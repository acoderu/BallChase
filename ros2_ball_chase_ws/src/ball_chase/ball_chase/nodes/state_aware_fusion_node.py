#!/usr/bin/env python3

"""
Enhanced Fusion Node - Using ROS 2 Lifecycle Management
Converts the original implementation to use ROS 2 Lifecycle Nodes
for better state management and transitions.
Optimized for Raspberry Pi 5 with multi-node resource coordination.
"""
import rclpy
from rclpy.node import Node
import time
import numpy as np
from geometry_msgs.msg import PointStamped, TwistStamped, TransformStamped
from std_msgs.msg import Float32, Bool, String, Float32MultiArray
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from tf2_geometry_msgs import do_transform_point
from collections import deque
import math
import json
import os
import copy
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# ROS 2 Lifecycle imports
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn, State
from rclpy.lifecycle import Publisher as LifecyclePublisher
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

# Import from config
from ball_chase.config.config_loader import ConfigLoader

# Optional imports for Pi 5 resource monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Try to import for temperature monitoring
try:
    import platform
    IS_RASPBERRY_PI = 'arm' in platform.machine().lower()
except ImportError:
    IS_RASPBERRY_PI = False


class SensorBuffer:
    """
    A buffer for storing and synchronizing sensor measurements.
    Helps coordinate data from multiple sensors with different update rates.
    """
    
    def __init__(self, sensor_time_thresholds=None, max_time_diff=None):
        """
        Initialize the sensor buffer with per-sensor time thresholds.
        
        Args:
            sensor_time_thresholds (dict): Dict of {sensor_name: max_time_diff}
                for per-sensor synchronization thresholds
            max_time_diff (float): Default maximum time difference for all sensors
        """
        self.buffers = {}
        self.default_max_time_diff = max_time_diff if max_time_diff is not None else 0.1
        # Store sensor-specific time thresholds
        self.sensor_time_thresholds = sensor_time_thresholds or {}
        # Reference to parent node for motion state access
        self.parent_node = None
    
    def add_sensor(self, sensor_name, buffer_size=20):
        """
        Register a new sensor to the synchronization system.
        
        Args:
            sensor_name (str): Unique name of the sensor
            buffer_size (int): Maximum number of measurements to keep
        """
        self.buffers[sensor_name] = deque(maxlen=buffer_size)
    
    def add_measurement(self, sensor_name, data, timestamp):
        """
        Add a new measurement from a sensor.
        
        Args:
            sensor_name (str): Name of the sensor
            data: The measurement data
            timestamp: ROS timestamp of the measurement
        """
        if sensor_name in self.buffers:
            # Log frame ID for debugging
            if hasattr(self, 'parent_node') and hasattr(self.parent_node, 'debug_level') and self.parent_node.debug_level >= 2:
                self.parent_node.get_logger().debug(
                    f"Adding {sensor_name} measurement with frame_id={data.header.frame_id}"
                )
            
            self.buffers[sensor_name].append((self._ros_time_to_float(timestamp), data))
    
    def get_latest_measurement(self, sensor_name):
        """
        Get the most recent measurement for a specific sensor.
        
        Args:
            sensor_name (str): Name of the sensor
            
        Returns:
            The most recent measurement or None if no measurements available
        """
        if sensor_name in self.buffers and self.buffers[sensor_name]:
            return self.buffers[sensor_name][-1][1]  # Return most recent data
        return None
    
    def find_synchronized_measurements(self, min_sensors=1, primary_sensor=None):
        """
        Find measurements from different sensors taken at approximately the same time.
        
        Args:
            min_sensors (int): Minimum number of synchronized sensors required
            primary_sensor (str): Optional sensor to use as time reference
                
        Returns:
            dict: Dictionary of {sensor_name: measurement} for synchronized measurements
        """
        # Check if we have enough sensors with data
        sensors_with_data = [s for s, b in self.buffers.items() if len(b) > 0]
        if len(sensors_with_data) < min_sensors:
            return {}

        # Find sensor to use as reference time
        if primary_sensor and primary_sensor in sensors_with_data:
            ref_sensor = primary_sensor
        else:
            # Prioritize 3D sensors over 2D if available
            ref_sensor = next((s for s in sensors_with_data if not s.endswith('_2d')), None)
            if ref_sensor is None:
                ref_sensor = sensors_with_data[0]
        
        best_sync = {}
        best_score = 0.0
        
        # For each measurement from reference sensor
        for ref_time, ref_data in self.buffers[ref_sensor]:
            current_sync = {ref_sensor: ref_data}
            current_score = 0.0
            
            # Check each other sensor
            for sensor in sensors_with_data:
                if sensor == ref_sensor:
                    continue
                
                # Get sensor-specific time threshold
                max_time_diff = self.sensor_time_thresholds.get(
                    sensor, self.default_max_time_diff)
                
                best_match = None
                best_match_diff = float('inf')
                
                for time_val, data in self.buffers[sensor]:
                    time_diff = abs(ref_time - time_val)
                    
                    if time_diff < best_match_diff and time_diff <= max_time_diff:
                        best_match_diff = time_diff
                        best_match = (time_val, data)
                
                # If we found a match within threshold
                if best_match:
                    # Add to current synchronization set
                    current_sync[sensor] = best_match[1]
                    # Score is higher when time differences are smaller (perfect=1.0)
                    match_score = 1.0 - (best_match_diff / max_time_diff)
                    current_score += match_score
            
            # Update best sync if this set is better
            if len(current_sync) >= min_sensors and current_score > best_score:
                best_sync = current_sync
                best_score = current_score
                
                # If we have all sensors, no need to keep searching
                if len(current_sync) == len(sensors_with_data):
                    break
                    
        return best_sync
    
    def _ros_time_to_float(self, timestamp):
        """Convert ROS timestamp to float seconds."""
        return timestamp.sec + timestamp.nanosec / 1e9    
    
    def interpolate_measurement(self, sensor, target_time):
        """
        Interpolate sensor measurement at the target time.
        
        Args:
            sensor (str): Sensor name
            target_time (float): Target timestamp for interpolation
                
        Returns:
            tuple: (interpolated_data, quality) or (None, 0) if not possible
        """
        if sensor not in self.buffers or len(self.buffers[sensor]) < 2:
            return None, 0.0
        
        # Find measurements before and after target time
        before_data = None
        after_data = None
        before_time = 0
        after_time = 0
        
        for time_val, data in self.buffers[sensor]:
            if time_val <= target_time and (before_data is None or time_val > before_time):
                before_data = data
                before_time = time_val
            if time_val >= target_time and (after_data is None or time_val < after_time):
                after_data = data
                after_time = time_val
        
        # If we don't have points on both sides, can't interpolate
        if before_data is None or after_data is None:
            return None, 0.0
        
        # Don't interpolate over large time gaps
        max_interp_gap = 0.5  # Maximum time gap for interpolation in seconds
        if after_time - before_time > max_interp_gap:
            return None, 0.0
        
        # Calculate interpolation factor (0 to 1)
        if after_time == before_time:  # Avoid division by zero
            t = 0.0
        else:
            t = (target_time - before_time) / (after_time - before_time)
        
        # For PointStamped messages, linearly interpolate position
        if hasattr(before_data, 'point') and hasattr(after_data, 'point'):
            result = copy.deepcopy(before_data)
            result.point.x = before_data.point.x + t * (after_data.point.x - before_data.point.x)
            result.point.y = before_data.point.y + t * (after_data.point.y - before_data.point.y)
            result.point.z = before_data.point.z + t * (after_data.point.z - before_data.point.z)
            
            # Quality is higher when we're closer to an actual measurement
            quality = 1.0 - min(t, 1.0-t)  # 1.0 at measurements, 0.5 halfway between
            return result, quality
        
        return None, 0.0
        
    def synthesize_measurement(self, sensor_name, target_time, state_predictor_callback=None):
        """
        Synthesize a measurement for a sensor when real data is unavailable.
        
        Args:
            sensor_name (str): The sensor to synthesize data for
            target_time (float): The target timestamp
            state_predictor_callback: Optional function to provide state prediction
            
        Returns:
            tuple: (synthesized_measurement, confidence)
        """
        # Default low confidence
        confidence = 0.3
        
        # Method 1: Interpolation from existing measurements
        interpolated, interp_quality = self.interpolate_measurement(sensor_name, target_time)
        if interpolated is not None:
            # Interpolation successful
            return interpolated, interp_quality
            
        # Method 2: Extrapolation from recent measurements
        if sensor_name in self.buffers and len(self.buffers[sensor_name]) >= 2:
            # Get two most recent measurements
            measurements = list(self.buffers[sensor_name])
            measurements.sort(key=lambda x: x[0])  # Sort by timestamp
            
            if len(measurements) >= 2:
                time1, data1 = measurements[-2]
                time2, data2 = measurements[-1]
                
                # Simple linear extrapolation
                if time2 > time1:  # Ensure time moved forward
                    dt_orig = time2 - time1
                    dt_extrap = target_time - time2
                    
                    # Don't extrapolate too far
                    if dt_extrap < dt_orig * 2:
                        # Create new message
                        result = copy.deepcopy(data2)
                        
                        # Extrapolate position
                        if hasattr(data1, 'point') and hasattr(data2, 'point'):
                            # Calculate velocity
                            vx = (data2.point.x - data1.point.x) / dt_orig
                            vy = (data2.point.y - data1.point.y) / dt_orig
                            vz = (data2.point.z - data1.point.z) / dt_orig
                            
                            # Extrapolate
                            result.point.x = data2.point.x + vx * dt_extrap
                            result.point.y = data2.point.y + vy * dt_extrap
                            result.point.z = data2.point.z + vz * dt_extrap
                            
                            # Quality decreases with extrapolation distance
                            confidence = max(0.1, 0.6 - (dt_extrap / dt_orig) * 0.3)
                            
                            return result, confidence
        
        # Method 3: Use state prediction if available
        if state_predictor_callback is not None:
            predicted_state = state_predictor_callback(target_time)
            if predicted_state is not None:
                # Create synthetic measurement from predicted state
                if sensor_name.endswith('_2d'):
                    # Create 2D point
                    result = PointStamped()
                    result.header.stamp.sec = int(target_time)
                    result.header.stamp.nanosec = int((target_time - int(target_time)) * 1e9)
                    result.header.frame_id = "prediction_frame"  # Special frame with flag for synthesized data
                    result.point.x = predicted_state[0]  # x
                    result.point.y = predicted_state[1]  # y
                    result.point.z = 0.5  # Confidence score
                else:
                    # Create 3D point
                    result = PointStamped()
                    result.header.stamp.sec = int(target_time)
                    result.header.stamp.nanosec = int((target_time - int(target_time)) * 1e9)
                    result.header.frame_id = "prediction_frame"
                    result.point.x = predicted_state[0]  # x
                    result.point.y = predicted_state[1]  # y
                    result.point.z = predicted_state[2] if len(predicted_state) > 2 else 0.11  # z
                
                # Lower confidence for prediction
                confidence = 0.4
                return result, confidence
        
        # No synthesis possible
        return None, 0.0

    def calculate_adaptive_time_thresholds(self):
        """
        Dynamically calculate appropriate time thresholds based on observed sensor rates.
        This adapts synchronization windows to actual sensor behaviors.
        """
        thresholds = {}
        
        # Calculate average time between measurements for each sensor
        for sensor, buffer in self.buffers.items():
            if len(buffer) < 2:
                continue
                
            # Calculate average interval between measurements
            timestamps = [t for t, _ in buffer]
            intervals = []
            for i in range(1, len(timestamps)):
                intervals.append(timestamps[i] - timestamps[i-1])
            
            if intervals:
                # Use larger of (2x average interval) or default threshold
                # This ensures we can handle occasional doubled intervals
                avg_interval = sum(intervals) / len(intervals)
                thresholds[sensor] = max(2.0 * avg_interval, self.default_max_time_diff)
        
        # Adjust thresholds based on motion state
        if hasattr(self, 'parent_node') and hasattr(self.parent_node, 'detect_motion_state'):
            motion_state = self.parent_node.detect_motion_state()
            
            adjusted_thresholds = {}
            for sensor, threshold in thresholds.items():
                # For fast motion, use tighter synchronization for better accuracy
                if motion_state == "medium_fast":
                    adjusted_thresholds[sensor] = threshold * 0.8  # Tighter window
                elif motion_state == "stationary":
                    adjusted_thresholds[sensor] = threshold * 1.5  # Wider window
                else:  # small_movement or unknown
                    adjusted_thresholds[sensor] = threshold
            
            thresholds = adjusted_thresholds
            
        # Update the sensor time thresholds
        for sensor, threshold in thresholds.items():
            self.sensor_time_thresholds[sensor] = threshold
            
        return thresholds


class SensorReliabilityTracker:
    """
    Tracks reliability metrics for various sensors based on their historical performance.
    This allows dynamic weighting of sensor measurements in the fusion algorithm.
    """
    
    def __init__(self, sensors=None):
        """
        Initialize the reliability tracker with a list of sensors.
        
        Args:
            sensors (list): List of sensor names to track
        """
        self.sensors = sensors or []
        
        # Reliability scores (0.0 to 1.0)
        self.reliability_scores = {sensor: 0.5 for sensor in self.sensors}
        
        # Historical counters
        self.total_measurements = {sensor: 0 for sensor in self.sensors}
        self.valid_measurements = {sensor: 0 for sensor in self.sensors}
        self.rejected_measurements = {sensor: 0 for sensor in self.sensors}
        
        # Innovation history for each sensor (recent residuals)
        self.innovation_history = {sensor: deque(maxlen=20) for sensor in self.sensors}
        
        # Gap tracking
        self.total_gaps = {sensor: 0 for sensor in self.sensors}
        self.gap_duration = {sensor: 0.0 for sensor in self.sensors}
        
        # Timing metrics
        self.timing_stability = {sensor: 1.0 for sensor in self.sensors}  # 1.0 = perfectly stable
        
        # Set default decay rates
        self.reliability_decay_rate = 0.99  # Slow decay when no new data
        self.reliability_boost_rate = 1.1   # Moderate boost for good measurements
        
    def add_sensor(self, sensor_name):
        """Add a new sensor to track."""
        if sensor_name not in self.sensors:
            self.sensors.append(sensor_name)
            self.reliability_scores[sensor_name] = 0.5
            self.total_measurements[sensor_name] = 0
            self.valid_measurements[sensor_name] = 0
            self.rejected_measurements[sensor_name] = 0
            self.innovation_history[sensor_name] = deque(maxlen=20)
            self.total_gaps[sensor_name] = 0
            self.gap_duration[sensor_name] = 0.0
            self.timing_stability[sensor_name] = 1.0
            
    def record_measurement(self, sensor_name, is_valid, innovation=None):
        """
        Record a new measurement from a sensor.
        
        Args:
            sensor_name (str): Name of the sensor
            is_valid (bool): Whether the measurement was valid and used
            innovation (float): Optional innovation (residual) value
        """
        if sensor_name not in self.sensors:
            self.add_sensor(sensor_name)
            
        # Update counters
        self.total_measurements[sensor_name] += 1
        if is_valid:
            self.valid_measurements[sensor_name] += 1
        else:
            self.rejected_measurements[sensor_name] += 1
            
        # Store innovation if provided
        if innovation is not None:
            self.innovation_history[sensor_name].append(innovation)
            
        # Update reliability score
        if is_valid:
            # Boost score for valid measurements (faster than decay)
            self.reliability_scores[sensor_name] = min(
                1.0, 
                self.reliability_scores[sensor_name] * self.reliability_boost_rate
            )
        else:
            # Reduce score for invalid measurements
            self.reliability_scores[sensor_name] = max(
                0.1,  # Don't go below 0.1 to allow recovery
                self.reliability_scores[sensor_name] * 0.9  # Stronger penalty for rejection
            )
            
    def record_gap(self, sensor_name, gap_duration):
        """Record a data gap for a sensor."""
        if sensor_name not in self.sensors:
            self.add_sensor(sensor_name)
            
        self.total_gaps[sensor_name] += 1
        self.gap_duration[sensor_name] += gap_duration
        
        # Reduce reliability score based on gap duration
        gap_factor = min(1.0, gap_duration / 2.0)  # Scale by gap duration up to 2 seconds
        self.reliability_scores[sensor_name] = max(
            0.1,  # Don't go below 0.1
            self.reliability_scores[sensor_name] * (1.0 - 0.2 * gap_factor)  # Up to 20% reduction
        )
        
    def record_timing_variance(self, sensor_name, timing_variance):
        """
        Record timing variance for a sensor to track stability.
        
        Args:
            sensor_name (str): Sensor name
            timing_variance (float): Variance in time between measurements
        """
        if sensor_name not in self.sensors:
            self.add_sensor(sensor_name)
            
        # Convert variance to stability score (1.0 = perfectly stable)
        # Assume variance under 0.001 is very stable, over 0.01 is unstable
        stability = max(0.5, min(1.0, 1.0 - (timing_variance * 100)))
        
        # Update using exponential moving average
        alpha = 0.2  # Weight for new measurement
        self.timing_stability[sensor_name] = (
            (1 - alpha) * self.timing_stability[sensor_name] + alpha * stability
        )
        
    def decay_unused_sensors(self):
        """Slightly decay reliability scores for sensors that haven't been updated."""
        for sensor in self.sensors:
            self.reliability_scores[sensor] = max(
                0.1,  # Don't go below 0.1
                self.reliability_scores[sensor] * self.reliability_decay_rate
            )
            
    def get_reliability(self, sensor_name):
        """Get the current reliability score for a sensor."""
        return self.reliability_scores.get(sensor_name, 0.5)
        
    def get_all_reliabilities(self):
        """Get all current reliability scores."""
        return self.reliability_scores.copy()
        
    def get_adaptive_measurement_noise(self, sensor_name, base_noise, motion_state="unknown"):
        """
        Get adaptive measurement noise based on reliability and motion state.
        
        Args:
            sensor_name (str): Sensor name
            base_noise (float or array): Base noise value(s)
            motion_state (str): Current motion state
            
        Returns:
            float or array: Adjusted noise value(s)
        """
        if sensor_name not in self.sensors:
            return base_noise
            
        reliability = self.reliability_scores[sensor_name]
        
        # Calculate adjustment factor based on reliability
        # For low reliability, increase noise up to 3x
        # For high reliability, decrease noise down to 0.8x
        if reliability < 0.5:
            # Scale 0.1-0.5 reliability to 1.0-3.0 factor
            factor = 3.0 - 5.0 * (reliability - 0.1)
        else:
            # Scale 0.5-1.0 reliability to 0.8-1.0 factor
            factor = 1.0 - 0.4 * (reliability - 0.5)
            
        # Apply motion state adjustment
        if motion_state == "medium_fast":
            factor *= 1.2  # More uncertainty during fast motion
        elif motion_state == "stationary":
            factor *= 0.9  # Less uncertainty when stationary
            
        # Apply the factor to the base noise
        if isinstance(base_noise, (list, tuple, np.ndarray)):
            return base_noise * factor
        else:
            return base_noise * factor


class AdaptiveValidationManager:
    """
    Manages adaptive validation thresholds for different sensors and conditions.
    Uses learning from past measurements to improve validation decisions.
    """
    
    def __init__(self, sensors=None):
        """Initialize with a list of expected sensors."""
        self.sensors = sensors or []
        
        # Validation history for each sensor
        self.validation_history = {sensor: deque(maxlen=50) for sensor in self.sensors}
        
        # Performance metrics
        self.false_positive_rate = {sensor: 0.0 for sensor in self.sensors}
        self.false_negative_rate = {sensor: 0.0 for sensor in self.sensors}
        
        # Adaptive thresholds with learning
        self.base_thresholds = {sensor: 5.0 for sensor in self.sensors}
        self.adaptive_thresholds = {sensor: 5.0 for sensor in self.sensors}
        
        # Contextual factors
        self.context = {
            'motion_state': 'unknown',
            'uncertainty': 1.0,
            'sensor_gaps': {},
            'consecutive_rejections': {}
        }
        
    def add_sensor(self, sensor_name):
        """Add a new sensor to track."""
        if sensor_name not in self.sensors:
            self.sensors.append(sensor_name)
            self.validation_history[sensor_name] = deque(maxlen=50)
            self.false_positive_rate[sensor_name] = 0.0
            self.false_negative_rate[sensor_name] = 0.0
            self.base_thresholds[sensor_name] = 5.0
            self.adaptive_thresholds[sensor_name] = 5.0
    
    def record_validation_result(self, sensor, innovation, was_accepted, was_correct):
        """
        Record the result of a validation decision.
        
        Args:
            sensor (str): Sensor name
            innovation (float): Innovation/residual value
            was_accepted (bool): Whether the measurement was accepted
            was_correct (bool): Whether the decision was correct (determined later)
        """
        if sensor not in self.sensors:
            self.add_sensor(sensor)
            
        # Store result in history
        self.validation_history[sensor].append({
            'innovation': innovation,
            'accepted': was_accepted,
            'correct': was_correct
        })
        
        # Update error rates
        false_positives = sum(1 for r in self.validation_history[sensor] 
                              if r['accepted'] and not r['correct'])
        false_negatives = sum(1 for r in self.validation_history[sensor] 
                             if not r['accepted'] and r['correct'])
        total = len(self.validation_history[sensor])
        
        if total > 0:
            self.false_positive_rate[sensor] = false_positives / total
            self.false_negative_rate[sensor] = false_negatives / total
            
        # Adjust thresholds based on error rates
        # Higher false negative rate -> lower threshold to accept more
        # Higher false positive rate -> higher threshold to reject more
        adjustment = 1.0 + (self.false_negative_rate[sensor] - self.false_positive_rate[sensor])
        self.adaptive_thresholds[sensor] = self.base_thresholds[sensor] * adjustment
    
    def update_context(self, context_dict):
        """Update contextual factors that affect validation."""
        self.context.update(context_dict)
        
    def get_validation_threshold_for_2d_derived(self, source, motion_state, bbox_age=0.0):
        """
        Get a specialized validation threshold for 3D estimates derived from 2D detections,
        which need more permissive thresholds due to their inherent uncertainty.
        
        Args:
            source (str): Source of the 2D sensor (e.g., 'yolo_2d_est3d')
            motion_state (str): Current motion state
            bbox_age (float): Age of the bounding box used for estimation
            
        Returns:
            float: Validation threshold for this 2D-derived 3D estimate
        """
        # Start with base thresholds that are much higher than standard 3D measurements
        if source.startswith('yolo'):
            # YOLO-based estimates tend to be more reliable than HSV
            base_threshold = 16.0  # High base threshold for YOLO
        else:
            base_threshold = 18.0  # Even higher for HSV
        
        # Apply motion state adjustments (use higher thresholds for all states)
        if motion_state == "stationary":
            motion_multiplier = 1.0
        elif motion_state == "long_stationary":
            motion_multiplier = 0.9  # Slightly lower for long-stationary
        elif motion_state == "small_movement":
            motion_multiplier = 1.2  # Higher for movement states
        else:  # medium_fast or unknown
            motion_multiplier = 1.4  # Much higher for faster movement
        
        threshold = base_threshold * motion_multiplier
        
        # Add additional allowance for bbox age
        if bbox_age > 0.0:
            # Increase threshold for older bboxes, up to 50% more for bboxes approaching max age
            max_age_factor = 1.5
            age_adjustment = min(max_age_factor - 1.0, bbox_age / 5.0)  # Scale up to 5 seconds of age
            threshold *= (1.0 + age_adjustment)
        
        # Check for state transitions
        if hasattr(self, 'prev_motion_state') and hasattr(self, 'motion_state'):
            if self.prev_motion_state != self.motion_state:
                # During transitions, especially from stationary, be much more permissive
                if self.prev_motion_state in ["stationary", "long_stationary"]:
                    threshold *= 2.0  # Double threshold during transition from stationary
                    
                    if self.debug_level >= 1:
                        self.get_logger().info(
                            f"Applying transition boost for {source}: threshold={threshold:.2f} "
                            f"during {self.prev_motion_state}->{self.motion_state} transition"
                        )
        
        # Check for consecutive rejections
        if hasattr(self, 'consecutive_rejections_per_sensor'):
            # For 2D-derived estimates, use the base 2D sensor name to check rejections
            base_sensor = source.split('_')[0] + '_2d'
            rejections = self.consecutive_rejections_per_sensor.get(base_sensor, 0)
            
            if rejections >= 2:  # Low threshold to respond quickly
                boost_factor = 1.0 + (min(rejections, 10) * 0.2)  # Higher per-rejection boost
                threshold *= boost_factor
                
                if self.debug_level >= 1 and boost_factor > 1.4:
                    self.get_logger().info(
                        f"Applying large rejection-break boost for {source}: "
                        f"{boost_factor:.2f}x → {threshold:.2f}"
                    )
        
        # HARD LIMIT: Cap maximum thresholds at a safety value to prevent extreme values
        max_safe_threshold = 40.0  # Very high for 2D-derived
        threshold = min(threshold, max_safe_threshold)
        
        return threshold
    
    def get_validation_threshold(self, sensor, innovation=None):
        """
        Get adaptive validation threshold for a sensor.
        
        Args:
            sensor (str): Sensor name
            innovation (float, optional): Current innovation/residual
            
        Returns:
            float: Adaptive threshold value
        """
        if sensor not in self.sensors:
            self.add_sensor(sensor)
            
        # Start with adaptive threshold from learning
        threshold = self.adaptive_thresholds[sensor]
        
        # Apply contextual adjustments
        
        # 1. Motion state adjustment
        motion_state = self.context.get('motion_state', 'unknown')
        if motion_state == 'stationary':
            threshold *= 0.8  # More strict during stationary
        elif motion_state == 'long_stationary':
            threshold *= 0.7  # Even more strict for long-term stationary
        elif motion_state == 'medium_fast':
            threshold *= 1.5  # More permissive during fast motion
        
        # 2. Uncertainty adjustment
        uncertainty = self.context.get('uncertainty', 1.0)
        threshold *= max(1.0, min(3.0, uncertainty))  # Scale with uncertainty
        
        # 3. Sensor gap adjustment
        sensor_gaps = self.context.get('sensor_gaps', {})
        if sensor in sensor_gaps and sensor_gaps[sensor]:
            # More permissive after gaps
            gap_level = sensor_gaps[sensor].get('gap_level', 0.0)
            threshold *= (1.0 + gap_level)
        
        # 4. Consecutive rejections adjustment
        consecutive_rejections = self.context.get('consecutive_rejections', {}).get(sensor, 0)
        if consecutive_rejections > 2:
            # Increase threshold with more rejections to prevent lockout
            threshold *= (1.0 + (consecutive_rejections * 0.1))
        
        return threshold


class SmoothedStateEstimator:
    """
    Provides temporal smoothing for state estimates to reduce uncertainty spikes
    during sensor recoveries and transitions.
    """
    
    def __init__(self, window_size=5):
        """
        Initialize the smoothed state estimator.
        
        Args:
            window_size (int): Size of the smoothing window
        """
        self.window_size = window_size
        self.position_buffer = deque(maxlen=window_size)
        self.velocity_buffer = deque(maxlen=window_size)
        self.uncertainty_buffer = deque(maxlen=window_size)
        self.timestamp_buffer = deque(maxlen=window_size)
        
        # Weights for exponential smoothing (newest to oldest)
        self.weights = np.array([0.4, 0.25, 0.15, 0.1, 0.1])
        # Truncate weights if window_size is smaller
        if window_size < len(self.weights):
            self.weights = self.weights[:window_size]
            # Renormalize
            self.weights = self.weights / np.sum(self.weights)
            
    def add_state(self, position, velocity, uncertainty, timestamp):
        """
        Add a new state estimate to the buffer.
        
        Args:
            position (np.ndarray): Position vector [x, y, z]
            velocity (np.ndarray): Velocity vector [vx, vy, vz]
            uncertainty (float): Position uncertainty scalar
            timestamp (float): Timestamp of the estimate
        """
        self.position_buffer.append(position.copy())
        self.velocity_buffer.append(velocity.copy())
        self.uncertainty_buffer.append(uncertainty)
        self.timestamp_buffer.append(timestamp)
        
    def get_smoothed_state(self):
        """
        Get the temporally smoothed state estimate.
        
        Returns:
            dict: Smoothed state with keys 'position', 'velocity', 'uncertainty'
        """
        if len(self.position_buffer) == 0:
            return None
            
        # If we have only one estimate, return it directly
        if len(self.position_buffer) == 1:
            return {
                'position': self.position_buffer[0].copy(),
                'velocity': self.velocity_buffer[0].copy(),
                'uncertainty': self.uncertainty_buffer[0],
                'timestamp': self.timestamp_buffer[0]
            }
            
        # Calculate active weights based on buffer size
        active_weights = self.weights[:len(self.position_buffer)]
        # Renormalize weights
        active_weights = active_weights / np.sum(active_weights)
        
        # Calculate weighted average of position
        smoothed_position = np.zeros_like(self.position_buffer[0])
        for i, pos in enumerate(self.position_buffer):
            smoothed_position += active_weights[i] * pos
            
        # Calculate weighted average of velocity
        smoothed_velocity = np.zeros_like(self.velocity_buffer[0])
        for i, vel in enumerate(self.velocity_buffer):
            smoothed_velocity += active_weights[i] * vel
            
        # Calculate smoothed uncertainty (use minimum to avoid overconfidence)
        # This helps maintain conservative uncertainty estimates
        smoothed_uncertainty = np.min(self.uncertainty_buffer)
        
        # Use newest timestamp
        latest_timestamp = self.timestamp_buffer[-1]
        
        return {
            'position': smoothed_position,
            'velocity': smoothed_velocity,
            'uncertainty': smoothed_uncertainty,
            'timestamp': latest_timestamp
        }
        
    def reset(self):
        """Reset the smoothing buffer."""
        self.position_buffer.clear()
        self.velocity_buffer.clear()
        self.uncertainty_buffer.clear()
        self.timestamp_buffer.clear()


class EnhancedFusionLifecycleNode(LifecycleNode):
    """
    Enhanced fusion node using ROS 2 Lifecycle framework for state management.
    This provides a clear state machine model for managing node transitions.
    """
    
    def __init__(self, node_name='state_aware_fusion_node'):
        super().__init__(node_name)
        
        self.get_logger().info("====== Enhanced Fusion Lifecycle Node Starting ======")
        
        # Core tracking variables
        self.start_time = time.time()
        self.transform_available = False
        self.transform_checks = 0
        self.transform_successes = 0
        self.transform_failures = 0
        self.transform_confirmed = False  # Flag to track if transform is permanently confirmed
        self.is_ready = False  # Flag to track if the node is ready for processing
        self._transform_available_count = 0
        
        # Counter for lidar messages
        self.lidar_msg_counter = 0
        
        # Initialize publishers list with a different name to avoid conflicts
        self._publishers = []
        
        # Lifecycle-specific flags
        self.is_configured = False
        self.is_activated = False
        
        # Use camera_frame as the reference coordinate system instead of map
        self.reference_frame = "base_link"
        self.get_logger().info(f"Using {self.reference_frame} as reference coordinate frame for fusion")
        
        # Lifecycle requirement: Store timers in a list to manage them in lifecycle transitions
        self._timer_list = []
        self.subscribers = []
        
        # We'll initialize these in on_configure and on_activate
        self.tf_buffer = None
        self.tf_listener = None
        self.tf_static_broadcaster = None
        
        # Initialize tracking variables that are checked with hasattr later
        self.position_anchors = {}
        self.sync_quality_metrics = {
            'success_rate': 0.0,
            'avg_time_diff': 0.0,
            'sensor_availability': {},
            'sync_counts': 0,
            'attempt_counts': 0
        }
        
        # Initialize motion state tracking
        self.init_motion_state_tracking()
        
        # Initialize flat ground tracking
        self.flat_ground_detected = False
        self.flat_ground_count = 0
        
        # Initialize sensor recovery tracking
        self.sensor_gap_detection = {}
        
        # Initialize reliability buffer
        self.reliability_buffer = deque([False] * 3, maxlen=5)
        self.last_tracking_state = False
        
        # Sensor gap tolerance window tracking
        self.sensor_gap_window = {
            'active': False,
            'start_time': 0.0,
            'previous_reliability': False,
            'tolerance_seconds': 2.0,  # Base tolerance in seconds
            'base_tolerance': 2.0,     # Store base tolerance value for adaptive calculations
            'adaptive_enabled': True   # Enable adaptive tolerance adjustment
        }
        
        # Add sensor history for adaptive gap tolerance
        self.sensor_update_intervals = {sensor: deque(maxlen=10) for sensor in ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']}
        self.last_sensor_update_times = {sensor: 0.0 for sensor in ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']}
        
        # Tracking variables
        self.initialized = False
        self.tracking_reliable = False
        self.last_update_time = None
        self.consecutive_updates = 0
        self.debug_level = 1  # Default until loaded from config
    
    def on_configure(self, state):
        """
        Lifecycle configure callback - called when transitioning from Unconfigured to Inactive.
        Handle core system setup that doesn't require activation yet.
        
        Returns:
            TransitionCallbackReturn: Success if configuration completes successfully
        """
        self.get_logger().info("Lifecycle transition: on_configure")
        
        try:
            # PHASE 1: Initialize transform system
            self.init_transform_system()
            
            # Cache static transforms
            self.cache_static_transforms()
            
            # Add verification of transform tree
            self.create_timer(2.0, self.check_transform_availability, callback_group=None)
            
            # PHASE 2: Load configuration
            self.load_configuration()
            
            # PHASE 3: Initialize state tracking and buffers
            self.init_state_tracking()
            self.init_sensor_synchronization()
            
            # PHASE 4: Initialize diagnostic tracking
            self.init_diagnostics()
            
            # Motion state protection with reduced long_stationary threshold
            self.motion_state_protection = {
                'long_stationary_confirmed_time': 0.0,
                'long_stationary_established': False,
                'consecutive_stationary_after_long': 0,
                'min_time_in_long_stationary': 2.0,  # Seconds required in long_stationary
                'post_gap_cooldown_active': False,
                'post_gap_cooldown_end': 0.0,
                'post_gap_protected_state': None,
                'last_gap_recovery_time': 0.0,
                'protection_violation_count': 0
            }
            
            # Mark configuration as complete
            self.is_configured = True
            self.get_logger().info("Configuration completed successfully")
            
            return TransitionCallbackReturn.SUCCESS
            
        except Exception as e:
            self.get_logger().error(f"Error during configuration: {str(e)}")
            return TransitionCallbackReturn.ERROR
    
    def on_cleanup(self, state):
        """
        Lifecycle cleanup callback - called when transitioning from Inactive to Unconfigured.
        Clean up resources allocated during on_configure.
        
        Returns:
            TransitionCallbackReturn: Success if cleanup completes successfully
        """
        self.get_logger().info("Lifecycle transition: on_cleanup")
        
        # Reset configuration state
        self.is_configured = False
        
        # Clean up transform resources
        self.tf_listener = None
        self.tf_buffer = None
        self.tf_static_broadcaster = None
        
        return TransitionCallbackReturn.SUCCESS
    
    def on_activate(self, state):
        """
        Modified lifecycle activate callback to prioritize fast initialization
        """
        self.get_logger().info("Lifecycle transition: on_activate")
        
        # First check if transforms are available
        if not self.check_transform_availability():
            self.get_logger().warn("Transform not available yet - cannot activate")
            # Create a one-shot timer to retry activation after a delay
            self.create_timer(5.0, self.retry_activation, callback_group=None)
            return TransitionCallbackReturn.FAILURE
        
        try:
            # PHASE 4: Set up publishers
            self.setup_publishers()
            
            # Activate lifecycle publishers
            for pub in self._publishers:
                pub.on_activate(state)
            
            # PHASE 5: Set up subscriptions
            self.setup_subscriptions()
            
            # Set flag for fast initialization
            self.pending_initialization = True
            self.initialization_attempts = 0
            
            # Set flag for refinement phase (will be used after initialization)
            self.in_refinement_phase = False
            self.refinement_measurements = 0
            self.refinement_start_time = 0.0
            
            # PHASE 7: Set up processing timers
            self.setup_timers()
            
            # Add a one-shot timer for initialization attempt, but with shorter delay
            self.create_timer(0.2, self.delayed_initialization, callback_group=None)
            
            # Mark as activated and ready
            self.is_activated = True
            self.is_ready = True
            
            self.get_logger().info("Node activated - will attempt fast initialization with first sensor data")
            
            return TransitionCallbackReturn.SUCCESS
        
        except Exception as e:
            self.get_logger().error(f"Error during activation: {str(e)}")
            return TransitionCallbackReturn.ERROR

    def on_deactivate(self, state):
        """
        Lifecycle deactivate callback - called when transitioning from Active to Inactive.
        Stop subscriptions and timers.
        
        Returns:
            TransitionCallbackReturn: Success if deactivation completes successfully
        """
        self.get_logger().info("Lifecycle transition: on_deactivate")
        
        # Deactivate lifecycle publishers
        for pub in self._publishers:
            pub.on_deactivate()
        
        # Reset activation flags
        self.is_activated = False
        self.is_ready = False
        
        # Clean up timers
        self._cleanup_timers()
        
        # Clean up subscribers
        for sub in self.subscribers:
            self.destroy_subscription(sub)
        self.subscribers = []
        
        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state):
        """
        Lifecycle shutdown callback - called when shutting down from any state.
        Perform final cleanup.
        
        Returns:
            TransitionCallbackReturn: Success if shutdown completes successfully
        """
        self.get_logger().info("Lifecycle transition: on_shutdown")
        
        # Clean up any remaining resources
        if hasattr(self, 'tf_listener') and self.tf_listener is not None:
            self.tf_listener = None
        
        if hasattr(self, 'tf_buffer') and self.tf_buffer is not None:
            self.tf_buffer = None
        
        if hasattr(self, 'tf_static_broadcaster') and self.tf_static_broadcaster is not None:
            self.tf_static_broadcaster = None
        
        return TransitionCallbackReturn.SUCCESS
    
    def init_transform_system(self):
        """Initialize just the transform system."""
        # CRITICAL STEP: Set up transform system FIRST
        self.tf_buffer = Buffer()  
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize cached transform variables
        self.tf_camera_to_base = None
        self.tf_lidar_to_base = None
        
        self.get_logger().info("Transform system initialized - waiting for transforms")
    
    def check_transform_availability(self):
        """
        Check if transforms are available and verify that test transforms are properly received.
        Returns True if transform is available, False otherwise.
        """
        # Increment check counter
        self.transform_checks += 1
        
        # Define the sensor frames we need to check
        sensor_frames = []
        
        # Use expected_frames from config if available
        if hasattr(self, 'expected_frames') and self.expected_frames:
            sensor_frames = list(self.expected_frames.values())
        else:
            # Fallback to hardcoded frames
            sensor_frames = [
                "lidar_frame",
                "ascamera_color_0",  # For both depth and RGB camera
            ]
        
        transforms_available = True
        for frame in sensor_frames:
            try:
                when = rclpy.time.Time()
                timeoutP = rclpy.duration.Duration(seconds=0.1)
                
                # Check if transform exists
                if self.tf_buffer.can_transform(
                    self.reference_frame, frame, when, timeout=timeoutP
                ):
                    self.transform_successes += 1
                    self.get_logger().debug(f"Transform {frame} → {self.reference_frame} is available")
                    
                    # Log actual transform details occasionally
                    if self.transform_successes % 10 == 0:
                        transform = self.tf_buffer.lookup_transform(
                            self.reference_frame, frame, when, timeout=timeoutP
                        )
                        self.get_logger().info(
                            f"Transform details for {frame}: translation=[{transform.transform.translation.x:.4f}, "
                            f"{transform.transform.translation.y:.4f}, {transform.transform.translation.z:.4f}]"
                        )
                else:
                    self.transform_failures += 1
                    self.get_logger().warn(f"Transform {frame} → {self.reference_frame} is NOT available")
                    transforms_available = False
            except Exception as e:
                self.transform_failures += 1
                self.get_logger().error(f"Error checking transform {frame}: {str(e)}")
                transforms_available = False
        
        # Update transform health based on success rate
        if self.transform_checks > 0:
            success_rate = self.transform_successes / (self.transform_successes + self.transform_failures)
            self.transform_health = min(1.0, success_rate)
        
        # Mark transform as confirmed after consistent availability
        if transforms_available and not self.transform_confirmed:
            if not hasattr(self, '_transform_available_count'):
                self._transform_available_count = 1
            else:
                self._transform_available_count += 1
                
            if self._transform_available_count >= 3:
                self.transform_confirmed = True
                self.get_logger().info("Transform availability confirmed after multiple consecutive checks")
                
                # Disable the transform check timer since transforms are static
                if hasattr(self, '_transform_check_timer'):
                    self.destroy_timer(self._transform_check_timer)
                    self._transform_check_timer = None
                    self.get_logger().info("Transform check timer disabled - transforms are static")
        elif not transforms_available and hasattr(self, '_transform_available_count'):
            self._transform_available_count = 0
        
        self.transform_available = transforms_available
        return transforms_available

    def retry_activation(self):
        """
        Retry the activation process if transforms weren't available the first time.
        This is called by a timer set in on_activate when transforms aren't ready.
        """
        self.get_logger().info("Retrying activation...")
        
        # Check if transforms are available now
        if self.check_transform_availability():
            self.get_logger().info("Transform is now available - triggering activation")
            # Manually trigger the transition to active state
            if self.trigger_transition(
                rclpy.lifecycle.msg.Transition.TRANSITION_ACTIVATE):
                self.get_logger().info("Activation triggered successfully")
            else:
                self.get_logger().error("Failed to trigger activation transition")
        else:
            # List all the frames in our transform buffer
            try:
                frames = self.tf_buffer.all_frames_as_string()
                if frames:
                    self.get_logger().info(f"Available frames:\n{frames}")
            except Exception as e:
                self.get_logger().warn(f"Could not list frames: {str(e)}")
            
            self.get_logger().warn("Transform still not available - will retry later")
            # Create another one-shot timer to retry again
            self.create_timer(5.0, self.retry_activation, callback_group=None)

    def assess_sensor_data_quality(self, measurement, source):
        """
        Assess the quality of a sensor measurement for initialization purposes.
        
        Args:
            measurement (PointStamped): The measurement to assess
            source (str): Source of the measurement
            
        Returns:
            tuple: (quality_score, is_valid)
                quality_score: 0.0-1.0 quality assessment
                is_valid: Boolean indicating if measurement is valid
        """
        # Default score
        quality_score = 0.5
        is_valid = True
        
        # Check valid range
        max_valid_range = 5.0  # Maximum valid range in meters
        measurement_distance = math.sqrt(
            measurement.point.x**2 + measurement.point.y**2 + measurement.point.z**2
        )
        
        if measurement_distance > max_valid_range:
            self.get_logger().warn(
                f"Measurement distance {measurement_distance:.2f}m exceeds valid range {max_valid_range:.1f}m"
            )
            is_valid = False
            return 0.0, False
        
        # Check for NaN or inf values
        if (math.isnan(measurement.point.x) or math.isnan(measurement.point.y) or 
            math.isinf(measurement.point.x) or math.isinf(measurement.point.y)):
            self.get_logger().warn(f"Measurement contains NaN or Inf values")
            is_valid = False
            return 0.0, False
        
        # Source-specific quality assessment
        if source == 'lidar':
            # LiDAR is generally more reliable
            quality_score = 0.9
            
            # Check if position is reasonable (e.g., z height)
            if abs(measurement.point.z - self.basketball_z_height) > 0.2:
                quality_score *= 0.7  # Reduce score if height is off
        
        elif source == 'yolo_2d_est3d':
            # Estimated 3D position is less reliable
            quality_score = 0.7
            
            # Check bbox age
            if hasattr(self, 'bbox_data') and 'yolo_2d' in self.bbox_data:
                bbox_age = time.time() - self.bbox_data['yolo_2d'].get('timestamp', 0)
                if bbox_age > 1.0:
                    quality_score *= 0.8  # Reduce score for older bbox data
        
        # Log quality assessment
        self.get_logger().debug(
            f"Sensor quality assessment: {source} quality={quality_score:.2f}, valid={is_valid}"
        )
        
        return quality_score, is_valid

    def delayed_initialization(self):
        """Modified delayed initialization that uses fast initialization"""
        if not self.initialized and self.pending_initialization:
            lidar_msg = self.sensor_buffer.get_latest_measurement('lidar')
            yolo_2d_msg = self.sensor_buffer.get_latest_measurement('yolo_2d')
            
            # First priority: try with LiDAR
            if lidar_msg:
                # Transform to reference frame
                transformed = self.transform_point(lidar_msg, self.reference_frame, False)
                if transformed:
                    self.get_logger().info("Attempting fast initialization with LiDAR data")
                    # Try fast initialization with LiDAR
                    if self.fast_initialize_with_first_measurement(transformed, 'lidar'):
                        self.pending_initialization = False
                        return
            
            # Second priority: try with YOLO 2D if bbox data available
            if not self.initialized and yolo_2d_msg and 'yolo_2d' in self.bbox_data:
                # Estimate 3D position from 2D detection
                estimated_3d = self.estimate_3d_from_2d(yolo_2d_msg, self.bbox_data['yolo_2d'])
                if estimated_3d:
                    self.get_logger().info("Attempting fast initialization with estimated 3D from YOLO 2D")
                    # Try fast initialization with estimated 3D
                    if self.fast_initialize_with_first_measurement(estimated_3d, 'yolo_2d_est3d'):
                        self.pending_initialization = False
                        return
            
            # If we're still not initialized, try again later (up to a limit)
            self.initialization_attempts += 1
            if self.initialization_attempts < 5:
                self.get_logger().info(f"No sensor data yet for initialization (attempt {self.initialization_attempts})")
                self.create_timer(0.5, self.delayed_initialization, callback_group=None)
            else:
                # Fall back to default initialization after multiple attempts
                self.get_logger().warn("No sensor data available after multiple attempts - initializing with defaults")                
                self.pending_initialization = False

    def load_configuration(self):
        """Load configuration from fusion_config.yaml."""
        try:
            config_loader = ConfigLoader()
            self.config = config_loader.load_yaml('fusion_config.yaml')  # Store the whole config
            
            # Extract topic configuration
            topics = self.config.get('topics', {})
            input_topics = topics.get('input', {})
            output_topics = topics.get('output', {})
            
            self.lidar_topic = input_topics.get('lidar', '/basketball/lidar/position')
            self.hsv_3d_topic = input_topics.get('hsv_3d', '/basketball/hsv/position_3d')
            self.yolo_3d_topic = input_topics.get('yolo_3d', '/basketball/yolo/position_3d')
            self.hsv_2d_topic = input_topics.get('hsv_2d', '/basketball/hsv/position')
            self.yolo_2d_topic = input_topics.get('yolo_2d', '/basketball/yolo/position')
            self.combined_topic = output_topics.get('combined', '/basketball/detected_position')
            
            # 2D bounding box topics for distance estimation
            self.hsv_bbox_topic = input_topics.get('hsv_bbox', '/basketball/hsv/bbox')
            self.yolo_bbox_topic = input_topics.get('yolo_bbox', '/basketball/yolo/bbox')
            
            self.position_topic = output_topics.get('position', '/basketball/fused/position')
            self.velocity_topic = output_topics.get('velocity', '/basketball/fused/velocity')
            self.status_topic = output_topics.get('tracking_status', '/basketball/fused/tracking_status')
            self.uncertainty_topic = output_topics.get('uncertainty', '/basketball/fused/position_uncertainty')
            self.diagnostics_topic = output_topics.get('diagnostics', '/basketball/fusion/diagnostics')
            
            # Load expected frame IDs
            expected_frames = self.config.get('expected_frames', {})
            self.expected_frames = {}
            
            for sensor, frame in expected_frames.items():
                if sensor != 'reference_frame':  # Skip reference_frame
                    self.expected_frames[sensor] = frame
            
            # Override reference_frame if specified in config
            if 'reference_frame' in expected_frames:
                self.reference_frame = expected_frames['reference_frame']
            
            # Set the specific frame for yolo_3d, hsv_3d, and combined topics
            self.expected_frames['yolo_3d'] = 'ascamera_camera_link_0'
            self.expected_frames['hsv_3d'] = 'ascamera_camera_link_0'
            self.expected_frames['combined'] = 'ascamera_camera_link_0'
                
            self.get_logger().info(f"Expected sensor frames: {self.expected_frames}")
                
            # Process noise parameters
            self.process_noise_pos = self.config.get('process_noise', {}).get('position', 0.1)
            self.process_noise_vel = self.config.get('process_noise', {}).get('velocity', 1.0)
            
            # Measurement noise parameters
            measurement_noise = self.config.get('measurement_noise', {})
            self.measurement_noise_lidar = measurement_noise.get('lidar', 0.03)
            self.measurement_noise_hsv_3d = measurement_noise.get('hsv_3d', 0.05)
            self.measurement_noise_yolo_3d = measurement_noise.get('yolo_3d', 0.04)
            self.measurement_noise_hsv_2d = measurement_noise.get('hsv_2d', 50.0)
            self.measurement_noise_yolo_2d = measurement_noise.get('yolo_2d', 30.0)
            
            # Add measurement noise for estimated 3D from 2D
            self.measurement_noise_hsv_2d_est3d = measurement_noise.get('hsv_2d_est3d', 0.15)
            self.measurement_noise_yolo_2d_est3d = measurement_noise.get('yolo_2d_est3d', 0.12)
            
            # Filter parameters
            filter_params = self.config.get('filter', {})
            self.max_time_diff = filter_params.get('max_time_diff', 0.2)
            self.min_confidence_threshold = filter_params.get('min_confidence_threshold', 0.5)
            self.detection_timeout = filter_params.get('detection_timeout', 0.5)
            
            # Tracking parameters
            tracking_params = self.config.get('tracking', {})
            self.position_uncertainty_threshold = tracking_params.get('position_uncertainty_threshold', 0.5)
            self.velocity_uncertainty_threshold = tracking_params.get('velocity_uncertainty_threshold', 1.0)
            
            # Advanced features
            advanced_features = self.config.get('advanced_features', {})
            self.use_bbox_distance_estimation = advanced_features.get('use_bbox_distance_estimation', True)
            self.allow_tracking_with_2d_only = advanced_features.get('allow_tracking_with_2d_only', True)
            self.increased_uncertainty_mode = advanced_features.get('increased_uncertainty_mode', True)
            
            # Diagnostic parameters
            diag_params = self.config.get('diagnostics', {})
            self.history_length = diag_params.get('history_length', 100)
            self.debug_level = diag_params.get('debug_level', 1)
            
            # Store base noise values for adaptive adjustment
            self.base_measurement_noise_lidar = self.measurement_noise_lidar
            self.base_measurement_noise_hsv_3d = self.measurement_noise_hsv_3d
            self.base_measurement_noise_yolo_3d = self.measurement_noise_yolo_3d
            self.base_measurement_noise_hsv_2d = self.measurement_noise_hsv_2d
            self.base_measurement_noise_yolo_2d = self.measurement_noise_yolo_2d
            self.base_measurement_noise_hsv_2d_est3d = self.measurement_noise_hsv_2d_est3d
            self.base_measurement_noise_yolo_2d_est3d = self.measurement_noise_yolo_2d_est3d

            # Add configurable parameter for maximum message age
            self.max_message_age = diag_params.get('max_message_age', 1.0)  # Default to 1.0 second
            
            self.get_logger().info("Configuration loaded successfully")
        except Exception as e:
            self.get_logger().error(f"Error loading config: {str(e)}")
            # Set reasonable defaults
            self.lidar_topic = '/basketball/lidar/position'
            self.hsv_3d_topic = '/basketball/hsv/position_3d'
            self.yolo_3d_topic = '/basketball/yolo/position_3d'
            self.hsv_2d_topic = '/basketball/hsv/position'
            self.yolo_2d_topic = '/basketball/yolo/position'
            self.hsv_bbox_topic = '/basketball/hsv/bbox'
            self.yolo_bbox_topic = '/basketball/yolo/bbox'
            self.position_topic = '/basketball/fused/position'
            self.velocity_topic = '/basketball/fused/velocity'
            self.status_topic = '/basketball/fused/tracking_status'
            self.uncertainty_topic = '/basketball/fused/position_uncertainty'
            self.diagnostics_topic = '/basketball/fusion/diagnostics'
            self.process_noise_pos = 0.1
            self.process_noise_vel = 1.0
            self.measurement_noise_lidar = 0.03
            self.measurement_noise_hsv_3d = 0.05
            self.measurement_noise_yolo_3d = 0.04
            self.measurement_noise_hsv_2d = 50.0
            self.measurement_noise_yolo_2d = 30.0
            self.measurement_noise_hsv_2d_est3d = 0.15
            self.measurement_noise_yolo_2d_est3d = 0.12
            self.max_time_diff = 0.2
            self.min_confidence_threshold = 0.5
            self.detection_timeout = 0.5
            self.position_uncertainty_threshold = 0.5
            self.velocity_uncertainty_threshold = 1.0
            self.use_bbox_distance_estimation = True
            self.allow_tracking_with_2d_only = True
            self.increased_uncertainty_mode = True
            self.history_length = 100
            self.debug_level = 1
            
            # Base noise (for adaptive adjustment)
            self.base_measurement_noise_lidar = self.measurement_noise_lidar
            self.base_measurement_noise_hsv_3d = self.measurement_noise_hsv_3d
            self.base_measurement_noise_yolo_3d = self.measurement_noise_yolo_3d
            self.base_measurement_noise_hsv_2d = self.measurement_noise_hsv_2d
            self.base_measurement_noise_yolo_2d = self.measurement_noise_yolo_2d
            self.base_measurement_noise_hsv_2d_est3d = self.measurement_noise_hsv_2d_est3d
            self.base_measurement_noise_yolo_2d_est3d = self.measurement_noise_yolo_2d_est3d

            # Add default for maximum message age
            self.max_message_age = 1.0

    def fast_initialize_with_first_measurement(self, measurement, source):
        """
        Quickly initialize the filter with the first reliable measurement.
        
        Args:
            measurement (PointStamped): The first reliable measurement
            source (str): Source of the measurement (e.g., 'lidar', 'yolo_2d')
            
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        # Ensure measurement is in reference frame
        if measurement.header.frame_id != self.reference_frame:
            transformed = self.transform_point(measurement, self.reference_frame, source.endswith('_2d'))
            if transformed is None:
                self.get_logger().warn(f"Cannot fast-initialize filter - transform failed")
                return False
            measurement = transformed
        
        # Initialize state vector with measurement
        # For 4D state [x, y, vx, vy]
        self.state = np.zeros(4, dtype=np.float32)
        self.state[0] = measurement.point.x  # x position
        self.state[1] = measurement.point.y  # y position
        # Velocities are initialized to zero
        
        # For LiDAR, trust the measurement more
        if source == 'lidar':
            position_variance = 0.05  # Lower variance = higher confidence
        else:
            position_variance = 0.1   # Higher variance for other sensors
        
        # Create covariance matrix with appropriate initial uncertainties
        self.covariance = np.eye(4, dtype=np.float32)
        self.covariance[0:2, 0:2] *= position_variance  # Position uncertainty
        self.covariance[2:4, 2:4] *= 1.0  # Velocity uncertainty
        
        # Mark as initialized
        self.initialized = True
        self.last_update_time = time.time()
        self.initialization_source = f"fast_init_{source}"
        
        # Update uncertainty metrics
        self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:2, 0:2]) / 2.0)
        self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[2:4, 2:4]) / 2.0)
        
        # Set flag to indicate we're in refinement phase
        self.in_refinement_phase = True
        self.refinement_measurements = 0
        self.refinement_start_time = time.time()
        
        # Log the initialization
        self.get_logger().info(
            f"Fast initialization with {source}: position=({measurement.point.x:.2f}, "
            f"{measurement.point.y:.2f}), uncertainty={self.position_uncertainty:.3f}m"
        )
        
        # Start active tracking
        return True
    
    def init_state_tracking(self):
        """Initialize state tracking variables with 4D state optimized for ground-only basketball movement."""
        # Kalman filter state: [x, y, vx, vy] - reduced from 6D to 4D
        # We eliminate z and vz since the basketball only moves on the ground
        self.state = np.zeros(4, dtype=np.float32)
        
        # Create initial covariance matrix (uncertainty) - reduced to 4x4
        if not hasattr(self, 'config') or self.config is None:
            # Use default values if config is not available
            position_variance = 10.0
            velocity_variance = 100.0
            self.get_logger().warn("Config not available in init_state_tracking - using default values")
        else:
            # Get values from config if available
            position_variance = float(self.config.get('initialization', {}).get('position_variance_initial', 10.0))
            velocity_variance = float(self.config.get('initialization', {}).get('velocity_variance_initial', 100.0))
            
        # Reduced 4x4 covariance matrix (x, y, vx, vy)
        self.covariance = np.eye(4, dtype=np.float32)
        self.covariance[0:2, 0:2] *= position_variance  # x,y position variance
        self.covariance[2:4, 2:4] *= velocity_variance  # x,y velocity variance
        
        # State tracking flags
        self.initialized = False
        self.tracking_reliable = False
        self.last_update_time = None
        self.consecutive_updates = 0
        
        # Uncertainty metrics
        self.position_uncertainty = float('inf')
        self.velocity_uncertainty = float('inf')
        
        # Define basketball properties
        self.basketball_radius = 0.1143  # 4.5 inches in meters (half of 9-inch diameter)
        self.basketball_z_height = self.basketball_radius  # Basketball center height above ground
        
        # Sensor health tracking - focus on horizontal plane reliability
        self.sensor_reliability = {
            'lidar': 0.5,
            'hsv_3d': 0.5,
            'yolo_3d': 0.5,
            'hsv_2d': 0.5,
            'yolo_2d': 0.6  # Slightly higher initial reliability for YOLO 2D
        }
        
        # Store base reliability for reverting after adjustments
        self.base_sensor_reliability = self.sensor_reliability.copy()
        
        # Pre-allocate filter matrices for efficiency (using float32)
        # Reduced to 4x4 for state transition and process noise
        self._F_matrix = np.eye(4, dtype=np.float32)  # State transition matrix
        self._Q_matrix = np.zeros((4, 4), dtype=np.float32)  # Process noise matrix
        
        # Special handling for measurement matrices:
        # For 3D sensors, we'll extract only x,y and ignore z (or set constant z)
        self._H_matrix_3d = np.zeros((3, 4), dtype=np.float32)  # Still has 3 rows for measurements
        self._H_matrix_2d = np.zeros((2, 4), dtype=np.float32)  # 2D measurement matrix
        
        # Set up constant components of measurement matrices
        self._H_matrix_3d[0, 0] = 1.0  # x position
        self._H_matrix_3d[1, 1] = 1.0  # y position
        # Note: The 3rd row (for z) is now not directly mapped to any state variable
        # We'll handle z separately since it's fixed at basketball height
        
        self._H_matrix_2d[0, 0] = 1.0  # x position
        self._H_matrix_2d[1, 1] = 1.0  # y position
        
        # History collections with adaptive length for Pi 5 memory management
        self.history_length = max(10, min(100, getattr(self, 'history_length', 100)))  # Bound history length
        self.position_history = deque(maxlen=self.history_length)
        self.velocity_history = deque(maxlen=self.history_length)
        self.time_history = deque(maxlen=self.history_length)
        self.innovation_history = deque(maxlen=self.history_length)
        
        # Initialize the ground position filter as a second stage
        from ball_chase.utilities.ground_position_filter import GroundPositionFilter
        self.ground_filter = GroundPositionFilter({
            "max_speed": 5.0,                # Maximum allowed speed in m/s
            "position_filter_alpha": 0.7,    # Position smoothing factor
            "ground_plane_z": self.basketball_radius,  # Basketball center height
            "basketball_radius": self.basketball_radius
        })
        
        self.get_logger().info(f"State tracking initialized with 4D state optimized for ground-only movement")
        
        # Initialize motion state tracking
        self.init_motion_state_tracking()
        
        # Initialize motion state protection
        self.motion_state_protection = {
            'long_stationary_confirmed_time': 0.0,  # When long_stationary was confirmed
            'long_stationary_established': False,   # Whether long_stationary is established
            'consecutive_stationary_after_long': 0, # Count of stationary detections after long_stationary
            'min_time_in_long_stationary': 2.0,     # Minimum time required in long_stationary
            'post_gap_cooldown_active': False,      # Whether we're in post-gap cooldown
            'post_gap_cooldown_end': 0.0,           # When post-gap cooldown ends
            'post_gap_protected_state': None,       # The state protected during cooldown
            'last_gap_recovery_time': 0.0,          # When last sensor recovery happened
            'protection_violation_count': 0         # Count of attempted transitions blocked by protection
        }
    
    def init_sensor_synchronization(self):
        """Initialize sensor synchronization system with extended buffer sizes."""
        # Create sensor buffer with increased buffer size and time tolerance
        self.sensor_buffer = SensorBuffer(max_time_diff=0.5)
        # Add parent reference for motion state access
        self.sensor_buffer.parent_node = self
        
        # Define all expected sensors
        self.expected_sensors = ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']
        
        # Initialize adaptive validation manager
        self.validation_manager = AdaptiveValidationManager(self.expected_sensors)
        
        # Increase buffer sizes for predictive buffering
        buffer_sizes = {
            'lidar': 30,     # Increase from 20 to 30
            'hsv_3d': 30,    # Increase from 20 to 30
            'yolo_3d': 30,   # Increase from 20 to 30
            'hsv_2d': 25,    # Increase from 20 to 25
            'yolo_2d': 25    # Increase from 20 to 25
        }
        
        # Add sensors to the buffer with larger buffer sizes
        for sensor in self.expected_sensors:
            size = buffer_sizes.get(sensor, 25)  # Default to 25 if not specified
            self.sensor_buffer.add_sensor(sensor, buffer_size=size)
        
        # Track last detection time for each sensor
        self.last_detection_time = {sensor: 0.0 for sensor in self.expected_sensors}
        self.sensor_counts = {sensor: 0 for sensor in self.expected_sensors}
        
        # Add sensor timing statistics for FPS calculation
        # Use larger buffer to better handle irregular timing
        self.sensor_frame_times = {sensor: deque(maxlen=40) for sensor in self.expected_sensors}
        self.sensor_fps = {sensor: 0.0 for sensor in self.expected_sensors}
        
        # Store bounding box information for distance estimation
        self.bbox_data = {
            'hsv_2d': {'width': 30, 'height': 30, 'timestamp': 0.0},
            'yolo_2d': {'width': 30, 'height': 30, 'timestamp': 0.0}
        }
        
        # Initialize sensor recovery tracking for all expected sensors
        self.sensor_gap_detection = {}
        for sensor in self.expected_sensors:
            self.sensor_gap_detection[sensor] = {
                'gap_detected': False,
                'gap_start_time': 0.0,
                'gap_level': 0.0,  # Track gap severity (0.0-1.0)
                'recent_gaps': deque(maxlen=5)  # Store recent gap durations for pattern analysis
            }
        
        # Create predictive buffer for position/velocity tracking
        self.position_prediction_buffer = deque(maxlen=10)  # Store recent predictions
        self.velocity_prediction_buffer = deque(maxlen=10)  # Store recent velocity predictions
        
        self.get_logger().info("Sensor synchronization system initialized with extended buffers")
        
        # Initialize sensor reliability tracker
        self.sensor_reliability_tracker = SensorReliabilityTracker(self.expected_sensors)

        # Initialize consecutive rejection tracking per sensor
        self.consecutive_rejections_per_sensor = {sensor: 0 for sensor in self.expected_sensors}

        # Initialize smoothed state estimator
        self.smoothed_state_estimator = SmoothedStateEstimator(window_size=5)

    def init_diagnostics(self):
        """Initialize diagnostic tracking."""
        # Performance tracking
        self.processing_times = deque(maxlen=100)
        
        # Error tracking
        self.errors = deque(maxlen=20)
        self.warnings = deque(maxlen=20)
        
        # Health metrics (0.0 to 1.0 scale)
        self.filter_health = 1.0
        self.transform_health = 0.0
        self.sensor_health = 0.0
        
        # When filter was last updated
        self.last_filter_update_time = 0.0
        
        self.get_logger().info("Diagnostic tracking initialized")

    def setup_publishers(self):
        """Set up lifecycle-managed publishers."""
        # Store publishers in list for lifecycle management
        self._publishers = []
        
        # Fused 3D position publisher
        self.position_pub = self.create_lifecycle_publisher(
            PointStamped,
            self.position_topic,
            10
        )
        self._publishers.append(self.position_pub)
        
        # Velocity publisher
        self.velocity_pub = self.create_lifecycle_publisher(
            TwistStamped,
            self.velocity_topic,
            10
        )
        self._publishers.append(self.velocity_pub)
        
        # Status publisher
        self.status_pub = self.create_lifecycle_publisher(
            Bool,
            self.status_topic,
            10
        )
        self._publishers.append(self.status_pub)
        
        # Uncertainty publisher
        self.uncertainty_pub = self.create_lifecycle_publisher(
            Float32,
            self.uncertainty_topic,
            10
        )
        self._publishers.append(self.uncertainty_pub)
        
        # Diagnostics publisher
        self.diagnostics_pub = self.create_lifecycle_publisher(
            String,
            self.diagnostics_topic,
            10
        )
        self._publishers.append(self.diagnostics_pub)
        
        self.get_logger().info("Publishers initialized")
        self.get_logger().info(f"Publishing to: {self.position_topic}, {self.velocity_topic}, {self.status_topic}, {self.uncertainty_topic}, {self.diagnostics_topic}")
        
    def setup_subscriptions(self):
        """Set up subscriptions (only called after transform is available)."""
        # 3D detections
        lidar_sub = self.create_subscription(
            PointStamped,
            self.lidar_topic,
            lambda msg: self.sensor_callback(msg, 'lidar'),
            10
        )
        self.subscribers.append(lidar_sub)
        
        hsv_3d_sub = self.create_subscription(
            PointStamped,
            self.hsv_3d_topic,
            lambda msg: self.sensor_callback(msg, 'hsv_3d'),
            10
        )
        self.subscribers.append(hsv_3d_sub)
        
        yolo_3d_sub = self.create_subscription(
            PointStamped,
            self.yolo_3d_topic,
            lambda msg: self.sensor_callback(msg, 'yolo_3d'),
             10
        )
        self.subscribers.append(yolo_3d_sub)
        
        # 2D detections
        hsv_2d_sub = self.create_subscription(
            PointStamped,
            self.hsv_2d_topic,
            lambda msg: self.sensor_callback(msg, 'hsv_2d'),
            10
        )
        self.subscribers.append(hsv_2d_sub)
        
        yolo_2d_sub = self.create_subscription(
            PointStamped,
            self.yolo_2d_topic,
            lambda msg: self.sensor_callback(msg, 'yolo_2d'),
            10
        )
        self.subscribers.append(yolo_2d_sub)
        
        # Bounding box subscriptions for distance estimation 
        # Use Float32MultiArray for YOLO
        from std_msgs.msg import Float32MultiArray
        
        yolo_bbox_sub = self.create_subscription(
            Float32MultiArray,
            self.yolo_bbox_topic,
            lambda msg: self.bbox_callback(msg, 'yolo_2d'),
            10
        )
        self.subscribers.append(yolo_bbox_sub)
        
        # Try to use BoundingBox2D for HSV if that's what the HSV node publishes
        try:
            from vision_msgs.msg import BoundingBox2D
            
            hsv_bbox_sub = self.create_subscription(
                BoundingBox2D,
                self.hsv_bbox_topic,
                lambda msg: self.bbox_callback_standard(msg, 'hsv_2d'),
                10
            )
            self.subscribers.append(hsv_bbox_sub)
        except ImportError:
            self.get_logger().warn("vision_msgs not available - standard bbox processing disabled")
        
        self.get_logger().info("Subscriptions initialized")
        self.get_logger().info(f"Subscribed to: {self.lidar_topic}, {self.hsv_3d_topic}, {self.yolo_3d_topic}, {self.hsv_2d_topic}, {self.yolo_2d_topic}")
            
        # Start fresh - we're subscribing only now, so messages received previously won't be processed
        # Reset the "last seen" timestamps
        current_time = time.time()
        for sensor in self.last_detection_time:
            self.last_detection_time[sensor] = current_time

    def setup_timers(self):
        """Set up regular processing timers."""
        # Status timer (1 Hz)
        status_timer = self.create_timer(1.0, self.publish_status)
        self._timer_list.append(status_timer)
        
        # Kalman filter update timer (20 Hz)
        filter_timer = self.create_timer(0.05, self.filter_update)
        self._timer_list.append(filter_timer)
        
        # Diagnostics timer (1 Hz)
        diagnostics_timer = self.create_timer(1.0, self.publish_diagnostics)
        self._timer_list.append(diagnostics_timer)
        
        # Transform check timer (5 Hz) - only keep until transform confirmed
        if not self.transform_confirmed:
            self._transform_check_timer = self.create_timer(5.0, self.check_transform_availability)
            self._timer_list.append(self._transform_check_timer)
            self.get_logger().info("Transform check timer started - will be disabled once transforms are confirmed")
        
        self.get_logger().info("Processing timers initialized")        

    def sensor_callback(self, msg, source):
        """Modified sensor callback with early initialization"""
        # Skip if not active yet
        if not self.is_activated:
            return
        
        try:
            # Get current time for timing statistics
            current_time = time.time()
            
            # Update sensor statistics
            self.sensor_counts[source] += 1
            self.last_detection_time[source] = current_time
            
            # Track frame time for FPS calculation
            self.sensor_frame_times[source].append(current_time)
            
            # Calculate FPS based on recent frames
            if len(self.sensor_frame_times[source]) >= 2:
                time_span = current_time - self.sensor_frame_times[source][0]
                if time_span > 0:
                    self.sensor_fps[source] = (len(self.sensor_frame_times[source]) - 1) / time_span
            
            # Add to synchronization buffer
            self.sensor_buffer.add_measurement(source, msg, msg.header.stamp)
            
            # If not initialized yet, try EARLY fast initialization
            if not self.initialized:
                # Check if LiDAR measurement (prioritize)
                if source == 'lidar':
                    self.get_logger().info(f"Received first {source} data - attempting fast initialization")
                    transformed = self.transform_point(msg, self.reference_frame, False)
                    if transformed:
                        # Try immediate initialization with first LiDAR measurement
                        self.fast_initialize_with_first_measurement(transformed, source)
                
                # If not LiDAR and still not initialized, try with 2D measurement
                elif not self.initialized and source == 'yolo_2d' and 'yolo_2d' in self.bbox_data:
                    self.get_logger().info(f"Received first {source} data - attempting fast initialization with 2D sensor")
                    # Estimate 3D position from 2D detection
                    estimated_3d = self.estimate_3d_from_2d(msg, self.bbox_data['yolo_2d'])
                    if estimated_3d:
                        # Try initialization with estimated 3D position from 2D
                        self.fast_initialize_with_first_measurement(estimated_3d, f"{source}_est3d")
            
            # For 2D YOLO data, also estimate 3D position
            if source == 'yolo_2d' and 'yolo_2d' in self.bbox_data:
                estimated_3d_point = self.estimate_3d_from_2d(msg, self.bbox_data['yolo_2d'])
                if estimated_3d_point:
                    self.sensor_buffer.add_measurement('yolo_2d_est3d', estimated_3d_point, msg.header.stamp)
            
            # Log first few detections with more detail
            if self.sensor_counts[source] <= 3:
                self.get_logger().info(
                    f"Received {source} detection #{self.sensor_counts[source]}: "
                    f"({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f}) in {msg.header.frame_id} frame"
                )
            
            # Increment lidar message counter and log every 3 messages
            if source == 'lidar':
                self.lidar_msg_counter += 1
                if self.lidar_msg_counter % 3 == 0:
                    self.get_logger().info(
                        f"Received lidar detection #{self.lidar_msg_counter}: "
                        f"({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f}) in {msg.header.frame_id} frame"
                    )
        except Exception as e:
            self.log_error(f"Error in {source} callback: {str(e)}")
                
    def bbox_callback(self, msg, source):
        """
        Callback for bounding box messages.
        
        Args:
            msg (Float32MultiArray): The bounding box message in Float32MultiArray format
            source (str): Source identifier (e.g., 'hsv_2d', 'yolo_2d')
        """
        # Skip if not active yet
        if not self.is_activated:
            return
        
        try:
            # Handle Float32MultiArray format for yolo
            if hasattr(msg, 'data') and hasattr(msg.data, '__len__') and len(msg.data) >= 4:                
                # Format: [center_x, center_y, width, height, confidence]
                width = msg.data[2]   # width is the 3rd value (index 2)
                height = msg.data[3]  # height is the 4th value (index 3)
                
                # Store the bounding box data with timestamp
                if source in self.bbox_data:                
                    self.bbox_data[source]['width'] = width
                    self.bbox_data[source]['height'] = height
                    self.bbox_data[source]['timestamp'] = time.time()
                    
                    # Initialize counter if not present
                    if not hasattr(self, '_bbox_log_counter'):
                        self._bbox_log_counter = {}
                    if source not in self._bbox_log_counter:
                        self._bbox_log_counter[source] = 0
                    
                    # Increment counter and log details every 3 times
                    self._bbox_log_counter[source] += 1
                    if self._bbox_log_counter[source] % 3 == 0:
                        self.get_logger().info(f"Received {source} bbox: {width:.1f}x{height:.1f}")
            else:
                self.get_logger().warn(f"Invalid format for {source} bounding box message")
                
        except Exception as e:
            self.log_error(f"Error in {source} bbox callback: {str(e)}")

    def bbox_callback_standard(self, msg, source):
        """
        Callback for standard BoundingBox2D messages.
        
        Args:
            msg (BoundingBox2D): The bounding box message
            source (str): Source identifier (e.g., 'hsv_2d')
        """
        # Skip if not active yet
        if not self.is_activated:
            return
        
        try:
            # Handle BoundingBox2D format
            if hasattr(msg, 'size_x') and hasattr(msg, 'size_y'):
                width = msg.size_x
                height = msg.size_y
                
                # Store the bounding box data with timestamp
                if source in self.bbox_data:
                    self.bbox_data[source]['width'] = width
                    self.bbox_data[source]['height'] = height
                    self.bbox_data[source]['timestamp'] = time.time()
                    
                    # Initialize counter if not present
                    if not hasattr(self, '_bbox_log_counter'):
                        self._bbox_log_counter = {}
                    if source not in self._bbox_log_counter:
                        self._bbox_log_counter[source] = 0
                    
                    # Increment counter and log details every 3 times
                    self._bbox_log_counter[source] += 1
                    if self._bbox_log_counter[source] % 3 == 0:
                        self.get_logger().info(f"Received {source} bbox: {width:.1f}x{height:.1f}")
            else:
                self.get_logger().warn(f"Invalid format for {source} bounding box message")
                
        except Exception as e:
            self.log_error(f"Error in {source} bbox callback: {str(e)}")

    def publish_status(self):
        """Publish and log brief status information."""
        # Skip if not active
        if not self.is_activated:
            return
        
        # Calculate uptime
        uptime = time.time() - self.start_time
        current_time = time.time()
        
        # Publish tracking status
        status_msg = Bool()
        status_msg.data = bool(self.tracking_reliable)
        self.status_pub.publish(status_msg)
        
        # Count active 2D and 3D sensors
        active_3d = sum(1 for sensor, last_time in self.last_detection_time.items() 
                        if not sensor.endswith('_2d') and current_time - last_time < 1.0)
        active_2d = sum(1 for sensor, last_time in self.last_detection_time.items() 
                        if sensor.endswith('_2d') and current_time - last_time < 1.0)
        
        # Determine operating mode
        if active_3d >= 1:
            mode = "3D tracking"
        elif active_2d >= 1 and self.allow_tracking_with_2d_only:
            mode = "2D-only tracking"
        else:
            mode = "Limited tracking"
        
        # Log basic status
        transform_status = "Confirmed" if self.transform_confirmed else ("OK" if self.transform_available else "Missing")
        self.get_logger().info(
            f"Status: Uptime={uptime:.1f}s, Transform={transform_status}, "
            f"Mode={mode}, 3D sensors={active_3d}, 2D sensors={active_2d}, "
            f"Initialized={self.initialized}, Tracking={self.tracking_reliable}, "
            f"Uncertainty={self.position_uncertainty:.3f}m, "
            f"Motion={self.motion_state}"
        )
        
        # Add sensor timing information to status
        active_sensors = []
        for sensor in ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']:
            if self.sensor_counts.get(sensor, 0) > 0:  # Only include if we've received data
                delay = current_time - self.last_detection_time.get(sensor, 0)
                fps = self.sensor_fps.get(sensor, 0.0)
                count = self.sensor_counts.get(sensor, 0)
                active_sensors.append(f"{sensor}: count={count}, {delay:.1f}s ago, {fps:.1f} FPS")
        if active_sensors:
            self.get_logger().info(f"Sensor data: {' | '.join(active_sensors)}")
        elif self.initialized:  # Only show warning if we're initialized
            self.get_logger().warn("No sensor data received - check if sensor nodes are running")    

    def detect_motion_state(self):
        """
        Detect the current motion state of the object with improved responsiveness
        to movement transitions, especially from stationary states.
        
        Returns:
            str: One of "stationary", "long_stationary", "small_movement", "medium_fast", or "unknown"
        """
        # If not initialized or insufficient velocity history, return unknown
        if not self.initialized or len(self.velocity_history) < 5:
            return "unknown"
        
        current_time = time.time()
        
        # Get the most recent velocity estimates
        recent_velocities = list(self.velocity_history)[-5:]
        
        # Calculate the average magnitude of these velocities
        valid_velocities = [vel for vel in recent_velocities if isinstance(vel, (list, tuple, np.ndarray)) and len(vel) >= 3]
        if not valid_velocities:
            return "unknown"
            
        # Calculate average velocity with gap-aware filtering
        filtered_velocities, avg_velocity, implausible_detected = self.process_velocity_measurements(valid_velocities, list(self.time_history)[-5:])
        
        # Initialize confidence dictionary if needed
        if not hasattr(self, 'motion_state_confidence'):
            self.motion_state_confidence = {
                "stationary": 0.5,
                "long_stationary": 0.0,
                "small_movement": 0.0, 
                "medium_fast": 0.0,
                "unknown": 0.0
            }
        
        # IMPROVED: Faster confidence changes for movement evidence
        # Update confidence values based on current velocity evidence with more aggressive changes
        if avg_velocity < 0.03:
            # Evidence for stationary state
            self.motion_state_confidence["stationary"] = min(1.0, self.motion_state_confidence["stationary"] + 0.1)
            self.motion_state_confidence["small_movement"] = max(0.0, self.motion_state_confidence["small_movement"] - 0.1)
            self.motion_state_confidence["medium_fast"] = max(0.0, self.motion_state_confidence["medium_fast"] - 0.2)
        elif avg_velocity < 0.20:  # REDUCED threshold from 0.25 to 0.20
            # Evidence for small movement - INCREASE confidence gain rate
            self.motion_state_confidence["small_movement"] = min(1.0, self.motion_state_confidence["small_movement"] + 0.15)  # Increased from 0.1
            self.motion_state_confidence["stationary"] = max(0.0, self.motion_state_confidence["stationary"] - 0.1)  # Increased from 0.05
            self.motion_state_confidence["medium_fast"] = max(0.0, self.motion_state_confidence["medium_fast"] - 0.1)
        else:
            # Evidence for medium/fast movement - INCREASE confidence gain rate
            self.motion_state_confidence["medium_fast"] = min(1.0, self.motion_state_confidence["medium_fast"] + 0.2)  # Increased from 0.15
            self.motion_state_confidence["small_movement"] = max(0.0, self.motion_state_confidence["small_movement"] - 0.1)  # Increased from 0.05
            self.motion_state_confidence["stationary"] = max(0.0, self.motion_state_confidence["stationary"] - 0.2)  # Increased from 0.15
        
        # Special handling for long_stationary confidence - FASTER DECAY
        if self.motion_state == "long_stationary":
            # GREATLY reduce confidence decay to be more responsive to new movement
            self.motion_state_confidence["long_stationary"] = max(0.4, self.motion_state_confidence["long_stationary"] * 0.9)  # Reduced from 0.95/0.7
        
        # Get average acceleration using velocity history
        acceleration = 0.0
        if len(self.velocity_history) >= 3 and len(self.time_history) >= 3:
            recent_vels = list(self.velocity_history)[-3:]
            recent_times = list(self.time_history)[-3:]
            
            if len(recent_vels) >= 2 and len(recent_times) >= 2:
                # Calculate change in velocity magnitude over time
                vel1_mag = math.sqrt(recent_vels[-2][0]**2 + recent_vels[-2][1]**2)
                vel2_mag = math.sqrt(recent_vels[-1][0]**2 + recent_vels[-1][1]**2)
                dt = recent_times[-1] - recent_times[-2]
                
                if dt > 0:
                    acceleration = abs(vel2_mag - vel1_mag) / dt

        # Apply REDUCED hysteresis for state classification based on current state
        current_state = getattr(self, 'motion_state', 'unknown')
        # REDUCED default thresholds
        stationary_thresh = 0.012  # Reduced from 0.015 m/s
        small_movement_thresh = 0.13  # Reduced from 0.15 m/s
        
        # Apply reduced hysteresis factors
        if current_state == "stationary":
            # Less hysteresis to leave stationary
            stationary_thresh = 0.020  # Reduced from 0.025
        elif current_state == "small_movement":
            # Adjusted thresholds for small_movement
            stationary_thresh = 0.008  # Reduced from 0.01
            small_movement_thresh = 0.15  # Reduced from 0.18
        elif current_state == "medium_fast":
            # Lower threshold to stay in medium_fast
            small_movement_thresh = 0.11  # Reduced from 0.13
        elif current_state == "long_stationary":
            # GREATLY REDUCED threshold to leave long_stationary - KEY CHANGE
            stationary_thresh = 0.025  # Reduced from 0.05 m/s
        
        # Use acceleration to detect rapid changes
        if acceleration > 1.5:  # Reduced from 2.0 for more sensitivity
            # Skip intermediate states for rapid acceleration
            base_motion_state = "medium_fast"
        elif avg_velocity < stationary_thresh:
            base_motion_state = "stationary"
        elif avg_velocity < small_movement_thresh:
            base_motion_state = "small_movement"
        else:
            base_motion_state = "medium_fast"
        
        # Initialize tracking attributes if they don't exist
        if not hasattr(self, 'stationary_start_time'):
            self.stationary_start_time = None
            
        if base_motion_state == "stationary":
            # Start or continue tracking stationary time
            if self.stationary_start_time is None:
                self.stationary_start_time = current_time
                motion_state = "stationary"
            else:
                # INCREASED time threshold for long-term stationary (5+ seconds)
                stationary_duration = current_time - self.stationary_start_time
                if stationary_duration > 5.0:  # No change here
                    motion_state = "long_stationary"
                else:
                    motion_state = "stationary"
        else:
            # Special handling for long_stationary state - IMPROVED RESPONSIVENESS
            if hasattr(self, 'motion_state') and self.motion_state == "long_stationary":
                # DRASTICALLY REDUCED threshold for exiting long_stationary
                if avg_velocity > 0.008:  # Changed from 0.025 to 0.008 m/s (less than 1cm/s)
                    # Log the override with increased visibility
                    self.get_logger().info(f"Movement detected: velocity {avg_velocity:.3f}m/s exceeds minimal threshold 0.008m/s - exiting long_stationary state")
                    motion_state = base_motion_state
                elif base_motion_state in ["small_movement", "medium_fast"]:
                    # Get confidence values
                    long_stationary_confidence = self.motion_state_confidence.get("long_stationary", 0.0)
                    movement_confidence = self.motion_state_confidence.get(base_motion_state, 0.0)
                    
                    # DRAMATICALLY REDUCED threshold to allow state transition
                    if movement_confidence > long_stationary_confidence * 0.05:  # Changed from 0.15 to 0.05 (5%)
                        self.get_logger().info(
                            f"Allowing movement transition: {base_motion_state} confidence ({movement_confidence:.2f}) > "
                            f"{long_stationary_confidence * 0.05:.2f} threshold (5% rule)"
                        )
                        motion_state = base_motion_state
                    else:
                        # Maintain long_stationary but with SEVERELY reduced confidence
                        motion_state = "long_stationary"
                        # Decay confidence MUCH faster when movement is detected
                        self.motion_state_confidence["long_stationary"] = max(0.2, long_stationary_confidence * 0.6)  # Changed from 0.8 to 0.6
                else:
                    motion_state = "long_stationary"
                    
                # ADDED: Additional protection against getting stuck
                # If we've spent too long in long_stationary, force a reset of confidence values
                if hasattr(self, 'stationary_start_time') and self.stationary_start_time is not None:
                    stationary_duration = current_time - self.stationary_start_time
                    if stationary_duration > 15.0:  # After 15 seconds in stationary state
                        # Reset confidence values to make transitions easier
                        self.motion_state_confidence["long_stationary"] = max(0.2, self.motion_state_confidence.get("long_stationary", 0.5) * 0.5)
                        self.motion_state_confidence["stationary"] = max(0.2, self.motion_state_confidence.get("stationary", 0.5) * 0.5)
                        # Boost movement confidence slightly
                        self.motion_state_confidence["small_movement"] = min(0.4, self.motion_state_confidence.get("small_movement", 0.0) + 0.1)
                        
                        # Log this intervention
                        if stationary_duration % 5.0 < 0.1:  # Log only once every ~5 seconds
                            self.get_logger().warn(
                                f"Extended stationary period detected ({stationary_duration:.1f}s) - "
                                f"reducing stationary confidence to prevent stuckness"
                            )
            else:
                # Reset stationary timer when moving
                self.stationary_start_time = None
                motion_state = base_motion_state
        
        # Log if motion state changes
        if motion_state != getattr(self, 'motion_state', 'unknown'):
            self.prev_motion_state = getattr(self, 'motion_state', 'unknown')
            self.motion_state = motion_state
            
            # Add confidence information to the log
            confidence_str = ""
            if hasattr(self, 'motion_state_confidence'):
                from_conf = self.motion_state_confidence.get(self.prev_motion_state, 0.0)
                to_conf = self.motion_state_confidence.get(self.motion_state, 0.0)
                confidence_str = f", confidence={to_conf:.2f}"
                
            self.get_logger().info(f"Motion state changed: {self.prev_motion_state} -> {self.motion_state} "
                                f"(velocity={avg_velocity:.3f}m/s{confidence_str})")
        
        return self.motion_state
    
    # Always enforce ground height constraint through the ground filter
    # This happens in publish_state when we run the state through the ground filter    
    def apply_flat_ground_constraints(self):
        """
        Apply flat ground constraints for a basketball that only moves on the ground.
        Optimized for 4D state vector [x, y, vx, vy].
        """
        # Since we know the basketball only moves on the ground,
        # we don't need to detect flat ground - it's always on the ground
        
        # For a 4D state, we don't have z or vz components
        # Instead, we use the ground filter's constraints directly
        
        # Update our velocity estimates based on recent positions if we have history
        if len(self.position_history) >= 3:
            recent_positions = list(self.position_history)[-3:]
            
            # If we have time history as well, calculate recent velocities
            if hasattr(self, 'time_history') and len(self.time_history) >= 3:
                recent_times = list(self.time_history)[-3:]
                
                # Calculate velocity between last two positions
                if recent_times[-1] > recent_times[-2]:  # Ensure time moved forward
                    dt = recent_times[-1] - recent_times[-2]
                    if dt > 0:
                        vx = (recent_positions[-1][0] - recent_positions[-2][0]) / dt
                        vy = (recent_positions[-1][1] - recent_positions[-2][1]) / dt
                        
                        # Apply physics constraints for a ball rolling on the ground
                        # (maximum speed, rolling friction, etc.)
                        speed = math.sqrt(vx*vx + vy*vy)
                        max_speed = 5.0  # Maximum rolling speed for basketball
                        
                        if speed > max_speed:
                            # Scale velocity to maximum speed
                            scale = max_speed / speed
                            vx *= scale
                            vy *= scale
                            
                        # Update state with constrained velocity
                        self.state[2] = vx
                        self.state[3] = vy

    
    def adjust_fusion_state_toward_sensor_data(self, sensor_pos, current_pos, motion_state):
        """
        Apply a more aggressive blend factor to adjust fusion state toward sensor data,
        especially during transitions from stationary to movement states.
        
        Args:
            sensor_pos (list): Position from sensor data [x, y, z]
            current_pos (list): Current fusion position estimate [x, y, z]
            motion_state (str): Current detected motion state
        
        Returns:
            list: Adjusted position after blending
        """
        # Calculate distance between sensor and fusion position
        dx = sensor_pos[0] - current_pos[0]
        dy = sensor_pos[1] - current_pos[1]
        distance_diff = math.sqrt(dx*dx + dy*dy)
        
        # Base blend factor - increased to 0.85 to be much more responsive
        blend_factor = 0.85
        
        # Apply even higher blend factors during transitions
        if hasattr(self, 'prev_motion_state') and self.motion_state != self.prev_motion_state:
            # If transitioning from stationary to any movement state
            if self.prev_motion_state in ["stationary", "long_stationary"] and \
            self.motion_state not in ["stationary", "long_stationary"]:
                # Use an extremely aggressive blend factor (95%) to rapidly trust sensor data
                blend_factor = 0.95
                self.get_logger().info(
                    f"Using aggressive blend factor during transition from {self.prev_motion_state} to {self.motion_state}: {blend_factor:.2f}"
                )
        
        # Apply special handling for long_stationary state
        if self.motion_state == "long_stationary" and distance_diff > 0.1:
            # If significant discrepancy during long_stationary, trust sensor more
            blend_factor = 0.90
            self.get_logger().info(
                f"Using higher blend factor to escape long_stationary: {blend_factor:.2f} (distance_diff={distance_diff:.3f}m)"
            )
        
        # Apply even more aggressive factors for larger discrepancies
        if distance_diff > 0.3:  # Lowered from 0.5 to be more sensitive
            blend_factor = min(0.98, blend_factor + 0.1)  # Increased cap to 0.98
            self.get_logger().info(
                f"Applying high blend factor for large discrepancy: {blend_factor:.2f} (distance_diff={distance_diff:.3f}m)"
            )
        
        # Apply more trust to LiDAR data which is typically more accurate
        if isinstance(sensor_pos, dict) and sensor_pos.get('source', '') == 'lidar':
            blend_factor = min(0.98, blend_factor + 0.1)
        elif isinstance(sensor_pos, list) and hasattr(self, 'latest_sensor_source') and getattr(self, 'latest_sensor_source') == 'lidar':
            blend_factor = min(0.98, blend_factor + 0.1)
        
        # Apply blend with strong minimum threshold to ensure some movement
        min_blend = 0.7  # Ensure we blend at least 70% of the sensor data
        blend_factor = max(min_blend, blend_factor)
        
        # Apply blend
        blended_pos = [
            (1 - blend_factor) * current_pos[0] + blend_factor * sensor_pos[0],
            (1 - blend_factor) * current_pos[1] + blend_factor * sensor_pos[1],
            self.basketball_z_height  # Keep fixed height
        ]
        
        # Update state with blended position
        self.state[0] = blended_pos[0]
        self.state[1] = blended_pos[1]
        
        # Log this adjustment
        if self.debug_level >= 1:
            self.get_logger().info(
                f"Adjusted fusion state toward sensor data: blend={blend_factor:.2f}, "
                f"motion={self.motion_state}, distance_diff={distance_diff:.3f}m, uncertainty={self.position_uncertainty:.3f}m"
            )
        
        return blended_pos

    def handle_sensor_recovery(self):
        """
        Monitor sensor availability patterns and handle recovery after gaps with
        improved responsiveness to resume tracking quickly after temporary sensor losses.
        """
        current_time = time.time()
        recovery_detected = False
        recovered_sensors = []
        
        for sensor in ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']:
            # Skip sensors we haven't seen yet
            if self.sensor_counts.get(sensor, 0) == 0:
                continue
                
            last_time = self.last_detection_time.get(sensor, 0)
            gap_duration = current_time - last_time
            
            # Initialize recovery tracking if needed or ensure all fields exist
            if sensor not in self.sensor_gap_detection:
                self.sensor_gap_detection[sensor] = {
                    'gap_detected': False,
                    'gap_start_time': 0.0,
                    'gap_level': 0.0,  # Track gap severity (0.0-1.0)
                    'recent_gaps': deque(maxlen=5),  # Store recent gap durations for pattern analysis
                    'last_recovery_time': 0.0,  # Track when sensor last recovered
                    'recovery_boost_active': False,  # Track if recovery boost is active
                    'recovery_boost_end': 0.0  # When recovery boost period ends
                }
            else:
                # Ensure all required keys exist (this fixes the KeyError)
                required_keys = ['gap_detected', 'gap_start_time', 'gap_level', 'recent_gaps', 
                                'last_recovery_time', 'recovery_boost_active', 'recovery_boost_end']
                for key in required_keys:
                    if key not in self.sensor_gap_detection[sensor]:
                        if key == 'recent_gaps':
                            self.sensor_gap_detection[sensor][key] = deque(maxlen=5)
                        else:
                            self.sensor_gap_detection[sensor][key] = False if key == 'gap_detected' or key == 'recovery_boost_active' else 0.0
            
            # First check if sensor was in gap state but now has fresh data
            if self.sensor_gap_detection[sensor]['gap_detected']:
                # Get newest measurement
                msg = self.sensor_buffer.get_latest_measurement(sensor)
                if msg is not None and gap_duration < 0.5:  # Fresh data - sensor has recovered
                    total_gap = current_time - self.sensor_gap_detection[sensor]['gap_start_time']
                    
                    # Log recovery with higher visibility if significant gap
                    if total_gap > 1.0:
                        self.get_logger().info(f"{sensor} recovered after {total_gap:.1f}s gap")
                    else:
                        self.get_logger().debug(f"{sensor} recovered after {total_gap:.1f}s gap")
                    
                    # IMPROVED: Apply much more aggressive covariance adjustment for faster recovery
                    # Scale by gap duration - longer gaps need more adjustment
                    adjustment_factor = min(5.0, 1.5 + (total_gap / 0.5))  # Massively increased from 3.0
                    
                    # Directly modify the state uncertainty to allow faster state changes
                    if hasattr(self, 'covariance'):
                        self.covariance[0:2, 0:2] *= adjustment_factor  # Increase position uncertainty
                        self.covariance[2:4, 2:4] *= adjustment_factor * 2.0  # Increase velocity uncertainty more
                        
                        # Ensure covariance remains symmetric
                        self.covariance = 0.5 * (self.covariance + self.covariance.T)
                        
                        # Update uncertainty metrics
                        self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:2, 0:2]) / 2.0)
                        self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[2:4, 2:4]) / 2.0)
                        
                        self.get_logger().info(
                            f"Applied gap recovery boost: factor={adjustment_factor:.1f}, "
                            f"new uncertainty={self.position_uncertainty:.3f}m"
                        )
                    
                    # ENHANCED: Set up longer recovery boost period for significant gaps
                    self.sensor_gap_detection[sensor]['recovery_boost_active'] = True
                    self.sensor_gap_detection[sensor]['recovery_boost_end'] = current_time + min(4.0, total_gap * 1.5)  # Extended from 2.0
                    self.sensor_gap_detection[sensor]['last_recovery_time'] = current_time
                    
                    # Store gap duration for pattern analysis
                    self.sensor_gap_detection[sensor]['recent_gaps'].append(total_gap)
                    
                    # Clear gap flag
                    self.sensor_gap_detection[sensor]['gap_detected'] = False
                    self.sensor_gap_detection[sensor]['gap_level'] = 0.0
                    
                    # Mark that a recovery was detected this cycle
                    recovery_detected = True
                    recovered_sensors.append(sensor)
            
            # IMPROVED: Detect gaps earlier
            elif gap_duration > 0.5:  # REDUCED threshold from 0.8s to 0.5s to detect gaps earlier
                # New gap detected
                if not self.sensor_gap_detection[sensor]['gap_detected']:
                    self.sensor_gap_detection[sensor]['gap_detected'] = True
                    self.sensor_gap_detection[sensor]['gap_start_time'] = current_time
                    
                    # ADDED: Calculate gap level based on sensor importance
                    importance = 1.0  # Default importance
                    if sensor == 'lidar':
                        importance = 1.5  # Lidar is more important
                    elif sensor.endswith('_2d'):
                        importance = 0.7  # 2D sensors are less important
                    
                    # Set initial gap level
                    self.sensor_gap_detection[sensor]['gap_level'] = 0.3 * importance
                    
                    # Log at debug level for shorter gaps
                    if gap_duration < 1.5:
                        self.get_logger().debug(f"{sensor} gap detected: {gap_duration:.1f}s")
                    else:
                        self.get_logger().info(f"{sensor} gap detected: {gap_duration:.1f}s")
                else:
                    # Update gap level based on duration (up to a maximum of 1.0)
                    duration_factor = min(1.0, gap_duration / 3.0)  # Normalize up to 3 seconds
                    self.sensor_gap_detection[sensor]['gap_level'] = duration_factor
            
            # FIXED: Check if recovery boost should still be active (with safe access)
            elif self.sensor_gap_detection[sensor].get('recovery_boost_active', False):
                if current_time > self.sensor_gap_detection[sensor].get('recovery_boost_end', 0.0):
                    # Deactivate recovery boost
                    self.sensor_gap_detection[sensor]['recovery_boost_active'] = False
                    self.get_logger().debug(f"Recovery boost for {sensor} ended")
        
        # ENHANCED: Consider state transition if recovery detected - much more aggressive
        if recovery_detected and hasattr(self, 'motion_state'):
            # If recovering from gap and currently in a stationary state, check for movement evidence
            if self.motion_state in ["stationary", "long_stationary"]:
                # Get average of recent velocities
                if len(self.velocity_history) >= 3:
                    recent_vels = list(self.velocity_history)[-3:]
                    avg_velocity = np.mean([math.sqrt(v[0]**2 + v[1]**2) for v in recent_vels])
                    
                    # Use an extremely low threshold for movement detection after recovery - 1 cm/s
                    if avg_velocity > 0.01:  # Drastically reduced from 0.02
                        self.prev_motion_state = self.motion_state
                        
                        # Skip directly to medium_fast state for better responsiveness
                        self.motion_state = "medium_fast"  # Changed from small_movement
                        
                        self.get_logger().info(
                            f"Forcing aggressive state transition after sensor recovery: {self.prev_motion_state} -> "
                            f"{self.motion_state} (velocity={avg_velocity:.3f}m/s)"
                        )
                        
                        # Update motion state confidence values - strongly favor movement
                        if hasattr(self, 'motion_state_confidence'):
                            self.motion_state_confidence["medium_fast"] = 0.8  # Boosted from 0.7
                            self.motion_state_confidence["small_movement"] = 0.5  # Added
                            self.motion_state_confidence["stationary"] = 0.2  # Reduced from 0.3
                            self.motion_state_confidence["long_stationary"] = 0.1  # Reduced from 0.2
                
                # Also check for any sensor data suggesting movement
                for sensor in recovered_sensors:
                    msg = self.sensor_buffer.get_latest_measurement(sensor)
                    if msg is not None:
                        # Transform to reference frame if needed
                        if msg.header.frame_id != self.reference_frame:
                            transformed = self.transform_point(msg, self.reference_frame, sensor.endswith('_2d'))
                            if transformed is None:
                                continue
                            pos = [transformed.point.x, transformed.point.y, transformed.point.z]
                        else:
                            pos = [msg.point.x, msg.point.y, msg.point.z]
                        
                        # Check distance from current state
                        dx = pos[0] - self.state[0]
                        dy = pos[1] - self.state[1]
                        distance_diff = math.sqrt(dx*dx + dy*dy)
                        
                        # If sensor position differs significantly from state, force movement state
                        if distance_diff > 0.15:  # Reduced from 0.3 (15cm threshold)
                            self.prev_motion_state = self.motion_state
                            
                            # Skip directly to medium_fast state for better responsiveness
                            self.motion_state = "medium_fast"  # Changed from small_movement
                            
                            self.get_logger().info(
                                f"Forcing state transition based on sensor position: {self.prev_motion_state} -> "
                                f"{self.motion_state} (position_diff={distance_diff:.3f}m)"
                            )
                            
                            # Update motion state confidence values - strongly favor movement
                            if hasattr(self, 'motion_state_confidence'):
                                self.motion_state_confidence["medium_fast"] = 0.8
                                self.motion_state_confidence["small_movement"] = 0.5
                                self.motion_state_confidence["stationary"] = 0.2
                                self.motion_state_confidence["long_stationary"] = 0.1
                            
                            break  # Exit loop after forcing state change
        
        # ENHANCED: Apply global adjustments when multiple sensors recover simultaneously
        if len(recovered_sensors) >= 2:
            # Major recovery event - force state reevaluation
            self.get_logger().info(f"Multiple sensor recovery detected: {', '.join(recovered_sensors)}")
            
            # Reset consecutive rejections to accept new measurements more easily
            if hasattr(self, 'consecutive_rejections_per_sensor'):
                for sensor in recovered_sensors:
                    self.consecutive_rejections_per_sensor[sensor] = 0
                    
            # Increase uncertainty to accept more measurements
            if hasattr(self, 'covariance'):
                # Apply a much larger uncertainty boost for multi-sensor recovery
                multi_recovery_factor = 3.0  # Doubled from 1.5
                self.covariance[0:2, 0:2] *= multi_recovery_factor
                self.covariance[2:4, 2:4] *= multi_recovery_factor * 1.5  # Additional boost for velocity
                
                # Update uncertainty metrics
                self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:2, 0:2]) / 2.0)
                self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[2:4, 2:4]) / 2.0)
                
                self.get_logger().info(
                    f"Applied multi-sensor recovery boost: factor={multi_recovery_factor:.1f}, "
                    f"new uncertainty={self.position_uncertainty:.3f}m"
                )
                
                # ADDED: Force medium_fast motion state after multiple sensor recovery
                if hasattr(self, 'motion_state'):
                    self.prev_motion_state = self.motion_state
                    self.motion_state = "medium_fast"
                    self.get_logger().info(
                        f"Forcing motion state to medium_fast after multi-sensor recovery (from {self.prev_motion_state})"
                    )
                    
                    # Update confidence values
                    if hasattr(self, 'motion_state_confidence'):
                        self.motion_state_confidence["medium_fast"] = 0.9
                        self.motion_state_confidence["small_movement"] = 0.6
                        self.motion_state_confidence["stationary"] = 0.1
                        self.motion_state_confidence["long_stationary"] = 0.0
    
    def predict_state(self, dt):
        """
        Predict the state forward by dt seconds, optimized for ground-only movement
        with enhanced responsiveness to state transitions.
        
        Args:
            dt (float): Time step in seconds
        """
        # Reset the state transition matrix to identity - 4D state vector
        self._F_matrix = np.eye(4, dtype=np.float32)
        
        # Set time-dependent values for x,y position updates from velocity
        self._F_matrix[0, 2] = dt  # x += vx*dt
        self._F_matrix[1, 3] = dt  # y += vy*dt
        
        # Reset the process noise matrix to zeros
        self._Q_matrix = np.zeros((4, 4), dtype=np.float32)
        
        # Apply adaptive process noise based on prediction duration and motion state
        gap_factor = min(5.0, max(1.0, dt / 0.1))  # Scale factor based on gap length
        
        # ADDED: Apply motion state-based scaling to process noise
        motion_state = getattr(self, 'motion_state', 'unknown')
        motion_scale = 1.0  # Default scaling
        
        if motion_state == "stationary":
            motion_scale = 1.2  # Increased from default for stationary
        elif motion_state == "long_stationary":
            motion_scale = 1.5  # Much higher for long_stationary to allow escaping
        elif motion_state == "small_movement":
            motion_scale = 1.3  # Higher for small_movement
        elif motion_state == "medium_fast":
            motion_scale = 1.1  # Slightly higher for medium_fast
        
        # ADDED: Check for recent state transitions and apply higher noise
        if hasattr(self, 'prev_motion_state') and hasattr(self, 'motion_state'):
            if self.motion_state != self.prev_motion_state:
                # If transitioning from stationary to movement
                if self.prev_motion_state in ["stationary", "long_stationary"]:
                    motion_scale *= 2.0  # Double process noise during transition
                    self.get_logger().info(
                        f"Doubling process noise during {self.prev_motion_state}->{self.motion_state} transition: scale={motion_scale:.1f}"
                    )
        
        # Factor in sensor gaps
        if hasattr(self, 'sensor_gap_detection'):
            current_time = time.time()
            avg_gap_level = 0.0
            gap_count = 0
            
            for sensor, gap_info in self.sensor_gap_detection.items():
                if gap_info.get('gap_detected', False):
                    avg_gap_level += gap_info.get('gap_level', 0.0)
                    gap_count += 1
                    
            if gap_count > 0:
                avg_gap_level /= gap_count
                gap_factor *= (1.0 + avg_gap_level * 1.5)  # Increased from 1.0 to 1.5
        
        # Apply combined scaling to the process noise
        combined_scale = gap_factor * motion_scale
        
        # Process noise parameters with adaptive scaling - INCREASED BASE VALUES
        q_pos = self.process_noise_pos * dt * combined_scale * 1.5  # Increased from base
        q_vel = self.process_noise_vel * dt * combined_scale * 1.5  # Increased from base
        
        # MODIFIED: Apply physics-based rolling friction for ground movement with decreased friction
        # (Basketball always rolls on ground, never bounces)
        friction_coef = 0.02  # Reduced from 0.03 - Lower rolling friction coefficient for better responsiveness
        
        # Apply deceleration to horizontal velocity components
        current_velocity = np.linalg.norm(self.state[2:4])  # x-y plane velocity
        if current_velocity > 0:
            # Calculate friction deceleration: a = μg
            deceleration = friction_coef * 9.81  # μg in m/s²
            
            # Don't decelerate more than the current velocity
            max_dv = current_velocity
            dv = min(max_dv, deceleration * dt)
            
            # Only apply friction if not in transition from stationary to moving
            apply_friction = True
            if hasattr(self, 'prev_motion_state') and hasattr(self, 'motion_state'):
                if (self.prev_motion_state in ["stationary", "long_stationary"] and 
                    self.motion_state in ["small_movement", "medium_fast"]):
                    # Skip friction during transition to movement
                    apply_friction = False
                    self.get_logger().debug("Skipping friction application during transition to movement")
            
            # Apply proportional deceleration to velocity components if needed
            if apply_friction and dv > 0 and current_velocity > 0:
                factor = 1.0 - (dv / current_velocity)
                self.state[2] *= factor  # Reduce x velocity
                self.state[3] *= factor  # Reduce y velocity
        
        # Fill in the 4x4 process noise matrix with INCREASED values
        # Position variances
        self._Q_matrix[0, 0] = q_pos * dt**3 / 3.0  # x position variance
        self._Q_matrix[1, 1] = q_pos * dt**3 / 3.0  # y position variance
        
        # Velocity variances
        self._Q_matrix[2, 2] = q_vel * dt          # x velocity variance
        self._Q_matrix[3, 3] = q_vel * dt          # y velocity variance
        
        # Position-velocity covariances
        self._Q_matrix[0, 2] = self._Q_matrix[2, 0] = q_pos * dt**2 / 2.0  # x position-velocity
        self._Q_matrix[1, 3] = self._Q_matrix[3, 1] = q_pos * dt**2 / 2.0  # y position-velocity
        
        # ADDED: Check for consecutive rejected measurements and increase process noise
        if hasattr(self, 'consecutive_rejections_per_sensor'):
            total_rejections = sum(self.consecutive_rejections_per_sensor.values())
            if total_rejections > 5:  # If significant rejections across sensors
                rejection_scale = min(3.0, 1.0 + (total_rejections / 10.0))
                self._Q_matrix *= rejection_scale
                self.get_logger().info(
                    f"Increasing process noise due to {total_rejections} total rejections: scale={rejection_scale:.1f}"
                )
        
        # ADDED: Check if velocity is consistently near zero despite position changes
        if (hasattr(self, 'position_history') and len(self.position_history) >= 3 and
            current_velocity < 0.05 and self.motion_state in ["small_movement", "medium_fast"]):
            # Calculate position changes
            recent_pos = list(self.position_history)[-3:]
            if len(recent_pos) >= 3:
                # Calculate distances between consecutive positions
                dist1 = math.sqrt((recent_pos[-2][0] - recent_pos[-3][0])**2 + 
                                (recent_pos[-2][1] - recent_pos[-3][1])**2)
                dist2 = math.sqrt((recent_pos[-1][0] - recent_pos[-2][0])**2 + 
                                (recent_pos[-1][1] - recent_pos[-2][1])**2)
                
                # If positions are changing but velocity is near zero, boost velocity variance
                if (dist1 > 0.05 or dist2 > 0.05) and current_velocity < 0.05:
                    self._Q_matrix[2, 2] *= 3.0  # Triple x velocity variance
                    self._Q_matrix[3, 3] *= 3.0  # Triple y velocity variance
                    self.get_logger().info(
                        f"Boosting velocity variance: position changes detected ({dist1:.2f}m, {dist2:.2f}m) "
                        f"but current velocity is only {current_velocity:.3f}m/s"
                    )
        
        # Predict state using state transition matrix
        self.state = np.dot(self._F_matrix, self.state)
        
        # Predict covariance
        self.covariance = np.dot(np.dot(self._F_matrix, self.covariance), self._F_matrix.T) + self._Q_matrix
        
        # Ensure covariance remains symmetric
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def check_filter_divergence(self):
        """
        Check for filter divergence and reset if needed.
        This helps recover from situations where validation thresholds 
        have allowed bad measurements to corrupt the filter state.
        """
        if not self.initialized:
            return
            
        # Check for persistent measurement rejections
        excessive_rejections = False
        if hasattr(self, 'consecutive_rejections_per_sensor'):
            # Count sensors with excessive rejections
            excessive_count = 0
            for sensor, count in self.consecutive_rejections_per_sensor.items():
                if count > 5:  # 5+ consecutive rejections indicates a problem
                    excessive_count += 1
            
            # If multiple sensors are being consistently rejected, we may have diverged
            if excessive_count >= 2:
                excessive_rejections = True
                self.get_logger().warn(
                    f"Detected filter divergence: {excessive_count} sensors with excessive rejections"
                )
        
        # Check for excessively high uncertainty
        high_uncertainty = self.position_uncertainty > 1.5  # Very high position uncertainty
        
        # If we have both excessive rejections and high uncertainty, reset the filter
        if excessive_rejections and high_uncertainty:
            self.get_logger().error("Filter divergence detected - reinitializing filter")
            
            # Save the current state for comparison
            old_state = self.state.copy()
            
            # Get latest available data for reinitialization
            lidar_msg = self.sensor_buffer.get_latest_measurement('lidar')
            yolo_2d_msg = self.sensor_buffer.get_latest_measurement('yolo_2d')
            
            # Attempt to reinitialize with sensor data
            if lidar_msg or (yolo_2d_msg and 'yolo_2d' in self.bbox_data):
                # Reset filter to default state
                self.initialize_filter_with_defaults()
                
                # Log the change in position
                new_distance = math.sqrt(self.state[0]**2 + self.state[1]**2)
                old_distance = math.sqrt(old_state[0]**2 + old_state[1]**2)
                self.get_logger().info(
                    f"Filter reinitialized: old=({old_state[0]:.2f}, {old_state[1]:.2f}), "
                    f"new=({self.state[0]:.2f}, {self.state[1]:.2f}), "
                    f"distance_change={new_distance-old_distance:.2f}m"
                )
                
                # Reset all consecutive rejection counters
                for sensor in self.consecutive_rejections_per_sensor:
                    self.consecutive_rejections_per_sensor[sensor] = 0
                
                return True
        
        return False

    def initialize_filter_with_defaults(self):
        """Initialize filter with default values for reinitialization after divergence."""
        # Reset state to zeros
        self.state = np.zeros(4, dtype=np.float32)
        
        # Reset covariance with large initial uncertainty
        self.covariance = np.eye(4, dtype=np.float32)
        self.covariance[0:2, 0:2] *= 1.0  # More conservative position uncertainty for restarts
        self.covariance[2:4, 2:4] *= 2.0  # More conservative velocity uncertainty for restarts
        
        # Update uncertainty metrics
        self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:2, 0:2]) / 2.0)
        self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[2:4, 2:4]) / 2.0)
        
        # Record reset
        self.last_update_time = time.time()
        
        # Reset smoothed state estimator
        if hasattr(self, 'smoothed_state_estimator'):
            self.smoothed_state_estimator.reset()
        
        # Log the reset
        self.get_logger().info(
            f"Filter initialized with defaults: uncertainty={self.position_uncertainty:.3f}m"
        )
    
    def update_state(self, measurements):
        """
        Modified update_state method with improved handling of 2D-derived 3D estimates.
        """
        # Store successful update flag
        successful_update = False
        
        # Check if we're in the refinement phase after fast initialization
        in_refinement = getattr(self, 'in_refinement_phase', False)
        
        # Get current motion state
        motion_state = self.detect_motion_state()
        
        # For each measurement in the synchronized set
        for sensor, msg in measurements.items():
            # Transform measurement to reference frame
            is_2d_sensor = sensor.endswith('_2d')
            is_2d_derived = '_est3d' in sensor
            
            # For 2D sensors, estimate 3D position
            if is_2d_sensor:
                if sensor in self.bbox_data:
                    transformed = self.estimate_3d_from_2d(msg, self.bbox_data[sensor])
                    if transformed is None:
                        continue
                else:
                    continue
            else:
                # For 3D sensors, transform point
                transformed = self.transform_point(msg, self.reference_frame, False)
                if transformed is None:
                    continue
            
            # Check hard position limits
            max_coord = 5.0
            if abs(transformed.point.x) > max_coord or abs(transformed.point.y) > max_coord:
                self.get_logger().warn(
                    f"Rejecting {sensor} measurement outside hard limits: "
                    f"pos=({transformed.point.x:.2f}, {transformed.point.y:.2f}), limits=±{max_coord}m"
                )
                self.consecutive_rejections_per_sensor[sensor] = self.consecutive_rejections_per_sensor.get(sensor, 0) + 1
                continue
            
            # Setup measurement and matrices
            if sensor.endswith('_2d'):
                z = np.array([transformed.point.x, transformed.point.y], dtype=np.float32)
                H = np.zeros((2, 4), dtype=np.float32)
                H[0, 0] = 1.0  # Extract x position
                H[1, 1] = 1.0  # Extract y position
                
                # Get noise matrix
                if sensor == 'hsv_2d':
                    R = np.diag([self.measurement_noise_hsv_2d, self.measurement_noise_hsv_2d]).astype(np.float32)
                elif sensor == 'yolo_2d':
                    R = np.diag([self.measurement_noise_yolo_2d, self.measurement_noise_yolo_2d]).astype(np.float32)
                else:
                    R = np.diag([50.0, 50.0]).astype(np.float32)
            elif is_2d_derived:
                # Handle 2D-derived 3D estimates
                z = np.array([transformed.point.x, transformed.point.y], dtype=np.float32)
                H = np.zeros((2, 4), dtype=np.float32)
                H[0, 0] = 1.0
                H[1, 1] = 1.0
                
                # Get noise matrix with higher uncertainty
                if 'yolo' in sensor:
                    R = np.diag([self.measurement_noise_yolo_2d_est3d, self.measurement_noise_yolo_2d_est3d]).astype(np.float32)
                elif 'hsv' in sensor:
                    R = np.diag([self.measurement_noise_hsv_2d_est3d, self.measurement_noise_hsv_2d_est3d]).astype(np.float32)
                else:
                    R = np.diag([0.15, 0.15]).astype(np.float32)
            else:
                z = np.array([transformed.point.x, transformed.point.y], dtype=np.float32)
                H = np.zeros((2, 4), dtype=np.float32)
                H[0, 0] = 1.0
                H[1, 1] = 1.0
                
                # Get noise matrix
                if sensor == 'lidar':
                    R = np.diag([self.measurement_noise_lidar, self.measurement_noise_lidar]).astype(np.float32)
                elif sensor == 'hsv_3d':
                    R = np.diag([self.measurement_noise_hsv_3d, self.measurement_noise_hsv_3d]).astype(np.float32)
                elif sensor == 'yolo_3d':
                    R = np.diag([self.measurement_noise_yolo_3d, self.measurement_noise_yolo_3d]).astype(np.float32)
                else:
                    R = np.diag([0.1, 0.1]).astype(np.float32)
            
            # Innovation (measurement residual)
            y = z - np.dot(H, self.state)
            
            # Innovation covariance
            S = np.dot(np.dot(H, self.covariance), H.T) + R
            
            # Get validation threshold - MODIFIED to use specialized threshold for 2D-derived estimates
            if is_2d_derived:
                # Get bbox age for this sensor
                bbox_age = 0.0
                base_sensor = sensor.split('_')[0] + '_2d'
                if base_sensor in self.bbox_data:
                    bbox_age = time.time() - self.bbox_data[base_sensor].get('timestamp', 0)
                
                # Use specialized threshold calculation with bbox age information
                threshold = self.get_validation_threshold_for_2d_derived(sensor, motion_state, bbox_age)
                
                if self.debug_level >= 2:
                    self.get_logger().debug(
                        f"Using specialized 2D-derived threshold for {sensor}: {threshold:.2f} (bbox_age={bbox_age:.1f}s)"
                    )
            elif hasattr(self, 'validation_manager'):
                threshold = self.validation_manager.get_validation_threshold(sensor)
            else:
                threshold = self.get_innovation_threshold(sensor, motion_state)
            
            # Cap threshold
            max_threshold = 40.0  # Increased from 25.0 for more permissiveness
            original_threshold = threshold
            threshold = min(threshold, max_threshold)
            
            # IMPORTANT: Use much more permissive validation during refinement phase
            if in_refinement:
                # Double the threshold during refinement phase
                threshold *= 3.0
                
                # Log this adjustment
                if self.debug_level >= 2:
                    self.get_logger().debug(
                        f"Using permissive validation during refinement: {original_threshold:.2f} -> {threshold:.2f}"
                    )
            
            # Mahalanobis distance calculation
            try:
                S_inv = np.linalg.inv(S)
                mahalanobis_dist = np.sqrt(np.dot(np.dot(y.T, S_inv), y))
                
                # Check if in initialization phase for permissive validation
                initialization_phase = hasattr(self, 'in_initialization_phase') and self.in_initialization_phase
                if initialization_phase:
                    # Use more permissive threshold during initialization
                    threshold *= 3.0
                    max_init_threshold = 10.0
                    if threshold > max_init_threshold:
                        threshold = max_init_threshold
                
                # Skip measurement if it fails validation
                if mahalanobis_dist > threshold:
                    # Log rejection
                    if mahalanobis_dist > (threshold * 2):
                        self.get_logger().warn(
                            f"Rejecting {sensor} measurement: innovation {mahalanobis_dist:.2f} > threshold {threshold:.2f}"
                        )
                    else:
                        self.get_logger().debug(
                            f"Rejecting {sensor} measurement: innovation {mahalanobis_dist:.2f} > threshold {threshold:.2f}"
                        )
                    
                    # Record validation decision
                    if hasattr(self, 'validation_manager'):
                        self.validation_manager.record_validation_result(sensor, mahalanobis_dist, False, True)
                    
                    # Increment rejection counter
                    self.consecutive_rejections_per_sensor[sensor] = self.consecutive_rejections_per_sensor.get(sensor, 0) + 1
                    self.consecutive_updates = 0
                    continue
                
                # Reset rejection counter
                self.consecutive_rejections_per_sensor[sensor] = 0
                
                # Record validation acceptance
                if hasattr(self, 'validation_manager'):
                    self.validation_manager.record_validation_result(sensor, mahalanobis_dist, True, True)
                
                # Update consecutive updates counter
                self.consecutive_updates += 1
                
            except np.linalg.LinAlgError:
                self.get_logger().warn(f"Matrix inversion failed during validation for {sensor}")
                continue
            
            # Kalman gain
            try:
                K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))
                
                # MODIFIED: Special handling for 2D-derived estimates during movement
                if is_2d_derived and motion_state in ["small_movement", "medium_fast"]:
                    # During movement, trust 2D-derived estimates more by increasing their influence
                    # Calculate a higher blend factor for movement detection
                    movement_blend = 0.4  # 40% direct influence for 2D-derived estimates during movement
                    
                    # Apply blended update for 2D-derived during movement
                    direct_influence = np.zeros_like(self.state)
                    direct_influence[0:2] = y[0:2] * movement_blend
                    
                    # Log this special handling
                    if self.debug_level >= 2:
                        self.get_logger().debug(
                            f"Applying movement-sensitive blend for {sensor}: "
                            f"direct_influence={movement_blend:.2f}, motion={motion_state}"
                        )
                    
                    # Apply combined update (standard Kalman + direct influence)
                    self.state = self.state + np.dot(K, y) + direct_influence
                elif in_refinement:
                    # Special refinement phase handling (existing logic)
                    # ...existing refinement phase code...
                    current_time = time.time()
                    refinement_duration = current_time - getattr(self, 'refinement_start_time', current_time)
                    self.refinement_measurements = getattr(self, 'refinement_measurements', 0) + 1
                    
                    # Calculate blend factor for refinement
                    if refinement_duration > 2.0 or self.refinement_measurements >= 5:
                        # Exit refinement phase
                        self.in_refinement_phase = False
                        self.get_logger().info(
                            f"Exiting refinement phase after {self.refinement_measurements} measurements and "
                            f"{refinement_duration:.1f}s"
                        )
                        # Normal update
                        self.state = self.state + np.dot(K, y)
                    else:
                        # Still in refinement - use blended update
                        blend_factor = max(0.3, 0.7 - (0.1 * self.refinement_measurements))
                        
                        # Blend between direct measurement and Kalman update for position
                        direct_influence = np.zeros_like(self.state)
                        direct_influence[0:2] = y[0:2] * blend_factor
                        
                        # Apply blended update
                        self.state = self.state + np.dot(K, y) + direct_influence
                else:
                    # Normal update outside of special cases
                    self.state = self.state + np.dot(K, y)
                
                # Update covariance using Joseph form for numerical stability
                I = np.eye(self.state.shape[0], dtype=np.float32)
                self.covariance = np.dot(np.dot(I - np.dot(K, H), self.covariance), 
                                        (I - np.dot(K, H)).T) + np.dot(np.dot(K, R), K.T)
                
                # Ensure covariance remains symmetric
                self.covariance = 0.5 * (self.covariance + self.covariance.T)
                
                # Mark that we had a successful update
                successful_update = True
                
                # For logging
                if self.debug_level >= 2:
                    self.get_logger().debug(
                        f"Successfully incorporated {sensor} measurement: "
                        f"innovation={mahalanobis_dist:.2f}, position=({transformed.point.x:.2f}, {transformed.point.y:.2f})"
                    )
                
            except np.linalg.LinAlgError:
                self.get_logger().warn(f"Matrix inversion failed during Kalman update for {sensor}")
                continue
        
        return successful_update
    
    def get_innovation_threshold(self, source, motion_state):
        """
        Get adaptive innovation threshold based on sensor type and motion state.
        Modified to be much more accepting of measurements showing movement,
        especially during transitions from stationary states.
        
        Args:
            source (str): Sensor source identifier
            motion_state (str): Current motion state
                
        Returns:
            float: Innovation threshold for measurement validation
        """
        # Determine sensor type
        if source == 'lidar':
            sensor_type = "lidar"
        elif source.endswith('_3d'):
            sensor_type = "3d_vision"
        else:
            sensor_type = "2d"
        
        # GREATLY INCREASED base thresholds for each sensor type and motion state
        base_thresholds = {
            "lidar": {
                "stationary": (10.0, 5.0),       # Increased from (7.5, 3.5)
                "long_stationary": (10.0, 5.0),  # Increased from (7.5, 3.5)
                "small_movement": (12.0, 6.0),   # Increased from (9.0, 4.5)
                "medium_fast": (15.0, 7.5),      # Increased from (12.0, 6.0)
                "unknown": (16.0, 8.0)           # Increased from (13.0, 6.5)
            },
            "3d_vision": {
                "stationary": (14.0, 7.0),       # Increased from (11.0, 5.0)
                "long_stationary": (14.0, 7.0),  # Increased from (11.0, 5.0)
                "small_movement": (16.0, 8.0),   # Increased from (13.0, 6.0)
                "medium_fast": (20.0, 10.0),     # Increased from (16.0, 8.0)
                "unknown": (22.0, 11.0)          # Increased from (18.0, 9.0)
            },
            "2d": {
                "stationary": (18.0, 9.0),       # Increased from (15.0, 7.5)
                "long_stationary": (18.0, 9.0),  # Increased from (15.0, 7.5)
                "small_movement": (22.0, 11.0),  # Increased from (18.0, 9.0)
                "medium_fast": (28.0, 18.0),     # Increased from (22.0, 15.0)
                "unknown": (30.0, 20.0)          # Increased from (25.0, 16.0)
            }
        }
        
        # Get thresholds for this state and sensor type
        # If motion_state is not in the dictionary, fall back to "stationary" for "long_stationary" 
        # or "unknown" for any other missing state
        if motion_state not in base_thresholds[sensor_type]:
            if motion_state == "long_stationary":
                motion_state = "stationary"
            else:
                motion_state = "unknown"
        
        initial, minimum = base_thresholds[sensor_type][motion_state]
        
        # SLOWER DECAY for thresholds to maintain higher values longer
        # Modify decay calculation to decrease much more slowly with consecutive updates
        decay_factor = max(0.7, min(1.0, 20.0 / (self.consecutive_updates + 1)))  # Increased from 15.0/0.5
        threshold = minimum + (initial - minimum) * decay_factor
        
        # Check for recent transitions and recent sensor updates
        just_transitioned = False
        if hasattr(self, 'prev_motion_state') and hasattr(self, 'motion_state'):
            # Check if we've had a motion state change in the last few updates
            if hasattr(self, 'state_transition_time'):
                time_since_transition = time.time() - self.state_transition_time
                if time_since_transition < 3.0:  # Extended from 2.0 to 3.0 seconds after transition
                    just_transitioned = True
            
            # If no transition time is stored but states are different, assume recent transition
            elif self.prev_motion_state != self.motion_state:
                just_transitioned = True
                # Store the transition time for future reference
                self.state_transition_time = time.time()
        
        # GREATLY BOOST threshold during state transitions
        if just_transitioned:
            # Apply much larger boost, especially for transitions from stationary states
            if hasattr(self, 'prev_motion_state') and self.prev_motion_state in ["stationary", "long_stationary"]:
                threshold *= 4.0  # Greatly increased from 2.5
                if self.debug_level >= 1:
                    self.get_logger().info(
                        f"Applying massive transition boost: threshold={threshold:.2f} "
                        f"for {source} during {self.prev_motion_state}->{motion_state} transition"
                    )
            else:
                # Still boost more for other transitions
                threshold *= 3.0  # Increased from 2.0
        
        # ADDED: Extra boost during movement motion states to accept more sensor data
        if motion_state in ["small_movement", "medium_fast"]:
            threshold *= 1.6  # Increased from 1.3 - 60% increase during movement states
        
        # Apply additional adjustments based on sensor type
        if motion_state == "medium_fast" and source == "lidar":
            threshold *= 1.5  # Increased from 1.3 - Extra permissiveness for primary sensor during fast motion
        
        # Detect if significant position changes are being rejected
        if hasattr(self, 'consecutive_rejections_per_sensor'):
            rejections = self.consecutive_rejections_per_sensor.get(source, 0)
            if rejections >= 1:  # Reduced from 2 - respond faster to rejections
                # LARGER boost factor when rejecting consecutive measurements
                boost_factor = 1.0 + (min(rejections, 10) * 0.25)  # Increased from 0.15
                threshold *= boost_factor
                
                if self.debug_level >= 1 and boost_factor > 1.2:
                    self.get_logger().info(
                        f"Applying rejection-break threshold boost for {source}: "
                        f"{boost_factor:.2f}x → {threshold:.2f}"
                    )
        
        # Special handling for rolling balls with lidar
        if source == "lidar" and hasattr(self, 'flat_ground_detected') and self.flat_ground_detected:
            # For a rolling ball, lidar measurements at the bottom of the ball might be inconsistent
            threshold *= 2.0  # Increased from 1.5 - 100% increase for rolling ball with lidar
                
        # ADDED: Check for sensor recovery after gaps
        current_time = time.time()
        if hasattr(self, 'sensor_gap_detection') and source in self.sensor_gap_detection:
            gap_info = self.sensor_gap_detection[source]
            if gap_info.get('gap_detected', False):
                # Recently detected gap - be more permissive to allow recovery
                threshold *= 2.5  # Increased from 1.6 - 150% increase after gaps
                self.get_logger().info(f"Post-gap validation boost for {source}: threshold={threshold:.2f}")
            else:
                # Check if we recently recovered from a gap
                last_time = self.last_detection_time.get(source, 0)
                if current_time - last_time < 1.0:  # Fresh data
                    recovery_time = current_time - gap_info.get('last_recovery_time', 0)
                    if recovery_time < 5.0:  # Extended from 3.0 to 5.0 seconds after recovery
                        # Apply post-recovery boost
                        threshold *= 2.0  # Increased from 1.4 - 100% increase after recovery
        
        # HARD LIMIT: Cap maximum thresholds at a safety value to prevent extreme values
        max_safe_threshold = 50.0  # Increased from 35.0
        threshold = min(threshold, max_safe_threshold)
        
        return threshold

    def transform_point(self, point_msg, target_frame, is_2d=False):
        """
        Transform a point message to the target reference frame.
        
        Args:
            point_msg (PointStamped): The point message to transform
            target_frame (str): The target reference frame
            is_2d (bool): Whether the point is a 2D point (z component ignored)
            
        Returns:
            PointStamped: The transformed point or None if transformation failed
        """
        if not self.transform_available:
            # Check again if transforms are available - we might have just missed initialization
            if self.check_transform_availability():
                self.get_logger().info(f"Transform became available - will attempt transformation")
            else:
                self.get_logger().warn(f"Transform not available - cannot transform point from {point_msg.header.frame_id} to {target_frame}")
                return None
        
        try:
            # Return original message if already in target frame
            if point_msg.header.frame_id == target_frame:
                return point_msg
            
            # Use cached transforms for static relationships
            transform = None
            if point_msg.header.frame_id == 'ascamera_color_0' and target_frame == self.reference_frame and self.tf_camera_to_base is not None:
                # Use cached camera to base transform
                transform = self.tf_camera_to_base
                if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 50 == 0:
                    self.get_logger().debug("Using cached camera to base transform")
            elif point_msg.header.frame_id == 'lidar_frame' and target_frame == self.reference_frame and self.tf_lidar_to_base is not None:
                # Use cached lidar to base transform
                transform = self.tf_lidar_to_base
                if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 50 == 0:
                    self.get_logger().debug("Using cached lidar to base transform")
            else:
                # Fall back to standard transform lookup for non-cached relationships
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    point_msg.header.frame_id,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.2)
                )
            
            # For 2D points, set z=0 before transform then restore confidence value after transform
            if is_2d:
                # Make a copy of the point to avoid modifying the original
                temp_point = copy.deepcopy(point_msg)
                confidence = point_msg.point.z  # Save confidence value
                temp_point.point.z = 0.0  # Set z to 0 for proper transformation
                
                # Transform the point
                transformed = do_transform_point(temp_point, transform)
                
                # Restore confidence value
                transformed.point.z = confidence
            else:
                # Normal 3D point transformation
                transformed = do_transform_point(point_msg, transform)
            
            # Increment success counter
            self.transform_successes += 1
            return transformed
            
        except Exception as e:
            # Increment failure counter
            self.transform_failures += 1
            self.get_logger().warn(f"Transform error: {str(e)}")
            return None

    def adjust_covariance_for_gaps(self):
        """
        Adjust the filter covariance based on sensor gap patterns,
        with improved handling for stationary objects and movement detection.
        """
        # Need sensors to be initialized first
        if not hasattr(self, 'sensor_gap_detection') or not hasattr(self, 'consecutive_rejections_per_sensor'):
            return

        current_time = time.time()

        # Count active sensors and their gap levels
        active_3d_sensors = 0
        active_2d_sensors = 0
        total_gap_level = 0.0
        sensor_count = 0
        rejected_sensor_count = 0 # Track sensors rejected consecutively
        rejection_threshold = 2 # Consider sensor 'gapped' after this many consecutive rejections (reduced from 3)

        for sensor, gap_info in self.sensor_gap_detection.items():
            last_time = self.last_detection_time.get(sensor, 0)
            gap_duration = current_time - last_time

            # Skip sensors we've never seen
            if self.sensor_counts.get(sensor, 0) == 0:
                continue

            sensor_count += 1

            # Check for consecutive rejections
            rejections = self.consecutive_rejections_per_sensor.get(sensor, 0)
            is_rejected = rejections >= rejection_threshold
            is_gap = gap_duration > 0.3 # Reduced from 0.5 - Standard gap definition

            gap_level = 0.0
            contributes_to_gap = False

            if is_gap:
                # Calculate gap level based on duration
                if gap_duration < 0.1:
                    gap_level = 0.0
                elif gap_duration < 1.0: # Reduced from 1.5 - Shorter time range for gap level calculation
                    gap_level = gap_duration / 1.0
                else:
                    gap_level = 1.0
                contributes_to_gap = True
            elif is_rejected:
                # Assign a higher gap level for rejected sensors even if data is recent
                gap_level = 0.6 + (min(rejections, 10) * 0.05) # Increased from 0.5 - Higher gap level with more rejections, capped
                gap_level = min(gap_level, 1.0) # Cap at 1.0
                rejected_sensor_count += 1
                contributes_to_gap = True

            if contributes_to_gap:
                total_gap_level += gap_level
            else:
                # Count active sensors only if data is recent AND not consistently rejected
                if not sensor.endswith('_2d'):
                    active_3d_sensors += 1
                else:
                    active_2d_sensors += 1

        # Calculate average gap level (considering actual gaps and rejections)
        avg_gap_level = total_gap_level / max(1, sensor_count)

        # Get current motion state
        motion_state = self.detect_motion_state()

        # GREATLY INCREASED base growth rates for better responsiveness
        if active_3d_sensors >= 2:
            growth_rate = 1.10 + (avg_gap_level * 0.25)  # Increased from 1.05 + 0.15
        elif active_3d_sensors == 1:
            growth_rate = 1.15 + (avg_gap_level * 0.35)  # Increased from 1.08 + 0.20
        elif active_2d_sensors > 0:
            growth_rate = 1.20 + (avg_gap_level * 0.40)  # Increased from 1.12 + 0.25
        else: # No effectively active sensors (all gapped or rejected)
            growth_rate = 1.30 + (avg_gap_level * 0.70)  # Increased from 1.15 + 0.50

        # Further increase growth rate if many sensors are being rejected
        if rejected_sensor_count >= 2:
            growth_rate *= (1.0 + 0.15 * rejected_sensor_count)  # Increased from 0.08
            if self.debug_level >= 2:
                self.get_logger().debug(f"Boosting growth rate due to {rejected_sensor_count} rejected sensors: new rate={growth_rate:.3f}")

        # MAJOR CHANGE: Allow much more covariance growth for stationary objects
        if motion_state in ["stationary", "long_stationary"]:
            # DRASTICALLY reduced suppression factors to allow more uncertainty growth
            if motion_state == "stationary":
                growth_rate = 1.0 + ((growth_rate - 1.0) * 0.8)  # Changed from 0.6 to 0.8
            else:  # long_stationary
                growth_rate = 1.0 + ((growth_rate - 1.0) * 0.7)  # Changed from 0.4 to 0.7
                
            # INCREASED minimum growth rate for stationary objects
            min_stationary_growth = 1.05  # Increased from 1.03 - At least 5% growth per update
            growth_rate = max(growth_rate, min_stationary_growth)
            
            if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 20 == 0:
                self.get_logger().debug(f"Modified covariance growth for {motion_state} object: {growth_rate:.3f}")

        # Apply growth limit based on uncertainties - with HIGHER CAPS
        pos_uncertainty = math.sqrt(max(0.0, np.trace(self.covariance[0:2, 0:2]) / 2.0))
        if pos_uncertainty > 0.7:  # Increased from 0.5
            # MODIFIED: Allow even higher maximum growth rate during gaps
            max_growth_rate = 1.4  # Increased from 1.2
            growth_rate = min(growth_rate, max_growth_rate)
            
        # Apply growth factor
        if growth_rate > 1.0:
            # MODIFIED: Apply growth more aggressively
            self.covariance[0:2, 0:2] *= growth_rate
            self.covariance[2:4, 2:4] *= (growth_rate * 1.2)  # Increased from 1.1
            
            # Ensure symmetry and minimum values
            self.covariance = 0.5 * (self.covariance + self.covariance.T)
            for i in range(4):
                self.covariance[i, i] = max(0.01, self.covariance[i, i])
                
            # Add motion-specific uncertainty caps with MUCH HIGHER CAPS
            uncertainty_caps = self.get_motion_based_uncertainty_caps(motion_state)
            max_pos_uncertainty = uncertainty_caps["position_uncertainty_cap"]
            max_vel_uncertainty = uncertainty_caps["velocity_uncertainty_cap"]
            
            current_pos_uncertainty = math.sqrt(max(0.0, np.trace(self.covariance[0:2, 0:2]) / 2.0))
            current_vel_uncertainty = math.sqrt(max(0.0, np.trace(self.covariance[2:4, 2:4]) / 2.0))
            
            if current_pos_uncertainty > max_pos_uncertainty:
                scale = (max_pos_uncertainty / current_pos_uncertainty) ** 2
                self.covariance[0:2, 0:2] *= scale
            if current_vel_uncertainty > max_vel_uncertainty:
                scale = (max_vel_uncertainty / current_vel_uncertainty) ** 2
                self.covariance[2:4, 2:4] *= scale
            self.covariance = 0.5 * (self.covariance + self.covariance.T)

            # Debug log for significant growth
            if growth_rate > 1.1 and self.debug_level >= 1: # Log even at level 1 if growth is significant
                self.get_logger().debug(
                    f"Applying covariance growth: rate={growth_rate:.3f}, avg_gap_level={avg_gap_level:.2f}, "
                    f"rejected_sensors={rejected_sensor_count}, "
                    f"pos_uncertainty={current_pos_uncertainty:.3f}m, caps=({max_pos_uncertainty:.1f}, {max_vel_uncertainty:.1f})"
                )

    def get_motion_based_uncertainty_caps(self, motion_state):
        """
        Returns maximum uncertainty caps based on the current motion state.
        Different motion states allow for different maximum uncertainty values.
        
        Args:
            motion_state (str): Current motion state of the basketball
            
        Returns:
            dict: Dictionary with position and velocity uncertainty caps
        """
        if motion_state == "stationary":
            return {
                "position_uncertainty_cap": 1.2,  # Increased from 0.8
                "velocity_uncertainty_cap": 2.0   # Increased from 1.5
            }
        elif motion_state == "long_stationary":
            return {
                "position_uncertainty_cap": 1.0,  # Increased from 0.7
                "velocity_uncertainty_cap": 1.8   # Increased from 1.2
            }
        elif motion_state == "small_movement":
            return {
                "position_uncertainty_cap": 1.6,  # Increased from 1.2
                "velocity_uncertainty_cap": 2.5   # Increased from 2.0
            }
        elif motion_state == "medium_fast":
            return {
                "position_uncertainty_cap": 2.0,  # Increased from 1.5
                "velocity_uncertainty_cap": 3.5   # Increased from 2.5
            }
        else:  # unknown or other states
            return {
                "position_uncertainty_cap": 2.0,  # Increased from 1.5
                "velocity_uncertainty_cap": 3.0   # Increased from 2.2
            }

    def update_tracking_status(self):
        """
        Updated method to determine tracking status with improved gap handling
        and faster response to movement after gaps.
        """
        current_time = time.time()
        
        # Get current uncertainty metrics
        pos_uncertainty = self.position_uncertainty
        vel_uncertainty = self.velocity_uncertainty
        
        # Count active sensors
        active_3d_sensors = 0
        active_2d_sensors = 0
        
        for sensor in self.expected_sensors:
            last_time = self.last_detection_time.get(sensor, 0)
            if current_time - last_time < 1.5:  # Extended from 1.0 to 1.5 seconds to be more lenient
                if sensor.endswith('_2d'):
                    active_2d_sensors += 1
                else:
                    active_3d_sensors += 1
        
        # Track sensor gap conditions
        all_sensors_gap = (active_3d_sensors == 0 and active_2d_sensors == 0)
        
        # Initialize sensor gap tolerance window tracking if not already present
        if not hasattr(self, 'sensor_gap_window'):
            self.sensor_gap_window = {
                'active': False,
                'start_time': 0.0,
                'previous_reliability': False,
                'tolerance_seconds': 0.8,  # Reduced from 2.0
                'base_tolerance': 1.0,     # Store base tolerance value
                'adaptive_enabled': True,   # Enable adaptive adjustment
                'recent_sensor_data': {},   # NEW: Track sensor data before gap
                'position_before_gap': None,  # NEW: Track position before gap
                'motion_state_before_gap': None  # NEW: Track motion state before gap
            }
        else:
            # Fix: Ensure all required keys exist
            required_keys = ['active', 'start_time', 'previous_reliability', 'tolerance_seconds', 
                            'base_tolerance', 'adaptive_enabled', 'recent_sensor_data',
                            'position_before_gap', 'motion_state_before_gap']
            for key in required_keys:
                if key not in self.sensor_gap_window:
                    if key == 'recent_sensor_data':
                        self.sensor_gap_window[key] = {}
                    elif key in ['active', 'adaptive_enabled', 'previous_reliability']:
                        self.sensor_gap_window[key] = False
                    elif key in ['position_before_gap', 'motion_state_before_gap']:
                        self.sensor_gap_window[key] = None
                    else:
                        self.sensor_gap_window[key] = 0.0
                
        # MODIFIED: Detect start of a new sensor gap
        if all_sensors_gap and not self.sensor_gap_window['active']:
            # Only activate gap mode if we were previously tracking
            if self.tracking_reliable:
                self.sensor_gap_window['active'] = True
                self.sensor_gap_window['start_time'] = current_time
                self.sensor_gap_window['previous_reliability'] = True
                
                # NEW: Store sensor data before gap
                for sensor in self.expected_sensors:
                    last_msg = self.sensor_buffer.get_latest_measurement(sensor)
                    if last_msg is not None:
                        # Store simplified version of the message
                        self.sensor_gap_window['recent_sensor_data'][sensor] = {
                            'x': last_msg.point.x,
                            'y': last_msg.point.y,
                            'z': last_msg.point.z,
                            'time': self.last_detection_time.get(sensor, 0)
                        }
                
                # NEW: Store position and motion state before gap
                self.sensor_gap_window['position_before_gap'] = [self.state[0], self.state[1]]
                self.sensor_gap_window['motion_state_before_gap'] = self.motion_state
                
                # MODIFIED: Motion-Aware Tolerance with Reduced Values
                # Start with the base tolerance from configuration but with lower values
                base_tolerance = self.sensor_gap_window.get('base_tolerance', 1.0)  # Reduced from 2.0
                
                # Apply motion-based multipliers with reduced values
                motion_state = self.detect_motion_state()
                if motion_state == "long_stationary":
                    # For long-stationary objects, allow longer gaps but reduced from before
                    tolerance = base_tolerance * 2.0  # Reduced from 3.0
                elif motion_state == "stationary":
                    # For regular stationary, use slightly longer gaps
                    tolerance = base_tolerance * 1.2  # Reduced from 1.5
                else:
                    # For moving objects, use shorter tolerance
                    tolerance = base_tolerance * 0.7  # Kept the same
                    
                # Store the adjusted tolerance
                self.sensor_gap_window['tolerance_seconds'] = tolerance
                
                if self.debug_level >= 1:
                    self.get_logger().info(
                        f"Sensor gap tolerance activated: window={self.sensor_gap_window['tolerance_seconds']:.1f}s, "
                        f"state={motion_state}, uncertainty={pos_uncertainty:.3f}m"
                    )
        
        # Check if we're within the tolerance window during a sensor gap
        within_gap_tolerance = False
        if self.sensor_gap_window['active']:
            # If any sensors have recovered, exit gap mode
            if not all_sensors_gap:
                self.sensor_gap_window['active'] = False
                if self.debug_level >= 2:
                    self.get_logger().debug("Sensor gap tolerance deactivated: sensors recovered")
            else:
                # Check if we're still within the tolerance window
                gap_duration = current_time - self.sensor_gap_window['start_time']
                within_gap_tolerance = gap_duration < self.sensor_gap_window['tolerance_seconds']
                
                # If gap exceeds tolerance window, exit gap mode
                if not within_gap_tolerance:
                    self.sensor_gap_window['active'] = False
                    if self.debug_level >= 1:
                        self.get_logger().info(
                            f"Sensor gap tolerance expired after {gap_duration:.2f}s > "
                            f"{self.sensor_gap_window['tolerance_seconds']:.1f}s threshold"
                        )
        
        # MODIFIED: Use different thresholds based on tracking state
        # If currently tracking, use much more lenient thresholds to maintain tracking
        if self.tracking_reliable:
            pos_threshold = self.position_uncertainty_threshold * 2.5  # Increased from 1.8
            vel_threshold = self.velocity_uncertainty_threshold * 2.5  # Increased from 1.8
        else:
            # Use more lenient thresholds to START tracking (easier to start than before)
            pos_threshold = self.position_uncertainty_threshold * 1.5  # Increased from 1.1
            vel_threshold = self.velocity_uncertainty_threshold * 1.5  # Increased from 1.1
        
        # Apply motion-aware threshold adjustments with increased values
        motion_state = self.detect_motion_state()
        if motion_state == "stationary":
            # MORE lenient with uncertainty thresholds for stationary objects
            pos_threshold *= 2.5  # Increased from 2.0
            vel_threshold *= 2.5  # Increased from 2.0
        elif motion_state == "long_stationary":
            # MORE lenient for long-term stationary objects
            pos_threshold *= 3.0  # Increased from 2.5
            vel_threshold *= 3.0  # Increased from 2.5
        elif motion_state == "medium_fast":
            # ADDED: Special handling for medium_fast to maintain tracking during movement
            pos_threshold *= 1.5  # New multiplier for movement
            vel_threshold *= 1.5  # New multiplier for movement
        
        # NEW: After sensor recovery with movement detection, be much more permissive
        if hasattr(self, 'sensor_gap_detection'):
            recent_recovery = False
            for sensor, gap_info in self.sensor_gap_detection.items():
                if gap_info.get('recovery_boost_active', False):
                    recent_recovery = True
                    break
                    
            if recent_recovery:
                # Triple the thresholds to ensure tracking during recovery
                pos_threshold *= 3.0
                vel_threshold *= 3.0
                if self.debug_level >= 1:
                    self.get_logger().info(
                        f"Using expanded thresholds during sensor recovery: pos={pos_threshold:.2f}, vel={vel_threshold:.2f}"
                    )
        
        # Modified reliability assessment with gap tolerance window
        # During sensor gaps within tolerance window for stationary objects, 
        # bypass the normal sensor count check
        if within_gap_tolerance and motion_state in ["stationary", "long_stationary"]:
            # During gap tolerance window, only check uncertainty thresholds, ignore sensor counts
            reliable = (pos_uncertainty < pos_threshold and vel_uncertainty < vel_threshold)
            
            # Log this special condition occasionally
            if self.debug_level >= 2:
                self.get_logger().debug(
                    f"Gap tolerance active: uncertainty={pos_uncertainty:.3f}m < threshold={pos_threshold:.3f}m"
                )
        else:
            # MODIFIED: Relaxed sensor requirements for tracking
            # Allow tracking with either 1 3D sensor OR 1 2D sensor (if enabled)
            reliable = (pos_uncertainty < pos_threshold and 
                        vel_uncertainty < vel_threshold and
                        (active_3d_sensors >= 1 or (active_2d_sensors >= 1 and self.allow_tracking_with_2d_only)))
        
        # Use time-based stability buffer with REDUCED size
        if len(self.reliability_buffer) == 0:
            # Initialize buffer if empty - REDUCED size from 5 to 3 for faster response
            self.reliability_buffer = deque([reliable] * 2, maxlen=3)  # Changed from 3 items maxlen=5
        else:
            # Add newest value
            self.reliability_buffer.append(reliable)
        
        # Analyze buffer for stability with MUCH more responsive thresholds
        true_count = sum(1 for r in self.reliability_buffer if r)
        
        # MODIFIED: Apply less aggressive hysteresis, especially for movement states
        if motion_state in ["stationary", "long_stationary"]:
            # For stationary objects: 
            # - Need just 1/3 reliable to start tracking (much easier to start tracking)
            # - Need all 3/3 unreliable to stop tracking (much harder to stop)
            if not self.tracking_reliable and true_count >= 1:  # Changed from 2/5 to 1/3
                self.tracking_reliable = True
                if self.last_tracking_state != self.tracking_reliable:
                    self.get_logger().info(
                        f"Tracking started: uncertainty={pos_uncertainty:.3f}m, sensors={active_3d_sensors}(3D)+{active_2d_sensors}(2D)"
                    )
            elif self.tracking_reliable and true_count == 0:  # All 3 must be unreliable (was 1/5)
                # Special case for sensor gaps with very low uncertainty
                if all_sensors_gap and pos_uncertainty < (self.position_uncertainty_threshold * 2.0) and motion_state == "long_stationary":
                    # Temporary loss - maintain tracking during brief gaps for long-stationary objects with low uncertainty
                    if self.debug_level >= 1:
                        self.get_logger().info(
                            f"Maintaining tracking despite sensor gap: uncertainty={pos_uncertainty:.3f}m < special threshold"
                        )
                else:
                    self.tracking_reliable = False
                    if self.last_tracking_state != self.tracking_reliable:
                        self.get_logger().info(
                            f"Tracking lost: uncertainty={pos_uncertainty:.3f}m, sensors={active_3d_sensors}(3D)+{active_2d_sensors}(2D)"
                        )
        else:
            # For moving objects, be even more responsive:
            # 1/3 reliable to start, 3/3 unreliable to stop (much harder to lose tracking)
            if not self.tracking_reliable and true_count >= 1:  # Changed from 2/5 to 1/3
                self.tracking_reliable = True
                if self.last_tracking_state != self.tracking_reliable:
                    self.get_logger().info(
                        f"Tracking started: uncertainty={pos_uncertainty:.3f}m, sensors={active_3d_sensors}(3D)+{active_2d_sensors}(2D)"
                    )
            elif self.tracking_reliable and true_count == 0:  # Changed from 2/5 to 0/3 (All 3 must be unreliable)
                self.tracking_reliable = False
                if self.last_tracking_state != self.tracking_reliable:
                    self.get_logger().info(
                        f"Tracking lost: uncertainty={pos_uncertainty:.3f}m, sensors={active_3d_sensors}(3D)+{active_2d_sensors}(2D)"
                    )
                    
        self.last_tracking_state = self.tracking_reliable
        return self.tracking_reliable

    def init_motion_state_tracking(self):
        """Initialize motion state tracking variables."""
        # Initialize basic motion state tracking
        self.motion_state = "unknown"
        self.prev_motion_state = "unknown"
        self.motion_state_counts = {
            "stationary": 0,
            "long_stationary": 0,
            "small_movement": 0,
            "medium_fast": 0,
            "unknown": 0
        }
        
        # Motion state confidence tracking
        self.motion_state_confidence = {
            "stationary": 0.5,
            "long_stationary": 0.5,
            "small_movement": 0.5, 
            "medium_fast": 0.5,
            "unknown": 0.5
        }
        
        # Velocity credibility tracking
        self.velocity_credibility = 1.0
        self.velocity_confidence = 1.0
        self.state_transition_evidence = {
            "stationary": 0,
            "small_movement": 0,
            "medium_fast": 0
        }
        self.stationary_start_time = None
        self.last_long_stationary_log = 0
        
        self.get_logger().info("Motion state tracking initialized")

    def process_velocity_measurements(self, velocities, times=None):
        """
        Process velocity measurements to filter out implausible values.
        
        Args:
            velocities (list): List of velocity vectors
            times (list): Optional list of timestamps for the velocities
            
        Returns:
            tuple: (filtered_velocities, avg_velocity, implausible_detected)
        """
        # Initialize result variables
        filtered_velocities = []
        implausible_detected = False
        
        # Basic motion state estimation (without recursion)
        avg_speed = 0.0
        valid_count = 0
        
        for vel in velocities:
            if isinstance(vel, (list, tuple, np.ndarray)) and len(vel) >= 2:
                # For 2D velocities, calculate magnitude in the x-y plane
                speed = math.sqrt(vel[0]**2 + vel[1]**2)
                avg_speed += speed
                valid_count += 1
        
        # Calculate average speed
        if valid_count > 0:
            avg_speed /= valid_count
        
        # Simple heuristic for motion state based on speed
        if avg_speed < 0.03:
            simple_motion_state = "stationary"
        elif avg_speed < 0.25:
            simple_motion_state = "small_movement"
        else:
            simple_motion_state = "medium_fast"
        
        # Apply filtering based on the simple motion state
        max_speed_threshold = 5.0  # Default max speed
        
        if simple_motion_state == "stationary":
            max_speed_threshold = 0.5  # Lower threshold for stationary
        elif simple_motion_state == "small_movement":
            max_speed_threshold = 2.0  # Medium threshold
        else:
            max_speed_threshold = 5.0  # Higher threshold for fast movement
        
        # Filter velocities
        for vel in velocities:
            if isinstance(vel, (list, tuple, np.ndarray)) and len(vel) >= 2:
                # Calculate speed (magnitude) in the horizontal plane
                speed = math.sqrt(vel[0]**2 + vel[1]**2)
                
                if speed <= max_speed_threshold:
                    filtered_velocities.append(vel)
                else:
                    # This velocity is implausible - filter it out
                    implausible_detected = True
                    
                    # Add a scaled-down version to maintain continuity
                    scale_factor = max_speed_threshold / speed
                    if isinstance(vel, np.ndarray):
                        scaled_vel = vel.copy() * scale_factor
                    else:
                        scaled_vel = [v * scale_factor for v in vel]
                    filtered_velocities.append(scaled_vel)
        
        # Calculate final average velocity
        if filtered_velocities:
            if isinstance(filtered_velocities[0], np.ndarray):
                avg_velocity = np.mean([math.sqrt(v[0]**2 + v[1]**2) for v in filtered_velocities])
            else:
                avg_velocity = sum([math.sqrt(v[0]**2 + v[1]**2) for v in filtered_velocities]) / len(filtered_velocities)
        else:
            avg_velocity = 0.0
        
        return filtered_velocities, avg_velocity, implausible_detected

    def estimate_3d_from_2d(self, detection_msg, bbox_data):
        """
        Estimate a 3D position from a 2D detection and bounding box.
        
        Args:
            detection_msg (PointStamped): The 2D detection message
            bbox_data (dict): Bounding box data with width, height, and timestamp
            
        Returns:
            PointStamped: Estimated 3D position or None if estimation fails
        """
        try:
            current_time = time.time()
            
            # Get current motion state for adaptive threshold
            motion_state = self.detect_motion_state() if hasattr(self, 'detect_motion_state') else "unknown"

            # Set age threshold based on motion state
            if motion_state == "stationary":
                max_bbox_age = 5.0  # Allow older bbox data for stationary objects
            elif motion_state == "long_stationary":
                max_bbox_age = 7.0  # Even longer for long-term stationary objects
            elif motion_state == "small_movement":
                max_bbox_age = 3.0  # Slightly increased for slow movement
            else:  # medium_fast or unknown
                max_bbox_age = 2.5  # Keep default for fast movement
            
            # Get the actual age
            bbox_age = current_time - bbox_data.get('timestamp', 0)

            # Check if bbox data is recent enough
            if bbox_age > max_bbox_age:
                self.get_logger().warn(f"Bbox data too old: {bbox_age:.2f}s > {max_bbox_age:.1f}s threshold ({motion_state} state)")
                return None

            # For slightly outdated bbox data, apply a confidence penalty
            age_penalty = 1.0
            if bbox_age > (max_bbox_age * 0.75):
                # If we're using the data despite it being somewhat old, log this
                age_penalty = 1.0 + (bbox_age / max_bbox_age) * 0.2  # Up to 20% penalty
                self.get_logger().debug(f"Using older bbox data: {bbox_age:.2f}s (applying {(age_penalty-1.0)*100:.1f}% distance penalty)")
                
            # Get bounding box dimensions
            bbox_width = bbox_data.get('width', 0)
            bbox_height = bbox_data.get('height', 0)
            
            if bbox_width <= 0 or bbox_height <= 0:
                self.get_logger().warn(f"Invalid bbox dimensions: {bbox_width}x{bbox_height}")
                return None
                
            # Known basketball diameter in meters
            basketball_diameter_meters = self.basketball_radius * 2
            
            # Calculate distance based on apparent size vs actual size
            focal_length_pixels = 345.58  # Calibrated focal length for camera
            estimated_distance = (basketball_diameter_meters * focal_length_pixels) / bbox_width
            
            # Apply age penalty to increase distance estimate for older data
            estimated_distance *= age_penalty
            
            # Get camera to reference frame transform
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.reference_frame,
                    'ascamera_color_0',  # Frame of the YOLO camera
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.2)
                )
            except Exception as te:
                self.get_logger().error(f"Transform lookup failed: {str(te)}")
                return None
            
            # Camera's position in reference frame
            camera_pos_x = transform.transform.translation.x
            camera_pos_y = transform.transform.translation.y
            camera_pos_z = transform.transform.translation.z
            
            # Get image dimensions
            image_width = 320  # Width of the camera image
            image_height = 320  # Height of the camera image
            image_center_x = image_width / 2
            image_center_y = image_height / 2
            
            # Get the detection coordinates
            detection_x = detection_msg.point.x  # X pixel coordinate in image
            detection_y = detection_msg.point.y  # Y pixel coordinate in image
            
            # Calculate offsets from center of image
            offset_x = detection_x - image_center_x
            offset_y = detection_y - image_center_y
            
            # Camera coordinate system mapping:
            # - Z axis points forward
            # - X axis points right
            # - Y axis points down
            
            # Convert pixel offsets to direction vector using focal length
            camera_dir_z = focal_length_pixels  # Z is forward in camera frame
            camera_dir_x = offset_x             # X is right in camera frame
            camera_dir_y = offset_y             # Y is down in camera frame
            
            # Normalize the direction vector
            dir_magnitude = math.sqrt(camera_dir_x**2 + camera_dir_y**2 + camera_dir_z**2)
            if dir_magnitude > 0:
                camera_dir_x /= dir_magnitude
                camera_dir_y /= dir_magnitude
                camera_dir_z /= dir_magnitude
            
            # Extract rotation quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            
            # Convert quaternion to rotation matrix
            norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            qw /= norm
            qx /= norm
            qy /= norm
            qz /= norm
            
            # Convert to rotation matrix elements
            xx = qx * qx
            xy = qx * qy
            xz = qx * qz
            xw = qx * qw
            yy = qy * qy
            yz = qy * qz
            yw = qy * qw
            zz = qz * qz
            zw = qz * qw
            
            # Rotation matrix
            r00 = 1 - 2 * (yy + zz)
            r01 = 2 * (xy - zw)
            r02 = 2 * (xz + yw)
            r10 = 2 * (xy + zw)
            r11 = 1 - 2 * (xx + zz)
            r12 = 2 * (yz - xw)
            r20 = 2 * (xz - yw)
            r21 = 2 * (yz + xw)
            r22 = 1 - 2 * (xx + yy)
            
            # Apply rotation to camera direction
            ref_dir_x = r00 * camera_dir_x + r01 * camera_dir_y + r02 * camera_dir_z
            ref_dir_y = r10 * camera_dir_x + r11 * camera_dir_y + r12 * camera_dir_z
            ref_dir_z = r20 * camera_dir_x + r21 * camera_dir_y + r22 * camera_dir_z
            
            # Normalize direction vector
            dir_magnitude = math.sqrt(ref_dir_x*ref_dir_x + ref_dir_y*ref_dir_y + ref_dir_z*ref_dir_z)
            if dir_magnitude > 0:
                ref_dir_x /= dir_magnitude
                ref_dir_y /= dir_magnitude
                ref_dir_z /= dir_magnitude
            
            # Calculate estimated position in reference frame
            est_x = camera_pos_x + estimated_distance * ref_dir_x
            est_y = camera_pos_y + estimated_distance * ref_dir_y
            est_z = self.basketball_z_height  # Always at basketball height above ground
            
            # Create and return a new 3D point message in the reference frame
            estimated_point = PointStamped()
            estimated_point.header.stamp = detection_msg.header.stamp
            estimated_point.header.frame_id = self.reference_frame
            estimated_point.point.x = est_x
            estimated_point.point.y = est_y
            estimated_point.point.z = est_z
            
            # Initialize estimation counter if not present
            if not hasattr(self, '_3d_estimation_counter'):
                self._3d_estimation_counter = 0
            
            # Increment counter and log details every 3 times
            self._3d_estimation_counter += 1
            if self._3d_estimation_counter % 3 == 0:
                self.get_logger().info(
                    f"3D estimation details: bbox={bbox_width}x{bbox_height}, "
                    f"distance={estimated_distance:.2f}m, "
                    f"camera_dir=({camera_dir_x:.2f}, {camera_dir_y:.2f}, {camera_dir_z:.2f}), "
                    f"pos=({est_x:.2f}, {est_y:.2f}, {est_z:.2f})"
                )
                
            return estimated_point
            
        except Exception as e:
            self.get_logger().error(f"Error estimating 3D from YOLO 2D: {str(e)}")
            self.get_logger().error(traceback.format_exc())
            return None

    def cache_static_transforms(self):
        """Cache static transforms to avoid repeated lookups during execution."""
        self.get_logger().info("Caching static transforms...")

        try:
            # Cache camera to base transform
            self.tf_camera_to_base = self.tf_buffer.lookup_transform(
                self.reference_frame, 'ascamera_color_0', 
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1)
            )
            self.get_logger().info(
                f"Cached static transform: ascamera_color_0 → {self.reference_frame}: "
                f"trans=({self.tf_camera_to_base.transform.translation.x:.3f}, "
                f"{self.tf_camera_to_base.transform.translation.y:.3f}, "
                f"{self.tf_camera_to_base.transform.translation.z:.3f})"
            )
            
            # Cache lidar to base transform
            self.tf_lidar_to_base = self.tf_buffer.lookup_transform(
                self.reference_frame, 'lidar_frame', 
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1)
            )
            self.get_logger().info(
                f"Cached static transform: lidar_frame → {self.reference_frame}: "
                f"trans=({self.tf_lidar_to_base.transform.translation.x:.3f}, "
                f"{self.tf_lidar_to_base.transform.translation.y:.3f}, "
                f"{self.tf_lidar_to_base.transform.translation.z:.3f})"
            )
            
        except Exception as e:
            self.get_logger().error(f"Failed to cache static transforms: {str(e)}")
            self.tf_camera_to_base = None
            self.tf_lidar_to_base = None

    def filter_update(self):
        """Modified filter update method to be more responsive to sensor data showing movement"""
        # Skip if not active or not initialized
        if not self.is_activated or not self.initialized:
            return
        
        try:
            # Get current time
            current_time = time.time()
            
            # Calculate time step since last update
            if self.last_update_time is None:
                dt = 0.1  # Default initial time step
            else:
                dt = current_time - self.last_update_time
                # Limit dt to reasonable values
                dt = min(dt, 0.5)  # Cap at 0.5 seconds
            
            # Apply gap-aware covariance adjustment
            self.adjust_covariance_for_gaps()
            
            # Find synchronized measurements
            measurements = self.sensor_buffer.find_synchronized_measurements(min_sensors=1)
            
            # ADDED: Check for position evidence suggesting movement
            motion_evidence_detected = False
            largest_discrepancy = 0.0
            largest_discrepancy_sensor = None
            
            # Check all measurements for significant discrepancies from current state
            for sensor, msg in measurements.items():
                try:
                    # Transform measurement to reference frame
                    if sensor.endswith('_2d'):
                        if sensor in self.bbox_data:
                            transformed = self.estimate_3d_from_2d(msg, self.bbox_data[sensor])
                            if transformed is None:
                                continue
                        else:
                            continue
                    else:
                        # For 3D sensors, transform point
                        transformed = self.transform_point(msg, self.reference_frame, False)
                        if transformed is None:
                            continue
                    
                    # Calculate discrepancy from current state
                    dx = transformed.point.x - self.state[0]
                    dy = transformed.point.y - self.state[1]
                    discrepancy = math.sqrt(dx*dx + dy*dy)
                    
                    # Track largest discrepancy
                    if discrepancy > largest_discrepancy:
                        largest_discrepancy = discrepancy
                        largest_discrepancy_sensor = sensor
                    
                    # Check for significant discrepancy
                    # Use lower threshold (15cm) when in long_stationary to be more responsive
                    threshold = 0.15 if self.motion_state == "long_stationary" else 0.20
                    if discrepancy > threshold:
                        motion_evidence_detected = True
                        self.get_logger().info(
                            f"Motion evidence detected: {sensor} shows {discrepancy:.2f}m discrepancy "
                            f"(sensor=({transformed.point.x:.2f}, {transformed.point.y:.2f}), state=({self.state[0]:.2f}, {self.state[1]:.2f}))"
                        )
                except Exception as e:
                    self.get_logger().warn(f"Error processing {sensor} for motion evidence: {str(e)}")
            
            # ADDED: Force motion state update when clear evidence is present
            if motion_evidence_detected and self.motion_state in ["stationary", "long_stationary"]:
                self.prev_motion_state = self.motion_state
                self.motion_state = "medium_fast"  # Skip directly to medium_fast for faster response
                self.get_logger().info(
                    f"Forcing motion state change based on position evidence: {self.prev_motion_state} -> "
                    f"medium_fast (discrepancy={largest_discrepancy:.2f}m from {largest_discrepancy_sensor})"
                )
                
                # Update confidence values
                if hasattr(self, 'motion_state_confidence'):
                    self.motion_state_confidence["medium_fast"] = 0.8
                    self.motion_state_confidence["small_movement"] = 0.5
                    self.motion_state_confidence["stationary"] = 0.1
                    self.motion_state_confidence["long_stationary"] = 0.0
            
            # ENHANCED: Handle synthesizing measurements during gaps with more aggressive approach
            if len(measurements) < 2:
                # Try to synthesize measurements for missing sensors
                missing_sensors = []
                for sensor in self.expected_sensors:
                    if sensor not in measurements and self.sensor_counts.get(sensor, 0) > 0:
                        last_time = self.last_detection_time.get(sensor, 0)
                        gap_duration = current_time - last_time
                        
                        # More aggressive gap threshold - 0.3s instead of 0.5s
                        if gap_duration > 0.3 and gap_duration < 3.5:  # Extended from 3.0
                            missing_sensors.append(sensor)
                
                # Attempt to synthesize measurements more aggressively
                for sensor in missing_sensors:
                    # Create a state predictor callback
                    def state_predictor(target_time):
                        dt = target_time - self.last_update_time if self.last_update_time else 0.1
                        # Use higher prediction speed when in movement state
                        speed_factor = 1.2 if self.motion_state in ["small_movement", "medium_fast"] else 1.0
                        predicted = [
                            self.state[0] + self.state[2] * dt * speed_factor,  # x + vx*dt (with boost)
                            self.state[1] + self.state[3] * dt * speed_factor,  # y + vy*dt (with boost)
                            self.basketball_z_height  # z is fixed
                        ]
                        return predicted
                    
                    # Try to synthesize a measurement
                    synth_measurement, confidence = self.sensor_buffer.synthesize_measurement(
                        sensor, current_time, state_predictor
                    )
                    
                    if synth_measurement is not None:
                        # Boost confidence for synthesis during movement
                        if self.motion_state in ["small_movement", "medium_fast"]:
                            confidence = min(0.8, confidence + 0.1)  # Boost up to 0.8 max
                        
                        # Track synthetic measurement
                        if not hasattr(self, 'synthetic_measurement_info'):
                            self.synthetic_measurement_info = {}
                        
                        # Store information about this measurement
                        synth_id = f"{sensor}_synth_{current_time}"
                        self.synthetic_measurement_info[synth_id] = {
                            'is_synthesized': True,
                            'synthesis_confidence': confidence,
                            'timestamp': current_time
                        }
                        
                        # Add to measurements
                        measurements[f"{sensor}_synthesized"] = synth_measurement
                        
                        # Log synthesis
                        if self.debug_level >= 1:
                            self.get_logger().info(
                                f"Synthesized measurement for {sensor} during "
                                f"{current_time - self.last_detection_time.get(sensor, 0):.2f}s gap "
                                f"with confidence={confidence:.2f}"
                            )
            
            # Update sync quality metrics
            self.sync_quality_metrics['attempt_counts'] += 1
            if measurements:
                self.sync_quality_metrics['sync_counts'] += 1
                self.sync_quality_metrics['success_rate'] = (
                    self.sync_quality_metrics['sync_counts'] / self.sync_quality_metrics['attempt_counts']
                )
                
                # Track sensor availability
                for sensor in self.expected_sensors:
                    if sensor in measurements:
                        if sensor not in self.sync_quality_metrics['sensor_availability']:
                            self.sync_quality_metrics['sensor_availability'][sensor] = 1
                        else:
                            self.sync_quality_metrics['sensor_availability'][sensor] += 1
            
            # Predict state forward to current time
            self.predict_state(dt)
            
            # Check for filter divergence
            self.check_filter_divergence()
            
            # Check if we're in refinement phase
            in_refinement = getattr(self, 'in_refinement_phase', False)
            
            # Update state with measurements if available
            successful_update = False
            if measurements:
                successful_update = self.update_state(measurements)
                
                # Record sensor reliability
                for sensor in measurements.keys():
                    self.sensor_reliability_tracker.record_measurement(sensor, successful_update)
            
            # Update last update time
            self.last_update_time = current_time
            
            # Update uncertainty metrics
            self.position_uncertainty = math.sqrt(max(0.0, np.trace(self.covariance[0:2, 0:2]) / 2.0))
            self.velocity_uncertainty = math.sqrt(max(0.0, np.trace(self.covariance[2:4, 2:4]) / 2.0))
            
            # Store state in history buffers
            if hasattr(self, 'position_history'):
                # Create 3D position with fixed z-height
                pos_3d = [self.state[0], self.state[1], self.basketball_z_height]
                self.position_history.append(pos_3d)
            
            if hasattr(self, 'velocity_history'):
                # Create 3D velocity with zero z component
                vel_3d = [self.state[2], self.state[3], 0.0]
                self.velocity_history.append(vel_3d)
            
            if hasattr(self, 'time_history'):
                self.time_history.append(current_time)
            
            # Add state to smoother
            if hasattr(self, 'smoothed_state_estimator'):
                pos_3d = np.array([self.state[0], self.state[1], self.basketball_z_height])
                vel_3d = np.array([self.state[2], self.state[3], 0.0])
                
                self.smoothed_state_estimator.add_state(
                    pos_3d, vel_3d, self.position_uncertainty, current_time
                )
                
                # Get smoothed state - MODIFIED to be less aggressive during movement
                smoothed = self.smoothed_state_estimator.get_smoothed_state()
                if smoothed and self.position_uncertainty > 0.1:
                    # Use less smoothing for moving objects - only partial smoothing
                    if self.motion_state in ["small_movement", "medium_fast"]:
                        # Blend factor - 40% from smoothed state, 60% from raw state
                        blend = 0.4
                        smooth_pos = smoothed['position'][0:2].copy()
                        smooth_vel = smoothed['velocity'][0:2].copy()
                        
                        # Partially apply smoothing
                        self.state[0] = blend * smooth_pos[0] + (1 - blend) * self.state[0]
                        self.state[1] = blend * smooth_pos[1] + (1 - blend) * self.state[1]
                        self.state[2] = blend * smooth_vel[0] + (1 - blend) * self.state[2]
                        self.state[3] = blend * smooth_vel[1] + (1 - blend) * self.state[3]
                    else:
                        # Full smoothing for stationary objects
                        self.state[0:2] = smoothed['position'][0:2].copy()
                        self.state[2:4] = smoothed['velocity'][0:2].copy()
            
            # Update tracking status
            self.update_tracking_status()
            
            # Publish state
            self.publish_state()
            
            # Publish uncertainty
            self.publish_uncertainty()
            
            # Update diagnostics
            self.update_diagnostics()
            
            # Apply flat ground constraints
            self.apply_flat_ground_constraints()
            
            # Handle sensor recovery
            self.handle_sensor_recovery()
            
            # Update motion state
            self.detect_motion_state()
            
            # Decay sensor reliability
            if hasattr(self, 'sensor_reliability_tracker'):
                self.sensor_reliability_tracker.decay_unused_sensors()
            
        except Exception as e:
            self.get_logger().error(f"Error during filter update: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            
    def publish_state(self):
        """
        Publish the current state estimate with improved responsiveness to sensor data,
        especially during movement transitions.
        """
        # Skip if not active
        if not self.is_activated:
            return
            
        # Create current position as 3D point from our 4D state
        current_pos = [float(self.state[0]), float(self.state[1]), float(self.basketball_z_height)]
        current_time = time.time()
        
        # ENHANCED: Check for significant discrepancy between ALL sensor data and fusion state
        latest_sensor_pos = None
        latest_sensor_time = 0
        latest_sensor_source = None
        significant_discrepancy = False
        movement_detected = self.motion_state in ["small_movement", "medium_fast"]
        
        # Lower discrepancy thresholds to be much more responsive
        movement_threshold = 0.15  # Reduced from 0.3 - detect 15cm discrepancy during movement
        stationary_threshold = 0.25  # Reduced from 0.5 - detect 25cm discrepancy during stationary
        threshold = movement_threshold if movement_detected else stationary_threshold
        
        # Check the most recent sensor data from ALL sensors (not just 3D)
        for sensor in ['lidar', 'yolo_3d', 'hsv_3d', 'yolo_2d', 'hsv_2d']:
            sensor_msg = self.sensor_buffer.get_latest_measurement(sensor)
            if sensor_msg is None:
                continue
                
            # Get timestamp from message
            sensor_time = sensor_msg.header.stamp.sec + sensor_msg.header.stamp.nanosec / 1e9
            
            # Check freshness - EXTENDED window to 1.0 second
            if current_time - self.last_detection_time.get(sensor, 0) < 1.0:  # Extended from 0.5
                # For 2D sensors, we need to estimate 3D position
                if sensor.endswith('_2d'):
                    # Check if we have bbox data for this sensor
                    if sensor in self.bbox_data:
                        # Estimate 3D from 2D
                        transformed = self.estimate_3d_from_2d(sensor_msg, self.bbox_data[sensor])
                        if transformed is None:
                            continue
                    else:
                        continue
                else:
                    # For 3D sensors, transform to reference frame
                    transformed = self.transform_point(sensor_msg, self.reference_frame, False)
                    if transformed is None:
                        continue
                
                # Check if this measurement is the most recent
                if sensor_time > latest_sensor_time:
                    latest_sensor_time = sensor_time
                    latest_sensor_pos = [transformed.point.x, transformed.point.y, transformed.point.z]
                    latest_sensor_source = sensor
                    
                    # Calculate distance between sensor reading and fusion state
                    dx = transformed.point.x - current_pos[0]
                    dy = transformed.point.y - current_pos[1]
                    distance_diff = math.sqrt(dx*dx + dy*dy)
                    
                    # Detect significant discrepancy with LOWER threshold
                    if distance_diff > threshold:
                        significant_discrepancy = True
                        
                        # Store distance for logging
                        discrepancy_distance = distance_diff
                        
                        if self.debug_level >= 1:
                            self.get_logger().info(
                                f"Detected position discrepancy: {sensor}={distance_diff:.2f}m from fusion state, "
                                f"sensor=({transformed.point.x:.2f}, {transformed.point.y:.2f}), "
                                f"fusion=({current_pos[0]:.2f}, {current_pos[1]:.2f})"
                            )
        
        # IMPROVED: If significant discrepancy detected, adjust state toward sensor data with more responsive approach
        if significant_discrepancy and latest_sensor_pos is not None:
            # Base blend factor - increased to be MUCH more aggressive (0.9)
            blend_factor = 0.9  # Increased from 0.85
            
            # Apply additional boost for stationary->movement transitions
            if hasattr(self, 'prev_motion_state') and self.motion_state != self.prev_motion_state:
                # If transitioning from stationary to any movement state
                if self.prev_motion_state in ["stationary", "long_stationary"] and \
                self.motion_state not in ["stationary", "long_stationary"]:
                    # Use an extremely aggressive blend factor (98%) to rapidly trust sensor data
                    blend_factor = 0.98
                    self.get_logger().info(
                        f"Using aggressive blend factor during transition from {self.prev_motion_state} to {self.motion_state}: {blend_factor:.2f}"
                    )
            
            # ADDED: Force strong blend when in long_stationary with discrepancy
            if self.motion_state == "long_stationary" and discrepancy_distance > 0.1:
                # For significant discrepancy during long_stationary, trust sensor data almost completely
                blend_factor = 0.95
                self.get_logger().info(
                    f"Using high blend factor to overcome long_stationary state: {blend_factor:.2f} (distance_diff={discrepancy_distance:.3f}m)"
                )
            
            # ADDED: Apply extreme blend factor for large discrepancies
            if discrepancy_distance > 0.25:  # 25cm+ discrepancy
                blend_factor = 0.98  # Almost completely trust sensor data
                self.get_logger().info(
                    f"Applying extreme blend factor for large discrepancy: {blend_factor:.2f} (distance_diff={discrepancy_distance:.3f}m)"
                )
            
            # Apply more trust to LiDAR data which is typically more accurate
            if latest_sensor_source == 'lidar':
                blend_factor = min(0.99, blend_factor + 0.05)  # Increased from 0.1 to 0.05 but with higher cap
            
            # Apply minimum blend threshold to ensure significant movement
            min_blend = 0.8  # Increased from 0.7
            blend_factor = max(min_blend, blend_factor)
            
            # Apply blend
            blended_pos = [
                (1 - blend_factor) * current_pos[0] + blend_factor * latest_sensor_pos[0],
                (1 - blend_factor) * current_pos[1] + blend_factor * latest_sensor_pos[1],
                self.basketball_z_height  # Keep fixed height
            ]
            
            # Update state with blended position
            self.state[0] = blended_pos[0]
            self.state[1] = blended_pos[1]
            
            # ADDED: Update velocity state to reflect movement
            if discrepancy_distance > 0.1 and hasattr(self, 'last_update_time'):
                dt = current_time - self.last_update_time
                if dt > 0.001:  # Avoid division by zero
                    # Calculate implied velocity from position change
                    vx = (latest_sensor_pos[0] - current_pos[0]) / dt
                    vy = (latest_sensor_pos[1] - current_pos[1]) / dt
                    
                    # Cap velocity to reasonable values
                    speed = math.sqrt(vx*vx + vy*vy)
                    max_speed = 5.0  # Maximum reasonable speed
                    if speed > max_speed:
                        scale = max_speed / speed
                        vx *= scale
                        vy *= scale
                    
                    # Blend velocity (70% from position change, 30% from current state)
                    self.state[2] = 0.3 * self.state[2] + 0.7 * vx
                    self.state[3] = 0.3 * self.state[3] + 0.7 * vy
                    
                    self.get_logger().info(
                        f"Updated velocity from position change: v=({self.state[2]:.2f}, {self.state[3]:.2f}) m/s, speed={speed:.2f}m/s"
                    )
            
            # Log this adjustment
            if self.debug_level >= 1:
                self.get_logger().info(
                    f"Adjusted fusion state toward sensor data: blend={blend_factor:.2f}, "
                    f"motion={self.motion_state}, distance_diff={discrepancy_distance:.3f}m, uncertainty={self.position_uncertainty:.3f}m"
                )
                
            # Use the blended position for the rest of this method
            current_pos = blended_pos
        
        # SIMPLIFIED: Pass through GroundPositionFilter as second stage
        filtered_pos = self.ground_filter.update(current_pos, current_time)
        
        # Update our state with the filtered position
        # This keeps the Kalman filter state in sync with published positions
        self.state[0:2] = filtered_pos[0:2]  # Update x,y position
        
        # Get velocity from ground filter (more accurate for rolling balls)
        ground_velocity = self.ground_filter.get_velocity()
        
        # IMPROVED: Update velocity state from ground filter for ANY motion 
        # Don't restrict to stronger movements to improve responsiveness
        ground_speed = self.ground_filter.get_speed()
        if ground_speed > 0.05:  # Lowered from 0.1 - respond to just 5cm/s
            # Only use x,y components of the ground_velocity (which is 3D)
            self.state[2] = ground_velocity[0]  # x velocity
            self.state[3] = ground_velocity[1]  # y velocity
        
        # Calculate distance and direction to the ball
        distance = math.sqrt(filtered_pos[0]**2 + filtered_pos[1]**2)
        direction = math.degrees(math.atan2(filtered_pos[1], filtered_pos[0]))
        
        # Log the distance and direction periodically to avoid log flooding
        if hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 5 == 0:
            self.get_logger().info(
                f"Ball position: distance={distance:.2f}m, direction={direction:.1f} degrees, "
                f"pos=({filtered_pos[0]:.2f}, {filtered_pos[1]:.2f}, {filtered_pos[2]:.2f})"
            )
        
        # Create position message
        pos_msg = PointStamped()
        pos_msg.header.stamp = self.get_clock().now().to_msg()
        pos_msg.header.frame_id = self.reference_frame
        pos_msg.point.x = float(filtered_pos[0])
        pos_msg.point.y = float(filtered_pos[1])
        pos_msg.point.z = float(filtered_pos[2])  # Always at basketball height
        self.position_pub.publish(pos_msg)
        
        # Create velocity message
        vel_msg = TwistStamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = self.reference_frame
        
        # Use ground filter velocity if significant, otherwise use Kalman filter
        if ground_speed > 0.05:  # Lowered from 0.1
            vel_msg.twist.linear.x = float(ground_velocity[0])
            vel_msg.twist.linear.y = float(ground_velocity[1])
        else:
            vel_msg.twist.linear.x = float(self.state[2])
            vel_msg.twist.linear.y = float(self.state[3])
            
        vel_msg.twist.linear.z = 0.0  # Zero vertical velocity for ground movement
        self.velocity_pub.publish(vel_msg)
        
        # Get statistics from ground filter occasionally
        if hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 100 == 0:
            stats = self.ground_filter.get_statistics()
            if stats:
                self.get_logger().info(
                    f"Ground filter stats: speed={stats['current_speed']:.2f}m/s, "
                    f"avg={stats['average_speed']:.2f}m/s, jumps={stats['position_jumps']}"
                )

    def publish_uncertainty(self):
        """Publish the position uncertainty."""
        msg = Float32()
        msg.data = self.position_uncertainty
        self.uncertainty_pub.publish(msg)
    
    def publish_diagnostics(self):
        """Publish diagnostic information."""
        # Log validation performance periodically
        if hasattr(self, 'validation_manager') and hasattr(self, 'sync_quality_metrics'):
            if self.sync_quality_metrics.get('attempt_counts', 0) % 50 == 0:
                self.log_validation_performance()
                
        # Create a diagnostics dictionary
        diag = {
            'filter_health': self.filter_health,
            'transform_health': self.transform_health,
            'sensor_health': self.sensor_health,
            'position_uncertainty': self.position_uncertainty,
            'velocity_uncertainty': self.velocity_uncertainty,
            'last_filter_update_time': self.last_filter_update_time,
            'motion_state': self.motion_state
        }
        
        # Add frame and transform diagnostics
        transform_diag = {}
        frames = ["lidar_frame", "ascamera_color_0", "ascamera_camera_link_0"]
        for frame in frames:
            try:
                transform_diag[frame] = {
                    "available": self.tf_buffer.can_transform(self.reference_frame, frame, rclpy.time.Time()),
                    "last_check": time.time()
                }
                
                # Add actual transform details if available
                if transform_diag[frame]["available"]:
                    transform = self.tf_buffer.lookup_transform(
                        self.reference_frame, frame, rclpy.time.Time()
                    )
                    transform_diag[frame]["translation"] = {
                        "x": transform.transform.translation.x,
                        "y": transform.transform.translation.y,
                        "z": transform.transform.translation.z
                    }
            except Exception as e:
                transform_diag[frame] = {"available": False, "error": True}
        
        diag["transform_health"] = {
            "transform_checks": self.transform_checks,
            "transform_successes": self.transform_successes,
            "transform_failures": self.transform_failures,
            "frames": transform_diag
        }
        
        msg = String()
        msg.data = json.dumps(diag)
        self.diagnostics_pub.publish(msg)

    def update_diagnostics(self):
        """Update diagnostic information."""
        self.filter_health = max(0.0, min(1.0, 1.0 - (self.position_uncertainty / 10.0)))
        self.transform_health = 1.0 if self.transform_confirmed else 0.0
        self.sensor_health = max(0.0, min(1.0, 1.0 - (self.position_uncertainty / 10.0)))
        self.last_filter_update_time = time.time()

    def log_error(self, msg):
        """Log an error message and add to error queue."""
        self.get_logger().error(msg)
        self.errors.append(msg)

    def log_warning(self, msg):
        """Log a warning message and add to warning queue."""
        self.get_logger().warn(msg)
        self.warnings.append(msg)

    def _cleanup_timers(self):
        """Clean up timers."""
        for timer in self._timer_list:
            self.destroy_timer(timer)
        self._timer_list = []

    def log_validation_performance(self):
        """
        Log statistics about the validation system's performance.
        Shows false positive/negative rates and adaptive thresholds.
        """
        if not hasattr(self, 'validation_manager'):
            return
            
        try:
            # Get a summary of false positive and negative rates
            false_positive_rates = []
            false_negative_rates = []
            
            # Collect statistics for each sensor
            sensor_stats = []
            for sensor in self.validation_manager.sensors:
                fp_rate = self.validation_manager.false_positive_rate.get(sensor, 0.0)
                fn_rate = self.validation_manager.false_negative_rate.get(sensor, 0.0)
                
                if hasattr(self.validation_manager, 'adaptive_thresholds'):
                    threshold = self.validation_manager.adaptive_thresholds.get(sensor, 0.0)
                else:
                    threshold = 0.0
                    
                false_positive_rates.append(fp_rate)
                false_negative_rates.append(fn_rate)
                
                # Track validation history counts
                history_count = 0
                if sensor in self.validation_manager.validation_history:
                    history_count = len(self.validation_manager.validation_history[sensor])
                
                # Add to sensor-specific stats
                sensor_stats.append(f"{sensor}: FP={fp_rate:.2f}, FN={fn_rate:.2f}, threshold={threshold:.1f}, samples={history_count}")
                
            # Calculate average rates across all sensors
            avg_fp_rate = sum(false_positive_rates) / max(1, len(false_positive_rates))
            avg_fn_rate = sum(false_negative_rates) / max(1, len(false_negative_rates))
            
            # Log the overall performance
            self.get_logger().info(
                f"Validation performance: Avg FP rate={avg_fp_rate:.2f}, Avg FN rate={avg_fn_rate:.2f}"
            )
            
            # Log per-sensor statistics
            for stat in sensor_stats:
                self.get_logger().info(f"  {stat}")
                
            # Get current motion state for context
            motion_state = self.detect_motion_state() if hasattr(self, 'detect_motion_state') else "unknown"
            
            # Update context information for validation manager
            if hasattr(self, 'sensor_gap_detection'):
                # Count active gaps
                gap_count = sum(1 for sensor, gap_info in self.sensor_gap_detection.items() 
                                if gap_info.get('gap_detected', False))
                
                if gap_count > 0:
                    self.get_logger().info(f"  Currently {gap_count} sensor gaps detected during {motion_state} state")
                    
            # Log number of consecutive rejections
            if hasattr(self, 'consecutive_rejections_per_sensor'):
                reject_sensors = []
                for sensor, count in self.consecutive_rejections_per_sensor.items():
                    if count > 1:  # Only log if more than 1 rejection
                        reject_sensors.append(f"{sensor}={count}")
                        
                if reject_sensors:
                    self.get_logger().info(f"  Consecutive rejections: {', '.join(reject_sensors)}")
                
        except Exception as e:
            self.get_logger().error(f"Error in log_validation_performance: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    
    # Use MultiThreadedExecutor for better performance on Pi 5
    executor = MultiThreadedExecutor()
    
    # Create the lifecycle node
    node = EnhancedFusionLifecycleNode()
    
    # Add the node to the executor
    executor.add_node(node)
    
    try:
        # Spin the executor
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown the node and executor
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()