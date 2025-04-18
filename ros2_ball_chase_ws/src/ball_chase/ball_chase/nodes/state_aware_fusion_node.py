#!/usr/bin/env python3

"""
Enhanced Fusion Node - Using ROS 2 Lifecycle Management
Converts the original implementation to use ROS 2 Lifecycle Nodes
for better state management and transitions.
Optimized for Raspberry Pi 5 with multi-node resource coordination.
The code needs to stay light on resources
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

    
    def predict_state(self, dt):
        """
        Predict the state forward by dt seconds with zero-tolerance
        for staying in stationary states.
        
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
        gap_factor = min(10.0, max(1.0, dt / 0.1))  # Scale factor based on gap length (doubled max)
        
        # ZERO TOLERANCE: Apply extreme motion state-based scaling
        motion_state = getattr(self, 'motion_state', 'unknown')
        motion_scale = 1.0  # Default scaling
        
        if motion_state == "stationary":
            motion_scale = 3.0  # TRIPLED from default for stationary
        elif motion_state == "long_stationary":
            motion_scale = 5.0  # EXTREME value for long_stationary
        elif motion_state == "small_movement":
            motion_scale = 2.0  # Increased for small_movement
        elif motion_state == "medium_fast":
            motion_scale = 1.5  # Increased for medium_fast
        
        # ZERO TOLERANCE: Super boost for transitions
        if hasattr(self, 'prev_motion_state') and hasattr(self, 'motion_state'):
            if self.motion_state != self.prev_motion_state:
                # If transitioning from stationary to movement
                if self.prev_motion_state in ["stationary", "long_stationary"]:
                    motion_scale *= 5.0  # 5x process noise during transition (was 2.0)
                    self.get_logger().info(
                        f"ZERO TOLERANCE: 5x process noise during {self.prev_motion_state}->{self.motion_state} transition: scale={motion_scale:.1f}"
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
                gap_factor *= (1.0 + avg_gap_level * 3.0)  # Increased from 1.5 to 3.0
        
        # Apply combined scaling to the process noise
        combined_scale = gap_factor * motion_scale
        
        # ZERO TOLERANCE: Massively increased process noise
        # Process noise parameters with adaptive scaling
        q_pos = self.process_noise_pos * dt * combined_scale * 3.0  # Doubled from 1.5
        q_vel = self.process_noise_vel * dt * combined_scale * 3.0  # Doubled from 1.5
        
        # ZERO TOLERANCE: Minimal rolling friction
        # Apply extremely minimal rolling friction - nearly frictionless
        friction_coef = 0.005  # Reduced by 75% from 0.02
        
        # Apply deceleration to horizontal velocity components
        current_velocity = np.linalg.norm(self.state[2:4])  # x-y plane velocity
        if current_velocity > 0:
            # Calculate friction deceleration: a = μg
            deceleration = friction_coef * 9.81  # μg in m/s²
            
            # Don't decelerate more than the current velocity
            max_dv = current_velocity
            dv = min(max_dv, deceleration * dt)
            
            # ZERO TOLERANCE: Skip friction entirely for stationary-to-moving transitions
            apply_friction = True
            if hasattr(self, 'prev_motion_state') and hasattr(self, 'motion_state'):
                if (self.prev_motion_state in ["stationary", "long_stationary"] or 
                    self.motion_state in ["small_movement", "medium_fast"]):
                    # Skip friction during ANY state involving movement
                    apply_friction = False
                    if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 10 == 0:
                        self.get_logger().debug("ZERO TOLERANCE: Skipping friction for movement states")
            
            # Apply proportional deceleration to velocity components if needed
            if apply_friction and dv > 0 and current_velocity > 0:
                factor = 1.0 - (dv / current_velocity)
                self.state[2] *= factor  # Reduce x velocity
                self.state[3] *= factor  # Reduce y velocity
        
        # Fill in the 4x4 process noise matrix with MASSIVELY INCREASED values
        # Position variances
        self._Q_matrix[0, 0] = q_pos * dt**3 / 3.0  # x position variance
        self._Q_matrix[1, 1] = q_pos * dt**3 / 3.0  # y position variance
        
        # Velocity variances
        self._Q_matrix[2, 2] = q_vel * dt          # x velocity variance
        self._Q_matrix[3, 3] = q_vel * dt          # y velocity variance
        
        # Position-velocity covariances
        self._Q_matrix[0, 2] = self._Q_matrix[2, 0] = q_pos * dt**2 / 2.0  # x position-velocity
        self._Q_matrix[1, 3] = self._Q_matrix[3, 1] = q_pos * dt**2 / 2.0  # y position-velocity
        
        # ADDED: Check for zero velocity in non-stationary state - force non-zero velocity
        if hasattr(self, 'motion_state') and self.motion_state in ["small_movement", "medium_fast"]:
            current_velocity = np.linalg.norm(self.state[2:4])
            if current_velocity < 0.05:
                # Force minimum velocity for movement states
                self.get_logger().info(
                    f"ZERO TOLERANCE: Forcing non-zero velocity in {self.motion_state} state "
                    f"(current={current_velocity:.4f}m/s, forcing to 0.05m/s)"
                )
                
                # Set minimal velocity (0.05 m/s) in x direction
                direction = 1.0 if self.state[0] >= 0 else -1.0  # Use position sign for direction
                self.state[2] = 0.05 * direction  # Set minimal x velocity
                self.state[3] = 0.0               # Reset y velocity
        
        # Check for consecutive rejected measurements and increase process noise
        if hasattr(self, 'consecutive_rejections_per_sensor'):
            total_rejections = sum(self.consecutive_rejections_per_sensor.values())
            if total_rejections > 3:  # Reduced from 5
                rejection_scale = min(5.0, 1.0 + (total_rejections / 5.0))  # Up from 3.0
                self._Q_matrix *= rejection_scale
                self.get_logger().info(
                    f"ZERO TOLERANCE: Extreme process noise due to {total_rejections} rejections: scale={rejection_scale:.1f}"
                )
        
        # Predict state using state transition matrix
        self.state = np.dot(self._F_matrix, self.state)
        
        # Predict covariance
        self.covariance = np.dot(np.dot(self._F_matrix, self.covariance), self._F_matrix.T) + self._Q_matrix
        
        # Ensure covariance remains symmetric
        self.covariance = 0.5 * (self.covariance + self.covariance.T)
    
    def detect_and_override_stuck_state(self):
        """
        Ultra-aggressive override mechanism that directly forces state changes
        when ANY sensor shows evidence of movement after being in stationary states.
        """
        # Skip if not initialized or not tracking
        if not self.initialized or not self.tracking_reliable:
            return False
        
        # Get current time for freshness checks
        current_time = time.time()
        
        # STEP 1: Check if we're potentially in a problematic state (ANY stationary state)
        problematic_state = self.motion_state in ["long_stationary", "stationary", "unknown"]
        
        # STEP 2: Examine sensor data from ALL sensors
        sensor_evidence = []
        movement_detected = False
        total_discrepancy = 0.0
        active_sensors = 0
        
        # ULTRA-LOW threshold for movement detection - just 2cm (was 8cm)
        movement_threshold = 0.02
        
        # Examine each sensor
        for sensor in ['lidar', 'yolo_3d', 'hsv_3d', 'yolo_2d', 'hsv_2d']:
            sensor_msg = self.sensor_buffer.get_latest_measurement(sensor)
            if sensor_msg is None:
                continue
            
            # Check freshness - use a generous window of 2 seconds
            if current_time - self.last_detection_time.get(sensor, 0) > 2.0:
                continue
            
            active_sensors += 1
            
            # Transform to get comparable position
            if sensor.endswith('_2d'):
                if sensor in self.bbox_data:
                    # Estimate 3D from 2D
                    transformed = self.estimate_3d_from_2d(sensor_msg, self.bbox_data[sensor])
                    if transformed is None:
                        continue
                else:
                    continue
            else:
                # For 3D sensors, transform point
                transformed = self.transform_point(sensor_msg, self.reference_frame, False)
                if transformed is None:
                    continue
            
            # Calculate discrepancy from current state
            dx = transformed.point.x - self.state[0]
            dy = transformed.point.y - self.state[1]
            discrepancy = math.sqrt(dx*dx + dy*dy)
            
            total_discrepancy += discrepancy
            
            # Check if this sensor shows ANY movement
            if discrepancy > movement_threshold:
                movement_detected = True
                sensor_evidence.append({
                    'sensor': sensor,
                    'discrepancy': discrepancy,
                    'position': [transformed.point.x, transformed.point.y, transformed.point.z]
                })
        
        # STEP 3: ZERO TOLERANCE intervention - ANY movement evidence triggers override
        if active_sensors == 0:
            # No sensors active, can't make a determination
            return False
        
        # Calculate average discrepancy
        avg_discrepancy = total_discrepancy / active_sensors
        
        # Initialize override counter if not present
        if not hasattr(self, 'movement_override_counter'):
            self.movement_override_counter = 0
            self.last_override_time = 0
            self.override_cooldown = False
        
        # ZERO TOLERANCE: Apply override with ANY movement evidence
        if movement_detected:
            # Update counter
            self.movement_override_counter += 1
            
            # Check cooldown - REDUCED to 2 seconds (was 5)
            if current_time - self.last_override_time < 2.0:
                self.override_cooldown = True
            else:
                self.override_cooldown = False
            
            # STEP 4: Apply intervention IMMEDIATELY
            if not self.override_cooldown:
                self.get_logger().warn(
                    f"ZERO TOLERANCE OVERRIDE: Detected {len(sensor_evidence)} sensors showing average discrepancy "
                    f"of {avg_discrepancy:.3f}m while in {self.motion_state} state"
                )
                
                # Log details of each sensor's evidence
                for ev in sensor_evidence:
                    self.get_logger().warn(
                        f"  Evidence from {ev['sensor']}: discrepancy={ev['discrepancy']:.3f}m, "
                        f"position=({ev['position'][0]:.2f}, {ev['position'][1]:.2f})"
                    )
                
                # STEP 5: Force state change and update filter state
                self.prev_motion_state = self.motion_state
                self.motion_state = "medium_fast"  # Force to fast movement state
                
                # Update confidence values to maximum
                if hasattr(self, 'motion_state_confidence'):
                    self.motion_state_confidence["medium_fast"] = 1.0  # MAXIMUM confidence
                    self.motion_state_confidence["small_movement"] = 0.8
                    self.motion_state_confidence["stationary"] = 0.0  # ZERO
                    self.motion_state_confidence["long_stationary"] = 0.0  # ZERO
                    
                # Reset tracking variables for stationary detection
                if hasattr(self, 'stationary_start_time'):
                    self.stationary_start_time = None
                if hasattr(self, 'long_stationary_since'):
                    self.long_stationary_since = None
                
                # STEP 6: MASSIVELY increase covariance to accept new measurements
                self.covariance[0:2, 0:2] *= 10.0  # 10x position uncertainty (was 5x)
                self.covariance[2:4, 2:4] *= 20.0  # 20x velocity uncertainty (was 10x)
                
                # Update uncertainty metrics
                self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:2, 0:2]) / 2.0)
                self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[2:4, 2:4]) / 2.0)
                
                # STEP 7: COMPLETE REPLACEMENT with sensor data
                if sensor_evidence:
                    # Find most discrepant sensor
                    most_discrepant = max(sensor_evidence, key=lambda x: x['discrepancy'])
                    
                    # DIRECT STATE REPLACEMENT - 100% sensor data (was 95%)
                    pos = most_discrepant['position']
                    
                    # Direct replacement of position
                    self.state[0] = pos[0]
                    self.state[1] = pos[1]
                    
                    # Calculate implied velocity based on position change
                    if self.last_update_time is not None:
                        dt = current_time - self.last_update_time
                        if dt > 0.001:  # Avoid division by zero
                            # Calculate implied motion direction
                            dx = pos[0] - self.state[0]
                            dy = pos[1] - self.state[1]
                            
                            # Use significant non-zero velocity in movement direction
                            implied_vx = dx / dt
                            implied_vy = dy / dt
                            
                            # Use implied velocity, with minimum magnitude
                            speed = math.sqrt(implied_vx**2 + implied_vy**2)
                            min_speed = 0.2  # Ensure some velocity
                            
                            if speed < min_speed and speed > 0:
                                # Scale up to minimum speed
                                scale = min_speed / speed
                                implied_vx *= scale
                                implied_vy *= scale
                            elif speed == 0:
                                # If no implied velocity, set some default velocity
                                implied_vx = 0.1
                                implied_vy = 0.1
                            
                            # Set velocity directly - NO blending
                            self.state[2] = implied_vx
                            self.state[3] = implied_vy
                            
                            self.get_logger().warn(
                                f"Setting non-zero velocity: v=({implied_vx:.2f}, {implied_vy:.2f}) m/s"
                            )
                
                # STEP 8: Reset ALL rejection counters
                if hasattr(self, 'consecutive_rejections_per_sensor'):
                    for sensor in self.consecutive_rejections_per_sensor:
                        self.consecutive_rejections_per_sensor[sensor] = 0
                
                # Record override time
                self.last_override_time = current_time
                self.movement_override_counter = 0  # Reset counter
                
                # Log the intervention
                self.get_logger().warn(
                    f"Applied ZERO TOLERANCE override: state={self.motion_state}, "
                    f"position=({self.state[0]:.2f}, {self.state[1]:.2f}), "
                    f"velocity=({self.state[2]:.2f}, {self.state[3]:.2f}), "
                    f"uncertainty={self.position_uncertainty:.3f}m"
                )
                
                return True  # Override applied
        
        # If no intervention was applied but no evidence either, reset counter
        if not movement_detected:
            self.movement_override_counter = 0
        
        return False  # No override needed

    
    def predict_state(self, dt):
        """
        Predict the state forward by dt seconds with balanced approach
        to staying in stationary states.
        
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
        
        # Apply balanced process noise based on prediction duration and motion state
        gap_factor = min(5.0, max(1.0, dt / 0.1))  # Scale factor based on gap length (reduced max)
        
        # Get current motion state once to avoid repeated calls
        motion_state = getattr(self, 'motion_state', 'unknown')
        
        # Apply reasonable motion state-based scaling
        motion_scale = 1.0  # Default scaling
        
        if motion_state == "stationary":
            motion_scale = 1.5  # Moderate boost for stationary
        elif motion_state == "long_stationary":
            motion_scale = 2.0  # Higher boost for long_stationary but not extreme
        elif motion_state == "small_movement":
            motion_scale = 1.3  # Moderate boost for small_movement
        elif motion_state == "medium_fast":
            motion_scale = 1.1  # Small boost for medium_fast
        
        # Apply balanced boost for transitions
        if hasattr(self, 'prev_motion_state') and hasattr(self, 'motion_state'):
            if self.motion_state != self.prev_motion_state:
                # If transitioning from stationary to movement
                if self.prev_motion_state in ["stationary", "long_stationary"]:
                    motion_scale *= 2.5  # Reasonable boost during transition (was 5.0)
                    if self.debug_level >= 1:
                        self.get_logger().info(
                            f"Applying process noise boost during {self.prev_motion_state}->{self.motion_state} transition: scale={motion_scale:.1f}"
                        )
        
        # Factor in sensor gaps with simplified calculation
        if hasattr(self, 'sensor_gap_detection'):
            gap_level = 0.0
            gap_count = 0
            
            # Only check key sensors to reduce computations
            for sensor in ['lidar', 'yolo_3d', 'yolo_2d']:
                if sensor in self.sensor_gap_detection and self.sensor_gap_detection[sensor].get('gap_detected', False):
                    gap_level += self.sensor_gap_detection[sensor].get('gap_level', 0.0)
                    gap_count += 1
                    
            if gap_count > 0:
                avg_gap_level = gap_level / gap_count
                gap_factor *= (1.0 + avg_gap_level * 1.5)  # Reduced from 3.0
        
        # Apply combined scaling to the process noise
        combined_scale = gap_factor * motion_scale
        
        # Process noise parameters with moderate scaling
        q_pos = self.process_noise_pos * dt * combined_scale * 1.5  # Reduced from 3.0
        q_vel = self.process_noise_vel * dt * combined_scale * 1.5  # Reduced from 3.0
        
        # Apply reasonable rolling friction
        friction_coef = 0.01  # Balanced from original 0.02 and reduced 0.005
        
        # Apply deceleration to horizontal velocity components
        current_velocity = np.linalg.norm(self.state[2:4])  # x-y plane velocity
        if current_velocity > 0:
            # Calculate friction deceleration: a = μg
            deceleration = friction_coef * 9.81  # μg in m/s²
            
            # Don't decelerate more than the current velocity
            max_dv = current_velocity
            dv = min(max_dv, deceleration * dt)
            
            # Apply reasonable friction adjustment for movement states
            apply_friction = True
            if motion_state in ["small_movement", "medium_fast"]:
                # Use reduced friction during movement states
                dv *= 0.7  # 30% reduction in friction
            
            # Apply proportional deceleration to velocity components if needed
            if apply_friction and dv > 0 and current_velocity > 0:
                factor = 1.0 - (dv / current_velocity)
                self.state[2] *= factor  # Reduce x velocity
                self.state[3] *= factor  # Reduce y velocity
        
        # Fill in the 4x4 process noise matrix with reasonable values
        # Position variances
        self._Q_matrix[0, 0] = q_pos * dt**3 / 3.0  # x position variance
        self._Q_matrix[1, 1] = q_pos * dt**3 / 3.0  # y position variance
        
        # Velocity variances
        self._Q_matrix[2, 2] = q_vel * dt          # x velocity variance
        self._Q_matrix[3, 3] = q_vel * dt          # y velocity variance
        
        # Position-velocity covariances
        self._Q_matrix[0, 2] = self._Q_matrix[2, 0] = q_pos * dt**2 / 2.0  # x position-velocity
        self._Q_matrix[1, 3] = self._Q_matrix[3, 1] = q_pos * dt**2 / 2.0  # y position-velocity
        
        # Use a reasonable non-zero velocity in non-stationary state
        if motion_state in ["small_movement", "medium_fast"]:
            current_velocity = np.linalg.norm(self.state[2:4])
            if current_velocity < 0.02:  # Apply minimum velocity only when very slow
                # Apply a very small minimum velocity for movement states
                if self.debug_level >= 1:
                    self.get_logger().info(
                        f"Applying minimum velocity in {motion_state} state "
                        f"(current={current_velocity:.4f}m/s, setting to 0.02m/s)"
                    )
                    
                # Set minimal velocity in current direction or default to x
                if current_velocity > 0.001:
                    # Keep direction, scale magnitude
                    scale = 0.02 / current_velocity
                    self.state[2] *= scale
                    self.state[3] *= scale
                else:
                    # Default to x direction if essentially zero
                    self.state[2] = 0.02  # Set minimal x velocity
                    self.state[3] = 0.0   # Reset y velocity
        
        # Check for consecutive rejected measurements with simplified approach
        if hasattr(self, 'consecutive_rejections_per_sensor'):
            # Count rejections only for key sensors
            total_rejections = sum(count for sensor, count in self.consecutive_rejections_per_sensor.items() 
                                if sensor in ['lidar', 'yolo_3d', 'yolo_2d'] and count > 2)
            
            if total_rejections > 3:  # Only apply when significant rejections
                rejection_scale = min(3.0, 1.0 + (total_rejections / 10.0))  # Reduced scale
                self._Q_matrix *= rejection_scale
                if self.debug_level >= 1:
                    self.get_logger().info(
                        f"Increasing process noise due to {total_rejections} rejections: scale={rejection_scale:.1f}"
                    )
        
        # Predict state using state transition matrix
        self.state = np.dot(self._F_matrix, self.state)
        
        # Predict covariance
        self.covariance = np.dot(np.dot(self._F_matrix, self.covariance), self._F_matrix.T) + self._Q_matrix
        
        # Ensure covariance remains symmetric
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def log_validation_performance(self):
        """
        Log statistics about the validation system's performance.
        Optimized with reduced verbosity for performance.
        """
        if not hasattr(self, 'validation_manager'):
            return
            
        try:
            # Only track the most important sensors to reduce computation
            key_sensors = ['lidar', 'yolo_3d', 'yolo_2d']
            sensor_stats = []
            
            for sensor in key_sensors:
                if sensor not in self.validation_manager.sensors:
                    continue
                    
                fp_rate = self.validation_manager.false_positive_rate.get(sensor, 0.0)
                fn_rate = self.validation_manager.false_negative_rate.get(sensor, 0.0)
                
                if hasattr(self.validation_manager, 'adaptive_thresholds'):
                    threshold = self.validation_manager.adaptive_thresholds.get(sensor, 0.0)
                else:
                    threshold = 0.0
                    
                # Track validation history counts
                history_count = 0
                if sensor in self.validation_manager.validation_history:
                    history_count = len(self.validation_manager.validation_history[sensor])
                    
                # Add to sensor-specific stats
                sensor_stats.append(f"{sensor}: FP={fp_rate:.2f}, FN={fn_rate:.2f}, threshold={threshold:.1f}, samples={history_count}")
                
            # Log the overall performance with simplified calculation
            if sensor_stats:
                self.get_logger().info("Validation performance: Avg FP rate=%.2f, Avg FN rate=%.2f", 
                                sum([float(s.split('FP=')[1].split(',')[0]) for s in sensor_stats])/len(sensor_stats),
                                sum([float(s.split('FN=')[1].split(',')[0]) for s in sensor_stats])/len(sensor_stats))
                
                # Log per-sensor statistics
                for stat in sensor_stats:
                    self.get_logger().info("  %s", stat)
                
            # Log consecutive rejections only if significant
            if hasattr(self, 'consecutive_rejections_per_sensor'):
                reject_sensors = []
                for sensor, count in self.consecutive_rejections_per_sensor.items():
                    if count > 2:  # Only log if more than 2 rejections
                        reject_sensors.append(f"{sensor}={count}")
                        
                if reject_sensors:
                    self.get_logger().info("  Consecutive rejections: %s", ', '.join(reject_sensors))
                    
        except Exception as e:
            if self.debug_level >= 1:
                self.get_logger().error("Error in log_validation_performance: %s", str(e))

    def transform_point(self, point_msg, target_frame, is_2d=False):
        """
        Optimized transform point method for better performance on Pi.
        
        Args:
            point_msg (PointStamped): The point message to transform
            target_frame (str): The target reference frame
            is_2d (bool): Whether the point is a 2D point (z component ignored)
            
        Returns:
            PointStamped: The transformed point or None if transformation failed
        """
        # Quick return if transform not available
        if not self.transform_available:
            if hasattr(self, '_transform_warning_time'):
                # Limit warnings to avoid log spam
                if time.time() - self._transform_warning_time > 5.0:
                    self.get_logger().warn(f"Transform not available - cannot transform point")
                    self._transform_warning_time = time.time()
            else:
                self._transform_warning_time = time.time()
                self.get_logger().warn(f"Transform not available - cannot transform point")
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
            elif point_msg.header.frame_id == 'lidar_frame' and target_frame == self.reference_frame and self.tf_lidar_to_base is not None:
                # Use cached lidar to base transform
                transform = self.tf_lidar_to_base
            else:
                # Fall back to standard transform lookup for non-cached relationships
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    point_msg.header.frame_id,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)  # Reduced timeout
                )
            
            # Transform with simplified handling
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
            
            return transformed
            
        except Exception as e:
            # Log errors less frequently to reduce overhead
            if hasattr(self, '_transform_error_counter'):
                self._transform_error_counter += 1
                # Log every 10th error to reduce verbosity
                if self._transform_error_counter % 10 == 0 and self.debug_level >= 1:
                    self.get_logger().warn(f"Transform error: {str(e)}")
            else:
                self._transform_error_counter = 1
                if self.debug_level >= 1:
                    self.get_logger().warn(f"Transform error: {str(e)}")
            return None
    
    
    def get_adaptive_validation_threshold(self, sensor, innovation=None):
        """
        Balanced validation threshold calculation with reasonable permissiveness
        for measurements showing movement after stationary periods.
        
        Args:
            sensor (str): The sensor name
            innovation (float, optional): The innovation value for context
                
        Returns:
            float: The adaptive validation threshold
        """
        # Get current motion state - store locally to avoid repeated calls
        motion_state = getattr(self, 'motion_state', 'unknown')
        
        # Set balanced base thresholds - higher than original but not extreme
        if sensor == 'lidar':
            base_threshold = 20.0  # Balanced from original 15.0
        elif sensor.endswith('_3d'):
            base_threshold = 25.0  # Balanced from original 20.0
        elif '_est3d' in sensor:
            base_threshold = 35.0  # Balanced from original 30.0
        else:  # 2D sensors
            base_threshold = 30.0  # Balanced from original 25.0
        
        # Apply motion state adjustments with balanced approach
        motion_multipliers = {
            "stationary": 1.2,
            "long_stationary": 1.5,
            "small_movement": 1.8,
            "medium_fast": 2.2,
            "unknown": 2.0
        }
        
        # Apply motion multiplier
        threshold = base_threshold * motion_multipliers.get(motion_state, 2.0)
        
        # Check for state transitions - balanced boost
        in_transition = False
        if hasattr(self, 'prev_motion_state') and hasattr(self, 'motion_state'):
            if self.motion_state != self.prev_motion_state:
                # Track time since transition if needed
                if not hasattr(self, 'last_state_transition_time'):
                    self.last_state_transition_time = time.time()
                
                # Consider a transition active for 3 seconds (reduced from 5)
                if time.time() - self.last_state_transition_time < 3.0:
                    in_transition = True
                    
                    # Apply reasonable boost for transitions OUT OF stationary
                    if self.prev_motion_state in ["stationary", "long_stationary"] and \
                    self.motion_state not in ["stationary", "long_stationary"]:
                        # Moving out of stationary states - significant boost
                        threshold *= 3.0  # Reduced from 10.0
                        
                        if self.debug_level >= 1:
                            self.get_logger().info(
                                f"Applying transition boost for {sensor}: threshold={threshold:.2f} "
                                f"during {self.prev_motion_state}->{self.motion_state} transition"
                            )
                    else:
                        # Other transitions get normal boost
                        threshold *= 1.5  # Reduced from 3.0
                else:
                    # Update transition time
                    self.last_state_transition_time = time.time()
        
        # Check for sensor gap recovery
        if hasattr(self, 'sensor_gap_detection') and sensor in self.sensor_gap_detection:
            # Check if this sensor has recently recovered from a gap
            gap_info = self.sensor_gap_detection[sensor]
            if gap_info.get('recovery_boost_active', False):
                # Apply reasonable boost for recovering sensors
                threshold *= 2.5  # Reduced from 6.0
        
        # Check for consecutive rejections - balanced response
        if hasattr(self, 'consecutive_rejections_per_sensor'):
            rejections = self.consecutive_rejections_per_sensor.get(sensor, 0)
            if rejections > 1:  # Require at least 2 rejections (reduced responsiveness)
                # Calculate reasonable boost factor
                rejection_boost = 1.0 + (min(rejections, 5) * 0.3)  # Reduced from 0.6
                threshold *= rejection_boost
                
                if self.debug_level >= 1 and rejections >= 3:
                    self.get_logger().info(
                        f"Applying rejection-break boost for {sensor}: "
                        f"{rejection_boost:.2f}x → {threshold:.2f}"
                    )
        
        # Check for movement evidence from the sensor position
        if hasattr(self, 'state') and self.initialized:
            # Get the latest measurement
            latest_msg = self.sensor_buffer.get_latest_measurement(sensor)
            if latest_msg is not None:
                # Transform to get comparable position
                if sensor.endswith('_2d') and sensor in self.bbox_data:
                    transformed = self.estimate_3d_from_2d(latest_msg, self.bbox_data[sensor])
                elif not sensor.endswith('_2d'):
                    transformed = self.transform_point(latest_msg, self.reference_frame, False)
                else:
                    transformed = None
                    
                if transformed is not None:
                    # Calculate discrepancy from current state
                    dx = transformed.point.x - self.state[0]
                    dy = transformed.point.y - self.state[1]
                    discrepancy = math.sqrt(dx*dx + dy*dy)
                    
                    # Consider moderate discrepancy as movement evidence
                    if discrepancy > 0.08:  # Increased from 0.02 (8cm instead of 2cm)
                        # Apply balanced movement evidence boost
                        movement_boost = 1.5 + min(2.0, discrepancy / 0.1)  # Reduced from 2.0 + 5.0
                        threshold *= movement_boost
                        
                        if self.debug_level >= 1 and discrepancy > 0.2:
                            self.get_logger().info(
                                f"Applying movement evidence boost for {sensor}: "
                                f"{movement_boost:.2f}x with {discrepancy:.3f}m discrepancy"
                            )
        
        # Apply special override for complex situations with balanced approach
        # If multiple factors suggest we should be more permissive
        complex_situation = sum([in_transition, 
                            hasattr(self, 'sensor_gap_detection') and
                            sensor in self.sensor_gap_detection and
                            self.sensor_gap_detection[sensor].get('recovery_boost_active', False),
                            hasattr(self, 'consecutive_rejections_per_sensor') and
                            self.consecutive_rejections_per_sensor.get(sensor, 0) >= 3]) >= 2  # Increased from 1
        
        if complex_situation:
            # Apply a reasonable override multiplier
            override_boost = 1.5  # Reduced from 3.0
            threshold *= override_boost
            
            if self.debug_level >= 1:
                self.get_logger().info(
                    f"Complex situation detected: applying {override_boost:.1f}x validation override. "
                    f"Threshold={threshold:.2f}"
                )
        
        # Apply safety limits with reasonable caps
        max_safe_threshold = 250.0  # Reduced from 1000.0
        threshold = min(threshold, max_safe_threshold)
        
        return threshold

    def handle_sensor_recovery_balanced(self):
        """
        Balanced sensor recovery handler with improved logging.
        
        Returns:
            bool: True if a recovery was detected in this update
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        recovery_detected = False
        recovered_sensors = []
        
        # Check for recovery only on most important sensors to reduce computation
        priority_sensors = ['lidar', 'yolo_3d', 'yolo_2d']
        
        for sensor in priority_sensors:
            # Skip sensors we haven't seen yet
            if self.sensor_counts.get(sensor, 0) == 0:
                continue
                
            last_time = self.last_detection_time.get(sensor, 0)
            gap_duration = current_time - last_time
            
            # Initialize recovery tracking if needed
            if sensor not in self.sensor_gap_detection:
                self.sensor_gap_detection[sensor] = {
                    'gap_detected': False,
                    'gap_start_time': 0.0,
                    'gap_level': 0.0,
                    'recent_gaps': deque(maxlen=5),
                    'last_recovery_time': 0.0,
                    'recovery_boost_active': False,
                    'recovery_boost_end': 0.0
                }
            
            # Check if sensor was in gap state but now has fresh data
            if self.sensor_gap_detection[sensor]['gap_detected']:
                # Get newest measurement
                msg = self.sensor_buffer.get_latest_measurement(sensor)
                if msg is not None and gap_duration < 0.5:  # Fresh data confirmed
                    # Calculate total gap duration
                    total_gap = current_time - self.sensor_gap_detection[sensor]['gap_start_time']
                    
                    # Log recovery for significant gaps
                    if total_gap > 1.0:
                        self.get_logger().info(
                            f"✨ [T+{elapsed_time:.1f}s][RECOVERY] {sensor} recovered after {total_gap:.1f}s gap"
                        )
                    
                    # Apply balanced covariance adjustment
                    adjustment_factor = min(3.0, 1.5 + total_gap)  # More moderate than before
                    
                    # Modify state uncertainty to allow faster state changes
                    if hasattr(self, 'covariance'):
                        self.covariance[0:2, 0:2] *= adjustment_factor
                        self.covariance[2:4, 2:4] *= adjustment_factor
                        
                        # Update uncertainty metrics
                        self.position_uncertainty = math.sqrt((self.covariance[0, 0] + self.covariance[1, 1]) / 2.0)
                        self.velocity_uncertainty = math.sqrt((self.covariance[2, 2] + self.covariance[3, 3]) / 2.0)
                        
                        if self.debug_level >= 1 and total_gap > 1.0:
                            self.get_logger().info(
                                f"👐 [T+{elapsed_time:.1f}s][RECOVERY] Boost: factor={adjustment_factor:.1f}, "
                                f"uncertainty={self.position_uncertainty:.3f}m"
                            )
                    
                    # Set up recovery boost period
                    self.sensor_gap_detection[sensor]['recovery_boost_active'] = True
                    self.sensor_gap_detection[sensor]['recovery_boost_end'] = current_time + min(3.0, total_gap)
                    self.sensor_gap_detection[sensor]['last_recovery_time'] = current_time
                    
                    # Store gap duration for pattern analysis
                    self.sensor_gap_detection[sensor]['recent_gaps'].append(total_gap)
                    
                    # Clear gap flag
                    self.sensor_gap_detection[sensor]['gap_detected'] = False
                    self.sensor_gap_detection[sensor]['gap_level'] = 0.0
                    
                    # Mark that a recovery was detected this cycle
                    recovery_detected = True
                    recovered_sensors.append(sensor)
            
            # Check for new gaps with balanced threshold (0.5s)
            elif gap_duration > 0.5:
                # New gap detected
                if not self.sensor_gap_detection[sensor]['gap_detected']:
                    self.sensor_gap_detection[sensor]['gap_detected'] = True
                    self.sensor_gap_detection[sensor]['gap_start_time'] = current_time
                    
                    # Calculate gap level based on sensor importance
                    importance = 1.0
                    if sensor == 'lidar':
                        importance = 1.3  # Lidar is more important
                    
                    # Set initial gap level
                    self.sensor_gap_detection[sensor]['gap_level'] = 0.3 * importance
                    
                    # Log only for significant gaps to reduce verbosity
                    if gap_duration > 1.0:
                        # Calculate expected rate based on FPS
                        expected_rate = self.sensor_fps.get(sensor, 1.0)
                        expected_interval = 1.0 / max(0.1, expected_rate)
                        
                        # Calculate multiples of expected rate
                        rate_multiples = gap_duration / max(0.1, expected_interval)
                        
                        self.get_logger().info(
                            f"⚠️ [T+{elapsed_time:.1f}s][GAP] {sensor}: {gap_duration:.1f}s "
                            f"({rate_multiples:.1f}x expected interval)"
                        )
                else:
                    # Update gap level based on duration (up to a maximum of 1.0)
                    duration_factor = min(1.0, gap_duration / 3.0)
                    self.sensor_gap_detection[sensor]['gap_level'] = duration_factor
            
            # Check if recovery boost should still be active
            elif self.sensor_gap_detection[sensor].get('recovery_boost_active', False):
                if current_time > self.sensor_gap_detection[sensor].get('recovery_boost_end', 0.0):
                    # Deactivate recovery boost
                    self.sensor_gap_detection[sensor]['recovery_boost_active'] = False
        
        # Handle multiple sensor recovery
        if len(recovered_sensors) >= 2:
            # Use motion state to inform recovery response
            motion_state = getattr(self, 'motion_state', 'unknown')
            
            # Reset rejection counters for recovered sensors
            for sensor in recovered_sensors:
                if hasattr(self, 'consecutive_rejections_per_sensor'):
                    self.consecutive_rejections_per_sensor[sensor] = 0
                    
            # Apply reasonable uncertainty boost for multi-sensor recovery
            if hasattr(self, 'covariance'):
                multi_recovery_factor = 2.5  # Reduced from 6.0
                self.covariance[0:2, 0:2] *= multi_recovery_factor
                self.covariance[2:4, 2:4] *= multi_recovery_factor * 1.5
                
                # Update uncertainty metrics
                self.position_uncertainty = math.sqrt((self.covariance[0, 0] + self.covariance[1, 1]) / 2.0)
                self.velocity_uncertainty = math.sqrt((self.covariance[2, 2] + self.covariance[3, 3]) / 2.0)
                
                self.get_logger().info(
                    f"✨ [T+{elapsed_time:.1f}s][RECOVERY] Multi-sensor boost: factor={multi_recovery_factor:.1f}, "
                    f"uncertainty={self.position_uncertainty:.3f}m"
                )
        
        return recovery_detected

    def publish_status(self):
        """Publish and log brief status information with improved formatting."""
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
        
        # Create status indicators
        transform_status = "✓" if self.transform_confirmed else ("⟳" if self.transform_available else "✗")
        tracking_status = "✓" if self.tracking_reliable else "✗"
        initialized_status = "✓" if self.initialized else "✗"
        
        # Log status - only once per 5 seconds or if the state changes
        if not hasattr(self, '_last_status_state'):
            self._last_status_state = ""
            self._last_status_time = 0
        
        # Create current state string
        current_state = f"{mode}|3D:{active_3d}|2D:{active_2d}|{self.motion_state}|{self.position_uncertainty:.3f}"
        
        # Check if status changed or time threshold reached
        if (current_state != self._last_status_state or 
                current_time - self._last_status_time > 5.0):  # Reduced frequency
            
            # Update status tracking
            self._last_status_state = current_state
            self._last_status_time = current_time
            
            # Log with improved formatting
            self.get_logger().info(
                f"📊 [T+{uptime:.1f}s][STATUS] Transform={transform_status}, "
                f"Mode={mode}, 3D sensors={active_3d}, 2D sensors={active_2d}, "
                f"Init={initialized_status}, Track={tracking_status}, "
                f"Uncert={self.position_uncertainty:.3f}m, Motion={self.motion_state}"
            )
            
            # Log active sensor information if anything is active
            active_sensors = []
            for sensor in ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']:
                if self.sensor_counts.get(sensor, 0) > 0:
                    delay = current_time - self.last_detection_time.get(sensor, 0)
                    if delay < 5.0:  # Only include recently active sensors
                        fps = self.sensor_fps.get(sensor, 0.0)
                        count = self.sensor_counts.get(sensor, 0)
                        active_sensors.append(f"{sensor}: {count}, {delay:.1f}s ago, {fps:.1f}Hz")
            
            if active_sensors:
                self.get_logger().info(f"📡 [T+{uptime:.1f}s][SENSORS] {' | '.join(active_sensors)}")
                
    def publish_state(self):
        """
        Publish the current state estimate with improved logging.
        """
        # Skip if not active
        if not self.is_activated:
            return
                    
        # Create current position as 3D point from our 4D state
        current_pos = [float(self.state[0]), float(self.state[1]), float(self.basketball_z_height)]
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Store last published position if not yet set
        if not hasattr(self, 'last_published_position'):
            self.last_published_position = current_pos.copy()
            self.last_position_log_time = current_time
        
        # Calculate position change since last published
        pos_change = math.sqrt((current_pos[0] - self.last_published_position[0])**2 + 
                            (current_pos[1] - self.last_published_position[1])**2)
        
        # Calculate distance and direction to the ball
        distance = math.sqrt(current_pos[0]**2 + current_pos[1]**2)
        direction = math.degrees(math.atan2(current_pos[1], current_pos[0]))
        
        # Only log when position changes significantly (>5cm) or time threshold (0.5s) reached
        significant_change = pos_change > 0.05
        time_threshold_reached = current_time - self.last_position_log_time > 0.5
        
        if significant_change or time_threshold_reached:
            # Calculate progress indicator (textual representation)
            if hasattr(self, 'initial_position'):
                progress = "⋯"
                init_dist = math.sqrt(self.initial_position[0]**2 + self.initial_position[1]**2)
                if abs(distance - init_dist) > 0.1:  # >10cm change
                    if distance < init_dist:
                        progress = "⟶⟶ CLOSER"  # Moving closer
                    else:
                        progress = "⟵⟵ FURTHER"  # Moving further
            else:
                # Store initial position for progress tracking
                self.initial_position = current_pos.copy()
                progress = "START"
            
            # Log with improved formatting
            self.get_logger().info(
                f"📍 [T+{elapsed_time:.1f}s][FUSION] dist={distance:.2f}m, dir={direction:.1f}°, "
                f"pos=({current_pos[0]:.2f}, {current_pos[1]:.2f}), "
                f"uncert=±{self.position_uncertainty:.2f}m {progress}"
            )
            
            # Update last published position and time
            self.last_published_position = current_pos.copy()
            self.last_position_log_time = current_time
        
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
        
        # Get statistics from ground filter occasionally (only once per 5 seconds)
        if not hasattr(self, '_ground_stats_timer'):
            self._ground_stats_timer = 0
        
        self._ground_stats_timer += 1
        if self._ground_stats_timer % 100 == 0:  # Reduced frequency
            stats = self.ground_filter.get_statistics()
            if stats:
                self.get_logger().info(
                    f"🚄 [T+{elapsed_time:.1f}s][SPEED] current={stats['current_speed']:.2f}m/s, "
                    f"avg={stats['average_speed']:.2f}m/s, jumps={stats['position_jumps']}"
                )
                
    def detect_motion_state(self):
        """
        Balanced motion state detector with improved logging.
        
        Returns:
            str: One of "stationary", "long_stationary", "small_movement", "medium_fast", or "unknown"
        """
        # If not initialized or insufficient velocity history, return unknown
        if not self.initialized or len(self.velocity_history) < 3:
            return "unknown"
        
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Get the most recent velocity estimates
        recent_velocities = list(self.velocity_history)[-3:]
        
        # Calculate the average magnitude of these velocities
        valid_velocities = [vel for vel in recent_velocities 
                            if isinstance(vel, (list, tuple, np.ndarray)) and len(vel) >= 2]
        if not valid_velocities:
            return "unknown"
            
        # Calculate average velocity
        avg_speed = 0.0
        valid_count = 0
        
        for vel in valid_velocities:
            # For 2D velocities, calculate magnitude in the x-y plane
            speed = math.sqrt(vel[0]**2 + vel[1]**2)
            avg_speed += speed
            valid_count += 1
        
        # Calculate average speed
        if valid_count > 0:
            avg_speed /= valid_count
        
        # Initialize confidence dictionary if needed
        if not hasattr(self, 'motion_state_confidence'):
            self.motion_state_confidence = {
                "stationary": 0.5,
                "long_stationary": 0.0,
                "small_movement": 0.0, 
                "medium_fast": 0.0,
                "unknown": 0.0
            }
        
        # Update confidence values based on current velocity - more balanced approach
        if avg_speed < 0.01:  # Increased from 0.003 to filter noise
            # Evidence for stationary state
            self.motion_state_confidence["stationary"] = min(1.0, self.motion_state_confidence["stationary"] + 0.1)
            self.motion_state_confidence["small_movement"] = max(0.0, self.motion_state_confidence["small_movement"] - 0.1)
            self.motion_state_confidence["medium_fast"] = max(0.0, self.motion_state_confidence["medium_fast"] - 0.1)
        elif avg_speed < 0.15:  # 
            # Evidence for small movement
            self.motion_state_confidence["small_movement"] = min(1.0, self.motion_state_confidence["small_movement"] + 0.15)
            self.motion_state_confidence["stationary"] = max(0.0, self.motion_state_confidence["stationary"] - 0.1)
            self.motion_state_confidence["medium_fast"] = max(0.0, self.motion_state_confidence["medium_fast"] - 0.05)
        else:
            # Evidence for medium/fast movement
            self.motion_state_confidence["medium_fast"] = min(1.0, self.motion_state_confidence["medium_fast"] + 0.2)
            self.motion_state_confidence["small_movement"] = max(0.0, self.motion_state_confidence["small_movement"] - 0.1)
            self.motion_state_confidence["stationary"] = max(0.0, self.motion_state_confidence["stationary"] - 0.15)
        
        # Define thresholds for state classification - more balanced
        stationary_thresh = 0.01    # Increased from 0.003 to filter sensor noise
        small_movement_thresh = 0.15 # Restored to original value
        
        # For movement detection, apply minimal hysteresis to filter noise
        # but still be responsive to actual movement
        if hasattr(self, 'motion_state') and self.motion_state in ["stationary", "long_stationary"]:
            # Apply slight hysteresis when exiting stationary - require slightly more evidence
            if avg_speed > stationary_thresh * 1.5:  # 50% higher threshold to exit stationary
                # Check for consistent movement (reduced oscillation)
                if len(valid_velocities) >= 2 and all(math.sqrt(v[0]**2 + v[1]**2) > stationary_thresh for v in valid_velocities[-2:]):
                    # Reset stationary tracking variables
                    self.stationary_start_time = None
                    if hasattr(self, 'long_stationary_since'):
                        self.long_stationary_since = None
        
        # Apply velocity-based classification
        if avg_speed < stationary_thresh:
            base_motion_state = "stationary"
        elif avg_speed < small_movement_thresh:
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
                # Reduced time threshold for long-term stationary to 5 seconds
                stationary_duration = current_time - self.stationary_start_time
                if stationary_duration > 5.0:
                    motion_state = "long_stationary"
                    
                    # Track how long we've been in long_stationary
                    if not hasattr(self, 'long_stationary_since'):
                        self.long_stationary_since = current_time
                else:
                    motion_state = "stationary"
        else:
            # Handle movement states 
            if hasattr(self, 'motion_state') and self.motion_state == "long_stationary":
                # Respond quickly to movement from long_stationary
                # but still require a bit more evidence
                if avg_speed > stationary_thresh * 1.2:  # 20% higher threshold
                    motion_state = base_motion_state
                    
                    # Reset tracking variables
                    self.stationary_start_time = None
                    self.long_stationary_since = None
                else:
                    motion_state = "long_stationary"  # Stay in long_stationary
            else:
                # Reset stationary timer when moving
                self.stationary_start_time = None
                motion_state = base_motion_state
        
        # Log if motion state changes with duration information
        if motion_state != getattr(self, 'motion_state', 'unknown'):
            self.prev_motion_state = getattr(self, 'motion_state', 'unknown')
            
            # Calculate duration in previous state
            state_duration = 0.0
            if hasattr(self, 'last_state_change_time'):
                state_duration = current_time - self.last_state_change_time
            
            # Update last state change time
            self.last_state_change_time = current_time
            
            # Store current state
            self.motion_state = motion_state
            
            # Reset long_stationary tracking if exiting that state
            if self.prev_motion_state == "long_stationary" and motion_state != "long_stationary":
                self.long_stationary_since = None
            
            # Log with improved formatting including duration
            duration_text = f" after {state_duration:.1f}s" if state_duration > 0.0 else ""
            self.get_logger().info(
                f"🔄 [T+{elapsed_time:.1f}s][STATE] {self.prev_motion_state} → {self.motion_state}{duration_text} "
                f"(v={avg_speed:.3f}m/s)"
            )
        
        return self.motion_state    
    
    def sensor_callback(self, msg, source):
        """Modified sensor callback with improved logging and LiDAR tracking"""
        # Skip if not active yet
        if not self.is_activated:
            return
        
        try:
            # Get current time for timing statistics
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
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
                    self.get_logger().info(f"✨ [T+{elapsed_time:.1f}s][INIT] Attempting fast initialization with LiDAR")
                    transformed = self.transform_point(msg, self.reference_frame, False)
                    if transformed:
                        # Try immediate initialization with first LiDAR measurement
                        self.fast_initialize_with_first_measurement(transformed, source)
                
                # If not LiDAR and still not initialized, try with 2D measurement
                elif not self.initialized and source == 'yolo_2d' and 'yolo_2d' in self.bbox_data:
                    self.get_logger().info(f"✨ [T+{elapsed_time:.1f}s][INIT] Attempting fast initialization with 2D sensor")
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
            
            # For LiDAR data, log cartesian distance and direction every 3rd message
            if source == 'lidar':
                # Initialize LiDAR counter if needed
                if not hasattr(self, '_lidar_log_counter'):
                    self._lidar_log_counter = 0
                
                self._lidar_log_counter += 1
                
                # Get position information
                transformed = self.transform_point(msg, self.reference_frame, False)
                if transformed and self._lidar_log_counter % 3 == 0:  # Log every 3rd entry
                    # Calculate cartesian distance and direction
                    distance = math.sqrt(transformed.point.x**2 + transformed.point.y**2)
                    direction = math.degrees(math.atan2(transformed.point.y, transformed.point.x))
                    
                    self.get_logger().info(
                        f"📡 [T+{elapsed_time:.1f}s][LIDAR] #{self._lidar_log_counter}: dist={distance:.2f}m, "
                        f"dir={direction:.1f}°, pos=({transformed.point.x:.2f}, {transformed.point.y:.2f}, {transformed.point.z:.2f})"
                    )
                
        except Exception as e:
            self.log_error(f"❌ [T+{elapsed_time:.1f}s][{source.upper()}] Error: {str(e)}")


    def update_state(self, measurements):
        """
        Balanced update_state method with improved logging.
        
        Args:
            measurements (dict): Dictionary of sensor measurements
                    
        Returns:
            bool: True if state was successfully updated
        """
        # Store successful update flag
        successful_update = False
        
        # Get current time and elapsed time
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Get current motion state once
        motion_state = getattr(self, 'motion_state', 'unknown')
        
        # Apply moderate validation boost for stationary states
        validation_boost = 1.0
        if motion_state in ["stationary", "long_stationary"]:
            validation_boost = 2.0
        
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
            
            # Check hard position limits with balanced values
            max_coord = 7.5
            if abs(transformed.point.x) > max_coord or abs(transformed.point.y) > max_coord:
                self.get_logger().warn(
                    f"⚠️ [T+{elapsed_time:.1f}s][VALIDATION] Rejecting {sensor}: out of bounds "
                    f"({transformed.point.x:.2f}, {transformed.point.y:.2f}), limits=±{max_coord}m"
                )
                self.consecutive_rejections_per_sensor[sensor] = self.consecutive_rejections_per_sensor.get(sensor, 0) + 1
                continue
            
            # Setup measurement and matrices - optimized to reduce operations
            if sensor.endswith('_2d'):
                z = np.array([transformed.point.x, transformed.point.y], dtype=np.float32)
                H = np.zeros((2, 4), dtype=np.float32)
                H[0, 0] = 1.0  # Extract x position
                H[1, 1] = 1.0  # Extract y position
                
                # Moderate noise for 2D sensors
                if sensor == 'hsv_2d':
                    R = np.diag([self.measurement_noise_hsv_2d * 0.7, self.measurement_noise_hsv_2d * 0.7]).astype(np.float32)
                elif sensor == 'yolo_2d':
                    R = np.diag([self.measurement_noise_yolo_2d * 0.7, self.measurement_noise_yolo_2d * 0.7]).astype(np.float32)
                else:
                    R = np.diag([30.0, 30.0]).astype(np.float32)
            elif is_2d_derived:
                # Handle 2D-derived 3D estimates
                z = np.array([transformed.point.x, transformed.point.y], dtype=np.float32)
                H = np.zeros((2, 4), dtype=np.float32)
                H[0, 0] = 1.0
                H[1, 1] = 1.0
                
                # Moderate noise for 2D-derived 3D
                if 'yolo' in sensor:
                    R = np.diag([self.measurement_noise_yolo_2d_est3d * 0.7, self.measurement_noise_yolo_2d_est3d * 0.7]).astype(np.float32)
                elif 'hsv' in sensor:
                    R = np.diag([self.measurement_noise_hsv_2d_est3d * 0.7, self.measurement_noise_hsv_2d_est3d * 0.7]).astype(np.float32)
                else:
                    R = np.diag([0.1, 0.1]).astype(np.float32)
            else:
                z = np.array([transformed.point.x, transformed.point.y], dtype=np.float32)
                H = np.zeros((2, 4), dtype=np.float32)
                H[0, 0] = 1.0
                H[1, 1] = 1.0
                
                # Moderate noise for 3D sensors
                if sensor == 'lidar':
                    R = np.diag([self.measurement_noise_lidar * 0.7, self.measurement_noise_lidar * 0.7]).astype(np.float32)
                elif sensor == 'hsv_3d':
                    R = np.diag([self.measurement_noise_hsv_3d * 0.7, self.measurement_noise_hsv_3d * 0.7]).astype(np.float32)
                elif sensor == 'yolo_3d':
                    R = np.diag([self.measurement_noise_yolo_3d * 0.7, self.measurement_noise_yolo_3d * 0.7]).astype(np.float32)
                else:
                    R = np.diag([0.07, 0.07]).astype(np.float32)
            
            # Innovation (measurement residual)
            y = z - np.dot(H, self.state)
            
            # Innovation covariance
            S = np.dot(np.dot(H, self.covariance), H.T) + R
            
            # Calculate Mahalanobis distance for validation
            mahalanobis_dist = 0.0
            try:
                S_inv = np.linalg.inv(S)
                mahalanobis_dist = np.sqrt(np.dot(np.dot(y.T, S_inv), y))
                
                # Apply balanced approach to movement detection
                should_force_accept = False
                
                # Calculate position discrepancy for movement detection
                pos_diff = np.linalg.norm(z - np.array([self.state[0], self.state[1]]))
                
                # Check for significant movement evidence with balanced threshold
                if motion_state in ["stationary", "long_stationary"] and pos_diff > 0.08:
                    # Significant movement from stationary states gets acceptance boost
                    should_force_accept = True
                    self.get_logger().info(
                        f"✅ [T+{elapsed_time:.1f}s][VALIDATION] Accepting {sensor}: {pos_diff:.3f}m discrepancy in {motion_state}"
                    )
                
                # Get threshold with validation boost applied
                threshold = self.get_adaptive_validation_threshold(sensor, mahalanobis_dist) * validation_boost
                
                # Skip measurement if it fails validation and isn't force-accepted
                if mahalanobis_dist > threshold and not should_force_accept:
                    # Log rejection with more details for significant rejections only
                    if mahalanobis_dist > threshold * 1.5:
                        # Calculate position discrepancy for context
                        if is_2d_sensor:
                            discrepancy_text = "N/A (2D sensor)"
                        else:
                            # For 3D estimated or direct
                            z_pos = np.array([transformed.point.x, transformed.point.y])
                            state_pos = np.array([self.state[0], self.state[1]])
                            pos_diff = np.linalg.norm(z_pos - state_pos)
                            discrepancy_text = f"{pos_diff:.3f}m"
                        
                        self.get_logger().info(
                            f"❌ [T+{elapsed_time:.1f}s][VALIDATION] Rejecting {sensor}: "
                            f"innovation {mahalanobis_dist:.2f} > threshold {threshold:.2f}, diff={discrepancy_text}"
                        )
                    
                    # Increment rejection counter
                    self.consecutive_rejections_per_sensor[sensor] = self.consecutive_rejections_per_sensor.get(sensor, 0) + 1
                    self.consecutive_updates = 0
                    continue
                
                # Reset rejection counter
                self.consecutive_rejections_per_sensor[sensor] = 0
                
                # Update consecutive updates counter
                self.consecutive_updates += 1
                
            except np.linalg.LinAlgError:
                self.get_logger().warn(f"⚠️ [T+{elapsed_time:.1f}s][MATRIX] Inversion failed for {sensor}")
                continue
                    
            # Kalman gain
            try:
                K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))
                
                # Store current state for comparison
                old_state = self.state.copy()
                
                # Calculate position discrepancy for blending decisions
                pos_discrepancy = np.linalg.norm(z - np.array([self.state[0], self.state[1]]))
                
                # Apply balanced blending approach based on conditions
                if motion_state in ["stationary", "long_stationary"] and pos_discrepancy > 0.08:
                    # Balanced approach for stationary->movement transition
                    blend_factor = 0.8
                    
                    # Apply direct blend for position with simplified approach
                    updated_state = self.state.copy()
                    updated_state += np.dot(K, y)  # Standard Kalman update
                    
                    # Apply additional position blend
                    updated_state[0] = (1 - blend_factor) * self.state[0] + blend_factor * z[0]
                    updated_state[1] = (1 - blend_factor) * self.state[1] + blend_factor * z[1]
                    
                    # Update the state
                    self.state = updated_state
                    
                    self.get_logger().info(
                        f"🔄 [T+{elapsed_time:.1f}s][UPDATE] Enhanced {sensor} update: {pos_discrepancy:.3f}m discrepancy, "
                        f"blend={blend_factor:.2f}"
                    )
                
                # Special case for moderate discrepancies > 10cm
                elif pos_discrepancy > 0.1:
                    # Calculate balanced blend factor
                    blend_factor = min(0.75, 0.6 + (pos_discrepancy / 1.0))
                    
                    # More trust for lidar
                    if sensor == 'lidar':
                        blend_factor = min(0.85, blend_factor + 0.1)
                    
                    # Apply standard Kalman update with position bias
                    updated_state = self.state.copy()
                    updated_state += np.dot(K, y)  # Standard Kalman update
                    
                    # Apply additional position blend - only for significant discrepancies
                    updated_state[0] = (1 - blend_factor) * updated_state[0] + blend_factor * z[0]
                    updated_state[1] = (1 - blend_factor) * updated_state[1] + blend_factor * z[1]
                    
                    # Update the state
                    self.state = updated_state
                    
                    if pos_discrepancy > 0.2:
                        self.get_logger().info(
                            f"🔄 [T+{elapsed_time:.1f}s][UPDATE] Balanced {sensor} update: {pos_discrepancy:.3f}m discrepancy, "
                            f"blend={blend_factor:.2f}"
                        )
                
                # Default case: Standard Kalman update
                else:
                    self.state = self.state + np.dot(K, y)
                
                # Update covariance using simplified Joseph form for numerical stability
                I_KH = np.eye(4, dtype=np.float32) - np.dot(K, H)
                self.covariance = np.dot(np.dot(I_KH, self.covariance), I_KH.T) + np.dot(np.dot(K, R), K.T)
                
                # Ensure covariance remains symmetric with single operation
                self.covariance = 0.5 * (self.covariance + self.covariance.T)
                
                # Check for significant position changes with moderate threshold
                pos_change = np.linalg.norm(self.state[0:2] - old_state[0:2])
                if pos_change > 0.05 and motion_state in ["stationary", "long_stationary"]:
                    self.get_logger().info(
                        f"📊 [T+{elapsed_time:.1f}s][CHANGE] Position changed during {motion_state}: "
                        f"{pos_change:.3f}m from {sensor} data"
                    )
                
                # Mark that we had a successful update
                successful_update = True
                
            except np.linalg.LinAlgError:
                self.get_logger().warn(f"⚠️ [T+{elapsed_time:.1f}s][MATRIX] Kalman update failed for {sensor}")
                continue
        
        return successful_update

    def adjust_covariance_for_gaps(self):
        """
        Balanced covariance adjustment with improved logging.
        """
        # Need sensors to be initialized first
        if not hasattr(self, 'sensor_gap_detection') or not hasattr(self, 'consecutive_rejections_per_sensor'):
            return

        current_time = time.time()
        elapsed_time = current_time - self.start_time

        # Count active sensors with simplified approach
        active_sensors = 0
        avg_gap_level = 0.0
        rejected_sensor_count = 0
        sensor_count = 0
        
        # Get current motion state - store locally to avoid repeated calls
        motion_state = getattr(self, 'motion_state', 'unknown')

        # Check only the most important sensors
        key_sensors = ['lidar', 'yolo_3d', 'yolo_2d']
        for sensor in key_sensors:
            # Skip sensors we've never seen
            if self.sensor_counts.get(sensor, 0) == 0:
                continue

            sensor_count += 1
            last_time = self.last_detection_time.get(sensor, 0)
            gap_duration = current_time - last_time
            
            # Use simplified gap detection
            if gap_duration > 0.5:
                avg_gap_level += min(1.0, gap_duration / 2.0)
            elif self.consecutive_rejections_per_sensor.get(sensor, 0) >= 2:
                rejected_sensor_count += 1
                avg_gap_level += 0.5
            else:
                active_sensors += 1

        # Calculate average gap level 
        if sensor_count > 0:
            avg_gap_level /= sensor_count
        
        # Set base growth rate
        if active_sensors >= 2:
            growth_rate = 1.1 + (avg_gap_level * 0.2)
        elif active_sensors == 1:
            growth_rate = 1.2 + (avg_gap_level * 0.3)
        else:  # No effectively active sensors
            growth_rate = 1.3 + (avg_gap_level * 0.5)

        # Apply balanced growth rate for motion states
        if motion_state in ["stationary", "long_stationary"]:
            if motion_state == "long_stationary":
                growth_rate *= 1.3
            else:
                growth_rate *= 1.2
        
        # Apply growth to covariance matrix
        self.covariance[0:2, 0:2] *= growth_rate  # Position uncertainty
        self.covariance[2:4, 2:4] *= growth_rate * 1.2  # Velocity uncertainty
        
        # Ensure symmetry and minimum values with fewer operations
        for i in range(4):
            self.covariance[i, i] = max(0.01, self.covariance[i, i])
                
        # Apply motion-specific uncertainty caps
        if motion_state == "stationary":
            max_pos_uncertainty = 0.8
            max_vel_uncertainty = 1.5
        elif motion_state == "long_stationary":
            max_pos_uncertainty = 0.7
            max_vel_uncertainty = 1.2
        elif motion_state == "small_movement":
            max_pos_uncertainty = 1.2
            max_vel_uncertainty = 2.0
        elif motion_state == "medium_fast":
            max_pos_uncertainty = 1.5
            max_vel_uncertainty = 2.5
        else:  # unknown
            max_pos_uncertainty = 1.5
            max_vel_uncertainty = 2.2

        # Calculate uncertainties once for efficiency
        current_pos_uncertainty = math.sqrt(max(0.0, (self.covariance[0, 0] + self.covariance[1, 1]) / 2.0))
        current_vel_uncertainty = math.sqrt(max(0.0, (self.covariance[2, 2] + self.covariance[3, 3]) / 2.0))
        
        # Apply minimums for stationary states
        if motion_state in ["stationary", "long_stationary"]:
            min_pos_uncertainty = 0.1
            min_vel_uncertainty = 0.2
            
            if current_pos_uncertainty < min_pos_uncertainty:
                scale = (min_pos_uncertainty / current_pos_uncertainty) ** 2
                self.covariance[0:2, 0:2] *= scale
                    
            if current_vel_uncertainty < min_vel_uncertainty:
                scale = (min_vel_uncertainty / current_vel_uncertainty) ** 2
                self.covariance[2:4, 2:4] *= scale
        
        # Check against maximum caps to prevent extreme uncertainty
        if current_pos_uncertainty > max_pos_uncertainty:
            scale = (max_pos_uncertainty / current_pos_uncertainty) ** 2
            self.covariance[0:2, 0:2] *= scale
        if current_vel_uncertainty > max_vel_uncertainty:
            scale = (max_vel_uncertainty / current_vel_uncertainty) ** 2
            self.covariance[2:4, 2:4] *= scale

        # Only log significant growth events
        if growth_rate > 1.3 and avg_gap_level > 0.5:
            self.get_logger().info(
                f"📈 [T+{elapsed_time:.1f}s][COVAR] Growth rate={growth_rate:.2f}, gaps={avg_gap_level:.1f}, "
                f"rejected={rejected_sensor_count}, uncertainty={current_pos_uncertainty:.3f}m"
            )

    def filter_update(self):
        """
        Optimized filter update with improved logging.
        """
        # Skip if not active or not initialized
        if not self.is_activated or not self.initialized:
            return
        
        try:
            # Get current time
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Calculate time step since last update
            if self.last_update_time is None:
                dt = 0.1  # Default initial time step
            else:
                dt = current_time - self.last_update_time
                # Limit dt to reasonable values
                dt = min(dt, 0.5)  # Cap at 0.5 seconds
            
            # Apply balanced covariance adjustment
            self.adjust_covariance_for_gaps()
            
            # Predict state
            self.predict_state(dt)
            
            # Find synchronized measurements with minimal processing
            measurements = self.sensor_buffer.find_synchronized_measurements(min_sensors=1)
            
            # Update state with measurements if available
            successful_update = False
            if measurements:
                # Log what sensors were synchronized - but only if multiple sensors
                if len(measurements) > 1 and self.debug_level >= 2:
                    sensor_list = ', '.join(measurements.keys())
                    self.get_logger().debug(f"🔄 [T+{elapsed_time:.1f}s][SYNC] Synchronized sensors: {sensor_list}")
                
                successful_update = self.update_state(measurements)
            
            # Update last update time
            self.last_update_time = current_time
            
            # Update uncertainty metrics using trace calculation
            self.position_uncertainty = math.sqrt(max(0.0, (self.covariance[0, 0] + self.covariance[1, 1]) / 2.0))
            self.velocity_uncertainty = math.sqrt(max(0.0, (self.covariance[2, 2] + self.covariance[3, 3]) / 2.0))
            
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
            
            # Update tracking status
            self.update_tracking_status()
            
            # Publish state
            self.publish_state()
            
            # Publish uncertainty (less frequent to reduce overhead)
            if hasattr(self, 'status_count'):
                self.status_count = (self.status_count + 1) % 5
                if self.status_count == 0:
                    self.publish_uncertainty()
            else:
                self.status_count = 0
                self.publish_uncertainty()
            
            # Apply flat ground constraints
            self.apply_flat_ground_constraints()
            
            # Handle sensor recovery with balanced approach
            recovery_detected = self.handle_sensor_recovery_balanced()
            
            # Update motion state on successful updates
            if successful_update:
                self.detect_motion_state()
            
            # Log performance metrics periodically (once every 10 seconds)
            if not hasattr(self, '_perf_log_time'):
                self._perf_log_time = 0
                
            if current_time - self._perf_log_time > 10.0:
                self._perf_log_time = current_time
                
                # Get system stats if available
                if HAS_PSUTIL:
                    try:
                        cpu_percent = psutil.cpu_percent(interval=0.1)
                        mem = psutil.virtual_memory()
                        
                        self.get_logger().info(
                            f"🖥️ [T+{elapsed_time:.1f}s][SYSTEM] CPU: {cpu_percent:.1f}%, Memory: {mem.percent:.1f}%, "
                            f"Active sensors: {sum(1 for s, t in self.last_detection_time.items() if current_time - t < 1.0)}"
                        )
                    except:
                        pass
            
        except Exception as e:
            self.get_logger().error(f"❌ [T+{elapsed_time:.1f}s][ERROR] Filter update: {str(e)}")
            if self.debug_level >= 2:
                import traceback
                self.get_logger().error(traceback.format_exc())

    def update_tracking_status(self):
        """
        Updated method to determine tracking status with improved logging
        and faster response to movement after gaps.
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
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
                
                # Log with improved formatting
                self.get_logger().info(
                    f"⏳ [T+{elapsed_time:.1f}s][GAP] Tolerance window activated: {tolerance:.1f}s, "
                    f"state={motion_state}, uncertainty={pos_uncertainty:.3f}m"
                )
        
        # Check if we're within the tolerance window during a sensor gap
        within_gap_tolerance = False
        if self.sensor_gap_window['active']:
            # If any sensors have recovered, exit gap mode
            if not all_sensors_gap:
                self.sensor_gap_window['active'] = False
                if self.debug_level >= 2:
                    self.get_logger().debug(f"🔄 [T+{elapsed_time:.1f}s][GAP] Tolerance window ended: sensors recovered")
            else:
                # Check if we're still within the tolerance window
                gap_duration = current_time - self.sensor_gap_window['start_time']
                within_gap_tolerance = gap_duration < self.sensor_gap_window['tolerance_seconds']
                
                # If gap exceeds tolerance window, exit gap mode
                if not within_gap_tolerance:
                    self.sensor_gap_window['active'] = False
                    self.get_logger().info(
                        f"⚠️ [T+{elapsed_time:.1f}s][GAP] Tolerance expired after {gap_duration:.2f}s > "
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
                self.get_logger().info(
                    f"🔍 [T+{elapsed_time:.1f}s][TRACKING] Expanded thresholds during recovery: "
                    f"pos={pos_threshold:.2f}, vel={vel_threshold:.2f}"
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
                    f"🕰️ [T+{elapsed_time:.1f}s][GAP] Maintaining tracking: uncertainty={pos_uncertainty:.3f}m < {pos_threshold:.3f}m"
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
                        f"✅ [T+{elapsed_time:.1f}s][TRACKING] Started: uncertainty={pos_uncertainty:.3f}m, "
                        f"sensors={active_3d_sensors}(3D)+{active_2d_sensors}(2D)"
                    )
            elif self.tracking_reliable and true_count == 0:  # All 3 must be unreliable (was 1/5)
                # Special case for sensor gaps with very low uncertainty
                if all_sensors_gap and pos_uncertainty < (self.position_uncertainty_threshold * 2.0) and motion_state == "long_stationary":
                    # Temporary loss - maintain tracking during brief gaps for long-stationary objects with low uncertainty
                    if self.debug_level >= 1:
                        self.get_logger().info(
                            f"🔄 [T+{elapsed_time:.1f}s][TRACKING] Maintaining despite gap: uncertainty={pos_uncertainty:.3f}m"
                        )
                else:
                    self.tracking_reliable = False
                    if self.last_tracking_state != self.tracking_reliable:
                        self.get_logger().info(
                            f"❌ [T+{elapsed_time:.1f}s][TRACKING] Lost: uncertainty={pos_uncertainty:.3f}m, "
                            f"sensors={active_3d_sensors}(3D)+{active_2d_sensors}(2D)"
                        )
        else:
            # For moving objects, be even more responsive:
            # 1/3 reliable to start, 3/3 unreliable to stop (much harder to lose tracking)
            if not self.tracking_reliable and true_count >= 1:  # Changed from 2/5 to 1/3
                self.tracking_reliable = True
                if self.last_tracking_state != self.tracking_reliable:
                    self.get_logger().info(
                        f"✅ [T+{elapsed_time:.1f}s][TRACKING] Started: uncertainty={pos_uncertainty:.3f}m, "
                        f"sensors={active_3d_sensors}(3D)+{active_2d_sensors}(2D)"
                    )
            elif self.tracking_reliable and true_count == 0:  # Changed from 2/5 to 0/3 (All 3 must be unreliable)
                self.tracking_reliable = False
                if self.last_tracking_state != self.tracking_reliable:
                    self.get_logger().info(
                        f"❌ [T+{elapsed_time:.1f}s][TRACKING] Lost: uncertainty={pos_uncertainty:.3f}m, "
                        f"sensors={active_3d_sensors}(3D)+{active_2d_sensors}(2D)"
                    )
                    
        self.last_tracking_state = self.tracking_reliable
        return self.tracking_reliable

    def publish_diagnostics(self):
        """Optimized diagnostics publishing with performance metrics."""
        # Get current time
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate periodic diagnostics (once per 10 seconds)
        if not hasattr(self, 'last_full_diagnostics_time'):
            self.last_full_diagnostics_time = 0
        
        do_full_diagnostics = current_time - self.last_full_diagnostics_time > 10.0
        
        if do_full_diagnostics:
            self.last_full_diagnostics_time = current_time
            
            # Get system resource metrics if psutil is available
            cpu_usage = "N/A"
            memory_usage = "N/A"
            if HAS_PSUTIL:
                try:
                    cpu_usage = f"{psutil.cpu_percent(interval=0.1):.1f}%"
                    memory = psutil.virtual_memory()
                    memory_usage = f"{memory.percent:.1f}% ({memory.used / (1024*1024):.1f}MB)"
                except:
                    pass
            
            # Log system performance
            self.get_logger().info(
                f"🔧 [T+{elapsed_time:.1f}s][SYSTEM] CPU: {cpu_usage}, Memory: {memory_usage}, "
                f"Position uncertainty: {self.position_uncertainty:.3f}m"
            )
            
            # Log validation performance, but with reduced frequency
            if hasattr(self, 'validation_manager'):
                # Only track the most important sensors
                key_sensors = ['lidar', 'yolo_3d', 'yolo_2d']
                rejection_counts = []
                
                for sensor in key_sensors:
                    if sensor in self.consecutive_rejections_per_sensor and self.consecutive_rejections_per_sensor[sensor] > 0:
                        rejection_counts.append(f"{sensor}={self.consecutive_rejections_per_sensor[sensor]}")
                
                if rejection_counts:
                    self.get_logger().info(
                        f"⚠️ [T+{elapsed_time:.1f}s][VALIDATION] Rejections: {', '.join(rejection_counts)}"
                    )
        
        # Create a simplified diagnostics dictionary with only essential info
        diag = {
            'elapsed_time': elapsed_time,
            'position_uncertainty': self.position_uncertainty,
            'velocity_uncertainty': self.velocity_uncertainty,
            'motion_state': self.motion_state,
            'sensor_health': {
                'lidar': current_time - self.last_detection_time.get('lidar', 0) < 1.0,
                'yolo_3d': current_time - self.last_detection_time.get('yolo_3d', 0) < 1.0,
                'yolo_2d': current_time - self.last_detection_time.get('yolo_2d', 0) < 1.0
            },
            'transform_available': self.transform_confirmed
        }
        
        # Publish diagnostics 
        msg = String()
        msg.data = json.dumps(diag)
        self.diagnostics_pub.publish(msg)    

    def estimate_3d_from_2d(self, detection_msg, bbox_data):
        """
        Optimized 3D position estimation from 2D detection with improved logging.
        
        Args:
            detection_msg (PointStamped): The 2D detection message
            bbox_data (dict): Bounding box data with width, height, and timestamp
            
        Returns:
            PointStamped: Estimated 3D position or None if estimation fails
        """
        try:
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Use direct motion state access to avoid recursion
            motion_state = getattr(self, 'motion_state', 'unknown')

            # Set age threshold based on motion state
            if motion_state == "stationary":
                max_bbox_age = 3.0  # Balanced threshold for stationary
            elif motion_state == "long_stationary":
                max_bbox_age = 4.0  # Still generous for long-term stationary
            elif motion_state == "small_movement":
                max_bbox_age = 2.0  # Reduced for movement states
            else:  # medium_fast or unknown
                max_bbox_age = 1.5  # Stricter for fast movement
            
            # Get the actual age
            bbox_age = current_time - bbox_data.get('timestamp', 0)

            # Check if bbox data is recent enough
            if bbox_age > max_bbox_age:
                # Reduce logging frequency for old bbox warnings
                if hasattr(self, '_bbox_age_warning_count'):
                    self._bbox_age_warning_count += 1
                    if self._bbox_age_warning_count % 5 == 0:  # Log every 5th warning
                        self.get_logger().warn(f"⚠️ [T+{elapsed_time:.1f}s][BBOX] Too old: {bbox_age:.2f}s > {max_bbox_age:.1f}s ({motion_state})")
                else:
                    self._bbox_age_warning_count = 1
                    self.get_logger().warn(f"⚠️ [T+{elapsed_time:.1f}s][BBOX] Too old: {bbox_age:.2f}s > {max_bbox_age:.1f}s ({motion_state})")
                return None

            # For slightly outdated bbox data, apply a moderate confidence penalty
            age_penalty = 1.0
            if bbox_age > (max_bbox_age * 0.6):  # Reduced threshold for penalty start
                # Linear penalty up to 15%
                age_penalty = 1.0 + (bbox_age / max_bbox_age) * 0.15
                
            # Get bounding box dimensions
            bbox_width = bbox_data.get('width', 0)
            bbox_height = bbox_data.get('height', 0)
            
            if bbox_width <= 0 or bbox_height <= 0:
                self.get_logger().warn(f"⚠️ [T+{elapsed_time:.1f}s][BBOX] Invalid dimensions: {bbox_width}x{bbox_height}")
                return None
                
            # Known basketball diameter in meters
            basketball_diameter_meters = self.basketball_radius * 2
            
            # Calculate distance based on apparent size vs actual size
            focal_length_pixels = 345.58  # Calibrated focal length for camera
            estimated_distance = (basketball_diameter_meters * focal_length_pixels) / bbox_width
            
            # Apply age penalty to increase distance estimate for older data
            estimated_distance *= age_penalty
            
            # Get camera to reference frame transform
            # Use cached transform when available to reduce overhead
            if hasattr(self, 'tf_camera_to_base') and self.tf_camera_to_base is not None:
                transform = self.tf_camera_to_base
            else:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.reference_frame,
                        'ascamera_color_0',  # Frame of the YOLO camera
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.1)  # Reduced timeout
                    )
                except Exception as te:
                    if self.debug_level >= 1:
                        self.get_logger().error(f"❌ [T+{elapsed_time:.1f}s][TRANSFORM] Lookup failed: {str(te)}")
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
            
            # Convert pixel offsets to direction vector using focal length
            camera_dir_z = focal_length_pixels  # Z is forward in camera frame
            camera_dir_x = offset_x             # X is right in camera frame
            camera_dir_y = offset_y             # Y is down in camera frame
            
            # Normalize the direction vector with single sqrt call for efficiency
            dir_magnitude = math.sqrt(camera_dir_x**2 + camera_dir_y**2 + camera_dir_z**2)
            if dir_magnitude > 0:
                inv_magnitude = 1.0 / dir_magnitude  # Calculate once
                camera_dir_x *= inv_magnitude
                camera_dir_y *= inv_magnitude
                camera_dir_z *= inv_magnitude
            
            # Extract rotation quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            
            # Convert to rotation matrix elements
            # Optimized calculations to reduce operations
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
            
            # Normalize direction vector with single sqrt for efficiency
            dir_magnitude = math.sqrt(ref_dir_x**2 + ref_dir_y**2 + ref_dir_z**2)
            if dir_magnitude > 0:
                inv_magnitude = 1.0 / dir_magnitude  # Calculate once
                ref_dir_x *= inv_magnitude
                ref_dir_y *= inv_magnitude
                ref_dir_z *= inv_magnitude
            
            # Calculate estimated position in reference frame
            est_x = camera_pos_x + estimated_distance * ref_dir_x
            est_y = camera_pos_y + estimated_distance * ref_dir_y
            est_z = self.basketball_z_height  # Always at basketball height above ground
            
            # Calculate cartesian distance and direction
            cartesian_distance = math.sqrt(est_x**2 + est_y**2)
            direction_deg = math.degrees(math.atan2(est_y, est_x))
            
            # Create and return a new 3D point message in the reference frame
            estimated_point = PointStamped()
            estimated_point.header.stamp = detection_msg.header.stamp
            estimated_point.header.frame_id = self.reference_frame
            estimated_point.point.x = est_x
            estimated_point.point.y = est_y
            estimated_point.point.z = est_z
            
            # Log estimation details using counter to reduce frequency
            if not hasattr(self, '_3d_estimation_counter'):
                self._3d_estimation_counter = 0
            
            self._3d_estimation_counter += 1
            
            if self._3d_estimation_counter % 3 == 0:  # Print every 3rd entry
                self.get_logger().info(
                    f"📏 [T+{elapsed_time:.1f}s][YOLO3D] dist={cartesian_distance:.2f}m, dir={direction_deg:.1f}°, "
                    f"pos=({est_x:.2f}, {est_y:.2f}), bbox={bbox_width:.1f}x{bbox_height:.1f}"
                )
                
            return estimated_point
                
        except Exception as e:
            # Log errors less frequently
            if hasattr(self, '_3d_estimation_error_count'):
                self._3d_estimation_error_count += 1
                if self._3d_estimation_error_count % 10 == 0 and self.debug_level >= 1:
                    self.get_logger().warn(f"⚠️ [T+{time.time()-self.start_time:.1f}s][YOLO3D] Error: {str(e)}")
            else:
                self._3d_estimation_error_count = 1
                if self.debug_level >= 1:
                    self.get_logger().warn(f"⚠️ [T+{time.time()-self.start_time:.1f}s][YOLO3D] Error: {str(e)}")
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

    def publish_uncertainty(self):
        """Publish the position uncertainty."""
        msg = Float32()
        msg.data = self.position_uncertainty
        self.uncertainty_pub.publish(msg)
    
    
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