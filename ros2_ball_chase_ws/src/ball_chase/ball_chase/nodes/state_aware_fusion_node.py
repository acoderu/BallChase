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
import sys
import copy
import functools
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
        
        # ENHANCEMENT 8: Adjust thresholds based on motion state
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
        
        self.get_logger().info("======xxxxx Enhanced Fusion Lifecycle Node Starting ======")
        
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
        
        # ENHANCEMENT 1: Initialize motion state tracking
        self.init_motion_state_tracking()
        
        # ENHANCEMENT 4: Initialize flat ground tracking
        self.flat_ground_detected = False
        self.flat_ground_count = 0
        
        # ENHANCEMENT 5: Initialize sensor recovery tracking
        self.sensor_gap_detection = {}
        
        # ENHANCEMENT 6: Initialize reliability buffer
        self.reliability_buffer = deque([False] * 3, maxlen=5)
        self.last_tracking_state = False
        
        # Sensor gap tolerance window tracking
        self.sensor_gap_window = {
            'active': False,
            'start_time': 0.0,
            'previous_reliability': False,
            'tolerance_seconds': 2.0,  # Increased from 0.8 to 2.0 seconds as default
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
            
            self.sync_quality_metrics = {
                'success_rate': 0.0,
                'avg_time_diff': 0.0,
                'sensor_availability': {},
                'sync_counts': 0,
                'attempt_counts': 0
            }
            
            # ENHANCEMENT 1: Initialize motion state tracking
            self.init_motion_state_tracking()
            
            # ENHANCEMENT 4: Initialize flat ground tracking
            self.flat_ground_detected = False
            self.flat_ground_count = 0
            
            # ENHANCEMENT 5: Initialize sensor recovery tracking
            self.sensor_gap_detection = {}
            
            # ENHANCEMENT 6: Initialize reliability buffer
            self.reliability_buffer = deque([False] * 3, maxlen=5)
            self.last_tracking_state = False
            
            # Motion state protection with reduced long_stationary threshold (5.0 -> 2.0 seconds)
            self.motion_state_protection = {
                'long_stationary_confirmed_time': 0.0,
                'long_stationary_established': False,
                'consecutive_stationary_after_long': 0,
                'min_time_in_long_stationary': 2.0,  # Reduced from 5.0 to 2.0 seconds
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
        Lifecycle activate callback - called when transitioning from Inactive to Active.
        Check transform availability and set up publishers, subscribers, and timers.
        
        Returns:
            TransitionCallbackReturn: Success if activation completes successfully,
                                    FAILURE if transforms are not available
        """
        self.get_logger().info("Lifecycle transition: on_activate")
        
        # First check if transforms are available
        if not self.check_transform_availability():
            self.get_logger().warn("Transform not available yet - cannot activate")
            # Create a one-shot timer to retry activation after a delay
            self.create_timer(5.0, self.retry_activation, callback_group=None)
            return TransitionCallbackReturn.FAILURE
        
        try:
            # PHASE 4: Set up publishers (these can be managed by lifecycle node)
            self.setup_publishers()
            
            # Activate lifecycle publishers - fix by passing state parameter
            for pub in self._publishers:
                pub.on_activate(state)
            
            # PHASE 5: Set up subscriptions (only now that transform is available)
            self.setup_subscriptions()
            
            # DO NOT initialize filter here - instead, set a flag for delayed initialization
            self.pending_initialization = True
            self.initialization_attempts = 0
            
            # PHASE 7: Set up processing timers
            self.setup_timers()
            
            # Add a one-shot timer to attempt initialization after callbacks have had time to run
            self.create_timer(0.5, self.delayed_initialization, callback_group=None)
            
            # Mark as activated and ready
            self.is_activated = True
            self.is_ready = True
            
            self.get_logger().info("Node activated - waiting for sensor data to initialize filter")
            
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Error during activation: {str(e)}")
            return TransitionCallbackReturn.ERROR

    def delayed_initialization(self):
        """Attempt delayed initialization after sensors have had time to provide data."""
        if not self.initialized and self.pending_initialization:
            lidar_msg = self.sensor_buffer.get_latest_measurement('lidar')
            yolo_2d_msg = self.sensor_buffer.get_latest_measurement('yolo_2d')
            
            if lidar_msg or (yolo_2d_msg and 'yolo_2d' in self.bbox_data):
                # We have sensor data - initialize now
                self.initialize_filter_with_defaults()
                self.pending_initialization = False
                self.get_logger().info("Delayed initialization completed with sensor data")
            else:
                # Try again if we haven't made too many attempts
                self.initialization_attempts += 1
                if self.initialization_attempts < 5:
                    self.get_logger().info(f"No sensor data yet for initialization (attempt {self.initialization_attempts})")
                    self.create_timer(0.5, self.delayed_initialization, callback_group=None)
                else:
                    # Fall back to default initialization after multiple attempts
                    self.get_logger().warn("No sensor data available after multiple attempts - initializing with defaults")
                    self.initialize_filter_with_defaults()
                    self.pending_initialization = False
    
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
        
        # Remove static transform broadcaster
        # self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        
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
            self.get_logger().info(f"Using {self.reference_frame} as reference frame")
            self.get_logger().info(f"Using ascamera_camera_link_0 frame for yolo_3d, hsv_3d, and combined topics")
                
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
            
            # NEW: Add measurement noise for estimated 3D from 2D
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
        
        # ENHANCEMENT 1: Initialize motion state tracking
        self.init_motion_state_tracking()
        
        # NEW: Enhanced motion state memory system
        self.motion_state_memory = {
            'long_stationary_confirmed_time': 0.0,  # When long_stationary was confirmed
            'requires_cooldown': False,             # Flag for post-gap cooldown period
            'cooldown_end_time': 0.0,               # When cooldown period ends
            'stationary_detections_after_gap': 0,   # Counter for stationary detections after a gap
            'last_reset_time': 0.0,                 # When counters were last reset
            'continuous_movement_start': 0.0,       # When continuous movement started
            'state_confidence': {                   # Confidence levels for each state (0-1)
                "stationary": 0.5,
                "long_stationary": 0.0,
                "small_movement": 0.0,
                "medium_fast": 0.0,
                "unknown": 0.0
            },
            'gap_recovery_time': 0.0,               # When the last gap recovery happened
            'protected_state': None,                # State that's currently under protection
            'protection_violation_count': 0         # Count of attempted transitions blocked by protection
        }
        
        # NEW: Velocity credibility system
        self.velocity_credibility = {
            'score': 1.0,                   # Current credibility score (0.0-1.0)
            'history': deque(maxlen=10),    # Recent credibility scores
            'last_significant_change': 0.0, # Timestamp of last significant velocity change
            'transition_times': {},         # Timestamps of state transitions
            'false_transition_count': 0,    # Counter for suspected false transitions
            'recovery_cooldown': False,     # Flag for gap recovery cooldown period
            'cooldown_end_time': 0.0        # When cooldown period ends
        }
        
        self.get_logger().info(f"State tracking initialized with 4D state optimized for ground-only movement")
        
        # ENHANCEMENT 1: Initialize motion state tracking
        self.init_motion_state_tracking()
        
        # NEW: Initialize motion state memory and protection system
        self.motion_state_protection = {
            'long_stationary_confirmed_time': 0.0,  # When long_stationary was confirmed
            'long_stationary_established': False,   # Whether long_stationary is established
            'consecutive_stationary_after_long': 0, # Count of stationary detections after long_stationary
            'min_time_in_long_stationary': 2.0,     # Reduced from 5.0 to 2.0 seconds
            'post_gap_cooldown_active': False,      # Whether we're in post-gap cooldown
            'post_gap_cooldown_end': 0.0,           # When post-gap cooldown ends
            'post_gap_protected_state': None,       # The state protected during cooldown
            'last_gap_recovery_time': 0.0,          # When last sensor recovery happened
        }
        
        # NEW: Add state confidence tracking
        self.motion_state_confidence = {
            "stationary": 0.5,
            "long_stationary": 0.5,
            "small_movement": 0.5, 
            "medium_fast": 0.5,
            "unknown": 0.5
        }

    def init_sensor_synchronization(self):
        """Initialize sensor synchronization system with extended buffer sizes."""
        """Initialize sensor synchronization system with extended buffer sizes."""
        # Create sensor buffer with increased buffer size and time tolerance
        self.sensor_buffer = SensorBuffer(max_time_diff=0.5)
        # Add parent reference for motion state access
        self.sensor_buffer.parent_node = self
        
        # Define all expected sensors
        self.expected_sensors = ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']
        
        # ENHANCEMENT 6: Increase buffer sizes for predictive buffering
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
        self.sensor_frame_times = {sensor: deque(maxlen=40) for sensor in self.expected_sensors}  # Increased from 30 to 40
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
        
        # ENHANCEMENT 6: Create predictive buffer for position/velocity tracking
        self.position_prediction_buffer = deque(maxlen=10)  # Store recent predictions
        self.velocity_prediction_buffer = deque(maxlen=10)  # Store recent velocity predictions
        
        self.get_logger().info("Sensor synchronization system initialized with extended buffers")
        
        # ENHANCEMENT 6: Initialize sensor reliability tracker
        self.sensor_reliability_tracker = SensorReliabilityTracker(self.expected_sensors)

        # NEW: Initialize consecutive rejection tracking per sensor
        self.consecutive_rejections_per_sensor = {sensor: 0 for sensor in self.expected_sensors}

        # ENHANCEMENT 6: Initialize smoothed state estimator
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
        
        # NEW: Bounding box subscriptions for distance estimation 
        # Use Float32MultiArray for YOLO
        from std_msgs.msg import Float32MultiArray
        
        yolo_bbox_sub = self.create_subscription(
            Float32MultiArray,
            self.yolo_bbox_topic,
            lambda msg: self.bbox_callback(msg, 'yolo_2d'),
            10
        )
        self.subscribers.append(yolo_bbox_sub)
        
        # Keep BoundingBox2D for HSV if that's what the HSV node publishes
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
            f"Motion={self.motion_state}"  # Added motion state to status log
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
    
    def setup_timers(self):
        """Set up regular processing timers."""
        # Status timer (1 Hz)
        status_timer = self.create_timer(1.0, self.publish_status)
        self._timer_list.append(status_timer)  # Changed from self.timers
        
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

    def initialize_filter_with_defaults(self):
        """Initialize filter with default values, using first available sensor data if possible."""
        try:
            # Check for any existing sensor data to use for initialization
            lidar_msg = self.sensor_buffer.get_latest_measurement('lidar')
            yolo_2d_msg = self.sensor_buffer.get_latest_measurement('yolo_2d')
            
            # Initialize position from sensor data if available
            if lidar_msg:
                transformed = self.transform_point(lidar_msg, self.reference_frame, False)
                if transformed:
                    self.state = np.zeros(4, dtype=np.float32)
                    self.state[0] = transformed.point.x  # x position
                    self.state[1] = transformed.point.y  # y position
                    self.get_logger().info(f"Filter initialized with lidar data: pos=({self.state[0]:.2f}, {self.state[1]:.2f})")
            elif yolo_2d_msg and 'yolo_2d' in self.bbox_data:
                # Attempt to estimate 3D position from 2D yolo data
                estimated_3d = self.estimate_3d_from_2d(yolo_2d_msg, self.bbox_data['yolo_2d'])
                if estimated_3d:
                    self.state = np.zeros(4, dtype=np.float32)
                    self.state[0] = estimated_3d.point.x  # x position
                    self.state[1] = estimated_3d.point.y  # y position
                    self.get_logger().info(f"Filter initialized with estimated 3D from yolo_2d: pos=({self.state[0]:.2f}, {self.state[1]:.2f})")
            else:
                # Fall back to zeros if no sensor data available
                self.state = np.zeros(4, dtype=np.float32)
                self.get_logger().info("Filter initialized with zeros - waiting for sensor data to update position")
            
            # Set initial covariance (high uncertainty since this is a guess)
            self.covariance = np.eye(4, dtype=np.float32)
            self.covariance[0:2, 0:2] *= 1.0  # Position uncertainty
            self.covariance[2:4, 2:4] *= 2.0  # Velocity uncertainty
            
            self.initialized = True
            self.last_update_time = time.time()
            
            # Update uncertainty metrics
            self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:2, 0:2]) / 2.0)
            self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[2:4, 2:4]) / 2.0)
            
            self.get_logger().info(
                f"Filter initialized with default values. Beginning active tracking with higher uncertainty."
            )
            return True
        except Exception as e:
            self.get_logger().error(f"Error during default filter initialization: {str(e)}")
            return False

    def sensor_callback(self, msg, source):
        """
        Common callback for all sensor measurements.
        
        Args:
            msg (PointStamped): The point message from sensor
            source (str): Sensor source identifier
        """
        # Skip if not active yet
        if not self.is_activated:
            return
        
        try:
            # Get current time for timing statistics
            current_time = time.time()
            
            # Update statistics
            self.sensor_counts[source] += 1
            self.last_detection_time[source] = current_time
            
            # Track frame time for FPS calculation
            self.sensor_frame_times[source].append(current_time)
            
            # Calculate FPS based on recent frames
            if len(self.sensor_frame_times[source]) >= 2:
                # Use time difference between oldest and newest frame
                time_span = current_time - self.sensor_frame_times[source][0]
                if time_span > 0:
                    # Calculate frames per second (number of frames - 1) / time span
                    self.sensor_fps[source] = (len(self.sensor_frame_times[source]) - 1) / time_span
            
            # Log first few detections with more detail
            if self.sensor_counts[source] <= 3:
                self.get_logger().info(
                    f"Received {source} detection #{self.sensor_counts[source]}: "
                    f"({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f}) in {msg.header.frame_id} frame"
                )
            
            # For 2D YOLO data, estimate 3D position here instead of waiting until publish time
            if source == 'yolo_2d' and 'yolo_2d' in self.bbox_data:
                try:
                    # Only estimate 3D position if we have recent bbox data
                    estimated_3d_point = self.estimate_3d_from_2d(msg, self.bbox_data['yolo_2d'])
                    if estimated_3d_point:
                        # Add estimated 3D point to the sensor buffer as a new sensor type
                        self.sensor_buffer.add_measurement('yolo_2d_est3d', estimated_3d_point, msg.header.stamp)
                        
                        # Initialize this sensor type in the reliability tracker if needed
                        if hasattr(self, 'sensor_reliability_tracker'):
                            if 'yolo_2d_est3d' not in self.sensor_reliability_tracker.sensors:
                                self.sensor_reliability_tracker.add_sensor('yolo_2d_est3d')
                        
                        # Initialize gap tracking for this sensor type
                        if hasattr(self, 'sensor_gap_detection') and 'yolo_2d_est3d' not in self.sensor_gap_detection:
                            self.sensor_gap_detection['yolo_2d_est3d'] = {
                                'gap_detected': False,
                                'gap_start_time': 0.0,
                                'gap_level': 0.0,
                                'recent_gaps': deque(maxlen=5)
                            }
                            
                        # Also update last detection time for this derived sensor
                        if hasattr(self, 'last_detection_time'):
                            self.last_detection_time['yolo_2d_est3d'] = current_time
                            
                        # Initialize counts if needed
                        if 'yolo_2d_est3d' not in self.sensor_counts:
                            self.sensor_counts['yolo_2d_est3d'] = 0
                        self.sensor_counts['yolo_2d_est3d'] += 1
                        
                        # Initialize FPS tracking
                        if 'yolo_2d_est3d' not in self.sensor_frame_times:
                            self.sensor_frame_times['yolo_2d_est3d'] = deque(maxlen=40)
                        self.sensor_frame_times['yolo_2d_est3d'].append(current_time)
                        
                        # Log occasionally for debugging
                        if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 20 == 0:
                            self.get_logger().debug(
                                f"Created 3D estimate from YOLO 2D: pos=({estimated_3d_point.point.x:.2f}, "
                                f"{estimated_3d_point.point.y:.2f}, {estimated_3d_point.point.z:.2f})"
                            )
                except Exception as e:
                    if self.debug_level >= 1:
                        self.get_logger().warn(f"Error creating 3D estimate from YOLO 2D: {str(e)}")
            
            # Add to synchronization buffer (always add original measurement too)
            self.sensor_buffer.add_measurement(source, msg, msg.header.stamp)
            
            # If this is a 3D source and we're not initialized yet, try initializing
            if not self.initialized and not source.endswith('_2d'):
                self.get_logger().info(f"Received {source} data - attempting initialization")
                transformed = self.transform_point(msg, self.reference_frame, False)  # 3D data, so is_2d=False
                if transformed:
                    self.initialize_filter_with_measurement(transformed, source)
            if self.debug_level >= 2:
                self.get_logger().debug(
                    f"{source} detection: ({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f}) in {msg.header.frame_id} frame"
                )
            # Increment lidar message counter and log every 5 messages
            if source == 'lidar':
                self.lidar_msg_counter += 1
                if self.lidar_msg_counter % 3 == 0:
                    # Log detailed lidar position data once after every 3 messages
                    self.get_logger().info(f"[state_aware_fusion_node]: Received lidar detection #{self.lidar_msg_counter}: ({msg.point.x:.2f}, {msg.point.y:.2f}, {msg.point.z:.2f}) in {msg.header.frame_id} frame")
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
                    if source not in self._bbox_log_counter:self._bbox_log_counter[source] = 0
                    
                    # Increment counter and log details every 3 times
                    self._bbox_log_counter[source] += 1
                    if self._bbox_log_counter[source] % 3 == 0:
                        self.get_logger().info(f"Received {source} bbox: {width:.1f}x{height:.1f}")
            else:
                self.get_logger().warn(f"Invalid format for {source} bounding box message")
                
        except Exception as e:
            self.log_error(f"Error in {source} bbox callback: {str(e)}")

    def initialize_filter_with_measurement(self, msg, source):
        """
        Initialize the filter with a specific measurement using the blank slate approach.
        This treats the first reliable measurement as ground truth with high confidence.
        But first confirms agreement between multiple sensors or consistent readings.
        
        Args:
            msg (PointStamped): The point message for initialization
            source (str): Sensor source identifier
            
        Returns:
            bool: Whether initialization was successful
        """
        try:
            # Ensure message is in the reference frame
            if msg.header.frame_id != self.reference_frame:
                transformed = self.transform_point(msg, self.reference_frame, source.endswith('_2d'))
                if transformed is None:
                    self.get_logger().warn(f"Cannot initialize filter - transform failed from {msg.header.frame_id} to {self.reference_frame}")
                    return False
                msg = transformed
            
            # Initialize sensor agreement tracking if it doesn't exist
            if not hasattr(self, 'initialization_candidates'):
                self.initialization_candidates = {}
                self.initialization_start_time = time.time()
                self.initialization_consensus = None
                self.initialization_confidence = 0.0
                self.initialization_required_time = 2.0  # Require 2 seconds of agreement
                self.initialization_required_sensors = 2  # Require at least 2 sensors to agree
                self.initialization_distance_threshold = 0.15  # 15cm agreement threshold
            
            current_time = time.time()
            
            # Add this measurement to candidates
            if source not in self.initialization_candidates:
                self.initialization_candidates[source] = {
                    'position': [msg.point.x, msg.point.y, msg.point.z],
                    'last_update': current_time,
                    'first_seen': current_time,
                    'update_count': 1
                }
            else:
                # Update existing candidate
                self.initialization_candidates[source]['position'] = [msg.point.x, msg.point.y, msg.point.z]
                self.initialization_candidates[source]['last_update'] = current_time
                self.initialization_candidates[source]['update_count'] += 1
            
            # Remove stale candidates (older than 2 seconds)
            stale_sources = []
            for s, data in self.initialization_candidates.items():
                if current_time - data['last_update'] > 2.0:
                    stale_sources.append(s)
            
            for s in stale_sources:
                del self.initialization_candidates[s]
            
            # Find agreement between sensors
            agreement_groups = []
            for s1, data1 in self.initialization_candidates.items():
                group = [s1]
                pos1 = data1['position']
                
                for s2, data2 in self.initialization_candidates.items():
                    if s1 == s2:
                        continue
                    
                    pos2 = data2['position']
                    # Calculate distance between positions (x,y only for ground plane)
                    distance = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    if distance < self.initialization_distance_threshold:
                        group.append(s2)
                
                if len(group) >= self.initialization_required_sensors:
                    agreement_groups.append(group)
            
            # Sort groups by size (largest first)
            agreement_groups.sort(key=len, reverse=True)
            
            if agreement_groups:
                # Get the largest agreement group
                largest_group = agreement_groups[0]
                
                # Calculate average position from this group
                avg_x, avg_y, avg_z = 0, 0, 0
                for s in largest_group:
                    pos = self.initialization_candidates[s]['position']
                    avg_x += pos[0]
                    avg_y += pos[1]
                    avg_z += pos[2]
                
                avg_x /= len(largest_group)
                avg_y /= len(largest_group)
                avg_z /= len(largest_group)
                
                # Check if we have a consensus position
                if self.initialization_consensus is None:
                    # First consensus
                    self.initialization_consensus = [avg_x, avg_y, avg_z]
                    self.initialization_consensus_time = current_time
                    self.initialization_consensus_group = largest_group
                    self.get_logger().info(
                        f"Initial consensus found between {len(largest_group)} sensors: "
                        f"position=({avg_x:.2f}, {avg_y:.2f})"
                    )
                else:
                    # Check if this consensus is consistent with previous one
                    prev_x, prev_y, _ = self.initialization_consensus
                    consensus_distance = math.sqrt((avg_x - prev_x)**2 + (avg_y - prev_y)**2)
                    
                    if consensus_distance < self.initialization_distance_threshold:
                        # Consistent consensus - update the running average
                        alpha = 0.3  # Weight for new measurement
                        self.initialization_consensus[0] = (1-alpha) * self.initialization_consensus[0] + alpha * avg_x
                        self.initialization_consensus[1] = (1-alpha) * self.initialization_consensus[1] + alpha * avg_y
                        self.initialization_consensus[2] = self.basketball_z_height  # Always use standard height
                        
                        # Check if we've had consistent consensus for the required time
                        consensus_duration = current_time - self.initialization_consensus_time
                        
                        if consensus_duration >= self.initialization_required_time:
                            # We have enough consensus over time - initialize the filter
                            self.get_logger().info(
                                f"Initialization criteria met! {len(largest_group)} sensors agree for {consensus_duration:.1f}s"
                            )
                            
                            # --- Blank slate initialization approach ---
                            # Extract x and y position from the consensus position
                            # For a 4D state vector [x, y, vx, vy]
                            self.state = np.zeros(4, dtype=np.float32)
                            self.state[0] = self.initialization_consensus[0]  # x position
                            self.state[1] = self.initialization_consensus[1]  # y position
                            # Initialize velocities to zero
                            self.state[2] = 0.0  # vx (zero initial velocity)
                            self.state[3] = 0.0  # vy (zero initial velocity)
                            
                            # Set covariance with high confidence (very low uncertainty) in position
                            # and moderate uncertainty in velocity
                            self.covariance = np.eye(4, dtype=np.float32)
                            
                            # Very high confidence in position (low uncertainty values)
                            # More sensors = higher confidence
                            position_variance = 0.01 / min(1.0, len(largest_group) / 3.0)
                            self.covariance[0, 0] = position_variance  # x position variance
                            self.covariance[1, 1] = position_variance  # y position variance
                            
                            # Moderate uncertainty in velocity (we don't know velocity yet)
                            self.covariance[2, 2] = 0.5   # vx velocity variance
                            self.covariance[3, 3] = 0.5   # vy velocity variance
                            
                            self.initialized = True
                            self.last_update_time = current_time
                            
                            # Record initialization details
                            self.initialization_source = "+".join(largest_group)
                            self.initialization_time = current_time
                            
                            # Update uncertainty metrics
                            self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:2, 0:2]) / 2.0)
                            self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[2:4, 2:4]) / 2.0)
                            
                            self.get_logger().info(
                                f"Filter initialized using consensus from {len(largest_group)} sensors: "
                                f"position=({self.initialization_consensus[0]:.2f}, {self.initialization_consensus[1]:.2f}), "
                                f"uncertainty={self.position_uncertainty:.3f}m"
                            )
                            
                            # Temporarily disable motion state protection during initial measurement
                            if hasattr(self, 'motion_state_protection'):
                                self.motion_state_protection['initialization_mode'] = True
                                
                            # Start active tracking
                            self.get_logger().info("Filter initialized with high confidence - beginning active tracking")
                            
                            return True
                        else:
                            # Still waiting for required duration
                            if hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 10 == 0:
                                self.get_logger().info(
                                    f"Building initialization consensus: {len(largest_group)} sensors agree for {consensus_duration:.1f}s "
                                    f"(need {self.initialization_required_time:.1f}s)"
                                )
                    else:
                        # Consensus changed significantly - reset timer
                        self.initialization_consensus = [avg_x, avg_y, avg_z]
                        self.initialization_consensus_time = current_time
                        self.initialization_consensus_group = largest_group
                        self.get_logger().info(
                            f"Consensus position changed - restarting initialization timer. New position: ({avg_x:.2f}, {avg_y:.2f})"
                        )
            else:
                # Not enough sensors agree yet
                if hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 30 == 0:
                    self.get_logger().info(
                        f"Waiting for sensor agreement for initialization. Have {len(self.initialization_candidates)} candidates."
                    )
            
            # If we've been trying to initialize for too long (10+ seconds), fall back to simpler initialization
            if (current_time - self.initialization_start_time > 10.0) and not self.initialized:
                self.get_logger().warn("Falling back to single-sensor initialization after 10s without consensus")
                
                # --- Simple initialization with current measurement ---
                self.state = np.zeros(4, dtype=np.float32)
                self.state[0] = msg.point.x  # x position
                self.state[1] = msg.point.y  # y position
                self.state[2] = 0.0  # vx (zero initial velocity)
                self.state[3] = 0.0  # vy (zero initial velocity)
                
                # Set covariance with moderate confidence
                self.covariance = np.eye(4, dtype=np.float32)
                self.covariance[0, 0] = 0.05  # Higher uncertainty than consensus initialization
                self.covariance[1, 1] = 0.05
                self.covariance[2, 2] = 0.8
                self.covariance[3, 3] = 0.8
                
                self.initialized = True
                self.last_update_time = current_time
                self.initialization_source = f"{source} (fallback)"
                self.initialization_time = current_time
                
                # Update uncertainty metrics
                self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:2, 0:2]) / 2.0)
                self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[2:4, 2:4]) / 2.0)
                
                self.get_logger().info(
                    f"Filter initialized with fallback method using {source}: "
                    f"position=({msg.point.x:.2f}, {msg.point.y:.2f}), "
                    f"uncertainty={self.position_uncertainty:.3f}m"
                )
                
                # Start active tracking
                self.get_logger().info("Filter initialized with fallback method - beginning active tracking")
                
                return True
            
            return False  # Not initialized yet
            
        except Exception as e:
            self.get_logger().error(f"Error during filter initialization: {str(e)}")
            return False

    # ENHANCEMENT 8: Enhanced Motion State Detection with State Protection
    def detect_motion_state(self):
        """
        Detect the current motion state of the object with enhanced protection
        against false transitions, especially after sensor gaps.
        
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
            
        # ---------------------------------------------------------------------
        # NEW: Check if we're in a post-gap cooldown period
        # ---------------------------------------------------------------------
        if hasattr(self, 'motion_state_protection'):
            # If we're in cooldown, strictly enforce the protected state
            if self.motion_state_protection['post_gap_cooldown_active']:
                if current_time < self.motion_state_protection['post_gap_cooldown_end']:
                    protected_state = self.motion_state_protection['post_gap_protected_state']
                    
                    # Log this occasionally for debugging
                    if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 10 == 0:
                        self.get_logger().debug(
                            f"Motion state cooldown active: enforcing '{protected_state}' for {remaining:.1f}s more"
                        )
                    
                    return protected_state
                else:
                    # Cooldown period has ended
                    self.motion_state_protection['post_gap_cooldown_active'] = False
                    if self.debug_level >= 2:
                        self.get_logger().debug("Motion state cooldown period ended")
        
        # ---------------------------------------------------------------------
        # Check for gap recovery conditions to activate cooldown
        # ---------------------------------------------------------------------
        recent_gap_recovery = False
        if hasattr(self, 'sensor_gap_detection'):
            # Check for sensors that just recovered from gaps
            for sensor, gap_info in self.sensor_gap_detection.items():
                if not gap_info.get('gap_detected', True):  # Not currently in a gap
                    last_time = self.last_detection_time.get(sensor, 0)
                    if current_time - last_time < 0.5:  # Recent data
                        # This sensor may have just recovered
                        if gap_info.get('gap_start_time', 0) > 0:
                            gap_duration = current_time - gap_info.get('gap_start_time', 0)
                            if gap_duration > 0.5:  # It was a significant gap
                                recent_gap_recovery = True
                                # Record recovery time
                                if hasattr(self, 'motion_state_protection'):
                                    self.motion_state_protection['last_gap_recovery_time'] = current_time
        
        # If we just recovered from a gap, activate cooldown protection
        if recent_gap_recovery and hasattr(self, 'motion_state_protection') and not self.motion_state_protection['post_gap_cooldown_active']:
            # Set up cooldown period
            self.motion_state_protection['post_gap_cooldown_active'] = True
            # Protected state is the current motion state (before the gap)
            self.motion_state_protection['post_gap_protected_state'] = getattr(self, 'motion_state', 'unknown')
            # Set cooldown end time (2 seconds after recovery)
            self.motion_state_protection['post_gap_cooldown_end'] = current_time + 2.0
            
            if self.debug_level >= 1:
                self.get_logger().info(
                    f"Post-gap cooldown activated: protecting '{self.motion_state_protection['post_gap_protected_state']}' state for 2.0s"
                )
            
            # Return the protected state immediately
            return self.motion_state_protection['post_gap_protected_state']
        
        # IMPROVEMENT 3: Detect and filter out implausible velocity spikes during sensor gaps
        # Check for gaps in our sensor data
        current_time = time.time()
        has_recent_gap = False
        max_gap_level = 0.0
        
        if hasattr(self, 'sensor_gap_detection'):
            for sensor, gap_info in self.sensor_gap_detection.items():
                if gap_info.get('gap_detected', False):
                    has_recent_gap = True
                    max_gap_level = max(max_gap_level, gap_info.get('gap_level', 0.0))
        
        # Initialize velocity confidence if not already present
        if not hasattr(self, 'velocity_confidence'):
            self.velocity_confidence = 1.0  # Start with full confidence
            
        # Reduce confidence during sensor gaps
        if has_recent_gap:
            # Reduce confidence based on gap level (0.0-1.0)
            self.velocity_confidence = max(0.1, self.velocity_confidence * (1.0 - (max_gap_level * 0.5)))
        else:
            # Gradually restore confidence when no gaps
            self.velocity_confidence = min(1.0, self.velocity_confidence + 0.1)
            
        # Calculate average velocity with gap-aware filtering
        filtered_velocities, avg_velocity, implausible_detected = self.process_velocity_measurements(valid_velocities, list(self.time_history)[-5:])
        
        # ---------------------------------------------------------------------
        # NEW: Update state confidence levels based on velocity evidence
        # ---------------------------------------------------------------------
        
        # Initialize confidence dictionary if needed
        if not hasattr(self, 'motion_state_confidence'):
            self.motion_state_confidence = {
                "stationary": 0.5,
                "long_stationary": 0.0,
                "small_movement": 0.0, 
                "medium_fast": 0.0,
                "unknown": 0.0
            }
        
        # Update confidence values based on current velocity evidence
        if avg_velocity < 0.03:
            # Strong evidence for stationary state
            self.motion_state_confidence["stationary"] = min(1.0, self.motion_state_confidence["stationary"] + 0.1)
            self.motion_state_confidence["small_movement"] = max(0.0, self.motion_state_confidence["small_movement"] - 0.1)
            self.motion_state_confidence["medium_fast"] = max(0.0, self.motion_state_confidence["medium_fast"] - 0.2)
        elif avg_velocity < 0.25:
            # Evidence for small movement
            self.motion_state_confidence["small_movement"] = min(1.0, self.motion_state_confidence["small_movement"] + 0.1)
            self.motion_state_confidence["stationary"] = max(0.0, self.motion_state_confidence["stationary"] - 0.05)
            self.motion_state_confidence["medium_fast"] = max(0.0, self.motion_state_confidence["medium_fast"] - 0.1)
        else:
            # Evidence for medium/fast movement
            self.motion_state_confidence["medium_fast"] = min(1.0, self.motion_state_confidence["medium_fast"] + 0.15)
            self.motion_state_confidence["small_movement"] = max(0.0, self.motion_state_confidence["small_movement"] - 0.05)
            self.motion_state_confidence["stationary"] = max(0.0, self.motion_state_confidence["stationary"] - 0.15)
        
        # Special handling for long_stationary confidence
        if self.motion_state == "long_stationary":
            # Once established, long_stationary confidence decays very slowly
            self.motion_state_confidence["long_stationary"] = max(0.8, self.motion_state_confidence["long_stationary"])
            
            # During gaps, boost the confidence to prevent state transition
            if has_recent_gap:
                self.motion_state_confidence["long_stationary"] = 1.0
        
        # ---------------------------------------------------------------------
        # NEW: Apply confidence-based transition thresholds
        # ---------------------------------------------------------------------
        
        # Classifier with confidence-adjusted thresholds
        if has_recent_gap and hasattr(self, 'motion_state') and self.motion_state == "long_stationary":
            # Special case: during gaps, require 1.5x higher evidence to transition out of long_stationary
            # Previously required 3x more velocity - reduced to 1.5x
            if avg_velocity < 0.045:  # Modified from 0.03 * 3 (0.09) to 0.03 * 1.5 (0.045)
                base_motion_state = "stationary"
            elif avg_velocity < 0.375:  # Modified from 0.25 * 3 (0.75) to 0.25 * 1.5 (0.375)
                base_motion_state = "small_movement"
            else:
                base_motion_state = "medium_fast"
        elif self.velocity_confidence < 0.5 and hasattr(self, 'motion_state') and self.motion_state in ["stationary", "long_stationary"]:
            # Require stronger evidence to leave stationary state during low confidence periods
            if avg_velocity < 0.1:  # Increased threshold during low confidence
                base_motion_state = "stationary"
            elif avg_velocity < 0.5:  # Increased threshold during low confidence
                base_motion_state = "small_movement"
            else:
                base_motion_state = "medium_fast"
        else:        # Get average acceleration using velocity history
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

            # Apply hysteresis for state classification based on current state
            current_state = getattr(self, 'motion_state', 'unknown')

            # Default thresholds
            stationary_thresh = 0.02  # m/s
            small_movement_thresh = 0.20  # m/s

            # Apply hysteresis: harder to leave current state
            if current_state == "stationary":
                # Higher threshold to leave stationary
                stationary_thresh = 0.04  # Doubled
            elif current_state == "small_movement":
                # Adjusted thresholds for small_movement
                stationary_thresh = 0.015  # Lower to stay in small_movement
                small_movement_thresh = 0.25  # Higher to stay in small_movement
            elif current_state == "medium_fast":
                # Lower threshold to stay in medium_fast
                small_movement_thresh = 0.18

            # Use acceleration to detect rapid changes
            if acceleration > 2.0:  # Significant acceleration
                # Skip intermediate states for rapid acceleration
                base_motion_state = "medium_fast"
            elif avg_velocity < stationary_thresh:
                base_motion_state = "stationary"
            elif avg_velocity < small_movement_thresh:
                base_motion_state = "small_movement"
            else:
                base_motion_state = "medium_fast"
                
            # Log acceleration for debugging
            if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 10 == 0:
                self.get_logger().debug(f"Motion detection: velocity={avg_velocity:.3f}m/s, acceleration={acceleration:.2f}m/s², state={base_motion_state}")
        
        # IMPROVEMENT 3: Add transition inertia - require consistent evidence for state changes
        # Initialize state transition evidence if not already present
        if not hasattr(self, 'state_transition_evidence'):
            self.state_transition_evidence = {
                "stationary": 0,
                "small_movement": 0,
                "medium_fast": 0
            }
            
        # Update evidence counters
        self.state_transition_evidence[base_motion_state] += 1
        
        # Decay other evidence counters
        for state in self.state_transition_evidence:
            if state != base_motion_state:
                self.state_transition_evidence[state] = max(0, self.state_transition_evidence[state] - 1)
        
        # ---------------------------------------------------------------------
        # RELAXED: Special protection for long_stationary -> stationary transition
        # ---------------------------------------------------------------------
        if hasattr(self, 'motion_state') and self.motion_state == "long_stationary":
            # Already in long_stationary state - this needs special protection
            
            # NEW: Add stronger override for clear movement detection
            if base_motion_state in ["small_movement", "medium_fast"] and avg_velocity > 0.15:
                # If clear movement detected, override protection immediately
                self.get_logger().info(f"Movement override: velocity {avg_velocity:.2f}m/s exceeds threshold 0.15m/s")
                return base_motion_state
            
            # Check if long_stationary is established in memory
            if not hasattr(self, 'motion_state_protection') or not self.motion_state_protection.get('long_stationary_established', False):
                # Mark it as established and record the time
                if not hasattr(self, 'motion_state_protection'):
                    self.motion_state_protection = {}
                self.motion_state_protection['long_stationary_established'] = True
                self.motion_state_protection['long_stationary_confirmed_time'] = current_time
                
                if self.debug_level >= 1:
                    self.get_logger().info("Long stationary state confirmed and protected against transitions")
            
            # Special protection against transitioning back to regular stationary
            if base_motion_state == "stationary":
                # -----------------------------------------------------------------
                # FIX: Restructure to decide if transition should even be considered
                # -----------------------------------------------------------------
                
                # Initialize protection if needed
                if not hasattr(self, 'motion_state_protection'):
                    self.motion_state_protection = {}
                
                if 'consecutive_stationary_after_long' not in self.motion_state_protection:
                    self.motion_state_protection['consecutive_stationary_after_long'] = 0
                
                # Determine if we should even consider this transition
                consider_transition = False
                
                # Check velocity stability 
                if avg_velocity < 0.01:  # Very stable velocity
                    # Get time since long_stationary was established
                    time_in_long_stationary = current_time - self.motion_state_protection.get('long_stationary_confirmed_time', 0)
                    
                    # Only consider transitions if:
                    # 1. Object has been in long_stationary for at least 20 seconds AND
                    # 2. Velocity confidence is high
                    if time_in_long_stationary > 20.0 and self.velocity_confidence > 0.8:
                        consider_transition = True
                
                # Skip all processing if transition isn't being considered
                if not consider_transition:
                    # Silently maintain long_stationary without logging
                    base_motion_state = "long_stationary"
                    
                    # Reset counter to avoid accumulating incorrect statistics
                    self.motion_state_protection['consecutive_stationary_after_long'] = 0
                else:
                    # We've decided to consider this transition - now track evidence
                    
                    # Increment counter
                    self.motion_state_protection['consecutive_stationary_after_long'] += 1
                    
                    # Log only after every 20 consecutive detections
                    if self.motion_state_protection['consecutive_stationary_after_long'] % 20 == 0:
                        self.get_logger().info(
                            f"Long stationary -> stationary transition accepted after "
                            f"{self.motion_state_protection['consecutive_stationary_after_long']} consecutive stationary detections"
                        )
                    
                    # Reduce consecutive detection requirement from 5 to 2
                    if self.motion_state_protection['consecutive_stationary_after_long'] < 2:
                        # Not enough evidence - remain in long_stationary
                        base_motion_state = "long_stationary"
                    else:
                        # Log this transition (only if truly changing state)
                        self.get_logger().info(
                            f"Long stationary -> stationary transition accepted after "
                            f"{self.motion_state_protection['consecutive_stationary_after_long']} consecutive stationary detections"
                        )
            else:
                # Reset consecutive counter if not detecting stationary
                if hasattr(self, 'motion_state_protection'):
                    self.motion_state_protection['consecutive_stationary_after_long'] = 0
        
        # RELAXED: Special protection for stationary state
        if hasattr(self, 'motion_state') and self.motion_state in ["stationary", "long_stationary"]:
            # Require 2+ consecutive motion samples to transition away from stationary to movement
            # Reduced from 3+ to be more responsive
            evidence_needed = 2
            # During gaps, require more evidence but reduced from previous requirements
            if has_recent_gap:
                evidence_needed = 3  # Reduced from 5
                
            # Check if we have enough evidence for the new state
            if base_motion_state not in ["stationary", "long_stationary"] and self.state_transition_evidence[base_motion_state] < evidence_needed:
                # Not enough evidence to leave stationary state yet
                if self.motion_state == "long_stationary":
                    base_motion_state = "long_stationary"
                else:
                    base_motion_state = "stationary"
        
        # IMPROVEMENT 2: Detect long-term stationary state
        
        # Initialize tracking attributes if they don't exist
        if not hasattr(self, 'stationary_start_time'):
            self.stationary_start_time = None
            
        if base_motion_state == "stationary":
            # Start or continue tracking stationary time
            if self.stationary_start_time is None:
                self.stationary_start_time = current_time
                motion_state = "stationary"
            else:
                # Check if it's been stationary for a significant period (5+ seconds)
                stationary_duration = current_time - self.stationary_start_time
                if stationary_duration > 5.0:  # 5 seconds threshold for long-term stationary
                    motion_state = "long_stationary"
                    
                    # Record when long_stationary was confirmed
                    if hasattr(self, 'motion_state_protection'):
                        if not self.motion_state_protection.get('long_stationary_established', False):
                            self.motion_state_protection['long_stationary_established'] = True
                            self.motion_state_protection['long_stationary_confirmed_time'] = current_time
                            
                    # Set confidence to high for long_stationary
                    if hasattr(self, 'motion_state_confidence'):
                        self.motion_state_confidence["long_stationary"] = 1.0
                    
                    # Log transition to long-term stationary occasionally
                    if not hasattr(self, 'last_long_stationary_log') or current_time - self.last_long_stationary_log > 10.0:
                        self.get_logger().info(f"Object has been stationary for {stationary_duration:.1f}s - using long-term stationary mode")
                        self.last_long_stationary_log = current_time
                else:
                    motion_state = "stationary"
        else:
            # ---------------------------------------------------------------------
            # RELAXED: Check for minimum time requirement in long_stationary state
            # ---------------------------------------------------------------------
            if hasattr(self, 'motion_state') and self.motion_state == "long_stationary" and hasattr(self, 'motion_state_protection'):
                # Get time since long_stationary was established
                time_in_long_stationary = current_time - self.motion_state_protection.get('long_stationary_confirmed_time', 0)
                min_time_required = self.motion_state_protection.get('min_time_in_long_stationary', 1.0)  # Reduced from 2.0 to 1.0
                
                if time_in_long_stationary < min_time_required:
                    # Not been in long_stationary state long enough to leave it for movement
                    # Only apply this protection for actual movement transitions, not back to stationary
                    if base_motion_state not in ["stationary", "unknown"]:
                        # Check if movement is very significant, which would override protection
                        if avg_velocity > 0.5:  # Clear movement detected
                            motion_state = base_motion_state
                            # Log this special override
                            self.get_logger().info(
                                f"Significant movement (v={avg_velocity:.2f}m/s) overriding long_stationary protection"
                            )
                        else:
                            # Movement not significant enough - maintain long_stationary
                            motion_state = "long_stationary"
                            
                            # Log this protection occasionally
                            if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 20 == 0:
                                self.get_logger().debug(
                                    f"Protected long_stationary state (established for {time_in_long_stationary:.1f}s < required {min_time_required:.1f}s)"
                                )
                    else:
                        # For non-movement transitions, use normal logic
                        motion_state = base_motion_state
                else:
                    # Been in long_stationary state long enough, normal transitions allowed
                    motion_state = base_motion_state
            else:
                # Reset stationary timer when moving
                self.stationary_start_time = None
                motion_state = base_motion_state
        
        # Update motion state counts for stability
        if not hasattr(self, 'motion_state_counts'):
            self.motion_state_counts = {
                "stationary": 0,
                "long_stationary": 0,
                "small_movement": 0,
                "medium_fast": 0,
                "unknown": 0
            }
        
        self.motion_state_counts[motion_state] += 1
        for state in self.motion_state_counts:
            if state != motion_state:
                self.motion_state_counts[state] = max(0, self.motion_state_counts[state] - 1)
        
        # Get the most frequent state for stability
        dominant_state = max(self.motion_state_counts, key=self.motion_state_counts.get)
        
        # Initialize motion state if not already present
        if not hasattr(self, 'motion_state'):
            self.motion_state = "unknown"
            self.prev_motion_state = "unknown"
        
        # ---------------------------------------------------------------------
        # RELAXED: Apply confidence-based transition protection
        # ---------------------------------------------------------------------
        # Only for transitions FROM long_stationary TO stationary
        if self.motion_state == "long_stationary" and dominant_state == "stationary":
            # Check if confidence for long_stationary is still high
            long_stationary_confidence = self.motion_state_confidence.get("long_stationary", 0.0)
            stationary_confidence = self.motion_state_confidence.get("stationary", 0.0)
            
            # RELAXED: Require lower confidence to transition from long_stationary to stationary
            # Changed from 3x to 2x higher confidence
            if long_stationary_confidence > stationary_confidence / 2.0:
                # Confidence in long_stationary is still strong enough - block transition
                dominant_state = "long_stationary"
                
                # If debugging enabled, log this protection
                if self.debug_level >= 2:
                    self.get_logger().debug(
                        f"Protected long_stationary (conf={long_stationary_confidence:.2f}) from transition to stationary (conf={stationary_confidence:.2f})"
                    )
                
                # Track count of blocked transitions
                if hasattr(self, 'motion_state_protection'):
                    if 'protection_violation_count' not in self.motion_state_protection:
                        self.motion_state_protection['protection_violation_count'] = 0
                    self.motion_state_protection['protection_violation_count'] += 1
        
        # RELAXED: Special handling for transitions from long_stationary to movement states
        if self.motion_state == "long_stationary" and dominant_state in ["small_movement", "medium_fast"]:
            # Get confidence values
            long_stationary_confidence = self.motion_state_confidence.get("long_stationary", 0.0)
            movement_confidence = self.motion_state_confidence.get(dominant_state, 0.0)
            
            # RELAXED: Only need a small amount of evidence to override previous state
            # If movement confidence is > 50% of long_stationary confidence, allow transition
            if movement_confidence > long_stationary_confidence * 0.5:
                # Log this relaxed transition
                if self.debug_level >= 1:
                    self.get_logger().info(
                        f"Allowing movement transition: {dominant_state} confidence ({movement_confidence:.2f}) > "
                        f"{long_stationary_confidence * 0.5:.2f} threshold"
                    )
                # Let transition proceed
            else:
                # Still not enough confidence, maintain long_stationary
                dominant_state = "long_stationary"
        
        # Log if motion state changes
        if dominant_state != self.motion_state:
            self.prev_motion_state = self.motion_state
            self.motion_state = dominant_state
            
            # Add confidence information to the log
            confidence_str = ""
            if hasattr(self, 'motion_state_confidence'):
                from_conf = self.motion_state_confidence.get(self.prev_motion_state, 0.0)
                to_conf = self.motion_state_confidence.get(self.motion_state, 0.0)
                confidence_str = f", confidence={to_conf:.2f}"
                
            self.get_logger().info(f"Motion state changed: {self.prev_motion_state} -> {self.motion_state} "
                                f"(velocity={avg_velocity:.3f}m/s{confidence_str})")
        
        return self.motion_state

    # ENHANCEMENT 3: Dynamic Measurement Validation
    def get_innovation_threshold(self, source, motion_state):
        """
        Get adaptive innovation threshold based on sensor type and motion state.
        
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
        
        # Base thresholds for each sensor type and motion state
        base_thresholds = {
            "lidar": {
                "stationary": (3.0, 1.5),  # (initial_threshold, min_threshold)
                "long_stationary": (2.5, 1.2),  # Even more permissive for long-term stationary objects
                "small_movement": (5.0, 2.0),
                "medium_fast": (8.0, 3.0),
                "unknown": (10.0, 3.0)
            },
            "3d_vision": {
                "stationary": (6.0, 2.0),
                "long_stationary": (5.0, 1.8),  # More permissive for long-term stationary
                "small_movement": (9.0, 3.0),
                "medium_fast": (12.0, 4.0),
                "unknown": (15.0, 5.0)
            },
            "2d": {
                "stationary": (10.0, 3.0),
                "long_stationary": (8.0, 2.5),  # More permissive for long-term stationary
                "small_movement": (15.0, 5.0),
                "medium_fast": (20.0, 8.0),
                "unknown": (25.0, 10.0)
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
        
        # Decay toward minimum with consecutive successful updates
        decay_factor = max(0.1, min(1.0, 8.0 / (self.consecutive_updates + 1)))
        threshold = minimum + (initial - minimum) * decay_factor
        
        # Apply additional adjustments
        if motion_state == "medium_fast" and source == "lidar":
            threshold *= 1.2  # Extra permissiveness for primary sensor during fast motion
        
        # ENHANCEMENT: Special handling for rolling balls with lidar
        # If the ball is detected as rolling on the ground, modify the lidar threshold
        if source == "lidar" and hasattr(self, 'flat_ground_detected') and self.flat_ground_detected:
            # For a rolling ball, lidar measurements at the bottom of the ball might be inconsistent
            # So we increase the threshold to allow more variation in measurements
            threshold *= 1.4  # 40% increase for rolling ball with lidar
            
            # Occasionally log this adjustment
            if hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 20 == 0:
                self.get_logger().info(f"Adjusting lidar threshold for rolling ball: {threshold:.2f}")
        
        return threshold

    # ENHANCEMENT 4: Flat Ground Movement Handling
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
        
        # Always enforce ground height constraint through the ground filter
        # This happens in publish_state when we run the state through the ground filter

    # ENHANCEMENT 5: Smart Sensor Recovery
    def handle_sensor_recovery(self):
        """
        Monitor sensor availability patterns and handle recovery after gaps.
        Fixes the "dead zone" between 0.5s and 2.0s where gaps weren't tracked.
        """
        current_time = time.time()
        
        for sensor in ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']:
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
                    'gap_level': 0.0  # New field to track gap severity (0.0-1.0)
                }
            
            # Check if sensor just recovered after a gap
            if self.sensor_gap_detection[sensor]['gap_detected']:
                # Get newest measurement
                msg = self.sensor_buffer.get_latest_measurement(sensor)
                if msg is not None and gap_duration < 0.5:  # Fresh data
                    total_gap = current_time - self.sensor_gap_detection[sensor]['gap_start_time']
                    self.get_logger().info(f"{sensor} recovered after {total_gap:.1f}s gap")
                    
                    # Temporarily increase covariance for more measurement acceptance
                    # Scale by gap duration - longer gaps need more adjustment
                    adjustment_factor = min(2.0, 1.0 + (total_gap / 2.0))
                    self.covariance[0:3, 0:3] *= adjustment_factor
                    
                    # Clear gap flag
                    self.sensor_gap_detection[sensor]['gap_detected'] = False
                    self.sensor_gap_detection[sensor]['gap_level'] = 0.0

    def filter_update(self):
        """
        Perform a Kalman filter update based on synchronized sensor measurements.
        Enhanced with new reliability tracking and smoothing capabilities.
        """
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
                dt = min(dt, 0.5)  # Cap at 0.5 seconds to prevent big jumps
            
            # IMPROVEMENT 7: Apply gap-aware covariance adjustment
            self.adjust_covariance_for_gaps()
            
            # Find synchronized measurements
            measurements = self.sensor_buffer.find_synchronized_measurements(min_sensors=1)
            
            # Add debug logging for sensor synchronization
            #self.get_logger().info(f"Synchronized measurements found: {list(measurements.keys())}")
            
            # Update sync quality metrics
            self.sync_quality_metrics['attempt_counts'] += 1
            if measurements:
                self.sync_quality_metrics['sync_counts'] += 1
                self.sync_quality_metrics['success_rate'] = self.sync_quality_metrics['sync_counts'] / self.sync_quality_metrics['attempt_counts']
                
                # Track sensor availability
                for sensor in self.expected_sensors:
                    if sensor in measurements:
                        if sensor not in self.sync_quality_metrics['sensor_availability']:
                            self.sync_quality_metrics['sensor_availability'][sensor] = 1
                        else:
                            self.sync_quality_metrics['sensor_availability'][sensor] += 1
            
            # Predict state forward to current time
            self.predict_state(dt)
            
            # Update state with measurements if available
            successful_update = False
            if measurements:
                successful_update = self.update_state(measurements)
                
                # IMPROVEMENT 3: Record sensor reliability
                for sensor in measurements.keys():
                    self.sensor_reliability_tracker.record_measurement(sensor, successful_update)            # Update last update time
            self.last_update_time = current_time
                
            # Update uncertainty metrics - FIX: Use [0:2, 0:2] for position part of 4D state
            self.position_uncertainty = math.sqrt(max(0.0, np.trace(self.covariance[0:2, 0:2]) / 2.0))
            self.velocity_uncertainty = math.sqrt(max(0.0, np.trace(self.covariance[2:4, 2:4]) / 2.0))
            
            # Store current state in history buffers - FIX: Handle 4D state properly for history tracking
            if hasattr(self, 'position_history'):
                # For position history, create a 3D position with fixed z-height
                pos_3d = [self.state[0], self.state[1], self.basketball_z_height]
                self.position_history.append(pos_3d)
                
            if hasattr(self, 'velocity_history'):
                # For velocity history, create a 3D velocity with zero z component
                vel_3d = [self.state[2], self.state[3], 0.0]
                self.velocity_history.append(vel_3d)
                
            if hasattr(self, 'time_history'):
                self.time_history.append(current_time)
            
            # IMPROVEMENT 3: Add current state to smoother
            if hasattr(self, 'smoothed_state_estimator'):
                # Create 3D position and velocity vectors for the smoother
                pos_3d = np.array([self.state[0], self.state[1], self.basketball_z_height])
                vel_3d = np.array([self.state[2], self.state[3], 0.0])
                
                self.smoothed_state_estimator.add_state(
                    pos_3d,
                    vel_3d,
                    self.position_uncertainty,
                    current_time
                )
                
                # Get smoothed state if available
                smoothed = self.smoothed_state_estimator.get_smoothed_state()
                if smoothed and self.position_uncertainty > 0.1:
                    # Use smoothed state for final output to reduce uncertainty spikes
                    # But only if current uncertainty is significant
                    # FIX: Only copy x,y to 4D state
                    self.state[0:2] = smoothed['position'][0:2].copy()
                    self.state[2:4] = smoothed['velocity'][0:2].copy()
                    
                    if self.debug_level >= 2:
                        smoothed_uncertainty = smoothed['uncertainty']
                        self.get_logger().debug(
                            f"Applied smoothing: uncertainty {self.position_uncertainty:.3f}m -> {smoothed_uncertainty:.3f}m"
                        )
            
            # IMPROVEMENT 8: Update tracking status using confidence-based approach
            self.update_tracking_status()
            
            # Log the state before publishing
            #self.get_logger().info(f"State before publishing: pos=({self.state[0]:.2f}, {self.state[1]:.2f}), velocity=({self.state[2]:.2f}, {self.state[3]:.2f})")
            
            # Publish fused position and velocity
            self.publish_state()
            
            # Publish uncertainty
            self.publish_uncertainty()
            
            # Update diagnostics
            self.update_diagnostics()
            
            # Apply flat ground constraints if needed
            self.apply_flat_ground_constraints()
            
            # Handle sensor recovery
            self.handle_sensor_recovery()
            
            # Update motion state
            self.detect_motion_state()
            
            # Decay reliability for unused sensors
            if hasattr(self, 'sensor_reliability_tracker'):
                self.sensor_reliability_tracker.decay_unused_sensors()
                
        except Exception as e:
            self.get_logger().error(f"Error during filter update: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def predict_state(self, dt):
        """
        Predict the state forward by dt seconds, optimized for ground-only movement.
        
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
        
        # Apply adaptive process noise based on prediction duration
        gap_factor = min(5.0, max(1.0, dt / 0.1))  # Scale factor based on gap length
        
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
                gap_factor *= (1.0 + avg_gap_level)
        
        # Process noise parameters with adaptive scaling
        q_pos = self.process_noise_pos * dt * gap_factor
        q_vel = self.process_noise_vel * dt * gap_factor
        
        # Apply physics-based rolling friction for ground movement
        # (Basketball always rolls on ground, never bounces)
        friction_coef = 0.03  # Rolling friction coefficient
        
        # Apply deceleration to horizontal velocity components
        current_velocity = np.linalg.norm(self.state[2:4])  # x-y plane velocity
        if current_velocity > 0:
            # Calculate friction deceleration: a = μg
            deceleration = friction_coef * 9.81  # μg in m/s²
            
            # Don't decelerate more than the current velocity
            max_dv = current_velocity
            dv = min(max_dv, deceleration * dt)
            
            # Apply proportional deceleration to velocity components
            if dv > 0 and current_velocity > 0:
                factor = 1.0 - (dv / current_velocity)
                self.state[2] *= factor  # Reduce x velocity
                self.state[3] *= factor  # Reduce y velocity
        
        # Fill in the 4x4 process noise matrix
        # Position variances
        self._Q_matrix[0, 0] = q_pos * dt**3 / 3.0  # x position variance
        self._Q_matrix[1, 1] = q_pos * dt**3 / 3.0  # y position variance
        
        # Velocity variances
        self._Q_matrix[2, 2] = q_vel * dt          # x velocity variance
        self._Q_matrix[3, 3] = q_vel * dt          # y velocity variance
        
        # Position-velocity covariances
        self._Q_matrix[0, 2] = self._Q_matrix[2, 0] = q_pos * dt**2 / 2.0  # x position-velocity
        self._Q_matrix[1, 3] = self._Q_matrix[3, 1] = q_pos * dt**2 / 2.0  # y position-velocity
        
        # Predict state using state transition matrix
        self.state = np.dot(self._F_matrix, self.state)
        
        # Predict covariance
        self.covariance = np.dot(np.dot(self._F_matrix, self.covariance), self._F_matrix.T) + self._Q_matrix
        
        # Ensure covariance remains symmetric
        self.covariance = 0.5 * (self.covariance + self.covariance.T)    
    
    def update_state(self, measurements):
        """
        Update the state with synchronized measurements (4D state version).
        
        Args:
            measurements (dict): Dictionary of {sensor_name: measurement}
            
        Returns:
            bool: Whether any measurements were successfully processed
        """
        # Store successful update flag to track if any measurements were processed
        successful_update = False
        
        # Add debug log for synchronized measurements
        #self.get_logger().info(f"Synchronized measurements found: {list(measurements.keys())}")
        
        # Motion state for adaptive validation
        motion_state = self.detect_motion_state()
        
        for sensor, msg in measurements.items():
            transformed = None # Initialize transformed variable
            is_2d_sensor = sensor.endswith('_2d')

            # For 2D sensors, estimate 3D position first
            if is_2d_sensor:
                # For 2D sensors, estimate 3D position first
                if sensor in self.bbox_data:
                    transformed = self.estimate_3d_from_2d(msg, self.bbox_data[sensor])
                    if transformed is None:
                        if self.debug_level >= 1:
                            self.get_logger().warn(f"Failed to estimate 3D position for {sensor}, skipping measurement.")
                        continue # Skip this measurement if estimation failed
                    # Note: transformed is already in the reference_frame
                else:
                    if self.debug_level >= 1:
                        self.get_logger().warn(f"No bounding box data available for {sensor}, cannot estimate 3D position.")
                    continue # Skip if no bbox data
            else:
                # For 3D sensors, transform the point as before
                transformed = self.transform_point(msg, self.reference_frame, False)
                if not transformed:
                    continue # Skip if transformation failed

            # --- BEGIN ADDITION: Hard Position Limits ---
            max_coord = 5.0 # Maximum plausible distance from origin (e.g., 5 meters)
            if abs(transformed.point.x) > max_coord or abs(transformed.point.y) > max_coord:
                self.get_logger().warn(
                    f"Rejecting {sensor} measurement outside hard limits: "
                    f"pos=({transformed.point.x:.2f}, {transformed.point.y:.2f}), limits=±{max_coord}m"
                )
                # Increment rejection counter and potentially increase covariance slightly
                self.consecutive_rejections_per_sensor[sensor] = self.consecutive_rejections_per_sensor.get(sensor, 0) + 1
                # Optional: Slightly increase covariance on hard rejection
                # self.covariance[0:2, 0:2] *= 1.01
                # self.covariance = 0.5 * (self.covariance + self.covariance.T)
                continue # Skip this measurement
            # --- END ADDITION ---
            
            # For ground-only basketball movement, we need special handling for measurements
            # 1. For 3D sensors, we use x,y and ignore z (basketball is always at fixed height)
            # 2. For 2D sensors, we use x,y coordinates
            
            if sensor.endswith('_2d'):
                # For 2D sensors, we create a 2x4 measurement matrix that only extracts x,y
                z = np.array([transformed.point.x, transformed.point.y], dtype=np.float32)
                H = np.zeros((2, 4), dtype=np.float32)  # 2x4 matrix for 2D sensors with 4D state
                H[0, 0] = 1.0  # Extract x position
                H[1, 1] = 1.0  # Extract y position
                
                # Get appropriate noise matrix
                if sensor == 'hsv_2d':
                    R = np.diag([self.measurement_noise_hsv_2d, self.measurement_noise_hsv_2d]).astype(np.float32)
                elif sensor == 'yolo_2d':
                    R = np.diag([self.measurement_noise_yolo_2d, self.measurement_noise_yolo_2d]).astype(np.float32)
                else:
                    R = np.diag([50.0, 50.0]).astype(np.float32)  # Default noise
            else:
                # For 3D sensors, we extract x,y and optionally use z as a consistency check
                # (since basketball is always at a known height)
                z = np.array([transformed.point.x, transformed.point.y], dtype=np.float32)
                H = np.zeros((2, 4), dtype=np.float32)  # 2x4 matrix for position-only with 4D state
                H[0, 0] = 1.0  # Extract x position
                H[1, 1] = 1.0  # Extract y position
                
                # Get appropriate noise matrix based on sensor type
                if sensor == 'lidar':
                    R = np.diag([self.measurement_noise_lidar, self.measurement_noise_lidar]).astype(np.float32)
                elif sensor == 'hsv_3d':
                    R = np.diag([self.measurement_noise_hsv_3d, self.measurement_noise_hsv_3d]).astype(np.float32)
                elif sensor == 'yolo_3d':
                    R = np.diag([self.measurement_noise_yolo_3d, self.measurement_noise_yolo_3d]).astype(np.float32)
                else:
                    R = np.diag([0.1, 0.1]).astype(np.float32)  # Default noise
                
                # Optional check if z-height is reasonable (within tolerance of expected height)
                # Skip if the measurement's z-value is too far from expected basketball height
                z_tolerance = 0.1  # 10cm tolerance for z-height
                if abs(transformed.point.z - self.basketball_z_height) > z_tolerance:
                    if self.debug_level >= 2:
                        self.get_logger().debug(
                            f"Skipping {sensor} measurement with unusual z-height: "
                            f"{transformed.point.z:.3f}m (expected: {self.basketball_z_height:.3f}m ±{z_tolerance:.2f}m)"
                        )
                    continue
            
            # Calculate innovation (measurement residual)
            y = z - np.dot(H, self.state)
            
            # Innovation covariance
            S = np.dot(np.dot(H, self.covariance), H.T) + R
            
            # Apply dynamic measurement validation based on motion state
            threshold = self.get_innovation_threshold(sensor, motion_state)

            # --- BEGIN ADDITION: Cap the Mahalanobis Threshold ---
            max_threshold = 25.0 # Set a maximum allowable threshold
            original_threshold = threshold
            threshold = min(threshold, max_threshold)
            # --- END ADDITION ---
            
            # Compute Mahalanobis distance for validation
            try:
                S_inv = np.linalg.inv(S)
                mahalanobis_dist = np.sqrt(np.dot(np.dot(y.T, S_inv), y))
                
                # --- BEGIN ADDITION: Debug Logging ---
                if self.debug_level >= 2:
                    self.get_logger().debug(
                        f"Validation Check [{sensor}]: "
                        f"Measurement z=({z[0]:.2f}, {z[1]:.2f}), "
                        f"Innovation y=({y[0]:.2f}, {y[1]:.2f}), "
                        f"Mahalanobis dist={mahalanobis_dist:.2f}, "
                        f"Threshold={threshold:.2f} (Original={original_threshold:.2f}, Cap={max_threshold:.1f})"
                    )
                # --- END ADDITION ---

                # Store innovation for diagnostic purposes
                if hasattr(self, 'innovation_history'):
                    self.innovation_history.append(mahalanobis_dist)
                  # Skip measurement if it fails validation
                if mahalanobis_dist > threshold:
                    # --- MODIFIED LOG ---
                    self.get_logger().debug(
                        f"Rejecting {sensor} measurement: innovation {mahalanobis_dist:.2f} > threshold {threshold:.2f}"
                    )
                    # --- END MODIFIED LOG ---
                    # --- BEGIN MODIFICATION ---
                    # Increment consecutive rejection counter for this sensor
                    consecutive_rejections = self.consecutive_rejections_per_sensor.get(sensor, 0) + 1
                    self.consecutive_rejections_per_sensor[sensor] = consecutive_rejections

                    # Adaptive rejection growth - increase factor with more consecutive rejections
                    base_growth = 1.02  # Base growth factor (2%)
                    adaptive_factor = min(3.0, 1.0 + (consecutive_rejections * 0.05))  # Scale up to 3x with more rejections
                    rejection_growth_factor = base_growth * adaptive_factor

                    # Log when rejection count is significant
                    if consecutive_rejections >= 3 and self.debug_level >= 1:
                        self.get_logger().info(
                            f"Increasing uncertainty after {consecutive_rejections} consecutive rejections for {sensor}: "
                            f"factor={rejection_growth_factor:.2f}"
                        )

                    # Apply larger growth for position than velocity during rejections
                    self.covariance[0:2, 0:2] *= rejection_growth_factor  # Position uncertainty
                    self.covariance[2:4, 2:4] *= (rejection_growth_factor * 0.8)  # Slightly less for velocity
                    # Ensure symmetry
                    self.covariance = 0.5 * (self.covariance + self.covariance.T)
                    # Reset consecutive updates counter on rejection
                    self.consecutive_updates = 0
                    # --- END MODIFICATION ---
                    continue

                # --- BEGIN MODIFICATION ---
                # Reset consecutive rejection counter for this sensor on successful validation
                self.consecutive_rejections_per_sensor[sensor] = 0
                # --- END MODIFICATION ---

                # Update consecutive updates counter for threshold adjustment
                self.consecutive_updates += 1
                
            except np.linalg.LinAlgError:
                self.get_logger().warn(f"Matrix inversion failed during validation for {sensor}")
                continue
            
            # Kalman gain
            try:
                K = np.dot(np.dot(self.covariance, H.T), np.linalg.inv(S))
                
                # Update state
                self.state = self.state + np.dot(K, y)
                
                # Update covariance using Joseph form for numerical stability
                I = np.eye(self.state.shape[0], dtype=np.float32)
                self.covariance = np.dot(np.dot(I - np.dot(K, H), self.covariance), 
                                        (I - np.dot(K, H)).T) + np.dot(np.dot(K, R), K.T)
                
                # Ensure covariance remains symmetric and positive definite
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
        
        # Add debug log for state after update
        #self.get_logger().info(
        #    f"State after update: pos=({self.state[0]:.2f}, {self.state[1]:.2f}), uncertainty={self.position_uncertainty:.2f}"
        #)
        
        # Return flag indicating if any measurements were successfully processed
        return successful_update

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
    
    def get_measurement_noise(self, sensor, is_2d):
        """
        Get measurement noise values for a sensor, factoring in quality information.
        
        Args:
            sensor (str): Sensor name
            is_2d (bool): Whether the sensor is 2D
            
        Returns:
            numpy.ndarray: Measurement noise covariance matrix
        """
        # Start with base noise values
        if sensor == 'lidar':
            base_noise = self.base_measurement_noise_lidar
        elif sensor == 'hsv_3d':
            base_noise = self.base_measurement_noise_hsv_3d
        elif sensor == 'yolo_3d': 
            base_noise = self.base_measurement_noise_yolo_3d
        elif sensor == 'hsv_2d':
            base_noise = self.base_measurement_noise_hsv_2d
        elif sensor == 'yolo_2d':
            base_noise = self.base_measurement_noise_yolo_2d
        elif sensor == 'hsv_2d_est3d':
            base_noise = self.base_measurement_noise_hsv_2d_est3d
        elif sensor == 'yolo_2d_est3d':
            base_noise = self.base_measurement_noise_yolo_2d_est3d
        else:
            base_noise = 0.1  # Default fallback
        
        # Get sensor confidence based on quality
        confidence = self.get_measurement_confidence(sensor)
        
        # Adjust noise based on confidence - higher confidence = lower noise
        if confidence > 0.8:
            # High confidence - reduce noise
            adjusted_noise = base_noise * 0.7
        elif confidence < 0.4:
            # Low confidence - increase noise
            adjusted_noise = base_noise * 2.0
        else:
            # Moderate confidence - linear scaling
            factor = 2.0 - 1.5 * confidence  # Maps 0.4->1.4, 0.8->0.8
            adjusted_noise = base_noise * factor
            
        # Get current motion state for additional adjustment
        motion_state = self.detect_motion_state()
        
        # Adjust for motion state
        if motion_state == "medium_fast":
            adjusted_noise *= 1.2  # Increase noise during fast motion
        elif motion_state == "long_stationary":
            adjusted_noise *= 0.8  # Decrease noise when stationary for long periods
            
        # Create appropriate noise matrix based on sensor dimensionality
        if is_2d:
            # 2D measurements - 2x2 noise matrix
            R = np.eye(2, dtype=np.float32) * adjusted_noise
        else:
            # 3D measurements - 3x3 noise matrix
            R = np.eye(3, dtype=np.float32) * adjusted_noise
            
        return R"""
        Get the measurement noise covariance matrix for a sensor.
        
        Args:
            sensor (str): Sensor name
            is_2d (bool): Whether the sensor is 2D or 3D
            
        Returns:
            np.ndarray: The measurement noise covariance matrix
        """
        if is_2d:
            if sensor == 'hsv_2d':
                return np.diag([self.measurement_noise_hsv_2d, self.measurement_noise_hsv_2d]).astype(np.float32)
            elif sensor == 'yolo_2d':
                return np.diag([self.measurement_noise_yolo_2d, self.measurement_noise_yolo_2d]).astype(np.float32)
            else:
                return np.diag([1.0, 1.0]).astype(np.float32)
        else:
            if sensor == 'lidar':
                return np.diag([self.measurement_noise_lidar, self.measurement_noise_lidar, self.measurement_noise_lidar]).astype(np.float32)
            elif sensor == 'hsv_3d':
                return np.diag([self.measurement_noise_hsv_3d, self.measurement_noise_hsv_3d, self.measurement_noise_hsv_3d]).astype(np.float32)
            elif sensor == 'yolo_3d':
                return np.diag([self.measurement_noise_yolo_3d, self.measurement_noise_yolo_3d, self.measurement_noise_yolo_3d]).astype(np.float32)
            else:
                return np.diag([1.0, 1.0, 1.0]).astype(np.float32)

    def publish_position(self):
        """Publish the fused position."""
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.reference_frame
        msg.point.x = self.state[0]
        msg.point.y = self.state[1]
        msg.point.z = self.state[2]
        self.position_pub.publish(msg)

    def publish_velocity(self):
        """Publish the velocity."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.reference_frame
        msg.twist.linear.x = self.state[3]
        msg.twist.linear.y = self.state[4]
        msg.twist.linear.z = self.state[5]
        self.velocity_pub.publish(msg)

    def publish_state(self):
        """
        Publish the current state estimate using the GroundPositionFilter as a second stage.
        This ensures all published positions adhere to ground-only basketball movement constraints.
        """
        # Skip if not active
        if not self.is_activated:
            return
            
        # Create current position as 3D point from our 4D state
        current_pos = [float(self.state[0]), float(self.state[1]), float(self.basketball_z_height)]
        current_time = time.time()
        
        # Check if we're in 2D-only mode and need to handle distance estimation differently
        using_2d_only = True
        for sensor in ['lidar', 'hsv_3d', 'yolo_3d']:
            if current_time - self.last_detection_time.get(sensor, 0) < 1.0:
                using_2d_only = False
                break
        
        # For 2D-only mode with YOLO data, perform distance estimation from bounding box
        estimated_pos = None
        if using_2d_only and current_time - self.last_detection_time.get('yolo_2d', 0) < 1.0:
            # Check if we have bbox data for distance estimation
            if 'yolo_2d' in self.bbox_data and current_time - self.bbox_data['yolo_2d'].get('timestamp', 0) < 1.0:
                # Use bounding box to estimate distance
                bbox_width = self.bbox_data['yolo_2d'].get('width', 0)
                bbox_height = self.bbox_data['yolo_2d'].get('height', 0)
                
                if bbox_width > 0 and bbox_height > 0:
                    # Known basketball diameter in meters
                    basketball_diameter_meters = self.basketball_radius * 2
                    
                    # Use width for horizontal field of view (assuming camera is calibrated)
                    # This is a simplified model that assumes camera focal length is known
                    # We'd use a more accurate model in production with actual camera parameters
                    horizontal_fov_degrees = 70.0  # Typical camera horizontal FOV
                    image_width_pixels = 640  # Typical camera resolution width
                    
                    # Calculate distance based on apparent size vs actual size
                    # distance = (actual_size * focal_length) / apparent_size
                    focal_length_pixels = 345.58  # Calibrated focal length for this camera
                    estimated_distance = (basketball_diameter_meters * focal_length_pixels) / bbox_width
                    
                    # Get the last known YOLO detection for direction
                    yolo_detection = self.sensor_buffer.get_latest_measurement('yolo_2d')
                    if yolo_detection is not None:
                        # Get camera to reference frame transform
                        try:
                            transform = self.tf_buffer.lookup_transform(
                                self.reference_frame,
                                'ascamera_color_0',  # Frame of the YOLO camera
                                rclpy.time.Time(),
                                rclpy.duration.Duration(seconds=0.2)
                            )
                            
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
                            detection_x = yolo_detection.point.x  # X pixel coordinate in image
                            detection_y = yolo_detection.point.y  # Y pixel coordinate in image
                            
                            # Calculate offsets from center of image
                            offset_x = detection_x - image_center_x
                            offset_y = detection_y - image_center_y
                            
                            # FIXED CAMERA COORDINATE SYSTEM MAPPING:
                            # Camera coordinate system needs to map correctly to robot frame
                            # In this robot's setup, the camera's:
                            #   - Z axis points forward (not X as we previously assumed)
                            #   - X axis points right
                            #   - Y axis points down
                            
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
                            
                            self.get_logger().info(f"Received yolo_2d detection: ({detection_x:.2f}, {detection_y:.2f}, {yolo_detection.point.z:.2f}) in {yolo_detection.header.frame_id} frame")
                            
                            # Log the camera direction vector for debugging
                            if hasattr(self, 'debug_level') and self.debug_level >= 2:
                                self.get_logger().debug(f"Camera direction vector from pixel ({detection_x:.1f}, {detection_y:.1f}): vector=({camera_dir_x:.2f}, {camera_dir_y:.2f}, {camera_dir_z:.2f})")
                            
                            # Extract rotation quaternion
                            qx = transform.transform.rotation.x
                            qy = transform.transform.rotation.y
                            qz = transform.transform.rotation.z
                            qw = transform.transform.rotation.w
                            
                            # Convert quaternion to rotation matrix to transform direction vector
                            # This is a simplified quaternion to rotation calculation
                            # Full implementation would use proper quaternion conversion
                            norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
                            qw /= norm
                            qx /= norm
                            qy /= norm
                            qz /= norm
                            
                            # Convert to rotation matrix elements (simplified)
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
                            
                            estimated_pos = [est_x, est_y, est_z]
                            
                            # Log this special calculation occasionally
                            if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 10 == 0:
                                self.get_logger().debug(
                                    f"Estimated 3D position from YOLO 2D: distance={estimated_distance:.2f}m, "
                                    f"pos=({est_x:.2f}, {est_y:.2f}, {est_z:.2f})"
                                )
                        except Exception as e:
                            if self.debug_level >= 2:
                                self.get_logger().warn(f"Could not list frames: {str(e)}")
        
        # IMPROVEMENT: Pass through GroundPositionFilter as second stage
        # If we have an estimated position from 2D data, use that instead of Kalman state
        filtered_pos = None
        if estimated_pos is not None:
            # Use the estimated position from 2D data
            filtered_pos = self.ground_filter.update(estimated_pos, current_time)
            
            # Update Kalman state with this new estimate to keep everything in sync
            self.state[0:2] = filtered_pos[0:2]  # Update x,y position
        else:
            # Use normal Kalman filter state
            filtered_pos = self.ground_filter.update(current_pos, current_time)
            
            # Update our state with the filtered position
            # This keeps the Kalman filter state in sync with published positions
            self.state[0:2] = filtered_pos[0:2]  # Update x,y position
        
        # Get velocity from ground filter (more accurate for rolling balls)
        ground_velocity = self.ground_filter.get_velocity()
        
        # Optionally update velocity state from ground filter's estimate
        # Only do this for stronger movements to avoid noise in stationary case
        ground_speed = self.ground_filter.get_speed()
        if ground_speed > 0.1:  # Only use ground filter velocity for significant movement
            # Fix: Only use x,y components of the ground_velocity (which is 3D)
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
        if ground_speed > 0.1:
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
        # Create a diagnostics dictionary
        diag = {
            'filter_health': self.filter_health,
            'transform_health': self.transform_health,
            'sensor_health': self.sensor_health,
            'position_uncertainty': self.position_uncertainty,
            'velocity_uncertainty': self.velocity_uncertainty,
            'last_filter_update_time': self.last_filter_update_time,
            'processing_times': list(self.processing_times),
            'errors': list(self.errors),
            'warnings': list(self.warnings)
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

    # ENHANCEMENT 7: Gap-Aware Covariance Adjustment
    def adjust_covariance_for_gaps(self):
        """
        Adjust the filter covariance based on sensor gap patterns,
        using a refined approach optimized for 4D state.
        NOW CONSIDERS REJECTED MEASUREMENTS.
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
        rejection_threshold = 3 # Consider sensor 'gapped' after this many consecutive rejections

        for sensor, gap_info in self.sensor_gap_detection.items():
            last_time = self.last_detection_time.get(sensor, 0)
            gap_duration = current_time - last_time

            # Skip sensors we've never seen
            if self.sensor_counts.get(sensor, 0) == 0:
                continue

            sensor_count += 1

            # --- BEGIN MODIFICATION ---
            # Check for consecutive rejections
            rejections = self.consecutive_rejections_per_sensor.get(sensor, 0)
            is_rejected = rejections >= rejection_threshold
            is_gap = gap_duration > 0.5 # Standard gap definition

            gap_level = 0.0
            contributes_to_gap = False

            if is_gap:
                # Calculate gap level based on duration
                if gap_duration < 0.1:
                    gap_level = 0.0
                elif gap_duration < 1.5: # Extend time range for gap level calculation
                    gap_level = gap_duration / 1.5
                else:
                    gap_level = 1.0
                contributes_to_gap = True
            elif is_rejected:
                # Assign a moderate gap level for rejected sensors even if data is recent
                gap_level = 0.5 + (min(rejections, 10) * 0.05) # Increase gap level slightly with more rejections, capped
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
            # --- END MODIFICATION ---

        # Calculate average gap level (considering actual gaps and rejections)
        avg_gap_level = total_gap_level / max(1, sensor_count)

        # IMPROVEMENT 7: Slower covariance growth for stationary objects
        motion_state = self.detect_motion_state()

        # FIX: Use more conservative growth rates based on *effective* sensor availability
        if active_3d_sensors >= 2:
            growth_rate = 1.02 + (avg_gap_level * 0.10)
        elif active_3d_sensors == 1:
            growth_rate = 1.05 + (avg_gap_level * 0.15)
        elif active_2d_sensors > 0:
            growth_rate = 1.08 + (avg_gap_level * 0.20)
        else: # No effectively active sensors (all gapped or rejected)
            growth_rate = 1.10 + (avg_gap_level * 0.40)

        # --- BEGIN MODIFICATION ---
        # Further increase growth rate if many sensors are being rejected
        if rejected_sensor_count >= 2:
             growth_rate *= (1.0 + 0.05 * rejected_sensor_count) # Boost growth by 5% per rejected sensor (capped implicitly)
             if self.debug_level >= 2:
                 self.get_logger().debug(f"Boosting growth rate due to {rejected_sensor_count} rejected sensors: new rate={growth_rate:.3f}")
        # --- END MODIFICATION ---

        # IMPROVEMENT 7: Reduce covariance growth for stationary objects
        if motion_state in ["stationary", "long_stationary"]:
            # ... existing code to reduce growth_rate for stationary ...
            growth_rate = 1.0 + ((growth_rate - 1.0) * 0.3)
            if motion_state == "long_stationary":
                growth_rate = 1.0 + ((growth_rate - 1.0) * 0.15)
            if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 30 == 0:
                self.get_logger().debug(f"Reduced covariance growth for {motion_state} object: {growth_rate:.3f}")

        # Apply growth limit based on uncertainties
        # ... existing code ...
        pos_uncertainty = math.sqrt(max(0.0, np.trace(self.covariance[0:2, 0:2]) / 2.0))
        if pos_uncertainty > 0.5:
            growth_rate = min(growth_rate, 1.1)            # Apply growth factor
            if growth_rate > 1.0:
                # Get current position and calculate unit vector for direction
                x, y = self.state[0], self.state[1]
                distance = math.sqrt(x*x + y*y)
                
                # Only apply special handling if ball is not near origin
                if distance > 0.3:
                    # Calculate unit vector components for current direction
                    unit_x, unit_y = x/distance, y/distance
                    
                    # Create rotation matrix for transforming covariance to radial coordinates
                    rot = np.array([
                        [unit_x, unit_y],
                        [-unit_y, unit_x]
                    ], dtype=np.float32)
                    rot_T = rot.T
                    
                    # Transform position covariance to radial coordinates
                    radial_cov = np.dot(np.dot(rot, self.covariance[0:2, 0:2]), rot_T)
                    
                    # Apply growth differently: more for distance, less for direction
                    radial_cov[0, 0] *= growth_rate               # Radial (distance) uncertainty
                    radial_cov[1, 1] *= (1.0 + (growth_rate-1.0) * 0.3)  # Angular uncertainty (much less growth)
                    
                    # Transform back to cartesian coordinates
                    self.covariance[0:2, 0:2] = np.dot(np.dot(rot_T, radial_cov), rot)
                    
                    # Apply normal growth to velocity
                    self.covariance[2:4, 2:4] *= (growth_rate * 1.05)
                else:
                    # For positions near origin, apply standard growth
                    self.covariance[0:2, 0:2] *= growth_rate
                    self.covariance[2:4, 2:4] *= (growth_rate * 1.05)
                
                # Ensure symmetry and minimum values
                self.covariance = 0.5 * (self.covariance + self.covariance.T)
                for i in range(4):
                    self.covariance[i, i] = max(0.01, self.covariance[i, i])
                    
                # Add debug log for significant direction-preserving growth
                if growth_rate > 1.1 and self.debug_level >= 1:
                    self.get_logger().debug(
                        f"Applied direction-preserving covariance growth: rate={growth_rate:.3f}"
                    )

            # FIX: Add motion-specific uncertainty caps to prevent excessive uncertainty
            # ... existing code to get caps ...
            uncertainty_caps = self.get_motion_based_uncertainty_caps(motion_state)
            max_pos_uncertainty = uncertainty_caps["position_uncertainty_cap"]
            max_vel_uncertainty = uncertainty_caps["velocity_uncertainty_cap"]
            # ... existing code to calculate current uncertainties ...
            current_pos_uncertainty = math.sqrt(max(0.0, np.trace(self.covariance[0:2, 0:2]) / 2.0))
            current_vel_uncertainty = math.sqrt(max(0.0, np.trace(self.covariance[2:4, 2:4]) / 2.0))
            # ... existing code to apply caps ...
            if current_pos_uncertainty > max_pos_uncertainty:
                scale = (max_pos_uncertainty / current_pos_uncertainty) ** 2
                self.covariance[0:2, 0:2] *= scale
            if current_vel_uncertainty > max_vel_uncertainty:
                scale = (max_vel_uncertainty / current_vel_uncertainty) ** 2
                self.covariance[2:4, 2:4] *= scale
            self.covariance = 0.5 * (self.covariance + self.covariance.T)

            # Debug log for significant growth
            # --- MODIFIED LOG ---
            if growth_rate > 1.1 and self.debug_level >= 1: # Log even at level 1 if growth is significant
                self.get_logger().debug(
                    f"Applying covariance growth: rate={growth_rate:.3f}, avg_gap_level={avg_gap_level:.2f}, "
                    f"rejected_sensors={rejected_sensor_count}, "
                    f"pos_uncertainty={current_pos_uncertainty:.3f}m, caps=({max_pos_uncertainty:.1f}, {max_vel_uncertainty:.1f})"
                )
            # --- END MODIFIED LOG ---

    # New method to provide motion-based uncertainty caps
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
                "position_uncertainty_cap": 0.5,  # Tight cap for stationary
                "velocity_uncertainty_cap": 1.0
            }
        elif motion_state == "long_stationary":
            return {
                "position_uncertainty_cap": 0.4,  # Even tighter cap for long-term stationary
                "velocity_uncertainty_cap": 0.8
            }
        elif motion_state == "small_movement":
            return {
                "position_uncertainty_cap": 0.8,  # Medium cap for small movements
                "velocity_uncertainty_cap": 1.5
            }
        elif motion_state == "medium_fast":
            return {
                "position_uncertainty_cap": 1.2,  # Higher but still reasonable cap
                "velocity_uncertainty_cap": 2.0
            }
        else:  # unknown or other states
            return {
                "position_uncertainty_cap": 1.0,  # Default caps
                "velocity_uncertainty_cap": 1.8
            }

    # ENHANCEMENT 8: Confidence-Based Tracking
    def update_tracking_status(self):
        """
        Updated method to determine tracking status using a confidence-based approach.
        Allows maintaining tracking at lower thresholds once it's established.
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
            if current_time - last_time < 1.0:  # Consider sensors active within last 1 second
                if sensor.endswith('_2d'):
                    active_2d_sensors += 1
                else:
                    active_3d_sensors += 1
        
        # NEW: Track sensor gap conditions
        all_sensors_gap = (active_3d_sensors == 0 and active_2d_sensors == 0)
        
        # Initialize sensor gap tolerance window tracking if not already present
        if not hasattr(self, 'sensor_gap_window'):
            self.sensor_gap_window = {
                'active': False,
                'start_time': 0.0,
                'previous_reliability': False,
                'tolerance_seconds': 0.8  # Default 0.8 second tolerance window
            }
            
        # Detect start of a new sensor gap
        if all_sensors_gap and not self.sensor_gap_window['active']:
            # Only activate gap mode if we were previously tracking
            if self.tracking_reliable:
                self.sensor_gap_window['active'] = True
                self.sensor_gap_window['start_time'] = current_time
                self.sensor_gap_window['previous_reliability'] = True
                
                # 1. Motion-Aware Base Tolerance
                # Start with the base tolerance from configuration
                base_tolerance = self.sensor_gap_window.get('base_tolerance', 2.0)
                
                # Apply motion-based multipliers instead of hardcoded values
                motion_state = self.detect_motion_state()
                if motion_state == "long_stationary":
                    # For long-stationary objects, allow MUCH longer gaps (5x)
                    tolerance = base_tolerance * 5.0  # 10 seconds for long-stationary
                elif motion_state == "stationary":
                    # For regular stationary, use 2.5x longer gaps
                    tolerance = base_tolerance * 2.5  # 5 seconds for stationary
                else:
                    # For moving objects, use the base tolerance
                    tolerance = base_tolerance
                    
                # Store the adjusted tolerance
                self.sensor_gap_window['tolerance_seconds'] = tolerance
                
                # 2. Implement the Adaptive Tolerance Mechanism
                if self.sensor_gap_window.get('adaptive_enabled', True):
                    max_avg_interval = 0.0
                    for sensor in self.expected_sensors:
                        intervals = self.sensor_update_intervals.get(sensor, [])
                        if intervals:
                            avg_interval = sum(intervals) / len(intervals)
                            max_avg_interval = max(max_avg_interval, avg_interval)
                    
                    # Set tolerance to at least 4x the maximum average update interval
                    adaptive_tolerance = max(tolerance, max_avg_interval * 4.0)
                    
                    # Cap the maximum tolerance at a reasonable value (15 seconds)
                    adaptive_tolerance = min(15.0, adaptive_tolerance)
                    
                    # Use the adaptive tolerance if it's higher than the motion-based tolerance
                    self.sensor_gap_window['tolerance_seconds'] = max(tolerance, adaptive_tolerance)
                
                # 3. Track Sensor-Specific Gap Patterns
                if motion_state in ["stationary", "long_stationary"]:
                    for sensor in self.expected_sensors:
                        # If this sensor has recent data, it's not showing a stationary gap pattern
                        if current_time - self.last_detection_time.get(sensor, 0) < 1.0:
                            continue
                            
                        # If this is a known gap-prone sensor (like lidar during stationary periods)
                        if sensor == 'lidar' or not sensor.endswith('_2d'):
                            # Create a flag for specific sensors if it doesn't exist
                            if not hasattr(self, 'stationary_gap_patterns'):
                                self.stationary_gap_patterns = {}
                            if sensor not in self.stationary_gap_patterns:
                                self.stationary_gap_patterns[sensor] = False
                                
                            # Mark this sensor as having a stationary gap pattern
                            self.stationary_gap_patterns[sensor] = True
                            
                            # Log this pattern recognition occasionally
                            if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 20 == 0:
                                self.get_logger().debug(f"Recognized {sensor} gap pattern during {motion_state} state")
                
                # 4. Apply Very Long Tolerance for Known Gap Patterns
                if motion_state in ["stationary", "long_stationary"] and hasattr(self, 'stationary_gap_patterns'):
                    gap_pattern_sensors = sum(1 for s, has_pattern in self.stationary_gap_patterns.items() if has_pattern)
                    
                    # If most sensors show stationary gap patterns, use a much larger tolerance
                    if gap_pattern_sensors >= 2:  # At least 2 sensors showing the pattern
                        extended_tolerance = 30.0 if motion_state == "long_stationary" else 15.0
                        self.sensor_gap_window['tolerance_seconds'] = max(self.sensor_gap_window['tolerance_seconds'], extended_tolerance)
                        
                        # Log this special adjustment occasionally
                        if self.debug_level >= 1 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 30 == 0:
                            self.get_logger().info(
                                f"Using extended gap tolerance for {motion_state} with known gap patterns: {self.sensor_gap_window['tolerance_seconds']:.1f}s"
                            )
                
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
        
        # IMPROVEMENT 1: Use different thresholds based on tracking state
        # If currently tracking, use more lenient thresholds to maintain tracking
        if self.tracking_reliable:
            pos_threshold = self.position_uncertainty_threshold * 2.0  # Increased from 1.5 to 2.0
            vel_threshold = self.velocity_uncertainty_threshold * 2.0  # Increased from 1.5 to 2.0
        else:
            pos_threshold = self.position_uncertainty_threshold * 1.2  # Add 20% margin for establishing tracking
            vel_threshold = self.velocity_uncertainty_threshold * 1.2  # Add 20% margin
        
        # IMPROVEMENT 4: Apply motion-aware threshold adjustments
        motion_state = self.detect_motion_state()
        if motion_state == "stationary":
            # Be MUCH more lenient with uncertainty thresholds for stationary objects
            # Increased from 2.5x to 3.0x to better maintain tracking during stationary periods
            pos_threshold *= 3.0
            vel_threshold *= 3.0
        elif motion_state == "long_stationary":
            # Even more permissive for long-term stationary objects
            pos_threshold *= 3.5
            vel_threshold *= 3.5
        
        # IMPROVEMENT 7: Special handling for yolo_2d-only mode
        # If only yolo_2d is active, still allow tracking with appropriate thresholds
        if active_3d_sensors == 0 and active_2d_sensors > 0 and self.allow_tracking_with_2d_only:
            # Check if yolo_2d is one of the active sensors
            yolo_2d_active = False
            for sensor in ['yolo_2d']:
                if current_time - self.last_detection_time.get(sensor, 0) < 1.0:
                    yolo_2d_active = True
                    break
                    
            if yolo_2d_active:
                # Use a more lenient threshold when only yolo_2d is available
                pos_threshold *= 1.25
                
                # IMPROVEMENT 4: For stationary objects with yolo_2d only, be even more permissive
                if motion_state in ["stationary", "long_stationary"]:
                    pos_threshold *= 2.0  # Increased from 1.5x to 2.0x for stationary objects with only 2D data
                    
                    # NEW: For long_stationary objects with 2D-only data, consider always reliable
                    if motion_state == "long_stationary":
                        reliable = True  # Override reliability check - trust 2D data for long-stationary objects
                        
                        # Log this special enhancement occasionally
                        if hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 20 == 0:
                            self.get_logger().info(f"Maintaining tracking with 2D-only data for long-stationary object")
                
                # Log this special case occasionally
                if hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 50 == 0:
                    self.get_logger().info(f"Operating in yolo_2d-only mode with adjusted thresholds")
        
        # NEW: Modified reliability assessment with gap tolerance window
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
            # Normal reliability assessment including sensor availability
            reliable = (pos_uncertainty < pos_threshold and 
                        vel_uncertainty < vel_threshold and
                        (active_3d_sensors >= 1 or (active_2d_sensors >= 1 and self.allow_tracking_with_2d_only)))
        
        # Use time-based stability buffer
        if len(self.reliability_buffer) == 0:
            # Initialize buffer if empty
            self.reliability_buffer = deque([reliable] * 3, maxlen=5)
        else:
            # Add newest value
            self.reliability_buffer.append(reliable)
        
        # Analyze buffer for stability
        true_count = sum(1 for r in self.reliability_buffer if r)
        
        # IMPROVEMENT 3: Apply stronger hysteresis to tracking status based on motion state
        # When stationary, be MUCH more reluctant to lose tracking
        if motion_state in ["stationary", "long_stationary"]:
            # For stationary objects: 
            # - Need 2/5 reliable to start tracking (easier to start tracking)
            # - Need 5/5 unreliable to stop tracking (much harder to lose tracking)
            if not self.tracking_reliable and true_count >= 2:
                self.tracking_reliable = True
                if self.last_tracking_state != self.tracking_reliable:
                    self.get_logger().info(
                        f"Tracking started: uncertainty={pos_uncertainty:.3f}m, sensors={active_3d_sensors}(3D)+{active_2d_sensors}(2D)"
                    )
            elif self.tracking_reliable and true_count == 0:  # Need ALL 5/5 unreliable to stop tracking
                # NEW: Add special case for sensor gaps with very low uncertainty
                if all_sensors_gap and pos_uncertainty < (self.position_uncertainty_threshold * 1.5) and motion_state == "long_stationary":
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
            # For moving objects, use standard criteria:
            # Need 3/5 reliable to start tracking, need 4/5 unreliable to stop tracking
            if not self.tracking_reliable and true_count >= 3:  # Need 3/5 reliable to start tracking
                self.tracking_reliable = True
                if self.last_tracking_state != self.tracking_reliable:
                    self.get_logger().info(
                        f"Tracking started: uncertainty={pos_uncertainty:.3f}m, sensors={active_3d_sensors}(3D)+{active_2d_sensors}(2D)"
                    )
            elif self.tracking_reliable and true_count <= 1:  # Need 4/5 unreliable to stop tracking (more conservative)
                self.tracking_reliable = False
                if self.last_tracking_state != self.tracking_reliable:
                    self.get_logger().info(
                        f"Tracking lost: uncertainty={pos_uncertainty:.3f}m, sensors={active_3d_sensors}(3D)+{active_2d_sensors}(2D)"
                    )
                
        self.last_tracking_state = self.tracking_reliable
        return self.tracking_reliable

    def init_motion_state_tracking(self):
        """Initialize motion state tracking variables in one place to avoid duplication."""
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
        
        # Motion state protection
        self.motion_state_protection = {
            'long_stationary_confirmed_time': 0.0,
            'long_stationary_established': False,
            'consecutive_stationary_after_long': 0,
            'min_time_in_long_stationary': 2.0,     # Reduced from 5.0 to 2.0 seconds
            'post_gap_cooldown_active': False,
            'post_gap_cooldown_end': 0.0,
            'post_gap_protected_state': None,
            'last_gap_recovery_time': 0.0,
            'protection_violation_count': 0
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

    def monitor_sensor_health(self):
        """
        Monitor sensor health and detect gaps or degradation in a centralized way.
        This consolidates duplicate gap detection logic found in multiple places.
        """
        current_time = time.time()
        
        # Track overall sensor status
        active_3d_sensors = 0
        active_2d_sensors = 0
        total_gap_level = 0.0
        sensor_count = 0
        
        # Process each expected sensor
        for sensor in self.expected_sensors:
            # Skip sensors we've never seen
            if self.sensor_counts.get(sensor, 0) == 0:
                continue
                
            sensor_count += 1
            last_time = self.last_detection_time.get(sensor, 0)
            gap_duration = current_time - last_time
            
            # Calculate effective gap level
            if gap_duration < 0.1:
                gap_level = 0.0  # Very recent update - no gap
            elif gap_duration < 0.5:
                gap_level = gap_duration / 0.5  # Linear increase to 1.0
            else:
                gap_level = 1.0  # Full gap level
                
            total_gap_level += gap_level
            
            # Count active sensors by type (with recent enough data)
            if gap_duration < 1.0:
                if not sensor.endswith('_2d'):
                    active_3d_sensors += 1
                else:
                    active_2d_sensors += 1
            
            # Update sensor gap detection
            if sensor in self.sensor_gap_detection:
                if gap_duration > 0.5:  # Start tracking gaps after 0.5s
                    # Calculate gap level
                    self.sensor_gap_detection[sensor]['gap_level'] = gap_level
                    
                    # Check if gap just started
                    if gap_level >= 0.5 and not self.sensor_gap_detection[sensor]['gap_detected']:
                        self.sensor_gap_detection[sensor]['gap_detected'] = True
                        self.sensor_gap_detection[sensor]['gap_start_time'] = current_time
                        
                        # Record gap in sensor reliability tracker
                        if hasattr(self, 'sensor_reliability_tracker'):
                            self.sensor_reliability_tracker.record_gap(sensor, gap_duration)
                            
                        self.get_logger().warn(f"{sensor} gap detected (level={gap_level:.2f})")
                else:
                    # Check if sensor just recovered after a gap
                    if self.sensor_gap_detection[sensor]['gap_detected']:
                        total_gap = current_time - self.sensor_gap_detection[sensor]['gap_start_time']
                        self.get_logger().info(f"{sensor} recovered after {total_gap:.1f}s gap")
                        
                        # Record recovery in reliability tracker
                        if hasattr(self, 'sensor_reliability_tracker'):
                            # Add the final gap duration
                            self.sensor_reliability_tracker.record_gap(sensor, total_gap)
                        
                        # Clear gap flag
                        self.sensor_gap_detection[sensor]['gap_detected'] = False
                        self.sensor_gap_detection[sensor]['gap_level'] = 0.0
                        
                        # Store gap in recent gaps history for pattern analysis
                        if 'recent_gaps' in self.sensor_gap_detection[sensor]:
                            self.sensor_gap_detection[sensor]['recent_gaps'].append(total_gap)
        
        # Calculate average gap level
        avg_gap_level = total_gap_level / max(1, sensor_count)
        
        # Update global sensor state
        self.sensor_status = {
            'active_3d_sensors': active_3d_sensors,
            'active_2d_sensors': active_2d_sensors,
            'average_gap_level': avg_gap_level,
            'all_sensors_gap': (active_3d_sensors == 0 and active_2d_sensors == 0)
        }
        
        return self.sensor_status

    def process_velocity_measurements(self, velocities, times=None):
        """
        Process velocity measurements to filter out implausible values.
        
        Args:
            velocities (list): List of velocity vectors
            times (list): Optional list of timestamps for the velocities
            
        Returns:
            tuple: (filtered_velocities, avg_velocity, implausible_detected)
        """
        # Avoid calling detect_motion_state to prevent recursion
        # Instead, use a simple heuristic based on recent velocities
        
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

    def handle_state_transition(self, new_state, current_state=None):
        """
        Centralized handler for motion state transitions with proper logging and protection.
        
        Args:
            new_state (str): The proposed new state
            current_state (str, optional): Current state, or None to use self.motion_state
            
        Returns:
            str: The actual state to use (may be different from new_state if protected)
        """
        if current_state is None and hasattr(self, 'motion_state'):
            current_state = self.motion_state
        elif current_state is None:
            current_state = "unknown"
            
        # If no change, just return current state
        if new_state == current_state:
            return current_state
            
        current_time = time.time()
        
        # Handle transition protection rules
        protected_state = None
        protection_reason = None
        
        # Check cooldown protection if active
        if hasattr(self, 'motion_state_protection') and self.motion_state_protection.get('post_gap_cooldown_active', False):
            if current_time < self.motion_state_protection.get('post_gap_cooldown_end', 0):
                protected_state = self.motion_state_protection.get('post_gap_protected_state')
                protection_reason = "post-gap cooldown active"
        
        # Check special protection for long_stationary state
        if current_state == "long_stationary" and new_state == "stationary":
            # Check if long_stationary is established
            if hasattr(self, 'motion_state_protection') and self.motion_state_protection.get('long_stationary_established', False):
                # Get confidence levels
                long_conf = self.motion_state_confidence.get("long_stationary", 0.5) if hasattr(self, 'motion_state_confidence') else 0.5
                stat_conf = self.motion_state_confidence.get("stationary", 0.5) if hasattr(self, 'motion_state_confidence') else 0.5
                
                # Require significant confidence difference to allow transition
                if long_conf > stat_conf / 3.0:
                    protected_state = "long_stationary"
                    protection_reason = "confidence levels protect long_stationary"
        
        # Ensure minimum time spent in long_stationary state
        if current_state == "long_stationary" and new_state not in ["stationary", "long_stationary", "unknown"]:
            if hasattr(self, 'motion_state_protection'):
                # Check if we've spent minimum time in long_stationary
                time_in_long = current_time - self.motion_state_protection.get('long_stationary_confirmed_time', 0)
                min_time = self.motion_state_protection.get('min_time_in_long_stationary', 5.0)
                
                if time_in_long < min_time:
                    protected_state = "long_stationary"
                    protection_reason = f"minimum time requirement ({time_in_long:.1f}s < {min_time:.1f}s)"
        
        # Apply protection if needed
        actual_state = protected_state if protected_state else new_state
        
        # Record transition for logging and metrics
        if hasattr(self, 'prev_motion_state'):
            self.prev_motion_state = current_state
        
        # Log the transition with appropriate detail
        if protected_state:
            # Log blocked transition if debugging enabled
            if hasattr(self, 'debug_level') and self.debug_level >= 2:
                self.get_logger().debug(
                    f"Protected state transition: {current_state} -> {new_state} blocked, "
                    f"maintaining {protected_state} ({protection_reason})"
                )
                
            # Count protection events
            if hasattr(self, 'motion_state_protection'):
                if 'protection_violation_count' not in self.motion_state_protection:
                    self.motion_state_protection['protection_violation_count'] = 0
                self.motion_state_protection['protection_violation_count'] += 1
        else:
            # Log actual transition
            confidence_str = ""
            if hasattr(self, 'motion_state_confidence'):
                from_conf = self.motion_state_confidence.get(current_state, 0.0)
                to_conf = self.motion_state_confidence.get(new_state, 0.0)
                confidence_str = f", confidence={to_conf:.2f}"
                
            # Calculate velocity for context in the log
            avg_velocity = 0.0
            if hasattr(self, 'velocity_history') and len(self.velocity_history) > 0:
                recent_velocities = list(self.velocity_history)[-5:]
                velocities = [np.linalg.norm(vel) for vel in recent_velocities if isinstance(vel, (list, tuple, np.ndarray))]
                if velocities:
                    avg_velocity = sum(velocities) / len(velocities)
                    
            self.get_logger().info(f"Motion state changed: {current_state} -> {new_state} "
                                  f"(velocity={avg_velocity:.3f}m/s{confidence_str})")
                                
        return actual_state

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
                max_bbox_age = 3.0  # Allow older bbox data for stationary objects
            elif motion_state == "long_stationary":
                max_bbox_age = 5.0  # Even longer for long-term stationary objects
            elif motion_state == "small_movement":
                max_bbox_age = 2.5  # Slightly increased for slow movement
            else:  # medium_fast or unknown
                max_bbox_age = 2.0  # Keep default for fast movement

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
                
                # Log transform details for debugging
                #self.get_logger().info(f"Transform details for {detection_msg.header.frame_id}: translation=[{transform.transform.translation.x:.4f}, {transform.transform.translation.y:.4f}, {transform.transform.translation.z:.4f}]")
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

    def calculate_velocity_consistency(self):
        """
        Calculate how consistent the recent velocity patterns are to determine predictability of motion.
        Returns a damping factor between 0.5-1.0 where higher values indicate more consistent motion.
        """
        # Default value if we don't have enough history
        if not hasattr(self, 'velocity_history') or len(self.velocity_history) < 5:
            return 0.7  # Default mid-range value
        
        # Get recent velocity history (last 5 measurements)
        recent_velocities = list(self.velocity_history)[-5:]
        
        # Filter out invalid velocities
        valid_velocities = []
        for vel in recent_velocities:
            if isinstance(vel, (list, tuple, np.ndarray)) and len(vel) >= 2:
                valid_velocities.append(vel)
        
        # If we don't have enough valid velocities, return default
        if len(valid_velocities) < 3:
            return 0.7
        
        # Calculate velocity magnitudes and directions
        magnitudes = []
        directions = []
        
        for vel in valid_velocities:
            # For 2D velocities (x,y components)
            vx, vy = vel[0], vel[1]
            magnitude = math.sqrt(vx**2 + vy**2)
            
            # Only calculate direction for non-zero velocities
            if magnitude > 0.02:  # Ignore very small magnitudes for direction
                direction = math.atan2(vy, vx)  # Range: -PI to PI
                directions.append(direction)
            
            magnitudes.append(magnitude)
        
        # Calculate statistics about velocity magnitudes
        avg_magnitude = sum(magnitudes) / len(magnitudes)
        if avg_magnitude < 0.01:  # Effectively stationary
            return 0.9  # High consistency for stationary objects
        
        # Calculate std dev of magnitudes
        magnitude_variance = sum((m - avg_magnitude)**2 for m in magnitudes) / len(magnitudes)
        magnitude_std_dev = math.sqrt(magnitude_variance)
        
        # Normalize to 0-1 range (lower std_dev = higher consistency)
        # Avoid division by zero
        if avg_magnitude > 0:
            magnitude_consistency = 1.0 - min(1.0, magnitude_std_dev / avg_magnitude)
        else:
            magnitude_consistency = 1.0
        
        # Calculate direction consistency if we have enough direction data
        if len(directions) >= 3:
            # Calculate circular statistics for directions
            # Convert directions to unit vectors, then average
            x_sum, y_sum = 0, 0
            for direction in directions:
                x_sum += math.cos(direction)
                y_sum += math.sin(direction)
            
            # Calculate mean resultant length (measure of circular variance)
            mean_resultant_length = math.sqrt(x_sum**2 + y_sum**2) / len(directions)
            
            # Convert to a measure of dispersion (0 = perfectly consistent, 1 = completely random)
            direction_dispersion = 1.0 - mean_resultant_length
            
            # Invert to get consistency (higher is better)
            direction_consistency = 1.0 - direction_dispersion
        else:
            # Not enough direction data
            direction_consistency = 0.5
        
        # Combine with higher weight on direction consistency
        combined_consistency = (0.4 * magnitude_consistency) + (0.6 * direction_consistency)
        
        # Map to damping factor between 0.5-1.0
        damping_factor = 0.5 + (combined_consistency * 0.5)
        
        if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 20 == 0:
            self.get_logger().debug(
                f"Velocity consistency: magnitude={magnitude_consistency:.2f}, direction={direction_consistency:.2f}, "
                f"combined={combined_consistency:.2f}, damping={damping_factor:.2f}"
            )
        
        return damping_factor

    def calculate_blended_motion_factors(self, average_velocity):
        """
        Calculate continuous blending factors between stationary and moving states.
        Returns factors that add up to 1.0 for smooth transitions between motion states.
        
        Args:
            average_velocity (float): The average velocity magnitude
            
        Returns:
            dict: Dictionary with stationary_factor and movement_factor that sum to 1.0
        """
        # Calculate a continuous scale from 0 (stationary) to 1 (fast)
        # Use a threshold of 0.5 m/s as the cutoff for full movement
        velocity_scale = min(1.0, max(0.0, average_velocity / 0.5))
        
        # Calculate motion state factors that add up to 1.0
        stationary_factor = 1.0 - velocity_scale
        movement_factor = velocity_scale
        
        # Occasionally log the factors for debugging
        if self.debug_level >= 2 and hasattr(self, 'sync_quality_metrics') and self.sync_quality_metrics.get('attempt_counts', 0) % 20 == 0:
            self.get_logger().debug(
                f"Motion factors: velocity={average_velocity:.3f}m/s, stationary={stationary_factor:.2f}, "
                f"movement={movement_factor:.2f}"
            )
        
        return {
            "stationary_factor": stationary_factor,
            "movement_factor": movement_factor
        }

    def get_measurement_confidence(self, sensor):
        """
        Get confidence value for a sensor based on recent quality measurements.
        
        Args:
            sensor (str): Sensor name
            
        Returns:
            float: Confidence value between 0.0 and 1.0
        """
        # Default confidence
        confidence = 0.5
        
        # For lidar, use the quality value provided directly in the message
        if sensor == 'lidar' and hasattr(self, 'lidar_quality_history') and len(self.lidar_quality_history) > 0:
            # Use average of recent qualities
            confidence = sum(self.lidar_quality_history) / len(self.lidar_quality_history)
            
            # Ensure it's in 0-1 range
            confidence = min(1.0, max(0.0, confidence))
        
        # For other sensors, use the reliability tracker if available
        elif hasattr(self, 'sensor_reliability_tracker'):
            confidence = self.sensor_reliability_tracker.get_reliability(sensor)
        
        return confidence
    
    
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