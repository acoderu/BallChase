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
from std_msgs.msg import Float32, Bool, String
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


class EnhancedFusionLifecycleNode(LifecycleNode):
    """
    Enhanced fusion node using ROS 2 Lifecycle framework for state management.
    This provides a clear state machine model for managing node transitions.
    """
    
    def __init__(self):
        super().__init__('state_fusion_node')
        
        self.get_logger().info("======xxxxx Enhanced Fusion Lifecycle Node Starting ======")
        
        # Core tracking variables
        self.start_time = time.time()
        self.transform_available = False
        self.transform_checks = 0
        self.transform_successes = 0
        self.transform_failures = 0
        self.transform_confirmed = False  # Flag to track if transform is permanently confirmed
        self.is_ready = False  # Flag to track if the node is ready for processing
        
        # Initialize publishers list with a different name to avoid conflicts
        self._publishers = []
        
        # Lifecycle-specific flags
        self.is_configured = False
        self.is_activated = False
        
        # Use camera_frame as the reference coordinate system instead of map
        self.reference_frame = "base_link"
        self.get_logger().info(f"Using {self.reference_frame} as reference coordinate frame for fusion")
        
        # Lifecycle requirement: Store timers in a list to manage them in lifecycle transitions
        self._timer_list = []  # Changed from self.timers
        self.subscribers = []
        
        # We'll initialize these in on_configure and on_activate
        self.tf_buffer = None
        self.tf_listener = None
        self.tf_static_broadcaster = None
    
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
            self.motion_state = "unknown"
            self.prev_motion_state = "unknown"
            self.motion_state_counts = {
                "stationary": 0,
                "small_movement": 0,
                "medium_fast": 0,
                "unknown": 0
            }
            
            # ENHANCEMENT 4: Initialize flat ground tracking
            self.flat_ground_detected = False
            self.flat_ground_count = 0
            
            # ENHANCEMENT 5: Initialize sensor recovery tracking
            self.sensor_gap_detection = {}
            
            # ENHANCEMENT 6: Initialize reliability buffer
            self.reliability_buffer = deque([False] * 3, maxlen=5)
            self.last_tracking_state = False
            
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
            
            # PHASE 6: Initialize filter with defaults
            self.initialize_filter_with_defaults()
            
            # PHASE 7: Set up processing timers
            self.setup_timers()
            
            # Mark as activated and ready
            self.is_activated = True
            self.is_ready = True
            
            self.get_logger().info("Node activated successfully and is now ready")
            
            return TransitionCallbackReturn.SUCCESS
            
        except Exception as e:
            self.get_logger().error(f"Error during activation: {str(e)}")
            return TransitionCallbackReturn.ERROR
    
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
                
            self.get_logger().info(f"Expected sensor frames: {self.expected_frames}")
            self.get_logger().info(f"Using {self.reference_frame} as reference frame")
                
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

    def init_state_tracking(self):
        """Initialize state tracking variables with 32-bit floats for better Pi 5 performance."""
        # Kalman filter state: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6, dtype=np.float32)
        
        # Create initial covariance matrix (uncertainty)
        if not hasattr(self, 'config') or self.config is None:
            # Use default values if config is not available
            position_variance = 10.0
            velocity_variance = 100.0
            self.get_logger().warn("Config not available in init_state_tracking - using default values")
        else:
            # Get values from config if available
            position_variance = float(self.config.get('initialization', {}).get('position_variance_initial', 10.0))
            velocity_variance = float(self.config.get('initialization', {}).get('velocity_variance_initial', 100.0))
            
        self.covariance = np.eye(6, dtype=np.float32)
        self.covariance[0:3, 0:3] *= position_variance
        self.covariance[3:6, 3:6] *= velocity_variance
        
        # State tracking flags
        self.initialized = False
        self.tracking_reliable = False
        self.last_update_time = None
        self.consecutive_updates = 0
        
        # Uncertainty metrics
        self.position_uncertainty = float('inf')
        self.velocity_uncertainty = float('inf')
        
        # Sensor health tracking
        self.sensor_reliability = {
            'lidar': 0.5,
            'hsv_3d': 0.5,
            'yolo_3d': 0.5,
            'hsv_2d': 0.5,
            'yolo_2d': 0.6  # Slightly higher initial reliability for YOLO 2D
        }
        
        # Store base reliability for reverting after adjustments
        self.base_sensor_reliability = self.sensor_reliability.copy()
        
        # Direction change detection
        self.direction_change_detected = False
        self.last_direction_change_time = 0.0
        
        # Pre-allocate filter matrices for efficiency (using float32)
        self._F_matrix = np.eye(6, dtype=np.float32)  # State transition matrix
        self._Q_matrix = np.zeros((6, 6), dtype=np.float32)  # Process noise matrix
        self._H_matrix_3d = np.zeros((3, 6), dtype=np.float32)  # Measurement matrix for 3D
        self._H_matrix_2d = np.zeros((2, 6), dtype=np.float32)  # Measurement matrix for 2D
        
        # Set up constant components of measurement matrices
        self._H_matrix_3d[0, 0] = 1.0  # x position
        self._H_matrix_3d[1, 1] = 1.0  # y position
        self._H_matrix_3d[2, 2] = 1.0  # z position
        
        self._H_matrix_2d[0, 0] = 1.0  # x position
        self._H_matrix_2d[1, 1] = 1.0  # y position
        
        # History collections with adaptive length for Pi 5 memory management
        self.history_length = max(10, min(100, getattr(self, 'history_length', 100)))  # Bound history length
        self.position_history = deque(maxlen=self.history_length)
        self.velocity_history = deque(maxlen=self.history_length)
        self.time_history = deque(maxlen=self.history_length)
        self.innovation_history = deque(maxlen=self.history_length)
        
        self.get_logger().info(f"State tracking initialized with float32 precision, history length: {self.history_length}")

    def init_sensor_synchronization(self):
        """Initialize sensor synchronization system."""
        # Create sensor buffer with increased time tolerance (0.5s instead of 0.1s)
        self.sensor_buffer = SensorBuffer(max_time_diff=0.5)
        # ENHANCEMENT 8: Add parent reference for motion state access
        self.sensor_buffer.parent_node = self
        
        # Define all expected sensors
        self.expected_sensors = ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']
        
        # Add sensors to the buffer
        for sensor in self.expected_sensors:
            self.sensor_buffer.add_sensor(sensor, buffer_size=20)
        
        # Track last detection time for each sensor
        self.last_detection_time = {sensor: 0.0 for sensor in self.expected_sensors}
        self.sensor_counts = {sensor: 0 for sensor in self.expected_sensors}
        
        # Add sensor timing statistics for FPS calculation
        self.sensor_frame_times = {sensor: deque(maxlen=30) for sensor in self.expected_sensors}  # Store last 30 frame times
        self.sensor_fps = {sensor: 0.0 for sensor in self.expected_sensors}  # Calculated FPS for each sensor
        
        # Store bounding box information for distance estimation
        self.bbox_data = {
            'hsv_2d': {'width': 30, 'height': 30, 'timestamp': 0.0},
            'yolo_2d': {'width': 30, 'height': 30, 'timestamp': 0.0}
        }
        
        # ENHANCEMENT 5: Initialize sensor recovery tracking for all expected sensors
        self.sensor_gap_detection = {}
        for sensor in self.expected_sensors:
            self.sensor_gap_detection[sensor] = {
                'gap_detected': False,
                'gap_start_time': 0.0,
                'gap_level': 0.0  # Track gap severity (0.0-1.0)
            }
        
        self.get_logger().info("Sensor synchronization system initialized")

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
        # Note: You'll need to import the appropriate message type for bounding boxes
        # This is just a placeholder assuming BoundingBox message type
        try:
            from vision_msgs.msg import BoundingBox2D
            
            hsv_bbox_sub = self.create_subscription(
                BoundingBox2D,
                self.hsv_bbox_topic,
                lambda msg: self.bbox_callback(msg, 'hsv_2d'),
                10
            )
            self.subscribers.append(hsv_bbox_sub)
            
            yolo_bbox_sub = self.create_subscription(
                BoundingBox2D,
                self.yolo_bbox_topic,
                lambda msg: self.bbox_callback(msg, 'yolo_2d'),
                10
            )
            self.subscribers.append(yolo_bbox_sub)
        except ImportError:
            self.get_logger().warn("vision_msgs not available - bounding box processing disabled")
        
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
            transform_check_timer = self.create_timer(5.0, self.check_transform_availability)
            self._timer_list.append(transform_check_timer)
        
        self.get_logger().info("Processing timers initialized")

    def initialize_filter_with_defaults(self):
        """Initialize filter with default values."""
        try:
            # Set default state with zero position and velocity
            self.state = np.zeros(6)
            
            # Set initial covariance (high uncertainty since this is a guess)
            self.covariance = np.eye(6)
            self.covariance[0:3, 0:3] *= 1.0  # Position uncertainty
            self.covariance[3:6, 3:6] *= 2.0  # Velocity uncertainty
            
            self.initialized = True
            self.last_update_time = time.time()
            
            # Update uncertainty metrics
            self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:3, 0:3]) / 3.0)
            self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[3:6, 3:6]) / 3.0)
            
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
            msg (PointStamped): The point measurement from sensor
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
            
            # Add to synchronization buffer
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
        except Exception as e:
            self.log_error(f"Error in {source} callback: {str(e)}")
                
    def bbox_callback(self, msg, source):
        """
        Callback for bounding box messages.
        
        Args:
            msg (BoundingBox2D): The bounding box message
            source (str): Source identifier (e.g., 'hsv_2d', 'yolo_2d')
        """
        # Skip if not active yet
        if not self.is_activated:
            return
        
        try:
            # Extract width and height from the bounding box message
            # Handle different possible message field structures
            width = 0
            height = 0
            
            # Try different field names that might exist in BoundingBox2D message
            if hasattr(msg, 'size_x') and hasattr(msg, 'size_y'):
                width = msg.size_x
                height = msg.size_y
            elif hasattr(msg, 'width') and hasattr(msg, 'height'):
                width = msg.width
                height = msg.height
            elif hasattr(msg, 'bbox') and hasattr(msg.bbox, 'size_x') and hasattr(msg.bbox, 'size_y'):
                width = msg.bbox.size_x
                height = msg.bbox.size_y
            elif hasattr(msg, 'bbox') and hasattr(msg.bbox, 'width') and hasattr(msg.bbox, 'height'):
                width = msg.bbox.width
                height = msg.bbox.height
            elif hasattr(msg, 'size') and hasattr(msg.size, 'x') and hasattr(msg.size, 'y'):
                width = msg.size.x
                height = msg.size.y
            else:
                self.get_logger().warn(f"Could not extract width and height from {source} bounding box message")
                return
            
            # Store the bounding box data with timestamp
            if source in self.bbox_data:
                self.bbox_data[source]['width'] = width
                self.bbox_data[source]['height'] = height
                self.bbox_data[source]['timestamp'] = time.time()
                
                if self.debug_level >= 2:
                    self.get_logger().debug(f"Received {source} bbox: {width:.1f}x{height:.1f}")
        except Exception as e:
            self.log_error(f"Error in {source} bbox callback: {str(e)}")

    def initialize_filter_with_measurement(self, msg, source):
        """Initialize the filter with a specific measurement."""
        try:
            # Ensure message is in the reference frame
            if msg.header.frame_id != self.reference_frame:
                transformed = self.transform_point(msg, self.reference_frame, source.endswith('_2d'))
                if transformed is None:
                    self.get_logger().warn(f"Cannot initialize filter - transform failed from {msg.header.frame_id} to {self.reference_frame}")
                    return False
                msg = transformed
            
            # Initialize with this measurement
            self.state[0:3] = [
                msg.point.x,
                msg.point.y,
                msg.point.z
            ]
            self.state[3:6] = [0.0, 0.0, 0.0]  # Start with zero velocity
            
            # Reset covariance with reasonable values
            self.covariance = np.eye(6)
            self.covariance[0:3, 0:3] *= 0.1  # Position uncertainty
            self.covariance[3:6, 3:6] *= 1.0  # Velocity uncertainty
            
            self.initialized = True
            self.last_update_time = time.time()
            
            self.get_logger().info(
                f"Filter initialized with {source} measurement: ({msg.point.x:.2f}, "
                f"{msg.point.y:.2f}, {msg.point.z:.2f})"
            )
            
            # Update position uncertainty for status tracking
            self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:3, 0:3]) / 3.0)
            self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[3:6, 3:6]) / 3.0)
            
            # Start active tracking
            self.get_logger().info("Kalman filter initialized - beginning active tracking")
            
            return True
        except Exception as e:
            self.get_logger().error(f"Error during filter initialization: {str(e)}")
            return False

    # ENHANCEMENT 1: Motion State Detection System
    def detect_motion_state(self):
        """
        Detect the current motion state of the object based on recent velocity history.
        
        Returns:
            str: One of "stationary", "small_movement", "medium_fast", or "unknown"
        """
        # If not initialized or insufficient velocity history, return unknown
        if not self.initialized or len(self.velocity_history) < 5:
            return "unknown"
        
        # Get the most recent velocity estimates
        recent_velocities = list(self.velocity_history)[-5:]
        
        # Calculate the average magnitude of these velocities
        valid_velocities = [vel for vel in recent_velocities if isinstance(vel, (list, tuple, np.ndarray)) and len(vel) >= 3]
        if not valid_velocities:
            return "unknown"
        
        avg_velocity = 0.0
        for vel in valid_velocities:
            avg_velocity += np.linalg.norm(vel)
        avg_velocity /= len(valid_velocities)
        
        # Classify based on thresholds
        if avg_velocity < 0.03:
            motion_state = "stationary"
        elif avg_velocity < 0.25:
            motion_state = "small_movement"
        else:
            motion_state = "medium_fast"
        
        # Update motion state counts for stability
        if not hasattr(self, 'motion_state_counts'):
            self.motion_state_counts = {
                "stationary": 0,
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
                 
        # Log if motion state changes
        if dominant_state != self.motion_state:
            self.prev_motion_state = self.motion_state
            self.motion_state = dominant_state
            self.get_logger().info(f"Motion state changed: {self.prev_motion_state} -> {self.motion_state} (velocity={avg_velocity:.3f}m/s)")
        
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
                "small_movement": (5.0, 2.0),
                "medium_fast": (8.0, 3.0),
                "unknown": (10.0, 3.0)
            },
            "3d_vision": {
                "stationary": (6.0, 2.0),
                "small_movement": (9.0, 3.0),
                "medium_fast": (12.0, 4.0),
                "unknown": (15.0, 5.0)
            },
            "2d": {
                "stationary": (10.0, 3.0),
                "small_movement": (15.0, 5.0),
                "medium_fast": (20.0, 8.0),
                "unknown": (25.0, 10.0)
            }
        }
        
        # Get thresholds for this state and sensor type
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
        Apply gentle constraints based on detected flat ground movement.
        """
        # Need sufficient history to determine flat ground movement
        if not self.initialized or len(self.position_history) < 10:
            self.flat_ground_detected = False
            self.flat_ground_count = 0
            return
            
        # Analyze recent z-dimension behavior
        recent_positions = list(self.position_history)[-10:]
        
        # Ensure positions have z-component before accessing
        if all(len(pos) >= 3 for pos in recent_positions):
            recent_z_values = [pos[2] for pos in recent_positions]
            z_variance = np.var(recent_z_values)
            z_range = max(recent_z_values) - min(recent_z_values)
            
            # Use motion-aware thresholds to detect flat ground movement
            motion_state = self.detect_motion_state()
            z_variance_threshold = 0.0005 if motion_state == "stationary" else 0.002
            
            # Detect flat ground movement
            flat_ground = z_variance < z_variance_threshold and z_range < 0.05
            
            # Update counter and detection state
            if flat_ground:
                self.flat_ground_count = min(self.flat_ground_count + 1, 20)
            else:
                self.flat_ground_count = max(self.flat_ground_count - 1, 0)
                
            # Set detected state based on counter
            self.flat_ground_detected = self.flat_ground_count > 5
            
            if self.flat_ground_detected:
                # Apply strict constraints for a basketball rolling on the ground
                # Fix Z height to a reasonable value for a basketball on the ground
                self.state[2] = 0.08  # Basketball radius above ground
                
                # Zero out vertical velocity for ground rolling
                self.state[5] = 0.0
                
                if hasattr(self, 'debug_level') and hasattr(self, 'sync_quality_metrics') and self.debug_level >= 2 and self.sync_quality_metrics['attempt_counts'] % 20 == 0:
                    self.get_logger().debug(
                        f"Ground rolling ball detected (count={self.flat_ground_count}). "
                        f"z_variance={z_variance:.6f}, z_range={z_range:.3f}, fixing height to 0.08m"
                    )

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
            
            # Update gap level continuously based on time since last detection
            if gap_duration > 0.5:  # Start tracking gaps after 0.5s - Now a separate if statement
                # Calculate gap level on a scale from 0.0 to 1.0
                gap_level = min(1.0, (gap_duration - 0.5) / 1.5)  # 0.5s to 2.0s range
                self.sensor_gap_detection[sensor]['gap_level'] = gap_level
                
                # If gap level exceeds threshold, mark as detected
                if gap_level >= 0.5 and not self.sensor_gap_detection[sensor]['gap_detected']:
                    self.sensor_gap_detection[sensor]['gap_detected'] = True
                    self.sensor_gap_detection[sensor]['gap_start_time'] = current_time
                    self.get_logger().warn(f"{sensor} gap detected (level={gap_level:.2f})")

    def filter_update(self):
        """
        Perform a Kalman filter update based on synchronized sensor measurements.
        """
        # Skip if not active or not initialized
        if not self.is_activated or not self.initialized:
            return
        
        try:
            # Get current time
            current_time = time.time()
            dt = current_time - self.last_update_time
            
            # Find synchronized measurements
            measurements = self.sensor_buffer.find_synchronized_measurements(min_sensors=2)
            
            # If no synchronized measurements, skip update
            if not measurements:
                return
            
            # Predict state forward to current time
            self.predict_state(dt)
            
            # Update state with measurements
            self.update_state(measurements)
            
            # Update last update time
            self.last_update_time = current_time
            
            # Update uncertainty metrics
            self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:3, 0:3]) / 3.0)
            self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[3:6, 3:6]) / 3.0)
            
            # Publish fused position and velocity
            if hasattr(self, 'position_history'):
                self.position_history.append(self.state[0:3].copy())
            if hasattr(self, 'velocity_history'):
                self.velocity_history.append(self.state[3:6].copy())
            if hasattr(self, 'time_history'):
                self.time_history.append(current_time)
            
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
        except Exception as e:
            self.get_logger().error(f"Error during filter update: {str(e)}")

    def predict_state(self, dt):
        """
        Predict the state forward by dt seconds.
        
        Args:
            dt (float): Time step in seconds
        """
        # Reset the state transition matrix to identity first
        self._F_matrix = np.eye(6, dtype=np.float32)
        
        # Then set the time-dependent values
        self._F_matrix[0, 3] = dt
        self._F_matrix[1, 4] = dt
        self._F_matrix[2, 5] = dt
        
        # Reset the process noise matrix to zeros
        self._Q_matrix = np.zeros((6, 6), dtype=np.float32)
        
        # Process noise parameters
        q_pos = self.process_noise_pos * dt
        q_vel = self.process_noise_vel * dt
        
        # Fill in the process noise matrix properly
        # Position variances
        self._Q_matrix[0, 0] = q_pos * dt**3 / 3.0  # x position variance
        self._Q_matrix[1, 1] = q_pos * dt**3 / 3.0  # y position variance
        self._Q_matrix[2, 2] = q_pos * dt**3 / 3.0  # z position variance
        
        # Velocity variances
        self._Q_matrix[3, 3] = q_vel * dt          # x velocity variance
        self._Q_matrix[4, 4] = q_vel * dt          # y velocity variance
        self._Q_matrix[5, 5] = q_vel * dt          # z velocity variance
        
        
        # Position-velocity covariances
        self._Q_matrix[0, 3] = self._Q_matrix[3, 0] = q_pos * dt**2 / 2.0  # x position-velocity
        self._Q_matrix[1, 4] = self._Q_matrix[4, 1] = q_pos * dt**2 / 2.0  # y position-velocity
        self._Q_matrix[2, 5] = self._Q_matrix[5, 2] = q_pos * dt**2 / 2.0  # z position-velocity
        
        # Predict state
        self.state = np.dot(self._F_matrix, self.state)
        
        # Predict covariance
        self.covariance = np.dot(np.dot(self._F_matrix, self.covariance), self._F_matrix.T) + self._Q_matrix
        
        # Ensure covariance remains symmetric after prediction too
        self.covariance = 0.5 * (self.covariance + self.covariance.T)

    def update_state(self, measurements):
        """
        Update the state with synchronized measurements.
        
        Args:
            measurements (dict): Dictionary of {sensor_name: measurement}
        """
        # Store successful update flag to track if any measurements were processed
        successful_update = False
        
        for sensor, msg in measurements.items():
            # Transform measurement to reference frame
            transformed = self.transform_point(msg, self.reference_frame, sensor.endswith('_2d'))
            if not transformed:
                continue
            
            # Measurement vector
            if sensor.endswith('_2d'):
                z = np.array([transformed.point.x, transformed.point.y], dtype=np.float32)
                H = self._H_matrix_2d
                R = self.get_measurement_noise(sensor, is_2d=True)
            else:
                z = np.array([transformed.point.x, transformed.point.y, transformed.point.z], dtype=np.float32)
                H = self._H_matrix_3d
                R = self.get_measurement_noise(sensor, is_2d=False)
            
            # Calculate innovation (measurement residual)
            y = z - np.dot(H, self.state)
            
            # Innovation covariance
            S = np.dot(np.dot(H, self.covariance), H.T) + R
            
            # ENHANCEMENT 3: Apply dynamic measurement validation
            # Get appropriate innovation threshold based on sensor type and motion state
            motion_state = self.detect_motion_state()
            threshold = self.get_innovation_threshold(sensor, motion_state)
            
            # Compute Mahalanobis distance for validation
            try:
                # Use proper dimensionality for S_inv based on sensor type
                S_inv = np.linalg.inv(S)
                mahalanobis_dist = np.sqrt(np.dot(np.dot(y.T, S_inv), y))
                
                # Store innovation for diagnostic purposes
                if hasattr(self, 'innovation_history'):
                    self.innovation_history.append(mahalanobis_dist)
                
                # Skip measurement if it fails validation
                if mahalanobis_dist > threshold:
                    self.get_logger().debug(
                        f"Rejecting {sensor} measurement: innovation {mahalanobis_dist:.2f} > threshold {threshold:.2f}"
                    )
                    continue
                    
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
                
            except np.linalg.LinAlgError:
                self.get_logger().warn(f"Matrix inversion failed during Kalman update for {sensor}")
                continue
        
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
            
            # Get transform from source to target frame with appropriate timeout
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
        """Publish the current state estimate."""
        # Skip if not active
        if not self.is_activated:
            return
        
        # Create position message
        pos_msg = PointStamped()
        pos_msg.header.stamp = self.get_clock().now().to_msg()
        pos_msg.header.frame_id = self.reference_frame
        pos_msg.point.x = self.state[0]
        pos_msg.point.y = self.state[1]
        pos_msg.point.z = self.state[2]
        self.position_pub.publish(pos_msg)
        
        # Create velocity message
        vel_msg = TwistStamped()
        vel_msg.header.stamp = self.get_clock().now().to_msg()
        vel_msg.header.frame_id = self.reference_frame
        vel_msg.twist.linear.x = self.state[3]
        vel_msg.twist.linear.y = self.state[4]
        vel_msg.twist.linear.z = self.state[5]
        self.velocity_pub.publish(vel_msg)

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