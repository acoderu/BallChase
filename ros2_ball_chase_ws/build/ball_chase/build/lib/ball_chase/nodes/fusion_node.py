#!/usr/bin/env python3

"""
Enhanced Fusion Node - Building on Working Foundation
This version starts with the working transform handling and adds advanced features.
"""
import threading
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
from rclpy.executors import MultiThreadedExecutor
# Import from config
from ball_chase.config.config_loader import ConfigLoader


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


class EnhancedFusionNode(Node):
    """
    Enhanced fusion node that builds on our working transform initialization design
    while adding advanced features for better state estimation.
    """
    
    def __init__(self):
        super().__init__('enhanced_fusion_node')
        
        self.get_logger().info("======xxxxx Enhanced Fusion Node Starting ======")
        
        
        # Core tracking variables
        self.start_time = time.time()
        self.transform_available = False
        self.transform_checks = 0
        self.transform_successes = 0
        self.transform_failures = 0
        self.transform_confirmed = False  # Flag to track if transform is permanently confirmed
        self.is_ready = False  # Flag to track if the node is ready for processing
        
        # Use camera_frame as the reference coordinate system instead of map
        self.reference_frame = "camera_frame"
        self.get_logger().info(f"Using {self.reference_frame} as reference coordinate frame for fusion")
        
        # PHASE 1: Initialize transform system FIRST before anything else
        self.init_transform_system()
        
        time.sleep(3)
        # PHASE 1.5: Wait for transform synchronously before proceeding
        self.get_logger().info("Waiting for transform to be available (camera_frame -> lidar_frame)...")
        
        # PHASE 2: Load configuration
        self.load_configuration()
        
        # PHASE 3: Initialize state tracking and buffers
        self.init_state_tracking()
        self.init_sensor_synchronization()
        
        self.transform_check_timer = self.create_timer(5.0, self.check_transform_availability)
        
                
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
    
    def init_transform_system(self):
        """Initialize just the transform system."""
        # CRITICAL STEP: Set up transform system FIRST
        self.tf_buffer = Buffer()  
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Also create a static transform broadcaster in case we need to publish our own
        self.tf_static_broadcaster = StaticTransformBroadcaster(self)
        
        self.get_logger().info("Transform system initialized - waiting for transforms")
    
    def check_transform_availability(self):
        """
        Check if transform is available without trying to fix anything.
        Returns True if transform is available, False otherwise.
        """
        self.transform_checks += 1
        print ("aaaa")
        try:

            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "testA"
            transform.child_frame_id = "testB"
                
            # Values from configuration
            transform.transform.translation.x = 1.0
            transform.transform.rotation.x = 1.0
                
            # Clear any existing transforms with the same parent/child frames
            # This is not directly supported in tf2_ros, but we can ensure a fresh transform
                
            # THIS IS CRITICAL: StaticTransformBroadcaster.sendTransform expects a LIST of transforms
            self.tf_static_broadcaster.sendTransform([transform])
            time.sleep(1)

            # Set up check parameters
            when = rclpy.time.Time()
            timeoutP = rclpy.duration.Duration(seconds=0.1)  # Slightly longer timeout
            parent_frame = "camera_frame"
            child_frame = "lidar_frame"
            
            self.get_logger().info(f"Transform check #{self.transform_checks} at {time.time()-self.start_time:.1f}s")
            
            # Check both directions
            forward_available = False
            reverse_available = False
            
            try:
                forward_available = self.tf_buffer.can_transform(
                    parent_frame,
                    child_frame,
                    when,
                    timeout=timeoutP
                )
            except Exception as e:
                self.get_logger().warn(f"Forward transform check error: {str(e)}")
            
            try:
                reverse_available = self.tf_buffer.can_transform(
                    child_frame,
                    parent_frame,
                    when,
                    timeout=timeoutP
                )
            except Exception as e:
                self.get_logger().warn(f"Reverse transform check error: {str(e)}")
            
            # Consider transform available if either direction works
            self.transform_available = forward_available or reverse_available
            
            if self.transform_available:
                self.transform_successes += 1
                self.get_logger().info(
                    f"✓ Transform check #{self.transform_checks}: Transform available (Forward={forward_available}, Reverse={reverse_available})"
                )
                
                # Display transform details
                if forward_available:
                    transform = self.tf_buffer.lookup_transform(
                        parent_frame,
                        child_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=1.0)
                    )
                    self.get_logger().info(
                        f"Transform details: translation=[{transform.transform.translation.x:.4f}, "
                        f"{transform.transform.translation.y:.4f}, {transform.transform.translation.z:.4f}]"
                    )
                
                # Once we've confirmed the transform is available, we don't need to keep checking
                if not self.transform_confirmed and self.transform_successes >= 2:
                    self.transform_confirmed = True
                    self.get_logger().info("Transform availability confirmed permanently")
            else:
                self.transform_failures += 1
                self.get_logger().warn(f"✗ Transform check #{self.transform_checks}: Transform NOT available")
                                
                # List available frames
                try:
                    frames = self.tf_buffer.all_frames_as_string()
                    if frames and frames.strip():
                        self.get_logger().info(f"Available frames:\n{frames}")
                    else:
                        self.get_logger().info("No frames available in transform buffer")
                except Exception as e:
                    self.get_logger().error(f"Error listing frames: {str(e)}")
            
            if self.transform_confirmed:
                self.transform_check_timer.cancel()
            return self.transform_available
        except Exception as e:
            self.get_logger().error(f"Error checking transform: {str(e)}")
            return False
    
    def load_configuration(self):
        """Load configuration from fusion_config.yaml."""
        try:
            config_loader = ConfigLoader()
            fusion_config = config_loader.load_yaml('fusion_config.yaml')
            
            # Extract topic configuration
            topics = fusion_config.get('topics', {})
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
            
            # Process noise parameters
            self.process_noise_pos = fusion_config.get('process_noise', {}).get('position', 0.1)
            self.process_noise_vel = fusion_config.get('process_noise', {}).get('velocity', 1.0)
            
            # Measurement noise parameters
            measurement_noise = fusion_config.get('measurement_noise', {})
            self.measurement_noise_lidar = measurement_noise.get('lidar', 0.03)
            self.measurement_noise_hsv_3d = measurement_noise.get('hsv_3d', 0.05)
            self.measurement_noise_yolo_3d = measurement_noise.get('yolo_3d', 0.04)
            self.measurement_noise_hsv_2d = measurement_noise.get('hsv_2d', 50.0)
            self.measurement_noise_yolo_2d = measurement_noise.get('yolo_2d', 30.0)
            
            # NEW: Add measurement noise for estimated 3D from 2D
            self.measurement_noise_hsv_2d_est3d = measurement_noise.get('hsv_2d_est3d', 0.15)
            self.measurement_noise_yolo_2d_est3d = measurement_noise.get('yolo_2d_est3d', 0.12)
            
            # Filter parameters
            filter_params = fusion_config.get('filter', {})
            self.max_time_diff = filter_params.get('max_time_diff', 0.2)
            self.min_confidence_threshold = filter_params.get('min_confidence_threshold', 0.5)
            self.detection_timeout = filter_params.get('detection_timeout', 0.5)
            
            # Tracking parameters
            tracking_params = fusion_config.get('tracking', {})
            self.position_uncertainty_threshold = tracking_params.get('position_uncertainty_threshold', 0.5)
            self.velocity_uncertainty_threshold = tracking_params.get('velocity_uncertainty_threshold', 1.0)
            
            # Advanced features
            advanced_features = fusion_config.get('advanced_features', {})
            self.use_bbox_distance_estimation = advanced_features.get('use_bbox_distance_estimation', True)
            self.allow_tracking_with_2d_only = advanced_features.get('allow_tracking_with_2d_only', True)
            self.increased_uncertainty_mode = advanced_features.get('increased_uncertainty_mode', True)
            
            # Diagnostic parameters
            diag_params = fusion_config.get('diagnostics', {})
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
        """Initialize state tracking variables."""
        # Kalman filter state: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        
        # Create initial covariance matrix (uncertainty)
        position_variance = 10.0
        velocity_variance = 100.0
        
        self.covariance = np.eye(6)
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
            'yolo_2d': 0.5
        }
        
        # Pre-allocate filter matrices for efficiency
        self._F_matrix = np.eye(6)  # State transition matrix
        self._Q_matrix = np.zeros((6, 6))  # Process noise matrix
        
        # History collections
        self.position_history = deque(maxlen=self.history_length)
        self.velocity_history = deque(maxlen=self.history_length)
        self.time_history = deque(maxlen=self.history_length)
        self.innovation_history = deque(maxlen=self.history_length)
        
        self.get_logger().info("State tracking variables initialized")
    
    def init_sensor_synchronization(self):
        """Initialize sensor synchronization system."""
        # Create sensor buffer with increased time tolerance (0.5s instead of 0.1s)
        self.sensor_buffer = SensorBuffer(max_time_diff=0.5)
        # ENHANCEMENT 8: Add parent reference for motion state access
        self.sensor_buffer.parent_node = self
        
        # Add sensors to the buffer
        sensor_names = ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']
        for sensor in sensor_names:
            self.sensor_buffer.add_sensor(sensor, buffer_size=20)
        
        # Track last detection time for each sensor
        self.last_detection_time = {sensor: 0.0 for sensor in sensor_names}
        self.sensor_counts = {sensor: 0 for sensor in sensor_names}
        
        # Add sensor timing statistics for FPS calculation
        self.sensor_frame_times = {sensor: deque(maxlen=30) for sensor in sensor_names}  # Store last 30 frame times
        self.sensor_fps = {sensor: 0.0 for sensor in sensor_names}  # Calculated FPS for each sensor
        
        # Store bounding box information for distance estimation
        self.bbox_data = {
            'hsv_2d': {'width': 30, 'height': 30, 'timestamp': 0.0},
            'yolo_2d': {'width': 30, 'height': 30, 'timestamp': 0.0}
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
        """Set up publishers (these can be created immediately)."""
        # Fused 3D position
        self.position_pub = self.create_publisher(
            PointStamped,
            self.position_topic,
            10
        )
        
        # Velocity publisher
        self.velocity_pub = self.create_publisher(
            TwistStamped,
            self.velocity_topic,
            10
        )
        
        # Status publisher
        self.status_pub = self.create_publisher(
            Bool,
            self.status_topic,
            10
        )
        
        # Uncertainty publisher
        self.uncertainty_pub = self.create_publisher(
            Float32,
            self.uncertainty_topic,
            10
        )
        
        # Diagnostics publisher
        self.diagnostics_pub = self.create_publisher(
            String,
            self.diagnostics_topic,
            10
        )
        
        self.get_logger().info("Publishers initialized")
        self.get_logger().info(f"Publishing to: {self.position_topic}, {self.velocity_topic}, {self.status_topic}, {self.uncertainty_topic}, {self.diagnostics_topic}")
    
    def setup_subscriptions(self):
        """Set up subscriptions (only called after transform is available)."""
        # 3D detections
        self.lidar_sub = self.create_subscription(
            PointStamped,
            self.lidar_topic,
            lambda msg: self.sensor_callback(msg, 'lidar'),
            10
        )
        
        self.hsv_3d_sub = self.create_subscription(
            PointStamped,
            self.hsv_3d_topic,
            lambda msg: self.sensor_callback(msg, 'hsv_3d'),
            10
        )
        
        self.yolo_3d_sub = self.create_subscription(
            PointStamped,
            self.yolo_3d_topic,
            lambda msg: self.sensor_callback(msg, 'yolo_3d'),
            10
        )
        
        # 2D detections
        self.hsv_2d_sub = self.create_subscription(
            PointStamped,
            self.hsv_2d_topic,
            lambda msg: self.sensor_callback(msg, 'hsv_2d'),
            10
        )
        
        self.yolo_2d_sub = self.create_subscription(
            PointStamped,
            self.yolo_2d_topic,
            lambda msg: self.sensor_callback(msg, 'yolo_2d'),
            10
        )
        
        # NEW: Bounding box subscriptions for distance estimation 
        # Note: You'll need to import the appropriate message type for bounding boxes
        # This is just a placeholder assuming BoundingBox message type
        try:
            from vision_msgs.msg import BoundingBox2D
            
            self.hsv_bbox_sub = self.create_subscription(
                BoundingBox2D,
                self.hsv_bbox_topic,
                lambda msg: self.bbox_callback(msg, 'hsv_2d'),
                10
            )
            
            self.yolo_bbox_sub = self.create_subscription(
                BoundingBox2D,
                self.yolo_bbox_topic,
                lambda msg: self.bbox_callback(msg, 'yolo_2d'),
                10
            )
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
        # Skip if not ready
        if not self.is_ready:
            self.get_logger().info(f"Status: Node not ready yet, waiting for transform (elapsed time: {time.time() - self.transform_wait_start:.1f}s)")
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
        self.status_timer = self.create_timer(1.0, self.publish_status)
        
        # Kalman filter update timer (20 Hz)
        self.filter_timer = self.create_timer(0.05, self.filter_update)
        
        # Diagnostics timer (1 Hz)
        self.diagnostics_timer = self.create_timer(1.0, self.publish_diagnostics)
        
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
        # Skip if not ready yet
        if not self.is_ready:
            self.get_logger().warn(f"Received {source} data before node was ready - ignoring")
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
        print ("YOLO CBBBB")
        # Skip if not ready yet
        if not self.is_ready:
            return
            
        try:
            # Extract width and height from the bounding box message
            # Note: Adjust this based on your actual message structure
            width = msg.size_x  # Assuming these fields exist in your BoundingBox2D message
            height = msg.size_y
            
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
        avg_velocity = 0.0
        for vel in recent_velocities:
            avg_velocity += np.linalg.norm(vel)
        avg_velocity /= len(recent_velocities)
        
        # Classify based on thresholds
        if avg_velocity < 0.03:
            motion_state = "stationary"
        elif avg_velocity < 0.25:
            motion_state = "small_movement"
        else:
            motion_state = "medium_fast"
        
        # Update motion state counts for stability
        self.motion_state_counts[motion_state] += 1
        for state in self.motion_state_counts:
            if state != motion_state:
                self.motion_state_counts[state] = max(0, self.motion_state_counts[state] - 1)
        
        # Get the most frequent state for stability
        dominant_state = max(self.motion_state_counts, key=self.motion_state_counts.get)
        
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
        recent_z_values = [pos[2] for pos in list(self.position_history)[-10:]]
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
            
        self.flat_ground_detected = self.flat_ground_count > 5
        
        # Apply gentle constraints when flat ground movement is detected
        if self.flat_ground_detected:
            # Dampen vertical velocity
            self.state[5] *= 0.7
            
            # Apply mild constraint toward average height
            if motion_state != "medium_fast":  # Don't constrain during fast motion
                avg_z = sum(recent_z_values) / len(recent_z_values)
                z_diff = avg_z - self.state[2]
                self.state[2] += z_diff * 0.1  # Gentle pull toward average
                
            if self.debug_level >= 2 and self.sync_quality_metrics['attempt_counts'] % 20 == 0:
                self.get_logger().debug(
                    f"Flat ground movement detected (count={self.flat_ground_count}). "
                    f"z_variance={z_variance:.6f}, z_range={z_range:.3f}, avg_z={avg_z:.3f}"
                )

    # ENHANCEMENT 5: Smart Sensor Recovery
    def handle_sensor_recovery(self):
        """
        Monitor sensor availability patterns and handle recovery after gaps.
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
                    'gap_start_time': 0.0
                }
            
            # Check if sensor just recovered after a gap
            if self.sensor_gap_detection[sensor]['gap_detected']:
                # Get newest measurement
                msg = self.sensor_buffer.get_latest_measurement(sensor)
                if msg is not None and current_time - last_time < 0.5:  # Fresh data
                    total_gap = current_time - self.sensor_gap_detection[sensor]['gap_start_time']
                    self.get_logger().info(f"{sensor} recovered after {total_gap:.1f}s gap")
                    
                    # Temporarily increase covariance for more measurement acceptance
                    self.covariance[0:3, 0:3] *= 1.5
                    
                    # Clear gap flag
                    self.sensor_gap_detection[sensor]['gap_detected'] = False
            
            # Set gap flag if sensor has been quiet for too long
            elif gap_duration > 2.0 and not self.sensor_gap_detection[sensor]['gap_detected']:
                self.sensor_gap_detection[sensor]['gap_detected'] = True
                self.sensor_gap_detection[sensor]['gap_start_time'] = current_time
                self.get_logger().warn(f"{sensor} data gap detected: {gap_duration:.1f}s")

    def filter_update(self):
        """Update the Kalman filter with synchronized measurements."""
        # Skip if not ready
        if not self.is_ready or not self.initialized:
            return
        
        # For performance tracking
        update_start = time.time()
        
        # ENHANCEMENT 1: Update motion state
        motion_state = self.detect_motion_state()
        
        # ENHANCEMENT 5: Handle sensor recovery
        self.handle_sensor_recovery()
        
        # Dynamically adjust which sensor is primary based on recency
        current_time = time.time()
        freshest_3d_sensor = None
        newest_time = 0
        
        # Find the most recently updated 3D sensor
        for sensor in ['lidar', 'hsv_3d', 'yolo_3d']:
            last_time = self.last_detection_time.get(sensor, 0)
            if last_time > newest_time:
                newest_time = last_time
                freshest_3d_sensor = sensor
        
        # Calculate adaptive time thresholds every 10 updates
        if self.sync_quality_metrics['attempt_counts'] % 10 == 0:
            thresholds = self.sensor_buffer.calculate_adaptive_time_thresholds()
            if self.debug_level >= 2:
                threshold_info = ", ".join([f"{s}: {t:.3f}s" for s, t in thresholds.items()])
                self.get_logger().debug(f"Adaptive sync thresholds: {threshold_info}")

        # Get synchronized measurements with the improved method
        sync_data = self.sensor_buffer.find_synchronized_measurements(
            min_sensors=1, 
            primary_sensor=freshest_3d_sensor)
        
        self.sync_quality_metrics['attempt_counts'] += 1
        
        # Log whether synchronized measurements were found
        if sync_data:
            self.get_logger().info(f"Found synchronized data from {len(sync_data)} sensors: {', '.join(sync_data.keys())}")
        elif self.debug_level >= 1:
            self.get_logger().debug("No synchronized measurements found")
        
        # Calculate time since last update (even if no new data)
        current_time = time.time()
        if self.last_update_time is None:
            dt = 0.05  # Default time step (20 Hz)
        else:
            dt = current_time - self.last_update_time
        
        # Clamp dt to reasonable range
        dt = max(0.01, min(dt, 0.2))
        
        # Predict step (move state forward in time) - always do this
        self.predict(dt)
        
        processed_measurements = False
        
        # If we have synchronized data, use it
        if sync_data:
            self.sync_quality_metrics['sync_counts'] += 1
    
            # Calculate average time difference between measurements
            timestamps = []
            for source, msg in sync_data.items():
                if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                    timestamps.append(self._ros_time_to_float(msg.header.stamp))
    
            if len(timestamps) > 1:
                max_diff = max(timestamps) - min(timestamps)
                self.sync_quality_metrics['avg_time_diff'] = (
                    0.9 * self.sync_quality_metrics['avg_time_diff'] + 
                    0.1 * max_diff
                )
    
            # Calculate success rate
            self.sync_quality_metrics['success_rate'] = (
                self.sync_quality_metrics['sync_counts'] / 
                self.sync_quality_metrics['attempt_counts']
            )
    
            # Transform measurements to reference frame
            transformed_data = {}
            
            for source, msg in sync_data.items():
                is_2d = source.endswith('_2d')  # Check if this is a 2D source
                
                # Skip transformation if already in reference frame
                if msg.header.frame_id == self.reference_frame:
                    transformed_data[source] = msg
                    self.get_logger().debug(f"{source} data already in {self.reference_frame} - no transform needed")
                else:
                    transformed = self.transform_point(msg, self.reference_frame, is_2d)
                    if transformed:
                        transformed_data[source] = transformed
            
            # Process measurements
            measurements_processed = self.process_measurements(transformed_data)
            processed_measurements = measurements_processed > 0
        
        # Fallback: If no synchronized data or processing failed, try using latest individual measurements
        if not processed_measurements and current_time - self.last_filter_update_time > 0.5:
            self.get_logger().info("Trying fallback with latest individual measurements")
            # Try to process the most recent measurement from each sensor
            latest_data = {}
            
            # Create sensor status string with timing info
            current_time = time.time()
            sensor_status = []
            for source in ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']:
                if self.sensor_counts.get(source, 0) > 0:  # Only include if we've received data
                    time_since_last = current_time - self.last_detection_time.get(source, 0)
                    fps = self.sensor_fps.get(source, 0.0)
                    sensor_status.append(f"{source}: {time_since_last:.1f}s ago ({fps:.1f} FPS)")
                
                # Try to process the most recent measurement from each sensor
                msg = self.sensor_buffer.get_latest_measurement(source)
                if msg is not None and current_time - self.last_detection_time.get(source, 0) < 1.0:
                    is_2d = source.endswith('_2d')  # Check if this is a 2D source
                    
                    # Skip transformation if already in reference frame
                    if msg.header.frame_id == self.reference_frame:
                        latest_data[source] = msg
                    else:
                        transformed = self.transform_point(msg, self.reference_frame, is_2d)
                        if transformed:
                            latest_data[source] = transformed
            
            # Log sensor status alongside fallback attempts
            if sensor_status:
                self.get_logger().info(f"Sensor timing: {', '.join(sensor_status)}")
            else:
                self.get_logger().info("No sensor data received yet")
            
            if latest_data:
                self.get_logger().info(f"Processing latest data from {len(latest_data)} sensors")
                measurements_processed = self.process_measurements(latest_data)
                processed_measurements = measurements_processed > 0
        
        # If no measurements were processed, increase uncertainty
        if not processed_measurements:
            # If no measurements, increment covariance slightly to show increasing uncertainty
            uncertainty_factor = 1.05
            self.covariance[0:3, 0:3] *= uncertainty_factor
            
            # Log occasionally if we're going a long time without measurements
            if current_time - self.last_filter_update_time > 5.0:
                # Create a detailed sensor status string for debugging
                sensor_details = []
                for sensor in ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']:
                    if self.sensor_counts.get(sensor, 0) > 0:  # Only include if we've received data
                        time_since_last = current_time - self.last_detection_time.get(sensor, 0)
                        fps = self.sensor_fps.get(sensor, 0.0)
                        count = self.sensor_counts.get(sensor, 0)
                        sensor_details.append(f"{sensor}: count={count}, {time_since_last:.1f}s ago, {fps:.1f} FPS")
                
                if sensor_details:
                    self.get_logger().warn(
                        f"No measurements processed for {current_time - self.last_filter_update_time:.1f} seconds. "
                        f"Check if sensors are publishing data. Sensor details: {', '.join(sensor_details)}"
                    )
                else:
                    self.get_logger().warn(
                        f"No measurements processed for {current_time - self.last_filter_update_time:.1f} seconds. "
                        f"No sensor data has been received."
                    )
        else:
            # If measurements were processed, update the last update time
            self.last_filter_update_time = current_time
        
        # ENHANCEMENT 4: Apply flat ground constraints
        self.apply_flat_ground_constraints()
        
        # Update timing
        self.last_update_time = current_time
        
        # Calculate uncertainties
        self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:3, 0:3]) / 3.0)
        self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[3:6, 3:6]) / 3.0)
        
        # Update tracking reliability
        self.update_tracking_reliability()
        
        # Store state in history
        self.position_history.append(np.copy(self.state[0:3]))
        self.velocity_history.append(np.copy(self.state[3:6]))
        self.time_history.append(current_time)
        
        # Publish updated state
        self.publish_state()
        
        # Record processing time for diagnostics
        execution_time = (time.time() - update_start) * 1000  # milliseconds
        self.processing_times.append(execution_time)
        
        # After determining which sensors have fresh data
        # Log when switching to 2D-only mode
        fresh_3d_sensors = 0
        fresh_2d_sensors = 0
        
        for sensor, last_time in self.last_detection_time.items():
            if (current_time - last_time) < self.detection_timeout:
                if sensor.endswith('_2d'):
                    fresh_2d_sensors += 1
                else:
                    fresh_3d_sensors += 1

        if fresh_3d_sensors == 0 and fresh_2d_sensors > 0:
            # Only log this when we first switch to 2D-only mode
            if not hasattr(self, '_last_mode') or self._last_mode != '2d_only':
                self._last_mode = '2d_only'
                self.get_logger().info("Switching to 2D-only tracking mode - using bounding box for distance estimation")
                # Log which 2D sensors are active
                active_2d = [s for s, t in self.last_detection_time.items() 
                             if s.endswith('_2d') and (current_time - t) < self.detection_timeout]
                self.get_logger().info(f"Active 2D sensors: {', '.join(active_2d)}")
        elif fresh_3d_sensors > 0:
            if not hasattr(self, '_last_mode') or self._last_mode != '3d':
                self._last_mode = '3d'
                self.get_logger().info("Using 3D tracking mode")
    
    def predict(self, dt):
        """
        Predict the state forward in time.
        
        Args:
            dt (float): Time step in seconds
        """
        # Update state transition matrix F
        self._F_matrix[0, 3] = dt  # x += vx*dt
        self._F_matrix[1, 4] = dt  # y += vy*dt
        self._F_matrix[2, 5] = dt  # z += vz*dt
        
        # Get adaptive process noise values based on motion state
        adaptive_noise_pos, adaptive_noise_vel = self.update_adaptive_process_noise()
        
        # Update process noise matrix Q with adaptive values
        # Position noise grows with dt²
        self._Q_matrix[0:3, 0:3] = np.eye(3) * adaptive_noise_pos * dt**2
        # Velocity noise grows with dt
        self._Q_matrix[3:6, 3:6] = np.eye(3) * adaptive_noise_vel * dt
        
        # Predict state: x = Fx
        self.state = self._F_matrix @ self.state
        
        # Predict covariance: P = FPF' + Q
        self.covariance = self._F_matrix @ self.covariance @ self._F_matrix.T + self._Q_matrix
    
    def process_measurements(self, measurements):
        """
        Process multiple measurements from different sensors.
        
        Args:
            measurements (dict): Dictionary of {sensor_name: measurement}
            
        Returns:
            int: Number of measurements successfully processed
        """
        if not measurements:
            return 0
            
        processed_count = 0
        
        # ENHANCEMENT 1: Get current motion state
        motion_state = self.detect_motion_state()
        
        # Sort measurements to prioritize 3D sensors (more accurate)
        # Process in this order: lidar, 3D vision sensors, 2D sensors
        priority_order = ['lidar', 'hsv_3d', 'yolo_3d', 'hsv_2d', 'yolo_2d']
        
        sorted_measurements = sorted(
            measurements.items(),
            key=lambda x: priority_order.index(x[0]) if x[0] in priority_order else 999
        )
        
        for source, msg in sorted_measurements:
            try:
                success = False
                
                # Check if this is a 2D source
                if source.endswith('_2d'):
                    # For 2D measurements, use the confidence (z) to filter low confidence detections
                    confidence = float(msg.point.z)
                    if confidence >= self.min_confidence_threshold:
                        success = self.update_2d(msg, source, motion_state)
                    else:
                        self.get_logger().debug(f"Skipping low confidence {source} measurement: {confidence:.2f} < {self.min_confidence_threshold:.2f}")
                else:
                    # For 3D measurements, always process
                    success = self.update_3d(msg, source, motion_state)
                
                if success:
                    processed_count += 1
                    
                    # If we get a successful update, update the consecutive updates counter
                    self.consecutive_updates += 1
                
            except Exception as e:
                self.log_error(f"Error processing {source} measurement: {str(e)}")
        
        # If no measurements were successfully processed, reset consecutive updates counter
        if processed_count == 0 and self.consecutive_updates > 0:
            self.consecutive_updates = 0
            self.get_logger().debug("No measurements processed - resetting consecutive updates counter")
        
        return processed_count

    def estimate_distance_from_bbox(self, bbox_width, bbox_height, source):
        """
        Estimate distance to the ball based on 2D bounding box size.
        Uses a simple inverse relationship between distance and size.
        
        Args:
            bbox_width (float): Width of the bounding box in pixels
            bbox_height (float): Height of the bounding box in pixels
            source (str): Source of the detection ('yolo_2d' or 'hsv_2d')
            
        Returns:
            float: Estimated distance in meters
            float: Confidence of the estimate (0.0-1.0)
        """
        # Use the area of the box (width * height) for better distance estimation
        box_area = bbox_width * bbox_height
        
        # Adjust parameters based on the source
        if source == 'yolo_2d':
            # YOLO provides more reliable boxes, so we use better parameters
            # Increase confidence in YOLO bbox-based distance estimates
            scale_factor = 45000.0
            min_distance = 0.3
            max_distance = 8.0
            min_box_area = 200  # Minimum reliable detection size
            # Higher base confidence for YOLO detections
            base_confidence = 0.8  # Increased from implicit 0.7
        else:  # hsv_2d
            # HSV detections might be less accurate
            scale_factor = 35000.0
            min_distance = 0.2
            max_distance = 6.0
            min_box_area = 300 
            base_confidence = 0.7  # Default base confidence
        
        # Safety check for very small boxes (likely noise)
        if box_area < min_box_area:
            return max_distance, 0.1  # Low confidence for very small boxes
        
        # Calculate distance using inverse relationship with area
        # Distance ∝ 1/√(area) for a spherical object like a ball
        estimated_distance = scale_factor / box_area
        
        # Clamp distance to reasonable range
        estimated_distance = max(min_distance, min(estimated_distance, max_distance))
        
        # Calculate confidence - higher for mid-range distances, lower for extremes
        if estimated_distance < 1.0:
            confidence = base_confidence * (estimated_distance / 1.0)
        elif estimated_distance > 5.0:
            confidence = base_confidence * (1.0 - ((estimated_distance - 5.0) / 3.0))
        else:
            confidence = base_confidence  # Use the source-specific base confidence
        
        self.get_logger().debug(
            f"Distance estimate from {source}: Box {bbox_width}x{bbox_height} (area={box_area}) "
            f"-> distance={estimated_distance:.2f}m, confidence={confidence:.2f}"
        )
        
        return estimated_distance, confidence

    def update_2d(self, msg, source, motion_state="unknown"):
        """
        Update the filter with a 2D measurement (x,y and optional distance estimation).
        
        Args:
            msg (PointStamped): The 2D measurement
            source (str): Sensor source identifier ('yolo_2d' or 'hsv_2d')
            motion_state (str): Current motion state
                
        Returns:
            bool: True if update was successful, False otherwise
        """
        # Extract position and confidence
        x_meas = float(msg.point.x)
        y_meas = float(msg.point.y)
        confidence = float(msg.point.z)
        
        # Check if we should attempt a 3D update with estimated distance
        if self.use_bbox_distance_estimation and source in self.bbox_data:
            try:
                # Get the bounding box data
                bbox_data = self.bbox_data[source]
                bbox_width = bbox_data['width']
                bbox_height = bbox_data['height']
                bbox_timestamp = bbox_data['timestamp']
                
                # Check if the bounding box data is fresh enough (within 0.5 seconds)
                current_time = time.time()
                if current_time - bbox_timestamp < 0.5:
                    # Get distance estimate
                    distance, distance_confidence = self.estimate_distance_from_bbox(
                        bbox_width, bbox_height, source
                    )
                    
                    # If we have a reasonable distance estimate, do a full 3D update
                    # Lower the confidence threshold to use distance estimates more often
                    if distance_confidence > 0.15:  # Reduced from 0.2 to use more distance estimates
                        # Create an estimated 3D position
                        # Use simple projective geometry to estimate z from x,y and distance
                        # This is a simplification - in a real scenario you'd use proper camera calibration
                        
                        # Estimate z position
                        # For simplicity, we assume the ball is on the ground or at a fixed height
                        # In reality, you might use more sophisticated projection methods
                        z_meas = 0.3  # Default height (e.g., ball radius above ground)
                        
                        # Check if we can use current state's z as a starting point
                        if self.initialized:
                            # Mix current state's z with a small default value
                            z_meas = 0.8 * self.state[2] + 0.2 * 0.3
                        
                        # Create measurement matrix for 3D update
                        H = np.zeros((3, 6))
                        H[0, 0] = 1.0  # x position
                        H[1, 1] = 1.0  # y position
                        H[2, 2] = 1.0  # z position
                        
                        # Measurement vector
                        z = np.array([x_meas, y_meas, z_meas])
                        
                        # Expected measurement based on current state
                        z_pred = H @ self.state
                        
                        # Innovation (measurement - prediction)
                        innovation = z - z_pred
                        
                        # Get base noise level for this estimated 3D from 2D sensor
                        base_noise = getattr(self, f"measurement_noise_{source}_est3d")
                        
                        # Adjust noise based on confidence and distance estimation uncertainty
                        adjusted_noise_xy = base_noise * (1.0 + (1.0 - confidence))
                        adjusted_noise_z = base_noise * 3.0  # Much higher noise for estimated z
                        
                        # For YOLO specifically, trust the distance estimates more
                        if source == 'yolo_2d':
                            # Reduce noise for YOLO distance estimates due to their reliability
                            adjusted_noise_xy = base_noise * 0.9 * (1.0 + (1.0 - confidence))
                            adjusted_noise_z = base_noise * 2.5  # Reduced from 3.0 for more trust
                        else:
                            adjusted_noise_xy = base_noise * (1.0 + (1.0 - confidence))
                            adjusted_noise_z = base_noise * 3.0
                        
                        # Measurement noise matrix - different noise for xy vs z
                        R = np.diag([adjusted_noise_xy, adjusted_noise_xy, adjusted_noise_z])
                        
                        # Innovation covariance
                        S = H @ self.covariance @ H.T + R
                        
                        try:
                            # Compute Mahalanobis distance for outlier detection
                            S_inv = np.linalg.inv(S)
                            mahalanobis_dist = math.sqrt(innovation.T @ S_inv @ innovation)
                            
                            # Store for diagnostics
                            self.innovation_history.append(mahalanobis_dist)
                            
                            # ENHANCEMENT 3: Get dynamic threshold based on motion state and sensor
                            threshold = self.get_innovation_threshold(source, motion_state)
                            
                            # Reject obvious outliers, but log for diagnostics
                            if mahalanobis_dist > threshold:
                                self.get_logger().info(
                                    f"Rejecting {source} estimated 3D update: innovation distance {mahalanobis_dist:.2f} > threshold {threshold:.2f}. "
                                    f"Confidence {confidence:.2f}, measurement [{x_meas:.2f}, {y_meas:.2f}, {z_meas:.2f}], "
                                    f"distance estimate: {distance:.2f}m (confidence: {distance_confidence:.2f})"
                                )
                                # Fall back to regular 2D update
                                return self._update_2d_only(msg, source, confidence, motion_state)
                                
                            # Debug logging for accepted measurements
                            self.get_logger().info(
                                f"Accepting {source} estimated 3D update: innovation distance {mahalanobis_dist:.2f} < threshold {threshold:.2f}. "
                                f"Confidence {confidence:.2f}, measurement [{x_meas:.2f}, {y_meas:.2f}, {z_meas:.2f}], "
                                f"distance estimate: {distance:.2f}m (confidence: {distance_confidence:.2f})"
                            )
                            
                            # Kalman gain
                            K = self.covariance @ H.T @ S_inv
                            
                            # Update state
                            self.state = self.state + K @ innovation
                            
                            # Update covariance (Joseph form for better numerical stability)
                            I = np.eye(6)
                            self.covariance = (I - K @ H) @ self.covariance @ (I - K @ H).T + K @ R @ K.T
                            
                            # Update sensor reliability
                            reliability_factor = 1.0 - min(1.0, mahalanobis_dist / threshold)
                            # Scale by distance confidence
                            reliability_factor *= distance_confidence
                            self.sensor_reliability[source] = 0.9 * self.sensor_reliability[source] + 0.1 * reliability_factor
                            
                            return True
                            
                        except np.linalg.LinAlgError as e:
                            self.log_error(f"Matrix inversion error in update_2d 3D estimation mode for {source}: {str(e)}")
                            # Fall back to regular 2D update
                            return self._update_2d_only(msg, source, confidence, motion_state)
                else:
                    # Bounding box data is too old
                    if self.debug_level >= 2:
                        self.get_logger().debug(
                            f"Bounding box data for {source} is too old: {current_time - bbox_timestamp:.2f}s > 0.5s. Using 2D-only update."
                        )
                    return self._update_2d_only(msg, source, confidence, motion_state)
                    
            except Exception as e:
                self.log_error(f"Error in 3D estimation from {source} 2D data: {str(e)}")
                # Fall back to regular 2D update
                return self._update_2d_only(msg, source, confidence, motion_state)
        
        # Default case: use 2D-only update
        return self._update_2d_only(msg, source, confidence, motion_state)

    def _update_2d_only(self, msg, source, confidence, motion_state="unknown"):
        """
        Original 2D update method (xy only).
        
        Args:
            msg (PointStamped): The 2D measurement
            source (str): Sensor source identifier
            confidence (float): Detection confidence
            motion_state (str): Current motion state
                
        Returns:
            bool: True if update was successful, False otherwise
        """
        # Extract position
        x_meas = float(msg.point.x)
        y_meas = float(msg.point.y)
        
        # For 2D, we only measure x and y position
        H = np.zeros((2, 6))
        H[0, 0] = 1.0  # x position
        H[1, 1] = 1.0  # y position
        
        # Measurement vector
        z = np.array([x_meas, y_meas])
        
        # Expected measurement based on current state
        z_pred = H @ self.state
        
        # Innovation (measurement - prediction)
        innovation = z - z_pred
        
        # Get base noise level for this sensor
        base_noise = getattr(self, f"measurement_noise_{source}")
        
        # Adjust noise based on confidence (higher confidence = lower noise)
        adjusted_noise = base_noise * (1.0 + (1.0 - confidence))
        
        # Measurement noise matrix
        R = np.eye(2) * adjusted_noise
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + R
        
        try:
            # Compute Mahalanobis distance for outlier detection
            S_inv = np.linalg.inv(S)
            mahalanobis_dist = math.sqrt(innovation.T @ S_inv @ innovation)
            
            # Store for diagnostics
            self.innovation_history.append(mahalanobis_dist)
            
            # ENHANCEMENT 3: Get dynamic threshold based on motion state and sensor
            threshold = self.get_innovation_threshold(source, motion_state)
            
            # Reject obvious outliers, but log for diagnostics
            if mahalanobis_dist > threshold:
                self.get_logger().info(
                    f"Rejecting {source} 2D update: innovation distance {mahalanobis_dist:.2f} > threshold {threshold:.2f}. "
                    f"Confidence {confidence:.2f}, measurement [{x_meas:.2f}, {y_meas:.2f}]"
                )
                return False
                
            # Debug logging for accepted measurements
            self.get_logger().info(
                f"Accepting {source} 2D update: innovation distance {mahalanobis_dist:.2f} < threshold {threshold:.2f}. "
                f"Confidence {confidence:.2f}, measurement [{x_meas:.2f}, {y_meas:.2f}]"
            )
            
            # Kalman gain
            K = self.covariance @ H.T @ S_inv
            
            # Update state
            self.state = self.state + K @ innovation
            
            # Update covariance (Joseph form for better numerical stability)
            I = np.eye(6)
            self.covariance = (I - K @ H) @ self.covariance @ (I - K @ H).T + K @ R @ K.T
            
            # Update sensor reliability
            reliability_factor = 1.0 - min(1.0, mahalanobis_dist / threshold)
            self.sensor_reliability[source] = 0.9 * self.sensor_reliability[source] + 0.1 * reliability_factor
            
            return True
            
        except np.linalg.LinAlgError as e:
            self.log_error(f"Matrix inversion error in _update_2d_only for {source}: {str(e)}")
            return False

    def update_3d(self, msg, source, motion_state="unknown"):
        """
        Update the filter with a 3D measurement.
        
        Args:
            msg (PointStamped): The 3D measurement
            source (str): Sensor source identifier
            motion_state (str): Current motion state
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        # Extract position
        x_meas = float(msg.point.x)
        y_meas = float(msg.point.y)
        z_meas = float(msg.point.z)
        
        # For 3D, we measure x, y, and z position
        H = np.zeros((3, 6))
        H[0, 0] = 1.0  # x position
        H[1, 1] = 1.0  # y position
        H[2, 2] = 1.0  # z position
        
        # Measurement vector
        z = np.array([x_meas, y_meas, z_meas])
        
        # Expected measurement based on current state
        z_pred = H @ self.state
        
        # Innovation (measurement - prediction)
        innovation = z - z_pred
        
        # Get base noise for this sensor
        base_noise = getattr(self, f"measurement_noise_{source}")
        
        # Adjust noise based on distance and sensor characteristics
        adjusted_noise = self.adjust_measurement_noise(source, z, base_noise)
        
        # Measurement noise matrix
        R = np.eye(3) * adjusted_noise
        
        # Innovation covariance
        S = H @ self.covariance @ H.T + R
        
        try:
            # Compute Mahalanobis distance for outlier detection
            S_inv = np.linalg.inv(S)
            mahalanobis_dist = math.sqrt(innovation.T @ S_inv @ innovation)
            
            # Store for diagnostics
            self.innovation_history.append(mahalanobis_dist)
            
            # ENHANCEMENT 3: Get dynamic threshold based on motion state and sensor
            threshold = self.get_innovation_threshold(source, motion_state)
            
            # Reject outliers, but log for diagnostics
            if mahalanobis_dist > threshold:
                self.get_logger().info(
                    f"Rejecting {source} update: innovation distance {mahalanobis_dist:.2f} > threshold {threshold:.2f}. "
                    f"Measurement {z}, prediction {z_pred}, distance={np.linalg.norm(innovation):.2f}m"
                )
                return False
                
            # Debug logging for accepted measurements
            self.get_logger().info(
                f"Accepting {source} update: innovation distance {mahalanobis_dist:.2f} < threshold {threshold:.2f}. "
                f"Measurement [{x_meas:.2f}, {y_meas:.2f}, {z_meas:.2f}]"
            )
            
            # Kalman gain
            K = self.covariance @ H.T @ S_inv
            
            # Update state
            self.state = self.state + K @ innovation
            
            # Update covariance (Joseph form for better numerical stability)
            I = np.eye(6)
            self.covariance = (I - K @ H) @ self.covariance @ (I - K @ H).T + K @ R @ K.T
            
            # Ensure covariance remains positive definite
            min_variance = 1e-6
            for i in range(6):
                if self.covariance[i, i] < min_variance:
                    self.covariance[i, i] = min_variance
            
            # Update sensor reliability
            reliability_factor = 1.0 - min(1.0, mahalanobis_dist / threshold)
            self.sensor_reliability[source] = 0.9 * self.sensor_reliability[source] + 0.1 * reliability_factor
            
            return True
            
        except np.linalg.LinAlgError as e:
            self.log_error(f"Matrix inversion error in update_3d for {source}: {str(e)}")
            return False
    
    def adjust_measurement_noise(self, source, position, base_noise):
        """
        Dynamically adjust measurement noise based on position and sensor.
        
        Args:
            source (str): Sensor source identifier
            position (numpy.ndarray): 3D position measurement
            base_noise (float): Base noise level for this sensor
            
        Returns:
            float: Adjusted noise value
        """
        # Calculate distance from origin (typical sensor location)
        distance = np.linalg.norm(position)
        
        # For LIDAR, adjust based on distance
        if source == 'lidar':
            if distance < 1.0:  # Too close
                return base_noise * 2.0
            elif distance < 3.0:  # Optimal range
                return base_noise
            else:  # Getting too far
                return base_noise * (1.0 + 0.2 * (distance - 3.0))
        
        # For depth cameras (3D vision)
        elif source.endswith('_3d'):
            if distance < 0.5:  # Too close
                return base_noise * 3.0
            elif distance < 2.0:  # Good range
                return base_noise * 0.8
            elif distance < 4.0:  # Decent range
                return base_noise
            else:  # Far range
                return base_noise * (1.0 + 0.3 * (distance - 4.0))
        
        # Default case
        return base_noise
    
    # ENHANCEMENT 6: Mode Transition Stabilization
    def update_tracking_reliability(self):
        """Update tracking reliability metrics based on current state.""" 
        # Check if we have fresh data from enough sensors
        current_time = time.time()
        
        # Count recent sensors by type
        fresh_3d_sensors = 0
        fresh_2d_sensors = 0
        
        for sensor, last_time in self.last_detection_time.items():
            if (current_time - last_time) < self.detection_timeout:
                if sensor.endswith('_2d'):
                    fresh_2d_sensors += 1
                else:
                    fresh_3d_sensors += 1
        
        # Update sensor health based on fresh data
        if fresh_3d_sensors >= 1:
            # We have at least one 3D sensor - great!
            target_health = 1.0
        elif fresh_2d_sensors >= 1 and self.allow_tracking_with_2d_only:
            # No 3D sensors but we have 2D sensors and allow 2D-only tracking
            # Increase health from 0.6 to 0.8 to show more confidence in 2D tracking
            target_health = 0.8  # Increased from 0.6 for better 2D-only confidence
        else:
            target_health = 0.0  # Poor health - no usable sensors
        
        # Smoothly adjust health
        self.sensor_health = 0.9 * self.sensor_health + 0.1 * target_health
        
        # Update transform health based on success rate
        if self.transform_checks > 0:
            success_rate = self.transform_successes / self.transform_checks
            self.transform_health = success_rate
        
        # Use more lenient uncertainty threshold initially, tightening over time
        # Start with 3.0m threshold, decrease to position_uncertainty_threshold over time
        time_since_init = current_time - self.last_update_time if self.last_update_time else 0
        adaptive_threshold = self.position_uncertainty_threshold + 2.5 * math.exp(-time_since_init / 30.0)
        
        # Allow for increased uncertainty when using 2D-only data
        if fresh_3d_sensors == 0 and fresh_2d_sensors > 0 and self.increased_uncertainty_mode:
            # Increase the allowable uncertainty if we only have 2D data
            adaptive_threshold *= 1.5
            self.get_logger().debug(f"2D-only mode: increasing allowable uncertainty to {adaptive_threshold:.2f}m")
        
        # Get current motion state
        motion_state = self.detect_motion_state()
        
        # Determine raw reliability based on uncertainty and available sensors
        raw_reliable = (
            self.position_uncertainty < adaptive_threshold and
            self.velocity_uncertainty < self.velocity_uncertainty_threshold * 2.0 and  # More lenient velocity threshold
            ((fresh_3d_sensors >= 1) or                                               # Either have 3D sensors
             (fresh_2d_sensors >= 1 and self.allow_tracking_with_2d_only)) and        # Or have 2D sensors and allow 2D-only
            self.consecutive_updates >= 1  # Only need one consecutive update
        )
        
        # ENHANCEMENT 6: Apply hysteresis buffer for stability
        # Add current state to reliability buffer
        self.reliability_buffer.append(raw_reliable)
        
        # Count true states in the buffer
        true_count = sum(1 for state in self.reliability_buffer if state)
        
        # Need strong evidence to change state
        if true_count >= 3:  # 3+ out of 5 = reliable
            self.tracking_reliable = True
        elif true_count <= 1:  # 0-1 out of 5 = unreliable
            self.tracking_reliable = False
        # If 2 out of 5, maintain previous state (hysteresis)
        
        # Log state transitions
        if hasattr(self, 'last_tracking_state') and self.last_tracking_state != self.tracking_reliable:
            self.get_logger().info(
                f"Tracking state changed to {self.tracking_reliable}, "
                f"motion_state={motion_state}, uncertainty={self.position_uncertainty:.3f}m"
            )
        self.last_tracking_state = self.tracking_reliable
        
        # Log adaptive thresholds occasionally
        if self.debug_level >= 1 and current_time % 10 < 0.1:
            self.get_logger().debug(
                f"Adaptive threshold: position={adaptive_threshold:.2f}m, "
                f"velocity={self.velocity_uncertainty_threshold * 2.0:.2f}m/s, "
                f"3D sensors: {fresh_3d_sensors}, 2D sensors: {fresh_2d_sensors}"
            )
    
    def transform_point(self, point_msg, target_frame, is_2d=False):
        """
        Transform a point from its original frame to the target frame.
        
        Args:
            point_msg (PointStamped): The point to transform
            target_frame (str): Target coordinate frame
            is_2d (bool): Whether this is a 2D point with confidence in z
            
        Returns:
            PointStamped: Transformed point or None if transformation failed
        """
        if point_msg.header.frame_id == target_frame:
            return point_msg  # Already in the right frame
        
        try:
            # For 2D points, save the confidence value before transform
            confidence = None
            if is_2d:
                confidence = point_msg.point.z
            
            # Get the transform
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                point_msg.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=1.0)
            )
            
            # Apply the transform
            transformed_point = do_transform_point(point_msg, transform)
            
            # Restore confidence value for 2D points
            if is_2d and confidence is not None:
                transformed_point.point.z = confidence
            
            # Debug logging (only log occasionally to avoid flooding)
            if self.sensor_counts.get(point_msg.header.frame_id, 0) % 10 == 0:  # Log every 10th message
                self.get_logger().info(
                    f"Transform successful: {point_msg.header.frame_id}→{target_frame}: "
                    f"({point_msg.point.x:.2f},{point_msg.point.y:.2f},{point_msg.point.z:.2f}) → "
                    f"({transformed_point.point.x:.2f},{transformed_point.point.y:.2f},{transformed_point.point.z:.2f})"
                    f"{' (preserving confidence)' if is_2d else ''}"
                )
            
            return transformed_point
            
        except Exception as e:
            self.get_logger().warn(f"Transform error {point_msg.header.frame_id}→{target_frame}: {str(e)}")
            return None

        if not self.tf_buffer.can_transform(target_frame, source_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0)):
            self.get_logger().warn("Transform not available yet, skipping...")
            return
    
    def publish_state(self):
        """Publish the current state estimate."""
        # Generate timestamp
        timestamp = self.get_clock().now().to_msg()
        
        # Position message
        pos_msg = PointStamped()
        pos_msg.header.frame_id = self.reference_frame
        pos_msg.header.stamp = timestamp
        pos_msg.point.x = float(self.state[0])
        pos_msg.point.y = float(self.state[1])
        pos_msg.point.z = float(self.state[2])
        self.position_pub.publish(pos_msg)
        
        # Velocity message
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = self.reference_frame
        vel_msg.header.stamp = timestamp
        vel_msg.twist.linear.x = float(self.state[3])
        vel_msg.twist.linear.y = float(self.state[4])
        vel_msg.twist.linear.z = float(self.state[5])
        self.velocity_pub.publish(vel_msg)
        
        # Uncertainty message
        unc_msg = Float32()
        unc_msg.data = float(self.position_uncertainty)
        self.uncertainty_pub.publish(unc_msg)
    
    # ENHANCEMENT 7: Comprehensive Motion Analytics
    def _calculate_distance_traveled(self, num_positions=20):
        """
        Calculate total distance traveled over last n positions.
        
        Args:
            num_positions (int): Number of recent positions to consider
            
        Returns:
            float: Total distance traveled
        """
        if not hasattr(self, 'position_history') or len(self.position_history) < 2:
            return 0.0
            
        positions = list(self.position_history)[-min(num_positions, len(self.position_history)):]
        
        total_distance = 0.0
        for i in range(1, len(positions)):
            total_distance += np.linalg.norm(positions[i] - positions[i-1])
            
        return total_distance
    
    # ENHANCEMENT 7: Enhanced Diagnostics with Motion Analytics
    def publish_diagnostics(self):
        """Publish detailed diagnostic information."""
        current_time = time.time()
        
        # Calculate velocities
        vel_mag = 0.0
        if self.initialized:
            vel_mag = math.sqrt(self.state[3]**2 + self.state[4]**2 + self.state[5]**2)
        
        # ENHANCEMENT 7: Calculate comprehensive motion metrics
        acceleration = 0.0
        avg_velocity = 0.0
        max_velocity = 0.0
        
        if self.initialized and len(self.velocity_history) > 10:
            # Calculate statistics on recent movement
            recent_velocities = [np.linalg.norm(v) for v in list(self.velocity_history)[-20:]]
            avg_velocity = sum(recent_velocities) / len(recent_velocities)
            max_velocity = max(recent_velocities)
            
            # Calculate acceleration
            if len(self.velocity_history) >= 3 and len(self.time_history) >= 3:
                v1 = self.velocity_history[-3]
                v2 = self.velocity_history[-1]
                t1 = self.time_history[-3]
                t2 = self.time_history[-1]
                
                if t2 > t1:
                    acceleration = np.linalg.norm(v2 - v1) / (t2 - t1)
        
        # Create diagnostic message
        diagnostics = {
            "timestamp": current_time,
            "uptime": current_time - self.start_time,
            "node_state": {
                "is_ready": self.is_ready,
                "transform_available": self.transform_available,
                "transform_confirmed": self.transform_confirmed
            },
            "transform": {
                "available": self.transform_available,
                "success_rate": self.transform_successes / max(1, self.transform_checks),
                "health": self.transform_health
            },
            "filter": {
                "initialized": self.initialized,
                "tracking": self.tracking_reliable,
                "position_uncertainty": float(self.position_uncertainty),
                "velocity_uncertainty": float(self.velocity_uncertainty),
                "position": [float(self.state[0]), float(self.state[1]), float(self.state[2])],
                "velocity": [float(self.state[3]), float(self.state[4]), float(self.state[5])],
                "velocity_magnitude": float(vel_mag),
                "consecutive_updates": self.consecutive_updates,
                "health": self.filter_health
            },
            "sensors": {
                sensor: {
                    "count": count,
                    "age": current_time - self.last_detection_time.get(sensor, 0),
                    "reliability": self.sensor_reliability.get(sensor, 0.0),
                    "fps": self.sensor_fps.get(sensor, 0.0)
                } for sensor, count in self.sensor_counts.items()
            },
            "performance": {
                "avg_process_time": np.mean(self.processing_times) if self.processing_times else 0.0,
                "max_process_time": np.max(self.processing_times) if self.processing_times else 0.0
            },
            "mode": {
                "using_2d_only": not any(current_time - self.last_detection_time.get(s, 0) < 1.0 
                                        for s in ['lidar', 'hsv_3d', 'yolo_3d']),
                "using_bbox_distance": self.use_bbox_distance_estimation,
                "increased_uncertainty": self.increased_uncertainty_mode
            },
            # ENHANCEMENT 7: Add motion analysis metrics
            "motion_analysis": {
                "state": self.motion_state,
                "avg_velocity": float(avg_velocity),
                "max_velocity": float(max_velocity),
                "acceleration": float(acceleration),
                "flat_ground_movement": getattr(self, 'flat_ground_detected', False),
                "z_variance": float(np.var([pos[2] for pos in list(self.position_history)[-10:]]) if len(self.position_history) >= 10 else 0.0),
                "distance_traveled": float(self._calculate_distance_traveled(20))
            }
        }
        
        # Publish as JSON string
        diag_msg = String()
        diag_msg.data = json.dumps(diagnostics)
        self.diagnostics_pub.publish(diag_msg)
        
        # Also log a summary of the current state
        if self.initialized and self.debug_level >= 1:
            pos = self.state[0:3]
            vel = self.state[3:6]
            self.get_logger().info(
                f"State: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]m, "
                f"vel=[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]m/s, "
                f"uncertainty={self.position_uncertainty:.3f}m, "
                f"motion={self.motion_state}"  # Added motion state to log
            )
    
    def log_error(self, message, is_warning=False):
        """
        Log an error or warning with timestamp.
        
        Args:
            message (str): The error message
            is_warning (bool): True if this is a warning, False for error
        """
        timestamp = time.time()
        
        if is_warning:
            self.get_logger().warn(message)
            self.warnings.append({
                "timestamp": timestamp,
                "message": message
            })
        else:
            self.get_logger().error(message)
            self.errors.append({
                "timestamp": timestamp,
                "message": message
            })
            
            # Reduce filter health score temporarily
            self.filter_health = max(0.3, self.filter_health - 0.2)

    # ENHANCEMENT 2: Adaptive Process Noise Model
    def update_adaptive_process_noise(self):
        """
        Adapt process noise based on detected motion state (moving vs stationary).
        Higher process noise when moving, lower when stationary to reduce jitter.
        """
        # Get current motion state
        motion_state = self.detect_motion_state()
        
        # Set noise factors based on motion state
        if motion_state == "stationary":
            position_noise_factor = 0.3
            velocity_noise_factor = 0.2
        elif motion_state == "small_movement":
            position_noise_factor = 0.8
            velocity_noise_factor = 1.0
        elif motion_state == "medium_fast":
            position_noise_factor = 1.5
            velocity_noise_factor = 2.0
        else:  # unknown
            position_noise_factor = 1.0
            velocity_noise_factor = 1.0
        
        # Calculate target noise values
        target_pos_noise = self.process_noise_pos * position_noise_factor
        target_vel_noise = self.process_noise_vel * velocity_noise_factor
        
        # Implement asymmetric smoothing for transitions
        prev_state = getattr(self, 'prev_motion_state', 'unknown')
        
        # Faster adaptation when transitioning to higher speeds
        if (prev_state == "stationary" and motion_state != "stationary") or \
           (prev_state == "small_movement" and motion_state == "medium_fast"):
            alpha = 0.4  # Faster adaptation
        else:
            alpha = 0.2  # Normal adaptation
        
        # Initialize adaptive noise if needed
        if not hasattr(self, 'adaptive_process_noise_pos'):
            self.adaptive_process_noise_pos = self.process_noise_pos
            self.adaptive_process_noise_vel = self.process_noise_vel
        
        # Apply exponential smoothing with asymmetric adaptation
        self.adaptive_process_noise_pos = (1-alpha) * self.adaptive_process_noise_pos + alpha * target_pos_noise
        self.adaptive_process_noise_vel = (1-alpha) * self.adaptive_process_noise_vel + alpha * target_vel_noise
        
        # Log motion state occasionally
        if self.debug_level >= 2 and self.sync_quality_metrics['attempt_counts'] % 20 == 0:
            self.get_logger().debug(
                f"Motion state: {motion_state} (prev={prev_state}), "
                f"adaptive noise: pos={self.adaptive_process_noise_pos:.3f}, vel={self.adaptive_process_noise_vel:.3f}"
            )
        
        return self.adaptive_process_noise_pos, self.adaptive_process_noise_vel

    def _ros_time_to_float(self, timestamp):
        """Convert ROS timestamp to float seconds."""
        return timestamp.sec + timestamp.nanosec / 1e9

    def wait_subscribers_until_transform(self):
        time.sleep(5)
        reTry = 0        
        while reTry < 10:
            if self.transform_confirmed:
                break
            time.sleep(5)
            reTry = reTry + 1

                
        if self.transform_confirmed:
            self.get_logger().info("✓ Transform confirmed - proceeding with initialization")            
            #only subscribe if transform is good
            self.setup_subscriptions()
        else:
            # This should never happen since we exit on timeout, but just in case
            self.get_logger().error("✗ Transform unexpectedly unavailable - exiting")
            self.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

         # PHASE 4: Set up publishers (these can be created immediately)
        self.setup_publishers()
        
        # PHASE 5: Initialize diagnostics
        self.init_diagnostics()
        
        # PHASE 7: Initialize filter with defaults
        self.initialize_filter_with_defaults()
        
        # PHASE 8: Set up processing timers
        self.setup_timers()
        # Mark as ready
        self.is_ready = True

        self.get_logger().info("Initialization complete - node is ready")


def main(args=None):
    """Main function to start the node."""
    rclpy.init(args=args)
    node = EnhancedFusionNode()
    
    # Schedule my_method to run once after 5 seconds
    threading.Timer(5.0, node.wait_subscribers_until_transform).start()

    try:
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Error: {str(e)}")
    finally:        
        node.destroy_node()
        rclpy.shutdown()