#!/usr/bin/env python3

"""
Basketball Tracking Robot - LIDAR Detection Node
===============================================

This node processes 2D LIDAR data to detect a basketball and provide 3D position information.
It correlates LIDAR data with camera-based detections from YOLO and HSV nodes.

Features:
- Processes 2D LIDAR scans to find circular patterns matching a basketball (9-inch diameter)
- Uses YOLO and HSV detections to trigger validation of potential basketball locations
- Publishes the basketball's 3D position in the robot's coordinate frame
- Provides visualization markers for debugging in RViz
- Includes simplified detection algorithms optimized for Raspberry Pi 5

Physical Setup:
- LIDAR mounted 6 inches (15.24 cm) above ground
- Basketball diameter: 9 inches (22.86 cm)
- LIDAR beam intersects basketball at a consistent height
- Basketball rolls on ground only (center is always 4.5 inches above ground)
"""
# Standard library imports
import sys
import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
from collections import deque
import threading
import psutil  # Add this for CPU monitoring

# ROS2 messages
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Float32
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import PointStamped as TF2PointStamped  # Add TF2 point support
import json

import os
# Import GroundPositionFilter for shared ground movement tracking
from ball_chase.config.config_loader import ConfigLoader
from ball_chase.utilities.ground_position_filter import GroundPositionFilter


class BasketballLidarDetector(Node):
    """
    A ROS2 node to detect basketballs using a 2D laser scanner.
    
    Correlates LIDAR data with camera detections to provide 3D position
    information for detected basketballs.
    """
    
    def __init__(self):
        """Initialize the basketball LIDAR detector node."""
        super().__init__('basketball_lidar_detector')
        
        # Load configuration
        self.config_loader = ConfigLoader()
        try:
            self.config = self.config_loader.load_yaml('lidar_config.yaml')
        except Exception as e:
            self.get_logger().error(f"Failed to load config: {str(e)}")
            self.config = {}
        
        # Load performance configuration
        self._load_performance_config()
        
        # Initialize state
        self._init_state()
        
        # Initialize coordinate transform parameters first
        self._init_transform_parameters()
        
        # Set up TF system - Initialize buffer and listener FIRST
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Set up subscribers
        self._setup_subscribers()
        
        # Set up publishers
        self._setup_publishers()
        
        # Set up a periodic timer to check transform availability (for debugging)
        self.transform_check_timer = self.create_timer(5.0, self.check_transform)
        
        # Set up diagnostics timer
        diag_interval = self.config.get('diagnostics', {}).get('publish_interval', 3.0)
        self.diagnostics_timer = self.create_timer(diag_interval, self.publish_diagnostics)
        
        # Load basketball parameters
        self._load_basketball_parameters()
        
        # Create a lock for thread safety
        self.lock = threading.RLock()
        
        # Debug timer - COMMENTED OUT FOR CALIBRATION
        # Uncomment when not doing calibration
        # self.debug_timer = self.create_timer(2.0, self.publish_debug_point)
        
        # NEW: Set up autonomous detection timer for independent ball finding
        autonomous_detection_config = self.config.get('autonomous_detection', {})
        self.autonomous_detection_enabled = autonomous_detection_config.get('enabled', True)
        self.autonomous_interval = autonomous_detection_config.get('interval', 1.0)  # seconds
        self.autonomous_detection_timer = self.create_timer(
            self.autonomous_interval, self.autonomous_detection_callback
        )
        
        # NEW: Track detection modes and confidence
        self.detection_mode = "HYBRID"  # Can be "HYBRID", "INDEPENDENT", or "CAMERA_DEPENDENT"
        self.independent_confidence = 0.5  # Initial confidence in independent detection
        self.last_independent_detection_time = 0.0
        self.last_camera_detection_time = 0.0
        self.recovery_attempts = 0
        self.max_recovery_attempts = autonomous_detection_config.get('max_recovery_attempts', 5)
        
        self.get_logger().info("Basketball LIDAR detector initialized")
        
        # NEW: Create a flag to track successful transforms
        self.transform_published_successfully = False
    
    def _load_performance_config(self):
        """Load performance-related configuration."""
        perf_config = self.config.get('performance', {})
        
        # Visualization settings (disabled by default)
        viz_config = self.config.get('visualization', {})
        self.visualization_enabled = viz_config.get('enabled', False)
        
        # Performance adaptation settings
        self.adaptive_processing = perf_config.get('adaptive_processing', True)
        # Adjusted thresholds with wider gap to prevent frequent switching
        self.high_load_threshold = perf_config.get('high_load_threshold', 85.0)  # CPU % (increased from 80)
        self.low_load_threshold = perf_config.get('low_load_threshold', 40.0)    # CPU % (decreased from 50)
        
        # NEW: Hysteresis parameters for performance mode transitions
        self.mode_transition_hysteresis = perf_config.get('mode_transition_hysteresis', 5.0)  # % hysteresis
        self.mode_stability_time = perf_config.get('mode_stability_time', 10.0)  # seconds to maintain mode
        self.last_mode_change_time = time.time()
        self.mode_samples = deque(maxlen=5)  # Track recent CPU measurements for smoothing
        
        # Processing settings
        self.max_point_limit = perf_config.get('max_point_limit', 500)
        self.dynamic_ransac_iterations = perf_config.get('dynamic_ransac_iterations', True)
        # Increased minimum RANSAC iterations for better quality in all modes
        self.min_ransac_iterations = perf_config.get('min_ransac_iterations', 15)  # Increased from 10
        
        # NEW: Quality protection parameters
        self.min_quality_threshold = perf_config.get('min_quality_threshold', 0.3)  # Minimum acceptable quality
        self.critical_point_radius = perf_config.get('critical_point_radius', 0.5)  # Radius around previous detection
        self.min_critical_points = perf_config.get('min_critical_points', 15)  # Minimum points to keep for detection
        
        # Timer frequencies
        self.diagnostics_interval_normal = self.config.get('diagnostics', {}).get('publish_interval', 3.0)
        self.diagnostics_interval_high_load = perf_config.get('diagnostics_interval_high_load', 10.0)
        
        # System resources
        self.current_cpu_load = 0.0
        self.performance_mode = "NORMAL"  # Can be "NORMAL", "EFFICIENT", "MINIMAL"
        
        # Resource monitoring timer - check every 5 seconds
        self.resource_timer = self.create_timer(5.0, self.monitor_resources)
        
    def _init_state(self):
        """Initialize internal state tracking."""
        # Scan data
        self.latest_scan = None
        self.scan_timestamp = None
        self.scan_frame_id = None
        self.points_array = None
        
        # Performance tracking
        self.start_time = time.time()
        self.processed_scans = 0
        self.successful_detections = 0
        self.detection_times = deque(maxlen=100)
        
        # Detection sources
        self.yolo_detections = 0
        self.hsv_detections = 0
        
        # Position tracking
        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)  # Track recent velocity for consistency checks
        self.previous_ball_position = None
        self.previous_timestamp = None  # Track timestamp for velocity calculation
        self.consecutive_failures = 0
        self.last_successful_detection_time = 0
        self.predicted_position = None
        
        # Initialize ground position filter for shared ground movement tracking
        self.position_filter = GroundPositionFilter()
        
        # Health monitoring
        self.lidar_health = 1.0
        self.detection_health = 1.0
        self.detection_latency = 0.0
        self.errors = deque(maxlen=10)
        self.last_error_time = 0
        
        # NEW: Transform publishing tracking
        self.transform_publish_attempts = 0
        self.transform_publish_successes = 0
        
        # New performance metrics
        self.processing_skips = 0
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        
        # Position smoothing parameters
        self.smoothing_alpha = 0.3  # Smoothing factor (0-1), lower = more smoothing
        self.max_position_jump = 0.5  # Maximum allowed jump in meters between frames
        self.max_speed = 2.0  # Maximum physically plausible speed in m/s
    
    def check_transform(self):
        """Periodically check if transform is available in TF tree."""
        try:
            test_time = rclpy.time.Time()
            
            # First check: Direct transform between frames (most straightforward)
            direct_transform_available = self.tf_buffer.can_transform(
                "lidar_frame",           # Target frame
                "ascamera_color_0",      # Source frame (camera)
                test_time,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Second check: Look for transform via common parent (base_link)
            # Check from camera to base_link
            camera_to_base = self.tf_buffer.can_transform(
                "base_link",
                "ascamera_color_0",
                test_time,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Check from base_link to lidar
            base_to_lidar = self.tf_buffer.can_transform(
                "lidar_frame",
                "base_link",
                test_time,
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            # Combined check (can transform via common parent)
            chain_transform_available = camera_to_base and base_to_lidar
            
            transform_available = direct_transform_available or chain_transform_available
            
            if transform_available:
                # Try to actually get the transform (direct or through chain)
                try:
                    transform = self.tf_buffer.lookup_transform(
                        "lidar_frame",
                        "ascamera_color_0",
                        test_time
                    )
                    self.transform_published_successfully = True
                    self.transform_publish_successes += 1
                    
                    self.get_logger().info(
                        f"✓ Transform check: transform from ascamera_color_0 to lidar_frame is available. "
                        f"Translation=[{transform.transform.translation.x:.4f}, "
                        f"{transform.transform.translation.y:.4f}, "
                        f"{transform.transform.translation.z:.4f}]"
                    )
                    
                    # Check if transform is also available in reverse direction
                    reverse_available = self.tf_buffer.can_transform(
                        "ascamera_color_0",
                        "lidar_frame",
                        test_time,
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    
                    if reverse_available:
                        self.get_logger().info("✓ Transform also available in reverse direction")
                    else:
                        self.get_logger().warn("✗ Transform NOT available in reverse direction")
                except Exception as e:
                    self.get_logger().error(f"Cannot lookup transform despite availability check: {str(e)}")
                    transform_available = False
            
            if not transform_available:
                self.get_logger().warn("✗ Transform check: transform is NOT available")
                
                # Detailed debug info about the transform chain
                if camera_to_base:
                    self.get_logger().info("✓ Transform available: ascamera_color_0 to base_link")
                else:
                    self.get_logger().warn("✗ Missing transform: ascamera_color_0 to base_link")
                
                if base_to_lidar:
                    self.get_logger().info("✓ Transform available: base_link to lidar_frame")
                else: 
                    self.get_logger().warn("✗ Missing transform: base_link to lidar_frame")
                
                # Log all available frames to debug
                try:
                    frames = self.tf_buffer.all_frames_as_string()
                    self.get_logger().info(f"Available frames:\n{frames}")
                except Exception as e:
                    self.get_logger().warn(f"Could not list frames: {str(e)}")
                
                # Re-publish the transform if we have a publishing function
                if hasattr(self, 'publish_static_transform') and callable(self.publish_static_transform):
                    self.get_logger().info("Re-publishing static transform...")
                    self.publish_static_transform()
                else:
                    self.get_logger().info("No static transform publisher available")
                
        except Exception as e:
            self.get_logger().error(f"Error checking transform: {str(e)}")
    
    def _load_basketball_parameters(self):
        """Load basketball physical parameters from config."""
        # Get basketball configuration
        basketball_config = self.config.get('basketball', {})
        
        # Core parameters - ensure basketball sized (9 inch diameter, updated from 10 inch)
        self.ball_radius = basketball_config.get('radius', 0.1143)  # 4.5 inches (9 inch diameter)
        self.max_distance = basketball_config.get('max_distance', 0.2)
        self.min_points = basketball_config.get('min_points', 6)
        self.detection_samples = basketball_config.get('detection_samples', 30)
        
        # Quality thresholds
        quality_thresholds = basketball_config.get('quality_threshold', {})
        self.quality_low = quality_thresholds.get('low', 0.35)
        self.quality_medium = quality_thresholds.get('medium', 0.6)
        self.quality_high = quality_thresholds.get('high', 0.8)
        
        # NEW: Adaptive quality parameters
        adaptive_quality = basketball_config.get('adaptive_quality', {})
        self.quality_distance_factor = adaptive_quality.get('distance_factor', 0.1)  # How much distance affects quality
        self.quality_point_count_factor = adaptive_quality.get('point_count_factor', 0.02)  # How much point count affects threshold
        self.quality_min_threshold = adaptive_quality.get('min_threshold', 0.25)  # Absolute minimum threshold
        self.quality_max_threshold = adaptive_quality.get('max_threshold', 0.9)  # Absolute maximum threshold
        self.quality_history_size = adaptive_quality.get('history_size', 10)  # Size of quality history buffer
        self.quality_history = deque(maxlen=self.quality_history_size)  # Track recent quality scores
        self.detection_confidence_weight = adaptive_quality.get('detection_confidence_weight', 0.7)  # Weight for distance-based detection confidence
        
        # Physical measurements - matching basketball & setup
        physical = self.config.get('physical_measurements', {})
        self.lidar_height = physical.get('lidar_height', 0.1524)  # 6 inches
        # Ball center is always 5 inches above ground (radius) for a basketball rolling on floor
        self.ball_center_height = physical.get('ball_center_height', 0.127)  # 5 inches
        
        # Detection reliability
        reliability = self.config.get('detection_reliability', {})
        # Reduced from 0.5m to 0.2m to improve close-range detection reliability
        self.min_reliable_distance = reliability.get('min_reliable_distance', 0.2)
        self.publish_unreliable = reliability.get('publish_unreliable', True)
        # New parameters for quality-weighted reliability assessment
        self.close_range_threshold = reliability.get('close_range_threshold', 0.3)
        self.close_range_min_quality = reliability.get('close_range_min_quality', 0.4)
        # Hysteresis parameters for reliability assessment
        self.reliability_history_size = reliability.get('history_size', 5)
        self.reliability_history = deque(maxlen=self.reliability_history_size)
        
        # RANSAC parameters
        ransac_config = self.config.get('ransac', {})
        self.ransac_enabled = ransac_config.get('enabled', True)
        self.ransac_max_iterations = ransac_config.get('max_iterations', 30)
        self.ransac_inlier_threshold = ransac_config.get('inlier_threshold', 0.02)
        self.ransac_min_inliers = ransac_config.get('min_inliers', 5)
        
        # For ground movement tracking
        self.ground_movement = True  # Basketball always moves on ground
        self.z_variance_threshold = 0.02  # Small threshold for height variation (2cm)
    
    def _init_transform_parameters(self):
        """Initialize coordinate transform parameters."""
        transform_config = self.config.get('transform', {})
        
        # Frame IDs - Update default camera frame to match what we need
        self.transform_parent_frame = transform_config.get('parent_frame', 'ascamera_color_0')
        self.transform_child_frame = transform_config.get('child_frame', 'lidar_frame')
        
        # Translation vector
        translation = transform_config.get('translation', {})
        self.transform_translation = {
            'x': translation.get('x', 0.0),
            'y': translation.get('y', 0.0),
            'z': translation.get('z', 0.0)
        }
        
        # Rotation quaternion
        rotation = transform_config.get('rotation', {})
        self.transform_rotation = {
            'x': rotation.get('x', 0.0),
            'y': rotation.get('y', 0.0),
            'z': rotation.get('z', 0.0),
            'w': rotation.get('w', 1.0)
        }
        
        # Log transform interval
        self.last_transform_log = 0.0
    
    def _setup_subscribers(self):
        """Set up subscribers for this node."""
        # Get topic config
        topics = self.config.get('topics', {})
        input_topics = topics.get('input', {})
        queue_size = topics.get('queue_size', 10)
        
        # LIDAR scan subscription
        lidar_topic = input_topics.get('lidar_scan', '/scan')
        self.scan_subscription = self.create_subscription(
            LaserScan,
            lidar_topic,
            self.scan_callback,
            queue_size
        )
        
        # YOLO detection subscription
        yolo_topic = input_topics.get('yolo_detection', '/basketball/yolo/position')
        self.yolo_subscription = self.create_subscription(
            PointStamped,
            yolo_topic,
            self.yolo_callback,
            queue_size
        )
        
        # HSV detection subscription
        hsv_topic = input_topics.get('hsv_detection', '/basketball/hsv/position')
        self.hsv_subscription = self.create_subscription(
            PointStamped,
            hsv_topic,
            self.hsv_callback,
            queue_size
        )
        
        # YOLO bounding box subscription for 3D position estimation
        from std_msgs.msg import Float32MultiArray
        yolo_bbox_topic = input_topics.get('yolo_bbox', '/basketball/yolo/bbox')
        self.yolo_bbox_subscription = self.create_subscription(
            Float32MultiArray,
            yolo_bbox_topic,
            self.yolo_bbox_callback,
            queue_size
        )
        
        # Initialize bounding box data storage
        self.yolo_bbox_data = {
            'width': 0.0,
            'height': 0.0,
            'timestamp': 0.0
        }
    
    def _setup_publishers(self):
        """Set up publishers for this node."""
        # Get topic config
        topics = self.config.get('topics', {})
        output_topics = topics.get('output', {})
        queue_size = topics.get('queue_size', 10)
        
        # Ball position publisher
        position_topic = output_topics.get('ball_position', '/basketball/lidar/position')
        self.position_publisher = self.create_publisher(
            PointStamped,
            position_topic,
            queue_size
        )
        
        # Debug position publisher
        debug_topic = output_topics.get('debug_position', '/basketball/lidar/debug_position')
        self.debug_publisher = self.create_publisher(
            PointStamped,
            debug_topic,
            queue_size
        )
        
        # Conditionally create visualization publisher only if enabled
        self.marker_publisher = None
        if self.visualization_enabled:
            viz_topic = output_topics.get('visualization', '/basketball/lidar/visualization')
            self.marker_publisher = self.create_publisher(
                MarkerArray,
                viz_topic,
                queue_size
            )
        
        # Diagnostics publisher
        diag_topic = output_topics.get('diagnostics', '/basketball/lidar/diagnostics')
        self.diagnostics_publisher = self.create_publisher(
            String,
            diag_topic,
            queue_size
        )
        
        # Publisher for sharing system load with other nodes
        load_topic = output_topics.get('system_load', '/system/load')
        self.load_publisher = self.create_publisher(
            Float32,
            load_topic,
            queue_size
        )

    def monitor_resources(self):
        """
        Monitor system resources and adapt processing accordingly.
        Implements hysteresis and gradual mode transitions to prevent thrashing.
        """
        try:
            # Get CPU and memory usage
            current_cpu = psutil.cpu_percent()
            self.current_cpu_load = current_cpu
            self.current_memory_usage = psutil.virtual_memory().percent
            
            # Add to rolling window for smoothing
            self.mode_samples.append(current_cpu)
            
            # Calculate smoothed CPU load (more stable than instantaneous readings)
            smoothed_cpu = sum(self.mode_samples) / len(self.mode_samples) if self.mode_samples else current_cpu
            
            # Determine tentative performance mode based on system load
            new_mode = self.determine_performance_mode(smoothed_cpu)
            
            # Apply hysteresis to mode transitions
            current_time = time.time()
            mode_stable_duration = current_time - self.last_mode_change_time
            
            # Only change modes if:
            # 1. We've been in the current mode for at least mode_stability_time seconds
            # 2. The new mode is different AND the CPU load exceeds the threshold plus hysteresis
            should_change_mode = False
            
            if new_mode != self.performance_mode:
                if mode_stable_duration >= self.mode_stability_time:
                    # Add hysteresis to thresholds - require more significant changes to switch modes
                    if new_mode == "MINIMAL" and smoothed_cpu > self.high_load_threshold + self.mode_transition_hysteresis:
                        should_change_mode = True
                    elif new_mode == "NORMAL" and smoothed_cpu < self.low_load_threshold - self.mode_transition_hysteresis:
                        should_change_mode = True
                    elif new_mode == "EFFICIENT" and (
                        (self.performance_mode == "NORMAL" and smoothed_cpu > self.low_load_threshold + self.mode_transition_hysteresis) or
                        (self.performance_mode == "MINIMAL" and smoothed_cpu < self.high_load_threshold - self.mode_transition_hysteresis)
                    ):
                        should_change_mode = True
            
            # Apply the mode change if conditions are met
            if should_change_mode:
                # Log mode changes
                self.get_logger().info(
                    f"Performance mode change: {self.performance_mode} -> {new_mode} "
                    f"(CPU: {smoothed_cpu:.1f}%, Memory: {self.current_memory_usage:.1f}%, "
                    f"stable for {mode_stable_duration:.1f}s)"
                )
                
                self.performance_mode = new_mode
                self.last_mode_change_time = current_time
                
                # Adjust diagnostic timer for high load
                if new_mode == "MINIMAL":
                    self.diagnostics_timer.timer_period_ns = int(self.diagnostics_interval_high_load * 1e9)
                else:
                    # Restore normal diagnostic frequency if coming from high load
                    if self.performance_mode != "MINIMAL":
                        self.diagnostics_timer.timer_period_ns = int(self.diagnostics_interval_normal * 1e9)
            
            # Publish system load for other nodes
            load_msg = Float32()
            load_msg.data = float(smoothed_cpu)
            self.load_publisher.publish(load_msg)
            
        except Exception as e:
            self.get_logger().warn(f"Error monitoring resources: {str(e)}")
            
    def determine_performance_mode(self, cpu_load):
        """
        Determine the appropriate performance mode based on CPU load.
        Separates the decision logic from the mode transition logic.
        
        Args:
            cpu_load: Current CPU load percentage
            
        Returns:
            str: The appropriate performance mode ("NORMAL", "EFFICIENT", or "MINIMAL")
        """
        if cpu_load > self.high_load_threshold:
            return "MINIMAL"
        elif cpu_load > self.low_load_threshold:
            return "EFFICIENT"
        else:
            return "NORMAL"

    def scan_callback(self, msg):
        """
        Process LaserScan messages from the LIDAR.
        
        Converts polar coordinates to Cartesian coordinates.
        Implements intelligent sampling to preserve ball detection accuracy under load.
        """
        try:
            # If system is under very high load, we might skip processing some scans
            if self.adaptive_processing and self.performance_mode == "MINIMAL" and self.processed_scans % 2 != 0:
                self.processing_skips += 1
                return
            
            # Store scan metadata
            self.latest_scan = msg
            self.scan_timestamp = msg.header.stamp
            self.scan_frame_id = "lidar_frame"
            
            # Extract scan parameters
            angle_min = msg.angle_min
            angle_increment = msg.angle_increment
            ranges = np.array(msg.ranges)
            
            # Filter out invalid measurements
            valid_indices = np.isfinite(ranges)
            
            # Filter out very short ranges (robot body reflections)
            min_valid_range = 0.05
            valid_indices = valid_indices & (ranges > min_valid_range)
            
            # Skip if no valid ranges
            if np.sum(valid_indices) == 0:
                self.get_logger().warn("No valid range measurements in scan")
                self.points_array = None
                return
            
            valid_ranges = ranges[valid_indices]
            angles = angle_min + angle_increment * np.arange(len(ranges))[valid_indices]
            
            # Optimize for high CPU load - limit points processed if needed
            point_limit = self.max_point_limit
            if self.adaptive_processing:
                if self.performance_mode == "EFFICIENT":
                    point_limit = self.max_point_limit // 2
                elif self.performance_mode == "MINIMAL":
                    point_limit = self.max_point_limit // 4
            
            # Intelligent sampling: Create regions of interest based on recent detections and movement patterns
            # Initialize all points as non-priority to start
            priority_weights = np.ones_like(valid_ranges)
            prioritize_regions = False
            
            # Convert scan to Cartesian for region analysis
            x_all = valid_ranges * np.cos(angles)
            y_all = valid_ranges * np.sin(angles)
            
            # Define regions of interest with different priority levels
            roi_regions = []
            
            # Region 1: Previous ball detection (highest priority)
            if self.previous_ball_position is not None:
                prev_ball_x = self.previous_ball_position[0]
                prev_ball_y = self.previous_ball_position[1]
                
                # Distance from each point to previous detection
                distances_to_prev = np.sqrt((x_all - prev_ball_x)**2 + (y_all - prev_ball_y)**2)
                
                # Critical region: points very close to previous detection (highest weight)
                critical_mask = distances_to_prev < self.critical_point_radius
                
                # Extended region: slightly wider area around previous detection
                extended_mask = distances_to_prev < self.critical_point_radius * 2.0
                
                # Add critical region with highest weight
                roi_regions.append({"mask": critical_mask, "weight": 10.0, "name": "previous_detection"})
                
                # Add extended region with medium-high weight
                roi_regions.append({"mask": extended_mask & ~critical_mask, "weight": 5.0, "name": "extended_previous"})
                
                # Flag that we have meaningful regions to prioritize
                prioritize_regions = True
            
            # Region 2: Predicted movement direction (medium priority)
            # Use velocity history to predict where the ball is likely moving
            if len(self.position_history) >= 2 and len(self.velocity_history) > 0:
                # Get last two positions to determine direction
                last_pos = self.position_history[-1]
                prev_pos = self.position_history[-2] if len(self.position_history) >= 2 else None
                
                if prev_pos is not None and last_pos is not None:
                    # Calculate movement direction vector
                    movement_dir = last_pos[:2] - prev_pos[:2]
                    movement_mag = np.linalg.norm(movement_dir)
                    
                    # Only use direction if we have meaningful movement
                    if movement_mag > 0.05:
                        # Normalize direction vector
                        movement_dir = movement_dir / movement_mag
                        
                        # Average recent velocity
                        avg_velocity = np.mean(self.velocity_history)
                        
                        # Predict next position (simple linear extrapolation)
                        # Using multiple time horizons to create a path prediction
                        time_horizons = [0.2, 0.4, 0.6]  # Look ahead 0.2, 0.4, 0.6 seconds
                        
                        for dt in time_horizons:
                            # Calculate predicted position
                            pred_x = last_pos[0] + movement_dir[0] * avg_velocity * dt
                            pred_y = last_pos[1] + movement_dir[1] * avg_velocity * dt
                            
                            # Define search radius - wider for further predictions
                            search_radius = self.critical_point_radius * (1.0 + dt)
                            
                            # Calculate distances to this predicted position
                            distances_to_pred = np.sqrt((x_all - pred_x)**2 + (y_all - pred_y)**2)
                            
                            # Create mask for points near this prediction
                            pred_mask = distances_to_pred < search_radius
                            
                            # Add this prediction region with weight that decreases with time horizon
                            weight = 8.0 / (1.0 + dt * 5.0)  # Decreases for further predictions
                            roi_regions.append({
                                "mask": pred_mask, 
                                "weight": weight, 
                                "name": f"prediction_{dt:.1f}s"
                            })
                            
                        # Flag that we have meaningful regions to prioritize
                        prioritize_regions = True
            
            # Region 3: Edge detection - find potential circular patterns (medium priority)
            # This helps detect new basketballs not related to previous detections
            # Only do this in NORMAL mode as it's more computationally expensive
            if self.performance_mode == "NORMAL" and len(valid_ranges) > 20:
                try:
                    # Quick and simple edge detection using range differences
                    range_diffs = np.abs(np.diff(valid_ranges, prepend=valid_ranges[0]))
                    
                    # Find significant jumps in range values (potential edges)
                    edge_threshold = 0.1  # 10cm jumps can indicate object edges
                    edge_points = range_diffs > edge_threshold
                    
                    # Dilate the edge points slightly to capture surrounding points
                    # Simple dilation by considering neighbors
                    edge_dilated = np.zeros_like(edge_points)
                    for i in range(1, len(edge_points)-1):
                        if edge_points[i-1] or edge_points[i] or edge_points[i+1]:
                            edge_dilated[i] = True
                    
                    # Add edge regions with medium weight
                    if np.any(edge_dilated):
                        roi_regions.append({"mask": edge_dilated, "weight": 4.0, "name": "edges"})
                        prioritize_regions = True
                except Exception as e:
                    # Skip edge detection if it fails
                    self.get_logger().debug(f"Edge detection skipped: {str(e)}")
            
            # Region 4: Uniform sampling across entire scan (lowest priority)
            # Ensure we sample some points from the entire scan for global awareness
            # Give uniform low weight to all points
            roi_regions.append({"mask": np.ones_like(valid_ranges, dtype=bool), "weight": 1.0, "name": "uniform"})
            
            # Apply ROI weights to create priority sampling weights
            if prioritize_regions:
                # Start with base weights
                priority_weights = np.ones_like(valid_ranges, dtype=float)
                
                # Apply each region's weights
                for region in roi_regions:
                    # Add weights where the mask is True
                    priority_weights[region["mask"]] += region["weight"]
                
                # Count points in high priority regions
                high_priority_mask = priority_weights > 2.0  # Points with more than base+2 weight
                high_priority_count = np.sum(high_priority_mask)
                
                # Log information about high-priority points (if not in MINIMAL mode)
                if self.performance_mode != "MINIMAL" and high_priority_count > 0:
                    region_counts = {}
                    for region in roi_regions:
                        if region["name"] != "uniform":  # Skip logging uniform region
                            region_counts[region["name"]] = np.sum(region["mask"])
                    
                    self.get_logger().debug(
                        f"Smart sampling: {high_priority_count} high-priority points. "
                        f"Regions: {region_counts}"
                    )
            
            # Sample points based on priority weights if we need to reduce
            if len(valid_ranges) > point_limit and prioritize_regions:
                # For weighted sampling without replacement
                # Normalize weights to probability distribution
                p = priority_weights / np.sum(priority_weights)
                
                # Set minimum count for high-priority points 
                high_priority_mask = priority_weights > 2.0
                high_priority_count = np.sum(high_priority_mask)
                
                # Ensure we keep a minimum number of critical points, select the rest with weighted sampling
                if high_priority_count > 0:
                    # Determine how many high-priority points to keep
                    # Ensure we keep at least min_critical_points of high priority points
                    critical_points_to_keep = min(high_priority_count, 
                                                max(self.min_critical_points, int(point_limit * 0.7)))
                    
                    # Directly select all high-priority points if there are few enough
                    if high_priority_count <= critical_points_to_keep:
                        critical_indices = np.where(high_priority_mask)[0]
                        remaining_count = point_limit - high_priority_count
                    else:
                        # Otherwise, sample from high-priority points weighted by their importance
                        high_priority_indices = np.where(high_priority_mask)[0]
                        high_priority_weights = priority_weights[high_priority_mask]
                        high_priority_p = high_priority_weights / np.sum(high_priority_weights)
                        
                        # Sample critical_points_to_keep from high-priority points
                        critical_indices = np.random.choice(
                            high_priority_indices, 
                            size=critical_points_to_keep, 
                            replace=False, 
                            p=high_priority_p
                        )
                        remaining_count = point_limit - critical_points_to_keep
                    
                    # Sample the remaining points from all points with distance-weighted probabilities
                    if remaining_count > 0:
                        # Create mask for non-critical points
                        non_critical_mask = np.ones(len(valid_ranges), dtype=bool)
                        non_critical_mask[critical_indices] = False
                        
                        # Get indices of non-critical points
                        non_critical_indices = np.where(non_critical_mask)[0]
                        
                        # No remaining points case
                        if len(non_critical_indices) == 0:
                            # Just use the critical indices
                            selected_indices = critical_indices
                        else:
                            # Get weights for non-critical points
                            non_critical_weights = priority_weights[non_critical_mask]
                            
                            # Normalize to probability
                            if np.sum(non_critical_weights) > 0:
                                non_critical_p = non_critical_weights / np.sum(non_critical_weights)
                            else:
                                non_critical_p = None  # Uniform sampling if all weights are zero
                            
                            # Sample remaining points from non-critical points
                            remaining_indices = np.random.choice(
                                non_critical_indices,
                                size=min(remaining_count, len(non_critical_indices)),
                                replace=False,
                                p=non_critical_p
                            )
                            
                            # Combine critical and remaining indices
                            selected_indices = np.concatenate([critical_indices, remaining_indices])
                    else:
                        # Only use critical indices if no room for other points
                        selected_indices = critical_indices
                    
                    # Sort indices to preserve original order (improves spatial coherence)
                    selected_indices = np.sort(selected_indices)
                    
                    # Select final points using these indices
                    valid_ranges = valid_ranges[selected_indices]
                    angles = angles[selected_indices]
                else:
                    # Fallback to weighted sampling if no high-priority points
                    selected_indices = np.random.choice(
                        len(valid_ranges),
                        size=point_limit,
                        replace=False,
                        p=p
                    )
                    
                    # Sort indices to preserve original order
                    selected_indices = np.sort(selected_indices)
                    
                    # Select final points using these indices
                    valid_ranges = valid_ranges[selected_indices]
                    angles = angles[selected_indices]
            elif len(valid_ranges) > point_limit:
                # Fallback to uniform sampling if we have no priority regions
                sample_step = len(valid_ranges) // point_limit
                valid_ranges = valid_ranges[::sample_step]
                angles = angles[::sample_step]
            
            # Convert to Cartesian coordinates - optimized for minimal memory allocation
            x = valid_ranges * np.cos(angles)
            y = valid_ranges * np.sin(angles)
            z = np.zeros_like(x)  # No need to allocate new memory
            
            # Stack coordinates - memory efficient with preallocated array
            self.points_array = np.column_stack((x, y, z))
            
            # Update statistics
            self.processed_scans += 1
            
            # Log scan information only in normal mode
            if self.performance_mode == "NORMAL":
                log_interval = self.config.get('diagnostics', {}).get('log_scan_interval', 20)
                if self.processed_scans % log_interval == 0:
                    self.get_logger().debug(
                        f"Processed scan #{self.processed_scans} with "
                        f"{len(self.points_array)} valid points"
                    )
            
        except Exception as e:
            self.log_error(f"Error processing scan: {str(e)}")
            self.points_array = None
    
    def yolo_callback(self, msg):
        """
        Handle ball detections from the YOLO neural network.
        """
        self.yolo_detections += 1
        self.camera_detection_callback(msg, "YOLO")
    
    def hsv_callback(self, msg):
        """
        Handle ball detections from the HSV color detector.
        """
        self.hsv_detections += 1
        self.camera_detection_callback(msg, "HSV")
    
    def camera_detection_callback(self, msg, source):
        """
        Process ball detections from camera systems (YOLO or HSV).
        Find matching points in LIDAR data.
        """
        detection_start_time = time.time()
        
        try:
            # Check if we have valid scan data
            if self.latest_scan is None or self.points_array is None or len(self.points_array) == 0:
                if self.performance_mode != "MINIMAL":  # Skip log in minimal mode
                    self.get_logger().info(f"LIDAR: Waiting for scan data for {source} detection")
                return
            
            # Extract camera detection info
            x_2d = msg.point.x
            y_2d = msg.point.y
            confidence = msg.point.z
            
            # Get camera frame - need to know which coordinate system the point is in
            camera_frame = msg.header.frame_id
            if not camera_frame:
                camera_frame = "ascamera_color_0"  # Default to standard camera frame
            
            # Transform point from camera frame to lidar frame
            transformed_point = None
            try:
                # Create a TF2PointStamped from the incoming detection
                camera_point = TF2PointStamped()
                camera_point.header = msg.header
                camera_point.point = msg.point
                
                # Check if transform is available - only check the transform we need
                transform_available = self.tf_buffer.can_transform(
                    "lidar_frame",
                    camera_frame,
                    msg.header.stamp,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
                
                if transform_available:
                    # Transform the point from camera to lidar frame
                    lidar_point = self.tf_buffer.transform(
                        camera_point,
                        "lidar_frame",
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    transformed_point = np.array([lidar_point.point.x, lidar_point.point.y, lidar_point.point.z])
                    
                    if self.performance_mode != "MINIMAL":  # Skip log in minimal mode
                        self.get_logger().info(
                            f"{source}: Transformed point from {camera_frame} to lidar_frame: "
                            f"({transformed_point[0]:.2f}, {transformed_point[1]:.2f}, {transformed_point[2]:.2f})"
                        )
                else:
                    self.get_logger().warn(f"Transform not available from {camera_frame} to lidar_frame")
            except Exception as e:
                self.get_logger().warn(f"Transform error: {str(e)}")
            
            if self.performance_mode != "MINIMAL":  # Skip log in minimal mode
                self.get_logger().info(
                    f"{source}: Ball detected at pixel ({x_2d:.1f}, {y_2d:.1f}) "
                    f"with confidence {confidence:.2f}"
                )
            
            # Get 3D point estimate from YOLO bbox if available
            estimated_3d_point = None
            if source == "YOLO" and hasattr(self, 'yolo_bbox_data') and self.yolo_bbox_data.get('timestamp', 0) > time.time() - 1.0:
                bbox_width = self.yolo_bbox_data.get('width', 0)
                bbox_height = self.yolo_bbox_data.get('height', 0)
                
                if bbox_width > 0 and bbox_height > 0:
                    estimated_3d_point = self.estimate_3d_from_2d(msg, bbox_width, bbox_height)
                    
                    if estimated_3d_point is not None and self.performance_mode != "MINIMAL":
                        self.get_logger().info(
                            f"Using estimated 3D position from YOLO 2D: "
                            f"({estimated_3d_point[0]:.2f}, {estimated_3d_point[1]:.2f}, {estimated_3d_point[2]:.2f})"
                        )
            
            # Use estimated 3D point as primary seed if available, transformed point as backup
            seed_point = estimated_3d_point if estimated_3d_point is not None else transformed_point
            
            # Find basketball in LIDAR data
            ball_results = self.find_basketball_ransac(seed_point)
            
            # If no ball found with RANSAC but we have an estimated 3D position from bbox,
            # directly use that instead of relying only on LIDAR points
            if (not ball_results or len(ball_results) == 0) and estimated_3d_point is not None:
                if self.performance_mode != "MINIMAL":
                    self.get_logger().info("No matching ball found with RANSAC, using estimated 3D position directly")
                
                # Calculate a default quality score based on confidence
                quality = 0.6  # Base quality for bbox-derived positions
                if hasattr(msg.point, 'z'):
                    # Adjust quality based on confidence if available (0.0-1.0)
                    quality = min(0.9, quality + msg.point.z * 0.3)
                
                # Publish the estimated position
                self.publish_ball_position(
                    estimated_3d_point,  # Use the estimated 3D point
                    10,                  # Default cluster size
                    quality,             # Quality score
                    f"{source}_3D_EST",  # Mark as estimated
                    msg.header.stamp     # Use original timestamp
                )
                return
            
            # Process the best detected ball (if any)
            if ball_results and len(ball_results) > 0:
                # Get the best match
                best_match = ball_results[0]
                center, cluster_size, circle_quality = best_match
                
                # Publish ball position
                self.publish_ball_position(center, cluster_size, circle_quality, source, msg.header.stamp)
            else:
                if self.performance_mode != "MINIMAL":  # Skip log in minimal mode
                    self.get_logger().info(f"LIDAR: No matching ball found for {source} detection")
                self.consecutive_failures += 1
            
        except Exception as e:
            self.log_error(f"Error processing {source} detection: {str(e)}")
        
        # Log processing time
        processing_time = (time.time() - detection_start_time) * 1000  # in ms
        self.detection_times.append(processing_time)
        self.detection_latency = processing_time
        if self.performance_mode != "MINIMAL":  # Skip log in minimal mode
            self.get_logger().debug(f"LIDAR: {source} processing took {processing_time:.2f}ms")
    
    def find_basketball_ransac(self, camera_seed_point=None):
        """
        Find a basketball in LIDAR data using RANSAC for robust circle fitting.
        Optimized for a basketball (10-inch diameter) rolling on the ground.
        
        Args:
            camera_seed_point: Optional point in LIDAR frame transformed from camera detection
        
        Returns:
            list: List of (center, cluster_size, quality) tuples for detected basketballs
        """
        if self.points_array is None or len(self.points_array) == 0:
            return []
            
        # Create seed points for RANSAC
        seed_points = []
        
        # If camera detection provided a transformed point, prioritize it
        if camera_seed_point is not None and len(camera_seed_point) >= 2:
            # Use only x,y coordinates for 2D search in LIDAR data
            seed_points.append([camera_seed_point[0], camera_seed_point[1], 0])
        
        # Include previous ball position if available
        if self.previous_ball_position is not None:
            seed_points.append(self.previous_ball_position)
        
        # Include current points array
        if len(self.points_array) > 0:
            # Create a few seed points based on point clusters
            # Focus on points within a reasonable range
            distances = np.sqrt(self.points_array[:, 0]**2 + self.points_array[:, 1]**2)
            
            # Adjust valid range for a larger basketball (can be detected from further away)
            valid_indices = np.where((distances > 0.3) & (distances < 3.0))[0]
            
            if len(valid_indices) > 0:
                # Adjust sample count based on performance mode
                if self.performance_mode == "NORMAL":
                    sample_count = min(10, len(valid_indices))
                elif self.performance_mode == "EFFICIENT":
                    sample_count = min(5, len(valid_indices))
                else:  # MINIMAL
                    sample_count = min(3, len(valid_indices))
                    
                indices = np.random.choice(valid_indices, sample_count, replace=False)
                for idx in indices:
                    seed_points.append(self.points_array[idx])
        
        # Best result tracking
        best_center = None
        best_inlier_count = 0
        best_quality = 0
        best_distance = 0
        
        # Try RANSAC with each seed point
        for seed_point in seed_points:
            # Find all points near this seed - using vectorized operations for speed
            distances = np.sqrt(
                (self.points_array[:, 0] - seed_point[0])**2 + 
                (self.points_array[:, 1] - seed_point[1])**2
            )
            
            # For basketball, use a larger search radius based on the basketball's size
            nearby_indices = np.where(distances < self.max_distance * 3)[0]
            
            if len(nearby_indices) < self.min_points:
                continue
                
            # Get points near seed
            nearby_points = self.points_array[nearby_indices]
            
            # Calculate distance from LIDAR for quality adaptation
            seed_distance = np.sqrt(seed_point[0]**2 + seed_point[1]**2)
            
            # Determine iterations based on system load
            max_iterations = self.ransac_max_iterations
            if self.dynamic_ransac_iterations:
                if self.performance_mode == "EFFICIENT":
                    max_iterations = max(self.min_ransac_iterations, self.ransac_max_iterations // 2)
                elif self.performance_mode == "MINIMAL":
                    max_iterations = max(self.min_ransac_iterations, self.ransac_max_iterations // 4)
                    
            # Try fitting a circle using RANSAC
            center, inlier_count, quality = self.ransac_circle_fit(
                nearby_points, 
                max_iterations,
                self.ransac_inlier_threshold,
                distance=seed_distance  # Pass distance for quality adaptation
            )
            
            # Check if this is better than current best
            if quality > best_quality and inlier_count >= self.min_points:
                best_center = center
                best_inlier_count = inlier_count
                best_quality = quality
                best_distance = seed_distance
        
        # Calculate adaptive quality threshold based on detection conditions
        adaptive_threshold = self.calculate_adaptive_quality_threshold(best_distance, best_inlier_count if best_center is not None else 0)
        
        # Return result if found and exceeds adaptive threshold
        if best_center is not None and best_quality >= adaptive_threshold:
            # Add quality score to history for trend analysis
            self.quality_history.append(best_quality)
            
            # Store the position for future reference
            self.previous_ball_position = best_center
            
            # Update statistics
            self.consecutive_failures = 0
            self.last_successful_detection_time = time.time()
            
            # Return as list of results (keeping same format as original code)
            return [(best_center, best_inlier_count, best_quality)]
        
        return []
    
    def calculate_adaptive_quality_threshold(self, distance, point_count):
        """
        Calculate an adaptive quality threshold based on detection conditions.
        
        Args:
            distance: Distance to the detected ball in meters
            point_count: Number of inlier points in the detection
            
        Returns:
            float: Adaptive quality threshold
        """
        # Base threshold from configuration
        base_threshold = self.quality_low
        
        # Distance-based adjustment: closer objects need higher quality (more reliable)
        # and farther objects can accept lower quality (harder to detect reliably)
        if distance < 0.5:
            # Require higher quality for very close detections (can be noisy)
            distance_adjustment = self.quality_distance_factor * 1.0
        elif distance < 1.0:
            # Slight increase for close detections
            distance_adjustment = self.quality_distance_factor * 0.5
        elif distance > 2.0:
            # Allow lower quality for far detections
            distance_adjustment = -self.quality_distance_factor * (min(distance, 4.0) - 2.0)
        else:
            # Neutral zone - no adjustment
            distance_adjustment = 0.0
            
        # Point count adjustment: more points should yield more reliable detection
        # Linear scaling based on point count
        if point_count > 20:
            # With many points, we can require higher quality
            point_adjustment = self.quality_point_count_factor * min(point_count, 50) / 10.0
        elif point_count < 10:
            # With few points, accept lower quality
            point_adjustment = -self.quality_point_count_factor * (10 - point_count) / 2.0
        else:
            # Neutral zone - no adjustment
            point_adjustment = 0.0
            
        # Recent quality trend adjustment
        trend_adjustment = 0.0
        if len(self.quality_history) > 3:
            # Look at recent quality trend
            recent_avg = sum(list(self.quality_history)[-3:]) / 3.0
            if recent_avg > 0.7:
                # If recent detections were high quality, slightly increase threshold
                trend_adjustment = 0.05
            elif recent_avg < 0.5:
                # If recent detections were poor, slightly decrease threshold
                trend_adjustment = -0.05
                
        # Calculate final adaptive threshold with constraints
        adaptive_threshold = base_threshold + distance_adjustment + point_adjustment + trend_adjustment
        
        # Clamp to configured min/max
        adaptive_threshold = max(self.quality_min_threshold, min(self.quality_max_threshold, adaptive_threshold))
        
        # Log the adaptive threshold calculation if not in MINIMAL mode
        if self.performance_mode != "MINIMAL":
            self.get_logger().debug(
                f"Adaptive quality threshold: {adaptive_threshold:.3f} = Base({base_threshold:.2f}) + "
                f"Distance({distance_adjustment:.2f}) + Points({point_adjustment:.2f}) + Trend({trend_adjustment:.2f})"
            )
            
        return adaptive_threshold

    def ransac_circle_fit(self, points, max_iterations=30, threshold=0.02, distance=1.0):
        """
        Use RANSAC to fit a circle to points, robust to outliers.
        Optimized for performance with adaptive quality assessment based on distance and point conditions.
        
        Args:
            points: Array of points to fit circle to
            max_iterations: Maximum RANSAC iterations
            threshold: Inlier threshold distance
            distance: Distance of the points from LIDAR (for quality adaptation)
            
        Returns:
            tuple: (center, inlier_count, quality) or (None, 0, 0) if no fit found
        """
        if points is None or len(points) < 3:
            return None, 0, 0
            
        best_inlier_count = 0
        best_center = None
        best_radius = 0
        
        # Limit iterations based on point count for better performance
        actual_iterations = min(max_iterations, len(points) // 2)
        actual_iterations = max(self.min_ransac_iterations, actual_iterations)  # At least min iterations
        
        # Pre-compute point coordinates for vector operations
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Pre-compute average distance to adjust radius tolerance
        avg_distance = np.mean(np.sqrt(x_coords**2 + y_coords**2))
        
        # Adjust inlier threshold based on distance
        # For closer points, we need tighter thresholds as measurement density is higher
        distance_adjusted_threshold = threshold * (1.0 + 0.5 * min(avg_distance, 3.0))
        
        # Calculate adaptive radius tolerance based on distance
        # Allow more variation at closer ranges due to perspective effects and partial views
        base_radius_tolerance = 0.5  # Original tolerance was 0.5 (50%)
        # Closer distances need more flexibility as perspective effects are stronger
        distance_factor = 1.0
        if avg_distance < 1.0:
            # Increase tolerance for close objects (up to 2x at very close range)
            distance_factor = 2.0 - avg_distance
        elif avg_distance > 2.0:
            # Slightly tighter tolerance for distant objects
            distance_factor = 0.9
            
        radius_tolerance = base_radius_tolerance * distance_factor
        
        # Increase minimum inliers based on distance
        # For closer objects we expect more inliers
        min_inliers = self.ransac_min_inliers
        if avg_distance < 1.0:
            min_inliers = max(min_inliers + 2, 7)  # More inliers expected when close
        
        # Track multiple candidate fits with different quality metrics
        candidates = []
        
        for _ in range(actual_iterations):
            # Randomly sample 3 points
            if len(points) < 3:
                continue
                
            sample_indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_indices]
            
            # Fit circle to these points
            try:
                center, radius = self.fit_circle(sample_points)
                
                # Apply adaptive radius tolerance check
                radius_error = abs(radius - self.ball_radius) / self.ball_radius
                
                # Adjust expected radius based on distance (perspective correction)
                # Very close balls may appear larger in scan data
                expected_radius = self.ball_radius
                if avg_distance < 0.8:
                    # Adjust expected radius for very close objects
                    # At very close range, apparent radius can be larger
                    expected_radius = self.ball_radius * (1.0 + (0.8 - avg_distance) * 0.2)
                    
                # Check against adjusted radius with adaptive tolerance
                adjusted_radius_error = abs(radius - expected_radius) / expected_radius
                
                # Allow more flexibility for partial circle detection
                # Instead of immediately skipping bad radii, score them but with penalty
                use_sample = True
                radius_penalty = 0.0
                
                # Apply base check with adaptive tolerance
                if adjusted_radius_error > radius_tolerance:
                    # For significant deviations:
                    if adjusted_radius_error > radius_tolerance * 1.5:
                        # Too far off - skip this sample
                        continue
                    else:
                        # Borderline case - accept but with penalty
                        radius_penalty = adjusted_radius_error / radius_tolerance
                        use_sample = True
                
                # Count inliers using vectorized operations for speed
                distances = np.sqrt(
                    (x_coords - center[0])**2 + 
                    (y_coords - center[1])**2
                )
                
                # Inliers are points close to the expected circle
                inliers = np.abs(distances - radius) < distance_adjusted_threshold
                inlier_count = np.sum(inliers)
                
                # Calculate coverage angle to detect partial circles
                if inlier_count >= min_inliers:
                    # Get angles of inlier points relative to center
                    inlier_angles = np.arctan2(
                        y_coords[inliers] - center[1],
                        x_coords[inliers] - center[0]
                    )
                    
                    # Normalize angles to 0-2π range
                    inlier_angles = (inlier_angles + 2*np.pi) % (2*np.pi)
                    
                    # Sort angles to find gaps
                    sorted_angles = np.sort(inlier_angles)
                    
                    # Calculate gaps between consecutive angles
                    angle_diffs = np.diff(sorted_angles)
                    angle_diffs = np.append(angle_diffs, sorted_angles[0] + 2*np.pi - sorted_angles[-1])
                    
                    # Find largest gap
                    max_gap = np.max(angle_diffs)
                    
                    # Calculate coverage (percent of full circle)
                    coverage = 1.0 - max_gap / (2*np.pi)
                    
                    # Adjust inlier count based on coverage for partial circles
                    adjusted_inlier_count = inlier_count
                    
                    # Track both the raw and adjusted candidate with quality metrics
                    if use_sample:
                        # Calculate base quality
                        inlier_ratio = inlier_count / len(points)
                        
                        # NEW: Multi-factor quality scoring
                        # 1. Inlier ratio (how many points fit the model)
                        # 2. Radius accuracy (how close to expected radius)
                        # 3. Coverage (how completely the circle is detected)
                        # 4. Point density (higher density = better detection)
                        # 5. Distance-based weighting (closer = more reliable, with exceptions)
                        
                        # Calculate point density factor (points per radian)
                        point_density = inlier_count / (2 * np.pi)
                        density_factor = min(1.0, point_density / 5.0)  # Normalize with max at ~5 points per radian
                        
                        # Distance factor - higher weight for medium distances (not too close, not too far)
                        if distance < 0.5:
                            # Very close can be noisy due to occlusion effects
                            distance_weight = 0.8
                        elif distance < 1.2:
                            # Medium distance is ideal
                            distance_weight = 1.0
                        elif distance < 2.5:
                            # Farther is less reliable but still good
                            distance_weight = 0.9 - (distance - 1.2) * 0.1
                        else:
                            # Very far is least reliable
                            distance_weight = 0.8
                        
                        # Weight components differently based on their reliability
                        quality = (0.4 * inlier_ratio +                # Inlier ratio (most important)
                                  0.2 * (1.0 - min(adjusted_radius_error, 1.0)) +  # Radius accuracy
                                  0.15 * coverage +                    # Circle completeness
                                  0.15 * density_factor +              # Point density
                                  0.1 * distance_weight)               # Distance reliability
                        
                        # Apply radius penalty if it was borderline
                        quality -= radius_penalty * 0.2
                        
                        # Store this candidate
                        candidates.append({
                            'center': center,
                            'radius': radius,
                            'inlier_count': inlier_count,
                            'quality': quality,
                            'coverage': coverage,
                            'radius_error': adjusted_radius_error,
                            'distance': avg_distance
                        })
                
                # Also track best model directly for backward compatibility
                if inlier_count > best_inlier_count and adjusted_radius_error <= radius_tolerance:
                    best_inlier_count = inlier_count
                    best_center = center
                    best_radius = radius
            except Exception:
                continue
        
        # Choose the best candidate from the tracked list
        if candidates:
            # Sort candidates by quality
            candidates.sort(key=lambda x: x['quality'], reverse=True)
            best_candidate = candidates[0]
            
            # Update best_center if we found a better candidate
            if best_candidate['quality'] > 0.0:
                best_center = best_candidate['center']
                best_inlier_count = best_candidate['inlier_count']
                best_quality = best_candidate['quality']
                
                # Add z-coordinate for 3D position
                center_3d = np.array([best_center[0], best_center[1], self.ball_center_height])
                
                # Log candidate details if not in MINIMAL mode
                if self.performance_mode != "MINIMAL" and len(candidates) > 1:
                    self.get_logger().debug(
                        f"Selected best of {len(candidates)} circle candidates: "
                        f"quality={best_candidate['quality']:.2f}, "
                        f"inliers={best_candidate['inlier_count']}, "
                        f"coverage={best_candidate['coverage']:.2f}, "
                        f"radius_error={best_candidate['radius_error']:.2f}, "
                        f"distance={best_candidate['distance']:.2f}m"
                    )
                
                return center_3d, best_inlier_count, best_quality
        
        # Original fallback code
        if best_center is None or best_inlier_count < min_inliers:
            return None, 0, 0
        
        # Refine with all inliers if we have enough
        if best_inlier_count >= min_inliers:
            # Calculate quality metrics
            inlier_ratio = best_inlier_count / len(points)
            radius_error = abs(best_radius - self.ball_radius) / self.ball_radius
            
            # Enhanced quality metric with distance weighting
            distance_factor = 1.0
            if distance < 0.5:
                distance_factor = 0.8  # Lower weight for very close detections
            elif distance > 2.0:
                distance_factor = 0.9 - min(0.2, (distance - 2.0) * 0.1)  # Lower for far detections
                
            # Combine factors for final quality score    
            quality = (0.6 * inlier_ratio + 
                     0.3 * (1.0 - min(radius_error, 1.0)) +
                     0.1 * distance_factor)
            
            # Add z-coordinate for 3D position - reuse existing array
            center_3d = np.array([best_center[0], best_center[1], self.ball_center_height])
            
            return center_3d, best_inlier_count, quality
        
        return None, 0, 0

    def fit_circle(self, points):
        """
        Fit a circle to 2D or 3D points.
        Optimized for basketball size (10-inch diameter) on Raspberry Pi.
        """
        # Extract 2D coordinates - avoid copying if possible
        if points.shape[1] > 2:
            points_2d = points[:, 0:2]
        else:
            points_2d = points
        
        # Need at least 3 points
        if len(points_2d) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        # Direct calculation for exactly 3 points (most efficient)
        if len(points_2d) == 3:
            # Get coordinates
            x1, y1 = points_2d[0]
            x2, y2 = points_2d[1]
            x3, y3 = points_2d[2]
            
            # Calculate circle parameters
            A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
            B = (x1**2 + y1**2) * (y3 - y2) + (x2**2 + y2**2) * (y1 - y3) + (x3**2 + y3**2) * (y2 - y1)
            C = (x1**2 + y1**2) * (x2 - x3) + (x2**2 + y2**2) * (x3 - x1) + (x3**2 + y3**2) * (x1 - x2)
            
            if abs(A) < 1e-10:
                raise ValueError("Points are collinear, cannot fit circle")
                
            x0 = -B / (2 * A)
            y0 = -C / (2 * A)
            r = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            
            # For basketball, enforce stricter radius check (basketball has consistent size)
            if abs(r - self.ball_radius) > self.ball_radius * 0.4:
                raise ValueError("Fitted radius too different from basketball radius")
                
            return np.array([x0, y0]), r
        
        # For more points, use optimized least squares method
        # Center the data for numerical stability - use mean instead of full centering
        mean_x = np.mean(points_2d[:, 0])
        mean_y = np.mean(points_2d[:, 1])
        x = points_2d[:, 0] - mean_x
        y = points_2d[:, 1]
        
        # Simplified matrix calculations - avoid full matrix ops when possible
        sum_x2 = np.sum(x**2)
        sum_y2 = np.sum(y**2)
        sum_xy = np.sum(x*y)
        sum_x3 = np.sum(x**3)
        sum_y3 = np.sum(y**3)
        sum_xy2 = np.sum(x*y**2)
        sum_x2y = np.sum(x**2*y)
        
        # Calculate matrix A and B
        A = np.array([[sum_x2, sum_xy], [sum_xy, sum_y2]])
        B = np.array([sum_x3 + sum_xy2, sum_x2y + sum_y3]) / 2
        
        # Solve linear system using direct method (faster than lstsq for 2x2)
        det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
        if abs(det) < 1e-10:
            # Fallback to standard method if matrix is singular
            c = np.linalg.lstsq(A, B, rcond=None)[0]
        else:
            c = np.array([
                (A[1,1]*B[0] - A[0,1]*B[1])/det,
                (A[0,0]*B[1] - A[1,0]*B[0])/det
            ])
        
        # Calculate center and radius
        x0 = c[0] + mean_x
        y0 = c[1] + mean_y
        r = np.sqrt(c[0]**2 + c[1]**2 + (sum_x2 + sum_y2)/len(points_2d))
        
        return np.array([x0, y0]), r
    
    def publish_ball_position(self, center, cluster_size, circle_quality, trigger_source, timestamp=None):
        """
        Publish the detected basketball position.
        Assumes basketball is always on the ground (z-height is ball radius).
        Includes temporal smoothing and consistency checking to reduce sudden jumps.
        """
        # Always set z to basketball center height since ball rolls on ground
        # This ensures we always assume basketball center is at the correct height above ground
        center[2] = self.ball_center_height
        
        # Calculate distance
        distance = np.sqrt(center[0]**2 + center[1]**2)
        
        # Apply position smoothing and consistency checks
        filtered_position = self.apply_position_smoothing(center, timestamp)
        
        # Calculate reliability score using a quality-weighted assessment instead of a fixed threshold
        # For close-range detections, use a different quality threshold
        if distance < self.close_range_threshold:
            # Special case for close-range: use quality assessment
            is_reliable = circle_quality >= self.close_range_min_quality
            reliability_score = circle_quality
            
            # Add to reliability history for hysteresis
            self.reliability_history.append(is_reliable)
            
            # Apply hysteresis: require multiple consecutive reliable or unreliable detections to change state
            if len(self.reliability_history) >= 3:
                # Consider reliable if majority of recent detections were reliable
                is_reliable = sum(self.reliability_history) > len(self.reliability_history) / 2
            
            if is_reliable:
                reliability_text = f"RELIABLE (Quality: {circle_quality:.2f})"
            else:
                reliability_text = f"UNRELIABLE (Quality: {circle_quality:.2f} < {self.close_range_min_quality:.2f})"
        else:
            # For normal range: use a graduated reliability scale
            distance_factor = 1.0
            if distance < self.min_reliable_distance:
                # Scale reliability based on how close to threshold
                distance_factor = max(0.5, distance / self.min_reliable_distance)
            
            # Combined reliability score (quality and distance)
            reliability_score = circle_quality * distance_factor
            is_reliable = reliability_score >= 0.4  # Lower threshold for combined score
            
            if is_reliable:
                reliability_text = f"RELIABLE (Score: {reliability_score:.2f})"
            else:
                reliability_text = f"UNRELIABLE (Score: {reliability_score:.2f} < 0.4)"
        
        # Skip unreliable detections if configured to do so
        if not is_reliable and not self.publish_unreliable:
            if self.performance_mode != "MINIMAL":
                self.get_logger().info("Skipping publication of unreliable detection")
            return
        
        # Filter position using the shared ground position filter
        current_time = time.time()
        filtered_position = self.position_filter.update(filtered_position, current_time)
        
        # Log the detection - only in normal or efficient modes
        if self.performance_mode != "MINIMAL":
            self.get_logger().info(
                f"LIDAR: Basketball at ({filtered_position[0]:.2f}, {filtered_position[1]:.2f}, {filtered_position[2]:.2f}) meters | "
                f"Distance: {distance:.2f}m | {reliability_text} | "
                f"Quality: {circle_quality:.2f} | Triggered by: {trigger_source}"
            )
        
        # Create and publish position message
        msg = PointStamped()
        
        # Use original timestamp if provided, otherwise use current time
        if timestamp is not None:
            msg.header.stamp = timestamp
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        
        msg.header.frame_id = "lidar_frame"
        msg.point.x = float(filtered_position[0])
        msg.point.y = float(filtered_position[1])
        msg.point.z = float(filtered_position[2])
        
        # Publish position
        self.position_publisher.publish(msg)
        
        # Update statistics
        self.successful_detections += 1
        
        # Only visualize if enabled and not in MINIMAL mode
        if self.visualization_enabled and self.marker_publisher is not None and self.performance_mode != "MINIMAL":
            self.visualize_detection(filtered_position, circle_quality, trigger_source)
    
    def visualize_detection(self, center, quality, source):
        """
        Create visualization markers for the detected ball.
        Only called when visualization is enabled and we're not in MINIMAL mode.
        """
        # Skip if visualization is disabled or publisher wasn't created
        if not self.visualization_enabled or self.marker_publisher is None:
            return
            
        markers = MarkerArray()
        
        # Get visualization settings
        viz_config = self.config.get('visualization', {})
        marker_lifetime = viz_config.get('marker_lifetime', 1.0)
        
        # Create sphere marker for the ball
        ball_marker = Marker()
        ball_marker.header.frame_id = "lidar_frame"
        ball_marker.header.stamp = self.scan_timestamp
        ball_marker.ns = "basketball"
        ball_marker.id = 1
        ball_marker.type = Marker.SPHERE
        ball_marker.action = Marker.ADD
        
        # Set position
        ball_marker.pose.position.x = center[0]
        ball_marker.pose.position.y = center[1]
        ball_marker.pose.position.z = center[2]
        ball_marker.pose.orientation.w = 1.0
        
        # Set color based on source
        colors = viz_config.get('colors', {})
        
        if source.lower() == "yolo":
            color_config = colors.get('yolo', {'r': 0.0, 'g': 1.0, 'b': 0.3, 'base_alpha': 0.5})
        else:  # HSV
            color_config = colors.get('hsv', {'r': 1.0, 'g': 0.6, 'b': 0.0, 'base_alpha': 0.5})
        
        ball_marker.color.r = color_config.get('r', 0.0)
        ball_marker.color.g = color_config.get('g', 1.0)
        ball_marker.color.b = color_config.get('b', 0.3)
        
        # Adjust transparency based on quality
        base_alpha = color_config.get('base_alpha', 0.5)
        ball_marker.color.a = min(base_alpha + quality * 0.5, 1.0)
        
        # Set size (basketball diameter)
        ball_marker.scale.x = self.ball_radius * 2.0
        ball_marker.scale.y = self.ball_radius * 2.0
        ball_marker.scale.z = self.ball_radius * 2.0
        
        # Set marker lifetime
        ball_marker.lifetime.sec = int(marker_lifetime)
        ball_marker.lifetime.nanosec = int((marker_lifetime % 1) * 1e9)
        
        markers.markers.append(ball_marker)
        
        # Add text marker
        text_marker = Marker()
        text_marker.header.frame_id = "lidar_frame"
        text_marker.header.stamp = self.scan_timestamp
        text_marker.ns = "basketball_text"
        text_marker.id = 2
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        # Position text above the ball
        text_height_offset = viz_config.get('text_height_offset', 0.2)
        text_marker.pose.position.x = center[0]
        text_marker.pose.position.y = center[1]
        text_marker.pose.position.z = center[2] + text_height_offset
        text_marker.pose.orientation.w = 1.0
        
        # Set text content
        quality_pct = int(quality * 100)
        text_marker.text = f"{source}: {quality_pct}%"
        
        # Set text appearance
        text_size = viz_config.get('text_size', 0.05)
        text_marker.scale.z = text_size
        
        text_color = colors.get('text', {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0})
        text_marker.color.r = text_color.get('r', 1.0)
        text_marker.color.g = text_color.get('g', 1.0)
        text_marker.color.b = text_color.get('b', 1.0)
        text_marker.color.a = text_color.get('a', 1.0)
        
        text_marker.lifetime.sec = int(marker_lifetime)
        text_marker.lifetime.nanosec = int((marker_lifetime % 1) * 1e9)
        
        markers.markers.append(text_marker)
        
        # Publish all markers
        self.marker_publisher.publish(markers)
    
    def publish_diagnostics(self):
        """Publish diagnostic information about the node."""
        try:
            # Calculate statistics
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            if elapsed < 0.1:
                return
            
            # Calculate rates
            scan_rate = self.processed_scans / elapsed if elapsed > 0 else 0
            detection_rate = self.successful_detections / elapsed if elapsed > 0 else 0
            
            # Calculate average processing time
            avg_time = 0
            if self.detection_times:
                avg_time = sum(self.detection_times) / len(self.detection_times)
            
            # Create diagnostics message
            diagnostics = {
                "timestamp": current_time,
                "node": "lidar",
                "uptime_seconds": elapsed,
                "status": "active",
                "performance_mode": self.performance_mode,
                "system": {
                    "cpu_load": self.current_cpu_load,
                    "memory_usage": self.current_memory_usage
                },
                "health": {
                    "lidar_health": self.lidar_health,
                    "detection_health": self.detection_health,
                    "overall": (self.lidar_health * 0.7 + self.detection_health * 0.3)
                },
                "metrics": {
                    "processed_scans": self.processed_scans,
                    "successful_detections": self.successful_detections,
                    "processing_skips": self.processing_skips,
                    "scan_rate": scan_rate,
                    "detection_rate": detection_rate,
                    "avg_processing_time_ms": avg_time * 1000,
                    "sources": {
                        "yolo_detections": self.yolo_detections,
                        "hsv_detections": self.hsv_detections
                    }
                },
                "config": {
                    "ball_radius": self.ball_radius,
                    "max_distance": self.max_distance,
                    "min_points": self.min_points,
                    "visualization_enabled": self.visualization_enabled
                },
                "transforms": {
                    "camera_frame": "ascamera_color_0",  # Add this to make it clear which frame we expect
                    "published_successfully": self.transform_published_successfully,
                    "publish_attempts": self.transform_publish_attempts,
                    "publish_successes": self.transform_publish_successes
                }
            }
            
            # Publish as JSON string
            msg = String()
            msg.data = json.dumps(diagnostics)
            self.diagnostics_publisher.publish(msg)
            
            # Log basic summary - reduced logging in MINIMAL mode
            if self.performance_mode != "MINIMAL" or self.processed_scans % 5 == 0:
                self.get_logger().info(
                    f"LIDAR: Status: {scan_rate:.1f} scans/sec, "
                    f"{detection_rate:.1f} detections/sec, "
                    f"Mode: {self.performance_mode}, CPU: {self.current_cpu_load:.1f}%"
                )
            
        except Exception as e:
            self.log_error(f"Error publishing diagnostics: {str(e)}")

    def publish_debug_point(self):
        """
        Publish a debug point for calibration purposes.
        Only executed when not in MINIMAL mode.
        """
        # Skip in MINIMAL performance mode
        if self.performance_mode == "MINIMAL":
            return
            
        if self.points_array is None or len(self.points_array) == 0:
            return
        
        try:
            # Group points by distance ranges
            points = self.points_array
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            
            # Find points in different ranges
            close_indices = np.where((distances >= 0.5) & (distances < 1.0))[0]
            mid_indices = np.where((distances >= 1.0) & (distances < 2.0))[0]
            far_indices = np.where((distances >= 2.0) & (distances < 3.0))[0]
            
            # Select which range to use
            if hasattr(self, 'last_debug_range'):
                if self.last_debug_range == "close" and len(mid_indices) > 0:
                    indices = mid_indices
                    range_name = "mid"
                elif self.last_debug_range == "mid" and len(far_indices) > 0:
                    indices = far_indices
                    range_name = "far"
                elif self.last_debug_range == "far" and len(close_indices) > 0:
                    indices = close_indices
                    range_name = "close"
                # Default if can't follow pattern
                elif len(mid_indices) > 0:
                    indices = mid_indices
                    range_name = "mid"
                elif len(far_indices) > 0:
                    indices = far_indices
                    range_name = "far"
                elif len(close_indices) > 0:
                    indices = close_indices
                    range_name = "close"
                else:
                    # No suitable points
                    return
            else:
                # First run, prefer mid-range
                if len(mid_indices) > 0:
                    indices = mid_indices
                    range_name = "mid"
                elif len(far_indices) > 0:
                    indices = far_indices
                    range_name = "far"
                elif len(close_indices) > 0:
                    indices = close_indices
                    range_name = "close"
                else:
                    # No suitable points
                    return
            
            # Save range for next time
            self.last_debug_range = range_name
            
            # Select a point with good Y variation
            selected_points = points[indices]
            y_values = np.abs(selected_points[:, 1])
            max_y_idx = np.argmax(y_values)
            selected_point = selected_points[max_y_idx]
            
            # Create and publish message
            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = "lidar_frame"
            point_msg.point.x = float(selected_point[0])
            point_msg.point.y = float(selected_point[1])
            point_msg.point.z = float(self.ball_center_height)  # Set to expected height
            
            # Use debug publisher instead of position publisher
            self.debug_publisher.publish(point_msg)
            
            # Log for calibration
            distance = np.sqrt(selected_point[0]**2 + selected_point[1]**2)
            self.get_logger().info(
                f"CALIBRATION: Debug point at ({selected_point[0]:.3f}, "
                f"{selected_point[1]:.3f}, {self.ball_center_height:.3f}), "
                f"distance: {distance:.2f}m, range: {range_name}"
            )
            
        except Exception as e:
            self.log_error(f"Error publishing debug point: {str(e)}")
    
    def log_error(self, message):
        """Log an error and update health status."""
        # Add to error collection
        current_time = time.time()
        self.errors.append({
            "timestamp": current_time,
            "message": message
        })
        
        # Update health
        self.last_error_time = current_time
        self.lidar_health = max(0.3, self.lidar_health - 0.2)
        
        # Log the error
        self.get_logger().error(f"LIDAR ERROR: {message}")

    def estimate_3d_from_2d(self, detection_msg, bbox_width, bbox_height):
        """
        Estimate a 3D position from a 2D detection and bbox dimensions.
        Similar to the fusion node's implementation but optimized for LIDAR use.
        
        Args:
            detection_msg (PointStamped): The 2D detection message
            bbox_width (float): Width of bounding box in pixels
            bbox_height (float): Height of bounding box in pixels
            
        Returns:
            np.ndarray: Estimated 3D position [x, y, z] or None if estimation fails
        """
        try:
            # Known basketball diameter in meters
            basketball_diameter_meters = self.ball_radius * 2
            
            # Calculate distance based on apparent size vs actual size
            focal_length_pixels = 345.58  # Calibrated focal length for camera
            estimated_distance = (basketball_diameter_meters * focal_length_pixels) / bbox_width
            
            # Get camera frame
            camera_frame = detection_msg.header.frame_id or "ascamera_color_0"
            
            # Get camera to lidar frame transform
            transform = self.tf_buffer.lookup_transform(
                "lidar_frame",
                camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            
            # Extract camera position in lidar frame
            camera_pos_x = transform.transform.translation.x
            camera_pos_y = transform.transform.translation.y
            camera_pos_z = transform.transform.translation.z
            
            # Get image dimensions
            image_width = 320  # Width of the camera image
            image_height = 320  # Height of the camera image
            image_center_x = image_width / 2
            image_center_y = image_height / 2
            
            # Get detection coordinates
            detection_x = detection_msg.point.x
            detection_y = detection_msg.point.y
            
            # Calculate offsets from center
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
            est_z = self.ball_center_height  # Always at basketball height above ground
            
            if self.performance_mode != "MINIMAL":
                self.get_logger().info(
                    f"Estimated 3D from YOLO 2D: distance={estimated_distance:.2f}m, "
                    f"pos=({est_x:.2f}, {est_y:.2f}, {est_z:.2f})"
                )
                
            return np.array([est_x, est_y, est_z])
            
        except Exception as e:
            self.get_logger().warn(f"Error estimating 3D from 2D: {str(e)}")
            return None

    def yolo_bbox_callback(self, msg):
        """
        Process bounding box information from YOLO detection.
        
        Args:
            msg (Float32MultiArray): Bounding box data formatted as [center_x, center_y, width, height, confidence]
        """
        try:
            # Handle Float32MultiArray format for YOLO
            if hasattr(msg, 'data') and len(msg.data) >= 4:
                # Format: [center_x, center_y, width, height, confidence]
                width = msg.data[2]   # width is the 3rd value (index 2)
                height = msg.data[3]  # height is the 4th value (index 3)
                
                # Store bounding box data with timestamp
                self.yolo_bbox_data = {
                    'width': width,
                    'height': height,
                    'timestamp': time.time()
                }
                
                if self.performance_mode != "MINIMAL":
                    self.get_logger().info(f"Received YOLO bbox: {width:.1f}x{height:.1f}")
                    
        except Exception as e:
            self.get_logger().warn(f"Error processing YOLO bbox: {str(e)}")

    def apply_position_smoothing(self, new_position, timestamp=None):
        """
        Apply temporal smoothing to position estimates to reduce sudden jumps.
        Also implements velocity-based position prediction and consistency checking.
        
        Args:
            new_position: The newly detected position [x, y, z]
            timestamp: Optional timestamp for velocity calculation
            
        Returns:
            np.array: Smoothed/filtered position
        """
        # Initialize with the new position if no history
        if not self.position_history:
            self.position_history.append(new_position)
            self.previous_timestamp = time.time() if timestamp is None else timestamp.sec + timestamp.nanosec/1e9
            return new_position
        
        # Get the current position from history
        current_position = self.position_history[-1]
        current_time = time.time() if timestamp is None else timestamp.sec + timestamp.nanosec/1e9
        
        # Calculate time delta for velocity-based checks
        if self.previous_timestamp is not None:
            dt = current_time - self.previous_timestamp
        else:
            dt = 0.1  # Default if no previous timestamp
        
        # Prevent division by zero or negative time
        dt = max(dt, 0.01)
        
        # Consistency check 1: Maximum position jump
        position_delta = np.linalg.norm(new_position[:2] - current_position[:2])
        
        # Calculate current velocity and check if jump is physically plausible
        current_velocity = position_delta / dt
        
        # Check if the jump exceeds maximum allowed distance
        if position_delta > self.max_position_jump:
            # Log suspect movement (only in normal mode)
            if self.performance_mode != "MINIMAL":
                self.get_logger().warn(
                    f"Position jump too large: {position_delta:.2f}m > {self.max_position_jump:.2f}m, "
                    f"velocity: {current_velocity:.2f}m/s"
                )
            
            # Velocity-based prediction: If we have velocity history, use it
            if len(self.velocity_history) > 0:
                avg_velocity = np.mean(self.velocity_history)
                # Consider if current velocity is physically plausible compared to recent
                if current_velocity > avg_velocity * 2 and current_velocity > self.max_speed:
                    # Position is likely bad, use predicted position instead
                    if self.performance_mode != "MINIMAL":
                        self.get_logger().info(f"Using velocity-based prediction instead of suspect position")
                    
                    # Predict position based on prior velocity, limiting to max speed
                    limited_velocity = min(avg_velocity, self.max_speed) 
                    
                    # Calculate predicted position based on prior movements
                    if len(self.position_history) >= 2:
                        last_pos = self.position_history[-1]
                        prev_pos = self.position_history[-2]
                        movement_dir = last_pos[:2] - prev_pos[:2]
                        if np.linalg.norm(movement_dir) > 0.01:  # Ensure we have meaningful direction
                            movement_dir = movement_dir / np.linalg.norm(movement_dir)
                            
                            # Create predicted position
                            pred_x = last_pos[0] + movement_dir[0] * limited_velocity * dt
                            pred_y = last_pos[1] + movement_dir[1] * limited_velocity * dt
                            pred_z = self.ball_center_height  # Ball always rolls on ground
                            
                            predicted_position = np.array([pred_x, pred_y, pred_z])
                            
                            # Store prediction for debugging
                            self.predicted_position = predicted_position
                            
                            # Use smoothed position between prediction and new position (closer to prediction)
                            filtered_position = 0.8 * predicted_position + 0.2 * new_position
                            
                            # Store for future reference
                            self.position_history.append(filtered_position)
                            self.previous_timestamp = current_time
                            
                            return filtered_position
            
            # If we can't predict based on velocity, use stronger smoothing
            self.smoothing_alpha = 0.2  # Use stronger smoothing for suspect jumps
        else:
            # Normal smoothing for reasonable movement
            self.smoothing_alpha = 0.3
        
        # Apply exponential smoothing filter
        filtered_position = (
            self.smoothing_alpha * new_position + 
            (1 - self.smoothing_alpha) * current_position
        )
        
        # Calculate and store velocity for future checks
        if dt > 0:
            velocity = position_delta / dt
            self.velocity_history.append(velocity)
        
        # Store position and timestamp for next iteration
        self.position_history.append(filtered_position)
        self.previous_timestamp = current_time
        
        return filtered_position

    def autonomous_detection_callback(self):
        """
        Periodically try to find the basketball without camera detection triggers.
        Implements autonomous basketball tracking with quality-weighted confidence.
        """
        # Skip if disabled or if we're under high load
        if not self.autonomous_detection_enabled or self.performance_mode == "MINIMAL":
            return
            
        # Skip if no valid scan data
        if self.points_array is None or len(self.points_array) == 0:
            return
        
        try:
            # Try to find the basketball using RANSAC
            ball_results = self.find_basketball_ransac()
            
            # Process the result if found
            if ball_results and len(ball_results) > 0:
                # Get the best match
                best_match = ball_results[0]
                center, cluster_size, circle_quality = best_match
                
                # Get current time
                current_time = time.time()
                
                # Calculate distance from LIDAR
                distance = np.sqrt(center[0]**2 + center[1]**2)
                
                # Calculate detection confidence based on quality and distance
                # For independent detections, we apply a distance-based scaling
                distance_factor = 1.0
                if distance < 1.0:
                    distance_factor = 1.2  # Higher confidence for close detections
                elif distance > 2.0:
                    distance_factor = 0.8  # Lower confidence for far detections
                
                # Weigh quality by distance factor for final confidence
                detection_confidence = min(1.0, circle_quality * distance_factor)
                
                # Only update tracking state if we have a good confidence
                if detection_confidence >= 0.5:
                    # Update timers
                    self.last_independent_detection_time = current_time
                    
                    # Update independent confidence
                    if self.independent_confidence < 0.9:
                        # Gradually increase confidence with successful detections
                        self.independent_confidence += 0.1
                    
                    # Publish ball position with AUTONOMOUS source
                    self.publish_ball_position(center, cluster_size, circle_quality, "AUTONOMOUS")
                    
                    # Reset recovery attempts
                    self.recovery_attempts = 0
                else:
                    # Lower confidence detection, don't publish but update detection time
                    # This helps track when we last saw anything
                    if detection_confidence >= 0.3:  # At least some reasonable confidence
                        self.last_independent_detection_time = current_time
                        
                        # Log low confidence detection
                        if self.performance_mode != "MINIMAL":
                            self.get_logger().debug(
                                f"Low confidence autonomous detection: {detection_confidence:.2f}, "
                                f"quality: {circle_quality:.2f}, distance: {distance:.2f}m"
                            )
            else:
                # No ball found, handle recovery if we've had recent detections
                current_time = time.time()
                time_since_last_detection = current_time - self.last_independent_detection_time
                
                # If we recently had autonomous detections but lost them, try recovery
                if (time_since_last_detection < 2.0 and 
                    self.recovery_attempts < self.max_recovery_attempts and
                    self.previous_ball_position is not None):
                    
                    if self.performance_mode != "MINIMAL":
                        self.get_logger().info(
                            f"Attempting recovery of recent autonomous detection "
                            f"(attempt {self.recovery_attempts+1}/{self.max_recovery_attempts})"
                        )
                    
                    # Try a targeted search around the last known position
                    # This could expand the search radius or use prediction
                    recovery_results = self.attempt_recovery()
                    
                    if recovery_results:
                        # Handle successful recovery similarly to regular detection
                        rec_center, rec_cluster_size, rec_quality = recovery_results
                        
                        # Lower quality threshold for recovery, but still maintain standards
                        if rec_quality >= 0.3:  # Lower threshold during recovery
                            if self.performance_mode != "MINIMAL":
                                self.get_logger().info("Successfully recovered autonomous detection")
                                
                            # Publish the recovered position
                            self.publish_ball_position(
                                rec_center, rec_cluster_size, rec_quality, "AUTONOMOUS_RECOVERY"
                            )
                            
                            # Update timers
                            self.last_independent_detection_time = current_time
                            
                            # Reset recovery attempts after success
                            self.recovery_attempts = 0
                            return
                    
                    # Increment recovery attempts
                    self.recovery_attempts += 1
                
        except Exception as e:
            self.log_error(f"Error in autonomous detection: {str(e)}")
    
    def attempt_recovery(self):
        """
        Attempt to recover a recently lost basketball detection.
        Uses expanded search parameters and relaxed criteria.
        
        Returns:
            tuple: (center, cluster_size, quality) if successful, None otherwise
        """
        # Need a previous position to attempt recovery
        if self.previous_ball_position is None:
            return None
        
        try:
            # Create expanded search parameters
            prev_pos = self.previous_ball_position
            
            # Get all points near the previous position
            if self.points_array is None or len(self.points_array) == 0:
                return None
                
            # Use a larger search radius for recovery
            recovery_radius = self.max_distance * 5  # Much larger radius
            
            # Find all points near previous position - vectorized for speed
            distances = np.sqrt(
                (self.points_array[:, 0] - prev_pos[0])**2 + 
                (self.points_array[:, 1] - prev_pos[1])**2
            )
            
            nearby_indices = np.where(distances < recovery_radius)[0]
            
            if len(nearby_indices) < self.min_points:
                return None
                
            # Get points near previous position
            nearby_points = self.points_array[nearby_indices]
            
            # Calculate distance from LIDAR for quality adaptation
            prev_distance = np.sqrt(prev_pos[0]**2 + prev_pos[1]**2)
            
            # Try fitting a circle with more iterations and looser thresholds
            center, inlier_count, quality = self.ransac_circle_fit(
                nearby_points,
                self.ransac_max_iterations * 2,  # Use more iterations for recovery
                self.ransac_inlier_threshold * 1.5,  # Looser threshold
                distance=prev_distance
            )
            
            # Check if we found anything
            if center is not None and inlier_count >= self.min_points:
                # Calculate distance from previous position for continuity check
                recovery_jump = np.sqrt(
                    (center[0] - prev_pos[0])**2 + 
                    (center[1] - prev_pos[1])**2
                )
                
                # Only accept recovery if it's not too far from previous position
                # Allow larger jumps for longer time since detection
                time_since_last = time.time() - self.last_independent_detection_time
                max_allowed_jump = 0.3 + time_since_last * 0.5  # Allow 30cm + 50cm/s * time
                
                if recovery_jump <= max_allowed_jump:
                    return (center, inlier_count, quality)
            
            return None
            
        except Exception as e:
            self.get_logger().warn(f"Recovery attempt failed: {str(e)}")
            return None

def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    # Create and spin node
    detector = BasketballLidarDetector()
    
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        detector.get_logger().info("Shutting down (Ctrl+C)")
    except Exception as e:
        detector.get_logger().error(f"Error: {str(e)}")
    finally:
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()