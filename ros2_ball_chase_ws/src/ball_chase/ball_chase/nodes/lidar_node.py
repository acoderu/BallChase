#!/usr/bin/env python3

"""
Basketball Tracking Robot - LIDAR Detection Node
===============================================

This node processes 2D LIDAR data to detect a basketball and provide 3D position information.
It correlates LIDAR data with camera-based detections from YOLO and HSV nodes.

Features:
- Processes 2D LIDAR scans to find circular patterns matching a basketball (10-inch diameter)
- Uses YOLO and HSV detections to trigger validation of potential basketball locations
- Publishes the basketball's 3D position in the robot's coordinate frame
- Provides visualization markers for debugging in RViz
- Includes simplified detection algorithms optimized for Raspberry Pi 5

Physical Setup:
- LIDAR mounted 6 inches (15.24 cm) above ground
- Basketball diameter: 10 inches (25.4 cm)
- LIDAR beam intersects basketball at a consistent height
- Basketball rolls on ground only (center is always 5 inches above ground)
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
        self.high_load_threshold = perf_config.get('high_load_threshold', 80.0)  # CPU %
        self.low_load_threshold = perf_config.get('low_load_threshold', 50.0)    # CPU %
        
        # Processing settings
        self.max_point_limit = perf_config.get('max_point_limit', 500)
        self.dynamic_ransac_iterations = perf_config.get('dynamic_ransac_iterations', True)
        self.min_ransac_iterations = perf_config.get('min_ransac_iterations', 10)
        
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
        self.previous_ball_position = None
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
        
        # Core parameters - ensure basketball sized (10 inch diameter)
        self.ball_radius = basketball_config.get('radius', 0.127)  # 5 inches (10 inch diameter)
        self.max_distance = basketball_config.get('max_distance', 0.2)
        self.min_points = basketball_config.get('min_points', 6)
        self.detection_samples = basketball_config.get('detection_samples', 30)
        
        # Quality thresholds
        quality_thresholds = basketball_config.get('quality_threshold', {})
        self.quality_low = quality_thresholds.get('low', 0.35)
        self.quality_medium = quality_thresholds.get('medium', 0.6)
        self.quality_high = quality_thresholds.get('high', 0.8)
        
        # Physical measurements - matching basketball & setup
        physical = self.config.get('physical_measurements', {})
        self.lidar_height = physical.get('lidar_height', 0.1524)  # 6 inches
        # Ball center is always 5 inches above ground (radius) for a basketball rolling on floor
        self.ball_center_height = physical.get('ball_center_height', 0.127)  # 5 inches
        
        # Detection reliability
        reliability = self.config.get('detection_reliability', {})
        # Increased from default 0.5 to improve reliability for larger basketball
        self.min_reliable_distance = reliability.get('min_reliable_distance', 0.8)
        self.publish_unreliable = reliability.get('publish_unreliable', True)
        
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
        """Monitor system resources and adapt processing accordingly."""
        try:
            # Get CPU and memory usage
            self.current_cpu_load = psutil.cpu_percent()
            self.current_memory_usage = psutil.virtual_memory().percent
            
            # Determine performance mode based on system load
            if self.current_cpu_load > self.high_load_threshold:
                new_mode = "MINIMAL"
                # Adjust diagnostic timer for high load
                if self.performance_mode != "MINIMAL":
                    self.diagnostics_timer.timer_period_ns = int(self.diagnostics_interval_high_load * 1e9)
            elif self.current_cpu_load > self.low_load_threshold:
                new_mode = "EFFICIENT"
            else:
                new_mode = "NORMAL"
                # Restore normal diagnostic frequency if coming from high load
                if self.performance_mode == "MINIMAL":
                    self.diagnostics_timer.timer_period_ns = int(self.diagnostics_interval_normal * 1e9)
            
            # Log mode changes
            if new_mode != self.performance_mode:
                self.get_logger().info(
                    f"Performance mode change: {self.performance_mode} -> {new_mode} "
                    f"(CPU: {self.current_cpu_load:.1f}%, Memory: {self.current_memory_usage:.1f}%)"
                )
                self.performance_mode = new_mode
            
            # Publish system load for other nodes
            load_msg = Float32()
            load_msg.data = float(self.current_cpu_load)
            self.load_publisher.publish(load_msg)
            
        except Exception as e:
            self.get_logger().warn(f"Error monitoring resources: {str(e)}")

    def scan_callback(self, msg):
        """
        Process LaserScan messages from the LIDAR.
        
        Converts polar coordinates to Cartesian coordinates.
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
            
            # Sample points if there are too many (improves performance)
            if len(valid_ranges) > point_limit:
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
            
            # Find basketball in LIDAR data
            ball_results = self.find_basketball_ransac(transformed_point)
            
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
                self.ransac_inlier_threshold
            )
            
            # Check if this is better than current best
            if quality > best_quality and inlier_count >= self.min_points:
                best_center = center
                best_inlier_count = inlier_count
                best_quality = quality
        
        # Return result if found
        if best_center is not None and best_quality >= self.quality_low:
            # Store the position for future reference
            self.previous_ball_position = best_center
            
            # Update statistics
            self.consecutive_failures = 0
            self.last_successful_detection_time = time.time()
            
            # Return as list of results (keeping same format as original code)
            return [(best_center, best_inlier_count, best_quality)]
        
        return []
    
    def ransac_circle_fit(self, points, max_iterations=30, threshold=0.02):
        """
        Use RANSAC to fit a circle to points, robust to outliers.
        Optimized for performance.
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
        
        for _ in range(actual_iterations):
            # Randomly sample 3 points
            if len(points) < 3:
                continue
                
            sample_indices = np.random.choice(len(points), 3, replace=False)
            sample_points = points[sample_indices]
            
            # Fit circle to these points
            try:
                center, radius = self.fit_circle(sample_points)
                
                # Skip if radius is too different from expected
                if abs(radius - self.ball_radius) > self.ball_radius * 0.5:
                    continue
                
                # Count inliers using vectorized operations for speed
                distances = np.sqrt(
                    (x_coords - center[0])**2 + 
                    (y_coords - center[1])**2
                )
                
                # Inliers are points close to the expected circle
                inliers = np.abs(distances - radius) < threshold
                inlier_count = np.sum(inliers)
                
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_center = center
                    best_radius = radius
            except Exception:
                continue
        
        if best_center is None:
            return None, 0, 0
        
        # Refine with all inliers if we have enough
        if best_inlier_count >= 5:
            # Calculate quality metrics
            inlier_ratio = best_inlier_count / len(points)
            radius_error = abs(best_radius - self.ball_radius) / self.ball_radius
            quality = 0.7 * inlier_ratio + 0.3 * (1.0 - min(radius_error, 1.0))
            
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
        y = points_2d[:, 1] - mean_y
        
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
        """
        # Always set z to basketball radius since ball rolls on ground
        # This ensures we always assume basketball center is 5 inches above ground
        center[2] = self.ball_center_height
        
        # Calculate distance and reliability
        distance = np.sqrt(center[0]**2 + center[1]**2)
        is_reliable = distance >= self.min_reliable_distance
        
        # Adjust quality based on distance
        if not is_reliable:
            distance_factor = max(0.1, distance / self.min_reliable_distance)
            adjusted_quality = circle_quality * distance_factor
            reliability_text = f"UNRELIABLE ({distance:.2f}m < {self.min_reliable_distance:.1f}m)"
        else:
            adjusted_quality = circle_quality
            reliability_text = "RELIABLE"
        
        # Skip unreliable detections if configured to do so
        if not is_reliable and not self.publish_unreliable:
            if self.performance_mode != "MINIMAL":
                self.get_logger().info("Skipping publication of unreliable detection")
            return
        
        # Filter position using the shared ground position filter
        current_time = time.time()
        filtered_position = self.position_filter.update(center, current_time)
        
        # Log the detection - only in normal or efficient modes
        if self.performance_mode != "MINIMAL":
            self.get_logger().info(
                f"LIDAR: Basketball at ({filtered_position[0]:.2f}, {filtered_position[1]:.2f}, {filtered_position[2]:.2f}) meters | "
                f"Distance: {distance:.2f}m | {reliability_text} | "
                f"Quality: {adjusted_quality:.2f} | Triggered by: {trigger_source}"
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
        
        # With the lock, update position history
        with self.lock:
            self.position_history.append(filtered_position)
    
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