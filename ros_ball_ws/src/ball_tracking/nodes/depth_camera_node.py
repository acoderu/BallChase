#!/usr/bin/env python3

"""
Raspberry Pi 5 Optimized Basketball Tracking - Depth Camera Node
================================================================

High-performance implementation for basketball tracking designed 
specifically for the Raspberry Pi 5's resource constraints.
"""
# Standard library imports
import sys
import os
import yaml
from collections import deque
import time

# Add the parent directory of 'config' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# ROS2 message types
from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
import tf2_geometry_msgs

# Third-party libraries
import numpy as np
from cv_bridge import CvBridge

# Project utilities
from utilities.resource_monitor import ResourceMonitor
from utilities.time_utils import TimeUtils
from config.config_loader import ConfigLoader


# Config loading
config_loader = ConfigLoader()
config = config_loader.load_yaml('depth_config.yaml')

# Configuration from config file
DEPTH_CONFIG = config.get('depth', {
    "scale": 0.001,           # Depth scale factor (converts raw depth to meters)
    "min_depth": 0.1,         # Minimum valid depth in meters
    "max_depth": 8.0,         # Maximum valid depth in meters
    "radius": 5,              # Radius around detection point to sample depth values
    "min_valid_points": 5,    # Minimum number of valid points required for reliable estimation
    "calibration_file": "depth_camera_calibration.yaml"  # Calibration parameters file
})

# Topic configuration from config file
TOPICS = config.get('topics', {
    "input": {
        "camera_info": "/ascamera/camera_publisher/depth0/camera_info",
        "depth_image": "/ascamera/camera_publisher/depth0/image_raw",
        "yolo_detection": "/tennis_ball/yolo/position",
        "hsv_detection": "/tennis_ball/hsv/position"
    },
    "output": {
        "yolo_3d": "/tennis_ball/yolo/position_3d",
        "hsv_3d": "/tennis_ball/hsv/position_3d",
        "combined": "/tennis_ball/detected_position"  # Legacy/combined topic
    }
})

# Diagnostic configuration
DIAG_CONFIG = config.get('diagnostics', {
    "log_interval": 15.0,      # Increased from 10.0 to 15.0 seconds
    "threads": 3,             # Using 3 threads for Raspberry Pi 5
})

# Define the common reference frame for the robot
COMMON_REFERENCE_FRAME = config.get('frames', {
    "reference_frame": "base_link",  # Common reference frame for all sensors
    "transform_timeout": 0.1          # Timeout for transform lookups in seconds
})

# Minimal logging by default
MINIMAL_LOGGING = True
LOG_INTERVAL = 15.0  # Log interval in seconds


class DepthCorrector:
    """Applies calibration corrections to depth camera measurements."""
    
    def __init__(self, calibration_file):
        """Initialize the depth corrector with calibration parameters."""
        self.correction_type = None
        self.parameters = None
        self.mean_error = None
        self.loaded = self.load_calibration(calibration_file)
        
        if not self.loaded:
            print("WARNING: No valid calibration found - using identity correction")
            self.correction_type = "identity"
    
    def load_calibration(self, calibration_file):
        """Load calibration parameters from file."""
        try:
            if os.path.exists(calibration_file):
                with open(calibration_file, 'r') as f:
                    calibration = yaml.safe_load(f)
                
                # Extract calibration parameters
                correction_data = calibration.get('depth_correction', {})
                self.correction_type = correction_data.get('type')
                self.parameters = correction_data.get('parameters', [])
                self.mean_error = correction_data.get('mean_error')
                
                if self.correction_type and self.parameters:
                    print(f"Loaded depth correction: {self.correction_type}")
                    print(f"Parameters: {self.parameters}")
                    print(f"Expected accuracy: ±{self.mean_error:.3f}m")
                    return True
                else:
                    print(f"WARNING: Invalid calibration format in {calibration_file}")
                    return False
            else:
                print(f"WARNING: Calibration file '{calibration_file}' not found")
                return False
        except Exception as e:
            print(f"ERROR loading calibration: {str(e)}")
            return False
    
    def correct_depth(self, measured_depth):
        """Apply calibration correction to a depth measurement."""
        if not self.loaded or self.correction_type is None:
            return measured_depth  # No correction available
        
        try:
            if self.correction_type == "linear":
                # Linear correction: true = a * measured + b
                a, b = self.parameters
                return a * measured_depth + b
            
            elif self.correction_type == "polynomial":
                # Polynomial correction: true = a * measured^2 + b * measured + c
                a, b, c = self.parameters
                return a * measured_depth**2 + b * measured_depth + c
            
            else:
                # Unknown or identity correction type
                return measured_depth
        except Exception:
            # If any error occurs, return original measurement
            return measured_depth


class OptimizedPositionEstimator(Node):
    """
    A ROS2 node that converts 2D ball detections to 3D positions.
    Optimized for basketball tracking on Raspberry Pi 5.
    """
    
    def __init__(self):
        """Initialize the 3D position estimator node with all required components."""
        super().__init__('tennis_ball_3d_position_estimator')
        
        # Initialize core attributes
        self._init_attributes()
        
        # Initialize the depth corrector for calibration
        calibration_file = DEPTH_CONFIG.get("calibration_file", "depth_camera_calibration.yaml")
        self.depth_corrector = DepthCorrector(calibration_file)
        
        # Setup in logical order
        self._setup_callback_group()
        self._init_camera_parameters()
        self._setup_tf2()
        self._setup_subscriptions()
        self._setup_publishers()
        
        # Performance optimization settings
        self.use_roi_only = True  # Process only regions of interest
        self.roi_size = 20       # 30x30 pixel region around detection
        self.log_cache_info = True  # Enable cache debugging
        self.max_cpu_target = 80.0  # Target max CPU usage
        self.last_cpu_log = 0  # Timestamp for CPU logging
        
        # Fixed caching structure that separates 2D and 3D positions
        self.detection_cache = {
            'YOLO': {'detection_2d': None, 'position_3d': None, 'timestamp': 0},
            'HSV': {'detection_2d': None, 'position_3d': None, 'timestamp': 0}
        }
        self.cache_hits = 0
        self.total_attempts = 0
        
        # Initialize the resource monitor with reduced frequency
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=30.0,  # Increased from 20.0 to 30.0 seconds
            enable_temperature=False
        )
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.resource_monitor.start()
        
        # Performance adjustment timer (reduced frequency)
        self.performance_timer = self.create_timer(10.0, self._adjust_performance)
        self.diagnostics_timer = self.create_timer(15.0, self.publish_system_diagnostics)
        
        # Log initialization (minimal)
        self.get_logger().info("Pi-Optimized 3D Position Estimator initialized")
    
    def _init_attributes(self):
        """Initialize all attributes with default values."""
        # Performance settings
        self.process_every_n_frames = 2  # Process every 2nd frame by default
        self.frame_counter = 0
        self.current_cpu_usage = 0.0
        self._scale_factor = float(DEPTH_CONFIG["scale"])
        self._min_valid_depth = float(DEPTH_CONFIG["min_depth"])
        self._max_valid_depth = float(DEPTH_CONFIG["max_depth"])
        
        # Debug flags
        self.debug_mode = False  # Full debug with verbose logging - off by default
        self.last_debug_log = 0
        
        # Performance tracking
        self.start_time = TimeUtils.now_as_float()
        self.successful_conversions = 0
        self.fps_history = deque(maxlen=5)  # Reduced from 10 to 5
        self.verified_transform = False
        self.camera_info_logged = False
        
        # Position tracking and filtering - minimal
        self.last_position = None
        self.last_position_time = 0
        self.position_filter_alpha = 0.8  # Higher alpha for more responsive tracking
        
        # Max allowed position change per second
        self.max_position_change = 4.0  # Increased for more responsive tracking
        
        # Detection tracking
        self.detection_history = {
            'YOLO': {'latest_position': None, 'last_time': 0},
            'HSV': {'latest_position': None, 'last_time': 0}
        }
        
        # Error tracking
        self.error_last_logged = {}
        
        # Timestamps
        self.last_resource_alert_time = 0
        self.last_cache_log_time = 0
        self.last_diag_log_time = 0
        self.transform_not_verified_logged = False
        self.last_detection_time = TimeUtils.now_as_float()
        
        # Transform cache
        self.transform_cache = {}
        self.transform_cache_lifetime = 10.0  # 10 seconds for Pi optimizations
        
        # Bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Status counters
        self.current_fps = 0.0
        self.last_fps_update = 0
    
    def _setup_callback_group(self):
        """Set up callback group and QoS profile for subscriptions."""
        # Single reentrant callback group
        self.callback_group = ReentrantCallbackGroup()
        
        # QoS profile with minimal buffer sizes
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # Minimal buffering
        )
    
    def _init_camera_parameters(self):
        """Initialize camera and detection parameters."""
        # Camera parameters (will be updated from camera_info)
        self.camera_info = None
        self.depth_array = None
        self.depth_header = None
        
        # Camera intrinsics
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        
        # Depth image resolution (default values)
        self.depth_width = 640
        self.depth_height = 480
        
        # Coordinate scaling factors
        self.x_scale = 2.0  # Default values
        self.y_scale = 2.0
    
    def _setup_tf2(self):
        """Set up tf2 components for coordinate transformations."""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Add common reference frame
        self.reference_frame = COMMON_REFERENCE_FRAME["reference_frame"]
        self.transform_timeout = COMMON_REFERENCE_FRAME["transform_timeout"]
        
        # Create static transform broadcaster for camera transform
        self.static_broadcaster = StaticTransformBroadcaster(self)
        
        # Publish the static transform immediately
        self._publish_static_transform()
        
        # Schedule a check to verify transform is properly registered
        self.transform_check_timer = self.create_timer(2.0, self._verify_transform)
    
    def _publish_static_transform(self):
        """Publish static transform between camera_frame and base_link."""
        # Create the transform
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.reference_frame
        transform.child_frame_id = "camera_frame"
        
        # Set the camera position relative to base_link (in meters)
        transform.transform.translation.x = 0.1016  # Camera is 4 inches in front of LIDAR
        transform.transform.translation.y = 0.0     # Camera is centered
        transform.transform.translation.z = 0.1524  # Camera is 6 inches ABOVE base_link
        
        # +90-degree rotation around Y-axis for proper coordinate mapping
        import math
        angle = math.pi/2  # +90 degrees in radians
        transform.transform.rotation.w = math.cos(angle/2)
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = math.sin(angle/2)
        transform.transform.rotation.z = 0.0
        
        # Publish the transform
        self.static_broadcaster.sendTransform(transform)
        
        # Only log once
        if not hasattr(self, 'transform_published') or not self.transform_published:
            self.get_logger().info("Published static transform: camera_frame -> base_link")
            self.transform_published = True
    
    def _verify_transform(self):
        """Verify transform is registered and cancel verification timer if successful."""
        try:
            if self.tf_buffer.can_transform(
                self.reference_frame,
                "camera_frame",
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            ):
                self.verified_transform = True
                self.get_logger().info("Transform verification successful")
                self.transform_check_timer.cancel()
                return
        except Exception:
            pass
            
        # If we get here, either transform is not ready or an exception occurred
        # Republish transform and continue checking
        self._publish_static_transform()
    
    def _setup_subscriptions(self):
        """Set up all subscriptions for this node."""
        # Subscribe to camera calibration information
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            TOPICS["input"]["camera_info"],
            self.camera_info_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Subscribe to depth image with optimized QoS
        self.depth_sub = self.create_subscription(
            Image,
            TOPICS["input"]["depth_image"],
            self.depth_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )
        
        # Subscribe to YOLO ball detections
        self.yolo_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["yolo_detection"],
            self.yolo_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )
        
        # Subscribe to HSV ball detections
        self.hsv_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["hsv_detection"],
            self.hsv_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )
    
    def _setup_publishers(self):
        """Set up all publishers for this node."""
        # Separate publishers for YOLO and HSV 3D positions
        self.yolo_3d_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["yolo_3d"],
            10
        )
        
        self.hsv_3d_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["hsv_3d"],
            10
        )
        
        # Combined publisher (for backward compatibility)
        self.position_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["combined"],
            10
        )
        
        # Diagnostics publisher
        self.system_diagnostics_publisher = self.create_publisher(
            String,
            "/tennis_ball/depth_camera/diagnostics",
            10
        )
    
    def log_error(self, error_message, is_warning=False):
        """Simplified error logging with rate limiting."""
        current_time = TimeUtils.now_as_float()
        
        # Simple rate limiting - log each error type at most once every 20 seconds
        if error_message not in self.error_last_logged or current_time - self.error_last_logged[error_message] > 20.0:
            self.error_last_logged[error_message] = current_time
            
            if is_warning:
                self.get_logger().warning(f"DEPTH: {error_message}")
            else:
                self.get_logger().error(f"DEPTH: {error_message}")
    
    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        self.camera_info = msg
        
        # Cache intrinsics for faster access
        self.fx = float(msg.k[0])  # Focal length x
        self.fy = float(msg.k[4])  # Focal length y
        self.cx = float(msg.k[2])  # Principal point x (optical center)
        self.cy = float(msg.k[5])  # Principal point y (optical center)
        
        # Update image dimensions and scaling factors
        self.depth_width = msg.width
        self.depth_height = msg.height
        
        # Log camera info once (first time received)
        if not self.camera_info_logged:
            self.get_logger().info(f"Camera info received: {self.depth_width}x{self.depth_height}")
            self.get_logger().info(f"Scaling factors: x_scale={self.x_scale}, y_scale={self.y_scale}")
            self.camera_info_logged = True

    def depth_callback(self, msg):
        """Process depth image with efficient frame skipping."""
        try:
            # Skip frames based on current setting
            self.frame_counter += 1
            if self.frame_counter % self.process_every_n_frames != 0:
                return
                
            # Convert with CvBridge
            self.depth_array = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_header = msg.header
            
        except Exception as e:
            self.log_error(f"Depth processing error: {str(e)}")
    
    def yolo_callback(self, msg):
        """Handle YOLO detections."""
        self._process_detection(msg, 'YOLO')
    
    def hsv_callback(self, msg):
        """Handle HSV detections."""
        self._process_detection(msg, 'HSV')
        
    def _process_detection(self, msg, source):
        """Ultra-optimized detection processing."""
        # Skip if we don't have depth data yet
        if self.depth_array is None or self.camera_info is None:
            return
            
        # Skip if transform not verified
        if not self.verified_transform:
            return
            
        # Check cache first for performance
        if self._check_position_cache(msg, source):
            return
            
        # Process detection
        self._get_3d_position(msg, source)
    
    def _check_position_cache(self, msg, source):
        """Compare in the same coordinate space for proper caching."""
        # Skip cache check if no cached position
        if self.detection_cache[source]['detection_2d'] is None:
            return False
                
        # Get current detection and cached 2D detection (same coordinate space)
        curr_x, curr_y = msg.point.x, msg.point.y
        cached_detection = self.detection_cache[source]['detection_2d']
        cached_x, cached_y = cached_detection.point.x, cached_detection.point.y
        
        # Calculate 2D distance in the SAME coordinate space
        dx = curr_x - cached_x
        dy = curr_y - cached_y
        dist_sq = dx*dx + dy*dy
        
        # Cache timing check
        curr_time = time.time()
        cached_time = self.detection_cache[source]['timestamp']
        
        # Reasonable thresholds for 2D image space
        movement_threshold = 0.3  # Much smaller in 2D space
        cache_duration = 0.5       # 500ms validity
        
        # Debug log the actual values
        if self.log_cache_info and self.total_attempts % 20 == 0:
            self.get_logger().info(
                f"Cache check: current=({curr_x:.2f},{curr_y:.2f}), "
                f"cached=({cached_x:.2f},{cached_y:.2f}), dist={dist_sq:.4f}"
            )
        
        # Use cache if position is similar and cache is fresh
        if dist_sq < movement_threshold and curr_time - cached_time < cache_duration:
            cached_3d = self.detection_cache[source]['position_3d']
            
            # Create new message with updated timestamp
            new_msg = PointStamped()
            new_msg.header.frame_id = cached_3d.header.frame_id
            new_msg.header.stamp = self.get_clock().now().to_msg()
            new_msg.point = cached_3d.point
            
            # Publish
            if source == 'YOLO':
                self.yolo_3d_publisher.publish(new_msg)
            else:
                self.hsv_3d_publisher.publish(new_msg)
                
            self.position_publisher.publish(new_msg)
            
            # Count cache hit
            self.cache_hits += 1
            self.successful_conversions += 1
            
            # Log cache hit
            if self.log_cache_info and self.cache_hits % 10 == 0:
                self.get_logger().info(f"Cache hit #{self.cache_hits}: dist={dist_sq:.4f}")
            
            return True
        
        # Log cache miss occasionally
        if self.log_cache_info and self.total_attempts % 20 == 0:
            self.get_logger().info(
                f"Cache miss: dist={dist_sq:.4f} > {movement_threshold}, "
                f"age={curr_time-cached_time:.3f}s > {cache_duration}"
            )
        
        # Update total attempts
        self.total_attempts += 1
        
        return False
    
    def _ultra_fast_depth(self, pixel_x, pixel_y):
        """Ultra-minimal depth processing with ROI."""
        try:
            # Use ROI-only mode for drastic performance improvement
            # Process ONLY a tiny area around the detection instead of full frame
            
            # Define a very small region size
            roi_size = self.roi_size if hasattr(self, 'roi_size') else 30
            
            # Calculate region bounds
            y_min = max(0, pixel_y - roi_size//2)
            y_max = min(self.depth_array.shape[0], y_min + roi_size)
            x_min = max(0, pixel_x - roi_size//2)
            x_max = min(self.depth_array.shape[1], x_min + roi_size)
            
            # Extract just the ROI - dramatically smaller data
            roi = self.depth_array[y_min:y_max, x_min:x_max]
            
            # Direct calculation with minimal operations
            nonzeros = roi[roi > 0]
            
            # If we have any valid points, use median
            if len(nonzeros) >= 3:
                # Convert to meters directly
                depth = float(np.median(nonzeros)) * self._scale_factor
                
                # Validate range (simple bounds check)
                if self._min_valid_depth < depth < self._max_valid_depth:
                    return depth, len(nonzeros)
            
            # Fallback: Check just the immediate neighbors
            # These are direct array accesses - extremely fast
            radius = 2
            values = []
            for y in range(pixel_y-radius, pixel_y+radius+1):
                for x in range(pixel_x-radius, pixel_x+radius+1):
                    if 0 <= y < self.depth_array.shape[0] and 0 <= x < self.depth_array.shape[1]:
                        val = int(self.depth_array[y, x])
                        if val > 0:
                            values.append(val * self._scale_factor)
            
            # If we found any valid neighbors
            if values:
                # Simple average - faster than median for small lists
                depth = sum(values) / len(values)
                if self._min_valid_depth < depth < self._max_valid_depth:
                    return depth, len(values)
                
            # Ultimate fallback - use default
            return 1.2, 0  # Default 1.2m depth
                
        except Exception:
            return 1.2, 0  # Default on error
    
    def _get_3d_position(self, msg, source):
        """Convert a 2D ball detection to a 3D position using depth data."""
        try:
            # Get 2D coordinates from detection
            orig_x = float(msg.point.x)
            orig_y = float(msg.point.y)
            
            # Scale coordinates to depth image space
            pixel_x = int(round(orig_x * self.x_scale))
            pixel_y = int(round(orig_y * self.y_scale))
            
            # Constrain to valid image bounds with margin
            depth_height, depth_width = self.depth_array.shape
            margin = 10
            
            pixel_x = max(margin, min(pixel_x, depth_width - margin - 1))
            pixel_y = max(margin, min(pixel_y, depth_height - margin - 1))
            
            # Get depth using fast estimation
            median_depth, valid_points = self._ultra_fast_depth(pixel_x, pixel_y)
            
            # Convert to 3D using the pinhole camera model
            x = float((pixel_x - self.cx) * median_depth / self.fx)
            y = float((pixel_y - self.cy) * median_depth / self.fy)
            z = float(median_depth)
            
            # Create the 3D position message in camera frame
            camera_position_msg = PointStamped()
            camera_position_msg.header.stamp = self.get_clock().now().to_msg()
            camera_position_msg.header.frame_id = "camera_frame"
            camera_position_msg.point.x = x
            camera_position_msg.point.y = y
            camera_position_msg.point.z = z
            
            # Transform position to common reference frame
            transformed_msg = self._fast_transform(camera_position_msg)
            if transformed_msg is not None:
                # Apply position filtering
                filtered_position = self._simple_position_filter(
                    (transformed_msg.point.x, transformed_msg.point.y, transformed_msg.point.z))
                
                # Update the message with filtered position
                filtered_msg = PointStamped()
                filtered_msg.header = transformed_msg.header
                filtered_msg.point.x = filtered_position[0]
                filtered_msg.point.y = filtered_position[1]
                filtered_msg.point.z = filtered_position[2]
                
                # Publish to source-specific topic
                if source == "YOLO":
                    self.yolo_3d_publisher.publish(filtered_msg)
                else:  # HSV
                    self.hsv_3d_publisher.publish(filtered_msg)
                
                # Also publish to combined topic
                self.position_publisher.publish(filtered_msg)
                
                # Update cache - store both 2D and 3D positions
                self.detection_cache[source]['detection_2d'] = msg         # Original 2D detection
                self.detection_cache[source]['position_3d'] = filtered_msg  # Processed 3D result
                self.detection_cache[source]['timestamp'] = time.time()
                
                # Count successful conversion
                self.successful_conversions += 1
                
                # Log position every 50 successful conversions
                if self.successful_conversions % 50 == 0:
                    self._update_fps()
                    self.get_logger().info(
                        f"3D position ({source}): "
                        f"({filtered_position[0]:.2f}, {filtered_position[1]:.2f}, {filtered_position[2]:.2f})m | "
                        f"FPS: {self.current_fps:.1f}"
                    )
                
                return True
            else:
                return False
                
        except Exception as e:
            self.log_error(f"Error in 3D conversion: {str(e)}")
            return False
    
    def _fast_transform(self, point_stamped):
        """Optimized transform with aggressive caching."""
        # Unique key for this transform
        frame_key = f"{self.reference_frame}_{point_stamped.header.frame_id}"
        curr_time = time.time()
        
        # Check cache first
        if frame_key in self.transform_cache:
            cached_time, cached_transform = self.transform_cache[frame_key]
            
            # Use cache if fresh (10 second validity)
            if curr_time - cached_time < self.transform_cache_lifetime:
                try:
                    transformed = tf2_geometry_msgs.do_transform_point(point_stamped, cached_transform)
                    return transformed
                except Exception:
                    # If transform fails, remove from cache and try new lookup
                    del self.transform_cache[frame_key]
        
        # Get new transform
        try:
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                point_stamped.header.frame_id,
                rclpy.time.Time())
            
            # Cache it
            self.transform_cache[frame_key] = (curr_time, transform)
            
            # Apply transform
            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed
        except Exception as e:
            if self.debug_mode:
                self.get_logger().error(f"Transform lookup error: {str(e)}")
            return None
    
    def _simple_position_filter(self, position):
        """Simple position filter that smooths tracking without much overhead."""
        # If this is first position, just use it
        if self.last_position is None:
            self.last_position = position
            self.last_position_time = time.time()
            return position
        
        # Apply simple exponential filter
        alpha = self.position_filter_alpha
        x, y, z = position
        prev_x, prev_y, prev_z = self.last_position
        
        # Calculate filtered position
        filtered_x = alpha * x + (1 - alpha) * prev_x
        filtered_y = alpha * y + (1 - alpha) * prev_y
        filtered_z = alpha * z + (1 - alpha) * prev_z
        
        # Create filtered position tuple
        filtered_position = (float(filtered_x), float(filtered_y), float(filtered_z))
        
        # Update last position
        self.last_position = filtered_position
        self.last_position_time = time.time()
        
        return filtered_position
    
    def _update_fps(self):
        """Update FPS calculation."""
        curr_time = time.time()
        
        # Only update every second
        if curr_time - self.last_fps_update > 1.0:
            elapsed = curr_time - self.start_time
            
            if elapsed > 0:
                self.current_fps = self.successful_conversions / elapsed
                self.fps_history.append(self.current_fps)
                
            self.last_fps_update = curr_time
    
    def _adjust_performance(self):
        """Adaptive performance adjustment based on CPU usage and FPS."""
        # Update FPS
        self._update_fps()
        
        # Get current settings
        old_skip = self.process_every_n_frames
        cpu = self.current_cpu_usage
        
        # Adjust based on CPU usage
        if cpu > 90.0 and self.process_every_n_frames < 5:
            # CPU too high, skip more frames
            self.process_every_n_frames += 1
        elif cpu < 50.0 and self.current_fps < 3.0 and self.process_every_n_frames > 1:
            # CPU has room and FPS is below target, process more frames
            self.process_every_n_frames -= 1
        
        # Constrain to valid range
        self.process_every_n_frames = max(1, min(5, self.process_every_n_frames))
        
        # Log only if changed
        if old_skip != self.process_every_n_frames:
            self.get_logger().info(
                f"Adjusted processing: 1 in {self.process_every_n_frames} frames "
                f"(CPU: {cpu:.1f}%, FPS: {self.current_fps:.1f})"
            )
    
    def publish_system_diagnostics(self):
        """Publish system diagnostics with minimal overhead."""
        # Update FPS
        self._update_fps()
        
        # Calculate metrics
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else self.current_fps
        frame_rate_pct = 100.0 / self.process_every_n_frames
        cache_hit_rate = (self.cache_hits / max(1, self.total_attempts)) * 100.0 if self.total_attempts > 0 else 0.0
        
        # Get latest position
        pos_str = ""
        if self.last_position is not None:
            pos_str = f" | Pos: ({self.last_position[0]:.2f}, {self.last_position[1]:.2f}, {self.last_position[2]:.2f})m"
        
        # Log status
        self.get_logger().info(
            f"Depth camera: {self.current_fps:.1f} FPS (avg: {avg_fps:.1f}), "
            f"CPU: {self.current_cpu_usage:.1f}%, "
            f"Frames: 1:{self.process_every_n_frames} ({frame_rate_pct:.1f}%), "
            f"Cache hits: {cache_hit_rate:.1f}%{pos_str}"
        )
        
        # Publish diagnostics
        diag_msg = String()
        diag_data = {
            "fps": self.current_fps,
            "avg_fps": avg_fps,
            "cpu": self.current_cpu_usage,
            "frame_skip": self.process_every_n_frames,
            "frame_rate_pct": frame_rate_pct,
            "cache_hit_rate": cache_hit_rate,
            "last_position": self.last_position,
            "timestamp": time.time()
        }
        diag_msg.data = str(diag_data)
        self.system_diagnostics_publisher.publish(diag_msg)
    
    def _handle_resource_alert(self, resource_type, value):
        """Fix CPU usage reporting."""
        if resource_type == 'cpu':
            # Ensure we get non-zero values
            try:
                cpu_value = float(value)
                self.current_cpu_usage = max(0.1, cpu_value)
                
                # Log significant CPU changes
                current_time = time.time()
                if cpu_value > 90.0 and current_time - self.last_cpu_log > 30.0:
                    self.get_logger().info(f"High CPU usage: {cpu_value:.1f}%")
                    self.last_cpu_log = current_time
            except (ValueError, TypeError):
                # Default value if conversion fails
                self.current_cpu_usage = 50.0
    
    def destroy_node(self):
        """Clean shutdown of the node."""
        # Clear any large stored data
        self.depth_array = None
        self.camera_info = None
        
        # Stop resource monitor
        if hasattr(self, 'resource_monitor') and self.resource_monitor:
            try:
                self.resource_monitor.stop()
            except:
                pass
        
        super().destroy_node()


def main(args=None):
    """Main function to initialize and run the 3D position estimator node."""
    rclpy.init(args=args)
    
    # Set Raspberry Pi environment variable
    os.environ['RASPBERRY_PI'] = '1'
    
    # Create and initialize the node
    node = OptimizedPositionEstimator()
    
    # Use multiple threads but not too many for Pi 5
    thread_count = DIAG_CONFIG.get("threads", 3)
    executor = MultiThreadedExecutor(num_threads=thread_count)
    executor.add_node(node)
    
    print("=================================================")
    print("Ultra-Optimized Basketball Tracking")
    print("=================================================")
    print(f"Using {thread_count} threads on Raspberry Pi 5")
    
    try:
        node.get_logger().info(f"Starting with {thread_count} threads and process_every_n_frames={node.process_every_n_frames}")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user.")
    except Exception as e:
        node.get_logger().error(f"Error: {str(e)}")
    finally:
        # Clean up
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()