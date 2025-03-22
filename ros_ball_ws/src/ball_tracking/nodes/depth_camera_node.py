#!/usr/bin/env python3

"""
Optimized Tennis/Basketball Tracking Robot - Depth Camera Node
==============================================================

This node converts 2D ball detections from YOLO and HSV into 3D positions
using depth camera data. Optimized for basketball tracking on Raspberry Pi 5.
"""
# Standard library imports
import sys
import os
import yaml
from collections import deque

# Add the parent directory of 'config' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the 'src' directory to the Python path
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
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
import tf2_geometry_msgs

# Third-party libraries
import numpy as np
import psutil
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
    "radius": 5,              # Radius around detection point to sample depth values (increased for basketball)
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
LOG_INTERVAL = 15.0  # Increased from 10 to 15 seconds


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


class TennisBall3DPositionEstimator(Node):
    """
    A ROS2 node that converts 2D ball detections to 3D positions.
    Optimized for basketball tracking on Raspberry Pi 5.
    """
    
    def __init__(self):
        """Initialize the 3D position estimator node with all required components."""
        super().__init__('tennis_ball_3d_position_estimator')
        
        # Initialize core attributes
        self.init_attributes()
        
        # Initialize the depth corrector for calibration
        calibration_file = DEPTH_CONFIG.get("calibration_file", "depth_camera_calibration.yaml")
        self.depth_corrector = DepthCorrector(calibration_file)
        
        # Setup in logical order
        self._setup_callback_group()
        self._init_camera_parameters()
        self._setup_tf2()
        self._setup_subscriptions()
        self._setup_publishers()
        
        # Initialize the resource monitor with reduced monitoring frequency
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
        self.get_logger().info("3D Position Estimator initialized")
        
        # Log calibration status (only if available)
        if self.depth_corrector.correction_type and self.depth_corrector.parameters:
            if self.depth_corrector.correction_type != "identity":
                self.get_logger().info(f"Depth calibration active: {self.depth_corrector.correction_type}")
    
    def init_attributes(self):
        """Initialize all attributes with default values."""
        # Performance settings
        self.process_every_n_frames = 3  # Default to every 3rd frame for efficient processing
        self.frame_counter = 0
        self.current_cpu_usage = 0.0
        self._scale_factor = float(DEPTH_CONFIG["scale"])
        self._min_valid_depth = float(DEPTH_CONFIG["min_depth"])
        self._max_valid_depth = float(DEPTH_CONFIG["max_depth"])
        
        # Debug flags
        self.debug_mode = True  # Enable debug mode for moving ball tracking development
        self.last_debug_log = 0
        
        # Performance tracking
        self.start_time = TimeUtils.now_as_float()
        self.successful_conversions = 0
        self.processing_times = deque(maxlen=10)
        self.depth_camera_health = 1.0
        self.last_fps_log_time = 0
        self.fps_history = deque(maxlen=10)  # Track recent FPS values
        self.verified_transform = False
        self.camera_info_logged = False
        
        # Position tracking and filtering
        self.position_history = deque(maxlen=5)  # Store last 5 valid positions
        self.velocity = [0.0, 0.0, 0.0]  # Simple velocity estimate (m/s)
        self.last_position_time = 0
        self.position_filter_alpha = 0.3  # Weight for position smoothing (lower = more smoothing)
        
        # Max allowed position change per second (in meters)
        self.max_position_change = 2.0  # m/s
        
        # Detection tracking
        self.detection_history = {
            'YOLO': {'count': 0, 'latest_position': None, 'last_time': 0},
            'HSV': {'count': 0, 'latest_position': None, 'last_time': 0}
        }
        
        # Optimized caching mechanism for basketball
        self.position_cache = {
            'YOLO': {'position': None, 'timestamp': 0, '3d_position': None},
            'HSV': {'position': None, 'timestamp': 0, '3d_position': None}
        }
        self.cache_hits = {'YOLO': 0, 'HSV': 0}
        self.cache_validity_duration = 0.3  # 300ms validity for basketball
        self.movement_threshold = 0.2  # 20% movement threshold for basketball
        
        # Error tracking (minimal)
        self.error_counts = {}
        self.error_last_logged = {}
        
        # Timestamps
        self.last_resource_alert_time = 0
        self.last_cache_log_time = 0
        self.last_diag_log_time = 0
        self.transform_not_verified_logged = False
        self.last_detection_time = TimeUtils.now_as_float()
        self.last_batch_process = TimeUtils.now_as_float()
        
        # Simplified detection buffer
        self.detection_buffer = {'YOLO': [], 'HSV': []}
        
        # Transform cache for optimization
        self.transform_cache = {}
        self.transform_cache_lifetime = 2.0  # 2 seconds
        
        # Bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Initialize attempt counter to track detection conversion attempts
        self.attempt_counter = {'YOLO': 0, 'HSV': 0}
    
    def _setup_callback_group(self):
        """Set up callback group and QoS profile for subscriptions."""
        self.callback_group = ReentrantCallbackGroup()
        
        # QoS profile with increased depth
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=3  # Reduced from 5 to 3
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
        self.x_scale = 1.0
        self.y_scale = 1.0
    
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
        
        # INVERTED HEIGHT RELATIONSHIP:
        # The camera appears to be above the base_link, not below it
        transform.transform.translation.z = 0.1524  # Camera is 6 inches ABOVE base_link
        
        # Keep the correct +90-degree rotation around Y-axis
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
            self.get_logger().info("Published static transform: camera_frame -> base_link with proper coordinate rotation")
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
        
        # Update scaling factors with proper aspect ratio handling
        detect_width = DEPTH_CONFIG.get("detection_resolution", {}).get("width", 320)
        detect_height = DEPTH_CONFIG.get("detection_resolution", {}).get("height", 320)
        
        # Maintain aspect ratio in scaling
        if (self.depth_width / self.depth_height) > (detect_width / detect_height):
            # Width-constrained
            self.x_scale = float(self.depth_width / detect_width)
            self.y_scale = self.x_scale
        else:
            # Height-constrained
            self.y_scale = float(self.depth_height / detect_height)
            self.x_scale = self.y_scale
        
        # Log camera info once (first time received)
        if not self.camera_info_logged:
            self.get_logger().info(f"Camera info received: {self.depth_width}x{self.depth_height}")
            self.get_logger().info(f"Camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")
            self.get_logger().info(f"Scaling factors: x_scale={self.x_scale}, y_scale={self.y_scale}")
            self.camera_info_logged = True
        
        # Update camera health
        self.depth_camera_health = 1.0

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
            
            # Only log depth information in debug mode
            if self.debug_mode and self.frame_counter % 100 == 0:
                self.get_logger().info(f"Depth array shape: {self.depth_array.shape}, " 
                                      f"min: {np.min(self.depth_array)}, max: {np.max(self.depth_array)}")
            
            # Process any pending detections
            self._process_detection_batch()
            
        except Exception as e:
            self.log_error(f"Depth processing error: {str(e)}")
    
    def yolo_callback(self, msg):
        """Handle YOLO detections."""
        self.detection_callback(msg, 'YOLO')
    
    def hsv_callback(self, msg):
        """Handle HSV detections."""
        self.detection_callback(msg, 'HSV')
        
    def detection_callback(self, msg, source):
        """Generic callback for processing detections."""
        # Only log detection information in debug mode
        if self.debug_mode:
            self.get_logger().info(
                f"Detection ({source}): ({msg.point.x:.2f}, {msg.point.y:.2f})"
            )
        
        # Add to buffer
        current_time = TimeUtils.now_as_float()
        self.detection_buffer[source].append((msg, current_time))
        
        # Process immediately if we have depth data
        if self.depth_array is not None:
            if self.debug_mode:
                self.get_logger().info(f"Processing detection immediately")
            self._process_detection_batch()
    
    def _process_detection_batch(self):
        """Process all buffered detections efficiently."""
        current_time = TimeUtils.now_as_float()
        self.last_batch_process = current_time
        
        # Skip if missing required data
        if self.camera_info is None or self.depth_array is None or self.fx == 0:
            if self.debug_mode:
                self.get_logger().warning("Missing required data for processing")
            return
            
        # Process only the latest detection for each source
        for source in ['YOLO', 'HSV']:
            if self.detection_buffer[source]:
                # Get the most recent detection
                latest_msg, _ = max(self.detection_buffer[source], key=lambda x: x[1])
                
                # Debug info
                if self.debug_mode:
                    self.get_logger().info(f"Processing {source} detection in batch")
                
                # Process if needed - use caching for efficiency
                if self._should_process_detection(latest_msg, source):
                    self.get_3d_position(latest_msg, source)
                
                # Clear buffer
                self.detection_buffer[source] = []
    
    def _should_process_detection(self, msg, source):
        """Determine if we should process this detection or use cached result."""
        current_time = TimeUtils.now_as_float()
        curr_x, curr_y = msg.point.x, msg.point.y
        
        # Check cache
        if self.position_cache[source]['position'] is not None:
            prev_x, prev_y = self.position_cache[source]['position']
            cache_time = self.position_cache[source]['timestamp']
            
            # Calculate distance and time since cache
            dist_squared = (curr_x - prev_x)**2 + (curr_y - prev_y)**2
            time_since_cache = current_time - cache_time
            
            # Basketball-optimized threshold (20% of image size)
            threshold = self.movement_threshold
            
            # Log cache decisions occasionally (debug only)
            if self.debug_mode and current_time - self.last_cache_log_time > 60.0:
                self.get_logger().info(f"Cache decision: dist={dist_squared:.6f}, threshold={threshold:.6f}, time={time_since_cache:.1f}s")
                self.last_cache_log_time = current_time
            
            # Use cache if position hasn't changed much and cache is fresh
            if dist_squared < threshold and time_since_cache < self.cache_validity_duration:
                if self.position_cache[source]['3d_position'] is not None:
                    success = self._republish_cached_position(source, 
                        self.yolo_3d_publisher if source == "YOLO" else self.hsv_3d_publisher)
                    
                    if success:
                        self.cache_hits[source] += 1
                        # Log cache hit occasionally
                        if self.debug_mode and self.cache_hits[source] % 100 == 0:
                            self.get_logger().info(f"Cache hit {self.cache_hits[source]} for {source}")
                        return False  # Skip processing
        
        # Update cache with new position
        self.position_cache[source]['position'] = (curr_x, curr_y)
        self.position_cache[source]['timestamp'] = current_time
        
        return True  # Process this detection
    
    def _republish_cached_position(self, source, publisher):
        """Republish the cached 3D position with an updated timestamp."""
        if self.position_cache[source]['3d_position'] is None:
            return False
            
        # Get cached position
        cached_pos = self.position_cache[source]['3d_position']
        
        # Create new message with updated timestamp
        msg = PointStamped()
        msg.header.frame_id = cached_pos.header.frame_id
        msg.header.stamp = TimeUtils.now_as_ros_time()
        msg.point = cached_pos.point
        
        # Publish to specific publisher and to combined topic
        publisher.publish(msg)
        self.position_publisher.publish(msg)
        
        # Count as a successful conversion
        self.successful_conversions += 1
        
        return True
    
    def _get_reliable_depth_vectorized(self, depth_array, pixel_x, pixel_y):
        """
        Adaptive depth sampling for basketball tracking.
        Progressively increases sampling radius until finding enough valid points.
        """
        # Try with increasing radius sizes to handle moving ball
        for radius in [5, 10, 15, 20]:
            try:
                # Create a slice of the depth array around the target pixel
                y_min = max(0, pixel_y - radius)
                y_max = min(depth_array.shape[0], pixel_y + radius + 1)
                x_min = max(0, pixel_x - radius)
                x_max = min(depth_array.shape[1], pixel_x + radius + 1)
                
                # Get the depth values in the region
                region = depth_array[y_min:y_max, x_min:x_max].astype(np.float32)
                
                # Create a mask for valid depths (non-zero values)
                valid_mask = (region > 0)
                
                # If no valid depths, try larger radius
                if not np.any(valid_mask):
                    if self.debug_mode:
                        self.get_logger().info(f"No valid depths with radius {radius}, trying larger")
                    continue
                
                # Convert to meters
                depths_m = region[valid_mask] * self._scale_factor
                
                # Apply calibration to all valid depths
                if hasattr(self, 'depth_corrector') and self.depth_corrector.correction_type == "linear":
                    a, b = self.depth_corrector.parameters
                    depths_m = a * depths_m + b
                elif hasattr(self, 'depth_corrector') and self.depth_corrector.correction_type == "polynomial":
                    a, b, c = self.depth_corrector.parameters
                    depths_m = a * depths_m**2 + b * depths_m + c
                
                # Filter by min/max depth
                valid_depths_mask = (depths_m > self._min_valid_depth) & (depths_m < self._max_valid_depth)
                valid_depths = depths_m[valid_depths_mask]
                
                # For basketball, median is more stable than mean
                if len(valid_depths) >= 10:  # Lowered threshold from original 49
                    # Convert NumPy median to Python float for ROS2 compatibility
                    median_depth = float(np.median(valid_depths))
                    if self.debug_mode:
                        self.get_logger().info(f"Found reliable depth with radius {radius}, {len(valid_depths)} points")
                    return median_depth, 0.9, len(valid_depths)
                
                # If we didn't get enough valid points, try larger radius
                if self.debug_mode:
                    self.get_logger().info(f"Only {len(valid_depths)} valid points with radius {radius}")
                
            except Exception as e:
                if self.debug_mode:
                    self.get_logger().error(f"Error with radius {radius}: {str(e)}")
        
        # If we still don't have a reliable depth, try to estimate from previous data
        estimated_depth = self._estimate_depth_from_previous(pixel_x, pixel_y)
        if estimated_depth is not None:
            return estimated_depth, 0.5, 5  # Lower reliability for estimated values
        
        # If all else fails
        return None, 0.0, 0
    
    def _estimate_depth_from_previous(self, pixel_x, pixel_y):
        """
        Estimate depth based on previous readings and position history.
        Returns estimated depth or None if no reliable estimation can be made.
        """
        # Check if we have any previous detection history from YOLO or HSV
        for source in ['YOLO', 'HSV']:
            if source in self.detection_history and self.detection_history[source].get('latest_position'):
                # Get the most recent valid 3D position
                latest_pos = self.detection_history[source]['latest_position']
                last_time = self.detection_history[source]['last_time']
                current_time = TimeUtils.now_as_float()
                
                # Only use recent history (within 1 second)
                if current_time - last_time < 1.0:
                    if self.debug_mode:
                        self.get_logger().info(f"Estimating depth from previous {source} position: {latest_pos}")
                    # Return the Z-coordinate (depth) from the latest position
                    return float(latest_pos[2])
        
        # If no recent history, use a reasonable default for a basketball at playing distance
        if self.debug_mode:
            self.get_logger().info("No recent history available, using default depth")
        return 1.2  # Default depth value in meters (adjust based on your specific use case)
    
    def _apply_position_filter(self, position, source):
        """Apply a simple position filter to smooth the tracking."""
        current_time = TimeUtils.now_as_float()
        
        # Unpack position tuple
        x, y, z = position
        
        # If we don't have a position history yet, create it
        if len(self.position_history) == 0:
            # Just add the current position
            self.position_history.append(position)
            return position
            
        # Simple Exponential Moving Average filter
        # Apply different smoothing based on source reliability
        alpha = self.position_filter_alpha
        if source == 'HSV':
            # HSV is typically less stable, more smoothing
            alpha = max(0.2, alpha - 0.1)
            
        # Calculate filtered position
        prev_pos = self.position_history[-1]
        smoothed_x = alpha * x + (1 - alpha) * prev_pos[0]
        smoothed_y = alpha * y + (1 - alpha) * prev_pos[1]
        smoothed_z = alpha * z + (1 - alpha) * prev_pos[2]
        
        # Validate position change (limit max change)
        if len(self.position_history) > 1 and self.last_position_time > 0:
            dt = current_time - self.last_position_time
            if dt > 0:
                # Calculate change per second for each dimension
                dx_per_sec = abs(smoothed_x - prev_pos[0]) / dt
                dy_per_sec = abs(smoothed_y - prev_pos[1]) / dt
                dz_per_sec = abs(smoothed_z - prev_pos[2]) / dt
                
                # Check if any dimension exceeds max allowed change
                if (dx_per_sec > self.max_position_change or
                    dy_per_sec > self.max_position_change or
                    dz_per_sec > self.max_position_change):
                    
                    # Use a blend of previous position and validated change
                    max_change = self.max_position_change * dt
                    
                    # Limit change in each dimension
                    dx = smoothed_x - prev_pos[0]
                    dy = smoothed_y - prev_pos[1]
                    dz = smoothed_z - prev_pos[2]
                    
                    # Calculate scaling factor to limit change
                    max_component = max(abs(dx), abs(dy), abs(dz))
                    if max_component > max_change:
                        scale = max_change / max_component
                        dx *= scale
                        dy *= scale
                        dz *= scale
                    
                    # Apply limited change
                    smoothed_x = prev_pos[0] + dx
                    smoothed_y = prev_pos[1] + dy
                    smoothed_z = prev_pos[2] + dz
                    
                    if self.debug_mode:
                        self.get_logger().info(f"Limited position change: {dx_per_sec:.2f}, {dy_per_sec:.2f}, {dz_per_sec:.2f} m/s")
        
        # Create filtered position tuple
        filtered_position = (float(smoothed_x), float(smoothed_y), float(smoothed_z))
        
        # Update history and time
        self.position_history.append(filtered_position)
        self.last_position_time = current_time
        
        # Update velocity estimate (if we have at least 2 positions)
        if len(self.position_history) >= 2 and self.last_position_time > 0:
            dt = current_time - self.last_position_time
            if dt > 0:
                # Position at t-1 and t
                pos_1 = self.position_history[-2]
                pos_2 = filtered_position
                
                # Velocity = (pos_2 - pos_1) / dt
                self.velocity = [
                    (pos_2[0] - pos_1[0]) / dt,
                    (pos_2[1] - pos_1[1]) / dt,
                    (pos_2[2] - pos_1[2]) / dt
                ]
        
        return filtered_position
    
    def get_3d_position(self, detection_msg, source):
        """
        Convert a 2D ball detection to a 3D position using depth data.
        Optimized for basketball tracking with improved handling of moving balls.
        """
        # Track attempt counts
        self.attempt_counter[source] += 1
        
        # Skip processing if we're missing required data
        if self.camera_info is None or self.depth_array is None or self.fx == 0:
            if self.debug_mode:
                self.get_logger().warning(
                    f"Missing data for {source}: camera_info={self.camera_info is not None}, "
                    f"depth_array={self.depth_array is not None}, fx={self.fx}"
                )
            return False
        
        # Skip if transform is not verified yet
        if not self.verified_transform:
            if not self.transform_not_verified_logged:
                self.log_error("Transform not yet verified", True)
                self.transform_not_verified_logged = True
            return False
        
        try:
            # Get 2D coordinates from detection
            orig_x = float(detection_msg.point.x)
            orig_y = float(detection_msg.point.y)
            
            if self.debug_mode:
                self.get_logger().info(f"Processing {source} detection at ({orig_x:.2f}, {orig_y:.2f})")
            
            # Scale coordinates to depth image space
            pixel_x = int(round(orig_x * self.x_scale))
            pixel_y = int(round(orig_y * self.y_scale))
            
            # Constrain to valid image bounds with margin
            depth_height, depth_width = self.depth_array.shape
            margin = 20 + 2  # Increased margin for adaptive sampling 
            
            pixel_x = max(margin, min(pixel_x, depth_width - margin - 1))
            pixel_y = max(margin, min(pixel_y, depth_height - margin - 1))
            
            # Get raw depth at center pixel (debug only)
            if self.debug_mode:
                raw_depth = float(self.depth_array[pixel_y, pixel_x])
                self.get_logger().info(
                    f"Depth at pixel ({pixel_x}, {pixel_y}): raw={raw_depth}, "
                    f"meters={(raw_depth * self._scale_factor):.3f}m"
                )
            
            # Get depth using adaptive vectorized method
            median_depth, reliability, valid_points = self._get_reliable_depth_vectorized(
                self.depth_array, pixel_x, pixel_y)
            
            # If no reliable depth found even with adaptive sampling, try to estimate
            if median_depth is None:
                if self.debug_mode:
                    self.get_logger().warning(f"No reliable depth found at ({pixel_x}, {pixel_y}), trying to estimate")
                
                # Check if we have recent position history
                if source in self.detection_history and self.detection_history[source].get('latest_position'):
                    prev_pos = self.detection_history[source]['latest_position']
                    prev_time = self.detection_history[source]['last_time']
                    current_time = TimeUtils.now_as_float()
                    
                    # Only use recent history (within 1 second)
                    if current_time - prev_time < 1.0:
                        # Use previous depth with estimated change
                        x = float((pixel_x - self.cx) * prev_pos[2] / self.fx)
                        y = float((pixel_y - self.cy) * prev_pos[2] / self.fy)
                        z = float(prev_pos[2])  # Use previous depth
                        
                        if self.debug_mode:
                            self.get_logger().info(f"Using previous depth: {z:.3f}m from {current_time - prev_time:.3f}s ago")
                        
                        # Skip forward to transform and publish step
                        depth_is_estimated = True
                    else:
                        return False  # No recent history to use
                else:
                    return False  # No history for this source
            else:
                # Depth found successfully!
                depth_is_estimated = False
                
                # Log depth information (debug only)
                if self.debug_mode:
                    self.get_logger().info(f"Reliable depth: {median_depth:.3f}m, valid points: {valid_points}")
                
                # Convert to 3D using the pinhole camera model
                # Ensure all values are Python floats for ROS2 compatibility
                x = float((pixel_x - self.cx) * median_depth / self.fx)
                y = float((pixel_y - self.cy) * median_depth / self.fy)
                z = float(median_depth)
            
            if self.debug_mode:
                self.get_logger().info(f"3D coordinates (camera frame): x={x:.2f}, y={y:.2f}, z={z:.2f}")
            
            # Create the 3D position message in camera frame
            camera_position_msg = PointStamped()
            
            # Use current time for timestamp to avoid potential issues
            camera_position_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Set the frame ID to camera frame
            camera_position_msg.header.frame_id = "camera_frame"
            camera_position_msg.point.x = x
            camera_position_msg.point.y = y
            camera_position_msg.point.z = z
            
            # Transform position to common reference frame
            transformed_msg = self._transform_to_reference_frame(camera_position_msg)
            if transformed_msg is not None:
                # Apply position filtering to smooth tracking
                raw_position = (
                    transformed_msg.point.x,
                    transformed_msg.point.y, 
                    transformed_msg.point.z
                )
                
                # If depth was estimated, give less weight to this position
                old_alpha = self.position_filter_alpha
                if depth_is_estimated:
                    self.position_filter_alpha = 0.2  # Lower alpha for estimated positions
                
                # Apply position filtering
                filtered_position = self._apply_position_filter(raw_position, source)
                
                # Restore original alpha
                if depth_is_estimated:
                    self.position_filter_alpha = old_alpha
                
                # Update the message with filtered position
                transformed_msg.point.x = filtered_position[0]
                transformed_msg.point.y = filtered_position[1]
                transformed_msg.point.z = filtered_position[2]
                
                # Publish to source-specific topic
                if source == "YOLO":
                    self.yolo_3d_publisher.publish(transformed_msg)
                else:  # HSV
                    self.hsv_3d_publisher.publish(transformed_msg)
                
                # Also publish to combined topic
                self.position_publisher.publish(transformed_msg)
                
                # Update tracking stats
                self.successful_conversions += 1
                self.detection_history[source]['count'] += 1
                self.detection_history[source]['latest_position'] = filtered_position
                self.detection_history[source]['last_time'] = TimeUtils.now_as_float()
                
                # Store in cache for future reuse
                self.position_cache[source]['3d_position'] = transformed_msg
                
                # Calculate FPS for occasional logging
                elapsed = TimeUtils.now_as_float() - self.start_time
                current_fps = self.successful_conversions / elapsed if elapsed > 0 else 0
                if len(self.fps_history) < 10 or self.successful_conversions % 10 == 0:
                    self.fps_history.append(current_fps)
                
                # Only log occasionally for reduced verbosity
                if self.successful_conversions % 10 == 0:
                    self.get_logger().info(
                        f"3D position ({source}): ({filtered_position[0]:.2f}, "
                        f"{filtered_position[1]:.2f}, {filtered_position[2]:.2f})m | "
                        f"FPS: {current_fps:.1f}"
                    )
                
                return True
            else:
                if self.debug_mode:
                    self.get_logger().warning("Transform to reference frame failed")
                return False
            
        except Exception as e:
            self.log_error(f"Error in 3D conversion: {str(e)}")
            if self.debug_mode:
                import traceback
                self.get_logger().error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _transform_to_reference_frame(self, point_stamped):
        """Transform with caching for efficiency."""
        now = TimeUtils.now_as_float()
        frame_key = f"{self.reference_frame}_{point_stamped.header.frame_id}"
        
        if self.debug_mode:
            self.get_logger().info(f"Transforming from {point_stamped.header.frame_id} to {self.reference_frame}")
        
        # Check if we have this transform in cache and it's still valid
        if frame_key in self.transform_cache:
            cached_time, cached_transform = self.transform_cache[frame_key]
            if now - cached_time < self.transform_cache_lifetime:
                try:
                    if self.debug_mode:
                        self.get_logger().info("Using cached transform")
                    transformed = tf2_geometry_msgs.do_transform_point(point_stamped, cached_transform)
                    return transformed
                except Exception as e:
                    if self.debug_mode:
                        self.get_logger().error(f"Error applying cached transform: {str(e)}")
                    # If transform fails, remove from cache
                    del self.transform_cache[frame_key]
        
        # Otherwise get a new transform
        try:
            if self.debug_mode:
                self.get_logger().info("Looking up new transform")
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                point_stamped.header.frame_id,
                rclpy.time.Time())
            
            # Cache it
            self.transform_cache[frame_key] = (now, transform)
            
            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            
            # Log transform result (debug only)
            if self.debug_mode:
                self.get_logger().info(
                    f"Transform successful: ({point_stamped.point.x:.2f}, {point_stamped.point.y:.2f}, {point_stamped.point.z:.2f}) -> "
                    f"({transformed.point.x:.2f}, {transformed.point.y:.2f}, {transformed.point.z:.2f})"
                )
            
            return transformed
        except Exception as e:
            if self.debug_mode:
                self.get_logger().error(f"Transform lookup error: {str(e)}")
                import traceback
                self.get_logger().error(f"Transform traceback: {traceback.format_exc()}")
            return None
    
    def _adjust_performance(self):
        """
        Enhanced adaptive performance adjustment based on CPU usage, ball distance,
        and detection frequency. Optimized for basketball tracking.
        """
        cpu_usage = self.current_cpu_usage
        old_frame_skip = self.process_every_n_frames
        current_time = TimeUtils.now_as_float()
        
        # Track performance metrics
        elapsed = current_time - self.start_time
        current_fps = self.successful_conversions / elapsed if elapsed > 0 else 0
        
        # CPU-based adjustments (more conservative to leave resources for other nodes)
        if cpu_usage > 80:  # Very high CPU load
            self.process_every_n_frames = min(15, self.process_every_n_frames + 2)
        elif cpu_usage > 60:  # High CPU load
            self.process_every_n_frames = min(10, self.process_every_n_frames + 1)
        elif cpu_usage > 40:  # Medium CPU load 
            self.process_every_n_frames = min(8, max(5, self.process_every_n_frames))
        elif cpu_usage < 20 and current_fps < 3.0:  # Low CPU load and low frame rate
            # Only decrease if we're not already processing at good rate
            self.process_every_n_frames = max(3, self.process_every_n_frames - 1)
            
        # Analyze detection rates from YOLO and HSV
        yolo_active = False
        hsv_active = False
        yolo_recent = False
        hsv_recent = False
        
        if 'YOLO' in self.detection_history:
            time_since_yolo = current_time - self.detection_history['YOLO'].get('last_time', 0)
            yolo_active = time_since_yolo < 1.0  # Active in the last second
            yolo_recent = time_since_yolo < 0.3  # Very recent detection
            
        if 'HSV' in self.detection_history:
            time_since_hsv = current_time - self.detection_history['HSV'].get('last_time', 0)
            hsv_active = time_since_hsv < 1.0  # Active in the last second
            hsv_recent = time_since_hsv < 0.3  # Very recent detection
            
        # If both detectors are inactive, conserve resources
        if not yolo_active and not hsv_active and self.process_every_n_frames < 10:
            self.process_every_n_frames += 1
            
        # If at least one detector is very active, ensure we're keeping up
        if (yolo_recent or hsv_recent) and self.process_every_n_frames > 5:
            self.process_every_n_frames -= 1
            
        # Distance-based processing adjustments
        position_available = False
        estimated_distance = None
        
        # Check if we have a recent position from YOLO (preferred) or HSV
        for source in ['YOLO', 'HSV']:
            if source in self.detection_history:
                time_since_detection = current_time - self.detection_history[source].get('last_time', 0)
                if time_since_detection < 0.5 and self.detection_history[source].get('latest_position'):
                    position = self.detection_history[source]['latest_position']
                    # Calculate planar distance (ignoring height)
                    estimated_distance = np.sqrt(position[0]**2 + position[2]**2)
                    position_available = True
                    break
        
        # Adjust based on distance if available
        if position_available and estimated_distance is not None:
            if estimated_distance < 1.0:  # Very close - prioritize depth camera
                # Process more frames for better close range tracking (at least every 3rd)
                self.process_every_n_frames = max(1, min(self.process_every_n_frames, 3))
            elif estimated_distance > 4.0:  # Far - reduce depth camera processing
                # Skip more frames to save resources (at most every 12th)
                self.process_every_n_frames = min(12, max(self.process_every_n_frames, 6))
        
        # Constrain frame skip values to reasonable bounds
        self.process_every_n_frames = max(1, min(15, self.process_every_n_frames))
        
        # Only log when changes occur
        if old_frame_skip != self.process_every_n_frames:
            self.get_logger().info(
                f"Adjusted processing: 1 in {self.process_every_n_frames} frames "
                f"(CPU: {cpu_usage:.1f}%, FPS: {current_fps:.1f})"
            )
            
        # Force a log of key metrics periodically
        if not hasattr(self, 'last_debug_log') or current_time - self.last_debug_log > 30.0:
            self.last_debug_log = current_time
            
            # Format distance properly handling None
            dist_str = f"{estimated_distance:.2f}m" if estimated_distance is not None else "None"
            
            # Only log in debug mode
            if self.debug_mode:
                self.get_logger().info(
                    f"Status: position_avail={position_available}, "
                    f"dist={dist_str}, "
                    f"YOLO_active={yolo_active}, "
                    f"HSV_active={hsv_active}, "
                    f"FPS={current_fps:.1f}"
                )
    
    def publish_system_diagnostics(self):
        """Enhanced diagnostics with FPS metrics and velocity information."""
        current_time = TimeUtils.now_as_float()
        
        # Only publish every LOG_INTERVAL seconds
        if current_time - self.last_diag_log_time < LOG_INTERVAL:
            return
            
        self.last_diag_log_time = current_time
        
        # Calculate FPS more reliably
        elapsed = current_time - self.start_time
        fps = self.successful_conversions / elapsed if elapsed > 0 else 0
        
        # Calculate average FPS from recent history
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else fps
        
        # Calculate frame processing rate as percentage
        frame_rate_percentage = 100.0 / self.process_every_n_frames
        
        # Calculate cache hit rate
        total_attempts = sum(self.attempt_counter.values())
        total_hits = sum(self.cache_hits.values())
        cache_hit_rate = (total_hits / total_attempts * 100) if total_attempts > 0 else 0
        
        # Get the latest position and velocity if available (YOLO preferred)
        latest_pos = None
        for source in ['YOLO', 'HSV']:
            if source in self.detection_history and self.detection_history[source].get('latest_position'):
                latest_pos = self.detection_history[source]['latest_position']
                if source == 'YOLO':  # Prefer YOLO position if available
                    break
        
        # Format position string if available
        pos_str = ""
        if latest_pos:
            pos_str = f" | Pos: ({latest_pos[0]:.2f}, {latest_pos[1]:.2f}, {latest_pos[2]:.2f})m"
            
            # Add velocity data if available
            if hasattr(self, 'velocity') and self.velocity[0] != 0:
                vel_magnitude = (self.velocity[0]**2 + self.velocity[1]**2 + self.velocity[2]**2)**0.5
                pos_str += f" | Vel: {vel_magnitude:.2f}m/s"
        
        # Log comprehensive status
        self.get_logger().info(
            f"Depth camera: {fps:.1f} FPS (avg: {avg_fps:.1f}), CPU: {self.current_cpu_usage:.1f}%, "
            f"Frames: 1:{self.process_every_n_frames} ({frame_rate_percentage:.1f}%), "
            f"Cache hits: {cache_hit_rate:.1f}%{pos_str}"
        )
        
        # Publish detailed diagnostics message
        diag_msg = String()
        diag_data = {
            "fps": fps,
            "avg_fps": avg_fps,
            "cpu": self.current_cpu_usage,
            "frame_skip": self.process_every_n_frames,
            "frame_rate_pct": frame_rate_percentage,
            "cache_hit_rate": cache_hit_rate,
            "positions": {
                "YOLO": self.detection_history['YOLO'].get('latest_position'),
                "HSV": self.detection_history['HSV'].get('latest_position')
            },
            "velocity": self.velocity if hasattr(self, 'velocity') else [0, 0, 0],
            "timestamp": current_time
        }
        diag_msg.data = str(diag_data)
        self.system_diagnostics_publisher.publish(diag_msg)
    
    def _handle_resource_alert(self, resource_type, value):
        """Simple handler for resource alerts that updates CPU usage."""
        if resource_type == 'cpu':
            self.current_cpu_usage = value
            
            # Only log critical alerts once per minute
            if value > 90.0:
                current_time = TimeUtils.now_as_float()
                if current_time - self.last_resource_alert_time > 60.0:
                    self.last_resource_alert_time = current_time
                    self.get_logger().warning(f"Critical CPU usage: {value:.1f}%")
    
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
    node = TennisBall3DPositionEstimator()
    
    # Use multiple threads
    thread_count = DIAG_CONFIG.get("threads", 3)
    executor = MultiThreadedExecutor(num_threads=thread_count)
    executor.add_node(node)
    
    print("=================================================")
    print("Optimized 3D Position Estimator for Basketball")
    print("=================================================")
    print(f"Using {thread_count} threads on Raspberry Pi 5")
    
    # Log startup message to ROS logger
    node.get_logger().info(f"Starting with {thread_count} threads and process_every_n_frames={node.process_every_n_frames}")
    
    try:
        node.get_logger().info("3D Position Estimator running.")
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