#!/usr/bin/env python3

"""
Tennis Ball Tracking Robot - Depth Camera Node
==============================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving tennis ball.
The system uses multiple sensing modalities for robust detection:
- YOLO neural network detection (subscribes to '/tennis_ball/yolo/position')
- HSV color-based detection (subscribes to '/tennis_ball/hsv/position') 
- LIDAR for depth sensing
- Depth camera for additional depth information (this node)

This Node's Purpose:
------------------
This node converts 2D tennis ball detections from camera-based nodes (YOLO and HSV)
into 3D positions by using depth camera data. Understanding the 3D position is essential
for accurately calculating the distance to the ball, which is needed for proper following behavior.

How 2D to 3D Conversion Works:
----------------------------
1. We receive 2D locations (x,y) of the tennis ball from YOLO and HSV detectors
2. We scale these coordinates to match the depth camera's resolution
3. We lookup the depth value at that point from the depth camera
4. We convert the 2D+depth information into a 3D position using the camera's intrinsic parameters:
   - X = (pixel_x - cx) * depth / fx
   - Y = (pixel_y - cy) * depth / fy
   - Z = depth
   Where fx, fy, cx, cy are the camera's focal lengths and optical centers

Data Pipeline:
-------------
1. Camera images are processed by:
   - YOLO detection node publishing to '/tennis_ball/yolo/position'
   - HSV color detector publishing to '/tennis_ball/hsv/position'

2. This depth camera node:
   - Subscribes to 2D positions from YOLO and HSV
   - Subscribes to depth images and camera calibration
   - Publishes 3D positions to '/tennis_ball/yolo/position_3d' and '/tennis_ball/hsv/position_3d'

3. These 3D positions are then used by:
   - Sensor fusion node to combine all detection methods
   - State management node for decision-making
   - PID controller for motor control
"""
# Standard library imports
import sys
import os
import json
import time
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
from rclpy.time import Time
from rclpy.duration import Duration

# TF2 imports
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
import tf2_geometry_msgs
from tf2_geometry_msgs import PointStamped as TF2PointStamped

# ROS2 message types
from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

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
    "radius": 3,              # Radius around detection point to sample depth values
    "min_valid_points": 5,    # Minimum number of valid points required for reliable estimation
    "adaptive_radius": True,  # Whether to try larger radius if not enough valid points
    "max_radius": 7,          # Maximum radius to try when using adaptive sampling
    "detection_resolution": {  # Resolution of detection images (YOLO/HSV)
        "width": 320,
        "height": 320
    }
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
    "log_interval": 5.0,      # How often to log performance stats (seconds)
    "debug_level": 1,         # 0=minimal, 1=normal, 2=verbose
    "threads": 6,             # Number of threads for parallel processing
    "error_history_size": 10  # Keep track of last 10 errors
})

# Add optimized thread count for Raspberry Pi 5
# The Pi 5 has 4 cores, so we adjust threads accordingly
if os.environ.get('RASPBERRY_PI') == '1':
    DIAG_CONFIG['threads'] = 3  # Leave one core for system processes


# Define the common reference frame for the robot
COMMON_REFERENCE_FRAME = config.get('frames', {
    "reference_frame": "base_link",  # Common reference frame for all sensors
    "transform_timeout": 0.1          # Timeout for transform lookups in seconds
})

class TennisBall3DPositionEstimator(Node):
    """
    A ROS2 node that converts 2D tennis ball detections to 3D positions.
    
    This node takes the 2D position of a tennis ball (from YOLO or HSV detectors)
    and uses depth camera data to estimate its 3D position in space. This is
    essential for the robot to understand how far away the ball is and approach
    it correctly.
    
    Subscribed Topics:
    - Camera info ({TOPICS["input"]["camera_info"]})
    - Depth image ({TOPICS["input"]["depth_image"]})
    - YOLO 2D detections ({TOPICS["input"]["yolo_detection"]})
    - HSV 2D detections ({TOPICS["input"]["hsv_detection"]})
    
    Published Topics:
    - YOLO 3D positions ({TOPICS["output"]["yolo_3d"]})
    - HSV 3D positions ({TOPICS["output"]["hsv_3d"]})
    - Combined 3D positions ({TOPICS["output"]["combined"]})
    """
    
    def __init__(self):
        """Initialize the 3D position estimator node with all required components."""
        super().__init__('tennis_ball_3d_position_estimator')
        
        # Initialize performance tracking BEFORE setting up resource monitor
        self._init_performance_tracking()
        
        # Add Raspberry Pi resource monitoring
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=20.0,
            enable_temperature=False
        )
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.resource_monitor.start()
        
        # Set up callback group for efficient multi-threading
        self._setup_callback_group()
        
        # Initialize camera and detection parameters
        self._init_camera_parameters()
        
        # Set up tf2 for coordinate transformations
        self._setup_tf2()
        
        # Set up subscriptions to receive data
        self._setup_subscriptions()
        
        # Set up publishers to send out 3D positions
        self._setup_publishers()
        
        # Add downsampling flag for Raspberry Pi
        self.enable_downsampling = True
        # More aggressive frame skipping at startup for better performance
        self.process_every_n_frames = 10  # Start with 1 in 10 frames, can adjust later
        self.frame_counter = 0
        self.depth_frame_counter = 0
        
        # Initialize CPU usage variable
        self.current_cpu_usage = 0.0
        
        # Start the adaptive frame skipping system
        self.frame_skipping_timer = self.create_timer(5.0, self._adjust_frame_skipping)

        # Initialize transform check flag
        self.verified_transform = False

        # Debug visualization mode
        self.debug_mode = DIAG_CONFIG.get('debug_level', 1) > 1
        
        self.get_logger().info("Tennis Ball 3D Position Estimator initialized")
        self.get_logger().info(f"Using coordinate scaling: detection ({DEPTH_CONFIG['detection_resolution']['width']}x"
                              f"{DEPTH_CONFIG['detection_resolution']['height']}) -> depth (will be updated when received)")
        self.get_logger().info(f"Processing every {self.process_every_n_frames} frames for performance optimization")
        
        # Pre-allocate these arrays for faster processing
        self.downsampled_array = None
        self.downsampled_width = 0
        self.downsampled_height = 0
        
        # Add memory optimization timer (every 30 seconds)
        self.memory_optimization_timer = self.create_timer(30.0, self._optimize_memory_use)
        
        # Add processing flags
        self.enable_motion_filtering = True  # Only process when ball moves
        self.enable_transform_caching = True  # Cache transforms
        self.transform_cache_lifetime = 0.2   # Short lifetime for accuracy
        
        # Add cached constants for faster math
        self._scale_factor = DEPTH_CONFIG["scale"]
        self._min_valid_depth = DEPTH_CONFIG["min_depth"]
        self._max_valid_depth = DEPTH_CONFIG["max_depth"]
    
    def _setup_callback_group(self):
        """Set up callback group and QoS profile for subscriptions."""
        # Single callback group for all subscriptions for maximum concurrency
        self.callback_group = ReentrantCallbackGroup()
        
        # Increase QoS history to avoid dropping messages
        self.qos_profile = rclpy.qos.QoSProfile(
            depth=10,
            reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT,  
            durability=rclpy.qos.DurabilityPolicy.VOLATILE
        )
    
    def _adjust_frame_skipping(self):
        """Periodically adjust frame skipping based on CPU load and processing performance"""
        # Get current CPU usage
        try:
            cpu_usage = psutil.cpu_percent(interval=None)
            self.current_cpu_usage = cpu_usage  # Store for other methods
            
            # Adjust frame skipping based on CPU load
            if cpu_usage < 70 and self.process_every_n_frames > 5:
                self.process_every_n_frames -= 1
                self.get_logger().info(f"CPU load allows reduced frame skipping: now 1 in {self.process_every_n_frames}")
            elif cpu_usage > 90 and self.process_every_n_frames < 15:
                self.process_every_n_frames += 1
                self.get_logger().info(f"High CPU load, increasing frame skipping: now 1 in {self.process_every_n_frames}")
        except Exception as e:
            pass
    
    def _optimize_memory_use(self):
        """Periodically optimize memory usage."""
        # Clear unnecessary caches when memory pressure is high
        if hasattr(self, 'resource_monitor') and hasattr(self.resource_monitor, 'last_memory_percent'):
            if self.resource_monitor.last_memory_percent > 85:
                # Clear transform cache
                if hasattr(self, 'recent_transforms'):
                    self.recent_transforms.clear()
                    
                # Clear error history caches if they're large
                if hasattr(self, 'error_counts') and len(self.error_counts) > 20:
                    self.error_counts.clear()
                    self.error_last_logged.clear()
                    
                # Force garbage collection
                import gc
                gc.collect()
                
                self.get_logger().info("Memory optimization performed due to high usage")

    def _init_camera_parameters(self):
        """Initialize camera and detection parameters."""
        # Camera parameters (will be updated from camera_info)
        self.camera_info = None
        self.depth_image = None
        self.depth_header = None
        
        # Camera intrinsics (from camera calibration)
        self.fx = 0.0  # Focal length x
        self.fy = 0.0  # Focal length y
        self.cx = 0.0  # Optical center x
        self.cy = 0.0  # Optical center y
        
        # Latest detections
        self.latest_yolo_detection = None
        self.latest_hsv_detection = None
        
        # Pre-create bridge for faster conversion
        self.cv_bridge = CvBridge()
        
        # Depth image resolution (will be updated from camera_info)
        self.depth_width = 640   # Default/initial value
        self.depth_height = 480  # Default/initial value
        
        # Coordinate scaling factors (updated when camera_info is received)
        self.x_scale = 1.0  # Will be properly calculated when camera_info is received
        self.y_scale = 1.0  # Will be properly calculated when camera_info is received
        
        # Configuration for depth sampling
        self.radius = DEPTH_CONFIG["radius"]
        
        # Pre-compute radius search offsets for faster lookup
        self.offsets = []
        for y in range(-self.radius, self.radius+1):
            for x in range(-self.radius, self.radius+1):
                self.offsets.append((x, y))
    
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
        # Add transform cache
        self.transform_cache = {}
        self.transform_cache_lifetime = 1.0  # Cache transforms for 1 second
        
        self.get_logger().info(f"Transform listener initialized - using '{self.reference_frame}' as reference frame")

    def _publish_static_transform(self):
        """Publish static transform between camera_frame and base_link."""
        # Create the transform
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.reference_frame
        transform.child_frame_id = "camera_frame"
        
        # Set the camera position relative to base_link
        # Adjust these values based on your actual camera position
        transform.transform.translation.x = 0.2  # 20cm forward
        transform.transform.translation.y = 0.0  # centered
        transform.transform.translation.z = 0.1  # 10cm up
        
        # Set rotation - identity quaternion if camera points same direction as base
        transform.transform.rotation.w = 1.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        
        # Publish the transform
        self.static_broadcaster.sendTransform(transform)
        self.get_logger().info("Published static transform: camera_frame -> base_link")
    
    def _verify_transform(self):
        """Verify that the transform is correctly registered."""
        try:
            # Check if transform is available
            if self.tf_buffer.can_transform(
                self.reference_frame,
                "camera_frame",
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            ):
                self.verified_transform = True
                self.get_logger().info("Transform verification successful. Transform chain is complete.")
                self.transform_check_timer.cancel()
            else:
                # If not available, republish transform
                self.get_logger().warning("Transform not yet available. Republishing...")
                self._publish_static_transform()
        except Exception as e:
            # If there's an error, republish transform
            self.get_logger().error(f"Transform verification failed: {str(e)}. Republishing...")
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
        
        # Subscribe to depth image
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
        
        # Update diagnostics publisher to match the format expected by system diagnostics node
        self.system_diagnostics_publisher = self.create_publisher(
            String,
            "/tennis_ball/depth_camera/diagnostics",  # Updated topic to match diagnostics node expectations
            10
        )
        
        # Add timer for publishing diagnostics
        self.diagnostics_timer = self.create_timer(5.0, self.publish_system_diagnostics)
    
    def _init_performance_tracking(self):
        """Initialize performance tracking variables."""
        self.start_time = TimeUtils.now_as_float()
        self.yolo_count = 0
        self.hsv_count = 0
        self.successful_conversions = 0
        self.last_fps_log_time = TimeUtils.now_as_float()
        
        # Use deque with maxlen instead of lists for bounded buffer sizes
        max_history = DIAG_CONFIG.get('history_length', 100)
        self.processing_times = deque(maxlen=max_history)
        
        # Define error history size FIRST
        self.error_history_size = DIAG_CONFIG.get('error_history_size', 10)
        
        # THEN create the deques with the size
        self.errors = deque(maxlen=self.error_history_size)
        self.warnings = deque(maxlen=self.error_history_size)
        self.last_error_time = 0
        
        # Error tracking by type to detect repeated errors
        self.error_counts = {}
        self.error_last_logged = {}
        
        # Add health metrics
        self.depth_camera_health = 1.0  # 0.0 to 1.0 scale
        self.processing_health = 1.0
        self.detection_health = 1.0
    
    def log_error(self, error_message, is_warning=False):
        """Log an error and add it to error history for diagnostics."""
        current_time = TimeUtils.now_as_float()
        
        # Track error frequency by type
        if error_message not in self.error_counts:
            self.error_counts[error_message] = 1
            self.error_last_logged[error_message] = 0  # Never logged before
        else:
            self.error_counts[error_message] += 1
        
        # Determine if we should log this error (always log first occurrence and then rate-limit)
        should_log = False
        time_since_last_log = current_time - self.error_last_logged.get(error_message, 0)
        
        # Always log the first occurrence or if it's been a while since we logged this error
        if self.error_counts[error_message] == 1 or time_since_last_log > 10.0:
            should_log = True
            self.error_last_logged[error_message] = current_time
            
            # For repeated errors, include the count
            if self.error_counts[error_message] > 1:
                error_message = f"{error_message} (occurred {self.error_counts[error_message]} times)"
        
        if should_log:
            if is_warning:
                self.get_logger().warning(f"DEPTH: {error_message}")
                # Add to warning list for diagnostics
                self.warnings.append({
                    "timestamp": current_time,
                    "message": error_message
                })
            else:
                self.get_logger().error(f"DEPTH: {error_message}")
                # Add to error list for diagnostics
                self.errors.append({
                    "timestamp": current_time,
                    "message": error_message
                })
                
                # Update health based on error frequency
                self.last_error_time = current_time
                
                # Reduce health score temporarily after an error
                self.depth_camera_health = max(0.3, self.depth_camera_health - 0.2)
    
    def camera_info_callback(self, msg):
        """
        Process camera calibration information.
        
        This callback stores the camera's intrinsic parameters which are essential
        for converting from pixel coordinates to 3D world coordinates.
        
        Args:
            msg (CameraInfo): Camera calibration information
        """
        self.camera_info = msg
        
        # Cache intrinsics for faster access
        # The camera matrix K contains:
        # [fx  0  cx]
        # [ 0 fy  cy]
        # [ 0  0   1]
        self.fx = msg.k[0]  # Focal length x
        self.fy = msg.k[4]  # Focal length y
        self.cx = msg.k[2]  # Principal point x (optical center)
        self.cy = msg.k[5]  # Principal point y (optical center)
        
        # Update image dimensions and scaling factors
        self.depth_width = msg.width
        self.depth_height = msg.height
        
        # Update scaling factors with proper aspect ratio handling
        detect_width = DEPTH_CONFIG["detection_resolution"]["width"]
        detect_height = DEPTH_CONFIG["detection_resolution"]["height"]
        
        # Maintain aspect ratio in scaling
        if (self.depth_width / self.depth_height) > (detect_width / detect_height):
            # Width-constrained
            self.x_scale = self.depth_width / detect_width
            self.y_scale = self.x_scale  # Use same scale factor
        else:
            # Height-constrained
            self.y_scale = self.depth_height / detect_height
            self.x_scale = self.y_scale  # Use same scale factor
        
        # Log camera info once (first time received)
        if not hasattr(self, 'camera_info_logged'):
            self.get_logger().info(f"Camera info received: {self.depth_width}x{self.depth_height}, "
                                  f"fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
            self.get_logger().info(
                f"Updated coordinate scaling: detection ({detect_width}x{detect_height}) -> "
                f"depth ({self.depth_width}x{self.depth_height}), scales: x={self.x_scale:.2f}, y={self.y_scale:.2f}"
            )
            self.camera_info_logged = True
        
        # Update camera health if calibration is received
        if hasattr(self, 'depth_camera_health'):
            self.depth_camera_health = 1.0  # Camera is working well

    # Optimize depth image processing
    def depth_callback(self, msg):
        """Process depth image from camera."""
        # Log the first few depth images
        if not hasattr(self, '_depth_msg_count'):
            self._depth_msg_count = 0
        
        self._depth_msg_count += 1
        if self._depth_msg_count <= 3:
            # Log basic info about the image
            self.get_logger().info(f"Depth image received: encoding={msg.encoding}, "
                                  f"step={msg.step}, shape={msg.width}x{msg.height}")
        
        try:
            # Convert the depth image to a numpy array
            depth_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # Store the depth array and header
            self.depth_array = depth_img
            self.depth_header = msg.header
            
            # Log depth statistics occasionally
            if not hasattr(self, 'depth_image_counter'):
                self.depth_image_counter = 0
            self.depth_image_counter += 1
            
            if self.depth_image_counter % 50 == 0:
                non_zero = np.count_nonzero(depth_img)
                total = depth_img.size
                percent = (non_zero / total) * 100 if total > 0 else 0
                min_val = np.min(depth_img[depth_img > 0]) if non_zero > 0 else 0
                max_val = np.max(depth_img)
                mean_val = np.mean(depth_img[depth_img > 0]) if non_zero > 0 else 0
                self.get_logger().info(
                    f"Depth image stats: shape={depth_img.shape}, "
                    f"non-zero={non_zero}/{total} ({percent:.1f}%), "
                    f"range={min_val}-{max_val}, mean={mean_val:.2f}"
                )
        except Exception as e:
            self.log_error(f"Error processing depth image: {str(e)}")
    
    def yolo_callback(self, msg):
        # Skip processing based on CPU load
        if hasattr(self, 'process_every_n_frames') and hasattr(self, 'frame_counter'):
            self.frame_counter += 1
            if self.frame_counter % self.process_every_n_frames != 0:
                return  # Skip this frame
        
        # Also skip if the ball hasn't moved enough
        if not self._should_process_detection(msg, "YOLO"):
            return
        
        start_time = TimeUtils.now_as_float()
        self.latest_yolo_detection = msg
        
        # Process immediately for lowest latency
        if self.get_3d_position(msg, "YOLO"):
            self.yolo_count += 1
            process_time = (TimeUtils.now_as_float() - start_time) * 1000  # in milliseconds
            self._update_processing_stats(process_time)
    
    def hsv_callback(self, msg):
        """Handle tennis ball detections from HSV."""
        # Skip processing based on CPU load
        if hasattr(self, 'process_every_n_frames') and hasattr(self, 'hsv_frame_counter'):
            self.hsv_frame_counter += 1
            if self.hsv_frame_counter % self.process_every_n_frames != 0:
                return  # Skip this frame
        else:
            self.hsv_frame_counter = 0
        
        # Also skip if the ball hasn't moved enough
        if not self._should_process_detection(msg, "HSV"):
            return
        
        # Record start time for performance tracking
        start_time = TimeUtils.now_as_float()
        self.latest_hsv_detection = msg
        
        # Process immediately for lowest latency
        if self.get_3d_position(msg, "HSV"):
            self.hsv_count += 1
            process_time = (TimeUtils.now_as_float() - start_time) * 1000  # in milliseconds
            self._update_processing_stats(process_time)
    
    def _update_processing_stats(self, process_time):
        """Update processing statistics for performance tracking."""
        self.processing_times.append(process_time)
        # No need to check size and pop - deque handles this automatically
    
    def get_3d_position(self, detection_msg, source):
        """
        Convert a 2D ball detection to a 3D position using depth data.
        
        Args:
            detection_msg (PointStamped): 2D position from detection node
            source (str): Detection source ("YOLO" or "HSV")
            
        Returns:
            bool: True if 3D position was successfully calculated and published
        """
        # Initialize failure counters if not existing
        if not hasattr(self, 'failure_reasons'):
            self.failure_reasons = {
                'missing_data': 0,
                'transform_not_verified': 0,
                'coordinates_out_of_bounds': 0,
                'no_reliable_depth': 0,
                'transform_failed': 0,
                'exceptions': 0,
                'hsv_priority_skip': 0,
                'successful': 0
            }
            
        # Add counter for tracking attempts
        if not hasattr(self, 'attempt_counter'):
            self.attempt_counter = {'YOLO': 0, 'HSV': 0}
        self.attempt_counter[source] += 1
        
        # Skip processing if we're missing required data
        if self.camera_info is None or not hasattr(self, 'depth_array') or self.fx == 0:
            # Every 50 attempts, log status
            if self.attempt_counter[source] % 50 == 0:
                self.get_logger().warn(f"Missing data: camera_info={self.camera_info is not None}, depth_array={hasattr(self, 'depth_array')}, fx={self.fx}")
            
            self.failure_reasons['missing_data'] += 1
            return False
        
        # Skip if transform is not verified yet
        if not self.verified_transform:
            if not hasattr(self, '_transform_not_verified_logged'):
                self.log_error("Transform not yet verified - waiting for transform chain to be complete", True)
                self._transform_not_verified_logged = True
            
            self.failure_reasons['transform_not_verified'] += 1
            return False
        
        # Prioritize HSV over YOLO when system is under load
        if source == "YOLO" and self.current_cpu_usage > 85 and hasattr(self, 'detection_history'):
            if 'HSV' in self.detection_history:
                current_time = TimeUtils.now_as_float()
                time_since_last_hsv = current_time - self.detection_history['HSV'].get('last_time', 0)
                if time_since_last_hsv < 0.2:  # If HSV was processed recently
                    self.failure_reasons['hsv_priority_skip'] += 1
                    return False  # Skip YOLO processing to prioritize HSV
                
        try:
            # Get 2D coordinates from detection
            orig_x = detection_msg.point.x
            orig_y = detection_msg.point.y
            
            # Print out the detection coords occasionally
            if self.attempt_counter[source] % 100 == 0:
                self.get_logger().info(f"Processing {source} detection at ({orig_x:.2f}, {orig_y:.2f})")
            
            # Step 1: Scale coordinates to depth image space
            # (YOLO/HSV work in 320x320, depth might be 640x480)
            pixel_x = int(round(orig_x * self.x_scale))
            pixel_y = int(round(orig_y * self.y_scale))
            
            # Step 2: Check if coordinates are within valid bounds
            # Ensure enough margin for adaptive radius sampling
            max_possible_radius = DEPTH_CONFIG.get("max_radius", 7)
            margin = max_possible_radius + 5  # Extra margin for safety
            
            # Check if depth array exists and has correct shape
            if not hasattr(self, 'depth_array') or self.depth_array is None or self.depth_array.size == 0:
                self.failure_reasons['missing_data'] += 1
                return False
                
            depth_height, depth_width = self.depth_array.shape
                
            if (pixel_x < margin or pixel_x >= depth_width - margin or 
                pixel_y < margin or pixel_y >= depth_height - margin):
                
                if self.attempt_counter[source] % 50 == 0:
                    self.get_logger().warning(
                        f"Detection coordinates out of bounds: ({pixel_x},{pixel_y}) "
                        f"max=({depth_width},{depth_height})"
                    )
                
                self.failure_reasons['coordinates_out_of_bounds'] += 1
                return False
            
            # Step 3: Get depth array
            depth_array = self.depth_array
            
            # Step 4: Use optimized depth sampling method
            median_depth, depth_reliability, valid_points = self._get_reliable_depth(
                depth_array, pixel_x, pixel_y)
            
            # If no reliable depth was found, return False
            if median_depth is None:
                if self.attempt_counter[source] % 50 == 0:
                    self.get_logger().warning(
                        f"No reliable depth found for {source} at ({pixel_x},{pixel_y})"
                    )
                
                self.failure_reasons['no_reliable_depth'] += 1
                return False
            
            # Step 5: Use the pinhole camera model to convert to 3D
            # These equations convert from pixel coordinates to 3D coordinates:
            # X = (u - cx) * Z / fx
            # Y = (v - cy) * Z / fy
            # Z = depth
            x = (pixel_x - self.cx) * median_depth / self.fx
            y = (pixel_y - self.cy) * median_depth / self.fy
            z = median_depth
            
            # Step 6: Create the 3D position message in camera frame
            camera_position_msg = PointStamped()
            
            # IMPORTANT: Use the timestamp from original detection
            # Validate timestamp before using it
            if TimeUtils.is_timestamp_valid(detection_msg.header.stamp):
                camera_position_msg.header.stamp = detection_msg.header.stamp
            else:
                camera_position_msg.header.stamp = TimeUtils.now_as_ros_time()
            
            # Set the frame ID to the actual camera frame
            camera_position_msg.header.frame_id = "camera_frame"
            
            # Track sequence internally but don't try to set it on the header
            if not hasattr(self, 'seq_counter'):
                self.seq_counter = 0
            self.seq_counter += 1
            
            camera_position_msg.point.x = x
            camera_position_msg.point.y = y
            camera_position_msg.point.z = z
            
            # Step 7: Transform position to common reference frame before publishing
            transformed_msg = self._transform_to_reference_frame(camera_position_msg)
            if transformed_msg:
                # Publish to source-specific topic
                if source == "YOLO":
                    self.yolo_3d_publisher.publish(transformed_msg)
                else:  # HSV
                    self.hsv_3d_publisher.publish(transformed_msg)
                
                # Also publish to combined topic for backward compatibility
                self.position_publisher.publish(transformed_msg)
                
                self.successful_conversions += 1
                self.failure_reasons['successful'] += 1
                
                # Log performance periodically (use transformed coordinates)
                self._log_performance(
                    transformed_msg.point.x,
                    transformed_msg.point.y, 
                    transformed_msg.point.z,
                    source
                )
                
                # Update detection health on successful conversion
                if hasattr(self, 'detection_health'):
                    self.detection_health = min(1.0, self.detection_health + 0.05)
                
                # Add depth reliability to point metadata (store in a class variable)
                if not hasattr(self, 'depth_reliability'):
                    self.depth_reliability = {}
                self.depth_reliability[source] = {
                    'value': depth_reliability,
                    'valid_points': valid_points,
                    'timestamp': TimeUtils.now_as_float()
                }
                
                return True
            else:
                # If transformation failed, log the error but not too frequently
                if not hasattr(self, '_transform_error_logged'):
                    self.log_error(f"Failed to transform position from camera_frame to {self.reference_frame}", True)
                    self._transform_error_logged = True
                    # Reset after a while to allow future logging
                    self.create_timer(10.0, lambda: setattr(self, '_transform_error_logged', False))
                
                self.failure_reasons['transform_failed'] += 1
                return False
            
        except Exception as e:
            # Log all errors - no random filtering
            self.log_error(f"Error in 3D conversion: {str(e)}")
            
            # Reduce detection health on errors
            if hasattr(self, 'detection_health'):
                self.detection_health = max(0.3, self.detection_health - 0.1)
            
            self.failure_reasons['exceptions'] += 1
            return False
    
    def _transform_to_reference_frame(self, point_msg):
        """Optimized transform function with aggressive caching."""
        # Create a cache key based on the message
        cache_key = (point_msg.header.frame_id, 
                     round(point_msg.point.x, 3), 
                     round(point_msg.point.y, 3), 
                     round(point_msg.point.z, 3))
        
        # Check if we have a cached result
        if hasattr(self, 'recent_transforms'):
            if cache_key in self.recent_transforms:
                cached_result, timestamp = self.recent_transforms[cache_key]
                # Only use cache if it's recent (200ms)
                if TimeUtils.now_as_float() - timestamp < 0.2:
                    return cached_result
        else:
            self.recent_transforms = {}
        
        # Rest of the transform code...
        try:
            # Look up the transformation
            
            # Use the timestamp from the original message for the transform
            transform_time = point_msg.header.stamp
            
            # Ensure TF2 knows how to transform PointStamped
            # This is handled by the import at the top: tf2_geometry_msgs.PointStamped
            
            # If the transform isn't available yet, try with a small delay
            try:
                self.tf_buffer.can_transform(
                    self.reference_frame,
                    point_msg.header.frame_id,
                    transform_time,
                    rclpy.duration.Duration(seconds=self.transform_timeout)
                )
            except Exception:
                # If we failed with the message timestamp, try with current time
                self.get_logger().debug(
                    f"Transform not available at timestamp {transform_time}, trying with current time"
                )
                transform_time = TimeUtils.now_as_ros_time()
            
            # Do the actual transformation
            transformed_point = self.tf_buffer.transform(
                point_msg,
                self.reference_frame,
                rclpy.duration.Duration(seconds=self.transform_timeout)
            )
            
            # Log the transformation occasionally, but track it more systematically
            if not hasattr(self, 'transform_log_counter'):
                self.transform_log_counter = 0
            
            self.transform_log_counter += 1
            if self.transform_log_counter % 100 == 0:  # Log every 100th transform
                self.get_logger().debug(
                    f"Transformed point from {point_msg.header.frame_id} to {self.reference_frame}: "
                    f"({point_msg.point.x:.2f}, {point_msg.point.y:.2f}, {point_msg.point.z:.2f}) -> "
                    f"({transformed_point.point.x:.2f}, {transformed_point.point.y:.2f}, {transformed_point.point.z:.2f})"
                )
            
            # Cache the result before returning
            if transformed_point:
                self.recent_transforms[cache_key] = (transformed_point, TimeUtils.now_as_float())
                
                # Limit cache size
                if len(self.recent_transforms) > 30:
                    # Remove oldest entries
                    oldest_key = min(self.recent_transforms, key=lambda k: self.recent_transforms[k][1])
                    del self.recent_transforms[oldest_key]
            
            return transformed_point
            
        except Exception as e:
            # Log all transform errors properly
            if not hasattr(self, '_transform_exception_logged'):
                self.log_error(f"Transform exception: {str(e)}", True)
                self._transform_exception_logged = True
                # Reset after a while to allow future logging
                self.create_timer(10.0, lambda: setattr(self, '_transform_exception_logged', False))
            return None
    
    def _log_performance(self, x, y, z, source):
        """
        Log performance metrics and position data.
        
        Args:
            x, y, z (float): 3D position coordinates
            source (str): Detection source ("YOLO" or "HSV")
        """
        # Calculate FPS and other metrics periodically
        current_time = TimeUtils.now_as_float()  # Use TimeUtils
        time_since_last_log = current_time - self.last_fps_log_time
        
        # Log FPS every N seconds (from config)
        if time_since_last_log >= DIAG_CONFIG.get('log_interval', 5.0):
            elapsed = current_time - self.start_time
            conversion_rate = self.successful_conversions / elapsed if elapsed > 0 else 0
            yolo_rate = self.yolo_count / elapsed if elapsed > 0 else 0
            hsv_rate = self.hsv_count / elapsed if elapsed > 0 else 0
            
            self.get_logger().info(
                f"3D position ({source}): ({x:.2f}, {y:.2f}, {z:.2f}) meters | "
                f"FPS: {conversion_rate:.1f} | YOLO: {yolo_rate:.1f} | HSV: {hsv_rate:.1f}"
            )
            self.last_fps_log_time = current_time
        
        # Store this detection data for diagnostics
        if not hasattr(self, 'detection_history'):
            self.detection_history = {
                'YOLO': {'count': 0, 'latest_position': None, 'last_time': 0},
                'HSV': {'count': 0, 'latest_position': None, 'last_time': 0}
            }
        
        # Update detection history
        self.detection_history[source]['count'] += 1
        self.detection_history[source]['latest_position'] = (x, y, z)
        self.detection_history[source]['last_time'] = TimeUtils.now_as_float()  # Use TimeUtils
    
    def publish_system_diagnostics(self):
        """Reduced overhead diagnostics that adapt to CPU load."""
        # Skip heavy diagnostics when CPU is high
        if hasattr(self, 'current_cpu_usage') and self.current_cpu_usage > 95:
            # Very lightweight diagnostics only
            self.get_logger().info(
                f"Depth camera: {self.successful_conversions / (TimeUtils.now_as_float() - self.start_time):.1f} FPS, "
                f"CPU: {self.current_cpu_usage:.1f}%, Status: active"
            )
            return
            
        # Regular diagnostics code follows...
        current_time = TimeUtils.now_as_float()  # Use TimeUtils
        elapsed = current_time - self.start_time
        
        # Performance metrics
        total_fps = self.successful_conversions / elapsed if elapsed > 0 else 0
        yolo_fps = self.yolo_count / elapsed if elapsed > 0 else 0
        hsv_fps = self.hsv_count / elapsed if elapsed > 0 else 0
        avg_processing = np.mean(self.processing_times) if self.processing_times else 0
        
        # Check camera info and depth availability
        has_camera_info = self.camera_info is not None
        has_depth_image = self.depth_image is not None
        
        # Calculate time since last detection from each source
        detection_status = {}
        
        if hasattr(self, 'detection_history'):
            for source, data in self.detection_history.items():
                time_since_last = current_time - data['last_time'] if data['last_time'] > 0 else float('inf')
                detection_status[source] = {
                    'count': data['count'],
                    'time_since_last_s': time_since_last,
                    'latest_position': data['latest_position'],
                    'active': time_since_last < 2.0  # Consider active if seen in last 2 seconds
                }
        
        # Collect errors and warnings
        error_messages = []
        warning_messages = []
        
        # Check for missing camera info
        if not has_camera_info:
            warning_messages.append("Missing camera intrinsics - waiting for camera_info")
        
        # Check for missing depth images
        if not has_depth_image:
            warning_messages.append("No depth image received yet")
        
        # If we've run for a while but have no successful conversions, that's concerning
        if elapsed > 10.0 and self.successful_conversions == 0:
            error_messages.append("No successful 3D conversions despite running for >10s")
        
        # Check for processing time issues
        if avg_processing > 50.0:  # If processing takes >50ms, it's slow
            warning_messages.append(f"High processing time: {avg_processing:.1f}ms")
        
        # Add tracked errors and warnings
        for error in self.errors:
            if current_time - error["timestamp"] < 300:  # Only include recent errors (last 5 minutes)
                error_messages.append(error["message"])
        
        # System resources
        system_metrics = {}
        try:
            system_metrics = {
                'cpu_percent': psutil.cpu_percent(interval=None),
                'ram_percent': psutil.virtual_memory().percent
            }
            
            # Add temperature info if on Raspberry Pi
            temps = {}
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps and 'cpu_thermal' in temps:
                    system_metrics['temperature'] = temps['cpu_thermal'][0].current
        except Exception as e:
            self.get_logger().debug(f"Could not get system metrics: {str(e)}")
            pass
        
        # Health recovery over time (errors become less relevant)
        time_since_last_error = current_time - self.last_error_time
        if time_since_last_error > 30.0:  # After 30 seconds with no errors
            self.depth_camera_health = min(1.0, self.depth_camera_health + 0.05)  # Gradually recover
        
        # Calculate overall health (weighted average)
        overall_health = (
            self.depth_camera_health * 0.4 + 
            self.processing_health * 0.3 + 
            self.detection_health * 0.3
        )
        
        # Determine status based on errors/warnings
        status = "active"
        if error_messages:
            status = "error"
        elif warning_messages:
            status = "warning"
        
        # Add depth sampling reliability metrics to diagnostics
        depth_reliability_metrics = {}
        if hasattr(self, 'depth_reliability'):
            current_time = TimeUtils.now_as_float()
            for source, data in self.depth_reliability.items():
                # Only include recent data (last 5 seconds)
                if current_time - data['timestamp'] < 5.0:
                    depth_reliability_metrics[source] = {
                        'reliability': data['value'],
                        'valid_points': data['valid_points'],
                        'age_seconds': current_time - data['timestamp']
                    }
        
        # Create diagnostics data structure formatted for the diagnostics node
        diag_data = {
            "timestamp": current_time,
            "node": "depth_camera",
            "uptime_seconds": elapsed,
            "status": status,
            "health": {
                "camera_health": self.depth_camera_health,
                "processing_health": self.processing_health,
                "detection_health": self.detection_health,
                "overall": overall_health
            },
            "metrics": {
                "successful_conversions": self.successful_conversions,
                "conversion_rate": total_fps,
                "yolo_conversions": self.yolo_count,
                "yolo_rate": yolo_fps,
                "hsv_conversions": self.hsv_count,
                "hsv_rate": hsv_fps,
                "avg_processing_time_ms": avg_processing
            },
            "camera_status": {
                "has_camera_info": has_camera_info,
                "has_depth_image": has_depth_image,
                "camera_resolution": f"{self.depth_width}x{self.depth_height}" if has_camera_info else "unknown"
            },
            "detections": detection_status,
            "resources": system_metrics,
            "errors": error_messages,
            "warnings": warning_messages,
            "config": {
                "detection_resolution": {
                    "width": DEPTH_CONFIG["detection_resolution"]["width"],
                    "height": DEPTH_CONFIG["detection_resolution"]["height"]
                },
                "depth_scale": DEPTH_CONFIG["scale"],
                "sampling_radius": self.radius
            },
            "depth_sampling": {
                "reliability": depth_reliability_metrics,
                "min_valid_points": DEPTH_CONFIG.get("min_valid_points", 5),
                "adaptive_radius": DEPTH_CONFIG.get("adaptive_radius", True),
                "current_radius": self.radius
            }
        }
        
        # Add transform diagnostics
        transform_available = False
        try:
            transform_available = self.tf_buffer.can_transform(
                self.reference_frame,
                "camera_frame",
                TimeUtils.now_as_ros_time(),
                rclpy.duration.Duration(seconds=0.1)
            )
        except Exception as e:
            pass
        
        # Add to diagnostics data
        diag_data["transform_status"] = {
            "available": transform_available,
            "reference_frame": self.reference_frame,
            "camera_frame": "camera_frame"
        }
        
        # Include frame skipping info
        diag_data["performance_adaptations"] = {
            "process_every_n_frames": getattr(self, "process_every_n_frames", 1),
            "current_radius": self.radius
        }
        
        # Publish as JSON
        msg = String()
        msg.data = json.dumps(diag_data)
        self.system_diagnostics_publisher.publish(msg)
        
        # Also log summary to console
        self.get_logger().info(
            f"Depth camera: {total_fps:.1f} FPS, "
            f"Processing: {avg_processing:.1f}ms, "
            f"Health: {overall_health:.2f}, "
            f"Status: {status}"
        )

        # Add failure reason statistics to diagnostics
        if hasattr(self, 'failure_reasons') and sum(self.failure_reasons.values()) > 0:
            total = sum(self.failure_reasons.values())
            self.get_logger().info("3D conversion statistics:")
            for reason, count in self.failure_reasons.items():
                percentage = (count / total) * 100 if total > 0 else 0
                self.get_logger().info(f"  - {reason}: {count} ({percentage:.1f}%)")
    
    def _handle_resource_alert(self, resource_type, value):
        """Handle resource alerts by adjusting processing parameters."""
        if hasattr(self, 'last_resource_alert_time'):
            current_time = TimeUtils.now_as_float()
            if current_time - self.last_resource_alert_time < 5.0:
                return  # Avoid too frequent adjustments
            
        self.last_resource_alert_time = TimeUtils.now_as_float()
        
        if resource_type == 'cpu' and value > 90.0:
            # Only log once per 10 seconds to reduce overhead
            self.log_error(f"High CPU usage ({value:.1f}%) - reducing processing load", True)
        
        if not hasattr(self, 'process_every_n_frames'):
            self.process_every_n_frames = 1
        
        if self.process_every_n_frames < 10:  # Max: process 1 in 10 frames
            self.process_every_n_frames += 1
            
        # Only reduce radius if it's not already at minimum
        if self.radius > 1:
            self.radius = max(1, self.radius - 1)

        # CPU recovery - gradually return to normal processing when CPU usage is lower
        elif resource_type == 'cpu' and value < 70.0 and hasattr(self, 'process_every_n_frames'):
            if self.process_every_n_frames > 1:
                self.process_every_n_frames -= 1
                self.get_logger().info(f"CPU usage normalized, processing 1 in {self.process_every_n_frames} frames")
    
    def destroy_node(self):
        """Clean shutdown of the node."""
        # Clear any large stored images
        if hasattr(self, 'depth_image'):
            self.depth_image = None
        if hasattr(self, 'depth_array'):
            self.depth_array = None
        if hasattr(self, 'camera_info'):
            self.camera_info = None
        
        # Release TF resources
        if hasattr(self, 'tf_buffer'):
            self.tf_buffer = None
        if hasattr(self, 'tf_listener'):
            self.tf_listener = None
        
        # Stop resource monitor and join thread more safely
        if hasattr(self, 'resource_monitor') and self.resource_monitor:
            try:
                self.resource_monitor.stop()
                # Only join if thread exists
                if hasattr(self.resource_monitor, 'monitor_thread') and self.resource_monitor.monitor_thread:
                    self.resource_monitor.monitor_thread.join(timeout=1.0)
            except Exception as e:
                self.get_logger().error(f"Error shutting down resource monitor: {str(e)}")
        
        super().destroy_node()

    def _get_reliable_depth(self, depth_array, pixel_x, pixel_y):
        """
        Get reliable depth with global sampling when local sampling fails.
        Returns the best available depth value.
        """
        # Log detailed info for the first few attempts
        if not hasattr(self, '_depth_debug_count'):
            self._depth_debug_count = 0
        
        self._depth_debug_count += 1
        do_detailed_logging = self._depth_debug_count < 5
        
        # APPROACH 1: Direct center pixel approach first (fastest)
        try:
            depth_value = depth_array[pixel_y, pixel_x]
            if do_detailed_logging:
                self.get_logger().info(f"Center pixel value at ({pixel_x},{pixel_y}): {depth_value}")
                
            if depth_value > 0:
                depth_meters = depth_value * DEPTH_CONFIG["scale"]
                if DEPTH_CONFIG["min_depth"] < depth_meters < DEPTH_CONFIG["max_depth"]:
                    if do_detailed_logging:
                        self.get_logger().info(f"Using center pixel depth: {depth_meters:.3f}m")
                    return depth_meters, 1.0, 1
        except IndexError:
            if do_detailed_logging:
                self.get_logger().warn(f"Index error for center pixel ({pixel_x},{pixel_y}) - shape is {depth_array.shape}")
        
        # APPROACH 2: Try an expanding window search
        max_radius = 20  # Search up to 20 pixels away
        for radius in range(5, max_radius+1, 5):  # Start with 5, then 10, then 15, then 20
            # Calculate region with boundary check
            min_y = max(0, pixel_y - radius)
            max_y = min(depth_array.shape[0], pixel_y + radius + 1)
            min_x = max(0, pixel_x - radius)
            max_x = min(depth_array.shape[1], pixel_x + radius + 1)
            
            if do_detailed_logging and radius == 5:
                self.get_logger().info(f"Sampling region: x={min_x}-{max_x}, y={min_y}-{max_y}, radius={radius}")
            
            # Get the region
            try:
                region = depth_array[min_y:max_y, min_x:max_x]
                
                # Find non-zero values
                non_zero_mask = region > 0
                non_zero_count = np.count_nonzero(non_zero_mask)
                
                # If we found non-zero values, use them
                if non_zero_count > 0:
                    # Scale all depths to meters
                    non_zero_values = region[non_zero_mask]
                    scaled_values = non_zero_values * DEPTH_CONFIG["scale"]
                    
                    # Filter to valid range
                    valid_mask = (scaled_values > 0.1) & (scaled_values < 10.0)
                    valid_values = scaled_values[valid_mask]
                    
                    valid_count = len(valid_values)
                    if valid_count > 0:
                        # Use median depth
                        if valid_count >= 3:
                            depth_meters = float(np.median(valid_values))
                        else:
                            depth_meters = float(np.mean(valid_values))
                        
                        if do_detailed_logging:
                            self.get_logger().info(f"Found {valid_count} valid depth points in radius {radius}")
                            self.get_logger().info(f"Using depth: {depth_meters:.3f}m")
                        
                        reliability = min(1.0, valid_count/10)  # Higher count = higher reliability
                        return depth_meters, reliability, valid_count
            except Exception as e:
                if do_detailed_logging:
                    self.get_logger().error(f"Error in region sampling: {str(e)}")
        
        # APPROACH 3: Global sampling - try the area where the depth camera has valid data
        try:
            # Get a downsampled version of the depth array for speed
            downsampled = depth_array[::10, ::10]  # Take every 10th pixel
            non_zero_mask = downsampled > 0
            non_zero_count = np.count_nonzero(non_zero_mask)
            
            if non_zero_count > 0:
                # Get valid depth values from anywhere in the frame
                non_zero_values = downsampled[non_zero_mask]
                scaled_values = non_zero_values * DEPTH_CONFIG["scale"]
                
                # Filter to reasonable range
                valid_mask = (scaled_values > 0.5) & (scaled_values < 5.0)
                valid_values = scaled_values[valid_mask]
                
                if len(valid_values) > 0:
                    depth_meters = float(np.median(valid_values))
                    
                    if do_detailed_logging:
                        self.get_logger().warn(f"Using global depth sampling as fallback: {depth_meters:.3f}m")
                    
                    return depth_meters, 0.3, 1  # Low reliability since it's from elsewhere in frame
        except Exception as e:
            if do_detailed_logging:
                self.get_logger().error(f"Error in global sampling: {str(e)}")
        
        # APPROACH 4: Last resort - use a default value if all else fails
        if do_detailed_logging:
            self.get_logger().warn("All depth sampling methods failed. Using default depth.")
        
        # Return a reasonable default value as absolute last resort
        return 1.0, 0.1, 0  # 1 meter with very low reliability

    def _check_idle_state(self):
        """Check if the node has been idle for a while and adjust resource usage."""
        current_time = TimeUtils.now_as_float()
        time_since_last_detection = current_time - self.last_detection_time
        
        # If no detections for 15 seconds, enter idle mode
        if time_since_last_detection > 15.0 and not self.is_idle:
            self.is_idle = True
            self.get_logger().info("No recent detections - entering idle mode to conserve resources")
            
            # Increase frame skipping in idle mode
            self.process_every_n_frames = 10  # Process 1 in 10 frames
            
            # Lower the resource monitor check frequency
            if hasattr(self, 'resource_monitor') and self.resource_monitor:
                self.resource_monitor.set_publish_interval(15.0)  # Check less frequently
        
        # If we've had a recent detection but were in idle mode, exit idle mode
        elif time_since_last_detection < 5.0 and self.is_idle:
            self.is_idle = False
            self.get_logger().info("Detections resumed - exiting idle mode")
            
            # Return to previous frame processing rate
            self.process_every_n_frames = 5
            
            # Restore resource monitor frequency
            if hasattr(self, 'resource_monitor') and self.resource_monitor:
                self.resource_monitor.set_publish_interval(10.0)

    def _should_process_detection(self, msg, source):
        """Determine if a detection needs processing based on previous position."""
        if not hasattr(self, 'last_processed_position'):
            self.last_processed_position = {}
            
        # Always process if we haven't seen this source before
        if source not in self.last_processed_position:
            self.last_processed_position[source] = (msg.point.x, msg.point.y)
            return True
            
        # Calculate movement since last processing
        prev_x, prev_y = self.last_processed_position[source]
        new_x, new_y = msg.point.x, msg.point.y
        
        # Calculate squared distance (avoid sqrt for speed)
        dist_squared = (new_x - prev_x)**2 + (new_y - prev_y)**2
        
        # Adaptive threshold based on CPU load
        threshold = 0.001  # Base threshold (1cm movement)
        if hasattr(self, 'current_cpu_usage'):
            # Make threshold larger when CPU is high
            threshold *= (1.0 + self.current_cpu_usage / 50.0)
        
        should_process = dist_squared > threshold
        
        # Update position if we're processing
        if should_process:
            self.last_processed_position[source] = (new_x, new_y)
            
        return should_process

def main(args=None):
    """Main function to initialize and run the 3D position estimator node."""
    # Add debug mode for troubleshooting
    if "--debug-depth" in sys.argv:
        print("Starting in depth debugging mode")
        # Reduce minimum points required and other constraints
        DEPTH_CONFIG["min_valid_points"] = 1
        DEPTH_CONFIG["radius"] = 5
        DEPTH_CONFIG["adaptive_radius"] = True
    
    rclpy.init(args=args)
    
    # Set Raspberry Pi environment variable for other components
    os.environ['RASPBERRY_PI'] = '1'
    
    node = TennisBall3DPositionEstimator()
    
    # Use multiple threads but leave one core for system processes on Raspberry Pi 5
    thread_count = DIAG_CONFIG["threads"]
    executor = MultiThreadedExecutor(num_threads=thread_count)
    executor.add_node(node)
    
    print("=================================================")
    print("Tennis Ball 3D Position Estimator")
    print("=================================================")
    print("This node converts 2D ball detections to 3D positions")
    print("using depth camera data.")
    print("")
    print(f"Subscribing to:")
    print(f"  - Camera info: {TOPICS['input']['camera_info']}")
    print(f"  - Depth image: {TOPICS['input']['depth_image']}")
    print(f"  - YOLO detections: {TOPICS['input']['yolo_detection']}")
    print(f"  - HSV detections: {TOPICS['input']['hsv_detection']}")
    print("")
    print(f"Publishing to:")
    print(f"  - YOLO 3D positions: {TOPICS['output']['yolo_3d']}")
    print(f"  - HSV 3D positions: {TOPICS['output']['hsv_3d']}")
    print(f"  - Combined positions: {TOPICS['output']['combined']}")
    print("")
    print("Press Ctrl+C to stop.")
    print("=================================================")
    
    try:
        node.get_logger().info("3D Position Estimator running. Press Ctrl+C to stop.")
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

# performance_launch.py
def generate_launch_description():
    return LaunchDescription([
        # Performance mode parameter
        DeclareLaunchArgument(
            'performance_mode',
            default_value='balanced',
            description='Performance mode: high_quality, balanced, low_latency, power_saving'
        ),
        
        Node(
            package='ball_tracking',
            executable='depth_camera_node.py',
            name='tennis_ball_3d_position_estimator',
            parameters=[{
                'performance_mode': LaunchConfiguration('performance_mode')
            }],
            arguments=['--performance', LaunchConfiguration('performance_mode')],
            output='screen'
        )
    ])

def main(args=None):
    # Performance profile settings
    performance_mode = 'balanced'  # Default
    
    for i, arg in enumerate(sys.argv):
        if arg == '--performance' and i + 1 < len(sys.argv):
            performance_mode = sys.argv[i + 1]
    
    # Apply performance profile settings
    if performance_mode == 'high_quality':
        DEPTH_CONFIG["min_valid_points"] = 5
        DEPTH_CONFIG["radius"] = 5
        DEPTH_CONFIG["adaptive_radius"] = True
        os.environ['PROCESS_EVERY_N_FRAMES'] = '3'
    elif performance_mode == 'low_latency':
        DEPTH_CONFIG["min_valid_points"] = 1
        DEPTH_CONFIG["radius"] = 2
        DEPTH_CONFIG["adaptive_radius"] = False
        os.environ['PROCESS_EVERY_N_FRAMES'] = '1'
    elif performance_mode == 'power_saving':
        DEPTH_CONFIG["min_valid_points"] = 3
        DEPTH_CONFIG["radius"] = 1
        DEPTH_CONFIG["adaptive_radius"] = False
        os.environ['PROCESS_EVERY_N_FRAMES'] = '15'
    # balanced uses defaults