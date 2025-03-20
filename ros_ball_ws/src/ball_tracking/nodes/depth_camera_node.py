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
    "log_interval": 10.0,      # Increase from 5.0 to 10.0 seconds
    "debug_level": 0,          # Set to minimal (0)
    "threads": 1,             # Number of threads for parallel processing
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

# Add minimal logging option
MINIMAL_LOGGING = True  # Set to False for debug/development
LOG_INTERVAL = 10.0     # Only log every 10 seconds

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
        
        # Initialize core attributes once with default values
        self.init_attributes()
        
        # Setup in logical order - ONCE only
        self._setup_callback_group()
        self._init_camera_parameters()
        self._setup_tf2()
        self._setup_subscriptions()
        self._setup_publishers()
        
        # Initialize the resource monitor
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=20.0,
            enable_temperature=False
        )
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.resource_monitor.start()
        
        # A single timer for performance adjustment
        self.performance_timer = self.create_timer(5.0, self._adjust_performance)
        self.diagnostics_timer = self.create_timer(10.0, self.publish_system_diagnostics)
        
        # Log initialization
        self.get_logger().info("Tennis Ball 3D Position Estimator initialized")
        self.get_logger().info(f"Processing every {self.process_every_n_frames} frames for optimization")
    
    def init_attributes(self):
        """Initialize all attributes with default values."""
        # Performance settings
        self.process_every_n_frames = 5  # Start balanced
        self.frame_counter = 0
        self.current_cpu_usage = 0.0
        self.radius = 3
        self._min_valid_points = 3
        self._scale_factor = DEPTH_CONFIG["scale"]
        self._min_valid_depth = DEPTH_CONFIG["min_depth"]
        self._max_valid_depth = DEPTH_CONFIG["max_depth"]
        
        # Add these lines to fix the error
        self.yolo_count = 0
        self.hsv_count = 0
        
        # Camera parameters (will be updated later)
        self.camera_info = None
        self.depth_array = None
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        self.depth_width = 640
        self.depth_height = 480
        self.x_scale = 1.0
        self.y_scale = 1.0
        
        # Tracking variables
        self.start_time = TimeUtils.now_as_float()
        self.successful_conversions = 0
        self.processing_times = deque(maxlen=20)
        self.depth_camera_health = 1.0
        self.last_fps_log_time = 0
        self.verified_transform = False
        self.camera_info_logged = False
        
        # Caching mechanism
        self.position_cache = {
            'YOLO': {'position': None, 'depth': None, 'timestamp': 0, '3d_position': None},
            'HSV': {'position': None, 'depth': None, 'timestamp': 0, '3d_position': None}
        }
        self.cache_hits = {'YOLO': 0, 'HSV': 0}
        self.attempt_counter = {'YOLO': 0, 'HSV': 0}
        self.cache_validity_duration = 0.5  # 500ms for all sources
        self.yolo_movement_threshold = 0.1  # 10% of image for YOLO
        self.hsv_movement_threshold = 0.4   # 40% of image for HSV (more jitter)
        
        # Error tracking
        self.error_counts = {}
        self.error_last_logged = {}
        
        # Bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Initialize everything that would be checked with hasattr()
        self.last_resource_alert_time = 0
        self.last_high_cpu_time = 0
        self.last_cache_debug_time = 0
        self.last_diag_log_time = 0
        self.last_cache_log = 0
        self.last_minimal_log_time = 0
        self.transform_not_verified_logged = False
        self.is_idle = False
        self.last_detection_time = TimeUtils.now_as_float()
        self.last_batch_process = TimeUtils.now_as_float()
        
        # Initialize containers
        self.detection_buffer = {'YOLO': [], 'HSV': []}
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
        
        # Simplified caching mechanism with fixed parameters
        self.position_cache = {
            'YOLO': {'position': None, 'depth': None, 'timestamp': 0, '3d_position': None},
            'HSV': {'position': None, 'depth': None, 'timestamp': 0, '3d_position': None}
        }
        self.cache_hits = {'YOLO': 0, 'HSV': 0}
        self.attempt_counter = {'YOLO': 0, 'HSV': 0}
        
        # Fixed cache parameters - no dynamic adjustment needed
        self.cache_validity_duration = 0.5  # 500ms for all sources
        self.yolo_movement_threshold = 0.1  # 10% of image for YOLO
        self.hsv_movement_threshold = 0.4   # 40% of image for HSV (more jitter)
        self.cache_validity_duration = 0.5  # Default for moving balls
        self.yolo_movement_threshold = 0.1
        self.hsv_movement_threshold = 0.4
    
    def _setup_callback_group(self):
        """Set up callback group and QoS profile for subscriptions."""
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        
        # Create a callback group for concurrent callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        # Create QoS profiles for various subscription types
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
        )
    
    def _adjust_frame_skipping(self):
        """More aggressive frame skipping when needed."""
        cpu_usage = self.current_cpu_usage
        
        # Ultra-aggressive CPU management
        if cpu_usage > 95:
            self.process_every_n_frames = min(20, self.process_every_n_frames + 3)
            # Log this change but only once in a while
            if not hasattr(self, 'last_high_cpu_time') or TimeUtils.now_as_float() - self.last_high_cpu_time > 30.0:
                self.get_logger().warn(f"Very high CPU ({cpu_usage:.1f}%), skipping {self.process_every_n_frames} frames")
                self.last_high_cpu_time = TimeUtils.now_as_float()
        elif cpu_usage > 85:
            self.process_every_n_frames = min(15, self.process_every_n_frames + 1)
        elif cpu_usage > 75:
            self.process_every_n_frames = min(10, max(5, self.process_every_n_frames))
        elif cpu_usage < 55 and self.process_every_n_frames > 5:
            self.process_every_n_frames = max(5, self.process_every_n_frames - 1)
        elif cpu_usage < 35 and self.process_every_n_frames > 3:
            self.process_every_n_frames = max(3, self.process_every_n_frames - 1)
    
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
        offsets = [(-5, 0), (5, 0), (0, -5), (0, 5), (-3, -3), (-3, 3), (3, -3), (3, 3)]
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
        """Verify transform is registered and cancel verification timer if successful."""
        try:
            if self.tf_buffer.can_transform(
                self.reference_frame,
                "camera_frame",
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            ):
                self.verified_transform = True
                self.get_logger().info("Transform verification successful. Transform chain is complete.")
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
        """Simplified error logging with rate limiting."""
        current_time = TimeUtils.now_as_float()
        
        # Simple rate limiting - log each error type at most once every 10 seconds
        if error_message not in self.error_last_logged or current_time - self.error_last_logged[error_message] > 10.0:
            self.error_last_logged[error_message] = current_time
            
            if is_warning:
                self.get_logger().warning(f"DEPTH: {error_message}")
            else:
                self.get_logger().error(f"DEPTH: {error_message}")
    
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

    def depth_callback(self, msg):
        """Process depth image with simple frame skipping for efficiency."""
        try:
            # Skip frames based on current setting
            self.frame_counter += 1
            if self.frame_counter % self.process_every_n_frames != 0:
                return
                
            # Convert with CvBridge - no conditional downsampling needed
            self.depth_array = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_header = msg.header
            
            # Process any pending detections now that we have depth data
            self._process_detection_batch()
        except Exception as e:
            pass
    
    def yolo_callback(self, msg):
        """Buffer YOLO detections for batch processing."""
        if not hasattr(self, 'detection_buffer'):
            self.detection_buffer = {'YOLO': [], 'HSV': []}
            self.last_batch_process = TimeUtils.now_as_float()
            
        # Add to buffer
        self.detection_buffer['YOLO'].append((msg, TimeUtils.now_as_float()))
        
        # Process if enough time has passed or buffer is large
        now = TimeUtils.now_as_float()
        if now - self.last_batch_process > 0.2 or len(self.detection_buffer['YOLO']) > 5:
            self._process_detection_batch()
        
    def hsv_callback(self, msg):
        """Buffer HSV detections for batch processing."""
        if not hasattr(self, 'detection_buffer'):
            self.detection_buffer = {'YOLO': [], 'HSV': []}
            self.last_batch_process = TimeUtils.now_as_float()
            
        # Add to buffer
        self.detection_buffer['HSV'].append((msg, TimeUtils.now_as_float()))
        
        # Process if enough time has passed or buffer is large
        now = TimeUtils.now_as_float()
        if now - self.last_batch_process > 0.2 or len(self.detection_buffer['HSV']) > 5:
            self._process_detection_batch()
        
    def _process_detection_batch(self):
        """Process all buffered detections at once."""
        if not hasattr(self, 'detection_buffer') or not self.detection_buffer:
            return
            
        # Skip if missing data
        if self.camera_info is None or self.depth_array is None:
            return
        
        start_time = TimeUtils.now_as_float()
        processed = 0
        
        # Process latest detections only
        for source in ['YOLO', 'HSV']:
            if self.detection_buffer[source]:
                latest_msg, _ = max(self.detection_buffer[source], key=lambda x: x[1])
                if self._should_process_detection(latest_msg, source):
                    self.get_3d_position(latest_msg, source)
                    processed += 1
                self.detection_buffer[source] = []
        
        # Track processing time
        if processed > 0:
            processing_time = (TimeUtils.now_as_float() - start_time) * 1000
            self.processing_times.append(processing_time / processed)
    
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
                
            # Adjust coordinates if they're out of bounds rather than failing
            if pixel_x < margin:
                pixel_x = margin
            elif pixel_x >= depth_width - margin:
                pixel_x = depth_width - margin - 1
                
            if pixel_y < margin:
                pixel_y = margin
            elif pixel_y >= depth_height - margin:
                pixel_y = depth_height - margin - 1
            
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

                # Store in cache for future reuse
                self.position_cache[source]['3d_position'] = transformed_msg
                self.position_cache[source]['depth'] = median_depth
                
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
    
    def _transform_to_reference_frame(self, point_stamped):
        """Transform with caching for efficiency."""
        now = TimeUtils.now_as_float()
        
        # Use cached transform if valid and recent
        if hasattr(self, '_last_transform') and hasattr(self, '_last_transform_time'):
            if now - self._last_transform_time < 0.5:  # Valid for 500ms
                transformed = tf2_geometry_msgs.do_transform_point(point_stamped, self._last_transform)
                return transformed
        
        # Otherwise get a new transform
        try:
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                point_stamped.header.frame_id,
                rclpy.time.Time())
            
            # Cache it
            self._last_transform = transform
            self._last_transform_time = now
            
            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed
        except:
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
        """Simplified diagnostics output."""
        current_time = TimeUtils.now_as_float()
        
        # Only log once per LOG_INTERVAL seconds
        if not hasattr(self, 'last_minimal_log_time') or current_time - self.last_minimal_log_time > LOG_INTERVAL:
            self.last_minimal_log_time = current_time
            total_fps = self.successful_conversions / (current_time - self.start_time) if current_time > self.start_time else 0
            self.get_logger().info(
                f"Depth camera: {total_fps:.1f} FPS, CPU: {self.current_cpu_usage:.1f}%, "
                f"Frames: 1:{self.process_every_n_frames}"
            )
    
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
        Optimized depth sampling with better reliability and caching.
        """
        # Try center pixel first (fastest approach)
        try:
            center_value = depth_array[pixel_y, pixel_x]
            if center_value > 0:
                depth_m = float(center_value * self._scale_factor)
                if self._min_valid_depth < depth_m < self._max_valid_depth:
                    return depth_m, 1.0, 1
        except:
            pass
        
        # If center fails, sample at fixed distances in a cross pattern
        offsets = [(0, -3), (0, 3), (-3, 0), (3, 0)]
        valid_depths = []
        h, w = depth_array.shape
        
        for offset_y, offset_x in offsets:
            y = pixel_y + offset_y
            x = pixel_x + offset_x
            
            if 0 <= y < h and 0 <= x < w:
                try:
                    val = depth_array[y, x]
                    if val > 0:
                        depth_m = float(val * self._scale_factor)
                        if self._min_valid_depth < depth_m < self._max_valid_depth:
                            valid_depths.append(depth_m)
                            # Early exit if we have at least 2 valid points
                            if len(valid_depths) >= 2:
                                return sum(valid_depths) / len(valid_depths), 0.9, len(valid_depths)
                except:
                    continue
        
        # If we found any valid points
        if valid_depths:
            return valid_depths[0], 0.7, len(valid_depths)
        
        # Last resort - use a fixed reasonable value with low confidence
        return 1.6, 0.1, 0  # Use 1.6m based on logs showing the ball at that distance

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
        """Simple check if we should process this detection or use cache."""
        current_time = TimeUtils.now_as_float()
        curr_x, curr_y = msg.point.x, msg.point.y  # Fixed
        
        # Count attempt for diagnostics
        if not hasattr(self, 'attempt_counter'):
            self.attempt_counter = {'YOLO': 0, 'HSV': 0}
        self.attempt_counter[source] += 1
        
        # Simple cache check
        if source in self.position_cache and self.position_cache[source]['position'] is not None:
            prev_x, prev_y = self.position_cache[source]['position']
            cache_time = self.position_cache[source]['timestamp']
            
            # Calculate distance and time since cache
            dist_squared = (curr_x - prev_x)**2 + (curr_y - prev_y)**2
            time_since_cache = current_time - cache_time
            
            # Different threshold for HSV vs YOLO
            threshold = 0.1 if source == 'YOLO' else 0.4  # Higher for HSV's jitter
            
            # Log occasionally for debugging
            if not hasattr(self, 'last_cache_debug_time') or current_time - self.last_cache_debug_time > 60.0:
                self.get_logger().info(f"Cache decision: dist={dist_squared:.6f}, threshold={threshold:.6f}, time={time_since_cache:.1f}s")
                self.last_cache_debug_time = current_time
            
            # Use cache if position hasn't changed much and cache is fresh
            if dist_squared < threshold and time_since_cache < 0.5:  # 500ms cache validity
                if self.position_cache[source]['3d_position'] is not None:
                    success = self._republish_cached_position(source, 
                        self.yolo_3d_publisher if source == "YOLO" else self.hsv_3d_publisher)
                    
                    if success:
                        # Only count cache hit here, not in republish function
                        if not hasattr(self, 'cache_hits'):
                            self.cache_hits = {'YOLO': 0, 'HSV': 0}
                        self.cache_hits[source] += 1
                        
                        # Update timestamp
                        self.position_cache[source]['timestamp'] = current_time
                        return False  # Skip processing
        
        # Update cache with new position
        self.position_cache[source]['position'] = (curr_x, curr_y)
        self.position_cache[source]['timestamp'] = current_time
        self.position_cache[source]['3d_position'] = None
        
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
        msg.header.stamp = TimeUtils.now_as_ros_time()  # Use current time
        msg.point.x = cached_pos.point.x
        msg.point.y = cached_pos.point.y
        msg.point.z = cached_pos.point.z
        
        # Publish to specific publisher
        publisher.publish(msg)
        
        # Also publish to combined topic for compatibility
        self.position_publisher.publish(msg)
        
        # Count as a successful conversion
        self.successful_conversions += 1
        
        # Mark as a cache hit for diagnostics
        if not hasattr(self, 'cache_hits'):
            self.cache_hits = {'YOLO': 0, 'HSV': 0}
        self.cache_hits[source] += 1
        
        # Update the timestamp to extend cache validity
        self.position_cache[source]['timestamp'] = TimeUtils.now_as_float()
        
        # Log cache hit every 100 hits
        if self.cache_hits[source] % 100 == 0:
            self.get_logger().info(f"Cache hit {self.cache_hits[source]} for {source}")
        
        return True

    def _process_detection_queue(self):
        """Process all queued detections efficiently."""
        start_time = TimeUtils.now_as_float()
        
        # Skip processing if we're missing data
        if self.camera_info is None or not hasattr(self, 'depth_array') or self.depth_array is None:
            return
            
        # Process up to 5 detections at once to avoid getting behind
        processed_count = 0
        while not self.detection_queue.empty() and processed_count < 5:
            try:
                msg, source, timestamp = self.detection_queue.get_nowait()
                
                # Quick check if we should process this detection
                if self._should_process_detection(msg, source):
                    self._process_single_detection(msg, source, timestamp)
                    
                # Always mark task as done
                self.detection_queue.task_done()
                processed_count += 1
                
            except Empty:
                break
                
        # Track processing time
        if processed_count > 0:
            processing_time = (TimeUtils.now_as_float() - start_time) * 1000
            self.processing_times.append(processing_time / processed_count)
            
            # Keep only recent times
            if len(self.processing_times) > 30:
                self.processing_times = self.processing_times[-30:]

    def _process_single_detection(self, msg, source, timestamp):
        """Process a single detection efficiently with minimal logging."""
        try:
            # Scale coordinates to depth image space
            pixel_x = int(round(msg.point.x * self.x_scale))
            pixel_y = int(round(msg.point.y * self.y_scale))
            
            # Get depth and reliability
            depth_meters, reliability, points = self._get_reliable_depth(self.depth_array, pixel_x, pixel_y)
            if depth_meters <= 0.0:
                self.failure_reasons['no_reliable_depth'] += 1
                return
                
            # Convert 2D to 3D using camera model
            x = (pixel_x - self.cx) * depth_meters / self.fx
            y = (pixel_y - self.cy) * depth_meters / self.fy
            z = depth_meters
            
            # Create position message
            camera_position = PointStamped()
            camera_position.header.frame_id = "camera_frame"
            camera_position.header.stamp = timestamp or TimeUtils.now_as_ros_time()
            camera_position.point.x = x
            camera_position.point.y = y
            camera_position.point.z = z
            
            # Transform and publish
            transformed = self._transform_to_reference_frame(camera_position)
            if transformed:
                # Source-specific publishing
                if source == "YOLO":
                    self.yolo_3d_publisher.publish(transformed)
                else:  # HSV
                    self.hsv_3d_publisher.publish(transformed)
                    
                # Also publish to combined topic
                self.position_publisher.publish(transformed)
                
                # Update stats
                self.successful_conversions += 1
                self.failure_reasons['successful'] += 1
                self.last_detection_time = TimeUtils.now_as_float()
                
                # Update detection history (only store the latest few)
                if hasattr(self, 'detection_history'):
                    self.detection_history[source] = {
                        'x': transformed.point.x,
                        'y': transformed.point.y,
                        'z': transformed.point.z,
                        'last_time': self.last_detection_time
                    }
                    
                # Log performance less frequently
                if self.successful_conversions % 10 == 0:
                    self._log_performance(transformed.point.x, 
                                          transformed.point.y, 
                                          transformed.point.z, source)
            else:
                self.failure_reasons['transform_failed'] += 1
                
        except Exception as e:
            self.failure_reasons['exceptions'] += 1
            # Only log serious errors that aren't common
            if "out of bounds" not in str(e).lower():
                self.get_logger().error(f"Error: {str(e)}")

    def _auto_adjust_performance(self):
        """Automatically adjust performance settings based on system load."""
        if not hasattr(self, 'last_performance_check'):
            self.last_performance_check = TimeUtils.now_as_float()
            self.performance_check_interval = 5.0  # Check every 5 seconds
            return
        
        now = TimeUtils.now_as_float()
        if now - self.last_performance_check < self.performance_check_interval:
            return
        
        self.last_performance_check = now
        
        # Get current CPU usage
        cpu_usage = self.current_cpu_usage if hasattr(self, 'current_cpu_usage') else 50.0
        
        # Ultra light mode
        if cpu_usage > 90:
            self.process_every_n_frames = min(15, self.process_every_n_frames + 1)
            self.radius = 1
            self._min_valid_points = 1
            self.enable_downsampling = True
            self.enable_transform_caching = True
            self.get_logger().info(f"Switched to ultra light mode: skipping {self.process_every_n_frames} frames")
        
        # Performance mode
        elif cpu_usage > 80:
            self.process_every_n_frames = min(10, max(5, self.process_every_n_frames))
            self.radius = 2
            self._min_valid_points = 2
        
        # Balanced mode
        elif cpu_usage > 60:
            self.process_every_n_frames = 5
            self.radius = 3
            self._min_valid_points = 3
        
        # Quality mode
        else:
            self.process_every_n_frames

        # Dynamically adjust caching parameters based on ball movement patterns
        if hasattr(self, 'last_movement_samples'):
            avg_movement = sum(self.last_movement_samples) / len(self.last_movement_samples)
            
            # If ball is mostly stationary, use longer cache validity
            if avg_movement < 0.002:  # Very little movement
                self.cache_validity_duration = 1.0  # 1 second cache 
            elif avg_movement < 0.01:  # Small movements
                self.cache_validity_duration = 0.5  # 500ms cache
            else:  # Significant movement
                self.cache_validity_duration = 0.2  # 200ms cache

        # Increase cache duration for better efficiency with stationary objects
        if not hasattr(self, 'last_movement_samples'):
            self.last_movement_samples = deque(maxlen=10)
            self.last_movement_samples.extend([0.0] * 10)  # Initialize with no movement
        
        # Always increase cache validity duration for stationary balls
        self.cache_validity_duration = 5.0  # Increase to 5 seconds for better cache use
        
        # Only adjust other parameters based on movement when we actually have samples
        if sum(self.last_movement_samples) > 0:
            avg_movement = sum(self.last_movement_samples) / len(self.last_movement_samples)
            
            # If ball is mostly stationary, use even longer cache validity
            if avg_movement < 0.002:  # Very little movement
                self.cache_validity_duration = 3.0  # Increase to 3 seconds
            elif avg_movement < 0.01:  # Small movements
                self.cache_validity_duration = 2.0  # 2 seconds cache
            else:  # Significant movement
                self.cache_validity_duration = 0.5  # 500ms cache for moving balls

    def _adjust_performance(self):
        """Single unified performance adjustment method."""
        cpu_usage = self.current_cpu_usage if hasattr(self, 'current_cpu_usage') else 50.0
        old_frame_skip = self.process_every_n_frames
        
        # Simple three-tier adjustment
        if cpu_usage > 85:
            self.process_every_n_frames = min(15, self.process_every_n_frames + 1)
            self.radius = 1
            self._min_valid_points = 1
        elif cpu_usage > 65:
            self.process_every_n_frames = min(10, max(5, self.process_every_n_frames))
            self.radius = 2
            self._min_valid_points = 2
        elif cpu_usage < 50:
            self.process_every_n_frames = max(3, self.process_every_n_frames - 1)
            self.radius = 3
            self._min_valid_points = 3
        
        # Only log when changes occur
        if old_frame_skip != self.process_every_n_frames:
            self.get_logger().info(f"Adjusted processing: 1 in {self.process_every_n_frames} frames (CPU: {cpu_usage:.1f}%)")

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

def publish_system_diagnostics(self):
    # Skip heavy diagnostics when performance matters
    current_time = TimeUtils.now_as_float()
    if MINIMAL_LOGGING or self.current_cpu_usage > 80:
        # Only log once per LOG_INTERVAL seconds
        if not hasattr(self, 'last_minimal_log_time') or current_time - self.last_minimal_log_time > LOG_INTERVAL:
            self.last_minimal_log_time = current_time
            total_fps = self.successful_conversions / (current_time - self.start_time) if current_time > self.start_time else 0
            self.get_logger().info(
                f"Depth camera: {total_fps:.1f} FPS, CPU: {self.current_cpu_usage:.1f}%, "
                f"Frames: 1:{self.process_every_n_frames}"
            )
        return