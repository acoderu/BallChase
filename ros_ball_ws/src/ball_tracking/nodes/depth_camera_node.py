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
        self._init_camera_parameters()  # Keep this one
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
        
        # Tracking counters
        self.yolo_count = 0
        self.hsv_count = 0
        
        # Add this line to initialize detection_history
        self.detection_history = {
            'YOLO': {'count': 0, 'latest_position': None, 'last_time': 0},
            'HSV': {'count': 0, 'latest_position': None, 'last_time': 0}
        }
        
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
        
        # Initialize timestamps to avoid hasattr checks
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
        
        # Call existing methods to initialize other parameters
        self._init_camera_parameters()
        
        # Add these to your existing attributes
        self.seq_counter = 0
        self.last_critical_alert = 0
        self._transform_error_logged = False
        self.detection_health = 0.8
        self.depth_reliability = {
            'YOLO': {'value': 0.0, 'valid_points': 0, 'timestamp': 0},
            'HSV': {'value': 0.0, 'valid_points': 0, 'timestamp': 0}
        }
    
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
        
        # Configuration for depth sampling
        self.radius = DEPTH_CONFIG["radius"]
        
        # Bridge for faster conversion
        self.cv_bridge = CvBridge()
    
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
            "/tennis_ball/depth_camera/diagnostics",
            10
        )
    
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
        if not self.camera_info_logged:
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
    
    def detection_callback(self, msg, source):
        """Generic callback for buffering detections."""
        # Add to buffer
        self.detection_buffer[source].append((msg, TimeUtils.now_as_float()))
        
        # Process if enough time has passed or buffer is large
        now = TimeUtils.now_as_float()
        if now - self.last_batch_process > 0.2 or len(self.detection_buffer[source]) > 5:
            self._process_detection_batch()

    def yolo_callback(self, msg):
        """Buffer YOLO detections."""
        self.detection_callback(msg, 'YOLO')
    
    def hsv_callback(self, msg):
        """Buffer HSV detections."""
        self.detection_callback(msg, 'HSV')
        
    def _process_detection_batch(self):
        """Process all buffered detections at once."""
        if not self.detection_buffer:
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
    
    def get_3d_position(self, detection_msg, source):
        """
        Convert a 2D ball detection to a 3D position using depth data.
        
        Args:
            detection_msg (PointStamped): 2D position from detection node
            source (str): Detection source ("YOLO" or "HSV")
            
        Returns:
            bool: True if 3D position was successfully calculated and published
        """
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
            if not self.transform_not_verified_logged:
                self.log_error("Transform not yet verified - waiting for transform chain to be complete", True)
                self.transform_not_verified_logged = True
            
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
        frame_key = f"{self.reference_frame}_{point_stamped.header.frame_id}"
        
        # Check if we have this transform in cache and it's still valid
        if frame_key in self.transform_cache:
            cached_time, cached_transform = self.transform_cache[frame_key]
            if now - cached_time < self.transform_cache_lifetime:
                transformed = tf2_geometry_msgs.do_transform_point(point_stamped, cached_transform)
                return transformed
        
        # Otherwise get a new transform
        try:
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                point_stamped.header.frame_id,
                rclpy.time.Time())
            
            # Cache it
            self.transform_cache[frame_key] = (now, transform)
            
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
        
        # Update detection history
        self.detection_history[source]['count'] += 1
        self.detection_history[source]['latest_position'] = (x, y, z)
        self.detection_history[source]['last_time'] = current_time
    
    def publish_system_diagnostics(self):
        """Simplified diagnostics output with minimal logging."""
        current_time = TimeUtils.now_as_float()
        
        # Only publish every LOG_INTERVAL seconds
        if current_time - self.last_diag_log_time < LOG_INTERVAL:
            return
            
        self.last_diag_log_time = current_time
        
        # Calculate core metrics
        elapsed = current_time - self.start_time
        fps = self.successful_conversions / elapsed if elapsed > 0 else 0
        
        # Simple status line (more concise for high CPU)
        if MINIMAL_LOGGING or self.current_cpu_usage > 80:
            self.get_logger().info(
                f"Depth camera: {fps:.1f} FPS, CPU: {self.current_cpu_usage:.1f}%, "
                f"Frames: 1:{self.process_every_n_frames}"
            )
        else:
            # More detailed logging when not in minimal mode
            self.get_logger().info(
                f"Depth camera: {fps:.1f} FPS, CPU: {self.current_cpu_usage:.1f}%, "
                f"Frames: 1:{self.process_every_n_frames}, "
                f"Cache hit rate: {self.cache_hits['YOLO']}/{self.attempt_counter['YOLO']} YOLO, "
                f"{self.cache_hits['HSV']}/{self.attempt_counter['HSV']} HSV"
            )
    
    def _handle_resource_alert(self, resource_type, value):
        """Log resource alerts and trigger performance adjustment."""
        current_time = TimeUtils.now_as_float()
        if current_time - self.last_resource_alert_time < 5.0:
            return  # Avoid too frequent alerts
            
        self.last_resource_alert_time = current_time
        
        if resource_type == 'cpu' and value > 90.0:
            self.log_error(f"High {resource_type} usage ({value:.1f}%) - reducing processing load", True)
        
        # Let the main adjustment method handle the actual changes
        self.current_cpu_usage = value if resource_type == 'cpu' else self.current_cpu_usage
        self._adjust_performance()
    
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
        """Get reliable depth at the specified pixel coordinates."""
        try:
            # Try center pixel first
            center_value = depth_array[pixel_y, pixel_x]
            if center_value > 0:
                depth_m = float(center_value * self._scale_factor)
                if self._min_valid_depth < depth_m < self._max_valid_depth:
                    return depth_m, 1.0, 1
            
            # If center fails, try cross pattern
            valid_depths = []
            offsets = [(0, -3), (0, 3), (-3, 0), (3, 0), (-2, -2), (2, 2), (-2, 2), (2, -2)]
            
            for offset_y, offset_x in offsets:
                y = pixel_y + offset_y
                x = pixel_x + offset_x
                
                try:
                    val = depth_array[y, x]
                    if val > 0:
                        depth_m = float(val * self._scale_factor)
                        if self._min_valid_depth < depth_m < self._max_valid_depth:
                            valid_depths.append(depth_m)
                except:
                    pass
            
            # If we found valid depths, use their median
            if valid_depths:
                return sum(valid_depths) / len(valid_depths), 0.8, len(valid_depths)
            
            # Fallback value
            return 1.6, 0.1, 0
            
        except Exception as e:
            return 1.6, 0.1, 0

    def _should_process_detection(self, msg, source):
        """Simple check if we should process this detection or use cache."""
        current_time = TimeUtils.now_as_float()
        curr_x, curr_y = msg.point.x, msg.point.y
        
        # Check cache
        if self.position_cache[source]['position'] is not None:
            prev_x, prev_y = self.position_cache[source]['position']
            cache_time = self.position_cache[source]['timestamp']
            
            # Calculate distance and time since cache
            dist_squared = (curr_x - prev_x)**2 + (curr_y - prev_y)**2
            time_since_cache = current_time - cache_time
            
            # Different threshold for HSV vs YOLO
            threshold = self.yolo_movement_threshold if source == 'YOLO' else self.hsv_movement_threshold
            
            # Log occasionally for debugging
            if current_time - self.last_cache_debug_time > 60.0:
                self.get_logger().info(f"Cache decision: dist={dist_squared:.6f}, threshold={threshold:.6f}, time={time_since_cache:.1f}s")
                self.last_cache_debug_time = current_time
            
            # Use cache if position hasn't changed much and cache is fresh
            if dist_squared < threshold and time_since_cache < self.cache_validity_duration:
                if self.position_cache[source]['3d_position'] is not None:
                    success = self._republish_cached_position(source, 
                        self.yolo_3d_publisher if source == "YOLO" else self.hsv_3d_publisher)
                    
                    if success:
                        self.cache_hits[source] += 1
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
        msg.point = cached_pos.point  # Copy the entire point structure
        
        # Publish to specific publisher and to combined topic
        publisher.publish(msg)
        self.position_publisher.publish(msg)
        
        # Count as a successful conversion and cache hit
        self.successful_conversions += 1
        
        # Update the timestamp to extend cache validity
        self.position_cache[source]['timestamp'] = TimeUtils.now_as_float()
        
        # Log cache hit every 100 hits
        if self.cache_hits[source] % 100 == 0:
            self.get_logger().info(f"Cache hit {self.cache_hits[source]} for {source}")
        
        return True

    def _adjust_performance(self):
        """Single unified performance adjustment method."""
        cpu_usage = self.current_cpu_usage
        old_frame_skip = self.process_every_n_frames
        
        # CPU-based adjustments
        if cpu_usage > 90:  # Critical CPU load - most aggressive
            self.process_every_n_frames = min(20, self.process_every_n_frames + 2)
            self.radius = 1
            self._min_valid_points = 1
            
            # Log critical situation (rate-limited)
            current_time = TimeUtils.now_as_float()
            if current_time - self.last_critical_alert > 30.0:
                self.get_logger().warning(f"Critical CPU load ({cpu_usage:.1f}%), reducing processing load")
                self.last_critical_alert = current_time
        elif cpu_usage > 85:  # High CPU load
            self.process_every_n_frames = min(15, self.process_every_n_frames + 1)
            self.radius = 1
            self._min_valid_points = 1
        elif cpu_usage > 65:  # Medium CPU load
            self.process_every_n_frames = min(10, max(5, self.process_every_n_frames))
            self.radius = 2
            self._min_valid_points = 2
        elif cpu_usage < 50:  # Low CPU load
            self.process_every_n_frames = max(3, self.process_every_n_frames - 1)
            self.radius = 3
            self._min_valid_points = 3
        
        # Only log when changes occur
        if old_frame_skip != self.process_every_n_frames:
            self.get_logger().info(f"Adjusted processing: 1 in {self.process_every_n_frames} frames (CPU: {cpu_usage:.1f}%)")

def apply_debug_flags():
    """Apply all command line debug flags."""
    if "--debug-depth" in sys.argv:
        print("Starting in depth debugging mode")
        DEPTH_CONFIG["min_valid_points"] = 1
        DEPTH_CONFIG["radius"] = 5
        DEPTH_CONFIG["adaptive_radius"] = True
        
    if "--performance" in sys.argv:
        idx = sys.argv.index("--performance")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]
            apply_performance_mode(mode)

def apply_performance_mode(mode):
    """Apply performance mode settings."""
    if mode == 'high_quality':
        DEPTH_CONFIG["min_valid_points"] = 5
        DEPTH_CONFIG["radius"] = 5
        DEPTH_CONFIG["adaptive_radius"] = True
        os.environ['PROCESS_EVERY_N_FRAMES'] = '3'
    elif mode == 'low_latency':
        DEPTH_CONFIG["min_valid_points"] = 1
        DEPTH_CONFIG["radius"] = 2
        DEPTH_CONFIG["adaptive_radius"] = False
        os.environ['PROCESS_EVERY_N_FRAMES'] = '1'
    elif mode == 'power_saving':
        DEPTH_CONFIG["min_valid_points"] = 3
        DEPTH_CONFIG["radius"] = 1
        DEPTH_CONFIG["adaptive_radius"] = False
        os.environ['PROCESS_EVERY_N_FRAMES'] = '15'

def main(args=None):
    """Main function to initialize and run the 3D position estimator node."""
    # Apply debug and performance flags
    apply_debug_flags()
    
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
    except Exception as e:
        node.get_logger().error(f"Error: {str(e)}")
    finally:
        # Stop any monitoring or background tasks
        if hasattr(node, 'resource_monitor'):
            node.resource_monitor.stop()
            
        # Clean up
        node.destroy_node()
        rclpy.shutdown()

# Launch file function (keep this separate from the main function)
def generate_launch_description():
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'performance_mode',
            default_value='balanced',
            description='Performance mode: high_quality, balanced, low_latency, power_saving'
        ),
        
        Node(
            package='ball_tracking',
            executable='depth_camera_node',
            name='tennis_ball_3d_position_estimator',
            parameters=[{
                'performance_mode': LaunchConfiguration('performance_mode')
            }],
            arguments=['--performance', LaunchConfiguration('performance_mode')],
            output='screen'
        )
    ])

if __name__ == '__main__':
    main()
