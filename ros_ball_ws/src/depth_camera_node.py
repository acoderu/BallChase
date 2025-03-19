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

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CameraInfo
import numpy as np
import time
from cv_bridge import CvBridge
import os
from config.config_loader import ConfigLoader  # Import ConfigLoader
import json
from ball_tracking.time_utils import TimeUtils  # Add TimeUtils import
from std_msgs.msg import String

# Load configuration from file
config_loader = ConfigLoader()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'depth_config.yaml')
config = config_loader.load_yaml(config_path)

# Configuration from config file
DEPTH_CONFIG = config.get('depth', {
    "scale": 0.001,           # Depth scale factor (converts raw depth to meters)
    "min_depth": 0.1,         # Minimum valid depth in meters
    "max_depth": 8.0,         # Maximum valid depth in meters
    "radius": 3,              # Radius around detection point to sample depth values
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
    
# Add resource monitoring import
from ball_tracking.resource_monitor import ResourceMonitor

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
        
        # Add Raspberry Pi resource monitoring
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=10.0,
            enable_temperature=True
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
        
        # Performance tracking variables
        self._init_performance_tracking()
        
        # Add downsampling flag for Raspberry Pi
        self.enable_downsampling = True
        if self.enable_downsampling:
            self.get_logger().info("Depth processing downsampling enabled for Raspberry Pi optimization")
        
        self.get_logger().info("Tennis Ball 3D Position Estimator initialized")
        self.get_logger().info(f"Using coordinate scaling: detection ({DEPTH_CONFIG['detection_resolution']['width']}x"
                              f"{DEPTH_CONFIG['detection_resolution']['height']}) -> depth (will be updated when received)")
    
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
        self.x_scale = self.depth_width / DEPTH_CONFIG["detection_resolution"]["width"]
        self.y_scale = self.depth_height / DEPTH_CONFIG["detection_resolution"]["height"]
        
        # Configuration for depth sampling
        self.radius = DEPTH_CONFIG["radius"]
        
        # Pre-compute radius search offsets for faster lookup
        self.offsets = []
        for y in range(-self.radius, self.radius+1):
            for x in range(-self.radius, self.radius+1):
                self.offsets.append((x, y))
    
    def _setup_tf2(self):
        """Set up tf2 components for coordinate transformations."""
        # Import tf2 modules here to avoid circular imports
        from tf2_ros import Buffer, TransformListener
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("Transform listener initialized for cross-sensor coordination")
    
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
        self.diagnostics_timer = self.create_timer(2.0, self.publish_system_diagnostics)
    
    def _init_performance_tracking(self):
        """Initialize performance tracking variables."""
        self.start_time = TimeUtils.now_as_float()  # Use TimeUtils instead of time.time()
        self.yolo_count = 0
        self.hsv_count = 0
        self.successful_conversions = 0
        self.last_fps_log_time = TimeUtils.now_as_float()  # Use TimeUtils
        self.processing_times = []
        
        # Add error tracking
        self.errors = []
        self.warnings = []
        self.error_history_size = DIAG_CONFIG.get('error_history_size', 10)
        self.last_error_time = 0
        
        # Add health metrics
        self.depth_camera_health = 1.0  # 0.0 to 1.0 scale
        self.processing_health = 1.0
        self.detection_health = 1.0
    
    def log_error(self, error_message, is_warning=False):
        """Log an error and add it to error history for diagnostics."""
        if is_warning:
            self.get_logger().warning(f"DEPTH: {error_message}")
            # Add to warning list for diagnostics
            self.warnings.append({
                "timestamp": TimeUtils.now_as_float(),  # Use TimeUtils
                "message": error_message
            })
        else:
            self.get_logger().error(f"DEPTH: {error_message}")
            # Add to error list for diagnostics
            self.errors.append({
                "timestamp": TimeUtils.now_as_float(),  # Use TimeUtils
                "message": error_message
            })
            
            # Keep error list at appropriate size
            if len(self.errors) > self.error_history_size:
                self.errors.pop(0)
                
            # Update health based on error frequency
            self.last_error_time = TimeUtils.now_as_float()  # Use TimeUtils
            
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
        
        # Update scaling factors between detection and depth image coordinates
        self.x_scale = self.depth_width / DEPTH_CONFIG["detection_resolution"]["width"]
        self.y_scale = self.depth_height / DEPTH_CONFIG["detection_resolution"]["height"]
        
        # Log camera info once (first time received)
        if not hasattr(self, 'camera_info_logged'):
            self.get_logger().info(f"Camera info received: {self.depth_width}x{self.depth_height}, "
                                  f"fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
            self.camera_info_logged = True
        
        # Update camera health if calibration is received
        if hasattr(self, 'depth_camera_health'):
            self.depth_camera_health = 1.0  # Camera is working well

    def depth_callback(self, msg):
        """
        Store the latest depth image data.
        
        Args:
            msg (Image): Depth image from camera
        """
        self.depth_image = msg
        self.depth_header = msg.header
        
        # Update camera health if depth data is being received
        if hasattr(self, 'depth_camera_health'):
            self.depth_camera_health = min(1.0, self.depth_camera_health + 0.1)  # Gradually improve health
    
    def yolo_callback(self, msg):
        """
        Handle tennis ball detections from YOLO.
        
        // ...existing docstring...
        """
        start_time = TimeUtils.now_as_float()  # Use TimeUtils
        
        self.latest_yolo_detection = msg
        
        # Process immediately for lowest latency
        # IMPORTANT: Preserve the original timestamp for synchronization
        if self.get_3d_position(msg, "YOLO"):
            self.yolo_count += 1
            process_time = (TimeUtils.now_as_float() - start_time) * 1000  # in milliseconds
            self._update_processing_stats(process_time)
    
    def hsv_callback(self, msg):
        """
        Handle tennis ball detections from HSV.
        
        // ...existing docstring...
        """
        start_time = TimeUtils.now_as_float()  # Use TimeUtils
        
        self.latest_hsv_detection = msg
        
        # Process all HSV detections (no skipping)
        # IMPORTANT: Preserve the original timestamp for synchronization
        if self.get_3d_position(msg, "HSV"):
            self.hsv_count += 1
            process_time = (TimeUtils.now_as_float() - start_time) * 1000  # in milliseconds
            self._update_processing_stats(process_time)
    
    def _update_processing_stats(self, process_time):
        """Update processing statistics for performance tracking."""
        self.processing_times.append(process_time)
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
    
    def get_3d_position(self, detection_msg, source):
        """
        Convert a 2D ball detection to a 3D position using depth data.
        
        // ...existing docstring...
        """
        # Skip processing if we're missing required data
        if self.camera_info is None or self.depth_image is None or self.fx == 0:
            if not hasattr(self, '_missing_data_logged'):
                self.log_error("Missing camera info or depth image - cannot convert to 3D", True)
                self._missing_data_logged = True
            return False
        
        try:
            # Get 2D coordinates from detection
            orig_x = detection_msg.point.x
            orig_y = detection_msg.point.y
            
            # Step 1: Scale coordinates to depth image space
            # (YOLO/HSV work in 320x320, depth might be 640x480)
            pixel_x = int(round(orig_x * self.x_scale))
            pixel_y = int(round(orig_y * self.y_scale))
            
            # Step 2: Check if coordinates are within valid bounds
            # (leaving room for radius sampling around the point)
            if (pixel_x < self.radius or pixel_x >= self.depth_width - self.radius or 
                pixel_y < self.radius or pixel_y >= self.depth_height - self.radius):
                return False
            
            # Step 3: Convert depth image to a numpy array
            depth_array = self.cv_bridge.imgmsg_to_cv2(self.depth_image)
            
            # Step 4: Extract a small region around the detection point
            # Use a smaller radius on Raspberry Pi 5 if downsampling is enabled
            radius = self.radius
            if self.enable_downsampling:
                # For every other frame, use an even smaller radius
                if hasattr(self, 'frame_counter'):
                    self.frame_counter += 1
                else:
                    self.frame_counter = 0
                
                if self.frame_counter % 2 == 0:
                    radius = max(1, radius - 1)  # Use a smaller radius
            
            min_y = max(0, pixel_y - radius)
            max_y = min(self.depth_height, pixel_y + radius + 1)
            min_x = max(0, pixel_x - radius)
            max_x = min(self.depth_width, pixel_x + radius + 1)
            
            depth_region = depth_array[min_y:max_y, min_x:max_x].astype(np.float32)
            
            # Step 5: Convert raw depth values to meters
            depth_region *= DEPTH_CONFIG["scale"]
            
            # Step 6: Filter out invalid depths
            valid_mask = (depth_region > DEPTH_CONFIG["min_depth"]) & (depth_region < DEPTH_CONFIG["max_depth"])
            if not np.any(valid_mask):
                return False
            
            # Step 7: Calculate median of valid depths
            # (median is more robust to outliers than mean)
            median_depth = float(np.median(depth_region[valid_mask]))
            
            # Step 8: Use the pinhole camera model to convert to 3D
            # These equations convert from pixel coordinates to 3D coordinates:
            # X = (u - cx) * Z / fx
            # Y = (v - cy) * Z / fy
            # Z = depth
            x = (pixel_x - self.cx) * median_depth / self.fx
            y = (pixel_y - self.cy) * median_depth / self.fy
            z = median_depth
            
            # Step 9: Create and publish 3D position message
            position_msg = PointStamped()
            
            # IMPORTANT: Use the timestamp from original detection
            # Validate timestamp before using it
            if TimeUtils.is_timestamp_valid(detection_msg.header.stamp):
                position_msg.header.stamp = detection_msg.header.stamp
                self.get_logger().debug(f"Using original timestamp from {source} for synchronization")
            else:
                position_msg.header.stamp = TimeUtils.now_as_ros_time()
                self.get_logger().debug(f"Using current time as timestamp (invalid original timestamp)")
            
            # Set the correct frame ID for consistent coordinate system
            position_msg.header.frame_id = "camera_frame"
            
            # Add sequence number for better synchronization
            if not hasattr(self, 'seq_counter'):
                self.seq_counter = 0
            self.seq_counter += 1
            position_msg.header.seq = self.seq_counter
            
            # Log that we're preserving the timestamp for synchronization
            self.get_logger().debug(
                f"Publishing 3D position with preserved timestamp from {source} for synchronization"
            )
            
            position_msg.point.x = x
            position_msg.point.y = y
            position_msg.point.z = z
            
            # Publish to source-specific topic
            if source == "YOLO":
                self.yolo_3d_publisher.publish(position_msg)
            else:  # HSV
                self.hsv_3d_publisher.publish(position_msg)
            
            # Also publish to combined topic for backward compatibility
            self.position_publisher.publish(position_msg)
            
            self.successful_conversions += 1
            
            # Log performance periodically
            self._log_performance(x, y, z, source)
            
            # Update detection health on successful conversion
            if hasattr(self, 'detection_health'):
                self.detection_health = min(1.0, self.detection_health + 0.05)
            
            return True
            
        except Exception as e:
            # Log error with reduced frequency
            if np.random.random() < 0.05:  # Log ~5% of errors
                self.log_error(f"Error in 3D conversion: {str(e)}")
            
            # Reduce detection health on errors
            if hasattr(self, 'detection_health'):
                self.detection_health = max(0.3, self.detection_health - 0.1)
                
            return False
    
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
        """Publish comprehensive system diagnostics for the diagnostics node."""
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
            import psutil
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
        except ImportError:
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
            }
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
    
    def _handle_resource_alert(self, resource_type, value):
        """Handle resource alerts by adjusting processing parameters."""
        self.get_logger().warn(f"Resource alert: {resource_type} at {value:.1f}%")
        
        if resource_type == 'cpu' and value > 90.0:
            # Log this as a warning for diagnostics
            self.log_error(f"High CPU usage ({value:.1f}%) - reducing processing load", True)
            
            # Increase downsampling to reduce CPU load
            self.radius = max(1, self.radius - 1)
            self.get_logger().warn(f"Reducing sampling radius to {self.radius} to conserve CPU")
            
            # Record adaptation for diagnostics
            if not hasattr(self, 'resource_adaptations'):
                self.resource_adaptations = []
            
            self.resource_adaptations.append({
                "timestamp": TimeUtils.now_as_float(),  # Use TimeUtils
                "resource_type": resource_type,
                "value": value,
                "action": f"Reduced sampling radius to {self.radius}"
            })
    
    def destroy_node(self):
        """Clean shutdown of the node."""
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop()
        super().destroy_node()

def main(args=None):
    """Main function to initialize and run the 3D position estimator node."""
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