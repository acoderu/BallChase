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

# Configuration constants
DEPTH_CONFIG = {
    "scale": 0.001,           # Depth scale factor (converts raw depth to meters)
    "min_depth": 0.1,         # Minimum valid depth in meters
    "max_depth": 8.0,         # Maximum valid depth in meters
    "radius": 3,              # Radius around detection point to sample depth values
    "detection_resolution": {  # Resolution of detection images (YOLO/HSV)
        "width": 320,
        "height": 320
    }
}

# Topic configuration (ensures consistency with other nodes)
TOPICS = {
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
}


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
        
        # Set up callback group for efficient multi-threading
        self._setup_callback_group()
        
        # Initialize camera and detection parameters
        self._init_camera_parameters()
        
        # Set up subscriptions to receive data
        self._setup_subscriptions()
        
        # Set up publishers to send out 3D positions
        self._setup_publishers()
        
        # Performance tracking variables
        self._init_performance_tracking()
        
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
    
    def _init_performance_tracking(self):
        """Initialize performance tracking variables."""
        self.start_time = time.time()
        self.yolo_count = 0
        self.hsv_count = 0
        self.successful_conversions = 0
        self.last_fps_log_time = time.time()
        self.processing_times = []
    
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
            self.get_logger().info(f"Received camera calibration with:")
            self.get_logger().info(f"  - Resolution: {self.depth_width}x{self.depth_height}")
            self.get_logger().info(f"  - Focal length: fx={self.fx:.1f}, fy={self.fy:.1f}")
            self.get_logger().info(f"  - Optical center: cx={self.cx:.1f}, cy={self.cy:.1f}")
            self.get_logger().info(f"  - Scaling factors: x={self.x_scale:.3f}, y={self.y_scale:.3f}")
            self.camera_info_logged = True
    
    def depth_callback(self, msg):
        """
        Store the latest depth image data.
        
        Args:
            msg (Image): Depth image from camera
        """
        self.depth_image = msg
        self.depth_header = msg.header
    
    def yolo_callback(self, msg):
        """
        Handle tennis ball detections from YOLO.
        
        Args:
            msg (PointStamped): 2D position of ball detected by YOLO
        """
        start_time = time.time()
        
        self.latest_yolo_detection = msg
        
        # Process immediately for lowest latency
        if self.get_3d_position(msg, "YOLO"):
            self.yolo_count += 1
            process_time = (time.time() - start_time) * 1000  # in milliseconds
            self._update_processing_stats(process_time)
    
    def hsv_callback(self, msg):
        """
        Handle tennis ball detections from HSV.
        
        Args:
            msg (PointStamped): 2D position of ball detected by HSV
        """
        start_time = time.time()
        
        self.latest_hsv_detection = msg
        
        # Process all HSV detections (no skipping)
        if self.get_3d_position(msg, "HSV"):
            self.hsv_count += 1
            process_time = (time.time() - start_time) * 1000  # in milliseconds
            self._update_processing_stats(process_time)
    
    def _update_processing_stats(self, process_time):
        """
        Update processing time statistics.
        
        Args:
            process_time (float): Processing time in milliseconds
        """
        self.processing_times.append(process_time)
        if len(self.processing_times) > 50:  # Keep a reasonable history
            self.processing_times.pop(0)
    
    def get_3d_position(self, detection_msg, source):
        """
        Convert a 2D ball detection to a 3D position using depth data.
        
        This is the core function that:
        1. Takes a 2D position from a camera-based detector
        2. Scales it to match the depth camera's resolution
        3. Samples depth values around that position
        4. Uses camera intrinsics to calculate the 3D position
        
        Args:
            detection_msg (PointStamped): 2D ball position from detector
            source (str): Which detector it came from ("YOLO" or "HSV")
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Skip processing if we're missing required data
        if self.camera_info is None or self.depth_image is None or self.fx == 0:
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
            min_y = pixel_y - self.radius
            max_y = pixel_y + self.radius + 1
            min_x = pixel_x - self.radius
            max_x = pixel_x + self.radius + 1
            
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
            position_msg.header = self.depth_header
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
            
            return True
            
        except Exception as e:
            # Only log occasional errors to reduce overhead
            if np.random.random() < 0.05:  # Log ~5% of errors
                self.get_logger().warn(f"Error in 3D conversion: {str(e)}")
            return False
    
    def _log_performance(self, x, y, z, source):
        """
        Log performance metrics and position data.
        
        Args:
            x, y, z (float): 3D position coordinates
            source (str): Detection source ("YOLO" or "HSV")
        """
        # Log performance less frequently to reduce CPU overhead
        current_time = time.time()
        if current_time - self.last_fps_log_time >= 5.0:  # Every 5 seconds
            elapsed = current_time - self.start_time
            total_fps = self.successful_conversions / elapsed
            yolo_fps = self.yolo_count / elapsed
            hsv_fps = self.hsv_count / elapsed
            avg_processing = np.mean(self.processing_times) if self.processing_times else 0
            
            self.get_logger().info(
                f"PERFORMANCE: Total: {total_fps:.1f}fps, YOLO: {yolo_fps:.1f}fps, "
                f"HSV: {hsv_fps:.1f}fps, Processing time: {avg_processing:.1f}ms"
            )
            
            self.get_logger().info(
                f"Published 3D position from {source}: ({x:.3f}, {y:.3f}, {z:.3f}) meters"
            )
            
            self.last_fps_log_time = current_time


def main(args=None):
    """Main function to initialize and run the 3D position estimator node."""
    rclpy.init(args=args)
    node = TennisBall3DPositionEstimator()
    
    # Use multiple threads for better performance with parallel processing
    executor = MultiThreadedExecutor(num_threads=6)
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