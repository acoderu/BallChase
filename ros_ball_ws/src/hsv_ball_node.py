#!/usr/bin/env python3

"""
Tennis Ball Tracking Robot - HSV Ball Detector Node
==================================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving tennis ball.
The system uses multiple sensing modalities for robust detection:
- YOLO neural network detection (more accurate but computationally intensive)
- HSV color-based detection (this node - fast and efficient)
- LIDAR for depth sensing
- Depth camera for additional depth information

This Node's Purpose:
------------------
This HSV detector node uses traditional computer vision techniques to detect tennis balls
based on their distinctive yellow-green color. It processes camera images, applies color
filtering in the HSV color space, and identifies circular objects of the right size and shape.

HSV color detection offers several advantages:
- Very fast processing compared to neural network approaches
- More resilient to changes in lighting conditions than RGB
- Can be fine-tuned for specific color targets

Data Pipeline:
-------------
1. Camera images are received from '/ascamera/camera_publisher/rgb0/image'
2. Images are processed to extract tennis ball position using HSV filtering
3. Detected positions are published to '/tennis_ball/hsv/position'
4. These positions are then used by:
   - Depth camera node for 3D position estimation
   - Sensor fusion node for combining with other detection methods
   - State manager for decision making
   - PID controller for motor control
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

# Topic configuration (ensures consistency with other nodes)
TOPICS = {
    "input": {
        "camera": "/ascamera/camera_publisher/rgb0/image"
    },
    "output": {
        "position": "/tennis_ball/hsv/position"
    }
}

# Tennis ball detection configuration
BALL_CONFIG = {
    "hsv_range": {
        "lower": np.array([27, 58, 77], dtype=np.uint8),  # Lower HSV boundary for tennis ball
        "upper": np.array([45, 255, 255], dtype=np.uint8)  # Upper HSV boundary for tennis ball
    },
    "size": {
        "min_area": 100,     # Minimum area in pixels for 320x320 image
        "max_area": 1500,    # Maximum area in pixels for 320x320 image
        "ideal_area": 600    # Ideal area for confidence calculation
    },
    "shape": {
        "min_circularity": 0.5,   # Minimum circularity (0.7 is a perfect circle)
        "max_circularity": 1.3,   # Maximum circularity
        "ideal_circularity": 0.7  # Ideal circularity for confidence calculation
    }
}

class HSVTennisBallTracker(Node):
    """
    A ROS2 node that uses HSV color filtering to detect a yellow tennis ball
    in camera images and publishes its position.
    
    HSV (Hue, Saturation, Value) color space is better for color detection than RGB
    because it separates color (hue) from intensity (value) and color purity (saturation).
    This makes it more robust to lighting changes.
    
    This detector works with 320x320 images to match YOLO's input size for consistency
    across the detection pipeline.
    """
    
    def __init__(self):
        """Initialize the HSV tennis ball tracker with all required components."""
        # Initialize our ROS node
        super().__init__('hsv_tennis_ball_tracker')
        
        # Set up parameters
        self._declare_parameters()
        
        # Set up subscriptions and publishers
        self._setup_communication()
        
        # Initialize state variables
        self._init_state_variables()
        
        # Set up visualization if enabled
        self._setup_visualization()
        
        self.get_logger().info("HSV Tennis Ball Tracker has started!")
        self.get_logger().info(f"Processing images at {self.target_width}x{self.target_height} to match YOLO")
        self.get_logger().info(f"Looking for balls with area between {self.min_ball_area} and {self.max_ball_area} pixels")
        self.get_logger().info(f"HSV color range: Lower={BALL_CONFIG['hsv_range']['lower']}, Upper={BALL_CONFIG['hsv_range']['upper']}")

    def _declare_parameters(self):
        """Declare and get all node parameters."""
        self.declare_parameters(
            namespace='',
            parameters=[
                ('target_width', 320),      # Target width for processing
                ('target_height', 320),     # Target height for processing
                ('enable_visualization', False),  # Whether to show detection visualization
                ('debug_level', 1),         # 0=errors only, 1=info, 2=debug
                ('log_interval', 10),       # Log every N frames for performance stats
            ]
        )
        
        # Get parameters
        self.target_width = self.get_parameter('target_width').value
        self.target_height = self.get_parameter('target_height').value
        self.enable_visualization = self.get_parameter('enable_visualization').value
        self.debug_level = self.get_parameter('debug_level').value
        self.log_interval = self.get_parameter('log_interval').value
        
        # Get ball detection parameters from configuration
        self.lower_yellow = BALL_CONFIG['hsv_range']['lower']
        self.upper_yellow = BALL_CONFIG['hsv_range']['upper']
        self.min_ball_area = BALL_CONFIG['size']['min_area']
        self.max_ball_area = BALL_CONFIG['size']['max_area']
        self.ideal_area = BALL_CONFIG['size']['ideal_area']
        self.min_circularity = BALL_CONFIG['shape']['min_circularity']
        self.max_circularity = BALL_CONFIG['shape']['max_circularity']
        self.ideal_circularity = BALL_CONFIG['shape']['ideal_circularity']

    def _setup_communication(self):
        """Set up all subscriptions and publishers."""
        # Subscribe to the camera feed
        self.subscription = self.create_subscription(
            Image, 
            TOPICS["input"]["camera"], 
            self.image_callback, 
            10
        )
        
        # Create a publisher for ball position
        self.ball_publisher = self.create_publisher(
            PointStamped, 
            TOPICS["output"]["position"], 
            10
        )
        
        # Bridge to convert between ROS images and OpenCV images
        self.bridge = CvBridge()

    def _init_state_variables(self):
        """Initialize all state tracking variables."""
        # Performance tracking
        self.start_time = time.time()
        self.frame_count = 0
        self.no_detection_count = 0
        self.last_detection_time = None
        
        # Detection statistics
        self.detection_count = 0
        self.detection_sizes = []  # List of detected ball sizes
        self.detection_confidences = []  # List of detection confidences
        
        # Processing timing
        self.processing_times = []

    def _setup_visualization(self):
        """Set up visualization windows if enabled."""
        if self.enable_visualization:
            cv2.namedWindow("Tennis Ball Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tennis Ball Detector", 800, 600)
            self.get_logger().info("Visualization enabled - showing detection window")

    def image_callback(self, msg):
        """
        Process each incoming camera image to detect tennis balls.
        
        This method:
        1. Converts the ROS image to OpenCV format
        2. Resizes to match YOLO's input size
        3. Filters the image to isolate yellow pixels (HSV color space)
        4. Finds contours in the filtered image
        5. Evaluates each contour to find the best tennis ball candidate
        6. Publishes the position of the detected ball
        
        Args:
            msg (Image): The incoming camera image from ROS
        """
        # Start timing for performance metrics
        processing_start = time.time()
        self.frame_count += 1
        
        try:
            # Step 1: Convert ROS image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Step 2: Resize to target resolution (320x320 to match YOLO)
            original_height, original_width = frame.shape[:2]
            frame = cv2.resize(frame, (self.target_width, self.target_height))
            
            # Create a copy for visualization if enabled
            if self.enable_visualization:
                display_frame = frame.copy()
            
            # Step 3: Apply HSV color filtering to detect the tennis ball
            detected_ball = self._detect_ball_in_frame(frame, msg.header)
            
            # Step 4: Update visualization if enabled
            if self.enable_visualization:
                self._update_visualization(frame, detected_ball, processing_start)
            
            # Log performance metrics occasionally
            if self.frame_count % self.log_interval == 0:
                self._log_performance_metrics(processing_start)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _detect_ball_in_frame(self, frame, header):
        """
        Apply HSV color filtering to detect a tennis ball in the frame.
        
        Args:
            frame (numpy.ndarray): OpenCV image in BGR format
            header (Header): ROS message header from the original image
            
        Returns:
            dict: Detection information or None if no ball found
        """
        # Create a copy for visualization
        if self.enable_visualization:
            display_frame = frame.copy()
        
        # Step 1: Convert from BGR to HSV color space
        # HSV is better for color detection than RGB
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Step 2: Create a mask that only shows yellow pixels
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        
        # Step 3: Clean up the mask with morphological operations
        # Remove noise and fill small holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Save the processed mask for visualization
        if self.enable_visualization:
            display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Step 4: Find contours (outlines) of yellow objects
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Variables to track the best ball candidate
        best_contour = None
        best_radius = 0.0
        best_confidence = 0.0
        best_center = (0, 0)
        best_area = 0.0
        best_circularity = 0.0
        
        # Step 5: Check each yellow object to see if it's a tennis ball
        for cnt in contours:
            # Calculate area of the contour
            area = cv2.contourArea(cnt)
            
            # Skip tiny contours (noise)
            if area < 20:  # Adjusted for 320x320
                continue
            
            # Find the smallest circle that can enclose the contour
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            
            # Calculate "circularity" - how close to a perfect circle
            # A perfect circle has circularity close to 1.0
            circle_area = np.pi * (radius ** 2) if radius > 0 else 1
            circularity = area / circle_area
            
            # Draw all contours on the visualization
            if self.enable_visualization:
                # Draw contour in blue
                cv2.drawContours(display_frame, [cnt], -1, (255, 0, 0), 1)
                
                # Add area text
                cv2.putText(display_frame, f"{area:.0f}", (int(cx), int(cy)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Check if this contour matches our criteria for a tennis ball
            if (self.min_ball_area <= area <= self.max_ball_area and 
                self.min_circularity <= circularity <= self.max_circularity):
                
                # Calculate a confidence score based on how well it matches ideal parameters
                circularity_score = 1.0 - min(abs(circularity - self.ideal_circularity) / 
                                             self.ideal_circularity, 1.0)
                size_score = 1.0 - (abs(area - self.ideal_area) / self.ideal_area)
                
                # Combined confidence score (weighted average)
                confidence = (circularity_score * 0.7) + (size_score * 0.3)
                
                # Keep the highest confidence match
                if confidence > best_confidence:
                    best_contour = cnt
                    best_radius = radius
                    best_confidence = confidence
                    best_center = (cx, cy)
                    best_area = area
                    best_circularity = circularity
        
        # Step 6: Process the best match if found
        if best_contour is not None:
            # Unpack the center coordinates
            center_x, center_y = best_center
            
            # Log the detection
            if self.debug_level >= 1:
                self.get_logger().info(
                    f"FOUND BALL at ({center_x:.1f}, {center_y:.1f}) "
                    f"radius: {best_radius:.1f}, area: {best_area:.1f}, "
                    f"circularity: {best_circularity:.2f}, confidence: {best_confidence:.2f}"
                )
            
            # Create and publish the position message
            position_msg = PointStamped()
            position_msg.header.stamp = header.stamp
            position_msg.header.frame_id = header.frame_id
            position_msg.point.x = float(center_x)
            position_msg.point.y = float(center_y)
            position_msg.point.z = float(best_confidence)  # Use z for confidence
            
            # Publish the ball position
            self.ball_publisher.publish(position_msg)
            
            # Reset no detection counter and update statistics
            self.no_detection_count = 0
            self.detection_count += 1
            self.last_detection_time = time.time()
            
            # Store for statistics (keep last 50)
            self.detection_sizes.append(best_area)
            self.detection_confidences.append(best_confidence)
            if len(self.detection_sizes) > 50:
                self.detection_sizes.pop(0)
                self.detection_confidences.pop(0)
            
            # Return detection information
            return {
                'center': best_center,
                'radius': best_radius,
                'area': best_area,
                'circularity': best_circularity,
                'confidence': best_confidence,
                'contour': best_contour
            }
        else:
            # No ball found
            self.no_detection_count += 1
            
            # Log "no ball found" at specified intervals
            if self.no_detection_count % self.log_interval == 0:
                self._log_no_detection_info(contours)
            
            return None

    def _log_no_detection_info(self, contours):
        """
        Log detailed information about why no ball was detected.
        
        Args:
            contours (list): List of detected contours
        """
        self.get_logger().info(f"NO BALL FOUND (for {self.no_detection_count} consecutive frames)")
        
        # If there were yellow objects, explain why they weren't detected as balls
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            
            if largest_area > 20:  # Only report significant blobs (adjusted for 320x320)
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                circle_area = np.pi * (radius ** 2) if radius > 0 else 1
                circularity = largest_area / circle_area
                
                # Explain why it was rejected
                reason = "unknown reason"
                if largest_area < self.min_ball_area:
                    reason = f"too small (area={largest_area:.0f}, min={self.min_ball_area})"
                elif largest_area > self.max_ball_area:
                    reason = f"too large (area={largest_area:.0f}, max={self.max_ball_area})"
                elif circularity < self.min_circularity:
                    reason = f"not circular enough (circularity={circularity:.2f}, min={self.min_circularity})"
                elif circularity > self.max_circularity:
                    reason = f"too circular (circularity={circularity:.2f}, max={self.max_circularity})"
                
                self.get_logger().info(f"Largest yellow object rejected because: {reason}")

    def _update_visualization(self, frame, detected_ball, processing_start):
        """
        Update the visualization window with detection results.
        
        Args:
            frame (numpy.ndarray): Original frame
            detected_ball (dict): Detection information or None if no ball found
            processing_start (float): When processing started for timing
        """
        if not self.enable_visualization:
            return
            
        # Create display frame and mask
        display_frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
        # Draw the detected ball if found
        if detected_ball:
            # Unpack values
            center_x, center_y = detected_ball['center']
            radius = detected_ball['radius']
            area = detected_ball['area']
            circularity = detected_ball['circularity']
            confidence = detected_ball['confidence']
            contour = detected_ball['contour']
            
            # Draw the best contour in green
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)
            
            # Draw the circle and center point
            cv2.circle(display_frame, (int(center_x), int(center_y)), 
                      int(radius), (0, 255, 0), 2)
            cv2.circle(display_frame, (int(center_x), int(center_y)), 
                      5, (0, 0, 255), -1)
            
            # Add text with ball info
            cv2.putText(display_frame, 
                       f"Tennis Ball: ({center_x:.0f}, {center_y:.0f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, 
                       f"Area: {area:.0f} px, Circ: {circularity:.2f}, Conf: {confidence:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Combine the original frame and mask side by side
        h, w = frame.shape[:2]
        combined = np.zeros((h, w*2, 3), dtype=np.uint8)
        combined[:, :w] = display_frame
        combined[:, w:] = display_mask
        
        # Add resolution and FPS counter
        processing_time = (time.time() - processing_start) * 1000
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(combined, f"{self.target_width}x{self.target_height}", 
                   (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                   
        cv2.putText(combined, f"FPS: {fps:.1f}, Time: {processing_time:.1f}ms", 
                   (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show the combined view
        cv2.imshow("Tennis Ball Detector", combined)
        cv2.waitKey(1)

    def _log_performance_metrics(self, processing_start):
        """
        Log detailed performance metrics.
        
        Args:
            processing_start (float): When processing started for timing
        """
        # Calculate timing metrics
        processing_time = (time.time() - processing_start) * 1000
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 50:
            self.processing_times.pop(0)
            
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        # Calculate overall performance
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        detection_rate = self.detection_count / self.frame_count if self.frame_count > 0 else 0
        
        # Log basic metrics
        self.get_logger().info(
            f"PERFORMANCE: Processing: {processing_time:.1f}ms (avg: {avg_processing_time:.1f}ms), "
            f"FPS: {fps:.1f}, Detection rate: {detection_rate*100:.1f}%"
        )
        
        # Log detection statistics if we have them
        if self.detection_sizes and self.debug_level >= 2:
            avg_size = sum(self.detection_sizes) / len(self.detection_sizes)
            avg_confidence = sum(self.detection_confidences) / len(self.detection_confidences)
            self.get_logger().info(
                f"DETECTION STATS: Avg size: {avg_size:.1f}px, "
                f"Avg confidence: {avg_confidence:.2f}"
            )

def main(args=None):
    """Main function to initialize and run the HSV Tennis Ball Tracker node."""
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create our HSV tennis ball tracker node
    node = HSVTennisBallTracker()
    
    # Welcome message
    print("=================================================")
    print("Tennis Ball Tracking - HSV Ball Detector Node")
    print("=================================================")
    print("This node uses HSV color filtering to detect tennis balls.")
    print(f"Processing images at {node.target_width}x{node.target_height} to match YOLO")
    print("")
    print("Subscriptions:")
    print(f"  - Camera: {TOPICS['input']['camera']}")
    print("")
    print("Publications:")
    print(f"  - Ball position: {TOPICS['output']['position']}")
    print("")
    print("Press Ctrl+C to stop the program")
    print("=================================================")
    
    try:
        # Keep the node running until interrupted
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Stopping HSV tracker (Ctrl+C pressed)")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Clean shutdown
        if node.enable_visualization:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        print("HSV Tennis Ball Tracker has been shut down.")

if __name__ == '__main__':
    main()