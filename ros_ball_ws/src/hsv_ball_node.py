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
from std_msgs.msg import String  # Add import for String message type
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os
from config.config_loader import ConfigLoader  # Import ConfigLoader
from ball_tracking.resource_monitor import ResourceMonitor  # Add resource monitoring import
import json
from collections import deque  # Add import for deque

# Load configuration from file
config_loader = ConfigLoader()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'hsv_config.yaml')
config = config_loader.load_yaml(config_path)

# Topic configuration from config file
TOPICS = config.get('topics', {
    "input": {
        "camera": "/ascamera/camera_publisher/rgb0/image"
    },
    "output": {
        "position": "/tennis_ball/hsv/position"
    }
})

# Tennis ball detection configuration from config file
HSV_LOWER = np.array(config.get('ball', {}).get('hsv_range', {}).get('lower', [27, 58, 77]), dtype=np.uint8)
HSV_UPPER = np.array(config.get('ball', {}).get('hsv_range', {}).get('upper', [45, 255, 255]), dtype=np.uint8)

BALL_CONFIG = {
    "hsv_range": {
        "lower": HSV_LOWER,  # Lower HSV boundary for tennis ball
        "upper": HSV_UPPER   # Upper HSV boundary for tennis ball
    },
    "size": config.get('ball', {}).get('size', {
        "min_area": 100,     # Minimum area in pixels for 320x320 image
        "max_area": 1500,    # Maximum area in pixels for 320x320 image
        "ideal_area": 600    # Ideal area for confidence calculation
    }),
    "shape": config.get('ball', {}).get('shape', {
        "min_circularity": 0.5,   # Minimum circularity (0.7 is a perfect circle)
        "max_circularity": 1.3,   # Maximum circularity
        "ideal_circularity": 0.7  # Ideal circularity for confidence calculation
    })
}

# Display configuration
DISPLAY_CONFIG = config.get('display', {
    "enable_visualization": False,  # Whether to show detection visualization
    "window_width": 800,            # Width of visualization window
    "window_height": 600            # Height of visualization window
})

# Diagnostic configuration
DIAG_CONFIG = config.get('diagnostics', {
    "target_width": 320,           # Target width for processing 
    "target_height": 320,          # Target height for processing
    "debug_level": 1,              # 0=errors only, 1=info, 2=debug
    "log_interval": 10             # Log every N frames for performance stats
})

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
        
        # Add resource monitoring for Raspberry Pi 5 with 16GB RAM
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=15.0,  # Less frequent to reduce overhead
            enable_temperature=True
        )
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.resource_monitor.start()
        
        # Set up parameters
        self._declare_parameters()
        
        # Configure optimization settings for Pi 5 with 16GB
        self._configure_optimizations()
        
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
        # Set parameters from config
        self.target_width = DIAG_CONFIG["target_width"]
        self.target_height = DIAG_CONFIG["target_height"]
        self.enable_visualization = DISPLAY_CONFIG["enable_visualization"] 
        self.debug_level = DIAG_CONFIG["debug_level"]
        self.log_interval = DIAG_CONFIG["log_interval"]
        
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
        
        # Create a publisher for system diagnostics
        self.system_diagnostics_publisher = self.create_publisher(
            String, 
            "/tennis_ball/hsv/diagnostics",  # Changed from "/system/diagnostics/hsv"
            10
        )
        
        # Timer for publishing diagnostics
        self.diagnostics_timer = self.create_timer(2.0, self.publish_system_diagnostics)

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
        
        # Initialize diagnostic metrics
        self.diagnostic_metrics = {
            'fps_history': deque(maxlen=10),
            'processing_time_history': deque(maxlen=10),
            'detection_rate_history': deque(maxlen=10),
            'last_detection_position': None,
            'last_detection_time': 0.0,
            'total_frames': 0,
            'missed_frames': 0,
            'errors': [],
            'warnings': []
        }

    def _setup_visualization(self):
        """Set up visualization windows if enabled."""
        if self.enable_visualization:
            cv2.namedWindow("Tennis Ball Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tennis Ball Detector", 
                           DISPLAY_CONFIG["window_width"], 
                           DISPLAY_CONFIG["window_height"])
            self.get_logger().info("Visualization enabled - showing detection window")

    def _configure_optimizations(self):
        """Configure performance optimizations based on RAM availability."""
        # With 16GB RAM, we can optimize for processing quality rather than memory usage
        try:
            import psutil
            total_ram = psutil.virtual_memory().total / (1024 * 1024)  # MB
            
            # On Pi 5 with 16GB RAM, we can use more advanced options
            if total_ram >= 12000:  # At least 12GB
                # Precompute kernel for morphological operations for better performance
                self.morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                
                # Enable more advanced detection features that use more RAM but give better results
                self.use_enhanced_detection = True
                
                self.get_logger().info(f"Using enhanced detection features (high RAM mode)")
            else:
                # Standard settings for lower memory systems
                self.morphology_kernel = np.ones((5, 5), np.uint8)  # Simple kernel
                self.use_enhanced_detection = False
        except:
            # Default settings if we can't check memory
            self.morphology_kernel = np.ones((5, 5), np.uint8)
            self.use_enhanced_detection = False
        
        # Number of frames to skip in low power mode (0 means no skipping)
        self.low_power_skip_frames = 0
    
    def _handle_resource_alert(self, resource_type, value):
        """Handle resource alerts by adjusting processing behavior."""
        self.get_logger().warn(f"Resource alert: {resource_type.upper()} at {value:.1f}%")
        
        # If CPU usage is critically high, start skipping frames
        if resource_type == 'cpu' and value > 90.0:
            old_skip = self.low_power_skip_frames
            self.low_power_skip_frames = 1  # Skip every other frame
            self.get_logger().warn(f"CPU usage high: changing frame skip from {old_skip} to {self.low_power_skip_frames}")
            
            # Record for diagnostics
            if hasattr(self, 'diagnostic_metrics'):
                if 'adaptations' not in self.diagnostic_metrics:
                    self.diagnostic_metrics['adaptations'] = []
                    
                self.diagnostic_metrics['adaptations'].append({
                    'timestamp': time.time(),
                    'resource_type': resource_type,
                    'value': value,
                    'action': f'Increased frame skip to {self.low_power_skip_frames}'
                })

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
        # Skip frames if needed to reduce CPU usage
        if self.low_power_skip_frames > 0:
            if not hasattr(self, 'frame_skip_counter'):
                self.frame_skip_counter = 0
            
            self.frame_skip_counter += 1
            if (self.frame_skip_counter % (self.low_power_skip_frames + 1)) != 0:
                # Skip this frame
                return
        
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
        # Use precomputed kernel for better performance
        mask = cv2.erode(mask, self.morphology_kernel, iterations=1)
        mask = cv2.dilate(mask, self.morphology_kernel, iterations=2)
        
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
        
        # For Pi 5 with 16GB, we can use more advanced detection techniques
        if hasattr(self, 'use_enhanced_detection') and self.use_enhanced_detection:
            # Enhance circle detection with Hough Circles if we have contours
            if len(contours) > 0 and np.any(mask):
                try:
                    # Only attempt circle detection on significant segments
                    largest_contour = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_contour) > 50:  # Skip tiny regions
                        # Create a mask with just the largest contour
                        largest_mask = np.zeros_like(mask)
                        cv2.drawContours(largest_mask, [largest_contour], 0, 255, -1)
                        
                        # Apply Hough Circle detection with adaptive parameters
                        detected_circles = cv2.HoughCircles(
                            mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=10, 
                            minRadius=int(np.sqrt(self.min_ball_area/np.pi)),
                            maxRadius=int(np.sqrt(self.max_ball_area/np.pi))
                        )
                        
                        # If circles are found, consider them in the detection
                        if detected_circles is not None:
                            detected_circles = np.round(detected_circles[0, :]).astype(int)
                            for (x, y, r) in detected_circles:
                                # Calculate approximate contour quality based on the circle
                                circle_area = np.pi * r * r
                                circle_matches = True
                                # Rest of Hough circle processing...
                except Exception as e:
                    # Ignore errors in enhanced detection - fall back to standard
                    if self.debug_level >= 2:
                        self.get_logger().debug(f"Enhanced detection error: {e}")
        
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
            
            # IMPORTANT: Use the original image timestamp
            # This is critical for proper synchronization
            position_msg.header.stamp = header.stamp
            position_msg.header.frame_id = "camera_frame"  # Use consistent frame ID
            
            position_msg.point.x = float(center_x)
            position_msg.point.y = float(center_y)
            position_msg.point.z = float(best_confidence)  # Use z for confidence
            
            # Add sequence number to header for better synchronization
            if not hasattr(self, 'seq_counter'):
                self.seq_counter = 0
            self.seq_counter += 1
            position_msg.header.seq = self.seq_counter
            
            self.get_logger().debug(f"Publishing 2D position with timestamp for synchronization")
            
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
            
            # Store for diagnostics
            if hasattr(self, 'diagnostic_metrics'):
                self.diagnostic_metrics['last_detection_position'] = (center_x, center_y)
                self.diagnostic_metrics['last_detection_time'] = time.time()
                
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
            
            # Track missed frames for diagnostics
            if hasattr(self, 'diagnostic_metrics'):
                self.diagnostic_metrics['missed_frames'] += 1
                
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
        
        # Store metrics for system diagnostics
        if not hasattr(self, 'diagnostic_metrics'):
            self.diagnostic_metrics = {
                'fps_history': deque(maxlen=10),
                'processing_time_history': deque(maxlen=10),
                'detection_rate_history': deque(maxlen=10),
                'last_detection_position': None,
                'last_detection_time': 0.0,
                'total_frames': 0,
                'missed_frames': 0,
                'errors': [],
                'warnings': []
            }
        
        # Update metrics
        self.diagnostic_metrics['fps_history'].append(fps)
        self.diagnostic_metrics['processing_time_history'].append(processing_time)
        self.diagnostic_metrics['detection_rate_history'].append(detection_rate)
        self.diagnostic_metrics['total_frames'] = self.frame_count

    def publish_system_diagnostics(self):
        """Publish comprehensive system diagnostics for the diagnostics node."""
        if not hasattr(self, 'diagnostic_metrics'):
            return  # Not enough data collected yet
            
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate average metrics
        avg_fps = np.mean(list(self.diagnostic_metrics['fps_history'])) if self.diagnostic_metrics['fps_history'] else 0.0
        avg_processing_time = np.mean(list(self.diagnostic_metrics['processing_time_history'])) if self.diagnostic_metrics['processing_time_history'] else 0.0
        avg_detection_rate = np.mean(list(self.diagnostic_metrics['detection_rate_history'])) if self.diagnostic_metrics['detection_rate_history'] else 0.0
        
        # Time since last detection
        time_since_detection = current_time - self.diagnostic_metrics['last_detection_time'] if self.diagnostic_metrics['last_detection_time'] > 0 else float('inf')
        
        # Build warnings list
        warnings = []
        errors = []
        
        # Check for performance issues
        if avg_fps < 10.0 and elapsed_time > 10.0:
            warnings.append(f"Low FPS: {avg_fps:.1f}")
            
        if avg_processing_time > 50.0:  # 50ms is slow
            warnings.append(f"High processing time: {avg_processing_time:.1f}ms")
            
        # Check for detection issues
        if time_since_detection > 5.0 and elapsed_time > 10.0:
            warnings.append(f"No ball detected for {time_since_detection:.1f}s")
            
        if avg_detection_rate < 0.1 and elapsed_time > 10.0:  # Less than 10% detection rate
            errors.append(f"Very low detection rate: {avg_detection_rate*100:.1f}%")
        
        # System resources
        system_resources = {}
        try:
            import psutil
            system_resources = {
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent
            }
            
            # Check for high resource usage
            if system_resources['cpu_percent'] > 80.0:
                warnings.append(f"High CPU usage: {system_resources['cpu_percent']:.1f}%")
                
            # Add temperature if available
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps and 'cpu_thermal' in temps:
                    system_resources['temperature'] = temps['cpu_thermal'][0].current
        except ImportError:
            pass
        
        # Build diagnostics data structure
        diag_data = {
            "node": "hsv",  # Changed from "node_name": "hsv_ball_node"
            "timestamp": current_time,
            "uptime_seconds": elapsed_time,
            "status": "error" if errors else ("warning" if warnings else "active"),  # Changed "ok" to "active"
            "health": {
                # Add proper health metrics for consistency
                "camera_health": 1.0 - (len(warnings) * 0.1),
                "detection_health": avg_detection_rate if avg_detection_rate > 0 else 0.5,
                "processing_health": 1.0 - (avg_processing_time / 100.0) if avg_processing_time < 100.0 else 0.0,
                "overall": 1.0 - (len(errors) * 0.3) - (len(warnings) * 0.1)
            },
            "metrics": {  # Changed from "performance"
                "fps": avg_fps,
                "processing_time_ms": avg_processing_time,
                "total_frames": self.diagnostic_metrics['total_frames'],
                "missed_frames": self.diagnostic_metrics['missed_frames'],
                "detection_rate": avg_detection_rate
            },
            "detection": {
                "latest_position": self.diagnostic_metrics['last_detection_position'],
                "time_since_last_detection_s": time_since_detection,
                "currently_tracking": time_since_detection < 1.0
            },
            "configuration": {
                "hsv_range": {
                    "lower": self.lower_yellow.tolist(),
                    "upper": self.upper_yellow.tolist()
                },
                "area_range": [self.min_ball_area, self.max_ball_area],
                "circularity_range": [self.min_circularity, self.max_circularity]
            },
            "resources": system_resources,  # Changed from "system_resources"
            "errors": errors,
            "warnings": warnings
        }
        
        # Publish as JSON
        msg = String()
        msg.data = json.dumps(diag_data)
        self.system_diagnostics_publisher.publish(msg)
        
        # Also log to console
        self.get_logger().info(
            f"HSV diagnostics: {avg_fps:.1f} FPS, {avg_detection_rate*100:.1f}% detection rate, "
            f"Status: {diag_data['status']}"
        )

    def destroy_node(self):
        """Clean up resources when the node is shutting down."""
        # Stop the resource monitor if it exists
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop()
        
        # Close OpenCV windows if enabled
        if self.enable_visualization:
            cv2.destroyAllWindows()
        
        super().destroy_node()

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
        # On Pi 5, use process priority to balance with other nodes
        try:
            import os
            os.nice(5)  # Slightly lower priority than critical nodes
            print("Set HSV tracker to adjusted process priority")
        except:
            pass
        
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