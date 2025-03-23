#!/usr/bin/env python3  

import rclpy  
from rclpy.node import Node  
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge  
import cv2  
import numpy as np  

class HSVTuner(Node):  
    """  
    A ROS2 node for interactively tuning HSV parameters to detect tennis balls.  
    
    This node:  
    1. Subscribes to a camera topic  
    2. Provides trackbars for adjusting HSV color ranges  
    3. Displays the original image and the resulting mask  
    4. Prints the current HSV values to use in your detection code  
    """  
    
    def __init__(self):  
        super().__init__('hsv_tuner')  
        
        # Declare parameters  
        self.declare_parameter('camera_topic', '/ascamera/camera_publisher/rgb0/image')  
        
        # Get parameters  
        self.camera_topic = self.get_parameter('camera_topic').value  
        
        # Create subscription to camera images  
        self.subscription = self.create_subscription(  
            Image,  
            self.camera_topic,  
            self.image_callback,  
            10  
        )  
        
        # Bridge to convert between ROS images and OpenCV images  
        self.bridge = CvBridge()  
        
        # Create trackbar window  
        cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)  
        cv2.resizeWindow("Trackbars", 640, 300)  
        
        # Create trackbars for HSV tuning  
        # Tennis ball default starting values (yellow)  
        cv2.createTrackbar("LowerH", "Trackbars", 25, 179, self.nothing)  
        cv2.createTrackbar("LowerS", "Trackbars", 50, 255, self.nothing)  
        cv2.createTrackbar("LowerV", "Trackbars", 80, 255, self.nothing)  
        
        cv2.createTrackbar("UpperH", "Trackbars", 45, 179, self.nothing)  
        cv2.createTrackbar("UpperS", "Trackbars", 255, 255, self.nothing)  
        cv2.createTrackbar("UpperV", "Trackbars", 255, 255, self.nothing)  
        
        # Window for displaying HSV values  
        cv2.namedWindow("HSV Values", cv2.WINDOW_NORMAL)  
        cv2.resizeWindow("HSV Values", 400, 100)  
        
        # For displaying images  
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)  
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)  
        
        # Initialize other variables  
        self.current_frame = None  
        self.hsv_values = np.zeros((100, 400, 3), dtype=np.uint8)  
        
        self.get_logger().info(f"HSV Tuner started. Listening to {self.camera_topic}")  
        self.get_logger().info("Press 'q' in any window to quit, 's' to save current values")  
    
    def nothing(self, x):  
        """Dummy callback for trackbars"""  
        pass  
    
    def image_callback(self, msg):  
        """Process incoming camera images and update displays"""  
        try:  
            # Convert ROS Image to OpenCV format  
            self.current_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  
            
            # Clone the frame to avoid modifying the original  
            frame_display = self.current_frame.copy()  
            
            # Convert to HSV  
            hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2HSV)  
            
            # Get current trackbar positions  
            lowerH = cv2.getTrackbarPos("LowerH", "Trackbars")  
            lowerS = cv2.getTrackbarPos("LowerS", "Trackbars")  
            lowerV = cv2.getTrackbarPos("LowerV", "Trackbars")  
            
            upperH = cv2.getTrackbarPos("UpperH", "Trackbars")  
            upperS = cv2.getTrackbarPos("UpperS", "Trackbars")  
            upperV = cv2.getTrackbarPos("UpperV", "Trackbars")  
            
            # Create HSV range arrays  
            lower_bound = np.array([lowerH, lowerS, lowerV], dtype=np.uint8)  
            upper_bound = np.array([upperH, upperS, upperV], dtype=np.uint8)  
            
            # Create a mask  
            mask = cv2.inRange(hsv, lower_bound, upper_bound)  
            
            # Apply morphological operations to clean up the mask  
            mask = cv2.erode(mask, None, iterations=2)  
            mask = cv2.dilate(mask, None, iterations=2)  
            
            # Find contours in the mask  
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  
            
            # Initialize text for contour information  
            contour_info = ""  
            
            # Process contours to find potential tennis balls  
            if len(contours) > 0:  
                # Find the largest contour by area  
                c = max(contours, key=cv2.contourArea)  
                area = cv2.contourArea(c)  
                
                # Only process if the contour is large enough  
                if area > 100:  
                    # Find the minimum enclosing circle  
                    ((x, y), radius) = cv2.minEnclosingCircle(c)  
                    
                    # Calculate circularity (1.0 is a perfect circle)  
                    circle_area = np.pi * (radius ** 2)  
                    circularity = area / circle_area if circle_area > 0 else 0  
                    
                    # Draw the circle and center on the frame  
                    cv2.circle(frame_display, (int(x), int(y)), int(radius), (0, 255, 0), 2)  
                    cv2.circle(frame_display, (int(x), int(y)), 5, (0, 0, 255), -1)  
                    
                    # Add contour info to display  
                    contour_info = f"Area: {area:.1f}, Radius: {radius:.1f}, Circularity: {circularity:.2f}"  
                    cv2.putText(frame_display, contour_info, (10, 30),   
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  
                    
                    # Log coordinates of potential ball  
                    if 0.6 <= circularity <= 1.2:  # Reasonable range for tennis ball  
                        self.get_logger().info(f"Potential ball at ({x:.1f}, {y:.1f}), {contour_info}")  
            
            # Create a display to show current HSV values  
            # Fill with black  
            self.hsv_values.fill(0)  
            
            # Add HSV value text  
            cv2.putText(self.hsv_values, f"Lower HSV: [{lowerH}, {lowerS}, {lowerV}]",   
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  
            cv2.putText(self.hsv_values, f"Upper HSV: [{upperH}, {upperS}, {upperV}]",   
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  
            
            # Display the results  
            cv2.imshow("Original", frame_display)  
            cv2.imshow("Mask", mask)  
            cv2.imshow("HSV Values", self.hsv_values)  
            
            # Handle key presses  
            key = cv2.waitKey(1) & 0xFF  
            
            if key == ord('q'):  
                # Quit  
                self.get_logger().info("Quitting HSV tuner")  
                cv2.destroyAllWindows()  
                self.destroy_node()  
                rclpy.shutdown()  
                
            elif key == ord('s'):  
                # Save current values to log  
                self.get_logger().info(f"SAVED HSV VALUES:")  
                self.get_logger().info(f"lower_yellow = np.array([{lowerH}, {lowerS}, {lowerV}], dtype=np.uint8)")  
                self.get_logger().info(f"upper_yellow = np.array([{upperH}, {upperS}, {upperV}], dtype=np.uint8)")  
                
        except Exception as e:  
            self.get_logger().error(f"Error processing image: {str(e)}")  

def main(args=None):  
    # Initialize ROS  
    rclpy.init(args=args)  
    
    # Create HSV tuner node  
    hsv_tuner = HSVTuner()  
    
    try:  
        rclpy.spin(hsv_tuner)  
    except KeyboardInterrupt:  
        print("HSV Tuner stopped by user")  
    except Exception as e:  
        print(f"Unexpected error: {str(e)}")  
    finally:  
        # Clean shutdown  
        cv2.destroyAllWindows()  
        hsv_tuner.destroy_node()  
        rclpy.shutdown()  

if __name__ == '__main__':  
    main()