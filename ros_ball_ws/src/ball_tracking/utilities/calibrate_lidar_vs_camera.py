#!/usr/bin/env python3

"""
LIDAR-Camera Calibration Tool for Basketball Tracking Robot
==========================================================

This script helps calibrate the transformation between LIDAR and camera coordinate systems
for a basketball tracking robot. It's been simplified and optimized for:

1. A basketball instead of a tennis ball (larger, more reliable detection)
2. Flat LIDAR orientation (not tilted)
3. Raspberry Pi 5 performance considerations

Features:
- Interactive calibration procedure
- Physical constraint validation (ensuring transform respects actual hardware setup)
- Quality assessment and guidance
- ROS2 TF2 code generation

Usage:
------
1. Place the basketball at various positions in view of both sensors
2. For each position, make sure both sensors detect the ball
3. Capture point pairs to build the calibration dataset
4. Calculate and validate the transformation

Commands:
- 'c': Capture current point pair
- 'l': List captured points
- 'r': Remove last point pair
- 'x': Calculate transformation (standard method)
- 'p': Calculate with physical constraints (recommended)
- 'd': Analyze point distribution
- 'a': Assess calibration quality
- 't': Test transformation on current points
- 'q': Quit
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import String
# Import QoS-related classes
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, HistoryPolicy
import numpy as np
from scipy.spatial.transform import Rotation
import tf2_ros
import time
import threading
import math
import json


class LidarCameraCalibrator(Node):
    """
    Node for calibrating the transformation between LIDAR and camera coordinate systems
    for basketball tracking.
    """
    
    def __init__(self):
        """Initialize the calibrator node."""
        super().__init__('lidar_camera_calibrator')
        
        # Storage for calibration points
        self.camera_points = []  # Points from camera
        self.lidar_points = []   # Corresponding points from LIDAR
        
        # Latest received points (for capturing)
        self.latest_camera_point = None
        self.latest_lidar_point = None
        self.camera_time = 0
        self.lidar_time = 0
        
        # Quality and stability tracking for LIDAR
        self.latest_lidar_quality = 0.0
        self.previous_lidar_points = []
        self.lidar_quality_threshold = 0.4  # Reduced threshold for reliable calibration 
        self.position_stability_threshold = 0.3  # Increased threshold for movement between readings (meters)
        self.consecutive_filtered_points = 0  # Track consecutive filtered points
        self.max_valid_data_age = 2.0  # Maximum age for valid data in seconds
        
        # Physical measurements (in meters)
        self.lidar_height = 0.1524      # LIDAR height from ground (6 inches)
        self.camera_height = 0.1016     # Camera height from ground (4 inches)
        self.ball_radius = 0.127        # Basketball radius (5 inches)
        self.ball_center_height = 0.127 # Basketball center height from ground (5 inches)
        
        # Create lock for thread safety
        self.lock = threading.Lock()
        
        # Create a sensor-compatible QoS profile
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Create subscribers with appropriate QoS settings
        self.camera_sub = self.create_subscription(
            PointStamped,
            '/basketball/yolo/position_3d',  # Position from YOLO/depth camera
            self.camera_callback,
            qos_profile=sensor_qos  # Use sensor-compatible QoS
        )
        
        self.lidar_sub = self.create_subscription(
            PointStamped,
            '/basketball/lidar/position',  # Position from LIDAR
            self.lidar_callback,
            qos_profile=sensor_qos  # Use sensor-compatible QoS
        )
        
        # Create TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Create timer for periodic updates
        self.timer = self.create_timer(0.5, self.timer_callback)
        
        # Transformation parameters
        self.translation = np.zeros(3)
        self.rotation_matrix = np.eye(3)
        self.has_transform = False
        
        # Print welcome message
        self.print_welcome_message()
        
        # Start input thread
        self.running = True
        self.input_thread = threading.Thread(target=self.input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()
    
    def print_welcome_message(self):
        """Print welcome message with instructions."""
        self.get_logger().info("====================================================")
        self.get_logger().info("   LIDAR-Camera Calibration Tool for Basketball")
        self.get_logger().info("====================================================")
        self.get_logger().info("This tool helps align LIDAR and camera coordinate frames")
        self.get_logger().info("")
        self.get_logger().info("IMPORTANT - READ BEFORE CALIBRATING:")
        self.get_logger().info("1. Make sure lidar_node.py is running with debug points")
        self.get_logger().info("   publishing to a separate topic")
        self.get_logger().info("2. Move the basketball to generate new detections")
        self.get_logger().info("3. Watch the logs - capture within 2 seconds of")
        self.get_logger().info("   seeing a valid detection")
        self.get_logger().info("")
        self.get_logger().info("Commands:")
        self.get_logger().info("  'c' - Capture current point pair")
        self.get_logger().info("  'l' - List captured points")
        self.get_logger().info("  'r' - Remove last point pair")
        self.get_logger().info("  'x' - Calculate transformation (standard)")
        self.get_logger().info("  'p' - Calculate with physical constraints (recommended)")
        self.get_logger().info("  'd' - Analyze point distribution")
        self.get_logger().info("  'a' - Assess calibration quality")
        self.get_logger().info("  't' - Test transformation on current points")
        self.get_logger().info("  'q' - Quit")
        self.get_logger().info("====================================================")
        self.get_logger().info("Tips for Good Calibration:")
        self.get_logger().info("- Ensure your lidar_node and camera are working properly")
        self.get_logger().info("- Collect at least 5-6 point pairs at different positions")
        self.get_logger().info("- Include points at different distances (0.8-2m)")
        self.get_logger().info("- If data is too old (>2s), move the ball to get new data")
        self.get_logger().info("- If capture fails, check if camera and LIDAR detect the same ball")
        self.get_logger().info("- Watch for 'Large distance mismatch' warnings")
        self.get_logger().info("- Remove bad points with 'r' if needed")
        self.get_logger().info("====================================================")
    
    def camera_callback(self, msg):
        """Process camera position messages."""
        # Debug print to confirm callback is being triggered
        #self.get_logger().info(f"Camera callback triggered: ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})")
        
        with self.lock:
            self.latest_camera_point = np.array([
                msg.point.x,
                msg.point.y,
                msg.point.z
            ])
            self.camera_time = time.time()
    
    def lidar_callback(self, msg):
        """
        Process LIDAR position messages with quality filtering and stability checks.
        """
        # Extract position
        position = np.array([
            msg.point.x,
            msg.point.y, 
            msg.point.z
        ])
        
        # Calculate distance
        distance = np.sqrt(position[0]**2 + position[1]**2)
        
        # Debug print to confirm callback is being triggered
        #self.get_logger().info(f"LIDAR callback triggered: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
        
        # SIMPLIFIED FILTERING - be much more lenient about accepting points
        is_filtered = False
        
        # Only filter extreme cases - points near -1.0y that are clear debug points
        if abs(position[1] + 1.0) < 0.05 and abs(distance - 1.0) < 0.05:
            self.get_logger().warn(f"Filtered out likely debug point: exact -1.0y, 1.0m distance")
            is_filtered = True
        
        # Basic stability check - only if we have previous points and only for very large jumps
        if len(self.previous_lidar_points) > 0:
            # Get last position
            last_position = self.previous_lidar_points[-1]
            position_change = np.linalg.norm(position - last_position)
            
            # Only filter extremely large changes (much more permissive)
            if position_change > 1.0:  # Changed from 0.5 to 1.0
                self.get_logger().warn(f"Filtered out extremely unstable point: {position_change:.2f}m position change")
                is_filtered = True
                self.consecutive_filtered_points += 1
            else:
                self.consecutive_filtered_points = 0
        
        # If we've filtered too many points in a row, reset our history to stop rejecting everything
        if self.consecutive_filtered_points > 5:
            self.get_logger().warn("Too many filtered points in a row - resetting position history")
            self.previous_lidar_points = []
            self.consecutive_filtered_points = 0
        
        # Skip filtered points
        if is_filtered:
            return
            
        # Store position for point stability tracking - max 3 points to avoid old history problems
        if len(self.previous_lidar_points) >= 3:
            self.previous_lidar_points.pop(0)  # Remove oldest
        self.previous_lidar_points.append(position)
            
        # If passed all filters, use this point
        with self.lock:
            self.latest_lidar_point = position
            self.lidar_time = time.time()
    
    def timer_callback(self):
        """Periodic updates including TF broadcasting."""
        # If we have a transform, publish it
        if self.has_transform:
            self.publish_transform()
        
        # Check data freshness less frequently
        current_time = time.time()
        
        # Only check every 1 second to reduce log spam
        if not hasattr(self, 'last_data_check') or (current_time - self.last_data_check) > 1.0:
            self.last_data_check = current_time
            
            with self.lock:
                camera_age = current_time - self.camera_time if self.camera_time > 0 else float('inf')
                lidar_age = current_time - self.lidar_time if self.lidar_time > 0 else float('inf')
            
            # Track if data was ready
            is_ready_now = camera_age <= self.max_valid_data_age and lidar_age <= self.max_valid_data_age
            
            # Only log when state changes or on a timer
            if not hasattr(self, 'was_ready_last_check'):
                self.was_ready_last_check = False
                
            if is_ready_now:
                # Show ready message when state changes or every 3 seconds
                if not self.was_ready_last_check or int(current_time * 10) % 30 == 0:
                    self.get_logger().info(f"✓ READY TO CAPTURE - Data fresh: camera={camera_age:.1f}s, lidar={lidar_age:.1f}s")
            else:
                # Only show waiting message occasionally (every 5 seconds)
                if not hasattr(self, 'last_waiting_message') or (current_time - self.last_waiting_message) > 5.0:
                    self.get_logger().info("Waiting for fresh sensor data... Move the basketball to trigger detection.")
                    self.last_waiting_message = current_time
            
            # Check position stability but only when data is fresh
            if is_ready_now and len(self.previous_lidar_points) >= 2:
                stability = self.check_position_stability()
                if stability >= self.position_stability_threshold:
                    self.get_logger().warn(f"LIDAR position unstable (variance: {stability:.3f}m)")
            
            # Update state
            self.was_ready_last_check = is_ready_now
    
    def check_position_stability(self):
        """
        Check the stability of recent LIDAR readings.
        
        Returns:
            float: Average distance between consecutive points (lower is more stable)
        """
        if len(self.previous_lidar_points) < 2:
            return 0.0
            
        # Use a better stability metric - average deviation from mean position
        # This is more robust than just looking at consecutive points
        if len(self.previous_lidar_points) >= 3:
            # Calculate mean position
            points_array = np.array(self.previous_lidar_points)
            mean_position = np.mean(points_array, axis=0)
            
            # Calculate average deviation from mean
            total_deviation = 0.0
            for point in self.previous_lidar_points:
                deviation = np.linalg.norm(point - mean_position)
                total_deviation += deviation
                
            return total_deviation / len(self.previous_lidar_points)
        else:
            # Fall back to original method for just 2 points
            total_distance = 0.0
            for i in range(1, len(self.previous_lidar_points)):
                distance = np.linalg.norm(
                    self.previous_lidar_points[i] - self.previous_lidar_points[i-1]
                )
                total_distance += distance
                
            return total_distance / (len(self.previous_lidar_points) - 1)
    
    def input_loop(self):
        """Handle user input for calibration process."""
        while self.running:
            cmd = input("> ").strip().lower()
            
            if cmd == 'c':  # Capture
                self.capture_point_pair()
            elif cmd == 'l':  # List
                self.list_points()
            elif cmd == 'r':  # Remove
                self.remove_last_point()
            elif cmd == 'f':  # Filter outliers automatically
                self.filter_outliers()
            elif cmd == 'x':  # Calculate
                self.calculate_transformation()
            elif cmd == 'p':  # Physical constraints
                self.calibrate_with_physical_constraints()
            elif cmd == 'a':  # Assess
                self.assess_calibration_quality()
            elif cmd == 'd':  # Distribution
                self.analyze_point_distribution()
            elif cmd == 't':  # Test
                self.test_transformation()
            elif cmd == 'q':  # Quit
                self.running = False
                self.get_logger().info("Exiting...")
                break
            else:
                self.get_logger().info("Unknown command")
                self.get_logger().info("Commands: c=capture, l=list, r=remove, f=filter outliers, x=calculate, p=physical, a=assess, t=test, q=quit")
    
    def capture_point_pair(self):
        """
        Capture a pair of corresponding points from both sensors,
        with stability and quality checking.
        """
        with self.lock:
            camera_point = self.latest_camera_point
            lidar_point = self.latest_lidar_point
            camera_age = time.time() - self.camera_time if self.camera_time > 0 else float('inf')
            lidar_age = time.time() - self.lidar_time if self.lidar_time > 0 else float('inf')
        
        # Check if we have recent data
        if camera_point is None or lidar_point is None:
            self.get_logger().error("Missing data from one or both sensors")
            return
        
        # Check if data is too old (important to prevent using very stale data)
        if camera_age > self.max_valid_data_age or lidar_age > self.max_valid_data_age:
            self.get_logger().error(f"Data too old to use: camera={camera_age:.2f}s, lidar={lidar_age:.2f}s")
            self.get_logger().error(f"Maximum valid age is {self.max_valid_data_age}s")
            self.get_logger().error("Try moving the basketball to trigger new detections")
            return
        
        # Warn about slightly stale data but still allow it
        if camera_age > 0.5 or lidar_age > 0.5:
            self.get_logger().warn(f"Using moderately stale data: camera={camera_age:.2f}s, lidar={lidar_age:.2f}s")
        
        # Calculate distance to check for reasonable correspondence
        lidar_distance = np.sqrt(lidar_point[0]**2 + lidar_point[1]**2)
        camera_distance = np.sqrt(camera_point[0]**2 + camera_point[1]**2)
        
        # Check for major distance mismatch (likely indicates something wrong)
        distance_ratio = abs(lidar_distance - camera_distance) / max(camera_distance, 0.1)
        if distance_ratio > 0.8:  # If distances differ by more than 80%
            self.get_logger().error(f"Large distance mismatch: camera={camera_distance:.2f}m, lidar={lidar_distance:.2f}m")
            self.get_logger().error("This suggests misaligned detections or debug points")
            self.get_logger().error("Try again with the ball in a different position")
            return
        
        # Less strict position stability check
        if len(self.previous_lidar_points) >= 3:
            stability = self.check_position_stability()
            if stability > self.position_stability_threshold:
                self.get_logger().warn(f"LIDAR position somewhat unstable ({stability:.3f}m > {self.position_stability_threshold:.3f}m)")
                self.get_logger().warn("Consider trying again for better results")
                # But still continue with capture
        
        # Add the points to calibration sets
        self.camera_points.append(np.copy(camera_point))
        self.lidar_points.append(np.copy(lidar_point))
        
        # Print capture info
        self.get_logger().info(f"Captured point pair #{len(self.camera_points)}:")
        self.get_logger().info(f"  Camera: ({camera_point[0]:.3f}, {camera_point[1]:.3f}, {camera_point[2]:.3f}), dist={camera_distance:.2f}m")
        self.get_logger().info(f"  LIDAR:  ({lidar_point[0]:.3f}, {lidar_point[1]:.3f}, {lidar_point[2]:.3f}), dist={lidar_distance:.2f}m")
        
        # Provide guidance
        if lidar_distance < 0.8:
            self.get_logger().warn("Point is close to LIDAR. Reliability may be low.")
        elif 1.0 <= lidar_distance <= 2.0:
            self.get_logger().info("Point is in optimal LIDAR range (1-2m). Good calibration point.")
        
        # After 3+ points, analyze distribution
        if len(self.lidar_points) >= 3:
            self.analyze_point_distribution(silent=True)
    
    def list_points(self):
        """List all captured point pairs."""
        if not self.camera_points:
            self.get_logger().info("No points captured yet")
            return
        
        self.get_logger().info(f"Captured {len(self.camera_points)} point pairs:")
        for i, (camera_point, lidar_point) in enumerate(zip(self.camera_points, self.lidar_points)):
            # Calculate distances
            lidar_distance = np.sqrt(lidar_point[0]**2 + lidar_point[1]**2)
            camera_distance = np.sqrt(camera_point[0]**2 + camera_point[1]**2)
            
            self.get_logger().info(f"Pair #{i+1}:")
            self.get_logger().info(f"  Camera: ({camera_point[0]:.3f}, {camera_point[1]:.3f}, {camera_point[2]:.3f}), dist={camera_distance:.2f}m")
            self.get_logger().info(f"  LIDAR:  ({lidar_point[0]:.3f}, {lidar_point[1]:.3f}, {lidar_point[2]:.3f}), dist={lidar_distance:.2f}m")
    
    def remove_last_point(self):
        """Remove the last captured point pair."""
        if not self.camera_points:
            self.get_logger().info("No points to remove")
            return
        
        self.camera_points.pop()
        self.lidar_points.pop()
        self.get_logger().info(f"Removed last point pair. {len(self.camera_points)} pairs remaining.")
    
    def calculate_transformation(self):
        """Calculate the transformation from LIDAR to camera frame."""
        if len(self.camera_points) < 3:
            self.get_logger().error("Need at least 3 point pairs for calibration!")
            return
        
        # Convert to numpy arrays
        camera_array = np.array(self.camera_points)
        lidar_array = np.array(self.lidar_points)
        
        try:
            # Compute centroids
            camera_centroid = np.mean(camera_array, axis=0)
            lidar_centroid = np.mean(lidar_array, axis=0)
            
            # Center the points
            camera_centered = camera_array - camera_centroid
            lidar_centered = lidar_array - lidar_centroid
            
            # Compute cross-covariance matrix
            H = lidar_centered.T @ camera_centered
            
            # SVD for rotation
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # Fix reflection if needed
            if np.linalg.det(R) < 0:
                self.get_logger().warn("Reflection detected in rotation. Fixing...")
                V = Vt.T
                V[:, -1] *= -1
                R = V @ U.T
            
            # Compute translation
            t = camera_centroid - (R @ lidar_centroid)
            
            # Store transformation
            self.rotation_matrix = R
            self.translation = t
            self.has_transform = True
            
            # Calculate and display error
            error = self.calculate_error()
            
            # Convert to Euler angles for better understanding
            rotation = Rotation.from_matrix(R)
            euler_angles = rotation.as_euler('xyz', degrees=True)
            
            # Log results
            self.get_logger().info("=== LIDAR to Camera Transformation ===")
            self.get_logger().info("Translation vector:")
            self.get_logger().info(f"  [{t[0]:.6f}, {t[1]:.6f}, {t[2]:.6f}]")
            self.get_logger().info("Rotation matrix:")
            for i in range(3):
                self.get_logger().info(f"  [{R[i, 0]:.6f}, {R[i, 1]:.6f}, {R[i, 2]:.6f}]")
            self.get_logger().info("Euler angles (degrees):")
            self.get_logger().info(f"  [{euler_angles[0]:.2f}, {euler_angles[1]:.2f}, {euler_angles[2]:.2f}]")
            self.get_logger().info(f"Mean squared error: {error:.6f} meters")
            
            # Output ROS TF2 code
            self.output_tf2_code()
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error calculating transformation: {str(e)}")
            return False
    
    def calibrate_with_physical_constraints(self):
        """
        Calibrate with physical constraints to ensure accuracy for ground tracking.
        This is recommended for basketball tracking on ground.
        """
        # Calculate standard transformation first
        if not self.calculate_transformation():
            return
        
        # Now apply physical constraints
        
        # 1. Ensure Z transformation matches physical setup
        # For ground objects, Z difference should match height difference
        height_diff = self.camera_height - self.lidar_height
        
        # Store original values for comparison
        original_z = self.translation[2]
        
        # Update Z translation
        self.translation[2] = height_diff
        
        # 2. Ensure rotation preserves ground plane alignment
        # Simplify rotation matrix to focus on X-Y plane
        # This assumes both sensors are mounted level with the ground
        self.rotation_matrix[2, 0] = 0.0
        self.rotation_matrix[2, 1] = 0.0
        self.rotation_matrix[0, 2] = 0.0
        self.rotation_matrix[1, 2] = 0.0
        self.rotation_matrix[2, 2] = 1.0
        
        # Calculate error with constraints
        constrained_error = self.calculate_error()
        
        # Convert to useful formats
        rotation = Rotation.from_matrix(self.rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        
        # Log results
        self.get_logger().info("=== LIDAR to Camera Transformation with Physical Constraints ===")
        self.get_logger().info(f"Applied physical constraints: Z-translation set to {height_diff:.4f}m")
        self.get_logger().info(f"Original Z-translation was: {original_z:.4f}m")
        self.get_logger().info(f"Camera height: {self.camera_height}m, LIDAR height: {self.lidar_height}m")
        self.get_logger().info("\nFinal transformation:")
        self.get_logger().info("Translation vector:")
        self.get_logger().info(f"  [{self.translation[0]:.6f}, {self.translation[1]:.6f}, {self.translation[2]:.6f}]")
        self.get_logger().info("Rotation matrix:")
        for i in range(3):
            self.get_logger().info(f"  [{self.rotation_matrix[i, 0]:.6f}, {self.rotation_matrix[i, 1]:.6f}, {self.rotation_matrix[i, 2]:.6f}]")
        self.get_logger().info("Quaternion [x, y, z, w]:")
        self.get_logger().info(f"  [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]")
        self.get_logger().info(f"Mean squared error with constraints: {constrained_error:.6f} meters")
        
        # Output ROS TF2 code
        self.output_tf2_code()
    
    def calculate_error(self):
        """Calculate mean squared error of the transformation."""
        if not self.has_transform:
            return float('inf')
        
        total_error = 0.0
        n = len(self.lidar_points)
        
        for i in range(n):
            lidar_point = self.lidar_points[i]
            camera_point = self.camera_points[i]
            
            # Transform LIDAR point to camera frame
            transformed_point = self.rotation_matrix @ lidar_point + self.translation
            
            # Calculate squared error
            error = np.sum((transformed_point - camera_point) ** 2)
            total_error += error
        
        return total_error / n
    
    def test_transformation(self):
        """Test the transformation on current sensor readings."""
        if not self.has_transform:
            self.get_logger().error("No transformation calculated yet!")
            return
        
        with self.lock:
            camera_point = self.latest_camera_point
            lidar_point = self.latest_lidar_point
        
        if camera_point is None or lidar_point is None:
            self.get_logger().error("Missing data from one or both sensors")
            return
        
        # Transform LIDAR point to camera frame
        transformed_point = self.rotation_matrix @ lidar_point + self.translation
        
        # Calculate error
        error = np.sqrt(np.sum((transformed_point - camera_point) ** 2))
        error_percent = (error / np.linalg.norm(camera_point)) * 100 if np.linalg.norm(camera_point) > 0 else 0
        
        # Calculate distances
        lidar_dist = np.sqrt(lidar_point[0]**2 + lidar_point[1]**2)
        camera_dist = np.sqrt(camera_point[0]**2 + camera_point[1]**2)
        transformed_dist = np.sqrt(transformed_point[0]**2 + transformed_point[1]**2)
        
        self.get_logger().info("=== Transformation Test ===")
        self.get_logger().info(f"Camera:     ({camera_point[0]:.3f}, {camera_point[1]:.3f}, {camera_point[2]:.3f}), dist={camera_dist:.2f}m")
        self.get_logger().info(f"LIDAR:      ({lidar_point[0]:.3f}, {lidar_point[1]:.3f}, {lidar_point[2]:.3f}), dist={lidar_dist:.2f}m")
        self.get_logger().info(f"Transformed: ({transformed_point[0]:.3f}, {transformed_point[1]:.3f}, {transformed_point[2]:.3f}), dist={transformed_dist:.2f}m")
        self.get_logger().info(f"Error:      {error:.3f} meters ({error_percent:.1f}%)")
        
        # Provide assessment
        if error < 0.05:
            self.get_logger().info("Transformation quality: EXCELLENT")
        elif error < 0.10:
            self.get_logger().info("Transformation quality: GOOD")
        elif error < 0.20:
            self.get_logger().info("Transformation quality: FAIR")
        else:
            self.get_logger().info("Transformation quality: POOR - Consider recalibration")
    
    def publish_transform(self):
        """Publish the current transformation as a TF2 transform."""
        if not self.has_transform:
            return
        
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "camera_frame"  # Parent frame
        transform.child_frame_id = "lidar_frame"    # Child frame
        
        # Set translation
        transform.transform.translation.x = float(self.translation[0])
        transform.transform.translation.y = float(self.translation[1])
        transform.transform.translation.z = float(self.translation[2])
        
        # Convert rotation matrix to quaternion
        rotation = Rotation.from_matrix(self.rotation_matrix)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        
        transform.transform.rotation.x = float(quaternion[0])
        transform.transform.rotation.y = float(quaternion[1])
        transform.transform.rotation.z = float(quaternion[2])
        transform.transform.rotation.w = float(quaternion[3])
        
        self.tf_broadcaster.sendTransform(transform)
    
    def output_tf2_code(self):
        """Output code snippets for using the transformation in ROS TF2."""
        # Convert rotation to quaternion
        rotation = Rotation.from_matrix(self.rotation_matrix)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        
        # Output Python code
        self.get_logger().info("\n=== Python Code for ROS TF2 ===")
        python_code = f"""
# Add this to your configuration file (basketball_lidar_config.yaml)
transform:
  parent_frame: "camera_frame"
  child_frame: "lidar_frame"
  translation:
    x: {self.translation[0]:.6f}
    y: {self.translation[1]:.6f}
    z: {self.translation[2]:.6f}
  rotation:
    x: {quaternion[0]:.6f}
    y: {quaternion[1]:.6f}
    z: {quaternion[2]:.6f}
    w: {quaternion[3]:.6f}
  publish_frequency: 10.0
  log_interval: 60.0
"""
        self.get_logger().info(python_code)
        
        # Output JSON for copying
        json_data = {
            "transform": {
                "parent_frame": "camera_frame",
                "child_frame": "lidar_frame",
                "translation": {
                    "x": float(self.translation[0]),
                    "y": float(self.translation[1]),
                    "z": float(self.translation[2])
                },
                "rotation": {
                    "x": float(quaternion[0]),
                    "y": float(quaternion[1]),
                    "z": float(quaternion[2]),
                    "w": float(quaternion[3])
                }
            }
        }
        
        self.get_logger().info("\n=== JSON for Copy/Paste ===")
        self.get_logger().info(json.dumps(json_data, indent=2))
    
    def assess_calibration_quality(self):
        """
        Assess the quality of the current calibration and provide guidance.
        """
        if not self.has_transform:
            self.get_logger().error("No transformation calculated yet!")
            return "None", float('inf')
        
        # Calculate error for each point
        errors = []
        for lidar_point, camera_point in zip(self.lidar_points, self.camera_points):
            # Transform LIDAR point to camera frame
            transformed = self.rotation_matrix @ lidar_point + self.translation
            # Calculate error magnitude
            error = np.linalg.norm(transformed - camera_point)
            errors.append(error)
        
        # Calculate statistics
        mean_error = np.mean(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)
        std_error = np.std(errors)
        
        # Evaluate calibration quality
        if mean_error < 0.05:
            quality = "Excellent"
        elif mean_error < 0.10:
            quality = "Good"
        elif mean_error < 0.20:
            quality = "Fair"
        else:
            quality = "Poor"
        
        # Print assessment
        self.get_logger().info("=== Calibration Quality Assessment ===")
        self.get_logger().info(f"Quality: {quality}")
        self.get_logger().info(f"Mean error: {mean_error:.3f}m")
        self.get_logger().info(f"Error range: {min_error:.3f}m - {max_error:.3f}m")
        self.get_logger().info(f"Standard deviation: {std_error:.3f}m")
        
        # Point-by-point breakdown
        self.get_logger().info("\nPoint-by-Point Error Analysis:")
        for i, error in enumerate(errors):
            status = "GOOD" if error < 0.1 else "FAIR" if error < 0.2 else "POOR"
            self.get_logger().info(f"  Point {i+1}: {error:.3f}m - {status}")
        
        # Recommendations
        self.get_logger().info("\nRecommendations:")
        if quality == "Poor":
            self.get_logger().info("- Recalibrate with more well-distributed points")
            self.get_logger().info("- Focus on collecting points in the 1-2m range")
            self.get_logger().info("- Make sure the basketball is stable when capturing points")
        elif std_error > 0.05:
            self.get_logger().info("- Add more calibration points to improve consistency")
            self.get_logger().info("- Try the 'p' option to apply physical constraints")
        
        # Analyze distribution
        self.analyze_point_distribution(silent=True)
        
        return quality, mean_error
    
    def analyze_point_distribution(self, silent=False):
        """
        Analyze the distribution of calibration points to guide collection.
        
        Args:
            silent: If True, only output recommendations, not full analysis
        """
        if len(self.lidar_points) < 2:
            if not silent:
                self.get_logger().info("Need at least 2 points for distribution analysis")
            return
        
        lidar_array = np.array(self.lidar_points)
        
        # Calculate ranges
        x_min, y_min = np.min(lidar_array[:, :2], axis=0)
        x_max, y_max = np.max(lidar_array[:, :2], axis=0)
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        # Calculate distances
        distances = np.sqrt(lidar_array[:, 0]**2 + lidar_array[:, 1]**2)
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        # Count points in different regions
        close_points = sum(1 for d in distances if d < 1.0)
        mid_points = sum(1 for d in distances if 1.0 <= d < 2.0)
        far_points = sum(1 for d in distances if d >= 2.0)
        
        # Calculate quadrant distribution
        quadrants = [
            sum(1 for x, y in lidar_array[:, :2] if x >= 0 and y >= 0),  # Q1: +x, +y
            sum(1 for x, y in lidar_array[:, :2] if x < 0 and y >= 0),   # Q2: -x, +y
            sum(1 for x, y in lidar_array[:, :2] if x < 0 and y < 0),    # Q3: -x, -y
            sum(1 for x, y in lidar_array[:, :2] if x >= 0 and y < 0)    # Q4: +x, -y
        ]
        
        # Skip full analysis if silent
        if not silent:
            # Print analysis
            self.get_logger().info("=== Calibration Point Distribution Analysis ===")
            self.get_logger().info(f"Total points: {len(self.lidar_points)}")
            self.get_logger().info(f"X range: {x_min:.2f}m to {x_max:.2f}m (span: {x_range:.2f}m)")
            self.get_logger().info(f"Y range: {y_min:.2f}m to {y_max:.2f}m (span: {y_range:.2f}m)")
            self.get_logger().info(f"Distance range: {min_distance:.2f}m to {max_distance:.2f}m")
            self.get_logger().info("\nDistance Distribution:")
            self.get_logger().info(f"  Close range (<1m): {close_points}/{len(self.lidar_points)}")
            self.get_logger().info(f"  Medium range (1-2m): {mid_points}/{len(self.lidar_points)}")
            self.get_logger().info(f"  Far range (>2m): {far_points}/{len(self.lidar_points)}")
            self.get_logger().info("\nQuadrant Distribution:")
            self.get_logger().info(f"  Q1 (+x, +y): {quadrants[0]}/{len(self.lidar_points)}")
            self.get_logger().info(f"  Q2 (-x, +y): {quadrants[1]}/{len(self.lidar_points)}")
            self.get_logger().info(f"  Q3 (-x, -y): {quadrants[2]}/{len(self.lidar_points)}")
            self.get_logger().info(f"  Q4 (+x, -y): {quadrants[3]}/{len(self.lidar_points)}")
        
        # Provide recommendations
        recommendations = []
        
        if x_range < 0.5:
            recommendations.append("Add points with more X variation (different distances)")
        if y_range < 0.2:
            recommendations.append("Add points with more Y variation (different sides)")
        
        # Recommend based on distance distribution
        if close_points < 2 and mid_points < 2:
            recommendations.append("Add more points in the 0.8-2m range")
        if mid_points < 2:
            recommendations.append("Add more medium range points (1-2m)")
        
        # Recommend based on quadrant
        min_quadrant = min(quadrants)
        for i, count in enumerate(quadrants):
            if count == min_quadrant and count < 2:
                quadrant_names = ["Q1 (+x, +y)", "Q2 (-x, +y)", "Q3 (-x, -y)", "Q4 (+x, -y)"]
                recommendations.append(f"Add more points in {quadrant_names[i]}")
        
        # Output recommendations if we have any
        if recommendations:
            self.get_logger().info("\nRecommendations for Better Calibration:")
            for i, rec in enumerate(recommendations):
                self.get_logger().info(f"- {rec}")
    
    def filter_outliers(self):
        """
        Automatically identify and remove outlier calibration points.
        
        This function removes points that:
        1. Have large discrepancies between camera and LIDAR distances
        2. Show high errors after transformation
        3. Are statistical outliers based on error metrics
        
        Returns:
            int: Number of points removed
        """
        if len(self.camera_points) < 4:
            self.get_logger().error("Need at least 4 points to identify outliers")
            return 0
        
        # First calculate a transformation to identify errors
        if not self.has_transform:
            self.calculate_transformation()
        
        # Get error for each point
        errors = []
        for i, (camera_point, lidar_point) in enumerate(zip(self.camera_points, self.lidar_points)):
            # 1. Calculate distance discrepancy
            camera_dist = np.linalg.norm(camera_point[:2])  # XY plane distance
            lidar_dist = np.linalg.norm(lidar_point[:2])    # XY plane distance
            dist_error = abs(camera_dist - lidar_dist) / max(camera_dist, 0.1)
            
            # 2. Calculate transformation error
            transformed = self.rotation_matrix @ lidar_point + self.translation
            transform_error = np.linalg.norm(transformed - camera_point)
            
            # Store error metrics
            total_error = transform_error
            errors.append({
                'index': i,
                'dist_error': dist_error,
                'transform_error': transform_error,
                'total_error': total_error,
                'camera_point': camera_point,
                'lidar_point': lidar_point
            })
        
        # Calculate mean and standard deviation of transform errors
        error_values = [e['transform_error'] for e in errors]
        mean_error = np.mean(error_values)
        std_error = np.std(error_values)
        
        # Set thresholds for different error types
        transform_threshold = mean_error + 1.5 * std_error
        distance_threshold = 0.4  # 40% difference in distance
        
        # Identify outliers
        outliers = []
        for e in errors:
            # Mark as outlier if transform error is high
            if e['transform_error'] > transform_threshold:
                outliers.append(e)
                continue
                
            # Mark as outlier if distance discrepancy is high
            if e['dist_error'] > distance_threshold:
                outliers.append(e)
                continue
        
        if not outliers:
            self.get_logger().info("No clear outliers detected in calibration data")
            return 0
        
        # Sort outliers by index in reverse order (to remove from end first)
        outliers.sort(key=lambda x: x['index'], reverse=True)
        
        # Remove outliers
        removed_count = 0
        for outlier in outliers:
            idx = outlier['index']
            camera_point = outlier['camera_point']
            lidar_point = outlier['lidar_point']
            
            # Get the point distances for logging
            camera_dist = np.linalg.norm(camera_point[:2])
            lidar_dist = np.linalg.norm(lidar_point[:2])
            
            # Log the removal
            self.get_logger().warn(f"Removing outlier point {idx+1}:")
            self.get_logger().warn(f"  Camera: ({camera_point[0]:.3f}, {camera_point[1]:.3f}, {camera_point[2]:.3f}), dist={camera_dist:.2f}m")
            self.get_logger().warn(f"  LIDAR:  ({lidar_point[0]:.3f}, {lidar_point[1]:.3f}, {lidar_point[2]:.3f}), dist={lidar_dist:.2f}m")
            self.get_logger().warn(f"  Error: {outlier['transform_error']:.3f}m, Distance mismatch: {outlier['dist_error']*100:.1f}%")
            
            # Remove the points
            del self.camera_points[idx]
            del self.lidar_points[idx]
            removed_count += 1
        
        # Recalculate transformation after removing outliers
        if removed_count > 0:
            self.get_logger().info(f"Removed {removed_count} outlier points")
            self.calculate_transformation()
            self.get_logger().info("New calibration quality after filtering:")
            self.assess_calibration_quality()
        
        return removed_count


def main(args=None):
    """Main function."""
    rclpy.init(args=args)
    
    node = LidarCameraCalibrator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Stopping calibration (Ctrl+C pressed)")
    except Exception as e:
        node.get_logger().error(f"Error: {str(e)}")
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()