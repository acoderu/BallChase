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
        self.get_logger().info("Instructions:")
        self.get_logger().info("1. Place the basketball in view of both sensors")
        self.get_logger().info("2. Wait for both sensors to detect the ball")
        self.get_logger().info("3. Use commands below to capture points & calculate transform")
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
        self.get_logger().info("- Collect at least 5-6 point pairs at different positions")
        self.get_logger().info("- Include points at different distances (0.5-3m)")
        self.get_logger().info("- Include points at different sides of the robot")
        self.get_logger().info("- Best results in the 1-2m range for both sensors")
        self.get_logger().info("====================================================")
    
    def camera_callback(self, msg):
        """Process camera position messages."""
        # Debug print to confirm callback is being triggered
        self.get_logger().info(f"Camera callback triggered: ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})")
        
        with self.lock:
            self.latest_camera_point = np.array([
                msg.point.x,
                msg.point.y,
                msg.point.z
            ])
            self.camera_time = time.time()
    
    def lidar_callback(self, msg):
        """Process LIDAR position messages."""
        # Debug print to confirm callback is being triggered
        self.get_logger().info(f"LIDAR callback triggered: ({msg.point.x:.3f}, {msg.point.y:.3f}, {msg.point.z:.3f})")
        
        with self.lock:
            self.latest_lidar_point = np.array([
                msg.point.x,
                msg.point.y,
                msg.point.z
            ])
            self.lidar_time = time.time()
    
    def timer_callback(self):
        """Periodic updates including TF broadcasting."""
        # If we have a transform, publish it
        if self.has_transform:
            self.publish_transform()
        
        # Check data freshness
        current_time = time.time()
        with self.lock:
            camera_age = current_time - self.camera_time if self.camera_time > 0 else float('inf')
            lidar_age = current_time - self.lidar_time if self.lidar_time > 0 else float('inf')
        
        # Report if data is stale
        if camera_age > 10.0:
            self.get_logger().warn(f"No recent camera data (last seen {camera_age:.1f}s ago)")
        if lidar_age > 10.0:
            self.get_logger().warn(f"No recent LIDAR data (last seen {lidar_age:.1f}s ago)")
    
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
                self.get_logger().info("Commands: c=capture, l=list, r=remove, x=calculate, p=physical, a=assess, t=test, q=quit")
    
    def capture_point_pair(self):
        """Capture a pair of corresponding points from both sensors."""
        with self.lock:
            camera_point = self.latest_camera_point
            lidar_point = self.latest_lidar_point
            camera_age = time.time() - self.camera_time if self.camera_time > 0 else float('inf')
            lidar_age = time.time() - self.lidar_time if self.lidar_time > 0 else float('inf')
        
        # Check if we have recent data
        if camera_point is None or lidar_point is None:
            self.get_logger().error("Missing data from one or both sensors")
            return
        
        if camera_age > 0.5 or lidar_age > 0.5:
            self.get_logger().warn(f"Using stale data: camera={camera_age:.2f}s, lidar={lidar_age:.2f}s")
        
        # Add the points to calibration sets
        self.camera_points.append(np.copy(camera_point))
        self.lidar_points.append(np.copy(lidar_point))
        
        # Calculate distance for reference
        lidar_distance = np.sqrt(lidar_point[0]**2 + lidar_point[1]**2)
        camera_distance = np.sqrt(camera_point[0]**2 + camera_point[1]**2)
        
        # Print capture info
        self.get_logger().info(f"Captured point pair #{len(self.camera_points)}:")
        self.get_logger().info(f"  Camera: ({camera_point[0]:.3f}, {camera_point[1]:.3f}, {camera_point[2]:.3f}), dist={camera_distance:.2f}m")
        self.get_logger().info(f"  LIDAR:  ({lidar_point[0]:.3f}, {lidar_point[1]:.3f}, {lidar_point[2]:.3f}), dist={lidar_distance:.2f}m")
        
        # Provide guidance
        if lidar_distance < 0.5:
            self.get_logger().warn("Point is very close to LIDAR. Reliability may be low.")
        elif 1.0 <= lidar_distance <= 3.0:
            self.get_logger().info("Point is in optimal LIDAR range (1-3m). Good calibration point.")
        
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
            self.get_logger().info("- Focus on collecting points in the 1-3m range")
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
        if close_points < 2:
            recommendations.append("Add more close range points (<1m)")
        if mid_points < 2:
            recommendations.append("Add more medium range points (1-2m)")
        if far_points < 2:
            recommendations.append("Add more far range points (>2m)")
        
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