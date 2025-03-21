#!/usr/bin/env python3

"""
LIDAR-Camera Calibration Tool for Tennis Ball Tracking Robot
===========================================================

This script helps calibrate the transformation between LIDAR and camera coordinate systems.
It works by:
1. Collecting pairs of 3D positions from both sensors observing the same tennis ball
2. Computing the transformation matrix that best aligns the LIDAR points to camera points
3. Outputting the transformation parameters to use in your ROS TF2 setup

Usage:
------
1. Place the tennis ball at various positions in front of the robot
2. For each position, record both sensors' readings using this tool
3. Generate the transformation and test it

Requirements:
- ROS2 Humble
- numpy
- scipy
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
import numpy as np
from scipy.spatial.transform import Rotation
import tf2_ros
import time
import threading
import math


class LidarCameraCalibrator(Node):
    """
    Node for calibrating the transformation between LIDAR and camera coordinate systems.
    """
    
    def __init__(self):
        super().__init__('lidar_camera_calibrator')
        
        # Storage for calibration points
        self.camera_points = []  # Points from camera
        self.lidar_points = []   # Corresponding points from LIDAR
        
        # Latest received points (for capturing)
        self.latest_camera_point = None
        self.latest_lidar_point = None
        self.camera_time = 0
        self.lidar_time = 0
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Create subscribers for both sensors
        self.camera_sub = self.create_subscription(
            PointStamped,
            '/tennis_ball/detected_position',  # 3D position from depth camera
            self.camera_callback,
            10
        )
        
        self.lidar_sub = self.create_subscription(
            PointStamped,
            '/tennis_ball/lidar/position',  # LIDAR detection
            self.lidar_callback,
            10
        )
        
        # Publisher for TF visualization
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Create a timer for periodic updates and user interface
        self.timer = self.create_timer(0.5, self.timer_callback)
        
        # Transformation parameters (will be computed)
        self.translation = np.zeros(3)
        self.rotation_matrix = np.eye(3)
        self.has_transform = False
        
        self.get_logger().info("=== LIDAR-Camera Calibration Tool ===")
        self.get_logger().info("Place the tennis ball at different positions")
        self.get_logger().info("and use these commands:")
        self.get_logger().info("  'c' - Capture current point pair")
        self.get_logger().info("  'l' - List captured points")
        self.get_logger().info("  'r' - Remove last point pair")
        self.get_logger().info("  'x' - Calculate transformation")
        self.get_logger().info("  't' - Test transformation on current points")
        self.get_logger().info("  'q' - Quit")
        
        # Physical measurements for ground-level tracking
        self.lidar_height = 0.1524  # 6 inches in meters
        self.camera_height = 0.1016  # 4 inches in meters
        self.ball_height = 0.0381  # 1.5 inches in meters (ball center)
        self.ball_radius = 0.03429  # 1.35 inches in meters
        
        # Start input thread
        self.running = True
        self.input_thread = threading.Thread(target=self.input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()

    def camera_callback(self, msg):
        """Process camera position messages."""
        with self.lock:
            self.latest_camera_point = np.array([
                msg.point.x,
                msg.point.y, 
                msg.point.z
            ])
            self.camera_time = time.time()

    def lidar_callback(self, msg):
        """Process LIDAR position messages."""
        with self.lock:
            self.latest_lidar_point = np.array([
                msg.point.x,
                msg.point.y,
                msg.point.z
            ])
            self.lidar_time = time.time()

    def timer_callback(self):
        """Periodic updates including TF broadcasting."""
        # If we have a transformation, publish it
        if self.has_transform:
            self.publish_transform()
        
        # Check data freshness
        current_time = time.time()
        with self.lock:
            camera_age = current_time - self.camera_time if self.camera_time > 0 else float('inf')
            lidar_age = current_time - self.lidar_time if self.lidar_time > 0 else float('inf')
        
        # Only report if data is stale (more than 2 seconds old)
        if camera_age > 2.0:
            self.get_logger().warn(f"No recent camera data (last seen {camera_age:.1f}s ago)")
        if lidar_age > 2.0:
            self.get_logger().warn(f"No recent LIDAR data (last seen {lidar_age:.1f}s ago)")

    def input_loop(self):
        """Handle user input for the calibration process."""
        while self.running:
            cmd = input("> ").strip().lower()
            
            if cmd == 'c':  # Capture point pair
                self.capture_point_pair()
            elif cmd == 'l':  # List points
                self.list_points()
            elif cmd == 'r':  # Remove last point
                self.remove_last_point()
            elif cmd == 'x':  # Calculate transformation
                self.calculate_transformation()
            elif cmd == 'p':  # Calculate with physical constraints
                self.calibrate_with_physical_constraints()  # Add this option
            elif cmd == 'a':  # Assess calibration quality
                self.assess_calibration_quality()
            elif cmd == 't':  # Test transformation
                self.test_transformation()
            elif cmd == 'q':  # Quit
                self.running = False
                self.get_logger().info("Exiting...")
                break
            else:
                self.get_logger().info("Unknown command")
                self.get_logger().info("Commands: c=capture, l=list, r=remove, x=calculate, p=physical, a=assess, t=test, q=quit")

    def capture_point_pair(self):
        """Capture a pair of corresponding points with distance information."""
        with self.lock:
            camera_point = self.latest_camera_point
            lidar_point = self.latest_lidar_point
            camera_age = time.time() - self.camera_time if self.camera_time > 0 else float('inf')
            lidar_age = time.time() - self.lidar_time if self.lidar_time > 0 else float('inf')
        
        # Check if we have recent data from both sensors
        if camera_point is None or lidar_point is None:
            self.get_logger().error("Missing data from one or both sensors")
            return
            
        if camera_age > 0.5 or lidar_age > 0.5:
            self.get_logger().warn(f"Using stale data: camera={camera_age:.2f}s, lidar={lidar_age:.2f}s")
        
        # Calculate distance from LIDAR
        lidar_distance = np.sqrt(lidar_point[0]**2 + lidar_point[1]**2)
        
        # Add the points to our calibration sets
        self.camera_points.append(np.copy(camera_point))
        self.lidar_points.append(np.copy(lidar_point))
        
        # Print distance information for calibration reference
        self.get_logger().info(f"Captured point pair #{len(self.camera_points)}:")
        self.get_logger().info(f"  Camera: ({camera_point[0]:.3f}, {camera_point[1]:.3f}, {camera_point[2]:.3f})")
        self.get_logger().info(f"  LIDAR:  ({lidar_point[0]:.3f}, {lidar_point[1]:.3f}, {lidar_point[2]:.3f})")
        self.get_logger().info(f"  Distance from LIDAR: {lidar_distance:.3f}m")
        
        # Provide guidance on calibration ranges
        if lidar_distance < 0.5:
            self.get_logger().warn("Point is very close to LIDAR. Reliability may be low.")
        elif 1.0 <= lidar_distance <= 3.0:
            self.get_logger().info("Point is in optimal LIDAR range (1-3m). Good calibration point.")

    def list_points(self):
        """List all captured point pairs."""
        if not self.camera_points:
            self.get_logger().info("No points captured yet")
            return
            
        self.get_logger().info(f"Captured {len(self.camera_points)} point pairs:")
        for i, (camera_point, lidar_point) in enumerate(zip(self.camera_points, self.lidar_points)):
            self.get_logger().info(f"  Pair #{i+1}:")
            self.get_logger().info(f"    Camera: ({camera_point[0]:.3f}, {camera_point[1]:.3f}, {camera_point[2]:.3f})")
            self.get_logger().info(f"    LIDAR:  ({lidar_point[0]:.3f}, {lidar_point[1]:.3f}, {lidar_point[2]:.3f})")

    def remove_last_point(self):
        """Remove the last captured point pair."""
        if not self.camera_points:
            self.get_logger().info("No points to remove")
            return
            
        self.camera_points.pop()
        self.lidar_points.pop()
        self.get_logger().info(f"Removed last point pair. {len(self.camera_points)} pairs remaining.")

    def calculate_transformation(self):
        """Calculate the rigid transformation from LIDAR to camera frame, optimized for ground plane."""
        if len(self.camera_points) < 3:
            self.get_logger().error("Need at least 3 point pairs for calibration!")
            return
            
        # Convert to numpy arrays
        camera_array = np.array(self.camera_points)
        lidar_array = np.array(self.lidar_points)
        
        # Check if points are well-distributed in 3D space
        lidar_spread = np.std(lidar_array, axis=0)
        camera_spread = np.std(camera_array, axis=0)
        if np.any(lidar_spread < 0.1) or np.any(camera_spread < 0.1):
            self.get_logger().warn("Points are not well distributed in 3D space!")
            self.get_logger().warn(f"LIDAR spread: {lidar_spread}, Camera spread: {camera_spread}")
            self.get_logger().warn("For better calibration, place the ball at widely different positions")
        
        self.get_logger().warn("Note: Since LIDAR Z values are constant, calibration will be accurate in X and Y but not in Z.")
        self.get_logger().warn("Z transformation should be manually set based on physical measurements of sensor positions.")
        
        try:
            # --- Step 1: Find 2D transformation in X-Y plane ---
            camera_xy = camera_array[:, :2]  # Just X,Y coordinates
            lidar_xy = lidar_array[:, :2]    # Just X,Y coordinates
            
            # Compute 2D centroids
            camera_centroid_xy = np.mean(camera_xy, axis=0)
            lidar_centroid_xy = np.mean(lidar_xy, axis=0)
            
            # Center the points in 2D
            camera_centered_xy = camera_xy - camera_centroid_xy
            lidar_centered_xy = lidar_xy - lidar_centroid_xy
            
            # Find best 2D rotation
            H_xy = lidar_centered_xy.T @ camera_centered_xy
            U, _, Vt = np.linalg.svd(H_xy)
            R_xy = Vt.T @ U.T
            
            if np.linalg.det(R_xy) < 0:
                self.get_logger().warn("Reflection detected in 2D rotation. Fixing...")
                Vt[-1, :] *= -1
                R_xy = Vt.T @ U.T
            
            # --- Step 2: Construct full 3D transformation ---
            # Full rotation matrix (only rotate in X-Y plane)
            R = np.eye(3)
            R[:2, :2] = R_xy
            
            # Calculate Z offset directly (average difference)
            z_offset = np.mean(camera_array[:, 2]) - np.mean(lidar_array[:, 2])
            
            # Translation vector
            t = np.zeros(3)
            t[:2] = camera_centroid_xy - (R_xy @ lidar_centroid_xy)
            t[2] = z_offset
            
            # Store the transformation
            self.rotation_matrix = R
            self.translation = t
            self.has_transform = True
            
            # Convert to Euler angles for better understanding
            rotation = Rotation.from_matrix(R)
            euler_angles = rotation.as_euler('xyz', degrees=True)
            
            self.get_logger().info("=== LIDAR to Camera Transformation (Ground-Plane Optimized) ===")
            self.get_logger().info("Translation vector:")
            self.get_logger().info(f"  [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
            self.get_logger().info("Rotation matrix (planar rotation only):")
            for i in range(3):
                self.get_logger().info(f"  [{R[i, 0]:.4f}, {R[i, 1]:.4f}, {R[i, 2]:.4f}]")
            self.get_logger().info("Euler angles (degrees):")
            self.get_logger().info(f"  [{euler_angles[0]:.2f}, {euler_angles[1]:.2f}, {euler_angles[2]:.2f}]")
            
            # Calculate error
            error = self.calculate_error()
            self.get_logger().info(f"Mean squared error: {error:.6f} meters")
            
            # Provide ROS TF2 code
            self.output_tf2_code(R, t, euler_angles)

            self.get_logger().info("\n=== 3D Error Visualization ===")
            for i, (lidar, camera, transformed) in enumerate(zip(
                self.lidar_points, 
                self.camera_points,
                [self.rotation_matrix @ lp + self.translation for lp in self.lidar_points]
            )):
                err = np.linalg.norm(transformed - camera)
                self.get_logger().info(f"Point {i+1}: Error = {err:.3f}m {'<!>' if err > 0.1 else ''}")
                
            self.get_logger().info("\n=== Manual Z Adjustment ===")
            self.get_logger().info("IMPORTANT: Since LIDAR values have constant Z, the Z component of translation")
            self.get_logger().info("should be manually set to the physical difference between sensors:")
            self.get_logger().info("1. Measure the height of the LIDAR from the ground")
            self.get_logger().info("2. Measure the height of the camera from the ground")
            self.get_logger().info("3. Set transform_translation.z to (camera_height - lidar_height)")
            self.get_logger().info("For example, with LIDAR at 15cm and camera at 10cm, use z = -0.05")
            
        except Exception as e:
            self.get_logger().error(f"Error calculating transformation: {str(e)}")

    def calibrate_with_physical_constraints(self):
        """
        Calibrate transformation with physical measurements as constraints.
        This ensures accurate Z-axis calibration for ground-level ball tracking.
        """
        # Physical measurements in meters
        lidar_height = self.lidar_height  # 6 inches
        camera_height = self.camera_height  # 4 inches
        ball_height = self.ball_height  # 1.5 inches
        
        # Calculate standard transformation first (for X-Y plane)
        self.calculate_transformation()
        
        # Override Z component based on physical measurements
        height_diff = camera_height - lidar_height
        self.translation[2] = height_diff
        
        # Apply identity rotation for Z-axis (assuming sensors are level)
        # This preserves X-Y rotation but ensures Z is handled by translation only
        self.rotation_matrix[2, 2] = 1.0
        self.rotation_matrix[0, 2] = 0.0
        self.rotation_matrix[1, 2] = 0.0
        self.rotation_matrix[2, 0] = 0.0
        self.rotation_matrix[2, 1] = 0.0
        
        # Update the has_transform flag
        self.has_transform = True
        
        # Log the applied constraints
        self.get_logger().info("Applied physical constraint: Z-translation set to measured height difference")
        self.get_logger().info(f"Camera height: {camera_height}m, LIDAR height: {lidar_height}m")
        self.get_logger().info(f"Height difference: {height_diff}m")
        
        # Calculate and log error
        error = self.calculate_error()
        self.get_logger().info(f"Mean squared error after constraint: {error:.6f} meters")
        
        # Convert to useful formats
        rotation = Rotation.from_matrix(self.rotation_matrix)
        euler_angles = rotation.as_euler('xyz', degrees=True)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        
        # Output full transformation details
        self.get_logger().info("=== LIDAR to Camera Transformation with Physical Constraints ===")
        self.get_logger().info("Translation vector:")
        self.get_logger().info(f"  [{self.translation[0]:.6f}, {self.translation[1]:.6f}, {self.translation[2]:.6f}]")
        self.get_logger().info("Rotation matrix:")
        for i in range(3):
            self.get_logger().info(f"  [{self.rotation_matrix[i, 0]:.6f}, {self.rotation_matrix[i, 1]:.6f}, {self.rotation_matrix[i, 2]:.6f}]")
        self.get_logger().info("Quaternion [x, y, z, w]:")
        self.get_logger().info(f"  [{quaternion[0]:.6f}, {quaternion[1]:.6f}, {quaternion[2]:.6f}, {quaternion[3]:.6f}]")

    def calculate_error(self):
        """Calculate the mean squared error of the transformation."""
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
        """Test the transformation on the current sensor readings."""
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
        
        self.get_logger().info("=== Transformation Test ===")
        self.get_logger().info(f"Camera:     ({camera_point[0]:.3f}, {camera_point[1]:.3f}, {camera_point[2]:.3f})")
        self.get_logger().info(f"LIDAR:      ({lidar_point[0]:.3f}, {lidar_point[1]:.3f}, {lidar_point[2]:.3f})")
        self.get_logger().info(f"Transformed: ({transformed_point[0]:.3f}, {transformed_point[1]:.3f}, {transformed_point[2]:.3f})")
        self.get_logger().info(f"Error:      {error:.3f} meters")

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

    def output_tf2_code(self, R, t, euler_angles):
        """Output code snippets for using the transformation in ROS TF2."""
        # Convert rotation to quaternion
        rotation = Rotation.from_matrix(R)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        
        # Output Python code
        self.get_logger().info("\n=== Python Code for ROS TF2 ===")
        python_code = f"""
# Add this to your LIDAR node
def publish_transform(self):
    transform = TransformStamped()
    transform.header.stamp = self.get_clock().now().to_msg()
    transform.header.frame_id = "camera_frame"  # Parent frame
    transform.child_frame_id = "lidar_frame"    # Child frame
    
    # Translation from LIDAR to camera
    transform.transform.translation.x = {t[0]:.6f}
    transform.transform.translation.y = {t[1]:.6f}
    transform.transform.translation.z = {t[2]:.6f}
    
    # Rotation from LIDAR to camera (as quaternion)
    transform.transform.rotation.x = {quaternion[0]:.6f}
    transform.transform.rotation.y = {quaternion[1]:.6f}
    transform.transform.rotation.z = {quaternion[2]:.6f}
    transform.transform.rotation.w = {quaternion[3]:.6f}
    
    self.tf_broadcaster.sendTransform(transform)
"""
        self.get_logger().info(python_code)

        # Output URDF XML snippet
        self.get_logger().info("\n=== URDF XML for Static Transform ===")
        urdf_code = f"""
<link name="camera_frame"/>
<link name="lidar_frame"/>

<joint name="lidar_to_camera_joint" type="fixed">
    <parent link="camera_frame"/>
    <child link="lidar_frame"/>
    <origin xyz="{t[0]} {t[1]} {t[2]}" rpy="{math.radians(euler_angles[0])} {math.radians(euler_angles[1])} {math.radians(euler_angles[2])}"/>
</joint>
"""
        self.get_logger().info(urdf_code)

    def assess_calibration_quality(self):
        """
        Assess the quality of the current calibration and provide guidance.
        
        Returns:
            tuple: (quality_string, mean_error)
        """
        if not self.has_transform:
            self.get_logger().error("No transformation calculated yet!")
            return "None", float('inf')
            
        # Calculate error statistics
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
        if mean_error < 0.03:
            quality = "Excellent"
        elif mean_error < 0.07:
            quality = "Good"
        elif mean_error < 0.15:
            quality = "Fair"
        else:
            quality = "Poor"
        
        # Print assessment
        self.get_logger().info("=== Calibration Quality Assessment ===")
        self.get_logger().info(f"Quality: {quality}")
        self.get_logger().info(f"Mean error: {mean_error:.3f}m")
        self.get_logger().info(f"Error range: {min_error:.3f}m - {max_error:.3f}m")
        self.get_logger().info(f"Standard deviation: {std_error:.3f}m")
        
        # Recommendations
        if quality == "Poor":
            self.get_logger().info("Recommendation: Recalibrate with more well-distributed points")
            self.get_logger().info("Focus on collecting points in the 1-3m range for optimal LIDAR detection")
        elif std_error > 0.05:
            self.get_logger().info("Recommendation: Add more calibration points to improve consistency")
        
        # Distance distribution analysis
        distances = [np.sqrt(p[0]**2 + p[1]**2) for p in self.lidar_points]
        close_points = sum(1 for d in distances if d < 1.0)
        optimal_points = sum(1 for d in distances if 1.0 <= d <= 3.0)
        far_points = sum(1 for d in distances if d > 3.0)
        
        self.get_logger().info("Distance distribution of calibration points:")
        self.get_logger().info(f"  Close range (<1m): {close_points}/{len(distances)}")
        self.get_logger().info(f"  Optimal range (1-3m): {optimal_points}/{len(distances)}")
        self.get_logger().info(f"  Far range (>3m): {far_points}/{len(distances)}")
        
        if optimal_points < 3:
            self.get_logger().info("Recommendation: Add more points in the optimal 1-3m range")
        
        return quality, mean_error


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