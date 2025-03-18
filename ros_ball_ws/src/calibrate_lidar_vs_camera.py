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
            elif cmd == 't':  # Test transformation
                self.test_transformation()
            elif cmd == 'q':  # Quit
                self.running = False
                self.get_logger().info("Exiting...")
                break
            else:
                self.get_logger().info("Unknown command")

    def capture_point_pair(self):
        """Capture a pair of corresponding points from camera and LIDAR."""
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
            
        # Add the points to our calibration sets
        self.camera_points.append(np.copy(camera_point))
        self.lidar_points.append(np.copy(lidar_point))
        
        self.get_logger().info(f"Captured point pair #{len(self.camera_points)}:")
        self.get_logger().info(f"  Camera: ({camera_point[0]:.3f}, {camera_point[1]:.3f}, {camera_point[2]:.3f})")
        self.get_logger().info(f"  LIDAR:  ({lidar_point[0]:.3f}, {lidar_point[1]:.3f}, {lidar_point[2]:.3f})")

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
        """Calculate the rigid transformation from LIDAR to camera frame."""
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
            
            # Compute the covariance matrix
            H = lidar_centered.T @ camera_centered
            
            # Find the rotation using SVD
            U, _, Vt = np.linalg.svd(H)
            
            # Ensure a right-handed coordinate system
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                self.get_logger().warn("Reflection detected in rotation. Fixing...")
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            # Calculate the translation
            t = camera_centroid - R @ lidar_centroid
            
            # Store the transformation
            self.rotation_matrix = R
            self.translation = t
            self.has_transform = True
            
            # Convert to Euler angles for better understanding
            rotation = Rotation.from_matrix(R)
            euler_angles = rotation.as_euler('xyz', degrees=True)
            
            self.get_logger().info("=== LIDAR to Camera Transformation ===")
            self.get_logger().info("Translation vector:")
            self.get_logger().info(f"  [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
            self.get_logger().info("Rotation matrix:")
            for i in range(3):
                self.get_logger().info(f"  [{R[i, 0]:.4f}, {R[i, 1]:.4f}, {R[i, 2]:.4f}]")
            self.get_logger().info("Euler angles (degrees):")
            self.get_logger().info(f"  [{euler_angles[0]:.2f}, {euler_angles[1]:.2f}, {euler_angles[2]:.2f}]")
            
            # Calculate error
            error = self.calculate_error()
            self.get_logger().info(f"Mean squared error: {error:.6f} meters")
            
            # Provide ROS TF2 code
            self.output_tf2_code(R, t, euler_angles)
            
        except Exception as e:
            self.get_logger().error(f"Error calculating transformation: {str(e)}")

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