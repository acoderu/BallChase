#!/usr/bin/env python3

"""
Terminal-Based Depth Camera Calibration
=======================================

A simplified script that works entirely in the terminal,
no display windows required.
"""

import os
import sys
import time
import numpy as np
import yaml
import threading
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class TerminalCalibrator(Node):
    def __init__(self):
        super().__init__('terminal_depth_calibrator')
        
        # Initialize
        self.cv_bridge = CvBridge()
        self.calibration_data = []
        self.depth_array = None
        self.depth_header = None
        self.depth_scale = 0.001  # Default scale
        self.ready_to_measure = False
        self.measurement_taken = False
        self.depth_msg_count = 0
        self.last_depth_time = 0
        
        # Configure these to match your system
        self.depth_topic = "/ascamera/camera_publisher/depth0/image_raw"
        
        # Subscription
        self.depth_sub = self.create_subscription(
            Image, 
            self.depth_topic,
            self.depth_callback,
            10
        )
        
        self.get_logger().info(f"Terminal Depth Calibrator initialized")
        self.get_logger().info(f"Listening to depth topic: {self.depth_topic}")
    
    def depth_callback(self, msg):
        try:
            # Convert to numpy array
            self.depth_array = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_header = msg.header
            self.last_depth_time = time.time()
            self.depth_msg_count += 1
            
            # If we're waiting for a measurement, take it
            if self.ready_to_measure and not self.measurement_taken:
                depth_value = self.take_measurement()
                if depth_value is not None:
                    self.latest_measurement = depth_value
                    self.measurement_taken = True
                self.ready_to_measure = False
        except Exception as e:
            self.get_logger().error(f"Error processing depth image: {str(e)}")
    
    def take_measurement(self):
        """Take a simple depth measurement from the center of the image."""
        if self.depth_array is None:
            self.get_logger().error("No depth data available")
            print("ERROR: No depth data available")
            return None
        
        # Get center region of image
        height, width = self.depth_array.shape
        center_y, center_x = height // 2, width // 2
        
        # Define a region around center (10% of image size)
        region_size = min(width, height) // 10
        y_start = max(0, center_y - region_size)
        y_end = min(height, center_y + region_size)
        x_start = max(0, center_x - region_size)
        x_end = min(width, center_x + region_size)
        
        # Extract region
        region = self.depth_array[y_start:y_end, x_start:x_end]
        
        # Remove invalid depths (zeros)
        valid_depths = region[region > 0]
        
        if len(valid_depths) < 10:
            print(f"WARNING: Only {len(valid_depths)} valid depth points found")
            if len(valid_depths) == 0:
                print("No valid depth points found. Is the object visible to the camera?")
                return None
        
        # Calculate median depth (more robust than mean)
        depth_value = np.median(valid_depths) * self.depth_scale
        
        # Show depth stats
        min_depth = np.min(valid_depths) * self.depth_scale
        max_depth = np.max(valid_depths) * self.depth_scale
        mean_depth = np.mean(valid_depths) * self.depth_scale
        std_depth = np.std(valid_depths) * self.depth_scale
        
        print(f"\nDepth Statistics (center region):")
        print(f"  Median depth: {depth_value:.3f} meters")
        print(f"  Mean depth:   {mean_depth:.3f} meters")
        print(f"  Min depth:    {min_depth:.3f} meters")
        print(f"  Max depth:    {max_depth:.3f} meters")
        print(f"  Std dev:      {std_depth:.3f} meters")
        print(f"  Valid points: {len(valid_depths)}")
        
        return depth_value

    def linear_correction(self, x, a, b):
        """Linear correction function: y = ax + b"""
        return a * x + b
    
    def polynomial_correction(self, x, a, b, c):
        """Quadratic correction function: y = ax^2 + bx + c"""
        return a * x**2 + b * x + c

    def compute_calibration(self):
        """Compute calibration from collected measurements."""
        if len(self.calibration_data) < 3:
            print("Need at least 3 measurements for calibration")
            return False
            
        # Extract data
        true_distances = np.array([d[0] for d in self.calibration_data])
        measured_distances = np.array([d[1] for d in self.calibration_data])
        
        # Fit models
        try:
            # Linear fit
            linear_params, _ = curve_fit(self.linear_correction, measured_distances, true_distances)
            linear_a, linear_b = linear_params
            
            # Polynomial fit if we have enough data points
            if len(self.calibration_data) >= 5:
                poly_params, _ = curve_fit(self.polynomial_correction, measured_distances, true_distances)
                poly_a, poly_b, poly_c = poly_params
                
                # Compute errors
                poly_predicted = self.polynomial_correction(measured_distances, poly_a, poly_b, poly_c)
                poly_errors = np.abs(poly_predicted - true_distances)
                poly_mean_error = np.mean(poly_errors)
            else:
                poly_mean_error = float('inf')
            
            # Compute linear errors
            linear_predicted = self.linear_correction(measured_distances, linear_a, linear_b)
            linear_errors = np.abs(linear_predicted - true_distances)
            linear_mean_error = np.mean(linear_errors)
            
            # Choose better model
            if poly_mean_error < linear_mean_error and len(self.calibration_data) >= 5:
                correction_type = "polynomial"
                params = [float(poly_a), float(poly_b), float(poly_c)]
                predicted = poly_predicted
                mean_error = poly_mean_error
                print(f"\nSelected polynomial correction: y = {poly_a:.6f}x² + {poly_b:.6f}x + {poly_c:.6f}")
            else:
                correction_type = "linear"
                params = [float(linear_a), float(linear_b)]
                predicted = linear_predicted
                mean_error = linear_mean_error
                print(f"\nSelected linear correction: y = {linear_a:.6f}x + {linear_b:.6f}")
            
            # Print detailed statistics
            print(f"Mean absolute error: {mean_error:.4f}m")
            print(f"Max error: {np.max(np.abs(predicted - true_distances)):.4f}m")
            
            # Show results for each data point
            print("\nCalibration results for each data point:")
            print("  True      Measured   Corrected  Error     Improvement")
            print("  --------------------------------------------------------")
            for i in range(len(true_distances)):
                true = true_distances[i]
                measured = measured_distances[i]
                corrected = predicted[i]
                orig_error = abs(measured - true)
                new_error = abs(corrected - true)
                improvement = orig_error - new_error
                print(f"  {true:.3f}m    {measured:.3f}m    {corrected:.3f}m    {new_error:.3f}m    {improvement:.3f}m")
            
            # Save to YAML
            calibration_data = {
                "depth_correction": {
                    "type": correction_type,
                    "parameters": params,
                    "mean_error": float(mean_error),
                    "calibration_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "num_samples": len(self.calibration_data),
                    "raw_data": [
                        {"true": float(t), "measured": float(m)} 
                        for t, m in self.calibration_data
                    ]
                }
            }
            
            # Save to file
            filename = "depth_camera_calibration.yaml"
            with open(filename, "w") as f:
                yaml.dump(calibration_data, f, default_flow_style=False)
            
            print(f"\nCalibration saved to {filename}")
            
            # Plot results without showing (save to file only)
            plt.figure(figsize=(10, 6))
            
            # Identity line
            min_val = min(np.min(measured_distances), np.min(true_distances))
            max_val = max(np.max(measured_distances), np.max(true_distances))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal (No Error)')
            
            # Original measurements
            plt.scatter(measured_distances, true_distances, c='blue', marker='o', label='Calibration Points')
            
            # Corrected values
            plt.scatter(measured_distances, predicted, c='red', marker='x', label='Corrected')
            
            plt.xlabel('Measured Distance (m)')
            plt.ylabel('True Distance (m)')
            plt.title('Depth Camera Calibration')
            plt.grid(True)
            plt.legend()
            
            plt.savefig('depth_calibration_plot.png')
            print("Plot saved to depth_calibration_plot.png")
            
            return True
            
        except Exception as e:
            print(f"Error computing calibration: {str(e)}")
            return False

def main(args=None):
    # Initialize ROS
    rclpy.init(args=args)
    calibrator = TerminalCalibrator()
    
    # Create a separate thread for ROS spinning
    spin_thread = threading.Thread(target=lambda: rclpy.spin(calibrator))
    spin_thread.daemon = True
    spin_thread.start()
    
    # Wait for depth connection
    print("Waiting for depth camera connection...")
    connection_wait = 0
    while calibrator.depth_msg_count == 0 and connection_wait < 50 and rclpy.ok():
        time.sleep(0.1)
        connection_wait += 1
    
    if calibrator.depth_msg_count == 0:
        print("\nERROR: Failed to connect to depth camera!")
        print(f"Check if topic '{calibrator.depth_topic}' exists and is publishing data")
        print("Use 'ros2 topic list' and 'ros2 topic echo <topic> -n 1' to debug")
        rclpy.shutdown()
        return
    
    print(f"\nConnected to depth camera! Received {calibrator.depth_msg_count} depth frames.")
    print("\n===== DEPTH CAMERA CALIBRATION =====")
    print("Place an object (wall, board, etc.) at known distances")
    print("from the front of the camera and take measurements.")
    print("Measure distances in meters (e.g., 0.5 for 50cm)")
    print("Recommended distances:")
    print("  - 20 inches (0.508 meters)")
    print("  - 30 inches (0.762 meters)")
    print("  - 40 inches (1.016 meters)")
    print("  - 60 inches (1.524 meters)")
    print("  - 80 inches (2.032 meters)")
    print("  - 100 inches (2.54 meters)")
    print("\nAim camera at a flat surface for each measurement.")
    
    # Calibration loop
    collecting = True
    while collecting and rclpy.ok():
        print("\nCalibration data:")
        if len(calibrator.calibration_data) > 0:
            for i, (true_dist, measured_dist) in enumerate(calibrator.calibration_data):
                error = measured_dist - true_dist
                print(f"{i+1}. True: {true_dist:.3f}m, Measured: {measured_dist:.3f}m, Error: {error:.3f}m")
        else:
            print("No data collected yet")
        
        print("\nOptions:")
        print("1. Take new measurement")
        print("2. Remove last measurement")
        print("3. Compute calibration")
        print("4. Exit")
        
        try:
            choice = int(input("\nSelect option (1-4): "))
            
            if choice == 1:
                # Take measurement
                try:
                    true_distance = float(input("Enter actual distance (meters): "))
                    if true_distance <= 0:
                        print("Distance must be positive")
                        continue
                    
                    print(f"Position object at exactly {true_distance:.3f}m...")
                    input("Press Enter when ready...")
                    
                    # Take measurement
                    print("Taking measurement...")
                    calibrator.ready_to_measure = True
                    calibrator.measurement_taken = False
                    calibrator.latest_measurement = None
                    
                    # Wait for measurement
                    wait_count = 0
                    while not calibrator.measurement_taken and wait_count < 30 and rclpy.ok():
                        time.sleep(0.1)
                        wait_count += 1
                        if wait_count % 10 == 0:
                            print(f"Still waiting... ({wait_count/10}s)")
                    
                    if not calibrator.measurement_taken:
                        print("ERROR: Failed to take measurement - timeout")
                        print("Check if depth camera is publishing data")
                        continue
                    
                    # Get result
                    if hasattr(calibrator, 'latest_measurement') and calibrator.latest_measurement is not None:
                        depth_value = calibrator.latest_measurement
                        calibrator.calibration_data.append((true_distance, depth_value))
                        print(f"Measurement added: True={true_distance:.3f}m, Measured={depth_value:.3f}m")
                    else:
                        print("Error: Failed to get valid measurement")
                except ValueError:
                    print("Invalid input - please enter a numeric value")
            
            elif choice == 2:
                if len(calibrator.calibration_data) > 0:
                    calibrator.calibration_data.pop()
                    print("Last measurement removed")
                else:
                    print("No data to remove")
            
            elif choice == 3:
                if calibrator.compute_calibration():
                    print("Calibration completed successfully")
                    collecting = False
                else:
                    print("Failed to compute calibration")
            
            elif choice == 4:
                print("Exiting calibration")
                collecting = False
            
            else:
                print("Invalid choice - please select 1-4")
        
        except ValueError:
            print("Invalid input - please enter a number")
    
    # Clean up
    rclpy.shutdown()
    print("\nCalibration complete!")

    if os.path.exists("depth_camera_calibration.yaml"):
        print("\nNext steps:")
        print("1. Add this to your depth_camera_node.py:")
        print("   ```python")
        print("   # In __init__ method:")
        print("   self.depth_corrector = DepthCorrector('depth_camera_calibration.yaml')")
        print("   ```")
        print("2. Then update the _get_reliable_depth method to apply correction:")
        print("   ```python")
        print("   # Old code:")
        print("   if center_value > 0:")
        print("       depth_m = float(center_value * self._scale_factor)")
        print("       if self._min_valid_depth < depth_m < self._max_valid_depth:")
        print("           return depth_m, 1.0, 1")
        print("   # New code:")
        print("   if center_value > 0:")
        print("       raw_depth_m = float(center_value * self._scale_factor)")
        print("       depth_m = self.depth_corrector.correct_depth(raw_depth_m)")
        print("       if self._min_valid_depth < depth_m < self._max_valid_depth:")
        print("           return depth_m, 1.0, 1")
        print("   ```")

if __name__ == "__main__":
    main()