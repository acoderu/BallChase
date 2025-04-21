#!/usr/bin/env python3

"""
Robot Rotation Test Script
=========================

This script commands a robot to perform a sequence of rotational movements
to verify how it interprets angular velocity commands:

1. Clockwise rotation
2. Counterclockwise rotation
3. Left turn 
4. Right turn

In ROS conventions:
- Positive angular velocity (z-axis) = counterclockwise/left turn
- Negative angular velocity (z-axis) = clockwise/right turn

Use this to verify that the robot's behavior matches expectations.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class RotationTestNode(Node):
    """
    A simple node to test robot rotation commands.
    """
    
    def __init__(self):
        """Initialize the node and set up publishers."""
        super().__init__('rotation_test')
        
        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/controller/cmd_vel',  # Use the same topic as your PID controller
            10
        )
        
        # Create a reusable command message
        self.cmd_vel_msg = Twist()
        
        # Start the test after a short delay
        self.create_timer(3.0, self.run_test_sequence)  # Increased initial delay to 3 seconds
        
        # Log startup
        self.get_logger().info("Rotation Test Node initialized")
        
    def send_velocity(self, linear_x, linear_y, angular_z, duration):
        """
        Send a velocity command for a specified duration.
        
        Args:
            linear_x (float): Linear velocity in X direction (m/s)
            linear_y (float): Linear velocity in Y direction (m/s)
            angular_z (float): Angular velocity around Z axis (rad/s)
            duration (float): How long to send the command (seconds)
        """
        # Update the command message
        self.cmd_vel_msg.linear.x = linear_x
        self.cmd_vel_msg.linear.y = linear_y
        self.cmd_vel_msg.angular.z = angular_z
        
        # Log what we're doing
        direction = "CLOCKWISE" if angular_z < 0 else "COUNTERCLOCKWISE" if angular_z > 0 else "NONE"
        text_direction = "RIGHT" if angular_z < 0 else "LEFT" if angular_z > 0 else "NONE"
        
        self.get_logger().info(
            f"Commanding rotation: {angular_z} rad/s ({direction}, {text_direction}) for {duration} seconds"
        )
        
        # Send command repeatedly over the duration
        start_time = time.time()
        while time.time() - start_time < duration:
            self.cmd_vel_pub.publish(self.cmd_vel_msg)
            time.sleep(0.1)  # 10 Hz update rate
            
        # Stop the robot after the duration
        self.stop_robot()
        
    def stop_robot(self):
        """Send a command to stop robot motion."""
        self.cmd_vel_msg.linear.x = 0.0
        self.cmd_vel_msg.linear.y = 0.0
        self.cmd_vel_msg.angular.z = 0.0
        self.cmd_vel_pub.publish(self.cmd_vel_msg)
        self.get_logger().info("Robot stopped")
        
    def run_test_sequence(self):
        """Run the full test sequence once."""
        # Cancel the timer to prevent repeated execution - fixed method
        for timer in self.timers:
            self.destroy_timer(timer)
        
        self.get_logger().info("Starting test sequence...")
        
        # 1. Clockwise rotation (negative angular velocity in ROS)
        self.get_logger().info("TEST 1: CLOCKWISE rotation (negative angular velocity)")
        self.send_velocity(0.0, 0.0, -0.5, 3.0)  # Rotate at -0.5 rad/s for 3 seconds
        time.sleep(3.0)  # Pause between tests
        
        # 2. Counterclockwise rotation (positive angular velocity in ROS)
        self.get_logger().info("TEST 2: COUNTERCLOCKWISE rotation (positive angular velocity)")
        self.send_velocity(0.0, 0.0, 0.5, 3.0)  # Rotate at 0.5 rad/s for 3 seconds
        time.sleep(3.0)  # Pause between tests
        
        # 3. Lateral movement left (positive Y in ROS)
        self.get_logger().info("TEST 3: LATERAL movement LEFT (positive Y)")
        self.send_velocity(0.0, 0.15, 0.0, 3.0)  # Move laterally left at 0.15 m/s
        time.sleep(3.0)  # Pause between tests
        
        # 4. Lateral movement right (negative Y in ROS)
        self.get_logger().info("TEST 4: LATERAL movement RIGHT (negative Y)")
        self.send_velocity(0.0, -0.15, 0.0, 3.0)  # Move laterally right at 0.15 m/s
        
        self.get_logger().info("Test sequence completed")
        
def main(args=None):
    """Main function to initialize and run the test node."""
    rclpy.init(args=args)
    node = RotationTestNode()
    
    # Welcome message
    print("=================================================")
    print("Robot Rotation Test")
    print("=================================================")
    print("This node will perform the following tests:")
    print("1. Clockwise rotation (negative angular velocity)")
    print("2. Counterclockwise rotation (positive angular velocity)")
    print("3. Left turn (should be positive angular velocity)")
    print("4. Right turn (should be negative angular velocity)")
    print("")
    print("Each test will run for 3 seconds with 2 seconds pause between.")
    print("Watch the robot to confirm if behavior matches expectations.")
    print("=================================================")
    
    # Set up signal handler for proper shutdown
    import signal
    def signal_handler(sig, frame):
        print(f"\nSignal {sig} received, stopping robot...")
        # Stop the robot before shutdown
        node.stop_robot()
        rclpy.shutdown()
        import sys
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Keyboard interrupt received, stopping robot...")
        node.stop_robot()
    finally:
        # Make sure the robot stops before shutdown
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()