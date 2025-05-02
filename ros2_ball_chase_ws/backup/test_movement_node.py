#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import signal
import sys

class TestMovementNode(Node):
    def __init__(self):
        super().__init__('test_movement_node')
        # Changed to use the controller/cmd_vel topic which actually moves the robot
        self.publisher = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Test Movement Node initialized")
        self.get_logger().info("Publishing to /controller/cmd_vel")
        
        # Extended movement sequence to demonstrate Mecanum capabilities:
        # forward, stop, backward, stop, right turn, stop, left turn, stop,
        # strafe right, stop, strafe left, stop
        self.movement_state = 0
        self.state_change_time = time.time()
        self.pause_duration = 3.0  # seconds between movements
        self.total_states = 12  # Updated to include all movement states

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        self.get_logger().info("Signal handlers registered (Ctrl+C will safely stop the robot)")

    def timer_callback(self):
        current_time = time.time()
        time_in_state = current_time - self.state_change_time
        
        # Only change state if enough time has passed
        if time_in_state >= self.pause_duration:
            self.movement_state = (self.movement_state + 1) % self.total_states
            self.state_change_time = current_time
            
        twist = Twist()
        
        # Execute movement based on current state
        if self.movement_state == 0:
            # Move forward
            twist.linear.x = 0.15  # Forward within recommended range (-0.2 to 0.2)
            self.get_logger().info("Moving forward")
        elif self.movement_state == 1:
            # Stop (pause)
            self.get_logger().info("Pausing")
        elif self.movement_state == 2:
            # Move backward
            twist.linear.x = -0.15  # Backward within recommended range
            self.get_logger().info("Moving backward")
        elif self.movement_state == 3:
            # Stop (pause)
            self.get_logger().info("Pausing")
        elif self.movement_state == 4:
            # Turn right (clockwise)
            twist.angular.z = -0.2  # Within recommended range
            self.get_logger().info("Turning right (clockwise)")
        elif self.movement_state == 5:
            # Stop (pause)
            self.get_logger().info("Pausing")
        elif self.movement_state == 6:
            # Turn left (counterclockwise)
            twist.angular.z = 0.2  # Within recommended range
            self.get_logger().info("Turning left (counterclockwise)")
        elif self.movement_state == 7:
            # Stop (pause)
            self.get_logger().info("Pausing")
        elif self.movement_state == 8:
            # Strafe right (Mecanum specific)
            twist.linear.y = -0.15  # Negative Y moves right
            self.get_logger().info("Strafing right")
        elif self.movement_state == 9:
            # Stop (pause)
            self.get_logger().info("Pausing")
        elif self.movement_state == 10:
            # Strafe left (Mecanum specific)
            twist.linear.y = 0.15  # Positive Y moves left
            self.get_logger().info("Strafing left")
        elif self.movement_state == 11:
            # Stop (pause)
            self.get_logger().info("Pausing")
        
        self.publisher.publish(twist)
        self.get_logger().info(f"Published: lin_x={twist.linear.x}, lin_y={twist.linear.y}, ang_z={twist.angular.z}")

    def stop_robot(self):
        """Send stop commands to the robot"""
        stop_twist = Twist()
        # Set all motion commands to zero
        stop_twist.linear.x = 0.0
        stop_twist.linear.y = 0.0
        stop_twist.linear.z = 0.0
        stop_twist.angular.x = 0.0
        stop_twist.angular.y = 0.0
        stop_twist.angular.z = 0.0
        
        # Publish the stop command multiple times to ensure it's received
        self.get_logger().info("Emergency stop initiated")
        for _ in range(5):
            self.publisher.publish(stop_twist)
            time.sleep(0.1)
        self.get_logger().info("Robot stopped")

    def signal_handler(self, sig, frame):
        """Handle shutdown signals gracefully"""
        self.get_logger().info(f"Received termination signal {sig}")
        self.stop_robot()
        self.destroy_node()
        rclpy.shutdown()
        sys.exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = TestMovementNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # This block is kept for redundancy but the signal handler should catch SIGINT first
        node.get_logger().info("Keyboard interrupt received")
        node.stop_robot()
    finally:
        node.get_logger().info("Shutting down Test Movement Node")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()