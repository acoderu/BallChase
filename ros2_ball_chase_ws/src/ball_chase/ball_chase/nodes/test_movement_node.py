#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class TestMovementNode(Node):
    def __init__(self):
        super().__init__('test_movement_node')
        # Changed to use the controller/cmd_vel topic which actually moves the robot
        self.publisher = self.create_publisher(Twist, '/controller/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Test Movement Node initialized")
        self.get_logger().info("Publishing to /controller/cmd_vel")
        
        # Movement sequence: forward, back, right 90 degrees, left 90 degrees, stop
        self.movement_state = 0
        self.state_change_time = time.time()
        self.pause_duration = 5.0  # seconds between movements

    def timer_callback(self):
        current_time = time.time()
        time_in_state = current_time - self.state_change_time
        
        # Only change state if enough time has passed
        if time_in_state >= self.pause_duration:
            self.movement_state = (self.movement_state + 1) % 6
            self.state_change_time = current_time
            
        twist = Twist()
        
        # Execute movement based on current state
        if self.movement_state == 0:
            # Move forward
            twist.linear.x = 0.2  # Max allowed is 0.2 m/s
            self.get_logger().info("Moving forward")
        elif self.movement_state == 1:
            # Stop (pause)
            self.get_logger().info("Pausing")
        elif self.movement_state == 2:
            # Move backward
            twist.linear.x = -0.2  # Max allowed is -0.2 m/s
            self.get_logger().info("Moving backward")
        elif self.movement_state == 3:
            # Turn 90 degrees right
            twist.angular.z = -0.5  # Max allowed is -0.5 rad/s
            self.get_logger().info("Turning right")
        elif self.movement_state == 4:
            # Turn 90 degrees left
            twist.angular.z = 0.5  # Max allowed is 0.5 rad/s
            self.get_logger().info("Turning left")
        elif self.movement_state == 5:
            # Stop
            self.get_logger().info("Stopping")
        
        self.publisher.publish(twist)
        self.get_logger().info(f"Published: lin_x={twist.linear.x}, lin_y={twist.linear.y}, ang_z={twist.angular.z}")

def main(args=None):
    rclpy.init(args=args)
    node = TestMovementNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Send a stop command before shutting down
        stop_twist = Twist()
        stop_twist.linear.x = 0.0
        stop_twist.linear.y = 0.0
        stop_twist.linear.z = 0.0
        stop_twist.angular.x = 0.0
        stop_twist.angular.y = 0.0
        stop_twist.angular.z = 0.0
        
        # Publish the stop command multiple times to ensure it's received
        for _ in range(5):
            node.publisher.publish(stop_twist)
            time.sleep(0.1)
            
        node.get_logger().info("Emergency stop sent. Shutting down Test Movement Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()