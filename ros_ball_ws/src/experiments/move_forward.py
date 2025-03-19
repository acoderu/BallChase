import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class MoveRobot(Node):
    def __init__(self):
        super().__init__('move_robot')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        time.sleep(1)  # Allow publisher to register

    def move_forward(self, speed=0.2, distance=0.3048):  # 1 foot = 0.3048 meters
        msg = Twist()
        msg.linear.x = speed  # Move forward
        msg.angular.z = 0.0   # No rotation

        duration = distance / speed  # Time = Distance / Speed

        self.get_logger().info(f"Moving forward {distance} meters at {speed} m/s for {duration:.2f} seconds")

        start_time = time.time()
        while time.time() - start_time < duration:
            self.publisher_.publish(msg)
            time.sleep(0.1)  # Small delay to prevent overwhelming the topic

        # Stop the robot after moving the desired distance
        msg.linear.x = 0.0
        self.publisher_.publish(msg)
        self.get_logger().info("Stopped moving")

def main(args=None):
    rclpy.init(args=args)
    node = MoveRobot()
    node.move_forward()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
