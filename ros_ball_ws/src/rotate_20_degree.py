import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class RotateRobot(Node):
    def __init__(self):
        super().__init__('rotate_robot')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        time.sleep(1)  # Allow publisher to register

    def rotate_clockwise(self, angular_speed=0.5, angle=20):
        """ Rotates the robot clockwise by a given angle. """
        msg = Twist()
        msg.angular.z = -abs(angular_speed)  # Clockwise rotation (negative value)

        # Convert degrees to radians (ROS uses radians)
        angle_radians = angle * (3.14159265359 / 180.0)
        duration = angle_radians / abs(angular_speed)  # Time = Angle / Speed

        self.get_logger().info(f"Rotating clockwise {angle} degrees at {angular_speed} rad/s for {duration:.2f} seconds")

        start_time = time.time()
        while time.time() - start_time < duration:
            self.publisher_.publish(msg)
            time.sleep(0.1)  # Small delay to prevent overwhelming the topic

        # Stop rotation
        msg.angular.z = 0.0
        self.publisher_.publish(msg)
        self.get_logger().info("Stopped rotating")

def main(args=None):
    rclpy.init(args=args)
    node = RotateRobot()
    node.rotate_clockwise()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
