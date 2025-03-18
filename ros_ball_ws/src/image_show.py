import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import os

# Constants
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 640
IMAGE_SAVE_PATH = "ros2_camera_feed_640.jpg"  # Save location

class ObjectDetectionNode(Node):
    """
    ROS2 Node to capture and process images in headless mode.
    """

    def __init__(self):
        super().__init__('object_detection_node')

        # ROS2 Subscriber
        self.subscription = self.create_subscription(
            Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 10
        )

        # CV Bridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Debug message
        self.get_logger().info("ObjectDetectionNode initialized. Waiting for images...")

    def image_callback(self, msg):
        """Process incoming images and save them instead of displaying."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Resize the image
            resized_frame = cv2.resize(cv_image, (CAMERA_WIDTH, CAMERA_HEIGHT))

            # Save image instead of displaying
            cv2.imwrite(IMAGE_SAVE_PATH, resized_frame)
            self.get_logger().info(f"Image saved at {IMAGE_SAVE_PATH}")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
