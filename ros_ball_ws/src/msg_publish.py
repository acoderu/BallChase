import rclpy  
from rclpy.node import Node  
import cv2  
import numpy as np  
from sensor_msgs.msg import Image  
from std_msgs.msg import UInt8MultiArray, MultiArrayDimension  
from cv_bridge import CvBridge  

class ImageProcessorNode(Node):  
    def __init__(self):  
        super().__init__('image_processor_node')  

        self.subscription = self.create_subscription(  
            Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 10  
        )  
        
        self.publisher = self.create_publisher(UInt8MultiArray, '/processed_image_bytes', 10)  

        self.bridge = CvBridge()  
        self.get_logger().info("ImageProcessorNode initialized.")  

    def image_callback(self, msg):  
        try:  
            # Convert ROS image to OpenCV  
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  
            
            # Resize if needed  
            cv_image = cv2.resize(cv_image, (640, 640))  
            
            # Create ROS message  
            msg = UInt8MultiArray()  
            
            # Flatten the array and convert to list  
            msg.data = cv_image.flatten().tolist()  # Keep BGR format from OpenCV  
            
            # Attach metadata  
            msg.layout.dim.append(MultiArrayDimension(label="height", size=cv_image.shape[0]))  
            msg.layout.dim.append(MultiArrayDimension(label="width", size=cv_image.shape[1]))  
            msg.layout.dim.append(MultiArrayDimension(label="channels", size=cv_image.shape[2]))  
            
            # Publish the array  
            self.publisher.publish(msg)  
            self.get_logger().info("Published processed image data.")  
            
        except Exception as e:  
            self.get_logger().error(f"Error processing image: {str(e)}")  

def main(args=None):  
    rclpy.init(args=args)  
    node = ImageProcessorNode()  
    try:  
        rclpy.spin(node)  
    except KeyboardInterrupt:  
        pass  
    node.destroy_node()  
    rclpy.shutdown()  

if __name__ == '__main__':  
    main()