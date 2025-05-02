#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
import time

class TransformPublisherNode(Node):
    def __init__(self):
        super().__init__('transform_publisher')
        
        # Create static transform broadcaster
        self.broadcaster = StaticTransformBroadcaster(self)
        
        # Publish immediately on startup
        self.publish_transform()
        
        # Create timer for periodic transform publishing
        self.timer = self.create_timer(5.0, self.publish_transform)
        
        self.get_logger().info("Transform publisher started")
    
    def publish_transform(self):
        """Publish static transform from camera_frame to lidar_frame."""
        # Create transform message
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "camera_frame"  # Parent frame
        transform.child_frame_id = "lidar_frame"    # Child frame
        
        # Set transform values (matching your existing calibration)
        transform.transform.translation.x = -0.0606
        transform.transform.translation.y = 0.0929
        transform.transform.translation.z = -0.0508
        
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0100
        transform.transform.rotation.w = 0.9999
        
        # IMPORTANT: StaticTransformBroadcaster.sendTransform expects a LIST of transforms
        self.broadcaster.sendTransform([transform])
        
        self.get_logger().info(f"Published transform: camera_frame -> lidar_frame at {self.get_clock().now().nanoseconds/1e9:.2f}s")

def main(args=None):
    rclpy.init(args=args)
    node = TransformPublisherNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()