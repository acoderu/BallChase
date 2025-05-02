#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
import time

class TransformReceiverNode(Node):
    def __init__(self):
        super().__init__('transform_receiver')
        
        # Create tf buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Add a short delay before first check (if enabled)
        self.use_initial_delay = True
        
        if self.use_initial_delay:
            self.get_logger().info("Waiting 3 seconds before checking transforms...")
            time.sleep(3.0)
        
        # Start periodic transform check timer
        self.check_timer = self.create_timer(1.0, self.check_transform)
        self.check_count = 0
        
        self.get_logger().info("Transform receiver started")
    
    def check_transform(self):
        """Check if transform from camera_frame to lidar_frame exists."""
        self.check_count += 1
        self.get_logger().info(f"Transform check #{self.check_count} at {self.get_clock().now().nanoseconds/1e9:.2f}s")
        
        # Try both directions
        try:
            when = rclpy.time.Time()
            timeout = rclpy.duration.Duration(seconds=0.2)
            
            # Camera frame to lidar frame
            cam_to_lidar = self.tf_buffer.can_transform(
                "camera_frame", 
                "lidar_frame", 
                when, 
                timeout=timeout
            )
            
            # Lidar frame to camera frame
            lidar_to_cam = self.tf_buffer.can_transform(
                "lidar_frame", 
                "camera_frame", 
                when, 
                timeout=timeout
            )
            
            if cam_to_lidar or lidar_to_cam:
                self.get_logger().info(f"✓ Transform found! camera→lidar: {cam_to_lidar}, lidar→camera: {lidar_to_cam}")
                
                # Get the actual transform details
                if cam_to_lidar:
                    transform = self.tf_buffer.lookup_transform(
                        "camera_frame", 
                        "lidar_frame", 
                        when
                    )
                    self.get_logger().info(
                        f"Transform details: translation=[{transform.transform.translation.x:.4f}, "
                        f"{transform.transform.translation.y:.4f}, {transform.transform.translation.z:.4f}]"
                    )
            else:
                self.get_logger().warn("✗ Transform not found in either direction")
                
                # Try to debug by listing all frames
                try:
                    frames = self.tf_buffer.all_frames_as_string()
                    if frames and frames.strip():
                        self.get_logger().info(f"Available frames:\n{frames}")
                    else:
                        self.get_logger().info("No frames available in transform buffer")
                except Exception as e:
                    self.get_logger().error(f"Error listing frames: {str(e)}")
                    
        except Exception as e:
            self.get_logger().error(f"Error checking transform: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = TransformReceiverNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()