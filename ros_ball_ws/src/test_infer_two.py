import rclpy  
from rclpy.node import Node  
import MNN  
import MNN.cv as mnn_cv2  
import MNN.numpy as mnn_np  
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge  
import cv2 as std_cv2  
import numpy as np  
import io  
import time  
import os  

# Pi 5 optimized configuration - fixed for container  
MODEL_PATH = "yolo12n_320.mnn"  
IMG_WIDTH = 320  
IMG_HEIGHT = 320  
PRECISION = "lowBF"  # Changed from lowBF  
BACKEND = "CPU"      # Force CPU backend since OpenCL isn't available  
THREAD_COUNT = 4     # Pi 5 has 4 cores  
NMS_THRESHOLD = 0.45  
CONFIDENCE_THRESHOLD = 0.25  
MAX_DETECTIONS = 100  

class MNNInferenceNode(Node):  
    def __init__(self):  
        super().__init__('mnn_inference_node')  
        
        self.subscription = self.create_subscription(  
            Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 10  
        )  

        self.bridge = CvBridge()  
        
        # Simplified configuration for container environment  
        config = {  
            "precision": PRECISION,  
            "backend": BACKEND,  
            "numThread": THREAD_COUNT,  
        }  
        
        self.get_logger().info(f"MNNInferenceNode initializing with {IMG_WIDTH}x{IMG_HEIGHT} image size.")  
        self.load_model(config)  

        self.start_time = time.time()  
        self.image_count = 0  
        
    def load_model(self, config):  
        try:  
            # Initialize model  
            self.runtime_manager = MNN.nn.create_runtime_manager((config,))  
            self.net = MNN.nn.load_module_from_file(MODEL_PATH, [], [], runtime_manager=self.runtime_manager)  
            
            # Create a dummy tensor with the correct shape for warmup  
            # CRITICAL: Make sure the shape is correct - need to check the model's expected format  
            # Most YOLO models expect RGB input in NCHW format (batch, channels, height, width)  
            dummy_image = np.zeros((3, IMG_HEIGHT, IMG_WIDTH), dtype=np.float32)  # NCHW format  
            dummy_tensor = MNN.expr.const(dummy_image, [3, IMG_HEIGHT, IMG_WIDTH], MNN.expr.NCHW)  
            dummy_tensor = MNN.expr.convert(dummy_tensor, MNN.expr.NC4HW4)  
            
            # Add batch dimension  
            dummy_input = MNN.expr.reshape(dummy_tensor, [1, 3, IMG_HEIGHT, IMG_WIDTH])  
            
            # Warm up  
            self.net.forward(dummy_input)  
            self.get_logger().info("Model loaded and warmed up successfully.")  
            
        except Exception as e:  
            self.get_logger().error(f"Model loading failed: {str(e)}")  
            raise  

    def image_callback(self, msg):  
        inference_start = time.time()  
        self.image_count += 1  
        
        try:  
            # Convert ROS image to OpenCV format  
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  
            
            # Resize if needed  
            if cv_image.shape[0] != IMG_HEIGHT or cv_image.shape[1] != IMG_WIDTH:  
                cv_image = std_cv2.resize(cv_image, (IMG_WIDTH, IMG_HEIGHT))  
            
            # Convert BGR to RGB   
            rgb_image = cv_image[..., ::-1]  
            
            # Convert to float and normalize [0,1]  
            rgb_image = rgb_image.astype(np.float32) * (1.0/255.0)  
            
            # CRITICAL: Transpose from HWC to CHW format (what most ML models expect)  
            # From (height, width, channels) to (channels, height, width)  
            chw_image = np.transpose(rgb_image, (2, 0, 1))  
            
            # Create MNN tensor from numpy array (explicitly specifying NCHW format)  
            input_tensor = MNN.expr.const(chw_image, [3, IMG_HEIGHT, IMG_WIDTH], MNN.expr.NCHW)  
            
            # Convert to NC4HW4 format used by MNN internally  
            input_tensor = MNN.expr.convert(input_tensor, MNN.expr.NC4HW4)  
            
            # Add batch dimension to get (1, 3, height, width)  
            input_tensor = MNN.expr.reshape(input_tensor, [1, 3, IMG_HEIGHT, IMG_WIDTH])  
            
            # Run inference  
            infer_start = time.time()  
            output_var = self.net.forward(input_tensor)  
            infer_time = (time.time() - infer_start) * 1000  

            # Post-processing  
            output_var = MNN.expr.convert(output_var, MNN.expr.NCHW).squeeze()  
            
            # Extract detection data  
            cx, cy = output_var[0], output_var[1]  
            w, h = output_var[2], output_var[3]  
            probs = output_var[4:]  

            # Convert box format  
            x0 = cx - w * 0.5  
            y0 = cy - h * 0.5  
            x1 = cx + w * 0.5  
            y1 = cy + h * 0.5  

            # Stack to boxes  
            boxes = mnn_np.stack([x0, y0, x1, y1], axis=1)  

            # Get scores and class IDs  
            scores = mnn_np.max(probs, axis=0)  
            class_ids = mnn_np.argmax(probs, axis=0)  

            # Filter by confidence threshold  
            valid_indices = mnn_np.where(scores > CONFIDENCE_THRESHOLD)[0]  
            
            # Process detections if any found  
            if len(valid_indices) > 0:  
                filtered_boxes = boxes[valid_indices]  
                filtered_scores = scores[valid_indices]  
                filtered_class_ids = class_ids[valid_indices]  
                
                # Apply NMS  
                result_ids = MNN.expr.nms(filtered_boxes, filtered_scores, MAX_DETECTIONS, NMS_THRESHOLD)  
                
                # Process results  
                for i in range(len(result_ids)):  
                    box = filtered_boxes[result_ids[i]]  
                    x0_val, y0_val, x1_val, y1_val = box.read_as_tuple()  
                    cls_id = filtered_class_ids[result_ids[i]]  
                    score = filtered_scores[result_ids[i]]  
    
                    self.get_logger().info(f"Detection: Class={cls_id}, Score={score:.2f}, "  
                          f"Box=({x0_val:.1f}, {y0_val:.1f}, {x1_val:.1f}, {y1_val:.1f})")  
            else:  
                self.get_logger().debug("No detections above threshold")  

            # Calculate FPS  
            total_time = (time.time() - inference_start) * 1000  
            elapsed_time = time.time() - self.start_time  
            fps = self.image_count / elapsed_time if elapsed_time > 0 else 0  

            self.get_logger().info(f"Inference={infer_time:.1f}ms, Total={total_time:.1f}ms, FPS={fps:.1f}")  

        except Exception as e:  
            self.get_logger().error(f"Error: {str(e)}")  
            import traceback  
            self.get_logger().error(traceback.format_exc())  

def main(args=None):  
    # Try to set CPU governor, but this might not work in a container  
    try:  
        os.system("echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null")  
        print("Set CPU to performance mode")  
    except:  
        print("Could not set CPU governor (expected in container)")  
    
    rclpy.init(args=args)  
    node = MNNInferenceNode()  
    
    print("MNNInferenceNode is running in container. Press Ctrl+C to exit.")  
    
    try:  
        rclpy.spin(node)  
    except KeyboardInterrupt:  
        print("Node stopped by keyboard interrupt")  
    except Exception as e:  
        print(f"Error during node execution: {str(e)}")  
        import traceback  
        print(traceback.format_exc())  
    finally:  
        node.destroy_node()  
        rclpy.shutdown()  
        print("Node shut down successfully")  

if __name__ == '__main__':  
    main()