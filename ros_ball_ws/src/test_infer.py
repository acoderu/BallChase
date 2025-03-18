import rclpy  
from rclpy.node import Node  
import MNN  
import MNN.cv as mnn_cv2  # MNN's version of cv2  
import MNN.numpy as mnn_np  # MNN's version of numpy  
from sensor_msgs.msg import Image  
from cv_bridge import CvBridge  
import cv2 as std_cv2  # Standard OpenCV  
import os  
import tempfile  
import time  

MODEL_PATH = "yolo12n_640.mnn"  
PRECISION = "normal"  
BACKEND = "CPU"   
THREAD_COUNT = 4  

class MNNInferenceNode(Node):  
    def __init__(self):  
        super().__init__('mnn_inference_node')  

        # Direct subscription to camera topic  
        self.subscription = self.create_subscription(  
            Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 10  
        )  

        self.bridge = CvBridge()  
        self.temp_dir = tempfile.mkdtemp()  
        self.temp_image_path = os.path.join(self.temp_dir, "temp_image.jpg")  
        
        # MNN model setup  
        config = {"precision": PRECISION, "backend": BACKEND, "numThread": THREAD_COUNT}  
        self.runtime_manager = MNN.nn.create_runtime_manager((config,))  
        self.net = MNN.nn.load_module_from_file(MODEL_PATH, [], [], runtime_manager=self.runtime_manager)  

        self.start_time = time.time()  
        self.image_count = 0  
        self.get_logger().info("MNNInferenceNode initialized.")  

    def image_callback(self, msg):  
        start_time = time.time()  
        self.image_count += 1  
        
        try:  
            # Convert ROS image to OpenCV format using cv_bridge  
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  
            
            # Resize if needed  
            cv_image = std_cv2.resize(cv_image, (640, 640))  
            
            # Save temporarily with standard OpenCV  
            std_cv2.imwrite(self.temp_image_path, cv_image)  
            
            # Read with MNN's cv2  
            mnn_image = mnn_cv2.imread(self.temp_image_path)  
            if mnn_image is None:  
                raise FileNotFoundError(f"MNN could not load image: {self.temp_image_path}")  
            
            # Convert BGR to RGB for the model  
            mnn_image = mnn_image[..., ::-1]  
            
            # Normalize to [0,1]  
            mnn_image = mnn_image.astype(mnn_np.float32) / 255.0  
            
            # Add batch dimension and convert  
            input_var = mnn_np.expand_dims(mnn_image, 0)  
            input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)  
            
            # Run inference  
            infer_start = time.time()  
            output_var = self.net.forward(input_var)  
            infer_time = (time.time() - infer_start) * 1000  

            # Convert output to standard format  
            output_var = MNN.expr.convert(output_var, MNN.expr.NCHW).squeeze()  
            
            # Extract detection data  
            cx = output_var[0]  
            cy = output_var[1]  
            w = output_var[2]  
            h = output_var[3]  
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

            # Apply Non-Max Suppression (NMS)  
            result_ids = MNN.expr.nms(boxes, scores, 100, 0.45, 0.25)  

            # Print results  
            ih, iw = 640, 640  # Image dimensions  
            for i in range(len(result_ids)):  
                box = boxes[result_ids[i]]  
                x0_val, y0_val, x1_val, y1_val = box.read_as_tuple()  
                cls_id = class_ids[result_ids[i]]  
                score = scores[result_ids[i]]  

                self.get_logger().info(f"Detection: Class={cls_id}, Score={score:.2f}, "  
                      f"Box=({x0_val:.1f}, {y0_val:.1f}, {x1_val:.1f}, {y1_val:.1f})")  
                
                # Optional: Draw detection on the original image  
                x0_int, y0_int = max(0, int(x0_val)), max(0, int(y0_val))  
                x1_int, y1_int = min(iw, int(x1_val)), min(ih, int(y1_val))  
                std_cv2.rectangle(cv_image, (x0_int, y0_int), (x1_int, y1_int), (0, 255, 0), 2)  

            # Calculate FPS  
            elapsed_time = time.time() - self.start_time  
            fps = self.image_count / elapsed_time if elapsed_time > 0 else 0  
            post_time = (time.time() - start_time) * 1000  

            self.get_logger().info(f"Preproc: {post_time:.2f}ms, Infer: {infer_time:.2f}ms, FPS: {fps:.2f}")  
            
            # Optional: Save the image with detections   
            # std_cv2.imwrite("detections.jpg", cv_image)  

        except Exception as e:  
            self.get_logger().error(f"Error processing image: {str(e)}")  
            import traceback  
            self.get_logger().error(traceback.format_exc())  

    def __del__(self):  
        # Clean up temporary directory  
        import shutil  
        try:  
            shutil.rmtree(self.temp_dir)  
            self.get_logger().info(f"Cleaned up temporary directory: {self.temp_dir}")  
        except Exception as e:  
            self.get_logger().error(f"Error cleaning up: {str(e)}")  

def main(args=None):  
    rclpy.init(args=args)  
    node = MNNInferenceNode()  
    
    #self.get_logger().info("MNNInferenceNode is running. Press Ctrl+C to exit.")  
    
    try:  
        rclpy.spin(node)  
    except KeyboardInterrupt:  
        #self.get_logger().info("Node stopped by keyboard interrupt")  
        i=1
    except Exception as e:  
        #self.get_logger().error(f"Error during node execution: {str(e)}")  
        import traceback  
        #self.get_logger().error(traceback.format_exc())  
    finally:  
        # Clean up  
        node.destroy_node()  
        rclpy.shutdown()  
        #self.get_logger().info("Node shut down successfully")  

if __name__ == '__main__':  
    main()