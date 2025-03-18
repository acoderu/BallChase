import rclpy  
from rclpy.node import Node  
import MNN  
import MNN.cv as cv2  # MNN's version of cv2  
import MNN.numpy as np  # MNN's version of numpy  
from std_msgs.msg import UInt8MultiArray  
import time  
import numpy as orig_np  # Original numpy for initial conversion  
import os  
import tempfile  

MODEL_PATH = "yolo12n_640.mnn"  
PRECISION = "normal"  
BACKEND = "CPU"   
THREAD_COUNT = 4  

class MNNInferenceNode(Node):  
    def __init__(self):  
        super().__init__('mnn_inference_node')  

        self.subscription = self.create_subscription(  
            UInt8MultiArray, '/processed_image_bytes', self.image_callback, 10  
        )  

        config = {"precision": PRECISION, "backend": BACKEND, "numThread": THREAD_COUNT}  
        self.runtime_manager = MNN.nn.create_runtime_manager((config,))  
        self.net = MNN.nn.load_module_from_file(MODEL_PATH, [], [], runtime_manager=self.runtime_manager)  

        self.start_time = time.time()  
        self.image_count = 0  
        self.temp_dir = tempfile.mkdtemp()  
        self.temp_image_path = os.path.join(self.temp_dir, "temp_image.jpg")  
        self.get_logger().info(f"MNNInferenceNode initialized. Temp dir: {self.temp_dir}")  

    def image_callback(self, msg):  
        start_time = time.time()  
        self.image_count += 1  

        try:  
            # Extract shape information  
            height = int(msg.layout.dim[0].size)  
            width = int(msg.layout.dim[1].size)  
            channels = int(msg.layout.dim[2].size)  
            
            # Convert data to regular numpy and reshape to image  
            raw_data = orig_np.array(msg.data, dtype=orig_np.uint8)  
            image = raw_data.reshape((height, width, channels))  
            
            # Save the image to a temporary file  
            # Convert from RGB to BGR for OpenCV  
            image_bgr = image[..., ::-1] if channels == 3 else image  
            
            # Use standard OpenCV to save the image  
            import cv2 as std_cv2  
            std_cv2.imwrite(self.temp_image_path, image_bgr)  
            
            # Now use MNN's cv2 to read the image (this works in your working code)  
            original_image = cv2.imread(self.temp_image_path)  
            if original_image is None:  
                raise FileNotFoundError(f"MNN could not load image: {self.temp_image_path}")  
                
            # Convert BGR to RGB  
            image = original_image[..., ::-1]  # Convert BGR to RGB format  

            # Normalize image: [0, 255] to [0, 1]  
            image = image.astype(np.float32) / 255.0  
            
            # Expand dimensions and convert to MNN tensor format  
            input_var = np.expand_dims(image, 0)  
            input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)  
            
            # Run inference  
            infer_start = time.time()  
            output_var = self.net.forward(input_var)  
            infer_time = (time.time() - infer_start) * 1000  

            # Convert output to standard format  
            output_var = MNN.expr.convert(output_var, MNN.expr.NCHW).squeeze()  
            
            # Extract predictions  
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
            boxes = np.stack([x0, y0, x1, y1], axis=1)  

            # Get scores and class IDs  
            scores = np.max(probs, axis=0)  
            class_ids = np.argmax(probs, axis=0)  

            # Apply Non-Max Suppression (NMS)  
            result_ids = MNN.expr.nms(boxes, scores, 100, 0.45, 0.25)  

            # Print results  
            for i in range(len(result_ids)):  
                box = boxes[result_ids[i]]  
                x0_val, y0_val, x1_val, y1_val = box.read_as_tuple()  
                cls_id = class_ids[result_ids[i]]  
                score = scores[result_ids[i]]  

                self.get_logger().info(f"Detection: Class={cls_id}, Score={score:.2f}, "  
                      f"Box=({x0_val:.1f}, {y0_val:.1f}, {x1_val:.1f}, {y1_val:.1f})")  

            # Calculate FPS  
            elapsed_time = time.time() - self.start_time  
            fps = self.image_count / elapsed_time if elapsed_time > 0 else 0  
            post_time = (time.time() - start_time) * 1000  

            self.get_logger().info(f"Preproc: {post_time:.2f}ms, Infer: {infer_time:.2f}ms, FPS: {fps:.2f}")  

        except Exception as e:  
            self.get_logger().error(f"Error processing image: {str(e)}")  
            import traceback  
            self.get_logger().error(traceback.format_exc())  

    def __del__(self):  
        # Clean up temporary directory  
        import shutil  
        try:  
            shutil.rmtree(self.temp_dir)  
        except:  
            pass  

def main(args=None):  
    rclpy.init(args=args)  
    node = MNNInferenceNode()  
    try:  
        rclpy.spin(node)  
    except KeyboardInterrupt:  
        pass  
    node.destroy_node()  
    rclpy.shutdown()  

if __name__ == '__main__':  
    main()