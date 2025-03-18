import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import ncnn  # Import NCNN for inference
import time

# Constants
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 320
MODEL_PARAM_PATH = "yolo12n_ncnn_model_320/model.ncnn.param"
MODEL_BIN_PATH = "yolo12n_ncnn_model_320/model.ncnn.bin"

class ObjectDetectionNode(Node):
    """
    ROS2 Node to run YOLO NCNN inference on camera frames with extensive debug logging.
    """

    def __init__(self):
        super().__init__('object_detection_node')

        self.get_logger().info("Initializing ObjectDetectionNode...")

        # ROS2 Subscriber
        self.subscription = self.create_subscription(
            Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 10
        )

        self.get_logger().info("ROS2 subscription set up.")
        
        # CV Bridge to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Load YOLO NCNN Model
        self.get_logger().info("Loading YOLO NCNN model...")
        self.net = self.load_ncnn_model()
        if self.net is None:
            self.get_logger().error("NCNN model failed to load.")
        else:
            self.get_logger().info("Model loaded successfully.")

    def load_ncnn_model(self):
        """ Load the YOLO NCNN model with detailed debug logging """
        try:
            net = ncnn.Net()
            net.opt.use_vulkan_compute = False
            net.opt.num_threads = 1
            net.opt.lightmode = True
            self.get_logger().info(f"Loading NCNN model param: {MODEL_PARAM_PATH}")
            
            if net.load_param(MODEL_PARAM_PATH) != 0:
                self.get_logger().error(f"Failed to load model param: {MODEL_PARAM_PATH}")
                return None

            self.get_logger().info(f"Loading NCNN model bin: {MODEL_BIN_PATH}")
            
            if net.load_model(MODEL_BIN_PATH) != 0:
                self.get_logger().error(f"Failed to load model bin: {MODEL_BIN_PATH}")
                return None
            
            return net
        except Exception as e:
            self.get_logger().error(f"Error loading NCNN model: {str(e)}")
            return None

    def preprocess_image(self, cv_image):
        """ Preprocess input image: resize, normalize, and format for NCNN """
        try:
            self.get_logger().info("Preprocessing image...")
            start_time = time.time()
            
            resized_frame = cv2.resize(cv_image, (CAMERA_WIDTH, CAMERA_HEIGHT))
            resized_frame = resized_frame.astype(np.float32) #/ 255.0  # Normalize
            resized_frame = np.transpose(resized_frame, (2, 0, 1))  # Convert to CHW format
            
            elapsed_time = time.time() - start_time
            self.get_logger().info(f"Processed image shape: {resized_frame.shape}, Time taken: {elapsed_time:.4f}s")

            return ncnn.Mat(resized_frame)
        except Exception as e:
            self.get_logger().error(f"Error in image preprocessing: {str(e)}")
            return None

    def image_callback(self, msg):
        """Process incoming images and run YOLO NCNN inference."""
        try:
            print("got image...")
            self.get_logger().info("Received image message.")
            start_time_total = time.time()
            
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.get_logger().info(f"Converted ROS image to OpenCV format, shape: {cv_image.shape}")
            print("A")
            # Preprocess image for NCNN
            ncnn_input = self.preprocess_image(cv_image)
            print("B")
            if ncnn_input is None:
                self.get_logger().error("Preprocessing failed, skipping inference.")
                return

            if self.net is None:
                self.get_logger().error("NCNN model not loaded, skipping inference.")
                return
            
            self.get_logger().info("Running NCNN inference...")
            start_time_inference = time.time()
            
            with self.net.create_extractor() as ex:
                print("ncnn_input shape:", ncnn_input.w, ncnn_input.h, ncnn_input.c)
                self.get_logger().info("Created NCNN extractor.")
                ex.input("in0", ncnn_input)
                self.get_logger().info("NCNN input set.")
                
                ret, out0 = ex.extract("out0")
                if ret != 0:
                    self.get_logger().error(f"Extract out0 failed with code {ret}")
                    return
                self.get_logger().info("NCNN inference completed.")
                
            elapsed_time_inference = time.time() - start_time_inference
            self.get_logger().info(f"Inference time: {elapsed_time_inference:.4f}s")

            # Parse detection results
            detections = self.parse_ncnn_output(out0)
            
            elapsed_time_total = time.time() - start_time_total
            self.get_logger().info(f"Total image processing time: {elapsed_time_total:.4f}s")

            # Print detected objects
            for det in detections:
                class_id, confidence, bbox = det
                self.get_logger().info(f"Detected: Class {class_id}, Confidence {confidence:.2f}, BBox {bbox}")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def parse_ncnn_output(self, out0):
        """ Parse NCNN YOLO output to extract bounding boxes and class IDs """
        try:
            self.get_logger().info("Parsing NCNN output...")
            start_time = time.time()
            
            results = []
            out_array = np.array(out0)  # Convert NCNN output to NumPy
            self.get_logger().info(f"NCNN output shape: {out_array.shape}")

            # Each row contains: [x_min, y_min, x_max, y_max, confidence, class_id]
            for row in out_array:
                bbox = row[:4].tolist()
                confidence = row[4]
                class_id = int(row[5])
                if confidence > 0.5:  # Confidence threshold
                    results.append((class_id, confidence, bbox))
            
            elapsed_time = time.time() - start_time
            self.get_logger().info(f"Detections parsed: {len(results)} objects found, Time taken: {elapsed_time:.4f}s")
            return results

        except Exception as e:
            self.get_logger().error(f"Error parsing NCNN output: {str(e)}")
            return []


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
