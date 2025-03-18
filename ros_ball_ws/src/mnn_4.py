import rclpy
from rclpy.node import Node

# ---------------------------
# MNN imports
# ---------------------------
import MNN
import MNN.cv as mnn_cv       # MNN's version of cv2
import MNN.numpy as mnn_np    # MNN's version of numpy

# ---------------------------
# ROS/Bridge imports
# ---------------------------
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

import time

# Configuration
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
MODEL_PATH = "yolo12n_640.mnn"  # Path to the MNN model
PRECISION = "normal"            # Inference precision
BACKEND = "CPU"                 # Keep CPU as the backend
THREAD_COUNT = 4                # Number of threads for inference


class MNNObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('mnn_object_detection_node')

        # 1) Create subscription
        self.subscription = self.create_subscription(
            Image, '/ascamera/camera_publisher/rgb0/image', self.image_callback, 10
        )

        # 2) Set up runtime and model
        config = {
            "precision": PRECISION,
            "backend": BACKEND,
            "numThread": THREAD_COUNT
        }
        self.runtime_manager = MNN.nn.create_runtime_manager((config,))
        self.net = MNN.nn.load_module_from_file(MODEL_PATH, [], [], runtime_manager=self.runtime_manager)

        # 3) cv_bridge for converting ROS -> NumPy array
        self.bridge = CvBridge()

        # For simple performance tracking
        self.start_time = time.time()
        self.image_count = 0

        self.get_logger().info("MNNObjectDetectionNode initialized. Waiting for images...")

    def image_callback(self, msg):
        """Processes each incoming ROS image message."""
        start_time = time.time()
        self.image_count += 1

        try:
            # ----------------------------------------------------
            # A) Convert ROS message into a standard NumPy array
            # ----------------------------------------------------
            ros_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  
            # shape: (H, W, 3) in standard Python/NumPy

            # ----------------------------------------------------
            # B) Convert standard NumPy into MNN numpy
            # ----------------------------------------------------
            # MNN provides a helper to wrap a Python array:
            mnn_image = mnn_np.array(ros_bgr)  
            # shape: (H, W, 3) but as MNN numpy

            # ----------------------------------------------------
            # C) Resize with MNN's cv if desired
            # ----------------------------------------------------
            # (MNN.cv.resize expects an MNN numpy array, returns an MNN numpy array)
            mnn_image = mnn_cv.resize(mnn_image, (IMAGE_WIDTH, IMAGE_HEIGHT), mnn_cv.INTER_LINEAR)

            # ----------------------------------------------------
            # D) Convert BGR to RGB if your model expects RGB
            # ----------------------------------------------------
            # MNN doesn't provide a single call for BGR->RGB, so swap channels manually
            # shape is (H, W, 3); do mnn_image[..., ::-1]
            mnn_image = mnn_image[..., ::-1]

            # ----------------------------------------------------
            # E) Normalize from [0,255] to [0,1]
            # ----------------------------------------------------
            mnn_image = mnn_image.astype(mnn_np.float32) / 255.0

            # ----------------------------------------------------
            # F) Expand dims: (H, W, C) -> (1, H, W, C)
            # ----------------------------------------------------
            mnn_image = mnn_image.expand_dims(0)  # now shape: (1, H, W, 3)

            # ----------------------------------------------------
            # G) Convert to NC4HW4 format for MNN inference
            # ----------------------------------------------------
            input_tensor = MNN.expr.convert(mnn_image, MNN.expr.NC4HW4)

            # Preprocessing time
            pre_time = (time.time() - start_time) * 1000

            # ----------------------------------------------------
            # H) Run MNN inference
            # ----------------------------------------------------
            infer_start = time.time()
            output_var = self.net.forward(input_tensor)
            infer_time = (time.time() - infer_start) * 1000

            # ----------------------------------------------------
            # I) Post-processing steps (like your existing code)
            # ----------------------------------------------------
            post_start = time.time()
            output_var = MNN.expr.convert(output_var, MNN.expr.NCHW).squeeze()
            # Suppose the model outputs [84, num_boxes], or similarly.

            # 0..3 are box coords
            cx, cy, w, h = output_var[0], output_var[1], output_var[2], output_var[3]
            probs = output_var[4:]  # class probabilities

            # Convert from center coords to corners
            x0 = cx - w * 0.5
            y0 = cy - h * 0.5
            x1 = cx + w * 0.5
            y1 = cy + h * 0.5

            # Make a boxes array
            boxes = mnn_np.stack([x0, y0, x1, y1], axis=1)
            # Clip boxes
            h_out, w_out = IMAGE_HEIGHT, IMAGE_WIDTH
            boxes[:, 0] = mnn_np.clip(boxes[:, 0], 0, w_out)
            boxes[:, 1] = mnn_np.clip(boxes[:, 1], 0, h_out)
            boxes[:, 2] = mnn_np.clip(boxes[:, 2], 0, w_out)
            boxes[:, 3] = mnn_np.clip(boxes[:, 3], 0, h_out)

            # Confidence scores & class IDs
            scores = mnn_np.max(probs, axis=0)
            class_ids = mnn_np.argmax(probs, axis=0)

            # Non-Max Suppression
            result_ids = MNN.expr.nms(boxes, scores, 100, 0.45, 0.25)
            post_time = (time.time() - post_start) * 1000

            # ----------------------------------------------------
            # Printing results
            # ----------------------------------------------------
            for i in range(len(result_ids)):
                x0_val, y0_val, x1_val, y1_val = boxes[result_ids[i]].read_as_tuple()
                score_val = scores[result_ids[i]]
                cls_id_val = class_ids[result_ids[i]]

                print(f"Detection {i}: Class {cls_id_val}, Score {score_val:.2f}, "
                      f"x0={x0_val}, y0={y0_val}, x1={x1_val}, y1={y1_val}")

            total_time = (time.time() - start_time) * 1000
            elapsed_time = time.time() - self.start_time
            fps = self.image_count / elapsed_time if elapsed_time > 0 else 0

            print(f"Preproc time: {pre_time:.2f} ms, Inference time: {infer_time:.2f} ms, "
                  f"Postproc time: {post_time:.2f} ms, FPS: {fps:.2f}")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    node = MNNObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
