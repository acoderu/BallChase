import rclpy
from rclpy.node import Node

# Standard OpenCV and NumPy
import cv2
import numpy as np

import MNN

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import time

# Config
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
MODEL_PATH = "yolo12n_640.mnn"
PRECISION = "normal"
BACKEND = "CPU"
THREAD_COUNT = 4
IMAGE_SAVE_PATH = "ros2_camera_feed_640.jpg"

class MNNObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('mnn_object_detection_node')
        self.subscription = self.create_subscription(
            Image,
            '/ascamera/camera_publisher/rgb0/image',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()

        # Prepare MNN runtime
        config = {
            "precision": PRECISION,
            "backend": BACKEND,
            "numThread": THREAD_COUNT
        }
        self.runtime_manager = MNN.nn.create_runtime_manager((config,))
        self.net = MNN.nn.load_module_from_file(MODEL_PATH, [], [], runtime_manager=self.runtime_manager)

        self.start_time = time.time()
        self.image_count = 0
        self.get_logger().info("MNNObjectDetectionNode initialized. Waiting for images...")

    def image_callback(self, msg):
        start_time = time.time()
        self.image_count += 1
        try:
            # 1) Convert ROS -> standard NumPy (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Optionally resize with standard cv2
            cv_image = cv2.resize(cv_image, (IMAGE_WIDTH, IMAGE_HEIGHT))

            # 2) (Optional) Save to file for debugging
            cv2.imwrite(IMAGE_SAVE_PATH, cv_image)

            # 3) Convert from BGR->RGB if your model expects RGB
            rgb_image = cv_image[..., ::-1]  # shape: (H, W, 3)

            # 4) Normalize to [0,1]
            rgb_image = rgb_image.astype(np.float32) / 255.0

            # 5) Add batch dimension: (1, H, W, C)
            input_array = np.expand_dims(rgb_image, axis=0)

            # 6) Convert to MNN tensor (NC4HW4)
            input_var = MNN.expr.convert(input_array, MNN.expr.NC4HW4)

            pre_time = (time.time() - start_time) * 1000

            # 7) Inference
            infer_start = time.time()
            output_var = self.net.forward(input_var)
            infer_time = (time.time() - infer_start) * 1000

            # 8) Post-processing
            post_start = time.time()
            # Convert output to NCHW, squeeze
            output_var = MNN.expr.convert(output_var, MNN.expr.NCHW).squeeze()
            # Suppose your model's shape is [84, num_boxes], same as your separate script
            cx, cy, w, h = output_var[0], output_var[1], output_var[2], output_var[3]
            probs = output_var[4:]

            # Convert center boxes to corners
            x0 = cx - w * 0.5
            y0 = cy - h * 0.5
            x1 = cx + w * 0.5
            y1 = cy + h * 0.5

            boxes = np.stack([x0, y0, x1, y1], axis=1)
            # clip to [0, W] or [0, H] if needed...
            # e.g. boxes[:, 0] = np.clip(boxes[:, 0], 0, IMAGE_WIDTH)
            # ...

            scores = np.max(probs, axis=0)
            class_ids = np.argmax(probs, axis=0)

            # Non-Max Suppression
            result_ids = MNN.expr.nms(boxes, scores, 100, 0.45, 0.25)

            post_time = (time.time() - post_start) * 1000

            # 9) Print results
            for i in range(len(result_ids)):
                # read_as_tuple is safe on MNN expression results
                # but here we used standard np in the final boxes stack, so just index
                x0_val, y0_val, x1_val, y1_val = boxes[result_ids[i]]
                cls_id_val = class_ids[result_ids[i]]
                score_val = scores[result_ids[i]]

                print(f"Detection {i}: Class={cls_id_val}, Score={score_val:.2f}, "
                      f"Box=({x0_val}, {y0_val}, {x1_val}, {y1_val})")

            total_time = (time.time() - start_time) * 1000
            elapsed_time = time.time() - self.start_time
            fps = self.image_count / elapsed_time if elapsed_time > 0 else 0

            print(f"Preproc: {pre_time:.2f}ms, Infer: {infer_time:.2f}ms, "
                  f"Postproc: {post_time:.2f}ms, FPS: {fps:.2f}")

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
