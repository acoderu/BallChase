"""
Tennis Ball Tracking Robot - YOLO Detection Node
================================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving tennis ball.
The system uses multiple sensing modalities for robust detection:
- YOLO object detection (this node)
- HSV color-based detection
- LiDAR for depth sensing
- Depth camera for additional depth information

These different sensing modalities are combined through sensor fusion to provide reliable
tracking even when one sensing method may fail.

This Node's Purpose:
------------------
This specific node uses a lightweight YOLO neural network to detect tennis balls in camera
images. The node processes raw camera frames, runs them through a pre-trained YOLO model,
and publishes the detected ball's position with a confidence score.

Coordinate System:
----------------
- Input images are resized to 320x320 pixels
- Output coordinates are in the same 320x320 image space
- (0,0) represents the top-left corner of the image
- Published positions include:
  - x: horizontal position (0-320)
  - y: vertical position (0-320)
  - z: confidence score (0-1)

Part of the Pipeline:
-------------------
1. Multiple detection nodes run in parallel (YOLO, HSV, depth)
2. Sensor fusion node combines detections and handles conflicting information
3. State manager determines robot behavior based on ball position/movement
4. PID controller drives the motors to follow the ball

Dependencies:
-----------
- ROS2 Humble
- MNN (for efficient neural network inference)
- OpenCV
- Numpy
"""

import rclpy
from rclpy.node import Node
import MNN
import MNN.cv as mnn_cv2
import MNN.numpy as mnn_np
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String  # Add String message type
from cv_bridge import CvBridge
import cv2 as std_cv2
import numpy as np
import time
import os
from config.config_loader import ConfigLoader  # Import ConfigLoader
from ball_tracking.resource_monitor import ResourceMonitor  # Add resource monitoring import
from collections import deque  # Add import for deque
import json

# Load configuration
config_loader = ConfigLoader()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'yolo_config.yaml')
config = config_loader.load_yaml(config_path)

# Model configuration from config file
MODEL_CONFIG = config.get('model', {
    "path": "yolo12n_320.mnn",    # Path to our YOLO model file
    "input_width": 320,           # Width our model expects
    "input_height": 320,          # Height our model expects
    "precision": "lowBF",         # Lower precision for faster inference
    "backend": "CPU",             # Using CPU for inference
    "thread_count": 4,            # Number of CPU threads to use
    "confidence_threshold": 0.25  # Only keep detections above this confidence
})

# COCO dataset class ID for "sports ball" - this includes tennis balls
TENNIS_BALL_CLASS_ID = config.get('model', {}).get('tennis_ball_class_id', 32)

# Topic configuration from config file
TOPICS = config.get('topics', {
    "input": {
        "camera": "/ascamera/camera_publisher/rgb0/image"
    },
    "output": {
        "position": "/tennis_ball/yolo/position"
    }
})

# Diagnostic configuration
DIAG_CONFIG = config.get('diagnostics', {
    "log_interval": 15,
    "performance_log_interval": 30
})

class TennisBallDetector(Node):
    """
    A ROS2 node that uses a YOLO neural network to detect tennis balls
    in camera images and publishes their positions with timestamp information.
    
    This node subscribes to camera images, performs tennis ball detection using
    a YOLO neural network model, and publishes the detected ball position
    along with confidence information.
    
    Published Topics:
    - /tennis_ball/yolo/position (geometry_msgs/PointStamped): 
      The detected ball position with timestamp
      
    Subscribed Topics:
    - /ascamera/camera_publisher/rgb0/image (sensor_msgs/Image): 
      RGB camera feed
    """
    
    def __init__(self):
        """Initialize the tennis ball detector node."""
        # Initialize our ROS node
        super().__init__('tennis_ball_detector')
        
        # Add Raspberry Pi resource monitoring
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=15.0,  # Less frequent to reduce overhead
            enable_temperature=True
        )
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        
        # Use higher thresholds since YOLO is compute-intensive
        self.resource_monitor.set_threshold('cpu', 90.0)
        self.resource_monitor.set_threshold('memory', 90.0)
        self.resource_monitor.start()
        
        # Enable low power mode for Raspberry Pi if available in config
        self.low_power_mode = config.get('raspberry_pi', {}).get('low_power_mode', False)
        if self.low_power_mode:
            self.get_logger().info("YOLO detector running in low power mode for Raspberry Pi")
        
        # Subscribe to the camera feed
        self.subscription = self.create_subscription(
            Image, 
            TOPICS["input"]["camera"], 
            self.image_callback, 
            10
        )  

        # Create a publisher for tennis ball coordinates
        # Using PointStamped to include timestamp and frame information
        self.ball_publisher = self.create_publisher(
            PointStamped, 
            TOPICS["output"]["position"], 
            10
        )  

        # Create a publisher for system diagnostics
        self.system_diagnostics_publisher = self.create_publisher(
            String,
            "/tennis_ball/yolo/diagnostics",  # Changed from "/system/diagnostics/yolo"
            10
        )
        
        # Timer for publishing diagnostics
        self.diagnostics_timer = self.create_timer(3.0, self.publish_system_diagnostics)

        # Bridge to convert between ROS images and OpenCV images
        self.bridge = CvBridge()
        
        # Configuration for the MNN model
        self.mnn_config = {
            "precision": MODEL_CONFIG["precision"],
            "backend": MODEL_CONFIG["backend"],
            "numThread": MODEL_CONFIG["thread_count"],
        }
        
        self.get_logger().info("YOLO Tennis Ball Detector starting up!")
        
        # Load the YOLO model
        self.load_model(self.mnn_config)

        # Performance tracking variables
        self.start_time = time.time()
        self.image_count = 0
        
        # Initialize diagnostic metrics
        self._init_diagnostic_metrics()
        
        self.get_logger().info("Initialization complete, waiting for camera images")
        
    def _init_diagnostic_metrics(self):
        """Initialize metrics for diagnostic monitoring"""
        self.diagnostic_metrics = {
            'fps_history': deque(maxlen=10),
            'processing_time_history': deque(maxlen=10),
            'inference_time_history': deque(maxlen=10),
            'detection_rate_history': deque(maxlen=10),
            'confidence_history': deque(maxlen=20),
            'last_detection_position': None,
            'last_detection_time': 0.0,
            'total_frames': 0,
            'detected_frames': 0,
            'errors': [],
            'warnings': []
        }

    def load_model(self, config):
        """
        Load the YOLO model for tennis ball detection.
        
        Args:
            config (dict): Configuration parameters for the MNN runtime
        """
        try:
            self.get_logger().info(f"Loading model from {MODEL_CONFIG['path']}...")
            
            # Initialize MNN runtime manager with our configuration
            self.runtime_manager = MNN.nn.create_runtime_manager((config,))
            
            # Load the YOLO model from file
            self.net = MNN.nn.load_module_from_file(
                MODEL_CONFIG["path"], [], [], runtime_manager=self.runtime_manager
            )
            
            # Create a test image to warm up the model
            # This helps the first real detection run faster
            dummy_image = np.zeros(
                (3, MODEL_CONFIG["input_height"], MODEL_CONFIG["input_width"]), 
                dtype=np.float32
            )
            dummy_tensor = MNN.expr.const(
                dummy_image, 
                [3, MODEL_CONFIG["input_height"], MODEL_CONFIG["input_width"]], 
                MNN.expr.NCHW
            )
            dummy_tensor = MNN.expr.convert(dummy_tensor, MNN.expr.NC4HW4)
            dummy_input = MNN.expr.reshape(
                dummy_tensor, 
                [1, 3, MODEL_CONFIG["input_height"], MODEL_CONFIG["input_width"]]
            )
            
            # Run the model once to warm it up
            self.net.forward(dummy_input)
            self.get_logger().info("YOLO tennis ball detector ready!")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            raise

    def preprocess_image(self, cv_image):
        """
        Preprocess the camera image for YOLO inference.
        
        Args:
            cv_image (numpy.ndarray): Raw OpenCV image in BGR format
            
        Returns:
            MNN.expr.Var: Preprocessed image tensor ready for model inference
        """
        # Resize the image to what our model expects
        if (cv_image.shape[0] != MODEL_CONFIG["input_height"] or 
            cv_image.shape[1] != MODEL_CONFIG["input_width"]):
            cv_image = std_cv2.resize(
                cv_image, 
                (MODEL_CONFIG["input_width"], MODEL_CONFIG["input_height"])
            )
        
        # Convert from BGR (OpenCV format) to RGB (what our model expects)
        rgb_image = cv_image[..., ::-1]
        
        # Normalize pixel values to [0,1] range
        rgb_image = rgb_image.astype(np.float32) * (1.0/255.0)
        
        # Change image format from HWC to CHW
        # HWC = Height, Width, Channels
        # CHW = Channels, Height, Width (what neural networks typically expect)
        chw_image = np.transpose(rgb_image, (2, 0, 1))
        
        # Create an MNN tensor from our image
        input_tensor = MNN.expr.const(
            chw_image, 
            [3, MODEL_CONFIG["input_height"], MODEL_CONFIG["input_width"]], 
            MNN.expr.NCHW
        )
        input_tensor = MNN.expr.convert(input_tensor, MNN.expr.NC4HW4)
        input_tensor = MNN.expr.reshape(
            input_tensor, 
            [1, 3, MODEL_CONFIG["input_height"], MODEL_CONFIG["input_width"]]
        )
        
        return input_tensor

    def process_detections(self, output_var):
        """
        Process YOLO output to extract tennis ball detections.
        
        Args:
            output_var (MNN.expr.Var): Raw output from YOLO model
            
        Returns:
            tuple: (best_box, confidence) if tennis ball found, else (None, 0)
                - best_box: [x0, y0, x1, y1] coordinates of the detection box
                - confidence: final confidence score (0-1)
        """
        # Convert model output to NCHW format and remove batch dimension
        output_var = MNN.expr.convert(output_var, MNN.expr.NCHW).squeeze()
        
        # Extract detection data
        cx, cy = output_var[0], output_var[1]  # Center coordinates
        w, h = output_var[2], output_var[3]    # Width and height
        probs = output_var[4:]                 # Class probabilities
        
        # Convert from center format to corner format
        # (x0,y0) is top-left corner, (x1,y1) is bottom-right corner
        x0 = cx - w * 0.5  # Left side of box
        y0 = cy - h * 0.5  # Top of box
        x1 = cx + w * 0.5  # Right side of box
        y1 = cy + h * 0.5  # Bottom of box
        
        # Combine into boxes array
        boxes = mnn_np.stack([x0, y0, x1, y1], axis=1)
        
        # Get confidence scores and class IDs for each detection
        scores = mnn_np.max(probs, axis=0)         # Highest probability for any class
        class_ids = mnn_np.argmax(probs, axis=0)   # Which class has highest probability
        
        # Find all tennis ball detections with confidence above threshold
        tennis_ball_indices = []
        for i in range(len(class_ids)):
            if (class_ids[i] == TENNIS_BALL_CLASS_ID and 
                scores[i] > MODEL_CONFIG["confidence_threshold"]):
                tennis_ball_indices.append(i)
        
        # If no tennis balls found, return None
        if not tennis_ball_indices:
            return None, 0.0
            
        # If multiple tennis balls detected, take the one with highest confidence
        best_idx = tennis_ball_indices[0]
        for idx in tennis_ball_indices:
            if scores[idx] > scores[best_idx]:
                best_idx = idx
        
        # Get the box coordinates for our best detection
        box = boxes[best_idx]
        x0_val, y0_val, x1_val, y1_val = box.read_as_tuple()
        
        # Calculate confidence adjustments
        base_confidence = scores[best_idx]
        
        # Adjust confidence based on aspect ratio (tennis balls should be round)
        width = x1_val - x0_val
        height = y1_val - y0_val
        aspect_ratio = width / height if height > 0 else 1.0
        size_confidence = 1.0 - abs(1.0 - aspect_ratio) * 0.5
        
        # Final confidence combines model confidence and size confidence
        final_confidence = base_confidence * size_confidence
        
        best_box = [x0_val, y0_val, x1_val, y1_val]
        return best_box, final_confidence

    def image_callback(self, msg):
        """
        Process each incoming camera image to detect tennis balls.
        
        Args:
            msg (sensor_msgs.msg.Image): The incoming camera image from ROS
        """
        # Skip frames in low power mode to reduce CPU usage
        if hasattr(self, 'frame_counter'):
            self.frame_counter += 1
        else:
            self.frame_counter = 0
            
        if self.low_power_mode and self.frame_counter % 2 != 0:
            # Process only every other frame in low power mode
            return

        inference_start = time.time()
        self.image_count += 1
        
        # Update total frames in diagnostic metrics
        self.diagnostic_metrics['total_frames'] = self.image_count
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Preprocess the image for model input
            input_tensor = self.preprocess_image(cv_image)
            
            # Run the neural network to detect objects
            infer_start = time.time()
            output_var = self.net.forward(input_tensor)
            infer_time = (time.time() - infer_start) * 1000  # milliseconds
            
            # Track inference time for diagnostics
            self.diagnostic_metrics['inference_time_history'].append(infer_time)
            
            # Process the model output to find tennis balls
            best_box, confidence = self.process_detections(output_var)
            
            # If a tennis ball was detected, publish its position
            if best_box is not None:
                x0, y0, x1, y1 = best_box
                
                # Calculate center point of the tennis ball
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2
                width = x1 - x0
                height = y1 - y0
                
                # Update diagnostic metrics for detection
                self.diagnostic_metrics['detected_frames'] += 1
                self.diagnostic_metrics['last_detection_position'] = (center_x, center_y)
                self.diagnostic_metrics['last_detection_time'] = time.time()
                self.diagnostic_metrics['confidence_history'].append(confidence)
                
                # Calculate detection rate
                detection_rate = self.diagnostic_metrics['detected_frames'] / self.image_count
                self.diagnostic_metrics['detection_rate_history'].append(detection_rate)
                
                # Log significant detections to avoid console flooding
                if self.image_count % DIAG_CONFIG["log_interval"] == 0 or confidence > 0.6:
                    self.get_logger().info(
                        f"YOLO detected ball: ({center_x:.1f}, {center_y:.1f}), "
                        f"Size: {width:.1f}x{height:.1f}, Confidence: {confidence:.2f}"
                    )
                
                # Create and publish the position message with timestamp
                position_msg = PointStamped()
                
                # IMPORTANT: Copy the timestamp from the original image
                # This is critical for fusion to know when this detection occurred
                position_msg.header.stamp = msg.header.stamp
                position_msg.header.frame_id = "camera_frame"  # Use consistent frame ID
                
                # Add sequence number for better synchronization
                if not hasattr(self, 'seq_counter'):
                    self.seq_counter = 0
                self.seq_counter += 1
                position_msg.header.seq = self.seq_counter
                
                self.get_logger().debug(f"Publishing YOLO detection with original timestamp for synchronization")
                
                position_msg.point.x = float(center_x)
                position_msg.point.y = float(center_y)
                position_msg.point.z = float(confidence)  # Using z for confidence
                self.ball_publisher.publish(position_msg)
            else:
                # Only log "no detection" occasionally to avoid flooding logs
                if self.image_count % 30 == 0:
                    self.get_logger().debug("No tennis ball detected in recent frames")

            # Calculate and display performance metrics
            total_time = (time.time() - inference_start) * 1000  # milliseconds
            elapsed_time = time.time() - self.start_time
            fps = self.image_count / elapsed_time if elapsed_time > 0 else 0
            
            # Update metrics for diagnostics
            self.diagnostic_metrics['fps_history'].append(fps)
            self.diagnostic_metrics['processing_time_history'].append(total_time)

            # Log performance periodically to avoid flooding the console
            if self.image_count % DIAG_CONFIG["performance_log_interval"] == 0:
                self.get_logger().info(
                    f"Performance: {fps:.1f} FPS, "
                    f"Processing: {total_time:.1f}ms, "
                    f"Inference: {infer_time:.1f}ms"
                )

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            # Track errors for diagnostics
            self.diagnostic_metrics['errors'].append(str(e))

    def publish_system_diagnostics(self):
        """Publish comprehensive system diagnostics for the diagnostics node."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate average metrics
        avg_fps = np.mean(list(self.diagnostic_metrics['fps_history'])) if self.diagnostic_metrics['fps_history'] else 0.0
        avg_processing_time = np.mean(list(self.diagnostic_metrics['processing_time_history'])) if self.diagnostic_metrics['processing_time_history'] else 0.0
        avg_inference_time = np.mean(list(self.diagnostic_metrics['inference_time_history'])) if self.diagnostic_metrics['inference_time_history'] else 0.0
        avg_detection_rate = np.mean(list(self.diagnostic_metrics['detection_rate_history'])) if self.diagnostic_metrics['detection_rate_history'] else 0.0
        avg_confidence = np.mean(list(self.diagnostic_metrics['confidence_history'])) if self.diagnostic_metrics['confidence_history'] else 0.0
        
        # Time since last detection
        time_since_detection = current_time - self.diagnostic_metrics['last_detection_time'] if self.diagnostic_metrics['last_detection_time'] > 0 else float('inf')
        
        # Build warnings list
        warnings = []
        errors = []
        
        # Check for performance issues
        if avg_fps < 5.0 and elapsed_time > 20.0:
            warnings.append(f"Low FPS: {avg_fps:.1f}")
            
        if avg_processing_time > 150.0:  # YOLO is more resource-intensive, so higher threshold
            warnings.append(f"High processing time: {avg_processing_time:.1f}ms")
            
        # Check for detection issues
        if time_since_detection > 10.0 and elapsed_time > 20.0:
            warnings.append(f"No ball detected for {time_since_detection:.1f}s")
            
        if avg_detection_rate < 0.05 and elapsed_time > 20.0:  # Less than 5% detection rate
            errors.append(f"Very low detection rate: {avg_detection_rate*100:.1f}%")
        
        # System resources
        system_resources = {}
        try:
            import psutil
            system_resources = {
                'cpu_percent': psutil.cpu_percent(interval=None),
                'memory_percent': psutil.virtual_memory().percent,
                'threads': MODEL_CONFIG["thread_count"]
            }
            
            # Check for high resource usage
            if system_resources['cpu_percent'] > 90.0:
                warnings.append(f"Critical CPU usage: {system_resources['cpu_percent']:.1f}%")
                
            # Add temperature if available
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                if temps and 'cpu_thermal' in temps:
                    system_resources['temperature'] = temps['cpu_thermal'][0].current
                    if system_resources['temperature'] > 80.0:  # Temperature in Celsius
                        warnings.append(f"High CPU temperature: {system_resources['temperature']:.1f}°C")
        except ImportError:
            pass
        
        # Build diagnostics data structure
        diag_data = {
            "node": "yolo",  # Changed from "node_name": "yolo_ball_node"
            "timestamp": current_time,
            "uptime_seconds": elapsed_time,
            "status": "error" if errors else ("warning" if warnings else "active"),  # Changed "ok" to "active"
            "health": {
                # Add proper health metrics for consistency
                "model_health": avg_confidence * 0.8 if avg_confidence > 0 else 0.5,
                "detection_health": avg_detection_rate if avg_detection_rate > 0 else 0.5,
                "processing_health": 1.0 - (avg_processing_time / 200.0) if avg_processing_time < 200.0 else 0.0,
                "overall": 1.0 - (len(errors) * 0.3) - (len(warnings) * 0.1)
            },
            "metrics": {  # Changed from "performance"
                "fps": avg_fps,
                "processing_time_ms": avg_processing_time,
                "inference_time_ms": avg_inference_time,
                "total_frames": self.diagnostic_metrics['total_frames'],
                "detected_frames": self.diagnostic_metrics['detected_frames'],
                "detection_rate": avg_detection_rate
            },
            "detection": {
                "latest_position": self.diagnostic_metrics['last_detection_position'],
                "time_since_last_detection_s": time_since_detection,
                "currently_tracking": time_since_detection < 2.0,
                "average_confidence": avg_confidence
            },
            "configuration": {
                "model": MODEL_CONFIG["path"],
                "precision": MODEL_CONFIG["precision"],
                "backend": MODEL_CONFIG["backend"],
                "confidence_threshold": MODEL_CONFIG["confidence_threshold"],
                "low_power_mode": self.low_power_mode
            },
            "resources": system_resources,  # Changed from "system_resources"
            "errors": errors,
            "warnings": warnings
        }
        
        # Publish as JSON
        msg = String()
        msg.data = json.dumps(diag_data)
        self.system_diagnostics_publisher.publish(msg)
        
        # Also log to console
        self.get_logger().info(
            f"YOLO diagnostics: {avg_fps:.1f} FPS, {avg_detection_rate*100:.1f}% detection rate, "
            f"Avg inference: {avg_inference_time:.1f}ms, Status: {diag_data['status']}"
        )

    def _handle_resource_alert(self, resource_type, value):
        """Handle high resource usage by adjusting detector behavior."""
        self.get_logger().warn(f"Resource alert: {resource_type} at {value:.1f}% - may affect performance")
        
        # Track warnings for diagnostics
        if not resource_type in [w for w in self.diagnostic_metrics['warnings']]:
            self.diagnostic_metrics['warnings'].append(f"High {resource_type}: {value:.1f}%")
        
        # Automatically enable low power mode if CPU is critically high
        if resource_type == 'cpu' and value > 95.0 and not self.low_power_mode:
            self.get_logger().warn("Enabling low power mode due to high CPU usage")
            self.low_power_mode = True

    def destroy_node(self):
        """Clean up resources when the node is shutting down."""
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop()
        super().destroy_node()

def main(args=None):
    """Main function to initialize and run the tennis ball detector."""
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create our tennis ball detector node
    node = TennisBallDetector()
    
    print("YOLO Tennis Ball Detector is now running! Press Ctrl+C to stop.")
    
    try:
        # Keep the node running until interrupted
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Stopping YOLO detector (Ctrl+C pressed)")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean shutdown
        node.destroy_node()
        rclpy.shutdown()
        print("YOLO Tennis Ball Detector has been shut down.")

if __name__ == '__main__':
    main()