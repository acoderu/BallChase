"""
Basketball Tracking Robot - Optimized YOLO Detection Node
==========================================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving basketball.
The system uses multiple sensing modalities for robust detection:
- YOLO object detection (this node)
- HSV color-based detection
- LiDAR for depth sensing
- Depth camera for additional depth information

These different sensing modalities are combined through sensor fusion to provide reliable
tracking even when one sensing method may fail.

This Node's Purpose:
------------------
This specific node uses a lightweight YOLO neural network to detect basketballs in camera
images. The node processes raw camera frames, runs them through a pre-trained YOLO model,
and publishes the detected ball's position with a confidence score.

Optimizations Added:
------------------
- Memory reuse and pre-allocation for better performance
- Fixed-size buffer implementation for history tracking
- Reuse of message objects for publishing
- Improved logging system with throttling and categories
- Resource monitoring with adaptive behavior

Coordinate System:
----------------
- Input images are resized to 320x320 pixels
- Output coordinates are in the same 320x320 image space
- (0,0) represents the top-left corner of the image
- Published positions include:
  - x: horizontal position (0-320)
  - y: vertical position (0-320)
  - z: confidence score (0-1)
"""
# Standard imports
import sys
import os
import json
import time
from collections import deque

# Add the parent directory of 'config' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Pose2D
from std_msgs.msg import String, Float32MultiArray
from vision_msgs.msg import BoundingBox2D
from cv_bridge import CvBridge

# ML/Vision imports
import MNN
import MNN.cv as mnn_cv2
import MNN.numpy as mnn_np
import cv2 as std_cv2
import numpy as np

# System monitoring
import psutil

# Project imports
from ball_chase.utilities.resource_monitor import ResourceMonitor
from ball_chase.utilities.time_utils import TimeUtils
from ball_chase.config.config_loader import ConfigLoader

# Load configuration
config_loader = ConfigLoader()
config = config_loader.load_yaml('yolo_config.yaml')

# Model configuration from config file - keeping as is per requirements
MODEL_CONFIG = config.get('model', {
    "path": "yolo12s_320.mnn",    # Path to our YOLO model file
    "input_width": 320,           # Width our model expects
    "input_height": 320,          # Height our model expects
    #"precision": "lowBF",        # Lower precision for faster inference
    "precision": "medium",        # Lower precision for faster inference
    "backend": "CPU",             # Using CPU for inference
    "thread_count": 1,            # Number of CPU threads to use
    "confidence_threshold": 0.2   # Only keep detections above this confidence
})

# COCO dataset class ID for "sports ball" - this includes basketballs
BASKETBALL_CLASS_ID = config.get('model', {}).get('basketball_class_id', 32)

# Topic configuration from config file
TOPICS = config.get('topics', {
    "input": {
        "camera": "/ascamera/camera_publisher/rgb0/image"
    },
    "output": {
        "position": "/basketball/yolo/position",
        "bbox": "/basketball/yolo/bbox"  # Added bbox topic configuration
    }
})

# Diagnostic configuration
DIAG_CONFIG = config.get('diagnostics', {
    "log_interval": 15,
    "performance_log_interval": 30
})

class LightweightBuffer:
    """
    Fixed-size buffer for storing recent detections with efficient memory usage.
    Uses pre-allocated memory to avoid dynamic allocations during runtime.
    """
    def __init__(self, max_size=10):
        """Initialize a fixed-size buffer."""
        self.data = [None] * max_size  # Pre-allocate list
        self.max_size = max_size
        self.next_index = 0
        self.is_full = False
        self.count = 0
    
    def add(self, timestamp, value):
        """Add a timestamped value to the buffer."""
        try:
            self.data[self.next_index] = (timestamp, value)
            self.next_index = (self.next_index + 1) % self.max_size
            self.count += 1
            if self.count >= self.max_size:
                self.is_full = True
                self.count = self.max_size
        except Exception as e:
            # Silently handle errors to prevent crashes during data collection
            # This could happen if the buffer is accessed before initialization
            pass
    
    def get_latest(self, count=1):
        """Get the latest entries from the buffer."""
        if self.count == 0:
            return []
        
        count = min(count, self.count)
        
        if self.is_full:
            start_idx = (self.next_index - count) % self.max_size
            if start_idx < self.next_index:
                return self.data[start_idx:self.next_index]
            else:
                return self.data[start_idx:] + self.data[:self.next_index]
        else:
            return self.data[max(0, self.next_index-count):self.next_index]
    
    def clear(self):
        """Clear the buffer."""
        self.next_index = 0
        self.is_full = False
        self.count = 0

class OptimizedBasketballDetector(Node):
    """
    An optimized ROS2 node that uses a YOLO neural network to detect basketballs
    in camera images and publishes their positions with timestamp information.
    
    This node subscribes to camera images, performs basketball detection using
    a YOLO neural network model, and publishes the detected ball position
    along with confidence information.
    
    Published Topics:
    - /basketball/yolo/position (geometry_msgs/PointStamped): 
      The detected ball position with timestamp
    - /basketball/yolo/bbox (std_msgs/Float32MultiArray):
      The bounding box of the detected ball for distance estimation
      
    Subscribed Topics:
    - /ascamera/camera_publisher/rgb0/image (sensor_msgs.Image): 
      RGB camera feed
    """
    
    def __init__(self):
        """Initialize the basketball detector node."""
        super().__init__('optimized_basketball_detector')
        
        # Initialize performance tracking variables first - needed for logging
        self.start_time = TimeUtils.now_as_float()
        self.image_count = 0
        
        # Allocate throttle tracking dictionary early for _log function
        self._throttle_times = {}
        
        # Initialize parameters with reasonable defaults
        self._init_parameters()
        
        # Initialize diagnostic metrics with pre-allocated buffers
        self._init_diagnostic_metrics()
        
        # Add Raspberry Pi resource monitoring
        self._setup_resource_monitoring()
        
        # Initialize publishers and subscribers
        self._init_ros_communication()
        
        # Bridge to convert between ROS images and OpenCV images
        self.bridge = CvBridge()
        
        # Initialize diagnostic metrics with pre-allocated buffers
        self._init_diagnostic_metrics()
        
        # Preallocate resources for reuse
        self._preallocate_resources()
        
        # Configure logging based on config
        log_config = config.get('logging', {
            'console_level': 'info',
            'publish_level': 'debug',
            'detection_log_rate': 10,
            'max_errors': 50,
            'performance_log_interval': 30
        })
        
        # Set log level
        self._configure_logging(log_config)
        
        # Initialize log file if configured
        self._init_log_file()
        
        # Log successful initialization
        self._log('info', 'INIT', f"Initialization complete, YOLO detector ready at {self.get_clock().now().to_msg().sec}.{self.get_clock().now().to_msg().nanosec//1000000:03d}s")

    def _init_parameters(self):
        """Initialize node parameters with reasonable defaults."""
        # Configuration for the MNN model
        self.mnn_config = {
            "precision": MODEL_CONFIG["precision"],
            "backend": MODEL_CONFIG["backend"],
            "numThread": MODEL_CONFIG["thread_count"],
        }
        
        # Settings for adaptable behavior
        self.low_power_mode = config.get('raspberry_pi', {}).get('low_power_mode', False)
        self.frame_skip_count = 1 if self.low_power_mode else 0
        self.detection_threshold = MODEL_CONFIG["confidence_threshold"]
        
        # Frame tracking
        self.frame_counter = 0
        self.seq_counter = 0
        
        # Error state tracking
        self.model_load_failed = False
        self.last_error_time = 0
        
    def _setup_resource_monitoring(self):
        """Set up resource monitoring for the node."""
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
        
        if self.low_power_mode:
            self._log('info', 'SYSTEM', "Running in low power mode for Raspberry Pi", throttle=0)

    def _init_ros_communication(self):
        """Initialize publishers and subscribers."""
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

        # Create a publisher for bounding box information
        self.bbox_publisher = self.create_publisher(
            Float32MultiArray, 
            TOPICS["output"].get("bbox", "/basketball/yolo/bbox"), 
            10
        )

        # Create a publisher for system diagnostics
        self.system_diagnostics_publisher = self.create_publisher(
            String,
            "/basketball/yolo/diagnostics",
            10
        )
        
        # Timer for publishing diagnostics
        self.diagnostics_timer = self.create_timer(3.0, self.publish_system_diagnostics)

    def _configure_logging(self, log_config):
        """Configure logger levels and parameters."""
        # Map string log levels to rclpy.logging.LoggingSeverity values
        level_map = {
            'debug': rclpy.logging.LoggingSeverity.DEBUG,
            'info': rclpy.logging.LoggingSeverity.INFO,
            'warn': rclpy.logging.LoggingSeverity.WARN,
            'error': rclpy.logging.LoggingSeverity.ERROR
        }
        
        # Set logger level 
        log_level = level_map.get(log_config.get('console_level', 'info').lower(), 
                                  rclpy.logging.LoggingSeverity.INFO)
        self.get_logger().set_level(log_level)
        
        # Store log config
        self.log_config = log_config

    def _init_log_file(self):
        """Initialize log file if configured."""
        log_file = self.log_config.get('log_file')
        if not log_file:
            return
            
        import logging
        file_handler = logging.FileHandler(log_file, mode='a')
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Get the logger used by rclpy
        logger_name = self.get_name()
        logger = logging.getLogger(logger_name)
        logger.addHandler(file_handler)
        
        # Set file logging level
        file_level = self.log_config.get('file_level', 'debug').upper()
        file_handler.setLevel(getattr(logging, file_level))

    def _init_diagnostic_metrics(self):
        """Initialize metrics for diagnostic monitoring with fixed buffers."""
        # Use fixed-size buffers to avoid memory allocations
        buffer_size = 10  # Keep last 10 entries for each metric
        
        self.diagnostic_metrics = {
            # Use fixed-size buffers for time series data
            'fps_history': np.zeros(buffer_size, dtype=np.float32),
            'processing_time_history': np.zeros(buffer_size, dtype=np.float32),
            'inference_time_history': np.zeros(buffer_size, dtype=np.float32),
            'detection_rate_history': np.zeros(buffer_size, dtype=np.float32),
            'confidence_history': np.zeros(buffer_size, dtype=np.float32),
            
            # Indices for circular buffers
            'fps_idx': 0,
            'processing_idx': 0,
            'inference_idx': 0,
            'detection_rate_idx': 0,
            'confidence_idx': 0,
            
            # Initialize counters
            'total_frames': 0,
            'detected_frames': 0,
            'missed_frames': 0,
            
            # Initialize detection data
            'last_detection_position': (0.0, 0.0),
            'last_detection_time': 0.0,
            
            # Limited-size error tracking
            'errors': [],
            'warnings': []
        }
        
        # Pre-allocate error and warning arrays with max size
        self.diagnostic_metrics['errors'] = [None] * 10
        self.diagnostic_metrics['warnings'] = [None] * 10
        self.error_count = 0
        self.warning_count = 0
        
        # Use LightweightBuffer for recent detections
        self.recent_detections = LightweightBuffer(max_size=10)

    def _preallocate_resources(self):
        """Preallocate resources that will be reused during operation."""
        # Preallocate message objects for reuse
        self._position_msg = PointStamped()
        self._position_msg.header.frame_id = "ascamera_color_0"
        
        # Preallocate bbox message
        self._bbox_msg = Float32MultiArray()
        self._bbox_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0]  # [center_x, center_y, width, height, confidence]
        
        # Preallocate tensor for reuse
        self._input_tensor = None
        
        # Preallocate arrays for detection processing
        self._best_box = np.zeros(4, dtype=np.float32)  # [x0, y0, x1, y1]

    def _log(self, level, category, message, data=None, throttle=0):
        """
        Unified logging with component tags, throttling, and optional data.
        
        Args:
            level (str): Log level ('debug', 'info', 'warn', 'error')
            category (str): Log category ('YOLO', 'DETECT', 'STATE', etc.)
            message (str): Main log message
            data (dict): Optional data to include in debug logs
            throttle (int): Throttle interval in seconds (0 = no throttling)
        """
        # Check if throttled
        now = TimeUtils.now_as_float()
        throttle_key = f"{category}:{message}"
        
        if throttle > 0:
            # Skip if within throttle period
            if throttle_key in self._throttle_times:
                if now - self._throttle_times[throttle_key] < throttle:
                    return
            
            # Update last log time
            self._throttle_times[throttle_key] = now
        
        # Format timestamp as [T+Ns][ROS:timestamp] for precise timing
        ros_time = self.get_clock().now().to_msg()
        ros_time_str = f"{ros_time.sec}.{ros_time.nanosec//1000000:03d}"
        
        # For important messages, include both timestamps
        if category in ['DETECT', 'FUSION', 'STATE']:
            # Only calculate uptime if we need it
            uptime = now - self.start_time
            time_prefix = f"[T+{uptime:.1f}s][ROS:{ros_time_str}]"
        else:
            time_prefix = ""
        
        # Format the log message with component and optional timestamp
        if time_prefix:
            formatted_msg = f"{time_prefix}[{category}] {message}"
        else:
            formatted_msg = f"[{category}] {message}"
        
        # Add data as JSON if provided and in debug mode
        if data and level == 'debug':
            # Truncate data for console
            data_str = str(data)
            if len(data_str) > 100:
                data_str = data_str[:100] + "..."
            formatted_msg += f" | {data_str}"
        
        # Log with appropriate level
        if level == 'debug':
            self.get_logger().debug(formatted_msg)
        elif level == 'info':
            self.get_logger().info(formatted_msg)
        elif level == 'warn':
            self.get_logger().warn(formatted_msg)
        elif level == 'error':
            self.get_logger().error(formatted_msg)
            
        # Store in error/warning history if needed
        if level == 'error':
            self.diagnostic_metrics['errors'][self.error_count % 10] = {
                "time": now,
                "message": message
            }
            self.error_count += 1
        elif level == 'warn':
            self.diagnostic_metrics['warnings'][self.warning_count % 10] = {
                "time": now,
                "message": message
            }
            self.warning_count += 1

    def _add_to_metric_history(self, metric_name, value):
        """Add a value to a circular buffer metric history."""
        if metric_name + '_history' in self.diagnostic_metrics:
            idx_name = metric_name + '_idx'
            history_name = metric_name + '_history'
            
            # Ensure index exists
            if idx_name not in self.diagnostic_metrics:
                self.diagnostic_metrics[idx_name] = 0
                
            # Get current index
            current_idx = self.diagnostic_metrics[idx_name]
            
            # Validate value before storing (prevent NaN/inf)
            if isinstance(value, (int, float)) and np.isfinite(value):
                # Store value
                self.diagnostic_metrics[history_name][current_idx] = value
                
                # Update index
                self.diagnostic_metrics[idx_name] = (current_idx + 1) % len(self.diagnostic_metrics[history_name])
            else:
                # Log invalid value but don't store it
                self._log('warn', 'METRIC', f"Invalid value for {metric_name}: {value}", throttle=30)

    def _get_metric_average(self, metric_name):
        """Get the average of a metric history."""
        try:
            if metric_name + '_history' in self.diagnostic_metrics:
                values = self.diagnostic_metrics[metric_name + '_history']
                # Filter out zeros (uninitialized values) and non-finite values
                valid_values = values[(values > 0) & np.isfinite(values)]
                if len(valid_values) > 0:
                    return float(np.mean(valid_values))
        except Exception as e:
            # Quietly handle exceptions to prevent diagnostics from breaking the node
            self._log('debug', 'METRIC', f"Error getting metric average for {metric_name}: {str(e)}", throttle=30)
        return 0.0

    def _trace_start(self, operation):
        """Start timing an operation."""
        if not hasattr(self, 'trace_points'):
            self.trace_points = {}
        self.trace_points[operation] = TimeUtils.now_as_float()

    def _trace_end(self, operation):
        """End timing an operation and return elapsed time in ms."""
        if hasattr(self, 'trace_points') and operation in self.trace_points:
            elapsed = (TimeUtils.now_as_float() - self.trace_points[operation]) * 1000
            return elapsed
        return 0.0

    def load_model(self, config):
        """
        Load the YOLO model for tennis ball detection.
        
        Args:
            config (dict): Configuration parameters for the MNN runtime
        """
        try:
            # Get model filename from the path in config
            model_filename = os.path.basename(MODEL_CONFIG["path"])
            
            # Use get_package_share_directory to find the model path
            from ament_index_python.packages import get_package_share_directory
            package_share_dir = get_package_share_directory('ball_chase')
            model_path = os.path.join(package_share_dir, 'models', model_filename)
            
            self._log('info', 'MODEL', f"Loading model from {model_path}...")
            
            # Check if model exists
            if not os.path.exists(model_path):
                self._log('error', 'MODEL', f"Model not found at {model_path}")
                self.model_load_failed = True
                self.net = None
                return
            
            # Initialize MNN runtime manager with our configuration
            self.runtime_manager = MNN.nn.create_runtime_manager((config,))
            
            # Load the YOLO model from file using the absolute path
            self.net = MNN.nn.load_module_from_file(
                model_path, [], [], runtime_manager=self.runtime_manager
            )
            
            # Create a test image to warm up the model
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
            
            try:
                # Run the model once to warm it up
                self.net.forward(dummy_input)
                self._log('info', 'MODEL', "YOLO model loaded and warmed up successfully")
            except Exception as e:
                self._log('warn', 'MODEL', f"Model loaded but warmup failed: {str(e)}")
                # Continue even if warmup fails - the model might still work
                
        except Exception as e:
            self._log('error', 'MODEL', f"Failed to load model: {str(e)}")
            # Don't raise, continue with graceful degradation
            self.model_load_failed = True

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
        Process YOLO output to extract basketball detections.
        
        Args:
            output_var (MNN.expr.Var): Raw output from YOLO model
            
        Returns:
            tuple: (best_box, confidence) if basketball found, else (None, 0)
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
        
        # Find all basketball detections with confidence above threshold
        basketball_indices = []
        for i in range(len(class_ids)):
            if scores[i] < 0.01:
                continue
            if (class_ids[i] == BASKETBALL_CLASS_ID and 
                scores[i] > MODEL_CONFIG["confidence_threshold"]):
                basketball_indices.append(i)
        
        # If no basketballs found, return None
        if not basketball_indices:
            return None, 0.0
            
        # If multiple basketballs detected, take the one with highest confidence
        best_idx = basketball_indices[0]
        for idx in basketball_indices:
            if scores[idx] > scores[best_idx]:
                best_idx = idx
        
        # Get the box coordinates for our best detection
        box = boxes[best_idx]
        x0_val, y0_val, x1_val, y1_val = box.read_as_tuple()
        
        # Calculate confidence adjustments
        base_confidence = scores[best_idx]
        
        # Adjust confidence based on aspect ratio (basketballs should be round)
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
        self._trace_start('overall')
        
        # Skip frames based on low power mode
        self.frame_counter += 1
        if self.frame_skip_count > 0 and (self.frame_counter % (self.frame_skip_count + 1)) != 0:
            return

        # Check if model failed to load or is not available
        if not hasattr(self, 'net') or self.net is None:
            now = TimeUtils.now_as_float()
            # Only log error every 10 seconds to avoid flooding
            if not hasattr(self, 'last_error_time') or now - self.last_error_time > 10:
                self._log('error', 'MODEL', "Cannot process image - model not available", throttle=10)
                self.last_error_time = now
            return

        # Update statistics
        inference_start = TimeUtils.now_as_float()
        self.last_callback_time = inference_start  # Track last callback time
        self.image_count += 1
        self.diagnostic_metrics['total_frames'] = self.image_count
        
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Preprocess the image for model input
            self._trace_start('preprocess')
            input_tensor = self.preprocess_image(cv_image)
            preprocess_time = self._trace_end('preprocess')
            
            # Run the neural network to detect objects
            self._trace_start('inference')
            output_var = self.net.forward(input_tensor)
            inference_time = self._trace_end('inference')
            
            # Track inference time for diagnostics
            self._add_to_metric_history('inference_time', inference_time)
            
            # Process the model output to find tennis balls
            self._trace_start('postprocess')
            best_box, confidence = self.process_detections(output_var)
            postprocess_time = self._trace_end('postprocess')
            
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
                self.diagnostic_metrics['last_detection_time'] = TimeUtils.now_as_float()
                
                # Update metrics
                self._add_to_metric_history('confidence', confidence)
                
                # Calculate detection rate
                detection_rate = self.diagnostic_metrics['detected_frames'] / self.image_count
                self._add_to_metric_history('detection_rate', detection_rate)
                
                # Log detection with improved formatting similar to fusion node
                # Only log if confidence is above threshold or periodically
                should_log = (confidence > 0.3 or 
                             (self.image_count % self.log_config.get('detection_log_rate', 10) == 0))
                
                if should_log:
                    self._log('info', 'DETECT', 
                             f"Ball at ({center_x:.1f}, {center_y:.1f}), Size: {width:.1f}x{height:.1f}, Conf: {confidence:.2f}", 
                             throttle=0)  # No throttling for actual detections
                
                # Create and publish the position message with timestamp
                # Use pre-allocated message object
                
                # Validate the timestamp
                if TimeUtils.is_timestamp_valid(msg.header.stamp):
                    self._position_msg.header.stamp = msg.header.stamp
                else:
                    self._position_msg.header.stamp = TimeUtils.now_as_ros_time()
                
                # Increment sequence counter
                self.seq_counter += 1
                
                # Set position data
                self._position_msg.point.x = float(center_x)
                self._position_msg.point.y = float(center_y)
                self._position_msg.point.z = float(confidence)  # Using z for confidence
                
                # Publish position
                self.ball_publisher.publish(self._position_msg)

                # Use pre-allocated bbox message
                # Format: [center_x, center_y, width, height, confidence]
                self._bbox_msg.data[0] = float(center_x)
                self._bbox_msg.data[1] = float(center_y)
                self._bbox_msg.data[2] = float(width)
                self._bbox_msg.data[3] = float(height)
                self._bbox_msg.data[4] = float(confidence)
                
                # Publish the bounding box
                self.bbox_publisher.publish(self._bbox_msg)

                # Store detailed data for high-confidence detections
                if confidence > 0.5:
                    # Use lightweight buffer instead of deque
                    self.recent_detections.add(TimeUtils.now_as_float(), {
                        "position": (float(center_x), float(center_y)),
                        "size": (float(width), float(height)),
                        "confidence": float(confidence),
                        "aspect_ratio": float(width/height) if height > 0 else 1.0,
                        "frame_id": self.seq_counter
                    })
            else:
                # Only log "no detection" occasionally to avoid flooding logs
                if self.image_count % 30 == 0:
                    self._log('debug', 'DETECT', "No basketball detected in recent frames", throttle=5)

            # Calculate and display performance metrics
            total_time = (TimeUtils.now_as_float() - inference_start) * 1000  # milliseconds
            elapsed_time = TimeUtils.now_as_float() - self.start_time
            fps = self.image_count / elapsed_time if elapsed_time > 0 else 0
            
            # Update metrics for diagnostics
            self._add_to_metric_history('fps', fps)
            self._add_to_metric_history('processing_time', total_time)

            # Log performance periodically (throttled)
            if self.image_count % DIAG_CONFIG["performance_log_interval"] == 0:
                self._log('info', 'PERF', 
                         f"FPS: {fps:.1f}, Processing: {total_time:.1f}ms, Inference: {inference_time:.1f}ms", 
                         throttle=DIAG_CONFIG["performance_log_interval"])

            overall_time = self._trace_end('overall')
            
            # Log detailed timing occasionally (highly throttled)
            if self.image_count % self.log_config.get('trace_log_interval', 100) == 0:
                detail = {
                    "preprocess_ms": preprocess_time,
                    "inference_ms": inference_time,
                    "postprocess_ms": postprocess_time,
                    "overhead_ms": overall_time - (preprocess_time + inference_time + postprocess_time)
                }
                self._log('debug', 'TRACE', "YOLO timing breakdown", data=detail, throttle=30)

        except Exception as e:
            self._log('error', 'ERROR', f"Error processing image: {str(e)}")

    def publish_system_diagnostics(self):
        """Publish comprehensive system diagnostics."""
        current_time = TimeUtils.now_as_float()
        elapsed_time = current_time - self.start_time
        ros_time = self.get_clock().now().to_msg()
        
        # Calculate average metrics
        avg_fps = self._get_metric_average('fps')
        avg_processing_time = self._get_metric_average('processing_time')
        avg_inference_time = self._get_metric_average('inference_time')
        avg_detection_rate = self._get_metric_average('detection_rate')
        avg_confidence = self._get_metric_average('confidence')
        
        # Time since last detection
        if self.diagnostic_metrics['last_detection_time'] > 0:
            time_since_detection = current_time - self.diagnostic_metrics['last_detection_time']
        else:
            time_since_detection = float('inf')
        
        # Build warnings list
        warnings = []
        errors = []
        
        # Check for performance issues
        if avg_fps < 5.0 and elapsed_time > 20.0:
            warnings.append(f"Low FPS: {avg_fps:.1f}")
            
        if avg_processing_time > 150.0:
            warnings.append(f"High processing time: {avg_processing_time:.1f}ms")
            
        # Check for detection issues
        if time_since_detection > 10.0 and elapsed_time > 20.0:
            warnings.append(f"No ball detected for {time_since_detection:.1f}s")
            
        if avg_detection_rate < 0.05 and elapsed_time > 20.0:
            errors.append(f"Very low detection rate: {avg_detection_rate*100:.1f}%")
        
        # Check model status
        if hasattr(self, 'model_load_failed') and self.model_load_failed:
            errors.append("Model failed to load")
        elif not hasattr(self, 'net') or self.net is None:
            errors.append("Model not available")
        
        # System resources
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
                if system_resources['temperature'] > 80.0:
                    warnings.append(f"High CPU temperature: {system_resources['temperature']:.1f}°C")
        
        # Build diagnostics data structure
        diag_data = {
            "node": "yolo",
            "timestamp": current_time,
            "ros_time": f"{ros_time.sec}.{ros_time.nanosec//1000000:03d}",
            "uptime_seconds": elapsed_time,
            "status": "error" if errors else ("warning" if warnings else "active"),
            "health": {
                "model_health": 0.0 if (not hasattr(self, 'net') or self.net is None) else (avg_confidence * 0.8 if avg_confidence > 0 else 0.5),
                "detection_health": avg_detection_rate if avg_detection_rate > 0 else 0.0,
                "processing_health": 1.0 - (avg_processing_time / 200.0) if avg_processing_time < 200.0 else 0.0,
                "overall": 0.0 if (not hasattr(self, 'net') or self.net is None) else (1.0 - (len(errors) * 0.3) - (len(warnings) * 0.1))
            },
            "metrics": {
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
            "resources": system_resources,
            "errors": errors,
            "warnings": warnings
        }

        # Add recent detections if available
        if hasattr(self, 'recent_detections'):
            recent_data = self.recent_detections.get_latest(5)  # Get last 5 detections
            if recent_data and len(recent_data) > 0 and recent_data[0] is not None:
                diag_data["detection"]["recent_detections"] = recent_data
        
        # Run health check
        health_issues = self.perform_health_check()
        if health_issues:
            for issue in health_issues:
                self._log('warn', 'HEALTH', issue, throttle=30)
            # Add to warnings
            warnings.extend(health_issues)

        # Publish as JSON
        msg = String()
        msg.data = json.dumps(diag_data)
        self.system_diagnostics_publisher.publish(msg)
        
        # Also log to console in a consistent format like the fusion node
        status_char = "✓" if diag_data['status'] == "active" else ("!" if diag_data['status'] == "warning" else "✗")
        self._log('info', 'DIAG', 
                 f"{status_char} {avg_fps:.1f} FPS, {avg_detection_rate*100:.1f}% detection rate, "
                 f"Avg inference: {avg_inference_time:.1f}ms, Status: {diag_data['status']}",
                 throttle=0)  # Don't throttle diagnostics

    def perform_health_check(self):
        """Run periodic health checks and self-diagnostics."""
        health_issues = []
        
        # Check model state - prioritize model checks
        if hasattr(self, 'model_load_failed') and self.model_load_failed:
            health_issues.append("YOLO model failed to load")
        elif not hasattr(self, 'net') or self.net is None:
            health_issues.append("YOLO model not loaded")
            
            # Try to reload the model if it's been a while since the last attempt
            current_time = TimeUtils.now_as_float()
            if (not hasattr(self, 'last_model_reload_attempt') or 
                current_time - self.last_model_reload_attempt > 60.0):  # Try once per minute
                self._log('info', 'MODEL', "Attempting to reload YOLO model...")
                self.last_model_reload_attempt = current_time
                self.load_model(self.mnn_config)
                
                # Check if reload was successful
                if hasattr(self, 'net') and self.net is not None:
                    self._log('info', 'MODEL', "Model reload successful!")


    def _handle_resource_alert(self, resource_type, value):
        """Handle high resource usage by adjusting detector behavior."""
        self._log('warn', 'SYSTEM', 
                 f"Resource alert: {resource_type} at {value:.1f}% - may affect performance",
                 throttle=30)
        
        # Check if diagnostic_metrics are initialized
        if hasattr(self, 'diagnostic_metrics') and 'warnings' in self.diagnostic_metrics:
            # Add warning to system warnings if not already present
            warning_msg = f"High {resource_type}: {value:.1f}%"
            if warning_msg not in [w.get('message', '') for w in self.diagnostic_metrics['warnings'] if w]:
                if self.warning_count < len(self.diagnostic_metrics['warnings']):
                    self.diagnostic_metrics['warnings'][self.warning_count] = {
                        "time": TimeUtils.now_as_float(),
                        "message": warning_msg
                    }
                    self.warning_count += 1
        
        # Automatically enable low power mode if CPU is critically high
        if resource_type == 'cpu' and value > 95.0 and not self.low_power_mode:
            self._log('warn', 'SYSTEM', 
                     "Enabling low power mode due to high CPU usage",
                     throttle=60)
            self.low_power_mode = True
            self.frame_skip_count = 1  # Skip every other frame

    def destroy_node(self):
        """Clean up YOLO model resources."""
        self._log('info', 'SHUTDOWN', "Cleaning up resources...")
        
        # Release model resources
        if hasattr(self, 'net'):
            del self.net
        
        # Release pre-allocated tensors
        if hasattr(self, '_input_tensor'):
            self._input_tensor = None
        
        # Stop resource monitor
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop()
            
        # Force garbage collection to clean up ML resources
        import gc
        gc.collect()
        
        self._log('info', 'SHUTDOWN', "YOLO detector resources released")
        super().destroy_node()

def main(args=None):
    """Main function to initialize and run the basketball detector."""
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create our basketball detector node
    node = OptimizedBasketballDetector()
    
    print("Optimized YOLO Basketball Detector is now running! Press Ctrl+C to stop.")
    
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
        print("YOLO Basketball Detector has been shut down.")

if __name__ == '__main__':
    main()