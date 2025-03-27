#!/usr/bin/env python3

"""
Tennis Ball Tracking Robot - HSV Ball Detector Node
==================================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving tennis ball.
The system uses multiple sensing modalities for robust detection:
- YOLO neural network detection (more accurate but computationally intensive)
- HSV color-based detection (this node - fast and efficient)
- LIDAR for depth sensing
- Depth camera for additional depth information

This Node's Purpose:
------------------
This HSV detector node uses traditional computer vision techniques to detect tennis balls
based on their distinctive yellow-green color. It processes camera images, applies color
filtering in the HSV color space, and identifies circular objects of the right size and shape.

HSV color detection offers several advantages:
- Very fast processing compared to neural network approaches
- More resilient to changes in lighting conditions than RGB
- Can be fine-tuned for specific color targets

Data Pipeline:
-------------
1. Camera images are received from '/ascamera/camera_publisher/rgb0/image'
2. Images are processed to extract tennis ball position using HSV filtering
3. Detected positions are published to '/tennis_ball/hsv/position'
4. These positions are then used by:
   - Depth camera node for 3D position estimation
   - Sensor fusion node for combining with other detection methods
   - State manager for decision making
   - PID controller for motor control
"""

import sys
import os
# Add the parent directory of 'config' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Float32  # Added Float32 for CPU load publishing
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import psutil
import json
from collections import deque
from functools import lru_cache  # Added for caching expensive operations

from utilities.resource_monitor import ResourceMonitor
from utilities.time_utils import TimeUtils

# Load configuration from file
from ball_chase.config.config_loader import ConfigLoader
config_loader = ConfigLoader()
config = config_loader.load_yaml('hsv_config.yaml')

# Topic configuration from config file
TOPICS = config.get('topics', {
    "input": {
        "camera": "/ascamera/camera_publisher/rgb0/image"
    },
    "output": {
        "position": "/tennis_ball/hsv/position"
    }
})

# Tennis ball detection configuration from config file
HSV_LOWER = np.array(config.get('ball', {}).get('hsv_range', {}).get('lower', [27, 58, 77]), dtype=np.uint8)
HSV_UPPER = np.array(config.get('ball', {}).get('hsv_range', {}).get('upper', [45, 255, 255]), dtype=np.uint8)

BALL_CONFIG = {
    "hsv_range": {
        "lower": HSV_LOWER,  # Lower HSV boundary for tennis ball
        "upper": HSV_UPPER   # Upper HSV boundary for tennis ball
    },
    "size": config.get('ball', {}).get('size', {
        "min_area": 100,     # Minimum area in pixels for 320x320 image
        "max_area": 1500,    # Maximum area in pixels for 320x320 image
        "ideal_area": 600    # Ideal area for confidence calculation
    }),
    "shape": config.get('ball', {}).get('shape', {
        "min_circularity": 0.5,   # Minimum circularity (0.7 is a perfect circle)
        "max_circularity": 1.3,   # Maximum circularity
        "ideal_circularity": 0.7  # Ideal circularity for confidence calculation
    })
}

# Display configuration
DISPLAY_CONFIG = config.get('display', {
    "enable_visualization": False,  # Whether to show detection visualization
    "window_width": 800,            # Width of visualization window
    "window_height": 600            # Height of visualization window
})

# Diagnostic configuration
DIAG_CONFIG = config.get('diagnostics', {
    "target_width": 320,           # Target width for processing 
    "target_height": 320,          # Target height for processing
    "debug_level": 1,              # 0=errors only, 1=info, 2=debug
    "log_interval": 10             # Log every N frames for performance stats
})

# New performance configuration
PERF_CONFIG = config.get('performance', {
    # CPU thresholds for reducing processing
    "cpu_high_threshold": 85.0,       # Above this threshold, reduce processing dramatically
    "cpu_medium_threshold": 70.0,     # Above this threshold, start reducing processing
    "cpu_low_threshold": 60.0,        # Below this threshold, process at full quality
    
    # Resolution downscaling factors for different CPU loads
    "high_load_scale": 0.5,           # Scale down to 50% resolution in high load
    "medium_load_scale": 0.75,        # Scale down to 75% resolution in medium load
    
    # Processing frequency control
    "min_processing_interval": 0.05,  # At least 50ms between frames in high load
    "cpu_check_interval": 1.0         # Check CPU usage every 1 second
})

class HSVTennisBallTracker(Node):
    """
    A ROS2 node that uses HSV color filtering to detect a yellow tennis ball
    in camera images and publishes its position.
    
    HSV (Hue, Saturation, Value) color space is better for color detection than RGB
    because it separates color (hue) from intensity (value) and color purity (saturation).
    This makes it more robust to lighting changes.
    
    This detector works with 320x320 images to match YOLO's input size for consistency
    across the detection pipeline.
    """
    
    def __init__(self):
        """Initialize the HSV tennis ball tracker with all required components."""
        # Initialize our ROS node
        super().__init__('hsv_tennis_ball_tracker')
        
        # Add resource monitoring for Raspberry Pi 5 with 16GB RAM
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=15.0,  # Less frequent to reduce overhead
            enable_temperature=True
        )
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.resource_monitor.start()
        
        # Set up parameters
        self._declare_parameters()
        
        # Configure optimization settings for Pi 5 with 16GB
        self._configure_optimizations()
        
        # Set up subscriptions and publishers
        self._setup_communication()
        
        # Initialize state variables
        self._init_state_variables()
        
        # Set up visualization if enabled
        self._setup_visualization()
        
        # Use bounded collections for metrics
        max_history = 100  # Default or from config
        self.fps_history = deque(maxlen=max_history)
        self.processing_times = deque(maxlen=max_history)
        self.detection_history = deque(maxlen=max_history)
        self.errors = deque(maxlen=50)
        self.warnings = deque(maxlen=50)
        
        # Initialize adaptive processing variables
        self.last_frame_time = 0.0
        self.current_scale_factor = 1.0
        self.current_cpu_usage = 0.0
        self.last_cpu_check_time = 0.0
        self.skip_count = 0
        
        # Create a CPU usage publisher
        self.cpu_usage_publisher = self.create_publisher(
            Float32,
            '/system/resources/cpu_load',
            10
        )
        
        # Add timer to check CPU and adjust processing rate/quality
        self.cpu_check_timer = self.create_timer(
            PERF_CONFIG.get('cpu_check_interval', 1.0),  # Check CPU every 1 second by default
            self._check_cpu_and_adjust_processing
        )
        
        # Pre-allocate memory for image operations
        self._init_image_buffers()
        
        self.get_logger().info("HSV Tennis Ball Tracker has started!")
        self.get_logger().info(f"Processing images at {self.target_width}x{self.target_height} to match YOLO")
        self.get_logger().info(f"Looking for balls with area between {self.min_ball_area} and {self.max_ball_area} pixels")
        self.get_logger().info(f"HSV color range: Lower={BALL_CONFIG['hsv_range']['lower']}, Upper={BALL_CONFIG['hsv_range']['upper']}")
        self.get_logger().info(f"CPU adaptivity enabled: high={PERF_CONFIG.get('cpu_high_threshold')}%, medium={PERF_CONFIG.get('cpu_medium_threshold')}%")

    def _init_image_buffers(self):
        """Pre-allocate memory for image operations to avoid frequent allocations."""
        # Pre-allocate buffers for image processing
        self.target_width = DIAG_CONFIG["target_width"]
        self.target_height = DIAG_CONFIG["target_height"]
        
        # Pre-allocate buffers for different scales
        self.image_buffers = {}
        for scale in [1.0, PERF_CONFIG.get('medium_load_scale', 0.75), PERF_CONFIG.get('high_load_scale', 0.5)]:
            width = int(self.target_width * scale)
            height = int(self.target_height * scale)
            # Only create if sensible dimensions (at least 32x32)
            if width >= 32 and height >= 32:
                self.image_buffers[scale] = {
                    'bgr': np.zeros((height, width, 3), dtype=np.uint8),
                    'hsv': np.zeros((height, width, 3), dtype=np.uint8),
                    'mask': np.zeros((height, width), dtype=np.uint8)
                }
        
        # Create morphological kernels at startup to avoid runtime creation
        self.morph_kernels = {
            'small': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            'medium': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            'large': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        }

    def _declare_parameters(self):
        """Declare and get all node parameters."""
        # Set parameters from config
        self.target_width = DIAG_CONFIG["target_width"]
        self.target_height = DIAG_CONFIG["target_height"]
        self.enable_visualization = DISPLAY_CONFIG["enable_visualization"] 
        self.debug_level = DIAG_CONFIG["debug_level"]
        self.log_interval = DIAG_CONFIG["log_interval"]
        
        # Get ball detection parameters from configuration
        self.lower_yellow = BALL_CONFIG['hsv_range']['lower']
        self.upper_yellow = BALL_CONFIG['hsv_range']['upper']
        self.min_ball_area = BALL_CONFIG['size']['min_area']
        self.max_ball_area = BALL_CONFIG['size']['max_area']
        self.ideal_area = BALL_CONFIG['size']['ideal_area']
        self.min_circularity = BALL_CONFIG['shape']['min_circularity']
        self.max_circularity = BALL_CONFIG['shape']['max_circularity']
        self.ideal_circularity = BALL_CONFIG['shape']['ideal_circularity']
        
        # Add performance parameters
        self.cpu_high_threshold = PERF_CONFIG.get('cpu_high_threshold', 85.0)
        self.cpu_medium_threshold = PERF_CONFIG.get('cpu_medium_threshold', 70.0)
        self.cpu_low_threshold = PERF_CONFIG.get('cpu_low_threshold', 60.0)
        self.high_load_scale = PERF_CONFIG.get('high_load_scale', 0.5) 
        self.medium_load_scale = PERF_CONFIG.get('medium_load_scale', 0.75)
        self.min_processing_interval = PERF_CONFIG.get('min_processing_interval', 0.05)

    def _setup_communication(self):
        """Set up all subscriptions and publishers."""
        # Subscribe to the camera feed
        self.subscription = self.create_subscription(
            Image, 
            TOPICS["input"]["camera"], 
            self.image_callback, 
            10
        )
        
        # Create a publisher for ball position
        self.ball_publisher = self.create_publisher(
            PointStamped, 
            TOPICS["output"]["position"], 
            10
        )
        
        # Bridge to convert between ROS images and OpenCV images
        self.bridge = CvBridge()
        
        # Create a publisher for system diagnostics
        self.system_diagnostics_publisher = self.create_publisher(
            String, 
            "/tennis_ball/hsv/diagnostics",  
            10
        )
        
        # Timer for publishing diagnostics
        self.diagnostics_timer = self.create_timer(2.0, self.publish_system_diagnostics)

    def _init_state_variables(self):
        """Initialize all state tracking variables."""
        # Performance tracking
        self.start_time = TimeUtils.now_as_float()
        self.frame_count = 0
        self.no_detection_count = 0
        self.last_detection_time = None
        
        # Detection statistics - replace unbounded lists with deque
        self.detection_count = 0
        self.detection_sizes = deque(maxlen=50)
        self.detection_confidences = deque(maxlen=50)
        
        # Processing timing
        self.processing_times = deque(maxlen=50)
        
        # Initialize diagnostic metrics
        self.diagnostic_metrics = {
            'fps_history': deque(maxlen=10),
            'processing_time_history': deque(maxlen=10),
            'detection_rate_history': deque(maxlen=10),
            'last_detection_position': None,
            'last_detection_time': 0.0,
            'total_frames': 0,
            'missed_frames': 0,
            'errors': deque(maxlen=10),
            'warnings': deque(maxlen=10),
            'adaptations': deque(maxlen=20)  # Track adaptive processing changes
        }
        
        # Track CPU usage over time for trends
        self.cpu_history = deque(maxlen=30)  # Track last 30 seconds
        self.adaptation_history = deque(maxlen=20)  # Track adaptation changes

    def _setup_visualization(self):
        """Set up visualization windows if enabled."""
        if self.enable_visualization:
            cv2.namedWindow("Tennis Ball Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tennis Ball Detector", 
                           DISPLAY_CONFIG["window_width"], 
                           DISPLAY_CONFIG["window_height"])
            self.get_logger().info("Visualization enabled - showing detection window")

    def _configure_optimizations(self):
        """Configure performance optimizations based on RAM availability."""
        # With 16GB RAM, we can optimize for processing quality rather than memory usage
        try:
            total_ram = psutil.virtual_memory().total / (1024 * 1024)  # MB
            
            # On Pi 5 with 16GB RAM, we can use more advanced options
            if total_ram >= 12000:  # At least 12GB
                # Precompute kernel for morphological operations for better performance
                self.morphology_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                
                # Enable more advanced detection features that use more RAM but give better results
                self.use_enhanced_detection = True
                
                # Use full-sized images when CPU allows
                self.default_scale_factor = 1.0
                
                self.get_logger().info(f"Using enhanced detection features (high RAM mode)")
            else:
                # Standard settings for lower memory systems
                self.morphology_kernel = np.ones((5, 5), np.uint8)  # Simple kernel
                self.use_enhanced_detection = False
                
                # Use slightly smaller images by default to conserve RAM
                self.default_scale_factor = 0.75
                
                self.get_logger().info(f"Using standard detection features (limited RAM mode)")
        except Exception as e:
            # Default settings if we can't check memory
            self.morphology_kernel = np.ones((5, 5), np.uint8)
            self.use_enhanced_detection = False
            self.default_scale_factor = 0.75
            self.get_logger().warn(f"Could not determine system memory. Using default settings: {e}")
        
        # Initialize current scale to default
        self.current_scale_factor = self.default_scale_factor
        
        # Number of frames to skip in low power mode (0 means no skipping)
        self.low_power_skip_frames = 0

    def _check_cpu_and_adjust_processing(self):
        """Check CPU usage and adjust processing quality/rate accordingly."""
        try:
            # Get current CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)  # Quick sampling
            self.current_cpu_usage = cpu_usage
            
            # Add to history
            self.cpu_history.append((TimeUtils.now_as_float(), cpu_usage))
            
            # Publish CPU usage
            cpu_msg = Float32()
            cpu_msg.data = float(cpu_usage)
            self.cpu_usage_publisher.publish(cpu_msg)
            
            # Current scale and skip settings before adjustments
            old_scale = self.current_scale_factor
            old_skip = self.low_power_skip_frames
            
            # Adjust processing based on CPU load
            if cpu_usage > self.cpu_high_threshold:
                # Very high CPU - dramatic reduction
                self.current_scale_factor = self.high_load_scale
                self.low_power_skip_frames = 2  # Process only 1 in 3 frames
                
                if old_scale != self.current_scale_factor or old_skip != self.low_power_skip_frames:
                    self.get_logger().warn(
                        f"CPU usage very high ({cpu_usage:.1f}%): reducing resolution to "
                        f"{int(100*self.current_scale_factor)}% and processing 1 in {self.low_power_skip_frames+1} frames"
                    )
                    
                    # Record adaptation
                    self.adaptation_history.append({
                        'time': TimeUtils.now_as_float(),
                        'cpu': cpu_usage,
                        'action': 'high_reduction',
                        'scale': self.current_scale_factor,
                        'skip': self.low_power_skip_frames
                    })
                    
            elif cpu_usage > self.cpu_medium_threshold:
                # Moderately high CPU - medium reduction
                self.current_scale_factor = self.medium_load_scale
                self.low_power_skip_frames = 1  # Process every other frame
                
                if old_scale != self.current_scale_factor or old_skip != self.low_power_skip_frames:
                    self.get_logger().info(
                        f"CPU usage high ({cpu_usage:.1f}%): reducing resolution to "
                        f"{int(100*self.current_scale_factor)}% and processing every other frame"
                    )
                    
                    # Record adaptation
                    self.adaptation_history.append({
                        'time': TimeUtils.now_as_float(),
                        'cpu': cpu_usage,
                        'action': 'medium_reduction',
                        'scale': self.current_scale_factor,
                        'skip': self.low_power_skip_frames
                    })
                    
            elif cpu_usage < self.cpu_low_threshold:
                # Low CPU - restore full processing if we were reducing
                if self.current_scale_factor < 1.0 or self.low_power_skip_frames > 0:
                    self.current_scale_factor = self.default_scale_factor
                    self.low_power_skip_frames = 0  # Process all frames
                    
                    self.get_logger().info(
                        f"CPU usage normal ({cpu_usage:.1f}%): restoring normal processing "
                        f"at {int(100*self.current_scale_factor)}% resolution"
                    )
                    
                    # Record adaptation
                    self.adaptation_history.append({
                        'time': TimeUtils.now_as_float(),
                        'cpu': cpu_usage,
                        'action': 'restore_normal',
                        'scale': self.current_scale_factor,
                        'skip': self.low_power_skip_frames
                    })
            
            # Update diagnostic metrics with current CPU usage and adaptations
            if hasattr(self, 'diagnostic_metrics'):
                # Store current settings
                self.diagnostic_metrics['current_cpu'] = cpu_usage
                self.diagnostic_metrics['current_scale'] = self.current_scale_factor
                self.diagnostic_metrics['frame_skip'] = self.low_power_skip_frames
                
                # Record adaptation if changed
                if old_scale != self.current_scale_factor or old_skip != self.low_power_skip_frames:
                    self.diagnostic_metrics['adaptations'].append({
                        'timestamp': TimeUtils.now_as_float(),
                        'cpu': cpu_usage,
                        'old_scale': old_scale,
                        'new_scale': self.current_scale_factor,
                        'old_skip': old_skip,
                        'new_skip': self.low_power_skip_frames
                    })
            
        except Exception as e:
            self.get_logger().error(f"Error in CPU monitoring: {e}")

    def _handle_resource_alert(self, resource_type, value):
        """Handle resource alerts by adjusting processing behavior."""
        self.get_logger().warn(f"Resource alert: {resource_type.upper()} at {value:.1f}%")
        
        # If CPU usage is critically high, implement more aggressive measures
        if resource_type == 'cpu' and value > 95.0:  # Extremely high CPU
            old_skip = self.low_power_skip_frames
            old_scale = self.current_scale_factor
            
            # Emergency measures - very low resolution and high frame skipping
            self.low_power_skip_frames = 3  # Skip 3 frames, process 1
            self.current_scale_factor = 0.4  # 40% of original resolution
            
            self.get_logger().warn(
                f"CRITICAL CPU USAGE: Emergency reduction to {int(100*self.current_scale_factor)}% "
                f"resolution and 1 in {self.low_power_skip_frames+1} frames"
            )
            
            # Record for diagnostics
            if hasattr(self, 'diagnostic_metrics'):
                self.diagnostic_metrics['adaptations'].append({
                    'timestamp': TimeUtils.now_as_float(),
                    'resource_type': resource_type,
                    'value': value,
                    'action': 'emergency_reduction',
                    'old_scale': old_scale,
                    'new_scale': self.current_scale_factor,
                    'old_skip': old_skip,
                    'new_skip': self.low_power_skip_frames
                })

    def _configure_logging(self, log_config):
        # ...existing code...
        pass

    def _log(self, level, context, message, data=None):
        # ...existing code...
        pass

    def _generate_trace_id(self):
        # ...existing code...
        pass

    def _init_log_file(self):
        # ...existing code...
        pass

    def image_callback(self, msg):
        """
        Process each incoming camera image to detect tennis balls.
        
        This method:
        1. Checks if we should process this frame based on CPU load
        2. Converts the ROS image to OpenCV format
        3. Resizes to appropriate scale for CPU load
        4. Applies optimized HSV detection pipeline
        5. Publishes the position of the detected ball if found
        
        Args:
            msg (Image): The incoming camera image from ROS
        """
        # Check if we need to skip this frame based on CPU load
        if self.low_power_skip_frames > 0:
            if not hasattr(self, 'frame_skip_counter'):
                self.frame_skip_counter = 0
            
            self.frame_skip_counter += 1
            if (self.frame_skip_counter % (self.low_power_skip_frames + 1)) != 0:
                # Skip this frame - just count it but don't process
                self.frame_count += 1  # Still count it for metrics
                self.skip_count += 1
                return
        
        # Check minimum time between frames for rate limiting
        current_time = TimeUtils.now_as_float()
        time_since_last_frame = current_time - self.last_frame_time
        
        if time_since_last_frame < self.min_processing_interval:
            # Too soon since last frame - enforce minimum interval
            self.skip_count += 1
            return
            
        # Update last frame time
        self.last_frame_time = current_time
        
        # Start timing for performance metrics
        processing_start = TimeUtils.now_as_float()
        self.frame_count += 1
        
        trace_id = self._generate_trace_id()
        if self.debug_level >= 2:  # Only log at debug level 2+
            self._log('debug', 'FRAME', f"Processing frame {self.frame_count}", {'trace_id': trace_id})

        try:
            # STEP 1: Convert ROS image to OpenCV format efficiently
            # Use direct array reference when possible instead of copy
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # STEP 2: Apply resolution scaling based on CPU load
            # Scale the image based on current CPU load factor
            width = int(self.target_width * self.current_scale_factor)
            height = int(self.target_height * self.current_scale_factor)
            
            # Use pre-allocated buffer if available
            if self.current_scale_factor in self.image_buffers:
                # Resize directly into pre-allocated buffer
                cv2.resize(frame, (width, height), dst=self.image_buffers[self.current_scale_factor]['bgr'])
                frame = self.image_buffers[self.current_scale_factor]['bgr']
            else:
                # Fallback - create a new buffer
                frame = cv2.resize(frame, (width, height))
            
            # Create a copy for visualization if enabled (only when needed)
            if self.enable_visualization:
                display_frame = frame.copy()
            
            # STEP 3: Apply HSV color filtering with optimized pipeline
            # Pass current scale factor to adjust detection parameters
            detected_ball = self._detect_ball_in_frame(frame, msg.header, trace_id)
            
            # STEP 4: Update visualization if enabled
            if self.enable_visualization:
                self._update_visualization(frame, detected_ball, processing_start)
            
            # Record processing time
            processing_time = TimeUtils.now_as_float() - processing_start
            self.processing_times.append(processing_time)
            
            # Log performance metrics occasionally
            if self.frame_count % self.log_interval == 0:
                self._log_performance_metrics(processing_start)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())

    def _detect_ball_in_frame(self, frame, header, trace_id=None):
        """
        Apply optimized HSV color filtering to detect a tennis ball in the frame.
        
        Args:
            frame (numpy.ndarray): OpenCV image in BGR format
            header (Header): ROS message header from the original image
            trace_id (str, optional): Trace ID for correlation in logs
            
        Returns:
            dict: Detection information or None if no ball found
        """
        # Get current scale factor for adjusting parameters
        scale_factor = self.current_scale_factor
        
        # Create a copy for visualization if enabled
        if self.enable_visualization:
            display_frame = frame.copy()
        
        # STEP 1: Convert from BGR to HSV color space efficiently
        # Use pre-allocated buffer if available
        if scale_factor in self.image_buffers:
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self.image_buffers[scale_factor]['hsv'])
            hsv = self.image_buffers[scale_factor]['hsv']
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # STEP 2: Create a mask that only shows yellow pixels (optimized)
        # Use pre-allocated buffer if available
        if scale_factor in self.image_buffers:
            cv2.inRange(hsv, self.lower_yellow, self.upper_yellow, dst=self.image_buffers[scale_factor]['mask'])
            mask = self.image_buffers[scale_factor]['mask']
        else:
            mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        
        # STEP 3: Clean up the mask with morphological operations
        # Use smaller kernel size for lower resolutions
        if scale_factor < 0.6:  # Very small images
            kernel = self.morph_kernels['small']  # Use 3x3 kernel
        elif scale_factor < 0.8:
            kernel = self.morph_kernels['medium']  # Use 5x5 kernel
        else:
            kernel = self.morph_kernels['medium']  # Use 5x5 kernel for full resolution
            
        # Apply morphology in-place to avoid allocations
        cv2.erode(mask, kernel, dst=mask, iterations=1)
        cv2.dilate(mask, kernel, dst=mask, iterations=2)
        
        # Save the processed mask for visualization
        if self.enable_visualization:
            display_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # STEP 4: Find contours efficiently
        # Use CHAIN_APPROX_SIMPLE to reduce points
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Skip processing if no significant contours
        if not contours or len(contours) == 0:
            self.no_detection_count += 1
            return None
        
        # STEP 5: Filter the contours efficiently 
        # Scale area thresholds based on current resolution
        area_scale = scale_factor * scale_factor  # Area scales with square of linear dimension
        min_area = self.min_ball_area * area_scale
        max_area = self.max_ball_area * area_scale
        ideal_area = self.ideal_area * area_scale
        
        # Pre-filter tiny contours before detailed analysis
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area * 0.5:  # Use 50% of min as pre-filter
                filtered_contours.append((cnt, area))
        
        # If no contours pass pre-filtering, return early
        if not filtered_contours:
            self.no_detection_count += 1
            return None
            
        # Variables to track the best ball candidate
        best_contour = None
        best_radius = 0.0
        best_confidence = 0.0
        best_center = (0, 0)
        best_area = 0.0
        best_circularity = 0.0
        
        # STEP 6: Find the best ball candidate
        for cnt, area in filtered_contours:
            # Skip contours outside our area range
            if area < min_area or area > max_area:
                continue
                
            # Find the smallest circle that can enclose the contour
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            
            # Calculate "circularity" - how close to a perfect circle
            # A perfect circle has circularity close to 1.0
            circle_area = np.pi * (radius ** 2) if radius > 0 else 1
            circularity = area / circle_area
            
            # Skip if circularity is outside acceptable range
            if circularity < self.min_circularity or circularity > self.max_circularity:
                continue
            
            # Draw contour on visualization if enabled
            if self.enable_visualization:
                cv2.drawContours(display_frame, [cnt], -1, (255, 0, 0), 1)
                cv2.putText(display_frame, f"{area:.0f}", (int(cx), int(cy)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # Calculate optimized confidence score
            circularity_score = 1.0 - min(abs(circularity - self.ideal_circularity) / 
                                         self.ideal_circularity, 1.0)
            size_score = 1.0 - min(abs(area - ideal_area) / ideal_area, 1.0)
            
            # Combined confidence score (weighted average)
            confidence = (circularity_score * 0.7) + (size_score * 0.3)
            
            # Keep the highest confidence match
            if confidence > best_confidence:
                best_contour = cnt
                best_radius = radius
                best_confidence = confidence
                best_center = (cx, cy)
                best_area = area
                best_circularity = circularity
        
        # Optional enhanced detection for Pi 5 with 16GB
        # Only run if we have enough CPU resources (not in high load)
        if (hasattr(self, 'use_enhanced_detection') and 
            self.use_enhanced_detection and 
            self.current_cpu_usage < self.cpu_high_threshold and
            len(filtered_contours) > 0):
            
            try:
                # Get the largest contour by area for enhanced processing
                largest_cnt, largest_area = max(filtered_contours, key=lambda x: x[1])
                
                # Only attempt circle detection if we have a significant contour
                if largest_area > min_area and largest_area < max_area:
                    # Create a mask with just the largest contour
                    largest_mask = np.zeros_like(mask)
                    cv2.drawContours(largest_mask, [largest_cnt], 0, 255, -1)
                    
                    # Apply Hough Circle detection with adaptive parameters based on scale
                    min_radius = int(np.sqrt(min_area/np.pi))
                    max_radius = int(np.sqrt(max_area/np.pi))
                    
                    # Only run HoughCircles if we have a valid contour
                    detected_circles = cv2.HoughCircles(
                        mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                        param1=50, param2=10, 
                        minRadius=min_radius,
                        maxRadius=max_radius
                    )
                    
                    # If circles are found, incorporate into detection
                    if detected_circles is not None:
                        # Convert circles to integer coordinates
                        detected_circles = np.round(detected_circles[0, :]).astype(int)
                        
                        # Find the best circle (most confident based on HoughCircles)
                        for (x, y, r) in detected_circles:
                            # Calculate how well circle matches contour
                            circle_center = (float(x), float(y))
                            circle_radius = float(r)
                            circle_area = np.pi * r * r
                            
                            # Only consider if better than current best
                            if circle_area > min_area and circle_area < max_area:
                                # Check if this circle improves detection
                                if best_confidence < 0.7:  # Only replace if current confidence is low
                                    circle_confidence = 0.8  # Default confidence from HoughCircles
                                    
                                    # Use the circle instead of contour if it's more reliable
                                    best_center = circle_center
                                    best_radius = circle_radius
                                    best_area = circle_area
                                    best_confidence = max(best_confidence, circle_confidence)
                                    
                                    # Don't update best_contour - keep original
                                    
                                    if self.debug_level >= 2:
                                        self.get_logger().debug("Enhanced detection improved result")
                                    break
            except Exception as e:
                # Ignore errors in enhanced detection - fall back to standard
                if self.debug_level >= 2:
                    self.get_logger().debug(f"Enhanced detection error: {e}")
        
        # STEP 7: Process the best match if found
        if best_contour is not None:
            # Scale coordinates back to original resolution (320x320) for consistent reporting
            scale_back = 1.0 / scale_factor
            
            # Unpack the center coordinates and scale back
            center_x, center_y = best_center
            center_x *= scale_back
            center_y *= scale_back
            
            # Log the detection (less frequently when CPU is high)
            log_this_detection = self.debug_level >= 1 and (
                self.current_cpu_usage < self.cpu_medium_threshold or 
                self.frame_count % 10 == 0  # Less logging in high CPU
            )
            
            if log_this_detection:
                self.get_logger().info(
                    f"FOUND BALL at ({center_x:.1f}, {center_y:.1f}) "
                    f"radius: {best_radius*scale_back:.1f}, area: {best_area*scale_back*scale_back:.1f}, "
                    f"confidence: {best_confidence:.2f}, scale: {scale_factor:.2f}"
                )
            
            # Create and publish the position message
            position_msg = PointStamped()
            
            # Use original image timestamp for synchronization
            if TimeUtils.is_timestamp_valid(header.stamp):
                position_msg.header.stamp = header.stamp
            else:
                position_msg.header.stamp = TimeUtils.now_as_ros_time()
                if self.debug_level >= 2:
                    self.get_logger().debug("Using current time (invalid original timestamp)")
                
            position_msg.header.frame_id = "ascamera_color_0"  # Camera frame
            
            position_msg.point.x = float(center_x)
            position_msg.point.y = float(center_y)
            position_msg.point.z = float(best_confidence)  # Use z for confidence
            
            # Publish the ball position
            self.ball_publisher.publish(position_msg)
            
            # Reset no detection counter and update statistics
            self.no_detection_count = 0
            self.detection_count += 1
            self.last_detection_time = TimeUtils.now_as_float()
            
            # Store detection metrics for statistics
            self.detection_sizes.append(best_area * scale_back * scale_back)  # Scale back area
            self.detection_confidences.append(best_confidence)
            
            # Store for diagnostics
            if hasattr(self, 'diagnostic_metrics'):
                self.diagnostic_metrics['last_detection_position'] = (center_x, center_y)
                self.diagnostic_metrics['last_detection_time'] = TimeUtils.now_as_float()
            
            # Return detection information
            return {
                'center': (center_x, center_y),
                'radius': best_radius * scale_back,
                'area': best_area * scale_back * scale_back,
                'circularity': best_circularity,
                'confidence': best_confidence,
                'contour': best_contour,
                'scale_factor': scale_factor
            }
        else:
            # No ball found
            self.no_detection_count += 1
            
            # Track missed frames for diagnostics
            if hasattr(self, 'diagnostic_metrics'):
                self.diagnostic_metrics['missed_frames'] += 1
                
            # Log "no ball found" at specified intervals (less frequently when CPU is high)
            if self.no_detection_count % (self.log_interval * (1 + int(self.current_cpu_usage > 70))) == 0:
                self._log_no_detection_info(contours)
            
            return None

    def _log_no_detection_info(self, contours):
        """Log detailed information about why no ball was detected."""
        # Only log detailed info if CPU isn't too high
        if self.current_cpu_usage > self.cpu_high_threshold:
            # Just brief logging at high CPU
            if self.no_detection_count % 20 == 0:  # Very occasional logging
                self.get_logger().info(f"No ball detected for {self.no_detection_count} frames (high CPU mode)")
            return
            
        # Normal detailed logging
        self.get_logger().info(f"NO BALL FOUND (for {self.no_detection_count} consecutive frames)")
        
        # Get scaled area thresholds
        scale_factor = self.current_scale_factor
        area_scale = scale_factor * scale_factor
        min_area = self.min_ball_area * area_scale
        max_area = self.max_ball_area * area_scale
        
        # If there were yellow objects, explain why they weren't detected as balls
        if contours and len(contours) > 0:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            
            if largest_area > 20 * area_scale:  # Only report significant blobs
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                circle_area = np.pi * (radius ** 2) if radius > 0 else 1
                circularity = largest_area / circle_area
                
                # Explain why it was rejected
                reason = "unknown reason"
                if largest_area < min_area:
                    reason = f"too small (area={largest_area:.0f}, min={min_area:.0f})"
                elif largest_area > max_area:
                    reason = f"too large (area={largest_area:.0f}, max={max_area:.0f})"
                elif circularity < self.min_circularity:
                    reason = f"not circular enough (circularity={circularity:.2f}, min={self.min_circularity})"
                elif circularity > self.max_circularity:
                    reason = f"too circular (circularity={circularity:.2f}, max={self.max_circularity})"
                
                self.get_logger().info(f"Largest yellow object rejected because: {reason}")

    def _update_visualization(self, frame, detected_ball, processing_start):
        # ...existing code...
        pass

    def _log_performance_metrics(self, processing_start):
        # ...existing code...
        pass

    def publish_system_diagnostics(self):
        """Publish comprehensive system diagnostics including adaptive processing info."""
        if not hasattr(self, 'diagnostic_metrics'):
            return  # Not enough data collected yet
            
        current_time = TimeUtils.now_as_float()
        elapsed_time = current_time - self.start_time
        
        # Calculate average metrics
        avg_fps = np.mean(list(self.diagnostic_metrics['fps_history'])) if self.diagnostic_metrics['fps_history'] else 0.0
        avg_processing_time = np.mean(list(self.diagnostic_metrics['processing_time_history'])) if self.diagnostic_metrics['processing_time_history'] else 0.0
        avg_detection_rate = np.mean(list(self.diagnostic_metrics['detection_rate_history'])) if self.diagnostic_metrics['detection_rate_history'] else 0.0
        
        # Time since last detection
        time_since_detection = 0
        if self.diagnostic_metrics['last_detection_time'] > 0:
            time_since_detection = current_time - self.diagnostic_metrics['last_detection_time']
        else:
            time_since_detection = float('inf')
        
        # Build warnings list
        warnings = []
        errors = []
        
        # Check for performance issues
        if avg_fps < 10.0 and elapsed_time > 10.0:
            warnings.append(f"Low FPS: {avg_fps:.1f}")
            
        if avg_processing_time > 50.0:  # 50ms is slow
            warnings.append(f"High processing time: {avg_processing_time:.1f}ms")
            
        # Check for detection issues
        if time_since_detection > 5.0 and elapsed_time > 10.0:
            warnings.append(f"No ball detected for {time_since_detection:.1f}s")
            
        if avg_detection_rate < 0.1 and elapsed_time > 10.0:  # Less than 10% detection rate
            errors.append(f"Very low detection rate: {avg_detection_rate*100:.1f}%")
        
        # System resources
        system_resources = {}
        try:
            system_resources = {
                'cpu_percent': self.current_cpu_usage,  # Use stored value
                'memory_percent': psutil.virtual_memory().percent
            }
            
            # Check for high resource usage
            if system_resources['cpu_percent'] > 80.0:
                warnings.append(f"High CPU usage: {system_resources['cpu_percent']:.1f}%")
                
            # Add temperature if available
            if hasattr(psutil, 'sensors_temperatures'):
                try:
                    temps = psutil.sensors_temperatures()
                    if temps and 'cpu_thermal' in temps:
                        system_resources['temperature'] = temps['cpu_thermal'][0].current
                except:
                    # Temperature reading can fail silently
                    pass
        except Exception as e:
            # Handle any errors accessing system metrics
            self.get_logger().warn(f"Error getting system resources: {e}")
        
        # Build diagnostics data structure
        diag_data = {
            "node": "hsv",
            "timestamp": current_time,
            "uptime_seconds": elapsed_time,
            "status": "error" if errors else ("warning" if warnings else "active"),
            "health": {
                "camera_health": 1.0 - (len(warnings) * 0.1),
                "detection_health": avg_detection_rate if avg_detection_rate > 0 else 0.5,
                "processing_health": 1.0 - (avg_processing_time / 100.0) if avg_processing_time < 100.0 else 0.0,
                "overall": 1.0 - (len(errors) * 0.3) - (len(warnings) * 0.1)
            },
            "metrics": {
                "fps": avg_fps,
                "processing_time_ms": avg_processing_time,
                "total_frames": self.diagnostic_metrics['total_frames'],
                "missed_frames": self.diagnostic_metrics['missed_frames'],
                "detection_rate": avg_detection_rate
            },
            "detection": {
                "latest_position": self.diagnostic_metrics['last_detection_position'],
                "time_since_last_detection_s": time_since_detection,
                "currently_tracking": time_since_detection < 1.0
            },
            "configuration": {
                "hsv_range": {
                    "lower": self.lower_yellow.tolist(),
                    "upper": self.upper_yellow.tolist()
                },
                "area_range": [self.min_ball_area, self.max_ball_area],
                "circularity_range": [self.min_circularity, self.max_circularity]
            },
            "resources": system_resources,
            "adaptive_processing": {
                "current_scale_factor": self.current_scale_factor,
                "frame_skip_count": self.low_power_skip_frames,
                "skipped_frames": self.skip_count,
                "cpu_thresholds": {
                    "high": self.cpu_high_threshold,
                    "medium": self.cpu_medium_threshold,
                    "low": self.cpu_low_threshold
                }
            },
            "errors": errors,
            "warnings": warnings
        }
        
        # Publish as JSON
        msg = String()
        msg.data = json.dumps(diag_data)
        self.system_diagnostics_publisher.publish(msg)
        
        # Also log to console (condensed in high CPU)
        if self.current_cpu_usage < self.cpu_medium_threshold:
            # Normal detailed logging
            self.get_logger().info(
                f"HSV diagnostics: {avg_fps:.1f} FPS, {avg_detection_rate*100:.1f}% detection rate, "
                f"Status: {diag_data['status']}, Scale: {self.current_scale_factor:.2f}, "
                f"Skip: {self.low_power_skip_frames}"
            )
        else:
            # Condensed logging in high CPU
            self.get_logger().info(
                f"HSV status: {diag_data['status']}, CPU: {self.current_cpu_usage:.1f}%, "
                f"Scale: {self.current_scale_factor:.2f}"
            )

    @lru_cache(maxsize=8)  # Cache results for better performance
    def _calculate_scaled_thresholds(self, scale_factor):
        """Calculate area thresholds scaled by current resolution factor."""
        # Area scales with the square of linear dimensions
        area_scale = scale_factor * scale_factor
        return {
            'min_area': self.min_ball_area * area_scale,
            'max_area': self.max_ball_area * area_scale,
            'ideal_area': self.ideal_area * area_scale
        }

    def _check_health_metrics(self):
        """Evaluate system health and make adjustments if needed."""
        # Most of the health adaptation is now handled by _check_cpu_and_adjust_processing
        pass

    def destroy_node(self):
        """Ensure proper cleanup of resources."""
        # Release any capture objects
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        # Close OpenCV windows if enabled
        if hasattr(self, 'enable_visualization') and self.enable_visualization:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                self.get_logger().warn(f"Error closing OpenCV windows: {str(e)}")
        
        # Stop threads and timers
        if hasattr(self, 'resource_monitor') and self.resource_monitor:
            self.resource_monitor.stop()
            
        if hasattr(self, 'cpu_check_timer'):
            self.cpu_check_timer.cancel()
            
        # Clear cached data
        if hasattr(self, '_calculate_scaled_thresholds'):
            self._calculate_scaled_thresholds.cache_clear()
            
        # Clear image buffers
        if hasattr(self, 'image_buffers'):
            self.image_buffers.clear()
            
        super().destroy_node()

    def _collect_debug_data(self, frame, mask, contours, detection):
        # ...existing code...
        pass

def main(args=None):
    """Main function to initialize and run the HSV Tennis Ball Tracker node."""
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create our HSV tennis ball tracker node
    node = HSVTennisBallTracker()
    
    # Welcome message
    print("=================================================")
    print("Tennis Ball Tracking - HSV Ball Detector Node")
    print("=================================================")
    print("This node uses HSV color filtering to detect tennis balls.")
    print(f"Processing images at {node.target_width}x{node.target_height} to match YOLO")
    print("")
    print("Subscriptions:")
    print(f"  - Camera: {TOPICS['input']['camera']}")
    print("")
    print("Publications:")
    print(f"  - Ball position: {TOPICS['output']['position']}")
    print(f"  - CPU usage: /system/resources/cpu_load")
    print("")
    print("Performance Adaptation:")
    print(f"  - High CPU threshold: {node.cpu_high_threshold}%")
    print(f"  - Medium CPU threshold: {node.cpu_medium_threshold}%")
    print("")
    print("Press Ctrl+C to stop the program")
    print("=================================================")
    
    try:
        # On Pi 5, use process priority to balance with other nodes
        try:
            import os
            os.nice(5)  # Slightly lower priority than critical nodes
            print("Set HSV tracker to adjusted process priority")
        except:
            pass
        
        # Keep the node running until interrupted
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Stopping HSV tracker (Ctrl+C pressed)")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Clean shutdown
        if node.enable_visualization:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        print("HSV Tennis Ball Tracker has been shut down.")

if __name__ == '__main__':
    main()