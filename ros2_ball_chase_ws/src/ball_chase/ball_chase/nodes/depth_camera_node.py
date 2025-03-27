#!/usr/bin/env python3

"""
Raspberry Pi 5 Ultra-Optimized Basketball Tracking - Depth Camera Node
======================================================================

Highly efficient implementation for basketball tracking designed
specifically for the Raspberry Pi 5's resource constraints.
"""
# Standard library imports - only import what's needed
import os
import time
from collections import deque
import psutil  # Add psutil for accurate CPU monitoring
import sys  # For immediate log flushing

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor  # Changed from MultiThreadedExecutor

# ROS2 message types
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
# Third-party libraries
import numpy as np
from cv_bridge import CvBridge

# Project utilities
from ball_chase.utilities.resource_monitor import ResourceMonitor
from ball_chase.utilities.time_utils import TimeUtils
from ball_chase.config.config_loader import ConfigLoader
from ball_chase.utilities.ground_position_filter import GroundPositionFilter  # Import the shared filter
from ball_chase.utilities.performance_metrics import PerformanceMetrics

# Config loading - done once at module level
config_loader = ConfigLoader()
config = config_loader.load_yaml('depth_config.yaml')

# Configuration from config file
DEPTH_CONFIG = config.get('depth', {
    "scale": 0.001,           # Depth scale factor (converts raw depth to meters)
    "min_depth": 0.1,         # Minimum valid depth in meters
    "max_depth": 8.0,         # Maximum valid depth in meters
    "radius": 5,              # Radius around detection point to sample depth values
    "min_valid_points": 5,    # Minimum number of valid points required for reliable estimation
    "calibration_file": "depth_camera_calibration.yaml",  # Calibration parameters file
    "enable_visualization": False,  # Disable visualization by default for performance
    "ultra_low_power_mode": False,  # New option for extreme power saving
    "adaptive_roi": True,     # New option to enable adaptive ROI sizing
    "debug_depth": True,      # New option to debug depth sampling issues
    "use_depth_history": True, # Use history-based depth recovery
    "max_roi_size": 60,       # Maximum ROI size for problematic regions
    "history_max_age": 3.0,   # Maximum age for depth history entries (seconds)
    "temporal_blending": 0.7, # Weight for temporal depth blending (0-1, higher = more weight to new data)
    "use_dynamic_sampling": True, # Use dynamic sampling strategy that expands from center
    "use_neighbor_data": True, # Allow using depth data from neighboring regions
    "ultra_fast_path": False,     # Use ultra-fast but less reliable direct pixel access
    "min_roi_size": 15,           # Restored to 15 (was reduced to 12)
    "quality_preference": 0.7,    # Restored to 0.7 (was reduced to 0.5) - favor quality
    "parallel_processing": True,  # Keep parallel processing 
    "fast_path_optimization": True,  # Keep fast path optimization
    "adaptive_frame_skip": True,    # Keep adaptive frame skipping
    "min_points_threshold": 3,    # Restored to 3 (was reduced to 2)
    "aggressive_caching": False,   # Disabled aggressive caching that hurt quality
    "skip_roi_expansion": False    # Disabled ROI expansion skipping
})

# Topic configuration from config file
TOPICS = config.get('topics', {
    "input": {
        "camera_info": "/ascamera/camera_publisher/depth0/camera_info",
        "depth_image": "/ascamera/camera_publisher/depth0/image_raw",
        "yolo_detection": "/basketball/yolo/position",
        "hsv_detection": "/basketball/hsv/position"
    },
    "output": {
        "yolo_3d": "/basketball/yolo/position_3d",
        "hsv_3d": "/basketball/hsv/position_3d",
        "combined": "/basketball/detected_position"  # Legacy/combined topic
    }
})

# Diagnostic configuration
DIAG_CONFIG = config.get('diagnostics', {
    "log_interval": 30.0,      # Increased from 15.0 to 30.0 seconds
})

# Define the common reference frame for the robot
COMMON_REFERENCE_FRAME = config.get('frames', {
    "reference_frame": "base_link",  # Common reference frame for all sensors
    "transform_timeout": 0.1          # Timeout for transform lookups in seconds
})

# Minimal logging by default
MINIMAL_LOGGING = False  # Changed to False to allow more verbose logging
LOG_INTERVAL = 10.0  # Reduced from 30.0 to 10.0 seconds for more frequent logs


class DepthCorrector:
    """Applies calibration corrections to depth camera measurements."""
    
    def __init__(self, calibration_file):
        """Initialize the depth corrector with calibration parameters."""
        self.correction_type = None
        self.parameters = None
        self.mean_error = None
        self.loaded = self.load_calibration(calibration_file)
        
        # Initialize cache for frequent depth values
        self.correction_cache = {}
        self.max_cache_size = 2000  # Increased from 1000 to 2000 for better hit rate
        
        if not self.loaded:
            print("WARNING: No valid calibration found - using identity correction")
            self.correction_type = "identity"
    
    def load_calibration(self, calibration_file):
        """Load calibration parameters from file."""
        try:
            if os.path.exists(calibration_file):
                with open(calibration_file, 'r') as f:
                    calibration = yaml.safe_load(f)
                
                # Extract calibration parameters
                correction_data = calibration.get('depth_correction', {})
                self.correction_type = correction_data.get('type')
                self.parameters = correction_data.get('parameters', [])
                self.mean_error = correction_data.get('mean_error')
                
                if self.correction_type and self.parameters:
                    print(f"Loaded depth correction: {self.correction_type}")
                    print(f"Parameters: {self.parameters}")
                    print(f"Expected accuracy: ±{self.mean_error:.3f}m")
                    return True
                else:
                    print(f"WARNING: Invalid calibration format in {calibration_file}")
                    return False
            else:
                print(f"WARNING: Calibration file '{calibration_file}' not found")
                return False
        except Exception as e:
            print(f"ERROR loading calibration: {str(e)}")
            return False
    
    def correct_depth(self, measured_depth):
        """Apply calibration correction to a depth measurement with caching."""
        if not self.loaded or self.correction_type is None:
            return measured_depth  # No correction available
        
        # Round to 3 decimals for cache key
        cache_key = round(measured_depth, 3)
        
        # Check cache first
        if cache_key in self.correction_cache:
            return self.correction_cache[cache_key]
        
        try:
            result = measured_depth  # Default to identity
            
            if self.correction_type == "linear":
                # Linear correction: true = a * measured + b
                a, b = self.parameters
                result = a * measured_depth + b
            
            elif self.correction_type == "polynomial":
                # Polynomial correction: true = a * measured^2 + b * measured + c
                a, b, c = self.parameters
                result = a * measured_depth**2 + b * measured_depth + c
            
            # Cache the result if cache isn't too large
            if len(self.correction_cache) < self.max_cache_size:
                self.correction_cache[cache_key] = result
            
            return result
            
        except Exception:
            # If any error occurs, return original measurement
            return measured_depth


class OptimizedPositionEstimator(Node):
    """
    A ROS2 node that converts 2D ball detections to 3D positions.
    Optimized for basketball tracking on Raspberry Pi 5.
    """
    
    def __init__(self):
        """Initialize the 3D position estimator node with all required components."""
        super().__init__('basketball_3d_position_estimator')
        
        # Initialize core attributes
        self._init_attributes()
        
        # Initialize the depth corrector for calibration
        calibration_file = DEPTH_CONFIG.get("calibration_file", "depth_camera_calibration.yaml")
        self.depth_corrector = DepthCorrector(calibration_file)
        
        # Setup in logical order
        self._setup_callback_group()
        self._init_camera_parameters()
        self._setup_tf2()
        self._setup_subscriptions()
        self._setup_publishers()
        
        # Performance optimization settings
        self.use_roi_only = True  # Process only regions of interest
        self.roi_size = 20       # Default: 20x20 pixel region around detection (smaller than original)
        self.log_cache_info = False  # Disable cache debugging by default to reduce overhead
        self.max_cpu_target = 75.0  # Reduced target max CPU usage from 80.0
        self.last_cpu_log = 0  # Timestamp for CPU logging
        
        # Fixed caching structure that separates 2D and 3D positions
        self.detection_cache = {
            'YOLO': {'detection_2d': None, 'position_3d': None, 'timestamp': 0},
            'HSV': {'detection_2d': None, 'position_3d': None, 'timestamp': 0}
        }
        self.cache_hits = 0
        self.total_attempts = 0
        
        # Initialize the resource monitor with reduced frequency
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=45.0,  # Increased from 30.0 to 45.0 seconds
            enable_temperature=False
        )
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.resource_monitor.start()
        
        # Performance adjustment timer (reduced frequency)
        self.performance_timer = self.create_timer(15.0, self._adjust_performance)  # Increased from 10.0 to 15.0
        self.diagnostics_timer = self.create_timer(30.0, self.publish_system_diagnostics)  # Increased from 15.0 to 30.0
        
        # Pre-allocate message objects for reuse (reduces memory allocations)
        self._preallocate_messages()
        
        # Ultra low power mode check
        self.ultra_low_power = DEPTH_CONFIG.get("ultra_low_power_mode", False)
        if self.ultra_low_power:
            # Start with very conservative settings in ultra-low-power mode
            self.process_every_n_frames = 4
            self.roi_size = 15
            self.get_logger().info("Starting in ULTRA-LOW-POWER mode with aggressive optimization")
        
        # Debug flags
        self.debug_mode = False  # Full debug with verbose logging - off by default
        self.debug_depth = DEPTH_CONFIG.get("debug_depth", False)  # Specific flag for depth debugging
        self.last_debug_log = 0
        
        # Log initialization (minimal)
        self.get_logger().info("Pi-Optimized 3D Position Estimator initialized")
        
        # Force flush logs
        sys.stdout.flush()
        
    def _init_attributes(self):
        """Initialize all attributes with default values."""
        # Performance settings
        self.process_every_n_frames = 2  # Reverting to process every 2nd frame for better performance
        self.frame_counter = 0
        self.current_cpu_usage = 0.0
        self._scale_factor = float(DEPTH_CONFIG["scale"])
        self._min_valid_depth = float(DEPTH_CONFIG["min_depth"])
        self._max_valid_depth = float(DEPTH_CONFIG["max_depth"])
        
        # Debug flags
        self.debug_mode = False  # Full debug with verbose logging - off by default
        self.last_debug_log = 0
        
        # Add region-based historical depth cache
        self.depth_region_cache = {}  # Cache for depth values by region
        self.region_grid_size = 20    # Split the image into regions of this size
        self.depth_region_ttl = 1.0   # Time-to-live for cached region depths (seconds)
        self.depth_history_max_age = DEPTH_CONFIG.get("history_max_age", 3.0)  # Maximum age for cached values
        self.use_depth_history = DEPTH_CONFIG.get("use_depth_history", True)   # Enable depth history
        self.max_roi_size = DEPTH_CONFIG.get("max_roi_size", 60)  # Maximum ROI size
        self.depth_region_stats = {}  # Statistics about depth measurements by region
        self.depth_history = {}       # Historical depth values by region
        
        # Performance tracking - use the new PerformanceMetrics utility
        
        self.performance_metrics = PerformanceMetrics(window_size=30)
        self.performance_metrics.start_monitoring(interval=0.5)  # Background monitoring
        
        self.start_time = TimeUtils.now_as_float()
        self.successful_conversions = 0
        self.fps_history = deque(maxlen=3)  # Reduced from 5 to 3 for less memory
        self.verified_transform = False
        self.camera_info_logged = False
        
        # Position tracking and filtering - minimal
        self.last_position = None
        self.last_position_time = 0
        self.position_filter_alpha = 0.8  # Higher alpha for more responsive tracking
        
        # Max allowed position change per second
        self.max_position_change = 4.0  # Increased for more responsive tracking
        
        # Detection tracking
        self.detection_history = {
            'YOLO': {'latest_position': None, 'last_time': 0},
            'HSV': {'latest_position': None, 'last_time': 0}
        }
        
        # Error tracking
        self.error_last_logged = {}
        
        # Timestamps
        self.last_resource_alert_time = 0
        self.last_cache_log_time = 0
        self.last_diag_log_time = 0
        self.transform_not_verified_logged = False
        self.last_detection_time = TimeUtils.now_as_float()
        
        # Transform cache
        self.transform_cache = {}
        self.transform_cache_lifetime = 20.0  # Increased from 10.0 to 20.0 seconds for Pi optimizations
        
        # Bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Status counters
        self.current_fps = 0.0
        self.last_fps_update = 0
        
        # Initialize the ground position filter
        self.position_filter = GroundPositionFilter()
        
        # Enhanced temporal depth tracking
        self.use_temporal_blending = DEPTH_CONFIG.get("temporal_blending", 0.7)
        self.use_dynamic_sampling = DEPTH_CONFIG.get("use_dynamic_sampling", True)
        self.use_neighbor_data = DEPTH_CONFIG.get("use_neighbor_data", True)
        
        # Expanded depth history tracking
        self.depth_sequence_by_region = {}    # Store recent depth values by region
        self.max_sequence_length = 5          # Maximum number of historical depths to store
        self.neighbor_search_radius = 2       # Number of regions to search for neighboring data
        
        # Tracking depth stability
        self.depth_stability_map = {}         # Track stability of depth data by region
        self.last_frame_had_depth = False     # Whether the last frame had valid depth
        self.consecutive_no_depth_frames = 0  # Count frames with no depth data
        
        # Quality settings with balanced approach
        self.ultra_fast_path = DEPTH_CONFIG.get("ultra_fast_path", False)  # Back to False
        self.min_roi_size = DEPTH_CONFIG.get("min_roi_size", 15)  # Restored to 15
        self.quality_preference = DEPTH_CONFIG.get("quality_preference", 0.7)  # Restored to 0.7
        self.historical_fallback_always = True  # Always try historical fallback
        self.min_points_threshold = DEPTH_CONFIG.get("min_points_threshold", 3)  # Restored to 3
        self.aggressive_caching = DEPTH_CONFIG.get("aggressive_caching", False)
        self.skip_roi_expansion = DEPTH_CONFIG.get("skip_roi_expansion", False)
        
        # Enhanced performance settings
        self.process_every_n_frames = 1  # Process every frame by default, adaptive skipping later
        self.parallel_processing = DEPTH_CONFIG.get("parallel_processing", True)
        self.fast_path_optimization = DEPTH_CONFIG.get("fast_path_optimization", True)
        self.adaptive_frame_skip = DEPTH_CONFIG.get("adaptive_frame_skip", True)
        
        # Cache previous depth results for faster processing
        self.last_full_depth_scan_time = 0
        self.full_depth_scan_interval = 0.5  # Seconds between full depth scans
        self.last_movement = (0, 0)  # Last movement vector
        self.detection_locations = {}  # Track previous detection locations
        self.low_movement_threshold = 3.0  # Threshold for low movement detection
        
        # Enable/use threading for depth operations
        self.depth_workers_enabled = self.parallel_processing
        self.depth_result = None   # Storage for async depth result
        self.depth_processing = False  # Flag for active depth processing
        self.depth_queue = deque(maxlen=5)  # Queue for depth processing requests
    
    def _preallocate_messages(self):
        """Pre-allocate message objects to reduce memory allocations."""
        # Create reusable message objects
        self.reusable_point = PointStamped()
        self.reusable_hsv_point = PointStamped()
        self.reusable_yolo_point = PointStamped()
        self.reusable_diag = String()
    
    def _setup_callback_group(self):
        """Set up callback group and QoS profile for subscriptions."""
        # Single reentrant callback group
        self.callback_group = ReentrantCallbackGroup()
        
        # QoS profile with minimal buffer sizes
        
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # Minimal buffering
        )
    
    def _init_camera_parameters(self):
        """Initialize camera and detection parameters."""
        # Camera parameters (will be updated from camera_info)
        self.camera_info = None
        self.depth_array = None
        self.depth_header = None
        
        # Camera intrinsics
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        
        # Depth image resolution (default values)
        self.depth_width = 640
        self.depth_height = 480
        
        # Coordinate scaling factors
        self.x_scale = 2.0  # Default values
        self.y_scale = 2.0
    
    def _setup_tf2(self):
        """Set up tf2 components for coordinate transformations."""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Add common reference frame
        self.reference_frame = COMMON_REFERENCE_FRAME["reference_frame"]
        self.transform_timeout = COMMON_REFERENCE_FRAME["transform_timeout"]
        
        # Schedule a check to verify transform is properly registered
        self.transform_check_timer = self.create_timer(2.0, self._verify_transform)
    
    def _verify_transform(self):
        """Verify transform is registered and cancel verification timer if successful."""
        try:
            if self.tf_buffer.can_transform(
                self.reference_frame,
                "ascamera_color_0",
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            ):
                self.verified_transform = True
                self.get_logger().info("Transform verification successful")
                self.transform_check_timer.cancel()
                return
        except Exception:
            pass
            
        # If transform is not ready, log warning
        if not self.transform_not_verified_logged:
            self.get_logger().warning("Transform not yet available: base_link -> ascamera_color_0")
            self.transform_not_verified_logged = True
    
    def _setup_subscriptions(self):
        """Set up all subscriptions for this node."""
        # Subscribe to camera calibration information
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            TOPICS["input"]["camera_info"],
            self.camera_info_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Subscribe to depth image with optimized QoS
        self.depth_sub = self.create_subscription(
            Image,
            TOPICS["input"]["depth_image"],
            self.depth_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )
        
        # Subscribe to YOLO ball detections
        self.yolo_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["yolo_detection"],
            self.yolo_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )
        
        # Subscribe to HSV ball detections
        self.hsv_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["hsv_detection"],
            self.hsv_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )
    
    def _setup_publishers(self):
        """Set up all publishers for this node."""
        # Separate publishers for YOLO and HSV 3D positions
        self.yolo_3d_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["yolo_3d"],
            10
        )
        
        self.hsv_3d_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["hsv_3d"],
            10
        )
        
        # Combined publisher (for backward compatibility)
        self.position_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["combined"],
            10
        )
        
        # Diagnostics publisher
        self.system_diagnostics_publisher = self.create_publisher(
            String,
            "/basketball/depth_camera/diagnostics",
            10
        )
    
    def log_error(self, error_message, is_warning=False):
        """Simplified error logging with rate limiting."""
        current_time = TimeUtils.now_as_float()
        
        # Simple rate limiting - log each error type at most once every 20 seconds
        if error_message not in self.error_last_logged or current_time - self.error_last_logged[error_message] > 20.0:
            self.error_last_logged[error_message] = current_time
            
            if is_warning:
                self.get_logger().warning(f"DEPTH: {error_message}")
            else:
                self.get_logger().error(f"DEPTH: {error_message}")
            
            # Force flush stdout to reduce log delay
            sys.stdout.flush()
    
    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        self.camera_info = msg
        
        # Cache intrinsics for faster access
        self.fx = float(msg.k[0])  # Focal length x
        self.fy = float(msg.k[4])  # Focal length y
        self.cx = float(msg.k[2])  # Principal point x (optical center)
        self.cy = float(msg.k[5])  # Principal point y (optical center)
        
        # Update image dimensions and scaling factors
        self.depth_width = msg.width
        self.depth_height = msg.height
        
        # Log camera info once (first time received)
        if not self.camera_info_logged:
            self.get_logger().info(f"Camera info received: {self.depth_width}x{self.depth_height}")
            self.get_logger().info(f"Scaling factors: x_scale={self.x_scale}, y_scale={self.y_scale}")
            self.camera_info_logged = True

    def depth_callback(self, msg):
        """Process depth image with efficient frame skipping."""
        try:
            # Adaptive frame skipping based on movement
            if self.adaptive_frame_skip:
                skip = self._determine_frame_skip()
                
                self.frame_counter += 1
                if self.frame_counter % skip != 0:
                    return
            else:
                # Standard frame skipping
                self.frame_counter += 1
                if self.frame_counter % self.process_every_n_frames != 0:
                    return
            
            # Use direct imgmsg_to_cv2 for better performance
            # Avoid unnecessary copying by using 'passthrough' encoding
            self.depth_array = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_header = msg.header
            
        except Exception as e:
            self.log_error(f"Depth processing error: {str(e)}")
    
    def yolo_callback(self, msg):
        """Handle YOLO detections by processing them."""
        self._process_detection(msg, 'YOLO')
    
    def hsv_callback(self, msg):
        """Handle HSV detections by processing them."""
        self._process_detection(msg, 'HSV')
    
    def _determine_frame_skip(self):
        """Adaptively determine frame skip based on movement."""
        # Default to the current setting
        skip = self.process_every_n_frames
        
        # If we have previous detections to compare
        if len(self.detection_locations) > 0:
            # Calculate movement magnitude
            total_movement = 0
            count = 0
            current_time = time.time()
            
            for source, data in self.detection_locations.items():
                if current_time - data['time'] < 1.0:  # Only consider recent detections
                    total_movement += data['movement']
                    count += 1
            
            if count > 0:
                avg_movement = total_movement / count
                
                # Adjust skip rate based on movement
                if avg_movement < self.low_movement_threshold:
                    # Low movement - skip more frames
                    skip = min(4, skip + 1)
                else:
                    # High movement - process more frames
                    skip = max(1, skip - 1)
        
        return skip
    
    def _process_detection(self, msg, source):
        """Ultra-optimized detection processing."""
        # Skip if we don't have depth data yet
        if self.depth_array is None or self.camera_info is None:
            return
            
        # Skip if transform not verified
        if not self.verified_transform:
            return
            
        # Track detection position and movement for adaptive frame skipping
        if self.adaptive_frame_skip:
            current_pos = (msg.point.x, msg.point.y)
            current_time = time.time()
            
            if source in self.detection_locations:
                # Calculate movement magnitude
                prev_pos = self.detection_locations[source]['position']
                dx = current_pos[0] - prev_pos[0]
                dy = current_pos[1] - prev_pos[1]
                movement = (dx**2 + dy**2)**0.5
                
                # Update movement info
                self.detection_locations[source] = {
                    'position': current_pos,
                    'time': current_time,
                    'movement': movement
                }
            else:
                # First detection for this source
                self.detection_locations[source] = {
                    'position': current_pos,
                    'time': current_time,
                    'movement': 0.0
                }
        
        # Check cache first for performance
        if self._check_position_cache(msg, source):
            return
            
        # Process detection
        self._get_3d_position(msg, source)
    
    def _check_position_cache(self, msg, source):
        """Compare in the same coordinate space for proper caching."""
        # Skip cache check if no cached position
        if self.detection_cache[source]['detection_2d'] is None:
            return False
                
        # Get current detection and cached 2D detection (same coordinate space)
        curr_x, curr_y = msg.point.x, msg.point.y
        cached_detection = self.detection_cache[source]['detection_2d']
        cached_x, cached_y = cached_detection.point.x, cached_detection.point.y
        
        # Calculate 2D distance in the SAME coordinate space - use squared distance to avoid sqrt
        dx = curr_x - cached_x
        dy = curr_y - cached_y
        dist_sq = dx*dx + dy*dy
        
        # Cache timing check
        curr_time = time.time()
        cached_time = self.detection_cache[source]['timestamp']
        
        # More modest thresholds
        movement_threshold = 0.3  # Restored to default
        if self.current_cpu_usage > 90.0:
            # Still use aggressive caching when CPU is very high
            movement_threshold = 0.6
        elif self.current_cpu_usage > 85.0:
            # Moderately aggressive caching when CPU is high
            movement_threshold = 0.45
        
        cache_duration = 0.5  # Restored default duration
        # Extend cache duration under high load
        if self.current_cpu_usage > 90.0:
            cache_duration = 1.0  # Reduced from 1.5
        elif self.current_cpu_usage > 85.0:
            cache_duration = 0.8
        
        # Use cache if position is similar and cache is fresh
        if dist_sq < movement_threshold and curr_time - cached_time < cache_duration:
            cached_3d = self.detection_cache[source]['position_3d']
            
            # Reuse message object for this source
            new_msg = self.reusable_yolo_point if source == 'YOLO' else self.reusable_hsv_point
            new_msg.header.frame_id = cached_3d.header.frame_id
            new_msg.header.stamp = self.get_clock().now().to_msg()
            new_msg.point = cached_3d.point
            
            # Publish directly (avoiding extra processing)
            if source == 'YOLO':
                self.yolo_3d_publisher.publish(new_msg)
            else:
                self.hsv_3d_publisher.publish(new_msg)
                
            self.position_publisher.publish(new_msg)
            
            # Update cache hits counter
            self.cache_hits += 1
            self.successful_conversions += 1
            
            return True
        
        # Update total attempts
        self.total_attempts += 1
        
        return False
    
    def _get_region_key(self, x, y):
        """Get a key for a spatial region of the depth image."""
        region_x = x // self.region_grid_size
        region_y = y // self.region_grid_size
        return f"{region_x}_{region_y}"
    
    def _store_depth_history(self, pixel_x, pixel_y, depth, valid_points):
        """Store depth measurement in history for future reference."""
        if not self.use_depth_history or depth <= 0 or valid_points < 3:
            return
            
        region_key = self._get_region_key(pixel_x, pixel_y)
        current_time = time.time()
        
        # Store in history with timestamp
        self.depth_history[region_key] = {
            'depth': depth,
            'timestamp': current_time,
            'valid_points': valid_points,
            'pixel_x': pixel_x,
            'pixel_y': pixel_y
        }
        
        # Also store in the sequence history for temporal blending
        if region_key not in self.depth_sequence_by_region:
            self.depth_sequence_by_region[region_key] = []
            
        # Add to sequence with timestamp
        self.depth_sequence_by_region[region_key].append({
            'depth': depth,
            'timestamp': current_time,
            'valid_points': valid_points
        })
        
        # Trim the sequence to max length
        if len(self.depth_sequence_by_region[region_key]) > self.max_sequence_length:
            self.depth_sequence_by_region[region_key].pop(0)
            
        # Update region stability score
        if region_key not in self.depth_stability_map:
            self.depth_stability_map[region_key] = {
                'success_count': 0,
                'failure_count': 0,
                'stability_score': 0.5  # Default middle value
            }
        
        # Increase success count for this region
        self.depth_stability_map[region_key]['success_count'] += 1
        # Recalculate stability score
        total = (self.depth_stability_map[region_key]['success_count'] + 
                 self.depth_stability_map[region_key]['failure_count'])
        if total > 0:
            self.depth_stability_map[region_key]['stability_score'] = (
                self.depth_stability_map[region_key]['success_count'] / total
            )
        
        # Update region statistics
        if region_key not in self.depth_region_stats:
            self.depth_region_stats[region_key] = {
                'total_attempts': 0,
                'successful': 0,
                'success_rate': 0.0,
                'last_updated': current_time
            }
            
        stats = self.depth_region_stats[region_key]
        stats['total_attempts'] += 1
        stats['successful'] += 1 if valid_points > 0 else 0
        stats['success_rate'] = stats['successful'] / stats['total_attempts']
        stats['last_updated'] = current_time
        
        # Reset global counter since we had success
        self.consecutive_no_depth_frames = 0
        self.last_frame_had_depth = True
    
    def _update_depth_failure(self, pixel_x, pixel_y):
        """Record a depth measurement failure for stability tracking."""
        region_key = self._get_region_key(pixel_x, pixel_y)
        
        # Initialize if not exists
        if region_key not in self.depth_stability_map:
            self.depth_stability_map[region_key] = {
                'success_count': 0,
                'failure_count': 0,
                'stability_score': 0.5  # Default middle value
            }
        
        # Increase failure count for this region
        self.depth_stability_map[region_key]['failure_count'] += 1
        # Recalculate stability score
        total = (self.depth_stability_map[region_key]['success_count'] + 
                 self.depth_stability_map[region_key]['failure_count'])
        if total > 0:
            self.depth_stability_map[region_key]['stability_score'] = (
                self.depth_stability_map[region_key]['success_count'] / total
            )
        
        # Increment consecutive failure counter
        if self.last_frame_had_depth:
            self.consecutive_no_depth_frames = 1
            self.last_frame_had_depth = False
        else:
            self.consecutive_no_depth_frames += 1
    
    def _get_neighbor_regions(self, region_key):
        """Get a list of neighboring region keys."""
        base_x, base_y = map(int, region_key.split('_'))
        neighbors = []
        
        # Generate a list of neighbor regions in a square pattern
        for dx in range(-self.neighbor_search_radius, self.neighbor_search_radius + 1):
            for dy in range(-self.neighbor_search_radius, self.neighbor_search_radius + 1):
                if dx == 0 and dy == 0:  # Skip self
                    continue
                neighbors.append(f"{base_x + dx}_{base_y + dy}")
        
        return neighbors
    
    def _get_historical_depth_with_blending(self, pixel_x, pixel_y):
        """Get depth using temporal blending across multiple frames."""
        if not self.use_depth_history:
            return None, 0
        
        region_key = self._get_region_key(pixel_x, pixel_y)
        current_time = time.time()
        
        # Try current region first
        if region_key in self.depth_sequence_by_region and self.depth_sequence_by_region[region_key]:
            sequence = self.depth_sequence_by_region[region_key]
            # Filter for recent entries only
            recent_entries = [entry for entry in sequence if current_time - entry['timestamp'] < self.depth_history_max_age]
            
            if recent_entries:
                # Sort by timestamp, newest first
                recent_entries.sort(key=lambda x: x['timestamp'], reverse=True)
                
                if len(recent_entries) == 1:
                    # Only one entry, no need to blend
                    return recent_entries[0]['depth'], recent_entries[0]['valid_points']
                else:
                    # Use temporal blending - newer entries have more weight
                    total_weight = 0
                    weighted_depth = 0
                    max_points = 0
                    
                    # Apply weights based on how recent each entry is
                    for i, entry in enumerate(recent_entries):
                        # Weight decays with age (newest = highest weight)
                        weight = pow(0.7, i)  # Exponential decay of weight
                        weighted_depth += entry['depth'] * weight
                        total_weight += weight
                        # Track maximum number of valid points for quality indicator
                        max_points = max(max_points, entry['valid_points'])
                    
                    if total_weight > 0:
                        blended_depth = weighted_depth / total_weight
                        if self.debug_depth:
                            self.get_logger().info(
                                f"DEPTH DEBUG: Temporal blending used {len(recent_entries)} entries for depth {blended_depth:.3f}m"
                            )
                            sys.stdout.flush()
                        return blended_depth, max_points
        
        # If we don't have data for this region, check neighboring regions if enabled
        if self.use_neighbor_data:
            neighbors = self._get_neighbor_regions(region_key)
            neighbor_data = []
            
            for neighbor_key in neighbors:
                if neighbor_key in self.depth_history:
                    entry = self.depth_history[neighbor_key]
                    age = current_time - entry['timestamp']
                    # Only use fresh neighbor data
                    if age < self.depth_history_max_age:
                        neighbor_data.append((entry, age))
            
            if neighbor_data:
                # Sort by age (youngest first)
                neighbor_data.sort(key=lambda x: x[1])
                newest_entry = neighbor_data[0][0]
                if self.debug_depth:
                    self.get_logger().info(
                        f"DEPTH DEBUG: Using neighbor region data: {newest_entry['depth']:.3f}m from region {neighbor_data[0][0]})"
                    )
                    sys.stdout.flush()
                return newest_entry['depth'], newest_entry['valid_points'] // 2  # Reduce quality score for neighbor data
        
        # Fall back to basic historical data for this region
        if region_key in self.depth_history:
            history = self.depth_history[region_key]
            age = current_time - history['timestamp']
            
            # Use historical data only if it's fresh enough
            if age < self.depth_history_max_age:
                if self.debug_depth:
                    self.get_logger().info(
                        f"DEPTH DEBUG: Using historical depth {history['depth']:.3f}m from {age:.1f}s ago for region {region_key}"
                    )
                    sys.stdout.flush()
                return history['depth'], max(1, history['valid_points'] // 2)  # Reduced quality for historical data
        
        return None, 0
    
    def _dynamic_depth_sampling(self, pixel_x, pixel_y, initial_size=10):
        """Dynamic sampling that starts small and expands outward until valid data is found."""
        # Start with a small region
        current_size = initial_size
        max_size = self.max_roi_size
        
        # Early exit - check if pixel has direct valid depth first
        # This is an ultra-fast single-pixel check that can skip the whole algorithm
        direct_depth = self.depth_array[pixel_y, pixel_x]
        if direct_depth > 0:
            scaled_depth = float(direct_depth) * self._scale_factor
            if self._min_valid_depth < scaled_depth < self._max_valid_depth:
                # Found valid depth at exact pixel - return immediately
                if self.debug_depth:
                    self.get_logger().info(
                        f"DEPTH DEBUG: Direct pixel hit with depth {scaled_depth:.3f}m"
                    )
                    sys.stdout.flush()
                
                # Store successful depth measurement for future reference
                self._store_depth_history(pixel_x, pixel_y, scaled_depth, 1)
                return scaled_depth, 1
                
        # FPS boost: Check 4 neighbor pixels before doing ROI sampling
        # This is still very fast but more reliable than a single pixel
        num_neighbors = 0
        sum_depth = 0
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = pixel_y + dy, pixel_x + dx
            if 0 <= ny < self.depth_array.shape[0] and 0 <= nx < self.depth_array.shape[1]:
                neighbor_depth = self.depth_array[ny, nx]
                if neighbor_depth > 0:
                    scaled_neighbor = float(neighbor_depth) * self._scale_factor
                    if self._min_valid_depth < scaled_neighbor < self._max_valid_depth:
                        sum_depth += scaled_neighbor
                        num_neighbors += 1
        
        if num_neighbors >= 2:  # If we have at least 2 valid neighbors
            avg_depth = sum_depth / num_neighbors
            if self.debug_depth:
                self.get_logger().info(
                    f"DEPTH DEBUG: Quick neighbor check found {num_neighbors} points with depth {avg_depth:.3f}m"
                )
                sys.stdout.flush()
                
            # Store successful depth measurement
            self._store_depth_history(pixel_x, pixel_y, avg_depth, num_neighbors)
            return avg_depth, num_neighbors
                
        # Track the best result so far
        best_depth = None
        best_points = 0
        
        # Try progressively larger regions until we find valid data or hit max size
        while current_size <= max_size:
            y_min = max(0, pixel_y - current_size//2)
            y_max = min(self.depth_array.shape[0], y_min + current_size)
            x_min = max(0, pixel_x - current_size//2)
            x_max = min(self.depth_array.shape[1], x_min + current_size)
            
            if self.debug_depth:
                self.get_logger().info(f"DEPTH DEBUG: Sampling with size {current_size}: x=[{x_min}:{x_max}], y=[{y_min}:{y_max}]")
                sys.stdout.flush()
            
            # FPS boost: Use strided sampling for larger ROIs
            # Only sample every other pixel when ROI gets larger than 20x20
            if current_size > 20:
                # Extract ROI and use stride sampling
                stride = 2
                roi = self.depth_array[y_min:y_max:stride, x_min:x_max:stride]
            else:
                # For small ROIs, sample all pixels
                roi = self.depth_array[y_min:y_max, x_min:x_max]
                
            nonzero_count = np.count_nonzero(roi)
            
            if nonzero_count >= 3:  # Found sufficient data
                nonzeros = roi[roi > 0]
                
                # Apply statistical outlier rejection to remove noisy points
                if nonzero_count >= 10:
                    # Get median and standard deviation
                    median = np.median(nonzeros)
                    std = np.std(nonzeros)
                    
                    # Filter outliers (values more than 2 standard deviations from median)
                    filtered_nonzeros = nonzeros[np.abs(nonzeros - median) < 2 * std]
                    
                    # Only use filtered data if we didn't lose too many points
                    if len(filtered_nonzeros) >= max(3, nonzero_count * 0.7):
                        nonzeros = filtered_nonzeros
                        nonzero_count = len(filtered_nonzeros)
                
                # Convert to meters
                depth = float(np.median(nonzeros)) * self._scale_factor
                
                # Keep track of the best result
                if best_points < nonzero_count:
                    best_depth = depth
                    best_points = nonzero_count
                
                # FPS boost: Be less strict about the required number of points
                # For higher FPS, reduce minimum point count from 10 to 5
                if nonzero_count >= 5 and self._min_valid_depth < depth < self._max_valid_depth:
                    if self.debug_depth:
                        self.get_logger().info(
                            f"DEPTH DEBUG: Dynamic sampling found {nonzero_count} points with depth {depth:.3f}m at size {current_size}"
                        )
                        sys.stdout.flush()
                    
                    # Store successful depth measurement for future reference
                    self._store_depth_history(pixel_x, pixel_y, depth, nonzero_count)
                    return depth, nonzero_count
                
            # If this size failed or didn't find enough points, increase by 50%
            # But use a more aggressive growth rate for faster convergence
            current_size = int(current_size * 1.75)  # Increased from 1.5 to 1.75 for faster convergence
        
        # If we tried all sizes and found some points, return the best result
        if best_depth is not None and best_points >= 3 and self._min_valid_depth < best_depth < self._max_valid_depth:
            if self.debug_depth:
                self.get_logger().info(
                    f"DEPTH DEBUG: Dynamic sampling using best result: {best_points} points with depth {best_depth:.3f}m"
                )
                sys.stdout.flush()
            
            # Store successful depth measurement
            self._store_depth_history(pixel_x, pixel_y, best_depth, best_points)
            return best_depth, best_points
        
        # Try searching neighboring regions as a last resort
        if self.debug_depth:
            self.get_logger().info("DEPTH DEBUG: Dynamic sampling failed, trying neighboring regions")
            sys.stdout.flush()
            
        # More comprehensive spiral search pattern for neighbor regions
        # Search in expanding spiral pattern with varying distances
        for search_dist in [20, 40]:
            # Try diagonal and cardinal directions
            for offset in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,-1), (1,-1), (-1,1)]:
                neighbor_x = pixel_x + offset[0] * search_dist
                neighbor_y = pixel_y + offset[1] * search_dist
                
                # Skip if outside bounds
                if not (0 <= neighbor_x < self.depth_array.shape[1] and 0 <= neighbor_y < self.depth_array.shape[0]):
                    continue
                    
                # Try a region around this neighbor
                for neighbor_size in [20, 30]:
                    y_min = max(0, neighbor_y - neighbor_size//2)
                    y_max = min(self.depth_array.shape[0], y_min + neighbor_size)
                    x_min = max(0, neighbor_x - neighbor_size//2)
                    x_max = min(self.depth_array.shape[1], x_min + neighbor_size)
                    
                    roi = self.depth_array[y_min:y_max, x_min:x_max]
                    nonzero_count = np.count_nonzero(roi)
                    
                    if nonzero_count >= 3:
                        nonzeros = roi[roi > 0]
                        depth = float(np.median(nonzeros)) * self._scale_factor
                        
                        if self._min_valid_depth < depth < self._max_valid_depth:
                            if self.debug_depth:
                                self.get_logger().info(
                                    f"DEPTH DEBUG: Found {nonzero_count} points in neighbor region at offset {offset} with depth {depth:.3f}m"
                                )
                                sys.stdout.flush()
                            
                            # Record this neighbor region's success for future reference
                            neighbor_region_key = self._get_region_key(neighbor_x, neighbor_y)
                            self._store_depth_history(neighbor_x, neighbor_y, depth, nonzero_count)
                            
                            return depth, nonzero_count
        
        # Use historical data as a last resort
        historical_depth = self._get_historical_depth_anywhere(pixel_x, pixel_y)
        if historical_depth is not None:
            return historical_depth
        
        return None, 0
    
    def _ultra_fast_depth(self, pixel_x, pixel_y):
        """Ultra-minimal depth processing with ROI - balanced for quality and performance."""
        try:
            # Fast path optimization - check if we've recently processed this region
            if self.fast_path_optimization:
                region_key = self._get_region_key(pixel_x, pixel_y)
                current_time = time.time()
                
                # If we have recent depth data for this region, use it directly
                if region_key in self.depth_history:
                    entry = self.depth_history[region_key]
                    age = current_time - entry['timestamp']
                    
                    # Only use very fresh data for fast path
                    if age < 0.2 and entry['valid_points'] >= 5:
                        return entry['depth'], entry['valid_points']
            
            # Debug - log input pixel coordinates if debug_depth enabled
            if self.debug_depth:
                self.get_logger().info(f"DEPTH DEBUG: Processing pixel ({pixel_x}, {pixel_y})")
                sys.stdout.flush()  # Force flush
            
            # Use balanced ROI size - not too small, not too large
            # Minimum default size balances quality and performance
            base_roi_size = max(self.min_roi_size, 15) 
            
            # FPS vs Quality - adjust size based on preference setting
            quality_adjusted_size = int(base_roi_size + (10 * self.quality_preference))  # Restored multiplier to 10
            
            # Adjust for CPU usage
            if self.current_cpu_usage > 90.0:
                roi_size = max(12, int(quality_adjusted_size * 0.6))  # Less aggressive
            elif self.current_cpu_usage > 80.0:
                roi_size = max(15, int(quality_adjusted_size * 0.75))
            else:
                roi_size = quality_adjusted_size
            
            # NEW: Fast path for depth extraction
            # First try fast methods: direct pixel and small radius check
            # This yields good quality with much better performance
            
            # Check direct pixel - if it works, this is the fastest path
            direct_pixel = self.depth_array[pixel_y, pixel_x]
            if direct_pixel > 0:
                scaled_depth = float(direct_pixel) * self._scale_factor
                if self._min_valid_depth < scaled_depth < self._max_valid_depth:
                    # Store this for future reference, but with reduced point count
                    self._store_depth_history(pixel_x, pixel_y, scaled_depth, 3)
                    return scaled_depth, 3
            
            # Try a small 3x3 window - very fast but yields good quality
            valid_depths = []
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    y, x = pixel_y + dy, pixel_x + dx
                    if 0 <= y < self.depth_array.shape[0] and 0 <= x < self.depth_array.shape[1]:
                        d = self.depth_array[y, x]
                        if d > 0:
                            scaled = d * self._scale_factor
                            if self._min_valid_depth < scaled < self._max_valid_depth:
                                valid_depths.append(scaled)
            
            # If we found enough valid depths in the 3x3 window, use them
            if len(valid_depths) >= 3:  # Restored threshold to 3
                depth = np.median(valid_depths)
                self._store_depth_history(pixel_x, pixel_y, depth, len(valid_depths))
                return depth, len(valid_depths)
            
            # Fast detection failed, now try with proper ROI
            # Calculate region bounds (standard approach)
            half_size = roi_size // 2
            y_min = max(0, pixel_y - half_size)
            y_max = min(self.depth_array.shape[0], y_min + roi_size)
            x_min = max(0, pixel_x - half_size)
            x_max = min(self.depth_array.shape[1], x_min + roi_size)
            
            if self.debug_depth:
                self.get_logger().info(f"DEPTH DEBUG: ROI bounds: x=[{x_min}:{x_max}], y=[{y_min}:{y_max}], size={roi_size}")
                sys.stdout.flush()  # Force flush
            
            # Extract ROI - using view not copy for performance
            roi = self.depth_array[y_min:y_max, x_min:x_max]
            
            # OPTIMIZATION: Apply stride sampling only for large ROIs to reduce computation
            if roi.size > 400:  # Only for larger ROIs
                # Use stride of 2 to sample 1/4 of the pixels (every other row and column)
                roi_sampled = roi[::2, ::2]
                nonzeros = roi_sampled[roi_sampled > 0]
                sample_factor = 4  # We're only sampling 1/4 of pixels
            else:
                # For smaller ROIs, use all pixels for better quality
                nonzeros = roi[roi > 0]
                sample_factor = 1
                
            nonzero_count = len(nonzeros) * sample_factor  # Scale count by sample factor
            
            # Debug ROI stats
            if self.debug_depth:
                min_val = np.min(roi) if roi.size > 0 and len(nonzeros) > 0 else 0
                max_val = np.max(roi) if len(nonzeros) > 0 else 0
                self.get_logger().info(
                    f"DEPTH DEBUG: ROI stats: nonzero={nonzero_count}/{roi.size} "
                    f"({nonzero_count/roi.size*100:.1f}%), range=[{min_val*self._scale_factor:.3f}m:{max_val*self._scale_factor:.3f}m]"
                )
                self.get_logger().info(f"DEPTH DEBUG: Found {nonzero_count} nonzero points in ROI")
                sys.stdout.flush()  # Force flush
            
            # If we have enough valid points, use median
            if len(nonzeros) >= 3:  # Restored minimum threshold to 3
                # Convert to meters
                depth = float(np.median(nonzeros)) * self._scale_factor
                
                # Validate range
                if self._min_valid_depth < depth < self._max_valid_depth:
                    if self.debug_depth:
                        self.get_logger().info(f"DEPTH DEBUG: Valid depth found: {depth:.3f}m from {nonzero_count} points")
                        sys.stdout.flush()  # Force flush
                    
                    # Store for future use
                    self._store_depth_history(pixel_x, pixel_y, depth, nonzero_count)
                    return depth, nonzero_count
            
            # Record failure before trying fallbacks
            self._update_depth_failure(pixel_x, pixel_y)
            
            # Always use historical data as fallback for consistency
            if self.historical_fallback_always:
                historical_depth, historical_points = self._get_historical_depth_with_blending(pixel_x, pixel_y)
                if historical_depth is not None:
                    return historical_depth, historical_points
                
                # Wider search as last resort
                historical_depth, historical_points = self._get_historical_depth_anywhere(pixel_x, pixel_y)
                if historical_depth is not None:
                    return historical_depth, historical_points
            
            # Last resort - default value
            if self.debug_depth:
                self.get_logger().info("DEPTH DEBUG: No valid depth found, returning default (1.2m)")
                sys.stdout.flush()  # Force flush
                
            return 1.2, 0  # Default depth = 1.2m
                
        except Exception as e:
            if self.debug_depth:
                self.get_logger().error(f"DEPTH DEBUG: Exception in depth processing: {str(e)}")
                sys.stdout.flush()  # Force flush
            return 1.2, 0  # Default on error
    
    def _get_3d_position(self, msg, source):
        """Convert a 2D ball detection to a 3D position using depth data."""
        try:
            # Get 2D coordinates from detection
            orig_x = float(msg.point.x)
            orig_y = float(msg.point.y)
            
            # Debug log detection coordinates 
            if self.debug_depth:
                self.get_logger().info(f"DEPTH DEBUG: Processing {source} detection at ({orig_x:.2f}, {orig_y:.2f})")
                sys.stdout.flush()  # Force flush
            
            # Scale coordinates to depth image space
            pixel_x = int(round(orig_x * self.x_scale))
            pixel_y = int(round(orig_y * self.y_scale))
            
            # Constrain to valid image bounds with margin
            depth_height, depth_width = self.depth_array.shape
            margin = 10
            
            pixel_x = max(margin, min(pixel_x, depth_width - margin - 1))
            pixel_y = max(margin, min(pixel_y, depth_height - margin - 1))
            
            if self.debug_depth:
                self.get_logger().info(f"DEPTH DEBUG: Mapped to depth pixel coordinates: ({pixel_x}, {pixel_y})")
                self.get_logger().info(f"DEPTH DEBUG: Depth image dimensions: {depth_width}x{depth_height}")
                sys.stdout.flush()  # Force flush
            
            # Get depth using fast estimation
            median_depth, valid_points = self._ultra_fast_depth(pixel_x, pixel_y)
            
            # Record detection quality metrics
            detection_quality = "good" if valid_points >= 8 else "fair" if valid_points >= 3 else "poor"
            
            # Convert to 3D using the pinhole camera model
            x = float((pixel_x - self.cx) * median_depth / self.fx)
            y = float((pixel_y - self.cy) * median_depth / self.fy)
            z = float(median_depth)
            
            # Create the 3D position message in camera frame
            camera_position_msg = PointStamped()
            camera_position_msg.header.stamp = self.get_clock().now().to_msg()
            camera_position_msg.header.frame_id = "ascamera_color_0"
            camera_position_msg.point.x = x
            camera_position_msg.point.y = y
            camera_position_msg.point.z = z
            
            # Transform position to common reference frame
            transformed_msg = self._fast_transform(camera_position_msg)
            if transformed_msg is not None:
                # Apply position filtering using shared ground position filter
                position = (transformed_msg.point.x, transformed_msg.point.y, transformed_msg.point.z)
                filtered_position = self._filter_position(position)
                
                # Update the message with filtered position
                filtered_msg = PointStamped()
                filtered_msg.header = transformed_msg.header
                filtered_msg.point.x = filtered_position[0]
                filtered_msg.point.y = filtered_position[1]
                filtered_msg.point.z = filtered_position[2]
                
                # Publish to source-specific topic
                if source == "YOLO":
                    self.yolo_3d_publisher.publish(filtered_msg)
                else:  # HSV
                    self.hsv_3d_publisher.publish(filtered_msg)
                
                # Also publish to combined topic
                self.position_publisher.publish(filtered_msg)
                
                # Update cache - store both 2D and 3D positions
                self.detection_cache[source]['detection_2d'] = msg         # Original 2D detection
                self.detection_cache[source]['position_3d'] = filtered_msg  # Processed 3D result
                self.detection_cache[source]['timestamp'] = time.time()
                
                # Count successful conversion
                self.successful_conversions += 1
                
                # Log position every 10 successful conversions (increased frequency from 20)
                if self.successful_conversions % 10 == 0:
                    self._update_fps()
                    actual_cpu = psutil.cpu_percent(interval=0.1)  # Quick CPU check with minimal delay
                    self.get_logger().info(
                        f"3D position ({source}): "
                        f"({filtered_position[0]:.2f}, {filtered_position[1]:.2f}, {filtered_position[2]:.2f})m | "
                        f"FPS: {self.current_fps:.1f} | "
                        f"Quality: {detection_quality} ({valid_points} points) | "
                        f"CPU: {actual_cpu:.1f}%"
                    )
                    sys.stdout.flush()  # Force flush
                
                return True
            else:
                return False
                
        except Exception as e:
            self.log_error(f"Error in 3D conversion: {str(e)}")
            return False
    
    def _fast_transform(self, point_stamped):
        """Optimized transform with aggressive caching."""
        # Unique key for this transform
        frame_key = f"{self.reference_frame}_{point_stamped.header.frame_id}"
        curr_time = time.time()
        
        # Check cache first
        if frame_key in self.transform_cache:
            cached_time, cached_transform = self.transform_cache[frame_key]
            
            # Use cache if fresh (10 second validity)
            if curr_time - cached_time < self.transform_cache_lifetime:
                try:
                    transformed = tf2_geometry_msgs.do_transform_point(point_stamped, cached_transform)
                    return transformed
                except Exception:
                    # If transform fails, remove from cache and try new lookup
                    del self.transform_cache[frame_key]
        
        # Get new transform
        try:
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                point_stamped.header.frame_id,
                rclpy.time.Time())
            
            # Cache it
            self.transform_cache[frame_key] = (curr_time, transform)
            
            # Apply transform
            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed
        except Exception as e:
            if self.debug_mode:
                self.get_logger().error(f"Transform lookup error: {str(e)}")
            return None
    
    def _update_fps(self):
        """Update FPS calculation."""
        curr_time = time.time()
        
        # Only update every second
        if curr_time - self.last_fps_update > 1.0:
            elapsed = curr_time - self.start_time
            
            if elapsed > 0:
                self.current_fps = self.successful_conversions / elapsed
                self.fps_history.append(self.current_fps)
                
            self.last_fps_update = curr_time
    
    def _adjust_performance(self):
        """Adaptive performance adjustment based on CPU usage and FPS."""
        # Update FPS
        self._update_fps()
        
        # Get current settings
        old_skip = self.process_every_n_frames
        cpu = self.current_cpu_usage
        
        # More aggressive adaptive throttling based on system load
        if cpu > 95.0:
            # Critical CPU usage - ultra-aggressive throttling
            target_skip = min(8, self.process_every_n_frames + 2)
            self.roi_size = 10  # Ultra-small ROI
            self.transform_cache_lifetime = 60.0  # Very long cache lifetime
        elif cpu > 90.0:
            # Critical CPU usage - aggressive throttling
            target_skip = min(6, self.process_every_n_frames + 1)
            self.roi_size = 12  # Even smaller ROI size
            self.transform_cache_lifetime = 30.0  # Extended cache lifetime
        elif cpu > 85.0:
            # Very high CPU usage - strong throttling
            target_skip = min(5, self.process_every_n_frames + 1)
            self.roi_size = 15  # Reduce ROI size even further
            self.transform_cache_lifetime = 20.0  # Extend cache lifetime
        elif cpu > 75.0:
            # High CPU usage - moderate throttling
            target_skip = min(4, self.process_every_n_frames)
            self.roi_size = 20
            self.transform_cache_lifetime = 15.0
        elif cpu > 60.0:
            # Moderate CPU usage - light throttling
            target_skip = min(3, max(2, self.process_every_n_frames))
            self.roi_size = 25
            self.transform_cache_lifetime = 10.0
        elif cpu < 40.0 and self.current_fps < 5.0:
            # Low CPU - can process more frames
            target_skip = max(1, self.process_every_n_frames - 1)
            self.roi_size = 30
            self.transform_cache_lifetime = 5.0
        else:
            # Maintain current settings
            target_skip = self.process_every_n_frames
        
        # Only change if needed
        if target_skip != self.process_every_n_frames:
            self.process_every_n_frames = target_skip
            # Only log significant changes to reduce logging overhead
            if abs(old_skip - target_skip) > 1:
                self.get_logger().info(
                    f"Adjusted processing: 1 in {self.process_every_n_frames} frames "
                    f"(CPU: {cpu:.1f}%, FPS: {self.current_fps:.1f}, ROI: {self.roi_size})"
                )
                sys.stdout.flush()  # Force flush

    def publish_system_diagnostics(self):
        """Publish comprehensive system diagnostics with detection quality metrics."""
        # Only run at specified interval - REDUCED from 5s to 2s for more frequent updates
        current_time = time.time()
        if current_time - self.last_diag_log_time < 2.0:  
            return
            
        self.last_diag_log_time = current_time
        
        # Update FPS with accurate measurement
        self._update_fps()
        
        # Get accurate CPU usage (not 0%)
        actual_cpu = psutil.cpu_percent(interval=0.05)  # Reduced from 0.1s to 0.05s
        self.current_cpu_usage = actual_cpu  # Update the stored value
        
        # Calculate metrics (only if we have frames processed)
        if self.fps_history:
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            # Calculate frame rate percentage and cache hit rate
            frame_rate_pct = 100.0 / self.process_every_n_frames
            cache_hit_rate = (self.cache_hits / max(1, self.total_attempts)) * 100.0 if self.total_attempts > 0 else 0.0
            
            # Calculate reliability metrics from detector cache
            detection_age_yolo = 0
            detection_age_hsv = 0
            if self.detection_cache['YOLO']['timestamp'] > 0:
                detection_age_yolo = current_time - self.detection_cache['YOLO']['timestamp']
            if self.detection_cache['HSV']['timestamp'] > 0:
                detection_age_hsv = current_time - self.detection_cache['HSV']['timestamp']
                
            # Reliability score (0-100)
            reliability = 100.0
            if detection_age_yolo > 2.0 or detection_age_hsv > 2.0:
                reliability = 75.0
            if detection_age_yolo > 5.0 or detection_age_hsv > 5.0:
                reliability = 50.0
            
            # Log comprehensive status with immediate flush
            self.get_logger().info(
                f"Depth camera: {self.current_fps:.1f} FPS (avg: {avg_fps:.1f}), "
                f"CPU: {actual_cpu:.1f}%, "
                f"RAM: {psutil.virtual_memory().percent:.1f}%, "
                f"Reliability: {reliability:.1f}%, "
                f"Frames: 1:{self.process_every_n_frames}, "
                f"Cache: {cache_hit_rate:.1f}%"
            )
            sys.stdout.flush()  # Force flush
        
            # Publish detailed diagnostics (reusing message object)
            diag_msg = self.reusable_diag
            diag_data = {
                "fps": self.current_fps,
                "avg_fps": avg_fps,
                "cpu": actual_cpu,
                "ram": psutil.virtual_memory().percent,
                "frame_skip": self.process_every_n_frames,
                "frame_rate_pct": frame_rate_pct, 
                "cache_hit_rate": cache_hit_rate,
                "reliability": reliability,
                "detection_age_yolo": round(detection_age_yolo, 2),
                "detection_age_hsv": round(detection_age_hsv, 2),
                "timestamp": current_time
            }
            diag_msg.data = str(diag_data)
            self.system_diagnostics_publisher.publish(diag_msg)
    
    def _handle_resource_alert(self, resource_type, value):
        """Fix CPU usage reporting."""
        if resource_type == 'cpu':
            # Ensure we get non-zero values
            try:
                cpu_value = float(value)
                self.current_cpu_usage = max(0.1, cpu_value)
                
                # Log significant CPU changes
                current_time = time.time()
                if cpu_value > 90.0 and current_time - self.last_cpu_log > 30.0:
                    self.get_logger().info(f"High CPU usage: {cpu_value:.1f}%")
                    self.last_cpu_log = current_time
            except (ValueError, TypeError):
                # Default value if conversion fails
                self.current_cpu_usage = 50.0
    
    def destroy_node(self):
        """Clean shutdown of the node."""
        # Clear any large stored data
        self.depth_array = None
        self.camera_info = None
        
        # Stop resource monitor
        if hasattr(self, 'resource_monitor') and self.resource_monitor:
            try:
                self.resource_monitor.stop()
            except:
                pass
        
        super().destroy_node()

    def _filter_position(self, position):
        """
        Apply position filtering using the shared GroundPositionFilter class.
        This ensures consistent ground movement tracking between both nodes.
        
        Args:
            position: (x, y, z) position tuple/list
            
        Returns:
            Filtered position as (x, y, z) tuple
        """
        # Use the shared ground position filter
        current_time = time.time()
        filtered_position = self.position_filter.update(position, current_time)
        return filtered_position

    def _get_adaptive_roi_size(self, pixel_x, pixel_y):
        """Calculate appropriate ROI size based on depth reliability in this region."""
        region_key = self._get_region_key(pixel_x, pixel_y)
        
        # Default ROI size based on current settings
        base_roi_size = self.roi_size
        
        # FPS optimization: use even smaller ROIs under high load
        if self.current_cpu_usage > 90.0 and base_roi_size > 8:
            base_roi_size = 8  # Ultra-small ROI for high CPU

        # Check if we have statistics for this region
        if region_key in self.depth_region_stats:
            stats = self.depth_region_stats[region_key]
            
            # Only adapt if we have enough data
            if stats['total_attempts'] >= 3:
                success_rate = stats['success_rate']
                
                # Adjust ROI size inversely to success rate
                if success_rate < 0.2:  # Very unreliable
                    return min(self.max_roi_size, int(base_roi_size * 2.5))
                elif success_rate < 0.5:  # Moderately unreliable
                    return min(self.max_roi_size, int(base_roi_size * 1.75))
                elif success_rate < 0.7:  # Somewhat reliable
                    return min(self.max_roi_size, int(base_roi_size * 1.25))
                
        return base_roi_size  # Default size

    def _update_depth_stats(self, pixel_x, pixel_y, success, nonzero_count):
        """Update statistics for depth measurements in this region."""
        region_key = self._get_region_key(pixel_x, pixel_y)
        current_time = time.time()
        
        # Initialize stats for this region if needed
        if region_key not in self.depth_region_stats:
            self.depth_region_stats[region_key] = {
                'total_attempts': 0,
                'successful': 0,
                'success_rate': 0.0,
                'last_updated': current_time
            }
            
        # Update stats
        stats = self.depth_region_stats[region_key]
        stats['total_attempts'] += 1
        if success and nonzero_count > 0:
            stats['successful'] += 1
        stats['success_rate'] = stats['successful'] / stats['total_attempts']
        stats['last_updated'] = current_time

    def _get_historical_depth_anywhere(self, pixel_x, pixel_y):
        """Look for ANY valid historical depth data in the whole scene."""
        current_time = time.time()
        
        # First check direct region
        region_key = self._get_region_key(pixel_x, pixel_y)
        if region_key in self.depth_history:
            entry = self.depth_history[region_key]
            age = current_time - entry['timestamp']
            if age < self.depth_history_max_age:
                if self.debug_depth:
                    self.get_logger().info(
                        f"DEPTH DEBUG: Using direct region history: {entry['depth']:.3f}m from {age:.1f}s ago"
                    )
                return entry['depth'], entry['valid_points']
        
        # Find ANY history entry, starting with newest
        if self.depth_history:
            candidates = []
            for key, entry in self.depth_history.items():
                age = current_time - entry['timestamp']
                if age < self.depth_history_max_age:
                    candidates.append((entry, age))
            
            if candidates:
                # Sort by age (newest first)
                candidates.sort(key=lambda x: x[1])
                entry = candidates[0][0]
                if self.debug_depth:
                    self.get_logger().info(
                        f"DEPTH DEBUG: Using ANY available depth history: {entry['depth']:.3f}m (age: {candidates[0][1]:.1f}s)"
                    )
                    sys.stdout.flush()
                return entry['depth'], max(1, entry['valid_points'] // 2)  # Reduce quality score
        
        return None, 0


def main(args=None):
    """Main function to initialize and run the 3D position estimator node."""
    rclpy.init(args=args)
    
    # Set Raspberry Pi environment variable
    os.environ['RASPBERRY_PI'] = '1'
    
    # Enable depth debugging if requested via command line
    if '--debug-depth' in (args or sys.argv):
        DEPTH_CONFIG['debug_depth'] = True
        print("Depth debugging enabled via command line")
    
    # Create and initialize the node
    node = OptimizedPositionEstimator()
    
    # Use a SingleThreadedExecutor instead of MultiThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    
    print("=================================================")
    print("Ultra-Optimized Basketball Tracking")
    print("=================================================")
    
    try:
        
        sys.stdout.flush()  # Force flush logs
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user.")
    except Exception as e:
        node.get_logger().error(f"Error: {str(e)}")
    finally:
        # Clean up
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()