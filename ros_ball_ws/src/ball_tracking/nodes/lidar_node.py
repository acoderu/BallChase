#!/usr/bin/env python3

"""
Tennis Ball Tracking Robot - LIDAR Detection Node
=================================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving tennis ball.
The system uses multiple sensing modalities for robust detection:
- YOLO neural network detection (subscribes to '/tennis_ball/yolo/position')
- HSV color-based detection (subscribes to '/tennis_ball/hsv/position')
- LIDAR for depth sensing (this node)
- Depth camera for additional depth information

Data Pipeline:
-------------
1. Camera images are processed by:
   - YOLO detection node (yolo_ball_node.py) publishing to '/tennis_ball/yolo/position'
   - HSV color detector (hsv_ball_node.py) publishing to '/tennis_ball/hsv/position'

2. This LIDAR node:
   - Subscribes to raw LIDAR scans from '/scan_raw'
   - Subscribes to both YOLO and HSV detection results
   - Correlates 2D camera detections with 3D LIDAR point data
   - Publishes 3D positions to '/tennis_ball/lidar/position'

3. Data is then passed to:
   - Sensor fusion node to combine all detection methods
   - State management node for decision-making
   - PID controller for motor control

This Node's Purpose:
------------------
This LIDAR node adds depth information to the tennis ball tracking by:
- Processing 2D laser scan data to find circular patterns matching a tennis ball
- Confirming these patterns when triggered by camera-based detections
- Publishing the ball's 3D position in the robot's coordinate frame
- Providing visualization markers for debugging in RViz
"""


import sys
import os
# Add the parent directory of 'config' to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import rclpy
from rclpy.node import Node
import time
import json
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import TransformStamped  
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster
import psutil  # Move psutil import here with other imports
import os
import threading
from collections import deque  # Add import for deque
from utilities.resource_monitor import ResourceMonitor
from utilities.time_utils import TimeUtils  # Add TimeUtils import
from config.config_loader import ConfigLoader  # Import ConfigLoader
from scipy.linalg import block_diag



# Load configuration from file
config_loader = ConfigLoader()
lidar_config = config_loader.load_yaml('lidar_config.yaml')



# Tennis ball configuration from config file
TENNIS_BALL_CONFIG = lidar_config.get('tennis_ball', {
    "radius": 0.033,         # Tennis ball radius in meters
    "height": -0.20,         # Expected height of ball center relative to LIDAR
    "max_distance": 0.17,    # Increased from 0.15 to 0.17
    "min_points": 6,         # Further reduce from 7 to 6
    "quality_threshold": {
        "low": 0.32,         # Further reduce from 0.35 to 0.32
        "medium": 0.45,      # Further reduce from 0.50 to 0.45
        "high": 0.65         # Further reduce from 0.70 to 0.65
    },
    "detection_samples": 40
})

# Topic configuration from config file
TOPICS = lidar_config.get('topics', {
    "input": {
        "lidar_scan": "/scan_raw",
        "yolo_detection": "/tennis_ball/yolo/position",
        "hsv_detection": "/tennis_ball/hsv/position"
    },
    "output": {
        "ball_position": "/tennis_ball/lidar/position",
        "visualization": "/tennis_ball/lidar/visualization",
        "diagnostics": "/tennis_ball/lidar/diagnostics"
    }
})

# Get default queue size from config
DEFAULT_QUEUE_SIZE = lidar_config.get('topics', {}).get('queue_size', 10)

# Get visualization settings
VIZ_CONFIG = lidar_config.get('visualization', {
    'marker_lifetime': 1.0,
    'text_height_offset': 0.1,
    'text_size': 0.05,
    'colors': {
        'yolo': {'r': 0.0, 'g': 1.0, 'b': 0.3, 'base_alpha': 0.5},
        'hsv': {'r': 1.0, 'g': 0.6, 'b': 0.0, 'base_alpha': 0.5},
        'text': {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0}
    }
})

# Get diagnostic settings
DIAG_CONFIG = lidar_config.get('diagnostics', {
    'publish_interval': 3.0,
    'debug_level': 1,
    'log_scan_interval': 20,
    'max_detection_times': 100,
    'error_history_size': 10  # Keep track of last 10 errors
})


class TennisBallLidarDetector(Node):
    """
    A ROS2 node to detect tennis balls using a 2D laser scanner.
    
    This node correlates LIDAR data with camera detections to provide 3D position 
    information for detected tennis balls, which is essential for accurate tracking
    and following behavior in the robot.
    
    Subscribed Topics:
    - LaserScan from 2D LIDAR ({TOPICS["input"]["lidar_scan"]})
    - PointStamped from YOLO detection ({TOPICS["input"]["yolo_detection"]})
    - PointStamped from HSV detection ({TOPICS["input"]["hsv_detection"]})
    
    Published Topics:
    - PointStamped with 3D position ({TOPICS["output"]["ball_position"]})
    - MarkerArray for visualization ({TOPICS["output"]["visualization"]})
    - String with diagnostic information ({TOPICS["output"]["diagnostics"]})
    """
    
    def __init__(self):
        """Initialize the tennis ball LIDAR detector node."""
        super().__init__('tennis_ball_lidar_detector')
        
        # Initialize state tracking (replaces _init_performance_tracking)
        # MOVE THIS UP before resource monitor initialization
        self._init_state_tracking()
        
        # Initialize detection parameters early
        self.detection_samples = TENNIS_BALL_CONFIG["detection_samples"]
        
        # Add resource monitoring for Raspberry Pi 5
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=20.0,  # Less frequent updates to reduce overhead
            enable_temperature=True
        )
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.resource_monitor.start()
        
        # Use PyTorch/NumPy optimized for Raspberry Pi if available
        try:
            # Try to set number of threads for better Pi 5 performance
            import torch
            if hasattr(torch, 'set_num_threads'):
                # Leave one core free for system processes
                torch.set_num_threads(3)
                self.get_logger().info(f"Set PyTorch to use 3 threads on Pi 5")
        except ImportError:
            pass
            
        try:
            # Configure NumPy for better performance
            np.set_printoptions(precision=4, suppress=True)
        except:
            pass
        
        # Load physical parameters of the tennis ball
        self.ball_radius = TENNIS_BALL_CONFIG["radius"]
        self.ball_height = TENNIS_BALL_CONFIG["height"]
        self.max_distance = TENNIS_BALL_CONFIG["max_distance"]
        self.min_points = TENNIS_BALL_CONFIG["min_points"]
        
        # Initialize state variables
        self.latest_scan = None
        self.scan_timestamp = None
        self.scan_frame_id = None
        self.points_array = None
        self.previous_ball_position = None  # For tracking the ball over time
        
        # Set up subscribers
        self._setup_subscribers()
        
        # Set up publishers
        self._setup_publishers()
        
        # Timer for publishing status
        self.status_timer = self.create_timer(
            lidar_config.get('diagnostics', {}).get('publish_interval', 3.0), 
            self.publish_status
        )
        
        # Initialize the TF Broadcaster for publishing coordinate transforms
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_transform_log = 0.0  # Track when we last logged transform info
        
        # Set up a timer for publishing the transform regularly
        # Publishing frequency from config
        self.transform_timer = self.create_timer(
            1.0 / lidar_config.get('transform', {}).get('publish_frequency', 10.0), 
            self.publish_transform
        )
        
        # Load transform parameters from config
        transform_config = lidar_config.get('transform', {})
        self.transform_parent_frame = transform_config.get('parent_frame', 'camera_frame')
        self.transform_child_frame = transform_config.get('child_frame', 'lidar_frame')
        self.transform_translation = transform_config.get('translation', {
            'x': -0.326256, 'y': 0.210052, 'z': 0.504021
        })
        self.transform_rotation = transform_config.get('rotation', {
            'x': -0.091584, 'y': 0.663308, 'z': 0.725666, 'w': 0.158248
        })
        
        # Log that we're publishing the calibrated transform
        self.get_logger().info(f"Publishing calibrated LIDAR-to-camera transform (from calibration)")
        self.get_logger().info(
            f"Transform: [{self.transform_translation['x']}, {self.transform_translation['y']}, "
            f"{self.transform_translation['z']}], Quaternion: [{self.transform_rotation['x']}, "
            f"{self.transform_rotation['y']}, {self.transform_rotation['z']}, {self.transform_rotation['w']}]"
        )

        self.get_logger().info("LIDAR: Tennis ball detector initialized and ready")
        self.get_logger().info(f"LIDAR: Listening for LaserScan on {TOPICS['input']['lidar_scan']}")
        self.get_logger().info(f"LIDAR: Listening for YOLO detections on {TOPICS['input']['yolo_detection']}")
        self.get_logger().info(f"LIDAR: Listening for HSV detections on {TOPICS['input']['hsv_detection']}")
        
        # Configure detection algorithm based on hardware
        self._configure_detection_algorithm()
        
        # Pre-allocate buffers for point cloud processing
        self._points_buffer = None  # Will be allocated based on first point cloud
        self._filtered_buffer = None
        self._cluster_buffer = None

        # Configure logging levels from config
        log_config = lidar_config.get('logging', {
            'console_level': 'info',   # Options: debug, info, warn, error
            'file_level': 'debug',     # Options: debug, info, warn, error
            'log_file': 'lidar_node.log',
            'max_file_size_mb': 10
        })

        # Set logger level programmatically
        self._configure_logging(log_config)

        # Debug data collection (only enabled in debug mode)
        self.debug_mode = DIAG_CONFIG.get('debug_level', 0) > 1
        self.last_point_clouds = deque(maxlen=5) if self.debug_mode else None

        self.position_history = deque(maxlen=10)  # Increase history from 5 to 10 positions

        # Add after other initializations
        self.kalman_filter = KalmanFilter()
        self.filtered_positions = deque(maxlen=10)  # Keep track of filtered positions
        self.use_kalman = True  # Flag to enable/disable Kalman filtering
        
        # Add to parameter declarations
        self.declare_parameter('use_kalman_filter', True)
        self.use_kalman = self.get_parameter('use_kalman_filter').value
        
        if self.use_kalman:
            self.get_logger().info("Using Kalman filter for position smoothing")
        
        # Add diagnostic logging flags (controlled via parameters)
        self.declare_parameter('detailed_logging', False)
        self.declare_parameter('cluster_debug', False)
        self.declare_parameter('kalman_debug', False)
        
        self.detailed_logging = self.get_parameter('detailed_logging').value
        self.cluster_debug = self.get_parameter('cluster_debug').value
        self.kalman_debug = self.get_parameter('kalman_debug').value
        
        # Create dedicated logging files for different analysis aspects
        self.cluster_log_file = None
        self.kalman_log_file = None
        
        if self.cluster_debug:
            try:
                self.cluster_log_file = open("cluster_analysis.log", "a")
                self.cluster_log_file.write("-------- New Session Started --------\n")
                self.cluster_log_file.write(f"Timestamp: {TimeUtils.now_as_float()}\n\n")
            except Exception as e:
                self.get_logger().error(f"Could not create cluster log file: {e}")
        
        if self.kalman_debug:
            try:
                self.kalman_log_file = open("kalman_analysis.log", "a")
                self.kalman_log_file.write("-------- New Session Started --------\n")
                self.kalman_log_file.write(f"Timestamp: {TimeUtils.now_as_float()}\n\n")
            except Exception as e:
                self.get_logger().error(f"Could not create Kalman log file: {e}")

        # Add to __init__ after other initializations
        if not hasattr(self, 'position_history') or self.position_history is None:
            self.position_history = deque(maxlen=10)

        if not hasattr(self, 'previous_ball_position'):
            self.previous_ball_position = None

        if not hasattr(self, 'consecutive_failures'):
            self.consecutive_failures = 0

        if not hasattr(self, 'last_successful_detection_time'):
            self.last_successful_detection_time = 0

        if not hasattr(self, 'predicted_position'):
            self.predicted_position = None

        if not hasattr(self, 'prediction_time'):
            self.prediction_time = 0

        # Add extra debug logging mode
        self.declare_parameter('super_verbose_logging', False)
        self.super_verbose_logging = self.get_parameter('super_verbose_logging').value
        if self.super_verbose_logging:
            self.get_logger().info("SUPER VERBOSE LOGGING MODE ENABLED - Will show detailed debugging information")
    
    def _init_state_tracking(self):
        """Initialize state tracking for all system components."""
        # Performance tracking from _init_performance_tracking will be moved here
        self.start_time = TimeUtils.now_as_float()  # Use TimeUtils instead of time.time()
        self.processed_scans = 0
        self.successful_detections = 0
        
        # Use deque with maxlen for detection times instead of unbounded list
        max_detection_times = DIAG_CONFIG['max_detection_times']
        self.detection_times = deque(maxlen=max_detection_times)
        
        # Detection source statisti
        self.yolo_detections = 0
        self.hsv_detections = 0
        
        # Error tracking (for diagnostics) - use deque with maxlen
        error_history_size = DIAG_CONFIG.get('error_history_size', 10)
        self.errors = deque(maxlen=error_history_size)
        self.warnings = deque(maxlen=error_history_size) # Add warnings collection too
        self.last_error_time = 0
        
        # System health indicators
        self.lidar_health = 1.0  # 0.0 to 1.0 scale
        self.detection_health = 1.0
        self.detection_latency = 0.0
        
        # Add point data history with bounded size
        self.point_history = deque(maxlen=100)  # Keep last 100 sets of points
        
        # Add detection failure tracking
        self.consecutive_failures = 0
        self.last_successful_detection_time = 0
        self.predicted_position = None
    
    def _setup_subscribers(self):
        """Set up all subscribers for this node."""
        # Subscribe to LIDAR scan data
        self.scan_subscription = self.create_subscription(
            LaserScan,
            TOPICS["input"]["lidar_scan"],
            self.scan_callback,
            DEFAULT_QUEUE_SIZE
        )
        
        # Subscribe to YOLO detections
        self.yolo_subscription = self.create_subscription(
            PointStamped,
            TOPICS["input"]["yolo_detection"],
            self.yolo_callback,
            DEFAULT_QUEUE_SIZE
        )
        
        # Subscribe to HSV detections
        self.hsv_subscription = self.create_subscription(
            PointStamped,
            TOPICS["input"]["hsv_detection"],
            self.hsv_callback,
            DEFAULT_QUEUE_SIZE
        )
    
    def _setup_publishers(self):
        """Set up all publishers for this node."""
        # Publisher for 3D tennis ball position
        self.position_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["ball_position"],
            DEFAULT_QUEUE_SIZE
        )
        
        # Publisher for visualization markers (for RViz)
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            TOPICS["output"]["visualization"],
            DEFAULT_QUEUE_SIZE
        )
        
        # Update diagnostics publisher to match the expected topic from the diagnostics node
        self.diagnostics_publisher = self.create_publisher(
            String,
            "/tennis_ball/lidar/diagnostics",
            DEFAULT_QUEUE_SIZE
        )
    
    def yolo_callback(self, msg):
        """
        Handle ball detections from the YOLO neural network.
        
        Args:
            msg (PointStamped): The detected ball position from YOLO
        """
        self.yolo_detections += 1
        self.camera_detection_callback(msg, "YOLO")
    
    def hsv_callback(self, msg):
        """
        Handle ball detections from the HSV color detector.
        
        Args:
            msg (PointStamped): The detected ball position from HSV detector
        """
        self.hsv_detections += 1
        self.camera_detection_callback(msg, "HSV")
    
    def scan_callback(self, msg):
        """
        Process LaserScan messages from the LIDAR.
        
        Extracts point cloud data from 2D laser scan by converting
        polar coordinates to Cartesian coordinates.
        
        Args:
            msg (LaserScan): The laser scan message
        """
        self.start_time = time.time()  # Reset timer for this scan
        
        # Store the scan metadata - preserve original frame_id for proper transformation
        self.latest_scan = msg
        self.scan_timestamp = msg.header.stamp  # Use original timestamp, not current time
        self.scan_frame_id = "lidar_frame"  # Always use our consistent frame ID
        
        try:
            # Extract basic scan parameters
            angle_min = msg.angle_min
            angle_increment = msg.angle_increment
            ranges = np.array(msg.ranges)
            
            # Filter out invalid measurements (inf, NaN)
            valid_indices = np.isfinite(ranges)
            valid_ranges = ranges[valid_indices]
            
            # Also filter out very short ranges that might be robot body reflections
            min_valid_range = 0.05  # Minimum valid range in meters
            valid_indices = valid_indices & (ranges > min_valid_range)
            valid_ranges = ranges[valid_indices]
            
            # Skip processing if no valid ranges
            if len(valid_ranges) == 0:
                self.get_logger().warn("LIDAR: No valid range measurements in scan")
                self.points_array = None
                return
            
            # Convert polar coordinates to Cartesian coordinates
            angles = angle_min + angle_increment * np.arange(len(ranges))[valid_indices]
            
            # Vectorized computation is much faster
            x = valid_ranges * np.cos(angles)  # x = r * cos(θ)
            y = valid_ranges * np.sin(angles)  # y = r * sin(θ)
            
            # Add Z coordinate (assumes LIDAR is parallel to ground)
            z = np.full_like(x, self.ball_height)
            
            # Stack coordinates to create points array [x, y, z]
            self.points_array = np.column_stack((x, y, z))
            
            # Update statistics
            self.processed_scans += 1
            
            # Log scan information (debug level to avoid flooding logs)
            log_scan_interval = DIAG_CONFIG['log_scan_interval']
            if self.processed_scans % log_scan_interval == 0:  # Log every Nth scan
                self.get_logger().debug(
                    f"LIDAR: Processed scan #{self.processed_scans} with "
                    f"{len(self.points_array)} valid points"
                )
            
            # Gradually recover health if we're successfully processing scans
            if hasattr(self, 'lidar_health'):
                self.lidar_health = min(1.0, self.lidar_health + 0.01)
            
            if self.debug_mode and self.points_array is not None:
                # Store a sample of points for debugging
                sample_size = min(100, len(self.points_array))
                indices = np.random.choice(len(self.points_array), sample_size, replace=False)
                self.last_point_clouds.append({
                    "timestamp": self.scan_timestamp.sec + self.scan_timestamp.nanosec/1e9,
                    "points_sample": self.points_array[indices].tolist(),
                    "total_points": len(self.points_array)
                })
            
        except Exception as e:
            error_msg = f"Error processing scan: {str(e)}"
            self.log_error(error_msg)
            self.points_array = None
    
    
    
    def find_tennis_balls(self, trigger_source):
        """
        Search for tennis balls using a more deterministic, grid-based approach.
        Optimized for single-ball detection scenario with cluster merging.
        """
        # Start timing
        operation_start = TimeUtils.now_as_float()
        
        if self.points_array is None or len(self.points_array) == 0:
            self.get_logger().warn("LIDAR: No points available for analysis")
            return []

        # Add at the beginning of find_tennis_balls method
        if self.detailed_logging and trigger_source.endswith("_DEBUG"):
            self.get_logger().info(f"DEBUG MODE: Using min_points={self.min_points}, min_quality={TENNIS_BALL_CONFIG['quality_threshold']['low']}")
        
        self.get_logger().debug(
            f"LIDAR: Analyzing {len(self.points_array)} points triggered by {trigger_source}"
        )
        
        points = self.points_array
        balls_found = []
        
        # Current time for adaptive parameters
        current_time = TimeUtils.now_as_float()
        
        # Dynamically adjust parameters based on consecutive failures
        # If we've had several failures in a row, we should be more lenient
        dynamic_min_points = self.min_points
        dynamic_quality_threshold = TENNIS_BALL_CONFIG["quality_threshold"]["low"]
        
        # If we have consecutive failures, gradually reduce requirements
        if hasattr(self, 'consecutive_failures') and self.consecutive_failures > 2:
            # Reduce minimum points requirement (but not below 7)
            dynamic_min_points = max(7, int(self.min_points * (1.0 - 0.1 * min(self.consecutive_failures, 5))))
            
            # Reduce quality threshold (but not below 0.3)
            dynamic_quality_threshold = max(0.3, dynamic_quality_threshold * (1.0 - 0.1 * min(self.consecutive_failures, 5)))
            
            if self.detailed_logging:
                self.get_logger().info(
                    f"ADAPTIVE: After {self.consecutive_failures} failures, using reduced requirements: "
                    f"min_points={dynamic_min_points} (from {self.min_points}), "
                    f"quality={dynamic_quality_threshold:.2f} (from {TENNIS_BALL_CONFIG['quality_threshold']['low']:.2f})"
                )
        
        # CHANGE: Create deterministic seed points using a grid-based approach
        # This is more systematic than random sampling
        seed_points = []
        
        # Always include previous ball position if available (highest priority)
        if hasattr(self, 'previous_ball_position') and self.previous_ball_position is not None:
            if isinstance(self.previous_ball_position, np.ndarray):
                seed_points.append((self.previous_ball_position, 3.0))  # Add highest priority
            else:
                self.get_logger().warn(f"Previous ball position has wrong type: {type(self.previous_ball_position)}")
        
        # Add predicted position if available (from Kalman filter)
        if hasattr(self, 'predicted_position') and self.predicted_position is not None:
            # Only use prediction if it's recent
            time_since_prediction = current_time - getattr(self, 'prediction_time', 0)
            if time_since_prediction < 1.0:  # Only use predictions less than 1 second old
                if isinstance(self.predicted_position, np.ndarray):
                    seed_points.append((self.predicted_position, 2.5))  # Add high priority
                else:
                    self.get_logger().warn(f"Predicted position has wrong type: {type(self.predicted_position)}")
        
        # CHANGE: Create a grid of seed points in the area where balls are likely to be found
        # Find bounds of point cloud to create a reasonable grid
        if len(points) > 0:
            x_min, y_min = np.min(points[:, 0:2], axis=0) 
            x_max, y_max = np.max(points[:, 0:2], axis=0)
            
            # Create a grid with reasonable spacing (tennis ball diameter)
            grid_spacing = self.ball_radius * 1.25  # Decrease from 1.5x to 1.25x for denser sampling in important areas
            x_grid = np.arange(x_min, x_max, grid_spacing)
            y_grid = np.arange(y_min, y_max, grid_spacing)
            
            # Limit grid size to avoid excessive computation
            max_grid_points = 30  # Increased from 25 to 30 for better coverage
            if len(x_grid) > max_grid_points:
                x_grid = np.linspace(x_min, x_max, max_grid_points)
            if len(y_grid) > max_grid_points:
                y_grid = np.linspace(y_min, y_max, max_grid_points)
            
            # Create grid points - only use points in reasonable range
            for x in x_grid:
                for y in y_grid:
                    # Add distance-based priority - favor points within 1.5 meters
                    distance = np.sqrt(x**2 + y**2)
                    # Only add grid points within reasonable range (increased from 1.5 to 2.0 meters)
                    if distance < 2.0:
                        # Give higher priority to central region and closer points
                        priority = 1.0
                        if distance < 1.0 and abs(y) < 0.7:  # Expand central forward area (increased from 0.5 to 0.7)
                            priority = 2.0  # Higher priority for central area
                        
                        # Prioritize points at the distance where we've seen good detections
                        if 0.05 < distance < 0.3:  # Very close range priority (new)
                            priority += 0.5
                        seed_points.append((np.array([x, y, self.ball_height]), priority))
        
        # Cap the number of seed points to avoid excessive computation
        max_seed_points = int(self.detection_samples)
        if len(seed_points) > max_seed_points:
            # Keep the first point (previous position) and sample from the rest
            rest_indices = np.random.choice(
                len(seed_points) - 1, 
                size=max_seed_points - 1, 
                replace=False
            ) + 1
            seed_points = [seed_points[0]] + [seed_points[i] for i in rest_indices]
        
        # Sort and select seed points by priority
        seed_points.sort(key=lambda sp: sp[1], reverse=True)
        seed_points = [sp[0] for sp in seed_points[:max_seed_points]]
        
        # CHANGE: Dynamically adjust minimum points based on distance
        # Objects closer to the LIDAR should have more points
        for seed_point in seed_points:
            # Find points close to this seed point (Euclidean distance in XY plane)
            distances = np.sqrt(
                (points[:, 0] - seed_point[0])**2 + 
                (points[:, 1] - seed_point[1])**2
            )
            
            # Increased max distance for clustering when having detection failures
            effective_max_distance = self.max_distance
            if hasattr(self, 'consecutive_failures') and self.consecutive_failures > 3:
                # Gradually increase cluster radius for persistent failures
                effective_max_distance = min(0.18, self.max_distance * (1.0 + 0.15 * min(self.consecutive_failures, 5)))
                # Only log once per call to find_tennis_balls
                if self.detailed_logging and not hasattr(self, '_logged_cluster_distance_this_run'):
                    self.get_logger().info(f"ADAPTIVE: Increased cluster distance to {effective_max_distance:.3f}m")
                    self._logged_cluster_distance_this_run = True
            
            cluster_indices = np.where(distances < effective_max_distance)[0]
            cluster = points[cluster_indices]
            
            # CHANGE: Dynamically adjust minimum points threshold based on distance
            # Closer objects should have more points to be valid
            distance_to_lidar = np.sqrt(seed_point[0]**2 + seed_point[1]**2)
            
            # Scale minimum points - closer objects need more points to be valid
            # This helps reject false positives from reflections
            distance_factor = max(0.5, min(1.5, 2.0 - distance_to_lidar))
            adaptive_min_points = max(7, int(dynamic_min_points * distance_factor))
            
            # Skip if cluster is too small for its distance
            if len(cluster) < adaptive_min_points:
                continue
            
            # Calculate the center of the cluster (centroid)
            center = np.mean(cluster, axis=0)
            
            # Use vectorized operations for circle quality check
            center_distances = np.sqrt(
                (cluster[:, 0] - center[0])**2 + 
                (cluster[:, 1] - center[1])**2
            )
            
            # CHANGE: Reject clusters with too much variance in radius
            # This helps filter out non-circular shapes
            radius_std = np.std(center_distances)
            if radius_std > self.ball_radius * 0.5:  # Std dev should be less than 50% of radius
                continue
            
            # A tennis ball should have points at approximately ball_radius distance
            radius_errors = np.abs(center_distances - self.ball_radius)
            avg_error = np.mean(radius_errors)
            
            # Calculate quality metric (1.0 = perfect circle of exactly ball_radius)
            circle_quality = 1.0 - (avg_error / self.ball_radius)
            
            # Calculate point density score (normalized against max observed of ~60 points)
            # More points = more reliable detection
            point_density_score = min(1.0, len(cluster) / 50.0)  # Reduced from 60 to 50
            
            # CHANGE: Add distance factor to quality score
            # Favor balls at reasonable distances (not too far, not too close)
            distance_quality = 1.0 - min(1.0, distance_to_lidar / 2.5)  # Increased from 2.0 to 2.5
            
            # Combine circle quality, point density, and distance into a weighted score
            combined_quality = (
                circle_quality * 0.4 +           # Increased from 30% to 40% circle shape
                point_density_score * 0.5 +      # Decreased from 60% to 50% point density
                distance_quality * 0.1           # Keep 10% distance factor
            )
            
            # Only consider clusters that reasonably match a tennis ball's shape
            if circle_quality > dynamic_quality_threshold:
                balls_found.append((center, len(cluster), combined_quality))
        
        if self.detailed_logging:
            if len(balls_found) == 0:
                self.get_logger().info(f"LIDAR: No potential clusters found for {trigger_source} trigger")
            else:
                self.get_logger().info(
                    f"LIDAR: Found {len(balls_found)} potential clusters for {trigger_source} trigger")
        
        # Sort by combined quality
        sorted_balls = sorted(balls_found, key=lambda x: x[2], reverse=True)
        
        # Keep only the best candidate for single ball scenario
        if len(sorted_balls) > 1:
            # If we have a previous position, use it to filter
            if hasattr(self, 'previous_ball_position') and self.previous_ball_position is not None:
                consistent_balls = []
                # Increase max position change when we have consecutive failures
                max_position_change = 0.7  # Increased base value
                
                # Get time since last detection for adaptive thresholds
                time_since_detection = current_time - getattr(self, 'last_successful_detection_time', 0)
                
                # Adaptive threshold based on time and consecutive failures
                if time_since_detection > 0.3:  # Reduced time threshold
                    max_position_change *= (1.0 + 0.8 * min(time_since_detection, 2.5))  # More aggressive scaling
                    
                if hasattr(self, 'consecutive_failures') and self.consecutive_failures > 2:
                    # Allow larger jumps if we've had several failures
                    max_position_change = min(1.2, max_position_change * (1.0 + 0.15 * self.consecutive_failures))
                
                for ball in sorted_balls:
                    center, points, quality = ball
                    dist_from_previous = np.linalg.norm(center - self.previous_ball_position)
                    
                    # If this cluster is close to previous position, it's likely the same ball
                    if dist_from_previous < max_position_change:
                        consistent_balls.append(ball)
                    elif self.detailed_logging:
                        self.get_logger().info(
                            f"CONSISTENCY: Rejected cluster at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) "
                            f"with {points} points and quality {quality:.2f} - "
                            f"too far from previous position ({dist_from_previous:.2f}m > {max_position_change:.2f}m)")
                
                # If we found any consistent balls, use those instead
                if consistent_balls:
                    if self.detailed_logging:
                        self.get_logger().info(
                            f"CONSISTENCY: Found {len(consistent_balls)} clusters consistent with previous position")
                    sorted_balls = sorted(consistent_balls, key=lambda x: x[2], reverse=True)
        
        # Update our tracked position if we found a good candidate
        if sorted_balls:
            best_ball = sorted_balls[0]
            new_position = best_ball[0]
            
            # Store in position history
            self.position_history.append(new_position)
            
            # Set current position 
            self.previous_ball_position = new_position
            
            # Reset consecutive failures since we found a ball
            if hasattr(self, 'consecutive_failures'):
                self.consecutive_failures = 0
            
            # Update last successful detection time
            if hasattr(self, 'last_successful_detection_time'):
                self.last_successful_detection_time = current_time
            
            # Debug log to show which position was selected
            self.get_logger().debug(
                f"Selected position: ({new_position[0]:.2f}, {new_position[1]:.2f}, {new_position[2]:.2f}), "
                f"Points: {best_ball[1]}, Quality: {best_ball[2]:.2f}"
            )
        else:
            # Increment consecutive failures
            if hasattr(self, 'consecutive_failures'):
                self.consecutive_failures += 1
            
            # Generate prediction from Kalman filter if we have one
            if hasattr(self, 'kalman_filter') and self.kalman_filter.initialized:
                # Only predict if we had a previous successful detection
                if hasattr(self, 'last_successful_detection_time'):
                    time_since_detection = current_time - self.last_successful_detection_time
                    # Only predict for reasonable time periods (avoid extrapolating too far)
                    if time_since_detection < 2.0:
                        self.predicted_position = self.kalman_filter.predict(time_since_detection)
                        self.prediction_time = current_time
                        
                        if self.detailed_logging:
                            self.get_logger().info(
                                f"PREDICTION: No detection for {time_since_detection:.2f}s. "
                                f"Predicted position: ({self.predicted_position[0]:.2f}, "
                                f"{self.predicted_position[1]:.2f}, {self.predicted_position[2]:.2f})"
                            )
        
        # In the find_tennis_balls method:
        if hasattr(self, 'position_history') and len(self.position_history) > 0:
            # Calculate weighted average of recent positions (newer positions have higher weight)
            recent_positions = np.array(list(self.position_history))
            
            # Create weights that favor more recent positions
            weights = np.linspace(0.5, 1.0, len(recent_positions))
            weights = weights / np.sum(weights)  # Normalize
            
            # Calculate weighted average
            average_position = np.zeros(3)
            for i, pos in enumerate(recent_positions):
                average_position += pos * weights[i]
            
            if self.detailed_logging:
                self.get_logger().info(
                    f"HISTORY: Using weighted average position: ({average_position[0]:.2f}, "
                    f"{average_position[1]:.2f}, {average_position[2]:.2f}) from {len(self.position_history)} samples"
                )
            
            # Add strong bias toward positions near the average
            for i, ball in enumerate(sorted_balls):
                center = ball[0]
                distance_to_average = np.linalg.norm(center - average_position)
                
                # Adjust quality score based on consistency with history
                # Positions closer to average get a significant boost
                consistency_bonus = max(0, 0.3 * (1.0 - min(1.0, distance_to_average / 0.3)))
                
                if self.detailed_logging:
                    self.get_logger().info(
                        f"HISTORY: Cluster at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) "
                        f"distance to avg: {distance_to_average:.2f}m, bonus: +{consistency_bonus:.2f}"
                    )
                
                sorted_balls[i] = (center, ball[1], ball[2] + consistency_bonus)
                
            # Re-sort with the consistency bonus
            sorted_balls = sorted(sorted_balls, key=lambda x: x[2], reverse=True)

        # In find_tennis_balls method, add temporal consistency tracking:

        # Near line 694 (after position history weighted average calculation):
        # Add temporal consistency tracking for stable detections
        if self.position_history and len(self.position_history) >= 3:
            # Track frequency of positions in similar locations
            position_clusters = {}
            grid_size = 0.2  # 20cm grid for position binning
            
            # Group recent positions into spatial bins
            for pos in self.position_history:
                # Create a position bin key
                bin_key = (
                    round(pos[0] / grid_size) * grid_size,
                    round(pos[1] / grid_size) * grid_size
                )
                
                if bin_key in position_clusters:
                    position_clusters[bin_key].append(pos)
                else:
                    position_clusters[bin_key] = [pos]
            
            # Find the most consistent cluster (most positions in same area)
            most_consistent = None
            max_count = 0
            for bin_key, positions in position_clusters.items():
                if len(positions) > max_count:
                    max_count = len(positions)
                    most_consistent = bin_key
            
            # If we have a consistent cluster, give bonus to nearby detections
            if most_consistent and max_count >= 3:  # At least 3 detections in similar location
                consistency_center = np.mean(position_clusters[most_consistent], axis=0)
                
                if self.detailed_logging:
                    self.get_logger().info(
                        f"CONSISTENCY: Found stable region at ({consistency_center[0]:.2f}, "
                        f"{consistency_center[1]:.2f}) with {max_count} samples"
                    )
                
                # Add bonus to detections near this consistent region
                for i, ball in enumerate(sorted_balls):
                    center = ball[0]
                    distance_to_consistent = np.linalg.norm(center[0:2] - consistency_center[0:2])
                    
                    # Strong bonus for being in a temporally consistent region
                    if distance_to_consistent < 0.3:  # Within 30cm of consistent detections
                        temporal_bonus = 0.20 * (1.0 - min(1.0, distance_to_consistent / 0.3))  # Increased from 0.15 to 0.20
                        sorted_balls[i] = (center, ball[1], ball[2] + temporal_bonus)
                        
                        if self.detailed_logging:
                            self.get_logger().info(
                                f"CONSISTENCY: Added +{temporal_bonus:.2f} bonus to cluster at "
                                f"({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) for temporal consistency"
                            )
            
            # Re-sort with the consistency bonus applied
            sorted_balls = sorted(sorted_balls, key=lambda x: x[2], reverse=True)

        # In find_tennis_balls method:

        # CHANGE: Merge nearby clusters before quality assessment
        # This helps combine evidence from reflections of the same ball
        merged_balls = []
        processed_indices = set()

        # Make merge threshold adaptive to detection difficulties
        base_merge_threshold = self.ball_radius * 18  # Increased from 15 to 18
        if hasattr(self, 'consecutive_failures'):
            # Increase merge radius with consecutive failures
            failure_factor = 1.0 + min(0.7, 0.15 * self.consecutive_failures)  # Increased from 0.5 to 0.7
            merge_threshold = base_merge_threshold * failure_factor
        else:
            merge_threshold = base_merge_threshold

        # Log the merging threshold for debugging
        if self.detailed_logging and len(balls_found) > 1:
            self.get_logger().info(f"MERGE: Using merge threshold of {merge_threshold:.3f}m")
        
        # Add counter for diagnostics
        clusters_merged_total = 0
        
        # Process each detected ball
        for i, ball in enumerate(balls_found):
            if i in processed_indices:
                continue  # Skip already-merged clusters
                
            center_i, points_i, quality_i = ball
            merged_center = center_i * points_i  # Weighted by point count
            merged_points = points_i
            clusters_merged = 1
            
            # Keep track of which clusters are being merged with this one
            merged_clusters_info = []
            
            # Look for nearby clusters to merge
            for j, other_ball in enumerate(balls_found):
                if j == i or j in processed_indices:
                    continue  # Skip self or already-merged clusters
                    
                center_j, points_j, quality_j = other_ball
                distance = np.linalg.norm(center_i - center_j)
                
                # If clusters are close enough, merge them
                if distance < merge_threshold:
                    # Add logging to diagnose merging issues
                    if self.detailed_logging:
                        self.get_logger().info(
                            f"MERGE ATTEMPT: Merging cluster {i} with {j}, distance={distance:.3f}m < threshold={merge_threshold:.3f}m"
                        )
                    merged_center += center_j * points_j  # Add weighted by point count
                    merged_points += points_j
                    clusters_merged += 1
                    processed_indices.add(j)
                    
                    merged_clusters_info.append({
                        "index": j,
                        "center": (center_j[0], center_j[1], center_j[2]),
                        "points": points_j,
                        "quality": quality_j,
                        "distance": distance
                    })
                    
                    if self.cluster_debug and self.cluster_log_file:
                        self.cluster_log_file.write(f"Merging cluster {i} with {j}: distance={distance:.3f}m\n")
                        self.cluster_log_file.write(f"  Cluster {i}: ({center_i[0]:.3f}, {center_i[1]:.3f}, {center_i[2]:.3f}) pts={points_i} q={quality_i:.2f}\n")
                        self.cluster_log_file.write(f"  Cluster {j}: ({center_j[0]:.3f}, {center_j[1]:.3f}, {center_j[2]:.3f}) pts={points_j} q={quality_j:.2f}\n")
            
            # Finalize the merged cluster
            if clusters_merged > 1:
                # Recalculate center as weighted average 
                merged_center = merged_center / merged_points
                
                # Calculate weighted quality score for the merged cluster
                total_quality = quality_i * points_i
                for info in merged_clusters_info:
                    total_quality += info["quality"] * info["points"]

                # Final quality is weighted average plus a bonus for having multiple confirmations
                merged_quality = (total_quality / merged_points) * (1.0 + 0.1 * (clusters_merged - 1))

                # Cap quality at 1.0
                merged_quality = min(1.0, merged_quality)
                
                # Update cluster merge counter
                clusters_merged_total += clusters_merged
                
                # Detailed logging
                if self.detailed_logging:
                    merge_details = [
                        f"({info['center'][0]:.2f}, {info['center'][1]:.2f}) with {info['points']} pts" 
                        for info in merged_clusters_info
                    ]
                    merge_details_str = ", ".join(merge_details)
                    self.get_logger().info(
                        f"MERGE: Merged {clusters_merged} clusters into ({merged_center[0]:.2f}, {merged_center[1]:.2f}, {merged_center[2]:.2f}) "
                        f"with {merged_points} total points. Merged: {merge_details_str}")
                
                # ... rest of merge recalculation code ...
                
                # Add log about merged clusters
                self.get_logger().info(
                    f"Merged {clusters_merged} clusters into position " 
                    f"({merged_center[0]:.2f}, {merged_center[1]:.2f}, {merged_center[2]:.2f}) "
                    f"with {merged_points} total points, quality: {merged_quality:.2f}"
                )
                
                merged_balls.append((merged_center, merged_points, merged_quality))
            
            elif i not in processed_indices:
                # Keep original ball if it wasn't merged with anything
                merged_balls.append(ball)
                
                if self.detailed_logging:
                    self.get_logger().info(
                        f"MERGE: Cluster at ({center_i[0]:.2f}, {center_i[1]:.2f}, {center_i[2]:.2f}) "
                        f"with {points_i} points remained unmerged (no nearby clusters)")
        
        # Log summary of merging operation
        if self.detailed_logging:
            if clusters_merged_total > 0:
                self.get_logger().info(f"MERGE: Merged {clusters_merged_total} clusters into {len(merged_balls)} final clusters")
            else:
                self.get_logger().info(f"MERGE: No clusters were merged, {len(merged_balls)} clusters remain")
        
        # Replace original balls list with merged version
        sorted_balls = sorted(merged_balls, key=lambda x: x[2], reverse=True)
        
        if hasattr(self, '_logged_cluster_distance_this_run'):
            delattr(self, '_logged_cluster_distance_this_run')
        
        # Add to find_tennis_balls method:

        # After we've had multiple failures, try to extract more information from the point cloud
        if hasattr(self, 'consecutive_failures') and self.consecutive_failures >= 5:
            # Get time since last detection
            time_since_detection = current_time - getattr(self, 'last_successful_detection_time', 0)
            
            # If we're in a prolonged detection drought, try even more aggressive clustering
            if time_since_detection > 1.0:
                self.get_logger().info(f"DEEP RECOVERY: Using aggressive clustering after {time_since_detection:.1f}s without detection")
                
                # Create a denser grid (2x normal density) in the predicted area
                if hasattr(self, 'predicted_position') and self.predicted_position is not None:
                    pred_x, pred_y = self.predicted_position[0:2]
                    search_radius = 0.5  # 50cm search radius around prediction
                    
                    # Create a dense grid in this area
                    dense_x = np.linspace(pred_x - search_radius, pred_x + search_radius, 15)
                    dense_y = np.linspace(pred_y - search_radius, pred_y + search_radius, 15)
                    
                    # Add these points to seed_points with high priority
                    for x in dense_x:
                        for y in dense_y:
                            # Higher priority for points closer to prediction
                            dist = np.sqrt((x - pred_x)**2 + (y - pred_y)**2)
                            if dist < search_radius:
                                priority = 2.5 - (dist / search_radius)
                                seed_points.append((np.array([x, y, self.ball_height]), priority))
        
        # Add after finding the best ball in find_tennis_balls:

        # If we have a detection that's far from history, apply a quality penalty
        if sorted_balls and len(self.position_history) >= 3:
            best_ball = sorted_balls[0]
            center, points, quality = best_ball
            
            # Calculate average of recent positions
            recent_positions = np.array(list(self.position_history)[-3:])  # Last 3 positions
            avg_position = np.mean(recent_positions, axis=0)
            
            # Check if this position is a major outlier
            distance_from_avg = np.linalg.norm(center - avg_position)
            if distance_from_avg > 0.8:  # If more than 80cm from recent average
                # Apply quality penalty based on distance
                outlier_factor = distance_from_avg / 0.8  # Normalized distance
                quality_penalty = min(0.2, quality * 0.3)  # Cap at 20% reduction
                
                # Apply penalty
                new_quality = max(0.2, quality - quality_penalty)
                sorted_balls[0] = (center, points, new_quality)
                
                if self.detailed_logging:
                    self.get_logger().info(
                        f"OUTLIER: Position ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) is {distance_from_avg:.2f}m "
                        f"from history average - quality reduced from {quality:.2f} to {new_quality:.2f}"
                    )
        
        # Improve prediction code:

        # When using prediction, apply a weighted blend with history
        if hasattr(self, 'predicted_position') and self.predicted_position is not None and hasattr(self, 'position_history') and len(self.position_history) > 0:
            # Calculate average of recent historical positions
            recent_positions = list(self.position_history)[-3:]  # Last 3 positions
            if recent_positions:
                try:
                    history_avg = np.mean(recent_positions, axis=0)
                    
                    # Blend prediction with history (75% prediction, 25% history)
                    self.predicted_position = self.predicted_position * 0.75 + history_avg * 0.25
                    
                    if self.detailed_logging:
                        self.get_logger().info(f"PREDICTION: Using history-blended position: ({self.predicted_position[0]:.2f}, {self.predicted_position[1]:.2f}, {self.predicted_position[2]:.2f})")
                except Exception as e:
                    self.get_logger().warn(f"Error blending prediction with history: {str(e)}")
        
        return sorted_balls
    
    def publish_ball_position(self, center, cluster_size, circle_quality, trigger_source, original_timestamp=None):
        """
        Publish the detected tennis ball position with quality information.
        """
        # Apply Kalman filtering if enabled
        filtered_center = center
        
        if self.use_kalman:
            # Use safe update method instead of direct update
            filtered_center = self.safe_update_kalman(center, circle_quality, cluster_size)
            
            # Calculate how much the filter corrected the position
            correction = np.linalg.norm(filtered_center - center)
            
            # Log the correction amount if significant
            if correction > 0.05 and self.detailed_logging:
                self.get_logger().info(
                    f"Kalman filter corrected position by {correction:.3f}m: "
                    f"Raw: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) → "
                    f"Filtered: ({filtered_center[0]:.2f}, {filtered_center[1]:.2f}, {filtered_center[2]:.2f})"
                )
        
        # Create message for ball position (3D point with timestamp)
        point_msg = PointStamped()
        
        # IMPORTANT: Use original timestamp if provided, otherwise use current time
        if original_timestamp and TimeUtils.is_timestamp_valid(original_timestamp):
            point_msg.header.stamp = original_timestamp
        else:
            point_msg.header.stamp = TimeUtils.now_as_ros_time()
        
        # Always use our consistent frame ID for proper transformation
        point_msg.header.frame_id = "lidar_frame"
        
        # Use the filtered position in the message
        point_msg.point.x = float(filtered_center[0])
        point_msg.point.y = float(filtered_center[1])
        point_msg.point.z = float(filtered_center[2])
        
        # Publish the ball position
        self.position_publisher.publish(point_msg)
        
        # Update statistics
        self.successful_detections += 1
        
        # Determine confidence level based on circle quality
        if circle_quality > TENNIS_BALL_CONFIG["quality_threshold"]["high"]:
            confidence_text = "HIGH"
        elif circle_quality > TENNIS_BALL_CONFIG["quality_threshold"]["medium"]:
            confidence_text = "MEDIUM"
        else:
            confidence_text = "LOW"
            
        # Log the detection with detailed information
        self.get_logger().info(
            f"LIDAR: Tennis ball detected at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) meters "
            f"with {cluster_size} points | Quality: {confidence_text} ({circle_quality:.2f}) | "
            f"Triggered by: {trigger_source}"
        )
        
        # Create visualization markers for RViz
        self.visualize_detection(center, cluster_size, circle_quality, trigger_source)
    
    def publish_transform(self):
        """
        Publish the LIDAR-to-camera transformation obtained through calibration.
        
        This transformation allows converting coordinates between the LIDAR
        and camera reference frames, which is essential for proper sensor fusion.
        """
        transform = TransformStamped()
        # Use current time from TimeUtils for the transform to ensure it's considered valid
        transform.header.stamp = TimeUtils.now_as_ros_time()
        transform.header.frame_id = self.transform_parent_frame  # Parent frame from config
        transform.child_frame_id = self.transform_child_frame    # Child frame from config
        
        # Translation from LIDAR to camera (from config)
        transform.transform.translation.x = self.transform_translation['x']
        transform.transform.translation.y = self.transform_translation['y']
        transform.transform.translation.z = self.transform_translation['z']
        
        # Rotation from LIDAR to camera as quaternion (from config)
        transform.transform.rotation.x = self.transform_rotation['x']
        transform.transform.rotation.y = self.transform_rotation['y']
        transform.transform.rotation.z = self.transform_rotation['z']
        transform.transform.rotation.w = self.transform_rotation['w']
        
        self.tf_broadcaster.sendTransform(transform)
        
        # Log the transform occasionally to verify it's being published
        current_time = TimeUtils.now_as_float()
        transform_log_interval = lidar_config.get('transform', {}).get('log_interval', 60.0)
        if not hasattr(self, 'last_transform_log') or current_time - self.last_transform_log > transform_log_interval:
            self.get_logger().info("Publishing LIDAR-to-camera transform from calibration")
            self.last_transform_log = current_time

    def visualize_detection(self, center, cluster_size, circle_quality, trigger_source):
        """
        Create visualization markers for the detected ball in RViz.
        
        Creates markers:
        1. A sphere representing the tennis ball (filtered position)
        2. A small sphere showing the raw detection (if Kalman is enabled)
        3. A text label showing detection source and quality
        
        Args:
            center (numpy.ndarray): Center of the detected ball [x, y, z]
            cluster_size (int): Number of points in the cluster
            circle_quality (float): How well the points match a circle (0-1)
            trigger_source (str): Which detector triggered this detection
        """
        markers = MarkerArray()
        
        # Use filtered position for the main marker if available
        display_center = center
        if self.use_kalman and hasattr(self, 'kalman_filter') and self.kalman_filter.initialized:
            display_center = self.kalman_filter.state[0:3]
        
        # Create a sphere marker for the ball - Main sphere (filtered position)
        ball_marker = Marker()
        ball_marker.header.frame_id = "lidar_frame"  # Use consistent frame ID
        ball_marker.header.stamp = self.scan_timestamp
        ball_marker.ns = "tennis_ball"
        ball_marker.id = 1
        ball_marker.type = Marker.SPHERE  # Show as a sphere
        ball_marker.action = Marker.ADD
        
        # Set position
        ball_marker.pose.position.x = display_center[0]
        ball_marker.pose.position.y = display_center[1]
        ball_marker.pose.position.z = display_center[2]
        ball_marker.pose.orientation.w = 1.0  # No rotation
        
        # Set color based on quality and source
        color_config = None
        if trigger_source.lower() == "yolo":
            color_config = VIZ_CONFIG['colors']['yolo']
        else:  # HSV
            color_config = VIZ_CONFIG['colors']['hsv']
        
        ball_marker.color.r = color_config['r']
        ball_marker.color.g = color_config['g']
        ball_marker.color.b = color_config['b']
        
        # Adjust transparency based on confidence
        base_alpha = color_config.get('base_alpha', 0.5)
        ball_marker.color.a = min(base_alpha + circle_quality * 0.5, 1.0)  # Higher quality = more opaque
        
        # Set size (tennis ball diameter)
        ball_marker.scale.x = self.ball_radius * 2.0
        ball_marker.scale.y = self.ball_radius * 2.0
        ball_marker.scale.z = self.ball_radius * 2.0
        
        # Set how long to display (from config)
        ball_marker.lifetime.sec = int(VIZ_CONFIG['marker_lifetime'])
        ball_marker.lifetime.nanosec = int((VIZ_CONFIG['marker_lifetime'] % 1) * 1e9)
        
        markers.markers.append(ball_marker)
        
        # Add a text marker to show source and quality
        text_marker = Marker()
        text_marker.header.frame_id = self.scan_frame_id
        text_marker.header.stamp = self.scan_timestamp
        text_marker.ns = "tennis_ball_text"
        text_marker.id = 2
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        # Position text above the ball (height from config)
        text_marker.pose.position.x = center[0]
        text_marker.pose.position.y = center[1]
        text_marker.pose.position.z = center[2] + VIZ_CONFIG['text_height_offset']
        text_marker.pose.orientation.w = 1.0
        
        # Set text content
        quality_pct = int(circle_quality * 100)
        text_marker.text = f"{trigger_source}: {quality_pct}%"
        
        # Set text appearance (from config)
        text_marker.scale.z = VIZ_CONFIG['text_size']
        text_color = VIZ_CONFIG['colors']['text']
        text_marker.color.r = text_color['r']
        text_marker.color.g = text_color['g']
        text_marker.color.b = text_color['b']
        text_marker.color.a = text_color['a']
        text_marker.lifetime.sec = int(VIZ_CONFIG['marker_lifetime'])
        text_marker.lifetime.nanosec = int((VIZ_CONFIG['marker_lifetime'] % 1) * 1e9)
        
        markers.markers.append(text_marker)
        
        # If Kalman filter is enabled, show the raw detection as a smaller sphere
        if self.use_kalman and not np.array_equal(center, display_center):
            raw_marker = Marker()
            raw_marker.header.frame_id = "lidar_frame"
            raw_marker.header.stamp = self.scan_timestamp
            raw_marker.ns = "tennis_ball_raw"
            raw_marker.id = 3
            raw_marker.type = Marker.SPHERE
            raw_marker.action = Marker.ADD
            
            # Position at the raw detection
            raw_marker.pose.position.x = center[0]
            raw_marker.pose.position.y = center[1]
            raw_marker.pose.position.z = center[2]
            raw_marker.pose.orientation.w = 1.0
            
            # Make it smaller and semi-transparent
            raw_marker.scale.x = self.ball_radius
            raw_marker.scale.y = self.ball_radius
            raw_marker.scale.z = self.ball_radius
            
            # Red color for raw detections
            raw_marker.color.r = 1.0
            raw_marker.color.g = 0.0
            raw_marker.color.b = 0.0
            raw_marker.color.a = 0.3
            
            raw_marker.lifetime.sec = int(VIZ_CONFIG['marker_lifetime'])
            raw_marker.lifetime.nanosec = int((VIZ_CONFIG['marker_lifetime'] % 1) * 1e9)
            
            markers.markers.append(raw_marker)
        
        # Publish the visualization
        self.marker_publisher.publish(markers)
    
    def publish_status(self):
        """
        Publish diagnostic information about the node's performance.
        
        Format is compatible with the central diagnostics node.
        """
        try:
            # Calculate running time
            current_time = TimeUtils.now_as_float()
            elapsed = current_time - self.start_time
            
            # Skip if we just started
            if elapsed < 0.1:
                return
                
            # Calculate performance statistics
            scan_rate = self.processed_scans / elapsed if elapsed > 0 else 0
            detection_rate = self.successful_detections / elapsed if elapsed > 0 else 0
            
            # Calculate average processing time
            avg_time = 0
            if self.detection_times:
                avg_time = sum(self.detection_times) / len(self.detection_times)
            
            # Health recovery over time (errors become less relevant)
            time_since_last_error = current_time - self.last_error_time
            if time_since_last_error > 30.0:  # After 30 seconds with no errors
                self.lidar_health = min(1.0, self.lidar_health + 0.05)  # Gradually recover
            
            # Calculate detection health (based on detection rate)
            expected_rate = 1.0  # expected detections/sec
            self.detection_health = min(1.0, detection_rate / expected_rate) if expected_rate > 0 else 0.5
            
            # Extract recent errors for diagnostics
            recent_errors = []
            cutoff_time = current_time - 300  # Last 5 minutes
            for error in self.errors:
                if error["timestamp"] > cutoff_time:
                    recent_errors.append(error["message"])
            
            # Create comprehensive diagnostics message in expected format
            diagnostics = {
                "timestamp": current_time,
                "node": "lidar",
                "uptime_seconds": elapsed,
                "status": "active",
                "health": {
                    "lidar_health": self.lidar_health,
                    "detection_health": self.detection_health,
                    "overall": (self.lidar_health * 0.7 + self.detection_health * 0.3)  # weighted average
                },
                "metrics": {
                    "processed_scans": self.processed_scans,
                    "successful_detections": self.successful_detections,
                    "scan_rate": scan_rate,
                    "detection_rate": detection_rate,
                    "avg_processing_time_ms": avg_time * 1000,
                    "detection_latency_ms": self.detection_latency,
                    "sources": {
                        "yolo_detections": self.yolo_detections,
                        "hsv_detections": self.hsv_detections,
                    }
                },
                "resources": {
                    "cpu_usage": getattr(self.resource_monitor, 'cpu_percent', 0),
                    "memory_usage": getattr(self.resource_monitor, 'mem_percent', 0),
                    "temperature": getattr(self.resource_monitor, 'temperature', 0)
                },
                "errors": recent_errors,
                "config": {
                    "ball_radius": self.ball_radius,
                    "max_distance": self.max_distance,
                    "min_points": self.min_points,
                    "detection_samples": self.detection_samples
                }
            }
            
            # Calculate additional health metrics
            scan_frequency_health = min(1.0, scan_rate / lidar_config.get('expected_scan_rate', 10.0))
            detection_latency_health = 1.0 - min(1.0, self.detection_latency / 100.0)

            diagnostics["health"]["scan_frequency_health"] = scan_frequency_health
            diagnostics["health"]["detection_latency_health"] = detection_latency_health 
            diagnostics["health"]["resource_health"] = 1.0 - (getattr(self.resource_monitor, 'cpu_percent', 0) / 100.0)

            # Overall health is weighted average of all components
            diagnostics["health"]["overall"] = (
                self.lidar_health * 0.3 + 
                self.detection_health * 0.3 + 
                scan_frequency_health * 0.2 +
                detection_latency_health * 0.1 +
                diagnostics["health"]["resource_health"] * 0.1
            )
            
            # Publish diagnostics as JSON string
            msg = String()
            msg.data = json.dumps(diagnostics)
            self.diagnostics_publisher.publish(msg)
            
            # Log basic summary (not the full diagnostics)
            self.get_logger().info(
                f"LIDAR: Status: {scan_rate:.1f} scans/sec, "
                f"{detection_rate:.1f} detections/sec, "
                f"YOLO: {self.yolo_detections}, HSV: {self.hsv_detections}, "
                f"Health: {diagnostics['health']['overall']:.2f}, "
                f"Errors: {len(recent_errors)}"
            )
            
        except Exception as e:
            self.log_error(f"Error publishing diagnostics: {str(e)}")
    
    def _configure_detection_algorithm(self):
        """Configure the detection algorithm based on hardware capabilities."""
        # Detection parameters
        # On Pi 5 with 16GB RAM, we can use more detection samples 
        # and keep full point resolution
        self.detection_samples = TENNIS_BALL_CONFIG["detection_samples"]
        
        # Check for Raspberry Pi 5
        is_raspberry_pi = os.environ.get('RASPBERRY_PI') == '1'
        
        if is_raspberry_pi:
            # On Raspberry Pi, use a more efficient detection approach
            self.get_logger().info("Configured for Raspberry Pi 5: using optimized detection")
            # Increase detection samples slightly since the Pi 5 has good performance
            self.detection_samples = min(40, self.detection_samples)
        else:
            # On desktop/laptop, use more thorough detection approach
            self.get_logger().info("Configured for desktop: using high-quality detection")
            # More detection samples for higher-powered systems
            self.detection_samples = min(50, self.detection_samples)
    
    def _handle_resource_alert(self, resource_type, value):
        """Handle resource alerts from the resource monitor."""
        # Track when we last had a resource alert
        current_time = TimeUtils.now_as_float()
        
        if resource_type == 'cpu' and value > 90:
            # Only log high CPU usage once per minute
            if not hasattr(self, 'last_high_cpu_time') or current_time - self.last_high_cpu_time > 60:
                self.get_logger().warning(f"High CPU usage detected: {value:.1f}%")
                self.last_high_cpu_time = current_time
                
                # Adjust detection algorithm to be more efficient
                self.detection_samples = max(15, self.detection_samples - 5)
                self.get_logger().info(f"Reducing detection samples to {self.detection_samples} to conserve resources")
        
        elif resource_type == 'memory' and value > 85:
            self.get_logger().warning(f"High memory usage detected: {value:.1f}%")
            
        elif resource_type == 'temperature' and value > 80:
            self.get_logger().error(f"Critical temperature detected: {value:.1f}°C")
    
    def log_error(self, error_message):
        """Log errors with consistent formatting and track for diagnostics."""
        # Add error to tracking list
        current_time = TimeUtils.now_as_float()
        self.errors.append({
            "timestamp": current_time,
            "message": error_message
        })
        
        # Update last error time for health tracking
        self.last_error_time = current_time
        
        # Reduce health score when errors occur
        if hasattr(self, 'lidar_health'):
            self.lidar_health = max(0.3, self.lidar_health - 0.2)
        
        # Log the error (but rate limit repeating errors)
        self.get_logger().error(f"LIDAR ERROR: {error_message}")
    
    def destroy_node(self):
        """Clean up resources when the node is shutting down."""
        # Clear any large stored data
        self.points_array = None
        if hasattr(self, '_points_buffer'):
            self._points_buffer = None
        if hasattr(self, '_filtered_buffer'):
            self._filtered_buffer = None
        if hasattr(self, '_cluster_buffer'):
            self._cluster_buffer = None
        
        # Stop any running threads
        if hasattr(self, 'resource_monitor') and self.resource_monitor:
            self.resource_monitor.stop()
        
        # Close log files if open
        if hasattr(self, 'cluster_log_file') and self.cluster_log_file:
            self.cluster_log_file.close()
        
        if hasattr(self, 'kalman_log_file') and self.kalman_log_file:
            self.kalman_log_file.close()
        
        super().destroy_node()

    def _configure_logging(self, log_config):
        """Configure logger levels based on config."""
        level_map = {
            'debug': rclpy.logging.LoggingSeverity.DEBUG,
            'info': rclpy.logging.LoggingSeverity.INFO,
            'warn': rclpy.logging.LoggingSeverity.WARN,
            'error': rclpy.logging.LoggingSeverity.ERROR
        }
        
        console_level = level_map.get(log_config.get('console_level', 'info').lower(), 
                                      rclpy.logging.LoggingSeverity.INFO)
        
        # Set logger level
        self.get_logger().set_level(console_level)
        self.get_logger().info(f"Logging level set to: {log_config.get('console_level', 'info').upper()}")

    # Create a helper method for consistent log formatting:

    def _log(self, level, component, message, extra_data=None):
        """Unified logging with component tags and optional data."""
        tagged_msg = f"[{component}] {message}"
        
        if extra_data and isinstance(extra_data, dict) and self.debug_mode:
            # Add data as JSON if in debug mode
            data_str = json.dumps(extra_data)
            if len(data_str) > 100:  # Truncate long data
                data_str = data_str[:97] + "..."
            tagged_msg += f" | {data_str}"
        
        if level == 'debug':
            self.get_logger().debug(tagged_msg)
        elif level == 'info':
            self.get_logger().info(tagged_msg)
        elif level == 'warn':
            self.get_logger().warn(tagged_msg)
        elif level == 'error':
            self.get_logger().error(tagged_msg)

    def safe_get_position_history(self):
        """Safely access the position history with proper validation."""
        if not hasattr(self, 'position_history') or self.position_history is None:
            self.position_history = deque(maxlen=10)
            return []
        
        # Filter out any None values that might have made it into history
        valid_positions = [pos for pos in self.position_history if pos is not None]
        return valid_positions

    def safe_update_kalman(self, center, circle_quality, cluster_size):
        """
        Safely update the Kalman filter, ensuring we never return None.
        
        Args:
            center (numpy.ndarray): Position measurement to update the filter with
            circle_quality (float): Quality of the detection (0-1)
            cluster_size (int): Number of points in the detection cluster
            
        Returns:
            numpy.ndarray: Filtered position, or original position if an error occurs
        """
        if center is None:
            self.get_logger().error("Attempted to update Kalman filter with None center")
            return np.zeros(3)
        
        try:
            # Verify Kalman filter has been properly initialized
            if not hasattr(self.kalman_filter, 'state'):
                self.get_logger().error("Kalman filter not properly initialized, reinitializing")
                # Reinitialize the Kalman filter
                self.kalman_filter = KalmanFilter()
                
            # If this is the first update, make sure we're initialized
            if not self.kalman_filter.initialized:
                self.get_logger().info("Initializing Kalman filter with first position")
                # Initialize directly with this measurement
                self.kalman_filter.state[0:3] = center
                self.kalman_filter.last_update_time = time.time()
                self.kalman_filter.initialized = True
                return center
            
            # Update with the measurement
            filtered_center = self.kalman_filter.update(center, circle_quality, cluster_size)
            
            # Double-check for None result
            if filtered_center is None:
                self.get_logger().error("Kalman filter returned None result, using original position")
                return np.array(center)
            
            # Ensure result is a numpy array
            if not isinstance(filtered_center, np.ndarray):
                self.get_logger().error(f"Kalman filter returned non-array: {type(filtered_center)}, using original position")
                return np.array(center)
            
            # Validate size
            if len(filtered_center) != 3:
                self.get_logger().error(f"Kalman filter returned array of wrong size: {len(filtered_center)}, using original position")
                return np.array(center)
            
            return filtered_center
            
        except Exception as e:
            self.get_logger().error(f"Error in Kalman filter update: {str(e)}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
            return np.array(center)  # Return original position if any error occurs

    def camera_detection_callback(self, msg, source):
        """
        Process ball detections from the camera and find matching points in LIDAR data.
        """
        detection_start_time = TimeUtils.now_as_float()
        
        try:
            # Check if we have scan data
            if self.latest_scan is None or self.points_array is None or len(self.points_array) == 0:
                self.get_logger().info("LIDAR: Waiting for scan data...")
                return
                
            # Extract the camera's detected position and confidence
            x_2d = msg.point.x
            y_2d = msg.point.y
            confidence = msg.point.z
            
            self.get_logger().info(
                f"{source}: Ball detected at pixel ({x_2d:.1f}, {y_2d:.1f}) "
                f"with confidence {confidence:.2f}"
            )
            
            # Find tennis ball patterns in LIDAR data
            ball_results = self.find_tennis_balls(source)
            
            # Process the best detected ball (if any)
            if ball_results and len(ball_results) > 0:
                # Get the best match (first in the list, already sorted by quality)
                best_match = ball_results[0]
                
                # Ensure best_match has the expected structure
                if len(best_match) == 3:
                    center, cluster_size, circle_quality = best_match
                    
                    # Safety check for center before publishing
                    if center is not None:
                        # Use original timestamp for synchronization
                        if TimeUtils.is_timestamp_valid(msg.header.stamp):
                            self.publish_ball_position(center, cluster_size, circle_quality, source, msg.header.stamp)
                        else:
                            self.get_logger().warn(f"Received invalid timestamp from {source}, using current time instead")
                            self.publish_ball_position(center, cluster_size, circle_quality, source, None)
                    else:
                        self.get_logger().warn(f"LIDAR: Invalid center point in ball result for {source}")
                else:
                    self.get_logger().warn(f"LIDAR: Unexpected structure in ball result for {source}")
            else:
                self.get_logger().info(f"LIDAR: No matching ball found for {source} detection")
                
        except Exception as e:
            error_msg = f"Error processing {source} detection: {str(e)}"
            self.log_error(error_msg)
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
        
        # Log processing time for this detection
        processing_time = (TimeUtils.now_as_float() - detection_start_time) * 1000  # in ms
        self.detection_times.append(processing_time)

class KalmanFilter:
    """Simple Kalman filter for tracking a tennis ball in 3D space."""
    
    def __init__(self):
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        
        # Initial uncertainty is high
        self.P = np.eye(6) * 0.9  # Decreased from 1.0 to 0.9
        
        # Process noise (how much we expect the state to change between predictions)
        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= 0.015  # Decreased from 0.02 to 0.015
        self.Q[3:6, 3:6] *= 0.12   # Decreased from 0.15 to 0.12
        
        # Measurement noise (how much we trust the measurements)
        self.R = np.eye(3) * 0.18  # Decreased from 0.25 to 0.18 - trust measurements even more
        
        # State transition matrix (physics model)
        self.F = np.eye(6)
        # During prediction, position += velocity * dt
        self.dt = 0.1  # default time step
        
        # Measurement matrix (maps state to measurement)
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)  # We only measure position, not velocity
        
        self.initialized = False
        self.last_update_time = None
    
    def predict(self, dt=None):
        """Predict next state based on motion model."""
        if not self.initialized:
            return
            
        # Use provided dt or default
        if dt is not None:
            self.dt = dt
        
        # Update state transition matrix with current dt
        self.F[0:3, 3:6] = np.eye(3) * self.dt
        
        # Predict next state: x = F * x
        self.state = self.F @ self.state
        
        # Update covariance: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state[0:3]  # Return predicted position
    
    def update(self, measurement, measurement_quality=1.0, cluster_size=0):
        """Update the filter with a new measurement."""
        # Validate measurement
        if measurement is None or not isinstance(measurement, np.ndarray) or len(measurement) != 3:
            # Log the issue and return the current state or zeros if not initialized
            return self.state[0:3] if self.initialized else np.zeros(3)
            
        current_time = time.time()
        
        # Get node reference for logging if available
        node = None
        for obj in globals().values():
            if isinstance(obj, TennisBallLidarDetector):
                node = obj
                break
        
        # Initialize if this is the first measurement
        if not self.initialized:
            self.state[0:3] = measurement
            self.last_update_time = current_time
            self.initialized = True
            
            if node and hasattr(node, 'kalman_debug') and node.kalman_debug:
                if hasattr(node, 'kalman_log_file') and node.kalman_log_file:
                    node.kalman_log_file.write(f"Kalman initialized with position: ({measurement[0]:.3f}, {measurement[1]:.3f}, {measurement[2]:.3f})\n")
            
            return self.state[0:3]
        
        # Calculate time since last update for prediction
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Save pre-update state for logging
        old_state = self.state.copy()
        
        # Don't use negative or very large dt values
        if dt > 0 and dt < 1.0:
            self.predict(dt)
        else:
            self.predict()
        
        # Adjust measurement noise based on quality AND point count
        # Lower quality = higher noise = less weight to this measurement
        quality_factor = max(0.2, measurement_quality)  # Ensure minimum quality factor
        
        # Add point count weighting - more points = more trust
        if cluster_size > 0:
            # More aggressive scaling based on point count
            if cluster_size < 10:
                # Very low point counts get less trust
                point_factor = 0.4 + (cluster_size / 20.0)  # 0.4 to 0.9 for 1-10 points
            else:
                # Higher point counts get more trust
                point_factor = min(2.5, 0.9 + (cluster_size - 10) / 20.0)  # 0.9 to 2.5
                
            quality_factor *= point_factor
        
        R_adjusted = self.R / quality_factor
        
        # For sudden large changes, increase the noise further
        if self.initialized:
            position_change = np.linalg.norm(measurement - self.state[0:3])
            if position_change > 0.3:  # If position jumped significantly
                jump_factor = min(3.0, position_change / 0.3)  # Scale up to 3x
                R_adjusted = R_adjusted * jump_factor
                
                # Add debug logging for large jumps
                if node and hasattr(node, 'detailed_logging') and node.detailed_logging:
                    node.get_logger().info(
                        f"KALMAN: Large position jump detected ({position_change:.2f}m), "
                        f"increasing measurement noise by {jump_factor:.1f}x"
                    )
        
        # Calculate innovation: y = z - H*x
        y = measurement - self.H @ self.state
        
        # Calculate innovation covariance: S = H*P*H^T + R
        S = self.H @ self.P @ self.H.T + R_adjusted
        
        # Default K to None for safer checking later
        K = None
        
        try:
            # Calculate Kalman gain: K = P*H^T*S^-1
            K = self.P @ self.H.T @ np.linalg.inv(S)
            
            # Update state: x = x + K*y
            self.state = self.state + K @ y
            
            # Update covariance: P = (I - K*H)*P
            I = np.eye(self.state.shape[0])
            self.P = (I - K @ self.H) @ self.P
        except np.linalg.LinAlgError:
            # Handle potential matrix inversion issues
            if node:
                node.get_logger().warn("Kalman filter: Matrix inversion failed, skipping update")
            # Don't update state - we'll use the predicted state
        
        # Enhanced logging if enabled
        if node and hasattr(node, 'kalman_debug') and node.kalman_debug:
            if hasattr(node, 'kalman_log_file') and node.kalman_log_file:
                try:
                    # Log detailed Kalman filter state
                    log_file = node.kalman_log_file
                    log_file.write(f"--- Kalman Update at t={current_time:.3f}, dt={dt:.3f}s ---\n")
                    log_file.write(f"Measurement: ({measurement[0]:.3f}, {measurement[1]:.3f}, {measurement[2]:.3f}) quality={measurement_quality:.2f}\n")
                    log_file.write(f"Pre-Update Position: ({old_state[0]:.3f}, {old_state[1]:.3f}, {old_state[2]:.3f})\n")
                    log_file.write(f"Pre-Update Velocity: ({old_state[3]:.3f}, {old_state[4]:.3f}, {old_state[5]:.3f}) m/s\n")
                    log_file.write(f"Innovation: ({y[0]:.3f}, {y[1]:.3f}, {y[2]:.3f})\n")
                    
                    if K is not None:
                        log_file.write(f"Kalman Gain: [{K[0,0]:.3f}, {K[1,1]:.3f}, {K[2,2]:.3f}]\n")
                    else:
                        log_file.write("Kalman Gain: [ERROR - matrix inversion failed]\n")
                        
                    log_file.write(f"Post-Update Position: ({self.state[0]:.3f}, {self.state[1]:.3f}, {self.state[2]:.3f})\n")
                    log_file.write(f"Post-Update Velocity: ({self.state[3]:.3f}, {self.state[4]:.3f}, {self.state[5]:.3f}) m/s\n")
                    log_file.write(f"Position Change: ({(self.state[0]-old_state[0])::.3f}, {(self.state[1]-old_state[1]):.3f}, {(self.state[2]-old_state[2]):.3f})\n")
                    log_file.write(f"Velocity Change: ({(self.state[3]-old_state[3])::.3f}, {(self.state[4]-old_state[4])::.3f}, {(self.state[5]-old_state[5])::.3f})\n\n")
                except Exception as e:
                    if node:
                        node.get_logger().error(f"Error writing to Kalman log file: {str(e)}")
        
        # After calculating filtered position, limit the maximum correction
        position_change = np.linalg.norm(self.state[0:3] - old_state[0:3])
        if position_change > 0.4:  # Reduce max correction from 0.5m to 0.4m
            correction_vector = self.state[0:3] - old_state[0:3]
            normalized = correction_vector / position_change
            self.state[0:3] = old_state[0:3] + normalized * 0.4
            if node and hasattr(node, 'detailed_logging') and node.detailed_logging:
                node.get_logger().info(f"KALMAN: Limited correction to 0.4m (was {position_change:.2f}m)")
                
        return self.state[0:3]  # Return updated position


def main():
    """Main entry point for the LIDAR node."""
    # Initialize ROS
    rclpy.init()
    
    # Create node
    detector = TennisBallLidarDetector()
    
    # Print welcome message
    print("=================================================")
    print("Tennis Ball LIDAR Detector")
    print("=================================================")
    print("This node detects tennis balls using a 2D laser scanner.")
    print(f"Subscribing to LIDAR data on: {TOPICS['input']['lidar_scan']}")
    print(f"Subscribing to YOLO detections on: {TOPICS['input']['yolo_detection']}")
    print(f"Subscribing to HSV detections on: {TOPICS['input']['hsv_detection']}")
    print(f"Publishing ball positions to: {TOPICS['output']['ball_position']}")
    print("Use RViz to see the visualizations.")
    print("Press Ctrl+C to stop the program.")
    print("=================================================")
    
    try:
        # Set thread priority for better real-time performance on Linux (Pi)
        try:
            import os
            os.nice(10)  # Lower priority slightly to favor critical nodes
        except:
            pass
            
        # Keep the node running
        rclpy.spin(detector)
    except KeyboardInterrupt:
        print("LIDAR: Shutting down (Ctrl+C pressed)")
    except Exception as e:
        print(f"LIDAR: Error: {str(e)}")
    finally:
        # Clean shutdown
        detector.destroy_node()
        rclpy.shutdown()
        print("LIDAR: Tennis Ball LIDAR Detector has been shut down.")

if __name__ == '__main__':
    main()