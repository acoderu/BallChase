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



# Tennis ball configuration from config file - fix parameter values for better detection
TENNIS_BALL_CONFIG = lidar_config.get('tennis_ball', {
    "radius": 0.033,         # Tennis ball radius in meters
    "height": -0.20,         # Expected height of ball center relative to LIDAR
    "max_distance": 0.12,    # Optimal distance for clustering
    "min_points": 6,         # Minimum points needed for valid detection
    "quality_threshold": {   # Thresholds for circle quality assessment
        "low": 0.35,         # Base threshold for valid detection
        "medium": 0.6,       # Medium confidence threshold
        "high": 0.8          # High confidence threshold
    },
    "detection_samples": 40  # Number of sampling points to try
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
        
        # Simplify debug flags - use a single detailed_logging parameter
        self.declare_parameter('detailed_logging', False)
        self.detailed_logging = self.get_parameter('detailed_logging').value
        
        # Add logging files based on detailed_logging flag
        self.cluster_log_file = None
        self.kalman_log_file = None
        
        if self.detailed_logging:
            try:
                self.cluster_log_file = open("cluster_analysis.log", "a")
                self.cluster_log_file.write("-------- New Session Started --------\n")
                self.cluster_log_file.write(f"Timestamp: {TimeUtils.now_as_float()}\n\n")
                
                self.kalman_log_file = open("kalman_analysis.log", "a")
                self.kalman_log_file.write("-------- New Session Started --------\n")
                self.kalman_log_file.write(f"Timestamp: {TimeUtils.now_as_float()}\n\n")
            except Exception as e:
                self.get_logger().error(f"Could not create log files: {e}")

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
        """Search for tennis balls using a grid-based approach."""
        # Start timing
        operation_start = TimeUtils.now_as_float()
        
        if self.points_array is None or len(self.points_array) == 0:
            self.get_logger().warn("LIDAR: No points available for analysis")
            return []

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
        dynamic_min_points = self.min_points
        dynamic_quality_threshold = TENNIS_BALL_CONFIG["quality_threshold"]["low"]
        
        # If we have consecutive failures, gradually reduce requirements
        if self.consecutive_failures > 2:
            # Reduce minimum points requirement (but not below 5)
            dynamic_min_points = max(5, int(self.min_points * (1.0 - 0.1 * min(self.consecutive_failures, 5))))
            
            # Reduce quality threshold (but not below 0.3)
            dynamic_quality_threshold = max(0.3, dynamic_quality_threshold * (1.0 - 0.1 * min(self.consecutive_failures, 5)))
            
            if self.detailed_logging:
                self.get_logger().info(
                    f"ADAPTIVE: After {self.consecutive_failures} failures, using reduced requirements: "
                    f"min_points={dynamic_min_points} (from {self.min_points}), "
                    f"quality={dynamic_quality_threshold:.2f} (from {TENNIS_BALL_CONFIG['quality_threshold']['low']:.2f})"
                )
        
        # SIMPLIFIED: Create seed points using a more straightforward approach
        seed_points = []
        
        # Always include previous ball position if available (highest priority)
        if self.previous_ball_position is not None:
            seed_points.append(self.previous_ball_position)
            
        # Add predicted position if available (from Kalman filter)
        if self.predicted_position is not None:
            # Only use prediction if it's recent
            time_since_prediction = current_time - getattr(self, 'prediction_time', 0)
            if time_since_prediction < 1.0:  # Only use predictions less than 1 second old
                seed_points.append(self.predicted_position)
                
                if self.detailed_logging:
                    self.get_logger().info(
                        f"PREDICTION: Using predicted position at "
                        f"({self.predicted_position[0]:.2f}, {self.predicted_position[1]:.2f}, {self.predicted_position[2]:.2f})"
                    )
        
        # Create a grid of seed points in the area where balls are likely to be found
        if len(points) > 0:
            x_min, y_min = np.min(points[:, 0:2], axis=0) 
            x_max, y_max = np.max(points[:, 0:2], axis=0)
            
            # Create a grid with reasonable spacing
            grid_spacing = self.ball_radius * 1.5
            x_grid = np.arange(x_min, x_max, grid_spacing)
            y_grid = np.arange(y_min, y_max, grid_spacing)
            
            # Limit grid size to avoid excessive computation
            max_grid_points = 25
            if len(x_grid) > max_grid_points:
                x_grid = np.linspace(x_min, x_max, max_grid_points)
            if len(y_grid) > max_grid_points:
                y_grid = np.linspace(y_min, y_max, max_grid_points)
            
            # Create grid points - focus on forward-facing area
            for x in x_grid:
                for y in y_grid:
                    distance = np.sqrt(x**2 + y**2)
                    # Only add grid points within reasonable range
                    if distance < 2.0:
                        seed_points.append(np.array([x, y, self.ball_height]))
        
        # Cap the number of seed points
        max_seed_points = int(self.detection_samples)
        if len(seed_points) > max_seed_points:
            # Keep the first few points (previous position, prediction) and sample from the rest
            keep_count = min(2, len(seed_points))
            points_to_keep = seed_points[:keep_count]
            points_to_sample = seed_points[keep_count:]
            
            if len(points_to_sample) > max_seed_points - keep_count:
                sample_indices = np.random.choice(
                    len(points_to_sample),
                    size=max_seed_points - keep_count,
                    replace=False
                )
                seed_points = points_to_keep + [points_to_sample[i] for i in sample_indices]
            else:
                seed_points = points_to_keep + points_to_sample
        
        # Process each seed point to find clusters
        for seed_point in seed_points:
            # Find points close to this seed point
            distances = np.sqrt(
                (points[:, 0] - seed_point[0])**2 + 
                (points[:, 1] - seed_point[1])**2
            )
            
            # Use adaptive clustering radius based on failures
            effective_max_distance = self.max_distance
            if self.consecutive_failures > 3:
                # Gradually increase cluster radius for persistent failures
                effective_max_distance = min(0.18, self.max_distance * (1.0 + 0.1 * min(self.consecutive_failures, 5)))
            
            cluster_indices = np.where(distances < effective_max_distance)[0]
            cluster = points[cluster_indices]
            
            # Skip if cluster is too small
            if len(cluster) < dynamic_min_points:
                continue
            
            # Calculate the center of the cluster (centroid)
            center = np.mean(cluster, axis=0)
            
            # Use vectorized operations for circle quality check
            center_distances = np.sqrt(
                (cluster[:, 0] - center[0])**2 + 
                (cluster[:, 1] - center[1])**2
            )
            
            # Reject clusters with too much variance in radius
            radius_std = np.std(center_distances)
            if radius_std > self.ball_radius * 0.5:  # Std dev should be less than 50% of radius
                continue
            
            # Calculate quality metrics
            radius_errors = np.abs(center_distances - self.ball_radius)
            avg_error = np.mean(radius_errors)
            circle_quality = 1.0 - (avg_error / self.ball_radius)
            
            # Calculate additional quality factors
            point_density_score = min(1.0, len(cluster) / 50.0)
            distance_to_lidar = np.sqrt(center[0]**2 + center[1]**2)
            distance_quality = 1.0 - min(1.0, distance_to_lidar / 2.0)
            
            # Combine into a weighted score
            combined_quality = (
                circle_quality * 0.4 +
                point_density_score * 0.5 +
                distance_quality * 0.1
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
        
        # Implement position consistency checking for multiple candidates
        if len(sorted_balls) > 1 and self.previous_ball_position is not None:
            consistent_balls = []
            # Adaptive position change threshold
            max_position_change = 0.5  # Base value
            
            # Increase threshold if we've had failures
            if self.consecutive_failures > 2:
                max_position_change = min(1.0, max_position_change * (1.0 + 0.1 * self.consecutive_failures))
            
            for ball in sorted_balls:
                center, points, quality = ball
                dist_from_previous = np.linalg.norm(center - self.previous_ball_position)
                
                # If this cluster is close to previous position, it's likely the same ball
                if dist_from_previous < max_position_change:
                    consistent_balls.append(ball)
                elif self.detailed_logging:
                    self.get_logger().info(
                        f"CONSISTENCY: Rejected cluster at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) "
                        f"too far from previous position ({dist_from_previous:.2f}m > {max_position_change:.2f}m)")
            
            # If we found any consistent balls, use those instead
            if consistent_balls:
                sorted_balls = sorted(consistent_balls, key=lambda x: x[2], reverse=True)
        
        # Update tracking based on detection results
        if sorted_balls:
            best_ball = sorted_balls[0]
            new_position = best_ball[0]
            
            # Store in position history
            self.position_history.append(new_position)
            
            # Set current position 
            self.previous_ball_position = new_position
            
            # Reset consecutive failures since we found a ball
            self.consecutive_failures = 0
            
            # Update last successful detection time
            self.last_successful_detection_time = current_time
        else:
            # Increment consecutive failures
            self.consecutive_failures += 1
            
            # Generate prediction from Kalman filter if we have one
            if hasattr(self, 'kalman_filter') and self.kalman_filter.initialized:
                # Only predict if we had a previous successful detection
                time_since_detection = current_time - self.last_successful_detection_time
                # Only predict for reasonable time periods
                if time_since_detection < 2.0:
                    self.predicted_position = self.kalman_filter.predict(time_since_detection)
                    self.prediction_time = current_time
        
        # SIMPLIFIED APPROACH: Simple cluster merging for nearby detections
        # This combines evidence from reflections of the same ball
        if len(sorted_balls) > 1:
            merge_threshold = self.ball_radius * 9  # ~30cm for tennis ball
            merged_balls = []
            processed = set()
            
            for i, ball in enumerate(sorted_balls):
                if i in processed:
                    continue
                    
                center_i, points_i, quality_i = ball
                merged_points = points_i
                weighted_center = center_i * points_i  # Weight by point count
                merged_quality = quality_i * points_i  # Weighted quality
                
                # Look for balls to merge with this one
                for j, other_ball in enumerate(sorted_balls):
                    if j == i or j in processed:
                        continue
                        
                    center_j, points_j, quality_j = other_ball
                    distance = np.linalg.norm(center_i - center_j)
                    
                    if distance < merge_threshold:
                        weighted_center += center_j * points_j
                        merged_points += points_j
                        merged_quality += quality_j * points_j
                        processed.add(j)
                
                # Calculate merged center and quality
                merged_center = weighted_center / merged_points
                merged_quality_score = merged_quality / merged_points
                
                merged_balls.append((merged_center, merged_points, merged_quality_score))
            
            # Replace with merged result if we did any merging
            if merged_balls:
                sorted_balls = sorted(merged_balls, key=lambda x: x[2], reverse=True)
        
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
        """Process ball detections from the camera and find matching points in LIDAR data."""
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
                # Get the best match (first in the list)
                best_match = ball_results[0]
                center, cluster_size, circle_quality = best_match
                
                # Use original timestamp for synchronization if valid
                if TimeUtils.is_timestamp_valid(msg.header.stamp):
                    self.publish_ball_position(center, cluster_size, circle_quality, source, msg.header.stamp)
                else:
                    self.get_logger().warn(f"Received invalid timestamp from {source}, using current time instead")
                    self.publish_ball_position(center, cluster_size, circle_quality, source, None)
            else:
                self.get_logger().info(f"LIDAR: No matching ball found for {source} detection")
                
                # Enhanced recovery for failed detections
                if self.detailed_logging:
                    self.get_logger().info(f"NO_MATCH: Point cloud has {len(self.points_array)} points")
                
                # Try with reduced requirements if needed
                if self.consecutive_failures >= 3:
                    saved_min_points = self.min_points
                    self.min_points = max(5, int(self.min_points * 0.5))  # Try with half the minimum points
                    debug_results = self.find_tennis_balls(f"{source}_DEBUG")
                    self.min_points = saved_min_points  # Restore original
                    
                    if debug_results:
                        recovery_center, recovery_points, recovery_quality = debug_results[0]
                        self.get_logger().info(f"RECOVERY: Found possible match with {recovery_points} points")
                        
                        # Use a reduced quality score for this fallback detection
                        fallback_quality = min(0.4, recovery_quality)
                        if TimeUtils.is_timestamp_valid(msg.header.stamp):
                            self.publish_ball_position(recovery_center, recovery_points, fallback_quality, f"{source}_FALLBACK", msg.header.stamp)
                        else:
                            self.publish_ball_position(recovery_center, recovery_points, fallback_quality, f"{source}_FALLBACK", None)
                
        except Exception as e:
            self.log_error(f"Error processing {source} detection: {str(e)}")
        
        # Log processing time
        processing_time = (TimeUtils.now_as_float() - detection_start_time) * 1000  # in ms
        self.detection_times.append(processing_time)
        
        # Update detection latency metric
        self.detection_latency = processing_time
        
        self.get_logger().debug(f"LIDAR: {source} processing took {processing_time:.2f}ms")

    # SIMPLIFIED KALMAN FILTER
class KalmanFilter:
    """Simple Kalman filter for tracking a tennis ball in 3D space."""
    
    def __init__(self):
        # State: [x, y, z, vx, vy, vz]
        self.state = np.zeros(6)
        
        # Initial uncertainty is high
        self.P = np.eye(6) * 0.8
        
        # Process noise (how much we expect the state to change between predictions)
        self.Q = np.eye(6)
        self.Q[0:3, 0:3] *= 0.02  # Position variance
        self.Q[3:6, 3:6] *= 0.2   # Velocity variance
        
        # Measurement noise (how much we trust the measurements)
        self.R = np.eye(3) * 0.2
        
        # State transition matrix (physics model)
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * 0.1  # Default dt = 0.1
        self.dt = 0.1  # default time step
        
        # Measurement matrix (maps state to measurement)
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)  # We only measure position, not velocity
        
        self.initialized = False
        self.last_update_time = None
    
    def predict(self, dt=None):
        """Predict next state based on motion model."""
        if not self.initialized:
            return np.zeros(3)
            
        # Use provided dt or default
        if dt is not None:
            self.dt = dt
            self.F[0:3, 3:6] = np.eye(3) * dt
        
        # Predict next state: x = F * x
        self.state = self.F @ self.state
        
        # Update covariance: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.state[0:3]  # Return predicted position
    
    def update(self, measurement, measurement_quality=1.0, cluster_size=0):
        """Update the filter with a new measurement."""
        current_time = time.time()
        
        # Initialize if this is the first measurement
        if not self.initialized:
            self.state[0:3] = measurement
            self.last_update_time = current_time
            self.initialized = True
            return self.state[0:3]
        
        # Calculate time since last update for prediction
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Don't use negative or very large dt values
        if dt > 0 and dt < 1.0:
            self.predict(dt)
        else:
            self.predict()
        
        # Adjust measurement noise based on quality and cluster size
        quality_factor = max(0.1, measurement_quality)
        
        # Better points = more trust
        if cluster_size > 0:
            point_factor = min(1.5, 0.5 + (cluster_size / 20.0))
            quality_factor *= point_factor
        
        R_adjusted = self.R / quality_factor
        
        try:
            # Calculate innovation: y = z - H*x
            y = measurement - self.H @ self.state
            
            # Calculate innovation covariance: S = H*P*H^T + R
            S = self.H @ self.P @ self.H.T + R_adjusted
            
            # Calculate Kalman gain: K = P*H^T*S^-1
            K = self.P @ self.H.T @ np.linalg.inv(S)
            
            # Update state: x = x + K*y
            self.state = self.state + K @ y
            
            # Update covariance: P = (I - K*H)*P
            I = np.eye(self.state.shape[0])
            self.P = (I - K @ self.H) @ self.P
        except np.linalg.LinAlgError:
            # Handle matrix inversion issues gracefully
            pass
        
        # Limit the maximum velocity to avoid instability
        max_velocity = 2.0  # m/s
        speed = np.linalg.norm(self.state[3:6])
        if speed > max_velocity:
            self.state[3:6] = self.state[3:6] * (max_velocity / speed)
        
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