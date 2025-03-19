#!/usr/bin/env python3

"""
Tennis Ball Tracking Robot - Sensor Fusion Node
===============================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving tennis ball.
The system uses multiple sensing modalities which are combined in this fusion node:
- YOLO neural network detection (fast but only 2D)
- HSV color-based detection (very fast but only 2D)
- LIDAR for depth sensing (accurate 3D but sparse)
- Depth camera for additional depth information (dense 3D but slower)

This Node's Purpose:
------------------
This sensor fusion node integrates data from all detection methods to create a single,
reliable estimate of the tennis ball's 3D position and velocity. It uses a Kalman filter,
which is a mathematical technique for estimating the true state of a system from noisy
measurements.

Key Concepts:
-----------
1. Kalman Filter: A recursive algorithm that uses:
   - A process model (how we expect the ball to move)
   - Measurements from various sensors
   - Knowledge of noise/uncertainty in both
   To produce an optimal estimate of the ball's position and velocity

2. State Vector: What we're tracking with our Kalman filter
   - [x, y, z, vx, vy, vz] = 3D position + 3D velocity

3. Prediction Step: Predicts where the ball will be based on physics
   - Uses constant velocity model: position += velocity * time
   - Adds uncertainty based on how much time has passed

4. Update Step: Incorporates new measurements
   - Weighs measurements based on their reliability
   - Updates both the estimated position/velocity and their uncertainty

5. Covariance Matrix: Represents uncertainty in our estimates
   - Diagonal elements: variance (uncertainty) of each state variable
   - Off-diagonal elements: correlation between state variables

Data Pipeline:
-------------
1. Previous nodes publish to:
   - '/tennis_ball/yolo/position' (2D from YOLO)
   - '/tennis_ball/hsv/position' (2D from HSV)
   - '/tennis_ball/yolo/position_3d' (3D from depth camera with YOLO)
   - '/tennis_ball/hsv/position_3d' (3D from depth camera with HSV)
   - '/tennis_ball/lidar/position' (3D from LIDAR)

2. This fusion node:
   - Subscribes to all these topics
   - Maintains a Kalman filter to estimate the true ball state
   - Publishes the fused 3D position and velocity
   - Publishes diagnostic information and tracking reliability metrics

3. Next in pipeline:
   - State management node uses the fused position and tracking reliability 
   - PID controller uses position and velocity for smooth following
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TwistStamped
from std_msgs.msg import String, Float32, Bool, Float64MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import math
from collections import deque
import time
import json
import os  # Add os import

# New imports for synchronization
from ball_tracking.sensor_sync_buffer import SimpleSensorBuffer
from ball_tracking.time_utils import TimeUtils  # Add TimeUtils import

# Import ConfigLoader
from config.config_loader import ConfigLoader

# Load configuration
config_loader = ConfigLoader()
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'fusion_config.yaml')
fusion_config = config_loader.load_yaml(config_path)

# Add startup configuration for transform waiting
STARTUP_CONFIG = fusion_config.get('startup', {
    'wait_for_transform': True,
    'transform_retry_count': 20,
    'transform_retry_delay': 1.0,  # seconds
    'required_transforms': ['lidar_frame', 'camera_frame']
})

# Topic configuration (ensures consistency with other nodes)
TOPICS = fusion_config.get('topics', {
    "input": {
        "yolo_2d": "/tennis_ball/yolo/position",
        "hsv_2d": "/tennis_ball/hsv/position",
        "yolo_3d": "/tennis_ball/yolo/position_3d",
        "hsv_3d": "/tennis_ball/hsv/position_3d",
        "lidar": "/tennis_ball/lidar/position"  # Note: This is already 3D
    },
    "output": {
        "position": "/tennis_ball/fused/position",
        "velocity": "/tennis_ball/fused/velocity",
        "diagnostics": "/tennis_ball/fusion/diagnostics",  # Updated to match diagnostics node expectations
        "uncertainty": "/tennis_ball/fused/position_uncertainty",
        "tracking_status": "/tennis_ball/fused/tracking_status"
    }
})

# Add resource monitoring import
from ball_tracking.resource_monitor import ResourceMonitor

class KalmanFilterFusion(Node):
    """
    A ROS2 node that fuses multiple tennis ball detection sources using a Kalman filter.
    
    This node integrates data from:
    1. HSV detection (2D)
    2. YOLO detection (2D)
    3. Depth camera with YOLO (3D)
    4. Depth camera with HSV (3D)
    5. LIDAR (3D)
    
    It maintains a Kalman filter to estimate the true 3D position and velocity of the tennis ball,
    publishing a fused position estimate with confidence metrics.
    
    The node includes reliability metrics essential for state management, allowing the robot
    to determine when to track the ball versus when to search for it.
    """
    
    def __init__(self):
        """Initialize the Kalman filter fusion node with all required components."""
        super().__init__('kalman_filter_fusion')
        
        # Add resource monitoring for Raspberry Pi
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=10.0,  # Less frequent updates to reduce overhead
            enable_temperature=True
        )
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.resource_monitor.set_threshold('cpu', 85.0)  # Adjust thresholds for Raspberry Pi
        self.resource_monitor.set_threshold('memory', 90.0)
        self.resource_monitor.start()
        
        # Set up filter parameters
        self._declare_parameters()
        
        # Add a ROS tf2 listener to transform coordinates between frames
        self._setup_transform_listener()
        
        # Wait for required transforms to become available to avoid race conditions
        if STARTUP_CONFIG['wait_for_transform']:
            self._wait_for_transforms()
        
        # Create the synchronization buffer
        self.sync_buffer = SimpleSensorBuffer(
            sensor_names=['hsv_2d', 'yolo_2d', 'hsv_3d', 'yolo_3d', 'lidar'],
            buffer_size=30,      # Keep 30 recent measurements per sensor
            max_time_diff=self.max_time_diff  # Use parameter for time difference
        )
        
        # Set up subscriptions to receive data from all sensors
        self._setup_subscriptions()
        
        # Set up publishers to send out fused data
        self._setup_publishers()
        
        # Initialize Kalman filter state and variables
        self._init_kalman_filter()
        
        # Set up timers for periodic status updates and filter updates
        self._setup_timers()
        
        # Set up visualization and debugging tools
        self._init_debugging_tools()
        
        self.get_logger().info("Kalman Filter Fusion Node has started with synchronized input!")
        self.get_logger().info("Using synchronization window of %.3f seconds", self.max_time_diff)
        self.log_parameters()
        
        # Log topic connections for debugging
        self._log_topic_connections()

    def _wait_for_transforms(self):
        """
        Wait for required transforms to become available.
        
        This avoids race conditions on startup where transforms might not be
        immediately available from other nodes like the LIDAR node.
        """
        required_transforms = STARTUP_CONFIG['required_transforms']
        retry_count = STARTUP_CONFIG['transform_retry_count']
        retry_delay = STARTUP_CONFIG['transform_retry_delay']
        
        # Get the first frame as parent, second as child
        if len(required_transforms) >= 2:
            parent_frame = required_transforms[0]
            child_frame = required_transforms[1]
            
            self.get_logger().info(f"Waiting for transform from '{parent_frame}' to '{child_frame}'...")
            
            # Try multiple times to get the transform
            transform_available = False
            
            for attempt in range(retry_count):
                try:
                    when = rclpy.time.Time()
                    transform_available = self.tf_buffer.can_transform(
                        parent_frame,
                        child_frame,
                        when,
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    
                    if transform_available:
                        self.get_logger().info(f"Transform from '{parent_frame}' to '{child_frame}' is now available (attempt {attempt+1}/{retry_count})")
                        break
                    else:
                        self.get_logger().warn(f"Transform not yet available, waiting... ({attempt+1}/{retry_count})")
                        
                except Exception as e:
                    self.get_logger().warn(f"Error checking transform: {str(e)}")
                
                # Sleep before retrying
                time.sleep(retry_delay)
            
            # Final status
            if transform_available:
                self.get_logger().info("All required transforms are available. Proceeding with initialization.")
            else:
                self.get_logger().warn(
                    "Could not verify transform availability after multiple attempts. "
                    "Will proceed anyway, but transforms might fail until all nodes are fully initialized."
                )
        else:
            self.get_logger().warn("No transform frames specified in config. Skipping transform check.")
    
    def _setup_transform_listener(self):
        """Set up tf2 listener for coordinate transformations between sensors."""
        # Import tf2 modules here to avoid circular imports
        from tf2_ros import Buffer, TransformListener
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("Transform listener initialized for cross-sensor coordination")

    def _declare_parameters(self):
        """Declare and load all node parameters with descriptive comments."""
        self.declare_parameters(
            namespace='',
            parameters=[
                # Process noise: how much uncertainty to add during prediction steps
                ('process_noise_pos', fusion_config.get('process_noise', {}).get('position', 0.1)),
                ('process_noise_vel', fusion_config.get('process_noise', {}).get('velocity', 1.0)),
                
                # Measurement noise: how much to trust each sensor type
                ('measurement_noise_hsv_2d', fusion_config.get('measurement_noise', {}).get('hsv_2d', 50.0)),
                ('measurement_noise_yolo_2d', fusion_config.get('measurement_noise', {}).get('yolo_2d', 30.0)),
                ('measurement_noise_hsv_3d', fusion_config.get('measurement_noise', {}).get('hsv_3d', 0.05)),
                ('measurement_noise_yolo_3d', fusion_config.get('measurement_noise', {}).get('yolo_3d', 0.04)),
                ('measurement_noise_lidar', fusion_config.get('measurement_noise', {}).get('lidar', 0.03)),
                
                # Filter tuning parameters
                ('max_time_diff', fusion_config.get('filter', {}).get('max_time_diff', 0.2)),
                ('min_confidence_threshold', fusion_config.get('filter', {}).get('min_confidence_threshold', 0.5)),
                ('detection_timeout', fusion_config.get('filter', {}).get('detection_timeout', 0.5)),
                
                # Tracking reliability thresholds
                ('position_uncertainty_threshold', fusion_config.get('tracking', {}).get('position_uncertainty_threshold', 0.5)),
                ('velocity_uncertainty_threshold', fusion_config.get('tracking', {}).get('velocity_uncertainty_threshold', 1.0)),
                
                # Debugging and diagnostics
                ('history_length', fusion_config.get('diagnostics', {}).get('history_length', 100)),
                ('debug_level', fusion_config.get('diagnostics', {}).get('debug_level', 1)),
                ('log_to_file', fusion_config.get('diagnostics', {}).get('log_to_file', False))
            ]
        )
        
        # Get all parameters
        self.process_noise_pos = self.get_parameter('process_noise_pos').value
        self.process_noise_vel = self.get_parameter('process_noise_vel').value
        self.measurement_noise_hsv_2d = self.get_parameter('measurement_noise_hsv_2d').value
        self.measurement_noise_yolo_2d = self.get_parameter('measurement_noise_yolo_2d').value
        self.measurement_noise_hsv_3d = self.get_parameter('measurement_noise_hsv_3d').value
        self.measurement_noise_yolo_3d = self.get_parameter('measurement_noise_yolo_3d').value
        self.measurement_noise_lidar = self.get_parameter('measurement_noise_lidar').value
        self.max_time_diff = self.get_parameter('max_time_diff').value
        self.min_confidence_threshold = self.get_parameter('min_confidence_threshold').value
        self.detection_timeout = self.get_parameter('detection_timeout').value
        self.position_uncertainty_threshold = self.get_parameter('position_uncertainty_threshold').value
        self.velocity_uncertainty_threshold = self.get_parameter('velocity_uncertainty_threshold').value
        self.history_length = self.get_parameter('history_length').value
        self.debug_level = self.get_parameter('debug_level').value
        self.log_to_file = self.get_parameter('log_to_file').value

    def _setup_subscriptions(self):
        """Set up all subscriptions for this node."""
        # Create subscribers for all detection sources
        # 2D detections (from camera)
        self.hsv_2d_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["hsv_2d"],
            lambda msg: self.sensor_callback(msg, 'hsv_2d'),
            10
        )
        
        self.yolo_2d_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["yolo_2d"],
            lambda msg: self.sensor_callback(msg, 'yolo_2d'),
            10
        )
        
        # 3D detections (from depth camera)
        self.hsv_3d_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["hsv_3d"],
            lambda msg: self.sensor_callback(msg, 'hsv_3d'),
            10
        )
        
        self.yolo_3d_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["yolo_3d"],
            lambda msg: self.sensor_callback(msg, 'yolo_3d'),
            10
        )
        
        # LIDAR detections (already 3D)
        self.lidar_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["lidar"],
            lambda msg: self.sensor_callback(msg, 'lidar'),
            10
        )

    def _setup_publishers(self):
        """Set up all publishers for this node."""
        # Publisher for fused 3D position
        self.position_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["position"],
            10
        )
        
        # Publisher for velocity (useful for PID)
        self.velocity_publisher = self.create_publisher(
            TwistStamped,
            TOPICS["output"]["velocity"],
            10
        )
        
        # Publisher for tracking reliability metrics (used by state manager)
        self.diagnostics_publisher = self.create_publisher(
            String,
            TOPICS["output"]["diagnostics"],
            10
        )
        
        # Publisher for position uncertainty (for visualization and debugging)
        self.uncertainty_publisher = self.create_publisher(
            Float32,
            TOPICS["output"]["uncertainty"],
            10
        )
        
        # Publisher for tracking status (boolean indicating if tracking is reliable)
        self.tracking_status_publisher = self.create_publisher(
            Bool,
            TOPICS["output"]["tracking_status"],
            10
        )
        
        # Add a more comprehensive system diagnostics publisher for the central diagnostics node
        self.system_diagnostics_publisher = self.create_publisher(
            String,
            TOPICS["output"]["diagnostics"],  # Use the configured diagnostics topic
            10
        )

    def _init_kalman_filter(self):
        """Initialize the Kalman filter state and variables."""
        # Kalman filter state: [x, y, z, vx, vy, vz]
        # x, y, z = 3D position coordinates
        # vx, vy, vz = 3D velocity components
        self.state = np.zeros(6)
        
        # Kalman filter covariance matrix (6x6)
        # Represents uncertainty in our state estimate
        # Diagonal elements = variance (uncertainty squared) of each state variable
        self.covariance = np.eye(6) * 1000  # Start with high uncertainty
        
        # Flag to track if filter has been initialized
        self.initialized = False
        
        # Last update timestamp (as float seconds)
        self.last_update_time = None
        
        # History of states for analysis and debugging
        self.state_history = deque(maxlen=self.history_length)
        self.time_history = deque(maxlen=self.history_length)
        
        # Sensor statistics for diagnostics
        self.sensor_counts = {
            'hsv_2d': 0, 'yolo_2d': 0, 
            'hsv_3d': 0, 'yolo_3d': 0, 
            'lidar': 0
        }
        self.sensor_last_seen = {
            'hsv_2d': 0, 'yolo_2d': 0, 
            'hsv_3d': 0, 'yolo_3d': 0, 
            'lidar': 0
        }
        
        # Tracking reliability metrics
        self.consecutive_updates = 0
        self.position_uncertainty = float('inf')
        self.velocity_uncertainty = float('inf')
        self.tracking_reliable = False
        
        # Start time for performance tracking
        self.start_time = TimeUtils.now_as_float()  # Use TimeUtils instead of time.time()
        self.updates_processed = 0

        # Add error tracking for diagnostics
        self.errors = []
        self.warnings = []
        self.error_history_size = fusion_config.get('diagnostics', {}).get('error_history_size', 10)
        self.last_error_time = 0
        
        # Add health metrics
        self.filter_health = 1.0  # 0.0 to 1.0 scale
        self.tracking_health = 1.0
        self.sensor_health = 1.0

    def _setup_timers(self):
        """Set up timer callbacks for periodic tasks."""
        # Diagnostic timer (publishes detailed diagnostics every 1 second)
        self.diagnostic_timer = self.create_timer(1.0, self.publish_diagnostics)
        
        # Fast status timer (publishes brief status updates at 10Hz)
        self.status_timer = self.create_timer(0.1, self.publish_status)
        
        # Kalman filter update timer (runs at 20Hz)
        self.filter_timer = self.create_timer(0.05, self.filter_update)

    def _init_debugging_tools(self):
        """Initialize tools for debugging and visualization."""
        # Keep track of timing for performance analysis
        self.processing_times = deque(maxlen=50)
        
        # Store innovation values for diagnostics
        self.innovation_history = deque(maxlen=20)
        
        # Log file for detailed analysis (if enabled)
        self.log_file = None
        if self.log_to_file:
            try:
                self.log_file = open('/tmp/kalman_filter_log.csv', 'w')
                self.log_file.write('time,x,y,z,vx,vy,vz,pos_uncertainty,vel_uncertainty,source\n')
            except Exception as e:
                self.get_logger().error(f"Failed to open log file: {str(e)}")
                self.log_to_file = False

    def _log_topic_connections(self):
        """Log information about topics being subscribed to and published."""
        self.get_logger().info("Topic connections:")
        self.get_logger().info("Subscriptions:")
        for name, topic in TOPICS["input"].items():
            self.get_logger().info(f"  - {name:<10}: {topic}")
        
        self.get_logger().info("Publications:")
        for name, topic in TOPICS["output"].items():
            self.get_logger().info(f"  - {name:<10}: {topic}")

    def log_parameters(self):
        """Log all the current parameter values for reference."""
        self.get_logger().info("=== Kalman Filter Parameters ===")
        self.get_logger().info("Process noise (how quickly uncertainty grows with time):")
        self.get_logger().info(f"  Position noise: {self.process_noise_pos} m/s²")
        self.get_logger().info(f"  Velocity noise: {self.process_noise_vel} m/s²")
        
        self.get_logger().info("Measurement noise (lower = more trust in sensor):")
        self.get_logger().info(f"  HSV 2D: {self.measurement_noise_hsv_2d} pixels")
        self.get_logger().info(f"  YOLO 2D: {self.measurement_noise_yolo_2d} pixels")
        self.get_logger().info(f"  HSV 3D: {self.measurement_noise_hsv_3d} meters")
        self.get_logger().info(f"  YOLO 3D: {self.measurement_noise_yolo_3d} meters")
        self.get_logger().info(f"  LIDAR: {self.measurement_noise_lidar} meters")
        
        self.get_logger().info("Timing and thresholds:")
        self.get_logger().info(f"  Max time difference: {self.max_time_diff} seconds")
        self.get_logger().info(f"  Min confidence threshold: {self.min_confidence_threshold}")
        self.get_logger().info(f"  Detection timeout: {self.detection_timeout} seconds")
        
        self.get_logger().info("Tracking reliability thresholds:")
        self.get_logger().info(f"  Position uncertainty threshold: {self.position_uncertainty_threshold} meters")
        self.get_logger().info(f"  Velocity uncertainty threshold: {self.velocity_uncertainty_threshold} m/s")
        
        self.get_logger().info("Debugging:")
        self.get_logger().info(f"  Debug level: {self.debug_level}")
        self.get_logger().info(f"  History length: {self.history_length} states")
        self.get_logger().info(f"  Log to file: {self.log_to_file}")
        
        self.get_logger().info("===============================")

    def sensor_callback(self, msg, source):
        """
        Process measurements from any sensor and add to synchronization buffer.
        
        This unified callback adds all incoming measurements to the appropriate
        buffer based on their source.
        
        Args:
            msg (PointStamped): The incoming measurement
            source (str): Which sensor it came from ('hsv_2d', 'yolo_2d', etc.)
        """
        # Add measurement to synchronization buffer
        self.sync_buffer.add_measurement(
            sensor_name=source, 
            data=msg, 
            timestamp=msg.header.stamp
        )
        
        # Update sensor statistics
        self.sensor_counts[source] += 1
        self.sensor_last_seen[source] = TimeUtils.now_as_float()  # Use TimeUtils instead of time.time()
        
        # Log incoming data if in debug mode
        if self.debug_level >= 2:
            self.get_logger().debug(
                f"Received {source} measurement with timestamp: " + 
                f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}"
            )

    def filter_update(self, event=None):
        """
        Update the Kalman filter with synchronized measurements.
        
        This is called regularly by our timer and uses the synchronization buffer
        to get measurements from multiple sensors that were taken at approximately
        the same time.
        """
        # For Raspberry Pi optimization: Track execution time
        update_start = TimeUtils.now_as_float()  # Use TimeUtils instead of time.time()
        
        if not self.initialized and not self._try_initialize():
            self.get_logger().debug("Filter not yet initialized, waiting for 3D measurement...")
            return

        # Get synchronized measurements from the buffer
        sync_data = self.sync_buffer.find_synchronized_data()
        
        if not sync_data:
            self.get_logger().debug("No synchronized measurements found")
            return
        
        # Log what sensors we have synchronized data from
        available_sensors = list(sync_data.keys())
        if self.debug_level >= 1:
            self.get_logger().info(f"Found synchronized data from: {', '.join(available_sensors)}")
        
        # Calculate average timestamp of the synchronized measurements
        total_time = 0.0
        count = 0
        for source, msg in sync_data.items():
            ros_time = msg.header.stamp
            timestamp = TimeUtils.ros_time_to_float(ros_time)  # Use TimeUtils for conversion
            total_time += timestamp
            count += 1
        
        if count == 0:
            self.get_logger().warn("Found sync_data but no valid timestamps")
            return
            
        avg_time = total_time / count
        
        # If we have a last update time, calculate dt
        if self.last_update_time is None:
            dt = 0.033  # Default time step of ~30 Hz
            self.get_logger().debug(f"First update, using default dt={dt}")
        else:
            dt = avg_time - self.last_update_time
            self.get_logger().debug(f"Time since last update: {dt:.3f} seconds")
        
        # Handle potential timing issues using TimeUtils
        dt = TimeUtils.handle_time_jump(dt)
        
        # Predict state forward to the current time
        self.predict(dt)
        
        # Transform all measurements to a common coordinate frame before updating
        transformed_data = {}
        
        # Check if transform is available yet (it might take a moment after startup)
        transform_available = False
        if 'lidar' in sync_data:
            transform_available = True
            self.get_logger().debug("Transform is available")
        
        # Try to transform all measurements to a common frame
        for source, msg in sync_data.items():
            transformed_msg = self._transform_point(msg, "map")
            if transformed_msg:
                transformed_data[source] = transformed_msg
            else:
                self.get_logger().warn(f"Could not transform {source} measurement")
        
        # Update with each transformed measurement
        for source, msg in transformed_data.items():
            if source.endswith('_2d'):
                # 2D measurements only have x and y (no depth)
                confidence = msg.point.z  # Usually contains confidence value
                self.update_2d(msg, 
                               getattr(self, f"measurement_noise_{source}"),
                               source, confidence)
            else:
                # 3D measurements have full position
                self.update_3d(msg, 
                               getattr(self, f"measurement_noise_{source}"),
                               source)
        
        # Update last update time
        self.last_update_time = avg_time
        
        # Store state in history
        self.state_history.append(np.copy(self.state))
        self.time_history.append(avg_time)
        
        # Update tracking reliability metrics
        self.update_tracking_reliability()
        
        # Publish the updated state
        self.publish_state()
        
        # Update stats
        self.updates_processed += 1
        
        # Measure execution time for performance monitoring
        execution_time = (TimeUtils.now_as_float() - update_start) * 1000  # milliseconds
        self.processing_times.append(execution_time)
        
        # Automatically adjust filter frequency based on execution time
        # This prevents overloading the Raspberry Pi CPU
        avg_execution = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        
        # If average execution time is more than 80% of our update interval,
        # we might want to slow down to prevent CPU overload
        current_period = getattr(self.filter_timer, 'timer_period_ns', 50000000) / 1e9  # Convert ns to seconds
        if avg_execution > (current_period * 1000 * 0.8):
            new_period = current_period * 1.2  # Increase period by 20%
            self.filter_timer.timer_period_ns = int(new_period * 1e9)
            self.get_logger().warn(
                f"High execution time ({avg_execution:.1f}ms), increasing filter period to {new_period:.3f}s"
            )

    def _try_initialize(self):
        """
        Try to initialize the Kalman filter with the best available 3D measurement.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        # Get the latest data from each sensor
        best_source = None
        best_measurement = None
        
        # First look for 3D sources in this priority order
        for source in ['lidar', 'hsv_3d', 'yolo_3d']:
            latest_data = self.sync_buffer.get_latest_measurement(source)
            if latest_data is not None:
                best_source = source
                best_measurement = latest_data
                break
        
        if best_source and best_measurement:
            # Initialize state with this measurement
            x, y, z = best_measurement.point.x, best_measurement.point.y, best_measurement.point.z
            self.state[0:3] = [x, y, z]  # Position
            self.state[3:6] = [0, 0, 0]  # Initial velocity = 0
            
            # Set last update time
            self.last_update_time = TimeUtils.ros_time_to_float(best_measurement.header.stamp)
            
            self.initialized = True
            self.get_logger().info(
                f"Kalman filter initialized with {best_source} measurement: "
                f"({x:.2f}, {y:.2f}, {z:.2f}) m"
            )
            return True
        
        return False

    def predict(self, dt):
        """
        Predict the state forward in time (time update step of Kalman filter).
        
        This function implements the prediction step of the Kalman filter, which:
        1. Predicts how the ball will move based on physics (constant velocity)
        2. Updates the covariance matrix to reflect increased uncertainty
        
        Args:
            dt (float): Time elapsed since last update in seconds
        """
        if dt <= 0 or dt > 1.0:
            self.get_logger().warn(f"Invalid dt: {dt}, using default")
            dt = 0.033  # Use default time step
        
        # State transition matrix (constant velocity model)
        # This matrix describes how the state evolves over time
        F = np.eye(6)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        
        # Process noise covariance matrix
        # This represents how much uncertainty to add due to imperfections in our model
        Q = np.zeros((6, 6))
        # Position noise (increases with dt^2)
        Q[0:3, 0:3] = np.eye(3) * self.process_noise_pos * dt**2
        # Velocity noise (increases with dt)
        Q[3:6, 3:6] = np.eye(3) * self.process_noise_vel * dt
        
        # Predict state: x = F*x
        # This applies the constant velocity model to move the ball forward in time
        self.state = F @ self.state
        
        # Predict covariance: P = F*P*F' + Q
        # This updates our uncertainty based on the model and process noise
        self.covariance = F @ self.covariance @ F.T + Q
        
        # Update uncertainty metrics
        self.update_tracking_reliability()
        
        # Gradually improve filter health during successful predictions
        if hasattr(self, 'filter_health'):
            self.filter_health = min(1.0, self.filter_health + 0.01)
        
        if self.debug_level >= 2:
            self.get_logger().debug(
                f"Predicted: dt={dt:.3f}s, "
                f"pos=({self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f}) m, "
                f"vel=({self.state[3]:.2f}, {self.state[4]:.2f}, {self.state[5]:.2f}) m/s"
            )

    def update_2d(self, msg, noise_level, source, confidence):
        """
        Update the filter with a 2D measurement (x,y only).
        
        This implements the update step of the Kalman filter for 2D measurements,
        which corrects our predicted state based on new observations.
        
        Args:
            msg (PointStamped): 2D position measurement
            noise_level (float): Base measurement noise level
            source (str): Source of measurement (for logging)
            confidence (float): Confidence value (0-1) to adjust noise
        """
        # Extract position from message
        x_meas = float(msg.point.x)
        y_meas = float(msg.point.y)
        
        # For 2D, we only measure x and y position
        # Measurement model H maps state to expected measurement
        H = np.zeros((2, 6))
        H[0, 0] = 1.0  # x position
        H[1, 1] = 1.0  # y position
        
        # Measurement vector (what we actually observed)
        z = np.array([x_meas, y_meas])
        
        # Expected measurement based on current state
        expected_z = H @ self.state
        
        # Innovation (difference between observation and prediction)
        innovation = z - expected_z
        
        # Scale noise by inverse of confidence (higher confidence = lower noise)
        adjusted_noise = noise_level / (confidence + 0.1)
        
        # Measurement noise covariance matrix
        R = np.eye(2) * adjusted_noise
        
        # Innovation covariance (uncertainty in the innovation)
        S = H @ self.covariance @ H.T + R
        
        # Check if innovation is reasonable using Mahalanobis distance
        # This measures how many standard deviations away the measurement is
        try:
            S_inv = np.linalg.inv(S)
            innovation_magnitude = np.sqrt(innovation.T @ S_inv @ innovation)
            
            # Store for diagnostics
            self.innovation_history.append(innovation_magnitude)
            
            # Skip updates that are too far from prediction (outliers)
            if innovation_magnitude > 10.0:
                self.log_error(
                    f"Rejecting {source} update - too far from prediction "
                    f"({innovation_magnitude:.2f} sigma)",
                    True  # This is a warning, not a critical error
                )
                return
        except np.linalg.LinAlgError:
            self.get_logger().error(f"Error computing innovation covariance inverse for {source}")
            return
        
        # Kalman gain (how much to trust this measurement vs. our prediction)
        K = self.covariance @ H.T @ S_inv
        
        # Update state: x = x + K*y
        self.state = self.state + K @ innovation
        
        # Update covariance: P = (I - K*H)*P
        I = np.eye(6)
        self.covariance = (I - K @ H) @ self.covariance
        
        # Update tracking reliability
        self.consecutive_updates += 1
        self.update_tracking_reliability()
        
        # Update tracking health after successful update
        if hasattr(self, 'tracking_health'):
            self.tracking_health = min(1.0, self.tracking_health + 0.02)
        
        if self.debug_level >= 2:
            self.get_logger().debug(
                f"Updated with {source} 2D: "
                f"measured=({x_meas:.2f}, {y_meas:.2f}), "
                f"innovation=({innovation[0]:.2f}, {innovation[1]:.2f}), "
                f"magnitude={innovation_magnitude:.2f}"
            )

    def update_3d(self, msg, noise_level, source):
        """
        Update the filter with a 3D measurement (x,y,z).
        
        This implements the update step of the Kalman filter for 3D measurements,
        which provides more complete information about the ball's position.
        
        Args:
            msg (PointStamped): 3D position measurement
            noise_level (float): Measurement noise level in meters
            source (str): Source of measurement (for logging)
        """
        # Extract position from message
        x_meas = float(msg.point.x)
        y_meas = float(msg.point.y)
        z_meas = float(msg.point.z)
        
        # For 3D, we measure x, y, and z position
        # Measurement model H maps state to expected measurement
        H = np.zeros((3, 6))
        H[0, 0] = 1.0  # x position
        H[1, 1] = 1.0  # y position
        H[2, 2] = 1.0  # z position
        
        # Measurement vector (what we actually observed)
        z = np.array([x_meas, y_meas, z_meas])
        
        # Expected measurement based on current state
        expected_z = H @ self.state
        
        # Innovation (difference between observation and prediction)
        innovation = z - expected_z
        
        # Measurement noise covariance
        R = np.eye(3) * noise_level
        
        # Innovation covariance (uncertainty in the innovation)
        S = H @ self.covariance @ H.T + R
        
        # Check if innovation is reasonable using Mahalanobis distance
        try:
            S_inv = np.linalg.inv(S)
            innovation_magnitude = np.sqrt(innovation.T @ S_inv @ innovation)
            
            # Store for diagnostics
            self.innovation_history.append(innovation_magnitude)
            
            # Skip updates that are too far from prediction (outliers)
            if innovation_magnitude > 5.0:
                self.log_error(
                    f"Rejecting {source} 3D update - too far from prediction: "
                    f"{innovation_magnitude:.2f} sigma",
                    True
                )
                return
        except np.linalg.LinAlgError:
            self.get_logger().error(f"Error computing innovation covariance inverse for {source}")
            return
        
        # Kalman gain (how much to trust this measurement vs. our prediction)
        K = self.covariance @ H.T @ S_inv
        
        # Update state: x = x + K*y
        self.state = self.state + K @ innovation
        
        # Update covariance: P = (I - K*H)*P
        I = np.eye(6)
        self.covariance = (I - K @ H) @ self.covariance
        
        # Update tracking reliability
        self.consecutive_updates += 1
        self.update_tracking_reliability()
        
        # Update tracking health after successful update
        if hasattr(self, 'tracking_health'):
            self.tracking_health = min(1.0, self.tracking_health + 0.02)
        
        if self.debug_level >= 1:
            self.get_logger().debug(
                f"Updated with {source} 3D: "
                f"measured=({x_meas:.2f}, {y_meas:.2f}, {z_meas:.2f}), "
                f"inno_mag={innovation_magnitude:.2f}"
            )

    def update_tracking_reliability(self):
        """
        Update metrics for tracking reliability.
        
        This calculates and updates metrics that help the state manager decide
        when tracking is reliable versus when to search for the ball.
        
        Key metrics:
        - Position uncertainty: How certain we are about the ball's position
        - Velocity uncertainty: How certain we are about the ball's velocity
        - Fresh sensor data: Whether we've seen recent measurements
        - Consecutive updates: How many successful updates we've had
        """
        # Calculate position and velocity uncertainty from covariance matrix diagonal
        # Taking the square root of the trace divided by dimension gives RMS uncertainty
        self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:3, 0:3]) / 3.0)
        self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[3:6, 3:6]) / 3.0)
        
        # Check if we have fresh data from enough sensors
        current_time = TimeUtils.now_as_float()  # Use TimeUtils instead of time.time()
        
        # Count number of sensors with recent data
        fresh_sensors = 0
        for sensor, last_seen in self.sensor_last_seen.items():
            if (current_time - last_seen) < self.detection_timeout:
                fresh_sensors += 1
        
        # Update sensor health based on fresh sensors
        if hasattr(self, 'sensor_health'):
            target_health = 0.3  # Base health
            if fresh_sensors >= 3:
                target_health = 1.0  # Excellent health with 3+ sensors
            elif fresh_sensors == 2:
                target_health = 0.8  # Good health with 2 sensors
            elif fresh_sensors == 1:
                target_health = 0.5  # Mediocre health with 1 sensor
                
            # Move sensor_health toward target gradually
            if self.sensor_health < target_health:
                self.sensor_health = min(target_health, self.sensor_health + 0.05)
            else:
                self.sensor_health = max(target_health, self.sensor_health - 0.05)
        
        # Determine if tracking is reliable based on multiple criteria
        self.tracking_reliable = (
            # Position uncertainty is below threshold
            self.position_uncertainty < self.position_uncertainty_threshold and
            # Velocity uncertainty is below threshold
            self.velocity_uncertainty < self.velocity_uncertainty_threshold and
            # At least two sensors with recent data
            fresh_sensors >= 2 and
            # At least three consecutive successful updates
            self.consecutive_updates >= 3
        )
        
        # Reset consecutive updates if tracking becomes unreliable after being reliable
        if not self.tracking_reliable and self.consecutive_updates > 10:
            self.consecutive_updates = 0
            self.get_logger().info("Tracking reliability lost - resetting consecutive update counter")

    def publish_state(self, timestamp=None):
        """
        Publish the current state estimate for the PID controller.
        
        Args:
            timestamp (Time, optional): Timestamp to use for the message.
                If None, current time will be used.
        """
        # Use current time if no timestamp provided
        if timestamp is None:
            timestamp = TimeUtils.now_as_ros_time()  # Use TimeUtils instead of self.get_clock().now().to_msg()
            
        # Create and publish position message
        pos_msg = PointStamped()
        pos_msg.header.stamp = timestamp
        pos_msg.header.frame_id = "map"  # Assuming map frame for global coordinates
        
        pos_msg.point.x = float(self.state[0])
        pos_msg.point.y = float(self.state[1])
        pos_msg.point.z = float(self.state[2])
        
        self.position_publisher.publish(pos_msg)
        
        # Create and publish velocity message (useful for PID)
        vel_msg = TwistStamped()
        vel_msg.header.stamp = timestamp
        vel_msg.header.frame_id = "map"
        
        vel_msg.twist.linear.x = float(self.state[3])
        vel_msg.twist.linear.y = float(self.state[4])
        vel_msg.twist.linear.z = float(self.state[5])
        
        self.velocity_publisher.publish(vel_msg)
        
        # Publish position uncertainty
        uncertainty_msg = Float32()
        uncertainty_msg.data = float(self.position_uncertainty)
        self.uncertainty_publisher.publish(uncertainty_msg)

    def publish_status(self):
        """
        Publish a quick status update (high frequency) for state management.
        
        This is called frequently (10Hz) to provide the state management node
        with the current tracking status.
        """
        if not self.initialized:
            return
            
        # Publish tracking status
        status_msg = Bool()
        status_msg.data = self.tracking_reliable
        self.tracking_status_publisher.publish(status_msg)

    def publish_diagnostics(self):
        """
        Publish comprehensive diagnostic information about filter performance
        and tracking reliability.
        
        This is called periodically (1Hz) to provide detailed diagnostic information
        about the filter's performance and current state.
        """
        if not self.initialized:
            self.get_logger().info("Kalman filter not yet initialized")
            return
            
        # Calculate time since last update from each sensor
        current_time = TimeUtils.now_as_float()  # Use TimeUtils instead of time.time()
        sensor_ages = {}
        for sensor, last_seen in self.sensor_last_seen.items():
            if last_seen > 0:
                sensor_ages[sensor] = current_time - last_seen
            else:
                sensor_ages[sensor] = float('inf')
        
        # Calculate velocity magnitude
        vel_mag = math.sqrt(self.state[3]**2 + self.state[4]**2 + self.state[5]**2)
        
        # Log diagnostic information
        self.get_logger().info("=== Kalman Filter Diagnostics ===")
        self.get_logger().info(f"Position: ({self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f}) m")
        self.get_logger().info(
            f"Velocity: ({self.state[3]:.2f}, {self.state[4]:.2f}, {self.state[5]:.2f}) m/s, "
            f"magnitude: {vel_mag:.2f} m/s"
        )
        self.get_logger().info(f"Position uncertainty: {self.position_uncertainty:.3f} m")
        self.get_logger().info(f"Velocity uncertainty: {self.velocity_uncertainty:.3f} m/s")
        self.get_logger().info(f"Tracking reliable: {self.tracking_reliable}")
        self.get_logger().info(f"Consecutive updates: {self.consecutive_updates}")
        
        # Sensor update counts
        self.get_logger().info("Sensor updates:")
        for sensor, count in self.sensor_counts.items():
            self.get_logger().info(f"  - {sensor}: {count} updates, last seen {sensor_ages[sensor]:.1f}s ago")
        
        # Check for sensor issues and log warnings
        recent_sensors = sum(1 for age in sensor_ages.values() if age < 1.0)
        if recent_sensors < 2:
            self.get_logger().warning(
                f"Only {recent_sensors} sensor(s) with recent data - tracking may be unreliable"
            )
        
        # Calculate update rate
        elapsed = current_time - self.start_time
        rate = self.updates_processed / elapsed if elapsed > 0 else 0
        avg_processing = np.mean(self.processing_times) if self.processing_times else 0
        
        self.get_logger().info(f"Update rate: {rate:.1f} Hz, Avg processing time: {avg_processing:.1f} ms")
        
        # Publish structured diagnostics for state manager as a JSON string for parsing
        diag_data = {
            "timestamp": current_time,
            "position": {
                "x": float(self.state[0]),
                "y": float(self.state[1]),
                "z": float(self.state[2]),
                "uncertainty": float(self.position_uncertainty)
            },
            "velocity": {
                "x": float(self.state[3]),
                "y": float(self.state[4]),
                "z": float(self.state[5]),
                "magnitude": float(vel_mag),
                "uncertainty": float(self.velocity_uncertainty)
            },
            "tracking": {
                "reliable": bool(self.tracking_reliable),
                "consecutive_updates": int(self.consecutive_updates)
            },
            "sensors": {
                sensor: {
                    "count": int(count),
                    "age": float(sensor_ages[sensor])
                } for sensor, count in self.sensor_counts.items()
            },
            "performance": {
                "update_rate": float(rate),
                "avg_processing_time_ms": float(avg_processing)
            }
        }
        
        diag_msg = String()
        diag_msg.data = json.dumps(diag_data)
        self.diagnostics_publisher.publish(diag_msg)
        
        # Publish enhanced system-wide diagnostics
        self.publish_system_diagnostics(diag_data, sensor_ages, recent_sensors)

    def publish_system_diagnostics(self, diag_data, sensor_ages, recent_sensors):
        """Publish enhanced diagnostics for the system-wide diagnostics node"""
        # Get system resource information
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=None)
            mem_percent = psutil.virtual_memory().percent
            
            # Get temperature if available (especially on Raspberry Pi)
            temp = None
            if hasattr(psutil, "sensors_temperatures") and callable(getattr(psutil, "sensors_temperatures")):
                temps = psutil.sensors_temperatures()
                if temps and 'cpu_thermal' in temps:
                    temp = temps['cpu_thermal'][0].current
                elif temps and 'coretemp' in temps:
                    temp = temps['coretemp'][0].current
        except ImportError:
            cpu_percent = None
            mem_percent = None
            temp = None
            
        # Identify any errors or warnings
        errors = []
        warnings = []
        
        # Check for sensor issues
        if recent_sensors < 2:
            warnings.append(f"Only {recent_sensors} sensor(s) with recent data - tracking may be unreliable")
        
        # Check for high uncertainties
        if self.position_uncertainty > self.position_uncertainty_threshold * 2:
            warnings.append(f"Very high position uncertainty: {self.position_uncertainty:.3f}m")
        
        # Check for sensor timeouts
        for sensor, age in sensor_ages.items():
            if age > self.detection_timeout * 2:
                warnings.append(f"Sensor {sensor} not seen for {age:.1f}s")
        
        # Calculate innovation statistics if available
        innovation_stats = {}
        if hasattr(self, 'innovation_history') and self.innovation_history:
            innovations = list(self.innovation_history)
            innovation_stats = {
                "mean": float(np.mean(innovations)),
                "max": float(np.max(innovations)),
                "latest": float(innovations[-1]) if innovations else 0.0
            }
            
            # Check for consistently high innovations (could indicate model problems)
            if innovation_stats["mean"] > 2.0:
                warnings.append(f"High average innovation: {innovation_stats['mean']:.2f} sigma")

        # Add any tracked errors and warnings
        current_time = TimeUtils.now_as_float()  # Use TimeUtils instead of time.time()
        for error in self.errors:
            if current_time - error["timestamp"] < 300:  # Last 5 minutes
                errors.append(error["message"])
                
        for warning in self.warnings:
            if current_time - warning["timestamp"] < 300:  # Last 5 minutes
                warnings.append(warning["message"])
        
        # Health recovery over time (errors become less relevant)
        time_since_last_error = current_time - self.last_error_time
        if time_since_last_error > 30.0:  # After 30 seconds with no errors
            self.filter_health = min(1.0, self.filter_health + 0.05)  # Gradually recover
        
        # Calculate overall health (weighted average)
        overall_health = (
            self.filter_health * 0.4 +
            self.tracking_health * 0.4 +
            self.sensor_health * 0.2
        )
        
        # Determine status based on overall health and tracking
        status = "active" if self.tracking_reliable else "searching"
        if errors:
            status = "error"
        elif warnings:
            status = "warning"
        
        # Format diagnostics message to match expected structure
        system_diag_data = {
            "timestamp": TimeUtils.now_as_float(),  # Use TimeUtils instead of time.time()
            "node": "fusion",
            "uptime_seconds": TimeUtils.now_as_float() - self.start_time,  # Use TimeUtils instead of time.time()
            "status": status,
            "health": {
                "filter_health": float(self.filter_health),
                "tracking_health": float(self.tracking_health),
                "sensor_health": float(self.sensor_health),
                "overall": float(overall_health)
            },
            "tracking": {
                "reliable": bool(self.tracking_reliable),
                "consecutive_updates": int(self.consecutive_updates),
                "position": {
                    "x": float(self.state[0]),
                    "y": float(self.state[1]),
                    "z": float(self.state[2]),
                    "uncertainty": float(self.position_uncertainty)
                },
                "velocity": {
                    "x": float(self.state[3]),
                    "y": float(self.state[4]),
                    "z": float(self.state[5]),
                    "uncertainty": float(self.velocity_uncertainty),
                    "magnitude": float(np.linalg.norm(self.state[3:6]))
                }
            },
            "sensors": {
                sensor: {
                    "online": age < self.detection_timeout,
                    "count": int(self.sensor_counts[sensor]),
                    "age_seconds": float(age)
                } for sensor, age in sensor_ages.items()
            },
            "metrics": {
                "update_rate_hz": float(diag_data["performance"]["update_rate"]),
                "processing_time_ms": float(diag_data["performance"]["avg_processing_time_ms"]),
                "recent_sensors": int(recent_sensors),
                "innovation": innovation_stats
            },
            "resources": {
                "cpu_percent": getattr(self.resource_monitor, 'cpu_percent', 0),
                "memory_percent": getattr(self.resource_monitor, 'mem_percent', 0),
                "temperature": getattr(self.resource_monitor, 'temperature', 0)
            },
            "errors": errors,
            "warnings": warnings
        }
        
        # Publish to system-wide diagnostics
        system_diag_msg = String()
        system_diag_msg.data = json.dumps(system_diag_data)
        self.system_diagnostics_publisher.publish(system_diag_msg)
        
        # Log a summary to console
        self.get_logger().info(
            f"Fusion status: {status}, Health: {overall_health:.2f}, "
            f"Tracking: {self.tracking_reliable}, "
            f"Sensors: {recent_sensors}/5"
        )

    def log_filter_state(self):
        """
        Log detailed information about the current filter state.
        
        This provides a snapshot of the current state and uncertainty,
        which is useful for debugging and understanding the filter's behavior.
        """
        self.get_logger().debug("--- Kalman Filter State ---")
        self.get_logger().debug(f"Position: ({self.state[0]:.3f}, {self.state[1]:.3f}, {self.state[2]:.3f}) m")
        self.get_logger().debug(f"Velocity: ({self.state[3]:.3f}, {self.state[4]:.3f}, {self.state[5]:.3f}) m/s")
        
        # Calculate and log standard deviations for each state variable
        pos_std = [math.sqrt(self.covariance[i,i]) for i in range(3)]
        vel_std = [math.sqrt(self.covariance[i+3,i+3]) for i in range(3)]
        
        self.get_logger().debug(
            f"Position std dev: ({pos_std[0]:.3f}, {pos_std[1]:.3f}, {pos_std[2]:.3f}) m"
        )
        self.get_logger().debug(
            f"Velocity std dev: ({vel_std[0]:.3f}, {vel_std[1]:.3f}, {vel_std[2]:.3f}) m/s"
        )
        
        # Log recent innovation magnitudes
        if self.innovation_history:
            recent_innovations = list(self.innovation_history)[-5:]
            self.get_logger().debug(
                f"Recent innovation magnitudes: {[f'{i:.2f}' for i in recent_innovations]}"
            )

    def log_to_file_csv(self, timestamp, source):
        """
        Log detailed state information to a CSV file.
        
        This provides data for offline analysis, visualizations, and debugging.
        
        Args:
            timestamp (float): Current timestamp
            source (str): Source of the update
        """
        if self.log_file:
            try:
                # Format: time,x,y,z,vx,vy,vz,pos_uncertainty,vel_uncertainty,source
                self.log_file.write(
                    f"{timestamp:.6f},"
                    f"{self.state[0]:.6f},{self.state[1]:.6f},{self.state[2]:.6f},"
                    f"{self.state[3]:.6f},{self.state[4]:.6f},{self.state[5]:.6f},"
                    f"{self.position_uncertainty:.6f},{self.velocity_uncertainty:.6f},"
                    f"{source}\n"
                )
                self.log_file.flush()  # Ensure data is written immediately
            except Exception as e:
                self.get_logger().error(f"Error writing to log file: {str(e)}")

    def _transform_point(self, point_msg, target_frame):
        """
        Transform a point from its original frame to the target frame.
        
        Args:
            point_msg (PointStamped): Point to transform
            target_frame (str): Target frame ID
            
        Returns:
            PointStamped: Transformed point or None if transformation failed
        """
        if point_msg.header.frame_id == target_frame:
            return point_msg  # Already in the right frame
            
        try:
            # Wait for transform to be available with shorter timeout now that we've
            # already waited during initialization
            when = rclpy.time.Time()
            self.tf_buffer.can_transform(
                target_frame,
                point_msg.header.frame_id,
                when,
                timeout=rclpy.duration.Duration(seconds=0.05)  # Shorter timeout for runtime checks
            )
            
            # Transform the point
            from geometry_msgs.msg import TransformStamped
            from tf2_geometry_msgs import do_transform_point
            
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                point_msg.header.frame_id,
                when
            )
            
            transformed_point = do_transform_point(point_msg, transform)
            return transformed_point
            
        except Exception as e:
            self.get_logger().warn(
                f"Failed to transform from {point_msg.header.frame_id} to {target_frame}: {str(e)}"
            )
            return None

    def _handle_resource_alert(self, resource_type, value):
        """Handle resource alerts by logging warnings and potentially reducing workload."""
        self.log_error(f"Resource alert: {resource_type.upper()} at {value:.1f}% - performance may be affected", True)
        
        # If CPU usage is critically high, try to reduce workload
        if resource_type == 'cpu' and value > 95.0:
            self.log_error("Critical CPU usage detected - reducing update frequency", True)
            # Adjust filter update frequency if CPU usage is too high
            if hasattr(self, 'filter_timer') and self.filter_timer:
                # Get current period and increase it to reduce CPU load
                current_period = 1.0 / 20.0  # Default 20Hz
                new_period = current_period * 1.5  # Reduce rate by 33%
                self.filter_timer.cancel()
                self.filter_timer = self.create_timer(new_period, self.filter_update)
                self.get_logger().warn(f"Reduced update rate to {1.0/new_period:.1f} Hz")
                
                # Also report this in next system diagnostics
                if not hasattr(self, 'resource_warnings'):
                    self.resource_warnings = []
                self.resource_warnings.append({
                    "timestamp": time.time(),
                    "type": resource_type,
                    "value": value,
                    "action": f"Reduced update rate to {1.0/new_period:.1f} Hz"
                })

    def destroy_node(self):
        """Clean shutdown of the node, stopping all resources."""
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop()
        super().destroy_node()

    def log_error(self, error_message, is_warning=False):
        """Log an error or warning and add it to history for diagnostics."""
        if is_warning:
            self.get_logger().warning(f"FUSION: {error_message}")
            # Add to warning list for diagnostics
            self.warnings.append({
                "timestamp": TimeUtils.now_as_float(),  # Use TimeUtils instead of time.time()
                "message": error_message
            })
        else:
            self.get_logger().error(f"FUSION: {error_message}")
            # Add to error list for diagnostics
            self.errors.append({
                "timestamp": TimeUtils.now_as_float(),  # Use TimeUtils instead of time.time()
                "message": error_message
            })
            
            # Keep error list at appropriate size
            if len(self.errors) > self.error_history_size:
                self.errors.pop(0)
                
            # Update health based on error frequency
            self.last_error_time = TimeUtils.now_as_float()  # Use TimeUtils instead of time.time()
            
            # Reduce filter health score temporarily after an error
            self.filter_health = max(0.3, self.filter_health - 0.2)

def main(args=None):
    """Main function to initialize and run the fusion node."""
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create Kalman filter fusion node
    node = KalmanFilterFusion()
    
    # Welcome message
    print("=================================================")
    print("Tennis Ball Tracking - Kalman Filter Fusion Node")
    print("=================================================")
    print("This node fuses multiple sensor inputs to track a tennis ball in 3D")
    print("")
    print("Subscriptions:")
    for name, topic in TOPICS["input"].items():
        print(f"  - {name:<10}: {topic}")
    print("")
    print("Publications:")
    for name, topic in TOPICS["output"].items():
        print(f"  - {name:<10}: {topic}")
    print("")
    print("Press Ctrl+C to stop")
    print("=================================================")
    
    try:
        import psutil  # For process priority
        
        # On Linux (Raspberry Pi), try to set higher process priority
        try:
            process = psutil.Process(os.getpid())
            process.nice(-10)  # Higher priority (needs appropriate permissions)
            print("Set fusion node to higher process priority")
        except:
            print("Could not increase process priority - requires root permissions")
            
        # Run the node
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Stopping Kalman Filter Fusion Node (Ctrl+C pressed)")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Close log file if open
        if node.log_to_file and node.log_file:
            node.log_file.close()
        
        # Clean shutdown
        node.destroy_node()
        rclpy.shutdown()
        print("Kalman Filter Fusion Node has been shut down.")

if __name__ == '__main__':
    main()
