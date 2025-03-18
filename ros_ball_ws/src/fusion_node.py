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
from std_msgs.msg import String, Float32, Bool
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import math
from collections import deque
import time
import json

# Topic configuration (ensures consistency with other nodes)
TOPICS = {
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
        "diagnostics": "/tennis_ball/fused/diagnostics",
        "uncertainty": "/tennis_ball/fused/position_uncertainty",
        "tracking_status": "/tennis_ball/fused/tracking_status"
    }
}

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
        
        # Set up filter parameters
        self._declare_parameters()
        
        # Set up subscriptions to receive data from all sensors
        self._setup_subscriptions()
        
        # Set up publishers to send out fused data
        self._setup_publishers()
        
        # Initialize Kalman filter state and variables
        self._init_kalman_filter()
        
        # Set up timers for periodic status updates
        self._setup_timers()
        
        # Set up visualization and debugging tools
        self._init_debugging_tools()
        
        self.get_logger().info("Kalman Filter Fusion Node has started!")
        self.log_parameters()
        
        # Log topic connections for debugging
        self._log_topic_connections()

    def _declare_parameters(self):
        """Declare and load all node parameters with descriptive comments."""
        self.declare_parameters(
            namespace='',
            parameters=[
                # Process noise: how much uncertainty to add during prediction steps
                ('process_noise_pos', 0.1),        # Position uncertainty per second squared
                ('process_noise_vel', 1.0),        # Velocity uncertainty per second
                
                # Measurement noise: how much to trust each sensor type
                ('measurement_noise_hsv_2d', 50.0),   # Pixels - high because 2D only
                ('measurement_noise_yolo_2d', 30.0),  # Pixels - lower because more accurate
                ('measurement_noise_hsv_3d', 0.05),  # Meters - from depth camera with HSV
                ('measurement_noise_yolo_3d', 0.04), # Meters - from depth camera with YOLO
                ('measurement_noise_lidar', 0.03),   # Meters - most accurate for 3D
                
                # Filter tuning parameters
                ('max_time_diff', 0.2),           # Maximum time difference for fusion (seconds)
                ('min_confidence_threshold', 0.5), # Minimum confidence threshold for detections
                ('detection_timeout', 0.5),        # Time after which a detection is considered stale
                
                # Tracking reliability thresholds
                ('position_uncertainty_threshold', 0.5),  # Position uncertainty threshold for reliable tracking
                ('velocity_uncertainty_threshold', 1.0),  # Velocity uncertainty threshold for reliable tracking
                
                # Debugging and diagnostics
                ('history_length', 100),           # Number of states to keep in history
                ('debug_level', 1),                # Debug level (0=minimal, 1=normal, 2=verbose)
                ('log_to_file', False)             # Whether to log detailed data to file
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
            lambda msg: self.detection_callback(msg, 'hsv_2d'),
            10
        )
        
        self.yolo_2d_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["yolo_2d"],
            lambda msg: self.detection_callback(msg, 'yolo_2d'),
            10
        )
        
        # 3D detections (from depth camera)
        self.hsv_3d_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["hsv_3d"],
            lambda msg: self.detection_callback(msg, 'hsv_3d'),
            10
        )
        
        self.yolo_3d_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["yolo_3d"],
            lambda msg: self.detection_callback(msg, 'yolo_3d'),
            10
        )
        
        # LIDAR detections (already 3D)
        self.lidar_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["lidar"],
            lambda msg: self.detection_callback(msg, 'lidar'),
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
        self.start_time = time.time()
        self.updates_processed = 0

    def _setup_timers(self):
        """Set up timer callbacks for periodic tasks."""
        # Diagnostic timer (publishes detailed diagnostics every 1 second)
        self.diagnostic_timer = self.create_timer(1.0, self.publish_diagnostics)
        
        # Fast status timer (publishes brief status updates at 10Hz)
        self.status_timer = self.create_timer(0.1, self.publish_status)

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

    def detection_callback(self, msg, source):
        """
        Process detections from any source and update the Kalman filter.
        
        This is the main entry point for all sensor data. It:
        1. Validates the incoming detection
        2. Predicts the state forward to the current time
        3. Updates the filter with the new measurement
        4. Updates tracking reliability metrics
        5. Publishes the updated state
        
        Args:
            msg (PointStamped): Detection message with position and timestamp
            source (str): Source of the detection ('hsv_2d', 'yolo_2d', 'hsv_3d', 'yolo_3d', 'lidar')
        """
        start_time = time.time()
        detection_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        try:
            # Log the detection with source
            if self.debug_level >= 2:
                x, y, z = msg.point.x, msg.point.y, msg.point.z
                self.get_logger().debug(f"Received {source} detection: ({x:.2f}, {y:.2f}, {z:.2f})")
            
            # Update sensor statistics
            self.sensor_counts[source] += 1
            self.sensor_last_seen[source] = time.time()  # Use current time
            
            # Extract confidence information if available
            confidence = 1.0  # Default high confidence
            if source == 'hsv_2d' or source == 'yolo_2d':
                # For 2D detections, z coordinate is often used for confidence
                confidence = msg.point.z
            
            # Skip low confidence detections
            if confidence < self.min_confidence_threshold:
                if self.debug_level >= 1:
                    self.get_logger().debug(
                        f"Skipping low confidence {source} detection: {confidence:.2f}"
                    )
                return
            
            # If not initialized and this is a 3D source, initialize the filter
            if not self.initialized and source in ['hsv_3d', 'yolo_3d', 'lidar']:
                self.initialize_filter(msg, source, detection_time)
                return
                
            # Skip if filter isn't initialized yet
            if not self.initialized:
                self.get_logger().debug(f"Received {source} detection but filter not yet initialized")
                return
                
            # Get time difference since last update
            if self.last_update_time is None:
                dt = 0.033  # Assume ~30Hz if this is the first measurement
            else:
                dt = detection_time - self.last_update_time
                
            # Skip if time went backwards significantly or too far forward
            if dt < -0.1:  # Allow small backward jumps (clock sync issues)
                self.get_logger().warning(
                    f"Time went backwards: {dt:.3f}s. Skipping update."
                )
                return
                
            if dt < 0:  # Small backward time jumps - use small positive value
                dt = 0.001
                
            if dt > 1.0:  # Too much time passed, reset filter
                self.get_logger().warning(
                    f"Too much time since last update ({dt:.3f}s). Resetting filter."
                )
                self.initialized = False
                return
                
            # First predict state forward to current time
            self.predict(dt)
            
            # Then update with the measurement based on source type
            if source == 'hsv_2d':
                self.update_2d(msg, self.measurement_noise_hsv_2d, source, confidence)
            elif source == 'yolo_2d':
                self.update_2d(msg, self.measurement_noise_yolo_2d, source, confidence)
            elif source == 'hsv_3d':
                self.update_3d(msg, self.measurement_noise_hsv_3d, source)
            elif source == 'yolo_3d':
                self.update_3d(msg, self.measurement_noise_yolo_3d, source)
            elif source == 'lidar':
                self.update_3d(msg, self.measurement_noise_lidar, source)
                
            # Update the last update time
            self.last_update_time = detection_time
            
            # Store state in history
            self.state_history.append(np.copy(self.state))
            self.time_history.append(detection_time)
            
            # Update tracking reliability metrics
            self.update_tracking_reliability()
            
            # Publish the updated state
            self.publish_state(msg.header.stamp)
            
            # Update stats
            self.updates_processed += 1
            
            # Log to file if enabled
            if self.log_to_file and self.log_file:
                self.log_to_file_csv(detection_time, source)
                
            # Log filter state periodically
            if (self.debug_level >= 2 or 
                (self.debug_level >= 1 and self.updates_processed % 30 == 0)):
                self.log_filter_state()
                
            # Measure and store processing time
            processing_time = (time.time() - start_time) * 1000  # to ms
            self.processing_times.append(processing_time)
                
        except Exception as e:
            self.get_logger().error(f"Error in {source} detection callback: {str(e)}")

    def initialize_filter(self, msg, source, detection_time):
        """
        Initialize the Kalman filter with the first 3D detection.
        
        This is called when we receive the first 3D measurement and sets up the
        initial state and covariance matrix.
        
        Args:
            msg (PointStamped): First 3D detection
            source (str): Source of the detection ('hsv_3d', 'yolo_3d', or 'lidar')
            detection_time (float): Timestamp of the detection in seconds
        """
        # Initialize state with position and zero velocity
        self.state[0] = msg.point.x  # x position
        self.state[1] = msg.point.y  # y position
        self.state[2] = msg.point.z  # z position
        self.state[3] = 0.0  # x velocity (initially zero)
        self.state[4] = 0.0  # y velocity (initially zero)
        self.state[5] = 0.0  # z velocity (initially zero)
        
        # Initialize covariance - high uncertainty for velocity
        self.covariance = np.diag([0.1, 0.1, 0.1, 10.0, 10.0, 10.0])
        
        # Record time
        self.last_update_time = detection_time
        
        # Mark as initialized
        self.initialized = True
        
        # Reset tracking metrics
        self.consecutive_updates = 1
        self.update_tracking_reliability()
        
        self.get_logger().info(f"Kalman filter initialized with {source} detection")
        self.get_logger().info(
            f"Initial position: ({self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f}) meters"
        )
        
        # Add to history
        self.state_history.append(np.copy(self.state))
        self.time_history.append(detection_time)
        
        # Publish initial state
        self.publish_state(msg.header.stamp)

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
            self.get_logger().warning(f"Invalid dt in predict step: {dt}. Using default.")
            dt = 0.033  # Fallback to ~30Hz
        
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
        
        if self.debug_level >= 2:
            self.get_logger().debug(
                f"Predicted state after {dt:.3f}s: "
                f"position=({self.state[0]:.2f}, {self.state[1]:.2f}, {self.state[2]:.2f}), "
                f"velocity=({self.state[3]:.2f}, {self.state[4]:.2f}, {self.state[5]:.2f})"
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
                self.get_logger().warning(
                    f"Rejecting {source} update - too far from prediction "
                    f"({innovation_magnitude:.2f} sigma)"
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
                self.get_logger().warning(
                    f"Rejecting {source} 3D update - too far from prediction: "
                    f"{innovation_magnitude:.2f} sigma"
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
        current_time = time.time()
        
        # Count number of sensors with recent data
        fresh_sensors = 0
        for sensor, last_seen in self.sensor_last_seen.items():
            if current_time - last_seen < self.detection_timeout:
                fresh_sensors += 1
                
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

    def publish_state(self, timestamp):
        """
        Publish the current state estimate for the PID controller.
        
        This publishes:
        - Current position (x, y, z)
        - Current velocity (vx, vy, vz)
        - Current position uncertainty
        
        Args:
            timestamp (Time): Timestamp to use for the message
        """
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
        current_time = time.time()
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
        # Keep the node running until interrupted
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