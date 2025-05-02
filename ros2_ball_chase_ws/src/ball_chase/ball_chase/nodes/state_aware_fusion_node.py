#!/usr/bin/env python3

"""
Highly Optimized State-Aware Fusion Node for ROS 2
Designed for resource-constrained systems like Raspberry Pi 5
Focuses on efficient state management, reduced CPU usage, and robust tracking

This file implements a sensor fusion node for tracking a basketball using multiple sensors (LiDAR, YOLO 3D, YOLO 2D) in a robotics context.

---

# What is Sensor Fusion?
Sensor fusion is the process of combining data from multiple sensors to get a more accurate, reliable, and robust estimate of the state of an object (like its position and velocity) than any single sensor could provide alone. This is important because each sensor has its own strengths and weaknesses (for example, LiDAR is very accurate but can be blocked, while cameras can see more but are less precise).

# What is a Kalman Filter?
A Kalman filter is a mathematical algorithm that estimates the state of a system (like the position and velocity of a ball) by combining predictions (from physics) and measurements (from sensors), while keeping track of uncertainty. It works in two steps:
1. **Prediction:** Use the previous state and physics to predict where the object should be now.
2. **Update:** Use new sensor measurements to correct the prediction, weighting them by how certain we are about each.

The filter also keeps track of how uncertain it is about its estimate, and this uncertainty changes as new data comes in.

# What is State Management?
State management here means keeping track of whether the ball is moving, stationary, or in between. This is important because we want to treat measurements differently depending on how the ball is behaving (for example, we can trust our estimate more if the ball is stationary).

---

This code is heavily optimized for performance, but the comments and docstrings below will explain the intuition and math behind each part, so a high school student can follow along.
"""
import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.lifecycle import Publisher as LifecyclePublisher
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
import numpy as np
import time
import math
import json
from collections import deque
from functools import lru_cache
from tf2_ros import Buffer, TransformListener, StaticTransformBroadcaster
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped, TwistStamped
from std_msgs.msg import Float32, Bool, String

# Try to import for resource monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class LightweightBuffer:
    """
    LightweightBuffer is a circular buffer (fixed-size queue) for storing recent sensor measurements.
    This is like a small notebook where you always write the newest value on top, and erase the oldest when you run out of space.
    This helps us keep memory usage low and access the most recent data quickly.
    """
    
    def __init__(self, max_size=10):
        """Initialize with fixed buffer size."""
        # Pre-allocate the full buffer with None values
        self.data = [(0.0, None) for _ in range(max_size)]
        self.max_size = max_size
        self.next_index = 0
        self.is_full = False
        self.count = 0
    
    def add(self, timestamp, value):
        """Add value to buffer with fixed memory allocation."""
        self.data[self.next_index] = (timestamp, value)
        self.next_index = (self.next_index + 1) % self.max_size
        if not self.is_full and self.next_index == 0:
            self.is_full = True
        self.count = self.max_size if self.is_full else self.next_index
    
    def get_latest(self):
        """Get the most recent value."""
        if self.count == 0:
            return None
        latest_idx = (self.next_index - 1) % self.max_size
        return self.data[latest_idx][1]
    
    def get_latest_before(self, timestamp, max_age=1.0):
        """Get the most recent value before the given timestamp."""
        if self.count == 0:
            return None
        
        best_time_diff = float('inf')
        best_value = None
        
        # Only iterate through actual data entries
        for i in range(min(self.count, self.max_size)):
            idx = (self.next_index - 1 - i) % self.max_size
            t, value = self.data[idx]
            time_diff = timestamp - t
            if 0 <= time_diff < best_time_diff and time_diff <= max_age:
                best_time_diff = time_diff
                best_value = value
                # Early exit if we find a very recent value
                if time_diff < 0.05:
                    break
        
        return best_value
    
    def get_all_within(self, start_time, end_time):
        """Get all values within a time range."""
        result = []
        # Only iterate through actual data entries
        for i in range(min(self.count, self.max_size)):
            idx = (self.next_index - 1 - i) % self.max_size
            t, v = self.data[idx]
            if start_time <= t <= end_time:
                result.append((t, v))
        return result
    
    def clear(self):
        """Clear the buffer."""
        # Don't reallocate, just reset indicators
        self.next_index = 0
        self.is_full = False
        self.count = 0


class SensorManager:
    """
    SensorManager keeps track of all the sensors, their recent data, and their health (are they working?).
    It uses LightweightBuffer for each sensor to store recent measurements.
    It also tracks how often each sensor is updating, and whether it's currently active.
    """
    
    def __init__(self, sensor_names=None):
        """Initialize with specified sensors."""
        self.sensors = sensor_names or []
        self.data_buffers = {}
        self.last_update_time = {}
        self.update_count = {}
        self.fps_estimates = {}
        
        # Initialize buffers for all sensors - optimize buffer sizes based on importance
        for sensor in self.sensors:
            # LiDAR and YOLO 3D are more important, so they get larger buffers
            buffer_size = 12 if sensor in ['lidar', 'yolo_3d'] else 8
            self.data_buffers[sensor] = LightweightBuffer(buffer_size)
            self.last_update_time[sensor] = 0.0
            self.update_count[sensor] = 0
            self.fps_estimates[sensor] = 0.0
        
        # Track sensor health
        self.sensor_active = {sensor: False for sensor in self.sensors}
        self.sensor_gap_durations = {sensor: 0.0 for sensor in self.sensors}
        self.sensor_update_intervals = {sensor: 0.0 for sensor in self.sensors}
        
        # Tracking for active sensor count to avoid recomputing
        self._active_sensor_count = 0
        self._active_high_quality_sensors = 0
        self._last_health_update = 0.0
    
    def add_measurement(self, sensor, timestamp, data):
        """Add a new measurement for a sensor."""
        if sensor not in self.sensors:
            return
        
        current_time = timestamp
        
        # Update buffer
        self.data_buffers[sensor].add(current_time, data)
        
        # Update statistics
        if self.last_update_time[sensor] > 0:
            interval = current_time - self.last_update_time[sensor]
            # Store interval for tracking
            self.sensor_update_intervals[sensor] = interval
            # Use exponential moving average for FPS estimate
            if self.fps_estimates[sensor] > 0:
                alpha = 0.3  # Smoothing factor
                self.fps_estimates[sensor] = (1 - alpha) * self.fps_estimates[sensor] + alpha * (1.0 / max(interval, 0.001))
            else:
                self.fps_estimates[sensor] = 1.0 / max(interval, 0.001)
        
        self.last_update_time[sensor] = current_time
        self.update_count[sensor] += 1
        
        # Mark sensor as active
        if not self.sensor_active[sensor]:
            self.sensor_active[sensor] = True
            # Force health update to reflect new active sensor
            self._last_health_update = 0.0
        
        self.sensor_gap_durations[sensor] = 0.0
    
    def get_latest(self, sensor):
        """Get the latest measurement for a sensor."""
        if sensor not in self.sensors:
            return None
        return self.data_buffers[sensor].get_latest()
    
    def update_sensor_health(self, current_time, max_gap=1.0):
        """Update sensor health status based on update times."""
        # Only update health every 100ms to reduce CPU usage
        if current_time - self._last_health_update < 0.1:
            return
        
        self._active_sensor_count = 0
        self._active_high_quality_sensors = 0
        
        for sensor in self.sensors:
            gap_duration = current_time - self.last_update_time.get(sensor, 0)
            self.sensor_gap_durations[sensor] = gap_duration
            
            # Mark as inactive if gap exceeds threshold
            sensor_specific_max_gap = max_gap
            # More lenient for 2D sensors
            if sensor.startswith('yolo_2d'):
                sensor_specific_max_gap = max_gap * 1.5
            
            old_active = self.sensor_active[sensor]
            # Mark as inactive if gap exceeds threshold
            self.sensor_active[sensor] = gap_duration <= sensor_specific_max_gap
            
            # Update active sensor counts
            if self.sensor_active[sensor]:
                self._active_sensor_count += 1
                if sensor in ['lidar', 'yolo_3d']:
                    self._active_high_quality_sensors += 1
        
        self._last_health_update = current_time
    
    def get_active_sensor_count(self):
        """Get the count of currently active sensors."""
        return self._active_sensor_count
    
    def get_active_high_quality_sensors(self):
        """Get the count of active high-quality (3D) sensors."""
        return self._active_high_quality_sensors
    
    def get_diagnostic_info(self):
        """Get diagnostic information about sensors."""
        current_time = time.time()
        
        info = {}
        for sensor in self.sensors:
            gap = current_time - self.last_update_time.get(sensor, 0)
            info[sensor] = {
                'active': self.sensor_active[sensor],
                'gap': gap,
                'count': self.update_count[sensor],
                'fps': self.fps_estimates[sensor]
            }
        
        return info


class MotionStateManager:
    """
    MotionStateManager keeps track of the motion state of the ball (is it stationary, moving slowly, or moving fast?).
    This is important because we want to treat measurements differently depending on how the ball is behaving.
    
    ## States:
    - UNKNOWN: We don't know yet.
    - STATIONARY: The ball is not moving.
    - LONG_STATIONARY: The ball has been still for a long time.
    - SMALL_MOVEMENT: The ball is moving a little.
    - MEDIUM_FAST: The ball is moving quickly.

    The state is determined by looking at the velocity (speed) of the ball and using thresholds (cutoff values).
    To avoid switching states too quickly due to noise, we use 'hysteresis' (requiring several measurements before changing state).
    """
    
    # Define motion states as class constants for better performance 
    # (avoids string comparisons)
    UNKNOWN = 0
    STATIONARY = 1
    LONG_STATIONARY = 2
    SMALL_MOVEMENT = 3
    MEDIUM_FAST = 4
    
    # State name mapping for logging
    STATE_NAMES = {
        0: "unknown",
        1: "stationary",
        2: "long_stationary",
        3: "small_movement",
        4: "medium_fast"
    }
    
    def __init__(self):
        """
        Initializes the motion state manager.
        - Sets the current and previous state to UNKNOWN.
        - Sets up confidence values for each state (how sure are we that we're in that state?).
        - Sets up counters for how much evidence we have for each state (used for hysteresis).
        - Sets up thresholds for what counts as stationary or moving.
        - Precomputes a lookup table for mapping velocities to states for fast decision-making.
        """
        # Current state
        self.current_state = self.UNKNOWN
        self.previous_state = self.UNKNOWN
        
        # State confidence (0-1)
        self.state_confidence = {
            self.UNKNOWN: 1.0,
            self.STATIONARY: 0.0,
            self.LONG_STATIONARY: 0.0,
            self.SMALL_MOVEMENT: 0.0,
            self.MEDIUM_FAST: 0.0
        }
        
        # State transition counters
        self.state_evidence = {
            self.STATIONARY: 0,
            self.SMALL_MOVEMENT: 0,
            self.MEDIUM_FAST: 0
        }
        
        # Timing trackers
        self.stationary_start_time = None
        self.last_state_change_time = time.time()
        
        # Speed thresholds (increased to reduce noise sensitivity)
        self.stationary_threshold = 0.05  # m/s
        self.small_movement_threshold = 0.20  # m/s
        
        # Required evidence for state changes (symmetric hysteresis)
        self.evidence_threshold = 3
        
        # Override protection
        self.last_override_time = 0
        self.override_count = 0
        
        # History for debugging - use deque with fixed size
        self.state_history = deque(maxlen=8)
        
        # Pre-compute threshold comparisons for a range of velocities
        self._velocity_state_map = {}
        for v in range(0, 100):  # 0.00 to 0.99 m/s
            vel = v / 100.0
            if vel < self.stationary_threshold:
                self._velocity_state_map[vel] = self.STATIONARY
            elif vel < self.small_movement_threshold:
                self._velocity_state_map[vel] = self.SMALL_MOVEMENT
            else:
                self._velocity_state_map[vel] = self.MEDIUM_FAST
    
    def get_state_name(self, state=None):
        """
        Returns the name of a state (e.g., 'stationary', 'small_movement').
        """
        if state is None:
            state = self.current_state
        return self.STATE_NAMES.get(state, "unknown")
    
    def update(self, velocity, position=None, force_state=None):
        """
        Updates the motion state based on the current velocity.
        
        **How does it work?**
        - Looks up what state the current velocity should correspond to (using thresholds).
        - Uses 'evidence counters' to require several consistent measurements before changing state (hysteresis).
        - Handles special cases like transitioning to LONG_STATIONARY if the ball has been still for a long time.
        - If a state change happens, updates confidence values and logs the change.
        
        **Why do we need this?**
        - Sensors are noisy, so we don't want to switch states just because of a single odd measurement.
        - By requiring several consistent measurements, we make the system more robust.
        
        Returns True if the state changed, False otherwise.
        """
        current_time = time.time()
        state_changed = False
        
        # Store previous state
        old_state = self.current_state
        
        # Handle forced state override
        if force_state is not None:
            if force_state in [self.STATIONARY, self.SMALL_MOVEMENT, self.MEDIUM_FAST]:
                self.previous_state = self.current_state
                self.current_state = force_state
                self.last_state_change_time = current_time
                
                # Reset confidence values
                for state in self.state_confidence:
                    self.state_confidence[state] = 0.2
                self.state_confidence[force_state] = 0.8
                
                # Reset evidence counters
                for state in self.state_evidence:
                    self.state_evidence[state] = 0
                
                self.state_history.append((current_time, force_state, "forced"))
                return True
        
        # Optimize velocity state determination using pre-computed map
        # Round to 2 decimal places for lookup
        lookup_vel = round(min(velocity, 0.99), 2)
        base_state = self._velocity_state_map.get(lookup_vel)
        
        # If not in map (velocity >= 1.0), use MEDIUM_FAST
        if base_state is None:
            base_state = self.MEDIUM_FAST
            
        # Update evidence counters
        for state in self.state_evidence:
            if state == base_state:
                self.state_evidence[state] += 1
            else:
                self.state_evidence[state] = 0
        
        # Apply symmetric hysteresis with evidence thresholds
        if self.current_state == self.STATIONARY:
            # Check for transition to LONG_STATIONARY
            if self.stationary_start_time is None:
                self.stationary_start_time = current_time
            elif current_time - self.stationary_start_time > 5.0:
                self.current_state = self.LONG_STATIONARY
                state_changed = True
            
            # Check for exit from STATIONARY
            elif base_state != self.STATIONARY and self.state_evidence[base_state] >= self.evidence_threshold:
                self.current_state = base_state
                self.stationary_start_time = None
                state_changed = True
        
        elif self.current_state == self.LONG_STATIONARY:
            # Check for exit from LONG_STATIONARY (require more evidence)
            if base_state != self.STATIONARY and self.state_evidence[base_state] >= self.evidence_threshold + 1:
                self.current_state = base_state
                self.stationary_start_time = None
                state_changed = True
        
        else:  # SMALL_MOVEMENT, MEDIUM_FAST, or UNKNOWN
            # Check for transition to STATIONARY
            if base_state == self.STATIONARY and self.state_evidence[self.STATIONARY] >= self.evidence_threshold:
                self.current_state = self.STATIONARY
                self.stationary_start_time = current_time
                state_changed = True
            
            # Check for transition between movement states
            elif (base_state in [self.SMALL_MOVEMENT, self.MEDIUM_FAST] and 
                  base_state != self.current_state and 
                  self.state_evidence[base_state] >= self.evidence_threshold):
                self.current_state = base_state
                state_changed = True
        
        # If in UNKNOWN, transition directly to detected state
        if self.current_state == self.UNKNOWN and self.state_evidence[base_state] >= 1:
            self.current_state = base_state
            if base_state == self.STATIONARY:
                self.stationary_start_time = current_time
            state_changed = True
        
        # Update confidence values
        if state_changed:
            # Update state transition tracking
            self.previous_state = old_state
            self.last_state_change_time = current_time
            
            # Reset/update confidence values
            for state in self.state_confidence:
                if state == self.current_state:
                    self.state_confidence[state] = 0.7
                else:
                    self.state_confidence[state] = max(0.0, self.state_confidence[state] - 0.3)
            
            # Add to history
            self.state_history.append((current_time, self.current_state, "normal"))
        
        return state_changed
    
    def get_validation_multiplier(self):
        """
        Returns a multiplier for how strict we should be when validating new measurements, depending on the current state.
        For example, if the ball is moving fast, we allow bigger changes; if it's stationary, we are more strict.
        """
        # Use direct return based on state for better performance
        if self.current_state == self.STATIONARY:
            return 1.0
        elif self.current_state == self.LONG_STATIONARY:
            return 0.9
        elif self.current_state == self.SMALL_MOVEMENT:
            return 1.3
        elif self.current_state == self.MEDIUM_FAST:
            return 1.5
        else:
            return 1.2
    
    def is_in_transition(self, max_transition_time=1.0):
        """
        Returns True if we recently changed state (within max_transition_time seconds).
        This helps us be more forgiving right after a state change.
        """
        return time.time() - self.last_state_change_time < max_transition_time
    
    def get_state_age(self):
        """
        Returns how long we've been in the current state (in seconds).
        """
        if self.last_state_change_time:
            return time.time() - self.last_state_change_time
        return 0.0


class OptimizedFusionNode(LifecycleNode):
    """
    =============================
    KALMAN FILTERS: THE MATHEMATICAL HEART OF SENSOR FUSION
    =============================
    
    What is a Kalman Filter?
    -----------------------
    A Kalman filter is a mathematical algorithm that estimates the state of a system (like the position and velocity of a ball) by combining:
      - Predictions (from physics: e.g., if the ball is moving, where should it be next?)
      - Measurements (from sensors: e.g., where does LiDAR or a camera say the ball is?)
    It also keeps track of how uncertain it is about its estimate, and updates this uncertainty as new data comes in.
    
    Why use a Kalman Filter?
    -----------------------
    - Sensors are noisy: Each sensor gives slightly different answers, and sometimes they are wrong.
    - Physics is not perfect: The ball might bounce, slow down, or be blocked from view.
    - We want the best possible estimate, using all available information, and we want to know how sure we are about it.
    
    How does it work? (Intuitive Explanation)
    ----------------------------------------
    1. **Prediction Step:**
        - Use the current state (position, velocity) and physics (e.g., velocity, friction) to predict where the ball should be after a small time step.
        - Increase the uncertainty a little, because the future is never perfectly predictable.
    2. **Update Step:**
        - When a new sensor measurement arrives, compare it to the prediction.
        - If the measurement is close to the prediction, trust it more; if it's far, trust it less (maybe it's an outlier).
        - Combine the prediction and measurement, weighted by their uncertainties, to get a new, improved estimate.
        - Reduce the uncertainty, because now we have more information.
    
    Why is this appropriate for basketball tracking?
    -----------------------------------------------
    - The ball moves according to physics, but sensors can lose track or be noisy.
    - We want to smoothly combine all available data, ignore outliers, and know when we're not sure.
    - Kalman filters are optimal (best possible) for linear systems with Gaussian noise, which is a good approximation for this problem.
    
    How does this code use the Kalman filter?
    ----------------------------------------
    - The state is [x, y, vx, vy]: position and velocity in 2D.
    - The prediction step uses physics (velocity, friction) to predict the next state.
    - The update step uses new sensor data (LiDAR, YOLO 3D, YOLO 2D) to correct the prediction.
    - The code adapts the filter based on the ball's motion state (stationary, moving, etc.), and handles uncertainty dynamically.
    - The code is optimized for speed and robustness, but all the core Kalman math is present and explained below.
    """
    
    def __init__(self, node_name='optimized_fusion_node'):
        super().__init__(node_name)
        
        # Create callback groups to manage concurrency
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.subscription_cb_group = ReentrantCallbackGroup()
        
        # Core tracking variables
        self.start_time = time.time()
        self.initialized = False
        self.transform_available = False
        self.tracking_reliable = False
        self._publishers = []
        
        # Reference frame
        self.reference_frame = "base_link"
        
        # Create timers list
        self._timer_list = []
        self.subscribers = []
        
        # Lifecycle flags
        self.is_configured = False
        self.is_activated = False
        
        # Basketball parameters
        self.basketball_radius = 0.1143  # meters (9-inch diameter / 2)
        self.basketball_z_height = self.basketball_radius  # Center height above ground
        
        # Initialize logging helper
        self._last_throttled_logs = {}
        
        # Performance monitoring
        self._last_performance_check = 0.0
        self._cpu_usage_history = deque(maxlen=5)
        self._adaptive_rates = {}
        
        # Cache for frequently computed values
        self._position_vector = np.zeros(2, dtype=np.float32)
        self._velocity_vector = np.zeros(2, dtype=np.float32)
        self._position_vector_previous = np.zeros(2, dtype=np.float32)
        
        # Optimized transformation lookup
        self._transform_cache = {}
        self._transform_cache_ttl = {}  # Infinite TTL for static transforms
        
        self.get_logger().info("Highly Optimized Fusion Node initialized with resource-efficient design")
    
    def on_configure(self, state):
        """
        Called when the node is being configured (set up).
        - Sets up the transform system (for converting between coordinate frames)
        - Loads configuration parameters (like noise values, topic names)
        - Initializes the Kalman filter state and covariance
        - Sets up the motion state manager and sensor manager
        - Precomputes matrices for fast filter updates
        """
        self.get_logger().info("Configuring node...")
        
        try:
            # Initialize transform system - with larger buffer capacity
            self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
            self.tf_listener = TransformListener(self.tf_buffer, self)
            
            # Load configuration
            self.load_configuration()
            
            # Initialize core state tracking - use float32 for better performance on Raspberry Pi
            self.state = np.zeros(4, dtype=np.float32)  # [x, y, vx, vy]
            self.covariance = np.eye(4, dtype=np.float32)
            self.covariance[0:2, 0:2] *= 10.0  # Position uncertainty
            self.covariance[2:4, 2:4] *= 10.0  # Velocity uncertainty
            
            # Initialize uncertainty metrics
            self.position_uncertainty = float('inf')
            self.velocity_uncertainty = float('inf')
            
            # Initialize motion state manager
            self.motion_manager = MotionStateManager()
            
            # Initialize sensor manager with optimized buffers
            self.sensors = ['lidar', 'yolo_3d', 'yolo_2d']
            self.sensor_manager = SensorManager(self.sensors)
            
            # BBOX data storage with fixed memory allocation
            self.bbox_data = {
                'yolo_2d': {'width': 0, 'height': 0, 'timestamp': 0}
            }
            
            # Pre-allocate matrices for filter operations
            self._F = np.eye(4, dtype=np.float32)  # State transition matrix
            self._Q = np.zeros((4, 4), dtype=np.float32)  # Process noise
            self._H_2d = np.zeros((2, 4), dtype=np.float32)  # Measurement matrix for 2D
            self._H_2d[0, 0] = 1.0  # x position
            self._H_2d[1, 1] = 1.0  # y position
            
            # Optimizations for matrix operations
            self._I_2x2 = np.eye(2, dtype=np.float32)  # Identity matrix for 2x2
            self._I_4x4 = np.eye(4, dtype=np.float32)  # Identity matrix for 4x4
            
            # Pre-compute commonly used values
            self._dt_values = {}
            for dt_ms in range(25, 501, 25):  # 25ms to 500ms in 25ms steps
                dt = dt_ms / 1000.0
                self._dt_values[dt_ms] = {
                    'dt': dt,
                    'dt2': dt * dt,
                    'dt3': dt * dt * dt
                }
            
            # Set up adaptive rate control
            self._adaptive_rates = {
                'filter_update': {'base': 0.1, 'current': 0.1, 'min': 0.05, 'max': 0.2},
                'publish_state': {'base': 0.5, 'current': 0.5, 'min': 0.2, 'max': 1.0},
                'publish_status': {'base': 1.0, 'current': 1.0, 'min': 1.0, 'max': 3.0},
                'publish_diagnostics': {'base': 5.0, 'current': 5.0, 'min': 5.0, 'max': 10.0}
            }
            
            self.is_configured = True
            self.get_logger().info("Node configured successfully")
            return TransitionCallbackReturn.SUCCESS
            
        except Exception as e:
            self.get_logger().error(f"Configuration error: {str(e)}")
            return TransitionCallbackReturn.ERROR
    
    def on_activate(self, state):
        """
        Called when the node is activated (starts running).
        - Sets up publishers (for sending out estimated position, velocity, etc.)
        - Sets up subscriptions to sensor topics
        - Sets up timers for periodic tasks (filter update, publishing, diagnostics)
        - Uses staged startup to avoid overloading the CPU at once
        """
        self.get_logger().info("Activating node...")
        
        try:
            # Setup publishers with minimal creation
            self.position_pub = self.create_lifecycle_publisher(
                PointStamped, '/basketball/fused/position', 10)
            self.velocity_pub = self.create_lifecycle_publisher(
                TwistStamped, '/basketball/fused/velocity', 10)
            self.status_pub = self.create_lifecycle_publisher(
                Bool, '/basketball/fused/tracking_status', 10)
            self.uncertainty_pub = self.create_lifecycle_publisher(
                Float32, '/basketball/fused/position_uncertainty', 5)
            self.diagnostics_pub = self.create_lifecycle_publisher(
                String, '/basketball/fusion/diagnostics', 3)
                
            self._publishers = [self.position_pub, self.velocity_pub, 
                               self.status_pub, self.uncertainty_pub, 
                               self.diagnostics_pub]
            
            # Activate publishers
            for pub in self._publishers:
                pub.on_activate(state)
            
            # Setup subscriptions with staggered creation (100ms between each)
            self.setup_lidar_subscription()
            
            # Setup timers with adaptive frequencies
            self._timer_list.append(self.create_timer(
                self._adaptive_rates['filter_update']['current'], 
                self.filter_update, 
                callback_group=self.timer_cb_group
            ))
            self._timer_list.append(self.create_timer(
                self._adaptive_rates['publish_state']['current'], 
                self.publish_state, 
                callback_group=self.timer_cb_group
            ))
            self._timer_list.append(self.create_timer(
                self._adaptive_rates['publish_status']['current'], 
                self.publish_status, 
                callback_group=self.timer_cb_group
            ))
            self._timer_list.append(self.create_timer(
                self._adaptive_rates['publish_diagnostics']['current'], 
                self.publish_diagnostics, 
                callback_group=self.timer_cb_group
            ))
            
            # Delayed setup for other subscriptions
            self._timer_list.append(self.create_timer(
                0.2, self.setup_remaining_subscriptions, callback_group=self.timer_cb_group))
            
            # Cache transforms after a delay
            self._timer_list.append(self.create_timer(
                0.5, self.cache_transforms, callback_group=self.timer_cb_group))
            
            # Add performance monitoring timer
            self._timer_list.append(self.create_timer(
                2.0, self.adjust_processing_rates, callback_group=self.timer_cb_group))
            
            self.is_activated = True
            self.get_logger().info("Node activated with staged startup")
            return TransitionCallbackReturn.SUCCESS
            
        except Exception as e:
            self.get_logger().error(f"Activation error: {str(e)}")
            return TransitionCallbackReturn.ERROR
    
    def on_deactivate(self, state):
        """
        Called when the node is deactivated (stops running).
        - Cleans up publishers, timers, and subscriptions.
        """
        self.get_logger().info("Deactivating node...")
        
        # Deactivate publishers
        for pub in self._publishers:
            pub.on_deactivate()
        
        # Clean up timers
        for timer in self._timer_list:
            self.destroy_timer(timer)
        self._timer_list = []
        
        # Clean up subscriptions
        for sub in self.subscribers:
            self.destroy_subscription(sub)
        self.subscribers = []
        
        self.is_activated = False
        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state):
        """
        Called when the node is being cleaned up (resources released).
        - Releases transform resources and clears caches.
        """
        self.get_logger().info("Cleaning up resources...")
        
        # Release transform resources
        self.tf_listener = None
        self.tf_buffer = None
        
        # Clear caches
        self._transform_cache.clear()
        self._transform_cache_ttl.clear()
        
        self.is_configured = False
        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state):
        """
        Called when the node is shutting down.
        """
        self.get_logger().info("Shutting down...")
        return TransitionCallbackReturn.SUCCESS
    
    def load_configuration(self):
        """
        Loads configuration parameters for the node.
        - Sets up noise values, topic names, thresholds, and performance settings.
        """
        # Process noise
        self.process_noise_pos = 0.1
        self.process_noise_vel = 0.8
        
        # Measurement noise
        self.measurement_noise = {
            'lidar': 0.04,
            'yolo_3d': 0.06,
            'yolo_2d': 0.15
        }
        
        # Validation thresholds with lower values
        self.validation_threshold = {
            'lidar': 15.0,
            'yolo_3d': 20.0,
            'yolo_2d': 30.0
        }
        
        # Topic names with defaults
        self.topics = {
            'lidar': '/basketball/lidar/position',
            'yolo_3d': '/basketball/yolo/position_3d',
            'yolo_2d': '/basketball/yolo/position',
            'yolo_bbox': '/basketball/yolo/bbox'
        }
        
        # Maximum sensor gap durations for processing
        self.max_sensor_gap = {
            'lidar': 1.0,
            'yolo_3d': 1.5,
            'yolo_2d': 1.5
        }
        
        # Uncertainty thresholds
        self.position_uncertainty_threshold = 0.5
        self.velocity_uncertainty_threshold = 1.0
        
        # Camera parameters for 3D estimation
        self.camera_parameters = {
            'focal_length': 345.58,
            'image_width': 320,
            'image_height': 320
        }
        
        # Performance settings - CPU thresholds for adaptive control
        self.performance_thresholds = {
            'high_cpu': 80.0,  # Reduce processing if above this
            'normal_cpu': 65.0,  # Normal operation below this
            'low_cpu': 40.0,    # Increase processing if below this
        }
        
        self.get_logger().info("Configuration loaded with efficient defaults")
    
    def setup_lidar_subscription(self):
        """
        Sets up the subscription to the LiDAR sensor topic.
        LiDAR is prioritized because it is usually the most accurate.
        """
        # LiDAR subscription
        lidar_sub = self.create_subscription(
            PointStamped,
            self.topics['lidar'],
            lambda msg: self.sensor_callback(msg, 'lidar'),
            10,
            callback_group=self.subscription_cb_group
        )
        self.subscribers.append(lidar_sub)
        self.get_logger().info(f"Subscribed to LiDAR: {self.topics['lidar']}")
    
    def setup_remaining_subscriptions(self):
        """
        Sets up subscriptions to the other sensors (YOLO 3D, YOLO 2D, and bounding box data).
        This is done after a short delay to avoid overloading the system at startup.
        """
        # Destroy the timer
        for timer in self._timer_list:
            if timer.callback == self.setup_remaining_subscriptions:
                self.destroy_timer(timer)
                self._timer_list.remove(timer)
                break
        
        # YOLO 3D subscription
        yolo_3d_sub = self.create_subscription(
            PointStamped,
            self.topics['yolo_3d'],
            lambda msg: self.sensor_callback(msg, 'yolo_3d'),
            10,
            callback_group=self.subscription_cb_group
        )
        self.subscribers.append(yolo_3d_sub)
        
        # YOLO 2D subscription
        yolo_2d_sub = self.create_subscription(
            PointStamped,
            self.topics['yolo_2d'],
            lambda msg: self.sensor_callback(msg, 'yolo_2d'),
            10,
            callback_group=self.subscription_cb_group
        )
        self.subscribers.append(yolo_2d_sub)
        
        # YOLO bbox subscription
        from std_msgs.msg import Float32MultiArray
        yolo_bbox_sub = self.create_subscription(
            Float32MultiArray,
            self.topics['yolo_bbox'],
            lambda msg: self.bbox_callback(msg, 'yolo_2d'),
            10,
            callback_group=self.subscription_cb_group
        )
        self.subscribers.append(yolo_bbox_sub)
        
        self.get_logger().info("Remaining subscriptions set up")
    
    def cache_transforms(self):
        """
        Caches coordinate transforms between different sensor frames and the robot's reference frame.
        
        Textbook Explanation:
        ---------------------
        In robotics, each sensor (like a camera or LiDAR) may report positions in its own coordinate system (frame).
        To combine data from multiple sensors, we need to convert all positions to a common frame (the robot's base).
        This process is called a coordinate transformation.
        
        Mathematically, a transformation consists of a rotation (to align axes) and a translation (to shift origins).
        We use a 3D rotation matrix (for orientation) and a translation vector (for position offset).
        
        This function looks up the required transforms using ROS's tf2 system, caches them for fast access, and precomputes rotation matrices for efficient repeated use.
        """
        # Destroy the timer once called (so we don't keep retrying)
        for timer in self._timer_list:
            if (timer.callback == self.cache_transforms):
                self.destroy_timer(timer)
                self._timer_list.remove(timer)
                break
        try:
            # Define important transform pairs (sensor frame, robot base frame)
            transform_pairs = [
                ('ascamera_color_0', self.reference_frame),
                ('lidar_frame', self.reference_frame),
                ('ascamera_camera_link_0', self.reference_frame)
            ]
            # For each pair, look up and cache the transform
            for source, target in transform_pairs:
                # Look up the transform using the tf2 buffer (this gives us translation and rotation)
                transform = self.tf_buffer.lookup_transform(
                    target, source, 
                    rclpy.time.Time()
                )
                # Store in cache with infinite TTL (static hardware, so transforms don't change)
                cache_key = f"{source}_{target}"
                self._transform_cache[cache_key] = transform
                self._transform_cache_ttl[cache_key] = float('inf')
                # Also cache reverse transformation if possible
                rev_key = f"{target}_{source}"
                try:
                    rev_transform = self.tf_buffer.lookup_transform(
                        source, target, 
                        rclpy.time.Time()
                    )
                    self._transform_cache[rev_key] = rev_transform
                    self._transform_cache_ttl[rev_key] = float('inf')
                except Exception:
                    # If reverse lookup fails, don't add to cache
                    pass
                # Log success for educational feedback
                self.get_logger().info(
                    f"Cached transform: {source} → {target}: "
                    f"translation=({transform.transform.translation.x:.3f}, "
                    f"{transform.transform.translation.y:.3f}, "
                    f"{transform.transform.translation.z:.3f})"
                )
            # Pre-compute rotation matrices for fast coordinate transformations
            self._precompute_rotation_matrices()
            self.transform_available = True
            self.get_logger().info("Transform caching completed")
        except Exception as e:
            self.get_logger().error(f"Transform caching error: {str(e)}")
            # Schedule another attempt to cache transforms in 1 second
            self._timer_list.append(self.create_timer(
                1.0, self.cache_transforms, callback_group=self.timer_cb_group))

    def _precompute_rotation_matrices(self):
        """
        Precomputes rotation matrices for fast coordinate transformations.
        
        Textbook Explanation:
        ---------------------
        A rotation in 3D can be represented by a 3x3 matrix. ROS uses quaternions (a 4D vector) to represent orientation because they avoid problems like gimbal lock and are efficient for interpolation.
        However, for fast repeated transformations, we convert quaternions to rotation matrices once and reuse them.
        
        The math:
        - A quaternion (qx, qy, qz, qw) can be converted to a 3x3 rotation matrix using a standard formula.
        - This matrix can then be used to rotate any 3D vector: new_vec = R * old_vec
        - This is much faster than recalculating the matrix every time.
        """
        self._rotation_matrices = {}
        for cache_key, transform in self._transform_cache.items():
            # Extract quaternion components from the transform message
            qx = transform.transform.rotation.x  # x component of quaternion (rotation)
            qy = transform.transform.rotation.y  # y component of quaternion
            qz = transform.transform.rotation.z  # z component of quaternion
            qw = transform.transform.rotation.w  # w (scalar) component of quaternion
            # Convert quaternion to rotation matrix (see math explanation above)
            xx = qx * qx
            xy = qx * qy
            xz = qx * qz
            xw = qx * qw
            yy = qy * qy
            yz = qy * qz
            yw = qy * qw
            zz = qz * qz
            zw = qz * qw
            # Build the 3x3 rotation matrix using the above terms
            rot_mat = np.array([
                [1 - 2 * (yy + zz), 2 * (xy - zw),     2 * (xz + yw)],   # First row
                [2 * (xy + zw),     1 - 2 * (xx + zz), 2 * (yz - xw)],   # Second row
                [2 * (xz - yw),     2 * (yz + xw),     1 - 2 * (xx + yy)]# Third row
            ], dtype=np.float32)
            # Store the matrix for later fast use
            self._rotation_matrices[cache_key] = rot_mat

    def transform_point(self, point_msg, target_frame):
        """
        Transforms a point from the sensor's coordinate frame to the robot's reference frame.
        
        Textbook Explanation:
        ---------------------
        This function takes a point measured in one coordinate system (e.g., the camera's) and converts it to another (the robot's base).
        
        Mathematically, this is done by applying a rotation (to align axes) and a translation (to shift the origin).
        - If the point is already in the target frame, we return it as is.
        - Otherwise, we use the cached transform (rotation + translation) to convert the point.
        - This is a fundamental operation in robotics and computer vision, allowing us to combine data from multiple sensors.
        """
        if not self.transform_available:
            return None
        try:
            # If already in the target frame, no transformation needed
            if point_msg.header.frame_id == target_frame:
                return point_msg
            # Build cache key for this transform
            cache_key = f"{point_msg.header.frame_id}_{target_frame}"
            transform = None
            # Use cached transform if available (fast path)
            if cache_key in self._transform_cache:
                transform = self._transform_cache[cache_key]
            else:
                # If not cached, look up and cache for future use
                try:
                    transform = self.tf_buffer.lookup_transform(
                        target_frame,
                        point_msg.header.frame_id,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.1)
                    )
                    self._transform_cache[cache_key] = transform
                    self._transform_cache_ttl[cache_key] = float('inf')
                except Exception as e:
                    self.throttled_log(
                        f"Transform lookup failed for {point_msg.header.frame_id}->{target_frame}: {str(e)}",
                        key=f"transform_lookup_{cache_key}",
                        min_interval=5.0,
                        level="error"
                    )
                    return None
            # Actually perform the transformation using tf2 helper
            # This will apply both translation and rotation to the point
            return do_transform_point(point_msg, transform)
        except Exception as e:
            self.throttled_log(
                f"Transform error: {str(e)}",
                key="transform_error",
                min_interval=5.0,
                level="error"
            )
            return None
    
    def bbox_callback(self, msg, source):
        """
        Receives bounding box data from YOLO 2D detections.
        Stores the width and height of the detected object for later use in 3D estimation.
        """
        if not self.is_activated:
            return
            
        try:
            if hasattr(msg, 'data') and len(msg.data) >= 4:
                width = msg.data[2]   # width
                height = msg.data[3]  # height
                
                # Store bbox data
                current_time = time.time()
                self.bbox_data[source]['width'] = width
                self.bbox_data[source]['height'] = height
                self.bbox_data[source]['timestamp'] = current_time
                
                # Only log occasionally to reduce overhead
                self.throttled_log(
                    f"Received {source} bbox: {width:.1f}x{height:.1f}",
                    key=f"{source}_bbox",
                    min_interval=2.0
                )
        except Exception as e:
            self.get_logger().error(f"Error in bbox callback: {str(e)}")
    
    def bbox_callback_standard(self, msg, source):
        """
        Receives bounding box data in a different format (BoundingBox2D).
        Stores the width and height for later use.
        """
        if not self.is_activated:
            return
            
        try:
            if hasattr(msg, 'size_x') and hasattr(msg, 'size_y'):
                width = msg.size_x
                height = msg.size_y
                
                # Store bbox data
                self.bbox_data[source]['width'] = width
                self.bbox_data[source]['height'] = height
                self.bbox_data[source]['timestamp'] = time.time()
                
                # Only log occasionally to reduce overhead
                self.throttled_log(
                    f"Received {source} bbox: {width:.1f}x{height:.1f}",
                    key=f"{source}_bbox",
                    min_interval=2.0
                )
        except Exception as e:
            self.get_logger().error(f"Error in bbox callback: {str(e)}")
    
    def sensor_callback(self, msg, source):
        """
        Receives new sensor data (position measurements) from LiDAR, YOLO 3D, or YOLO 2D.
        - Adds the measurement to the sensor manager.
        - If this is the first LiDAR measurement, uses it to initialize the filter.
        - Logs the measurement occasionally for debugging.
        """
        if not self.is_activated:
            return
            
        try:
            # Add to sensor manager
            current_time = time.time()
            self.sensor_manager.add_measurement(source, current_time, msg)
            
            # For LiDAR, attempt early initialization
            if not self.initialized and source == 'lidar':
                transformed = self.transform_point(msg, self.reference_frame)
                if transformed:
                    self.initialize_with_measurement(transformed, source)
            
            # Log one in every 5 sensor updates for key sensors (reduced from 3)
            if source in ['lidar', 'yolo_3d', 'yolo_2d']:
                # Initialize counters if needed
                if not hasattr(self, '_sensor_log_counters'):
                    self._sensor_log_counters = {'lidar': 0, 'yolo_3d': 0, 'yolo_2d': 0}
                
                # Update counter for this sensor
                self._sensor_log_counters[source] = (self._sensor_log_counters[source] + 1) % 5
                
                # Log every fifth update
                if self._sensor_log_counters[source] == 0:
                    # Only transform if we need to log - avoid unnecessary computations
                    transformed = self.transform_point(msg, self.reference_frame)
                    if transformed:
                        # Use both elapsed and ROS time for logs
                        time_prefix = self.get_time_prefix()
                        
                        if source == 'lidar':
                            # Log LiDAR position
                            distance = math.sqrt(transformed.point.x**2 + transformed.point.y**2)
                            direction = math.degrees(math.atan2(transformed.point.y, transformed.point.x))
                            
                            self.get_logger().info(
                                f"{time_prefix}[LIDAR_POS] dist={distance:.2f}m, "
                                f"dir={direction:.1f}°, pos=({transformed.point.x:.2f}, "
                                f"{transformed.point.y:.2f}, {transformed.point.z:.2f})"
                            )
                        
                        elif source == 'yolo_3d':
                            # Log YOLO 3D position
                            distance = math.sqrt(transformed.point.x**2 + transformed.point.y**2)
                            direction = math.degrees(math.atan2(transformed.point.y, transformed.point.x))
                            
                            self.get_logger().info(
                                f"{time_prefix}[YOLO3D_POS] dist={distance:.2f}m, "
                                f"dir={direction:.1f}°, pos=({transformed.point.x:.2f}, "
                                f"{transformed.point.y:.2f}, {transformed.point.z:.2f})"
                            )
                        
                        elif source == 'yolo_2d':
                            # For YOLO 2D, log both the 2D position and bbox if available
                            self.get_logger().info(
                                f"{time_prefix}[YOLO2D_POS] image_pos=({msg.point.x:.1f}, {msg.point.y:.1f})"
                            )
                            
                            # Log bbox if available
                            if 'yolo_2d' in self.bbox_data and self.bbox_data['yolo_2d']['timestamp'] > 0:
                                bbox_width = self.bbox_data['yolo_2d']['width']
                                bbox_height = self.bbox_data['yolo_2d']['height']
                                
                                self.get_logger().info(
                                    f"{time_prefix}[YOLO2D_BBOX] {bbox_width:.1f}x{bbox_height:.1f}"
                                )
                
        except Exception as e:
            self.get_logger().error(f"Error in sensor callback: {str(e)}")
    
    def initialize_with_measurement(self, measurement, source):
        """
        Initializes the Kalman filter state with the first measurement.
        - Sets the position to the measured value.
        - Sets the velocity to zero (since we don't know it yet).
        - Sets the initial uncertainty based on the sensor type.
        """
        # Skip if already initialized
        if self.initialized:
            return False
            
        try:
            # Initialize state with position
            self.state[0] = measurement.point.x  # x
            self.state[1] = measurement.point.y  # y
            self.state[2] = 0.0  # vx
            self.state[3] = 0.0  # vy
            
            # Set initial covariance based on source reliability
            if source == 'lidar':
                pos_variance = 0.05
            else:
                pos_variance = 0.10
                
            # Update covariance
            self.covariance = np.eye(4, dtype=np.float32)
            self.covariance[0:2, 0:2] *= pos_variance
            self.covariance[2:4, 2:4] *= 1.0
            
            # Mark as initialized
            self.initialized = True
            self.last_update_time = time.time()
            
            # Calculate uncertainty metrics
            self.position_uncertainty = math.sqrt(np.trace(self.covariance[0:2, 0:2]) / 2.0)
            self.velocity_uncertainty = math.sqrt(np.trace(self.covariance[2:4, 2:4]) / 2.0)
            
            # Log initialization
            self.get_logger().info(
                f"Filter initialized with {source}: position=({measurement.point.x:.2f}, "
                f"{measurement.point.y:.2f}), uncertainty={self.position_uncertainty:.3f}m"
            )
            
            return True
            
        except Exception as e:
            self.get_logger().error(f"Initialization error: {str(e)}")
            return False
    
    def estimate_3d_from_2d(self, detection_msg, bbox_data):
        """
        Estimates the 3D position of the ball from a 2D camera detection and its bounding box size.
        
        Textbook Explanation:
        ---------------------
        Cameras see the world in 2D, but we want to know where the ball is in 3D space. We can estimate the distance to the ball using the size of its image (bounding box) and the camera's focal length, using the concept of similar triangles.
        
        Math:
        - The real diameter of the ball and its size in the image form similar triangles with the camera's focal length.
        - distance = (real_diameter * focal_length) / image_diameter
        - We then compute the direction from the camera to the ball in 3D, rotate it into the robot's frame, and scale it by the estimated distance.
        - This gives us the 3D position of the ball in the robot's reference frame.
        """
        try:
            current_time = time.time()
            
            # Check bbox freshness
            bbox_age = current_time - bbox_data.get('timestamp', 0)
            max_age = 2.0  # Conservative max age
            
            # Adjust max age based on motion state
            if hasattr(self, 'motion_manager'):
                motion_state = self.motion_manager.current_state
                if motion_state == MotionStateManager.STATIONARY:
                    max_age = 3.0
                elif motion_state == MotionStateManager.LONG_STATIONARY:
                    max_age = 4.0
                elif motion_state == MotionStateManager.MEDIUM_FAST:
                    max_age = 1.5
            
            if bbox_age > max_age:
                self.throttled_log(
                    f"Bbox too old: {bbox_age:.2f}s > {max_age:.1f}s",
                    key="old_bbox",
                    min_interval=1.0
                )
                return None
            
            # Get bbox dimensions
            bbox_width = bbox_data.get('width', 0)
            bbox_height = bbox_data.get('height', 0)
            
            if bbox_width <= 0 or bbox_height <= 0:
                return None
            
            # Basketball diameter
            basketball_diameter = self.basketball_radius * 2
            
            # Calculate ball diameter using geometric mean instead of max dimension
            ball_diameter_pixels = math.sqrt(bbox_width * bbox_height) 

            # Calculate distance
            focal_length = self.camera_parameters['focal_length']
            estimated_distance = (basketball_diameter * focal_length) / ball_diameter_pixels
            
            # Age penalty for older bboxes
            age_factor = min(1.0 + (bbox_age / max_age) * 0.15, 1.15)
            estimated_distance *= age_factor
            
            # Get transform - use cached transform for performance
            cache_key = f"ascamera_color_0_{self.reference_frame}"
            
            if cache_key not in self._transform_cache:
                self.throttled_log(
                    f"Missing camera transform for 3D estimation",
                    key="missing_transform",
                    min_interval=1.0,
                    level="error"
                )
                return None
                
            transform = self._transform_cache[cache_key]
            
            # Camera position
            camera_pos = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            
            # Normalized direction vector calculation - optimized with fewer operations
            image_width = self.camera_parameters['image_width']
            image_height = self.camera_parameters['image_height']
            image_center_x = image_width / 2
            image_center_y = image_height / 2
            
            # Detection coordinates
            detection_x = detection_msg.point.x
            detection_y = detection_msg.point.y
            
            # Offsets from center
            offset_x = detection_x - image_center_x
            offset_y = detection_y - image_center_y
            
            # Camera direction vector
            focal_length = self.camera_parameters['focal_length']
            
            # Pre-normalize for better numerical stability
            # Calculate magnitude first
            magnitude = math.sqrt(offset_x*offset_x + offset_y*offset_y + focal_length*focal_length)
            
            # Create normalized direction vector
            dir_vec = np.array([
                offset_x / magnitude,
                offset_y / magnitude, 
                focal_length / magnitude
            ], dtype=np.float32)
            
            # Use pre-computed rotation matrix if available
            if cache_key in self._rotation_matrices:
                # Apply rotation with optimized matrix multiplication
                rotated_dir = np.dot(self._rotation_matrices[cache_key], dir_vec)
            else:
                # Fallback to quaternion calculation
                # Extract quaternion components from the transform message
                qx = transform.transform.rotation.x  # x component of quaternion (rotation)
                qy = transform.transform.rotation.y  # y component of quaternion
                qz = transform.transform.rotation.z  # z component of quaternion
                qw = transform.transform.rotation.w  # w (scalar) component of quaternion
                # Convert quaternion to rotation matrix (see math explanation above)
                xx = qx * qx
                xy = qx * qy
                xz = qx * qz
                xw = qx * qw
                yy = qy * qy
                yz = qy * qz
                yw = qy * qw
                zz = qz * qz
                zw = qz * qw
                # Build the 3x3 rotation matrix using the above terms
                rot_mat = np.array([
                    [1 - 2 * (yy + zz), 2 * (xy - zw),     2 * (xz + yw)],
                    [2 * (xy + zw),     1 - 2 * (xx + zz), 2 * (yz - xw)],
                    [2 * (xz - yw),     2 * (yz + xw),     1 - 2 * (xx + yy)]
                ], dtype=np.float32)
                # Apply the rotation matrix to the direction vector
                rotated_dir = np.dot(rot_mat, dir_vec)
            
            # Calculate 3D position
            position = [
                camera_pos[0] + estimated_distance * rotated_dir[0],
                camera_pos[1] + estimated_distance * rotated_dir[1],
                self.basketball_z_height  # Fixed z height
            ]
            
            # Create point message
            result = PointStamped()
            result.header.stamp = detection_msg.header.stamp
            result.header.frame_id = self.reference_frame
            result.point.x = float(position[0])
            result.point.y = float(position[1])
            result.point.z = float(position[2])
            
            # Log 3D estimation occasionally
            self.throttled_log(
                f"3D estimation from {detection_msg.header.frame_id}: pos=({position[0]:.2f}, "
                f"{position[1]:.2f}), distance={estimated_distance:.2f}m, bbox={bbox_width:.1f}x{bbox_height:.1f}",
                key="3d_estimation",
                min_interval=1.0
            )
            
            return result
            
        except Exception as e:
            self.throttled_log(
                f"3D estimation error: {str(e)}",
                key="3d_estimation_error",
                min_interval=5.0,
                level="error"
            )
            return None
    
    def process_sensor_data(self):
        """
        Processes sensor data in order of priority (LiDAR first, then YOLO 3D, then YOLO 2D).
        - For each sensor, checks if the data is fresh and valid.
        - Updates the filter state if a good measurement is found.
        - If no good data is found, increases uncertainty (because we're less sure about the ball's position).
        """
        # Update sensor health status
        self.sensor_manager.update_sensor_health(time.time())
        
        # Get motion state to determine processing strategy
        if hasattr(self, 'motion_manager'):
            motion_state = self.motion_manager.current_state
        else:
            motion_state = MotionStateManager.UNKNOWN
        
        # Process sensors in priority order
        processed_any = False
        
        # Always try LiDAR
        if self.process_sensor('lidar'):
            processed_any = True
            
            # If in stationary state, we can skip other sensors to reduce CPU
            if motion_state in [MotionStateManager.STATIONARY, MotionStateManager.LONG_STATIONARY]:
                # For stationary objects with good LiDAR data, we can skip 
                # processing other sensors half the time
                if self.position_uncertainty < 0.3 and time.time() % 2 < 1:
                    return
        
        # Then try YOLO 3D
        if self.process_sensor('yolo_3d'):
            processed_any = True
        
        # Process 2D sensors only if not enough 3D sensors or higher uncertainty
        if (self.sensor_manager.get_active_high_quality_sensors() < 1 or 
                self.position_uncertainty > 0.3):
            # Try YOLO 2D with 3D estimation
            if self.process_2d_sensor('yolo_2d'):
                processed_any = True
        
        # If no sensor processed, increase covariance slightly
        if not processed_any:
            # Covariance growth rate based on motion state
            growth_rate = 1.05
            if motion_state == MotionStateManager.MEDIUM_FAST:
                growth_rate = 1.1
            
            # Apply growth
            self.covariance[0:2, 0:2] *= growth_rate  # Position uncertainty
            self.covariance[2:4, 2:4] *= growth_rate  # Velocity uncertainty
    
    def process_sensor(self, sensor):
        """
        Processes a single 3D sensor measurement.
        - Gets the latest measurement.
        - Checks if it's fresh enough.
        - Transforms it to the reference frame.
        - Validates it (checks if it makes sense given our current estimate).
        - Updates the filter state if valid.
        Returns True if the measurement was used, False otherwise.
        """
        # Get latest measurement
        msg = self.sensor_manager.get_latest(sensor)
        if not msg:
            return False
        
        # Check freshness
        current_time = time.time()
        last_update = self.sensor_manager.last_update_time.get(sensor, 0)
        age = current_time - last_update
        
        # Skip if too old
        max_age = self.max_sensor_gap.get(sensor, 1.0)
        if age > max_age:
            return False
        
        # Transform point
        transformed = self.transform_point(msg, self.reference_frame)
        if not transformed:
            return False
        
        # Validate measurement
        valid, innovation = self.validate_measurement(transformed, sensor)
        if not valid:
            return False
        
        # Update state with valid measurement
        self.update_state_with_measurement(transformed, sensor)
        return True
    
    def process_2d_sensor(self, sensor):
        """
        Processes a 2D sensor measurement by estimating its 3D position.
        - Gets the latest 2D measurement and bounding box.
        - Estimates the 3D position.
        - Validates and updates the filter state if valid.
        Returns True if the measurement was used, False otherwise.
        """
        # Get latest measurement
        msg = self.sensor_manager.get_latest(sensor)
        if not msg:
            return False
        
        # Check freshness
        current_time = time.time()
        last_update = self.sensor_manager.last_update_time.get(sensor, 0)
        age = current_time - last_update
        
        # Skip if too old
        max_age = self.max_sensor_gap.get(sensor, 1.0)
        if age > max_age:
            return False
        
        # Check if we have bbox data
        if sensor not in self.bbox_data or self.bbox_data[sensor]['timestamp'] == 0:
            return False
        
        # Estimate 3D from 2D
        estimated_3d = self.estimate_3d_from_2d(msg, self.bbox_data[sensor])
        if not estimated_3d:
            return False
        
        # Validate measurement
        valid, innovation = self.validate_measurement(estimated_3d, f"{sensor}_est3d")
        if not valid:
            return False
        
        # Update state with valid measurement
        self.update_state_with_measurement(estimated_3d, f"{sensor}_est3d")
        return True
    
    def validate_measurement(self, measurement, source):
        """
        Checks if a new measurement is reasonable given our current estimate.
        
        Textbook Explanation:
        ---------------------
        In sensor fusion, not all measurements are trustworthy. We use the Mahalanobis distance to check if a new measurement is consistent with our current estimate, considering both our uncertainty and the sensor's noise.
        
        Math:
        - Innovation y = measurement - prediction
        - Innovation covariance S = prediction_uncertainty + measurement_noise
        - Mahalanobis distance = sqrt(y^T S^-1 y)
        - If the distance is too large, the measurement is likely an outlier and is rejected.
        - The threshold for acceptance is adapted based on how fast the ball is moving (stricter for stationary, looser for fast movement).
        """
        # Prepare measurement vector
        z = np.array([measurement.point.x, measurement.point.y], dtype=np.float32)
        
        # Calculate innovation - use direct array subtraction for better performance
        y = z - self.state[0:2]
        
        # Create measurement noise matrix
        R = np.eye(2, dtype=np.float32) * self.measurement_noise.get(source, 0.1)
        
        # Calculate innovation covariance S = H*P*H' + R
        # For 2D position measurement, this simplifies to S = P[0:2,0:2] + R
        S = self.covariance[0:2, 0:2] + R
        
        try:
            # Calculate squared Mahalanobis distance more efficiently
            # For 2x2 matrices, we can compute the inverse directly
            det_S = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
            
            # Avoid division by zero
            if abs(det_S) < 1e-10:
                return False, float('inf')
                
            # Calculate inverse manually for 2x2 matrix
            S_inv = np.array([
                [S[1, 1] / det_S, -S[0, 1] / det_S],
                [-S[1, 0] / det_S, S[0, 0] / det_S]
            ], dtype=np.float32)
            
            # Compute innovation score efficiently
            innovation_value = y[0] * (S_inv[0, 0] * y[0] + S_inv[0, 1] * y[1]) + \
                              y[1] * (S_inv[1, 0] * y[0] + S_inv[1, 1] * y[1])
            
            # Take square root for actual Mahalanobis distance
            innovation_sqrt = math.sqrt(innovation_value)
            
            # Get validation threshold
            base_threshold = self.validation_threshold.get(source, 20.0)
            
            # Apply motion state multiplier
            if hasattr(self, 'motion_manager'):
                multiplier = self.motion_manager.get_validation_multiplier()
                threshold = base_threshold * multiplier
                
                # Apply transition boost if in transition
                if self.motion_manager.is_in_transition():
                    # Apply stronger boost for transitions from stationary
                    if self.motion_manager.previous_state in [
                            MotionStateManager.STATIONARY, 
                            MotionStateManager.LONG_STATIONARY]:
                        threshold *= 1.5
                    else:
                        threshold *= 1.2
            else:
                threshold = base_threshold
            
            # Cap threshold
            threshold = min(threshold, 50.0)
            
            # Check if measurement passes validation
            valid = innovation_sqrt <= threshold
            
            # Log validation occasionally
            if not valid and source in ['lidar', 'yolo_3d']:
                self.throttled_log(
                    f"Rejected {source}: innovation={innovation_sqrt:.2f} > threshold={threshold:.2f}",
                    key=f"{source}_reject",
                    min_interval=1.0
                )
            
            return valid, innovation_sqrt
            
        except Exception as e:
            self.throttled_log(
                f"Validation error for {source}: {str(e)}",
                key="validation_error",
                min_interval=5.0,
                level="error"
            )
            return False, float('inf')
    
    def update_state_with_measurement(self, measurement, source):
        """
        Updates the filter state with a new, validated measurement.
        
        Textbook Explanation:
        ---------------------
        This is the core of the Kalman filter's update step. We combine our prediction and the new measurement, weighted by their uncertainties, to get a new, improved estimate.
        
        Math:
        - Kalman gain K = P S^-1 (how much to trust the measurement vs. prediction)
        - New state = old state + K * (measurement - prediction)
        - New uncertainty = (I - K H) * old uncertainty
        - For stationary states and small changes, we use a simple weighted average (blending) for efficiency and stability.
        - For moving states or large changes, we use the full Kalman update equations.
        """
        # Prepare measurement
        z = np.array([measurement.point.x, measurement.point.y], dtype=np.float32)
        
        # Store original state for position change calculation and revert if needed
        self._position_vector_previous[0] = self.state[0]
        self._position_vector_previous[1] = self.state[1]
        old_velocity = self.state[2:4].copy()
        
        # Setup measurement noise
        R = np.eye(2, dtype=np.float32) * self.measurement_noise.get(source, 0.1)
        
        # Calculate innovation
        y = z - self.state[0:2]
        
        # Get motion state for context
        motion_state = self.motion_manager.current_state if hasattr(self, 'motion_manager') else MotionStateManager.UNKNOWN
        
        # For stationary states with small innovations, use simplified update
        if motion_state in [MotionStateManager.STATIONARY, MotionStateManager.LONG_STATIONARY] and \
           np.linalg.norm(y) < 0.1:
            # Calculate blend factor based on source
            blend_factor = 0.0
            if source == 'lidar':
                blend_factor = 0.5  # Reduced from 0.8 for stationary
            elif source.startswith('yolo_3d'):
                blend_factor = 0.4  # Reduced from 0.6 for stationary
            elif source.endswith('_est3d'):
                blend_factor = 0.3  # Reduced from 0.5 for stationary
            else:
                blend_factor = 0.2  # Reduced from 0.4 for stationary
            
            # Simple weighted average for position
            self.state[0] = (1.0 - blend_factor) * self.state[0] + blend_factor * z[0]
            self.state[1] = (1.0 - blend_factor) * self.state[1] + blend_factor * z[1]
            
            # Slightly reduce velocity towards zero for stationary
            self.state[2] *= 0.95
            self.state[3] *= 0.95
            
            # Simplified covariance update - just scale position covariance
            pos_scale = (1.0 - blend_factor) ** 2
            self.covariance[0:2, 0:2] *= pos_scale
            
            return True
        
        # For moving objects or larger innovations, use optimized Kalman update
        try:
            # Calculate innovation covariance S = P[0:2,0:2] + R
            S = self.covariance[0:2, 0:2] + R
            
            # Calculate determinant of S
            det_S = S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0]
            
            # Avoid division by zero
            if abs(det_S) < 1e-10:
                return False
                
            # Calculate inverse manually for 2x2 matrix - faster than np.linalg.inv
            inv_det_S = 1.0 / det_S
            S_inv = np.array([
                [S[1, 1] * inv_det_S, -S[0, 1] * inv_det_S],
                [-S[1, 0] * inv_det_S, S[0, 0] * inv_det_S]
            ], dtype=np.float32)
            
            # Calculate Kalman gain directly - avoiding full matrix multiplication
            # K_top = P[0:2,0:2] * S_inv
            K_top = np.array([
                [self.covariance[0, 0] * S_inv[0, 0] + self.covariance[0, 1] * S_inv[1, 0],
                 self.covariance[0, 0] * S_inv[0, 1] + self.covariance[0, 1] * S_inv[1, 1]],
                [self.covariance[1, 0] * S_inv[0, 0] + self.covariance[1, 1] * S_inv[1, 0],
                 self.covariance[1, 0] * S_inv[0, 1] + self.covariance[1, 1] * S_inv[1, 1]]
            ], dtype=np.float32)
            
            # K_bottom = P[2:4,0:2] * S_inv
            K_bottom = np.array([
                [self.covariance[2, 0] * S_inv[0, 0] + self.covariance[2, 1] * S_inv[1, 0],
                 self.covariance[2, 0] * S_inv[0, 1] + self.covariance[2, 1] * S_inv[1, 1]],
                [self.covariance[3, 0] * S_inv[0, 0] + self.covariance[3, 1] * S_inv[1, 0],
                 self.covariance[3, 0] * S_inv[0, 1] + self.covariance[3, 1] * S_inv[1, 1]]
            ], dtype=np.float32)
            
            # Calculate blend factor based on source
            blend_factor = 0.0
            if source == 'lidar':
                blend_factor = 0.8  # High trust for LiDAR
            elif source.startswith('yolo_3d'):
                blend_factor = 0.6  # Medium trust for YOLO 3D
            elif source.endswith('_est3d'):
                blend_factor = 0.5  # Medium-low trust for 2D-derived 3D
            else:
                blend_factor = 0.4  # Low trust for others
            
            # Apply different update strategy based on motion state
            if motion_state in [MotionStateManager.STATIONARY, MotionStateManager.LONG_STATIONARY]:
                # For stationary objects, use more conservative blend
                pos_discrepancy = np.linalg.norm(z - self.state[0:2])
                
                if pos_discrepancy > 0.1:
                    # Significant discrepancy during stationary - likely valid movement
                    # Apply stronger blend to allow escaping stationary state
                    blend_factor = min(blend_factor + 0.2, 0.9)
                    
                    self.throttled_log(
                        f"Detected movement during {self.motion_manager.get_state_name()}: {pos_discrepancy:.3f}m, "
                        f"blend={blend_factor:.2f}",
                        key="stationary_movement",
                        min_interval=0.5
                    )
                    
                    # Update position with strong blend
                    self.state[0] = (1.0 - blend_factor) * self.state[0] + blend_factor * z[0]
                    self.state[1] = (1.0 - blend_factor) * self.state[1] + blend_factor * z[1]
                    
                    # Calculate implied velocity
                    implied_vx = (z[0] - self._position_vector_previous[0]) / 0.1  # Assume recent measurement
                    implied_vy = (z[1] - self._position_vector_previous[1]) / 0.1
                    
                    # Update velocity with reduced blend
                    vel_blend = blend_factor * 0.5
                    self.state[2] = (1.0 - vel_blend) * self.state[2] + vel_blend * implied_vx
                    self.state[3] = (1.0 - vel_blend) * self.state[3] + vel_blend * implied_vy
                    
                else:
                    # Normal update for stationary with small discrepancy
                    # Use direct update with Kalman gain
                    self.state[0] += K_top[0, 0] * y[0] + K_top[0, 1] * y[1]
                    self.state[1] += K_top[1, 0] * y[0] + K_top[1, 1] * y[1]
                    self.state[2] += K_bottom[0, 0] * y[0] + K_bottom[0, 1] * y[1]
                    self.state[3] += K_bottom[1, 0] * y[0] + K_bottom[1, 1] * y[1]
            
            elif motion_state == MotionStateManager.MEDIUM_FAST:
                # For fast-moving objects, trust measurements more
                # Especially from LiDAR and YOLO 3D
                if source in ['lidar', 'yolo_3d']:
                    blend_factor = min(blend_factor + 0.1, 0.9)
                
                # Update position with blend
                self.state[0] = (1.0 - blend_factor) * self.state[0] + blend_factor * z[0]
                self.state[1] = (1.0 - blend_factor) * self.state[1] + blend_factor * z[1]
                
                # Update velocity with partial Kalman
                self.state[2] += K_bottom[0, 0] * y[0] + K_bottom[0, 1] * y[1]
                self.state[3] += K_bottom[1, 0] * y[0] + K_bottom[1, 1] * y[1]
                
            else:  # SMALL_MOVEMENT or UNKNOWN
                # Standard Kalman update with direct computation (avoiding matrix multiplications)
                self.state[0] += K_top[0, 0] * y[0] + K_top[0, 1] * y[1]
                self.state[1] += K_top[1, 0] * y[0] + K_top[1, 1] * y[1]
                self.state[2] += K_bottom[0, 0] * y[0] + K_bottom[0, 1] * y[1]
                self.state[3] += K_bottom[1, 0] * y[0] + K_bottom[1, 1] * y[1]
            
            # Calculate position change for logging
            pos_change = np.linalg.norm(self.state[0:2] - self._position_vector_previous)
            if pos_change > 0.1:
                self.throttled_log(
                    f"Position updated: {pos_change:.3f}m from {source}",
                    key="position_update",
                    min_interval=0.5
                )
            
            # Optimized Joseph form covariance update
            # Note: We skip the full Joseph form update (P = (I-KH)P(I-KH)^T + KRK^T) which is expensive
            # Instead, we use a simplified form that works well in practice and is much faster
            
            # 1. Build the I-KH matrix
            I_KH = np.eye(4, dtype=np.float32)
            I_KH[0, 0] -= K_top[0, 0]
            I_KH[0, 1] -= K_top[0, 1]
            I_KH[1, 0] -= K_top[1, 0]
            I_KH[1, 1] -= K_top[1, 1]
            
            # 2. Calculate P = (I-KH)P - the main covariance reduction
            # Only calculate the upper triangular part and then mirror for symmetry
            # This takes advantage of the structure of I-KH to reduce computations
            P_new = np.zeros_like(self.covariance)
            
            # Update top-left 2x2 block (position covariance)
            for i in range(2):
                for j in range(2):
                    for k in range(4):
                        P_new[i, j] += I_KH[i, k] * self.covariance[k, j]
            
            # Update top-right 2x2 block (position-velocity cross-covariance)
            for i in range(2):
                for j in range(2, 4):
                    for k in range(4):
                        P_new[i, j] += I_KH[i, k] * self.covariance[k, j]
            
            # Update bottom-left 2x2 block (velocity-position cross-covariance)
            for i in range(2, 4):
                for j in range(2):
                    for k in range(4):
                        P_new[i, j] += I_KH[i, k] * self.covariance[k, j]
            
            # Update bottom-right 2x2 block (velocity covariance)
            for i in range(2, 4):
                for j in range(2, 4):
                    for k in range(4):
                        P_new[i, j] += I_KH[i, k] * self.covariance[k, j]
            
            # 3. Ensure symmetry by averaging with the transpose
            for i in range(4):
                for j in range(i):
                    P_new[i, j] = P_new[j, i]
            
            # 4. Add small stabilizing term to ensure positive definiteness
            for i in range(4):
                P_new[i, i] += 1e-5
            
            # Update the covariance matrix
            self.covariance = P_new
            
            return True
            
        except Exception as e:
            self.throttled_log(
                f"Kalman update failed for {source}: {str(e)}",
                key="kalman_error",
                min_interval=5.0,
                level="error"
            )
            
            # Revert state changes to avoid instability
            self.state[0] = self._position_vector_previous[0]
            self.state[1] = self._position_vector_previous[1]
            self.state[2:4] = old_velocity
            
            return False
    
    def update_motion_state(self):
        """
        Updates the motion state (stationary, moving, etc.) based on the current velocity.
        
        Textbook Explanation:
        ---------------------
        The ball's behavior changes how we should treat measurements. This function uses the current velocity to update the motion state (stationary, moving, etc.), which in turn affects how the filter processes new data and applies constraints.
        """
        if not hasattr(self, 'motion_manager'):
            return
        
        # Calculate velocity magnitude
        velocity = math.sqrt(self.state[2]**2 + self.state[3]**2)
        
        # Update motion state
        state_changed = self.motion_manager.update(velocity)
        
        # Log state changes with dual-time format
        if state_changed:
            # Get time prefix with both elapsed and ROS time
            time_prefix = self.get_time_prefix()
            
            state_age = 0
            if hasattr(self.motion_manager, 'previous_state'):
                state_age = self.motion_manager.get_state_age()
                
            self.get_logger().info(
                f"{time_prefix}[STATE] {self.motion_manager.get_state_name(self.motion_manager.previous_state)} → "
                f"{self.motion_manager.get_state_name()} (v={velocity:.3f}m/s, age={state_age:.1f}s)"
            )
    
    def apply_physics_constraints(self):
        """
        Applies physical constraints to the estimated state.
        
        Textbook Explanation:
        ---------------------
        The real world has limits: the ball can't move infinitely fast, and friction slows it down. This function applies speed limits and friction to keep the estimate realistic. This is a key part of making the filter robust to sensor dropouts and noise.
        """
        # Get motion state
        motion_state = self.motion_manager.current_state if hasattr(self, 'motion_manager') else MotionStateManager.UNKNOWN
        
        # Apply speed limit based on motion state using direct comparison
        vx, vy = self.state[2], self.state[3]
        v_squared = vx*vx + vy*vy
        
        # Define maximum speed based on state - squared for efficient comparison
        max_speed_squared = 25.0  # 5.0^2 Default
        
        if motion_state == MotionStateManager.STATIONARY:
            max_speed_squared = 0.01  # 0.1^2
        elif motion_state == MotionStateManager.LONG_STATIONARY:
            max_speed_squared = 0.0025  # 0.05^2
        elif motion_state == MotionStateManager.SMALL_MOVEMENT:
            max_speed_squared = 4.0  # 2.0^2
        
        # Apply speed limit if exceeded - avoid sqrt when possible
        if v_squared > max_speed_squared:
            scale = math.sqrt(max_speed_squared / v_squared)
            self.state[2] *= scale
            self.state[3] *= scale
        
        # Apply friction based on motion state for smoother deceleration
        if motion_state != MotionStateManager.MEDIUM_FAST and v_squared > 0.0001:  # Only if moving
            # Friction coefficients based on motion state
            friction_coef = 0.015  # Baseline friction
            
            if motion_state == MotionStateManager.STATIONARY:
                friction_coef = 0.03
            elif motion_state == MotionStateManager.LONG_STATIONARY:
                friction_coef = 0.04
            
            # Calculate deceleration (μg)
            deceleration = friction_coef * 9.81
            
            # Calculate current speed only when needed
            speed = math.sqrt(v_squared)
            
            # Calculate velocity reduction
            dv = min(speed, deceleration * 0.1)  # Assuming 10Hz update rate
            
            # Apply proportional deceleration if significant
            if dv > 0.001:
                factor = 1.0 - (dv / speed)
                self.state[2] *= factor
                self.state[3] *= factor
    
    def update_tracking_status(self):
        """
        Updates the tracking status (are we tracking the ball reliably?).
        
        Textbook Explanation:
        ---------------------
        We want to know if our estimate is reliable enough to use. We require a minimum number of active sensors and low enough uncertainty. Hysteresis means we are more strict about losing tracking than gaining it, to avoid rapid flipping. We publish the status and log changes for monitoring.
        """
        # Get active sensor counts - already calculated in sensor_manager
        active_3d = self.sensor_manager.get_active_high_quality_sensors()
        active_2d = self.sensor_manager._active_sensor_count - active_3d
        
        # Calculate sensor requirement based on motion state
        if hasattr(self, 'motion_manager'):
            motion_state = self.motion_manager.current_state
            
            # More lenient requirements for stationary states
            if motion_state in [MotionStateManager.STATIONARY, MotionStateManager.LONG_STATIONARY]:
                min_3d_sensors = 0
                min_2d_sensors = 1
            else:
                min_3d_sensors = 1
                min_2d_sensors = 0
        else:
            min_3d_sensors = 1
            min_2d_sensors = 0
            
        # Check uncertainty thresholds
        pos_uncertainty_ok = self.position_uncertainty < self.position_uncertainty_threshold
        vel_uncertainty_ok = self.velocity_uncertainty < self.velocity_uncertainty_threshold
        
        # Determine if tracking is reliable
        sensors_ok = (active_3d >= min_3d_sensors) or (active_2d >= min_2d_sensors)
        uncertainty_ok = pos_uncertainty_ok and vel_uncertainty_ok
        
        # Optimize previous tracking state detection for hysteresis
        if not self.tracking_reliable:
            # More lenient to start tracking
            new_tracking_reliable = sensors_ok and uncertainty_ok
        else:
            # More strict to lose tracking (need to lose both)
            new_tracking_reliable = sensors_ok or uncertainty_ok
        
        # Only update if changed to avoid redundant publishing
        if new_tracking_reliable != self.tracking_reliable:
            self.tracking_reliable = new_tracking_reliable
            
            # Publish status change immediately
            status_msg = Bool()
            status_msg.data = self.tracking_reliable
            self.status_pub.publish(status_msg)
            
            # Log status change
            self.get_logger().info(
                f"{self.get_time_prefix()}[TRACKING] {'Started' if self.tracking_reliable else 'Lost'} "
                f"tracking: 3D={active_3d}, 2D={active_2d}, PosUnc={self.position_uncertainty:.3f}m"
            )
        
        # Publish uncertainty at reduced frequency
        uncertainty_msg = Float32()
        uncertainty_msg.data = self.position_uncertainty
        self.uncertainty_pub.publish(uncertainty_msg)
    
    def publish_state(self):
        """
        Publishes the current estimated position and velocity of the ball.
        - Also logs the state occasionally for debugging.
        """
        if not self.is_activated or not self.initialized:
            return
            
        try:
            # Create position message
            pos_msg = PointStamped()
            pos_msg.header.stamp = self.get_clock().now().to_msg()
            pos_msg.header.frame_id = self.reference_frame
            pos_msg.point.x = float(self.state[0])
            pos_msg.point.y = float(self.state[1])
            pos_msg.point.z = float(self.basketball_z_height)  # Fixed height
            
            # Publish position
            self.position_pub.publish(pos_msg)
            
            # Create velocity message
            vel_msg = TwistStamped()
            vel_msg.header.stamp = self.get_clock().now().to_msg()
            vel_msg.header.frame_id = self.reference_frame
            vel_msg.twist.linear.x = float(self.state[2])
            vel_msg.twist.linear.y = float(self.state[3])
            vel_msg.twist.linear.z = 0.0  # Zero vertical velocity
            
            # Publish velocity
            self.velocity_pub.publish(vel_msg)
            
            # Get motion state
            motion_state = "unknown"
            if hasattr(self, 'motion_manager'):
                motion_state = self.motion_manager.get_state_name()
            
            # Calculate required metrics for logging - only when needed
            # Reduce computations with adaptive logging frequency
            current_time = time.time()
            log_key = "fusion_output"
            last_log_time = self._last_throttled_logs.get(log_key, 0)
            log_interval = 1.0  # Default 1 second
            
            # Adjust log interval based on motion state
            if motion_state in ["stationary", "long_stationary"]:
                log_interval = 2.0  # Log less frequently for stationary objects
            
            # Only log if enough time has passed
            if current_time - last_log_time >= log_interval:
                # Calculate metrics only when logging
                distance = math.sqrt(self.state[0]**2 + self.state[1]**2)
                direction = math.degrees(math.atan2(self.state[1], self.state[0]))
                speed = math.sqrt(self.state[2]**2 + self.state[3]**2)
                
                # Get time prefix with both elapsed and ROS time
                time_prefix = self.get_time_prefix()
                
                # Log fusion output with specific format for tabular extraction
                self.get_logger().info(
                    f"{time_prefix}[FUSION] dist={distance:.2f}m, dir={direction:.1f}°, "
                    f"pos=({self.state[0]:.2f}, {self.state[1]:.2f}), "
                    f"vel=({self.state[2]:.2f}, {self.state[3]:.2f}), "
                    f"speed={speed:.3f}m/s, state={motion_state}, "
                    f"uncert={self.position_uncertainty:.3f}m"
                )
                
                # Update last log time
                self._last_throttled_logs[log_key] = current_time
                
        except Exception as e:
            self.get_logger().error(f"Error publishing state: {str(e)}")
    
    def publish_status(self):
        """
        Publishes a summary of the current tracking status (active sensors, uncertainty, etc.).
        - Uses throttled logging to avoid spamming the log.
        """
        if not self.is_activated:
            return
            
        try:
            # Calculate uptime
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Count active sensors - use the cached values from sensor_manager
            active_3d = self.sensor_manager.get_active_high_quality_sensors()
            active_2d = self.sensor_manager._active_sensor_count - active_3d
            
            # Get motion state
            motion_state = "unknown"
            if hasattr(self, 'motion_manager'):
                motion_state = self.motion_manager.get_state_name()
                
            # Make sure position_uncertainty is initialized
            if not hasattr(self, 'position_uncertainty'):
                self.position_uncertainty = float('inf')
            
            # Log status with reduced frequency - only meaningful changes
            time_prefix = self.get_time_prefix()
            
            self.throttled_log(
                f"Status: Track={self.tracking_reliable}, Mode={motion_state}, "
                f"3D={active_3d}, 2D={active_2d}, Uncert={self.position_uncertainty:.3f}m",
                key="status",
                min_interval=5.0
            )
            
        except Exception as e:
            self.get_logger().error(f"Error publishing status: {str(e)}")
    
    def publish_diagnostics(self):
        """
        Publishes detailed diagnostics, including system resource usage and sensor health.
        - Useful for debugging and monitoring the system.
        """
        if not self.is_activated:
            return
            
        try:
            # Get CPU and memory usage if available - only check once per second
            current_time = time.time()
            cpu_usage = None
            memory_usage = None
            
            if HAS_PSUTIL and (current_time - self._last_performance_check >= 1.0):
                try:
                    cpu_usage = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    memory_usage = {
                        'percent': memory.percent,
                        'used_mb': memory.used / (1024 * 1024)
                    }
                    
                    # Update performance check time
                    self._last_performance_check = current_time
                    
                    # Add to CPU history for adaptive rate control
                    self._cpu_usage_history.append(cpu_usage)
                except:
                    pass
            
            # Get sensor diagnostics - limit to active sensors only
            sensor_info = self.sensor_manager.get_diagnostic_info()
            
            # Simplified sensor diagnostics with only active sensors
            active_sensors = {}
            for sensor, info in sensor_info.items():
                if info['active']:
                    active_sensors[sensor] = {
                        'count': info['count'],
                        'fps': round(float(info['fps']), 1)  # Convert to standard Python float
                    }
            
            # Build diagnostics message - convert all numpy types to Python native types
            diag = {
                'tracking': bool(self.tracking_reliable),
                'position': [float(self.state[0]), float(self.state[1])],
                'velocity': [float(self.state[2]), float(self.state[3])],
                'uncertainty': float(self.position_uncertainty),
                'motion_state': self.motion_manager.get_state_name() if hasattr(self, 'motion_manager') else "unknown",
                'active_sensors': active_sensors
            }
            
            # Add system metrics if available
            if cpu_usage is not None:
                diag['cpu'] = float(cpu_usage)
            if memory_usage is not None:
                diag['memory'] = {
                    'percent': float(memory_usage['percent']),
                    'used_mb': float(memory_usage['used_mb'])
                }
            
            # Create diagnostics message - ensure all values are JSON serializable
            diag_msg = String()
            diag_msg.data = json.dumps(diag)
            self.diagnostics_pub.publish(diag_msg)
            
            # Log system metrics with reduced frequency and only if significant change
            if cpu_usage is not None:
                # Check if CPU usage has changed significantly
                avg_cpu = sum(self._cpu_usage_history) / len(self._cpu_usage_history) if self._cpu_usage_history else cpu_usage
                
                # Log if CPU usage is high or has changed significantly
                if cpu_usage > 70.0 or abs(cpu_usage - avg_cpu) > 10.0:
                    self.throttled_log(
                        f"System: CPU={float(cpu_usage):.1f}%, Mem={float(memory_usage['percent']):.1f}%, "
                        f"Active sensors: {len(active_sensors)}",
                        key="system",
                        min_interval=15.0
                    )
                    
        except Exception as e:
            self.get_logger().error(f"Error publishing diagnostics: {str(e)}")
    
    def get_time_prefix(self):
        """
        Returns a string with the elapsed time since startup and the current ROS time.
        - Used to prefix log messages for easier debugging.
        """
        # Get elapsed time since startup
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Get ROS 2 time
        ros_time = self.get_clock().now()
        ros_seconds = ros_time.seconds_nanoseconds()
        ros_time_str = f"{ros_seconds[0]}.{str(ros_seconds[1]).zfill(9)[:6]}"
        
        # Return formatted prefix
        return f"[T+{elapsed_time:.1f}s][ROS:{ros_time_str}]"
    
    def adjust_processing_rates(self):
        """
        Dynamically adjusts how often the node processes data, based on CPU usage.
        - If the CPU is busy, slows down processing to avoid overloading the system.
        - If the CPU is idle, speeds up processing for better performance.
        """
        # Skip if psutil not available
        if not HAS_PSUTIL:
            return
        
        try:
            # Get current CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Calculate average CPU usage from history
            self._cpu_usage_history.append(cpu_usage)
            avg_cpu = sum(self._cpu_usage_history) / len(self._cpu_usage_history)
            
            # Adjust rates based on CPU usage
            if avg_cpu > self.performance_thresholds['high_cpu']:
                # High CPU usage - reduce processing rates
                self._adjust_rates('increase')
                self.throttled_log(
                    f"High CPU load detected ({avg_cpu:.1f}%) - reducing processing rates",
                    key="cpu_high",
                    min_interval=10.0
                )
            elif avg_cpu < self.performance_thresholds['low_cpu']:
                # Low CPU usage - increase processing rates if below base rates
                self._adjust_rates('decrease')
                self.throttled_log(
                    f"Low CPU load detected ({avg_cpu:.1f}%) - increasing processing rates",
                    key="cpu_low",
                    min_interval=20.0
                )
            elif avg_cpu < self.performance_thresholds['normal_cpu']:
                # Normal CPU usage - restore base rates
                self._adjust_rates('normal')
            
        except Exception as e:
            self.get_logger().error(f"Error adjusting processing rates: {str(e)}")
    
    def _adjust_rates(self, direction):
        """
        Helper function to adjust timer rates up or down.
        This is used for adaptive processing: if the CPU is busy, we slow down processing;
        if the CPU is idle, we speed up. This helps keep the system responsive and efficient.
        """
        needs_update = False
        for timer_name, rate_info in self._adaptive_rates.items():
            old_rate = rate_info['current']
            if direction == 'increase':
                # Increase period (reduce frequency)
                new_rate = min(rate_info['current'] * 1.2, rate_info['max'])
            elif direction == 'decrease':
                # Decrease period (increase frequency)
                new_rate = max(rate_info['current'] * 0.8, rate_info['min'])
            else:  # 'normal'
                # Restore base rate
                new_rate = rate_info['base']
            # Only update if rate changed significantly
            if abs(new_rate - old_rate) > 0.01:
                rate_info['current'] = new_rate
                needs_update = True
        # If any rates changed, update the timers
        if needs_update:
            self._update_timers()

    def _update_timers(self):
        """
        Updates the timers to use the new rates after an adjustment.
        This function destroys the old timers and creates new ones with the updated frequencies.
        This is important for adaptive processing, so the node can respond to CPU load or other conditions.
        """
        # Destroy old timers for the main periodic tasks
        old_timers = []
        for timer in self._timer_list:
            if timer.callback == self.filter_update:
                old_timers.append(timer)
            elif timer.callback == self.publish_state:
                old_timers.append(timer)
            elif timer.callback == self.publish_status:
                old_timers.append(timer)
            elif timer.callback == self.publish_diagnostics:
                old_timers.append(timer)
        # Remove and destroy old timers
        for timer in old_timers:
            if timer in self._timer_list:
                self._timer_list.remove(timer)
                self.destroy_timer(timer)
        # Create new timers with updated rates
        self._timer_list.append(self.create_timer(
            self._adaptive_rates['filter_update']['current'], 
            self.filter_update, 
            callback_group=self.timer_cb_group
        ))
        self._timer_list.append(self.create_timer(
            self._adaptive_rates['publish_state']['current'], 
            self.publish_state, 
            callback_group=self.timer_cb_group
        ))
        self._timer_list.append(self.create_timer(
            self._adaptive_rates['publish_status']['current'], 
            self.publish_status, 
            callback_group=self.timer_cb_group
        ))
        self._timer_list.append(self.create_timer(
            self._adaptive_rates['publish_diagnostics']['current'], 
            self.publish_diagnostics, 
            callback_group=self.timer_cb_group
        ))
        # Log timer updates for educational feedback
        self.throttled_log(
            f"Updated timer rates: filter={self._adaptive_rates['filter_update']['current']:.2f}s, "
            f"publish={self._adaptive_rates['publish_state']['current']:.2f}s",
            key="timer_update",
            min_interval=5.0
        )
    
    def throttled_log(self, message, key, min_interval=1.0, level="info"):
        """
        Logs a message, but only if enough time has passed since the last log with the same key.
        - Used to avoid spamming the log with repeated messages.
        """
        current_time = time.time()
        
        # Initialize tracking dict if needed
        if not hasattr(self, '_last_throttled_logs'):
            self._last_throttled_logs = {}
            
        # Check if enough time has passed since last log
        if key in self._last_throttled_logs:
            elapsed = current_time - self._last_throttled_logs[key]
            if elapsed < min_interval:
                return
                
        # Update last log time
        self._last_throttled_logs[key] = current_time
        
        # Add time prefix with both elapsed and ROS times
        time_prefix = self.get_time_prefix()
        
        # Format the message with the time prefix
        message = f"{time_prefix} {message}"
        
        # Log with appropriate level
        if level == "error":
            self.get_logger().error(message)
        elif level == "warn":
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    def dynamic_uncertainty_recovery(self):
        """
        Tries to reduce uncertainty when sensors are missing or unreliable.
        
        Textbook Explanation:
        ---------------------
        If we lose all sensors, we don't want our uncertainty to grow forever. This function gently reduces uncertainty and slows down the ball, assuming it probably stopped or is not moving much. This prevents the filter from becoming useless during sensor outages.
        - If a few sensors are active, we recover uncertainty more slowly.
        """
        # Only apply when uncertainty exceeds normal thresholds
        if self.position_uncertainty < 0.3:  # Lowered from 0.4 to be closer to normal operation
            return False
            
        # Check sensor health
        current_time = time.time()
        active_sensors = self.sensor_manager.get_active_sensor_count()
        
        # Get motion state for context
        motion_state = self.motion_manager.current_state if hasattr(self, 'motion_manager') else MotionStateManager.UNKNOWN
        
        # Apply different strategies based on sensor availability
        if active_sensors == 0:
            # No active sensors - use last known good velocity with damping
            # Calculate time since last sensor update
            time_since_update = 0.0
            if hasattr(self.sensor_manager, 'last_update_time') and self.sensor_manager.last_update_time:
                last_update_times = list(self.sensor_manager.last_update_time.values())
                if last_update_times:
                    time_since_update = min(current_time - max(last_update_times), 2.0)
            
            # Apply damping to velocity based on time without updates
            damping_factor = max(0.0, 1.0 - (time_since_update / 1.5))  # More aggressive damping
            self.state[2] *= damping_factor  # Dampen vx
            self.state[3] *= damping_factor  # Dampen vy
            
            # Gradually reduce uncertainty with a floor - more aggressive reduction
            if self.position_uncertainty > 0.25:
                # Use a more aggressive recovery rate
                recovery_rate = 0.95  # Faster recovery rate
                
                # Lower uncertainty floor based on motion state
                if motion_state == MotionStateManager.STATIONARY:
                    uncertainty_floor = 0.2  # Lower floor for stationary objects
                else:
                    uncertainty_floor = 0.25  # Slightly higher for moving objects
                
                # Calculate new uncertainty with more aggressive reduction
                new_uncertainty = max(
                    self.position_uncertainty * recovery_rate,
                    uncertainty_floor
                )
                
                # Only log and apply if actually changed
                if abs(new_uncertainty - self.position_uncertainty) > 0.001:
                    # Apply to covariance
                    scale = (new_uncertainty / self.position_uncertainty) ** 2
                    self.covariance[0:2, 0:2] *= scale
                    
                    # Store original for logging
                    old_uncertainty = self.position_uncertainty
                    
                    # Update uncertainty metric
                    self.position_uncertainty = new_uncertainty
                    
                    self.get_logger().info(
                        f"{self.get_time_prefix()}[RECOVERY] Uncertainty adjusted from "
                        f"{old_uncertainty:.3f}m to {new_uncertainty:.3f}m"
                    )
                    return True
        elif active_sensors < 2:
            # Limited sensors - apply milder recovery
            if self.position_uncertainty > 0.35:
                recovery_rate = 0.97  # Milder recovery
                uncertainty_floor = 0.25
                
                # Calculate new uncertainty
                new_uncertainty = max(
                    self.position_uncertainty * recovery_rate,
                    uncertainty_floor
                )
                
                # Only apply if actually changed
                if abs(new_uncertainty - self.position_uncertainty) > 0.001:
                    # Apply to covariance
                    scale = (new_uncertainty / self.position_uncertainty) ** 2
                    self.covariance[0:2, 0:2] *= scale
                    
                    # Store original for logging
                    old_uncertainty = self.position_uncertainty
                    
                    # Update uncertainty metric
                    self.position_uncertainty = new_uncertainty
                    
                    self.get_logger().info(
                        f"{self.get_time_prefix()}[RECOVERY] Partial uncertainty recovery: "
                        f"{old_uncertainty:.3f}m to {new_uncertainty:.3f}m (limited sensors)"
                    )
                    return True
        
        return False

    def update_uncertainty_metrics(self):
        """
        Updates the uncertainty metrics (how sure are we about position and velocity?)
        
        Textbook Explanation:
        ---------------------
        The Kalman filter keeps track of how uncertain it is about its estimate. This function calculates the uncertainty from the covariance matrix, smooths it to avoid rapid changes, and caps it based on the ball's motion state.
        
        Math:
        - Position uncertainty = sqrt(average of position variances)
        - Velocity uncertainty = sqrt(average of velocity variances)
        - We apply smoothing (weighted average) to avoid rapid jumps.
        - We cap the uncertainty based on the motion state (stationary objects should not have high uncertainty).
        - We add a minimum floor to avoid being overconfident.
        """
        # Original position uncertainty calculation from position covariance
        raw_position_uncertainty = math.sqrt((self.covariance[0, 0] + self.covariance[1, 1]) / 2.0)
        
        # Apply a smoothing factor to avoid rapid changes
        if hasattr(self, 'position_uncertainty'):
            smoothing_factor = 0.8  # Weight toward previous value for stability
            self.position_uncertainty = smoothing_factor * self.position_uncertainty + (1 - smoothing_factor) * raw_position_uncertainty
        else:
            self.position_uncertainty = raw_position_uncertainty
        
        # Velocity uncertainty from velocity covariance
        self.velocity_uncertainty = math.sqrt((self.covariance[2, 2] + self.covariance[3, 3]) / 2.0)
        
        # Apply state-based uncertainty caps
        if hasattr(self, 'motion_manager'):
            motion_state = self.motion_manager.current_state
            
            if motion_state == MotionStateManager.STATIONARY:
                max_pos_uncertainty = 0.6  # Reduced from 0.8
                max_vel_uncertainty = 1.2  # Reduced from 1.5
            elif motion_state == MotionStateManager.LONG_STATIONARY:
                max_pos_uncertainty = 0.5  # Reduced from 0.7
                max_vel_uncertainty = 1.0  # Reduced from 1.2
            elif motion_state == MotionStateManager.SMALL_MOVEMENT:
                max_pos_uncertainty = 0.8  # Reduced from 1.2
                max_vel_uncertainty = 1.5  # Reduced from 2.0
            elif motion_state == MotionStateManager.MEDIUM_FAST:
                max_pos_uncertainty = 1.0  # Reduced from 1.5
                max_vel_uncertainty = 2.0  # Reduced from 2.5
            else:  # unknown
                max_pos_uncertainty = 1.0  # Reduced from 1.5
                max_vel_uncertainty = 1.8  # Reduced from 2.2
                
            # Apply caps with scale factors
            if self.position_uncertainty > max_pos_uncertainty:
                scale = (max_pos_uncertainty / self.position_uncertainty) ** 2
                self.covariance[0:2, 0:2] *= scale
                self.position_uncertainty = max_pos_uncertainty
                
            if self.velocity_uncertainty > max_vel_uncertainty:
                scale = (max_vel_uncertainty / self.velocity_uncertainty) ** 2
                self.covariance[2:4, 2:4] *= scale
                self.velocity_uncertainty = max_vel_uncertainty
                
            # Add minimum uncertainty floors to avoid overconfidence
            if self.position_uncertainty < 0.1:
                floor_scale = (0.1 / self.position_uncertainty) ** 2
                self.covariance[0:2, 0:2] *= floor_scale
                self.position_uncertainty = 0.1

    def filter_update(self):
        """
        The main filter update loop.
        
        Textbook Explanation:
        ---------------------
        This function runs the Kalman filter's two main steps:
        1. Prediction: Use physics to predict where the ball should be.
        2. Update: Use new sensor data to correct the prediction.
        It also updates the motion state, applies physical constraints, updates uncertainty, and tracking status.
        This loop is the heart of the sensor fusion process.
        """
        if not self.is_activated:
            return
            
        current_time = time.time()
        
        # Skip if not initialized yet (wait for initialization)
        if not self.initialized:
            # Try initialization with sensor data
            for sensor in ['lidar', 'yolo_3d']:
                msg = self.sensor_manager.get_latest(sensor)
                if msg:
                    transformed = self.transform_point(msg, self.reference_frame)
                    if transformed:
                        if self.initialize_with_measurement(transformed, sensor):
                            break
            
            # Try with 2D sensors if 3D not available
            if not self.initialized:
                for sensor in ['yolo_2d']:
                    if sensor in self.bbox_data and self.bbox_data[sensor]['timestamp'] > 0:
                        msg = self.sensor_manager.get_latest(sensor)
                        if msg:
                            estimated_3d = self.estimate_3d_from_2d(msg, self.bbox_data[sensor])
                            if estimated_3d:
                                self.initialize_with_measurement(estimated_3d, f"{sensor}_est3d")
                                break
            
            return
        
        try:
            # Calculate time step
            if not hasattr(self, 'last_update_time') or self.last_update_time is None:
                dt = 0.1  # Default time step
            else:
                dt = current_time - self.last_update_time
                # Limit dt to reasonable values
                dt = min(dt, 0.25)  # Reduced cap from 0.5 to 0.25 seconds
            
            # Round dt to nearest pre-computed value for optimized matrix operations
            dt_ms = round(dt * 1000)
            dt_key = (dt_ms // 25) * 25  # Round to nearest 25ms
            dt_key = max(25, min(500, dt_key))  # Clamp between 25ms and 500ms
            
            # Adaptive update rate based on motion state
            if hasattr(self, 'motion_manager'):
                motion_state = self.motion_manager.current_state
                
                # For stationary objects with low uncertainty, we can skip some updates
                if motion_state in [MotionStateManager.STATIONARY, MotionStateManager.LONG_STATIONARY] and \
                   self.position_uncertainty < 0.2:
                    # Skip updates more aggressively for long stationary objects
                    if motion_state == MotionStateManager.LONG_STATIONARY:
                        # Only update every 3rd call for long stationary
                        if not hasattr(self, '_stationary_counter'):
                            self._stationary_counter = 0
                        
                        self._stationary_counter = (self._stationary_counter + 1) % 3
                        if self._stationary_counter != 0:
                            # Just update tracking status and return
                            self.update_tracking_status()
                            return
                    else:
                        # Only update every other call for stationary
                        if not hasattr(self, '_stationary_counter'):
                            self._stationary_counter = 0
                        
                        self._stationary_counter = (self._stationary_counter + 1) % 2
                        if self._stationary_counter != 0:
                            # Just update tracking status and return
                            self.update_tracking_status()
                            return
            
            # Check sensor health before prediction
            self.sensor_manager.update_sensor_health(current_time)
            active_sensors = self.sensor_manager.get_active_sensor_count()
            
            # Apply preliminary uncertainty damping if no sensors active
            # This prevents runaway uncertainty during prediction when no sensors
            if active_sensors == 0 and self.position_uncertainty > 0.3:
                # Pre-recovery for prediction phase
                self.state[2] *= 0.9  # Dampen velocity before prediction
                self.state[3] *= 0.9
                
            # 1. Prediction stage with optimized dt values
            self.predict_state(dt, dt_key)
            
            # 2. Update stage with motion-aware sensor processing
            self.process_sensor_data()
            
            # 3. Update motion state
            self.update_motion_state()
            
            # 4. Apply physics constraints
            self.apply_physics_constraints()
            
            # 5. Update uncertainty metrics
            self.update_uncertainty_metrics()
            
            # 6. Apply dynamic uncertainty recovery for improved stability
            self.dynamic_uncertainty_recovery()
            
            # 7. Update tracking status
            self.update_tracking_status()
            
            # Update last update time
            self.last_update_time = current_time
            
        except Exception as e:
            self.get_logger().error(f"Filter update error: {str(e)}")

    def predict_state(self, dt, dt_key=None):
        """
        Predicts the next state of the ball using physics (where should it be, given its velocity?).
        
        Textbook Explanation:
        ---------------------
        This is the prediction step of the Kalman filter. We use the current state (position, velocity) and physics (velocity, friction) to predict where the ball should be after a small time step.
        
        Math:
        - State prediction: x = F x (where F is the state transition matrix)
        - Uncertainty prediction: P = F P F^T + Q (where Q is the process noise matrix)
        - We also apply friction to slow the ball down if it's supposed to be stationary.
        - This step increases our uncertainty, because the future is never perfectly predictable.
        """
        # Use pre-computed dt values if available
        dt_info = None
        if dt_key is not None and dt_key in self._dt_values:
            dt_info = self._dt_values[dt_key]
            dt = dt_info['dt']  # Use exact dt from pre-computed values
        
        # Update state transition matrix for current dt
        self._F[0, 2] = dt  # x += vx*dt (position update from velocity)
        self._F[1, 3] = dt  # y += vy*dt
        
        # Get motion state-based scaling with more precise tuning
        motion_state = self.motion_manager.current_state if hasattr(self, 'motion_manager') else MotionStateManager.UNKNOWN
        motion_scale = 1.0
        
        if motion_state == MotionStateManager.STATIONARY:
            motion_scale = 0.7  # Reduced from 0.8
        elif motion_state == MotionStateManager.LONG_STATIONARY:
            motion_scale = 0.5  # Reduced from 0.6
        elif motion_state == MotionStateManager.SMALL_MOVEMENT:
            motion_scale = 0.9  # Reduced from 1.0
        elif motion_state == MotionStateManager.MEDIUM_FAST:
            motion_scale = 1.1  # Reduced from 1.2
        
        # Process noise parameters
        q_pos = self.process_noise_pos * dt * motion_scale
        q_vel = self.process_noise_vel * dt * motion_scale
        
        # Apply friction based on state with improved physics
        if motion_state != MotionStateManager.MEDIUM_FAST:
            # Calculate current velocity
            vx, vy = self.state[2], self.state[3]
            v_squared = vx*vx + vy*vy
            
            if v_squared > 0.0001:  # Avoid sqrt for near-zero velocities
                current_velocity = math.sqrt(v_squared)
                
                # Apply friction based on state - more aggressive for stationary
                friction_coef = 0.015  # Increased from 0.01
                if motion_state == MotionStateManager.STATIONARY:
                    friction_coef = 0.03  # Increased from 0.02
                elif motion_state == MotionStateManager.LONG_STATIONARY:
                    friction_coef = 0.04  # Increased from 0.03
                
                # Adjust friction based on uncertainty
                if hasattr(self, 'position_uncertainty') and self.position_uncertainty > 0.3:
                    # Apply stronger friction during high uncertainty
                    friction_coef *= 1.5
                
                # Calculate deceleration
                deceleration = friction_coef * 9.81  # μg
                dv = min(current_velocity, deceleration * dt)
                
                # Apply proportional deceleration
                if dv > 0.001:  # Only apply if significant
                    factor = 1.0 - (dv / current_velocity)
                    self.state[2] *= factor
                    self.state[3] *= factor
        
        # Performance optimization: If state is nearly stationary, skip complex matrix operations
        if abs(self.state[2]) < 0.01 and abs(self.state[3]) < 0.01 and \
           motion_state in [MotionStateManager.STATIONARY, MotionStateManager.LONG_STATIONARY]:
            # Just add process noise to covariance for stationary objects
            self.covariance[0, 0] += q_pos * dt * dt / 3.0
            self.covariance[1, 1] += q_pos * dt * dt / 3.0
            self.covariance[2, 2] += q_vel * dt
            self.covariance[3, 3] += q_vel * dt
            
            # Add cross-terms for symmetry
            self.covariance[0, 2] = self.covariance[2, 0] = q_pos * dt * dt / 2.0
            self.covariance[1, 3] = self.covariance[3, 1] = q_pos * dt * dt / 2.0
            
            return
            
        # For moving objects, do the full prediction
        
        # Reset process noise matrix
        self._Q.fill(0.0)
        
        # Use pre-computed dt values for Q if available
        if dt_info is not None:
            dt2 = dt_info['dt2']
            dt3 = dt_info['dt3']
        else:
            dt2 = dt * dt
            dt3 = dt2 * dt
        
        # Fill in process noise with optimized access
        # Position variances
        self._Q[0, 0] = q_pos * dt3 / 3.0
        self._Q[1, 1] = q_pos * dt3 / 3.0
        
        # Velocity variances
        self._Q[2, 2] = q_vel * dt
        self._Q[3, 3] = q_vel * dt
        
        # Covariances
        self._Q[0, 2] = self._Q[2, 0] = q_pos * dt2 / 2.0
        self._Q[1, 3] = self._Q[3, 1] = q_pos * dt2 / 2.0
        
        # Check for uncertainty-based scaling - limit growth for high uncertainty
        if hasattr(self, 'position_uncertainty') and self.position_uncertainty > 0.3:
            # Scale down process noise when uncertainty is already high
            uncertainty_factor = 0.3 / self.position_uncertainty
            self._Q *= max(0.5, uncertainty_factor)
        
        # Direct state update for position - faster than matrix multiplication
        self.state[0] += dt * self.state[2]  # x = x + vx*dt
        self.state[1] += dt * self.state[3]  # y = y + vy*dt
        
        # Predict covariance using direct matrix operations
        # This is an optimized version of: P = F*P*F^T + Q
        
        # Calculate F*P first
        FP = np.zeros((4, 4), dtype=np.float32)
        
        # Explicit matrix multiplication for F*P
        # First row
        FP[0, 0] = self.covariance[0, 0] + dt * self.covariance[2, 0]
        FP[0, 1] = self.covariance[0, 1] + dt * self.covariance[2, 1]
        FP[0, 2] = self.covariance[0, 2] + dt * self.covariance[2, 2]
        FP[0, 3] = self.covariance[0, 3] + dt * self.covariance[2, 3]
        
        # Second row
        FP[1, 0] = self.covariance[1, 0] + dt * self.covariance[3, 0]
        FP[1, 1] = self.covariance[1, 1] + dt * self.covariance[3, 1]
        FP[1, 2] = self.covariance[1, 2] + dt * self.covariance[3, 2]
        FP[1, 3] = self.covariance[1, 3] + dt * self.covariance[3, 3]
        
        # Third and fourth rows (no change from original)
        FP[2, :] = self.covariance[2, :]
        FP[3, :] = self.covariance[3, :]
        
        # Now calculate FP*F^T (transpose multiply)
        FPFT = np.zeros((4, 4), dtype=np.float32)
        
        # First row
        FPFT[0, 0] = FP[0, 0]
        FPFT[0, 1] = FP[0, 1]
        FPFT[0, 2] = FP[0, 2] + dt * FP[0, 0]
        FPFT[0, 3] = FP[0, 3] + dt * FP[0, 1]
        
        # Second row
        FPFT[1, 0] = FP[1, 0]
        FPFT[1, 1] = FP[1, 1]
        FPFT[1, 2] = FP[1, 2] + dt * FP[1, 0]
        FPFT[1, 3] = FP[1, 3] + dt * FP[1, 1]
        
        # Third row
        FPFT[2, 0] = FP[2, 0] + dt * FP[0, 0]
        FPFT[2, 1] = FP[2, 1] + dt * FP[0, 1]
        FPFT[2, 2] = FP[2, 2] + dt * (FP[2, 0] + FP[0, 2]) + dt*dt * FP[0, 0]
        FPFT[2, 3] = FP[2, 3] + dt * (FP[2, 1] + FP[0, 3]) + dt*dt * FP[0, 1]
        
        # Fourth row
        FPFT[3, 0] = FP[3, 0] + dt * FP[1, 0]
        FPFT[3, 1] = FP[3, 1] + dt * FP[1, 1]
        FPFT[3, 2] = FP[3, 2] + dt * (FP[3, 0] + FP[1, 2]) + dt*dt * FP[1, 0]
        FPFT[3, 3] = FP[3, 3] + dt * (FP[3, 1] + FP[1, 3]) + dt*dt * FP[1, 1]
        
        # Final update: P = F*P*F^T + Q
        self.covariance = FPFT + self._Q


def main(args=None):
    """
    Main function to start the node.
    - Initializes ROS 2.
    - Creates the fusion node.
    - Runs the node using a multi-threaded executor.
    - Handles clean shutdown on exit.
    """
    rclpy.init(args=args)
    
    # Use MultiThreadedExecutor with reduced thread count
    # Pi has 4 cores, and we need to share with other nodes
    executor = MultiThreadedExecutor(num_threads=2)
    
    # Create node
    node = OptimizedFusionNode()
    
    # Add to executor
    executor.add_node(node)
    
    try:
        # Spin the executor
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Clean shutdown
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()