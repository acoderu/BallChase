#!/usr/bin/env python3

"""
Basketball Tracking Robot - Optimized LIDAR Detection Node
=========================================================

This node processes 2D LIDAR data to detect a basketball and provide 3D position information.
It correlates LIDAR data with camera-based detections from YOLO.

Key optimizations:
- Static transform caching with one-time initialization
- Lightweight buffer implementation with fixed memory allocation
- Message object reuse to reduce allocations
- Motion-aware processing strategies
- Optimized NumPy operations with explicit data types
- Enhanced RANSAC algorithm with early termination
- Efficient QoS profiles
- Adaptive processing based on system load
- Visualization components removed for performance
- Optimized for Raspberry Pi 5 single-thread execution

Physical Setup:
- LIDAR mounted 6 inches (15.24 cm) above ground
- Basketball diameter: 9 inches (22.86 cm)
- Basketball rolls on ground only (center is always 4.5 inches above ground)
"""
# Standard library imports
import sys
import rclpy
from rclpy.node import Node
import numpy as np
import math
import time
from collections import deque
import threading
import psutil  # For CPU monitoring
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.logging import LoggingSeverity

# ROS2 messages
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, TransformStamped
from std_msgs.msg import String, Float32, Bool
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import PointStamped as TF2PointStamped
import json

import os
# Import GroundPositionFilter for shared ground movement tracking
from ball_chase.config.config_loader import ConfigLoader
from ball_chase.utilities.ground_position_filter import GroundPositionFilter


class ObjectPool:
    """
    ObjectPool is a design pattern that helps manage a set of reusable objects.
    Instead of creating and destroying objects all the time (which can be slow and use a lot of memory),
    we keep a pool (a collection) of objects that we can reuse. This is especially useful in robotics
    where we need to create many similar objects (like arrays for LIDAR points) very quickly and often.
    
    Why use an object pool?
    - Saves time: Creating new objects is slower than reusing existing ones.
    - Saves memory: We avoid memory fragmentation and reduce garbage collection.
    - Makes the program more efficient, especially on devices with limited resources (like Raspberry Pi).
    """
    
    def __init__(self, factory_func, initial_size=5, max_size=20):
        """Initialize the object pool.
        
        Args:
            factory_func: Function that creates a new object
            initial_size: Initial number of objects to create
            max_size: Maximum pool size
        """
        self.factory_func = factory_func
        self.max_size = max_size
        self.pool = []
        
        # Pre-populate pool
        for _ in range(initial_size):
            self.pool.append(factory_func())
    
    def get(self):
        """Get an object from the pool or create a new one."""
        if not self.pool:
            return self.factory_func()
        return self.pool.pop()
    
    def put(self, obj):
        """Return an object to the pool."""
        if len(self.pool) < self.max_size:
            self.pool.append(obj)
        # Otherwise object is left for garbage collection


class LightweightBuffer:
    """
    LightweightBuffer is a simple, efficient way to store a fixed number of recent values.
    It's like a small notebook that only keeps the last N things you write in it.
    
    Why use a lightweight buffer?
    - Uses a fixed amount of memory, so it won't grow too large and slow down the program.
    - Fast to add and retrieve recent data, which is important for real-time robotics.
    - Useful for keeping track of recent positions, errors, or other time-series data.
    """
    
    def __init__(self, max_size=10):
        """Initialize with fixed buffer size."""
        self.data = []
        self.max_size = max_size
        self.next_index = 0
        self.is_full = False
    
    def add(self, timestamp, value):
        """Add value to buffer with fixed memory allocation."""
        if len(self.data) < self.max_size:
            self.data.append((timestamp, value))
        else:
            self.data[self.next_index] = (timestamp, value)
            self.next_index = (self.next_index + 1) % self.max_size
            self.is_full = True
    
    def get_latest(self):
        """Get the most recent value."""
        if not self.data:
            return None
        latest_idx = (self.next_index - 1) % len(self.data)
        return self.data[latest_idx][1]
    
    def get_latest_before(self, timestamp, max_age=1.0):
        """Get the most recent value before the given timestamp."""
        if not self.data:
            return None
        
        best_time_diff = float('inf')
        best_value = None
        
        for t, value in self.data:
            time_diff = timestamp - t
            if 0 <= time_diff < best_time_diff and time_diff <= max_age:
                best_time_diff = time_diff
                best_value = value
        
        return best_value
    
    def get_all_within(self, start_time, end_time):
        """Get all values within a time range."""
        return [(t, v) for t, v in self.data if start_time <= t <= end_time]
    
    def clear(self):
        """Clear the buffer."""
        self.data = []
        self.next_index = 0
        self.is_full = False


class MotionStateManager:
    """
    MotionStateManager keeps track of how the basketball is moving: is it stationary, moving slowly, or moving fast?
    It uses a concept called "hysteresis" to avoid switching states too quickly due to small changes or noise.
    
    Why is this important?
    - In robotics, we want to know if the ball is stopped, rolling slowly, or moving fast so we can react appropriately.
    - Hysteresis means we require several pieces of evidence before changing state, which makes the system more stable.
    - This class also keeps a history of state changes for debugging and analysis.
    """
    
    # Define motion states
    UNKNOWN = "unknown"
    STATIONARY = "stationary"
    SMALL_MOVEMENT = "small_movement"
    MEDIUM_FAST = "medium_fast"
    
    def __init__(self):
        """Initialize state manager."""
        # Current state
        self.current_state = self.UNKNOWN
        self.previous_state = self.UNKNOWN
        
        # State confidence (0-1)
        self.state_confidence = {
            self.UNKNOWN: 1.0,
            self.STATIONARY: 0.0,
            self.SMALL_MOVEMENT: 0.0,
            self.MEDIUM_FAST: 0.0
        }
        
        # State transition counters with hysteresis
        self.state_evidence = {
            self.STATIONARY: 0,
            self.SMALL_MOVEMENT: 0,
            self.MEDIUM_FAST: 0
        }
        
        # Timing trackers
        self.stationary_start_time = None
        self.last_state_change_time = time.time()
        
        # Speed thresholds
        self.stationary_threshold = 0.05  # m/s
        self.small_movement_threshold = 0.20  # m/s
        
        # Required evidence for state changes
        self.evidence_threshold = 3
        
        # History for debugging
        self.state_history = deque(maxlen=10)
    
    def update(self, velocity, position=None, force_state=None):
        """
        Update motion state based on velocity or force a specific state.
        
        Args:
            velocity: Current velocity estimate (magnitude in m/s)
            position: Optional position for continuity checks
            force_state: Optional state to force (for overrides)
        
        Returns:
            bool: True if state changed
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
        
        # Determine base state from velocity
        if velocity < self.stationary_threshold:
            base_state = self.STATIONARY
            self.state_evidence[self.STATIONARY] += 1
            self.state_evidence[self.SMALL_MOVEMENT] = 0
            self.state_evidence[self.MEDIUM_FAST] = 0
        elif velocity < self.small_movement_threshold:
            base_state = self.SMALL_MOVEMENT
            self.state_evidence[self.STATIONARY] = 0
            self.state_evidence[self.SMALL_MOVEMENT] += 1
            self.state_evidence[self.MEDIUM_FAST] = 0
        else:
            base_state = self.MEDIUM_FAST
            self.state_evidence[self.STATIONARY] = 0
            self.state_evidence[self.SMALL_MOVEMENT] = 0
            self.state_evidence[self.MEDIUM_FAST] += 1
        
        # Apply symmetric hysteresis with evidence thresholds
        if self.current_state == self.STATIONARY:
            # Check for stationary start time
            if self.stationary_start_time is None:
                self.stationary_start_time = current_time
            
            # Check for exit from STATIONARY
            if base_state != self.STATIONARY and self.state_evidence[base_state] >= self.evidence_threshold:
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
        """Get validation threshold multiplier based on current state."""
        if self.current_state == self.STATIONARY:
            return 1.0
        elif self.current_state == self.SMALL_MOVEMENT:
            return 1.3
        elif self.current_state == self.MEDIUM_FAST:
            return 1.5
        else:
            return 1.2
    
    def is_in_transition(self, max_transition_time=1.0):
        """Check if currently in state transition."""
        return time.time() - self.last_state_change_time < max_transition_time
    
    def get_state_age(self):
        """Get the age of the current state."""
        if self.last_state_change_time:
            return time.time() - self.last_state_change_time
        return 0.0


class BasketballLidarDetector(Node):
    """
    BasketballLidarDetector is the main class for this node. It brings together all the components needed to detect a basketball using LIDAR and camera data.
    
    What does this class do?
    - Sets up ROS2 publishers and subscribers to communicate with other parts of the robot.
    - Manages object pools and buffers for efficient memory use.
    - Processes LIDAR scans to find the basketball using circle fitting and RANSAC.
    - Correlates LIDAR data with camera detections for more accurate 3D position estimation.
    - Tracks the motion state of the ball (stopped, slow, fast) and adapts processing accordingly.
    - Publishes the detected position and diagnostics for monitoring.
    - Handles system resource monitoring and adapts performance if the CPU is overloaded.
    
    Why is this class important?
    - It is the "brain" of the basketball detection system, coordinating all the parts to work together efficiently.
    - It is optimized for running on a Raspberry Pi, which has limited resources, so every optimization helps!
    """
    
    def __init__(self):
        """Initialize the basketball LIDAR detector node."""
        super().__init__('basketball_lidar_detector')
        
        # Track timers explicitly since Node doesn't have get_timers()
        self.node_timers = []
        
        # Initialize transform timestamps dictionary 
        self.transform_timestamps = {}
        
        # Create object pools
        self._create_object_pools()
        
        # Load configuration
        self.config_loader = ConfigLoader()
        try:
            self.config = self.config_loader.load_yaml('lidar_config.yaml')
        except Exception as e:
            self.get_logger().error(f"Failed to load config: {str(e)}")
            self.config = {}
        
        # Initialize callback groups for concurrency management
        self.timer_cb_group = rclpy.callback_groups.MutuallyExclusiveCallbackGroup()
        self.subscription_cb_group = rclpy.callback_groups.ReentrantCallbackGroup()
        
        # Load performance configuration
        self._load_performance_config()
        
        # Initialize state
        self._init_state()
        
        # Initialize coordinate transform parameters first
        self._init_transform_parameters()
        
        # EMA smoothing for YOLO distance
        self.smoothed_yolo_distance = None
        self.ema_alpha = 0.5  # Smoothing factor between 0 (more smoothing) and 1 (no smoothing)
        
        # Set up TF system - Initialize buffer and listener FIRST
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize motion state manager
        self.motion_manager = MotionStateManager()
        
        # Manage publisher objects for reuse
        self._initialize_publisher_objects()
        
        # Initialize transform cache
        self.cached_transforms = {}
        
        # Pre-allocate arrays for vector operations
        self._init_vector_arrays()
        
        # Load basketball parameters
        self._load_basketball_parameters()
        
        # Create a lock for thread safety
        self.lock = threading.RLock()
        
        # Set up a periodic timer to check transform availability (for debugging)
        timer = self.create_timer(
            5.0, self.check_transform, callback_group=self.timer_cb_group)
        self.node_timers.append(timer)
        
        # Set up diagnostics timer with throttled interval
        diag_interval = self.config.get('diagnostics', {}).get('publish_interval', 3.0)
        self.diagnostics_timer = self.create_timer(
            diag_interval, self.publish_diagnostics, callback_group=self.timer_cb_group)
        self.node_timers.append(self.diagnostics_timer)
        
        # Set up staged startup to reduce initial CPU spikes
        timer = self.create_timer(
            0.1, self.staged_startup, callback_group=self.timer_cb_group)
        self.node_timers.append(timer)
        
        # Set up transform cache cleanup timer - longer interval for Pi 5
        timer = self.create_timer(
            900.0, self.clean_transform_cache, callback_group=self.timer_cb_group)  # 15 minutes
        self.node_timers.append(timer)
        
        self.get_logger().info("Basketball LIDAR detector initialized with optimized memory management")
        
        # Flag to track successful transforms
        self.transform_published_successfully = False
    
    def _create_object_pools(self):
        """
        Create object pools for frequently used objects.
        This helps us avoid creating and destroying large arrays all the time, which saves time and memory.
        For example, we use pools for arrays that store LIDAR points, distances, and masks.
        """
        # Pool for point arrays
        self.point_pool = ObjectPool(
            lambda: np.zeros((500, 3), dtype=np.float32),
            initial_size=3,
            max_size=5
        )
        
        # Pool for distance arrays
        self.distance_pool = ObjectPool(
            lambda: np.zeros(500, dtype=np.float32),
            initial_size=3,
            max_size=5
        )
        
        # Pool for mask arrays
        self.mask_pool = ObjectPool(
            lambda: np.zeros(500, dtype=bool),
            initial_size=3,
            max_size=5
        )
    
    def _initialize_publisher_objects(self):
        """
        Initialize publisher message objects for reuse.
        Instead of creating new message objects every time we want to publish, we reuse the same ones.
        This is more efficient and reduces memory usage.
        """
        # Pre-create common message objects that will be reused
        self._pos_msg = PointStamped()
        self._debug_msg = PointStamped()
        self._diag_msg = String()
        self._status_msg = Bool()
    
    def _init_vector_arrays(self):
        """
        Pre-allocate arrays for vector operations.
        By creating arrays ahead of time, we avoid having to allocate memory inside loops, which makes the code faster.
        This is especially important for real-time robotics where every millisecond counts.
        """
        # Small array for circle fitting
        self._circle_points = np.zeros((3, 2), dtype=np.float32)
        
        # Array for transform calculations
        self._transform_matrix = np.eye(4, dtype=np.float32)
        
        # Arrays for RANSAC
        max_size = self.max_point_limit if hasattr(self, 'max_point_limit') else 500
        self._inlier_mask = np.zeros(max_size, dtype=bool)
        
        # Reusable arrays for distance calculations
        self._distances_array = np.zeros(max_size, dtype=np.float32)
        
        # Pre-allocate array for angle calculations to avoid allocation inside loops
        self._angles_array = np.zeros(max_size, dtype=np.float32)
    
    def staged_startup(self):
        """
        Staged startup to reduce initial CPU load spikes.
        Instead of starting everything at once (which can overload the CPU), we start subscribers, publishers, and transform caching in stages.
        This makes the node start up more smoothly, especially on slower hardware.
        """
        # Remove the timer
        for i, timer in enumerate(self.node_timers):
            if timer.callback == self.staged_startup:
                self.destroy_timer(timer)
                self.node_timers.pop(i)
                break
        
        # Set up subscribers with staggered creation
        self._setup_subscribers()
        
        # Set up publishers after a small delay
        timer = self.create_timer(
            0.1, self._setup_publishers, callback_group=self.timer_cb_group)
        self.node_timers.append(timer)
        
        # Cache transforms after a delay
        timer = self.create_timer(
            0.5, self.cache_transforms, callback_group=self.timer_cb_group)
        self.node_timers.append(timer)
    
    def create_transform_retry_timer(self, source, target):
        """
        Create a timer to retry a specific transform cache operation.
        If a transform (coordinate conversion) isn't available at startup, we schedule a retry after a short delay.
        This helps ensure the system can recover if something isn't ready right away.
        """
        retry_timer = None  # Declare in outer scope

        def transform_retry_callback():
            nonlocal retry_timer
            # Remove this timer first
            if retry_timer in self.node_timers:
                self.node_timers.remove(retry_timer)
            self.destroy_timer(retry_timer)

            # Now retry the transform
            self.get_logger().info(f"Retrying cache for transform {source} → {target}")
            try:
                transform = self.tf_buffer.lookup_transform(
                    target, source, 
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.5)
                )

                # Store in cache
                cache_key = f"{source}_{target}"
                self.cached_transforms[cache_key] = transform
                self.transform_timestamps[cache_key] = time.time()

                self.get_logger().info(f"Successfully cached transform on retry: {source} → {target}")
                self.transform_published_successfully = True

            except Exception as e:
                self.get_logger().warn(f"Retry failed for transform {source} → {target}: {str(e)}")

        # Create the timer with the dedicated callback
        retry_timer = self.create_timer(
            2.0,  # Wait 2 seconds before retry
            transform_retry_callback,
            callback_group=self.timer_cb_group
        )
        self.node_timers.append(retry_timer)
        return retry_timer

    def cache_transforms(self):
        """
        Cache static transforms with long TTL for the Raspberry Pi 5.
        Transforms are used to convert positions between different coordinate frames (like camera and LIDAR).
        By caching them, we avoid having to look them up every time, which saves time and CPU.
        """
        # Destroy the timer once called
        timer_index_to_remove = None
        for i, timer in enumerate(self.node_timers):
            if hasattr(timer, 'callback') and timer.callback == self.cache_transforms:
                timer_index_to_remove = i
                break
        
        if timer_index_to_remove is not None:
            self.destroy_timer(self.node_timers[timer_index_to_remove])
            self.node_timers.pop(timer_index_to_remove)
        
        try:
            # Define important transform pairs - these are static in our setup
            transform_pairs = [
                ('ascamera_color_0', 'lidar_frame'),
                ('lidar_frame', 'ascamera_color_0'),
                ('base_link', 'lidar_frame'),
                ('lidar_frame', 'base_link')
            ]
            
            # Cache each transform
            for source, target in transform_pairs:
                try:
                    # Use a timeout to avoid blocking if transform isn't available
                    transform = self.tf_buffer.lookup_transform(
                        target, source, 
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.5)
                    )
                    
                    # Store in cache with very long TTL since transforms are static
                    cache_key = f"{source}_{target}"
                    self.cached_transforms[cache_key] = transform
                    # Set initial timestamp (for static transforms this will rarely change)
                    self.transform_timestamps[cache_key] = time.time()
                    
                    # Pre-compute and cache transform matrices for faster 3D calculations
                    if source == 'ascamera_color_0' and target == 'lidar_frame':
                        self._precompute_transform_matrix(transform, f"{source}_{target}_matrix")
                    
                    # Log success (throttled)
                    self.throttled_log(
                        f"Cached transform: {source} → {target}: "
                        f"translation=({transform.transform.translation.x:.3f}, "
                        f"{transform.transform.translation.y:.3f}, "
                        f"{transform.transform.translation.z:.3f})",
                        key="transform_cache",
                        min_interval=30.0
                    )
                except Exception as e:
                    self.throttled_log(
                        f"Failed to cache transform {source} → {target}: {str(e)}",
                        key="transform_cache_fail",
                        min_interval=10.0,
                        level="warn"
                    )
                    # Schedule a retry for transforms that failed
                    # Only do this for critical transforms
                    if (source == 'ascamera_color_0' and target == 'lidar_frame') or \
                    (source == 'lidar_frame' and target == 'ascamera_color_0'):
                        self.get_logger().info(f"Scheduling retry for transform {source} → {target}")                        
                        # Capture current source and target values to use in the callback
                        s, t = source, target
                        retry_timer = self.create_transform_retry_timer(s, t)
                        self.node_timers.append(retry_timer)
                    continue
            
            self.transform_published_successfully = True
            self.get_logger().info("Transform caching completed")
            
        except Exception as e:
            self.throttled_log(
                f"Transform caching error: {str(e)}",
                key="transform_error",
                min_interval=30.0,
                level="error"
            )
            # Schedule a retry for the entire caching process
            cache_retry_timer = self.create_timer(
                3.0,  # Wait 3 seconds before retry
                lambda _: self.retry_all_transform_caches(),  # Use underscore to indicate unused parameter
                callback_group=self.timer_cb_group
            )
            self.node_timers.append(cache_retry_timer)
    
    def _precompute_transform_matrix(self, transform, cache_key):
        """
        Pre-compute transformation matrix for a transform and cache it.
        This allows us to quickly convert points between coordinate frames using matrix multiplication, which is much faster than recalculating every time.
        
        ---
        TEXTBOOK EXPLANATION FOR STUDENTS:
        Why do we need transformations in robotics?
        - On a robot, sensors like cameras and LIDARs are mounted at different physical locations and orientations.
        - Each sensor "sees" the world from its own point of view (called a coordinate frame).
        - To combine data from different sensors, we need to convert (transform) points from one frame to another.
        - For example, if the camera sees a ball at (x, y, z) in its frame, we need to know where that is in the LIDAR's frame to compare or fuse the data.
        
        What is a transformation matrix?
        - A transformation matrix is a mathematical tool that combines rotation and translation.
        - It lets us convert a point from one coordinate frame to another using matrix multiplication.
        - In 3D, we use a 4x4 matrix (homogeneous coordinates) to handle both rotation and translation in one step.
        
        How does this function work?
        1. **Extract translation:**
           - Translation is the shift in position between the two frames (e.g., the camera is 10cm to the right of the LIDAR).
           - In code: `tx`, `ty`, `tz` are the translation components.
        2. **Extract rotation (quaternion):**
           - Rotation describes how the two frames are turned relative to each other (e.g., the camera is tilted up).
           - Quaternions (`qx`, `qy`, `qz`, `qw`) are a way to represent 3D rotations without gimbal lock.
        3. **Convert quaternion to rotation matrix:**
           - The code computes the 3x3 rotation part of the 4x4 matrix using the quaternion values.
           - This math fills in the top-left 3x3 part of the matrix (see the code for how each element is calculated).
        4. **Fill in translation:**
           - The translation values go in the last column of the matrix (except the bottom row).
        5. **Result:**
           - The final 4x4 matrix can be used to transform any 3D point from the camera frame to the LIDAR frame (or vice versa).
           - To transform a point, you multiply the matrix by the point (in homogeneous coordinates).
        6. **Cache the matrix:**
           - The matrix is stored for fast reuse, so we don't have to recalculate it every time.
        
        Intuition:
        - Imagine you are standing at the camera, and you want to tell your friend at the LIDAR where the ball is.
        - You need to account for how far away your friend is (translation) and which way they are facing (rotation).
        - The transformation matrix does all this math for you!
        
        This is a fundamental concept in robotics and computer vision, and mastering it will help you work with any multi-sensor robot.
        """
        try:
            # Extract translation
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z

            # Extract rotation quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w

            # Normalize quaternion
            norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if norm > 0.001:
                qw /= norm
                qx /= norm
                qy /= norm
                qz /= norm

            # Calculate rotation matrix elements
            xx = qx * qx
            xy = qx * qy
            xz = qx * qz
            xw = qx * qw
            yy = qy * qy
            yz = qy * qz
            yw = qy * qw
            zz = qz * qz
            zw = qz * qw

            # Create transformation matrix (4x4)
            matrix = np.eye(4, dtype=np.float32)
            
            # Rotation part (top-left 3x3)
            matrix[0, 0] = 1 - 2 * (yy + zz)
            matrix[0, 1] = 2 * (xy - zw)
            matrix[0, 2] = 2 * (xz + yw)
            matrix[1, 0] = 2 * (xy + zw)
            matrix[1, 1] = 1 - 2 * (xx + zz)
            matrix[1, 2] = 2 * (yz - xw)
            matrix[2, 0] = 2 * (xz - yw)
            matrix[2, 1] = 2 * (yz + xw)
            matrix[2, 2] = 1 - 2 * (xx + yy)
            
            # Translation part (top-right 3x1)
            matrix[0, 3] = tx
            matrix[1, 3] = ty
            matrix[2, 3] = tz
            
            # Store in cache
            self.cached_transforms[cache_key] = matrix
            
            # Also cache the inverse matrix
            inv_matrix = np.linalg.inv(matrix)
            self.cached_transforms[f"{cache_key}_inv"] = inv_matrix
            
            # Log success
            self.get_logger().info(f"Pre-computed transform matrix for {cache_key}")
            
        except Exception as e:
            self.get_logger().error(f"Error pre-computing transform matrix: {str(e)}")
    
    def retry_transform_cache(self, source, target, timer=None):
        """
        Retry caching a specific transform that failed earlier.
        This is a recovery mechanism to make the system more robust if something goes wrong at startup.
        """
        # Find and remove any timer with this callback
        timers_to_remove = []
        for i, t in enumerate(self.node_timers):
            if hasattr(t, 'callback') and t.callback.__name__ == '<lambda>' and t.callback.__code__.co_freevars:
                timers_to_remove.append(i)
        
        # Remove timers in reverse order to avoid index issues
        for i in sorted(timers_to_remove, reverse=True):
            self.destroy_timer(self.node_timers[i])
            self.node_timers.pop(i)
                
        try:
            self.get_logger().info(f"Retrying cache for transform {source} → {target}")
            # Use a timeout to avoid blocking if transform isn't available
            transform = self.tf_buffer.lookup_transform(
                target, source, 
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.5)
            )
            
            # Store in cache
            cache_key = f"{source}_{target}"
            self.cached_transforms[cache_key] = transform
            # Add timestamp for cache management
            self.transform_timestamps[cache_key] = time.time()
            
            # Pre-compute matrix for key transforms
            if source == 'ascamera_color_0' and target == 'lidar_frame':
                self._precompute_transform_matrix(transform, f"{source}_{target}_matrix")
            
            self.get_logger().info(f"Successfully cached transform on retry: {source} → {target}")
            self.transform_published_successfully = True
            
        except Exception as e:
            self.get_logger().warn(f"Retry failed for transform {source} → {target}: {str(e)}")

    def retry_all_transform_caches(self, timer=None):
        """
        Retry the entire transform caching process.
        If multiple transforms failed, this method tries to cache all of them again.
        """
        # Remove all lambda timers
        timers_to_remove = []
        for i, t in enumerate(self.node_timers):
            if hasattr(t, 'callback') and t.callback.__name__ == '<lambda>':
                timers_to_remove.append(i)
        
        # Remove timers in reverse order to avoid index issues
        for i in sorted(timers_to_remove, reverse=True):
            self.destroy_timer(self.node_timers[i])
            self.node_timers.pop(i)
        
        # If timer was passed, remove it too
        if timer in self.node_timers:
            self.node_timers.remove(timer)
            self.destroy_timer(timer)
        
        # Just call the main caching function again
        self.get_logger().info("Retrying all transform caches")
        self.cache_transforms()
    
    def clean_transform_cache(self):
        """
        Periodically clean the transform cache to prevent memory growth.
        This keeps the program from using too much memory over time by removing old or unused transforms.
        """
        # This function mainly handles edge cases, as most transforms are static
        if len(self.cached_transforms) > 30:  # Higher limit for Pi 5 with 16GB RAM
            # Keep only the most used transforms
            current_time = time.time()
            # Remove unused transforms older than 1 hour (much longer for static transforms)
            old_keys = []
            
            if not hasattr(self, 'transform_timestamps'):
                self.transform_timestamps = {}
                return
                
            for key in self.cached_transforms:
                if key not in self.transform_timestamps:
                    self.transform_timestamps[key] = current_time
                elif current_time - self.transform_timestamps[key] > 3600:  # 1 hour
                    # Skip fundamental transforms - never expire them
                    if any(k in key for k in ['ascamera_color_0_lidar_frame', 'lidar_frame_ascamera_color_0']):
                        self.transform_timestamps[key] = current_time
                        continue
                    old_keys.append(key)
            
            # Remove old transforms
            for key in old_keys:
                del self.cached_transforms[key]
                if key in self.transform_timestamps:
                    del self.transform_timestamps[key]
            
            self.throttled_log(
                f"Cleaned transform cache, removed {len(old_keys)} old transforms",
                key="cache_cleanup",
                min_interval=300.0
            )
    
    def _load_performance_config(self):
        """
        Load performance-related configuration.
        This method reads settings from the configuration file to control how the node adapts to system load (like CPU usage).
        """
        perf_config = self.config.get('performance', {})
        
        # Performance adaptation settings
        self.adaptive_processing = perf_config.get('adaptive_processing', True)
        self.high_load_threshold = perf_config.get('high_load_threshold', 90.0)  # CPU %
        self.low_load_threshold = perf_config.get('low_load_threshold', 50.0)    # CPU %
        
        # Processing settings
        self.max_point_limit = perf_config.get('max_point_limit', 500)
        self.dynamic_ransac_iterations = perf_config.get('dynamic_ransac_iterations', True)
        self.min_ransac_iterations = perf_config.get('min_ransac_iterations', 10)
        
        # Timer frequencies
        self.diagnostics_interval_normal = self.config.get('diagnostics', {}).get('publish_interval', 3.0)
        self.diagnostics_interval_high_load = perf_config.get('diagnostics_interval_high_load', 10.0)
        
        # System resources
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        self.performance_mode = "NORMAL"  # Can be "NORMAL", "EFFICIENT", "MINIMAL"
        
        # Resource monitoring timer - check every 5 seconds
        timer = self.create_timer(
            5.0, self.monitor_resources, callback_group=self.timer_cb_group)
        self.node_timers.append(timer)
        
        # Initialize throttled logging helper
        self._last_throttled_logs = {}
        
        # Configure logging thresholds
        self.configure_logging()
    
    def configure_logging(self):
        """
        Configure logging levels based on performance settings.
        This lets us control how much information is printed to the console, which can help with debugging or reduce CPU usage.
        """
        # Set default logging level
        if self.performance_mode == "MINIMAL":
            self.get_logger().set_level(LoggingSeverity.WARN)
        elif self.performance_mode == "EFFICIENT":
            self.get_logger().set_level(LoggingSeverity.INFO)
        else:
            self.get_logger().set_level(LoggingSeverity.DEBUG)
    
    def _init_state(self):
        """
        Initialize internal state tracking with optimized data structures.
        This sets up all the variables and buffers needed to keep track of LIDAR scans, positions, errors, and performance metrics.
        """
        # Scan data
        self.latest_scan = None
        self.scan_timestamp = None
        self.scan_frame_id = None
        self.points_array = None
        
        # Performance tracking
        self.start_time = time.time()
        self.processed_scans = 0
        self.successful_detections = 0
        
        # Use LightweightBuffer instead of deque
        self.detection_times = LightweightBuffer(max_size=20)
        # Detection sources
        self.yolo_detections = 0
        
        # Position tracking with LightweightBuffer
        self.position_history = LightweightBuffer(max_size=10)
        self.previous_ball_position = None
        self.consecutive_failures = 0
        self.last_successful_detection_time = 0
        self.predicted_position = None
        
        # Initialize ground position filter for shared ground movement tracking
        self.position_filter = GroundPositionFilter()
        
        # Health monitoring with optimized buffers
        self.lidar_health = 1.0
        self.detection_health = 1.0
        self.detection_latency = 0.0
        
        # Use LightweightBuffer for errors
        self.errors = LightweightBuffer(max_size=10)
        self.last_error_time = 0
        
        # Transform publishing tracking
        self.transform_publish_attempts = 0
        self.transform_publish_successes = 0
        
        # Performance metrics
        self.processing_skips = 0
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        
        # Motion tracking
        self.current_velocity = 0.0
        self.last_position = None
        self.last_position_time = time.time()
    
    def check_transform(self):
        """
        Periodically check if transform is available in TF tree.
        This helps ensure that coordinate conversions between camera and LIDAR are working, and logs any problems for debugging.
        """
        try:
            test_time = rclpy.time.Time()
            
            # First check cached transforms
            direct_transform_available = False
            
            if 'ascamera_color_0_lidar_frame' in self.cached_transforms:
                direct_transform_available = True
                # Update the last used timestamp for cache management
                if hasattr(self, 'transform_timestamps'):
                    self.transform_timestamps['ascamera_color_0_lidar_frame'] = time.time()
            else:
                # Try to get from TF tree
                direct_transform_available = self.tf_buffer.can_transform(
                    "lidar_frame",           # Target frame
                    "ascamera_color_0",      # Source frame (camera)
                    test_time,
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
            
            if direct_transform_available:
                # Try to actually get the transform (direct or through chain)
                try:
                    # First check cache
                    transform = None
                    cache_key = 'ascamera_color_0_lidar_frame'
                    
                    if cache_key in self.cached_transforms:
                        transform = self.cached_transforms[cache_key]
                        # Update last used timestamp
                        self.transform_timestamps[cache_key] = time.time()
                    else:
                        transform = self.tf_buffer.lookup_transform(
                            "lidar_frame",
                            "ascamera_color_0",
                            test_time
                        )
                        # Cache the transform and pre-compute matrix
                        self.cached_transforms[cache_key] = transform
                        self._precompute_transform_matrix(transform, f"{cache_key}_matrix")
                        self.transform_timestamps[cache_key] = time.time()
                    
                    self.transform_published_successfully = True
                    self.transform_publish_successes += 1
                    
                    # Throttled logging
                    self.throttled_log(
                        f"Transform check: transform from ascamera_color_0 to lidar_frame is available. "
                        f"Translation=[{transform.transform.translation.x:.4f}, "
                        f"{transform.transform.translation.y:.4f}, "
                        f"{transform.transform.translation.z:.4f}]",
                        key="transform_check",
                        min_interval=30.0
                    )
                    
                except Exception as e:
                    self.throttled_log(
                        f"Cannot lookup transform despite availability check: {str(e)}",
                        key="transform_error",
                        min_interval=10.0,
                        level="error"
                    )
                    direct_transform_available = False
            
            if not direct_transform_available:
                self.throttled_log(
                    "Transform check: transform is NOT available",
                    key="transform_missing",
                    min_interval=10.0,
                    level="warn"
                )
                
        except Exception as e:
            self.throttled_log(
                f"Error checking transform: {str(e)}",
                key="transform_check_error",
                min_interval=30.0,
                level="error"
            )
    
    def _load_basketball_parameters(self):
        """
        Load basketball physical parameters from config.
        Reads the size and other properties of the basketball from the configuration file, so the detection algorithms know what to look for.
        """
        # Get basketball configuration
        basketball_config = self.config.get('basketball', {})
        
        # Core parameters - ensure basketball sized (9 inch diameter)
        self.ball_radius = basketball_config.get('radius', 0.1143)  # 4.5 inches (9 inch diameter)
        self.max_distance = basketball_config.get('max_distance', 0.2)
        self.min_points = basketball_config.get('min_points', 3)
        
        self.detection_samples = basketball_config.get('detection_samples', 30)
        
        # Quality thresholds
        quality_thresholds = basketball_config.get('quality_threshold', {})
        self.quality_low = quality_thresholds.get('low', 0.35)
        self.quality_medium = quality_thresholds.get('medium', 0.6)
        self.quality_high = quality_thresholds.get('high', 0.8)
        
        # Physical measurements - matching basketball & setup
        physical = self.config.get('physical_measurements', {})
        self.lidar_height = physical.get('lidar_height', 0.1524)  # 6 inches
        # Ball center is always 5 inches above ground (radius) for a basketball rolling on floor
        self.ball_center_height = physical.get('ball_center_height', 0.127)  # 5 inches
        
        # Detection reliability
        reliability = self.config.get('detection_reliability', {})
        # Increased from default 0.5 to improve reliability for larger basketball
        self.min_reliable_distance = reliability.get('min_reliable_distance', 0.8)
        self.publish_unreliable = reliability.get('publish_unreliable', True)
        
        # RANSAC parameters
        ransac_config = self.config.get('ransac', {})
        self.ransac_enabled = ransac_config.get('enabled', True)
        self.ransac_max_iterations = ransac_config.get('max_iterations', 30)
        self.ransac_inlier_threshold = ransac_config.get('inlier_threshold', 0.02)
        self.ransac_min_inliers = ransac_config.get('min_inliers', 5)
        
        # For ground movement tracking
        self.ground_movement = True  # Basketball always moves on ground
        self.z_variance_threshold = 0.02  # Small threshold for height variation (2cm)
        
        # New parameter for early stopping in RANSAC
        self.ransac_early_stop_quality = 0.85  # Early stop if quality exceeds this
        self.ransac_min_iterations_before_early_stop = 8  # Minimum iterations before early stopping
    
    def _init_transform_parameters(self):
        """
        Initialize coordinate transform parameters.
        Sets up the default frames and translation/rotation values for converting between camera and LIDAR coordinates.
        """
        transform_config = self.config.get('transform', {})
        
        # Frame IDs - Update default camera frame to match what we need
        self.transform_parent_frame = transform_config.get('parent_frame', 'ascamera_color_0')
        self.transform_child_frame = transform_config.get('child_frame', 'lidar_frame')
        
        # Translation vector
        translation = transform_config.get('translation', {})
        self.transform_translation = {
            'x': translation.get('x', 0.0),
            'y': translation.get('y', 0.0),
            'z': translation.get('z', 0.0)
        }
        
        # Rotation quaternion
        rotation = transform_config.get('rotation', {})
        self.transform_rotation = {
            'x': rotation.get('x', 0.0),
            'y': rotation.get('y', 0.0),
            'z': rotation.get('z', 0.0),
            'w': rotation.get('w', 1.0)
        }
        
        # Log transform interval
        self.last_transform_log = 0.0
    
    def _setup_subscribers(self):
        """
        Set up subscribers with optimized QoS profiles.
        Subscribers listen for messages from other parts of the robot (like LIDAR scans or camera detections).
        QoS (Quality of Service) settings control how messages are delivered and stored.
        """
        # Get topic config
        topics = self.config.get('topics', {})
        input_topics = topics.get('input', {})
        
        # Create optimized QoS profiles
        # For LIDAR scans - best effort, keep only latest
        lidar_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # Only keep the most recent scan
        )
        
        # For YOLO detections - reliable delivery but minimal history
        yolo_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2  # Keep 2 most recent detections
        )
        
        # LIDAR scan subscription - best effort QoS
        lidar_topic = input_topics.get('lidar_scan', '/scan')
        self.scan_subscription = self.create_subscription(
            LaserScan,
            lidar_topic,
            self.scan_callback,
            qos_profile=lidar_qos,
            callback_group=self.subscription_cb_group
        )
        
        # YOLO detection subscription - reliable QoS
        yolo_topic = input_topics.get('yolo_detection', '/basketball/yolo/position')
        self.yolo_subscription = self.create_subscription(
            PointStamped,
            yolo_topic,
            lambda msg: self.sensor_callback(msg, 'yolo'),
            qos_profile=yolo_qos,
            callback_group=self.subscription_cb_group
        )
        
        # YOLO bounding box subscription for 3D position estimation - reliable QoS
        from std_msgs.msg import Float32MultiArray
        yolo_bbox_topic = input_topics.get('yolo_bbox', '/basketball/yolo/bbox')
        self.yolo_bbox_subscription = self.create_subscription(
            Float32MultiArray,
            yolo_bbox_topic,
            self.yolo_bbox_callback,
            qos_profile=yolo_qos,
            callback_group=self.subscription_cb_group
        )
        
        # Initialize bounding box data storage
        self.yolo_bbox_data = {
            'width': 0.0,
            'height': 0.0,
            'timestamp': 0.0
        }
        
        self.get_logger().info("Core subscriptions established with optimized QoS profiles")
    
    def _setup_publishers(self):
        """
        Set up publishers with optimized QoS profiles.
        Publishers send messages to other parts of the robot (like the detected ball position or diagnostics).
        QoS settings help ensure important messages are delivered reliably.
        """
        # Remove the timer that triggered this
        for i, timer in enumerate(self.node_timers):
            if timer.callback == self._setup_publishers:
                self.destroy_timer(timer)
                self.node_timers.pop(i)
                break
        
        # Get topic config
        topics = self.config.get('topics', {})
        output_topics = topics.get('output', {})
        
        # Create optimized QoS profiles
        # For position data - reliable delivery with history
        position_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,  # Late joiners can get last message
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5  # Keep several recent positions
        )
        
        # For debug data - best effort
        debug_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # For diagnostics - best effort
        diag_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Ball position publisher - reliable QoS
        position_topic = output_topics.get('ball_position', '/basketball/lidar/position')
        self.position_publisher = self.create_publisher(
            PointStamped,
            position_topic,
            qos_profile=position_qos
        )
        
        # Debug position publisher - best effort QoS
        debug_topic = output_topics.get('debug_position', '/basketball/lidar/debug_position')
        self.debug_publisher = self.create_publisher(
            PointStamped,
            debug_topic,
            qos_profile=debug_qos
        )
        
        # Diagnostics publisher - best effort QoS
        diag_topic = output_topics.get('diagnostics', '/basketball/lidar/diagnostics')
        self.diagnostics_publisher = self.create_publisher(
            String,
            diag_topic,
            qos_profile=diag_qos
        )
        
        # Publisher for sharing system load with other nodes - best effort QoS
        load_topic = output_topics.get('system_load', '/system/load')
        self.load_publisher = self.create_publisher(
            Float32,
            load_topic,
            qos_profile=debug_qos
        )
        
        self.get_logger().info("Publishers established with optimized QoS profiles")
    
    def throttled_log(self, message, key, min_interval=1.0, level="info"):
        """
        Log with throttling to reduce overhead.
        Throttling means we only print a message if enough time has passed since the last one with the same key.
        This prevents flooding the console with too many messages, which can slow down the program.
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
        
        # Skip most informational logging in MINIMAL mode
        if level == "info" and self.performance_mode == "MINIMAL":
            return
        
        # Log with appropriate level
        if level == "error":
            self.get_logger().error(message)
        elif level == "warn":
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    def monitor_resources(self):
        """
        Monitor system resources and adapt processing accordingly.
        Checks CPU and memory usage, and changes the node's performance mode if the system is overloaded.
        This helps keep the robot running smoothly even if the CPU is very busy.
        """
        try:
            # Get CPU and memory usage
            self.current_cpu_load = psutil.cpu_percent()
            self.current_memory_usage = psutil.virtual_memory().percent
            
            # Determine performance mode based on system load
            if self.current_cpu_load > self.high_load_threshold:
                new_mode = "MINIMAL"
                # Adjust diagnostic timer for high load
                if self.performance_mode != "MINIMAL":
                    self.diagnostics_timer.timer_period_ns = int(self.diagnostics_interval_high_load * 1e9)
            elif self.current_cpu_load > self.low_load_threshold:
                new_mode = "EFFICIENT"
            else:
                new_mode = "NORMAL"
                # Restore normal diagnostic frequency if coming from high load
                if self.performance_mode == "MINIMAL":
                    self.diagnostics_timer.timer_period_ns = int(self.diagnostics_interval_normal * 1e9)
            
            # Log mode changes (throttled)
            if new_mode != self.performance_mode:
                self.throttled_log(
                    f"Performance mode change: {self.performance_mode} -> {new_mode} "
                    f"(CPU: {self.current_cpu_load:.1f}%, Memory: {self.current_memory_usage:.1f}%)",
                    key="performance_mode",
                    min_interval=2.0
                )
                self.performance_mode = new_mode
                
                # Update logging configuration when mode changes
                self.configure_logging()
            
            # Publish system load for other nodes
            load_msg = Float32()
            load_msg.data = float(self.current_cpu_load)
            self.load_publisher.publish(load_msg)
            
        except Exception as e:
            self.throttled_log(
                f"Error monitoring resources: {str(e)}",
                key="resource_error",
                min_interval=30.0,
                level="error"
            )

    def scan_callback(self, msg):
        """
        Process LaserScan messages from the LIDAR.
        
        Converts polar coordinates to Cartesian coordinates with optimized memory operations.
        """
        try:
            # If system is under very high load, we might skip processing some scans
            if self.adaptive_processing and self.performance_mode == "MINIMAL" and self.processed_scans % 2 != 0:
                self.processing_skips += 1
                return
            
            # Store scan metadata
            self.latest_scan = msg
            self.scan_timestamp = msg.header.stamp
            self.scan_frame_id = "lidar_frame"
            
            # Extract scan parameters
            angle_min = msg.angle_min
            angle_increment = msg.angle_increment
            ranges = np.array(msg.ranges, dtype=np.float32)  # Explicit dtype
            
            # Filter out invalid measurements
            valid_indices = np.isfinite(ranges)
            
            # Filter out very short ranges (robot body reflections)
            min_valid_range = 0.05
            valid_indices &= (ranges > min_valid_range)  # In-place operation
            
            # Skip if no valid ranges
            if np.sum(valid_indices) == 0:
                self.throttled_log(
                    "No valid range measurements in scan",
                    key="no_valid_ranges",
                    min_interval=5.0,
                    level="warn"
                )
                self.points_array = None
                return
            
            valid_ranges = ranges[valid_indices]
            
            # Pre-calculated angles array - reuse arrays when possible
            if len(self._angles_array) < len(valid_indices):
                # Resize if needed
                self._angles_array = np.zeros(len(ranges), dtype=np.float32)
            
            # Calculate angles using pre-allocated array
            indices = np.arange(len(ranges), dtype=np.int32)[valid_indices]
            np.multiply(indices, angle_increment, out=self._angles_array[:len(indices)])
            np.add(self._angles_array[:len(indices)], angle_min, out=self._angles_array[:len(indices)])
            angles = self._angles_array[:len(indices)]
            
            # Optimize for high CPU load - limit points processed if needed
            point_limit = self.max_point_limit
            if self.adaptive_processing:
                if self.performance_mode == "EFFICIENT":
                    point_limit = self.max_point_limit // 2
                elif self.performance_mode == "MINIMAL":
                    point_limit = self.max_point_limit // 4
            
            # Sample points if there are too many (improves performance)
            if len(valid_ranges) > point_limit:
                sample_step = len(valid_ranges) // point_limit
                valid_ranges = valid_ranges[::sample_step]
                angles = angles[::sample_step]
            
            # Get a point array from the pool or create a new one if needed
            if self.points_array is not None and len(self.points_array) >= len(valid_ranges):
                # Reuse existing array if it's large enough
                points = self.points_array[:len(valid_ranges)]
            else:
                # Get from pool or create new
                if len(valid_ranges) <= 500:  # Standard pool size
                    points = self.point_pool.get()[:len(valid_ranges)]
                else:
                    # Create a new array for unusually large point sets
                    points = np.zeros((len(valid_ranges), 3), dtype=np.float32)
            
            # Convert to Cartesian coordinates - using direct indexing for speed
            points[:, 0] = valid_ranges * np.cos(angles)  # x
            points[:, 1] = valid_ranges * np.sin(angles)  # y
            points[:, 2] = 0.0  # z (all points on ground plane)
            
            # Store reference to points array
            self.points_array = points
            
            # Update statistics
            self.processed_scans += 1
            
            # Estimate velocity from consecutive scans
            self.update_motion_state()
            
            # Log scan information only in normal mode (throttled)
            if self.performance_mode == "NORMAL":
                self.throttled_log(
                    f"Processed scan #{self.processed_scans} with "
                    f"{len(self.points_array)} valid points",
                    key="scan_processed",
                    min_interval=10.0
                )
            
        except Exception as e:
            self.log_error(f"Error processing scan: {str(e)}")
            self.points_array = None
    
    def sensor_callback(self, msg, source):
        """
        Handle ball detections from camera systems (YOLO).
        Matches camera detections with LIDAR points to improve accuracy.
        If no LIDAR match is found, can fall back to using the camera's estimated 3D position.
        """
        detection_start_time = time.time()
        
        try:
            # Check if we have valid scan data
            if self.latest_scan is None or self.points_array is None or len(self.points_array) == 0:
                if self.performance_mode != "MINIMAL":  # Skip log in minimal mode
                    self.throttled_log(
                        f"Waiting for scan data for {source} detection",
                        key=f"{source}_waiting",
                        min_interval=3.0
                    )
                return
            
            # Extract camera detection info
            x_2d = msg.point.x
            y_2d = msg.point.y
            confidence = msg.point.z if hasattr(msg.point, 'z') else 0.8
            
            # Get camera frame - need to know which coordinate system the point is in
            camera_frame = msg.header.frame_id
            if not camera_frame:
                camera_frame = "ascamera_color_0"  # Default to standard camera frame
            
            # Get 3D point estimate from YOLO bbox if available
            estimated_3d_point = None
            if source == "yolo" and hasattr(self, 'yolo_bbox_data') and self.yolo_bbox_data.get('timestamp', 0) > time.time() - 1.0:
                bbox_width = self.yolo_bbox_data.get('width', 0)
                bbox_height = self.yolo_bbox_data.get('height', 0)
                
                if bbox_width > 0 and bbox_height > 0:
                    estimated_3d_point = self.estimate_3d_from_2d(msg, bbox_width, bbox_height)
                    
                    if estimated_3d_point is not None and self.performance_mode != "MINIMAL":
                        self.throttled_log(
                            f"Using estimated 3D position from {source.upper()} 2D: "
                            f"({estimated_3d_point[0]:.2f}, {estimated_3d_point[1]:.2f}, {estimated_3d_point[2]:.2f})",
                            key=f"{source}_3d_est",
                            min_interval=1.0
                        )
            
            # Find basketball in LIDAR data using the estimated 3D point as seed
            ball_results = None
            if estimated_3d_point is not None:
                ball_results = self.find_basketball_ransac(estimated_3d_point)
            
            # If no ball found with RANSAC but we have an estimated 3D position from bbox,
            # directly use that instead of relying only on LIDAR points
            if (not ball_results or len(ball_results) == 0) and estimated_3d_point is not None:
                if self.performance_mode != "MINIMAL":
                    self.throttled_log(
                        "No matching ball found with RANSAC, using estimated 3D position directly",
                        key="use_estimate_direct",
                        min_interval=1.0
                    )
                
                # Calculate a default quality score based on confidence
                quality = 0.6  # Base quality for bbox-derived positions
                if hasattr(msg.point, 'z'):
                    # Adjust quality based on confidence if available (0.0-1.0)
                    quality = min(0.9, quality + msg.point.z * 0.3)
                
                # Publish the estimated position
                self.publish_ball_position(
                    estimated_3d_point,  # Use the estimated 3D point
                    10,                  # Default cluster size
                    quality,             # Quality score
                    f"{source.upper()}_3D_EST",  # Mark as estimated
                    msg.header.stamp     # Use original timestamp
                )
                return
            
            # Process the best detected ball (if any)
            if ball_results and len(ball_results) > 0:
                # Get the best match
                best_match = ball_results[0]
                center, cluster_size, circle_quality = best_match
                
                # Track YOLO detections
                if source == "yolo":
                    self.yolo_detections += 1
                
                # Publish ball position
                self.publish_ball_position(center, cluster_size, circle_quality, source.upper(), msg.header.stamp)
            else:
                if self.performance_mode != "MINIMAL":  # Skip log in minimal mode
                    self.throttled_log(
                        f"No matching ball found for {source.upper()} detection",
                        key=f"{source}_no_match",
                        min_interval=1.0
                    )
                self.consecutive_failures += 1
            
        except Exception as e:
            self.log_error(f"Error processing {source} detection: {str(e)}")
        
        # Log processing time
        processing_time = (time.time() - detection_start_time) * 1000  # in ms
        self.detection_times.add(time.time(), processing_time)
        self.detection_latency = processing_time
        
        # Throttled logging for processing time
        if self.performance_mode != "MINIMAL":
            self.throttled_log(
                f"{source.upper()} processing took {processing_time:.2f}ms",
                key=f"{source}_processing_time",
                min_interval=5.0
            )
    
    def find_basketball_ransac(self, camera_seed_point=None):
        """
        Find a basketball in LIDAR data using RANSAC for robust circle fitting.
        Optimized for a basketball (9-inch diameter) rolling on the ground.
        
        ---
        TEXTBOOK EXPLANATION FOR STUDENTS:
        This function uses a famous algorithm called RANSAC (Random Sample Consensus) to find circles in noisy data.
        Imagine you have a bunch of points from a LIDAR scan, and you want to find which ones form a circle (the basketball).
        But some points are just noise (not part of the ball). RANSAC helps us find the best circle by:
        1. Randomly picking 3 points (since 3 points define a circle).
        2. Fitting a circle through those 3 points.
        3. Counting how many other points are close to that circle (these are called "inliers").
        4. Repeating this many times, keeping the circle with the most inliers and best fit.
        5. The best circle is our detected basketball!
        
        Why do we use RANSAC? Because it's robust to outliers (bad points), so even if there is a lot of noise, it can still find the real circle.
        
        Args:
            camera_seed_point: Optional point in LIDAR frame transformed from camera detection
        Returns:
            list: List of (center, cluster_size, quality) tuples for detected basketballs
        """
        if self.points_array is None or len(self.points_array) == 0:
            return []
        
        # Create seed points for RANSAC
        seed_points = []
        
        # If camera detection provided a transformed point, prioritize it
        filtered_points = None
        
        if camera_seed_point is not None and len(camera_seed_point) >= 2:
            
            # Use only x,y coordinates for 2D search in LIDAR data
            seed_points.append([camera_seed_point[0], camera_seed_point[1], 0])
            
            # Convert YOLO's (x, y) to polar coordinates for filtering
            estimated_x = camera_seed_point[0]
            estimated_y = camera_seed_point[1]
            r_est = math.sqrt(estimated_x**2 + estimated_y**2)
            theta_est = math.atan2(estimated_y, estimated_x)
            
            # Filter LIDAR points by distance and angle - use pre-allocated arrays
            px = self.points_array[:, 0]
            py = self.points_array[:, 1]
            
            # Get a distances array from the pool or reuse existing
            if len(self._distances_array) >= len(px):
                distances = self._distances_array[:len(px)]
            else:
                # Get from pool or create new
                distances_array = self.distance_pool.get()
                if len(distances_array) >= len(px):
                    distances = distances_array[:len(px)]
                else:
                    # Create new if pool object is too small
                    distances = np.zeros(len(px), dtype=np.float32)
            
            # Calculate distances in-place
            np.sqrt(px**2 + py**2, out=distances)
            
            angles = np.arctan2(py, px)
            
            # Set tolerances - adjust these based on motion state if available
            distance_tolerance = 0.3  # meters
            angular_tolerance = math.radians(15)  # 15 degrees in radians
            
            # Adjust tolerances based on motion state
            if hasattr(self, 'motion_manager'):
                motion_state = self.motion_manager.current_state
                if motion_state == MotionStateManager.STATIONARY:
                    distance_tolerance = 0.2  # Tighter tolerance for stationary
                    angular_tolerance = math.radians(10)
                elif motion_state == MotionStateManager.MEDIUM_FAST:
                    distance_tolerance = 0.4  # Wider tolerance for fast movement
                    angular_tolerance = math.radians(20)
            
            # Get mask from pool or create new
            if len(self._inlier_mask) >= len(px):
                mask = self._inlier_mask[:len(px)]
            else:
                # Get from pool or create new
                mask_array = self.mask_pool.get()
                if len(mask_array) >= len(px):
                    mask = mask_array[:len(px)]
                else:
                    # Create new if needed
                    mask = np.zeros(len(px), dtype=bool)
            
            # Apply distance filter (in-place)
            valid_dist = (distances >= (r_est - distance_tolerance)) & (distances <= (r_est + distance_tolerance))
            
            # Apply angle filter (in-place)
            delta = np.abs(angles - theta_est)
            delta = np.where(delta > math.pi, 2*math.pi - delta, delta)  # handle angle wrap-around
            valid_angle = delta <= angular_tolerance
            
            # Combine both masks (in-place)
            np.logical_and(valid_dist, valid_angle, out=mask[:len(valid_dist)])
            
            # Extract filtered points
            filtered_indices = np.where(mask[:len(valid_dist)])[0]
            if len(filtered_indices) > 0:
                # Get point array from pool
                if len(filtered_indices) <= 500:  # Standard pool size
                    filtered_points_array = self.point_pool.get()
                    filtered_points = filtered_points_array[:len(filtered_indices)]
                else:
                    # Create new for large sets
                    filtered_points = np.zeros((len(filtered_indices), 3), dtype=np.float32)
                
                # Copy selected points
                for i, idx in enumerate(filtered_indices):
                    filtered_points[i, 0] = self.points_array[idx, 0]
                    filtered_points[i, 1] = self.points_array[idx, 1]
                    filtered_points[i, 2] = self.points_array[idx, 2]
            else:
                filtered_points = None
            
            # Log filtering results (throttled)
            if filtered_points is not None and len(filtered_points) >= self.min_points:
                self.throttled_log(
                    f"Filtered LIDAR points from {len(self.points_array)} to {len(filtered_points)} "
                    f"using cone at distance {r_est:.2f}m, angle {math.degrees(theta_est):.1f}°",
                    key="filter_success",
                    min_interval=1.0
                )
            else:
                self.throttled_log(
                    f"Not enough points ({0 if filtered_points is None else len(filtered_points)}) in detection cone. "
                    f"Falling back to standard detection.",
                    key="filter_fallback",
                    min_interval=1.0,
                    level="warn"
                )
                filtered_points = None  # Fall back to standard search
        
        # Include previous ball position if available
        if self.previous_ball_position is not None:
            seed_points.append(self.previous_ball_position)
        
        # Include current points array for seed points if we don't have filtered points yet
        if filtered_points is None and len(self.points_array) > 0:
            # Create a few seed points based on point clusters
            # Focus on points within a reasonable range
            distances = np.sqrt(self.points_array[:, 0]**2 + self.points_array[:, 1]**2)
            
            # Adjust valid range for a larger basketball (can be detected from further away)
            valid_indices = np.where((distances > 0.3) & (distances < 3.0))[0]
            
            if len(valid_indices) > 0:
                # Adjust sample count based on performance mode
                if self.performance_mode == "NORMAL":
                    sample_count = min(8, len(valid_indices))
                elif self.performance_mode == "EFFICIENT":
                    sample_count = min(4, len(valid_indices))
                else:  # MINIMAL
                    sample_count = min(2, len(valid_indices))
                    
                indices = np.random.choice(valid_indices, sample_count, replace=False)
                for idx in indices:
                    seed_points.append(self.points_array[idx])
        
        # Best result tracking
        best_center = None
        best_inlier_count = 0
        best_quality = 0
        
        # Try RANSAC with each seed point
        for seed_point in seed_points:
            # Points to search in - use filtered points if available, otherwise full array
            points_to_search = filtered_points if filtered_points is not None else self.points_array
            
            # Find all points near this seed - using vectorized operations for speed
            distances = np.sqrt(
                (points_to_search[:, 0] - seed_point[0])**2 + 
                (points_to_search[:, 1] - seed_point[1])**2
            )
            
            # For basketball, use a larger search radius based on the basketball's size
            nearby_indices = np.where(distances < self.max_distance * 3)[0]
            
            if len(nearby_indices) < self.min_points:
                continue
                
            # Get points near seed
            nearby_points = points_to_search[nearby_indices]
            
            # Determine iterations based on system load
            max_iterations = self.ransac_max_iterations
            if self.dynamic_ransac_iterations:
                if self.performance_mode == "EFFICIENT":
                    max_iterations = max(self.min_ransac_iterations, self.ransac_max_iterations // 2)
                elif self.performance_mode == "MINIMAL":
                    max_iterations = max(self.min_ransac_iterations, self.ransac_max_iterations // 4)
                    
            # Try fitting a circle using RANSAC
            center, inlier_count, quality = self.ransac_circle_fit(
                nearby_points, 
                max_iterations,
                self.ransac_inlier_threshold
            )
            
            # Check if this is better than current best
            if quality > best_quality and inlier_count >= self.min_points:
                best_center = center
                best_inlier_count = inlier_count
                best_quality = quality
                
                # Early stopping if we have a good enough result
                if (i >= self.ransac_min_iterations_before_early_stop and 
                    quality > self.ransac_early_stop_quality and
                    inlier_count >= len(points) * 0.7):
                    break
            
        # Return result if found
        if best_center is not None and best_quality >= self.quality_low:
            # Store the position for future reference
            self.previous_ball_position = best_center
            
            # Update statistics
            self.consecutive_failures = 0
            self.last_successful_detection_time = time.time()
            
            # Return as list of results (keeping same format as original code)
            return [(best_center, best_inlier_count, best_quality)]
        
        # If no ball found with RANSAC but we have an estimated 3D position from camera,
        # still return that as a fallback with lower quality
        if best_center is None and camera_seed_point is not None:
            if self.performance_mode != "MINIMAL":
                self.throttled_log(
                    "No ball found with RANSAC, using camera seed point as fallback",
                    key="camera_fallback",
                    min_interval=1.0
                )
            
            # Collect any available LiDAR points near the estimated position for fusion
            filtered_lidar_points = []
            if filtered_points is not None and len(filtered_points) > 0:
                # Find points close to the camera seed point for fusion
                px = filtered_points[:, 0]
                py = filtered_points[:, 1]
                distances = np.sqrt((px - camera_seed_point[0])**2 + (py - camera_seed_point[1])**2)
                
                # Collect points within reasonable distance of the seed
                fusion_distance_threshold = self.ball_radius * 3.0  # Use a larger threshold for fusion
                close_indices = np.where(distances < fusion_distance_threshold)[0]
                
                # Add valid points to our collection
                if len(close_indices) > 0:
                    for idx in close_indices:
                        # Add z-coordinate for 3D position
                        point = np.array([filtered_points[idx, 0], filtered_points[idx, 1], self.ball_center_height])
                        filtered_lidar_points.append(point)
            
            # Default fallback is the camera seed point
            fallback_center = np.array([camera_seed_point[0], camera_seed_point[1], self.ball_center_height])
            fallback_quality = self.quality_low * 0.8  # Lower quality than normal detection
            
            # If we have LiDAR points, fuse with YOLO estimate
            if len(filtered_lidar_points) > 0:
                # Compute average of LiDAR points
                lidar_avg = np.mean(np.array(filtered_lidar_points), axis=0)
                
                # Combine with YOLO position using weighted average
                w_yolo = 0.7  # YOLO weight
                w_lidar = 0.3  # LiDAR weight
                fused_position = (
                    w_yolo * fallback_center +
                    w_lidar * lidar_avg
                )
                fallback_center = fused_position
                
                # Slightly higher quality for fused result
                fallback_quality = self.quality_low * 0.9
                
                if self.performance_mode != "MINIMAL":
                    self.throttled_log(
                        f"Using fused YOLO+LiDAR fallback position with {len(filtered_lidar_points)} points: "
                        f"({fallback_center[0]:.2f}, {fallback_center[1]:.2f}, {fallback_center[2]:.2f})",
                        key="fused_fallback",
                        min_interval=1.0
                    )
            else:
                # If no LiDAR points, fallback to YOLO only
                if self.performance_mode != "MINIMAL":
                    self.throttled_log(
                        "Using YOLO-only fallback position (no LiDAR points)",
                        key="yolo_only_fallback",
                        min_interval=1.0
                    )
            
            # Store for future reference
            self.previous_ball_position = fallback_center
            
            # Return as list with minimal inlier count
            return [(fallback_center, self.min_points, fallback_quality)]
        
        return []
    
    def ransac_circle_fit(self, points, max_iterations=30, threshold=0.02):
        """
        Use RANSAC to fit a circle to points, robust to outliers.
        Optimized for performance with explicit data types and minimal allocations.
        
        ---
        TEXTBOOK EXPLANATION FOR STUDENTS:
        RANSAC (Random Sample Consensus) is an algorithm for fitting models (like lines or circles) to data that may have lots of noise or outliers.
        Here, we use it to fit a circle to 2D points from the LIDAR.
        
        The steps are:
        1. Repeat for a number of iterations:
           a. Randomly pick 3 points from the data (since 3 points define a unique circle).
           b. Calculate the circle that passes through these 3 points (see fit_circle).
           c. For every other point, check if it lies close to this circle (within a threshold distance). If so, it's an "inlier".
           d. Count the number of inliers. If this is the best so far, remember this circle.
           e. If the fit is very good (enough inliers and high quality), stop early.
        2. After all iterations, return the best circle found.
        
        Why is this useful? Because in real data, not all points are perfect. Some are noise. RANSAC helps us ignore the noise and find the real shape.
        """
        if points is None or len(points) < 3:
            return None, 0, 0
            
        best_inlier_count = 0
        best_center = None
        best_radius = 0
        best_quality = 0.0
        
        # Limit iterations based on point count for better performance
        actual_iterations = min(max_iterations, len(points) // 2)
        actual_iterations = max(self.min_ransac_iterations, actual_iterations)  # At least min iterations
        
        # Pre-compute point coordinates for vector operations
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Pre-allocate inlier mask for reuse
        inlier_mask = self._inlier_mask[:len(points)]
        
        for i in range(actual_iterations):
            # Randomly sample 3 points
            if len(points) < 3:
                continue
                
            sample_indices = np.random.choice(len(points), 3, replace=False)
            
            # Directly use the pre-allocated circle points array
            for j in range(3):
                self._circle_points[j, 0] = points[sample_indices[j], 0]
                self._circle_points[j, 1] = points[sample_indices[j], 1]
            
            # Fit circle to these points
            try:
                center, radius = self.fit_circle(self._circle_points)
                
                # Skip if radius is too different from expected
                if abs(radius - self.ball_radius) > self.ball_radius * 0.5:
                    continue
                
                # Count inliers using vectorized operations for speed
                distances = np.sqrt(
                    (x_coords - center[0])**2 + 
                    (y_coords - center[1])**2
                )
                
                # Inliers are points close to the expected circle - use in-place operations
                np.abs(distances - radius, out=distances)
                np.less(distances, threshold, out=inlier_mask)
                inlier_count = np.sum(inlier_mask)
                
                # Calculate quality metrics
                inlier_ratio = inlier_count / len(points)
                radius_error = abs(radius - self.ball_radius) / self.ball_radius
                quality = 0.7 * inlier_ratio + 0.3 * (1.0 - min(radius_error, 1.0))
                
                if inlier_count > best_inlier_count or (inlier_count == best_inlier_count and quality > best_quality):
                    best_inlier_count = inlier_count
                    best_center = center
                    best_radius = radius
                    best_quality = quality
                    
                    # Early stopping if we have a good enough result
                    if (i >= self.ransac_min_iterations_before_early_stop and 
                        quality > self.ransac_early_stop_quality and
                        inlier_count >= len(points) * 0.7):
                        break
                
            except Exception:
                continue
        
        if best_center is None:
            return None, 0, 0
        
        # Add z-coordinate for 3D position - reuse existing array
        center_3d = np.array([best_center[0], best_center[1], self.ball_center_height], dtype=np.float32)
        
        return center_3d, best_inlier_count, best_quality
    
    def fit_circle(self, points_2d):
        """
        Fit a circle to 2D points.
        Optimized for basketball size (9-inch diameter) on Raspberry Pi.
        
        ---
        TEXTBOOK EXPLANATION FOR STUDENTS:
        This function finds the best-fitting circle for a set of 2D points. This is useful for detecting round objects like a basketball in LIDAR data.
        
        There are two main cases:
        1. **Exactly 3 points:**
           - Any 3 non-collinear points define a unique circle.
           - We use geometry to solve for the center (x0, y0) and radius r.
           - The math involves solving equations for the perpendicular bisectors of the lines between the points, which intersect at the circle's center.
           - The formulas used here are derived from the general equation of a circle: (x - x0)^2 + (y - y0)^2 = r^2.
           - We solve for x0 and y0 using determinants and then compute the radius as the distance from the center to any of the points.
        2. **More than 3 points:**
           - We use a method called "least squares fitting" to find the circle that best fits all the points, even if they are noisy.
           - The idea is to minimize the sum of squared differences between the distance from each point to the center and the radius.
           - We first center the data (subtract the mean) for numerical stability.
           - We then set up a system of equations based on the expanded circle equation and solve for the center using matrix algebra.
           - The solution involves solving a 2x2 linear system (matrix A and vector B), which gives us the center offset from the mean.
           - The radius is then calculated as the average distance from the center to the points.
        
        Why do we care about centering and least squares?
        - Centering helps avoid numerical errors when points are far from the origin.
        - Least squares gives the "best fit" even if the points are not perfectly on a circle (which is common with real sensor data).
        - This method is fast and works well for detecting round objects in robotics!
        """
        # Need at least 3 points
        if len(points_2d) < 3:
            raise ValueError("Need at least 3 points to fit a circle")
        
        # Direct calculation for exactly 3 points (most efficient)
        if len(points_2d) == 3:
            # Get coordinates
            x1, y1 = points_2d[0]
            x2, y2 = points_2d[1]
            x3, y3 = points_2d[2]
            
            # Calculate circle parameters
            A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2
            B = (x1**2 + y1**2) * (y3 - y2) + (x2**2 + y2**2) * (y1 - y3) + (x3**2 + y3**2) * (y2 - y1)
            C = (x1**2 + y1**2) * (x2 - x3) + (x2**2 + y2**2) * (x3 - x1) + (x3**2 + y3**2) * (x1 - x2)
            
            if abs(A) < 1e-10:
                raise ValueError("Points are collinear, cannot fit circle")
                
            x0 = -B / (2 * A)
            y0 = -C / (2 * A)
            r = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
            
            # For basketball, enforce stricter radius check (basketball has consistent size)
            if abs(r - self.ball_radius) > self.ball_radius * 0.4:
                raise ValueError("Fitted radius too different from basketball radius")
                
            return np.array([x0, y0], dtype=np.float32), r
        
        # For more points, use optimized least squares method
        # Center the data for numerical stability - use mean instead of full centering
        mean_x = np.mean(points_2d[:, 0])
        mean_y = np.mean(points_2d[:, 1])
        x = points_2d[:, 0] - mean_x
        y = points_2d[:, 1] - mean_y
        
        # Simplified matrix calculations - avoid full matrix ops when possible
        sum_x2 = np.sum(x**2)
        sum_y2 = np.sum(y**2)
        sum_xy = np.sum(x*y)
        sum_x3 = np.sum(x**3)
        sum_y3 = np.sum(y**3)
        sum_xy2 = np.sum(x*y**2)
        sum_x2y = np.sum(x**2*y)
        
        # Calculate matrix A and B
        A = np.array([[sum_x2, sum_xy], [sum_xy, sum_y2]], dtype=np.float32)
        B = np.array([sum_x3 + sum_xy2, sum_x2y + sum_y3], dtype=np.float32) / 2
        
        # Solve linear system using direct method (faster than lstsq for 2x2)
        det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
        if abs(det) < 1e-10:
            # Fallback to standard method if matrix is singular
            c = np.linalg.lstsq(A, B, rcond=None)[0]
        else:
            c = np.array([
                (A[1,1]*B[0] - A[0,1]*B[1])/det,
                (A[0,0]*B[1] - A[1,0]*B[0])/det
            ], dtype=np.float32)
        
        # Calculate center and radius
        x0 = c[0] + mean_x
        y0 = c[1] + mean_y
        r = np.sqrt(c[0]**2 + c[1]**2 + (sum_x2 + sum_y2)/len(points_2d))
        
        return np.array([x0, y0], dtype=np.float32), r
    
    def publish_ball_position(self, center, cluster_size, circle_quality, trigger_source, timestamp=None):
        """
        Publish the detected basketball position using reusable message objects.
        Assumes basketball is always on the ground (z-height is ball radius).
        """
        # Always set z to basketball radius since ball rolls on ground
        # This ensures we always assume basketball center is 5 inches above ground
        center[2] = self.ball_center_height
        
        # Calculate distance and reliability
        distance = np.sqrt(center[0]**2 + center[1]**2)
        is_reliable = distance >= self.min_reliable_distance
        
        # Adjust quality based on distance
        if not is_reliable:
            distance_factor = max(0.1, distance / self.min_reliable_distance)
            adjusted_quality = circle_quality * distance_factor
            reliability_text = f"UNRELIABLE ({distance:.2f}m < {self.min_reliable_distance:.1f}m)"
        else:
            adjusted_quality = circle_quality
            reliability_text = "RELIABLE"
        
        # Skip unreliable detections if configured to do so
        if not is_reliable and not self.publish_unreliable:
            self.throttled_log(
                "Skipping publication of unreliable detection",
                key="skip_unreliable",
                min_interval=2.0
            )
            return
        
        # Filter position using the shared ground position filter
        current_time = time.time()
        filtered_position = self.position_filter.update(center, current_time)
        
        # Log the detection - only in normal or efficient modes (throttled)
        if self.performance_mode != "MINIMAL":
            self.throttled_log(
                f"LIDAR: Basketball at ({filtered_position[0]:.2f}, {filtered_position[1]:.2f}, {filtered_position[2]:.2f}) meters | "
                f"Distance: {distance:.2f}m | {reliability_text} | "
                f"Quality: {adjusted_quality:.2f} | Triggered by: {trigger_source}",
                key="ball_position",
                min_interval=0.5
            )
        
        # Reuse position message object instead of creating new one
        # Use original timestamp if provided, otherwise use current time
        if timestamp is not None:
            self._pos_msg.header.stamp = timestamp
        else:
            self._pos_msg.header.stamp = self.get_clock().now().to_msg()
        
        self._pos_msg.header.frame_id = "lidar_frame"
        self._pos_msg.point.x = float(filtered_position[0])
        self._pos_msg.point.y = float(filtered_position[1])
        self._pos_msg.point.z = float(filtered_position[2])
        
        # Publish position
        self.position_publisher.publish(self._pos_msg)
        
        # Update statistics
        self.successful_detections += 1
        
        # With the lock, update position history
        with self.lock:
            self.position_history.add(current_time, filtered_position)
        
        # Update motion from new position
        self.update_velocity_from_positions(filtered_position)
        
        # Publish debug point for visualization occasionally
        if (self.successful_detections % 20 == 0) and (self.performance_mode != "MINIMAL"):
            self.publish_debug_point()
    
    def update_velocity_from_positions(self, new_position):
        """
        Update velocity estimate from position history.
        Used to update the motion state manager.
        
        ---
        TEXTBOOK EXPLANATION FOR STUDENTS:
        This function calculates how fast the basketball is moving (its velocity) using its recent positions.
        
        Why do we care about velocity?
        - In robotics, knowing how fast an object is moving helps us predict where it will be next.
        - Velocity is a key part of tracking and following moving objects.
        
        How is velocity calculated?
        - Velocity is the change in position over time: v = Δx / Δt
        - For 2D movement, we calculate the change in x and y separately, then combine them to get the speed (magnitude).
        - We use the most recent two positions and their timestamps to compute this.
        
        Why do we use smoothing (exponential moving average)?
        - Real sensor data is noisy: small errors or jitters can make the velocity jump around.
        - Smoothing helps us get a more stable estimate by blending the new velocity with the previous estimate.
        - The formula is: v_smoothed = α * v_new + (1 - α) * v_old, where α is a smoothing factor (between 0 and 1).
        - A smaller α means more smoothing (slower to react), a larger α means less smoothing (faster to react).
        
        What are the steps in this function?
        1. If this is the first position, just store it and return (can't calculate velocity yet).
        2. Calculate the time difference (dt) between the new and previous positions.
        3. If dt is too small or too large, skip the update (to avoid errors).
        4. Compute the change in x and y, divide by dt to get velocity components.
        5. Calculate the speed as the magnitude: sqrt(vx^2 + vy^2).
        6. Update the stored position and time.
        7. Apply smoothing to the velocity estimate.
        8. Store the smoothed velocity for use by the motion state manager.
        
        This approach is common in robotics, physics, and engineering for tracking moving objects!
        """
        # Need at least 2 positions to calculate velocity
        if not hasattr(self, 'last_position') or self.last_position is None:
            self.last_position = new_position
            self.last_position_time = time.time()
            return
        
        # Calculate time difference
        current_time = time.time()
        dt = current_time - self.last_position_time
        
        # Avoid division by zero and exclude old measurements
        if dt < 0.001 or dt > 1.0:
            self.last_position = new_position
            self.last_position_time = current_time
            return
        
        # Calculate velocity
        vx = (new_position[0] - self.last_position[0]) / dt
        vy = (new_position[1] - self.last_position[1]) / dt
        velocity = math.sqrt(vx*vx + vy*vy)
        
        # Update last position and time
        self.last_position = new_position
        self.last_position_time = current_time
        
        # Store velocity for motion state update
        if not hasattr(self, 'current_velocity'):
            self.current_velocity = velocity
        else:
            # Apply exponential smoothing
            alpha = 0.3  # Smoothing factor
            self.current_velocity = alpha * velocity + (1 - alpha) * self.current_velocity
    
    def update_motion_state(self):
        """Update motion state based on calculated velocity."""
        if not hasattr(self, 'motion_manager') or not hasattr(self, 'current_velocity'):
            return
        
        # Use the smoothed velocity calculated from position history
        velocity = self.current_velocity
        
        # Update motion state
        state_changed = self.motion_manager.update(velocity)
        
        # Log state changes with throttling
        if state_changed:
            state_age = 0
            if hasattr(self.motion_manager, 'previous_state'):
                state_age = self.motion_manager.get_state_age()
                
            self.throttled_log(
                f"Motion state change: {self.motion_manager.previous_state} → "
                f"{self.motion_manager.current_state} (v={velocity:.3f}m/s, age={state_age:.1f}s)",
                key="motion_state_change",
                min_interval=0.5
            )
    
    def estimate_3d_from_2d(self, detection_msg, bbox_width, bbox_height):
        """
        Estimate a 3D position from a 2D detection and bbox dimensions.
        Similar to the fusion node's implementation but optimized for LIDAR use.
        Uses cached transforms and reused matrices for efficiency.
        
        ---
        TEXTBOOK EXPLANATION FOR STUDENTS:
        This function takes a 2D detection from a camera (like the center and size of a bounding box around the basketball)
        and estimates where the ball is in 3D space (real world coordinates).
        
        The math behind it:
        1. The camera sees the ball as a circle of a certain size (in pixels). The real ball has a known diameter (in meters).
        2. Using the formula: distance = (real_diameter * focal_length) / observed_diameter_in_pixels
           - This comes from the pinhole camera model in geometry.
        3. The center of the bounding box tells us the direction from the camera center to the ball (in image coordinates).
        4. We convert this direction into a 3D vector using the camera's focal length.
        5. We then use the camera's position and orientation (from the transform) to convert this direction into the LIDAR's frame.
        6. Finally, we multiply the direction by the estimated distance to get the 3D position of the ball.
        
        This is a classic example of using geometry and camera calibration to go from 2D images to 3D positions!
        """
        try:
            # Known basketball diameter in meters
            basketball_diameter_meters = self.ball_radius * 2
            
            # Calculate ball diameter using geometric mean instead of max dimension
            ball_diameter_pixels = math.sqrt(bbox_width * bbox_height)
            
            # Calculate distance based on apparent size vs actual size
            focal_length_pixels = 345.58  # Calibrated focal length for camera
            distance = (basketball_diameter_meters * focal_length_pixels) / ball_diameter_pixels
            
            # Enhanced logging with all intermediate values (throttled)
            self.throttled_log(
                f"YOLO bbox: {bbox_width:.1f}x{bbox_height:.1f} pixels | "
                f"Pixel diameter (√w·h): {ball_diameter_pixels:.2f} px | "
                f"Focal length: {focal_length_pixels:.2f} px | "
                f"Ball real diameter: {basketball_diameter_meters:.2f} m | "
                f"Estimated distance: {distance:.2f} m",
                key="bbox_distance",
                min_interval=0.5
            )
            
            # Apply EMA smoothing to the distance
            raw_distance = distance
            if self.smoothed_yolo_distance is None:
                self.smoothed_yolo_distance = raw_distance
            else:
                self.smoothed_yolo_distance = (self.ema_alpha * raw_distance +
                                              (1 - self.ema_alpha) * self.smoothed_yolo_distance)
            distance = self.smoothed_yolo_distance

            # Log both raw and smoothed distance values (throttled)
            self.throttled_log(
                f"YOLO Distance (raw): {raw_distance:.2f} m | YOLO Distance (smoothed): {self.smoothed_yolo_distance:.2f} m",
                key="yolo_distance",
                min_interval=0.5
            )

            # Get camera frame
            camera_frame = detection_msg.header.frame_id or "ascamera_color_0"
            
            # Check transform cache first
            transform = None
            cache_key = f"{camera_frame}_lidar_frame"
            
            if cache_key in self.cached_transforms:
                transform = self.cached_transforms[cache_key]
                # Update last used timestamp
                self.transform_timestamps[cache_key] = time.time()
            else:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        "lidar_frame",
                        camera_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.2)
                    )
                    # Cache for future use
                    self.cached_transforms[cache_key] = transform
                    self.transform_timestamps[cache_key] = time.time()
                except Exception as e:
                    self.throttled_log(
                        f"Failed to lookup transform for 3D estimation: {str(e)}",
                        key="transform_lookup_fail",
                        min_interval=3.0,
                        level="warn"
                    )
                    return None  # Return None if transform isn't available
            
            # Extract camera position in lidar frame
            camera_pos_x = transform.transform.translation.x
            camera_pos_y = transform.transform.translation.y
            camera_pos_z = transform.transform.translation.z
            
            # Get image dimensions
            image_width = 320  # Width of the camera image
            image_height = 320  # Height of the camera image
            image_center_x = image_width / 2
            image_center_y = image_height / 2
            
            # Get detection coordinates
            detection_x = detection_msg.point.x
            detection_y = detection_msg.point.y
            
            # Calculate offsets from center
            offset_x = detection_x - image_center_x
            offset_y = detection_y - image_center_y
            
            # Convert pixel offsets to direction vector using focal length
            camera_dir_z = focal_length_pixels  # Z is forward in camera frame
            camera_dir_x = offset_x             # X is right in camera frame 
            camera_dir_y = offset_y             # Y is down in camera frame
            
            # Normalize the direction vector - avoid division by zero
            dir_magnitude = math.sqrt(camera_dir_x**2 + camera_dir_y**2 + camera_dir_z**2)
            if dir_magnitude > 0.001:
                camera_dir_x /= dir_magnitude
                camera_dir_y /= dir_magnitude
                camera_dir_z /= dir_magnitude
            
            # Extract rotation quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            
            # Convert quaternion to rotation matrix - use pre-allocated matrix
            # First normalize the quaternion
            norm = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
            if norm > 0.001:
                qw /= norm
                qx /= norm
                qy /= norm
                qz /= norm
            
            # Calculate rotation matrix elements
            xx = qx * qx
            xy = qx * qy
            xz = qx * qz
            xw = qx * qw
            yy = qy * qy
            yz = qy * qz
            yw = qy * qw
            zz = qz * qz
            zw = qz * qw
            
            # Fill rotation matrix (top 3x3 of transform matrix)
            self._transform_matrix[0, 0] = 1 - 2 * (yy + zz)
            self._transform_matrix[0, 1] = 2 * (xy - zw)
            self._transform_matrix[0, 2] = 2 * (xz + yw)
            self._transform_matrix[1, 0] = 2 * (xy + zw)
            self._transform_matrix[1, 1] = 1 - 2 * (xx + zz)
            self._transform_matrix[1, 2] = 2 * (yz - xw)
            self._transform_matrix[2, 0] = 2 * (xz - yw)
            self._transform_matrix[2, 1] = 2 * (yz + xw)
            self._transform_matrix[2, 2] = 1 - 2 * (xx + yy)
            
            # Apply rotation to camera direction vector
            ref_dir = np.array([
                self._transform_matrix[0, 0] * camera_dir_x + self._transform_matrix[0, 1] * camera_dir_y + self._transform_matrix[0, 2] * camera_dir_z,
                self._transform_matrix[1, 0] * camera_dir_x + self._transform_matrix[1, 1] * camera_dir_y + self._transform_matrix[1, 2] * camera_dir_z,
                self._transform_matrix[2, 0] * camera_dir_x + self._transform_matrix[2, 1] * camera_dir_y + self._transform_matrix[2, 2] * camera_dir_z
            ])
            
            # Normalize direction vector
            dir_magnitude = np.linalg.norm(ref_dir)
            if dir_magnitude > 0.001:
                ref_dir /= dir_magnitude
            
            # Calculate estimated position in reference frame
            estimated_position = np.array([
                camera_pos_x + distance * ref_dir[0],
                camera_pos_y + distance * ref_dir[1],
                self.ball_center_height  # Always at basketball height above ground
            ], dtype=np.float32)
            
            # Throttled logging
            self.throttled_log(
                f"Estimated 3D from 2D: distance={distance:.2f}m, "
                f"pos=({estimated_position[0]:.2f}, {estimated_position[1]:.2f}, {estimated_position[2]:.2f})",
                key="3d_estimation",
                min_interval=0.5
            )
                
            return estimated_position
            
        except Exception as e:
            self.throttled_log(
                f"Error estimating 3D from 2D: {str(e)}",
                key="3d_est_error",
                min_interval=3.0,
                level="warn"
            )
            return None
    
    def log_error(self, message):
        """
        Log an error and update health status.
        Adds the error to a buffer for diagnostics, and reduces the health score of the LIDAR system.
        """
        # Add to error collection using LightweightBuffer
        current_time = time.time()
        self.errors.add(current_time, message)
        
        # Update health
        self.last_error_time = current_time
        self.lidar_health = max(0.3, self.lidar_health - 0.2)
        
        # Log the error
        self.get_logger().error(f"LIDAR ERROR: {message}")
    
    def yolo_bbox_callback(self, msg):
        """
        Process bounding box information from YOLO detection with optimized storage.
        
        Args:
            msg (Float32MultiArray): Bounding box data formatted as [center_x, center_y, width, height, confidence]
        """
        try:
            # Handle Float32MultiArray format for YOLO
            if hasattr(msg, 'data') and len(msg.data) >= 4:
                # Format: [center_x, center_y, width, height, confidence]
                width = msg.data[2]   # width is the 3rd value (index 2)
                height = msg.data[3]  # height is the 4th value (index 3)
                
                # Store bounding box data with timestamp
                self.yolo_bbox_data = {
                    'width': width,
                    'height': height,
                    'timestamp': time.time()
                }
                
                # Throttled logging to reduce overhead
                self.throttled_log(
                    f"Received YOLO bbox: {width:.1f}x{height:.1f}",
                    key="yolo_bbox",
                    min_interval=1.0
                )
                    
        except Exception as e:
            self.throttled_log(
                f"Error processing YOLO bbox: {str(e)}",
                key="bbox_error",
                min_interval=5.0,
                level="warn"
            )
    
    def publish_diagnostics(self):
        """
        Publish diagnostic information about the node with optimized message reuse.
        Sends a summary of the node's status, performance, and health to other parts of the system for monitoring.
        """
        try:
            # Calculate statistics
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            if elapsed < 0.1:
                return
            
            # Calculate rates
            scan_rate = self.processed_scans / elapsed if elapsed > 0 else 0
            detection_rate = self.successful_detections / elapsed if elapsed > 0 else 0
            
            # Calculate average processing time
            avg_time = 0
            detection_times_latest = None
            
            # Access latest time from LightweightBuffer
            if hasattr(self.detection_times, 'get_latest'):
                detection_times_latest = self.detection_times.get_latest()
                if detection_times_latest is not None:
                    avg_time = detection_times_latest
            
            # Get motion state
            motion_state = "unknown"
            if hasattr(self, 'motion_manager'):
                motion_state = self.motion_manager.current_state
            
            # Create diagnostics message - reuse existing object
            diagnostics = {
                "timestamp": current_time,
                "node": "lidar",
                "uptime_seconds": elapsed,
                "status": "active",
                "performance_mode": self.performance_mode,
                "motion_state": motion_state,
                "system": {
                    "cpu_load": self.current_cpu_load,
                    "memory_usage": self.current_memory_usage
                },
                "health": {
                    "lidar_health": self.lidar_health,
                    "detection_health": self.detection_health,
                    "overall": (self.lidar_health * 0.7 + self.detection_health * 0.3)
                },
                "metrics": {
                    "processed_scans": self.processed_scans,
                    "successful_detections": self.successful_detections,
                    "processing_skips": self.processing_skips,
                    "scan_rate": scan_rate,
                    "detection_rate": detection_rate,
                    "avg_processing_time_ms": avg_time * 1000 if avg_time else 0,
                    "sources": {
                        "yolo_detections": self.yolo_detections
                    }
                },
                "config": {
                    "ball_radius": self.ball_radius,
                    "max_distance": self.max_distance,
                    "min_points": self.min_points
                },
                "transforms": {
                    "camera_frame": "ascamera_color_0",
                    "published_successfully": self.transform_published_successfully,
                    "publish_attempts": self.transform_publish_attempts,
                    "publish_successes": self.transform_publish_successes,
                    "cached_transforms": len(self.cached_transforms) if hasattr(self, 'cached_transforms') else 0
                }
            }
            
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return convert_numpy_types(obj.tolist())
                else:
                    return obj
            
            # Convert all numpy types to Python types
            diagnostics_converted = convert_numpy_types(diagnostics)
            
            # Publish as JSON string - reuse message object
            self._diag_msg.data = json.dumps(diagnostics_converted)
            self.diagnostics_publisher.publish(self._diag_msg)
            
            # Log basic summary - reduced logging in MINIMAL mode with throttling
            log_interval = 5.0 if self.performance_mode == "MINIMAL" else 3.0
            self.throttled_log(
                f"LIDAR: Status: {scan_rate:.1f} scans/sec, "
                f"{detection_rate:.1f} detections/sec, "
                f"Mode: {self.performance_mode}, CPU: {self.current_cpu_load:.1f}%, "
                f"State: {motion_state}",
                key="status_summary",
                min_interval=log_interval
            )
            
        except Exception as e:
            self.log_error(f"Error publishing diagnostics: {str(e)}")
    
    def publish_debug_point(self):
        """
        Publish a debug point for calibration purposes.
        Selects a point from the LIDAR data to help with calibration and visualization.
        Only runs when not in minimal performance mode.
        """
        # Skip in MINIMAL performance mode
        if self.performance_mode == "MINIMAL":
            return
            
        if self.points_array is None or len(self.points_array) == 0:
            return
        
        try:
            # Group points by distance ranges
            points = self.points_array
            distances = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
            
            # Find points in different ranges
            close_indices = np.where((distances >= 0.5) & (distances < 1.0))[0]
            mid_indices = np.where((distances >= 1.0) & (distances < 2.0))[0]
            far_indices = np.where((distances >= 2.0) & (distances < 3.0))[0]
            
            # Select which range to use
            if hasattr(self, 'last_debug_range'):
                if self.last_debug_range == "close" and len(mid_indices) > 0:
                    indices = mid_indices
                    range_name = "mid"
                elif self.last_debug_range == "mid" and len(far_indices) > 0:
                    indices = far_indices
                    range_name = "far"
                elif self.last_debug_range == "far" and len(close_indices) > 0:
                    indices = close_indices
                    range_name = "close"
                # Default if can't follow pattern
                elif len(mid_indices) > 0:
                    indices = mid_indices
                    range_name = "mid"
                elif len(far_indices) > 0:
                    indices = far_indices
                    range_name = "far"
                elif len(close_indices) > 0:
                    indices = close_indices
                    range_name = "close"
                else:
                    # No suitable points
                    return
            else:
                # First run, prefer mid-range
                if len(mid_indices) > 0:
                    indices = mid_indices
                    range_name = "mid"
                elif len(far_indices) > 0:
                    indices = far_indices
                    range_name = "far"
                elif len(close_indices) > 0:
                    indices = close_indices
                    range_name = "close"
                else:
                    # No suitable points
                    return
            
            # Save range for next time
            self.last_debug_range = range_name
            
            # Select a point with good Y variation
            selected_points = points[indices]
            y_values = np.abs(selected_points[:, 1])
            max_y_idx = np.argmax(y_values)
            selected_point = selected_points[max_y_idx]
            
            # Reuse debug point message object
            self._debug_msg.header.stamp = self.get_clock().now().to_msg()
            self._debug_msg.header.frame_id = "lidar_frame"
            self._debug_msg.point.x = float(selected_point[0])
            self._debug_msg.point.y = float(selected_point[1])
            self._debug_msg.point.z = float(self.ball_center_height)  # Set to expected height
            
            # Use debug publisher instead of position publisher
            self.debug_publisher.publish(self._debug_msg)
            
            # Log for calibration (throttled)
            distance = np.sqrt(selected_point[0]**2 + selected_point[1]**2)
            self.throttled_log(
                f"CALIBRATION: Debug point at ({selected_point[0]:.3f}, "
                f"{selected_point[1]:.3f}, {self.ball_center_height:.3f}), "
                f"distance: {distance:.2f}m, range: {range_name}",
                key="calibration_point",
                min_interval=5.0
            )
            
        except Exception as e:
            self.throttled_log(
                f"Error publishing debug point: {str(e)}",
                key="debug_error",
                min_interval=10.0,
                level="error"
            )
    
    def shutdown(self):
        """
        Clean shutdown of the node.
        Destroys timers, clears caches and buffers, and logs that the node has shut down.
        This helps prevent memory leaks and ensures a clean exit.
        """
        # Destroy timers - iterate through the timers list we maintain
        for timer in self.node_timers:
            self.destroy_timer(timer)
        
        # Clear cached data
        if hasattr(self, 'cached_transforms'):
            self.cached_transforms.clear()
        
        # Clear buffers
        if hasattr(self, 'position_history'):
            self.position_history.clear()
        if hasattr(self, 'detection_times'):
            self.detection_times.clear()
        
        # Log shutdown
        self.get_logger().info("LIDAR node shutdown complete")


# Main function with reusable executor
def main(args=None):
    """Main entry point with optimized executor and graceful shutdown."""
    rclpy.init(args=args)
    
    # Create and spin node
    detector = BasketballLidarDetector()
    
    # Use MultiThreadedExecutor with controlled thread count
    # Lower thread count is better for Raspberry Pi to avoid oversubscription
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=2)
    executor.add_node(detector)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        detector.get_logger().info("Shutting down gracefully (Ctrl+C)")
    except Exception as e:
        detector.get_logger().error(f"Error during execution: {str(e)}")
    finally:
        # Clean shutdown
        detector.shutdown()
        detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()