#!/usr/bin/env python3

"""
Basketball Tracking Robot - Optimized LIDAR Detection Node
=========================================================

This node processes 2D LIDAR data to detect a basketball and provide 3D position information.
It correlates LIDAR data with camera-based detections from YOLO.

Key optimizations:
- Lightweight buffer implementation with fixed memory allocation
- Message object reuse to reduce allocations
- Transform caching for efficient lookups
- Motion-aware processing strategies
- Optimized NumPy operations with explicit data types
- Throttled logging to reduce overhead

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

# ROS2 messages
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Float32, Bool
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import PointStamped as TF2PointStamped
import json

import os
# Import GroundPositionFilter for shared ground movement tracking
from ball_chase.config.config_loader import ConfigLoader
from ball_chase.utilities.ground_position_filter import GroundPositionFilter


class LightweightBuffer:
    """Lightweight buffer implementation with fixed memory allocation."""
    
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
    """Efficient motion state management with hysteresis."""
    
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
    A ROS2 node to detect basketballs using a 2D laser scanner.
    
    Correlates LIDAR data with camera detections to provide 3D position
    information for detected basketballs. Optimized for Raspberry Pi with
    memory-efficient operations and adaptive processing.
    """
    
    def __init__(self):
        """Initialize the basketball LIDAR detector node."""
        super().__init__('basketball_lidar_detector')
        
        # Track timers explicitly since Node doesn't have get_timers()
        self.node_timers = []
        
        # Initialize transform timestamps dictionary
        self.transform_timestamps = {}
        
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
        
        # Set up transform cache cleanup timer
        timer = self.create_timer(
            300.0, self.clean_transform_cache, callback_group=self.timer_cb_group)
        self.node_timers.append(timer)
        
        self.get_logger().info("Basketball LIDAR detector initialized with optimized memory management")
        
        # NEW: Create a flag to track successful transforms
        self.transform_published_successfully = False
    
    def _initialize_publisher_objects(self):
        """Initialize publisher message objects for reuse."""
        # Pre-create common message objects that will be reused
        self._pos_msg = PointStamped()
        self._debug_msg = PointStamped()
        self._diag_msg = String()
        self._status_msg = Bool()
        
        # Pre-create markers for visualization if enabled
        if self.config.get('visualization', {}).get('enabled', False):
            self._ball_marker = Marker()
            self._text_marker = Marker()
            self._marker_array = MarkerArray()
    
    def _init_vector_arrays(self):
        """Pre-allocate arrays for vector operations."""
        # Small array for circle fitting
        self._circle_points = np.zeros((3, 2), dtype=np.float32)
        
        # Array for transform calculations
        self._transform_matrix = np.eye(4, dtype=np.float32)
        
        # Arrays for RANSAC
        max_size = self.max_point_limit if hasattr(self, 'max_point_limit') else 500
        self._inlier_mask = np.zeros(max_size, dtype=bool)
        
        # Reusable arrays for distance calculations
        self._distances_array = np.zeros(max_size, dtype=np.float32)
    
    def staged_startup(self):
        """Staged startup to reduce initial CPU load spikes."""
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
    
    def cache_transforms(self):
        """Cache transforms for efficient lookup."""
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
            # Define important transform pairs
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
                    
                    # Store in cache
                    cache_key = f"{source}_{target}"
                    self.cached_transforms[cache_key] = transform
                    # Add timestamp for cache management (using a dictionary)
                    self.transform_timestamps[cache_key] = time.time()
                    
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
                        
                        # Create a one-shot timer with proper callback
                        # Note: Using different variable name for the timer parameter
                        #retry_timer = self.create_timer(
                        #    2.0,  # Wait 2 seconds before retry
                        #    lambda callback_timer, source=s, target=t: self.retry_transform_cache(source, target),
                        #    callback_group=self.timer_cb_group
                        #)
                        retry_timer = self.create_timer(
                            2.0,  # Wait 2 seconds before retry
                            lambda callback_timer, source=s, target=t: self.retry_transform_cache(source, target, callback_timer),
                            callback_group=self.timer_cb_group
                        )
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
            # Using a different name for the timer parameter
            cache_retry_timer = self.create_timer(
                3.0,  # Wait 3 seconds before retry
                lambda _: self.retry_all_transform_caches(),  # Use underscore to indicate unused parameter
                callback_group=self.timer_cb_group
            )
            self.node_timers.append(cache_retry_timer)

    def retry_transform_cache(self, source, target, timer=None):
        """Retry caching a specific transform that failed earlier.
        
        Args:
            source: Source frame
            target: Target frame
            timer: The timer that triggered this callback (optional)
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
            
            self.get_logger().info(f"Successfully cached transform on retry: {source} → {target}")
            self.transform_published_successfully = True
            
        except Exception as e:
            self.get_logger().warn(f"Retry failed for transform {source} → {target}: {str(e)}")
            # Could schedule another retry here if needed

    def retry_all_transform_caches(self, timer=None):
        """Retry the entire transform caching process.
        
        Args:
            timer: The timer that triggered this callback (optional)
        """
        # Find and remove any timer with this callback
        timers_to_remove = []
        for i, t in enumerate(self.node_timers):
            if hasattr(t, 'callback') and t.callback.__name__ == '<lambda>':
                timers_to_remove.append(i)
        
        # Remove timers in reverse order to avoid index issues
        for i in sorted(timers_to_remove, reverse=True):
            self.destroy_timer(self.node_timers[i])
            self.node_timers.pop(i)
                    
        self.get_logger().info("Retrying all transform caches")
        # Just call the main caching function again
        self.cache_transforms()
    
    def clean_transform_cache(self):
        """Periodically clean the transform cache to prevent memory growth."""
        if len(self.cached_transforms) > 20:  # Arbitrary limit
            # Keep only the most used transforms
            current_time = time.time()
            # Remove transforms older than 5 minutes
            old_keys = []
            
            if not hasattr(self, 'transform_timestamps'):
                self.transform_timestamps = {}
                return
                
            for key in self.cached_transforms:
                if key not in self.transform_timestamps:
                    self.transform_timestamps[key] = current_time
                elif current_time - self.transform_timestamps[key] > 300:  # 5 minutes
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
        """Load performance-related configuration."""
        perf_config = self.config.get('performance', {})
        
        # Visualization settings (disabled by default)
        viz_config = self.config.get('visualization', {})
        self.visualization_enabled = viz_config.get('enabled', False)
        
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
    
    def _init_state(self):
        """Initialize internal state tracking with optimized data structures."""
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
        
        # NEW: Transform publishing tracking
        self.transform_publish_attempts = 0
        self.transform_publish_successes = 0
        
        # New performance metrics
        self.processing_skips = 0
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        
        # Motion tracking
        self.current_velocity = 0.0
        self.last_position = None
        self.last_position_time = time.time()
    
    def check_transform(self):
        """Periodically check if transform is available in TF tree."""
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
                        # Cache the transform
                        self.cached_transforms[cache_key] = transform
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
        """Load basketball physical parameters from config."""
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
    
    def _init_transform_parameters(self):
        """Initialize coordinate transform parameters."""
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
        """Set up subscribers for this node."""
        # Get topic config
        topics = self.config.get('topics', {})
        input_topics = topics.get('input', {})
        queue_size = topics.get('queue_size', 10)
        
        # LIDAR scan subscription
        lidar_topic = input_topics.get('lidar_scan', '/scan')
        self.scan_subscription = self.create_subscription(
            LaserScan,
            lidar_topic,
            self.scan_callback,
            queue_size,
            callback_group=self.subscription_cb_group
        )
        
        # YOLO detection subscription
        yolo_topic = input_topics.get('yolo_detection', '/basketball/yolo/position')
        self.yolo_subscription = self.create_subscription(
            PointStamped,
            yolo_topic,
            lambda msg: self.sensor_callback(msg, 'yolo'),
            queue_size,
            callback_group=self.subscription_cb_group
        )
          # HSV subscription removed
        
        # YOLO bounding box subscription for 3D position estimation
        from std_msgs.msg import Float32MultiArray
        yolo_bbox_topic = input_topics.get('yolo_bbox', '/basketball/yolo/bbox')
        self.yolo_bbox_subscription = self.create_subscription(
            Float32MultiArray,
            yolo_bbox_topic,
            self.yolo_bbox_callback,
            queue_size,
            callback_group=self.subscription_cb_group
        )
        
        # Initialize bounding box data storage
        self.yolo_bbox_data = {
            'width': 0.0,
            'height': 0.0,
            'timestamp': 0.0
        }
        
        self.get_logger().info("Core subscriptions established")
    
    def _setup_publishers(self):
        """Set up publishers after a delay to spread CPU load."""
        # Remove the timer that triggered this
        for i, timer in enumerate(self.node_timers):
            if timer.callback == self._setup_publishers:
                self.destroy_timer(timer)
                self.node_timers.pop(i)
                break
        
        # Get topic config
        topics = self.config.get('topics', {})
        output_topics = topics.get('output', {})
        queue_size = topics.get('queue_size', 10)
        
        # Ball position publisher
        position_topic = output_topics.get('ball_position', '/basketball/lidar/position')
        self.position_publisher = self.create_publisher(
            PointStamped,
            position_topic,
            queue_size
        )
        
        # Debug position publisher
        debug_topic = output_topics.get('debug_position', '/basketball/lidar/debug_position')
        self.debug_publisher = self.create_publisher(
            PointStamped,
            debug_topic,
            queue_size
        )
        
        # Conditionally create visualization publisher only if enabled
        self.marker_publisher = None
        if self.visualization_enabled:
            viz_topic = output_topics.get('visualization', '/basketball/lidar/visualization')
            self.marker_publisher = self.create_publisher(
                MarkerArray,
                viz_topic,
                queue_size
            )
        
        # Diagnostics publisher
        diag_topic = output_topics.get('diagnostics', '/basketball/lidar/diagnostics')
        self.diagnostics_publisher = self.create_publisher(
            String,
            diag_topic,
            queue_size
        )
        
        # Publisher for sharing system load with other nodes
        load_topic = output_topics.get('system_load', '/system/load')
        self.load_publisher = self.create_publisher(
            Float32,
            load_topic,
            queue_size
        )
        
        self.get_logger().info("Publishers established")
    
    def throttled_log(self, message, key, min_interval=1.0, level="info"):
        """Log with throttling to reduce overhead."""
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
        
        # Log with appropriate level
        if level == "error":
            self.get_logger().error(message)
        elif level == "warn":
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    def monitor_resources(self):
        """Monitor system resources and adapt processing accordingly."""
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
            angles = angle_min + angle_increment * np.arange(len(ranges), dtype=np.float32)[valid_indices]
            
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
            
            # Convert to Cartesian coordinates - optimized for minimal memory allocation
            x = valid_ranges * np.cos(angles)
            y = valid_ranges * np.sin(angles)
            z = np.zeros_like(x)  # No new memory allocation
            
            # Stack coordinates - memory efficient with preallocated array
            self.points_array = np.column_stack((x, y, z))
            
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
        Handle ball detections from camera systems (YOLO or HSV).
        Find matching points in LIDAR data with optimized processing.
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
            
            # NEW: Convert YOLO's (x, y) to polar coordinates for filtering
            estimated_x = camera_seed_point[0]
            estimated_y = camera_seed_point[1]
            r_est = math.sqrt(estimated_x**2 + estimated_y**2)
            theta_est = math.atan2(estimated_y, estimated_x)
            
            # NEW: Filter LIDAR points by distance and angle - use pre-allocated arrays
            px = self.points_array[:, 0]
            py = self.points_array[:, 1]
            
            # Use pre-allocated distance array
            np.sqrt(px**2 + py**2, out=self._distances_array[:len(px)])
            distances = self._distances_array[:len(px)]
            
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
            
            # Apply filters
            valid_dist = (distances >= (r_est - distance_tolerance)) & (distances <= (r_est + distance_tolerance))
            delta = np.abs(angles - theta_est)
            delta = np.where(delta > math.pi, 2*math.pi - delta, delta)  # handle angle wrap-around
            valid_angle = delta <= angular_tolerance
            
            # Combine both masks
            mask = valid_dist & valid_angle
            filtered_points = self.points_array[mask]
            
            # Log filtering results (throttled)
            if len(filtered_points) >= self.min_points:
                self.throttled_log(
                    f"Filtered LIDAR points from {len(self.points_array)} to {len(filtered_points)} "
                    f"using cone at distance {r_est:.2f}m, angle {math.degrees(theta_est):.1f}°",
                    key="filter_success",
                    min_interval=1.0
                )
            else:
                self.throttled_log(
                    f"Not enough points ({len(filtered_points)}) in detection cone. "
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
        """
        if points is None or len(points) < 3:
            return None, 0, 0
            
        best_inlier_count = 0
        best_center = None
        best_radius = 0
        
        # Limit iterations based on point count for better performance
        actual_iterations = min(max_iterations, len(points) // 2)
        actual_iterations = max(self.min_ransac_iterations, actual_iterations)  # At least min iterations
        
        # Pre-compute point coordinates for vector operations
        x_coords = points[:, 0]
        y_coords = points[:, 1]
        
        # Pre-allocate inlier mask for reuse
        inlier_mask = self._inlier_mask[:len(points)]
        
        for _ in range(actual_iterations):
            # Randomly sample 3 points
            if len(points) < 3:
                continue
                
            sample_indices = np.random.choice(len(points), 3, replace=False)
            
            # Directly use the pre-allocated circle points array
            for i in range(3):
                self._circle_points[i, 0] = points[sample_indices[i], 0]
                self._circle_points[i, 1] = points[sample_indices[i], 1]
            
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
                
                if inlier_count > best_inlier_count:
                    best_inlier_count = inlier_count
                    best_center = center
                    best_radius = radius
            except Exception:
                continue
        
        if best_center is None:
            return None, 0, 0
        
        # Refine with all inliers if we have enough
        if best_inlier_count >= 5:
            # Calculate quality metrics
            inlier_ratio = best_inlier_count / len(points)
            radius_error = abs(best_radius - self.ball_radius) / self.ball_radius
            quality = 0.7 * inlier_ratio + 0.3 * (1.0 - min(radius_error, 1.0))
            
            # Add z-coordinate for 3D position - reuse existing array
            center_3d = np.array([best_center[0], best_center[1], self.ball_center_height], dtype=np.float32)
            
            return center_3d, best_inlier_count, quality
        
        return None, 0, 0

    def fit_circle(self, points_2d):
        """
        Fit a circle to 2D points.
        Optimized for basketball size (9-inch diameter) on Raspberry Pi.
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
        
        # Only visualize if enabled and not in MINIMAL mode
        if self.visualization_enabled and self.marker_publisher is not None and self.performance_mode != "MINIMAL":
            self.visualize_detection(filtered_position, circle_quality, trigger_source)
        
        # With the lock, update position history
        with self.lock:
            self.position_history.add(current_time, filtered_position)
        
        # Update motion from new position
        self.update_velocity_from_positions(filtered_position)
    
    def update_velocity_from_positions(self, new_position):
        """
        Update velocity estimate from position history.
        Used to update the motion state manager.
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
    
    def visualize_detection(self, center, quality, source):
        """
        Create visualization markers for the detected ball.
        Only called when visualization is enabled and we're not in MINIMAL mode.
        Uses pre-allocated marker objects for efficiency.
        """
        # Skip if visualization is disabled or publisher wasn't created
        if not self.visualization_enabled or self.marker_publisher is None:
            return
            
        # Get visualization settings
        viz_config = self.config.get('visualization', {})
        marker_lifetime = viz_config.get('marker_lifetime', 1.0)
        
        # Reuse pre-allocated ball marker
        self._ball_marker.header.frame_id = "lidar_frame"
        self._ball_marker.header.stamp = self.scan_timestamp
        self._ball_marker.ns = "basketball"
        self._ball_marker.id = 1
        self._ball_marker.type = Marker.SPHERE
        self._ball_marker.action = Marker.ADD
        
        # Set position
        self._ball_marker.pose.position.x = center[0]
        self._ball_marker.pose.position.y = center[1]
        self._ball_marker.pose.position.z = center[2]
        self._ball_marker.pose.orientation.w = 1.0
        
        # Set color based on source
        colors = viz_config.get('colors', {})
        
        # Set color based on source and motion state for better visualization
        motion_state = self.motion_manager.current_state if hasattr(self, 'motion_manager') else "unknown"
          # Always use YOLO color config since HSV is removed
        color_config = colors.get('yolo', {'r': 0.0, 'g': 1.0, 'b': 0.3, 'base_alpha': 0.5})
        
        # Adjust color based on motion state
        if motion_state == MotionStateManager.STATIONARY:
            # More blue for stationary
            self._ball_marker.color.r = color_config.get('r', 0.0) * 0.7
            self._ball_marker.color.g = color_config.get('g', 1.0) * 0.7
            self._ball_marker.color.b = min(color_config.get('b', 0.3) + 0.4, 1.0)  # More blue
        elif motion_state == MotionStateManager.MEDIUM_FAST:
            # More red for fast movement
            self._ball_marker.color.r = min(color_config.get('r', 0.0) + 0.5, 1.0)  # More red
            self._ball_marker.color.g = color_config.get('g', 1.0) * 0.8
            self._ball_marker.color.b = color_config.get('b', 0.3) * 0.8
        else:
            # Default color
            self._ball_marker.color.r = color_config.get('r', 0.0)
            self._ball_marker.color.g = color_config.get('g', 1.0)
            self._ball_marker.color.b = color_config.get('b', 0.3)
        
        # Adjust transparency based on quality
        base_alpha = color_config.get('base_alpha', 0.5)
        self._ball_marker.color.a = min(base_alpha + quality * 0.5, 1.0)
        
        # Set size (basketball diameter)
        self._ball_marker.scale.x = self.ball_radius * 2.0
        self._ball_marker.scale.y = self.ball_radius * 2.0
        self._ball_marker.scale.z = self.ball_radius * 2.0
        
        # Set marker lifetime
        self._ball_marker.lifetime.sec = int(marker_lifetime)
        self._ball_marker.lifetime.nanosec = int((marker_lifetime % 1) * 1e9)
        
        # Reuse pre-allocated text marker
        self._text_marker.header.frame_id = "lidar_frame"
        self._text_marker.header.stamp = self.scan_timestamp
        self._text_marker.ns = "basketball_text"
        self._text_marker.id = 2
        self._text_marker.type = Marker.TEXT_VIEW_FACING
        self._text_marker.action = Marker.ADD
        
        # Position text above the ball
        text_height_offset = viz_config.get('text_height_offset', 0.2)
        self._text_marker.pose.position.x = center[0]
        self._text_marker.pose.position.y = center[1]
        self._text_marker.pose.position.z = center[2] + text_height_offset
        self._text_marker.pose.orientation.w = 1.0
        
        # Set text content with motion state
        quality_pct = int(quality * 100)
        motion_abbr = motion_state[:3].upper() if hasattr(self, 'motion_manager') else "UNK"
        self._text_marker.text = f"{source}: {quality_pct}% ({motion_abbr})"
        
        # Set text appearance
        text_size = viz_config.get('text_size', 0.05)
        self._text_marker.scale.z = text_size
        
        text_color = colors.get('text', {'r': 1.0, 'g': 1.0, 'b': 1.0, 'a': 1.0})
        self._text_marker.color.r = text_color.get('r', 1.0)
        self._text_marker.color.g = text_color.get('g', 1.0)
        self._text_marker.color.b = text_color.get('b', 1.0)
        self._text_marker.color.a = text_color.get('a', 1.0)
        
        self._text_marker.lifetime.sec = int(marker_lifetime)
        self._text_marker.lifetime.nanosec = int((marker_lifetime % 1) * 1e9)
        
        # Reuse pre-allocated marker array
        self._marker_array.markers = [self._ball_marker, self._text_marker]
        
        # Publish markers
        self.marker_publisher.publish(self._marker_array)
    
    def estimate_3d_from_2d(self, detection_msg, bbox_width, bbox_height):
        """
        Estimate a 3D position from a 2D detection and bbox dimensions.
        Similar to the fusion node's implementation but optimized for LIDAR use.
        Uses cached transforms and reused matrices for efficiency.
        
        Args:
            detection_msg (PointStamped): The 2D detection message
            bbox_width (float): Width of bounding box in pixels
            bbox_height (float): Height of bounding box in pixels
            
        Returns:
            np.ndarray: Estimated 3D position [x, y, z] or None if estimation fails
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
        """Log an error and update health status."""
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
        """Publish diagnostic information about the node with optimized message reuse."""
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
                    "avg_processing_time_ms": avg_time * 1000 if avg_time else 0,                    "sources": {
                        "yolo_detections": self.yolo_detections
                    }
                },
                "config": {
                    "ball_radius": self.ball_radius,
                    "max_distance": self.max_distance,
                    "min_points": self.min_points,
                    "visualization_enabled": self.visualization_enabled
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
        Only executed when not in MINIMAL mode.
        Uses message object reuse for efficiency.
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
        """Clean shutdown of the node."""
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