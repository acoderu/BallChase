#!/usr/bin/env python3

"""
Optimized State-Aware Fusion Node for ROS 2
Designed for resource-constrained systems like Raspberry Pi 5
Focuses on efficient state management, reduced CPU usage, and robust tracking
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
    """Lightweight buffer for sensor measurements with fixed memory allocation."""
    
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


class SensorManager:
    """Efficient sensor data management with fixed memory allocation."""
    
    def __init__(self, sensor_names=None):
        """Initialize with specified sensors."""
        self.sensors = sensor_names or []
        self.data_buffers = {}
        self.last_update_time = {}
        self.update_count = {}
        self.fps_estimates = {}
        
        # Initialize buffers for all sensors
        for sensor in self.sensors:
            buffer_size = 15 if sensor in ['lidar', 'yolo_3d'] else 10
            self.data_buffers[sensor] = LightweightBuffer(buffer_size)
            self.last_update_time[sensor] = 0.0
            self.update_count[sensor] = 0
            self.fps_estimates[sensor] = 0.0
        
        # Track sensor health
        self.sensor_active = {sensor: False for sensor in self.sensors}
        self.sensor_gap_durations = {sensor: 0.0 for sensor in self.sensors}
    
    def add_measurement(self, sensor, timestamp, data):
        """Add a new measurement for a sensor."""
        if sensor not in self.sensors:
            return
        
        current_time = time.time()
        
        # Update buffer
        self.data_buffers[sensor].add(current_time, data)
        
        # Update statistics
        if self.last_update_time[sensor] > 0:
            interval = current_time - self.last_update_time[sensor]
            # Use exponential moving average for FPS estimate
            if self.fps_estimates[sensor] > 0:
                alpha = 0.3  # Smoothing factor
                self.fps_estimates[sensor] = (1 - alpha) * self.fps_estimates[sensor] + alpha * (1.0 / max(interval, 0.001))
            else:
                self.fps_estimates[sensor] = 1.0 / max(interval, 0.001)
        
        self.last_update_time[sensor] = current_time
        self.update_count[sensor] += 1
        
        # Mark sensor as active
        self.sensor_active[sensor] = True
        self.sensor_gap_durations[sensor] = 0.0
    
    def get_latest(self, sensor):
        """Get the latest measurement for a sensor."""
        if sensor not in self.sensors:
            return None
        return self.data_buffers[sensor].get_latest()
    
    def update_sensor_health(self, current_time, max_gap=1.0):
        """Update sensor health status based on update times."""
        for sensor in self.sensors:
            gap_duration = current_time - self.last_update_time.get(sensor, 0)
            self.sensor_gap_durations[sensor] = gap_duration
            
            # Mark as inactive if gap exceeds threshold
            if gap_duration > max_gap:
                self.sensor_active[sensor] = False
    
    def get_active_sensor_count(self):
        """Get the count of currently active sensors."""
        return sum(1 for sensor, active in self.sensor_active.items() if active)
    
    def get_active_high_quality_sensors(self):
        """Get the count of active high-quality (3D) sensors."""
        return sum(1 for sensor in ['lidar', 'yolo_3d'] 
                  if sensor in self.sensor_active and self.sensor_active[sensor])
    
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
    """Efficient state management with hysteresis and confidence tracking."""
    
    # Define motion states
    UNKNOWN = "unknown"
    STATIONARY = "stationary"
    LONG_STATIONARY = "long_stationary"
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
        """Get validation threshold multiplier based on current state."""
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
        """Check if currently in state transition."""
        return time.time() - self.last_state_change_time < max_transition_time
    
    def get_state_age(self):
        """Get the age of the current state."""
        if self.last_state_change_time:
            return time.time() - self.last_state_change_time
        return 0.0


class OptimizedFusionNode(LifecycleNode):
    """
    Optimized fusion node with reduced CPU usage and improved state management.
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
        
        self.get_logger().info("Optimized Fusion Node initialized with resource-efficient design")
    
    def on_configure(self, state):
        """Configure node with minimized resource usage."""
        self.get_logger().info("Configuring node...")
        
        try:
            # Initialize transform system
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            
            # Load configuration
            self.load_configuration()
            
            # Initialize core state tracking
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
            
            # Cache common transforms
            self.cached_transforms = {}
            
            self.is_configured = True
            self.get_logger().info("Node configured successfully")
            return TransitionCallbackReturn.SUCCESS
            
        except Exception as e:
            self.get_logger().error(f"Configuration error: {str(e)}")
            return TransitionCallbackReturn.ERROR
    
    def on_activate(self, state):
        """Activate node with staged startup for lower initial CPU spike."""
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
            
            # Setup timers with low frequencies
            self._timer_list.append(self.create_timer(
                0.1, self.filter_update, callback_group=self.timer_cb_group))
            self._timer_list.append(self.create_timer(
                0.5, self.publish_state, callback_group=self.timer_cb_group))
            self._timer_list.append(self.create_timer(
                1.0, self.publish_status, callback_group=self.timer_cb_group))
            self._timer_list.append(self.create_timer(
                5.0, self.publish_diagnostics, callback_group=self.timer_cb_group))
            
            # Delayed setup for other subscriptions
            self._timer_list.append(self.create_timer(
                0.2, self.setup_remaining_subscriptions, callback_group=self.timer_cb_group))
            
            # Cache transforms after a delay
            self._timer_list.append(self.create_timer(
                0.5, self.cache_transforms, callback_group=self.timer_cb_group))
            
            self.is_activated = True
            self.get_logger().info("Node activated with staged startup")
            return TransitionCallbackReturn.SUCCESS
            
        except Exception as e:
            self.get_logger().error(f"Activation error: {str(e)}")
            return TransitionCallbackReturn.ERROR
    
    def on_deactivate(self, state):
        """Deactivate node and clean up resources."""
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
        """Clean up resources."""
        self.get_logger().info("Cleaning up resources...")
        
        # Release transform resources
        self.tf_listener = None
        self.tf_buffer = None
        
        self.is_configured = False
        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state):
        """Perform final shutdown."""
        self.get_logger().info("Shutting down...")
        return TransitionCallbackReturn.SUCCESS
    
    def load_configuration(self):
        """Load configuration with resource-efficient defaults."""
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
        
        self.get_logger().info("Configuration loaded with efficient defaults")
    
    def setup_lidar_subscription(self):
        """Setup LiDAR subscription first (highest priority)."""
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
        """Setup remaining subscriptions after a delay."""
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
        """Cache transforms for efficient lookup."""
        # Destroy the timer once called
        for timer in self._timer_list:
            if timer.callback == self.cache_transforms:
                self.destroy_timer(timer)
                self._timer_list.remove(timer)
                break
        
        try:
            # Define important transform pairs
            transform_pairs = [
                ('ascamera_color_0', self.reference_frame),
                ('lidar_frame', self.reference_frame),
                ('ascamera_camera_link_0', self.reference_frame)
            ]
            
            # Cache each transform
            for source, target in transform_pairs:
                transform = self.tf_buffer.lookup_transform(
                    target, source, 
                    rclpy.time.Time()
                )
                
                # Store in cache
                cache_key = f"{source}_{target}"
                self.cached_transforms[cache_key] = transform
                
                # Log success
                self.get_logger().info(
                    f"Cached transform: {source} → {target}: "
                    f"translation=({transform.transform.translation.x:.3f}, "
                    f"{transform.transform.translation.y:.3f}, "
                    f"{transform.transform.translation.z:.3f})"
                )
            
            self.transform_available = True
            self.get_logger().info("Transform caching completed")
            
        except Exception as e:
            self.get_logger().error(f"Transform caching error: {str(e)}")
    
    def bbox_callback(self, msg, source):
        """
        Optimized bbox callback for Float32MultiArray format.
        Used primarily by YOLO detection.
        """
        if not self.is_activated:
            return
            
        try:
            if hasattr(msg, 'data') and len(msg.data) >= 4:
                width = msg.data[2]   # width
                height = msg.data[3]  # height
                
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
    
    def bbox_callback_standard(self, msg, source):
        """Bbox callback for standard BoundingBox2D format."""
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
        Optimized sensor callback with minimal processing.
        Defers expensive operations to filter update.
        Logs one in every 3 sensor updates for key sensors.
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
            
            # Log one in every 3 sensor updates for key sensors
            if source in ['lidar', 'yolo_3d', 'yolo_2d']:
                # Initialize counters if needed
                if not hasattr(self, '_sensor_log_counters'):
                    self._sensor_log_counters = {'lidar': 0, 'yolo_3d': 0, 'yolo_2d': 0}
                
                # Update counter for this sensor
                self._sensor_log_counters[source] = (self._sensor_log_counters[source] + 1) % 3
                
                # Log every third update
                if self._sensor_log_counters[source] == 0:
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
        """Initialize filter state with first measurement."""
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
    
    def transform_point(self, point_msg, target_frame):
        """Transform point with cached transform lookup."""
        if not self.transform_available:
            return None
            
        try:
            # Return original if already in target frame
            if point_msg.header.frame_id == target_frame:
                return point_msg
                
            # Check cache first
            cache_key = f"{point_msg.header.frame_id}_{target_frame}"
            transform = None
            
            if cache_key in self.cached_transforms:
                transform = self.cached_transforms[cache_key]
            else:
                # Lookup and cache
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    point_msg.header.frame_id,
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)
                )
                self.cached_transforms[cache_key] = transform
            
            # Apply transform
            return do_transform_point(point_msg, transform)
            
        except Exception as e:
            self.throttled_log(
                f"Transform error: {str(e)}",
                key="transform_error",
                min_interval=5.0,
                level="error"
            )
            return None
    
    def estimate_3d_from_2d(self, detection_msg, bbox_data):
        """
        Optimized 3D position estimation from 2D detection.
        Only called when needed during filter update.
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
            
            # Calculate distance
            focal_length = self.camera_parameters['focal_length']
            estimated_distance = (basketball_diameter * focal_length) / bbox_width
            
            # Age penalty for older bboxes
            age_factor = min(1.0 + (bbox_age / max_age) * 0.15, 1.15)
            estimated_distance *= age_factor
            
            # Get transform
            transform = None
            cache_key = f"ascamera_color_0_{self.reference_frame}"
            
            if cache_key in self.cached_transforms:
                transform = self.cached_transforms[cache_key]
            else:
                transform = self.tf_buffer.lookup_transform(
                    self.reference_frame,
                    'ascamera_color_0',
                    rclpy.time.Time(),
                    rclpy.duration.Duration(seconds=0.1)
                )
                self.cached_transforms[cache_key] = transform
            
            # Camera position
            camera_pos = [
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z
            ]
            
            # Normalized direction vector calculation
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
            dir_vec = [offset_x, offset_y, focal_length]
            
            # Normalize
            magnitude = math.sqrt(sum(c*c for c in dir_vec))
            if magnitude > 0:
                dir_vec = [c/magnitude for c in dir_vec]
            
            # Get quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            
            # Convert to rotation matrix (optimized calculation)
            xx = qx * qx
            xy = qx * qy
            xz = qx * qz
            xw = qx * qw
            yy = qy * qy
            yz = qy * qz
            yw = qy * qw
            zz = qz * qz
            zw = qz * qw
            
            # Rotation matrix
            rot_mat = [
                [1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw)],
                [2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw)],
                [2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)]
            ]
            
            # Apply rotation
            rotated_dir = [0, 0, 0]
            for i in range(3):
                rotated_dir[i] = sum(rot_mat[i][j] * dir_vec[j] for j in range(3))
            
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
            result.point.x = position[0]
            result.point.y = position[1]
            result.point.z = position[2]
            
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
        """Process sensor data by priority order."""
        # Update sensor health status
        self.sensor_manager.update_sensor_health(time.time())
        
        # Process sensors in priority order
        processed_any = False
        
        # First try LiDAR
        if self.process_sensor('lidar'):
            processed_any = True
        
        # Then try YOLO 3D
        if self.process_sensor('yolo_3d'):
            processed_any = True
        
        # Process 2D sensors only if not enough 3D sensors
        if self.sensor_manager.get_active_high_quality_sensors() < 1:
            # Try YOLO 2D with 3D estimation
            if self.process_2d_sensor('yolo_2d'):
                processed_any = True
        
        # If no sensor processed, increase covariance slightly
        if not processed_any:
            # Get current state for context
            motion_state = self.motion_manager.current_state
            
            # Covariance growth rate based on motion state
            growth_rate = 1.05
            if motion_state == MotionStateManager.MEDIUM_FAST:
                growth_rate = 1.1
            
            # Apply growth
            self.covariance[0:2, 0:2] *= growth_rate  # Position uncertainty
            self.covariance[2:4, 2:4] *= growth_rate  # Velocity uncertainty
    
    def process_sensor(self, sensor):
        """Process a single 3D sensor measurement."""
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
        """Process 2D sensor with 3D estimation."""
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
        Validate measurement with motion-aware thresholds.
        
        Returns:
            tuple: (valid, innovation)
        """
        # Prepare measurement vector
        z = np.array([measurement.point.x, measurement.point.y], dtype=np.float32)
        
        # Calculate innovation
        y = z - self.state[0:2]
        
        # Calculate innovation magnitude (Mahalanobis distance)
        # Innovation covariance S = H*P*H' + R
        R = np.eye(2, dtype=np.float32) * self.measurement_noise.get(source, 0.1)
        
        # Calculate S directly with slices for efficiency
        S = self.covariance[0:2, 0:2] + R
        
        try:
            # Calculate inverse of S
            S_inv = np.linalg.inv(S)
            
            # Calculate squared Mahalanobis distance
            innovation_value = float(np.dot(np.dot(y, S_inv), y))
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
            
        except np.linalg.LinAlgError:
            self.throttled_log(
                f"Matrix inversion failed for {source}",
                key="matrix_error",
                min_interval=5.0,
                level="error"
            )
            return False, float('inf')
    
    def update_state_with_measurement(self, measurement, source):
        """Update state with validated measurement."""
        # Prepare measurement
        z = np.array([measurement.point.x, measurement.point.y], dtype=np.float32)
        
        # Setup measurement noise
        R = np.eye(2, dtype=np.float32) * self.measurement_noise.get(source, 0.1)
        
        # Calculate innovation
        y = z - self.state[0:2]
        
        # Calculate innovation covariance
        S = self.covariance[0:2, 0:2] + R
        
        try:
            # Calculate Kalman gain
            S_inv = np.linalg.inv(S)
            K_top = np.dot(self.covariance[0:2, 0:2], S_inv)
            K_bottom = np.dot(self.covariance[2:4, 0:2], S_inv)
            
            # Store original state for position change calculation
            old_state = self.state.copy()
            
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
            
            # Get motion state for context
            motion_state = self.motion_manager.current_state
            
            # Apply different update strategy based on motion state
            if motion_state in [MotionStateManager.STATIONARY, MotionStateManager.LONG_STATIONARY]:
                # For stationary objects, use more conservative blend
                pos_discrepancy = np.linalg.norm(z - np.array([self.state[0], self.state[1]]))
                
                if pos_discrepancy > 0.1:
                    # Significant discrepancy during stationary - likely valid movement
                    # Apply stronger blend to allow escaping stationary state
                    blend_factor = min(blend_factor + 0.2, 0.9)
                    
                    self.throttled_log(
                        f"Detected movement during {motion_state}: {pos_discrepancy:.3f}m, "
                        f"blend={blend_factor:.2f}",
                        key="stationary_movement",
                        min_interval=0.5
                    )
                    
                    # Update position with strong blend
                    self.state[0] = (1.0 - blend_factor) * self.state[0] + blend_factor * z[0]
                    self.state[1] = (1.0 - blend_factor) * self.state[1] + blend_factor * z[1]
                    
                    # Calculate implied velocity
                    implied_vx = (z[0] - old_state[0]) / 0.1  # Assume recent measurement
                    implied_vy = (z[1] - old_state[1]) / 0.1
                    
                    # Update velocity with reduced blend
                    vel_blend = blend_factor * 0.5
                    self.state[2] = (1.0 - vel_blend) * self.state[2] + vel_blend * implied_vx
                    self.state[3] = (1.0 - vel_blend) * self.state[3] + vel_blend * implied_vy
                    
                else:
                    # Normal update for stationary with small discrepancy
                    self.state[0:2] += np.dot(K_top, y)
                    self.state[2:4] += np.dot(K_bottom, y)
            
            elif motion_state == MotionStateManager.MEDIUM_FAST:
                # For fast-moving objects, trust measurements more
                # Especially from LiDAR and YOLO 3D
                if source in ['lidar', 'yolo_3d']:
                    blend_factor = min(blend_factor + 0.1, 0.9)
                
                # Update position with blend
                self.state[0] = (1.0 - blend_factor) * self.state[0] + blend_factor * z[0]
                self.state[1] = (1.0 - blend_factor) * self.state[1] + blend_factor * z[1]
                
                # Update velocity with partial Kalman
                self.state[2:4] += np.dot(K_bottom, y)
                
            else:  # SMALL_MOVEMENT or UNKNOWN
                # Standard Kalman update
                self.state[0:2] += np.dot(K_top, y)
                self.state[2:4] += np.dot(K_bottom, y)
            
            # Calculate position change for logging
            pos_change = np.linalg.norm(self.state[0:2] - old_state[0:2])
            if pos_change > 0.1:
                self.throttled_log(
                    f"Position updated: {pos_change:.3f}m from {source}",
                    key="position_update",
                    min_interval=0.5
                )
            
            # Update covariance with Joseph form
            I_KH = np.eye(4, dtype=np.float32)
            I_KH[0:2, 0:2] -= np.dot(K_top, np.eye(2))
            
            # Build full K matrix for covariance update
            K = np.zeros((4, 2), dtype=np.float32)
            K[0:2, 0:2] = K_top
            K[2:4, 0:2] = K_bottom
            
            # Update covariance
            self.covariance = np.dot(np.dot(I_KH, self.covariance), I_KH.T) + np.dot(np.dot(K, R), K.T)
            
            # Ensure covariance is symmetric
            self.covariance = 0.5 * (self.covariance + self.covariance.T)
            
            return True
            
        except np.linalg.LinAlgError:
            self.throttled_log(
                f"Kalman update failed for {source}",
                key="kalman_error",
                min_interval=5.0,
                level="error"
            )
            return False
    
    def update_motion_state(self):
        """Update motion state based on velocity."""
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
                f"{time_prefix}[STATE] {self.motion_manager.previous_state} → "
                f"{self.motion_manager.current_state} (v={velocity:.3f}m/s, age={state_age:.1f}s)"
            )
    
    def apply_physics_constraints(self):
        """Apply physics constraints based on ground movement."""
        # Get motion state
        motion_state = self.motion_manager.current_state if hasattr(self, 'motion_manager') else "unknown"
        
        # Apply speed limit based on motion state
        vx, vy = self.state[2], self.state[3]
        speed = math.sqrt(vx*vx + vy*vy)
        
        # Define maximum speed based on state
        max_speed = 5.0  # Default
        if motion_state == MotionStateManager.STATIONARY:
            max_speed = 0.1
        elif motion_state == MotionStateManager.LONG_STATIONARY:
            max_speed = 0.05
        elif motion_state == MotionStateManager.SMALL_MOVEMENT:
            max_speed = 2.0
        
        # Apply speed limit if exceeded
        if speed > max_speed:
            scale = max_speed / speed
            self.state[2] *= scale
            self.state[3] *= scale
            
        # Enforce z-height constraint
        # This is a basketball that rolls on the ground, so z is fixed
        # Note: z is not in our state vector, but we enforce it in published messages
    
    def update_tracking_status(self):
        """Update tracking reliability status."""
        # Count active sensors
        active_3d = self.sensor_manager.get_active_high_quality_sensors()
        active_2d = sum(1 for sensor in ['yolo_2d'] 
                      if sensor in self.sensor_manager.sensor_active and 
                      self.sensor_manager.sensor_active[sensor])
        
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
        
        # Update tracking status with hysteresis
        if not self.tracking_reliable:
            # More lenient to start tracking
            self.tracking_reliable = sensors_ok and uncertainty_ok
        else:
            # More strict to lose tracking (need to lose both)
            self.tracking_reliable = sensors_ok or uncertainty_ok
        
        # Publish status
        status_msg = Bool()
        status_msg.data = self.tracking_reliable
        self.status_pub.publish(status_msg)
        
        # Publish uncertainty
        uncertainty_msg = Float32()
        uncertainty_msg.data = self.position_uncertainty
        self.uncertainty_pub.publish(uncertainty_msg)
    
    def publish_state(self):
        """Publish the current state estimate."""
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
            
            # Log fusion output with dual-time format for tabular extraction
            # Calculate required metrics
            distance = math.sqrt(self.state[0]**2 + self.state[1]**2)
            direction = math.degrees(math.atan2(self.state[1], self.state[0]))
            speed = math.sqrt(self.state[2]**2 + self.state[3]**2)
            
            # Get motion state
            motion_state = "unknown"
            if hasattr(self, 'motion_manager'):
                motion_state = self.motion_manager.current_state
            
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
                
        except Exception as e:
            self.get_logger().error(f"Error publishing state: {str(e)}")
    
    def publish_status(self):
        """Publish status information."""
        if not self.is_activated:
            return
            
        try:
            # Calculate uptime
            current_time = time.time()
            uptime = current_time - self.start_time
              # Count active sensors
            active_3d = self.sensor_manager.get_active_high_quality_sensors()
            active_2d = sum(1 for sensor in ['yolo_2d'] 
                          if sensor in self.sensor_manager.sensor_active and 
                          self.sensor_manager.sensor_active[sensor])
            
            # Get motion state
            motion_state = "unknown"
            if hasattr(self, 'motion_manager'):
                motion_state = self.motion_manager.current_state
                
            # Make sure position_uncertainty is initialized
            if not hasattr(self, 'position_uncertainty'):
                self.position_uncertainty = float('inf')
            
            # Build status message
            status = {
                'tracking': self.tracking_reliable,
                'initialized': self.initialized,
                'motion_state': motion_state,
                'sensors_3d': active_3d,
                'sensors_2d': active_2d,
                'position_uncertainty': round(float(self.position_uncertainty), 3),
                'uptime': round(uptime, 1)
            }
            
            # Log status with reduced frequency
            elapsed_time = current_time - self.start_time
            
            # Get time prefix with both elapsed and ROS time
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
        """Publish detailed diagnostics."""
        if not self.is_activated:
            return
            
        try:
            # Get CPU and memory usage if available
            cpu_usage = None
            memory_usage = None
            if HAS_PSUTIL:
                try:
                    cpu_usage = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    memory_usage = {
                        'percent': memory.percent,
                        'used_mb': memory.used / (1024 * 1024)
                    }
                except:
                    pass
            
            # Get sensor diagnostics
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
                'position': [float(self.state[0]), float(self.state[1])],  # Convert numpy.float32 to Python float
                'velocity': [float(self.state[2]), float(self.state[3])],  # Convert numpy.float32 to Python float
                'uncertainty': float(self.position_uncertainty),  # Convert numpy.float32 to Python float
                'motion_state': self.motion_manager.current_state if hasattr(self, 'motion_manager') else "unknown",
                'active_sensors': active_sensors
            }
            
            # Add system metrics if available
            if cpu_usage is not None:
                diag['cpu'] = float(cpu_usage)  # Ensure it's a Python float
            if memory_usage is not None:
                diag['memory'] = {
                    'percent': float(memory_usage['percent']),  # Ensure it's a Python float
                    'used_mb': float(memory_usage['used_mb'])   # Ensure it's a Python float
                }
            
            # Create diagnostics message - ensure all values are JSON serializable
            diag_msg = String()
            diag_msg.data = json.dumps(diag)
            self.diagnostics_pub.publish(diag_msg)
            
            # Log system metrics with reduced frequency
            if cpu_usage is not None:
                self.throttled_log(
                    f"System: CPU={float(cpu_usage):.1f}%, Mem={float(memory_usage['percent']):.1f}%, "
                    f"Active sensors: {len(active_sensors)}",
                    key="system",
                    min_interval=15.0
                )
                
        except Exception as e:
            self.get_logger().error(f"Error publishing diagnostics: {str(e)}")
    
    def get_time_prefix(self):
        """Create a standardized time prefix with both elapsed and ROS times."""
        # Get elapsed time since startup
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Get ROS 2 time
        ros_time = self.get_clock().now()
        ros_seconds = ros_time.seconds_nanoseconds()
        ros_time_str = f"{ros_seconds[0]}.{str(ros_seconds[1]).zfill(9)[:6]}"
        
        # Return formatted prefix
        return f"[T+{elapsed_time:.1f}s][ROS:{ros_time_str}]"
        
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
        """Implement adaptive uncertainty recovery to improve tracking stability during sensor gaps."""
        # Only apply when uncertainty exceeds normal thresholds
        if self.position_uncertainty < 0.3:  # Lowered from 0.4 to be closer to normal operation
            return False
            
        # Check sensor health
        current_time = time.time()
        active_sensors = self.sensor_manager.get_active_sensor_count()
        
        # Get motion state for context
        motion_state = self.motion_manager.current_state if hasattr(self, 'motion_manager') else "unknown"
        
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
                # Use a more aggressive recovery rate (0.95 instead of 0.98)
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
        """Update uncertainty metrics from covariance matrix with improved heuristics."""
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
        Optimized Kalman filter update with reduced operations.
        Core algorithm split into stages for better organization.
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
            
            # Check sensor health before prediction
            self.sensor_manager.update_sensor_health(current_time)
            active_sensors = self.sensor_manager.get_active_sensor_count()
            
            # Apply preliminary uncertainty damping if no sensors active
            # This prevents runaway uncertainty during prediction when no sensors
            if active_sensors == 0 and self.position_uncertainty > 0.3:
                # Pre-recovery for prediction phase
                self.state[2] *= 0.9  # Dampen velocity before prediction
                self.state[3] *= 0.9
                
            # 1. Prediction stage
            self.predict_state(dt)
            
            # 2. Update stage
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

    def predict_state(self, dt):
        """Optimized state prediction with minimal matrix operations and improved motion-aware processing."""
        # Update state transition matrix for current dt
        self._F[0, 2] = dt  # x += vx*dt
        self._F[1, 3] = dt  # y += vy*dt
        
        # Reset process noise matrix
        self._Q.fill(0.0)
        
        # Get motion state-based scaling with more precise tuning
        motion_state = self.motion_manager.current_state if hasattr(self, 'motion_manager') else "unknown"
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
            current_velocity = math.sqrt(vx*vx + vy*vy)
            
            if current_velocity > 0.01:
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
                if dv > 0 and current_velocity > 0:
                    factor = 1.0 - (dv / current_velocity)
                    self.state[2] *= factor
                    self.state[3] *= factor
        
        # Fill in process noise with optimized access
        # Position variances
        self._Q[0, 0] = q_pos * dt * dt / 3.0
        self._Q[1, 1] = q_pos * dt * dt / 3.0
        
        # Velocity variances
        self._Q[2, 2] = q_vel * dt
        self._Q[3, 3] = q_vel * dt
        
        # Covariances
        self._Q[0, 2] = self._Q[2, 0] = q_pos * dt * dt / 2.0
        self._Q[1, 3] = self._Q[3, 1] = q_pos * dt * dt / 2.0
        
        # Check for uncertainty-based scaling - limit growth for high uncertainty
        if hasattr(self, 'position_uncertainty') and self.position_uncertainty > 0.3:
            # Scale down process noise when uncertainty is already high
            uncertainty_factor = 0.3 / self.position_uncertainty
            self._Q *= max(0.5, uncertainty_factor)
        
        # Predict state with optimized operations
        self.state = np.dot(self._F, self.state)
        
        # Predict covariance
        self.covariance = np.dot(np.dot(self._F, self.covariance), self._F.T) + self._Q
def main(args=None):
    """Main function with optimized executor."""
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