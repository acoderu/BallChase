"""
Basketball Tracking Robot - Optimized PID Controller Node
=======================================================

This controller implements efficient movement patterns for a mecanum-wheeled
basketball tracking robot with several enhancements:
- Angular-first control strategy for diagonal movements
- Fast strategy transitions for responsive tracking
- Enhanced integral term management to prevent windup
- Coordinated angular-lateral control with balanced parameters
- Balanced error thresholds and hysteresis for smooth behavior
- Data freshness detection with graduated response
- Fusion rate detection and control rate adaptation
- Continuous motion tracking with trajectory prediction
- Optimized for Raspberry Pi 5 performance with resource monitoring
- Performance optimizations for CPU-constrained environments
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped, Vector3, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray
from tf2_ros import TransformListener, Buffer
from rclpy.logging import LoggingSeverity
import math
import time
import numpy as np
import signal
import sys
from collections import deque
import logging
import traceback  

# Import modules from refactored files
from pid_helpers import LightweightBuffer, CircularBuffer, ThrottledLogger, FastTrigonometry, ResourceMonitor
from pid_target_filter import EnhancedTargetFilter, ErrorTracker
from pid_computation import PIDControllers
from pid_target_tracking import TargetTrackingModule, MovementStrategyModule, VelocityControlModule, TransformSystem
from pid_target_tracking import RecoveryBehaviorModule, TransformStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pid_controller')
throttled_logger = ThrottledLogger(logger)

# Topic configuration
TOPICS = {
    "input": {
        "target": "/basketball/fused/position",
        "state": "/robot/state",
        "orientation": "/imu/rpy/filtered",  # Orientation from IMU
        "odometry": "/odom"                  # Odometry data
    },
    "output": {
        "cmd_vel": "/controller/cmd_vel",
        "diagnostics": "/pid/diagnostics",
        "performance": "/pid/performance"    # Performance metrics topic
    }
}

# Log throttling parameters
LOG_THROTTLE_CONTROL = 2.0     # Seconds between control loop status logs
LOG_THROTTLE_STATE = 0.5       # Seconds between state change logs
LOG_THROTTLE_DIAG = 1.0        # Seconds between diagnostic logs


# Centralized object pool manager
class ObjectPoolManager:
    """Manages pools of reusable objects to reduce memory allocations."""
    
    def __init__(self, max_twist=10, max_vector3=15):
        """Initialize object pools."""
        self.twist_pool = [Twist() for _ in range(max_twist)]
        self.vector3_pool = [Vector3() for _ in range(max_vector3)]
        
        # Pool statistics for monitoring
        self.twist_misses = 0
        self.vector3_misses = 0
        self.twist_max_usage = 0
        self.vector3_max_usage = 0
        
    def get_twist(self):
        """Get a Twist message from the pool, or create a new one if pool is empty."""
        if not self.twist_pool:
            self.twist_misses += 1
            return Twist()
        
        # Track max usage
        self.twist_max_usage = max(self.twist_max_usage, len(self.twist_pool))
        
        # Get from pool
        twist = self.twist_pool.pop()
        
        # Reset all fields
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        
        return twist
    
    def return_twist(self, twist):
        """Return a Twist to the pool if below capacity."""
        if len(self.twist_pool) < 10:
            self.twist_pool.append(twist)
    
    def get_vector3(self):
        """Get a Vector3 from the pool, or create a new one if pool is empty."""
        if not self.vector3_pool:
            self.vector3_misses += 1
            return Vector3()
        
        # Track max usage
        self.vector3_max_usage = max(self.vector3_max_usage, len(self.vector3_pool))
        
        # Get from pool
        vector = self.vector3_pool.pop()
        
        # Reset all fields
        vector.x = 0.0
        vector.y = 0.0
        vector.z = 0.0
        
        return vector
    
    def return_vector3(self, vector):
        """Return a Vector3 to the pool if below capacity."""
        if len(self.vector3_pool) < 15:
            self.vector3_pool.append(vector)
    
    def get_stats(self):
        """Return pool usage statistics."""
        return {
            'twist_pool_size': len(self.twist_pool),
            'vector3_pool_size': len(self.vector3_pool),
            'twist_misses': self.twist_misses,
            'vector3_misses': self.vector3_misses,
            'twist_max_usage': self.twist_max_usage,
            'vector3_max_usage': self.vector3_max_usage
        }


class InitializationError(Exception):
    """Exception raised when initialization fails."""
    pass

def handle_initialization_error(logger, message, original_exception=None):
    """Handle initialization errors consistently."""
    error_message = f"{message}"
    if original_exception:
        error_message += f": {str(original_exception)}"
    logger.error(error_message)
    raise InitializationError(error_message) from original_exception

class OptimizedPIDControllerNode(Node):
    """Enhanced PID Controller node with improved movement strategy and error handling."""
    def __init__(self):
        """Initialize the enhanced PID controller node with phased, dependency-driven initialization."""
        super().__init__('pid_controller')
        # Set ROS2 logger level based on debug_level
        
        
        try:
            # Phase 1: Parameter initialization (must come first)
            self._initialize_parameters()
            if self.debug_level >= 2:
                self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
            else:
                self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
            # Phase 2: Core initialization
            self._initialize_core_components()
            # Phase 3: Basic utility components
            self._initialize_utility_components()
            # Phase 4: Dependent components
            self._initialize_dependent_components()
            # Phase 5: Final setup
            self._initialize_final_components()
            # Validate complete initialization
            self._validate_initialization()
            self.get_logger().info("Initialization complete - all components ready")
        except Exception as e:
            self.get_logger().error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"PID Controller initialization failed: {str(e)}")

    def _initialize_core_components(self):
        """Initialize core components with no dependencies."""
        self.callback_group = ReentrantCallbackGroup()
        self.fast_trig = FastTrigonometry()
        self._init_memory_pools()
        self._init_state_variables()

    def _initialize_parameters(self):
        """Initialize and validate all parameters."""
        self._declare_parameters()
        self._validate_parameters()

    def _initialize_utility_components(self):
        """Initialize utility components that depend only on parameters."""
        # Pass self.get_logger() to all helper modules for consistent logging
        self.resource_monitor = ResourceMonitor(throttled_logger, debug_level=self.debug_level)
        if hasattr(self.resource_monitor, 'set_rate_limits'):
            self.resource_monitor.set_rate_limits(
                min_rate=self.min_control_rate,
                max_rate=self.max_control_rate,
                base_rate=self.update_rate
            )
        if hasattr(self.resource_monitor, 'set_cpu_thresholds'):
            self.resource_monitor.set_cpu_thresholds(
                low_threshold=self.cpu_low_threshold,
                high_threshold=self.cpu_high_threshold
            )
        self.resource_monitor.start()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        from pid_target_tracking import TransformSystem
        self.transform_system = TransformSystem(self, throttled_logger, self.tf_buffer)
        self.transform_system.add_transform_dependency("base_link", "imu_link", required=True)
        if not self.transform_system.start_initialization():
            raise RuntimeError("Failed to start transform system initialization")

    def _initialize_dependent_components(self):
        """Initialize components that depend on utility components."""
        self._init_controllers()
        self.target_tracker = TargetTrackingModule(
            throttled_logger,
            filter_buffer_size=self.filter_buffer_size,
            prediction_horizon=self.prediction_horizon,
            debug_level=self.debug_level
        )
        if not hasattr(self.target_tracker, 'target_filter') or self.target_tracker.target_filter is None:
            raise RuntimeError("Target tracking filter was not properly initialized")
        self.strategy_module = MovementStrategyModule(throttled_logger, self.debug_level)
        self.velocity_control = VelocityControlModule(throttled_logger)
        self.velocity_control.set_approach_parameters(
            self.approach_distance, 
            self.min_approach_factor
        )
        self.recovery_module = RecoveryBehaviorModule(throttled_logger)

    def _initialize_final_components(self):
        """Initialize communication and timer components."""
        self._setup_publishers()
        self._setup_subscriptions()
        self._setup_timers()

    def _validate_parameters(self):
        """Validate that all required parameters are properly set."""
        required_params = [
            'update_rate', 'min_control_rate', 'max_control_rate',
            'approach_distance', 'min_approach_factor',
            'distance_threshold', 'lateral_threshold', 'angular_threshold'
        ]
        for param in required_params:
            if not hasattr(self, param) or getattr(self, param) is None:
                raise ValueError(f"Required parameter '{param}' is not initialized")

    def _validate_initialization(self):
        """Validate that all components are properly initialized."""
        required_components = [
            'resource_monitor', 'transform_system', 'target_tracker',
            'strategy_module', 'velocity_control', 'recovery_module'
        ]
        for component in required_components:
            if not hasattr(self, component) or getattr(self, component) is None:
                raise RuntimeError(f"Required component '{component}' is not initialized")
       
    def _declare_parameters(self):
        """Declare and get all node parameters with improved defaults for Raspberry Pi 5."""
        # Declare parameters with improved defaults
        self.declare_parameters(
            namespace='',
            parameters=[
                # Linear X velocity PID parameters - adjusted for moderate velocity increase
                ('linear_x_kp', 1.2),  # controls overshoot
                ('linear_x_ki', 0.05), # handle steady state errors
                ('linear_x_kd', 0.25), # controls dampening during approach
                ('linear_x_min', 0.0),
                ('linear_x_max', 0.1),  
                
                # Linear Y velocity PID parameters - improved lateral damping
                ('linear_y_kp', 0.08),
                ('linear_y_ki', 0.06),  # Reduced from 0.08
                ('linear_y_kd', 0.4),   # Increased from 0.12
                ('linear_y_min', -0.2),
                ('linear_y_max', 0.3),
                
                # Angular velocity PID parameters - improved to prevent overshoot
                ('angular_kp', 0.9),   # Reduced from 1.5
                ('angular_ki', 0.1),   # Reduced from 0.05
                ('angular_kd', 0.8),   # Reduced from 0.8
                ('angular_min', -0.5),
                ('angular_max', 0.7),
                
                # Control parameters
                ('min_distance', 0.9),
                ('max_distance', 2.0),
                ('target_offset_x', 0.0),
                ('target_offset_y', 0.0),
                ('target_update_rate', 3.0),   # CHANGED: Default rate from 20Hz to 3Hz for Pi
                ('diagnostics_rate', 0.5),
                ('debug_level', 1),
                ('adaptive_gains', True),
                ('use_lateral_control', True),
                
                # Balanced error thresholds - increased angular threshold
                ('distance_threshold', 0.08),
                ('lateral_threshold', 0.06),  # Increased from 0.05
                ('angular_threshold', 2.5),   # Increased from 1.5 to 2.5 degrees
                
                # New parameter for scaling angular threshold when at target distance
                ('angular_at_target_factor', 2.5),  # Multiply threshold by this when at target distance
                
                # Resource monitoring parameters
                ('adaptive_control_rate', True),      # CHANGED: Enable by default
                ('enable_resource_monitoring', True), # CHANGED: Enable by default
                ('cpu_high_threshold', 60.0),         # CHANGED: Lower threshold for Pi
                ('cpu_low_threshold', 30.0),          # CHANGED: Lower threshold for Pi
                
                # Performance optimization
                ('enable_transform_caching', True),
                ('transform_cache_ttl', 1.0),
                
                # Strategy configuration
                ('angular_first_control', True),
                ('strategy_blend_duration', 0.2),  # Faster blending
                ('coordinated_movement', True),
                
                # Target filter parameters
                ('filter_buffer_size', 8),
                ('prediction_horizon', 0.3),
                
                # Approach configuration
                ('approach_distance', 0.3),    # Distance at which to start slowing down
                ('min_approach_factor', 0.2),  # Minimum velocity factor when very close
                
                # New optimized control parameters
                ('use_simplified_control_when_possible', True),
                ('cpu_optimization_threshold', 70.0),      # CHANGED: Lower threshold for Pi
                ('use_fast_trigonometry', True),
                
                # ADDED: New rate control parameters for fusion rate adaptation
                ('min_control_rate', 2.0),       # Minimum control rate (Hz)
                ('max_control_rate', 5.0),       # Maximum control rate (Hz)
                ('enable_fusion_rate_detection', True), # Enable fusion rate detection
                ('fresh_data_timeout', 0.5),     # Maximum age for fresh data (seconds)
                ('stale_data_timeout', 1.0),     # Maximum age for stale data (seconds)
                
                # ADDED: New CPU throttling parameters
                ('cpu_throttle_interval', 0.5),   # Check CPU every 0.5 seconds
                ('enable_cycle_skipping', True),  # Enable cycle skipping for CPU relief
                ('max_cpu_skip_threshold', 80.0), # Skip cycles when CPU exceeds this
            ]
        )
        
        # Get key parameters
        self.linear_x_kp = self.get_parameter('linear_x_kp').value
        self.linear_x_ki = self.get_parameter('linear_x_ki').value
        self.linear_x_kd = self.get_parameter('linear_x_kd').value
        self.linear_x_min = self.get_parameter('linear_x_min').value
        self.linear_x_max = self.get_parameter('linear_x_max').value
        
        self.linear_y_kp = self.get_parameter('linear_y_kp').value
        self.linear_y_ki = self.get_parameter('linear_y_ki').value
        self.linear_y_kd = self.get_parameter('linear_y_kd').value
        self.linear_y_min = self.get_parameter('linear_y_min').value
        self.linear_y_max = self.get_parameter('linear_y_max').value
        
        self.angular_kp = self.get_parameter('angular_kp').value
        self.angular_ki = self.get_parameter('angular_ki').value
        self.angular_kd = self.get_parameter('angular_kd').value
        self.angular_min = self.get_parameter('angular_min').value
        self.angular_max = self.get_parameter('angular_max').value
        
        # Get error thresholds
        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.lateral_threshold = self.get_parameter('lateral_threshold').value
        self.angular_threshold = self.get_parameter('angular_threshold').value
        self.angular_at_target_factor = self.get_parameter('angular_at_target_factor').value
        
        # Get approach parameters
        self.approach_distance = self.get_parameter('approach_distance').value
        self.min_approach_factor = self.get_parameter('min_approach_factor').value
        
        # Get resource monitoring parameters
        self.adaptive_control_rate = self.get_parameter('adaptive_control_rate').value
        self.enable_resource_monitoring = self.get_parameter('enable_resource_monitoring').value
        self.cpu_high_threshold = self.get_parameter('cpu_high_threshold').value
        self.cpu_low_threshold = self.get_parameter('cpu_low_threshold').value
        
        # Get transform parameters
        self.enable_transform_caching = self.get_parameter('enable_transform_caching').value
        self.transform_cache_ttl = self.get_parameter('transform_cache_ttl').value
        
        # Get movement strategy parameters
        self.angular_first_control = self.get_parameter('angular_first_control').value
        self.strategy_blend_duration = self.get_parameter('strategy_blend_duration').value
        self.coordinated_movement = self.get_parameter('coordinated_movement').value
        
        # Target filter parameters
        self.filter_buffer_size = self.get_parameter('filter_buffer_size').value
        self.prediction_horizon = self.get_parameter('prediction_horizon').value
        
        # Target update rate from parameters
        # CHANGED: Using a lower default rate (3Hz) instead of 20Hz
        self.update_rate = self.get_parameter('target_update_rate').value
        
        # All other parameter assignments would be here
        self.diagnostics_rate = self.get_parameter('diagnostics_rate').value
        self.debug_level = self.get_parameter('debug_level').value
        
        # Get optimization parameters
        self.use_simplified_control_when_possible = self.get_parameter('use_simplified_control_when_possible').value
        self.cpu_optimization_threshold = self.get_parameter('cpu_optimization_threshold').value
        self.use_fast_trigonometry = self.get_parameter('use_fast_trigonometry').value
        
        # ADDED: Get new rate control parameters
        self.min_control_rate = self.get_parameter('min_control_rate').value
        self.max_control_rate = self.get_parameter('max_control_rate').value
        self.enable_fusion_rate_detection = self.get_parameter('enable_fusion_rate_detection').value
        self.fresh_data_timeout = self.get_parameter('fresh_data_timeout').value
        self.stale_data_timeout = self.get_parameter('stale_data_timeout').value
        
        # ADDED: Get new CPU throttling parameters
        self.cpu_throttle_interval = self.get_parameter('cpu_throttle_interval').value
        self.enable_cycle_skipping = self.get_parameter('enable_cycle_skipping').value
        self.max_cpu_skip_threshold = self.get_parameter('max_cpu_skip_threshold').value
        
        # Desired distance (cached calculation)
        self.desired_distance = 1.0  # Default value
        
        # Log important parameters
        self.get_logger().info(
            f"Control rate settings: base={self.update_rate:.1f}Hz, "
            f"min={self.min_control_rate:.1f}Hz, max={self.max_control_rate:.1f}Hz, "
            f"adaptive={self.adaptive_control_rate}"
        )
        
        self.get_logger().info(
            f"Controller parameters: linear_x=[{self.linear_x_kp}, {self.linear_x_ki}, {self.linear_x_kd}], "
            f"linear_y=[{self.linear_y_kp}, {self.linear_y_ki}, {self.linear_y_kd}], "
            f"angular=[{self.angular_kp}, {self.angular_ki}, {self.angular_kd}]"
        )
        
        self.get_logger().info(
            f"Error thresholds: distance={self.distance_threshold}, "
            f"lateral={self.lateral_threshold}, angular={self.angular_threshold} "
            f"(at_target_factor={self.angular_at_target_factor})"
        )
        
        self.get_logger().info(
            f"Approach configuration: approach_distance={self.approach_distance}m, "
            f"min_approach_factor={self.min_approach_factor}"
        )
        
        # Log optimization parameters
        self.get_logger().info(
            f"Optimization settings: simplified_control={self.use_simplified_control_when_possible}, "
            f"cpu_threshold={self.cpu_optimization_threshold}, "
            f"fast_trig={self.use_fast_trigonometry}"
        )
        
        # Log freshness parameters
        self.get_logger().info(
            f"Data freshness: fresh_timeout={self.fresh_data_timeout:.1f}s, "
            f"stale_timeout={self.stale_data_timeout:.1f}s, "
            f"fusion_detection={self.enable_fusion_rate_detection}"
        )
               
    def _init_controllers(self):
        """Initialize the controllers with improved tuning for controlled velocity."""
        try:
            self.pid_linear_x, self.distance_error_tracker = PIDControllers.create_controller_with_tracker(
                PIDControllers.ControllerType.LINEAR_X,
                self.linear_x_kp, self.linear_x_ki, self.linear_x_kd,
                self.linear_x_min, self.linear_x_max,
                "distance", throttled_logger, max_history=8
            )
            self.pid_linear_y, self.lateral_error_tracker = PIDControllers.create_controller_with_tracker(
                PIDControllers.ControllerType.LINEAR_Y,
                self.linear_y_kp, self.linear_y_ki, self.linear_y_kd,
                self.linear_y_min, self.linear_y_max,
                "lateral", throttled_logger, max_history=8
            )
            self.pid_angular, self.angular_error_tracker = PIDControllers.create_controller_with_tracker(
                PIDControllers.ControllerType.ANGULAR,
                self.angular_kp, self.angular_ki, self.angular_kd,
                self.angular_min, self.angular_max,
                "angular", throttled_logger, max_history=8
            )
            self.pid_linear_x.validate_initialization()
            self.pid_linear_y.validate_initialization()
            self.pid_angular.validate_initialization()
            self.coordinated_controller = PIDControllers.CoordinatedController(
                self.pid_linear_y, 
                self.pid_angular,
                throttled_logger,
                {
                    'coupling_factor': 0.3,
                    'smoothing_factor': 0.7,
                    'min_angle_for_reduction': 0.06,
                    'zero_angle_threshold': 0.02,
                    'max_angle_factor': 0.2,
                    'same_sign_scale': 0.7,
                    'opposite_sign_scale': 1.1,
                }
            )
            self.get_logger().info("PID controllers initialized successfully with error trackers")
        except Exception as e:
            handle_initialization_error(
                self.get_logger(),
                "Failed to initialize PID controllers",
                e
            )

    def _init_memory_pools(self):
        """Setup memory pools and reusable objects for efficiency."""
        # Initialize object pool manager
        self.object_pool = ObjectPoolManager(max_twist=10, max_vector3=15)
        
        # Pre-allocate commonly used arrays
        self._limited_velocities = np.zeros(3, dtype=np.float32)
        self._prev_velocities = np.zeros(3, dtype=np.float32)
        self._target_velocities = np.zeros(3, dtype=np.float32)
        self._vel_diffs = np.zeros(3, dtype=np.float32)
        
        # Pre-allocate reusable message objects
        self._cmd_vel_msg = Twist()
        self._diag_msg = Float32MultiArray()
        self._diag_data = np.zeros(14, dtype=np.float32)
        
        # Pre-allocated objects for frequent operations
        self._key_tuple = ["none", "none", "none"]  # Use list instead of tuple for mutability
        
        # Pre-allocated velocity tuple
        self._velocity_tuple = [0.0, 0.0, 0.0]
        
        # Pre-allocated velocity change check
        self._velocity_change_check = [False, False, False]
        
        # Pre-allocated error container
        self._current_errors = [0.0, 0.0, 0.0]  # distance, lateral, angular
        
        # Adaptive control rate calculation variables
        self._adaptive_rate_history = deque(maxlen=10)
        self._last_rate_adjustment_time = time.time()
        
        # ADDED: Fusion rate tracking variables
        self._detected_fusion_rate = 1.0  # Default assumption (1Hz)
        self._fusion_rate_updated = False
        self._last_fusion_check_time = time.time()
        
        # ADDED: Skip cycle flag
        self._skip_next_cycle = False
        self._skipped_cycle_count = 0
    
    def _init_state_variables(self):
        """Initialize all state tracking variables."""
        # Robot state
        self.robot_state = "initializing"
        self.previous_state = None
        self.last_control_time = time.time()
        
        # Robot orientation
        self.robot_orientation = 0.0  # Current yaw in radians
        self.last_orientation_time = None  # Time of last orientation update
        
        # For tracking when state changed (for duration calculations)
        self._last_state_change_time = time.time()
        
        # For tracking time between events
        self.last_velocity_log_time = 0.0
        self.last_cpu_warning_time = 0.0
        self.last_pool_log_time = 0.0
        
        # Motion smoothing
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        self.last_logged_cmd = (0.0, 0.0, 0.0)
        
        # Diagnostic information
        self.cycle_count = 0
        
        # Stopped state tracking with hysteresis
        self._robot_stopped = False
        self._stop_time = 0.0
        self._last_stop_position = (0.0, 0.0, 0.0)
        self._movement_hysteresis = 0.0  # Used to prevent oscillating between movement/stopped states
        
        # Recovery state tracking
        self.in_recovery = False
        self.recovery_start_time = 0.0
        self.recovery_phase = "none"  # none, stop, orient, approach
        self.force_target_reacquisition = False
        
        # Flag to track if we're shutting down
        self._shutting_down = False
        
        # Optimization tracking
        self._using_simplified_control = False
        self._computation_level = 3  # Full computation
        self._last_full_computation_time = 0.0
        self._simplified_control_count = 0
        
        # ADDED: Data freshness tracking
        self._data_freshness_level = "unknown"  # unknown, fresh, stale, critical
        
        # ADDED: Event-based control tracking
        self._last_timer_execution = 0.0
        self._last_event_execution = 0.0
        self._event_control_count = 0
        self._timer_control_count = 0
        
        # ADDED: CPU throttling tracking
        self._last_cpu_check_time = 0.0

        self.get_logger().info(
            f"Initialized state: desired_distance={self.desired_distance:.3f}m, "
            f"robot_state={self.robot_state}"
        )
    
    def _setup_subscriptions(self):
        """Set up all subscriptions for this node."""
        self.state_sub = self.create_subscription(
            String,
            TOPICS["input"]["state"],
            self.state_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.target_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["target"],
            self.target_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Subscribe to orientation data
        self.orientation_sub = self.create_subscription(
            Vector3Stamped,
            TOPICS["input"]["orientation"],
            self.orientation_callback,
            10,
            callback_group=self.callback_group
        )
    
    def _setup_publishers(self):
        """Set up all publishers for this node."""
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            TOPICS["output"]["cmd_vel"],
            10
        )
        
        self.pid_diag_pub = self.create_publisher(
            Float32MultiArray,
            TOPICS["output"]["diagnostics"],
            10
        )
        
        # Performance metrics publisher
        self.performance_pub = self.create_publisher(
            String,
            TOPICS["output"]["performance"],
            10
        )
    
    def _setup_timers(self):
        """Set up timer callbacks for periodic tasks with tiered frequencies."""
        # Main control loop timer
        self.timer = self.create_timer(1.0 / self.update_rate, self.control_loop_callback)
        
        # Diagnostic timer
        self.diagnostic_timer = self.create_timer(1.0 / self.diagnostics_rate, self.publish_diagnostics)
        
        # Resource monitoring timer
        self.resource_timer = self.create_timer(self.cpu_throttle_interval, self._update_resource_monitoring)
        
    def orientation_callback(self, msg):
        """Handle orientation updates from the IMU with improved transform handling."""
        # Extract yaw (z component) from the Vector3Stamped message
        raw_orientation = msg.vector.z
        
        # Store timestamp for freshness checking
        self.last_orientation_time = time.time()
        
        # Check if transform system is ready before attempting transforms
        if (not hasattr(self, 'transform_utils') or 
            not self.transform_utils.is_transform_system_ready()):
            # If transforms aren't ready, use raw orientation
            self.robot_orientation = raw_orientation
            return
        
        # If we need to transform the orientation to another frame
        if self.transform_utils.imu_frame != self.transform_utils.reference_frame:
            try:
                # First approach: direct transform using quaternion math
                transform = self.transform_utils.get_transform_between_frames(
                    self.transform_utils.imu_frame, 
                    self.transform_utils.reference_frame
                )
                if (transform and hasattr(transform, 'transform') and 
                    hasattr(transform.transform, 'rotation')):
                    # Extract quaternion components from transform
                    qx = transform.transform.rotation.x
                    qy = transform.transform.rotation.y
                    qz = transform.transform.rotation.z
                    qw = transform.transform.rotation.w
                    
                    # Create forward unit vector in IMU frame
                    # Use optimized trigonometry if enabled
                    if self.use_fast_trigonometry:
                        forward_x = self.fast_trig.cos(raw_orientation)
                        forward_y = self.fast_trig.sin(raw_orientation)
                    else:
                        forward_x = math.cos(raw_orientation)
                        forward_y = math.sin(raw_orientation)
                    forward_z = 0.0
                    
                    # Calculate rotation matrix elements from quaternion - optimized
                    # Pre-calculate common terms
                    xx = qx * qx
                    xy = qx * qy
                    xz = qx * qz
                    xw = qx * qw
                    yy = qy * qy
                    yz = qy * qz
                    yw = qy * qw
                    zz = qz * qz
                    zw = qz * qw
                    
                    # Apply rotation to forward vector
                    r00 = 1.0 - 2.0 * (yy + zz)
                    r01 = 2.0 * (xy - zw)
                    r02 = 2.0 * (xz + yw)
                    r10 = 2.0 * (xy + zw)
                    r11 = 1.0 - 2.0 * (xx + zz)
                    r12 = 2.0 * (yz - xw)
                    
                    # Transform forward vector
                    tx = r00 * forward_x + r01 * forward_y + r02 * forward_z
                    ty = r10 * forward_x + r11 * forward_y + r12 * forward_z
                    
                    # Calculate new orientation angle
                    # Use optimized atan2 if enabled
                    if self.use_fast_trigonometry:
                        self.robot_orientation = self.fast_trig.atan2(ty, tx)
                    else:
                        self.robot_orientation = math.atan2(ty, tx)
                else:
                    # If transform not available, use raw orientation
                    self.robot_orientation = raw_orientation
                    
            except Exception as e:
                # In case of error, fall back to raw orientation
                if self.debug_level >= 2:
                    self.get_logger().warning(f"Orientation transform error: {str(e)}")
                self.robot_orientation = raw_orientation
        else:
            # No transform needed
            self.robot_orientation = raw_orientation
        
        # Log orientation updates at high debug level
        if self.debug_level >= 3:
            # Use pre-computed table for degrees conversion if fast trig is enabled
            if self.use_fast_trigonometry:
                orientation_degrees = self.robot_orientation * 57.29578  # 180/pi
            else:
                orientation_degrees = math.degrees(self.robot_orientation)
                
            self.get_logger().debug(f"Orientation update: yaw={orientation_degrees:.2f}°")
    
    def _is_orientation_fresh(self):
        """Check if orientation data is fresh enough to use."""
        if self.last_orientation_time is None:
            return False
            
        current_time = time.time()
        age = current_time - self.last_orientation_time
        
        # Consider orientation data older than 0.5 seconds as stale
        return age < 0.5
    
    def target_callback(self, msg):
        """
        Process incoming target message with event-based control capability.
        
        This enhanced callback not only updates target data but also
        can trigger immediate control cycle for improved responsiveness.
        """
        # Early exit if shutting down
        if self._shutting_down:
            return
        
        # ADDED: Record time for event tracking
        event_time = time.time()
        
        # Update target data in the target tracking module
        update_success = self.target_tracker.update_target(msg, self.debug_level)
        
        if not update_success:
            return
        
        # ADDED: Check for fusion rate updates
        if self.enable_fusion_rate_detection:
            try:
                # Make sure the method exists and handle possible exceptions
                if hasattr(self.target_tracker, 'get_fusion_rate'):
                    fusion_rate, was_updated = self.target_tracker.get_fusion_rate()
                    
                    if was_updated and fusion_rate > 0:  # Ensure rate is positive
                        # Update our stored fusion rate
                        self._detected_fusion_rate = fusion_rate
                        self._fusion_rate_updated = True
                        
                        # Update the resource monitor with the new fusion rate
                        if hasattr(self.resource_monitor, 'set_fusion_rate'):
                            self.resource_monitor.set_fusion_rate(fusion_rate)
                            
                        self.get_logger().info(f"Detected fusion rate: {fusion_rate:.2f}Hz")
            except Exception as e:
                self.get_logger().warning(f"Error getting fusion rate: {str(e)}")
        
        # ADDED: Event-based control execution
        # Only trigger if we're in tracking mode and it's been long enough since last execution
        # This prevents excessive CPU usage while ensuring responsiveness
        try:
            # Initialize time tracking attributes if they don't exist
            if not hasattr(self, '_last_event_execution'):
                self._last_event_execution = 0.0
            if not hasattr(self, '_last_timer_execution'):
                self._last_timer_execution = 0.0
            if not hasattr(self, '_event_control_count'):
                self._event_control_count = 0
                
            if (self.robot_state == "tracking" and 
                (event_time - self._last_event_execution) > (1.0 / self.max_control_rate) and
                (event_time - self._last_timer_execution) > 0.05):  # Minimum 50ms between executions
                
                # Execute control loop directly in response to new data
                self.execute_control_cycle(event_triggered=True)
                self._last_event_execution = event_time
                self._event_control_count += 1
        except Exception as e:
            self.get_logger().error(f"Error in event-based control: {str(e)}")
    
    def state_callback(self, msg):
        """Handle robot state updates with improved recovery behavior."""
        new_state = msg.data
        
        # If state changed, handle the transition
        if new_state != self.robot_state:
            # Log state transition with built-in throttling
            self.get_logger().info(
                f"STATE TRANSITION: {self.robot_state} → {new_state}",
                throttle_duration_sec=LOG_THROTTLE_STATE
            )

            # Additional logging with explicit time tracking
            current_time = time.time()
            
            # Additional logging
            if self.debug_level >= 2:
                # Calculate time in previous state
                time_in_state = current_time - self._last_state_change_time
                # Log the duration
                self.get_logger().info(f"Time in state '{self.robot_state}': {time_in_state:.2f}s")
                # Update last change time
                self._last_state_change_time = current_time
            
            self.previous_state = self.robot_state
            self.robot_state = new_state
            
            # Handle recovery state transitions
            if new_state == "recovery":
                self.in_recovery = True
                self.recovery_start_time = time.time()
                self.recovery_phase = "stop"
                # Stop robot immediately when entering recovery
                stop_cmd = self.recovery_module.stop_robot()
                self.cmd_vel_pub.publish(stop_cmd)
                self.get_logger().info("Entering recovery mode - stopping robot")
            elif self.previous_state == "recovery" and new_state != "recovery":
                self.in_recovery = False
                self.recovery_phase = "none"
                self.get_logger().info("Exiting recovery mode")
            
            # Complete controller reset when transitioning between tracking and other states
            if new_state == "tracking" or self.previous_state == "tracking":
                self._complete_controller_reset()
                
                # Force target reacquisition when re-entering tracking mode
                if new_state == "tracking":
                    self.force_target_reacquisition = True
                
            # If we're not in tracking mode, ensure the robot is stopped
            # (unless it's in searching or lost_ball mode, where the state manager controls motion)
            if new_state != "tracking" and new_state != "searching" and new_state != "lost_ball":
                stop_cmd = self.recovery_module.stop_robot()
                self.cmd_vel_pub.publish(stop_cmd)
    
    def _handle_non_tracking_state(self):
        """Handle robot behavior when not in tracking mode."""
        # When not tracking, ensure robot is stopped (unless controlled by another node)
        if self.robot_state not in ["searching", "lost_ball"]:
            stop_cmd = self.recovery_module.stop_robot()
            self.cmd_vel_pub.publish(stop_cmd)
        return True  # Indicate that the method handled the situation

    def _calculate_errors(self):
        """Calculate tracking errors using filtered values."""
        # Get filtered position data from target tracker
        position_data = self.target_tracker.get_position_data()
        
        # Handle the case where position data is missing or incomplete
        if not position_data or not all(k in position_data for k in ['distance', 'lateral', 'bearing']):
            # Set default values
            return 0.0, 0.0, 0.0, 0.0
        
        # Extract position values directly to local variables to avoid repeated dict lookups
        distance = position_data['distance']
        lateral = position_data['lateral']
        bearing = position_data['bearing']
        
        # Calculate errors using pre-allocated array
        self._current_errors[0] = distance - self.desired_distance  # distance_error
        self._current_errors[1] = lateral - 0.0                    # lateral_error 
        self._current_errors[2] = bearing                          # angular_error
        
        # Convert angular error to degrees for logging
        # Use optimized conversion if fast trigonometry is enabled
        if self.use_fast_trigonometry:
            angular_degrees = bearing * 57.29578  # 180/pi - faster than math.degrees
        else:
            angular_degrees = math.degrees(bearing)
        
        return distance, lateral, bearing, angular_degrees

    def _handle_stop_conditions(self, distance, lateral, angular_degrees, dt):
        """Check and handle stop conditions if needed."""
        # Check if we need to reset stopped state based on errors with enhanced hysteresis
        state_reset = self._reset_stopped_state_if_needed(
            self._current_errors[0], 
            self._current_errors[1], 
            angular_degrees
        )
        
        # If state was reset, skip the normal stop condition check this cycle
        if not state_reset:
            # Check stop conditions
            should_stop, stop_reason = self._evaluate_stop_conditions(
                distance, lateral, angular_degrees, self._robot_stopped
            )
            
            if should_stop:
                if not self._robot_stopped:
                    self.get_logger().info(stop_reason)
                    self._robot_stopped = True
                    self._stop_time = time.time()
                    # Generate stop command
                    stop_cmd = self.recovery_module.stop_robot()
                    # Publish stop command to actually stop the robot
                    self.cmd_vel_pub.publish(stop_cmd)  # Add this line to actually send the 
                return True  # Handled by stopping the robot
        
        # Update error trackers
        self.distance_error_tracker.update(self._current_errors[0], dt)
        self.lateral_error_tracker.update(self._current_errors[1], dt)
        self.angular_error_tracker.update(self._current_errors[2], dt)
        
        return False  # Not handled, continue with normal control

    def _determine_and_apply_strategy(self, dt):
        """Determine movement strategy and apply it to calculate velocities."""
        # Determine the optimal movement strategy with hysteresis
        strategy = self.strategy_module.determine_strategy(
            self._current_errors[0], 
            self._current_errors[1], 
            math.degrees(self._current_errors[2]),
            self._robot_stopped
        )
        
        # Apply strategy to movement decisions (use object attributes)
        use_forward = strategy.use_forward
        use_lateral = strategy.use_lateral
        use_angular = strategy.use_angular
        
        forward_scale = strategy.forward_scale
        lateral_scale = strategy.lateral_scale
        angular_scale = strategy.angular_scale
        
        if self.debug_level >= 3:
            strategy_log = (
                f"Using strategy: {strategy.strategy_name}, "
                f"forward={use_forward}, lateral={use_lateral}, angular={use_angular}"
            )
            throttled_logger.info(strategy_log, throttle_duration_sec=0.5, log_id='strategy')

        # Compute velocities based on the selected strategy
        current_time = time.time()
        
        # Compute velocities
        if self.coordinated_movement and use_lateral and use_angular:
            # Use coordinated controller for lateral and angular movements
            linear_x_velocity = self.pid_linear_x.compute(
                self._current_errors[0], 
                current_time, 
                not use_forward,
                self.distance_error_tracker.get_trend()
            )
            
            # Use coordinated control for lateral and angular velocities
            lateral_velocity, angular_velocity = self.coordinated_controller.compute(
                self._current_errors[1],   # lateral error
                self._current_errors[2],   # angular error
                current_time,              # current time
                self.robot_orientation     # current orientation from IMU
            )
            
            # Disable individual components if strategy requires
            if not use_lateral:
                lateral_velocity = 0.0
            if not use_angular:
                angular_velocity = 0.0
        else:
            # Traditional separate PID controllers
            linear_x_velocity = self.pid_linear_x.compute(
                self._current_errors[0], 
                current_time, 
                not use_forward,
                self.distance_error_tracker.get_trend()
            )
            
            lateral_velocity = self.pid_linear_y.compute(
                self._current_errors[1], 
                current_time, 
                not use_lateral,
                self.lateral_error_tracker.get_trend()
            )
            
            angular_velocity = self.pid_angular.compute(
                self._current_errors[2], 
                current_time, 
                not use_angular,
                self.angular_error_tracker.get_trend()
            )
        
        # Apply strategy scaling factors
        linear_x_velocity *= forward_scale
        lateral_velocity *= lateral_scale
        angular_velocity *= angular_scale
        
        # ADDED: Apply freshness-based velocity scaling
        if self._data_freshness_level == "stale":
            # Reduced speed (50%) for stale data
            stale_scale = 0.5
            linear_x_velocity *= stale_scale
            lateral_velocity *= stale_scale
            angular_velocity *= stale_scale
            
            if self.debug_level >= 1:
                self.get_logger().warning(
                    f"Using stale sensor data - scaling velocities to {stale_scale*100:.0f}%",
                    throttle_duration_sec=2.0
                )
        elif self._data_freshness_level == "critical" or self._data_freshness_level == "invalid":
            # Stop for critical or invalid data
            linear_x_velocity = 0.0
            lateral_velocity = 0.0
            angular_velocity = 0.0
        
        if self.debug_level >= 2:
            velocity_log = (
                f"After scaling: linear_x={linear_x_velocity:.3f}, "
                f"lateral={lateral_velocity:.3f}, angular={angular_velocity:.3f}"
            )
            throttled_logger.info(velocity_log, throttle_duration_sec=1.0, log_id='velocity')
            
        return linear_x_velocity, lateral_velocity, angular_velocity

    def _apply_and_publish_velocities(self, linear_x_velocity, lateral_velocity, angular_velocity):
        """Apply velocity limits and publish command velocities."""
        # Apply velocity and acceleration limits
        position_data = self.target_tracker.get_position_data()
        target_distance = position_data['distance'] if position_data and 'distance' in position_data else 0.0
        # Use VelocityControlModule for all velocity state
        limited_velocities = self.velocity_control.process_velocities(
            linear_x_velocity, 
            lateral_velocity, 
            angular_velocity, 
            target_distance, 
            self.desired_distance,
            freshness_level=self._data_freshness_level
        )
        linear_x_velocity = limited_velocities[0]
        lateral_velocity = limited_velocities[1]
        angular_velocity = limited_velocities[2]
        # Publish command
        cmd_vel_msg = self._cmd_vel_msg
        cmd_vel_msg.linear.x = float(linear_x_velocity)
        cmd_vel_msg.linear.y = float(lateral_velocity)
        cmd_vel_msg.angular.z = float(angular_velocity)
        self.cmd_vel_pub.publish(cmd_vel_msg)
        # Update error trackers only if significant movement is occurring
        motion_occurred = False
        if abs(linear_x_velocity) > 0.05:
            self.distance_error_tracker.record_correction()
            motion_occurred = True
        if abs(lateral_velocity) > 0.05:
            self.lateral_error_tracker.record_correction()
            motion_occurred = True
        if abs(angular_velocity) > 0.1:
            self.angular_error_tracker.record_correction()
            motion_occurred = True
        return motion_occurred

    def _check_data_freshness(self):
        """
        Check the freshness of target data and update system state accordingly.
        
        Returns:
            tuple: (is_fresh, freshness_level, age)
        """
        try:
            # Get freshness information from the target tracker
            is_fresh, freshness_level, data_age = self.target_tracker.is_target_fresh(
                max_age=self.fresh_data_timeout
            )
            
            # Validate returned data (guard against unexpected return values)
            if not isinstance(freshness_level, str):
                self.get_logger().warning(f"Invalid freshness level type: {type(freshness_level)}")
                freshness_level = "critical"  # Default to critical for safety
            if not isinstance(data_age, (int, float)):
                self.get_logger().warning(f"Invalid data age type: {type(data_age)}")
                data_age = 999.0  # Default to high age for safety
            
            # Check if the freshness level has changed
            if freshness_level != self._data_freshness_level:
                # Log the transition
                if self._data_freshness_level != "unknown":  # Skip initial setting
                    self.get_logger().info(
                        f"Data freshness changed: {self._data_freshness_level} → {freshness_level} "
                        f"(age: {data_age:.3f}s)"
                    )
                # Record the time of state change
                self._freshness_state_change_time = time.time()
                # Update the freshness level
                self._data_freshness_level = freshness_level
            
            #caller handles the stop when data is not fresh
            return is_fresh, freshness_level, data_age            
        except Exception as e:
            self.get_logger().error(f"Error checking data freshness: {str(e)}")
            # Default to critical for safety in case of errors
            return False, "critical", 999.0

    def _determine_computation_level(self):
        """
        Determine the level of computation needed for the current cycle.
        
        Returns:
            int: 0-3 indicating computation level (0=minimal, 3=full)
        Notes:
            - Full computation (level 3) is now allowed up to 85% CPU usage.
            - Scaling down only starts above 85% CPU.
        """
        computation_level = 3
        cpu_usage = self.resource_monitor.current_cpu_usage
        # Adjusted thresholds for high baseline CPU
        if cpu_usage > 95.0:
            return 0  # Minimal computation only at extreme load
        elif cpu_usage > 90.0:
            computation_level = min(computation_level, 1)
        elif cpu_usage > 85.0:
            computation_level = min(computation_level, 2)
        # Always perform full computation periodically to ensure accuracy
        time_since_full = time.time() - self._last_full_computation_time
        if time_since_full > 0.5:
            return 3
        position_data = self.target_tracker.get_position_data()
        if not position_data or not all(k in position_data for k in ['distance', 'lateral', 'bearing']):
            return computation_level
        distance_error = abs(position_data['distance'] - self.desired_distance)
        lateral_error = abs(position_data['lateral'])
        angular_error = abs(position_data['bearing'])
        if self.use_fast_trigonometry:
            angular_degrees = angular_error * 57.29578
        else:
            angular_degrees = math.degrees(angular_error)
        # Use velocity_control for velocity state
        velocity_stable = all(abs(v) < 0.05 for v in self.velocity_control.last_cmd_vel)
        if (distance_error < self.distance_threshold * 0.3 and
            lateral_error < self.lateral_threshold * 0.3 and
            angular_degrees < self.angular_threshold * 0.3):
            computation_level = min(computation_level, 0)
        elif (distance_error < self.distance_threshold * 0.7 and
              lateral_error < self.lateral_threshold * 0.7 and
              angular_degrees < self.angular_threshold * 0.7):
            computation_level = min(computation_level, 1)
        elif (distance_error < self.distance_threshold * 1.5 and
              lateral_error < self.lateral_threshold * 1.5 and
              angular_degrees < self.angular_threshold * 1.5):
            computation_level = min(computation_level, 2)
        if velocity_stable and computation_level > 0:
            computation_level -= 1
        if (self.distance_error_tracker.get_trend() == "stable" and
            self.lateral_error_tracker.get_trend() == "stable" and
            self.angular_error_tracker.get_trend() == "stable"):
            computation_level = min(computation_level, 1)
        return computation_level

    def calculate_adaptive_rate(self, base_rate, cpu_usage):
        """
        Calculate adaptive control rate based on CPU usage and fusion rate.
        
        Args:
            base_rate: The base update rate
            cpu_usage: Current CPU usage percentage
            
        Returns:
            float: Adjusted update rate
        Notes:
            - Full control rate is now allowed up to 85% CPU usage.
            - Scaling down only starts above 85% CPU.
        """
        try:
            current_time = time.time()
            if not isinstance(base_rate, (int, float)) or base_rate <= 0:
                self.get_logger().warning(f"Invalid base_rate: {base_rate}, using default 3.0Hz")
                base_rate = 3.0
            if not isinstance(cpu_usage, (int, float)) or cpu_usage < 0:
                self.get_logger().warning(f"Invalid cpu_usage: {cpu_usage}, using 50%")
                cpu_usage = 50.0
            if not hasattr(self, '_last_rate_adjustment_time'):
                self._last_rate_adjustment_time = 0.0
            if current_time - self._last_rate_adjustment_time < 1.0:
                return getattr(self, '_current_rate', base_rate)
            self._last_rate_adjustment_time = current_time
            if hasattr(self, '_fusion_rate_updated') and hasattr(self, '_detected_fusion_rate') and \
               self._fusion_rate_updated and self.enable_fusion_rate_detection:
                fusion_rate = self._detected_fusion_rate
                if fusion_rate > 0 and fusion_rate < 100:
                    adjusted_base_rate = min(max(fusion_rate * 1.2, self.min_control_rate), self.max_control_rate)
                    if abs(adjusted_base_rate - base_rate) > 0.3:
                        self.get_logger().info(
                            f"Adjusted base rate using fusion detection: {base_rate:.1f}Hz -> {adjusted_base_rate:.1f}Hz "
                            f"(fusion_rate: {fusion_rate:.1f}Hz)"
                        )
                    base_rate = adjusted_base_rate
            # Adjusted thresholds for high baseline CPU
            if cpu_usage > 95.0:
                new_rate = self.min_control_rate
            elif cpu_usage > 90.0:
                new_rate = base_rate * 0.5
            elif cpu_usage > 85.0:
                new_rate = base_rate * 0.7
            elif cpu_usage > 80.0:
                new_rate = base_rate * 0.85
            elif cpu_usage < 40.0:
                new_rate = base_rate * 1.1
            else:
                new_rate = base_rate
            new_rate = max(self.min_control_rate, min(self.max_control_rate, new_rate))
            self._current_rate = new_rate
            if not hasattr(self, '_adaptive_rate_history'):
                self._adaptive_rate_history = deque(maxlen=10)
            self._adaptive_rate_history.append((current_time, new_rate))
            if hasattr(self, '_last_logged_rate') and abs(new_rate - self._last_logged_rate) > 0.5:
                self.get_logger().info(
                    f"Adaptive rate adjusted: {self._last_logged_rate:.1f}Hz -> {new_rate:.1f}Hz "
                    f"(CPU: {cpu_usage:.1f}%)"
                )
                self._last_logged_rate = new_rate
            elif not hasattr(self, '_last_logged_rate'):
                self._last_logged_rate = new_rate
            return new_rate
        except Exception as e:
            self.get_logger().error(f"Error in calculate_adaptive_rate: {str(e)}")
            return max(base_rate, self.min_control_rate)

    def execute_control_cycle(self, event_triggered=False):
        """Execute one complete control cycle with improved error handling and state management."""
        try:
            # Skip if shutting down
            if hasattr(self, '_shutting_down') and self._shutting_down:
                return
            
            # Skip if transform system is not initialized and required for this operation
            if (hasattr(self, 'transform_system') and 
                not self.transform_system.is_transform_system_ready()):
                return
            
            # Track execution time source for metrics
            if event_triggered:
                self._last_event_execution = time.time()
            else:
                # Initialize timer control count if it doesn't exist
                if not hasattr(self, '_timer_control_count'):
                    self._timer_control_count = 0
                self._last_timer_execution = time.time()
                self._timer_control_count += 1
            
            # Mark cycle start for performance tracking
            self.cycle_start_time = time.time()
            
            # Calculate dt since last control
            current_time = time.time()
            dt = current_time - self.last_control_time
            # Ensure dt is reasonable (protect against time jumps)
            if dt > 1.0:
                self.get_logger().warning(f"Large time step detected: {dt:.3f}s - capping at 0.1s")
                dt = 0.1  # Cap at 100ms to prevent instability
            elif dt <= 0.0:
                dt = 0.001  # Ensure positive dt to prevent division by zero
            
            self.last_control_time = current_time
            
            # Increment cycle counter
            self.cycle_count += 1
            
            # Check data freshness - critical for safety
            is_fresh, freshness_level, data_age = self._check_data_freshness()
            
            # Log periodic status updates (once every 50 cycles)
            if self.cycle_count % 50 == 0:
                self._log_periodic_status()
            
            # Handle critical data freshness (prioritized over other state handling)
            if freshness_level == "critical":
                if not self._robot_stopped:
                    self.get_logger().warning(f"CRITICAL DATA AGE: {data_age:.3f}s - Safety stop triggered")
                    stop_cmd = self.recovery_module.stop_robot()
                    self.cmd_vel_pub.publish(stop_cmd)  # Ensure command is published
                    self._robot_stopped = True
                    self._stop_time = time.time()
                return  # Exit early when data is critically stale
                
            # Special handling for recovery mode (higher priority than other states)
            if self.in_recovery:
                # Get position data for recovery
                position_data = self.target_tracker.get_position_data()
                orientation_data = {'yaw': self.robot_orientation}
                
                # Delegate to recovery module
                cmd_vel, is_complete = self.recovery_module.handle_recovery(
                    current_time, position_data, orientation_data
                )
                
                # Always publish the command if in recovery mode
                self.cmd_vel_pub.publish(cmd_vel)
                
                # If recovery is complete, transition back to normal mode
                if is_complete:
                    self.in_recovery = False
                    self.get_logger().info("Recovery sequence completed")
                
                return  # Exit after handling recovery
            
            # Handle non-tracking states (searching/lost_ball have their own handling)
            if self.robot_state != "tracking":
                if self._handle_non_tracking_state():
                    return  # Exit after handling non-tracking state
            
            # Check if orientation data is fresh (prevents race conditions)
            if not self._is_orientation_fresh():
                if self.debug_level >= 2:
                    self.get_logger().warning("Skipping control cycle - orientation data is stale", 
                                            throttle_duration_sec=1.0)
                return
            
            # Determine computation level needed
            # The key CPU optimization - avoid expensive calculations when appropriate
            if self.use_simplified_control_when_possible:
                computation_level = self._determine_computation_level()
                
                if computation_level < 3 and self._apply_simplified_control(computation_level):
                    # Simplified control was applied, skip the rest of the processing
                    return
                else:
                    # Record that we did full computation
                    self._last_full_computation_time = time.time()
            else:
                # Standard control path
                self._using_simplified_control = False
                
            # Perform expensive transform operations at reduced frequency
            #self._optimize_transforms_and_filtering()
            
            # Calculate current errors
            distance, lateral, bearing, angular_degrees = self._calculate_errors()
            
            if self.debug_level >= 2:
                debug_msg = (
                    f"PRE-STOP CHECK: distance={distance:.3f}m (target={self.desired_distance:.3f}m), "
                    f"lateral={lateral:.3f}m, angular={angular_degrees:.2f}°, "
                    f"is_stopped={self._robot_stopped}"
                )
                self.get_logger().info(debug_msg, throttle_duration_sec=2.0)
            
            # Check stop conditions and handle if needed
            if self._handle_stop_conditions(distance, lateral, angular_degrees, dt):
                return  # Exit if stop conditions were met
            
            # Apply stale data handling - reduce velocity if data is stale
            velocity_scale = 1.0
            if freshness_level == "stale":
                # Apply significant reduction for stale data
                velocity_scale = 0.5  # 50% reduction
                if self.debug_level >= 1:
                    self.get_logger().warning(
                        f"Using stale sensor data ({data_age:.3f}s old) - scaling velocities to {velocity_scale*100:.0f}%",
                        throttle_duration_sec=2.0
                    )
            
            # Determine strategy and calculate velocities
            linear_x_velocity, lateral_velocity, angular_velocity = self._determine_and_apply_strategy(dt)
            
            # Apply stale data velocity scaling
            if velocity_scale < 1.0:
                linear_x_velocity *= velocity_scale
                lateral_velocity *= velocity_scale
                angular_velocity *= velocity_scale
            
            # Guard against NaN values which can cause silent failures
            if (math.isnan(linear_x_velocity) or math.isnan(lateral_velocity) or 
                math.isnan(angular_velocity)):
                self.get_logger().error("NaN velocity detected - resetting to zero")
                linear_x_velocity = 0.0
                lateral_velocity = 0.0
                angular_velocity = 0.0
            
            # Apply velocity limits and publish commands
            motion_occurred = self._apply_and_publish_velocities(linear_x_velocity, lateral_velocity, angular_velocity)
            
            # Update performance stats
            cycle_duration = time.time() - self.cycle_start_time
            self.update_performance_stats(cycle_duration)
                
        except Exception as e:
            stack_trace = traceback.format_exc()
        
            # Log the error with stack trace
            self.get_logger().error(f"Unexpected error in execute_control_cycle: {str(e)}\nStack trace:\n{stack_trace}")

            self.get_logger().error(f"Unexpected error in control cycle: {str(e)}")
            # Try to safely stop the robot
            try:
                stop_cmd = self.recovery_module.stop_robot()
                self.cmd_vel_pub.publish(stop_cmd)  # Ensure the stop command is published
            except Exception as stop_error:
                self.get_logger().error(f"Failed to stop robot after error: {str(stop_error)}")

    def _apply_simplified_control(self, computation_level):
        """
        Apply a simplified control update based on the computation level.
        
        Args:
            computation_level: 0-3 indicating computation level (0=minimal, 3=full)
            
        Returns:
            bool: True if simplified control was applied, False if full computation is needed
        """
        # Validate input
        if not isinstance(computation_level, int) or computation_level < 0:
            self.get_logger().warning(f"Invalid computation level: {computation_level}, using full computation")
            return False
            
        if computation_level == 3:
            # Full computation needed
            return False
            
        # Track that we're using simplified control
        self._using_simplified_control = True
        
        # Ensure _simplified_control_count is initialized
        if not hasattr(self, '_simplified_control_count'):
            self._simplified_control_count = 0
            
        self._simplified_control_count += 1
        
        # Get current time for performance tracking
        start_time = time.time()
        
        # For minimal computation (level 0), just dampen previous velocities
        if computation_level == 0:
            # Reuse previous velocity commands with significant damping
            damping = 0.85  # Stronger reduction for level 0
            
            # Use pre-allocated message
            cmd_vel_msg = self._cmd_vel_msg
            cmd_vel_msg.linear.x = float(self.last_cmd_vel[0] * damping)
            cmd_vel_msg.linear.y = float(self.last_cmd_vel[1] * damping)
            cmd_vel_msg.angular.z = float(self.last_cmd_vel[2] * damping)
            
            # Publish command
            self.cmd_vel_pub.publish(cmd_vel_msg)
            
            # Update history without expensive diagnostics
            new_velocity = (cmd_vel_msg.linear.x, cmd_vel_msg.linear.y, cmd_vel_msg.angular.z)
            self.last_cmd_vel = new_velocity
            
        # For basic computation (level 1), apply simple proportional control
        elif computation_level == 1:
            # Get current errors
            position_data = self.target_tracker.get_position_data()
            
            if position_data and all(k in position_data for k in ['distance', 'lateral', 'bearing']):
                # Calculate basic errors
                distance_error = position_data['distance'] - self.desired_distance
                lateral_error = position_data['lateral']
                angular_error = position_data['bearing']
                
                # Simple proportional control with reduced gains
                kp_factor = 0.7  # Reduce gains for smoother control
                linear_x = max(-0.1, min(0.1, distance_error * self.linear_x_kp * kp_factor))
                lateral_y = max(-0.1, min(0.1, lateral_error * self.linear_y_kp * kp_factor))
                angular_z = max(-0.3, min(0.3, angular_error * self.angular_kp * kp_factor))
                
                # Apply damping from previous velocities for smoothness
                damping = 0.3  # 30% of previous velocity
                linear_x = linear_x * (1.0 - damping) + self.last_cmd_vel[0] * damping
                lateral_y = lateral_y * (1.0 - damping) + self.last_cmd_vel[1] * damping
                angular_z = angular_z * (1.0 - damping) + self.last_cmd_vel[2] * damping
                
                # Publish
                cmd_vel_msg = self._cmd_vel_msg
                cmd_vel_msg.linear.x = float(linear_x)
                cmd_vel_msg.linear.y = float(lateral_y)
                cmd_vel_msg.angular.z = float(angular_z)
                
                self.cmd_vel_pub.publish(cmd_vel_msg)
                
                # Update history
                new_velocity = (linear_x, lateral_y, angular_z)
                self.last_cmd_vel = new_velocity
            else:
                # No valid position data, apply strong damping
                damping = 0.7
                cmd_vel_msg = self._cmd_vel_msg
                cmd_vel_msg.linear.x = float(self.last_cmd_vel[0] * damping)
                cmd_vel_msg.linear.y = float(self.last_cmd_vel[1] * damping)
                cmd_vel_msg.angular.z = float(self.last_cmd_vel[2] * damping)
                
                # Publish command
                self.cmd_vel_pub.publish(cmd_vel_msg)
                
                # Update history
                new_velocity = (cmd_vel_msg.linear.x, cmd_vel_msg.linear.y, cmd_vel_msg.angular.z)
                self.last_cmd_vel = new_velocity
                
        # For medium computation (level 2), use PID but skip coordinated control
        elif computation_level == 2:
            # Calculate errors - reuse existing method
            distance, lateral, bearing, angular_degrees = self._calculate_errors()
            
            # Calculate current time and dt
            current_time = time.time()
            dt = current_time - self.last_control_time
            self.last_control_time = current_time
            
            # Update error trackers
            self.distance_error_tracker.update(self._current_errors[0], dt)
            self.lateral_error_tracker.update(self._current_errors[1], dt)
            self.angular_error_tracker.update(self._current_errors[2], dt)
            
            # Use separate PID controllers for efficiency - no coordination
            linear_x_velocity = self.pid_linear_x.compute(
                self._current_errors[0], 
                current_time, 
                False,
                self.distance_error_tracker.get_trend()
            )
            
            lateral_velocity = self.pid_linear_y.compute(
                self._current_errors[1], 
                current_time, 
                False,
                self.lateral_error_tracker.get_trend()
            )
            
            angular_velocity = self.pid_angular.compute(
                self._current_errors[2], 
                current_time, 
                False,
                self.angular_error_tracker.get_trend()
            )
            
            # Apply velocity limits directly (simplified)
            linear_x_velocity = max(self.linear_x_min, min(self.linear_x_max, linear_x_velocity))
            lateral_velocity = max(self.linear_y_min, min(self.linear_y_max, lateral_velocity))
            angular_velocity = max(self.angular_min, min(self.angular_max, angular_velocity))
            
            # Publish
            cmd_vel_msg = self._cmd_vel_msg
            cmd_vel_msg.linear.x = float(linear_x_velocity)
            cmd_vel_msg.linear.y = float(lateral_velocity)
            cmd_vel_msg.angular.z = float(angular_velocity)
            
            self.cmd_vel_pub.publish(cmd_vel_msg)
            
            # Update history
            new_velocity = (linear_x_velocity, lateral_velocity, angular_velocity)
            self.last_cmd_vel = new_velocity
        
        # Calculate cycle duration for performance monitoring
        cycle_duration = time.time() - start_time
        if hasattr(self.resource_monitor, '_update_cycle_stats'):
            self.resource_monitor._update_cycle_stats(cycle_duration)
        
        return True
        
    def control_loop_callback(self):
        """Regular control loop to calculate and publish velocity commands with CPU optimization."""
        try:
            # Skip if shutting down
            if hasattr(self, '_shutting_down') and self._shutting_down:
                return
            
            # Skip if transform system is not initialized and required
            if (hasattr(self, 'transform_system') and 
                not self.transform_system.is_transform_system_ready()):
                
                # Log at most once per 20 cycles to avoid spamming
                if self.cycle_count % 20 == 0:
                    status = self.transform_system.get_status()
                    self.get_logger().warn(
                        f"Control loop waiting for transform initialization: {status['message']}"
                    )
                    
                    # Try to restart initialization if it's in error state
                    if status['status'] == TransformStatus.ERROR:
                        self.get_logger().warn("Attempting to restart transform initialization")
                        self.transform_system.start_initialization()
                
                # Still increment cycle count
                self.cycle_count += 1
                return
                    
            # Apply adaptive rate control if enabled
            if (self.adaptive_control_rate and 
                hasattr(self.resource_monitor, 'current_cpu_usage') and 
                hasattr(self, 'update_rate')):
                
                # Calculate adaptive rate
                adaptive_rate = self.calculate_adaptive_rate(
                    self.update_rate, 
                    self.resource_monitor.current_cpu_usage
                )
                
                # Check if we should skip this cycle based on adaptive rate
                current_time = time.time()
                time_since_last = current_time - getattr(self, 'last_control_time', current_time)
                
                # Skip if the time since last execution is too short
                if (time_since_last < (1.0 / adaptive_rate)):
                    return
            
                # Skip this cycle if requested for CPU relief
                if hasattr(self.resource_monitor, 'should_skip_cycle') and self.resource_monitor.should_skip_cycle():
                    # Ensure counter exists before incrementing
                    if not hasattr(self, '_skipped_cycle_count'):
                        self._skipped_cycle_count = 0
                    self._skipped_cycle_count += 1
                    return
                
                # Execute the control cycle
                self.execute_control_cycle(event_triggered=False)
                    
        except Exception as e:
            
            stack_trace = traceback.format_exc()
        
            # Log the error with stack trace
            self.get_logger().error(f"Unexpected error in control_loop_callback: {str(e)}\nStack trace:\n{stack_trace}")

            self.get_logger().error(f"Unexpected error in control_loop_callback: {str(e)}")
            # Try to safely stop the robot
            try:
                self.recovery_module.stop_robot()
            except Exception as stop_error:
                self.get_logger().error(f"Failed to stop robot after error: {str(stop_error)}")

    
    def _evaluate_stop_conditions(self, distance, lateral, angular_degrees, is_stopped):
        """
        Evaluate if the robot should stop based on current errors.
        
        Args:
            distance: Current distance to target
            lateral: Current lateral offset
            angular_degrees: Current angular error in degrees
            is_stopped: Whether the robot is currently stopped
            
        Returns:
            tuple: (should_stop, reason) - True if robot should stop, False if it should move
        """
        # Calculate error values
        distance_error = abs(distance - self.desired_distance)
        lateral_error = abs(lateral)
        angular_error = abs(angular_degrees)
        
        # Start with base thresholds
        distance_threshold = self.distance_threshold
        lateral_threshold = self.lateral_threshold
        angular_threshold = self.angular_threshold
        
        # Increase angular threshold when at target distance
        if distance_error < self.distance_threshold * 1.5:
            angular_threshold *= self.angular_at_target_factor
            
            if self.debug_level >= 2:
                threshold_msg = f"At target distance: increased angular threshold to {angular_threshold:.2f}°"
                self.get_logger().debug(threshold_msg, throttle_duration_sec=1.0)
        
        # Apply state-dependent hysteresis
        if is_stopped:
            # Higher thresholds to start moving (requires larger errors)
            hysteresis = 1.5 + self._movement_hysteresis
            distance_threshold *= hysteresis
            lateral_threshold *= hysteresis
            angular_threshold *= hysteresis
            
            # Cap maximum thresholds
            distance_threshold = min(distance_threshold, 0.2)
            lateral_threshold = min(lateral_threshold, 0.15)
            angular_threshold = min(angular_threshold, 15.0)
        else:
            # Lower thresholds to stop (more precision when near target)
            hysteresis = 0.8
            distance_threshold *= hysteresis
            lateral_threshold *= hysteresis
            angular_threshold *= hysteresis
        
        # Log thresholds for debugging
        if self.debug_level >= 2:
            debug_msg = (
                f"Stop thresholds: d={distance_error:.3f}/{distance_threshold:.3f}, "
                f"l={lateral_error:.3f}/{lateral_threshold:.3f}, "
                f"a={angular_error:.2f}/{angular_threshold:.2f}"
            )
            self.get_logger().debug(debug_msg, throttle_duration_sec=1.0)
        
        # Check if any error exceeds thresholds
        if (distance_error > distance_threshold or
            lateral_error > lateral_threshold or
            angular_error > angular_threshold):
            
            reason = (
                f"Movement needed: distance_error={distance_error:.3f}m, "
                f"lateral_error={lateral_error:.3f}m, "
                f"angular_error={angular_error:.2f}°"
            )
            return False, reason  # Return False to indicate robot should NOT stop
        
        # All errors within thresholds - robot should stop
        reason = (
            f"Target reached: distance_error={distance_error:.3f}m, "
            f"lateral_error={lateral_error:.3f}m, "
            f"angular_error={angular_error:.2f}°"
        )
        
        # Accumulate hysteresis for sustained stops
        if not is_stopped:
            self._movement_hysteresis += 0.05
            self._movement_hysteresis = min(0.3, self._movement_hysteresis)  # Cap at 0.3
        
        return True, reason  # Return True to indicate robot SHOULD stop

    def _reset_stopped_state_if_needed(self, distance_error, lateral_error, angular_error):
        """
        Reset stopped state if significant movement is required.
        
        Args:
            distance_error: Error in distance (meters)
            lateral_error: Error in lateral position (meters)
            angular_error: Error in angular position (degrees)
            
        Returns:
            bool: True if stopped state was reset, False otherwise
        """
        # If already moving, no need to reset
        if not self._robot_stopped:
            return False
        
        # Handle initialization for first movement after startup
        if not hasattr(self, '_initial_movement_boost'):
            self._initial_movement_boost = True
        
        # Calculate hysteresis factor based on stop duration
        stop_duration = time.time() - self._stop_time
        
        # Apply reduced hysteresis for first movement after startup
        if self._initial_movement_boost:
            hysteresis = 0.5  # Much lower hysteresis for first movement
            self._initial_movement_boost = False
        else:
            # Regular hysteresis calculation
            hysteresis = min(1.1, 1.0 + stop_duration * 0.1)
        
        # Calculate movement thresholds with hysteresis
        distance_threshold = self.distance_threshold * hysteresis * 0.7
        lateral_threshold = self.lateral_threshold * hysteresis * 0.7
        
        # Adjust angular threshold based on distance to target
        if abs(distance_error) < self.distance_threshold * 1.2:
            # More lenient when at target distance
            angular_threshold = self.angular_threshold * self.angular_at_target_factor * hysteresis
        else:
            # Standard threshold otherwise
            angular_threshold = self.angular_threshold * hysteresis
        
        # Check if any error exceeds thresholds
        if (abs(distance_error) > distance_threshold or
            abs(lateral_error) > lateral_threshold or
            abs(angular_error) > angular_threshold):
            
            # Log the decision to exit stopped state
            log_msg = (
                f"Exiting stopped state - Movement required: "
                f"distance_error={distance_error:.3f}m(threshold={distance_threshold:.3f}), "
                f"lateral_error={lateral_error:.3f}m(threshold={lateral_threshold:.3f}), "
                f"angular_error={angular_error:.2f}°(threshold={angular_threshold:.2f})"
            )
            self.get_logger().info(log_msg)
            
            # Reset stopped state
            self._robot_stopped = False
            
            # Reset movement hysteresis
            self._movement_hysteresis = 0.0
            
            return True
        
        return False
    
    def _complete_controller_reset(self):
        """Complete reset of all controllers and error states."""
        # Reset all PID controllers
        self.pid_linear_x.reset()
        self.pid_linear_y.reset()
        self.pid_angular.reset()
        # Reset coordinated controller
        self.coordinated_controller.reset()
        # Reset all error trackers
        self.distance_error_tracker.reset()
        self.lateral_error_tracker.reset()
        self.angular_error_tracker.reset()
        # Reset strategy module
        self.strategy_module.reset()
        # Reset velocity control module (centralized state)
        self.velocity_control.reset()
        # Reset target tracker if needed
        if hasattr(self, 'target_tracker'):
            if hasattr(self.target_tracker, 'reset'):
                self.target_tracker.reset()
        # Reset movement hysteresis
        self._movement_hysteresis = 0.0
        # Set stopped state
        self._robot_stopped = True
        self._stop_time = time.time()
        # Reset computation tracking
        self._using_simplified_control = False
        self._last_full_computation_time = time.time()
        # Reset data freshness tracking
        self._data_freshness_level = "unknown"
        self.get_logger().info("Complete controller reset performed")
    
    def _log_periodic_status(self):
        """Log periodic status updates."""
        if self.debug_level < 1:
            return
        perf_stats = self.resource_monitor.get_performance_stats()
        if not perf_stats or not all(k in perf_stats for k in ['cpu_avg', 'cycle_time_ms', 'update_rate']):
            perf_stats = {
                'cpu_avg': 0.0,
                'cycle_time_ms': 0.0,
                'update_rate': getattr(self, 'update_rate', 3.0),
                'skips': 0
            }
        strategy_name = getattr(self.strategy_module, 'current_strategy', 'unknown')
        total_cycles = max(1, self._event_control_count + self._timer_control_count)
        event_ratio = self._event_control_count / total_cycles * 100.0
        status_msg = (
            f"Status: Robot state={self.robot_state}, "
            f"Strategy={strategy_name}, "
            f"CPU={perf_stats['cpu_avg']:.1f}%, "
            f"Cycle time={perf_stats['cycle_time_ms']:.2f}ms, "
            f"Rate={perf_stats['update_rate']:.1f}Hz, "
            f"Simplified={self._using_simplified_control}, "
            f"Freshness={self._data_freshness_level}, "
            f"Event-driven={event_ratio:.1f}%, "
            f"Skips={self._skipped_cycle_count}"
        )
        throttled_logger.info(status_msg, throttle_duration_sec=LOG_THROTTLE_CONTROL, log_id='periodic_status')
    
    def calculate_adaptive_rate(self, base_rate, cpu_usage):
        """
        Calculate adaptive control rate based on CPU usage and fusion rate.
        
        Args:
            base_rate: The base update rate
            cpu_usage: Current CPU usage percentage
            
        Returns:
            float: Adjusted update rate
        Notes:
            - Full control rate is now allowed up to 85% CPU usage.
            - Scaling down only starts above 85% CPU.
        """
        try:
            # Store current time
            current_time = time.time()
            
            # Validate input parameters to prevent errors
            if not isinstance(base_rate, (int, float)) or base_rate <= 0:
                self.get_logger().warning(f"Invalid base_rate: {base_rate}, using default 3.0Hz")
                base_rate = 3.0
                
            if not isinstance(cpu_usage, (int, float)) or cpu_usage < 0:
                self.get_logger().warning(f"Invalid cpu_usage: {cpu_usage}, using 50%")
                cpu_usage = 50.0
            
            # Only adjust rate periodically to avoid oscillation
            if not hasattr(self, '_last_rate_adjustment_time'):
                self._last_rate_adjustment_time = 0.0
                
            if current_time - self._last_rate_adjustment_time < 1.0:  # At most once per second
                return getattr(self, '_current_rate', base_rate)
                
            self._last_rate_adjustment_time = current_time
            
            # Apply fusion rate consideration if detected
            if hasattr(self, '_fusion_rate_updated') and hasattr(self, '_detected_fusion_rate') and \
               self._fusion_rate_updated and self.enable_fusion_rate_detection:
                # Use fusion rate as a baseline if it's reliable
                fusion_rate = self._detected_fusion_rate
                
                # Ensure fusion rate is positive and reasonable
                if (fusion_rate > 0 and fusion_rate < 100):  # Sanity check
                    # Cap to reasonable limits
                    adjusted_base_rate = min(max(fusion_rate * 1.2, self.min_control_rate), self.max_control_rate)
                    
                    # Log if significant change
                    if abs(adjusted_base_rate - base_rate) > 0.3:
                        self.get_logger().info(
                            f"Adjusted base rate using fusion detection: {base_rate:.1f}Hz -> {adjusted_base_rate:.1f}Hz "
                            f"(fusion_rate: {fusion_rate:.1f}Hz)"
                        )
                        
                    base_rate = adjusted_base_rate
            
            # Calculate new rate based on CPU usage with progressive scaling
            if cpu_usage > 95.0:
                # Severe CPU load - drastic reduction
                new_rate = self.min_control_rate
            elif cpu_usage > 90.0:
                # Heavy CPU load
                new_rate = base_rate * 0.5
            elif cpu_usage > 85.0:
                # Moderate CPU load
                new_rate = base_rate * 0.7
            elif cpu_usage > 80.0:
                # Light CPU load
                new_rate = base_rate * 0.85
            elif cpu_usage < 30.0:
                # Very light CPU load - can increase slightly
                new_rate = base_rate * 1.1
            else:
                # Normal CPU load
                new_rate = base_rate
                
            # Constrain rate to reasonable bounds
            new_rate = max(self.min_control_rate, min(self.max_control_rate, new_rate))
            
            # Store current rate for reference
            self._current_rate = new_rate
            
            # Add to history (ensure history exists)
            if not hasattr(self, '_adaptive_rate_history'):
                self._adaptive_rate_history = deque(maxlen=10)
            self._adaptive_rate_history.append((current_time, new_rate))
            
            # Log significant rate changes
            if hasattr(self, '_last_logged_rate') and abs(new_rate - self._last_logged_rate) > 0.5:
                self.get_logger().info(
                    f"Adaptive rate adjusted: {self._last_logged_rate:.1f}Hz -> {new_rate:.1f}Hz "
                    f"(CPU: {cpu_usage:.1f}%)"
                )
                self._last_logged_rate = new_rate
            elif not hasattr(self, '_last_logged_rate'):
                self._last_logged_rate = new_rate
            
            return new_rate
            
        except Exception as e:
            self.get_logger().error(f"Error in calculate_adaptive_rate: {str(e)}")
            # Fall back to base rate or minimum rate in case of error
            return max(base_rate, self.min_control_rate)
    
    def _update_resource_monitoring(self):
        """Wrapper method to update resource monitoring."""
        try:
            # Update CPU stats if the method exists
            if hasattr(self.resource_monitor, 'update_cpu_stats'):
                self.resource_monitor.update_cpu_stats()
                
                # Check for high CPU - trigger cycle skipping if needed
                if (self.enable_cycle_skipping and 
                    hasattr(self.resource_monitor, 'current_cpu_usage') and
                    self.resource_monitor.current_cpu_usage > self.max_cpu_skip_threshold):
                    
                    self._skip_next_cycle = True
                    
                    current_time = time.time()
                    if current_time - self.last_cpu_warning_time >= 2.0:
                        self.get_logger().warning(
                            f"HIGH CPU LOAD: {self.resource_monitor.current_cpu_usage:.1f}% - skipping next cycle"
                        )
                        self.last_cpu_warning_time = current_time
                else:
                    self._skip_next_cycle = False
                
            # Log pool statistics periodically if in debug mode
            if self.debug_level >= 2 and hasattr(self, 'object_pool'):
                pool_stats = self.object_pool.get_stats()
                current_time = time.time()
                if current_time - self.last_pool_log_time >= 10.0:
                    pool_msg = (
                        f"Object pool stats: "
                        f"twist={pool_stats['twist_pool_size']}/{pool_stats['twist_max_usage']} "
                        f"(misses={pool_stats['twist_misses']}), "
                        f"vector3={pool_stats['vector3_pool_size']}/{pool_stats['vector3_max_usage']} "
                        f"(misses={pool_stats['vector3_misses']})"
                    )
                    self.get_logger().debug(pool_msg)
                    self.last_pool_log_time = current_time
        except Exception as e:
            self.get_logger().error(f"Error in resource monitoring: {str(e)}")
        
    def update_performance_stats(self, cycle_duration):
        """Update performance statistics for monitoring."""
        try:
            if hasattr(self.resource_monitor, '_update_cycle_stats'):
                self.resource_monitor._update_cycle_stats(cycle_duration)
                
                # Update adaptive rate if needed
                if self.adaptive_control_rate and hasattr(self, 'update_rate'):
                    adaptive_rate = self.calculate_adaptive_rate(
                        self.update_rate, 
                        self.resource_monitor.current_cpu_usage
                    )
                    # Store current rate
                    self._adaptive_rate = adaptive_rate
        except Exception as e:
            self.get_logger().warning(f"Error updating performance stats: {str(e)}")
    
    def publish_diagnostics(self):
        """Publish detailed diagnostic information at a slower rate."""
        try:
            if hasattr(self, '_shutting_down') and self._shutting_down:
                return
                
            # Calculate velocity statistics
            vel_data = self.velocity_control.get_velocity_history()
            if not vel_data:
                return
                
            # Use NumPy for vectorized calculations when possible
            if len(vel_data) > 0:
                # Convert to NumPy array for efficient calculation
                vel_array = np.array(vel_data)
                
                # Extract velocities by column (more efficient than list comprehensions)
                if vel_array.size > 0:  # Check that array has elements
                    lin_x_velocities = vel_array[:, 0]
                    lin_y_velocities = vel_array[:, 1]
                    ang_velocities = vel_array[:, 2]
                    
                    # Calculate statistics using NumPy
                    avg_lin_x_vel = np.mean(lin_x_velocities) if lin_x_velocities.size > 0 else 0.0
                    avg_lin_y_vel = np.mean(lin_y_velocities) if lin_y_velocities.size > 0 else 0.0
                    avg_ang_vel = np.mean(ang_velocities) if ang_velocities.size > 0 else 0.0
                else:
                    avg_lin_x_vel = avg_lin_y_vel = avg_ang_vel = 0.0
            else:
                avg_lin_x_vel = avg_lin_y_vel = avg_ang_vel = 0.0
            
            # Log detailed information with built-in throttling
            if hasattr(self, 'debug_level') and self.debug_level >= 1:
                # Get PID components
                p_x, i_x, d_x = self.pid_linear_x.get_components()
                p_y, i_y, d_y = self.pid_linear_y.get_components()
                p_a, i_a, d_a = self.pid_angular.get_components()
                
                # Get stats on simplified control usage
                simplified_pct = getattr(self, '_simplified_control_count', 0) / max(1, self.cycle_count) * 100.0
                
                diag_msg = (
                    f"DIAGNOSTICS: "
                    f"Avg Vel=[{avg_lin_x_vel:.2f}, {avg_lin_y_vel:.2f}, {avg_ang_vel:.2f}], "
                    f"PID X=[{p_x:.2f}, {i_x:.2f}, {d_x:.2f}], "
                    f"PID Y=[{p_y:.2f}, {i_y:.2f}, {d_y:.2f}], "
                    f"PID A=[{p_a:.2f}, {i_a:.2f}, {d_a:.2f}], "
                    f"Strategy={self.strategy_module.current_strategy}, "
                    f"Simp={simplified_pct:.1f}%, "
                    f"Freshness={self._data_freshness_level}"
                )
                throttled_logger.info(diag_msg, throttle_duration_sec=LOG_THROTTLE_DIAG, log_id='diagnostics')
            
            # Publish PID diagnostic data
            self._publish_pid_diagnostics()
            
            # Publish performance metrics
            self._publish_performance_metrics()
        except Exception as e:
            self.get_logger().error(f"Error in publish_diagnostics: {str(e)}")
    
    def _publish_pid_diagnostics(self):
        """Publish detailed PID diagnostics for analysis."""
        try:
            # Get PID components
            p_x, i_x, d_x = self.pid_linear_x.get_components()
            p_y, i_y, d_y = self.pid_linear_y.get_components()
            p_a, i_a, d_a = self.pid_angular.get_components()
            
            # Get current gains
            kp_x, ki_x, kd_x = self.pid_linear_x.get_current_gains()
            kp_y, ki_y, kd_y = self.pid_linear_y.get_current_gains()
            kp_a, ki_a, kd_a = self.pid_angular.get_current_gains()
            
            # Get current errors
            e_x = self.distance_error_tracker.current_error
            e_y = self.lateral_error_tracker.current_error
            e_a = self.angular_error_tracker.current_error
            
            # Pack all data into the array - no unnecessary float() conversions
            # Direct array access is more efficient
            self._diag_data[0] = p_x
            self._diag_data[1] = i_x
            self._diag_data[2] = d_x
            self._diag_data[3] = p_y
            self._diag_data[4] = i_y
            self._diag_data[5] = d_y
            self._diag_data[6] = p_a
            self._diag_data[7] = i_a
            self._diag_data[8] = d_a
            self._diag_data[9] = e_x
            self._diag_data[10] = e_y
            self._diag_data[11] = e_a
            self._diag_data[12] = kp_a  # Track angular P gain
            self._diag_data[13] = kd_a  # Track angular D gain
            
            # Update Float32MultiArray data
            self._diag_msg.data = self._diag_data.tolist()
            
            # Publish diagnostics
            self.pid_diag_pub.publish(self._diag_msg)
        except Exception as e:
            self.get_logger().error(f"Error in _publish_pid_diagnostics: {str(e)}")
    
    def _publish_performance_metrics(self):
        """Publish performance metrics for monitoring."""
        try:
            # Get performance stats
            perf_stats = self.resource_monitor.get_performance_stats()
            
            # Check if we have valid stats
            if not perf_stats or not all(k in perf_stats for k in ['cpu_avg', 'cycle_time_ms', 'update_rate']):
                # Create default stats if missing
                perf_stats = {
                    'cpu_avg': 0.0,
                    'cycle_time_ms': 0.0,
                    'skips': 0,
                    'update_rate': getattr(self, 'update_rate', 3.0)
                }
            
            # Get current strategy
            strategy_name = "unknown"
            if hasattr(self.strategy_module, 'current_strategy'):
                strategy_name = self.strategy_module.current_strategy
            
            # Add optimization stats
            adaptive_rate = getattr(self, '_adaptive_rate', perf_stats['update_rate'])
            using_simplified = 1 if getattr(self, '_using_simplified_control', False) else 0
            
            # Add freshness and event stats
            freshness_level = self._data_freshness_level
            event_ratio = 0.0
            if hasattr(self, '_event_control_count') and hasattr(self, '_timer_control_count'):
                total_cycles = max(1, self._event_control_count + self._timer_control_count)
                event_ratio = self._event_control_count / total_cycles * 100.0
            
            # Create performance message - use string formatting for better performance
            # in tight loops
            performance_msg = String()
            performance_msg.data = (
                '{{"cpu": {0:.1f}, '
                '"cycle_time_ms": {1:.2f}, '
                '"strategy": "{2}", '
                '"skips": {3}, '
                '"update_rate": {4:.1f}, '
                '"adaptive_rate": {5:.1f}, '
                '"simplified": {6}, '
                '"freshness": "{7}", '
                '"event_ratio": {8:.1f}}}'
            ).format(
                perf_stats["cpu_avg"],
                perf_stats["cycle_time_ms"],
                strategy_name,
                perf_stats.get("skips", self._skipped_cycle_count),
                perf_stats["update_rate"],
                adaptive_rate,
                using_simplified,
                freshness_level,
                event_ratio
            )
            
            # Publish
            self.performance_pub.publish(performance_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing performance metrics: {str(e)}")
    
    def prepare_shutdown(self):
        """Prepare for node shutdown."""
        if hasattr(self, '_shutting_down') and self._shutting_down:
            return  # Already shutting down
            
        print("Preparing for shutdown")  # Use print instead of ROS logger
        self._shutting_down = True
        
        # Cancel all timers to prevent them from continuing to fire
        for timer in [self.timer, self.diagnostic_timer, self.resource_timer, self.transform_check_timer]:
            if hasattr(self, timer.__name__) and timer.__name__ is not None:
                try:
                    timer.cancel()
                except Exception:
                    pass
                    
        try:
            # Create a stop command directly
            stop_cmd = Twist()  # All values default to 0.0
            
            # Try to publish but don't rely on logging
            try:
                self.cmd_vel_pub.publish(stop_cmd)
                print("Robot motion stopped during shutdown")
            except Exception:
                print("Failed to publish stop command - context may be invalid")
                
            # Quick sleep to allow message to be sent if context still valid
            time.sleep(0.1)
        except Exception as e:
            print(f"Error stopping robot during shutdown: {str(e)}")

# Main function
def main(args=None):
    """Main function to initialize and run the PID Controller node."""
    rclpy.init(args=args)
    node = None
    
    # Flag to track shutdown state
    shutdown_initiated = False
    
    try:
        print("=================================================")
        print("Optimized PID Controller for Basketball Tracking Robot")
        print("=================================================")
        try:
            node = OptimizedPIDControllerNode()
        except InitializationError as e:
            print(f"\nINITIALIZATION ERROR: {str(e)}")
            print("\nThe system cannot start due to critical initialization errors.")
            sys.exit(1)
            
        # Set up signal handler for graceful shutdown
        def signal_handler(sig, frame):
            nonlocal shutdown_initiated
            
            if shutdown_initiated:
                print("\nForce quitting (received multiple shutdown signals)")
                sys.exit(1)
                
            shutdown_initiated = True
            print("\nShutdown requested by user (Ctrl+C)")
            print("Stopping robot and shutting down...")
            
            # Stop the robot first, before rclpy.shutdown() invalidates the context
            if node is not None:
                # Use direct method to stop robot without logging
                stop_cmd = Twist()  # All values default to 0.0
                try:
                    node.cmd_vel_pub.publish(stop_cmd)
                    # Small delay to allow message to be sent
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error during emergency stop: {str(e)}")
                node._shutting_down = True
                
            # Important: Raise KeyboardInterrupt to break out of rclpy.spin()
            raise KeyboardInterrupt
            
        # Register the signal handler
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            # This will be raised by our signal handler
            print("Exiting main loop...")
        except Exception as e:
            print(f"\nRUNTIME ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    finally:
        try:
            # Restore original signal handler before cleanup
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            if node is not None:
                # No need to call prepare_shutdown() since signal handler does it
                node.destroy_node()
                print("Node destroyed successfully")
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
        
        print("Calling rclpy.shutdown()...")
        rclpy.shutdown()
        print("Shutdown complete")

if __name__ == '__main__':
    main()