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
from abc import ABC, abstractmethod

# Import modules from refactored files
from ball_chase.pid.pid_helpers import LightweightBuffer, CircularBuffer, ThrottledLogger, FastTrigonometry, ResourceMonitor
from ball_chase.pid.pid_target_filter import EnhancedTargetFilter, ErrorTracker
from ball_chase.pid.pid_computation import PIDControllers
from ball_chase.pid.pid_target_tracking import TargetTrackingModule, MovementStrategyModule, VelocityControlModule, TransformSystem
from ball_chase.pid.pid_target_tracking import RecoveryBehaviorModule, TransformStatus

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
LOG_THROTTLE_DIAG = 2.0        # Seconds between diagnostic logs


# Centralized object pool manager
class ObjectPoolManager:
    """Manages pools of reusable objects to reduce memory allocations."""
    def __init__(self, max_twist=10, max_vector3=15, max_float32multiarray=5, ttl=60.0):
        from geometry_msgs.msg import Twist, Vector3, PointStamped, Vector3Stamped, TransformStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Float32MultiArray, String
        from ball_chase.pid.pid_helpers import GenericObjectPool

        def reset_twist(twist):
            twist.linear.x = twist.linear.y = twist.linear.z = 0.0
            twist.angular.x = twist.angular.y = twist.angular.z = 0.0

        def reset_vector3(vec):
            vec.x = vec.y = vec.z = 0.0

        def reset_float32multiarray(arr):
            arr.data.clear()

        def reset_pointstamped(msg):
            msg.point.x = msg.point.y = msg.point.z = 0.0
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            msg.header.frame_id = ""

        def reset_vector3stamped(msg):
            msg.vector.x = msg.vector.y = msg.vector.z = 0.0
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            msg.header.frame_id = ""

        def reset_transformstamped(msg):
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            msg.header.frame_id = ""
            msg.child_frame_id = ""
            msg.transform.translation.x = 0.0
            msg.transform.translation.y = 0.0
            msg.transform.translation.z = 0.0
            msg.transform.rotation.x = 0.0
            msg.transform.rotation.y = 0.0
            msg.transform.rotation.z = 0.0
            msg.transform.rotation.w = 1.0

        def reset_odometry(msg):
            msg.pose.pose.position.x = 0.0
            msg.pose.pose.position.y = 0.0
            msg.pose.pose.position.z = 0.0
            msg.twist.twist.linear.x = 0.0
            msg.twist.twist.linear.y = 0.0
            msg.twist.twist.linear.z = 0.0
            # ...reset other fields as needed...

        def reset_string(msg):
            msg.data = ""

        self.pools = {
            'Twist': GenericObjectPool(Twist, max_twist, reset_twist, ttl),
            'Vector3': GenericObjectPool(Vector3, max_vector3, reset_vector3, ttl),
            'Float32MultiArray': GenericObjectPool(Float32MultiArray, max_float32multiarray, reset_float32multiarray, ttl),
            'PointStamped': GenericObjectPool(PointStamped, 10, reset_pointstamped, ttl),
            'Vector3Stamped': GenericObjectPool(Vector3Stamped, 10, reset_vector3stamped, ttl),
            'TransformStamped': GenericObjectPool(TransformStamped, 5, reset_transformstamped, ttl),
            'Odometry': GenericObjectPool(Odometry, 5, reset_odometry, ttl),
            'String': GenericObjectPool(String, 10, reset_string, ttl),
        }

    def get(self, msg_type):
        return self.pools[msg_type].get() if msg_type in self.pools else None

    def put(self, msg_type, obj):
        if msg_type in self.pools:
            self.pools[msg_type].put(obj)

    def get_stats(self):
        return {k: v.stats() for k, v in self.pools.items()}


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

class ParameterManager:
    """Handles parameter declaration and retrieval for the PID Controller Node."""
    def __init__(self, node):
        self.node = node
        self._declare_parameters()
        self._get_parameters()

    def _declare_parameters(self):
        self.node.declare_parameters(
            namespace='',
            parameters=[
                ('linear_x_kp', 1.0),
                ('linear_x_ki', 0.05),
                ('linear_x_kd', 0.3),
                ('linear_x_min', 0.0),
                ('linear_x_max', 0.1),
                ('linear_y_kp', 0.7),
                ('linear_y_ki', 0.005),
                ('linear_y_kd', 0.7),
                ('linear_y_min', -0.2),
                ('linear_y_max', 0.3),
                ('angular_kp', 0.8),
                ('angular_ki', 0.01),
                ('angular_kd', 0.9),
                ('angular_min', -0.5),
                ('angular_max', 0.5),
                ('min_distance', 0.9),
                ('max_distance', 2.0),
                ('target_offset_x', 0.0),
                ('target_offset_y', 0.0),
                ('target_update_rate', 3.0),
                ('diagnostics_rate', 0.5),
                ('debug_level', 1),
                ('adaptive_gains', True),
                ('use_lateral_control', True),
                ('distance_threshold', 0.1),
                ('lateral_threshold', 0.02),
                ('angular_threshold', 1.8),
                ('angular_at_target_factor', 1.0),
                ('adaptive_control_rate', True),
                ('enable_resource_monitoring', True),
                ('cpu_high_threshold', 80.0),
                ('cpu_low_threshold', 50.0),
                ('enable_transform_caching', True),
                ('transform_cache_ttl', 1.0),
                ('angular_first_control', True),
                ('strategy_blend_duration', 0.15),
                ('coordinated_movement', True),
                ('filter_buffer_size', 3),
                ('prediction_horizon', 0.04),
                ('approach_distance', 0.3),
                ('min_approach_factor', 0.15),
                ('use_simplified_control_when_possible', True),
                ('cpu_optimization_threshold', 70.0),
                ('use_fast_trigonometry', True),
                ('min_control_rate', 2.4),
                ('max_control_rate', 4.0),
                ('enable_fusion_rate_detection', True),
                ('fresh_data_timeout', 0.7),
                ('stale_data_timeout', 1.0),
                ('cpu_throttle_interval', 0.5),
                ('enable_cycle_skipping', False),
                ('max_cpu_skip_threshold', 90.0),
                ('coordinated_coupling_factor', 0.45),  
                ('coordinated_smoothing_factor', 0.6),  
                ('coordinated_min_angle_for_reduction', 0.1),
                ('coordinated_zero_angle_threshold', 0.015),
                ('coordinated_max_angle_factor', 0.2),
                ('coordinated_same_sign_scale', 0.8),  
                ('coordinated_opposite_sign_scale', 0.9),  
            ]
        )

    def _get_parameters(self):
        # PID gains
        self.linear_x_kp = self.node.get_parameter('linear_x_kp').value
        self.linear_x_ki = self.node.get_parameter('linear_x_ki').value
        self.linear_x_kd = self.node.get_parameter('linear_x_kd').value
        self.linear_x_min = self.node.get_parameter('linear_x_min').value
        self.linear_x_max = self.node.get_parameter('linear_x_max').value
        self.linear_y_kp = self.node.get_parameter('linear_y_kp').value
        self.linear_y_ki = self.node.get_parameter('linear_y_ki').value
        self.linear_y_kd = self.node.get_parameter('linear_y_kd').value
        self.linear_y_min = self.node.get_parameter('linear_y_min').value
        self.linear_y_max = self.node.get_parameter('linear_y_max').value
        self.angular_kp = self.node.get_parameter('angular_kp').value
        self.angular_ki = self.node.get_parameter('angular_ki').value
        self.angular_kd = self.node.get_parameter('angular_kd').value
        self.angular_min = self.node.get_parameter('angular_min').value
        self.angular_max = self.node.get_parameter('angular_max').value
        # Thresholds
        self.distance_threshold = self.node.get_parameter('distance_threshold').value
        self.lateral_threshold = self.node.get_parameter('lateral_threshold').value
        self.angular_threshold = self.node.get_parameter('angular_threshold').value
        self.angular_at_target_factor = self.node.get_parameter('angular_at_target_factor').value
        # Approach
        self.approach_distance = self.node.get_parameter('approach_distance').value
        self.min_approach_factor = self.node.get_parameter('min_approach_factor').value
        # Resource monitoring
        self.adaptive_control_rate = self.node.get_parameter('adaptive_control_rate').value
        self.enable_resource_monitoring = self.node.get_parameter('enable_resource_monitoring').value
        self.cpu_high_threshold = self.node.get_parameter('cpu_high_threshold').value
        self.cpu_low_threshold = self.node.get_parameter('cpu_low_threshold').value
        # Transform
        self.enable_transform_caching = self.node.get_parameter('enable_transform_caching').value
        self.transform_cache_ttl = self.node.get_parameter('transform_cache_ttl').value
        # Strategy
        self.angular_first_control = self.node.get_parameter('angular_first_control').value
        self.strategy_blend_duration = self.node.get_parameter('strategy_blend_duration').value
        self.coordinated_movement = self.node.get_parameter('coordinated_movement').value
        # Target filter
        self.filter_buffer_size = self.node.get_parameter('filter_buffer_size').value
        self.prediction_horizon = self.node.get_parameter('prediction_horizon').value
        # Target update rate
        self.update_rate = self.node.get_parameter('target_update_rate').value
        self.diagnostics_rate = self.node.get_parameter('diagnostics_rate').value
        self.debug_level = self.node.get_parameter('debug_level').value
        # Optimization
        self.use_simplified_control_when_possible = self.node.get_parameter('use_simplified_control_when_possible').value
        self.cpu_optimization_threshold = self.node.get_parameter('cpu_optimization_threshold').value
        self.use_fast_trigonometry = self.node.get_parameter('use_fast_trigonometry').value
        # Rate control
        self.min_control_rate = self.node.get_parameter('min_control_rate').value
        self.max_control_rate = self.node.get_parameter('max_control_rate').value
        self.enable_fusion_rate_detection = self.node.get_parameter('enable_fusion_rate_detection').value
        self.fresh_data_timeout = self.node.get_parameter('fresh_data_timeout').value
        self.stale_data_timeout = self.node.get_parameter('stale_data_timeout').value
        # CPU throttling
        self.cpu_throttle_interval = self.node.get_parameter('cpu_throttle_interval').value
        self.enable_cycle_skipping = self.node.get_parameter('enable_cycle_skipping').value
        self.max_cpu_skip_threshold = self.node.get_parameter('max_cpu_skip_threshold').value
        # Misc
        self.desired_distance = 1.0  # Default value
        # Coordinated controller parameters
        self.coordinated_coupling_factor = self.node.get_parameter('coordinated_coupling_factor').value
        self.coordinated_smoothing_factor = self.node.get_parameter('coordinated_smoothing_factor').value
        self.coordinated_min_angle_for_reduction = self.node.get_parameter('coordinated_min_angle_for_reduction').value
        self.coordinated_zero_angle_threshold = self.node.get_parameter('coordinated_zero_angle_threshold').value
        self.coordinated_max_angle_factor = self.node.get_parameter('coordinated_max_angle_factor').value
        self.coordinated_same_sign_scale = self.node.get_parameter('coordinated_same_sign_scale').value
        self.coordinated_opposite_sign_scale = self.node.get_parameter('coordinated_opposite_sign_scale').value


# Phase 3: Extract State Manager - Define component interfaces
class StateObserver(ABC):
    """Interface for classes that observe state changes."""
    @abstractmethod
    def on_state_change(self, old_state, new_state, reason=""):
        """Called when robot state changes."""
        pass
    
    @abstractmethod
    def on_freshness_change(self, freshness_level, data_age):
        """Called when data freshness level changes."""
        pass


# Extract these data classes from the original code
class RobotStateData:
    """Core robot state data"""
    def __init__(self):
        self.state = "initializing"
        self.previous_state = None
        self.last_state_change_time = time.time()
        self.robot_orientation = 0.0
        self.last_orientation_time = None
        self.cycle_count = 0
        self.force_target_reacquisition = False

class MovementStateData:
    """Movement-related state data"""
    def __init__(self):
        self.robot_stopped = True
        self.stop_time = time.time()
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        self.last_logged_cmd = (0.0, 0.0, 0.0)
        self.movement_hysteresis = 0.0
        self.last_stop_position = (0.0, 0.0, 0.0)
        self.using_simplified_control = False
        self.computation_level = 3
        self.last_full_computation_time = time.time()
        self.initial_movement_boost = True
        
        # Pre-allocated vectors for movement calculations
        self.prev_velocities = np.zeros(3, dtype=np.float32)
        self.target_velocities = np.zeros(3, dtype=np.float32)
        self.vel_diffs = np.zeros(3, dtype=np.float32)
        self.velocity_tuple = [0.0, 0.0, 0.0]
        self.velocity_change_check = [False, False, False]

class PerformanceData:
    """Performance monitoring state"""
    def __init__(self):
        self.simplified_control_count = 0
        self.last_cpu_check_time = 0.0
        self.last_cpu_warning_time = 0.0
        self.skipped_cycle_count = 0
        self.last_pool_log_time = 0.0
        self.current_rate = 0.0
        self.last_logged_rate = 0.0
        self.adaptive_rate_history = deque(maxlen=10)
        self.timer_control_count = 0
        self.event_control_count = 0
        self.last_timer_execution = 0.0
        self.last_event_execution = 0.0
        self.last_rate_adjustment_time = 0.0
        self.detected_fusion_rate = 1.0
        self.fusion_rate_updated = False
        self.adaptive_rate = 0.0

class DataFreshnessData:
    """Data freshness tracking state"""
    def __init__(self):
        self.level = "unknown"
        self.state_change_time = time.time()
        self.last_fusion_check_time = time.time()

class RecoveryStateData:
    """Recovery-related state data"""
    def __init__(self):
        self.in_recovery = False
        self.recovery_start_time = 0.0
        self.recovery_phase = "none"


# Phase 3: Enhanced State Manager 
class EnhancedStateManager:
    """Enhanced state manager that provides observation capabilities"""
    def __init__(self):
        self.robot = RobotStateData()
        self.movement = MovementStateData()
        self.perf = PerformanceData()
        self.freshness = DataFreshnessData()
        self.recovery = RecoveryStateData()
        self.shutting_down = False
        self.last_control_time = time.time()
        self.skip_next_cycle = False
        self.observers = []
        
        # Pre-allocate numpy arrays for movement calculations
        self._limited_velocities = np.zeros(3, dtype=np.float32)

    def register_observer(self, observer):
        """Register a state observer."""
        if isinstance(observer, StateObserver):
            self.observers.append(observer)
        
    def transition_state(self, new_state, reason=""):
        """Handle state transition and return True if state changed"""
        if new_state == self.robot.state:
            return False
            
        # Store previous state and update current state
        old_state = self.robot.state
        self.robot.previous_state = old_state
        self.robot.state = new_state
        self.robot.last_state_change_time = time.time()
        
        # Notify observers
        for observer in self.observers:
            observer.on_state_change(old_state, new_state, reason)
            
        return True
        
    def update_freshness(self, freshness_level, data_age):
        """Update data freshness state."""
        if freshness_level != self.freshness.level:
            old_level = self.freshness.level
            self.freshness.level = freshness_level
            self.freshness.state_change_time = time.time()
            
            # Notify observers
            for observer in self.observers:
                observer.on_freshness_change(freshness_level, data_age)
        
    def is_state(self, state_name):
        """Efficient state check"""
        return self.robot.state == state_name
        
    def was_state(self, state_name):
        """Check previous state"""
        return self.robot.previous_state == state_name


# Phase 1: Extract Diagnostics Publisher
class DiagnosticsPublisher:
    """Handles collection and publishing of diagnostic data."""
    def __init__(self, node, pid_controllers, target_tracker, strategy_module, velocity_control, resource_monitor, parameter_manager, logger, throttled_logger):
        self.node = node
        self.pid_controllers = pid_controllers
        self.target_tracker = target_tracker
        self.strategy_module = strategy_module
        self.velocity_control = velocity_control
        self.resource_monitor = resource_monitor
        self.parameter_manager = parameter_manager
        self.logger = logger
        self.throttled_logger = ThrottledLogger(logger)
        
        # Setup publishers
        self.pid_diag_pub = node.create_publisher(
            Float32MultiArray,
            TOPICS["output"]["diagnostics"],
            10
        )
        
        self.performance_pub = node.create_publisher(
            String,
            TOPICS["output"]["performance"],
            10
        )
        
        # Setup data containers
        self._diag_msg = Float32MultiArray()
        self._diag_data = np.zeros(14, dtype=np.float32)
        
        # Logging helper
        self.throttled_logger = throttled_logger  # Use the global throttled logger
    
    def publish_diagnostics(self, state_manager):
        """Publish detailed diagnostic information at a slower rate."""
        try:
            if state_manager.shutting_down:
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
            if hasattr(self.parameter_manager, 'debug_level') and self.parameter_manager.debug_level >= 1:
                # Get PID components
                p_x, i_x, d_x = self.pid_controllers['linear_x'].get_components()
                p_y, i_y, d_y = self.pid_controllers['linear_y'].get_components()
                p_a, i_a, d_a = self.pid_controllers['angular'].get_components()
                
                # Get stats on simplified control usage
                simplified_pct = state_manager.perf.simplified_control_count / max(1, state_manager.robot.cycle_count) * 100.0
                
                diag_msg = (
                    f"DIAGNOSTICS: "
                    f"Avg Vel=[{avg_lin_x_vel:.2f}, {avg_lin_y_vel:.2f}, {avg_ang_vel:.2f}], "
                    f"PID X=[{p_x:.2f}, {i_x:.2f}, {d_x:.2f}], "
                    f"PID Y=[{p_y:.2f}, {i_y:.2f}, {d_y:.2f}], "
                    f"PID A=[{p_a:.2f}, {i_a:.2f}, {d_a:.2f}], "
                    f"Strategy={self.strategy_module.current_strategy}, "
                    f"Simp={simplified_pct:.1f}%, "
                    f"Freshness={state_manager.freshness.level}"
                )
                if self.parameter_manager.debug_level >= 1:
                    self.throttled_logger.info(diag_msg, throttle_duration_sec=LOG_THROTTLE_DIAG, log_id='diagnostics')
            
            # Publish PID diagnostic data
            self._publish_pid_diagnostics()
            
            # Publish performance metrics
            self._publish_performance_metrics(state_manager)
        except Exception as e:
            if self.parameter_manager.debug_level >= 1:
                self.logger.error(f"Error in publish_diagnostics: {str(e)}")
    
    def _publish_pid_diagnostics(self):
        """Publish detailed PID diagnostics for analysis."""
        try:
            # Get PID components
            p_x, i_x, d_x = self.pid_controllers['linear_x'].get_components()
            p_y, i_y, d_y = self.pid_controllers['linear_y'].get_components()
            p_a, i_a, d_a = self.pid_controllers['angular'].get_components()
            
            # Get current gains
            kp_x, ki_x, kd_x = self.pid_controllers['linear_x'].get_current_gains()
            kp_y, ki_y, kd_y = self.pid_controllers['linear_y'].get_current_gains()
            kp_a, ki_a, kd_a = self.pid_controllers['angular'].get_current_gains()
            
            # Get current errors
            e_x = self.pid_controllers['linear_x'].error_tracker.current_error
            e_y = self.pid_controllers['linear_y'].error_tracker.current_error
            e_a = self.pid_controllers['angular'].error_tracker.current_error
            
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
            if self.parameter_manager.debug_level >= 1:
                self.logger.error(f"Error in _publish_pid_diagnostics: {str(e)}")
    
    def _publish_performance_metrics(self, state_manager):
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
                    'update_rate': getattr(self.parameter_manager, 'update_rate', 3.0)
                }
            
            # Get current strategy
            strategy_name = "unknown"
            if hasattr(self.strategy_module, 'current_strategy'):
                strategy_name = self.strategy_module.current_strategy
            
            # Add optimization stats
            adaptive_rate = getattr(state_manager.perf, 'adaptive_rate', perf_stats['update_rate'])
            using_simplified = 1 if state_manager.movement.using_simplified_control else 0
            
            # Add freshness and event stats
            freshness_level = state_manager.freshness.level
            event_ratio = 0.0
            total_cycles = max(1, state_manager.perf.event_control_count + state_manager.perf.timer_control_count)
            event_ratio = state_manager.perf.event_control_count / total_cycles * 100.0
            
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
                perf_stats.get("skips", state_manager.perf.skipped_cycle_count),
                perf_stats["update_rate"],
                adaptive_rate,
                using_simplified,
                state_manager.freshness.level,
                event_ratio
            )
            
            # Publish
            self.performance_pub.publish(performance_msg)
        except Exception as e:
            if self.parameter_manager.debug_level >= 1:
                self.logger.error(f"Error publishing performance metrics: {str(e)}")


# Phase 2: Extract Performance Monitor
class EnhancedPerformanceMonitor:
    """Enhanced performance monitoring with adaptive rate control."""
    def __init__(self, resource_monitor, parameter_manager, logger, throttled_logger):
        self.resource_monitor = resource_monitor
        self.parameter_manager = parameter_manager
        self.logger = logger
        self.throttled_logger = ThrottledLogger(logger)
        self.detected_fusion_rate = 1.0
        self.fusion_rate_updated = False
        self.last_rate_adjustment_time = 0.0
        self.current_rate = parameter_manager.update_rate
        self.last_logged_rate = 0.0
        self.adaptive_rate_history = deque(maxlen=10)
        
    def calculate_adaptive_rate(self, state_manager, base_rate, cpu_usage):
        """Calculate adaptive control rate based on CPU usage and fusion rate."""
        try:
            current_time = time.time()
            if not isinstance(base_rate, (int, float)) or base_rate <= 0:
                if self.parameter_manager.debug_level >= 1:
                    self.logger.warning(f"Invalid base_rate: {base_rate}, using default 3.0Hz")
                base_rate = 3.0
            if not isinstance(cpu_usage, (int, float)) or cpu_usage < 0:
                if self.parameter_manager.debug_level >= 1:
                    self.logger.warning(f"Invalid cpu_usage: {cpu_usage}, using 50%")
                cpu_usage = 50.0
                
            # Only adjust rate periodically to avoid oscillation
            if current_time - self.last_rate_adjustment_time < 1.0:
                return self.current_rate or base_rate
                
            self.last_rate_adjustment_time = current_time
            
            # Apply fusion rate consideration if detected
            if self.fusion_rate_updated and self.parameter_manager.enable_fusion_rate_detection:
                fusion_rate = self.detected_fusion_rate
                if fusion_rate > 0 and fusion_rate < 100:
                    adjusted_base_rate = min(max(fusion_rate * 1.2, self.parameter_manager.min_control_rate), self.parameter_manager.max_control_rate)
                    if abs(adjusted_base_rate - base_rate) > 0.3:
                        if self.parameter_manager.debug_level >= 1:
                            self.logger.info(
                                f"Adjusted base rate using fusion detection: {base_rate:.1f}Hz -> {adjusted_base_rate:.1f}Hz "
                                f"(fusion_rate: {fusion_rate:.1f}Hz)"
                            )
                    base_rate = adjusted_base_rate
                    
            # Adjusted thresholds for high baseline CPU
            if cpu_usage > 95.0:
                new_rate = self.parameter_manager.min_control_rate
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
                
            new_rate = max(self.parameter_manager.min_control_rate, min(self.parameter_manager.max_control_rate, new_rate))
            self.current_rate = new_rate
            self.adaptive_rate_history.append((current_time, new_rate))
            
            if abs(new_rate - self.last_logged_rate) > 0.5:
                if self.parameter_manager.debug_level >= 1:
                    self.throttled_logger.info(
                        f"Adaptive rate adjusted: {self.last_logged_rate:.1f}Hz -> {new_rate:.1f}Hz "
                        f"(CPU: {cpu_usage:.1f}%)",
                        throttle_duration_sec=2.0,
                        log_id='adaptive_rate'
                    )
                self.last_logged_rate = new_rate
                
            return new_rate
        except Exception as e:
            if self.parameter_manager.debug_level >= 1:
                self.logger.error(f"Error in calculate_adaptive_rate: {str(e)}")
            return max(base_rate, self.parameter_manager.min_control_rate)
    
    def update_fusion_rate(self, fusion_rate):
        """Update the detected fusion rate."""
        if fusion_rate > 0:
            self.detected_fusion_rate = fusion_rate
            self.fusion_rate_updated = True
            
            # Update the resource monitor with the new fusion rate
            if hasattr(self.resource_monitor, 'set_fusion_rate'):
                self.resource_monitor.set_fusion_rate(fusion_rate)
    
    def should_skip_cycle(self, cpu_usage, last_control_time, adaptive_rate):
        """Determine if the current cycle should be skipped based on CPU usage and timing."""
        # Skip based on CPU threshold
        if self.parameter_manager.enable_cycle_skipping and cpu_usage > self.parameter_manager.max_cpu_skip_threshold:
            return True
            
        # Skip based on timing
        if adaptive_rate > 0:
            current_time = time.time()
            time_since_last = current_time - last_control_time
            if time_since_last < (1.0 / adaptive_rate):
                return True
                
        return False
    
    def update_performance_stats(self, cycle_duration):
        """Update performance statistics in the resource monitor."""
        try:
            if hasattr(self.resource_monitor, '_update_cycle_stats'):
                self.resource_monitor._update_cycle_stats(cycle_duration)
        except Exception as e:
            if self.parameter_manager.debug_level >= 1:
                self.logger.warning(f"Error updating performance stats: {str(e)}")


# Phase 4: Extract Control Strategies
class ControlStrategy(ABC):
    """Base class for control strategies."""
    @abstractmethod
    def compute_velocity_command(self, errors, position_data, current_time, freshness_level="fresh"):
        """Compute velocity commands based on errors and other data."""
        pass
    
    @abstractmethod
    def get_strategy_name(self):
        """Get the name of this strategy."""
        pass


class StandardControlStrategy(ControlStrategy):
    """Full control strategy with PID and coordination."""
    def __init__(self, pid_controllers, coordinated_controller, parameter_manager, distance_error_tracker,
                 lateral_error_tracker, angular_error_tracker, velocity_control):
        self.pid_controllers = pid_controllers
        self.coordinated_controller = coordinated_controller
        self.parameter_manager = parameter_manager
        self.distance_error_tracker = distance_error_tracker
        self.lateral_error_tracker = lateral_error_tracker
        self.angular_error_tracker = angular_error_tracker
        self.velocity_control = velocity_control
    
    def compute_velocity_command(self, errors, position_data, current_time, freshness_level="fresh"):
        """Compute velocity commands using full PID control with coordination."""
        # Unpack errors
        distance_error, lateral_error, angular_error = errors
        
        # Determine strategy flags
        use_forward = True
        use_lateral = True
        use_angular = True
        
        # ADD THIS SECTION: Check for significant angular error and apply angular-first strategy
        significant_angular_error = False
        if self.parameter_manager.angular_first_control:
            # Convert to degrees for easier threshold comparison
            angular_degrees = angular_error * 57.29578  # 180/pi
            if abs(angular_degrees) > 11.0:  # ~0.2 radians
                significant_angular_error = True
        
        # Compute velocities based on the selected strategy
        if self.parameter_manager.coordinated_movement and use_lateral and use_angular:
            # Use coordinated controller for lateral and angular movements
            linear_x_velocity = self.pid_controllers['linear_x'].compute(
                distance_error, 
                current_time, 
                not use_forward,
                self.distance_error_tracker.get_trend()
            )
            
            # Use coordinated control for lateral and angular velocities
            lateral_velocity, angular_velocity = self.coordinated_controller.compute(
                lateral_error,   # lateral error
                angular_error,   # angular error
                current_time,    # current time
                0.0              # current orientation - this is taken directly from the coordinated controller
            )
            
            # ADD THIS SECTION: Apply angular-first strategy if significant angular error
            if significant_angular_error:
                # Reduce linear and lateral velocities until angular error is smaller
                linear_x_velocity *= 0.7
                lateral_velocity *= 0.8
            
            # Disable individual components if strategy requires
            if not use_lateral:
                lateral_velocity = 0.0
            if not use_angular:
                angular_velocity = 0.0
        else:
            # Traditional separate PID controllers
            linear_x_velocity = self.pid_controllers['linear_x'].compute(
                distance_error, 
                current_time, 
                not use_forward,
                self.distance_error_tracker.get_trend()
            )
            
            lateral_velocity = self.pid_controllers['linear_y'].compute(
                lateral_error, 
                current_time, 
                not use_lateral,
                self.lateral_error_tracker.get_trend()
            )
            
            angular_velocity = self.pid_controllers['angular'].compute(
                angular_error, 
                current_time, 
                not use_angular,
                self.angular_error_tracker.get_trend()
            )
            
            # ADD THIS SECTION: Apply angular-first strategy if significant angular error
            if significant_angular_error:
                # Reduce linear and lateral velocities until angular error is smaller
                linear_x_velocity *= 0.7
                lateral_velocity *= 0.8
        
        # Apply freshness-based velocity scaling
        if freshness_level == "stale":
            # Reduced speed (50%) for stale data
            stale_scale = 0.5
            linear_x_velocity *= stale_scale
            lateral_velocity *= stale_scale
            angular_velocity *= stale_scale
        elif freshness_level == "critical" or freshness_level == "invalid":
            # Stop for critical or invalid data
            linear_x_velocity = 0.0
            lateral_velocity = 0.0
            angular_velocity = 0.0
        
        # Apply velocity control limits
        target_distance = position_data['distance'] if position_data and 'distance' in position_data else 0.0
        limited_velocities = self.velocity_control.process_velocities(
            linear_x_velocity, 
            lateral_velocity, 
            angular_velocity, 
            target_distance, 
            self.parameter_manager.desired_distance,
            freshness_level=freshness_level
        )
        
        # ADD THIS LINE: Return the computed velocities
        return limited_velocities
    
    def get_strategy_name(self):
        return "standard"


class SimplifiedControlStrategy(ControlStrategy):
    """Simplified control strategy for reduced CPU usage."""
    def __init__(self, pid_controllers, parameter_manager, velocity_control, last_cmd_vel, level=1):
        self.pid_controllers = pid_controllers
        self.parameter_manager = parameter_manager
        self.velocity_control = velocity_control
        self.last_cmd_vel = last_cmd_vel
        self.level = level  # 0: minimal, 1: basic, 2: medium
        
    def compute_velocity_command(self, errors, position_data, current_time, freshness_level="fresh"):
        """Compute velocity commands using simplified control."""
        if self.level == 0:
            # Minimal computation - just dampen previous velocities
            damping = 0.85
            linear_x = self.last_cmd_vel[0] * damping
            lateral_y = self.last_cmd_vel[1] * damping
            angular_z = self.last_cmd_vel[2] * damping
            return [linear_x, lateral_y, angular_z]
            
        elif self.level == 1:
            # Basic computation - simple proportional control
            if position_data and all(k in position_data for k in ['distance', 'lateral', 'bearing']):
                # Calculate basic errors
                distance_error = position_data['distance'] - self.parameter_manager.desired_distance
                lateral_error = position_data['lateral']
                angular_error = position_data['bearing']
                
                # Simple proportional control with reduced gains
                kp_factor = 0.7  # Reduce gains for smoother control
                linear_x = max(-0.1, min(0.1, distance_error * self.parameter_manager.linear_x_kp * kp_factor))
                lateral_y = max(-0.1, min(0.1, lateral_error * self.parameter_manager.linear_y_kp * kp_factor))
                angular_z = max(-0.3, min(0.3, angular_error * self.parameter_manager.angular_kp * kp_factor))
                
                # Apply damping from previous velocities for smoothness
                damping = 0.3  # 30% of previous velocity
                linear_x = linear_x * (1.0 - damping) + self.last_cmd_vel[0] * damping
                lateral_y = lateral_y * (1.0 - damping) + self.last_cmd_vel[1] * damping
                angular_z = angular_z * (1.0 - damping) + self.last_cmd_vel[2] * damping
                
                return [linear_x, lateral_y, angular_z]
            else:
                # No valid position data, apply strong damping
                damping = 0.7
                linear_x = self.last_cmd_vel[0] * damping
                lateral_y = self.last_cmd_vel[1] * damping
                angular_z = self.last_cmd_vel[2] * damping
                return [linear_x, lateral_y, angular_z]
                
        elif self.level == 2:
            # Medium computation - use PID but skip coordinated control
            distance_error, lateral_error, angular_error = errors
            
            # Use separate PID controllers for efficiency - no coordination
            linear_x_velocity = self.pid_controllers['linear_x'].compute(
                distance_error, 
                current_time, 
                False,
                self.pid_controllers['linear_x'].error_tracker.get_trend()
            )
            
            lateral_velocity = self.pid_controllers['linear_y'].compute(
                lateral_error, 
                current_time, 
                False,
                self.pid_controllers['linear_y'].error_tracker.get_trend()
            )
            
            angular_velocity = self.pid_controllers['angular'].compute(
                angular_error, 
                current_time, 
                False,
                self.pid_controllers['angular'].error_tracker.get_trend()
            )
            
            # Apply velocity limits directly (simplified)
            linear_x_velocity = max(self.parameter_manager.linear_x_min, min(self.parameter_manager.linear_x_max, linear_x_velocity))
            lateral_velocity = max(self.parameter_manager.linear_y_min, min(self.parameter_manager.linear_y_max, lateral_velocity))
            angular_velocity = max(self.parameter_manager.angular_min, min(self.parameter_manager.angular_max, angular_velocity))
            
            return [linear_x_velocity, lateral_velocity, angular_velocity]
            
        # Default fallback
        return list(self.last_cmd_vel)
    
    def get_strategy_name(self):
        return f"simplified_level_{self.level}"


class RecoveryControlStrategy(ControlStrategy):
    """Strategy for recovery behavior."""
    def __init__(self, recovery_module):
        self.recovery_module = recovery_module
        
    def compute_velocity_command(self, errors, position_data, current_time, freshness_level="fresh"):
        """Compute velocity commands for recovery behavior."""
        orientation_data = {'yaw': 0.0}  # This will be set by the controller
        cmd_vel, is_complete = self.recovery_module.handle_recovery(
            current_time, position_data, orientation_data
        )
        return [cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z]
    
    def get_strategy_name(self):
        return "recovery"


class ControlStrategyFactory:
    """Factory for creating appropriate control strategies."""
    def __init__(self, pid_controllers, coordinated_controller, parameter_manager, 
                 distance_error_tracker, lateral_error_tracker, angular_error_tracker,
                 velocity_control, recovery_module):
        self.pid_controllers = pid_controllers
        self.coordinated_controller = coordinated_controller
        self.parameter_manager = parameter_manager
        self.distance_error_tracker = distance_error_tracker
        self.lateral_error_tracker = lateral_error_tracker
        self.angular_error_tracker = angular_error_tracker
        self.velocity_control = velocity_control
        self.recovery_module = recovery_module
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        
    def set_last_cmd_vel(self, cmd_vel):
        """Update the last command velocity."""
        self.last_cmd_vel = cmd_vel
        
    def create_strategy(self, robot_state, computation_level):
        """Create the appropriate control strategy based on state and computation level."""
        if robot_state == "recovery":
            return RecoveryControlStrategy(self.recovery_module)
        
        if computation_level < 3:
            return SimplifiedControlStrategy(
                self.pid_controllers,
                self.parameter_manager,
                self.velocity_control,
                self.last_cmd_vel,
                level=computation_level
            )
            
        # Default to standard control strategy
        return StandardControlStrategy(
            self.pid_controllers,
            self.coordinated_controller,
            self.parameter_manager,
            self.distance_error_tracker,
            self.lateral_error_tracker,
            self.angular_error_tracker,
            self.velocity_control
        )


# Phase 5: Refactored Controller Manager
class PIDControllerNode(Node, StateObserver):
    """Optimized PID Controller node with modular components."""
    def __init__(self):
        """Initialize the enhanced PID controller node with phased initialization."""
        super().__init__('pid_controller')
        self.throttled_logger = ThrottledLogger(self.get_logger())
        try:
            # Phase 1: Parameter initialization (must come first)
            self._initialize_parameters()
            if self.parameter_manager.debug_level >= 2:
                self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
            else:
                self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
            
            # Phase 2: Initialize core components
            self._initialize_core_components()
            
            # Phase 3: Initialize state manager and register as observer
            self._initialize_state_manager()
            
            # Phase 4: Initialize dependent components
            self._initialize_dependent_components()
            
            # Phase 5: Setup communications
            self._setup_publishers()
            self._setup_subscriptions()
            self._setup_timers()
            
            # Validate complete initialization
            self._validate_initialization()
            self.get_logger().info("Initialization complete - all components ready")
        except Exception as e:
            self.get_logger().error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"PID Controller initialization failed: {str(e)}")

    def _initialize_parameters(self):
        """Initialize and validate all parameters."""
        self.parameter_manager = ParameterManager(self)
        # No validation needed here - moved to _validate_initialization
        
    def _initialize_core_components(self):
        """Initialize core components with no dependencies."""
        self.callback_group = ReentrantCallbackGroup()
        self.fast_trig = FastTrigonometry()
        self._init_memory_pools()
        
        # Pre-allocated message objects
        self._cmd_vel_msg = Twist()
        
        # Pre-allocated objects for frequent operations
        self._key_tuple = ["none", "none", "none"]  # Use list instead of tuple for mutability
        
        # Pre-allocated error container
        self._current_errors = [0.0, 0.0, 0.0]  # distance, lateral, angular
        
        # Initialize transform check timer variable
        self.transform_check_timer = None
        
        # Create resource monitor
        self.resource_monitor = ResourceMonitor(throttled_logger, debug_level=self.parameter_manager.debug_level)
        
    def _initialize_state_manager(self):
        """Initialize state manager and register as observer."""
        self.state_manager = EnhancedStateManager()
        self.state_manager.register_observer(self)
        
    def _initialize_dependent_components(self):
        """Initialize components that depend on core components."""
        # Initialize transform system
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.transform_system = TransformSystem(self, throttled_logger, self.tf_buffer)
        self.transform_system.add_transform_dependency("base_link", "imu_link", required=True)
        if not self.transform_system.start_initialization():
            raise RuntimeError("Failed to start transform system initialization")
            
        # Initialize PID controllers
        self._init_controllers()
        
        # Initialize target tracking
        pm = self.parameter_manager
        self.target_tracker = TargetTrackingModule(
            throttled_logger,
            filter_buffer_size=pm.filter_buffer_size,
            prediction_horizon=pm.prediction_horizon,
            debug_level=pm.debug_level
        )
        
        # Initialize strategy module
        self.strategy_module = MovementStrategyModule(throttled_logger, pm.debug_level)
        
        # Initialize velocity control
        self.velocity_control = VelocityControlModule(
            throttled_logger,
            max_velocity=[
                self.parameter_manager.linear_x_max,
                self.parameter_manager.linear_y_max,
                self.parameter_manager.angular_max
            ]
        )
        self.velocity_control.set_approach_parameters(
            pm.approach_distance, 
            pm.min_approach_factor
        )
        
        # Initialize recovery module
        self.recovery_module = RecoveryBehaviorModule(throttled_logger)
        
        # Initialize control strategy factory
        self.strategy_factory = ControlStrategyFactory(
            self.pid_controllers,
            self.coordinated_controller,
            self.parameter_manager,
            self.distance_error_tracker,
            self.lateral_error_tracker,
            self.angular_error_tracker,
            self.velocity_control,
            self.recovery_module
        )
        
        # Initialize performance monitor
        self.performance_monitor = EnhancedPerformanceMonitor(
            self.resource_monitor,
            self.parameter_manager,
            self.get_logger(),
            throttled_logger
        )
        
        if hasattr(self.resource_monitor, 'set_rate_limits'):
            self.resource_monitor.set_rate_limits(
                min_rate=pm.min_control_rate,
                max_rate=pm.max_control_rate,
                base_rate=pm.update_rate
            )
        if hasattr(self.resource_monitor, 'set_cpu_thresholds'):
            self.resource_monitor.set_cpu_thresholds(
                low_threshold=pm.cpu_low_threshold,
                high_threshold=pm.cpu_high_threshold
            )
        self.resource_monitor.start()
        
    def _init_memory_pools(self):
        """Setup memory pools and reusable objects for efficiency."""
        # Initialize object pool manager
        self.object_pool = ObjectPoolManager(max_twist=10, max_vector3=15)
        
    def _init_controllers(self):
        """Initialize the controllers with improved tuning for controlled velocity."""
        pm = self.parameter_manager
        try:
            self.pid_linear_x, self.distance_error_tracker = PIDControllers.create_controller_with_tracker(
                PIDControllers.ControllerType.LINEAR_X,
                pm.linear_x_kp, pm.linear_x_ki, pm.linear_x_kd,
                pm.linear_x_min, pm.linear_x_max,
                "distance", throttled_logger, max_history=8
            )
            self.pid_linear_y, self.lateral_error_tracker = PIDControllers.create_controller_with_tracker(
                PIDControllers.ControllerType.LINEAR_Y,
                pm.linear_y_kp, pm.linear_y_ki, pm.linear_y_kd,
                pm.linear_y_min, pm.linear_y_max,
                "lateral", throttled_logger, max_history=8
            )
            self.pid_angular, self.angular_error_tracker = PIDControllers.create_controller_with_tracker(
                PIDControllers.ControllerType.ANGULAR,
                pm.angular_kp, pm.angular_ki, pm.angular_kd,
                pm.angular_min, pm.angular_max,
                "angular", throttled_logger, max_history=8
            )
            self.pid_linear_x.validate_initialization()
            self.pid_linear_y.validate_initialization()
            self.pid_angular.validate_initialization()
            
            # Store controllers in a dictionary for easy access
            self.pid_controllers = {
                'linear_x': self.pid_linear_x,
                'linear_y': self.pid_linear_y,
                'angular': self.pid_angular
            }
            
            self.coordinated_controller = PIDControllers.CoordinatedController(
                self.pid_linear_y, 
                self.pid_angular,
                throttled_logger,
                {
                    'coupling_factor': pm.coordinated_coupling_factor,
                    'smoothing_factor': pm.coordinated_smoothing_factor,
                    'min_angle_for_reduction': pm.coordinated_min_angle_for_reduction,
                    'zero_angle_threshold': pm.coordinated_zero_angle_threshold,
                    'max_angle_factor': pm.coordinated_max_angle_factor,
                    'same_sign_scale': pm.coordinated_same_sign_scale,
                    'opposite_sign_scale': pm.coordinated_opposite_sign_scale,
                }
            )
            
            self.get_logger().info("PID controllers initialized successfully with error trackers")
        except Exception as e:
            handle_initialization_error(
                self.get_logger(),
                "Failed to initialize PID controllers",
                e
            )
            
    def _setup_publishers(self):
        """Set up all publishers for this node."""
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            TOPICS["output"]["cmd_vel"],
            10
        )
        
        # Initialize diagnostics publisher
        self.diagnostics_publisher = DiagnosticsPublisher(
            self,
            self.pid_controllers,
            self.target_tracker,
            self.strategy_module,
            self.velocity_control,
            self.resource_monitor,
            self.parameter_manager,
            self.get_logger(),
            throttled_logger
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
        
    def _setup_timers(self):
        """Set up timer callbacks for periodic tasks with tiered frequencies."""
        # Main control loop timer
        self.timer = self.create_timer(1.0 / self.parameter_manager.update_rate, self.control_loop_callback)
        
        # Diagnostic timer
        self.diagnostic_timer = self.create_timer(
            1.0 / self.parameter_manager.diagnostics_rate, 
            lambda: self.diagnostics_publisher.publish_diagnostics(self.state_manager)
        )
        
        # Resource monitoring timer
        self.resource_timer = self.create_timer(
            self.parameter_manager.cpu_throttle_interval, 
            self._update_resource_monitoring
        )
    
    def _validate_initialization(self):
        """Validate that all components are properly initialized."""
        pm = self.parameter_manager
        
        # Validate parameters
        required_params = [
            'update_rate', 'min_control_rate', 'max_control_rate',
            'approach_distance', 'min_approach_factor',
            'distance_threshold', 'lateral_threshold', 'angular_threshold'
        ]
        for param in required_params:
            if not hasattr(pm, param) or getattr(pm, param) is None:
                raise ValueError(f"Required parameter '{param}' is not initialized")
                
        # Validate components
        required_components = [
            'resource_monitor', 'transform_system', 'target_tracker',
            'strategy_module', 'velocity_control', 'recovery_module',
            'diagnostics_publisher', 'performance_monitor', 'strategy_factory'
        ]
        for component in required_components:
            if not hasattr(self, component) or getattr(self, component) is None:
                raise RuntimeError(f"Required component '{component}' is not initialized")
    
    # StateObserver Implementation
    def on_state_change(self, old_state, new_state, reason=""):
        """Called when robot state changes."""
        if self.parameter_manager.debug_level >= 1:
            self.throttled_logger.info(
                f"STATE TRANSITION: {old_state} → {new_state} {reason}",
                throttle_duration_sec=LOG_THROTTLE_STATE,
                log_id='state_transition'
            )
        
        # Handle recovery state transitions
        if new_state == "recovery":
            stop_cmd = self.recovery_module.start_recovery()
            self.cmd_vel_pub.publish(stop_cmd)
        elif old_state == "recovery" and new_state != "recovery":
            self.recovery_module.reset()
            if self.parameter_manager.debug_level >= 1:
                self.throttled_logger.info("Exiting recovery mode", throttle_duration_sec=LOG_THROTTLE_STATE, log_id='recovery_exit')
            
        # Complete controller reset when transitioning between tracking and other states
        if new_state == "tracking" or old_state == "tracking":
            self._complete_controller_reset()
            if new_state == "tracking":
                self.state_manager.robot.force_target_reacquisition = True
                
        if new_state != "tracking" and new_state != "searching" and new_state != "lost_ball":
            stop_cmd = self.recovery_module.stop_robot()
            self.cmd_vel_pub.publish(stop_cmd)
    
    def on_freshness_change(self, freshness_level, data_age):
        """Called when data freshness level changes."""
        if freshness_level != "unknown" and self.parameter_manager.debug_level >= 1:
            self.throttled_logger.info(
                f"Data freshness changed: {freshness_level} "
                f"(age: {data_age:.3f}s)",
                throttle_duration_sec=1.0,
                log_id='freshness_change'
            )
            
        # Handle critical freshness
        if freshness_level == "critical" and not self.state_manager.movement.robot_stopped:
            self.get_logger().warning(f"CRITICAL DATA AGE: {data_age:.3f}s - Safety stop triggered")
            stop_cmd = self.recovery_module.stop_robot()
            self.cmd_vel_pub.publish(stop_cmd)
            
    def state_callback(self, msg):
        """Handle robot state updates with improved recovery behavior."""
        new_state = msg.data
        # If state changed, handle the transition using state manager
        if new_state != self.state_manager.robot.state:
            current_time = time.time()
            if self.parameter_manager.debug_level >= 2:
                time_in_state = current_time - self.state_manager.robot.last_state_change_time
                self.get_logger().info(f"Time in state '{self.state_manager.robot.state}': {time_in_state:.2f}s")
            
            # Use state manager to handle transition
            self.state_manager.transition_state(new_state)
    
    def orientation_callback(self, msg):
        """Handle orientation updates from the IMU with improved transform handling."""
        # Extract yaw (z component) from the Vector3Stamped message
        raw_orientation = msg.vector.z
        
        # Store timestamp for freshness checking
        self.state_manager.robot.last_orientation_time = time.time()
        
        # Check if transform system is ready before attempting transforms
        if (not hasattr(self, 'transform_system') or 
            not self.transform_system.is_transform_system_ready()):
            # If transforms aren't ready, use raw orientation
            self.state_manager.robot.robot_orientation = raw_orientation
            return
        
        # If we need to transform the orientation to another frame
        try:
            # First approach: direct transform using quaternion math
            transform = self.transform_system.get_transform_between_frames(
                self.transform_system.imu_frame, 
                self.transform_system.reference_frame
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
                if self.parameter_manager.use_fast_trigonometry:
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
                if self.parameter_manager.use_fast_trigonometry:
                    self.state_manager.robot.robot_orientation = self.fast_trig.atan2(ty, tx)
                else:
                    self.state_manager.robot.robot_orientation = math.atan2(ty, tx)
            else:
                # If transform not available, use raw orientation
                self.state_manager.robot.robot_orientation = raw_orientation
                
        except Exception as e:
            # In case of error, fall back to raw orientation
            if self.parameter_manager.debug_level >= 2:
                self.get_logger().warning(f"Orientation transform error: {str(e)}")
            self.state_manager.robot.robot_orientation = raw_orientation
        
        # Log orientation updates at high debug level
        if self.parameter_manager.debug_level >= 3:
            # Use pre-computed table for degrees conversion if fast trig is enabled
            if self.parameter_manager.use_fast_trigonometry:
                orientation_degrees = self.state_manager.robot.robot_orientation * 57.29578  # 180/pi
            else:
                orientation_degrees = math.degrees(self.state_manager.robot.robot_orientation)
                            
            self.throttled_logger.info(f"Orientation update: yaw={orientation_degrees:.2f}°", 
                throttle_duration_sec=LOG_THROTTLE_CONTROL, log_id='Orientation_update')
    
    def _is_orientation_fresh(self):
        """Check if orientation data is fresh enough to use."""
        if self.state_manager.robot.last_orientation_time is None:
            return False
            
        current_time = time.time()
        age = current_time - self.state_manager.robot.last_orientation_time
        
        # Consider orientation data older than 0.5 seconds as stale
        return age < 0.5
    
    def target_callback(self, msg):
        """
        Process incoming target message with event-based control capability.
        
        This enhanced callback not only updates target data but also
        can trigger immediate control cycle for improved responsiveness.
        """
        # Early exit if shutting down
        if self.state_manager.shutting_down:
            return
        
        # Record time for event tracking
        event_time = time.time()
        
        # Update target data in the target tracking module
        update_success = self.target_tracker.update_target(msg, self.parameter_manager.debug_level)
        
        if not update_success:
            return
        
        # Check for fusion rate updates
        if self.parameter_manager.enable_fusion_rate_detection:
            try:
                # Make sure the method exists and handle possible exceptions
                if hasattr(self.target_tracker, 'get_fusion_rate'):
                    fusion_rate, was_updated = self.target_tracker.get_fusion_rate()
                    
                    if was_updated and fusion_rate > 0:  # Ensure rate is positive
                        # Update performance monitor with fusion rate
                        self.performance_monitor.update_fusion_rate(fusion_rate)
                        
                        self.get_logger().info(f"Detected fusion rate: {fusion_rate:.2f}Hz")
            except Exception as e:
                self.get_logger().warning(f"Error getting fusion rate: {str(e)}")
        
        # Event-based control execution
        # Only trigger if we're in tracking mode and it's been long enough since last execution
        try:
            if (self.state_manager.robot.state == "tracking" and 
                (event_time - self.state_manager.perf.last_event_execution) > (1.0 / self.parameter_manager.max_control_rate) and
                (event_time - self.state_manager.perf.last_timer_execution) > 0.05):  # Minimum 50ms between executions
                
                # Execute control loop directly in response to new data
                self.execute_control_cycle(event_triggered=True)
                self.state_manager.perf.last_event_execution = event_time
                self.state_manager.perf.event_control_count += 1
        except Exception as e:
            self.get_logger().error(f"Error in event-based control: {str(e)}")
    
    def _update_resource_monitoring(self):
        """Update resource monitoring stats."""
        try:
            # Update CPU stats if the method exists
            if hasattr(self.resource_monitor, 'update_cpu_stats'):
                self.resource_monitor.update_cpu_stats()
                
                # Check for high CPU - trigger cycle skipping if needed
                if (self.parameter_manager.enable_cycle_skipping and 
                    hasattr(self.resource_monitor, 'current_cpu_usage') and
                    self.resource_monitor.current_cpu_usage > self.parameter_manager.max_cpu_skip_threshold):
                    
                    self.state_manager.skip_next_cycle = True
                    
                    current_time = time.time()
                    if current_time - self.state_manager.perf.last_cpu_warning_time >= 2.0:
                        self.get_logger().warning(
                            f"HIGH CPU LOAD: {self.resource_monitor.current_cpu_usage:.1f}% - skipping next cycle"
                        )
                        self.state_manager.perf.last_cpu_warning_time = current_time
                else:
                    self.state_manager.skip_next_cycle = False
                
            # Log pool statistics periodically if in debug mode
            if self.parameter_manager.debug_level >= 2 and hasattr(self, 'object_pool'):
                pool_stats = self.object_pool.get_stats()
                current_time = time.time()
                if current_time - self.state_manager.perf.last_pool_log_time >= 10.0:
                    pool_msg = (
                        f"Object pool stats: "
                        f"Twist={pool_stats['Twist']['pool_size']}/{pool_stats['Twist']['max_usage']} "
                        f"(misses={pool_stats['Twist']['misses']}), "
                        f"Vector3={pool_stats['Vector3']['pool_size']}/{pool_stats['Vector3']['max_usage']} "
                        f"(misses={pool_stats['Vector3']['misses']}), "
                        f"Float32MultiArray={pool_stats['Float32MultiArray']['pool_size']}/{pool_stats['Float32MultiArray']['max_usage']} "
                        f"(misses={pool_stats['Float32MultiArray']['misses']})"
                    )
                    self.throttled_logger.info(pool_msg, throttle_duration_sec=10.0, log_id='pool_stats')
                    self.state_manager.perf.last_pool_log_time = current_time
        except Exception as e:
            if self.parameter_manager.debug_level >= 1:
                self.get_logger().error(f"Error in resource monitoring: {str(e)}")
    
    def _log_periodic_status(self):
        """Log periodic status updates."""
        if self.parameter_manager.debug_level < 1:
            return
        perf_stats = self.resource_monitor.get_performance_stats()
        if not perf_stats or not all(k in perf_stats for k in ['cpu_avg', 'cycle_time_ms', 'update_rate']):
            perf_stats = {
                'cpu_avg': 0.0,
                'cycle_time_ms': 0.0,
                'update_rate': getattr(self.parameter_manager, 'update_rate', 3.0),
                'skips': 0
            }
        strategy_name = getattr(self.strategy_module, 'current_strategy', 'unknown')
        total_cycles = max(1, self.state_manager.perf.event_control_count + self.state_manager.perf.timer_control_count)
        event_ratio = self.state_manager.perf.event_control_count / total_cycles * 100.0
        status_msg = (
            f"Status: Robot state={self.state_manager.robot.state}, "
            f"Strategy={strategy_name}, "
            f"CPU={perf_stats['cpu_avg']:.1f}%, "
            f"Cycle time={perf_stats['cycle_time_ms']:.2f}ms, "
            f"Rate={perf_stats['update_rate']:.1f}Hz, "
            f"Simplified={self.state_manager.movement.using_simplified_control}, "
            f"Freshness={self.state_manager.freshness.level}, "
            f"Event-driven={event_ratio:.1f}%, "
            f"Skips={self.state_manager.perf.skipped_cycle_count}"
        )
        if self.parameter_manager.debug_level >= 1:
            self.throttled_logger.info(status_msg, throttle_duration_sec=LOG_THROTTLE_CONTROL, log_id='periodic_status')
    
    def control_loop_callback(self):
        """Regular control loop to calculate and publish velocity commands with CPU optimization."""
        try:
            # Skip if shutting down
            if self.state_manager.shutting_down:
                return
            
            # Skip if transform system is not initialized and required
            if (hasattr(self, 'transform_system') and 
                not self.transform_system.is_transform_system_ready()):
                
                # Log at most once per 20 cycles to avoid spamming
                if self.state_manager.robot.cycle_count % 20 == 0:
                    status = self.transform_system.get_status()
                    self.get_logger().warn(
                        f"Control loop waiting for transform initialization: {status['message']}"
                    )
                    
                    # Try to restart initialization if it's in error state
                    if status['status'] == TransformStatus.ERROR:
                        self.get_logger().warn("Attempting to restart transform initialization")
                        self.transform_system.start_initialization()
                
                # Still increment cycle count
                self.state_manager.robot.cycle_count += 1
                return
                    
            # Apply adaptive rate control if enabled
            if (self.parameter_manager.adaptive_control_rate and 
                hasattr(self.resource_monitor, 'current_cpu_usage') and 
                hasattr(self.parameter_manager, 'update_rate')):
                
                # Calculate adaptive rate
                adaptive_rate = self.performance_monitor.calculate_adaptive_rate(
                    self.state_manager,
                    self.parameter_manager.update_rate, 
                    self.resource_monitor.current_cpu_usage
                )
                
                # Check if we should skip this cycle
                if self.performance_monitor.should_skip_cycle(
                    self.resource_monitor.current_cpu_usage,
                    self.state_manager.last_control_time,
                    adaptive_rate
                ):
                    # Increment skipped cycle count
                    self.state_manager.perf.skipped_cycle_count += 1
                    return
            
            # Execute the control cycle
            self.execute_control_cycle(event_triggered=False)
                    
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.get_logger().error(f"Unexpected error in control_loop_callback: {str(e)}\nStack trace:\n{stack_trace}")
            
            # Try to safely stop the robot
            try:
                self.recovery_module.stop_robot()
            except Exception as stop_error:
                self.get_logger().error(f"Failed to stop robot after error: {str(stop_error)}")
    
    def execute_control_cycle(self, event_triggered=False):
        """Execute one complete control cycle with improved error handling and state management."""
        try:
            # Skip if shutting down
            if self.state_manager.shutting_down:
                return
            
            # Skip if transform system is not initialized and required
            if (hasattr(self, 'transform_system') and 
                not self.transform_system.is_transform_system_ready()):
                return
            
            # Track execution time source for metrics
            if event_triggered:
                self.state_manager.perf.last_event_execution = time.time()
            else:
                self.state_manager.perf.last_timer_execution = time.time()
                self.state_manager.perf.timer_control_count += 1
            
            # Mark cycle start for performance tracking
            cycle_start_time = time.time()
            
            # Calculate dt since last control
            current_time = time.time()
            dt = current_time - self.state_manager.last_control_time
            # Ensure dt is reasonable (protect against time jumps)
            if dt > 1.0:
                self.get_logger().warning(f"Large time step detected: {dt:.3f}s - capping at 0.1s")
                dt = 0.1  # Cap at 100ms to prevent instability
            elif dt <= 0.0:
                dt = 0.001  # Ensure positive dt to prevent division by zero
            
            self.state_manager.last_control_time = current_time
            
            # Increment cycle counter
            self.state_manager.robot.cycle_count += 1
            
            # Check data freshness - critical for safety
            is_fresh, freshness_level, data_age = self._check_data_freshness()
            
            # Update state manager with freshness info
            self.state_manager.update_freshness(freshness_level, data_age)
            
            # Log periodic status updates (once every 50 cycles)
            if self.state_manager.robot.cycle_count % 50 == 0:
                self._log_periodic_status()
            
            # Handle critical data freshness (prioritized over other state handling)
            if freshness_level == "critical":
                if not self.state_manager.movement.robot_stopped:
                    self.get_logger().warning(f"CRITICAL DATA AGE: {data_age:.3f}s - Safety stop triggered")
                    stop_cmd = self.recovery_module.stop_robot()
                    self.cmd_vel_pub.publish(stop_cmd)  # Ensure command is published
                    self.state_manager.movement.robot_stopped = True
                    self.state_manager.movement.stop_time = time.time()
                return  # Exit early when data is critically stale
                
            # Special handling for recovery mode (higher priority than other states)
            if self.state_manager.robot.state == "recovery":
                position_data = self.target_tracker.get_position_data()
                orientation_data = {'yaw': self.state_manager.robot.robot_orientation}
                cmd_vel, is_complete = self.recovery_module.handle_recovery(
                    time.time(), position_data, orientation_data
                )
                self.cmd_vel_pub.publish(cmd_vel)
                if is_complete:
                    self.recovery_module.reset()
                    self.get_logger().info("Recovery sequence completed")
                return
            
            # Handle non-tracking states (searching/lost_ball have their own handling)
            if self.state_manager.robot.state != "tracking":
                if self._handle_non_tracking_state():
                    return  # Exit after handling non-tracking state
            
            # Check if orientation data is fresh (prevents race conditions)
            if not self._is_orientation_fresh():
                if self.parameter_manager.debug_level >= 2:
                    self.get_logger().warning("Skipping control cycle - orientation data is stale")
                return
            
            # Determine computation level needed
            # The key CPU optimization - avoid expensive calculations when appropriate
            if self.parameter_manager.use_simplified_control_when_possible:
                computation_level = self._determine_computation_level()
                self.state_manager.movement.computation_level = computation_level
                
                # Create appropriate control strategy based on computation level
                current_strategy = self.strategy_factory.create_strategy(
                    self.state_manager.robot.state,
                    computation_level
                )
                
                if computation_level < 3:
                    self.state_manager.movement.using_simplified_control = True
                    self.state_manager.perf.simplified_control_count += 1
                else:
                    self.state_manager.movement.using_simplified_control = False
                    # Record that we did full computation
                    self.state_manager.movement.last_full_computation_time = time.time()
            else:
                # Standard control path
                self.state_manager.movement.using_simplified_control = False
                current_strategy = self.strategy_factory.create_strategy(
                    self.state_manager.robot.state,
                    3  # Full computation
                )
            
            # Calculate current errors
            distance, lateral, bearing, angular_degrees = self._calculate_errors()
            
            if self.parameter_manager.debug_level >= 2:
                debug_msg = (
                    f"PRE-STOP CHECK: distance={distance:.3f}m (target={self.parameter_manager.desired_distance:.3f}m), "
                    f"lateral={lateral:.3f}m, angular={angular_degrees:.2f}°, "
                    f"is_stopped={self.state_manager.movement.robot_stopped}"
                )
                self.get_logger().info(debug_msg)
            
            # Check stop conditions and handle if needed
            if self._handle_stop_conditions(distance, lateral, angular_degrees, dt):
                return  # Exit if stop conditions were met
            
            # Get position data for control strategy
            position_data = self.target_tracker.get_position_data()
            
            # Compute velocities using selected strategy
            velocities = current_strategy.compute_velocity_command(
                self._current_errors,
                position_data, 
                current_time,
                freshness_level
            )
            
            # Update strategy factory with last velocity
            self.strategy_factory.set_last_cmd_vel(velocities)
            
            # Guard against NaN values which can cause silent failures
            if any(math.isnan(v) for v in velocities):
                self.get_logger().error("NaN velocity detected - resetting to zero")
                velocities = [0.0, 0.0, 0.0]
            
            # Publish command
            cmd_vel_msg = self._cmd_vel_msg
            cmd_vel_msg.linear.x = float(velocities[0])
            cmd_vel_msg.linear.y = float(velocities[1])
            cmd_vel_msg.angular.z = float(velocities[2])
            self.cmd_vel_pub.publish(cmd_vel_msg)
            
            # Update last command velocity in state manager
            self.state_manager.movement.last_cmd_vel = (velocities[0], velocities[1], velocities[2])
            
            # Update performance stats
            cycle_duration = time.time() - cycle_start_time
            self.performance_monitor.update_performance_stats(cycle_duration)
                
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.get_logger().error(f"Unexpected error in execute_control_cycle: {str(e)}\nStack trace:\n{stack_trace}")
            
            # Try to safely stop the robot
            try:
                stop_cmd = self.recovery_module.stop_robot()
                self.cmd_vel_pub.publish(stop_cmd)  # Ensure the stop command is published
            except Exception as stop_error:
                self.get_logger().error(f"Failed to stop robot after error: {str(stop_error)}")
    
    def _check_data_freshness(self):
        """
        Check the freshness of target data and update system state accordingly.
        
        Returns:
            tuple: (is_fresh, freshness_level, age)
        """
        try:
            # Get freshness information from the target tracker
            is_fresh, freshness_level, data_age = self.target_tracker.is_target_fresh(
                max_age=self.parameter_manager.fresh_data_timeout
            )
            
            # Validate returned data (guard against unexpected return values)
            if not isinstance(freshness_level, str):
                self.get_logger().warning(f"Invalid freshness level type: {type(freshness_level)}")
                freshness_level = "critical"  # Default to critical for safety
            if not isinstance(data_age, (int, float)):
                self.get_logger().warning(f"Invalid data age type: {type(data_age)}")
                data_age = 999.0  # Default to high age for safety
            
            return is_fresh, freshness_level, data_age            
        except Exception as e:
            self.get_logger().error(f"Error checking data freshness: {str(e)}")
            # Default to critical for safety in case of errors
            return False, "critical", 999.0
    
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
        self._current_errors[0] = distance - self.parameter_manager.desired_distance  # distance_error
        self._current_errors[1] = lateral - 0.0                    # lateral_error 
        self._current_errors[2] = bearing                          # angular_error
        
        # Convert angular error to degrees for logging
        # Use optimized conversion if fast trigonometry is enabled
        if self.parameter_manager.use_fast_trigonometry:
            angular_degrees = bearing * 57.29578  # 180/pi - faster than math.degrees
        else:
            angular_degrees = math.degrees(bearing)
        
        return distance, lateral, bearing, angular_degrees
    
    def _handle_non_tracking_state(self):
        """Handle robot behavior when not in tracking mode."""
        # When not tracking, ensure robot is stopped (unless controlled by another node)
        if self.state_manager.robot.state not in ["searching", "lost_ball"]:
            stop_cmd = self.recovery_module.stop_robot()
            self.cmd_vel_pub.publish(stop_cmd)
        return True  # Indicate that the method handled the situation
    
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
                distance, lateral, angular_degrees, self.state_manager.movement.robot_stopped
            )
            
            if should_stop:
                if not self.state_manager.movement.robot_stopped:
                    self.get_logger().info(stop_reason)
                    self.state_manager.movement.robot_stopped = True
                    self.state_manager.movement.stop_time = time.time()
                    # Generate stop command
                    stop_cmd = self.recovery_module.stop_robot()
                    # Publish stop command to actually stop the robot
                    self.cmd_vel_pub.publish(stop_cmd)
                return True  # Handled by stopping the robot
        
        # Update error trackers
        self.distance_error_tracker.update(self._current_errors[0], dt)
        self.lateral_error_tracker.update(self._current_errors[1], dt)
        self.angular_error_tracker.update(self._current_errors[2], dt)
        
        return False  # Not handled, continue with normal control
    
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
        distance_error = abs(distance - self.parameter_manager.desired_distance)
        lateral_error = abs(lateral)
        angular_error = abs(angular_degrees)
        
        # Start with base thresholds
        distance_threshold = self.parameter_manager.distance_threshold
        lateral_threshold = self.parameter_manager.lateral_threshold
        angular_threshold = self.parameter_manager.angular_threshold
        
        # Increase angular threshold when at target distance
        if distance_error < self.parameter_manager.distance_threshold * 1.5:
            angular_threshold *= self.parameter_manager.angular_at_target_factor
            if self.parameter_manager.debug_level >= 2:
                threshold_msg = f"At target distance: increased angular threshold to {angular_threshold:.2f}°"
                self.throttled_logger.info(threshold_msg, throttle_duration_sec=LOG_THROTTLE_DIAG, log_id='angular_thresholds')
        
        # Apply state-dependent hysteresis
        if is_stopped:
            # Higher thresholds to start moving (requires larger errors)
            hysteresis = 1.5 + self.state_manager.movement.movement_hysteresis
            distance_threshold = min(distance_threshold * hysteresis, 0.2)
            lateral_threshold = min(lateral_threshold * hysteresis, 0.15)
            angular_threshold = min(angular_threshold * hysteresis, 15.0)
        else:
            # Lower thresholds to stop (more precision when near target)
            hysteresis = 0.8
            distance_threshold *= hysteresis
            lateral_threshold *= hysteresis
            angular_threshold *= hysteresis
        
        # Log thresholds for debugging
        if self.parameter_manager.debug_level >= 2:
            debug_msg = (
                f"Stop thresholds: d={distance_error:.3f}/{distance_threshold:.3f}, "
                f"l={lateral_error:.3f}/{lateral_threshold:.3f}, "
                f"a={angular_error:.2f}/{angular_threshold:.2f}"
            )
            self.throttled_logger.info(debug_msg, throttle_duration_sec=LOG_THROTTLE_DIAG, log_id='stop_thresholds')
            
        
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
            self.state_manager.movement.movement_hysteresis += 0.05
            self.state_manager.movement.movement_hysteresis = min(0.3, self.state_manager.movement.movement_hysteresis)  # Cap at 0.3
        
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
        if not self.state_manager.movement.robot_stopped:
            return False
        
        # Handle initialization for first movement after startup
        if self.state_manager.movement.initial_movement_boost:
            hysteresis = 0.5  # Much lower hysteresis for first movement
            self.state_manager.movement.initial_movement_boost = False
        else:
            # Regular hysteresis calculation
            stop_duration = time.time() - self.state_manager.movement.stop_time
            hysteresis = min(1.1, 1.0 + stop_duration * 0.1)
        
        # Calculate movement thresholds with hysteresis
        distance_threshold = self.parameter_manager.distance_threshold * hysteresis * 0.7
        lateral_threshold = self.parameter_manager.lateral_threshold * hysteresis * 0.7
        
        # Adjust angular threshold based on distance to target
        if abs(distance_error) < self.parameter_manager.distance_threshold * 1.2:
            # More lenient when at target distance
            angular_threshold = self.parameter_manager.angular_threshold * self.parameter_manager.angular_at_target_factor * hysteresis
        else:
            # Standard threshold otherwise
            angular_threshold = self.parameter_manager.angular_threshold * hysteresis
        
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
            if self.parameter_manager.debug_level >= 1:
                self.throttled_logger.info(log_msg, throttle_duration_sec=2.0, log_id='exit_stopped')
            
            # Reset stopped state
            self.state_manager.movement.robot_stopped = False
            
            # Reset movement hysteresis
            self.state_manager.movement.movement_hysteresis = 0.0
            
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
        self.state_manager.movement.movement_hysteresis = 0.0
        # Set stopped state
        self.state_manager.movement.robot_stopped = True
        self.state_manager.movement.stop_time = time.time()
        # Reset computation tracking
        self.state_manager.movement.using_simplified_control = False
        self.state_manager.movement.last_full_computation_time = time.time()
        # Reset data freshness tracking
        self.state_manager.freshness.level = "unknown"
        self.get_logger().info("Complete controller reset performed")
    
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
        time_since_full = time.time() - self.state_manager.movement.last_full_computation_time
        if time_since_full > 0.5:
            return 3
        position_data = self.target_tracker.get_position_data()
        if not position_data or not all(k in position_data for k in ['distance', 'lateral', 'bearing']):
            return computation_level
        distance_error = abs(position_data['distance'] - self.parameter_manager.desired_distance)
        lateral_error = abs(position_data['lateral'])
        angular_error = abs(position_data['bearing'])
        if self.parameter_manager.use_fast_trigonometry:
            angular_degrees = angular_error * 57.29578
        else:
            angular_degrees = math.degrees(angular_error)
        # Use velocity_control for velocity state
        velocity_stable = all(abs(v) < 0.05 for v in self.velocity_control.last_cmd_vel)
        if (distance_error < self.parameter_manager.distance_threshold * 0.3 and
            lateral_error < self.parameter_manager.lateral_threshold * 0.3 and
            angular_degrees < self.parameter_manager.angular_threshold * 0.3):
            computation_level = min(computation_level, 0)
        elif (distance_error < self.parameter_manager.distance_threshold * 0.7 and
              lateral_error < self.parameter_manager.lateral_threshold * 0.7 and
              angular_degrees < self.parameter_manager.angular_threshold * 0.7):
            computation_level = min(computation_level, 1)
        elif (distance_error < self.parameter_manager.distance_threshold * 1.5 and
              lateral_error < self.parameter_manager.lateral_threshold * 1.5 and
              angular_degrees < self.parameter_manager.angular_threshold * 1.5):
            computation_level = min(computation_level, 2)
        if velocity_stable and computation_level > 0:
            computation_level -= 1
        if (self.distance_error_tracker.get_trend() == "stable" and
            self.lateral_error_tracker.get_trend() == "stable" and
            self.angular_error_tracker.get_trend() == "stable"):
            computation_level = min(computation_level, 1)
        return computation_level
    
    def prepare_shutdown(self):
        """Prepare for node shutdown."""
        if self.state_manager.shutting_down:
            return  # Already shutting down
            
        print("Preparing for shutdown")  # Use print instead of ROS logger
        self.state_manager.shutting_down = True
        
        # Cancel all timers to prevent them from continuing to fire
        for timer in [self.timer, self.diagnostic_timer, self.resource_timer]:
            if hasattr(self, timer.__name__) and timer.__name__ is not None:
                try:
                    timer.cancel()
                except Exception:
                    pass
                    
        if hasattr(self, 'transform_check_timer') and self.transform_check_timer is not None:
            try:
                self.transform_check_timer.cancel()
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
    
    # Set up signal handler for graceful shutdown
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    
    try:
        print("=================================================")
        print("Optimized PID Controller for Basketball Tracking Robot")
        print("=================================================")
        try:
            node = PIDControllerNode()
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
                node.state_manager.shutting_down = True
                
            # Important: Raise KeyboardInterrupt to break out of rclpy.spin()
            raise KeyboardInterrupt
            
        # Register the signal handler
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