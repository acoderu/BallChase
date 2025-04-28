"""
Basketball Tracking Robot - Improved PID Controller Node
=======================================================

This controller implements efficient movement patterns for a mecanum-wheeled
basketball tracking robot with several enhancements:
- Angular-first control strategy for diagonal movements
- Fast strategy transitions for responsive tracking
- Enhanced integral term management to prevent windup
- Coordinated angular-lateral control with balanced parameters
- Balanced error thresholds and hysteresis for smooth behavior
- Continuous motion tracking with trajectory prediction
- Optimized for Raspberry Pi 5 performance with resource monitoring
"""

import numba
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped, Vector3, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray
from tf2_ros import TransformListener, Buffer
import math
import time
import numpy as np
import signal
import sys
from collections import deque
import psutil  # For CPU monitoring
import logging


# Global logging manager instance
_logging_manager = None
_dummy_logging_manager = None  # Will be initialized below

class LoggingManager:
    """Centralized manager for all logging in the PID controller."""
    
    def __init__(self, node, debug_level=1, log_verbosity=1):
        """
        Initialize logging system with proper hierarchy.
        
        Args:
            node: ROS node for ROS logging
            debug_level: Debug level (0=minimal, 1=normal, 2=verbose)
        """
        self.node = node
        self.debug_level = debug_level
        self.log_verbosity = log_verbosity
        self.throttle_timestamps = {}  # For tracking throttled logs
        
        # Create base logger
        self.root_logger = logging.getLogger('pid_controller')
        
        # Configure format
        formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] %(message)s')
        
        # Set up handlers
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Add file handler for persistent logs
        file_handler = logging.FileHandler('pid_controller_detailed.log')
        file_handler.setFormatter(formatter)
        
        # Add handlers (avoid duplicate handlers)
        if not self.root_logger.handlers:
            self.root_logger.addHandler(console_handler)
            self.root_logger.addHandler(file_handler)
        
        # Set level based on debug parameter
        if debug_level >= 2:
            self.root_logger.setLevel(logging.DEBUG)
        elif debug_level == 1:
            self.root_logger.setLevel(logging.INFO)
        else:
            self.root_logger.setLevel(logging.WARNING)
        
        # Pre-configure component loggers
        self.loggers = {}
        self._setup_component_loggers()
        
        # Log initialization
        self.log_structured('logging_manager', 'INIT', 
                         f"Logging system initialized with debug level {debug_level}")
    
    def should_log(self, verbosity_level):
        """Check if logging is enabled for the given verbosity level."""
        return self.log_verbosity >= verbosity_level

    def _setup_component_loggers(self):
        """Set up loggers for all components with correct propagation settings."""
        # List of all components
        component_names = [
            'target_filter',
            'target_tracking',
            'matrix4x4',
            'motion_control',
            'motion_commander',
            'strategy_selector',
            'pid',
            'pid_controller',
            'velocity_limiter',
            'coordinated_control',
            'error_tracking',
            'recovery',
            'resource_monitor',
            'state_monitor',
            'strategy_blender',
            'jit_compiler',
            'error_classifier',
            'state_manager',
            'logging_manager',
            'parameter_validator'
        ]
        
        # Create and configure each component logger
        for component in component_names:
            logger = logging.getLogger(f'pid_controller.{component}')
            
            # Critical: Set propagate to False to prevent duplicate logs
            logger.propagate = False
            
            # Set appropriate log level
            logger.setLevel(self.root_logger.level)
            
            # Add the same handlers
            if not logger.handlers:
                for handler in self.root_logger.handlers:
                    logger.addHandler(handler)
            
            # Store in dictionary for quick access
            self.loggers[component] = logger
    
    def get_logger(self, component):
        """
        Get logger for a specific component.
        
        Args:
            component: Component name
        
        Returns:
            Logger instance
        """
        if component not in self.loggers:
            # Create if doesn't exist yet
            logger = logging.getLogger(f'pid_controller.{component}')
            logger.propagate = False
            logger.setLevel(self.root_logger.level)
            
            if not logger.handlers:
                for handler in self.root_logger.handlers:
                    logger.addHandler(handler)
                    
            self.loggers[component] = logger
            
        return self.loggers[component]
    
    def log_structured(self, component, event_type, message, params=None, 
                     level=logging.INFO, throttle_key=None, throttle_seconds=None):
        """
        Log a structured message with component, event type, and relevant parameters.
        
        Args:
            component: Component generating the log
            event_type: Type of event
            message: Log message
            params: Optional dictionary of parameters
            level: Log level
            throttle_key: Optional key for throttling
            throttle_seconds: Seconds to throttle similar logs
        
        Returns:
            bool: True if log was written, False if throttled
        """
        if params is None:
            params = {}
        
        # Apply throttling if requested
        if throttle_key is not None and throttle_seconds is not None:
            current_time = time.time()
            full_key = f"{component}_{event_type}_{throttle_key}"
            
            if full_key in self.throttle_timestamps:
                last_time = self.throttle_timestamps[full_key]
                if current_time - last_time < throttle_seconds:
                    return False  # Skip this log due to throttling
            
            # Update throttle timestamp
            self.throttle_timestamps[full_key] = current_time
        
        # Get the appropriate logger
        logger = self.get_logger(component)
        
        # Format message with parameters
        structured_msg = f"[{event_type}] {message}"
        if params:
            param_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                                for k, v in params.items())
            structured_msg += f" | {param_str}"
        
        # Log using the component's logger
        logger.log(level, structured_msg)
        
        # For critical errors, also log to ROS logger if available
        if level >= logging.ERROR and hasattr(self.node, 'get_logger'):
            self.node.get_logger().error(f"{component}: {structured_msg}")
        
        return True

    def get_debug_level(self):
        return self.debug_level

# Create a dummy logging manager for use before initialization
class DummyLoggingManager:
    """Simple logging implementation for when the real manager isn't initialized yet."""
    def log_structured(self, component, event_type, message, params=None, 
                     level=logging.INFO, throttle_key=None, throttle_seconds=None):
        """Simple implementation that logs to console until real manager is ready."""
        if params is None:
            params = {}
        
        # Format message with parameters
        structured_msg = f"[{event_type}] {message}"
        if params:
            param_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                                for k, v in params.items())
            structured_msg += f" | {param_str}"
        
        print(f"EARLY LOG - {component}: {structured_msg}")
        return True
    
    def get_logger(self, component):
        """Return a basic logger that writes to stdout."""
        return logging.getLogger(f"dummy.{component}")

# Initialize the dummy manager
_dummy_logging_manager = DummyLoggingManager()

def init_logging_system(node, debug_level=1, log_verbosity=1):
    """Initialize the global logging system."""
    global _logging_manager
    _logging_manager = LoggingManager(node, debug_level, log_verbosity)
    return _logging_manager

# Global function interfaces that use the logging manager

# Modify the global log_structured function
def log_structured(component, event_type, message, params=None, level=logging.INFO, 
                throttle_key=None, throttle_seconds=None, verbosity_level=1):
    """Global shortcut for log_structured with verbosity level check."""
    global _logging_manager, _dummy_logging_manager
    
    # Check if we should log based on verbosity level
    if _logging_manager is not None:
        if _logging_manager.log_verbosity < verbosity_level:
            return False  # Skip logging due to verbosity setting
        
        # Use the global logging manager
        return _logging_manager.log_structured(component, event_type, message, 
                                            params, level, throttle_key, throttle_seconds)
    else:
        # Use dummy manager if not initialized yet
        return _dummy_logging_manager.log_structured(component, event_type, message, 
                                                   params, level, throttle_key, throttle_seconds)

def get_logger(component):
    """Global shortcut for get_logger."""
    global _logging_manager, _dummy_logging_manager
    if _logging_manager is None:
        return _dummy_logging_manager.get_logger(component)
    return _logging_manager.get_logger(component)

# Specialized logging functions that use the centralized system

def log_state_transition(node, old_state, new_state, reason, params=None):
    """
    Log a state transition with the reason and relevant parameters.
    State transitions are important, so we don't throttle these.
    """
    if params is None:
        params = {}
    
    # Add timestamp diff since last state change
    current_time = time.time()
    time_in_prev_state = current_time - getattr(node, 'last_state_change_time', current_time)
    params['time_in_prev_state'] = time_in_prev_state
    
    # Update last state change time
    node.last_state_change_time = current_time
    
    log_structured('state_manager', 'STATE_TRANSITION', 
                  f"{old_state} → {new_state}: {reason}", 
                  params)

def log_pid_state(controller_name, error, output, p_term, i_term, d_term, current_gains=None):
    """
    Log detailed PID controller state with throttling.
    Limits PID state logs to once per second per controller.
    """
    params = {
        'error': error,
        'output': output,
        'p_term': p_term,
        'i_term': i_term,
        'd_term': d_term
    }
    
    if current_gains:
        params['kp'] = current_gains[0]
        params['ki'] = current_gains[1]
        params['kd'] = current_gains[2]
    
    # Throttle PID state logs by controller name
    log_structured('pid_controller', 'PID_STATE', 
                  f"{controller_name} update", 
                  params,
                  throttle_key=controller_name,
                  throttle_seconds=2.0)  # Once per second

def log_strategy_selection(key, selected_strategy, candidate_strategies=None, reason=None):
    """
    Log strategy selection with reasoning and throttling.
    """
    params = {
        'key': key,
        'strategy_name': selected_strategy['strategy_name'],
        'forward_scale': selected_strategy['forward_scale'],
        'lateral_scale': selected_strategy['lateral_scale'],
        'angular_scale': selected_strategy['angular_scale']
    }
    
    msg = f"Selected strategy: {selected_strategy['strategy_name']}"
    if reason:
        msg += f" - {reason}"
    
    # Throttle strategy selection logs by strategy name
    log_structured('strategy_selector', 'STRATEGY_SELECTION', 
                  msg, 
                  params,
                  throttle_key=selected_strategy['strategy_name'],
                  throttle_seconds=1)  # Max once per second

def log_strategy_blending(start_strategy, target_strategy, blend_factor, blend_duration):
    """
    Log strategy blending progress with reduced frequency.
    """
    params = {
        'start': start_strategy.name,
        'target': target_strategy.name,
        'blend_factor': blend_factor,
        'blend_duration': blend_duration,
        'effective_duration': getattr(start_strategy, 'effective_blend_duration', blend_duration)
    }
    
    # Only log at start and end of blend to reduce noise (removed middle log point)
    should_log = (
        abs(blend_factor) < 0.05 or      # Start of blend        
        abs(blend_factor - 1.0) < 0.05    # End of blend
    )
    
    if should_log:
        log_structured('strategy_blender', 'STRATEGY_BLEND', 
                      f"Blending {blend_factor*100:.1f}% complete", 
                      params,
                      throttle_key=f"{start_strategy.name}_to_{target_strategy.name}",
                      throttle_seconds=0.6)  # Allow reasonable updates during blend

def log_error_categorization(error_type, raw_error, category, threshold, prev_category=None):
    """
    Log error categorization with thresholds.
    Only log when category changes to reduce noise.
    """
    # Only log if category has changed
    if prev_category is not None and category == prev_category:
        return  # Skip logging if category hasn't changed
    
    params = {
        'raw_error': raw_error,
        'category': category,
        'threshold': threshold
    }
    
    if prev_category:
        params['prev_category'] = prev_category
        msg = f"{error_type.capitalize()} error {raw_error:.3f} categorized as '{category}' (was '{prev_category}')"
    else:
        msg = f"{error_type.capitalize()} error {raw_error:.3f} categorized as '{category}'"
    
    log_structured('error_classifier', 'ERROR_CATEGORIZATION', 
                  msg, 
                  params)

def log_initialization_step(component, step, status, elapsed_time=None):
    """
    Log initialization progress.
    Initialization logs are important so we don't throttle them.
    """
    params = {'status': status}
    if elapsed_time is not None:
        params['elapsed_ms'] = elapsed_time * 1000  # convert to ms
    
    log_structured(component, 'INITIALIZATION', 
                  f"{step}: {status}", 
                  params)

def log_jit_compilation(function_name, success, elapsed_time=None):
    """
    Log JIT compilation status with reduced verbosity.
    Only log failures and the first successful compilation.
    """
    params = {'success': success}
    if elapsed_time is not None:
        params['elapsed_ms'] = elapsed_time * 1000  # convert to ms
    
    status = "completed" if success else "failed"
    
    # Use a higher log level (INFO) to reduce log volume
    # Only critical failures are logged at WARNING level
    log_level = logging.INFO if success else logging.WARNING
    
    # Only log the first successful compilation per function
    if success:
        throttle_key = f"success_{function_name}"
        throttle_seconds = 3600  # Only log success once per hour (effectively once)
    else:
        # Always log failures
        throttle_key = None
        throttle_seconds = None
    
    log_structured('jit_compiler', 'JIT_COMPILATION', 
                  f"Function '{function_name}' compilation {status}", 
                  params, log_level, throttle_key, throttle_seconds)

def log_target_filter_update(raw_position, filtered_position, predicted_position=None, confidence=None):
    """
    Log target filter updates and predictions with reduced frequency.
    """
    params = {
        'raw_dist': raw_position[0],
        'raw_lateral': raw_position[1],
        'raw_angle': raw_position[2],
        'filtered_dist': filtered_position[0],
        'filtered_lateral': filtered_position[1],
        'filtered_angle': filtered_position[2]
    }
    
    msg = "Target filter update"
    
    if predicted_position:
        params['pred_dist'] = predicted_position[0]
        params['pred_lateral'] = predicted_position[1]
        params['pred_angle'] = predicted_position[2]
        if confidence:
            params['confidence'] = confidence
        msg = "Target prediction"
        
        # Throttle prediction logs (these happen frequently)
        throttle_key = "prediction"
        throttle_seconds = 1.0  # Once per second max
    else:
        # Throttle regular updates even more aggressively
        throttle_key = "filter_update"
        throttle_seconds = 2.0  # Once per 2 seconds max
    
    log_structured('target_filter', 'FILTER_UPDATE', 
                  msg, 
                  params,
                  throttle_key=throttle_key,
                  throttle_seconds=throttle_seconds)

def log_velocity_limiting(raw_velocities, limited_velocities, reason=None):
    """
    Log velocity limiting decisions with throttling and significance threshold.
    Only log when velocity is limited significantly.
    """
    # Calculate how much velocity was limited
    limit_pct = [0, 0, 0]
    significant_limiting = False
    
    for i in range(3):
        if abs(raw_velocities[i]) > 0.01:  # Avoid division by zero
            limit_pct[i] = abs(limited_velocities[i] - raw_velocities[i]) / abs(raw_velocities[i]) * 100
            if limit_pct[i] > 25:  # Only consider significant if >15% change
                significant_limiting = True
    
    # Skip logging if velocity wasn't limited significantly
    if not significant_limiting:
        return
        
    params = {
        'raw_x': raw_velocities[0],
        'raw_y': raw_velocities[1],
        'raw_angular': raw_velocities[2],
        'limited_x': limited_velocities[0],
        'limited_y': limited_velocities[1],
        'limited_angular': limited_velocities[2]
    }
    
    msg = "Velocity limiting applied"
    if reason:
        msg += f": {reason}"
    
    # Throttle velocity limiting logs
    log_structured('velocity_limiter', 'VELOCITY_LIMIT', 
                  msg, 
                  params,
                  throttle_key="velocity_limit",
                  throttle_seconds=1.5)  

def log_resource_usage(cpu_usage, memory_usage, cycle_time=None, rate_adjustment=None):
    """
    Log system resource usage with reduced frequency.
    Only log on significant changes or every 10 seconds.
    """
    # Check if enough time has passed since last resource log
    current_time = time.time()
    
    # Use the centralized logging manager's throttling system instead
    params = {
        'cpu_pct': cpu_usage,
        'mem_pct': memory_usage
    }
    
    if cycle_time is not None:
        params['cycle_time_ms'] = cycle_time * 1000  # convert to ms
    
    if rate_adjustment is not None:
        params['rate_adj'] = rate_adjustment
    
    # Use significant change detection
    significant_change_key = "significant_change"
    if hasattr(log_resource_usage, "last_cpu") and hasattr(log_resource_usage, "last_mem"):
        last_cpu = log_resource_usage.last_cpu
        last_mem = log_resource_usage.last_mem
        
        if abs(cpu_usage - last_cpu) > 25 or abs(memory_usage - last_mem) > 15:
            # Log immediately for significant changes
            log_structured('resource_monitor', 'RESOURCE_USAGE', 
                         f"CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%", 
                         params,
                         throttle_key=significant_change_key,
                         throttle_seconds=4.0)  # Allow significant changes every 2s
    
    # Regular periodic logging with longer throttle time
    log_structured('resource_monitor', 'RESOURCE_USAGE', 
                 f"CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%", 
                 params,
                 throttle_key="periodic",
                 throttle_seconds=10.0)  # Regular updates every 10s
                
    # Update last logged values
    log_resource_usage.last_cpu = cpu_usage
    log_resource_usage.last_mem = memory_usage

def log_parameter_validation(component, param_name, expected_value, actual_value):
    """
    Log parameter validation checks.
    Only log mismatches at WARNING level, matches at DEBUG level.
    """
    match = expected_value == actual_value
    params = {
        'expected': expected_value,
        'actual': actual_value,
        'match': match
    }
    
    if match:
        msg = f"Parameter '{param_name}' correctly set"
        level = logging.DEBUG  # Successful validations at debug level only
    else:
        msg = f"Parameter '{param_name}' MISMATCH"
        level = logging.WARNING  # Mismatches are important to log
    
    log_structured(component, 'PARAMETER_VALIDATION', msg, params, level)

# Additional helper functions to reduce log volume for common events

def log_movement_status(status, error_value, threshold, robot_state=None, params=None):
    """
    Log movement status changes with throttling.
    """
    if params is None:
        params = {}
        
    if robot_state:
        params['robot_state'] = robot_state
        
    params['error'] = error_value
    params['threshold'] = threshold
    
    # Throttle movement status logs
    log_structured('motion_control', 'MOVEMENT_STATUS', 
                  status, 
                  params,
                  throttle_key=status.lower().replace(" ", "_"),
                  throttle_seconds=0.5)  # Max twice per second

def log_direction_change(component, prev_vel, new_vel, consistency=None):
    """
    Log direction changes with throttling and reduced verbosity.
    """
    params = {
        'prev_x': prev_vel[0],
        'prev_y': prev_vel[1],
        'new_x': new_vel[0],
        'new_y': new_vel[1]
    }
    
    if consistency is not None:
        params['consistency'] = consistency
    
    # Throttle direction change logs
    log_structured(component, 'DIRECTION_CHANGE', 
                  "Direction change detected", 
                  params,
                  throttle_key="direction_change",
                  throttle_seconds=0.5)  # Max twice per second

def log_target_movement(is_moving, velocity, threshold, params=None):
    """
    Log target movement status with throttling.
    """
    if params is None:
        params = {}
        
    params['velocity'] = velocity
    params['threshold'] = threshold
    
    status = "Target started moving" if is_moving else "Target stopped moving"
    
    # Throttle movement status logs
    log_structured('target_filter', 'MOVEMENT_STATUS', 
                  status, 
                  params,
                  throttle_key="movement_status",
                  throttle_seconds=1.0)  # Once per second max

def log_robot_motion(is_starting, velocity=None, prev_velocity=None, robot_state=None):
    """
    Log robot motion start/stop with throttling.
    """
    params = {}
    if robot_state:
        params['robot_state'] = robot_state
        
    if is_starting and velocity:
        params['velocity'] = velocity
        msg = "Robot starting movement"
    else:
        params['prev_velocity'] = prev_velocity
        msg = "Robot stopping"
    
    # Throttle motion start/stop logs
    log_structured('motion_commander', 'ROBOT_MOTION_' + ('START' if is_starting else 'STOP'), 
                  msg, 
                  params,
                  throttle_key="motion_" + ("start" if is_starting else "stop"),
                  throttle_seconds=0.5)  # Max twice per second

def log_performance_metrics(component, metrics, level=logging.INFO):
    """
    Log performance metrics with throttling.
    """
    # Throttle performance logs
    log_structured(component, 'PERFORMANCE_METRICS', 
                  f"Performance metrics", 
                  metrics,
                  level=level,
                  throttle_key="perf_metrics",
                  throttle_seconds=5.0)  # Max once per 5 seconds

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
LOG_THROTTLE_CONTROL = 4.0     # Seconds between control loop status logs
LOG_THROTTLE_STATE = 1       # Seconds between state change logs
LOG_THROTTLE_DIAG = 2.5        # Seconds between diagnostic logs


# Memory optimization classes
class LightweightBuffer:
    """Memory-efficient buffer for storing historical data with pre-allocated storage."""
    
    def __init__(self, max_size=10, default_value=(0.0, 0.0, 0.0)):
        """Initialize a fixed-size circular buffer."""
        self.data = [default_value] * max_size  # Pre-allocate with default values
        self.next_index = 0
        self.count = 0
        self.max_size = max_size
    
    def add(self, value):
        """Add a new value to the buffer, overwriting oldest if full."""
        self.data[self.next_index] = value
        self.next_index = (self.next_index + 1) % self.max_size
        self.count = min(self.count + 1, self.max_size)
    
    def get_all(self):
        """Get all values currently in the buffer in chronological order."""
        if self.count == 0:
            return []
        
        result = []
        
        if self.count < self.max_size:
            # Buffer isn't full yet, return all items from 0 to count
            for i in range(self.count):
                result.append(self.data[i])
        else:
            # Buffer is full, need to wrap around
            # Start from oldest item (at next_index) and go around
            for i in range(self.max_size):
                idx = (self.next_index + i) % self.max_size
                result.append(self.data[idx])
        
        return result
        
    def get_latest(self, n=1):
        """Get the latest n values (default is just the latest one)."""
        if self.count == 0:
            return []
        
        n = min(n, self.count)
        result = []
        
        # Calculate positions of the n latest elements
        for i in range(n):
            if self.count < self.max_size:
                # Simple case: buffer isn't full yet
                idx = self.count - n + i
            else:
                # Buffer is full, handle circular indexing
                idx = (self.next_index - n + i) % self.max_size
            
            # Append item at calculated index
            result.append(self.data[idx])
        
        return result
    
    def clear(self):
        """Clear the buffer."""
        self.next_index = 0
        self.count = 0
        # Reset all data to default values
        for i in range(self.max_size):
            self.data[i] = (0.0, 0.0, 0.0)

class TTLDict:
    """
    Dictionary with time-to-live functionality for entries.
    Automatically cleans up expired entries during access.
    """
    def __init__(self, default_ttl=60.0):
        """
        Initialize a TTL dictionary.
        
        Args:
            default_ttl (float): Default TTL in seconds for entries
        """
        self.data = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
    
    def __setitem__(self, key, value, ttl=None):
        """Set an item with a specific TTL."""
        self.data[key] = value
        self.timestamps[key] = time.time()
        # Store TTL with the key if provided
        if ttl is not None:
            if not hasattr(self, 'ttls'):
                self.ttls = {}
            self.ttls[key] = ttl
    
    def set(self, key, value, ttl=None):
        """Set an item with an optional custom TTL."""
        self.__setitem__(key, value, ttl)
    
    def __getitem__(self, key):
        """Get an item, removing if expired."""
        self._cleanup(key)
        if key in self.data:
            return self.data[key]
        raise KeyError(key)
    
    def get(self, key, default=None):
        """Get an item with a default value if missing or expired."""
        self._cleanup(key)
        return self.data.get(key, default)
    
    def _cleanup(self, key=None):
        """Clean up expired entries."""
        current_time = time.time()
        
        # Clean specific key if provided
        if key is not None:
            if key in self.timestamps:
                ttl = self.ttls.get(key, self.default_ttl) if hasattr(self, 'ttls') else self.default_ttl
                if current_time - self.timestamps[key] > ttl:
                    del self.data[key]
                    del self.timestamps[key]
                    if hasattr(self, 'ttls') and key in self.ttls:
                        del self.ttls[key]
            return
        
        # Full cleanup (expensive, use sparingly)
        keys_to_remove = []
        for k, timestamp in self.timestamps.items():
            ttl = self.ttls.get(k, self.default_ttl) if hasattr(self, 'ttls') else self.default_ttl
            if current_time - timestamp > ttl:
                keys_to_remove.append(k)
        
        for k in keys_to_remove:
            del self.data[k]
            del self.timestamps[k]
            if hasattr(self, 'ttls') and k in self.ttls:
                del self.ttls[k]
    
    def cleanup_all(self):
        """Force cleanup of all expired entries."""
        self._cleanup()
    
    def __contains__(self, key):
        """Check if key exists and is not expired."""
        self._cleanup(key)
        return key in self.data

logger = logging.getLogger('pid_controller.matrix4x4')

# Core matrix operations optimized with Numba
@numba.jit(nopython=True, fastmath=True)
def transform_point_jit(matrix_data, x, y, z):
    """
    Transform a 3D point using the provided matrix data.
    
    Args:
        matrix_data: 4x4 matrix as a numpy array
        x, y, z: Point coordinates
    
    Returns:
        tuple: Transformed (x, y, z) coordinates
    """
    tx = matrix_data[0, 0] * x + matrix_data[0, 1] * y + matrix_data[0, 2] * z + matrix_data[0, 3]
    ty = matrix_data[1, 0] * x + matrix_data[1, 1] * y + matrix_data[1, 2] * z + matrix_data[1, 3]
    tz = matrix_data[2, 0] * x + matrix_data[2, 1] * y + matrix_data[2, 2] * z + matrix_data[2, 3]
    
    return tx, ty, tz

@numba.jit(nopython=True, fastmath=True)
def transform_vector_jit(matrix_data, x, y, z):
    """
    Transform a 3D vector using the provided matrix data (no translation).
    
    Args:
        matrix_data: 4x4 matrix as a numpy array
        x, y, z: Vector components
    
    Returns:
        tuple: Transformed (x, y, z) vector
    """
    tx = matrix_data[0, 0] * x + matrix_data[0, 1] * y + matrix_data[0, 2] * z
    ty = matrix_data[1, 0] * x + matrix_data[1, 1] * y + matrix_data[1, 2] * z
    tz = matrix_data[2, 0] * x + matrix_data[2, 1] * y + matrix_data[2, 2] * z
    
    return tx, ty, tz

@numba.jit(nopython=True, fastmath=True)
def quaternion_to_matrix_jit(qx, qy, qz, qw, tx, ty, tz):
    """
    Convert quaternion and translation to 4x4 matrix.
    
    Args:
        qx, qy, qz, qw: Quaternion components
        tx, ty, tz: Translation components
    
    Returns:
        numpy.ndarray: 4x4 matrix
    """
    # Initialize matrix
    matrix = np.eye(4, dtype=np.float32)
    
    # Precompute common products
    xx = qx * qx
    xy = qx * qy
    xz = qx * qz
    xw = qx * qw
    yy = qy * qy
    yz = qy * qz
    yw = qy * qw
    zz = qz * qz
    zw = qz * qw
    
    # Fill rotation part (3x3 top-left)
    matrix[0, 0] = 1.0 - 2.0 * (yy + zz)
    matrix[0, 1] = 2.0 * (xy - zw)
    matrix[0, 2] = 2.0 * (xz + yw)
    
    matrix[1, 0] = 2.0 * (xy + zw)
    matrix[1, 1] = 1.0 - 2.0 * (xx + zz)
    matrix[1, 2] = 2.0 * (yz - xw)
    
    matrix[2, 0] = 2.0 * (xz - yw)
    matrix[2, 1] = 2.0 * (yz + xw)
    matrix[2, 2] = 1.0 - 2.0 * (xx + yy)
    
    # Fill translation part (right column)
    matrix[0, 3] = tx
    matrix[1, 3] = ty
    matrix[2, 3] = tz
    
    return matrix

@numba.jit(nopython=True, fastmath=True)
def matrix_multiply_jit(a, b):
    """
    Multiply two 4x4 matrices.
    
    Args:
        a: First 4x4 matrix
        b: Second 4x4 matrix
    
    Returns:
        numpy.ndarray: Resulting 4x4 matrix
    """
    result = np.zeros((4, 4), dtype=np.float32)
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                result[i, j] += a[i, k] * b[k, j]
    
    return result

@numba.jit(nopython=True, fastmath=True)
def matrix_inverse_jit(matrix):
    """
    Compute the inverse of a 4x4 transformation matrix more efficiently.
    This assumes the matrix is a valid transformation matrix (rotation + translation).
    
    Args:
        matrix: 4x4 transformation matrix
    
    Returns:
        numpy.ndarray: Inverse 4x4 matrix
    """
    result = np.eye(4, dtype=np.float32)
    
    # Extract the 3x3 rotation matrix
    r00 = matrix[0, 0]
    r01 = matrix[0, 1]
    r02 = matrix[0, 2]
    r10 = matrix[1, 0]
    r11 = matrix[1, 1]
    r12 = matrix[1, 2]
    r20 = matrix[2, 0]
    r21 = matrix[2, 1]
    r22 = matrix[2, 2]
    
    # Extract the translation
    tx = matrix[0, 3]
    ty = matrix[1, 3]
    tz = matrix[2, 3]
    
    # Transpose of rotation matrix
    result[0, 0] = r00
    result[0, 1] = r10
    result[0, 2] = r20
    result[1, 0] = r01
    result[1, 1] = r11
    result[1, 2] = r21
    result[2, 0] = r02
    result[2, 1] = r12
    result[2, 2] = r22
    
    # New translation = -R^T * t
    result[0, 3] = -(r00 * tx + r10 * ty + r20 * tz)
    result[1, 3] = -(r01 * tx + r11 * ty + r21 * tz)
    result[2, 3] = -(r02 * tx + r12 * ty + r22 * tz)
    
    return result


class Matrix4x4:
    """
    Efficient 4x4 matrix implementation for transform operations.
    Optimized with Numba for better performance.
    """
    def __init__(self):
        """Initialize identity matrix."""
        self.logger = get_logger('matrix4x4')
        # Initialize as identity matrix (row-major order)
        self.data = np.eye(4, dtype=np.float32)                
        # Warm up JIT compilation
        self._warmup_jit()
        
    def _warmup_jit(self):
        """Warm up JIT compilation to avoid delays during operation with reduced logging."""
    
        start_time = time.time()
        
        # Only log at DEBUG level to reduce verbosity        
        self.logger.debug("Starting JIT warmup for Matrix4x4")
        
        # Create dummy data
        dummy_matrix = np.eye(4, dtype=np.float32)
        
        # Call JIT functions with dummy data without logging each one
        _ = transform_point_jit(dummy_matrix, 1.0, 2.0, 3.0)
        _ = transform_vector_jit(dummy_matrix, 1.0, 2.0, 3.0)
        _ = quaternion_to_matrix_jit(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        _ = matrix_multiply_jit(dummy_matrix, dummy_matrix)
        _ = matrix_inverse_jit(dummy_matrix)
        
        elapsed_time = time.time() - start_time
        
        # Log completion once at INFO level
        self.logger.info(f"JIT compilation warmup completed for Matrix4x4 in {elapsed_time*1000:.1f}ms")

    @classmethod
    def from_tf_transform(cls, transform):
        """
        Create matrix from ROS transform with optimized computation.
        
        Args:
            transform: ROS Transform message
        
        Returns:
            Matrix4x4: New matrix with transform data
        """
        matrix = cls()
        
        try:
            # Extract quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            
            # Extract translation
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            tz = transform.transform.translation.z
            
            # Use Numba optimized function to convert quaternion to matrix
            matrix.data = quaternion_to_matrix_jit(qx, qy, qz, qw, tx, ty, tz)
            
        except Exception as e:
            self.logger.warning(f"Error in from_tf_transform: {str(e)}. Using fallback method.")
            
            # Fallback non-JIT method
            # Extract quaternion
            qx = transform.transform.rotation.x
            qy = transform.transform.rotation.y
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            
            # Convert quaternion to rotation matrix
            # Precompute common products
            xx = qx * qx
            xy = qx * qy
            xz = qx * qz
            xw = qx * qw
            yy = qy * qy
            yz = qy * qz
            yw = qy * qw
            zz = qz * qz
            zw = qz * qw
            
            # Fill rotation part (3x3 top-left)
            matrix.data[0, 0] = 1.0 - 2.0 * (yy + zz)
            matrix.data[0, 1] = 2.0 * (xy - zw)
            matrix.data[0, 2] = 2.0 * (xz + yw)
            
            matrix.data[1, 0] = 2.0 * (xy + zw)
            matrix.data[1, 1] = 1.0 - 2.0 * (xx + zz)
            matrix.data[1, 2] = 2.0 * (yz - xw)
            
            matrix.data[2, 0] = 2.0 * (xz - yw)
            matrix.data[2, 1] = 2.0 * (yz + xw)
            matrix.data[2, 2] = 1.0 - 2.0 * (xx + yy)
            
            # Fill translation part (right column)
            matrix.data[0, 3] = transform.transform.translation.x
            matrix.data[1, 3] = transform.transform.translation.y
            matrix.data[2, 3] = transform.transform.translation.z
            
        return matrix
    
    def transform_point(self, x, y, z):
        """
        Transform a 3D point using this matrix.
        
        Args:
            x, y, z (float): Point coordinates
        
        Returns:
            tuple: Transformed (x, y, z) coordinates
        """
        try:
            # Use Numba optimized function
            return transform_point_jit(self.data, x, y, z)
        except Exception as e:
            self.logger.warning(f"Error in transform_point: {str(e)}. Using fallback method.")
            
            # Fallback non-JIT method
            tx = self.data[0, 0] * x + self.data[0, 1] * y + self.data[0, 2] * z + self.data[0, 3]
            ty = self.data[1, 0] * x + self.data[1, 1] * y + self.data[1, 2] * z + self.data[1, 3]
            tz = self.data[2, 0] * x + self.data[2, 1] * y + self.data[2, 2] * z + self.data[2, 3]
            
            return (tx, ty, tz)
    
    def transform_vector(self, x, y, z):
        """
        Transform a 3D vector using this matrix (no translation).
        
        Args:
            x, y, z (float): Vector components
        
        Returns:
            tuple: Transformed (x, y, z) vector
        """
        try:
            # Use Numba optimized function
            return transform_vector_jit(self.data, x, y, z)
        except Exception as e:
            logger.warning(f"Error in transform_vector: {str(e)}. Using fallback method.")
            
            # Fallback non-JIT method
            tx = self.data[0, 0] * x + self.data[0, 1] * y + self.data[0, 2] * z
            ty = self.data[1, 0] * x + self.data[1, 1] * y + self.data[1, 2] * z
            tz = self.data[2, 0] * x + self.data[2, 1] * y + self.data[2, 2] * z
            
            return (tx, ty, tz)
    
    def multiply(self, other):
        """
        Multiply this matrix by another matrix.
        
        Args:
            other: Another Matrix4x4 instance
            
        Returns:
            Matrix4x4: Result of multiplication
        """
        result = Matrix4x4()
        
        try:
            # Use Numba optimized function
            result.data = matrix_multiply_jit(self.data, other.data)
        except Exception as e:
            logger.warning(f"Error in matrix multiply: {str(e)}. Using fallback method.")
            
            # Fallback to numpy matrix multiplication
            result.data = np.matmul(self.data, other.data)
            
        return result
    
    def inverse(self):
        """
        Compute the inverse of this matrix.
        
        Returns:
            Matrix4x4: Inverse matrix
        """
        result = Matrix4x4()
        
        try:
            # Use Numba optimized function for transform matrix inverse
            result.data = matrix_inverse_jit(self.data)
        except Exception as e:
            logger.warning(f"Error in matrix inverse: {str(e)}. Using fallback method.")
            
            # Fallback to numpy matrix inverse
            result.data = np.linalg.inv(self.data)
            
        return result
    
    def to_array(self):
        """
        Convert matrix to a flat array.
        
        Returns:
            list: Flattened 16-element array
        """
        return self.data.flatten().tolist()
    
    @classmethod
    def from_array(cls, array):
        """
        Create matrix from a 16-element array.
        
        Args:
            array: 16-element array in row-major order
            
        Returns:
            Matrix4x4: New matrix with array data
        """
        matrix = cls()
        matrix.data = np.array(array, dtype=np.float32).reshape(4, 4)
        return matrix
    
    def copy(self):
        """
        Create a copy of this matrix.
        
        Returns:
            Matrix4x4: Copy of this matrix
        """
        matrix = Matrix4x4()
        matrix.data = self.data.copy()
        return matrix

@numba.jit(nopython=True, fastmath=True)
def update_smoothed_values(current_velocity, current_acceleration, last_velocity, last_acceleration):
    """
    Apply low-pass filtering to smooth velocity and acceleration.
    
    Args:
        current_velocity: Raw calculated velocity
        current_acceleration: Raw calculated acceleration
        last_velocity: Previous smoothed velocity
        last_acceleration: Previous smoothed acceleration
        
    Returns:
        tuple: (smoothed_velocity, smoothed_acceleration)
    """
    # Initialize output arrays
    smoothed_velocity = np.zeros(3, dtype=np.float32)
    smoothed_acceleration = np.zeros(3, dtype=np.float32)
    
    # Apply low-pass filter for velocity with improved smoothing
    alpha = 0.75  # Reduced from 0.85 for more smoothing
    smoothed_velocity[0] = alpha * current_velocity[0] + (1 - alpha) * last_velocity[0]
    smoothed_velocity[1] = alpha * current_velocity[1] + (1 - alpha) * last_velocity[1]
    smoothed_velocity[2] = alpha * current_velocity[2] + (1 - alpha) * last_velocity[2]
    
    # Apply low-pass filter for acceleration
    alpha_a = 0.25  # Reduced from 0.3 for more smoothing
    smoothed_acceleration[0] = alpha_a * current_acceleration[0] + (1 - alpha_a) * last_acceleration[0]
    smoothed_acceleration[1] = alpha_a * current_acceleration[1] + (1 - alpha_a) * last_acceleration[1]
    smoothed_acceleration[2] = alpha_a * current_acceleration[2] + (1 - alpha_a) * last_acceleration[2]
    
    return smoothed_velocity, smoothed_acceleration

@numba.jit(nopython=True, fastmath=True)
def detect_direction_change(prev_velocity, new_velocity):
    """
    Detect significant direction changes in motion.
    
    Args:
        prev_velocity: Previous velocity vector
        new_velocity: New velocity vector
        
    Returns:
        bool: True if direction changed significantly
    """
    # Check if velocities are significant
    prev_mag = math.sqrt(prev_velocity[0]**2 + prev_velocity[1]**2)
    new_mag = math.sqrt(new_velocity[0]**2 + new_velocity[1]**2)
    
    # Only detect changes when there's significant movement
    if prev_mag > 0.05 and new_mag > 0.05:  # Increased threshold from 0.01 to 0.05
        # Calculate dot product to check direction change
        dot_product = prev_velocity[0] * new_velocity[0] + prev_velocity[1] * new_velocity[1]
        cos_angle = dot_product / (prev_mag * new_mag)
        
        # Consider significant direction change if angle > 45 degrees (instead of 30)
        if cos_angle < 0.707:  # cos(45°) ≈ 0.707
            return True
    
    return False

@numba.jit(nopython=True, fastmath=True)
def calculate_motion_direction(current_velocity, last_motion_direction):
    """
    Calculate normalized motion direction vector.
    
    Args:
        current_velocity: Current velocity vector
        last_motion_direction: Previous motion direction
        
    Returns:
        tuple: Motion direction as (x, y, angular)
    """
    # Calculate velocity magnitude
    vel_magnitude = math.sqrt(current_velocity[0]**2 + current_velocity[1]**2)
    
    # Initialize new direction vector
    new_direction = np.zeros(3, dtype=np.float32)
    new_direction[2] = current_velocity[2]  # Angular component
    
    if vel_magnitude > 0.05:  # Increased threshold from 0.01 to 0.05
        # Calculate normalized direction
        new_direction[0] = current_velocity[0] / vel_magnitude
        new_direction[1] = current_velocity[1] / vel_magnitude
        
        # Smooth direction updates with increased smoothing for stability
        alpha_dir = 0.6  # Reduced from 0.7 for more smoothing
        new_direction[0] = alpha_dir * new_direction[0] + (1 - alpha_dir) * last_motion_direction[0]
        new_direction[1] = alpha_dir * new_direction[1] + (1 - alpha_dir) * last_motion_direction[1]
    else:
        # Maintain previous direction if motion is too small
        new_direction[0] = last_motion_direction[0]
        new_direction[1] = last_motion_direction[1]
    
    return new_direction

@numba.jit(nopython=True, fastmath=True)
def calculate_weighted_position(recent_positions, weights):
    """
    Calculate weighted average position.
    
    Args:
        recent_positions: Array of recent positions
        weights: Array of weights for each position
        
    Returns:
        tuple: (x, y, angle) weighted position
    """
    # Initialize weighted sums
    x_sum = 0.0
    y_sum = 0.0
    angle_sum = 0.0
    
    # Apply weights to each position
    for i in range(len(recent_positions)):
        x_sum += recent_positions[i][0] * weights[i]
        y_sum += recent_positions[i][1] * weights[i]
        angle_sum += recent_positions[i][2] * weights[i]
    
    return (x_sum, y_sum, angle_sum)

@numba.jit(nopython=True, fastmath=True)
def predict_future_position(current_position, velocity, acceleration, t, movement_consistency):
    """
    Predict future position based on physics formulas with improved safety limits.
    
    Args:
        current_position: Current (x, y, angle) position
        velocity: Current (vx, vy, vangle) velocity
        acceleration: Current (ax, ay, aangle) acceleration
        t: Prediction time horizon
        movement_consistency: Consistency of movement pattern (0-1)
        
    Returns:
        tuple: Predicted (x, y, angle) position with safety limits
    """
    # Reduce acceleration weight for less aggressive predictions
    accel_weight = 0.3 * movement_consistency  # Reduced from 0.5
    
    # Calculate position using physics formulas
    pred_x = (current_position[0] + 
              velocity[0] * t + 
              0.5 * acceleration[0] * t**2 * accel_weight)
    
    pred_y = (current_position[1] + 
              velocity[1] * t + 
              0.5 * acceleration[1] * t**2 * accel_weight)
    
    pred_angle = (current_position[2] + 
                  velocity[2] * t)
    
    # Apply hard safety limits to predictions
    # 1. Limit distance prediction to 1.5x current distance (reduced from 2x)
    if abs(current_position[0]) > 0.1:
        max_distance = 1.5 * abs(current_position[0])
        if abs(pred_x) > max_distance:
            # Cap prediction at 1.5x current distance
            sign = 1.0 if pred_x >= 0 else -1.0
            pred_x = sign * max_distance
    
    # 2. Limit lateral prediction to 2x current lateral (reduced from 3x)
    if abs(current_position[1]) > 0.05:
        max_lateral = 2.0 * abs(current_position[1])
        if abs(pred_y) > max_lateral:
            # Cap prediction at 2x current lateral
            sign = 1.0 if pred_y >= 0 else -1.0
            pred_y = sign * max_lateral
    
    # 3. Limit angle prediction to reasonable range
    max_angle = 1.57  # π/2 ≈ 1.57
    if abs(pred_angle) > max_angle:
        # Cap angle prediction
        sign = 1.0 if pred_angle >= 0 else -1.0
        pred_angle = sign * max_angle
    
    return (pred_x, pred_y, pred_angle)

@numba.jit(nopython=True, fastmath=True)
def calculate_movement_consistency(trajectory_positions, trajectory_times):
    """
    Calculate how consistent the movement trajectory is.
    
    Args:
        trajectory_positions: Array of position tuples
        trajectory_times: Array of timestamps
        
    Returns:
        float: Movement consistency (0-1)
    """
    if len(trajectory_positions) < 5:
        return 0.0
    
    # Calculate average displacement direction
    dx = trajectory_positions[-1][0] - trajectory_positions[0][0]
    dy = trajectory_positions[-1][1] - trajectory_positions[0][1]
    
    overall_len = math.sqrt(dx**2 + dy**2)
    if overall_len < 0.05:  # Increased threshold from 0.01 to 0.05
        return 0.0
    
    # Normalize overall direction
    overall_dx = dx / overall_len
    overall_dy = dy / overall_len
    
    # Check how well each segment aligns with overall direction
    consistency_sum = 0.0
    count = 0
    
    for i in range(1, len(trajectory_positions)):
        segment_dx = trajectory_positions[i][0] - trajectory_positions[i-1][0]
        segment_dy = trajectory_positions[i][1] - trajectory_positions[i-1][1]
        
        # Skip tiny movements
        segment_len = math.sqrt(segment_dx**2 + segment_dy**2)
        if segment_len < 0.02:  # Increased threshold from 0.01 to 0.02
            continue
            
        # Normalize segment direction
        segment_dx /= segment_len
        segment_dy /= segment_len
        
        # Calculate alignment using dot product
        alignment = segment_dx * overall_dx + segment_dy * overall_dy
        consistency_sum += max(0.0, alignment)  # Only count positive alignment
        count += 1
    
    # Calculate average consistency
    if count > 0:
        return consistency_sum / count
    else:
        return 0.0

@numba.jit(nopython=True, fastmath=True)
def validate_prediction_jit(raw_position, filtered_position, predicted_position,
                         max_dist_ratio, max_lateral_ratio):
    """
    JIT-compiled prediction validation.
    
    Args:
        raw_position: Raw position (3-element array)
        filtered_position: Filtered position (3-element array)
        predicted_position: Predicted position (3-element array)
        max_dist_ratio: Maximum allowed ratio between predicted and filtered distance
        max_lateral_ratio: Maximum allowed ratio between predicted and filtered lateral
        
    Returns:
        tuple: (is_valid, confidence_adjustment, reason_code)
            reason_code: 0=valid, 1=distance_too_large, 2=lateral_too_large, 
                       3=angle_unrealistic, 4=direction_impossible
    """
    # Default confidence adjustment
    confidence_adj = 1.0
    
    # 1. Check if distance prediction is reasonable (below max_dist_ratio)
    filtered_dist = abs(filtered_position[0])
    predicted_dist = abs(predicted_position[0])
    
    if filtered_dist > 0.05:  # Only apply ratio check if filtered distance is significant
        dist_ratio = predicted_dist / filtered_dist
        
        if dist_ratio > max_dist_ratio:
            return False, 0.0, 1  # Distance too large
        
        # Reduce confidence as ratio approaches limit
        if dist_ratio > max_dist_ratio * 0.7:
            # Scale down confidence as we approach the limit
            confidence_adj *= 1.0 - (dist_ratio - max_dist_ratio * 0.7) / (max_dist_ratio * 0.3)
    
    # 2. Check if lateral prediction is reasonable (below max_lateral_ratio)
    filtered_lateral = abs(filtered_position[1])
    predicted_lateral = abs(predicted_position[1])
    
    if filtered_lateral > 0.05:  # Only apply ratio check if filtered lateral is significant
        lateral_ratio = predicted_lateral / filtered_lateral
        
        if lateral_ratio > max_lateral_ratio:
            return False, 0.0, 2  # Lateral too large
        
        # Reduce confidence as ratio approaches limit
        if lateral_ratio > max_lateral_ratio * 0.7:
            # Scale down confidence as we approach the limit
            confidence_adj *= 1.0 - (lateral_ratio - max_lateral_ratio * 0.7) / (max_lateral_ratio * 0.3)
    
    # 3. Check if angular prediction is reasonable
    # Assuming angles are in radians, limit to +/- π/2
    if abs(predicted_position[2]) > 1.57:  # π/2 ≈ 1.57
        return False, 0.0, 3  # Angle unrealistic
    
    # 4. Check if predicted direction is physically possible
    # This is a basic check that prevents impossible trajectories
    # More complex checks could be added if needed
    
    # If all checks pass, prediction is valid
    return True, confidence_adj, 0


@numba.jit(nopython=True, fastmath=True)
def detect_statistical_outlier_jit(history_array, history_size, new_prediction, z_threshold):
    """
    JIT-compiled statistical outlier detection.
    
    Args:
        history_array: Array of historical predictions (n x 3)
        history_size: Number of valid entries in history_array
        new_prediction: New prediction to check (3-element array)
        z_threshold: Z-score threshold for outlier detection
        
    Returns:
        tuple: (is_outlier, confidence_adjustment, z_score)
    """
    if history_size < 3:
        return False, 1.0, 0.0
    
    # Calculate mean and standard deviation for distance
    mean_distance = 0.0
    for i in range(history_size):
        mean_distance += history_array[i, 0]
    
    mean_distance /= history_size
    
    # Calculate variance
    variance = 0.0
    for i in range(history_size):
        diff = history_array[i, 0] - mean_distance
        variance += diff * diff
    
    variance /= history_size
    std_dev = max(0.001, np.sqrt(variance))  # Avoid division by zero
    
    # Calculate z-score
    z_score = abs(new_prediction[0] - mean_distance) / std_dev
    
    if z_score > z_threshold:
        # Calculate confidence adjustment based on how far beyond threshold
        confidence_adj = max(0.1, 1.0 - (z_score - z_threshold) / z_threshold)
        return True, confidence_adj, z_score
    
    return False, 1.0, z_score

@numba.jit(nopython=True, fastmath=True)
def calculate_velocity_and_acceleration(pos_buffer, dt):
    """
    Calculate velocity and acceleration from position buffer.
    
    Args:
        pos_buffer: Array of positions (x, y, angle, timestamp)
        dt: Time delta
        
    Returns:
        tuple: (velocity, acceleration) as 3-element arrays
    """
    buffer_size = len(pos_buffer)
    
    # Initialize with zeros
    velocity = np.zeros(3, dtype=np.float32)
    acceleration = np.zeros(3, dtype=np.float32)
    
    if buffer_size < 2:
        return velocity, acceleration
    
    # Get the two most recent positions
    curr_pos = pos_buffer[buffer_size - 1]
    prev_pos = pos_buffer[buffer_size - 2]
    
    # Calculate velocity
    velocity[0] = (curr_pos[0] - prev_pos[0]) / dt
    velocity[1] = (curr_pos[1] - prev_pos[1]) / dt
    velocity[2] = (curr_pos[2] - prev_pos[2]) / dt
    
    # Calculate acceleration if we have at least 3 points
    if buffer_size >= 3:
        # Get previous velocity
        prev_prev_pos = pos_buffer[buffer_size - 3]
        prev_dt = prev_pos[3] - prev_prev_pos[3]
        prev_dt = max(0.001, prev_dt)  # Avoid division by zero
        
        prev_vel_x = (prev_pos[0] - prev_prev_pos[0]) / prev_dt
        prev_vel_y = (prev_pos[1] - prev_prev_pos[1]) / prev_dt
        prev_vel_angle = (prev_pos[2] - prev_prev_pos[2]) / prev_dt
        
        # Calculate acceleration
        acceleration[0] = (velocity[0] - prev_vel_x) / dt
        acceleration[1] = (velocity[1] - prev_vel_y) / dt
        acceleration[2] = (velocity[2] - prev_vel_angle) / dt
    
    return velocity, acceleration

class EnhancedTargetFilter:
    """Enhanced filter for target position data with improved prediction and stability."""
    
    def __init__(self, buffer_size=10, prediction_horizon=0.2):
        """Initialize with more conservative settings and additional validation parameters."""
        self.logger = logging.getLogger('pid_controller.target_filter') 
        self.buffer_size = buffer_size
        self.prediction_horizon = prediction_horizon  # Shorter horizon for more conservative prediction
        self.position_buffer = deque(maxlen=buffer_size)
        self.last_update_time = None
        self.current_velocity = np.zeros(3, dtype=np.float32)  # x, y, angular
        self.filtered_position = None
        self.predicted_position = None
        self.acceleration = np.zeros(3, dtype=np.float32)  # x, y, angular acceleration
        self.is_moving = False
        self.direction_change_detected = False
        self.motion_direction = np.zeros(3, dtype=np.float32)  # normalized direction vector
        self.trajectory_history = deque(maxlen=10)
        
        # Enhanced parameters for validation
        self.max_prediction_distance_ratio = 1.5  # Reduced from 2.0
        self.max_prediction_lateral_ratio = 2.0   # Reduced from 3.0
        self.outlier_detection_window = 5         # Size of window to detect outliers
        self.prediction_history = deque(maxlen=5) # Store recent predictions for validation
        self.recovery_counter = 0                 # Counter for recovery from bad predictions
        self.prediction_disabled_until = 0        # Timestamp when prediction can be re-enabled
        
        # More conservative consistency threshold
        self.consistency_threshold = 0.7  # Reduced from 0.8 for easier prediction usage
        self.movement_consistency = 0.0  # 0.0-1.0 measure of consistent movement
        
        # More conservative velocity threshold
        self.min_velocity_for_prediction = 0.15  # Reduced from 0.2 m/s
        
        # Add confidence tracking for predictions
        self.prediction_confidence = 0.0  # 0.0-1.0 confidence in predictions
        self.prev_prediction_confidence = 0.0  # For temporal smoothing
        
        # Lower weights for filtered position calculation (more conservative)
        # Give more weight to recent positions (0.5) and less to older ones
        self.position_weights = np.array([0.2, 0.3, 0.5], dtype=np.float32)
        
        # For internal calculations
        self._trajectory_positions = np.zeros((10, 3), dtype=np.float32)
        self._trajectory_times = np.zeros(10, dtype=np.float32)
        self._recent_positions = np.zeros((3, 3), dtype=np.float32)
        
        # For prediction validation
        self._validation_errors = deque(maxlen=10)  # Store recent validation errors
        self._prediction_errors = deque(maxlen=10)  # Store prediction vs actual errors
        
        # New stability tracking
        self.position_stability = 1.0
        self.last_significant_change_time = 0
        self.position_stability_counter = 0
        self.position_history_consistency = 0.8
        
        # Adaptive exponential smoothing parameters
        self.exp_smoothing_alpha = 0.6  # Base smoothing factor
        self.min_smoothing_alpha = 0.3  # More smoothing for slow movements
        self.max_smoothing_alpha = 0.8  # Less smoothing for fast movements
        
        # Initialize the meta-stability tracking
        self.short_term_stability = 1.0
        self.medium_term_stability = 1.0
        self.target_stability = 0.8  # Start with middle value
        
        # Debounce parameters for resets
        self.min_reset_interval = 0.5  # Minimum time between resets in seconds
        self.last_reset_time = 0.0
    
    def _warmup_jit(self):
        """Warm up JIT compilation for EnhancedTargetFilter with improved logging."""
        try:
            start_time = time.time()
            self.logger.info("Starting JIT warmup for EnhancedTargetFilter")
            
            # Create dummy data for JIT warmup (implementation details omitted for brevity)
            # ...
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"JIT compilation warmup completed for EnhancedTargetFilter in {elapsed_time*1000:.1f}ms")
        except Exception as e:
            self.logger.warning(f"JIT warmup failed: {str(e)}. Will use non-optimized fallbacks.")

    def update(self, position, timestamp=None):
        """
        Update the filter with a new position measurement with improved smoothing and validation.
        
        Args:
            position: Tuple of (x, y, angle) for the target position
            timestamp: Time of measurement (defaults to current time)
        
        Returns:
            tuple: Filtered position
        """
        try:
            current_time = timestamp if timestamp is not None else time.time()
            
            # Store original position for comparisons
            original_position = position
            
            # Add to buffer with timestamp
            self.position_buffer.append((position[0], position[1], position[2], current_time))
            
            # Add to trajectory history
            self.trajectory_history.append((position, current_time))
            
            # Apply improved exponential smoothing for filtered position
            filtered_position = self._apply_exponential_smoothing(position, current_time)
            
            # Update meta-stability tracking
            self._update_stability_metrics()
            
            # Calculate velocity and acceleration
            self._update_motion_metrics(current_time)
            
            # Make better prediction for future position with enhanced validation
            if self.filtered_position is not None:
                # Check if prediction is still disabled from a previous reset
                if hasattr(self, 'prediction_disabled_until') and current_time < self.prediction_disabled_until:
                    # Skip prediction during recovery period
                    self.predicted_position = self.filtered_position
                    self.prediction_confidence = 0.0
                    self.using_prediction = False
                    return self.filtered_position
                
                # Only attempt prediction if conditions are good
                if self._should_use_prediction():
                    self._calculate_prediction(current_time)
                else:
                    # Not using prediction
                    self.predicted_position = self.filtered_position
                    self.prediction_confidence = 0.0
                    self.using_prediction = False
            else:
                # No filtered position yet, use raw position
                self.predicted_position = position
                self.prediction_confidence = 0.0
                self.using_prediction = False
            
            # Store for next iteration
            self.prev_using_prediction = getattr(self, 'using_prediction', False)
            self.last_update_time = current_time
            
            return self.filtered_position
            
        except Exception as e:
            self.logger.error(f"Error in EnhancedTargetFilter.update: {str(e)}")
            # Fallback to simple pass-through
            self.filtered_position = position
            self.predicted_position = position
            self.last_update_time = timestamp if timestamp is not None else time.time()
            return position
    
    def _apply_exponential_smoothing(self, position, current_time):
        """
        Apply adaptive exponential smoothing to position.
        
        Args:
            position: Raw position (distance, lateral, angle)
            current_time: Current time
            
        Returns:
            tuple: Smoothed position
        """
        # Initialize filtered position if not yet set
        if self.filtered_position is None:
            self.filtered_position = position
            return position
        
        # Adapt smoothing factor based on velocity magnitude
        vel_magnitude = 0.0
        if hasattr(self, 'current_velocity'):
            vel_magnitude = math.sqrt(
                self.current_velocity[0]**2 + 
                self.current_velocity[1]**2
            )
        
        # More smoothing (lower alpha) when slower, less when faster
        alpha = self.min_smoothing_alpha + (
            (self.max_smoothing_alpha - self.min_smoothing_alpha) * 
            min(1.0, vel_magnitude / 0.5)
        )
        
        # Also factor in stability - more smoothing when less stable
        if hasattr(self, 'target_stability'):
            # Adjust alpha based on stability (lower alpha for less stable targets)
            stability_factor = self.target_stability  # 0.0-1.0
            alpha = alpha * stability_factor + self.min_smoothing_alpha * (1.0 - stability_factor)
        
        # Apply exponential smoothing
        smoothed_position = (
            alpha * position[0] + (1 - alpha) * self.filtered_position[0],
            alpha * position[1] + (1 - alpha) * self.filtered_position[1],
            alpha * position[2] + (1 - alpha) * self.filtered_position[2]
        )
        
        # Update filtered position
        self.filtered_position = smoothed_position
        
        # Log significant changes
        if hasattr(self, 'last_logged_position') and self.last_logged_position is not None:
            filter_diff = abs(self.filtered_position[0] - self.last_logged_position[0]) + \
                        abs(self.filtered_position[1] - self.last_logged_position[1])
            if filter_diff > 0.05:  # Only log when filter changes significantly
                self.logger.info(f"Significant filter change detected | raw_dist={position[0]:.3f}, filtered_dist={self.filtered_position[0]:.3f}")
                self.last_logged_position = self.filtered_position
        else:
            self.last_logged_position = self.filtered_position
            
        return smoothed_position
    
    def _update_motion_metrics(self, current_time):
        """
        Update velocity, acceleration, and movement metrics.
        
        Args:
            current_time: Current time
        """
        # Calculate velocity and acceleration if we have enough data
        if len(self.position_buffer) >= 2 and self.last_update_time is not None:
            dt = current_time - self.last_update_time
            dt = max(0.001, dt)  # Avoid division by zero
            
            # Convert buffer to numpy array for Numba
            buffer_array = np.zeros((len(self.position_buffer), 4), dtype=np.float32)
            for i, pos in enumerate(self.position_buffer):
                buffer_array[i] = np.array(pos, dtype=np.float32)
            
            # Use Numba function to calculate velocity and acceleration
            raw_velocity, raw_acceleration = calculate_velocity_and_acceleration(buffer_array, dt)
            
            # Check for direction changes
            prev_vel = np.array(self.current_velocity, dtype=np.float32)
            prev_direction_change = self.direction_change_detected
            self.direction_change_detected = detect_direction_change(prev_vel, raw_velocity)
            
            # Log only when direction change status changes
            if self.direction_change_detected != prev_direction_change:
                if self.direction_change_detected:
                    self.logger.info(f"Direction change detected | prev_x={prev_vel[0]:.4f}, prev_y={prev_vel[1]:.4f}, new_x={raw_velocity[0]:.4f}, new_y={raw_velocity[1]:.4f}")
            
            # Use Numba function to smooth velocity and acceleration
            smoothed_velocity, smoothed_acceleration = update_smoothed_values(
                raw_velocity, raw_acceleration, prev_vel, self.acceleration)
            
            # Update state variables
            self.current_velocity = smoothed_velocity
            self.acceleration = smoothed_acceleration
            
            # Calculate normalized direction vector
            self.motion_direction = calculate_motion_direction(
                smoothed_velocity, self.motion_direction)
            
            # Determine if target is consistently moving with more conservative threshold
            vel_magnitude = math.sqrt(smoothed_velocity[0]**2 + smoothed_velocity[1]**2)
            vel_threshold = 0.05  # m/s
            prev_is_moving = self.is_moving
            self.is_moving = vel_magnitude > vel_threshold
            
            # Log only when movement status changes
            if self.is_moving != prev_is_moving:
                self.logger.info(f"Target {'started moving' if self.is_moving else 'stopped moving'} | velocity={vel_magnitude:.3f}, threshold={vel_threshold:.3f}")
            
            # Calculate movement consistency using Numba
            if len(self.trajectory_history) >= 5:
                # Extract position and time arrays for Numba
                for i, (pos, t) in enumerate(self.trajectory_history):
                    if i < 10:  # Max trajectory history size
                        self._trajectory_positions[i] = np.array(pos, dtype=np.float32)
                        self._trajectory_times[i] = t
                
                # Calculate consistency using Numba
                history_size = min(len(self.trajectory_history), 10)
                prev_consistency = self.movement_consistency
                self.movement_consistency = calculate_movement_consistency(
                    self._trajectory_positions[:history_size], 
                    self._trajectory_times[:history_size])
                
                # Log only significant consistency changes
                if abs(self.movement_consistency - prev_consistency) > 0.2:
                    self.logger.info(f"Movement consistency changed significantly | new_consistency={self.movement_consistency:.3f}, prev_consistency={prev_consistency:.3f}")
    
    def _update_stability_metrics(self):
        """
        Update the target stability metrics at multiple timescales.
        """
        if len(self.position_buffer) < 5:
            return
            
        # Calculate short-term stability (last 5 positions)
        recent = list(self.position_buffer)[-5:]
        short_term_variance = sum(
            ((p[0] - recent[0][0])**2 + (p[1] - recent[0][1])**2) 
            for p in recent[1:]
        ) / len(recent)
        
        self.short_term_stability = 1.0 / (1.0 + short_term_variance * 10.0)
        
        # Medium-term stability (all positions if available)
        if len(self.position_buffer) >= 10:
            medium = list(self.position_buffer)
            medium_term_variance = sum(
                ((p[0] - medium[0][0])**2 + (p[1] - medium[0][1])**2) 
                for p in medium[1:]
            ) / len(medium)
            
            self.medium_term_stability = 1.0 / (1.0 + medium_term_variance * 5.0)
        else:
            self.medium_term_stability = self.short_term_stability
            
        # Combine scores with weighting
        prev_stability = self.target_stability
        
        # Apply temporal smoothing to stability
        new_stability = self.short_term_stability * 0.6 + self.medium_term_stability * 0.4
        self.target_stability = prev_stability * 0.7 + new_stability * 0.3
        
        # Log significant stability changes
        if abs(self.target_stability - prev_stability) > 0.2:
            self.logger.info(f"Target stability changed significantly | new={self.target_stability:.2f}, "
                            f"prev={prev_stability:.2f}, short_term={self.short_term_stability:.2f}, "
                            f"medium_term={self.medium_term_stability:.2f}")
    
    def _should_use_prediction(self):
        """
        Determine if prediction should be used based on current conditions.
        
        Returns:
            bool: True if prediction should be used, False otherwise
        """
        # Calculate velocity magnitude
        vel_magnitude = math.sqrt(
            self.current_velocity[0]**2 + 
            self.current_velocity[1]**2
        )
        
        # Check criteria for prediction
        meets_velocity_criteria = vel_magnitude > self.min_velocity_for_prediction
        meets_consistency_criteria = self.movement_consistency > self.consistency_threshold
        is_stable_target = self.target_stability > 0.6
        no_direction_change = not self.direction_change_detected
        
        # Combine criteria
        should_predict = (
            meets_velocity_criteria and 
            meets_consistency_criteria and
            is_stable_target and
            no_direction_change
        )
        
        # Log prediction decision at debug level
        if hasattr(self, 'debug_level') and getattr(self, 'debug_level', 0) >= 2:
            if not should_predict:
                reasons = []
                if not meets_velocity_criteria:
                    reasons.append(f"velocity {vel_magnitude:.3f} < {self.min_velocity_for_prediction:.3f}")
                if not meets_consistency_criteria:
                    reasons.append(f"consistency {self.movement_consistency:.3f} < {self.consistency_threshold:.3f}")
                if not is_stable_target:
                    reasons.append(f"stability {self.target_stability:.3f} < 0.6")
                if not no_direction_change:
                    reasons.append("direction change detected")
                
                self.logger.debug(f"Skipping prediction: {', '.join(reasons)}")
        
        return should_predict
    
    def _calculate_prediction(self, current_time):
        """
        Calculate future position prediction with enhanced validation.
        
        Args:
            current_time: Current time
        """
        # Calculate base prediction confidence
        vel_magnitude = math.sqrt(
            self.current_velocity[0]**2 + 
            self.current_velocity[1]**2
        )
        
        base_confidence = self._calculate_prediction_confidence(
            vel_magnitude, 
            self.movement_consistency,
            self.filtered_position,
            self.filtered_position  # Using filtered as placeholder for now
        )
        
        # Only attempt prediction if base confidence is reasonable
        if base_confidence > 0.3:
            # Calculate effective horizon based on confidence and stability
            effective_horizon = (
                self.prediction_horizon * 
                self.movement_consistency * 
                self.target_stability
            )
            
            # Use physics-based prediction with JIT function
            current_pos = np.array(self.filtered_position, dtype=np.float32)
            pred_position = predict_future_position(
                current_pos,
                self.current_velocity,
                self.acceleration,
                effective_horizon,
                self.movement_consistency
            )
            
            # Validate the prediction
            is_valid, confidence_adj, reason = self._validate_prediction(
                position=self.position_buffer[-1][:3],  # most recent position
                filtered_position=self.filtered_position, 
                predicted_position=pred_position
            )
            
            # Check for statistical outliers
            is_outlier, outlier_conf_adj = self._detect_outliers(pred_position)
            
            if is_valid and not is_outlier:
                # Calculate final prediction confidence
                self.prediction_confidence = base_confidence * confidence_adj * outlier_conf_adj
                
                # Use prediction only if confidence is acceptable
                if self.prediction_confidence > 0.3:
                    self.predicted_position = pred_position
                    self.using_prediction = True
                    
                    # Add to prediction history for future validation
                    self.prediction_history.append(pred_position)
                    
                    # Reset recovery counter on successful prediction
                    self.recovery_counter = 0
                else:
                    # Low confidence prediction - fall back to filtered position
                    self.predicted_position = self.filtered_position
                    self.using_prediction = False
                    self.recovery_counter += 1
            else:
                # Invalid prediction - fall back to filtered position
                self.predicted_position = self.filtered_position
                self.prediction_confidence = 0.0
                self.using_prediction = False
                
                # Increment recovery counter
                self.recovery_counter += 1
                
                # If multiple consecutive invalid predictions, reset prediction state
                if self.recovery_counter > 2:
                    self._reset_prediction_state()
        else:
            # Base confidence too low - skip prediction 
            self.predicted_position = self.filtered_position
            self.prediction_confidence = 0.0
            self.using_prediction = False
    
    def _calculate_prediction_confidence(self, velocity_magnitude, consistency, filtered_position, predicted_position):
        """
        Enhanced confidence calculation with better stability metrics and temporal smoothing.
        
        Args:
            velocity_magnitude: Magnitude of current velocity
            consistency: Movement consistency
            filtered_position: Current filtered position
            predicted_position: Predicted position
            
        Returns:
            float: Confidence value (0.0-1.0)
        """
        # Use exponential weighting instead of linear scaling
        base_confidence = min(1.0, math.pow(consistency, 2) * velocity_magnitude / 0.5)
        
        # Add stability factor - less confidence for unstable targets
        if hasattr(self, 'target_stability'):
            # Square stability for more aggressive reduction of unstable targets
            stability_factor = math.pow(self.target_stability, 2)
            base_confidence *= stability_factor
        
        # More gradual confidence reduction for extreme predictions
        ratio = predicted_position[0] / (filtered_position[0] + 0.001)
        if ratio > 1.5 or ratio < 0.7:
            # Smoother confidence scaling using sigmoid function
            confidence_scale = 1.0 / (1.0 + math.exp(2 * (abs(ratio - 1.0) - 0.5)))
            base_confidence *= confidence_scale
        
        # Add temporal consistency component
        if hasattr(self, 'prev_prediction_confidence'):
            temporal_smoothing = 0.7  # Prefer temporal consistency
            base_confidence = (self.prev_prediction_confidence * temporal_smoothing + 
                              base_confidence * (1.0 - temporal_smoothing))
        
        # Store for next iteration
        self.prev_prediction_confidence = base_confidence
        
        return max(0.0, min(1.0, base_confidence))
    
    def _validate_prediction(self, position, filtered_position, predicted_position):
        """
        Validate predictions using multiple criteria.
        
        Args:
            position: Original position measurement
            filtered_position: Filtered position
            predicted_position: Position prediction to validate
            
        Returns:
            tuple: (is_valid, confidence_adjustment, reason)
        """
        try:
            # Convert to numpy arrays for JIT
            raw_arr = np.array(position, dtype=np.float32)
            filtered_arr = np.array(filtered_position, dtype=np.float32)
            predicted_arr = np.array(predicted_position, dtype=np.float32)
            
            # Use JIT-compiled validation
            is_valid, conf_adj, reason_code = validate_prediction_jit(
                raw_arr, 
                filtered_arr, 
                predicted_arr,
                self.max_prediction_distance_ratio,
                self.max_prediction_lateral_ratio
            )
            
            # Convert reason code to string
            reason_map = {
                0: "prediction_valid",
                1: "prediction_distance_too_large",
                2: "prediction_lateral_too_large",
                3: "prediction_angle_unrealistic",
                4: "prediction_direction_impossible"
            }
            reason = reason_map.get(reason_code, "prediction_invalid_unknown")
            
            # Additional validation for sudden jumps
            if is_valid and self.prediction_history:
                last_prediction = self.prediction_history[-1]
                jump_ratio = abs(predicted_position[0] - last_prediction[0]) / (abs(last_prediction[0]) + 0.001)
                
                if jump_ratio > 3.0:  # Reduced from 5.0 to 3.0 for stricter validation
                    # If new prediction is 3x different from last one
                    self._validation_errors.append("prediction_sudden_jump")
                    return False, 0.0, "prediction_sudden_jump"
            
            # Store validation result for monitoring
            if not is_valid:
                self._validation_errors.append(reason)
            
            return is_valid, conf_adj, reason
            
        except Exception as e:
            self.logger.warning(f"Error in validation: {str(e)}, falling back to safety checks")
            
            # Fallback to basic checks without JIT
            # 1. Check distance ratio
            distance_ratio = abs(predicted_position[0]) / (abs(filtered_position[0]) + 0.001)
            if distance_ratio > self.max_prediction_distance_ratio:
                return False, 0.0, "prediction_distance_too_large"
            
            # 2. Check lateral ratio
            lateral_ratio = abs(predicted_position[1]) / (abs(filtered_position[1]) + 0.001)
            if lateral_ratio > self.max_prediction_lateral_ratio:
                return False, 0.0, "prediction_lateral_too_large"
                
            return True, 1.0, "prediction_valid_fallback"
    
    def _detect_outliers(self, new_prediction):
        """
        Detect statistical outliers in predictions.
        
        Args:
            new_prediction: New prediction to check
            
        Returns:
            tuple: (is_outlier, confidence_adjustment)
        """
        if len(self.prediction_history) < 3:
            return False, 1.0
            
        try:
            # Convert prediction history to numpy array for JIT
            history_arr = np.zeros((len(self.prediction_history), 3), dtype=np.float32)
            for i, pred in enumerate(self.prediction_history):
                history_arr[i] = np.array(pred, dtype=np.float32)
            
            # Use JIT-compiled function for outlier detection    
            is_outlier, conf_adj, z_score = detect_statistical_outlier_jit(
                history_arr, 
                len(self.prediction_history),
                np.array(new_prediction, dtype=np.float32),
                2.5  # Z-score threshold reduced from 3.0 to 2.5 for stricter validation
            )
            
            if is_outlier:
                self._validation_errors.append(f"statistical_outlier_z{z_score:.1f}")
                
            return is_outlier, conf_adj
            
        except Exception as e:
            self.logger.warning(f"Error in outlier detection: {str(e)}, falling back to basic checks")
            
            # Fallback to simpler detection
            # Get recent predictions
            recent_distances = [p[0] for p in self.prediction_history]
            
            # Calculate mean and standard deviation
            mean_distance = sum(recent_distances) / len(recent_distances)
            variance = sum((d - mean_distance)**2 for d in recent_distances) / len(recent_distances)
            std_dev = max(0.001, math.sqrt(variance))  # Avoid division by zero
            
            # Check if new prediction is an outlier (> 2.5 standard deviations)
            z_score = abs(new_prediction[0] - mean_distance) / std_dev
            
            if z_score > 2.5:  # Reduced from 3.0 to 2.5
                return True, 0.2  # It's an outlier, reduce confidence to 20%
                
            return False, 1.0
    
    def _reset_prediction_state(self):
        """Reset prediction state after bad predictions."""
        # Don't reset too frequently
        current_time = time.time()
        if current_time - self.last_reset_time < self.min_reset_interval:
            return
            
        self.last_reset_time = current_time
        
        # Clear prediction history
        self.prediction_history.clear()
        
        # Reset confidence and state
        self.prediction_confidence = 0.0
        self.using_prediction = False
        
        # Disable prediction for a cooling-off period
        self.prediction_disabled_until = current_time + 0.5  # Disable for 0.5 seconds
        
        # Reset recovery counter
        self.recovery_counter = 0
        
        # Log the reset
        errors = list(self._validation_errors) if hasattr(self, '_validation_errors') else []
        self.logger.info(f"Prediction system reset due to validation failures | errors={errors}, disabled_until=0.5s")
        
        # Clear validation errors
        if hasattr(self, '_validation_errors'):
            self._validation_errors.clear()
    
    def validate_position_change(self, new_position, current_time):
        """
        Validate if a position change is significant and stable enough to trigger reacquisition.
        
        Args:
            new_position: New position (distance, lateral, angle)
            current_time: Current time
            
        Returns:
            bool: True if the change should trigger reacquisition, False otherwise
        """
        # Skip validation if no previous position
        if self.filtered_position is None:
            return True
            
        # Check how long since last significant change
        if not hasattr(self, 'last_significant_change_time'):
            self.last_significant_change_time = current_time
            self.position_stability_counter = 0
            
        # Calculate change magnitude
        distance_change = abs(new_position[0] - self.filtered_position[0])
        lateral_change = abs(new_position[1] - self.filtered_position[1])
        total_change = distance_change + lateral_change
        
        # If change is small, no need for reacquisition
        if total_change < 0.1:  # Small change threshold
            return False
            
        # For significant changes, require stability
        time_since_change = current_time - self.last_significant_change_time
        if time_since_change < 0.3:  # Require 300ms of stability
            # Reset stability counter
            self.position_stability_counter = 0
            self.last_significant_change_time = current_time
            return False
            
        # Increment stability counter
        self.position_stability_counter += 1
        
        # Only reacquire after position has been stable for a few cycles
        return self.position_stability_counter >= 3
    
    def is_direction_change_significant(self, prev_direction, new_direction, consistency):
        """
        Determine if a direction change is significant enough to trigger recovery.
        
        Args:
            prev_direction: Previous movement direction
            new_direction: New movement direction
            consistency: Movement consistency
            
        Returns:
            bool: True if the direction change is significant
        """
        if prev_direction is None or new_direction is None:
            return False
        
        # Calculate dot product to check direction change
        dot_product = (prev_direction[0] * new_direction[0] + 
                      prev_direction[1] * new_direction[1])
        
        # Normalize vectors
        prev_mag = math.sqrt(prev_direction[0]**2 + prev_direction[1]**2)
        new_mag = math.sqrt(new_direction[0]**2 + new_direction[1]**2)
        
        if prev_mag < 0.05 or new_mag < 0.05:
            return False  # Too small to be significant
        
        normalized_dot = dot_product / (prev_mag * new_mag)
        
        # Calculate angle between vectors
        angle_change = math.acos(max(-1.0, min(1.0, normalized_dot)))
        angle_degrees = math.degrees(angle_change)
        
        # Only consider significant if:
        # 1. Angle change is large (> 45 degrees)
        # 2. Movement is consistent enough to be meaningful
        # 3. Velocity magnitude is sufficient
        is_significant = (angle_degrees > 45 and 
                         consistency > 0.5 and
                         max(prev_mag, new_mag) > 0.1)
        
        return is_significant
    
    def get_filtered_position(self):
        """Get the current filtered position."""
        if self.filtered_position:
            return self.filtered_position
        elif self.position_buffer:
            return (self.position_buffer[-1][0], self.position_buffer[-1][1], self.position_buffer[-1][2])
        else:
            return None
    
    def get_predicted_position(self):
        """Get the predicted position with final validation check."""
        # Add one final sanity check before returning
        if self.predicted_position and self.filtered_position:
            # Calculate ratio between prediction and filtered position
            ratio = self.predicted_position[0] / (self.filtered_position[0] + 0.001)
            
            # If prediction is unreasonably different from filtered, return filtered
            if ratio > 2.0 or ratio < 0.5:  # More strict limits (was 3.0/0.3)
                self.logger.info(f"Final sanity check rejected prediction | "
                                f"pred_dist={self.predicted_position[0]:.3f}, "
                                f"filtered_dist={self.filtered_position[0]:.3f}, ratio={ratio:.2f}")
                return self.filtered_position
        
        return self.predicted_position if hasattr(self, 'predicted_position') else self.filtered_position
    
    def get_velocity(self):
        """Get the current velocity estimate."""
        return tuple(self.current_velocity)
    
    def get_acceleration(self):
        """Get the current acceleration estimate."""
        return tuple(self.acceleration)
    
    def get_movement_info(self):
        """Get information about the movement characteristics."""
        # Calculate velocity magnitude
        vel_magnitude = math.sqrt(self.current_velocity[0]**2 + self.current_velocity[1]**2)
        
        return {
            'is_moving': self.is_moving,
            'direction_change': self.direction_change_detected,
            'consistency': self.movement_consistency,
            'velocity_magnitude': vel_magnitude,
            'confidence': self.prediction_confidence,
            'stability': getattr(self, 'target_stability', 0.5)
        }
    
    def get_prediction_info(self):
        """Get information about the prediction quality."""
        return {
            'confidence': self.prediction_confidence,
            'horizon': getattr(self, 'prediction_horizon', 0.0) * self.prediction_confidence,
            'consistency': getattr(self, 'movement_consistency', 0.0),
            'is_moving': getattr(self, 'is_moving', False),
            'velocity_threshold': getattr(self, 'min_velocity_for_prediction', 0.0),
            'consistency_threshold': getattr(self, 'consistency_threshold', 0.0),
            'recovery_counter': getattr(self, 'recovery_counter', 0),
            'validation_errors': list(self._validation_errors) if hasattr(self, '_validation_errors') else [],
            'stability': getattr(self, 'target_stability', 0.5)
        }
    
    def adaptive_reset(self, reason=""):
        """
        Perform adaptive reset based on context rather than full reset.
        
        Args:
            reason: Reason for reset
        """
        # Don't reset too frequently
        current_time = time.time()
        if current_time - self.last_reset_time < self.min_reset_interval:
            self.logger.info(f"Skipping reset - too soon since last reset "
                           f"({current_time - self.last_reset_time:.2f}s < {self.min_reset_interval:.2f}s)")
            return
            
        self.last_reset_time = current_time
        
        if reason == "direction_change":
            # Only reset velocity and acceleration, keep position
            self.current_velocity = np.zeros(3, dtype=np.float32)
            self.acceleration = np.zeros(3, dtype=np.float32)
            self.direction_change_detected = True
            self.movement_consistency = 0.0
            
            # Keep position history but mark as unreliable for prediction
            self.prediction_confidence = 0.0
            self.prediction_disabled_until = current_time + 0.5
            
            self.logger.info(f"Partial filter reset due to {reason} | "
                           f"kept_position=True, reset_velocity=True, reset_acceleration=True")
            
        elif reason == "state_change":
            # Preserve some position information with high decay
            if self.filtered_position is not None:
                old_position = self.filtered_position
                # Complete reset of buffer
                self.position_buffer.clear()
                # But seed with old position and high uncertainty
                self.position_buffer.append((old_position[0], old_position[1], old_position[2], current_time))
                
                self.logger.info(f"Semi-preserved position in filter reset due to {reason} | seed_position=True")
            else:
                # Full reset if no position
                self._full_reset()
        else:
            # Default to full reset for other cases
            self._full_reset()
    
    def _full_reset(self):
        """Full reset of all filter state."""
        self.position_buffer.clear()
        self.trajectory_history.clear()
        self.last_update_time = None
        self.current_velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.filtered_position = None
        self.predicted_position = None
        self.is_moving = False
        self.direction_change_detected = False
        self.motion_direction = np.zeros(3, dtype=np.float32)
        self.movement_consistency = 0.0
        self.prediction_confidence = 0.0
        self.prev_prediction_confidence = 0.0
        
        # Reset prediction validation state
        self.prediction_history.clear()
        self.recovery_counter = 0
        self.prediction_disabled_until = 0
        
        # Reset validation errors
        if hasattr(self, '_validation_errors'):
            self._validation_errors.clear()
        
        # Reset prediction errors
        if hasattr(self, '_prediction_errors'):
            self._prediction_errors.clear()
        
        # Reset stability tracking
        self.position_stability = 0.5
        self.target_stability = 0.5
        self.short_term_stability = 0.5
        self.medium_term_stability = 0.5
        
        # Log the reset
        self.logger.info("Target filter completely reset")
    
    def reset(self):
        """Reset the filter state using the full reset method."""
        self._full_reset()

class ErrorTracker:
    """Lightweight error tracker that monitors error values over time."""
    
    def __init__(self, name, max_history=8):
        """Initialize error tracker with efficient storage."""
        self.name = name
        self.current_error = 0.0
        self.previous_error = 0.0
        self.previous_category = None  # For tracking error category with hysteresis
        # Pre-allocate error history with zeros to avoid frequent allocations
        self.error_history = deque([0.0] * max_history, maxlen=max_history)
        self.last_correction_time = 0.0
        self.sign_changes = 0  # Count of error sign changes (useful for oscillation detection)
        self.accumulated_error = 0.0
        self.decay_factor = 0.9  # Simplified decay factor
        self.error_increasing = False
        self.peak_error = 0.0
        self.last_sign = 0  # -1, 0, or 1
        
    def update(self, error, dt):
        """Update error tracking with new error value."""
        # Store previous error and sign
        self.previous_error = self.current_error
        prev_sign = 1 if self.current_error > 0 else (-1 if self.current_error < 0 else 0)
        
        # Update current error
        self.current_error = error
        current_sign = 1 if error > 0 else (-1 if error < 0 else 0)
        
        # Check for sign change
        if prev_sign != 0 and current_sign != 0 and prev_sign != current_sign:
            self.sign_changes += 1
            
        # Update error history
        self.error_history.append(error)
        
        # Track if error is increasing
        self.error_increasing = abs(error) > abs(self.previous_error) * 1.05  # 5% threshold
        
        # Update peak error if current error is larger
        if abs(error) > abs(self.peak_error):
            self.peak_error = error
        
        # Apply different decay rates based on error direction
        if current_sign == prev_sign:
            # Same direction - standard decay
            decay = self.decay_factor
        else:
            # Direction change - faster decay
            decay = self.decay_factor * 0.5
            
        # Update accumulated error with direction-aware decay
        self.accumulated_error = (self.accumulated_error + error * dt) * decay
        
        # Store last sign
        self.last_sign = current_sign
    
    def reset(self):
        """Reset all tracked errors."""
        self.current_error = 0.0
        self.previous_error = 0.0
        self.error_history.clear()
        # Refill with zeros
        for _ in range(self.error_history.maxlen):
            self.error_history.append(0.0)
        self.accumulated_error = 0.0
        self.sign_changes = 0
        self.error_increasing = False
        self.peak_error = 0.0
        self.last_sign = 0
        self.previous_category = None
    
    def is_error_growing(self):
        """Check if error is growing compared to previous value."""
        return self.error_increasing
        
    def record_correction(self):
        """Record that a correction was made for this error."""
        self.last_correction_time = time.time()
        # Reduce accumulated error when correction is made
        self.accumulated_error *= 0.5
        
    def get_trend(self, n=3):
        """Calculate trend of error (increasing/decreasing)."""
        if len(self.error_history) < n:
            return 0.0  # Not enough data
            
        # Get the last n values
        history = list(self.error_history)[-n:]
        
        # Simple linear regression to find trend
        x = list(range(len(history)))
        y = history
        n = len(x)
        
        if n < 2:
            return 0.0
            
        # Calculate slope with least squares
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi ** 2 for xi in x)
        
        # Slope formula: (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x^2)
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
            return slope
        except ZeroDivisionError:
            return 0.0
    
    def is_oscillating(self):
        """Determine if the error is oscillating."""
        # Consider oscillating if there are multiple sign changes recently
        return self.sign_changes >= 2


@numba.jit(nopython=True, fastmath=True)
def adjust_gains_jit(kp, ki, kd, base_kp, base_ki, base_kd, error, trend, 
                    zero_crossing_time, gain_adjust_rate, current_time, controller_type,
                    abs_error):
    """
    Adaptively adjust PID gains with enhanced gain scheduling.
    
    Args:
        kp, ki, kd: Current gains
        base_kp, base_ki, base_kd: Base gains
        error: Current error value
        trend: Error trend
        zero_crossing_time: Time of last zero crossing or None
        gain_adjust_rate: How quickly gains adjust
        current_time: Current time
        controller_type: 0 for Linear X, 1 for Linear Y, 2 for Angular
        abs_error: Absolute error value for gain scheduling
        
    Returns:
        tuple: (kp, ki, kd) adjusted gains
    """
    # Define target gains based on control phase (error magnitude)
    if abs_error > 0.5:  # Far from target
        # Use more aggressive P, reduced I, moderate D
        target_kp = base_kp * 1.2  # Higher P for faster response
        target_ki = base_ki * 0.5  # Reduced I to prevent overshoot
        target_kd = base_kd * 0.8  # Moderate D for stability
    elif abs_error > 0.2:  # Mid-range
        # Balanced gains
        target_kp = base_kp * 1.0
        target_ki = base_ki * 0.8
        target_kd = base_kd * 1.0
    else:  # Close to target
        # Reduced P, increased D for precision
        target_kp = base_kp * 0.8  # Softer P for less overshoot
        target_ki = base_ki * 1.0  # Full I for zero steady-state error
        target_kd = base_kd * 1.5  # Higher D for critically damped approach
    
    # Further adjust based on error trend
    if trend < -0.1:  # Error is decreasing
        # Approaching target, increase damping
        target_kp *= 0.9
        target_ki *= 0.8
        target_kd *= 1.2
    elif trend > 0.1:  # Error is increasing
        # Moving away from target, increase responsiveness
        target_kp *= 1.1
        target_ki *= 0.9
        target_kd *= 0.9
    
    # Special case for zero crossing
    if zero_crossing_time > 0.0:  # Time is provided, meaning we have a crossing
        time_since_crossing = current_time - zero_crossing_time
        if time_since_crossing < 0.5:  # Within 0.5 seconds of zero crossing
            # Enhance derivative, reduce integral during zero crossing
            target_kd *= 1.3
            target_ki *= 0.3
    
    # Controller-specific adjustments
    if controller_type == 1:  # Linear Y
        # More aggressive damping for lateral control
        target_kd *= 1.2
        # Less integral for lateral to prevent overshooting
        target_ki *= 0.7
    elif controller_type == 2:  # Angular
        # Reduced integral gain for angular control
        target_ki *= 0.6
        # Reduced derivative gain to prevent overshoot
        target_kd *= 0.6
    
    # Gradually adjust gains toward targets
    new_kp = kp * (1.0 - gain_adjust_rate) + target_kp * gain_adjust_rate
    new_ki = ki * (1.0 - gain_adjust_rate) + target_ki * gain_adjust_rate
    new_kd = kd * (1.0 - gain_adjust_rate) + target_kd * gain_adjust_rate
    
    return new_kp, new_ki, new_kd

def pid_compute_jit(error, prev_error, integral, dt, kp, ki, kd, 
                   output_min, output_max, name_idx, 
                   sign_change_count, prev_sign, zero_crossing_time, current_time,
                   integral_deadband=0.05, integral_decay=0.7, 
                   max_integral_multiplier=1.0, prev_error_change=0.0,
                   prev_output=0.0, integral_adjustment_factor=1.0):
    """
    Optimized PID computation function using Numba with enhanced anti-windup
    and integral management.
    
    Args:
        error: Current error value
        prev_error: Previous error value
        integral: Current integral value
        dt: Time since last update
        kp, ki, kd: Current PID gains
        output_min, output_max: Output limits
        name_idx: Controller type (0:Linear X, 1:Linear Y, 2:Angular)
        sign_change_count: Count of sign changes
        prev_sign: Previous error sign
        zero_crossing_time: Time of last zero crossing
        current_time: Current time
        integral_deadband: Deadband for integral term
        integral_decay: Decay factor for integral when in deadband
        max_integral_multiplier: Multiplier for integral limit
        prev_error_change: Previous error change for D-term smoothing
        prev_output: Previous output for saturation detection
        integral_adjustment_factor: Factor for dynamic integral adjustment
        
    Returns:
        tuple: (output, integral, p_term, i_term, d_term, sign_change_count, prev_sign, 
                zero_crossing_time, error_change)
    """
    # Starting values
    new_sign_change_count = sign_change_count
    new_prev_sign = prev_sign
    new_zero_crossing_time = zero_crossing_time
    new_integral = integral
    
    # Detect sign changes for improved zero-crossing behavior
    current_sign = 0
    if error > 0:
        current_sign = 1
    elif error < 0:
        current_sign = -1
    
    # Check for sign change (zero crossing)
    if prev_sign != 0 and current_sign != 0 and prev_sign != current_sign:
        # Sign changed - we crossed zero
        new_zero_crossing_time = current_time
        new_sign_change_count += 1
        
        # Reset integral based on controller type - ENHANCED
        if name_idx == 2:  # Angular
            new_integral = 0.0
        elif name_idx == 1:  # Linear Y
            new_integral = integral * 0.05  # More aggressive (was 0.1)
        else:  # Linear X
            new_integral = integral * 0.1  # More aggressive (was 0.2)
    else:
        # No sign change - gradually reduce sign change count for hysteresis
        new_sign_change_count = max(0, sign_change_count - 0.1)
    
    # Update previous sign
    if error != 0:
        new_prev_sign = current_sign
    
    # Calculate proportional term
    p_term = kp * error
    
    # Enhanced integral term handling with saturation awareness
    # Check if P term is already saturating output
    is_p_term_saturating = abs(p_term) >= output_max
    
    # Only calculate if integral term is used
    if ki > 0:
        if is_p_term_saturating:
            # If P term is already saturating output, don't accumulate integral
            # This prevents integral windup more effectively than back-calculation
            new_integral = integral * 0.9  # Gradually reduce existing integral
        elif abs(error) < integral_deadband:
            # ENHANCED: More aggressively reduce integral term when close to target
            # Different decay rates based on controller type
            if name_idx == 0:  # Linear X - most aggressive reset
                new_integral *= 0.4  # Much faster decay (was integral_decay)
            elif name_idx == 1:  # Linear Y 
                new_integral *= 0.5  # Faster decay
            else:  # Angular - least aggressive
                new_integral *= 0.6  # Still faster than original
        else:
            # Normal case - accumulate integral with direction awareness
            if error * prev_error < 0:
                # Error changed sign - reset integral more aggressively
                new_integral = error * dt * 0.5  # Start fresh with smaller value
            else:
                # Standard accumulation with adjustment factor
                new_integral += error * dt * integral_adjustment_factor
        
        # Calculate max integral limit based on controller type
        if name_idx == 2:  # Angular
            max_integral = (output_max - output_min) / ki * 0.7 * max_integral_multiplier
        else:
            max_integral = (output_max - output_min) / ki * max_integral_multiplier
        
        # Apply limits to integral
        if new_integral > max_integral:
            new_integral = max_integral
        elif new_integral < -max_integral:
            new_integral = -max_integral
        
        i_term = ki * new_integral
    else:
        i_term = 0.0
    
    # Enhanced derivative term with improved noise filtering
    error_change = error - prev_error
    
    # Calculate derivative term (with dt protection)
    if dt > 0.001:
        if name_idx == 2:  # Angular controller
            # For angular controller, use smoother derivative calculation
            # Calculate weighted average of current and previous change
            filtered_error_change = error_change * 0.5 + prev_error_change * 0.5  
            d_term = kd * filtered_error_change / dt
        else:
            # For position controllers, implement advanced derivative filtering
            if abs(error_change) > abs(prev_error_change) * 3:
                # Sudden large change - likely noise, use previous value
                d_term = kd * prev_error_change / dt
            else:
                # Normal case - apply smoothing
                d_term = kd * (error_change * 0.8 + prev_error_change * 0.2) / dt
    else:
        # Protect against very small dt
        d_term = kd * error_change / 0.001
    
    # Calculate raw output by summing all terms
    raw_output = p_term + i_term + d_term
    
    # Store original output before limiting
    pre_limit_output = raw_output
    
    # Apply output limits
    if raw_output > output_max:
        output = output_max
    elif raw_output < output_min:
        output = output_min
    else:
        output = raw_output
    
    # Enhanced anti-windup using back-calculation
    if ki > 0 and output != pre_limit_output:
        # Calculate how much was clipped
        clipped_amount = pre_limit_output - output
        
        # Reduce integral term proportionally based on controller type
        if name_idx == 0:  # Linear X - more aggressive
            back_calc_factor = 0.8
        elif name_idx == 1:  # Linear Y
            back_calc_factor = 0.7
        else:  # Angular
            back_calc_factor = 0.6
            
        # Apply back-calculation adjustment
        new_integral -= clipped_amount * back_calc_factor / ki
    
    # Return all necessary values for state tracking
    return output, new_integral, p_term, i_term, d_term, new_sign_change_count, new_prev_sign, new_zero_crossing_time, error_change


class ImprovedPID:
    """PID controller with enhanced integral handling and adaptive gains, optimized with Numba."""
    
    # Controller type enum for use with Numba
    LINEAR_X = 0
    LINEAR_Y = 1
    ANGULAR = 2
    
    def get_logger(self):
        return logging.getLogger('pid_controller.pid')

    def __init__(self, base_kp, base_ki, base_kd, output_min, output_max, name="PID"):
        """Initialize the improved PID controller."""
        
        self.logger = logging.getLogger('pid_controller.pid')

        # Base gains
        self.base_kp = base_kp
        self.base_ki = base_ki
        self.base_kd = base_kd
        
        # Current gains (will be adjusted adaptively)
        self.kp = base_kp
        self.ki = base_ki
        self.kd = base_kd
        
        # Output limits
        self.output_min = output_min
        self.output_max = output_max
        self.name = name
        
        # Convert name to numeric index for Numba
        if name == "Linear X":
            self.name_idx = self.LINEAR_X
        elif name == "Linear Y": 
            self.name_idx = self.LINEAR_Y
        elif name == "Angular":
            self.name_idx = self.ANGULAR
        else:
            self.name_idx = self.LINEAR_X  # Default
        
        # ADDED: Reference to parent controller
        self.pid_controller = None
        
        # PID state
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.last_output = 0.0
        
        # Diagnostic information
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        
        # Adaptation parameters
        self.gain_adjust_rate = 0.1  # How quickly gains adjust
        self.zero_crossing_time = None
        self.sign_change_count = 0
        self.prev_sign = 0
        
        # Enhanced parameters
        self.integral_deadband = 0.05
        self.integral_decay = 0.7
        self.max_integral = (output_max - output_min) / base_ki if base_ki > 0 else 1.0
        
        # Error tracker for trend analysis
        self.error_tracker = None
        
        # Performance metrics
        self.settling_time = 0.0
        self.overshoot = 0.0
        self.rise_time = 0.0
        self.steady_state = False
        
        # New enhanced parameters
        self.prev_error_change = 0.0  # For improved D-term calculation
        self.integral_adjustment_factor = 1.0  # Dynamic factor based on context
        self.last_movement_change_time = 0.0  # Track when movement direction changes
        self.state_history = deque(maxlen=10)  # For extended state monitoring
        
        # Logger for controller-specific logs
        self.logger = logging.getLogger(f'pid_controller.{name}')
        
        # Warm up JIT compilation by running the functions once
        self._warmup_jit()
    
    def _warmup_jit(self):
        """Warm up JIT compilation of Numba functions with improved logging."""
        try:
            start_time = time.time()
            self.logger.info(f"Starting JIT warmup for ImprovedPID ({self.name})")
            
            # Use dummy values to trigger compilation
            dummy_time = time.time()
            
            # Warm up adjust_gains_jit
            _, _, _ = adjust_gains_jit(1.0, 0.1, 0.1, 1.0, 0.1, 0.1, 0.5, 0.0, 
                                    0.0, 0.1, dummy_time, self.name_idx, 0.5)
            
            # Warm up pid_compute_jit with new parameters
            _, _, _, _, _, _, _, _, _ = pid_compute_jit(
                0.5, 0.0, 0.0, 0.1, 1.0, 0.1, 0.1, -1.0, 1.0, 
                self.name_idx, 0, 0, 0.0, dummy_time, 0.05, 0.7, 1.0, 0.0, 0.0, 1.0
            )
            
            elapsed_time = time.time() - start_time
            self.logger.info(f"JIT compilation warmup completed for ImprovedPID ({self.name}) in {elapsed_time*1000:.1f}ms")
            
        except Exception as e:
            self.logger.warning(f"JIT warmup failed for ImprovedPID ({self.name}): {str(e)}. Will use non-optimized version.")
        
    def adjust_gains(self, error, trend):
        """
        Adaptively adjust PID gains based on error magnitude and trend.
        This version uses enhanced gain scheduling by error magnitude.
        
        Args:
            error: Current error value
            trend: Error trend (positive for increasing, negative for decreasing)
        """
        # Convert None to numeric for Numba compatibility
        zero_time = 0.0
        if self.zero_crossing_time is not None:
            zero_time = self.zero_crossing_time
        
        # Get absolute error for gain scheduling
        abs_error = abs(error)
            
        try:
            # Call the JIT-compiled function with abs_error parameter
            new_kp, new_ki, new_kd = adjust_gains_jit(
                self.kp, self.ki, self.kd,
                self.base_kp, self.base_ki, self.base_kd,
                error, trend, zero_time, self.gain_adjust_rate,
                time.time(), self.name_idx, abs_error
            )
            
            # Update the gains
            self.kp = new_kp
            self.ki = new_ki
            self.kd = new_kd
            
        except Exception as e:
            # Fallback to non-JIT version if exception occurs
            if hasattr(self, 'logger'):
                self.logger.warning(f"JIT adjust_gains failed: {str(e)}. Using non-optimized version.")
            
            # Define target gains based on control phase (error magnitude)
            if abs_error > 0.5:  # Far from target
                # Use more aggressive P, reduced I, moderate D
                target_kp = self.base_kp * 1.2  # Higher P for faster response
                target_ki = self.base_ki * 0.5  # Reduced I to prevent overshoot
                target_kd = self.base_kd * 0.8  # Moderate D for stability
            elif abs_error > 0.2:  # Mid-range
                # Balanced gains
                target_kp = self.base_kp * 1.0
                target_ki = self.base_ki * 0.8
                target_kd = self.base_kd * 1.0
            else:  # Close to target
                # Reduced P, increased D for precision
                target_kp = self.base_kp * 0.8  # Softer P for less overshoot
                target_ki = self.base_ki * 1.0  # Full I for zero steady-state error
                target_kd = self.base_kd * 1.5  # Higher D for critically damped approach
            
            # Further adjust based on error trend
            if trend < -0.1:  # Error is decreasing
                # Approaching target, increase damping
                target_kp *= 0.9
                target_ki *= 0.8
                target_kd *= 1.2
            elif trend > 0.1:  # Error is increasing
                # Moving away from target, increase responsiveness
                target_kp *= 1.1
                target_ki *= 0.9
                target_kd *= 0.9
                
            # Special case for zero crossing
            if self.zero_crossing_time is not None:
                time_since_crossing = time.time() - self.zero_crossing_time
                if time_since_crossing < 0.5:  # Within 0.5 seconds of zero crossing
                    # Enhance derivative, reduce integral during zero crossing
                    target_kd *= 1.3
                    target_ki *= 0.3
            
            # Controller-specific adjustments
            if self.name == "Linear Y":
                # More aggressive damping for lateral control
                target_kd *= 1.2
                # Less integral for lateral to prevent overshooting
                target_ki *= 0.7
            elif self.name == "Angular":
                # Reduced integral gain for angular control
                target_ki *= 0.6
                # Adjusted derivative gain
                target_kd *= 0.6
            
            # Gradually adjust gains toward targets
            self.kp = self.kp * (1.0 - self.gain_adjust_rate) + target_kp * self.gain_adjust_rate
            self.ki = self.ki * (1.0 - self.gain_adjust_rate) + target_ki * self.gain_adjust_rate
            self.kd = self.kd * (1.0 - self.gain_adjust_rate) + target_kd * self.gain_adjust_rate
    
    def reset(self):
        """Reset controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.last_output = 0.0
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.sign_change_count = 0
        self.prev_sign = 0
        self.zero_crossing_time = None
        self.prev_error_change = 0.0
        
        # Reset new enhanced parameters
        self.integral_adjustment_factor = 1.0
        self.last_movement_change_time = time.time()
        
        # Reset gains to base values
        self.kp = self.base_kp
        self.ki = self.base_ki
        self.kd = self.base_kd
        
        # Reset error tracker if it exists
        if self.error_tracker is not None:
            self.error_tracker.reset()
        
        # Reset performance metrics
        self.settling_time = 0.0
        self.overshoot = 0.0
        self.rise_time = 0.0
        self.steady_state = False
        
        # Reset state history
        if hasattr(self, 'state_history'):
            self.state_history.clear()
        
        # Log reset
        self.logger.info(f"PID controller {self.name} reset")
    
    def compute(self, error, current_time=None, force_zero=False, error_trend=0.0):
        """
        Compute the control output based on the error with improved integral handling
        and error-aware derivative calculation.
        
        Args:
            error: Current error value
            current_time: Current time (defaults to now)
            force_zero: Whether to force zero output
            error_trend: Trend of error (from ErrorTracker)
                
        Returns:
            float: Calculated control output
        """
        try:
            # Throttled logging - only log every few calls for this controller
            should_log = hasattr(self, 'compute_count') and self.compute_count % 10 == 0
            
            # Track computation cycles
            if not hasattr(self, 'compute_count'):
                self.compute_count = 0
            self.compute_count += 1
            
            # Log initial PID state if debugging
            if should_log and hasattr(self, 'debug_level') and self.debug_level >= 2:
                self.logger.debug(f"PID {self.name} compute input: error={error:.3f}, trend={error_trend:.3f}")

            # If forcing zero output, bypass calculations
            if force_zero:
                if should_log:
                    self.logger.info(f"PID {self.name} forced to zero")
                    
                self.prev_error = error
                self.last_p_term = 0.0
                self.last_i_term = 0.0
                self.last_d_term = 0.0
                self.last_output = 0.0
                return 0.0
                
            # Use current time if not provided
            if current_time is None:
                current_time = time.time()
                
            # Initialize time on first call
            if self.last_time is None:
                self.last_time = current_time
                self.prev_error = error
                self.last_movement_change_time = current_time
                # P-only control on first iteration (no I or D)
                output = self.kp * error
                self.last_p_term = output
                self.last_output = max(self.output_min, min(self.output_max, output))
                
                # Log initial calculation
                if hasattr(self, 'debug_level') and self.debug_level >= 2:
                    self.logger.debug(f"PID {self.name} first compute: P-only={output:.3f}")
                    
                return float(self.last_output)
                
            # Calculate dt (time since last update)
            dt = current_time - self.last_time
            dt = max(0.001, min(0.1, dt))  # Bound dt to reasonable values
                
            # Adjust gains based on error and trend
            original_gains = (self.kp, self.ki, self.kd)
            self.adjust_gains(error, error_trend)
            adjusted_gains = (self.kp, self.ki, self.kd)
            
            # Log gain adjustments if significant changes occurred
            if hasattr(self, 'debug_level') and self.debug_level >= 2:
                if any(abs(a-b) > 0.01 for a, b in zip(original_gains, adjusted_gains)):
                    self.logger.debug(f"PID {self.name} gains adjusted: {original_gains} -> {adjusted_gains}")
            
            # Update dynamic integral adjustment factor based on context
            # Determine if error direction changed
            current_sign = 1 if error > 0 else (-1 if error < 0 else 0)
            prev_sign = 1 if self.prev_error > 0 else (-1 if self.prev_error < 0 else 0)
            direction_changed = (current_sign != 0 and prev_sign != 0 and current_sign != prev_sign)
            
            if direction_changed:
                # Error changed direction - record time and adjust factor
                self.last_movement_change_time = current_time
                self.integral_adjustment_factor = 0.2  # Greatly reduce integral effect
                
                # Log direction change
                if hasattr(self, 'debug_level') and self.debug_level >= 1:
                    self.logger.info(f"PID {self.name} direction change: integral_factor={self.integral_adjustment_factor:.2f}")
            else:
                # Gradually increase integral effect after direction change
                time_since_change = current_time - self.last_movement_change_time
                if time_since_change < 1.0:
                    # Recently changed direction - limit integral contribution
                    self.integral_adjustment_factor = 0.2 + 0.8 * time_since_change
                else:
                    # Normal operation
                    self.integral_adjustment_factor = 1.0
            
            # Prepare inputs for JIT function
            zero_time = 0.0
            if self.zero_crossing_time is not None:
                zero_time = self.zero_crossing_time
                
            # Check if we're approaching target in the X direction
            max_integral_multiplier = 1.0
            if self.name == "Linear X" and hasattr(self, 'pid_controller') and \
            hasattr(self.pid_controller, 'filtered_distance') and \
            hasattr(self.pid_controller, 'desired_distance') and \
            hasattr(self.pid_controller, 'approach_distance'):
                
                distance_error = abs(self.pid_controller.filtered_distance - self.pid_controller.desired_distance)
                approach_distance = self.pid_controller.approach_distance
                
                # Dynamically scale integral based on distance
                if distance_error < approach_distance:
                    # Calculate scaling factor (lower when closer)
                    max_integral_multiplier = max(0.1, distance_error / approach_distance)
                    
                    # Log integral scaling in approach mode
                    if should_log and hasattr(self, 'debug_level') and self.debug_level >= 2:
                        self.logger.debug(f"Approach scaling: integral_multiplier={max_integral_multiplier:.2f}")
            
            if hasattr(self, 'pid_controller') and hasattr(self.pid_controller, 'filtered_distance'):
                # If we're close to target, use even stricter integral limits
                if abs(self.pid_controller.filtered_distance - self.pid_controller.desired_distance) < 0.2:
                    max_integral_multiplier *= 0.5  # Further reduce integral limit near target

            # Call the enhanced JIT-compiled function with new parameters
            try:
                output, new_integral, p_term, i_term, d_term, new_sign_count, new_prev_sign, new_zero_time, error_change = pid_compute_jit(
                    error, self.prev_error, self.integral, dt, 
                    self.kp, self.ki, self.kd, 
                    self.output_min, self.output_max, 
                    self.name_idx, self.sign_change_count, self.prev_sign, 
                    zero_time, current_time,
                    self.integral_deadband, self.integral_decay, max_integral_multiplier,
                    self.prev_error_change, self.last_output, self.integral_adjustment_factor
                )
                
                # Store error change for next D-term calculation
                self.prev_error_change = error_change
                
                # Log zero crossings which are important events
                if zero_time == 0.0 and new_zero_time > 0.0:
                    # Zero crossing occurred
                    self.logger.info(f"PID {self.name} error crossed zero: integral reset {self.integral:.3f} -> {new_integral:.3f}")
                
                # Check for significant integral changes
                integral_change_ratio = abs(new_integral - self.integral) / (abs(self.integral) + 1e-6)
                if integral_change_ratio > 0.3 and hasattr(self, 'debug_level') and self.debug_level >= 2:
                    self.logger.debug(f"PID {self.name} integral changed: {self.integral:.3f} -> {new_integral:.3f}")
                
                # Update controller state
                self.integral = new_integral
                self.sign_change_count = new_sign_count
                self.prev_sign = new_prev_sign
                
                # Handle zero crossing time (convert 0.0 back to None)
                if new_zero_time > 0.0:
                    self.zero_crossing_time = new_zero_time
                else:
                    self.zero_crossing_time = None
                
                # Save individual terms for diagnostics
                self.last_p_term = p_term
                self.last_i_term = i_term
                self.last_d_term = d_term
                self.last_output = output
                
                # Save state for next iteration
                self.prev_error = error
                self.last_time = current_time
                
                # Store current state in history for monitoring
                self.log_extended_pid_state()
                
                # Apply predictive control elements if appropriate
                if self.name == "Linear X" and abs(error) < 0.3:
                    # Try predictive approach for final approach
                    adjusted_output = self.apply_predictive_control(error, output)
                    if adjusted_output != output:
                        output = adjusted_output
                        self.last_output = output
                
                # Ensure we return a proper float value
                return float(output)
                
            except Exception as e:
                self.logger.warning(f"JIT compute failed: {str(e)}. Using non-optimized version.")
                
                # Original non-JIT implementation (simplified version)
                if current_time is None:
                    current_time = time.time()
                    
                if self.last_time is None:
                    self.last_time = current_time
                    self.prev_error = error
                    output = self.kp * error
                    self.last_p_term = output
                    self.last_output = max(self.output_min, min(self.output_max, output))
                    return float(self.last_output)
                
                dt = current_time - self.last_time
                dt = max(0.001, min(0.1, dt))
                
                # Calculate PID terms
                p_term = self.kp * error
                
                # Enhanced integral handling
                if abs(error) < self.integral_deadband:
                    # Within deadband, decay integral
                    self.integral *= self.integral_decay
                else:
                    # Normal accumulation with direction awareness
                    if error * self.prev_error < 0:
                        # Error changed sign - reset integral more aggressively
                        self.integral = error * dt * 0.5
                    else:
                        # Apply integral adjustment factor
                        self.integral += error * dt * self.integral_adjustment_factor
                
                # Apply limits to integral
                max_integral = self.max_integral * max_integral_multiplier
                self.integral = max(-max_integral, min(max_integral, self.integral))
                
                i_term = self.ki * self.integral
                
                # Enhanced D-term calculation
                error_change = error - self.prev_error
                if abs(error_change) > abs(self.prev_error_change) * 3:
                    # Likely noise spike - use previous value
                    d_term = self.kd * self.prev_error_change / dt
                else:
                    # Smoothed derivative
                    filtered_change = error_change * 0.7 + self.prev_error_change * 0.3
                    d_term = self.kd * filtered_change / dt
                
                # Store for next iteration
                self.prev_error_change = error_change
                
                # Calculate output
                output = p_term + i_term + d_term
                output = max(self.output_min, min(self.output_max, output))
                
                # Update state
                self.prev_error = error
                self.last_time = current_time
                self.last_output = output
                self.last_p_term = p_term
                self.last_i_term = i_term
                self.last_d_term = d_term
                
                return float(output)
                
        except Exception as e:
            # Log critical error
            self.logger.error(f"Critical error in PID {self.name}: {str(e)}")
            
            # Safe fallback to prevent system failure
            return 0.0  # Return zero to prevent erratic behavior
    
    def apply_predictive_control(self, error, base_output):
        """
        Apply predictive control elements for better approach behavior.
        
        Args:
            error: Current error value
            base_output: Base PID output before predictive adjustment
            
        Returns:
            float: Adjusted output with predictive elements
        """
        # Only apply for small errors and positive outputs (approaching target)
        if abs(error) < 0.3 and base_output > 0.05:
            # Predict stopping distance based on current velocity
            velocity = base_output  # Use PID output as velocity estimate
            deceleration_rate = 1.5  # m/s²
            stopping_distance = (velocity * velocity) / (2 * deceleration_rate)
            
            # If stopping distance is close to error, start slowing down early
            if stopping_distance > abs(error) * 0.7:
                # Apply predictive braking factor
                prediction_factor = max(0.5, 1.0 - (stopping_distance / abs(error)))
                adjusted_output = base_output * prediction_factor
                
                # Log predictive action
                self.logger.info(
                    f"PID {self.name} predictive braking: vel={velocity:.3f}, "
                    f"stop_dist={stopping_distance:.3f}m, factor={prediction_factor:.2f}"
                )
                return adjusted_output
                
        return base_output  # No adjustment needed
            
    def get_components(self):
        """Get the last calculated PID components."""
        return (self.last_p_term, self.last_i_term, self.last_d_term)

    def get_current_gains(self):
        """Get the current adaptive gains."""
        return (self.kp, self.ki, self.kd)
    
    def log_extended_pid_state(self):
        """Log extended PID state information for debugging."""
        if not hasattr(self, 'state_history'):
            self.state_history = deque(maxlen=10)
        
        # Store current state
        state = {
            'time': time.time(),
            'error': self.prev_error,
            'output': self.last_output,
            'p_term': self.last_p_term,
            'i_term': self.last_i_term,
            'd_term': self.last_d_term,
            'integral': self.integral,
            'adjustment_factor': self.integral_adjustment_factor,
            'kp': self.kp,
            'ki': self.ki,
            'kd': self.kd
        }
        self.state_history.append(state)
        
        # Calculate rate of change for key metrics
        if len(self.state_history) >= 2:
            last = self.state_history[-1]
            prev = self.state_history[-2]
            dt = last['time'] - prev['time']
            
            if dt > 0:
                integral_change_rate = (last['integral'] - prev['integral']) / dt
                output_change_rate = (last['output'] - prev['output']) / dt
                
                # Log problematic conditions
                if abs(integral_change_rate) > 1 or abs(last['integral']) > self.max_integral * 0.9:
                    self.logger.warning(
                        f"PID {self.name} integral warning: value={self.integral:.3f}, "
                        f"change_rate={integral_change_rate:.3f}, error={self.prev_error:.3f}"
                    )

    
class StrategyBlender:
    """Handles smooth transitions between movement strategies with improved hysteresis."""
    
    def __init__(self, blend_duration=0.2, min_hold_time=0.3):
        """
        Initialize the strategy blender with smoother transitions and improved hysteresis.
        
        Args:
            blend_duration: Time for complete transition between strategies (seconds)
            min_hold_time: Minimum time to hold a strategy before switching (seconds)
        """
        self.current_strategy = None
        self.target_strategy = None
        self.blend_start_time = 0.0
        self.blending_active = False
        self.blend_duration = blend_duration
        
        # MODIFIED: Reduced direction change boost for smoother transitions
        self.direction_change_boost = 2.0  # Reduced from 3.0 to 2.0 to prevent excessive acceleration
        
        self.previous_direction = None
        self.strategy_activation_time = 0.0  # When current strategy became active
        self.min_hold_time = min_hold_time  # Minimum time to hold strategy
        self.strategy_history = []  # Track recent strategies (name, timestamp)
        self.last_strategy_switch_time = 0.0  # Time of last successful strategy switch

        global _logging_manager
        self.debug_level = _logging_manager.get_debug_level()

        # Logger
        self.logger = logging.getLogger('pid_controller.blender')
        
        # ADDED: Special exceptions for angular strategies
        self.angular_strategy_prefixes = ["ANGULAR_", "BALANCED_ANGULAR"]
   
    def update_target(self, target_strategy, current_time):
        """
        Update the target strategy with enhanced handling for angular strategies.
        
        Args:
            target_strategy: The target strategy to blend towards
            current_time: Current time
                
        Returns:
            bool: True if a new blend was started
        """
        # Gather parameters for potential logging
        blend_params = {
            'target_strategy': target_strategy.name,
            'blend_duration': self.blend_duration,
            'min_hold_time': self.min_hold_time
        }
        
        # Initialize if this is the first strategy
        if self.current_strategy is None:
            self.current_strategy = target_strategy
            self.strategy_activation_time = current_time
            self.previous_direction = self._get_strategy_direction(target_strategy)
            self.last_strategy_switch_time = current_time
            
            # Log initial strategy with detailed parameters
            log_strategy_selection(
                "initial", 
                target_strategy.as_dict(),
                reason="Initial strategy set"
            )
            return False
                
        # Check if target is different from current
        if target_strategy.name != self.current_strategy.name:
            # Add current state to parameters
            blend_params['current_strategy'] = self.current_strategy.name
            blend_params['time_in_strategy'] = current_time - self.strategy_activation_time
            
            # Check if we're already blending to this strategy
            if self.blending_active and self.target_strategy.name == target_strategy.name:
                # Log continued blend attempt
                if self.debug_level >= 2:
                    log_structured('strategy_blender', 'BLEND_CONTINUE', 
                                f"Continued blend request to {target_strategy.name} - already in progress",
                                blend_params)
                return False  # Already blending to this strategy
                    
            # MODIFIED: Special handling for angular strategies
            # If target is an angular strategy, use moderate (not very short) hold time
            is_angular_strategy = any(target_strategy.name.startswith(prefix) 
                                    for prefix in self.angular_strategy_prefixes)
            
            # Apply different hold time checks based on strategy type
            time_in_current_strategy = current_time - self.strategy_activation_time
            
            if is_angular_strategy:
                # MODIFIED: Increased from 0.1 to 0.15 for more stability
                angular_min_hold = 0.15  # Slightly longer hold time to avoid rapid oscillation
                if time_in_current_strategy < angular_min_hold:
                    # Still apply a minimal hold time to prevent oscillation
                    blend_params['reason'] = f"Minimal hold time not met ({time_in_current_strategy:.2f}s < {angular_min_hold:.2f}s)"
                    
                    # Log deferred blend with reason
                    log_structured('strategy_blender', 'BLEND_DEFERRED', 
                                f"Strategy switch to {target_strategy.name} deferred - minimal hold time not met", 
                                blend_params)
                    return False
            else:
                # Standard hold time check for non-angular strategies
                if time_in_current_strategy < self.min_hold_time:
                    # Haven't held current strategy long enough, don't switch yet
                    blend_params['reason'] = f"Minimum hold time not met ({time_in_current_strategy:.2f}s < {self.min_hold_time:.2f}s)"
                    
                    # Log deferred blend with reason
                    log_structured('strategy_blender', 'BLEND_DEFERRED', 
                                f"Strategy switch to {target_strategy.name} deferred - minimum hold time not met", 
                                blend_params)
                    return False
            
            # MODIFIED: Stricter similarity check to prevent unnecessary transitions
            # Check for similarity between strategies to avoid unneeded transitions
            strategies_similar = self._are_strategies_similar(self.current_strategy, target_strategy)
            
            # If target is angular strategy, use looser similarity criteria but still check
            if is_angular_strategy and not self.current_strategy.name.startswith("ANGULAR_"):
                # Still check similarity but with looser criteria
                if strategies_similar and abs(self.current_strategy.angular_scale - target_strategy.angular_scale) < 0.3:
                    # Strategies are too similar, don't switch
                    blend_params['reason'] = "Strategies too similar for angular transition"
                    
                    # Log similarity-based deferral
                    log_structured('strategy_blender', 'BLEND_SIMILARITY_SKIP', 
                                f"Strategy switch to {target_strategy.name} skipped - too similar to current strategy", 
                                blend_params)
                    return False
            else:
                if strategies_similar:
                    # Strategies are too similar, don't switch
                    blend_params['reason'] = "Strategies too similar"
                    blend_params['similarity_metrics'] = {
                        'forward_diff': abs(self.current_strategy.forward_scale - target_strategy.forward_scale),
                        'lateral_diff': abs(self.current_strategy.lateral_scale - target_strategy.lateral_scale),
                        'angular_diff': abs(self.current_strategy.angular_scale - target_strategy.angular_scale)
                    }
                    
                    # Log similarity-based deferral
                    log_structured('strategy_blender', 'BLEND_SIMILARITY_SKIP', 
                                f"Strategy switch to {target_strategy.name} skipped - too similar to current strategy", 
                                blend_params)
                    return False
            
            # MODIFIED: Improved oscillation checking for all strategies
            # Check for oscillation in strategy selection
            is_oscillating = self._is_oscillating_between_strategies(target_strategy, current_time)
                
            if is_oscillating:
                # We're oscillating between strategies, don't switch
                blend_params['reason'] = "Oscillation detected"
                blend_params['history'] = [s[0] for s in self.strategy_history[-3:]]
                
                # Log oscillation detection
                log_structured('strategy_blender', 'BLEND_OSCILLATION_DETECTED', 
                            f"Strategy switch to {target_strategy.name} deferred - oscillation pattern detected", 
                            blend_params)
                return False
            
            # Detect direction change for boosting transition speed
            current_direction = self._get_strategy_direction(target_strategy)
            direction_change = False
            
            if self.previous_direction is not None and current_direction is not None:
                # Check if direction components are opposite
                direction_change = (
                    (self.previous_direction[0] * current_direction[0] < 0) or
                    (self.previous_direction[1] * current_direction[1] < 0) or
                    (self.previous_direction[2] * current_direction[2] < 0)
                )
            
            # Start new blend
            self.target_strategy = target_strategy
            self.blend_start_time = current_time
            self.blending_active = True
            self.last_strategy_switch_time = current_time
            
            # Update strategy history
            self.strategy_history.append((self.current_strategy.name, current_time))
            if len(self.strategy_history) > 10:  # Keep last 10 strategies
                self.strategy_history.pop(0)
            
            # MODIFIED: Apply direct override for angular strategies
            # Apply boosting for direction changes or angular strategies
            if direction_change or is_angular_strategy:
                if is_angular_strategy:
                    # Direct override for angular strategies - fixed duration
                    self.effective_blend_duration = 0.05  # Fixed duration for angular strategies
                else:
                    # Use standard boost for direction changes
                    boost_factor = self.direction_change_boost
                    self.effective_blend_duration = self.blend_duration / boost_factor
                
                blend_params['direction_change'] = direction_change
                blend_params['is_angular'] = is_angular_strategy
                blend_params['effective_duration'] = self.effective_blend_duration
            else:
                self.effective_blend_duration = self.blend_duration
                blend_params['effective_duration'] = self.effective_blend_duration
                
            # Update previous direction
            self.previous_direction = current_direction
            
            # Log strategy transition with detailed parameters
            transition_reason = self._determine_transition_reason(
                self.current_strategy, target_strategy)
            
            log_strategy_blending(
                self.current_strategy, 
                target_strategy, 
                0.0,  # Initial blend factor
                self.effective_blend_duration
            )
            
            # Add detailed selection log
            log_strategy_selection(
                "transition", 
                target_strategy.as_dict(),
                candidate_strategies={self.current_strategy.name: self.current_strategy.as_dict()},
                reason=transition_reason
            )
            
            return True
                
        return False

    def _determine_transition_reason(self, current, target):
        """
        Determine a meaningful reason for the strategy transition.
        
        Args:
            current: Current strategy
            target: Target strategy
            
        Returns:
            str: Reason for transition
        """
        # Analyze the differences between strategies
        forward_change = target.forward_scale - current.forward_scale
        lateral_change = target.lateral_scale - current.lateral_scale
        angular_change = target.angular_scale - current.angular_scale
        
        # Determine dominant change
        changes = [
            ("forward", abs(forward_change)),
            ("lateral", abs(lateral_change)),
            ("angular", abs(angular_change))
        ]
        changes.sort(key=lambda x: x[1], reverse=True)
        
        # Get the primary change component
        primary_change = changes[0][0]
        primary_value = changes[0][1]
        
        # Only consider significant changes
        if primary_value < 0.1:
            return f"Minor parameter adjustments from {current.name} to {target.name}"
        
        # Create reason based on primary change
        if primary_change == "forward":
            if forward_change > 0:
                return f"Increasing forward movement ({current.forward_scale:.1f} → {target.forward_scale:.1f})"
            else:
                return f"Decreasing forward movement ({current.forward_scale:.1f} → {target.forward_scale:.1f})"
        elif primary_change == "lateral":
            if lateral_change > 0:
                return f"Increasing lateral movement ({current.lateral_scale:.1f} → {target.lateral_scale:.1f})"
            else:
                return f"Decreasing lateral movement ({current.lateral_scale:.1f} → {target.lateral_scale:.1f})"
        elif primary_change == "angular":
            if angular_change > 0:
                return f"Increasing angular movement ({current.angular_scale:.1f} → {target.angular_scale:.1f})"
            else:
                return f"Decreasing angular movement ({current.angular_scale:.1f} → {target.angular_scale:.1f})"
        
        return f"Strategy change from {current.name} to {target.name}"
    
    def _are_strategies_similar(self, strategy1, strategy2):
        """
        Determine if two strategies are functionally similar.
        
        Args:
            strategy1, strategy2: Strategies to compare
            
        Returns:
            bool: True if strategies have similar behavior
        """
        # Check if both strategies use same components
        components_match = (
            strategy1.use_forward == strategy2.use_forward and
            strategy1.use_lateral == strategy2.use_lateral and
            strategy1.use_angular == strategy2.use_angular
        )
        
        if not components_match:
            return False
        
        # MODIFIED: Tightened similarity thresholds for more stable transitions
        # Calculate similarity scores for each scale factor
        forward_sim = abs(strategy1.forward_scale - strategy2.forward_scale) < 0.15  # Reduced from 0.2
        lateral_sim = abs(strategy1.lateral_scale - strategy2.lateral_scale) < 0.15   # Reduced from 0.2
        angular_sim = abs(strategy1.angular_scale - strategy2.angular_scale) < 0.15  # Reduced from 0.2
        
        # Count number of similar components
        similar_count = sum([forward_sim, lateral_sim, angular_sim])
        
        # Consider similar if at least 2 components are similar
        return similar_count >= 2
    
    def _is_oscillating_between_strategies(self, target_strategy, current_time):
        """
        Detect if we're oscillating between two strategies.
        
        Args:
            target_strategy: New target strategy
            current_time: Current time
            
        Returns:
            bool: True if oscillation is detected
        """
        # Need at least 4 strategy changes to detect oscillation
        if len(self.strategy_history) < 4:
            return False
        
        # MODIFIED: Improved oscillation detection logic
        # Check if we've switched to this strategy recently
        recent_period = 2.0  # Reduced from 3.0 to 2.0 seconds
        matching_strategies = [s for s, t in self.strategy_history 
                              if s == target_strategy.name and current_time - t < recent_period]
        
        # If we've used this strategy 2+ times in the last 2 seconds, consider it oscillation
        if len(matching_strategies) >= 2:
            # Also check for pattern: A->B->A->B
            if len(self.strategy_history) >= 3:
                if (self.strategy_history[-1][0] == self.strategy_history[-3][0] and
                    target_strategy.name == self.strategy_history[-2][0]):
                    # Clear A->B->A->B pattern detected
                    return True
            
            return True
        
        return False
    
    def _get_strategy_direction(self, strategy):
        """
        Extract movement direction from a strategy.
        
        Args:
            strategy: Movement strategy
            
        Returns:
            tuple: Direction vector (forward, lateral, angular) or None
        """
        if not (strategy.use_forward or strategy.use_lateral or strategy.use_angular):
            return None
            
        return (
            1 if strategy.use_forward and strategy.forward_scale > 0 else 
            (-1 if strategy.use_forward and strategy.forward_scale < 0 else 0),
            
            1 if strategy.use_lateral and strategy.lateral_scale > 0 else 
            (-1 if strategy.use_lateral and strategy.lateral_scale < 0 else 0),
            
            1 if strategy.use_angular and strategy.angular_scale > 0 else 
            (-1 if strategy.use_angular and strategy.angular_scale < 0 else 0)
        )
    
    def _enhanced_smoothstep(self, x):
        """
        Improved smoothstep function for smoother transitions.
        Uses a higher-order polynomial for better start/end behavior.
        
        Args:
            x: Input value (0-1)
            
        Returns:
            float: Smoothed value
        """
        x = max(0.0, min(1.0, x))
        # 6th order polynomial for even smoother blending
        return x * x * x * (x * (x * (6 * x - 15) + 10))
    
    def get_current_strategy(self, current_time):
        """
        Get the current strategy, which might be a blend of two strategies.
        
        Args:
            current_time: Current time
            
        Returns:
            MovementStrategy: Current strategy (possibly blended)
        """
        if not self.blending_active:
            return self.current_strategy
            
        # Calculate blend factor
        elapsed_time = current_time - self.blend_start_time
        
        # Use effective blend duration that might be boosted for direction changes
        blend_duration = getattr(self, 'effective_blend_duration', self.blend_duration)
        
        linear_blend = min(1.0, elapsed_time / blend_duration)
        blend_factor = self._enhanced_smoothstep(linear_blend)
        
        # Check if blending is complete
        if blend_factor >= 0.999:
            self.current_strategy = self.target_strategy
            self.blending_active = False
            self.strategy_activation_time = current_time
            return self.current_strategy
        
        # Create blended strategy
        name = f"{self.current_strategy.name}→{self.target_strategy.name}"
        
        # MODIFIED: Improved blending logic for smoother transitions
        
        # Determine boolean flags using OR logic for smoother transitions
        # Extended to prevent jerky transitions during blending
        use_forward = self.target_strategy.use_forward or (
            self.current_strategy.use_forward and blend_factor < 0.9)  # Extended transition
            
        use_lateral = self.target_strategy.use_lateral or (
            self.current_strategy.use_lateral and blend_factor < 0.9)
            
        use_angular = self.target_strategy.use_angular or (
            self.current_strategy.use_angular and blend_factor < 0.9)
        
        # MODIFIED: Enhanced blending curve with smoother transitions
        # For angular transitions, use a different curve to avoid overshoot
        is_angular_transition = (
            self.target_strategy.name.startswith("ANGULAR_") or 
            self.current_strategy.name.startswith("ANGULAR_")
        )
        
        if is_angular_transition:
            # For angular transitions, use a more cautious blend approach
            # Bias blend factor to change more slowly at the beginning
            adj_blend_factor = pow(blend_factor, 1.3)  # Makes transition slower at start
            
            # Angular control blending
            angular_scale = self._blend_parameter_cautious(
                self.current_strategy.angular_scale,
                self.target_strategy.angular_scale,
                adj_blend_factor
            )
            
            # Use standard blending for other parameters
            forward_scale = self._blend_parameter(
                self.current_strategy.forward_scale,
                self.target_strategy.forward_scale,
                blend_factor
            )
            
            lateral_scale = self._blend_parameter(
                self.current_strategy.lateral_scale,
                self.target_strategy.lateral_scale,
                blend_factor
            )
        else:
            # Use standard enhanced blending for non-angular transitions
            forward_scale = self._blend_parameter(
                self.current_strategy.forward_scale,
                self.target_strategy.forward_scale,
                blend_factor
            )
            
            lateral_scale = self._blend_parameter(
                self.current_strategy.lateral_scale,
                self.target_strategy.lateral_scale,
                blend_factor
            )
            
            angular_scale = self._blend_parameter(
                self.current_strategy.angular_scale,
                self.target_strategy.angular_scale,
                blend_factor
            )
        
        reason = f"Blending strategies: {blend_factor*100:.0f}% complete"
        
        return MovementStrategy(
            name, use_forward, use_lateral, use_angular,
            forward_scale, lateral_scale, angular_scale, reason
        )
    
    def _blend_parameter(self, start_value, end_value, blend_factor):
        """
        Blend a numeric parameter with enhanced curve for smoother transitions.
        
        Args:
            start_value: Starting value
            end_value: Ending value
            blend_factor: Blend factor (0-1)
            
        Returns:
            float: Blended value
        """
        # Enhanced blending curve to make transitions smoother at the beginning and end
        if blend_factor < 0.2:
            # Slower at the start - ease in
            adjusted_factor = blend_factor * 0.5
        elif blend_factor > 0.8:
            # Slower at the end - ease out
            adjusted_factor = 0.5 + (blend_factor - 0.5) * 1.5
        else:
            # Normal in the middle
            adjusted_factor = blend_factor
            
        # Apply the adjusted factor
        return start_value * (1.0 - adjusted_factor) + end_value * adjusted_factor
    
    def _blend_parameter_cautious(self, start_value, end_value, blend_factor):
        """
        Blend a numeric parameter with a more aggressive curve for angular parameters.
        
        Args:
            start_value: Starting value
            end_value: Ending value
            blend_factor: Blend factor (0-1)
            
        Returns:
            float: Blended value
        """
        # MODIFIED: More aggressive blending curve for angular parameters
        # Faster at the start, controlled in middle, faster at the end
        if blend_factor < 0.3:
            # Faster at the start for quicker initial response
            adjusted_factor = blend_factor * 0.5  # Increased from 0.3 to 0.5
        elif blend_factor < 0.7:
            # Controlled in the middle to avoid overshoot
            adjusted_factor = 0.15 + (blend_factor - 0.3) * 0.7  # Adjusted curve
        else:
            # Faster at the end for quick completion
            adjusted_factor = 0.43 + (blend_factor - 0.7) * 2.5  # More aggressive end
            
        # Cap at 1.0 in case we exceed it
        adjusted_factor = min(1.0, adjusted_factor)
            
        # Apply the adjusted factor
        return start_value * (1.0 - adjusted_factor) + end_value * adjusted_factor
    
    def is_blending(self):
        """Check if a blend is currently in progress."""
        return self.blending_active
    
    def get_blend_progress(self, current_time):
        """Get the current blend progress as a percentage."""
        if not self.blending_active:
            return 100.0
            
        # Use effective blend duration for calculating progress
        blend_duration = getattr(self, 'effective_blend_duration', self.blend_duration)
        elapsed_time = current_time - self.blend_start_time
        
        # Apply non-linear curve to progress for more intuitive reporting
        linear_progress = min(1.0, elapsed_time / blend_duration)
        curved_progress = self._enhanced_smoothstep(linear_progress)
        
        return curved_progress * 100.0
    
    def get_strategy_stability(self, current_time):
        """
        Get a measure of how stable the current strategy selection has been.
        
        Args:
            current_time: Current time
            
        Returns:
            float: Stability score (0-1), higher means more stable
        """
        # If we're blending, stability is low
        if self.blending_active:
            return 0.5
        
        # Calculate time since last strategy switch
        time_since_switch = current_time - self.last_strategy_switch_time
        
        # Normalize to a 0-1 scale, capping at 3 seconds (3s = fully stable)
        stability = min(1.0, time_since_switch / 3.0)
        
        return stability
    
    def debug_enabled(self):
        """Check if debug logging is enabled."""
        # Helper method to reduce log spam
        return hasattr(self, 'debug_level') and getattr(self, 'debug_level', 0) >= 2
    
    def reset(self):
        """Reset the blender state."""
        self.current_strategy = None
        self.target_strategy = None
        self.blending_active = False
        self.previous_direction = None
        self.strategy_activation_time = 0.0
        self.last_strategy_switch_time = 0.0
        self.strategy_history = []
        self.logger.info("Strategy blender reset")


class ResourceMonitor:
    """
    Lightweight resource monitor for tracking CPU and memory usage.
    Provides callbacks for resource threshold alerts.
    """
    
    def __init__(self, update_interval=5.0):
        """
        Initialize the resource monitor.
        
        Args:
            update_interval: How often to update resource metrics (seconds)
        """
        self.update_interval = update_interval
        self.last_update_time = 0
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.alert_callbacks = []
        self.cpu_threshold = 85.0  # Default CPU usage threshold (%)
        self.memory_threshold = 85.0  # Default memory usage threshold (%)
        self.logger = logging.getLogger('pid_controller.resource_monitor')
    
    def update(self):
        """Update resource metrics with improved logging."""
        current_time = time.time()
        
        # Only update at specified interval
        if current_time - self.last_update_time >= self.update_interval:
            # Get previous values for change detection
            prev_cpu = self.cpu_usage
            prev_memory = self.memory_usage
            
            # Update metrics
            self.cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            self.memory_usage = memory.percent
            
            # Check for significant changes (>10% absolute or >20% relative)
            cpu_change = abs(self.cpu_usage - prev_cpu)
            mem_change = abs(self.memory_usage - prev_memory)
            
            significant_change = (cpu_change > 10.0 or 
                                mem_change > 10.0 or 
                                (prev_cpu > 0 and cpu_change / prev_cpu > 0.2) or
                                (prev_memory > 0 and mem_change / prev_memory > 0.2))
            
            # Check thresholds for alerts
            triggered_alert = self._check_thresholds()
            
            # Log on significant changes or threshold triggers
            if significant_change or triggered_alert:
                # Only log changes or alerts to reduce volume
                log_resource_usage(self.cpu_usage, self.memory_usage)
            
            self.last_update_time = current_time
            return True
        
        return False
    
    def _check_thresholds(self):
        """Check if any resource metrics exceed thresholds and trigger callbacks."""
        triggered = False
        
        # Check CPU
        if self.cpu_usage > self.cpu_threshold:
            for callback in self.alert_callbacks:
                callback('cpu', self.cpu_usage)
            triggered = True
        
        # Check memory
        if self.memory_usage > self.memory_threshold:
            for callback in self.alert_callbacks:
                callback('memory', self.memory_usage)
            triggered = True
        
        return triggered
    
    def add_alert_callback(self, callback):
        """
        Add a callback to be called when resource thresholds are exceeded.
        
        Args:
            callback: Function to call with (resource_type, value) parameters
        """
        self.alert_callbacks.append(callback)
    
    def get_cpu_usage(self):
        """Get the last measured CPU usage."""
        return self.cpu_usage
    
    def get_memory_usage(self):
        """Get the last measured memory usage."""
        return self.memory_usage
        
    def log_stats(self):
        """Log current resource statistics using specialized logging function."""
        # Use the specialized logging function
        log_resource_usage(self.cpu_usage, self.memory_usage)


# Core computational functions optimized with Numba
@numba.jit(nopython=True, fastmath=True)
def compute_coordination_jit(lateral_error, angular_error, current_time, last_update_time,
                           last_lateral_velocity, last_angular_velocity, 
                           config_values, robot_orientation):
    """
    Compute coordinated lateral and angular velocities with Numba optimization.
    
    Args:
        lateral_error: Current lateral error
        angular_error: Current angular error in radians
        current_time: Current time
        last_update_time: Time of last update
        last_lateral_velocity: Last lateral velocity
        last_angular_velocity: Last angular velocity
        config_values: Array of configuration values [coupling_factor, min_angle_for_reduction,
                                                    zero_angle_threshold, max_angle_factor,
                                                    same_sign_scale, opposite_sign_scale,
                                                    smoothing_factor]
        robot_orientation: Current robot orientation in radians
        
    Returns:
        tuple: (lateral_velocity, angular_velocity, dt) where dt is time since last update
    """
    # Calculate dt
    dt = current_time - last_update_time
    dt = max(0.001, min(dt, 0.1))  # Bound dt to reasonable values
    
    # Extract configuration parameters
    coupling_factor = config_values[0]
    min_angle_for_reduction = config_values[1]
    zero_angle_threshold = config_values[2]
    max_angle_factor = config_values[3]
    same_sign_scale = config_values[4]
    opposite_sign_scale = config_values[5]
    smoothing_factor = config_values[6]
    
    # Get angular error magnitude
    angular_magnitude = abs(angular_error)
    
    # Initialize velocity values (will be modified below)
    lateral_velocity = lateral_error
    angular_velocity = angular_error
    
    # Apply coordination logic
    # 1. Calculate coupling based on angular error magnitude
    if angular_magnitude > min_angle_for_reduction:
        # Normalize angular error to 0-1 range up to max_angle_factor
        normalized_angle = min(1.0, angular_magnitude / max_angle_factor)
        
        # Calculate lateral velocity reduction
        lateral_reduction = normalized_angle * coupling_factor
        
        # Reduce lateral velocity when angular error is large
        lateral_velocity = lateral_velocity * (1.0 - lateral_reduction)
    
    # 2. Adjust based on sign relationship between errors
    same_sign = (lateral_error * angular_error) > 0
    
    if angular_magnitude > zero_angle_threshold:
        if same_sign:
            # Same sign - needs coordinated movement
            lateral_velocity = lateral_velocity * same_sign_scale
        else:
            # Opposite sign - movement naturally helps correction
            lateral_velocity = lateral_velocity * opposite_sign_scale
    
    # 3. Apply smoothing to prevent jerky transitions
    if last_lateral_velocity != 0.0 or last_angular_velocity != 0.0:
        # Apply smoothing
        lateral_velocity = last_lateral_velocity * (1.0 - smoothing_factor) + lateral_velocity * smoothing_factor
        angular_velocity = last_angular_velocity * (1.0 - smoothing_factor) + angular_velocity * smoothing_factor
    
    return lateral_velocity, angular_velocity, dt


def prepare_config_array(config_dict):
    """
    Prepare configuration array from dictionary for use with Numba.
    
    Args:
        config_dict: Dictionary with configuration parameters
        
    Returns:
        numpy.ndarray: Array of configuration values
    """
    # Create array with default values
    config_array = np.zeros(7, dtype=np.float32)
    
    # Fill with values from config dictionary, with defaults if not present
    config_array[0] = config_dict.get('coupling_factor', 0.4)
    config_array[1] = config_dict.get('min_angle_for_reduction', 0.1)
    config_array[2] = config_dict.get('zero_angle_threshold', 0.03)
    config_array[3] = config_dict.get('max_angle_factor', 0.3)
    config_array[4] = config_dict.get('same_sign_scale', 0.8)
    config_array[5] = config_dict.get('opposite_sign_scale', 1.2)
    config_array[6] = config_dict.get('smoothing_factor', 0.4)
    
    return config_array


class CoordinatedController:
    """Controller that coordinates lateral and angular movements, optimized with Numba."""
    
    def __init__(self, linear_pid, angular_pid, config=None):
        """
        Initialize the coordinated controller.
        
        Args:
            linear_pid: PID controller for lateral movement
            angular_pid: PID controller for angular movement
            config: Configuration dictionary
        """
        self.logger = get_logger('coordinated_control')
        self.linear_pid = linear_pid
        self.angular_pid = angular_pid
        
        # Default configuration with improved values
        self.config = {
            'coupling_factor': 0.4,         # Reduced from 0.7 to allow more lateral movement
            'min_angle_for_reduction': 0.1,  # ~5.7 degrees
            'zero_angle_threshold': 0.03,    # ~1.7 degrees
            'max_angle_factor': 0.3,         # Reduced from 0.5 to be less aggressive
            'same_sign_scale': 0.8,          # Scaling when errors have same sign
            'opposite_sign_scale': 1.2,      # Increased from 1.0 to prioritize when errors help each other
            'smoothing_factor': 0.4,         # Reduced from 0.6 for faster response
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Convert config to array format for Numba
        self.config_array = prepare_config_array(self.config)
            
        # State variables
        self.last_lateral_velocity = 0.0
        self.last_angular_velocity = 0.0
        self.last_update_time = None
        
        # For logging and debugging
        self.compute_count = 0
        
        # Warm up JIT compilation
        self._warmup_jit()
        
    def _warmup_jit(self):
        """Warm up JIT compilation to avoid delays during operation with improved logging."""
        try:
            start_time = time.time()
            log_structured('jit_compiler', 'JIT_WARMUP_START', 
                        f"Starting JIT warmup for CoordinatedController",
                        {'linear_pid': self.linear_pid.name, 
                        'angular_pid': self.angular_pid.name})
            
            # Create dummy data
            dummy_time = time.time()
            
            t1 = time.time()
            # Call JIT-compiled function with dummy data
            lateral_velocity, angular_velocity, _ = compute_coordination_jit(
                0.1, 0.1, dummy_time, dummy_time - 0.1,
                0.0, 0.0, self.config_array, 0.0
            )
            log_jit_compilation("compute_coordination_jit", True, time.time() - t1)
            
            elapsed_time = time.time() - start_time
            log_structured('jit_compiler', 'JIT_WARMUP_COMPLETE', 
                        f"JIT compilation warmup completed for CoordinatedController",
                        {'elapsed_ms': elapsed_time * 1000,
                        'result': f"({lateral_velocity:.3f}, {angular_velocity:.3f})"})
            
            self.logger.info(f"JIT compilation warmup completed for CoordinatedController in {elapsed_time*1000:.1f}ms")
        except Exception as e:
            log_jit_compilation("CoordinatedController_warmup", False, None)
            self.logger.warning(f"JIT warmup failed: {str(e)}. Will use non-optimized fallbacks.")
    
    def compute(self, lateral_error, angular_error, current_time=None, robot_orientation=0.0):
        """
        Compute coordinated lateral and angular velocities.
        
        Args:
            lateral_error: Current lateral error
            angular_error: Current angular error in radians
            current_time: Current time (defaults to now)
            robot_orientation: Current robot orientation in radians
            
        Returns:
            tuple: (lateral_velocity, angular_velocity)
        """
        try:
            if current_time is None:
                current_time = time.time()
                
            # Initialize time on first call
            if self.last_update_time is None:
                self.last_update_time = current_time
                
            # Get individual PID outputs
            lateral_trend = self.linear_pid.error_tracker.get_trend() if hasattr(self.linear_pid, 'error_tracker') else 0.0
            angular_trend = self.angular_pid.error_tracker.get_trend() if hasattr(self.angular_pid, 'error_tracker') else 0.0
            
            # Control strategy adaptation - prioritize angular control for larger angular errors
            angular_magnitude = abs(angular_error)
            large_angle_threshold = 0.1  # ~5.7 degrees
            
            # Angular-first strategy - for larger angular errors, compute angular first
            # and modulate lateral based on angular progress
            if angular_magnitude > large_angle_threshold:
                # Get raw angular velocity first
                raw_angular_velocity = self.angular_pid.compute(
                    angular_error, 
                    current_time, 
                    force_zero=False,
                    error_trend=angular_trend
                )
                
                # Then compute lateral velocity, potentially with reduced effect
                angular_progress = 1.0 - min(1.0, angular_magnitude / self.config['max_angle_factor'])
                
                # Scale lateral control based on angular progress 
                lateral_force_zero = angular_magnitude > (large_angle_threshold * 2)
                
                raw_lateral_velocity = self.linear_pid.compute(
                    lateral_error, 
                    current_time, 
                    force_zero=lateral_force_zero,
                    error_trend=lateral_trend
                )
                
                # Gradually phase in lateral control as angular error reduces
                if not lateral_force_zero:
                    # Apply scaling that increases as angular error decreases
                    raw_lateral_velocity *= angular_progress
                    
            else:
                # For smaller angular errors, compute both normally
                raw_lateral_velocity = self.linear_pid.compute(
                    lateral_error, 
                    current_time, 
                    force_zero=False, 
                    error_trend=lateral_trend
                )
                
                raw_angular_velocity = self.angular_pid.compute(
                    angular_error, 
                    current_time, 
                    force_zero=False,
                    error_trend=angular_trend
                )
            
            # Apply Numba-optimized coordination calculations
            lateral_velocity, angular_velocity, dt = compute_coordination_jit(
                raw_lateral_velocity, raw_angular_velocity,
                current_time, self.last_update_time,
                self.last_lateral_velocity, self.last_angular_velocity,
                self.config_array, robot_orientation
            )
            
            # Store values for next iteration
            self.last_lateral_velocity = lateral_velocity
            self.last_angular_velocity = angular_velocity
            self.last_update_time = current_time
            
            # Log coordination details occasionally
            if hasattr(self, 'compute_count') and self.compute_count % 20 == 0:
                self.logger.info(
                    f"Coordinated control: lateral_err={lateral_error:.3f}, angular_err={angular_error:.3f}, "
                    f"lateral_vel={lateral_velocity:.3f}, angular_vel={angular_velocity:.3f}"
                )
                
            if not hasattr(self, 'compute_count'):
                self.compute_count = 0
            self.compute_count += 1
            
            return lateral_velocity, angular_velocity
            
        except Exception as e:
            self.logger.error(f"Error in coordinated compute: {str(e)}")
            
            # Fallback to non-JIT computation
            # Calculate dt
            dt = current_time - self.last_update_time
            dt = max(0.001, min(dt, 0.1))  # Bound dt to reasonable values
            
            # Apply coordination logic directly
            angular_magnitude = abs(angular_error)
            
            # Initialize with raw values
            lateral_velocity = raw_lateral_velocity
            angular_velocity = raw_angular_velocity
            
            # Apply coupling based on angular error magnitude
            if angular_magnitude > self.config['min_angle_for_reduction']:
                normalized_angle = min(1.0, angular_magnitude / self.config['max_angle_factor'])
                lateral_reduction = normalized_angle * self.config['coupling_factor']
                lateral_velocity *= (1.0 - lateral_reduction)
            
            # Adjust based on sign relationship
            same_sign = (lateral_error * angular_error) > 0
            if angular_magnitude > self.config['zero_angle_threshold']:
                if same_sign:
                    lateral_velocity *= self.config['same_sign_scale']
                else:
                    lateral_velocity *= self.config['opposite_sign_scale']
            
            # Apply smoothing
            smoothing = self.config['smoothing_factor']
            lateral_velocity = self.last_lateral_velocity * (1 - smoothing) + lateral_velocity * smoothing
            angular_velocity = self.last_angular_velocity * (1 - smoothing) + angular_velocity * smoothing
            
            # Store values for next iteration
            self.last_lateral_velocity = lateral_velocity
            self.last_angular_velocity = angular_velocity
            self.last_update_time = current_time
            
            return lateral_velocity, angular_velocity
    
    def reset(self):
        """Reset the controller state."""
        self.linear_pid.reset()
        self.angular_pid.reset()
        self.last_lateral_velocity = 0.0
        self.last_angular_velocity = 0.0
        self.last_update_time = None
        if hasattr(self, 'compute_count'):
            self.compute_count = 0
    
    def update_config(self, new_config):
        """
        Update controller configuration.
        
        Args:
            new_config: Dictionary with new configuration values
        """
        # Update configuration dictionary
        self.config.update(new_config)
        
        # Regenerate Numba-compatible array
        self.config_array = prepare_config_array(self.config)
        
        self.logger.info(f"Updated coordinated controller config: {self.config}")


class MovementStrategy:
    """Represents a robot movement strategy with blending capabilities."""
    
    def __init__(self, name, use_forward, use_lateral, use_angular, 
                 forward_scale, lateral_scale, angular_scale, reason):
        """Initialize a movement strategy."""
        self.name = name
        self.use_forward = use_forward
        self.use_lateral = use_lateral
        self.use_angular = use_angular
        self.forward_scale = forward_scale
        self.lateral_scale = lateral_scale
        self.angular_scale = angular_scale
        self.reason = reason
    
    def as_dict(self):
        """Convert to dictionary for compatibility with existing code."""
        return {
            "strategy_name": self.name,
            "use_forward": self.use_forward,
            "use_lateral": self.use_lateral,
            "use_angular": self.use_angular,
            "forward_scale": self.forward_scale,
            "lateral_scale": self.lateral_scale,
            "angular_scale": self.angular_scale,
            "reason": self.reason
        }

@numba.jit(nopython=True, fastmath=True)
def apply_velocity_limits_core(target_velocities, prev_velocities, dt, accel_limit, angular_accel_limit, 
                              distance_error=0.0, approach_distance=0.3, is_approaching=False, 
                              min_effective_velocity=0.01, min_angular_velocity=0.01):
    """
    JIT-compiled core of velocity limiting logic with enhanced acceleration limiting.
    """
    # Pre-allocate outputs
    limited_velocities = np.zeros(3, dtype=np.float32)
    vel_diffs = np.zeros(3, dtype=np.float32)
    
    # Calculate velocity differences
    for i in range(3):
        vel_diffs[i] = target_velocities[i] - prev_velocities[i]
    
    # Direction-specific acceleration limits
    forward_accel_limit = accel_limit * 0.8  # Reduce forward acceleration by 20%
    lateral_accel_limit = accel_limit * 1.0  # Keep lateral acceleration unchanged
    
    # Enhanced deceleration for approaching target
    decel_boost = 1.0
    if is_approaching and distance_error < approach_distance * 1.5:
        # Check if we're decelerating (slowing down) in forward direction
        is_decelerating = (prev_velocities[0] > 0.0 and target_velocities[0] < prev_velocities[0])
        
        if is_decelerating:
            # More aggressive deceleration as we get closer to target
            # The closer we are, the stronger the deceleration
            proximity_factor = 1.0 - (distance_error / approach_distance)
            proximity_factor = max(0.0, min(1.0, proximity_factor))  # Clamp to 0-1 range
            
            # Increase deceleration boost based on proximity (up to 2.0x)
            decel_boost = 1.0 + (1.0 * proximity_factor)
            
            # Apply the boost to the forward direction
            forward_accel_limit *= decel_boost
    
    # Apply acceleration limits for each direction
    for i in range(3):
        # Select the appropriate limit based on the direction
        if i == 0:  # Linear X (forward/backward)
            limit = forward_accel_limit * dt
        elif i == 1:  # Linear Y (lateral)
            limit = lateral_accel_limit * dt
        else:  # Angular Z
            limit = angular_accel_limit * dt
        
        # Special handling for acceleration from stop
        if abs(prev_velocities[i]) < 0.01 and abs(target_velocities[i]) > 0.01:
            # Less boost for forward direction, more for others
            if i == 0:  # Forward motion
                limit *= 2.0  # Reduced from 3.0 for more cautious starts
            else:
                limit *= 3.0  # Keep the original boost for other directions
        
        # If velocity change exceeds limit, scale it
        if abs(vel_diffs[i]) > limit:
            if vel_diffs[i] > 0:
                limited_velocities[i] = prev_velocities[i] + limit
            else:
                limited_velocities[i] = prev_velocities[i] - limit
        else:
            limited_velocities[i] = target_velocities[i]
    
    # The rest of the function remains the same...
    # Apply minimum velocity thresholds, etc.
    
    return limited_velocities


class StoppingStrategy:
    """Base class for stopping strategies."""
    
    def should_stop(self, robot_state, errors, thresholds, velocities):
        """
        Determine if the robot should stop based on this strategy.
        
        Args:
            robot_state: Dictionary with robot state info
            errors: Dictionary with current errors (distance, lateral, angular)
            thresholds: Dictionary with current thresholds
            velocities: Current robot velocities (linear_x, linear_y, angular)
            
        Returns:
            tuple: (should_stop, reason, importance)
        """
        raise NotImplementedError("Subclasses must implement this method")


class DistancePriorityStrategy(StoppingStrategy):
    """Strategy that prioritizes distance over other errors."""
    
    def should_stop(self, robot_state, errors, thresholds, velocities):
        # Extract errors and thresholds
        distance_error = abs(errors['distance'])
        distance_threshold = thresholds['distance']
        
        # Distance is good - prioritize stopping based on distance
        if distance_error <= distance_threshold:
            return True, f"Target distance reached: {distance_error:.3f}m ≤ {distance_threshold:.3f}m", 10
        
        # Distance is outside threshold - keep moving
        return False, f"Distance error: {distance_error:.3f}m > {distance_threshold:.3f}m", 10


class VelocityAwareStrategy(StoppingStrategy):
    """Strategy that anticipates stopping based on current velocity."""
    
    def should_stop(self, robot_state, errors, thresholds, velocities):
        # Extract errors, thresholds and velocities
        distance_error = errors['distance']
        distance_threshold = thresholds['distance']
        linear_x = velocities['linear_x']
        
        # Skip if not moving forward or if moving away from target
        if linear_x <= 0.01 or distance_error < 0:
            return False, "Not moving forward or moving away", 5
        
        # Calculate stopping distance based on velocity and deceleration rate
        # Assuming 1.5 m/s² deceleration capability
        deceleration_rate = 1.5
        stopping_distance = (linear_x * linear_x) / (2 * deceleration_rate)
        stopping_buffer = 0.05  # Additional buffer for safety
        
        # If stopping distance is close to error, start stopping
        if stopping_distance >= abs(distance_error) - stopping_buffer:
            return True, f"Early stop: velocity={linear_x:.2f}m/s requires {stopping_distance:.2f}m to stop", 8
            
        return False, "Can stop safely later", 5


class BrakeIfOvershootingStrategy(StoppingStrategy):
    """Emergency strategy that stops the robot if it's about to overshoot."""
    
    def should_stop(self, robot_state, errors, thresholds, velocities):
        # Extract errors and velocities
        distance_error = errors['distance']  # Negative means past target
        linear_x = velocities['linear_x']
        
        # If we've overshot and still moving forward - emergency stop
        if distance_error < 0 and linear_x > 0.01:
            return True, f"EMERGENCY STOP: Overshooting target, distance_error={distance_error:.3f}m", 20
            
        # If we're close to target and moving fast - emergency stop
        if abs(distance_error) < 0.1 and linear_x > 0.15:
            return True, f"EMERGENCY STOP: Too fast near target, velocity={linear_x:.2f}m/s", 20
            
        return False, "No emergency braking needed", 5


class AlignmentStrategy(StoppingStrategy):
    """Strategy focused on angular and lateral alignment."""
    
    def should_stop(self, robot_state, errors, thresholds, velocities):
        # Extract errors and thresholds
        lateral_error = abs(errors['lateral'])
        angular_error = abs(errors['angular'])
        distance_error = abs(errors['distance'])
        
        lateral_threshold = thresholds['lateral']
        angular_threshold = thresholds['angular']
        
        # Only care about alignment if distance is reasonable
        if distance_error > thresholds['distance'] * 1.5:
            return False, "Too far for alignment check", 2
            
        # Check if alignment is within thresholds
        if lateral_error <= lateral_threshold and angular_error <= angular_threshold:
            return True, f"Good alignment: lateral={lateral_error:.3f}m, angular={angular_error:.2f}°", 5
        
        # If at good distance but poor alignment
        if distance_error <= thresholds['distance']:
            # Bad alignment at good distance - report but don't force stop
            reason = []
            if lateral_error > lateral_threshold:
                reason.append(f"lateral={lateral_error:.3f}m > {lateral_threshold:.3f}m")
            if angular_error > angular_threshold:
                reason.append(f"angular={angular_error:.2f}° > {angular_threshold:.2f}°")
                
            return False, "Alignment needs improvement: " + ", ".join(reason), 3
            
        return False, "Alignment outside thresholds", 2


class ThresholdCalculator:
    """Calculator for dynamic thresholds with hysteresis."""
    
    def __init__(self, base_thresholds, debug_level=0):
        """
        Initialize with base thresholds.
        
        Args:
            base_thresholds: Dictionary with base threshold values
            debug_level: Debug level for logging (0-2)
        """
        self.base_thresholds = base_thresholds
        self.debug_level = debug_level
        self.movement_hysteresis = 0.0
        
    def calculate_thresholds(self, is_stopped, at_target_distance=False, angular_at_target_factor=1.0):
        """
        Calculate dynamic thresholds based on robot state with hysteresis.
        
        Args:
            is_stopped: Whether the robot is currently stopped
            at_target_distance: Whether the robot is at the target distance
            angular_at_target_factor: Factor to adjust angular threshold when at target
            
        Returns:
            dict: Calculated thresholds
        """
        thresholds = {}
        
        # Copy base thresholds
        for key, value in self.base_thresholds.items():
            thresholds[key] = value
        
        # Apply hysteresis based on current state
        if is_stopped:
            # Higher thresholds to start moving (more stable stopped state)
            hysteresis = 1.2 + self.movement_hysteresis  # Less aggressive than original 1.5
            
            # Apply hysteresis to all thresholds
            for key in thresholds:
                thresholds[key] *= hysteresis
                
            # Cap maximum thresholds for sanity
            thresholds['distance'] = min(thresholds['distance'], 0.15)
            thresholds['lateral'] = min(thresholds['lateral'], 0.15)
            thresholds['angular'] = min(thresholds['angular'], 10.0)
            
        else:
            # Lower thresholds to stop (easier to stop than to start)
            # Using 0.9 instead of the original 0.8 to make stopping slightly easier
            hysteresis = 0.9
            
            # Apply hysteresis to all thresholds
            for key in thresholds:
                thresholds[key] *= hysteresis
                
            # Accumulate a small amount of hysteresis to prevent oscillations
            # This gets reset when the robot actually moves
            self.movement_hysteresis += 0.03  # Reduced from 0.05
            self.movement_hysteresis = min(0.2, self.movement_hysteresis)  # Cap at 0.2 instead of 0.3
            
        # Apply increased angular threshold when at target distance
        if at_target_distance:
            thresholds['angular'] *= angular_at_target_factor
            
        return thresholds
        
    def reset_hysteresis(self):
        """Reset accumulated hysteresis."""
        self.movement_hysteresis = 0.0


class StopDecisionManager:
    """Manager for stop decisions using multiple strategies."""
    
    def __init__(self, threshold_calculator, debug_level=0):
        """
        Initialize with threshold calculator.
        
        Args:
            threshold_calculator: ThresholdCalculator instance
            debug_level: Debug level for logging (0-2)
        """
        self.threshold_calculator = threshold_calculator
        self.debug_level = debug_level
        self.strategies = []
        
    def add_strategy(self, strategy):
        """
        Add a stopping strategy.
        
        Args:
            strategy: StoppingStrategy instance
        """
        self.strategies.append(strategy)
        
    def evaluate_stop_conditions(self, distance_error, lateral_error, angular_error, 
                                is_stopped, velocities):
        """
        Evaluate if the robot should stop based on current conditions.
        
        Args:
            distance_error: Current distance error
            lateral_error: Current lateral error
            angular_error: Current angular error in degrees
            is_stopped: Whether the robot is currently stopped
            velocities: Dictionary with current velocities
            
        Returns:
            tuple: (should_stop, reason) - True if robot should stop, False if it should move
        """
        # Prepare errors dictionary
        errors = {
            'distance': distance_error,
            'lateral': lateral_error,
            'angular': angular_error
        }
        
        # Check if we're at target distance
        at_target_distance = abs(distance_error) < self.threshold_calculator.base_thresholds['distance'] * 1.5
        
        # Calculate thresholds with hysteresis
        thresholds = self.threshold_calculator.calculate_thresholds(
            is_stopped=is_stopped,
            at_target_distance=at_target_distance,
            angular_at_target_factor=1.5  # Reduced from what was likely a higher value
        )
        
        # Prepare robot state
        robot_state = {
            'is_stopped': is_stopped,
            'at_target_distance': at_target_distance
        }
        
        # Evaluate all strategies
        results = []
        for strategy in self.strategies:
            should_stop, reason, importance = strategy.should_stop(
                robot_state, errors, thresholds, velocities
            )
            
            results.append((should_stop, reason, importance))
            
        # Sort by importance (highest first)
        results.sort(key=lambda x: x[2], reverse=True)
        
        # Log all strategy results at high debug level
        if self.debug_level >= 2:
            for result in results:
                print(f"Strategy result: should_stop={result[0]}, "
                     f"reason={result[1]}, importance={result[2]}")
        
        # Use highest importance result
        should_stop, reason, _ = results[0]
        
        if should_stop:
            # If we're stopping, find supporting reasons from other strategies
            supporting_reasons = [r for s, r, _ in results if s and r != reason]
            if supporting_reasons:
                reason += " (Also: " + ", ".join(supporting_reasons[:2]) + ")"
        else:
            # If we're not stopping, reset hysteresis when we actually move
            # This prevents accumulated hysteresis from making it too hard to stop later
            if not is_stopped:
                self.threshold_calculator.reset_hysteresis()
        
        return should_stop, reason


class StartMovementStrategy(StoppingStrategy):
    """Strategy that determines if robot should start moving from a stopped state."""
    
    def __init__(self, threshold_multiplier=0.7):
        """
        Initialize with threshold multiplier.
        
        Args:
            threshold_multiplier: Factor to multiply thresholds (smaller = easier to start moving)
        """
        self.threshold_multiplier = threshold_multiplier
        self.initial_movement_boost = True  # Flag for first movement after startup
    
    def should_move(self, robot_state, errors, thresholds, velocities):
        """
        Determine if the robot should start moving based on errors exceeding thresholds.
        
        Args:
            robot_state: Dictionary with robot state info
            errors: Dictionary with current errors (distance, lateral, angular)
            thresholds: Dictionary with current thresholds
            velocities: Current robot velocities (linear_x, linear_y, angular)
            
        Returns:
            tuple: (should_move, reason, importance)
        """
        # Extract errors and thresholds
        distance_error = abs(errors['distance'])
        lateral_error = abs(errors['lateral'])
        angular_error = abs(errors['angular'])
        
        # Apply movement-specific threshold scaling
        # Use smaller thresholds to exit stopped state (easier to start moving)
        adjusted_thresholds = {}
        for key, value in thresholds.items():
            # Apply different multiplier for initial movement
            if self.initial_movement_boost:
                adjusted_thresholds[key] = value * 0.5  # Much lower threshold for first movement
            else:
                adjusted_thresholds[key] = value * self.threshold_multiplier
                
        # Check if any error exceeds its adjusted threshold
        triggers = []
        if distance_error > adjusted_thresholds['distance']:
            triggers.append(f"distance_error={distance_error:.3f}m > threshold={adjusted_thresholds['distance']:.3f}m")
        if lateral_error > adjusted_thresholds['lateral']:
            triggers.append(f"lateral_error={lateral_error:.3f}m > threshold={adjusted_thresholds['lateral']:.3f}m")
        if angular_error > adjusted_thresholds['angular']:
            triggers.append(f"angular_error={angular_error:.2f}° > threshold={adjusted_thresholds['angular']:.2f}°")
        
        # If any error exceeds threshold, recommend movement
        if triggers:
            # Clear initial movement boost flag after first use
            if self.initial_movement_boost:
                self.initial_movement_boost = False
                
            # Create detailed reason message
            reason = "Movement required: " + ", ".join(triggers)
            return True, reason, 10  # High importance to ensure movement
            
        # No errors exceed thresholds, stay stopped
        return False, "All errors within movement thresholds", 5
    
    def should_stop(self, robot_state, errors, thresholds, velocities):
        """
        Implementation for StoppingStrategy interface - delegates to should_move with inverted result.
        
        Returns:
            tuple: (should_stop, reason, importance)
        """
        should_move, reason, importance = self.should_move(robot_state, errors, thresholds, velocities)
        return not should_move, reason, importance
        
    def reset_initial_boost(self):
        """Reset the initial movement boost flag."""
        self.initial_movement_boost = True


class MovementDecisionManager:
    """Manager for movement decisions using multiple strategies."""
    
    def __init__(self, stop_decision_manager):
        """
        Initialize with reference to stop decision manager.
        
        Args:
            stop_decision_manager: StopDecisionManager instance for threshold sharing
        """
        self.stop_decision_manager = stop_decision_manager
        self.movement_strategy = StartMovementStrategy()
        self.stop_duration = 0.0
        self.last_stop_time = 0.0
        
    def evaluate_movement_conditions(self, distance_error, lateral_error, angular_error, 
                                   current_time):
        """
        Evaluate if the robot should start moving based on current conditions.
        
        Args:
            distance_error: Current distance error
            lateral_error: Current lateral error
            angular_error: Current angular error in degrees
            current_time: Current time
            
        Returns:
            tuple: (should_move, reason) - True if robot should move, False otherwise
        """
        # Prepare errors dictionary
        errors = {
            'distance': distance_error,
            'lateral': lateral_error,
            'angular': angular_error
        }
        
        # Calculate stop duration
        self.stop_duration = current_time - self.last_stop_time
        
        # Check if we're at target distance
        at_target_distance = abs(distance_error) < self.stop_decision_manager.threshold_calculator.base_thresholds['distance'] * 1.5
        
        # Calculate thresholds with hysteresis
        # Note: is_stopped=True because this function is only called when stopped
        thresholds = self.stop_decision_manager.threshold_calculator.calculate_thresholds(
            is_stopped=True,
            at_target_distance=at_target_distance,
            angular_at_target_factor=1.5
        )
        
        # Prepare robot state
        robot_state = {
            'is_stopped': True,
            'at_target_distance': at_target_distance,
            'stop_duration': self.stop_duration
        }
        
        # Prepare velocities (should be zero when stopped)
        velocities = {
            'linear_x': 0.0,
            'linear_y': 0.0,
            'angular': 0.0
        }
        
        # Evaluate movement strategy
        should_move, reason, _ = self.movement_strategy.should_move(
            robot_state, errors, thresholds, velocities
        )
        
        return should_move, reason
    
    def record_stop_time(self, current_time):
        """Record the time when the robot stopped."""
        self.last_stop_time = current_time
        
    def reset_initial_boost(self):
        """Reset the initial movement boost flag."""
        self.movement_strategy.reset_initial_boost()


class VelocityLimitingStrategy:
    """Base class for velocity limiting strategies."""
    
    def __init__(self, name, priority=5):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name for logging
            priority: Priority level (higher = applied first)
        """
        self.name = name
        self.priority = priority
        
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """
        Apply velocity limits according to this strategy.
        
        Args:
            velocities: Current commanded velocities [linear_x, linear_y, angular]
            prev_velocities: Previous velocities [linear_x, linear_y, angular]
            robot_state: Dict with robot state info
            dt: Time delta since last update
            
        Returns:
            tuple: (limited_velocities, reason, limiting_applied)
        """
        # Base implementation does nothing
        return velocities, None, False


class AccelerationLimiter(VelocityLimitingStrategy):
    """Limits acceleration to prevent jerky motion with improved angular response."""
    
    def __init__(self, accel_limit=1.5, angular_accel_limit=1.2, priority=5):  # MODIFIED: Increased angular_accel_limit from 2.0 to 3.0
        """Initialize with acceleration limits."""
        super().__init__("AccelerationLimiter", priority)
        self.accel_limit = accel_limit
        self.angular_accel_limit = angular_accel_limit
        
        # ADDED: Separate limits for deceleration to allow faster stopping
        self.decel_limit = accel_limit * 1.5  # 50% higher limit for deceleration
        self.angular_decel_limit = angular_accel_limit * 1.8  
        
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """Apply acceleration limits to all velocity components with enhanced angular handling."""
        # Copy velocities to avoid modifying the original
        limited_velocities = velocities.copy()
        
        # Extract current error information from robot_state if available
        distance_error = robot_state.get('distance_error', 0.0)
        angular_error = robot_state.get('angular_error', 0.0)
        is_approaching = robot_state.get('is_approaching', False)
        
        # ADDED: Conditionally apply asymmetric acceleration limits
        linear_limit = self.accel_limit * dt
        angular_limit = self.angular_accel_limit * dt
        
        # Apply limits to each component
        limited_dirs = []
        for i in range(3):
            # Determine if acceleration or deceleration
            is_decel = (velocities[i] * prev_velocities[i] >= 0 and  # Same direction
                        abs(velocities[i]) < abs(prev_velocities[i]))  # Reducing speed
            
            is_reversal = (velocities[i] * prev_velocities[i] < 0)  # Changing direction
                
            # MODIFIED: Select appropriate limit for this component and condition
            if i == 2:  # Angular component
                # For angular, use different limits based on error
                if abs(angular_error) > 0.1:  # Significant angular error
                    # Boost acceleration when substantial angular error exists
                    angular_boost = min(2.0, 1.0 + abs(angular_error) / 0.2)
                    limit = angular_limit * angular_boost
                    
                    # But if decelerating, use standard limit to prevent oscillation
                    if is_decel and not is_reversal:
                        limit = angular_limit
                else:
                    # Normal limit for small errors
                    limit = angular_limit
                    
                # Handle deceleration case
                if is_decel and not is_reversal:
                    limit = self.angular_decel_limit * dt
            else:
                # Linear component
                if is_decel and not is_reversal:
                    # Use higher limit for deceleration
                    limit = self.decel_limit * dt
                else:
                    # Use standard limit for acceleration
                    limit = linear_limit
            
            # ADDED: Special case for starting from stop
            # If previous velocity was very low but new velocity is significant
            if abs(prev_velocities[i]) < 0.01 and abs(velocities[i]) > 0.05:
                # Higher limit for starting movement (2x normal)
                if i == 2:  # Angular
                    # Even higher boost for angular starting (3x)
                    limit = limit * 1.5
                else:
                    # 2x boost for linear starting
                    limit = limit * 2.0
            
            # Calculate velocity change
            vel_diff = limited_velocities[i] - prev_velocities[i]
            
            # Apply limit if needed
            if abs(vel_diff) > limit:
                if vel_diff > 0:
                    limited_velocities[i] = prev_velocities[i] + limit
                    limited_dirs.append(f"+{['x', 'y', 'a'][i]}")
                else:
                    limited_velocities[i] = prev_velocities[i] - limit
                    limited_dirs.append(f"-{['x', 'y', 'a'][i]}")
        
        # ADDED: Special case for approaching target
        if is_approaching and abs(distance_error) < 0.3:
            # When close to target, apply special case priorities
            if abs(angular_error) > 0.05:  # Still has angular error
                # Allow faster angular correction when close to target
                angular_idx = 2
                angular_vel_diff = velocities[angular_idx] - prev_velocities[angular_idx]
                
                # Don't limit angular velocity if correcting alignment near target
                # Only apply if we're not already at max correction
                if abs(angular_vel_diff) > 0 and abs(limited_velocities[angular_idx]) < 0.8:
                    # Directly use requested angular velocity
                    limited_velocities[angular_idx] = velocities[angular_idx]
                    
                    # Remove angular from limited directions if it was there
                    limited_dirs = [d for d in limited_dirs if not d.endswith('a')]
        
        # Generate reason message if limiting was applied
        if limited_dirs:
            reason = f"Acceleration limited in {', '.join(limited_dirs)}"
            return limited_velocities, reason, True
            
        return limited_velocities, None, False


class EmergencyBrakingLimiter(VelocityLimitingStrategy):
    """Emergency braking to prevent overshooting."""
    
    def __init__(self, priority=20):
        """Initialize with high priority."""
        super().__init__("EmergencyBrakingLimiter", priority)
        
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """Apply emergency braking when necessary."""
        # Extract required state
        distance_error = robot_state.get('distance_error', 0.0)
        
        # Copy velocities to avoid modifying the original
        limited_velocities = velocities.copy()
        reason = None
        limiting_applied = False
        
        # Emergency case 1: Already overshot target
        if distance_error < 0 and limited_velocities[0] > 0.01:
            # Hard limit on forward velocity
            limited_velocities[0] = 0.0
            reason = f"EMERGENCY BRAKE: Overshot target by {abs(distance_error):.3f}m"
            limiting_applied = True
            
        # Emergency case 2: Too fast near target
        elif abs(distance_error) < 0.1 and limited_velocities[0] > 0.15:
            # Severe reduction in forward velocity
            limited_velocities[0] *= 0.3
            reason = f"EMERGENCY BRAKE: Too fast near target ({limited_velocities[0]:.2f}m/s)"
            limiting_applied = True
            
        return limited_velocities, reason, limiting_applied


class LookAheadLimiter(VelocityLimitingStrategy):
    """Implements predictive stopping based on look-ahead distance."""
    
    def __init__(self, deceleration_rate=1.5, safety_buffer=0.05, priority=8):
        """
        Initialize with deceleration parameters.
        
        Args:
            deceleration_rate: Expected deceleration capability (m/s²)
            safety_buffer: Additional safety margin (m)
            priority: Priority level
        """
        super().__init__("LookAheadLimiter", priority)
        self.deceleration_rate = deceleration_rate
        self.safety_buffer = safety_buffer
        
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """Apply look-ahead based velocity limiting."""
        # Extract required state
        distance_error = robot_state.get('distance_error', 0.0)
        
        # Don't apply if moving away or not moving forward
        if distance_error < 0 or velocities[0] <= 0.01:
            return velocities, None, False
        
        # Calculate stopping distance based on current velocity
        stopping_distance = (velocities[0] * velocities[0]) / (2 * self.deceleration_rate)
        
        # Apply limiting if we're getting too close
        if stopping_distance >= abs(distance_error) - self.safety_buffer:
            # Calculate what velocity would be appropriate at this distance
            # Using v = sqrt(2 * a * d) formula, where d is distance to target
            safe_velocity = math.sqrt(2 * self.deceleration_rate * 
                                     max(0.01, abs(distance_error) - self.safety_buffer))
            
            # We never want to exceed current velocity (only reduce)
            safe_velocity = min(safe_velocity, velocities[0])
            
            # Copy velocities for modification
            limited_velocities = velocities.copy()
            limited_velocities[0] = safe_velocity
            
            # Check if we made a significant change
            if velocities[0] - safe_velocity > 0.05:  # 5 cm/s change
                reason = (f"Look-ahead braking: distance={abs(distance_error):.2f}m, "
                         f"stopping_distance={stopping_distance:.2f}m")
                return limited_velocities, reason, True
                
        return velocities, None, False


class ApproachScalingLimiter(VelocityLimitingStrategy):
    """Scales velocity based on approach to target."""
    
    def __init__(self, approach_distance=0.7, min_approach_factor=0.1, priority=7):
        """
        Initialize with approach parameters.
        
        Args:
            approach_distance: Distance to start scaling (m)
            min_approach_factor: Minimum velocity factor when close to target
            priority: Priority level
        """
        super().__init__("ApproachScalingLimiter", priority)
        self.approach_distance = approach_distance
        self.min_approach_factor = min_approach_factor
        
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """Apply approach-based velocity scaling."""
        # Extract required state
        distance_error = robot_state.get('distance_error', 0.0)
        
        # Don't apply if moving away or not moving
        if distance_error < 0:
            return velocities, None, False
            
        # Calculate where we are in the approach zone
        approach_zone = self.approach_distance * 1.5  # Full zone is 1.5x approach distance
        
        # Check if we're in the approach zone
        if abs(distance_error) < approach_zone:
            # Copy velocities for modification
            limited_velocities = velocities.copy()
            
            # Calculate approach factor (1.0 at edge of zone, min_factor at target)
            # This creates a linear scaling from full speed to minimum
            proximity_factor = abs(distance_error) / approach_zone
            scale_factor = self.min_approach_factor + (1.0 - self.min_approach_factor) * proximity_factor
            
            # Apply scaling to forward velocity
            limited_velocities[0] = limited_velocities[0] * scale_factor
            
            # Apply less aggressive scaling to lateral and angular
            lateral_scale = min(1.0, scale_factor * 1.5)  # Less reduction
            limited_velocities[1] = limited_velocities[1] * lateral_scale
            
            angular_scale = min(1.0, scale_factor * 1.8)  # Even less reduction
            limited_velocities[2] = limited_velocities[2] * angular_scale
            
            # Check if we made a significant change
            if abs(1.0 - scale_factor) > 0.1:  # 10% change
                reason = (f"Approach scaling: distance={abs(distance_error):.2f}m, "
                         f"scale={scale_factor:.2f}, zone={approach_zone:.2f}m")
                return limited_velocities, reason, True
                
        return velocities, None, False


class AsymmetricDecelerationLimiter(VelocityLimitingStrategy):
    """Implements asymmetric deceleration for more responsive stopping."""
    
    def __init__(self, accel_factor=1.0, decel_factor=1.5, priority=6):
        """
        Initialize with acceleration/deceleration factors.
        
        Args:
            accel_factor: Factor for acceleration limits
            decel_factor: Factor for deceleration limits (higher = stronger deceleration)
            priority: Priority level
        """
        super().__init__("AsymmetricDecelerationLimiter", priority)
        self.accel_factor = accel_factor
        self.decel_factor = decel_factor
        
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """
        Apply asymmetric deceleration with enhanced angular awareness.
        
        Args:
            velocities: Current commanded velocities [linear_x, linear_y, angular]
            prev_velocities: Previous velocities [linear_x, linear_y, angular]
            robot_state: Dict with robot state info
            dt: Time delta since last update
            
        Returns:
            tuple: (limited_velocities, reason, limiting_applied)
        """
        # Only apply to linear x (forward) velocity and angular velocity
        is_decelerating_x = velocities[0] < prev_velocities[0]
        is_decelerating_angular = (velocities[2] * prev_velocities[2] > 0 and 
                                  abs(velocities[2]) < abs(prev_velocities[2]))
        
        # Get error values from robot state
        distance_error = robot_state.get('distance_error', 0.0)
        angular_error = robot_state.get('angular_error', 0.0)
        
        # Flag for whether limiting was applied
        limiting_applied = False
        reason = None
        
        # Create a copy of velocities for modification
        limited_velocities = velocities.copy()
        
        # Handle forward deceleration
        if is_decelerating_x:
            # Stronger deceleration when close to or past target
            if distance_error < 0.2 or abs(angular_error) < math.radians(7.0):
                # Enhance deceleration factor based on proximity or small angular error
                proximity_boost = 1.0 + max(0, 0.2 - abs(distance_error)) * 10
                
                # Additional boost for small angular errors
                if abs(angular_error) < math.radians(7.0):
                    angular_proximity = abs(angular_error) / math.radians(7.0)
                    angular_boost = 1.0 + (1.0 - angular_proximity) * 1.5
                    proximity_boost = max(proximity_boost, angular_boost)
                    
                decel_boost = self.decel_factor * proximity_boost
                
                # Apply enhanced deceleration to forward velocity
                accel_limit = robot_state.get('accel_limit', 1.5) * decel_boost * dt
                
                # Calculate deceleration
                decel = prev_velocities[0] - velocities[0]
                
                # If deceleration exceeds the enhanced limit
                if decel > accel_limit:
                    # Apply the enhanced limit
                    limited_velocities[0] = prev_velocities[0] - accel_limit
                    
                    reason = (f"Enhanced deceleration near target: "
                             f"boost={decel_boost:.2f}x, distance={abs(distance_error):.2f}m, "
                             f"angular={math.degrees(abs(angular_error)):.2f}°")
                    limiting_applied = True
        
        # Handle angular deceleration
        if is_decelerating_angular:
            # Stronger deceleration when close to target angle
            if abs(angular_error) < math.radians(7.0):
                # Enhance deceleration factor based on angular proximity
                angular_proximity = abs(angular_error) / math.radians(7.0)
                angular_boost = 1.0 + (1.0 - angular_proximity) * 2.0
                decel_boost = self.decel_factor * angular_boost
                
                # Apply enhanced deceleration to angular velocity
                angular_accel_limit = robot_state.get('angular_accel_limit', 0.8) * decel_boost * dt
                
                # Calculate deceleration (maintain sign)
                decel = abs(prev_velocities[2] - velocities[2])
                
                # If deceleration exceeds the enhanced limit
                if decel > angular_accel_limit:
                    # Apply the enhanced limit (preserve direction)
                    sign = 1.0 if velocities[2] >= 0 else -1.0
                    limited_velocities[2] = prev_velocities[2] - sign * angular_accel_limit
                    
                    # Update or append to reason
                    angular_reason = (f"Enhanced angular deceleration: "
                                    f"boost={angular_boost:.2f}x, angular={math.degrees(abs(angular_error)):.2f}°")
                    
                    if reason:
                        reason = reason + "; " + angular_reason
                    else:
                        reason = angular_reason
                        
                    limiting_applied = True
        
        # Return results
        if limiting_applied:
            return limited_velocities, reason, True
            
        return velocities, None, False


class VelocityHysteresisLimiter(VelocityLimitingStrategy):
    """Applies hysteresis to velocity changes to prevent oscillation."""
    
    def __init__(self, hysteresis_band=0.05, priority=4):
        """
        Initialize with hysteresis parameters.
        
        Args:
            hysteresis_band: Minimum velocity change to apply (m/s)
            priority: Priority level
        """
        super().__init__("VelocityHysteresisLimiter", priority)
        self.hysteresis_band = hysteresis_band
        
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """Apply velocity hysteresis."""
        # Check if velocity changes are above hysteresis band
        significant_changes = [False, False, False]
        for i in range(3):
            significant_changes[i] = abs(velocities[i] - prev_velocities[i]) > self.hysteresis_band
            
        # If any component has insignificant change, apply hysteresis
        if not all(significant_changes):
            # Copy velocities for modification
            limited_velocities = velocities.copy()
            
            # Apply hysteresis to each component
            filtered_dirs = []
            for i in range(3):
                if not significant_changes[i]:
                    # Change is below hysteresis band - keep previous velocity
                    limited_velocities[i] = prev_velocities[i]
                    filtered_dirs.append(f"{['x', 'y', 'a'][i]}")
                    
            if filtered_dirs:
                reason = f"Velocity hysteresis applied to {', '.join(filtered_dirs)}"
                return limited_velocities, reason, True
                
        return velocities, None, False


class MinVelocityLimiter(VelocityLimitingStrategy):
    """Enforces minimum effective velocity to prevent very small movements."""
    
    def __init__(self, min_linear=0.01, min_angular=0.01, priority=3):
        """
        Initialize with minimum velocity parameters.
        
        Args:
            min_linear: Minimum linear velocity (m/s)
            min_angular: Minimum angular velocity (rad/s)
            priority: Priority level
        """
        super().__init__("MinVelocityLimiter", priority)
        self.min_linear = min_linear
        self.min_angular = min_angular
        
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """Apply minimum velocity thresholds."""
        # Check if any velocity is below minimum threshold but not zero
        below_min = [False, False, False]
        
        # Check each component
        below_min[0] = 0 < abs(velocities[0]) < self.min_linear
        below_min[1] = 0 < abs(velocities[1]) < self.min_linear
        below_min[2] = 0 < abs(velocities[2]) < self.min_angular
        
        # If any component is below minimum, apply limits
        if any(below_min):
            # Copy velocities for modification
            limited_velocities = velocities.copy()
            
            # Apply minimum thresholds
            zeroed_dirs = []
            for i in range(3):
                if below_min[i]:
                    # Too small to be effective - zero it out
                    limited_velocities[i] = 0.0
                    zeroed_dirs.append(f"{['x', 'y', 'a'][i]}")
                    
            if zeroed_dirs:
                reason = f"Min velocity threshold applied to {', '.join(zeroed_dirs)}"
                return limited_velocities, reason, True
                
        return velocities, None, False


class MaxVelocityLimiter(VelocityLimitingStrategy):
    """Enforces maximum velocity limits."""
    
    def __init__(self, max_linear_x=0.3, max_linear_y=0.3, max_angular=0.7, priority=9):
        """
        Initialize with maximum velocity parameters.
        
        Args:
            max_linear_x: Maximum forward velocity (m/s)
            max_linear_y: Maximum lateral velocity (m/s)
            max_angular: Maximum angular velocity (rad/s)
            priority: Priority level
        """
        super().__init__("MaxVelocityLimiter", priority)
        self.max_velocities = [max_linear_x, max_linear_y, max_angular]
        
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """Apply maximum velocity limits."""
        # Check if any velocity exceeds maximum
        exceeds_max = [False, False, False]
        
        # Check each component
        for i in range(3):
            exceeds_max[i] = abs(velocities[i]) > self.max_velocities[i]
            
        # If any component exceeds maximum, apply limits
        if any(exceeds_max):
            # Copy velocities for modification
            limited_velocities = velocities.copy()
            
            # Apply maximum limits
            limited_dirs = []
            for i in range(3):
                if exceeds_max[i]:
                    # Apply limit with original sign
                    sign = 1 if velocities[i] >= 0 else -1
                    limited_velocities[i] = sign * self.max_velocities[i]
                    limited_dirs.append(f"{['x', 'y', 'a'][i]}")
                    
            if limited_dirs:
                reason = f"Max velocity limit applied to {', '.join(limited_dirs)}"
                return limited_velocities, reason, True
                
        return velocities, None, False


class VelocityLimiterPipeline:
    """Pipeline for applying multiple velocity limiting strategies."""
    
    def __init__(self, debug_level=0):
        """Initialize the pipeline with debugging level."""
        self.strategies = []
        self.debug_level = debug_level
        self.logger = logging.getLogger('pid_controller.velocity_limiter')
        
    def add_strategy(self, strategy):
        """
        Add a velocity limiting strategy to the pipeline.
        
        Args:
            strategy: VelocityLimitingStrategy instance
        """
        self.strategies.append(strategy)
        
        # Sort strategies by priority (highest first)
        self.strategies.sort(key=lambda s: s.priority, reverse=True)
        
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """
        Apply all strategies in priority order.
        
        Args:
            velocities: Current commanded velocities [linear_x, linear_y, angular]
            prev_velocities: Previous velocities [linear_x, linear_y, angular]
            robot_state: Dict with robot state info
            dt: Time delta since last update
            
        Returns:
            tuple: (limited_velocities, reasons)
        """
        # Convert to numpy array for efficient operations
        current_velocities = np.array(velocities, dtype=np.float32)
        
        # Track reasons for limiting
        reasons = []
        
        # Apply each strategy in priority order
        for strategy in self.strategies:
            # Apply this strategy
            limited_velocities, reason, limiting_applied = strategy.apply_limits(
                current_velocities, prev_velocities, robot_state, dt
            )
            
            # Update current velocities for next strategy
            if limiting_applied:
                current_velocities = limited_velocities
                reasons.append(reason)
                
                # Log at debug level
                if self.debug_level >= 2:
                    self.logger.debug(f"Strategy {strategy.name} applied: {reason}")
        
        # Convert back to list
        result_velocities = current_velocities.tolist()
        
        return result_velocities, reasons


class VelocityLimiterManager:
    """Manager for all velocity limiting functionality."""
    
    def __init__(self, robot_params, debug_level=0):
        """
        Initialize the velocity limiter manager.
        
        Args:
            robot_params: Dictionary with robot parameters
            debug_level: Debug level for logging
        """
        self.pipeline = VelocityLimiterPipeline(debug_level)
        self.debug_level = debug_level
        self.logger = logging.getLogger('pid_controller.velocity_limiter')
        
        # Extract parameters
        self.approach_distance = robot_params.get('approach_distance', 0.7)
        self.min_approach_factor = robot_params.get('min_approach_factor', 0.1)
        self.linear_x_max = robot_params.get('linear_x_max', 0.3)
        self.linear_y_max = robot_params.get('linear_y_max', 0.3)
        self.angular_max = robot_params.get('angular_max', 0.7)
        self.accel_limit = robot_params.get('accel_limit', 1.5)
        self.angular_accel_limit = robot_params.get('angular_accel_limit', 0.8)
        
        # Initialize strategies
        self._init_strategies()
        
    def _init_strategies(self):
        """Initialize all velocity limiting strategies."""
        # Add strategies in priority order
        
        # 1. Maximum velocity limits (highest priority)
        self.pipeline.add_strategy(MaxVelocityLimiter(
            self.linear_x_max, self.linear_y_max, self.angular_max
        ))
        
        # 2. Emergency braking
        self.pipeline.add_strategy(EmergencyBrakingLimiter())
        
        # 3. Look-ahead limiting
        self.pipeline.add_strategy(LookAheadLimiter())
        
        # 4. Approach scaling
        self.pipeline.add_strategy(ApproachScalingLimiter(
            self.approach_distance, self.min_approach_factor
        ))
        
        # 5. Asymmetric deceleration
        self.pipeline.add_strategy(AsymmetricDecelerationLimiter())
        
        # 6. Standard acceleration limiting
        self.pipeline.add_strategy(AccelerationLimiter(
            self.accel_limit, self.angular_accel_limit
        ))
        
        # 7. Velocity hysteresis
        self.pipeline.add_strategy(VelocityHysteresisLimiter())
        
        # 8. Minimum velocity threshold
        self.pipeline.add_strategy(MinVelocityLimiter())
        
    def apply_velocity_limits(self, target_velocities, prev_velocities, distance_error, 
                             is_approaching=False, dt=0.1):
        """
        Apply velocity limits to target velocities.
        
        Args:
            target_velocities: Target velocities [linear_x, linear_y, angular]
            prev_velocities: Previous velocities [linear_x, linear_y, angular]
            distance_error: Error in distance to target
            is_approaching: Whether robot is approaching target
            dt: Time delta since last update
            
        Returns:
            tuple: (limited_velocities, limited_components)
        """
        # Prepare robot state dictionary
        robot_state = {
            'distance_error': distance_error,
            'is_approaching': is_approaching,
            'accel_limit': self.accel_limit
        }
        
        # Apply all limiting strategies
        limited_velocities, reasons = self.pipeline.apply_limits(
            target_velocities, prev_velocities, robot_state, dt
        )
        
        # Log limiting if any occurred
        if reasons and self.debug_level >= 1:
            # Combine the top two reasons
            reason_msg = "; ".join(reasons[:2])
            self.logger.info(f"Velocity limiting: {reason_msg}")
            
            # Determine which components were limited
            limited_components = []
            for i, (target, limited) in enumerate(zip(target_velocities, limited_velocities)):
                if abs(target - limited) > 0.01:
                    pct = abs(target - limited) / max(0.01, abs(target)) * 100
                    limited_components.append(f"{['x', 'y', 'a'][i]}:{pct:.0f}%")
        else:
            limited_components = []
            
        return limited_velocities, limited_components

class ImprovedPIDControllerNode(Node):
    """Enhanced PID Controller node with improved movement strategy and error handling."""
    
    def get_logger(self):
        return get_logger('pid_controller')
    
    def __init__(self):
        """Initialize the enhanced PID controller node with comprehensive logging."""
        init_start_time = time.time()
        
        # Initialize the ROS node
        super().__init__('pid_controller')
        
        # Log initialization start using basic logging (no structured logging yet)
        self.get_logger().info("Starting PID controller initialization")
        
        

        # Set up callback group BEFORE any other operations that might need it
        self.callback_group = ReentrantCallbackGroup()
        
        try:
            # Set up parameters FIRST so debug_level is available
            start_time = time.time()
            self._declare_parameters()
            elapsed = time.time() - start_time

            # Initialize the global logging system FIRST
            init_logging_system(self, self.debug_level, self.log_verbosity)

            # Now set up logging with debug_level available
            start_time = time.time()
            self._init_logging()
            elapsed = time.time() - start_time
            log_initialization_step("pid_controller", "logging_setup", "completed", elapsed)
            
            # Now we can use structured logging for earlier steps
            log_initialization_step("pid_controller", "node_initialization", "started", None)
            log_initialization_step("parameters", "declaration", "completed", elapsed)
            log_initialization_step("callback_group", "setup", "completed", 0.0)

            # Initialize resource monitoring
            start_time = time.time()
            self._init_resource_monitoring()
            elapsed = time.time() - start_time
            log_initialization_step("resource_monitor", "initialization", "completed", elapsed)
            
            # Initialize memory pools for performance
            start_time = time.time()
            self._init_memory_pools()
            elapsed = time.time() - start_time
            log_initialization_step("memory_pools", "initialization", "completed", elapsed)
            
            # Initialize controllers
            start_time = time.time()
            self._init_controllers()
            elapsed = time.time() - start_time
            log_initialization_step("controllers", "initialization", "completed", elapsed)
            
            # Set up state variables
            start_time = time.time()
            self._init_state_variables()
            elapsed = time.time() - start_time
            log_initialization_step("state_variables", "initialization", "completed", elapsed)
            
            # Set up tf2 system
            start_time = time.time()
            self._setup_tf2()
            elapsed = time.time() - start_time
            log_initialization_step("tf2_system", "setup", "completed", elapsed)
            
            # Set up subscriptions
            start_time = time.time()
            self._setup_subscriptions()
            elapsed = time.time() - start_time
            log_initialization_step("subscriptions", "setup", "completed", elapsed)
            
            # Set up publishers
            start_time = time.time()
            self._setup_publishers()
            elapsed = time.time() - start_time
            log_initialization_step("publishers", "setup", "completed", elapsed)
            
            # Set up timers
            start_time = time.time()
            self._setup_timers()
            elapsed = time.time() - start_time
            log_initialization_step("timers", "setup", "completed", elapsed)
            
            # Define movement strategies
            start_time = time.time()
            self._init_strategy_table()
            elapsed = time.time() - start_time
            log_initialization_step("strategy_table", "initialization", "completed", elapsed)
            
            # Initialize target filter
            start_time = time.time()
            self.target_filter = EnhancedTargetFilter(buffer_size=5, prediction_horizon=0.2)
            elapsed = time.time() - start_time
            log_initialization_step("target_filter", "initialization", "completed", elapsed)
            
            # Initialize strategy blender
            start_time = time.time()
            self.strategy_blender = StrategyBlender(blend_duration=0.05)
            elapsed = time.time() - start_time
            log_initialization_step("strategy_blender", "initialization", "completed", elapsed)
            
            self._initialize_decision_managers()

            # Validate parameters were properly applied
            start_time = time.time()
            validation_success = self._validate_parameters()
            status = "success" if validation_success else "warnings_found"
            elapsed = time.time() - start_time
            log_initialization_step("parameters", "validation", status, elapsed)
            
            if not validation_success:
                log_structured("initialization", "VALIDATION_WARNING",
                            "Parameter validation issues detected during initialization",
                            level=logging.WARNING)
                self.get_logger().warning("Parameter validation failed during initialization")
            
            # Log total initialization time
            total_init_time = time.time() - init_start_time
            log_initialization_step("pid_controller", "node_initialization", "completed", total_init_time)
            
            # Log welcome message and version info
            important_params = {
                'angular_first': self.angular_first_control,
                'coordinated_movement': self.coordinated_movement,
                'approach_distance': self.approach_distance,
                'adaptive_control': self.adaptive_control_rate,
                'update_rate': self.update_rate
            }
            
            log_structured("initialization", "NODE_READY",
                        "Improved PID Controller initialized with angular-first strategy",
                        important_params)
            
            # Log startup info
            self.get_logger().info("Improved PID Controller initialized with angular-first strategy")
            
        except Exception as e:
            # Log initialization failure
            self.get_logger().error(f"Initialization failed: {str(e)}")
            # Try to use structured logging if available
            try:
                log_structured("initialization", "INIT_FAILURE",
                            f"Failed to initialize PID controller: {str(e)}",
                            {'component': traceback.extract_tb(sys.exc_info()[2])[-1].name},
                            level=logging.ERROR)
            except Exception as log_error:
                # If structured logging failed, just use basic logging
                self.get_logger().error(f"Failed to initialize PID controller: {str(e)}")
                self.get_logger().error(f"Failed to log error: {str(log_error)}")
            raise

    def _initialize_decision_managers(self):
        """Initialize the decision managers for stop and movement decisions if not already done."""
        if not hasattr(self, '_stop_decision_manager'):
            # Create threshold calculator
            base_thresholds = {
                'distance': self.distance_threshold,
                'lateral': self.lateral_threshold,
                'angular': self.angular_threshold  # In degrees
            }
            threshold_calculator = ThresholdCalculator(base_thresholds, self.debug_level)
            
            # Create stop decision manager
            stop_manager = StopDecisionManager(threshold_calculator, self.debug_level)
            
            # Add strategies in priority order (highest priority first)
            stop_manager.add_strategy(BrakeIfOvershootingStrategy())  # Highest priority
            stop_manager.add_strategy(DistancePriorityStrategy())     # Second priority
            stop_manager.add_strategy(VelocityAwareStrategy())        # Third priority
            stop_manager.add_strategy(AlignmentStrategy())            # Lowest priority
            
            # Store for future use
            self._stop_decision_manager = stop_manager
            
            # Create movement decision manager that shares threshold calculator
            self._movement_decision_manager = MovementDecisionManager(stop_manager)
            
            # Initialize stop time
            self._movement_decision_manager.record_stop_time(time.time())

    def _validate_parameters(self):
        """
        Validate that parameters were correctly applied to their target objects.
        
        This method checks that configuration values were properly propagated
        to the appropriate components and logs any discrepancies found.
        """
        log_structured('parameter_validator', 'VALIDATION_START', 
                    "Starting parameter validation",
                    {'debug_level': self.debug_level})
        
        validation_results = {
            'passed': 0,
            'failed': 0,
            'params_checked': []
        }
        
        # Strategy blender parameters
        self._validate_param('strategy_blend_duration', 'strategy_blender.blend_duration', 
                        self.get_parameter('strategy_blend_duration').value,
                        self.strategy_blender.blend_duration,
                        validation_results)
        
        self._validate_param('min_hold_time', 'strategy_blender.min_hold_time', 
                        self.get_parameter('strategy_blend_duration').value * 1.5,  # Default should be 1.5x blend duration
                        self.strategy_blender.min_hold_time,
                        validation_results)
        
        # Target filter parameters
        self._validate_param('prediction_horizon', 'target_filter.prediction_horizon', 
                        self.get_parameter('prediction_horizon').value,
                        self.target_filter.prediction_horizon,
                        validation_results)
        
        self._validate_param('filter_buffer_size', 'target_filter.buffer_size', 
                        self.get_parameter('filter_buffer_size').value,
                        self.target_filter.buffer_size,
                        validation_results)
        
        # Check consistency threshold
        self._validate_param('consistency_threshold', 'target_filter.consistency_threshold', 
                        0.7,  # Expected value after changes
                        self.target_filter.consistency_threshold,
                        validation_results)
        
        # Check min velocity threshold
        self._validate_param('min_velocity_for_prediction', 'target_filter.min_velocity_for_prediction', 
                        0.1,  # Expected value after changes
                        self.target_filter.min_velocity_for_prediction,
                        validation_results)
        
        # PID controller parameters
        pid_controllers = [
            ('linear_x_kp', 'pid_linear_x.base_kp', self.get_parameter('linear_x_kp').value, self.pid_linear_x.base_kp),
            ('linear_x_ki', 'pid_linear_x.base_ki', self.get_parameter('linear_x_ki').value, self.pid_linear_x.base_ki),
            ('linear_x_kd', 'pid_linear_x.base_kd', self.get_parameter('linear_x_kd').value, self.pid_linear_x.base_kd),
            ('linear_y_kp', 'pid_linear_y.base_kp', self.get_parameter('linear_y_kp').value, self.pid_linear_y.base_kp),
            ('linear_y_ki', 'pid_linear_y.base_ki', self.get_parameter('linear_y_ki').value, self.pid_linear_y.base_ki),
            ('linear_y_kd', 'pid_linear_y.base_kd', self.get_parameter('linear_y_kd').value, self.pid_linear_y.base_kd),
            ('angular_kp', 'pid_angular.base_kp', self.get_parameter('angular_kp').value, self.pid_angular.base_kp),
            ('angular_ki', 'pid_angular.base_ki', self.get_parameter('angular_ki').value, self.pid_angular.base_ki),
            ('angular_kd', 'pid_angular.base_kd', self.get_parameter('angular_kd').value, self.pid_angular.base_kd)
        ]
        
        for param_name, target_path, expected, actual in pid_controllers:
            self._validate_param(param_name, target_path, expected, actual, validation_results)
        
        # Velocity limits
        self._validate_param('linear_x_max', 'pid_linear_x.output_max', 
                        self.get_parameter('linear_x_max').value,
                        self.pid_linear_x.output_max,
                        validation_results)
        
        # Integral decay rates - check if properly applied after changes
        self._validate_param('linear_x_integral_decay', 'pid_compute_jit parameter',
                        0.4,  # Expected value after changes
                        getattr(self.pid_linear_x, 'integral_decay', None),  
                        validation_results)
        
        self._validate_param('linear_y_integral_decay', 'pid_compute_jit parameter',
                        0.5,  # Expected value after changes
                        getattr(self.pid_linear_y, 'integral_decay', None),
                        validation_results)
        
        self._validate_param('angular_integral_decay', 'pid_compute_jit parameter',
                        0.6,  # Expected value after changes
                        getattr(self.pid_angular, 'integral_decay', None),
                        validation_results)
        
        # Check coordinated controller settings
        if hasattr(self, 'coordinated_controller'):
            coupling_factor = self.coordinated_controller.config.get('coupling_factor', None)
            if coupling_factor is not None:
                self._validate_param('coupling_factor', 'coordinated_controller.config',
                                0.3,  # Expected value after changes
                                coupling_factor,
                                validation_results)

        # Log validation summary
        log_structured('parameter_validator', 'VALIDATION_COMPLETE', 
                    f"Parameter validation complete: {validation_results['passed']} passed, {validation_results['failed']} failed",
                    {'passed': validation_results['passed'],
                    'failed': validation_results['failed'],
                    'failed_params': [p for p in validation_results['params_checked'] if not p['passed']]})
        
        if validation_results['failed'] > 0:
            # Log a warning about failed validations
            failed_params = [p['param_name'] for p in validation_results['params_checked'] if not p['passed']]
            log_structured('parameter_validator', 'VALIDATION_WARNING',
                        f"Some parameters failed validation: {', '.join(failed_params)}",
                        {'failed_params': failed_params},
                        level=logging.WARNING)
            
            # If any strategic parameters failed, try to diagnose the issue
            strategic_params = ['strategy_blend_duration', 'min_hold_time', 'prediction_horizon', 'consistency_threshold']
            strategic_failures = [p for p in failed_params if p in strategic_params]
            
            if strategic_failures:
                log_structured('parameter_validator', 'STRATEGIC_PARAM_FAILURE',
                            "Strategic parameters failed validation - check initialization sequence",
                            {'strategic_failures': strategic_failures},
                            level=logging.WARNING)

        return validation_results['failed'] == 0  # Return True if all validations passed

    def _validate_param(self, param_name, target_path, expected_value, actual_value, results):
        """
        Validate a single parameter and log the result.
        
        Args:
            param_name: Name of the parameter
            target_path: Path to where the parameter should be applied
            expected_value: Expected value of the parameter
            actual_value: Actual value found
            results: Dictionary to track validation results
        """
        # Handle expected float precision/comparison issues
        if isinstance(expected_value, float) and isinstance(actual_value, float):
            match = abs(expected_value - actual_value) < 0.0001
        else:
            match = expected_value == actual_value
        
        # Log the validation result
        param_result = {
            'param_name': param_name,
            'target_path': target_path,
            'expected': expected_value,
            'actual': actual_value,
            'passed': match
        }
        
        results['params_checked'].append(param_result)
        
        if match:
            results['passed'] += 1
            
            # Only log successful validations at debug level to reduce verbosity
            if self.debug_level >= 2:
                log_parameter_validation(target_path, param_name, expected_value, actual_value)
        else:
            results['failed'] += 1
            
            # Always log failed validations
            log_parameter_validation(target_path, param_name, expected_value, actual_value)
            
            # Provide additional diagnostic information for critical parameters
            if param_name in ['strategy_blend_duration', 'prediction_horizon', 'min_velocity_for_prediction']:
                log_structured('parameter_validator', 'CRITICAL_PARAM_MISMATCH',
                            f"Critical parameter {param_name} mismatch in {target_path}",
                            {'expected': expected_value, 
                            'actual': actual_value,
                            'difference': expected_value - actual_value if isinstance(expected_value, (int, float)) else "N/A"},
                            level=logging.WARNING)
    
    def _init_logging(self):
        """Initialize enhanced logging system with reduced duplication."""
        # Configure base logger
        root_logger = logging.getLogger('pid_controller')
        
        # Configure format
        formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] %(message)s')
        
        # Set up handlers
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Add file handler for persistent logs
        file_handler = logging.FileHandler('pid_controller_detailed.log')
        file_handler.setFormatter(formatter)
        
        # Add handlers
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        # Set level based on debug parameter if it exists, otherwise use INFO level
        debug_level = getattr(self, 'debug_level', 1)  # Default to level 1 if not set yet
        root_logger.setLevel(logging.DEBUG if debug_level >= 2 else logging.INFO)
        
        self.logger = root_logger
        
        # Only use ROS logger for this message to avoid duplication
        self.get_logger().info("Enhanced logging system initialized")

    def _init_resource_monitoring(self):
        """Initialize the resource monitor for CPU and memory tracking with improved logging."""
        self.resource_monitor = ResourceMonitor(update_interval=5.0)
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.current_cpu_usage = 0.0
        self.performance_stats = {
            'cpu': deque(maxlen=12),  # 1 minute of 5-second samples
            'control_cycles': deque(maxlen=100),  # Last 100 control cycle times
            'control_skips': 0  # Count of skipped control cycles due to high CPU
        }
        
        # Performance adjustment parameters
        self.base_update_rate = 20.0  # Default 20Hz control rate
        self.adaptive_control_rate = True
        self.min_update_rate = 8.0   # Don't go below 8Hz
        self.max_update_rate = 20.0  # Don't go above 20Hz
        
        # Performance metrics tracking
        self.cycle_start_time = 0.0
        self.cycle_duration_avg = 0.0
        self.skip_next_cycle = False
        
        # Last logged status time to prevent duplicates
        self.last_resource_log_time = 0
        
        # Log initial resource configuration
        log_resource_usage(0.0, 0.0, None, None)
        
        log_structured('resource_monitor', 'MONITOR_INIT', 
                    "Resource monitor initialized",
                    {'update_interval': 5.0,
                    'base_rate': self.base_update_rate,
                    'min_rate': self.min_update_rate,
                    'max_rate': self.max_update_rate,
                    'adaptive_rate': self.adaptive_control_rate})
        
    def _declare_parameters(self):
        """Declare and get all node parameters with improved defaults."""
        # Most parameter declarations are omitted for brevity but would be here
        
        # Declare parameters with improved defaults
        self.declare_parameters(
            namespace='',
            parameters=[
                # Linear X velocity PID parameters - adjusted for moderate velocity increase
                ('linear_x_kp', 1.2),  # controls overshoot
                ('linear_x_ki', 0.05), # handle steady state errors
                ('linear_x_kd', 0.25), # controls dampening during approach
                ('linear_x_min', 0.0),
                ('linear_x_max', 0.3),  # controls how fast the robot moves forward. 
                
                # Linear Y velocity PID parameters - improved lateral damping
                ('linear_y_kp', 0.15),
                ('linear_y_ki', 0.06),  # Reduced from 0.08
                ('linear_y_kd', 0.4),  # Increased from 0.12
                ('linear_y_min', -0.2),
                ('linear_y_max', 0.3),
                
                # Angular velocity PID parameters - improved to prevent overshoot
                ('angular_kp', 0.9),  # Reduced from 1.5
                ('angular_ki', 0.1), # Reduced from 0.05
                ('angular_kd', 0.8),   # Reduced from 0.8
                ('angular_min', -0.5),
                ('angular_max', 0.9),
                
                # Control parameters
                ('min_distance', 0.9),
                ('max_distance', 2.0),
                ('target_offset_x', 0.0),
                ('target_offset_y', 0.0),
                ('target_update_rate', 20.0),
                ('diagnostics_rate', 0.5),
                ('debug_level', 1),
                ('adaptive_gains', True),
                ('use_lateral_control', True),
                
                # Balanced error thresholds - increased angular threshold
                ('distance_threshold', 0.08),
                ('lateral_threshold', 0.06),  # Increased from 0.05
                ('angular_threshold', 2.5),    # Increased from 1.5 to 3.0 degrees
                
                # New parameter for scaling angular threshold when at target distance
                ('angular_at_target_factor', 2.5),  # Multiply threshold by this when at target distance
                
                # Resource monitoring parameters
                ('adaptive_control_rate', True),
                ('enable_resource_monitoring', True),
                ('cpu_high_threshold', 85.0),
                ('cpu_low_threshold', 40.0),
                
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
                ('approach_distance', 0.7),    # Distance at which to start slowing down
                ('min_approach_factor', 0.1),  # Minimum velocity factor when very close
                ('log_verbosity', 1),  # 0=minimal, 1=normal, 2=verbose
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
        self.update_rate = self.get_parameter('target_update_rate').value
        
        # All other parameter assignments would be here
        self.diagnostics_rate = self.get_parameter('diagnostics_rate').value
        self.debug_level = self.get_parameter('debug_level').value

        self.log_verbosity = self.get_parameter('log_verbosity').value
        
        # Log important parameters
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
               
    def _init_controllers(self):
        """Initialize the controllers with improved tuning for angular response."""
        try:
            log_initialization_step("controllers", "initialization_start", "started", None)
            
            # Create error trackers
            start_time = time.time()
            self.distance_error_tracker = ErrorTracker("distance", max_history=8)
            self.lateral_error_tracker = ErrorTracker("lateral", max_history=8)
            self.angular_error_tracker = ErrorTracker("angular", max_history=8)
            elapsed = time.time() - start_time
            log_initialization_step("error_trackers", "creation", "completed", elapsed)
            
            # Initialize linear X PID controller
            start_time = time.time()
            self.pid_linear_x = ImprovedPID(
                self.linear_x_kp, self.linear_x_ki, self.linear_x_kd,
                self.linear_x_min, self.linear_x_max,
                name="Linear X"
            )
            self.pid_linear_x.error_tracker = self.distance_error_tracker
            self.pid_linear_x.debug_level = self.debug_level
            elapsed = time.time() - start_time
            log_initialization_step("pid_controller", "linear_x_init", "completed", elapsed)
            
            # Log PID parameters for linear X
            log_structured("controllers", "PID_LINEAR_X_CONFIG",
                        "Linear X PID controller configured",
                        {'kp': self.linear_x_kp, 'ki': self.linear_x_ki, 'kd': self.linear_x_kd,
                        'min': self.linear_x_min, 'max': self.linear_x_max})
            
            # Initialize linear Y PID controller
            start_time = time.time()
            self.pid_linear_y = ImprovedPID(
                self.linear_y_kp, self.linear_y_ki, self.linear_y_kd,
                self.linear_y_min, self.linear_y_max,
                name="Linear Y"
            )
            self.pid_linear_y.error_tracker = self.lateral_error_tracker
            self.pid_linear_y.debug_level = self.debug_level
            elapsed = time.time() - start_time
            log_initialization_step("pid_controller", "linear_y_init", "completed", elapsed)
            
            # Log PID parameters for linear Y
            log_structured("controllers", "PID_LINEAR_Y_CONFIG",
                        "Linear Y PID controller configured",
                        {'kp': self.linear_y_kp, 'ki': self.linear_y_ki, 'kd': self.linear_y_kd,
                        'min': self.linear_y_min, 'max': self.linear_y_max})
            
            # Initialize angular PID controller with IMPROVED parameters
            start_time = time.time()
            
            angular_kp = 0.7
            
            angular_ki = 0.01

            angular_kd = 0.8
            
            angular_min = -0.5  
            angular_max = 0.7   
            
            self.pid_angular = ImprovedPID(
                angular_kp, angular_ki, angular_kd,
                angular_min, angular_max,
                name="Angular"
            )
            self.pid_angular.error_tracker = self.angular_error_tracker
            self.pid_angular.debug_level = self.debug_level
            elapsed = time.time() - start_time
            log_initialization_step("pid_controller", "angular_init", "completed", elapsed)
            
            # Log modified PID parameters for angular
            log_structured("controllers", "PID_ANGULAR_CONFIG",
                        "Angular PID controller configured with improved parameters",
                        {'kp': angular_kp, 'ki': angular_ki, 'kd': angular_kd,
                        'min': angular_min, 'max': angular_max,
                        'modified': True})
            
            # Set parent controller references for context
            self.pid_linear_x.pid_controller = self
            self.pid_linear_y.pid_controller = self
            self.pid_angular.pid_controller = self
            
            # Initialize coordinated controller with IMPROVED parameters
            start_time = time.time()
            coord_config = {
                'coupling_factor': 0.3,       # Reduced from 0.3 to reduce lateral overcorrection during turning
                'smoothing_factor': 0.6,      # Reduced from 0.7 for faster transitions
                'min_angle_for_reduction': 0.05, # Reduced threshold
                'zero_angle_threshold': 0.02,
                'max_angle_factor': 0.24,      # Reduced from 0.25
                'same_sign_scale': 0.6,       # Reduced from 0.8
                'opposite_sign_scale': 1.1,   # Reduced from 1.2
            }
            
            self.coordinated_controller = CoordinatedController(
                self.pid_linear_y, 
                self.pid_angular,
                coord_config
            )
            elapsed = time.time() - start_time
            log_initialization_step("controllers", "coordinated_init", "completed", elapsed)
            
            # Log coordinated controller configuration
            log_structured("controllers", "COORDINATED_CONFIG",
                        "Coordinated controller configured with improved parameters",
                        {'coupling_factor': coord_config['coupling_factor'],
                        'smoothing_factor': coord_config['smoothing_factor'],
                        'min_angle': coord_config['min_angle_for_reduction'],
                        'same_sign_scale': coord_config['same_sign_scale'],
                        'opposite_sign_scale': coord_config['opposite_sign_scale'],
                        'modified': True})
            
            # Update instance variables to match modified parameters
            self.angular_min = angular_min
            self.angular_max = angular_max
            
            log_initialization_step("controllers", "all_controllers_init", "completed", None)
            
        except Exception as e:
            # Log initialization failure for controllers
            log_structured("controllers", "CONTROLLER_INIT_FAILURE",
                        f"Failed to initialize controllers: {str(e)}",
                        {'component': traceback.extract_tb(sys.exc_info()[2])[-1].name},
                        level=logging.ERROR)
            log_initialization_step("controllers", "all_controllers_init", "failed", None)
            raise
        
    def _init_memory_pools(self):
        """Setup memory pools and reusable objects for efficiency."""
        # Twist message pool
        self.twist_pool = []
        for _ in range(5):  # Pre-allocate 5 twist messages
            self.twist_pool.append(Twist())
        
        # Vector3 pool
        self.vector3_pool = []
        for _ in range(10):  # Pre-allocate 10 vector3 messages
            self.vector3_pool.append(Vector3())
        
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
        self._strategy_dict = {
            "strategy_name": "",
            "use_forward": False,
            "use_lateral": False,
            "use_angular": False,
            "forward_scale": 0.0,
            "lateral_scale": 0.0,
            "angular_scale": 0.0,
            "reason": ""
        }
        self._key_tuple = ["none", "none", "none"]  # Use list instead of tuple for mutability

        
        # Pre-allocated velocity tuple
        self._velocity_tuple = [0.0, 0.0, 0.0]
        
        # Pre-allocated velocity change check
        self._velocity_change_check = [False, False, False]
        
        # Pre-allocated error container
        self._current_errors = [0.0, 0.0, 0.0]  # distance, lateral, angular
    
    def get_twist_from_pool(self):
        """Get a Twist message from the pool."""
        if not self.twist_pool:
            # If pool is empty, create a new one
            return Twist()
        
        # Pop one from the pool
        twist = self.twist_pool.pop()
        
        # Clear all fields
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        
        return twist
    
    def return_twist_to_pool(self, twist):
        """Return a Twist to the pool if below capacity."""
        if len(self.twist_pool) < 10:
            self.twist_pool.append(twist)
    
    def get_vector3_from_pool(self):
        """Get a Vector3 from the pool."""
        if not self.vector3_pool:
            # If pool is empty, create a new one
            return Vector3()
        
        # Pop one from the pool
        vector = self.vector3_pool.pop()
        
        # Clear all fields
        vector.x = 0.0
        vector.y = 0.0
        vector.z = 0.0
        
        return vector
    
    def return_vector3_to_pool(self, vector):
        """Return a Vector3 to the pool if below capacity."""
        if len(self.vector3_pool) < 10:
            self.vector3_pool.append(vector)
    
    def _init_state_variables(self):
        """Initialize all state tracking variables with improved logging."""
        try:
            log_initialization_step("state_variables", "initialization_start", "started", None)
            
            # Target tracking
            self.current_target = None
            self.last_target_time = None
            
            # Robot state
            self.robot_state = "initializing"
            self.previous_state = None
            self.last_control_time = time.time()
            
            # Robot orientation
            self.robot_orientation = 0.0  # Current yaw in radians
            self.last_orientation_time = None  # Time of last orientation update
            
            # Log throttling timestamps
            self.last_control_log_time = 0.0
            self.last_state_log_time = 0.0
            self.last_diag_log_time = 0.0
            self.last_status_log_time = 0.0
            self.last_velocity_log_time = 0.0
            
            # Derived values
            self.current_distance = 0.0
            self.current_bearing = 0.0
            self.current_lateral = 0.0
            
            # Filtered values
            self.filtered_distance = 0.0
            self.filtered_lateral = 0.0
            self.filtered_bearing = 0.0
            
            # Motion smoothing
            self.last_cmd_vel = (0.0, 0.0, 0.0)
            self.last_logged_cmd = (0.0, 0.0, 0.0)
            
            # Diagnostic information
            self.cycle_count = 0
            
            # Velocity history
            start_time = time.time()
            self.velocity_history = LightweightBuffer(max_size=6)
            elapsed = time.time() - start_time
            log_initialization_step("state_variables", "velocity_history_buffer", "created", elapsed)
            
            # Calculate and set desired distance based on loaded parameters
            if hasattr(self, 'min_distance') and hasattr(self, 'max_distance'):
                self.desired_distance = (self.min_distance + self.max_distance) / 2.0
            else:
                # Default if parameters aren't loaded
                self.desired_distance = 1.0
                log_structured("state_variables", "DEFAULT_DISTANCE",
                            "Using default desired distance",
                            {'value': self.desired_distance},
                            level=logging.WARNING)
            
            # Stopped state tracking with hysteresis
            self._robot_stopped = True  # Start in stopped state
            self._stop_time = time.time()
            self._last_stop_position = (0.0, 0.0, 0.0)
            self._movement_hysteresis = 0.0
            
            # Movement strategy
            self.current_strategy = "IDLE"
            self.previous_strategy = None
            self.strategy_change_time = time.time()
            self.active_strategy = None  # Holds the current strategy object
            
            # Error categorization state
            self.prev_distance_category = "none"
            self.prev_lateral_category = "none"
            self.prev_angular_category = "none"
            
            # Recovery state tracking
            self.in_recovery = False
            self.recovery_start_time = 0.0
            self.recovery_phase = "none"  # none, stop, orient, approach
            self.force_target_reacquisition = False
            
            # Flag to track if we're shutting down
            self._shutting_down = False
            
            # Setup pre-allocation for error calculation
            self._current_errors = [0.0, 0.0, 0.0]  # distance, lateral, angular
            
            # Log successful initialization
            log_initialization_step("state_variables", "all_variables", "initialized", None)
            
            log_structured("state_variables", "INITIAL_STATE",
                        "State variables initialized",
                        {'desired_distance': self.desired_distance,
                        'robot_state': self.robot_state,
                        'robot_stopped': self._robot_stopped})
            
        except Exception as e:
            # Log initialization failure
            log_structured("state_variables", "INIT_FAILURE",
                        f"Failed to initialize state variables: {str(e)}",
                        {'function': '_init_state_variables'},
                        level=logging.ERROR)
            log_initialization_step("state_variables", "all_variables", "failed", None)
            raise
    
    def _setup_tf2(self):
        """Set up tf2 components for coordinate transformations."""
        # Create the buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Define frame IDs
        self.reference_frame = "base_link"
        self.imu_frame = "imu_link"
        
        # Define transform parameters
        self.transform_timeout = 0.1  # seconds
        
        # Set up transform verification
        self.transform_verified = False
        self.transform_check_timer = self.create_timer(2.0, self._verify_transform)
        
        # Set up transform caching
        self.transform_cache = TTLDict(default_ttl=self.transform_cache_ttl)
        self.matrix_cache = TTLDict(default_ttl=5.0)  # 5-second TTL for matrices
        
        # Set up matrix transform optimization
        self.use_matrix_transforms = self.enable_transform_caching
        
        # Pre-compute common transforms
        self.common_transforms_computed = False
        self.transform_check_timer_matrix = self.create_timer(5.0, self._cache_common_transforms)
    
    def _verify_transform(self):
        """Verify that all required transforms are available."""
        try:
            # Check transform between reference frame and IMU frame
            if self.tf_buffer.can_transform(
                self.reference_frame,
                self.imu_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            ):
                self.transform_verified = True
                self.get_logger().info(f"Transform verification successful between {self.reference_frame} and {self.imu_frame}")
                self.transform_check_timer.cancel()
                return
        except Exception:
            pass
            
        # If transform is not ready, log warning
        self.get_logger().warning(f"Transform not yet available between {self.reference_frame} and {self.imu_frame}")
    
    def _cache_common_transforms(self):
        """Cache commonly used transforms to avoid frequent lookups."""
        # Only try if we already verified the transform
        if not self.transform_verified:
            return
            
        try:
            # Look up transform from reference frame to IMU frame
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                self.imu_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            
            # Convert to matrix and cache
            ref_to_imu_matrix = Matrix4x4.from_tf_transform(transform)
            self.matrix_cache.set("ref_to_imu", ref_to_imu_matrix)
            
            # Look up inverse transform
            transform = self.tf_buffer.lookup_transform(
                self.imu_frame,
                self.reference_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            
            # Convert to matrix and cache
            imu_to_ref_matrix = Matrix4x4.from_tf_transform(transform)
            self.matrix_cache.set("imu_to_ref", imu_to_ref_matrix)
            
            # Mark as computed and cancel timer
            self.common_transforms_computed = True
            self.transform_check_timer_matrix.cancel()
            
            self.get_logger().info("Cached common transforms as matrices")
            
        except Exception as e:
            self.get_logger().warning(f"Failed to cache common transforms: {str(e)}")
    
    def get_transform_between_frames(self, source_frame, target_frame):
        """
        Get transform between two frames with caching.
        
        Args:
            source_frame: Source frame ID
            target_frame: Target frame ID
            
        Returns:
            TransformStamped or None if not available
        """
        # Create unique key for this transform
        frame_key = f"{target_frame}_{source_frame}"
        
        # Check standard transform cache first (this should return TransformStamped)
        if frame_key in self.transform_cache:
            return self.transform_cache.get(frame_key)
        
        # Look up new transform
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=self.transform_timeout)
            )
            
            # Cache the transform
            self.transform_cache.set(frame_key, transform)
            
            # Also cache as matrix for future use (but don't return it here)
            if self.use_matrix_transforms:
                matrix = Matrix4x4.from_tf_transform(transform)
                self.matrix_cache.set(frame_key, matrix)
            
            return transform
        except Exception as e:
            if hasattr(self, 'debug_level') and self.debug_level >= 2:
                self.get_logger().warning(f"Transform lookup error: {str(e)}")
            return None
    
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
        self.resource_timer = self.create_timer(1.0, self._monitor_resources)
        
        # Add cache cleanup timer
        self.cache_cleanup_timer = self.create_timer(60.0, self._cleanup_all_caches)
    
    def _cleanup_all_caches(self):
        """Perform cleanup of all caches to prevent memory growth."""
        # Force cleanup of all TTL dictionaries
        self.transform_cache.cleanup_all()
        self.matrix_cache.cleanup_all()
        
        # Log cache cleanup
        if self.debug_level >= 2:
            self.get_logger().info("Transform cache cleanup complete")
    
    def orientation_callback(self, msg):
        """Handle orientation updates from the IMU with improved transform handling."""
        # Extract yaw (z component) from the Vector3Stamped message
        raw_orientation = msg.vector.z
        
        # Store timestamp for freshness checking
        self.last_orientation_time = time.time()
        
        # If we need to transform the orientation to another frame
        if self.imu_frame != self.reference_frame:
            try:
                # First approach: direct transform using quaternion math
                transform = self.get_transform_between_frames(self.imu_frame, self.reference_frame)
                if transform:
                    # Extract quaternion components from transform
                    qx = transform.transform.rotation.x
                    qy = transform.transform.rotation.y
                    qz = transform.transform.rotation.z
                    qw = transform.transform.rotation.w
                    
                    # Create forward unit vector in IMU frame
                    forward_x = math.cos(raw_orientation)
                    forward_y = math.sin(raw_orientation)
                    forward_z = 0.0
                    
                    # Calculate rotation matrix elements from quaternion
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
                    self.robot_orientation = math.atan2(ty, tx)
                else:
                    # If transform not available, use raw orientation
                    self.robot_orientation = raw_orientation
                    
            except Exception as e:
                # In case of error, fall back to raw orientation
                self.get_logger().warning(f"Orientation transform error: {str(e)}")
                self.robot_orientation = raw_orientation
        else:
            # No transform needed
            self.robot_orientation = raw_orientation
        
        # Log orientation updates at high debug level
        if hasattr(self, 'debug_level') and self.debug_level >= 3:
            self.get_logger().debug(f"Orientation update: yaw={math.degrees(self.robot_orientation):.2f}°")
    
    def _is_orientation_fresh(self):
        """Check if orientation data is fresh enough to use."""
        if self.last_orientation_time is None:
            print ("orientation not received....")
            return False
            
        current_time = time.time()
        age = current_time - self.last_orientation_time
        
        # Consider orientation data older than 0.5 seconds as stale
        return age < 0.5
    
    def target_callback(self, msg):
        """Handle target position updates with enhanced filtering and strategic logging."""
        if self._shutting_down:
            return
        
        # Update timestamps
        self.last_target_time = time.time()
        self.current_target = msg
        
        # Extract key information from target
        target = msg.point
        
        # Calculate full 2D distance to target
        self.current_distance = math.sqrt(target.x**2 + target.y**2)
        
        # Store target frame for debugging
        frame_id = msg.header.frame_id if hasattr(msg.header, 'frame_id') else "unknown_frame"
        self.target_frame = frame_id

        # Calculate bearing/direction to ball based on frame
        if frame_id == "camera_frame" or frame_id == "camera_optical_frame":
            # Camera optical frame: Z forward, X right, Y down
            self.current_bearing = math.atan2(target.x, target.z)
            self.current_lateral = target.x
        else:
            # Standard robot frame: X forward, Y left
            self.current_bearing = math.atan2(target.y, target.x)
            self.current_lateral = target.y
        
        # Throttled logging for target updates
        if self.debug_level >= 2:
            self._log_throttled(
                self.get_logger().info,
                f"TARGET DATA: frame={frame_id}, pos=({target.x:.3f}, {target.y:.3f}, {target.z:.3f})",
                1.0,  # Throttle to once per second
                'last_target_log_time'
            )

        # Create original position tuple for filter update and logging
        original_position = (self.current_distance, self.current_lateral, self.current_bearing)
        
        # Apply target filtering
        # Update the filter with the new position
        try:
            filtered_position = self.target_filter.update(
                original_position,
                self.last_target_time
            )
        except Exception as e:
            # Log errors with filter updates
            log_structured('target_tracking', 'FILTER_ERROR', 
                        f"Error in target filter: {str(e)}", 
                        {'distance': self.current_distance}, 
                        level=logging.ERROR)
            
            # Fallback to using raw values
            filtered_position = original_position
        
        # Force target reacquisition if requested (e.g. after recovery)
        if self.force_target_reacquisition:
            # Log this significant event
            log_structured('target_tracking', 'REACQUISITION', 
                        "Forcing target filter reset for reacquisition", 
                        {'reason': getattr(self, 'reacquisition_reason', 'state_change')})
            
            self.target_filter.reset()
            filtered_position = self.target_filter.update(
                original_position,
                self.last_target_time
            )
            self.force_target_reacquisition = False
        
        # Use filtered/predicted position if available
        if filtered_position:
            # Check if we should use prediction or just filtering
            movement_info = self.target_filter.get_movement_info()
            
            if movement_info['is_moving'] and not movement_info['direction_change']:
                # For consistent movement, use prediction
                predicted_position = self.target_filter.get_predicted_position()
                
                if predicted_position:
                    # Use prediction for control
                    self.filtered_distance = predicted_position[0]
                    self.filtered_lateral = predicted_position[1]
                    self.filtered_bearing = predicted_position[2]
                    
                    # Only log when prediction is significantly different from filtered
                    prediction_diff = abs(predicted_position[0] - filtered_position[0]) + \
                                    abs(predicted_position[1] - filtered_position[1])
                    
                    if prediction_diff > 0.05 and self.debug_level >= 2:
                        # Log significant predictions
                        log_target_filter_update(
                            raw_position=original_position,
                            filtered_position=filtered_position,
                            predicted_position=predicted_position,
                            confidence=movement_info['confidence']
                        )
                else:
                    # Fall back to filtered values
                    self.filtered_distance = filtered_position[0]
                    self.filtered_lateral = filtered_position[1]
                    self.filtered_bearing = filtered_position[2]
            else:
                # Just use filtered values for inconsistent movement
                self.filtered_distance = filtered_position[0]
                self.filtered_lateral = filtered_position[1]
                self.filtered_bearing = filtered_position[2]
                
                # Log direction changes which are important events
                if movement_info['direction_change'] and self.debug_level >= 1:
                    log_structured('target_tracking', 'DIRECTION_CHANGE', 
                                "Target direction change detected", 
                                {'consistency': movement_info['consistency']})
        else:
            # Fall back to raw values if filtering not available
            self.filtered_distance = self.current_distance
            self.filtered_lateral = self.current_lateral
            self.filtered_bearing = self.current_bearing
    
    def state_callback(self, msg):
        """
        Handle robot state updates with improved transition logging and recovery behavior.
        
        Args:
            msg: State message containing the new state string
        """
        new_state = msg.data
        
        # If state changed, handle the transition
        if new_state != self.robot_state:
            # Calculate time in previous state
            time_in_state = 0.0
            if hasattr(self, 'state_change_time'):
                time_in_state = time.time() - self.state_change_time
            
            # Gather relevant parameters for state transition logging
            transition_params = {
                'previous_state': self.robot_state,
                'time_in_state': time_in_state,
                'distance_error': getattr(self, 'filtered_distance', 0.0) - self.desired_distance 
                                if hasattr(self, 'filtered_distance') else None,
                'lateral_error': getattr(self, 'filtered_lateral', 0.0) 
                            if hasattr(self, 'filtered_lateral') else None,
                'angular_error_deg': math.degrees(getattr(self, 'filtered_bearing', 0.0))
                                if hasattr(self, 'filtered_bearing') else None,
                'has_target': self.current_target is not None,
                'cycle_count': getattr(self, 'cycle_count', 0)
            }
            
            # Calculate reason for transition based on context
            if new_state == "tracking" and self.current_target is not None:
                reason = f"Target detected at {getattr(self, 'filtered_distance', 0.0):.2f}m"
            elif new_state == "recovery":
                reason = "Recovery triggered by state manager"
                # Add specific recovery trigger if known
                if hasattr(self, 'recovery_trigger'):
                    reason = f"Recovery triggered: {self.recovery_trigger}"
            elif new_state == "searching":
                # Add more context about search state entry
                if self.current_target is None:
                    reason = "Target lost - starting search"
                else:
                    time_since_target = time.time() - self.last_target_time if self.last_target_time else 0.0
                    reason = f"Target stale ({time_since_target:.1f}s) - starting search"
            elif new_state == "stopped":
                if hasattr(self, 'stop_reason'):
                    reason = f"Stop command received: {self.stop_reason}"
                else:
                    reason = "Stop command received"
            else:
                reason = "External state change via state topic"
            
            # Log state transition with detailed parameters
            log_state_transition(self, self.robot_state, new_state, reason, transition_params)
            
            # Store previous state and update current state
            self.previous_state = self.robot_state
            self.robot_state = new_state
            self.state_change_time = time.time()
            
            # Handle recovery state transitions
            if new_state == "recovery":
                self.in_recovery = True
                self.recovery_start_time = time.time()
                self.recovery_phase = "stop"
                
                # Log recovery initiation with more context
                recovery_params = {
                    'phase': self.recovery_phase,
                    'from_state': self.previous_state,
                    'error_distance': getattr(self, 'filtered_distance', 0.0) - self.desired_distance 
                                if hasattr(self, 'filtered_distance') else None,
                    'error_angle_deg': math.degrees(getattr(self, 'filtered_bearing', 0.0))
                                    if hasattr(self, 'filtered_bearing') else None
                }
                log_structured('recovery', 'RECOVERY_INIT', 
                            "Entering recovery mode - stopping robot", 
                            recovery_params)
                
                # Stop robot immediately when entering recovery
                self.stop_robot()
            elif self.previous_state == "recovery" and new_state != "recovery":
                # Log recovery completion
                recovery_duration = time.time() - getattr(self, 'recovery_start_time', time.time())
                log_structured('recovery', 'RECOVERY_COMPLETE', 
                            f"Exiting recovery mode after {recovery_duration:.1f}s", 
                            {'to_state': new_state,
                            'duration': recovery_duration,
                            'final_phase': getattr(self, 'recovery_phase', 'unknown')})
                self.in_recovery = False
            
            # Complete controller reset when transitioning between tracking and other states
            if new_state == "tracking" or self.previous_state == "tracking":
                # Reset controllers with reason logging
                controller_reset_reason = (f"State transition from {self.previous_state} to {new_state}" if 
                                        new_state != "tracking" else
                                        f"Entering tracking mode from {self.previous_state}")
                
                # Use parameter validation to check reset
                pre_reset_state = {
                    'linear_x_integral': self.pid_linear_x.integral if hasattr(self, 'pid_linear_x') else None,
                    'linear_y_integral': self.pid_linear_y.integral if hasattr(self, 'pid_linear_y') else None,
                    'angular_integral': self.pid_angular.integral if hasattr(self, 'pid_angular') else None
                }
                
                # Perform the reset
                self._complete_controller_reset()
                
                # Validate reset occurred
                post_reset_state = {
                    'linear_x_integral': self.pid_linear_x.integral if hasattr(self, 'pid_linear_x') else None,
                    'linear_y_integral': self.pid_linear_y.integral if hasattr(self, 'pid_linear_y') else None,
                    'angular_integral': self.pid_angular.integral if hasattr(self, 'pid_angular') else None
                }
                
                # Log controller reset with validation
                log_structured('pid_controller', 'CONTROLLER_RESET',
                            controller_reset_reason,
                            {'pre_reset': pre_reset_state, 'post_reset': post_reset_state})
                
                # Force target reacquisition when re-entering tracking mode
                if new_state == "tracking":
                    self.force_target_reacquisition = True
                    
                    # Log target reacquisition request
                    log_structured('target_filter', 'TARGET_REACQUISITION_REQUEST',
                                "Forcing target filter reset on tracking entry",
                                {'from_state': self.previous_state})
                    
            # If we're not in tracking mode, ensure the robot is stopped
            # (unless it's in searching or lost_ball mode, where state manager controls motion)
            if new_state != "tracking" and new_state != "searching" and new_state != "lost_ball":
                # Check if robot is already stopped to avoid redundant logs
                if hasattr(self, '_robot_stopped') and not self._robot_stopped:
                    stop_reason = f"State {new_state} requires stopping"
                    self.stop_reason = stop_reason  # Store for future reference
                    
                    # Log stop requirement
                    log_structured('motion_control', 'STOP_REQUIRED',
                                stop_reason,
                                {'new_state': new_state, 'from_state': self.previous_state})
                    
                    # Stop the robot
                    self.stop_robot()
        
        # Add periodic state validation for long-running states
        elif hasattr(self, 'cycle_count') and self.cycle_count % 100 == 0:
            # Check for stuck states
            if hasattr(self, 'state_change_time'):
                time_in_state = time.time() - self.state_change_time
                
                # Only log if in state for over 10 seconds
                if time_in_state > 10.0 and self.debug_level >= 1:
                    log_structured('state_monitor', 'EXTENDED_STATE',
                                f"In state '{self.robot_state}' for {time_in_state:.1f} seconds",
                                {'state': self.robot_state, 'time_in_state': time_in_state},
                                level=logging.INFO)
                    
                    # Validate appropriate behavior for long-running tracking state
                    if self.robot_state == "tracking" and time_in_state > 20.0:
                        is_moving = not getattr(self, '_robot_stopped', True)
                        
                        if not is_moving and hasattr(self, 'filtered_distance'):
                            # Log warning if robot is stationary in tracking mode for too long
                            distance_error = abs(self.filtered_distance - self.desired_distance)
                            
                            if distance_error > self.distance_threshold:
                                log_structured('state_monitor', 'TRACKING_ANOMALY',
                                            "Robot not moving despite being in tracking mode with significant error",
                                            {'distance_error': distance_error,
                                            'threshold': self.distance_threshold,
                                            'time_stationary': time_in_state},
                                            level=logging.WARNING)
    
    def _log_throttled(self, level_func, message, min_interval, last_time_attr):
        """Log messages with throttling to reduce log volume."""
        current_time = time.time()
        last_time = getattr(self, last_time_attr, 0)
        
        if current_time - last_time >= min_interval:
            level_func(message)
            setattr(self, last_time_attr, current_time)
            return True
            
        return False
    
    def _monitor_resources(self):
        """Monitor system resources and adjust control parameters with improved logging."""
        if not self.enable_resource_monitoring:
            return
            
        # Update resource metrics
        self.resource_monitor.update()
        current_cpu = self.resource_monitor.get_cpu_usage()
        current_memory = self.resource_monitor.get_memory_usage()
        
        # Store CPU history
        self.performance_stats['cpu'].append(current_cpu)
        self.current_cpu_usage = current_cpu
        
        # Adaptive control rate based on CPU usage
        if self.adaptive_control_rate:
            self._adjust_control_rate()
        
        # Log resources periodically (not every call)
        current_time = time.time()
        if current_time - self.last_resource_log_time >= 10.0:  # Every 10 seconds
            # Log using specialized function with cycle time info
            log_resource_usage(
                current_cpu, 
                current_memory, 
                self.cycle_duration_avg if hasattr(self, 'cycle_duration_avg') else None,
                getattr(self, 'last_rate_adjustment', None)
            )
            self.last_resource_log_time = current_time

    def _handle_resource_alert(self, resource_type, value):
        """Handle resource alerts from the resource monitor with improved logging."""
        current_time = time.time()
        
        # Add a grace period after startup before throttling
        if not hasattr(self, '_startup_time'):
            self._startup_time = current_time
            
        startup_elapsed = current_time - getattr(self, '_startup_time', 0)
        grace_period = startup_elapsed < 5.0  # 5 second grace period
        
        # Track the adjustment we're going to make
        rate_adjustment = None
        
        if resource_type == 'cpu':
            # Critical CPU threshold (90%+)
            if value > 90.0:
                if grace_period:
                    # During grace period, apply gentler throttling
                    log_structured('resource_monitor', 'CPU_ALERT_GRACE', 
                                f"CPU alert during startup grace period: {value:.1f}%", 
                                {'action': 'mild_throttling',
                                'startup_time': startup_elapsed})
                    
                    # Less aggressive rate adjustment during startup
                    if self.update_rate > self.min_update_rate:
                        new_rate = max(self.min_update_rate, self.update_rate * 0.9)  # 10% reduction
                        rate_adjustment = (self.update_rate, new_rate)
                        self._update_control_rate(new_rate)
                else:
                    # Normal throttling after grace period
                    log_structured('resource_monitor', 'CPU_ALERT_CRITICAL', 
                                f"Critical CPU usage: {value:.1f}%", 
                                {'action': 'aggressive_throttling'})
                    
                    self.skip_next_cycle = True
                    self.performance_stats["control_skips"] += 1
                    
                    if self.update_rate > self.min_update_rate:
                        new_rate = max(self.min_update_rate, self.update_rate * 0.7)  # 30% reduction
                        rate_adjustment = (self.update_rate, new_rate)
                        self._update_control_rate(new_rate)
            
            # High CPU threshold (80-90%)
            elif value > 80.0:
                # Only log if we haven't recently logged
                log_structured('resource_monitor', 'CPU_ALERT_HIGH', 
                            f"High CPU usage: {value:.1f}%", 
                            {'action': 'moderate_throttling'})
                
                if self.update_rate > self.min_update_rate:
                    new_rate = max(self.min_update_rate, self.update_rate * 0.8)  # 20% reduction
                    rate_adjustment = (self.update_rate, new_rate)
                    self._update_control_rate(new_rate)
            
            # Log CPU usage with any rate adjustments
            log_resource_usage(value, 
                            self.resource_monitor.get_memory_usage(), 
                            self.cycle_duration_avg if hasattr(self, 'cycle_duration_avg') else None,
                            rate_adjustment)
            
            # Store last adjustment for future logging
            if rate_adjustment:
                self.last_rate_adjustment = rate_adjustment[1] / rate_adjustment[0]  # Store as ratio
    
    def _adjust_control_rate(self):
        """Adjust control loop rate based on CPU usage with improved logging."""
        # Only adjust if we're using adaptive control
        if not self.adaptive_control_rate:
            return
            
        # Get average CPU usage
        avg_cpu = 0.0
        if self.performance_stats['cpu']:
            avg_cpu = sum(self.performance_stats['cpu']) / len(self.performance_stats['cpu'])
        
        # Track adjustments for logging
        rate_adjustment = None
        
        # Adjust based on average CPU usage
        if avg_cpu > self.cpu_high_threshold:
            # High CPU - reduce rate
            if self.update_rate > self.min_update_rate:
                current_rate = self.update_rate
                new_rate = max(self.min_update_rate, self.update_rate * 0.9)
                
                if abs(new_rate - current_rate) > 0.1:  # Only adjust if change is significant
                    rate_adjustment = (current_rate, new_rate)
                    self._update_control_rate(new_rate)
                    
                    # Log the adjustment
                    log_structured('resource_monitor', 'RATE_DECREASE', 
                                f"Decreasing control rate due to high CPU", 
                                {'cpu_avg': avg_cpu,
                                'old_rate': current_rate,
                                'new_rate': new_rate,
                                'threshold': self.cpu_high_threshold})
        
        elif avg_cpu < self.cpu_low_threshold and self.update_rate < self.base_update_rate:
            # Low CPU - increase rate, up to base rate
            current_rate = self.update_rate
            new_rate = min(self.base_update_rate, self.update_rate * 1.1)
            
            if abs(new_rate - current_rate) > 0.1:  # Only adjust if change is significant
                rate_adjustment = (current_rate, new_rate)
                self._update_control_rate(new_rate)
                
                # Log the adjustment
                log_structured('resource_monitor', 'RATE_INCREASE', 
                            f"Increasing control rate due to low CPU", 
                            {'cpu_avg': avg_cpu,
                            'old_rate': current_rate,
                            'new_rate': new_rate,
                            'threshold': self.cpu_low_threshold})
        
        # Log resource usage with rate adjustment info if an adjustment was made
        if rate_adjustment:
            log_resource_usage(
                avg_cpu, 
                self.resource_monitor.get_memory_usage(), 
                self.cycle_duration_avg if hasattr(self, 'cycle_duration_avg') else None,
                rate_adjustment[1] / rate_adjustment[0]  # Ratio of new/old rate
            )
            
            # Store adjustment for future reference
            self.last_rate_adjustment = rate_adjustment[1] / rate_adjustment[0]
    
    def _update_control_rate(self, new_rate):
        """Update the control loop rate with logging if it has changed significantly."""
        # Only update if change is significant
        if abs(new_rate - self.update_rate) < 0.1:
            return
        
        # Calculate percentage change for logging
        pct_change = (new_rate - self.update_rate) / self.update_rate * 100
        
        # Update rate
        self.update_rate = new_rate
        
        # Recreate timer with new rate
        self.timer.cancel()
        self.timer = self.create_timer(1.0 / self.update_rate, self.control_loop_callback)
        
        # Log the change
        log_structured('resource_monitor', 'RATE_UPDATED', 
                    f"Control rate updated: {pct_change:.1f}% change", 
                    {'new_rate': new_rate,
                    'period_ms': 1000.0 / new_rate})
    
    def _apply_simplified_control(self):
        """Apply a simplified control update when errors are small to save CPU."""
        # Reuse previous velocity commands with small damping
        damping = 0.95  # Slight reduction to avoid oscillation
        
        cmd_vel_msg = self._cmd_vel_msg
        cmd_vel_msg.linear.x = float(self.last_cmd_vel[0] * damping)
        cmd_vel_msg.linear.y = float(self.last_cmd_vel[1] * damping)
        cmd_vel_msg.angular.z = float(self.last_cmd_vel[2] * damping)
        
        # Publish command
        self.cmd_vel_pub.publish(cmd_vel_msg)
        
        # Update history but skip the expensive diagnostics and filtering
        self.velocity_history.add((cmd_vel_msg.linear.x, cmd_vel_msg.linear.y, cmd_vel_msg.angular.z))
        
        # Calculate cycle duration for performance monitoring
        cycle_duration = time.time() - self.cycle_start_time
        self.performance_stats['control_cycles'].append(cycle_duration)
        
        # Update running average
        if len(self.performance_stats['control_cycles']) > 0:
            self.cycle_duration_avg = sum(self.performance_stats['control_cycles']) / len(self.performance_stats['control_cycles'])

    def _can_use_simplified_control(self):
        """Check if simplified control can be used based on tracking error magnitudes."""
        # Only use simplified control when already in motion and errors are small
        if self._robot_stopped:
            return False
            
        # Check if errors are small enough to use simplified control
        small_errors = (
            abs(self.filtered_distance - self.desired_distance) < self.distance_threshold * 0.5 and
            abs(self.filtered_lateral) < self.lateral_threshold * 0.5 and
            abs(self.filtered_bearing) < math.radians(self.angular_threshold * 0.5)
        )
        
        # Don't use simplified control if we just started tracking
        if self.robot_state == "tracking" and time.time() - self.strategy_change_time < 1.0:
            return False
            
        return small_errors

    def _handle_non_tracking_state(self):
        """Handle robot behavior when not in tracking mode."""
        # When not tracking, ensure robot is stopped (unless controlled by another node)
        if self.robot_state not in ["searching", "lost_ball"]:
            self.stop_robot()
        return True  # Indicate that the method handled the situation

    def _update_performance_stats(self):
        """Update performance statistics for monitoring."""
        cycle_duration = time.time() - self.cycle_start_time
        self.performance_stats['control_cycles'].append(cycle_duration)
        
        # Update running average
        if len(self.performance_stats['control_cycles']) > 0:
            self.cycle_duration_avg = sum(self.performance_stats['control_cycles']) / len(self.performance_stats['control_cycles'])

    def _calculate_errors(self):
        """
        Calculate tracking errors using filtered values with improved logging.
        
        Returns:
            tuple: (distance, lateral, bearing, angular_degrees) - raw values and converted angular error
        """
        # Use filtered values for error calculation
        distance = self.filtered_distance
        lateral = self.filtered_lateral
        bearing = self.filtered_bearing
        
        # Calculate errors using pre-allocated array
        prev_distance_error = self._current_errors[0] if hasattr(self, '_current_errors') else 0.0
        prev_lateral_error = self._current_errors[1] if hasattr(self, '_current_errors') else 0.0
        prev_angular_error = self._current_errors[2] if hasattr(self, '_current_errors') else 0.0
        
        # Calculate current errors
        self._current_errors[0] = distance - self.desired_distance  # distance_error
        self._current_errors[1] = lateral - 0.0                    # lateral_error 
        self._current_errors[2] = bearing                          # angular_error
        
        # Convert angular error to degrees for logging
        angular_degrees = math.degrees(bearing)
        
        # Check for significant changes in error values
        distance_change = abs(self._current_errors[0] - prev_distance_error)
        lateral_change = abs(self._current_errors[1] - prev_lateral_error)
        angular_change = abs(math.degrees(self._current_errors[2] - prev_angular_error))
        
        # Log when errors change categories
        new_distance_category = self._categorize_error(
            self._current_errors[0], "distance", self.prev_distance_category)
        
        if new_distance_category != self.prev_distance_category:
            log_error_categorization(
                "distance",
                self._current_errors[0], 
                new_distance_category, 
                self.distance_threshold,
                prev_category=self.prev_distance_category
            )
            self.prev_distance_category = new_distance_category
        
        new_lateral_category = self._categorize_error(
            self._current_errors[1], "lateral", self.prev_lateral_category)
        
        if new_lateral_category != self.prev_lateral_category:
            log_error_categorization(
                "lateral", 
                self._current_errors[1], 
                new_lateral_category, 
                self.lateral_threshold,
                prev_category=self.prev_lateral_category
            )
            self.prev_lateral_category = new_lateral_category
        
        new_angular_category = self._categorize_error(
            angular_degrees, "angular", self.prev_angular_category)
        
        if new_angular_category != self.prev_angular_category:
            log_error_categorization(
                "angular", 
                angular_degrees, 
                new_angular_category, 
                self.angular_threshold,
                prev_category=self.prev_angular_category
            )
            self.prev_angular_category = new_angular_category
        
        # Log significant error changes regardless of category
        if distance_change > self.distance_threshold * 1.5 or \
        lateral_change > self.lateral_threshold * 1.5 or \
        angular_change > self.angular_threshold * 1.5:
            
            log_structured('error_tracking', 'ERROR_CHANGE', 
                        f"Significant change in tracking errors", 
                        {'distance_error': self._current_errors[0],
                        'distance_change': distance_change,
                        'lateral_error': self._current_errors[1],
                        'lateral_change': lateral_change,
                        'angular_error_deg': angular_degrees,
                        'angular_change_deg': angular_change})
        
        # When errors are all small, log this too (only when changing to "all small")
        all_errors_small = (abs(self._current_errors[0]) < self.distance_threshold and
                            abs(self._current_errors[1]) < self.lateral_threshold and
                            abs(angular_degrees) < self.angular_threshold)
        
        prev_all_small = getattr(self, '_prev_all_errors_small', False)
        
        if all_errors_small and not prev_all_small:
            # First time all errors are small - significant tracking milestone
            log_structured('error_tracking', 'TARGET_REACHED', 
                        f"All tracking errors within thresholds", 
                        {'distance_error': self._current_errors[0],
                        'distance_threshold': self.distance_threshold,
                        'lateral_error': self._current_errors[1],
                        'lateral_threshold': self.lateral_threshold,
                        'angular_error_deg': angular_degrees,
                        'angular_threshold': self.angular_threshold})
        
        # Store for next comparison
        self._prev_all_errors_small = all_errors_small
        
        # Log periodic error summaries (every 50 cycles) for monitoring
        if hasattr(self, 'cycle_count') and self.cycle_count % 100 == 0:
            log_structured('error_tracking', 'ERROR_SUMMARY', 
                        f"Tracking error summary", 
                        {'distance_error': self._current_errors[0],
                        'distance_category': new_distance_category,
                        'lateral_error': self._current_errors[1],
                        'lateral_category': new_lateral_category,
                        'angular_error_deg': angular_degrees,
                        'angular_category': new_angular_category},
                        level=logging.INFO)
        
        return distance, lateral, bearing, angular_degrees

    def _handle_stop_conditions(self, distance, lateral, angular_degrees, dt):
        """
        Check and handle stop conditions with improved state transition logging.
        
        Args:
            distance: Current distance to target
            lateral: Current lateral offset
            angular_degrees: Current angular error in degrees
            dt: Time since last control cycle
        
        Returns:
            bool: True if handled by stopping the robot, False to continue with normal control
        """
        # Check if we need to reset stopped state based on errors with enhanced hysteresis
        state_reset = self._reset_stopped_state_if_needed(
            self._current_errors[0], 
            self._current_errors[1], 
            angular_degrees
        )
        
        # If state was reset, skip the normal stop condition check this cycle
        if state_reset:
            # This was a transition from stopped to moving state
            # Already logged in _reset_stopped_state_if_needed
            return False  # Continue with normal control
        
        # Check stop conditions
        should_stop, stop_reason = self._evaluate_stop_conditions(
            distance, lateral, angular_degrees, self._robot_stopped
        )
        
        if should_stop:
            if not self._robot_stopped:
                # Transition from moving to stopped state
                self.stop_reason = stop_reason  # Store for use in stop_robot
                
                # Log state transition
                log_structured('motion_control', 'STOPPING_CONDITION_MET', 
                            stop_reason, 
                            {'distance_error': abs(distance - self.desired_distance),
                            'lateral_error': abs(lateral),
                            'angular_error': abs(angular_degrees),
                            'robot_state': self.robot_state})
                            
                self.get_logger().info(stop_reason)
                self.stop_robot()
            return True  # Handled by stopping the robot
        
        # Update error trackers
        self.distance_error_tracker.update(self._current_errors[0], dt)
        self.lateral_error_tracker.update(self._current_errors[1], dt)
        self.angular_error_tracker.update(self._current_errors[2], dt)
        
        return False  # Not handled, continue with normal control

    def _determine_and_apply_strategy(self, dt):
        """
        Determine movement strategy and apply it to calculate velocities with improved error handling.
        
        Args:
            dt: Time delta since last control cycle
                
        Returns:
            tuple: (linear_x_velocity, lateral_velocity, angular_velocity)
        """
        try:
            # Determine the optimal movement strategy
            strategy = self._select_strategy()
            
            # Compute velocities based on the strategy
            linear_x_velocity, lateral_velocity, angular_velocity = self._compute_velocities_from_strategy(strategy)
            
            # Apply velocity scaling and log results
            linear_x_velocity, lateral_velocity, angular_velocity = self._apply_velocity_scaling(
                linear_x_velocity, lateral_velocity, angular_velocity, strategy)
            
            return linear_x_velocity, lateral_velocity, angular_velocity
            
        except Exception as e:
            # Improved error handling - provide a safe fallback instead of stopping
            return self._handle_strategy_error(e)
            
    def _select_strategy(self):
        """
        Select the optimal movement strategy based on current errors and conditions.
        
        Returns:
            dict: The selected strategy
        """
        # Determine the optimal movement strategy with hysteresis
        prev_strategy_name = self.current_strategy
        
        # Log error values before strategy determination for context
        if self.debug_level >= 2:
            log_structured('strategy_selector', 'PRE_STRATEGY_ERRORS', 
                        f"Errors before strategy selection", 
                        {'distance_error': self._current_errors[0], 
                        'lateral_error': self._current_errors[1], 
                        'angular_error_deg': math.degrees(self._current_errors[2])})
        
        # Get error trends from trackers for dynamic adjustments
        distance_trend = self.distance_error_tracker.get_trend() if hasattr(self.distance_error_tracker, 'get_trend') else 0.0
        lateral_trend = self.lateral_error_tracker.get_trend() if hasattr(self.lateral_error_tracker, 'get_trend') else 0.0
        angular_trend = self.angular_error_tracker.get_trend() if hasattr(self.angular_error_tracker, 'get_trend') else 0.0
        
        # Current error values for convenient access
        distance_error = self._current_errors[0]
        lateral_error = self._current_errors[1]
        angular_error = self._current_errors[2]
        angular_error_degrees = math.degrees(angular_error)
        
        # Implement adaptive strategy hold time
        current_time = time.time()
        strategy_stability_needed = False

        # Check if we recently changed strategies and need stability
        if hasattr(self, 'strategy_change_time') and hasattr(self, 'current_strategy'):
            time_in_strategy = current_time - self.strategy_change_time
            # Base hold time on error magnitude - larger errors need longer hold times
            adaptive_hold_time = 0.5  # Base hold time
            if abs(distance_error) > 0.5 or abs(angular_error_degrees) > 10.0:
                adaptive_hold_time = 1.0  # Longer hold time for large errors
            
            # Only consider changing if we've held this strategy long enough
            if time_in_strategy < adaptive_hold_time:
                strategy_stability_needed = True
                
                # Log the strategy stability constraint
                if self.debug_level >= 2:
                    log_structured('strategy_selector', 'STRATEGY_STABILITY', 
                                f"Maintaining current strategy for stability", 
                                {'current_strategy': self.current_strategy,
                                'time_in_strategy': time_in_strategy,
                                'required_hold_time': adaptive_hold_time})
                
                # If we need strategy stability and have an active strategy, use it
                if hasattr(self, 'active_strategy'):
                    return self.active_strategy.as_dict()
        
        # Check for special case strategies
        strategy = self._check_special_case_strategies(distance_error, lateral_error, angular_error_degrees)
        
        # Apply strategy modifiers based on approach phase
        strategy = self._apply_strategy_modifiers(strategy, distance_error, distance_trend, lateral_trend, angular_trend)
        
        # Log strategy change if it occurred
        self._log_strategy_change(strategy, prev_strategy_name)
        
        return strategy

    def _check_special_case_strategies(self, distance_error, lateral_error, angular_error_degrees):
        """
        Check for special case strategies based on error conditions with enhanced
        angular prioritization.
        
        Args:
            distance_error: Current distance error
            lateral_error: Current lateral error
            angular_error_degrees: Current angular error in degrees
                
        Returns:
            dict: Selected strategy
        """
        # IMPROVED: Lowering angular threshold for special case handling from 10.0 to 7.0 degrees
        # This ensures earlier intervention for moderate angular errors
        if abs(angular_error_degrees) > 7.0:
            # Angular-first strategy for significant angular errors
            name = "ANGULAR_FIRST_APPROACH"
            use_forward = True
            use_lateral = True
            use_angular = True
            
            # IMPROVED: More aggressive angular scaling for faster correction
            # Original scales were forward=0.4, lateral=0.4, angular=1.0
            forward_scale = 0.3  # Reduced further to prioritize turning
            lateral_scale = 0.4  # Keep lateral scale the same 
            angular_scale = 0.9  # Increased from 1.0 to allow faster turning
            
            reason = f"Angular-first approach - correcting angular error: {angular_error_degrees:.1f}°"
            
            # Create and log special strategy
            special_strategy = {
                "strategy_name": name,
                "use_forward": use_forward,
                "use_lateral": use_lateral,
                "use_angular": use_angular,
                "forward_scale": forward_scale,
                "lateral_scale": lateral_scale,
                "angular_scale": angular_scale,
                "reason": reason
            }
            log_strategy_selection(
                "angular_override", 
                special_strategy,
                reason=f"Angular error ({angular_error_degrees:.1f}°) exceeds threshold"
            )
            
            # Store as active strategy
            self.active_strategy = MovementStrategy(
                name, use_forward, use_lateral, use_angular,
                forward_scale, lateral_scale, angular_scale, reason
            )
            
            return special_strategy
            
        # IMPROVED: Add additional special case for medium angular errors (3-7 degrees)
        # This creates a more gradual transition for angular corrections
        elif abs(angular_error_degrees) > 3.0:
            # Balanced approach with moderate angular priority
            name = "BALANCED_ANGULAR_CORRECTION"
            use_forward = True
            use_lateral = True
            use_angular = True
            
            # Scale factors that still allow forward movement but prioritize angular correction
            forward_scale = 0.6  # Moderate forward movement
            lateral_scale = 0.5  # Moderate lateral correction
            angular_scale = 0.7  # Strong angular priority (but not maximum)
            
            reason = f"Balanced approach with angular priority: {angular_error_degrees:.1f}°"
            
            # Create and log special strategy
            special_strategy = {
                "strategy_name": name,
                "use_forward": use_forward,
                "use_lateral": use_lateral,
                "use_angular": use_angular,
                "forward_scale": forward_scale,
                "lateral_scale": lateral_scale,
                "angular_scale": angular_scale,
                "reason": reason
            }
            log_strategy_selection(
                "medium_angular_override", 
                special_strategy,
                reason=f"Medium angular error ({angular_error_degrees:.1f}°) requires attention"
            )
            
            # Store as active strategy
            self.active_strategy = MovementStrategy(
                name, use_forward, use_lateral, use_angular,
                forward_scale, lateral_scale, angular_scale, reason
            )
            
            return special_strategy
            
        # Check for combined lateral and distance error - implement coordinated diagonal approach
        # This part remains similar to the original implementation
        elif abs(lateral_error) > 0.15 and abs(distance_error) > 0.3:
            # Calculate approach angle to determine optimal path
            approach_angle = math.degrees(math.atan2(lateral_error, distance_error))
            
            # For significant diagonal approaches, create a coordinated strategy
            if abs(approach_angle) > 10.0:
                name = "COORDINATED_DIAGONAL_APPROACH"
                use_forward = True
                use_lateral = True
                use_angular = True
                
                # Calculate coordinated scales based on approach angle
                # Higher lateral scale for larger angles
                lateral_factor = min(1.0, abs(approach_angle) / 45.0)
                forward_scale = 0.9
                lateral_scale = 0.8 * lateral_factor
                angular_scale = 0.5
                
                reason = f"Coordinated diagonal approach at {approach_angle:.1f}° angle"
                
                # Create coordinated strategy
                special_strategy = {
                    "strategy_name": name,
                    "use_forward": use_forward,
                    "use_lateral": use_lateral,
                    "use_angular": use_angular,
                    "forward_scale": forward_scale,
                    "lateral_scale": lateral_scale,
                    "angular_scale": angular_scale,
                    "reason": reason
                }
                
                log_strategy_selection(
                    "diagonal_override", 
                    special_strategy,
                    reason=f"Significant diagonal approach at {approach_angle:.1f}° angle"
                )
                
                # Store as active strategy
                self.active_strategy = MovementStrategy(
                    name, use_forward, use_lateral, use_angular,
                    forward_scale, lateral_scale, angular_scale, reason
                )
                
                return special_strategy
        
        # If no special case applies, determine strategy based on error categories
        strategy = self._determine_movement_strategy(
            distance_error, 
            lateral_error, 
            angular_error_degrees,
            self.prev_distance_category,
            self.prev_lateral_category,
            self.prev_angular_category
        )
        
        # Store as active strategy
        self.active_strategy = MovementStrategy(
            strategy["strategy_name"], 
            strategy["use_forward"], 
            strategy["use_lateral"], 
            strategy["use_angular"],
            strategy["forward_scale"], 
            strategy["lateral_scale"], 
            strategy["angular_scale"], 
            strategy["reason"]
        )
        
        return strategy

    def _apply_strategy_modifiers(self, strategy, distance_error, distance_trend, lateral_trend, angular_trend):
        """
        Apply modifiers to the strategy based on approach phase and error trends.
        
        Args:
            strategy: The base strategy to modify
            distance_error: Current distance error
            distance_trend, lateral_trend, angular_trend: Error trends
            
        Returns:
            dict: Modified strategy
        """
        # Apply final approach deceleration logic
        # Detect final approach phase - distance error between 0.1m and 0.5m
        if 0.1 < abs(distance_error) < 0.5:
            # Calculate progressive deceleration factor that scales with distance
            # Scales from 1.0 at 0.5m to 0.2 at 0.1m
            decel_factor = 0.2 + 0.8 * (abs(distance_error) - 0.1) / 0.4
            
            # Apply to forward movement
            original_scale = strategy['forward_scale']
            strategy['forward_scale'] = original_scale * decel_factor
            
            # Log the modification
            if self.debug_level >= 1:
                log_structured('motion_control', 'APPROACH_DECELERATION', 
                            f"Progressive deceleration applied", 
                            {'distance_error': distance_error,
                            'decel_factor': decel_factor,
                            'original_scale': original_scale,
                            'modified_scale': strategy['forward_scale']})
        
        # If errors are rapidly increasing, use more aggressive correction
        if distance_trend > 0.1 or lateral_trend > 0.1 or angular_trend > 0.1:
            # Errors are growing - use more aggressive strategy scales
            aggressive_factor = 1.2  # Increase scales by up to 20%
            
            # Apply to strategy scales
            if 'forward_scale' in strategy and distance_trend > 0:
                strategy['forward_scale'] = min(1.0, strategy['forward_scale'] * aggressive_factor)
            if 'lateral_scale' in strategy and lateral_trend > 0:
                strategy['lateral_scale'] = min(1.0, strategy['lateral_scale'] * aggressive_factor)
            if 'angular_scale' in strategy and angular_trend > 0:
                strategy['angular_scale'] = min(1.0, strategy['angular_scale'] * aggressive_factor)
                
            # Log the aggressive adjustment
            if self.debug_level >= 1:
                log_structured('strategy_selector', 'TREND_ADJUSTMENT', 
                            f"Strategy scales adjusted for increasing errors", 
                            {'distance_trend': distance_trend,
                            'lateral_trend': lateral_trend,
                            'angular_trend': angular_trend,
                            'aggressive_factor': aggressive_factor})
        
        return strategy

    def _log_strategy_change(self, strategy, prev_strategy_name):
        """
        Log strategy changes and update strategy tracking.
        
        Args:
            strategy: Current strategy
            prev_strategy_name: Previous strategy name
        """
        # Extract strategy name
        strategy_name = strategy["strategy_name"]
        
        # Track current strategy for next iteration
        self.current_strategy = strategy_name
        
        # Check if strategy changed
        if strategy_name != prev_strategy_name:
            # Strategy has changed - log the transition
            log_structured('strategy_selector', 'STRATEGY_CHANGED', 
                        f"Strategy changed: {prev_strategy_name} → {strategy_name}", 
                        {'reason': strategy["reason"],
                        'distance_cat': self.prev_distance_category,
                        'lateral_cat': self.prev_lateral_category, 
                        'angular_cat': self.prev_angular_category,
                        'forward_scale': strategy["forward_scale"],
                        'lateral_scale': strategy["lateral_scale"],
                        'angular_scale': strategy["angular_scale"]})
            
            # Update the strategy change time for oscillation detection
            self.strategy_change_time = time.time()
        elif self.debug_level >= 3:
            # Only log if at high debug level
            log_structured('strategy_selector', 'STRATEGY_MAINTAINED', 
                        f"Maintaining strategy: {strategy_name}", 
                        {'duration': time.time() - self.strategy_change_time})

    def _compute_velocities_from_strategy(self, strategy):
        """
        Compute velocities based on the selected strategy.
        
        Args:
            strategy: The movement strategy
            
        Returns:
            tuple: (linear_x_velocity, lateral_velocity, angular_velocity)
        """
        # Extract strategy parameters
        use_forward = strategy["use_forward"]
        use_lateral = strategy["use_lateral"]
        use_angular = strategy["use_angular"]
        
        # Current time for PID calculations
        current_time = time.time()
        
        # Track PID calls for debugging
        pid_results = {}
        
        # Compute velocities
        if self.coordinated_movement and use_lateral and use_angular:
            # Use coordinated controller for lateral and angular movements
            linear_x_velocity = self._compute_linear_x_velocity(use_forward, current_time, pid_results)
            
            # Compute coordinated lateral and angular velocities
            lateral_velocity, angular_velocity = self._compute_coordinated_velocities(
                use_lateral, use_angular, current_time, pid_results)
                
        else:
            # Traditional separate PID controllers
            linear_x_velocity = self._compute_linear_x_velocity(use_forward, current_time, pid_results)
            lateral_velocity = self._compute_lateral_velocity(use_lateral, current_time, pid_results)
            angular_velocity = self._compute_angular_velocity(use_angular, current_time, pid_results)
        
        # Log PID outputs periodically to reduce volume
        if hasattr(self, 'pid_log_counter'):
            self.pid_log_counter += 1
        else:
            self.pid_log_counter = 0
            
        # Log PID outputs every 50 cycles (increased from 20) or when error changes significantly
        significant_error_change = False
        if hasattr(self, 'last_logged_errors'):
            error_change = sum([abs(self._current_errors[i] - self.last_logged_errors[i]) for i in range(3)])
            significant_error_change = error_change > 0.2  # Increased threshold from 0.1 to 0.2

        if self.pid_log_counter % 50 == 0 or significant_error_change:
            # Log PID state
            for controller, data in pid_results.items():
                log_pid_state(
                    controller, 
                    data['error'], 
                    data['output'], 
                    data['p_term'], 
                    data['i_term'], 
                    data['d_term']
                )
            
            # Store current errors for next comparison
            self.last_logged_errors = self._current_errors.copy()
        
        return linear_x_velocity, lateral_velocity, angular_velocity

    def _compute_linear_x_velocity(self, use_forward, current_time, pid_results):
        """
        Compute linear X velocity using PID.
        
        Args:
            use_forward: Whether to use forward motion
            current_time: Current time for PID computation
            pid_results: Dictionary to store PID results for logging
            
        Returns:
            float: Computed linear X velocity
        """
        linear_x_velocity = self.pid_linear_x.compute(
            self._current_errors[0], 
            current_time, 
            not use_forward,
            self.distance_error_tracker.get_trend()
        )
        
        # Store PID results for logging
        pid_results['linear_x'] = {
            'error': self._current_errors[0],
            'output': linear_x_velocity,
            'p_term': self.pid_linear_x.last_p_term,
            'i_term': self.pid_linear_x.last_i_term,
            'd_term': self.pid_linear_x.last_d_term
        }
        
        return linear_x_velocity

    def _compute_lateral_velocity(self, use_lateral, current_time, pid_results):
        """
        Compute lateral velocity using PID.
        
        Args:
            use_lateral: Whether to use lateral motion
            current_time: Current time for PID computation
            pid_results: Dictionary to store PID results for logging
            
        Returns:
            float: Computed lateral velocity
        """
        lateral_velocity = self.pid_linear_y.compute(
            self._current_errors[1], 
            current_time, 
            not use_lateral,
            self.lateral_error_tracker.get_trend()
        )
        
        # Store PID results for logging
        pid_results['lateral'] = {
            'error': self._current_errors[1],
            'output': lateral_velocity,
            'p_term': self.pid_linear_y.last_p_term,
            'i_term': self.pid_linear_y.last_i_term,
            'd_term': self.pid_linear_y.last_d_term
        }
        
        return lateral_velocity

    def _detect_angular_oscillation(self, angular_error, angular_velocity):
        """
        Detect and prevent oscillations in angular control.
        
        Args:
            angular_error: Current angular error
            angular_velocity: Current angular velocity
            
        Returns:
            bool: True if oscillation is detected
        """
        # Check if error and velocity have opposite signs (indicating overshooting)
        opposite_signs = angular_error * angular_velocity < 0
        
        # Check if angular error is small but velocity is significant
        small_error_high_velocity = (abs(angular_error) < math.radians(3.0) and 
                                    abs(angular_velocity) > 0.3)
        
        # Check oscillation count in error tracker
        high_oscillation = (hasattr(self.angular_error_tracker, 'sign_changes') and 
                        self.angular_error_tracker.sign_changes > 2)
        
        # Detect oscillation when multiple conditions are met
        if (opposite_signs and small_error_high_velocity) or high_oscillation:
            log_structured('motion_control', 'ANGULAR_OSCILLATION', 
                        "Angular oscillation detected", 
                        {'error_rad': angular_error,
                        'error_deg': math.degrees(angular_error),
                        'velocity': angular_velocity,
                        'sign_changes': getattr(self.angular_error_tracker, 'sign_changes', 0)})
            return True
        
        return False

    def _compute_angular_velocity(self, use_angular, current_time, pid_results):
        # Compute base angular velocity using PID
        angular_velocity = self.pid_angular.compute(
            self._current_errors[2], 
            current_time, 
            not use_angular,
            self.angular_error_tracker.get_trend()
        )
        
        # Check for oscillation and reduce velocity if detected
        if self._detect_angular_oscillation(self._current_errors[2], angular_velocity):
            # Dampen velocity to break oscillation pattern
            angular_velocity *= 0.5  # Reduce velocity by 50% to dampen oscillation
            
            # Apply more aggressive damping for very small errors
            if abs(self._current_errors[2]) < math.radians(1.0):
                angular_velocity *= 0.5  # Further reduction for very small errors
        
        # Store PID results for logging
        pid_results['angular'] = {
            'error': math.degrees(self._current_errors[2]),
            'output': angular_velocity,
            'p_term': self.pid_angular.last_p_term,
            'i_term': self.pid_angular.last_i_term,
            'd_term': self.pid_angular.last_d_term
        }
        
        return angular_velocity

    def _compute_coordinated_velocities(self, use_lateral, use_angular, current_time, pid_results):
        """
        Compute coordinated lateral and angular velocities.
        
        Args:
            use_lateral: Whether to use lateral motion
            use_angular: Whether to use angular motion
            current_time: Current time for computation
            pid_results: Dictionary to store results for logging
            
        Returns:
            tuple: (lateral_velocity, angular_velocity)
        """
        start_coord_time = time.time()
        try:
            lateral_velocity, angular_velocity = self.coordinated_controller.compute(
                self._current_errors[1],   # lateral error
                self._current_errors[2],   # angular error
                current_time,              # current time
                self.robot_orientation     # current orientation from IMU
            )
            coord_time = time.time() - start_coord_time
            
            # Log coordinated control output on significant changes
            if hasattr(self, 'last_lateral_vel') and hasattr(self, 'last_angular_vel'):
                lat_change = abs(lateral_velocity - self.last_lateral_vel)
                ang_change = abs(angular_velocity - self.last_angular_vel)
                
                # Log on significant changes (>10%)
                if (lat_change > 0.10 or ang_change > 0.1) and self.debug_level >= 1:
                    log_structured('coordinated_control', 'COORDINATION_OUTPUT', 
                                f"Coordinated control output changed", 
                                {'lateral_error': self._current_errors[1],
                                'angular_error_deg': math.degrees(self._current_errors[2]),
                                'lateral_velocity': lateral_velocity,
                                'angular_velocity': angular_velocity,
                                'lateral_change': lat_change,
                                'angular_change': ang_change,
                                'compute_time_ms': coord_time * 1000})
            
            # Store for next comparison
            self.last_lateral_vel = lateral_velocity
            self.last_angular_vel = angular_velocity
        except Exception as e:
            # Handle coordinated controller error with fallback values
            self.get_logger().warning(f"Coordinated control failed: {str(e)}, using fallback values")
            
            # Compute lateral separately as fallback
            lateral_velocity = self.pid_linear_y.compute(
                self._current_errors[1], 
                current_time, 
                not use_lateral,
                self.lateral_error_tracker.get_trend()
            )
            
            # Compute angular separately as fallback
            angular_velocity = self.pid_angular.compute(
                self._current_errors[2], 
                current_time, 
                not use_angular,
                self.angular_error_tracker.get_trend()
            )
            
            # Log the fallback action
            log_structured('coordinated_control', 'COORDINATION_FALLBACK', 
                        f"Using fallback for coordinated control due to error", 
                        {'lateral_error': self._current_errors[1],
                        'angular_error_deg': math.degrees(self._current_errors[2]),
                        'error': str(e)})
        
        # Disable individual components if strategy requires
        if not use_lateral:
            lateral_velocity = 0.0
        if not use_angular:
            angular_velocity = 0.0
        
        return lateral_velocity, angular_velocity

    def _apply_velocity_scaling(self, linear_x_velocity, lateral_velocity, angular_velocity, strategy):
        """
        Apply velocity scaling based on the strategy and current errors with enhanced
        angular response control to prevent overshooting.
        
        Args:
            linear_x_velocity, lateral_velocity, angular_velocity: Unscaled velocities
            strategy: Current movement strategy
                
        Returns:
            tuple: (scaled_linear_x, scaled_lateral, scaled_angular)
        """
        # Extract strategy parameters
        forward_scale = strategy["forward_scale"]
        lateral_scale = strategy["lateral_scale"]
        angular_scale = strategy["angular_scale"]
        use_forward = strategy["use_forward"]
        use_angular = strategy["use_angular"]
        strategy_name = strategy["strategy_name"]
        
        # Store velocities before scaling
        unscaled_velocities = [linear_x_velocity, lateral_velocity, angular_velocity]
        
        # Apply enhanced angular velocity scaling with lower threshold (3.0 instead of 5.0 degrees)
        if use_forward and use_angular and abs(self._current_errors[2]) > math.radians(3.0):
            # Calculate angular error in degrees for readability
            angular_error_deg = math.degrees(abs(self._current_errors[2]))
            
            # More moderate reduction curve for forward velocity
            angular_scaling = max(0.3, 1.0 - pow(angular_error_deg / 15.0, 1.0))
            
            # Apply scaling to forward velocity
            original_linear_x = linear_x_velocity
            linear_x_velocity *= angular_scaling
            
            # Apply more conservative angular velocity boost with improved dampening curve
            # MODIFIED: Reduced from 1.05 to 1.03 maximum boost
            angular_boost = min(1.03, 1.0 + pow(angular_error_deg / 20.0, 0.5))
            angular_velocity *= angular_boost
            
            # Log the angular-based velocity scaling
            if self.debug_level >= 1:
                log_structured('motion_control', 'ANGULAR_VELOCITY_SCALING', 
                            f"Forward velocity scaled due to angular error", 
                            {'angular_error_deg': angular_error_deg,
                            'scaling_factor': angular_scaling,
                            'angular_boost': angular_boost,
                            'original_velocity': original_linear_x,
                            'scaled_velocity': linear_x_velocity})
        
        # Check if we're approaching target angle (getting close to zero error)
        # MODIFIED: Enhanced deceleration logic to prevent overshooting
        if abs(self._current_errors[2]) < math.radians(7.0) and abs(angular_velocity) > 0.2:
            # Apply stronger deceleration when approaching zero angular error
            # (scales down velocity more as error approaches zero)
            deceleration_factor = max(0.2, abs(self._current_errors[2]) / math.radians(7.0))
            original_angular = angular_velocity
            angular_velocity *= deceleration_factor
            
            # Log this important event
            if self.debug_level >= 1:
                log_structured('motion_control', 'ANGULAR_DECELERATION', 
                            f"Enhanced angular deceleration applied when approaching target", 
                            {'angular_error_deg': math.degrees(self._current_errors[2]),
                            'deceleration_factor': deceleration_factor,
                            'original_velocity': original_angular,
                            'scaled_velocity': angular_velocity})
        
        # Strategy-specific angular scaling
        # MODIFIED: Reduced all strategy-specific boost factors
        if "ANGULAR" in strategy_name:
            # Boost angular velocity for angular-focused strategies
            # MODIFIED: Reduced from 1.8 to 1.3
            angular_boost_factor = 1.3
            angular_velocity *= angular_boost_factor
        elif "APPROACH" in strategy_name or "DIAGONAL" in strategy_name:
            # Moderate boost for approach strategies
            # MODIFIED: Reduced from 1.5 to 1.2
            angular_boost_factor = 1.2
            angular_velocity *= angular_boost_factor
        
        # Check if angular and forward motions are working together or against each other
        if (linear_x_velocity > 0.05 and angular_velocity * self._current_errors[2] < 0):
            # Angular velocity is helping alignment during approach - boost it
            # MODIFIED: Reduced from 1.3 to 1.1
            corrective_boost = 1.1
            angular_velocity *= corrective_boost
        
        # When very close to target, prioritize angular alignment
        if abs(self._current_errors[0]) < 0.2:  # Close to target distance
            # Higher angular priority when close to target
            # MODIFIED: Limited maximum boost from 2.0 to 1.5
            close_target_boost = min(1.5, 1.0 + abs(self._current_errors[2]) / 0.12)
            angular_velocity *= close_target_boost
            
            # If angular error is significant when close to target, reduce forward speed further
            if abs(self._current_errors[2]) > math.radians(5.0):
                linear_x_velocity *= 0.5
        
        # Apply standard strategy scaling factors
        linear_x_velocity *= forward_scale
        lateral_velocity *= lateral_scale
        
        # MODIFIED: Apply a more conservative angular scaling
        # For strategies with high angular scale, apply a dampening factor
        if angular_scale > 0.9:
            # Scale down the scaling factor itself to prevent excessive angular velocity
            dampened_angular_scale = 0.9 + (angular_scale - 0.9) * 0.5  # Dampen values above 0.9
            angular_velocity *= dampened_angular_scale
            
            # Log when dampening is applied
            if self.debug_level >= 1 and abs(dampened_angular_scale - angular_scale) > 0.05:
                log_structured('motion_control', 'ANGULAR_SCALE_DAMPENING', 
                            f"Angular scale dampened to prevent overshooting", 
                            {'original_scale': angular_scale,
                            'dampened_scale': dampened_angular_scale})
        else:
            # Apply normal scaling for smaller angular scales
            angular_velocity *= angular_scale
        
        # Protect against excessive angular velocity after all boosts
        # MODIFIED: Reduced from 1.2 to 1.1 times max
        max_safe_angular = self.angular_max * 1.3  # Allow slightly higher than standard max
        if abs(angular_velocity) > max_safe_angular:
            original_angular = angular_velocity
            angular_velocity = math.copysign(max_safe_angular, angular_velocity)
            
            # Log velocity capping
            if self.debug_level >= 1:
                log_structured('motion_control', 'ANGULAR_VELOCITY_CAP',
                            f"Angular velocity capped to prevent overshooting",
                            {'original': original_angular,
                            'capped': angular_velocity,
                            'max_allowed': max_safe_angular})
        
        # Log velocity scaling results if significant
        if max(abs(linear_x_velocity - unscaled_velocities[0]),
            abs(lateral_velocity - unscaled_velocities[1]),
            abs(angular_velocity - unscaled_velocities[2])) > 0.05:
            
            log_structured('strategy_selector', 'VELOCITY_SCALING', 
                        f"Strategy scaling applied to velocities", 
                        {'strategy': strategy_name,
                        'unscaled_forward': unscaled_velocities[0],
                        'unscaled_lateral': unscaled_velocities[1],
                        'unscaled_angular': unscaled_velocities[2],
                        'scaled_forward': linear_x_velocity,
                        'scaled_lateral': lateral_velocity,
                        'scaled_angular': angular_velocity})
        
        return linear_x_velocity, lateral_velocity, angular_velocity

    def _handle_strategy_error(self, error):
        """
        Handle errors in strategy determination with safe fallback.
        
        Args:
            error: The exception that occurred
            
        Returns:
            tuple: Safe fallback velocities (linear_x, lateral, angular)
        """
        # Log the error
        self.get_logger().error(f"Strategy determination error: {str(error)}, using safe fallback strategy")
        
        # Get current error values for fallback strategy
        distance_error = self._current_errors[0]
        lateral_error = self._current_errors[1]
        angular_error = self._current_errors[2]
        angular_error_degrees = math.degrees(angular_error)
        
        # Create safe fallback velocities based on current errors
        # Don't move forward - prioritize lateral and angular corrections
        safe_linear_x = 0.0 
        
        # Small lateral correction if needed
        safe_lateral = 0.0
        if abs(lateral_error) > self.lateral_threshold:
            # Apply small lateral correction in the right direction
            safe_lateral = 0.05 * (1.0 if lateral_error > 0 else -1.0)
            
        # Small angular correction if needed
        safe_angular = 0.0
        if abs(angular_error_degrees) > self.angular_threshold:
            # Apply small angular correction in the right direction
            safe_angular = 0.05 * (1.0 if angular_error > 0 else -1.0)
        
        # Log the fallback strategy
        log_structured('strategy_selector', 'FALLBACK_STRATEGY', 
                    f"Using safe fallback strategy due to error",
                    {'original_error': str(error),
                    'distance_error': distance_error, 
                    'lateral_error': lateral_error,
                    'angular_error_deg': angular_error_degrees,
                    'vel_x': safe_linear_x,
                    'vel_y': safe_lateral,
                    'vel_angular': safe_angular})
        
        # Update current strategy for tracking
        self.current_strategy = "ERROR_FALLBACK"
        
        return safe_linear_x, safe_lateral, safe_angular

    def _apply_and_publish_velocities(self, linear_x_velocity, lateral_velocity, angular_velocity):
        """
        Apply velocity limits and publish command velocities with enhanced logging.
        
        Args:
            linear_x_velocity: Calculated forward velocity
            lateral_velocity: Calculated lateral velocity
            angular_velocity: Calculated angular velocity
        """
        # Store original velocities for logging
        original_velocities = (linear_x_velocity, lateral_velocity, angular_velocity)
        
        # Track command publication time
        start_time = time.time()
        
        # Apply velocity and acceleration limits
        current_time = time.time()
        limited_velocities = self._apply_velocity_limits(
            linear_x_velocity, lateral_velocity, angular_velocity, current_time
        )
        
        # Calculate time spent in velocity limiting
        velocity_limit_time = time.time() - start_time
        
        # Unpack limited velocities
        linear_x_velocity, lateral_velocity, angular_velocity = limited_velocities
        
        # Store new velocities in pre-allocated arrays for efficiency
        self._velocity_tuple[0] = linear_x_velocity
        self._velocity_tuple[1] = lateral_velocity
        self._velocity_tuple[2] = angular_velocity
        
        # Calculate if velocity changed significantly from what was previously commanded
        prev_cmd = self.last_cmd_vel
        significant_change = False
        change_components = []
        
        for i, component in enumerate(['forward', 'lateral', 'angular']):
            if abs(self._velocity_tuple[i] - prev_cmd[i]) > 0.1:  
                significant_change = True
                change_components.append(component)
        
        # Also check if any component changed sign (direction reversal)
        direction_change = False
        reversing_components = []
        
        for i, component in enumerate(['forward', 'lateral', 'angular']):
            if (prev_cmd[i] * self._velocity_tuple[i] < 0) and abs(self._velocity_tuple[i]) > 0.10:
                direction_change = True
                reversing_components.append(component)
        
        # Log significant velocity changes
        if significant_change:
            # Only log for significant changes to reduce log volume
            changed_dirs = ", ".join(change_components)
            
            # Include additional context for direction reversals
            if direction_change:
                log_structured('motion_commander', 'VELOCITY_CHANGE', 
                            f"Significant velocity change: {changed_dirs} (direction reversal in {', '.join(reversing_components)})", 
                            {'previous': prev_cmd,
                            'current': tuple(self._velocity_tuple),
                            'original': original_velocities,
                            'robot_state': self.robot_state})
            else:
                # Standard velocity change
                log_structured('motion_commander', 'VELOCITY_CHANGE', 
                            f"Significant velocity change: {changed_dirs}", 
                            {'previous': prev_cmd,
                            'current': tuple(self._velocity_tuple),
                            'original': original_velocities,
                            'robot_state': self.robot_state})
        
        # Set up for standard logging
        should_log_velocity = False
        
        # Log for cycle-based or significant events to avoid log spam
        if hasattr(self, '_vel_log_count'):
            self._vel_log_count += 1
            
            # Log periodically (every 15 cycles) or on significant changes
            should_log_velocity = (self._vel_log_count % 30 == 0) or significant_change
        else:
            self._vel_log_count = 0
            should_log_velocity = True  # Always log the first velocity
        
        # Log velocities (throttled)
        if should_log_velocity:
            # Compare original vs limited velocities to see limiting impact
            limiting_occurred = any(abs(o - l) > 0.01 for o, l in zip(original_velocities, limited_velocities))
            limiting_info = ""
            
            if limiting_occurred:
                # Calculate limitation percentages
                limit_percentages = []
                for i in range(3):
                    if abs(original_velocities[i]) > 0.01:
                        pct = abs(limited_velocities[i] - original_velocities[i]) / abs(original_velocities[i]) * 100
                        if pct > 5:  # Only note significant limiting
                            component = ['x', 'y', 'θ'][i]
                            limit_percentages.append(f"{component}:{pct:.0f}%")
                
                if limit_percentages:
                    limiting_info = f" (limited: {', '.join(limit_percentages)})"
            
            # Standard velocity logging with limiting info
            self.get_logger().info(
                f"MOTION: x={linear_x_velocity:.2f} y={lateral_velocity:.2f} θ={angular_velocity:.2f}{limiting_info}"
            )
            
            # Update last logged command
            self.last_logged_cmd = tuple(self._velocity_tuple)
        
        # Store for next cycle
        self.last_cmd_vel = tuple(self._velocity_tuple)
        
        # Get a Twist message from the pool (or use pre-allocated message)
        cmd_vel_msg = self._cmd_vel_msg  # Use pre-allocated message
        
        # Set velocity values
        cmd_vel_msg.linear.x = float(linear_x_velocity)
        cmd_vel_msg.linear.y = float(lateral_velocity)
        cmd_vel_msg.angular.z = float(angular_velocity)
        
        # Save velocity for history
        velocity_tuple = (float(linear_x_velocity), float(lateral_velocity), float(angular_velocity))
        self.velocity_history.add(velocity_tuple)
        
        # Track robot stopped status for other components
        was_stopped = self._robot_stopped
        self._robot_stopped = abs(linear_x_velocity) < 0.01 and abs(lateral_velocity) < 0.01 and abs(angular_velocity) < 0.01
        
        # Log stop/start transitions
        if was_stopped and not self._robot_stopped:
            # Robot starting to move
            log_structured('motion_commander', 'ROBOT_MOTION_START', 
                        f"Robot starting movement", 
                        {'velocity': velocity_tuple,
                        'robot_state': self.robot_state})
        elif not was_stopped and self._robot_stopped:
            # Robot stopping
            log_structured('motion_commander', 'ROBOT_MOTION_STOP', 
                        f"Robot stopping", 
                        {'prev_velocity': prev_cmd,
                        'robot_state': self.robot_state})
        
        # Track publication timing
        publication_start = time.time()
        
        # Publish command
        self.cmd_vel_pub.publish(cmd_vel_msg)
        
        # Track publication time for performance monitoring
        publication_time = time.time() - publication_start
        
        # Log performance metrics periodically
        if hasattr(self, '_perf_log_count'):
            self._perf_log_count += 1
            
            # Log every 50 cycles
            if self._perf_log_count % 100 == 0:
                log_structured('motion_commander', 'PERFORMANCE_METRICS', 
                            f"Motion command performance metrics", 
                            {'velocity_limit_time_ms': velocity_limit_time * 1000,
                            'publication_time_ms': publication_time * 1000,
                            'cycle_time_ms': self.cycle_duration_avg * 1000 if hasattr(self, 'cycle_duration_avg') else 0,
                            'cpu_usage': self.current_cpu_usage if hasattr(self, 'current_cpu_usage') else 0})
        else:
            self._perf_log_count = 0
        
        # Update error trackers if significant movement is occurring
        if abs(linear_x_velocity) > 0.05:
            self.distance_error_tracker.record_correction()
        if abs(lateral_velocity) > 0.05:
            self.lateral_error_tracker.record_correction()
        if abs(angular_velocity) > 0.1:
            self.angular_error_tracker.record_correction()

    def _optimize_transforms_and_filtering(self):
        """Execute expensive transform and filtering operations at reduced frequency."""
        # Only perform expensive operations periodically to save CPU
        if self.cycle_count % 3 == 0 or self.force_target_reacquisition:
            # Verify critical transforms
            if not self.transform_verified and not self.tf_buffer.can_transform(
                self.reference_frame,
                self.imu_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            ):
                self.get_logger().warning(f"Transform not yet available between {self.reference_frame} and {self.imu_frame}")
                
            # Cache common transforms if needed
            if self.enable_transform_caching and not self.common_transforms_computed:
                self._cache_common_transforms()
                
            return True
        
        return False  # No expensive operations performed

    def control_loop_callback(self):
        """Regular control loop to calculate and publish velocity commands with CPU optimization."""
        try:
            # Skip this cycle if requested for CPU relief
            if self.skip_next_cycle:
                self.skip_next_cycle = False
                self.performance_stats["control_skips"] += 1
                return
                    
            # Track performance
            self.cycle_start_time = time.time()
            
            if self._shutting_down:
                return
                        
            current_time = time.time()

            if self._robot_stopped and current_time - self._stop_time > 1.0:
                # If stopped for more than 1 second, decay integral term
                self.pid_linear_x.integral *= 0.5  # Reduce by 50% each second
                self.pid_linear_y.integral *= 0.5
                self.pid_angular.integral *= 0.5

            dt = current_time - self.last_control_time
            self.last_control_time = current_time
            self.cycle_count += 1
            
            # Log periodic status updates (once every 50 cycles)
            if self.cycle_count % 50 == 0:
                self._log_periodic_status()
            
            # Special handling for recovery mode
            if self.in_recovery:
                self._handle_recovery_mode(current_time)
                return
            
            # Only generate commands in tracking mode with a recent target
            if self.robot_state != "tracking" or self.current_target is None:
                if self._handle_non_tracking_state():
                    return
            
            # Check if orientation data is fresh (prevents race conditions)
            if not self._is_orientation_fresh():
                self.get_logger().warning("Skipping control cycle - orientation data is stale")
                return
            
            # Check if we can use simplified control to save CPU
            # This is the key CPU optimization - avoid expensive calculations when appropriate
            if self._can_use_simplified_control() or self.current_cpu_usage > 80.0:
                # Use simplified control approach when errors are small
                if self.debug_level >= 2:
                    self.get_logger().info("Using simplified control to save CPU - errors are small")
                self._apply_simplified_control()
                return
                
            # Perform expensive transform operations at reduced frequency
            self._optimize_transforms_and_filtering()
            
            # Calculate current errors
            distance, lateral, bearing, angular_degrees = self._calculate_errors()
            
            if self.debug_level >= 2:
                self.get_logger().info(
                    f"PRE-STOP CHECK: distance={distance:.3f}m (target={self.desired_distance:.3f}m), "
                    f"lateral={lateral:.3f}m, angular={angular_degrees:.2f}°, "
                    f"is_stopped={self._robot_stopped}"
                )
            
            # Check stop conditions and handle if needed
            if self._handle_stop_conditions(distance, lateral, angular_degrees, dt):
                return
            
            # Determine strategy and calculate velocities
            linear_x_velocity, lateral_velocity, angular_velocity = self._determine_and_apply_strategy(dt)
            
            # Apply velocity limits and publish commands
            self._apply_and_publish_velocities(linear_x_velocity, lateral_velocity, angular_velocity)
            
            # Update performance stats
            self._update_performance_stats()
                
        except Exception as e:
            self.get_logger().error(f"Unexpected error in control_loop_callback: {str(e)}")
            # Try to safely stop the robot
            try:
                self.stop_robot()
            except Exception as stop_error:
                self.get_logger().error(f"Failed to stop robot after error: {str(stop_error)}")
    
    def _handle_recovery_mode(self, current_time):
        """
        Handle recovery mode with a three-phase approach and improved transition logging:
        1. Stop - halt all movement
        2. Orient - align with target
        3. Approach - move to target
        """
        # Check if we should exit recovery mode
        if self.robot_state != "recovery":
            if self.in_recovery:
                # Log recovery exit
                log_structured('recovery', 'RECOVERY_FORCED_EXIT', 
                            "Forced exit from recovery due to state change", 
                            {'new_state': self.robot_state,
                            'was_in_phase': self.recovery_phase})
            self.in_recovery = False
            self.recovery_phase = "none"
            return
        
        recovery_duration = current_time - self.recovery_start_time
        prev_phase = self.recovery_phase
        
        # Phase 1: Stop (0-1 seconds)
        if self.recovery_phase == "stop":
            # Ensure robot is stopped
            self.stop_robot()
            
            # After 1 second, transition to orient phase
            if recovery_duration > 1.0:
                self.recovery_phase = "orient"
                
                # Log phase transition with context
                log_structured('recovery', 'PHASE_TRANSITION', 
                            "Recovery: Moving to orientation phase", 
                            {'from_phase': prev_phase,
                            'to_phase': self.recovery_phase,
                            'duration_in_prev': recovery_duration,
                            'robot_stopped': self._robot_stopped})
                
                # Reset angular controller
                self.pid_angular.reset()
                self.angular_error_tracker.reset()
                
        # Phase 2: Orient (1-3 seconds)
        elif self.recovery_phase == "orient":
            # Only orient if we have a target
            if self.current_target is not None and self._is_orientation_fresh():
                # Use filtered bearing
                bearing = self.filtered_bearing
                angular_degrees = math.degrees(bearing)
                
                # Only orient if angular error is significant
                if abs(angular_degrees) > 2.0:
                    # Compute angular velocity with PID
                    angular_velocity = self.pid_angular.compute(
                        bearing,
                        current_time,
                        force_zero=False
                    )
                    
                    # Apply conservative scaling
                    angular_velocity *= 0.8
                    
                    # Create and publish Twist message
                    cmd_vel_msg = self._cmd_vel_msg
                    cmd_vel_msg.linear.x = 0.0
                    cmd_vel_msg.linear.y = 0.0
                    cmd_vel_msg.angular.z = float(angular_velocity)
                    
                    self.cmd_vel_pub.publish(cmd_vel_msg)
                    
                    # Only log orientation progress periodically to reduce volume
                    if int(recovery_duration * 2) % 2 == 0:  # Log roughly twice per second
                        log_structured('recovery', 'ORIENT_PROGRESS', 
                                    f"Recovery orient progress: {angular_degrees:.2f}°", 
                                    {'angular_error': angular_degrees,
                                    'velocity': angular_velocity,
                                    'time_in_phase': recovery_duration - 1.0})
                else:
                    # If angular error is small, stop rotation
                    self.stop_robot()
                    
                    # Log alignment success
                    log_structured('recovery', 'ORIENT_COMPLETE', 
                                f"Recovery orient: good alignment achieved ({angular_degrees:.2f}°)", 
                                {'final_angle': angular_degrees})
                    
                    # Consider early completion if alignment is good
                    if recovery_duration > 2.0:
                        self.recovery_phase = "approach"
                        
                        # Log early phase transition
                        log_structured('recovery', 'PHASE_TRANSITION', 
                                    "Recovery: Moving to approach phase (early completion)", 
                                    {'from_phase': prev_phase,
                                    'to_phase': self.recovery_phase,
                                    'duration_in_prev': recovery_duration - 1.0,
                                    'reason': 'alignment_success'})
            else:
                # No target available
                if self.current_target is None:
                    log_structured('recovery', 'ORIENT_NO_TARGET', 
                                "Cannot orient in recovery - no target available", 
                                {'time_in_phase': recovery_duration - 1.0},
                                level=logging.WARNING)
                elif not self._is_orientation_fresh():
                    log_structured('recovery', 'ORIENT_STALE_ORIENTATION', 
                                "Cannot orient in recovery - orientation data stale", 
                                {'time_in_phase': recovery_duration - 1.0},
                                level=logging.WARNING)
            
            # After 3 seconds in orient phase, move to approach
            if recovery_duration > 3.0:
                self.recovery_phase = "approach"
                
                # Log phase transition
                log_structured('recovery', 'PHASE_TRANSITION', 
                            "Recovery: Moving to approach phase", 
                            {'from_phase': prev_phase,
                            'to_phase': self.recovery_phase,
                            'duration_in_prev': recovery_duration - 1.0})
                
                # Reset controllers for approach
                self.pid_linear_x.reset()
                self.pid_linear_y.reset()
                self.distance_error_tracker.reset()
                self.lateral_error_tracker.reset()
                
        # Phase 3: Approach (3+ seconds)
        elif self.recovery_phase == "approach":
            # Only approach if we have a target
            if self.current_target is not None and self._is_orientation_fresh():
                # Use filtered values
                distance = self.filtered_distance
                lateral = self.filtered_lateral
                
                # Calculate errors
                distance_error = distance - self.desired_distance
                lateral_error = lateral
                
                # Only move if errors are significant
                if abs(distance_error) > 0.1 or abs(lateral_error) > 0.1:
                    # Compute velocities
                    linear_x_velocity = self.pid_linear_x.compute(
                        distance_error,
                        current_time,
                        force_zero=False
                    ) * 0.7  # Apply conservative scaling
                    
                    lateral_velocity = self.pid_linear_y.compute(
                        lateral_error,
                        current_time,
                        force_zero=False
                    ) * 0.7  # Apply conservative scaling
                    
                    # Create and publish Twist message
                    cmd_vel_msg = self._cmd_vel_msg
                    cmd_vel_msg.linear.x = float(linear_x_velocity)
                    cmd_vel_msg.linear.y = float(lateral_velocity)
                    cmd_vel_msg.angular.z = 0.0
                    
                    self.cmd_vel_pub.publish(cmd_vel_msg)
                    
                    # Only log approach progress periodically to reduce volume
                    if int(recovery_duration * 2) % 2 == 0:  # Log roughly twice per second
                        log_structured('recovery', 'APPROACH_PROGRESS', 
                                    f"Recovery approach progress", 
                                    {'distance_error': distance_error,
                                    'lateral_error': lateral_error,
                                    'velocity_fwd': linear_x_velocity,
                                    'velocity_lat': lateral_velocity,
                                    'time_in_phase': recovery_duration - 3.0})
                else:
                    # If errors are small, stop movement
                    self.stop_robot()
                    
                    # Log approach success
                    log_structured('recovery', 'APPROACH_COMPLETE', 
                                "Recovery approach: good position achieved", 
                                {'distance_error': distance_error,
                                'lateral_error': lateral_error,
                                'within_threshold': abs(distance_error) < self.distance_threshold and 
                                                    abs(lateral_error) < self.lateral_threshold})
                    
                    # Consider recovery complete if position is good
                    recovery_trigger = "state_transition"
                    if hasattr(self, 'recovery_trigger'):
                        recovery_trigger = self.recovery_trigger
                        
                    log_structured('recovery', 'RECOVERY_SUCCESS', 
                                "Recovery sequence successful, ready to return to tracking", 
                                {'duration': recovery_duration,
                                'original_trigger': recovery_trigger})
            else:
                # No target available during approach
                if self.current_target is None:
                    log_structured('recovery', 'APPROACH_NO_TARGET', 
                                "Cannot approach in recovery - no target available", 
                                {'time_in_phase': recovery_duration - 3.0},
                                level=logging.WARNING)
            
            # After 6 seconds in recovery, suggest exiting recovery mode
            if recovery_duration > 6.0 and int(recovery_duration) % 2 == 0:  # Check every 2 seconds
                log_structured('recovery', 'RECOVERY_DURATION_LIMIT', 
                            "Recovery has been active for extended period", 
                            {'duration': recovery_duration,
                            'phase': self.recovery_phase,
                            'suggestion': "Consider state transition to tracking mode"})
    
    def _reset_stopped_state_if_needed(self, distance_error, lateral_error, angular_error):
        """
        Reset stopped state if significant movement is required, with improved logic.
        
        Args:
            distance_error: Error in distance (meters)
            lateral_error: Error in lateral position (meters)
            angular_error: Error in angular position (degrees)
            
        Returns:
            bool: True if stopped state was reset, False otherwise
        """
        # Skip if not in stopped state
        if not self._robot_stopped:
            return False
        
        # Initialize decision managers if needed
        self._initialize_decision_managers()
        
        # Evaluate movement conditions using manager
        current_time = time.time()
        should_move, reason = self._movement_decision_manager.evaluate_movement_conditions(
            distance_error, lateral_error, angular_error, current_time
        )
        
        if should_move:
            # Log the transition
            stop_duration = current_time - self._movement_decision_manager.last_stop_time
            
            log_structured('motion_control', 'MOVEMENT_STARTED',
                        reason,
                        {'stop_duration': stop_duration,
                        'initial_boost': self._movement_decision_manager.movement_strategy.initial_movement_boost,
                        'hysteresis': getattr(self, '_movement_hysteresis', 0.0),
                        'robot_state': self.robot_state})
            
            self.get_logger().info(
                f"Exiting stopped state - {reason}"
            )
            
            # Reset stopped state
            self._robot_stopped = False
            
            # Reset movement hysteresis
            self._movement_hysteresis = 0.0
            
            return True
        
        return False
    
    def _evaluate_stop_conditions(self, distance, lateral, angular_degrees, is_stopped):
        """
        Evaluate if the robot should move based on current conditions with improved logic.
        
        Args:
            distance: Current distance to target
            lateral: Current lateral offset
            angular_degrees: Current angular error in degrees
            is_stopped: Whether the robot is currently stopped
            
        Returns:
            tuple: (should_stop, reason) - True if robot should stop, False if it should move
        """
        # Calculate error values
        distance_error = distance - self.desired_distance  # Can be negative if overshot
        lateral_error = lateral
        angular_error = angular_degrees
        
        # Initialize decision managers if needed
        self._initialize_decision_managers()
        
        # Prepare current velocities
        velocities = {
            'linear_x': getattr(self, 'last_cmd_vel', (0.0, 0.0, 0.0))[0],
            'linear_y': getattr(self, 'last_cmd_vel', (0.0, 0.0, 0.0))[1],
            'angular': getattr(self, 'last_cmd_vel', (0.0, 0.0, 0.0))[2]
        }
        
        # Evaluate stop conditions using manager
        should_stop, reason = self._stop_decision_manager.evaluate_stop_conditions(
            distance_error, lateral_error, angular_error, is_stopped, velocities
        )
        
        # If stopping, update stop time
        if should_stop and not is_stopped:
            self._movement_decision_manager.record_stop_time(time.time())
        
        # Log detailed information at higher debug levels
        if self.debug_level >= 1:
            if should_stop != is_stopped:  # Only log state changes to reduce noise
                self.get_logger().info(
                    f"STOP DECISION: {'STOP' if should_stop else 'MOVE'} - {reason}"
                )
        
        return should_stop, reason
    
    def _apply_velocity_limits(self, linear_x, linear_y, angular_z, current_time):
        """
        Apply velocity and acceleration limits with enhanced approach behavior.
        
        Args:
            linear_x: Calculated forward velocity
            linear_y: Calculated lateral velocity
            angular_z: Calculated angular velocity
            current_time: Current time for acceleration limiting
                
        Returns:
            tuple: (limited_linear_x, limited_linear_y, limited_angular_z)
        """
        # Store original velocities for comparison and logging
        original_velocities = [linear_x, linear_y, angular_z]
        
        # Initialize velocity limiter if needed
        if not hasattr(self, '_velocity_limiter_manager'):
            # Collect robot parameters for limiter
            robot_params = {
                'approach_distance': self.approach_distance,
                'min_approach_factor': self.min_approach_factor,
                'linear_x_max': self.linear_x_max,
                'linear_y_max': self.linear_y_max,
                'angular_max': self.angular_max,
                'accel_limit': 1.5,  # Base acceleration limit
                'angular_accel_limit': 1.0  # Base angular acceleration limit
            }
            
            # Create the velocity limiter manager
            self._velocity_limiter_manager = VelocityLimiterManager(
                robot_params, self.debug_level
            )
            
        # Calculate time since last control step
        dt = current_time - getattr(self, "last_accel_time", current_time - 0.1)
        self.last_accel_time = current_time
        dt = max(0.001, min(dt, 0.1))  # Constrain dt to reasonable range
        
        # Get distance error if available
        distance_error = 0.0
        is_approaching = False
        if hasattr(self, 'filtered_distance') and hasattr(self, 'desired_distance'):
            distance_error = self.filtered_distance - self.desired_distance
            
            # Determine if we're approaching the target
            if abs(distance_error) < self.approach_distance * 1.5:
                is_approaching = True
        
        # Apply velocity limits
        limited_velocities, limited_components = self._velocity_limiter_manager.apply_velocity_limits(
            [linear_x, linear_y, angular_z],
            self.last_cmd_vel,
            distance_error,
            is_approaching,
            dt
        )
        
        # Log if significant limiting occurred
        if limited_components and self.debug_level >= 1:
            self.get_logger().info(f"Velocity limited: {', '.join(limited_components)}")
        
        return tuple(limited_velocities)    
    def stop_robot(self):
        """Send a command to stop all robot motion immediately and reset controllers with improved logging."""
        # Store previous motion state for transition logging
        was_moving = not getattr(self, '_robot_stopped', True)
        prev_velocity = getattr(self, 'last_cmd_vel', (0.0, 0.0, 0.0))
        
        # Reuse cmd_vel message and set all fields to 0
        self._cmd_vel_msg.linear.x = 0.0
        self._cmd_vel_msg.linear.y = 0.0
        self._cmd_vel_msg.linear.z = 0.0
        self._cmd_vel_msg.angular.x = 0.0
        self._cmd_vel_msg.angular.y = 0.0
        self._cmd_vel_msg.angular.z = 0.0

        # Add stronger integral reset
        prev_integral_values = {
            'linear_x': getattr(self.pid_linear_x, 'integral', 0.0),
            'linear_y': getattr(self.pid_linear_y, 'integral', 0.0),
            'angular': getattr(self.pid_angular, 'integral', 0.0)
        }
        
        self.pid_linear_x.integral = 0.0  # Complete reset
        self.pid_linear_y.integral = 0.0
        self.pid_angular.integral = 0.0
        
        # Log integral reset
        if was_moving:
            log_structured('motion_control', 'INTEGRAL_RESET', 
                        "PID integral terms reset on stop", 
                        {'prev_values': prev_integral_values,
                        'reason': getattr(self, 'stop_reason', 'stop_robot_called')})
        
        # Publish stop command multiple times to ensure it's received
        for _ in range(3):
            self.cmd_vel_pub.publish(self._cmd_vel_msg)
            time.sleep(0.01)  # Small delay between publishes
        
        # Reset last command velocity for acceleration limiting
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        
        # Clear velocity history
        self.velocity_history = LightweightBuffer(max_size=6)
        
        # Set a "stopped" state flag and timestamp
        was_stopped = self._robot_stopped
        self._robot_stopped = True
        self._stop_time = time.time()
        
        # Remember our position when we stopped
        self._last_stop_position = (
            self.current_distance,
            self.current_lateral, 
            self.current_bearing
        )
        
        # Only log if this is an actual transition from moving to stopped
        if was_moving:
            robot_state = getattr(self, 'robot_state', 'unknown')
            stop_reason = getattr(self, 'stop_reason', None)
            
            log_structured('motion_control', 'ROBOT_STOPPED', 
                        "Robot motion stopped", 
                        {'prev_velocity': prev_velocity,
                        'robot_state': robot_state,
                        'stop_reason': stop_reason,
                        'position': self._last_stop_position[:2]})  # Only include distance & lateral
                        
            # Clear the stop reason to avoid reusing it
            if hasattr(self, 'stop_reason'):
                del self.stop_reason
                
            self.get_logger().info("Robot stopped! All velocities reset.")
    
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
        
        # Reset target filter
        self.target_filter.reset()
        
        # Reset strategy blender
        self.strategy_blender.reset()
        
        # Reset error categorization state
        self.prev_distance_category = "none"
        self.prev_lateral_category = "none"
        self.prev_angular_category = "none"
        
        # Reset motion state
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        self.velocity_history = LightweightBuffer(max_size=6)
        
        # Reset last logged command
        self.last_logged_cmd = (0.0, 0.0, 0.0)
        
        # Reset movement hysteresis
        self._movement_hysteresis = 0.0
        
        # Set stopped state
        self._robot_stopped = True
        
        self.get_logger().info("Complete controller reset performed")
    
    def _init_strategy_table(self):
        """Initialize the table-driven movement strategy definitions with improved approach and balanced strategies."""
        # Define the strategy table with improved angular-first strategies and approach strategies
        self.strategy_table = {
            # All errors within deadbands - no movement
            ("none", "none", "none"): [
                "NO_MOVEMENT", False, False, False, 
                0.0, 0.0, 0.0, 
                "All errors within deadbands"
            ],
            
            # Very small errors - minimal corrections
            ("very_small", "very_small", "very_small"): [
                "MINIMAL_CORRECTION", True, True, True, 
                0.5, 0.4, 0.4,
                "Minimal corrections for very small errors"
            ],
            
            # New high-priority strategy for large distance + any lateral/angular error
            ("large", "*", "*"): [
                "DISTANCE_PRIORITY_APPROACH", True, True, True,
                0.9, 0.6, 0.5,  # Strong forward, good lateral, moderate angular
                "Distance-priority approach: {distance_error:.2f}m"
            ],
            
            # Large distance with large lateral - fast diagonal approach
            ("large", "large", "*"): [
                "FAST_DIAGONAL_APPROACH", True, True, True,
                0.9, 0.9, 0.4,  # Strong forward and lateral, moderate angular
                "Fast diagonal approach: {distance_error:.2f}m, {lateral_error:.2f}m"
            ],
            
            # Large distance with medium lateral
            ("large", "medium", "*"): [
                "FAST_DIAGONAL", True, True, True,
                1.0, 0.8, 0.5,  # Maximum forward, strong lateral, moderate angular
                "Fast diagonal approach: {distance_error:.2f}m, {lateral_error:.2f}m"
            ],
            
            # Medium distance with large lateral
            ("medium", "large", "*"): [
                "LATERAL_PRIORITY", True, True, True,
                0.7, 1.0, 0.4,  # Good forward, maximum lateral, moderate angular
                "Lateral-priority movement: {lateral_error:.2f}m"
            ],
            
            # Pure lateral correction at target distance
            ("none", "very_small", "none"): [
                "MICRO_LATERAL", False, True, False,
                0.0, 0.7, 0.0,
                "Micro lateral correction at target distance: {lateral_error:.2f}m"
            ],
            
            ("none", "small", "none"): [
                "LATERAL_CORRECTION", False, True, False,
                0.0, 0.9, 0.0,
                "Lateral correction at target distance: {lateral_error:.2f}m"
            ],
            
            ("none", "medium", "none"): [
                "STRONG_LATERAL", False, True, False, 
                0.0, 1.0, 0.0, 
                "Strong lateral correction at target distance: {lateral_error:.2f}m"
            ],
            
            ("none", "large", "none"): [
                "MAX_LATERAL", False, True, False, 
                0.0, 1.0, 0.0, 
                "Maximum lateral correction at target distance: {lateral_error:.2f}m"
            ],
            
            # Pure distance corrections (no lateral, no angular)
            ("very_small", "none", "none"): [
                "MICRO_APPROACH", True, False, False, 
                0.7, 0.0, 0.0,
                "Micro distance adjustment: {distance_error:.2f}m"
            ],
            
            ("small", "none", "none"): [
                "FORWARD_ADJUSTMENT", True, False, False, 
                0.8, 0.0, 0.0,
                "Small distance adjustment: {distance_error:.2f}m"
            ],
            
            ("medium", "none", "none"): [
                "FORWARD_APPROACH", True, False, False, 
                1.0, 0.0, 0.0,  # Increased from 0.9 to 1.0 for faster approach
                "Medium distance approach: {distance_error:.2f}m"
            ],
            
            ("large", "none", "none"): [
                "FULL_APPROACH", True, False, False, 
                1.0, 0.0, 0.0,
                "Full distance approach: {distance_error:.2f}m"
            ],
            
            # Angular-first strategies by magnitude - IMPROVED
            ("*", "*", "very_large"): [
                "ANGULAR_PRIMARY", True, True, True,  # Enabled lateral movement
                0.4, 0.6, 0.9,  # Increased forward from 0.3 to 0.4, added lateral 0.3
                "Angular correction with approach: {angular_error:.1f}°"
            ],
            
            ("*", "*", "large"): [
                "ANGULAR_PRIORITY", True, True, True,  # Enabled lateral
                0.5, 0.4, 0.8,  # Increased forward from 0.4, added lateral 0.4
                "Angular correction with steady approach: {angular_error:.1f}°"
            ],
            
            ("*", "*", "medium_large"): [
                "ANGULAR_BALANCED", True, True, True,
                0.6, 0.5, 0.7,  # Increased forward from 0.5 to 0.6, lateral from 0.2 to 0.5
                "Balanced approach with angular correction: {angular_error:.1f}°"
            ],
            
            ("*", "*", "medium"): [
                "BALANCED", True, True, True, 
                0.7, 0.5, 0.6,  # Increased forward from 0.6 to 0.7, lateral from 0.3 to 0.5
                "Balanced movement with angular correction: {angular_error:.1f}°"
            ],
            
            ("*", "*", "small_medium"): [
                "FORWARD_ANGULAR", True, True, True, 
                0.8, 0.6, 0.5,  # Increased forward from 0.7 to 0.8, lateral from 0.4 to 0.6
                "Forward movement with angular fine-tuning: {angular_error:.1f}°"
            ],
            
            ("*", "*", "small"): [
                "FORWARD_PRIMARY", True, True, True, 
                0.9, 0.7, 0.4,  # Increased forward from 0.8 to 0.9, lateral from 0.5 to 0.7
                "Forward-focused movement with minor angular correction: {angular_error:.1f}°"
            ],
            
            ("*", "*", "very_small"): [
                "POSITION_WITH_ALIGNMENT", True, True, True, 
                1.0, 0.8, 0.3,  # Increased forward from 0.9 to 1.0, lateral from 0.6 to 0.8
                "Position-focused movement with subtle alignment: {angular_error:.1f}°"
            ],
            
            # Special case strategies for when at target distance with angular error - IMPROVED
            ("none", "*", "medium"): [
                "AT_TARGET_ANGULAR", True, True, True,  # Enabled lateral movement
                0.4, 0.3, 0.6,  # Increased forward from 0.2 to 0.4, added lateral 0.3
                "At target distance - angular correction with movement: {angular_error:.1f}°"
            ],
            
            ("none", "*", "medium_large"): [
                "AT_TARGET_ANGULAR_STRONG", True, True, True,  # Enabled lateral movement
                0.3, 0.3, 0.7,  # Increased forward from 0.1 to 0.3, added lateral 0.3
                "At target distance - angular correction with movement: {angular_error:.1f}°"
            ],
            
            ("none", "*", "large"): [
                "AT_TARGET_ANGULAR_MAX", True, True, True,  # Enabled forward and lateral
                0.2, 0.3, 0.8,  # Added forward 0.2, lateral 0.3
                "At target distance - angular correction with movement: {angular_error:.1f}°"
            ],
            
            # Combined distance + lateral but no angular - IMPROVED
            ("very_small", "very_small", "none"): [
                "FINE_POSITION_ADJUSTMENT", True, True, False,
                0.5, 0.7, 0.0,  # Increased forward from 0.4 to 0.5, lateral from 0.6 to 0.7
                "Fine position adjustment with lateral emphasis"
            ],
            
            ("small", "small", "none"): [
                "POSITION_ADJUSTMENT", True, True, False,
                0.7, 0.8, 0.0,  # Increased forward from 0.6 to 0.7
                "Small position adjustment with lateral priority"
            ],
            
            ("medium", "small", "none"): [
                "APPROACH_WITH_LATERAL", True, True, False,
                0.9, 0.7, 0.0,  # Increased forward from 0.8 to 0.9, lateral from 0.6 to 0.7
                "Approach with lateral correction"
            ],
            
            ("small", "medium", "none"): [
                "LATERAL_WITH_APPROACH", True, True, False,
                0.7, 0.9, 0.0,  # Increased forward from 0.6 to 0.7, lateral from 0.8 to 0.9
                "Lateral correction with approach component"
            ],
            
            # Combined distance + angular without lateral - IMPROVED
            ("small", "none", "small"): [
                "APPROACH_WITH_ALIGNMENT", True, True, True,  # Enabled lateral for minor corrections
                0.8, 0.2, 0.5,  # Increased forward from 0.7 to 0.8, added lateral 0.2
                "Approach with alignment correction"
            ],
            
            ("medium", "none", "small"): [
                "APPROACH_WITH_MINOR_ALIGNMENT", True, True, True,  # Enabled lateral
                0.9, 0.2, 0.4,  # Increased forward from 0.8 to 0.9, added lateral 0.2
                "Focused approach with minor alignment"
            ],
            
            # Angular-first based on distance - IMPROVED
            ("large", "*", "medium"): [
                "ANGULAR_THEN_APPROACH", True, True, True,  # Enabled lateral
                0.7, 0.4, 0.6,  # Increased forward from 0.5 to 0.7, added lateral 0.4
                "Angular correction with approach from distance"
            ],
            
            # Diagonal movement - IMPROVED
            ("medium", "medium", "small"): [
                "DIAGONAL_MOVEMENT", True, True, True,
                0.8, 0.8, 0.4,  # Increased angular from 0.3 to 0.4
                "Diagonal movement with small angular correction"
            ],
            
            ("medium", "medium", "none"): [
                "PURE_DIAGONAL", True, True, False,
                0.9, 0.9, 0.0,
                "Pure diagonal movement without rotation"
            ],
            
            # Approach strategies for near-target behavior - IMPROVED
            ("small", "*", "*"): [
                "APPROACH", True, True, True, 
                0.8, 0.8, 0.4,  # Increased forward from 0.7 to 0.8, lateral from 0.7 to 0.8
                "Approach mode - nearing target: {distance_error:.2f}m"
            ],
                       
            
            # Position correction without rotation - IMPROVED
            ("*", "*", "none"): [
                "POSITION_ONLY", True, True, False,
                0.9, 0.9, 0.0,  # Increased from 0.8 to 0.9
                "Position correction without rotation"
            ],
            
            # Fallback strategy - IMPROVED
            ("*", "*", "*"): [
                "BALANCED", True, True, True, 
                0.8, 0.7, 0.5,  # Increased forward from 0.7 to 0.8, lateral from 0.6 to 0.7
                "Balanced movement strategy (fallback)"
            ],

            # ADDED: Deceleration strategy for controlled approach at close range
            ("very_small", "*", "*"): [
                "DECELERATION_APPROACH", True, True, True, 
                0.4, 0.6, 0.3,  # Reduced forward from 0.6 to 0.4 for controlled approach
                "Deceleration approach - very close to target: {distance_error:.2f}m"
            ],

            # Add a new strategy for close approaches
            ("very_small", "*", "*"): [
                "FINAL_APPROACH", True, True, True, 
                0.4, 0.6, 0.3,  # Reduced forward from 0.6 to 0.4
                "Final careful approach - very close to target: {distance_error:.2f}m"
            ],

            ("*", "medium", "medium"): [
                "HYBRID_ANGULAR_LATERAL", True, True, True,
                0.6, 0.8, 0.7,  # Good forward, strong lateral, strong angular
                "Hybrid angular-lateral correction: lateral={lateral_error:.2f}m, angular={angular_error:.1f}°"
            ]
        }
    
    def _categorize_error(self, error, error_type="distance", prev_category=None, lenient_factor=1.0):
        """
        Categorize an error value with hysteresis to prevent oscillation.
        Modified to support lenient categorization for angular errors.
        
        Args:
            error: The error value to categorize
            error_type: The type of error (distance, lateral, angular)
            prev_category: Previous category for hysteresis
            lenient_factor: Factor to make categories more lenient (higher means more lenient)
            
        Returns:
            String: The error category
        """
        abs_error = abs(error)
        
        # Select appropriate thresholds based on error type
        if error_type == "angular":
            # MODIFIED: Significantly increased angular thresholds
            deadband = 4.0  # Increased from 3.0 degrees
            very_small_threshold = deadband * lenient_factor
            small_threshold = deadband * 2.0 * lenient_factor
            small_medium_threshold = deadband * 3.0 * lenient_factor
            medium_threshold = deadband * 9.0/4.0 * lenient_factor  # Adjusted to make 9.0 degrees
            medium_large_threshold = deadband * 8.0 * lenient_factor  # Increased
            large_threshold = deadband * 12.0 * lenient_factor  # Increased
            very_large_threshold = deadband * 16.0 * lenient_factor  # Increased
        elif error_type == "lateral":
            # MODIFIED: Increased lateral thresholds
            deadband = 0.08  
            very_small_threshold = deadband
            small_threshold = deadband * 1.8
            small_medium_threshold = deadband * 3.0
            medium_threshold = deadband * 4.0
            medium_large_threshold = deadband * 6.0
            large_threshold = deadband * 8.0
            very_large_threshold = deadband * 10.0
        else:  # distance
            # MODIFIED: Increased distance thresholds
            deadband = 0.15  # Increased from 0.1 meters
            very_small_threshold = deadband
            small_threshold = deadband * 2.0  # Increased from 1.5
            small_medium_threshold = deadband * 3.0  # Increased from 2.5
            medium_threshold = deadband * 4.0
            medium_large_threshold = deadband * 6.0
            large_threshold = deadband * 8.0
            very_large_threshold = deadband * 10.0
        
        # Apply hysteresis if previous category is provided
        if prev_category and prev_category != "none":
            # MODIFIED: Apply different hysteresis factors based on error type
            if error_type == "angular":
                # Stronger hysteresis for angular errors to prevent oscillation
                hysteresis_down = 0.35  # 35% hysteresis for moving down a category
                hysteresis_up = 0.15    # 15% hysteresis for moving up
            else:
                # Standard hysteresis for other error types
                hysteresis_down = 0.2   # For moving down a category
                hysteresis_up = 0.1     # For moving up a category (easier to go up than down)
            
            # Going down from a higher category - apply stronger hysteresis
            if prev_category == "very_large" and abs_error > very_large_threshold * (1.0 - hysteresis_down):
                return "very_large"
            elif prev_category == "large":
                if abs_error > large_threshold * (1.0 - hysteresis_down):
                    return "large"
                # Moving up requires less hysteresis
                if abs_error > very_large_threshold * (1.0 - hysteresis_up):
                    return "very_large"
                
            elif prev_category == "medium_large":
                if abs_error > medium_large_threshold * (1.0 - hysteresis_down):
                    return "medium_large"
                # Moving up requires less hysteresis
                if abs_error > large_threshold * (1.0 - hysteresis_up):
                    return "large"
                
            elif prev_category == "medium":
                if abs_error > medium_threshold * (1.0 - hysteresis_down):
                    return "medium"
                # Moving up requires less hysteresis
                if abs_error > medium_large_threshold * (1.0 - hysteresis_up):
                    return "medium_large"
                    
            elif prev_category == "small_medium":
                if abs_error > small_medium_threshold * (1.0 - hysteresis_down):
                    return "small_medium"
                # Moving up requires less hysteresis
                if abs_error > medium_threshold * (1.0 - hysteresis_up):
                    return "medium"
                
            elif prev_category == "small":
                if abs_error > small_threshold * (1.0 - hysteresis_down):
                    return "small"
                # Moving up requires less hysteresis
                if abs_error > small_medium_threshold * (1.0 - hysteresis_up):
                    return "small_medium"
                
            elif prev_category == "very_small":
                if abs_error > very_small_threshold * (1.0 - hysteresis_down):
                    return "very_small"
                # Moving up requires less hysteresis
                if abs_error > small_threshold * (1.0 - hysteresis_up):
                    return "small"
        
        # Handle lateral control toggle
        if error_type == "lateral" and not getattr(self, 'use_lateral_control', True):
            return "none"
        
        # Categorize based on absolute error value
        if abs_error <= deadband:
            return "none"
        elif abs_error <= very_small_threshold:
            return "very_small"
        elif abs_error <= small_threshold:
            return "small"
        elif abs_error <= small_medium_threshold:
            return "small_medium"
        elif abs_error <= medium_threshold:
            return "medium"
        elif abs_error <= medium_large_threshold:
            return "medium_large"
        elif abs_error <= large_threshold:
            return "large"
        else:
            return "very_large"
    
    
    def _determine_movement_strategy(self, distance_error, lateral_error, angular_error_degrees,
                           prev_distance_category=None, prev_lateral_category=None, 
                           prev_angular_category=None):
        """
        Determine the optimal movement strategy using table-driven approach
        with hysteresis and angular-first prioritization.
        Modified to reduce angular corrections when at target distance and
        prioritize forward movement during startup.
        
        Args:
            distance_error: Error in distance (meters)
            lateral_error: Error in lateral position (meters)
            angular_error_degrees: Error in angular position (degrees)
            prev_*_category: Previous error categories for hysteresis
            
        Returns:
            dict: Strategy information including strategy name, movement flags, and scale factors
        """
        current_time = time.time()
        
        # Add startup movement counter if it doesn't exist
        if not hasattr(self, '_startup_movement_cycles'):
            self._startup_movement_cycles = 0
        
        # For first few cycles after movement starts, prioritize forward movement
        if self._robot_stopped == False and self._startup_movement_cycles < 5:
            self._startup_movement_cycles += 1
            
            # If significant distance error exists, prioritize forward movement
            if abs(distance_error) > 0.2:  # Significant distance error
                # Override strategy to prioritize forward movement
                name = "STARTUP_FORWARD_PRIORITY"
                use_forward = True
                use_lateral = True
                use_angular = True
                forward_scale = 0.7  # Start with moderate forward scale
                lateral_scale = 0.3  # Reduced lateral scale during startup
                angular_scale = 0.3  # Reduced angular scale during startup
                reason = "Startup forward priority - quick response mode"
                
                # Log the startup strategy selection
                log_strategy_selection(
                    "startup_forward", 
                    {
                        "strategy_name": name,
                        "use_forward": use_forward,
                        "use_lateral": use_lateral,
                        "use_angular": use_angular,
                        "forward_scale": forward_scale,
                        "lateral_scale": lateral_scale,
                        "angular_scale": angular_scale,
                        "reason": reason
                    },
                    reason=f"Startup cycle {self._startup_movement_cycles}/5, distance_error={distance_error:.3f}m"
                )
                
                # Create and return startup strategy
                return {
                    "strategy_name": name,
                    "use_forward": use_forward,
                    "use_lateral": use_lateral,
                    "use_angular": use_angular,
                    "forward_scale": forward_scale,
                    "lateral_scale": lateral_scale,
                    "angular_scale": angular_scale,
                    "reason": reason
                }
        
        # Check if robot is at target distance and reduce angular priority if so
        at_target_distance = abs(distance_error) < self.distance_threshold * 1.5
        
        # Categorize errors into states with hysteresis
        self._key_tuple[0] = self._categorize_error(
            distance_error, "distance", prev_distance_category)
        self._key_tuple[1] = self._categorize_error(
            lateral_error, "lateral", prev_lateral_category)
        
        # Modify angular error categorization when at target distance
        if at_target_distance and self._key_tuple[0] == "none":
            # At target distance, categorize angular errors more leniently
            lenient_factor = 1.5  # Make angular categories 50% more lenient
            
            angular_category = self._categorize_error(
                angular_error_degrees, 
                "angular", 
                prev_angular_category,
                lenient_factor=lenient_factor
            )
            
            # Optionally downgrade categories for angular errors when at target distance
            original_category = angular_category
            if angular_category == "medium":
                angular_category = "small_medium"  # Downgrade medium to small_medium
            elif angular_category == "small_medium":
                angular_category = "small"  # Downgrade small_medium to small
            elif angular_category == "small":
                angular_category = "very_small"  # Downgrade small to very_small
                    
            self._key_tuple[2] = angular_category
            
            # Only log if category actually changed
            if original_category != angular_category:
                log_error_categorization(
                    "angular", 
                    angular_error_degrees, 
                    angular_category, 
                    self.angular_threshold * lenient_factor,
                    prev_category=original_category
                )
        else:
            # Normal angular categorization
            self._key_tuple[2] = self._categorize_error(
                angular_error_degrees, "angular", prev_angular_category)
        
        # Save categories for next iteration's hysteresis
        self.prev_distance_category = self._key_tuple[0]
        self.prev_lateral_category = self._key_tuple[1]
        self.prev_angular_category = self._key_tuple[2]
        
        # Create lookup key
        key = tuple(self._key_tuple)
        
        # Get strategy definition from table
        strategy_def = self._match_strategy(key, self.strategy_table)
        
        # Format the reason string with actual error values
        name, use_forward, use_lateral, use_angular, forward_scale, lateral_scale, angular_scale, reason_template = strategy_def
        
        try:
            reason = reason_template.format(
                distance_error=abs(distance_error),
                lateral_error=abs(lateral_error),
                angular_error=abs(angular_error_degrees)
            )
        except KeyError as e:
            # Handle missing format key gracefully
            reason = reason_template
        
        # Create a MovementStrategy object
        target_strategy = MovementStrategy(
            name, use_forward, use_lateral, use_angular,
            forward_scale, lateral_scale, angular_scale, reason
        )
        
        # Additional logic for at-target-distance
        if at_target_distance and (name != "NO_MOVEMENT" and angular_scale > 0.0):
            # Reduce angular scale further if already at target distance
            original_angular_scale = target_strategy.angular_scale
            adjusted_angular_scale = original_angular_scale * 0.8  # Reduce by an additional 20%
            target_strategy.angular_scale = adjusted_angular_scale
        
        # Update the strategy blender with the target strategy
        blend_started = self.strategy_blender.update_target(target_strategy, current_time)
        
        # Get the current (possibly blended) strategy
        current_strategy = self.strategy_blender.get_current_strategy(current_time)
        
        # Log strategy blending if happening
        if self.strategy_blender.is_blending():
            blend_factor = self.strategy_blender.get_blend_progress(current_time) / 100.0
            blend_duration = getattr(self.strategy_blender, 'effective_blend_duration', 
                                    self.strategy_blender.blend_duration)
            
            log_strategy_blending(
                self.strategy_blender.current_strategy, 
                target_strategy, 
                blend_factor, 
                blend_duration
            )
        
        # Keep track of the current strategy name for logging
        self.current_strategy = current_strategy.name
        
        # Log strategy changes (throttled) with more detailed logging when strategy changes
        if blend_started or self.debug_level >= 2:
            if self._log_throttled(
                self.get_logger().info,
                f"Strategy selected: {current_strategy.name}, params: "
                f"forward={current_strategy.forward_scale:.1f}, "
                f"lateral={current_strategy.lateral_scale:.1f}, "
                f"angular={current_strategy.angular_scale:.1f}",
                1.0,  # Throttle to once per second
                'last_strategy_log_time'
            ):
                # Log error categories used for selection with this strategy change
                log_strategy_selection(
                    key, 
                    current_strategy.as_dict(),
                    candidate_strategies={name: strategy_def[0] for name, strategy_def in
                                        [(key, strategy_def)]},
                    reason=f"Error categories: distance={self._key_tuple[0]}, "
                        f"lateral={self._key_tuple[1]}, angular={self._key_tuple[2]}"
                )
        
        # Convert to dictionary for compatibility with existing code
        return current_strategy.as_dict()

    def _match_strategy(self, key, strategies):
        """
        Match a key against the strategy table with wildcard support.
        
        Args:
            key: Tuple of (distance_state, lateral_state, angular_state)
            strategies: Strategy table dictionary
            
        Returns:
            List: The matched strategy definition
        """
        # First try exact match
        if key in strategies:
            return strategies[key]
        
        # Support wildcards
        d_state, l_state, a_state = key
        
        # Try wildcard matches in order of specificity
        # First: match two specific states, one wildcard
        patterns_to_try = [
            # Two specific, one wildcard
            (d_state, l_state, "*"),
            (d_state, "*", a_state),
            ("*", l_state, a_state),
            
            # One specific, two wildcards
            (d_state, "*", "*"),
            ("*", l_state, "*"),
            ("*", "*", a_state),
        ]
        
        for pattern in patterns_to_try:
            if pattern in strategies:
                # Only log at debug level to reduce verbosity
                if self.debug_level >= 2:
                    self.get_logger().debug(
                        f"Strategy wildcard match: {key} → {pattern}, " 
                        f"strategy={strategies[pattern][0]}"
                    )
                return strategies[pattern]
        
        # Fallback
        return strategies[("*", "*", "*")]
    
    def _log_periodic_status(self):
        """Log periodic status updates."""
        # Only log if debug level supports it
        if self.debug_level < 1:
            return
            
        # Get current CPU usage
        cpu_usage = self.current_cpu_usage
        
        # Calculate cycle time stats
        cycle_time_avg = 0.0
        if hasattr(self, 'cycle_duration_avg'):
            cycle_time_avg = self.cycle_duration_avg * 1000.0  # Convert to ms
        
        # Log status
        self.get_logger().info(
            f"Status: Robot state={self.robot_state}, "
            f"Strategy={self.current_strategy}, "
            f"CPU={cpu_usage:.1f}%, "
            f"Cycle time={cycle_time_avg:.2f}ms, "
            f"Update rate={self.update_rate:.1f}Hz"
        )
    
    def publish_diagnostics(self):
        """Publish detailed diagnostic information at a slower rate."""
        if self._shutting_down:
            return
            
        # Calculate velocity statistics
        vel_data = self.velocity_history.get_all()
        if not vel_data:
            return
            
        # Extract linear and angular velocities
        lin_x_velocities = [v[0] for v in vel_data]
        lin_y_velocities = [v[1] for v in vel_data]
        ang_velocities = [v[2] for v in vel_data]
        
        # Calculate statistics
        avg_lin_x_vel = sum(lin_x_velocities) / len(lin_x_velocities) if lin_x_velocities else 0
        avg_lin_y_vel = sum(lin_y_velocities) / len(lin_y_velocities) if lin_y_velocities else 0
        avg_ang_vel = sum(ang_velocities) / len(ang_velocities) if ang_velocities else 0
        
        # Only log if time interval has passed (throttling)
        current_time = time.time()
        if (current_time - self.last_diag_log_time) < LOG_THROTTLE_DIAG:
            return
            
        self.last_diag_log_time = current_time
        
        # Log detailed information
        if self.debug_level >= 1:
            # Get PID components
            p_x, i_x, d_x = self.pid_linear_x.get_components()
            p_y, i_y, d_y = self.pid_linear_y.get_components()
            p_a, i_a, d_a = self.pid_angular.get_components()
            
            diag_msg = (
                f"DIAGNOSTICS: "
                f"Avg Vel=[{avg_lin_x_vel:.2f}, {avg_lin_y_vel:.2f}, {avg_ang_vel:.2f}], "
                f"PID X=[{p_x:.2f}, {i_x:.2f}, {d_x:.2f}], "
                f"PID Y=[{p_y:.2f}, {i_y:.2f}, {d_y:.2f}], "
                f"PID A=[{p_a:.2f}, {i_a:.2f}, {d_a:.2f}], "
                f"Strategy={self.current_strategy}"
            )
            
            self.get_logger().info(diag_msg)
        
        # Publish PID diagnostic data
        self._publish_pid_diagnostics()
        
        # Publish performance metrics
        self._publish_performance_metrics()
    
    def _publish_pid_diagnostics(self):
        """Publish detailed PID diagnostics for analysis."""
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
    
    def _publish_performance_metrics(self):
        """Publish performance metrics for monitoring."""
        # Calculate CPU average
        cpu_avg = 0.0
        if self.performance_stats['cpu']:
            cpu_avg = sum(self.performance_stats['cpu']) / len(self.performance_stats['cpu'])
        
        # Calculate cycle time average
        cycle_time_avg = 0.0
        if self.performance_stats['control_cycles']:
            cycle_time_avg = sum(self.performance_stats['control_cycles']) / len(self.performance_stats['control_cycles'])
            cycle_time_avg *= 1000.0  # Convert to ms
        
        # Create performance message
        performance_msg = String()
        performance_msg.data = (
            f'{{"cpu": {cpu_avg:.1f}, '
            f'"cycle_time_ms": {cycle_time_avg:.2f}, '
            f'"strategy": "{self.current_strategy}", '
            f'"skips": {self.performance_stats["control_skips"]}, '
            f'"update_rate": {self.update_rate:.1f}}}'
        )
        
        # Publish
        self.performance_pub.publish(performance_msg)
    
    def prepare_shutdown(self):
        """Prepare for node shutdown with state transition logging."""
        previous_state = self.robot_state
        shutdown_start = time.time()
        
        log_structured('system', 'SHUTDOWN_INITIATED',
                    "Node shutdown sequence initiated",
                    {'from_state': previous_state,
                    'cycle_count': getattr(self, 'cycle_count', 0),
                    'uptime': shutdown_start - getattr(self, '_startup_time', shutdown_start)})
        
        self.get_logger().info("Preparing for shutdown")
        
        # Set shutdown flag
        self._shutting_down = True
        self.robot_state = "shutting_down"
        
        # Log the state transition
        log_state_transition(self, previous_state, "shutting_down", 
                        "Shutdown requested", 
                        {'cycle_count': getattr(self, 'cycle_count', 0)})
        
        # Immediately stop the robot
        try:
            self.stop_reason = "System shutdown"
            self.stop_robot()
            stop_success = True
            self.get_logger().info("Robot motion stopped during shutdown")
        except Exception as e:
            stop_success = False
            self.get_logger().error(f"Error stopping robot during shutdown: {str(e)}")
        
        # Log shutdown completion
        shutdown_duration = time.time() - shutdown_start
        log_structured('system', 'SHUTDOWN_COMPLETE',
                    "Node shutdown sequence completed",
                    {'duration': shutdown_duration,
                    'stop_success': stop_success})

# Main function
def main(args=None):
    """Main function to initialize and run the PID Controller node."""
    rclpy.init(args=args)
    node = ImprovedPIDControllerNode()
    
    # Welcome message
    print("=================================================")
    print("Improved PID Controller for Basketball Tracking Robot")
    print("=================================================")
    print("This node implements several enhancements:")
    print("1. Angular-first control strategy for diagonal movements")
    print("2. Enhanced integral term management")
    print("3. Fast strategy transitions for responsive tracking")
    print("4. Balanced error thresholds with hysteresis")
    print("5. Continuous motion tracking with trajectory prediction")
    print("")
    print("Press Ctrl+C to stop the program")
    print("=================================================")
    
    # Register shutdown handler for proper cleanup
    def signal_handler(sig, frame):
        print(f"\nSignal {sig} received, stopping robot...")
        # Prepare for shutdown
        node.prepare_shutdown()
        # Then proceed with ROS shutdown
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Improved PID Controller shutdown requested via Ctrl+C")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {str(e)}")
    finally:
        # Stop the robot before shutdown
        node.prepare_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()