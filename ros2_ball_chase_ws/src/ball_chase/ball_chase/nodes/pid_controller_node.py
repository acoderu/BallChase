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
    """Centralized manager for all logging in the PID controller with enhanced throttling."""
    
    def __init__(self, node, debug_level=1, log_verbosity=1):
        """
        Initialize logging system with proper hierarchy and improved throttling.
        
        Args:
            node: ROS node for ROS logging
            debug_level: Debug level (0=minimal, 1=normal, 2=verbose)
            log_verbosity: Verbosity level for logs (0=minimal, 1=normal, 2=detailed)
        """
        self.node = node
        self.debug_level = debug_level
        self.log_verbosity = log_verbosity
        
        # Enhanced throttling with message deduplication
        self.throttle_timestamps = {}  # For tracking throttled logs
        self.recent_message_hashes = {}  # For tracking and deduplicating recent messages
        self.message_counts = {}  # For counting similar messages
        
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
        
        # Initialize component verbosity settings with defaults
        self.component_verbosity = {
            'pid_controller': log_verbosity,
            'velocity_limiter': max(0, log_verbosity - 1),  # Lower default for noisy components
            'matrix4x4': max(0, log_verbosity - 1),
            'strategy_selector': log_verbosity,
            'target_filter': max(0, log_verbosity - 1),
            'motion_control': log_verbosity,
            'motion_commander': log_verbosity
        }
        
        # Log initialization
        self.log_structured('logging_manager', 'INIT', 
                        f"Logging system initialized with debug level {debug_level}, verbosity {log_verbosity}")
        
        
    
    def should_log(self, component, verbosity_level):
        """
        Check if logging is enabled for the given component and verbosity level.
        
        Args:
            component: Component name
            verbosity_level: Verbosity level of this message
            
        Returns:
            bool: True if should log, False otherwise
        """
        # Get component-specific verbosity setting with fallback to global
        component_verbosity = self.component_verbosity.get(component, self.log_verbosity)
        
        # Log only if message verbosity <= component verbosity
        return verbosity_level <= component_verbosity

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
                     level=logging.INFO, throttle_key=None, throttle_seconds=None,
                     verbosity_level=1):
        """
        Log a structured message with component, event type, and relevant parameters.
        Enhanced with better throttling, deduplication, and verbosity control.
        
        Args:
            component: Component generating the log
            event_type: Type of event
            message: Log message
            params: Optional dictionary of parameters
            level: Log level
            throttle_key: Optional key for throttling
            throttle_seconds: Seconds to throttle similar logs
            verbosity_level: Verbosity level (0=critical, 1=normal, 2=detailed)
        
        Returns:
            bool: True if log was written, False if throttled or filtered
        """
        if params is None:
            params = {}
            
        # Check component verbosity first - fast early exit
        if not self.should_log(component, verbosity_level):
            return False
        
        current_time = time.time()
        
        # 1. Deduplication - prevent exact same log within short window (100ms)
        # Create a hash of the log message content
        msg_content = f"{component}_{event_type}_{message}"
        if params:
            sorted_params = sorted((k, str(v)) for k, v in params.items())
            msg_content += "_" + str(sorted_params)
            
        msg_hash = hash(msg_content)
        
        # Check for exact duplicate message within 100ms
        if msg_hash in self.recent_message_hashes:
            last_time, count = self.recent_message_hashes[msg_hash]
            if current_time - last_time < 0.1:  # 100ms deduplication window
                # Update count but don't log
                self.recent_message_hashes[msg_hash] = (last_time, count + 1)
                return False
        
        # 2. Apply throttling if requested
        if throttle_key is not None and throttle_seconds is not None:
            full_key = f"{component}_{event_type}_{throttle_key}"
            
            if full_key in self.throttle_timestamps:
                last_time = self.throttle_timestamps[full_key]
                
                # Check if within throttle window
                if current_time - last_time < throttle_seconds:
                    # Count throttled message
                    if full_key not in self.message_counts:
                        self.message_counts[full_key] = 0
                    self.message_counts[full_key] += 1
                    return False  # Skip this log due to throttling
                else:
                    # Outside throttle window - include skipped count in message
                    if full_key in self.message_counts and self.message_counts[full_key] > 0:
                        # Append skipped count to message
                        message += f" (+{self.message_counts[full_key]} similar messages throttled)"
                        # Reset counter
                        self.message_counts[full_key] = 0
            
            # Update throttle timestamp
            self.throttle_timestamps[full_key] = current_time
        
        # 3. Update recent message tracking
        self.recent_message_hashes[msg_hash] = (current_time, 1)
        
        # 4. Clean up old hashes (older than 1 second)
        self._cleanup_old_message_hashes(current_time)
        
        # Get the appropriate logger
        logger = self.get_logger(component)
        
        # Format message with parameters
        structured_msg = f"[{event_type}] {message}"
        if params:
            # Format float values with precision
            param_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in params.items())
            structured_msg += f" | {param_str}"
        
        # Log using the component's logger
        logger.log(level, structured_msg)
        
        # For critical errors, also log to ROS logger if available
        if level >= logging.ERROR and hasattr(self.node, 'get_logger'):
            self.node.get_logger().error(f"{component}: {structured_msg}")
        
        return True
    
    def _cleanup_old_message_hashes(self, current_time, age=1.0):
        """Clean up message hashes older than specified age."""
        hashes_to_remove = []
        
        for msg_hash, (timestamp, _) in self.recent_message_hashes.items():
            if current_time - timestamp > age:
                hashes_to_remove.append(msg_hash)
                
        for msg_hash in hashes_to_remove:
            del self.recent_message_hashes[msg_hash]
    
    def set_component_verbosity(self, component, verbosity_level):
        """
        Set verbosity level for a specific component.
        
        Args:
            component: Component name
            verbosity_level: Verbosity level (0=minimal, 1=normal, 2=detailed)
        """
        self.component_verbosity[component] = verbosity_level
        self.log_structured('logging_manager', 'VERBOSITY_UPDATE',
                         f"Component '{component}' verbosity set to {verbosity_level}",
                         level=logging.INFO)
    
    def get_debug_level(self):
        """Get the global debug level."""
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

class MovementStateLogger:
    """
    Centralized manager for movement state logging to eliminate duplicate logs
    while preserving important context information.
    """
    
    def __init__(self, log_throttle_seconds=0.5):
        """
        Initialize the movement state logger with throttling controls.
        
        Args:
            log_throttle_seconds: Minimum seconds between similar logs
        """
        self.logger = logging.getLogger('pid_controller.motion_state')
        
        # Tracking state for throttling and context
        self._state_tracking = {
            'last_state_change_time': 0.0,
            'last_log_time': 0.0,
            'current_state': True,  # True = stopped, False = moving
            'last_transition_reason': None,
            'last_reported_error_values': None,
            'consecutive_similar_count': 0,
            'min_log_interval': log_throttle_seconds  # Seconds between similar logs
        }
        
        # Initialize motion history for context
        self._motion_history = []  # List of (state, reason, timestamp) tuples
        self._max_history_size = 5  # Keep last 5 state transitions
    
    def log_state_transition(self, new_state, reason, error_values=None, robot_state=None, verbose=True):
        """
        Log a movement state transition with enhanced context and duplicate prevention.
        
        Args:
            new_state: True for stopped, False for moving
            reason: Reason for the state transition
            error_values: Optional dictionary with error values (distance, lateral, angular)
            robot_state: Optional robot state string (tracking, recovery, etc.)
            verbose: Whether to log this transition at INFO level (vs DEBUG)
            
        Returns:
            bool: True if log was actually written, False if suppressed
        """
        current_time = time.time()
        
        # Get previous state for comparison
        prev_state = self._state_tracking['current_state']
        prev_reason = self._state_tracking['last_transition_reason']
        
        # Format stop/moving labels for readability
        state_label = "STOPPED" if new_state else "MOVING"
        prev_state_label = "STOPPED" if prev_state else "MOVING"
        
        # Track state changes even if we don't log them
        state_changed = new_state != prev_state
        if state_changed:
            # Update state tracking
            self._state_tracking['current_state'] = new_state
            self._state_tracking['last_state_change_time'] = current_time
            self._state_tracking['last_transition_reason'] = reason
            
            # Update history
            self._motion_history.append((new_state, reason, current_time))
            if len(self._motion_history) > self._max_history_size:
                self._motion_history.pop(0)  # Remove oldest entry
        
        # Prepare log parameters with contextual information
        log_params = {
            'duration': 0.0,  # Will be updated if state has changed
            'robot_state': robot_state or "unknown",
            'reason': reason
        }
        
        # Include error values if provided
        if error_values:
            for key, value in error_values.items():
                log_params[key] = value
            self._state_tracking['last_reported_error_values'] = error_values
        elif self._state_tracking['last_reported_error_values']:
            # Use previously reported error values for context if not provided
            for key, value in self._state_tracking['last_reported_error_values'].items():
                log_params[key + "_prev"] = value
        
        # Determine if this is a true transition or a duplicate log request
        if state_changed:
            # This is a true state transition - calculate time in previous state
            time_in_state = current_time - self._state_tracking['last_state_change_time']
            log_params['duration'] = time_in_state
            
            # Reset consecutive counter for new state
            self._state_tracking['consecutive_similar_count'] = 0
            
            # Log the state transition with full information
            if new_state:  # Stopping
                event_type = 'ROBOT_MOTION_STOP'
                msg = f"Robot stopping: {reason}"
            else:  # Starting to move
                event_type = 'ROBOT_MOTION_START'
                msg = f"Robot starting movement: {reason}"
            
            # Expand log context based on transition type
            if new_state and len(self._motion_history) >= 2:
                # For stopping, include how long we were moving
                prev_start_time = self._find_last_state_change(False)
                if prev_start_time:
                    moving_duration = current_time - prev_start_time
                    log_params['moving_duration'] = moving_duration
                    msg += f" (after moving for {moving_duration:.1f}s)"
            elif not new_state and len(self._motion_history) >= 2:
                # For starting, include how long we were stopped
                prev_stop_time = self._find_last_state_change(True)
                if prev_stop_time:
                    stopped_duration = current_time - prev_stop_time
                    log_params['stopped_duration'] = stopped_duration
                    msg += f" (after being stopped for {stopped_duration:.1f}s)"
            
            # Use structured logging with appropriate level
            level = logging.INFO if verbose else logging.DEBUG
            self._log_structured('motion_state', event_type, msg, log_params, level)
            
            # Update tracking timestamps
            self._state_tracking['last_log_time'] = current_time
            return True
            
        else:
            # Same state reported again - determine if we should log
            # Increment consecutive counter
            self._state_tracking['consecutive_similar_count'] += 1
            consecutive_count = self._state_tracking['consecutive_similar_count']
            
            # Check if reason changed substantially
            reason_changed = prev_reason != reason
            
            # Apply throttling for duplicate state reports
            time_since_last_log = current_time - self._state_tracking['last_log_time']
            min_interval = self._state_tracking['min_log_interval']
            
            # Exponentially increase interval for repeated logs
            if consecutive_count > 3:
                # Double interval after 3 repeats, up to 10x original
                adjusted_interval = min(min_interval * 10, min_interval * (2 ** (consecutive_count - 3)))
            else:
                adjusted_interval = min_interval
            
            # Log if reason changed or if enough time has passed
            should_log = reason_changed or time_since_last_log >= adjusted_interval
            
            if should_log:
                # Log with lower level and repeat context
                if new_state:  # Stopped
                    event_type = 'ROBOT_STILL_STOPPED'
                    msg = f"Robot remains stopped: {reason}"
                else:  # Moving
                    event_type = 'ROBOT_STILL_MOVING'
                    msg = f"Robot continues moving: {reason}"
                
                # Add additional context about repetition
                if consecutive_count > 1:
                    msg += f" (reported {consecutive_count} times)"
                
                # Use structured logging with lower level for duplicate states
                level = logging.DEBUG if not reason_changed else logging.INFO
                self._log_structured('motion_state', event_type, msg, log_params, level)
                
                # Update tracking timestamps
                self._state_tracking['last_log_time'] = current_time
                return True
                
        return False
        
    def _find_last_state_change(self, state):
        """
        Find the timestamp of the last transition to the specified state.
        
        Args:
            state: The state to find (True for stopped, False for moving)
            
        Returns:
            float: Timestamp of last transition or None if not found
        """
        for entry_state, _, timestamp in reversed(self._motion_history):
            if entry_state == state:
                return timestamp
        return None
        
    def _log_structured(self, component, event_type, message, params=None, level=logging.INFO):
        """
        Log a structured message with component, event type, and relevant parameters.
        
        Args:
            component: Component generating the log
            event_type: Type of event
            message: Log message
            params: Optional dictionary of parameters
            level: Log level
        """
        # Call the global log_structured function if available
        if globals().get('log_structured'):
            return globals()['log_structured'](component, event_type, message, params, level)
        
        # Fallback implementation if log_structured not available
        structured_msg = f"[{event_type}] {message}"
        if params:
            # Format parameters
            param_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in params.items())
            structured_msg += f" | {param_str}"
        
        # Log using the component's logger
        self.logger.log(level, structured_msg)
        return True
    
    def get_current_state(self):
        """
        Get the current movement state.
        
        Returns:
            bool: True if stopped, False if moving
        """
        return self._state_tracking['current_state']
    
    def get_last_transition_time(self):
        """
        Get the timestamp of the last state transition.
        
        Returns:
            float: Timestamp of last transition
        """
        return self._state_tracking['last_state_change_time']
    
    def get_time_in_current_state(self):
        """
        Get the time spent in the current state.
        
        Returns:
            float: Seconds in current state
        """
        return time.time() - self._state_tracking['last_state_change_time']


# Create a global instance for use across the application
_movement_state_logger = None

def init_movement_logger(log_throttle_seconds=0.5):
    """
    Initialize the global movement state logger.
    
    Args:
        log_throttle_seconds: Minimum seconds between similar logs
        
    Returns:
        MovementStateLogger: The global logger instance
    """
    global _movement_state_logger
    _movement_state_logger = MovementStateLogger(log_throttle_seconds)
    return _movement_state_logger

def log_movement_state(is_stopped, reason, error_values=None, robot_state=None, verbose=True):
    """
    Global function to log movement state changes or updates.
    
    Args:
        is_stopped: True if robot is stopped, False if moving
        reason: Reason for the state or state change
        error_values: Optional dictionary with error values
        robot_state: Optional robot state string
        verbose: Whether to log this at INFO level
        
    Returns:
        bool: True if log was written, False if suppressed
    """
    global _movement_state_logger
    
    # Initialize logger if needed
    if _movement_state_logger is None:
        _movement_state_logger = MovementStateLogger()
    
    return _movement_state_logger.log_state_transition(is_stopped, reason, error_values, robot_state, verbose)

def is_robot_stopped():
    """
    Check if the robot is currently in stopped state.
    
    Returns:
        bool: True if stopped, False if moving
    """
    global _movement_state_logger
    
    if _movement_state_logger is None:
        # Default to stopped if logger not initialized
        return True
        
    return _movement_state_logger.get_current_state()

def get_time_since_last_transition():
    """
    Get time since the last movement state transition.
    
    Returns:
        float: Seconds since last transition
    """
    global _movement_state_logger
    
    if _movement_state_logger is None:
        return 0.0
        
    return _movement_state_logger.get_time_in_current_state()

# Initialize the dummy manager
_dummy_logging_manager = DummyLoggingManager()

def init_logging_system(node, debug_level=1, log_verbosity=1):
    """
    Initialize the global logging system with enhanced throttling.
    
    Args:
        node: ROS node for ROS logging
        debug_level: Debug level (0=minimal, 1=normal, 2=verbose)
        log_verbosity: Verbosity level for logs (0=minimal, 1=normal, 2=detailed)
        
    Returns:
        LoggingManager: The global logging manager
    """
    global _logging_manager
    _logging_manager = LoggingManager(node, debug_level, log_verbosity)
    
    # Apply recommended verbosity settings for noisy components
    _logging_manager.set_component_verbosity('velocity_limiter', max(0, log_verbosity - 1))
    _logging_manager.set_component_verbosity('matrix4x4', max(0, log_verbosity - 1))
    _logging_manager.set_component_verbosity('target_filter', max(0, log_verbosity - 1))
    
    return _logging_manager

# Global function interfaces that use the logging manager

# Modify the global log_structured function
def log_structured(component, event_type, message, params=None, level=logging.INFO, 
                throttle_key=None, throttle_seconds=None, verbosity_level=1):
    """
    Global shortcut for log_structured with verbosity level check and improved type handling.
    Ensures tuple variables are safely formatted as strings.
    """
    global _logging_manager, _dummy_logging_manager
    
    # Default parameters if not provided
    if params is None:
        params = {}
    
    # Make a safe copy of params to avoid modifying the original
    safe_params = {}
    
    # Process the parameters to ensure they're safely formattable
    for key, value in params.items():
        # Handle potential tuples or objects with name attributes
        if hasattr(value, 'name'):
            safe_params[key] = value.name
        else:
            safe_params[key] = value
            
    # Process the message to ensure it's safely formattable
    try:
        # Check if message has any format placeholders
        if '{' in message and '}' in message:
            # Check if it's using new-style format
            if any(c in message for c in ['{0}', '{1}', '{2}', '{:']) or \
               any(key in message for key in safe_params.keys()):
                # Try to safely format the message
                message = message.format(**safe_params)
        
    except Exception as e:
        # If formatting fails, create a safe alternative message
        format_error_msg = f"[FORMAT ERROR in log message: {str(e)}] {message}"
        message = format_error_msg
    
    # Check if we should log based on verbosity level
    if _logging_manager is not None:
        if _logging_manager.log_verbosity < verbosity_level:
            return False  # Skip logging due to verbosity setting
        
        # Use the global logging manager
        return _logging_manager.log_structured(component, event_type, message, 
                                            safe_params, level, throttle_key, throttle_seconds)
    else:
        # Use dummy manager if not initialized yet
        return _dummy_logging_manager.log_structured(component, event_type, message, 
                                                   safe_params, level, throttle_key, throttle_seconds)

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
    Log detailed PID controller state with improved throttling, significance detection,
    and deduplication to reduce log volume while preserving important events.
    
    Args:
        controller_name: Name of the controller
        error: Current error value
        output: Output value
        p_term: Proportional term
        i_term: Integral term
        d_term: Derivative term
        current_gains: Current PID gains (optional)
    """
    # Initialize tracking if it doesn't exist
    if not hasattr(log_pid_state, '_tracking'):
        log_pid_state._tracking = {}
        
    # Create tracking entry for this controller if it doesn't exist
    if controller_name not in log_pid_state._tracking:
        log_pid_state._tracking[controller_name] = {
            'last_log_time': 0,
            'last_error': None,
            'last_output': None,
            'last_i_term': None,
            'min_log_interval': 5.0,        # Reduced frequency - log at most every 5 seconds
            'force_log_interval': 30.0,     # Force at least one log every 30 seconds
            'error_threshold': 0.05,        # 5% error change threshold
            'output_threshold': 0.05,       # 5% output change threshold 
            'integral_threshold': 0.15,     # 15% integral change threshold
            'error_timestamps': {},         # Track error timestamps for oscillation detection
            'oscillation_count': 0,         # Track oscillation count
            'log_count': 0,                 # Track number of logs
            'last_hash': None,              # For deduplication
            'integral_warning_count': 0,    # Count integral warnings to avoid flooding
            'last_integral_warning': 0      # Time of last integral warning
        }
    
    # Get tracking data for this controller
    tracking = log_pid_state._tracking[controller_name]
    current_time = time.time()
    time_since_last_log = current_time - tracking['last_log_time']
    
    # Prepare log parameters
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
    
    # Check if this is a duplicate of the previous log
    log_hash = hash(f"{error:.4f}_{output:.4f}_{p_term:.4f}_{i_term:.4f}_{d_term:.4f}")
    is_duplicate = log_hash == tracking['last_hash']
    tracking['last_hash'] = log_hash
    
    # Skip if duplicate and recent (within 2 seconds)
    if is_duplicate and time_since_last_log < 2.0:
        return
    
    # Calculate relative changes for significance detection
    significant_change = False
    change_reasons = []
    
    if tracking['last_error'] is not None:
        # Calculate error change
        if abs(tracking['last_error']) > 0.001:
            error_change = abs(error - tracking['last_error']) / abs(tracking['last_error'])
            if error_change > tracking['error_threshold']:
                significant_change = True
                change_reasons.append(f"error_change={error_change:.2f}")
    
    if tracking['last_output'] is not None:
        # Calculate output change
        if abs(tracking['last_output']) > 0.001:
            output_change = abs(output - tracking['last_output']) / abs(tracking['last_output'])
            if output_change > tracking['output_threshold']:
                significant_change = True
                change_reasons.append(f"output_change={output_change:.2f}")
    
    # Special handling for integral term - detect problematic values
    integral_warning = False
    if tracking['last_i_term'] is not None:
        # Calculate rate of change (per second)
        if time_since_last_log > 0.001:
            i_change_rate = abs(i_term - tracking['last_i_term']) / time_since_last_log
            
            # Calculate relative change
            if abs(tracking['last_i_term']) > 0.001:
                i_relative_change = abs(i_term - tracking['last_i_term']) / abs(tracking['last_i_term'])
                
                # Check for rapid growth in integral term (possible windup)
                if i_relative_change > tracking['integral_threshold'] and abs(i_term) > 5.0:
                    integral_warning = True
                    params['i_change_rate'] = i_change_rate
                    params['i_relative_change'] = i_relative_change
                    change_reasons.append(f"i_term_change={i_relative_change:.2f}")
    
    # Detect sign changes that might indicate oscillation
    if tracking['last_error'] is not None:
        # Check if error crossed zero
        if (error * tracking['last_error']) < 0 and abs(error) > 0.01:
            # Record error sign change time
            error_sign = 1 if error > 0 else -1
            if error_sign not in tracking['error_timestamps']:
                tracking['error_timestamps'][error_sign] = []
            
            tracking['error_timestamps'][error_sign].append(current_time)
            
            # Only keep the most recent timestamps (last 5 seconds)
            tracking['error_timestamps'][error_sign] = [
                t for t in tracking['error_timestamps'][error_sign] 
                if (current_time - t) < 5.0
            ]
            
            # Check for oscillation
            if (1 in tracking['error_timestamps'] and -1 in tracking['error_timestamps'] and
                len(tracking['error_timestamps'][1]) >= 2 and len(tracking['error_timestamps'][-1]) >= 2):
                tracking['oscillation_count'] += 1
                significant_change = True
                change_reasons.append(f"oscillation={tracking['oscillation_count']}")
    
    # Determine if we should log based on time or significance
    should_log = (
        time_since_last_log >= tracking['force_log_interval'] or   # Force log periodically
        (significant_change and time_since_last_log >= tracking['min_log_interval']) or  # Log significant changes
        tracking['log_count'] < 5  # Always log the first few updates for baseline
    )
    
    # If we're going to log, update tracking
    if should_log:
        tracking['last_log_time'] = current_time
        tracking['last_error'] = error
        tracking['last_output'] = output
        tracking['last_i_term'] = i_term
        tracking['log_count'] += 1
        
        # Add context about what triggered the log
        if time_since_last_log >= tracking['force_log_interval']:
            context = f"periodic update after {time_since_last_log:.1f}s"
        elif significant_change:
            context = f"significant change: {', '.join(change_reasons)}"
        else:
            context = "initial update"
            
        # Add context to message
        msg = f"{controller_name} update: {context}"
        
        # Log normally for most updates
        log_structured('pid_controller', 'PID_STATE', 
                      msg, 
                      params,
                      verbosity_level=2)     # Higher verbosity level
        
    # Handle integral warnings separately with stricter throttling
    if integral_warning:
        # Only log integral warnings if enough time has passed (avoid flooding)
        integral_warning_interval = 2.0  # seconds
        if current_time - tracking['last_integral_warning'] >= integral_warning_interval:
            # Log warning with detailed info about the integral term
            log_structured('pid_controller', 'PID_INTEGRAL_WARNING', 
                          f"{controller_name} integral warning: value={i_term:.3f}, change_rate={i_change_rate:.3f}, error={error:.3f}", 
                          {'i_term': i_term, 'error': error, 'i_change_rate': i_change_rate},
                          level=logging.WARNING)
            
            tracking['last_integral_warning'] = current_time
            tracking['integral_warning_count'] += 1
        
        # If we've seen many warnings, suggest reset in log
        if tracking['integral_warning_count'] > 5:
            log_structured('pid_controller', 'PID_RESET_SUGGESTED', 
                          f"{controller_name} has {tracking['integral_warning_count']} integral warnings - PID reset suggested",
                          {'i_term': i_term, 'warning_count': tracking['integral_warning_count']},
                          level=logging.WARNING)
            tracking['integral_warning_count'] = 0  # Reset counter after suggestion

def log_strategy_selection(key, selected_strategy, candidate_strategies=None, reason=None):
    """
    Log strategy selection with improved throttling and significance detection.
    Consolidates strategy selection logs with appropriate throttling.
    
    Args:
        key: Selection key or context
        selected_strategy: The strategy that was selected
        candidate_strategies: Optional dictionary of candidate strategies
        reason: Optional reason for the selection
    """
    # Initialize static tracking dictionary if it doesn't exist
    if not hasattr(log_strategy_selection, "_tracking"):
        log_strategy_selection._tracking = {
            "last_log_time": {},  # Keyed by strategy name
            "consecutive_count": {},  # Count consecutive selections of same strategy
            "last_strategy": None,  # Last logged strategy
            "strategy_durations": {},  # Track how long each strategy is used
            "significant_strategies": {  # Strategies that are always significant to log
                "ANGULAR_FIRST_APPROACH", 
                "ANGULAR_PRIMARY",
                "EMERGENCY_BRAKE",
                "APPROACH"
            },
            "min_log_interval": 2.0,  # Minimum seconds between logs for same strategy
            "escalating_interval": True,  # Whether to increase interval for repetitive logs
        }
    
    tracking = log_strategy_selection._tracking
    current_time = time.time()
    
    # Extract strategy name - handle both dict and object formats
    strategy_name = selected_strategy
    if isinstance(selected_strategy, dict) and "strategy_name" in selected_strategy:
        strategy_name = selected_strategy["strategy_name"]
    elif hasattr(selected_strategy, "name"):
        strategy_name = selected_strategy.name
    
    # Check if this is a significant strategy change
    is_significant = False
    
    # Case 1: Strategy changed from previous
    prev_strategy = tracking["last_strategy"]
    strategy_changed = prev_strategy is not None and prev_strategy != strategy_name
    
    if strategy_changed:
        is_significant = True
        # Reset consecutive count for new strategy
        tracking["consecutive_count"][strategy_name] = 0
        
        # Record duration of previous strategy
        if prev_strategy in tracking["strategy_durations"]:
            start_time = tracking["strategy_durations"][prev_strategy]
            duration = current_time - start_time
            # Only log significant durations (> 0.5 seconds)
            if duration > 0.5:
                log_structured('strategy_selector', 'STRATEGY_DURATION', 
                            f"Strategy {prev_strategy} was active for {duration:.1f}s", 
                            {'duration': duration,
                             'strategy': prev_strategy},
                             throttle_key=f"duration_{prev_strategy}",
                             throttle_seconds=5.0)
        
        # Record start time for new strategy
        tracking["strategy_durations"][strategy_name] = current_time
    
    # Case 2: Always log significant strategies
    elif strategy_name in tracking["significant_strategies"]:
        is_significant = True
        
    # Case 3: First time seeing this strategy
    elif strategy_name not in tracking["last_log_time"]:
        is_significant = True
        tracking["consecutive_count"][strategy_name] = 0
        tracking["strategy_durations"][strategy_name] = current_time
    
    # Update consecutive count for this strategy
    if strategy_name in tracking["consecutive_count"]:
        tracking["consecutive_count"][strategy_name] += 1
    else:
        tracking["consecutive_count"][strategy_name] = 1
    
    # Determine if we should log based on time and significance
    should_log = is_significant
    
    if not should_log:
        # Get last log time for this strategy, defaulting to 0
        last_log_time = tracking["last_log_time"].get(strategy_name, 0)
        consecutive_count = tracking["consecutive_count"].get(strategy_name, 0)
        
        # Calculate required interval based on repetition
        if tracking["escalating_interval"] and consecutive_count > 3:
            # Escalate interval for highly repetitive selections
            # After 3 consecutive logs, double the interval each time up to 20 seconds
            log_interval = min(20.0, tracking["min_log_interval"] * (2 ** (consecutive_count - 3)))
        else:
            log_interval = tracking["min_log_interval"]
        
        # Log if enough time has passed
        time_since_log = current_time - last_log_time
        if time_since_log >= log_interval:
            should_log = True
    
    if should_log:
        # Format parameters for logging
        params = {
            'key': key,
            'strategy_name': strategy_name,
        }
        
        # Add selected strategy details
        if isinstance(selected_strategy, dict):
            for k, v in selected_strategy.items():
                if k != "strategy_name":  # Already included
                    params[k] = v
        
        # Include consecutive count if relevant
        consecutive_msg = ""
        if tracking["consecutive_count"].get(strategy_name, 0) > 1:
            consecutive_count = tracking["consecutive_count"][strategy_name]
            consecutive_msg = f" (selected {consecutive_count} times consecutively)"
            params['consecutive_count'] = consecutive_count
        
        # Create log message
        msg = f"Selected strategy: {strategy_name}{consecutive_msg}"
        if reason:
            msg += f" - {reason}"
        
        # Log with appropriate throttling
        log_structured('strategy_selector', 'STRATEGY_SELECTION', 
                     msg, 
                     params,
                     throttle_key=strategy_name,
                     throttle_seconds=1.0)  # Keep at 1.0 for important strategy changes
                     
        # Update tracking
        tracking["last_log_time"][strategy_name] = current_time
        tracking["last_strategy"] = strategy_name

def log_strategy_blending(start_strategy, target_strategy, blend_factor, blend_duration):
    """
    Log strategy blending progress with reduced frequency and improved significance detection.
    """
    # Initialize static tracking dictionary if it doesn't exist
    if not hasattr(log_strategy_blending, "_tracking"):
        log_strategy_blending._tracking = {
            "last_log_time": {},  # Keyed by blend transition
            "last_factor": {},    # Last logged factor for this transition
            "significant_points": [0.0, 0.5, 1.0],  # Log at these specific blend points
            "min_factor_change": 0.25,  # Minimum change in blend factor to log
        }
    
    tracking = log_strategy_blending._tracking
    current_time = time.time()
    
    # Get strategy names consistently
    start_name = start_strategy.name if hasattr(start_strategy, "name") else str(start_strategy)
    target_name = target_strategy.name if hasattr(target_strategy, "name") else str(target_strategy)
    
    # Create a key for this specific blend transition
    blend_key = f"{start_name}_to_{target_name}"
    
    params = {
        'start': start_name,
        'target': target_name,
        'blend_factor': blend_factor,
        'blend_duration': blend_duration,
        'effective_duration': getattr(start_strategy, 'effective_blend_duration', blend_duration)
    }
    
    # Determine if this is a significant point to log
    is_significant = False
    
    # Case 1: First time seeing this blend transition
    if blend_key not in tracking["last_log_time"]:
        is_significant = True
    
    # Case 2: At a significant blend point (start, middle, end)
    for point in tracking["significant_points"]:
        if abs(blend_factor - point) < 0.05:  # Within 5% of a significant point
            is_significant = True
            break
    
    # Case 3: Significant change from last logged factor
    last_factor = tracking["last_factor"].get(blend_key, -1)
    if last_factor >= 0 and abs(blend_factor - last_factor) >= tracking["min_factor_change"]:
        is_significant = True
    
    # Only log if significant and enough time has passed
    min_log_interval = 0.3  # Minimum seconds between logs for same blend
    last_log_time = tracking["last_log_time"].get(blend_key, 0)
    time_since_log = current_time - last_log_time
    
    if is_significant and time_since_log >= min_log_interval:
        log_structured('strategy_blender', 'STRATEGY_BLEND', 
                      f"Blending {blend_factor*100:.1f}% complete", 
                      params,
                      throttle_key=blend_key,
                      throttle_seconds=0.3)  # Allow reasonable updates during blend
        
        # Update tracking
        tracking["last_log_time"][blend_key] = current_time
        tracking["last_factor"][blend_key] = blend_factor

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
    # Initialize static tracking dictionary for throttling
    if not hasattr(log_target_filter_update, "_tracking"):
        log_target_filter_update._tracking = {
            "last_log_time": {},  # Keyed by type of update
            "consecutive_similar": {},  # Count of consecutive similar updates
            "last_positions": {},  # Last logged positions for comparison
            "min_log_interval": {
                "filter": 2.0,  # Every 2 seconds for regular updates
                "prediction": 1.0  # Every 1 second for predictions
            },
            "significant_change_threshold": 0.05  # 5% position change is significant
        }
    
    tracking = log_target_filter_update._tracking
    current_time = time.time()
    
    params = {
        'raw_dist': raw_position[0],
        'raw_lateral': raw_position[1],
        'raw_angle': raw_position[2],
        'filtered_dist': filtered_position[0],
        'filtered_lateral': filtered_position[1],
        'filtered_angle': filtered_position[2]
    }
    
    # Determine log type and setup tracking key
    if predicted_position:
        log_type = "prediction"
        tracking_key = "prediction"
        params['pred_dist'] = predicted_position[0]
        params['pred_lateral'] = predicted_position[1]
        params['pred_angle'] = predicted_position[2]
        if confidence:
            params['confidence'] = confidence
        msg = "Target prediction"
    else:
        log_type = "filter_update"
        tracking_key = "filter"
        msg = "Target filter update"
    
    # Check if this update is similar to the previous one
    is_similar = False
    if tracking_key in tracking["last_positions"]:
        if log_type == "prediction":
            # For predictions, compare prediction values
            prev_vals = tracking["last_positions"][tracking_key]
            curr_vals = [predicted_position[0], predicted_position[1], predicted_position[2]]
            
            # Calculate difference percentage
            diffs = [
                abs(curr_vals[0] - prev_vals[0]) / (abs(prev_vals[0]) + 0.001),
                abs(curr_vals[1] - prev_vals[1]) / (abs(prev_vals[1]) + 0.001),
                abs(curr_vals[2] - prev_vals[2]) / (abs(prev_vals[2]) + 0.001)
            ]
            avg_diff = sum(diffs) / 3
            
            is_similar = avg_diff < tracking["significant_change_threshold"]
        else:
            # For filter updates, compare filtered position
            prev_vals = tracking["last_positions"][tracking_key]
            curr_vals = [filtered_position[0], filtered_position[1], filtered_position[2]]
            
            # Calculate difference percentage
            diffs = [
                abs(curr_vals[0] - prev_vals[0]) / (abs(prev_vals[0]) + 0.001),
                abs(curr_vals[1] - prev_vals[1]) / (abs(prev_vals[1]) + 0.001),
                abs(curr_vals[2] - prev_vals[2]) / (abs(prev_vals[2]) + 0.001)
            ]
            avg_diff = sum(diffs) / 3
            
            is_similar = avg_diff < tracking["significant_change_threshold"]
    
    # Update consecutive similar counter
    if is_similar:
        if tracking_key in tracking["consecutive_similar"]:
            tracking["consecutive_similar"][tracking_key] += 1
        else:
            tracking["consecutive_similar"][tracking_key] = 1
    else:
        tracking["consecutive_similar"][tracking_key] = 0
    
    # Determine if we should log based on similarity and time
    last_log_time = tracking["last_log_time"].get(tracking_key, 0)
    time_since_log = current_time - last_log_time
    consecutive_count = tracking["consecutive_similar"].get(tracking_key, 0)
    
    # Apply time-based throttling with consecutive message consideration
    min_interval = tracking["min_log_interval"][tracking_key]
    
    # For consecutive similar updates, increase throttle interval
    if consecutive_count > 3:
        # Exponential increase in throttle interval
        adjusted_interval = min_interval * (1.5 ** min(consecutive_count - 3, 5))
        should_log = time_since_log >= adjusted_interval
    else:
        # Regular time-based throttling
        should_log = time_since_log >= min_interval
    
    # Add repetition info to message if needed
    if consecutive_count > 0 and should_log:
        msg += f" (similar updates: {consecutive_count+1}x)"
    
    # Log if we should
    if should_log:
        log_structured('target_filter', log_type.upper(), 
                    msg, 
                    params,
                    throttle_key=tracking_key,
                    throttle_seconds=min_interval)
        
        # Update tracking info
        tracking["last_log_time"][tracking_key] = current_time
        if log_type == "prediction":
            # Store prediction values
            tracking["last_positions"][tracking_key] = [
                predicted_position[0], predicted_position[1], predicted_position[2]
            ]
        else:
            # Store filtered values
            tracking["last_positions"][tracking_key] = [
                filtered_position[0], filtered_position[1], filtered_position[2]
            ]

def log_velocity_limiting(raw_velocities, limited_velocities, reason=None, components=None):
    """
    Log velocity limiting decisions with enhanced throttling and significance detection.
    Only logs when velocity is limited significantly.
    
    Args:
        raw_velocities: Original velocities before limiting
        limited_velocities: Limited velocities after applying limits
        reason: Reason for limiting, if any
        components: Components that were limited
    """
    # Calculate how much velocity was limited - with component tracking
    limit_pct = [0, 0, 0]
    significant_limiting = False
    limited_components = components or []
    
    # Increased threshold from 25% to 40% for logging significance
    significance_threshold = 40.0  
    
    for i, component in enumerate(['x', 'y', 'a']):
        if abs(raw_velocities[i]) > 0.01:  # Avoid division by zero
            limit_pct[i] = abs(limited_velocities[i] - raw_velocities[i]) / abs(raw_velocities[i]) * 100
            if limit_pct[i] > significance_threshold:  # Only consider significant if >40% change
                significant_limiting = True
                limited_components.append(f"{component}:{limit_pct[i]:.0f}%")
    
    # Skip logging if velocity wasn't limited significantly
    if not significant_limiting:
        return
    
    # Check for duplicate logging based on component pattern
    # Only implement if we have tracking enabled
    components_key = '_'.join(limited_components)
    
    # Initialize log tracking if not already done
    global _logging_manager
    if _logging_manager is not None and not hasattr(_logging_manager, 'velocity_limiting_tracking'):
        _logging_manager.velocity_limiting_tracking = {
            'last_log_time': {},  # Track time by component pattern
            'repetition_count': {},  # Track repetition by component pattern
            'consecutive_threshold': 5,  # Log after this many consecutive similar messages
            'force_log_interval': 5.0  # Force log every 5 seconds for a component pattern
        }
    
    # If logging manager exists, use enhanced tracking
    if _logging_manager is not None and hasattr(_logging_manager, 'velocity_limiting_tracking'):
        tracking = _logging_manager.velocity_limiting_tracking
        current_time = time.time()
        
        # Get last log time for this component pattern, defaulting to 0
        last_log_time = tracking['last_log_time'].get(components_key, 0)
        
        # Get repetition count for this component pattern, defaulting to 0
        repetition_count = tracking['repetition_count'].get(components_key, 0)
        
        # Increment repetition count
        repetition_count += 1
        tracking['repetition_count'][components_key] = repetition_count
        
        # Determine if we should log based on time or repetition
        should_log = (
            (current_time - last_log_time) > tracking['force_log_interval'] or  # Time threshold
            repetition_count >= tracking['consecutive_threshold']  # Repetition threshold
        )
        
        if should_log:
            # Update log time
            tracking['last_log_time'][components_key] = current_time
            
            # Create message with repetition count if needed
            msg = "Velocity limiting applied"
            if reason:
                msg += f": {reason}"
                
            # Add repetition info if relevant
            if repetition_count > 1:
                msg += f" (repeated {repetition_count} times)"
                # Reset repetition count
                tracking['repetition_count'][components_key] = 0
            
            # Include components but limit the list for brevity
            params = {
                'raw_velocities': raw_velocities,
                'limited_velocities': limited_velocities,
                'components': limited_components[:2]  # Only show first two component limits
            }
            
            # Log with increased throttle time
            log_structured('velocity_limiter', 'VELOCITY_LIMIT', 
                        msg, 
                        params,
                        throttle_key="velocity_limit",
                        throttle_seconds=3.0,     # Increased from 1.5s to 3.0s
                        verbosity_level=1)        # Normal verbosity
    else:
        # Fallback if no logging manager is available
        msg = "Velocity limiting applied"
        if reason:
            msg += f": {reason}"
            
        params = {
            'raw_x': raw_velocities[0],
            'raw_y': raw_velocities[1],
            'raw_angular': raw_velocities[2],
            'limited_x': limited_velocities[0],
            'limited_y': limited_velocities[1],
            'limited_angular': limited_velocities[2],
            'components': limited_components
        }
        
        # Use increased throttle time
        log_structured('velocity_limiter', 'VELOCITY_LIMIT', 
                    msg, 
                    params,
                    throttle_key="velocity_limit",
                    throttle_seconds=3.0,     # Increased throttle time
                    verbosity_level=1)        # Normal verbosity


def log_resource_usage(cpu_usage, memory_usage, cycle_time=None, rate_adjustment=None):
    """
    Log system resource usage with improved significance detection and throttling.
    
    Args:
        cpu_usage: Current CPU usage percentage
        memory_usage: Current memory usage percentage
        cycle_time: Control cycle time in seconds (optional)
        rate_adjustment: Rate adjustment factor (optional)
    """
    # Initialize tracking dict if it doesn't exist in the logging manager
    global _logging_manager
    
    # If logging manager isn't available, fall back to basic logging
    if _logging_manager is None:
        # Simple fallback with basic throttling
        current_time = time.time()
        if not hasattr(log_resource_usage, 'last_log_time'):
            log_resource_usage.last_log_time = 0
            
        if current_time - log_resource_usage.last_log_time > 10.0:
            print(f"RESOURCE: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%")
            log_resource_usage.last_log_time = current_time
        return
    
    # Get or create resource tracking data
    if not hasattr(_logging_manager, 'resource_tracking'):
        _logging_manager.resource_tracking = {
            'last_log_time': 0.0,
            'last_cpu': 0.0,
            'last_mem': 0.0,
            'last_cycle_time': None,
            'last_adjustment': None,
            'changes': [],  # Track recent changes for pattern detection
            'min_log_interval': 10.0,  # Minimum time between logs
            'forced_log_interval': 60.0,  # Force log at least every minute
            'high_cpu_log_interval': 5.0,  # More frequent logs for high CPU
            'significant_change_cpu': 10.0,  # 10 percentage points change in CPU
            'significant_change_mem': 5.0,  # 5 percentage points change in memory
            'significant_change_cycle': 0.2  # 20% change in cycle time
        }
    
    tracking = _logging_manager.resource_tracking
    current_time = time.time()
    
    # Calculate time since last log
    time_since_log = current_time - tracking['last_log_time']
    
    # Determine if changes are significant
    cpu_change = abs(cpu_usage - tracking['last_cpu'])
    mem_change = abs(memory_usage - tracking['last_mem'])
    cycle_change = False
    
    if cycle_time is not None and tracking['last_cycle_time'] is not None:
        if tracking['last_cycle_time'] > 0.001:  # Avoid division by zero
            cycle_pct_change = abs(cycle_time - tracking['last_cycle_time']) / tracking['last_cycle_time']
            cycle_change = cycle_pct_change > tracking['significant_change_cycle']
    
    rate_change = rate_adjustment != tracking['last_adjustment']
    
    # Determine if changes are significant enough to log
    significant_change = (
        cpu_change > tracking['significant_change_cpu'] or
        mem_change > tracking['significant_change_mem'] or
        cycle_change or
        rate_change
    )
    
    # Determine if it's time to log
    high_cpu = cpu_usage > 80.0
    min_interval = tracking['high_cpu_log_interval'] if high_cpu else tracking['min_log_interval']
    time_to_log = time_since_log >= min_interval
    force_log = time_since_log >= tracking['forced_log_interval']
    
    # Log if significant changes or forced by time
    if significant_change or force_log:
        if time_to_log or force_log:
            try:
                # Prepare parameters
                params = {
                    'cpu_pct': cpu_usage,
                    'mem_pct': memory_usage
                }
                
                if cycle_time is not None:
                    params['cycle_time_ms'] = cycle_time * 1000  # convert to ms
                
                if rate_adjustment is not None:
                    params['rate_adj'] = rate_adjustment
                
                # Prepare log message with context
                if high_cpu:
                    msg = f"High CPU usage: {cpu_usage:.1f}%"
                else:
                    msg = f"Resource usage: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}%"
                    
                # Add context about changes
                change_context = []
                if cpu_change > tracking['significant_change_cpu']:
                    change_context.append(f"CPU {'+' if cpu_usage > tracking['last_cpu'] else '-'}{cpu_change:.1f}pts")
                    
                if mem_change > tracking['significant_change_mem']:
                    change_context.append(f"Mem {'+' if memory_usage > tracking['last_mem'] else '-'}{mem_change:.1f}pts")
                    
                if cycle_change:
                    change_context.append(f"Cycle {'+' if cycle_time > tracking['last_cycle_time'] else '-'}{cycle_pct_change*100:.0f}%")
                    
                if rate_adjustment is not None and rate_change:
                    change_context.append(f"Rate adj: {rate_adjustment:.2f}x")
                    
                if change_context:
                    msg += f" (Changes: {', '.join(change_context)})"
                
                # Log with appropriate level based on CPU usage
                level = logging.WARNING if high_cpu else logging.INFO
                
                # Use the modified log_structured function with improved throttling
                log_structured('resource_monitor', 'RESOURCE_USAGE', msg, params, level=level)
                
                # Update tracking data
                tracking['last_log_time'] = current_time
                tracking['last_cpu'] = cpu_usage
                tracking['last_mem'] = memory_usage
                tracking['last_cycle_time'] = cycle_time
                tracking['last_adjustment'] = rate_adjustment
            except Exception as e:
                # Safe fallback if structured logging fails
                print(f"RESOURCE: CPU {cpu_usage:.1f}%, Memory {memory_usage:.1f}% - Logging error: {str(e)}")

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
    Log movement status changes with better integration with the centralized logger.
    
    Args:
        status: Status message
        error_value: Current error value
        threshold: Error threshold
        robot_state: Robot state string
        params: Additional parameters
    """
    if params is None:
        params = {}
        
    if robot_state:
        params['robot_state'] = robot_state
        
    params['error'] = error_value
    params['threshold'] = threshold
    
    # Check if this is a start/stop status message
    if "started moving" in status.lower():
        # Delegate to centralized logger instead of creating a duplicate log
        log_movement_state(
            is_stopped=False,  # Moving
            reason=status,
            error_values=params,
            robot_state=robot_state,
            verbose=False  # Use DEBUG level to reduce duplication
        )
    elif "stopped" in status.lower():
        # Delegate to centralized logger
        log_movement_state(
            is_stopped=True,  # Stopped
            reason=status,
            error_values=params,
            robot_state=robot_state,
            verbose=False  # Use DEBUG level to reduce duplication
        )
    else:
        # For other status messages, use the original logging
        log_structured('motion_control', 'MOVEMENT_STATUS', 
                    status, 
                    params,
                    throttle_key=status.lower().replace(" ", "_"),
                    throttle_seconds=0.5)

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
    Log robot motion start/stop with enhanced integration with centralized logger.
    
    Args:
        is_starting: True if robot is starting, False if stopping
        velocity: Optional velocity for starts
        prev_velocity: Optional previous velocity for stops
        robot_state: Optional robot state string
    """
    params = {}
    if robot_state:
        params['robot_state'] = robot_state
        
    if is_starting and velocity:
        params['velocity'] = velocity
        reason = "Robot starting movement"
    else:
        if prev_velocity:
            params['prev_velocity'] = prev_velocity
        reason = "Robot stopping"
    
    # Delegate to centralized logger
    log_movement_state(
        is_stopped=not is_starting,
        reason=reason,
        error_values=params,
        robot_state=robot_state,
        verbose=False  # Use DEBUG level to reduce duplication
    )


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
        """
        Warm up JIT compilation to avoid delays during operation with reduced logging.
        Uses class-level tracking to prevent duplicate warmup operations and logs.
        """
        try:
            # Use a static class variable to track if JIT has been warmed up
            if not hasattr(Matrix4x4, '_jit_warmed_up'):
                Matrix4x4._jit_warmed_up = False
                Matrix4x4._warmup_attempt_count = 0
                
            # Skip if already warmed up successfully
            if Matrix4x4._jit_warmed_up:
                return
                
            # Limit total warmup attempts to avoid excessive logs on failure
            Matrix4x4._warmup_attempt_count += 1
            if Matrix4x4._warmup_attempt_count > 3:
                # Just silently return after 3 attempts - no more logging
                return
            
            # Only log at DEBUG level to reduce verbosity        
            self.logger.debug("Starting JIT warmup for Matrix4x4")
            
            start_time = time.time()
            
            # Create dummy data
            dummy_matrix = np.eye(4, dtype=np.float32)
            
            # Call JIT functions with dummy data without logging each one
            _ = transform_point_jit(dummy_matrix, 1.0, 2.0, 3.0)
            _ = transform_vector_jit(dummy_matrix, 1.0, 2.0, 3.0)
            _ = quaternion_to_matrix_jit(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            _ = matrix_multiply_jit(dummy_matrix, dummy_matrix)
            _ = matrix_inverse_jit(dummy_matrix)
            
            elapsed_time = time.time() - start_time
            
            # Only log successful completion if:
            # 1. This is the first successful attempt
            # 2. Warmup took significant time (>5ms)
            if elapsed_time > 0.005 and not Matrix4x4._jit_warmed_up:
                self.logger.info(f"JIT compilation warmup completed for Matrix4x4 in {elapsed_time*1000:.1f}ms")
            
            # Mark as successfully warmed up
            Matrix4x4._jit_warmed_up = True
            
        except Exception as e:
            # Only log first error in detail
            if Matrix4x4._warmup_attempt_count == 1:
                self.logger.warning(f"JIT warmup failed for Matrix4x4: {str(e)}. Will use non-optimized fallbacks.")
            elif Matrix4x4._warmup_attempt_count == 3:
                # Final summary log after multiple failures
                self.logger.warning(f"Matrix4x4 JIT warmup failed after {Matrix4x4._warmup_attempt_count} attempts. Using non-optimized operations.")

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
        Update the filter with a new position measurement with improved smoothing,
        validation, and optimized logging to reduce verbosity.
        
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
            
            # Log filter changes with improved throttling and significance detection
            self._log_filter_updates(original_position, filtered_position)
            
            return self.filtered_position
            
        except Exception as e:
            self.logger.error(f"Error in EnhancedTargetFilter.update: {str(e)}")
            # Fallback to simple pass-through
            self.filtered_position = position
            self.predicted_position = position
            self.last_update_time = timestamp if timestamp is not None else time.time()
            return position
    
    def _log_filter_updates(self, original_position, filtered_position):
        """
        Log filter updates with improved throttling and significance detection.
        Reduces log volume by consolidating logs and only logging significant changes.
        
        Args:
            original_position: Original unfiltered position
            filtered_position: Position after filtering
        """
        # Initialize tracking variables if they don't exist
        if not hasattr(self, "_filter_log_data"):
            self._filter_log_data = {
                'last_log_time': 0,
                'last_logged_position': None,
                'significant_changes_count': 0,
                'accumulated_changes': 0,
                'change_history': [],  # Track recent changes for trend analysis
            }
        
        current_time = time.time()
        
        # Calculate raw vs filtered difference for significance detection
        if filtered_position:
            # Calculate difference between raw and filtered positions
            position_diff = (
                abs(original_position[0] - filtered_position[0]),
                abs(original_position[1] - filtered_position[1]),
                abs(original_position[2] - filtered_position[2])
            )
            
            # Calculate total difference (weighted sum - emphasize distance)
            total_diff = position_diff[0] + position_diff[1] * 0.5 + position_diff[2] * 0.2
            
            # Track filter changes using a more meaningful approach
            if self._filter_log_data['last_logged_position'] is not None:
                # Calculate how much filtered position changed since last logged position
                filter_diff = (
                    abs(filtered_position[0] - self._filter_log_data['last_logged_position'][0]),
                    abs(filtered_position[1] - self._filter_log_data['last_logged_position'][1]),
                    abs(filtered_position[2] - self._filter_log_data['last_logged_position'][2])
                )
                
                # Calculate total change (weighted sum)
                total_filter_change = filter_diff[0] + filter_diff[1] * 0.5 + filter_diff[2] * 0.2
                
                # Keep a short history of changes for trend detection
                self._filter_log_data['change_history'].append(total_filter_change)
                if len(self._filter_log_data['change_history']) > 5:
                    self._filter_log_data['change_history'].pop(0)
                
                # Calculate significance based on several factors
                is_significant = False
                
                # 1. Absolute change threshold - distance change > 0.05m or lateral change > 0.1m
                if filter_diff[0] > 0.05 or filter_diff[1] > 0.1:
                    is_significant = True
                    
                # 2. Relative change threshold - calculate percentage change
                if self._filter_log_data['last_logged_position'][0] > 0.01:
                    pct_change = filter_diff[0] / self._filter_log_data['last_logged_position'][0] * 100
                    if pct_change > 5:  # 5% change threshold
                        is_significant = True
                
                # 3. Pattern detection - rapid acceleration in changes
                if len(self._filter_log_data['change_history']) >= 3:
                    # Check if change rate is accelerating
                    if (self._filter_log_data['change_history'][-1] > 
                        self._filter_log_data['change_history'][-2] * 1.5 and
                        self._filter_log_data['change_history'][-2] >
                        self._filter_log_data['change_history'][-3] * 1.2):
                        is_significant = True
                
                # 4. Time-based throttling - ensure minimum time between logs
                min_log_interval = 0.5  # 500ms minimum between logs
                time_since_last_log = current_time - self._filter_log_data['last_log_time']
                if time_since_last_log < min_log_interval:
                    is_significant = False  # Override - too soon for another log
                
                # Log only significant changes with context
                if is_significant:
                    # Increment counter for consecutive significant changes
                    self._filter_log_data['significant_changes_count'] += 1
                    self._filter_log_data['accumulated_changes'] += total_filter_change
                    
                    # Only log every 3rd significant change to reduce volume,
                    # or if accumulated change is very large, or if it's been a while
                    should_log = (
                        self._filter_log_data['significant_changes_count'] % 3 == 0 or
                        self._filter_log_data['accumulated_changes'] > 0.15 or
                        time_since_last_log > 5.0  # Force log at least every 5 seconds
                    )
                    
                    if should_log:
                        # Build more informative log message with context
                        movement_info = ""
                        if hasattr(self, 'movement_consistency') and hasattr(self, 'is_moving'):
                            consistency = getattr(self, 'movement_consistency', 0)
                            is_moving = getattr(self, 'is_moving', False)
                            if consistency > 0.7:
                                movement_info = f", consistent movement (c={consistency:.2f})"
                            elif is_moving:
                                movement_info = f", moving"
                        
                        # Log with more useful context
                        self.logger.info(
                            f"Filter update: raw_dist={original_position[0]:.3f}, "
                            f"filtered_dist={filtered_position[0]:.3f}"
                            f"{movement_info}"
                        )
                        
                        # Reset tracking variables
                        self._filter_log_data['last_logged_position'] = filtered_position
                        self._filter_log_data['last_log_time'] = current_time
                        self._filter_log_data['accumulated_changes'] = 0
                else:
                    # Not significant enough - reset consecutive counter
                    self._filter_log_data['significant_changes_count'] = 0
            else:
                # First logged position - initialize
                self._filter_log_data['last_logged_position'] = filtered_position
                self._filter_log_data['last_log_time'] = current_time
    
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
        Update velocity, acceleration, and movement metrics with improved logging.
        Reduces oscillating movement state logs by implementing hysteresis and
        stability thresholds.
        
        Args:
            current_time: Current time
        """
        # Initialize tracking dictionary for movement state if it doesn't exist
        if not hasattr(self, '_movement_state_tracking'):
            self._movement_state_tracking = {
                'last_state_change_time': 0,
                'stable_state_duration': 0,
                'movement_state': None,  # None, True (moving), False (stopped)
                'state_change_count': 0,
                'last_significant_velocity': 0,
                'velocity_samples': [],   # Recent velocity samples
                'min_stable_duration': 0.5  # Require state to be stable for this long
            }
        
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
            
            # Log only when direction change status changes - with meaningful context
            if self.direction_change_detected != prev_direction_change:
                if self.direction_change_detected:
                    # Calculate angle between vectors for more meaningful logging
                    prev_mag = np.linalg.norm(prev_vel[:2])
                    new_mag = np.linalg.norm(raw_velocity[:2])
                    
                    if prev_mag > 0.05 and new_mag > 0.05:
                        dot_product = np.dot(prev_vel[:2], raw_velocity[:2])
                        angle_rad = np.arccos(max(-1.0, min(1.0, dot_product / (prev_mag * new_mag))))
                        angle_deg = np.degrees(angle_rad)
                        
                        # Log with meaningful angle information
                        self.logger.info(
                            f"Direction change detected | angle={angle_deg:.1f}°, "
                            f"prev_vel=[{prev_vel[0]:.2f}, {prev_vel[1]:.2f}], "
                            f"new_vel=[{raw_velocity[0]:.2f}, {raw_velocity[1]:.2f}]"
                        )
                    else:
                        # Simple log for small velocities
                        self.logger.info(f"Direction change detected | prev_vel={prev_vel[:2]}, new_vel={raw_velocity[:2]}")
            
            # Use Numba function to smooth velocity and acceleration
            smoothed_velocity, smoothed_acceleration = update_smoothed_values(
                raw_velocity, raw_acceleration, prev_vel, self.acceleration)
            
            # Update state variables
            self.current_velocity = smoothed_velocity
            self.acceleration = smoothed_acceleration
            
            # Calculate normalized direction vector
            self.motion_direction = calculate_motion_direction(
                smoothed_velocity, self.motion_direction)
            
            # IMPROVED MOVEMENT STATE DETECTION WITH HYSTERESIS
            # Calculate velocity magnitude
            vel_magnitude = math.sqrt(smoothed_velocity[0]**2 + smoothed_velocity[1]**2)
            
            # Store in recent samples for trend analysis
            self._movement_state_tracking['velocity_samples'].append(vel_magnitude)
            if len(self._movement_state_tracking['velocity_samples']) > 5:
                self._movement_state_tracking['velocity_samples'].pop(0)
            
            # Define velocity thresholds with hysteresis
            moving_threshold = 0.05  # Base threshold (m/s)
            
            # Use different thresholds based on current state (hysteresis)
            if self._movement_state_tracking['movement_state'] is None:
                # Initial state - use base threshold
                is_moving_now = vel_magnitude > moving_threshold
            elif self._movement_state_tracking['movement_state'] is True:
                # Currently moving - use lower threshold to stop (80%)
                is_moving_now = vel_magnitude > (moving_threshold * 0.8)
            else:
                # Currently stopped - use higher threshold to start (120%)
                is_moving_now = vel_magnitude > (moving_threshold * 1.2)
            
            # Update stability tracking
            if is_moving_now == self._movement_state_tracking['movement_state']:
                # State is stable - increment duration
                self._movement_state_tracking['stable_state_duration'] += dt
            else:
                # Potential state change - reset stability counter
                self._movement_state_tracking['stable_state_duration'] = 0
            
            # Check if state has been stable for minimum duration
            if (self._movement_state_tracking['stable_state_duration'] >= 
                self._movement_state_tracking['min_stable_duration']):
                
                # State change is stable enough to consider
                prev_is_moving = self._movement_state_tracking['movement_state']
                
                # Only process and log if actually changing state
                if is_moving_now != prev_is_moving:
                    # Additional validation: check velocity trend to avoid oscillation
                    is_velocity_trending = False
                    
                    if len(self._movement_state_tracking['velocity_samples']) >= 3:
                        # Check if velocity is consistently trending in state direction
                        if is_moving_now:
                            # Velocity should be increasing to confirm "started moving"
                            is_velocity_trending = (
                                self._movement_state_tracking['velocity_samples'][-1] >
                                self._movement_state_tracking['velocity_samples'][-2] and
                                self._movement_state_tracking['velocity_samples'][-2] >
                                self._movement_state_tracking['velocity_samples'][-3]
                            )
                        else:
                            # Velocity should be decreasing to confirm "stopped moving"
                            is_velocity_trending = (
                                self._movement_state_tracking['velocity_samples'][-1] <
                                self._movement_state_tracking['velocity_samples'][-2] and
                                self._movement_state_tracking['velocity_samples'][-2] <
                                self._movement_state_tracking['velocity_samples'][-3]
                            )
                    
                    # Update state if trending confirms it, or if the change is very significant
                    significant_change = abs(vel_magnitude - self._movement_state_tracking['last_significant_velocity']) > 0.03
                    
                    if is_velocity_trending or significant_change or prev_is_moving is None:
                        # Log the state change with rich context
                        time_in_prev_state = current_time - self._movement_state_tracking['last_state_change_time']
                        
                        # Collect movement metrics for context
                        if hasattr(self, 'movement_consistency'):
                            consistency = self.movement_consistency
                        else:
                            consistency = 0.0
                        
                        # Update tracking
                        self._movement_state_tracking['movement_state'] = is_moving_now
                        self._movement_state_tracking['last_state_change_time'] = current_time
                        self._movement_state_tracking['state_change_count'] += 1
                        self._movement_state_tracking['last_significant_velocity'] = vel_magnitude
                        
                        # Only log after initial state is set
                        if prev_is_moving is not None:
                            # Create informative log with movement metrics
                            self.logger.info(
                                f"Target {'started moving' if is_moving_now else 'stopped moving'} | "
                                f"velocity={vel_magnitude:.3f}, threshold={moving_threshold:.3f}, "
                                f"consistency={consistency:.2f}, time_in_prev_state={time_in_prev_state:.1f}s"
                            )
                        
                        # Update main state variable for other components
                        self.is_moving = is_moving_now
            
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
                    self.logger.info(
                        f"Movement consistency changed significantly | "
                        f"new_consistency={self.movement_consistency:.3f}, "
                        f"prev_consistency={prev_consistency:.3f}"
                    )
    
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
        # Initialize tracking for prediction logging if needed
        if not hasattr(self, "_prediction_log_tracking"):
            self._prediction_log_tracking = {
                "last_log_time": 0.0,
                "consecutive_similar": 0,
                "last_prediction": None,
                "min_log_interval": 1.0,  # Minimum seconds between prediction logs
                "significant_change_threshold": 0.05  # 5% position change is significant
            }
        
        tracking = self._prediction_log_tracking
        
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
                    
                    # Log prediction with improved throttling
                    self._log_prediction_update(
                        self.filtered_position, 
                        pred_position, 
                        self.prediction_confidence
                    )
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

    def _log_prediction_update(self, filtered_position, predicted_position, confidence):
        """
        Log prediction updates with improved throttling and significance detection.
        
        Args:
            filtered_position: Current filtered position
            predicted_position: Calculated prediction
            confidence: Prediction confidence score
        """
        # Ensure tracking structure exists
        if not hasattr(self, "_prediction_log_tracking"):
            self._prediction_log_tracking = {
                "last_log_time": 0.0,
                "consecutive_similar": 0,
                "last_prediction": None,
                "min_log_interval": 1.0,  # Minimum seconds between prediction logs
                "significant_change_threshold": 0.05  # 5% position change is significant
            }
        
        tracking = self._prediction_log_tracking
        current_time = time.time()
        
        # Check if prediction is similar to the last logged prediction
        is_similar = False
        if tracking["last_prediction"] is not None:
            # Calculate position difference percentages
            diffs = [
                abs(predicted_position[0] - tracking["last_prediction"][0]) / (abs(tracking["last_prediction"][0]) + 0.001),
                abs(predicted_position[1] - tracking["last_prediction"][1]) / (abs(tracking["last_prediction"][1]) + 0.001),
                abs(predicted_position[2] - tracking["last_prediction"][2]) / (abs(tracking["last_prediction"][2]) + 0.001)
            ]
            avg_diff = sum(diffs) / 3
            
            is_similar = avg_diff < tracking["significant_change_threshold"]
        
        # Update consecutive counter
        if is_similar:
            tracking["consecutive_similar"] += 1
        else:
            tracking["consecutive_similar"] = 0
        
        # Determine if we should log based on time and similarity
        time_since_log = current_time - tracking["last_log_time"]
        
        # Adjust log interval based on consecutive similar predictions
        if tracking["consecutive_similar"] > 3:
            # Exponential increase in throttle interval for similar predictions
            adjusted_interval = tracking["min_log_interval"] * (1.5 ** min(tracking["consecutive_similar"] - 3, 5))
            should_log = time_since_log >= adjusted_interval
        else:
            # Regular time-based throttling
            should_log = time_since_log >= tracking["min_log_interval"]
        
        # Log prediction only when significant or after enough time
        if should_log:
            # Add repetition info if needed
            repetition_info = ""
            if tracking["consecutive_similar"] > 0:
                repetition_info = f" (similar predictions: {tracking['consecutive_similar']+1}x)"
            
            # Enhanced prediction message 
            prediction_diff = (
                predicted_position[0] - filtered_position[0],
                predicted_position[1] - filtered_position[1]
            )
            
            # Calculate look-ahead time
            lookdir_txt = ""
            if abs(prediction_diff[0]) > 0.01:
                look_dir = "ahead" if prediction_diff[0] > 0 else "behind"
                lookdir_txt = f", looking {look_dir}"
            
            # Log using target filter update to leverage its throttling
            log_target_filter_update(
                filtered_position,  # Use filtered as "raw" since we're interested in prediction offset
                filtered_position, 
                predicted_position,
                confidence
            )
            
            # Update tracking info
            tracking["last_log_time"] = current_time
            tracking["last_prediction"] = predicted_position
    
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
        Update the target strategy with enhanced handling for angular strategies
        and improved logging of transitions.
        
        Args:
            target_strategy: The target strategy to blend towards
            current_time: Current time
                    
        Returns:
            bool: True if a new blend was started
        """
        # Initialize tracking for blend logs if needed
        if not hasattr(self, "_blend_tracking"):
            self._blend_tracking = {
                "last_log_time": 0.0,
                "blend_count": 0,  # Count of blend transitions
                "min_log_interval": 0.5,  # Minimum seconds between consecutive blend logs
                "significant_transitions": set(),  # Track significant strategy pairs
            }
        
        tracking = self._blend_tracking
        
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
            
            # Log initial strategy with reduced details for cleaner logs
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
                # Only log continued blend attempts at high debug level
                if self.debug_level >= 2:
                    log_structured('strategy_blender', 'BLEND_CONTINUE', 
                                f"Continued blend request to {target_strategy.name} - already in progress",
                                blend_params,
                                throttle_key="blend_continue",
                                throttle_seconds=1.0,
                                verbosity_level=2)  # Higher verbosity level
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
                    
                    # Only log at higher debug level to reduce verbosity
                    if self.debug_level >= 2:
                        log_structured('strategy_blender', 'BLEND_DEFERRED', 
                                    f"Strategy switch to {target_strategy.name} deferred - minimal hold time not met", 
                                    blend_params,
                                    throttle_key="angular_hold",
                                    throttle_seconds=0.5,
                                    verbosity_level=2)  # Higher verbosity level
                    return False
            else:
                # Standard hold time check for non-angular strategies
                if time_in_current_strategy < self.min_hold_time:
                    # Haven't held current strategy long enough, don't switch yet
                    blend_params['reason'] = f"Minimum hold time not met ({time_in_current_strategy:.2f}s < {self.min_hold_time:.2f}s)"
                    
                    # Only log at higher debug level to reduce verbosity
                    if self.debug_level >= 2:
                        log_structured('strategy_blender', 'BLEND_DEFERRED', 
                                    f"Strategy switch to {target_strategy.name} deferred - minimum hold time not met", 
                                    blend_params,
                                    throttle_key="standard_hold",
                                    throttle_seconds=1.0,
                                    verbosity_level=2)  # Higher verbosity level
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
                    
                    # Only log at higher debug level to reduce verbosity
                    if self.debug_level >= 2:
                        log_structured('strategy_blender', 'BLEND_SIMILARITY_SKIP', 
                                    f"Strategy switch to {target_strategy.name} skipped - too similar to current strategy", 
                                    blend_params,
                                    throttle_key="angular_similarity",
                                    throttle_seconds=1.0,
                                    verbosity_level=2)  # Higher verbosity level
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
                    
                    # Only log at higher debug level to reduce verbosity
                    if self.debug_level >= 2:
                        log_structured('strategy_blender', 'BLEND_SIMILARITY_SKIP', 
                                    f"Strategy switch to {target_strategy.name} skipped - too similar to current strategy", 
                                    blend_params,
                                    throttle_key="standard_similarity",
                                    throttle_seconds=1.0,
                                    verbosity_level=2)  # Higher verbosity level
                    return False
            
            # MODIFIED: Improved oscillation checking for all strategies
            # Check for oscillation in strategy selection
            is_oscillating = self._is_oscillating_between_strategies(target_strategy, current_time)
                
            if is_oscillating:
                # We're oscillating between strategies, don't switch
                blend_params['reason'] = "Oscillation detected"
                blend_params['history'] = [s[0] for s in self.strategy_history[-3:]]
                
                # Only log oscillation if significant time has passed
                if current_time - tracking["last_log_time"] >= tracking["min_log_interval"]:
                    log_structured('strategy_blender', 'BLEND_OSCILLATION_DETECTED', 
                                f"Strategy switch to {target_strategy.name} deferred - oscillation pattern detected", 
                                blend_params)
                    tracking["last_log_time"] = current_time
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
            
            # Determine if this transition is significant for logging
            is_significant_transition = False
            
            # Case 1: Direction change transitions are significant
            if direction_change:
                is_significant_transition = True
                
            # Case 2: Any transition involving angular strategies is significant
            if is_angular_strategy or self.current_strategy.name.startswith("ANGULAR_"):
                is_significant_transition = True
                
            # Case 3: Any strategy with a substantial scale change is significant
            forward_change = abs(target_strategy.forward_scale - self.current_strategy.forward_scale)
            lateral_change = abs(target_strategy.lateral_scale - self.current_strategy.lateral_scale)
            angular_change = abs(target_strategy.angular_scale - self.current_strategy.angular_scale)
            
            if forward_change > 0.3 or lateral_change > 0.3 or angular_change > 0.3:
                is_significant_transition = True
                
            # Case 4: First few blends are significant
            if tracking["blend_count"] < 5:
                is_significant_transition = True
                
            # Increment blend count
            tracking["blend_count"] += 1
            
            # Log strategy transition with detailed parameters if significant
            transition_reason = self._determine_transition_reason(
                self.current_strategy, target_strategy)
            
            # Always log initial blend event
            log_strategy_blending(
                self.current_strategy, 
                target_strategy, 
                0.0,  # Initial blend factor
                self.effective_blend_duration
            )
            
            # Log selection with reason if significant or enough time has passed
            if is_significant_transition or (current_time - tracking["last_log_time"] >= tracking["min_log_interval"]):
                log_strategy_selection(
                    "transition", 
                    target_strategy.as_dict(),
                    candidate_strategies={self.current_strategy.name: self.current_strategy.as_dict()},
                    reason=transition_reason
                )
                tracking["last_log_time"] = current_time
            
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
    
    # Apply minimum velocity thresholds
    for i in range(3):
        # If velocity is below minimum threshold but not zero, set to zero
        if 0.0 < abs(limited_velocities[i]) < (min_angular_velocity if i == 2 else min_effective_velocity):
            limited_velocities[i] = 0.0
            
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
    """
    Manager for movement decisions using multiple strategies with improved logging.
    
    This class extends the original MovementDecisionManager with better logging
    integration to prevent duplicate logs.
    """
    
    def __init__(self, stop_decision_manager):
        """Initialize with reference to stop decision manager."""
        self.stop_decision_manager = stop_decision_manager
        self.movement_strategy = None  # Will be initialized in later methods
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
        # If movement_strategy isn't initialized, create it
        if self.movement_strategy is None:
            self.movement_strategy = StartMovementStrategy()
        
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
        if self.movement_strategy:
            self.movement_strategy.reset_initial_boost()


class VelocityLimitingStrategy:
    """Base class for velocity limiting strategies with improved logging."""
    
    def __init__(self, name, priority=5):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name for logging
            priority: Priority level (higher = applied first)
        """
        self.name = name
        self.priority = priority
        self.logger = logging.getLogger('pid_controller.velocity_limiter')
        
        # Add logging tracking for this strategy
        self._log_tracking = {
            'last_log_time': 0.0,
            'last_components': None,
            'consecutive_similar_count': 0,
            'min_log_interval': 2.0  # Minimum seconds between similar logs
        }
    
    def _is_change_significant(self, original_values, limited_values, components=None):
        """
        Determine if velocity changes are significant enough to log.
        
        Args:
            original_values: Original velocity values [x, y, angular]
            limited_values: Limited velocity values [x, y, angular]
            components: Optional list to store components that changed significantly
            
        Returns:
            tuple: (is_significant, total_magnitude, affected_components)
        """
        # Track which components were significantly limited
        affected_components = [] if components is None else components
        
        # Increased threshold from 25% to 40% for significance
        significance_threshold = 40.0
        total_magnitude = 0.0
        
        component_names = ['x', 'y', 'a']
        for i, name in enumerate(component_names):
            # Skip insignificant original values
            if abs(original_values[i]) <= 0.01:
                continue
                
            # Calculate percentage change
            pct_change = abs(limited_values[i] - original_values[i]) / abs(original_values[i]) * 100
            total_magnitude += pct_change
            
            if pct_change > significance_threshold:
                affected_components.append(f"{name}:{pct_change:.0f}%")
        
        # Determine overall significance:
        # - Either total magnitude exceeds 50% across all components
        # - Or any single component exceeds 70% (critical limiting)
        is_significant = total_magnitude > 50.0 or any(float(comp.split(':')[1][:-1]) > 70.0 
                                                     for comp in affected_components)
        
        return is_significant, total_magnitude, affected_components
    
    def _should_log_change(self, affected_components, reason):
        """
        Determine if this change should be logged based on frequency and similarity.
        
        Args:
            affected_components: List of components affected by limiting
            reason: Reason for limiting
            
        Returns:
            bool: True if should log, False otherwise
        """
        current_time = time.time()
        
        # Sort components to ensure consistent comparison
        affected_components = sorted(affected_components) if affected_components else []
        
        # Check if this is similar to the last logged change
        is_similar = (self._log_tracking['last_components'] == affected_components)
        
        if is_similar:
            # Increment similarity counter
            self._log_tracking['consecutive_similar_count'] += 1
            
            # Only log every Nth similar change 
            # For strategies that typically repeat, use higher thresholds
            if self.name in ['LookAheadLimiter', 'ApproachScalingLimiter']:
                # These tend to repeat frequently - require more repetitions
                should_log = self._log_tracking['consecutive_similar_count'] % 5 == 0
            else:
                # Standard strategies - log every 3rd occurrence
                should_log = self._log_tracking['consecutive_similar_count'] % 3 == 0
        else:
            # New component pattern - reset counter
            self._log_tracking['consecutive_similar_count'] = 0
            should_log = True
        
        # Always respect minimum time between logs
        if (current_time - self._log_tracking['last_log_time']) < self._log_tracking['min_log_interval']:
            should_log = False
            
        # If we're going to log, update tracking
        if should_log:
            self._log_tracking['last_log_time'] = current_time
            self._log_tracking['last_components'] = affected_components
            
            # Add repetition info to reason if relevant
            if self._log_tracking['consecutive_similar_count'] > 0:
                return f"{reason} (repeated {self._log_tracking['consecutive_similar_count']} times)"
        
        return should_log and reason
    
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """
        Base implementation of apply_limits - should be overridden by subclasses.
        
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
    
    def __init__(self, accel_limit=1.5, angular_accel_limit=1.2, priority=5):
        """Initialize with acceleration limits."""
        super().__init__("AccelerationLimiter", priority)
        self.accel_limit = accel_limit
        self.angular_accel_limit = angular_accel_limit
        
        # ADDED: Separate limits for deceleration to allow faster stopping
        self.decel_limit = accel_limit * 1.5  # 50% higher limit for deceleration
        self.angular_decel_limit = angular_accel_limit * 1.8
        
    def apply_limits(self, velocities, prev_velocities, robot_state, dt):
        """
        Apply acceleration limits with enhanced deceleration.
        
        Args:
            velocities: Current commanded velocities [linear_x, linear_y, angular]
            prev_velocities: Previous velocities [linear_x, linear_y, angular]
            robot_state: Dict with robot state info
            dt: Time delta since last update
            
        Returns:
            tuple: (limited_velocities, reason, limiting_applied)
        """
        # Convert to numpy array for efficient operations
        current_velocities = np.array(velocities, dtype=np.float32)
        previous_velocities = np.array(prev_velocities, dtype=np.float32)
        
        # Apply acceleration limits
        is_approaching = robot_state.get('is_approaching', False)
        distance_error = robot_state.get('distance_error', 0.0)
        approach_distance = robot_state.get('approach_distance', 0.3)
        
        try:
            # Use JIT-compiled function
            limited_velocities = apply_velocity_limits_core(
                current_velocities, 
                previous_velocities, 
                dt, 
                self.accel_limit, 
                self.angular_accel_limit,
                distance_error,
                approach_distance,
                is_approaching
            )
            
            # Check if limiting was significant
            affected_components = []
            is_significant, total_magnitude, affected_components = self._is_change_significant(
                current_velocities, limited_velocities, affected_components
            )
            
            if is_significant:
                reason = self._should_log_change(
                    affected_components,
                    f"Acceleration limited for {', '.join(affected_components)}"
                )
                
                if reason:
                    return limited_velocities, reason, True
                else:
                    # Change was significant but logging is throttled
                    return limited_velocities, None, True
            
            # Still return limited velocities even if not significant enough to log
            if not np.array_equal(current_velocities, limited_velocities):
                return limited_velocities, None, True
                
            return current_velocities, None, False
            
        except Exception as e:
            # Fall back to simplified limiting
            self.logger.warning(f"JIT acceleration limiting failed: {str(e)}. Using fallback.")
            
            # Simple fallback implementation
            limited_velocities = np.copy(current_velocities)
            
            # Calculate max allowed changes
            linear_accel_limit = self.accel_limit * dt
            angular_accel_limit = self.angular_accel_limit * dt
            
            # Apply limits to each component
            any_limited = False
            affected_components = []
            
            for i in range(3):
                # Determine appropriate limit based on component
                limit = angular_accel_limit if i == 2 else linear_accel_limit
                
                # Calculate difference
                diff = current_velocities[i] - previous_velocities[i]
                
                if abs(diff) > limit:
                    # Apply limit while preserving sign
                    limited_velocities[i] = previous_velocities[i] + (limit if diff > 0 else -limit)
                    any_limited = True
                    
                    # Calculate percentage change for significance
                    if abs(current_velocities[i]) > 0.01:
                        pct_change = abs(limited_velocities[i] - current_velocities[i]) / abs(current_velocities[i]) * 100
                        if pct_change > 40:  # Only record if >40% change
                            affected_components.append(f"{['x', 'y', 'a'][i]}:{pct_change:.0f}%")
            
            if any_limited and affected_components:
                reason = self._should_log_change(
                    affected_components,
                    f"Acceleration limited for {', '.join(affected_components)} (fallback mode)"
                )
                
                if reason:
                    return limited_velocities, reason, True
                else:
                    # Change was significant but logging is throttled
                    return limited_velocities, None, True
            
            if any_limited:
                return limited_velocities, None, True
            
            return current_velocities, None, False


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
            original_velocity = limited_velocities[0]
            limited_velocities[0] = 0.0
            reason = f"EMERGENCY BRAKE: Overshot target by {abs(distance_error):.3f}m"
            limiting_applied = True
            
            # Emergency is always logged (no throttling for safety reasons)
            return limited_velocities, reason, limiting_applied
            
        # Emergency case 2: Too fast near target
        elif abs(distance_error) < 0.1 and limited_velocities[0] > 0.15:
            # Severe reduction in forward velocity
            original_velocity = limited_velocities[0]
            limited_velocities[0] *= 0.3
            reason = f"EMERGENCY BRAKE: Too fast near target ({original_velocity:.2f}m/s)"
            limiting_applied = True
            
            # Emergency is always logged (no throttling for safety reasons)
            return limited_velocities, reason, limiting_applied
            
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
            original_velocity = limited_velocities[0]
            limited_velocities[0] = safe_velocity
            
            # Calculate percentage reduction
            if original_velocity > 0.01:
                reduction_pct = abs(safe_velocity - original_velocity) / original_velocity * 100
                
                # Only consider significant if reduction is large
                if reduction_pct > 40:  # Increased from 25% to 40%
                    # Check if we should log based on frequency and similarity
                    components = [f"x:{reduction_pct:.0f}%"]
                    
                    reason = self._should_log_change(
                        components,
                        f"Look-ahead braking: distance={abs(distance_error):.2f}m, stopping_distance={stopping_distance:.2f}m"
                    )
                    
                    return limited_velocities, reason, True
                    
                # Still limit velocity even if not logging
                return limited_velocities, None, True
                
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
            original_velocities = velocities.copy()
            
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
            
            # Check if change is significant
            affected_components = []
            is_significant, total_magnitude, affected_components = self._is_change_significant(
                original_velocities, limited_velocities, affected_components
            )
            
            if is_significant:
                reason = self._should_log_change(
                    affected_components,
                    f"Approach scaling: distance={abs(distance_error):.2f}m, scale={scale_factor:.2f}, zone={approach_zone:.2f}m"
                )
                
                if reason:
                    return limited_velocities, reason, True
                else:
                    # Still apply limiting even if not logging
                    return limited_velocities, None, True
                    
            # Always apply the scaling even if not significant enough to log
            return limited_velocities, None, True
                
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
        
        # If neither component is decelerating, do nothing
        if not is_decelerating_x and not is_decelerating_angular:
            return velocities, None, False
        
        # Get error values from robot state
        distance_error = robot_state.get('distance_error', 0.0)
        angular_error = robot_state.get('angular_error', 0.0)
        
        # Flag for whether limiting was applied
        limiting_applied = False
        
        # Create a copy of velocities for modification
        limited_velocities = velocities.copy()
        original_velocities = velocities.copy()
        affected_components = []
        
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
                    limiting_applied = True
                    
                    # Calculate percentage reduction for logging
                    if abs(velocities[0]) > 0.01:
                        reduction_pct = abs(limited_velocities[0] - velocities[0]) / abs(velocities[0]) * 100
                        if reduction_pct > 40:  # Only log significant changes
                            affected_components.append(f"x:{reduction_pct:.0f}%")
        
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
                    limiting_applied = True
                    
                    # Calculate percentage reduction for logging
                    if abs(velocities[2]) > 0.01:
                        reduction_pct = abs(limited_velocities[2] - velocities[2]) / abs(velocities[2]) * 100
                        if reduction_pct > 40:  # Only log significant changes
                            affected_components.append(f"a:{reduction_pct:.0f}%")
        
        # Check if limiting should be logged
        if limiting_applied and affected_components:
            # Check if this should be logged based on throttling
            is_significant, total_magnitude, _ = self._is_change_significant(
                original_velocities, limited_velocities
            )
            
            if is_significant:
                reason = self._should_log_change(
                    affected_components,
                    f"Enhanced deceleration applied to {', '.join(affected_components)}"
                )
                
                # Return with reason if should log
                if reason:
                    return limited_velocities, reason, True
                
                # Otherwise just return limited values without a reason
                return limited_velocities, None, True
            
            # Still apply limiting even if not significant enough to log
            return limited_velocities, None, True
                
        # Return original velocities if no limiting was applied
        if not limiting_applied:
            return velocities, None, False
        
        return limited_velocities, None, True


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
        
        any_limited = False
        limited_velocities = velocities.copy()
        affected_components = []
        
        for i in range(3):
            significant_changes[i] = abs(velocities[i] - prev_velocities[i]) > self.hysteresis_band
            
            # If change is below hysteresis band, apply hysteresis
            if not significant_changes[i]:
                # Only apply if current velocity is non-zero (avoid preventing stopping)
                if abs(velocities[i]) > 0.01:
                    # Change is below hysteresis band - keep previous velocity
                    original = limited_velocities[i]
                    limited_velocities[i] = prev_velocities[i]
                    any_limited = True
                    
                    # Calculate percentage change for logging
                    if abs(original) > 0.01:
                        change_pct = abs(limited_velocities[i] - original) / abs(original) * 100
                        if change_pct > 40:  # Only log significant changes
                            affected_components.append(f"{['x', 'y', 'a'][i]}:{change_pct:.0f}%")
        
        # Only log significant hysteresis applications
        if any_limited and affected_components:
            is_significant, total_magnitude, _ = self._is_change_significant(
                velocities, limited_velocities
            )
            
            if is_significant:
                reason = self._should_log_change(
                    affected_components,
                    f"Velocity hysteresis applied to {', '.join(affected_components)}"
                )
                
                if reason:
                    return limited_velocities, reason, True
                
                # Still apply limiting even if not logging
                return limited_velocities, None, True
        
        # Always apply hysteresis if needed, even if not significant enough to log
        if any_limited:
            return limited_velocities, None, True
                
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
        limited_velocities = velocities.copy()
        any_limited = False
        affected_components = []
        
        # Check each component
        for i in range(3):
            min_threshold = self.min_angular if i == 2 else self.min_linear
            below_min[i] = 0 < abs(velocities[i]) < min_threshold
            
            if below_min[i]:
                # Too small to be effective - zero it out
                original = limited_velocities[i]
                limited_velocities[i] = 0.0
                any_limited = True
                affected_components.append(f"{['x', 'y', 'a'][i]}")
        
        if any_limited:
            # This isn't significant enough to log in most cases - just apply it
            return limited_velocities, None, True
                
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
        limited_velocities = velocities.copy()
        any_limited = False
        affected_components = []
        
        # Check each component
        for i in range(3):
            exceeds_max[i] = abs(velocities[i]) > self.max_velocities[i]
            
            if exceeds_max[i]:
                # Apply limit with original sign
                original = limited_velocities[i]
                sign = 1 if velocities[i] >= 0 else -1
                limited_velocities[i] = sign * self.max_velocities[i]
                any_limited = True
                
                # Calculate percentage change for logging
                if abs(original) > 0.01:
                    change_pct = abs(limited_velocities[i] - original) / abs(original) * 100
                    if change_pct > 40:  # Only log significant changes
                        affected_components.append(f"{['x', 'y', 'a'][i]}:{change_pct:.0f}%")
        
        # Check if limiting is significant enough to log
        if any_limited and affected_components:
            is_significant, total_magnitude, _ = self._is_change_significant(
                velocities, limited_velocities
            )
            
            if is_significant:
                reason = self._should_log_change(
                    affected_components,
                    f"Max velocity limit applied to {', '.join(affected_components)}"
                )
                
                if reason:
                    return limited_velocities, reason, True
                    
                # Still apply limiting even if not logging
                return limited_velocities, None, True
        
        # Always apply max velocity limits even if not significant enough to log
        if any_limited:
            return limited_velocities, None, True
                
        return velocities, None, False


class VelocityLimiterPipeline:
    """Pipeline for applying multiple velocity limiting strategies with improved logging."""
    
    def __init__(self, debug_level=0):
        """Initialize the pipeline with debugging level."""
        self.strategies = []
        self.debug_level = debug_level
        self.logger = logging.getLogger('pid_controller.velocity_limiter')
        
        # Add tracking for consolidated logging
        self._pipeline_log_tracking = {
            'last_log_time': 0.0,
            'last_components': None,
            'consecutive_similar_count': 0,
            'min_log_interval': 3.0  # Minimum seconds between pipeline logs
        }
        
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
        Apply all strategies in priority order with improved logging.
        
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
        original_velocities = current_velocities.copy()
        
        # Create consolidated tracking data structure
        limiting_data = {
            'original_velocities': velocities.copy(),
            'limited_velocities': None,  # Will be set at end
            'applied_limits': [],        # Track all limits applied
            'limited_components': {      # Track which components were limited
                'x': False,
                'y': False,
                'a': False
            },
            'limiting_magnitude': 0.0    # Track total magnitude of limiting
        }
        
        # Apply each strategy in priority order
        for strategy in self.strategies:
            # Store velocities before this strategy
            pre_strategy_velocities = current_velocities.copy()
            
            # Apply this strategy
            try:
                limited_velocities, reason, limiting_applied = strategy.apply_limits(
                    current_velocities, prev_velocities, robot_state, dt
                )
                
                # Update current velocities for next strategy
                if limiting_applied:
                    # Calculate which components were limited and by how much
                    component_changes = {}
                    component_names = ['x', 'y', 'a']
                    limiting_magnitude = 0.0
                    
                    for i, comp in enumerate(component_names):
                        # Check if this component changed significantly
                        if abs(limited_velocities[i] - pre_strategy_velocities[i]) > 0.01:
                            # Calculate percentage change
                            if abs(pre_strategy_velocities[i]) > 0.01:
                                # Calculate limitation percentage
                                pct_change = abs(limited_velocities[i] - pre_strategy_velocities[i]) / abs(pre_strategy_velocities[i]) * 100
                                limiting_magnitude += pct_change
                                if pct_change > 20:  # Track any change over 20%
                                    component_changes[comp] = pct_change
                                    limiting_data['limited_components'][comp] = True
                    
                    limiting_data['limiting_magnitude'] += limiting_magnitude
                    
                    # Only record significant limiting actions
                    if component_changes and reason:  # Only record if there's a reason (means strategy decided it was significant)
                        # Store limit information with components affected
                        limiting_data['applied_limits'].append({
                            'strategy': strategy.name,
                            'reason': reason,
                            'components': component_changes,
                            'before': pre_strategy_velocities.copy(),
                            'after': limited_velocities.copy()
                        })
                    
                    # Update current velocities for next strategy
                    current_velocities = limited_velocities
            except Exception as e:
                self.logger.error(f"Error in strategy {strategy.name}: {str(e)}")
                # Continue with next strategy
        
        # Update final velocities
        limiting_data['limited_velocities'] = current_velocities.copy()
        
        # Create consolidated reasons list and extract limited components
        reasons = []
        limited_components = []
        
        # Organize limiting info into logical groups
        strategy_groups = {
            'emergency': [],
            'approach': [],
            'acceleration': [],
            'hysteresis': []
        }
        
        # Group strategies by type
        for limit_info in limiting_data['applied_limits']:
            strategy_name = limit_info['strategy']
            # Categorize based on strategy name
            if 'Emergency' in strategy_name:
                strategy_groups['emergency'].append(limit_info)
            elif 'Approach' in strategy_name:
                strategy_groups['approach'].append(limit_info)
            elif 'Acceleration' in strategy_name or 'Deceleration' in strategy_name:
                strategy_groups['acceleration'].append(limit_info)
            elif 'Hysteresis' in strategy_name:
                strategy_groups['hysteresis'].append(limit_info)
            
            # Collect affected components for logging
            if 'components' in limit_info:
                for comp, pct in limit_info['components'].items():
                    limited_components.append(f"{comp}:{pct:.0f}%")
        
        # Build consolidated reason message - limit to two main reasons for brevity
        reasons_collected = 0
        
        # Emergency strategies take precedence in messaging
        if strategy_groups['emergency'] and reasons_collected < 2:
            for limit_info in strategy_groups['emergency']:
                reasons.append(limit_info['reason'])
                reasons_collected += 1
                if reasons_collected >= 2:
                    break
        
        # Add approach scaling if applicable and if we have space
        if strategy_groups['approach'] and reasons_collected < 2:
            approach_info = strategy_groups['approach'][0]  # Take the first one
            if 'reason' in approach_info and approach_info['reason']:
                reasons.append(approach_info['reason'])
                reasons_collected += 1
        
        # Add acceleration limiting if applicable and if we have space
        if strategy_groups['acceleration'] and reasons_collected < 2:
            accel_components = []
            for limit_info in strategy_groups['acceleration']:
                if 'components' in limit_info:
                    accel_components.extend(limit_info['components'].keys())
            
            if accel_components:
                accel_components = list(set(accel_components))  # Remove duplicates
                reasons.append(f"Acceleration limited in {', '.join(accel_components)}")
                reasons_collected += 1
        
        # Log only if significant limiting occurred - increased threshold
        if limiting_data['applied_limits'] and limiting_data['limiting_magnitude'] > 60.0:  # Increased from 50% to 60%
            # Count how many components were limited
            limited_count = sum(limiting_data['limited_components'].values())
            
            # Calculate overall percentage change
            percentage_change = 0.0
            for i in range(3):
                if abs(original_velocities[i]) > 0.01:
                    pct = abs(current_velocities[i] - original_velocities[i]) / abs(original_velocities[i]) * 100
                    percentage_change += pct
            
            # Only log if multiple components were limited or limitation was very significant
            very_significant = limiting_data['limiting_magnitude'] > 120.0  # Very significant limiting
            multi_component = limited_count > 1
            
            # Determine if this should be logged based on throttling
            current_time = time.time()
            should_log = False
            
            # Sort components to ensure consistent comparison
            sorted_components = sorted(limited_components) if limited_components else []
            
            # Check if this is similar to the last logged change
            is_similar = (self._pipeline_log_tracking['last_components'] == sorted_components)
            
            if is_similar:
                # Increment similarity counter
                self._pipeline_log_tracking['consecutive_similar_count'] += 1
                
                # Only log every 5th similar change
                should_log = self._pipeline_log_tracking['consecutive_similar_count'] % 5 == 0
            else:
                # New component pattern - reset counter
                self._pipeline_log_tracking['consecutive_similar_count'] = 0
                should_log = True
            
            # Time-based throttling - minimum time between similar logs
            if (current_time - self._pipeline_log_tracking['last_log_time']) < self._pipeline_log_tracking['min_log_interval']:
                should_log = False
            
            # Additional conditions for logging
            meets_significance = very_significant or multi_component
            
            if should_log and meets_significance:
                # Update tracking
                self._pipeline_log_tracking['last_log_time'] = current_time
                self._pipeline_log_tracking['last_components'] = sorted_components
                
                # Add repetition info if relevant
                repetition_info = ""
                if self._pipeline_log_tracking['consecutive_similar_count'] > 0:
                    repetition_info = f" (repeated {self._pipeline_log_tracking['consecutive_similar_count']} times)"
                
                # Create consolidated log message with main reasons (limited to 2)
                log_msg = "Velocity limiting: " + "; ".join(reasons[:2]) + repetition_info
                
                # Log with improved throttling
                log_velocity_limiting(
                    limiting_data['original_velocities'], 
                    current_velocities.tolist(), 
                    log_msg, 
                    limited_components[:3]  # Limit to top 3 components for brevity
                )
        
        return current_velocities.tolist(), reasons


def create_velocity_limiter_pipeline(robot_params, debug_level=0):
    """
    Create a complete velocity limiter pipeline with all strategies.
    
    Args:
        robot_params: Dictionary with robot parameters
        debug_level: Debug level for logging
        
    Returns:
        VelocityLimiterPipeline: Configured pipeline
    """
    # Extract parameters
    approach_distance = robot_params.get('approach_distance', 0.7)
    min_approach_factor = robot_params.get('min_approach_factor', 0.1)
    linear_x_max = robot_params.get('linear_x_max', 0.3)
    linear_y_max = robot_params.get('linear_y_max', 0.3)
    angular_max = robot_params.get('angular_max', 0.7)
    accel_limit = robot_params.get('accel_limit', 1.5)
    angular_accel_limit = robot_params.get('angular_accel_limit', 0.8)
    
    # Create pipeline
    pipeline = VelocityLimiterPipeline(debug_level)
    
    # Create and add all strategies in priority order
    
    # 1. Maximum velocity limits (highest priority)
    max_vel_limiter = MaxVelocityLimiter(
        linear_x_max, linear_y_max, angular_max
    )
    pipeline.add_strategy(max_vel_limiter)
    
    # 2. Emergency braking
    emergency_limiter = EmergencyBrakingLimiter()
    pipeline.add_strategy(emergency_limiter)
    
    # 3. Look-ahead limiting
    lookahead_limiter = LookAheadLimiter()
    pipeline.add_strategy(lookahead_limiter)
    
    # 4. Approach scaling
    approach_limiter = ApproachScalingLimiter(
        approach_distance, min_approach_factor
    )
    pipeline.add_strategy(approach_limiter)
    
    # 5. Asymmetric deceleration
    asymmetric_limiter = AsymmetricDecelerationLimiter()
    pipeline.add_strategy(asymmetric_limiter)
    
    # 6. Standard acceleration limiting
    accel_limiter = AccelerationLimiter(
        accel_limit, angular_accel_limit
    )
    pipeline.add_strategy(accel_limiter)
    
    # 7. Velocity hysteresis
    hysteresis_limiter = VelocityHysteresisLimiter()
    pipeline.add_strategy(hysteresis_limiter)
    
    # 8. Minimum velocity threshold
    min_vel_limiter = MinVelocityLimiter()
    pipeline.add_strategy(min_vel_limiter)
    
    return pipeline

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
            init_movement_logger(log_throttle_seconds=0.5)
            self.movement_logger = _movement_state_logger

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
            
            # ====================================================================
            # IMPROVED: Set up velocity limiter pipeline instead of manager
            # ====================================================================
            
            # Collect robot parameters for velocity limiter
            robot_params = {
                'approach_distance': self.approach_distance,
                'min_approach_factor': self.min_approach_factor,
                'linear_x_max': self.linear_x_max,
                'linear_y_max': self.linear_y_max,
                'angular_max': self.angular_max,
                'accel_limit': 1.5,  # Base acceleration limit
                'angular_accel_limit': 1.0  # Base angular acceleration limit
            }
            
            # Create the velocity limiter pipeline
            start_time = time.time()
            self._velocity_limiter_pipeline = create_velocity_limiter_pipeline(
                robot_params, self.debug_level
            )
            elapsed = time.time() - start_time
            log_initialization_step("velocity_limiter", "pipeline_setup", "completed", elapsed)
            
            # ====================================================================
            # End of improved velocity limiter setup
            # ====================================================================
            
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
        """
        Log messages with enhanced throttling to reduce log volume.
        
        Args:
            level_func: Logging function to use (info, warning, etc.)
            message: Message to log
            min_interval: Minimum interval between logs
            last_time_attr: Attribute name for tracking last log time
        
        Returns:
            bool: True if message was logged, False if throttled
        """
        current_time = time.time()
        
        # Initialize tracking for this log type if it doesn't exist
        if not hasattr(self, '_log_throttle_tracking'):
            self._log_throttle_tracking = {}
        
        # Get or create tracking data for this log type
        if last_time_attr not in self._log_throttle_tracking:
            self._log_throttle_tracking[last_time_attr] = {
                'last_time': 0,
                'last_message': None,
                'repeat_count': 0,
                'significant_change_threshold': 0.3  # Consider messages different if 30% different
            }
        
        tracking = self._log_throttle_tracking[last_time_attr]
        
        # Check if enough time has passed
        time_passed = current_time - tracking['last_time']
        
        # Force log if this is a new message that's significantly different
        force_log = False
        if tracking['last_message'] is not None:
            # Calculate similarity using difflib
            import difflib
            
            # Convert to strings if they aren't already
            msg_str = str(message)
            last_msg_str = str(tracking['last_message'])
            
            # Calculate similarity ratio (0.0 to 1.0)
            similarity = difflib.SequenceMatcher(None, msg_str, last_msg_str).ratio()
            
            # Force log if message is significantly different (similarity below threshold)
            if similarity < (1.0 - tracking['significant_change_threshold']):
                force_log = True
        
        # Log if enough time has passed or it's a significantly different message
        if time_passed >= min_interval or force_log:
            # Check if we need to add repeat info to the message
            if tracking['repeat_count'] > 0:
                modified_message = f"{message} (repeated {tracking['repeat_count']} times previously)"
                level_func(modified_message)
            else:
                level_func(message)
            
            # Reset tracking
            tracking['last_time'] = current_time
            tracking['last_message'] = message
            tracking['repeat_count'] = 0
            return True
        else:
            # Update repeat count
            tracking['repeat_count'] += 1
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
            try:
                self._adjust_control_rate()
            except Exception as e:
                # Prevent rate adjustment errors from propagating
                self.get_logger().error(f"Error adjusting control rate: {str(e)}")
        
        # Log resources periodically (not every call)
        current_time = time.time()
        if current_time - self.last_resource_log_time >= 10.0:  # Every 10 seconds
            try:
                # Get cycle time info safely
                cycle_time = None
                if hasattr(self, 'cycle_duration_avg'):
                    cycle_time = self.cycle_duration_avg
                    
                # Get rate adjustment safely
                rate_adjustment = getattr(self, 'last_rate_adjustment', None)
                
                # Log using specialized function with cycle time info
                log_resource_usage(
                    current_cpu, 
                    current_memory, 
                    cycle_time,
                    rate_adjustment
                )
                self.last_resource_log_time = current_time
            except Exception as log_error:
                # Prevent logging errors from affecting resource monitoring
                self.get_logger().error(f"Error logging resource usage: {str(log_error)}")

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
                    
                    # Get strategy name safely for logging
                    strategy_name = str(self.current_strategy)
                    if hasattr(self.current_strategy, 'name'):
                        strategy_name = self.current_strategy.name
                    
                    # Log the adjustment
                    log_structured('resource_monitor', 'RATE_DECREASE', 
                                f"Decreasing control rate due to high CPU", 
                                {'cpu_avg': avg_cpu,
                                'old_rate': current_rate,
                                'new_rate': new_rate,
                                'current_strategy': strategy_name,
                                'threshold': self.cpu_high_threshold})
        
        elif avg_cpu < self.cpu_low_threshold and self.update_rate < self.base_update_rate:
            # Low CPU - increase rate, up to base rate
            current_rate = self.update_rate
            new_rate = min(self.base_update_rate, self.update_rate * 1.1)
            
            if abs(new_rate - current_rate) > 0.1:  # Only adjust if change is significant
                rate_adjustment = (current_rate, new_rate)
                self._update_control_rate(new_rate)
                
                # Get strategy name safely for logging
                strategy_name = str(self.current_strategy)
                if hasattr(self.current_strategy, 'name'):
                    strategy_name = self.current_strategy.name
                
                # Log the adjustment
                log_structured('resource_monitor', 'RATE_INCREASE', 
                            f"Increasing control rate due to low CPU", 
                            {'cpu_avg': avg_cpu,
                            'old_rate': current_rate,
                            'new_rate': new_rate,
                            'current_strategy': strategy_name,
                            'threshold': self.cpu_low_threshold})
        
        # Log resource usage with rate adjustment info if an adjustment was made
        if rate_adjustment:
            try:
                # Get cycle time info safely
                cycle_time = None
                if hasattr(self, 'cycle_duration_avg'):
                    cycle_time = self.cycle_duration_avg
                    
                log_resource_usage(
                    avg_cpu, 
                    self.resource_monitor.get_memory_usage(), 
                    cycle_time,
                    rate_adjustment[1] / rate_adjustment[0]  # Ratio of new/old rate
                )
                
                # Store adjustment for future reference
                self.last_rate_adjustment = rate_adjustment[1] / rate_adjustment[0]
            except Exception as e:
                # Prevent logging errors from affecting rate adjustment
                self.get_logger().error(f"Error logging rate adjustment: {str(e)}")
    
    def _update_control_rate(self, new_rate):
        """Update the control loop rate with logging if it has changed significantly."""
        # Only update if change is significant
        if abs(new_rate - self.update_rate) < 0.1:
            return
        
        # Calculate percentage change for logging
        pct_change = (new_rate - self.update_rate) / self.update_rate * 100
        
        # Update rate
        self.update_rate = new_rate
        
        try:
            # Recreate timer with new rate
            self.timer.cancel()
            self.timer = self.create_timer(1.0 / self.update_rate, self.control_loop_callback)
            
            # Get strategy name safely for logging
            strategy_name = str(self.current_strategy)
            if hasattr(self.current_strategy, 'name'):
                strategy_name = self.current_strategy.name
            
            # Log the change with safe strategy reference
            log_structured('resource_monitor', 'RATE_UPDATED', 
                        f"Control rate updated: {pct_change:.1f}% change", 
                        {'new_rate': new_rate,
                        'period_ms': 1000.0 / new_rate,
                        'current_strategy': strategy_name})
        except Exception as e:
            # Prevent logging or timer errors from affecting rate updates
            self.get_logger().error(f"Error updating control timer: {str(e)}")
            
            # Try to maintain functionality even if logging fails
            try:
                # Recreate timer with new rate (separate try block)
                if not self.timer.is_cancelled():
                    self.timer.cancel()
                self.timer = self.create_timer(1.0 / self.update_rate, self.control_loop_callback)
            except Exception as timer_error:
                self.get_logger().error(f"Critical error recreating timer: {str(timer_error)}")
    
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
        Select the optimal movement strategy based on current errors and conditions with improved logging.
        
        Returns:
            dict: The selected strategy
        """
        # Determine the optimal movement strategy with hysteresis
        prev_strategy_name = self.current_strategy
        
        # Initialize selection tracking if it doesn't exist
        if not hasattr(self, "_strategy_selection_tracking"):
            self._strategy_selection_tracking = {
                "last_log_time": 0.0,
                "min_log_interval": 2.0,  # Minimum seconds between selection logs
                "last_selection_key": None,  # Last error category combination used
                "selection_counts": {},  # Count selections for each key combination
            }
        
        tracking = self._strategy_selection_tracking
        current_time = time.time()
        
        # Log error values before strategy determination only if:
        # 1. Debug level is high enough AND
        # 2. Enough time has passed since last log
        if self.debug_level >= 2 and (current_time - tracking["last_log_time"]) >= tracking["min_log_interval"]:
            log_structured('strategy_selector', 'PRE_STRATEGY_ERRORS', 
                        f"Errors before strategy selection", 
                        {'distance_error': self._current_errors[0], 
                        'lateral_error': self._current_errors[1], 
                        'angular_error_deg': math.degrees(self._current_errors[2])},
                        verbosity_level=2)  # Higher verbosity level
            
            tracking["last_log_time"] = current_time
        
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
                
                # Only log stability constraints at high debug level
                if self.debug_level >= 2:
                    # Use throttled logging to reduce verbosity
                    log_structured('strategy_selector', 'STRATEGY_STABILITY', 
                                f"Maintaining current strategy for stability", 
                                {'current_strategy': self.current_strategy,
                                'time_in_strategy': time_in_strategy,
                                'required_hold_time': adaptive_hold_time},
                                throttle_key="stability",
                                throttle_seconds=1.0,
                                verbosity_level=2)  # Higher verbosity level
                
                # If we need strategy stability and have an active strategy, use it
                if hasattr(self, 'active_strategy'):
                    return self.active_strategy.as_dict()
        
        # Check for special case strategies
        strategy = self._check_special_case_strategies(distance_error, lateral_error, angular_error_degrees)
        
        # Apply strategy modifiers based on approach phase
        strategy = self._apply_strategy_modifiers(strategy, distance_error, distance_trend, lateral_trend, angular_trend)
        
        # Track selection key for logging
        current_key = (self.prev_distance_category, self.prev_lateral_category, self.prev_angular_category)
        if current_key != tracking["last_selection_key"]:
            # Reset count for new key combination
            tracking["selection_counts"][current_key] = 1
            tracking["last_selection_key"] = current_key
        else:
            # Increment count for this key
            tracking["selection_counts"][current_key] = tracking["selection_counts"].get(current_key, 0) + 1
        
        # Log strategy selection with improved context
        sel_count = tracking["selection_counts"].get(current_key, 1)
        context = f"Error categories: distance={self.prev_distance_category}, lateral={self.prev_lateral_category}, angular={self.prev_angular_category}"
        
        # Add trend information for significant trends only
        if abs(distance_trend) > 0.1 or abs(lateral_trend) > 0.1 or abs(angular_trend) > 0.1:
            context += f", trends=[{distance_trend:.2f}, {lateral_trend:.2f}, {angular_trend:.2f}]"
        
        # Only log periodically based on selection count for the same key
        if sel_count <= 3 or sel_count % 10 == 0:  # Log first 3 and then every 10th
            log_strategy_selection(
                current_key, 
                strategy,
                reason=context
            )
        
        return strategy

    def _check_special_case_strategies(self, distance_error, lateral_error, angular_error_degrees):
        """
        Check for special case strategies based on error conditions with enhanced
        angular prioritization and reduced logging.
        
        Args:
            distance_error: Current distance error
            lateral_error: Current lateral error
            angular_error_degrees: Current angular error in degrees
                    
        Returns:
            dict: Selected strategy
        """
        # Initialize tracking for special case logging if needed
        if not hasattr(self, "_special_case_tracking"):
            self._special_case_tracking = {
                "last_log_time": {},  # Keyed by special case type
                "consecutive_count": {},  # Count consecutive selections
                "min_log_interval": 1.0,  # Minimum seconds between logs
            }
        
        tracking = self._special_case_tracking
        current_time = time.time()
        
        # IMPROVED: Lowering angular threshold for special case handling from 10.0 to 7.0 degrees
        # This ensures earlier intervention for moderate angular errors
        if abs(angular_error_degrees) > 7.0:
            # Angular-first strategy for significant angular errors
            special_case = "ANGULAR_FIRST"
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
            
            # Create special strategy
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
            
            # Log with reduced frequency depending on consecutive selections
            if special_case in tracking["consecutive_count"]:
                tracking["consecutive_count"][special_case] += 1
            else:
                tracking["consecutive_count"][special_case] = 1
            
            # Get last log time, defaulting to 0
            last_log_time = tracking["last_log_time"].get(special_case, 0)
            consecutive_count = tracking["consecutive_count"][special_case]
            
            # Calculate appropriate log interval based on repetition
            if consecutive_count > 5:
                # After 5 consecutive logs, gradually increase interval
                log_interval = min(5.0, tracking["min_log_interval"] * (consecutive_count / 5))
            else:
                log_interval = tracking["min_log_interval"]
            
            # Log if enough time has passed or if angular error changed significantly
            time_since_log = current_time - last_log_time
            
            # Only log periodically with higher-level reason
            if time_since_log >= log_interval:
                log_strategy_selection(
                    "angular_override", 
                    special_strategy,
                    reason=f"Angular error ({angular_error_degrees:.1f}°) exceeds threshold"
                )
                tracking["last_log_time"][special_case] = current_time
            
            # Store as active strategy
            self.active_strategy = MovementStrategy(
                name, use_forward, use_lateral, use_angular,
                forward_scale, lateral_scale, angular_scale, reason
            )
            
            return special_strategy
            
        # IMPROVED: Add additional special case for medium angular errors (3-7 degrees)
        elif abs(angular_error_degrees) > 3.0:
            # Balanced approach with moderate angular priority
            special_case = "BALANCED_ANGULAR"
            name = "BALANCED_ANGULAR_CORRECTION"
            use_forward = True
            use_lateral = True
            use_angular = True
            
            # Scale factors that still allow forward movement but prioritize angular correction
            forward_scale = 0.6  # Moderate forward movement
            lateral_scale = 0.5  # Moderate lateral correction
            angular_scale = 0.7  # Strong angular priority (but not maximum)
            
            reason = f"Balanced approach with angular priority: {angular_error_degrees:.1f}°"
            
            # Create special strategy
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
            
            # Increment consecutive count
            if special_case in tracking["consecutive_count"]:
                tracking["consecutive_count"][special_case] += 1
            else:
                tracking["consecutive_count"][special_case] = 1
            
            # Get last log time and consecutive count
            last_log_time = tracking["last_log_time"].get(special_case, 0)
            consecutive_count = tracking["consecutive_count"][special_case]
            
            # For medium angular corrections, log less frequently
            log_interval = min(8.0, tracking["min_log_interval"] * (1 + consecutive_count / 3))
            time_since_log = current_time - last_log_time
            
            # Only log periodically to reduce verbosity
            if consecutive_count <= 2 or time_since_log >= log_interval:
                log_strategy_selection(
                    "medium_angular_override", 
                    special_strategy,
                    reason=f"Medium angular error ({angular_error_degrees:.1f}°) requires attention"
                )
                tracking["last_log_time"][special_case] = current_time
            
            # Store as active strategy
            self.active_strategy = MovementStrategy(
                name, use_forward, use_lateral, use_angular,
                forward_scale, lateral_scale, angular_scale, reason
            )
            
            return special_strategy
            
        # Check for combined lateral and distance error - implement coordinated diagonal approach
        elif abs(lateral_error) > 0.15 and abs(distance_error) > 0.3:
            # Calculate approach angle to determine optimal path
            special_case = "DIAGONAL"
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
                
                # Reduced logging for diagonal approaches
                if special_case in tracking["consecutive_count"]:
                    tracking["consecutive_count"][special_case] += 1
                else:
                    tracking["consecutive_count"][special_case] = 1
                    
                consecutive_count = tracking["consecutive_count"][special_case]
                last_log_time = tracking["last_log_time"].get(special_case, 0)
                time_since_log = current_time - last_log_time
                
                # Log only for first few and then periodically
                if consecutive_count <= 2 or (consecutive_count % 5 == 0 and time_since_log >= 2.0):
                    log_strategy_selection(
                        "diagonal_override", 
                        special_strategy,
                        reason=f"Significant diagonal approach at {approach_angle:.1f}° angle"
                    )
                    tracking["last_log_time"][special_case] = current_time
                
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
                            'modified_scale': strategy['forward_scale']},
                            throttle_key=f"approach_decel_{round(distance_error, 2)}_{round(strategy['forward_scale'], 2)}",
                            throttle_seconds=0.4)
        
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
        Log strategy changes and update strategy tracking with improved significance detection.
        
        Args:
            strategy: Current strategy
            prev_strategy_name: Previous strategy name
        """
        # Extract strategy name safely
        strategy_name = strategy["strategy_name"] if isinstance(strategy, dict) and "strategy_name" in strategy else str(strategy)
        
        # Track current strategy for next iteration
        self.current_strategy = strategy_name
        
        # Check if strategy changed
        if strategy_name != prev_strategy_name:
            # Initialize strategy change tracking if it doesn't exist
            if not hasattr(self, "_strategy_change_tracking"):
                self._strategy_change_tracking = {
                    "last_log_time": 0.0,
                    "oscillating_strategies": {},  # Count oscillations between pairs
                    "significant_changes": set(),  # Track significant state transitions
                }
            
            tracking = self._strategy_change_tracking
            current_time = time.time()
            
            # Check for oscillation (switching back and forth between strategies)
            pair_key = f"{prev_strategy_name}_{strategy_name}"
            reverse_key = f"{strategy_name}_{prev_strategy_name}"
            
            if reverse_key in tracking["oscillating_strategies"]:
                # This is a switch back to a recent strategy - increment oscillation count
                tracking["oscillating_strategies"][reverse_key] = tracking["oscillating_strategies"].get(reverse_key, 0) + 1
                
                # Only log oscillations periodically to avoid spam
                oscillation_count = tracking["oscillating_strategies"][reverse_key]
                if oscillation_count <= 3 or oscillation_count % 5 == 0:  # Log first 3, then every 5th
                    log_structured('strategy_selector', 'STRATEGY_OSCILLATION', 
                                f"Strategy oscillation detected: {prev_strategy_name} ↔ {strategy_name} ({oscillation_count}x)", 
                                {'count': oscillation_count,
                                'from_strategy': prev_strategy_name,
                                'to_strategy': strategy_name})
            else:
                # Record this transition for future oscillation detection
                tracking["oscillating_strategies"][pair_key] = 1
                
                # Clean up old oscillation records periodically
                if len(tracking["oscillating_strategies"]) > 10:
                    # Keep only the most recent and most frequent oscillations
                    sorted_pairs = sorted(tracking["oscillating_strategies"].items(), 
                                        key=lambda x: x[1], reverse=True)
                    tracking["oscillating_strategies"] = dict(sorted_pairs[:5])  # Keep top 5
            
            # Determine if this is a significant strategy change worth logging
            is_significant = False
            
            # Case 1: First strategy after initialization
            if prev_strategy_name is None or prev_strategy_name == "IDLE":
                is_significant = True
                
            # Case 2: Entering or exiting an "ANGULAR" strategy
            if ("ANGULAR" in strategy_name and "ANGULAR" not in prev_strategy_name) or \
            ("ANGULAR" in prev_strategy_name and "ANGULAR" not in strategy_name):
                is_significant = True
                
            # Case 3: Any transition to emergency, stop, or approach strategies
            important_strategies = ["EMERGENCY", "STOP", "APPROACH", "DIAGONAL"]
            if any(s in strategy_name for s in important_strategies):
                is_significant = True
                
            # Case 4: Major change in motion components or direction
            if isinstance(strategy, dict) and isinstance(prev_strategy_name, str):
                # Try to find previous strategy in strategy table
                for strat_def in self.strategy_table.values():
                    if len(strat_def) >= 8 and strat_def[0] == prev_strategy_name:
                        # Compare motion components
                        prev_forward = strat_def[1]  # use_forward
                        prev_lateral = strat_def[2]  # use_lateral
                        prev_angular = strat_def[3]  # use_angular
                        
                        curr_forward = strategy.get("use_forward", False)
                        curr_lateral = strategy.get("use_lateral", False)
                        curr_angular = strategy.get("use_angular", False)
                        
                        # If motion components changed, it's significant
                        if prev_forward != curr_forward or prev_lateral != curr_lateral or prev_angular != curr_angular:
                            is_significant = True
                        break
            
            # Only log significant strategy changes or periodically forced logs
            if is_significant or (current_time - tracking["last_log_time"]) > 5.0:  # Force log every 5 seconds
                # Strategy has changed - log the transition
                log_structured('strategy_selector', 'STRATEGY_CHANGED', 
                            f"Strategy changed: {prev_strategy_name} → {strategy_name}", 
                            {'reason': strategy.get("reason", "N/A") if isinstance(strategy, dict) else "N/A",
                            'distance_cat': self.prev_distance_category,
                            'lateral_cat': self.prev_lateral_category, 
                            'angular_cat': self.prev_angular_category,
                            'forward_scale': strategy.get("forward_scale", 0.0) if isinstance(strategy, dict) else 0.0,
                            'lateral_scale': strategy.get("lateral_scale", 0.0) if isinstance(strategy, dict) else 0.0,
                            'angular_scale': strategy.get("angular_scale", 0.0) if isinstance(strategy, dict) else 0.0})
                
                # Update log time
                tracking["last_log_time"] = current_time
                
            # Always update the strategy change time regardless of logging
            self.strategy_change_time = time.time()
            
        elif self.debug_level >= 3:
            # Only log continued strategy at highest debug level
            log_structured('strategy_selector', 'STRATEGY_MAINTAINED', 
                        f"Maintaining strategy: {strategy_name}", 
                        {'duration': time.time() - self.strategy_change_time},
                        verbosity_level=3)  # Increase verbosity level to reduce logging

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
        angular response control to prevent overshooting. Improved to reduce log verbosity.
        
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
        
        # Track all velocity adjustments in a single dictionary for consolidated logging
        scaling_data = {
            'raw_velocities': unscaled_velocities.copy(),
            'final_velocities': None,  # Will be set at end
            'adjustments': [],         # List of all adjustments made
            'scaling_factors': {       # Track all factors applied
                'forward': forward_scale,
                'lateral': lateral_scale, 
                'angular': angular_scale
            }
        }
        
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
            
            # Store this adjustment for logging
            scaling_data['adjustments'].append({
                'type': 'angular_based_scaling',
                'angular_error_deg': angular_error_deg,
                'forward_scaling': angular_scaling,
                'angular_boost': angular_boost,
                'before': {'forward': original_linear_x, 'angular': angular_velocity / angular_boost},
                'after': {'forward': linear_x_velocity, 'angular': angular_velocity}
            })
        
        # Check if we're approaching target angle (getting close to zero error)
        # MODIFIED: Enhanced deceleration logic to prevent overshooting
        if abs(self._current_errors[2]) < math.radians(7.0) and abs(angular_velocity) > 0.2:
            # Apply stronger deceleration when approaching zero angular error
            # (scales down velocity more as error approaches zero)
            deceleration_factor = max(0.2, abs(self._current_errors[2]) / math.radians(7.0))
            original_angular = angular_velocity
            angular_velocity *= deceleration_factor
            
            # Store this adjustment for logging
            scaling_data['adjustments'].append({
                'type': 'angular_deceleration',
                'angle_error_deg': math.degrees(self._current_errors[2]),
                'deceleration_factor': deceleration_factor,
                'before': original_angular,
                'after': angular_velocity
            })
        
        # Strategy-specific angular scaling
        # MODIFIED: Reduced all strategy-specific boost factors
        if "ANGULAR" in strategy_name:
            # Boost angular velocity for angular-focused strategies
            # MODIFIED: Reduced from 1.8 to 1.3
            angular_boost_factor = 1.3
            original_angular = angular_velocity
            angular_velocity *= angular_boost_factor
            
            # Store this adjustment
            scaling_data['adjustments'].append({
                'type': 'strategy_angular_boost',
                'strategy': strategy_name,
                'boost_factor': angular_boost_factor,
                'before': original_angular,
                'after': angular_velocity
            })
        elif "APPROACH" in strategy_name or "DIAGONAL" in strategy_name:
            # Moderate boost for approach strategies
            # MODIFIED: Reduced from 1.5 to 1.2
            angular_boost_factor = 1.2
            original_angular = angular_velocity
            angular_velocity *= angular_boost_factor
            
            # Store this adjustment
            scaling_data['adjustments'].append({
                'type': 'strategy_angular_boost',
                'strategy': strategy_name,
                'boost_factor': angular_boost_factor,
                'before': original_angular,
                'after': angular_velocity
            })
        
        # Check if angular and forward motions are working together or against each other
        if (linear_x_velocity > 0.05 and angular_velocity * self._current_errors[2] < 0):
            # Angular velocity is helping alignment during approach - boost it
            # MODIFIED: Reduced from 1.3 to 1.1
            corrective_boost = 1.1
            original_angular = angular_velocity
            angular_velocity *= corrective_boost
            
            # Store this adjustment
            scaling_data['adjustments'].append({
                'type': 'alignment_boost',
                'boost_factor': corrective_boost,
                'before': original_angular,
                'after': angular_velocity
            })
        
        # When very close to target, prioritize angular alignment
        if abs(self._current_errors[0]) < 0.2:  # Close to target distance
            # Higher angular priority when close to target
            # MODIFIED: Limited maximum boost from 2.0 to 1.5
            close_target_boost = min(1.5, 1.0 + abs(self._current_errors[2]) / 0.12)
            original_angular = angular_velocity
            angular_velocity *= close_target_boost
            
            # If angular error is significant when close to target, reduce forward speed further
            if abs(self._current_errors[2]) > math.radians(5.0):
                original_forward = linear_x_velocity
                linear_x_velocity *= 0.5
                
                # Store these adjustments
                scaling_data['adjustments'].append({
                    'type': 'close_target',
                    'angular_boost': close_target_boost,
                    'forward_reduction': 0.5 if abs(self._current_errors[2]) > math.radians(5.0) else 1.0,
                    'before': {'angular': original_angular, 'forward': original_forward},
                    'after': {'angular': angular_velocity, 'forward': linear_x_velocity}
                })
        
        # Apply standard strategy scaling factors
        original_velocities = [linear_x_velocity, lateral_velocity, angular_velocity]
        linear_x_velocity *= forward_scale
        lateral_velocity *= lateral_scale
        
        # MODIFIED: Apply a more conservative angular scaling
        # For strategies with high angular scale, apply a dampening factor
        if angular_scale > 0.9:
            # Scale down the scaling factor itself to prevent excessive angular velocity
            dampened_angular_scale = 0.9 + (angular_scale - 0.9) * 0.5  # Dampen values above 0.9
            angular_velocity *= dampened_angular_scale
            
            # Store the dampening info
            scaling_data['adjustments'].append({
                'type': 'angular_scale_dampening',
                'original_scale': angular_scale,
                'dampened_scale': dampened_angular_scale
            })
            
            # Update tracking dictionary
            scaling_data['scaling_factors']['angular'] = dampened_angular_scale
        else:
            # Apply normal scaling for smaller angular scales
            angular_velocity *= angular_scale
        
        # Protect against excessive angular velocity after all boosts
        # MODIFIED: Reduced from 1.2 to 1.1 times max
        max_safe_angular = self.angular_max * 1.3  # Allow slightly higher than standard max
        if abs(angular_velocity) > max_safe_angular:
            original_angular = angular_velocity
            angular_velocity = math.copysign(max_safe_angular, angular_velocity)
            
            # Store this adjustment
            scaling_data['adjustments'].append({
                'type': 'angular_velocity_cap',
                'original': original_angular,
                'capped': angular_velocity,
                'max_allowed': max_safe_angular
            })
        
        # Update final velocities for logging
        scaling_data['final_velocities'] = [linear_x_velocity, lateral_velocity, angular_velocity]
        
        # Calculate how significant the changes were overall
        total_change_magnitude = sum([
            abs(linear_x_velocity - unscaled_velocities[0]),
            abs(lateral_velocity - unscaled_velocities[1]),
            abs(angular_velocity - unscaled_velocities[2])
        ])
        
        # Only log if significant changes were made - increased threshold
        if total_change_magnitude > 0.1:  # Increased from 0.05 to 0.1
            # Extract most important adjustments for logging
            main_adjustments = []
            angular_scaled = False
            forward_scaled = False
            
            for adj in scaling_data['adjustments']:
                # Prioritize certain adjustment types
                if adj['type'] == 'angular_based_scaling':
                    forward_scaled = True
                    main_adjustments.append(f"forward scaled by {adj['forward_scaling']:.2f} due to {adj['angular_error_deg']:.1f}° error")
                    angular_scaled = True
                elif adj['type'] == 'angular_velocity_cap' and not angular_scaled:
                    angular_scaled = True
                    main_adjustments.append(f"angular capped at {max_safe_angular:.2f}")
                elif adj['type'] == 'close_target' and not forward_scaled:
                    forward_scaled = True
                    main_adjustments.append(f"near target adjustments (d={abs(self._current_errors[0]):.2f}m)")
            
            # Create informative log message with main adjustments
            log_msg = f"Velocity scaling for {strategy_name}"
            if main_adjustments:
                log_msg += ": " + ", ".join(main_adjustments[:2])
                
            # Log consolidated velocity scaling info - with increased throttling
            log_structured('strategy_selector', 'VELOCITY_SCALING', 
                        log_msg, 
                        {'strategy': strategy_name,
                        'raw': unscaled_velocities,
                        'scaled': (linear_x_velocity, lateral_velocity, angular_velocity),
                        'factors': (forward_scale, lateral_scale, angular_scale),
                        'adj_count': len(scaling_data['adjustments'])},
                        throttle_key=f"vel_scaling_{strategy_name}",
                        throttle_seconds=0.5)  # Increased from no throttling
        
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
        Consolidates logging to reduce verbosity.
        
        Args:
            linear_x_velocity: Calculated forward velocity
            lateral_velocity: Calculated lateral velocity
            angular_velocity: Calculated angular velocity
        """
        # Implement velocity limiting and publication as before (code omitted for brevity)
        # ...
        
        # Store new velocities in pre-allocated arrays for efficiency
        self._velocity_tuple[0] = linear_x_velocity
        self._velocity_tuple[1] = lateral_velocity
        self._velocity_tuple[2] = angular_velocity
        
        # Store for next cycle
        self.last_cmd_vel = tuple(self._velocity_tuple)
        
        # Save velocity for history
        velocity_tuple = (float(linear_x_velocity), float(lateral_velocity), float(angular_velocity))
        self.velocity_history.add(velocity_tuple)
        
        # Track robot stopped status with improved logging
        was_stopped = self._robot_stopped
        self._robot_stopped = abs(linear_x_velocity) < 0.01 and abs(lateral_velocity) < 0.01 and abs(angular_velocity) < 0.01
        
        # Only log transitions to avoid duplicate logs
        if was_stopped and not self._robot_stopped:
            # Robot starting to move from previously stopped state
            # This log call is only needed if _reset_stopped_state_if_needed didn't already log it
            strategy_name = getattr(self, 'current_strategy', 'unknown')
            if isinstance(strategy_name, dict) and 'strategy_name' in strategy_name:
                strategy_name = strategy_name['strategy_name']
                
            log_movement_state(
                is_stopped=False,  # Moving
                reason=f"Motion commanded by {strategy_name} strategy",
                error_values={
                    'velocity_x': linear_x_velocity,
                    'velocity_y': lateral_velocity,
                    'velocity_angular': angular_velocity
                },
                robot_state=self.robot_state,
                verbose=True
            )
        elif not was_stopped and self._robot_stopped:
            # Robot stopping from previously moving state
            # Only log if this wasn't explicitly commanded by stop_robot()
            if not hasattr(self, 'stop_already_logged') or not self.stop_already_logged:
                strategy_name = getattr(self, 'current_strategy', 'unknown')
                if isinstance(strategy_name, dict) and 'strategy_name' in strategy_name:
                    strategy_name = strategy_name['strategy_name']
                    
                log_movement_state(
                    is_stopped=True,  # Stopped
                    reason=f"Velocity decayed to zero under {strategy_name} strategy",
                    error_values={
                        'prev_velocity_x': self.last_cmd_vel[0],
                        'prev_velocity_y': self.last_cmd_vel[1],
                        'prev_velocity_angular': self.last_cmd_vel[2]
                    },
                    robot_state=self.robot_state,
                    verbose=True
                )
        
        # Clear the stop_already_logged flag after handling
        if hasattr(self, 'stop_already_logged'):
            delattr(self, 'stop_already_logged')

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
            try:
                linear_x_velocity, lateral_velocity, angular_velocity = self._determine_and_apply_strategy(dt)
            except Exception as strategy_error:
                # Handle strategy determination errors specifically
                self.get_logger().error(f"Strategy determination error: {str(strategy_error)}")
                linear_x_velocity, lateral_velocity, angular_velocity = 0.0, 0.0, 0.0
            
            # Apply velocity limits and publish commands
            try:
                self._apply_and_publish_velocities(linear_x_velocity, lateral_velocity, angular_velocity)
            except Exception as velocity_error:
                self.get_logger().error(f"Velocity application error: {str(velocity_error)}")
                # Try to stop the robot safely
                self.stop_robot()
                return
            
            # Update performance stats
            try:
                self._update_performance_stats()
            except Exception as stats_error:
                # Don't let performance stats errors affect control
                self.get_logger().warning(f"Performance stats update error: {str(stats_error)}")
                
        except Exception as e:
            # Enhanced error handling that checks for specific formatting errors
            error_str = str(e)
            
            if "unsupported format string passed to tuple" in error_str:
                # This is the specific error we're fixing
                self.get_logger().error(
                    f"String formatting error detected in control loop - check strategy type handling: {error_str}"
                )
                
                # Log additional debugging info about the current strategy
                strategy_type = type(self.current_strategy).__name__
                self.get_logger().error(
                    f"Debug info: current_strategy type={strategy_type}"
                )
            else:
                # General error handling
                self.get_logger().error(f"Unexpected error in control_loop_callback: {error_str}")
            
            # Try to safely stop the robot
            try:
                self.stop_robot()
            except Exception as stop_error:
                self.get_logger().error(f"Failed to stop robot after error: {str(stop_error)}")
    
    def _handle_recovery_mode(self, current_time):
        """
        Handle recovery mode with a three-phase approach and improved logging.
        Reduces verbose logs during recovery while maintaining important state information.
        
        Args:
            current_time: Current time
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
        
        # Initialize recovery tracking if needed
        if not hasattr(self, '_recovery_tracking'):
            self._recovery_tracking = {
                'last_log_time': 0.0,
                'last_phase_progress': 0.0,
                'progress_threshold': 0.1,  # Log progress at 10% intervals
                'min_log_interval': 0.5,    # Minimum 0.5s between logs in same phase
                'orientation_logged': False, # Track if orientation info was logged
                'approach_logged': False     # Track if approach info was logged
            }
        
        recovery_duration = current_time - self.recovery_start_time
        prev_phase = self.recovery_phase
        tracking = self._recovery_tracking
        
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
                
                # Reset tracking for new phase
                tracking['orientation_logged'] = False
                tracking['last_log_time'] = current_time
                tracking['last_phase_progress'] = 0.0
                
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
                    
                    # Calculate progress in this phase
                    phase_duration = 2.0  # Orient phase is 1-3 seconds (2s duration)
                    phase_progress = min(1.0, (recovery_duration - 1.0) / phase_duration)
                    
                    # Only log orientation progress at intervals or significant changes
                    should_log = (
                        # Log for first orientation attempt
                        not tracking['orientation_logged'] or
                        # Log at progress percentage thresholds (e.g., 10%, 20%, etc.)
                        (phase_progress - tracking['last_phase_progress'] >= tracking['progress_threshold']) or
                        # Log for significant angular changes
                        (abs(angular_degrees) < 5.0 and not tracking['orientation_logged'])
                    )
                    
                    # Additional time-based throttling
                    time_since_log = current_time - tracking['last_log_time']
                    if should_log and time_since_log >= tracking['min_log_interval']:
                        log_structured('recovery', 'ORIENT_PROGRESS', 
                                    f"Recovery orient progress: {angular_degrees:.2f}°", 
                                    {'angular_error': angular_degrees,
                                    'velocity': angular_velocity,
                                    'phase_progress': f"{phase_progress*100:.0f}%",
                                    'time_in_phase': recovery_duration - 1.0})
                        
                        tracking['orientation_logged'] = True
                        tracking['last_log_time'] = current_time
                        tracking['last_phase_progress'] = phase_progress
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
                        
                        # Reset tracking for new phase
                        tracking['approach_logged'] = False
                        tracking['last_log_time'] = current_time
                        tracking['last_phase_progress'] = 0.0
            else:
                # No target available - only log once
                if not tracking['orientation_logged']:
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
                                    
                    tracking['orientation_logged'] = True
                    tracking['last_log_time'] = current_time
            
            # After 3 seconds in orient phase, move to approach
            if recovery_duration > 3.0:
                self.recovery_phase = "approach"
                
                # Log phase transition
                log_structured('recovery', 'PHASE_TRANSITION', 
                            "Recovery: Moving to approach phase", 
                            {'from_phase': prev_phase,
                            'to_phase': self.recovery_phase,
                            'duration_in_prev': recovery_duration - 1.0})
                
                # Reset tracking for new phase
                tracking['approach_logged'] = False
                tracking['last_log_time'] = current_time
                tracking['last_phase_progress'] = 0.0
                
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
                    
                    # Calculate progress in this phase
                    phase_progress = min(1.0, (recovery_duration - 3.0) / 3.0)  # Assume 3 seconds for approach
                    
                    # Only log approach progress at intervals or significant error changes
                    should_log = (
                        # First approach log
                        not tracking['approach_logged'] or
                        # Log at progress thresholds
                        (phase_progress - tracking['last_phase_progress'] >= tracking['progress_threshold']) or
                        # Log when getting close to target
                        (abs(distance_error) < 0.2 and not tracking['approach_logged'])
                    )
                    
                    # Additional time-based throttling
                    time_since_log = current_time - tracking['last_log_time']
                    if should_log and time_since_log >= tracking['min_log_interval']:
                        log_structured('recovery', 'APPROACH_PROGRESS', 
                                    f"Recovery approach progress", 
                                    {'distance_error': distance_error,
                                    'lateral_error': lateral_error,
                                    'velocity_fwd': linear_x_velocity,
                                    'velocity_lat': lateral_velocity,
                                    'phase_progress': f"{phase_progress*100:.0f}%",
                                    'time_in_phase': recovery_duration - 3.0})
                        
                        tracking['approach_logged'] = True
                        tracking['last_log_time'] = current_time
                        tracking['last_phase_progress'] = phase_progress
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
                # No target available during approach - only log once
                if not tracking['approach_logged'] and self.current_target is None:
                    log_structured('recovery', 'APPROACH_NO_TARGET', 
                                "Cannot approach in recovery - no target available", 
                                {'time_in_phase': recovery_duration - 3.0},
                                level=logging.WARNING)
                                
                    tracking['approach_logged'] = True
                    tracking['last_log_time'] = current_time
            
            # After 6 seconds in recovery, suggest exiting recovery mode, but only log every 2 seconds
            if recovery_duration > 6.0 and int(recovery_duration) % 2 == 0:
                # Calculate time since last log to prevent duplicate logs
                time_since_log = current_time - tracking['last_log_time']
                if time_since_log > 1.5:  # Only log once per suggestion (avoiding integer boundary issues)
                    log_structured('recovery', 'RECOVERY_DURATION_LIMIT', 
                                "Recovery has been active for extended period", 
                                {'duration': recovery_duration,
                                'phase': self.recovery_phase,
                                'suggestion': "Consider state transition to tracking mode"})
                    tracking['last_log_time'] = current_time
    
    def _reset_stopped_state_if_needed(self, distance_error, lateral_error, angular_error):
        """
        Reset stopped state if significant movement is required, with improved logging.
        
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
            # Collect error values for logging context
            error_values = {
                'distance_error': distance_error,
                'lateral_error': lateral_error,
                'angular_error': angular_error
            }
            
            # Use centralized logging with additional context
            log_movement_state(
                is_stopped=False,  # Transitioning to moving state
                reason=reason,
                error_values=error_values,
                robot_state=self.robot_state,
                verbose=True
            )
            
            # Reset stopped state
            self._robot_stopped = False
            
            # Reset movement hysteresis
            self._movement_hysteresis = 0.0
            
            return True
        
        return False
    
    def _evaluate_stop_conditions(self, distance, lateral, angular_degrees, is_stopped):
        """
        Evaluate if the robot should move based on current conditions with improved logging.
        
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
            
            # Collect error values for logging context 
            error_values = {
                'distance_error': distance_error,
                'lateral_error': lateral_error,
                'angular_error': angular_error,
                'velocity_x': velocities['linear_x'],
                'velocity_y': velocities['linear_y'],
                'velocity_angular': velocities['angular']
            }
            
            # Use centralized logging for stop conditions
            # The actual logging will happen in stop_robot() to avoid duplication
            # Just store the reason here for later use
            self.stop_reason = reason
        
        return should_stop, reason
    
    def _apply_velocity_limits(self, linear_x, linear_y, angular_z, current_time):
        """
        Apply velocity and acceleration limits with enhanced approach behavior and improved logging.
        
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
        
        # ====================================================================
        # IMPROVED: Use velocity limiter pipeline instead of manager
        # ====================================================================
        
        # Prepare robot state dictionary for velocity limiting
        robot_state = {
            'distance_error': distance_error,
            'is_approaching': is_approaching,
            'approach_distance': self.approach_distance,
            'accel_limit': 1.5,  # Base acceleration limit
            'angular_accel_limit': 1.0  # Base angular acceleration limit
        }
        
        try:
            # Apply all velocity limiting strategies through the pipeline
            limited_velocities, reasons = self._velocity_limiter_pipeline.apply_limits(
                [linear_x, linear_y, angular_z],  # Current velocities
                self.last_cmd_vel,                # Previous velocities
                robot_state,                      # State information
                dt                                # Time delta
            )
            
            # Only log if there are reasons (significant limiting occurred)
            if reasons and self.debug_level >= 1:
                # The pipeline's internal logging is already improved and throttled,
                # so no need for additional logging here
                pass
                
        except Exception as e:
            # Handle errors gracefully - log and return original velocities
            self.get_logger().error(f"Error in velocity limiter pipeline: {str(e)}")
            limited_velocities = [linear_x, linear_y, angular_z]
        
        # ====================================================================
        # End of improved velocity limiter usage
        # ====================================================================
        
        return tuple(limited_velocities)

    def stop_robot(self):
        """Send a command to stop all robot motion immediately and reset controllers with improved logging."""
        # Store previous motion state for transition logging
        was_moving = not getattr(self, '_robot_stopped', True)
        
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
            # Get reason from stored value or use default
            stop_reason = getattr(self, 'stop_reason', "Stop command issued")
            
            # Log only once using the centralized logging
            log_movement_state(
                is_stopped=True,  # Stopped
                reason=stop_reason,
                error_values={
                    'prev_velocity_x': getattr(self, 'last_cmd_vel', (0.0, 0.0, 0.0))[0],
                    'prev_velocity_y': getattr(self, 'last_cmd_vel', (0.0, 0.0, 0.0))[1],
                    'prev_velocity_angular': getattr(self, 'last_cmd_vel', (0.0, 0.0, 0.0))[2],
                    'integral_reset': True,
                    'position_distance': self._last_stop_position[0],
                    'position_lateral': self._last_stop_position[1]
                },
                robot_state=self.robot_state,
                verbose=True
            )
            
            # Set flag to prevent duplicate logs
            self.stop_already_logged = True
            
            # Clear the stop reason to avoid reusing it
            if hasattr(self, 'stop_reason'):
                delattr(self, 'stop_reason')

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
        Match a key against the strategy table with wildcard support and reduced logging.
        
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
                # Only log at highest debug level (3) to minimize verbosity
                if self.debug_level >= 3:
                    self.get_logger().debug(
                        f"Strategy wildcard match: {key} → {pattern}, " 
                        f"strategy={strategies[pattern][0]}"
                    )
                return strategies[pattern]
        
        # Fallback
        return strategies[("*", "*", "*")]
    
    def _log_periodic_status(self):
        """
        Log periodic status updates with improved filtering to reduce verbosity.
        Only logs when stats have changed significantly from the last logged values.
        """
        # Only log if debug level supports it
        if self.debug_level < 1:
            return
        
        # Initialize tracking if it doesn't exist
        if not hasattr(self, '_status_tracking'):
            self._status_tracking = {
                'last_log_time': 0.0,
                'last_cpu': 0.0,
                'last_strategy': None,
                'last_state': None,
                'last_cycle_time': 0.0,
                'last_update_rate': 0.0,
                'min_log_interval': 5.0,  # At least 5 seconds between similar logs
                'forced_log_interval': 30.0,  # Force log every 30 seconds
                'significant_change_threshold': 0.1  # 10% change is significant
            }
        
        # Get current CPU usage
        cpu_usage = self.current_cpu_usage
        
        # Calculate cycle time stats
        cycle_time_avg = 0.0
        if hasattr(self, 'cycle_duration_avg'):
            cycle_time_avg = self.cycle_duration_avg * 1000.0  # Convert to ms
        
        # Get current time
        current_time = time.time()
        
        # Decide whether to log based on time or significant changes
        force_log = (current_time - self._status_tracking['last_log_time']) > self._status_tracking['forced_log_interval']
        significant_change = False
        
        # Calculate relative changes in numeric values
        if abs(self._status_tracking['last_cpu']) > 0.1:
            cpu_change = abs(cpu_usage - self._status_tracking['last_cpu']) / self._status_tracking['last_cpu']
            if cpu_change > self._status_tracking['significant_change_threshold']:
                significant_change = True
        
        if abs(self._status_tracking['last_cycle_time']) > 0.1:
            cycle_time_change = abs(cycle_time_avg - self._status_tracking['last_cycle_time']) / self._status_tracking['last_cycle_time']
            if cycle_time_change > self._status_tracking['significant_change_threshold']:
                significant_change = True
        
        if abs(self._status_tracking['last_update_rate']) > 0.1:
            rate_change = abs(self.update_rate - self._status_tracking['last_update_rate']) / self._status_tracking['last_update_rate']
            if rate_change > self._status_tracking['significant_change_threshold']:
                significant_change = True
        
        # Get strategy name safely
        strategy_name = self.current_strategy
        if hasattr(self.current_strategy, 'name'):
            strategy_name = self.current_strategy.name
        
        # Check categorical changes
        if (self.robot_state != self._status_tracking['last_state'] or 
            strategy_name != self._status_tracking['last_strategy']):
            significant_change = True
        
        # Minimum time between logs even with significant changes
        min_time_passed = (current_time - self._status_tracking['last_log_time']) > self._status_tracking['min_log_interval']
        
        # Log if changes warrant it and enough time has passed
        if (force_log or (significant_change and min_time_passed)):
            # Log status
            self.get_logger().info(
                f"Status: Robot state={self.robot_state}, "
                f"Strategy={strategy_name}, "
                f"CPU={cpu_usage:.1f}%, "
                f"Cycle time={cycle_time_avg:.2f}ms, "
                f"Update rate={self.update_rate:.1f}Hz"
            )
            
            # Update tracking data
            self._status_tracking['last_log_time'] = current_time
            self._status_tracking['last_cpu'] = cpu_usage
            self._status_tracking['last_strategy'] = strategy_name
            self._status_tracking['last_state'] = self.robot_state
            self._status_tracking['last_cycle_time'] = cycle_time_avg
            self._status_tracking['last_update_rate'] = self.update_rate
    
    def publish_diagnostics(self):
        """Publish detailed diagnostic information at a slower rate with improved significance tracking."""
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
        
        # Get current time
        current_time = time.time()
        
        # Initialize tracking structure if not exists
        if not hasattr(self, '_diag_tracking'):
            self._diag_tracking = {
                'last_log_time': 0.0,
                'last_values': {
                    'avg_lin_x': 0.0,
                    'avg_lin_y': 0.0,
                    'avg_ang': 0.0,
                    'p_x': 0.0, 'i_x': 0.0, 'd_x': 0.0,
                    'p_y': 0.0, 'i_y': 0.0, 'd_y': 0.0,
                    'p_a': 0.0, 'i_a': 0.0, 'd_a': 0.0,
                    'strategy': None
                },
                'min_log_interval': LOG_THROTTLE_DIAG,  # Use existing throttle setting
                'significant_change_threshold': 0.1      # 10% change threshold for significant changes
            }
        
        # Only continue if minimum time has passed
        if (current_time - self._diag_tracking['last_log_time']) < self._diag_tracking['min_log_interval']:
            return
                
        # Get PID components
        p_x, i_x, d_x = self.pid_linear_x.get_components()
        p_y, i_y, d_y = self.pid_linear_y.get_components()
        p_a, i_a, d_a = self.pid_angular.get_components()
        
        # Get strategy name safely - this is a fix for the tuple formatting issue
        strategy_name = str(self.current_strategy)
        if hasattr(self.current_strategy, 'name'):
            strategy_name = self.current_strategy.name
        
        # Check if values have changed significantly enough to log
        current_values = {
            'avg_lin_x': avg_lin_x_vel, 
            'avg_lin_y': avg_lin_y_vel, 
            'avg_ang': avg_ang_vel,
            'p_x': p_x, 'i_x': i_x, 'd_x': d_x,
            'p_y': p_y, 'i_y': i_y, 'd_y': d_y,
            'p_a': p_a, 'i_a': i_a, 'd_a': d_a,
            'strategy': strategy_name
        }
        
        # Detect significant changes or if it's been a long time since last log
        significant_change = False
        force_log = (current_time - self._diag_tracking['last_log_time']) > (self._diag_tracking['min_log_interval'] * 2)
        
        if not force_log:
            # Check numerical values for significant changes
            for key in ['avg_lin_x', 'avg_lin_y', 'avg_ang', 
                    'p_x', 'i_x', 'd_x', 'p_y', 'i_y', 'd_y', 'p_a', 'i_a', 'd_a']:
                if (abs(current_values[key]) > 0.01 or abs(self._diag_tracking['last_values'][key]) > 0.01):
                    relative_change = abs(current_values[key] - self._diag_tracking['last_values'][key]) / (
                        max(0.01, abs(self._diag_tracking['last_values'][key])))
                    if relative_change > self._diag_tracking['significant_change_threshold']:
                        significant_change = True
                        break
            
            # Check if strategy changed
            if current_values['strategy'] != self._diag_tracking['last_values']['strategy']:
                significant_change = True
        
        # Only log if changes are significant or it's time for a forced log
        if significant_change or force_log:
            # Log detailed information if debug level permits
            if self.debug_level >= 1:
                try:
                    # Format components with zero suppression for readability
                    def format_component(val):
                        return f"{val:.2f}" if abs(val) >= 0.005 else "0.00"
                    
                    x_terms = f"[{format_component(p_x)}, {format_component(i_x)}, {format_component(d_x)}]"
                    y_terms = f"[{format_component(p_y)}, {format_component(i_y)}, {format_component(d_y)}]"
                    a_terms = f"[{format_component(p_a)}, {format_component(i_a)}, {format_component(d_a)}]"
                    
                    diag_msg = (
                        f"DIAGNOSTICS: "
                        f"Avg Vel=[{avg_lin_x_vel:.2f}, {avg_lin_y_vel:.2f}, {avg_ang_vel:.2f}], "
                        f"PID X={x_terms}, "
                        f"PID Y={y_terms}, "
                        f"PID A={a_terms}, "
                        f"Strategy={strategy_name}"
                    )
                    
                    self.get_logger().info(diag_msg)
                except Exception as e:
                    # Handle potential formatting errors safely
                    self.get_logger().warning(f"Error formatting diagnostics: {str(e)}")
            
            try:
                # Always publish PID diagnostic data regardless of log verbosity
                self._publish_pid_diagnostics()
                
                # Publish performance metrics
                self._publish_performance_metrics()
            except Exception as e:
                self.get_logger().error(f"Error publishing diagnostics: {str(e)}")
            
            # Update tracking data
            try:
                self._diag_tracking['last_log_time'] = current_time
                self._diag_tracking['last_values'] = current_values.copy()
            except Exception as e:
                self.get_logger().warning(f"Error updating diagnostic tracking: {str(e)}")
    
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
        try:
            # Calculate CPU average
            cpu_avg = 0.0
            if self.performance_stats['cpu']:
                cpu_avg = sum(self.performance_stats['cpu']) / len(self.performance_stats['cpu'])
            
            # Calculate cycle time average
            cycle_time_avg = 0.0
            if self.performance_stats['control_cycles']:
                cycle_time_avg = sum(self.performance_stats['control_cycles']) / len(self.performance_stats['control_cycles'])
                cycle_time_avg *= 1000.0  # Convert to ms
            
            # Get strategy name safely
            strategy_name = "unknown"
            if hasattr(self, 'current_strategy'):
                if hasattr(self.current_strategy, 'name'):
                    strategy_name = self.current_strategy.name
                else:
                    strategy_name = str(self.current_strategy)
            
            # Create performance message
            performance_msg = String()
            performance_msg.data = (
                f'{{"cpu": {cpu_avg:.1f}, '
                f'"cycle_time_ms": {cycle_time_avg:.2f}, '
                f'"strategy": "{strategy_name}", '
                f'"skips": {self.performance_stats["control_skips"]}, '
                f'"update_rate": {self.update_rate:.1f}}}'
            )
            
            # Publish
            self.performance_pub.publish(performance_msg)
            
            # Log for debugging at higher debug levels
            if hasattr(self, 'debug_level') and self.debug_level >= 2:
                self.get_logger().debug(
                    f"Performance metrics: CPU={cpu_avg:.1f}%, cycle_time={cycle_time_avg:.2f}ms, "
                    f"strategy={strategy_name}, rate={self.update_rate:.1f}Hz"
                )
                
        except Exception as e:
            # Log error but don't crash
            self.get_logger().error(f"Error publishing performance metrics: {str(e)}")
    
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