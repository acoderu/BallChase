"""
Enhanced Basketball Tracking Robot - PID Controller Node with Natural Movement
========================================================================

This controller implements natural movement patterns for a basketball tracking robot:
- Uses lookahead control for rotation to prevent overshooting
- Implements overrotation protection to detect and correct errors
- Properly coordinates lateral and angular movements
- Makes deliberate, purposeful movements with proper coordination
- Uses intelligent strategy selection with persistence to prevent thrashing
- Enhanced stop logic with drift detection and correction
- Memory-efficient implementation for Raspberry Pi 5 constraints
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import String, Float32MultiArray
import math
import time
import numpy as np
import signal
import sys
from collections import deque

# Topic configuration
TOPICS = {
    "input": {
        "target": "/basketball/fused/position",
        "state": "/robot/state"
    },
    "output": {
        "cmd_vel": "/controller/cmd_vel",
        "diagnostics": "/pid/diagnostics"
    }
}

# Log throttling parameters
LOG_THROTTLE_CONTROL = 1.0     # Seconds between control loop status logs
LOG_THROTTLE_STATE = 0.5       # Seconds between state change logs
LOG_THROTTLE_DIAG = 0.2        # Seconds between diagnostic logs

class LightweightBuffer:
    """
    Memory-efficient buffer for storing historical data with pre-allocated storage.
    """
    
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
        if self.count < self.max_size:
            return self.data[:self.count]
        # Reconstruct in chronological order
        start_idx = self.next_index
        return self.data[start_idx:] + self.data[:start_idx]
        
    def get_latest(self, n=1):
        """Get the latest n values (default is just the latest one)."""
        if self.count == 0:
            return []
        n = min(n, self.count)
        if self.count < self.max_size:
            return self.data[self.count - n:self.count]
        # Calculate indices for the latest n elements
        start_idx = (self.next_index - n) % self.max_size
        if start_idx < self.next_index:
            return self.data[start_idx:self.next_index]
        return self.data[start_idx:] + self.data[:self.next_index]
        
    def get_trend(self):
        """Calculate trend (positive means increasing, negative means decreasing)."""
        if self.count < 3:
            return 0.0
        # Get the latest values
        values = self.get_all()
        # Calculate average of newest 3 values and oldest 3 values
        if len(values) <= 6:
            mid_point = len(values) // 2
            newer_avg = sum(values[mid_point:]) / len(values[mid_point:])
            older_avg = sum(values[:mid_point]) / len(values[:mid_point])
        else:
            newer_avg = sum(values[-3:]) / 3.0
            older_avg = sum(values[:3]) / 3.0
        # Return difference (positive means increasing trend)
        return newer_avg - older_avg

class ErrorTracker:
    """
    Lightweight error tracker that monitors error values over time.
    Optimized for performance on resource-constrained systems.
    """
    
    def __init__(self, name, max_history=8):
        """Initialize error tracker with efficient storage."""
        self.name = name
        self.current_error = 0.0
        self.previous_error = 0.0
        self.error_history = deque(maxlen=max_history)  # Fixed-size deque for memory efficiency
        self.last_correction_time = 0.0
        self.sign_changes = 0  # Count of error sign changes (useful for oscillation detection)
        self.accumulated_error = 0.0
        self.decay_factor = 0.9  # Simplified decay factor
        
    def update(self, error, dt):
        """Update error tracking with new error value."""
        # Check for sign change
        if self.current_error != 0 and error != 0 and (self.current_error * error) < 0:
            self.sign_changes += 1
            
        # Update error history
        self.previous_error = self.current_error
        self.current_error = error
        self.error_history.append(error)
        
        # Update accumulated error with decay
        self.accumulated_error = (self.accumulated_error + error * dt) * self.decay_factor
    
    def reset(self):
        """Reset all tracked errors."""
        self.current_error = 0.0
        self.previous_error = 0.0
        self.error_history.clear()
        self.accumulated_error = 0.0
        self.sign_changes = 0
    
    def get_trend(self):
        """Calculate error trend (positive means growing error, negative means shrinking)."""
        if len(self.error_history) < 4:
            return 0.0
            
        # Use absolute values to detect magnitude changes regardless of direction
        abs_history = [abs(e) for e in self.error_history]
        
        # Calculate averages of newest and oldest halves
        half = len(abs_history) // 2
        recent_avg = sum(abs_history[half:]) / max(1, len(abs_history) - half)
        older_avg = sum(abs_history[:half]) / max(1, half)
        
        return recent_avg - older_avg
    
    def is_error_growing(self):
        """Check if error is growing compared to previous value."""
        return abs(self.current_error) > abs(self.previous_error) * 1.05  # 5% threshold
        
    def is_error_oscillating(self):
        """Check if error is oscillating (frequent sign changes)."""
        return self.sign_changes >= 2 and len(self.error_history) >= 5
        
    def record_correction(self):
        """Record that a correction was made for this error."""
        self.last_correction_time = time.time()
        # Reduce accumulated error when correction is made
        self.accumulated_error *= 0.5

class OptimizedPIDController:
    """
    Memory-efficient PID controller implementation with anti-windup protection.
    Optimized for performance on Raspberry Pi.
    """
    
    def __init__(self, kp, ki, kd, output_min, output_max, anti_windup=True, name="PID"):
        """Initialize a new PID controller with optimized settings."""
        # Control parameters
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.output_min = output_min  # Minimum output value
        self.output_max = output_max  # Maximum output value
        self.anti_windup = anti_windup  # Whether to use anti-windup
        self.name = name  # Controller name for logging
        
        # Internal state
        self.prev_error = 0.0  # Previous error value
        self.integral = 0.0    # Accumulated error (integral term)
        self.last_time = None  # Time of last update
        
        # Diagnostic information
        self.last_p_term = 0.0  # Last proportional term
        self.last_i_term = 0.0  # Last integral term
        self.last_d_term = 0.0  # Last derivative term
        
        # Performance optimization
        self.integral_deadband = 0.05  # Only accumulate integral when error is significant
        self.integral_decay = 0.7      # Decay rate for integral when error is small
        
    def reset(self):
        """Reset controller state, restarting from scratch."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        
    def compute(self, error, current_time=None, force_zero=False):
        """
        Compute the control output based on the error, with optimizations for
        resource-constrained environments.
        """
        # If forcing zero output, bypass calculations
        if force_zero:
            self.last_p_term = 0.0
            self.last_i_term = 0.0
            self.last_d_term = 0.0
            return 0.0
            
        # Use current time if not provided
        if current_time is None:
            current_time = time.time()
            
        # Initialize time on first call
        if self.last_time is None:
            self.last_time = current_time
            self.prev_error = error
            # P-only control on first iteration (no I or D)
            output = self.kp * error
            self.last_p_term = output
            return max(self.output_min, min(self.output_max, output))
            
        # Calculate dt (time since last update)
        dt = current_time - self.last_time
        if dt <= 0.001:  # Protect against very small or negative dt
            dt = 0.01  # Fallback to prevent division by zero (assume 100Hz)
            
        # Calculate each PID term
        # Proportional term (proportional to error)
        p_term = self.kp * error
        
        # Improved integral term handling with resource optimization
        # Only accumulate integral when error is significant
        if abs(error) > self.integral_deadband:
            self.integral += error * dt
        else:
            # More aggressively reduce integral term when close to target
            self.integral *= self.integral_decay
            
        i_term = self.ki * self.integral
        
        # Derivative term (rate of change of error)
        error_change = error - self.prev_error
        d_term = self.kd * error_change / max(dt, 0.001)  # Protect against division by zero
        
        # Calculate raw output by summing all terms
        output = p_term + i_term + d_term
        
        # Apply output limits
        output_limited = max(self.output_min, min(self.output_max, output))
        
        # Improved anti-windup logic
        if self.anti_windup:
            if output != output_limited and abs(self.ki) > 1e-10:  # Avoid division by zero
                # Reduce integral by the excess output scaled by Ki
                self.integral -= (output - output_limited) / self.ki
                # Recalculate integral term
                i_term = self.ki * self.integral
                
            # Additional anti-windup when error changes sign
            if error * self.prev_error < 0:  # Error changed sign
                # Reduce integral more aggressively to avoid overshooting
                self.integral *= 0.3
                i_term = self.ki * self.integral
                
        # Save individual terms for diagnostics
        self.last_p_term = p_term
        self.last_i_term = i_term
        self.last_d_term = d_term
        
        # Save state for next iteration
        self.prev_error = error
        self.last_time = current_time
        
        return output_limited
        
    def get_components(self):
        """Get the last calculated PID components."""
        return (self.last_p_term, self.last_i_term, self.last_d_term)

class EnhancedPIDControllerNode(Node):
    """
    Enhanced PID Controller node for basketball tracking with natural movement.
    Optimized for Raspberry Pi 5 with limited resources.
    """
    
    def __init__(self):
        """Initialize the PID controller node with all required components."""
        super().__init__('pid_controller')
        
        # Set up parameters
        self._declare_parameters()
        
        # Initialize controllers
        self._init_controllers()
        
        # Set up state variables
        self._init_state_variables()
        
        # Pre-allocate ROS messages (optimization for memory reuse)
        self._init_reusable_messages()
        
        # Set up subscriptions
        self._setup_subscriptions()
        
        # Set up publishers
        self._setup_publishers()
        
        # Set up timers with tiered frequencies
        self._setup_timers()
        
        # Flag to track if we're shutting down
        self._shutting_down = False
        
        # Log startup info (just once)
        self.get_logger().info("Enhanced PID Controller initialized with natural movement behaviors")
        
        # Use throttled logging for parameters
        self._log_parameters_throttled()
        
    def _declare_parameters(self):
        """Declare and get all node parameters with descriptive comments."""
        self.declare_parameters(
            namespace='',
            parameters=[
                # Linear X velocity PID parameters - controls forward/backward movement
                ('linear_x_kp', 1.0),     # Proportional gain
                ('linear_x_ki', 0.03),    # Integral gain
                ('linear_x_kd', 0.1),     # Derivative gain 
                ('linear_x_min', 0.0),    # Backward limit (0.0 to prevent backward motion)
                ('linear_x_max', 0.2),    # Forward limit (m/s)
                
                # Linear Y velocity PID parameters - controls lateral movement (strafing)
                ('linear_y_kp', 0.8),     # Proportional gain
                ('linear_y_ki', 0.15),    # Integral gain
                ('linear_y_kd', 0.05),    # Derivative gain
                ('linear_y_min', -0.2),   # Right strafe limit (m/s)
                ('linear_y_max', 0.2),    # Left strafe limit (m/s)
                
                # Angular velocity PID parameters - controls turning
                ('angular_kp', 3.0),      # Proportional gain (reduced from 3.5 to prevent overshoot)
                ('angular_ki', 0.1),      # Integral gain (reduced from 0.15)
                ('angular_kd', 0.4),      # Derivative gain (increased from 0.3 for better damping)
                ('angular_min', -0.5),    # Right turn limit (rad/s)
                ('angular_max', 0.5),     # Left turn limit (rad/s)
                
                # Control parameters
                ('min_distance', 0.9),       # Minimum distance to keep from ball (meters)
                ('max_distance', 2.0),       # Maximum tracking distance (meters)
                ('target_offset_x', 0.0),    # Desired offset from ball in x direction
                ('target_offset_y', 0.0),    # Desired offset from ball in y direction
                ('target_update_rate', 10.0),# Control loop update rate (Hz)
                ('diagnostics_rate', 0.5),   # Rate for detailed diagnostics (Hz) - reduced from 1.0
                ('frame_check_rate', 0.1),   # Rate for coordinate frame checks (Hz) - reduced from 0.2
                ('debug_level', 0),          # 0=errors only, 1=info, 2=debug - default to minimal logging
                ('adaptive_gains', True),    # Whether to adjust gains based on distance
                ('use_lateral_control', True), # Whether to use Y-axis control for lateral movement
                
                # Natural movement parameters
                ('min_effective_velocity', 0.05),    # Minimum effective velocity (m/s)
                ('max_allowed_velocity', 2.0),       # Maximum allowed velocity (m/s)
                ('min_angular_velocity', 0.1),       # Minimum angular velocity (rad/s)
                
                # Improved deadband parameters
                ('distance_deadband', 0.08),         # Deadband for distance error (m) - increased
                ('lateral_deadband', 0.08),          # Deadband for lateral error (m) - increased
                ('angular_deadband', 5.0),           # Deadband for angular error (degrees)
                ('error_significance_threshold', 0.15), # Threshold for significant error (multiple of deadband)
                
                # Accumulated error parameters
                ('accumulated_error_threshold', 0.1), # Threshold for accumulated error (m-s)
                ('accumulated_angular_threshold', 15.0), # Threshold for accumulated angular error (deg-s)
                ('max_time_without_correction', 1.5),   # Maximum time before forcing correction (s)
                
                # Stop parameters
                ('stop_zone_size', 0.2),           # Size of zone where robot will stop (m)
                ('safety_min_distance', 0.4),      # Emergency stop distance (m)
                ('complete_stop_time', 1.0),       # Time to remain stopped before resuming (s)
                
                # Acceleration and motion parameters
                ('max_accel', 0.6),                # Maximum acceleration per control cycle
                ('max_angular_accel', 1.0),        # Maximum angular acceleration per control cycle
                ('accel_boost_factor', 3.0),       # Boost factor for starting from stop
                ('forward_scale_with_angle', True), # Whether to scale forward velocity based on angular error
                ('angular_scale_threshold', 10.0),  # Angle error in degrees at which to start scaling
                ('always_use_angular_control', True), # Always use angular velocity for alignment
                ('enable_debug_velocity_publish', False), # Whether to log velocity commands
                
                # Target age parameters
                ('target_max_age', 0.5),           # Maximum age of target before stopping (s)
                ('drift_detection_threshold', 0.05), # Threshold for detecting drift after stop (m)
                
                # Decision-making parameters
                ('error_trend_influence', 0.5),    # How much error trend affects decisions (0-1)
                ('forward_first_distance', 0.5),   # Distance threshold for forward-first mode (m)
                ('enable_forward_only_mode', True), # Enable pure forward movement when well-aligned
                ('coordination_factor', 0.8),      # Factor for coordinating multi-dimensional movement (0-1)
                ('approach_slowdown_distance', 0.6), # Distance at which to start slowing down (m)
                
                # New lookahead and protection parameters
                ('rotation_lookahead_factor', 0.7), # Apply only 70% of calculated rotation initially
                ('overrotation_protection', True),  # Enable protection against overrotation
                ('overrotation_threshold', 1.2),    # Threshold for detecting overrotation (ratio of errors)
                ('strategy_min_duration', 0.2),     # Minimum duration for a strategy (s)
                ('min_reassessment_time', 0.1),     # Time to wait before reassessing rotation (s)
                ('error_history_size', 8),          # Size of error history buffer
                ('velocity_history_size', 6),       # Size of velocity history buffer
                ('dynamic_deadband_factor', 1.3),   # Factor to increase deadband after corrections
                ('enable_emergency_reversal', True), # Enable emergency direction reversal for severe overrotation
            ]
        )
        
        # Get all parameters (standard parameters from original code)
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
        
        self.min_distance = self.get_parameter('min_distance').value
        self.max_distance = self.get_parameter('max_distance').value
        self.target_offset_x = self.get_parameter('target_offset_x').value
        self.target_offset_y = self.get_parameter('target_offset_y').value
        self.update_rate = self.get_parameter('target_update_rate').value
        self.diagnostics_rate = self.get_parameter('diagnostics_rate').value
        self.frame_check_rate = self.get_parameter('frame_check_rate').value
        self.debug_level = self.get_parameter('debug_level').value
        self.adaptive_gains = self.get_parameter('adaptive_gains').value
        self.use_lateral_control = self.get_parameter('use_lateral_control').value
        
        # Natural movement parameters
        self.min_effective_velocity = self.get_parameter('min_effective_velocity').value
        self.max_allowed_velocity = self.get_parameter('max_allowed_velocity').value
        self.min_angular_velocity = self.get_parameter('min_angular_velocity').value
        
        # Improved deadband parameters
        self.distance_deadband = self.get_parameter('distance_deadband').value
        self.lateral_deadband = self.get_parameter('lateral_deadband').value
        self.angular_deadband = self.get_parameter('angular_deadband').value
        self.error_significance_threshold = self.get_parameter('error_significance_threshold').value
        
        # Accumulated error parameters
        self.accumulated_error_threshold = self.get_parameter('accumulated_error_threshold').value
        self.accumulated_angular_threshold = self.get_parameter('accumulated_angular_threshold').value
        self.max_time_without_correction = self.get_parameter('max_time_without_correction').value
        
        # Stop parameters
        self.stop_zone_size = self.get_parameter('stop_zone_size').value
        self.safety_min_distance = self.get_parameter('safety_min_distance').value
        self.complete_stop_time = self.get_parameter('complete_stop_time').value
        
        # Acceleration and motion parameters
        self.max_accel = self.get_parameter('max_accel').value
        self.max_angular_accel = self.get_parameter('max_angular_accel').value
        self.accel_boost_factor = self.get_parameter('accel_boost_factor').value
        self.forward_scale_with_angle = self.get_parameter('forward_scale_with_angle').value
        self.angular_scale_threshold = self.get_parameter('angular_scale_threshold').value
        self.always_use_angular_control = self.get_parameter('always_use_angular_control').value
        self.enable_debug_velocity_publish = self.get_parameter('enable_debug_velocity_publish').value
        
        # Target age parameters
        self.target_max_age = self.get_parameter('target_max_age').value
        self.drift_detection_threshold = self.get_parameter('drift_detection_threshold').value
        
        # Decision-making parameters
        self.error_trend_influence = self.get_parameter('error_trend_influence').value
        self.forward_first_distance = self.get_parameter('forward_first_distance').value
        self.enable_forward_only_mode = self.get_parameter('enable_forward_only_mode').value
        self.coordination_factor = self.get_parameter('coordination_factor').value
        self.approach_slowdown_distance = self.get_parameter('approach_slowdown_distance').value
        
        # New lookahead and protection parameters
        self.rotation_lookahead_factor = self.get_parameter('rotation_lookahead_factor').value
        self.overrotation_protection = self.get_parameter('overrotation_protection').value
        self.overrotation_threshold = self.get_parameter('overrotation_threshold').value
        self.strategy_min_duration = self.get_parameter('strategy_min_duration').value
        self.min_reassessment_time = self.get_parameter('min_reassessment_time').value
        self.error_history_size = self.get_parameter('error_history_size').value
        self.velocity_history_size = self.get_parameter('velocity_history_size').value
        self.dynamic_deadband_factor = self.get_parameter('dynamic_deadband_factor').value
        self.enable_emergency_reversal = self.get_parameter('enable_emergency_reversal').value
        
    def _init_controllers(self):
        """Initialize the PID controllers."""
        # Initialize PID controllers with descriptive names
        self.pid_linear_x = OptimizedPIDController(
            self.linear_x_kp, self.linear_x_ki, self.linear_x_kd,
            self.linear_x_min, self.linear_x_max,
            name="Linear X"
        )
        
        self.pid_linear_y = OptimizedPIDController(
            self.linear_y_kp, self.linear_y_ki, self.linear_y_kd,
            self.linear_y_min, self.linear_y_max,
            name="Linear Y"
        )
        
        self.pid_angular = OptimizedPIDController(
            self.angular_kp, self.angular_ki, self.angular_kd,
            self.angular_min, self.angular_max,
            name="Angular"
        )
        
    def _init_reusable_messages(self):
        """Pre-allocate ROS messages for reuse to avoid memory churn."""
        # Pre-allocate velocity command message
        self._cmd_vel_msg = Twist()
        
        # Pre-allocate diagnostics message
        self._diag_msg = Float32MultiArray()
        # Pre-allocate data array with fixed size for diagnostics
        self._diag_data = np.zeros(11, dtype=np.float32)
        
    def _init_state_variables(self):
        """Initialize all state tracking variables."""
        # Target tracking
        self.current_target = None      # Latest target position
        self.last_target_time = None    # When we last received a target
        self.previous_target_time = None # Previous target time for update rate calculation
        
        # Robot state
        self.robot_state = "initializing"  # Current state from state manager
        self.previous_state = None         # For state transition detection
        self.last_control_time = time.time()  # For periodic logging
        
        # Log throttling timestamps
        self.last_control_log_time = 0.0
        self.last_state_log_time = 0.0
        self.last_diag_log_time = 0.0
        self.last_frame_check_time = 0.0
        self.last_velocity_publish_log_time = 0.0
        
        # Derived values
        self.current_distance = 0.0     # Current distance to target
        self.current_bearing = 0.0      # Current bearing to target
        self.current_lateral = 0.0      # Current lateral offset to target
        
        # Motion smoothing
        self.last_cmd_vel = (0.0, 0.0, 0.0)  # (lin_x, lin_y, ang_z)
        
        # Diagnostic information
        self.cycle_count = 0            # Number of control cycles
        
        # Use LightweightBuffer for velocity history (reduced size for memory efficiency)
        self.velocity_history = LightweightBuffer(max_size=self.velocity_history_size)
        
        # Cached values
        self.desired_distance = self.min_distance + self.target_offset_x
        
        # Error tracking using optimized ErrorTracker class
        self.distance_error_tracker = ErrorTracker("distance", max_history=self.error_history_size)
        self.lateral_error_tracker = ErrorTracker("lateral", max_history=self.error_history_size)
        self.angular_error_tracker = ErrorTracker("angular", max_history=self.error_history_size)
        
        # Stopped state tracking
        self._robot_stopped = False
        self._stop_time = 0.0
        
        # Last stop position for drift detection
        self._last_stop_position = (0.0, 0.0, 0.0)  # distance, lateral, bearing
        
        # Strategy tracking
        self.current_strategy = "IDLE"
        self.strategy_start_time = time.time()
        
        # Strategy persistence tracking (to prevent thrashing)
        self.strategy_change_time = time.time()
        self.previous_strategy = None
        
        # Movement coordination
        self.movement_phase = 0.0  # Used for coordinated movement (0.0-1.0)
        self.last_phase_update = time.time()
        
        # Target quality assessment
        self.target_update_intervals = deque(maxlen=5)  # Reduced buffer size
        self.target_quality_score = 1.0
        
        # New variables for lookahead control and overrotation protection
        self.rotation_in_progress = False
        self.rotation_start_time = 0.0
        self.rotation_reassessment_time = 0.0
        self.rotation_initial_error = 0.0
        self.rotation_direction = 0.0  # Last rotation direction
        
        # Dynamic deadbands (initialized to base values)
        self.effective_distance_deadband = self.distance_deadband
        self.effective_lateral_deadband = self.lateral_deadband
        self.effective_angular_deadband = self.angular_deadband
        
        # Overrotation emergency intervention
        self.emergency_intervention_active = False
        self.emergency_intervention_start = 0.0
        self.emergency_intervention_duration = 0.3  # seconds
        
        # Count consecutive overrotations for escalating response
        self.consecutive_overrotations = 0
        
        # Pre-calculated constants for optimization
        self._precalc_constants()
        
    def _precalc_constants(self):
        """Pre-calculate constants to avoid repeated calculations."""
        # Angular conversions to save computation
        self.deg_to_rad = math.pi / 180.0
        self.rad_to_deg = 180.0 / math.pi
        
        # Combined velocity limits (sqrt calculation is expensive)
        self.combined_velocity_limit = math.sqrt(
            self.linear_x_max**2 + self.linear_y_max**2 + self.angular_max**2
        )
        
        # Stop zone thresholds
        self.stop_forward_threshold = self.stop_zone_size
        self.stop_lateral_threshold = self.lateral_deadband * 1.5
        self.stop_angular_threshold = self.angular_deadband * 1.5
        
    def _setup_subscriptions(self):
        """Set up all subscriptions for this node."""
        # Subscribe to robot state
        self.state_sub = self.create_subscription(
            String,
            TOPICS["input"]["state"],
            self.state_callback,
            10
        )
        
        # Subscribe to basketball target
        self.target_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["target"],
            self.target_callback,
            10
        )
        
    def _setup_publishers(self):
        """Set up all publishers for this node."""
        # Publisher for velocity commands
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            TOPICS["output"]["cmd_vel"],
            10
        )
        
        # Publisher for PID diagnostic info
        self.pid_diag_pub = self.create_publisher(
            Float32MultiArray,
            TOPICS["output"]["diagnostics"],
            10
        )
        
    def _setup_timers(self):
        """Set up timer callbacks for periodic tasks with tiered frequencies."""
        # Create control loop timer at specified update rate
        self.timer = self.create_timer(1.0 / self.update_rate, self.control_loop_callback)
        
        # Create a slower timer for detailed diagnostics
        self.diagnostic_timer = self.create_timer(1.0 / self.diagnostics_rate, self.publish_detailed_diagnostics)
        
        # Create a timer for coordinate frame diagnostics at lowest frequency
        self.frame_diagnostic_timer = self.create_timer(1.0 / self.frame_check_rate, self.run_coordinate_frame_diagnostics)
        
    def _log_parameters_throttled(self):
        """Log parameters with throttling to avoid log spam."""
        if self.debug_level < 1:
            return
            
        self.get_logger().info("=== PID Controller Parameters ===")
        self.get_logger().info("Linear X velocity PID:")
        self.get_logger().info(f"  Kp: {self.linear_x_kp}, Ki: {self.linear_x_ki}, Kd: {self.linear_x_kd}")
        
        self.get_logger().info("Linear Y velocity PID:")
        self.get_logger().info(f"  Kp: {self.linear_y_kp}, Ki: {self.linear_y_ki}, Kd: {self.linear_y_kd}")
        
        self.get_logger().info("Angular velocity PID:")
        self.get_logger().info(f"  Kp: {self.angular_kp}, Ki: {self.angular_ki}, Kd: {self.angular_kd}")
        
        self.get_logger().info("Natural movement parameters:")
        self.get_logger().info(f"  Lookahead factor: {self.rotation_lookahead_factor}")
        self.get_logger().info(f"  Overrotation protection: {self.overrotation_protection}")
        self.get_logger().info(f"  Min distance: {self.min_distance} m")
        self.get_logger().info("==================================")
        
    def _log_throttled(self, level_func, message, min_interval, last_time_attr):
        """
        Log messages with throttling to reduce log volume.
        
        Args:
            level_func: Logger function (e.g., self.get_logger().info)
            message: Message to log
            min_interval: Minimum time between logs in seconds
            last_time_attr: Attribute name storing last log time
        
        Returns:
            bool: True if message was logged, False if throttled
        """
        current_time = time.time()
        last_time = getattr(self, last_time_attr, 0)
        
        if current_time - last_time >= min_interval:
            level_func(message)
            setattr(self, last_time_attr, current_time)
            return True
            
        return False
    
    def state_callback(self, msg):
        """
        Handle robot state updates from the state manager.
        
        This ensures the PID controller behaves appropriately based on the
        current operational state of the robot.
        """
        new_state = msg.data
        
        # If state changed, handle the transition
        if new_state != self.robot_state:
            # Throttled logging for state changes
            self._log_throttled(
                self.get_logger().info,
                f"Robot state changed: {self.robot_state} → {new_state}",
                LOG_THROTTLE_STATE,
                'last_state_log_time'
            )
            
            self.previous_state = self.robot_state
            self.robot_state = new_state
            
            # Only reset PIDs when switching to/from tracking
            if new_state == "tracking" or self.previous_state == "tracking":
                self.pid_linear_x.reset()
                self.pid_linear_y.reset()
                self.pid_angular.reset()
                
                # Reset error trackers
                self.distance_error_tracker.reset()
                self.lateral_error_tracker.reset()
                self.angular_error_tracker.reset()
                
                # Reset rotation tracking
                self.rotation_in_progress = False
                self.emergency_intervention_active = False
                self.consecutive_overrotations = 0
                
                if self.debug_level >= 1:
                    self.get_logger().debug("PID controllers and error trackers reset due to state change")
                
            # If we're not in tracking mode, ensure the robot is stopped
            # (unless it's in searching or lost_ball mode, where the state manager controls motion)
            if new_state != "tracking" and new_state != "searching" and new_state != "lost_ball":
                self.stop_robot()
    
    def target_callback(self, msg):
        """
        Handle target position updates from the state manager.
        
        This receives the 3D position of the basketball and updates
        target tracking variables.
        """
        if self._shutting_down:
            return
            
        # Update target update interval tracking
        current_time = time.time()
        if self.last_target_time is not None:
            interval = current_time - self.last_target_time
            # Only track reasonable intervals (avoid bogus values after long gaps)
            if 0.005 < interval < 1.0:
                self.target_update_intervals.append(interval)
        
        self.previous_target_time = self.last_target_time
        self.last_target_time = current_time
        self.current_target = msg
        
        # Extract key information from target
        target = msg.point
        
        # Calculate full 3D distance to target (more robust)
        full_distance = math.sqrt(target.x**2 + target.y**2 + target.z**2)
        
        # Store raw target components for logging
        self.raw_target_x = target.x
        self.raw_target_y = target.y
        self.raw_target_z = target.z
        
        # Handle coordinates based on frame_id
        frame_id = msg.header.frame_id if hasattr(msg.header, 'frame_id') else "unknown_frame"
        
        # Store for logging and diagnostics
        self.target_frame = frame_id
        
        # Use appropriate interpretation based on coordinate frame
        # State manager appears to be publishing in a frame where distance is the 
        # magnitude of the position vector, not just the z component
        self.current_distance = full_distance
        
        # Calculate bearing/direction to ball
        # Most robot/world frames have x forward, y left - adapt based on frame
        # Use atan2 for consistent angle calculation
        if frame_id == "camera_frame" or frame_id == "camera_optical_frame":
            # Camera optical frame: Z forward, X right, Y down
            self.current_bearing = math.atan2(target.x, target.z)
            # For camera frame, positive X is right, which is the correct sign convention
            self.current_lateral = target.x
        else:
            # Standard robot frame: X forward, Y left
            self.current_bearing = math.atan2(target.y, target.x)
            
            # In base_link frame, positive Y is left
            # To make lateral consistent: positive = ball is to the left
            self.current_lateral = target.y
            
        # Calculate target quality score based on update frequency stability
        if len(self.target_update_intervals) >= 3:
            # Get mean and standard deviation
            mean_interval = sum(self.target_update_intervals) / len(self.target_update_intervals)
            variance = sum((x - mean_interval) ** 2 for x in self.target_update_intervals) / len(self.target_update_intervals)
            std_dev = math.sqrt(variance)
            
            # Coefficient of variation (lower is better)
            cov = std_dev / mean_interval if mean_interval > 0 else 1.0
            
            # Quality score from 0.0 to 1.0 based on coefficient of variation
            # A perfectly stable update rate would have cov=0 and quality=1.0
            self.target_quality_score = max(0.0, min(1.0, 1.0 - cov))

        # Check for safety distance - if ball is too close, log it immediately
        if self.current_distance < self.safety_min_distance:
            self.get_logger().warn(
                f"Ball very close to robot! Distance={self.current_distance:.2f}m, "
                f"safety threshold={self.safety_min_distance:.2f}m"
            )
            
        # Throttled logging for target updates (only at debug level >= 1)
        if self.debug_level >= 1:
            self._log_throttled(
                self.get_logger().info,
                f"Target update: raw=[{target.x:.2f}, {target.y:.2f}, {target.z:.2f}], "
                f"frame={frame_id}, "
                f"calculated: distance={self.current_distance:.2f}m, "
                f"lateral={self.current_lateral:.2f}m, "
                f"bearing={math.degrees(self.current_bearing):.1f}°",
                0.5,  # Every 0.5 seconds max
                'last_target_log_time'
            )
    
    def _determine_alignment_strategy(self, distance_error, lateral_error, angular_error_degrees, distance, dt):
        """
        Determine the optimal alignment strategy based on current errors with improved
        strategy persistence to prevent thrashing.
        
        Args:
            distance_error (float): Error in distance (meters)
            lateral_error (float): Error in lateral position (meters)
            angular_error_degrees (float): Error in angular position (degrees)
            distance (float): Current distance to target (meters)
            dt (float): Time since last control cycle
            
        Returns:
            dict: Strategy information including strategy name, movement flags, and scale factors
        """
        # Get absolute error values
        abs_distance_error = abs(distance_error)
        abs_lateral_error = abs(lateral_error)
        abs_angular_error = abs(angular_error_degrees)
        
        # First check if we need to continue with current strategy
        current_time = time.time()
        should_persist_strategy = (
            current_time - self.strategy_change_time < self.strategy_min_duration and
            self.current_strategy != "NO_MOVEMENT" and
            self.current_strategy != "IDLE"
        )
        
        # If we're in the middle of a rotation with lookahead, continue until reassessment
        if (self.rotation_in_progress and 
            current_time < self.rotation_reassessment_time):
            # Continue with angular-only control during lookahead rotation
            return {
                "strategy_name": "LOOKAHEAD_ROTATION",
                "use_forward": False,
                "use_lateral": False,
                "use_angular": True,
                "forward_scale": 0.0,
                "lateral_scale": 0.0,
                "angular_scale": 1.0,
                "reason": f"Continuing lookahead rotation (reassessment in {self.rotation_reassessment_time - current_time:.2f}s)"
            }
            
        # If emergency intervention for overrotation is active, override with corrective strategy
        if self.emergency_intervention_active:
            if current_time - self.emergency_intervention_start < self.emergency_intervention_duration:
                # Reverse the previous rotation direction
                return {
                    "strategy_name": "EMERGENCY_CORRECTION",
                    "use_forward": False,
                    "use_lateral": False,
                    "use_angular": True,
                    "forward_scale": 0.0,
                    "lateral_scale": 0.0,
                    "angular_scale": 0.7,  # Use 70% of calculated correction
                    "reason": "Emergency correction for overrotation"
                }
            else:
                # Emergency intervention complete
                self.emergency_intervention_active = False
        
        # Check for strategy persistence (prevent thrashing)
        if should_persist_strategy:
            # Continue with current strategy
            # Get the last strategy parameters from stored values
            if self.current_strategy == "FORWARD_ONLY":
                return {
                    "strategy_name": "FORWARD_ONLY",
                    "use_forward": True,
                    "use_lateral": False,
                    "use_angular": False,
                    "forward_scale": 1.0,
                    "lateral_scale": 0.0,
                    "angular_scale": 0.0,
                    "reason": f"Persisting FORWARD_ONLY (min duration: {self.strategy_min_duration}s)"
                }
            elif self.current_strategy == "LATERAL_PRIMARY":
                return {
                    "strategy_name": "LATERAL_PRIMARY",
                    "use_forward": abs_distance_error > self.effective_distance_deadband,
                    "use_lateral": True,
                    "use_angular": abs_angular_error > self.effective_angular_deadband,
                    "forward_scale": 0.3 if abs_distance_error > self.effective_distance_deadband else 0.0,
                    "lateral_scale": 1.0,
                    "angular_scale": 0.3 if abs_angular_error > self.effective_angular_deadband else 0.0,
                    "reason": f"Persisting LATERAL_PRIMARY (min duration: {self.strategy_min_duration}s)"
                }
            elif self.current_strategy == "ANGULAR_ONLY" or self.current_strategy == "ANGULAR_EFFICIENT":
                return {
                    "strategy_name": "ANGULAR_ONLY",
                    "use_forward": False,
                    "use_lateral": False,
                    "use_angular": True,
                    "forward_scale": 0.0,
                    "lateral_scale": 0.0,
                    "angular_scale": 1.0,
                    "reason": f"Persisting {self.current_strategy} (min duration: {self.strategy_min_duration}s)"
                }
        
        # Define error thresholds (with dynamic deadbands)
        LARGE_DISTANCE_ERROR = self.effective_distance_deadband * 3.0
        MEDIUM_DISTANCE_ERROR = self.effective_distance_deadband * 1.5
        
        LARGE_LATERAL_ERROR = self.effective_lateral_deadband * 3.0
        MEDIUM_LATERAL_ERROR = self.effective_lateral_deadband * 1.5
        
        LARGE_ANGULAR_ERROR = self.effective_angular_deadband * 3.0
        MEDIUM_ANGULAR_ERROR = self.effective_angular_deadband * 1.5
        
        # Default strategy: no movement
        strategy = {
            "strategy_name": "NO_MOVEMENT",
            "use_forward": False,
            "use_lateral": False,
            "use_angular": False,
            "forward_scale": 0.0,
            "lateral_scale": 0.0,
            "angular_scale": 0.0,
            "reason": "Default no movement strategy"
        }
        
        # 1. Check if all errors are very small (within deadbands)
        if (abs_distance_error < self.effective_distance_deadband and
            abs_lateral_error < self.effective_lateral_deadband and
            abs_angular_error < self.effective_angular_deadband):
            
            # Check if any accumulated errors are significant
            accumulated_distance = abs(self.distance_error_tracker.accumulated_error)
            accumulated_lateral = abs(self.lateral_error_tracker.accumulated_error)
            accumulated_angular = abs(self.angular_error_tracker.accumulated_error)
            
            # Only correct accumulated errors if they've grown significantly
            if (accumulated_distance > self.accumulated_error_threshold or
                accumulated_lateral > self.accumulated_error_threshold or
                accumulated_angular > math.radians(self.accumulated_angular_threshold)):
                
                # Determine which accumulated error is most significant
                distance_ratio = accumulated_distance / self.accumulated_error_threshold
                lateral_ratio = accumulated_lateral / self.accumulated_error_threshold
                angular_ratio = accumulated_angular / math.radians(self.accumulated_angular_threshold)
                
                max_ratio = max(distance_ratio, lateral_ratio, angular_ratio)
                
                # Correct the dimension with the most significant accumulated error
                if max_ratio == distance_ratio:
                    strategy = {
                        "strategy_name": "ACCUMULATED_DISTANCE",
                        "use_forward": True,
                        "use_lateral": False,
                        "use_angular": False,
                        "forward_scale": 0.5,
                        "lateral_scale": 0.0,
                        "angular_scale": 0.0,
                        "reason": f"Correcting accumulated distance error: {accumulated_distance:.3f}m-s"
                    }
                elif max_ratio == lateral_ratio:
                    strategy = {
                        "strategy_name": "ACCUMULATED_LATERAL",
                        "use_forward": False,
                        "use_lateral": True,
                        "use_angular": False,
                        "forward_scale": 0.0,
                        "lateral_scale": 0.5,
                        "angular_scale": 0.0,
                        "reason": f"Correcting accumulated lateral error: {accumulated_lateral:.3f}m-s"
                    }
                else:  # Angular has max ratio
                    strategy = {
                        "strategy_name": "ACCUMULATED_ANGULAR",
                        "use_forward": False,
                        "use_lateral": False,
                        "use_angular": True,
                        "forward_scale": 0.0,
                        "lateral_scale": 0.0,
                        "angular_scale": 0.5,
                        "reason": f"Correcting accumulated angular error: {math.degrees(accumulated_angular):.1f}°-s"
                    }
            else:
                # No significant errors - can use FORWARD_ONLY if needed
                if self.enable_forward_only_mode and distance_error > 0.01:
                    strategy = {
                        "strategy_name": "FORWARD_ONLY",
                        "use_forward": True,
                        "use_lateral": False,
                        "use_angular": False,
                        "forward_scale": 1.0,
                        "lateral_scale": 0.0,
                        "angular_scale": 0.0,
                        "reason": "All errors small, pure forward movement"
                    }
                else:
                    strategy = {
                        "strategy_name": "NO_MOVEMENT",
                        "use_forward": False,
                        "use_lateral": False,
                        "use_angular": False,
                        "forward_scale": 0.0,
                        "lateral_scale": 0.0,
                        "angular_scale": 0.0,
                        "reason": "All errors within deadbands"
                    }
            
            # Record time of strategy change if different
            if strategy["strategy_name"] != self.current_strategy:
                self.strategy_change_time = current_time
                self.previous_strategy = self.current_strategy
                
            return strategy
        
        # 2. Emergency strategies for large errors
        # If angular error is very large, fix that first with lookahead control
        if abs_angular_error > LARGE_ANGULAR_ERROR:
            # Mark that we're starting a rotation with lookahead control
            if not self.rotation_in_progress:
                self.rotation_in_progress = True
                self.rotation_start_time = current_time
                self.rotation_initial_error = angular_error_degrees
                self.rotation_direction = math.copysign(1.0, angular_error_degrees)
                # Set reassessment time
                self.rotation_reassessment_time = current_time + self.min_reassessment_time
                
            strategy = {
                "strategy_name": "ANGULAR_ONLY",
                "use_forward": False,
                "use_lateral": False,
                "use_angular": True,
                "forward_scale": 0.0,
                "lateral_scale": 0.0,
                "angular_scale": self.rotation_lookahead_factor,  # Use lookahead factor
                "reason": f"Large angular error with lookahead: {abs_angular_error:.1f}°"
            }
            
            # Record time of strategy change if different
            if strategy["strategy_name"] != self.current_strategy:
                self.strategy_change_time = current_time
                self.previous_strategy = self.current_strategy
                
            return strategy
            
        # Reset rotation tracking if we're not doing a rotation
        self.rotation_in_progress = False
            
        # If lateral error is very large, fix that first
        if abs_lateral_error > LARGE_LATERAL_ERROR:
            strategy = {
                "strategy_name": "LATERAL_PRIMARY",
                "use_forward": abs_distance_error > MEDIUM_DISTANCE_ERROR,
                "use_lateral": True,
                "use_angular": abs_angular_error > self.effective_angular_deadband,
                "forward_scale": 0.3 if abs_distance_error > MEDIUM_DISTANCE_ERROR else 0.0,
                "lateral_scale": 1.0,
                "angular_scale": 0.3 if abs_angular_error > self.effective_angular_deadband else 0.0,
                "reason": f"Large lateral error: {abs_lateral_error:.3f}m"
            }
            
            # Record time of strategy change if different
            if strategy["strategy_name"] != self.current_strategy:
                self.strategy_change_time = current_time
                self.previous_strategy = self.current_strategy
                
            return strategy
            
        # If distance error is very large, fix that first
        if abs_distance_error > LARGE_DISTANCE_ERROR:
            strategy = {
                "strategy_name": "FORWARD_PRIMARY",
                "use_forward": True,
                "use_lateral": abs_lateral_error > self.effective_lateral_deadband,
                "use_angular": abs_angular_error > self.effective_angular_deadband,
                "forward_scale": 1.0,
                "lateral_scale": 0.3 if abs_lateral_error > self.effective_lateral_deadband else 0.0,
                "angular_scale": 0.3 if abs_angular_error > self.effective_angular_deadband else 0.0,
                "reason": f"Large distance error: {abs_distance_error:.3f}m"
            }
            
            # Record time of strategy change if different
            if strategy["strategy_name"] != self.current_strategy:
                self.strategy_change_time = current_time
                self.previous_strategy = self.current_strategy
                
            return strategy
        
        # 3. Efficiency-based strategies for medium errors
        
        # Calculate the effective distance each correction would move the robot
        # toward perfect alignment (using geometric approach)
        distance_via_rotation = abs(math.sin(math.radians(abs_angular_error)) * distance)
        
        # For medium angular errors, consider if rotation is efficient
        if abs_angular_error > MEDIUM_ANGULAR_ERROR:
            # Is rotation more efficient than lateral movement for alignment?
            if distance_via_rotation < abs_lateral_error or abs_lateral_error < self.effective_lateral_deadband:
                strategy = {
                    "strategy_name": "ANGULAR_EFFICIENT",
                    "use_forward": abs_distance_error > self.effective_distance_deadband,
                    "use_lateral": False,
                    "use_angular": True,
                    "forward_scale": 0.7 if abs_distance_error > self.effective_distance_deadband else 0.0,
                    "lateral_scale": 0.0,
                    "angular_scale": 0.9,
                    "reason": f"Angular correction efficient: {distance_via_rotation:.3f}m vs {abs_lateral_error:.3f}m"
                }
                
                # Record time of strategy change if different
                if strategy["strategy_name"] != self.current_strategy:
                    self.strategy_change_time = current_time
                    self.previous_strategy = self.current_strategy
                    
                return strategy
                
        # For medium lateral errors, prioritize lateral movement
        if abs_lateral_error > MEDIUM_LATERAL_ERROR:
            strategy = {
                "strategy_name": "LATERAL_EFFICIENT",
                "use_forward": abs_distance_error > self.effective_distance_deadband,
                "use_lateral": True,
                "use_angular": False,
                "forward_scale": 0.7 if abs_distance_error > self.effective_distance_deadband else 0.0,
                "lateral_scale": 0.9,
                "angular_scale": 0.0,
                "reason": f"Lateral correction efficient: {abs_lateral_error:.3f}m"
            }
            
            # Record time of strategy change if different
            if strategy["strategy_name"] != self.current_strategy:
                self.strategy_change_time = current_time
                self.previous_strategy = self.current_strategy
                
            return strategy
        
        # 4. Coordinated movement for multiple small-medium errors
        # Count dimensions with significant errors
        error_dimensions = 0
        if abs_distance_error > self.effective_distance_deadband:
            error_dimensions += 1
        if abs_lateral_error > self.effective_lateral_deadband:
            error_dimensions += 1
        if abs_angular_error > self.effective_angular_deadband:
            error_dimensions += 1
        
        # If multiple errors, use coordinated strategy with phase-based movement
        if error_dimensions >= 2:
            # Update movement phase (0.0 to 1.0) based on time
            dt = current_time - self.last_phase_update
            self.last_phase_update = current_time
            
            # Advance phase at a rate that completes full cycle in 2.0 seconds
            self.movement_phase += dt / 2.0
            self.movement_phase = self.movement_phase % 1.0
            
            # Determine which dimension to prioritize based on phase
            if self.movement_phase < 0.5:  # Simplified to 2 phases for efficiency
                # First half: prioritize forward movement
                strategy = {
                    "strategy_name": "COORDINATED_FORWARD",
                    "use_forward": True,
                    "use_lateral": abs_lateral_error > self.effective_lateral_deadband,
                    "use_angular": abs_angular_error > self.effective_angular_deadband,
                    "forward_scale": 1.0,
                    "lateral_scale": 0.3 if abs_lateral_error > self.effective_lateral_deadband else 0.0,
                    "angular_scale": 0.3 if abs_angular_error > self.effective_angular_deadband else 0.0,
                    "reason": f"Coordinated forward priority (phase={self.movement_phase:.2f})"
                }
            else:
                # Second half: prioritize angular movement
                strategy = {
                    "strategy_name": "COORDINATED_ANGULAR",
                    "use_forward": abs_distance_error > self.effective_distance_deadband,
                    "use_lateral": abs_lateral_error > self.effective_lateral_deadband,
                    "use_angular": True,
                    "forward_scale": 0.3 if abs_distance_error > self.effective_distance_deadband else 0.0,
                    "lateral_scale": 0.3 if abs_lateral_error > self.effective_lateral_deadband else 0.0,
                    "angular_scale": 1.0,
                    "reason": f"Coordinated angular priority (phase={self.movement_phase:.2f})"
                }
            
            # Record time of strategy change if different
            if strategy["strategy_name"] != self.current_strategy:
                self.strategy_change_time = current_time
                self.previous_strategy = self.current_strategy
                
            return strategy
            
        # 5. Single dimension corrections when only one error is significant
        
        # Only distance error is significant
        if abs_distance_error > self.effective_distance_deadband and abs_lateral_error <= self.effective_lateral_deadband and abs_angular_error <= self.effective_angular_deadband:
            strategy = {
                "strategy_name": "DISTANCE_ONLY",
                "use_forward": True,
                "use_lateral": False,
                "use_angular": False,
                "forward_scale": 1.0,
                "lateral_scale": 0.0,
                "angular_scale": 0.0,
                "reason": f"Only distance error significant: {abs_distance_error:.3f}m"
            }
            
            # Record time of strategy change if different
            if strategy["strategy_name"] != self.current_strategy:
                self.strategy_change_time = current_time
                self.previous_strategy = self.current_strategy
                
            return strategy
            
        # Only lateral error is significant
        if abs_lateral_error > self.effective_lateral_deadband and abs_distance_error <= self.effective_distance_deadband and abs_angular_error <= self.effective_angular_deadband:
            strategy = {
                "strategy_name": "LATERAL_ONLY",
                "use_forward": False,
                "use_lateral": True,
                "use_angular": False,
                "forward_scale": 0.0,
                "lateral_scale": 1.0,
                "angular_scale": 0.0,
                "reason": f"Only lateral error significant: {abs_lateral_error:.3f}m"
            }
            
            # Record time of strategy change if different
            if strategy["strategy_name"] != self.current_strategy:
                self.strategy_change_time = current_time
                self.previous_strategy = self.current_strategy
                
            return strategy
            
        # Only angular error is significant
        if abs_angular_error > self.effective_angular_deadband and abs_distance_error <= self.effective_distance_deadband and abs_lateral_error <= self.effective_lateral_deadband:
            strategy = {
                "strategy_name": "ANGULAR_ONLY",
                "use_forward": False,
                "use_lateral": False,
                "use_angular": True,
                "forward_scale": 0.0,
                "lateral_scale": 0.0,
                "angular_scale": 1.0,
                "reason": f"Only angular error significant: {abs_angular_error:.1f}°"
            }
            
            # Record time of strategy change if different
            if strategy["strategy_name"] != self.current_strategy:
                self.strategy_change_time = current_time
                self.previous_strategy = self.current_strategy
                
            return strategy
        
        # 6. Fallback strategy for any cases not handled above
        strategy = {
            "strategy_name": "MINIMAL_CORRECTION",
            "use_forward": abs_distance_error > self.effective_distance_deadband / 2,
            "use_lateral": abs_lateral_error > self.effective_lateral_deadband / 2,
            "use_angular": abs_angular_error > self.effective_angular_deadband / 2,
            "forward_scale": 0.5 if abs_distance_error > self.effective_distance_deadband / 2 else 0.0,
            "lateral_scale": 0.5 if abs_lateral_error > self.effective_lateral_deadband / 2 else 0.0,
            "angular_scale": 0.5 if abs_angular_error > self.effective_angular_deadband / 2 else 0.0,
            "reason": "Fallback minimal correction strategy"
        }
        
        # Record time of strategy change if different
        if strategy["strategy_name"] != self.current_strategy:
            self.strategy_change_time = current_time
            self.previous_strategy = self.current_strategy
            
        return strategy
    
    def _apply_velocity_bounds(self, velocity, min_non_zero=None, max_allowed=None, name=None):
        """
        Apply velocity bounds to ensure movement is either zero or within acceptable range.
        
        Args:
            velocity: The calculated velocity
            min_non_zero: Minimum non-zero velocity to apply (defaults to class parameter)
            max_allowed: Maximum allowed velocity (defaults to class parameter)
            name: Name of the dimension for logging (optional)
            
        Returns:
            Bounded velocity
        """
        # Use class parameters if not provided
        if min_non_zero is None:
            min_non_zero = self.min_effective_velocity
        if max_allowed is None:
            max_allowed = self.max_allowed_velocity
            
        # Original velocity for comparison
        original_velocity = velocity
        
        # If velocity is very small, set to exactly zero
        if abs(velocity) < 0.001:
            velocity = 0.0
        
        # If velocity exceeds maximum, cap it
        if abs(velocity) > max_allowed:
            velocity = math.copysign(max_allowed, velocity)
            
        # Log significant changes if name was provided (only in debug level 2+)
        if name and abs(velocity - original_velocity) > 0.01 and self.debug_level >= 2:
            self.get_logger().debug(
                f"Velocity bounds applied to {name}: {original_velocity:.3f} -> {velocity:.3f} "
                f"(min={min_non_zero:.3f}, max={max_allowed:.3f})"
            )
        
        return velocity
        
    def _apply_acceleration_limit(self, current_velocity, target_velocity, max_accel, current_time, name=None):
        """
        Apply acceleration limiting for smoother motion.
        
        Args:
            current_velocity (float): Current velocity
            target_velocity (float): Desired velocity
            max_accel (float): Maximum acceleration per control cycle
            current_time (float): Current time
            name (str, optional): Name of dimension for logging
            
        Returns:
            float: Limited velocity that doesn't exceed acceleration constraints
        """
        # Calculate time since last control step
        dt = current_time - getattr(self, "last_accel_time", current_time - 0.1)
        self.last_accel_time = current_time
        
        # Ensure dt is reasonable
        dt = max(0.001, min(dt, 0.1))  # Clamp to reasonable range
        
        # Scale acceleration limit by time
        accel_limit = max_accel * dt * 10.0  # Scale by dt
        
        # Calculate difference between current and target velocity
        vel_diff = target_velocity - current_velocity
        original_target = target_velocity
        
        # Special case for starting movement from zero
        if abs(current_velocity) < 0.01 and abs(target_velocity) > 0.01:
            # When starting from stopped position, allow higher initial acceleration
            boost_factor = self.accel_boost_factor
            accel_limit *= boost_factor
            
            if name and self.debug_level >= 1:
                self.get_logger().info(
                    f"ACCELERATION BOOST: Starting {name} movement with increased limit: {accel_limit:.3f} "
                    f"(boost factor: {boost_factor:.1f})"
                )
        
        # Limit acceleration if needed
        if abs(vel_diff) > accel_limit:
            # Apply limit with correct sign
            target_velocity = current_velocity + math.copysign(accel_limit, vel_diff)
            
            # Log significant changes (only in debug level 2+)
            if name and self.debug_level >= 2:
                self.get_logger().debug(
                    f"Acceleration limit applied to {name}: {original_target:.3f} -> {target_velocity:.3f} "
                    f"(current={current_velocity:.3f}, limit={accel_limit:.3f})"
                )
                
        return target_velocity
            
    def _adjust_gains_for_distance(self, distance):
        """
        Adjust PID gains based on distance to target.
        
        This makes the controller more aggressive when the ball is far away
        and more gentle when it's close.
        
        Args:
            distance (float): Current distance to target in meters
        """
        # If adaptive gains are disabled, return immediately
        if not self.adaptive_gains:
            return
            
        # Scale factor based on distance (1.0 at max_distance, 0.7 at min_distance)
        # Constrain to avoid negative scaling
        normalized_distance = max(0.0, (distance - self.min_distance) / 
                                max(0.1, self.max_distance - self.min_distance))
        scale = 0.7 + 0.3 * min(1.0, normalized_distance)
        
        # Apply scaling consistently to each controller
        # Linear X controller: less aggressive when close
        self.pid_linear_x.kp = self.linear_x_kp * scale
        
        # Further reduce integral gain when close to target
        if distance < self.min_distance + 0.2:
            # Very close to target, use minimal integral gain to prevent overshoot
            self.pid_linear_x.ki = self.linear_x_ki * scale * 0.3
        else:
            self.pid_linear_x.ki = self.linear_x_ki * scale
        
        # Linear Y controller: less aggressive when close
        self.pid_linear_y.kp = self.linear_y_kp * scale
        self.pid_linear_y.ki = self.linear_y_ki * scale
        
        # Angular controller: more precise when close
        # Use a different scale factor for angular to prioritize alignment
        precision_scale = 1.0  # Default no change
        if distance < self.min_distance + 0.3:
            # When close, increase precision
            precision_scale = 1.2
        elif distance > self.max_distance - 0.3:
            # When far, reduce precision to avoid overturning
            precision_scale = 0.9
            
        self.pid_angular.kp = self.angular_kp * precision_scale
        
        # Log the gain adjustments for debugging (only in debug level 1+ and every 20 cycles)
        if self.debug_level >= 1 and self.cycle_count % 20 == 0:
            self.get_logger().info(
                f"ADAPTIVE GAINS: distance={distance:.2f}m, "
                f"lin_x=[P:{self.pid_linear_x.kp:.2f}, I:{self.pid_linear_x.ki:.3f}], "
                f"lin_y=[P:{self.pid_linear_y.kp:.2f}, I:{self.pid_linear_y.ki:.3f}], "
                f"ang=[P:{self.pid_angular.kp:.2f}, I:{self.pid_angular.ki:.3f}]"
            )

    def _check_for_overrotation(self, angular_error, angular_velocity):
        """
        Check for overrotation and implement protective measures.
        
        Args:
            angular_error (float): Current angular error in radians
            angular_velocity (float): Current calculated angular velocity
            
        Returns:
            float: Potentially adjusted angular velocity
        """
        # Skip overrotation protection if disabled
        if not self.overrotation_protection:
            return angular_velocity
            
        # Only check for overrotation if we're actively rotating
        if abs(angular_velocity) < self.min_angular_velocity:
            return angular_velocity
            
        # Check if error is growing (overrotation)
        if self.angular_error_tracker.is_error_growing():
            # Log warning
            self.get_logger().warn(
                f"Overrotation detected! Current={math.degrees(angular_error):.1f}°, "
                f"Previous={math.degrees(self.angular_error_tracker.previous_error):.1f}°"
            )
            
            # Count consecutive overrotations for escalating response
            self.consecutive_overrotations += 1
            
            # Apply immediate velocity reduction to mitigate overrotation
            reduced_velocity = angular_velocity * 0.5
            
            # For severe overrotation, implement emergency intervention
            if self.consecutive_overrotations >= 2 and self.enable_emergency_reversal:
                # Reverse direction completely
                self.get_logger().warn("EMERGENCY DIRECTION REVERSAL ACTIVATED!")
                
                self.emergency_intervention_active = True
                self.emergency_intervention_start = time.time()
                
                # Reverse direction with 70% of current magnitude in opposite direction
                return -1.0 * abs(angular_velocity) * 0.7 * math.copysign(1.0, angular_error)
            
            return reduced_velocity
        else:
            # Reset consecutive counter if no overrotation
            self.consecutive_overrotations = 0
            return angular_velocity

    def _update_dynamic_deadbands(self, linear_x_velocity, linear_y_velocity, angular_velocity):
        """
        Update dynamic deadbands based on recent corrections.
        
        When a correction is made in any dimension, temporarily increase that dimension's
        deadband to prevent oscillation.
        
        Args:
            linear_x_velocity (float): Current forward velocity
            linear_y_velocity (float): Current lateral velocity
            angular_velocity (float): Current angular velocity
        """
        # Define the decay rate for deadbands (slowly return to base values)
        decay_rate = 0.95  # Reduce by 5% each control cycle
        
        # Update distance deadband based on recent forward movement
        if abs(linear_x_velocity) > self.min_effective_velocity:
            # Increase deadband when active correction is happening
            self.effective_distance_deadband = self.distance_deadband * self.dynamic_deadband_factor
        else:
            # Gradually decay back to base value
            self.effective_distance_deadband = max(
                self.distance_deadband,
                self.effective_distance_deadband * decay_rate
            )
            
        # Update lateral deadband based on recent lateral movement
        if abs(linear_y_velocity) > self.min_effective_velocity:
            # Increase deadband when active correction is happening
            self.effective_lateral_deadband = self.lateral_deadband * self.dynamic_deadband_factor
        else:
            # Gradually decay back to base value
            self.effective_lateral_deadband = max(
                self.lateral_deadband,
                self.effective_lateral_deadband * decay_rate
            )
            
        # Update angular deadband based on recent angular movement
        if abs(angular_velocity) > self.min_angular_velocity:
            # Increase deadband when active correction is happening
            self.effective_angular_deadband = self.angular_deadband * self.dynamic_deadband_factor
        else:
            # Gradually decay back to base value
            self.effective_angular_deadband = max(
                self.angular_deadband,
                self.effective_angular_deadband * decay_rate
            )

    def control_loop_callback(self):
        """
        Regular control loop to calculate and publish velocity commands with natural movement behaviors.
        
        This is the core function that:
        1. Checks if we should be controlling the robot in the current state
        2. Calculates appropriate linear and angular velocities using PID controllers
        3. Implements natural movement behavior with error accumulation
        4. Publishes velocity commands to control the robot's motion
        
        Enhanced with lookahead control and overrotation protection.
        """
        if self._shutting_down:
            return
            
        current_time = time.time()
        dt = current_time - self.last_control_time
        self.last_control_time = current_time
        self.cycle_count += 1
        
        # "Stay stopped" delay after stopping
        if self._robot_stopped:
            # Stay stopped for specified time
            if time.time() - self._stop_time < self.complete_stop_time:
                # Keep publishing stop commands
                self._cmd_vel_msg.linear.x = 0.0
                self._cmd_vel_msg.linear.y = 0.0
                self._cmd_vel_msg.angular.z = 0.0
                self.cmd_vel_pub.publish(self._cmd_vel_msg)
                return
            else:
                # After delay, check for drift
                distance_drift = abs(self.current_distance - self._last_stop_position[0])
                lateral_drift = abs(self.current_lateral - self._last_stop_position[1])
                angular_drift = abs(math.degrees(self.current_bearing - self._last_stop_position[2]))
                
                # If significant drift detected, log it
                if (distance_drift > self.drift_detection_threshold or
                    lateral_drift > self.drift_detection_threshold or
                    angular_drift > self.angular_deadband):
                    self.get_logger().warn(
                        f"Drift detected after stop: distance={distance_drift:.3f}m, "
                        f"lateral={lateral_drift:.3f}m, angular={angular_drift:.1f}°"
                    )
                
                # Clear the stopped flag
                self._robot_stopped = False
        
        # Only generate commands in tracking mode with a recent target
        if self.robot_state != "tracking" or self.current_target is None:
            # When not tracking, ensure robot is stopped (unless in a state where another node controls movement)
            if self.robot_state not in ["searching", "lost_ball"]:
                self.stop_robot()
            # Log the reason we're not controlling, but throttled to avoid log spam
            if self.debug_level >= 1 and self.cycle_count % 50 == 0:
                if self.robot_state != "tracking":
                    self._log_throttled(
                        self.get_logger().info,
                        f"Not controlling robot - current state: {self.robot_state}",
                        LOG_THROTTLE_CONTROL,
                        'last_control_log_time'
                    )
                elif self.current_target is None:
                    self._log_throttled(
                        self.get_logger().info,
                        "Not controlling robot - no target data received yet",
                        LOG_THROTTLE_CONTROL,
                        'last_control_log_time'
                    )
            return
            
        # Check if target is too old
        if self.last_target_time is None or (current_time - self.last_target_time) > self.target_max_age:
            self.get_logger().warn(f"Target is too old (>{self.target_max_age*1000:.0f}ms), stopping robot")
            self.stop_robot()
            return
        
        # Use the processed values from target_callback
        distance = self.current_distance
        lateral = self.current_lateral
        bearing = self.current_bearing
        angular_degrees = math.degrees(bearing)
        
        # Enhanced safety checks
        
        # Emergency safety check - if ball is too close, stop immediately
        if distance < self.safety_min_distance:
            self.get_logger().warn(f"Emergency stop! Ball too close: {distance:.2f}m < {self.safety_min_distance:.2f}m")
            self.stop_robot()
            return
        
        # Comprehensive stop condition that considers all dimensions
        # Calculate adjusted stop size based on velocity
        adjusted_stop_size = self.stop_zone_size * (1.0 + max(0.0, min(1.0, abs(self.last_cmd_vel[0]) * 5.0)))
        
        if (abs(distance - self.desired_distance) < adjusted_stop_size and
            abs(lateral - self.target_offset_y) < self.stop_lateral_threshold and
            abs(angular_degrees) < self.stop_angular_threshold):
            self.get_logger().info(
                f"Target fully reached! Distance: {distance:.2f}m, "
                f"lateral: {lateral:.2f}m, bearing: {angular_degrees:.2f}°"
            )
            self.stop_robot()
            return
        
        # Calculate errors
        distance_error = distance - self.desired_distance
        lateral_error = lateral - self.target_offset_y
        angular_error = bearing
        angular_error_degrees = math.degrees(angular_error)
        
        # Update error trackers
        self.distance_error_tracker.update(distance_error, dt)
        self.lateral_error_tracker.update(lateral_error, dt)
        self.angular_error_tracker.update(angular_error, dt)
        
        # Throttled logging for control errors (only at debug level >= 1)
        if self.debug_level >= 1 and self.cycle_count % 10 == 0:
            self._log_throttled(
                self.get_logger().info,
                f"Control errors: distance_error={distance_error:.2f}m, "
                f"lateral_error={lateral_error:.2f}m, "
                f"angular_error={angular_error_degrees:.1f}°",
                0.2,  # Every 0.2 seconds max
                'last_error_log_time'
            )
        
        # First, check if we should switch to final alignment mode
        in_final_alignment = abs(distance - self.desired_distance) < adjusted_stop_size
        if in_final_alignment:
            self.get_logger().info(
                f"Distance reached ({distance:.2f}m), performing final alignment... "
                f"lateral: {lateral:.2f}m, bearing: {angular_degrees:.2f}°"
            )
        
        # Apply adaptive gains if enabled
        self._adjust_gains_for_distance(distance)
        
        # Determine the optimal alignment strategy with improved lookahead control
        strategy = self._determine_alignment_strategy(
            distance_error, lateral_error, angular_error_degrees, distance, dt
        )
        
        # Log the selected strategy
        strategy_name = strategy["strategy_name"]
        if self.debug_level >= 1:
            self.get_logger().info(f"Alignment strategy: {strategy_name} ({strategy['reason']})")
        
        # Save as current strategy
        self.current_strategy = strategy_name
        
        # Apply strategy to movement decisions
        use_forward = strategy["use_forward"]
        use_lateral = strategy["use_lateral"] and self.use_lateral_control
        use_angular = strategy["use_angular"] or self.always_use_angular_control
        
        forward_scale = strategy["forward_scale"]
        lateral_scale = strategy["lateral_scale"]
        angular_scale = strategy["angular_scale"]
        
        # Compute PID outputs using the errors
        linear_x_velocity = self.pid_linear_x.compute(distance_error, current_time, not use_forward)
        lateral_velocity = self.pid_linear_y.compute(lateral_error, current_time, not use_lateral)
        angular_velocity = self.pid_angular.compute(angular_error, current_time, not use_angular)
        
        # Apply strategy scaling factors
        linear_x_velocity *= forward_scale
        lateral_velocity *= lateral_scale
        angular_velocity *= angular_scale
        
        # Implement velocity ramp-down as we approach target
        approach_distance = self.approach_slowdown_distance
        if abs(distance_error) < approach_distance:
            approach_factor = max(0.1, abs(distance_error) / approach_distance)
            if self.debug_level >= 1:
                self.get_logger().info(f"Ramping down velocity as approaching target, factor: {approach_factor:.2f}")
            linear_x_velocity *= approach_factor
            
            # Also scale down lateral and angular for final approach
            if in_final_alignment:
                lateral_velocity *= approach_factor
                angular_velocity *= approach_factor
        
        # Enhanced debugging for motion planning decisions
        if self.debug_level >= 1:
            self.get_logger().info(
                f"MOTION PLANNING: dist_err={distance_error:.3f}m, "
                f"lat_err={lateral_error:.3f}m, "
                f"ang_err={angular_error_degrees:.2f}°, "
                f"use_lateral={use_lateral}, "
                f"raw_lat_vel={lateral_velocity:.3f}m/s, "
                f"raw_ang_vel={angular_velocity:.3f}rad/s"
            )
            
        # Debug log pre-velocity limiting values
        if self.debug_level >= 1:
            self.get_logger().info(f"PRE-LIMIT VELOCITIES: lin_x={linear_x_velocity:.3f}, lin_y={lateral_velocity:.3f}, ang_z={angular_velocity:.3f}")
        
        # Apply acceleration limiting for smoother motion
        linear_x_velocity = self._apply_acceleration_limit(
            self.last_cmd_vel[0], linear_x_velocity, self.max_accel, current_time, "forward")
        
        # Apply acceleration limit for lateral movement with higher limit for responsiveness
        linear_y_velocity = self._apply_acceleration_limit(
            self.last_cmd_vel[1], lateral_velocity, self.max_accel * 1.5, current_time, "lateral")
        
        # Apply acceleration limit for angular movement
        angular_velocity = self._apply_acceleration_limit(
            self.last_cmd_vel[2], angular_velocity, self.max_angular_accel, current_time, "angular")
        
        # Debug log post-acceleration limiting values
        if self.debug_level >= 1:
            self.get_logger().info(f"POST-LIMIT VELOCITIES: lin_x={linear_x_velocity:.3f}, lin_y={linear_y_velocity:.3f}, ang_z={angular_velocity:.3f}")
        
        # Apply overrotation protection for angular velocity
        angular_velocity = self._check_for_overrotation(angular_error, angular_velocity)
        
        # Apply velocity bounds to ensure movement is either zero or effective
        linear_x_velocity = self._apply_velocity_bounds(linear_x_velocity, 
                                                   self.min_effective_velocity,
                                                   self.max_allowed_velocity,
                                                   "forward")
        
        linear_y_velocity = self._apply_velocity_bounds(linear_y_velocity,
                                                   self.min_effective_velocity,
                                                   self.max_allowed_velocity,
                                                   "lateral")
        
        angular_velocity = self._apply_velocity_bounds(angular_velocity,
                                                  self.min_angular_velocity,
                                                  self.max_angular_accel,
                                                  "angular")
        
        # In special circumstances, enforce minimum velocities to ensure movement
        # Only do this when errors are significantly above deadband
        
        # For lateral movement, ensure minimum velocity for significant errors
        if use_lateral and abs(lateral_error) > self.effective_lateral_deadband * 2.0 and abs(linear_y_velocity) < self.min_effective_velocity:
            if abs(linear_y_velocity) > 0.001:  # Only if already non-zero
                linear_y_velocity = math.copysign(self.min_effective_velocity, lateral_error)
                if self.debug_level >= 1:
                    self.get_logger().info(f"LATERAL CORRECTION: Setting minimum velocity: {linear_y_velocity:.3f} m/s")
        
        # For angular movement, ensure minimum velocity for significant errors
        if use_angular and abs(angular_error_degrees) > self.effective_angular_deadband * 2.0 and abs(angular_velocity) < self.min_angular_velocity:
            if abs(angular_velocity) > 0.001:  # Only if already non-zero
                angular_velocity = math.copysign(self.min_angular_velocity, angular_error)
                if self.debug_level >= 1:
                    self.get_logger().info(f"ANGULAR CORRECTION: Setting minimum velocity: {angular_velocity:.3f} rad/s")
        
        # Store for next cycle
        self.last_cmd_vel = (linear_x_velocity, linear_y_velocity, angular_velocity)
        
        # Update error trackers if significant movement is occurring
        if abs(linear_x_velocity) > self.min_effective_velocity / 2.0:
            self.distance_error_tracker.record_correction()
        if abs(linear_y_velocity) > self.min_effective_velocity / 2.0:
            self.lateral_error_tracker.record_correction()
        if abs(angular_velocity) > self.min_angular_velocity / 2.0:
            self.angular_error_tracker.record_correction()
        
        # Optional: scale forward velocity based on angular error
        # This makes the robot slow down when trying to turn to face the ball
        if self.forward_scale_with_angle and abs(angular_error_degrees) > self.angular_scale_threshold:
            # Scale factor: 1.0 at threshold, linearly decreasing to 0.3 at 45 degrees
            forward_scale = max(0.3, 1.0 - ((abs(angular_error_degrees) - self.angular_scale_threshold) / 
                                         (45.0 - self.angular_scale_threshold)))
            # Apply scaling to forward velocity
            linear_x_velocity *= forward_scale
            
            if self.debug_level >= 2 and self.cycle_count % 20 == 0:
                self.get_logger().info(
                    f"Scaling forward velocity: angular_error={angular_error_degrees:.1f}°, "
                    f"scale={forward_scale:.2f}, "
                    f"adjusted_velocity={linear_x_velocity:.3f}m/s"
                )
        
        # Update dynamic deadbands based on current velocities
        self._update_dynamic_deadbands(linear_x_velocity, linear_y_velocity, angular_velocity)
        
        # Log final velocity values
        if self.debug_level >= 1:
            self.get_logger().info(f"FINAL VELOCITY: lin_x={linear_x_velocity:.3f}, lin_y={linear_y_velocity:.3f}, ang_z={angular_velocity:.3f}")
        
        # Update reusable velocity command message (memory optimization)
        self._cmd_vel_msg.linear.x = linear_x_velocity    # Forward/backward
        self._cmd_vel_msg.linear.y = linear_y_velocity    # Left/right strafe
        self._cmd_vel_msg.angular.z = angular_velocity    # Rotation
        
        # Enhanced debugging to trace commands actually being sent
        if self.enable_debug_velocity_publish:
            self._log_throttled(
                self.get_logger().info,
                f"[#{self.cycle_count}] Velocity cmd: linear_x={linear_x_velocity:.3f}m/s, "
                f"linear_y={linear_y_velocity:.3f}m/s, "
                f"angular={angular_velocity:.3f}rad/s",
                0.2,  # Every 0.2 seconds max
                'last_velocity_publish_log_time'
            )
        
        # Save for history using the LightweightBuffer
        self.velocity_history.add((linear_x_velocity, linear_y_velocity, angular_velocity))
        
        # Publish command with pre-allocated message
        self.cmd_vel_pub.publish(self._cmd_vel_msg)
        
        # Publish basic diagnostics every cycle
        self.publish_basic_diagnostics(distance_error, lateral_error, angular_error,
                                    linear_x_velocity, linear_y_velocity, angular_velocity)
    
    def stop_robot(self):
        """Send a command to stop all robot motion immediately."""
        # Reuse cmd_vel message and just set all fields to 0
        self._cmd_vel_msg.linear.x = 0.0
        self._cmd_vel_msg.linear.y = 0.0
        self._cmd_vel_msg.angular.z = 0.0
        
        # Publish stop command multiple times to ensure it's received
        for _ in range(3):
            self.cmd_vel_pub.publish(self._cmd_vel_msg)
            time.sleep(0.01)  # Small delay between publishes
        
        # Debug logging for stop command
        if self.enable_debug_velocity_publish:
            self.get_logger().info("STOP COMMAND PUBLISHED: All velocities set to 0.0")
        
        # Reset last command velocity for acceleration limiting
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        
        # Clear velocity history
        self.velocity_history = LightweightBuffer(max_size=self.velocity_history_size)
        
        # Reset error trackers
        self.distance_error_tracker.reset()
        self.lateral_error_tracker.reset()
        self.angular_error_tracker.reset()
        
        # Reset PID controllers to clear any lingering integral terms
        self.pid_linear_x.reset()
        self.pid_linear_y.reset()
        self.pid_angular.reset()
        
        # Reset rotation tracking
        self.rotation_in_progress = False
        self.emergency_intervention_active = False
        self.consecutive_overrotations = 0
        
        # Reset dynamic deadbands to base values
        self.effective_distance_deadband = self.distance_deadband
        self.effective_lateral_deadband = self.lateral_deadband
        self.effective_angular_deadband = self.angular_deadband
        
        # Set a "stopped" state flag and timestamp
        self._robot_stopped = True
        self._stop_time = time.time()
        
        # Remember our position when we stopped
        self._last_stop_position = (
            self.current_distance,
            self.current_lateral, 
            self.current_bearing
        )
        
        self.get_logger().info("Robot stopped! All velocities and errors reset.")
    
    def publish_basic_diagnostics(self, distance_error, lateral_error, angular_error,
                              linear_x_velocity, linear_y_velocity, angular_velocity):
        """
        Publish basic diagnostic information for PID controllers.
        
        This is called at the full control loop rate and includes just
        the essential metrics for other nodes.
        """
        if self._shutting_down:
            return
            
        # Fill pre-allocated numpy array with diagnostic data
        self._diag_data[0] = distance_error
        self._diag_data[1] = lateral_error
        self._diag_data[2] = angular_error
        self._diag_data[3] = linear_x_velocity
        self._diag_data[4] = linear_y_velocity
        self._diag_data[5] = angular_velocity
        self._diag_data[6] = self.pid_linear_x.integral
        self._diag_data[7] = self.pid_linear_y.integral
        self._diag_data[8] = self.pid_angular.integral
        self._diag_data[9] = self.current_distance
        self._diag_data[10] = float(abs(distance_error) < self.stop_zone_size)
        
        # Update and publish pre-allocated message
        self._diag_msg.data = self._diag_data.tolist()  # Convert to Python list for ROS message
        self.pid_diag_pub.publish(self._diag_msg)
        
    def publish_detailed_diagnostics(self):
        """
        Publish comprehensive diagnostic information at a slower rate.
        
        This provides more detailed information for debugging and tuning,
        but at a lower frequency to avoid flooding the system.
        """
        if self._shutting_down or not self.robot_state == "tracking":
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
        max_lin_x_vel = max(abs(v) for v in lin_x_velocities) if lin_x_velocities else 0
        max_lin_y_vel = max(abs(v) for v in lin_y_velocities) if lin_y_velocities else 0
        max_ang_vel = max(abs(v) for v in ang_velocities) if ang_velocities else 0
        
        # Only log if time interval has passed (throttling)
        current_time = time.time()
        if (current_time - self.last_diag_log_time) < LOG_THROTTLE_DIAG:
            return
            
        self.last_diag_log_time = current_time
        
        # Log detailed information (only in debug level >= 1)
        if self.debug_level >= 1:
            self.get_logger().info("=== PID Detailed Diagnostics ===")
            self.get_logger().info(f"Target: distance={self.current_distance:.2f}m, lateral={self.current_lateral:.2f}m, bearing={math.degrees(self.current_bearing):.1f}°")
            self.get_logger().info(f"Linear X velocity: avg={avg_lin_x_vel:.2f}m/s, max={max_lin_x_vel:.2f}m/s")
            self.get_logger().info(f"Linear Y velocity: avg={avg_lin_y_vel:.2f}m/s, max={max_lin_y_vel:.2f}m/s")
            self.get_logger().info(f"Angular velocity: avg={avg_ang_vel:.2f}rad/s, max={max_ang_vel:.2f}rad/s")
            
            # Get PID components
            lin_x_p, lin_x_i, lin_x_d = self.pid_linear_x.get_components()
            lin_y_p, lin_y_i, lin_y_d = self.pid_linear_y.get_components()
            ang_p, ang_i, ang_d = self.pid_angular.get_components()
            
            self.get_logger().info(f"Linear X PID components: P={lin_x_p:.2f}, I={lin_x_i:.2f}, D={lin_x_d:.2f}")
            self.get_logger().info(f"Linear Y PID components: P={lin_y_p:.2f}, I={lin_y_i:.2f}, D={lin_y_d:.2f}")
            self.get_logger().info(f"Angular PID components: P={ang_p:.2f}, I={ang_i:.2f}, D={ang_d:.2f}")
            
            if self.adaptive_gains:
                self.get_logger().info(f"Adaptive gains: linear_x_kp={self.pid_linear_x.kp:.2f}, linear_y_kp={self.pid_linear_y.kp:.2f}, angular_kp={self.pid_angular.kp:.2f}")
            
            # Enhanced diagnostics for natural movement
            self.get_logger().info(f"Accumulated errors: distance={self.distance_error_tracker.accumulated_error:.3f}m-s, lateral={self.lateral_error_tracker.accumulated_error:.3f}m-s, angular={math.degrees(self.angular_error_tracker.accumulated_error):.1f}°-s")
            
            # Enhanced diagnostics for rotation protection
            if self.rotation_in_progress:
                time_since_start = current_time - self.rotation_start_time
                self.get_logger().info(f"Lookahead rotation in progress: {time_since_start:.1f}s elapsed, initial error={self.rotation_initial_error:.1f}°")
            if self.consecutive_overrotations > 0:
                self.get_logger().info(f"Overrotation protection: consecutive count={self.consecutive_overrotations}")
                
            # Dynamic deadband info
            self.get_logger().info(f"Dynamic deadbands: distance={self.effective_distance_deadband:.3f}m, lateral={self.effective_lateral_deadband:.3f}m, angular={self.effective_angular_deadband:.1f}°")
            
            # Control strategy
            self.get_logger().info(f"Current strategy: {self.current_strategy}, previous: {self.previous_strategy}")
            
            self.get_logger().info(f"Control cycles: {self.cycle_count}")
            self.get_logger().info("================================")

    def run_coordinate_frame_diagnostics(self):
        """Periodic function to check for coordinate frame issues."""
        if self._shutting_down or self.current_target is None:
            return

        # Only run full diagnostics if debug level >= 1 (to save CPU)
        if self.debug_level < 1:
            return
            
        # Only run full diagnostics if:
        # 1. We're in tracking mode (where it matters most)
        # 2. The frame appears to have changed 
        # 3. A minimum time has passed since last check
        current_frame = getattr(self.current_target.header, 'frame_id', None)
        frame_changed = current_frame != getattr(self, '_last_diagnostics_frame', None)
        
        current_time = time.time()
        time_passed = (current_time - getattr(self, 'last_frame_check_time', 0)) >= 1.0/self.frame_check_rate
        
        if (self.robot_state == "tracking" and (frame_changed or time_passed)):
            # Update tracking variables
            self._last_diagnostics_frame = current_frame
            self.last_frame_check_time = current_time
            
            # Run the diagnostics
            diagnostics = self.handle_coordinate_frame_diagnostics()
            
            # Only log warnings if we detected an issue
            if diagnostics.get("detected_issue", False):
                self.get_logger().warn(f"Possible coordinate frame issue detected - {diagnostics}")
            
    def handle_coordinate_frame_diagnostics(self):
        """
        Special diagnostic function to detect and report coordinate frame issues.
        
        This function analyzes the latest target data and logs detailed information
        about potential coordinate frame mismatches or interpretation issues.
        """
        if self._shutting_down or self.current_target is None:
            return {}
            
        target = self.current_target.point
        frame_id = getattr(self.current_target.header, 'frame_id', 'unknown')
        
        # Calculate distances in different ways to detect coordinate mismatches
        # Full 3D distance
        dist_3d = math.sqrt(target.x**2 + target.y**2 + target.z**2)
        
        # Compare the different calculations to how we're interpreting in target_callback
        # If these vary widely, we may have a coordinate frame issue
        
        frame_mismatch_detected = False
        
        # Check if our interpretation differs significantly from other interpretations
        if abs(self.current_distance - dist_3d) > 0.1:
            frame_mismatch_detected = True
        
        # Check which axis seems to be the primary distance component
        primary_axis = "unknown"
        dist_x = abs(target.x)
        dist_y = abs(target.y)
        dist_z = abs(target.z)
        
        if dist_x > dist_y and dist_x > dist_z:
            primary_axis = "x"
        elif dist_y > dist_x and dist_y > dist_z:
            primary_axis = "y"
        elif dist_z > dist_x and dist_z > dist_y:
            primary_axis = "z"
            
        # Our interpretation vs what we calculated
        frame_diagnostics = {
            "detected_issue": frame_mismatch_detected,
            "target_frame": frame_id,
            "raw_position": [round(target.x, 2), round(target.y, 2), round(target.z, 2)],
            "3d_distance": round(dist_3d, 2),
            "our_interpretation": {
                "distance": round(self.current_distance, 2),
                "lateral": round(self.current_lateral, 2),
                "bearing_deg": round(math.degrees(self.current_bearing), 1)
            },
            "primary_axis": primary_axis
        }
        
        return frame_diagnostics

    def prepare_shutdown(self):
        """
        Prepare for node shutdown.
        
        This method ensures the robot is stopped and sets a flag to
        prevent further callbacks during shutdown.
        """
        self.get_logger().info("Preparing for shutdown")
        
        # Immediately stop the robot using the stop_robot method
        try:
            self.stop_robot()
            self.get_logger().info("Robot motion stopped - velocity and rotation set to 0")
        except Exception as e:
            self.get_logger().error(f"Error stopping robot during shutdown: {str(e)}")
            
        # Set shutdown flag after stopping to prevent further actions
        self._shutting_down = True

# Main function outside of class definition
def main(args=None):
    """Main function to initialize and run the PID Controller node."""
    rclpy.init(args=args)
    node = EnhancedPIDControllerNode()
    
    # Welcome message
    print("=================================================")
    print("Enhanced PID Controller for Basketball Tracking Robot")
    print("=================================================")
    print("This node implements three PID controllers with natural movement behaviors:")
    print("1. Linear X velocity (forward/backward movement)")
    print("2. Linear Y velocity (lateral/strafing movement)")
    print("3. Angular velocity (turning/rotation)")
    print("")
    print("Enhanced Movement Features:")
    print("- Lookahead control for rotation to prevent overshooting")
    print("- Overrotation protection to detect and correct errors")
    print("- Coordinated lateral and angular movements")
    print("- Strategy persistence to prevent thrashing")
    print("- Drift detection and correction after stopping")
    print("- Dynamic deadbands to prevent oscillation")
    print("- Resource-efficient implementation for Raspberry Pi")
    print("")
    print("Subscriptions:")
    for name, topic in TOPICS["input"].items():
        print(f"  - {name:<15}: {topic}")
    print("")
    print("Publications:")
    for name, topic in TOPICS["output"].items():
        print(f"  - {name:<15}: {topic}")
    print("")
    print("Press Ctrl+C to stop the program")
    print("=================================================")
    
    # Define shutdown handler to ensure robot stops on any exit
    def shutdown_handler():
        node.prepare_shutdown()
        node.destroy_node()
    
    # Register shutdown handler
    rclpy.get_global_executor().add_node(node)
    rclpy.get_default_context().on_shutdown(shutdown_handler)
    
    # Register signal handlers for proper shutdown
    def signal_handler(sig, frame):
        print(f"\nSignal {sig} received, stopping robot...")
        # First prepare for shutdown to stop the robot
        node.prepare_shutdown()
        # Then proceed with ROS shutdown
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("PID Controller shutdown requested via Ctrl+C")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {str(e)}")
    finally:
        # Explicitly stop the robot before shutdown
        node.prepare_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()