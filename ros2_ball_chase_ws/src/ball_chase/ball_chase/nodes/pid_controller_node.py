"""""
Basketball Tracking Robot - PID Controller Node
==============================================

This controller implements efficient movement patterns for a mecanum-wheeled
basketball tracking robot using a table-driven approach:
- Simplified state logic with discrete movement states
- Clear separation of control and decision logic
- Hysteresis for stable behavior near state boundaries
- Optimized for Raspberry Pi 5 performance
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
LOG_THROTTLE_CONTROL = 2.0     # Seconds between control loop status logs
LOG_THROTTLE_STATE = 0.5       # Seconds between state change logs
LOG_THROTTLE_DIAG = 1.0        # Seconds between diagnostic logs

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

class ErrorTracker:
    """Lightweight error tracker that monitors error values over time."""
    
    def __init__(self, name, max_history=8):
        """Initialize error tracker with efficient storage."""
        self.name = name
        self.current_error = 0.0
        self.previous_error = 0.0
        # Pre-allocate error history with zeros to avoid frequent allocations
        self.error_history = deque([0.0] * max_history, maxlen=max_history)
        self.error_history_index = 0
        self.error_history_count = 0
        self.error_history_size = max_history
        self.last_correction_time = 0.0
        self.sign_changes = 0  # Count of error sign changes (useful for oscillation detection)
        self.accumulated_error = 0.0
        self.decay_factor = 0.9  # Simplified decay factor
        
    def update(self, error, dt):
        """Update error tracking with new error value."""
        # Check for sign change
        if self.current_error != 0 and error != 0 and (self.current_error * error) < 0:
            self.sign_changes += 1
            
        # Update error history using in-place modification
        self.previous_error = self.current_error
        self.current_error = error
        
        # Update error history with in-place modification
        self.error_history[self.error_history_index] = error
        self.error_history_index = (self.error_history_index + 1) % self.error_history_size
        if self.error_history_count < self.error_history_size:
            self.error_history_count += 1
        
        # Update accumulated error with decay
        self.accumulated_error = (self.accumulated_error + error * dt) * self.decay_factor
    
    def reset(self):
        """Reset all tracked errors."""
        self.current_error = 0.0
        self.previous_error = 0.0
        # Reset history indices rather than allocating new memory
        self.error_history_index = 0
        self.error_history_count = 0
        # Zero out the existing history array instead of clearing and reallocating
        for i in range(len(self.error_history)):
            self.error_history[i] = 0.0
        self.accumulated_error = 0.0
        self.sign_changes = 0
    
    def is_error_growing(self):
        """Check if error is growing compared to previous value."""
        return abs(self.current_error) > abs(self.previous_error) * 1.05  # 5% threshold
        
    def record_correction(self):
        """Record that a correction was made for this error."""
        self.last_correction_time = time.time()
        # Reduce accumulated error when correction is made
        self.accumulated_error *= 0.5

class PIDController:
    """Efficient PID controller implementation with enhanced anti-windup protection."""
    
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
        self.last_output = 0.0 # Last output value (for transition smoothing)
        
        # Diagnostic information
        self.last_p_term = 0.0  # Last proportional term
        self.last_i_term = 0.0  # Last integral term
        self.last_d_term = 0.0  # Last derivative term
        
        # Performance optimization
        self.integral_deadband = 0.05  # Only accumulate integral when error is significant
        self.integral_decay = 0.7      # Decay rate for integral when error is small
        self.max_integral = (output_max - output_min) / ki if ki > 0 else 1.0  # Prevent excessive integral buildup
        
    def reset(self):
        """Reset controller state, restarting from scratch."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.last_output = 0.0
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        
    def compute(self, error, current_time=None, force_zero=False):
        """Compute the control output based on the error."""
        try:
            # If forcing zero output, bypass calculations
            if force_zero:
                # Reset derivative term when forcing zero to prevent future spikes
                self.prev_error = error
                # Keep integral to avoid losing accumulated correction
                
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
                # P-only control on first iteration (no I or D)
                output = self.kp * error
                self.last_p_term = output
                self.last_output = max(self.output_min, min(self.output_max, output))
                return float(self.last_output)
                
            # Calculate dt (time since last update)
            dt = current_time - self.last_time
            if dt <= 0.001:  # Protect against very small or negative dt
                dt = 0.01  # Fallback to prevent division by zero (assume 100Hz)
                
            # Calculate each PID term
            # Proportional term (proportional to error)
            p_term = self.kp * error
            
            # Improved integral term handling with better anti-windup
            # Detect if output is likely to saturate
            predicted_output = p_term + self.last_i_term + self.last_d_term
            is_saturated = (predicted_output >= self.output_max) or (predicted_output <= self.output_min)
            
            if self.anti_windup and is_saturated:
                # Don't accumulate integral when saturated
                i_term = self.last_i_term
            else:
                # Only accumulate integral when error is significant
                if abs(error) > self.integral_deadband:
                    self.integral += error * dt
                else:
                    # More aggressively reduce integral term when close to target
                    self.integral *= self.integral_decay
                
                # Apply integral limit to prevent excessive buildup
                self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
                
                i_term = self.ki * self.integral
            
            # Derivative term with improved noise handling
            # Use filtered error derivative to reduce noise sensitivity
            error_change = error - self.prev_error
            d_term = self.kd * error_change / max(dt, 0.001)  # Protect against division by zero
            
            # Calculate raw output by summing all terms
            output = p_term + i_term + d_term
            
            # Apply output limits
            output_limited = max(self.output_min, min(self.output_max, output))
            
            # Improved anti-windup logic for when we exceed limits
            if self.anti_windup and output != output_limited:
                # Back-calculate integral to match limited output
                # This adjusts the integral term to account for saturation
                if abs(self.ki) > 1e-10:  # Avoid division by zero
                    self.integral = (output_limited - p_term - d_term) / self.ki
                    # Recalculate integral term
                    i_term = self.ki * self.integral
                    
            # Additional anti-windup when error changes sign
            if error * self.prev_error < 0 and abs(error) < abs(self.prev_error):
                # Error crossed zero and is decreasing - reduce integral more aggressively
                self.integral *= 0.5
                i_term = self.ki * self.integral
                    
            # Apply transition smoothing for rapid control changes
            # This helps reduce jerky movements when control mode changes
            if abs(output_limited - self.last_output) > (self.output_max - self.output_min) * 0.5:
                # Blend between previous and current output for large changes
                output_limited = 0.7 * output_limited + 0.3 * self.last_output
                
            # Save individual terms for diagnostics
            self.last_p_term = p_term
            self.last_i_term = i_term
            self.last_d_term = d_term
            self.last_output = output_limited
            
            # Save state for next iteration
            self.prev_error = error
            self.last_time = current_time
            
            # Ensure we return a proper float value
            return float(output_limited)
            
        except Exception as e:
            self.get_logger().error(f"Error in PID compute: {str(e)}")
            return 0.0  # Return a safe default value
        
    def get_components(self):
        """Get the last calculated PID components."""
        return (self.last_p_term, self.last_i_term, self.last_d_term)

class PIDControllerNode(Node):
    """PID Controller node for basketball tracking with table-driven state logic."""
    
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
        
        # Define movement strategies using table-driven approach
        self._init_strategy_table()
        
        # Log startup info (just once)
        self.get_logger().info("PID Controller initialized with table-driven state logic")
        
    def _declare_parameters(self):
        """Declare and get all node parameters."""
        self.declare_parameters(
            namespace='',
            parameters=[
                # Linear X velocity PID parameters
                ('linear_x_kp', 1.0),
                ('linear_x_ki', 0.03),
                ('linear_x_kd', 0.1),
                ('linear_x_min', 0.0),
                ('linear_x_max', 0.2),
                
                # Linear Y velocity PID parameters
                ('linear_y_kp', 0.8),
                ('linear_y_ki', 0.15),
                ('linear_y_kd', 0.05),
                ('linear_y_min', -0.2),
                ('linear_y_max', 0.2),
                
                # Angular velocity PID parameters
                ('angular_kp', 3.0),
                ('angular_ki', 0.1),
                ('angular_kd', 0.4),
                ('angular_min', -0.5),
                ('angular_max', 0.5),
                
                # Control parameters
                ('min_distance', 0.9),
                ('max_distance', 2.0),
                ('target_offset_x', 0.0),
                ('target_offset_y', 0.0),
                ('target_update_rate', 10.0),
                ('diagnostics_rate', 0.5),
                ('debug_level', 0),
                ('adaptive_gains', True),
                ('use_lateral_control', True),
                
                # Movement parameters
                ('min_effective_velocity', 0.05),
                ('max_allowed_velocity', 2.0),
                ('min_angular_velocity', 0.1),
                
                # Deadband parameters
                ('distance_deadband', 0.08),
                ('lateral_deadband', 0.08),
                ('angular_deadband', 5.0),
                
                # Stop parameters
                ('stop_zone_size', 0.2),
                ('safety_min_distance', 0.4),
                ('complete_stop_time', 3.0),
                ('stop_hysteresis_factor', 1.5),
                
                # Acceleration parameters
                ('max_accel', 0.6),
                ('max_angular_accel', 1.0),
                ('accel_boost_factor', 3.0),
                
                # Target age parameters
                ('target_max_age', 0.5),
                ('drift_detection_threshold', 0.2),
                
                # Error categorization thresholds
                ('error_small_factor', 1.0),        # Error is "small" if > deadband * this factor
                ('error_medium_factor', 2.0),       # Error is "medium" if > deadband * this factor
                ('error_large_factor', 4.0),        # Error is "large" if > deadband * this factor
                
                # Stop condition threshold factors
                ('stop_lateral_factor', 1.5),       # Lateral threshold = lateral_deadband * this factor
                ('stop_angular_factor', 1.5),       # Angular threshold = angular_deadband * this factor
                
                # Strategy selection parameters
                ('strategy_wildcard_enabled', True), # Whether to use wildcard matching for strategies
                ('strategy_log_level', 1),          # 0=none, 1=changes only, 2=all selections
                
                # Velocity logging threshold
                ('velocity_change_threshold', 0.05), # Log velocity changes greater than this
            ]
        )
        
        # Get all parameters
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
        self.debug_level = self.get_parameter('debug_level').value
        self.adaptive_gains = self.get_parameter('adaptive_gains').value
        self.use_lateral_control = self.get_parameter('use_lateral_control').value
        
        self.min_effective_velocity = self.get_parameter('min_effective_velocity').value
        self.max_allowed_velocity = self.get_parameter('max_allowed_velocity').value
        self.min_angular_velocity = self.get_parameter('min_angular_velocity').value
        
        self.distance_deadband = self.get_parameter('distance_deadband').value
        self.lateral_deadband = self.get_parameter('lateral_deadband').value
        self.angular_deadband = self.get_parameter('angular_deadband').value
        
        self.stop_zone_size = self.get_parameter('stop_zone_size').value
        self.safety_min_distance = self.get_parameter('safety_min_distance').value
        self.complete_stop_time = self.get_parameter('complete_stop_time').value
        self.stop_hysteresis_factor = self.get_parameter('stop_hysteresis_factor').value
        
        self.max_accel = self.get_parameter('max_accel').value
        self.max_angular_accel = self.get_parameter('max_angular_accel').value
        self.accel_boost_factor = self.get_parameter('accel_boost_factor').value
        
        self.target_max_age = self.get_parameter('target_max_age').value
        self.drift_detection_threshold = self.get_parameter('drift_detection_threshold').value
        
        # Error categorization thresholds
        self.error_small_factor = self.get_parameter('error_small_factor').value
        self.error_medium_factor = self.get_parameter('error_medium_factor').value
        self.error_large_factor = self.get_parameter('error_large_factor').value
        
        # Stop condition threshold factors
        self.stop_lateral_factor = self.get_parameter('stop_lateral_factor').value
        self.stop_angular_factor = self.get_parameter('stop_angular_factor').value
        
        # Strategy selection parameters
        self.strategy_wildcard_enabled = self.get_parameter('strategy_wildcard_enabled').value
        self.strategy_log_level = self.get_parameter('strategy_log_level').value
        
        # Velocity logging threshold
        self.velocity_change_threshold = self.get_parameter('velocity_change_threshold').value
        
    def _init_controllers(self):
        """Initialize the PID controllers."""
        self.pid_linear_x = PIDController(
            self.linear_x_kp, self.linear_x_ki, self.linear_x_kd,
            self.linear_x_min, self.linear_x_max,
            name="Linear X"
        )
        
        self.pid_linear_y = PIDController(
            self.linear_y_kp, self.linear_y_ki, self.linear_y_kd,
            self.linear_y_min, self.linear_y_max,
            name="Linear Y"
        )
        
        self.pid_angular = PIDController(
            self.angular_kp, self.angular_ki, self.angular_kd,
            self.angular_min, self.angular_max,
            name="Angular"
        )
        
    def _init_reusable_messages(self):
        """Pre-allocate ROS messages for reuse to avoid memory churn."""
        self._cmd_vel_msg = Twist()
        self._diag_msg = Float32MultiArray()
        self._diag_data = np.zeros(11, dtype=np.float32)
        
    def _init_state_variables(self):
        """Initialize all state tracking variables."""
        # Target tracking
        self.current_target = None
        self.last_target_time = None
        
        # Robot state
        self.robot_state = "initializing"
        self.previous_state = None
        self.last_control_time = time.time()
        
        # Log throttling timestamps
        self.last_control_log_time = 0.0
        self.last_state_log_time = 0.0
        self.last_diag_log_time = 0.0
        self.last_status_log_time = 0.0
        
        # Derived values
        self.current_distance = 0.0
        self.current_bearing = 0.0
        self.current_lateral = 0.0
        
        # Motion smoothing
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        self.last_logged_cmd = (0.0, 0.0, 0.0)
        
        # Diagnostic information
        self.cycle_count = 0
        
        # Velocity history
        self.velocity_history = LightweightBuffer(max_size=6)
        
        # Cached values
        self.desired_distance = self.min_distance + self.target_offset_x
        
        # Error tracking
        self.distance_error_tracker = ErrorTracker("distance", max_history=8)
        self.lateral_error_tracker = ErrorTracker("lateral", max_history=8)
        self.angular_error_tracker = ErrorTracker("angular", max_history=8)
        
        # Stopped state tracking
        self._robot_stopped = False
        self._stop_time = 0.0
        self._last_stop_position = (0.0, 0.0, 0.0)
        
        # Movement strategy
        self.current_strategy = "IDLE"
        self.previous_strategy = None
        self.strategy_change_time = time.time()
        
    def _init_strategy_table(self):
        """
        Initialize the table-driven movement strategy definitions.
        This table defines how the robot should move based on different error states.
        """
        # Define the strategy table
        # Format: (distance_state, lateral_state, angular_state): [
        #    name, use_forward, use_lateral, use_angular, 
        #    forward_scale, lateral_scale, angular_scale, reason_template
        # ]
        self.strategy_table = {
            # All errors within deadbands - no movement
            ("none", "none", "none"): [
                "NO_MOVEMENT", False, False, False, 
                0.0, 0.0, 0.0, 
                "All errors within deadbands"
            ],
            
            # Single dimension errors - focused corrections
            ("small", "none", "none"): [
                "FORWARD_ONLY", True, False, False, 
                0.8, 0.0, 0.0, 
                "Small distance error correction: {distance_error:.2f}m"
            ],
            ("medium", "none", "none"): [
                "FORWARD_ONLY", True, False, False, 
                1.0, 0.0, 0.0, 
                "Medium distance error correction: {distance_error:.2f}m"
            ],
            ("large", "none", "none"): [
                "FORWARD_ONLY", True, False, False, 
                1.0, 0.0, 0.0, 
                "Large distance error correction: {distance_error:.2f}m"
            ],
            
            ("none", "small", "none"): [
                "LATERAL_ONLY", False, True, False, 
                0.0, 0.8, 0.0, 
                "Small lateral error correction: {lateral_error:.2f}m"
            ],
            ("none", "medium", "none"): [
                "LATERAL_ONLY", False, True, False, 
                0.0, 1.0, 0.0, 
                "Medium lateral error correction: {lateral_error:.2f}m"
            ],
            ("none", "large", "none"): [
                "LATERAL_ONLY", False, True, False, 
                0.0, 1.0, 0.0, 
                "Large lateral error correction: {lateral_error:.2f}m"
            ],
            
            ("none", "none", "small"): [
                "ANGULAR_ONLY", False, False, True, 
                0.0, 0.0, 0.8, 
                "Small angular error correction: {angular_error:.1f}°"
            ],
            ("none", "none", "medium"): [
                "ANGULAR_ONLY", False, False, True, 
                0.0, 0.0, 1.0, 
                "Medium angular error correction: {angular_error:.1f}°"
            ],
            ("none", "none", "large"): [
                "ANGULAR_ONLY", False, False, True, 
                0.0, 0.0, 1.0, 
                "Large angular error correction: {angular_error:.1f}°"
            ],
            
            # Large angular error takes precedence over others
            ("*", "*", "large"): [
                "ANGULAR_PRIMARY", False, False, True, 
                0.0, 0.0, 1.0, 
                "Angular correction prioritized: {angular_error:.1f}°"
            ],
            
            # Combined movement strategies - coordinated approaches
            ("small", "small", "none"): [
                "COORDINATED", True, True, False, 
                0.8, 0.8, 0.0, 
                "Coordinated distance and lateral correction"
            ],
            ("small", "none", "small"): [
                "COORDINATED", True, False, True, 
                0.8, 0.0, 0.8, 
                "Coordinated distance and angular correction"
            ],
            ("none", "small", "small"): [
                "COORDINATED", False, True, True, 
                0.0, 0.8, 0.8, 
                "Coordinated lateral and angular correction"
            ],
            
            # Medium-to-large lateral with other errors
            ("*", "medium", "small"): [
                "LATERAL_PRIMARY", True, True, True, 
                0.4, 1.0, 0.3, 
                "Lateral-focused coordination with small angular correction"
            ],
            ("*", "large", "small"): [
                "LATERAL_PRIMARY", True, True, True, 
                0.3, 1.0, 0.3, 
                "Lateral-primary movement with small angular correction"
            ],
            
            # Medium-to-large distance with other errors
            ("medium", "*", "small"): [
                "FORWARD_PRIMARY", True, True, True, 
                1.0, 0.4, 0.3, 
                "Forward-focused coordination with small corrections"
            ],
            ("large", "*", "small"): [
                "FORWARD_PRIMARY", True, True, True, 
                1.0, 0.3, 0.3, 
                "Forward-primary movement with small corrections"
            ],
            
            # Medium angular with other errors
            ("small", "small", "medium"): [
                "ANGULAR_PRIMARY", True, True, True, 
                0.3, 0.3, 1.0, 
                "Angular-focused coordination with small corrections"
            ],
            
            # Fallback strategy
            ("*", "*", "*"): [
                "BALANCED", True, True, True, 
                0.6, 0.6, 0.6, 
                "Balanced movement strategy (fallback)"
            ]
        }
        
    def _setup_subscriptions(self):
        """Set up all subscriptions for this node."""
        self.state_sub = self.create_subscription(
            String,
            TOPICS["input"]["state"],
            self.state_callback,
            10
        )
        
        self.target_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["target"],
            self.target_callback,
            10
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
        
    def _setup_timers(self):
        """Set up timer callbacks for periodic tasks with tiered frequencies."""
        self.timer = self.create_timer(1.0 / self.update_rate, self.control_loop_callback)
        self.diagnostic_timer = self.create_timer(1.0 / self.diagnostics_rate, self.publish_diagnostics)
        
    def _log_throttled(self, level_func, message, min_interval, last_time_attr):
        """Log messages with throttling to reduce log volume."""
        current_time = time.time()
        last_time = getattr(self, last_time_attr, 0)
        
        if current_time - last_time >= min_interval:
            level_func(message)
            setattr(self, last_time_attr, current_time)
            return True
            
        return False
    
    def state_callback(self, msg):
        """Handle robot state updates from the state manager."""
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
                
            # If we're not in tracking mode, ensure the robot is stopped
            # (unless it's in searching or lost_ball mode, where the state manager controls motion)
            if new_state != "tracking" and new_state != "searching" and new_state != "lost_ball":
                self.stop_robot()
    
    def target_callback(self, msg):
        """Handle target position updates from the state manager."""
        if self._shutting_down:
            return
        
        # Update timestamps
        self.last_target_time = time.time()
        self.current_target = msg
        
        # Extract key information from target
        target = msg.point
        
        # Calculate full 2D distance to target
        self.current_distance = math.sqrt(target.x**2 + target.y**2 )
        
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
            
        # Check for safety distance
        if self.current_distance < self.safety_min_distance:
            self.get_logger().warn(
                f"Ball very close to robot! Distance={self.current_distance:.2f}m, "
                f"safety threshold={self.safety_min_distance:.2f}m"
            )
            
    def _categorize_error(self, error, error_type="distance"):
        """
        Categorize an error value into none, small, medium, or large.
        
        Args:
            error: The error value to categorize
            error_type: The type of error (distance, lateral, angular)
            
        Returns:
            String: The error category (none, small, medium, large)
        """
        abs_error = abs(error)
        
        # Select appropriate deadband and thresholds based on error type
        if error_type == "angular":
            deadband = self.angular_deadband
            small_threshold = deadband * self.error_small_factor
            medium_threshold = deadband * self.error_medium_factor
            large_threshold = deadband * self.error_large_factor
        elif error_type == "lateral":
            deadband = self.lateral_deadband
            small_threshold = deadband * self.error_small_factor
            medium_threshold = deadband * self.error_medium_factor
            large_threshold = deadband * self.error_large_factor
        else:  # distance
            deadband = self.distance_deadband
            small_threshold = deadband * self.error_small_factor
            medium_threshold = deadband * self.error_medium_factor
            large_threshold = deadband * self.error_large_factor
        
        # Handle case where lateral control is disabled
        if error_type == "lateral" and not self.use_lateral_control:
            return "none"
        
        # Categorize based on thresholds
        if abs_error <= deadband:
            return "none"
        elif abs_error <= small_threshold:
            return "small"
        elif abs_error <= medium_threshold:
            return "medium"
        else:
            return "large"
    
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
        
        # If wildcards are disabled, use fallback
        if not self.strategy_wildcard_enabled:
            return strategies[("*", "*", "*")]
        
        # Try wildcard matches with progressively fewer specifics
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
                return strategies[pattern]
        
        # Fallback - should never reach here if table has a ("*", "*", "*") entry
        return strategies[("*", "*", "*")]
    
    def _populate_strategy(self, strategy_dict, strategy_def, distance_error, lateral_error, angular_error):
        """
        Populate a pre-allocated strategy dictionary with values from strategy definition.
        
        Args:
            strategy_dict: Pre-allocated strategy dictionary to populate
            strategy_def: Strategy definition from table
            distance_error: Current distance error
            lateral_error: Current lateral error
            angular_error: Current angular error in degrees
        """
        # Unpack strategy definition
        name, use_forward, use_lateral, use_angular, forward_scale, lateral_scale, angular_scale, reason_template = strategy_def
        
        # Format the reason string with actual error values
        reason = reason_template.format(
            distance_error=abs(distance_error),
            lateral_error=abs(lateral_error),
            angular_error=abs(angular_error)
        )
        
        # Populate the strategy dictionary
        strategy_dict["strategy_name"] = name
        strategy_dict["use_forward"] = use_forward
        strategy_dict["use_lateral"] = use_lateral and self.use_lateral_control
        strategy_dict["use_angular"] = use_angular
        strategy_dict["forward_scale"] = forward_scale
        strategy_dict["lateral_scale"] = lateral_scale
        strategy_dict["angular_scale"] = angular_scale
        strategy_dict["reason"] = reason
    
    def _init_state_variables(self):
        """Initialize all state tracking variables."""
        # Target tracking
        self.current_target = None
        self.last_target_time = None
        
        # Robot state
        self.robot_state = "initializing"
        self.previous_state = None
        self.last_control_time = time.time()
        
        # Log throttling timestamps
        self.last_control_log_time = 0.0
        self.last_state_log_time = 0.0
        self.last_diag_log_time = 0.0
        self.last_status_log_time = 0.0
        
        # Derived values
        self.current_distance = 0.0
        self.current_bearing = 0.0
        self.current_lateral = 0.0
        
        # Motion smoothing
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        self.last_logged_cmd = (0.0, 0.0, 0.0)
        
        # Diagnostic information
        self.cycle_count = 0
        
        # Velocity history
        self.velocity_history = LightweightBuffer(max_size=6)
        
        # Cached values
        self.desired_distance = self.min_distance + self.target_offset_x
        
        # Error tracking
        self.distance_error_tracker = ErrorTracker("distance", max_history=8)
        self.lateral_error_tracker = ErrorTracker("lateral", max_history=8)
        self.angular_error_tracker = ErrorTracker("angular", max_history=8)
        
        # Stopped state tracking
        self._robot_stopped = False
        self._stop_time = 0.0
        self._last_stop_position = (0.0, 0.0, 0.0)
        
        # Movement strategy
        self.current_strategy = "IDLE"
        self.previous_strategy = None
        self.strategy_change_time = time.time()
        
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
        self._key_tuple = ["none", "none", "none"]  # Will be modified in-place and used as key
        
    def _determine_movement_strategy(self, distance_error, lateral_error, angular_error_degrees):
        """
        Determine the optimal movement strategy using table-driven approach.
        
        Args:
            distance_error: Error in distance (meters)
            lateral_error: Error in lateral position (meters)
            angular_error_degrees: Error in angular position (degrees)
            
        Returns:
            dict: Strategy information including strategy name, movement flags, and scale factors
        """
        # FIXED: If we just exited stopped state, use an immediate response strategy
        if (not self._robot_stopped and 
            (time.time() - self._stop_time) < 0.5):  # Within 0.5 seconds of exiting stopped state
            
            # Use a more responsive strategy when resuming from stop
            urgent_strategy = {
                "strategy_name": "RESUME_TRACKING",
                "use_forward": abs(distance_error) > self.distance_deadband,
                "use_lateral": self.use_lateral_control and abs(lateral_error) > self.lateral_deadband,
                "use_angular": abs(angular_error_degrees) > self.angular_deadband,
                "forward_scale": 1.0,
                "lateral_scale": 1.0,
                "angular_scale": 1.0,
                "reason": f"Resuming tracking after stop: dist_err={distance_error:.2f}m, lat_err={lateral_error:.2f}m, ang_err={angular_error_degrees:.1f}°"
            }
            return urgent_strategy
        
        # Reuse pre-allocated strategy dict
        strategy = self._strategy_dict
        
        # Categorize errors into states: "none", "small", "medium", "large"
        # Reuse pre-allocated tuple for key
        self._key_tuple[0] = self._categorize_error(distance_error, "distance")
        self._key_tuple[1] = self._categorize_error(lateral_error, "lateral")
        self._key_tuple[2] = self._categorize_error(angular_error_degrees, "angular")
        
        # Create lookup key (convert to tuple for dictionary lookup)
        key = tuple(self._key_tuple)
        
        # Get strategy definition from table
        strategy_def = self._match_strategy(key, self.strategy_table)
        
        # Populate the strategy dict
        self._populate_strategy(strategy, strategy_def, distance_error, lateral_error, angular_error_degrees)
        
        return strategy
    
    def _init_reusable_messages(self):
        """Pre-allocate ROS messages for reuse to avoid memory churn."""
        self._cmd_vel_msg = Twist()
        self._diag_msg = Float32MultiArray()
        self._diag_data = np.zeros(11, dtype=np.float32)
        
        # Pre-allocated arrays for velocity processing
        self._limited_velocities = np.zeros(3, dtype=np.float32)  # For storing limited velocities
        self._prev_velocities = np.zeros(3, dtype=np.float32)     # For storing previous velocities
        self._target_velocities = np.zeros(3, dtype=np.float32)   # For storing target velocities
        self._vel_diffs = np.zeros(3, dtype=np.float32)           # For storing velocity differences
        
    def _apply_velocity_limits(self, linear_x, linear_y, angular_z, current_time):
        """
        Apply velocity and acceleration limits for smooth, natural movement.
        
        Args:
            linear_x: Calculated forward velocity
            linear_y: Calculated lateral velocity
            angular_z: Calculated angular velocity
            current_time: Current time for acceleration limiting
            
        Returns:
            tuple: (limited_linear_x, limited_linear_y, limited_angular_z)
        """

        if not isinstance(self.last_cmd_vel, tuple):
            self.get_logger().warn(f"last_cmd_vel is not a tuple: {type(self.last_cmd_vel)}")
            self.last_cmd_vel = (0.0, 0.0, 0.0)

        # Store target velocities in pre-allocated array
        self._target_velocities[0] = linear_x
        self._target_velocities[1] = linear_y
        self._target_velocities[2] = angular_z
        
        # Store previous velocities in pre-allocated array
        self._prev_velocities[0] = self.last_cmd_vel[0]  # x
        self._prev_velocities[1] = self.last_cmd_vel[1]  # y
        self._prev_velocities[2] = self.last_cmd_vel[2]  # angular
        
        # Calculate time since last control step
        dt = current_time - getattr(self, "last_accel_time", current_time - 0.1)
        self.last_accel_time = current_time
        
        # Ensure dt is reasonable
        dt = max(0.001, min(dt, 0.1))
        
        # Scale acceleration limit by time
        accel_limit = self.max_accel * dt * 10.0
        angular_accel_limit = self.max_angular_accel * dt * 10.0
        
        # Calculate velocity differences in-place
        np.subtract(self._target_velocities, self._prev_velocities, out=self._vel_diffs)
        
        # Apply acceleration limits with vectorized operations where possible
        
        # Forward velocity
        if abs(self._vel_diffs[0]) > accel_limit:
            # Apply boost factor when starting from stop
            boost = self.accel_boost_factor if abs(self._prev_velocities[0]) < 0.01 and abs(self._target_velocities[0]) > 0.01 else 1.0
            
            # Calculate limited velocity
            self._limited_velocities[0] = self._prev_velocities[0] + math.copysign(
                min(abs(self._vel_diffs[0]), accel_limit * boost), 
                self._vel_diffs[0]
            )
        else:
            self._limited_velocities[0] = self._target_velocities[0]
            
        # Lateral velocity
        if abs(self._vel_diffs[1]) > accel_limit:
            # Apply boost factor when starting from stop
            boost = self.accel_boost_factor if abs(self._prev_velocities[1]) < 0.01 and abs(self._target_velocities[1]) > 0.01 else 1.0
            
            # Calculate limited velocity
            self._limited_velocities[1] = self._prev_velocities[1] + math.copysign(
                min(abs(self._vel_diffs[1]), accel_limit * boost), 
                self._vel_diffs[1]
            )
        else:
            self._limited_velocities[1] = self._target_velocities[1]
            
        # Angular velocity
        if abs(self._vel_diffs[2]) > angular_accel_limit:
            # Apply boost factor when starting from stop
            boost = self.accel_boost_factor if abs(self._prev_velocities[2]) < 0.01 and abs(self._target_velocities[2]) > 0.01 else 1.0
            
            # Calculate limited velocity
            self._limited_velocities[2] = self._prev_velocities[2] + math.copysign(
                min(abs(self._vel_diffs[2]), angular_accel_limit * boost), 
                self._vel_diffs[2]
            )
        else:
            self._limited_velocities[2] = self._target_velocities[2]
        
        # Apply minimum velocity thresholds (to avoid ineffective small movements)
        if abs(self._limited_velocities[0]) < self.min_effective_velocity:
            self._limited_velocities[0] = 0.0
            
        if abs(self._limited_velocities[1]) < self.min_effective_velocity:
            self._limited_velocities[1] = 0.0
            
        if abs(self._limited_velocities[2]) < self.min_angular_velocity:
            self._limited_velocities[2] = 0.0
            
        # Apply maximum velocity limits
        self._limited_velocities[0] = max(-self.linear_x_max, min(self.linear_x_max, self._limited_velocities[0]))
        self._limited_velocities[1] = max(self.linear_y_min, min(self.linear_y_max, self._limited_velocities[1]))
        self._limited_velocities[2] = max(self.angular_min, min(self.angular_max, self._limited_velocities[2]))
        
        return (self._limited_velocities[0], self._limited_velocities[1], self._limited_velocities[2])

    def _init_reusable_objects(self):
        """Initialize additional reusable objects for the control loop"""
        # Pre-allocated velocity tuple to avoid creating tuples in hot loop
        self._velocity_tuple = [0.0, 0.0, 0.0]
        
        # Pre-allocated check for velocity changes
        self._velocity_change_check = [False, False, False]
        
        # Pre-allocated error container
        self._current_errors = [0.0, 0.0, 0.0]  # distance, lateral, angular
        
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
        
        # Pre-allocate additional objects used in hot paths
        self._init_reusable_objects()
        
        # Set up subscriptions
        self._setup_subscriptions()
        
        # Set up publishers
        self._setup_publishers()
        
        # Set up timers with tiered frequencies
        self._setup_timers()
        
        # Flag to track if we're shutting down
        self._shutting_down = False
        
        # Define movement strategies using table-driven approach
        self._init_strategy_table()
        
        # Log startup info (just once)
        self.get_logger().info("PID Controller initialized with table-driven state logic")
    
    def control_loop_callback(self):
        """Regular control loop to calculate and publish velocity commands."""
        try:
            if self._shutting_down:
                return
                    
            current_time = time.time()
            dt = current_time - self.last_control_time
            self.last_control_time = current_time
            self.cycle_count += 1
            
            # Log periodic status updates
            if self.cycle_count % 50 == 0:
                self._log_periodic_status()
            
            # Only generate commands in tracking mode with a recent target
            if self.robot_state != "tracking" or self.current_target is None:
                # When not tracking, ensure robot is stopped (unless controlled by another node)
                if self.robot_state not in ["searching", "lost_ball"]:
                    self.stop_robot()
                return
            
            # Use values from target callback
            distance = self.current_distance
            lateral = self.current_lateral
            bearing = self.current_bearing
            angular_degrees = math.degrees(bearing)
            
            # Debug log for target values
            self.get_logger().debug(f"TARGET VALUES: distance={distance:.3f}, lateral={lateral:.3f}, bearing={bearing:.3f}, angular_deg={angular_degrees:.3f}")
            
            # Calculate errors using pre-allocated array
            self._current_errors[0] = distance - self.desired_distance  # distance_error
            self._current_errors[1] = lateral - self.target_offset_y    # lateral_error
            self._current_errors[2] = bearing                          # angular_error
            
            # FIXED: Check if we need to reset stopped state based on errors
            state_reset = self._reset_stopped_state_if_needed(
                self._current_errors[0], 
                self._current_errors[1], 
                angular_degrees
            )
            
            # If state was reset, skip the normal stop condition check this cycle
            if not state_reset:
                # Unified stop condition check
                try:
                    stop_result = self._evaluate_stop_conditions(
                        distance, lateral, angular_degrees, self._robot_stopped
                    )
                    
                    # Check the type of the returned value
                    if not isinstance(stop_result, tuple) or len(stop_result) != 2:
                        self.get_logger().error(f"ERROR: _evaluate_stop_conditions returned invalid type: {type(stop_result)}, value: {stop_result}")
                        # Provide default values
                        should_stop, stop_reason = False, "Invalid stop check result"
                    else:
                        should_stop, stop_reason = stop_result
                except Exception as e:
                    self.get_logger().error(f"ERROR in _evaluate_stop_conditions: {str(e)}")
                    should_stop, stop_reason = False, "Error in stop check"
                
                if should_stop:
                    if not self._robot_stopped:
                        self.get_logger().info(stop_reason)
                        self.stop_robot()
                    return
            
            # Log calculated errors
            self.get_logger().debug(f"ERRORS: distance={self._current_errors[0]:.3f}, lateral={self._current_errors[1]:.3f}, angular={self._current_errors[2]:.3f}")
            
            # Update error trackers
            self.distance_error_tracker.update(self._current_errors[0], dt)
            self.lateral_error_tracker.update(self._current_errors[1], dt)
            self.angular_error_tracker.update(self._current_errors[2], dt)
            
            # Determine the optimal movement strategy
            try:
                strategy = self._determine_movement_strategy(
                    self._current_errors[0], self._current_errors[1], angular_degrees
                )
                
                # Check if strategy is a valid dictionary
                if not isinstance(strategy, dict):
                    self.get_logger().error(f"ERROR: _determine_movement_strategy returned invalid type: {type(strategy)}, value: {strategy}")
                    # Create a default strategy dictionary
                    strategy = {
                        "strategy_name": "DEFAULT_FALLBACK",
                        "use_forward": True,
                        "use_lateral": self.use_lateral_control,
                        "use_angular": True,
                        "forward_scale": 0.5,
                        "lateral_scale": 0.5,
                        "angular_scale": 0.5,
                        "reason": "Fallback due to error in strategy determination"
                    }
            except Exception as e:
                self.get_logger().error(f"ERROR in _determine_movement_strategy: {str(e)}")
                # Create a default strategy dictionary
                strategy = {
                    "strategy_name": "ERROR_FALLBACK",
                    "use_forward": True,
                    "use_lateral": self.use_lateral_control,
                    "use_angular": True,
                    "forward_scale": 0.3,
                    "lateral_scale": 0.3,
                    "angular_scale": 0.3,
                    "reason": f"Error in strategy determination: {str(e)}"
                }
            
            # Log strategy details for debugging
            self.get_logger().debug(f"STRATEGY: {strategy['strategy_name']} - {strategy['reason']}")
            
            # Log strategy change if it changed
            if strategy["strategy_name"] != self.current_strategy:
                if self.strategy_log_level >= 1:
                    self.get_logger().info(f"STRATEGY: {self.current_strategy} → {strategy['strategy_name']}: {strategy['reason']}")
                self.previous_strategy = self.current_strategy
                self.current_strategy = strategy["strategy_name"]
                self.strategy_change_time = current_time
            elif self.strategy_log_level >= 2:
                # Log every strategy selection at highest verbosity
                self.get_logger().debug(f"STRATEGY: Maintaining {strategy['strategy_name']}: {strategy['reason']}")
            
            # Apply strategy to movement decisions
            use_forward = strategy["use_forward"]
            use_lateral = strategy["use_lateral"]
            use_angular = strategy["use_angular"]
            
            forward_scale = strategy["forward_scale"]
            lateral_scale = strategy["lateral_scale"]
            angular_scale = strategy["angular_scale"]
            
            # Log movement flags and scale factors
            self.get_logger().debug(f"MOVEMENT FLAGS: forward={use_forward}, lateral={use_lateral}, angular={use_angular}")
            self.get_logger().debug(f"SCALE FACTORS: forward={forward_scale:.2f}, lateral={lateral_scale:.2f}, angular={angular_scale:.2f}")
            
            # Compute PID outputs using the errors
            try:
                linear_x_velocity = self.pid_linear_x.compute(self._current_errors[0], current_time, not use_forward)
                if not isinstance(linear_x_velocity, (int, float)):
                    self.get_logger().error(f"ERROR: pid_linear_x.compute returned invalid type: {type(linear_x_velocity)}, value: {linear_x_velocity}")
                    linear_x_velocity = 0.0
            except Exception as e:
                self.get_logger().error(f"ERROR in pid_linear_x.compute: {str(e)}")
                linear_x_velocity = 0.0
                
            try:
                lateral_velocity = self.pid_linear_y.compute(self._current_errors[1], current_time, not use_lateral)
                if not isinstance(lateral_velocity, (int, float)):
                    self.get_logger().error(f"ERROR: pid_linear_y.compute returned invalid type: {type(lateral_velocity)}, value: {lateral_velocity}")
                    lateral_velocity = 0.0
            except Exception as e:
                self.get_logger().error(f"ERROR in pid_linear_y.compute: {str(e)}")
                lateral_velocity = 0.0
                
            try:
                angular_velocity = self.pid_angular.compute(self._current_errors[2], current_time, not use_angular)
                if not isinstance(angular_velocity, (int, float)):
                    self.get_logger().error(f"ERROR: pid_angular.compute returned invalid type: {type(angular_velocity)}, value: {angular_velocity}")
                    angular_velocity = 0.0
            except Exception as e:
                self.get_logger().error(f"ERROR in pid_angular.compute: {str(e)}")
                angular_velocity = 0.0
            
            # Log raw PID outputs
            self.get_logger().debug(f"RAW PID OUTPUTS: x={linear_x_velocity:.3f}, y={lateral_velocity:.3f}, θ={angular_velocity:.3f}")
            
            # Apply strategy scaling factors
            linear_x_velocity *= forward_scale
            lateral_velocity *= lateral_scale
            angular_velocity *= angular_scale
            
            # Log scaled PID outputs
            self.get_logger().debug(f"SCALED PID OUTPUTS: x={linear_x_velocity:.3f}, y={lateral_velocity:.3f}, θ={angular_velocity:.3f}")
            
            # Apply velocity and acceleration limits with extensive error handling
            try:
                # Verify self.last_cmd_vel is a valid tuple before using it
                if not isinstance(self.last_cmd_vel, tuple) or len(self.last_cmd_vel) != 3:
                    self.get_logger().error(f"ERROR: last_cmd_vel is invalid: type={type(self.last_cmd_vel)}, value={self.last_cmd_vel}")
                    # Reset to default tuple
                    self.last_cmd_vel = (0.0, 0.0, 0.0)
                
                # Log the values being passed to _apply_velocity_limits
                self.get_logger().debug(
                    f"APPLYING VELOCITY LIMITS: x={linear_x_velocity:.3f}, y={lateral_velocity:.3f}, θ={angular_velocity:.3f}, "
                    f"last_cmd_vel={self.last_cmd_vel}"
                )
                
                # Call the velocity limits function with error handling
                result = self._apply_velocity_limits(
                    linear_x_velocity, lateral_velocity, angular_velocity, current_time
                )
                
                # Verify the result is a valid tuple
                if not isinstance(result, tuple) or len(result) != 3:
                    self.get_logger().error(f"ERROR: _apply_velocity_limits returned invalid type: {type(result)}, value: {result}")
                    # Use pre-limited values
                    limited_velocities = (linear_x_velocity, lateral_velocity, angular_velocity)
                else:
                    limited_velocities = result
                    
                # Unpack the limited velocities
                linear_x_velocity, lateral_velocity, angular_velocity = limited_velocities
                
            except Exception as e:
                self.get_logger().error(f"ERROR in _apply_velocity_limits: {str(e)}")
                # In case of error, keep the pre-limited values
                # (already initialized above)
            
            # Log limited velocities
            self.get_logger().debug(f"LIMITED VELOCITIES: x={linear_x_velocity:.3f}, y={lateral_velocity:.3f}, θ={angular_velocity:.3f}")
            
            # Store new velocities in pre-allocated arrays
            self._velocity_tuple[0] = linear_x_velocity
            self._velocity_tuple[1] = lateral_velocity
            self._velocity_tuple[2] = angular_velocity
            
            # Calculate if velocity changed significantly for logging
            for i in range(3):
                self._velocity_change_check[i] = abs(self._velocity_tuple[i] - self.last_logged_cmd[i]) > self.velocity_change_threshold
            
            # Log velocity commands (throttled)
            if self.debug_level >= 1 or any(self._velocity_change_check):
                self._log_throttled(
                    self.get_logger().info,
                    f"MOTION: x={linear_x_velocity:.2f} y={lateral_velocity:.2f} θ={angular_velocity:.2f}",
                    0.5,  # Throttle to every 0.5 seconds 
                    'last_velocity_log_time'
                )
                # Update last logged command without creating new tuple
                self.last_logged_cmd = tuple(self._velocity_tuple)
            
            # Store for next cycle (safely convert to tuple)
            self.last_cmd_vel = tuple(self._velocity_tuple)
            
            # Update error trackers if significant movement is occurring
            if abs(linear_x_velocity) > self.min_effective_velocity / 2.0:
                self.distance_error_tracker.record_correction()
            if abs(lateral_velocity) > self.min_effective_velocity / 2.0:
                self.lateral_error_tracker.record_correction()
            if abs(angular_velocity) > self.min_angular_velocity / 2.0:
                self.angular_error_tracker.record_correction()
            
            # Ensure all velocity values are explicitly converted to float
            # to avoid type errors in the message fields
            try:
                self._cmd_vel_msg.linear.x = float(linear_x_velocity)
                self._cmd_vel_msg.linear.y = float(lateral_velocity)
                self._cmd_vel_msg.angular.z = float(angular_velocity)
            except (TypeError, ValueError) as e:
                self.get_logger().error(f"ERROR converting velocities to float: {str(e)}")
                self.get_logger().error(f"Velocity values were: x={linear_x_velocity}, y={lateral_velocity}, θ={angular_velocity}")
                self.get_logger().error(f"Types: x={type(linear_x_velocity)}, y={type(lateral_velocity)}, θ={type(angular_velocity)}")
                # Use safe default values
                self._cmd_vel_msg.linear.x = 0.0
                self._cmd_vel_msg.linear.y = 0.0
                self._cmd_vel_msg.angular.z = 0.0
            
            # Save for history (reuse existing tuple)
            try:
                velocity_tuple = (float(linear_x_velocity), float(lateral_velocity), float(angular_velocity))
                self.velocity_history.add(velocity_tuple)
            except Exception as e:
                self.get_logger().error(f"ERROR adding to velocity history: {str(e)}")
            
            # Publish command
            try:
                self.cmd_vel_pub.publish(self._cmd_vel_msg)
            except Exception as e:
                self.get_logger().error(f"ERROR publishing cmd_vel: {str(e)}")
            
            # Publish basic diagnostics
            try:
                self._publish_basic_diagnostics(self._current_errors[0], self._current_errors[1], self._current_errors[2],
                                            linear_x_velocity, lateral_velocity, angular_velocity)
            except Exception as e:
                self.get_logger().error(f"ERROR publishing diagnostics: {str(e)}")
                
        except Exception as e:
            self.get_logger().error(f"Unexpected error in control_loop_callback: {str(e)}")
            import traceback
            self.get_logger().error(f"Traceback: {traceback.format_exc()}")
            try:
                # Try to safely stop the robot
                self.stop_robot()
            except Exception as stop_error:
                self.get_logger().error(f"Failed to stop robot after error: {str(stop_error)}")

    def _evaluate_stop_conditions(self, distance, lateral, angular_degrees, currently_stopped=False):
        """
        Evaluate all stop conditions in one place to ensure consistent logic.
        
        Args:
            distance: Current distance to target
            lateral: Current lateral offset 
            angular_degrees: Current angular error in degrees
            currently_stopped: Whether the robot is already stopped
            
        Returns:
            tuple: (should_stop, stop_reason)
        """
        try:
            # Emergency safety stop (highest priority)
            if distance < self.safety_min_distance:
                return True, f"SAFETY: Ball too close: {distance:.2f}m < {self.safety_min_distance:.2f}m"
                    
            # Target age check
            current_time = time.time()
            if self.last_target_time is None or (current_time - self.last_target_time) > self.target_max_age:
                return True, f"Target data too old: >{self.target_max_age*1000:.0f}ms"
            
            # Target reached check with hysteresis
            distance_error = abs(distance - self.desired_distance)
            lateral_error = abs(lateral - self.target_offset_y)
            angular_error = abs(angular_degrees)
            
            # Mandatory stop period check
            if currently_stopped and (current_time - self._stop_time) < self.complete_stop_time:
                return True, f"Mandatory stop period ({self.complete_stop_time:.1f}s)"
            
            # Use different thresholds based on current state (hysteresis)
            if currently_stopped:
                # Larger threshold to exit stopped state (hysteresis)
                if (distance_error < self.stop_zone_size * self.stop_hysteresis_factor and
                    lateral_error < self.lateral_deadband * self.stop_hysteresis_factor and
                    angular_error < self.angular_deadband * self.stop_hysteresis_factor):
                    return True, "Remaining stopped (within hysteresis thresholds)"
            else:
                # Normal threshold to enter stopped state
                if (distance_error < self.stop_zone_size and
                    lateral_error < self.lateral_deadband * self.stop_lateral_factor and
                    angular_error < self.angular_deadband * self.stop_angular_factor):
                    return True, f"TARGET: Reached position! Distance: {distance:.2f}m, lateral: {lateral:.2f}m, bearing: {angular_degrees:.2f}°"
            
            # Check for drift after stop - ONLY when not actively tracking
            # FIXED: Only check drift when not in tracking mode
            if currently_stopped and self.robot_state != "tracking":
                # Current position vs position when stopped
                distance_drift = abs(distance - self._last_stop_position[0])
                lateral_drift = abs(lateral - self._last_stop_position[1])
                angular_drift = abs(angular_degrees - math.degrees(self._last_stop_position[2]))
                
                # Log significant drift
                if (distance_drift > self.drift_detection_threshold or
                    lateral_drift > self.drift_detection_threshold or
                    angular_drift > self.angular_deadband):
                    self.get_logger().warn(
                        f"Drift detected after stop: distance={distance_drift:.3f}m, "
                        f"lateral={lateral_drift:.3f}m, angular={angular_drift:.1f}°"
                    )
                    # Don't return anything here, fall through to end of function
            
            # No stop condition met
            return False, "No stop conditions met"
        except Exception as e:
            self.get_logger().error(f"Error in _evaluate_stop_conditions: {str(e)}")
            # Safe default
            return False, f"Error in stop conditions: {str(e)}"  

    def _reset_stopped_state_if_needed(self, distance_error, lateral_error, angular_error):
        """
        Reset stopped state if significant movement is required.
        
        Args:
            distance_error: Current error in distance to target
            lateral_error: Current error in lateral position
            angular_error: Current error in angular position (degrees)
            
        Returns:
            bool: True if stopped state was reset, False otherwise
        """
        if not self._robot_stopped:
            return False  # Already in movement state
            
        # If any error exceeds the movement threshold, exit stopped state
        distance_threshold = self.stop_zone_size * self.stop_hysteresis_factor  
        lateral_threshold = self.lateral_deadband * self.stop_hysteresis_factor
        angular_threshold = self.angular_deadband * self.stop_hysteresis_factor
        
        if (abs(distance_error) > distance_threshold or
            abs(lateral_error) > lateral_threshold or
            abs(angular_error) > angular_threshold):
            
            self.get_logger().info(
                f"Exiting stopped state - Movement required: "
                f"distance_error={distance_error:.3f}m, "
                f"lateral_error={lateral_error:.3f}m, "
                f"angular_error={angular_error:.2f}°"
            )
            
            # Reset stopped state
            self._robot_stopped = False
            
            # Don't reset the full PID controllers, just allow movement
            return True
            
        return False

    def stop_robot(self):
        """Send a command to stop all robot motion immediately."""
        # Reuse cmd_vel message and set all fields to 0
        self._cmd_vel_msg.linear.x = 0.0
        self._cmd_vel_msg.linear.y = 0.0
        self._cmd_vel_msg.angular.z = 0.0
        
        # Publish stop command multiple times to ensure it's received
        for _ in range(3):
            self.cmd_vel_pub.publish(self._cmd_vel_msg)
            time.sleep(0.01)  # Small delay between publishes
        
        # Reset last command velocity for acceleration limiting
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        
        # Clear velocity history
        self.velocity_history = LightweightBuffer(max_size=6)
        
        # FIXED: Only reset PID controllers if transitioning out of tracking mode
        # This prevents losing integral terms during active tracking
        # Reset error trackers
        self.distance_error_tracker.reset()
        self.lateral_error_tracker.reset()
        self.angular_error_tracker.reset()
            
        # Reset PID controllers to clear any lingering integral terms
        self.pid_linear_x.reset()
        self.pid_linear_y.reset()
        self.pid_angular.reset()
        
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
    
    def _publish_basic_diagnostics(self, distance_error, lateral_error, angular_error,
                                linear_x_velocity, lateral_velocity, angular_velocity):
        """Publish basic diagnostic information for PID controllers."""
        if self._shutting_down:
            return
            
        # Fill pre-allocated numpy array with diagnostic data
        self._diag_data[0] = distance_error
        self._diag_data[1] = lateral_error
        self._diag_data[2] = angular_error
        self._diag_data[3] = linear_x_velocity
        self._diag_data[4] = lateral_velocity
        self._diag_data[5] = angular_velocity
        self._diag_data[6] = self.pid_linear_x.integral
        self._diag_data[7] = self.pid_linear_y.integral
        self._diag_data[8] = self.pid_angular.integral
        self._diag_data[9] = self.current_distance
        self._diag_data[10] = float(self._robot_stopped)
        
        # Update and publish pre-allocated message
        self._diag_msg.data = self._diag_data.tolist()
        self.pid_diag_pub.publish(self._diag_msg)
    
    def _log_periodic_status(self):
        """Log a comprehensive status update at periodic intervals."""
        if self._shutting_down or self.debug_level < 1:
            return
            
        current_time = time.time()
        if current_time - getattr(self, "last_status_log_time", 0) < 5.0:  # Every 5 seconds
            return
            
        self.last_status_log_time = current_time
        
        # Basic status info
        if self.current_target is None:
            self.get_logger().info(f"STATUS: State={self.robot_state}, No target detected")
            return
            
        # With target info
        self.get_logger().info(
            f"STATUS: State={self.robot_state}, "
            f"Strategy={self.current_strategy}, "
            f"Target=[{self.current_distance:.2f}m, "
            f"{self.current_lateral:.2f}m, "
            f"{math.degrees(self.current_bearing):.1f}°], "
            f"Stopped={self._robot_stopped}"
        )
        
    def publish_diagnostics(self):
        """Publish detailed diagnostic information at a slower rate."""
        if self._shutting_down or not self.robot_state == "tracking" or self.debug_level < 1:
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
        self.get_logger().info(
            f"DIAG: Avg Vel=[{avg_lin_x_vel:.2f}, {avg_lin_y_vel:.2f}, {avg_ang_vel:.2f}], "
            f"Strategy={self.current_strategy}, "
            f"E=[{self.distance_error_tracker.current_error:.2f}m, "
            f"{self.lateral_error_tracker.current_error:.2f}m, "
            f"{math.degrees(self.angular_error_tracker.current_error):.1f}°]"
        )

    def prepare_shutdown(self):
        """Prepare for node shutdown."""
        self.get_logger().info("Preparing for shutdown")
        
        # Immediately stop the robot
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
    node = PIDControllerNode()
    
    # Welcome message
    print("=================================================")
    print("PID Controller for Basketball Tracking Robot")
    print("=================================================")
    print("This node implements three PID controllers with table-driven logic:")
    print("1. Linear X velocity (forward/backward movement)")
    print("2. Linear Y velocity (lateral/strafing movement)")
    print("3. Angular velocity (turning/rotation)")
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