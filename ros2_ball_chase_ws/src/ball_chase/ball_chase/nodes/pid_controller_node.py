"""
Tennis Ball Tracking Robot - Improved PID Controller Node
========================================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving tennis ball.
The system uses multiple sensing modalities which are combined in a fusion node, passed to a
state manager, and finally to this PID controller node that generates motor commands.

This Node's Purpose:
------------------
The PID controller node is responsible for the actual motion control of the robot when
it's actively tracking a tennis ball. It implements three separate PID controllers:
1. Linear X velocity controller - Controls forward/backward movement to maintain ideal distance
2. Linear Y velocity controller - Controls lateral movement (strafing) for mecanum wheels
3. Angular velocity controller - Controls turning to keep the ball centered in view

PID Control Explained:
--------------------
PID (Proportional-Integral-Derivative) control is a feedback mechanism that:
- P term: Responds proportionally to the current error
- I term: Accumulates past errors to address systematic biases
- D term: Anticipates future errors based on rate of change

These three components are weighted and combined to produce smooth, accurate control:
Output = Kp*error + Ki*∫error·dt + Kd*(d/dt)error

Optimizations:
------------
This version includes significant optimizations for better performance on Raspberry Pi:
- Memory reuse for ROS messages and buffers
- Reduced computational load with caching and lazy evaluation
- Tiered update frequencies for different operations
- Throttled logging to reduce overhead
- Improved concurrency and CPU utilization
- Deadzone for lateral movement to prevent small oscillations

Data Pipeline:
-------------
1. Fusion node integrates sensor data about the ball position
2. State manager determines the robot's operational state
3. This PID controller:
   - Receives target positions from state manager
   - Calculates appropriate motor commands using PID algorithms 
   - Publishes velocity commands to the robot's motors

The controller automatically adapts its behavior based on the robot's current state
and the distance to the target.
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

# Topic configuration (ensures consistency with other nodes)
TOPICS = {
    "input": {
        "target": "/basketball/fused/position",
        "state": "/robot/state"
    },
    "output": {
        "cmd_vel": "/controller/cmd_vel",  # Changed from "/cmd_vel" to "/controller/cmd_vel"
        "diagnostics": "/pid/diagnostics"
    }
}

# Log throttling parameters
LOG_THROTTLE_CONTROL = 1.0     # Seconds between control loop status logs
LOG_THROTTLE_STATE = 0.5       # Seconds between state change logs
LOG_THROTTLE_DIAG = 0.2        # Seconds between diagnostic logs

class LightweightBuffer:
    """
    Memory-efficient buffer for storing historical data.
    
    This implementation pre-allocates its storage and operates as a circular 
    buffer, overwriting the oldest entries when full to avoid dynamic allocations.
    """
    
    def __init__(self, max_size=20, default_value=(0.0, 0.0, 0.0)):
        """
        Initialize a fixed-size circular buffer.
        
        Args:
            max_size (int): Maximum number of elements to store
            default_value: Default value for pre-allocation
        """
        self.data = [default_value] * max_size  # Pre-allocate with default values
        self.next_index = 0
        self.count = 0
        self.max_size = max_size
    
    def add(self, value):
        """
        Add a new value to the buffer, overwriting oldest if full.
        
        Args:
            value: Value to add to the buffer
        """
        self.data[self.next_index] = value
        self.next_index = (self.next_index + 1) % self.max_size
        self.count = min(self.count + 1, self.max_size)
    
    def get_all(self):
        """
        Get all values currently in the buffer.
        
        Returns:
            list: All values in chronological order
        """
        if self.count < self.max_size:
            return self.data[:self.count]
        # Reconstruct in chronological order
        start_idx = self.next_index
        return self.data[start_idx:] + self.data[:start_idx]

class PIDController:
    """
    A memory-efficient PID controller implementation.
    
    This class provides a complete PID controller with:
    - Anti-windup protection to prevent integral term saturation
    - Output limiting to ensure safe operation
    - Automatic time calculation for correct derivative and integral terms
    - Memory reuse for improved performance
    
    The PID formula used is:
    output = Kp*error + Ki*∫error·dt + Kd*Δerror/Δt
    
    Where:
    - Kp, Ki, Kd are the gains for each term
    - error is the difference between setpoint and measured value
    - dt is the time delta between updates
    """
    
    def __init__(self, kp, ki, kd, output_min, output_max, anti_windup=True, name="PID"):
        """
        Initialize a new PID controller.
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
            output_min (float): Minimum allowable output value
            output_max (float): Maximum allowable output value
            anti_windup (bool): Whether to use anti-windup protection
            name (str): Name for this controller (for debugging)
        """
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
        
    def reset(self):
        """
        Reset controller state.
        
        This clears the integral accumulator, previous error, and timing information,
        essentially restarting the controller from scratch.
        """
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        
    def compute(self, error, current_time=None):
        """
        Compute the control output based on the error.
        
        This method implements the complete PID algorithm:
        1. Calculate time delta since last update
        2. Calculate proportional, integral, and derivative terms
        3. Apply anti-windup if enabled
        4. Limit output to configured range
        
        Args:
            error (float): Current error value (setpoint - measured_value)
            current_time (float, optional): Current time in seconds
                                           (if None, will use time.time())
            
        Returns:
            float: Control output value
        """
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
        if dt <= 0:
            dt = 0.01  # Fallback to prevent division by zero (assume 100Hz)
            
        # Calculate each PID term
        # Proportional term (proportional to error)
        p_term = self.kp * error
        
        # FIXED: Improved integral term handling
        # Integral term (accumulates error over time)
        # Only accumulate integral when error is significant
        # This prevents integral windup when close to target
        if abs(error) > 0.05:  # Small deadband for integral accumulation
            self.integral += error * dt
        else:
            # Gradually reduce integral term when close to target
            self.integral *= 0.95  # 5% decay rate when near target
            
        i_term = self.ki * self.integral
        
        # Derivative term (rate of change of error)
        # Note: derivative on error, not measurement, can cause "derivative kick"
        d_term = self.kd * (error - self.prev_error) / dt
        
        # Calculate raw output by summing all terms
        output = p_term + i_term + d_term
        
        # Apply output limits
        output_limited = max(self.output_min, min(self.output_max, output))
        
        # FIXED: Enhanced anti-windup
        # Anti-windup: adjust integral term if output is saturated
        # This prevents integral windup when the controller cannot achieve the desired output
        if self.anti_windup:
            if output != output_limited and abs(self.ki) > 1e-10:  # Avoid division by zero
                # Reduce integral by the excess output scaled by Ki
                self.integral -= (output - output_limited) / self.ki
                # Recalculate integral term
                i_term = self.ki * self.integral
            # Additional anti-windup when error changes sign
            if error * self.prev_error < 0:  # Error changed sign
                # Reduce integral to avoid overshooting
                self.integral *= 0.5
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
        """
        Get the last calculated PID components.
        
        Returns:
            tuple: (p_term, i_term, d_term) - The three components of the PID output
        """
        return (self.last_p_term, self.last_i_term, self.last_d_term)

class PIDControllerNode(Node):
    """
    Optimized PID Controller node for tennis ball tracking.
    
    This node uses separate PID controllers for movement:
    - Linear X velocity controller: Adjusts forward/backward speed to maintain ideal distance
    - Linear Y velocity controller: Controls lateral movement (strafing) for mecanum wheels
    - Angular velocity controller: Adjusts turning to keep the ball centered
    
    The node:
    1. Receives target positions from the state manager
    2. Uses PID controllers to generate linear and angular velocities
    3. Publishes velocity commands to control the robot
    4. Adapts control parameters based on robot state and target distance
    5. Provides detailed diagnostic information for tuning and debugging
    
    Optimizations include:
    - Message reuse to reduce memory allocations
    - Fixed-size buffers with pre-allocation
    - Throttled logging to reduce overhead
    - Tiered update frequencies for different operations
    - Caching of frequently used calculations
    - Deadzones for lateral movement
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
        self.get_logger().info("PID Controller initialized with optimizations")
        
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
                ('linear_x_max', 0.2),    # Forward limit (m/s) - REDUCED from 0.25 to 0.2
                
                # Linear Y velocity PID parameters - controls lateral movement (strafing)
                ('linear_y_kp', 0.8),     # Proportional gain - INCREASED from 0.7 to 0.8
                ('linear_y_ki', 0.15),    # Integral gain - INCREASED from 0.1 to 0.15
                ('linear_y_kd', 0.05),    # Derivative gain
                ('linear_y_min', -0.2),   # Right strafe limit (m/s)
                ('linear_y_max', 0.2),    # Left strafe limit (m/s)
                
                # Angular velocity PID parameters - controls turning
                ('angular_kp', 3.5),      # Proportional gain - INCREASED from 3.0 to 3.5
                ('angular_ki', 0.15),     # Integral gain - INCREASED from 0.1 to 0.15
                ('angular_kd', 0.3),      # Derivative gain
                ('angular_min', -0.5),    # Right turn limit (rad/s)
                ('angular_max', 0.5),     # Left turn limit (rad/s)
                
                # Control parameters
                ('min_distance', 0.9),       # Minimum distance to keep from ball (meters)
                ('max_distance', 2.0),       # Maximum tracking distance (meters)
                ('target_offset_x', 0.0),    # Desired offset from ball in x direction
                ('target_offset_y', 0.0),    # Desired offset from ball in y direction
                ('target_update_rate', 10.0),# Control loop update rate (Hz)
                ('diagnostics_rate', 1.0),   # Rate for detailed diagnostics (Hz)
                ('frame_check_rate', 0.2),   # Rate for coordinate frame checks (Hz)
                ('debug_level', 1),          # 0=errors only, 1=info, 2=debug - REDUCED from 2 to 1 after debugging
                ('adaptive_gains', True),    # Whether to adjust gains based on distance
                ('use_lateral_control', True), # Whether to use Y-axis control for lateral movement
                ('lateral_deadband', 0.03),  # Deadband for lateral error - reduced from 0.05 to 0.03
                ('deadband_distance', 0.02), # Deadband for distance error (to prevent minor oscillations)
                ('stop_zone_size', 0.2),     # Size of zone where robot will stop
                ('safety_min_distance', 0.4), # Emergency stop distance (m) - stop if closer than this
                ('min_angular_velocity', 0.1), # Minimum angular velocity to apply (rad/s) - INCREASED from 0.01 to 0.1
                ('max_accel', 0.4),          # Maximum acceleration per control cycle
                ('max_angular_accel', 0.8),  # Maximum angular acceleration per control cycle
                ('forward_scale_with_angle', True), # Whether to scale forward velocity based on angular error
                ('angular_scale_threshold', 10.0),  # Angle error in degrees at which to start scaling forward velocity
                ('always_use_angular_control', True), # Always use angular velocity for alignment
                ('enable_debug_velocity_publish', False), # Whether to log velocity commands being published - DISABLED after debugging
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
        self.frame_check_rate = self.get_parameter('frame_check_rate').value
        self.debug_level = self.get_parameter('debug_level').value
        self.adaptive_gains = self.get_parameter('adaptive_gains').value
        self.use_lateral_control = self.get_parameter('use_lateral_control').value
        self.lateral_deadband = self.get_parameter('lateral_deadband').value
        self.deadband_distance = self.get_parameter('deadband_distance').value
        self.stop_zone_size = self.get_parameter('stop_zone_size').value
        self.safety_min_distance = self.get_parameter('safety_min_distance').value
        self.min_angular_velocity = self.get_parameter('min_angular_velocity').value
        self.max_accel = self.get_parameter('max_accel').value
        self.max_angular_accel = self.get_parameter('max_angular_accel').value
        self.forward_scale_with_angle = self.get_parameter('forward_scale_with_angle').value
        self.angular_scale_threshold = self.get_parameter('angular_scale_threshold').value
        self.always_use_angular_control = self.get_parameter('always_use_angular_control').value
        self.enable_debug_velocity_publish = self.get_parameter('enable_debug_velocity_publish').value
        
    def _init_controllers(self):
        """Initialize the PID controllers."""
        # Initialize PID controllers with descriptive names
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
        
        # Error tracking
        self.last_distance_error = 0.0  # Previous distance error
        
        # Motion smoothing
        self.last_cmd_vel = (0.0, 0.0, 0.0)  # (lin_x, lin_y, ang_z)
        
        # Diagnostic information
        self.cycle_count = 0            # Number of control cycles
        
        # Use LightweightBuffer for velocity history
        self.velocity_history = LightweightBuffer(max_size=20)
        
        # Cached values
        self.desired_distance = self.min_distance + self.target_offset_x
        self._cached_transforms = {}    # Store transforms for reuse
        self._last_diagnostics_frame = None  # Track frame changes for diagnostics
        
    def _setup_subscriptions(self):
        """Set up all subscriptions for this node."""
        # Subscribe to robot state
        self.state_sub = self.create_subscription(
            String,
            TOPICS["input"]["state"],
            self.state_callback,
            10
        )
        
        # Subscribe to tennis ball target
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
        self.get_logger().info(f"  Limits: [{self.linear_x_min}, {self.linear_x_max}] m/s")
        
        self.get_logger().info("Linear Y velocity PID:")
        self.get_logger().info(f"  Kp: {self.linear_y_kp}, Ki: {self.linear_y_ki}, Kd: {self.linear_y_kd}")
        self.get_logger().info(f"  Limits: [{self.linear_y_min}, {self.linear_y_max}] m/s")
        
        self.get_logger().info("Angular velocity PID:")
        self.get_logger().info(f"  Kp: {self.angular_kp}, Ki: {self.angular_ki}, Kd: {self.angular_kd}")
        self.get_logger().info(f"  Limits: [{self.angular_min}, {self.angular_max}] rad/s")
        
        self.get_logger().info("Control parameters:")
        self.get_logger().info(f"  Min distance: {self.min_distance} m")
        self.get_logger().info(f"  Max distance: {self.max_distance} m")
        self.get_logger().info(f"  Target offset X: {self.target_offset_x} m")
        self.get_logger().info(f"  Target offset Y: {self.target_offset_y} m")
        self.get_logger().info(f"  Update rate: {self.update_rate} Hz")
        self.get_logger().info(f"  Adaptive gains: {self.adaptive_gains}")
        self.get_logger().info(f"  Use lateral control: {self.use_lateral_control}")
        self.get_logger().info(f"  Lateral deadband: {self.lateral_deadband} m")
        self.get_logger().info(f"  Distance deadband: {self.deadband_distance} m")
        self.get_logger().info(f"  Stop zone size: {self.stop_zone_size} m")
        self.get_logger().info(f"  Safety min distance: {self.safety_min_distance} m")
        self.get_logger().info(f"  Min angular velocity: {self.min_angular_velocity} rad/s")
        self.get_logger().info(f"  Max acceleration: {self.max_accel}")
        self.get_logger().info(f"  Max angular acceleration: {self.max_angular_accel}")
        self.get_logger().info(f"  Forward scale with angle: {self.forward_scale_with_angle}")
        self.get_logger().info(f"  Angular scale threshold: {self.angular_scale_threshold} degrees")
        self.get_logger().info(f"  Always use angular control: {self.always_use_angular_control}")
        self.get_logger().info(f"  Enable debug velocity publish: {self.enable_debug_velocity_publish}")
        self.get_logger().info(f"  Debug level: {self.debug_level}")
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
        
        Args:
            msg (String): Current robot state
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
                
                if self.debug_level >= 1:
                    self.get_logger().debug("PID controllers reset due to state change")
                
            # If we're not in tracking mode, ensure the robot is stopped
            # (unless it's in searching or lost_ball mode, where the state manager controls motion)
            if new_state != "tracking" and new_state != "searching" and new_state != "lost_ball":
                self.stop_robot()
        
    def target_callback(self, msg):
        """
        Handle target position updates from the state manager.
        
        This receives the 3D position of the tennis ball from the
        state manager and updates our target tracking variables.
        
        Args:
            msg (PointStamped): 3D position of the target
        """
        if self._shutting_down:
            return
            
        self.current_target = msg
        self.last_target_time = time.time()
        
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
            # Keep same bearing calculation, but we'll invert the direction in the control logic
            self.current_bearing = math.atan2(target.y, target.x)
            
            # In base_link frame, positive Y is left
            # To make lateral consistent: positive = ball is to the left
            self.current_lateral = target.y
            
            # Enhanced logging for coordinate frame understanding
            self.get_logger().debug(
                f"Coordinate frame details: "
                f"raw=[{target.x:.3f}, {target.y:.3f}, {target.z:.3f}], "
                f"bearing={math.degrees(self.current_bearing):.2f}°, "
                f"lateral={self.current_lateral:.3f}m"
            )
            
            # For base frame, we might need to handle z component differently
            if abs(target.z) > 0.1 and self.debug_level >= 2:  # If there's significant height difference
                self.get_logger().debug(f"Target has height component: z={target.z:.2f}m")

        # Check for safety distance - if ball is too close, log it immediately
        if self.current_distance < self.safety_min_distance:
            self.get_logger().warn(
                f"Ball very close to robot! Distance={self.current_distance:.2f}m, "
                f"safety threshold={self.safety_min_distance:.2f}m"
            )
            
        # Throttled logging for target updates
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
        
    def control_loop_callback(self):
        """
        Regular control loop to calculate and publish velocity commands.
        
        This is the core function that:
        1. Checks if we should be controlling the robot in the current state
        2. Calculates appropriate linear and angular velocities using PID controllers
        3. Publishes velocity commands to control the robot's motion
        
        Optimizations include:
        - Early exit conditions to skip unnecessary processing
        - Reuse of pre-allocated message objects
        - Acceleration limiting for smoother motion
        """
        if self._shutting_down:
            return
            
        current_time = time.time()
        self.cycle_count += 1
        
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
            
        # Check if target is too old (500ms timeout)
        if self.last_target_time is None or (current_time - self.last_target_time) > 0.5:
            self.get_logger().warn("Target is too old (>500ms), stopping robot")
            self.stop_robot()
            return
        
        # Log current target position and properties for diagnostics, but throttled
        if self.debug_level >= 2 and self.cycle_count % 50 == 0:
            raw_point = self.current_target.point
            frame_id = getattr(self.current_target.header, 'frame_id', 'unknown')
            self._log_throttled(
                self.get_logger().info,
                f"Raw target data: pos=[{raw_point.x:.2f}, {raw_point.y:.2f}, {raw_point.z:.2f}], "
                f"frame={frame_id}",
                1.0,  # Every 1 second max
                'last_raw_target_log_time'
            )
            
        # Use the processed values from target_callback instead of recalculating
        # This ensures consistent interpretation of coordinate frames
        distance = self.current_distance
        lateral = self.current_lateral
        bearing = self.current_bearing
        
        # Calculate distance error (target - current)
        raw_distance_error = distance - self.desired_distance
        
        # Throttled logging for control errors
        if self.debug_level >= 1 and self.cycle_count % 10 == 0:
            self._log_throttled(
                self.get_logger().info,
                f"Control errors: distance_error={raw_distance_error:.2f}m, "
                f"lateral_error={lateral:.2f}m, "
                f"angular_error={math.degrees(bearing):.1f}°",
                0.2,  # Every 0.2 seconds max
                'last_error_log_time'
            )
        
        # Emergency safety check - if ball is too close, stop immediately
        if distance < self.safety_min_distance:
            self.get_logger().warn(f"Emergency stop! Ball too close: {distance:.2f}m < {self.safety_min_distance:.2f}m")
            self.stop_robot()
            return
            
        # Calculate distance error with appropriate sign (negative when too close)
        distance_error = raw_distance_error
        
        # FIXED: Improved deadband to prevent oscillations
        # Apply larger deadband to distance error to prevent small oscillations
        # Distance deadband is now implemented as a smooth transition
        if abs(distance_error) < self.deadband_distance * 2.0:
            # Scale down the error within the expanded deadband
            if abs(distance_error) < self.deadband_distance:
                distance_error = 0.0  # Zero when very close to target
            else:
                # Linear scaling for smoother transition at deadband edge
                scale = (abs(distance_error) - self.deadband_distance) / self.deadband_distance
                distance_error *= scale
        
        # Calculate lateral error and apply lateral deadband
        lateral_error = lateral - self.target_offset_y
        
        # FIXED: Reduced lateral deadband to improve responsiveness
        if abs(lateral_error) < self.lateral_deadband:
            lateral_error = 0.0
            
        # Angular error (we want bearing to be 0 - centered)
        angular_error = bearing
        
        # Apply adaptive gains if enabled
        if self.adaptive_gains:
            self._adjust_gains_for_distance(distance)
        
        # Compute PID outputs
        linear_x_velocity = self.pid_linear_x.compute(distance_error, current_time)
        
        # Modified control strategy: always use angular control for alignment,
        # and use lateral control when appropriate
        
        # Always compute lateral velocity if configured, regardless of deadband
        # This ensures we compute the value for diagnostics even if we don't use it
        lateral_velocity = self.pid_linear_y.compute(lateral_error, current_time)
        
        # Determine whether to use lateral velocity based on configuration and deadband
        use_lateral = self.use_lateral_control and abs(lateral_error) >= self.lateral_deadband
        
        # Always compute angular velocity for alignment
        angular_velocity = self.pid_angular.compute(angular_error, current_time)
        
        # When using both controls, slightly reduce the influence of each
        # to prevent over-correction
        if use_lateral and self.always_use_angular_control:
            # Apply scaling factors - use both lateral and angular control together
            lateral_scale = 0.8  # Scale down lateral velocity slightly when using both
            angular_scale = 0.8  # Scale down angular velocity slightly when using both
            linear_y_velocity = lateral_velocity * lateral_scale
            angular_velocity = angular_velocity * angular_scale
        elif use_lateral and not self.always_use_angular_control:
            # Use only lateral control
            linear_y_velocity = lateral_velocity
            angular_velocity = 0.0
        else:
            # Use only angular control
            linear_y_velocity = 0.0
            angular_scale = 1.0
            # No change to angular_velocity, already calculated above
            
        # FIXED: Add minimum lateral velocity threshold similar to angular
        # Apply minimum lateral velocity threshold if needed
        if 0 < abs(linear_y_velocity) < 0.05:  # Minimum lateral velocity to apply
            # Apply minimum threshold with correct sign
            linear_y_velocity = math.copysign(0.05, linear_y_velocity)
            self.get_logger().info(
                f"LATERAL CORRECTION: Forcing minimum lateral velocity: {linear_y_velocity:.3f} m/s "
                f"(error: {lateral_error:.3f}m, direction: {'LEFT' if linear_y_velocity > 0 else 'RIGHT'})"
            )
            
        # Apply minimum angular velocity threshold if needed
        if 0 < abs(angular_velocity) < self.min_angular_velocity:
            # Apply minimum threshold with correct sign
            angular_velocity = math.copysign(self.min_angular_velocity, angular_velocity)
        
        # Optional: scale forward velocity based on angular error
        # This makes the robot slow down when trying to turn to face the ball
        if self.forward_scale_with_angle:
            angular_error_degrees = math.degrees(abs(angular_error))
            
            # Only scale forward velocity if angle error exceeds threshold
            if angular_error_degrees > self.angular_scale_threshold:
                # Scale factor: 1.0 at threshold, linearly decreasing to 0.3 at 45 degrees
                forward_scale = max(0.3, 1.0 - ((angular_error_degrees - self.angular_scale_threshold) / (45.0 - self.angular_scale_threshold)))
                # Apply scaling to forward velocity
                linear_x_velocity *= forward_scale
                
                if self.debug_level >= 2 and self.cycle_count % 20 == 0:
                    self.get_logger().info(
                        f"Scaling forward velocity: angular_error={angular_error_degrees:.1f}°, "
                        f"scale={forward_scale:.2f}, "
                        f"adjusted_velocity={linear_x_velocity:.3f}m/s"
                    )
            
        # Enhanced debugging for motion planning decisions
        if self.debug_level >= 1:
            self.get_logger().info(
                f"MOTION PLANNING: dist_err={distance_error:.3f}m, "
                f"lat_err={lateral_error:.3f}m, "
                f"ang_err={math.degrees(angular_error):.2f}°, "
                f"use_lateral={use_lateral}, "
                f"raw_lat_vel={lateral_velocity:.3f}m/s, "
                f"raw_ang_vel={angular_velocity:.3f}rad/s"
            )
            
        # Debug log pre-acceleration limiting values
        self.get_logger().info(f"PRE-LIMIT VELOCITIES: lin_x={linear_x_velocity:.3f}, lin_y={linear_y_velocity:.3f}, ang_z={angular_velocity:.3f}")
        
        # Apply acceleration limiting for smoother motion
        linear_x_velocity = self._apply_acceleration_limit(
            self.last_cmd_vel[0], linear_x_velocity, self.max_accel, current_time)
        
        # FIXED: Apply higher acceleration limit for lateral movement
        # This allows lateral movement to start more quickly from zero
        linear_y_velocity = self._apply_acceleration_limit(
            self.last_cmd_vel[1], linear_y_velocity, self.max_accel * 1.5, current_time)
        
        angular_velocity = self._apply_acceleration_limit(
            self.last_cmd_vel[2], angular_velocity, self.max_angular_accel, current_time)
        
        # Debug log post-acceleration limiting values
        self.get_logger().info(f"POST-LIMIT VELOCITIES: lin_x={linear_x_velocity:.3f}, lin_y={linear_y_velocity:.3f}, ang_z={angular_velocity:.3f}")
            
        # Store for next cycle
        self.last_cmd_vel = (linear_x_velocity, linear_y_velocity, angular_velocity)
        
        # Update reusable velocity command message (memory optimization)
        self._cmd_vel_msg.linear.x = linear_x_velocity    # Forward/backward
        self._cmd_vel_msg.linear.y = linear_y_velocity    # Left/right strafe
        
        # IMPROVED ANGULAR VELOCITY HANDLING
        # If there's a significant angular error but angular velocity is near zero, force it
        if abs(angular_error) > 0.05 and abs(angular_velocity) < 0.1:  # About 3 degrees of error
            # In base_link frame:
            # - Positive Y means ball is to the LEFT of the robot
            # - Negative Y means ball is to the RIGHT of the robot
            # - For turning:
            #   - Positive angular velocity turns LEFT (counterclockwise)
            #   - Negative angular velocity turns RIGHT (clockwise)
            # Therefore, we should turn:
            # - LEFT (positive angular velocity) when ball is to the LEFT (positive Y)
            # - RIGHT (negative angular velocity) when ball is to the RIGHT (negative Y)
            
            # Use the sign of the angular error directly for turning direction
            forced_angular = math.copysign(self.min_angular_velocity, angular_error)
            direction_text = 'LEFT' if forced_angular > 0 else 'RIGHT'
            
            self.get_logger().info(
                f"ANGULAR CORRECTION: Forcing angular velocity: {forced_angular:.3f} rad/s "
                f"(error: {math.degrees(angular_error):.1f}°, direction: {direction_text}, "
                f"ball position Y: {lateral_error:.3f}m)"
            )
            self._cmd_vel_msg.angular.z = forced_angular
        else:
            # Use regular angular velocity calculation
            self._cmd_vel_msg.angular.z = angular_velocity
            direction_text = 'LEFT' if self._cmd_vel_msg.angular.z > 0 else 'RIGHT' if self._cmd_vel_msg.angular.z < 0 else 'NONE'
            self.get_logger().info(
                f"ANGULAR VELOCITY: {self._cmd_vel_msg.angular.z:.3f} rad/s "
                f"(error: {math.degrees(angular_error):.1f}°, direction: {direction_text}, "
                f"ball position Y: {lateral_error:.3f}m)"
            )
            
        # ENHANCED LATERAL MOVEMENT LOGGING
        if abs(linear_y_velocity) > 0.01:
            self.get_logger().info(
                f"LATERAL MOVEMENT: {linear_y_velocity:.3f} m/s "
                f"(error: {lateral_error:.3f}m, direction: {'LEFT' if linear_y_velocity > 0 else 'RIGHT'})"
            )
        elif abs(lateral_error) > self.lateral_deadband:
            self.get_logger().info(
                f"LATERAL ERROR PRESENT BUT NO MOVEMENT: error={lateral_error:.3f}m, "
                f"calculated_velocity={lateral_velocity:.3f}m/s, "
                f"final_velocity={linear_y_velocity:.3f}m/s"
            )
            
        # Log final velocity values for debugging
        self.get_logger().info(f"FINAL VELOCITY: lin_x={self._cmd_vel_msg.linear.x:.3f}, lin_y={self._cmd_vel_msg.linear.y:.3f}, ang_z={self._cmd_vel_msg.angular.z:.3f}")
        
        # Enhanced debugging to trace commands actually being sent
        if self.enable_debug_velocity_publish:
            self._log_throttled(
                self.get_logger().info,
                f"PUBLISHING VELOCITY: lin_x={self._cmd_vel_msg.linear.x:.3f}m/s, "
                f"lin_y={self._cmd_vel_msg.linear.y:.3f}m/s, "
                f"ang_z={self._cmd_vel_msg.angular.z:.3f}rad/s",
                0.2,  # Every 0.2 seconds max
                'last_velocity_publish_log_time'
            )
        
        # Log velocity values less frequently
        if self.debug_level >= 1 and self.cycle_count % 20 == 0:
            self._log_throttled(
                self.get_logger().info,
                f"[#{self.cycle_count}] Velocity cmd: linear_x={linear_x_velocity:.3f}m/s, "
                f"linear_y={linear_y_velocity:.3f}m/s, "
                f"angular={angular_velocity:.3f}rad/s",
                0.5,  # Every 0.5 seconds max
                'last_velocity_log_time'
            )
        
        # Save for history using the LightweightBuffer
        self.velocity_history.add((linear_x_velocity, linear_y_velocity, angular_velocity))
        
        # Publish command with pre-allocated message
        self.cmd_vel_pub.publish(self._cmd_vel_msg)
        
        # Publish basic diagnostics every cycle
        self.publish_basic_diagnostics(distance_error, lateral_error, angular_error,
                                      linear_x_velocity, linear_y_velocity, angular_velocity)
        
        # Log periodic status with throttling
        if self.debug_level >= 1:
            current_interval = current_time - self.last_control_time
            if current_interval >= LOG_THROTTLE_CONTROL:
                # Get PID components for debugging
                lin_x_p, lin_x_i, lin_x_d = self.pid_linear_x.get_components()
                lin_y_p, lin_y_i, lin_y_d = self.pid_linear_y.get_components()
                ang_p, ang_i, ang_d = self.pid_angular.get_components()
                
                self.get_logger().info(
                    f"PID Control: dist_err={distance_error:.2f}m, "
                    f"lat_err={lateral_error:.2f}m, "
                    f"ang_err={math.degrees(angular_error):.1f}°, "
                    f"lin_x={linear_x_velocity:.2f}m/s, "
                    f"lin_y={linear_y_velocity:.2f}m/s, "
                    f"ang_v={angular_velocity:.2f}rad/s"
                )
                
                if self.debug_level >= 2:
                    self.get_logger().debug(
                        f"Linear X PID: P={lin_x_p:.2f}, I={lin_x_i:.2f}, D={lin_x_d:.2f}"
                    )
                    self.get_logger().debug(
                        f"Linear Y PID: P={lin_y_p:.2f}, I={lin_y_i:.2f}, D={lin_y_d:.2f}"
                    )
                    self.get_logger().debug(
                        f"Angular PID: P={ang_p:.2f}, I={ang_i:.2f}, D={ang_d:.2f}"
                    )
                    
                self.last_control_time = current_time
                
    def _apply_acceleration_limit(self, current_velocity, target_velocity, max_accel, current_time):
        """
        Apply acceleration limiting for smoother motion.
        
        Args:
            current_velocity (float): Current velocity
            target_velocity (float): Desired velocity
            max_accel (float): Maximum acceleration per control cycle
            current_time (float): Current time
            
        Returns:
            float: Limited velocity that doesn't exceed acceleration constraints
        """
        # Calculate time since last control step
        dt = current_time - getattr(self, "last_accel_time", current_time - 0.1)
        self.last_accel_time = current_time
        
        # Scale acceleration limit by time
        accel_limit = max_accel * dt * 12.0  # Scale by dt and by 12 to get reasonable units
        
        # Calculate difference between current and target velocity
        vel_diff = target_velocity - current_velocity
        
        # FIXED: Special case for starting movement from zero
        # Allow quicker acceleration when starting from zero
        if abs(current_velocity) < 0.01 and abs(target_velocity) > 0.05:
            # When starting from stopped position, allow higher initial acceleration
            accel_limit *= 2.0
            self.get_logger().info(f"ACCELERATION BOOST: Starting movement with increased limit: {accel_limit:.3f}")
        
        # Limit acceleration if needed
        if abs(vel_diff) > accel_limit:
            # Apply limit with correct sign
            limited_velocity = current_velocity + math.copysign(accel_limit, vel_diff)
            return limited_velocity
            
        # No limiting needed
        return target_velocity
            
    def _adjust_gains_for_distance(self, distance):
        """
        Adjust PID gains based on distance to target.
        
        This makes the controller more aggressive when the ball is far away
        and more gentle when it's close.
        
        Args:
            distance (float): Current distance to target in meters
        """
        # Scale factor based on distance (1.0 at max_distance, 0.7 at min_distance)
        scale = 0.7 + 0.3 * min(1.0, max(0.0, (distance - self.min_distance) / 
                                        (self.max_distance - self.min_distance)))
        
        # Apply scaling to controllers
        # Linear X controller: less aggressive when close
        self.pid_linear_x.kp = self.linear_x_kp * scale
        
        # FIXED: Further reduce integral gain when close to target
        if distance < self.min_distance + 0.1:
            # Very close to target, use minimal integral gain to prevent overshoot
            self.pid_linear_x.ki = self.linear_x_ki * scale * 0.5
        else:
            self.pid_linear_x.ki = self.linear_x_ki * scale
        
        # Linear Y controller: less aggressive when close
        self.pid_linear_y.kp = self.linear_y_kp * scale
        self.pid_linear_y.ki = self.linear_y_ki * scale
        
        # Angular controller: more precise when close
        precision_scale = 1.5 - 0.5 * scale  # 1.25 when close, 1.0 when far
        self.pid_angular.kp = self.angular_kp * precision_scale
        
        # Log the gain adjustments for debugging
        if self.debug_level >= 1 and self.cycle_count % 20 == 0:
            self.get_logger().info(
                f"ADAPTIVE GAINS: distance={distance:.2f}m, "
                f"lin_x=[P:{self.pid_linear_x.kp:.2f}, I:{self.pid_linear_x.ki:.3f}], "
                f"lin_y=[P:{self.pid_linear_y.kp:.2f}, I:{self.pid_linear_y.ki:.3f}], "
                f"ang=[P:{self.pid_angular.kp:.2f}, I:{self.pid_angular.ki:.3f}]"
            )
            
    def stop_robot(self):
        """Send a command to stop all robot motion immediately."""
        # Reuse cmd_vel message and just set all fields to 0
        self._cmd_vel_msg.linear.x = 0.0
        self._cmd_vel_msg.linear.y = 0.0
        self._cmd_vel_msg.angular.z = 0.0
        
        # Publish stop command
        self.cmd_vel_pub.publish(self._cmd_vel_msg)
        
        # Debug logging for stop command
        if self.enable_debug_velocity_publish:
            self.get_logger().info("STOP COMMAND PUBLISHED: All velocities set to 0.0")
        
        # Reset last command velocity for acceleration limiting
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        
        # Clear velocity history
        self.velocity_history = LightweightBuffer(max_size=20)
        
    def publish_basic_diagnostics(self, distance_error, lateral_error, angular_error,
                               linear_x_velocity, linear_y_velocity, angular_velocity):
        """
        Publish basic diagnostic information for PID controllers.
        
        This is called at the full control loop rate and includes just
        the essential metrics for other nodes.
        
        Args:
            distance_error: Error in distance from target (meters)
            lateral_error: Error in lateral position (meters)
            angular_error: Error in angular position (radians)
            linear_x_velocity: Computed forward/backward velocity (m/s)
            linear_y_velocity: Computed left/right velocity (m/s)
            angular_velocity: Computed angular velocity (rad/s)
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
        max_lin_x_vel = max(lin_x_velocities) if lin_x_velocities else 0
        max_lin_y_vel = max(lin_y_velocities) if lin_y_velocities else 0
        max_ang_vel = max(abs(v) for v in ang_velocities) if ang_velocities else 0
        
        # Only log if time interval has passed (throttling)
        current_time = time.time()
        if (current_time - self.last_diag_log_time) < LOG_THROTTLE_DIAG:
            return
            
        self.last_diag_log_time = current_time
        
        # Log detailed information
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
        
        # Enhanced diagnostics for control logic state
        angular_error_degrees = math.degrees(abs(self.current_bearing))
        lateral_error = self.current_lateral - self.target_offset_y
        using_lateral = self.use_lateral_control and abs(lateral_error) >= self.lateral_deadband
        
        self.get_logger().info(f"Control strategy: using_lateral={using_lateral}, always_use_angular={self.always_use_angular_control}")
        
        if self.forward_scale_with_angle and angular_error_degrees > self.angular_scale_threshold:
            forward_scale = max(0.3, 1.0 - ((angular_error_degrees - self.angular_scale_threshold) / (45.0 - self.angular_scale_threshold)))
            self.get_logger().info(f"Forward scaling: angle={angular_error_degrees:.1f}°, scaling_factor={forward_scale:.2f}")
        
        self.get_logger().info(f"Control cycles: {self.cycle_count}")
        self.get_logger().info("================================")
            
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
        
        # XY plane distance (commonly used for ground robots)
        dist_xy = math.sqrt(target.x**2 + target.y**2)
        
        # Individual axis distances that could be the "forward" direction
        dist_x = abs(target.x)
        dist_y = abs(target.y)
        dist_z = abs(target.z)
        
        # Direction calculations in different possible interpretations
        dir_xy = math.degrees(math.atan2(target.y, target.x))
        dir_yz = math.degrees(math.atan2(target.z, target.y))
        dir_xz = math.degrees(math.atan2(target.z, target.x))
        
        # Compare the different calculations to how we're interpreting in target_callback
        # If these vary widely, we may have a coordinate frame issue
        
        frame_mismatch_detected = False
        
        # Check if our interpretation differs significantly from other interpretations
        if abs(self.current_distance - dist_3d) > 0.1:
            frame_mismatch_detected = True
        
        # Check which axis seems to be the primary distance component
        primary_axis = "unknown"
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
            "distance_calculations": {
                "3d_distance": round(dist_3d, 2),
                "xy_plane": round(dist_xy, 2),
                "x_axis": round(dist_x, 2),
                "y_axis": round(dist_y, 2),
                "z_axis": round(dist_z, 2)
            },
            "bearing_calculations": {
                "xy_plane": round(dir_xy, 1),
                "yz_plane": round(dir_yz, 1),
                "xz_plane": round(dir_xz, 1)
            },
            "our_interpretation": {
                "distance": round(self.current_distance, 2),
                "lateral": round(self.current_lateral, 2),
                "bearing_deg": round(math.degrees(self.current_bearing), 1)
            },
            "primary_axis": primary_axis
        }
        
        return frame_diagnostics

    def run_coordinate_frame_diagnostics(self):
        """Periodic function to check for coordinate frame issues."""
        if self._shutting_down or self.current_target is None:
            return

        # Only run full diagnostics if:
        # 1. We're in tracking mode (where it matters most)
        # 2. The frame appears to have changed 
        # 3. A minimum time has passed since last check
        current_frame = getattr(self.current_target.header, 'frame_id', None)
        frame_changed = current_frame != self._last_diagnostics_frame
        
        current_time = time.time()
        time_passed = (current_time - self.last_frame_check_time) >= 1.0/self.frame_check_rate
        
        if (self.robot_state == "tracking" and (frame_changed or time_passed)):
            # Update tracking variables
            self._last_diagnostics_frame = current_frame
            self.last_frame_check_time = current_time
            
            # Run the diagnostics
            diagnostics = self.handle_coordinate_frame_diagnostics()
            
            # Only log warnings if we detected an issue
            if diagnostics.get("detected_issue", False):
                self.get_logger().warn(f"Possible coordinate frame issue detected - {diagnostics}")
            elif self.debug_level >= 2:
                self.get_logger().debug(f"Coordinate frame diagnostics: {diagnostics}")
            
            # If we're actively tracking, log a brief summary of the coordinate frames
            if self.debug_level >= 1:
                # Extract the latest position values from state manager 
                target = self.current_target.point
                
                # Throttled logging for frame checks
                self._log_throttled(
                    self.get_logger().info,
                    f"Frame check: Target=[{target.x:.2f}, {target.y:.2f}, {target.z:.2f}], "
                    f"interpreted as dist={self.current_distance:.2f}m, "
                    f"bearing={math.degrees(self.current_bearing):.1f}°, "
                    f"primary_component={diagnostics.get('primary_axis', 'unknown')}",
                    5.0,  # Every 5 seconds max
                    'last_frame_summary_time'
                )
                
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


def main(args=None):
    """Main function to initialize and run the PID Controller node."""
    rclpy.init(args=args)
    node = PIDControllerNode()
    
    # Welcome message
    print("=================================================")
    print("Tennis Ball Tracking - Improved PID Controller Node")
    print("=================================================")
    print("This node implements three PID controllers:")
    print("1. Linear X velocity (forward/backward movement)")
    print("2. Linear Y velocity (lateral/strafing movement)")
    print("3. Angular velocity (turning/rotation)")
    print("")
    print("Optimizations:")
    print("- Memory reuse for ROS messages")
    print("- Fixed-size buffers to prevent dynamic allocations")
    print("- Throttled logging to reduce overhead")
    print("- Tiered update frequencies")
    print("- Deadzone for lateral movement")
    print("- Acceleration limiting for smoother motion")
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