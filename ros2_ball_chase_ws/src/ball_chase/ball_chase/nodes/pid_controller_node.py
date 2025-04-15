#!/usr/bin/env python3

"""
Tennis Ball Tracking Robot - PID Controller Node
===============================================

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
        "cmd_vel": "/cmd_vel",
        "diagnostics": "/pid/diagnostics"
    }
}

class PIDController:
    """
    A general-purpose PID controller implementation.
    
    This class provides a complete PID controller with:
    - Anti-windup protection to prevent integral term saturation
    - Output limiting to ensure safe operation
    - Automatic time calculation for correct derivative and integral terms
    
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
        
        # Integral term (accumulates error over time)
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term (rate of change of error)
        # Note: derivative on error, not measurement, can cause "derivative kick"
        d_term = self.kd * (error - self.prev_error) / dt
        
        # Calculate raw output by summing all terms
        output = p_term + i_term + d_term
        
        # Apply output limits
        output_limited = max(self.output_min, min(self.output_max, output))
        
        # Anti-windup: adjust integral term if output is saturated
        # This prevents integral windup when the controller cannot achieve the desired output
        if self.anti_windup and output != output_limited:
            # Reduce integral by the excess output scaled by Ki
            if abs(self.ki) > 1e-10:  # Avoid division by zero
                self.integral -= (output - output_limited) / self.ki
                # Recalculate integral term
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
    PID Controller node for tennis ball tracking.
    
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
        
        # Set up subscriptions
        self._setup_subscriptions()
        
        # Set up publishers
        self._setup_publishers()
        
        # Set up timers
        self._setup_timers()
        
        self.get_logger().info("PID Controller initialized")
        self.log_parameters()
        
    def _declare_parameters(self):
        """Declare and get all node parameters with descriptive comments."""
        self.declare_parameters(
            namespace='',
            parameters=[
                # Linear X velocity PID parameters - controls forward/backward movement
                ('linear_x_kp', 0.5),     # Proportional gain
                ('linear_x_ki', 0.1),     # Integral gain
                ('linear_x_kd', 0.05),    # Derivative gain
                ('linear_x_min', -0.2),   # Backward limit (m/s)
                ('linear_x_max', 0.2),    # Forward limit (m/s)
                
                # Linear Y velocity PID parameters - controls lateral movement (strafing)
                ('linear_y_kp', 0.5),     # Proportional gain
                ('linear_y_ki', 0.1),     # Integral gain
                ('linear_y_kd', 0.05),    # Derivative gain
                ('linear_y_min', -0.2),   # Right strafe limit (m/s)
                ('linear_y_max', 0.2),    # Left strafe limit (m/s)
                
                # Angular velocity PID parameters - controls turning
                ('angular_kp', 1.0),    # Proportional gain
                ('angular_ki', 0.1),    # Integral gain
                ('angular_kd', 0.2),    # Derivative gain
                ('angular_min', -0.2),  # Right turn limit (rad/s)
                ('angular_max', 0.2),   # Left turn limit (rad/s)
                
                # Control parameters
                ('min_distance', 0.5),       # Minimum distance to keep from ball (meters)
                ('max_distance', 2.0),       # Maximum tracking distance (meters)
                ('target_offset_x', 0.0),    # Desired offset from ball in x direction
                ('target_offset_y', 0.0),    # Desired offset from ball in y direction
                ('target_update_rate', 10.0),# Control loop update rate (Hz)
                ('debug_level', 1),          # 0=errors only, 1=info, 2=debug
                ('adaptive_gains', True),    # Whether to adjust gains based on distance
                ('use_lateral_control', True), # Whether to use Y-axis control for lateral movement
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
        self.debug_level = self.get_parameter('debug_level').value
        self.adaptive_gains = self.get_parameter('adaptive_gains').value
        self.use_lateral_control = self.get_parameter('use_lateral_control').value
        
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
        
    def _init_state_variables(self):
        """Initialize all state tracking variables."""
        # Target tracking
        self.current_target = None      # Latest target position
        self.last_target_time = None    # When we last received a target
        
        # Robot state
        self.robot_state = "initializing"  # Current state from state manager
        self.last_control_time = time.time()  # For periodic logging
        
        # Derived values
        self.current_distance = 0.0     # Current distance to target
        self.current_bearing = 0.0      # Current bearing to target
        self.current_lateral = 0.0      # Current lateral offset to target
        
        # Diagnostic information
        self.cycle_count = 0            # Number of control cycles
        self.velocity_history = []      # Recent velocity commands
        
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
        """Set up timer callbacks for periodic tasks."""
        # Create control loop timer at specified update rate
        self.timer = self.create_timer(1.0 / self.update_rate, self.control_loop_callback)
        
        # Create a slower timer for detailed diagnostics
        self.diagnostic_timer = self.create_timer(1.0, self.publish_detailed_diagnostics)
        
        # Create a timer for coordinate frame diagnostics to catch mismatches early
        self.frame_diagnostic_timer = self.create_timer(3.0, self.run_coordinate_frame_diagnostics)
        
    def log_parameters(self):
        """Log all the current parameter values for reference."""
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
        self.get_logger().info(f"  Debug level: {self.debug_level}")
        self.get_logger().info("==================================")
        
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
            self.get_logger().info(f"Robot state changed: {self.robot_state} → {new_state}")
            self.robot_state = new_state
            
            # Only reset PIDs when switching to/from tracking
            if new_state == "tracking" or self.robot_state == "tracking":
                self.pid_linear_x.reset()
                self.pid_linear_y.reset()
                self.pid_angular.reset()
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
            self.current_bearing = math.atan2(target.y, target.x)
            # FIX: In base_link frame, positive Y is left, but we need negative lateral for leftward movement
            # Negating Y to maintain the correct sign convention (negative = right, positive = left)
            self.current_lateral = -target.y
            
            # For base frame, we might need to handle z component differently
            if abs(target.z) > 0.1:  # If there's significant height difference
                self.get_logger().debug(f"Target has height component: z={target.z:.2f}m")

        # Enhanced detailed logging for every target update to catch issues early
        self.get_logger().info(
            f"Target update: raw=[{target.x:.2f}, {target.y:.2f}, {target.z:.2f}], "
            f"frame={frame_id}, "
            f"calculated: distance={self.current_distance:.2f}m, "
            f"lateral={self.current_lateral:.2f}m, "
            f"bearing={math.degrees(self.current_bearing):.1f}°"
        )
        
    def control_loop_callback(self):
        """
        Regular control loop to calculate and publish velocity commands.
        
        This is the core function that:
        1. Checks if we should be controlling the robot in the current state
        2. Calculates appropriate linear and angular velocities using PID controllers
        3. Publishes velocity commands to control the robot's motion
        """
        current_time = time.time()
        self.cycle_count += 1
        
        # Only generate commands in tracking mode with a recent target
        if self.robot_state != "tracking" or self.current_target is None:
            # When not tracking, ensure robot is stopped (unless in a state where another node controls movement)
            if self.robot_state not in ["searching", "lost_ball"]:
                self.stop_robot()
            # Log the reason we're not controlling, but only every 10 cycles to avoid log spam
            if self.debug_level >= 1 and self.cycle_count % 10 == 0:
                if self.robot_state != "tracking":
                    self.get_logger().info(f"Not controlling robot - current state: {self.robot_state}")
                elif self.current_target is None:
                    self.get_logger().info("Not controlling robot - no target data received yet")
            return
            
        # Check if target is too old (500ms timeout)
        if self.last_target_time is None or (current_time - self.last_target_time) > 0.5:
            self.get_logger().warn("Target is too old (>500ms), stopping robot")
            self.stop_robot()
            return
        
        # Log current target position and properties for diagnostics
        # This will help catch any coordinate frame mismatches
        if self.debug_level >= 2 and self.cycle_count % 10 == 0:
            raw_point = self.current_target.point
            frame_id = getattr(self.current_target.header, 'frame_id', 'unknown')
            self.get_logger().info(
                f"Raw target data: pos=[{raw_point.x:.2f}, {raw_point.y:.2f}, {raw_point.z:.2f}], "
                f"frame={frame_id}"
            )
            
        # Use the processed values from target_callback instead of recalculating
        # This ensures consistent interpretation of coordinate frames
        distance = self.current_distance
        lateral = self.current_lateral
        bearing = self.current_bearing
        
        # Log values each cycle to diagnose coordinate frame issues
        self.get_logger().info(f"Control errors: distance_error={distance - self.min_distance:.2f}m, lateral_error={lateral:.2f}m, angular_error={math.degrees(bearing):.1f}°")
        
        # Calculate desired distance (with minimum safe distance)
        desired_distance = max(self.min_distance, min(distance, self.max_distance))
        
        # Calculate errors - ball_distance is already the full 3D distance
        distance_error = distance - desired_distance - self.target_offset_x
        lateral_error = lateral - self.target_offset_y
        angular_error = bearing  # We want bearing to be 0 (centered)
        
        # Apply adaptive gains if enabled
        if self.adaptive_gains:
            self._adjust_gains_for_distance(distance)
        
        # Compute PID outputs
        linear_x_velocity = self.pid_linear_x.compute(distance_error, current_time)
        
        # Handle lateral movement based on configuration
        if self.use_lateral_control:
            # Use direct lateral control with mecanum wheels
            linear_y_velocity = self.pid_linear_y.compute(lateral_error, current_time)
            # Reduce angular velocity influence when using lateral control
            angular_scale = 0.7 
        else:
            # No lateral control, use only angular velocity for alignment
            linear_y_velocity = 0.0
            angular_scale = 1.0
            
        angular_velocity = self.pid_angular.compute(angular_error, current_time) * angular_scale
        
        # Create velocity command for mecanum wheels
        cmd_vel = Twist()
        cmd_vel.linear.x = linear_x_velocity   # Forward/backward
        cmd_vel.linear.y = linear_y_velocity   # Left/right strafe (positive = left)
        cmd_vel.angular.z = angular_velocity   # Rotation (positive = counterclockwise)
        
        # Log velocity values less frequently
        if self.debug_level >= 1 and self.cycle_count % 20 == 0:
            self.get_logger().info(
                f"[#{self.cycle_count}] Velocity cmd: linear_x={linear_x_velocity:.3f}m/s, "
                f"linear_y={linear_y_velocity:.3f}m/s, angular={angular_velocity:.3f}rad/s"
            )
        
        # Save for history
        self.velocity_history.append((linear_x_velocity, linear_y_velocity, angular_velocity))
        if len(self.velocity_history) > 20:  # Keep last 20 commands
            self.velocity_history.pop(0)
        
        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Publish basic diagnostics every cycle
        self.publish_basic_diagnostics(distance_error, lateral_error, angular_error,
                                      linear_x_velocity, linear_y_velocity, angular_velocity)
        
        # Log periodic status (approximately once per second)
        if self.debug_level >= 1 and (current_time - self.last_control_time) >= 1.0:
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
            
    def _adjust_gains_for_distance(self, distance):
        """
        Adjust PID gains based on distance to target.
        
        This makes the controller more aggressive when the ball is far away
        and more gentle when it's close.
        
        Args:
            distance (float): Current distance to target in meters
        """
        # Scale factor based on distance (1.0 at max_distance, 0.5 at min_distance)
        scale = 0.5 + 0.5 * min(1.0, max(0.0, (distance - self.min_distance) / 
                                        (self.max_distance - self.min_distance)))
        
        # Apply scaling to controllers
        # Linear X controller: less aggressive when close
        self.pid_linear_x.kp = self.linear_x_kp * scale
        self.pid_linear_x.ki = self.linear_x_ki * scale
        
        # Linear Y controller: less aggressive when close
        self.pid_linear_y.kp = self.linear_y_kp * scale
        self.pid_linear_y.ki = self.linear_y_ki * scale
        
        # Angular controller: more precise when close
        precision_scale = 1.5 - 0.5 * scale  # 1.25 when close, 1.0 when far
        self.pid_angular.kp = self.angular_kp * precision_scale
            
    def stop_robot(self):
        """Send a command to stop all robot motion immediately."""
        cmd_vel = Twist()  # All fields initialize to 0
        self.cmd_vel_pub.publish(cmd_vel)
        
        # Clear velocity history
        self.velocity_history = []
        
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
        diag_msg = Float32MultiArray()
        diag_msg.data = [
            distance_error,
            lateral_error,
            angular_error,
            linear_x_velocity,
            linear_y_velocity,
            angular_velocity,
            self.pid_linear_x.integral,
            self.pid_linear_y.integral,
            self.pid_angular.integral
        ]
        self.pid_diag_pub.publish(diag_msg)
        
    def publish_detailed_diagnostics(self):
        """
        Publish comprehensive diagnostic information at a slower rate.
        
        This provides more detailed information for debugging and tuning,
        but at a lower frequency to avoid flooding the system.
        """
        if not self.robot_state == "tracking":
            return
            
        # Calculate velocity statistics
        if self.velocity_history:
            # Extract linear and angular velocities
            lin_x_velocities = [v[0] for v in self.velocity_history]
            lin_y_velocities = [v[1] for v in self.velocity_history]
            ang_velocities = [v[2] for v in self.velocity_history]
            
            # Calculate statistics
            avg_lin_x_vel = sum(lin_x_velocities) / len(lin_x_velocities)
            avg_lin_y_vel = sum(lin_y_velocities) / len(lin_y_velocities)
            avg_ang_vel = sum(ang_velocities) / len(ang_velocities)
            max_lin_x_vel = max(lin_x_velocities)
            max_lin_y_vel = max(lin_y_velocities) if any(lin_y_velocities) else 0
            max_ang_vel = max(ang_velocities)
            
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
            
            self.get_logger().info(f"Control cycles: {self.cycle_count}")
            self.get_logger().info("================================")
            
    def handle_coordinate_frame_diagnostics(self):
        """
        Special diagnostic function to detect and report coordinate frame issues.
        
        This function analyzes the latest target data and logs detailed information
        about potential coordinate frame mismatches or interpretation issues.
        """
        if self.current_target is None:
            return
            
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
        
        # Log warning if we detect a potential coordinate frame mismatch
        if frame_mismatch_detected:
            self.get_logger().warn(f"Possible coordinate frame issue detected - {frame_diagnostics}")
        else:
            self.get_logger().debug(f"Coordinate frame diagnostics: {frame_diagnostics}")
            
        return frame_diagnostics

    def run_coordinate_frame_diagnostics(self):
        """Periodic function to check for coordinate frame issues."""
        if self.current_target is None:
            return

        # Run the diagnostics
        diagnostics = self.handle_coordinate_frame_diagnostics()
        
        # If we're actively tracking, log a bit more detail about the coordinate frames
        if self.robot_state == "tracking":
            # Extract the latest position values from state manager 
            target = self.current_target.point
            
            # Additional logging to help diagnose frame issues
            self.get_logger().info(
                f"Frame check: Target=[{target.x:.2f}, {target.y:.2f}, {target.z:.2f}], "
                f"interpreted as dist={self.current_distance:.2f}m, "
                f"bearing={math.degrees(self.current_bearing):.1f}°, "
                f"primary_component={diagnostics['primary_axis'] if diagnostics else 'unknown'}"
            )


def main(args=None):
    """Main function to initialize and run the PID Controller node."""
    rclpy.init(args=args)
    node = PIDControllerNode()
    
    # Welcome message
    print("=================================================")
    print("Tennis Ball Tracking - PID Controller Node")
    print("=================================================")
    print("This node implements three PID controllers:")
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
        node.get_logger().info("Shutdown handler called, stopping robot...")
        # Ensure robot stops on shutdown
        node.stop_robot()
        node.destroy_node()
    
    # Register shutdown handler
    rclpy.get_global_executor().add_node(node)
    rclpy.get_default_context().on_shutdown(shutdown_handler)
    
    # Register signal handlers for proper shutdown
    def signal_handler(sig, frame):
        node.get_logger().info(f"Signal {sig} received, stopping robot...")
        shutdown_handler()
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
        node.stop_robot()
        node.get_logger().info("Robot motion stopped - setting velocity and rotation to 0")
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()