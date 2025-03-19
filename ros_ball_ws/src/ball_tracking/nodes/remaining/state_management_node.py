#!/usr/bin/env python3

"""
Tennis Ball Tracking Robot - State Management Node
=================================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving tennis ball.
The system uses multiple sensing modalities which are combined in a fusion node, and the
state management node (this one) determines the robot's behavior based on tracking quality.

This Node's Purpose:
------------------
The state management node serves as the "brain" of the robot, deciding what the robot should
do at any given moment based on sensor data about the tennis ball. It transitions between
different operational states and executes appropriate behaviors for each state.

States:
------
1. INITIALIZING: Starting up, waiting for first reliable detection
2. TRACKING: Actively following the tennis ball when it's reliably detected
3. SEARCHING: Executing a methodical search pattern when the ball is temporarily lost
4. LOST_BALL: Stationary waiting mode when the ball cannot be found after extensive searching
5. STOPPED: Robot stops when the ball is very close and not moving

Data Pipeline:
-------------
1. Fusion node publishes to:
   - '/tennis_ball/fused/position' (3D position)
   - '/tennis_ball/fused/tracking_status' (reliability)
   - '/tennis_ball/fused/diagnostics' (detailed info)

2. This state manager:
   - Subscribes to fusion node outputs
   - Determines appropriate robot state
   - Publishes target position for PID controller when tracking
   - Publishes movement commands when searching
   - Publishes current state for other nodes

3. Next in pipeline:
   - PID controller handles actual motor control
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TwistStamped, Twist
from std_msgs.msg import String, Bool, Float32
import numpy as np
import time
import math
import json

# Define robot states as an enumeration
class RobotState:
    """Enumeration of robot operational states"""
    INITIALIZING = "initializing"  # Startup state, waiting for first reliable detection
    TRACKING = "tracking"          # Actively tracking the ball with reliable detections
    SEARCHING = "searching"        # Looking for a lost ball using search patterns
    LOST_BALL = "lost_ball"        # Ball not found after extensive searching
    STOPPED = "stopped"            # Stationary state (e.g., emergency stop or close to ball)

# Define search pattern phases
class SearchPhase:
    """Enumeration of search pattern phases"""
    ROTATE = "rotate"              # Rotating to a new angle
    PAUSE_AFTER_ROTATE = "pause1"  # Pausing after rotation to let sensors catch up
    FORWARD = "forward"            # Moving forward to explore
    PAUSE_AFTER_FORWARD = "pause2" # Pausing after forward movement
    BACKWARD = "backward"          # Moving backward to original position
    PAUSE_AFTER_BACKWARD = "pause3" # Pausing after backward movement

# Topic configuration (ensures consistency with other nodes)
TOPICS = {
    "input": {
        "position": "/tennis_ball/fused/position",
        "tracking_status": "/tennis_ball/fused/tracking_status",
        "diagnostics": "/tennis_ball/fused/diagnostics"
    },
    "output": {
        "cmd_vel": "/cmd_vel",
        "target": "/tennis_ball/target",
        "state": "/robot/state"
    }
}

class StateManagerNode(Node):
    """
    Manages the robot's operational states based on ball tracking quality.
    
    This node serves as the decision-making center that:
    1. Monitors ball tracking reliability from the fusion node
    2. Determines the appropriate robot state (tracking/searching/etc.)
    3. Executes an intelligent search pattern when the ball is lost
    4. Forwards position targets to the PID controller when tracking
    5. Provides diagnostic information about state transitions
    6. Detects when ball is very close and stationary to safely stop
    
    The search pattern is designed to account for sensor processing latency
    on the Raspberry Pi 5, incorporating pauses between movements to allow
    sensor updates to complete.
    """
    
    def __init__(self):
        """Initialize the state manager with all required components."""
        super().__init__('state_manager')
        
        # Set up parameters
        self._declare_parameters()
        
        # Initialize state variables
        self._init_state_variables()
        
        # Set up subscriptions
        self._setup_subscriptions()
        
        # Set up publishers
        self._setup_publishers()
        
        # Set up timers
        self._setup_timers()
        
        self.get_logger().info("State Manager initialized in INITIALIZING state")
        self.publish_state()
        self.log_parameters()

    def _declare_parameters(self):
        """Declare and get all node parameters with descriptive comments."""
        self.declare_parameters(
            namespace='',
            parameters=[
                # Timing thresholds
                ('lost_ball_timeout', 1.0),       # Seconds without detection to consider ball lost
                ('max_search_time', 30.0),        # Seconds to search before giving up and entering LOST_BALL state
                ('stationary_time_threshold', 1.5), # Time ball needs to be stationary before stopping (seconds)
                
                # Search pattern parameters
                ('search_rotation_speed', 0.3),   # Rotation speed during search (rad/s)
                ('search_forward_speed', 0.15),   # Forward speed during search (m/s)
                ('search_rotation_angle', 30.0),  # Initial rotation increment in degrees
                ('search_forward_distance', 0.3), # Forward movement distance in meters
                ('pause_duration', 1.2),          # Pause duration to let sensors process (seconds)
                ('max_rotation_angle', 180.0),    # Maximum rotation angle in degrees
                
                # Detection and tracking thresholds
                ('min_tracking_detections', 3),   # Consecutive detections to confirm tracking
                ('position_threshold', 0.5),      # Position threshold (meters)
                ('proximity_threshold', 0.3),     # Distance to consider ball as "close" (meters)
                ('stationary_threshold', 0.03),   # Maximum movement to consider ball "stationary" (meters)
                
                # Debugging and data collection
                ('debug_level', 1),               # 0=errors only, 1=info, 2=debug
                ('position_history_length', 10),  # Number of positions to keep in history
            ]
        )
        
        # Get all parameters
        self.lost_ball_timeout = self.get_parameter('lost_ball_timeout').value
        self.max_search_time = self.get_parameter('max_search_time').value
        self.stationary_time_threshold = self.get_parameter('stationary_time_threshold').value
        
        self.search_rotation_speed = self.get_parameter('search_rotation_speed').value
        self.search_forward_speed = self.get_parameter('search_forward_speed').value
        self.search_rotation_angle = self.get_parameter('search_rotation_angle').value
        self.search_forward_distance = self.get_parameter('search_forward_distance').value
        self.pause_duration = self.get_parameter('pause_duration').value
        self.max_rotation_angle = self.get_parameter('max_rotation_angle').value
        
        self.min_tracking_detections = self.get_parameter('min_tracking_detections').value
        self.position_threshold = self.get_parameter('position_threshold').value
        self.proximity_threshold = self.get_parameter('proximity_threshold').value
        self.stationary_threshold = self.get_parameter('stationary_threshold').value
        
        self.debug_level = self.get_parameter('debug_level').value
        self.position_history_length = self.get_parameter('position_history_length').value

    def _init_state_variables(self):
        """Initialize all state tracking variables."""
        # Primary state
        self.current_state = RobotState.INITIALIZING
        self.state_start_time = time.time()
        self.last_state_change_time = time.time()
        
        # Detection tracking
        self.last_detection_time = None
        self.consecutive_detections = 0
        self.position_uncertainty = float('inf')
        self.last_position = None
        self.tracking_reliable = False
        
        # Ball proximity and movement detection
        self.position_history = []  # List of (position, timestamp) tuples
        self.ball_distance = float('inf')  # Distance to ball in meters
        self.is_ball_close = False  # Flag for when ball is within proximity threshold
        self.is_ball_stationary = False  # Flag for when ball hasn't moved significantly
        self.stationary_start_time = None  # When the ball was first detected as stationary
        
        # Search pattern variables
        self.search_direction = 1   # 1 or -1 for clockwise/counterclockwise
        self.search_cycle_count = 0 # Number of complete search cycles
        self.current_search_phase = SearchPhase.ROTATE
        self.search_phase_start_time = time.time()
        self.search_step_complete = False
        self.current_rotation_angle = self.search_rotation_angle  # Current rotation angle in degrees
        self.current_forward_distance = self.search_forward_distance  # Current forward distance in meters
        self.cumulative_rotation = 0.0  # Track total rotation during search to detect 360° completion
        
        # Motion tracking for state search
        self.rotation_target = 0.0  # Target rotation in radians
        self.distance_target = 0.0  # Target distance in meters
        self.motion_start_time = 0.0  # Start time for current motion
        
        # Recovery attempt tracking
        self.recovery_attempts = 0  # Number of times we've tried to recover from LOST_BALL state
        self.last_recovery_time = 0.0  # When we last attempted recovery
        
        # Total search time before giving up
        self.total_search_time = 0.0  # Accumulated time spent searching

    def _setup_subscriptions(self):
        """Set up all subscriptions for this node."""
        # Subscribe to ball position from fusion node
        self.position_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["position"],
            self.position_callback,
            10
        )
        
        # Subscribe to tracking reliability indicator
        self.tracking_status_sub = self.create_subscription(
            Bool,
            TOPICS["input"]["tracking_status"],
            self.tracking_status_callback,
            10
        )
        
        # Subscribe to fusion node diagnostics
        self.diagnostics_sub = self.create_subscription(
            String,
            TOPICS["input"]["diagnostics"],
            self.diagnostics_callback,
            10
        )

    def _setup_publishers(self):
        """Set up all publishers for this node."""
        # Publishers
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            TOPICS["output"]["cmd_vel"],
            10
        )
        
        self.target_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["target"],
            10
        )
        
        self.state_publisher = self.create_publisher(
            String,
            TOPICS["output"]["state"],
            10
        )

    def _setup_timers(self):
        """Set up timer callbacks for periodic tasks."""
        # Create timers
        self.timer = self.create_timer(0.1, self.state_manager_callback)  # 10Hz state management
        self.diagnostic_timer = self.create_timer(1.0, self.publish_diagnostics)  # 1Hz diagnostics

    def log_parameters(self):
        """Log all the current parameter values for reference."""
        self.get_logger().info("=== State Manager Parameters ===")
        self.get_logger().info("Timing thresholds:")
        self.get_logger().info(f"  Lost ball timeout: {self.lost_ball_timeout} seconds")
        self.get_logger().info(f"  Max search time: {self.max_search_time} seconds")
        self.get_logger().info(f"  Stationary time threshold: {self.stationary_time_threshold} seconds")
        
        self.get_logger().info("Search pattern parameters:")
        self.get_logger().info(f"  Rotation speed: {self.search_rotation_speed} rad/s")
        self.get_logger().info(f"  Forward speed: {self.search_forward_speed} m/s")
        self.get_logger().info(f"  Initial rotation angle: {self.search_rotation_angle} degrees")
        self.get_logger().info(f"  Forward distance: {self.search_forward_distance} meters")
        self.get_logger().info(f"  Pause duration: {self.pause_duration} seconds")
        self.get_logger().info(f"  Max rotation angle: {self.max_rotation_angle} degrees")
        
        self.get_logger().info("Detection thresholds:")
        self.get_logger().info(f"  Min tracking detections: {self.min_tracking_detections}")
        self.get_logger().info(f"  Position threshold: {self.position_threshold} meters")
        self.get_logger().info(f"  Proximity threshold: {self.proximity_threshold} meters")
        self.get_logger().info(f"  Stationary threshold: {self.stationary_threshold} meters")
        
        self.get_logger().info("Debug settings:")
        self.get_logger().info(f"  Debug level: {self.debug_level}")
        self.get_logger().info(f"  Position history length: {self.position_history_length}")
        self.get_logger().info("===============================")

    def tracking_status_callback(self, msg):
        """
        Process tracking reliability flag from the fusion node.
        
        Args:
            msg (Bool): Whether tracking is reliable according to fusion node
        """
        self.tracking_reliable = msg.data
        
        if self.debug_level >= 2:
            self.get_logger().debug(f"Tracking status: {self.tracking_reliable}")

    def position_callback(self, msg):
        """
        Process ball position updates from the fusion node.
        
        This method is triggered whenever we receive a new position update from
        the sensor fusion node. It updates our knowledge about the ball and
        potentially triggers state transitions.
        
        Args:
            msg (PointStamped): 3D position of the ball
        """
        current_time = time.time()
        
        # Extract position
        position = np.array([msg.point.x, msg.point.y, msg.point.z])
        
        # Update last detection time
        self.last_detection_time = current_time
        
        # Update position history for stationary detection
        self.position_history.append((position, current_time))
        if len(self.position_history) > self.position_history_length:
            self.position_history.pop(0)
        
        # Calculate position change if we have a previous position
        if self.last_position is not None:
            position_change = np.linalg.norm(position - self.last_position)
            
            # If position is changing reasonably, count as valid detection
            if position_change < 1.0:  # Max 1 meter change between updates
                self.consecutive_detections += 1
            else:
                # Large jump might be a false positive
                self.consecutive_detections = max(0, self.consecutive_detections - 1)
                if self.debug_level >= 1:
                    self.get_logger().info(f"Large position jump detected: {position_change:.2f}m")
        else:
            # First detection
            self.consecutive_detections = 1
        
        # Extract the distance to the ball (typically z-coordinate)
        self.ball_distance = np.linalg.norm(position)
        
        # Update close ball detection
        self.is_ball_close = self.ball_distance <= self.proximity_threshold
        
        # Update stationary ball detection
        self.check_if_ball_stationary()
            
        # Store current position for next comparison
        self.last_position = position
        
        # Forward position to target if we're in tracking mode
        if self.current_state == RobotState.TRACKING:
            self.publish_target(msg)
            
        # Handle state transitions based on position updates
        self._handle_position_based_transitions(current_time)

    def _handle_position_based_transitions(self, current_time):
        """
        Handle state transitions based on new position information.
        
        Args:
            current_time (float): Current timestamp
        """
        # Transition from INITIALIZING to TRACKING if we have reliable detections
        if self.current_state == RobotState.INITIALIZING:
            if self.consecutive_detections >= self.min_tracking_detections and self.tracking_reliable:
                self.transition_to_state(RobotState.TRACKING)
                
        # Transition from SEARCHING to TRACKING if we find the ball again
        elif self.current_state == RobotState.SEARCHING:
            if self.consecutive_detections >= self.min_tracking_detections and self.tracking_reliable:
                self.transition_to_state(RobotState.TRACKING)
                
        # Transition from LOST_BALL to TRACKING if the ball reappears
        elif self.current_state == RobotState.LOST_BALL:
            if self.consecutive_detections >= self.min_tracking_detections and self.tracking_reliable:
                self.transition_to_state(RobotState.TRACKING)
                
        # Handle transition to STOPPED state if ball is close and stationary
        elif self.current_state == RobotState.TRACKING:
            if self.is_ball_close and self.is_ball_stationary:
                if self.stationary_start_time is None:
                    # First time detecting stationary ball
                    self.stationary_start_time = current_time
                elif current_time - self.stationary_start_time >= self.stationary_time_threshold:
                    # Ball has been stationary and close for the required time
                    self.transition_to_state(RobotState.STOPPED)
                    self.get_logger().info(f"Ball is close ({self.ball_distance:.2f}m) and stationary - stopping")
            else:
                # Reset stationary timer if conditions aren't met
                self.stationary_start_time = None
        
        # Handle transition back from STOPPED state if ball moves or is no longer close
        elif self.current_state == RobotState.STOPPED:
            if not self.is_ball_close or not self.is_ball_stationary:
                reason = "moved away" if not self.is_ball_close else "started moving"
                self.get_logger().info(f"Ball has {reason} - resuming tracking")
                self.transition_to_state(RobotState.TRACKING)

    def check_if_ball_stationary(self):
        """
        Check if the ball hasn't moved significantly over recent history.
        
        This method analyzes the position history to determine if the ball
        has been relatively stationary over time.
        """
        if len(self.position_history) < 3:  # Need a minimum history to determine if stationary
            self.is_ball_stationary = False
            return
        
        # Get the most recent position
        latest_position, _ = self.position_history[-1]
        
        # Check movement against all positions in history
        max_movement = 0.0
        for pos, _ in self.position_history:
            movement = np.linalg.norm(latest_position - pos)
            max_movement = max(max_movement, movement)
        
        # Ball is stationary if maximum movement is below threshold
        self.is_ball_stationary = max_movement <= self.stationary_threshold
        
        if self.debug_level >= 2:
            self.get_logger().debug(
                f"Ball stationary check: max_movement={max_movement:.3f}m, "
                f"is_stationary={self.is_ball_stationary}"
            )

    def diagnostics_callback(self, msg):
        """
        Process diagnostic information from the fusion node.
        
        Args:
            msg (String): Diagnostic information in JSON format
        """
        try:
            # Parse diagnostic information (new format is JSON)
            info = json.loads(msg.data)
            
            # Update position uncertainty if available
            if "position" in info and "uncertainty" in info["position"]:
                self.position_uncertainty = info["position"]["uncertainty"]
            
            # Get sensor freshness info if available
            if "sensors" in info:
                sensor_ages = {}
                for sensor, data in info["sensors"].items():
                    if "age" in data:
                        sensor_ages[sensor] = data["age"]
                
                # Count fresh sensors (less than timeout seconds old)
                fresh_sensors = sum(1 for age in sensor_ages.values() 
                                  if age < self.lost_ball_timeout)
                
                if self.debug_level >= 2:
                    self.get_logger().debug(f"Fresh sensors: {fresh_sensors}")
                    
        except json.JSONDecodeError:
            # Fall back to old string parsing format for backward compatibility
            try:
                # Parse diagnostic information in key:value;key:value format
                info = {}
                for item in msg.data.split(';'):
                    if ':' in item:
                        key, value = item.split(':')
                        # Convert numeric values to float, boolean-like to bool
                        if key in ['tracking_reliable']:
                            info[key] = int(value) == 1
                        else:
                            try:
                                info[key] = float(value)
                            except ValueError:
                                info[key] = value
                
                # Update position uncertainty if available
                if 'uncertainty' in info:
                    self.position_uncertainty = info['uncertainty']
                
                # Get sensor freshness info
                sensor_freshness = {
                    'hsv_2d': info.get('hsv_2d_age', float('inf')),
                    'yolo_2d': info.get('yolo_2d_age', float('inf')),
                    'hsv_3d': info.get('hsv_3d_age', float('inf')),
                    'yolo_3d': info.get('yolo_3d_age', float('inf')),
                    'lidar': info.get('lidar_age', float('inf'))
                }
                
                # Count fresh sensors (less than timeout seconds old)
                fresh_sensors = sum(1 for age in sensor_freshness.values() 
                                  if age < self.lost_ball_timeout)
                
                if self.debug_level >= 2:
                    self.get_logger().debug(f"Fresh sensors: {fresh_sensors}")
            except Exception as e:
                self.get_logger().error(f"Error parsing diagnostics string format: {str(e)}")
        except Exception as e:
            self.get_logger().error(f"Error processing diagnostics: {str(e)}")
            
    def state_manager_callback(self):
        """
        Regular timer callback to manage state transitions and actions.
        
        This is called at 10Hz to:
        1. Monitor tracking quality and perform state transitions
        2. Execute appropriate actions for the current state
        3. Handle timeouts and state transitions
        """
        current_time = time.time()
        
        if self.current_state == RobotState.TRACKING:
            # Check if ball is lost based on tracking reliability and timeout
            time_since_detection = (current_time - self.last_detection_time 
                                   if self.last_detection_time is not None else float('inf'))
            
            if not self.tracking_reliable or time_since_detection > self.lost_ball_timeout:
                reason = "unreliable tracking" if not self.tracking_reliable else "detection timeout"
                self.get_logger().info(f"Ball lost! Reason: {reason}")
                self.transition_to_state(RobotState.SEARCHING)
                
        elif self.current_state == RobotState.SEARCHING:
            # Execute the current search pattern
            self.execute_search_pattern()
            
            # Check if we've been searching too long and should give up
            search_duration = current_time - self.state_start_time
            self.total_search_time += 0.1  # Add time since this runs at 10Hz
            
            # Check for 360-degree rotation completion
            if self.cumulative_rotation >= 360.0:
                self.get_logger().info("Search completed a full 360-degree rotation")
                if self.consecutive_detections < self.min_tracking_detections:
                    # If we still haven't found the ball after a full rotation, give up
                    self.transition_to_state(RobotState.LOST_BALL)
                    self.get_logger().info("Full rotation search complete, no ball found. Entering LOST_BALL state.")
            
            # Also transition to LOST_BALL if we've searched for too long
            elif self.total_search_time >= self.max_search_time:
                self.transition_to_state(RobotState.LOST_BALL)
                self.get_logger().info(f"Search timeout after {self.total_search_time:.1f} seconds. Entering LOST_BALL state.")
                
        elif self.current_state == RobotState.INITIALIZING:
            # Check if we should timeout initialization
            time_in_state = current_time - self.state_start_time
            if time_in_state > 5.0:  # 5 seconds to initialize
                self.transition_to_state(RobotState.SEARCHING)
                
        elif self.current_state == RobotState.LOST_BALL:
            # We are stopped because we couldn't find the ball
            # Occasionally try a brief recovery search
            time_in_lost_state = current_time - self.state_start_time
            time_since_last_recovery = current_time - self.last_recovery_time
            
            # Every 30 seconds, do a brief recovery search
            if time_since_last_recovery > 30.0:
                self.get_logger().info(f"Attempting recovery search after {time_in_lost_state:.1f} seconds in LOST_BALL state")
                self.last_recovery_time = current_time
                self.recovery_attempts += 1
                
                # Execute a quick rotation to look around
                twist = Twist()
                twist.angular.z = self.search_rotation_speed
                self.cmd_vel_publisher.publish(twist)
                
                # After 3 seconds, stop again
                rclpy.spin_once(self, timeout_sec=3.0)
                self.stop_robot()
                
        elif self.current_state == RobotState.STOPPED:
            # We are stopped because ball is close and stationary
            # No action needed - transitions handled in position_callback
            pass

    def publish_target(self, position_msg):
        """
        Publish the target position for the PID controller.
        
        Args:
            position_msg (PointStamped): The position message to forward
        """
        # Forward the position as a target
        self.target_publisher.publish(position_msg)
        
    def execute_search_pattern(self):
        """
        Execute a methodical search pattern designed to account for sensor processing latency.
        
        The pattern:
        1. Rotates in small increments (with pauses between to process sensor data)
        2. Moves forward a short distance (with pauses)
        3. Moves backward to original position (with pauses)
        4. Repeats with progressively wider rotations
        5. After a full 360-degree rotation, gives up and enters LOST_BALL state
        
        This pattern accounts for the Raspberry Pi's processing limitations by
        giving sensors time to update between movements.
        """
        current_time = time.time()
        phase_duration = current_time - self.search_phase_start_time
        
        # Execute the appropriate phase of the search pattern
        if self.current_search_phase == SearchPhase.ROTATE:
            if not self.search_step_complete:
                # Calculate rotation parameters
                # Convert degrees to radians
                angle_radians = math.radians(self.current_rotation_angle) * self.search_direction
                duration = abs(angle_radians) / self.search_rotation_speed
                
                # Send rotation command
                twist = Twist()
                twist.angular.z = self.search_rotation_speed * self.search_direction
                self.cmd_vel_publisher.publish(twist)
                
                # Check if rotation is complete
                if phase_duration >= duration:
                    self.stop_robot()
                    self.search_step_complete = True
                    
                    # Track cumulative rotation for detecting 360° completion
                    self.cumulative_rotation += self.current_rotation_angle
                    
                    self.get_logger().debug(
                        f"Rotation complete: {self.current_rotation_angle}° "
                        f"{'clockwise' if self.search_direction < 0 else 'counter-clockwise'}, "
                        f"cumulative: {self.cumulative_rotation}°"
                    )
            else:
                # Move to next phase
                self.current_search_phase = SearchPhase.PAUSE_AFTER_ROTATE
                self.search_phase_start_time = current_time
                self.search_step_complete = False
                self.get_logger().debug("Search phase: Rotate → Pause1")
                
        elif self.current_search_phase == SearchPhase.PAUSE_AFTER_ROTATE:
            # Just wait during the pause to allow sensors to process
            if phase_duration >= self.pause_duration:
                self.current_search_phase = SearchPhase.FORWARD
                self.search_phase_start_time = current_time
                self.search_step_complete = False
                self.get_logger().debug("Search phase: Pause1 → Forward")
                
        elif self.current_search_phase == SearchPhase.FORWARD:
            if not self.search_step_complete:
                # Calculate forward movement parameters
                duration = self.current_forward_distance / self.search_forward_speed
                
                # Send forward command
                twist = Twist()
                twist.linear.x = self.search_forward_speed
                self.cmd_vel_publisher.publish(twist)
                
                # Check if forward movement is complete
                if phase_duration >= duration:
                    self.stop_robot()
                    self.search_step_complete = True
                    self.get_logger().debug(f"Forward movement complete: {self.current_forward_distance}m")
            else:
                # Move to next phase
                self.current_search_phase = SearchPhase.PAUSE_AFTER_FORWARD
                self.search_phase_start_time = current_time
                self.search_step_complete = False
                self.get_logger().debug("Search phase: Forward → Pause2")
                
        elif self.current_search_phase == SearchPhase.PAUSE_AFTER_FORWARD:
            # Just wait during the pause
            if phase_duration >= self.pause_duration:
                self.current_search_phase = SearchPhase.BACKWARD
                self.search_phase_start_time = current_time
                self.search_step_complete = False
                self.get_logger().debug("Search phase: Pause2 → Backward")
                
        elif self.current_search_phase == SearchPhase.BACKWARD:
            if not self.search_step_complete:
                # Calculate backward movement parameters
                duration = self.current_forward_distance / self.search_forward_speed
                
                # Send backward command
                twist = Twist()
                twist.linear.x = -self.search_forward_speed
                self.cmd_vel_publisher.publish(twist)
                
                # Check if backward movement is complete
                if phase_duration >= duration:
                    self.stop_robot()
                    self.search_step_complete = True
                    self.get_logger().debug(f"Backward movement complete: {self.current_forward_distance}m")
            else:
                # Move to next phase
                self.current_search_phase = SearchPhase.PAUSE_AFTER_BACKWARD
                self.search_phase_start_time = current_time
                self.search_step_complete = False
                self.get_logger().debug("Search phase: Backward → Pause3")
                
        elif self.current_search_phase == SearchPhase.PAUSE_AFTER_BACKWARD:
            # Just wait during the pause
            if phase_duration >= self.pause_duration:
                # One complete search cycle finished
                self.search_cycle_count += 1
                
                # Every 3 cycles, change rotation direction
                if self.search_cycle_count % 3 == 0:
                    self.search_direction *= -1
                    self.get_logger().info(f"Search cycle {self.search_cycle_count}: Changing direction")
                
                # Increase rotation angle gradually (up to max specified degrees)
                rotation_increment = min(self.search_cycle_count * 5, 30)  # 5 degree increments, max 30 additional degrees
                self.current_rotation_angle = min(self.search_rotation_angle + rotation_increment, self.max_rotation_angle)
                
                # Back to rotate phase to start next cycle
                self.current_search_phase = SearchPhase.ROTATE
                self.search_phase_start_time = current_time
                self.search_step_complete = False
                self.get_logger().debug(f"Search phase: Pause3 → Rotate (Cycle {self.search_cycle_count + 1})")

    def transition_to_state(self, new_state):
        """
        Handle state transitions with proper logging and actions.
        
        This method ensures clean transitions between states, performs any
        necessary cleanup actions for the old state, and initializes the 
        new state properly.
        
        Args:
            new_state (str): The state to transition to
        """
        if new_state == self.current_state:
            return
            
        # Log the transition with timing information
        time_in_prev_state = time.time() - self.state_start_time
        self.get_logger().info(
            f"State transition: {self.current_state} → {new_state} "
            f"(after {time_in_prev_state:.1f}s)"
        )
        
        # Handle exit actions for the current state
        if self.current_state == RobotState.SEARCHING:
            # Stop movement when leaving search mode
            self.stop_robot()
            
            # Record total search time
            self.total_search_time += time_in_prev_state
            
        elif self.current_state == RobotState.LOST_BALL:
            # Reset recovery attempt counter when leaving lost ball state
            self.recovery_attempts = 0
            
        # Store previous state for potential recovery actions
        previous_state = self.current_state
            
        # Update state and reset state timer
        self.current_state = new_state
        self.state_start_time = time.time()
        self.last_state_change_time = time.time()
        
        # Handle entry actions for the new state
        if new_state == RobotState.TRACKING:
            self.get_logger().info("Ball tracking initiated")
            
        elif new_state == RobotState.SEARCHING:
            self.get_logger().info("Starting methodical ball search pattern")
            
            # If coming from LOST_BALL, we need a more aggressive search
            if previous_state == RobotState.LOST_BALL:
                self.get_logger().info("Starting recovery search with higher rotation speed")
                # Use more aggressive search parameters for recovery
                self.current_rotation_angle = self.search_rotation_angle * 2
                self.search_rotation_speed *= 1.5
            else:
                # Reset search parameters to defaults
                self.current_rotation_angle = self.search_rotation_angle
            
            # Reset search tracking variables
            self.consecutive_detections = 0
            self.search_cycle_count = 0
            self.current_search_phase = SearchPhase.ROTATE
            self.search_phase_start_time = time.time()
            self.search_step_complete = False
            self.current_forward_distance = self.search_forward_distance
            self.cumulative_rotation = 0.0  # Reset rotation counter
            
        elif new_state == RobotState.LOST_BALL:
            self.get_logger().info(
                f"Ball not found after {self.total_search_time:.1f}s of searching. "
                f"Entering stationary wait mode."
            )
            self.stop_robot()  # Make sure robot is stopped
            self.last_recovery_time = time.time()  # Initialize recovery timer
            
        elif new_state == RobotState.STOPPED:
            self.get_logger().info("Ball is close and stationary - stopping robot")
            self.stop_robot()  # Make sure robot is stopped
            
        # Publish the new state
        self.publish_state()
            
    def stop_robot(self):
        """
        Send a command to stop all robot motion immediately.
        
        This publishes a zero-velocity command to ensure the robot stops.
        """
        twist = Twist()  # All fields initialize to 0
        self.cmd_vel_publisher.publish(twist)
        if self.debug_level >= 2:
            self.get_logger().debug("Robot motion stopped")
        
    def publish_state(self):
        """
        Publish the current robot state for other nodes to consume.
        
        This allows other nodes (like a visualization or UI) to know
        what the robot is currently doing.
        """
        msg = String()
        msg.data = self.current_state
        self.state_publisher.publish(msg)
        
    def publish_diagnostics(self):
        """
        Publish comprehensive diagnostic information about the state manager.
        
        This provides detailed status information for monitoring and debugging.
        """
        if self.last_detection_time is not None:
            time_since_detection = time.time() - self.last_detection_time
        else:
            time_since_detection = float('inf')
            
        state_duration = time.time() - self.state_start_time
        
        # Create a structured diagnostic log
        self.get_logger().info("=== State Manager Status ===")
        self.get_logger().info(f"Current state: {self.current_state} (for {state_duration:.1f}s)")
        
        # Format tracking information
        self.get_logger().info(
            f"Ball tracking: reliable={self.tracking_reliable}, "
            f"uncertainty={self.position_uncertainty:.3f}m"
        )
        self.get_logger().info(
            f"Detection stats: consecutive={self.consecutive_detections}, "
            f"time_since_last={time_since_detection:.2f}s"
        )
        self.get_logger().info(
            f"Ball proximity: distance={self.ball_distance:.2f}m, "
            f"close={self.is_ball_close}, stationary={self.is_ball_stationary}"
        )
        
        # Add state-specific diagnostics
        if self.current_state == RobotState.SEARCHING:
            # Show search pattern details
            self.get_logger().info(
                f"Search status: cycle={self.search_cycle_count}, " 
                f"phase={self.current_search_phase}, "
                f"angle={self.current_rotation_angle}°, "
                f"cumulative={self.cumulative_rotation:.1f}°, "
                f"direction={'CW' if self.search_direction < 0 else 'CCW'}"
            )
            self.get_logger().info(
                f"Search timing: duration={self.total_search_time:.1f}s, "
                f"timeout={self.max_search_time}s"
            )
            
        elif self.current_state == RobotState.LOST_BALL:
            # Show recovery information
            time_to_recovery = max(0, 30.0 - (time.time() - self.last_recovery_time))
            self.get_logger().info(
                f"Lost ball status: recovery_attempts={self.recovery_attempts}, "
                f"next_attempt_in={time_to_recovery:.1f}s"
            )
            
        elif self.current_state == RobotState.STOPPED:
            # Show stationary ball information
            if self.stationary_start_time is not None:
                stationary_duration = time.time() - self.stationary_start_time
                self.get_logger().info(f"Ball stationary for {stationary_duration:.1f}s")
        
        self.get_logger().info("============================")


def main(args=None):
    """Main function to initialize and run the State Manager node."""
    rclpy.init(args=args)
    node = StateManagerNode()
    
    # Welcome message
    print("=================================================")
    print("Tennis Ball Tracking - State Manager Node")
    print("=================================================")
    print("This node manages the robot's operational states:")
    print("- INITIALIZING: Startup, waiting for ball detection")
    print("- TRACKING: Following the tennis ball")
    print("- SEARCHING: Looking for a lost ball")
    print("- LOST_BALL: Stationary waiting for ball to reappear")
    print("- STOPPED: Ball is close and stationary")
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
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("State Manager shutdown requested (Ctrl+C)")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {str(e)}")
    finally:
        # Make sure robot stops before shutting down
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()