#!/usr/bin/env python3

"""
Basketball Chaser - State Management Node

This node determines the robot's behavior based on tracking reliability,
transitioning between states like initialization, tracking, lost ball, 
and stopped states.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TwistStamped, Twist
from std_msgs.msg import String, Bool, Float32
import numpy as np
import time
import math
import json
from collections import deque  # For tracking history

# Add a custom JSON encoder for NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder that can handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyJSONEncoder, self).default(obj)

class RobotState:
    """Enumeration of robot operational states"""
    INITIALIZING = "initializing"  # Startup state, waiting for first reliable detection
    TRACKING = "tracking"          # Actively tracking the ball with reliable detections
    LOST_BALL = "lost_ball"        # Ball not found after extensive searching
    STOPPED = "stopped"            # Stationary state when ball is close and stationary

class BallChaseStateManager(Node):
    """
    State management node for the basketball chasing robot.
    
    Determines appropriate robot behavior based on tracking reliability
    and handles transitions between different operational states.
    """
    
    def __init__(self):
        """Initialize the state manager node."""
        super().__init__('ball_chase_state_manager')
        
        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                # Timing thresholds
                ('lost_ball_timeout', 1.0),          # Seconds without detection to consider ball lost
                ('max_search_time', 30.0),           # Seconds to search before giving up
                ('stationary_time_threshold', 1.5),  # Time ball needs to be stationary before stopping
                
                # Search pattern parameters
                ('search_rotation_speed', 0.5),      # Rotation speed during search (rad/s)
                ('max_rotation_time', 15.0),         # Maximum time to rotate before giving up
                
                # Detection thresholds
                ('min_tracking_detections', 3),      # Consecutive detections to confirm tracking
                ('proximity_threshold', 0.5),        # Distance to consider ball "close" (meters)
                ('stationary_threshold', 0.05),      # Max movement to consider ball stationary
            ]
        )
        
        # Load parameters
        self._load_parameters()
        
        # Initialize state variables
        self._init_state_variables()
        
        # Set up subscriptions
        self._setup_subscriptions()
        
        # Set up publishers
        self._setup_publishers()
        
        # Timer for state management
        self.timer = self.create_timer(0.1, self.state_manager_callback)  # 10Hz state management
        self.diagnostic_timer = self.create_timer(1.0, self.publish_diagnostics)  # 1Hz diagnostics
        
        self.get_logger().info("Basketball Chaser State Manager initialized in INITIALIZING state")
        self.publish_state()
    
    def _load_parameters(self):
        """Load parameters from ROS parameter server."""
        self.lost_ball_timeout = self.get_parameter('lost_ball_timeout').value
        self.max_search_time = self.get_parameter('max_search_time').value
        self.stationary_time_threshold = self.get_parameter('stationary_time_threshold').value
        
        self.search_rotation_speed = self.get_parameter('search_rotation_speed').value
        self.max_rotation_time = self.get_parameter('max_rotation_time').value
        
        self.min_tracking_detections = self.get_parameter('min_tracking_detections').value
        self.proximity_threshold = self.get_parameter('proximity_threshold').value
        self.stationary_threshold = self.get_parameter('stationary_threshold').value
    
    def _init_state_variables(self):
        """Initialize state tracking variables."""
        # Current state
        self.current_state = RobotState.INITIALIZING
        self.state_start_time = time.time()
        
        # Ball tracking
        self.last_position = None
        self.last_detection_time = None
        self.consecutive_detections = 0
        self.tracking_reliable = False
        self.position_uncertainty = float('inf')
        
        # Position history for stationary detection
        self.position_history = []  # List of (position, timestamp) tuples
        self.position_history_max_len = 10
        
        # Ball proximity and movement detection
        self.ball_distance = float('inf')
        self.is_ball_close = False
        self.is_ball_stationary = False
        self.stationary_start_time = None
        
        # Search variables
        self.search_direction = 1  # 1 for counter-clockwise, -1 for clockwise
        self.total_search_time = 0.0
        self.search_rotation_start_time = None
        self.search_angle_accumulated = 0.0  # Track rotation during search
    
    def _setup_subscriptions(self):
        """Set up subscriptions to fusion node topics."""
        # Ball position
        self.position_sub = self.create_subscription(
            PointStamped,
            '/basketball/fused/position',
            self.position_callback,
            10
        )
        
        # Tracking status
        self.tracking_status_sub = self.create_subscription(
            Bool,
            '/basketball/fused/tracking_status',
            self.tracking_status_callback,
            10
        )
        
        # Position uncertainty
        self.uncertainty_sub = self.create_subscription(
            Float32,
            '/basketball/fused/position_uncertainty',
            self.uncertainty_callback,
            10
        )
        
        # 1. Motion State Integration - Subscribe to motion state
        self.motion_state_sub = self.create_subscription(
            String,
            '/basketball/fused/motion_state',
            self.motion_state_callback,
            10
        )
        
        # 2. Confidence-Based Decision Making - Subscribe to tracking confidence
        self.confidence_sub = self.create_subscription(
            Float32,
            '/basketball/fused/tracking_confidence',
            self.tracking_confidence_callback,
            10
        )
        
        # 3. Gap Tolerance Enhancement - Subscribe to gap information
        self.gap_detection_sub = self.create_subscription(
            Bool,
            '/basketball/fused/sensor_gap',
            self.sensor_gap_callback,
            10
        )
    
    def _setup_publishers(self):
        """Set up publishers for commands and state information."""
        # Robot motion commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Current robot state
        self.state_publisher = self.create_publisher(
            String,
            '/robot/state',
            10
        )
    
    def tracking_status_callback(self, msg):
        """
        Process tracking reliability flag from the fusion node.
        
        Args:
            msg (Bool): Whether tracking is reliable
        """
        # Initialize message counter if not already created
        if not hasattr(self, 'tracking_status_msg_count'):
            self.tracking_status_msg_count = 0
        
        # Increment message counter
        self.tracking_status_msg_count += 1
        
        # Store tracking status
        self.tracking_reliable = msg.data
        
        # Log every 10th message with more detail
        if self.tracking_status_msg_count % 10 == 0:
            self.get_logger().info(f"Fusion node tracking status message #{self.tracking_status_msg_count}: reliable={self.tracking_reliable}")
        else:
            self.get_logger().debug(f"Tracking status update: {self.tracking_reliable}")
    
    def uncertainty_callback(self, msg):
        """
        Process position uncertainty from the fusion node.
        
        Args:
            msg (Float32): Position uncertainty in meters
        """
        # Initialize message counter if not already created
        if not hasattr(self, 'uncertainty_msg_count'):
            self.uncertainty_msg_count = 0
        
        # Increment message counter
        self.uncertainty_msg_count += 1
        
        # Store uncertainty value
        self.position_uncertainty = msg.data
        
        # Log every 10th message with more detail
        if self.uncertainty_msg_count % 10 == 0:
            self.get_logger().info(f"Fusion node uncertainty message #{self.uncertainty_msg_count}: {self.position_uncertainty:.3f}m")
        else:
            self.get_logger().debug(f"Position uncertainty update: {self.position_uncertainty:.3f}m")
    
    def position_callback(self, msg):
        """
        Process ball position updates from the fusion node.
        
        Args:
            msg (PointStamped): 3D position of the ball
        """
        current_time = time.time()
        
        # Extract position - only use 3D sensor data, ignore 2D
        position = np.array([msg.point.x, msg.point.y, msg.point.z])
        
        # Update detection time
        self.last_detection_time = current_time
        
        # Update position history for stationary detection
        self.position_history.append((position, current_time))
        if len(self.position_history) > self.position_history_max_len:
            self.position_history.pop(0)
        
        # Calculate position change if we have a previous position
        if self.last_position is not None:
            position_change = np.linalg.norm(position - self.last_position)
            
            # Count as valid detection if change is reasonable
            if position_change < 1.0:  # Max 1 meter change between updates
                self.consecutive_detections += 1
            else:
                # Large jump might be a false positive
                self.consecutive_detections = max(0, self.consecutive_detections - 1)
                self.get_logger().info(f"Large position jump detected: {position_change:.2f}m")
        else:
            # First detection
            self.consecutive_detections = 1
        
        # Calculate distance to ball correctly - use full 3D position 
        # (ignoring 2D sensor data which adds noise)
        self.ball_distance = np.linalg.norm(position[:2])  # Only consider XY plane distance
        
        # Update close ball detection
        self.is_ball_close = self.ball_distance <= self.proximity_threshold
        
        # Update stationary ball detection
        self.update_ball_stationary_status()
        
        # Store current position for next comparison
        self.last_position = position
        
        # Handle state transitions based on position updates
        self.handle_position_based_transitions(current_time)
    
    def update_ball_stationary_status(self):
        """Check if the ball hasn't moved significantly over recent history."""
        if len(self.position_history) < 3:  # Need multiple samples to determine
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
        
        self.get_logger().debug(
            f"Ball stationary check: max_movement={max_movement:.3f}m, "
            f"is_stationary={self.is_ball_stationary}"
        )
    
    def handle_position_based_transitions(self, current_time):
        """
        Handle state transitions based on new position information.
        
        Args:
            current_time (float): Current timestamp
        """
        # Transition from INITIALIZING to TRACKING if reliable detections
        if self.current_state == RobotState.INITIALIZING:
            # MODIFIED: Relax reliability requirement if ball is stationary
            if (self.consecutive_detections >= self.min_tracking_detections and 
                (self.tracking_reliable or 
                 (hasattr(self, 'is_ball_stationary') and self.is_ball_stationary) or
                 (hasattr(self, 'motion_state') and 
                  self.motion_state in ["stationary", "long_stationary"]))):
                self.get_logger().info("Transitioning to TRACKING: ball is detected with sufficient confidence or is stationary")
                self.transition_to_state(RobotState.TRACKING)
        
        # Transition from LOST_BALL to TRACKING if ball reappears
        elif self.current_state == RobotState.LOST_BALL:
            # MODIFIED: Same relaxed condition for LOST_BALL to TRACKING
            if (self.consecutive_detections >= self.min_tracking_detections and 
                (self.tracking_reliable or 
                 (hasattr(self, 'is_ball_stationary') and self.is_ball_stationary) or
                 (hasattr(self, 'motion_state') and 
                  self.motion_state in ["stationary", "long_stationary"]))):
                self.get_logger().info("Transitioning from LOST_BALL to TRACKING: ball reappeared with reliability or is stationary")
                self.transition_to_state(RobotState.TRACKING)
        
        # Handle transition to STOPPED if ball is close and stationary
        elif self.current_state == RobotState.TRACKING:
            if self.is_ball_close and self.is_ball_stationary:
                if self.stationary_start_time is None:
                    # First time detecting stationary ball
                    self.stationary_start_time = current_time
                elif current_time - self.stationary_start_time >= self.stationary_time_threshold:
                    # Ball has been stationary and close for required time
                    self.transition_to_state(RobotState.STOPPED)
                    self.get_logger().info(f"Ball is close ({self.ball_distance:.2f}m) and stationary - stopping")
            else:
                # Reset stationary timer if conditions aren't met
                self.stationary_start_time = None
        
        # Handle transition back from STOPPED if ball moves or is no longer close
        elif self.current_state == RobotState.STOPPED:
            if not self.is_ball_close or not self.is_ball_stationary:
                reason = "moved away" if not self.is_ball_close else "started moving"
                self.get_logger().info(f"Ball has {reason} - resuming tracking")
                self.transition_to_state(RobotState.TRACKING)
    
    def state_manager_callback(self):
        """
        Regular timer callback to manage state transitions and actions.
        
        Called at 10Hz to:
        1. Monitor tracking quality and perform state transitions
        2. Execute appropriate actions for the current state
        3. Handle timeouts and recovery actions
        """
        current_time = time.time()
        
        if self.current_state == RobotState.TRACKING:
            # Check if ball is lost based on tracking reliability and timeout
            time_since_detection = (current_time - self.last_detection_time 
                                   if self.last_detection_time is not None else float('inf'))
            
            # Don't transition to LOST_BALL if ball is stationary
            ignore_reliability = (hasattr(self, 'is_ball_stationary') and self.is_ball_stationary) or \
                               (hasattr(self, 'motion_state') and 
                                self.motion_state in ["stationary", "long_stationary"] and
                                time_since_detection < self.lost_ball_timeout * 1.5)  # Give more time for stationary balls
            
            if (not self.tracking_reliable and not ignore_reliability) or time_since_detection > self.lost_ball_timeout:
                reason = "unreliable tracking" if not self.tracking_reliable else "detection timeout"
                self.get_logger().info(f"Ball lost! Reason: {reason}")
                self.transition_to_state(RobotState.LOST_BALL)
        
        elif self.current_state == RobotState.INITIALIZING:
            # Check if we should timeout initialization
            time_in_state = current_time - self.state_start_time
            if time_in_state > 5.0:  # 5 seconds to initialize
                self.transition_to_state(RobotState.LOST_BALL)
        
        elif self.current_state == RobotState.LOST_BALL:
            # Just stay in LOST_BALL state - we don't search for the ball
            # The transition to TRACKING happens in handle_position_based_transitions
            # when the ball is detected again
            self.stop_robot()
    
    def execute_search_rotation(self):
        """
        This method is deprecated - we no longer search for the ball by moving.
        Keeping the method as a stub for compatibility.
        """
        # Just publish zero velocity - no searching
        twist = Twist()
        self.cmd_vel_publisher.publish(twist)
    
    def transition_to_state(self, new_state):
        """
        Handle state transitions with proper cleanup and initialization.
        
        Args:
            new_state (str): The state to transition to
        """
        if new_state == self.current_state:
            return
        
        # Log the transition
        time_in_prev_state = time.time() - self.state_start_time
        self.get_logger().info(
            f"State transition: {self.current_state} → {new_state} "
            f"(after {time_in_prev_state:.1f}s)"
        )
        
        # Handle exit actions for current state
        if self.current_state == RobotState.LOST_BALL:
            # Record total time ball was lost
            self.total_lost_time = getattr(self, 'total_lost_time', 0) + time_in_prev_state
        
        # Update state and reset state timer
        prev_state = self.current_state
        self.current_state = new_state
        self.state_start_time = time.time()
        
        # Handle entry actions for the new state
        if new_state == RobotState.TRACKING:
            self.get_logger().info("Ball tracking initiated")
        
        elif new_state == RobotState.LOST_BALL:
            self.get_logger().info("Ball lost. Entering wait mode.")
            self.stop_robot()
        
        elif new_state == RobotState.STOPPED:
            self.get_logger().info("Ball is close and stationary - stopping robot")
            self.stop_robot()
        
        # Publish the new state
        self.publish_state()
    
    def stop_robot(self):
        """Send command to stop all robot motion immediately."""
        twist = Twist()  # All fields initialize to 0
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().debug("Robot motion stopped")
    
    def publish_state(self):
        """Publish current robot state for other nodes to consume."""
        msg = String()
        msg.data = self.current_state
        self.state_publisher.publish(msg)
    
    def publish_diagnostics(self):
        """Publish diagnostic information about the state manager."""
        if self.last_detection_time is not None:
            time_since_detection = time.time() - self.last_detection_time
        else:
            time_since_detection = float('inf')
            
        state_duration = time.time() - self.state_start_time
        
        # Create a structured diagnostic log
        diagnostic_info = {
            "state": {
                "current": self.current_state,
                "duration": f"{state_duration:.1f}s"
            },
            "tracking": {
                "reliable": self.tracking_reliable,
                "consecutive_detections": self.consecutive_detections,
                "uncertainty": f"{self.position_uncertainty:.3f}m",
                "time_since_detection": f"{time_since_detection:.2f}s"
            },
            "ball": {
                "distance": f"{self.ball_distance:.2f}m",
                "is_close": self.is_ball_close,
                "is_stationary": self.is_ball_stationary
            }
        }
        
        # Log diagnostic information
        self.get_logger().info(f"State Manager Diagnostics: {json.dumps(diagnostic_info, cls=NumpyJSONEncoder)}")
    
    # 1. Motion State Integration
    def motion_state_callback(self, msg):
        """
        Process motion state updates from the fusion node.
        
        Args:
            msg (String): Current motion state (stationary, long_stationary, small_movement, medium_fast)
        """
        # Initialize message counter if not already created
        if not hasattr(self, 'motion_state_msg_count'):
            self.motion_state_msg_count = 0
        
        # Increment message counter
        self.motion_state_msg_count += 1
        
        # Store motion state
        self.motion_state = msg.data
        
        # Log every 10th message with detailed information
        if self.motion_state_msg_count % 10 == 0:
            self.get_logger().info(f"Fusion node motion state message #{self.motion_state_msg_count}: {self.motion_state}")
        else:
            self.get_logger().debug(f"Received motion state update: {self.motion_state}")
        
        # When ball is stationary or long-stationary, we can be more lenient with tracking reliability
        if self.motion_state in ["stationary", "long_stationary"]:
            # During stationary states, consecutive detections are more meaningful than tracking_reliable flag
            if self.consecutive_detections >= self.min_tracking_detections:
                # Force a position-based transition check
                self.handle_position_based_transitions(time.time())
                
        # Update adaptive thresholds based on motion state
        if hasattr(self, 'update_adaptive_thresholds'):
            self.update_adaptive_thresholds()
    
    # 2. Confidence-Based Decision Making
    def tracking_confidence_callback(self, msg):
        """
        Process tracking confidence values from fusion node.
        
        Args:
            msg (Float32): Confidence level (0.0-1.0) of current tracking
        """
        self.tracking_confidence = msg.data
        
        # Update confidence history for trend analysis
        if not hasattr(self, 'confidence_history'):
            self.confidence_history = deque(maxlen=10)
        self.confidence_history.append(self.tracking_confidence)
        
        # Use confidence for more nuanced state transitions
        if self.current_state == RobotState.TRACKING:
            # Only transition to SEARCHING if confidence is consistently low
            if self.tracking_confidence < 0.3:
                # Check if confidence has been consistently low
                if len(self.confidence_history) >= 3 and all(c < 0.4 for c in list(self.confidence_history)[-3:]):
                    self.get_logger().info(f"Low tracking confidence ({self.tracking_confidence:.2f}) - transitioning to SEARCHING")
                    self.transition_to_state(RobotState.SEARCHING)
            
        elif self.current_state == RobotState.SEARCHING:
            # Use confidence to return to TRACKING sooner
            if self.tracking_confidence > 0.7:
                self.get_logger().info(f"High confidence detection during search ({self.tracking_confidence:.2f}) - resuming TRACKING")
                self.transition_to_state(RobotState.TRACKING)
    
    # 3. Gap Tolerance Enhancement
    def sensor_gap_callback(self, msg):
        """
        Process sensor gap information from fusion node.
        
        Args:
            msg (Bool): Whether the system is currently in a sensor gap
        """
        self.in_sensor_gap = msg.data
        
        if self.in_sensor_gap:
            # Start tracking gap duration if not already tracking
            if not hasattr(self, 'gap_start_time') or self.gap_start_time is None:
                self.gap_start_time = time.time()
                self.get_logger().info("Sensor gap detected - entering gap tolerance mode")
        else:
            # Reset gap tracking
            self.gap_start_time = None
        
        # Add special handling for sensor gaps during TRACKING state
        if self.current_state == RobotState.TRACKING and self.in_sensor_gap:
            # Keep tracking during short gaps if confidence was high
            # and we've had reliable tracking recently
            if hasattr(self, 'tracking_confidence') and self.tracking_confidence > 0.7:
                # Calculate gap duration
                gap_duration = time.time() - self.gap_start_time
                
                # Adaptive gap tolerance based on motion state
                tolerance_time = 1.0  # Default tolerance
                
                if hasattr(self, 'motion_state'):
                    if self.motion_state in ["stationary", "long_stationary"]:
                        # Much longer tolerance for stationary balls
                        tolerance_time = 3.0
                    elif self.motion_state == "small_movement":
                        # Medium tolerance for slow movement
                        tolerance_time = 1.5
                
                # Stay in TRACKING state during short gaps with adaptive tolerance
                if gap_duration < tolerance_time:
                    # Temporarily override the timeout logic
                    self.last_detection_time = time.time() - (self.lost_ball_timeout * 0.5)
                    self.get_logger().debug(f"Gap tolerance active: {gap_duration:.1f}s/{tolerance_time:.1f}s")
    
    # 4. Adaptive Thresholds
    def update_adaptive_thresholds(self):
        """Update thresholds based on current conditions."""
        # Create adaptive parameters if not existing
        if not hasattr(self, 'adaptive_thresholds'):
            self.adaptive_thresholds = {
                'lost_ball_timeout': self.lost_ball_timeout,
                'stationary_threshold': self.stationary_threshold,
                'proximity_threshold': self.proximity_threshold
            }
            self.base_thresholds = self.adaptive_thresholds.copy()
        
        # Update based on motion state
        if hasattr(self, 'motion_state'):
            if self.motion_state == "stationary":
                # Higher movement threshold for stationary detection
                self.adaptive_thresholds['stationary_threshold'] = self.base_thresholds['stationary_threshold'] * 1.5
                # Longer timeout for stationary balls
                self.adaptive_thresholds['lost_ball_timeout'] = self.base_thresholds['lost_ball_timeout'] * 1.5
            elif self.motion_state == "long_stationary":
                # Even higher stationary threshold and timeout for long-term stationary
                self.adaptive_thresholds['stationary_threshold'] = self.base_thresholds['stationary_threshold'] * 2.0
                self.adaptive_thresholds['lost_ball_timeout'] = self.base_thresholds['lost_ball_timeout'] * 2.0
            elif self.motion_state == "medium_fast":
                # Lower stationary threshold for fast motion
                self.adaptive_thresholds['stationary_threshold'] = self.base_thresholds['stationary_threshold'] * 0.8
                # Shorter lost ball timeout for fast moving balls
                self.adaptive_thresholds['lost_ball_timeout'] = self.base_thresholds['lost_ball_timeout'] * 0.7
            else:
                # Default thresholds
                self.adaptive_thresholds = self.base_thresholds.copy()
        
        # Update based on tracking confidence
        if hasattr(self, 'tracking_confidence'):
            # Adjust lost ball timeout based on confidence
            confidence_factor = 0.5 + (self.tracking_confidence * 0.5)  # 0.5-1.0 range
            self.adaptive_thresholds['lost_ball_timeout'] *= confidence_factor
        
        # Apply the adapted thresholds
        self.lost_ball_timeout = self.adaptive_thresholds.get('lost_ball_timeout', self.lost_ball_timeout)
        self.stationary_threshold = self.adaptive_thresholds.get('stationary_threshold', self.stationary_threshold)
        self.proximity_threshold = self.adaptive_thresholds.get('proximity_threshold', self.proximity_threshold)
    
    # 5. State Protection Mechanisms
    def apply_state_protection(self, proposed_state):
        """
        Apply protection against rapid state oscillations.
        
        Args:
            proposed_state (str): The proposed new state
            
        Returns:
            str: The actual state to transition to (may be different from proposed)
        """
        # Initialize state transition history if not present
        if not hasattr(self, 'state_transition_history'):
            self.state_transition_history = deque(maxlen=10)
            self.transition_times = {}
            self.min_time_in_state = {
                RobotState.TRACKING: 1.0,    # At least 1 second in TRACKING
                RobotState.SEARCHING: 2.0,   # At least 2 seconds in SEARCHING
                RobotState.STOPPED: 0.5,     # At least 0.5 second in STOPPED
                RobotState.LOST_BALL: 5.0    # At least 5 seconds in LOST_BALL
            }
            self.hysteresis_counts = {}  # Count of blocked transitions
        
        current_time = time.time()
        time_in_state = current_time - self.state_start_time
        
        # Always allow transitions from INITIALIZING (no protection needed)
        if self.current_state == RobotState.INITIALIZING:
            return proposed_state
            
        # Check minimum time in current state
        min_time = self.min_time_in_state.get(self.current_state, 0.0)
        
        # Block transitions if not enough time in current state
        if time_in_state < min_time:
            # Count blocked transitions
            transition_key = f"{self.current_state}->{proposed_state}"
            self.hysteresis_counts[transition_key] = self.hysteresis_counts.get(transition_key, 0) + 1
            
            if proposed_state != self.current_state:
                self.get_logger().debug(
                    f"Blocked transition to {proposed_state}: too soon "
                    f"({time_in_state:.1f}s < {min_time:.1f}s required)"
                )
            
            return self.current_state
            
        # Check for oscillating transitions (ping-pong between states)
        # Example: TRACKING -> SEARCHING -> TRACKING happening too frequently
        if len(self.state_transition_history) >= 4:
            recent_states = list(self.state_transition_history)
            
            # If we detect a pattern like A->B->A->B
            if (recent_states[-1] == recent_states[-3] and 
                recent_states[-2] == recent_states[-4] and
                proposed_state == recent_states[-2]):
                
                # Check if these transitions happened in quick succession
                if current_time - self.transition_times.get(recent_states[-4], 0) < 5.0:
                    # This is an oscillation - apply hysteresis by remaining in current state
                    self.get_logger().info(
                        f"Detected state oscillation pattern. "
                        f"Remaining in {self.current_state} for stability."
                    )
                    return self.current_state
        
        # Special case: Protect STOPPED state during gaps for stationary balls
        if (self.current_state == RobotState.STOPPED and
            proposed_state == RobotState.TRACKING and
            hasattr(self, 'in_sensor_gap') and self.in_sensor_gap and
            hasattr(self, 'motion_state') and 
            self.motion_state in ["stationary", "long_stationary"]):
            
            self.get_logger().info(
                f"Protecting STOPPED state during sensor gap for {self.motion_state} ball"
            )
            return self.current_state
            
        # Allow the transition
        if proposed_state != self.current_state:
            # Record transition for pattern detection
            self.state_transition_history.append(self.current_state)
            self.transition_times[self.current_state] = current_time
            
        return proposed_state
        
    # 6. Enhanced Motion State Awareness
    def update_motion_based_behavior(self):
        """Update robot behavior based on ball motion state."""
        if not hasattr(self, 'motion_state'):
            return
            
        # Initialize motion state confidence if not present
        if not hasattr(self, 'motion_state_confidence'):
            self.motion_state_confidence = {
                "stationary": 0.5,
                "long_stationary": 0.0,
                "small_movement": 0.0,
                "medium_fast": 0.0
            }
            
        # Update behavior based on current state and motion state
        if self.current_state == RobotState.TRACKING:
            cmd = Twist()
            
            # Adjust approach behavior based on motion state
            if self.motion_state == "stationary" or self.motion_state == "long_stationary":
                # Approach stationary ball more carefully with slower speed
                # (implementation would depend on your tracking algorithm)
                pass
                
            elif self.motion_state == "medium_fast":
                # Be more aggressive in following fast-moving balls
                # (implementation would depend on your tracking algorithm)
                pass
                
    # 7. Predictive State Management
    def predict_state(self):
        """
        Predict future state based on current trends.
        This helps maintain state during brief sensor gaps.
        """
        # Initialize prediction data if not present
        if not hasattr(self, 'prediction_data'):
            self.prediction_data = {
                'predicted_position': None,
                'predicted_velocity': None,
                'last_prediction_time': 0.0,
                'lost_position': None,
                'lost_time': 0.0
            }
            
        current_time = time.time()
        
        # Check for transition from TRACKING to SEARCHING (ball just lost)
        # This is where we want to start prediction
        if (self.current_state == RobotState.SEARCHING and 
            hasattr(self, 'prev_state') and 
            self.prev_state == RobotState.TRACKING and
            self.last_position is not None):
            
            # Start prediction when we first lose the ball
            if self.prediction_data['lost_position'] is None:
                # Store last known position and time
                self.prediction_data['lost_position'] = self.last_position
                self.prediction_data['lost_time'] = current_time
                
                # Calculate a simple velocity from recent history if available
                if len(self.position_history) >= 2:
                    pos1, time1 = self.position_history[-1]
                    pos2, time2 = self.position_history[-2]
                    
                    # Only calculate if times are different
                    if time2 < time1:
                        dt = time1 - time2
                        # Simple velocity calculation based on last two positions
                        velocity = (pos1 - pos2) / dt
                        self.prediction_data['predicted_velocity'] = velocity
                        
                        # Log the predicted velocity
                        speed = np.linalg.norm(velocity)
                        self.get_logger().info(
                            f"Ball lost. Starting prediction with velocity of {speed:.2f} m/s"
                        )
                        
                        # Make initial position prediction
                        self.update_position_prediction(current_time)
        
        # Update prediction periodically while searching
        elif (self.current_state == RobotState.SEARCHING and 
              self.prediction_data['lost_position'] is not None and
              self.prediction_data['predicted_velocity'] is not None):
            
            # Update prediction every 0.5 seconds
            if current_time - self.prediction_data['last_prediction_time'] > 0.5:
                self.update_position_prediction(current_time)
                
    def update_position_prediction(self, current_time):
        """
        Update predicted position based on elapsed time and predicted velocity.
        
        Args:
            current_time (float): Current timestamp
        """
        # Calculate time since ball was lost
        dt = current_time - self.prediction_data['lost_time']
        
        # Don't predict beyond a reasonable time window (5 seconds)
        if dt > 5.0:
            self.get_logger().info("Prediction timeout - ball lost too long ago")
            # Clear prediction
            self.prediction_data['predicted_position'] = None
            return
            
        # Simple linear prediction with deceleration
        # v = v0 - kt where k is a deceleration factor due to friction
        friction_decel = 0.5  # Deceleration in m/s²
        
        # Get original velocity magnitude
        v0 = np.linalg.norm(self.prediction_data['predicted_velocity'])
        
        # Only apply deceleration until velocity would reach zero
        decel_time = v0 / friction_decel if friction_decel > 0 else float('inf')
        
        if dt < decel_time:
            # Ball is still moving, apply deceleration
            velocity_factor = 1.0 - (dt / decel_time)
            current_velocity = self.prediction_data['predicted_velocity'] * velocity_factor
            
            # Calculate displacement from initial lost position
            displacement = self.prediction_data['predicted_velocity'] * dt - 0.5 * friction_decel * dt**2 * self.prediction_data['predicted_velocity'] / v0
            
            # Update predicted position
            self.prediction_data['predicted_position'] = self.prediction_data['lost_position'] + displacement
        else:
            # Ball has stopped due to friction
            # Use the final resting position
            travel_distance = (v0**2) / (2 * friction_decel)
            displacement = travel_distance * self.prediction_data['predicted_velocity'] / v0
            self.prediction_data['predicted_position'] = self.prediction_data['lost_position'] + displacement
            
        # Update last prediction time
        self.prediction_data['last_prediction_time'] = current_time
        
        # Bias search direction toward predicted position
        if hasattr(self, 'search_direction'):
            # Get angle to predicted position (in robot's frame)
            predicted_pos = self.prediction_data['predicted_position']
            angle = math.atan2(predicted_pos[1], predicted_pos[0])
            
            # Set search direction based on angle to predicted position
            self.search_direction = 1 if angle > 0 else -1
            
            self.get_logger().debug(
                f"Ball predicted at ({predicted_pos[0]:.2f}, {predicted_pos[1]:.2f}). "
                f"Search direction: {'+CCW' if self.search_direction > 0 else '-CW'}"
            )
    
    # 8. Enhanced Diagnostic Integration
    def publish_enhanced_diagnostics(self):
        """Publish enhanced diagnostic information including fusion state integration."""
        # Basic diagnostics from original method
        if self.last_detection_time is not None:
            time_since_detection = time.time() - self.last_detection_time
        else:
            time_since_detection = float('inf')
            
        state_duration = time.time() - self.state_start_time
        
        # Create a structured diagnostic log with enhanced information
        diagnostic_info = {
            "state": {
                "current": self.current_state,
                "duration": f"{state_duration:.1f}s"
            },
            "tracking": {
                "reliable": self.tracking_reliable,
                "consecutive_detections": self.consecutive_detections,
                "uncertainty": f"{self.position_uncertainty:.3f}m",
                "time_since_detection": f"{time_since_detection:.2f}s"
            },
            "ball": {
                "distance": f"{self.ball_distance:.2f}m",
                "is_close": self.is_ball_close,
                "is_stationary": self.is_ball_stationary
            }
        }
        
        # Add motion state info from fusion node
        if hasattr(self, 'motion_state'):
            diagnostic_info["motion_state"] = {
                "current": self.motion_state,
                "previous": getattr(self, 'last_motion_state', "unknown"),
            }
            
        # Add confidence-based metrics
        if hasattr(self, 'tracking_confidence'):
            diagnostic_info["confidence"] = {
                "tracking": f"{self.tracking_confidence:.2f}"
            }
            
        # Add gap tolerance diagnostics
        if hasattr(self, 'in_sensor_gap') and self.in_sensor_gap:
            gap_duration = 0.0
            if hasattr(self, 'gap_start_time') and self.gap_start_time is not None:
                gap_duration = time.time() - self.gap_start_time
                
            diagnostic_info["sensor_gap"] = {
                "active": True,
                "duration": f"{gap_duration:.2f}s",
            }
            
        # Add state protection info
        if hasattr(self, 'hysteresis_counts') and self.hysteresis_counts:
            diagnostic_info["state_protection"] = {
                "blocked_transitions": sum(self.hysteresis_counts.values()),
                "stability_score": self.calculate_stability_score()
            }
            
        # Add prediction info if active
        if (hasattr(self, 'prediction_data') and 
            self.prediction_data.get('predicted_position') is not None):
            
            pred_pos = self.prediction_data['predicted_position']
            diagnostic_info["prediction"] = {
                "active": True,
                "predicted_position": f"({pred_pos[0]:.2f}, {pred_pos[1]:.2f})",
                "prediction_age": f"{time.time() - self.prediction_data.get('last_prediction_time', 0):.1f}s"
            }
        
        # Add adaptive threshold info
        if hasattr(self, 'adaptive_thresholds'):
            diagnostic_info["adaptive_thresholds"] = {
                "lost_ball_timeout": f"{self.lost_ball_timeout:.2f}s",
                "stationary_threshold": f"{self.stationary_threshold:.3f}m"
            }
        
        # Log enhanced diagnostic information
        self.get_logger().info(f"Enhanced State Manager Diagnostics: {json.dumps(diagnostic_info, cls=NumpyJSONEncoder)}")
        
    def calculate_stability_score(self):
        """Calculate a stability score based on state transitions and protection metrics."""
        if not hasattr(self, 'hysteresis_counts') or not hasattr(self, 'state_transition_history'):
            return 1.0  # Default perfect score
            
        # More blocked transitions means less stability
        blocked_transitions = sum(self.hysteresis_counts.values())
        
        # More transitions in the recent history means less stability
        recent_transition_count = len(set(self.state_transition_history))
        
        # Calculate stability score (higher is better)
        # 1.0 is perfect stability, 0.0 is completely unstable
        stability = max(0.0, 1.0 - (0.1 * blocked_transitions) - (0.2 * recent_transition_count))
        
        return stability


def main(args=None):
    """Main function to initialize and run the state manager node."""
    rclpy.init(args=args)
    node = BallChaseStateManager()
    
    # Welcome message
    print("=================================================")
    print("Basketball Chaser - State Manager Node")
    print("=================================================")
    print("This node manages the robot's operational states:")
    print("- INITIALIZING: Startup, waiting for ball detection")
    print("- TRACKING: Following the tennis ball")
    print("- LOST_BALL: Ball not found, waiting for it to reappear")
    print("- STOPPED: Ball is close and stationary")
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