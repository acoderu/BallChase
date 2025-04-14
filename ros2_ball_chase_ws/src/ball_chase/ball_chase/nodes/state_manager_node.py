def _init_state_variables(self):
        """Initialize state variables for the state machine."""
        # Current state
        self.state = "initializing"
        
        # Timing variables
        self.state_start_time = time.time()
        self.last_detection_time = 0.0  # Initialize to 0 to ensure lost ball until first detection
        self.last_state_log_time = 0.0  # For throttling state logs
        
        # Ball detection variables
        self.ball_detected = False
        
        # State machine parameters
        self.init_delay = 2.0  # Time to stay in initializing state (seconds)
        self.detection_timeout = 0.5  # Time before considering ball lost (seconds)
        self.lost_timeout = 3.0  # Time to wait in lost state before searching (seconds)
        
        # Logging for state changes
        self.get_logger().info(f"Initial state: {self.state}")
        self.get_logger().info(f"Will transition from initializing after {self.init_delay}s")
        self.get_logger().info(f"Detection timeout: {self.detection_timeout}s")
        self.get_logger().info(f"Lost ball timeout: {self.lost_timeout}s")

def _check_state_transitions(self):
        """
        Check and perform state transitions based on current conditions.
        
        This is the core state machine update function that determines when to
        transition between different states based on sensor data and timing.
        """
        current_time = time.time()
        
        # Log current state with more debugging info every 3 seconds
        if current_time - self.last_state_log_time > 3.0:
            if self.state == "initializing":
                self.get_logger().info(
                    f"Current state: {self.state}, "
                    f"time in state: {current_time - self.state_start_time:.1f}s, "
                    f"ball_detected: {self.ball_detected}, "
                    f"ready_to_transition: {(current_time - self.state_start_time) > self.init_delay}"
                )
            else:
                self.get_logger().info(
                    f"Current state: {self.state}, "
                    f"time in state: {current_time - self.state_start_time:.1f}s, "
                    f"ball_detected: {self.ball_detected}, "
                    f"last_detection: {current_time - self.last_detection_time:.1f}s ago"
                )
            self.last_state_log_time = current_time
            
        # State transitions - implemented as a state machine
        if self.state == "initializing":
            # Transition from initializing after a short delay to allow sensors to stabilize
            if (current_time - self.state_start_time) > self.init_delay:
                if self.ball_detected:
                    self._change_state("tracking")
                else:
                    self._change_state("searching")
                    
        elif self.state == "searching":
            # If ball is found while searching, switch to tracking
            if self.ball_detected:
                self._change_state("tracking")
                
        elif self.state == "tracking":
            # If ball is lost while tracking, switch to lost_ball state
            if not self.ball_detected or (current_time - self.last_detection_time) > self.detection_timeout:
                self._change_state("lost_ball")
                
        elif self.state == "lost_ball":
            # If ball is re-detected while in lost_ball state, return to tracking
            if self.ball_detected:
                self._change_state("tracking")
                
            # If ball remains lost for too long, switch to searching
            elif (current_time - self.state_start_time) > self.lost_timeout:
                self._change_state("searching")
                
    def _change_state(self, new_state):
        """
        Change the robot's state and log the transition.
        
        Args:
            new_state (str): The new state to transition to
        """
        if new_state == self.state:
            return
            
        self.get_logger().info(f"State transition: {self.state} → {new_state}")
        old_state = self.state
        self.state = new_state
        self.state_start_time = time.time()
        
        # Publish state change immediately
        self._publish_state()
        
        # Special handling for certain transitions
        if new_state == "searching":
            self.get_logger().info("Starting search pattern for ball")
            
        elif new_state == "tracking":
            self.get_logger().info("Ball detected - tracking started")
            
        elif new_state == "lost_ball":
            self.get_logger().info(f"Ball lost - last seen {time.time() - self.last_detection_time:.1f}s ago")
            
        # Log detailed state transition for debugging
        self.get_logger().debug(f"State transition details - old: {old_state}, new: {new_state}, "
                               f"ball_detected: {self.ball_detected}, "
                               f"time since detection: {time.time() - self.last_detection_time:.1f}s")