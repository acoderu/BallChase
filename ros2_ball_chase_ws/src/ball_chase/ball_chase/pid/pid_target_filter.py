"""
Basketball Tracking Robot - Target Position Filtering and Prediction
===================================================================

EDUCATIONAL DOCUMENTATION
------------------------

This module provides sophisticated filtering and prediction algorithms for tracking
the basketball's position. It serves several critical functions in the robot's
control system:

1. Filtering noisy sensor data to create smooth, stable position estimates
2. Predicting the basketball's future position based on its movement patterns
3. Calculating velocity and acceleration for improved tracking
4. Detecting movement patterns and assessing consistency of motion
5. Tracking error trends to help the PID controller adapt

┌───────────────────────────────────────────────────────────────────────────┐
│               TARGET FILTERING AND PREDICTION PROCESS                     │
└───────────────────────────────────────────────────────────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
    ┌──────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │ Raw Sensor Data  │ │  Position   │ │ Error Trends    │
    │ Processing       │ │  History    │ │ Analysis        │
    └────────┬─────────┘ └─────┬───────┘ └────────┬────────┘
             │                 │                  │
             │                 │                  │
             ▼                 ▼                  ▼
    ┌──────────────────┐ ┌─────────────┐ ┌─────────────────┐
    │• Remove noise    │ │• Calculate  │ │• Monitor error  │
    │• Create weighted │ │  velocity   │ │  trends         │
    │  average         │ │  vectors    │ │• Detect zero    │
    │• Stabilize data  │ │• Determine  │ │  crossings      │
    │                  │ │  acceleration│ │• Analyze        │
    │                  │ │• Assess     │ │  oscillation    │
    │                  │ │  consistency │ │                 │
    └────────┬─────────┘ └─────┬───────┘ └────────┬────────┘
             │                 │                  │
             │                 │                  │
             └─────────────────┼──────────────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │ Predict Future    │
                     │ Position          │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │ To PID Controller │
                     └───────────────────┘

Key Concepts for Beginners:
--------------------------

1. FILTERING SENSOR DATA

   Raw sensor data is often noisy and inconsistent. Filtering helps create
   a more stable and accurate representation of reality:
   
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                      FILTERING OUT SENSOR NOISE                           │
   └───────────────────────────────────────────────────────────────────────────┘
    
    RAW DATA:                               FILTERED DATA:
    
    Position                                Position
      │                                       │
      │     ●                                 │
      │       ●   ●                           │       Smooth
      │    ●    ●  ●     ●                    │       curve
      │  ●           ●                        │      ●────●────●
      │                 ●  ●                  │    ●           ●
      │                    ●                  │   ●             ●
      │                                       │  ●               ●
      └───────────────────────               └───────────────────────
                  Time                                 Time
    
    Using Weighted Average:
    
    ┌─────────────────────────┐
    │ Latest reading: 40%     │
    │ 1 reading ago: 30%      │
    │ 2 readings ago: 20%     │
    │ 3 readings ago: 10%     │
    └─────────────────────────┘
   
   - WEIGHTED AVERAGING: Giving more importance to recent measurements
   - LOW-PASS FILTERING: Removing high-frequency noise while keeping trends
   - ADAPTIVE FILTERING: Adjusting filter parameters based on conditions
   
   These filtering techniques prevent the robot from reacting to sensor
   noise or brief glitches in detection.

2. MOTION PREDICTION

   For smooth tracking, the robot needs to anticipate where the ball will be:
   
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                      PHYSICS-BASED PREDICTION                             │
   └───────────────────────────────────────────────────────────────────────────┘
   
                    (Current position)      (Predicted position)
                           ●                        ●
                           │                       ╱
                           │                      ╱
   Past                    │                     ╱     Future
   positions     ●         │                    ╱      position
                  ╲        │                   ╱
                   ╲       │                  ╱
                    ╲      │                 ╱
                     ●─────●────────────────●
                   
                    │← Measured history →│← Prediction →│
                  
               Prediction formula: future_pos = current_pos + velocity*time + 0.5*accel*time²
   
   - PHYSICS-BASED PREDICTION: Using position, velocity, and acceleration
   - MOTION CONSISTENCY ANALYSIS: Detecting how predictably the ball is moving
   - ADAPTIVE PREDICTION HORIZON: Looking further ahead when movement is consistent
   
   Prediction allows the robot to move proactively rather than always
   reacting to where the ball was.

3. ERROR TREND ANALYSIS

   Understanding how error is changing helps the controller adapt:
   
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                      ERROR TREND ANALYSIS                                 │
   └───────────────────────────────────────────────────────────────────────────┘
   
   Error
   Value                 Zero-crossing
                              │
     +                        │
      │     ╱╲                │               ╱╲
      │    ╱  ╲              │              ╱  ╲
      │   ╱    ╲            │             ╱    ╲
      │  ╱      ╲          │            ╱      ╲
      │ ╱        ╲        │           ╱        ╲
   0 ──┼──────────╲──────┼─────────╱────────────
      │           ╲     │        ╱
      │            ╲   │       ╱        Trend = IMPROVING
      │             ╲ │      ╱          (errors getting smaller)
     -│              ╲│     ╱
      │               ╲    ╱
   
   ┌───────────────┐  ┌────────────────┐  ┌───────────────────┐
   │ STABLE        │  │ IMPROVING      │  │ WORSENING         │
   │ Error steady  │  │ Error shrinking│  │ Error growing     │
   └───────┬───────┘  └───────┬────────┘  └────────┬──────────┘
           │                  │                    │
           │                  │                    │
           ▼                  ▼                    ▼
   ┌───────────────┐  ┌────────────────┐  ┌───────────────────┐
   │ No gain       │  │ Reduce gains   │  │ Increase gains    │
   │ adjustment    │  │ (more gentle)  │  │ (more aggressive) │
   └───────────────┘  └────────────────┘  └───────────────────┘
   
   - TREND CALCULATION: Determining if errors are getting better or worse
   - SIGN CHANGE DETECTION: Identifying when the robot crosses its target
   - OSCILLATION DETECTION: Recognizing when the system is unstable
   
   This information feeds back to the PID controller to adjust its
   behavior dynamically.

4. DIRECTION CHANGE DETECTION

   Detecting when the ball changes direction helps the robot respond appropriately:
   
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                     DIRECTION CHANGE DETECTION                            │
   └───────────────────────────────────────────────────────────────────────────┘
   
   • Previous trajectory                     • Current trajectory
      ┌──────┐                                  ┌──────┐
      │      │                                  │      │
      │  ◎───┼───▶                             │  ◎───┼───▶
      │      │                                  │      │
      └──────┘                                  └──────┘
                          Vector
                          Comparison
                              │
                              ▼
                      ┌───────────────┐
                      │Dot product = 0.9│ → Similar direction (continue prediction)
                      └───────────────┘
   
   • Previous trajectory                     • Current trajectory
      ┌──────┐                                  ┌──────┐
      │      │                                  │      │
      │  ◎───┼───▶                             │  ◎───┼───┘
      │      │                                  │      │  ▼
      └──────┘                                  └──────┘
                          Vector
                          Comparison
                              │
                              ▼
                      ┌───────────────┐
                      │Dot product = 0.1│ → Direction change! (reset prediction)
                      └───────────────┘
   
   - VECTOR COMPARISON: Using dot products to measure direction changes
   - HYSTERESIS: Preventing false positives from minor variations
   - ADAPTIVE RESPONSE: Special handling for sharp direction changes
   
   This allows the robot to immediately adjust when the ball changes course.

Why Not a Kalman Filter Here?
---------------------------

While Kalman filters are excellent for sensor fusion (and are used in our fusion node),
this module uses simpler filtering methods for several important reasons:

1. COMPUTATIONAL EFFICIENCY
   - Kalman filters require matrix operations that are more CPU-intensive
   - This module runs on every control cycle (10-40Hz), demanding efficiency
   - The simpler weighted averaging and low-pass filters are fast enough for real-time control

2. ADAPTABILITY TO NON-LINEAR MOVEMENT
   - Standard Kalman filters assume linear motion models with Gaussian noise
   - Basketball movement can be highly non-linear (bouncing, being thrown, sudden stops)
   - Our adaptive approach can quickly adjust to these non-linear movements
   - Extended Kalman Filters could handle this but are even more computationally expensive

3. BEHAVIOR UNDERSTANDING VS. PURE STATISTICS
   - Kalman filters work on statistical principles without "understanding" the behavior
   - Our approach incorporates domain knowledge about basketball movement
   - We explicitly detect and handle special cases like direction changes
   - This makes the filter more intuitive to tune for the specific application

4. SEPARATION OF CONCERNS
   - Kalman filters are ideal for the fusion node, where multiple sensors with
     different error characteristics need to be combined
   - In the control pipeline, we need predictive filters optimized for control decisions
   - This specialization provides better performance than a one-size-fits-all approach

The fusion node uses Kalman filtering to combine data from multiple sensors
(camera, LIDAR, etc.), where dealing with different sensor noise characteristics
and conflicting measurements is crucial. In contrast, this module focuses on
processing already-fused data to extract movement patterns and make predictions
specifically for control purposes.

This module contains two main classes:
-------------------------------------

1. EnhancedTargetFilter: Advanced filtering and prediction for target positions
2. ErrorTracker: Monitors error trends to enhance PID controller adaptation

Together, these classes create a sophisticated tracking system that helps the
robot follow the basketball smoothly and efficiently, even when it moves
unpredictably or sensor data is imperfect.
"""

import time
import math
import numpy as np
from ball_chase.pid.pid_helpers import CircularBuffer
import logging

class EnhancedTargetFilter:
    """
    Advanced filtering and prediction system for basketball position tracking.
    
    EDUCATIONAL EXPLANATION:
    -----------------------
    This class transforms noisy, inconsistent sensor readings of the basketball's
    position into smooth, stable position estimates and predictions. Think of this
    as the robot's "vision processing center" - it takes raw data about the ball's
    position and creates a reliable understanding of where the ball is and where
    it's going.
    
    KEY FUNCTIONS:
    ------------
    
    1. POSITION FILTERING
       - Combines multiple recent measurements using weighted averaging
       - Gives more importance to newer readings while still using older ones
       - Creates a stable estimate that's resistant to sensor noise
       - Acts like a "mental average" of recent ball positions
    
    2. VELOCITY & ACCELERATION CALCULATION
       - Analyzes position changes over time to determine speed and direction
       - Applies low-pass filtering to smooth out noisy measurements
       - Calculates acceleration to understand how the ball's movement is changing
       - Adapts filtering based on movement consistency
    
    3. FUTURE POSITION PREDICTION
       - Uses physics equations to predict where the ball will be in the near future
       - Adjusts prediction confidence based on movement consistency
       - Reduces prediction confidence after direction changes
       - Allows the robot to move toward where the ball WILL be, not where it WAS
    
    4. MOVEMENT PATTERN ANALYSIS
       - Detects when the ball changes direction
       - Calculates how consistently the ball is moving
       - Identifies whether the ball is moving or stationary
       - Helps the robot anticipate the ball's behavior
    
    REAL-WORLD ANALOGY:
    -----------------
    This system works like a baseball outfielder tracking a fly ball:
    1. The player watches the ball's path (filtering)
    2. Estimates its speed and trajectory (velocity calculation)
    3. Predicts where it will land (future position prediction)
    4. Adjusts based on how the ball is moving (movement pattern analysis)
    5. Runs to where the ball will be, not where it is now
    
    The result is a tracking system that makes smooth, intelligent movements
    even when sensor data is imperfect or the ball's motion is complex.
    """
    
    def __init__(self, throttled_logger, buffer_size=8, prediction_horizon=0.3, debug_level=0):
        """
        Set up the robot's basketball tracking brain.
        
        IMAGINE THIS: 🏀
        ---------------
        Think of this like setting up a basketball player's mental abilities:
        
        - buffer_size (default=8): How many past positions to remember
          - Like a player's short-term memory of where the ball has been
          - Larger value = smoother tracking but slower reactions
          - Smaller value = quicker reactions but jerkier movements
          
        - prediction_horizon (default=0.3): How far ahead to predict (in seconds)
          - Like a player's ability to anticipate where the ball will be
          - 0.3 seconds is about how long it takes for the robot to move
          - This lets the robot "aim where the ball is going to be"
          
        - debug_level (default=0): How much the robot should talk about its thinking
          - 0 = Quiet, no extra information
          - 1 = Basic information about significant events
          - 2 = Detailed tracking information
          - 3 = Extremely detailed diagnostic data (for debugging)
        
        REAL-WORLD CONNECTIONS: 🧠
        -----------------------
        This is similar to how basketball players track the ball:
        - They don't just react to where the ball is right now
        - They remember its recent path (buffer)
        - They anticipate where it's heading (prediction)
        - They adjust their movement based on this information
        
        The key difference is that humans do this naturally, while
        our robot needs these careful calculations to achieve similar skills!
        """
        self.logger = throttled_logger
        self.debug_level = debug_level
        # Use CircularBuffer for position and trajectory history
        self.position_buffer = CircularBuffer(buffer_size, default=(0.0, 0.0, 0.0, 0.0))
        self.trajectory_history = CircularBuffer(10, default=((0.0, 0.0, 0.0), 0.0))
        
        # Configuration parameters
        self.prediction_horizon = prediction_horizon  # seconds
        
        # State variables
        self.last_update_time = None
        self.current_velocity = np.zeros(3)  # x, y, angular as numpy array
        self.filtered_position = None
        self.predicted_position = None
        self.acceleration = np.zeros(3)  # x, y, angular acceleration
        self.is_moving = False
        self.direction_change_detected = False
        self.motion_direction = np.zeros(3)  # normalized direction vector
        self.movement_consistency = 0.0  # 0.0-1.0 measure of consistent movement
        
        # Constants to avoid recomputation
        self.DIRECTION_CHANGE_THRESHOLD = 0.866  # cos(30°)
        self.VELOCITY_THRESHOLD_SQ = 0.0025  # (0.05 m/s)²
        self.MOVEMENT_THRESHOLD = 0.01  # Minimum distance to consider movement
        
    def _update_buffers(self, position, current_time):
        """Update position and trajectory buffers with new measurement."""
        self.position_buffer.add((position[0], position[1], position[2], current_time))
        self.trajectory_history.add((position, current_time))
        
        if self.debug_level >= 3:
            if not hasattr(self, '_last_logged_position') or \
               np.linalg.norm(np.array(position) - np.array(getattr(self, '_last_logged_position', (0,0,0)))) > 0.02:
                self.logger.info(f"Buffer updated with position: {position} at {current_time:.3f}", throttle_duration_sec=2.0)
                self._last_logged_position = position
        
    def _calculate_filtered_position(self):
        """Calculate filtered position from recent measurements."""
        pos_data = self.position_buffer.get_all()
        if len(pos_data) >= 3:
            # Get the three most recent positions
            recent = pos_data[-3:]
            
            # Weights for weighted average (more weight to recent measurements)
            weights = np.array([0.2, 0.3, 0.5])
            
            # Extract position components and calculate weighted average
            positions = np.array([(p[0], p[1], p[2]) for p in recent])
            self.filtered_position = tuple(np.sum(positions * weights[:, np.newaxis], axis=0))
        else:
            # Not enough data, use the latest position
            latest = pos_data[-1]
            self.filtered_position = (latest[0], latest[1], latest[2])
        
        if self.debug_level >= 3:
            if not hasattr(self, '_last_logged_filtered') or \
               np.linalg.norm(np.array(self.filtered_position) - np.array(getattr(self, '_last_logged_filtered', (0,0,0)))) > 0.02:
                self.logger.info(f"Filtered position: {self.filtered_position}", throttle_duration_sec=2.0)
                self._last_logged_filtered = self.filtered_position
    
    def _update_velocity(self, current_time):
        """Update velocity and acceleration estimates."""
        pos_data = self.position_buffer.get_all()
        if len(pos_data) < 2 or self.last_update_time is None:
            return
            
        dt = current_time - self.last_update_time
        if dt <= 0.001:  # Avoid division by zero
            return
            
        # Get the two most recent positions
        prev_pos = pos_data[-2]
        curr_pos = pos_data[-1]
        
        # Calculate raw velocity
        raw_velocity = np.array([
            (curr_pos[0] - prev_pos[0]) / dt,
            (curr_pos[1] - prev_pos[1]) / dt,
            (curr_pos[2] - prev_pos[2]) / dt
        ])
        
        # Check for direction changes
        prev_vel = self.current_velocity[:2]  # Just x, y components
        new_vel = raw_velocity[:2]
        
        # Calculate velocity magnitudes (squared)
        prev_vel_sq = prev_vel[0]**2 + prev_vel[1]**2
        new_vel_sq = new_vel[0]**2 + new_vel[1]**2
        
        # Detect significant direction changes using dot product
        self.direction_change_detected = False
        if prev_vel_sq > self.VELOCITY_THRESHOLD_SQ and new_vel_sq > self.VELOCITY_THRESHOLD_SQ:
            # Compute normalized dot product without unnecessary sqrt operations
            dot_product = prev_vel[0] * new_vel[0] + prev_vel[1] * new_vel[1]
            cos_angle = dot_product / (math.sqrt(prev_vel_sq * new_vel_sq))
            
            # Consider significant direction change if angle > 30 degrees
            if cos_angle < self.DIRECTION_CHANGE_THRESHOLD:
                self.direction_change_detected = True
                if hasattr(self, 'debug_level') and self.debug_level >= 3:
                    self.logger.debug(f"Direction change detected: {cos_angle:.3f}")
        
        if self.direction_change_detected and self.debug_level >= 2:
            self.logger.info(f"Direction change detected in velocity at {current_time:.3f}", throttle_duration_sec=2.0)
        
        # Calculate acceleration if we have enough data
        if len(pos_data) >= 3 and dt > 0.001:
            # Acceleration = change in velocity / time
            raw_accel = (raw_velocity - self.current_velocity) / dt
            
            # Apply low-pass filter for acceleration
            alpha_a = 0.3  # Lower value means more smoothing
            self.acceleration = alpha_a * raw_accel + (1 - alpha_a) * self.acceleration
        
        # Smooth velocity with low-pass filter (adaptive alpha based on consistency)
        alpha = 0.7 + 0.15 * self.movement_consistency  # 0.7-0.85 range
        self.current_velocity = alpha * raw_velocity + (1 - alpha) * self.current_velocity
        
        if self.is_moving and self.debug_level >= 3:
            if not hasattr(self, '_last_logged_velocity') or \
               np.linalg.norm(self.current_velocity - getattr(self, '_last_logged_velocity', np.zeros(3))) > 0.02:
                self.logger.debug(f"Current velocity: {self.current_velocity}", throttle_duration_sec=2.0)
                self._last_logged_velocity = self.current_velocity.copy()
        
        # Update movement status and direction
        self._update_movement_characteristics()
    
    def _update_movement_characteristics(self):
        """Update movement direction and consistency metrics."""
        # Calculate velocity magnitude squared (avoid sqrt for comparison)
        vel_mag_squared = self.current_velocity[0]**2 + self.current_velocity[1]**2
        
        # Determine if target is moving
        self.is_moving = vel_mag_squared > self.VELOCITY_THRESHOLD_SQ
        
        # Update direction vector if moving significantly
        if self.is_moving:
            # Calculate magnitude once (only when needed)
            vel_magnitude = math.sqrt(vel_mag_squared)
            
            # Calculate normalized direction
            new_direction = np.array([
                self.current_velocity[0] / vel_magnitude,
                self.current_velocity[1] / vel_magnitude,
                self.current_velocity[2]  # Angular component
            ])
            
            # Smooth direction updates
            alpha_dir = 0.7
            self.motion_direction = alpha_dir * new_direction + (1 - alpha_dir) * self.motion_direction
        
        # Calculate movement consistency from trajectory history
        self._calculate_movement_consistency()
        
        if self.debug_level >= 3:
            if not hasattr(self, '_last_logged_consistency') or \
               abs(self.movement_consistency - getattr(self, '_last_logged_consistency', 0.0)) > 0.05:
                self.logger.info(f"Movement consistency: {self.movement_consistency:.2f}", throttle_duration_sec=4.0)
                self._last_logged_consistency = self.movement_consistency
    
    def _calculate_movement_consistency(self):
        """Calculate how consistently the target is moving in one direction."""
        traj_data = self.trajectory_history.get_all()
        if len(traj_data) < 5:
            self.movement_consistency = 0.0
            return
            
        # Get recent positions
        recent_positions = [entry[0] for entry in traj_data[-5:]]
        
        # Calculate overall displacement vector
        start_pos = np.array(recent_positions[0][:2])  # x,y only
        end_pos = np.array(recent_positions[-1][:2])
        displacement = end_pos - start_pos
        displacement_len_sq = np.sum(displacement**2)
        
        # If total displacement is too small, consistency is low
        if displacement_len_sq < self.MOVEMENT_THRESHOLD**2:
            self.movement_consistency = 0.0
            return
            
        # Normalize overall direction
        displacement_len = math.sqrt(displacement_len_sq)
        overall_dir = displacement / displacement_len
        
        # Check how well each segment aligns with overall direction
        consistency_sum = 0
        count = 0
        
        for i in range(1, len(recent_positions)):
            prev = np.array(recent_positions[i-1][:2])
            curr = np.array(recent_positions[i][:2])
            segment = curr - prev
            segment_len_sq = np.sum(segment**2)
            
            # Skip tiny movements
            if segment_len_sq < self.MOVEMENT_THRESHOLD**2:
                continue
                
            # Normalize segment
            segment_len = math.sqrt(segment_len_sq)
            segment_dir = segment / segment_len
            
            # Calculate alignment using dot product
            alignment = np.dot(segment_dir, overall_dir)
            consistency_sum += max(0, alignment)  # Only count positive alignment
            count += 1
        
        # Update consistency metric
        self.movement_consistency = consistency_sum / count if count > 0 else 0.0
    
    def _predict_future_position(self):
        """
        Guess where the ball will be in the near future.
        
        IMAGINE THIS: 🔮
        ---------------
        Imagine you're playing basketball and need to intercept a moving ball.
        You don't run to where the ball IS - you run to where it WILL BE.
        
        This method helps the robot do exactly that by answering the question:
        "Where will the ball be in the next 0.3 seconds?"
        
        HOW IT WORKS: ✨
        -------------
        This uses the same physics you learn in school:
        
        When a ball moves through the air, it follows predictable patterns
        based on its current position, speed, and how that speed is changing.
        
        1. For a steady, predictable ball (like rolling across the floor):
           
           📝 Future Position = Current Position + Speed × Time + ½ × Acceleration × Time²
           
           This is exactly like calculating where a thrown baseball will land!
           
        2. For an unpredictable ball (like after a bounce or direction change):
           
           📝 Future Position = Current Position + Speed × Time × Safety Factor
           
           We're more cautious with our prediction because the ball's behavior
           is less predictable.
           
        REAL-WORLD EXAMPLE:
        -----------------
        Think about catching a frisbee:
        - When it's flying smoothly - you can predict exactly where it will go
        - Right after it hits something - its path becomes less predictable
        
        The robot does the same thing - it's more confident in its predictions
        when the ball has been moving consistently in one direction.
        
        This prediction is what makes the robot's movements look smooth and
        intelligent instead of constantly playing "catch-up" with the ball.
        """
        # Handle first measurement case
        if self.filtered_position is None:
            self.predicted_position = self.position_buffer.get_all()[-1][:3] if self.position_buffer else None
            return
            
        # Time horizon for prediction
        t = self.prediction_horizon
        
        if self.is_moving and not self.direction_change_detected:
            # Full physics-based prediction with acceleration for consistent movement
            accel_weight = 0.5 * self.movement_consistency
            
            # Position prediction using physics formula: x = x₀ + v₀t + ½at²
            pred_pos = np.array(self.filtered_position) + \
                      self.current_velocity * t + \
                      0.5 * self.acceleration * t**2 * accel_weight
                      
            self.predicted_position = tuple(pred_pos)
        else:
            # Simpler prediction for non-consistent movement
            damping = 0.7  # Reduce prediction confidence
            pred_pos = np.array(self.filtered_position) + self.current_velocity * t * damping
            self.predicted_position = tuple(pred_pos)
        
        if self.debug_level >= 3:
            if not hasattr(self, '_last_logged_predicted') or \
               np.linalg.norm(np.array(self.predicted_position) - np.array(getattr(self, '_last_logged_predicted', (0,0,0)))) > 0.02:
                self.logger.info(f"Predicted position: {self.predicted_position}", throttle_duration_sec=2.0)
                self._last_logged_predicted = self.predicted_position
    
    def update(self, position, timestamp=None):
        """
        Process a new ball position reading and make sense of it.
        
        IMAGINE THIS: 🧩
        ---------------
        Think of this as the main control center that takes raw ball position
        data and transforms it into something much more useful:
        
        📥 INPUT: "The ball is at position (x, y, z) right now"
        
        ⚙️ PROCESSING STEPS:
        
        1. REMEMBER 📝
           - Add this new position to our memory (buffer)
           - Keep track of when we saw the ball here (timestamp)
           - Build up a history of where the ball has been
        
        2. SMOOTH OUT THE NOISE 🧹
           - Calculate a "weighted average" of recent positions
             (newer positions count more than older ones)
           - This is like how your brain naturally "smooths out"
             small errors when you track an object with your eyes
           - Result: A more stable position that ignores random wobbles
        
        3. UNDERSTAND THE MOTION 📊
           - Calculate the ball's speed and direction (velocity)
           - Figure out if it's speeding up or slowing down (acceleration)
           - Detect when it changes direction (like after a bounce)
           - Determine how consistently it's moving (in a straight line
             vs. erratically)
        
        4. LOOK INTO THE FUTURE 🔮
           - Predict where the ball will be in the next fraction of a second
           - Use different prediction methods depending on how predictably
             the ball is moving
           - Be more conservative with predictions when the motion is erratic
        
        📤 OUTPUT: Filtered position (noise removed) and predicted future position
        
        The robot uses these processed values instead of raw sensor data
        because they create much smoother, more natural tracking behavior.
        
        Args:
            position: Tuple of (x, y, angle) coordinates of where we see the ball
            timestamp: When we saw the ball (uses current time if not provided)
        
        Returns:
            tuple: Filtered position (x, y, angle) with noise removed
        """
        current_time = timestamp if timestamp is not None else time.time()
        
        # Early return if this is the first measurement
        if self.last_update_time is None:
            self.position_buffer.add((position[0], position[1], position[2], current_time))
            self.filtered_position = position
            self.predicted_position = position
            self.last_update_time = current_time
            return position
        
        if self.debug_level >= 2:
            if not hasattr(self, '_last_logged_update') or \
               np.linalg.norm(np.array(position) - np.array(getattr(self, '_last_logged_update', (0,0,0)))) > 0.02:
                self.logger.info(f"Update called with position: {position} at {current_time:.3f}", throttle_duration_sec=2.0)
                self._last_logged_update = position
        
        # Update buffers with new measurement
        self._update_buffers(position, current_time)
        
        # Calculate filtered position
        self._calculate_filtered_position()
        
        # Update velocity and acceleration
        self._update_velocity(current_time)
        
        # Predict future position
        self._predict_future_position()
            
        self.last_update_time = current_time
        return self.filtered_position
    
    def get_filtered_position(self):
        """Get the current filtered position."""
        if self.filtered_position:
            return self.filtered_position
            
        # Fallback to last position if available
        return self.position_buffer.get_all()[-1][:3] if self.position_buffer else None
    
    def get_predicted_position(self):
        """Get the predicted future position based on velocity and acceleration."""
        return self.predicted_position
        
    def get_velocity(self):
        """Get the current velocity estimate."""
        return tuple(self.current_velocity)
    
    def get_acceleration(self):
        """Get the current acceleration estimate."""
        return tuple(self.acceleration)
    
    def get_recent_positions(self, n=3):
        return self.position_buffer.get_latest(n)
    
    def get_movement_info(self):
        """Get information about the movement characteristics."""
        # Calculate velocity magnitude only when needed
        vel_mag = math.sqrt(self.current_velocity[0]**2 + self.current_velocity[1]**2)
        
        return {
            'is_moving': self.is_moving,
            'direction_change': self.direction_change_detected,
            'consistency': self.movement_consistency,
            'velocity_magnitude': vel_mag
        }
    
    def reset(self):
        """Reset the filter state."""
        buffer_size = self.position_buffer.maxlen
        traj_size = self.trajectory_history.maxlen
        
        # Clear and pre-allocate buffers
        self.position_buffer.clear()
        self.trajectory_history.clear()
        
        # Pre-fill with zeros
        for _ in range(buffer_size):
            self.position_buffer.add((0.0, 0.0, 0.0, 0.0))
            
        for _ in range(traj_size):
            self.trajectory_history.add(((0.0, 0.0, 0.0), 0.0))
        
        # Reset state variables
        self.last_update_time = None
        self.current_velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.filtered_position = None
        self.predicted_position = None
        self.is_moving = False
        self.direction_change_detected = False
        self.motion_direction = np.zeros(3)
        self.movement_consistency = 0.0
        
        if hasattr(self, 'debug_level') and self.debug_level >= 2:
            self.logger.info("Target filter state reset", throttle_duration_sec=2.0)


class ErrorTracker:
    """
    Specialized system for analyzing error patterns to enhance PID control.
    
    EDUCATIONAL EXPLANATION:
    -----------------------
    The ErrorTracker monitors how error values change over time, extracting
    valuable information that helps the PID controller make better decisions.
    Think of it as an "error analyst" that studies patterns in the control
    system's performance.
    
    KEY FUNCTIONS:
    ------------
    
    1. ERROR TRENDING
       - Detects whether errors are increasing or decreasing
       - Calculates the slope of recent error values
       - Helps the PID controller adjust gains adaptively
       - Allows the controller to be more aggressive when needed
    
    2. SIGN CHANGE DETECTION
       - Identifies when error crosses zero (changes sign)
       - Counts oscillations (repeated sign changes)
       - Provides critical information for zero-crossing handling
       - Helps prevent overshoot and oscillation
    
    3. ACCUMULATED ERROR TRACKING
       - Maintains a running sum of error over time
       - Applies intelligent decay to prevent windup
       - Adjusts decay rate based on error patterns
       - Provides additional context beyond the I-term
    
    4. PEAK ERROR MONITORING
       - Tracks the largest error value seen
       - Helps assess overall performance
       - Identifies challenging control situations
       - Informs gain adaptation strategies
    
    WHY THIS MATTERS:
    ---------------
    Advanced PID control requires more than just current error values.
    Understanding error trends and patterns allows the controller to:
    
    - Adapt gains based on error behavior
    - Detect and respond to oscillations
    - Handle zero-crossings intelligently
    - Adjust integral terms more effectively
    - React differently to improving vs. worsening errors
    
    This dynamic error analysis is what transforms a basic PID controller
    into a sophisticated adaptive control system.
    """
    
    def __init__(self, name, throttled_logger, max_history=8, debug_level=0):
        """
        Set up a system to analyze how errors change over time.
        
        IMAGINE THIS: 📈
        ---------------
        Think of this like a basketball coach studying game tapes:
        
        - The coach doesn't just look at the current score
        - They watch how the score has changed over time
        - They note patterns (like "we're falling behind in the 3rd quarter")
        - They make strategic adjustments based on these patterns
        
        Similarly, our ErrorTracker doesn't just look at the current error
        (like "the ball is 0.5m too far to the left"). It studies how that
        error has been changing to make smarter decisions.
        
        WHAT IT TRACKS: 🔍
        --------------
        1. Is the error getting better or worse? (TREND)
           - "Are we getting closer to the ball or farther away?"
           
        2. Has the error changed direction? (SIGN CHANGE)
           - "Did we go from being too far left to too far right?"
           
        3. How much error have we accumulated? (ERROR SUM)
           - "How consistently have we been off-target?"
           
        4. What was our biggest error? (PEAK ERROR)
           - "What was our worst performance?"
        
        WHY THIS HELPS: 🌟
        --------------
        A basic controller only sees "now" - it's like driving by
        only looking at where the car is at this exact moment.
        
        The ErrorTracker adds the equivalent of looking ahead
        down the road and anticipating what's coming - making
        the movements much smoother and more intelligent.
        
        Args:
            name: The name of this tracker (like "ForwardControl")
            throttled_logger: A system for logging without spam
            max_history: How many past errors to remember (default=8)
            debug_level: How much information to share (0-3)
        """
        self.name = name
        self.logger = throttled_logger
        self.max_history = max_history
        self.debug_level = debug_level
        self.error_history = CircularBuffer(max_history, default=0.0)
        self.current_error = 0.0
        self.previous_error = 0.0
        self.previous_category = None  # For tracking error category with hysteresis
        
        # State variables
        self.last_correction_time = 0.0
        self.sign_changes = 0  # Count of error sign changes (useful for oscillation detection)
        self.accumulated_error = 0.0
        self.decay_factor = 0.9  # Simplified decay factor
        self.error_increasing = False
        self.peak_error = 0.0
        self.last_sign = 0  # -1, 0, or 1
        
    def update(self, error, dt):
        """
        Analyze the latest error to find helpful patterns.
        
        IMAGINE THIS: 🕵️
        ---------------
        Think of this like a detective investigating clues:
        
        1. THE SIGN DETECTIVE 🔍
           "Did we just cross over our target?"
           
           Example: Error changed from +0.2 to -0.1
           Meaning: We just passed the target! (sign changed)
           Action: The controller should be careful not to overreact
           
        2. THE TREND DETECTIVE 📈
           "Is our aim getting better or worse?"
           
           Example: Error went from 0.5 to 0.7
           Meaning: We're getting farther from target (error increasing)
           Action: The controller might need to be more aggressive
           
        3. THE HISTORY DETECTIVE 📚
           "What pattern do our recent errors show?"
           
           Example: Errors alternate positive/negative repeatedly
           Meaning: We're oscillating around the target
           Action: The controller needs to calm down its responses
        
        REAL-WORLD EXAMPLE: 🚗
        -------------------
        Think about parking a car:
        
        - SIGN CHANGE: You went from being too far forward to too far back
          (you just drove past the perfect position)
          
        - TREND: Your distance from the ideal spot is growing
          (you're backing up too quickly and getting farther away)
          
        - HISTORY: You keep going back and forth multiple times
          (you're overcorrecting and need to make smaller adjustments)
          
        The robot uses these same insights to make its movements more
        natural and avoid the "jerky robot syndrome" of constant 
        overcorrection.
        
        Args:
            error: How far we are from our target (positive or negative)
            dt: How much time passed since the last update (seconds)
        """
        # Store previous error and sign
        self.previous_error = self.current_error
        
        # Determine signs (avoid unnecessary comparisons)
        prev_sign = 1 if self.current_error > 0 else (-1 if self.current_error < 0 else 0)
        
        # Update current error
        self.current_error = error
        current_sign = 1 if error > 0 else (-1 if error < 0 else 0)
        
        # Check for sign change efficiently
        if prev_sign != 0 and current_sign != 0 and prev_sign != current_sign:
            self.sign_changes += 1
            
        # Update error history (deque handles the rolling window)
        self.error_history.add(error)
        
        # Track if error is increasing (with 5% threshold to avoid noise)
        error_abs = abs(error)
        prev_error_abs = abs(self.previous_error)
        self.error_increasing = error_abs > prev_error_abs * 1.05
        
        # Update peak error if current error is larger
        if error_abs > abs(self.peak_error):
            self.peak_error = error
        
        # Choose decay rate based on error direction
        decay = self.decay_factor * (0.5 if current_sign != prev_sign else 1.0)
            
        # Update accumulated error with direction-aware decay
        self.accumulated_error = (self.accumulated_error + error * dt) * decay
        
        # Store last sign
        self.last_sign = current_sign
        
        if self.debug_level >= 3:
            if not hasattr(self, '_last_logged_error') or \
               abs(error - getattr(self, '_last_logged_error', 0.0)) > 0.02:
                self.logger.info(f"Error updated: {error:.3f}, dt={dt:.3f}", throttle_duration_sec=2.0, log_id=f'{self.name}_error')
                self._last_logged_error = error
        if self.sign_changes > 0 and self.debug_level >= 3:
            if not hasattr(self, '_last_logged_sign_changes') or \
               self.sign_changes != getattr(self, '_last_logged_sign_changes', -1):
                self.logger.info(f"Error sign changes: {self.sign_changes}", throttle_duration_sec=4.0, log_id=f'{self.name}_sign_changes')
                self._last_logged_sign_changes = self.sign_changes
    
    def reset(self):
        """Reset all tracked errors."""
        # Reset error history buffer to its original size
        max_len = getattr(self.error_history, 'max_size', 8)
        self.error_history = CircularBuffer(max_len, default=0.0)
        self.current_error = 0.0
        self.previous_error = 0.0
        self.accumulated_error = 0.0
        self.sign_changes = 0
        self.error_increasing = False
        self.peak_error = 0.0
        self.last_sign = 0
        self.previous_category = None
        
        if hasattr(self, 'debug_level') and self.debug_level >= 2:
            self.logger.info("Error tracker state reset", throttle_duration_sec=2.0, log_id=f'{self.name}_reset')
    
    def is_error_growing(self):
        """Check if error is growing compared to previous value."""
        return self.error_increasing
        
    def record_correction(self):
        """Record that a correction was made for this error."""
        self.last_correction_time = time.time()
        # Reduce accumulated error when correction is made
        self.accumulated_error *= 0.5
        
        if hasattr(self, 'debug_level') and self.debug_level >= 2:
            self.logger.info("Correction recorded for error tracker", throttle_duration_sec=2.0)
        
    def get_trend(self, n=3):
        """
        Figure out if errors are getting better or worse over time.
        
        IMAGINE THIS: 📉
        ---------------
        Picture a coach looking at the last few plays:
        
        "Let's see... three possessions ago we were down by 7 points,
        then we were down by 5, and now we're down by 2. That's a clear
        IMPROVING trend - the team is catching up!"
        
        Similarly, this method looks at the last few error values and
        calculates whether we're getting closer to the target (improving)
        or farther away (worsening).
        
        HOW IT WORKS: 📊
        -------------
        1. Take the last few error measurements (usually 3)
        2. Draw the best-fit line through these points
        3. Calculate the slope of this line
        
        WHAT THE SLOPE MEANS:
        -------------------
           📉 Negative slope: We're getting CLOSER to the target (good!)
              - Errors: [0.8, 0.5, 0.2] → Slope = -0.3 (improving)
              
           📈 Positive slope: We're getting FARTHER from the target (bad!)
              - Errors: [0.2, 0.5, 0.8] → Slope = +0.3 (worsening)
              
           ➖ Flat slope (near zero): Error is staying about the same
              - Errors: [0.5, 0.5, 0.5] → Slope = 0.0 (steady)
        
        Why is this better than just comparing the last two values?
        It's more stable and less affected by random fluctuations.
        
        HOW THE ROBOT USES THIS: 🤖
        -----------------------
        - If errors are growing: "I need to be more aggressive!"
        - If errors are shrinking: "I can be more gentle and precise"
        - If errors are changing rapidly: "I need to be more responsive"
        - If errors are changing slowly: "I can be more patient"
        
        Args:
            n: How many recent errors to analyze (default: 3)
            
        Returns:
            A number telling us if errors are improving (negative)
            or getting worse (positive)
        """
        if len(self.error_history) < n:
            return 0.0  # Not enough data
            
        # Get the last n values as numpy array
        history = np.array(self.error_history.get_latest(n))
        x = np.arange(len(history))
        
        if len(x) < 2:
            return 0.0
            
        # More efficient slope calculation using numpy
        try:
            slope, _ = np.polyfit(x, history, 1)
            return slope
        except:
            return 0.0
    
    def is_oscillating(self):
        """Determine if the error is oscillating."""
        # Consider oscillating if there are multiple sign changes recently
        return self.sign_changes >= 2