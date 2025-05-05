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

Key Concepts for Beginners:
--------------------------

1. FILTERING SENSOR DATA

   Raw sensor data is often noisy and inconsistent. Filtering helps create
   a more stable and accurate representation of reality:
   
   - WEIGHTED AVERAGING: Giving more importance to recent measurements
   - LOW-PASS FILTERING: Removing high-frequency noise while keeping trends
   - ADAPTIVE FILTERING: Adjusting filter parameters based on conditions
   
   These filtering techniques prevent the robot from reacting to sensor
   noise or brief glitches in detection.

2. MOTION PREDICTION

   For smooth tracking, the robot needs to anticipate where the ball will be:
   
   - PHYSICS-BASED PREDICTION: Using position, velocity, and acceleration
   - MOTION CONSISTENCY ANALYSIS: Detecting how predictably the ball is moving
   - ADAPTIVE PREDICTION HORIZON: Looking further ahead when movement is consistent
   
   Prediction allows the robot to move proactively rather than always
   reacting to where the ball was.

3. ERROR TREND ANALYSIS

   Understanding how error is changing helps the controller adapt:
   
   - TREND CALCULATION: Determining if errors are getting better or worse
   - SIGN CHANGE DETECTION: Identifying when the robot crosses its target
   - OSCILLATION DETECTION: Recognizing when the system is unstable
   
   This information feeds back to the PID controller to adjust its
   behavior dynamically.

4. DIRECTION CHANGE DETECTION

   Detecting when the ball changes direction helps the robot respond appropriately:
   
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
        Initialize the advanced target filtering and prediction system.
        
        Args:
            throttled_logger: Logger with rate limiting to prevent log flooding
            buffer_size: Number of position measurements to store (default: 8)
            prediction_horizon: How far into the future to predict (seconds)
            debug_level: Controls verbosity of diagnostic output (0-3)
            
        The buffer_size determines how many past positions are used for filtering
        and prediction. Larger values create smoother filtering but slower response
        to sudden changes. The prediction_horizon controls how far ahead the system
        tries to predict the ball's position.
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
        Predict where the target will be in the near future based on its motion pattern.
        
        EDUCATIONAL EXPLANATION:
        -----------------------
        Position prediction is essential for smooth tracking. This method uses
        physics principles to estimate where the ball will be shortly in the future.
        
        PREDICTION APPROACHES:
        --------------------
        
        1. PHYSICS-BASED PREDICTION
           For consistent motion, we use the standard physics equation:
           
           Position = Initial Position + Velocity × Time + ½ × Acceleration × Time²
           
           This is the same equation used to predict where a thrown ball will land,
           allowing the robot to anticipate the ball's movement.
           
        2. SIMPLIFIED PREDICTION
           For inconsistent motion (like after direction changes), we use a simpler model:
           
           Position = Initial Position + Velocity × Time × Damping
           
           The damping factor reduces prediction confidence when the ball's
           movement is unpredictable.
           
        3. ADAPTIVE WEIGHTING
           The acceleration term is weighted by movement consistency:
           - High consistency → Full acceleration component
           - Low consistency → Reduced acceleration component
           
           This prevents over-prediction when the ball's movement is erratic.
           
        4. SPECIAL CASES
           - After direction changes: Use more conservative prediction
           - When stationary: Prediction matches current position
           - First measurement: Use raw position with no prediction
           
        These adaptive prediction strategies allow the robot to track both
        predictable and unpredictable ball movements effectively.
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
        Process a new position measurement through the filtering and prediction pipeline.
        
        EDUCATIONAL EXPLANATION:
        -----------------------
        This method is the main entry point for the filtering system. When a new
        position measurement arrives from sensors, this method processes it through
        several stages:
        
        1. BUFFERING
           - Stores the new position in the history buffers
           - Maintains chronological order with timestamps
           - Builds up the dataset needed for filtering and prediction
        
        2. FILTERING
           - Calculates a weighted average of recent positions
           - Reduces sensor noise and random fluctuations
           - Creates a more stable position estimate
        
        3. VELOCITY & ACCELERATION ANALYSIS
           - Calculates how fast the ball is moving and in what direction
           - Determines if the ball is accelerating or decelerating
           - Detects direction changes and movement consistency
        
        4. PREDICTION
           - Uses position, velocity, and acceleration to predict future position
           - Adapts prediction based on movement consistency
           - Uses different prediction models based on movement patterns
        
        This multi-stage process transforms a single noisy position reading into
        a complete understanding of the ball's current state and likely future
        position.
        
        Args:
            position: Tuple of (x, y, angle) for the target position
            timestamp: Time of measurement (defaults to current time)
        
        Returns:
            tuple: Filtered position (x, y, angle)
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
        Initialize error tracking and analysis system.
        
        Args:
            name: Identifier for this tracker (e.g., "LinearX", "Angular")
            throttled_logger: Logger with rate limiting to prevent log flooding
            max_history: Number of error values to track for trend analysis
            debug_level: Controls verbosity of diagnostic output (0-3)
            
        The tracker maintains a history of error values, analyzes trends,
        and provides insights about error behavior to enhance PID control.
        Different controllers (Linear X, Y, Angular) have their own trackers
        to separately analyze each dimension of control.
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
        Process a new error value and update all tracking metrics.
        
        EDUCATIONAL EXPLANATION:
        -----------------------
        This method analyzes a new error value to extract patterns and trends
        that help the PID controller make better decisions. The analysis
        happens in several stages:
        
        1. SIGN ANALYSIS
           - Determines if error is positive, negative, or zero
           - Detects sign changes (crossing zero)
           - Counts sign changes to identify oscillations
           - Example: Error changing from +0.2 to -0.1 indicates crossing the target
        
        2. TREND ANALYSIS
           - Compares current error magnitude with previous
           - Determines if error is growing or shrinking
           - Uses a 5% threshold to ignore minor fluctuations
           - Informs the PID controller about whether its actions are helping
        
        3. PEAK TRACKING
           - Monitors the largest error value observed
           - Helps assess overall control system performance
           - Provides context for controller tuning
        
        4. INTELLIGENT DECAY
           - Applies different decay rates based on error behavior
           - Uses faster decay when error changes sign
           - Prevents accumulated error from growing too large
           - Creates a more responsive control system
        
        5. HISTORY MANAGEMENT
           - Maintains a rolling window of recent error values
           - Supports trend calculation over longer timeframes
           - Preserves enough history for statistical analysis
        
        This comprehensive error analysis enables advanced PID features like
        adaptive gains, zero-crossing handling, and oscillation prevention.
        
        Args:
            error: The new error value to analyze
            dt: Time elapsed since last update (seconds)
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
        Calculate the mathematical trend of recent error values.
        
        EDUCATIONAL EXPLANATION:
        -----------------------
        This method performs linear regression on recent error values to
        determine their trend over time. This information is crucial for
        adaptive PID control.
        
        THE PROCESS:
        -----------
        
        1. LINEAR REGRESSION ANALYSIS
           - Takes the last n error values (typically 3)
           - Fits a straight line to these points using least squares method
           - Calculates the slope of this line
           - The slope indicates how quickly and in what direction error is changing
        
        2. INTERPRETING THE RESULT
           - Positive slope: Error is getting worse over time
           - Negative slope: Error is improving over time
           - Larger magnitude: Error is changing quickly
           - Smaller magnitude: Error is changing slowly
        
        3. HOW THE PID CONTROLLER USES THIS
           - Increasing errors → More aggressive control
           - Decreasing errors → More conservative control
           - Rapid changes → More derivative term influence
           - Slow changes → More integral term influence
        
        This mathematical trend analysis provides a more comprehensive view
        than simply comparing the last two values. It helps the controller
        make smarter decisions about gain adaptation and control strategy.
        
        Args:
            n: Number of recent values to analyze (default: 3)
            
        Returns:
            float: Slope of the error trend (positive = worsening, negative = improving)
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