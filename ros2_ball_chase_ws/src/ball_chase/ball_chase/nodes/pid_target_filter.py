import time
import math
import numpy as np
from collections import deque
import logging

class EnhancedTargetFilter:
    """Enhanced filter for target position data with better motion prediction."""
    
    def __init__(self, logger, buffer_size=8, prediction_horizon=0.3, debug_level=0):
        self.logger = logger
        self.debug_level = debug_level
        # Pre-allocate buffers to reduce memory allocations
        self.position_buffer = deque([(0.0, 0.0, 0.0, 0.0)] * buffer_size, maxlen=buffer_size)
        self.trajectory_history = deque([(((0.0, 0.0, 0.0), 0.0))] * 10, maxlen=10)
        
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
        # Add to position buffer
        self.position_buffer.append((position[0], position[1], position[2], current_time))
        
        # Add to trajectory history
        self.trajectory_history.append((position, current_time))
        
        if self.debug_level >= 3:
            if not hasattr(self, '_last_logged_position') or \
               np.linalg.norm(np.array(position) - np.array(getattr(self, '_last_logged_position', (0,0,0)))) > 0.02:
                self.logger.info(f"Buffer updated with position: {position} at {current_time:.3f}", throttle_duration_sec=2.0)
                self._last_logged_position = position
        
    def _calculate_filtered_position(self):
        """Calculate filtered position from recent measurements."""
        if len(self.position_buffer) >= 3:
            # Get the three most recent positions
            recent = list(self.position_buffer)[-3:]
            
            # Weights for weighted average (more weight to recent measurements)
            weights = np.array([0.2, 0.3, 0.5])
            
            # Extract position components and calculate weighted average
            positions = np.array([(p[0], p[1], p[2]) for p in recent])
            self.filtered_position = tuple(np.sum(positions * weights[:, np.newaxis], axis=0))
        else:
            # Not enough data, use the latest position
            latest = self.position_buffer[-1]
            self.filtered_position = (latest[0], latest[1], latest[2])
        
        if self.debug_level >= 3:
            if not hasattr(self, '_last_logged_filtered') or \
               np.linalg.norm(np.array(self.filtered_position) - np.array(getattr(self, '_last_logged_filtered', (0,0,0)))) > 0.02:
                self.logger.info(f"Filtered position: {self.filtered_position}", throttle_duration_sec=2.0)
                self._last_logged_filtered = self.filtered_position
    
    def _update_velocity(self, current_time):
        """Update velocity and acceleration estimates."""
        if len(self.position_buffer) < 2 or self.last_update_time is None:
            return
            
        dt = current_time - self.last_update_time
        if dt <= 0.001:  # Avoid division by zero
            return
            
        # Get the two most recent positions
        prev_pos = self.position_buffer[-2]
        curr_pos = self.position_buffer[-1]
        
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
        if len(self.position_buffer) >= 3 and dt > 0.001:
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
        if len(self.trajectory_history) < 5:
            self.movement_consistency = 0.0
            return
            
        # Get recent positions
        recent_positions = [entry[0] for entry in list(self.trajectory_history)[-5:]]
        
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
        """Predict future position based on current state."""
        if self.filtered_position is None:
            self.predicted_position = self.position_buffer[-1][:3] if self.position_buffer else None
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
        Update the filter with a new position measurement.
        
        Args:
            position: Tuple of (x, y, angle) for the target position
            timestamp: Time of measurement (defaults to current time)
        
        Returns:
            tuple: Filtered position
        """
        current_time = timestamp if timestamp is not None else time.time()
        
        # Early return if this is the first measurement
        if self.last_update_time is None:
            self.position_buffer.append((position[0], position[1], position[2], current_time))
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
        return self.position_buffer[-1][:3] if self.position_buffer else None
    
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
        """Get the n most recent positions."""
        # More efficient extraction of positions using list comprehension
        positions = list(self.position_buffer)[-n:] if len(self.position_buffer) >= n else list(self.position_buffer)
        return [p[:3] for p in positions]
    
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
            self.position_buffer.append((0.0, 0.0, 0.0, 0.0))
            
        for _ in range(traj_size):
            self.trajectory_history.append(((0.0, 0.0, 0.0), 0.0))
        
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
    """Lightweight error tracker that monitors error values over time."""
    
    def __init__(self, name, logger, max_history=8, debug_level=0):
        self.logger = logger
        self.debug_level = debug_level
        self.name = name
        self.current_error = 0.0
        self.previous_error = 0.0
        self.previous_category = None  # For tracking error category with hysteresis
        
        # Pre-allocate error history buffer
        self.error_history = deque([0.0] * max_history, maxlen=max_history)
        
        # State variables
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
        
        # Determine signs (avoid unnecessary comparisons)
        prev_sign = 1 if self.current_error > 0 else (-1 if self.current_error < 0 else 0)
        
        # Update current error
        self.current_error = error
        current_sign = 1 if error > 0 else (-1 if error < 0 else 0)
        
        # Check for sign change efficiently
        if prev_sign != 0 and current_sign != 0 and prev_sign != current_sign:
            self.sign_changes += 1
            
        # Update error history (deque handles the rolling window)
        self.error_history.append(error)
        
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
                self.logger.info(f"Error updated: {error:.3f}, dt={dt:.3f}", throttle_duration_sec=2.0)
                self._last_logged_error = error
        if self.sign_changes > 0 and self.debug_level >= 3:
            if not hasattr(self, '_last_logged_sign_changes') or \
               self.sign_changes != getattr(self, '_last_logged_sign_changes', -1):
                self.logger.info(f"Error sign changes: {self.sign_changes}", throttle_duration_sec=4.0)
                self._last_logged_sign_changes = self.sign_changes
    
    def reset(self):
        """Reset all tracked errors."""
        # Reset scalar values
        self.current_error = 0.0
        self.previous_error = 0.0
        self.accumulated_error = 0.0
        self.sign_changes = 0
        self.error_increasing = False
        self.peak_error = 0.0
        self.last_sign = 0
        self.previous_category = None
        
        # Clear and pre-fill error history
        max_len = self.error_history.maxlen
        self.error_history.clear()
        for _ in range(max_len):
            self.error_history.append(0.0)
        
        if hasattr(self, 'debug_level') and self.debug_level >= 2:
            self.logger.info("Error tracker state reset", throttle_duration_sec=2.0)
    
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
        """Calculate trend of error (increasing/decreasing)."""
        if len(self.error_history) < n:
            return 0.0  # Not enough data
            
        # Get the last n values as numpy array
        history = np.array(list(self.error_history)[-n:])
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