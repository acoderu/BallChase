#!/usr/bin/env python3

"""
Ground Position Filter - Specialized filter for basketball tracking

This filter is optimized for tracking basketballs that move on the ground,
with the ability to handle rapid direction changes, varying speeds,
and the physical characteristics of a basketball (10-inch diameter).
"""

import numpy as np
import math
from collections import deque
import time


class GroundPositionFilter:
    """
    Specialized filter for tracking basketballs moving on the ground.
    
    Features:
    - Constrains movement to ground plane
    - Handles varying movement speeds
    - Manages direction changes
    - Adjusts for basketball physics
    """
    
    def __init__(self, config=None):
        """Initialize the ground position filter with configuration parameters."""
        # Default configuration
        self.config = {
            "max_speed": 5.0,                # Maximum allowed speed in m/s
            "position_filter_alpha": 0.7,     # Position smoothing factor (higher = more responsive)
            "ground_plane_z": 0.127,         # Default expected z-coordinate for ball center (5 inches - half of basketball)
            "ground_plane_tolerance": 0.03,  # Tolerance for ground plane detection (3 cm)
            "min_speed_threshold": 0.05,     # Min speed to be considered moving (m/s)
            "direction_filter_size": 5,      # Number of samples for direction filtering
            "acceleration_limit": 8.0,       # Maximum allowed acceleration (m/s²)
            "basketball_radius": 0.127       # Basketball radius in meters (5 inches)
        }
        
        # Override defaults with provided config
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
        
        # Initialize state variables
        self.last_position = None         # Last filtered position
        self.last_raw_position = None     # Last raw (unfiltered) position
        self.last_velocity = [0.0, 0.0]   # Last estimated velocity [vx, vy]
        self.last_timestamp = 0           # Timestamp of last update
        
        # Position and velocity history
        self.position_history = deque(maxlen=10)
        self.direction_history = deque(maxlen=self.config["direction_filter_size"])
        self.velocity_magnitude_history = deque(maxlen=5)
        
        # Track statistics
        self.position_jumps = 0      # Count of discontinuous position jumps
        self.filtered_positions = 0  # Count of filtered positions
    
    def reset(self):
        """Reset the filter state."""
        self.last_position = None
        self.last_raw_position = None
        self.last_velocity = [0.0, 0.0]
        self.last_timestamp = 0
        self.position_history.clear()
        self.direction_history.clear()
        self.velocity_magnitude_history.clear()
        self.position_jumps = 0
        self.filtered_positions = 0
    
    def update(self, position, timestamp=None):
        """
        Process a new position measurement and return filtered position.
        
        Args:
            position: Tuple/list of (x, y, z) coordinates
            timestamp: Time of measurement (defaults to current time)
            
        Returns:
            Filtered position as (x, y, z) tuple
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Convert position to list for manipulation
        position = list(position)
        
        # If this is the first position, initialize and return
        if self.last_position is None:
            # Initialize with this position (but constrain z to ground plane)
            ground_position = self._constrain_to_ground_plane(position)
            self.last_position = ground_position
            self.last_raw_position = position
            self.last_timestamp = timestamp
            self.position_history.append(ground_position)
            return tuple(ground_position)
        
        # Calculate time delta
        dt = timestamp - self.last_timestamp
        if dt <= 0:
            # Prevent division by zero or negative time
            dt = 0.01  # 10ms minimum
        
        # Apply ground plane constraint
        ground_position = self._constrain_to_ground_plane(position)
        
        # Calculate current velocity
        curr_velocity = self._estimate_velocity(ground_position, dt)
        
        # Check for impossible movements (speed/acceleration limits)
        filtered_position = self._filter_impossible_movements(ground_position, curr_velocity, dt)
        
        # Apply adaptive smoothing
        smoothed_position = self._apply_smoothing(filtered_position)
        
        # Update state variables
        self.last_raw_position = position
        self.last_position = smoothed_position
        self.last_timestamp = timestamp
        self.position_history.append(smoothed_position)
        self.filtered_positions += 1
        
        return tuple(smoothed_position)
    
    def _constrain_to_ground_plane(self, position):
        """
        Constrain the position to the ground plane.
        The basketball center should be at basketball_radius height above ground.
        """
        # Copy to avoid modifying original
        position = list(position)
        
        # Calculate expected z-height for basketball center
        expected_z = self.config["basketball_radius"]
        
        # Check if measured z is within tolerance
        tolerance = self.config["ground_plane_tolerance"]
        if abs(position[2] - expected_z) > tolerance:
            # Replace with expected z
            position[2] = expected_z
        
        return position
    
    def _estimate_velocity(self, position, dt):
        """Estimate current velocity based on position change."""
        if self.last_position is None:
            return [0.0, 0.0]
        
        # Calculate raw velocity (only x-y plane for ground movement)
        vx = (position[0] - self.last_position[0]) / dt
        vy = (position[1] - self.last_position[1]) / dt
        
        # Calculate direction and magnitude
        magnitude = math.sqrt(vx*vx + vy*vy)
        
        # If moving significantly, record direction
        if magnitude > self.config["min_speed_threshold"]:
            direction = [vx/magnitude, vy/magnitude] if magnitude > 0 else [0, 0]
            self.direction_history.append(direction)
            self.velocity_magnitude_history.append(magnitude)
        
        return [vx, vy]
    
    def _filter_impossible_movements(self, position, velocity, dt):
        """Filter out physically impossible movements based on speed/acceleration."""
        if self.last_position is None:
            return position
        
        # Create a copy to work with
        filtered_pos = list(position)
        
        # Calculate distance and speed
        dx = position[0] - self.last_position[0]
        dy = position[1] - self.last_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        speed = distance / dt
        
        # Check if speed exceeds limit
        if speed > self.config["max_speed"]:
            # Position jump detected - could be noise or tracking error
            self.position_jumps += 1
            
            # Scale back to maximum allowed distance
            max_distance = self.config["max_speed"] * dt
            scale_factor = max_distance / distance if distance > 0 else 0
            
            # Apply scaled movement
            filtered_pos[0] = self.last_position[0] + dx * scale_factor
            filtered_pos[1] = self.last_position[1] + dy * scale_factor
        
        # Check for excessive acceleration
        if len(self.velocity_magnitude_history) > 0:
            # Calculate acceleration
            prev_velocity_magnitude = sum(self.velocity_magnitude_history) / len(self.velocity_magnitude_history)
            current_velocity_magnitude = math.sqrt(velocity[0]**2 + velocity[1]**2)
            acceleration = abs(current_velocity_magnitude - prev_velocity_magnitude) / dt
            
            # If acceleration exceeds limit, dampen the change
            if acceleration > self.config["acceleration_limit"]:
                # Calculate maximum allowed velocity change
                max_vel_change = self.config["acceleration_limit"] * dt
                
                # Direction of velocity change
                if current_velocity_magnitude > prev_velocity_magnitude:
                    # Speeding up - limit acceleration
                    allowed_velocity = prev_velocity_magnitude + max_vel_change
                else:
                    # Slowing down - limit deceleration
                    allowed_velocity = max(0, prev_velocity_magnitude - max_vel_change)
                
                # Scale current velocity
                if current_velocity_magnitude > 0:
                    scale = allowed_velocity / current_velocity_magnitude
                    scaled_vx = velocity[0] * scale
                    scaled_vy = velocity[1] * scale
                    
                    # Calculate new position based on scaled velocity
                    filtered_pos[0] = self.last_position[0] + scaled_vx * dt
                    filtered_pos[1] = self.last_position[1] + scaled_vy * dt
        
        return filtered_pos
    
    def _apply_smoothing(self, position):
        """Apply adaptive position smoothing based on movement dynamics."""
        if self.last_position is None:
            return position
        
        # Get basic smoothing factor
        alpha = self.config["position_filter_alpha"]
        
        # If there's enough history, adapt smoothing factor based on movement
        if len(self.velocity_magnitude_history) > 0:
            avg_speed = sum(self.velocity_magnitude_history) / len(self.velocity_magnitude_history)
            
            # More smoothing for very slow movements (reduce noise)
            if avg_speed < 0.1:  # Nearly stationary
                alpha = 0.3  # More smoothing
            # Less smoothing for fast movements (more responsive)
            elif avg_speed > 1.0:  # Fast movement
                alpha = 0.8  # Less smoothing
        
        # Apply exponential filter
        smoothed = [
            alpha * position[0] + (1 - alpha) * self.last_position[0],
            alpha * position[1] + (1 - alpha) * self.last_position[1],
            position[2]  # Z is already constrained to ground plane
        ]
        
        return smoothed
    
    def get_velocity(self):
        """Return the current estimated velocity as [vx, vy]."""
        return self.last_velocity
    
    def get_speed(self):
        """Return the current speed in m/s."""
        if self.last_velocity:
            return math.sqrt(self.last_velocity[0]**2 + self.last_velocity[1]**2)
        return 0.0
    
    def get_movement_direction(self):
        """Return the current movement direction as [dx, dy] unit vector."""
        # Use average direction for stability
        if not self.direction_history:
            return [0.0, 0.0]
        
        # Calculate average direction from history
        avg_x = 0.0
        avg_y = 0.0
        for direction in self.direction_history:
            avg_x += direction[0]
            avg_y += direction[1]
        
        # Normalize
        count = len(self.direction_history)
        if count > 0:
            avg_x /= count
            avg_y /= count
            
            # Convert back to unit vector
            magnitude = math.sqrt(avg_x*avg_x + avg_y*avg_y)
            if magnitude > 0.001:  # Avoid division by near-zero
                avg_x /= magnitude
                avg_y /= magnitude
        
        return [avg_x, avg_y]
    
    def predict_position(self, time_ahead):
        """
        Predict position after time_ahead seconds based on current state.
        
        Args:
            time_ahead: Time in seconds to predict ahead
            
        Returns:
            Predicted position as (x, y, z)
        """
        if self.last_position is None or not self.last_velocity:
            return None
        
        # Simple linear prediction with current velocity
        pred_x = self.last_position[0] + self.last_velocity[0] * time_ahead
        pred_y = self.last_position[1] + self.last_velocity[1] * time_ahead
        pred_z = self.last_position[2]  # Z stays constant (ground plane)
        
        return (pred_x, pred_y, pred_z)
    
    def get_statistics(self):
        """Return filter statistics as a dictionary."""
        return {
            "position_jumps": self.position_jumps,
            "filtered_positions": self.filtered_positions,
            "position_jump_rate": self.position_jumps / max(1, self.filtered_positions),
            "current_speed": self.get_speed(),
            "average_speed": sum(self.velocity_magnitude_history) / max(1, len(self.velocity_magnitude_history)) 
                            if self.velocity_magnitude_history else 0
        }