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
    Filter for tracking ground-constrained basketball movement.
    Improved with motion state feedback and enhanced jump detection.
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        
        # Basketball parameters
        self.max_speed = self.config.get("max_speed", 5.0)
        self.position_filter_alpha = self.config.get("position_filter_alpha", 0.7)
        self.ground_plane_z = self.config.get("ground_plane_z", 0.127)  # Default basketball radius
        self.basketball_radius = self.config.get("basketball_radius", 0.127)
        
        # Current state
        self.current_position = None
        self.filtered_position = None
        self.current_velocity = [0.0, 0.0, 0.0]
        self.last_update_time = None
        
        # History for velocity calculation
        self.position_history = []
        self.time_history = []
        
        # Increase history length for smoother filtering
        self.history_max_length = 10  # Increased from 5 to 10 for better smoothing
        
        # Statistics
        self.position_jumps = 0
        self.jump_threshold = 0.1  # Minimum jump distance to count
        self.total_distance = 0.0
        self.max_observed_speed = 0.0
        
        # NEW: Add stationary history for state-specific filtering
        self.stationary_positions = []
        self.stationary_max_length = 30  # Longer history for stationary objects
        
        # NEW: Add motion state information
        self.motion_state = "unknown"
        self.motion_state_count = {"stationary": 0, "long_stationary": 0, "moving": 0}
        
        # NEW: Velocity validation system
        self.velocity_history = []
        self.velocity_history_max_length = 10
        self.velocity_confidence = 1.0  # 0.0-1.0 scale
        
    def reset(self):
        """Reset the filter state."""
        self.current_position = None
        self.filtered_position = None
        self.current_velocity = [0.0, 0.0, 0.0]
        self.last_update_time = None
        self.position_history = []
        self.time_history = []
        self.position_jumps = 0
        self.total_distance = 0.0
        self.max_observed_speed = 0.0
        self.stationary_positions = []
        self.velocity_history = []
    
    def update(self, position, timestamp, motion_state=None):
        """
        Update the filter with a new position measurement.
        
        Args:
            position (list): [x, y, z] position
            timestamp (float): Current time
            motion_state (str, optional): Current motion state if available
            
        Returns:
            list: Filtered position
        """
        # Update motion state if provided
        if motion_state is not None:
            self.motion_state = motion_state
            # Update state counter
            if motion_state in ["stationary", "long_stationary"]:
                self.motion_state_count["stationary"] += 1
                if motion_state == "long_stationary":
                    self.motion_state_count["long_stationary"] += 1
                self.motion_state_count["moving"] = max(0, self.motion_state_count["moving"] - 1)
            else:
                self.motion_state_count["moving"] += 1
                self.motion_state_count["stationary"] = max(0, self.motion_state_count["stationary"] - 1)
                self.motion_state_count["long_stationary"] = max(0, self.motion_state_count["long_stationary"] - 1)
        
        # For the first update, just store the position
        if self.current_position is None:
            self.current_position = list(position)
            self.filtered_position = list(position)
            self.last_update_time = timestamp
            return list(position)
        
        # Calculate time delta
        dt = timestamp - self.last_update_time
        if dt <= 0:
            # Invalid time delta, just return current position
            return self.filtered_position
            
        # Store previous position for jump detection
        prev_position = list(self.filtered_position)
        
        # Calculate distance from previous position
        dx = position[0] - self.filtered_position[0]
        dy = position[1] - self.filtered_position[1]
        dz = position[2] - self.filtered_position[2]
        distance = (dx*dx + dy*dy + dz*dz) ** 0.5
        
        # Calculate speed
        speed = distance / dt
        
        # NEW: Apply velocity sanity check based on motion state
        velocity_valid = True
        expected_max_speed = self.max_speed
        
        # Adjust expected max speed based on motion state
        if self.motion_state == "stationary":
            expected_max_speed = 0.5  # 0.5 m/s max for stationary objects
        elif self.motion_state == "long_stationary":
            expected_max_speed = 0.2  # 0.2 m/s max for long-term stationary
        
        # Check if speed is implausible
        if speed > expected_max_speed:
            # Calculate how many standard deviations from recent speeds
            if len(self.velocity_history) >= 3:
                recent_speeds = [v[3] for v in self.velocity_history[-3:]]
                avg_speed = sum(recent_speeds) / len(recent_speeds)
                if speed > avg_speed * 3:  # More than 3x recent average
                    velocity_valid = False
                    # Treat as jump rather than valid movement
                    self.position_jumps += 1
        
        # NEW: Special handling for stationary and long-stationary objects
        if self.motion_state in ["stationary", "long_stationary"]:
            # For stationary objects, use stronger filtering to reject jumps
            filter_alpha = self.position_filter_alpha * 0.5  # More aggressive smoothing (0.35 vs 0.7)
            
            # For long-stationary objects, use even stronger filtering
            if self.motion_state == "long_stationary":
                filter_alpha = self.position_filter_alpha * 0.3  # Very aggressive smoothing (0.21 vs 0.7)
                
                # Add position to stationary history
                self.stationary_positions.append(list(position))
                if len(self.stationary_positions) > self.stationary_max_length:
                    self.stationary_positions.pop(0)
                
                # If we have enough history, use centroid of recent positions
                if len(self.stationary_positions) >= 5:
                    # Calculate centroid of recent positions
                    centroid = [0, 0, 0]
                    for pos in self.stationary_positions:
                        centroid[0] += pos[0]
                        centroid[1] += pos[1]
                        centroid[2] += pos[2]
                    centroid = [c / len(self.stationary_positions) for c in centroid]
                    
                    # Use centroid with very slight update from new position
                    self.filtered_position = [
                        centroid[0] * 0.9 + position[0] * 0.1,
                        centroid[1] * 0.9 + position[1] * 0.1,
                        self.ground_plane_z  # Force to ground plane
                    ]
                else:
                    # Standard EMA filter with lower alpha
                    self.filtered_position = [
                        self.filtered_position[0] * (1 - filter_alpha) + position[0] * filter_alpha,
                        self.filtered_position[1] * (1 - filter_alpha) + position[1] * filter_alpha,
                        self.ground_plane_z  # Force to ground plane
                    ]
            else:
                # Regular stationary - standard EMA filter with lower alpha
                self.filtered_position = [
                    self.filtered_position[0] * (1 - filter_alpha) + position[0] * filter_alpha,
                    self.filtered_position[1] * (1 - filter_alpha) + position[1] * filter_alpha,
                    self.ground_plane_z  # Force to ground plane
                ]
        else:
            # Moving object - use standard filter
            # Check if this is a position jump rather than real movement
            if distance > self.jump_threshold and speed > self.max_speed:
                self.position_jumps += 1
                # Use less filtering weight for suspected jumps (0.3 vs 0.7)
                filter_alpha = self.position_filter_alpha * 0.3
            else:
                # Normal EMA filter
                filter_alpha = self.position_filter_alpha
                
            # Apply EMA filter
            self.filtered_position = [
                self.filtered_position[0] * (1 - filter_alpha) + position[0] * filter_alpha,
                self.filtered_position[1] * (1 - filter_alpha) + position[1] * filter_alpha,
                self.ground_plane_z  # Force to ground plane
            ]
        
        # Calculate velocity vector
        velocity = [
            (self.filtered_position[0] - prev_position[0]) / dt,
            (self.filtered_position[1] - prev_position[1]) / dt,
            0.0  # No vertical velocity for ground movement
        ]
        
        # Calculate speed from velocity
        current_speed = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2) ** 0.5
        
        # NEW: Apply velocity validation
        if velocity_valid:
            # Store velocity with timestamp and speed
            self.velocity_history.append((velocity[0], velocity[1], velocity[2], current_speed, timestamp))
            if len(self.velocity_history) > self.velocity_history_max_length:
                self.velocity_history.pop(0)
                
            # Update current velocity with smoothing
            if len(self.velocity_history) >= 3:
                # Use weighted average of recent velocities
                total_weight = 0
                weighted_vel = [0, 0, 0]
                
                for i, (vx, vy, vz, spd, ts) in enumerate(self.velocity_history[-3:]):
                    # More recent velocities get higher weight
                    weight = i + 1
                    total_weight += weight
                    weighted_vel[0] += vx * weight
                    weighted_vel[1] += vy * weight
                    weighted_vel[2] += vz * weight
                
                self.current_velocity = [v / total_weight for v in weighted_vel]
            else:
                self.current_velocity = velocity
        
        # Update history
        self.position_history.append(list(self.filtered_position))
        self.time_history.append(timestamp)
        
        if len(self.position_history) > self.history_max_length:
            self.position_history.pop(0)
            self.time_history.pop(0)
        
        # Update statistics
        self.total_distance += distance
        self.max_observed_speed = max(self.max_observed_speed, current_speed)
        
        # Update last update time
        self.last_update_time = timestamp
        
        # Return filtered position
        return self.filtered_position
    
    def get_velocity(self):
        """Get current velocity vector."""
        return self.current_velocity
    
    def get_speed(self):
        """Get current speed magnitude."""
        return (self.current_velocity[0]**2 + 
                self.current_velocity[1]**2 + 
                self.current_velocity[2]**2) ** 0.5
    
    def get_statistics(self):
        """Get filter statistics for diagnostics."""
        current_speed = self.get_speed()
        
        # Calculate average speed from position history
        avg_speed = 0.0
        if len(self.position_history) >= 2 and len(self.time_history) >= 2:
            total_distance = 0.0
            for i in range(1, len(self.position_history)):
                p1 = self.position_history[i-1]
                p2 = self.position_history[i]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dz = p2[2] - p1[2]
                total_distance += (dx*dx + dy*dy + dz*dz) ** 0.5
            
            time_span = self.time_history[-1] - self.time_history[0]
            if time_span > 0:
                avg_speed = total_distance / time_span
        
        return {
            "current_speed": current_speed,
            "average_speed": avg_speed,
            "position_jumps": self.position_jumps,
            "total_distance": self.total_distance,
            "max_observed_speed": self.max_observed_speed,
            "motion_state": self.motion_state
        }
    
    def predict_position(self, time_ahead):
        """
        Predict position after time_ahead seconds based on current state.
        
        Args:
            time_ahead: Time in seconds to predict ahead
            
        Returns:
            Predicted position as (x, y, z)
        """
        if self.filtered_position is None or not self.current_velocity:
            return None
        
        # Simple linear prediction with current velocity
        pred_x = self.filtered_position[0] + self.current_velocity[0] * time_ahead
        pred_y = self.filtered_position[1] + self.current_velocity[1] * time_ahead
        pred_z = self.filtered_position[2]  # Z stays constant (ground plane)
        
        return (pred_x, pred_y, pred_z)