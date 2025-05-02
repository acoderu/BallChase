#!/usr/bin/env python3

"""
Basketball Movement Predictor 

This module provides common prediction functionality for basketball tracking across
different sensor types (LIDAR, depth camera). It builds on the GroundPositionFilter
to add advanced prediction capabilities while keeping code shared across nodes.
"""

import numpy as np
import math
from collections import deque
import time
from typing import List, Tuple, Dict, Optional, Union

# Import the base position filter
from ball_chase.utilities.ground_position_filter import GroundPositionFilter


class BasketballPredictor:
    """
    Advanced predictor for basketball movements that can be used by different sensor nodes.
    
    Features:
    - Short and long-term position prediction
    - Movement path prediction 
    - Confidence estimations for predictions
    - Basketball physical constraints (movement on ground)
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the basketball predictor.
        
        Args:
            config: Optional configuration dictionary
        """
        # Default configuration
        self.config = {
            "short_term_prediction": 0.2,   # Short-term prediction in seconds (200ms)
            "mid_term_prediction": 0.5,     # Mid-term prediction in seconds (500ms)
            "long_term_prediction": 1.0,    # Long-term prediction in seconds (1s)
            "confidence_decay_rate": 0.5,   # How quickly confidence decays with prediction time
            "min_history_for_prediction": 3, # Minimum position history needed for prediction
            "max_prediction_points": 10,    # Maximum number of prediction points
            "prediction_step": 0.1,         # Time step for prediction path in seconds
            "basketball_radius": 0.127,     # Basketball radius in meters (5 inches)
            "court_bounds_x": (-5.0, 5.0),  # Court boundaries in X direction (meters)
            "court_bounds_y": (-5.0, 5.0)   # Court boundaries in Y direction (meters)
        }
        
        # Override defaults with provided config
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
        
        # Initialize the position filter (used for current position filtering)
        self.position_filter = GroundPositionFilter(config)
        
        # Additional state for prediction
        self.velocity_history = deque(maxlen=10)  # Store recent velocities
        self.acceleration_history = deque(maxlen=8)  # Store recent accelerations
        
        # Last prediction info
        self.last_prediction_time = 0
        self.last_prediction = None
        self.last_prediction_confidence = 0.0
        
        # Path prediction
        self.predicted_path = []
        self.path_confidence = 0.0
    
    def update(self, position: Tuple[float, float, float], timestamp: float = None) -> Tuple[float, float, float]:
        """
        Update the predictor with a new position measurement.
        
        Args:
            position: Tuple (x, y, z) of current position
            timestamp: Measurement timestamp (defaults to current time)
            
        Returns:
            Filtered position as (x, y, z) tuple
        """
        # Delegate position filtering to the GroundPositionFilter
        filtered_position = self.position_filter.update(position, timestamp)
        
        # Update velocity history
        current_velocity = self.position_filter.get_velocity()
        if current_velocity:
            self.velocity_history.append((current_velocity, timestamp or time.time()))
            
            # Calculate acceleration if we have enough velocity history
            if len(self.velocity_history) >= 2:
                self._update_acceleration()
        
        return filtered_position
    
    def _update_acceleration(self):
        """Calculate and update acceleration history based on velocity changes."""
        if len(self.velocity_history) < 2:
            return
            
        # Get the two most recent velocity entries
        (v1, t1) = self.velocity_history[-1]  # Most recent
        (v0, t0) = self.velocity_history[-2]  # Second most recent
        
        # Calculate time delta
        dt = t1 - t0
        if dt <= 0.001:  # Avoid division by near-zero
            return
        
        # Calculate acceleration components
        ax = (v1[0] - v0[0]) / dt
        ay = (v1[1] - v0[1]) / dt
        
        # Store acceleration
        self.acceleration_history.append(([ax, ay], t1))
    
    def predict(self, time_ahead: float) -> Tuple[Optional[Tuple[float, float, float]], float]:
        """
        Predict basketball position at a specific time in the future.
        
        Args:
            time_ahead: Seconds in the future to predict
            
        Returns:
            Tuple of (predicted_position, confidence)
            where predicted_position is (x, y, z) or None if prediction not possible
        """
        # Check if we have enough history for prediction
        if len(self.velocity_history) < self.config["min_history_for_prediction"]:
            return None, 0.0
        
        # Get current position and velocity
        current_position = self.position_filter.last_position
        if not current_position:
            return None, 0.0
            
        current_velocity = self.position_filter.get_velocity()
        current_time = time.time()
        
        # Simple prediction with current velocity as baseline
        pred_position = self.position_filter.predict_position(time_ahead)
        if not pred_position:
            return None, 0.0
            
        # Enhance prediction with acceleration if available
        if len(self.acceleration_history) > 0:
            # Get average acceleration
            total_ax, total_ay = 0.0, 0.0
            count = 0
            
            # Use more recent acceleration samples with higher weight
            weight_sum = 0
            for i, (acc, _) in enumerate(reversed(self.acceleration_history)):
                # More recent accelerations get higher weight
                weight = 1.0 / (i + 1)
                weight_sum += weight
                
                total_ax += acc[0] * weight
                total_ay += acc[1] * weight
                count += 1
            
            # Calculate weighted average
            if count > 0 and weight_sum > 0:
                avg_ax = total_ax / weight_sum
                avg_ay = total_ay / weight_sum
                
                # Dampen acceleration effect for longer predictions
                accel_damping = max(0.0, 1.0 - time_ahead)
                
                # Adjust prediction with acceleration
                # x = x0 + v0*t + 0.5*a*t^2
                pred_x = pred_position[0] + 0.5 * avg_ax * time_ahead * time_ahead * accel_damping
                pred_y = pred_position[1] + 0.5 * avg_ay * time_ahead * time_ahead * accel_damping
                
                # Update prediction
                pred_position = (pred_x, pred_y, pred_position[2])
        
        # Apply physical constraints
        pred_position = self._apply_physical_constraints(pred_position)
        
        # Calculate confidence
        confidence = self._calculate_prediction_confidence(time_ahead, current_time)
        
        # Store this prediction
        self.last_prediction = pred_position
        self.last_prediction_time = current_time
        self.last_prediction_confidence = confidence
        
        return pred_position, confidence
    
    def _apply_physical_constraints(self, position: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Apply physical constraints to predicted position."""
        x, y, z = position
        
        # Enforce court boundaries if defined
        x_min, x_max = self.config["court_bounds_x"]
        y_min, y_max = self.config["court_bounds_y"]
        
        x = max(x_min, min(x_max, x))
        y = max(y_min, min(y_max, y))
        
        # Ensure z is at expected basketball height
        z = self.config["basketball_radius"]
        
        return (x, y, z)
    
    def _calculate_prediction_confidence(self, time_ahead: float, current_time: float) -> float:
        """
        Calculate confidence value for a prediction.
        
        Returns:
            Confidence value between 0.0 and 1.0
        """
        # Base confidence starts high and decreases with prediction time
        base_confidence = max(0.0, 1.0 - time_ahead * self.config["confidence_decay_rate"])
        
        # Adjust confidence based on velocity stability
        velocity_stability = self._calculate_velocity_stability()
        
        # Adjust confidence based on measurement consistency
        measurement_consistency = self._calculate_measurement_consistency()
        
        # Calculate overall confidence
        confidence = base_confidence * 0.4 + velocity_stability * 0.4 + measurement_consistency * 0.2
        
        # Cap between 0 and 1
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence
    
    def _calculate_velocity_stability(self) -> float:
        """
        Calculate stability of velocity measurements.
        
        Returns:
            Stability score between 0.0 and 1.0
        """
        if len(self.velocity_history) < 2:
            return 0.5  # Moderate confidence with minimal data
        
        # Calculate variance in velocity magnitude
        velocities = [v[0] for v, _ in self.velocity_history]
        magnitudes = [math.sqrt(v[0]**2 + v[1]**2) for v in velocities]
        
        if not magnitudes:
            return 0.5
            
        # Simple statistical measures
        mean_magnitude = sum(magnitudes) / len(magnitudes)
        if mean_magnitude < 0.001:  # Nearly stationary
            return 0.9  # High confidence for stationary objects
            
        # Calculate normalized variance
        squared_diffs = [(m - mean_magnitude)**2 for m in magnitudes]
        variance = sum(squared_diffs) / len(squared_diffs)
        normalized_variance = min(1.0, variance / (mean_magnitude**2))
        
        # Convert to stability (lower variance = higher stability)
        stability = 1.0 - normalized_variance
        
        return stability
    
    def _calculate_measurement_consistency(self) -> float:
        """
        Calculate consistency of recent position measurements.
        
        Returns:
            Consistency score between 0.0 and 1.0
        """
        # Get position history from filter
        history = list(self.position_filter.position_history)
        if len(history) < 3:
            return 0.5  # Default with limited data
            
        # Check how well positions fit a smooth path
        # For a simple measure, check if points roughly follow a line/curve
        
        # Look at the most recent positions
        recent = history[-min(5, len(history)):]
        
        # Calculate average deviation from expected positions
        deviations = []
        if len(recent) >= 3:
            for i in range(1, len(recent)-1):
                # Expected position if movement was perfectly smooth
                expected_x = (recent[i-1][0] + recent[i+1][0]) / 2
                expected_y = (recent[i-1][1] + recent[i+1][1]) / 2
                
                # Actual position
                actual_x = recent[i][0]
                actual_y = recent[i][1]
                
                # Calculate deviation
                deviation = math.sqrt((expected_x - actual_x)**2 + (expected_y - actual_y)**2)
                deviations.append(deviation)
        
        if not deviations:
            return 0.5
            
        # Calculate average deviation
        avg_deviation = sum(deviations) / len(deviations)
        
        # Convert to consistency score
        # Higher deviation = lower consistency
        consistency = max(0.0, 1.0 - avg_deviation * 2)  # Scale factor of 2 for sensitivity
        
        return consistency
    
    def predict_path(self, duration: float = None) -> List[Tuple[Tuple[float, float, float], float]]:
        """
        Predict the basketball's path over time.
        
        Args:
            duration: How far ahead to predict (defaults to long_term_prediction)
                
        Returns:
            List of (position, confidence) tuples along predicted path
        """
        if duration is None:
            duration = self.config["long_term_prediction"]
        
        # Calculate number of points to generate
        step = self.config["prediction_step"]
        num_points = min(self.config["max_prediction_points"], 
                          int(duration / step) + 1)
        
        # Generate predictions at each time step
        predicted_path = []
        for i in range(num_points):
            time_ahead = step * i
            pos, conf = self.predict(time_ahead)
            if pos:
                predicted_path.append((pos, conf))
        
        # Store for later use
        self.predicted_path = predicted_path
        if predicted_path:
            avg_confidence = sum(conf for _, conf in predicted_path) / len(predicted_path)
            self.path_confidence = avg_confidence
        
        return predicted_path
    
    def get_velocity(self) -> List[float]:
        """Get current velocity from the filter."""
        return self.position_filter.get_velocity()
    
    def get_speed(self) -> float:
        """Get current speed from the filter."""
        return self.position_filter.get_speed()
    
    def get_movement_direction(self) -> List[float]:
        """Get current movement direction as a unit vector."""
        return self.position_filter.get_movement_direction()
    
    def get_statistics(self) -> Dict:
        """Get combined statistics about tracking and prediction."""
        # Get base statistics from filter
        stats = self.position_filter.get_statistics()
        
        # Add prediction statistics
        stats.update({
            "prediction_confidence": self.last_prediction_confidence,
            "path_confidence": self.path_confidence,
            "predictions_generated": len(self.predicted_path),
            "last_prediction_time": self.last_prediction_time
        })
        
        return stats
    
    def reset(self):
        """Reset the predictor state."""
        self.position_filter.reset()
        self.velocity_history.clear()
        self.acceleration_history.clear()
        self.last_prediction = None
        self.last_prediction_time = 0
        self.last_prediction_confidence = 0.0
        self.predicted_path = []
        self.path_confidence = 0.0