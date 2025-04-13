import numpy as np
from collections import deque

class GroundFilter:
    """
    Simple filter for position measurements that reduces the effect of ground plane jumps.
    """
    
    def __init__(self, buffer_size=5):
        """
        Initialize the ground filter.
        
        Args:
            buffer_size: Size of position history buffer
        """
        self.positions = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.jumps_detected = 0
        self.avg_speed = 0.0
    
    def filter(self, position):
        """
        Filter position based on history.
        
        Args:
            position: Position measurement (np.ndarray)
        
        Returns:
            Filtered position
        """
        # Add new measurement to history
        current_time = np.datetime64('now').astype(float) / 1e9
        
        # Check for ground jumps
        if len(self.positions) > 0:
            last_position = self.positions[-1]
            last_time = self.timestamps[-1]
            
            # Calculate time difference and distance
            dt = current_time - last_time
            dist = np.linalg.norm(position - last_position)
            
            # If dt is very small, avoid division by zero
            if dt > 1e-6:
                speed = dist / dt
                
                # Update average speed with exponential smoothing
                if self.avg_speed == 0.0:
                    self.avg_speed = speed
                else:
                    alpha = 0.3  # Smoothing factor
                    self.avg_speed = (1 - alpha) * self.avg_speed + alpha * speed
                
                # Check for sudden jumps in z-axis (ground plane)
                # This is a simple heuristic - in a real system, you might want more sophisticated logic
                if abs(position[2] - last_position[2]) > 0.05 and speed > 0.5:
                    self.jumps_detected += 1
        
        # Store new position
        self.positions.append(position.copy())
        self.timestamps.append(current_time)
        
        # If we don't have enough history, return current measurement
        if len(self.positions) < 3:
            return position
        
        # Simple weighted average with more weight on recent positions
        weights = np.linspace(0.5, 1.0, len(self.positions))
        weighted_positions = [w * p for w, p in zip(weights, self.positions)]
        
        filtered_position = sum(weighted_positions) / sum(weights)
        
        return filtered_position
    
    def get_stats(self):
        """Get filter statistics"""
        return {
            'avg_speed': self.avg_speed,
            'jumps': self.jumps_detected
        }