#!/usr/bin/env python3

"""
MULTI-SENSOR SYNCHRONIZATION BUFFER FOR ROS2
------------------------------------------

This module provides a buffer system that stores recent measurements from
multiple sensors and helps find measurements that happened at approximately 
the same time, designed specifically for ROS2 Humble.

This optimized version is intended to run efficiently on Raspberry Pi 5 hardware.
"""

from collections import deque  # Efficient for our buffer implementation
import rclpy
from rclpy.time import Time
from typing import Dict, Any, Optional, List


class SimpleSensorBuffer:
    """
    A buffer that stores recent measurements from different sensors and
    helps find measurements that were taken at approximately the same time.
    
    This version is optimized for ROS2 Humble running on Raspberry Pi 5 hardware.
    """
    
    def __init__(self, sensor_names: List[str], buffer_size: int = 20, max_time_diff: float = 0.1):
        """
        Set up a new buffer for synchronizing sensor data
        
        Args:
            sensor_names: List of sensor names (e.g., ['camera', 'lidar', 'gps'])
            buffer_size: How many measurements to remember for each sensor
            max_time_diff: Maximum time difference (in seconds) to consider measurements "synchronized"
        """
        # Create buffers for each sensor
        self.buffers: Dict[str, deque] = {}
        for name in sensor_names:
            self.buffers[name] = deque(maxlen=buffer_size)
        
        self.max_time_diff = max_time_diff
        self.buffer_size = buffer_size
        self.sensor_names = sensor_names
        
        # Add timestamp tracking for performance analysis
        self.last_successful_sync = None
        self.last_failed_sync = None
    
    def add_measurement(self, sensor_name: str, data: Any, timestamp):
        """
        Add a new measurement from a sensor to our buffer
        
        Args:
            sensor_name: Which sensor this measurement came from
            data: The actual measurement data (could be any type)
            timestamp: When the measurement was taken (a ROS2 Time object)
        """
        if sensor_name not in self.buffers:
            return
        
        # Store the measurement with its timestamp
        measurement = {
            'data': data,
            'timestamp': timestamp
        }
        
        self.buffers[sensor_name].append(measurement)
    
    def find_synchronized_data(self) -> Optional[Dict[str, Any]]:
        """
        Find sensor measurements that were taken at approximately the same time
        
        Returns:
            Dictionary with sensor names as keys and their data as values,
            or None if synchronized measurements couldn't be found
        """
        # First check if we have data from each sensor
        for name, buffer in self.buffers.items():
            if not buffer:
                return None
                
        # Find the measurement with the most recent timestamp
        reference_time = None
        reference_sensor = None
        
        for name, buffer in self.buffers.items():
            latest = buffer[-1]
            if reference_time is None or self._timestamp_gt(latest['timestamp'], reference_time):
                reference_time = latest['timestamp']
                reference_sensor = name
        
        # Start collecting synchronized data with our reference measurement
        result = {
            reference_sensor: self.buffers[reference_sensor][-1]['data']
        }
        
        # For each other sensor, find its measurement closest to our reference time
        for name, buffer in self.buffers.items():
            # Skip the reference sensor since we already added it
            if name == reference_sensor:
                continue
                
            # Find the measurement with timestamp closest to reference_time
            closest_measurement = None
            smallest_time_diff = float('inf')
            
            for measurement in buffer:
                time_diff = abs(self._timestamp_diff(measurement['timestamp'], reference_time))
                
                if time_diff < smallest_time_diff:
                    smallest_time_diff = time_diff
                    closest_measurement = measurement
            
            # If within our allowed time difference, add it to our result
            if smallest_time_diff <= self.max_time_diff:
                result[name] = closest_measurement['data']
            else:
                # If not close enough, we can't provide synchronized data
                self.last_failed_sync = reference_time
                return None
                
        # Update successful sync time
        self.last_successful_sync = reference_time
        
        # Return the synchronized measurements
        return result
    
    def _timestamp_diff(self, t1, t2) -> float:
        """Calculate the difference between two ROS2 timestamps in seconds."""
        # Handle different timestamp formats (Time or builtin_interfaces.msg.Time)
        if hasattr(t1, 'sec') and hasattr(t1, 'nanosec'):
            ts1 = t1.sec + t1.nanosec * 1e-9
        else:
            ts1 = float(t1)
            
        if hasattr(t2, 'sec') and hasattr(t2, 'nanosec'):
            ts2 = t2.sec + t2.nanosec * 1e-9
        else:
            ts2 = float(t2)
            
        return ts1 - ts2
    
    def _timestamp_gt(self, t1, t2) -> bool:
        """Check if timestamp t1 is greater than t2."""
        return self._timestamp_diff(t1, t2) > 0
    
    def get_buffer_stats(self) -> Dict[str, int]:
        """Get the current size of each sensor's buffer."""
        return {name: len(buffer) for name, buffer in self.buffers.items()}
