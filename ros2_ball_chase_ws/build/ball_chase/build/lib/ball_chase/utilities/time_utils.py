#!/usr/bin/env python3

"""
Time Handling Utilities
----------------------
This module provides consistent time handling utilities for all nodes in the
tennis ball tracking system, ensuring uniform timestamp processing.
"""

import time
from typing import Union, Tuple, Optional
import rclpy
from rclpy.time import Time
from builtin_interfaces.msg import Time as TimeMsg


class TimeUtils:
    """
    Utilities for consistent handling of time across all nodes.
    
    This class provides methods for:
    - Converting between different time formats
    - Calculating time differences safely
    - Handling edge cases like backward time jumps
    """
    
    @staticmethod
    def ros_time_to_float(ros_time: TimeMsg) -> float:
        """
        Convert a ROS Time message to a float seconds value.
        
        Args:
            ros_time: ROS Time message
            
        Returns:
            Float representing seconds since epoch
        """
        if not isinstance(ros_time, TimeMsg):
            raise TypeError("Expected ROS Time message, got {}".format(type(ros_time)))
        return ros_time.sec + ros_time.nanosec * 1e-9
    
    @staticmethod
    def float_to_ros_time(timestamp: float) -> TimeMsg:
        """
        Convert a float seconds value to a ROS Time message.
        
        Args:
            timestamp: Float seconds value
            
        Returns:
            ROS Time message
        """
        if not isinstance(timestamp, (int, float)):
            raise TypeError("Expected float or int timestamp, got {}".format(type(timestamp)))
            
        sec = int(timestamp)
        nanosec = int((timestamp - sec) * 1e9)
        
        msg = TimeMsg()
        msg.sec = sec
        msg.nanosec = nanosec
        return msg
    
    @staticmethod
    def get_time_difference(newer: Union[TimeMsg, float], older: Union[TimeMsg, float]) -> float:
        """
        Calculate the time difference between two timestamps, handling type conversions.
        
        Args:
            newer: More recent timestamp (ROS Time or float)
            older: Older timestamp (ROS Time or float)
            
        Returns:
            Time difference in seconds (positive if newer > older)
        """
        if newer is None or older is None:
            raise ValueError("Cannot calculate time difference with None values")
            
        # Convert to float seconds if needed
        newer_float = newer if isinstance(newer, float) else TimeUtils.ros_time_to_float(newer)
        older_float = older if isinstance(older, float) else TimeUtils.ros_time_to_float(older)
        
        return newer_float - older_float
    
    @staticmethod
    def handle_time_jump(dt: float, default_dt: float = 0.033) -> float:
        """
        Handle common time jump issues in a consistent way.
        
        Args:
            dt: Time difference that may contain jumps
            default_dt: Default time step to use if dt is invalid
            
        Returns:
            Sanitized time difference suitable for further processing
        """
        if not isinstance(dt, (int, float)):
            return default_dt
            
        # Handle backward time jumps
        if dt < -0.1:  # Significant backward jump
            return default_dt  # Use default instead
        elif dt < 0:   # Small backward jump
            return 0.001  # Use small positive value
            
        # Handle excessive forward jumps
        if dt > 1.0:
            return default_dt  # Use reasonable default
            
        # Normal time progression
        return dt
    
    @staticmethod
    def is_timestamp_valid(timestamp: Union[TimeMsg, float, None]) -> bool:
        """
        Check if a timestamp is valid.
        
        Args:
            timestamp: The timestamp to check (ROS Time, float, or None)
            
        Returns:
            True if the timestamp is valid, False otherwise
        """
        if timestamp is None:
            return False
            
        if isinstance(timestamp, float):
            return timestamp > 0.0
        
        # For ROS Time messages - make sure we have either seconds or nanoseconds
        try:
            return (timestamp.sec > 0 or timestamp.nanosec > 0)
        except AttributeError:
            # If the timestamp doesn't have sec or nanosec attributes
            return False
    
    @staticmethod
    def now_as_float() -> float:
        """
        Get the current time as a float.
        
        Returns:
            Current time as float seconds since epoch
        """
        return time.time()
    
    @staticmethod
    def now_as_ros_time() -> TimeMsg:
        """
        Get the current time as a ROS Time message.
        
        Returns:
            Current time as ROS Time message
        """
        return TimeUtils.float_to_ros_time(time.time())
    
    @staticmethod
    def find_closest_timestamp(target_time: Union[TimeMsg, float], 
                              timestamps: list, 
                              max_difference: float = 0.1) -> Tuple[int, float]:
        """
        Find the timestamp closest to a target time.
        
        Args:
            target_time: The target time to match
            timestamps: List of timestamps to search
            max_difference: Maximum allowed time difference
            
        Returns:
            Tuple of (index of closest timestamp, time difference)
            Returns (-1, float('inf')) if no timestamp is close enough
        """
        if target_time is None or not timestamps:
            return -1, float('inf')
            
        target_float = target_time if isinstance(target_time, float) else TimeUtils.ros_time_to_float(target_time)
        
        closest_idx = -1
        min_diff = float('inf')
        
        for i, ts in enumerate(timestamps):
            if ts is None:
                continue
                
            ts_float = ts if isinstance(ts, float) else TimeUtils.ros_time_to_float(ts)
            diff = abs(ts_float - target_float)
            
            if diff < min_diff:
                min_diff = diff
                closest_idx = i
        
        # Check if the closest timestamp is close enough
        if min_diff > max_difference:
            return -1, float('inf')
            
        return closest_idx, min_diff
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Format a duration in seconds to a human-readable string.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Human-readable duration string
        """
        if not isinstance(seconds, (int, float)):
            return "invalid"
            
        if seconds < 0.001:
            return f"{seconds*1e6:.1f}μs"
        elif seconds < 1.0:
            return f"{seconds*1000:.1f}ms"
        elif seconds < 60.0:
            return f"{seconds:.2f}s"
        else:
            minutes = int(seconds / 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
    
    @staticmethod
    def is_timestamp_stale(timestamp: Union[TimeMsg, float], 
                         max_age: float,
                         reference_time: Optional[float] = None) -> bool:
        """
        Check if a timestamp is older than a maximum age.
        
        Args:
            timestamp: The timestamp to check
            max_age: Maximum allowed age in seconds
            reference_time: Reference time (defaults to current time)
            
        Returns:
            True if the timestamp is stale, False otherwise
        """
        if not TimeUtils.is_timestamp_valid(timestamp):
            return True
            
        ts_float = timestamp if isinstance(timestamp, float) else TimeUtils.ros_time_to_float(timestamp)
        ref_time = reference_time if reference_time is not None else TimeUtils.now_as_float()
        
        age = ref_time - ts_float
        return age > max_age
