#!/usr/bin/env python3

"""
Time Handling Utilities
----------------------
This module provides consistent time handling utilities for all nodes in the
tennis ball tracking system, ensuring uniform timestamp processing.

WHY DO WE NEED TIME UTILITIES? ⏰
==============================

┌─────────────────────────────────────────────────────────────────────────┐
│                 TIME CHALLENGES IN ROBOTICS                              │
└─────────────────────────────────────────────────────────────────────────┘

    In robotics, dealing with time correctly is CRITICAL:

    🔹 Different components use different time formats:
       ┌───────────────────┐   ┌────────────────────┐   ┌────────────────┐
       │ ROS TIME          │   │ UNIX TIME          │   │ OTHER FORMATS  │
       │ sec: 1683142517   │   │ 1683142517.423     │   │ "15:45:17.423" │
       │ nanosec: 423000000│   │ (float seconds)    │   │ (string)       │
       └───────────────────┘   └────────────────────┘   └────────────────┘
                        
    🔹 We must be able to compare times and calculate differences:
       
       "Is this camera frame older than this sensor reading?"
       "How much time has passed between these two events?"
       
    🔹 Time can sometimes jump backward due to system issues:
    
        TIME: 100.0s → 100.1s → 99.8s → 100.2s
                                ↑ Backward jump!

┌─────────────────────────────────────────────────────────────────────────┐
│                   HOW TIME UTILS HELPS                                   │
└─────────────────────────────────────────────────────────────────────────┘

    The TimeUtils class provides solutions for:
    
                            ┌───────────────────────────┐
                            │     TIME CONVERSIONS      │
                            └───────────────┬───────────┘
                                            │
                              ┌─────────────┴────────────┐
                              │                          │
     ┌────────────────────┐   │   ┌────────────────────┐ │  ┌────────────────────┐
     │  ROS TIME          │◀──┼──▶│  FLOAT TIME        │◀┼─▶│ STRING TIME        │
     │  sec: 1683142517   │   │   │  1683142517.423    │ │  │ "15:45:17.423"     │
     │  nanosec: 423000000│   │   │                    │ │  │                    │
     └────────────────────┘   │   └────────────────────┘ │  └────────────────────┘
                              │                          │
                              └──────────────────────────┘

                            ┌───────────────────────────┐
                            │     TIME SAFETY FEATURES  │
                            └───────────────┬───────────┘
                                            │
                            ┌───────────────┴───────────┐
                            │                           │
               ┌────────────┴─────────┐   ┌─────────────┴────────────┐
               │                      │   │                          │
     ┌─────────┴─────────┐   ┌────────┴───┴───────┐    ┌─────────────┴────────┐
     │  DETECT TIME JUMPS │   │ HANDLE BAD VALUES  │    │ FIND CLOSEST TIMES  │
     │                    │   │                    │    │                      │
     │  ✓ Backward jumps  │   │  ✓ Zero/negative   │    │  ✓ Match timestamps │
     │  ✓ Clock resets    │   │  ✓ Very old times  │    │  ✓ Find best match  │
     └────────────────────┘   └────────────────────┘    └──────────────────────┘

EVERYDAY ANALOGY:
===============

Think of TimeUtils like a universal translator for different clock systems.
Imagine your friend has a 24-hour military clock, another uses a 12-hour AM/PM 
clock, and you need to coordinate a meeting. This class helps convert between 
formats, check if times are valid, and handle problems like someone's watch 
being set incorrectly.

It's the time expert that makes sure everyone is talking about the same moment!
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
    
    IMAGINE THIS: 🕰️
    ---------------
    Think of TimeUtils as a highly skilled translator who helps people speaking
    different "time languages" understand each other. Just like languages have
    different words and grammar, different parts of a robot system use different
    ways to represent time.
    
    HOW IT WORKS:
    ------------
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                         TIME CONVERSIONS                             │
    └─────────────────────────────────────────────────────────────────────┘
                                   │
           ┌─────────────────────┬─┴─┬─────────────────────┐
           │                     │   │                     │
           ▼                     ▼   ▼                     ▼
    ┌─────────────┐      ┌─────────────┐           ┌─────────────┐
    │  ROS TIME   │      │   FLOAT     │           │   CHECK     │
    │ CONVERSIONS │      │   TIME      │           │   AND FIX   │
    └──────┬──────┘      └──────┬──────┘           └──────┬──────┘
           │                    │                         │
           ▼                    ▼                         ▼
    ┌─────────────┐      ┌─────────────┐           ┌─────────────┐
    │ros_time_to_ │      │get_time_    │           │handle_time_ │
    │float()      │      │difference() │           │jump()       │
    └─────────────┘      └─────────────┘           └─────────────┘
    ┌─────────────┐      ┌─────────────┐           ┌─────────────┐
    │float_to_ros_│      │find_closest_│           │is_timestamp_│
    │time()       │      │timestamp()  │           │valid()      │
    └─────────────┘      └─────────────┘           └─────────────┘
    
    EVERYDAY ANALOGY:
    ---------------
    In our daily lives, we have similar time conversion challenges:
    
    1. CONVERSION PROBLEMS:
       - Converting between 24-hour and 12-hour time (3:30 PM vs. 15:30)
       - Converting between time zones (8:00 AM EST vs. 5:00 AM PST)
       - Converting between date formats (MM/DD/YY vs. DD/MM/YY)
    
    2. TIME DIFFERENCE PROBLEMS:
       - "If one flight lands at 2:45 PM and another at 3:20 PM, how long
         do I have between them?" (35 minutes)
       - "How long ago did I send that message?" (calculating elapsed time)
    
    3. TIME VALIDATION PROBLEMS:
       - Computer clock suddenly jumps backwards (daylight savings time)
       - Setting a meeting for "February 30th" (invalid date)
       - Computer showing year 1970 after a reboot (clock reset)
    
    TimeUtils solves all these types of problems for our robot system,
    making sure all parts of the robot correctly understand when things happen.
    """
    
    @staticmethod
    def ros_time_to_float(ros_time: TimeMsg) -> float:
        """
        Convert a ROS Time message to a float seconds value.
        
        IMAGINE THIS: 🔄
        ---------------
        Think of this like converting currency. Just as you might convert euros 
        to dollars, this method converts ROS's way of representing time into a 
        simpler "universal" format (floating-point seconds).
        
        ROS Time looks like:   →  Float Time looks like:
        sec: 1234567890           1234567890.123
        nanosec: 123000000
        
        HOW IT WORKS:
        ------------
        This function:
        1. Takes the "seconds" part as is
        2. Converts "nanoseconds" to a fraction of a second (divides by a billion)
        3. Adds them together to get a single number
        
        EXAMPLE:
        -------
        Input ROS Time: {sec: 10, nanosec: 500000000}
        Float result: 10.5 seconds
        
        EVERYDAY ANALOGY:
        ---------------
        It's like converting feet and inches to just inches:
        5 feet and 6 inches → 66 inches
        
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
        
        IMAGINE THIS: 🧙‍♂️
        ---------------
        Think of this method as a time detective that solves mysteries when the
        clock behaves strangely. Sometimes computers hiccup and their clocks jump
        backward (like going from 10:00 to 9:55) or jump too far forward
        (from 10:00 to 11:30) when only a few seconds should have passed.
        
        HOW IT WORKS:
        ------------
        
        ┌─────────────────────────────────────────────────────────┐
        │                 TIME JUMP HANDLER                       │
        └───────────────────────────┬─────────────────────────────┘
                                    │
                      ┌─────────────┴──────────────┐
                      │                            │
                      ▼                            ▼
        ┌──────────────────────┐       ┌──────────────────────┐
        │  BACKWARD JUMP?      │       │  FORWARD JUMP?       │
        │                      │       │                      │
        │  dt < 0             │       │  dt > 1.0            │
        │  "Time went backward"│       │  "Time jumped way    │
        │                      │       │   too far forward"   │
        └──────────┬───────────┘       └──────────┬───────────┘
                   │                               │
                   ▼                               ▼
        ┌──────────────────────┐       ┌──────────────────────┐
        │  USE SAFE VALUE      │       │  USE SAFE VALUE      │
        │                      │       │                      │
        │  Return default_dt   │       │  Return default_dt   │
        │  (typically 0.033s)  │       │  (typically 0.033s)  │
        └──────────────────────┘       └──────────────────────┘
        
        EVERYDAY ANALOGY:
        ---------------
        It's like when your friend says, "I'll meet you in 5 minutes," but then
        later claims, "I said that 2 minutes from now!" Their clock is clearly
        wrong. Instead of accepting their bizarre time jump, you decide to use
        your watch's reasonable time (like our default_dt) to keep things on track.
        
        Think of it as a "reality check" for time differences.
        
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
        
        IMAGINE THIS: 🔍
        ---------------
        Think of this like finding the person closest to your age at a party.
        You're 25 years old and want to find someone similar in age. You check 
        everyone's age: 18, 32, 24, 55, 40. The person who is 24 is closest to 
        your age (just 1 year difference).
        
        In robotics, we do this with timestamps to match up sensor readings
        that happened at nearly the same time.
        
        HOW IT WORKS:
        ------------
                      ┌───────────────────────────┐
                      │   TARGET TIME: 10.5s      │
                      └───────────────┬───────────┘
                                      │
                                      ▼
        ┌─────────────────────────────────────────────────────┐
        │                                                     │
        │  AVAILABLE TIMESTAMPS:                              │
        │                                                     │
        │  [ 10.1s,   10.3s,   10.7s,   11.2s,   12.0s ]     │
        │     |        |        |        |        |          │
        │     |        |        |        |        |          │
        │     |        |        |        |        |          │
        │     ▼        ▼        ▼        ▼        ▼          │
        │                                                     │
        │  DIFFERENCES:                                       │
        │                                                     │
        │  [ 0.4s,    0.2s,    0.2s,    0.7s,    1.5s ]      │
        │                                                     │
        └────────────────────────┬────────────────────────────┘
                                 │
                                 ▼
                      ┌───────────────────────────┐
                      │   CLOSEST: 10.3s or 10.7s │
                      │   (Both 0.2s away)        │
                      │   Take the first one!     │
                      └───────────────────────────┘
        
        EVERYDAY ANALOGY:
        ---------------
        It's like when you miss your favorite TV show that airs at 8:00 PM, but
        there are reruns at 7:30 PM, 8:15 PM, 9:00 PM, and 11:00 PM. You want
        to watch the rerun closest to the original time. This method would tell
        you that the 8:15 PM showing is closest to your target of 8:00 PM.
        
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
