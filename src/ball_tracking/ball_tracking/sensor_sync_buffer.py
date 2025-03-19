#!/usr/bin/env python3

"""
MULTI-SENSOR SYNCHRONIZATION BUFFER
----------------------------------

This module provides a buffer system that stores recent measurements from
multiple sensors and helps find measurements that happened at approximately 
the same time.

WHY DO WE NEED THIS?
In a robot with multiple sensors, each sensor might report data at different rates
and with slightly different timing. For example:
- Camera: might send images at 30 frames per second
- LIDAR: might send scans at 10 times per second
- Other sensors: might report at their own rates

To make sense of all this data, we need to match up measurements that were
taken at approximately the same time, even if they arrive at different times.

CONCEPT: Think of this like organizing photos from different cameras that were
taken during the same event. Each camera has its own timestamp, but you want
to group photos that were taken at almost the same moment, even if the clocks
weren't perfectly synchronized.
"""

import rospy
from collections import deque  # A special list-like data structure that's efficient for our needs

class SimpleSensorBuffer:
    """
    A buffer that stores recent measurements from different sensors and
    helps find measurements that were taken at approximately the same time.
    
    This allows us to combine data from different sensors even when they
    don't perfectly align in time.
    """
    
    def __init__(self, sensor_names, buffer_size=20, max_time_diff=0.1):
        """
        Set up a new buffer for synchronizing sensor data
        
        Parameters:
            sensor_names: List of sensor names (e.g., ['camera', 'lidar', 'gps'])
            buffer_size: How many measurements to remember for each sensor
            max_time_diff: Maximum time difference (in seconds) to consider measurements "synchronized"
                          (e.g., 0.1 means within 100 milliseconds of each other)
        """
        # STEP 1: Create a separate buffer (storage container) for each sensor
        # ----------------------------------------------------------------
        # EXPLANATION: We use a "dictionary of deques" to organize our data:
        # - A dictionary allows us to use sensor names as keys
        # - A deque is a special type of list that's efficient for adding and removing items
        # - Each deque has a maximum size (buffer_size) so we don't use too much memory
        # ----------------------------------------------------------------
        self.buffers = {}
        for name in sensor_names:
            # Create an empty buffer with limited size for this sensor
            self.buffers[name] = deque(maxlen=buffer_size)
        
        # Store the maximum allowed time difference for synchronization
        self.max_time_diff = max_time_diff
        
        # Log information about our setup
        rospy.loginfo("==== SENSOR SYNCHRONIZATION BUFFER CREATED ====")
        rospy.loginfo(f"Tracking these sensors: {', '.join(sensor_names)}")
        rospy.loginfo(f"Keeping up to {buffer_size} measurements per sensor")
        rospy.loginfo(f"Will consider measurements synchronized if within {max_time_diff} seconds of each other")
    
    def add_measurement(self, sensor_name, data, timestamp):
        """
        Add a new measurement from a sensor to our buffer
        
        Parameters:
            sensor_name: Which sensor this measurement came from (must be one we're tracking)
            data: The actual measurement data (could be any type)
            timestamp: When the measurement was taken (a rospy.Time object)
        """
        # STEP 1: Check if this is a sensor we're tracking
        if sensor_name not in self.buffers:
            rospy.logwarn(f"Unknown sensor: '{sensor_name}'. We're only tracking: {list(self.buffers.keys())}")
            return
        
        # STEP 2: Create a measurement entry with both data and timestamp
        # We store this as a dictionary for clarity and flexibility
        measurement = {
            'data': data,           # The actual sensor measurement
            'timestamp': timestamp  # When the measurement was taken
        }
        
        # STEP 3: Add the measurement to the appropriate sensor's buffer
        self.buffers[sensor_name].append(measurement)
        
        # STEP 4: For debugging - show how many measurements we have for each sensor
        buffer_sizes = {name: len(buffer) for name, buffer in self.buffers.items()}
        rospy.logdebug(f"Current buffer sizes: {buffer_sizes}")
    
    def find_synchronized_data(self):
        """
        Look through our buffers and find sensor measurements that were taken
        at approximately the same time
        
        Returns:
            Dictionary with sensor names as keys and their data as values,
            or None if synchronized measurements couldn't be found
        """
        # STEP 1: Make sure we have at least some data from each sensor
        # ------------------------------------------------------------
        # EXPLANATION: Before we try to find synchronized data, let's check
        # that we have at least one measurement from each sensor.
        # If any sensor has no data yet, we can't synchronize.
        # ------------------------------------------------------------
        for name, buffer in self.buffers.items():
            if not buffer:  # If this buffer is empty
                rospy.logdebug(f"No data available yet from sensor: {name}")
                return None
                
        # STEP 2: Find the measurement with the most recent timestamp
        # ------------------------------------------------------------
        # EXPLANATION: We need a reference point for synchronization.
        # We'll use the most recent measurement from any sensor.
        # This will be our "anchor" that we try to match other measurements to.
        # ------------------------------------------------------------
        reference_time = None
        reference_sensor = None
        
        for name, buffer in self.buffers.items():
            latest = buffer[-1]  # Get the newest measurement from this sensor
            if reference_time is None or latest['timestamp'] > reference_time:
                reference_time = latest['timestamp']
                reference_sensor = name
        
        rospy.logdebug(f"Using {reference_sensor} as reference with timestamp {reference_time.to_sec():.3f}s")
                
        # STEP 3: Start collecting synchronized data with our reference measurement
        result = {
            reference_sensor: self.buffers[reference_sensor][-1]['data']  # Add reference sensor data
        }
        
        # STEP 4: For each other sensor, find its measurement closest to our reference time
        # ------------------------------------------------------------
        # EXPLANATION: This is the heart of our synchronization algorithm.
        # For each sensor, we search through its recent measurements and
        # find the one that was taken closest in time to our reference.
        # This is similar to finding the closest match in time.
        # ------------------------------------------------------------
        for name, buffer in self.buffers.items():
            # Skip the reference sensor since we already added it
            if name == reference_sensor:
                continue
                
            # STEP 4.1: Find the measurement with timestamp closest to reference_time
            closest_measurement = None
            smallest_time_diff = float('inf')  # Start with infinity (largest possible number)
            
            for measurement in buffer:
                # Calculate the time difference between this measurement and our reference
                time_diff = abs((measurement['timestamp'] - reference_time).to_sec())
                
                # If this is closer than our previous closest match, update
                if time_diff < smallest_time_diff:
                    smallest_time_diff = time_diff
                    closest_measurement = measurement
            
            # STEP 4.2: Check if this measurement is close enough to consider "synchronized"
            # If within our allowed time difference, add it to our result
            if smallest_time_diff <= self.max_time_diff:
                result[name] = closest_measurement['data']
                rospy.logdebug(f"Found matching {name} measurement {smallest_time_diff:.3f}s from reference")
            else:
                # If not close enough, we can't provide synchronized data
                rospy.logwarn(f"No {name} measurement found within {self.max_time_diff}s of reference")
                rospy.logwarn(f"Closest was {smallest_time_diff:.3f}s away - too far!")
                return None  # Return None if any sensor doesn't have a matching measurement
                
        # STEP 5: Return the synchronized measurements if we found matches for all sensors
        return result
