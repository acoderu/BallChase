#!/usr/bin/env python3

"""
SYNCHRONIZED KALMAN FILTER FUSION NODE
-------------------------------------

This module shows how to combine data from multiple sensors using a Kalman filter,
while properly handling the synchronization of measurements.

WHAT IS SENSOR FUSION?
Sensor fusion is the process of combining data from multiple sensors to get a more
accurate understanding of the world. Each sensor has strengths and weaknesses:

- HSV Camera: Good at finding colored objects, but only gives 2D position
- YOLO Camera: Good at recognizing objects, but slower than HSV processing
- LIDAR: Gives accurate distance measurements, but might miss some objects
- Depth Camera: Combines camera and depth information, but may have limited range

By combining data from all these sensors, we can get a more complete and accurate
picture of where objects are.

CONCEPT: Think of this like using multiple witnesses to describe an event. Each 
witness (sensor) sees things from a different perspective and has different strengths
in what they can observe. By combining all their accounts, we get a more complete
picture of what happened.
"""

import rospy
from sensor_msgs.msg import PointCloud2
from ball_tracking_msgs.msg import BallDetection
from ball_tracking.sensor_sync_buffer import SimpleSensorBuffer

class SimpleKalmanFusionNode:
    """
    A node that uses a Kalman filter to combine data from multiple sensors.
    
    The Kalman filter is a mathematical tool that can:
    1. Combine measurements from different sensors
    2. Track objects over time (including their velocity)
    3. Handle noisy measurements
    4. Predict where objects will be even when sensors temporarily don't see them
    
    This implementation focuses on properly handling the timing and synchronization
    of measurements from different sensors.
    """
    
    def __init__(self):
        """Set up the Kalman filter fusion node with all its components."""
        rospy.loginfo("=== STARTING KALMAN FILTER FUSION NODE ===")
        rospy.loginfo("This node will combine data from multiple sensors to track the tennis ball")
        
        # STEP 1: Create the buffer for synchronizing our sensor measurements
        # -----------------------------------------------------------------
        # EXPLANATION: Before we can fuse data from different sensors, we need
        # to make sure we're combining measurements taken at approximately the
        # same time. The SimpleSensorBuffer helps us match measurements based
        # on their timestamps.
        # -----------------------------------------------------------------
        rospy.loginfo("Creating sensor synchronization buffer")
        self.sensor_buffer = SimpleSensorBuffer(
            sensor_names=['hsv', 'yolo', 'lidar', 'depth'],  # Our sensor sources
            buffer_size=30,      # Keep 30 recent measurements per sensor
            max_time_diff=0.1    # Allow 100ms time difference between "synchronized" measurements
        )
        
        # STEP 2: Set up subscribers for all our sensor inputs
        # -----------------------------------------------------------------
        # EXPLANATION: For each sensor, we create a subscriber that listens
        # for messages on a specific topic. When a message arrives, we call
        # the appropriate callback function, which adds the measurement to
        # our synchronization buffer.
        # -----------------------------------------------------------------
        rospy.loginfo("Setting up subscribers for all sensors")
        rospy.Subscriber('/hsv/ball_detection', BallDetection, 
                         self.hsv_callback)
        rospy.Subscriber('/yolo/ball_detection', BallDetection,
                         self.yolo_callback)
        rospy.Subscriber('/lidar/ball_detection', BallDetection,
                         self.lidar_callback)
        rospy.Subscriber('/depth/ball_detection', BallDetection,
                         self.depth_callback)
        
        # STEP 3: Create a publisher for our fused output
        # We'll publish our best estimate of the ball's position after fusion
        self.fused_pub = rospy.Publisher('/fused_ball', BallDetection, queue_size=10)
        
        # STEP 4: Set up a timer to regularly update our Kalman filter
        # -----------------------------------------------------------------
        # EXPLANATION: Instead of updating our filter every time we get a new
        # measurement (which could happen at irregular intervals), we update
        # at a fixed rate of 20Hz (every 0.05 seconds). This gives us more
        # consistent behavior.
        # -----------------------------------------------------------------
        rospy.loginfo("Starting Kalman filter update timer (20Hz)")
        self.update_timer = rospy.Timer(rospy.Duration(0.05), self.filter_update)
        
        # STEP 5: Initialize the Kalman filter
        self.initialize_kalman_filter()
        
        rospy.loginfo("Kalman fusion node initialization complete!")
        rospy.loginfo("Waiting for sensor data...")
    
    def initialize_kalman_filter(self):
        """
        Set up the initial state of our Kalman filter.
        
        In a real implementation, this would initialize the state vector
        and covariance matrices of the Kalman filter. For this example,
        we use a simplified version.
        """
        # For this simplified example, we just track position and velocity
        self.position = [0.0, 0.0, 0.0]  # x, y, z position in meters
        self.velocity = [0.0, 0.0, 0.0]  # x, y, z velocity in meters/second
        self.initialized = False  # Flag to know if we've received our first measurement
        
        rospy.loginfo("Kalman filter initialized with default values")
        rospy.loginfo("Waiting for first valid measurements to start tracking")
    
    def hsv_callback(self, msg):
        """
        Process a new measurement from the HSV ball detector
        
        Parameters:
            msg: The BallDetection message from the HSV detector
        """
        # Add this measurement to our buffer with the 'hsv' label
        self.sensor_buffer.add_measurement('hsv', msg, msg.header.stamp)
        rospy.logdebug(f"Received HSV measurement at time {msg.header.stamp.to_sec():.3f}s")
    
    def yolo_callback(self, msg):
        """
        Process a new measurement from the YOLO object detector
        
        Parameters:
            msg: The BallDetection message from the YOLO detector
        """
        # Add this measurement to our buffer with the 'yolo' label
        self.sensor_buffer.add_measurement('yolo', msg, msg.header.stamp)
        rospy.logdebug(f"Received YOLO measurement at time {msg.header.stamp.to_sec():.3f}s")
    
    def lidar_callback(self, msg):
        """
        Process a new measurement from the LIDAR ball detector
        
        Parameters:
            msg: The BallDetection message from the LIDAR detector
        """
        # Add this measurement to our buffer with the 'lidar' label
        self.sensor_buffer.add_measurement('lidar', msg, msg.header.stamp)
        rospy.logdebug(f"Received LIDAR measurement at time {msg.header.stamp.to_sec():.3f}s")
    
    def depth_callback(self, msg):
        """
        Process a new measurement from the depth camera ball detector
        
        Parameters:
            msg: The BallDetection message from the depth camera detector
        """
        # Add this measurement to our buffer with the 'depth' label
        self.sensor_buffer.add_measurement('depth', msg, msg.header.stamp)
        rospy.logdebug(f"Received depth camera measurement at time {msg.header.stamp.to_sec():.3f}s")
    
    def filter_update(self, event):
        """
        Update the Kalman filter with synchronized measurements
        
        This function is called regularly by our timer (20 times per second)
        
        Parameters:
            event: Timer event (not used, but required by ROS timer callback)
        """
        # STEP 1: Try to get synchronized measurements from our buffer
        sync_data = self.sensor_buffer.find_synchronized_data()
        
        # If we couldn't find synchronized data, skip this update
        if not sync_data:
            rospy.logdebug("No synchronized data available for this update")
            return
        
        # STEP 2: Log information about what sensors we have data from
        available_sensors = list(sync_data.keys())
        rospy.loginfo(f"Found synchronized data from these sensors: {available_sensors}")
        
        # STEP 3: If we have data from at least two sensors, perform fusion
        # ----------------------------------------------------------------
        # EXPLANATION: The whole point of sensor fusion is to combine data
        # from multiple sensors. If we only have data from one sensor,
        # there's nothing to fuse!
        # ----------------------------------------------------------------
        if len(sync_data) >= 2:
            # STEP 3.1: Calculate the average timestamp of our measurements
            # This helps us understand how synchronized our data really is
            total_time = 0.0
            count = 0
            
            for sensor_name, data in sync_data.items():
                if hasattr(data, 'header'):  # Check if this data has a header
                    total_time += data.header.stamp.to_sec()
                    count += 1
            
            if count > 0:
                # Calculate the average timestamp
                avg_time = total_time / count
                max_diff = 0.0
                
                # Find the maximum time difference from the average
                for sensor_name, data in sync_data.items():
                    if hasattr(data, 'header'):
                        diff = abs(data.header.stamp.to_sec() - avg_time)
                        max_diff = max(max_diff, diff)
                
                rospy.loginfo(f"Measurements are within {max_diff:.3f}s of each other")
                
                # STEP 3.2: Update the Kalman filter with our synchronized data
                # -------------------------------------------------------------
                # EXPLANATION: This is where the actual Kalman filter math would go.
                # In a complete implementation, we would:
                # 1. Predict the new state based on the previous state
                # 2. Extract the measurements from each sensor
                # 3. Update the state with these measurements
                # 4. Calculate an updated estimate of position and velocity
                # -------------------------------------------------------------
                
                # For this example, we'll just log that we would update the filter
                rospy.loginfo("Would update Kalman filter with synchronized data")
                rospy.loginfo(f"Average timestamp: {avg_time:.3f}s")

                # In a real implementation, we would update our position and velocity
                # based on the Kalman filter estimation
                
                # STEP 3.3: Publish our fused estimate
                # Create a message with our fused position estimate
                fused_msg = BallDetection()
                # In a real implementation, we would populate this with our
                # Kalman filter's current estimate
                # ... (code to populate fused_msg with results from Kalman filter)
                
                # Publish our fused result
                self.fused_pub.publish(fused_msg)
