#!/usr/bin/env python3

"""
SENSOR SYNCHRONIZATION MODULE
-----------------------------

This file demonstrates how to synchronize data from multiple sensors that might 
report measurements at different times.

WHY IS THIS IMPORTANT?
In robotics, we often have several sensors (cameras, LIDAR, etc.) that all measure
the world at slightly different times. For example, a camera might take pictures
30 times per second, while a LIDAR might scan the environment 10 times per second.
If we want to combine information from both sensors, we need to match up measurements
that happened at approximately the same time.

CONCEPT: Think of it like a group project where different team members work at 
different speeds. To make decisions, you need to wait until everyone has contributed
their part for the same section.
"""

import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2
from ball_tracking_msgs.msg import BallDetection

class SynchronizedSensorProcessor:
    """
    This class shows how to synchronize camera images and LIDAR scans.
    
    It uses ROS's message_filters to wait for data from both sensors
    that were captured at nearly the same time, then processes them together.
    """
    
    def __init__(self):
        rospy.loginfo("=== SENSOR SYNCHRONIZATION NODE STARTING ===")
        rospy.loginfo("This node demonstrates how to match camera and LIDAR data by timestamp")
        
        # STEP 1: Define subscribers for different sensor topics
        # Instead of using normal subscribers, we use "message_filters.Subscriber"
        # which allows us to synchronize messages from different topics
        rospy.loginfo("Setting up synchronized subscribers for camera and LIDAR")
        image_sub = message_filters.Subscriber('/camera/image_raw', Image)
        lidar_sub = message_filters.Subscriber('/lidar_points', PointCloud2)
        
        # STEP 2: Create a synchronizer
        # This is like a "matchmaker" for messages from different sensors
        # -------------------------------------------------------------
        # EXPLANATION: The ApproximateTimeSynchronizer works like this:
        # 1. It receives messages from both sensors
        # 2. It looks at their timestamps
        # 3. If it finds messages from both sensors with timestamps that are
        #    within 'slop' seconds of each other, it considers them a match
        # 4. It calls our callback function with these matched messages
        # -------------------------------------------------------------
        rospy.loginfo("Creating synchronizer with 100ms (0.1s) time window")
        sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, lidar_sub],
            queue_size=10,  # Store 10 messages while waiting for matches
            slop=0.1        # 100ms time window - a tenth of a second
        )
        
        # STEP 3: Tell the synchronizer what function to call when it finds matching messages
        rospy.loginfo("Registering callback function for synchronized data")
        sync.registerCallback(self.synchronized_callback)
        
        # STEP 4: Create a publisher for our results
        self.fusion_pub = rospy.Publisher('/fused_detections', BallDetection, queue_size=10)
        
        rospy.loginfo("Synchronized sensor processor initialized and ready!")
        rospy.loginfo("Waiting for matching camera and LIDAR messages...")
    
    def synchronized_callback(self, image_msg, lidar_msg):
        """
        This function is called only when we receive camera and LIDAR data
        captured at nearly the same time.
        
        Parameters:
            image_msg: The camera image message
            lidar_msg: The LIDAR point cloud message
        """
        
        # Calculate time difference between the messages
        # The .to_sec() converts the ROS time to seconds (a regular float number)
        camera_time = image_msg.header.stamp.to_sec()
        lidar_time = lidar_msg.header.stamp.to_sec()
        time_diff = abs(camera_time - lidar_time)
        
        # Print timestamp info to verify our synchronization is working
        rospy.loginfo("===============================================")
        rospy.loginfo("EUREKA! Received synchronized sensor data!")
        rospy.loginfo(f"  Camera timestamp: {camera_time:.3f} seconds")
        rospy.loginfo(f"  LIDAR timestamp:  {lidar_time:.3f} seconds")
        rospy.loginfo(f"  Time difference:  {time_diff:.3f} seconds")
        
        # Now that we have synchronized data, we could:
        # 1. Detect objects in the camera image
        # 2. Find those same objects in the LIDAR data
        # 3. Get 3D position information from LIDAR
        # 4. Combine information from both sensors
        
        # For now, we'll just log that we received synchronized data
        # In a complete system, we would process the data here
        rospy.loginfo("Now we could process these synchronized measurements together!")
        rospy.loginfo("===============================================")
