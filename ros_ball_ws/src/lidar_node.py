#!/usr/bin/env python3

"""
Tennis Ball Tracking Robot - LIDAR Detection Node
=================================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving tennis ball.
The system uses multiple sensing modalities for robust detection:
- YOLO neural network detection (subscribes to '/tennis_ball/yolo/position')
- HSV color-based detection (subscribes to '/tennis_ball/hsv/position')
- LIDAR for depth sensing (this node)
- Depth camera for additional depth information

Data Pipeline:
-------------
1. Camera images are processed by:
   - YOLO detection node (yolo_ball_node.py) publishing to '/tennis_ball/yolo/position'
   - HSV color detector (hsv_ball_node.py) publishing to '/tennis_ball/hsv/position'

2. This LIDAR node:
   - Subscribes to raw LIDAR scans from '/scan_raw'
   - Subscribes to both YOLO and HSV detection results
   - Correlates 2D camera detections with 3D LIDAR point data
   - Publishes 3D positions to '/tennis_ball/lidar/position'

3. Data is then passed to:
   - Sensor fusion node to combine all detection methods
   - State management node for decision-making
   - PID controller for motor control

This Node's Purpose:
------------------
This LIDAR node adds depth information to the tennis ball tracking by:
- Processing 2D laser scan data to find circular patterns matching a tennis ball
- Confirming these patterns when triggered by camera-based detections
- Publishing the ball's 3D position in the robot's coordinate frame
- Providing visualization markers for debugging in RViz
"""

import rclpy
from rclpy.node import Node
import time
import json
import math
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import TransformStamped  
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster  

# Configuration constants
TENNIS_BALL_CONFIG = {
    "radius": 0.033,         # Tennis ball radius in meters
    "height": -0.20,         # Expected height of ball center relative to LIDAR
    "max_distance": 0.1,     # Maximum distance for clustering points
    "min_points": 10,        # Minimum points to consider a valid cluster
    "quality_threshold": {   # Thresholds for circle quality assessment
        "low": 0.5,
        "medium": 0.7,
        "high": 0.9
    },
    "detection_samples": 30  # Number of random starting points for clustering
}

# Topic configuration (ensures consistency with other nodes)
TOPICS = {
    "input": {
        "lidar_scan": "/scan_raw",
        "yolo_detection": "/tennis_ball/yolo/position",
        "hsv_detection": "/tennis_ball/hsv/position"
    },
    "output": {
        "ball_position": "/tennis_ball/lidar/position",
        "visualization": "/tennis_ball/lidar/visualization",
        "diagnostics": "/tennis_ball/lidar/diagnostics"
    }
}


class TennisBallLidarDetector(Node):
    """
    A ROS2 node to detect tennis balls using a 2D laser scanner.
    
    This node correlates LIDAR data with camera detections to provide 3D position 
    information for detected tennis balls, which is essential for accurate tracking
    and following behavior in the robot.
    
    Subscribed Topics:
    - LaserScan from 2D LIDAR ({TOPICS["input"]["lidar_scan"]})
    - PointStamped from YOLO detection ({TOPICS["input"]["yolo_detection"]})
    - PointStamped from HSV detection ({TOPICS["input"]["hsv_detection"]})
    
    Published Topics:
    - PointStamped with 3D position ({TOPICS["output"]["ball_position"]})
    - MarkerArray for visualization ({TOPICS["output"]["visualization"]})
    - String with diagnostic information ({TOPICS["output"]["diagnostics"]})
    """
    
    def __init__(self):
        """Initialize the tennis ball LIDAR detector node."""
        super().__init__('tennis_ball_lidar_detector')
        
        # Load physical parameters of the tennis ball
        self.ball_radius = TENNIS_BALL_CONFIG["radius"]
        self.ball_height = TENNIS_BALL_CONFIG["height"]
        self.max_distance = TENNIS_BALL_CONFIG["max_distance"]
        self.min_points = TENNIS_BALL_CONFIG["min_points"]
        
        # Initialize state variables
        self.latest_scan = None
        self.scan_timestamp = None
        self.scan_frame_id = None
        self.points_array = None
        
        # Initialize performance metrics
        self._init_performance_tracking()
        
        # Set up subscribers
        self._setup_subscribers()
        
        # Set up publishers
        self._setup_publishers()
        
        # Timer for publishing status
        self.status_timer = self.create_timer(3.0, self.publish_status)
        
        # Initialize the TF Broadcaster for publishing coordinate transforms
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_transform_log = 0.0  # Track when we last logged transform info
        
        # Set up a timer for publishing the transform regularly
        # Publishing at 10Hz ensures the transform is always available
        self.transform_timer = self.create_timer(0.1, self.publish_transform)
        
        # Log that we're publishing the calibrated transform
        self.get_logger().info("Publishing calibrated LIDAR-to-camera transform (from calibration)")
        self.get_logger().info("Transform: [-0.326256, 0.210052, 0.504021], Quaternion: [-0.091584, 0.663308, 0.725666, 0.158248]")

        self.get_logger().info("LIDAR: Tennis ball detector initialized and ready")
        self.get_logger().info(f"LIDAR: Listening for LaserScan on {TOPICS['input']['lidar_scan']}")
        self.get_logger().info(f"LIDAR: Listening for YOLO detections on {TOPICS['input']['yolo_detection']}")
        self.get_logger().info(f"LIDAR: Listening for HSV detections on {TOPICS['input']['hsv_detection']}")
    
    def _init_performance_tracking(self):
        """Initialize tracking variables for performance monitoring."""
        # Runtime statistics
        self.start_time = time.time()
        self.processed_scans = 0
        self.successful_detections = 0
        self.detection_times = []
        
        # Detection source statistics
        self.yolo_detections = 0
        self.hsv_detections = 0
    
    def _setup_subscribers(self):
        """Set up all subscribers for this node."""
        # Subscribe to LIDAR scan data
        self.scan_subscription = self.create_subscription(
            LaserScan,
            TOPICS["input"]["lidar_scan"],
            self.scan_callback,
            10
        )
        
        # Subscribe to YOLO detections
        self.yolo_subscription = self.create_subscription(
            PointStamped,
            TOPICS["input"]["yolo_detection"],
            self.yolo_callback,
            10
        )
        
        # Subscribe to HSV detections
        self.hsv_subscription = self.create_subscription(
            PointStamped,
            TOPICS["input"]["hsv_detection"],
            self.hsv_callback,
            10
        )
    
    def _setup_publishers(self):
        """Set up all publishers for this node."""
        # Publisher for 3D tennis ball position
        self.position_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["ball_position"],
            10
        )
        
        # Publisher for visualization markers (for RViz)
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            TOPICS["output"]["visualization"],
            10
        )
        
        # Publisher for diagnostics information
        self.diagnostics_publisher = self.create_publisher(
            String,
            TOPICS["output"]["diagnostics"],
            10
        )
    
    def yolo_callback(self, msg):
        """
        Handle ball detections from the YOLO neural network.
        
        Args:
            msg (PointStamped): The detected ball position from YOLO
        """
        self.yolo_detections += 1
        self.camera_detection_callback(msg, "YOLO")
    
    def hsv_callback(self, msg):
        """
        Handle ball detections from the HSV color detector.
        
        Args:
            msg (PointStamped): The detected ball position from HSV detector
        """
        self.hsv_detections += 1
        self.camera_detection_callback(msg, "HSV")
    
    def scan_callback(self, msg):
        """
        Process LaserScan messages from the LIDAR.
        
        Extracts point cloud data from 2D laser scan by converting
        polar coordinates to Cartesian coordinates.
        
        Args:
            msg (LaserScan): The laser scan message
        """
        self.start_time = time.time()  # Reset timer for this scan
        
        # Store the scan metadata - preserve original frame_id for proper transformation
        self.latest_scan = msg
        self.scan_timestamp = msg.header.stamp  # Use original timestamp, not current time
        self.scan_frame_id = "lidar_frame"  # Always use our consistent frame ID
        
        try:
            # Extract basic scan parameters
            angle_min = msg.angle_min
            angle_increment = msg.angle_increment
            ranges = np.array(msg.ranges)
            
            # Filter out invalid measurements (inf, NaN)
            valid_indices = np.isfinite(ranges)
            valid_ranges = ranges[valid_indices]
            
            # Skip processing if no valid ranges
            if len(valid_ranges) == 0:
                self.get_logger().warn("LIDAR: No valid range measurements in scan")
                self.points_array = None
                return
            
            # Convert polar coordinates to Cartesian coordinates
            angles = angle_min + angle_increment * np.arange(len(ranges))[valid_indices]
            x = valid_ranges * np.cos(angles)  # x = r * cos(θ)
            y = valid_ranges * np.sin(angles)  # y = r * sin(θ)
            
            # Add Z coordinate (assumes LIDAR is parallel to ground)
            z = np.full_like(x, self.ball_height)
            
            # Stack coordinates to create points array [x, y, z]
            self.points_array = np.column_stack((x, y, z))
            
            # Update statistics
            self.processed_scans += 1
            
            # Log scan information (debug level to avoid flooding logs)
            if self.processed_scans % 20 == 0:  # Log every 20th scan
                self.get_logger().debug(
                    f"LIDAR: Processed scan #{self.processed_scans} with "
                    f"{len(self.points_array)} valid points"
                )
            
        except Exception as e:
            self.get_logger().error(f"LIDAR: Error processing scan: {str(e)}")
            self.points_array = None
    
    def camera_detection_callback(self, msg, source):
        """
        Process ball detections from the camera and find matching points in LIDAR data.
        
        This method is triggered whenever one of the camera-based detectors
        (YOLO or HSV) reports a tennis ball detection. It attempts to correlate
        this 2D detection with 3D LIDAR point cloud data.
        
        Args:
            msg (PointStamped): 2D position (x,y) of ball detected by camera
            source (str): Which detector triggered this detection ("YOLO" or "HSV")
        """
        detection_start_time = time.time()
        
        # Check if we have scan data
        if self.latest_scan is None or self.points_array is None or len(self.points_array) == 0:
            self.get_logger().info("LIDAR: Waiting for scan data...")
            return
        
        try:
            # Extract the camera's detected position and confidence
            x_2d = msg.point.x
            y_2d = msg.point.y
            confidence = msg.point.z  # Usually contains confidence value
            
            self.get_logger().info(
                f"{source}: Ball detected at pixel ({x_2d:.1f}, {y_2d:.1f}) "
                f"with confidence {confidence:.2f}"
            )
            
            # Find tennis ball patterns in LIDAR data
            ball_results = self.find_tennis_balls(source)
            
            # Process the best detected ball (if any)
            if ball_results:
                # Get the best match (first in the list, already sorted by quality)
                best_match = ball_results[0]
                center, cluster_size, circle_quality = best_match
                
                # IMPORTANT: Use original timestamp for synchronization
                # Publish the ball position with the camera detection's timestamp
                self.publish_ball_position(center, cluster_size, circle_quality, source, msg.header.stamp)
            else:
                self.get_logger().info(f"LIDAR: No matching ball found for {source} detection")
            
        except Exception as e:
            self.get_logger().error(f"LIDAR: Error processing {source} detection: {str(e)}")
        
        # Log processing time for this detection
        processing_time = (time.time() - detection_start_time) * 1000  # in ms
        self.detection_times.append(processing_time)
        # Keep only the last 100 detection times
        if len(self.detection_times) > 100:
            self.detection_times.pop(0)
            
        self.get_logger().debug(f"LIDAR: {source} processing took {processing_time:.2f}ms")
    
    def find_tennis_balls(self, trigger_source):
        """
        Search for tennis balls in the latest scan data using a clustering algorithm.
        
        This method implements a simple circle-finding algorithm:
        1. Randomly select seed points from the point cloud
        2. For each seed, find nearby points within max_distance
        3. Check if the cluster forms a circular pattern matching a tennis ball
        4. Return the best matches sorted by quality
        
        Args:
            trigger_source (str): Source of the triggering detection (YOLO or HSV)
        
        Returns:
            list: List of tuples (center, cluster_size, circle_quality) for each detected ball
        """
        if self.points_array is None or len(self.points_array) == 0:
            self.get_logger().warn("LIDAR: No points available for analysis")
            return []
        
        self.get_logger().debug(
            f"LIDAR: Analyzing {len(self.points_array)} points triggered by {trigger_source}"
        )
        
        points = self.points_array
        balls_found = []
        
        # Try multiple random starting points
        for _ in range(TENNIS_BALL_CONFIG["detection_samples"]):
            # Pick a random point as a starting point
            seed_idx = np.random.randint(0, len(points))
            seed_point = points[seed_idx]
            
            # Find points close to this seed point (Euclidean distance in XY plane)
            distances = np.sqrt(
                (points[:, 0] - seed_point[0])**2 + 
                (points[:, 1] - seed_point[1])**2
            )
            cluster_indices = np.where(distances < self.max_distance)[0]
            cluster = points[cluster_indices]
            
            # Skip if cluster is too small
            if len(cluster) < self.min_points:
                continue
                
            # Calculate the center of the cluster (centroid)
            center = np.mean(cluster, axis=0)
            
            # Check if the cluster is approximately circular
            # by measuring distances from each point to the center
            center_distances = np.sqrt(
                (cluster[:, 0] - center[0])**2 + 
                (cluster[:, 1] - center[1])**2
            )
            
            # A tennis ball should have points at approximately ball_radius distance
            radius_errors = np.abs(center_distances - self.ball_radius)
            avg_error = np.mean(radius_errors)
            
            # Calculate quality metric (1.0 = perfect circle of exactly ball_radius)
            circle_quality = 1.0 - (avg_error / self.ball_radius)
            
            # Only consider clusters that reasonably match a tennis ball's shape
            quality_threshold = TENNIS_BALL_CONFIG["quality_threshold"]["low"]
            if circle_quality > quality_threshold:
                # Check if this ball is too close to any we've already found
                is_new_ball = True
                for existing_center, _, _ in balls_found:
                    dist = np.sqrt(
                        (center[0] - existing_center[0])**2 + 
                        (center[1] - existing_center[1])**2
                    )
                    # If centers are less than 2 radii apart, consider it the same ball
                    if dist < self.ball_radius * 2:
                        is_new_ball = False
                        break
                
                if is_new_ball:
                    balls_found.append((center, len(cluster), circle_quality))
        
        # Log results of the detection attempt
        if balls_found:
            self.get_logger().debug(
                f"LIDAR: Found {len(balls_found)} potential tennis balls"
            )
            
            # Log details of the best candidate
            best = balls_found[0]
            self.get_logger().debug(
                f"LIDAR: Best candidate at ({best[0][0]:.2f}, {best[0][1]:.2f}) "
                f"with {best[1]} points and quality {best[2]:.2f}"
            )
        else:
            self.get_logger().debug("LIDAR: No potential tennis balls found")
        
        # Sort balls by quality (best first) and return
        return sorted(balls_found, key=lambda x: x[2], reverse=True)
    
    def publish_ball_position(self, center, cluster_size, circle_quality, trigger_source, original_timestamp=None):
        """
        Publish the 3D position of a detected tennis ball.
        
        Creates and publishes messages with the ball's position along with
        visualization markers for RViz debugging.
        
        Args:
            center (numpy.ndarray): Center of the detected ball [x, y, z]
            cluster_size (int): Number of points in the cluster
            circle_quality (float): How well the points match a circle (0-1)
            trigger_source (str): Which detector triggered this detection (YOLO or HSV)
            original_timestamp (Time, optional): Original timestamp from the triggering detection
        """
        # Create message for ball position (3D point with timestamp)
        point_msg = PointStamped()
        
        # IMPORTANT: Use original timestamp if provided, otherwise use scan timestamp
        # This ensures we maintain the timing relationship for proper synchronization
        if original_timestamp:
            point_msg.header.stamp = original_timestamp
            self.get_logger().debug(f"Using original timestamp from {trigger_source} detection for synchronization")
        else:
            point_msg.header.stamp = self.scan_timestamp
        
        # Always use our consistent frame ID for proper transformation
        point_msg.header.frame_id = "lidar_frame"
        
        # Add sequence number for better synchronization
        if not hasattr(self, 'seq_counter'):
            self.seq_counter = 0
        self.seq_counter += 1
        point_msg.header.seq = self.seq_counter
        
        point_msg.point.x = float(center[0])
        point_msg.point.y = float(center[1])
        point_msg.point.z = float(center[2])
        
        # Publish the ball position
        self.position_publisher.publish(point_msg)
        
        # Update statistics
        self.successful_detections += 1
        
        # Determine confidence level based on circle quality
        if circle_quality > TENNIS_BALL_CONFIG["quality_threshold"]["high"]:
            confidence_text = "HIGH"
        elif circle_quality > TENNIS_BALL_CONFIG["quality_threshold"]["medium"]:
            confidence_text = "MEDIUM"
        else:
            confidence_text = "LOW"
            
        # Log the detection with detailed information
        self.get_logger().info(
            f"LIDAR: Tennis ball detected at ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) meters "
            f"with {cluster_size} points | Quality: {confidence_text} ({circle_quality:.2f}) | "
            f"Triggered by: {trigger_source}"
        )
        
        # Create visualization markers for RViz
        self.visualize_detection(center, cluster_size, circle_quality, trigger_source)
    
    def publish_transform(self):
        """
        Publish the LIDAR-to-camera transformation obtained through calibration.
        
        This transformation allows converting coordinates between the LIDAR
        and camera reference frames, which is essential for proper sensor fusion.
        """
        transform = TransformStamped()
        # Use current time for the transform to ensure it's considered valid
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "camera_frame"  # Parent frame
        transform.child_frame_id = "lidar_frame"    # Child frame
        
        # Translation from LIDAR to camera (from calibration)
        transform.transform.translation.x = -0.326256
        transform.transform.translation.y = 0.210052
        transform.transform.translation.z = 0.504021
        
        # Rotation from LIDAR to camera as quaternion (from calibration)
        transform.transform.rotation.x = -0.091584
        transform.transform.rotation.y = 0.663308
        transform.transform.rotation.z = 0.725666
        transform.transform.rotation.w = 0.158248
        
        self.tf_broadcaster.sendTransform(transform)
        
        # Log the transform occasionally to verify it's being published
        current_time = time.time()
        if not hasattr(self, 'last_transform_log') or current_time - self.last_transform_log > 60:
            self.get_logger().info("Publishing LIDAR-to-camera transform from calibration")
            self.last_transform_log = current_time

    def visualize_detection(self, center, cluster_size, circle_quality, trigger_source):
        """
        Create visualization markers for the detected ball in RViz.
        
        Creates two markers:
        1. A sphere representing the tennis ball
        2. A text label showing detection source and quality
        
        Args:
            center (numpy.ndarray): Center of the detected ball [x, y, z]
            cluster_size (int): Number of points in the cluster
            circle_quality (float): How well the points match a circle (0-1)
            trigger_source (str): Which detector triggered this detection
        """
        markers = MarkerArray()
        
        # Create a sphere marker for the ball
        ball_marker = Marker()
        ball_marker.header.frame_id = "lidar_frame"  # Use consistent frame ID
        ball_marker.header.stamp = self.scan_timestamp
        ball_marker.ns = "tennis_ball"
        ball_marker.id = 1
        ball_marker.type = Marker.SPHERE  # Show as a sphere
        ball_marker.action = Marker.ADD
        
        # Set position
        ball_marker.pose.position.x = center[0]
        ball_marker.pose.position.y = center[1]
        ball_marker.pose.position.z = center[2]
        ball_marker.pose.orientation.w = 1.0  # No rotation
        
        # Set color based on quality and source
        if trigger_source == "YOLO":
            # Green-ish for YOLO
            ball_marker.color.r = 0.0
            ball_marker.color.g = 1.0
            ball_marker.color.b = 0.3
        else:  # HSV
            # Orange-ish for HSV
            ball_marker.color.r = 1.0
            ball_marker.color.g = 0.6
            ball_marker.color.b = 0.0
        
        # Adjust transparency based on confidence
        ball_marker.color.a = min(0.5 + circle_quality * 0.5, 1.0)  # Higher quality = more opaque
        
        # Set size (tennis ball diameter)
        ball_marker.scale.x = self.ball_radius * 2.0
        ball_marker.scale.y = self.ball_radius * 2.0
        ball_marker.scale.z = self.ball_radius * 2.0
        
        # Set how long to display (1 second)
        ball_marker.lifetime.sec = 1
        
        markers.markers.append(ball_marker)
        
        # Add a text marker to show source and quality
        text_marker = Marker()
        text_marker.header.frame_id = self.scan_frame_id
        text_marker.header.stamp = self.scan_timestamp
        text_marker.ns = "tennis_ball_text"
        text_marker.id = 2
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        
        # Position text above the ball
        text_marker.pose.position.x = center[0]
        text_marker.pose.position.y = center[1]
        text_marker.pose.position.z = center[2] + 0.1  # 10cm above the ball
        text_marker.pose.orientation.w = 1.0
        
        # Set text content
        quality_pct = int(circle_quality * 100)
        text_marker.text = f"{trigger_source}: {quality_pct}%"
        
        # Set text appearance
        text_marker.scale.z = 0.05  # Text size
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        text_marker.lifetime.sec = 1
        
        markers.markers.append(text_marker)
        
        # Publish the visualization
        self.marker_publisher.publish(markers)
    
    def publish_status(self):
        """
        Publish diagnostic information about the node's performance.
        
        Calculates and publishes metrics including:
        - Scan processing rate
        - Detection success rate
        - Processing time statistics
        - Detection counts by source
        """
        try:
            # Calculate running time
            elapsed = time.time() - self.start_time
            
            # Skip if we just started
            if elapsed < 0.1:
                return
                
            # Calculate performance statistics
            scan_rate = self.processed_scans / elapsed
            detection_rate = self.successful_detections / elapsed
            
            # Calculate average processing time
            avg_time = 0
            if self.detection_times:
                avg_time = sum(self.detection_times) / len(self.detection_times)
            
            # Create status message with all metrics
            status = {
                "timestamp": time.time(),
                "uptime_seconds": elapsed,
                "processed_scans": self.processed_scans,
                "successful_detections": self.successful_detections,
                "detection_sources": {
                    "yolo_detections": self.yolo_detections,
                    "hsv_detections": self.hsv_detections
                },
                "performance": {
                    "scan_rate": scan_rate,
                    "detection_rate": detection_rate,
                    "avg_processing_time_ms": avg_time * 1000
                }
            }
            
            # Publish status as JSON string
            msg = String()
            msg.data = json.dumps(status)
            self.diagnostics_publisher.publish(msg)
            
            # Log summary of performance metrics
            self.get_logger().info(
                f"LIDAR: Status: {scan_rate:.1f} scans/sec, "
                f"{detection_rate:.1f} detections/sec, "
                f"YOLO: {self.yolo_detections}, HSV: {self.hsv_detections}, "
                f"avg processing: {avg_time*1000:.1f}ms"
            )
            
        except Exception as e:
            self.get_logger().error(f"LIDAR: Error publishing status: {str(e)}")


def main(args=None):
    """Main function to initialize and run the LIDAR detector node."""
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create detector node
    detector = TennisBallLidarDetector()
    
    # Print welcome message
    print("=================================================")
    print("Tennis Ball LIDAR Detector")
    print("=================================================")
    print("This node detects tennis balls using a 2D laser scanner.")
    print(f"Subscribing to LIDAR data on: {TOPICS['input']['lidar_scan']}")
    print(f"Subscribing to YOLO detections on: {TOPICS['input']['yolo_detection']}")
    print(f"Subscribing to HSV detections on: {TOPICS['input']['hsv_detection']}")
    print(f"Publishing ball positions to: {TOPICS['output']['ball_position']}")
    print("Use RViz to see the visualizations.")
    print("Press Ctrl+C to stop the program.")
    print("=================================================")
    
    try:
        # Keep the node running
        rclpy.spin(detector)
    except KeyboardInterrupt:
        print("LIDAR: Shutting down (Ctrl+C pressed)")
    except Exception as e:
        print(f"LIDAR: Error: {str(e)}")
    finally:
        # Clean shutdown
        detector.destroy_node()
        rclpy.shutdown()
        print("LIDAR: Tennis Ball LIDAR Detector has been shut down.")

if __name__ == '__main__':
    main()