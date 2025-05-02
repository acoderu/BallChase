#!/usr/bin/env python3

"""
Basketball Tracker Base Node

This provides a common base class for basketball tracking nodes (LIDAR, depth camera)
with shared functionality like prediction, tracking, visualization, and diagnostics.
"""

import time
import json
import math
from collections import deque

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from tf2_ros import Buffer, TransformListener

# ROS2 message types
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Float32
from visualization_msgs.msg import MarkerArray

# Project utilities
from ball_chase.utilities.movement_predictor import BasketballPredictor
from ball_chase.utilities.tracking_visualizer import TrackingVisualizer
from ball_chase.utilities.resource_monitor import ResourceMonitor
from ball_chase.config.config_loader import ConfigLoader


class BasketballTrackerNode(Node):
    """
    Base class for basketball tracking nodes providing common functionality.
    
    Features:
    - Common basketball physical parameters
    - Position prediction
    - Resource monitoring
    - Visualization
    - Diagnostics
    - Transform handling
    """
    
    def __init__(self, node_name, config_file=None):
        """
        Initialize the basketball tracker base node.
        
        Args:
            node_name: Name for this ROS2 node
            config_file: Optional configuration YAML file name
        """
        super().__init__(node_name)
        
        # Load configuration
        self.config_loader = ConfigLoader()
        if config_file:
            try:
                self.config = self.config_loader.load_yaml(config_file)
            except Exception as e:
                self.get_logger().error(f"Failed to load config: {str(e)}")
                self.config = {}
        else:
            self.config = {}
        
        # Initialize state tracking
        self._init_state()
        
        # Set up transforms
        self._setup_transform_system()
        
        # Load basketball parameters
        self._load_basketball_parameters()
        
        # Initialize the predictor
        predictor_config = self.config.get('predictor', {})
        self.predictor = BasketballPredictor(predictor_config)
        
        # Initialize the visualizer if visualization is enabled
        self._setup_visualizer()
        
        # Set up resource monitoring
        self._setup_resource_monitoring()
        
        # Create common publishers
        self._setup_common_publishers()
        
        # Diagnostics timer
        diag_interval = self.config.get('diagnostics', {}).get('publish_interval', 3.0)
        self.diagnostics_timer = self.create_timer(diag_interval, self.publish_diagnostics)
    
    def _init_state(self):
        """Initialize internal state tracking."""
        # Performance tracking
        self.start_time = time.time()
        self.processed_frames = 0
        self.successful_detections = 0
        self.processing_times = deque(maxlen=100)
        
        # Position tracking
        self.position_history = deque(maxlen=20)
        self.last_position = None
        self.consecutive_failures = 0
        self.last_successful_detection_time = 0
        
        # Health monitoring
        self.sensor_health = 1.0
        self.detection_health = 1.0
        self.detection_latency = 0.0
        self.errors = deque(maxlen=10)
        self.last_error_time = 0
        
        # Performance metrics
        self.processing_skips = 0
        self.current_cpu_load = 0.0
        self.current_memory_usage = 0.0
        
        # Performance management
        self.performance_mode = "NORMAL"  # Can be "NORMAL", "EFFICIENT", "MINIMAL"
    
    def _setup_transform_system(self):
        """Set up transforms for 3D coordinate systems."""
        # Set up TF system
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Get frame configuration
        frames_config = self.config.get('frames', {})
        
        # Reference frame - common coordinate system
        self.reference_frame = frames_config.get('reference_frame', 'base_link')
        self.sensor_frame = frames_config.get('sensor_frame', None)  # To be set by derived class
        
        # Transform parameters
        self.transform_timeout = frames_config.get('transform_timeout', 0.1)
        self.transform_cache = {}
        self.transform_cache_lifetime = 10.0  # seconds
        
        # Track transform availability
        self.transform_verified = False
        
        # Create a timer to verify transform availability
        self.transform_check_timer = self.create_timer(5.0, self.check_transform)
    
    def check_transform(self):
        """Periodically check if transforms are available."""
        # This should be customized by derived classes for their specific frames
        pass
    
    def _load_basketball_parameters(self):
        """Load basketball physical parameters from config."""
        # Get basketball configuration
        basketball_config = self.config.get('basketball', {})
        
        # Core parameters - ensure basketball sized (10 inch diameter)
        self.ball_radius = basketball_config.get('radius', 0.127)  # 5 inches (10 inch diameter)
        self.ball_diameter = self.ball_radius * 2.0
        self.ball_center_height = basketball_config.get('center_height', 0.127)  # 5 inches
        
        # Movement parameters
        self.max_speed = basketball_config.get('max_speed', 5.0)  # m/s
        self.min_speed_threshold = basketball_config.get('min_speed_threshold', 0.05)  # m/s
    
    def _setup_visualizer(self):
        """Set up visualization if enabled."""
        # Check if visualization is enabled
        viz_config = self.config.get('visualization', {})
        self.visualization_enabled = viz_config.get('enabled', False)
        
        # Initialize visualizer if enabled
        self.visualizer = None
        if self.visualization_enabled:
            viz_config['coordinate_frame'] = self.reference_frame
            viz_config['basketball_radius'] = self.ball_radius
            
            # Create visualizer
            self.visualizer = TrackingVisualizer(self, viz_config)
            
            # Set up visualization publisher
            output_topics = self.config.get('topics', {}).get('output', {})
            viz_topic = output_topics.get('visualization', '/basketball/visualization')
            self.viz_publisher = self.create_publisher(
                MarkerArray,
                viz_topic,
                10
            )
    
    def _setup_resource_monitoring(self):
        """Set up system resource monitoring."""
        resource_config = self.config.get('resources', {})
        monitor_enabled = resource_config.get('monitor_enabled', True)
        
        # Skip if disabled
        if not monitor_enabled:
            self.resource_monitor = None
            return
            
        # Create resource monitor
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=resource_config.get('publish_interval', 30.0),
            enable_temperature=resource_config.get('monitor_temperature', False)
        )
        
        # Set up callback
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        
        # Resource thresholds
        self.high_load_threshold = resource_config.get('high_load_threshold', 80.0)  # CPU %
        self.critical_load_threshold = resource_config.get('critical_load_threshold', 90.0)  # CPU %
        self.low_load_threshold = resource_config.get('low_load_threshold', 50.0)  # CPU %
        
        # Start monitoring
        self.resource_monitor.start()
        
        # Create a timer to periodically adjust performance based on load
        self.performance_timer = self.create_timer(10.0, self._adjust_performance)
    
    def _handle_resource_alert(self, resource_type, value):
        """Handle resource alerts from the monitor."""
        if resource_type == 'cpu':
            self.current_cpu_load = float(value)
        elif resource_type == 'memory':
            self.current_memory_usage = float(value)
    
    def _adjust_performance(self):
        """Adjust node performance based on system load."""
        # Default implementation - override in derived classes
        cpu = self.current_cpu_load
        
        # Determine performance mode based on CPU load
        if cpu > self.critical_load_threshold:
            new_mode = "MINIMAL"
        elif cpu > self.high_load_threshold:
            new_mode = "EFFICIENT"
        else:
            new_mode = "NORMAL"
        
        # Log mode changes
        if new_mode != self.performance_mode:
            self.get_logger().info(
                f"Performance mode change: {self.performance_mode} -> {new_mode} "
                f"(CPU: {self.current_cpu_load:.1f}%, Memory: {self.current_memory_usage:.1f}%)"
            )
            self.performance_mode = new_mode
    
    def _setup_common_publishers(self):
        """Set up common publishers used by all tracker nodes."""
        # Get topic configuration
        topics = self.config.get('topics', {})
        output_topics = topics.get('output', {})
        queue_size = topics.get('queue_size', 10)
        
        # Diagnostics publisher
        diag_topic = output_topics.get('diagnostics', '/basketball/diagnostics')
        self.diagnostics_publisher = self.create_publisher(
            String,
            diag_topic,
            queue_size
        )
        
        # System load publisher
        load_topic = output_topics.get('system_load', '/system/load')
        self.load_publisher = self.create_publisher(
            Float32,
            load_topic,
            queue_size
        )
    
    def publish_ball_position(self, position, confidence, trigger_source, timestamp=None):
        """
        Publish detected basketball position with prediction.
        
        Args:
            position: (x, y, z) position tuple/list
            confidence: Detection confidence (0.0 to 1.0)
            trigger_source: String identifying the detection source
            timestamp: Optional timestamp (defaults to current time)
        """
        # This method should be implemented by derived classes
        raise NotImplementedError("Derived classes must implement publish_ball_position")
    
    def visualize_detection(self, position, confidence, source):
        """
        Create and publish visualization markers.
        Only called when visualization is enabled.
        
        Args:
            position: (x, y, z) position tuple/list
            confidence: Detection confidence (0.0 to 1.0)  
            source: String identifying the detection source
        """
        # Skip if visualization is disabled or visualizer not created
        if not self.visualization_enabled or self.visualizer is None:
            return
            
        # Convert position history to list for visualization
        history = list(self.position_history)
        
        # Generate path prediction
        prediction_time = 1.0  # 1 second prediction
        predicted_path = self.predictor.predict_path(prediction_time)
        
        # Create visualization marker array
        markers = self.visualizer.create_marker_array(
            position, 
            confidence,
            history,
            predicted_path,
            source
        )
        
        # Publish markers
        self.viz_publisher.publish(markers)
    
    def publish_diagnostics(self):
        """Publish diagnostic information about the node."""
        try:
            # Calculate statistics
            current_time = time.time()
            elapsed = current_time - self.start_time
            
            if elapsed < 0.1:
                return
                
            # Calculate rates
            frame_rate = self.processed_frames / elapsed if elapsed > 0 else 0
            detection_rate = self.successful_detections / elapsed if elapsed > 0 else 0
            
            # Calculate average processing time
            avg_time = 0
            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
            
            # Get prediction statistics
            prediction_stats = self.predictor.get_statistics()
            
            # Create diagnostics message
            diagnostics = {
                "timestamp": current_time,
                "node": self._node_name,
                "uptime_seconds": elapsed,
                "status": "active",
                "performance_mode": self.performance_mode,
                "system": {
                    "cpu_load": self.current_cpu_load,
                    "memory_usage": self.current_memory_usage
                },
                "health": {
                    "sensor_health": self.sensor_health,
                    "detection_health": self.detection_health,
                    "overall": (self.sensor_health * 0.7 + self.detection_health * 0.3)
                },
                "metrics": {
                    "processed_frames": self.processed_frames,
                    "successful_detections": self.successful_detections,
                    "processing_skips": self.processing_skips,
                    "frame_rate": frame_rate,
                    "detection_rate": detection_rate,
                    "avg_processing_time_ms": avg_time * 1000
                },
                "prediction": {
                    "confidence": prediction_stats.get("prediction_confidence", 0),
                    "current_speed": prediction_stats.get("current_speed", 0),
                    "average_speed": prediction_stats.get("average_speed", 0)
                },
                "config": {
                    "ball_radius": self.ball_radius,
                    "visualization_enabled": self.visualization_enabled
                }
            }
            
            # Publish as JSON string
            msg = String()
            msg.data = json.dumps(diagnostics)
            self.diagnostics_publisher.publish(msg)
            
            # Publish system load for other nodes
            load_msg = Float32()
            load_msg.data = float(self.current_cpu_load)
            self.load_publisher.publish(load_msg)
            
            # Log basic summary - reduced logging in MINIMAL mode
            if self.performance_mode != "MINIMAL" or self.processed_frames % 5 == 0:
                self.get_logger().info(
                    f"{self._node_name}: Status: {frame_rate:.1f} frames/sec, "
                    f"{detection_rate:.1f} detections/sec, "
                    f"Mode: {self.performance_mode}, CPU: {self.current_cpu_load:.1f}%"
                )
                
        except Exception as e:
            self.log_error(f"Error publishing diagnostics: {str(e)}")
    
    def log_error(self, message):
        """Log an error and update health status."""
        # Add to error collection
        current_time = time.time()
        self.errors.append({
            "timestamp": current_time,
            "message": message
        })
        
        # Update health
        self.last_error_time = current_time
        self.sensor_health = max(0.3, self.sensor_health - 0.2)
        
        # Log the error
        self.get_logger().error(f"{self._node_name} ERROR: {message}")
    
    def destroy_node(self):
        """Clean shutdown of the node."""
        # Stop resource monitor if running
        if self.resource_monitor:
            try:
                self.resource_monitor.stop()
            except Exception:
                pass
        
        super().destroy_node()