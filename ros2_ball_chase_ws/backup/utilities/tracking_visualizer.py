#!/usr/bin/env python3

"""
Basketball Tracking Visualization Utilities

This module provides common visualization functionality that can be used by
different sensor nodes (LIDAR, depth camera) when visualizing basketball tracking
and prediction data in RViz.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA


class TrackingVisualizer:
    """
    Provides standardized visualization tools for basketball tracking across nodes.
    
    Features:
    - Basketball visualization
    - Path history visualization
    - Prediction path visualization
    - Detection quality indicators
    """
    
    def __init__(self, node, config: Dict = None):
        """
        Initialize the basketball tracking visualizer.
        
        Args:
            node: The ROS2 node this visualizer is attached to
            config: Optional configuration dictionary
        """
        self.node = node
        
        # Default configuration
        self.config = {
            "basketball_radius": 0.127,        # Basketball radius in meters (5 inches)
            "basketball_color": (1.0, 0.5, 0.0, 0.8),  # Orange with alpha
            "history_line_width": 0.02,        # Line width for history trail
            "history_color": (0.0, 0.7, 1.0, 0.6),  # Light blue with alpha
            "prediction_line_width": 0.02,     # Line width for prediction path
            "prediction_color": (1.0, 1.0, 0.0, 0.7),  # Yellow with alpha
            "max_history_points": 20,          # Maximum number of history points
            "text_height_offset": 0.2,         # Height above basketball for text
            "text_size": 0.05,                 # Size of text markers
            "coordinate_frame": "base_link",   # Default coordinate frame
            "marker_lifetime_sec": 1.0,        # How long markers persist
            "confidence_fade": True            # Fade markers based on confidence
        }
        
        # Override defaults with provided config
        if config:
            for key, value in config.items():
                if key in self.config:
                    self.config[key] = value
    
    def create_basketball_marker(self, 
                                position: Tuple[float, float, float], 
                                confidence: float = 1.0,
                                marker_id: int = 0,
                                source: str = "sensor") -> Marker:
        """
        Create a basketball visualization marker.
        
        Args:
            position: (x, y, z) position of basketball center
            confidence: Detection confidence (0.0 to 1.0)
            marker_id: Unique ID for this marker
            source: Source of the detection (e.g., "lidar", "depth")
            
        Returns:
            Marker message for the basketball
        """
        marker = Marker()
        marker.header.frame_id = self.config["coordinate_frame"]
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = f"basketball_{source}"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0
        
        # Set size (basketball diameter)
        marker.scale.x = self.config["basketball_radius"] * 2.0
        marker.scale.y = self.config["basketball_radius"] * 2.0
        marker.scale.z = self.config["basketball_radius"] * 2.0
        
        # Set color with alpha based on confidence
        r, g, b, a = self.config["basketball_color"]
        if self.config["confidence_fade"]:
            a = a * max(0.3, confidence)  # Minimum 30% opacity
            
        marker.color.r = float(r)
        marker.color.g = float(g)
        marker.color.b = float(b)
        marker.color.a = float(a)
        
        # Set marker lifetime
        lifetime_sec = self.config["marker_lifetime_sec"]
        marker.lifetime.sec = int(lifetime_sec)
        marker.lifetime.nanosec = int((lifetime_sec % 1) * 1e9)
        
        return marker
    
    def create_text_marker(self, 
                          position: Tuple[float, float, float], 
                          text: str,
                          marker_id: int = 100,
                          source: str = "sensor") -> Marker:
        """
        Create a text visualization marker.
        
        Args:
            position: (x, y, z) position for the text
            text: Text to display
            marker_id: Unique ID for this marker
            source: Source of the detection
            
        Returns:
            Marker message for the text
        """
        marker = Marker()
        marker.header.frame_id = self.config["coordinate_frame"]
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = f"text_{source}"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position text above the position
        offset = self.config["text_height_offset"]
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2] + offset
        marker.pose.orientation.w = 1.0
        
        # Set text
        marker.text = text
        
        # Set text size
        marker.scale.z = self.config["text_size"]
        
        # Set color (white by default)
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        
        # Set marker lifetime
        lifetime_sec = self.config["marker_lifetime_sec"]
        marker.lifetime.sec = int(lifetime_sec)
        marker.lifetime.nanosec = int((lifetime_sec % 1) * 1e9)
        
        return marker
    
    def create_path_marker(self, 
                          path_points: List[Tuple[float, float, float]], 
                          confidences: Optional[List[float]] = None,
                          marker_id: int = 200,
                          source: str = "history",
                          line_width: float = None,
                          color: Tuple[float, float, float, float] = None) -> Marker:
        """
        Create a path/line visualization marker.
        
        Args:
            path_points: List of (x, y, z) points in the path
            confidences: Optional list of confidence values (0.0-1.0) for each point
            marker_id: Unique ID for this marker
            source: Source identifier ("history" or "prediction")
            line_width: Override line width
            color: Override color tuple (r, g, b, a)
            
        Returns:
            Marker message for the path
        """
        if not path_points or len(path_points) < 2:
            return None
            
        marker = Marker()
        marker.header.frame_id = self.config["coordinate_frame"]
        marker.header.stamp = self.node.get_clock().now().to_msg()
        marker.ns = f"path_{source}"
        marker.id = marker_id
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Add points to the path
        for point in path_points:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            marker.points.append(p)
        
        # Add colors if confidences are provided
        if confidences and self.config["confidence_fade"]:
            if len(confidences) == len(path_points):
                # Default color (use specified or source-based default)
                if color:
                    default_r, default_g, default_b, default_a = color
                elif source == "prediction":
                    default_r, default_g, default_b, default_a = self.config["prediction_color"]
                else:
                    default_r, default_g, default_b, default_a = self.config["history_color"]
                
                # Create a color for each point based on confidence
                for confidence in confidences:
                    c = ColorRGBA()
                    c.r = default_r
                    c.g = default_g
                    c.b = default_b
                    c.a = default_a * max(0.3, confidence)  # Minimum 30% opacity
                    marker.colors.append(c)
        
        # Set line width
        if line_width is None:
            if source == "prediction":
                line_width = self.config["prediction_line_width"]
            else:
                line_width = self.config["history_line_width"]
                
        marker.scale.x = line_width  # LINE_STRIP uses only x for width
        
        # Set color if no per-point colors
        if not marker.colors:
            if color:
                r, g, b, a = color
            elif source == "prediction":
                r, g, b, a = self.config["prediction_color"]
            else:
                r, g, b, a = self.config["history_color"]
                
            marker.color.r = float(r)
            marker.color.g = float(g)
            marker.color.b = float(b)
            marker.color.a = float(a)
        
        # Set marker lifetime
        lifetime_sec = self.config["marker_lifetime_sec"]
        marker.lifetime.sec = int(lifetime_sec)
        marker.lifetime.nanosec = int((lifetime_sec % 1) * 1e9)
        
        return marker
    
    def create_history_marker(self, 
                            positions: List[Tuple[float, float, float]], 
                            marker_id: int = 200,
                            source: str = "sensor") -> Marker:
        """
        Create a marker visualizing recent position history.
        
        Args:
            positions: List of (x, y, z) historical positions
            marker_id: Unique ID for this marker
            source: Source identifier
            
        Returns:
            Marker message for the history path
        """
        # Limit to max history points
        if len(positions) > self.config["max_history_points"]:
            positions = positions[-self.config["max_history_points"]:]
            
        # No confidences for history - just use the path marker
        return self.create_path_marker(
            positions, 
            None, 
            marker_id,
            f"history_{source}", 
            self.config["history_line_width"],
            self.config["history_color"]
        )
    
    def create_prediction_markers(self, 
                                predicted_path: List[Tuple[Tuple[float, float, float], float]], 
                                marker_id_base: int = 300,
                                source: str = "sensor") -> Marker:
        """
        Create a marker visualizing the predicted path.
        
        Args:
            predicted_path: List of ((x, y, z), confidence) tuples
            marker_id_base: Base ID for markers
            source: Source identifier
            
        Returns:
            Marker message for the prediction path
        """
        if not predicted_path:
            return None
            
        # Extract positions and confidences
        positions = [pos for pos, _ in predicted_path]
        confidences = [conf for _, conf in predicted_path]
            
        # Create prediction path with confidences
        return self.create_path_marker(
            positions, 
            confidences, 
            marker_id_base,
            f"prediction_{source}", 
            self.config["prediction_line_width"],
            self.config["prediction_color"]
        )
    
    def create_marker_array(self, 
                          position: Tuple[float, float, float],
                          confidence: float = 1.0,
                          history: List[Tuple[float, float, float]] = None,
                          prediction: List[Tuple[Tuple[float, float, float], float]] = None,
                          source: str = "sensor",
                          include_text: bool = True,
                          text: str = None) -> MarkerArray:
        """
        Create a complete visualization of basketball tracking including:
        - Current position sphere
        - Text label
        - History trail
        - Prediction path
        
        Args:
            position: Current (x, y, z) position
            confidence: Detection confidence (0.0 to 1.0)
            history: Optional list of historical positions
            prediction: Optional list of predicted positions with confidences
            source: Source identifier
            include_text: Whether to include text label
            text: Custom text (defaults to source name with confidence)
            
        Returns:
            MarkerArray containing all visualization elements
        """
        markers = MarkerArray()
        
        # Add basketball marker
        ball_marker = self.create_basketball_marker(position, confidence, 0, source)
        markers.markers.append(ball_marker)
        
        # Add text marker
        if include_text:
            if text is None:
                conf_pct = int(confidence * 100)
                text = f"{source}: {conf_pct}%"
                
            text_marker = self.create_text_marker(position, text, 100, source)
            markers.markers.append(text_marker)
        
        # Add history path if provided
        if history and len(history) > 1:
            history_marker = self.create_history_marker(history, 200, source)
            if history_marker:
                markers.markers.append(history_marker)
        
        # Add prediction path if provided
        if prediction and len(prediction) > 1:
            prediction_marker = self.create_prediction_markers(prediction, 300, source)
            if prediction_marker:
                markers.markers.append(prediction_marker)
        
        return markers