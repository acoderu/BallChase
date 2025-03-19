#!/usr/bin/env python3

"""
Diagnostics Visualizer Node
----------------------------
This node subscribes to the system status topic and visualizes the
diagnostics information in a user-friendly format using RViz markers.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
import json
import time
from typing import Dict, List, Any, Optional
from collections import deque  # Add deque import

class DiagnosticsVisualizerNode(Node):
    """
    A ROS2 node that visualizes diagnostics information from the
    system_diagnostics node using RViz markers.
    """
    
    def __init__(self):
        """Initialize the diagnostics visualizer node."""
        super().__init__('diagnostics_visualizer')
        
        # Set up subscriber to system status topic
        self.subscription = self.create_subscription(
            String,
            '/tennis_ball/system/status',
            self.status_callback,
            10
        )
        
        # Individual node diagnostic subscribers
        self.node_subscribers = {
            'fusion': self.create_subscription(
                String, '/tennis_ball/fusion/diagnostics',
                lambda msg: self.node_diagnostic_callback(msg, 'fusion'), 10
            ),
            'hsv': self.create_subscription(
                String, '/tennis_ball/hsv/diagnostics',
                lambda msg: self.node_diagnostic_callback(msg, 'hsv'), 10
            ),
            'yolo': self.create_subscription(
                String, '/tennis_ball/yolo/diagnostics',
                lambda msg: self.node_diagnostic_callback(msg, 'yolo'), 10
            ),
            'depth_camera': self.create_subscription(
                String, '/tennis_ball/depth_camera/diagnostics',
                lambda msg: self.node_diagnostic_callback(msg, 'depth_camera'), 10
            ),
            'lidar': self.create_subscription(
                String, '/tennis_ball/lidar/diagnostics',
                lambda msg: self.node_diagnostic_callback(msg, 'lidar'), 10
            )
        }
        
        # Set up publisher for RViz markers
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/tennis_ball/diagnostics_visualization',
            10
        )
        
        # Store latest diagnostics for each node
        self.node_diagnostics = {}
        
        # Track last update time for each node
        self.last_update = {}
        
        # Store marker history with bounded deque
        self.marker_history = deque(maxlen=50)  # Store last 50 sets of markers
        
        # Store system status history with bounded deque
        self.status_history = deque(maxlen=100)  # Store last 100 status updates
        
        # Store error history with bounded deque
        self.error_history = deque(maxlen=50)   # Store last 50 errors
        
        # Visualization configuration
        self.viz_config = {
            'text_size': 0.15,
            'status_colors': {
                'active': {'r': 0.0, 'g': 1.0, 'b': 0.0},  # Green
                'warning': {'r': 1.0, 'g': 0.7, 'b': 0.0},  # Yellow
                'error': {'r': 1.0, 'g': 0.0, 'b': 0.0},    # Red
                'inactive': {'r': 0.5, 'g': 0.5, 'b': 0.5}  # Gray
            },
            'layout': {
                'start_x': 0.0,
                'start_y': 0.0,
                'node_spacing': 0.5,
                'section_spacing': 1.0
            }
        }
        
        # Create timer for regular visualization updates
        self.update_timer = self.create_timer(0.1, self.update_visualization)
        
        self.get_logger().info("Diagnostics Visualizer Node started")

    def node_diagnostic_callback(self, msg: String, node_name: str):
        """Process individual node diagnostic messages."""
        try:
            diag_data = json.loads(msg.data)
            self.node_diagnostics[node_name] = diag_data
            self.last_update[node_name] = time.time()
            
            # Save error/warning messages to history
            if 'errors' in diag_data and diag_data['errors']:
                for error in diag_data['errors']:
                    self.error_history.append({
                        'timestamp': time.time(),
                        'node': node_name,
                        'type': 'error',
                        'message': error
                    })
                    
            if 'warnings' in diag_data and diag_data['warnings']:
                for warning in diag_data['warnings']:
                    self.error_history.append({
                        'timestamp': time.time(),
                        'node': node_name,
                        'type': 'warning',
                        'message': warning
                    })
            
            if self.get_logger().get_effective_level() <= rclpy.logging.LoggingSeverity.DEBUG:
                self.get_logger().debug(f"Updated diagnostics for {node_name}")
        except json.JSONDecodeError:
            self.get_logger().error(f"Failed to parse diagnostic data from {node_name}")
        except Exception as e:
            self.get_logger().error(f"Error processing {node_name} diagnostics: {str(e)}")

    def create_markers(self, status_data: Dict[str, Any]) -> MarkerArray:
        """Create RViz markers based on system status data."""
        marker_array = MarkerArray()
        marker_id = 0
        
        # System health overview
        marker_array.markers.append(self.create_system_health_marker(
            status_data["system_health"],
            marker_id
        ))
        marker_id += 1
        
        # Node status section
        y_offset = -1.0  # Start below system health
        for node_name, node_data in status_data["nodes"].items():
            # Add detailed node info if we have it
            node_diag = self.node_diagnostics.get(node_name, {})
            
            marker_array.markers.extend(self.create_node_status_markers(
                node_name,
                node_data,
                node_diag,
                marker_id,
                y_offset
            ))
            marker_id += 2  # Each node uses 2 markers (status and details)
            y_offset -= 0.3  # Move down for next node
        
        # Resource usage section
        if "resources" in status_data:
            marker_array.markers.extend(self.create_resource_markers(
                status_data["resources"],
                marker_id,
                y_offset - 0.5
            ))
        
        return marker_array

    def create_system_health_marker(self, health_data: Dict[str, Any], marker_id: int) -> Marker:
        """Create a marker for overall system health."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "system_health"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position at top
        marker.pose.position.x = self.viz_config['layout']['start_x']
        marker.pose.position.y = self.viz_config['layout']['start_y']
        marker.pose.position.z = 0.5
        
        # Set size and color based on health
        marker.scale.z = self.viz_config['text_size'] * 1.5  # Larger text for system health
        score = health_data.get("score", 0.0)
        
        # Color based on health score
        if score >= 0.8:
            color = self.viz_config['status_colors']['active']
        elif score >= 0.6:
            color = self.viz_config['status_colors']['warning']
        else:
            color = self.viz_config['status_colors']['error']
            
        marker.color.r = color['r']
        marker.color.g = color['g']
        marker.color.b = color['b']
        marker.color.a = 1.0
        
        # Health text
        marker.text = (f"System Health: {health_data['status']} "
                      f"({score:.2f}) - {health_data['active_nodes']}/{health_data['total_nodes']} nodes active")
        
        return marker

    def create_node_status_markers(self, node_name: str, status: Dict[str, Any],
                                 diagnostics: Dict[str, Any], base_id: int,
                                 y_offset: float) -> List[Marker]:
        """Create status markers for a single node."""
        markers = []
        
        # Main status marker
        status_marker = Marker()
        status_marker.header.frame_id = "map"
        status_marker.header.stamp = self.get_clock().now().to_msg()
        status_marker.ns = "node_status"
        status_marker.id = base_id
        status_marker.type = Marker.TEXT_VIEW_FACING
        status_marker.action = Marker.ADD
        
        # Position
        status_marker.pose.position.x = self.viz_config['layout']['start_x']
        status_marker.pose.position.y = y_offset
        status_marker.pose.position.z = 0.0
        
        # Size
        status_marker.scale.z = self.viz_config['text_size']
        
        # Determine color based on node status and diagnostics
        node_active = status.get("active", False)
        error_count = status.get("error_count", 0)
        status_color = self.get_node_status_color(node_active, error_count, diagnostics)
        
        status_marker.color.r = status_color['r']
        status_marker.color.g = status_color['g']
        status_marker.color.b = status_color['b']
        status_marker.color.a = 1.0
        
        # Create status text
        status_text = self.format_node_status_text(node_name, status, diagnostics)
        status_marker.text = status_text
        markers.append(status_marker)
        
        # Add detailed info marker if available
        if diagnostics:
            detail_marker = self.create_node_detail_marker(
                node_name, diagnostics, base_id + 1,
                y_offset - 0.2  # Position below status
            )
            markers.append(detail_marker)
        
        return markers

    def get_node_status_color(self, active: bool, error_count: int,
                            diagnostics: Dict[str, Any]) -> Dict[str, float]:
        """Determine appropriate color for node status."""
        if not active:
            return self.viz_config['status_colors']['inactive']
        
        # Check diagnostic health if available
        if diagnostics and 'health' in diagnostics:
            health = diagnostics['health'].get('overall', 1.0)
            if health < 0.5:
                return self.viz_config['status_colors']['error']
            elif health < 0.8:
                return self.viz_config['status_colors']['warning']
        
        # Fallback to error count
        if error_count > 5:
            return self.viz_config['status_colors']['error']
        elif error_count > 0:
            return self.viz_config['status_colors']['warning']
        
        return self.viz_config['status_colors']['active']

    def format_node_status_text(self, node_name: str, status: Dict[str, Any],
                              diagnostics: Dict[str, Any]) -> str:
        """Format node status text with available information."""
        text_parts = [f"{node_name}:"]
        
        # Add status from system status
        if status.get("active", False):
            text_parts.append("ACTIVE")
        else:
            text_parts.append("INACTIVE")
        
        # Add error count
        if "error_count" in status:
            text_parts.append(f"Errors: {status['error_count']}")
        
        # Add diagnostic info if available
        if diagnostics:
            if 'status' in diagnostics:
                text_parts.append(f"Status: {diagnostics['status']}")
            if 'health' in diagnostics and 'overall' in diagnostics['health']:
                text_parts.append(f"Health: {diagnostics['health']['overall']:.2f}")
        
        return " | ".join(text_parts)

    def create_node_detail_marker(self, node_name: str, diagnostics: Dict[str, Any], 
                                marker_id: int, y_offset: float) -> Marker:
        """Create a detail marker showing more information about a node."""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "node_details"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # Position below main status
        marker.pose.position.x = self.viz_config['layout']['start_x'] + 1.0  # Indented
        marker.pose.position.y = y_offset
        marker.pose.position.z = 0.0
        
        # Smaller text for details
        marker.scale.z = self.viz_config['text_size'] * 0.85
        
        # Use lighter color for details
        marker.color.r = 0.8
        marker.color.g = 0.8
        marker.color.b = 0.8
        marker.color.a = 0.9
        
        # Build detail text based on available diagnostics
        details = []
        
        # Metrics (different nodes provide different metrics)
        if 'metrics' in diagnostics:
            metrics = diagnostics['metrics']
            
            # Common metrics
            if 'fps' in metrics:
                details.append(f"FPS: {metrics['fps']:.1f}")
            if 'processing_time_ms' in metrics:
                details.append(f"Proc: {metrics['processing_time_ms']:.1f}ms")
            if 'detection_rate' in metrics:
                details.append(f"Det: {metrics['detection_rate']*100:.1f}%")
                
            # Node-specific metrics
            if node_name == 'fusion' and 'update_rate_hz' in metrics:
                details.append(f"Update: {metrics['update_rate_hz']:.1f}Hz")
            if node_name == 'depth_camera' and 'conversion_rate' in metrics:
                details.append(f"Conv: {metrics['conversion_rate']:.1f}/s")
                
        # Detection information
        if 'detection' in diagnostics:
            detection = diagnostics['detection']
            if 'currently_tracking' in detection and detection['currently_tracking']:
                details.append("Tracking: YES")
            elif 'currently_tracking' in detection:
                details.append("Tracking: NO")
                
            # Add time since last detection
            if 'time_since_last_detection_s' in detection:
                t = detection['time_since_last_detection_s']
                if t < 5.0:  # Only show if recent
                    details.append(f"Last: {t:.1f}s ago")
        
        # Sensor-specific health
        if node_name == 'hsv' and 'health' in diagnostics:
            health = diagnostics['health']
            if 'camera_health' in health:
                details.append(f"Camera: {health['camera_health']:.2f}")
        
        # Resources (if available)
        if 'resources' in diagnostics:
            resources = diagnostics['resources']
            if 'temperature' in resources:
                details.append(f"Temp: {resources['temperature']:.1f}°C")
        
        # Join all details with commas
        marker.text = ", ".join(details)
        
        return marker

    def create_resource_markers(self, resource_data: Dict[str, Any],
                              base_id: int, y_offset: float) -> List[Marker]:
        """Create markers for system resource usage."""
        markers = []
        
        # CPU usage marker
        cpu_marker = Marker()
        cpu_marker.header.frame_id = "map"
        cpu_marker.header.stamp = self.get_clock().now().to_msg()
        cpu_marker.ns = "resources"
        cpu_marker.id = base_id
        cpu_marker.type = Marker.TEXT_VIEW_FACING
        cpu_marker.action = Marker.ADD
        
        cpu_marker.pose.position.x = self.viz_config['layout']['start_x']
        cpu_marker.pose.position.y = y_offset
        cpu_marker.pose.position.z = 0.0
        
        cpu_marker.scale.z = self.viz_config['text_size']
        
        # Color based on CPU usage
        cpu_usage = resource_data.get("cpu_usage", 0.0)
        cpu_marker.color.a = 1.0
        if cpu_usage > 90:
            cpu_marker.color.r = 1.0
            cpu_marker.color.g = 0.0
        elif cpu_usage > 75:
            cpu_marker.color.r = 1.0
            cpu_marker.color.g = 0.5
        else:
            cpu_marker.color.r = 0.0
            cpu_marker.color.g = 1.0
        
        cpu_marker.text = f"CPU: {cpu_usage:.1f}%"
        markers.append(cpu_marker)
        
        # Memory usage marker
        mem_marker = Marker()
        mem_marker.header.frame_id = "map"
        mem_marker.header.stamp = self.get_clock().now().to_msg()
        mem_marker.ns = "resources"
        mem_marker.id = base_id + 1
        mem_marker.type = Marker.TEXT_VIEW_FACING
        mem_marker.action = Marker.ADD
        
        mem_marker.pose.position.x = self.viz_config['layout']['start_x'] + 2.0  # Offset to the right
        mem_marker.pose.position.y = y_offset
        mem_marker.pose.position.z = 0.0
        
        mem_marker.scale.z = self.viz_config['text_size']
        
        # Color based on memory usage
        mem_usage = resource_data.get("memory_usage", 0.0)
        mem_marker.color.a = 1.0
        if mem_usage > 90:
            mem_marker.color.r = 1.0
            mem_marker.color.g = 0.0
        elif mem_usage > 75:
            mem_marker.color.r = 1.0
            mem_marker.color.g = 0.5
        else:
            mem_marker.color.r = 0.0
            mem_marker.color.g = 1.0
        
        mem_marker.text = f"Memory: {mem_usage:.1f}%"
        markers.append(mem_marker)
        
        # Add temperature marker for Raspberry Pi
        if "temperature" in resource_data:
            temp_marker = Marker()
            temp_marker.header.frame_id = "map"
            temp_marker.header.stamp = self.get_clock().now().to_msg()
            temp_marker.ns = "resources"
            temp_marker.id = base_id + 2
            temp_marker.type = Marker.TEXT_VIEW_FACING
            temp_marker.action = Marker.ADD
            
            temp_marker.pose.position.x = self.viz_config['layout']['start_x'] + 4.0  # Further right
            temp_marker.pose.position.y = y_offset
            temp_marker.pose.position.z = 0.0
            
            temp_marker.scale.z = self.viz_config['text_size']
            
            # Color based on temperature (Raspberry Pi throttles at 80°C)
            temp = resource_data["temperature"]
            temp_marker.color.a = 1.0
            if temp > 75:  # Near throttling
                temp_marker.color.r = 1.0
                temp_marker.color.g = 0.0
                temp_marker.color.b = 0.0
            elif temp > 65:  # Getting warm
                temp_marker.color.r = 1.0
                temp_marker.color.g = 0.5
                temp_marker.color.b = 0.0
            else:  # Normal operation
                temp_marker.color.r = 0.0
                temp_marker.color.g = 1.0
                temp_marker.color.b = 0.0
            
            temp_marker.text = f"Temp: {temp:.1f}°C"
            markers.append(temp_marker)
            
            # Add warning if temperature is high
            if temp > 75:
                warning_marker = Marker()
                warning_marker.header.frame_id = "map"
                warning_marker.header.stamp = self.get_clock().now().to_msg()
                warning_marker.ns = "resources"
                warning_marker.id = base_id + 3
                warning_marker.type = Marker.TEXT_VIEW_FACING
                warning_marker.action = Marker.ADD
                
                warning_marker.pose.position.x = self.viz_config['layout']['start_x']
                warning_marker.pose.position.y = y_offset - 0.3
                warning_marker.pose.position.z = 0.0
                
                warning_marker.scale.z = self.viz_config['text_size']
                warning_marker.color.r = 1.0
                warning_marker.color.g = 0.0
                warning_marker.color.b = 0.0
                warning_marker.color.a = 1.0
                
                warning_marker.text = "WARNING: High Temperature!"
                markers.append(warning_marker)
        
        return markers

    def update_visualization(self):
        """Update visualization markers periodically."""
        current_time = time.time()
        
        # Check for stale node data
        for node_name in list(self.node_diagnostics.keys()):
            if current_time - self.last_update.get(node_name, 0) > 5.0:
                # Clear stale data
                self.node_diagnostics.pop(node_name)
                self.get_logger().warn(f"Diagnostic data for {node_name} is stale")

    def status_callback(self, msg: String):
        """Process system status updates and publish RViz markers."""
        try:
            # Parse JSON data from the status message
            status_data = json.loads(msg.data)
            
            # Store in history
            self.status_history.append({
                'timestamp': time.time(),
                'data': status_data
            })
            
            # Create RViz markers based on the status data
            marker_array = self.create_markers(status_data)
            
            # Store marker history
            self.marker_history.append({
                'timestamp': time.time(),
                'markers': marker_array
            })
            
            # Publish the marker array
            self.marker_publisher.publish(marker_array)
            
            # Log system health status occasionally
            if not hasattr(self, 'last_health_log') or time.time() - self.last_health_log > 30.0:
                health = status_data.get("system_health", {})
                active_nodes = health.get("active_nodes", 0)
                total_nodes = health.get("total_nodes", 0)
                self.get_logger().info(
                    f"System health: {health.get('status', 'UNKNOWN')} ({health.get('score', 0.0):.2f}), "
                    f"Nodes: {active_nodes}/{total_nodes} active"
                )
                self.last_health_log = time.time()
            
        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse JSON from system status message")
        except Exception as e:
            self.get_logger().error(f"Error processing system status: {str(e)}")

def main(args=None):
    """Main function to run the diagnostics visualizer node."""
    rclpy.init(args=args)
    node = DiagnosticsVisualizerNode()
    
    try:
        print("Diagnostics Visualizer Node now listening for system status. Press Ctrl+C to stop.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down Diagnostics Visualizer Node...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()