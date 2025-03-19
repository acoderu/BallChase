#!/usr/bin/env python3

"""
System-Wide Diagnostics Node
----------------------------
This node centralizes diagnostics from all other nodes in the tennis ball
tracking system, providing a unified view of system health and performance.
It detects problems, logs diagnostics, and visualizes system status.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PointStamped, TwistStamped
import json
import time
import numpy as np
import psutil
import os
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime

# System-wide configuration
SYSTEM_NODES = ["fusion", "lidar", "hsv", "yolo", "depth_camera"]
LOG_FILE_DIR = os.path.expanduser("~/ball_chase_logs")

class SystemDiagnosticsNode(Node):
    """
    A ROS2 node that monitors and aggregates diagnostics from all nodes
    in the tennis ball tracking system, providing a unified view of system
    health and functionality.
    """
    
    def __init__(self):
        """Initialize the diagnostic monitoring node."""
        super().__init__('system_diagnostics')
        
        # Create log directory if it doesn't exist
        if not os.path.exists(LOG_FILE_DIR):
            os.makedirs(LOG_FILE_DIR)
        
        # Initialize state tracking
        self._init_state_tracking()
        
        # Set up diagnostic subscribers
        self._setup_subscribers()
        
        # Set up diagnostic publishers and status timer
        self._setup_publishers()
        
        # Monitor timers and watchdogs
        self.system_check_timer = self.create_timer(1.0, self.check_system_health)
        self.report_timer = self.create_timer(10.0, self.publish_system_report)
        
        # Set up CPU/memory usage monitoring
        self.resource_timer = self.create_timer(5.0, self.monitor_resources)
        
        # Open log file
        self._open_log_file()
        
        self.get_logger().info("System Diagnostics Node started - monitoring all components")
        
    def _init_state_tracking(self):
        """Initialize state tracking for all system components."""
        # Last seen timestamps for each node
        self.last_seen = {node: 0.0 for node in SYSTEM_NODES}
        
        # Latest diagnostic data from each node
        self.latest_diagnostics = {node: {} for node in SYSTEM_NODES}
        
        # Whether each node is currently considered active/healthy
        self.node_active = {node: False for node in SYSTEM_NODES}
        
        # Track restart attempts
        self.restart_attempts = {node: 0 for node in SYSTEM_NODES}
        self.last_restart = {node: 0.0 for node in SYSTEM_NODES}
        
        # Sensor statuses
        self.tracking_status = False
        self.ball_position = None
        self.ball_velocity = None
        
        # System resources
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        
        # Error counts
        self.error_counts = {node: 0 for node in SYSTEM_NODES}

    def _setup_subscribers(self):
        """Set up subscribers for all diagnostic topics."""
        # Node diagnostics
        self.create_subscription(
            String, "/tennis_ball/fusion/diagnostics", 
            lambda msg: self.diagnostic_callback(msg, "fusion"), 10)
            
        self.create_subscription(
            String, "/tennis_ball/lidar/diagnostics",
            lambda msg: self.diagnostic_callback(msg, "lidar"), 10)
        
        # Add subscriptions for other nodes
        self.create_subscription(
            String, "/tennis_ball/hsv/diagnostics",
            lambda msg: self.diagnostic_callback(msg, "hsv"), 10)
        
        self.create_subscription(
            String, "/tennis_ball/yolo/diagnostics",
            lambda msg: self.diagnostic_callback(msg, "yolo"), 10)
        
        self.create_subscription(
            String, "/tennis_ball/depth_camera/diagnostics",
            lambda msg: self.diagnostic_callback(msg, "depth_camera"), 10)
    
        # Other important system topics
        self.create_subscription(
            Bool, "/tennis_ball/fused/tracking_status",
            self.tracking_status_callback, 10)
            
        self.create_subscription(
            PointStamped, "/tennis_ball/fused/position",
            self.position_callback, 10)
            
        self.create_subscription(
            TwistStamped, "/tennis_ball/fused/velocity",
            self.velocity_callback, 10)
    
    def _setup_publishers(self):
        """Set up publishers for this node."""
        # System status publisher
        self.status_publisher = self.create_publisher(
            String, 
            "/tennis_ball/system/status", 
            10
        )
        
        # Health metrics publisher
        self.health_publisher = self.create_publisher(
            Float64MultiArray, 
            "/tennis_ball/system/health", 
            10
        )
        
        # Visualization publisher for RViz
        self.visualization_publisher = self.create_publisher(
            MarkerArray, 
            "/tennis_ball/system/visualization", 
            10
        )
    
    def _open_log_file(self):
        """Open log file for diagnostic data."""
        # Create log directory if it doesn't exist
        if not os.path.exists(LOG_FILE_DIR):
            os.makedirs(LOG_FILE_DIR)
        
        # Create a log file with timestamp in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filepath = os.path.join(LOG_FILE_DIR, f"diagnostics_{timestamp}.log")
        
        try:
            self.log_file = open(log_filepath, 'w')
            self.get_logger().info(f"Logging diagnostics to {log_filepath}")
        except Exception as e:
            self.get_logger().error(f"Failed to open log file: {str(e)}")
            self.log_file = None
    
    def log_event(self, event_type: str, node: str, details: str):
        """Log an event to both ROS log and the diagnostics log file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"{timestamp} [{event_type}] {node}: {details}"
        
        # Log to ROS
        if event_type == "ERROR":
            self.get_logger().error(log_entry)
        elif event_type == "WARNING":
            self.get_logger().warning(log_entry)
        else:
            self.get_logger().info(log_entry)
        
        # Log to file
        if hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.write(log_entry + "\n")
                self.log_file.flush()
            except Exception as e:
                self.get_logger().error(f"Failed to write to log file: {str(e)}")
    
    def diagnostic_callback(self, msg: String, node: str):
        """Process diagnostics data from a node."""
        try:
            # Parse JSON data
            diag_data = json.loads(msg.data)
            
            # Store the latest diagnostics
            self.latest_diagnostics[node] = diag_data
            
            # Update timestamp for when we last heard from this node
            self.last_seen[node] = time.time()
            
            # Update node status
            self.node_active[node] = True
            
            # Log errors and warnings from the node
            if "errors" in diag_data and diag_data["errors"]:
                for error in diag_data["errors"]:
                    self.log_event("ERROR", node, error)
                    self.error_counts[node] += 1
            
            if "warnings" in diag_data and diag_data["warnings"]:
                for warning in diag_data["warnings"]:
                    self.log_event("WARNING", node, warning)
            
            # Log significant health changes
            if "health" in diag_data and "overall" in diag_data["health"]:
                health_score = float(diag_data["health"]["overall"])
                if health_score < 0.5:
                    self.log_event("WARNING", node, f"Health is critical: {health_score:.2f}")
            
            # Log diagnostic summary at debug level
            self.get_logger().debug(f"Received diagnostics from {node}: status={diag_data.get('status', 'unknown')}")
            
        except json.JSONDecodeError:
            self.get_logger().error(f"Failed to parse diagnostics JSON from {node}")
            self.error_counts[node] += 1
        except Exception as e:
            self.get_logger().error(f"Error processing diagnostics from {node}: {str(e)}")
            self.error_counts[node] += 1
    
    def tracking_status_callback(self, msg: Bool):
        """Process tracking status updates."""
        self.tracking_status = msg.data
    
    def position_callback(self, msg: PointStamped):
        """Process ball position updates."""
        self.ball_position = {
            "x": msg.point.x,
            "y": msg.point.y,
            "z": msg.point.z,
            "timestamp": time.time()
        }
    
    def velocity_callback(self, msg: TwistStamped):
        """Process ball velocity updates."""
        self.ball_velocity = {
            "x": msg.twist.linear.x,
            "y": msg.twist.linear.y,
            "z": msg.twist.linear.z,
            "timestamp": time.time()
        }
    
    def check_system_health(self):
        """Check the health of all system nodes."""
        current_time = time.time()
        
        # Check if nodes have sent diagnostics recently (within 5 seconds)
        for node in SYSTEM_NODES:
            if current_time - self.last_seen[node] > 5.0:
                if self.node_active[node]:
                    # Node just became inactive
                    self.node_active[node] = False
                    self.log_event("WARNING", node, "Node appears to be inactive (no diagnostics for 5s)")
                    
                    # Attempt node recovery if it's been at least 30 seconds since last restart
                    if current_time - self.last_restart[node] > 30.0 and self.restart_attempts[node] < 3:
                        self.attempt_node_restart(node)
    
    def attempt_node_restart(self, node: str):
        """Attempt to restart a failed node."""
        self.restart_attempts[node] += 1
        self.last_restart[node] = time.time()
        
        self.log_event("WARNING", node, 
            f"Attempting to restart node (attempt #{self.restart_attempts[node]})")
        
        # In a real system, this would use ros2 lifecycle or some other restart mechanism
        # For now, we'll just log that we would restart it
        self.get_logger().warning(f"Would restart {node} node here (simulation only)")
        
        # In a real implementation:
        # try:
        #     # This is just an example - actual implementation would depend on your system
        #     subprocess.run(["ros2", "run", "ball_tracking", f"{node}_node"], 
        #                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5.0)
        #     self.log_event("INFO", node, "Restart command issued")
        # except Exception as e:
        #     self.log_event("ERROR", node, f"Failed to restart: {str(e)}")
    
    def monitor_resources(self):
        """Monitor system resources (CPU, memory)."""
        try:
            self.cpu_usage = psutil.cpu_percent(interval=None)
            self.memory_usage = psutil.virtual_memory().percent
            
            # Log if resources are constrained
            if self.cpu_usage > 90.0:
                self.log_event("WARNING", "system", f"High CPU usage: {self.cpu_usage:.1f}%")
            if self.memory_usage > 90.0:
                self.log_event("WARNING", "system", f"High memory usage: {self.memory_usage:.1f}%")
                
            # Adaptive behavior could be triggered here
            self.check_for_resource_constraints()
            
        except Exception as e:
            self.get_logger().error(f"Error monitoring system resources: {str(e)}")
    
    def check_for_resource_constraints(self):
        """Check for resource constraints and potentially adjust system behavior."""
        if self.cpu_usage > 80.0:
            # We could send messages to nodes to reduce their processing load
            # For example, decrease frame rates or detection frequency
            pass
    
    def publish_system_report(self):
        """Publish a comprehensive system status report."""
        # Compile system status
        report = {
            "timestamp": time.time(),
            "nodes": {
                node: {
                    "active": self.node_active[node],
                    "last_seen_seconds_ago": time.time() - self.last_seen[node] if self.last_seen[node] > 0 else None,
                    "error_count": self.error_counts[node],
                    "restart_attempts": self.restart_attempts[node]
                } for node in SYSTEM_NODES
            },
            "tracking": {
                "status": self.tracking_status,
                "has_position": self.ball_position is not None,
                "has_velocity": self.ball_velocity is not None
            },
            "resources": {
                "cpu_usage": self.cpu_usage,
                "memory_usage": self.memory_usage
            },
            "system_health": self.calculate_system_health()
        }
        
        # Publish the report
        status_msg = String()
        status_msg.data = json.dumps(report)
        self.system_status_pub.publish(status_msg)
        
        # Create visualization data
        self.publish_visualization_data(report)
        
        # Log overall system health
        health_score = report["system_health"]["score"]
        if health_score < 0.5:
            self.log_event("WARNING", "system", f"Poor system health: {health_score:.2f}/1.0")
        
    def calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score and status."""
        # Start with perfect health
        health_score = 1.0
        active_nodes = sum(1 for active in self.node_active.values() if active)
        
        # Reduce score for each inactive node
        health_score -= 0.2 * (len(SYSTEM_NODES) - active_nodes)
        
        # Reduce score for high resource usage
        if self.cpu_usage > 80.0:
            health_score -= 0.1
        if self.memory_usage > 80.0:
            health_score -= 0.1
            
        # Reduce score for error counts
        total_errors = sum(self.error_counts.values())
        health_score -= min(0.3, 0.01 * total_errors)
        
        # Don't allow negative scores
        health_score = max(0.0, health_score)
        
        health_status = "GOOD"
        if health_score < 0.6:
            health_status = "POOR"
        elif health_score < 0.8:
            health_status = "FAIR"
        
        return {
            "score": health_score,
            "status": health_status,
            "active_nodes": active_nodes,
            "total_nodes": len(SYSTEM_NODES)
        }
    
    def publish_visualization_data(self, report: Dict[str, Any]):
        """Publish data for visualization of system status."""
        # Build simplified visualization-friendly data structure
        viz_data = {
            "timestamp": report["timestamp"],
            "nodes": {
                node: {
                    "status": "active" if status else "inactive",
                    "errors": self.error_counts[node]
                } for node, status in self.node_active.items()
            },
            "ball_tracking": {
                "active": self.tracking_status,
                "position": self.ball_position,
                "velocity": self.ball_velocity
            },
            "system": {
                "cpu": self.cpu_usage,
                "memory": self.memory_usage,
                "health": report["system_health"]["status"]
            }
        }
        
        # Publish for visualization tools
        viz_msg = String()
        viz_msg.data = json.dumps(viz_data)
        self.system_viz_pub.publish(viz_msg)

    def __del__(self):
        """Clean up resources when node is shut down."""
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()


def main(args=None):
    """Main function to run the system diagnostics node."""
    rclpy.init(args=args)
    node = SystemDiagnosticsNode()
    
    try:
        print("System Diagnostics Node now monitoring all components. Press Ctrl+C to stop.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down System Diagnostics Node...")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if hasattr(node, 'log_file') and node.log_file:
            node.log_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
