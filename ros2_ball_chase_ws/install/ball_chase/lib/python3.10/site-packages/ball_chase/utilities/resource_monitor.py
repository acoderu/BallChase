"""
Resource Monitor for ROS2 on Raspberry Pi 5
------------------------------------------

This module provides utilities to monitor CPU, memory, and temperature
on Raspberry Pi hardware. It's designed to help track system performance
while running computationally intensive robotics applications.

Optimized for Raspberry Pi 5 with 16GB RAM.
"""

import os
import time
import threading
import psutil
from typing import Dict, Optional, List, Callable
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String


class ResourceMonitor:
    """
    Monitors system resources including CPU, memory, and temperature.
    
    This is especially useful for Raspberry Pi applications where
    resource constraints can impact performance.
    """
    
    def __init__(self, 
                 node: Optional[Node] = None,
                 publish_interval: float = 5.0,
                 enable_temperature: bool = True,
                 enable_publication: bool = True,
                 high_memory_mode: bool = True):  # New parameter for 16GB Pi
        """
        Initialize the resource monitor.
        
        Args:
            node: ROS2 node to use for publishing (if None, won't publish)
            publish_interval: How often to publish metrics (seconds)
            enable_temperature: Whether to monitor temperature (may require specific hardware)
            enable_publication: Whether to publish metrics to ROS topics
            high_memory_mode: Whether the system has ample RAM (16GB Pi 5)
        """
        self.node = node
        self.publish_interval = publish_interval
        self.enable_temperature = enable_temperature
        self.enable_publication = enable_publication and (node is not None)
        self.high_memory_mode = high_memory_mode
        
        # Resource data storage
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.temperature = 0.0
        self.last_update_time = 0.0
        
        # Storage for detailed stats (enabled in high memory mode)
        if high_memory_mode:
            self.cpu_per_core = []
            self.memory_details = {}
            self.network_io = {}
            self.disk_io = {}
        
        # ROS2 publishers
        if self.enable_publication and self.node:
            self.metrics_publisher = self.node.create_publisher(
                Float32MultiArray, 
                '/system/resources', 
                10
            )
            
            self.status_publisher = self.node.create_publisher(
                String,
                '/system/status',
                10
            )
            
            # In high memory mode, add more detailed publishers
            if high_memory_mode:
                self.detailed_publisher = self.node.create_publisher(
                    String,
                    '/system/resources/detailed',
                    10
                )
        
        # Running flag for the monitoring thread
        self.running = False
        self.monitor_thread = None
        
        # Callbacks for threshold alerts
        self.alert_callbacks: List[Callable[[str, float], None]] = []
        
        # With 16GB RAM, we can be less conservative with thresholds
        self.thresholds = {
            'cpu': 90.0,       # 90% CPU usage (vs 80% for low memory)
            'memory': 90.0,    # 90% memory usage (vs 85% for low memory) 
            'temperature': 80.0 # 80°C (Raspberry Pi typically throttles at 85°C)
        }
    
    def start(self):
        """Start the resource monitoring thread."""
        if self.running:
            return
            
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        if self.node:
            self.node.get_logger().info("Resource monitoring started")
    
    def stop(self):
        """Stop the resource monitoring thread."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            self.monitor_thread = None
            
        if self.node:
            self.node.get_logger().info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while self.running:
            try:
                self._update_metrics()
                
                if self.enable_publication and time.time() - self.last_update_time >= self.publish_interval:
                    self._publish_metrics()
                    self.last_update_time = time.time()
                    
                # Check for threshold violations
                self._check_thresholds()
                
                # Sleep to avoid excessive CPU usage
                time.sleep(1.0)
                
            except Exception as e:
                if self.node:
                    self.node.get_logger().error(f"Error in resource monitoring: {str(e)}")
                time.sleep(5.0)  # Longer sleep after error
    
    def _update_metrics(self):
        """Update all resource metrics."""
        try:
            # CPU usage (average across all cores)
            self.cpu_percent = psutil.cpu_percent(interval=None)
            
            # In high memory mode, get per-core stats
            if self.high_memory_mode:
                self.cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_percent = memory.percent
            
            # More detailed memory info in high memory mode
            if self.high_memory_mode:
                self.memory_details = {
                    'total': memory.total / (1024 * 1024),  # MB
                    'available': memory.available / (1024 * 1024),  # MB
                    'used': memory.used / (1024 * 1024),  # MB
                    'percent': memory.percent
                }
                
                # Get network I/O stats
                net_io = psutil.net_io_counters()
                self.network_io = {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                }
                
                # Get disk I/O stats
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    self.disk_io = {
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes
                    }
            
            # Temperature (Raspberry Pi specific)
            if self.enable_temperature:
                self._update_temperature()
                
        except Exception as e:
            if self.node:
                self.node.get_logger().warn(f"Failed to update system metrics: {str(e)}")
    
    def _update_temperature(self):
        """Update temperature metrics (Raspberry Pi specific)."""
        try:
            # Try Raspberry Pi specific temperature file first
            if os.path.exists('/sys/class/thermal/thermal_zone0/temp'):
                with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                    self.temperature = float(f.read().strip()) / 1000.0
            else:
                # Fallback to psutil for other platforms
                temps = psutil.sensors_temperatures()
                if temps and 'cpu_thermal' in temps:
                    self.temperature = temps['cpu_thermal'][0].current
                elif temps and 'coretemp' in temps:
                    self.temperature = temps['coretemp'][0].current
        except Exception:
            # Temperature monitoring is optional, so just set to 0 if unavailable
            self.temperature = 0.0
    
    def _publish_metrics(self):
        """Publish resource metrics to ROS topics."""
        if not self.enable_publication or not self.node:
            return
            
        try:
            # Publish numeric metrics
            metrics_msg = Float32MultiArray()
            metrics_msg.data = [self.cpu_percent, self.memory_percent, self.temperature]
            self.metrics_publisher.publish(metrics_msg)
            
            # Publish human-readable status
            status_msg = String()
            
            # Include more details for high memory mode
            if self.high_memory_mode and hasattr(self, 'memory_details'):
                status_msg.data = (
                    f"CPU: {self.cpu_percent:.1f}% | "
                    f"Memory: {self.memory_percent:.1f}% "
                    f"({self.memory_details['used']:.0f}/{self.memory_details['total']:.0f} MB) | "
                    f"Temp: {self.temperature:.1f}°C"
                )
            else:
                status_msg.data = (
                    f"CPU: {self.cpu_percent:.1f}% | "
                    f"Memory: {self.memory_percent:.1f}% | "
                    f"Temp: {self.temperature:.1f}°C"
                )
                
            self.status_publisher.publish(status_msg)
            
            # Publish detailed metrics in high memory mode
            if self.high_memory_mode and hasattr(self, 'detailed_publisher'):
                detailed_msg = String()
                
                # Create a detailed JSON string with all metrics
                import json
                detailed_data = {
                    'timestamp': time.time(),
                    'cpu': {
                        'total': self.cpu_percent,
                        'per_core': self.cpu_per_core
                    },
                    'memory': self.memory_details,
                    'temperature': self.temperature,
                    'network': self.network_io,
                    'disk': self.disk_io
                }
                
                detailed_msg.data = json.dumps(detailed_data)
                self.detailed_publisher.publish(detailed_msg)
            
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"Failed to publish resource metrics: {str(e)}")
    
    def _check_thresholds(self):
        """Check if any metrics exceed their thresholds and trigger callbacks."""
        if self.cpu_percent > self.thresholds['cpu']:
            self._trigger_alert('cpu', self.cpu_percent)
            
        if self.memory_percent > self.thresholds['memory']:
            self._trigger_alert('memory', self.memory_percent)
            
        if self.temperature > self.thresholds['temperature']:
            self._trigger_alert('temperature', self.temperature)
    
    def _trigger_alert(self, resource_type: str, value: float):
        """Trigger an alert for a threshold violation."""
        for callback in self.alert_callbacks:
            try:
                callback(resource_type, value)
            except Exception as e:
                if self.node:
                    self.node.get_logger().error(f"Error in alert callback: {str(e)}")
    
    def add_alert_callback(self, callback: Callable[[str, float], None]):
        """
        Add a callback to be called when a threshold is exceeded.
        
        Args:
            callback: Function taking (resource_type, value) arguments
        """
        self.alert_callbacks.append(callback)
    
    def set_threshold(self, resource_type: str, threshold: float):
        """
        Set a threshold for a specific resource.
        
        Args:
            resource_type: 'cpu', 'memory', or 'temperature'
            threshold: The threshold value
        """
        if resource_type in self.thresholds:
            self.thresholds[resource_type] = threshold
    
    def get_current_metrics(self) -> Dict[str, float]:
        """
        Get the current resource metrics.
        
        Returns:
            Dictionary with current metrics
        """
        return {
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'temperature': self.temperature,
            'timestamp': time.time()
        }


def print_alert(resource_type: str, value: float):
    """Example alert callback that prints to console."""
    print(f"ALERT: {resource_type.upper()} usage at {value:.1f}% exceeds threshold!")
    

# Simple usage example when run as a standalone script
if __name__ == "__main__":
    print("Starting resource monitor standalone test...")
    monitor = ResourceMonitor(node=None, enable_publication=False)
    monitor.add_alert_callback(print_alert)
    monitor.start()
    
    try:
        # Run for 30 seconds
        for i in range(30):
            metrics = monitor.get_current_metrics()
            print(f"CPU: {metrics['cpu_percent']:.1f}% | "
                  f"Memory: {metrics['memory_percent']:.1f}% | "
                  f"Temperature: {metrics['temperature']:.1f}°C")
            time.sleep(1)
    finally:
        monitor.stop()
        print("Resource monitor test completed")
