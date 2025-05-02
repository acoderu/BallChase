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
from typing import Dict, Optional, List, Callable, Any, Tuple
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from collections import deque


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
                 high_memory_mode: bool = True,
                 cpu_sample_window: int = 5,
                 enable_async_monitoring: bool = True):
        """
        Initialize the resource monitor.
        
        Args:
            node: ROS2 node to use for publishing (if None, won't publish)
            publish_interval: How often to publish metrics (seconds)
            enable_temperature: Whether to monitor temperature (may require specific hardware)
            enable_publication: Whether to publish metrics to ROS topics
            high_memory_mode: Whether the system has ample RAM (16GB Pi 5)
            cpu_sample_window: Number of samples to keep for CPU averaging
            enable_async_monitoring: Whether to use async monitoring (more efficient)
        """
        self.node = node
        self.publish_interval = publish_interval
        self.enable_temperature = enable_temperature
        self.enable_publication = enable_publication and (node is not None)
        self.high_memory_mode = high_memory_mode
        self.enable_async = enable_async_monitoring
        self.cpu_sample_window = cpu_sample_window
        
        # Resource data storage
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.temperature = 0.0
        self.last_update_time = 0.0
        
        # Use deque for CPU samples to efficiently calculate moving averages
        self.cpu_samples = deque(maxlen=cpu_sample_window)
        
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
        
        # Performance mode tracking for adaptive systems
        self.current_performance_mode = "normal"  # Options: "normal", "efficient", "critical"
        self.performance_change_callbacks: List[Callable[[str, Dict[str, float]], None]] = []
        
        # Initialize last CPU check time
        self._last_cpu_check = 0
        
        # Resource history tracking
        self._history_enabled = high_memory_mode
        if self._history_enabled:
            self._cpu_history = deque(maxlen=60)  # 1 minute of history at 1Hz
            self._memory_history = deque(maxlen=60)
            self._temperature_history = deque(maxlen=60)
    
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
        last_minor_update = time.time()
        last_major_update = time.time()
        last_publish = time.time()
        
        # Adjust update intervals based on performance mode to prevent overhead
        minor_interval = 0.25  # 4Hz for CPU-only sampling 
        major_interval = 2.0   # 0.5Hz for full system check
        
        while self.running:
            try:
                current_time = time.time()
                
                # Fast, lightweight CPU check
                if current_time - last_minor_update >= minor_interval:
                    self._update_cpu_metrics()
                    last_minor_update = current_time
                    self._check_cpu_threshold()
                    
                # Less frequent full system check
                if current_time - last_major_update >= major_interval:
                    self._update_all_metrics()
                    last_major_update = current_time
                    
                    # Check all thresholds
                    self._check_thresholds()
                    
                    # Check if performance mode should change
                    self._update_performance_mode()
                
                if self.enable_publication and current_time - last_publish >= self.publish_interval:
                    self._publish_metrics()
                    last_publish = current_time
                    
                # Adaptive sleep based on CPU load to minimize overhead
                # When CPU is high, monitor less frequently
                if self.cpu_percent > 80:
                    sleep_time = 0.5  # 2Hz when CPU is high
                else:
                    sleep_time = 0.1  # 10Hz when CPU is normal
                    
                time.sleep(sleep_time)
                
            except Exception as e:
                if self.node:
                    self.node.get_logger().error(f"Error in resource monitoring: {str(e)}")
                time.sleep(5.0)  # Longer sleep after error
    
    def _update_cpu_metrics(self):
        """Update just CPU metrics (lightweight)."""
        try:
            # Get current CPU usage
            current_cpu = psutil.cpu_percent()
            
            # Add to samples
            self.cpu_samples.append(current_cpu)
            
            # Calculate average CPU usage
            self.cpu_percent = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else current_cpu
            
            # Add to history if enabled
            if self._history_enabled:
                self._cpu_history.append((time.time(), self.cpu_percent))
                
            self._last_cpu_check = time.time()
            
            # Update per-core stats if in high memory mode (but less frequently)
            if self.high_memory_mode and hasattr(self, 'cpu_per_core') and time.time() % 5 < 0.1:
                self.cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
                
        except Exception as e:
            if self.node:
                self.node.get_logger().debug(f"Failed to update CPU metrics: {str(e)}")
    
    def _update_all_metrics(self):
        """Update all resource metrics."""
        try:
            # Memory usage (already lightweight so we keep it here)
            memory = psutil.virtual_memory()
            self.memory_percent = memory.percent
            
            # Add to history if enabled
            if self._history_enabled:
                self._memory_history.append((time.time(), self.memory_percent))
            
            # More detailed memory info in high memory mode
            if self.high_memory_mode and hasattr(self, 'memory_details'):
                self.memory_details = {
                    'total': memory.total / (1024 * 1024),  # MB
                    'available': memory.available / (1024 * 1024),  # MB
                    'used': memory.used / (1024 * 1024),  # MB
                    'percent': memory.percent
                }
                
                # Get network I/O stats (less frequently)
                if time.time() % 10 < 0.1:  # Only every ~10 seconds
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
                
                # Add to history if enabled
                if self._history_enabled:
                    self._temperature_history.append((time.time(), self.temperature))
                
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
            
            # Include performance mode in status
            performance_indicator = ""
            if self.current_performance_mode != "normal":
                performance_indicator = f"[{self.current_performance_mode.upper()}] "
            
            # Include more details for high memory mode
            if self.high_memory_mode and hasattr(self, 'memory_details'):
                status_msg.data = (
                    f"{performance_indicator}CPU: {self.cpu_percent:.1f}% | "
                    f"Memory: {self.memory_percent:.1f}% "
                    f"({self.memory_details['used']:.0f}/{self.memory_details['total']:.0f} MB) | "
                    f"Temp: {self.temperature:.1f}°C"
                )
            else:
                status_msg.data = (
                    f"{performance_indicator}CPU: {self.cpu_percent:.1f}% | "
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
                    'performance_mode': self.current_performance_mode
                }
                
                # Add network and disk info if available
                if hasattr(self, 'network_io'):
                    detailed_data['network'] = self.network_io
                    
                if hasattr(self, 'disk_io'):
                    detailed_data['disk'] = self.disk_io
                
                detailed_msg.data = json.dumps(detailed_data)
                self.detailed_publisher.publish(detailed_msg)
            
        except Exception as e:
            if self.node:
                self.node.get_logger().error(f"Failed to publish resource metrics: {str(e)}")
    
    def _check_cpu_threshold(self):
        """Quick check for just CPU threshold to enable fast response."""
        if self.cpu_percent > self.thresholds['cpu']:
            self._trigger_alert('cpu', self.cpu_percent)
    
    def _check_thresholds(self):
        """Check if any metrics exceed their thresholds and trigger callbacks."""
        # CPU is checked separately in _check_cpu_threshold
        # for faster response, so we don't duplicate here
            
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
    
    def _update_performance_mode(self):
        """
        Update the performance mode based on system metrics.
        
        Performance modes:
        - normal: All resources within reasonable limits
        - efficient: Some resources approaching limits, reduce usage
        - critical: Resources at critical levels, minimize usage
        """
        old_mode = self.current_performance_mode
        new_mode = old_mode
        
        # Determine the new performance mode
        if self.temperature >= self.thresholds['temperature'] or self.cpu_percent >= 95.0:
            new_mode = "critical"
        elif self.cpu_percent >= self.thresholds['cpu'] or self.memory_percent >= self.thresholds['memory']:
            new_mode = "efficient"
        else:
            new_mode = "normal"
        
        # If mode changed, notify callbacks
        if new_mode != old_mode:
            self.current_performance_mode = new_mode
            
            # Create metrics dict for callbacks
            metrics = {
                'cpu': self.cpu_percent,
                'memory': self.memory_percent,
                'temperature': self.temperature
            }
            
            for callback in self.performance_change_callbacks:
                try:
                    callback(new_mode, metrics)
                except Exception as e:
                    if self.node:
                        self.node.get_logger().error(f"Error in performance mode callback: {str(e)}")
            
            # Log the performance mode change
            if self.node:
                self.node.get_logger().info(
                    f"Performance mode changed: {old_mode} -> {new_mode} "
                    f"(CPU: {self.cpu_percent:.1f}%, Memory: {self.memory_percent:.1f}%, "
                    f"Temp: {self.temperature:.1f}°C)"
                )
    
    def add_alert_callback(self, callback: Callable[[str, float], None]):
        """
        Add a callback to be called when a threshold is exceeded.
        
        Args:
            callback: Function taking (resource_type, value) arguments
        """
        self.alert_callbacks.append(callback)
    
    def add_performance_mode_callback(self, callback: Callable[[str, Dict[str, float]], None]):
        """
        Add a callback to be called when performance mode changes.
        
        Args:
            callback: Function taking (new_mode, metrics) arguments
                new_mode: 'normal', 'efficient', or 'critical'
                metrics: Dict with current 'cpu', 'memory', and 'temperature' values
        """
        self.performance_change_callbacks.append(callback)
    
    def remove_callback(self, callback: Callable):
        """
        Remove a previously added callback.
        
        Args:
            callback: The callback function to remove
        """
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            
        if callback in self.performance_change_callbacks:
            self.performance_change_callbacks.remove(callback)
    
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
            'timestamp': time.time(),
            'performance_mode': self.current_performance_mode
        }
    
    def get_cpu_average(self, seconds: float = 5.0) -> float:
        """
        Get the average CPU usage over the specified time window.
        
        Args:
            seconds: Time window in seconds
            
        Returns:
            Average CPU percentage over the time window
        """
        if not self._history_enabled:
            # If history isn't enabled, return current average
            return self.cpu_percent
            
        now = time.time()
        window_start = now - seconds
        
        # Filter CPU history to the specified window
        relevant_samples = [(t, cpu) for t, cpu in self._cpu_history if t >= window_start]
        
        if not relevant_samples:
            return self.cpu_percent
            
        # Calculate average
        avg = sum(cpu for _, cpu in relevant_samples) / len(relevant_samples)
        return avg
    
    def get_trend(self, metric: str = 'cpu', seconds: float = 30.0) -> Dict[str, Any]:
        """
        Calculate trend information for a specific metric.
        
        Args:
            metric: 'cpu', 'memory', or 'temperature'
            seconds: Time window in seconds
            
        Returns:
            Dictionary with trend information
        """
        if not self._history_enabled:
            return {'trend': 'unknown', 'rate': 0.0}
            
        # Select the appropriate history
        if metric == 'cpu':
            history = self._cpu_history
        elif metric == 'memory':
            history = self._memory_history
        elif metric == 'temperature':
            history = self._temperature_history
        else:
            return {'trend': 'unknown', 'rate': 0.0}
            
        now = time.time()
        window_start = now - seconds
        
        # Filter to the specified window
        relevant_samples = [(t, value) for t, value in history if t >= window_start]
        
        if len(relevant_samples) < 2:
            return {'trend': 'stable', 'rate': 0.0}
            
        # Calculate trend
        first_time, first_value = relevant_samples[0]
        last_time, last_value = relevant_samples[-1]
        
        if last_time == first_time:
            rate = 0.0
        else:
            rate = (last_value - first_value) / (last_time - first_time)
        
        # Determine trend direction
        if abs(rate) < 0.1:  # Less than 0.1% change per second is considered stable
            trend = 'stable'
        elif rate > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
            
        return {
            'trend': trend,
            'rate': rate,
            'unit': 'percent/second',
            'window_seconds': seconds,
            'start_value': first_value,
            'end_value': last_value
        }
    
    def is_system_stable(self) -> Tuple[bool, str]:
        """
        Check if the system resources are stable.
        
        Returns:
            Tuple of (stable, reason)
            - stable: True if resources are stable, False otherwise
            - reason: Explanation if not stable
        """
        # Check for high values
        if self.temperature > self.thresholds['temperature']:
            return False, f"High temperature: {self.temperature:.1f}°C"
            
        if self.cpu_percent > self.thresholds['cpu']:
            # Check if CPU is trending up or stable at high level
            cpu_trend = self.get_trend('cpu', 10.0)
            if cpu_trend['trend'] == 'increasing':
                return False, f"CPU increasing: {self.cpu_percent:.1f}% and rising"
            return False, f"High CPU: {self.cpu_percent:.1f}%"
            
        if self.memory_percent > self.thresholds['memory']:
            return False, f"High memory: {self.memory_percent:.1f}%"
            
        # All resources within thresholds
        return True, "System stable"


def print_alert(resource_type: str, value: float):
    """Example alert callback that prints to console."""
    print(f"ALERT: {resource_type.upper()} usage at {value:.1f}% exceeds threshold!")


def performance_mode_changed(mode: str, metrics: Dict[str, float]):
    """Example performance mode callback."""
    print(f"Performance mode changed to {mode.upper()} - CPU: {metrics['cpu']:.1f}%, "
          f"Memory: {metrics['memory']:.1f}%, Temp: {metrics['temperature']:.1f}°C")
    

# Simple usage example when run as a standalone script
if __name__ == "__main__":
    print("Starting resource monitor standalone test...")
    monitor = ResourceMonitor(node=None, enable_publication=False)
    monitor.add_alert_callback(print_alert)
    monitor.add_performance_mode_callback(performance_mode_changed)
    monitor.start()
    
    try:
        # Run for 30 seconds
        for i in range(30):
            metrics = monitor.get_current_metrics()
            mode = metrics['performance_mode']
            print(f"[{mode.upper()}] CPU: {metrics['cpu_percent']:.1f}% | "
                  f"Memory: {metrics['memory_percent']:.1f}% | "
                  f"Temperature: {metrics['temperature']:.1f}°C")
            time.sleep(1)
    finally:
        monitor.stop()
        print("Resource monitor test completed")
