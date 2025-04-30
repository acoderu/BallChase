import time
import math
import numpy as np  # Still needed for some operations but usage is minimized
import psutil
import logging
import threading
from collections import deque
from functools import lru_cache

class CircularBuffer:
    """Generic fixed-size circular buffer for any data type."""
    def __init__(self, max_size, default=None):
        self.data = [default] * max_size
        self.max_size = max_size
        self.next_index = 0
        self.count = 0

    def __len__(self):
        return self.count

    def add(self, value):
        self.data[self.next_index] = value
        self.next_index = (self.next_index + 1) % self.max_size
        self.count = min(self.count + 1, self.max_size)

    def get_all(self):
        if self.count == 0:
            return []
        if self.count < self.max_size:
            return self.data[:self.count]
        return self.data[self.next_index:] + self.data[:self.next_index]

    def get_latest(self, n=1):
        if self.count == 0:
            return []
        n = min(n, self.count)
        result = []
        for i in range(n):
            idx = (self.next_index - 1 - i) % self.max_size
            result.append(self.data[idx])
        return result[::-1]

    def clear(self):
        self.next_index = 0
        self.count = 0
        self.data = [self.data[0]] * self.max_size

# Optionally, alias LightweightBuffer to CircularBuffer for backward compatibility
LightweightBuffer = CircularBuffer

class TTLDict:
    """
    Dictionary with time-to-live functionality for entries.
    Automatically cleans up expired entries during access with lazy cleanup.
    """
    def __init__(self, default_ttl=60.0, cleanup_threshold=100):
        """
        Initialize a TTL dictionary.
        
        Args:
            default_ttl (float): Default TTL in seconds for entries
            cleanup_threshold (int): Number of operations before full cleanup
        """
        self.data = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
        self.ttls = {}  # Store custom TTLs
        self.operation_count = 0
        self.cleanup_threshold = cleanup_threshold
    
    def __setitem__(self, key, value, ttl=None):
        """Set an item with a specific TTL."""
        self.data[key] = value
        self.timestamps[key] = time.time()
        # Store TTL with the key if provided
        if ttl is not None:
            self.ttls[key] = ttl
        
        # Increment operation counter and check if full cleanup needed
        self.operation_count += 1
        if self.operation_count >= self.cleanup_threshold:
            self._cleanup()
            self.operation_count = 0
    
    def set(self, key, value, ttl=None):
        """Set an item with an optional custom TTL."""
        self.__setitem__(key, value, ttl)
    
    def __getitem__(self, key):
        """Get an item, removing if expired."""
        self._cleanup_key(key)
        if key in self.data:
            return self.data[key]
        raise KeyError(key)
    
    def get(self, key, default=None):
        """Get an item with a default value if missing or expired."""
        self._cleanup_key(key)
        return self.data.get(key, default)
    
    def _cleanup_key(self, key):
        """Clean up a specific key if expired."""
        current_time = time.time()
        
        if key in self.timestamps:
            ttl = self.ttls.get(key, self.default_ttl)
            if current_time - self.timestamps[key] > ttl:
                del self.data[key]
                del self.timestamps[key]
                if key in self.ttls:
                    del self.ttls[key]
    
    def _cleanup(self):
        """Clean up all expired entries efficiently."""
        current_time = time.time()
        
        # Use a direct iteration over a copy of keys to avoid modification during iteration
        for k in list(self.timestamps.keys()):
            ttl = self.ttls.get(k, self.default_ttl)
            if current_time - self.timestamps[k] > ttl:
                del self.data[k]
                del self.timestamps[k]
                if k in self.ttls:
                    del self.ttls[k]
    
    def cleanup_all(self):
        """Force cleanup of all expired entries."""
        self._cleanup()
    
    def __contains__(self, key):
        """Check if key exists and is not expired."""
        self._cleanup_key(key)
        return key in self.data


class Matrix4x4:
    """
    Efficient 4x4 matrix implementation for transform operations.
    Uses standard Python lists instead of NumPy for better performance on Raspberry Pi.
    """
    def __init__(self):
        """Initialize identity matrix using standard Python list."""
        # Initialize as identity matrix (row-major order)
        self.data = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    
    @classmethod
    def from_tf_transform(cls, transform):
        """
        Create matrix from ROS transform.
        
        Args:
            transform: ROS Transform message
        
        Returns:
            Matrix4x4: New matrix with transform data
        """
        matrix = cls()
        
        # Extract quaternion
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        
        # Pre-compute quaternion products once and store them
        # These will be reused for all transformations
        matrix.qx_qx = qx * qx
        matrix.qx_qy = qx * qy
        matrix.qx_qz = qx * qz
        matrix.qx_qw = qx * qw
        matrix.qy_qy = qy * qy
        matrix.qy_qz = qy * qz
        matrix.qy_qw = qy * qw
        matrix.qz_qz = qz * qz
        matrix.qz_qw = qz * qw
        
        # Fill rotation part (3x3 top-left)
        matrix.data[0][0] = 1.0 - 2.0 * (matrix.qy_qy + matrix.qz_qz)
        matrix.data[0][1] = 2.0 * (matrix.qx_qy - matrix.qz_qw)
        matrix.data[0][2] = 2.0 * (matrix.qx_qz + matrix.qy_qw)
        
        matrix.data[1][0] = 2.0 * (matrix.qx_qy + matrix.qz_qw)
        matrix.data[1][1] = 1.0 - 2.0 * (matrix.qx_qx + matrix.qz_qz)
        matrix.data[1][2] = 2.0 * (matrix.qy_qz - matrix.qx_qw)
        
        matrix.data[2][0] = 2.0 * (matrix.qx_qz - matrix.qy_qw)
        matrix.data[2][1] = 2.0 * (matrix.qy_qz + matrix.qx_qw)
        matrix.data[2][2] = 1.0 - 2.0 * (matrix.qx_qx + matrix.qy_qy)
        
        # Fill translation part (right column)
        matrix.data[0][3] = transform.transform.translation.x
        matrix.data[1][3] = transform.transform.translation.y
        matrix.data[2][3] = transform.transform.translation.z
        
        return matrix
    
    # Add LRU cache for repeated transformations of the same point
    @lru_cache(maxsize=32)
    def transform_point(self, x, y, z):
        """
        Transform a 3D point using this matrix.
        Cached for efficiency when transforming the same points repeatedly.
        
        Args:
            x, y, z (float): Point coordinates
        
        Returns:
            tuple: Transformed (x, y, z) coordinates
        """
        # Apply transformation
        tx = self.data[0][0] * x + self.data[0][1] * y + self.data[0][2] * z + self.data[0][3]
        ty = self.data[1][0] * x + self.data[1][1] * y + self.data[1][2] * z + self.data[1][3]
        tz = self.data[2][0] * x + self.data[2][1] * y + self.data[2][2] * z + self.data[2][3]
        
        return (tx, ty, tz)
    
    @lru_cache(maxsize=32)
    def transform_vector(self, x, y, z):
        """
        Transform a 3D vector using this matrix (no translation).
        Cached for efficiency when transforming the same vectors repeatedly.
        
        Args:
            x, y, z (float): Vector components
        
        Returns:
            tuple: Transformed (x, y, z) vector
        """
        # Apply rotation only
        tx = self.data[0][0] * x + self.data[0][1] * y + self.data[0][2] * z
        ty = self.data[1][0] * x + self.data[1][1] * y + self.data[1][2] * z
        tz = self.data[2][0] * x + self.data[2][1] * y + self.data[2][2] * z
        
        return (tx, ty, tz)


class ResourceMonitor:
    """
    Lightweight resource monitor for tracking CPU and memory usage.
    Non-blocking implementation using background thread for resource monitoring.
    """
    
    def __init__(self, logger, update_interval=5.0):
        """
        Initialize the resource monitor.
        
        Args:
            logger: Logger instance to use for logging
            update_interval: How often to update resource metrics (seconds)
        """
        self.logger = logger
        self.update_interval = update_interval
        self.last_update_time = 0
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.alert_callback = None  # Single callback function for efficiency
        self.cpu_threshold = 85.0  # Default CPU usage threshold (%)
        self.memory_threshold = 85.0  # Default memory usage threshold (%)
        self.running = False
        self._monitor_thread = None
    
    def start(self):
        """Start background monitoring thread."""
        if not self.running:
            self.running = True
            self._monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
            self._monitor_thread.start()
    
    def stop(self):
        """Stop background monitoring thread."""
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _background_monitor(self):
        """Background thread for non-blocking resource monitoring."""
        while self.running:
            # Update resource metrics
            self.cpu_usage = psutil.cpu_percent(interval=None)  # Non-blocking
            memory = psutil.virtual_memory()
            self.memory_usage = memory.percent
            
            # Check thresholds for alerts
            self._check_thresholds()
            
            # Sleep for update interval
            time.sleep(self.update_interval)
    
    def update(self):
        """
        Legacy update method for compatibility.
        Not needed if using background thread mode.
        """
        # If not using background thread, force an update
        if not self.running:
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                self.cpu_usage = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                self.memory_usage = memory.percent
                self._check_thresholds()
                self.last_update_time = current_time
    
    def _check_thresholds(self):
        """Check if any resource metrics exceed thresholds and trigger callback."""
        if not self.alert_callback:
            return
            
        alerts = {}
        
        # Check CPU
        if self.cpu_usage > self.cpu_threshold:
            alerts['cpu'] = self.cpu_usage
        
        # Check memory
        if self.memory_usage > self.memory_threshold:
            alerts['memory'] = self.memory_usage
        
        # Call callback once with all alerts
        if alerts:
            self.alert_callback(alerts)
    
    def set_alert_callback(self, callback):
        """
        Set a callback to be called when resource thresholds are exceeded.
        
        Args:
            callback: Function to call with alerts dictionary {resource_type: value}
        """
        self.alert_callback = callback
    
    def add_alert_callback(self, callback):
        """
        Legacy method for compatibility.
        Wraps individual resource callbacks to work with new system.
        
        Args:
            callback: Function to call with (resource_type, value) parameters
        """
        # Create a wrapper that converts new format to old format
        def wrapper(alerts):
            for resource_type, value in alerts.items():
                callback(resource_type, value)
        
        self.alert_callback = wrapper
    
    def get_cpu_usage(self):
        """Get the last measured CPU usage."""
        return self.cpu_usage
    
    def get_memory_usage(self):
        """Get the last measured memory usage."""
        return self.memory_usage
        
    def log_stats(self):
        """Log current resource statistics efficiently."""
        # Use debug_level gating instead of isEnabledFor
        if hasattr(self, 'debug_level') and self.debug_level >= 2:
            self.logger.info(f"CPU: {self.cpu_usage:.1f}%, Memory: {self.memory_usage:.1f}%", throttle_duration_sec=2.0)

class ThrottledLogger:
    """Logger wrapper that supports throttled logging to avoid log spam."""
    def __init__(self, logger):
        self.logger = logger
        self._last_log_times = {}
        self._lock = threading.Lock()

    def info(self, msg, throttle_duration_sec=None, log_id=None):
        self._log('info', msg, throttle_duration_sec, log_id)

    def warning(self, msg, throttle_duration_sec=None, log_id=None):
        self._log('warning', msg, throttle_duration_sec, log_id)

    def debug(self, msg, throttle_duration_sec=None, log_id=None):
        self._log('debug', msg, throttle_duration_sec, log_id)

    def error(self, msg, throttle_duration_sec=None, log_id=None):
        self._log('error', msg, throttle_duration_sec, log_id)

    def _log(self, level, msg, throttle_duration_sec, log_id):
        if throttle_duration_sec is None:
            getattr(self.logger, level)(msg)
            return
        # Use log_id or msg as the key
        key = log_id if log_id is not None else msg
        now = time.time()
        with self._lock:
            last_time = self._last_log_times.get(key, 0)
            if now - last_time >= throttle_duration_sec:
                getattr(self.logger, level)(msg)
                self._last_log_times[key] = now