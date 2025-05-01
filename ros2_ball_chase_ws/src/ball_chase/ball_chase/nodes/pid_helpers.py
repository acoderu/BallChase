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
    """Unified resource monitor for tracking CPU and memory usage with alerting and stats."""
    def __init__(self, logger, update_interval=5.0, debug_level=0):
        self.logger = logger
        self.update_interval = update_interval
        self.debug_level = debug_level
        self.last_update_time = 0
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.current_cpu_usage = 0.0
        self.current_memory_usage = 0.0
        self.alert_callback = None
        self.cpu_threshold = 85.0
        self.memory_threshold = 85.0
        self.running = False
        self._monitor_thread = None
        self._cycle_stats = []
        self._max_cycle_stats = 100
        self._performance_stats = {
            'cpu_avg': 0.0,
            'cycle_time_ms': 0.0,
            'update_rate': 0.0,
            'skips': 0
        }

    def start(self):
        if not self.running:
            self.running = True
            self._monitor_thread = threading.Thread(target=self._background_monitor, daemon=True)
            self._monitor_thread.start()

    def stop(self):
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
            self._monitor_thread = None

    def _background_monitor(self):
        while self.running:
            self.update_cpu_stats()
            time.sleep(self.update_interval)

    def update_cpu_stats(self):
        self.cpu_usage = psutil.cpu_percent(interval=None)
        self.memory_usage = psutil.virtual_memory().percent
        self.current_cpu_usage = self.cpu_usage
        self.current_memory_usage = self.memory_usage
        self._check_thresholds()
        if self.debug_level >= 2:
            self.logger.info(
                f"CPU: {self.cpu_usage:.1f}%, Memory: {self.memory_usage:.1f}%",
                throttle_duration_sec=5.0,  
                log_id="resource_monitor_stats"  # <--- static log_id ensures throttling works!
            )

    def _check_thresholds(self):
        if not self.alert_callback:
            return
        alerts = {}
        if self.cpu_usage > self.cpu_threshold:
            alerts['cpu'] = self.cpu_usage
        if self.memory_usage > self.memory_threshold:
            alerts['memory'] = self.memory_usage
        if alerts:
            self.alert_callback(alerts)

    def set_alert_callback(self, callback):
        self.alert_callback = callback

    def add_alert_callback(self, callback):
        def wrapper(alerts):
            for resource_type, value in alerts.items():
                callback(resource_type, value)
        self.alert_callback = wrapper

    def set_cpu_thresholds(self, low_threshold, high_threshold):
        self.cpu_threshold = high_threshold
        # Optionally store low_threshold for future use

    def set_rate_limits(self, min_rate, max_rate, base_rate):
        self._performance_stats['update_rate'] = base_rate

    def set_fusion_rate(self, fusion_rate):
        self._performance_stats['update_rate'] = fusion_rate

    def should_skip_cycle(self):
        return self.cpu_usage > self.cpu_threshold

    def get_cpu_usage(self):
        return self.cpu_usage

    def get_memory_usage(self):
        return self.memory_usage

    def _update_cycle_stats(self, cycle_duration):
        self._cycle_stats.append(cycle_duration)
        if len(self._cycle_stats) > self._max_cycle_stats:
            self._cycle_stats.pop(0)
        avg_cycle = sum(self._cycle_stats) / len(self._cycle_stats) * 1000.0 if self._cycle_stats else 0.0
        self._performance_stats['cycle_time_ms'] = avg_cycle

    def get_performance_stats(self):
        # Optionally update CPU avg
        self._performance_stats['cpu_avg'] = self.cpu_usage
        return self._performance_stats

    def log_stats(self):
        if self.debug_level >= 2:
            self.logger.info(f"CPU: {self.cpu_usage:.1f}%, Memory: {self.memory_usage:.1f}%", throttle_duration_sec=5.0)

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

# Trigonometric optimization - pre-computed values for common angles
# Using 1-degree increments for reasonable accuracy/memory tradeoff
class FastTrigonometry:
    """Optimized trigonometric functions using look-up tables and approximations."""
    
    def __init__(self):
        """Initialize lookup tables for common angles."""
        # Create lookup tables with 1-degree increments from -180 to 180 degrees
        self.angles_rad = np.radians(np.arange(-180, 181, 1))
        self.sin_table = np.sin(self.angles_rad)
        self.cos_table = np.cos(self.angles_rad)
        
        # Small angle threshold in radians (approximately 5 degrees)
        self.small_angle_threshold = 0.087  # ~5 degrees
        
    def sin(self, angle_rad):
        """Fast sine calculation using lookup table and small angle approximation."""
        # Small angle approximation for very small angles
        if abs(angle_rad) < self.small_angle_threshold:
            return angle_rad  # sin(x) ≈ x for small x
        
        # Normalize angle to -π to π range
        angle_rad = (angle_rad + math.pi) % (2 * math.pi) - math.pi
        
        # Convert to degrees and find nearest index
        angle_deg = round(math.degrees(angle_rad))
        # Ensure index is within bounds
        index = max(-180, min(180, angle_deg)) + 180
        
        return self.sin_table[index]
    
    def cos(self, angle_rad):
        """Fast cosine calculation using lookup table and small angle approximation."""
        # Small angle approximation for very small angles
        if abs(angle_rad) < self.small_angle_threshold:
            return 1.0 - (angle_rad * angle_rad) / 2.0  # cos(x) ≈ 1 - x²/2 for small x
        
        # Normalize angle to -π to π range
        angle_rad = (angle_rad + math.pi) % (2 * math.pi) - math.pi
        
        # Convert to degrees and find nearest index
        angle_deg = round(math.degrees(angle_rad))
        # Ensure index is within bounds
        index = max(-180, min(180, angle_deg)) + 180
        
        return self.cos_table[index]
    
    def atan2(self, y, x):
        """Fast implementation of atan2 using lookup tables for common cases."""
        # Handle special cases
        if abs(x) < 1e-10:  # x is close to zero
            return math.pi/2 if y > 0 else -math.pi/2 if y < 0 else 0.0
        
        # Use standard atan2 for non-common cases
        # This could be further optimized with a 2D lookup table if needed
        return math.atan2(y, x)

class StateController:
    """Centralized state controller for robot and controller state."""
    def __init__(self):
        # Robot state
        self.robot_state = "initializing"
        self.previous_state = None
        self.last_control_time = 0.0
        self.robot_orientation = 0.0
        self.last_orientation_time = None
        self._last_state_change_time = 0.0
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        self.last_logged_cmd = (0.0, 0.0, 0.0)
        self.cycle_count = 0
        self.last_pool_log_time = 0.0
        self._robot_stopped = True
        self._stop_time = 0.0
        self._last_stop_position = (0.0, 0.0, 0.0)
        self._movement_hysteresis = 0.0
        self.in_recovery = False
        self.recovery_start_time = 0.0
        self.recovery_phase = "none"
        self.force_target_reacquisition = False
        self._shutting_down = False
        self._using_simplified_control = False
        self._computation_level = 3
        self._last_full_computation_time = 0.0
        self._simplified_control_count = 0
        self._data_freshness_level = "unknown"
        self._last_timer_execution = 0.0
        self._last_event_execution = 0.0
        self._event_control_count = 0
        self._timer_control_count = 0
        self._last_cpu_check_time = 0.0
        self.last_cpu_warning_time = 0.0
        self._skipped_cycle_count = 0
        self._freshness_state_change_time = 0.0
        self._last_logged_rate = 0.0
        self._current_rate = 0.0
        self._adaptive_rate_history = []
        self._initial_movement_boost = False
        self._prev_velocities = None
        self._target_velocities = None
        self._vel_diffs = None
        self._velocity_tuple = None
        self._velocity_change_check = None
        # Add more shared state as needed

class GenericObjectPool:
    """Generic object pool for any message type, with TTL-based cleanup and max size."""
    def __init__(self, cls, max_size=10, reset_fn=None, ttl=60.0):
        self.cls = cls
        self.max_size = max_size
        self.reset_fn = reset_fn
        self.ttl = ttl  # seconds
        self.pool = []  # List of (obj, timestamp)
        self.misses = 0
        self.max_usage = 0
        now = time.time()
        for _ in range(max_size):
            self.pool.append((cls(), now))

    def get(self):
        now = time.time()
        self._cleanup(now)
        if not self.pool:
            self.misses += 1
            return self.cls()
        self.max_usage = max(self.max_usage, self.max_size - len(self.pool))
        obj, _ = self.pool.pop()
        if self.reset_fn:
            self.reset_fn(obj)
        return obj

    def put(self, obj):
        now = time.time()
        self._cleanup(now)
        if len(self.pool) < self.max_size:
            self.pool.append((obj, now))
        # If full, discard the object

    def _cleanup(self, now=None):
        if now is None:
            now = time.time()
        self.pool = [(obj, ts) for (obj, ts) in self.pool if now - ts < self.ttl]

    def stats(self):
        return {
            'pool_size': len(self.pool),
            'misses': self.misses,
            'max_usage': self.max_usage
        }