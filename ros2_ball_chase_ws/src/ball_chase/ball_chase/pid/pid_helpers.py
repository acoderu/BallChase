"""
Basketball Tracking Robot - PID Helper Utilities
===============================================

EDUCATIONAL DOCUMENTATION
------------------------

This module provides optimized utility classes for the basketball tracking robot's 
control system. These utilities are designed for efficient operation on resource-constrained
systems like the Raspberry Pi. This file serves as a foundation for understanding how
real-world robotics systems handle performance optimization.

Key Concepts for Beginners:
-------------------------

1. MEMORY MANAGEMENT IN ROBOTICS

   Robotics systems must process data continuously in real-time. Efficient memory
   management is critical for several reasons:
   
   - PREDICTABLE PERFORMANCE: Avoiding unexpected pauses from garbage collection
   - RESOURCE CONSTRAINTS: Embedded systems like Raspberry Pi have limited RAM
   - REAL-TIME REQUIREMENTS: Control systems must respond within milliseconds

   This module demonstrates several memory optimization techniques like object pools,
   circular buffers, and time-to-live (TTL) dictionaries.

2. COMPUTATION OPTIMIZATION

   Robotics applications need to perform many calculations per second. Optimizing
   these calculations is essential:
   
   - TRIGONOMETRY: Sine/cosine calculations are common but computationally expensive
   - MATRIX OPERATIONS: 3D transforms require many matrix operations
   - CACHING: Many calculations can be cached to avoid redundant work

   The FastTrigonometry and Matrix4x4 classes demonstrate these optimization techniques.

3. RESOURCE MONITORING

   Robotics systems must adapt to changing conditions, including their own resource usage:
   
   - CPU MONITORING: Detect when the system is overloaded
   - ADAPTIVE BEHAVIOR: Reduce computation when resources are limited
   - PERFORMANCE STATISTICS: Track system performance over time

   The ResourceMonitor class implements these concepts.

4. THREAD SAFETY

   Robotics systems often use multiple threads to handle different tasks:
   
   - CONCURRENT ACCESS: Multiple components may access the same data
   - RACE CONDITIONS: Can cause unpredictable behavior if not handled properly
   - LOCKING: Properly synchronizing access to shared resources

   The ThrottledLogger demonstrates thread-safe operations.

This module contains the following utility classes:
-------------------------------------------------

- CircularBuffer: Fixed-size buffer for efficient data storage
- TTLDict: Dictionary with automatic expiration of old entries
- Matrix4x4: Optimized 4x4 matrix for coordinate transformations
- ResourceMonitor: CPU and memory usage monitoring
- ThrottledLogger: Prevents log flooding by rate-limiting messages
- FastTrigonometry: Optimized trigonometric functions using lookup tables
- GenericObjectPool: Memory-efficient object reuse system

"""

import time
import math
import numpy as np  # Still needed for some operations but usage is minimized
import psutil
import logging
import threading
from collections import deque
from functools import lru_cache

class CircularBuffer:
    """
    Memory-efficient fixed-size circular buffer implementation.
    
    EDUCATIONAL EXPLANATION:
    -----------------------
    A circular buffer (or ring buffer) is a data structure that:
    
    1. Uses a fixed amount of memory, regardless of how many items you add
    2. Automatically overwrites old data when full
    3. Provides efficient O(1) operations for adding and accessing data
    4. Avoids memory allocations/deallocations common in dynamic arrays
    
    This is ideal for robotics applications where you need to:
    - Store recent sensor readings (like the last 10 positions of a ball)
    - Monitor trends over a fixed time window
    - Implement sliding window algorithms
    - Avoid unpredictable memory allocation delays
    
    VISUALIZATION:
    -------------
    For a buffer of size 4:
                  
       Initial:   [None, None, None, None]  
                   ^next
                   
       Add "A":    [A, None, None, None]
                      ^next
                      
       Add "B":    [A, B, None, None]
                         ^next
                         
       Add "C,D":  [A, B, C, D]
                   ^next        
                   
       Add "E":    [E, B, C, D]
                      ^next
                      
    Note how "A" was overwritten when the buffer was full!
    """
    def __init__(self, max_size, default=None):
        """
        Initialize a fixed-size circular buffer.
        
        Args:
            max_size: Maximum number of items to store
            default: Initial value to fill the buffer with
        
        The buffer pre-allocates all memory upfront to avoid 
        dynamic allocations during operation.
        """
        self.data = [default] * max_size
        self.max_size = max_size
        self.next_index = 0  # Where the next item will be written
        self.count = 0       # How many valid items are in the buffer

    def __len__(self):
        """Return the current number of items in the buffer."""
        return self.count

    def add(self, value):
        """
        Add an item to the buffer, overwriting oldest if full.
        
        This operation is O(1) - constant time regardless of buffer size.
        """
        # Store the value at the current write position
        self.data[self.next_index] = value
        
        # Move the write position for next time, wrapping around if needed
        self.next_index = (self.next_index + 1) % self.max_size
        
        # Update count (but never exceed max_size)
        self.count = min(self.count + 1, self.max_size)

    def get_all(self):
        """
        Return all valid items in the buffer in order (oldest first).
        
        For a partially filled buffer: Returns only the valid items
        For a full buffer: Returns all items in the correct order
        """
        if self.count == 0:
            return []  # Empty buffer
            
        # For partially filled buffers, just return the filled portion
        if self.count < self.max_size:
            return self.data[:self.count]
            
        # For full buffers, create a properly ordered view
        # This handles the "wrapped around" case
        return self.data[self.next_index:] + self.data[:self.next_index]

    def get_latest(self, n=1):
        """
        Return the n most recent items, newest first.
        
        Args:
            n: Number of recent items to return (default=1)
            
        This is useful for analyzing recent trends in data.
        """
        if self.count == 0:
            return []  # Empty buffer
            
        # Limit n to actual item count
        n = min(n, self.count)
        
        result = []
        # Work backwards from the most recently added item
        for i in range(n):
            # Calculate index with wraparound
            idx = (self.next_index - 1 - i) % self.max_size
            result.append(self.data[idx])
            
        # Reverse to get items in chronological order
        return result[::-1]

    def clear(self):
        """
        Reset the buffer to empty state while preserving allocated memory.
        
        This efficiently clears the buffer without deallocating memory.
        """
        self.next_index = 0
        self.count = 0
        # Reuse first item's value to maintain consistent type
        self.data = [self.data[0]] * self.max_size

# Optionally, alias LightweightBuffer to CircularBuffer for backward compatibility
LightweightBuffer = CircularBuffer

class TTLDict:
    """
    Dictionary with automatic time-to-live (TTL) expiration functionality.
    
    EDUCATIONAL EXPLANATION:
    -----------------------
    A TTL Dictionary is a specialized data structure that:
    
    1. Works like a standard dictionary/map for storing key-value pairs
    2. Automatically removes entries after they've been around for a specified time
    3. Uses "lazy cleanup" to avoid performance overhead (only checks expiration when accessed)
    4. Enables efficient caching with automatic memory management
    
    This is particularly valuable in robotics applications for:
    - Caching sensor readings with automatic expiration of stale data
    - Implementing timeout mechanisms for operations
    - Managing temporary values without manual cleanup
    - Preventing memory leaks from forgotten data
    
    REAL-WORLD ANALOGY:
    ------------------
    Think of a TTL Dictionary like a refrigerator where each food item has an expiration date:
    - You put food items in (key-value pairs)
    - Each item has its own expiration date (TTL)
    - When you look for an item, you first check if it's expired
    - Periodically, you clean out all expired items
    
    PRACTICAL EXAMPLE:
    -----------------
    In our basketball tracking robot:
    - Store recent ball detections with 0.5 second TTL
    - If sensor readings are delayed, old values auto-expire
    - The system automatically adapts to changing conditions
    """
    def __init__(self, default_ttl=60.0, cleanup_threshold=100):
        """
        Initialize a TTL dictionary.
        
        Args:
            default_ttl (float): Default time-to-live in seconds for entries
            cleanup_threshold (int): Number of operations before performing a full cleanup
            
        This creates three internal dictionaries:
        - data: Stores the actual key-value pairs
        - timestamps: Records when each item was added
        - ttls: Stores custom TTL values for specific keys
        """
        self.data = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
        self.ttls = {}  # Store custom TTLs
        self.operation_count = 0
        self.cleanup_threshold = cleanup_threshold
    
    def __setitem__(self, key, value, ttl=None):
        """
        Set an item with a specific TTL.
        
        Args:
            key: Dictionary key
            value: Value to store
            ttl: Optional custom TTL in seconds for this specific entry
            
        This method is called when using the dictionary[key] = value syntax.
        It also triggers periodic full cleanup to prevent memory buildup.
        """
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
        """
        Set an item with an optional custom TTL.
        
        This is an alternative to the dictionary[key] = value syntax that
        allows explicitly setting a custom TTL.
        """
        self.__setitem__(key, value, ttl)
    
    def __getitem__(self, key):
        """
        Get an item, removing it if expired.
        
        This method is called when using the dictionary[key] syntax.
        It first checks if the key exists and hasn't expired before returning.
        """
        self._cleanup_key(key)  # Check if this specific key has expired
        if key in self.data:
            return self.data[key]
        raise KeyError(key)
    
    def get(self, key, default=None):
        """
        Get an item with a default value if missing or expired.
        
        This is safer than dictionary[key] as it returns a default value
        rather than raising an exception if the key doesn't exist.
        """
        self._cleanup_key(key)
        return self.data.get(key, default)
    
    def _cleanup_key(self, key):
        """
        Check if a specific key has expired and remove it if so.
        
        This is the "lazy cleanup" approach - we only check
        expiration when a key is accessed, which saves CPU time.
        """
        current_time = time.time()
        
        if key in self.timestamps:
            # Get this key's TTL (or use default if not specified)
            ttl = self.ttls.get(key, self.default_ttl)
            if current_time - self.timestamps[key] > ttl:
                # Key has expired - remove it from all dictionaries
                del self.data[key]
                del self.timestamps[key]
                if key in self.ttls:
                    del self.ttls[key]
    
    def _cleanup(self):
        """
        Clean up all expired entries efficiently.
        
        This performs a full scan of all keys to remove any that have expired.
        We use a list copy to avoid issues with modifying while iterating.
        """
        current_time = time.time()
        
        # Create a copy of keys to avoid modification during iteration
        for k in list(self.timestamps.keys()):
            ttl = self.ttls.get(k, self.default_ttl)
            if current_time - self.timestamps[k] > ttl:
                del self.data[k]
                del self.timestamps[k]
                if k in self.ttls:
                    del self.ttls[k]
    
    def cleanup_all(self):
        """Force cleanup of all expired entries regardless of threshold."""
        self._cleanup()
    
    def __contains__(self, key):
        """
        Check if key exists and has not expired.
        
        This method is called when using the "key in dictionary" syntax.
        It first removes the key if expired before checking existence.
        """
        self._cleanup_key(key)
        return key in self.data


class Matrix4x4:
    """
    Optimized 4x4 transformation matrix for 3D robotics applications.
    
    EDUCATIONAL EXPLANATION:
    -----------------------
    In 3D robotics, we need to transform coordinates between different reference frames:
    - From camera coordinates to robot coordinates
    - From robot coordinates to world coordinates
    - From one sensor's frame to another's
    
    A 4x4 matrix provides a unified way to represent both rotation and translation in 3D space.
    This implementation is optimized for Raspberry Pi by:
    1. Using Python lists instead of NumPy arrays (better for small matrices)
    2. Pre-computing values that will be reused
    3. Caching repeated transformations
    4. Specialized methods for different transformation types
    
    MATRIX STRUCTURE:
    ---------------
    A 4x4 transformation matrix has this structure:
    
    │ R₁₁  R₁₂  R₁₃  Tx │ 
    │ R₂₁  R₂₂  R₂₃  Ty │ 
    │ R₃₁  R₃₂  R₃₃  Tz │ 
    │  0    0    0    1 │ 
    
    Where:
    - R is a 3×3 rotation matrix (top-left)
    - T is a translation vector (right column)
    - Bottom row is always [0,0,0,1] for homogeneous coordinates
    
    QUATERNIONS AND ROTATIONS:
    ------------------------
    This class converts quaternions (a compact way to represent 3D rotations)
    into the rotation matrix elements. Quaternions are used because they:
    - Avoid "gimbal lock" issues that affect Euler angles
    - Require less storage (4 values vs. 9 values)
    - Can be smoothly interpolated
    - Result in numerically stable calculations
    
    The conversion formulas look complex but are derived from quaternion math principles.
    """
    def __init__(self):
        """
        Initialize a new matrix as the identity transformation.
        
        The identity matrix represents "no change" in a transformation:
        - Points transformed by this matrix stay in the same place
        - This provides a starting point for building other transformations
        """
        # Initialize as identity matrix (row-major order)
        self.data = [
            [1.0, 0.0, 0.0, 0.0],  # First row
            [0.0, 1.0, 0.0, 0.0],  # Second row
            [0.0, 0.0, 1.0, 0.0],  # Third row
            [0.0, 0.0, 0.0, 1.0]   # Fourth row
        ]
    
    @classmethod
    def from_tf_transform(cls, transform):
        """
        Create a transformation matrix from a ROS2 transform message.
        
        Args:
            transform: ROS Transform message containing:
                      - translation (x, y, z)
                      - rotation as quaternion (x, y, z, w)
        
        Returns:
            Matrix4x4: New matrix representing this transform
            
        Quaternion to Rotation Matrix Conversion:
        ---------------------------------------
        The complex formulas below convert quaternion orientation
        (x,y,z,w) into the 9 elements of a 3x3 rotation matrix.
        
        This is a standard conversion in 3D mathematics and robotics.
        """
        matrix = cls()
        
        # Extract quaternion components from the transform message
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        
        # OPTIMIZATION: Pre-compute quaternion products once
        # These values are used multiple times in the conversion formulas
        # By computing them once and storing, we reduce calculations
        matrix.qx_qx = qx * qx
        matrix.qx_qy = qx * qy
        matrix.qx_qz = qx * qz
        matrix.qx_qw = qx * qw
        matrix.qy_qy = qy * qy
        matrix.qy_qz = qy * qz
        matrix.qy_qw = qy * qw
        matrix.qz_qz = qz * qz
        matrix.qz_qw = qz * qw
        
        # Fill rotation part (3x3 top-left portion of matrix)
        # These are the standard formulas for converting quaternion to rotation matrix
        
        # First row of rotation matrix
        matrix.data[0][0] = 1.0 - 2.0 * (matrix.qy_qy + matrix.qz_qz)
        matrix.data[0][1] = 2.0 * (matrix.qx_qy - matrix.qz_qw)
        matrix.data[0][2] = 2.0 * (matrix.qx_qz + matrix.qy_qw)
        
        # Second row of rotation matrix
        matrix.data[1][0] = 2.0 * (matrix.qx_qy + matrix.qz_qw)
        matrix.data[1][1] = 1.0 - 2.0 * (matrix.qx_qx + matrix.qz_qz)
        matrix.data[1][2] = 2.0 * (matrix.qy_qz - matrix.qx_qw)
        
        # Third row of rotation matrix
        matrix.data[2][0] = 2.0 * (matrix.qx_qz - matrix.qy_qw)
        matrix.data[2][1] = 2.0 * (matrix.qy_qz + matrix.qx_qw)
        matrix.data[2][2] = 1.0 - 2.0 * (matrix.qx_qx + matrix.qy_qy)
        
        # Fill translation part (right column)
        matrix.data[0][3] = transform.transform.translation.x
        matrix.data[1][3] = transform.transform.translation.y
        matrix.data[2][3] = transform.transform.translation.z
        
        return matrix
    
    # OPTIMIZATION: Add LRU (Least Recently Used) cache for repeated transformations
    # This prevents recalculating the same transformation when points are transformed repeatedly
    @lru_cache(maxsize=32)
    def transform_point(self, x, y, z):
        """
        Transform a 3D point using this matrix (applies both rotation AND translation).
        
        Args:
            x, y, z (float): Point coordinates
        
        Returns:
            tuple: Transformed (x, y, z) coordinates
            
        EDUCATIONAL NOTE:
        ---------------
        Matrix-vector multiplication for point transformation works by:
        1. Multiplying each row of the matrix by the input vector
        2. Summing the products to get each output coordinate
        
        Written as equation:
        │ x' │   │ R₁₁  R₁₂  R₁₃  Tx │   │ x │
        │ y' │ = │ R₂₁  R₂₂  R₂₃  Ty │ × │ y │
        │ z' │   │ R₃₁  R₃₂  R₃₃  Tz │   │ z │
        │ 1  │   │  0    0    0    1 │   │ 1 │
        
        We're doing the actual calculation directly for speed.
        """
        # Apply full transformation (rotation + translation)
        tx = self.data[0][0] * x + self.data[0][1] * y + self.data[0][2] * z + self.data[0][3]
        ty = self.data[1][0] * x + self.data[1][1] * y + self.data[1][2] * z + self.data[1][3]
        tz = self.data[2][0] * x + self.data[2][1] * y + self.data[2][2] * z + self.data[2][3]
        
        return (tx, ty, tz)
    
    @lru_cache(maxsize=32)
    def transform_vector(self, x, y, z):
        """
        Transform a 3D vector using this matrix (applies ONLY rotation, NOT translation).
        
        Args:
            x, y, z (float): Vector components
        
        Returns:
            tuple: Transformed (x, y, z) vector
            
        EDUCATIONAL NOTE:
        ---------------
        Vectors differ from points because they represent directions, not locations.
        The key difference in transformation:
        - Points get both rotated AND translated
        - Vectors get ONLY rotated (no translation)
        
        Example: If you rotate a robot 90 degrees and move it 1 meter forward:
        - A point (like a physical object location) will be both rotated and moved
        - A vector (like a compass direction or velocity) will only be rotated
        """
        # Apply rotation only (skip the translation component)
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
    """
    Highly optimized trigonometric functions for robotics applications.
    
    EDUCATIONAL EXPLANATION:
    -----------------------
    Trigonometric functions (sin, cos, atan2) are essential in robotics for:
    - Converting between coordinate systems
    - Calculating angles and distances
    - Rotating vectors and positions
    - Control algorithms like PID
    
    However, standard trigonometric functions are computationally expensive:
    - They typically use Taylor series approximations
    - These require many floating-point operations
    - On a resource-constrained system like Raspberry Pi, they can be a bottleneck
    
    This class uses three optimization techniques:
    
    1. LOOKUP TABLES (LUT):
       - Pre-calculate sin/cos values for all angles at 1-degree intervals
       - Store these values in memory
       - When needed, find the closest pre-calculated value
       - Tradeoff: Uses memory to save computation time
    
    2. SMALL-ANGLE APPROXIMATIONS:
       - For very small angles, use mathematical shortcuts:
       - sin(x) ≈ x when x is small
       - cos(x) ≈ 1 - x²/2 when x is small
       - These are accurate enough for many robotics applications
    
    3. SPECIAL CASE HANDLING:
       - Handle common edge cases efficiently (like x=0 in atan2)
       - Avoid unnecessary computation in simple cases
    
    PERFORMANCE IMPACT:
    -----------------
    These optimizations can speed up trigonometric calculations by 5-20x
    on embedded systems like Raspberry Pi, which directly improves:
    - Maximum control loop frequency
    - System responsiveness
    - Available CPU for other tasks
    """
    
    def __init__(self):
        """
        Initialize lookup tables for common angles.
        
        This pre-computes sine and cosine values for all angles
        from -180 to +180 degrees in 1-degree increments.
        
        Memory usage: ~2.9KB (361 float values × 2 tables × 4 bytes)
        This is a reasonable tradeoff for the performance gain.
        """
        # Create lookup tables with 1-degree increments from -180 to 180 degrees
        # We use NumPy here during initialization, but the lookup operations will be pure Python
        self.angles_rad = np.radians(np.arange(-180, 181, 1))
        self.sin_table = np.sin(self.angles_rad)
        self.cos_table = np.cos(self.angles_rad)
        
        # Small angle threshold in radians (approximately 5 degrees)
        # Below this value, we'll use mathematical approximations
        self.small_angle_threshold = 0.087  # ~5 degrees
        
    def sin(self, angle_rad):
        """
        Fast sine calculation using lookup table and small angle approximation.
        
        Args:
            angle_rad (float): Angle in radians
            
        Returns:
            float: Sine of the angle
            
        EDUCATIONAL NOTES:
        ----------------
        The small angle approximation sin(x) ≈ x is from calculus:
        - Taylor series for sin(x) = x - x³/3! + x⁵/5! - ...
        - For small x, the higher terms become negligible
        - For angles < 5°, the error is less than 0.002
        
        This is accurate enough for most robotics control applications.
        """
        # Small angle approximation for very small angles
        if abs(angle_rad) < self.small_angle_threshold:
            return angle_rad  # sin(x) ≈ x for small x
        
        # Normalize angle to -π to π range
        # This handles angles outside our table range by wrapping them
        angle_rad = (angle_rad + math.pi) % (2 * math.pi) - math.pi
        
        # Convert to degrees and find nearest index in lookup table
        angle_deg = round(math.degrees(angle_rad))
        
        # Ensure index is within bounds (should be -180 to 180 + offset of 180)
        index = max(-180, min(180, angle_deg)) + 180
        
        # Return pre-computed value from table
        return self.sin_table[index]
    
    def cos(self, angle_rad):
        """
        Fast cosine calculation using lookup table and small angle approximation.
        
        Args:
            angle_rad (float): Angle in radians
            
        Returns:
            float: Cosine of the angle
            
        EDUCATIONAL NOTES:
        ----------------
        The small angle approximation cos(x) ≈ 1 - x²/2 is from calculus:
        - Taylor series for cos(x) = 1 - x²/2! + x⁴/4! - ...
        - For small x, higher terms become negligible
        - For angles < 5°, error is less than 0.001
        """
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
        """
        Fast implementation of arctangent (atan2).
        
        Args:
            y, x (float): Coordinates of point
            
        Returns:
            float: Angle in radians from origin to point (x,y)
            
        EDUCATIONAL NOTES:
        ----------------
        atan2(y,x) calculates the angle between the positive x-axis and the
        point (x,y), handling all quadrants correctly.
        
        It's commonly used in robotics to:
        - Find the angle to a target
        - Convert from cartesian to polar coordinates
        - Calculate heading directions
        
        This implementation optimizes the common edge case where x=0,
        which would otherwise cause division by zero.
        """
        # Handle special cases where x is zero (would cause division by zero)
        if abs(x) < 1e-10:  # x is close to zero
            return math.pi/2 if y > 0 else -math.pi/2 if y < 0 else 0.0
        
        # For general cases, use the standard function
        # This could be further optimized with a 2D lookup table if needed
        return math.atan2(y, x)

class GenericObjectPool:
    """
    Efficient object reuse system to eliminate costly object creation/destruction cycles.
    
    EDUCATIONAL EXPLANATION:
    -----------------------
    Object pooling is a memory management technique that pre-allocates and recycles objects
    instead of constantly creating and destroying them. This is especially important
    in robotics for several reasons:
    
    1. PERFORMANCE BENEFITS:
       - Creating objects is slow (memory allocation, constructor execution)
       - Garbage collection causes unpredictable pauses
       - Recycling objects avoids both these issues
    
    2. MEMORY FRAGMENTATION PREVENTION:
       - Repeatedly creating/destroying objects causes memory fragmentation
       - Fragmentation degrades performance over time
       - Object pools use a fixed memory region, preventing fragmentation
    
    3. PREDICTABLE RESOURCE USAGE:
       - The pool has a maximum size limit
       - This provides predictable memory usage
       - Essential for real-time systems like robots
    
    REAL-WORLD EXAMPLE:
    -----------------
    Think of object pooling like a library:
    - The library has a fixed number of books (pre-allocated objects)
    - People borrow books (get() from pool) and return them (put() to pool)
    - If all books are borrowed, you might have to wait
    - Books that haven't been borrowed for a long time might be removed
    
    In this basketball robot:
    - We pool message objects for position, velocity, etc.
    - These get reused hundreds of times per second
    - This dramatically reduces CPU usage and prevents pauses
    - The system runs smoother and more responsively
    """
    def __init__(self, cls, max_size=10, reset_fn=None, ttl=60.0):
        """
        Initialize an object pool for a specific class type.
        
        Args:
            cls: The class to create objects from
            max_size: Maximum number of objects to keep in the pool
            reset_fn: Optional function to reset an object before reuse
            ttl: Time-to-live in seconds for unused objects
            
        This pre-allocates max_size objects at initialization to prevent
        allocation delays during operation.
        """
        self.cls = cls                # Class to create objects
        self.max_size = max_size      # Maximum pool size
        self.reset_fn = reset_fn      # Function to reset objects before reuse
        self.ttl = ttl                # Time-to-live for unused objects
        self.pool = []                # List of (object, timestamp) pairs
        self.misses = 0               # Count of times pool was empty when needed
        self.max_usage = 0            # Maximum number of objects used at once
        
        # Pre-allocate objects to avoid allocation during operation
        now = time.time()
        for _ in range(max_size):
            self.pool.append((cls(), now))

    def get(self):
        """
        Get an object from the pool or create a new one if empty.
        
        Returns:
            An instance of the class that can be used
            
        This function:
        1. First cleans up expired objects
        2. Takes an object from the pool if available
        3. Creates a new object if the pool is empty
        4. Resets the object if a reset function was provided
        """
        now = time.time()
        self._cleanup(now)  # Remove expired objects
        
        # If pool is empty, create a new object (and count as a "miss")
        if not self.pool:
            self.misses += 1
            return self.cls()
            
        # Track the maximum usage for statistics
        self.max_usage = max(self.max_usage, self.max_size - len(self.pool))
        
        # Get an object from the pool
        obj, _ = self.pool.pop()
        
        # Reset it if a reset function was provided
        if self.reset_fn:
            self.reset_fn(obj)
            
        return obj

    def put(self, obj):
        """
        Return an object to the pool for future reuse.
        
        Args:
            obj: The object to return to the pool
            
        This function:
        1. First cleans up expired objects to make room
        2. Adds the object to the pool if there's space
        3. Discards the object if the pool is full
        """
        now = time.time()
        self._cleanup(now)  # Remove expired objects
        
        # Only add to the pool if there's space
        if len(self.pool) < self.max_size:
            self.pool.append((obj, now))
        # If the pool is full, the object is discarded and will be garbage collected

    def _cleanup(self, now=None):
        """
        Remove objects that haven't been used for longer than the TTL.
        
        Args:
            now: Current time (or None to use current time)
            
        This prevents the pool from holding onto objects indefinitely,
        which would waste memory if the system's needs change.
        """
        if now is None:
            now = time.time()
            
        # Keep only objects that haven't expired
        self.pool = [(obj, ts) for (obj, ts) in self.pool if now - ts < self.ttl]

    def stats(self):
        """
        Get statistics about the pool's usage.
        
        Returns:
            Dictionary with stats (current size, misses, max usage)
            
        This helps monitor how effectively the pool is being used
        and whether adjustments might be needed.
        """
        return {
            'pool_size': len(self.pool),  # Current number of objects in pool
            'misses': self.misses,        # Times pool was empty when needed
            'max_usage': self.max_usage   # Maximum simultaneous objects in use
        }