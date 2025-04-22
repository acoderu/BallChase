"""
Basketball Tracking Robot - Enhanced PID Controller Node
=======================================================

This controller implements efficient movement patterns for a mecanum-wheeled
basketball tracking robot with several enhancements:
- Coordinated angular-lateral control for coupled motion
- Gradual strategy transitions for smooth state changes
- Adaptive gain system based on tracking conditions
- Target filtering for smoother tracking
- Optimized for Raspberry Pi 5 performance with resource monitoring
- Enhanced transform handling with caching and optimizations
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped, Vector3, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray
from tf2_ros import TransformListener, Buffer
import tf2_geometry_msgs
import math
import time
import numpy as np
import signal
import sys
from collections import deque
import psutil  # For CPU monitoring

# Topic configuration
TOPICS = {
    "input": {
        "target": "/basketball/fused/position",
        "state": "/robot/state",
        "orientation": "/imu/rpy/filtered",  # Orientation from IMU
        "odometry": "/odom"                  # Odometry data
    },
    "output": {
        "cmd_vel": "/controller/cmd_vel",
        "diagnostics": "/pid/diagnostics",
        "performance": "/pid/performance"    # New performance metrics topic
    }
}

# Log throttling parameters
LOG_THROTTLE_CONTROL = 2.0     # Seconds between control loop status logs
LOG_THROTTLE_STATE = 0.5       # Seconds between state change logs
LOG_THROTTLE_DIAG = 1.0        # Seconds between diagnostic logs

# Memory optimization classes
class LightweightBuffer:
    """Memory-efficient buffer for storing historical data with pre-allocated storage."""
    
    def __init__(self, max_size=10, default_value=(0.0, 0.0, 0.0)):
        """Initialize a fixed-size circular buffer."""
        self.data = [default_value] * max_size  # Pre-allocate with default values
        self.next_index = 0
        self.count = 0
        self.max_size = max_size
    
    def add(self, value):
        """Add a new value to the buffer, overwriting oldest if full."""
        self.data[self.next_index] = value
        self.next_index = (self.next_index + 1) % self.max_size
        self.count = min(self.count + 1, self.max_size)
    
    def get_all(self):
        """Get all values currently in the buffer in chronological order."""
        if self.count < self.max_size:
            return self.data[:self.count]
        # Reconstruct in chronological order
        start_idx = self.next_index
        return self.data[start_idx:] + self.data[:start_idx]
        
    def get_latest(self, n=1):
        """Get the latest n values (default is just the latest one)."""
        if self.count == 0:
            return []
        n = min(n, self.count)
        if self.count < self.max_size:
            return self.data[self.count - n:self.count]
        # Calculate indices for the latest n elements
        start_idx = (self.next_index - n) % self.max_size
        if start_idx < self.next_index:
            return self.data[start_idx:self.next_index]
        return self.data[start_idx:] + self.data[:self.next_index]

class TTLDict:
    """
    Dictionary with time-to-live functionality for entries.
    Automatically cleans up expired entries during access.
    """
    def __init__(self, default_ttl=60.0):
        """
        Initialize a TTL dictionary.
        
        Args:
            default_ttl (float): Default TTL in seconds for entries
        """
        self.data = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
    
    def __setitem__(self, key, value, ttl=None):
        """Set an item with a specific TTL."""
        self.data[key] = value
        self.timestamps[key] = time.time()
        # Store TTL with the key if provided
        if ttl is not None:
            if not hasattr(self, 'ttls'):
                self.ttls = {}
            self.ttls[key] = ttl
    
    def set(self, key, value, ttl=None):
        """Set an item with an optional custom TTL."""
        self.__setitem__(key, value, ttl)
    
    def __getitem__(self, key):
        """Get an item, removing if expired."""
        self._cleanup(key)
        if key in self.data:
            return self.data[key]
        raise KeyError(key)
    
    def get(self, key, default=None):
        """Get an item with a default value if missing or expired."""
        self._cleanup(key)
        return self.data.get(key, default)
    
    def _cleanup(self, key=None):
        """Clean up expired entries."""
        current_time = time.time()
        
        # Clean specific key if provided
        if key is not None:
            if key in self.timestamps:
                ttl = self.ttls.get(key, self.default_ttl) if hasattr(self, 'ttls') else self.default_ttl
                if current_time - self.timestamps[key] > ttl:
                    del self.data[key]
                    del self.timestamps[key]
                    if hasattr(self, 'ttls') and key in self.ttls:
                        del self.ttls[key]
            return
        
        # Full cleanup (expensive, use sparingly)
        keys_to_remove = []
        for k, timestamp in self.timestamps.items():
            ttl = self.ttls.get(k, self.default_ttl) if hasattr(self, 'ttls') else self.default_ttl
            if current_time - timestamp > ttl:
                keys_to_remove.append(k)
        
        for k in keys_to_remove:
            del self.data[k]
            del self.timestamps[k]
            if hasattr(self, 'ttls') and k in self.ttls:
                del self.ttls[k]
    
    def cleanup_all(self):
        """Force cleanup of all expired entries."""
        self._cleanup()
    
    def __contains__(self, key):
        """Check if key exists and is not expired."""
        self._cleanup(key)
        return key in self.data

class Matrix4x4:
    """
    Efficient 4x4 matrix implementation for transform operations.
    Optimized for 3D transformations with minimal memory allocations.
    """
    def __init__(self):
        """Initialize identity matrix."""
        # Initialize as identity matrix (row-major order)
        self.data = np.eye(4, dtype=np.float32)
    
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
        
        # Convert quaternion to rotation matrix
        # Precompute common products
        xx = qx * qx
        xy = qx * qy
        xz = qx * qz
        xw = qx * qw
        yy = qy * qy
        yz = qy * qz
        yw = qy * qw
        zz = qz * qz
        zw = qz * qw
        
        # Fill rotation part (3x3 top-left)
        matrix.data[0, 0] = 1.0 - 2.0 * (yy + zz)
        matrix.data[0, 1] = 2.0 * (xy - zw)
        matrix.data[0, 2] = 2.0 * (xz + yw)
        
        matrix.data[1, 0] = 2.0 * (xy + zw)
        matrix.data[1, 1] = 1.0 - 2.0 * (xx + zz)
        matrix.data[1, 2] = 2.0 * (yz - xw)
        
        matrix.data[2, 0] = 2.0 * (xz - yw)
        matrix.data[2, 1] = 2.0 * (yz + xw)
        matrix.data[2, 2] = 1.0 - 2.0 * (xx + yy)
        
        # Fill translation part (right column)
        matrix.data[0, 3] = transform.transform.translation.x
        matrix.data[1, 3] = transform.transform.translation.y
        matrix.data[2, 3] = transform.transform.translation.z
        
        return matrix
    
    def transform_point(self, x, y, z):
        """
        Transform a 3D point using this matrix.
        
        Args:
            x, y, z (float): Point coordinates
        
        Returns:
            tuple: Transformed (x, y, z) coordinates
        """
        # Apply transformation
        tx = self.data[0, 0] * x + self.data[0, 1] * y + self.data[0, 2] * z + self.data[0, 3]
        ty = self.data[1, 0] * x + self.data[1, 1] * y + self.data[1, 2] * z + self.data[1, 3]
        tz = self.data[2, 0] * x + self.data[2, 1] * y + self.data[2, 2] * z + self.data[2, 3]
        
        return (tx, ty, tz)
    
    def transform_vector(self, x, y, z):
        """
        Transform a 3D vector using this matrix (no translation).
        
        Args:
            x, y, z (float): Vector components
        
        Returns:
            tuple: Transformed (x, y, z) vector
        """
        # Apply rotation only
        tx = self.data[0, 0] * x + self.data[0, 1] * y + self.data[0, 2] * z
        ty = self.data[1, 0] * x + self.data[1, 1] * y + self.data[1, 2] * z
        tz = self.data[2, 0] * x + self.data[2, 1] * y + self.data[2, 2] * z
        
        return (tx, ty, tz)

class TargetFilter:
    """Filter for target position data to reduce noise and predict movement."""
    
    def __init__(self, buffer_size=5, prediction_horizon=0.2):
        self.position_buffer = deque(maxlen=buffer_size)
        self.prediction_horizon = prediction_horizon  # seconds
        self.last_update_time = None
        self.current_velocity = (0.0, 0.0, 0.0)  # x, y, angular
        self.filtered_position = None
        self.predicted_position = None
        
    def update(self, position, timestamp=None):
        """
        Update the filter with a new position measurement.
        
        Args:
            position: Tuple of (x, y, angle) for the target position
            timestamp: Time of measurement (defaults to current time)
        """
        current_time = timestamp if timestamp is not None else time.time()
        
        # Add to buffer with timestamp
        self.position_buffer.append((position[0], position[1], position[2], current_time))
        
        # Basic filtering - moving average
        if len(self.position_buffer) >= 3:
            # Simple weighted average with more weight to recent measurements
            weights = [0.2, 0.3, 0.5]  # More weight to recent measurements
            x_sum = sum(pos[0] * w for pos, w in zip(self.get_recent_positions(3), weights))
            y_sum = sum(pos[1] * w for pos, w in zip(self.get_recent_positions(3), weights))
            angle_sum = sum(pos[2] * w for pos, w in zip(self.get_recent_positions(3), weights))
            self.filtered_position = (x_sum, y_sum, angle_sum)
        else:
            self.filtered_position = position
            
        # Calculate velocity if we have enough data
        if len(self.position_buffer) >= 2 and self.last_update_time is not None:
            dt = current_time - self.last_update_time
            if dt > 0.001:  # Avoid division by zero
                # Get two most recent positions
                prev_pos = self.position_buffer[-2]
                curr_pos = self.position_buffer[-1]
                
                # Calculate velocity components
                vx = (curr_pos[0] - prev_pos[0]) / dt
                vy = (curr_pos[1] - prev_pos[1]) / dt
                v_angle = (curr_pos[2] - prev_pos[2]) / dt
                
                # Smooth velocity estimate with low-pass filter
                alpha = 0.7  # Smoothing factor
                self.current_velocity = (
                    alpha * vx + (1 - alpha) * self.current_velocity[0],
                    alpha * vy + (1 - alpha) * self.current_velocity[1],
                    alpha * v_angle + (1 - alpha) * self.current_velocity[2]
                )
        
        # Make prediction for future position
        if self.current_velocity != (0.0, 0.0, 0.0) and self.filtered_position is not None:
            pred_x = self.filtered_position[0] + self.current_velocity[0] * self.prediction_horizon
            pred_y = self.filtered_position[1] + self.current_velocity[1] * self.prediction_horizon
            pred_angle = self.filtered_position[2] + self.current_velocity[2] * self.prediction_horizon
            self.predicted_position = (pred_x, pred_y, pred_angle)
        else:
            self.predicted_position = self.filtered_position if self.filtered_position else position
            
        self.last_update_time = current_time
        return self.filtered_position
    
    def get_filtered_position(self):
        """Get the current filtered position."""
        return self.filtered_position if self.filtered_position else (
            self.position_buffer[-1][:3] if self.position_buffer else None
        )
    
    def get_predicted_position(self):
        """Get the predicted future position based on velocity."""
        return self.predicted_position
        
    def get_velocity(self):
        """Get the current velocity estimate."""
        return self.current_velocity
    
    def get_recent_positions(self, n=3):
        """Get the n most recent positions."""
        return [p[:3] for p in list(self.position_buffer)[-n:]]
    
    def reset(self):
        """Reset the filter state."""
        self.position_buffer.clear()
        self.last_update_time = None
        self.current_velocity = (0.0, 0.0, 0.0)
        self.filtered_position = None
        self.predicted_position = None

class ErrorTracker:
    """Lightweight error tracker that monitors error values over time."""
    
    def __init__(self, name, max_history=8):
        """Initialize error tracker with efficient storage."""
        self.name = name
        self.current_error = 0.0
        self.previous_error = 0.0
        # Pre-allocate error history with zeros to avoid frequent allocations
        self.error_history = deque([0.0] * max_history, maxlen=max_history)
        self.error_history_index = 0
        self.error_history_count = 0
        self.error_history_size = max_history
        self.last_correction_time = 0.0
        self.sign_changes = 0  # Count of error sign changes (useful for oscillation detection)
        self.accumulated_error = 0.0
        self.decay_factor = 0.9  # Simplified decay factor
        
    def update(self, error, dt):
        """Update error tracking with new error value."""
        # Check for sign change
        if self.current_error != 0 and error != 0 and (self.current_error * error) < 0:
            self.sign_changes += 1
            
        # Update error history using in-place modification
        self.previous_error = self.current_error
        self.current_error = error
        
        # Update error history with in-place modification
        self.error_history[self.error_history_index] = error
        self.error_history_index = (self.error_history_index + 1) % self.error_history_size
        if self.error_history_count < self.error_history_size:
            self.error_history_count += 1
        
        # Update accumulated error with decay
        self.accumulated_error = (self.accumulated_error + error * dt) * self.decay_factor
    
    def reset(self):
        """Reset all tracked errors."""
        self.current_error = 0.0
        self.previous_error = 0.0
        # Reset history indices rather than allocating new memory
        self.error_history_index = 0
        self.error_history_count = 0
        # Zero out the existing history array instead of clearing and reallocating
        for i in range(len(self.error_history)):
            self.error_history[i] = 0.0
        self.accumulated_error = 0.0
        self.sign_changes = 0
    
    def is_error_growing(self):
        """Check if error is growing compared to previous value."""
        return abs(self.current_error) > abs(self.previous_error) * 1.05  # 5% threshold
        
    def record_correction(self):
        """Record that a correction was made for this error."""
        self.last_correction_time = time.time()
        # Reduce accumulated error when correction is made
        self.accumulated_error *= 0.5
        
    def get_trend(self, n=3):
        """Calculate trend of error (increasing/decreasing)."""
        if self.error_history_count < n:
            return 0.0  # Not enough data
            
        # Get the last n values
        history = list(self.error_history)[-n:]
        
        # Simple linear regression to find trend
        x = list(range(len(history)))
        y = history
        n = len(x)
        
        if n < 2:
            return 0.0
            
        # Calculate slope with least squares
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_xx = sum(xi ** 2 for xi in x)
        
        # Slope formula: (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x^2)
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x ** 2)
            return slope
        except ZeroDivisionError:
            return 0.0

class AdaptiveGainPID:
    """PID controller with adaptive gains based on error magnitude and trend."""
    
    def __init__(self, base_kp, base_ki, base_kd, output_min, output_max, name="PID"):
        """Initialize the adaptive PID controller."""
        # Base gains
        self.base_kp = base_kp
        self.base_ki = base_ki
        self.base_kd = base_kd
        
        # Current gains (will be adjusted adaptively)
        self.kp = base_kp
        self.ki = base_ki
        self.kd = base_kd
        
        # Output limits
        self.output_min = output_min
        self.output_max = output_max
        self.name = name
        
        # PID state
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.last_output = 0.0
        
        # Diagnostic information
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        
        # Adaptation parameters
        self.gain_adjust_rate = 0.1  # How quickly gains adjust
        self.zero_crossing_time = None
        self.sign_change_count = 0
        self.prev_sign = 0
        
        # Enhanced parameters
        self.integral_deadband = 0.05
        self.integral_decay = 0.7
        self.max_integral = (output_max - output_min) / base_ki if base_ki > 0 else 1.0
        
        # Error tracker for trend analysis
        self.error_tracker = None
        
        
    def adjust_gains(self, error, trend):
        """
        Adaptively adjust PID gains based on error magnitude and trend.
        
        Args:
            error: Current error value
            trend: Error trend (positive for increasing, negative for decreasing)
        """
        # Scale factor based on error magnitude (smaller errors -> smaller kp, larger kd)
        error_magnitude = abs(error)
        
        # Different gain profiles based on error trend and magnitude
        if trend < -0.1:  # Error is decreasing
            # Approaching target, increase damping
            kp_factor = max(0.8, 1.0 - error_magnitude * 0.5)
            ki_factor = max(0.5, 1.0 - error_magnitude)
            kd_factor = min(1.5, 1.0 + error_magnitude)
        elif trend > 0.1:  # Error is increasing
            # Moving away from target, increase responsiveness
            kp_factor = min(1.2, 1.0 + error_magnitude * 0.2)
            ki_factor = min(1.1, 1.0 + error_magnitude * 0.1)
            kd_factor = max(0.9, 1.0 - error_magnitude * 0.1)
        else:  # Error is stable
            kp_factor = 1.0
            ki_factor = 1.0
            kd_factor = 1.0
        
        # Special case for zero crossing
        if self.zero_crossing_time is not None:
            time_since_crossing = time.time() - self.zero_crossing_time
            if time_since_crossing < 0.5:  # Within 0.5 seconds of zero crossing
                # Enhance derivative, reduce integral during zero crossing
                kd_factor *= 1.2
                ki_factor *= 0.5
                
        # For lateral control specifically
        if self.name == "Linear Y":
            # More aggressive damping for lateral control
            kd_factor *= 1.2
            # Less integral for lateral to prevent overshooting
            ki_factor *= 0.8
        
        # Gradually adjust gains
        self.kp = self.kp * (1.0 - self.gain_adjust_rate) + (self.base_kp * kp_factor) * self.gain_adjust_rate
        self.ki = self.ki * (1.0 - self.gain_adjust_rate) + (self.base_ki * ki_factor) * self.gain_adjust_rate
        self.kd = self.kd * (1.0 - self.gain_adjust_rate) + (self.base_kd * kd_factor) * self.gain_adjust_rate
    
    def reset(self):
        """Reset controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.last_output = 0.0
        self.last_p_term = 0.0
        self.last_i_term = 0.0
        self.last_d_term = 0.0
        self.sign_change_count = 0
        self.prev_sign = 0
        self.zero_crossing_time = None
        
        # Reset gains to base values
        self.kp = self.base_kp
        self.ki = self.base_ki
        self.kd = self.base_kd
        
        # Reset error tracker if it exists
        if self.error_tracker is not None:
            self.error_tracker.reset()
        
    def compute(self, error, current_time=None, force_zero=False, error_trend=0.0):
        """
        Compute the control output based on the error with improved zero-crossing handling.
        
        Args:
            error: Current error value
            current_time: Current time (defaults to now)
            force_zero: Whether to force zero output
            error_trend: Trend of error (from ErrorTracker)
            
        Returns:
            float: Calculated control output
        """
        try:

            print(f"PID {self.name} compute: error={error:.3f}, force_zero={force_zero}")

            # If forcing zero output, bypass calculations
            if force_zero:
                self.prev_error = error
                self.last_p_term = 0.0
                self.last_i_term = 0.0
                self.last_d_term = 0.0
                self.last_output = 0.0
                return 0.0
                
            # Use current time if not provided
            if current_time is None:
                current_time = time.time()
                
            # Initialize time on first call
            if self.last_time is None:
                self.last_time = current_time
                self.prev_error = error
                # P-only control on first iteration (no I or D)
                output = self.kp * error
                self.last_p_term = output
                self.last_output = max(self.output_min, min(self.output_max, output))
                return float(self.last_output)
                
            # Calculate dt (time since last update)
            dt = current_time - self.last_time
            if dt <= 0.001:  # Protect against very small or negative dt
                dt = 0.01  # Fallback to prevent division by zero (assume 100Hz)
                
            # Detect sign changes for improved zero-crossing behavior
            current_sign = 1 if error > 0 else (-1 if error < 0 else 0)
            
            if self.prev_sign != 0 and current_sign != 0 and self.prev_sign != current_sign:
                # Sign changed - we crossed zero
                self.zero_crossing_time = current_time
                self.sign_change_count += 1
                
                # More aggressive integral reset on zero crossing for lateral movement
                if self.name == "Linear Y":
                    # More aggressive integral reset specifically for lateral PID
                    self.integral *= 0.1  # More aggressive than before (was 0.2)
                else:
                    # Default behavior for other controllers
                    self.integral *= 0.2  # More aggressive than before (was 0.3)
            else:
                # No sign change
                self.sign_change_count = max(0, self.sign_change_count - 1)
                
            # Update previous sign
            if error != 0:
                self.prev_sign = current_sign
                
            # Adjust gains based on error and trend
            self.adjust_gains(error, error_trend)
                
            # Calculate proportional term
            p_term = self.kp * error
            
            # Enhanced integral term handling with better anti-windup
            # Detect if output is likely to saturate
            predicted_output = p_term + self.last_i_term + self.last_d_term
            is_saturated = (predicted_output >= self.output_max) or (predicted_output <= self.output_min)
            
            # Enhanced integral behavior
            if is_saturated:
                # Don't accumulate integral when saturated
                i_term = self.last_i_term
            else:
                # Near zero-crossing, be more aggressive with integral changes
                recently_crossed_zero = (self.zero_crossing_time is not None and 
                                        (current_time - self.zero_crossing_time) < 0.5)
                
                if recently_crossed_zero:
                    # Near zero crossing, allow faster integral changes
                    if abs(error) > self.integral_deadband:
                        self.integral += error * dt * 1.2  # Boost integral after crossing zero
                    else:
                        self.integral *= 0.5  # More aggressively reduce integral near zero
                else:
                    # Normal integral handling
                    if abs(error) > self.integral_deadband:
                        self.integral += error * dt
                    else:
                        # More aggressively reduce integral term when close to target
                        self.integral *= self.integral_decay
                    
                # Apply integral limit to prevent excessive buildup
                self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
                
                i_term = self.ki * self.integral
            
            # Derivative term with improved noise handling
            # Use filtered error derivative to reduce noise sensitivity
            error_change = error - self.prev_error
            
            # Enhanced derivative handling
            if self.sign_change_count >= 2:
                # If oscillating (multiple sign changes), amplify derivative term
                d_term = self.kd * error_change / max(dt, 0.001) * 1.3
            else:
                # Normal derivative calculation
                d_term = self.kd * error_change / max(dt, 0.001)
            
            # Calculate raw output by summing all terms
            output = p_term + i_term + d_term
            
            # Apply output limits
            output_limited = max(self.output_min, min(self.output_max, output))
            
            # Additional anti-windup when error changes sign
            if error * self.prev_error < 0:
                # Error crossed zero - reduce integral more aggressively
                # If error is decreasing (approaching zero), be even more aggressive
                if abs(error) < abs(self.prev_error):
                    # More aggressive for lateral controller
                    if self.name == "Linear Y":
                        self.integral *= 0.1  # More aggressive than before (was 0.2)
                    else:
                        self.integral *= 0.2  # More aggressive than before (was 0.3)
                else:
                    if self.name == "Linear Y":
                        self.integral *= 0.2  # More aggressive than before (was 0.4)
                    else:
                        self.integral *= 0.3  # More aggressive than before (was 0.5)
                i_term = self.ki * self.integral
                    
            # Apply transition smoothing for rapid control changes
            if abs(output_limited - self.last_output) > (self.output_max - self.output_min) * 0.4:
                # Blend between previous and current output for large changes
                output_limited = 0.6 * output_limited + 0.4 * self.last_output
                
            # Save individual terms for diagnostics
            self.last_p_term = p_term
            self.last_i_term = i_term
            self.last_d_term = d_term
            self.last_output = output_limited
            
            # Save state for next iteration
            self.prev_error = error
            self.last_time = current_time
            
            print(f"PID {self.name} terms: P={self.last_p_term:.3f}, I={self.last_i_term:.3f}, D={self.last_d_term:.3f}")

            # Ensure we return a proper float value
            return float(output_limited)
            
        except Exception as e:
            # Log error and return safe value
            print(f"Error in PID compute: {str(e)}")
            return 0.0
        
    def get_components(self):
        """Get the last calculated PID components."""
        return (self.last_p_term, self.last_i_term, self.last_d_term)
    
    def get_current_gains(self):
        """Get the current adaptive gains."""
        return (self.kp, self.ki, self.kd)

class CoordinatedController:
    """Controller that coordinates lateral and angular movements."""
    
    def __init__(self, linear_pid, angular_pid, config=None):
        """
        Initialize the coordinated controller.
        
        Args:
            linear_pid: PID controller for lateral movement
            angular_pid: PID controller for angular movement
            config: Configuration dictionary
        """
        self.linear_pid = linear_pid
        self.angular_pid = angular_pid
        
        # Default configuration
        self.config = {
            'coupling_factor': 0.7,  # How strongly movements are coupled
            'min_angle_for_reduction': 0.1,  # ~5.7 degrees
            'zero_angle_threshold': 0.03,  # ~1.7 degrees
            'max_angle_factor': 0.5,  # 30 degrees
            'same_sign_scale': 0.8,  # Scaling when errors have same sign
            'opposite_sign_scale': 1.0,  # Scaling when errors have opposite signs
            'smoothing_factor': 0.6,  # Smoothing factor for velocity changes
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # State variables
        self.last_lateral_velocity = 0.0
        self.last_angular_velocity = 0.0
        self.last_update_time = None
    
    def compute(self, lateral_error, angular_error, current_time=None, robot_orientation=0.0):
        """
        Compute coordinated lateral and angular velocities.
        
        Args:
            lateral_error: Current lateral error
            angular_error: Current angular error in radians
            current_time: Current time (defaults to now)
            robot_orientation: Current robot orientation in radians
            
        Returns:
            tuple: (lateral_velocity, angular_velocity)
        """
        if current_time is None:
            current_time = time.time()
            
        # Initialize time on first call
        if self.last_update_time is None:
            self.last_update_time = current_time
            
        # Calculate dt
        dt = current_time - self.last_update_time
        dt = max(0.001, min(dt, 0.1))  # Bound dt to reasonable values
        
        # Get individual PID outputs
        lateral_trend = self.linear_pid.error_tracker.get_trend() if hasattr(self.linear_pid, 'error_tracker') else 0.0
        angular_trend = self.angular_pid.error_tracker.get_trend() if hasattr(self.angular_pid, 'error_tracker') else 0.0
        
        raw_lateral_velocity = self.linear_pid.compute(
            lateral_error, 
            current_time, 
            force_zero=False, 
            error_trend=lateral_trend
        )
        
        raw_angular_velocity = self.angular_pid.compute(
            angular_error, 
            current_time, 
            force_zero=False,
            error_trend=angular_trend
        )
        
        # Apply coordination logic
        # 1. Calculate coupling based on angular error magnitude
        angular_magnitude = abs(angular_error)
        
        # Only apply coupling if angular error is significant
        if angular_magnitude > self.config['min_angle_for_reduction']:
            # Normalize angular error to 0-1 range up to max_angle_factor
            normalized_angle = min(1.0, angular_magnitude / self.config['max_angle_factor'])
            
            # Calculate lateral velocity reduction
            lateral_reduction = normalized_angle * self.config['coupling_factor']
            
            # Reduce lateral velocity when angular error is large
            lateral_velocity = raw_lateral_velocity * (1.0 - lateral_reduction)
        else:
            lateral_velocity = raw_lateral_velocity
            
        # Set angular velocity directly from PID
        angular_velocity = raw_angular_velocity
        
        # 2. Adjust based on sign relationship between errors
        same_sign = (lateral_error * angular_error) > 0
        
        if abs(angular_error) > self.config['zero_angle_threshold']:
            if same_sign:
                # Same sign - needs coordinated movement
                lateral_velocity *= self.config['same_sign_scale']
            else:
                # Opposite sign - movement naturally helps correction
                lateral_velocity *= self.config['opposite_sign_scale']
                
        # 3. Apply smoothing to prevent jerky transitions
        if self.last_lateral_velocity is not None and self.last_angular_velocity is not None:
            lateral_velocity = self.last_lateral_velocity * (1 - self.config['smoothing_factor']) + \
                              lateral_velocity * self.config['smoothing_factor']
                              
            angular_velocity = self.last_angular_velocity * (1 - self.config['smoothing_factor']) + \
                              angular_velocity * self.config['smoothing_factor']
        
        # Store values for next iteration
        self.last_lateral_velocity = lateral_velocity
        self.last_angular_velocity = angular_velocity
        self.last_update_time = current_time
        
        return lateral_velocity, angular_velocity
    
    def reset(self):
        """Reset the controller state."""
        self.linear_pid.reset()
        self.angular_pid.reset()
        self.last_lateral_velocity = 0.0
        self.last_angular_velocity = 0.0
        self.last_update_time = None

class MovementStrategy:
    """Represents a robot movement strategy with blending capabilities."""
    
    def __init__(self, name, use_forward, use_lateral, use_angular, 
                 forward_scale, lateral_scale, angular_scale, reason):
        """Initialize a movement strategy."""
        self.name = name
        self.use_forward = use_forward
        self.use_lateral = use_lateral
        self.use_angular = use_angular
        self.forward_scale = forward_scale
        self.lateral_scale = lateral_scale
        self.angular_scale = angular_scale
        self.reason = reason
    
    def as_dict(self):
        """Convert to dictionary for compatibility with existing code."""
        return {
            "strategy_name": self.name,
            "use_forward": self.use_forward,
            "use_lateral": self.use_lateral,
            "use_angular": self.use_angular,
            "forward_scale": self.forward_scale,
            "lateral_scale": self.lateral_scale,
            "angular_scale": self.angular_scale,
            "reason": self.reason
        }

class StrategyBlender:
    """Handles smooth transitions between movement strategies."""
    
    def __init__(self, blend_duration=0.5):
        """Initialize the strategy blender."""
        self.current_strategy = None
        self.target_strategy = None
        self.blend_start_time = 0.0
        self.blending_active = False
        self.blend_duration = blend_duration
    
    def update_target(self, target_strategy, current_time):
        """
        Update the target strategy.
        
        Args:
            target_strategy: The target strategy to blend towards
            current_time: Current time
            
        Returns:
            bool: True if a new blend was started
        """
        # Initialize if this is the first strategy
        if self.current_strategy is None:
            self.current_strategy = target_strategy
            return False
            
        # Check if target is different from current
        if target_strategy.name != self.current_strategy.name:
            # Start new blend
            self.target_strategy = target_strategy
            self.blend_start_time = current_time
            self.blending_active = True
            return True
            
        return False
    
    def _smoothstep(self, x):
        """Apply smoothstep function to create smoother transitions."""
        x = max(0.0, min(1.0, x))  # Clamp input to 0-1
        return x * x * (3 - 2 * x)  # Smoothstep function
    
    def get_current_strategy(self, current_time):
        """
        Get the current strategy, which might be a blend of two strategies.
        
        Args:
            current_time: Current time
            
        Returns:
            MovementStrategy: Current strategy (possibly blended)
        """
        if not self.blending_active:
            return self.current_strategy
            
        # Calculate blend factor
        elapsed_time = current_time - self.blend_start_time
        linear_blend = min(1.0, elapsed_time / self.blend_duration)
        blend_factor = self._smoothstep(linear_blend)
        
        # Check if blending is complete
        if blend_factor >= 0.999:
            self.current_strategy = self.target_strategy
            self.blending_active = False
            return self.current_strategy
        
        # Create blended strategy
        name = f"{self.current_strategy.name}→{self.target_strategy.name}"
        use_forward = self.target_strategy.use_forward  # Boolean flags use target value
        use_lateral = self.target_strategy.use_lateral
        use_angular = self.target_strategy.use_angular
        
        # Blend continuous parameters
        forward_scale = self.current_strategy.forward_scale * (1.0 - blend_factor) + \
                       self.target_strategy.forward_scale * blend_factor
                       
        lateral_scale = self.current_strategy.lateral_scale * (1.0 - blend_factor) + \
                       self.target_strategy.lateral_scale * blend_factor
                       
        angular_scale = self.current_strategy.angular_scale * (1.0 - blend_factor) + \
                       self.target_strategy.angular_scale * blend_factor
        
        reason = f"Blending strategies: {blend_factor*100:.0f}% complete"
        
        return MovementStrategy(
            name, use_forward, use_lateral, use_angular,
            forward_scale, lateral_scale, angular_scale, reason
        )
    
    def is_blending(self):
        """Check if a blend is currently in progress."""
        return self.blending_active
    
    def get_blend_progress(self, current_time):
        """Get the current blend progress as a percentage."""
        if not self.blending_active:
            return 100.0
            
        elapsed_time = current_time - self.blend_start_time
        return min(100.0, (elapsed_time / self.blend_duration) * 100.0)
    
    def reset(self):
        """Reset the blender state."""
        self.current_strategy = None
        self.target_strategy = None
        self.blending_active = False

class ResourceMonitor:
    """
    Lightweight resource monitor for tracking CPU and memory usage.
    Provides callbacks for resource threshold alerts.
    """
    
    def __init__(self, update_interval=5.0):
        """
        Initialize the resource monitor.
        
        Args:
            update_interval: How often to update resource metrics (seconds)
        """
        self.update_interval = update_interval
        self.last_update_time = 0
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.alert_callbacks = []
        self.cpu_threshold = 85.0  # Default CPU usage threshold (%)
        self.memory_threshold = 85.0  # Default memory usage threshold (%)
    
    def update(self):
        """Update resource metrics if interval has elapsed."""
        current_time = time.time()
        
        # Only update at specified interval
        if current_time - self.last_update_time >= self.update_interval:
            self.cpu_usage = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            self.memory_usage = memory.percent
            
            # Check thresholds for alerts
            self._check_thresholds()
            
            self.last_update_time = current_time
    
    def _check_thresholds(self):
        """Check if any resource metrics exceed thresholds and trigger callbacks."""
        # Check CPU
        if self.cpu_usage > self.cpu_threshold:
            for callback in self.alert_callbacks:
                callback('cpu', self.cpu_usage)
        
        # Check memory
        if self.memory_usage > self.memory_threshold:
            for callback in self.alert_callbacks:
                callback('memory', self.memory_usage)
    
    def add_alert_callback(self, callback):
        """
        Add a callback to be called when resource thresholds are exceeded.
        
        Args:
            callback: Function to call with (resource_type, value) parameters
        """
        self.alert_callbacks.append(callback)
    
    def get_cpu_usage(self):
        """Get the last measured CPU usage."""
        return self.cpu_usage
    
    def get_memory_usage(self):
        """Get the last measured memory usage."""
        return self.memory_usage

class EnhancedPIDControllerNode(Node):
    """Enhanced PID Controller node with coordinated control and smooth transitions."""
    
    def __init__(self):
        """Initialize the enhanced PID controller node."""
        super().__init__('pid_controller')
        
        # Set up callback group for reentrant callbacks
        self.callback_group = ReentrantCallbackGroup()
        
        # Initialize resource monitoring
        self._init_resource_monitoring()
        
        # Set up parameters
        self._declare_parameters()
        
        # Initialize controllers
        self._init_controllers()
        
        # Set up state variables
        self._init_state_variables()
        
        # Set up object pools and reusable objects
        self._init_memory_pools()
        
        # Set up tf2 system
        self._setup_tf2()
        
        # Set up subscriptions
        self._setup_subscriptions()
        
        # Set up publishers
        self._setup_publishers()
        
        # Set up timers
        self._setup_timers()
        
        # Define movement strategies
        self._init_strategy_table()
        
        # Initialize target filter
        self.target_filter = TargetFilter(buffer_size=5, prediction_horizon=0.2)
        
        # Initialize strategy blender
        self.strategy_blender = StrategyBlender(blend_duration=0.5)
        
        # Log startup info
        self.get_logger().info("Enhanced PID Controller initialized with coordinated control and smooth transitions")
    
    def _init_resource_monitoring(self):
        """Initialize the resource monitor for CPU and memory tracking."""
        self.resource_monitor = ResourceMonitor(update_interval=5.0)
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.current_cpu_usage = 0.0
        self.performance_stats = {
            'cpu': deque(maxlen=12),  # 1 minute of 5-second samples
            'control_cycles': deque(maxlen=100),  # Last 100 control cycle times
            'control_skips': 0  # Count of skipped control cycles due to high CPU
        }
        
        # Performance adjustment parameters
        self.base_update_rate = 10.0  # Default 10Hz control rate
        self.adaptive_control_rate = True
        self.min_update_rate = 5.0  # Don't go below 5Hz
        self.max_update_rate = 20.0  # Don't go above 20Hz
        
        # Performance metrics tracking
        self.cycle_start_time = 0.0
        self.cycle_duration_avg = 0.0
        self.skip_next_cycle = False
        
    def _declare_parameters(self):
        """Declare and get all node parameters."""
        # Most parameter declarations are omitted for brevity but would be here
        
        # Only showing a few key parameters for example
        self.declare_parameters(
            namespace='',
            parameters=[
                # Linear X velocity PID parameters
                ('linear_x_kp', 1.0),
                ('linear_x_ki', 0.03),
                ('linear_x_kd', 0.1),
                ('linear_x_min', 0.0),
                ('linear_x_max', 0.2),
                
                # Linear Y velocity PID parameters
                ('linear_y_kp', 0.6),
                ('linear_y_ki', 0.08),
                ('linear_y_kd', 0.12),
                ('linear_y_min', -0.2),
                ('linear_y_max', 0.2),
                
                # Angular velocity PID parameters
                ('angular_kp', 1.5),
                ('angular_ki', 0.05),
                ('angular_kd', 0.8),
                ('angular_min', -0.5),
                ('angular_max', 0.5),
                
                # Control parameters
                ('min_distance', 0.9),
                ('max_distance', 2.0),
                ('target_offset_x', 0.0),
                ('target_offset_y', 0.0),
                ('target_update_rate', 10.0),
                ('diagnostics_rate', 0.5),
                ('debug_level', 0),
                ('adaptive_gains', True),
                ('use_lateral_control', True),
                
                # Resource monitoring parameters
                ('adaptive_control_rate', True),
                ('enable_resource_monitoring', True),
                ('cpu_high_threshold', 85.0),
                ('cpu_low_threshold', 40.0),
                
                # Performance optimization
                ('enable_transform_caching', True),
                ('transform_cache_ttl', 1.0),
                ('diagnostics_rate', 0.5),
                ('debug_level', 0),
            ]
        )
        
        # Get key parameters
        self.linear_x_kp = self.get_parameter('linear_x_kp').value
        self.linear_x_ki = self.get_parameter('linear_x_ki').value
        self.linear_x_kd = self.get_parameter('linear_x_kd').value
        self.linear_x_min = self.get_parameter('linear_x_min').value
        self.linear_x_max = self.get_parameter('linear_x_max').value
        
        self.linear_y_kp = self.get_parameter('linear_y_kp').value
        self.linear_y_ki = self.get_parameter('linear_y_ki').value
        self.linear_y_kd = self.get_parameter('linear_y_kd').value
        self.linear_y_min = self.get_parameter('linear_y_min').value
        self.linear_y_max = self.get_parameter('linear_y_max').value
        
        self.angular_kp = self.get_parameter('angular_kp').value
        self.angular_ki = self.get_parameter('angular_ki').value
        self.angular_kd = self.get_parameter('angular_kd').value
        self.angular_min = self.get_parameter('angular_min').value
        self.angular_max = self.get_parameter('angular_max').value
        
        # Get resource monitoring parameters
        self.adaptive_control_rate = self.get_parameter('adaptive_control_rate').value
        self.enable_resource_monitoring = self.get_parameter('enable_resource_monitoring').value
        self.cpu_high_threshold = self.get_parameter('cpu_high_threshold').value
        self.cpu_low_threshold = self.get_parameter('cpu_low_threshold').value
        
        # Get transform parameters
        self.enable_transform_caching = self.get_parameter('enable_transform_caching').value
        self.transform_cache_ttl = self.get_parameter('transform_cache_ttl').value
        
        # Target update rate from parameters
        self.update_rate = self.get_parameter('target_update_rate').value
        
        # All other parameter assignments would be here
        self.diagnostics_rate = self.get_parameter('diagnostics_rate').value

        self.debug_level = self.get_parameter('debug_level').value

        self.get_logger().info(
            f"CONTROLLER PARAMS: linear_x=[{self.linear_x_kp}, {self.linear_x_ki}, {self.linear_x_kd}], "
            f"linear_y=[{self.linear_y_kp}, {self.linear_y_ki}, {self.linear_y_kd}], "
            f"angular=[{self.angular_kp}, {self.angular_ki}, {self.angular_kd}]"
        )

        self.get_logger().info(f"MODIFIED GAINS: linear_x_kp={self.linear_x_kp}, angular_kp={self.angular_kp}")

               
    def _init_controllers(self):
        """Initialize the controllers with improved tuning."""
        # Create error trackers
        self.distance_error_tracker = ErrorTracker("distance", max_history=8)
        self.lateral_error_tracker = ErrorTracker("lateral", max_history=8)
        self.angular_error_tracker = ErrorTracker("angular", max_history=8)
        
        # Initialize individual PID controllers with adaptive gains
        self.pid_linear_x = AdaptiveGainPID(
            self.linear_x_kp, self.linear_x_ki, self.linear_x_kd,
            self.linear_x_min, self.linear_x_max,
            name="Linear X"
        )
        self.pid_linear_x.error_tracker = self.distance_error_tracker
        
        self.pid_linear_y = AdaptiveGainPID(
            self.linear_y_kp, self.linear_y_ki, self.linear_y_kd,
            self.linear_y_min, self.linear_y_max,
            name="Linear Y"
        )
        self.pid_linear_y.error_tracker = self.lateral_error_tracker
        
        self.pid_angular = AdaptiveGainPID(
            self.angular_kp, self.angular_ki, self.angular_kd,
            self.angular_min, self.angular_max,
            name="Angular"
        )
        
        self.pid_angular.error_tracker = self.angular_error_tracker
        
        # Initialize coordinated controller
        self.coordinated_controller = CoordinatedController(
            self.pid_linear_y, 
            self.pid_angular,
            {
                'coupling_factor': 0.7,
                'smoothing_factor': 0.6,
            }
        )
        
    def _init_memory_pools(self):
        """Setup memory pools and reusable objects for efficiency."""
        # Twist message pool
        self.twist_pool = []
        for _ in range(5):  # Pre-allocate 5 twist messages
            self.twist_pool.append(Twist())
        
        # Vector3 pool
        self.vector3_pool = []
        for _ in range(10):  # Pre-allocate 10 vector3 messages
            self.vector3_pool.append(Vector3())
        
        # Pre-allocate commonly used arrays
        self._limited_velocities = np.zeros(3, dtype=np.float32)
        self._prev_velocities = np.zeros(3, dtype=np.float32)
        self._target_velocities = np.zeros(3, dtype=np.float32)
        self._vel_diffs = np.zeros(3, dtype=np.float32)
        
        # Pre-allocate reusable message objects
        self._cmd_vel_msg = Twist()
        self._diag_msg = Float32MultiArray()
        self._diag_data = np.zeros(14, dtype=np.float32)
        
        # Pre-allocated objects for frequent operations
        self._strategy_dict = {
            "strategy_name": "",
            "use_forward": False,
            "use_lateral": False,
            "use_angular": False,
            "forward_scale": 0.0,
            "lateral_scale": 0.0,
            "angular_scale": 0.0,
            "reason": ""
        }
        self._key_tuple = ["none", "none", "none"]  # Will be modified in-place
        
        # Pre-allocated velocity tuple
        self._velocity_tuple = [0.0, 0.0, 0.0]
        
        # Pre-allocated velocity change check
        self._velocity_change_check = [False, False, False]
        
        # Pre-allocated error container
        self._current_errors = [0.0, 0.0, 0.0]  # distance, lateral, angular
    
    def get_twist_from_pool(self):
        """Get a Twist message from the pool."""
        if not self.twist_pool:
            # If pool is empty, create a new one
            return Twist()
        
        # Pop one from the pool
        twist = self.twist_pool.pop()
        
        # Clear all fields
        twist.linear.x = 0.0
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0
        
        return twist
    
    def return_twist_to_pool(self, twist):
        """Return a Twist to the pool if below capacity."""
        if len(self.twist_pool) < 10:
            self.twist_pool.append(twist)
    
    def get_vector3_from_pool(self):
        """Get a Vector3 from the pool."""
        if not self.vector3_pool:
            # If pool is empty, create a new one
            return Vector3()
        
        # Pop one from the pool
        vector = self.vector3_pool.pop()
        
        # Clear all fields
        vector.x = 0.0
        vector.y = 0.0
        vector.z = 0.0
        
        return vector
    
    def return_vector3_to_pool(self, vector):
        """Return a Vector3 to the pool if below capacity."""
        if len(self.vector3_pool) < 10:
            self.vector3_pool.append(vector)
    
    def _init_state_variables(self):
        """Initialize all state tracking variables."""
        # Target tracking
        self.current_target = None
        self.last_target_time = None
        
        # Robot state
        self.robot_state = "initializing"
        self.previous_state = None
        self.last_control_time = time.time()
        
        # Robot orientation
        self.robot_orientation = 0.0  # Current yaw in radians
        self.last_orientation_time = None  # Time of last orientation update
        
        # Log throttling timestamps
        self.last_control_log_time = 0.0
        self.last_state_log_time = 0.0
        self.last_diag_log_time = 0.0
        self.last_status_log_time = 0.0
        self.last_velocity_log_time = 0.0
        
        # Derived values
        self.current_distance = 0.0
        self.current_bearing = 0.0
        self.current_lateral = 0.0
        
        # Motion smoothing
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        self.last_logged_cmd = (0.0, 0.0, 0.0)
        
        # Diagnostic information
        self.cycle_count = 0
        
        # Velocity history
        self.velocity_history = LightweightBuffer(max_size=6)
        
        # Cached values
        self.desired_distance = 1.0  # Will be set properly after parameters are loaded
        
        # Stopped state tracking
        self._robot_stopped = False
        self._stop_time = 0.0
        self._last_stop_position = (0.0, 0.0, 0.0)
        
        # Movement strategy
        self.current_strategy = "IDLE"
        self.previous_strategy = None
        self.strategy_change_time = time.time()
        self.active_strategy = None  # Holds the current strategy object
        
        # Flag to track if we're shutting down
        self._shutting_down = False

        self.get_logger().info(
            f"INIT VALUES: desired_distance={self.desired_distance:.3f}m, "
            f"robot_state={self.robot_state}"
        )
    
    def _setup_tf2(self):
        """Set up tf2 components for coordinate transformations."""
        # Create the buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Define frame IDs
        self.reference_frame = "base_link"
        self.imu_frame = "imu_link"
        
        # Define transform parameters
        self.transform_timeout = 0.1  # seconds
        
        # Set up transform verification
        self.transform_verified = False
        self.transform_check_timer = self.create_timer(2.0, self._verify_transform)
        
        # Set up transform caching
        self.transform_cache = TTLDict(default_ttl=self.transform_cache_ttl)
        self.matrix_cache = TTLDict(default_ttl=5.0)  # 5-second TTL for matrices
        
        # Set up matrix transform optimization
        self.use_matrix_transforms = self.enable_transform_caching
        
        # Pre-compute common transforms
        self.common_transforms_computed = False
        self.transform_check_timer_matrix = self.create_timer(5.0, self._cache_common_transforms)
    
    def _verify_transform(self):
        """Verify that all required transforms are available."""
        try:
            # Check transform between reference frame and IMU frame
            if self.tf_buffer.can_transform(
                self.reference_frame,
                self.imu_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            ):
                self.transform_verified = True
                self.get_logger().info(f"Transform verification successful between {self.reference_frame} and {self.imu_frame}")
                self.transform_check_timer.cancel()
                return
        except Exception:
            pass
            
        # If transform is not ready, log warning
        self.get_logger().warning(f"Transform not yet available between {self.reference_frame} and {self.imu_frame}")
    
    def _cache_common_transforms(self):
        """Cache commonly used transforms to avoid frequent lookups."""
        # Only try if we already verified the transform
        if not self.transform_verified:
            return
            
        try:
            # Look up transform from reference frame to IMU frame
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                self.imu_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            
            # Convert to matrix and cache
            ref_to_imu_matrix = Matrix4x4.from_tf_transform(transform)
            self.matrix_cache.set("ref_to_imu", ref_to_imu_matrix)
            
            # Look up inverse transform
            transform = self.tf_buffer.lookup_transform(
                self.imu_frame,
                self.reference_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            
            # Convert to matrix and cache
            imu_to_ref_matrix = Matrix4x4.from_tf_transform(transform)
            self.matrix_cache.set("imu_to_ref", imu_to_ref_matrix)
            
            # Mark as computed and cancel timer
            self.common_transforms_computed = True
            self.transform_check_timer_matrix.cancel()
            
            self.get_logger().info("Cached common transforms as matrices")
            
        except Exception as e:
            self.get_logger().warning(f"Failed to cache common transforms: {str(e)}")
    
    def get_transform_between_frames(self, source_frame, target_frame):
        """
        Get transform between two frames with caching.
        
        Args:
            source_frame: Source frame ID
            target_frame: Target frame ID
            
        Returns:
            TransformStamped or None if not available
        """
        # Create unique key for this transform
        frame_key = f"{target_frame}_{source_frame}"
        
        # Check standard transform cache first (this should return TransformStamped)
        if frame_key in self.transform_cache:
            return self.transform_cache.get(frame_key)
        
        # Look up new transform
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=self.transform_timeout)
            )
            
            # Cache the transform
            self.transform_cache.set(frame_key, transform)
            
            # Also cache as matrix for future use (but don't return it here)
            if self.use_matrix_transforms:
                matrix = Matrix4x4.from_tf_transform(transform)
                self.matrix_cache.set(frame_key, matrix)
            
            return transform
        except Exception as e:
            if hasattr(self, 'debug_level') and self.debug_level >= 2:
                self.get_logger().warning(f"Transform lookup error: {str(e)}")
            return None
    
    def _setup_subscriptions(self):
        """Set up all subscriptions for this node."""
        self.state_sub = self.create_subscription(
            String,
            TOPICS["input"]["state"],
            self.state_callback,
            10,
            callback_group=self.callback_group
        )
        
        self.target_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["target"],
            self.target_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Subscribe to orientation data
        self.orientation_sub = self.create_subscription(
            Vector3Stamped,
            TOPICS["input"]["orientation"],
            self.orientation_callback,
            10,
            callback_group=self.callback_group
        )
    
    def _setup_publishers(self):
        """Set up all publishers for this node."""
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            TOPICS["output"]["cmd_vel"],
            10
        )
        
        self.pid_diag_pub = self.create_publisher(
            Float32MultiArray,
            TOPICS["output"]["diagnostics"],
            10
        )
        
        # Performance metrics publisher
        self.performance_pub = self.create_publisher(
            String,
            TOPICS["output"]["performance"],
            10
        )
    
    def _setup_timers(self):
        """Set up timer callbacks for periodic tasks with tiered frequencies."""
        # Main control loop timer
        self.timer = self.create_timer(1.0 / self.update_rate, self.control_loop_callback)
        
        # Diagnostic timer
        self.diagnostic_timer = self.create_timer(1.0 / self.diagnostics_rate, self.publish_diagnostics)
        
        # Resource monitoring timer
        self.resource_timer = self.create_timer(1.0, self._monitor_resources)
        
        # Add cache cleanup timer
        self.cache_cleanup_timer = self.create_timer(60.0, self._cleanup_all_caches)
    
    def _cleanup_all_caches(self):
        """Perform cleanup of all caches to prevent memory growth."""
        # Force cleanup of all TTL dictionaries
        self.transform_cache.cleanup_all()
        self.matrix_cache.cleanup_all()
        
        # Log cache cleanup
        if self.debug_level >= 2:
            self.get_logger().info("Transform cache cleanup complete")
    
    def orientation_callback(self, msg):
        """Handle orientation updates from the IMU with improved transform handling."""
        # Extract yaw (z component) from the Vector3Stamped message
        raw_orientation = msg.vector.z
        
        # Store timestamp for freshness checking
        self.last_orientation_time = time.time()
        
        # If we need to transform the orientation to another frame
        if self.imu_frame != self.reference_frame:
            try:
                # First approach: direct transform using quaternion math
                transform = self.get_transform_between_frames(self.imu_frame, self.reference_frame)
                if transform:
                    # Extract quaternion components from transform
                    qx = transform.transform.rotation.x
                    qy = transform.transform.rotation.y
                    qz = transform.transform.rotation.z
                    qw = transform.transform.rotation.w
                    
                    # Create forward unit vector in IMU frame
                    forward_x = math.cos(raw_orientation)
                    forward_y = math.sin(raw_orientation)
                    forward_z = 0.0
                    
                    # Calculate rotation matrix elements from quaternion
                    xx = qx * qx
                    xy = qx * qy
                    xz = qx * qz
                    xw = qx * qw
                    yy = qy * qy
                    yz = qy * qz
                    yw = qy * qw
                    zz = qz * qz
                    zw = qz * qw
                    
                    # Apply rotation to forward vector
                    r00 = 1.0 - 2.0 * (yy + zz)
                    r01 = 2.0 * (xy - zw)
                    r02 = 2.0 * (xz + yw)
                    r10 = 2.0 * (xy + zw)
                    r11 = 1.0 - 2.0 * (xx + zz)
                    r12 = 2.0 * (yz - xw)
                    
                    # Transform forward vector
                    tx = r00 * forward_x + r01 * forward_y + r02 * forward_z
                    ty = r10 * forward_x + r11 * forward_y + r12 * forward_z
                    
                    # Calculate new orientation angle
                    self.robot_orientation = math.atan2(ty, tx)
                else:
                    # If transform not available, use raw orientation
                    self.robot_orientation = raw_orientation
                    
            except Exception as e:
                # In case of error, fall back to raw orientation
                self.get_logger().warning(f"Orientation transform error: {str(e)}")
                self.robot_orientation = raw_orientation
        else:
            # No transform needed
            self.robot_orientation = raw_orientation
        
        # Log orientation updates at high debug level
        if hasattr(self, 'debug_level') and self.debug_level >= 3:
            self.get_logger().debug(f"Orientation update: yaw={math.degrees(self.robot_orientation):.2f}°")
    
    def _is_orientation_fresh(self):
        """Check if orientation data is fresh enough to use."""
        if self.last_orientation_time is None:
            return False
            
        current_time = time.time()
        age = current_time - self.last_orientation_time
        
        # Consider orientation data older than 0.5 seconds as stale
        return age < 0.5
    
    def target_callback(self, msg):
        """Handle target position updates."""
        if self._shutting_down:
            return
        
        # Update timestamps
        self.last_target_time = time.time()
        self.current_target = msg
        
        # Extract key information from target
        target = msg.point
        
        # Calculate full 2D distance to target
        self.current_distance = math.sqrt(target.x**2 + target.y**2)
        
        # Store target frame for debugging
        frame_id = msg.header.frame_id if hasattr(msg.header, 'frame_id') else "unknown_frame"
        self.target_frame = frame_id
        
        self.get_logger().info(
            f"TARGET DATA: frame={frame_id}, pos=({target.x:.3f}, {target.y:.3f}, {target.z:.3f}), "
            f"calculated: distance={self.current_distance:.3f}, lateral={self.current_lateral:.3f}, "
            f"bearing={math.degrees(self.current_bearing):.2f}°"
        )

        # Calculate bearing/direction to ball based on frame
        if frame_id == "camera_frame" or frame_id == "camera_optical_frame":
            # Camera optical frame: Z forward, X right, Y down
            self.current_bearing = math.atan2(target.x, target.z)
            self.current_lateral = target.x
        else:
            # Standard robot frame: X forward, Y left
            self.current_bearing = math.atan2(target.y, target.x)
            self.current_lateral = target.y
        
        # Apply target filtering if enabled
        if hasattr(self, 'target_filter'):
            # Update the filter with the new position
            filtered_position = self.target_filter.update(
                (self.current_distance, self.current_lateral, self.current_bearing),
                self.last_target_time
            )
            
            # Use filtered/predicted position if available
            if filtered_position:
                predicted_position = self.target_filter.get_predicted_position()
                if predicted_position:
                    # Use prediction for control, but keep unfiltered values for tracking
                    # and diagnostics
                    self.filtered_distance = predicted_position[0]
                    self.filtered_lateral = predicted_position[1]
                    self.filtered_bearing = predicted_position[2]
                    
                    if self.debug_level >= 2:
                        self.get_logger().debug(
                            f"Target filtering: raw=({self.current_distance:.2f}, {self.current_lateral:.2f}, "
                            f"{math.degrees(self.current_bearing):.1f}°), "
                            f"filtered=({self.filtered_distance:.2f}, {self.filtered_lateral:.2f}, "
                            f"{math.degrees(self.filtered_bearing):.1f}°)"
                        )
                else:
                    # Fall back to raw values if prediction not available
                    self.filtered_distance = self.current_distance
                    self.filtered_lateral = self.current_lateral
                    self.filtered_bearing = self.current_bearing
            else:
                # Fall back to raw values if filtering not available
                self.filtered_distance = self.current_distance
                self.filtered_lateral = self.current_lateral
                self.filtered_bearing = self.current_bearing
    
    def state_callback(self, msg):
        """Handle robot state updates from the state manager."""
        new_state = msg.data
        
        # If state changed, handle the transition
        if new_state != self.robot_state:

            self.get_logger().info(
                f"STATE TRANSITION: {self.robot_state} → {new_state}, "
                f"last_target_time={self.last_target_time}, "
                f"current_time={time.time()}"
            )

            # Throttled logging for state changes
            self._log_throttled(
                self.get_logger().info,
                f"Robot state changed: {self.robot_state} → {new_state}",
                LOG_THROTTLE_STATE,
                'last_state_log_time'
            )
            
            self.previous_state = self.robot_state
            self.robot_state = new_state
            
            # Only reset controllers when switching to/from tracking
            if new_state == "tracking" or self.previous_state == "tracking":
                self.pid_linear_x.reset()
                self.pid_linear_y.reset()
                self.pid_angular.reset()
                self.coordinated_controller.reset()
                
                # Reset error trackers
                self.distance_error_tracker.reset()
                self.lateral_error_tracker.reset()
                self.angular_error_tracker.reset()
                
                # Reset target filter
                self.target_filter.reset()
                
                # Reset strategy blender
                self.strategy_blender.reset()
                
            # If we're not in tracking mode, ensure the robot is stopped
            # (unless it's in searching or lost_ball mode, where the state manager controls motion)
            if new_state != "tracking" and new_state != "searching" and new_state != "lost_ball":
                self.stop_robot()
    
    def _log_throttled(self, level_func, message, min_interval, last_time_attr):
        """Log messages with throttling to reduce log volume."""
        current_time = time.time()
        last_time = getattr(self, last_time_attr, 0)
        
        if current_time - last_time >= min_interval:
            level_func(message)
            setattr(self, last_time_attr, current_time)
            return True
            
        return False
    
    def _monitor_resources(self):
        """Monitor system resources and adjust control parameters if needed."""
        if not self.enable_resource_monitoring:
            return
            
        # Update resource metrics
        self.resource_monitor.update()
        current_cpu = self.resource_monitor.get_cpu_usage()
        
        # Store CPU history
        self.performance_stats['cpu'].append(current_cpu)
        self.current_cpu_usage = current_cpu
        
        # Adaptive control rate based on CPU usage
        if self.adaptive_control_rate:
            self._adjust_control_rate()
    
    def _handle_resource_alert(self, resource_type, value):
        """Handle resource alerts from the resource monitor."""
        if resource_type == 'cpu':
            if value > 90.0:
                # Severe CPU alert - reduce control rate dramatically
                self.get_logger().warning(f"Severe CPU usage alert: {value:.1f}% - adjusting control rate")
                
                # Trigger skip of next cycle for immediate relief
                self.skip_next_cycle = True
                
                # Reduce control rate
                if self.update_rate > self.min_update_rate:
                    new_rate = max(self.min_update_rate, self.update_rate * 0.7)
                    self._update_control_rate(new_rate)
    
    def _adjust_control_rate(self):
        """Adjust control loop rate based on CPU usage."""
        # Only adjust if we're using adaptive control
        if not self.adaptive_control_rate:
            return
            
        # Get average CPU usage
        avg_cpu = 0.0
        if self.performance_stats['cpu']:
            avg_cpu = sum(self.performance_stats['cpu']) / len(self.performance_stats['cpu'])
        
        # Adjust based on average CPU usage
        if avg_cpu > self.cpu_high_threshold:
            # High CPU - reduce rate
            if self.update_rate > self.min_update_rate:
                new_rate = max(self.min_update_rate, self.update_rate * 0.9)
                self._update_control_rate(new_rate)
        elif avg_cpu < self.cpu_low_threshold and self.update_rate < self.base_update_rate:
            # Low CPU - increase rate, up to base rate
            new_rate = min(self.base_update_rate, self.update_rate * 1.1)
            self._update_control_rate(new_rate)
    
    def _update_control_rate(self, new_rate):
        """Update the control loop rate if it has changed significantly."""
        # Only update if change is significant
        if abs(new_rate - self.update_rate) < 0.1:
            return
            
        # Log the change
        self.get_logger().info(f"Adjusting control rate: {self.update_rate:.1f}Hz → {new_rate:.1f}Hz")
        
        # Update rate
        self.update_rate = new_rate
        
        # Recreate timer with new rate
        self.timer.cancel()
        self.timer = self.create_timer(1.0 / self.update_rate, self.control_loop_callback)
    
    def control_loop_callback(self):
        """Regular control loop to calculate and publish velocity commands."""
        try:
            # Skip this cycle if requested
            if self.skip_next_cycle:
                self.skip_next_cycle = False
                return
                
            # Track performance
            self.cycle_start_time = time.time()
            
            if self._shutting_down:
                return
                    
            current_time = time.time()
            dt = current_time - self.last_control_time
            self.last_control_time = current_time
            self.cycle_count += 1
            
            # Log periodic status updates
            if self.cycle_count % 50 == 0:
                self._log_periodic_status()
            
            # Only generate commands in tracking mode with a recent target
            if self.robot_state != "tracking" or self.current_target is None:
                # When not tracking, ensure robot is stopped (unless controlled by another node)
                if self.robot_state not in ["searching", "lost_ball"]:
                    self.stop_robot()
                return
            
            # Check if orientation data is fresh (prevents race conditions)
            if not self._is_orientation_fresh():
                self.get_logger().warning("Skipping control cycle - orientation data is stale")
                return
            
            # Use filtered values if available, otherwise use raw values from target callback
            if hasattr(self, 'filtered_distance'):
                distance = self.filtered_distance
                lateral = self.filtered_lateral
                bearing = self.filtered_bearing
            else:
                distance = self.current_distance
                lateral = self.current_lateral
                bearing = self.current_bearing
            
            angular_degrees = math.degrees(bearing)
            
            self.get_logger().info(
                f"PRE-STOP CHECK: distance={distance:.3f}m (target={self.desired_distance:.3f}m), "
                f"lateral={lateral:.3f}m, angular={angular_degrees:.2f}°, "
                f"is_stopped={self._robot_stopped}"
            )

            # Calculate errors using pre-allocated array
            self._current_errors[0] = distance - self.desired_distance  # distance_error
            self._current_errors[1] = lateral - 0.0    # lateral_error (target_offset_y)
            self._current_errors[2] = bearing          # angular_error
            
            # Check if we need to reset stopped state based on errors
            state_reset = self._reset_stopped_state_if_needed(
                self._current_errors[0], 
                self._current_errors[1], 
                angular_degrees
            )
            
            # If state was reset, skip the normal stop condition check this cycle
            if not state_reset:
                # Check stop conditions
                should_stop, stop_reason = self._evaluate_stop_conditions(
                    distance, lateral, angular_degrees, self._robot_stopped
                )
                
                if should_stop:
                    if not self._robot_stopped:
                        self.get_logger().info(stop_reason)
                        self.stop_robot()
                    return
            
            # Update error trackers
            self.distance_error_tracker.update(self._current_errors[0], dt)
            self.lateral_error_tracker.update(self._current_errors[1], dt)
            self.angular_error_tracker.update(self._current_errors[2], dt)
            
            # Determine the optimal movement strategy
            strategy = self._determine_movement_strategy(
                self._current_errors[0], self._current_errors[1], angular_degrees
            )
            
            # Apply strategy to movement decisions
            use_forward = strategy["use_forward"]
            use_lateral = strategy["use_lateral"]
            use_angular = strategy["use_angular"]
            
            forward_scale = strategy["forward_scale"]
            lateral_scale = strategy["lateral_scale"]
            angular_scale = strategy["angular_scale"]
            
            self.get_logger().info(f"Computing PID with: use_forward={use_forward}, use_lateral={use_lateral}, use_angular={use_angular}")

            # Compute PID outputs
            if hasattr(self, 'use_coordinated_control') and self.use_coordinated_control:
                # Use coordinated controller for lateral and angular movements
                linear_x_velocity = self.pid_linear_x.compute(
                    self._current_errors[0], 
                    current_time, 
                    not use_forward
                )
                
                # Use coordinated control for lateral and angular velocities
                lateral_velocity, angular_velocity = self.coordinated_controller.compute(
                    self._current_errors[1],   # lateral error
                    self._current_errors[2],   # angular error
                    current_time,              # current time
                    self.robot_orientation     # current orientation from IMU
                )
                
                # Disable individual components if strategy requires
                if not use_lateral:
                    lateral_velocity = 0.0
                if not use_angular:
                    angular_velocity = 0.0
            else:
                # Traditional separate PID controllers
                linear_x_velocity = self.pid_linear_x.compute(
                    self._current_errors[0], 
                    current_time, 
                    not use_forward
                )
                
                lateral_velocity = self.pid_linear_y.compute(
                    self._current_errors[1], 
                    current_time, 
                    not use_lateral
                )
                
                angular_velocity = self.pid_angular.compute(
                    self._current_errors[2], 
                    current_time, 
                    not use_angular
                )
            
            # Apply strategy scaling factors
            linear_x_velocity *= forward_scale
            lateral_velocity *= lateral_scale
            angular_velocity *= angular_scale
            
            self.get_logger().info(
                f"After scaling: linear_x={linear_x_velocity:.3f}, lateral={lateral_velocity:.3f}, "
                f"angular={angular_velocity:.3f}"
            )

            # Apply velocity and acceleration limits
            limited_velocities = self._apply_velocity_limits(
                linear_x_velocity, lateral_velocity, angular_velocity, current_time
            )
            
            linear_x_velocity, lateral_velocity, angular_velocity = limited_velocities
            
            # Store new velocities in pre-allocated arrays
            self._velocity_tuple[0] = linear_x_velocity
            self._velocity_tuple[1] = lateral_velocity
            self._velocity_tuple[2] = angular_velocity
            
            # Calculate if velocity changed significantly for logging
            for i in range(3):
                self._velocity_change_check[i] = abs(self._velocity_tuple[i] - self.last_logged_cmd[i]) > 0.05
            
            # Log velocity commands (throttled)
            if self.debug_level >= 1 or any(self._velocity_change_check):
                self._log_throttled(
                    self.get_logger().info,
                    f"MOTION: x={linear_x_velocity:.2f} y={lateral_velocity:.2f} θ={angular_velocity:.2f}",
                    0.5,  # Throttle to every 0.5 seconds 
                    'last_velocity_log_time'
                )
                # Update last logged command
                self.last_logged_cmd = tuple(self._velocity_tuple)
            
            # Store for next cycle
            self.last_cmd_vel = tuple(self._velocity_tuple)
            
            # Get a Twist message from the pool
            cmd_vel_msg = self._cmd_vel_msg  # Use pre-allocated message
            
            # Set velocity values
            cmd_vel_msg.linear.x = float(linear_x_velocity)
            cmd_vel_msg.linear.y = float(lateral_velocity)
            cmd_vel_msg.angular.z = float(angular_velocity)
            
            # Save for history
            velocity_tuple = (float(linear_x_velocity), float(lateral_velocity), float(angular_velocity))
            self.velocity_history.add(velocity_tuple)
            
            self.get_logger().info(f"PUBLISHING: linear_x={cmd_vel_msg.linear.x:.3f}, "
                          f"lateral={cmd_vel_msg.linear.y:.3f}, "
                          f"angular={cmd_vel_msg.angular.z:.3f}")

            # Publish command
            self.cmd_vel_pub.publish(cmd_vel_msg)
            
            # Update error trackers if significant movement is occurring
            if abs(linear_x_velocity) > 0.05:
                self.distance_error_tracker.record_correction()
            if abs(lateral_velocity) > 0.05:
                self.lateral_error_tracker.record_correction()
            if abs(angular_velocity) > 0.1:
                self.angular_error_tracker.record_correction()
            
            # Calculate cycle duration for performance monitoring
            cycle_duration = time.time() - self.cycle_start_time
            self.performance_stats['control_cycles'].append(cycle_duration)
            
            # Update running average
            if len(self.performance_stats['control_cycles']) > 0:
                self.cycle_duration_avg = sum(self.performance_stats['control_cycles']) / len(self.performance_stats['control_cycles'])
                
        except Exception as e:
            self.get_logger().error(f"Unexpected error in control_loop_callback: {str(e)}")
            # Try to safely stop the robot
            try:
                self.stop_robot()
            except Exception as stop_error:
                self.get_logger().error(f"Failed to stop robot after error: {str(stop_error)}")
    
    def _reset_stopped_state_if_needed(self, distance_error, lateral_error, angular_error):
        """
        Reset stopped state if significant movement is required.
        
        Args:
            distance_error: Current error in distance to target
            lateral_error: Current error in lateral position
            angular_error: Current error in angular position (degrees)
            
        Returns:
            bool: True if stopped state was reset, False otherwise
        """
        if not self._robot_stopped:
            return False  # Already in movement state
            
        # If any error exceeds the movement threshold, exit stopped state
        distance_threshold = 0.2 * 1.5  # Example threshold with hysteresis
        lateral_threshold = 0.08 * 1.5   # Example threshold with hysteresis
        angular_threshold = 5.0 * 1.5    # Example threshold with hysteresis
        
        if (abs(distance_error) > distance_threshold or
            abs(lateral_error) > lateral_threshold or
            abs(angular_error) > angular_threshold):
            
            self.get_logger().info(
                f"Exiting stopped state - Movement required: "
                f"distance_error={distance_error:.3f}m, "
                f"lateral_error={lateral_error:.3f}m, "
                f"angular_error={angular_error:.2f}°"
            )
            
            # Reset stopped state
            self._robot_stopped = False
            return True
        
        return False
    
    def _evaluate_stop_conditions(self, distance, lateral, angular_degrees, is_stopped):
        """
        Evaluate if the robot should move based on current conditions.
        
        Args:
            distance: Current distance to target
            lateral: Current lateral offset
            angular_degrees: Current angular error in degrees
            is_stopped: Whether the robot is currently stopped
            
        Returns:
            tuple: (should_move, reason) - True if robot should move, False if it should stop
        """
        # Calculate error values
        distance_error = abs(distance - self.desired_distance)
        lateral_error = abs(lateral)
        angular_error = abs(angular_degrees)
        
        # Define thresholds
        distance_threshold = 0.1  # meters
        lateral_threshold = 0.05  # meters
        angular_threshold = 3.0   # degrees
        
        # Apply hysteresis - different thresholds based on current state
        if is_stopped:
            # If already stopped, use higher thresholds to start moving
            # (requires larger errors to start moving)
            distance_threshold *= 1.5
            lateral_threshold *= 1.5
            angular_threshold *= 1.5
        
        self.get_logger().info(
            f"MOVEMENT THRESHOLDS: d_err={distance_error:.3f}/{distance_threshold:.3f}, "
            f"l_err={lateral_error:.3f}/{lateral_threshold:.3f}, "
            f"a_err={angular_error:.2f}/{angular_threshold:.2f}"
        )
        
        # Check if any error exceeds thresholds - robot should move
        if (distance_error > distance_threshold or
            lateral_error > lateral_threshold or
            angular_error > angular_threshold):
            
            reason = (
                f"Movement needed: distance_error={distance_error:.3f}m, "
                f"lateral_error={lateral_error:.3f}m, "
                f"angular_error={angular_error:.2f}°"
            )
            return False, reason  # Return False to indicate robot should NOT stop
        
        # All errors within thresholds - robot should stop
        reason = (
            f"Target reached: distance_error={distance_error:.3f}m, "
            f"lateral_error={lateral_error:.3f}m, "
            f"angular_error={angular_error:.2f}°"
        )
        return True, reason  # Return True to indicate robot SHOULD stop
    
    def _apply_velocity_limits(self, linear_x, linear_y, angular_z, current_time):
        """
        Apply velocity and acceleration limits for smooth, natural movement.
        With enhanced handling for combined movements.
        
        Args:
            linear_x: Calculated forward velocity
            linear_y: Calculated lateral velocity
            angular_z: Calculated angular velocity
            current_time: Current time for acceleration limiting
            
        Returns:
            tuple: (limited_linear_x, limited_linear_y, limited_angular_z)
        """

        self.get_logger().info(f"Before limits: linear_x={linear_x:.3f}, "
                          f"linear_y={linear_y:.3f}, angular_z={angular_z:.3f}")

        # Convert inputs to numpy arrays for vectorized operations
        self._target_velocities[0] = linear_x
        self._target_velocities[1] = linear_y
        self._target_velocities[2] = angular_z
        
        # Get previous velocities
        self._prev_velocities[0] = self.last_cmd_vel[0]
        self._prev_velocities[1] = self.last_cmd_vel[1]
        self._prev_velocities[2] = self.last_cmd_vel[2]
        
        # Calculate time since last control step
        dt = current_time - getattr(self, "last_accel_time", current_time - 0.1)
        self.last_accel_time = current_time
        
        # Ensure dt is reasonable
        dt = max(0.001, min(dt, 0.1))
        
        # Scale acceleration limit by time
        accel_limit = 0.6 * dt * 10.0  # Example accel_limit
        angular_accel_limit = 1.0 * dt * 10.0  # Example angular_accel_limit
        
        # Calculate velocity differences
        np.subtract(self._target_velocities, self._prev_velocities, out=self._vel_diffs)
        
        # Apply acceleration limits
        for i in range(3):
            if i < 2:  # Linear X and Y
                limit = accel_limit
                # Apply acceleration boosting when starting from stop
                if abs(self._prev_velocities[i]) < 0.01 and abs(self._target_velocities[i]) > 0.01:
                    boost = 3.0  # Acceleration boost factor
                    limit *= boost
            else:  # Angular Z
                limit = angular_accel_limit
                # Apply acceleration boosting for angular motion too
                if abs(self._prev_velocities[i]) < 0.01 and abs(self._target_velocities[i]) > 0.01:
                    boost = 2.0  # Angular acceleration boost factor
                    limit *= boost
            
            # If velocity change exceeds limit, scale it
            if abs(self._vel_diffs[i]) > limit:
                self._limited_velocities[i] = self._prev_velocities[i] + math.copysign(
                    limit, self._vel_diffs[i])
            else:
                self._limited_velocities[i] = self._target_velocities[i]
        
        # Apply minimum velocity thresholds
        min_effective_velocity = 0.05
        min_angular_velocity = 0.1
        
        # Forward velocity
        if abs(self._limited_velocities[0]) < min_effective_velocity:
            self._limited_velocities[0] = 0.0
            
        # Lateral velocity
        if abs(self._limited_velocities[1]) < min_effective_velocity:
            self._limited_velocities[1] = 0.0
            
        # Angular velocity
        if abs(self._limited_velocities[2]) < min_angular_velocity:
            self._limited_velocities[2] = 0.0
        
        # Limit combined lateral and angular movement
        if abs(self._limited_velocities[1]) > 0.15 and abs(self._limited_velocities[2]) > 0.3:
            # Scale down lateral velocity when combined with significant angular velocity
            self._limited_velocities[1] *= 0.6
        
        # Apply maximum velocity limits
        linear_x_max = 0.2  # Example max_velocity
        linear_y_max = 0.2  # Example max_velocity
        angular_max = 0.5   # Example max_angular_velocity
        
        self._limited_velocities[0] = max(-linear_x_max, min(linear_x_max, self._limited_velocities[0]))
        self._limited_velocities[1] = max(-linear_y_max, min(linear_y_max, self._limited_velocities[1]))
        self._limited_velocities[2] = max(-angular_max, min(angular_max, self._limited_velocities[2]))
        
        self.get_logger().info(f"After limits: linear_x={self._limited_velocities[0]:.3f}, "
                          f"linear_y={self._limited_velocities[1]:.3f}, "
                          f"angular_z={self._limited_velocities[2]:.3f}")

        return (self._limited_velocities[0], self._limited_velocities[1], self._limited_velocities[2])
    
    def stop_robot(self):
        """Send a command to stop all robot motion immediately."""
        # Reuse cmd_vel message and set all fields to 0
        self._cmd_vel_msg.linear.x = 0.0
        self._cmd_vel_msg.linear.y = 0.0
        self._cmd_vel_msg.linear.z = 0.0
        self._cmd_vel_msg.angular.x = 0.0
        self._cmd_vel_msg.angular.y = 0.0
        self._cmd_vel_msg.angular.z = 0.0
        
        # Publish stop command multiple times to ensure it's received
        for _ in range(3):
            self.cmd_vel_pub.publish(self._cmd_vel_msg)
            time.sleep(0.01)  # Small delay between publishes
        
        # Reset last command velocity for acceleration limiting
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        
        # Clear velocity history
        self.velocity_history = LightweightBuffer(max_size=6)
        
        # Reset controllers to clear any lingering integral terms
        self.pid_linear_x.reset()
        self.pid_linear_y.reset()
        self.pid_angular.reset()
        self.coordinated_controller.reset()
        
        # Reset error trackers
        self.distance_error_tracker.reset()
        self.lateral_error_tracker.reset()
        self.angular_error_tracker.reset()
        
        # Reset strategy blender
        self.strategy_blender.reset()
        
        # Set a "stopped" state flag and timestamp
        self._robot_stopped = True
        self._stop_time = time.time()
        
        # Remember our position when we stopped
        self._last_stop_position = (
            self.current_distance,
            self.current_lateral, 
            self.current_bearing
        )
        
        self.get_logger().info("Robot stopped! All velocities and errors reset.")
    
    def _init_strategy_table(self):
        """Initialize the table-driven movement strategy definitions."""
        # Define the strategy table
        # Format: (distance_state, lateral_state, angular_state): [
        #    name, use_forward, use_lateral, use_angular, 
        #    forward_scale, lateral_scale, angular_scale, reason_template
        # ]
        self.strategy_table = {
            # All errors within deadbands - no movement
            ("none", "none", "none"): [
                "NO_MOVEMENT", False, False, False, 
                0.0, 0.0, 0.0, 
                "All errors within deadbands"
            ],
            
            # Single dimension errors - focused corrections
            ("small", "none", "none"): [
                "FORWARD_ONLY", True, False, False, 
                0.8, 0.0, 0.0, 
                "Small distance error correction: {distance_error:.2f}m"
            ],
            ("medium", "none", "none"): [
                "FORWARD_ONLY", True, False, False, 
                1.0, 0.0, 0.0, 
                "Medium distance error correction: {distance_error:.2f}m"
            ],
            ("large", "none", "none"): [
                "FORWARD_ONLY", True, False, False, 
                1.0, 0.0, 0.0, 
                "Large distance error correction: {distance_error:.2f}m"
            ],
            
            ("none", "small", "none"): [
                "LATERAL_ONLY", False, True, False, 
                0.0, 0.8, 0.0, 
                "Small lateral error correction: {lateral_error:.2f}m"
            ],
            ("none", "medium", "none"): [
                "LATERAL_ONLY", False, True, False, 
                0.0, 1.0, 0.0, 
                "Medium lateral error correction: {lateral_error:.2f}m"
            ],
            ("none", "large", "none"): [
                "LATERAL_ONLY", False, True, False, 
                0.0, 1.0, 0.0, 
                "Large lateral error correction: {lateral_error:.2f}m"
            ],
            
            ("none", "none", "small"): [
                "ANGULAR_ONLY", False, False, True, 
                0.0, 0.0, 0.8, 
                "Small angular error correction: {angular_error:.1f}°"
            ],
            ("none", "none", "medium"): [
                "ANGULAR_ONLY", False, False, True, 
                0.0, 0.0, 1.0, 
                "Medium angular error correction: {angular_error:.1f}°"
            ],
            
            # More granular angular error categories
            ("none", "none", "medium_high"): [
                "ANGULAR_ONLY", False, False, True, 
                0.0, 0.0, 1.0, 
                "Medium-high angular error correction: {angular_error:.1f}°"
            ],
            ("none", "none", "medium_large"): [
                "ANGULAR_ONLY", False, False, True, 
                0.0, 0.0, 1.0, 
                "Medium-large angular error correction: {angular_error:.1f}°"
            ],
            ("none", "none", "large"): [
                "ANGULAR_ONLY", False, False, True, 
                0.0, 0.0, 1.0, 
                "Large angular error correction: {angular_error:.1f}°"
            ],
            
            # Small angular with other errors - balanced approach
            ("*", "*", "small"): [
                "BALANCED", True, True, True, 
                0.7, 0.7, 0.4, 
                "Balanced movement with small angular correction: {angular_error:.1f}°"
            ],
            
            # Medium angular takes precedence but allows other movements
            ("*", "*", "medium"): [
                "ANGULAR_BALANCED", True, True, True, 
                0.5, 0.4, 0.7, 
                "Angular-balanced movement: {angular_error:.1f}°"
            ],
            
            # Intermediate step for more gradual transition
            ("*", "*", "medium_high"): [
                "ANGULAR_PRIMARY_BALANCED", False, True, True, 
                0.0, 0.08, 0.7, 
                "Primarily angular correction with minimal lateral: {angular_error:.1f}°"
            ],
            
            # Medium-large angular starts to dominate but allows some lateral
            ("*", "*", "medium_large"): [
                "ANGULAR_PRIMARY_BALANCED", False, True, True, 
                0.0, 0.15, 0.8, 
                "Primarily angular correction with some lateral: {angular_error:.1f}°"
            ],
            
            # Only truly large angular errors get exclusive focus
            ("*", "*", "large"): [
                "ANGULAR_PRIMARY", False, False, True, 
                0.0, 0.0, 1.0, 
                "Angular correction prioritized: {angular_error:.1f}°"
            ],
            
            # Combined movement strategies
            ("small", "small", "none"): [
                "COORDINATED", True, True, False, 
                0.8, 0.8, 0.0, 
                "Coordinated distance and lateral correction"
            ],
            
            # Medium lateral with other errors
            ("*", "medium", "small"): [
                "LATERAL_PRIMARY", True, True, True, 
                0.4, 1.0, 0.3, 
                "Lateral-focused coordination with small angular correction"
            ],
            
            # Fallback strategy
            ("*", "*", "*"): [
                "BALANCED", True, True, True, 
                0.6, 0.6, 0.6, 
                "Balanced movement strategy (fallback)"
            ]
        }
    
    def _categorize_error(self, error, error_type="distance"):
        """
        Categorize an error value into none, small, medium, medium_high, medium_large, or large.
        
        Args:
            error: The error value to categorize
            error_type: The type of error (distance, lateral, angular)
            
        Returns:
            String: The error category
        """
        abs_error = abs(error)
        
        # Select appropriate deadband and thresholds based on error type
        if error_type == "angular":
            deadband = 5.0  # Example angular_deadband in degrees
            small_threshold = deadband * 1.0
            medium_threshold = deadband * 2.0
            medium_high_threshold = deadband * 3.0
            medium_large_threshold = deadband * 4.0
            large_threshold = deadband * 6.0
        elif error_type == "lateral":
            deadband = 0.08  # Example lateral_deadband
            small_threshold = deadband * 1.0
            medium_threshold = deadband * 2.0
            medium_high_threshold = deadband * 3.0
            medium_large_threshold = deadband * 4.0
            large_threshold = deadband * 6.0
        else:  # distance
            deadband = 0.08  # Example distance_deadband
            small_threshold = deadband * 1.0
            medium_threshold = deadband * 2.0
            medium_high_threshold = deadband * 3.0
            medium_large_threshold = deadband * 4.0
            large_threshold = deadband * 6.0
        
        # Handle lateral control toggle
        if error_type == "lateral" and not hasattr(self, 'use_lateral_control'):
            return "none"
        
        # Categorize based on thresholds
        if abs_error <= deadband:
            return "none"
        elif abs_error <= small_threshold:
            return "small"
        elif abs_error <= medium_threshold:
            return "medium"
        elif abs_error <= medium_high_threshold:
            return "medium_high"
        elif abs_error <= medium_large_threshold:
            return "medium_large"
        else:
            return "large"
    
    def _determine_movement_strategy(self, distance_error, lateral_error, angular_error_degrees):
        """
        Determine the optimal movement strategy using table-driven approach
        with blending between strategies for smooth transitions.
        
        Args:
            distance_error: Error in distance (meters)
            lateral_error: Error in lateral position (meters)
            angular_error_degrees: Error in angular position (degrees)
            
        Returns:
            dict: Strategy information including strategy name, movement flags, and scale factors
        """
        current_time = time.time()
        
        # Categorize errors into states: "none", "small", "medium", "medium_large", "large"
        self._key_tuple[0] = self._categorize_error(distance_error, "distance")
        self._key_tuple[1] = self._categorize_error(lateral_error, "lateral")
        self._key_tuple[2] = self._categorize_error(angular_error_degrees, "angular")
        
        # Create lookup key
        key = tuple(self._key_tuple)
        
        # Get strategy definition from table
        strategy_def = self._match_strategy(key, self.strategy_table)
        
        # Format the reason string with actual error values
        name, use_forward, use_lateral, use_angular, forward_scale, lateral_scale, angular_scale, reason_template = strategy_def
        
        reason = reason_template.format(
            distance_error=abs(distance_error),
            lateral_error=abs(lateral_error),
            angular_error=abs(angular_error_degrees)
        )
        
        # Create a MovementStrategy object
        target_strategy = MovementStrategy(
            name, use_forward, use_lateral, use_angular,
            forward_scale, lateral_scale, angular_scale, reason
        )
        
        # Update the strategy blender with the target strategy
        blend_started = self.strategy_blender.update_target(target_strategy, current_time)
        
        # Get the current (possibly blended) strategy
        current_strategy = self.strategy_blender.get_current_strategy(current_time)
        
        # Keep track of the current strategy name for logging
        self.current_strategy = current_strategy.name
        
        self.get_logger().info(
            f"Strategy selected: {current_strategy.name}, params: forward={current_strategy.forward_scale:.1f}, "
            f"lateral={current_strategy.lateral_scale:.1f}, angular={current_strategy.angular_scale:.1f}"
        )

        # Convert to dictionary for compatibility with existing code
        return current_strategy.as_dict()
    
    def _match_strategy(self, key, strategies):
        """
        Match a key against the strategy table with wildcard support.
        
        Args:
            key: Tuple of (distance_state, lateral_state, angular_state)
            strategies: Strategy table dictionary
            
        Returns:
            List: The matched strategy definition
        """
        # First try exact match
        if key in strategies:
            return strategies[key]
        
        # Support wildcards
        d_state, l_state, a_state = key
        
        # Try wildcard matches in order of specificity
        # First: match two specific states, one wildcard
        patterns_to_try = [
            # Two specific, one wildcard
            (d_state, l_state, "*"),
            (d_state, "*", a_state),
            ("*", l_state, a_state),
            
            # One specific, two wildcards
            (d_state, "*", "*"),
            ("*", l_state, "*"),
            ("*", "*", a_state),
        ]
        
        for pattern in patterns_to_try:
            if pattern in strategies:
                return strategies[pattern]
        
        # Fallback
        return strategies[("*", "*", "*")]
    
    def _log_periodic_status(self):
        """Log periodic status updates."""
        # Only log if debug level supports it
        if self.debug_level < 1:
            return
            
        # Get current CPU usage
        cpu_usage = self.current_cpu_usage
        
        # Calculate cycle time stats
        cycle_time_avg = 0.0
        if hasattr(self, 'cycle_duration_avg'):
            cycle_time_avg = self.cycle_duration_avg * 1000.0  # Convert to ms
        
        # Log status
        self.get_logger().info(
            f"Status: Robot state={self.robot_state}, "
            f"Strategy={self.current_strategy}, "
            f"CPU={cpu_usage:.1f}%, "
            f"Cycle time={cycle_time_avg:.2f}ms, "
            f"Update rate={self.update_rate:.1f}Hz"
        )
    
    def publish_diagnostics(self):
        """Publish detailed diagnostic information at a slower rate."""
        if self._shutting_down:
            return
            
        # Calculate velocity statistics
        vel_data = self.velocity_history.get_all()
        if not vel_data:
            return
            
        # Extract linear and angular velocities
        lin_x_velocities = [v[0] for v in vel_data]
        lin_y_velocities = [v[1] for v in vel_data]
        ang_velocities = [v[2] for v in vel_data]
        
        # Calculate statistics
        avg_lin_x_vel = sum(lin_x_velocities) / len(lin_x_velocities) if lin_x_velocities else 0
        avg_lin_y_vel = sum(lin_y_velocities) / len(lin_y_velocities) if lin_y_velocities else 0
        avg_ang_vel = sum(ang_velocities) / len(ang_velocities) if ang_velocities else 0
        
        # Only log if time interval has passed (throttling)
        current_time = time.time()
        if (current_time - self.last_diag_log_time) < LOG_THROTTLE_DIAG:
            return
            
        self.last_diag_log_time = current_time
        
        # Log detailed information
        if self.debug_level >= 1:
            diag_msg = (
                f"DIAG: Avg Vel=[{avg_lin_x_vel:.2f}, {avg_lin_y_vel:.2f}, {avg_ang_vel:.2f}], "
                f"Strategy={self.current_strategy}, "
                f"E=[{self.distance_error_tracker.current_error:.2f}m, "
                f"{self.lateral_error_tracker.current_error:.2f}m, "
                f"{math.degrees(self.angular_error_tracker.current_error):.1f}°]"
            )
            
            self.get_logger().info(diag_msg)
        
        # Publish performance metrics
        self._publish_performance_metrics()
    
    def _publish_performance_metrics(self):
        """Publish performance metrics for monitoring."""
        # Calculate CPU average
        cpu_avg = 0.0
        if self.performance_stats['cpu']:
            cpu_avg = sum(self.performance_stats['cpu']) / len(self.performance_stats['cpu'])
        
        # Calculate cycle time average
        cycle_time_avg = 0.0
        if self.performance_stats['control_cycles']:
            cycle_time_avg = sum(self.performance_stats['control_cycles']) / len(self.performance_stats['control_cycles'])
            cycle_time_avg *= 1000.0  # Convert to ms
        
        # Create performance message
        performance_msg = String()
        performance_msg.data = (
            f'{{"cpu": {cpu_avg:.1f}, '
            f'"cycle_time_ms": {cycle_time_avg:.2f}, '
            f'"strategy": "{self.current_strategy}", '
            f'"skips": {self.performance_stats["control_skips"]}, '
            f'"update_rate": {self.update_rate:.1f}}}'
        )
        
        # Publish
        self.performance_pub.publish(performance_msg)
    
    def prepare_shutdown(self):
        """Prepare for node shutdown."""
        self.get_logger().info("Preparing for shutdown")
        
        # Set shutdown flag
        self._shutting_down = True
        
        # Immediately stop the robot
        try:
            self.stop_robot()
            self.get_logger().info("Robot motion stopped - velocity and rotation set to 0")
        except Exception as e:
            self.get_logger().error(f"Error stopping robot during shutdown: {str(e)}")

# Main function
def main(args=None):
    """Main function to initialize and run the PID Controller node."""
    rclpy.init(args=args)
    node = EnhancedPIDControllerNode()
    
    # Welcome message
    print("=================================================")
    print("Enhanced PID Controller for Basketball Tracking Robot")
    print("=================================================")
    print("This node implements several enhancements:")
    print("1. Coordinated angular-lateral control")
    print("2. Gradual strategy transitions")
    print("3. Adaptive gain system")
    print("4. Target filtering")
    print("5. Optimized transform handling with caching")
    print("6. Resource monitoring and adaptive control rate")
    print("")
    print("Press Ctrl+C to stop the program")
    print("=================================================")
    
    # Register shutdown handler for proper cleanup
    def signal_handler(sig, frame):
        print(f"\nSignal {sig} received, stopping robot...")
        # Prepare for shutdown
        node.prepare_shutdown()
        # Then proceed with ROS shutdown
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Enhanced PID Controller shutdown requested via Ctrl+C")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {str(e)}")
    finally:
        # Stop the robot before shutdown
        node.prepare_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()