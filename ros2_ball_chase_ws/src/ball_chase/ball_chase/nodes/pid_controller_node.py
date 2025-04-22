"""
Basketball Tracking Robot - Improved PID Controller Node
=======================================================

This controller implements efficient movement patterns for a mecanum-wheeled
basketball tracking robot with several enhancements:
- Angular-first control strategy for diagonal movements
- Fast strategy transitions for responsive tracking
- Enhanced integral term management to prevent windup
- Coordinated angular-lateral control with balanced parameters
- Balanced error thresholds and hysteresis for smooth behavior
- Continuous motion tracking with trajectory prediction
- Optimized for Raspberry Pi 5 performance with resource monitoring
"""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped, Vector3, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray
from tf2_ros import TransformListener, Buffer
import math
import time
import numpy as np
import signal
import sys
from collections import deque
import psutil  # For CPU monitoring
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pid_controller')

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
        "performance": "/pid/performance"    # Performance metrics topic
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
        if self.count == 0:
            return []
        
        result = []
        
        if self.count < self.max_size:
            # Buffer isn't full yet, return all items from 0 to count
            for i in range(self.count):
                result.append(self.data[i])
        else:
            # Buffer is full, need to wrap around
            # Start from oldest item (at next_index) and go around
            for i in range(self.max_size):
                idx = (self.next_index + i) % self.max_size
                result.append(self.data[idx])
        
        return result
        
    def get_latest(self, n=1):
        """Get the latest n values (default is just the latest one)."""
        if self.count == 0:
            return []
        
        n = min(n, self.count)
        result = []
        
        # Calculate positions of the n latest elements
        for i in range(n):
            if self.count < self.max_size:
                # Simple case: buffer isn't full yet
                idx = self.count - n + i
            else:
                # Buffer is full, handle circular indexing
                idx = (self.next_index - n + i) % self.max_size
            
            # Append item at calculated index
            result.append(self.data[idx])
        
        return result
    
    def clear(self):
        """Clear the buffer."""
        self.next_index = 0
        self.count = 0
        # Reset all data to default values
        for i in range(self.max_size):
            self.data[i] = (0.0, 0.0, 0.0)

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

class EnhancedTargetFilter:
    """Enhanced filter for target position data with better motion prediction."""
    
    def __init__(self, buffer_size=8, prediction_horizon=0.3):
        self.position_buffer = deque(maxlen=buffer_size)
        self.prediction_horizon = prediction_horizon  # seconds
        self.last_update_time = None
        self.current_velocity = (0.0, 0.0, 0.0)  # x, y, angular
        self.filtered_position = None
        self.predicted_position = None
        self.acceleration = (0.0, 0.0, 0.0)  # x, y, angular acceleration
        self.is_moving = False
        self.direction_change_detected = False
        self.motion_direction = (0.0, 0.0, 0.0)  # normalized direction vector
        self.trajectory_history = deque(maxlen=10)
        self.movement_consistency = 0.0  # 0.0-1.0 measure of consistent movement
        
    def update(self, position, timestamp=None):
        """
        Update the filter with a new position measurement.
        
        Args:
            position: Tuple of (x, y, angle) for the target position
            timestamp: Time of measurement (defaults to current time)
        
        Returns:
            tuple: Filtered position
        """
        current_time = timestamp if timestamp is not None else time.time()
        
        # Add to buffer with timestamp
        self.position_buffer.append((position[0], position[1], position[2], current_time))
        
        # Add to trajectory history
        self.trajectory_history.append((position, current_time))
        
        # Calculate filtered position (weighted average with recent samples)
        if len(self.position_buffer) >= 3:
            # Get the three most recent positions manually without slicing
            buffer_size = len(self.position_buffer)
            recent = []
            for i in range(buffer_size - 3, buffer_size):
                recent.append(self.position_buffer[i])
            
            # Assign higher weights to more recent measurements
            weights = [0.2, 0.3, 0.5]  # More weight to recent measurements
            
            # Calculate weighted average for each dimension
            x_sum = sum(pos[0] * w for pos, w in zip(recent, weights))
            y_sum = sum(pos[1] * w for pos, w in zip(recent, weights))
            angle_sum = sum(pos[2] * w for pos, w in zip(recent, weights))
            
            self.filtered_position = (x_sum, y_sum, angle_sum)
        else:
            self.filtered_position = position
            
        # Calculate velocity and acceleration if we have enough data
        if len(self.position_buffer) >= 2 and self.last_update_time is not None:
            dt = current_time - self.last_update_time
            if dt > 0.001:  # Avoid division by zero
                # Calculate velocity
                # Get the two most recent positions without slicing
                buffer_size = len(self.position_buffer)
                prev_pos = self.position_buffer[buffer_size - 2]
                curr_pos = self.position_buffer[buffer_size - 1]
                
                vx = (curr_pos[0] - prev_pos[0]) / dt
                vy = (curr_pos[1] - prev_pos[1]) / dt
                v_angle = (curr_pos[2] - prev_pos[2]) / dt
                
                # Check for direction changes
                prev_vel = self.current_velocity
                new_vel = (vx, vy, v_angle)
                
                # Detect significant direction changes
                self.direction_change_detected = False
                if prev_vel[0] != 0 and prev_vel[1] != 0:
                    # Calculate dot product to check direction change
                    dot_product = prev_vel[0] * new_vel[0] + prev_vel[1] * new_vel[1]
                    prev_mag = math.sqrt(prev_vel[0]**2 + prev_vel[1]**2)
                    new_mag = math.sqrt(new_vel[0]**2 + new_vel[1]**2)
                    
                    # Avoid division by zero
                    if prev_mag > 0.01 and new_mag > 0.01:
                        cos_angle = dot_product / (prev_mag * new_mag)
                        # Consider significant direction change if angle > 30 degrees
                        if cos_angle < 0.866:  # cos(30°) ≈ 0.866
                            self.direction_change_detected = True
                
                # Calculate acceleration if we have at least 3 points
                if len(self.position_buffer) >= 3 and dt > 0.001:
                    # Simple acceleration calculation 
                    ax = (vx - self.current_velocity[0]) / dt
                    ay = (vy - self.current_velocity[1]) / dt
                    a_angle = (v_angle - self.current_velocity[2]) / dt
                    
                    # Low-pass filter for acceleration
                    alpha_a = 0.3  # Lower value means more smoothing
                    self.acceleration = (
                        alpha_a * ax + (1 - alpha_a) * self.acceleration[0],
                        alpha_a * ay + (1 - alpha_a) * self.acceleration[1],
                        alpha_a * a_angle + (1 - alpha_a) * self.acceleration[2]
                    )
                
                # Smooth velocity estimate with low-pass filter
                alpha = 0.6  # Decreased from 0.7 to be less reactive to noise
                self.current_velocity = (
                    alpha * vx + (1 - alpha) * self.current_velocity[0],
                    alpha * vy + (1 - alpha) * self.current_velocity[1],
                    alpha * v_angle + (1 - alpha) * self.current_velocity[2]
                )
                
                # Calculate normalized direction vector
                vel_magnitude = math.sqrt(self.current_velocity[0]**2 + self.current_velocity[1]**2)
                if vel_magnitude > 0.01:  # Only update if moving significantly
                    new_direction = (
                        self.current_velocity[0] / vel_magnitude,
                        self.current_velocity[1] / vel_magnitude
                    )
                    
                    # Smooth direction updates
                    alpha_dir = 0.7
                    self.motion_direction = (
                        alpha_dir * new_direction[0] + (1 - alpha_dir) * self.motion_direction[0],
                        alpha_dir * new_direction[1] + (1 - alpha_dir) * self.motion_direction[1],
                        self.current_velocity[2]  # Angular component
                    )
                
                # Determine if target is consistently moving
                vel_threshold = 0.05  # m/s
                self.is_moving = vel_magnitude > vel_threshold
                
                # Calculate movement consistency (higher values = more consistent trajectory)
                if len(self.trajectory_history) >= 5:
                    # Get recent positions without slicing
                    recent_positions = []
                    traj_size = len(self.trajectory_history)
                    for i in range(max(0, traj_size - 5), traj_size):
                        recent_positions.append(self.trajectory_history[i][0])
                    
                    # Calculate average displacement direction
                    dx = recent_positions[-1][0] - recent_positions[0][0]
                    dy = recent_positions[-1][1] - recent_positions[0][1]
                    
                    # Check how well each segment aligns with overall direction
                    consistency_sum = 0
                    count = 0
                    
                    for i in range(1, len(recent_positions)):
                        segment_dx = recent_positions[i][0] - recent_positions[i-1][0]
                        segment_dy = recent_positions[i][1] - recent_positions[i-1][1]
                        
                        # Skip tiny movements
                        segment_len = math.sqrt(segment_dx**2 + segment_dy**2)
                        if segment_len < 0.01:
                            continue
                            
                        # Normalize
                        segment_dx /= segment_len
                        segment_dy /= segment_len
                        
                        # Calculate alignment using dot product
                        overall_len = math.sqrt(dx**2 + dy**2)
                        if overall_len > 0.01:
                            overall_dx = dx / overall_len
                            overall_dy = dy / overall_len
                            
                            # Dot product (1.0 = perfect alignment, -1.0 = opposite direction)
                            alignment = segment_dx * overall_dx + segment_dy * overall_dy
                            consistency_sum += max(0, alignment)  # Only count positive alignment
                            count += 1
                    
                    # Update consistency metric
                    if count > 0:
                        self.movement_consistency = consistency_sum / count
                    else:
                        self.movement_consistency = 0.0
        
        # Make better prediction for future position
        if self.filtered_position is not None:
            # Base prediction on current velocity and acceleration
            if self.is_moving and not self.direction_change_detected:
                # Calculate position using physics formulas with quadratic acceleration term
                # x = x₀ + v₀t + ½at²
                t = self.prediction_horizon
                
                # More weight to acceleration when consistent movement is detected
                accel_weight = 0.5 * self.movement_consistency
                
                pred_x = (self.filtered_position[0] + 
                        self.current_velocity[0] * t + 
                        0.5 * self.acceleration[0] * t**2 * accel_weight)
                
                pred_y = (self.filtered_position[1] + 
                        self.current_velocity[1] * t + 
                        0.5 * self.acceleration[1] * t**2 * accel_weight)
                
                pred_angle = (self.filtered_position[2] + 
                            self.current_velocity[2] * t)
                
                self.predicted_position = (pred_x, pred_y, pred_angle)
            else:
                # For non-consistent movement or after direction changes,
                # use simpler prediction that's less sensitive to noise
                t = self.prediction_horizon
                pred_x = self.filtered_position[0] + self.current_velocity[0] * t * 0.7
                pred_y = self.filtered_position[1] + self.current_velocity[1] * t * 0.7
                pred_angle = self.filtered_position[2] + self.current_velocity[2] * t * 0.7
                
                self.predicted_position = (pred_x, pred_y, pred_angle)
        else:
            self.predicted_position = position
            
        self.last_update_time = current_time
        return self.filtered_position
    
    def get_filtered_position(self):
        """Get the current filtered position."""
        return self.filtered_position if self.filtered_position else (
            self.position_buffer[-1][:3] if self.position_buffer else None
        )
    
    def get_predicted_position(self):
        """Get the predicted future position based on velocity and acceleration."""
        return self.predicted_position
        
    def get_velocity(self):
        """Get the current velocity estimate."""
        return self.current_velocity
    
    def get_acceleration(self):
        """Get the current acceleration estimate."""
        return self.acceleration
    
    def get_recent_positions(self, n=3):
        """Get the n most recent positions."""
        # Use individual indexing instead of slicing
        result = []
        
        # Get the last n positions manually
        positions = []
        count = len(self.position_buffer)
        for i in range(max(0, count - n), count):
            positions.append(self.position_buffer[i])
        
        # Extract the first 3 elements of each position
        for p in positions:
            result.append(p[:3])
        
        return result
    
    def get_movement_info(self):
        """Get information about the movement characteristics."""
        return {
            'is_moving': self.is_moving,
            'direction_change': self.direction_change_detected,
            'consistency': self.movement_consistency,
            'velocity_magnitude': math.sqrt(self.current_velocity[0]**2 + self.current_velocity[1]**2)
        }
    
    def reset(self):
        """Reset the filter state."""
        self.position_buffer.clear()
        self.trajectory_history.clear()
        self.last_update_time = None
        self.current_velocity = (0.0, 0.0, 0.0)
        self.acceleration = (0.0, 0.0, 0.0)
        self.filtered_position = None
        self.predicted_position = None
        self.is_moving = False
        self.direction_change_detected = False
        self.motion_direction = (0.0, 0.0, 0.0)
        self.movement_consistency = 0.0

class ErrorTracker:
    """Lightweight error tracker that monitors error values over time."""
    
    def __init__(self, name, max_history=8):
        """Initialize error tracker with efficient storage."""
        self.name = name
        self.current_error = 0.0
        self.previous_error = 0.0
        self.previous_category = None  # For tracking error category with hysteresis
        # Pre-allocate error history with zeros to avoid frequent allocations
        self.error_history = deque([0.0] * max_history, maxlen=max_history)
        self.last_correction_time = 0.0
        self.sign_changes = 0  # Count of error sign changes (useful for oscillation detection)
        self.accumulated_error = 0.0
        self.decay_factor = 0.9  # Simplified decay factor
        self.error_increasing = False
        self.peak_error = 0.0
        self.last_sign = 0  # -1, 0, or 1
        
    def update(self, error, dt):
        """Update error tracking with new error value."""
        # Store previous error and sign
        self.previous_error = self.current_error
        prev_sign = 1 if self.current_error > 0 else (-1 if self.current_error < 0 else 0)
        
        # Update current error
        self.current_error = error
        current_sign = 1 if error > 0 else (-1 if error < 0 else 0)
        
        # Check for sign change
        if prev_sign != 0 and current_sign != 0 and prev_sign != current_sign:
            self.sign_changes += 1
            
        # Update error history
        self.error_history.append(error)
        
        # Track if error is increasing
        self.error_increasing = abs(error) > abs(self.previous_error) * 1.05  # 5% threshold
        
        # Update peak error if current error is larger
        if abs(error) > abs(self.peak_error):
            self.peak_error = error
        
        # Apply different decay rates based on error direction
        if current_sign == prev_sign:
            # Same direction - standard decay
            decay = self.decay_factor
        else:
            # Direction change - faster decay
            decay = self.decay_factor * 0.5
            
        # Update accumulated error with direction-aware decay
        self.accumulated_error = (self.accumulated_error + error * dt) * decay
        
        # Store last sign
        self.last_sign = current_sign
    
    def reset(self):
        """Reset all tracked errors."""
        self.current_error = 0.0
        self.previous_error = 0.0
        self.error_history.clear()
        # Refill with zeros
        for _ in range(self.error_history.maxlen):
            self.error_history.append(0.0)
        self.accumulated_error = 0.0
        self.sign_changes = 0
        self.error_increasing = False
        self.peak_error = 0.0
        self.last_sign = 0
        self.previous_category = None
    
    def is_error_growing(self):
        """Check if error is growing compared to previous value."""
        return self.error_increasing
        
    def record_correction(self):
        """Record that a correction was made for this error."""
        self.last_correction_time = time.time()
        # Reduce accumulated error when correction is made
        self.accumulated_error *= 0.5
        
    def get_trend(self, n=3):
        """Calculate trend of error (increasing/decreasing)."""
        if len(self.error_history) < n:
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
    
    def is_oscillating(self):
        """Determine if the error is oscillating."""
        # Consider oscillating if there are multiple sign changes recently
        return self.sign_changes >= 2

class ImprovedPID:
    """PID controller with enhanced integral handling and adaptive gains."""
    
    def __init__(self, base_kp, base_ki, base_kd, output_min, output_max, name="PID"):
        """Initialize the improved PID controller."""
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
        
        # Performance metrics
        self.settling_time = 0.0
        self.overshoot = 0.0
        self.rise_time = 0.0
        self.steady_state = False
        
        # Logger for controller-specific logs
        self.logger = logging.getLogger(f'pid_controller.{name}')
        
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
        
        # For angular control specifically - reduced integral effect
        if self.name == "Angular":
            # Reduced integral gain for angular control
            ki_factor *= 0.7
            # Reduced derivative gain to prevent overshoot
            kd_factor *= 0.6
        
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
        
        # Reset performance metrics
        self.settling_time = 0.0
        self.overshoot = 0.0
        self.rise_time = 0.0
        self.steady_state = False
        
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
            # Throttled logging - only log every few calls for this controller
            should_log = hasattr(self, 'compute_count') and self.compute_count % 10 == 0
            if not hasattr(self, 'compute_count'):
                self.compute_count = 0
            self.compute_count += 1
            
            if should_log:
                self.logger.info(f"PID {self.name} compute: error={error:.3f}, force_zero={force_zero}")

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
                
                # Much more aggressive integral reset on zero crossing for angular movement
                if self.name == "Angular":
                    # More aggressive integral reset
                    self.integral *= 0.05  # Much more aggressive (was 0.2 or 0.3)
                elif self.name == "Linear Y":
                    # More aggressive for lateral controller
                    self.integral *= 0.1  # More aggressive than before (was 0.2)
                else:
                    # Default behavior for other controllers
                    self.integral *= 0.2
            else:
                # No sign change - gradually reduce sign change count for hysteresis
                self.sign_change_count = max(0, self.sign_change_count - 0.1)
                
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
                        # For Angular controller, even more cautious integral accumulation
                        if self.name == "Angular":
                            self.integral += error * dt * 0.6  # Reduced accumulation rate
                        else:
                            self.integral += error * dt * 1.2  # Boost integral after crossing zero
                    else:
                        self.integral *= 0.5  # More aggressively reduce integral near zero
                else:
                    # Normal integral handling
                    if abs(error) > self.integral_deadband:
                        # For Angular controller, reduce integral accumulation rate
                        if self.name == "Angular":
                            self.integral += error * dt * 0.7  # 30% slower accumulation
                        else:
                            self.integral += error * dt
                    else:
                        # More aggressively reduce integral term when close to target
                        self.integral *= self.integral_decay
                    
                # Apply integral limit to prevent excessive buildup
                # More restrictive limit for Angular controller
                if self.name == "Angular":
                    max_integral = self.max_integral * 0.7  # 30% smaller limit for angular
                else:
                    max_integral = self.max_integral
                    
                self.integral = max(-max_integral, min(max_integral, self.integral))
                
                i_term = self.ki * self.integral
            
            # Derivative term with improved noise handling
            # Use filtered error derivative to reduce noise sensitivity
            error_change = error - self.prev_error
            
            # Enhanced derivative handling
            if self.sign_change_count >= 2:
                # If oscillating (multiple sign changes), amplify derivative term
                # But less amplification for Angular controller to prevent overshoot
                if self.name == "Angular":
                    d_term = self.kd * error_change / max(dt, 0.001) * 1.0  # No amplification
                else:
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
                    if self.name == "Angular":
                        self.integral *= 0.05  # Much more aggressive for angular
                    elif self.name == "Linear Y":
                        self.integral *= 0.1  # More aggressive for lateral
                    else:
                        self.integral *= 0.2  # Standard value
                else:
                    if self.name == "Angular":
                        self.integral *= 0.1  # More aggressive than before for angular
                    elif self.name == "Linear Y":
                        self.integral *= 0.2
                    else:
                        self.integral *= 0.3
                i_term = self.ki * self.integral
                    
            # Apply transition smoothing for rapid control changes
            # Less smoothing for Angular controller to improve responsiveness
            smoothing_factor = 0.4 if self.name == "Angular" else 0.6
            if abs(output_limited - self.last_output) > (self.output_max - self.output_min) * 0.4:
                # Blend between previous and current output for large changes
                output_limited = smoothing_factor * output_limited + (1 - smoothing_factor) * self.last_output
                
            # Save individual terms for diagnostics
            self.last_p_term = p_term
            self.last_i_term = i_term
            self.last_d_term = d_term
            self.last_output = output_limited
            
            # Save state for next iteration
            self.prev_error = error
            self.last_time = current_time
            
            if should_log:
                self.logger.info(f"PID {self.name} terms: P={self.last_p_term:.3f}, I={self.last_i_term:.3f}, D={self.last_d_term:.3f}")

            # Ensure we return a proper float value
            return float(output_limited)
            
        except Exception as e:
            # Log error and return safe value
            self.logger.error(f"Error in PID compute: {str(e)}")
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
        
        # Default configuration with improved values
        self.config = {
            'coupling_factor': 0.4,         # Reduced from 0.7 to allow more lateral movement
            'min_angle_for_reduction': 0.1,  # ~5.7 degrees
            'zero_angle_threshold': 0.03,    # ~1.7 degrees
            'max_angle_factor': 0.3,         # Reduced from 0.5 to be less aggressive
            'same_sign_scale': 0.8,          # Scaling when errors have same sign
            'opposite_sign_scale': 1.2,      # Increased from 1.0 to prioritize when errors help each other
            'smoothing_factor': 0.4,         # Reduced from 0.6 for faster response
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
            
        # State variables
        self.last_lateral_velocity = 0.0
        self.last_angular_velocity = 0.0
        self.last_update_time = None
        
        # Logger
        self.logger = logging.getLogger('pid_controller.coordinated')
    
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
        
        # Control strategy adaptation - prioritize angular control for larger angular errors
        angular_magnitude = abs(angular_error)
        large_angle_threshold = 0.1  # ~5.7 degrees
        
        # Angular-first strategy - for larger angular errors, compute angular first
        # and modulate lateral based on angular progress
        if angular_magnitude > large_angle_threshold:
            # Get raw angular velocity first
            raw_angular_velocity = self.angular_pid.compute(
                angular_error, 
                current_time, 
                force_zero=False,
                error_trend=angular_trend
            )
            
            # Then compute lateral velocity, potentially with reduced effect
            angular_progress = 1.0 - min(1.0, angular_magnitude / self.config['max_angle_factor'])
            
            # Scale lateral control based on angular progress 
            lateral_force_zero = angular_magnitude > (large_angle_threshold * 2)
            
            raw_lateral_velocity = self.linear_pid.compute(
                lateral_error, 
                current_time, 
                force_zero=lateral_force_zero,
                error_trend=lateral_trend
            )
            
            # Gradually phase in lateral control as angular error reduces
            if not lateral_force_zero:
                # Apply scaling that increases as angular error decreases
                raw_lateral_velocity *= angular_progress
                
        else:
            # For smaller angular errors, compute both normally
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
            
            # Calculate lateral velocity reduction - more moderate than before
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
            # Reduced smoothing factor for more responsive control
            smoothing = self.config['smoothing_factor']
            
            lateral_velocity = self.last_lateral_velocity * (1 - smoothing) + \
                              lateral_velocity * smoothing
                              
            angular_velocity = self.last_angular_velocity * (1 - smoothing) + \
                              angular_velocity * smoothing
        
        # Store values for next iteration
        self.last_lateral_velocity = lateral_velocity
        self.last_angular_velocity = angular_velocity
        self.last_update_time = current_time
        
        # Log coordination details occasionally
        if hasattr(self, 'compute_count') and self.compute_count % 20 == 0:
            coupling_str = f"{normalized_angle * self.config['coupling_factor']:.2f}" if 'normalized_angle' in locals() else "N/A"
            self.logger.info(
                f"Coordinated control: lateral_err={lateral_error:.3f}, angular_err={angular_error:.3f}, "
                f"lateral_vel={lateral_velocity:.3f}, angular_vel={angular_velocity:.3f}, "
                f"coupling={coupling_str}")
            
        if not hasattr(self, 'compute_count'):
            self.compute_count = 0
        self.compute_count += 1
        
        return lateral_velocity, angular_velocity
    
    def reset(self):
        """Reset the controller state."""
        self.linear_pid.reset()
        self.angular_pid.reset()
        self.last_lateral_velocity = 0.0
        self.last_angular_velocity = 0.0
        self.last_update_time = None
        if hasattr(self, 'compute_count'):
            self.compute_count = 0

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
    
    def __init__(self, blend_duration=0.2):  # Reduced from 0.5 to 0.2 seconds
        """Initialize the strategy blender with faster transitions."""
        self.current_strategy = None
        self.target_strategy = None
        self.blend_start_time = 0.0
        self.blending_active = False
        self.blend_duration = blend_duration
        self.direction_change_boost = 2.0  # Speed up transitions when direction changes
        self.previous_direction = None
        
        # Logger
        self.logger = logging.getLogger('pid_controller.blender')
    
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
            self.previous_direction = self._get_strategy_direction(target_strategy)
            return False
            
        # Check if target is different from current
        if target_strategy.name != self.current_strategy.name:
            # Detect direction change for boosting transition speed
            current_direction = self._get_strategy_direction(target_strategy)
            direction_change = False
            
            if self.previous_direction is not None and current_direction is not None:
                # Check if direction components are opposite
                direction_change = (
                    (self.previous_direction[0] * current_direction[0] < 0) or
                    (self.previous_direction[1] * current_direction[1] < 0) or
                    (self.previous_direction[2] * current_direction[2] < 0)
                )
            
            # Start new blend
            self.target_strategy = target_strategy
            self.blend_start_time = current_time
            self.blending_active = True
            
            # Apply boosting for direction changes
            if direction_change:
                # Use shorter blend duration for direction changes
                self.effective_blend_duration = self.blend_duration / self.direction_change_boost
                self.logger.info(f"Direction change detected: boosting transition speed")
            else:
                self.effective_blend_duration = self.blend_duration
            
            # Update previous direction
            self.previous_direction = current_direction
            
            return True
            
        return False
    
    def _get_strategy_direction(self, strategy):
        """Extract movement direction from a strategy."""
        if not (strategy.use_forward or strategy.use_lateral or strategy.use_angular):
            return None
            
        return (
            1 if strategy.use_forward and strategy.forward_scale > 0 else 
            (-1 if strategy.use_forward else 0),
            
            1 if strategy.use_lateral and strategy.lateral_scale > 0 else 
            (-1 if strategy.use_lateral else 0),
            
            1 if strategy.use_angular and strategy.angular_scale > 0 else 
            (-1 if strategy.use_angular else 0)
        )
    
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
        
        # Use effective blend duration that might be boosted for direction changes
        blend_duration = getattr(self, 'effective_blend_duration', self.blend_duration)
        
        linear_blend = min(1.0, elapsed_time / blend_duration)
        blend_factor = self._smoothstep(linear_blend)
        
        # Check if blending is complete
        if blend_factor >= 0.999:
            self.current_strategy = self.target_strategy
            self.blending_active = False
            return self.current_strategy
        
        # Create blended strategy
        name = f"{self.current_strategy.name}→{self.target_strategy.name}"
        
        # Determine boolean flags using OR logic for smoother transitions
        # e.g., if either strategy uses forward, the blended strategy should use it
        # This prevents sudden stopping of a movement axis during transitions
        use_forward = self.target_strategy.use_forward or (
            self.current_strategy.use_forward and blend_factor < 0.5)
            
        use_lateral = self.target_strategy.use_lateral or (
            self.current_strategy.use_lateral and blend_factor < 0.5)
            
        use_angular = self.target_strategy.use_angular or (
            self.current_strategy.use_angular and blend_factor < 0.5)
        
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
            
        # Use effective blend duration for calculating progress
        blend_duration = getattr(self, 'effective_blend_duration', self.blend_duration)
        elapsed_time = current_time - self.blend_start_time
        return min(100.0, (elapsed_time / blend_duration) * 100.0)
    
    def reset(self):
        """Reset the blender state."""
        self.current_strategy = None
        self.target_strategy = None
        self.blending_active = False
        self.previous_direction = None

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
        self.logger = logging.getLogger('pid_controller.resource_monitor')
    
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
        
    def log_stats(self):
        """Log current resource statistics."""
        self.logger.info(f"CPU: {self.cpu_usage:.1f}%, Memory: {self.memory_usage:.1f}%")

class ImprovedPIDControllerNode(Node):
    """Enhanced PID Controller node with improved movement strategy and error handling."""
    
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
        self.target_filter = EnhancedTargetFilter(buffer_size=8, prediction_horizon=0.3)
        
        # Initialize strategy blender (faster transition time)
        self.strategy_blender = StrategyBlender(blend_duration=0.2)
        
        # Log startup info
        self.get_logger().info("Improved PID Controller initialized with angular-first strategy")
    
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
        self.min_update_rate = 5.0   # Don't go below 5Hz
        self.max_update_rate = 20.0  # Don't go above 20Hz
        
        # Performance metrics tracking
        self.cycle_start_time = 0.0
        self.cycle_duration_avg = 0.0
        self.skip_next_cycle = False
        
    def _declare_parameters(self):
        """Declare and get all node parameters with improved defaults."""
        # Most parameter declarations are omitted for brevity but would be here
        
        # Declare parameters with improved defaults
        self.declare_parameters(
            namespace='',
            parameters=[
                # Linear X velocity PID parameters
                ('linear_x_kp', 1.0),
                ('linear_x_ki', 0.03),
                ('linear_x_kd', 0.1),
                ('linear_x_min', 0.0),
                ('linear_x_max', 0.2),
                
                # Linear Y velocity PID parameters - improved lateral damping
                ('linear_y_kp', 0.6),
                ('linear_y_ki', 0.06),  # Reduced from 0.08
                ('linear_y_kd', 0.14),  # Increased from 0.12
                ('linear_y_min', -0.2),
                ('linear_y_max', 0.2),
                
                # Angular velocity PID parameters - improved to prevent overshoot
                ('angular_kp', 1.35),  # Reduced from 1.5
                ('angular_ki', 0.035), # Reduced from 0.05
                ('angular_kd', 0.5),   # Reduced from 0.8
                ('angular_min', -0.5),
                ('angular_max', 0.5),
                
                # Control parameters
                ('min_distance', 0.9),
                ('max_distance', 2.0),
                ('target_offset_x', 0.0),
                ('target_offset_y', 0.0),
                ('target_update_rate', 10.0),
                ('diagnostics_rate', 0.5),
                ('debug_level', 1),
                ('adaptive_gains', True),
                ('use_lateral_control', True),
                
                # Balanced error thresholds
                ('distance_threshold', 0.1),
                ('lateral_threshold', 0.075),  # Increased from 0.05
                ('angular_threshold', 1.5),    # Decreased from 3.0
                
                # Resource monitoring parameters
                ('adaptive_control_rate', True),
                ('enable_resource_monitoring', True),
                ('cpu_high_threshold', 85.0),
                ('cpu_low_threshold', 40.0),
                
                # Performance optimization
                ('enable_transform_caching', True),
                ('transform_cache_ttl', 1.0),
                
                # Strategy configuration
                ('angular_first_control', True),
                ('strategy_blend_duration', 0.2),  # Faster blending
                ('coordinated_movement', True),
                
                # Target filter parameters
                ('filter_buffer_size', 8),
                ('prediction_horizon', 0.3),
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
        
        # Get error thresholds
        self.distance_threshold = self.get_parameter('distance_threshold').value
        self.lateral_threshold = self.get_parameter('lateral_threshold').value
        self.angular_threshold = self.get_parameter('angular_threshold').value
        
        # Get resource monitoring parameters
        self.adaptive_control_rate = self.get_parameter('adaptive_control_rate').value
        self.enable_resource_monitoring = self.get_parameter('enable_resource_monitoring').value
        self.cpu_high_threshold = self.get_parameter('cpu_high_threshold').value
        self.cpu_low_threshold = self.get_parameter('cpu_low_threshold').value
        
        # Get transform parameters
        self.enable_transform_caching = self.get_parameter('enable_transform_caching').value
        self.transform_cache_ttl = self.get_parameter('transform_cache_ttl').value
        
        # Get movement strategy parameters
        self.angular_first_control = self.get_parameter('angular_first_control').value
        self.strategy_blend_duration = self.get_parameter('strategy_blend_duration').value
        self.coordinated_movement = self.get_parameter('coordinated_movement').value
        
        # Target filter parameters
        self.filter_buffer_size = self.get_parameter('filter_buffer_size').value
        self.prediction_horizon = self.get_parameter('prediction_horizon').value
        
        # Target update rate from parameters
        self.update_rate = self.get_parameter('target_update_rate').value
        
        # All other parameter assignments would be here
        self.diagnostics_rate = self.get_parameter('diagnostics_rate').value
        self.debug_level = self.get_parameter('debug_level').value
        
        # Log important parameters
        self.get_logger().info(
            f"Controller parameters: linear_x=[{self.linear_x_kp}, {self.linear_x_ki}, {self.linear_x_kd}], "
            f"linear_y=[{self.linear_y_kp}, {self.linear_y_ki}, {self.linear_y_kd}], "
            f"angular=[{self.angular_kp}, {self.angular_ki}, {self.angular_kd}]"
        )
        
        self.get_logger().info(
            f"Error thresholds: distance={self.distance_threshold}, "
            f"lateral={self.lateral_threshold}, angular={self.angular_threshold}"
        )
               
    def _init_controllers(self):
        """Initialize the controllers with improved tuning."""
        # Create error trackers
        self.distance_error_tracker = ErrorTracker("distance", max_history=8)
        self.lateral_error_tracker = ErrorTracker("lateral", max_history=8)
        self.angular_error_tracker = ErrorTracker("angular", max_history=8)
        
        # Initialize individual PID controllers with improved parameters
        self.pid_linear_x = ImprovedPID(
            self.linear_x_kp, self.linear_x_ki, self.linear_x_kd,
            self.linear_x_min, self.linear_x_max,
            name="Linear X"
        )
        self.pid_linear_x.error_tracker = self.distance_error_tracker
        
        self.pid_linear_y = ImprovedPID(
            self.linear_y_kp, self.linear_y_ki, self.linear_y_kd,
            self.linear_y_min, self.linear_y_max,
            name="Linear Y"
        )
        self.pid_linear_y.error_tracker = self.lateral_error_tracker
        
        self.pid_angular = ImprovedPID(
            self.angular_kp, self.angular_ki, self.angular_kd,
            self.angular_min, self.angular_max,
            name="Angular"
        )
        
        self.pid_angular.error_tracker = self.angular_error_tracker
        
        # Initialize coordinated controller with improved parameters
        self.coordinated_controller = CoordinatedController(
            self.pid_linear_y, 
            self.pid_angular,
            {
                'coupling_factor': 0.4,        # Reduced from 0.7
                'smoothing_factor': 0.4,       # Reduced from 0.6 for faster response
                'min_angle_for_reduction': 0.1,
                'zero_angle_threshold': 0.03,
                'max_angle_factor': 0.3,       # Reduced from 0.5
                'same_sign_scale': 0.8,
                'opposite_sign_scale': 1.2,    # Increased from 1.0
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
        self._key_tuple = ["none", "none", "none"]  # Use list instead of tuple for mutability

        
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
        
        # Filtered values
        self.filtered_distance = 0.0
        self.filtered_lateral = 0.0
        self.filtered_bearing = 0.0
        
        # Motion smoothing
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        self.last_logged_cmd = (0.0, 0.0, 0.0)
        
        # Diagnostic information
        self.cycle_count = 0
        
        # Velocity history
        self.velocity_history = LightweightBuffer(max_size=6)
        
        # Cached values
        self.desired_distance = 1.0  # Will be set properly after parameters are loaded
        
        # Stopped state tracking with hysteresis
        self._robot_stopped = False
        self._stop_time = 0.0
        self._last_stop_position = (0.0, 0.0, 0.0)
        self._movement_hysteresis = 0.0  # Used to prevent oscillating between movement/stopped states
        
        # Movement strategy
        self.current_strategy = "IDLE"
        self.previous_strategy = None
        self.strategy_change_time = time.time()
        self.active_strategy = None  # Holds the current strategy object
        
        # Error categorization state
        self.prev_distance_category = "none"
        self.prev_lateral_category = "none"
        self.prev_angular_category = "none"
        
        # Recovery state tracking
        self.in_recovery = False
        self.recovery_start_time = 0.0
        self.recovery_phase = "none"  # none, stop, orient, approach
        self.force_target_reacquisition = False
        
        # Flag to track if we're shutting down
        self._shutting_down = False

        self.get_logger().info(
            f"Initialized state: desired_distance={self.desired_distance:.3f}m, "
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
            print ("orientation not received....")
            return False
            
        current_time = time.time()
        age = current_time - self.last_orientation_time
        
        # Consider orientation data older than 0.5 seconds as stale
        return age < 0.5
    
    def target_callback(self, msg):
        """Handle target position updates with enhanced filtering."""
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
        
        # Throttled logging for target updates
        if self.debug_level >= 2:
            self._log_throttled(
                self.get_logger().info,
                f"TARGET DATA: frame={frame_id}, pos=({target.x:.3f}, {target.y:.3f}, {target.z:.3f})",
                1.0,  # Throttle to once per second
                'last_target_log_time'
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
        
        # Apply target filtering
        # Update the filter with the new position
        filtered_position = self.target_filter.update(
            (self.current_distance, self.current_lateral, self.current_bearing),
            self.last_target_time
        )
        
        # Force target reacquisition if requested (e.g. after recovery)
        if self.force_target_reacquisition:
            self.target_filter.reset()
            filtered_position = self.target_filter.update(
                (self.current_distance, self.current_lateral, self.current_bearing),
                self.last_target_time
            )
            self.force_target_reacquisition = False
            self.get_logger().info("Forced target reacquisition - filter reset")
        
        # Use filtered/predicted position if available
        if filtered_position:
            # Check if we should use prediction or just filtering
            movement_info = self.target_filter.get_movement_info()
            
            if movement_info['is_moving'] and not movement_info['direction_change']:
                # For consistent movement, use prediction
                predicted_position = self.target_filter.get_predicted_position()
                if predicted_position:
                    # Use prediction for control
                    self.filtered_distance = predicted_position[0]
                    self.filtered_lateral = predicted_position[1]
                    self.filtered_bearing = predicted_position[2]
                    
                    if self.debug_level >= 2:
                        self.get_logger().debug(
                            f"Target prediction: raw=({self.current_distance:.2f}, {self.current_lateral:.2f}, "
                            f"{math.degrees(self.current_bearing):.1f}°), "
                            f"predicted=({self.filtered_distance:.2f}, {self.filtered_lateral:.2f}, "
                            f"{math.degrees(self.filtered_bearing):.1f}°)"
                        )
                else:
                    # Fall back to filtered values
                    self.filtered_distance = filtered_position[0]
                    self.filtered_lateral = filtered_position[1]
                    self.filtered_bearing = filtered_position[2]
            else:
                # Just use filtered values for inconsistent movement
                self.filtered_distance = filtered_position[0]
                self.filtered_lateral = filtered_position[1]
                self.filtered_bearing = filtered_position[2]
        else:
            # Fall back to raw values if filtering not available
            self.filtered_distance = self.current_distance
            self.filtered_lateral = self.current_lateral
            self.filtered_bearing = self.current_bearing
    
    def state_callback(self, msg):
        """Handle robot state updates with improved recovery behavior."""
        new_state = msg.data
        
        # If state changed, handle the transition
        if new_state != self.robot_state:
            # Throttled logging for state changes
            self._log_throttled(
                self.get_logger().info,
                f"STATE TRANSITION: {self.robot_state} → {new_state}",
                LOG_THROTTLE_STATE,
                'last_state_log_time'
            )
            
            self.previous_state = self.robot_state
            self.robot_state = new_state
            
            # Handle recovery state transitions
            if new_state == "recovery":
                self.in_recovery = True
                self.recovery_start_time = time.time()
                self.recovery_phase = "stop"
                # Stop robot immediately when entering recovery
                self.stop_robot()
                self.get_logger().info("Entering recovery mode - stopping robot")
            
            # Complete controller reset when transitioning between tracking and other states
            if new_state == "tracking" or self.previous_state == "tracking":
                self._complete_controller_reset()
                
                # Force target reacquisition when re-entering tracking mode
                if new_state == "tracking":
                    self.force_target_reacquisition = True
                
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
            
        # Log resource stats occasionally
        if self.cycle_count % 10 == 0:
            self.resource_monitor.log_stats()
    
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
                self.performance_stats["control_skips"] += 1
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
            
            # Special handling for recovery mode
            if self.in_recovery:
                self._handle_recovery_mode(current_time)
                return
            
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
            
            # Use filtered values
            distance = self.filtered_distance
            lateral = self.filtered_lateral
            bearing = self.filtered_bearing
            
            angular_degrees = math.degrees(bearing)
            
            if self.debug_level >= 2:
                self.get_logger().info(
                    f"PRE-STOP CHECK: distance={distance:.3f}m (target={self.desired_distance:.3f}m), "
                    f"lateral={lateral:.3f}m, angular={angular_degrees:.2f}°, "
                    f"is_stopped={self._robot_stopped}"
                )

            # Calculate errors using pre-allocated array
            self._current_errors[0] = distance - self.desired_distance  # distance_error
            self._current_errors[1] = lateral - 0.0                     # lateral_error 
            self._current_errors[2] = bearing                           # angular_error
            
            # Check if we need to reset stopped state based on errors with enhanced hysteresis
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
            
            # Determine the optimal movement strategy with hysteresis
            strategy = self._determine_movement_strategy(
                self._current_errors[0], 
                self._current_errors[1], 
                angular_degrees,
                self.prev_distance_category,
                self.prev_lateral_category,
                self.prev_angular_category
            )
            
            # Apply strategy to movement decisions
            use_forward = strategy["use_forward"]
            use_lateral = strategy["use_lateral"]
            use_angular = strategy["use_angular"]
            
            forward_scale = strategy["forward_scale"]
            lateral_scale = strategy["lateral_scale"]
            angular_scale = strategy["angular_scale"]
            
            if self.debug_level >= 3:
                self.get_logger().info(
                    f"Using strategy: {strategy['strategy_name']}, "
                    f"forward={use_forward}, lateral={use_lateral}, angular={use_angular}"
                )

            # Compute velocities
            if self.coordinated_movement and use_lateral and use_angular:
                # Use coordinated controller for lateral and angular movements
                linear_x_velocity = self.pid_linear_x.compute(
                    self._current_errors[0], 
                    current_time, 
                    not use_forward,
                    self.distance_error_tracker.get_trend()
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
                    not use_forward,
                    self.distance_error_tracker.get_trend()
                )
                
                lateral_velocity = self.pid_linear_y.compute(
                    self._current_errors[1], 
                    current_time, 
                    not use_lateral,
                    self.lateral_error_tracker.get_trend()
                )
                
                angular_velocity = self.pid_angular.compute(
                    self._current_errors[2], 
                    current_time, 
                    not use_angular,
                    self.angular_error_tracker.get_trend()
                )
            
            # Apply strategy scaling factors
            linear_x_velocity *= forward_scale
            lateral_velocity *= lateral_scale
            angular_velocity *= angular_scale
            
            if self.debug_level >= 2:
                self.get_logger().info(
                    f"After scaling: linear_x={linear_x_velocity:.3f}, "
                    f"lateral={lateral_velocity:.3f}, angular={angular_velocity:.3f}"
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
    
    def _handle_recovery_mode(self, current_time):
        """
        Handle recovery mode with a three-phase approach:
        1. Stop - halt all movement
        2. Orient - align with target
        3. Approach - move to target
        """
        # Check if we should exit recovery mode
        if self.robot_state != "recovery":
            self.in_recovery = False
            self.recovery_phase = "none"
            return
        
        recovery_duration = current_time - self.recovery_start_time
        
        # Phase 1: Stop (0-1 seconds)
        if self.recovery_phase == "stop":
            # Ensure robot is stopped
            self.stop_robot()
            
            # After 1 second, transition to orient phase
            if recovery_duration > 1.0:
                self.recovery_phase = "orient"
                self.get_logger().info("Recovery: Moving to orientation phase")
                
                # Reset angular controller
                self.pid_angular.reset()
                self.angular_error_tracker.reset()
                
        # Phase 2: Orient (1-3 seconds)
        elif self.recovery_phase == "orient":
            # Only orient if we have a target
            if self.current_target is not None and self._is_orientation_fresh():
                # Use filtered bearing
                bearing = self.filtered_bearing
                angular_degrees = math.degrees(bearing)
                
                # Only orient if angular error is significant
                if abs(angular_degrees) > 2.0:
                    # Compute angular velocity with PID
                    angular_velocity = self.pid_angular.compute(
                        bearing,
                        current_time,
                        force_zero=False
                    )
                    
                    # Apply conservative scaling
                    angular_velocity *= 0.8
                    
                    # Create and publish Twist message
                    cmd_vel_msg = self._cmd_vel_msg
                    cmd_vel_msg.linear.x = 0.0
                    cmd_vel_msg.linear.y = 0.0
                    cmd_vel_msg.angular.z = float(angular_velocity)
                    
                    self.cmd_vel_pub.publish(cmd_vel_msg)
                    
                    self.get_logger().info(f"Recovery orient: angular_error={angular_degrees:.2f}°, velocity={angular_velocity:.2f}")
                else:
                    # If angular error is small, stop rotation
                    self.stop_robot()
                    
                    # Log alignment success
                    self.get_logger().info(f"Recovery orient: good alignment achieved ({angular_degrees:.2f}°)")
            
            # After 2 seconds in orient phase, move to approach
            if recovery_duration > 3.0:
                self.recovery_phase = "approach"
                self.get_logger().info("Recovery: Moving to approach phase")
                
                # Reset all controllers for approach
                self.pid_linear_x.reset()
                self.pid_linear_y.reset()
                self.distance_error_tracker.reset()
                self.lateral_error_tracker.reset()
                
        # Phase 3: Approach (3+ seconds)
        elif self.recovery_phase == "approach":
            # Only approach if we have a target
            if self.current_target is not None and self._is_orientation_fresh():
                # Use filtered values
                distance = self.filtered_distance
                lateral = self.filtered_lateral
                
                # Calculate errors
                distance_error = distance - self.desired_distance
                lateral_error = lateral
                
                # Only move if errors are significant
                if abs(distance_error) > 0.1 or abs(lateral_error) > 0.1:
                    # Compute velocities
                    linear_x_velocity = self.pid_linear_x.compute(
                        distance_error,
                        current_time,
                        force_zero=False
                    ) * 0.7  # Apply conservative scaling
                    
                    lateral_velocity = self.pid_linear_y.compute(
                        lateral_error,
                        current_time,
                        force_zero=False
                    ) * 0.7  # Apply conservative scaling
                    
                    # Create and publish Twist message
                    cmd_vel_msg = self._cmd_vel_msg
                    cmd_vel_msg.linear.x = float(linear_x_velocity)
                    cmd_vel_msg.linear.y = float(lateral_velocity)
                    cmd_vel_msg.angular.z = 0.0
                    
                    self.cmd_vel_pub.publish(cmd_vel_msg)
                    
                    self.get_logger().info(
                        f"Recovery approach: distance_error={distance_error:.2f}m, "
                        f"lateral_error={lateral_error:.2f}m, "
                        f"velocity=({linear_x_velocity:.2f}, {lateral_velocity:.2f})"
                    )
                else:
                    # If errors are small, stop movement
                    self.stop_robot()
                    
                    # Log approach success
                    self.get_logger().info(
                        f"Recovery approach: good position achieved "
                        f"(distance_error={distance_error:.2f}m, lateral_error={lateral_error:.2f}m)"
                    )
            
            # After 6 seconds in recovery, suggest exiting recovery mode
            if recovery_duration > 6.0:
                self.get_logger().info(
                    "Recovery has been active for 6 seconds. "
                    "Consider transitioning back to tracking mode if recovery is complete."
                )
    
    def _reset_stopped_state_if_needed(self, distance_error, lateral_error, angular_error):
        """
        Reset stopped state if significant movement is required, with improved hysteresis.
        
        Args:
            distance_error: Current error in distance to target
            lateral_error: Current error in lateral position
            angular_error: Current error in angular position (degrees)
            
        Returns:
            bool: True if stopped state was reset, False otherwise
        """
        if not self._robot_stopped:
            return False  # Already in movement state
            
        # Calculate hysteresis factor based on stop time
        stop_duration = time.time() - self._stop_time
        
        # Hysteresis increases with stop duration to a max of 1.5
        # This helps prevent oscillating between stopped and moving states
        hysteresis = min(1.5, 1.0 + stop_duration * 0.2)
        
        # If any error exceeds the movement threshold with hysteresis, exit stopped state
        distance_threshold = self.distance_threshold * hysteresis
        lateral_threshold = self.lateral_threshold * hysteresis
        angular_threshold = self.angular_threshold * hysteresis
        
        if (abs(distance_error) > distance_threshold or
            abs(lateral_error) > lateral_threshold or
            abs(angular_error) > angular_threshold):
            
            self.get_logger().info(
                f"Exiting stopped state - Movement required: "
                f"distance_error={distance_error:.3f}m(threshold={distance_threshold:.3f}), "
                f"lateral_error={lateral_error:.3f}m(threshold={lateral_threshold:.3f}), "
                f"angular_error={angular_error:.2f}°(threshold={angular_threshold:.2f})"
            )
            
            # Reset stopped state
            self._robot_stopped = False
            
            # Reset movement hysteresis
            self._movement_hysteresis = 0.0
            
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
            tuple: (should_stop, reason) - True if robot should stop, False if it should move
        """
        # Calculate error values
        distance_error = abs(distance - self.desired_distance)
        lateral_error = abs(lateral)
        angular_error = abs(angular_degrees)
        
        # Start with base thresholds
        distance_threshold = self.distance_threshold
        lateral_threshold = self.lateral_threshold
        angular_threshold = self.angular_threshold
        
        # Apply hysteresis - different thresholds based on current state
        if is_stopped:
            # If already stopped, use higher thresholds to start moving
            # (requires larger errors to start moving)
            hysteresis = 1.5 + self._movement_hysteresis  # Additional accumulated hysteresis
            distance_threshold *= hysteresis
            lateral_threshold *= hysteresis
            angular_threshold *= hysteresis
            
            # Cap the maximum thresholds
            distance_threshold = min(distance_threshold, 0.2)
            lateral_threshold = min(lateral_threshold, 0.15)
            angular_threshold = min(angular_threshold, 6.0)
        else:
            # If already moving, use reduced thresholds to stop
            # (more precision when already near target)
            hysteresis = 0.8
            distance_threshold *= hysteresis
            lateral_threshold *= hysteresis
            angular_threshold *= hysteresis
        
        if self.debug_level >= 2:
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
        
        # If we're stopping, accumulate a small amount of hysteresis
        # This helps prevent oscillating when hovering near thresholds
        if not is_stopped:
            self._movement_hysteresis += 0.05
            self._movement_hysteresis = min(0.3, self._movement_hysteresis)  # Cap at 0.3
        
        return True, reason  # Return True to indicate robot SHOULD stop
    
    def _apply_velocity_limits(self, linear_x, linear_y, angular_z, current_time):
        """
        Apply velocity and acceleration limits for smooth, natural movement.
        With enhanced handling for combined movements and hysteresis.
        
        Args:
            linear_x: Calculated forward velocity
            linear_y: Calculated lateral velocity
            angular_z: Calculated angular velocity
            current_time: Current time for acceleration limiting
            
        Returns:
            tuple: (limited_linear_x, limited_linear_y, limited_angular_z)
        """
        if self.debug_level >= 2:  # Lowered from 3 for consistent debug level
            self.get_logger().info(
                f"Before limits: linear_x={linear_x:.3f}, linear_y={linear_y:.3f}, angular_z={angular_z:.3f}"
            )

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
        
        # Apply minimum velocity thresholds with hysteresis
        min_effective_velocity = 0.025  # Reduced from 0.05
        min_angular_velocity = 0.05     # Reduced from 0.1
        
        # Forward velocity threshold with hysteresis
        if abs(self._limited_velocities[0]) < min_effective_velocity:
            # Only zero if previously zero or very small
            if abs(self._prev_velocities[0]) < min_effective_velocity * 1.2:
                self._limited_velocities[0] = 0.0
            
        # Lateral velocity threshold with hysteresis
        if abs(self._limited_velocities[1]) < min_effective_velocity:
            # Only zero if previously zero or very small
            if abs(self._prev_velocities[1]) < min_effective_velocity * 1.2:
                self._limited_velocities[1] = 0.0
            
        # Angular velocity threshold with hysteresis
        if abs(self._limited_velocities[2]) < min_angular_velocity:
            # Only zero if previously zero or very small
            if abs(self._prev_velocities[2]) < min_angular_velocity * 1.2:
                self._limited_velocities[2] = 0.0
        
        # Limit combined lateral and angular movement
        if abs(self._limited_velocities[1]) > 0.15 and abs(self._limited_velocities[2]) > 0.3:
            # Scale down lateral velocity when combined with significant angular velocity
            # This prevents tipping during combined movements
            self._limited_velocities[1] *= 0.6
        
        # Apply maximum velocity limits
        linear_x_max = 0.2  # Example max_velocity
        linear_y_max = 0.2  # Example max_velocity
        angular_max = 0.5   # Example max_angular_velocity
        
        self._limited_velocities[0] = max(-linear_x_max, min(linear_x_max, self._limited_velocities[0]))
        self._limited_velocities[1] = max(-linear_y_max, min(linear_y_max, self._limited_velocities[1]))
        self._limited_velocities[2] = max(-angular_max, min(angular_max, self._limited_velocities[2]))
        
        if self.debug_level >= 2:  # Consistent debug level
            self.get_logger().info(
                f"After limits: linear_x={self._limited_velocities[0]:.3f}, "
                f"linear_y={self._limited_velocities[1]:.3f}, "
                f"angular_z={self._limited_velocities[2]:.3f}"
            )

        return (self._limited_velocities[0], self._limited_velocities[1], self._limited_velocities[2])
    
    def stop_robot(self):
        """Send a command to stop all robot motion immediately and reset controllers."""
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
        
        # Set a "stopped" state flag and timestamp
        self._robot_stopped = True
        self._stop_time = time.time()
        
        # Remember our position when we stopped
        self._last_stop_position = (
            self.current_distance,
            self.current_lateral, 
            self.current_bearing
        )
        
        self.get_logger().info("Robot stopped! All velocities reset.")
    
    def _complete_controller_reset(self):
        """Complete reset of all controllers and error states."""
        # Reset all PID controllers
        self.pid_linear_x.reset()
        self.pid_linear_y.reset()
        self.pid_angular.reset()
        
        # Reset coordinated controller
        self.coordinated_controller.reset()
        
        # Reset all error trackers
        self.distance_error_tracker.reset()
        self.lateral_error_tracker.reset()
        self.angular_error_tracker.reset()
        
        # Reset target filter
        self.target_filter.reset()
        
        # Reset strategy blender
        self.strategy_blender.reset()
        
        # Reset error categorization state
        self.prev_distance_category = "none"
        self.prev_lateral_category = "none"
        self.prev_angular_category = "none"
        
        # Reset motion state
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        self.velocity_history = LightweightBuffer(max_size=6)
        
        # Reset last logged command
        self.last_logged_cmd = (0.0, 0.0, 0.0)
        
        # Reset movement hysteresis
        self._movement_hysteresis = 0.0
        
        # Set stopped state
        self._robot_stopped = True
        
        self.get_logger().info("Complete controller reset performed")
    
    def _init_strategy_table(self):
        """Initialize the table-driven movement strategy definitions with angular-first approach."""
        # Define the strategy table with improved angular-first strategies
        self.strategy_table = {
            # All errors within deadbands - no movement
            ("none", "none", "none"): [
                "NO_MOVEMENT", False, False, False, 
                0.0, 0.0, 0.0, 
                "All errors within deadbands"
            ],
            
            # Very small errors - minimal corrections
            ("very_small", "very_small", "very_small"): [
                "MINIMAL_CORRECTION", True, True, True, 
                0.4, 0.4, 0.4, 
                "Minimal corrections for very small errors"
            ],
            
            # Angular error categories - prioritize angular correction
            ("*", "*", "very_large"): [
                "ANGULAR_ONLY", False, False, True, 
                0.0, 0.0, 1.0, 
                "Angular error correction only: {angular_error:.1f}°"
            ],
            
            ("*", "*", "large"): [
                "ANGULAR_PRIMARY", False, False, True, 
                0.0, 0.0, 1.0, 
                "Angular correction prioritized: {angular_error:.1f}°"
            ],
            
            ("*", "*", "medium_large"): [
                "ANGULAR_PRIMARY_BALANCED", False, True, True, 
                0.0, 0.2, 0.9, 
                "Primarily angular correction with some lateral: {angular_error:.1f}°"
            ],
            
            ("*", "*", "medium"): [
                "ANGULAR_BALANCED", False, True, True, 
                0.0, 0.4, 0.8, 
                "Angular-balanced movement: {angular_error:.1f}°"
            ],
            
            ("*", "*", "small_medium"): [
                "ANGULAR_THEN_LATERAL", True, True, True, 
                0.3, 0.6, 0.7, 
                "Angular-then-lateral transition: {angular_error:.1f}°"
            ],
            
            ("*", "*", "small"): [
                "BALANCED", True, True, True, 
                0.6, 0.7, 0.5, 
                "Balanced movement with small angular correction: {angular_error:.1f}°"
            ],
            
            ("*", "*", "very_small"): [
                "COMBINED_MOVEMENT", True, True, True, 
                0.8, 0.8, 0.3, 
                "Combined movement with minimal angular error: {angular_error:.1f}°"
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
            
            # Combined distance and lateral errors (only when angular error is very small)
            ("*", "*", "none"): [
                "POSITION_ONLY", True, True, False,
                0.9, 0.9, 0.0,
                "Position correction without rotation"
            ],
            
            # Special case for diagonal movement - gradual transition
            ("medium", "medium", "small"): [
                "DIAGONAL_MOVEMENT", True, True, True,
                0.8, 0.8, 0.4,
                "Diagonal movement with small angular correction"
            ],
            
            # Fallback strategy
            ("*", "*", "*"): [
                "BALANCED", True, True, True, 
                0.6, 0.6, 0.6, 
                "Balanced movement strategy (fallback)"
            ]
        }
    
    def _categorize_error(self, error, error_type="distance", prev_category=None):
        """
        Categorize an error value with hysteresis to prevent oscillation.
        
        Args:
            error: The error value to categorize
            error_type: The type of error (distance, lateral, angular)
            prev_category: Previous category for hysteresis
            
        Returns:
            String: The error category
        """
        abs_error = abs(error)
        
        # Select appropriate thresholds based on error type
        if error_type == "angular":
            deadband = 1.5  # Degrees (reduced from 3.0)
            very_small_threshold = deadband
            small_threshold = deadband * 2.0
            small_medium_threshold = deadband * 3.0
            medium_threshold = deadband * 4.0
            medium_large_threshold = deadband * 6.0
            large_threshold = deadband * 8.0
            very_large_threshold = deadband * 12.0
        elif error_type == "lateral":
            deadband = 0.075  # Meters (increased from 0.05)
            very_small_threshold = deadband
            small_threshold = deadband * 2.0
            small_medium_threshold = deadband * 3.0
            medium_threshold = deadband * 4.0
            medium_large_threshold = deadband * 6.0
            large_threshold = deadband * 8.0
            very_large_threshold = deadband * 10.0
        else:  # distance
            deadband = 0.1  # Meters
            very_small_threshold = deadband
            small_threshold = deadband * 2.0
            small_medium_threshold = deadband * 3.0
            medium_threshold = deadband * 4.0
            medium_large_threshold = deadband * 6.0
            large_threshold = deadband * 8.0
            very_large_threshold = deadband * 10.0
        
        # Apply hysteresis if previous category is provided
        if prev_category and prev_category != "none":
            # Apply 20% hysteresis factor
            hysteresis_down = 0.2  # For moving down a category
            hysteresis_up = 0.1    # For moving up a category (easier to go up than down)
            
            # Going down from a higher category - apply stronger hysteresis
            if prev_category == "very_large" and abs_error > very_large_threshold * (1.0 - hysteresis_down):
                return "very_large"
            elif prev_category == "large":
                if abs_error > large_threshold * (1.0 - hysteresis_down):
                    return "large"
                # Moving up requires less hysteresis
                if abs_error > very_large_threshold * (1.0 - hysteresis_up):
                    return "very_large"
                
            elif prev_category == "medium_large":
                if abs_error > medium_large_threshold * (1.0 - hysteresis_down):
                    return "medium_large"
                # Moving up requires less hysteresis
                if abs_error > large_threshold * (1.0 - hysteresis_up):
                    return "large"
                
            elif prev_category == "medium":
                if abs_error > medium_threshold * (1.0 - hysteresis_down):
                    return "medium"
                # Moving up requires less hysteresis
                if abs_error > medium_large_threshold * (1.0 - hysteresis_up):
                    return "medium_large"
                    
            elif prev_category == "small_medium":
                if abs_error > small_medium_threshold * (1.0 - hysteresis_down):
                    return "small_medium"
                # Moving up requires less hysteresis
                if abs_error > medium_threshold * (1.0 - hysteresis_up):
                    return "medium"
                
            elif prev_category == "small":
                if abs_error > small_threshold * (1.0 - hysteresis_down):
                    return "small"
                # Moving up requires less hysteresis
                if abs_error > small_medium_threshold * (1.0 - hysteresis_up):
                    return "small_medium"
                
            elif prev_category == "very_small":
                if abs_error > very_small_threshold * (1.0 - hysteresis_down):
                    return "very_small"
                # Moving up requires less hysteresis
                if abs_error > small_threshold * (1.0 - hysteresis_up):
                    return "small"
        
        # Handle lateral control toggle
        if error_type == "lateral" and not getattr(self, 'use_lateral_control', True):
            return "none"
        
        # Categorize based on absolute error value
        if abs_error <= deadband:
            return "none"
        elif abs_error <= very_small_threshold:
            return "very_small"
        elif abs_error <= small_threshold:
            return "small"
        elif abs_error <= small_medium_threshold:
            return "small_medium"
        elif abs_error <= medium_threshold:
            return "medium"
        elif abs_error <= medium_large_threshold:
            return "medium_large"
        elif abs_error <= large_threshold:
            return "large"
        else:
            return "very_large"
    
    def _determine_movement_strategy(self, distance_error, lateral_error, angular_error_degrees,
                                     prev_distance_category=None, prev_lateral_category=None, 
                                     prev_angular_category=None):
        """
        Determine the optimal movement strategy using table-driven approach
        with hysteresis and angular-first prioritization.
        
        Args:
            distance_error: Error in distance (meters)
            lateral_error: Error in lateral position (meters)
            angular_error_degrees: Error in angular position (degrees)
            prev_*_category: Previous error categories for hysteresis
            
        Returns:
            dict: Strategy information including strategy name, movement flags, and scale factors
        """
        current_time = time.time()
        
        # Categorize errors into states with hysteresis
        self._key_tuple[0] = self._categorize_error(
            distance_error, "distance", prev_distance_category)
        self._key_tuple[1] = self._categorize_error(
            lateral_error, "lateral", prev_lateral_category)
        self._key_tuple[2] = self._categorize_error(
            angular_error_degrees, "angular", prev_angular_category)
        
        # Save categories for next iteration's hysteresis
        self.prev_distance_category = self._key_tuple[0]
        self.prev_lateral_category = self._key_tuple[1]
        self.prev_angular_category = self._key_tuple[2]
        
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
        
        # Log strategy changes (throttled)
        if blend_started or self.debug_level >= 2:
            if self._log_throttled(
                self.get_logger().info,
                f"Strategy selected: {current_strategy.name}, params: "
                f"forward={current_strategy.forward_scale:.1f}, "
                f"lateral={current_strategy.lateral_scale:.1f}, "
                f"angular={current_strategy.angular_scale:.1f}",
                1.0,  # Throttle to once per second
                'last_strategy_log_time'
            ):
                # Log detailed error info with strategy change
                self.get_logger().info(
                    f"Error categories: distance={self._key_tuple[0]}, "
                    f"lateral={self._key_tuple[1]}, angular={self._key_tuple[2]}"
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
            # Get PID components
            p_x, i_x, d_x = self.pid_linear_x.get_components()
            p_y, i_y, d_y = self.pid_linear_y.get_components()
            p_a, i_a, d_a = self.pid_angular.get_components()
            
            diag_msg = (
                f"DIAGNOSTICS: "
                f"Avg Vel=[{avg_lin_x_vel:.2f}, {avg_lin_y_vel:.2f}, {avg_ang_vel:.2f}], "
                f"PID X=[{p_x:.2f}, {i_x:.2f}, {d_x:.2f}], "
                f"PID Y=[{p_y:.2f}, {i_y:.2f}, {d_y:.2f}], "
                f"PID A=[{p_a:.2f}, {i_a:.2f}, {d_a:.2f}], "
                f"Strategy={self.current_strategy}"
            )
            
            self.get_logger().info(diag_msg)
        
        # Publish PID diagnostic data
        self._publish_pid_diagnostics()
        
        # Publish performance metrics
        self._publish_performance_metrics()
    
    def _publish_pid_diagnostics(self):
        """Publish detailed PID diagnostics for analysis."""
        # Get PID components
        p_x, i_x, d_x = self.pid_linear_x.get_components()
        p_y, i_y, d_y = self.pid_linear_y.get_components()
        p_a, i_a, d_a = self.pid_angular.get_components()
        
        # Get current gains
        kp_x, ki_x, kd_x = self.pid_linear_x.get_current_gains()
        kp_y, ki_y, kd_y = self.pid_linear_y.get_current_gains()
        kp_a, ki_a, kd_a = self.pid_angular.get_current_gains()
        
        # Get current errors
        e_x = self.distance_error_tracker.current_error
        e_y = self.lateral_error_tracker.current_error
        e_a = self.angular_error_tracker.current_error
        
        # Pack all data into the array - no unnecessary float() conversions
        self._diag_data[0] = p_x
        self._diag_data[1] = i_x
        self._diag_data[2] = d_x
        self._diag_data[3] = p_y
        self._diag_data[4] = i_y
        self._diag_data[5] = d_y
        self._diag_data[6] = p_a
        self._diag_data[7] = i_a
        self._diag_data[8] = d_a
        self._diag_data[9] = e_x
        self._diag_data[10] = e_y
        self._diag_data[11] = e_a
        self._diag_data[12] = kp_a  # Track angular P gain
        self._diag_data[13] = kd_a  # Track angular D gain
        
        # Update Float32MultiArray data
        self._diag_msg.data = self._diag_data.tolist()
        
        # Publish diagnostics
        self.pid_diag_pub.publish(self._diag_msg)
    
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
            self.get_logger().info("Robot motion stopped during shutdown")
        except Exception as e:
            self.get_logger().error(f"Error stopping robot during shutdown: {str(e)}")

# Main function
def main(args=None):
    """Main function to initialize and run the PID Controller node."""
    rclpy.init(args=args)
    node = ImprovedPIDControllerNode()
    
    # Welcome message
    print("=================================================")
    print("Improved PID Controller for Basketball Tracking Robot")
    print("=================================================")
    print("This node implements several enhancements:")
    print("1. Angular-first control strategy for diagonal movements")
    print("2. Enhanced integral term management")
    print("3. Fast strategy transitions for responsive tracking")
    print("4. Balanced error thresholds with hysteresis")
    print("5. Continuous motion tracking with trajectory prediction")
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
        node.get_logger().info("Improved PID Controller shutdown requested via Ctrl+C")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {str(e)}")
    finally:
        # Stop the robot before shutdown
        node.prepare_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()