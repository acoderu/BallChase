"""
Basketball Tracking Robot - Optimized PID Controller Modules
===========================================================

This file contains refactored and performance-optimized modules for the PID controller.
Optimizations focus on:
- Matching control rate to actual fusion data rate (1-5Hz)
- Reducing CPU and memory usage on Raspberry Pi 5
- Enhanced data freshness detection
- Event-driven control capability
- Simplified resource management
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped, Vector3, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray
from tf2_ros import TransformListener, Buffer
import math
import time
import numpy as np
import psutil
import logging
from collections import deque
from enum import Enum, auto

# Import modules from refactored files
from pid_helpers import Matrix4x4, TTLDict, LightweightBuffer, ResourceMonitor
from pid_target_filter import EnhancedTargetFilter, ErrorTracker
from pid_computation import PIDControllers

#############################################
# Target Tracking Module
#############################################

class TargetTrackingModule:
    """Module that handles target tracking, filtering, and prediction with fusion rate detection."""
    
    def __init__(self, throttled_logger, filter_buffer_size=5, prediction_horizon=0.2, debug_level=0):
        """Initialize the target tracking module."""
        self.logger = throttled_logger
        self.debug_level = debug_level
        self.current_target = None
        self.last_target_time = None
        self.target_frame = "unknown_frame"
        
        # Pre-allocate arrays for metrics to reduce memory allocations
        # Current raw metrics
        self.current_metrics = np.zeros(3, dtype=np.float32)  # [distance, lateral, bearing]
        # Filtered metrics
        self.filtered_metrics = np.zeros(3, dtype=np.float32)  # [distance, lateral, bearing]
        
        # Initialize target filter
        try:
            self.target_filter = EnhancedTargetFilter(
                self.logger,
                buffer_size=filter_buffer_size,
                prediction_horizon=prediction_horizon,
                debug_level=self.debug_level
            )
            self.filter_initialized = True
        except Exception as e:
            self.logger.error(f"Failed to initialize target filter: {str(e)}")
            self.filter_initialized = False
        
        # Recovery flags
        self.force_target_reacquisition = False
        
        # Add timestamp tracking for fusion rate calculation
        self.update_timestamps = deque(maxlen=10)  # Store last 10 update times
        self.last_fusion_rate = 1.0  # Default assumption (1Hz)
        self.fusion_rate_updated = False
        self.last_rate_calculation = 0.0
    
    def update_target(self, target_msg, debug_level=None):
        """
        Process a new target message and update position data.
        
        Returns:
            bool: True if data was updated, False otherwise
        """
        if debug_level is None:
            debug_level = self.debug_level
        if target_msg is None:
            return False
            
        try:
            # Store current time once for efficiency
            current_time = time.time()
            
            # Safety check for target point
            if not hasattr(target_msg, 'point'):
                self.logger.warning("Target message has no point attribute")
                return False
                
            # Update timestamps for fusion rate detection
            self.update_timestamps.append(current_time)
            
            # Only calculate fusion rate periodically to save CPU
            if current_time - self.last_rate_calculation > 2.0 and len(self.update_timestamps) >= 3:
                self._calculate_fusion_rate(debug_level)
                self.last_rate_calculation = current_time
            
            # Update target data and timestamps
            self.last_target_time = current_time
            self.current_target = target_msg
            
            # Store target frame for debugging
            self.target_frame = target_msg.header.frame_id if hasattr(target_msg.header, 'frame_id') else "base_link"
            
            # Calculate raw target metrics
            self._calculate_raw_target_metrics(target_msg.point, self.target_frame)
            
            # Apply target filtering and prediction
            self._apply_target_filtering()
            
            # Log target update if debugging enabled
            if debug_level >= 2:
                now = time.time()
                if not hasattr(self, '_last_logged_target_update_info') or now - self._last_logged_target_update_info > 1.0:
                    self.logger.info(
                        f"Target update: frame={self.target_frame}, pos=({self.current_metrics[0]:.3f}, {self.current_metrics[1]:.3f}, {self.current_metrics[2]:.3f})"
                    )
                    self._last_logged_target_update_info = now
                
            return True
        except Exception as e:
            self.logger.error(f"Error updating target: {str(e)}")
            return False
    
    def _calculate_fusion_rate(self, debug_level=None):
        """Calculate the actual fusion data rate based on timestamps."""
        if debug_level is None:
            debug_level = self.debug_level
        try:
            # Safety check - need at least 2 timestamps
            if len(self.update_timestamps) < 2:
                return
                
            # Calculate time intervals between updates
            intervals = [self.update_timestamps[i] - self.update_timestamps[i-1] 
                        for i in range(1, len(self.update_timestamps))]
            
            # Skip outliers (very long gaps)
            valid_intervals = [dt for dt in intervals if dt < 2.0]
            
            if not valid_intervals:
                return
                
            # Calculate average interval
            avg_interval = sum(valid_intervals) / len(valid_intervals)
            new_rate = 1.0 / max(0.1, avg_interval)  # Avoid division by zero
            
            # Ensure rate is within reasonable bounds (0.5Hz to 10Hz)
            new_rate = max(0.5, min(10.0, new_rate))
            
            # Only update if significantly different (save unnecessary updates)
            if abs(new_rate - self.last_fusion_rate) > 0.2:
                self.last_fusion_rate = new_rate
                self.fusion_rate_updated = True
                if debug_level >= 1:
                    self.logger.info(f"Detected fusion data rate: {new_rate:.2f} Hz", throttle_duration_sec=2.0)
        except Exception as e:
            self.logger.error(f"Error calculating fusion rate: {str(e)}")
    
    def _calculate_raw_target_metrics(self, target_point, frame_id):
        """Calculate raw distance, bearing, and lateral offset to target."""
        if target_point is None:
            return
            
        try:
            # Calculate full 2D distance to target (index 0)
            self.current_metrics[0] = math.sqrt(target_point.x**2 + target_point.y**2)
            
            # Calculate bearing and lateral position based on frame
            if frame_id in ["camera_frame", "camera_optical_frame"]:
                # Camera optical frame: Z forward, X right, Y down
                self.current_metrics[2] = math.atan2(target_point.x, target_point.z)  # Bearing (index 2)
                self.current_metrics[1] = target_point.x  # Lateral (index 1)
            else:
                # Standard robot frame: X forward, Y left
                self.current_metrics[2] = math.atan2(target_point.y, target_point.x)  # Bearing (index 2)
                self.current_metrics[1] = target_point.y  # Lateral (index 1)
        except Exception as e:
            self.logger.error(f"Error calculating target metrics: {str(e)}")
    
    def _apply_target_filtering(self):
        """Apply filtering and prediction to target position."""
        if not self.filter_initialized:
            # Copy current values directly to filtered values
            np.copyto(self.filtered_metrics, self.current_metrics)
            return
            
        try:
            # Update the filter with new position data
            # Convert NumPy array to tuple for filter update
            current_position = (self.current_metrics[0], self.current_metrics[1], self.current_metrics[2])
            filtered_position = self.target_filter.update(current_position, self.last_target_time)
            
            # Handle forced target reacquisition (e.g., after recovery)
            if self.force_target_reacquisition:
                self._handle_target_reacquisition()
                return
            
            # Use filtered/predicted position values
            self._select_position_values(filtered_position)
            
            if self.filter_initialized and self.debug_level >= 2:
                now = time.time()
                if not hasattr(self, '_last_logged_applied_filter') or now - self._last_logged_applied_filter > 1.0:
                    self.logger.info(f"Applied target filtering: filtered={self.filtered_metrics}")
                    self._last_logged_applied_filter = now
                
        except Exception as e:
            self.logger.error(f"Error applying target filtering: {str(e)}")
            # Fall back to raw values on error
            np.copyto(self.filtered_metrics, self.current_metrics)
    
    def _handle_target_reacquisition(self):
        """Handle forced target reacquisition after recovery."""
        try:
            self.target_filter.reset()
            # Convert NumPy array to tuple for filter update
            current_position = (self.current_metrics[0], self.current_metrics[1], self.current_metrics[2])
            filtered_position = self.target_filter.update(current_position, self.last_target_time)
            self.force_target_reacquisition = False
            self.logger.info("Forced target reacquisition - filter reset", throttle_duration_sec=2.0)
            
            # Use filtered values (not prediction during reacquisition)
            if filtered_position:
                # Copy filtered values directly to the pre-allocated array
                self.filtered_metrics[0] = filtered_position[0]
                self.filtered_metrics[1] = filtered_position[1]
                self.filtered_metrics[2] = filtered_position[2]
            else:
                # Fall back to raw values if filter has issues
                np.copyto(self.filtered_metrics, self.current_metrics)
        except Exception as e:
            self.logger.error(f"Error during target reacquisition: {str(e)}")
            # Fall back to raw values on error
            np.copyto(self.filtered_metrics, self.current_metrics)
    
    def _select_position_values(self, filtered_position):
        """Select whether to use raw, filtered, or predicted position values."""
        if not filtered_position:
            # Fall back to raw values if filtering unavailable
            np.copyto(self.filtered_metrics, self.current_metrics)
            return
        
        try:
            # Check target movement characteristics
            movement_info = self.target_filter.get_movement_info()
            
            # Validate movement info has expected keys
            is_moving = movement_info.get('is_moving', False)
            direction_change = movement_info.get('direction_change', True)
            
            # For consistent movement, use prediction
            if is_moving and not direction_change:
                predicted_position = self.target_filter.get_predicted_position()
                if predicted_position and len(predicted_position) >= 3:
                    # Use prediction for control
                    self.filtered_metrics[0] = predicted_position[0]
                    self.filtered_metrics[1] = predicted_position[1]
                    self.filtered_metrics[2] = predicted_position[2]
                else:
                    # Fall back to filtered values if prediction fails
                    self.filtered_metrics[0] = filtered_position[0]
                    self.filtered_metrics[1] = filtered_position[1]
                    self.filtered_metrics[2] = filtered_position[2]
            else:
                # Just use filtered values for inconsistent movement
                self.filtered_metrics[0] = filtered_position[0]
                self.filtered_metrics[1] = filtered_position[1]
                self.filtered_metrics[2] = filtered_position[2]
                
            if self.debug_level >= 2:
                now = time.time()
                if not hasattr(self, '_last_logged_selected_position') or now - self._last_logged_selected_position > 1.0:
                    self.logger.info(f"Selected position values: {self.filtered_metrics}")
                    self._last_logged_selected_position = now
                
        except Exception as e:
            self.logger.error(f"Error selecting position values: {str(e)}")
            # Fall back to filtered values on error
            self.filtered_metrics[0] = filtered_position[0] if len(filtered_position) > 0 else self.current_metrics[0]
            self.filtered_metrics[1] = filtered_position[1] if len(filtered_position) > 1 else self.current_metrics[1]
            self.filtered_metrics[2] = filtered_position[2] if len(filtered_position) > 2 else self.current_metrics[2]
    
    def _log_target_update(self, msg):
        """Log target update information."""
        if msg is None or not hasattr(msg, 'point'):
            return
            
        target = msg.point
        frame_id = self.target_frame
        
        # Only format the log message if it will actually be logged
        if self.debug_level >= 2:
            now = time.time()
            if not hasattr(self, '_last_logged_target_update') or now - self._last_logged_target_update > 1.0:
                self.logger.info(
                    f"TARGET DATA: frame={frame_id}, pos=({target.x:.3f}, {target.y:.3f}, {target.z:.3f})"
                )
                self._last_logged_target_update = now
        
    def get_position_data(self):
        """Get the current filtered position data."""
        return {
            'distance': self.filtered_metrics[0],
            'lateral': self.filtered_metrics[1],
            'bearing': self.filtered_metrics[2],
            'raw_distance': self.current_metrics[0],
            'raw_lateral': self.current_metrics[1],
            'raw_bearing': self.current_metrics[2]
        }
    
    def is_target_fresh(self, max_age=None):
        """
        Check if the target data is fresh enough to use with graduated freshness levels.
        
        Args:
            max_age: Maximum age in seconds, if None calculated based on fusion rate
                  
        Returns:
            tuple: (is_fresh, freshness_level, age)
                is_fresh: Boolean indicating if data is usable at all
                freshness_level: String indicating freshness level ('fresh', 'stale', 'critical')
                age: Current age of the data in seconds
        """
        if self.last_target_time is None:
            return False, 'critical', float('inf')
            
        current_time = time.time()
        age = current_time - self.last_target_time
        
        # If no max_age provided, calculate based on fusion rate
        if max_age is None:
            # Calculate expected interval between updates
            expected_interval = 1.0 / max(0.5, self.last_fusion_rate)
            
            # Define freshness thresholds based on expected update interval
            fresh_threshold = expected_interval * 1.2    # Data is fully fresh within 1.2x update interval
            stale_threshold = expected_interval * 2.0    # Data is stale but usable up to 2x update interval
            critical_threshold = expected_interval * 3.0  # Data is critically old after 3x update interval
            
            # Determine freshness level
            if age <= fresh_threshold:
                return True, 'fresh', age
            elif age <= stale_threshold:
                return True, 'stale', age
            elif age <= critical_threshold:
                return False, 'critical', age
            else:
                return False, 'invalid', age
        else:
            # Use provided max_age with simple binary fresh/not fresh
            return age < max_age, 'fresh' if age < max_age else 'critical', age
    
    def get_fusion_rate(self):
        """
        Get the detected fusion data rate.
        
        Returns:
            tuple: (rate, was_updated) - Current rate and whether it was just updated
        """
        was_updated = self.fusion_rate_updated
        self.fusion_rate_updated = False  # Reset flag
        return self.last_fusion_rate, was_updated

#############################################
# Movement Strategy Module
#############################################

class MovementStrategyModule:
    """Module that handles movement strategy selection and blending by delegating to StrategyManager."""
    
    def __init__(self, throttled_logger, debug_level=0):
        """Initialize the movement strategy module."""
        self.logger = throttled_logger
        self.debug_level = debug_level
        # Use centralized StrategyManager from PIDControllers
        self.strategy_manager = PIDControllers.StrategyManager(throttled_logger)
        self.strategy_manager.set_debug_level(self.debug_level)
        self.strategy_blender = self.strategy_manager.strategy_blender
        self.initialized = True
        self.current_strategy = "IDLE"
        self.previous_strategy = None
        self.strategy_change_time = time.time()
        self.prev_error_categories = ["none", "none", "none"]  # [distance, lateral, angular]
        self._startup_movement_cycles = 0
        self._fallback_strategy = self.strategy_manager._fallback_strategy

    def determine_strategy(self, distance_error, lateral_error, angular_error_degrees, is_robot_stopped=False):
        """
        Determine the optimal movement strategy based on current errors.
        Delegates to the centralized StrategyManager.
        """
        strategy = self.strategy_manager.determine_strategy(
            distance_error, 
            lateral_error, 
            angular_error_degrees,
            is_robot_stopped
        )
        if self.debug_level >= 2:
            self.logger.info(
                f"Strategy selected: {strategy.strategy_name}, params: forward={strategy.forward_scale:.1f}, lateral={strategy.lateral_scale:.1f}, angular={strategy.angular_scale:.1f}",
                throttle_duration_sec=1.0
            )
        return strategy

    def reset(self):
        """Reset the movement strategy module state."""
        self.current_strategy = "IDLE"
        self.previous_strategy = None
        self.strategy_change_time = time.time()
        self.prev_error_categories = ["none", "none", "none"]
        self._startup_movement_cycles = 0
        if hasattr(self, 'strategy_blender') and self.strategy_blender is not None:
            self.strategy_blender.reset()
        if hasattr(self, 'strategy_manager') and self.strategy_manager is not None:
            self.strategy_manager.current_strategy = "IDLE"
            self.strategy_manager._startup_movement_cycles = 0
        self._fallback_strategy = self.strategy_manager._fallback_strategy

#############################################
# Velocity Control Module
#############################################

class VelocityControlModule:
    """Module to handle velocity generation, limiting, and coordination."""
    
    def __init__(self, throttled_logger):
        """Initialize velocity processor with logger."""
        self.logger = throttled_logger
        
        # Velocity parameters
        self.approach_distance = 0.3
        self.min_approach_factor = 0.2
        self.debug_level = 0
        
        # Previous velocities for acceleration limiting
        self.last_cmd_vel = np.zeros(3, dtype=np.float32)  # [x, y, angular_z]
        self.last_logged_cmd = np.zeros(3, dtype=np.float32)
        self.last_accel_time = time.time()
        
        # Pre-allocated arrays for performance
        self._limited_velocities = np.zeros(3, dtype=np.float32)
        self._target_velocities = np.zeros(3, dtype=np.float32)
        self._vel_diffs = np.zeros(3, dtype=np.float32)
        
        # Pre-allocated velocity change check
        self._velocity_change_check = np.zeros(3, dtype=bool)
        
        # Velocity history buffer - optimized for fixed size
        self.buffer_size = 6
        self.velocity_history = np.zeros((self.buffer_size, 3), dtype=np.float32)
        self.history_index = 0
        self.history_count = 0
        
        # Threshold lookup table
        self.min_linear_velocity = 0.01
        self.min_angular_velocity = 0.01
        
        # Cached velocity limits to avoid repeated calculations
        self.max_velocity_limits = np.array([0.5, 0.4, 0.6], dtype=np.float32)  # [x, y, angular_z]
    
    def set_approach_parameters(self, approach_distance, min_approach_factor):
        """Set parameters for approach behavior."""
        if approach_distance <= 0 or min_approach_factor < 0 or min_approach_factor > 1:
            self.logger.warning(f"Invalid approach parameters: distance={approach_distance}, factor={min_approach_factor}")
            return False
            
        self.approach_distance = approach_distance
        self.min_approach_factor = min_approach_factor
        return True
    
    def set_debug_level(self, debug_level):
        """Set debug level for logging."""
        self.debug_level = max(0, debug_level)
    
    def reset(self):
        """Reset processor state."""
        # Zero velocity commands
        self.last_cmd_vel.fill(0.0)
        self.last_logged_cmd.fill(0.0)
        
        # Reset timing
        self.last_accel_time = time.time()
        
        # Reset velocity history
        self.velocity_history.fill(0.0)
        self.history_index = 0
        self.history_count = 0
        
        # Reset velocity arrays
        self._limited_velocities.fill(0.0)
        self._target_velocities.fill(0.0)
        self._vel_diffs.fill(0.0)
        
        # Reset velocity change check 
        self._velocity_change_check.fill(False)
    
    def process_velocities(self, linear_x, linear_y, angular_z, filtered_distance, desired_distance, freshness_level='fresh'):
        """
        Process and limit velocities for smooth, natural movement.
        
        Args:
            linear_x: Calculated forward velocity
            linear_y: Calculated lateral velocity
            angular_z: Calculated angular velocity
            filtered_distance: Current filtered distance to target
            desired_distance: Desired distance to target
            freshness_level: Data freshness level ('fresh', 'stale', 'critical')
            
        Returns:
            tuple: (limited_linear_x, limited_linear_y, limited_angular_z)
        """
        try:
            # Validate inputs
            self._target_velocities[0] = float(linear_x) if linear_x is not None else 0.0
            self._target_velocities[1] = float(linear_y) if linear_y is not None else 0.0
            self._target_velocities[2] = float(angular_z) if angular_z is not None else 0.0
            filtered_distance = float(filtered_distance) if filtered_distance is not None else 1.0
            desired_distance = float(desired_distance) if desired_distance is not None else 1.0
            
            # If data is stale, apply conservative velocity scaling
            if (freshness_level == 'stale'):
                # Reduce all velocities to handle stale data
                velocity_scale = 0.5  # 50% of normal velocity
                self._target_velocities *= velocity_scale
                
                if self.debug_level >= 1:
                    self.logger.info(f"Applying stale data velocity reduction: {velocity_scale:.2f}")
            
            # Log incoming velocities at debug level
            if self.debug_level >= 2:
                self.logger.info(
                    f"Pre-limit velocities: x={self._target_velocities[0]:.3f}, y={self._target_velocities[1]:.3f}, θ={self._target_velocities[2]:.3f}",
                    throttle_duration_sec=1.0
                )
            
            # Apply distance-aware approach scaling for forward velocity
            self._target_velocities[0] = self._apply_approach_scaling(
                self._target_velocities[0], filtered_distance, desired_distance
            )
            
            # Apply predictive braking based on motion analysis
            self._target_velocities[0] = self._apply_predictive_braking(
                self._target_velocities[0], filtered_distance, desired_distance
            )
            
            # Apply acceleration limits
            current_time = time.time()
            self._apply_acceleration_limits(current_time)
            
            # Apply minimum velocity thresholds with hysteresis
            self._apply_minimum_thresholds()
            
            # Limit combined lateral and angular movement to prevent instability
            self._limit_combined_movements()
            
            # Apply maximum velocity limits using vectorized operation
            np.clip(
                self._limited_velocities, 
                -self.max_velocity_limits, 
                self.max_velocity_limits,
                out=self._limited_velocities
            )
            
            # Log limited velocities at debug level
            if self.debug_level >= 2:
                self.logger.info(
                    f"Post-limit velocities: x={self._limited_velocities[0]:.3f}, y={self._limited_velocities[1]:.3f}, θ={self._limited_velocities[2]:.3f}",
                    throttle_duration_sec=1.0
                )
            
            # Update last command for next cycle
            np.copyto(self.last_cmd_vel, self._limited_velocities)
            
            # Add to velocity history
            self._update_velocity_history()
            
            # Calculate if velocity changed significantly for logging
            self._velocity_change_check = np.abs(self.last_cmd_vel - self.last_logged_cmd) > 0.05
            
            # Log velocity commands if changed significantly
            if self.debug_level >= 1 or np.any(self._velocity_change_check):
                self.logger.info(
                    f"MOTION: x={self.last_cmd_vel[0]:.2f} y={self.last_cmd_vel[1]:.2f} θ={self.last_cmd_vel[2]:.2f}",
                    throttle_duration_sec=0.5
                )
                # Update last logged command
                np.copyto(self.last_logged_cmd, self.last_cmd_vel)
            
            # Return as tuple for compatibility
            return (self.last_cmd_vel[0], self.last_cmd_vel[1], self.last_cmd_vel[2])
            
        except Exception as e:
            self.logger.error(f"Error processing velocities: {str(e)}")
            # Return zero velocities on error for safety
            return (0.0, 0.0, 0.0)
    
    def _update_velocity_history(self):
        """Update velocity history using a circular buffer pattern."""
        # Store velocity in history using circular indexing
        self.velocity_history[self.history_index] = self.last_cmd_vel
        
        # Update index and count
        self.history_index = (self.history_index + 1) % self.buffer_size
        self.history_count = min(self.history_count + 1, self.buffer_size)
    
    def _apply_approach_scaling(self, linear_x, filtered_distance, desired_distance):
        """Apply distance-based scaling to forward velocity during approach."""
        try:
            # Calculate distance error (negative means too close)
            raw_distance_error = filtered_distance - desired_distance
            distance_error = abs(raw_distance_error)
            
            # Emergency stop for negative distance errors (robot too close)
            if raw_distance_error < -0.05:
                emergency_factor = max(0.0, 1.0 + raw_distance_error * 10.0)
                scaled_velocity = linear_x * emergency_factor
                
                if self.debug_level >= 1 and abs(scaled_velocity - linear_x) > 0.01:
                    self.logger.info(
                        f"Emergency approach reduction: error={raw_distance_error:.3f}m, "
                        f"factor={emergency_factor:.2f}, velocity={scaled_velocity:.3f}"
                    )
                return scaled_velocity
            
            # Apply progressive deceleration as robot approaches target
            if distance_error < self.approach_distance * 3.0:
                # Calculate approach scale with exponential curve
                normalized_distance = distance_error / self.approach_distance
                approach_factor = max(self.min_approach_factor, (normalized_distance)**2.5)
                
                # Apply stronger deceleration when very close
                if distance_error < self.approach_distance * 0.5:
                    approach_factor *= 0.3
                
                # Check closing speed using velocity history
                recent_velocities = self._get_recent_velocities(3)
                
                # Only calculate if we have history
                if self.history_count > 0:
                    # Get average forward velocity from history
                    avg_forward_vel = np.mean(recent_velocities[:, 0])
                    
                    # Apply more aggressive deceleration if approaching quickly
                    if avg_forward_vel > 0.1 and distance_error < self.approach_distance * 1.2:
                        speed_reduction = max(0.4, 1.0 - (avg_forward_vel / 0.4) * 0.5)
                        approach_factor *= speed_reduction
                        
                        if self.debug_level >= 1:
                            self.logger.info(
                                f"Enhanced deceleration: speed={avg_forward_vel:.2f}, "
                                f"additional factor={speed_reduction:.2f}"
                            )
                
                # Apply the scaling factor to forward velocity
                if abs(linear_x) > 0.01:
                    scaled_velocity = linear_x * approach_factor
                    
                    if self.debug_level >= 1 and abs(scaled_velocity - linear_x) > 0.02:
                        self.logger.info(
                            f"Approach scaling: distance_error={distance_error:.3f}m, "
                            f"factor={approach_factor:.2f}, velocity={scaled_velocity:.3f}"
                        )
                        
                    return scaled_velocity
            
            return linear_x
        except Exception as e:
            self.logger.error(f"Error in approach scaling: {str(e)}")
            return linear_x  # Return original value on error
    
    def _get_recent_velocities(self, count):
        """Get recent velocity history from circular buffer."""
        # Limit count to available history
        count = min(count, self.history_count)
        
        if count == 0:
            # Return empty array if no history
            return np.zeros((0, 3), dtype=np.float32)
            
        # Calculate indices to retrieve from buffer
        if self.history_index >= count:
            # Simple case: just get the last 'count' entries
            start_idx = self.history_index - count
            return self.velocity_history[start_idx:self.history_index]
        else:
            # Wrap-around case: get from end of buffer and beginning
            count_from_end = self.history_index
            count_from_start = count - count_from_end
            
            # Create a new array to hold the result
            result = np.zeros((count, 3), dtype=np.float32)
            
            # Copy data from end of buffer
            end_indices = np.arange(self.buffer_size - count_from_start, self.buffer_size)
            result[:count_from_start] = self.velocity_history[end_indices]
            
            # Copy data from beginning of buffer
            start_indices = np.arange(0, count_from_end)
            result[count_from_start:] = self.velocity_history[start_indices]
            
            return result
    
    def _apply_predictive_braking(self, linear_x, filtered_distance, desired_distance):
        """Apply predictive braking based on target motion analysis."""
        try:
            # Calculate distance error
            distance_error = abs(filtered_distance - desired_distance)
            
            # Only apply predictive braking during approach
            if distance_error < self.approach_distance * 2.0:
                # Only apply when we have velocity history
                if self.history_count > 0:
                    # Get robot's forward velocity (average of recent values)
                    recent_velocities = self._get_recent_velocities(min(3, self.history_count))
                    robot_forward_vel = np.mean(recent_velocities[:, 0])
                    
                    # Only apply when moving forward
                    if robot_forward_vel > 0.05:
                        # Estimate time to target
                        closing_speed = robot_forward_vel
                        if closing_speed > 0:
                            time_to_target = distance_error / closing_speed
                            
                            # Apply braking when getting close
                            if time_to_target < 1.5:
                                # Progressive braking curve
                                braking_factor = max(0.1, (time_to_target / 1.5)**1.5)
                                scaled_velocity = linear_x * braking_factor
                                
                                if self.debug_level >= 1 and abs(scaled_velocity - linear_x) > 0.02:
                                    self.logger.info(
                                        f"Predictive braking: time_to_target={time_to_target:.2f}s, "
                                        f"factor={braking_factor:.2f}, velocity={scaled_velocity:.3f}"
                                    )
                                
                                return scaled_velocity
            
            return linear_x
        except Exception as e:
            self.logger.error(f"Error in predictive braking: {str(e)}")
            return linear_x  # Return original value on error
    
    def _apply_acceleration_limits(self, current_time):
        """Apply acceleration limits to prevent jerky motion."""
        try:
            # Calculate time since last control step
            dt = current_time - self.last_accel_time
            self.last_accel_time = current_time
            dt = max(0.001, min(dt, 0.1))  # Bound dt to reasonable values
            
            # Base acceleration limits - scale with dt to handle varying control rates
            accel_limit = 2.5 * dt * 10.0
            angular_accel_limit = 3.0 * dt * 10.0
            
            # Safety check for NaN values
            if np.isnan(accel_limit) or np.isnan(angular_accel_limit):
                self.logger.warning("NaN detected in acceleration limits, using defaults")
                accel_limit = 0.25  # Default safe value
                angular_accel_limit = 0.3  # Default safe value
            
            # Calculate velocity differences
            np.subtract(self._target_velocities, self.last_cmd_vel, out=self._vel_diffs)
            
            # Apply acceleration limits
            for i in range(3):
                # Select appropriate limit based on axis
                limit = angular_accel_limit if i == 2 else accel_limit
                
                # Apply starting boost when beginning movement
                if abs(self.last_cmd_vel[i]) < 0.01 and abs(self._target_velocities[i]) > 0.01:
                    # Different boost factors by axis
                    boost = 3.0 if i == 0 else (5.0 if i == 1 else 3.0)
                    limit *= boost
                
                # Apply limit if change exceeds maximum
                if abs(self._vel_diffs[i]) > limit:
                    self._limited_velocities[i] = self.last_cmd_vel[i] + np.sign(self._vel_diffs[i]) * limit
                else:
                    self._limited_velocities[i] = self._target_velocities[i]
                
                # Safety check for NaN values in result
                if np.isnan(self._limited_velocities[i]):
                    self.logger.warning(f"NaN detected in velocity {i}, using previous value")
                    self._limited_velocities[i] = self.last_cmd_vel[i]
            
        except Exception as e:
            self.logger.error(f"Error applying acceleration limits: {str(e)}")
            # Copy target velocities directly on error
            np.copyto(self._limited_velocities, self._target_velocities)
    
    def _apply_minimum_thresholds(self):
        """Apply minimum velocity thresholds with hysteresis."""
        try:
            # Forward velocity (index 0)
            if abs(self._limited_velocities[0]) < self.min_linear_velocity:
                if abs(self._limited_velocities[0]) > self.min_linear_velocity * 0.3 and self._limited_velocities[0] != 0.0:
                    # Apply minimum threshold
                    self._limited_velocities[0] = self.min_linear_velocity * np.sign(self._limited_velocities[0])
                elif abs(self.last_cmd_vel[0]) < self.min_linear_velocity * 1.2:
                    # Zero when previously very small
                    self._limited_velocities[0] = 0.0
            
            # Lateral velocity (index 1)
            if abs(self._limited_velocities[1]) < self.min_linear_velocity:
                if abs(self.last_cmd_vel[1]) < self.min_linear_velocity * 1.2:
                    self._limited_velocities[1] = 0.0
            
            # Angular velocity (index 2)
            if abs(self._limited_velocities[2]) < self.min_angular_velocity:
                if abs(self.last_cmd_vel[2]) < self.min_angular_velocity * 1.2:
                    self._limited_velocities[2] = 0.0
            
        except Exception as e:
            self.logger.error(f"Error applying minimum thresholds: {str(e)}")
            # Leave values as is on error
    
    def _limit_combined_movements(self):
        """Limit combined lateral and angular movement to prevent instability."""
        try:
            # Check if both lateral and angular velocities are significant
            if abs(self._limited_velocities[1]) > 0.15 and abs(self._limited_velocities[2]) > 0.3:
                lateral_magnitude = abs(self._limited_velocities[1])
                angular_magnitude = abs(self._limited_velocities[2])
                
                # Apply stronger reduction for larger movements
                if lateral_magnitude > 0.2 and angular_magnitude > 0.4:
                    # Prevent tipping with significant reduction
                    self._limited_velocities[1] *= 0.6
                else:
                    # Milder reduction for smaller movements
                    scale_factor = 0.8 + (0.2 * (1.0 - min(1.0, lateral_magnitude / 0.2)))
                    self._limited_velocities[1] *= min(0.9, max(0.7, scale_factor))
            
        except Exception as e:
            self.logger.error(f"Error limiting combined movements: {str(e)}")
            # Leave values as is on error
    
    def get_average_velocity(self):
        """Get average velocity over recent history."""
        try:
            # Return zeros if no history
            if self.history_count == 0:
                return (0.0, 0.0, 0.0)
                
            # Calculate mean across all history
            if self.history_count < self.buffer_size:
                # Only use the filled part of the buffer
                mean_velocities = np.mean(self.velocity_history[:self.history_count], axis=0)
            else:
                # Use the entire buffer
                mean_velocities = np.mean(self.velocity_history, axis=0)
            
            return (mean_velocities[0], mean_velocities[1], mean_velocities[2])
        except Exception as e:
            self.logger.error(f"Error calculating average velocity: {str(e)}")
            return (0.0, 0.0, 0.0)  # Return zero velocity on error

#############################################
# Resource Monitoring Module
#############################################

class ResourceMonitoringModule:
    """Module for monitoring system resources and adapting behavior for Raspberry Pi 5."""
    
    def __init__(self, throttled_logger):
        """Initialize the resource monitoring module."""
        self.logger = throttled_logger
        
        # Initialize resource monitor from imported module
        try:
            self.resource_monitor = ResourceMonitor(logger=throttled_logger, update_interval=5.0)
            self.monitor_initialized = True
        except Exception as e:
            self.logger.error(f"Failed to initialize resource monitor: {str(e)}")
            self.monitor_initialized = False
        
        # Performance adjustment parameters - optimized for Raspberry Pi and fusion rate
        self.base_update_rate = 3.0   # Default control rate (changed from 20.0 to 3.0)
        self.current_update_rate = self.base_update_rate
        self.adaptive_control_rate = True
        self.min_update_rate = 2.0    # Don't go below 2Hz (changed from 8Hz)
        self.max_update_rate = 5.0    # Don't go above 5Hz - matches max fusion rate (changed from 20Hz)
        
        # Performance stats - optimize for fixed size
        self.cpu_samples = np.zeros(12, dtype=np.float32)  # 1 minute of 5-second samples
        self.cpu_samples_index = 0
        self.cpu_samples_count = 0
        
        self.control_cycles = np.zeros(100, dtype=np.float32)  # Last 100 control cycle times
        self.cycles_index = 0
        self.cycles_count = 0
        
        self.control_skips = 0  # Count of skipped control cycles due to high CPU
        
        # CPU thresholds - more aggressive for Raspberry Pi 5
        self.cpu_high_threshold = 50.0  # Lower threshold for earlier action
        self.cpu_low_threshold = 25.0   # Lower threshold for recovery
        
        # Fusion rate tracking
        self.current_fusion_rate = 1.0
        self.last_fusion_rate_update = 0.0
        
        # Flag to skip next cycle
        self.skip_next_cycle = False
        
        # Initialize startup time
        self._startup_time = time.time()
        
        # Current CPU usage (updated by update_cpu_stats)
        self.current_cpu_usage = 0.0
        
        # Register alert handler if monitor initialized
        if self.monitor_initialized:
            try:
                self.resource_monitor.add_alert_callback(self.handle_resource_alert)
            except Exception as e:
                self.logger.error(f"Failed to register resource alert callback: {str(e)}")
    
    def set_rate_limits(self, min_rate, max_rate, base_rate):
        """Set limits for rate adaptation."""
        if min_rate <= 0 or max_rate <= min_rate or base_rate < min_rate or base_rate > max_rate:
            self.logger.warning(f"Invalid rate limits: min={min_rate}, max={max_rate}, base={base_rate}")
            return False
            
        self.min_update_rate = min_rate
        self.max_update_rate = max_rate
        self.base_update_rate = base_rate
        self.current_update_rate = base_rate
        return True
    
    def set_cpu_thresholds(self, low_threshold, high_threshold):
        """Set CPU thresholds for rate adaptation."""
        if low_threshold >= high_threshold or low_threshold < 0 or high_threshold > 100:
            self.logger.warning(f"Invalid CPU thresholds: low={low_threshold}, high={high_threshold}")
            return False
            
        self.cpu_low_threshold = low_threshold
        self.cpu_high_threshold = high_threshold
        return True
    
    def set_fusion_rate(self, fusion_rate):
        """Set control rate based on detected fusion rate."""
        if fusion_rate <= 0:
            self.logger.warning(f"Invalid fusion rate: {fusion_rate}")
            return False
        
        # Bound fusion rate to reasonable values (0.5Hz to 10Hz)
        bounded_fusion_rate = max(0.5, min(10.0, fusion_rate))
        
        # Update fusion rate tracking
        self.current_fusion_rate = bounded_fusion_rate
        self.last_fusion_rate_update = time.time()
        
        # Set base rate to match fusion rate with small margin
        new_base_rate = min(self.max_update_rate, bounded_fusion_rate * 1.1)
        
        # Ensure rate is within bounds
        new_base_rate = max(self.min_update_rate, new_base_rate)
        
        # Update base rate if significantly different
        if abs(new_base_rate - self.base_update_rate) > 0.3:
            self.logger.info(f"Adjusting base control rate to match fusion: {self.base_update_rate:.1f}Hz → {new_base_rate:.1f}Hz")
            self.base_update_rate = new_base_rate
            
            # Also update current rate if it was at previous base rate
            if abs(self.current_update_rate - self.base_update_rate) < 0.5:
                self.current_update_rate = new_base_rate
            
            return True
        
        return False
    
    def handle_resource_alert(self, resource_type, value):
        """Handle resource alerts from the resource monitor."""
        try:
            if resource_type == 'cpu':
                startup_elapsed = time.time() - self._startup_time
                
                # Only throttle aggressively after grace period (5 seconds)
                if value > 80.0:  # More aggressive threshold for Raspberry Pi
                    if startup_elapsed < 5.0:
                        # During grace period, apply gentler throttling
                        self.logger.warning(f"CPU alert during startup grace period: {value:.1f}% - mild adjustment")
                        
                        # Less aggressive rate adjustment during startup
                        if self.current_update_rate > self.min_update_rate:
                            new_rate = max(self.min_update_rate, self.current_update_rate * 0.9)  # Only 10% reduction
                            self.update_control_rate(new_rate)
                    else:
                        # Normal throttling after grace period
                        self.logger.warning(f"Severe CPU usage alert: {value:.1f}% - adjusting control rate")
                        self.skip_next_cycle = True
                        
                        if self.current_update_rate > self.min_update_rate:
                            new_rate = max(self.min_update_rate, self.current_update_rate * 0.7)
                            self.update_control_rate(new_rate)
        except Exception as e:
            self.logger.error(f"Error handling resource alert: {str(e)}")
    
    def update_cpu_stats(self):
        """Update CPU usage statistics."""
        # Check if resource monitor is initialized
        if not self.monitor_initialized:
            return 0.0
            
        try:
            # Update the resource monitor
            self.resource_monitor.update()
            
            # Get current CPU usage and store in history
            cpu_usage = self.resource_monitor.get_cpu_usage()
            
            # Update current CPU usage for external access
            self.current_cpu_usage = cpu_usage
            
            # Store in circular buffer
            self.cpu_samples[self.cpu_samples_index] = cpu_usage
            self.cpu_samples_index = (self.cpu_samples_index + 1) % len(self.cpu_samples)
            self.cpu_samples_count = min(self.cpu_samples_count + 1, len(self.cpu_samples))
            
            # Adjust control rate if enabled
            if self.adaptive_control_rate:
                self._adjust_control_rate()
            
            return cpu_usage
        except Exception as e:
            self.logger.error(f"Error updating CPU stats: {str(e)}")
            return 0.0
    
    def _adjust_control_rate(self):
        """Adjust control loop rate based on CPU usage."""
        try:
            # Skip if no samples
            if self.cpu_samples_count == 0:
                return
                
            # Get average CPU usage
            avg_cpu = np.mean(self.cpu_samples[:self.cpu_samples_count])
            
            # More aggressive CPU-based adjustments for Raspberry Pi
            if avg_cpu > 80.0:
                # Very high CPU - use minimum rate
                if self.current_update_rate > self.min_update_rate:
                    self.update_control_rate(self.min_update_rate)
            elif avg_cpu > self.cpu_high_threshold:
                # High CPU - reduce rate more aggressively
                if self.current_update_rate > self.min_update_rate:
                    new_rate = max(self.min_update_rate, self.current_update_rate * 0.7)  # 30% reduction
                    self.update_control_rate(new_rate)
            elif avg_cpu < self.cpu_low_threshold and self.current_update_rate < self.base_update_rate:
                # Low CPU - increase rate, up to base rate, but more conservatively
                new_rate = min(self.base_update_rate, self.current_update_rate * 1.05)  # Only 5% increase
                self.update_control_rate(new_rate)
        except Exception as e:
            self.logger.error(f"Error adjusting control rate: {str(e)}")
    
    def update_control_rate(self, new_rate):
        """Update the control loop rate if it has changed significantly."""
        try:
            # Only update if change is significant
            if abs(new_rate - self.current_update_rate) < 0.1:
                return False
                
            # Log the change
            self.logger.info(f"Adjusting control rate: {self.current_update_rate:.1f}Hz → {new_rate:.1f}Hz")
            
            # Update rate
            self.current_update_rate = new_rate
            
            # Timer recreation would be handled by the main node
            return True
        except Exception as e:
            self.logger.error(f"Error updating control rate: {str(e)}")
            return False
    
    def _update_cycle_stats(self, cycle_duration):
        """Update control cycle statistics."""
        try:
            if cycle_duration is None:
                return 0.0
                
            # Store in circular buffer
            self.control_cycles[self.cycles_index] = cycle_duration
            self.cycles_index = (self.cycles_index + 1) % len(self.control_cycles)
            self.cycles_count = min(self.cycles_count + 1, len(self.control_cycles))
            
            # Calculate running average
            if self.cycles_count > 0:
                return np.mean(self.control_cycles[:self.cycles_count])
            return 0.0
        except Exception as e:
            self.logger.error(f"Error updating cycle stats: {str(e)}")
            return 0.0
    
    def should_skip_cycle(self):
        """Check if next cycle should be skipped due to resource constraints."""
        if self.skip_next_cycle:
            self.skip_next_cycle = False
            self.control_skips += 1
            return True
                
        return False
    
    def get_performance_stats(self):
        """Get current performance statistics."""
        try:
            # Add safety check for attribute initialization
            if not hasattr(self, 'cycles_count') or not hasattr(self, 'control_cycles'):
                self.logger.warning("Performance metrics not initialized correctly")
                return {
                    'cpu_avg': 0.0,
                    'cycle_time_ms': 0.0,
                    'skips': 0,
                    'update_rate': getattr(self, 'current_update_rate', 3.0),
                    'fusion_rate': getattr(self, 'current_fusion_rate', 1.0)
                }
            # Calculate averages with safety checks
            cpu_avg = 0.0
            if self.cpu_samples_count > 0:
                cpu_samples = self.cpu_samples[:self.cpu_samples_count]
                # Filter out invalid values
                valid_samples = cpu_samples[~np.isnan(cpu_samples)]
                if len(valid_samples) > 0:
                    cpu_avg = np.mean(valid_samples)
                    # Ensure CPU average is within valid range
                    cpu_avg = max(0.0, min(100.0, cpu_avg))
            cycle_time_avg = 0.0
            if self.cycles_count > 0:
                cycle_samples = self.control_cycles[:self.cycles_count]
                # FIX: Filter out NaN values from cycle_samples directly
                valid_cycles = cycle_samples[~np.isnan(cycle_samples)]
                if len(valid_cycles) > 0:
                    cycle_time_avg = np.mean(valid_cycles)
                    cycle_time_avg *= 1000.0  # Convert to ms
                    # Ensure cycle time is reasonable
                    cycle_time_avg = max(0.0, min(1000.0, cycle_time_avg))  # Cap at 1000ms
            # Ensure we have valid rates
            current_update_rate = max(0.1, min(20.0, self.current_update_rate))
            current_fusion_rate = max(0.1, min(10.0, self.current_fusion_rate))
            return {
                'cpu_avg': cpu_avg,
                'cycle_time_ms': cycle_time_avg,
                'skips': self.control_skips,
                'update_rate': current_update_rate,
                'fusion_rate': current_fusion_rate
            }
        except Exception as e:
            self.logger.error(f"Error getting performance stats: {str(e)}")
            return {
                'cpu_avg': 0.0,
                'cycle_time_ms': 0.0,
                'skips': 0,
                'update_rate': self.current_update_rate,
                'fusion_rate': self.current_fusion_rate
            }

class TransformStatus(Enum):
    """Enumeration of transform system status states."""
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    PARTIALLY_AVAILABLE = auto()
    READY = auto()
    ERROR = auto()

class TransformSystem:
    """Unified transform system that encapsulates both management and utilities."""
    def __init__(self, node, logger, tf_buffer):
        self.node = node
        self.logger = logger
        self.tf_buffer = tf_buffer
        self.status = TransformStatus.UNINITIALIZED
        self.status_message = "Transform system not initialized"
        self._initialized = False
        self._initialization_started = False
        self._initialization_failed = False
        self.verification_timer = None
        self.reference_frame = "base_link"
        self.imu_frame = "imu_link"
        self.transform_cache = {}
        self.matrix_cache = {}
        self.transform_ttl = 1.0
        self.matrix_ttl = 5.0
        self.transform_timeout = 0.1
        self.transform_verified = False
        self.use_matrix_transforms = True
        self.last_cleanup_time = time.time()
        self.transform_dependencies = []
        self.status_callbacks = []
        self.retry_count = 0
        self.max_retries = 30
        self.base_retry_interval = 0.2
        self.max_retry_interval = 5.0
        self.last_retry_time = 0.0
        self.initialization_start_time = 0.0
        self.status_publisher = node.create_publisher(String, '/transform_system/status', 10)
        self.logger.info("Unified TransformSystem initialized")

    def add_transform_dependency(self, source_frame, target_frame, required=True):
        self.transform_dependencies.append({
            'source': source_frame,
            'target': target_frame,
            'required': required
        })
        self.logger.debug(f"Added transform dependency: {source_frame} -> {target_frame}")
        return self

    def is_transform_system_ready(self):
        return self._initialized and self.status in (TransformStatus.READY, TransformStatus.PARTIALLY_AVAILABLE)

    def get_transform_between_frames(self, source_frame, target_frame, verify_only=False):        
        # Input validation
        if self.tf_buffer is None:
            if not verify_only:
                current_time = time.time()
                if current_time - getattr(self, '_last_transform_warning_time', 0.0) > 1.0:
                    self.logger.error("TF buffer is invalid, cannot get transform")
                    self._last_transform_warning_time = current_time
            return None
        if not source_frame or not target_frame:
            if not verify_only:
                self.logger.warning(f"Invalid frame IDs: source={source_frame}, target={target_frame}")
            return None
        if source_frame == target_frame:
            try:
                from geometry_msgs.msg import TransformStamped
                identity_transform = TransformStamped()
                identity_transform.header.frame_id = target_frame
                identity_transform.child_frame_id = source_frame
                identity_transform.header.stamp = rclpy.clock.Clock().now().to_msg()
                identity_transform.transform.rotation.w = 1.0
                identity_transform.transform.rotation.x = 0.0
                identity_transform.transform.rotation.y = 0.0
                identity_transform.transform.rotation.z = 0.0
                identity_transform.transform.translation.x = 0.0
                identity_transform.transform.translation.y = 0.0
                identity_transform.transform.translation.z = 0.0
                return identity_transform
            except Exception as e:
                if not verify_only:
                    self.logger.error(f"Error creating identity transform: {str(e)}")
                return None
        frame_key = f"{target_frame}_{source_frame}"
        current_time = time.time()
        if not verify_only:
            if frame_key in self.transform_cache:
                transform, timestamp = self.transform_cache[frame_key]
                if current_time - timestamp <= self.transform_ttl:
                    return transform
        if not verify_only and not self.is_transform_system_ready():
            if current_time - getattr(self, '_last_not_ready_warning', 0.0) > 1.0:
                self.logger.warning("Transform requested before system initialization complete")
                self._last_not_ready_warning = current_time
            return None
        try:
            transform_time = rclpy.time.Time()
            if not self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                transform_time,
                rclpy.duration.Duration(seconds=0.01)
            ):
                if not verify_only:
                    self.logger.debug(f"Frames not yet available: source={source_frame}, target={target_frame}")
                return None
        except Exception as e:
            if not verify_only:
                self.logger.debug(f"Transform existence check failed: {str(e)}")
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                transform_time,
                rclpy.duration.Duration(seconds=self.transform_timeout)
            )
            if not hasattr(transform, 'transform') or not hasattr(transform.transform, 'rotation'):
                if not verify_only:
                    self.logger.error(f"Invalid transform structure between {source_frame} and {target_frame}")
                return None
            rotation = transform.transform.rotation
            quat_norm = math.sqrt(rotation.x**2 + rotation.y**2 + rotation.z**2 + rotation.w**2)
            if abs(quat_norm - 1.0) > 0.01:
                if quat_norm > 0.0001:
                    rotation.x /= quat_norm
                    rotation.y /= quat_norm
                    rotation.z /= quat_norm
                    rotation.w /= quat_norm
                else:
                    rotation.x = 0.0
                    rotation.y = 0.0
                    rotation.z = 0.0
                    rotation.w = 1.0
                    if not verify_only:
                        self.logger.warning(
                            f"Invalid quaternion in transform between {source_frame} and {target_frame}, reset to identity"
                        )
            if not verify_only:
                self.transform_cache[frame_key] = (transform, current_time)
                if self.use_matrix_transforms:
                    try:
                        from pid_helpers import Matrix4x4
                        matrix = Matrix4x4.from_tf_transform(transform)
                        self.matrix_cache[frame_key] = (matrix, current_time)
                    except ImportError:
                        pass
                    except Exception as e:
                        self.logger.error(f"Error creating matrix from transform: {str(e)}")
            return transform
        except Exception as e:
            if verify_only:
                self.logger.debug(f"Transform lookup error: {str(e)}")
            else:
                if current_time - getattr(self, '_last_transform_warning_time', 0.0) > 1.0:
                    self.logger.warning(f"Transform lookup error: {str(e)}")
                    self._last_transform_warning_time = current_time
            return None

    def start_initialization(self):
        if self._initialization_started and not self._initialization_failed:
            return False
        self._initialization_started = True
        self._initialized = False
        self._initialization_failed = False
        self.initialization_start_time = time.time()
        self.status = TransformStatus.INITIALIZING
        self.status_message = "Transform verification in progress"
        self.retry_count = 0
        self._notify_status_change()
        if hasattr(self, 'verification_timer') and self.verification_timer is not None:
            self.verification_timer.cancel()
        self.verification_timer = self.node.create_timer(
            self.base_retry_interval,
            self._verify_transforms_callback
        )
        self.logger.info("Transform initialization started")
        return True

    def _verify_transforms_callback(self):
        try:
            all_available = True
            required_available = True
            total_deps = len(self.transform_dependencies)
            available_deps = 0
            if total_deps == 0:
                self.logger.warning("No transform dependencies defined - cannot verify")
                self._update_status(TransformStatus.ERROR, "No transform dependencies defined")
                self._initialization_failed = True
                if hasattr(self, 'verification_timer') and self.verification_timer is not None:
                    self.verification_timer.cancel()
                return
            for dep in self.transform_dependencies:
                source_frame = dep['source']
                target_frame = dep['target']
                try:
                    transform = self.get_transform_between_frames(
                        source_frame, target_frame, verify_only=True)
                    if transform is not None:
                        dep['available'] = True
                        available_deps += 1
                    else:
                        all_available = False
                        if dep['required']:
                            required_available = False
                except Exception as e:
                    all_available = False
                    if dep['required']:
                        required_available = False
                    if self.retry_count % 5 == 0:
                        self.logger.debug(f"Transform check failed: {source_frame} -> {target_frame}: {str(e)}")
            if all_available:
                self._update_status(TransformStatus.READY, "All transforms verified and available")
                self._initialized = True
                if hasattr(self, 'verification_timer') and self.verification_timer is not None:
                    self.verification_timer.cancel()
                    self.verification_timer = None
                init_time = time.time() - self.initialization_start_time
                self.logger.info(f"Transform initialization completed in {init_time:.2f} seconds")
            elif required_available:
                self._update_status(
                    TransformStatus.PARTIALLY_AVAILABLE,
                    f"Required transforms available ({available_deps}/{total_deps} total)"
                )
                self._initialized = True
            else:
                self._update_status(
                    TransformStatus.INITIALIZING,
                    f"Waiting for transforms ({available_deps}/{total_deps} available)"
                )
                self.retry_count += 1
                if self.retry_count >= self.max_retries:
                    self._update_status(
                        TransformStatus.ERROR,
                        f"Failed to initialize transforms after {self.max_retries} attempts"
                    )
                    self._initialization_failed = True
                    if hasattr(self, 'verification_timer') and self.verification_timer is not None:
                        self.verification_timer.cancel()
                        self.verification_timer = None
                    self.logger.error("Transform initialization failed - max retries exceeded")
                    return
                current_interval = min(
                    self.max_retry_interval,
                    self.base_retry_interval * (1.5 ** min(10, self.retry_count))
                )
                if (hasattr(self, 'verification_timer') and 
                    self.verification_timer is not None and
                    hasattr(self.verification_timer, 'timer_period_ns')):
                    try:
                        if abs(current_interval - self.verification_timer.timer_period_ns / 1e9) > 0.1:
                            self.verification_timer.cancel()
                            self.verification_timer = self.node.create_timer(
                                current_interval,
                                self._verify_transforms_callback
                            )
                            self.logger.debug(f"Retry interval adjusted to {current_interval:.2f}s")
                    except Exception as timer_e:
                        self.logger.warning(f"Error adjusting timer: {str(timer_e)}")
                        if self.verification_timer is not None:
                            try:
                                self.verification_timer.cancel()
                            except:
                                pass
                        self.verification_timer = self.node.create_timer(
                            current_interval,
                            self._verify_transforms_callback
                        )
                else:
                    self.verification_timer = self.node.create_timer(
                        current_interval,
                        self._verify_transforms_callback
                    )
                if self.retry_count % 5 == 0:
                    elapsed_time = time.time() - self.initialization_start_time
                    self.logger.info(
                        f"Transform initialization in progress: "
                        f"{available_deps}/{total_deps} transforms available "
                        f"(retry {self.retry_count}, elapsed {elapsed_time:.1f}s)"
                    )
        except Exception as e:
            self.logger.error(f"Error in transform verification: {str(e)}")
            self._update_status(TransformStatus.ERROR, f"Verification error: {str(e)}")
            self._initialization_failed = True
            if hasattr(self, 'verification_timer') and self.verification_timer is not None:
                try:
                    self.verification_timer.cancel()
                    self.verification_timer = None
                except:
                    pass

    def _update_status(self, status, message):
        if status != self.status:
            old_status = self.status
            self.status = status
            self.status_message = message
            self.logger.info(f"Transform system status: {old_status.name} -> {status.name}: {message}")
            status_msg = String()
            status_msg.data = f"{status.name}: {message}"
            self.status_publisher.publish(status_msg)
            self._notify_status_change()

    def _notify_status_change(self):
        for callback in self.status_callbacks:
            try:
                callback(self.status, self.status_message)
            except Exception as e:
                self.logger.error(f"Error in transform status callback: {str(e)}")

    def get_status(self):
        return {
            'status': self.status,
            'message': self.status_message,
            'initialization_started': self._initialization_started,
            'initialized': self._initialized,
            'initialization_failed': self._initialization_failed,
            'retry_count': self.retry_count,
            'dependencies': self.transform_dependencies
        }

    # ...other transform utility methods can be added here as needed...

#############################################
# Recovery Behavior Module
#############################################

class RecoveryBehaviorModule:
    """Module for handling recovery behaviors."""
    
    def __init__(self, throttled_logger):
        """Initialize recovery module with logger."""
        self.logger = throttled_logger
        
        # Recovery state tracking
        self.in_recovery = False
        self.recovery_start_time = 0.0
        self.recovery_phase = "none"  # none, stop, orient, approach
        
        # Command generator for recovery - reuse single instance
        self._cmd_vel_msg = Twist()
        
        # Exit suggestion flag
        self._exit_suggested = False
        
        # Pre-allocated parameters
        self._orient_gains = {
            'kp': 0.03  # Conservative gain
        }
        
        self._approach_gains = {
            'kp_distance': 0.2,  # Conservative gain
            'kp_lateral': 0.3    # Slightly more aggressive for lateral
        }
        
        # Data staleness tracking
        self._stale_data_stop_active = False
        self._last_staleness_log_time = 0.0
    
    def start_recovery(self):
        """Start the recovery mode sequence."""
        self.in_recovery = True
        self.recovery_start_time = time.time()
        self.recovery_phase = "stop"
        self._exit_suggested = False
        self.logger.info("Entering recovery mode - stopping robot")
        return self.get_stop_command()
        
    def handle_recovery(self, current_time, target_data=None, orientation_data=None):
        """
        Handle recovery mode with a three-phase approach.
        
        Args:
            current_time: Current time
            target_data: Dictionary with target position data
            orientation_data: Dictionary with orientation data
            
        Returns:
            tuple: (cmd_vel, is_complete) - Velocity command and whether recovery is complete
        """
        try:
            # Calculate recovery duration
            recovery_duration = current_time - self.recovery_start_time
            
            # Phase 1: Stop (0-1 seconds)
            if self.recovery_phase == "stop":
                if recovery_duration > 1.0:
                    self.recovery_phase = "orient"
                    self.logger.info("Recovery: Moving to orientation phase")
                    
                return self.get_stop_command(), False
                
            # Phase 2: Orient (1-3 seconds)
            if self.recovery_phase == "orient":
                # Only proceed if we have target and orientation data
                if not target_data or not orientation_data:
                    self.logger.warning("Missing data for recovery orient phase")
                    return self.get_stop_command(), False
                    
                cmd_vel = self._handle_orient_phase(target_data, orientation_data)
                
                # After 2 seconds in orient phase, move to approach
                if recovery_duration > 3.0:
                    self.recovery_phase = "approach"
                    self.logger.info("Recovery: Moving to approach phase")
                    
                return cmd_vel, False
            
            # Phase 3: Approach (3+ seconds)
            if self.recovery_phase == "approach":
                # Only proceed if we have target data
                if not target_data:
                    self.logger.warning("Missing target data for recovery approach phase")
                    return self.get_stop_command(), False
                    
                cmd_vel = self._handle_approach_phase(target_data)
                
                # Suggest exit after 6 seconds
                if recovery_duration > 6.0 and not self._exit_suggested:
                    self.logger.info(
                        "Recovery has been active for 6 seconds. "
                        "Consider transitioning back to tracking mode."
                    )
                    self._exit_suggested = True
                    
                return cmd_vel, True
            
            # Should never get here
            self.logger.error(f"Unknown recovery phase: {self.recovery_phase}")
            return self.get_stop_command(), False
        except Exception as e:
            self.logger.error(f"Error in recovery behavior: {str(e)}")
            return self.get_stop_command(), False
    
    def _handle_orient_phase(self, target_data, orientation_data):
        """Handle orientation phase of recovery."""
        # Validate input data
        if not isinstance(target_data, dict) or not target_data:
            self.logger.warning("Invalid target data for recovery orient phase")
            return self.get_stop_command()
            
        try:
            # Use filtered bearing from target data with validation
            bearing = target_data.get('bearing', 0.0)
            if not isinstance(bearing, (int, float)) or math.isnan(bearing):
                self.logger.warning(f"Invalid bearing value: {bearing}")
                bearing = 0.0
                
            # Validate orientation data
            if not isinstance(orientation_data, dict):
                self.logger.warning("Invalid orientation data format")
                orientation_data = {'yaw': 0.0}
                
            # Get yaw from orientation data if available
            yaw = orientation_data.get('yaw', 0.0)
            if not isinstance(yaw, (int, float)) or math.isnan(yaw):
                self.logger.warning(f"Invalid yaw value: {yaw}")
                yaw = 0.0
                
            # Convert to degrees for calculations and logging
            angular_degrees = math.degrees(bearing)
            
            # Ensure angular degrees is within reasonable range
            if abs(angular_degrees) > 180.0:
                angular_degrees = (angular_degrees + 180.0) % 360.0 - 180.0
                self.logger.warning(f"Normalized large angular error: {angular_degrees:.2f}°")
            
            # Reuse the single Twist message instance
            self._cmd_vel_msg.linear.x = 0.0
            self._cmd_vel_msg.linear.y = 0.0
            self._cmd_vel_msg.linear.z = 0.0
            self._cmd_vel_msg.angular.x = 0.0
            self._cmd_vel_msg.angular.y = 0.0
            
            # Only orient if angular error is significant
            if abs(angular_degrees) > 2.0:
                # Calculate angular velocity proportional to error using cached gain
                angular_velocity = self._orient_gains['kp'] * angular_degrees
                
                # Limit maximum velocity
                angular_velocity = max(-0.3, min(angular_velocity, 0.3))
                
                # Additional safety check for NaN
                if math.isnan(angular_velocity):
                    self.logger.warning("NaN detected in angular velocity calculation")
                    angular_velocity = 0.0
                
                self._cmd_vel_msg.angular.z = float(angular_velocity)
                
                self.logger.info(f"Recovery orient: angular_error={angular_degrees:.2f}°, velocity={angular_velocity:.2f}")
            else:
                # If angular error is small, stop rotation
                self._cmd_vel_msg.angular.z = 0.0
                self.logger.info(f"Recovery orient: good alignment achieved ({angular_degrees:.2f}°)")
            
            return self._cmd_vel_msg
        except Exception as e:
            self.logger.error(f"Error handling orient phase: {str(e)}")
            return self.get_stop_command()
    
    def _handle_approach_phase(self, target_data):
        """Handle approach phase of recovery."""
        # Validate target data
        if not isinstance(target_data, dict) or not target_data:
            self.logger.warning("Invalid target data for recovery approach phase")
            return self.get_stop_command()
            
        try:
            # Get filtered distance and lateral from target data
            distance = target_data.get('distance', 0.0)
            if not isinstance(distance, (int, float)):
                self.logger.warning(f"Invalid distance value: {distance}")
                distance = 0.0
                
            lateral = target_data.get('lateral', 0.0)
            if not isinstance(lateral, (int, float)):
                self.logger.warning(f"Invalid lateral value: {lateral}")
                lateral = 0.0
            
            # Set desired distance
            desired_distance = 1.0  # Default tracking distance
            
            # Calculate errors
            distance_error = distance - desired_distance
            lateral_error = lateral
            
            # Reuse the single Twist message instance
            self._cmd_vel_msg.linear.x = 0.0
            self._cmd_vel_msg.linear.y = 0.0
            self._cmd_vel_msg.linear.z = 0.0
            self._cmd_vel_msg.angular.x = 0.0
            self._cmd_vel_msg.angular.y = 0.0
            self._cmd_vel_msg.angular.z = 0.0
            
            # Only move if errors are significant
            if abs(distance_error) > 0.1 or abs(lateral_error) > 0.1:
                # Use cached gains for velocity calculation
                linear_velocity = self._approach_gains['kp_distance'] * distance_error
                lateral_velocity = self._approach_gains['kp_lateral'] * -lateral_error  # Invert for correct direction
                
                # Apply conservative scaling
                linear_velocity *= 0.7
                lateral_velocity *= 0.7
                
                # Limit maximum velocities
                linear_velocity = max(-0.1, min(linear_velocity, 0.1))
                lateral_velocity = max(-0.1, min(lateral_velocity, 0.1))
                
                self._cmd_vel_msg.linear.x = float(linear_velocity)
                self._cmd_vel_msg.linear.y = float(lateral_velocity)
                
                self.logger.info(
                    f"Recovery approach: distance_error={distance_error:.2f}m, "
                    f"lateral_error={lateral_error:.2f}m, "
                    f"velocity=({linear_velocity:.2f}, {lateral_velocity:.2f})"
                )
            else:
                # If errors are small, stop movement
                self.logger.info(
                    f"Recovery approach: good position achieved "
                    f"(distance_error={distance_error:.2f}m, lateral_error={lateral_error:.2f}m)"
                )
            
            return self._cmd_vel_msg
        except Exception as e:
            self.logger.error(f"Error handling approach phase: {str(e)}")
            return self.get_stop_command()
    
    def stop_robot(self):
        """Emergency stop method to immediately halt all robot motion."""
        try:
            # Get a zero-velocity command
            cmd_vel = self.get_stop_command()
            
            # Ensure we're not in recovery mode
            self.in_recovery = False
            self.recovery_phase = "none"
            
            return cmd_vel
        except Exception as e:
            self.logger.error(f"Error in stop_robot: {str(e)}")
            # Create a new Twist as fallback
            fallback = Twist()
            return fallback
    
    def get_stop_command(self):
        """Get a zero-velocity command."""
        self._cmd_vel_msg.linear.x = 0.0
        self._cmd_vel_msg.linear.y = 0.0
        self._cmd_vel_msg.linear.z = 0.0
        self._cmd_vel_msg.angular.x = 0.0
        self._cmd_vel_msg.angular.y = 0.0
        self._cmd_vel_msg.angular.z = 0.0
        return self._cmd_vel_msg
    
    def reset(self):
        """Reset recovery state."""
        self.in_recovery = False
        self.recovery_phase = "none"
        self._exit_suggested = False
        self._stale_data_stop_active = False
        self._last_staleness_log_time = 0.0
