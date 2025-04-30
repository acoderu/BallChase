"""
PID Controller Module

This module provides an improved PID controller implementation with enhanced features:
- Adaptive gains based on error trends
- Improved zero-crossing handling
- Coordinated movement strategies
- Anti-windup mechanisms

This controller is specifically designed for robotic applications requiring
precise motion control with smooth transitions.
"""

import time
import math
import logging
import numpy as np
from enum import Enum, auto

from pid_target_filter import ErrorTracker

class PIDControllers:
    """Namespace for PID controller classes and related functionality."""
    
    @staticmethod
    def create_controller_with_tracker(controller_type, kp, ki, kd, output_min, output_max, tracker_name, logger, max_history=8):
        """Factory method to create a controller with its error tracker properly initialized."""
        error_tracker = ErrorTracker(tracker_name, logger, max_history=max_history)
        controller = PIDControllers.create_controller(
            controller_type, kp, ki, kd, output_min, output_max
        )
        controller.error_tracker = error_tracker
        controller.logger = logger  
        if not hasattr(controller, 'error_tracker') or controller.error_tracker is None:
            raise RuntimeError(f"Failed to initialize error tracker for {controller_type.name} controller")
        return controller, error_tracker

    class StrategyManager:
        """
        Manages movement strategies for robotic control systems.
        
        This class centralizes the strategy table, error categorization, and
        strategy matching logic to provide a consistent interface for determining
        the appropriate movement strategy based on current error conditions.
        """
        
        def __init__(self, logger):
            """
            Initialize the strategy manager.
            
            Args:
                logger: Logger instance for diagnostic output
            """
            # Setup logging
            self.logger = logger
            
            # Initialize strategy table
            self.strategy_table = self._init_strategy_table()
            
            # Error categorization state
            self.prev_distance_category = "none"
            self.prev_lateral_category = "none"
            self.prev_angular_category = "none"
            
            # For wildcard matching
            self._key_tuple = ["none", "none", "none"]
            
            # Miscellaneous state
            self._robot_stopped = True
            self._startup_movement_cycles = 0
            self.current_strategy = "IDLE"
            self.debug_level = 0
            
            # For use by strategies that reduce angular corrections at target
            self.distance_threshold = 0.15  # meters
            
            # Strategy blender for smooth transitions (to be set externally)
            self.strategy_blender = None
            
            # Flag for using lateral control
            self.use_lateral_control = True
            
            # Define a fallback strategy
            self._fallback_strategy = PIDControllers.MovementStrategy(
                "SAFE_FALLBACK", True, False, False, 
                0.3, 0.0, 0.0, 
                "Fallback strategy due to selection error"
            )
            
        def _init_strategy_table(self):
            """
            Initialize the table-driven movement strategy definitions with 
            improved approach and balanced strategies.
            
            Returns:
                dict: The strategy table mapping error categories to strategies
            """
            # Define the strategy table with improved angular-first strategies and approach strategies
            return {
                # All errors within deadbands - no movement
                ("none", "none", "none"): [
                    "NO_MOVEMENT", False, False, False, 
                    0.0, 0.0, 0.0, 
                    "All errors within deadbands"
                ],
                
                # Very small errors - minimal corrections
                ("very_small", "very_small", "very_small"): [
                    "MINIMAL_CORRECTION", True, True, True, 
                    0.5, 0.4, 0.4,
                    "Minimal corrections for very small errors"
                ],
                
                # New high-priority strategy for large distance + any lateral/angular error
                ("large", "*", "*"): [
                    "DISTANCE_PRIORITY_APPROACH", True, True, True,
                    0.9, 0.6, 0.5,  # Strong forward, good lateral, moderate angular
                    "Distance-priority approach: {distance_error:.2f}m"
                ],
                
                # Large distance with large lateral - fast diagonal approach
                ("large", "large", "*"): [
                    "FAST_DIAGONAL_APPROACH", True, True, True,
                    0.9, 0.9, 0.4,  # Strong forward and lateral, moderate angular
                    "Fast diagonal approach: {distance_error:.2f}m, {lateral_error:.2f}m"
                ],
                
                # Large distance with medium lateral
                ("large", "medium", "*"): [
                    "FAST_DIAGONAL", True, True, True,
                    1.0, 0.8, 0.5,  # Maximum forward, strong lateral, moderate angular
                    "Fast diagonal approach: {distance_error:.2f}m, {lateral_error:.2f}m"
                ],
                
                # Medium distance with large lateral
                ("medium", "large", "*"): [
                    "LATERAL_PRIORITY", True, True, True,
                    0.7, 1.0, 0.4,  # Good forward, maximum lateral, moderate angular
                    "Lateral-priority movement: {lateral_error:.2f}m"
                ],
                
                # Pure lateral correction at target distance
                ("none", "very_small", "none"): [
                    "MICRO_LATERAL", False, True, False,
                    0.0, 0.7, 0.0,
                    "Micro lateral correction at target distance: {lateral_error:.2f}m"
                ],
                
                ("none", "small", "none"): [
                    "LATERAL_CORRECTION", False, True, False,
                    0.0, 0.9, 0.0,
                    "Lateral correction at target distance: {lateral_error:.2f}m"
                ],
                
                ("none", "medium", "none"): [
                    "STRONG_LATERAL", False, True, False, 
                    0.0, 1.0, 0.0, 
                    "Strong lateral correction at target distance: {lateral_error:.2f}m"
                ],
                
                ("none", "large", "none"): [
                    "MAX_LATERAL", False, True, False, 
                    0.0, 1.0, 0.0, 
                    "Maximum lateral correction at target distance: {lateral_error:.2f}m"
                ],
                
                # Pure distance corrections (no lateral, no angular)
                ("very_small", "none", "none"): [
                    "MICRO_APPROACH", True, False, False, 
                    0.7, 0.0, 0.0,
                    "Micro distance adjustment: {distance_error:.2f}m"
                ],
                
                ("small", "none", "none"): [
                    "FORWARD_ADJUSTMENT", True, False, False, 
                    0.8, 0.0, 0.0,
                    "Small distance adjustment: {distance_error:.2f}m"
                ],
                
                ("medium", "none", "none"): [
                    "FORWARD_APPROACH", True, False, False, 
                    1.0, 0.0, 0.0,  # Increased from 0.9 to 1.0 for faster approach
                    "Medium distance approach: {distance_error:.2f}m"
                ],
                
                ("large", "none", "none"): [
                    "FULL_APPROACH", True, False, False, 
                    1.0, 0.0, 0.0,
                    "Full distance approach: {distance_error:.2f}m"
                ],
                
                # Angular-first strategies by magnitude - IMPROVED
                ("*", "*", "very_large"): [
                    "ANGULAR_PRIMARY", True, True, True,  # Enabled lateral movement
                    0.4, 0.3, 0.9,  # Increased forward from 0.3 to 0.4, added lateral 0.3
                    "Angular correction with approach: {angular_error:.1f}°"
                ],
                
                ("*", "*", "large"): [
                    "ANGULAR_PRIORITY", True, True, True,  # Enabled lateral
                    0.5, 0.4, 0.8,  # Increased forward from 0.4, added lateral 0.4
                    "Angular correction with steady approach: {angular_error:.1f}°"
                ],
                
                ("*", "*", "medium_large"): [
                    "ANGULAR_BALANCED", True, True, True,
                    0.6, 0.5, 0.7,  # Increased forward from 0.5 to 0.6, lateral from 0.2 to 0.5
                    "Balanced approach with angular correction: {angular_error:.1f}°"
                ],
                
                ("*", "*", "medium"): [
                    "BALANCED", True, True, True, 
                    0.7, 0.5, 0.6,  # Increased forward from 0.6 to 0.7, lateral from 0.3 to 0.5
                    "Balanced movement with angular correction: {angular_error:.1f}°"
                ],
                
                ("*", "*", "small_medium"): [
                    "FORWARD_ANGULAR", True, True, True, 
                    0.8, 0.6, 0.5,  # Increased forward from 0.7 to 0.8, lateral from 0.4 to 0.6
                    "Forward movement with angular fine-tuning: {angular_error:.1f}°"
                ],
                
                ("*", "*", "small"): [
                    "FORWARD_PRIMARY", True, True, True, 
                    0.9, 0.7, 0.4,  # Increased forward from 0.8 to 0.9, lateral from 0.5 to 0.7
                    "Forward-focused movement with minor angular correction: {angular_error:.1f}°"
                ],
                
                ("*", "*", "very_small"): [
                    "POSITION_WITH_ALIGNMENT", True, True, True, 
                    1.0, 0.8, 0.3,  # Increased forward from 0.9 to 1.0, lateral from 0.6 to 0.8
                    "Position-focused movement with subtle alignment: {angular_error:.1f}°"
                ],
                
                # Special case strategies for when at target distance with angular error - IMPROVED
                ("none", "*", "medium"): [
                    "AT_TARGET_ANGULAR", True, True, True,  # Enabled lateral movement
                    0.4, 0.3, 0.6,  # Increased forward from 0.2 to 0.4, added lateral 0.3
                    "At target distance - angular correction with movement: {angular_error:.1f}°"
                ],
                
                ("none", "*", "medium_large"): [
                    "AT_TARGET_ANGULAR_STRONG", True, True, True,  # Enabled lateral movement
                    0.3, 0.3, 0.7,  # Increased forward from 0.1 to 0.3, added lateral 0.3
                    "At target distance - angular correction with movement: {angular_error:.1f}°"
                ],
                
                ("none", "*", "large"): [
                    "AT_TARGET_ANGULAR_MAX", True, True, True,  # Enabled forward and lateral
                    0.2, 0.3, 0.8,  # Added forward 0.2, lateral 0.3
                    "At target distance - angular correction with movement: {angular_error:.1f}°"
                ],
                
                # Combined distance + lateral but no angular - IMPROVED
                ("very_small", "very_small", "none"): [
                    "FINE_POSITION_ADJUSTMENT", True, True, False,
                    0.5, 0.7, 0.0,  # Increased forward from 0.4 to 0.5, lateral from 0.6 to 0.7
                    "Fine position adjustment with lateral emphasis"
                ],
                
                ("small", "small", "none"): [
                    "POSITION_ADJUSTMENT", True, True, False,
                    0.7, 0.8, 0.0,  # Increased forward from 0.6 to 0.7
                    "Small position adjustment with lateral priority"
                ],
                
                ("medium", "small", "none"): [
                    "APPROACH_WITH_LATERAL", True, True, False,
                    0.9, 0.7, 0.0,  # Increased forward from 0.8 to 0.9, lateral from 0.6 to 0.7
                    "Approach with lateral correction"
                ],
                
                ("small", "medium", "none"): [
                    "LATERAL_WITH_APPROACH", True, True, False,
                    0.7, 0.9, 0.0,  # Increased forward from 0.6 to 0.7, lateral from 0.8 to 0.9
                    "Lateral correction with approach component"
                ],
                
                # Combined distance + angular without lateral - IMPROVED
                ("small", "none", "small"): [
                    "APPROACH_WITH_ALIGNMENT", True, True, True,  # Enabled lateral for minor corrections
                    0.8, 0.2, 0.5,  # Increased forward from 0.7 to 0.8, added lateral 0.2
                    "Approach with alignment correction"
                ],
                
                ("medium", "none", "small"): [
                    "APPROACH_WITH_MINOR_ALIGNMENT", True, True, True,  # Enabled lateral
                    0.9, 0.2, 0.4,  # Increased forward from 0.8 to 0.9, added lateral 0.2
                    "Focused approach with minor alignment"
                ],
                
                # Angular-first based on distance - IMPROVED
                ("large", "*", "medium"): [
                    "ANGULAR_THEN_APPROACH", True, True, True,  # Enabled lateral
                    0.7, 0.4, 0.6,  # Increased forward from 0.5 to 0.7, added lateral 0.4
                    "Angular correction with approach from distance"
                ],
                
                # Diagonal movement - IMPROVED
                ("medium", "medium", "small"): [
                    "DIAGONAL_MOVEMENT", True, True, True,
                    0.8, 0.8, 0.4,  # Increased angular from 0.3 to 0.4
                    "Diagonal movement with small angular correction"
                ],
                
                ("medium", "medium", "none"): [
                    "PURE_DIAGONAL", True, True, False,
                    0.9, 0.9, 0.0,
                    "Pure diagonal movement without rotation"
                ],
                
                # Approach strategies for near-target behavior - IMPROVED
                ("small", "*", "*"): [
                    "APPROACH", True, True, True, 
                    0.8, 0.8, 0.4,  # Increased forward from 0.7 to 0.8, lateral from 0.7 to 0.8
                    "Approach mode - nearing target: {distance_error:.2f}m"
                ],
                        
                
                # Position correction without rotation - IMPROVED
                ("*", "*", "none"): [
                    "POSITION_ONLY", True, True, False,
                    0.9, 0.9, 0.0,  # Increased from 0.8 to 0.9
                    "Position correction without rotation"
                ],
                
                # ADDED: Deceleration strategy for controlled approach at close range
                ("very_small", "*", "*"): [
                    "DECELERATION_APPROACH", True, True, True, 
                    0.4, 0.6, 0.3,  # Reduced forward from 0.6 to 0.4 for controlled approach
                    "Deceleration approach - very close to target: {distance_error:.2f}m"
                ],

                # Add a new strategy for close approaches
                ("very_small", "*", "*"): [
                    "FINAL_APPROACH", True, True, True, 
                    0.4, 0.6, 0.3,  # Reduced forward from 0.6 to 0.4
                    "Final careful approach - very close to target: {distance_error:.2f}m"
                ],
                
                # Fallback strategy - IMPROVED
                ("*", "*", "*"): [
                    "BALANCED", True, True, True, 
                    0.8, 0.7, 0.5,  # Increased forward from 0.7 to 0.8, lateral from 0.6 to 0.7
                    "Balanced movement strategy (fallback)"
                ],
            }
        
        def categorize_error(self, error, error_type="distance", prev_category=None, lenient_factor=1.0):
            """
            Categorize an error value with hysteresis to prevent oscillation.
            Modified to support lenient categorization for angular errors.
            
            Args:
                error: The error value to categorize
                error_type: The type of error (distance, lateral, angular)
                prev_category: Previous category for hysteresis
                lenient_factor: Factor to make categories more lenient (higher means more lenient)
                
            Returns:
                String: The error category
            """
            abs_error = abs(error)
            
            # Select appropriate thresholds based on error type
            if error_type == "angular":
                # MODIFIED: Significantly increased angular thresholds
                deadband = 5.0  # Increased from 3.0 degrees
                very_small_threshold = deadband * lenient_factor
                small_threshold = deadband * 2.0 * lenient_factor
                small_medium_threshold = deadband * 3.0 * lenient_factor
                medium_threshold = deadband * 5.0 * lenient_factor  # Increased
                medium_large_threshold = deadband * 8.0 * lenient_factor  # Increased
                large_threshold = deadband * 12.0 * lenient_factor  # Increased
                very_large_threshold = deadband * 16.0 * lenient_factor  # Increased
            elif error_type == "lateral":
                # MODIFIED: Increased lateral thresholds
                deadband = 0.08  
                very_small_threshold = deadband
                small_threshold = deadband * 1.8
                small_medium_threshold = deadband * 3.0
                medium_threshold = deadband * 4.0
                medium_large_threshold = deadband * 6.0
                large_threshold = deadband * 8.0
                very_large_threshold = deadband * 10.0
            else:  # distance
                # MODIFIED: Increased distance thresholds
                deadband = 0.15  # Increased from 0.1 meters
                very_small_threshold = deadband
                small_threshold = deadband * 2.0  # Increased from 1.5
                small_medium_threshold = deadband * 3.0  # Increased from 2.5
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
        
        def match_strategy(self, key, strategies=None):
            """
            Match a key against the strategy table with wildcard support.
            
            Args:
                key: Tuple of (distance_state, lateral_state, angular_state)
                strategies: Optional strategy table to use (defaults to self.strategy_table)
                
            Returns:
                List: The matched strategy definition
            """
            # Use provided strategies or default to the built-in table
            strategies = strategies or self.strategy_table
            
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
        
        def determine_strategy(self, distance_error, lateral_error, angular_error_degrees, is_robot_stopped=False):
            """
            Determine the optimal movement strategy using table-driven approach
            with hysteresis and angular-first prioritization.
            Modified to reduce angular corrections when at target distance and
            prioritize forward movement during startup.
            
            Args:
                distance_error: Error in distance (meters)
                lateral_error: Error in lateral position (meters)
                angular_error_degrees: Error in angular position (degrees)
                is_robot_stopped: Whether the robot is currently stopped
                
            Returns:
                MovementStrategy: Strategy object including strategy name, movement flags, and scale factors
            """
            current_time = time.time()
            
            try:
                # Set robot stopped state for internal use
                self._robot_stopped = is_robot_stopped
                
                # Add startup movement counter if it doesn't exist
                if not hasattr(self, '_startup_movement_cycles'):
                    self._startup_movement_cycles = 0
                
                # For first few cycles after movement starts, prioritize forward movement
                if not is_robot_stopped and self._startup_movement_cycles < 5:
                    self._startup_movement_cycles += 1
                    
                    # If significant distance error exists, prioritize forward movement
                    if abs(distance_error) > 0.2:  # Significant distance error
                        # Override strategy to prioritize forward movement
                        startup_strategy = PIDControllers.MovementStrategy(
                            "STARTUP_FORWARD_PRIORITY",
                            True, True, True,
                            0.7, 0.3, 0.3,
                            "Startup forward priority - quick response mode"
                        )
                        return startup_strategy
                
                # Check if robot is at target distance and reduce angular priority if so
                at_target_distance = abs(distance_error) < self.distance_threshold * 1.5
                
                # Key tuple for strategy lookup
                self._key_tuple = ["none", "none", "none"]
                
                # Categorize errors into states with hysteresis
                self._key_tuple[0] = self.categorize_error(
                    distance_error, "distance", self.prev_distance_category)
                self._key_tuple[1] = self.categorize_error(
                    lateral_error, "lateral", self.prev_lateral_category)
                
                # Modify angular error categorization when at target distance
                if at_target_distance and self._key_tuple[0] == "none":
                    # At target distance, categorize angular errors more leniently
                    # This effectively makes the robot less concerned with perfect angular alignment
                    angular_category = self.categorize_error(
                        angular_error_degrees, 
                        "angular", 
                        self.prev_angular_category,
                        lenient_factor=1.5  # Make angular categories 50% more lenient
                    )
                    
                    # Optionally downgrade categories for angular errors when at target distance
                    if angular_category == "medium":
                        angular_category = "small_medium"  # Downgrade medium to small_medium
                    elif angular_category == "small_medium":
                        angular_category = "small"  # Downgrade small_medium to small
                    elif angular_category == "small":
                        angular_category = "very_small"  # Downgrade small to very_small
                        
                    self._key_tuple[2] = angular_category
                    
                    if self.debug_level >= 2:
                        self.logger.info(
                            f"At target distance: using more lenient angular categorization: {angular_category}"
                        )
                else:
                    # Normal angular categorization
                    self._key_tuple[2] = self.categorize_error(
                        angular_error_degrees, "angular", self.prev_angular_category)
                
                # Save categories for next iteration's hysteresis
                self.prev_distance_category = self._key_tuple[0]
                self.prev_lateral_category = self._key_tuple[1]
                self.prev_angular_category = self._key_tuple[2]
                
                # Create lookup key
                key = tuple(self._key_tuple)
                
                # Get strategy definition from table
                strategy_def = self.match_strategy(key, self.strategy_table)
                
                # Format the reason string with actual error values
                name, use_forward, use_lateral, use_angular, forward_scale, lateral_scale, angular_scale, reason_template = strategy_def
                
                reason = reason_template.format(
                    distance_error=abs(distance_error),
                    lateral_error=abs(lateral_error),
                    angular_error=abs(angular_error_degrees)
                )
                
                # Create a MovementStrategy object
                target_strategy = PIDControllers.MovementStrategy(
                    name, use_forward, use_lateral, use_angular,
                    forward_scale, lateral_scale, angular_scale, reason
                )
                
                # Additional logic for at-target-distance
                if at_target_distance and (name != "NO_MOVEMENT" and angular_scale > 0.0):
                    # Reduce angular scale further if already at target distance
                    adjusted_angular_scale = angular_scale * 0.8  # Reduce by an additional 20%
                    target_strategy.angular_scale = adjusted_angular_scale
                    
                    if self.debug_level >= 2:
                        self.logger.info(
                            f"At target distance: reducing angular scale from {angular_scale:.2f} to {adjusted_angular_scale:.2f}"
                        )
                
                # If blender exists, update it with the target strategy
                if hasattr(self, 'strategy_blender') and self.strategy_blender is not None:
                    blend_started = self.strategy_blender.update_target(target_strategy, current_time)
                    
                    # Get the current (possibly blended) strategy
                    current_strategy = self.strategy_blender.get_current_strategy(current_time)
                    
                    # Keep track of the current strategy name for logging
                    self.current_strategy = current_strategy.strategy_name
                    
                    # Log strategy changes (throttled)
                    if blend_started or self.debug_level >= 2:
                        self.logger.info(
                            f"Strategy selected: {current_strategy.strategy_name}, params: "
                            f"forward={current_strategy.forward_scale:.1f}, "
                            f"lateral={current_strategy.lateral_scale:.1f}, "
                            f"angular={current_strategy.angular_scale:.1f}"
                        )
                        # Log detailed error info with strategy change
                        self.logger.info(
                            f"Error categories: distance={self._key_tuple[0]}, "
                            f"lateral={self._key_tuple[1]}, angular={self._key_tuple[2]}"
                        )
                    
                    # Return strategy object
                    return current_strategy
                else:
                    # If no blender, return target strategy directly
                    self.current_strategy = target_strategy.strategy_name
                    return target_strategy
            except Exception as e:
                self.logger.error(f"Strategy determination error: {str(e)}")
                # Return fallback strategy on error
                return self._fallback_strategy

        @staticmethod
        def create_strategy_from_definition(strategy_def, distance_error=0.0, lateral_error=0.0, angular_error=0.0):
            """
            Create a MovementStrategy object from a strategy definition.
            
            Args:
                strategy_def: Strategy definition from the strategy table
                distance_error: Distance error value for reason formatting
                lateral_error: Lateral error value for reason formatting
                angular_error: Angular error value for reason formatting
                
            Returns:
                MovementStrategy: The created strategy object
            """
            try:
                # Extract strategy components
                name, use_forward, use_lateral, use_angular, forward_scale, lateral_scale, angular_scale, reason_template = strategy_def
                reason = reason_template.format(
                    distance_error=abs(distance_error),
                    lateral_error=abs(lateral_error),
                    angular_error=abs(angular_error)
                )
                return PIDControllers.MovementStrategy(
                    name, use_forward, use_lateral, use_angular,
                    forward_scale, lateral_scale, angular_scale, reason
                )
            except Exception as e:
                print(f"Error creating strategy from definition: {e}")
                return PIDControllers.MovementStrategy(
                    "FALLBACK", True, False, False,
                    0.3, 0.0, 0.0,
                    "Fallback due to definition error"
                )
        
        def set_debug_level(self, level):
            """Set the debug output level."""
            self.debug_level = level
        
        def set_distance_threshold(self, threshold):
            """Set the distance threshold for at-target behavior."""
            if threshold > 0:
                self.distance_threshold = threshold
        
        def set_lateral_control(self, enabled):
            """Enable or disable lateral control."""
            self.use_lateral_control = enabled

    class CoordinatedController:
        """Controller that coordinates lateral and angular movements."""
        
        def __init__(self, linear_pid, angular_pid, logger, config=None):
            """
            Initialize the coordinated controller.
            
            Args:
                linear_pid: PID controller for lateral movement
                angular_pid: PID controller for angular movement
                logger: Logger instance for diagnostic output
                config: Configuration dictionary
            """
            self.linear_pid = linear_pid
            self.angular_pid = angular_pid
            self.logger = logger
            
            # Default configuration with improved values - extract to instance variables
            self.coupling_factor = 0.4         # Reduced from 0.7 to allow more lateral movement
            self.min_angle_for_reduction = 0.1  # ~5.7 degrees
            self.zero_angle_threshold = 0.03    # ~1.7 degrees
            self.max_angle_factor = 0.3         # Reduced from 0.5 to be less aggressive
            self.same_sign_scale = 0.8          # Scaling when errors have same sign
            self.opposite_sign_scale = 1.2      # Increased from 1.0 to prioritize when errors help each other
            self.smoothing_factor = 0.4         # Reduced from 0.6 for faster response
            
            # Update with provided config
            if config:
                # Only set attributes that exist
                for key, value in config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                
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
                angular_progress = 1.0 - min(1.0, angular_magnitude / self.max_angle_factor)
                
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
            lateral_velocity = raw_lateral_velocity
            
            # Only apply coupling if angular error is significant
            if angular_magnitude > self.min_angle_for_reduction:
                # Normalize angular error to 0-1 range up to max_angle_factor
                normalized_angle = min(1.0, angular_magnitude / self.max_angle_factor)
                
                # Calculate lateral velocity reduction - more moderate than before
                lateral_reduction = normalized_angle * self.coupling_factor
                
                # Reduce lateral velocity when angular error is large
                lateral_velocity = raw_lateral_velocity * (1.0 - lateral_reduction)
                
            # Set angular velocity directly from PID
            angular_velocity = raw_angular_velocity
            
            # 2. Adjust based on sign relationship between errors
            same_sign = (lateral_error * angular_error) > 0
            
            if abs(angular_error) > self.zero_angle_threshold:
                if same_sign:
                    # Same sign - needs coordinated movement
                    lateral_velocity *= self.same_sign_scale
                else:
                    # Opposite sign - movement naturally helps correction
                    lateral_velocity *= self.opposite_sign_scale
                    
            # 3. Apply smoothing to prevent jerky transitions
            if self.last_lateral_velocity is not None and self.last_angular_velocity is not None:
                # Reduced smoothing factor for more responsive control
                lateral_velocity = self.last_lateral_velocity * (1 - self.smoothing_factor) + \
                                lateral_velocity * self.smoothing_factor
                                
                angular_velocity = self.last_angular_velocity * (1 - self.smoothing_factor) + \
                                angular_velocity * self.smoothing_factor
            
            # Store values for next iteration
            self.last_lateral_velocity = lateral_velocity
            self.last_angular_velocity = angular_velocity
            self.last_update_time = current_time
            
            # Log coordination details occasionally
            if hasattr(self, 'compute_count') and self.compute_count % 20 == 0:
                coupling_str = f"{normalized_angle * self.coupling_factor:.2f}" if 'normalized_angle' in locals() else "N/A"
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
            self.strategy_name = name
            self.use_forward = use_forward
            self.use_lateral = use_lateral
            self.use_angular = use_angular
            self.forward_scale = forward_scale
            self.lateral_scale = lateral_scale
            self.angular_scale = angular_scale
            self.reason = reason
    
    class StrategyBlender:
        """Handles smooth transitions between movement strategies."""
        
        def __init__(self, logger, blend_duration=0.1):  # Reduced from 0.5 to 0.2 seconds
            """Initialize the strategy blender with faster transitions."""
            self.current_strategy = None
            self.target_strategy = None
            self.blend_start_time = 0.0
            self.blending_active = False
            self.blend_duration = blend_duration
            self.direction_change_boost = 2.5  # Speed up transitions when direction changes
            self.previous_direction = None
            
            # Create a reusable blended strategy object
            self._blended_strategy = None
            
            # Logger
            self.logger = logger
        
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
            if target_strategy.strategy_name != self.current_strategy.strategy_name:
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
            """Simplified smoothstep function for smoother transitions."""
            # Bound x to [0,1]
            x = max(0.0, min(1.0, x))
            # Use cubic smoothstep: 3x^2 - 2x^3 (simpler than 6x^5 - 15x^4 + 10x^3)
            return x * x * (3.0 - 2.0 * x)
        
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
            
            # Initialize blended strategy object on first use
            if self._blended_strategy is None:
                self._blended_strategy = PIDControllers.MovementStrategy(
                    "blended", False, False, False, 0, 0, 0, "")
            
            # Update the reusable blended strategy object
            self._blended_strategy.strategy_name = "blended"
            
            # Determine boolean flags using OR logic for smoother transitions
            self._blended_strategy.use_forward = self.target_strategy.use_forward or (
                self.current_strategy.use_forward and blend_factor < 0.5)
                
            self._blended_strategy.use_lateral = self.target_strategy.use_lateral or (
                self.current_strategy.use_lateral and blend_factor < 0.5)
                
            self._blended_strategy.use_angular = self.target_strategy.use_angular or (
                self.current_strategy.use_angular and blend_factor < 0.5)
            
            # Blend continuous parameters
            inv_blend = 1.0 - blend_factor  # Cache the inverse blend factor
            self._blended_strategy.forward_scale = self.current_strategy.forward_scale * inv_blend + \
                        self.target_strategy.forward_scale * blend_factor
                        
            self._blended_strategy.lateral_scale = self.current_strategy.lateral_scale * inv_blend + \
                        self.target_strategy.lateral_scale * blend_factor
                        
            self._blended_strategy.angular_scale = self.current_strategy.angular_scale * inv_blend + \
                        self.target_strategy.angular_scale * blend_factor
            
            self._blended_strategy.reason = f"Blending strategies: {blend_factor*100:.0f}% complete"
            
            return self._blended_strategy
        
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
            # Don't reset _blended_strategy - it's reused

    class ControllerType(Enum):
        """Enum for different controller types."""
        LINEAR_X = auto()
        LINEAR_Y = auto()
        ANGULAR = auto()
        CUSTOM = auto()
        
    @staticmethod
    def create_controller(controller_type, kp, ki, kd, output_min, output_max):
        """Factory method to create a controller of the specified type with appropriate defaults."""
        # NOTE: Error trackers should be created at the node level
        # and assigned to the controller after creation
        if controller_type == PIDControllers.ControllerType.LINEAR_X:
            return PIDControllers.ImprovedPID(kp, ki, kd, output_min, output_max, "Linear X")
        elif controller_type == PIDControllers.ControllerType.LINEAR_Y:
            return PIDControllers.ImprovedPID(kp, ki, kd, output_min, output_max, "Linear Y")
        elif controller_type == PIDControllers.ControllerType.ANGULAR:
            return PIDControllers.ImprovedPID(kp, ki, kd, output_min, output_max, "Angular")
        else:
            return PIDControllers.ImprovedPID(kp, ki, kd, output_min, output_max, "Custom")
        
    @staticmethod
    def categorize_error(error, error_type="distance", prev_category=None, lenient_factor=1.0):
        """Static wrapper for error categorization to be used by other modules."""
        temp_manager = PIDControllers.StrategyManager()
        return temp_manager.categorize_error(error, error_type, prev_category, lenient_factor)

    class ImprovedPID:
        """PID controller with enhanced integral handling and adaptive gains."""
        # NOTE: Initialization errors should be raised explicitly and not masked, for consistency with the node's error handling policy.
        
        def __init__(self, base_kp, base_ki, base_kd, output_min, output_max, name="PID", logger=None):
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
            self.output_range = output_max - output_min  # Precompute this value
            self.name = name
            
            # Reference to parent controller
            self.pid_controller = None
            
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
            # Precompute max integral value
            self.max_integral = self.output_range / base_ki if base_ki > 0 else 1.0
            
            # Controller-specific max integral values
            if name == "Angular":
                self.max_integral *= 0.7  # 30% smaller limit for angular
            
            # Error tracker - MODIFIED to ensure it's initialized to None
            # and set externally by the node
            self.error_tracker = None
            
            # Performance metrics
            self.settling_time = 0.0
            self.overshoot = 0.0
            self.rise_time = 0.0
            self.steady_state = False
            
            # Logger for controller-specific logs at lower frequency
            self.logger = logger
        
        def validate_initialization(self):
            """Validate that the controller is properly initialized."""
            if not hasattr(self, 'error_tracker') or self.error_tracker is None:
                raise RuntimeError(f"PID controller '{self.name}' has no error tracker initialized")
            required_methods = ['update', 'get_trend', 'reset']
            for method in required_methods:
                if not hasattr(self.error_tracker, method):
                    raise RuntimeError(f"Error tracker for '{self.name}' missing required method '{method}'")
            return True

        def compute(self, error, current_time=None, force_zero=False, error_trend=None):
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
                # Validate error tracker is available if no error_trend is provided
                if error_trend is None and (not hasattr(self, 'error_tracker') or self.error_tracker is None):
                    raise RuntimeError(f"PID controller '{self.name}' has no error tracker and no trend provided")
                # Track call count and log only occasionally to reduce overhead
                if not hasattr(self, 'compute_count'):
                    self.compute_count = 0
                self.compute_count += 1
                
                should_log = self.compute_count % 50 == 0  # Reduced logging frequency
                # Log large output jumps or output saturation at debug_level >= 1
                large_jump = abs(self.last_output - self.prev_error) > self.output_range * 0.4
                debug_level = getattr(self, 'debug_level', 0)
                if debug_level >= 2 or should_log:
                    self.logger.info(f"PID {self.name} compute: error={error:.3f}, force_zero={force_zero}")
                if debug_level >= 1 and large_jump:
                    self.logger.info(
                        f"PID {self.name} large output change: prev={self.prev_error:.3f} -> output={self.last_output:.3f}",
                        throttle_duration_sec=1.0
                    )
                
                # Special case: forced zero output
                if force_zero:
                    self.prev_error = error
                    self.last_p_term = 0.0
                    self.last_i_term = 0.0
                    self.last_d_term = 0.0
                    self.last_output = 0.0
                    return 0.0
                    
                # Initialize time values
                if current_time is None:
                    current_time = time.time()
                
                # First-time initialization
                if self.last_time is None:
                    self.last_time = current_time
                    self.prev_error = error
                    # P-only control on first iteration (no I or D)
                    output = self.kp * error
                    self.last_p_term = output
                    self.last_output = max(self.output_min, min(self.output_max, output))
                    return float(self.last_output)
                    
                # Calculate time interval with safety check
                dt = current_time - self.last_time
                if dt <= 0.001:  # Protect against very small or negative dt
                    dt = 0.01  # Fallback to prevent division by zero (assume 100Hz)
                
                # Detect and handle zero crossings - inlined for efficiency
                current_sign = 1 if error > 0 else (-1 if error < 0 else 0)
                
                # Check for sign change (zero crossing)
                if self.prev_sign != 0 and current_sign != 0 and self.prev_sign != current_sign:
                    # Record the zero crossing
                    self.zero_crossing_time = current_time
                    self.sign_change_count += 1
                    
                    # Apply controller-specific integral reset factors
                    if self.name == "Angular":
                        self.integral *= 0.05  # Much more aggressive for angular
                    elif self.name == "Linear Y":
                        self.integral *= 0.1  # More aggressive for lateral
                    else:
                        self.integral *= 0.2  # Default behavior
                else:
                    # No sign change - gradually reduce sign change count for hysteresis
                    self.sign_change_count = max(0, self.sign_change_count - 0.1)
                
                # Update previous sign
                if error != 0:
                    self.prev_sign = current_sign
                
                # Adjust PID gains based on current conditions - flattened conditionals
                # Create a gains array for vectorized operations
                kp_factor, ki_factor, kd_factor = 1.0, 1.0, 1.0
                error_magnitude = abs(error)
                
                # Apply trend-based adjustments with flattened conditionals
                # MODIFIED to use provided error_trend if self.error_tracker is None
                if self.error_tracker is not None:
                    error_trend = self.error_tracker.get_trend()
                if error_trend < -0.1:  # Error is decreasing
                    kp_factor = max(0.8, 1.0 - error_magnitude * 0.5)
                    ki_factor = max(0.5, 1.0 - error_magnitude)
                    kd_factor = min(1.5, 1.0 + error_magnitude)
                elif error_trend > 0.1:  # Error is increasing
                    kp_factor = min(1.2, 1.0 + error_magnitude * 0.2)
                    ki_factor = min(1.1, 1.0 + error_magnitude * 0.1)
                    kd_factor = max(0.9, 1.0 - error_magnitude * 0.1)
                
                # Special case for zero crossing
                if self.zero_crossing_time is not None:
                    time_since_crossing = current_time - self.zero_crossing_time
                    if time_since_crossing < 0.5:  # Within 0.5 seconds of zero crossing
                        # Enhance derivative, reduce integral during zero crossing
                        kd_factor *= 1.2
                        ki_factor *= 0.5
                
                # Apply controller-specific adjustments
                if self.name == "Linear Y":
                    # More aggressive damping for lateral control
                    kd_factor *= 1.2
                    # Less integral for lateral to prevent overshooting
                    ki_factor *= 0.8
                elif self.name == "Angular":
                    # Reduced integral gain for angular control
                    ki_factor *= 0.7
                    # Reduced derivative gain to prevent overshoot
                    kd_factor *= 0.6
                
                # Gradually adjust gains - using NumPy for vectorized operations
                adjust_rate = self.gain_adjust_rate
                inv_adjust_rate = 1.0 - adjust_rate
                
                self.kp = self.kp * inv_adjust_rate + (self.base_kp * kp_factor) * adjust_rate
                self.ki = self.ki * inv_adjust_rate + (self.base_ki * ki_factor) * adjust_rate
                self.kd = self.kd * inv_adjust_rate + (self.base_kd * kd_factor) * adjust_rate
                
                # Calculate PID terms
                p_term = self.kp * error
                
                # Integral term calculation with anti-windup
                # Check if output is likely to saturate
                predicted_output = p_term + self.last_i_term + self.last_d_term
                is_saturated = (predicted_output >= self.output_max) or (predicted_output <= self.output_min)
                
                # Calculate integral term
                if not is_saturated:
                    # Check for recent zero crossing
                    recently_crossed_zero = (self.zero_crossing_time is not None and 
                                          (current_time - self.zero_crossing_time) < 0.5)
                    
                    if recently_crossed_zero:
                        # Near zero crossing, allow faster integral changes
                        if abs(error) > self.integral_deadband:
                            if self.name == "Angular":
                                self.integral += error * dt * 0.6  # Reduced accumulation rate
                            else:
                                self.integral += error * dt * 1.2  # Boost integral after crossing zero
                        else:
                            self.integral *= 0.5  # More aggressively reduce integral near zero
                    else:
                        # Normal integral handling
                        if abs(error) > self.integral_deadband:
                            if self.name == "Angular":
                                self.integral += error * dt * 0.7  # 30% slower accumulation
                            else:
                                self.integral += error * dt
                        else:
                            # More aggressively reduce integral term when close to target
                            self.integral *= self.integral_decay
                    
                    # Apply approach-specific integral adjustments for Linear X
                    if self.name == "Linear X" and hasattr(self, 'pid_controller'):
                        # Check if we have all required attributes
                        if (hasattr(self.pid_controller, 'filtered_distance') and 
                            hasattr(self.pid_controller, 'desired_distance') and 
                            hasattr(self.pid_controller, 'approach_distance')):
                            
                            distance_error = abs(self.pid_controller.filtered_distance - self.pid_controller.desired_distance)
                            approach_distance = self.pid_controller.approach_distance
                            
                            # Only apply when close to target
                            if distance_error < approach_distance:
                                # Calculate scaling factor (lower when closer)
                                proximity_factor = max(0.1, distance_error / approach_distance)
                                
                                # Apply stronger reduction to integral when close
                                self.integral *= proximity_factor
                                
                                # Additional aggressive reset when very close with significant integral
                                if abs(error) < 0.1 and abs(self.integral) > 0.1:
                                    self.integral *= 0.1
                
                # Apply controller-specific integral limits
                self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
                i_term = self.ki * self.integral
                
                # Calculate derivative term with enhanced handling for oscillations
                error_change = error - self.prev_error
                
                # Enhanced derivative handling for oscillations
                if self.sign_change_count >= 2:
                    # If oscillating (multiple sign changes), adjust derivative term
                    if self.name == "Angular":
                        d_term = self.kd * error_change / dt  # No amplification
                    else:
                        d_term = self.kd * error_change / dt * 1.3  # Amplify for non-angular
                else:
                    # Normal derivative calculation
                    d_term = self.kd * error_change / dt
                
                # Calculate raw output
                output = p_term + i_term + d_term
                
                # Apply output limits
                output_limited = max(self.output_min, min(self.output_max, output))
                
                # Apply anti-windup when error changes sign
                if error * self.prev_error < 0:
                    # Error crossed zero - reduce integral more aggressively
                    if abs(error) < abs(self.prev_error):
                        # Error is decreasing - be more aggressive
                        if self.name == "Angular":
                            self.integral *= 0.05  # Much more aggressive for angular
                        elif self.name == "Linear Y":
                            self.integral *= 0.1  # More aggressive for lateral
                        else:
                            self.integral *= 0.2  # Standard value
                    else:
                        # Error is increasing
                        if self.name == "Angular":
                            self.integral *= 0.1  # More aggressive than before for angular
                        elif self.name == "Linear Y":
                            self.integral *= 0.2
                        else:
                            self.integral *= 0.3
                
                # Apply transition smoothing for rapid control changes
                # Less smoothing for Angular controller to improve responsiveness
                smoothing_factor = 0.4 if self.name == "Angular" else 0.6
                
                if abs(output_limited - self.last_output) > self.output_range * 0.4:
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
                
                # Log PID terms if needed (reduced frequency)
                if debug_level >= 2 and should_log:
                    self.logger.info(f"PID {self.name} terms: P={p_term:.3f}, I={i_term:.3f}, D={d_term:.3f}")
                
                # After output limits
                if debug_level >= 1 and output != output_limited:
                    self.logger.info(
                        f"PID {self.name} output saturated: {output:.3f} limited to {output_limited:.3f}",
                        throttle_duration_sec=1.0
                    )
                
                # Log direction change (sign flip) at debug_level >= 1
                if debug_level >= 1 and error * self.prev_error < 0:
                    self.logger.info(
                        f"PID {self.name} direction change: error sign flip {self.prev_error:.3f} -> {error:.3f}",
                        throttle_duration_sec=1.0
                    )
                
                return float(output_limited)
            
            except Exception as e:
                self.logger.error(f"Error in PID compute: {str(e)}")
                return 0.0
        
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
        
        def get_components(self):
            """Get the last calculated PID components."""
            return (self.last_p_term, self.last_i_term, self.last_d_term)
        
        def get_current_gains(self):
            """Get the current adaptive gains."""
            return (self.kp, self.ki, self.kd)