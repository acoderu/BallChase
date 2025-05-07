"""
Basketball Tracking Robot - Advanced PID Control System
======================================================

EDUCATIONAL DOCUMENTATION
------------------------

This module implements a sophisticated PID control system specifically designed
for controlling a basketball-tracking robot with mecanum wheels. It goes far beyond
a basic PID controller, incorporating numerous advanced control techniques that
make the robot's movement smooth, efficient, and natural-looking.

┌───────────────────────────────────────────────────────────────────────────┐
│                   ADVANCED PID CONTROL ARCHITECTURE                       │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  ImprovedPID     │      │  StrategyManager │      │  Coordinated     │
│  Controllers     │      │  & Blender       │      │  Controller      │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                          │                          │
         │                          │                          │
         ▼                          ▼                          ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ • Adaptive gains │      │ • Chooses best   │      │ • Links lateral  │
│ • Anti-windup    │      │   movement       │      │   and angular    │
│ • Zero-crossing  │      │   strategy based │      │   movements      │
│   protection     │      │   on error       │      │ • Creates smooth │
│ • Trend analysis │      │   patterns       │      │   natural paths  │
│ • Dampening      │      │ • Smooth blending│      │ • Reduces        │
│                  │      │   between        │      │   conflicts      │
└──────────────────┘      └──────────────────┘      └──────────────────┘

Key Concepts for Beginners:
--------------------------

1. BEYOND BASIC PID CONTROL

   While standard PID control provides a foundation, this system adds several
   layers of sophistication:
   
   ┌───────────────────────────────────────────────────────────────────────┐
   │                          PID CONTROLLER EVOLUTION                     │
   └───────────────────────────────────────────────────────────────────────┘
    
    BASIC PID:                     ADVANCED PID:
    ┌─────────────────┐            ┌─────────────────────────────┐
    │                 │            │     Dynamic Gain Adjustment │
    │     Fixed       │            │ ┌─────┐      ┌─────┐        │
    │     Gains       │            │ │ Kp  │◄─┐   │ Dt  │◄───┐   │
    │                 │            │ └─────┘  │   └─────┘    │   │
    │ ┌───┐ ┌───┐ ┌───┐            │ ┌─────┐  │   ┌─────┐   │   │
    │ │ P │ │ I │ │ D │            │ │ Ki  │◄─┼───┤Error│───┘   │
    │ └─┬─┘ └─┬─┘ └─┬─┘            │ └─────┘  │   └─────┘       │
    │   │     │     │              │ ┌─────┐  │   ┌─────┐       │
    │   └─────┼─────┘              │ │ Kd  │◄─┴───┤Trend│       │
    │         │                    │ └─────┘      └─────┘       │
    │         ▼                    │                           │
    │      Output                  │    Anti-Windup Protection  │
    └─────────────────┘            └─────────────────────────────┘
   
   - ADAPTIVE GAIN ADJUSTMENT: PID gains change automatically based on the robot's situation
   - COORDINATED MULTI-DIMENSIONAL CONTROL: Handles forward, lateral, and angular movement together
   - MOVEMENT STRATEGIES: Different movement patterns for different situations
   - SMOOTH TRANSITIONS: Blends between strategies to prevent jerky movement
   - ANTI-WINDUP MECHANISMS: Prevents integral term from causing oscillations
   
   These enhancements transform basic PID control into a sophisticated 
   robotics control system.

2. MOVEMENT STRATEGIES AND THE STRATEGY PATTERN

   The robot uses different movement "strategies" based on the specific situation:
   
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                          STRATEGY PATTERN                                 │
   └───────────────────────────────────────────────────────────────────────────┘
   
   Error Condition                Strategy Selected              Movement Priority
   ┌────────────────┐            ┌────────────────┐            ┌────────────────┐
   │                │            │                │            │                │
   │ Large Angular  │────┐       │  ANGULAR_      │            │  90% Angular   │
   │ Error          │    │       │  FIRST         │────┐       │  40% Forward   │
   │                │    │       │                │    │       │  30% Lateral   │
   └────────────────┘    │       └────────────────┘    │       │                │
                         │                              │       └────────────────┘
   ┌────────────────┐    │       ┌────────────────┐    │       ┌────────────────┐
   │                │    │       │                │    │       │                │
   │ Large Lateral  │────┼─────▶│  LATERAL_      │────┼─────▶│  20% Angular   │
   │ Medium Distance│    │       │  PRIORITY      │    │       │  70% Forward   │
   │                │    │       │                │    │       │ 100% Lateral   │
   └────────────────┘    │       └────────────────┘    │       │                │
                         │                              │       └────────────────┘
   ┌────────────────┐    │       ┌────────────────┐    │       ┌────────────────┐
   │                │    │       │                │    │       │                │
   │ Ball Moving    │────┘       │  PREDICTIVE_   │────┘       │  50% Angular   │
   │ Diagonally     │            │  DIAGONAL      │            │  80% Forward   │
   │                │            │                │            │  80% Lateral   │
   └────────────────┘            └────────────────┘            └────────────────┘
   
   - Each strategy defines how to use forward, lateral, and angular movement
   - Strategies focus movement in different directions based on the current error pattern
   - The robot smoothly transitions between strategies using blending
   
   This approach comes from the "Strategy Pattern" in software design - switching
   algorithms at runtime based on conditions.

3. ERROR CATEGORIZATION WITH HYSTERESIS

   The system translates numerical errors into meaningful categories:
   
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                    ERROR CATEGORIZATION WITH HYSTERESIS                   │
   └───────────────────────────────────────────────────────────────────────────┘
   
           ┌─────────────────────────────────────────────────────────┐
   Error   │                                                         │
   Value   │                                                         │
           │                                                         │
     0.8   │                              ┌──────┐                   │
           │                              │LARGE │                   │
     0.6   │                 ┌──────┐     └──────┘                   │
           │                 │MEDIUM│                                │
     0.4   │    ┌──────┐     └──────┘                                │
           │    │SMALL │                                             │
     0.2   │    └──────┘                                             │
           │                                                         │
     0.0   │                                                         │
           └─────────────────────────────────────────────────────────┘
               Error increasing →                                    
                                                                     
                                                                     
           ┌─────────────────────────────────────────────────────────┐
   Error   │                                                         │
   Value   │                                                         │
           │                                                         │
     0.8   │                                                         │
           │                              ┌──────┐                   │
     0.6   │                              │LARGE │                   │
           │                 ┌──────┐     └──────┘                   │
     0.4   │                 │MEDIUM│                                │
           │    ┌──────┐     └──────┘                                │
     0.2   │    │SMALL │                                             │
           │    └──────┘                                             │
     0.0   │                                                         │
           └─────────────────────────────────────────────────────────┘
               ← Error decreasing                                    
   
   Note the HYSTERESIS effect: When error is increasing, the categories change
   at different thresholds than when error is decreasing. This "sticky" behavior
   prevents rapid oscillation between categories when error is near a threshold.
   
   - "none", "very_small", "small", "medium", "large", etc.
   - These categories are more intuitive for decision-making
   - Hysteresis ("stickiness") prevents rapid oscillation between categories
   
   This categorical approach allows robust decision-making without 
   being affected by minor sensor noise.

4. ZERO-CROSSING HANDLING

   Special handling occurs when errors change sign (cross zero):
   
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                          ZERO-CROSSING HANDLING                           │
   └───────────────────────────────────────────────────────────────────────────┘
   
   Error      │                 Zero-Crossing                                    
   Value      │                 Detection                                        
              │                     │                                            
       +      │      ╱              │               ╲                            
              │     ╱               │                ╲                           
       0 ─────┼────╱────────────────┼─────────────────╲────────                 
              │   ╱                 │                  ╲                         
       -      │  ╱                  │                   ╲                        
              │                     │                                            
              └─────────────────────┼─────────────────────────────────────      
                                    │                                            
                         ┌──────────▼──────────┐                                 
                         │ When Zero-Crossing  │                                 
                         │     Detected:       │                                 
                         └──────────┬──────────┘                                 
                                    │                                            
             ┌───────────────────────────────────────────────┐                   
             │                                               │                   
             ▼                                               ▼                   
    ┌─────────────────┐                             ┌─────────────────┐          
    │ Reset Integral  │                             │  Adjust Gains   │          
    │     Term        │                             │                 │          
    └─────────────────┘                             └─────────────────┘          
                                                                                
    Prevents accumulated                            Increases damping            
    integral value from                             for smoother                 
    causing overshoot                               approach to target           
   
   - The robot detects when it passes its target and adjusts control accordingly
   - Integral terms are reduced to prevent overshooting
   - Gains are adjusted to provide optimal damping
   
   This prevents the common problem of oscillating around the target point.

5. COORDINATED MOVEMENT

   The robot coordinates its movements across different dimensions:
   
   ┌───────────────────────────────────────────────────────────────────────────┐
   │                          COORDINATED MOVEMENT                             │
   └───────────────────────────────────────────────────────────────────────────┘
   
    BASIC APPROACH:                          COORDINATED APPROACH:
    
    ┌───────────────────────┐                ┌───────────────────────┐
    │                       │                │                       │
    │   Forward ──────────┐ │                │                       │
    │                     │ │                │   ┌─────────────────┐ │
    │   Lateral ──────────┼─┼─▶ Outputs      │   │   Coordinator   │ │
    │                     │ │                │   │                 │ │
    │   Angular ──────────┘ │                │   │  ┌─┐ ┌─┐   ┌─┐ │ │
    │                       │                │   │  │F│ │L├───┤A│ │ │
    └───────────────────────┘                │   │  └─┘ └─┘   └─┘ │ │
                                             │   │                 │ │
    Each dimension controlled                │   └─────────┬───────┘ │
    independently - can create               │             │         │
    jerky, robotic movement                  │             ▼         │
                                             │                       │
                                             │           Outputs     │
                                             └───────────────────────┘
                                             
                                             Dimensions influence each other
                                             creating smoother, more natural
                                             movement patterns
   
   - Lateral and angular movements are coordinated together
   - When turning, forward speed is appropriately reduced
   - Mutually beneficial movements are enhanced, conflicts are reduced
   
   This makes the robot move in a more natural, human-like manner.

Component Architecture:
---------------------

This module contains the following key components:

1. ImprovedPID: Enhanced PID controller with adaptive gains and anti-windup
2. StrategyManager: Selects movement strategies based on error patterns
3. MovementStrategy: Defines a specific movement pattern
4. StrategyBlender: Smoothly transitions between different strategies
5. CoordinatedController: Coordinates related movements to create natural motion

Together, these components create a control system that makes the robot move
efficiently and naturally, while being robust to real-world conditions like
sensor noise and mechanical limitations.
"""

import time
import math
import logging
import numpy as np
from enum import Enum, auto

from ball_chase.pid.pid_target_filter import ErrorTracker

class PIDControllers:
    """
    Collection of advanced PID control systems for the basketball tracking robot.
    
    Think of this class as a toolbox containing all the different control tools
    needed for making the robot move smoothly and efficiently. This includes:
    
    - PID controllers with advanced features
    - Movement strategy selection system
    - Coordinated movement patterns
    - Smooth transitions between behaviors
    
    What makes this different from basic PID controllers is the combination of
    multiple specialized components working together to create motion that appears
    natural and intelligent, rather than robotic and mechanical.
    """
    
    @staticmethod
    def create_controller_with_tracker(controller_type, kp, ki, kd, output_min, output_max, tracker_name, throttled_logger, max_history=8):
        """
        Creates a complete PID controller with an attached error tracker.
        
        Think of this like setting up a car with both an engine (the controller) 
        and dashboard instruments (the error tracker) at the same time.
        
        Args:
            controller_type: The kind of PID controller to create (BasicPID, ImprovedPID, etc.)
            kp: Proportional gain - how strongly to react to current error
            ki: Integral gain - how strongly to react to accumulated error over time
            kd: Derivative gain - how strongly to react to rate of change of error
            output_min: Minimum allowable output value (prevents commands that are too small)
            output_max: Maximum allowable output value (prevents commands that are too large)
            tracker_name: Name for the error tracker (for logging and debugging)
            throttled_logger: Logger that prevents too many repeated messages
            max_history: How many past error values to remember
            
        Returns:
            A tuple containing (controller, error_tracker)
            
        Raises:
            RuntimeError: If the error tracker fails to initialize properly
        """
        # Create the error tracker that will monitor error patterns
        error_tracker = ErrorTracker(tracker_name, throttled_logger, max_history=max_history)
        
        # Create the actual PID controller
        controller = PIDControllers.create_controller(
            controller_type, kp, ki, kd, output_min, output_max
        )
        
        # Attach the error tracker to the controller
        controller.error_tracker = error_tracker
        controller.logger = throttled_logger  
        
        # Make sure everything was set up correctly
        if not hasattr(controller, 'error_tracker') or controller.error_tracker is None:
            raise RuntimeError(f"Failed to initialize error tracker for {controller_type.name} controller")
            
        return controller, error_tracker

    class StrategyManager:
        """
        Manages movement strategies for robotic control systems.
        
        MOVEMENT STRATEGY SELECTION: HOW THE ROBOT DECIDES HOW TO MOVE
        -----------------------------------------------------------
        
        Imagine you're chasing a ball. How you move depends on where the ball is:
        - If it's far ahead, you run straight toward it
        - If it's to your side, you move diagonally
        - If it's behind you, you turn around first, then move
        
        The StrategyManager does this decision-making for the robot. It:
        1. Takes the current errors (distance, lateral, angular)
        2. Categorizes each error (none, very_small, small, medium, large, etc.)
        3. Looks up the appropriate strategy in a table
        4. Returns movement priorities and scaling factors
        
        This creates natural, human-like movement patterns that efficiently
        track the target while avoiding jerky or mechanical-looking motion.
        
        HOW THE STRATEGY TABLE WORKS:
        ---------------------------
        
        The strategy table is like a big decision matrix that maps error patterns
        to movement behaviors. For example:
        
        Error Pattern                 →  Strategy Name        →  Movement Behavior
        ----------------------------------------------------------------
        (large distance,  *,  *)      →  DISTANCE_PRIORITY    →  Focus on moving forward
        (*,  large lateral,  *)       →  LATERAL_PRIORITY     →  Focus on sideways alignment
        (*,  *,  large angular)       →  ANGULAR_PRIMARY      →  Turn to face target first
        (medium, medium, small)       →  DIAGONAL_MOVEMENT    →  Move forward+sideways together
        (none, medium, none)          →  LATERAL_CORRECTION   →  Pure sideways movement
        
        The * is a wildcard that matches any error category. This allows us to
        prioritize the most important errors.
        
        Each strategy defines:
        - Which dimensions to use (forward, lateral, angular)
        - Scaling factors for each dimension (0.0-1.0)
        - A human-readable description of the strategy
        
        For example, the ANGULAR_PRIMARY strategy might use:
        - Use forward motion? True (but scaled down)
        - Use lateral motion? True (but scaled down)
        - Use angular motion? True (at high priority)
        - Scaling: [0.4, 0.3, 0.9] - prioritizing rotation
        
        This means "turn to face the target, while moving slowly toward it."
        
        BENEFITS OF THIS APPROACH:
        -----------------------
        
        1. NATURAL MOVEMENT: Creates smooth, human-like movement patterns
        2. ADAPTABILITY: Easily adjusted for different robots or tasks
        3. EXPLAINABILITY: Clear reasoning behind movement decisions
        4. EFFICIENCY: Optimized paths to target
        5. CONFIGURABILITY: Easy to tune without changing code
        
        This class centralizes the strategy table, error categorization, and
        strategy matching logic to provide a consistent interface for determining
        the appropriate movement strategy based on current error conditions.
        """
        
        def __init__(self, throttled_logger):
            """
            Initialize the strategy manager.
            
            Args:
                logger: Logger instance for diagnostic output
            """
            # Setup logging
            self.logger = throttled_logger
            
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
            Turns numbers into words that describe how big the error is.
            
            IMAGINE THIS: ✨
            ---------------
            You're playing a game where you need to throw a ball into a basket.
            Instead of saying "you missed by 0.5 meters" or "you missed by 15 degrees",
            this method would tell you "you missed by a MEDIUM amount" or "you missed
            by a LARGE amount" - it translates exact measurements into useful categories
            that help decide how to adjust.
            
            TRANSLATING NUMERICAL ERRORS INTO MEANINGFUL CATEGORIES
            -----------------------------------------------------
            
            This method takes raw error values (like "0.5 meters too far" or "15 degrees off-angle")
            and translates them into simple words that are more intuitive and useful for
            decision-making:
            
              Error Categories:
              - "none": Perfect! No correction needed (like "bulls-eye!")
              - "very_small": Just a tiny bit off (like "almost perfect!")
              - "small": A little bit off (like "close but needs a small adjustment")
              - "small_medium": Somewhat off (like "getting there but needs work")
              - "medium": Noticeably off (like "definitely needs correction")
              - "medium_large": Quite far off (like "needs significant correction")
              - "large": Very far off (like "way off target")
              - "very_large": Extremely far off (like "completely missed")
            
            REAL-WORLD EXAMPLES:
            -----------------
            For a distance error (in meters):
              - 0.05m error → "none" (you're close enough - like parking within 5cm of the curb)
              - 0.15m error → "very_small" (like being just one step away from where you should stand)
              - 0.25m error → "small" (like missing a basketball shot by a small amount)
              - 0.5m error → "medium" (like throwing a dart that landed in the outer ring)
              - 1.2m error → "large" (like throwing a paper into a trash can from across the room)
            
            Each error type (distance, lateral, angular) has different thresholds that make
            sense for that measurement - just like how being off by 5cm when parking is minor,
            but being off by 5 degrees when flying an airplane could be a big deal!
            
            WHAT IS HYSTERESIS AND WHY WE NEED IT
            -----------------------------------
            
            Hysteresis is like "stickiness" that prevents flip-flopping between categories.
            It's easiest to understand with everyday examples:
            
            🌡️ THERMOSTAT EXAMPLE:
            A good thermostat doesn't turn on exactly at 70°F and off at 70°F.
            Instead, it might turn on at 68°F and off at 72°F. This 4-degree gap
            prevents the heater from rapidly turning on and off when the temperature
            hovers around 70°F.
            
            🚘 CRUISE CONTROL EXAMPLE:
            When you set cruise control to 65 mph, your car doesn't constantly
            adjust speed if you go 64.9, then 65.1, then 65.0 mph. Instead, it
            waits until you're maybe 2 mph off before making adjustments.
            
            WITHOUT hysteresis (bad):
              - Error changes from 0.14m to 0.16m
              - Category immediately changes from "none" to "very_small"
              - The robot keeps flip-flopping between strategies
              - Results in jittery, unstable movement (like a nervous driver)
            
            WITH hysteresis (good):
              - Error must increase to 0.16m to go from "none" to "very_small"
              - But must decrease to 0.12m (not just 0.14m) to go back to "none"
              - This 20% buffer creates "stickiness" at the boundaries
              - Results in smooth, stable movement (like a confident driver)
            
            HOW THIS MAKES THE ROBOT MOVE BETTER
            ----------------------------------
            
            Think of these categories like "driving modes":
            - When errors are "none" → the robot is in "I arrived!" mode
            - When errors are "small" → the robot is in "careful adjustment" mode
            - When errors are "large" → the robot is in "hurry to the target" mode
            
            Without hysteresis, the robot might rapidly switch between these modes
            when near the boundaries, making it seem indecisive and jittery - like
            a driver who can't decide whether to speed up or slow down.
            
            With hysteresis, the robot sticks with a mode until there's a meaningful
            change in the error, creating smoother, more natural-looking movement.
            
            Args:
                error: How far off we are (in meters or degrees)
                error_type: What kind of error ("distance", "lateral", or "angular")
                prev_category: What category we were in last time (for hysteresis)
                lenient_factor: How forgiving to be with categorization (higher = more forgiving)
                
            Returns:
                A word describing how big the error is ("none", "very_small", "small", etc.)
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
            Finds the right movement strategy for the current situation.
            
            IMAGINE THIS: 📚
            ---------------
            Think of this like looking up a recipe in a cookbook:
            
            1. You check if you have ingredients like: 
               - How far away is the target? (distance error)
               - How far to the side is the target? (lateral error)
               - How much do I need to turn? (angular error)
               
            2. You use these "ingredients" to find a matching recipe (strategy)
               for how the robot should move.
               
            WHAT MAKES THIS SMART: 🔍
            ---------------------
            This method can handle "wildcards" - which means if it can't find an
            exact match for the current situation, it will find the closest match:
            
            Example:
            - Exact match: "medium distance error, small lateral error, large angular error"
            - Wildcard match: "medium distance error, * lateral error, large angular error"
              (where * means "any value")
            
            This is like having a recipe that says "use any vegetable you have on hand"
            instead of requiring a specific vegetable.
            
            Args:
                key: The current error situation (distance, lateral, and angular categories)
                strategies: Optional special strategy table (usually you don't need this)
                
            Returns:
                The best movement strategy for the current situation
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
            Decides the best way for the robot to move right now.
            
            IMAGINE THIS: 🧠
            ---------------
            This is like the robot's "brain" that decides how to drive. It works like this:
            
            1. ASSESSMENT: "Where am I compared to where I should be?"
               - How far away am I? (distance_error)
               - How far to the side am I? (lateral_error) 
               - Am I facing the wrong direction? (angular_error_degrees)
               
            2. CATEGORIZATION: "How big are these errors?"
               - Turns numerical errors into categories like "small", "medium", "large"
               - Uses hysteresis to prevent flip-flopping between categories
               
            3. STRATEGY SELECTION: "What's the best way to move?"
               - Looks up the best movement pattern for the current situation
               - Decides which dimensions to move in and how strongly
               - Returns a complete movement plan
               
            SPECIAL FEATURES: 🌟
            ----------------
            - STARTUP BOOST: When first moving, prioritizes forward movement
              (like how a car needs extra gas to start moving from a standstill)
              
            - AT-TARGET BEHAVIOR: When at the right distance, cares less about
              perfect angular alignment (like being more relaxed with parking
              angle when you're already in the spot)
              
            - SMOOTH TRANSITIONS: Uses "strategy blending" to gradually transition
              between movement patterns (like how a good driver doesn't jerk the
              steering wheel but makes smooth adjustments)
            
            Args:
                distance_error: How far from target (in meters, + = too far, - = too close)
                lateral_error: How far to the side (in meters, + = too right, - = too left)
                angular_error_degrees: How much rotation needed (in degrees)
                is_robot_stopped: Whether the robot is currently stopped
                
            Returns:
                A complete movement strategy object with instructions for how to move
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
            Builds a complete movement strategy from a simple definition.
            
            IMAGINE THIS: 🏗️
            ---------------
            This is like assembling a toy from a box of parts:
            
            1. You have a list of parts (the strategy_def)
            2. You put them together into a complete, working toy (MovementStrategy)
            
            The strategy definition contains all the information needed:
            - What to name the strategy ("FORWARD_APPROACH", "ANGULAR_CORRECTION", etc.)
            - Which dimensions to use (forward, lateral, angular)
            - How strongly to use each dimension (scale factors)
            - A reason message template explaining why this strategy was chosen
            
            This method adds the current error values to the reason template
            to create a complete, informative message.
            
            Args:
                strategy_def: The blueprint for the strategy (from the strategy table)
                distance_error: How far from target (for message formatting)
                lateral_error: How far to the side (for message formatting)
                angular_error: How much rotation needed (for message formatting)
                
            Returns:
                A complete, ready-to-use MovementStrategy object
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
            """
            Controls how much information the robot shares about its decisions.
            
            Think of this like adjusting how chatty the robot is:
            - Level 0: Silent (only critical errors)
            - Level 1: Basic info (strategy changes)
            - Level 2: Detailed info (all decisions and calculations)
            - Level 3: Super detailed (absolutely everything)
            
            Higher levels are helpful when learning or troubleshooting,
            but can slow things down in regular operation.
            """
            self.debug_level = level
        
        def set_distance_threshold(self, threshold):
            """
            Sets how close is "close enough" for distance to target.
            
            This is like setting the "parking tolerance" - how precisely
            do we need to reach our exact distance target?
            
            - Smaller values (e.g., 0.1m): Very precise positioning required
            - Larger values (e.g., 0.3m): More relaxed about exact position
            
            This affects when the robot considers itself "at target distance"
            and starts using special at-target movement strategies.
            """
            if threshold > 0:
                self.distance_threshold = threshold
        
        def set_lateral_control(self, enabled):
            """
            Turns sideways movement on or off.
            
            When enabled = True:
              Robot will use sideways (lateral) movement to position itself
              (This is what makes mecanum wheels special - regular wheels can't do this!)
              
            When enabled = False:
              Robot will only use forward/backward movement and rotation
              (Like a regular car that can't slide sideways)
              
            You might disable lateral movement on rough surfaces where
            mecanum wheels don't slide well, or for testing purposes.
            """
            self.use_lateral_control = enabled

    class CoordinatedController:
        """
        Synchronized controller that coordinates lateral and angular movements.
        
        EDUCATIONAL EXPLANATION:
        -----------------------
        Imagine driving a car - when you make a turn, you naturally slow down.
        This controller brings that same intuitive coordination to the robot.
        
        THE COORDINATION PROBLEM:
        -----------------------
        In a multi-dimensional robot control system, separate PID controllers
        operate independently for each dimension:
        
        - Forward controller: Controls forward/backward movement
        - Lateral controller: Controls left/right movement (unique to mecanum wheels)
        - Angular controller: Controls rotation
        
        When operating independently, these controllers can create inefficient
        or unnatural movements. For example:
        
        - Trying to move sideways at full speed while also turning sharply
        - Moving forward at full speed while trying to correct a lateral error
        - Performing big lateral movements when a small rotation would be better
        
        THE SOLUTION: MOVEMENT COORDINATION
        ---------------------------------
        This class connects the controllers and applies human-like coordination:
        
        1. ANGULAR PRIORITY
           - When large angular errors exist, rotation takes priority
           - Forward and lateral movements are reduced until facing the target
           - This mimics how humans first turn toward a target before approaching
        
        2. COUPLING FACTOR
           - Lateral speed is reduced based on the angular error
           - The reduction is proportional to the angular error's magnitude
           - This creates smooth, natural combined movements
        
        3. SIGN RELATIONSHIP
           - When errors have the same sign, they compete with each other
           - When errors have opposite signs, they help each other
           - Adjusts the movement accordingly to optimize the combined effect
            
        4. SMOOTH TRANSITIONS
           - Prevents abrupt changes in velocity
           - Creates fluid, continuous movement
           - Makes the robot's motion appear natural and purposeful
        
        This coordination makes the robot move in a way that appears intelligent
        and human-like, rather than mechanical and robotic.
        """
        
        def __init__(self, linear_pid, angular_pid, throttled_logger, config=None):
            """
            Initialize a multi-dimensional movement coordinator.
            
            Args:
                linear_pid: PID controller for lateral (side-to-side) movement
                angular_pid: PID controller for rotational movement
                throttled_logger: Logger instance with throttling support
                config: Optional configuration dictionary for customization
                
            The coordinator sets up coupling factors, thresholds, and scaling values
            that determine how lateral and angular movements interact. These values
            are carefully tuned to create natural, efficient movement patterns.
            """
            self.linear_pid = linear_pid
            self.angular_pid = angular_pid
            self.logger = throttled_logger
            
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
        """
        Defines a specific pattern of robot movement across multiple dimensions.
        
        EDUCATIONAL EXPLANATION:
        -----------------------
        A MovementStrategy encapsulates a specific "way of moving" for the robot.
        Think of it like different driving styles for different situations:
        
        - Highway driving: High forward speed, minimal steering
        - Parallel parking: Low speed, high lateral and angular movement
        - Three-point turn: Coordinated forward/reverse with steering
        
        Each strategy defines:
        
        1. WHICH DIMENSIONS TO USE
           - Should the robot move forward/backward?
           - Should the robot move laterally (sideways)?
           - Should the robot rotate?
           
        2. HOW STRONGLY TO USE EACH DIMENSION
           - Scale factors (0.0-1.0) for each dimension
           - Higher values mean stronger movement in that dimension
           - These proportions create different movement patterns
        
        3. REASONING
           - Human-readable explanation of why this strategy was chosen
           - Useful for debugging and understanding robot behavior
        
        EXAMPLE STRATEGIES:
        -----------------
        
        - "ANGULAR_PRIMARY": Strong rotation, minimal forward/lateral
          Used when the robot needs to turn to face the target
          
        - "DIAGONAL_MOVEMENT": Equal forward and lateral, minimal angular
          Used when approaching the target from an angle
          
        - "FORWARD_APPROACH": Strong forward, no lateral or angular
          Used when aligned with target and just needs to approach
          
        - "LATERAL_CORRECTION": No forward or angular, only lateral
          Used when at correct distance but needs side-to-side alignment
        
        The system has dozens of strategies for different situations, and
        automatically selects the appropriate one based on current errors.
        """
        
        def __init__(self, name, use_forward, use_lateral, use_angular, 
                    forward_scale, lateral_scale, angular_scale, reason):
            """
            Initialize a movement strategy with specific characteristics.
            
            Args:
                name: Strategy identifier (e.g., "ANGULAR_PRIMARY")
                use_forward: Whether to use forward/backward movement
                use_lateral: Whether to use lateral (side-to-side) movement
                use_angular: Whether to use rotational movement
                forward_scale: Scaling factor for forward movement (0.0-1.0)
                lateral_scale: Scaling factor for lateral movement (0.0-1.0)
                angular_scale: Scaling factor for angular movement (0.0-1.0)
                reason: Human-readable explanation for this strategy
                
            Each strategy creates a specific movement pattern by enabling or
            disabling dimensions and setting their relative strengths.
            """
            self.strategy_name = name
            self.use_forward = use_forward
            self.use_lateral = use_lateral
            self.use_angular = use_angular
            self.forward_scale = forward_scale
            self.lateral_scale = lateral_scale
            self.angular_scale = angular_scale
            self.reason = reason
    
    class StrategyBlender:
        """
        Creates smooth transitions between movement strategies for natural motion.
        
        EDUCATIONAL EXPLANATION:
        -----------------------
        Without blending, switching between movement strategies would cause
        abrupt, jerky changes in the robot's motion. The StrategyBlender 
        solves this by gradually transitioning between strategies over time.
        
        THE BLENDING PROCESS:
        -------------------
        When a new strategy is selected, the blender:
        
        1. Stores both the current and target strategies
        2. Records the time when the transition begins
        3. Over a short duration (typically 0.1-0.5 seconds):
           - Gradually decreases the influence of the old strategy
           - Gradually increases the influence of the new strategy
           - Uses smoothstep function for acceleration/deceleration
        4. Creates a temporary "blended" strategy for each time step
        
        ENHANCED FEATURES:
        ---------------
        This implementation includes additional refinements:
        
        - DIRECTION CHANGE DETECTION
          When movements change direction (e.g., left to right), 
          transitions occur faster to maintain responsiveness
          
        - SMOOTHSTEP FUNCTION
          Uses a cubic smoothstep function (3x² - 2x³) for natural
          acceleration and deceleration during transitions
          
        - BOOLEAN LOGIC HANDLING
          Special handling for boolean flags (use_forward, etc.)
          to prevent flickering during transitions
        
        VISUAL ANALOGY:
        -------------
        Think of this like cross-fading between two songs:
        - Current strategy volume decreases
        - Target strategy volume increases
        - Both play simultaneously during the transition
        - Listeners experience a smooth audio transition
        
        This creates fluid, natural-looking robot movement that
        transitions seamlessly between different movement patterns.
        """
        
        def __init__(self, logger, blend_duration=0.1):  # Reduced from 0.5 to 0.2 seconds
            """
            Initialize a strategy blending system for smooth transitions.
            
            Args:
                logger: Logger instance for diagnostic output
                blend_duration: Time in seconds for transition (default: 0.1s)
                
            The blend_duration determines how long transitions take. Shorter
            durations are more responsive but may appear more mechanical,
            while longer durations create smoother transitions but might
            feel less responsive. This value has been optimized for the
            basketball tracking application.
            """
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
        """
        Advanced PID controller with adaptive gains and enhanced stability features.
        
        EDUCATIONAL EXPLANATION:
        -----------------------
        A PID controller calculates output based on three terms:
        
        1. PROPORTIONAL (P) - Responds directly to current error
           - Acts like a spring pulling toward the target
           - Larger errors create stronger responses
           - Provides immediate responsiveness
        
        2. INTEGRAL (I) - Accumulates error over time
           - Eliminates persistent errors (steady-state errors)
           - Helps overcome friction and other constant forces
           - Acts like a gradually increasing push
        
        3. DERIVATIVE (D) - Responds to rate of change of error
           - Acts like a damper or shock absorber
           - Prevents overshooting by counteracting rapid changes
           - Adds stability to the system
        
        IMAGINE THIS: 🚗
        --------------
        Think of driving a car toward a parking spot:

        - P term is like how hard you press the gas pedal based on distance
          (farther away = press harder, closer = press lighter)
          
        - I term is like adjusting for a hill - if you're on an incline and not
          moving despite pressing the gas, you gradually press harder until the
          car starts moving
          
        - D term is like how you ease off the gas as you approach the spot
          (slowing down to avoid overshooting your target)

        STANDARD PID VS. IMPROVED PID:
        ----------------------------
        A standard PID controller always uses fixed gains (kp, ki, kd).
        This advanced implementation adds:
        
        - ADAPTIVE GAINS: Gains change based on the situation (like a driver
          who becomes more cautious when approaching their destination)
          
        - ANTI-WINDUP: Prevents integral term from growing too large (like 
          avoiding pressing the gas pedal to the floor when stuck in mud)
          
        - ZERO-CROSSING HANDLING: Special care when passing the target (like
          special attention when you've just driven past your parking spot)
          
        - GAIN SCHEDULING: Different gains for different controller types (like
          different driving styles for highways vs. city streets)
          
        - TREND-BASED ADAPTATION: Adjusts control based on whether error is improving
          (like easing off the gas when already slowing down at the right rate)
        
        WHY THESE IMPROVEMENTS MATTER: 🎯
        ----------------------------
        These enhancements solve several common PID problems:
        
        1. OSCILLATION: Standard PIDs often oscillate around the target
           (like a driver who keeps overshooting and backing up repeatedly)
           Our zero-crossing detection prevents this
        
        2. OVERSHOOTING: Standard PIDs often overshoot the target
           (like driving past your parking spot before backing up)
           Our adaptive gains reduce this
        
        3. SLOW RESPONSE: Fixed gains must compromise between stability and speed
           (like a driver who's either too cautious or too aggressive)
           Our adaptive approach can be both stable AND fast
        
        4. INTEGRATOR WINDUP: When the system can't reach the target, the 
           integral term grows without limit (like pressing the gas harder and
           harder when stuck against a wall)
           Our anti-windup mechanisms prevent this
        
        The end result is a controller that creates smooth, natural movement
        while adapting to different conditions - just like an experienced driver!
        """
        # NOTE: Initialization errors should be raised explicitly and not masked, for consistency with the node's error handling policy.
        
        def __init__(self, base_kp, base_ki, base_kd, output_min, output_max, name="PID", logger=None):
            """
            Initialize an advanced PID controller with adaptive capabilities.
            
            Args:
                base_kp: Base proportional gain coefficient
                base_ki: Base integral gain coefficient
                base_kd: Base derivative gain coefficient
                output_min: Minimum output value
                output_max: Maximum output value
                name: Controller name (Linear X, Linear Y, Angular)
                logger: Logger instance for diagnostic output
                
            The controller uses the base gain values as starting points, but
            will adaptively adjust these values during operation based on
            error trends and system behavior.
            """
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
            Compute optimized control output with adaptive gains and zero-crossing logic.
            
            EDUCATIONAL EXPLANATION:
            -----------------------
            This method implements the core PID computation with many advanced features.
            Understanding its operation requires following several key processes:
            
            IMAGINE THIS: 🧠
            --------------
            Think of this method as the "brain" of an experienced robot driver:
            
            1. ERROR TREND ANALYSIS 📈
               - Is the error getting better or worse?
               - Adjust gains accordingly (more aggressive when worsening)
               - Use gentle control when error is already improving
               
               Example: Like noticing "I'm already slowing down at the right rate, 
               so I don't need to press the brake harder"
            
            2. ZERO-CROSSING DETECTION 🎯
               - Detect when the error changes sign (crossed the target)
               - Apply special handling to prevent oscillation
               - Reduce integral term to prevent overshooting 
               
               Example: Like realizing "I just drove past my parking spot! Let me
               quickly adjust to avoid going too far in the wrong direction"
               
            3. ADAPTIVE GAIN ADJUSTMENT 🔄
               - For Linear X (forward): Prioritize smooth approach
               - For Linear Y (lateral): More aggressive damping
               - For Angular (rotation): Reduced integral gain
               - All controllers: Adjust gains based on error magnitude
               
               Example: Like naturally driving differently on highways (fast, stable)
               versus in parking lots (slow, precise)
               
            4. INTELLIGENT INTEGRAL HANDLING 📊
               - Apply deadband (ignore tiny errors)
               - Use position-based decay for final approach
               - Reset integral when crossing zero
               - Anti-windup when output saturates
               
               Example: Like not bothering to adjust your steering for a tiny
               deviation, but making corrections for larger ones
               
            5. ENHANCED DERIVATIVE HANDLING 🛑
               - Amplify derivative during oscillations
               - Apply controller-specific adjustments
               - Protect against division by zero
               
               Example: Like applying extra braking when you notice you're swinging
               back and forth between too fast and too slow
            
            6. OUTPUT SMOOTHING 🧈
               - Detect and smooth out abrupt changes
               - Apply different smoothing based on controller type
               - Prevent jerky movements
               
               Example: Like how you don't slam on brakes or gas pedal, but
               apply them gradually for passenger comfort
            
            DAILY LIFE EXAMPLE: 🚶‍♀️
            -------------------
            Think about how you walk through a crowded mall to reach a specific store:
            
            - P term: You walk faster when far away, slower when close
            - I term: If people keep blocking your path, you gradually find a more
              open route
            - D term: You slow down when approaching your destination
            - Zero-crossing: If you walk past the store, you quickly turn around
            - Adaptive gains: You move more carefully in crowded areas
            - Output smoothing: You don't make jerky movements as you navigate
            
            All these mechanisms work together to create a controller that produces
            smooth, natural movement with minimal oscillation - just like how a skilled
            human navigates complex environments!
            
            Args:
                error: Current error value (target - current)
                current_time: Current time (defaults to now)
                force_zero: Whether to force zero output (overrides PID)
                error_trend: Trend of error (-1 to 1, from ErrorTracker)
                
            Returns:
                float: Calculated control output, limited to output_min/max range
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