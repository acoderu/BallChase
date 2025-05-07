#!/usr/bin/env python3

"""
Ground Position Filter - Specialized filter for basketball tracking
=================================================================

This filter is optimized for tracking basketballs that move on the ground,
with the ability to handle rapid direction changes, varying speeds,
and the physical characteristics of a basketball (10-inch diameter).

EDUCATIONAL VISUALIZATION - HOW THE GROUND POSITION FILTER WORKS:
----------------------------------------------------------------

┌─────────────────────────────────────────────────────────────────────────────┐
│                     BASKETBALL TRACKING CHALLENGES                           │
└─────────────────────────────────────────────────────────────────────────────┘

    1. SENSOR NOISE                 2. RAPID DIRECTION CHANGES      3. VARYING SPEEDS
    
    ┌────────────────┐             ┌────────────────┐             ┌────────────────┐
    │                │             │                │             │                │
    │  ACTUAL PATH:  │             │  BALL CHANGES  │             │   BALL SPEED:  │
    │    ━━━━━━━     │             │   DIRECTION:   │             │                │
    │                │             │      ↑         │             │   SLOW → FAST  │
    │  WHAT SENSORS  │             │     ╱ ╲        │             │                │
    │     "SEE":     │             │    ╱   ╲       │             │    •••→•→•→•   │
    │   ~ ~ ~ ~ ~    │             │   •     •      │             │   •→•→•→•→•→•  │
    │                │             │                │             │                │
    └────────────────┘             └────────────────┘             └────────────────┘
    
      PROBLEM: Camera               PROBLEM: Ball can               PROBLEM: Speed 
      and sensors add               suddenly change                 can change from
      "noise" to readings           direction when                  very slow to very
                                    bouncing or rolling             fast quickly
      
      SOLUTION: Filter              SOLUTION: Adaptive              SOLUTION: Physics-
      smooths out random            filtering based on              based constraints
      fluctuations                  motion patterns                 & speed validation

The Ground Position Filter solves these problems by combining:
1. Adaptive filtering based on motion state
2. Jump detection to identify and correct for implausible position changes
3. Physics constraints (ground plane, maximum speeds)


┌─────────────────────────────────────────────────────────────────────────────┐
│                   ADAPTIVE FILTERING BASED ON MOTION STATE                   │
└─────────────────────────────────────────────────────────────────────────────┘

     HOW MUCH TO TRUST NEW SENSOR READINGS VS. HISTORY
     ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

 +----------------+------------------+-------------------+----------------------+
 |    HOW BALL    |  TRUST LEVEL    |     FILTERING     |       REAL-LIFE      |
 |   IS MOVING    |  (0.0 to 1.0)   |     TECHNIQUE     |        EXAMPLE       |
 +----------------+------------------+-------------------+----------------------+
 |                |                  |                   |                      |
 |    "MOVING"    |      0.70       |  Standard Filter  |   Basketball rolling |
 |                |     [HIGH]       |                   |    across the floor  |
 |                |                  |                   |                      |
 +----------------+------------------+-------------------+----------------------+
 |                |                  |                   |                      |
 |  "STATIONARY"  |      0.35       |  Stronger Filter  |   Basketball sitting |
 |                |    [MEDIUM]      |                   |    still briefly     |
 |                |                  |                   |                      |
 +----------------+------------------+-------------------+----------------------+
 |                |                  |                   |                      |
 |     "LONG_     |      0.21       |   Very Strong     |   Basketball placed  |
 |   STATIONARY"  |      [LOW]      |   Filter + Math   |    in a fixed spot   |
 |                |                  |                   |                      |
 +----------------+------------------+-------------------+----------------------+

     THINK OF THIS LIKE:                   HIGHER VALUES = More responsive but noisier
     🔊 Volume control on                   (Trust new readings a lot)
         sensitive equipment         
                                     0.7    MEDIUM VALUES = Balance between smooth
     0.0 ◁──────────○───────▷ 1.0   0.35   and responsive
      LOW            |       HIGH    0.21
         (Smooth)    |    (Responsive)     LOWER VALUES = Smoother but slower to respond
                                           (Trust history more than new readings)


┌─────────────────────────────────────────────────────────────────────────────┐
│                       JUMP DETECTION AND HANDLING                            │
└─────────────────────────────────────────────────────────────────────────────┘

                 WHAT ARE "JUMPS"?
    Sensor readings that show impossible movements,
    like the ball "teleporting" across the floor

    +-----------------+    +------------------+    +------------------+
    |                 |    |                  |    |                  |
    |  SENSOR SHOWS   |    |  FILTER DETECTS  |    |  FILTER FIXES   |
    |  ERROR (JUMP)   |    |     THE JUMP     |    |    THE PROBLEM  |
    |                 |    |                  |    |                  |
    +-----------------+    +------------------+    +------------------+
    |                 |    |                  |    |                  |
    |    Ball path:   |    |                  |    |   Smooth path:   |
    |                 |    |    ⚠️  TOO FAST! |    |                  |
    |      •---•      |    |      •---•       |    |      •---•       |
    |        ╲        |    |        ╲         |    |        ╲         |
    |         ╲       |    |         ╲        |    |         ╲        |
    |          •      |    |          •       |    |          •       |
    |         ╱       |    |    Impossible!   |    |         ╱        |
    |        ╱        |    |        ╱         |    |        ╱         |
    |    JUMP↓        |    |    Jump detected:|    |    Smooth curve  |
    |       •         |    |       •          |    |       •          |
    |                 |    |                  |    |                  |
    +-----------------+    +------------------+    +------------------+
                                      |
                                      ↓
            HOW THE FILTER DECIDES IF A MOVEMENT IS IMPOSSIBLE:
               
            1. Calculate speed between position readings
            2. Compare to maximum possible basketball speed (5 m/s)
            3. Compare to recent average speeds (history)
            4. If much faster than physically possible → It's a JUMP!
            5. Apply stronger filtering to ignore most of the bad reading


┌─────────────────────────────────────────────────────────────────────────────┐
│                       MOTION PREDICTION SYSTEM                               │
└─────────────────────────────────────────────────────────────────────────────┘

                 TIME:        [ PAST ]        [ NOW ]        [ FUTURE ]
                              ◄──────────────┐                  ┌────▶
                                             │                  │
                 THE PROBLEM: Robot sensors have delay, so by the time
                              we process data, the ball has ALREADY MOVED!

    +----------------------+     +----------------------+     +--------------------+
    |                      |     |                      |     |                    |
    |  🕒 WHERE WAS IT?    |     |  📍 WHERE IS IT NOW? |     |  🔮 WHERE WILL    |
    |  (POSITION HISTORY)  |     |  (CURRENT POSITION)  |     |    IT BE NEXT?    |
    |                      |     |                      |     |                    |
    +----------------------+     +----------------------+     +--------------------+
    |                      |     |                      |     |                    |
    |   We remember where  |     |   We know where      |     |  We predict where  |
    |   the ball was:      |     |   the ball is now    |     |  ball is going:    |
    |                      |     |   and current speed  |     |                    |
    |      •               |     |          •           |     |           •        |
    |       \              |     |         /|\          |     |          /         |
    |        \             |     |        / | \         |     |         /          |
    |         •            |     |            Speed     |     |        /           |
    |          \           |     |          Direction   |     |       /            |
    |           \          |     |                      |     |      /             |
    |            •         |     |                      |     |     •              |
    |                      |     |                      |     |                    |
    +----------------------+     +----------------------+     +--------------------+
                                           │
                                           ▼
      HOW PREDICTION WORKS: Position_future = Position_now + (Velocity × Time)

      IMAGINE THIS: It's like predicting where to catch a thrown ball. You don't aim
      for where the ball IS - you aim for where it WILL BE when it reaches you.
      
      IN ROBOTICS: This helps the robot "aim ahead" of the moving ball, compensating
      for the delay between sensing, processing, and mechanical movement.
"""

import numpy as np
import math
from collections import deque
import time


class GroundPositionFilter:
    """
    Filter for tracking ground-constrained basketball movement.
    Improved with motion state feedback and enhanced jump detection.
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    GROUND POSITION FILTER ARCHITECTURE                       │
    └─────────────────────────────────────────────────────────────────────────────┘
    
        HOW THE FILTER WORKS - STEP BY STEP PROCESSING PIPELINE
        
    
        ┌───────────────────────────────────────────────────────────────────────┐
        │                                                                       │
        │  📥 INPUT: POSITION READINGS (x, y, z) FROM SENSORS                   │
        │                                                                       │
        └───────────────────────────────────┬───────────────────────────────────┘
                                            │
                                            ▼
        ┌───────────────────────────────────────────────────────────────────────┐
        │                                                                       │
        │  🚶‍♂️ STEP 1: MOTION STATE DETECTION                                 │
        │                                                                       │
        │    Is the ball:                                                       │
        │    • Moving? (rolling across floor)                                   │
        │    • Stationary? (sitting still briefly)                              │
        │    • Long_stationary? (placed in a fixed position)                    │
        │                                                                       │
        └───────────────────────────────────┬───────────────────────────────────┘
                                            │
                                            ▼
        ┌───────────────────────────────────────────────────────────────────────┐
        │                                                                       │
        │  ⚖️ STEP 2: ADAPTIVE FILTERING                                       │
        │                                                                       │
        │    Apply appropriate filtering based on motion state:                 │
        │    • Moving? → Trust 70% of new data                                  │
        │    • Stationary? → Trust 35% of new data                              │
        │    • Long_stationary? → Trust 21% of new data                         │
        │                                                                       │
        └───────────────────────────────────┬───────────────────────────────────┘
                                            │
                                            ▼
        ┌───────────────────────────────────────────────────────────────────────┐
        │                                                                       │
        │  🔍 STEP 3: JUMP DETECTION                                           │
        │                                                                       │
        │    Did the ball move impossibly fast?                                 │
        │    • Calculate speed                                                  │
        │    • Compare with maximum possible speed                              │
        │    • If suspicious → Apply stronger filtering                         │
        │                                                                       │
        └───────────────────────────────────┬───────────────────────────────────┘
                                            │
                                            ▼
        ┌───────────────────────────────────────────────────────────────────────┐
        │                                                                       │
        │  🧲 STEP 4: GROUND PLANE ENFORCEMENT                                 │
        │                                                                       │
        │    Keep the ball on the floor!                                        │
        │    • Force the z-coordinate to ground level                           │
        │    • Add basketball radius (5 inches / 0.127 meters)                  │
        │                                                                       │
        └───────────────────────────────────┬───────────────────────────────────┘
                                            │
                                            ▼
        ┌───────────────────────────────────────────────────────────────────────┐
        │                                                                       │
        │  ➡️ STEP 5: VELOCITY CALCULATION                                     │
        │                                                                       │
        │    How fast is the ball moving and in what direction?                 │
        │    • Calculate speed and direction                                    │
        │    • Store in history                                                 │
        │    • Apply smoothing to velocity                                      │
        │                                                                       │
        └───────────────────────────────────┬───────────────────────────────────┘
                                            │
                                            ▼
        ┌───────────────────────────────────────────────────────────────────────┐
        │                                                                       │
        │  📤 OUTPUT: FILTERED POSITION AND VELOCITY                            │
        │                                                                       │
        │    Cleaned, smoothed, and physically realistic basketball data        │
        │    for the robot to use in tracking and prediction                    │
        │                                                                       │
        └───────────────────────────────────────────────────────────────────────┘
    
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                   EXPONENTIAL MOVING AVERAGE CONCEPT                         │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    WHAT IS FILTERING?
    Think of it like noise-canceling headphones for data - removing unwanted "noise"
    while keeping the important "signal" (the actual ball movement)
    
    ┌──────────────────────────────┐     ┌──────────────────────────────┐
    │                              │     │                              │
    │   Raw Noisy Measurements     │     │      Filtered Output         │
    │   (What sensors report)      │     │   (What we actually use)     │
    │                              │     │                              │
    ├──────────────────────────────┤     ├──────────────────────────────┤
    │                              │     │                              │
    │     *      *                 │     │                              │
    │    * *    * *   *            │     │                              │
    │   *   *  *   * * *           │     │       Smooth curve           │
    │  *     **     *   *          │     │      _,-"~^~"-,_             │
    │ *      *       Noisy!        │     │ _,-"'           '~-,_        │
    │                              │     │                              │
    └──────────────────────────────┘     └──────────────────────────────┘
                    │                                   ▲
                    │                                   │
                    └─────────────► FILTER ────────────┘
    
    ┌──────────────────────────────────────────────────────────────────────────────┐
    │                                                                              │
    │  HOW THE EXPONENTIAL MOVING AVERAGE FILTER WORKS                             │
    │                                                                              │
    │  1. FORMULA:                                                                 │
    │     P_filtered = P_previous * (1-alpha) + P_new * alpha                      │
    │                                                                              │
    │     WHERE:                                                                   │
    │     • P_filtered = The new filtered position                                 │
    │     • P_previous = The last filtered position                                │
    │     • P_new = The new raw sensor reading                                     │
    │     • alpha = How much to trust new data (0.0-1.0)                           │
    │                                                                              │
    │  2. REAL-WORLD ANALOGY:                                                      │
    │     Imagine you're estimating the temperature outside. You have:             │
    │     • Your previous estimate: "It's about 70°F"                              │
    │     • A new reading: Thermometer shows 80°F                                  │
    │                                                                              │
    │     If alpha = 0.3 (low trust in new reading):                               │
    │     Temperature = 70 * 0.7 + 80 * 0.3 = 73°F                                 │
    │     (You mostly trust your previous estimate, adjusting it slightly)         │
    │                                                                              │
    │     If alpha = 0.7 (high trust in new reading):                              │
    │     Temperature = 70 * 0.3 + 80 * 0.7 = 77°F                                 │
    │     (You mostly trust the new reading, discounting your previous estimate)   │
    │                                                                              │
    └──────────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config=None):
        """
        Initialize the ground position filter.
        
        IMAGINE THIS: 🛠️
        ---------------
        Think of this method like setting up specialized equipment before a basketball game.
        The referee needs to check the ball's pressure, ensure the court dimensions are correct,
        and calibrate the shot clock - all before the game starts.
        
        Similarly, this method prepares all the components needed for our basketball tracking
        filter, configuring key parameters based on physical properties and desired behavior.
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. CONFIGURATION LOADING ("Setting up the equipment")
           - Accept external configuration or use defaults
           - This allows customization for different environments
           
        2. PHYSICAL PARAMETER SETUP ("Understanding the basketball")
           - max_speed: How fast can a basketball realistically move?
           - basketball_radius: How big is the basketball?
           - ground_plane_z: Where is the floor relative to our coordinate system?
           
        3. FILTER PARAMETER SETUP ("Tuning the detector")
           - position_filter_alpha: How much to trust new measurements vs. history
           - This is like adjusting the sensitivity of our tracking system
           
        EVERYDAY ANALOGY:
        ---------------
        It's like setting up a new smartphone - you first configure core settings like
        display brightness, notification sensitivity, and speaker volume. These settings 
        affect how the phone behaves in different environments. Our filter similarly
        needs basic configuration that matches the physical world it's modeling.
        
        The filter's "alpha" parameter (0.0-1.0) is particularly important - it's like
        a "trust dial" for new information:
          - Higher values (closer to 1.0): "Trust new measurements more" (responsive but noisy)
          - Lower values (closer to 0.0): "Trust history more" (smooth but laggy)
        """
        self.config = config or {}
        
        # Basketball parameters
        self.max_speed = self.config.get("max_speed", 5.0)
        self.position_filter_alpha = self.config.get("position_filter_alpha", 0.7)
        self.ground_plane_z = self.config.get("ground_plane_z", 0.127)  # Default basketball radius
        self.basketball_radius = self.config.get("basketball_radius", 0.127)
        
        # Current state
        self.current_position = None
        self.filtered_position = None
        self.current_velocity = [0.0, 0.0, 0.0]
        self.last_update_time = None
        
        # History for velocity calculation
        self.position_history = []
        self.time_history = []
        
        # Increase history length for smoother filtering
        self.history_max_length = 10  # Increased from 5 to 10 for better smoothing
        
        # Statistics
        self.position_jumps = 0
        self.jump_threshold = 0.1  # Minimum jump distance to count
        self.total_distance = 0.0
        self.max_observed_speed = 0.0
        
        # NEW: Add stationary history for state-specific filtering
        self.stationary_positions = []
        self.stationary_max_length = 30  # Longer history for stationary objects
        
        # NEW: Add motion state information
        self.motion_state = "unknown"
        self.motion_state_count = {"stationary": 0, "long_stationary": 0, "moving": 0}
        
        # NEW: Velocity validation system
        self.velocity_history = []
        self.velocity_history_max_length = 10
        self.velocity_confidence = 1.0  # 0.0-1.0 scale
        
    def reset(self):
        """
        Reset the filter state.
        
        IMAGINE THIS: 🔄
        ---------------
        Think of this like wiping the scoreboard clean before a new basketball game. 
        All the statistics, player positions, and game history from the previous game
        are cleared so we can start fresh with a clean slate.
        
        This method performs a similar function - it clears all the accumulated history,
        calculations, and state information so the filter can start tracking a new object
        or recover from a tracking failure.
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. POSITION RESET ("Forget where we were")
           - Clear current and filtered positions
           - Reset velocity to zero in all directions
           - Forget last update time
           
        2. HISTORY RESET ("Erase the memory")
           - Clear position history array
           - Clear time history array
           - Reset stationary position history
           - Reset velocity history
           
        3. STATISTICS RESET ("Zero the counters")
           - Reset position jump counter
           - Reset total distance traveled
           - Reset maximum observed speed
           
        EVERYDAY ANALOGY:
        ---------------
        It's like restarting your GPS navigation when you've taken a wrong turn.
        Sometimes the best solution is to clear all routing information and start
        fresh with your current position rather than trying to recover from a
        confused state.
        
        In robotics, the reset function is crucial when:
        1. The tracked object disappears and reappears
        2. The filter gets into an unstable state
        3. The system detects tracking has failed
        4. A completely new tracking session begins
        """
        self.current_position = None
        self.filtered_position = None
        self.current_velocity = [0.0, 0.0, 0.0]
        self.last_update_time = None
        self.position_history = []
        self.time_history = []
        self.position_jumps = 0
        self.total_distance = 0.0
        self.max_observed_speed = 0.0
        self.stationary_positions = []
        self.velocity_history = []
    
    def update(self, position, timestamp, motion_state=None):
        """
        Update the filter with a new position measurement.
        
        IMAGINE THIS: 🔮
        ---------------
        Think of this method like a basketball coach watching a game through a foggy window.
        The coach sees the player's position, but it's not always clear - sometimes there's
        a blur or the coach blinks and misses something. The coach has to mentally filter out
        what's real movement vs. just noise or mistakes in perception.
        
        When a stationary player is standing still, the coach knows they shouldn't appear to
        be "teleporting" around the court - that would be the foggy window playing tricks.
        When a player is moving fast, the coach can accept more dramatic position changes.
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. MOTION STATE TRACKING ("How is the ball moving?")
           - We keep track if the ball is "moving", "stationary", or "long_stationary"
           - Each state gets different filtering parameters
           - This is like adjusting how much you trust new information based on context
           
        2. SPEED & DISTANCE CALCULATION ("How far and fast did it move?")
           - Calculate the distance from previous position
           - Calculate speed based on distance and time elapsed
           - This gives us clues about whether the movement is realistic
        
        3. ADAPTIVE FILTERING ("Adjust filtering based on context")
           - Moving balls: Standard filtering (trust ~70% of new information)
           - Stationary balls: Stronger filtering (trust only ~35% of new information)
           - Long-stationary balls: Very strong filtering (trust only ~21% of new information)
           - Position jumps: Super strong filtering (treat as likely measurement errors)
           
        4. VELOCITY CALCULATION ("How is the ball traveling?")
           - Calculate direction and speed of movement
           - This helps predict where the ball is going
           - Also helps detect impossible movements
           
        EVERYDAY ANALOGY:
        ---------------
        It's like GPS navigation in your car. When you're moving on an open highway, the GPS
        updates your position frequently and mostly trusts the new readings. But when you're
        in a tunnel or between tall buildings (causing GPS errors), the system knows to be
        more skeptical of sudden position jumps and relies more on your previous trajectory.
        
        Args:
            position (list): [x, y, z] position
            timestamp (float): Current time
            motion_state (str, optional): Current motion state if available
            
        Returns:
            list: Filtered position
        """
        # Update motion state if provided
        if motion_state is not None:
            self.motion_state = motion_state
            # Update state counter
            if motion_state in ["stationary", "long_stationary"]:
                self.motion_state_count["stationary"] += 1
                if motion_state == "long_stationary":
                    self.motion_state_count["long_stationary"] += 1
                self.motion_state_count["moving"] = max(0, self.motion_state_count["moving"] - 1)
            else:
                self.motion_state_count["moving"] += 1
                self.motion_state_count["stationary"] = max(0, self.motion_state_count["stationary"] - 1)
                self.motion_state_count["long_stationary"] = max(0, self.motion_state_count["long_stationary"] - 1)
        
        # For the first update, just store the position
        if self.current_position is None:
            self.current_position = list(position)
            self.filtered_position = list(position)
            self.last_update_time = timestamp
            return list(position)
        
        # Calculate time delta
        dt = timestamp - self.last_update_time
        if dt <= 0:
            # Invalid time delta, just return current position
            return self.filtered_position
            
        # Store previous position for jump detection
        prev_position = list(self.filtered_position)
        
        # Calculate distance from previous position
        dx = position[0] - self.filtered_position[0]
        dy = position[1] - self.filtered_position[1]
        dz = position[2] - self.filtered_position[2]
        distance = (dx*dx + dy*dy + dz*dz) ** 0.5
        
        # Calculate speed
        speed = distance / dt
        
        # NEW: Apply velocity sanity check based on motion state
        velocity_valid = True
        expected_max_speed = self.max_speed
        
        # Adjust expected max speed based on motion state
        if self.motion_state == "stationary":
            expected_max_speed = 0.5  # 0.5 m/s max for stationary objects
        elif self.motion_state == "long_stationary":
            expected_max_speed = 0.2  # 0.2 m/s max for long-term stationary
        
        # Check if speed is implausible
        if speed > expected_max_speed:
            # Calculate how many standard deviations from recent speeds
            if len(self.velocity_history) >= 3:
                recent_speeds = [v[3] for v in self.velocity_history[-3:]]
                avg_speed = sum(recent_speeds) / len(recent_speeds)
                if speed > avg_speed * 3:  # More than 3x recent average
                    velocity_valid = False
                    # Treat as jump rather than valid movement
                    self.position_jumps += 1
        
        # NEW: Special handling for stationary and long-stationary objects
        if self.motion_state in ["stationary", "long_stationary"]:
            # For stationary objects, use stronger filtering to reject jumps
            filter_alpha = self.position_filter_alpha * 0.5  # More aggressive smoothing (0.35 vs 0.7)
            
            # For long-stationary objects, use even stronger filtering
            if self.motion_state == "long_stationary":
                filter_alpha = self.position_filter_alpha * 0.3  # Very aggressive smoothing (0.21 vs 0.7)
                
                # Add position to stationary history
                self.stationary_positions.append(list(position))
                if len(self.stationary_positions) > self.stationary_max_length:
                    self.stationary_positions.pop(0)
                
                # If we have enough history, use centroid of recent positions
                if len(self.stationary_positions) >= 5:
                    # Calculate centroid of recent positions
                    centroid = [0, 0, 0]
                    for pos in self.stationary_positions:
                        centroid[0] += pos[0]
                        centroid[1] += pos[1]
                        centroid[2] += pos[2]
                    centroid = [c / len(self.stationary_positions) for c in centroid]
                    
                    # Use centroid with very slight update from new position
                    self.filtered_position = [
                        centroid[0] * 0.9 + position[0] * 0.1,
                        centroid[1] * 0.9 + position[1] * 0.1,
                        self.ground_plane_z  # Force to ground plane
                    ]
                else:
                    # Standard EMA filter with lower alpha
                    self.filtered_position = [
                        self.filtered_position[0] * (1 - filter_alpha) + position[0] * filter_alpha,
                        self.filtered_position[1] * (1 - filter_alpha) + position[1] * filter_alpha,
                        self.ground_plane_z  # Force to ground plane
                    ]
            else:
                # Regular stationary - standard EMA filter with lower alpha
                self.filtered_position = [
                    self.filtered_position[0] * (1 - filter_alpha) + position[0] * filter_alpha,
                    self.filtered_position[1] * (1 - filter_alpha) + position[1] * filter_alpha,
                    self.ground_plane_z  # Force to ground plane
                ]
        else:
            # Moving object - use standard filter
            # Check if this is a position jump rather than real movement
            if distance > self.jump_threshold and speed > self.max_speed:
                self.position_jumps += 1
                # Use less filtering weight for suspected jumps (0.3 vs 0.7)
                filter_alpha = self.position_filter_alpha * 0.3
            else:
                # Normal EMA filter
                filter_alpha = self.position_filter_alpha
                
            # Apply EMA filter
            self.filtered_position = [
                self.filtered_position[0] * (1 - filter_alpha) + position[0] * filter_alpha,
                self.filtered_position[1] * (1 - filter_alpha) + position[1] * filter_alpha,
                self.ground_plane_z  # Force to ground plane
            ]
        
        # Calculate velocity vector
        velocity = [
            (self.filtered_position[0] - prev_position[0]) / dt,
            (self.filtered_position[1] - prev_position[1]) / dt,
            0.0  # No vertical velocity for ground movement
        ]
        
        # Calculate speed from velocity
        current_speed = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2) ** 0.5
        
        # NEW: Apply velocity validation
        if velocity_valid:
            # Store velocity with timestamp and speed
            self.velocity_history.append((velocity[0], velocity[1], velocity[2], current_speed, timestamp))
            if len(self.velocity_history) > self.velocity_history_max_length:
                self.velocity_history.pop(0)
                
            # Update current velocity with smoothing
            if len(self.velocity_history) >= 3:
                # Use weighted average of recent velocities
                total_weight = 0
                weighted_vel = [0, 0, 0]
                
                for i, (vx, vy, vz, spd, ts) in enumerate(self.velocity_history[-3:]):
                    # More recent velocities get higher weight
                    weight = i + 1
                    total_weight += weight
                    weighted_vel[0] += vx * weight
                    weighted_vel[1] += vy * weight
                    weighted_vel[2] += vz * weight
                
                self.current_velocity = [v / total_weight for v in weighted_vel]
            else:
                self.current_velocity = velocity
        
        # Update history
        self.position_history.append(list(self.filtered_position))
        self.time_history.append(timestamp)
        
        if len(self.position_history) > self.history_max_length:
            self.position_history.pop(0)
            self.time_history.pop(0)
        
        # Update statistics
        self.total_distance += distance
        self.max_observed_speed = max(self.max_observed_speed, current_speed)
        
        # Update last update time
        self.last_update_time = timestamp
        
        # Return filtered position
        return self.filtered_position
    
    def get_velocity(self):
        """
        Get current velocity vector.
        
        IMAGINE THIS: 🏀➡️
        ---------------
        Think of the velocity vector as an arrow attached to the basketball, pointing
        in the direction it's moving, with the arrow's length showing how fast it's going.
        This method returns that arrow's properties (how much the ball is moving in each
        of the x, y, and z directions).
        
        HOW IT WORKS:
        ------------
        Returns the 3D velocity vector [vx, vy, vz] showing:
        - vx: How fast the ball is moving east/west (positive = east)
        - vy: How fast the ball is moving north/south (positive = north)
        - vz: How fast the ball is moving up/down (usually 0 for ground movement)
        
        The values are in meters per second (m/s).
        
        EVERYDAY ANALOGY:
        ---------------
        It's like the wind direction and speed. A weather report might say "15 mph winds 
        from the northwest" - that's giving you both direction and magnitude. Our velocity 
        vector does the same thing but breaks it into x, y, and z components.
        """
        return self.current_velocity
    
    def get_speed(self):
        """
        Get current speed magnitude.
        
        IMAGINE THIS: 🏁
        ---------------
        Think of a car's speedometer, which shows how fast you're going without caring
        about the direction. This method is like the basketball's speedometer - it tells
        you how fast the ball is moving without specifying the direction.
        
        HOW IT WORKS:
        ------------
        1. Takes the velocity vector [vx, vy, vz]
        2. Calculates: sqrt(vx² + vy² + vz²)
        3. Returns a single number representing the total speed
        
        This is the Pythagorean theorem in 3D space, combining all movement directions
        into one overall speed value in meters per second (m/s).
        
        EVERYDAY ANALOGY:
        ---------------
        It's like the difference between saying "the car is going 30 mph north and 40 mph east" 
        versus simply "the car is going 50 mph." The speed (50 mph) doesn't tell you the 
        direction, just how fast the object is moving overall.
        """
        return (self.current_velocity[0]**2 + 
                self.current_velocity[1]**2 + 
                self.current_velocity[2]**2) ** 0.5
    
    def get_statistics(self):
        """
        Get filter statistics for diagnostics.
        
        IMAGINE THIS: 📊
        ---------------
        Think of this method like a sports analyst reviewing game footage. After watching
        a basketball game, the analyst compiles statistics like "player speed", "distance covered",
        and "successful passes" to understand performance and identify patterns.
        
        This method works similarly - it analyzes the filter's historical data to provide
        useful performance metrics about how the basketball has been moving.
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. CURRENT SPEED CALCULATION ("How fast is it moving right now?")
           - Calculate the magnitude of the current velocity vector
           - This tells us the instantaneous speed at this moment
           
        2. AVERAGE SPEED CALCULATION ("How fast has it been moving overall?")
           - Calculate total distance covered across position history
           - Divide by total time elapsed
           - This gives us the average speed over recent history
           
        3. STATISTICS COMPILATION ("Gather all diagnostic information")
           - Combine current metrics with historical counters
           - Package everything into a dictionary for easy access
           - Include motion state information for context
           
        EVERYDAY ANALOGY:
        ---------------
        It's like your car's trip computer that shows not just current speed, but also
        average speed, distance traveled, and fuel efficiency. These metrics help you
        understand both current performance and trends over time.
        
        For the robotics system, these statistics are crucial for:
        1. Debugging filter performance
        2. Tuning filter parameters
        3. Understanding the ball's behavior patterns
        4. Diagnosing sensor issues (e.g., too many position jumps)
        """
        current_speed = self.get_speed()
        
        # Calculate average speed from position history
        avg_speed = 0.0
        if len(self.position_history) >= 2 and len(self.time_history) >= 2:
            total_distance = 0.0
            for i in range(1, len(self.position_history)):
                p1 = self.position_history[i-1]
                p2 = self.position_history[i]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dz = p2[2] - p1[2]
                total_distance += (dx*dx + dy*dy + dz*dz) ** 0.5
            
            time_span = self.time_history[-1] - self.time_history[0]
            if time_span > 0:
                avg_speed = total_distance / time_span
        
        return {
            "current_speed": current_speed,
            "average_speed": avg_speed,
            "position_jumps": self.position_jumps,
            "total_distance": self.total_distance,
            "max_observed_speed": self.max_observed_speed,
            "motion_state": self.motion_state
        }
    
    def predict_position(self, time_ahead):
        """
        Predict position after time_ahead seconds based on current state.
        
        IMAGINE THIS: 🎯
        ---------------
        Imagine you're playing basketball and need to pass to a teammate who's running.
        You don't aim at where they are NOW - you aim at where they WILL BE when the ball
        arrives. You're making a prediction based on their current movement.
        
        This method does exactly that! It looks at how the basketball is currently moving
        and predicts where it will be in the future.
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. CURRENT STATE CHECK ("Do we have enough information?")
           - Make sure we have the ball's current position and velocity
           - Without these, we can't make any predictions
        
        2. LINEAR PROJECTION ("Where will it be if it keeps moving this way?")
           - Take current position (x, y, z)
           - Add (velocity × time) to position
           - This gives us the future position assuming constant velocity
           
        3. GROUND PLANE ENFORCEMENT ("Basketball stays on the ground")
           - The z-coordinate (height) stays constant
           - This enforces the physical constraint that the basketball rolls on the ground
        
        EVERYDAY ANALOGY:
        ---------------
        It's like predicting where a rolling billiard ball will be. If you see it moving
        across the table at a constant speed and direction, you can predict where it will
        be in one second by mentally extending its path. This prediction assumes no new
        forces act on the ball (no collisions, no friction changes).
        
        This prediction is essential for robot control since sensors and motors have delays.
        By the time the robot processes the ball's position and moves to intercept it,
        the ball has already moved - so we need to aim for where it WILL be.
        
        Args:
            time_ahead: Time in seconds to predict ahead
            
        Returns:
            Predicted position as (x, y, z)
        """
        if self.filtered_position is None or not self.current_velocity:
            return None
        
        # Simple linear prediction with current velocity
        pred_x = self.filtered_position[0] + self.current_velocity[0] * time_ahead
        pred_y = self.filtered_position[1] + self.current_velocity[1] * time_ahead
        pred_z = self.filtered_position[2]  # Z stays constant (ground plane)
        
        return (pred_x, pred_y, pred_z)