"""
Basketball Tracking Robot - Advanced Target Tracking and Movement Planning
==========================================================================

EDUCATIONAL DOCUMENTATION
------------------------

This module forms the backbone of the robot's tracking intelligence, turning
raw sensor data into smooth, intelligent movement decisions. It bridges the gap
between perception (where is the ball?) and action (how should the robot move?),
implementing sophisticated algorithms for target tracking, movement strategy
selection, and velocity control.

┌───────────────────────────────────────────────────────────────────────────┐
│                   ROBOTICS PERCEPTION-ACTION PIPELINE                     │
│                                                                           │
│  ┌───────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐  │
│  │ SENSORS   │     │ PERCEPTION │     │ PLANNING   │     │ ACTION     │  │
│  │           │     │            │     │            │     │            │  │
│  │ • Cameras │     │ • Filtering│     │ • Strategy │     │ • Motors   │  │
│  │ • LIDAR   │ ──► │ • Tracking │ ──► │ • Decision │ ──► │ • Actuators│  │
│  │ • Depth   │     │ • Fusion   │     │ • Prediction     │ • Feedback │  │
│  │   Sensors │     │            │     │            │     │            │  │
│  └───────────┘     └────────────┘     └────────────┘     └────────────┘  │
│                                                                           │
│  This module handles the critical middle components of this pipeline:     │
│  turning raw perceptions into intelligent movement commands.              │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

Key Concepts for Beginners:
--------------------------

1. MODULAR SOFTWARE ARCHITECTURE

   The code is organized into specialized modules, each with clear responsibilities:
   
   - TARGET TRACKING: Keeps track of ball position, filters data, predicts future positions
   - MOVEMENT STRATEGY: Decides HOW to move based on the current situation
   - VELOCITY CONTROL: Converts strategy decisions into safe, smooth motor commands
   
   This modular design makes the system easier to understand, test, and improve.
   Each module can be enhanced independently without affecting the others.

┌───────────────────────────────────────────────────────────────────────────┐
│                        MODULAR SYSTEM ARCHITECTURE                        │
│                                                                           │
│   ┌───────────────────┐    ┌────────────────────┐   ┌──────────────────┐ │
│   │TargetTrackingModule│   │MovementStrategyModule│  │VelocityControlModule│
│   └─────────┬─────────┘    └──────────┬─────────┘   └────────┬─────────┘ │
│             │                         │                      │           │
│             ▼                         ▼                      ▼           │
│   ┌───────────────────┐    ┌────────────────────┐   ┌──────────────────┐ │
│   │• Filter raw data  │    │• Select strategy   │   │• Apply limits    │ │
│   │• Handle sensor    │───►│• Plan movement     │──►│• Control accel.  │ │
│   │  fusion           │    │• Coordinate axes   │   │• Ensure safety   │ │
│   │• Predict movement │    │                    │   │                  │ │
│   └───────────────────┘    └────────────────────┘   └──────────────────┘ │
│             │                         │                      │           │
│             │                         │                      │           │
│             ▼                         ▼                      ▼           │
│    Filtered Position        Movement Strategies        Safe Velocities   │
│      Predictions              Blending Rules             Smoothing        │
│    Quality Assessment      Context Awareness          Emergency Braking   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

2. ADAPTIVE TRACKING AND PREDICTION

   The robot doesn't just react to where the ball IS - it anticipates where it WILL BE:
   
   - FUSION RATE DETECTION: Automatically detects how frequently sensor data arrives
   - FRESHNESS ANALYSIS: Determines if data is recent enough to be reliable
   - MOVEMENT PATTERN DETECTION: Identifies if the ball is moving consistently
   - DIAGONAL MOVEMENT HANDLING: Special processing for balls moving diagonally
   
   These capabilities allow the robot to track the ball smoothly even when
   sensor data is delayed or inconsistent.

┌───────────────────────────────────────────────────────────────────────────┐
│                     ADAPTIVE TRACKING AND PREDICTION                       │
│                                                                           │
│  ┌───────────────────────────────────────────────────────────────────┐    │
│  │                    PREDICTION VS. REACTION                        │    │
│  │                                                                   │    │
│  │  REACTIVE APPROACH         PREDICTIVE APPROACH                    │    │
│  │  ┌─────────────────┐       ┌─────────────────────────────┐       │    │
│  │  │                 │       │                             │       │    │
│  │  │    Ball         │       │    Previous                 │       │    │
│  │  │    Current      │       │    Positions    ┌────────┐  │       │    │
│  │  │    Position     │       │       ┌─────────► Predict│  │       │    │
│  │  │                 │       │       │         │ Future │  │       │    │
│  │  │       │         │       │       │         │Position│  │       │    │
│  │  │       │         │       │       │         └────────┘  │       │    │
│  │  │       ▼         │       │       │              │      │       │    │
│  │  │   Move Robot    │       │       │              │      │       │    │
│  │  │   To Current    │       │       │              ▼      │       │    │
│  │  │   Position      │       │  Move Robot to Predicted    │       │    │
│  │  │                 │       │  Position                   │       │    │
│  │  └─────────────────┘       └─────────────────────────────┘       │    │
│  │                                                                   │    │
│  └───────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  ADAPTIVE TECHNIQUES:                                                     │
│                                                                           │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐│
│  │ FUSION RATE         │  │ DATA FRESHNESS      │  │ MOVEMENT PATTERN    ││
│  │ DETECTION           │  │ ANALYSIS            │  │ DETECTION           ││
│  ├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤│
│  │• Measures actual    │  │• Evaluates data age │  │• Identifies patterns││
│  │  sensor update rate │  │• Graduated levels:  │  │• Detects:           ││
│  │• Adapts system to   │  │  - Fresh            │  │  - Consistent motion││
│  │  match real-world   │  │  - Stale            │  │  - Direction changes││
│  │  conditions         │  │  - Critical         │  │  - Diagonal movement││
│  │• Range: 0.5-10 Hz   │  │  - Invalid          │  │  - Erratic behavior ││
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘│
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

3. STRATEGY-BASED MOVEMENT PLANNING

   Rather than directly connecting errors to movements, the robot uses a
   strategy-based approach:
   
   - STRATEGY SELECTION: Chooses the best movement strategy based on ball position
   - STRATEGY BLENDING: Smoothly transitions between strategies to prevent jerky motion
   - COORDINATED MOVEMENT: Ensures forward, lateral, and rotational movements work together
   
   This approach creates much more natural and efficient movement than
   simple reactive control.

┌───────────────────────────────────────────────────────────────────────────┐
│                      STRATEGY-BASED MOVEMENT PLANNING                      │
│                                                                           │
│  TRADITIONAL PID                      STRATEGY-BASED APPROACH             │
│  ┌────────────────────┐               ┌────────────────────────────┐      │
│  │                    │               │                            │      │
│  │      Error         │               │      Error Pattern         │      │
│  │        │           │               │          │                 │      │
│  │        ▼           │               │          ▼                 │      │
│  │ ┌──────────────┐   │               │ ┌────────────────────┐    │      │
│  │ │ PID Formula  │   │               │ │ Strategy Selection │    │      │
│  │ └──────┬───────┘   │               │ └────────┬───────────┘    │      │
│  │        │           │               │          │                 │      │
│  │        ▼           │               │          ▼                 │      │
│  │   Motor Commands   │               │ ┌────────────────────┐    │      │
│  │                    │               │ │ Strategy Blending  │    │      │
│  │                    │               │ └────────┬───────────┘    │      │
│  │                    │               │          │                 │      │
│  │                    │               │          ▼                 │      │
│  │                    │               │ ┌────────────────────┐    │      │
│  │                    │               │ │ Motor Coordination │    │      │
│  │                    │               │ └────────┬───────────┘    │      │
│  │                    │               │          │                 │      │
│  │                    │               │          ▼                 │      │
│  │                    │               │     Motor Commands         │      │
│  └────────────────────┘               └────────────────────────────┘      │
│                                                                           │
│  STRATEGY EXAMPLES:                                                       │
│                                                                           │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐  │
│  │ DISTANCE PRIORITY  │  │ ANGULAR CORRECTION │  │ LATERAL ALIGNMENT  │  │
│  ├────────────────────┤  ├────────────────────┤  ├────────────────────┤  │
│  │• Used when ball is │  │• Used when ball is │  │• Used when offset  │  │
│  │  far away          │  │  at an angle       │  │  to the side       │  │
│  │• Prioritizes forward│  │• Rotate to face   │  │• Move sideways     │  │
│  │  movement          │  │  the ball          │  │  first             │  │
│  │• Rotates while     │  │• Forward movement  │  │• Then approach     │  │
│  │  moving forward    │  │  secondary         │  │  forward           │  │
│  └────────────────────┘  └────────────────────┘  └────────────────────┘  │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

4. SAFETY AND PERFORMANCE OPTIMIZATION

   The system incorporates numerous features to ensure safe, efficient operation:
   
   - VELOCITY LIMITING: Prevents unsafe speeds
   - ACCELERATION LIMITING: Ensures smooth acceleration and deceleration
   - RESOURCE MONITORING: Adapts processing based on available CPU
   - FAIL-SAFE MECHANISMS: Handles data loss and sensor failures gracefully
   
   These optimizations are especially important for running on resource-constrained
   systems like the Raspberry Pi 5.

┌───────────────────────────────────────────────────────────────────────────┐
│                    SAFETY AND PERFORMANCE OPTIMIZATION                     │
│                                                                           │
│  SAFETY SYSTEMS:                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                       VELOCITY SAFETY PIPELINE                     │   │
│  │                                                                    │   │
│  │    Raw        ┌─────────┐     ┌──────────┐      ┌──────────┐      │   │
│  │  Velocity     │ Max Vel │     │Accel.    │      │ Proximity│      │   │
│  │  Commands ───►│ Limiting│────►│Limiting  │─────►│ Scaling  │─────►│   │
│  │               └─────────┘     └──────────┘      └──────────┘      │   │
│  │                                                                    │   │
│  │  Maximum allowed velocities:                                       │   │
│  │  • Forward: 0.5 m/s (~1.1 mph)                                     │   │
│  │  • Lateral: 0.4 m/s (~0.9 mph)                                     │   │
│  │  • Rotation: 0.6 rad/s (~34° per second)                           │   │
│  │                                                                    │   │
│  │  Maximum acceleration limits:                                      │   │
│  │  • Forward: 1.8 m/s² (0→0.5m/s in ~0.28s)                           │   │
│  │  • Lateral: 1.5 m/s² (0→0.4m/s in ~0.27s)                           │   │
│  │  • Rotation: 2.0 rad/s² (0→0.6rad/s in ~0.3s)                       │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                           │
│  PERFORMANCE ADAPTATION:                                                  │
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐│
│  │ RESOURCE MONITORING │  │ ADAPTIVE CONTROL    │  │ FAIL-SAFE           ││
│  ├─────────────────────┤  ├─────────────────────┤  ├─────────────────────┤│
│  │• Monitors CPU usage │  │• Changes update     │  │• Graduated response ││
│  │• Tracks memory      │  │  frequency based on │  │  to stale data      ││
│  │  consumption        │  │  available resources│  │• Smooth deceleration││
│  │• Detects processing │  │• Skips update cycles│  │  when data lost     ││
│  │  bottlenecks        │  │  when CPU overloaded│  │• Safe recovery from ││
│  │                     │  │                     │  │  sensor failures    ││
│  └─────────────────────┘  └─────────────────────┘  └─────────────────────┘│
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

Module Architecture:
------------------

1. TargetTrackingModule: Processes and filters sensor data, predicts ball movement
2. MovementStrategyModule: Determines how the robot should move to follow the ball
3. VelocityControlModule: Converts movement decisions into safe motor commands

Together, these modules implement a sophisticated control pipeline that creates
smooth, intelligent robot movement with minimal computational overhead.

┌───────────────────────────────────────────────────────────────────────────┐
│                      CONTROL PIPELINE DATA FLOW                            │
│                                                                           │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐    │
│  │  SENSOR DATA      │   │  MOVEMENT ERRORS  │   │  MOTOR VELOCITIES │    │
│  │                   │   │                   │   │                   │    │
│  │• Ball position    │   │• Distance error   │   │• Linear X (forward)    │
│  │  (x, y, z)        │──►│• Lateral error    │──►│• Linear Y (lateral)    │
│  │• Sensor frame     │   │• Angular error    │   │• Angular Z (rotation)  │
│  │• Detection time   │   │                   │   │                   │    │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘    │
│           │                       │                       │               │
│           ▼                       ▼                       ▼               │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐    │
│  │Target Tracking    │   │Movement Strategy  │   │Velocity Control   │    │
│  │Module             │   │Module             │   │Module             │    │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘    │
│           │                       │                       │               │
│           ▼                       ▼                       ▼               │
│  ┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐    │
│  │• Position filtering│   │• Strategy selection│  │• Velocity limiting│    │
│  │• Freshness analysis│   │• Error mapping    │   │• Accel. limiting  │    │
│  │• Prediction       │   │• Movement planning │   │• Safety checks    │    │
│  │• Sensor fusion    │   │• Directional biasing│  │• Smoothing        │    │
│  └───────────────────┘   └───────────────────┘   └───────────────────┘    │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘

Performance Optimizations:
------------------------
┌───────────────────────────────────────────────────────────────────────────┐
│                        PERFORMANCE OPTIMIZATIONS                           │
│                                                                           │
│  ADAPTIVE PROCESSING                 MEMORY OPTIMIZATIONS                 │
│  ┌────────────────────────┐          ┌────────────────────────┐          │
│  │• Matches update rate to│          │• Pre-allocated buffers │          │
│  │  actual fusion rate    │          │• Fixed-size arrays     │          │
│  │• Adapts to 1-5Hz range │          │• Circular buffers      │          │
│  │• Scales computation    │          │• Reused objects        │          │
│  │  with data quality     │          │• Minimal allocations   │          │
│  └────────────────────────┘          └────────────────────────┘          │
│                                                                           │
│  COMPUTATIONAL EFFICIENCY           FAILURE HANDLING                      │
│  ┌────────────────────────┐          ┌────────────────────────┐          │
│  │• Vectorized operations │          │• Graduated freshness   │          │
│  │• NumPy for math        │          │  levels (fresh, stale) │          │
│  │• Cached calculations   │          │• Safe fallbacks        │          │
│  │• Small-angle approx.   │          │• Recovery strategies   │          │
│  │• Lookup tables         │          │• Consistent behaviors  │          │
│  └────────────────────────┘          └────────────────────────┘          │
│                                                                           │
│  PROCESSING PIPELINE OPTIMIZATION:                                        │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Minimized object creation in hot path                          │   │
│  │ 2. Pre-allocated arrays for all numerical operations              │   │
│  │ 3. Reused calculation buffers across processing steps             │   │
│  │ 4. Batch computations using vectorized operations                 │   │
│  │ 5. Early exits for common cases                                   │   │
│  │ 6. CPU monitoring and cycle skipping under high load              │   │
│  │ 7. Adaptive processing based on fusion data rates                 │   │
│  └────────────────────────────────────────────────────────────────────┘   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
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
from ball_chase.pid.pid_helpers import Matrix4x4, TTLDict, LightweightBuffer, ResourceMonitor
from ball_chase.pid.pid_target_filter import EnhancedTargetFilter, ErrorTracker
from ball_chase.pid.pid_computation import PIDControllers

#############################################
# Target Tracking Module
#############################################

class TargetTrackingModule:
    """
    Advanced target tracking, filtering, and prediction system for basketball tracking.
    
    EDUCATIONAL EXPLANATION:
    -----------------------
    The TargetTrackingModule is the sensory processing center of the robot. It takes
    raw positional data from sensors and transforms it into reliable, predictive
    information about where the basketball is and where it's going.
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                 TARGET TRACKING MODULE ARCHITECTURE                        │
    │                                                                           │
    │  ┌───────────────────┐                                                    │
    │  │   Sensor Data     │                                                    │
    │  │                   │                                                    │
    │  │ • Camera detection│                                                    │
    │  │ • LIDAR points    │                                                    │
    │  │ • Depth data      │                                                    │
    │  └─────────┬─────────┘                                                    │
    │            │                                                              │
    │            ▼                                                              │
    │  ┌───────────────────┐      ┌────────────────────┐                       │
    │  │ Target Processing │      │  Fusion Rate       │                       │
    │  │                   │◄────►│  Detection         │                       │
    │  │ • Noise filtering │      │                    │                       │
    │  │ • Outlier removal │      │ • Measures update  │                       │
    │  │ • Frame transforms│      │   frequency        │                       │
    │  │ • Coordinate conv.│      │ • Adapts thresholds│                       │
    │  └─────────┬─────────┘      └────────────────────┘                       │
    │            │                                                              │
    │            ▼                                                              │
    │  ┌───────────────────┐      ┌────────────────────┐                       │
    │  │  Movement Pattern │      │ Data Freshness     │                       │
    │  │  Detection        │◄────►│ Analysis           │                       │
    │  │                   │      │                    │                       │
    │  │ • Consistent      │      │ • Evaluates age    │                       │
    │  │   motion detection│      │ • Graduated levels │                       │
    │  │ • Direction change│      │ • Adaptive TTLs    │                       │
    │  │ • Diagonal motion │      │ • Confidence score │                       │
    │  └─────────┬─────────┘      └────────────────────┘                       │
    │            │                                                              │
    │            ▼                                                              │
    │  ┌───────────────────┐                                                    │
    │  │  Position         │                                                    │
    │  │  Prediction       │                                                    │
    │  │                   │                                                    │
    │  │ • Physics model   │                                                    │
    │  │ • Future position │                                                    │
    │  │ • Velocity vector │                                                    │
    │  │ • Confidence level│                                                    │
    │  └─────────┬─────────┘                                                    │
    │            │                                                              │
    │            ▼                                                              │
    │  ┌───────────────────┐                                                    │
    │  │ Processed Target  │                                                    │
    │  │ Information       │                                                    │
    │  │                   │                                                    │
    │  │ • Filtered position                                                    │
    │  │ • Predicted position                                                   │
    │  │ • Movement characteristics                                             │
    │  │ • Data quality assessment                                              │
    │  └───────────────────┘                                                    │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    KEY CAPABILITIES:
    ---------------
    
    1. FILTERING AND SMOOTHING
       - Removes sensor noise and random fluctuations
       - Creates a stable position estimate from potentially noisy data
       - Allows smooth control even with imperfect sensors
       - Prevents the robot from reacting to sensor glitches
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                    NOISE FILTERING AND SMOOTHING                           │
    │                                                                           │
    │  RAW SENSOR DATA                         FILTERED DATA                    │
    │                                                                           │
    │     Ball Position                          Ball Position                  │
    │         ^                                      ^                          │
    │         │                                      │                          │
    │         │   ●                                  │                          │
    │         │      ●                               │                          │
    │         │    ●   ●        FILTERING            │       ●                  │
    │         │  ●       ●     ─────────►            │    ●─────●               │
    │         │ ●                                    │  ●         ●             │
    │         │●           ●                         │ ●           ●            │
    │         │                                      │                          │
    │         │                                      │                          │
    │         └──────────────────►                   └──────────────────►       │
    │                 Time                                  Time                │
    │                                                                           │
    │  • Moving average filter removes random fluctuations                      │
    │  • Weighted filtering prioritizes more recent data                        │
    │  • Adaptive filter parameters based on ball movement                      │
    │  • Statistical outlier detection and rejection                           │
    │  • Automatic sensor quality assessment                                    │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    2. MOVEMENT PREDICTION
       - Estimates where the ball will be in the near future (prediction_horizon)
       - Uses physics-based prediction for consistent movement
       - Applies special handling for diagonal movements
       - Allows the robot to anticipate the ball's path
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                        MOVEMENT PREDICTION                                 │
    │                                                                           │
    │  ┌───────────────────────────────────────────────────────────────────┐    │
    │  │                    PHYSICS-BASED PREDICTION                       │    │
    │  │                                                                   │    │
    │  │  Current Time (t)                 Future Time (t + horizon)       │    │
    │  │                                                                   │    │
    │  │  position = [x₀, y₀]              predicted_position = [x₁, y₁]   │    │
    │  │  velocity = [vₓ, vᵧ]                                             │    │
    │  │                                                                   │    │
    │  │   Ball           velocity          Ball                           │    │
    │  │    ●─────────────────►              ●                             │    │
    │  │  position                       predicted_position                │    │
    │  │                                                                   │    │
    │  │  Simple Physics Model:                                            │    │
    │  │  x₁ = x₀ + vₓ * horizon                                          │    │
    │  │  y₁ = y₀ + vᵧ * horizon                                          │    │
    │  │                                                                   │    │
    │  └───────────────────────────────────────────────────────────────────┘    │
    │                                                                           │
    │  DIAGONAL MOVEMENT ENHANCEMENT:                                           │
    │                                                                           │
    │  ┌────────────────────────────┐    ┌───────────────────────────┐         │
    │  │ Standard Prediction        │    │ Enhanced Prediction       │         │
    │  │ (prediction weight = 0.5)  │    │ (prediction weight = 0.7) │         │
    │  │                            │    │                           │         │
    │  │ • 50% current position     │    │ • 30% current position    │         │
    │  │ • 50% predicted position   │    │ • 70% predicted position  │         │
    │  │ • Good for straight-line   │    │ • Better for diagonal and │         │
    │  │   movements                │    │   curved movements        │         │
    │  │                            │    │ • Compensates for sensing │         │
    │  │                            │    │   delays better           │         │
    │  └────────────────────────────┘    └───────────────────────────┘         │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    3. FUSION RATE DETECTION
       - Automatically detects how frequently new sensor data arrives
       - Adapts processing parameters based on actual data rate
       - Ensures control logic matches the system's capabilities
       - Provides crucial information for data freshness evaluation
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                       FUSION RATE DETECTION                               │
    │                                                                           │
    │  ADAPTIVE RATE CALCULATION:                                               │
    │                                                                           │
    │  ┌────────────────────────────────────────────────────────────────────┐   │
    │  │     Updates                                                        │   │
    │  │  ───────────►                                                      │   │
    │  │                                                                    │   │
    │  │  t₁    t₂    t₃    t₄    t₅    t₆    t₇    t₈    t₉    t₁₀        │   │
    │  │  │     │     │     │     │     │     │     │     │     │           │   │
    │  │  ●─────●─────●─────●─────●─────●─────●─────●─────●─────●─────►     │   │
    │  │                                                          time      │   │
    │  │                                                                    │   │
    │  │  ◄──▶ Δt₁  ◄──▶ Δt₂  ◄──▶ Δt₃  ◄──▶ Δt₄  ◄──▶ Δt₅  ◄──▶ ...      │   │
    │  │                                                                    │   │
    │  │  Average interval = (Δt₁ + Δt₂ + ... + Δtₙ) / n                   │   │
    │  │  Detected rate = 1.0 / average_interval                           │   │
    │  │                                                                    │   │
    │  │  • Store timestamps of last 10 updates                            │   │
    │  │  • Calculate time intervals between updates                       │   │
    │  │  • Discard outliers (large gaps > 2 seconds)                      │   │
    │  │  • Compute average interval                                       │   │
    │  │  • Convert to frequency (Hz)                                      │   │
    │  │  • Bounds check: 0.5Hz ≤ rate ≤ 10Hz                              │   │
    │  └────────────────────────────────────────────────────────────────────┘   │
    │                                                                           │
    │  SYSTEM ADAPTATION:                                                       │
    │                                                                           │
    │  ┌────────────────────────────────────────┐                              │
    │  │ EFFECTS OF FUSION RATE DETECTION       │                              │
    │  ├────────────────────┬───────────────────┤                              │
    │  │ Parameter          │ Adaptation         │                              │
    │  ├────────────────────┼───────────────────┤                              │
    │  │ Freshness          │ Sets thresholds    │                              │
    │  │ thresholds         │ relative to rate   │                              │
    │  ├────────────────────┼───────────────────┤                              │
    │  │ Target prediction  │ Adjusts prediction │                              │
    │  │ horizon            │ time window        │                              │
    │  ├────────────────────┼───────────────────┤                              │
    │  │ Filter window      │ Changes filtering  │                              │
    │  │ size               │ sensitivity        │                              │
    │  ├────────────────────┼───────────────────┤                              │
    │  │ Control loop       │ Matches control    │                              │
    │  │ frequency          │ rate to data rate  │                              │
    │  └────────────────────┴───────────────────┘                              │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    4. DATA FRESHNESS ANALYSIS
       - Determines if data is fresh enough to be reliable
       - Provides graduated freshness levels (fresh, stale, critical)
       - Adapts thresholds based on detected fusion rate
       - Enables graceful handling of sensor delays or failures
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                       DATA FRESHNESS ANALYSIS                              │
    │                                                                           │
    │  GRADUATED FRESHNESS LEVELS:                                              │
    │  ┌──────────────────────────────────────────────────────────────────┐     │
    │  │                                                                  │     │
    │  │  last_update                                            now      │     │
    │  │     │                                                     │      │     │
    │  │     ●─────────────────────────────────────────────────────●      │     │
    │  │     │                       data_age                       │      │     │
    │  │     │                                                      │      │     │
    │  │     │◄─────────┬─────────┬─────────┬───────────────────────┤      │     │
    │  │     │  FRESH   │  STALE  │ CRITICAL│      INVALID          │      │     │
    │  │     │          │         │         │                       │      │     │
    │  │  expected_     │expected_│expected_│                       │      │     │
    │  │  interval*1.2  │interval*2.0     │interval*3.0           │      │     │
    │  │                                                                  │     │
    │  └──────────────────────────────────────────────────────────────────┘     │
    │                                                                           │
    │  FRESHNESS LEVEL RESPONSES:                                               │
    │  ┌────────────────────────────────────────────────────────────┐          │
    │  │ Level     │ Age Range              │ System Response        │          │
    │  ├───────────┼───────────────────────┼─────────────────────────┤          │
    │  │ FRESH     │ < 1.2× expected update │ Normal operation        │          │
    │  │           │ interval              │ Full confidence         │          │
    │  ├───────────┼───────────────────────┼─────────────────────────┤          │
    │  │ STALE     │ 1.2× to 2.0× expected │ Reduced velocities (50%)│          │
    │  │           │ interval              │ More prediction weight  │          │
    │  ├───────────┼───────────────────────┼─────────────────────────┤          │
    │  │ CRITICAL  │ 2.0× to 3.0× expected │ Final controlled stop   │          │
    │  │           │ interval              │ Prepare for recovery    │          │
    │  ├───────────┼───────────────────────┼─────────────────────────┤          │
    │  │ INVALID   │ > 3.0× expected update│ Emergency stop          │          │
    │  │           │ interval              │ Sensor reset procedure  │          │
    │  └────────────────────────────────────────────────────────────┘          │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    This module embodies the concept of "smart sensing" - going beyond
    raw data to create a higher-level understanding of the basketball's
    behavior that directly supports intelligent decision-making.
    """
    
    def __init__(self, throttled_logger, filter_buffer_size=5, prediction_horizon=0.2, debug_level=0):
        """
        Initialize target tracking with configurable filtering and prediction parameters.
        
        Args:
            throttled_logger: Logger with rate limiting to prevent log flooding
            filter_buffer_size: Number of positions to keep in filtering buffer (default: 5)
            prediction_horizon: How far ahead to predict ball position in seconds (default: 0.2)
            debug_level: Controls verbosity of diagnostic output (0-3)
            
        The filter_buffer_size affects how many previous positions are used for filtering.
        Larger values create smoother filtering but slower response to sudden changes.
        
        The prediction_horizon controls how far ahead the system predicts the ball's
        position (typically 0.2-0.3 seconds). This compensates for processing delays
        and allows the robot to move toward where the ball will be, not where it was.
        """
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
        Process new sensor data to update tracking and prediction systems.
        
        EDUCATIONAL EXPLANATION:
        -----------------------
        This method is called whenever new position data arrives from sensors
        (typically via the fusion node). It's the entry point for all new
        basketball position information and performs several critical functions:
        
        1. DATA VALIDATION
           - Checks if the message contains valid position data
           - Verifies that the data structure has expected properties
           - Prevents invalid data from corrupting tracking
        
        2. FUSION RATE CALCULATION
           - Measures time intervals between incoming messages
           - Computes the actual rate at which sensor data is arriving
           - Updates internal parameters to match the detected rate
           - This adaptive approach handles varying sensor performance
        
        3. POSITION EXTRACTION AND CONVERSION
           - Extracts raw position data (x, y, z) from the message
           - Converts from sensor coordinates to robot-centric metrics
             (distance, lateral offset, bearing angle)
           - These more intuitive metrics are easier to use for control
        
        4. FILTERING AND PREDICTION
           - Passes the new position to the filtering system
           - Updates velocity and acceleration estimates
           - Recalculates predicted future positions
           - Produces filtered, stable position metrics
        
        The result is a set of reliable, filtered metrics (distance, lateral, bearing)
        that can be used for robot control, even if the original sensor data
        was noisy or inconsistent.
        
        Args:
            target_msg: Message containing target position information
            debug_level: Optional override for diagnostic output verbosity
            
        Returns:
            bool: True if data was successfully processed, False otherwise
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
            
            # CHANGE: More aggressive prediction for diagonal movements
            predicted_position = self.target_filter.get_predicted_position()
            
            # CHANGE: Check for diagonal movement by examining both lateral and distance changes
            diagonal_movement = False
            if predicted_position and len(predicted_position) >= 3 and is_moving:
                # Calculate if both lateral and distance are changing significantly
                lateral_change = abs(predicted_position[1] - filtered_position[1])
                distance_change = abs(predicted_position[0] - filtered_position[0])
                
                # If both are changing by more than 5cm, we have diagonal movement
                diagonal_movement = (lateral_change > 0.05 and distance_change > 0.05)
            
            # For consistent movement or diagonal movements, use prediction
            if (is_moving and not direction_change) or diagonal_movement:
                if predicted_position and len(predicted_position) >= 3:
                    # CHANGE: Use stronger prediction blend for diagonal movements
                    if diagonal_movement:
                        # Use 70% prediction for diagonal movements - look further ahead
                        prediction_weight = 0.7
                        
                        # Log the enhanced prediction for diagonal movement
                        if self.debug_level >= 2:
                            self.logger.info(f"Using enhanced prediction for diagonal movement: weight={prediction_weight:.1f}")
                    else:
                        # Use normal prediction weight
                        prediction_weight = 0.5
                        
                    # CHANGE: Blend filtered and predicted values based on movement type
                    self.filtered_metrics[0] = filtered_position[0] * (1-prediction_weight) + predicted_position[0] * prediction_weight
                    self.filtered_metrics[1] = filtered_position[1] * (1-prediction_weight) + predicted_position[1] * prediction_weight
                    self.filtered_metrics[2] = filtered_position[2] * (1-prediction_weight) + predicted_position[2] * prediction_weight
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
        Evaluate if target data is recent enough to be reliable with graduated freshness levels.
        
        EDUCATIONAL EXPLANATION:
        -----------------------
        Data freshness is critical for robotics - using outdated position information
        can lead to incorrect or unsafe movements. This method implements a sophisticated
        freshness evaluation system using the concept of "graduated freshness."
        
        GRADUATED FRESHNESS CONCEPT:
        --------------------------
        Rather than a simple binary fresh/not-fresh approach, this system recognizes
        different levels of freshness:
        
        1. FRESH (Level 1)
           - Data is very recent (within 1.2x expected update interval)
           - Full confidence in the data
           - Normal control operations proceed
        
        2. STALE (Level 2)
           - Data is somewhat old (1.2x to 2x expected update interval)
           - Reduced confidence in the data
           - Control proceeds with caution (e.g., reduced speed)
        
        3. CRITICAL (Level 3)
           - Data is old but potentially still usable (2x to 3x expected interval)
           - Very low confidence in the data
           - May trigger fallback behaviors or reduced functionality
        
        4. INVALID (Level 4)
           - Data is too old to use (beyond 3x expected interval)
           - No confidence in the data
           - Typically triggers safety stops or recovery behaviors
        
        ADAPTIVE THRESHOLDS:
        ------------------
        The brilliance of this approach is that the freshness thresholds automatically
        adapt to the actual fusion rate. If sensor data typically arrives every 0.5 seconds,
        the thresholds will be different than if data arrives every 0.1 seconds.
        
        This adaptive approach means the system works optimally regardless of:
        - Hardware differences (faster/slower sensors)
        - Processing load variations
        - Communication delays
        
        The graduated freshness concept allows for graceful degradation of
        performance when sensor data becomes delayed, rather than an abrupt
        failure or unsafe behavior.
        
        Args:
            max_age: Optional fixed maximum age (seconds), overrides adaptive calculation
                  
        Returns:
            tuple: (is_fresh, freshness_level, age)
                is_fresh: Boolean indicating if data is usable at all
                freshness_level: String indicating level ('fresh', 'stale', 'critical', 'invalid')
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
    """
    Plans how the robot should move to reach the ball.
    
    IMAGINE THIS: 🧠
    ---------------
    Think of this like the "thinking brain" of the robot - it decides HOW
    to move based on WHERE the ball is. This is similar to how basketball
    players make different movement decisions based on the situation:
    
    - When the ball is far away: run straight toward it
    - When the ball is at an angle: turn to face it first
    - When the ball is close but off to the side: side-step to align
    
    This module is what makes the robot's movements look intelligent rather 
    than mechanical or robotic.
    
    HOW IT WORKS: 📊
    -------------
    Instead of using simple "if the ball is left, turn left" rules, the robot 
    uses a sophisticated table of movement strategies:
    
    1. Categorize errors into meaningful groups:
       - Distance: How far from the ball? (far, medium, close, etc.)
       - Lateral: How far to the side? (left, right, centered)
       - Angular: How much rotation needed? (facing wrong way, slightly off)
       
    2. Look up the best strategy in a strategy catalog:
       - "ANGULAR_PRIMARY": Turn first, then move forward
       - "DIAGONAL_MOVEMENT": Move forward and sideways at the same time
       - "LATERAL_CORRECTION": Move sideways without changing angle
       - Many more specialized strategies for different situations
       
    3. Blend strategies together for smooth transitions:
       - Gradual shifts between strategies instead of abrupt changes
       - No jerky movements when switching from one strategy to another
       
    REAL-WORLD ANALOGY: 🏀
    -------------------
    The MovementStrategyModule works like an experienced basketball player's
    instincts - knowing exactly how to move in each situation to get to the
    ball efficiently, naturally, and smoothly.
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                    MOVEMENT STRATEGY MODULE ARCHITECTURE                   │
    │                                                                           │
    │  ┌───────────────────┐                                                    │
    │  │   Position Errors │                                                    │
    │  │                   │                                                    │
    │  │ • Distance error  │                                                    │
    │  │ • Lateral error   │                                                    │
    │  │ • Angular error   │                                                    │
    │  └─────────┬─────────┘                                                    │
    │            │                                                              │
    │            ▼                                                              │
    │  ┌───────────────────┐      ┌────────────────────┐                       │
    │  │ Error Pattern     │      │  Strategy Database │                       │
    │  │ Analysis          │◄────►│                    │                       │
    │  │                   │      │ • Predefined       │                       │
    │  │ • Categorize errors│      │   strategy catalog│                       │
    │  │ • Identify patterns│      │ • Movement styles │                       │
    │  │ • Determine context│      │ • Dimension scales│                       │
    │  └─────────┬─────────┘      └────────────────────┘                       │
    │            │                                                              │
    │            ▼                                                              │
    │  ┌───────────────────┐      ┌────────────────────┐                       │
    │  │ Strategy Selection│      │ History Tracking   │                       │
    │  │                   │◄────►│                    │                       │
    │  │ • Choose best     │      │ • Previous strategy│                       │
    │  │   strategy        │      │ • Transition state │                       │
    │  │ • Apply context   │      │ • Strategy changes │                       │
    │  │   rules           │      │ • Change timestamps│                       │
    │  └─────────┬─────────┘      └────────────────────┘                       │
    │            │                                                              │
    │            ▼                                                              │
    │  ┌───────────────────┐                                                    │
    │  │ Strategy Blending │                                                    │
    │  │                   │                                                    │
    │  │ • Smooth strategy │                                                    │
    │  │   transitions     │                                                    │
    │  │ • Temporal blending│                                                   │
    │  │ • Prevent jerky   │                                                    │
    │  │   movement changes│                                                    │
    │  └─────────┬─────────┘                                                    │
    │            │                                                              │
    │            ▼                                                              │
    │  ┌───────────────────┐                                                    │
    │  │ Movement Plan     │                                                    │
    │  │                   │                                                    │
    │  │ • Dimension scales│                                                    │
    │  │ • Coordinated     │                                                    │
    │  │   movement params │                                                    │
    │  │ • Movement style  │                                                    │
    │  └───────────────────┘                                                    │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    STRATEGY-BASED APPROACH:
    ----------------------
    Think of this like different driving strategies a human might use:
    
    - When far away: Focus on getting closer quickly (DISTANCE_PRIORITY)
    - When at an angle: Turn to face the ball first (ANGULAR_PRIMARY)
    - When aligned but offset: Move sideways to align (LATERAL_CORRECTION)
    - When approaching diagonally: Move forward and sideways together (DIAGONAL_MOVEMENT)
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                    SITUATION-SPECIFIC MOVEMENT STRATEGIES                  │
    │                                                                           │
    │  ┌───────────────────────────────────────────────────────────────────┐    │
    │  │                                                                   │    │
    │  │       Far Distance            Medium Distance      Close Distance │    │
    │  │       ┌──────────┐            ┌──────────┐         ┌──────────┐  │    │
    │  │       │  Robot   │            │  Robot   │         │  Robot   │  │    │
    │  │       │    ↑     │            │   ↗      │         │ ←        │  │    │
    │  │       │    │     │            │          │         │          │  │    │
    │  │       └────┼─────┘            └─────┬────┘         └────┬─────┘  │    │
    │  │            │                        │                   │        │    │
    │  │            │                        │                   │        │    │
    │  │            │                        │                   │        │    │
    │  │            │                        │                   ↓        │    │
    │  │            ↓                        ↓                            │    │
    │  │         ┌─────┐                  ┌─────┐                ┌─────┐  │    │
    │  │         │Ball │                  │Ball │                │Ball │  │    │
    │  │         └─────┘                  └─────┘                └─────┘  │    │
    │  │                                                                   │    │
    │  │    DISTANCE_PRIORITY         DIAGONAL_APPROACH     LATERAL_ALIGN  │    │
    │  │    Forward: 1.0              Forward: 0.7          Forward: 0.1   │    │
    │  │    Lateral: 0.3              Lateral: 0.7          Lateral: 1.0   │    │
    │  │    Angular: 0.5              Angular: 0.4          Angular: 0.2   │    │
    │  │                                                                   │    │
    │  └───────────────────────────────────────────────────────────────────┘    │
    │                                                                           │
    │  STRATEGY DEFINITION STRUCTURE:                                           │
    │  ┌────────────────────────────────────────────────┐                      │
    │  │                                                │                      │
    │  │  class MovementStrategy:                        │                      │
    │  │      strategy_name = "DISTANCE_PRIORITY"        │                      │
    │  │      description = "Prioritize forward movement"│                      │
    │  │      forward_scale = 1.0   # Full forward       │                      │
    │  │      lateral_scale = 0.3   # Some sideways      │                      │
    │  │      angular_scale = 0.5   # Medium rotation    │                      │
    │  │      transition_speed = 0.5 # Blend speed       │                      │
    │  │      error_thresholds = {                       │                      │
    │  │          "distance": {"small": 0.3, "large": 1.0},                    │
    │  │          "lateral": {"small": 0.2, "large": 0.5},                     │
    │  │          "angular": {"small": 15, "large": 45}  │                      │
    │  │      }                                          │                      │
    │  │                                                │                      │
    │  └────────────────────────────────────────────────┘                      │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    Rather than hardcoding these behaviors, the system selects from dozens of
    predefined strategies based on the current error pattern. Each strategy defines:
    
    1. WHICH DIMENSIONS TO USE
       - Forward/backward movement
       - Lateral (sideways) movement
       - Rotational movement
    
    2. HOW STRONGLY TO USE EACH DIMENSION
       - Scaling factors (0.0-1.0) for each dimension
       - These create different movement "styles"
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                  STRATEGY SELECTION AND BLENDING PROCESS                   │
    │                                                                           │
    │  STRATEGY SELECTION:                                                      │
    │                                                                           │
    │  ┌─────────────────────────────────────────────────────────────────┐      │
    │  │ 1. Categorize errors:                                          │      │
    │  │    - Distance: none/small/medium/large                         │      │
    │  │    - Lateral: none/small/medium/large                          │      │
    │  │    - Angular: none/small/medium/large                          │      │
    │  │                                                               │      │
    │  │ 2. Generate error pattern:                                     │      │
    │  │    Example: (LARGE_DISTANCE, SMALL_LATERAL, MEDIUM_ANGULAR)    │      │
    │  │                                                               │      │
    │  │ 3. Look up matching strategy in strategy database:            │      │
    │  │    Error Pattern → DISTANCE_PRIORITY strategy                  │      │
    │  │                                                               │      │
    │  │ 4. Apply context rules:                                       │      │
    │  │    - Is the robot stopped? → Prefer orientation first         │      │
    │  │    - Was a previous strategy working well? → Maintain it      │      │
    │  │    - Special cases (diagonal movement detected)               │      │
    │  │                                                               │      │
    │  └─────────────────────────────────────────────────────────────────┘      │
    │                                                                           │
    │  STRATEGY BLENDING:                                                       │
    │                                                                           │
    │  ┌──────────────────────────────────────────────────────────────┐        │
    │  │                                                             │        │
    │  │  Previous Strategy    Current Strategy      Blended Result   │        │
    │  │  ┌──────────────┐     ┌─────────────┐      ┌─────────────┐  │        │
    │  │  │ANGULAR_PRIMARY│     │LATERAL_ALIGN│      │TRANSITION   │  │        │
    │  │  │Fwd:  0.1     │     │Fwd:  0.2    │      │Fwd:  0.15   │  │        │
    │  │  │Lat:  0.3     │  +  │Lat:  0.9    │  =   │Lat:  0.6    │  │        │
    │  │  │Ang:  1.0     │     │Ang:  0.2    │      │Ang:  0.6    │  │        │
    │  │  └──────────────┘     └─────────────┘      └─────────────┘  │        │
    │  │                                                             │        │
    │  │  • Blend factor based on time since strategy change         │        │
    │  │  • Transition_speed controls blending duration             │        │
    │  │  • Linear interpolation between parameter values            │        │
    │  │  • Prevents sudden jerks when strategy changes              │        │
    │  │  • Creates natural, fluid transitions in robot movement     │        │
    │  │                                                             │        │
    │  └──────────────────────────────────────────────────────────────┘        │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    3. STRATEGY BLENDING
       - Smooth transitions between strategies
       - Prevents jerky changes in movement
       - Creates natural, fluid motion
    
    This approach creates much more natural and efficient movement than
    simple reactive control. It's similar to how a human would move - 
    we don't follow rigid formulas but adapt our movement style to
    the specific situation.
    """
    
    def __init__(self, throttled_logger, debug_level=0):
        """
        Initialize the strategic movement planning system.
        
        Args:
            throttled_logger: Logger with rate limiting to prevent log flooding
            debug_level: Controls verbosity of diagnostic output (0-3)
            
        This module delegates most of its functionality to the StrategyManager
        from the PIDControllers module, which maintains the strategy database
        and selection logic. The module adds tracking of strategy changes and
        transition management.
        """
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
    """
    Makes the robot move safely and smoothly like a professional driver.
    
    IMAGINE THIS: 🚗
    ---------------
    Think of this like the difference between how a beginner drives a car
    versus how a professional driver handles it:
    
    - Beginner driver: Jerky starts and stops, sudden acceleration, abrupt braking
    - Professional driver: Smooth acceleration, gentle deceleration, no sudden movements
    
    The VelocityControlModule is like having a professional driver controlling
    the robot's movement - it takes the raw commands ("go forward", "turn left")
    and transforms them into smooth, natural-looking motion.
    
    HOW IT WORKS: 🛡️
    -------------
    1. SAFETY FIRST
       - Prevents commands that would move too fast
       - Makes sure the robot slows down when getting close to the target
       - Reacts appropriately to stale or missing sensor data
       
    2. SMOOTH ACCELERATION
       - Gradually builds up speed instead of jerky starts
       - Limits how quickly velocity can change (just like a car can't go from
         0 to 60 mph instantly)
       - Makes movements look intentional and controlled
       
    3. INTELLIGENT BRAKING
       - Starts slowing down at just the right distance from the target
       - Uses quadratic deceleration curve for natural-feeling stops
       - More aggressive braking when approaching at high speed
       
    4. MULTI-DIMENSIONAL COORDINATION
       - Balances forward, sideways, and rotational movement
       - Prevents instability from trying to do too much at once
       - Creates proper proportions between different movement types
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                    VELOCITY CONTROL MODULE ARCHITECTURE                    │
    │                                                                           │
    │  ┌───────────────────┐                                                    │
    │  │  Target Velocities│                                                    │
    │  │                   │                                                    │
    │  │ • Forward (lin_x) │                                                    │
    │  │ • Lateral (lin_y) │                                                    │
    │  │ • Angular (ang_z) │                                                    │
    │  └─────────┬─────────┘                                                    │
    │            │                                                              │
    │            ▼                                                              │
    │  ┌───────────────────┐      ┌────────────────────┐                       │
    │  │ Maximum Velocity  │      │  Distance-Aware    │                       │
    │  │ Limiting          │◄────►│  Approach Scaling  │                       │
    │  │                   │      │                    │                       │
    │  │ • Enforce safety  │      │ • Slow down when   │                       │
    │  │   limits          │      │   approaching      │                       │
    │  │ • Different limits│      │ • Prevent overshoot│                       │
    │  │   per dimension   │      │ • Gentle final     │                       │
    │  └─────────┬─────────┘      │   approach         │                       │
    │            │                └────────────────────┘                       │
    │            ▼                                                              │
    │  ┌───────────────────┐      ┌────────────────────┐                       │
    │  │ Acceleration      │      │ Direction Change   │                       │
    │  │ Limiting          │◄────►│ Detection          │                       │
    │  │                   │      │                    │                       │
    │  │ • Smooth accel.   │      │ • Detect sudden    │                       │
    │  │   and deceleration│      │   direction changes│                       │
    │  │ • Natural motion  │      │ • Apply special    │                       │
    │  │   profiles        │      │   handling         │                       │
    │  └─────────┬─────────┘      └────────────────────┘                       │
    │            │                                                              │
    │            ▼                                                              │
    │  ┌───────────────────┐      ┌────────────────────┐                       │
    │  │ Combined Movement │      │ Velocity History   │                       │
    │  │ Optimization      │◄────►│ Tracking           │                       │
    │  │                   │      │                    │                       │
    │  │ • Balance movement│      │ • Track recent     │                       │
    │  │   dimensions      │      │   velocity commands│                       │
    │  │ • Prevent instable│      │ • Analyze movement │                       │
    │  │   combinations    │      │   patterns         │                       │
    │  └─────────┬─────────┘      └────────────────────┘                       │
    │            │                                                              │
    │            ▼                                                              │
    │  ┌───────────────────┐                                                    │
    │  │ Safe Velocity     │                                                    │
    │  │ Commands          │                                                    │
    │  │                   │                                                    │
    │  │ • Optimized       │                                                    │
    │  │ • Safety-checked  │                                                    │
    │  │ • Smooth          │                                                    │
    │  │ • Coordinated     │                                                    │
    │  └───────────────────┘                                                    │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    KEY CAPABILITIES:
    ---------------
    
    1. SAFETY CONSTRAINTS
       - Enforces maximum velocity limits in each dimension
       - Prevents the robot from moving too quickly
       - Supports different limits for forward, lateral, and rotational movement
       - These limits protect the robot and environment from damage
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                      SAFE VELOCITY LIMITS SYSTEM                           │
    │                                                                           │
    │  ROBOT SAFETY SPEED ENVELOPE:                                             │
    │  ┌──────────────────────────────────────────────────────────────┐        │
    │  │                                                             │        │
    │  │               Forward Speed (m/s)                           │        │
    │  │                      ↑                                      │        │
    │  │                      │                                      │        │
    │  │                    0.5 ┌─────────────────────┐               │        │
    │  │                      │ │                   │               │        │
    │  │                      │ │     Safe Zone     │               │        │
    │  │                      │ │                   │               │        │
    │  │                      │ │                   │               │        │
    │  │  Lateral    ◄────────┼─┼───────┼───────────┼──────────►    │        │
    │  │  Speed (m/s)       -0.4│       0          │0.4           │        │
    │  │                      │ │                   │               │        │
    │  │                      │ │                   │               │        │
    │  │                      │ │                   │               │        │
    │  │                      │ └─────────────────────┘               │        │
    │  │                   -0.5 │                                      │        │
    │  │                      │                                      │        │
    │  │                      ↓                                      │        │
    │  │                                                             │        │
    │  │  Angular Speed Limit: ±0.6 rad/s (~34° per second)          │        │
    │  │                                                             │        │
    │  └──────────────────────────────────────────────────────────────┘        │
    │                                                                           │
    │  IMPLEMENTATION:                                                          │
    │  ┌────────────────────────────────────────────────────────────────┐      │
    │  │                                                               │      │
    │  │  # Apply maximum velocity limits using vectorized operation    │      │
    │  │  np.clip(                                                     │      │
    │  │    velocities,                                                │      │
    │  │    -self.max_velocity_limits,                                 │      │
    │  │    self.max_velocity_limits,                                  │      │
    │  │    out=velocities                                             │      │
    │  │  )                                                            │      │
    │  │                                                               │      │
    │  │  # Safety checks for combined movements                       │      │
    │  │  if lateral_magnitude > 0.15 and angular_magnitude > 0.3:     │      │
    │  │    velocities[1] *= 0.7  # reduce lateral by 30%             │      │
    │  │                                                               │      │
    │  └────────────────────────────────────────────────────────────────┘      │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    2. SMOOTH ACCELERATION CONTROL
       - Limits how quickly velocity can change (acceleration limits)
       - Prevents jarring starts, stops, and direction changes
       - Creates natural, smooth motion profiles
       - Reduces mechanical stress and improves tracking stability
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                    SMOOTH ACCELERATION CONTROL SYSTEM                      │
    │                                                                           │
    │  ACCELERATION LIMITS:                                                     │
    │                                                                           │
    │  │  Target velocity: 0.5 m/s                                              │
    │  │                                       ┌───────────                     │
    │  │  Velocity                             │                                │
    │  │    ↑                                  │                                │
    │  │    │                                  │                                │
    │  │    │                               ┌──┘                                │
    │  │ 0.5├─ - - - - - - - - - - - - - - -┤                                   │
    │  │    │                            ┌─┘                                    │
    │  │    │                         ┌──┘                                      │
    │  │    │                      ┌──┘                                         │
    │  │    │                   ┌──┘                                            │
    │  │    │                ┌──┘                                               │
    │  │    │             ┌──┘                                                  │
    │  │    │          ┌──┘                                                     │
    │  │  0 └──────────┼──────────────────────────────────────────────►         │
    │  │               0                          time                          │
    │  │                                                                        │
    │  │  Acceleration limits: 1.8 m/s² forward, 1.5 m/s² lateral              │
    │  │  Time to reach 0.5 m/s from stop: ~0.28 seconds                        │
    │  │                                                                        │
    │  ACCELERATION LIMITING ALGORITHM:                                         │
    │  ┌────────────────────────────────────────────────────────────────┐      │
    │  │ 1. Calculate time elapsed since last update (dt)               │      │
    │  │ 2. Compute maximum allowed velocity change for this cycle:      │      │
    │  │    max_Δv = acceleration_limit * dt                            │      │
    │  │    Example: 1.8 m/s² * 0.05s = 0.09 m/s maximum change         │      │
    │  │ 3. Calculate actual requested velocity change:                  │      │
    │  │    Δv = target_velocity - current_velocity                     │      │
    │  │ 4. If |Δv| > max_Δv, limit the change:                         │      │
    │  │    new_v = current_v + sign(Δv) * max_Δv                      │      │
    │  │                                                               │      │
    │  │ Special case: Starting movement                                │      │
    │  │ • Apply boost factor (1.5-2.5×) when starting from stop       │      │
    │  │ • Allows faster initial response while maintaining safety      │      │
    │  └────────────────────────────────────────────────────────────────┘      │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    3. PROXIMITY-BASED VELOCITY SCALING
       - Automatically slows down as the robot approaches the target
       - Uses distance-based scaling factors for gentle final approach
       - Prevents overshooting and oscillation around the target
       - Mimics how humans naturally slow down when approaching a destination
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                PROXIMITY-BASED VELOCITY SCALING SYSTEM                     │
    │                                                                           │
    │  DISTANCE-AWARE APPROACH:                                                 │
    │  ┌────────────────────────────────────────────────────────────────────┐   │
    │  │                                                                   │   │
    │  │                                                                   │   │
    │  │  Speed                                                            │   │
    │  │    ↑                                                              │   │
    │  │    │                                                              │   │
    │  │ 1.0├───────────                                                   │   │
    │  │    │           └───────────                                       │   │
    │  │    │                       └───────────                           │   │
    │  │    │                                   └───────────               │   │
    │  │    │                                               └───────────   │   │
    │  │    │                                                           └──│   │
    │  │ 0.2├                                                              │   │
    │  │    │                                                              │   │
    │  │  0 └──────────┬─────────┬─────────┬─────────┬─────────┬──────────▶│   │
    │  │              far        0.9m      0.6m      0.3m      0.15m   0m  │   │
    │  │                                 Distance to target                │   │
    │  │                                                                   │   │
    │  │  • approach_distance = 0.3m (begin slowing at 0.9m)              │   │
    │  │  • min_approach_factor = 0.2 (20% of full speed when very close) │   │
    │  │  • Quadratic curve for natural deceleration profile              │   │
    │  │  • Enhanced deceleration when approaching at high speed          │   │
    │  └────────────────────────────────────────────────────────────────────┘   │
    │                                                                           │
    │  QUADRATIC SCALING FORMULA:                                               │
    │  ┌────────────────────────────────────────────────────────────────┐      │
    │  │                                                               │      │
    │  │  normalized_distance = distance_error / approach_distance      │      │
    │  │                                                               │      │
    │  │  approach_factor = max(min_approach_factor,                   │      │
    │  │                       normalized_distance²)                    │      │
    │  │                                                               │      │
    │  │  # Further reduce when very close (within half approach distance)  │      │
    │  │  if distance_error < approach_distance * 0.5:                  │      │
    │  │      approach_factor *= 0.5                                   │      │
    │  │                                                               │      │
    │  │  # Apply the scaling to forward velocity                       │      │
    │  │  scaled_velocity = linear_x * approach_factor                 │      │
    │  │                                                               │      │
    │  └────────────────────────────────────────────────────────────────┘      │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    4. MULTI-DIMENSIONAL COORDINATION
       - Handles forward, lateral, and rotational movements simultaneously
       - Ensures coordinated motion across all dimensions
       - Maintains proper velocity ratios for the selected strategy
       - Creates natural, efficient movement patterns
    
    ┌───────────────────────────────────────────────────────────────────────────┐
    │                  MULTI-DIMENSIONAL MOVEMENT COORDINATION                   │
    │                                                                           │
    │  BALANCING DIMENSIONS:                                                    │
    │  ┌────────────────────────────────────────────────────────────┐          │
    │  │                                                           │          │
    │  │  UNBALANCED MOVEMENT             BALANCED MOVEMENT        │          │
    │  │  ┌────────────────────┐          ┌────────────────────┐   │          │
    │  │  │ Linear X: 0.5 m/s  │          │ Linear X: 0.4 m/s  │   │          │
    │  │  │ Linear Y: 0.05 m/s │    →     │ Linear Y: 0.2 m/s  │   │          │
    │  │  │ Angular: 0.2 rad/s │          │ Angular: 0.2 rad/s │   │          │
    │  │  └────────────────────┘          └────────────────────┘   │          │
    │  │                                                           │          │
    │  │  Problem:                       Solution:                 │          │
    │  │  Forward movement               Redistribution creates    │          │
    │  │  dominates, making              better balance between    │          │
    │  │  lateral adjustment too         motion dimensions         │          │
    │  │  slow and ineffective                                     │          │
    │  │                                                           │          │
    │  └────────────────────────────────────────────────────────────┘          │
    │                                                                           │
    │  COMBINED MOVEMENT SAFETY:                                                │
    │  ┌────────────────────────────────────────────────────────────┐          │
    │  │                                                           │          │
    │  │  COMBINED MOVEMENT PHYSICS:                               │          │
    │  │                                                           │          │
    │  │  • Simultaneous large lateral movement and rotation       │          │
    │  │    creates centrifugal forces                             │          │
    │  │  • This can cause tipping or slipping in wheeled robots   │          │
    │  │  • Solution: Automatically reduce lateral velocity when   │          │
    │  │    combined with significant rotation                     │          │
    │  │                                                           │          │
    │  │  IMPLEMENTATION:                                          │          │
    │  │  # Check for potentially unstable combinations            │          │
    │  │  if lateral_magnitude > 0.2 and angular_magnitude > 0.4:  │          │
    │  │    # Apply 70% reduction to lateral velocity              │          │
    │  │    velocities[1] *= 0.7                                  │          │
    │  │                                                           │          │
    │  └────────────────────────────────────────────────────────────┘          │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘
    
    The result is a robot that moves smoothly and naturally, with acceleration
    and deceleration profiles that appear intentional and controlled, rather
    than mechanical and jerky.
    """
    
    def __init__(self, throttled_logger, history_size=10, max_velocity=None, acceleration_limits=None):
        """
        Initialize the advanced velocity control system with safety parameters.
        
        Args:
            throttled_logger: Logger with rate limiting to prevent log flooding
            history_size: Number of previous velocities to record for trending
            max_velocity: Optional custom velocity limits [forward, lateral, rotational]
                          in meters/second and radians/second
            acceleration_limits: Optional custom acceleration limits [forward, lateral, rotational]
                               in meters/second² and radians/second²
            
        The max_velocity limits prevent the robot from moving too quickly, which
        could be unsafe or cause control issues. The acceleration_limits control
        how quickly velocities can change, ensuring smooth motion.
        
        If not provided, these limits use conservative defaults that work well
        for most small-to-medium sized robots, balancing responsiveness with
        safe, smooth operation.
        """
        self.logger = throttled_logger
        self.history_size = history_size  # Number of velocity samples to keep for analysis
        
        # Initialize velocity history buffer
        self.velocity_history = []  # Always use a list initially for flexibility
        self.last_cmd_vel = np.zeros(3, dtype=np.float32)  # Last command velocity [x, y, angular]
        
        # Velocity approach parameters
        self.approach_distance = 0.3  # Distance (m) at which to start slowing down approach
        self.min_approach_factor = 0.2  # Minimum velocity factor when very close to target (20% of full speed)
        self.debug_level = 0  # Logging verbosity level: 0=minimal, 1=normal, 2=verbose, 3=debug
        
        # Previous velocities for acceleration limiting
        self.last_cmd_vel = np.zeros(3, dtype=np.float32)  # [x, y, angular_z] in m/s, m/s, rad/s
        self.last_logged_cmd = np.zeros(3, dtype=np.float32)  # Last logged command for change detection
        self.last_accel_time = time.time()  # Timestamp for calculating dt in acceleration limiting
        
        # Configure maximum velocity limits - allow overrides from parameters
        if max_velocity is not None and isinstance(max_velocity, (list, tuple)) and len(max_velocity) >= 3:
            self.max_velocity_limits = np.array(max_velocity, dtype=np.float32)
        else:
            # Default limits - same as original for compatibility
            # [forward m/s, lateral m/s, rotation rad/s]
            # 0.5 m/s forward = ~1.1 mph, 0.4 m/s lateral = ~0.9 mph, 0.6 rad/s = ~34 degrees/second
            self.max_velocity_limits = np.array([0.5, 0.4, 0.6], dtype=np.float32)  # [x, y, angular_z]
            
        # Configure acceleration limits - determines how quickly velocity can change
        if acceleration_limits is not None and isinstance(acceleration_limits, (list, tuple)) and len(acceleration_limits) >= 3:
            self.acceleration_limits = np.array(acceleration_limits, dtype=np.float32)
        else:
            # Default acceleration limits - m/s² for linear, rad/s² for angular
            # 1.8 m/s² forward = 0->0.5m/s in ~0.28s
            # 1.5 m/s² lateral = 0->0.4m/s in ~0.27s
            # 2.0 rad/s² angular = 0->0.6rad/s in ~0.3s
            self.acceleration_limits = np.array([1.8, 1.5, 2.0], dtype=np.float32)  # [x accel, y accel, angular accel]
        
        # Pre-allocated arrays for performance (minimize allocations in control loop)
        self._limited_velocities = np.zeros(3, dtype=np.float32)  # Output buffer for limited velocities
        self._target_velocities = np.zeros(3, dtype=np.float32)  # Input buffer for target velocities
        self._vel_diffs = np.zeros(3, dtype=np.float32)  # Buffer for velocity differences
        
        # Pre-allocated velocity change check array (for logging decisions)
        self._velocity_change_check = np.zeros(3, dtype=bool)  # [x_changed, y_changed, angular_changed]
        
        # Velocity history buffer - optimized for fixed size and efficient lookups
        self.buffer_size = 6  # Store 6 previous velocity commands (typically ~1-3 seconds of history)
        self.velocity_history = np.zeros((self.buffer_size, 3), dtype=np.float32)  # Circular buffer [time][x,y,θ]
        self.history_index = 0  # Current position in circular buffer
        self.history_count = 0  # Number of valid entries in history buffer
        
        # Threshold lookup table for minimum velocities
        self.min_linear_velocity = 0.01  # Minimum linear velocity (m/s) - below this will be zeroed
        self.min_angular_velocity = 0.01  # Minimum angular velocity (rad/s) - below this will be zeroed
        
        # Velocity ramping configuration - controls how quickly velocity changes are applied
        self.ramping_enabled = True  # Enable/disable ramping feature
        self.ramp_threshold = 0.05   # Velocity difference threshold (m/s) above which ramping applies
        self.max_ramp_factor = 0.6   # Maximum ramping factor (higher = faster changes, 1.0 = instant)
        
        # Direction change tracking - for smoothing direction changes
        self.prev_direction = np.zeros(3)  # Previous normalized velocity direction vector
        self.direction_change_factor = 0.5  # When direction changes, multiply velocity by this factor (50%)
    
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
        self.last_cmd_vel = np.zeros(3, dtype=np.float32)
        self.last_logged_cmd = np.zeros(3, dtype=np.float32)
        
        # Reset timing
        self.last_accel_time = time.time()
        
        # Reset velocity history
        self.velocity_history = np.zeros((self.buffer_size, 3), dtype=np.float32)
        self.history_index = 0
        self.history_count = 0
        
        # Reset velocity arrays
        self._limited_velocities.fill(0.0)
        self._target_velocities.fill(0.0)
        self._vel_diffs.fill(0.0)
        
        # Reset velocity change check 
        self._velocity_change_check.fill(False)
        
        # Reset direction tracking
        self.prev_direction.fill(0.0)
    

    def _balance_velocity_distribution(self):
        """
        Balance velocity distribution to improve multi-axis movements.
        Ensures one movement axis doesn't dominate at the expense of others.
        """
        try:
            # Only apply balancing when multiple significant velocities exist
            significant_axes = 0
            for i in range(3):
                if abs(self._limited_velocities[i]) > 0.05:  # 5cm/s or 0.05rad/s threshold
                    significant_axes += 1
                    
            # Only balance when we have multiple significant movement axes
            if significant_axes >= 2:
                # Calculate magnitude of total movement vector
                total_magnitude = np.sqrt(np.sum(self._limited_velocities[:2]**2))  # Just x and y components
                
                # Only apply balancing for significant movement
                if total_magnitude > 0.1:  # At least 10cm/s combined movement
                    # Calculate what portion of total movement each component represents
                    proportions = np.abs(self._limited_velocities) / max(0.001, total_magnitude)
                    
                    # Check if linear X is dominating (forward/backward)
                    if proportions[0] > 0.85 and abs(self._limited_velocities[1]) > 0.05:
                        # Linear X is taking >85% of movement budget while Y needs >5cm/s
                        # Redistribute to give lateral movement more priority
                        self._limited_velocities[0] *= 0.8  # Reduce X to 80%
                        
                        # Log the balancing if in debug mode
                        if self.debug_level >= 2:
                            self.logger.info(
                                f"Balanced velocity distribution: reduced X dominance "
                                f"from {self._limited_velocities[0]/0.8:.2f} to {self._limited_velocities[0]:.2f}"
                            )
                    
                    # Similar check for Y dominating
                    if proportions[1] > 0.85 and abs(self._limited_velocities[0]) > 0.05:
                        # Linear Y is taking >85% of movement budget while X needs >5cm/s
                        self._limited_velocities[1] *= 0.8  # Reduce Y to 80%
                        
                        # Log the balancing if in debug mode
                        if self.debug_level >= 2:
                            self.logger.info(
                                f"Balanced velocity distribution: reduced Y dominance "
                                f"from {self._limited_velocities[1]/0.8:.2f} to {self._limited_velocities[1]:.2f}"
                            )
        except Exception as e:
            self.logger.error(f"Error in velocity balancing: {str(e)}")
            # Continue without balancing on error
    
    def process_velocities(self, linear_x, linear_y, angular_z, filtered_distance, desired_distance, freshness_level='fresh'):
        """
        Transforms raw movement commands into smooth, natural robot motion.
        
        IMAGINE THIS: 🎮
        ---------------
        Imagine the difference between a racing game with basic controls vs. 
        one with advanced physics:
        
        - Basic game: Press forward = instant full speed, release = instant stop
        - Advanced game: Gradual acceleration, momentum, natural braking, etc.
        
        This method transforms those simple "move forward/turn left" commands
        into the realistic physics-based motion that makes the robot's movement
        look smooth and natural.
        
        THE PROCESS: 🔄
        -----------
        Your robot movement request goes through these enhancement steps:
        
        1. FRESHNESS CHECK 👁️
           "How recent is our sensor data? If it's old, let's move more cautiously."
           
        2. APPROACH CONTROL 🛑
           "We're getting close to the target - let's start slowing down
           gradually so we don't overshoot."
           
        3. PREDICTIVE BRAKING 🧠
           "At this speed, we need to start braking now to stop at the
           right place. Let's look ahead and plan our deceleration."
           
        4. DIRECTION CHANGES 🔄
           "We're changing direction from forward to right - let's make
           that transition feel smooth rather than jerky."
           
        5. ACCELERATION PHYSICS 📈
           "Real objects can't instantly change speed - let's apply
           natural acceleration and deceleration limits."
           
        6. FINAL REFINEMENTS ✨
           "Let's add those little details that make movement feel natural:
           minimum speed thresholds, balance between rotation and translation,
           and intelligent distribution between different movement directions."
            
        This is what transforms simple movement commands into motion that
        looks intelligent, smooth, and intentional instead of robotic.
        
        Args:
            linear_x: Forward/backward speed request
            linear_y: Left/right speed request
            angular_z: Rotation speed request
            filtered_distance: How far from target we are
            desired_distance: How far we want to be
            freshness_level: How recent our sensor data is
            
        Returns:
            Three refined velocity values that create smooth, natural movement
        """
        try:
            # Validate inputs
            self._target_velocities[0] = float(linear_x) if linear_x is not None else 0.0
            self._target_velocities[1] = float(linear_y) if linear_y is not None else 0.0
            self._target_velocities[2] = float(angular_z) if angular_z is not None else 0.0
            filtered_distance = float(filtered_distance) if filtered_distance is not None else 1.0
            desired_distance = float(desired_distance) if desired_distance is not None else 1.0
            
            # Calculate current time once for efficiency
            current_time = time.time()
            
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
            
            # Check for direction changes
            self._detect_direction_changes()
            
            # Apply acceleration limits
            self._apply_acceleration_limits(current_time)
            
            # Apply progressive velocity ramping for smoother transitions
            if self.ramping_enabled:
                self._apply_progressive_ramping()
            
            # Apply minimum velocity thresholds with hysteresis
            self._apply_minimum_thresholds()
            
            # Limit combined lateral and angular movement to prevent instability
            self._limit_combined_movements()
            
            # Apply velocity smoothing using sliding window
            self._apply_velocity_smoothing()
            
            # Balance velocity distribution for diagonal movements
            self._balance_velocity_distribution()
            
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
            return tuple(self.last_cmd_vel)
            
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
    
    def _detect_direction_changes(self):
        """
        Detect significant direction changes and apply damping.
        """
        # Initialize normalized direction vector
        current_dir = np.zeros(3)
        
        # Calculate velocity magnitude (speed) in the X-Y plane
        # Using Euclidean norm: sqrt(vx² + vy²)
        vel_mag = np.sqrt(np.sum(self._target_velocities[:2]**2))
        
        # Only calculate direction if robot is moving significantly
        # 0.05 m/s threshold filters out noise and very slow movements
        if vel_mag > 0.05:
            # Normalize velocity vector to get direction unit vector
            # This gives us a vector with magnitude 1 pointing in direction of travel
            current_dir[:2] = self._target_velocities[:2] / vel_mag
            
            # Check for direction change only if we have a previous direction
            if np.any(self.prev_direction != 0):
                # Calculate dot product between current and previous direction
                dot_product = np.sum(current_dir[:2] * self.prev_direction[:2])
                
                # Detect significant direction change based on dot product threshold
                # CHANGE: Reduced threshold from 0.7 to 0.5 (cos 60° instead of cos 45°)
                # This makes the controller more responsive during diagonal tracking
                # by only considering more severe direction changes (>60 degrees)
                if dot_product < 0.5:
                    # CHANGE: Apply less aggressive damping (0.7 instead of 0.5)
                    # This keeps more momentum during direction changes
                    damping_factor = 0.7  # Changed from 0.5 (50%) to 0.7 (70%)
                    self._target_velocities *= damping_factor
                    
                    # Log direction changes at debug level
                    if self.debug_level >= 2:
                        # Calculate angle in degrees for more intuitive logging
                        angle_rad = np.arccos(max(-1.0, min(1.0, dot_product)))
                        angle_deg = angle_rad * 180.0 / np.pi
                        
                        self.logger.info(
                            f"Direction change detected: {angle_deg:.1f}° change, "
                            f"damping velocities by factor {damping_factor}"
                        )
                        
            # Update previous direction for next cycle comparison
            self.prev_direction = current_dir
    
    def _apply_approach_scaling(self, linear_x, filtered_distance, desired_distance):
        """
        Makes the robot slow down naturally as it gets closer to the target.
        
        IMAGINE THIS: 🛑
        ---------------
        Think about how you would drive a car when approaching a stop sign:
        
        - When far away (100+ feet): Drive at normal speed
        - When getting closer (50-100 feet): Start gently slowing down
        - When very close (10-20 feet): Slow down more aggressively
        - If you overshoot: Quickly back up to the right position
        
        This method creates that same natural braking behavior for the robot,
        making it look smooth and intentional rather than jerky or robotic.
        
        HOW IT WORKS: 📉
        -------------
        1. GENTLE DECELERATION CURVE
           Instead of suddenly slowing down at a fixed distance, the robot
           gradually reduces speed using a special curve (quadratic) that
           feels natural - just like how humans naturally slow down.
        
        2. EMERGENCY BRAKING
           If the robot gets too close to the target (overshoots), it
           rapidly slows down or even backs up - similar to how you'd
           quickly brake if you rolled past a stop sign.
           
        3. ADAPTIVE TO CURRENT SPEED
           Just like you'd brake harder when traveling at high speed,
           the robot adjusts its deceleration based on how fast it's
           currently moving.
        
        Args:
            linear_x: How fast the robot wants to go forward (meters/second)
            filtered_distance: How far the robot is from the target (meters)
            desired_distance: How far the robot should stop from target (meters)
            
        Returns:
            A new forward speed that creates natural, smooth braking
        """
        try:
            # Calculate distance error (negative means robot is too close to target)
            raw_distance_error = filtered_distance - desired_distance  # Meters
            distance_error = abs(raw_distance_error)  # Absolute error
            
            # Emergency braking for negative distance errors (robot too close)
            # If robot is closer than desired_distance minus 5cm safety margin
            if raw_distance_error < -0.05:
                # Calculate emergency braking factor:
                # - Error = 0m → factor = 0 (full stop)
                # - Error = -0.05m → factor = 0.5 (half speed)
                # - Error = -0.1m → factor = 0 (full stop)
                emergency_factor = max(0.0, 1.0 + raw_distance_error * 10.0)
                scaled_velocity = linear_x * emergency_factor
                
                # Log emergency braking events
                if self.debug_level >= 1 and abs(scaled_velocity - linear_x) > 0.01:
                    self.logger.info(
                        f"Emergency approach reduction: error={raw_distance_error:.3f}m, "
                        f"factor={emergency_factor:.2f}, velocity={scaled_velocity:.3f}"
                    )
                return scaled_velocity
            
            # Apply progressive deceleration as robot approaches target
            # Start deceleration at approach_distance*3 (default 0.9m from target)
            if distance_error < self.approach_distance * 3.0:
                # Calculate approach scale with quadratic curve for smooth deceleration
                # Normalize distance to 0-1 range relative to approach distance
                normalized_distance = distance_error / self.approach_distance  # 1.0 = at approach boundary
                
                # Calculate approach factor with quadratic curve:
                # - At approach_distance*3 (0.9m): factor approaches 1.0 (full speed)
                # - At approach_distance (0.3m): factor = 1.0 (full speed)
                # - At approach_distance*0.5 (0.15m): factor ≈ 0.25 (25% speed)
                # - At approach_distance*0.0 (0.0m): factor = min_approach_factor (20% speed)
                # The squared curve creates gentle initial deceleration that increases as robot gets closer
                approach_factor = max(self.min_approach_factor, (normalized_distance)**2.0)
                
                # Apply stronger deceleration when very close (within half approach distance)
                if distance_error < self.approach_distance * 0.5:  # Within 15cm of target by default
                    approach_factor *= 0.5  # Further reduce to 50% of calculated value
                
                # Check closing speed using velocity history to create adaptive deceleration
                recent_velocities = self._get_recent_velocities(3)
                
                # Only calculate if we have velocity history
                if self.history_count > 0:
                    # Get average forward velocity from recent history
                    avg_forward_vel = np.mean(recent_velocities[:, 0])
                    
                    # Apply more aggressive deceleration if approaching target quickly
                    # High-speed approaches need more aggressive braking
                    if avg_forward_vel > 0.1 and distance_error < self.approach_distance * 1.2:
                        # Calculate speed-based reduction factor:
                        # - speeds < 0.1 m/s: no additional reduction
                        # - at 0.2 m/s: factor ≈ 0.8 (80% of normal approach factor)
                        # - at 0.4 m/s: factor = 0.6 (60% of normal approach factor)
                        # This creates more aggressive braking at higher speeds
                        speed_reduction = max(0.5, 1.0 - (avg_forward_vel / 0.4) * 0.4)
                        approach_factor *= speed_reduction
                        
                        # Log enhanced deceleration events
                        if self.debug_level >= 1:
                            self.logger.info(
                                f"Enhanced deceleration: speed={avg_forward_vel:.2f}m/s, "
                                f"additional factor={speed_reduction:.2f}"
                            )
                
                # Apply the scaling factor to forward velocity
                # Only apply if forward velocity is significant
                if abs(linear_x) > 0.01:
                    scaled_velocity = linear_x * approach_factor
                    
                    # Log significant approach scaling events
                    if self.debug_level >= 1 and abs(scaled_velocity - linear_x) > 0.02:
                        self.logger.info(
                            f"Approach scaling: distance_error={distance_error:.3f}m, "
                            f"factor={approach_factor:.2f}, velocity={scaled_velocity:.3f}m/s"
                        )
                        
                    return scaled_velocity
            
            # Return unmodified velocity if not in approach zone
            return linear_x
        except Exception as e:
            self.logger.error(f"Error in approach scaling: {str(e)}")
            return linear_x  # Return original value on error for safety
    
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
                                # Progressive braking curve - less aggressive
                                braking_factor = max(0.2, (time_to_target / 1.5)**1.2)  # Changed from 1.5 to 1.2 power
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
        """
        Apply acceleration limits to prevent jerky motion.
        
        This method:
        1. Calculates time elapsed since last control cycle
        2. Determines maximum allowed velocity change based on acceleration limits and dt
        3. Applies special handling for starting movement from a stop
        4. Limits velocity changes to prevent jerky motion
        
        Args:
            current_time: Current timestamp for calculating dt
        """
        try:
            # Calculate time since last control step (dt = delta time)
            dt = current_time - self.last_accel_time
            self.last_accel_time = current_time
            
            # Bound dt to reasonable values to handle timing anomalies
            # - Less than 0.001s (1ms): likely a timing error, use minimum
            # - More than 0.1s (100ms): unusually long delay, cap to avoid jumps
            dt = max(0.001, min(dt, 0.1))
            
            # Scale acceleration limits with dt to get max velocity change per axis
            # Formula: max_Δv = acceleration * dt
            accel_limits = self.acceleration_limits * dt
            
            # Safety check for NaN values (shouldn't happen but protect against math errors)
            if np.any(np.isnan(accel_limits)):
                self.logger.warning("NaN detected in acceleration limits, using defaults")
                # Default safe values for max Δv per cycle, assuming typical dt of ~0.05s:
                # [0.18 m/s per cycle, 0.15 m/s per cycle, 0.2 rad/s per cycle]
                accel_limits = np.array([0.18, 0.15, 0.2])
            
            # Calculate velocity differences between target and current
            np.subtract(self._target_velocities, self.last_cmd_vel, out=self._vel_diffs)
            
            # Apply acceleration limits for each axis (x, y, angular)
            for i in range(3):
                # Apply starting boost when beginning movement from stop
                # This makes the robot more responsive when starting from rest
                if abs(self.last_cmd_vel[i]) < 0.01 and abs(self._target_velocities[i]) > 0.01:
                    # Different boost factors by axis:
                    # - Forward (i=0): 2.0x boost (0->0.5m/s in ~0.14s vs normal 0.28s)
                    # - Lateral (i=1): 2.5x boost (0->0.4m/s in ~0.11s vs normal 0.27s)
                    # - Angular (i=2): 2.0x boost (0->0.6rad/s in ~0.15s vs normal 0.3s)
                    # These are reduced from original values [3.0, 5.0, 3.0] for smoother starts
                    boost = 2.0 if i == 0 else (2.5 if i == 1 else 2.0)
                    limit = accel_limits[i] * boost
                else:
                    limit = accel_limits[i]
                
                # Apply limit if velocity change exceeds maximum
                # This is the core acceleration limiting logic
                if abs(self._vel_diffs[i]) > limit:
                    # Limit the change to maximum allowed
                    # new_v = current_v + sign(Δv) * max_Δv
                    self._limited_velocities[i] = self.last_cmd_vel[i] + np.sign(self._vel_diffs[i]) * limit
                else:
                    # Change is within limits, allow full target velocity
                    self._limited_velocities[i] = self._target_velocities[i]
                
                # Safety check for NaN values in result (protect against math errors)
                if np.isnan(self._limited_velocities[i]):
                    self.logger.warning(f"NaN detected in velocity {i}, using previous value")
                    self._limited_velocities[i] = self.last_cmd_vel[i]
            
        except Exception as e:
            self.logger.error(f"Error applying acceleration limits: {str(e)}")
            # Copy target velocities directly on error (fall back to desired values)
            np.copyto(self._limited_velocities, self._target_velocities)
    
    def _apply_progressive_ramping(self):
        """
        Apply progressive velocity ramping to prevent velocity spikes.
        
        This method implements adaptive velocity ramping where:
        - Small changes are applied more quickly
        - Large changes are applied more gradually
        - The ramping factor scales inversely with the magnitude of velocity change
        
        This creates smoother transitions when velocities need to change
        significantly, preventing the sudden velocity jumps that cause jerky motion.
        """
        try:
            # Apply ramping to each velocity component (x, y, angular)
            for i in range(3):
                # Calculate absolute velocity difference between target and current
                vel_diff = abs(self._limited_velocities[i] - self.last_cmd_vel[i])
                
                # Only apply ramping for changes larger than threshold
                # (small changes don't need ramping)
                if vel_diff > self.ramp_threshold:  # threshold = 0.05 m/s or rad/s
                    # Calculate adaptive ramping factor based on magnitude of change:
                    # - For smaller changes: factor approaches max_ramp_factor (0.6)
                    # - For larger changes: factor decreases (slower transition)
                    # The formula creates an inverse relationship between change size and ramp speed
                    base_ramp = self.max_ramp_factor  # 0.6 = default maximum ramping rate
                    
                    # This formula creates a curve where:
                    # - vel_diff of 0.05 m/s → factor ≈ 0.5
                    # - vel_diff of 0.20 m/s → factor ≈ 0.25
                    # - vel_diff of 0.50 m/s → factor ≈ 0.12
                    ramp_factor = base_ramp / (1.0 + vel_diff * 2.0)
                    
                    # Ensure factor stays within reasonable bounds
                    # - Minimum 0.1 (10% of change per cycle) for very large changes
                    # - Maximum is base_ramp (default 0.6 = 60% of change per cycle)
                    ramp_factor = max(0.1, min(base_ramp, ramp_factor))
                    
                    # Apply weighted average for smooth transition:
                    # new_v = current_v + (target_v - current_v) * ramp_factor
                    # Example: If ramp_factor is 0.2, apply 20% of the desired change this cycle
                    self._limited_velocities[i] = self.last_cmd_vel[i] + (self._limited_velocities[i] - self.last_cmd_vel[i]) * ramp_factor
                    
                    # Log significant velocity ramps to help with debugging
                    if self.debug_level >= 2 and vel_diff > 0.1:
                        axis_name = "x" if i == 0 else ("y" if i == 1 else "θ")
                        self.logger.info(
                            f"Progressive ramping: {axis_name}={self.last_cmd_vel[i]:.2f} → {self._limited_velocities[i]:.2f} "
                            f"(diff: {vel_diff:.2f}, factor: {ramp_factor:.2f})"
                        )
        except Exception as e:
            self.logger.error(f"Error in progressive ramping: {str(e)}")
            # Continue without ramping on error - fallback to acceleration-limited values
    
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
        """
        Limit combined lateral and angular movement to prevent instability.
        
        This method prevents combinations of movements that could cause physical
        instability in the robot:
        
        1. When both lateral movement and rotation occur simultaneously,
           it can create tipping forces on the robot
        2. Larger movements create stronger destabilizing forces
        3. Selective reduction of lateral velocity maintains rotational response
           while preventing potential tipping
        
        The scaling is progressive, applying stronger reductions for more
        extreme movement combinations.
        """
        try:
            # Check if both lateral and angular velocities exceed stability thresholds
            # 0.15 m/s lateral = significant sideways motion
            # 0.3 rad/s angular = significant rotation (~17 degrees/second)
            if abs(self._limited_velocities[1]) > 0.15 and abs(self._limited_velocities[2]) > 0.3:
                lateral_magnitude = abs(self._limited_velocities[1])
                angular_magnitude = abs(self._limited_velocities[2])
                
                # Apply stronger reduction for larger combined movements
                # Large lateral (>0.2 m/s) + large angular (>0.4 rad/s) creates high risk of tipping
                if lateral_magnitude > 0.2 and angular_magnitude > 0.4:
                    # Significant reduction for high-risk scenarios
                    # Reduce lateral velocity to 70% to prevent tipping
                    # We modify lateral rather than angular to maintain responsive turning
                    self._limited_velocities[1] *= 0.7  # 70% of calculated lateral velocity
                    
                    if self.debug_level >= 2:
                        self.logger.info(
                            f"Strong combined movement limiting: lateral={lateral_magnitude:.2f}m/s, "
                            f"angular={angular_magnitude:.2f}rad/s, applying 70% lateral reduction"
                        )
                else:
                    # Adaptive scaling for moderate combined movements
                    # Formula creates a curve where:
                    # - At lateral=0.15 m/s: scale ≈ 0.95 (minimal reduction)
                    # - At lateral=0.2 m/s: scale ≈ 0.85 (moderate reduction)
                    # This progressively increases reduction as lateral velocity increases
                    scale_factor = 0.85 + (0.15 * (1.0 - min(1.0, lateral_magnitude / 0.2)))
                    
                    # Apply calculated scale factor with bounds for safety
                    # - Minimum 0.8 (80% of original) for moderate reduction
                    # - Maximum 0.95 (95% of original) for minimal reduction
                    final_scale = min(0.95, max(0.8, scale_factor))
                    self._limited_velocities[1] *= final_scale
                    
                    if self.debug_level >= 3:
                        self.logger.info(
                            f"Moderate combined movement limiting: applying {final_scale:.2f} lateral scale"
                        )
            
        except Exception as e:
            self.logger.error(f"Error limiting combined movements: {str(e)}")
            # Leave velocity values unchanged on error (fail-safe approach)
    
    def _apply_velocity_smoothing(self):
        """
        Apply velocity smoothing using a sliding window approach.
        
        This method implements a weighted moving average smoothing that:
        1. Uses recent velocity history to create a smoother trajectory
        2. Applies smoothing only when appropriate (not during rapid changes)
        3. Blends current velocity commands with historical average
        
        The weighted average gives more importance to recent commands and
        less to older ones, creating natural motion that reduces oscillations
        and small jitters while preserving responsiveness.
        """
        try:
            # Only apply smoothing if we have enough history (minimum 3 samples)
            if self.history_count >= 3:
                # Get recent velocity commands from history buffer
                recent_vels = self._get_recent_velocities(3)
                
                # Create a weighted average with exponential decay
                # - Most recent: 50% weight (highest importance)
                # - Second most recent: 30% weight
                # - Third most recent: 20% weight
                # This prioritizes recent commands while still considering history
                weights = np.array([0.5, 0.3, 0.2])
                weighted_avg = np.zeros(3)  # Initialize weighted average vector
                
                # Calculate weighted average of recent velocities
                for i in range(min(3, len(recent_vels))):
                    weighted_avg += weights[i] * recent_vels[i]
                
                # Only apply smoothing if velocities aren't changing drastically
                # We skip smoothing during rapid changes to maintain responsiveness
                # Calculate maximum velocity change across all axes
                vel_change = np.max(np.abs(self._limited_velocities - self.last_cmd_vel))
                
                # Apply smoothing only when changes are moderate
                # 0.15 m/s or rad/s threshold represents significant but not drastic change
                if vel_change < 0.15:
                    # Blend current velocities with smoothed history using weighted average:
                    # final_v = current_v * blend_factor + history_avg * (1-blend_factor)
                    
                    # 0.8 = 80% current velocity, 20% historical average
                    # This preserves responsiveness while reducing jitter
                    blend_factor = 0.8
                    
                    # Apply the blending to all velocity components
                    self._limited_velocities = self._limited_velocities * blend_factor + weighted_avg * (1.0 - blend_factor)
                    
                    # Log smoothing application at high debug levels
                    if self.debug_level >= 3:
                        self.logger.info(f"Applied velocity smoothing, blend: {blend_factor:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error in velocity smoothing: {str(e)}")
            # Continue without smoothing on error - fallback to acceleration-limited values
    
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

    def _add_to_history(self, vel_tuple):
        # Accepts tuple or array, always stores as numpy array
        if not isinstance(vel_tuple, np.ndarray):
            vel_tuple = np.array(vel_tuple, dtype=np.float32)
        if not hasattr(self, 'velocity_history') or not isinstance(self.velocity_history, list):
            self.velocity_history = []
        self.velocity_history.append(tuple(vel_tuple))
        if len(self.velocity_history) > self.history_size:
            self.velocity_history.pop(0)

    def get_velocity_history(self):
        """Return the velocity history for analysis."""
        # Convert numpy array to list of tuples for compatibility
        if self.history_count == 0:
            return []
            
        history_list = []
        for i in range(min(self.history_count, self.buffer_size)):
            idx = (self.history_index - i - 1) % self.buffer_size
            history_list.append(tuple(self.velocity_history[idx]))
            
        return history_list

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
                        from ball_chase.pid.pid_helpers import Matrix4x4
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
