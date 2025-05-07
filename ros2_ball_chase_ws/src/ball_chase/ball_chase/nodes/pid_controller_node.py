"""
Basketball Tracking Robot - Optimized PID Controller Node
=======================================================

ARCHITECTURAL OVERVIEW & TEACHING GUIDE
---------------------------------------

This controller implements a sophisticated robot control system for tracking and following
a basketball using a mecanum-wheeled robot. This document explains both WHAT the system
does and HOW it works, with a focus on teaching robotics programming concepts.

System Architecture Diagram:
--------------------------

                          ┌─────────────────────────┐
                          │ ROS2 Environment        │
                          └───────────┬─────────────┘
                                      │
                                      ▼
        ┌─────────┐         ┌─────────────────────┐
        │ YOLO    │───┐     │                     │
        │ Camera  │   │     │ ┌─────────────────┐ │         ┌─────────────────┐
        └─────────┘   │     │ │ Target Tracking │ │         │                 │
                      │     │ │ Module          │ │         │  PID Control    │
        ┌─────────┐   │ ┌──►│ │ - Filtering    ├─┼────────►│  Module         │
        │ LIDAR   │───┼─┘   │ │ - Prediction   │ │         │  - Error Calc   │
        └─────────┘   │     │ └─────────────────┘ │         │  - PID Compute  │
                      │     │                     │         └────────┬────────┘
        ┌─────────┐   │     │ ┌─────────────────┐ │                  │
        │ 3D Depth│───┼────►│ │ State Manager   │ │                  │
        │ Camera  │   │     │ │ - Robot State   │ │                  ▼
        └─────────┘   │     │ │ - Freshness     │ │         ┌─────────────────┐
                      │     │ │ - Recovery      │ │         │                 │
        ┌─────────┐   │     │ └─────────────────┘ │         │ Strategy Module │
        │ Fusion  │───┘     │                     │         │ - Chooses       │
        └─────────┘         │ ┌─────────────────┐ │         │   Movement     │
                            │ │ Transform System│ │         │   Strategy     │
        ┌─────────┐         │ └─────────────────┘ │         └────────┬────────┘
        │ External│         │  PIDControllerNode  │                  │
        │ IMU     │────────►└─────────────────────┘                  │
        └─────────┘                   │                              │
                                      ▼                              ▼
                            ┌─────────────────────┐         ┌─────────────────┐
                            │ Performance Monitor │         │                 │
                            │ - CPU Load          │         │ Velocity Module │
                            │ - Adaptation        │         │ - Coordination  │
                            └─────────────────────┘         │ - Safety Limits │
                                                            └────────┬────────┘
                                                                     │
                                                                     ▼
                                                            ┌─────────────────┐
                                                            │ Robot Motors    │
                                                            │ (cmd_vel output)│
                                                            └─────────────────┘

Key Concepts for Beginners:
-------------------------

1. SENSE-THINK-ACT LOOP

   ┌───────────────────────────────────────────────────────────────────────────┐
   │                     THE ROBOTICS CONTROL CYCLE                           │
   └───────────────────────────────────────────────────────────────────────────┘
                                     │
          ┌─────────────────────────┼─────────────────────────┐
          │                         │                         │
          ▼                         ▼                         ▼
    ┌────────────┐           ┌────────────┐           ┌────────────┐
    │   SENSE    │           │   THINK    │           │    ACT     │
    └──────┬─────┘           └──────┬─────┘           └──────┬─────┘
           │                        │                        │
           ▼                        ▼                        ▼
    ┌────────────┐           ┌────────────┐           ┌────────────┐
    │ Get data   │           │ Process    │           │ Send       │
    │ from       │───────────│ data and   │───────────│ commands   │
    │ sensors    │           │ make       │           │ to motors  │
    └────────────┘           │ decisions  │           └────────────┘
    • Camera                 └────────────┘           • Forward/back
    • LIDAR                  • Filter data            • Side-to-side
    • Depth                  • Calculate errors       • Rotation
    • IMU                    • Choose strategy
                            • Compute velocities
    
   This controller runs this loop 10-40 times per second!

2. WHAT IS PID CONTROL?

   ┌───────────────────────────────────────────────────────────────────────────┐
   │                 PID CONTROL: THREE WAYS TO RESPOND                       │
   └───────────────────────────────────────────────────────────────────────────┘
    
    Target Position                                 Current Position
         ┌───┐                                           ┌───┐
         │ T │                                           │ R │
         └───┘                                           └───┘
           │                                               │
           └───────────────────┬───────────────────────────┘
                               │
                               ▼
                            ERROR
    
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │ P: Proportional │   │  I: Integral    │   │  D: Derivative  │
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                     │
             ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │"How far am I    │   │"Have I been     │   │"Am I approaching│
    │ from target     │   │ stuck away from │   │ the target too  │
    │ right now?"     │   │ target too long?│   │ quickly?"       │
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                     │
             ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │Like a spring:   │   │Gradually builds │   │Acts as a brake: │
    │The further away,│   │force if error   │   │Slows down as you│
    │the stronger     │   │persists, helping│   │approach to      │
    │the pull         │   │overcome friction│   │prevent overshoot│
    └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
             │                     │                     │
             └───────────────────┬─────────────────────┬┘
                                 │                     │
                                 ▼                     ▼
                        ┌─────────────────────────────────┐
                        │           COMBINED              │
                        │      OUTPUT = P + I + D        │
                        └─────────────────────────────────┘

   These three terms are combined with specific "gains" (multiplication factors)
   to produce smooth, accurate movement.

3. WORKING WITH MULTIPLE DIMENSIONS

   ┌───────────────────────────────────────────────────────────────────────────┐
   │                      MECANUM WHEEL ROBOT MOVEMENT                        │
   └───────────────────────────────────────────────────────────────────────────┘
    
    Our robot has 3 movement dimensions with special "mecanum wheels":
    
    ┌────────────────────────┐      ┌────────────────────────┐
    │  LINEAR X (FORWARD)    │      │    LINEAR Y (SIDE)     │
    │                        │      │                        │
    │      ┌───────┐         │      │      ┌───────┐         │
    │     ↑│ ROBOT │         │      │      │ ROBOT │→        │
    │      └───────┘         │      │      └───────┘         │
    │                        │      │                        │
    └────────────────────────┘      └────────────────────────┘
    
                      ┌────────────────────────┐
                      │    ANGULAR Z (TURN)    │
                      │                        │
                      │      ┌───────┐         │
                      │      │ ROBOT │↻        │
                      │      └───────┘         │
                      │                        │
                      └────────────────────────┘
    
    Each dimension has its own PID controller, but they work together!
    
    • Forward/backward controller helps get proper distance to ball
    • Side-to-side controller keeps ball centered in front of robot
    • Rotational controller helps robot face directly toward ball

4. ERROR CALCULATION VISUALIZATION

   ┌───────────────────────────────────────────────────────────────────────────┐
   │                    THE THREE TRACKING ERRORS                            │
   └───────────────────────────────────────────────────────────────────────────┘
    
    TOP VIEW:                                  Desired position
                                                    ●
                                                    │
    Angular                                         │
    Error       ╲                                   │
      ╲         ╲                                   │
       ╲ θ       ╲                                  │
        ╲         ╲   Distance                      │
         ╲         ╲   Error                        │
          ╲         ╲                               │
           ▼          ╲                             │
        ┌─────┐        ╲                           ╱│╲ Lateral
        │Robot│─────────●                         ╱ │ ╲ Error
        └─────┘          Basketball               ╱  │  ╲
    
    • Distance Error: How far from desired distance to the ball
      (wants to be at a specific distance - not too close, not too far)
    
    • Lateral Error: How far left/right from being centered with the ball
      (wants the ball to be directly in front)
    
    • Angular Error: How far rotated from facing directly at the ball
      (wants to be pointed straight at the ball)

5. STRATEGY SELECTION LOGIC

   ┌───────────────────────────────────────────────────────────────────────────┐
   │                    CHOOSING THE RIGHT STRATEGY                           │
   └───────────────────────────────────────────────────────────────────────────┘
    
          ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
          │ Distance     │    │ Lateral      │    │ Angular      │
          │ Error        │    │ Error        │    │ Error        │
          └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
                 │                   │                   │
                 ▼                   ▼                   ▼
          ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
          │ Categorize:  │    │ Categorize:  │    │ Categorize:  │
          │ none         │    │ none         │    │ none         │
          │ very_small   │    │ very_small   │    │ very_small   │
          │ small        │    │ small        │    │ small        │
          │ medium       │    │ medium       │    │ medium       │
          │ large        │    │ large        │    │ large        │
          │ very_large   │    │ very_large   │    │ very_large   │
          └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
                 │                   │                   │
                 └───────────┬───────┴───────────┬───────┘
                             │                   │
                             ▼                   ▼
                      ┌─────────────────────────────────┐
                      │        Strategy Lookup          │
                      │                                 │
                      │ ("large", "*", "small")  →     │
                      │    APPROACH_WITH_ALIGNMENT      │
                      │                                 │
                      │ ("*", "large", "*")  →         │
                      │    LATERAL_PRIORITY             │
                      │                                 │
                      │ ("*", "*", "large")  →         │
                      │    ANGULAR_PRIORITY             │
                      └─────────────┬───────────────────┘
                                    │
                                    │
                                    ▼
                      ┌─────────────────────────────────┐
                      │        Selected Strategy        │
                      │                                 │
                      │  ✓ Use forward?    True         │
                      │  ✓ Use lateral?    True         │
                      │  ✓ Use angular?    True         │
                      │                                 │
                      │  ✓ Forward scale:  0.7          │
                      │  ✓ Lateral scale:  0.9          │
                      │  ✓ Angular scale:  0.4          │
                      └─────────────────────────────────┘

6. COORDINATED CONTROL VISUALIZATION

   ┌───────────────────────────────────────────────────────────────────────────┐
   │              COORDINATED MOVEMENT: WORKING TOGETHER                      │
   └───────────────────────────────────────────────────────────────────────────┘
    
    Instead of treating each movement independently (which can look robotic),
    our controller coordinates them - just like how humans naturally move!
    
    SCENARIO: Basketball is ahead and to the right
    
    UNCOORDINATED:                      COORDINATED:
    ┌─────────────────────┐            ┌─────────────────────┐
    │                     │            │                     │
    │        Ball         │            │        Ball         │
    │         ●           │            │         ●           │
    │                     │            │                     │
    │                     │            │         ↗           │
    │                     │            │        /            │
    │  ┌───┐    →         │            │  ┌───┐              │
    │  │ R │              │            │  │ R │              │
    │  └───┘              │            │  └───┘              │
    │    ↑                │            │                     │
    │    |                │            │                     │
    └─────────────────────┘            └─────────────────────┘
    Robot tries to move forward,       Robot creates a smooth diagonal
    sideways, and turn all at          path, reducing forward speed
    once - jerky movement!             during turning - natural movement!
    
    COORDINATED CONTROL BENEFITS:
    • More efficient paths to target
    • Smoother, more natural movement
    • Reduced mechanical stress
    • Lower energy consumption
    • Better tracking performance

7. DATA FRESHNESS AND SAFETY SYSTEM

   ┌───────────────────────────────────────────────────────────────────────────┐
   │                  DATA FRESHNESS SAFETY SYSTEM                           │
   └───────────────────────────────────────────────────────────────────────────┘
    
                           ┌──────────────────┐
                           │   Last Update    │◀───── New sensor data
                           │   Timestamp      │       arrives
                           └────────┬─────────┘
                                    │
                                    ▼
                           ┌──────────────────┐
                           │ Calculate data   │
                           │ age = current    │
                           │ time - timestamp │
                           └────────┬─────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────────┐
                    │       How old is the data?       │
                    └──────────────────────────────────┘
                      /            │             \
                     /             │              \
                    ▼              ▼               ▼
           ┌────────────┐  ┌────────────┐  ┌────────────┐
           │   FRESH    │  │   STALE    │  │  CRITICAL  │
           │  (< 0.7s)  │  │(0.7s-1.0s) │  │  (> 1.0s)  │
           └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
                  │               │               │
                  ▼               ▼               ▼
         ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
         │ Normal      │  │ Reduce      │  │ STOP        │
         │ operation   │  │ speed to    │  │ completely  │
         │             │  │ 50%         │  │ for safety  │
         └─────────────┘  └─────────────┘  └─────────────┘
    
    This is essential for safety - if we haven't received sensor data recently,
    we don't really know where the ball is, so we slow down or stop!

8. STATE MANAGEMENT DIAGRAM

   ┌───────────────────────────────────────────────────────────────────────────┐
   │                      ROBOT STATE MACHINE                                 │
   └───────────────────────────────────────────────────────────────────────────┘
    
                         ┌───────────────┐
                 ┌───────│ INITIALIZING  │
                 │       └───────┬───────┘
                 │               │ Components ready
                 │               ▼
                 │       ┌───────────────┐  No ball found
                 │       │  SEARCHING    │◀────────────┐
                 │       └───────┬───────┘             │
                 │               │ Ball found          │
                 │               ▼                     │
    Error        │       ┌───────────────┐             │
    detected     │       │   TRACKING    │─────────────┘
    ┌────────────┼───────┤               │   Ball lost
    │            │       └───────┬───────┘
    │            │               │ At target position
    │            │               ▼
    │            │       ┌───────────────┐
    │            └───────┤    STOPPED    │
    │                    └───────┬───────┘
    │                            │
    │                            │ Critical error
    ▼                            ▼
    ┌───────────────┐     ┌───────────────┐
    │   RECOVERY    │     │   EMERGENCY   │
    │               │     │     STOP      │
    └───────────────┘     └───────────────┘
    
    Each state has specific behaviors and transitions, allowing the robot
    to react appropriately to different situations.

9. PERFORMANCE OPTIMIZATION

   ┌───────────────────────────────────────────────────────────────────────────┐
   │                 ADAPTIVE PERFORMANCE MANAGEMENT                          │
   └───────────────────────────────────────────────────────────────────────────┘
    
    On resource-constrained platforms like Raspberry Pi, the system automatically 
    adjusts to available resources:
    
    ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
    │     CPU USAGE     │     │ CONTROL FREQUENCY │     │ CONTROL METHOD    │
    │                   │     │                   │     │                   │
    │  ┌───┬───┬───┬───┤     │    Updates/sec    │     │  ┌───────────┐    │
    │  │███│███│   │   │     │      ┌────┐       │     │  │SIMPLIFIED │    │
    │  │███│███│   │   │     │      │    │       │     │  └───────────┘    │
    │  │███│███│   │   │     │   ┌──┘    └──┐    │     │  ┌───────────┐    │
    │  │███│███│   │   │     │   │         │    │     │  │FULL PID    │    │
    │  └───┴───┴───┴───┘     │   └─────────┘    │     │  └───────────┘    │
    │   25% 50% 75% 100%     │    1  2  4  8    │     │  Simple  Complex  │
    └───────┬───────────┘     └───────┬───────────┘     └───────┬───────────┘
            │                         │                         │
            └─────────────────────────┼─────────────────────────┘
                                      │
                               ┌──────▼──────┐
                               │             │
                               │  ADAPTIVE   │
                               │  CONTROL    │
                               │             │
                               └─────────────┘
    
    • High CPU usage → Lower control frequency, simpler calculations
    • Low CPU usage → Higher control frequency, more sophisticated control
    • Control rate detection → Synchronizes with sensor fusion rate
    • Memory optimization → Object pools, pre-allocation
    
    This maintains reliable operation even under system load!

Key Features:
-----------
- Angular-first control strategy for diagonal movements
- Fast strategy transitions for responsive tracking
- Enhanced integral term management to prevent windup
- Coordinated angular-lateral control with balanced parameters
- Balanced error thresholds and hysteresis for smooth behavior
- Data freshness detection with graduated response
- Fusion rate detection and control rate adaptation
- Continuous motion tracking with trajectory prediction
- Optimized for Raspberry Pi 5 performance with resource monitoring
- Performance optimizations for CPU-constrained environments
"""

# Import ROS2 and standard Python modules
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PointStamped, Twist, Vector3Stamped, Vector3, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32MultiArray
from tf2_ros import TransformListener, Buffer
from rclpy.logging import LoggingSeverity
import math
import time
import numpy as np
import signal
import sys
from collections import deque
import logging
import traceback
from abc import ABC, abstractmethod

# Import custom modules for PID helpers, filtering, computation, and tracking
from ball_chase.pid.pid_helpers import LightweightBuffer, CircularBuffer, ThrottledLogger, FastTrigonometry, ResourceMonitor
from ball_chase.pid.pid_target_filter import EnhancedTargetFilter, ErrorTracker
from ball_chase.pid.pid_computation import PIDControllers
from ball_chase.pid.pid_target_tracking import TargetTrackingModule, MovementStrategyModule, VelocityControlModule, TransformSystem
from ball_chase.pid.pid_target_tracking import RecoveryBehaviorModule, TransformStatus

# Configure Python logging for this node
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('pid_controller')
throttled_logger = ThrottledLogger(logger)

# Define topic names for input and output
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

# Throttling intervals for log messages (in seconds)
LOG_THROTTLE_CONTROL = 2.0     # Control loop status logs
LOG_THROTTLE_STATE = 0.5       # State change logs
LOG_THROTTLE_DIAG = 2.0        # Diagnostic logs


# Centralized object pool manager for efficient message reuse
class ObjectPoolManager:
    """
    PURPOSE: Manages reusable object pools to minimize memory allocations.
    
    This class creates and manages collections of pre-allocated ROS2 message objects
    (like Twist, Vector3, etc.) that can be reused instead of creating new objects
    for each message. This significantly reduces memory allocations and garbage
    collection overhead, improving performance on resource-constrained systems
    like the Raspberry Pi 5.
    """
    """Manages pools of reusable objects to reduce memory allocations."""
    def __init__(self, max_twist=10, max_vector3=15, max_float32multiarray=5, ttl=60.0):
        from geometry_msgs.msg import Twist, Vector3, PointStamped, Vector3Stamped, TransformStamped
        from nav_msgs.msg import Odometry
        from std_msgs.msg import Float32MultiArray, String
        from ball_chase.pid.pid_helpers import GenericObjectPool

        def reset_twist(twist):
            twist.linear.x = twist.linear.y = twist.linear.z = 0.0
            twist.angular.x = twist.angular.y = twist.angular.z = 0.0

        def reset_vector3(vec):
            vec.x = vec.y = vec.z = 0.0

        def reset_float32multiarray(arr):
            arr.data.clear()

        def reset_pointstamped(msg):
            msg.point.x = msg.point.y = msg.point.z = 0.0
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            msg.header.frame_id = ""

        def reset_vector3stamped(msg):
            msg.vector.x = msg.vector.y = msg.vector.z = 0.0
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            msg.header.frame_id = ""

        def reset_transformstamped(msg):
            msg.header.stamp.sec = 0
            msg.header.stamp.nanosec = 0
            msg.header.frame_id = ""
            msg.child_frame_id = ""
            msg.transform.translation.x = 0.0
            msg.transform.translation.y = 0.0
            msg.transform.translation.z = 0.0
            msg.transform.rotation.x = 0.0
            msg.transform.rotation.y = 0.0
            msg.transform.rotation.z = 0.0
            msg.transform.rotation.w = 1.0

        def reset_odometry(msg):
            msg.pose.pose.position.x = 0.0
            msg.pose.pose.position.y = 0.0
            msg.pose.pose.position.z = 0.0
            msg.twist.twist.linear.x = 0.0
            msg.twist.twist.linear.y = 0.0
            msg.twist.twist.linear.z = 0.0
            # ...reset other fields as needed...

        def reset_string(msg):
            msg.data = ""

        self.pools = {
            'Twist': GenericObjectPool(Twist, max_twist, reset_twist, ttl),
            'Vector3': GenericObjectPool(Vector3, max_vector3, reset_vector3, ttl),
            'Float32MultiArray': GenericObjectPool(Float32MultiArray, max_float32multiarray, reset_float32multiarray, ttl),
            'PointStamped': GenericObjectPool(PointStamped, 10, reset_pointstamped, ttl),
            'Vector3Stamped': GenericObjectPool(Vector3Stamped, 10, reset_vector3stamped, ttl),
            'TransformStamped': GenericObjectPool(TransformStamped, 5, reset_transformstamped, ttl),
            'Odometry': GenericObjectPool(Odometry, 5, reset_odometry, ttl),
            'String': GenericObjectPool(String, 10, reset_string, ttl),
        }

    def get(self, msg_type):
        return self.pools[msg_type].get() if msg_type in self.pools else None

    def put(self, msg_type, obj):
        if msg_type in self.pools:
            self.pools[msg_type].put(obj)

    def get_stats(self):
        return {k: v.stats() for k, v in self.pools.items()}


# Custom exception for initialization errors
class InitializationError(Exception):
    """
    PURPOSE: Custom exception for initialization failures.
    
    This exception is thrown when component initialization fails, providing
    clearer error reporting and allowing specific handling of initialization
    problems as distinct from other runtime errors.
    """
    """Exception raised when initialization fails."""
    pass

# Helper function for consistent error handling during initialization
def handle_initialization_error(logger, message, original_exception=None):
    """Handle initialization errors consistently."""
    error_message = f"{message}"
    if original_exception:
        error_message += f": {str(original_exception)}"
    logger.error(error_message)
    raise InitializationError(error_message) from original_exception

# ParameterManager handles all ROS2 parameters for the node
class ParameterManager:
    """
    Handles parameter declaration and retrieval for the PID Controller Node.
    
    In robotics, parameters allow us to tune the robot's behavior without changing code.
    This class centralizes all parameter management for the PID controller, providing:
    1. Default values suitable for initial operation
    2. Access to parameters through a clean interface
    3. Organization of related parameters into logical groups
    
    Parameters fall into several categories:
    - PID control gains: Determine how robot responds to errors
    - Movement limits: Constrain maximum speeds for safety
    - Thresholds: Define when robot should start/stop moving
    - Performance adaptations: Adjust behavior based on system load
    - Strategy parameters: Control movement coordination behaviors
    
    ROS2 parameter system allows these values to be:
    - Set at launch time via launch files
    - Changed dynamically during operation
    - Saved and loaded from parameter files
    """
    def __init__(self, node):
        """
        Initialize parameter manager with reference to parent node.
        
        Args:
            node: The ROS2 node that will declare and own these parameters
        """
        self.node = node
        self._declare_parameters()
        self._get_parameters()

    def _declare_parameters(self):
        """
        Declare all parameters with their default values.
        
        This method defines the complete set of parameters used by the PID controller.
        Each parameter includes a default value that provides reasonable behavior
        without additional tuning.
        """
        self.node.declare_parameters(
            namespace='',
            parameters=[
                # ========== PID CONTROLLER GAINS ==========
                # Forward motion control (X-axis)
                # These gains affect how the robot moves forward/backward toward the ball
                ('linear_x_kp', 1.0),      # Proportional gain: How strongly to respond to distance errors
                                           # Higher values make the robot move more aggressively toward the target
                                           # Typical range: 0.5-2.0

                ('linear_x_ki', 0.05),     # Integral gain: Helps overcome persistent errors (like friction)
                                           # Accumulates error over time to ensure the robot reaches its target
                                           # Typical range: 0.01-0.1
                
                ('linear_x_kd', 0.3),      # Derivative gain: Provides damping to prevent oscillation
                                           # Predicts error trends to smooth the approach
                                           # Typical range: 0.1-0.5
                
                ('linear_x_min', 0.0),     # Minimum forward velocity command (m/s)
                                           # Zero allows the robot to stop completely
                
                ('linear_x_max', 0.1),     # Maximum forward velocity command (m/s)
                                           # Limits top speed for safety and control
                                           # For a small robot, 0.1-0.5 m/s is reasonable
                
                # Lateral motion control (Y-axis)
                # These gains affect how the robot moves side-to-side to align with the ball
                ('linear_y_kp', 0.7),      # Proportional gain for lateral movement
                                           # Lower than X because lateral movement is typically less stable
                                           # Typical range: 0.5-1.0
                
                ('linear_y_ki', 0.005),    # Integral gain for lateral movement
                                           # Very low to prevent side-to-side oscillation
                                           # Typical range: 0.001-0.01
                
                ('linear_y_kd', 0.7),      # Derivative gain for lateral movement
                                           # Higher to dampen lateral oscillations effectively
                                           # Typical range: 0.5-1.0
                
                ('linear_y_min', -0.2),    # Minimum lateral velocity command (m/s)
                                           # Negative values allow movement to the right
                
                ('linear_y_max', 0.3),     # Maximum lateral velocity command (m/s)
                                           # Limits side-to-side speed for stability
                
                # Rotational motion control (angular Z-axis)
                # These gains affect how the robot rotates to face the ball
                ('angular_kp', 0.8),       # Proportional gain for rotational movement
                                           # How quickly the robot turns to face the target
                                           # Typical range: 0.5-1.0
                
                ('angular_ki', 0.01),      # Integral gain for rotational movement
                                           # Very low to prevent overshooting target heading
                                           # Typical range: 0.005-0.02
                
                ('angular_kd', 0.9),       # Derivative gain for rotational movement
                                           # High to prevent rotational oscillation (wobbling)
                                           # Typical range: 0.7-1.2
                
                ('angular_min', -0.5),     # Minimum angular velocity command (rad/s)
                                           # Negative values allow counterclockwise rotation
                
                ('angular_max', 0.5),      # Maximum angular velocity command (rad/s)
                                           # ~30 degrees per second to prevent dizzying rotation
                
                # ========== TARGET POSITIONING ==========
                ('min_distance', 0.9),     # Minimum desired distance from target (meters)
                                           # How close the robot will try to get to the ball
                
                ('max_distance', 2.0),     # Maximum tracking distance (meters)
                                           # Beyond this, the robot may use different behavior
                
                ('target_offset_x', 0.0),  # X offset from target center (meters)
                                           # Allows tracking offset from center (e.g., to follow behind)
                
                ('target_offset_y', 0.0),  # Y offset from target center (meters)
                                           # Allows tracking offset from center (e.g., to follow from side)
                
                # ========== TIMING PARAMETERS ==========
                ('target_update_rate', 3.0),  # Target position update frequency (Hz)
                                             # How often to process new target positions
                                             # Higher values: more responsive but more CPU usage
                
                ('diagnostics_rate', 0.5),   # Diagnostics publish frequency (Hz)
                                            # How often to publish diagnostic data
                                            # Lower values reduce bandwidth usage
                
                # ========== DEBUG SETTINGS ==========
                ('debug_level', 1),          # Debug verbosity level (0-3)
                                            # 0: Errors only
                                            # 1: Basic info
                                            # 2: Detailed info
                                            # 3: All debug data (very verbose)
                
                # ========== CONTROL BEHAVIOR SETTINGS ==========
                ('adaptive_gains', True),    # Enable adaptive PID gain adjustment
                                            # Automatically adjusts gains based on error magnitude
                                            # Makes control more stable at different distances
                
                ('use_lateral_control', True),  # Enable lateral (side-to-side) movement
                                              # Mecanum wheels allow sideways motion
                                              # Disable for differential drive robots
                
                # These thresholds determine when the robot considers itself "at target"
                # and can stop moving
                ('distance_threshold', 0.1),    # Distance error threshold (meters)
                                               # Robot stops if closer than this
                
                ('lateral_threshold', 0.02),    # Lateral error threshold (meters)
                                               # Robot stops lateral movement if aligned within this
                
                ('angular_threshold', 1.8),     # Angular error threshold (degrees)
                                               # Robot stops rotation if aligned within this
                
                ('angular_at_target_factor', 1.0),  # Factor to relax angular control when at target distance
                                                  # Makes robot less concerned with perfect alignment when close
                
                # ========== RESOURCE OPTIMIZATION SETTINGS ==========
                ('adaptive_control_rate', True),  # Adjust control frequency based on CPU load
                                                # Reduces processing frequency when CPU is high
                
                ('enable_resource_monitoring', True),  # Monitor CPU, memory, and temperature
                                                     # Adapts behavior to prevent overloading
                
                ('cpu_high_threshold', 80.0),  # CPU usage threshold (%) to reduce processing
                                             # Above this, control rate is reduced
                
                ('cpu_low_threshold', 50.0),   # CPU usage threshold (%) to return to normal processing
                                             # Below this, control rate returns to normal
                
                # ========== TRANSFORM OPTIMIZATION ==========
                ('enable_transform_caching', True),  # Cache coordinate transforms to reduce CPU usage
                                                   # Transforms between coordinate frames are expensive
                
                ('transform_cache_ttl', 1.0),  # Transform cache time-to-live (seconds)
                                             # How long to keep transforms before recalculating
                
                # ========== MOVEMENT STRATEGY SETTINGS ==========
                ('angular_first_control', True),  # Prioritize angular alignment before moving
                                                # Makes robot face target before approaching
                
                ('strategy_blend_duration', 0.15),  # Time to blend between movement strategies (seconds)
                                                  # Prevents jerky transitions between strategies
                
                ('coordinated_movement', True),  # Enable coordinated movement (combining rotation and translation)
                                               # Creates smoother, more efficient paths
                
                # ========== FILTERING AND PREDICTION ==========
                ('filter_buffer_size', 3),  # Target position filter buffer size
                                          # Larger values: smoother but more latency
                
                ('prediction_horizon', 0.04),  # Target prediction time horizon (seconds)
                                             # How far to predict target movement
                                             # Compensates for system processing delays
                
                # ========== APPROACH BEHAVIOR ==========
                ('approach_distance', 0.3),  # Distance to start slowing approach (meters)
                                           # Starts deceleration when this close to target
                
                ('min_approach_factor', 0.15),  # Minimum approach speed factor
                                              # Prevents robot from moving too slowly at final approach
                
                # ========== ADVANCED OPTIMIZATION ==========
                ('use_simplified_control_when_possible', True),  # Use simpler control algorithms when appropriate
                                                               # Reduces CPU load when full control isn't needed
                
                ('cpu_optimization_threshold', 70.0),  # CPU threshold for optimization (%)
                                                     # Above this, optimizations are activated
                
                ('use_fast_trigonometry', True),  # Use approximate trigonometric functions
                                                # Faster but slightly less accurate
                
                # ========== CONTROL RATE SETTINGS ==========
                ('min_control_rate', 2.4),  # Minimum control frequency (Hz)
                                          # Never goes below this rate, even under high load
                
                ('max_control_rate', 4.0),  # Maximum control frequency (Hz)
                                          # Never exceeds this rate, even under low load
                
                ('enable_fusion_rate_detection', True),  # Automatically detect sensor fusion rate
                                                       # Adapts control rate to match sensor data
                
                ('fresh_data_timeout', 0.7),  # Time until data considered non-fresh (seconds)
                                            # After this, control gets more conservative
                
                ('stale_data_timeout', 1.0),  # Time until data considered stale (seconds)
                                            # After this, robot may stop for safety
                
                # ========== CPU MANAGEMENT ==========
                ('cpu_throttle_interval', 0.5),  # Time between CPU checks (seconds)
                                               # Prevents too-frequent system queries
                
                ('enable_cycle_skipping', False),  # Skip control cycles when CPU is high
                                                 # More aggressive optimization for CPU constrained systems
                
                ('max_cpu_skip_threshold', 90.0),  # CPU threshold (%) to skip cycles
                                                 # Only active if cycle skipping enabled
                
                # ========== COORDINATED CONTROL PARAMETERS ==========
                # These parameters affect how lateral and angular movements are coordinated
                # Proper coordination creates more efficient, natural movement patterns
                ('coordinated_coupling_factor', 0.45),  # How strongly lateral/angular movements affect each other
                                                      # Higher values: more coordination but less independent control
                
                ('coordinated_smoothing_factor', 0.6),  # Smoothing factor for coordinated movements
                                                      # Higher values: smoother but less responsive
                
                ('coordinated_min_angle_for_reduction', 0.1),  # Angle threshold for coordination (radians)
                                                             # Below this angle, less coordination is applied
                
                ('coordinated_zero_angle_threshold', 0.015),  # Angle threshold to consider "zero" (radians)
                                                            # Provides a small deadband for stability
                
                ('coordinated_max_angle_factor', 0.2),  # Maximum effect angle has on lateral movement
                                                      # Prevents excessive lateral speed reduction
                
                ('coordinated_same_sign_scale', 0.8),  # Scaling when errors have same sign
                                                     # Same sign means errors "fight" each other
                
                ('coordinated_opposite_sign_scale', 0.9),  # Scaling when errors have opposite signs
                                                         # Opposite signs means errors "help" each other
            ]
        )

    def _get_parameters(self):
        # PID gains
        self.linear_x_kp = self.node.get_parameter('linear_x_kp').value
        self.linear_x_ki = self.node.get_parameter('linear_x_ki').value
        self.linear_x_kd = self.node.get_parameter('linear_x_kd').value
        self.linear_x_min = self.node.get_parameter('linear_x_min').value
        self.linear_x_max = self.node.get_parameter('linear_x_max').value
        self.linear_y_kp = self.node.get_parameter('linear_y_kp').value
        self.linear_y_ki = self.node.get_parameter('linear_y_ki').value
        self.linear_y_kd = self.node.get_parameter('linear_y_kd').value
        self.linear_y_min = self.node.get_parameter('linear_y_min').value
        self.linear_y_max = self.node.get_parameter('linear_y_max').value
        self.angular_kp = self.node.get_parameter('angular_kp').value
        self.angular_ki = self.node.get_parameter('angular_ki').value
        self.angular_kd = self.node.get_parameter('angular_kd').value
        self.angular_min = self.node.get_parameter('angular_min').value
        self.angular_max = self.node.get_parameter('angular_max').value
        # Thresholds
        self.distance_threshold = self.node.get_parameter('distance_threshold').value
        self.lateral_threshold = self.node.get_parameter('lateral_threshold').value
        self.angular_threshold = self.node.get_parameter('angular_threshold').value
        self.angular_at_target_factor = self.node.get_parameter('angular_at_target_factor').value
        # Approach
        self.approach_distance = self.node.get_parameter('approach_distance').value
        self.min_approach_factor = self.node.get_parameter('min_approach_factor').value
        # Resource monitoring
        self.adaptive_control_rate = self.node.get_parameter('adaptive_control_rate').value
        self.enable_resource_monitoring = self.node.get_parameter('enable_resource_monitoring').value
        self.cpu_high_threshold = self.node.get_parameter('cpu_high_threshold').value
        self.cpu_low_threshold = self.node.get_parameter('cpu_low_threshold').value
        # Transform
        self.enable_transform_caching = self.node.get_parameter('enable_transform_caching').value
        self.transform_cache_ttl = self.node.get_parameter('transform_cache_ttl').value
        # Strategy
        self.angular_first_control = self.node.get_parameter('angular_first_control').value
        self.strategy_blend_duration = self.node.get_parameter('strategy_blend_duration').value
        self.coordinated_movement = self.node.get_parameter('coordinated_movement').value
        # Target filter
        self.filter_buffer_size = self.node.get_parameter('filter_buffer_size').value
        self.prediction_horizon = self.node.get_parameter('prediction_horizon').value
        # Target update rate
        self.update_rate = self.node.get_parameter('target_update_rate').value
        self.diagnostics_rate = self.node.get_parameter('diagnostics_rate').value
        self.debug_level = self.node.get_parameter('debug_level').value
        # Optimization
        self.use_simplified_control_when_possible = self.node.get_parameter('use_simplified_control_when_possible').value
        self.cpu_optimization_threshold = self.node.get_parameter('cpu_optimization_threshold').value
        self.use_fast_trigonometry = self.node.get_parameter('use_fast_trigonometry').value
        # Rate control
        self.min_control_rate = self.node.get_parameter('min_control_rate').value
        self.max_control_rate = self.node.get_parameter('max_control_rate').value
        self.enable_fusion_rate_detection = self.node.get_parameter('enable_fusion_rate_detection').value
        self.fresh_data_timeout = self.node.get_parameter('fresh_data_timeout').value
        self.stale_data_timeout = self.node.get_parameter('stale_data_timeout').value
        # CPU throttling
        self.cpu_throttle_interval = self.node.get_parameter('cpu_throttle_interval').value
        self.enable_cycle_skipping = self.node.get_parameter('enable_cycle_skipping').value
        self.max_cpu_skip_threshold = self.node.get_parameter('max_cpu_skip_threshold').value
        # Misc
        self.desired_distance = 1.0  # Default value
        # Coordinated controller parameters
        self.coordinated_coupling_factor = self.node.get_parameter('coordinated_coupling_factor').value
        self.coordinated_smoothing_factor = self.node.get_parameter('coordinated_smoothing_factor').value
        self.coordinated_min_angle_for_reduction = self.node.get_parameter('coordinated_min_angle_for_reduction').value
        self.coordinated_zero_angle_threshold = self.node.get_parameter('coordinated_zero_angle_threshold').value
        self.coordinated_max_angle_factor = self.node.get_parameter('coordinated_max_angle_factor').value
        self.coordinated_same_sign_scale = self.node.get_parameter('coordinated_same_sign_scale').value
        self.coordinated_opposite_sign_scale = self.node.get_parameter('coordinated_opposite_sign_scale').value


# StateObserver interface for classes that want to observe state changes
class StateObserver(ABC):
    """
    PURPOSE: Interface for classes that need to observe robot state changes.
    
    This abstract base class defines the interface that observer classes must implement
    to receive notifications when the robot's state changes or when data freshness
    levels change. It enables a publish-subscribe pattern for state events.
    """
    """Interface for classes that observe state changes."""
    @abstractmethod
    def on_state_change(self, old_state, new_state, reason=""):
        """Called when robot state changes."""
        pass
    
    @abstractmethod
    def on_freshness_change(self, freshness_level, data_age):
        """Called when data freshness level changes."""
        pass


# Data classes for holding robot, movement, performance, freshness, and recovery state
class RobotStateData:
    """
    PURPOSE: Holds core robot state information.
    
    This data class encapsulates the robot's current operational state
    (initializing, searching, tracking, etc.), previous state, and timing
    information. It provides a central place to track the robot's high-level status.
    """
    """Core robot state data"""
    def __init__(self):
        self.state = "initializing"
        self.previous_state = None
        self.last_state_change_time = time.time()
        self.robot_orientation = 0.0
        self.last_orientation_time = None
        self.cycle_count = 0
        self.force_target_reacquisition = False

class MovementStateData:
    """
    PURPOSE: Tracks movement-related state information.
    
    This data class maintains information about the robot's current movement,
    including whether it's stopped, last velocity commands, and pre-allocated
    vectors for efficient movement calculations. It helps manage the transition
    between moving and stopped states.
    """
    """Movement-related state data"""
    def __init__(self):
        self.robot_stopped = True
        self.stop_time = time.time()
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        self.last_logged_cmd = (0.0, 0.0, 0.0)
        self.movement_hysteresis = 0.0
        self.last_stop_position = (0.0, 0.0, 0.0)
        self.using_simplified_control = False
        self.computation_level = 3
        self.last_full_computation_time = time.time()
        self.initial_movement_boost = True
        
        # Pre-allocated vectors for movement calculations
        self.prev_velocities = np.zeros(3, dtype=np.float32)
        self.target_velocities = np.zeros(3, dtype=np.float32)
        self.vel_diffs = np.zeros(3, dtype=np.float32)
        self.velocity_tuple = [0.0, 0.0, 0.0]
        self.velocity_change_check = [False, False, False]

class PerformanceData:
    """
    PURPOSE: Tracks performance metrics and runtime statistics.
    
    This data class maintains information about system performance metrics,
    including CPU usage patterns, control cycle counts, and execution timing.
    It helps monitor and optimize the controller's resource usage.
    """
    """Performance monitoring state"""
    def __init__(self):
        self.simplified_control_count = 0
        self.last_cpu_check_time = 0.0
        self.last_cpu_warning_time = 0.0
        self.skipped_cycle_count = 0
        self.last_pool_log_time = 0.0
        self.current_rate = 0.0
        self.last_logged_rate = 0.0
        self.adaptive_rate_history = deque(maxlen=10)
        self.timer_control_count = 0
        self.event_control_count = 0
        self.last_timer_execution = 0.0
        self.last_event_execution = 0.0
        self.last_rate_adjustment_time = 0.0
        self.detected_fusion_rate = 1.0
        self.fusion_rate_updated = False
        self.adaptive_rate = 0.0

class DataFreshnessData:
    """
    PURPOSE: Tracks sensor data freshness information.
    
    This data class monitors how recent and reliable the sensor data is,
    tracking the freshness level (fresh, stale, critical) and timestamps
    to ensure the robot operates safely with up-to-date information.
    """
    """Data freshness tracking state"""
    def __init__(self):
        self.level = "unknown"
        self.state_change_time = time.time()
        self.last_fusion_check_time = time.time()

class RecoveryStateData:
    """
    PURPOSE: Manages error recovery state information.
    
    This data class tracks the robot's recovery status when errors or
    exceptional conditions occur, including recovery phases and timing
    information for handling error conditions safely.
    """
    """Recovery-related state data"""
    def __init__(self):
        self.in_recovery = False
        self.recovery_start_time = 0.0
        self.recovery_phase = "none"


# EnhancedStateManager manages all state and notifies observers
class EnhancedStateManager:
    """
    PURPOSE: Centrally manages all robot state information.
    
    This class serves as the central coordinator for all state-related
    information, including robot state, movement state, performance metrics,
    data freshness, and recovery state. It implements the observer pattern
    to notify components when state changes occur.
    """
    """Enhanced state manager that provides observation capabilities"""
    def __init__(self):
        self.robot = RobotStateData()
        self.movement = MovementStateData()
        self.perf = PerformanceData()
        self.freshness = DataFreshnessData()
        self.recovery = RecoveryStateData()
        self.shutting_down = False
        self.last_control_time = time.time()
        self.skip_next_cycle = False
        self.observers = []
        
        # Pre-allocate numpy arrays for movement calculations
        self._limited_velocities = np.zeros(3, dtype=np.float32)

    def register_observer(self, observer):
        """Register a state observer."""
        if isinstance(observer, StateObserver):
            self.observers.append(observer)
        
    def transition_state(self, new_state, reason=""):
        """Handle state transition and return True if state changed"""
        if new_state == self.robot.state:
            return False
            
        # Store previous state and update current state
        old_state = self.robot.state
        self.robot.previous_state = old_state
        self.robot.state = new_state
        self.robot.last_state_change_time = time.time()
        
        # Notify observers
        for observer in self.observers:
            observer.on_state_change(old_state, new_state, reason)
            
        return True
        
    def update_freshness(self, freshness_level, data_age):
        """Update data freshness state."""
        if freshness_level != self.freshness.level:
            old_level = self.freshness.level
            self.freshness.level = freshness_level
            self.freshness.state_change_time = time.time()
            
            # Notify observers
            for observer in self.observers:
                observer.on_freshness_change(freshness_level, data_age)
        
    def is_state(self, state_name):
        """Efficient state check"""
        return self.robot.state == state_name
        
    def was_state(self, state_name):
        """Check previous state"""
        return self.robot.previous_state == state_name


# DiagnosticsPublisher publishes diagnostic and performance data for monitoring
class DiagnosticsPublisher:
    """
    PURPOSE: Collects and publishes diagnostic information about the controller.
    
    This class gathers performance metrics, PID status information, and other
    diagnostic data from various components of the control system and publishes
    them to ROS topics for monitoring and debugging purposes.
    """
    """Handles collection and publishing of diagnostic data."""
    def __init__(self, node, pid_controllers, target_tracker, strategy_module, velocity_control, resource_monitor, parameter_manager, logger, throttled_logger):
        self.node = node
        self.pid_controllers = pid_controllers
        self.target_tracker = target_tracker
        self.strategy_module = strategy_module
        self.velocity_control = velocity_control
        self.resource_monitor = resource_monitor
        self.parameter_manager = parameter_manager
        self.logger = logger
        self.throttled_logger = ThrottledLogger(logger)
        
        # Setup publishers
        self.pid_diag_pub = node.create_publisher(
            Float32MultiArray,
            TOPICS["output"]["diagnostics"],
            10
        )
        
        self.performance_pub = node.create_publisher(
            String,
            TOPICS["output"]["performance"],
            10
        )
        
        # Setup data containers
        self._diag_msg = Float32MultiArray()
        self._diag_data = np.zeros(14, dtype=np.float32)
        
        # Logging helper
        self.throttled_logger = throttled_logger  # Use the global throttled logger
    
    def publish_diagnostics(self, state_manager):
        """Publish detailed diagnostic information at a slower rate."""
        try:
            if state_manager.shutting_down:
                return
                
            # Calculate velocity statistics
            vel_data = self.velocity_control.get_velocity_history()
            if not vel_data:
                return
                
            # Use NumPy for vectorized calculations when possible
            if len(vel_data) > 0:
                # Convert to NumPy array for efficient calculation
                vel_array = np.array(vel_data)
                
                # Extract velocities by column (more efficient than list comprehensions)
                if vel_array.size > 0:  # Check that array has elements
                    lin_x_velocities = vel_array[:, 0]
                    lin_y_velocities = vel_array[:, 1]
                    ang_velocities = vel_array[:, 2]
                    
                    # Calculate statistics using NumPy
                    avg_lin_x_vel = np.mean(lin_x_velocities) if lin_x_velocities.size > 0 else 0.0
                    avg_lin_y_vel = np.mean(lin_y_velocities) if lin_y_velocities.size > 0 else 0.0
                    avg_ang_vel = np.mean(ang_velocities) if ang_velocities.size > 0 else 0.0
                else:
                    avg_lin_x_vel = avg_lin_y_vel = avg_ang_vel = 0.0
            else:
                avg_lin_x_vel = avg_lin_y_vel = avg_ang_vel = 0.0
            
            # Log detailed information with built-in throttling
            if hasattr(self.parameter_manager, 'debug_level') and self.parameter_manager.debug_level >= 1:
                # Get PID components
                p_x, i_x, d_x = self.pid_controllers['linear_x'].get_components()
                p_y, i_y, d_y = self.pid_controllers['linear_y'].get_components()
                p_a, i_a, d_a = self.pid_controllers['angular'].get_components()
                
                # Get stats on simplified control usage
                simplified_pct = state_manager.perf.simplified_control_count / max(1, state_manager.robot.cycle_count) * 100.0
                
                diag_msg = (
                    f"DIAGNOSTICS: "
                    f"Avg Vel=[{avg_lin_x_vel:.2f}, {avg_lin_y_vel:.2f}, {avg_ang_vel:.2f}], "
                    f"PID X=[{p_x:.2f}, {i_x:.2f}, {d_x:.2f}], "
                    f"PID Y=[{p_y:.2f}, {i_y:.2f}, {d_y:.2f}], "
                    f"PID A=[{p_a:.2f}, {i_a:.2f}, {d_a:.2f}], "
                    f"Strategy={self.strategy_module.current_strategy}, "
                    f"Simp={simplified_pct:.1f}%, "
                    f"Freshness={state_manager.freshness.level}"
                )
                if self.parameter_manager.debug_level >= 1:
                    self.throttled_logger.info(diag_msg, throttle_duration_sec=LOG_THROTTLE_DIAG, log_id='diagnostics')
            
            # Publish PID diagnostic data
            self._publish_pid_diagnostics()
            
            # Publish performance metrics
            self._publish_performance_metrics(state_manager)
        except Exception as e:
            if self.parameter_manager.debug_level >= 1:
                self.logger.error(f"Error in publish_diagnostics: {str(e)}")
    
    def _publish_pid_diagnostics(self):
        """Publish detailed PID diagnostics for analysis."""
        try:
            # Get PID components
            p_x, i_x, d_x = self.pid_controllers['linear_x'].get_components()
            p_y, i_y, d_y = self.pid_controllers['linear_y'].get_components()
            p_a, i_a, d_a = self.pid_controllers['angular'].get_components()
            
            # Get current gains
            kp_x, ki_x, kd_x = self.pid_controllers['linear_x'].get_current_gains()
            kp_y, ki_y, kd_y = self.pid_controllers['linear_y'].get_current_gains()
            kp_a, ki_a, kd_a = self.pid_controllers['angular'].get_current_gains()
            
            # Get current errors
            e_x = self.pid_controllers['linear_x'].error_tracker.current_error
            e_y = self.pid_controllers['linear_y'].error_tracker.current_error
            e_a = self.pid_controllers['angular'].error_tracker.current_error
            
            # Pack all data into the array - no unnecessary float() conversions
            # Direct array access is more efficient
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
        except Exception as e:
            if self.parameter_manager.debug_level >= 1:
                self.logger.error(f"Error in _publish_pid_diagnostics: {str(e)}")
    
    def _publish_performance_metrics(self, state_manager):
        """Publish performance metrics for monitoring."""
        try:
            # Get performance stats
            perf_stats = self.resource_monitor.get_performance_stats()
            
            # Check if we have valid stats
            if not perf_stats or not all(k in perf_stats for k in ['cpu_avg', 'cycle_time_ms', 'update_rate']):
                # Create default stats if missing
                perf_stats = {
                    'cpu_avg': 0.0,
                    'cycle_time_ms': 0.0,
                    'skips': 0,
                    'update_rate': getattr(self.parameter_manager, 'update_rate', 3.0)
                }
            
            # Get current strategy
            strategy_name = "unknown"
            if hasattr(self.strategy_module, 'current_strategy'):
                strategy_name = self.strategy_module.current_strategy
            
            # Add optimization stats
            adaptive_rate = getattr(state_manager.perf, 'adaptive_rate', perf_stats['update_rate'])
            using_simplified = 1 if state_manager.movement.using_simplified_control else 0
            
            # Add freshness and event stats
            freshness_level = state_manager.freshness.level
            event_ratio = 0.0
            total_cycles = max(1, state_manager.perf.event_control_count + state_manager.perf.timer_control_count)
            event_ratio = state_manager.perf.event_control_count / total_cycles * 100.0
            
            # Create performance message - use string formatting for better performance
            # in tight loops
            performance_msg = String()
            performance_msg.data = (
                '{{"cpu": {0:.1f}, '
                '"cycle_time_ms": {1:.2f}, '
                '"strategy": "{2}", '
                '"skips": {3}, '
                '"update_rate": {4:.1f}, '
                '"adaptive_rate": {5:.1f}, '
                '"simplified": {6}, '
                '"freshness": "{7}", '
                '"event_ratio": {8:.1f}}}'
            ).format(
                perf_stats["cpu_avg"],
                perf_stats["cycle_time_ms"],
                strategy_name,
                perf_stats.get("skips", state_manager.perf.skipped_cycle_count),
                perf_stats["update_rate"],
                adaptive_rate,
                using_simplified,
                state_manager.freshness.level,
                event_ratio
            )
            
            # Publish
            self.performance_pub.publish(performance_msg)
        except Exception as e:
            if self.parameter_manager.debug_level >= 1:
                self.logger.error(f"Error publishing performance metrics: {str(e)}")


# EnhancedPerformanceMonitor adapts control rate based on CPU and fusion rate
class EnhancedPerformanceMonitor:
    """
    PURPOSE: Monitors and optimizes system performance.
    
    This class tracks CPU usage, memory usage, and execution timing to
    dynamically adjust the control rate and processing complexity based on
    system load. It helps ensure reliable operation even under resource
    constraints on the Raspberry Pi 5.
    """
    """Enhanced performance monitoring with adaptive rate control."""
    def __init__(self, resource_monitor, parameter_manager, logger, throttled_logger):
        self.resource_monitor = resource_monitor
        self.parameter_manager = parameter_manager
        self.logger = logger
        self.throttled_logger = ThrottledLogger(logger)
        self.detected_fusion_rate = 1.0
        self.fusion_rate_updated = False
        self.last_rate_adjustment_time = 0.0
        self.current_rate = parameter_manager.update_rate
        self.last_logged_rate = 0.0
        self.adaptive_rate_history = deque(maxlen=10)
        
    def calculate_adaptive_rate(self, state_manager, base_rate, cpu_usage):
        """Calculate adaptive control rate based on CPU usage and fusion rate."""
        try:
            current_time = time.time()
            if not isinstance(base_rate, (int, float)) or base_rate <= 0:
                if self.parameter_manager.debug_level >= 1:
                    self.logger.warning(f"Invalid base_rate: {base_rate}, using default 3.0Hz")
                base_rate = 3.0
            if not isinstance(cpu_usage, (int, float)) or cpu_usage < 0:
                if self.parameter_manager.debug_level >= 1:
                    self.logger.warning(f"Invalid cpu_usage: {cpu_usage}, using 50%")
                cpu_usage = 50.0
                
            # Only adjust rate periodically to avoid oscillation
            if current_time - self.last_rate_adjustment_time < 1.0:
                return self.current_rate or base_rate
                
            self.last_rate_adjustment_time = current_time
            
            # Apply fusion rate consideration if detected
            if self.fusion_rate_updated and self.parameter_manager.enable_fusion_rate_detection:
                fusion_rate = self.detected_fusion_rate
                if fusion_rate > 0 and fusion_rate < 100:
                    adjusted_base_rate = min(max(fusion_rate * 1.2, self.parameter_manager.min_control_rate), self.parameter_manager.max_control_rate)
                    if abs(adjusted_base_rate - base_rate) > 0.3:
                        if self.parameter_manager.debug_level >= 1:
                            self.logger.info(
                                f"Adjusted base rate using fusion detection: {base_rate:.1f}Hz -> {adjusted_base_rate:.1f}Hz "
                                f"(fusion_rate: {fusion_rate:.1f}Hz)"
                            )
                    base_rate = adjusted_base_rate
                    
            # Adjusted thresholds for high baseline CPU
            if cpu_usage > 95.0:
                new_rate = self.parameter_manager.min_control_rate
            elif cpu_usage > 90.0:
                new_rate = base_rate * 0.5
            elif cpu_usage > 85.0:
                new_rate = base_rate * 0.7
            elif cpu_usage > 80.0:
                new_rate = base_rate * 0.85
            elif cpu_usage < 40.0:
                new_rate = base_rate * 1.1
            else:
                new_rate = base_rate
                
            new_rate = max(self.parameter_manager.min_control_rate, min(self.parameter_manager.max_control_rate, new_rate))
            self.current_rate = new_rate
            self.adaptive_rate_history.append((current_time, new_rate))
            
            if abs(new_rate - self.last_logged_rate) > 0.5:
                if self.parameter_manager.debug_level >= 1:
                    self.throttled_logger.info(
                        f"Adaptive rate adjusted: {self.last_logged_rate:.1f}Hz -> {new_rate:.1f}Hz "
                        f"(CPU: {cpu_usage:.1f}%)",
                        throttle_duration_sec=2.0,
                        log_id='adaptive_rate'
                    )
                self.last_logged_rate = new_rate
                
            return new_rate
        except Exception as e:
            if self.parameter_manager.debug_level >= 1:
                self.logger.error(f"Error in calculate_adaptive_rate: {str(e)}")
            return max(base_rate, self.parameter_manager.min_control_rate)
    
    def update_fusion_rate(self, fusion_rate):
        """Update the detected fusion rate."""
        if fusion_rate > 0:
            self.detected_fusion_rate = fusion_rate
            self.fusion_rate_updated = True
            
            # Update the resource monitor with the new fusion rate
            if hasattr(self.resource_monitor, 'set_fusion_rate'):
                self.resource_monitor.set_fusion_rate(fusion_rate)
    
    def should_skip_cycle(self, cpu_usage, last_control_time, adaptive_rate):
        """Determine if the current cycle should be skipped based on CPU usage and timing."""
        # Skip based on CPU threshold
        if self.parameter_manager.enable_cycle_skipping and cpu_usage > self.parameter_manager.max_cpu_skip_threshold:
            return True
            
        # Skip based on timing
        if adaptive_rate > 0:
            current_time = time.time()
            time_since_last = current_time - last_control_time
            if time_since_last < (1.0 / adaptive_rate):
                return True
                
        return False
    
    def update_performance_stats(self, cycle_duration):
        """Update performance statistics in the resource monitor."""
        try:
            if hasattr(self.resource_monitor, '_update_cycle_stats'):
                self.resource_monitor._update_cycle_stats(cycle_duration)
        except Exception as e:
            if self.parameter_manager.debug_level >= 1:
                self.logger.warning(f"Error updating performance stats: {str(e)}")


# Abstract base class for all control strategies
class ControlStrategy(ABC):
    """
    PURPOSE: Abstract base class for different control strategies.
    
    This class defines the interface that all control strategies must implement.
    Control strategies convert error values into velocity commands using different
    approaches optimized for different situations, such as full PID control,
    simplified control, or recovery behavior.
    
    CONTROL STRATEGY ARCHITECTURAL PATTERN
    -------------------------------------
    
    The control strategy pattern allows the robot to adapt its movement calculation
    approach based on different situations without changing the overall control flow.
    This is a form of the Strategy design pattern from software engineering.
    
    Benefits of this pattern:
    1. Separation of concerns - Each strategy focuses on one approach to control
    2. Runtime adaptation - System can switch strategies without stopping
    3. Extensibility - New strategies can be added without changing existing code
    4. Resource optimization - Can select strategies based on CPU availability
    
    The three main strategies implemented are:
    - StandardControlStrategy: Full PID with coordination (best accuracy, higher CPU)
    - SimplifiedControlStrategy: Basic control (lower accuracy, saves CPU)
    - RecoveryControlStrategy: Safety-focused movements for error recovery
    
    The ControlStrategyFactory selects which strategy to use based on:
    - Current robot state (tracking, recovery, etc.)
    - CPU load and resource availability
    - Data freshness and sensor quality
    """
    """Base class for control strategies."""
    @abstractmethod
    def compute_velocity_command(self, errors, position_data, current_time, freshness_level="fresh"):
        """
        Compute velocity commands based on errors and other data.
        
        Args:
            errors: Tuple of (distance_error, lateral_error, angular_error)
            position_data: Dictionary with target position information
            current_time: Current timestamp for time-based calculations
            freshness_level: How recent the sensor data is ('fresh', 'stale', 'critical')
            
        Returns:
            Tuple of (linear_x, linear_y, angular_z) velocity commands
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self):
        """
        Return a unique name for this strategy.
        
        Used for logging, diagnostics, and strategy transition tracking.
        """
        pass


# StandardControlStrategy: full PID and coordinated control
class StandardControlStrategy(ControlStrategy):
    """
    PURPOSE: Implements full PID control with movement coordination.
    
    This class provides the primary control strategy used during normal operation,
    applying full PID control to each movement dimension with coordination between
    lateral and angular movements for smooth, natural motion patterns.
    
    STANDARD CONTROL STRATEGY OPERATION
    ----------------------------------
    
    This is the primary, high-precision control strategy that provides the best
    tracking quality and movement smoothness. It is used whenever system resources
    allow, and is responsible for the robot's primary ball-tracking behavior.
    
    Key Features:
    
    1. FULL PID CONTROL
       - Applies complete PID control to all three movement dimensions
       - Considers error history, trends, and rates of change
       - Adapts PID gains based on error patterns
       - Handles integral windup and zero-crossing conditions
    
    2. COORDINATED MOVEMENT
       - Coordinates lateral (side-to-side) and angular (rotational) movements
       - Prevents fighting between movement dimensions
       - Scales movements based on error patterns
       - Creates more natural, human-like motion
    
    3. ANGULAR-FIRST BEHAVIOR
       - Prioritizes facing the target when angular error is large
       - Reduces forward/lateral speeds during significant rotation
       - Makes movement paths more efficient and natural
    
    4. DATA FRESHNESS SAFETY
       - Reduces speed when sensor data is stale
       - Stops completely when data is critically old
       
    This strategy has the highest computational cost but provides the best
    tracking performance and is used whenever CPU resources permit.
    """
    """Full control strategy with PID and coordination."""
    def __init__(self, pid_controllers, coordinated_controller, parameter_manager, distance_error_tracker,
                 lateral_error_tracker, angular_error_tracker, velocity_control):
        self.pid_controllers = pid_controllers
        self.coordinated_controller = coordinated_controller
        self.parameter_manager = parameter_manager
        self.distance_error_tracker = distance_error_tracker
        self.lateral_error_tracker = lateral_error_tracker
        self.angular_error_tracker = angular_error_tracker
        self.velocity_control = velocity_control
    
    def compute_velocity_command(self, errors, position_data, current_time, freshness_level="fresh"):
        """
        Compute velocity commands using full PID control with coordination.
        
        EDUCATIONAL WALKTHROUGH: FROM ERRORS TO ROBOT MOVEMENT
        ----------------------------------------------------
        
        This method is where the core decision-making happens. It takes the errors
        (how far we are from the ball) and computes velocity commands that will
        move the robot smoothly and precisely toward the target.
        
        STEP BY STEP CONTROL FLOW:
        
        1. UNDERSTAND THE INPUTS:
           - errors: Three values telling us how far we are from perfect position
           - position_data: Details about the ball's position and movement
           - current_time: Current timestamp for calculating time differences
           - freshness_level: How recent our sensor data is
        
        2. DECIDE ON BASIC MOVEMENT CAPABILITIES:
           We decide which dimensions to use:
           - use_forward: Can we move forward/backward? (Almost always True)
           - use_lateral: Can we move side-to-side? (True with mecanum wheels)
           - use_angular: Can we rotate? (Almost always True)
        
        3. CHECK FOR SPECIAL CASES:
           If the ball is at a significant angle (>11°), we prioritize turning
           to face it before moving forward too quickly (angular-first strategy).
           This creates more natural movement patterns - just like a person would
           turn to face an object before moving toward it.
           
        4. COMPUTE VELOCITY COMMANDS:
           Here we have two approaches:
           
           A. COORDINATED CONTROL (preferred):
              - Calculate forward velocity independently
              - Calculate lateral and angular velocities in coordination
              - This creates smooth curves and natural movements
              - Coordination prevents jerky movements when both turning and 
                moving sideways at the same time
                
           B. TRADITIONAL SEPARATE CONTROL:
              - Calculate each dimension independently
              - Used as fallback when coordinated control is disabled
              - Simpler but can create less natural movements
        
        5. APPLY ANGULAR-FIRST STRATEGY:
           If the angle to the ball is large (>11°), reduce forward and lateral
           speeds until we're better aligned. This is like slowing down to take
           a turn more smoothly.
        
        6. APPLY DATA FRESHNESS SAFETY:
           - Fresh data: Normal operation
           - Stale data: Move at 50% speed for safety
           - Critical/invalid data: STOP completely
           
           This is a critical safety feature - old sensor data means we're
           not sure where the ball really is, so we reduce speed or stop.
        
        7. APPLY VELOCITY CONTROL LIMITS:
           - Ensure speeds don't exceed safe maximums
           - Scale speeds based on distance to target
           - Apply smoother acceleration/deceleration
           - Apply additional safety checks
        
        The final output is a tuple of 3 velocity commands:
        [forward velocity, lateral velocity, angular velocity]
        These values directly control the robot's wheels to create movement.
        """
        # Unpack errors - these tell us how far we are from the desired position
        distance_error, lateral_error, angular_error = errors
        
        # Determine which movement dimensions we should use
        # For a mecanum-wheeled robot, all three are typically enabled
        use_forward = True    # Forward/backward movement (X-axis)
        use_lateral = True    # Side-to-side movement (Y-axis)
        use_angular = True    # Rotational movement (angular Z-axis)
        
        # Check if we need to prioritize turning to face the ball
        # This creates more natural movement, like a person turning to face an object
        significant_angular_error = False
        if self.parameter_manager.angular_first_control:
            # Convert radians to degrees for easier threshold comparison
            # 57.29578 = 180/π (conversion factor from radians to degrees)
            angular_degrees = angular_error * 57.29578
            if abs(angular_degrees) > 11.0:  # If ball is more than 11° off-center
                significant_angular_error = True
                # We'll use this flag later to reduce forward/lateral speeds
        
        # VELOCITY COMPUTATION: We have two methods to calculate velocities
        if self.parameter_manager.coordinated_movement and use_lateral and use_angular:
            # METHOD 1: COORDINATED CONTROL (preferred)
            # ----------------------------------------
            # This method coordinates lateral and angular movements for smoother paths
            
            # 1.A: Compute forward velocity independently using PID
            # This controls how fast we approach or back away from the ball
            linear_x_velocity = self.pid_controllers['linear_x'].compute(
                distance_error,                            # Error input (how far from desired distance)
                current_time,                             # Current time for dt calculation
                not use_forward,                          # Whether to force zero output
                self.distance_error_tracker.get_trend()   # Error trend (is error increasing or decreasing?)
            )
            
            # 1.B: Use special coordinated controller for lateral and angular movements
            # This ensures smooth coordination between side motion and rotation
            lateral_velocity, angular_velocity = self.coordinated_controller.compute(
                lateral_error,   # How far left/right of target (meters)
                angular_error,   # How far rotated from facing target (radians)
                current_time,    # Current time for calculating time differences
                0.0              # Current orientation (handled internally by controller)
            )
            
            # If we need to turn significantly to face the ball, reduce forward and lateral speeds
            # This creates more natural movement, like how you'd slow down to take a sharp turn
            if significant_angular_error:
                linear_x_velocity *= 0.7    # Reduce forward speed to 70%
                lateral_velocity *= 0.8     # Reduce lateral speed to 80%
            
            # If specific movements are disabled, zero out those velocities
            if not use_lateral:
                lateral_velocity = 0.0
            if not use_angular:
                angular_velocity = 0.0
        else:
            # METHOD 2: TRADITIONAL SEPARATE PID CONTROLLERS
            # ---------------------------------------------
            # This method computes each dimension independently
            
            # 2.A: Compute forward velocity
            linear_x_velocity = self.pid_controllers['linear_x'].compute(
                distance_error, 
                current_time, 
                not use_forward,
                self.distance_error_tracker.get_trend()
            )
            
            # 2.B: Compute lateral (side-to-side) velocity
            lateral_velocity = self.pid_controllers['linear_y'].compute(
                lateral_error, 
                current_time, 
                not use_lateral,
                self.lateral_error_tracker.get_trend()
            )
            
            # 2.C: Compute angular (rotational) velocity
            angular_velocity = self.pid_controllers['angular'].compute(
                angular_error, 
                current_time, 
                not use_angular,
                self.angular_error_tracker.get_trend()
            )
            
            # Apply angular-first strategy as before
            if significant_angular_error:
                linear_x_velocity *= 0.7
                lateral_velocity *= 0.8
        
        # SAFETY: Adjust speeds based on how fresh (recent) our sensor data is
        # This is critical for safety - old data means we're not sure where the ball really is
        if freshness_level == "stale":
            # Data is old but still usable - reduce speed to 50% for safety
            stale_scale = 0.5
            linear_x_velocity *= stale_scale
            lateral_velocity *= stale_scale
            angular_velocity *= stale_scale
        elif freshness_level == "critical" or freshness_level == "invalid":
            # Data is too old or invalid - STOP the robot completely for safety
            linear_x_velocity = 0.0
            lateral_velocity = 0.0
            angular_velocity = 0.0
        
        # FINAL PROCESSING: Apply velocity limits and safety constraints
        # This ensures our commands don't exceed safe speeds and accelerations
        target_distance = position_data['distance'] if position_data and 'distance' in position_data else 0.0
        limited_velocities = self.velocity_control.process_velocities(
            linear_x_velocity,                           # Forward velocity (m/s)
            lateral_velocity,                            # Lateral velocity (m/s)
            angular_velocity,                            # Angular velocity (rad/s)
            target_distance,                             # Current distance to target
            self.parameter_manager.desired_distance,     # Desired distance to maintain
            freshness_level=freshness_level              # Data freshness for safety
        )
        
        # Return the final velocity commands that will control the robot's motion
        return limited_velocities
    
    def get_strategy_name(self):
        return "standard"


# SimplifiedControlStrategy: reduced computation for CPU savings
class SimplifiedControlStrategy(ControlStrategy):
    """
    PURPOSE: Provides computationally efficient control for high CPU loads.
    
    This class implements a simplified control strategy that uses less CPU
    resources by applying basic velocity commands without full PID computation.
    It's automatically activated when system load is high or when precise
    control is less critical.
    
    SIMPLIFIED CONTROL STRATEGY OPERATION
    -----------------------------------
    
    This strategy is designed to reduce computational load while still maintaining
    basic tracking functionality. It's automatically activated when:
    - CPU usage exceeds configured thresholds
    - Errors are small and precise control is less critical
    - During transitions between other system states
    
    There are multiple levels of simplification (set by the 'level' parameter):
    
    LEVEL 0: MINIMAL COMPUTATION
    - Simply dampens previous velocities
    - Essentially coasts with gradually decreasing speed
    - Used in extremely high CPU load situations
    - Minimal overhead, saves maximum CPU
    
    LEVEL 1: BASIC PROPORTIONAL CONTROL
    - Uses simple proportional control (no I or D terms)
    - Basic movement toward target without advanced features
    - No coordination between dimensions
    - Approximately 60% reduction in computation vs. standard
    
    LEVEL 2: ENHANCED PROPORTIONAL CONTROL
    - Adds basic lateral/angular coordination
    - Simple error-based movement scaling
    - Minor prediction capability
    - Approximately 40% reduction in computation vs. standard
    
    The EnhancedPerformanceMonitor decides when to switch to this strategy
    and which level to use based on current CPU load and tracking requirements.
    """
    """Simplified control strategy for reduced CPU usage."""
    def __init__(self, pid_controllers, parameter_manager, velocity_control, last_cmd_vel, level=1):
        self.pid_controllers = pid_controllers
        self.parameter_manager = parameter_manager
        self.velocity_control = velocity_control
        self.last_cmd_vel = last_cmd_vel
        self.level = level  # 0: minimal, 1: basic, 2: medium
        
    def compute_velocity_command(self, errors, position_data, current_time, freshness_level="fresh"):
        """Compute velocity commands using simplified control."""
        if self.level == 0:
            # Minimal computation - just dampen previous velocities
            damping = 0.85
            linear_x = self.last_cmd_vel[0] * damping
            lateral_y = self.last_cmd_vel[1] * damping
            angular_z = self.last_cmd_vel[2] * damping
            return [linear_x, lateral_y, angular_z]
            
        elif self.level == 1:
            # Basic computation - simple proportional control
            if position_data and all(k in position_data for k in ['distance', 'lateral', 'bearing']):
                # Calculate basic errors
                distance_error = position_data['distance'] - self.parameter_manager.desired_distance
                lateral_error = position_data['lateral']
                angular_error = position_data['bearing']
                
                # Simple proportional control with reduced gains
                kp_factor = 0.7  # Reduce gains for smoother control
                linear_x = max(-0.1, min(0.1, distance_error * self.parameter_manager.linear_x_kp * kp_factor))
                lateral_y = max(-0.1, min(0.1, lateral_error * self.parameter_manager.linear_y_kp * kp_factor))
                angular_z = max(-0.3, min(0.3, angular_error * self.parameter_manager.angular_kp * kp_factor))
                
                # Apply damping from previous velocities for smoothness
                damping = 0.3  # 30% of previous velocity
                linear_x = linear_x * (1.0 - damping) + self.last_cmd_vel[0] * damping
                lateral_y = lateral_y * (1.0 - damping) + self.last_cmd_vel[1] * damping
                angular_z = angular_z * (1.0 - damping) + self.last_cmd_vel[2] * damping
                
                return [linear_x, lateral_y, angular_z]
            else:
                # No valid position data, apply strong damping
                damping = 0.7
                linear_x = self.last_cmd_vel[0] * damping
                lateral_y = self.last_cmd_vel[1] * damping
                angular_z = self.last_cmd_vel[2] * damping
                return [linear_x, lateral_y, angular_z]
                
        elif self.level == 2:
            # Medium computation - use PID but skip coordinated control
            distance_error, lateral_error, angular_error = errors
            
            # Use separate PID controllers for efficiency - no coordination
            linear_x_velocity = self.pid_controllers['linear_x'].compute(
                distance_error, 
                current_time, 
                False,
                self.pid_controllers['linear_x'].error_tracker.get_trend()
            )
            
            lateral_velocity = self.pid_controllers['linear_y'].compute(
                lateral_error, 
                current_time, 
                False,
                self.pid_controllers['linear_y'].error_tracker.get_trend()
            )
            
            angular_velocity = self.pid_controllers['angular'].compute(
                angular_error, 
                current_time, 
                False,
                self.pid_controllers['angular'].error_tracker.get_trend()
            )
            
            # Apply velocity limits directly (simplified)
            linear_x_velocity = max(self.parameter_manager.linear_x_min, min(self.parameter_manager.linear_x_max, linear_x_velocity))
            lateral_velocity = max(self.parameter_manager.linear_y_min, min(self.parameter_manager.linear_y_max, lateral_velocity))
            angular_velocity = max(self.parameter_manager.angular_min, min(self.parameter_manager.angular_max, angular_velocity))
            
            return [linear_x_velocity, lateral_velocity, angular_velocity]
            
        # Default fallback
        return list(self.last_cmd_vel)
    
    def get_strategy_name(self):
        return f"simplified_level_{self.level}"


# RecoveryControlStrategy: handles recovery behavior
class RecoveryControlStrategy(ControlStrategy):
    """
    PURPOSE: Handles control during error recovery and exceptional conditions.
    
    This strategy takes over when the system encounters errors or exceptional
    conditions, implementing safety behaviors like stopping, backing up, or
    performing specific recovery movements to restore normal operation.
    
    RECOVERY CONTROL STRATEGY OPERATION
    ---------------------------------
    
    This strategy is a safety mechanism that takes control during exceptional
    conditions when normal tracking isn't possible. It's activated when:
    
    - Sensor data is missing or critically stale
    - System errors or exceptions occur
    - Target is lost for extended periods
    - Other error conditions are detected
    
    The recovery strategy operates through several phases:
    
    1. INITIAL PHASE: SAFE STOP
       - Immediately stops all movement
       - Ensures robot is in a safe, stable state
       - Prevents further problems in case of sensor malfunction
    
    2. DIAGNOSIS PHASE
       - Assesses what caused the recovery condition
       - Determines if recovery is possible
       - Selects appropriate recovery behavior
    
    3. RECOVERY PHASE
       - Implements specific recovery behaviors, which might include:
         * Rotating in place to search for the ball
         * Backing up to get a better view
         * Moving to a known good position
         * Waiting for sensor data to return
    
    4. VALIDATION PHASE
       - Checks if recovery was successful
       - Returns to normal operation if recovered
       - Attempts alternative recovery if not
    
    This strategy prioritizes safety over performance and ensures the robot
    can gracefully handle exceptional conditions without requiring human
    intervention.
    """
    """Strategy for recovery behavior."""
    def __init__(self, recovery_module):
        self.recovery_module = recovery_module
        
    def compute_velocity_command(self, errors, position_data, current_time, freshness_level="fresh"):
        """Compute velocity commands for recovery behavior."""
        orientation_data = {'yaw': 0.0}  # This will be set by the controller
        cmd_vel, is_complete = self.recovery_module.handle_recovery(
            current_time, position_data, orientation_data
        )
        return [cmd_vel.linear.x, cmd_vel.linear.y, cmd_vel.angular.z]
    
    def get_strategy_name(self):
        return "recovery"


# ControlStrategyFactory: creates the right strategy for the current state
class ControlStrategyFactory:
    """
    PURPOSE: Creates and manages control strategy instances.
    
    This factory class creates the appropriate control strategy based on system
    state, CPU load, and other factors. It implements the Factory pattern to
    dynamically select between standard, simplified, and recovery strategies.
    
    CONTROL STRATEGY SELECTION LOGIC
    ------------------------------
    
    This factory implements the Factory design pattern from software engineering,
    which centralizes the creation of objects and encapsulates the decision logic
    for which type to create. This allows the system to:
    
    1. Dynamically select control strategies at runtime
    2. Keep selection logic in one place
    3. Ensure consistency in control strategy initialization
    4. Easily extend with new strategy types as needed
    
    Strategy Selection Algorithm:
    
    START SELECTION
    
    1. CHECK RECOVERY STATE:
       - Is the robot in recovery mode?
         → YES: Return RecoveryControlStrategy
         → NO: Continue
    
    2. CHECK CRITICAL DATA CONDITIONS:
       - Is sensor data critically stale or missing?
         → YES: Return RecoveryControlStrategy
         → NO: Continue
    
    3. CHECK CPU LOAD CONDITIONS:
       - Is CPU usage extremely high (above max threshold)?
         → YES: Return SimplifiedControlStrategy (Level 0 - minimal)
         → NO: Continue
    
    4. CHECK RESOURCE OPTIMIZATION:
       - Is CPU usage high (above optimization threshold)?
       - AND simplified control enabled?
         → YES: Return SimplifiedControlStrategy (Level 1 or 2)
         → NO: Continue
    
    5. CHECK ERROR MAGNITUDE:
       - Are all errors very small (fine positioning phase)?
       - AND simplified control when possible enabled?
         → YES: Return SimplifiedControlStrategy (Level 2)
         → NO: Continue
    
    6. DEFAULT CASE:
       - Return StandardControlStrategy (full PID with coordination)
    
    END SELECTION
    
    The factory creates and initializes each strategy with the appropriate
    dependencies, ensuring they have everything needed to properly function.
    """
    """Factory for creating appropriate control strategies."""
    def __init__(self, pid_controllers, coordinated_controller, parameter_manager, 
                 distance_error_tracker, lateral_error_tracker, angular_error_tracker,
                 velocity_control, recovery_module):
        self.pid_controllers = pid_controllers
        self.coordinated_controller = coordinated_controller
        self.parameter_manager = parameter_manager
        self.distance_error_tracker = distance_error_tracker
        self.lateral_error_tracker = lateral_error_tracker
        self.angular_error_tracker = angular_error_tracker
        self.velocity_control = velocity_control
        self.recovery_module = recovery_module
        self.last_cmd_vel = (0.0, 0.0, 0.0)
        
    def set_last_cmd_vel(self, cmd_vel):
        """Update the last command velocity."""
        self.last_cmd_vel = cmd_vel
        
    def create_strategy(self, robot_state, computation_level):
        """Create the appropriate control strategy based on state and computation level."""
        if robot_state == "recovery":
            return RecoveryControlStrategy(self.recovery_module)
        
        if computation_level < 3:
            return SimplifiedControlStrategy(
                self.pid_controllers,
                self.parameter_manager,
                self.velocity_control,
                self.last_cmd_vel,
                level=computation_level
            )
            
        # Default to standard control strategy
        return StandardControlStrategy(
            self.pid_controllers,
            self.coordinated_controller,
            self.parameter_manager,
            self.distance_error_tracker,
            self.lateral_error_tracker,
            self.angular_error_tracker,
            self.velocity_control
        )


# PIDControllerNode: the main ROS2 node for PID control
class PIDControllerNode(Node, StateObserver):
    """
    Optimized PID Controller node with modular components.
    
    HOW THE BALL-TRACKING SYSTEM WORKS IN PRACTICE
    ---------------------------------------------
    
    This controller is the central coordinator for the robot's movement. It continuously
    processes sensor data, decides how to move, and controls the motors to follow the ball.
    Let's walk through the entire process with practical examples:
    
    1. TARGET TRACKING:
       The controller receives ball position data from the sensor fusion system.
       
       Raw position data example: x=1.5m (ahead), y=0.2m (left), z=0.1m (up)
       
       The TargetTrackingModule then:
       - Filters this data to remove noise (small fluctuations)
       - Calculates velocity by comparing positions over time
       - Predicts where the ball will be in the near future
       
       Example calculation:
       - Previous position (100ms ago): [1.47m, 0.21m, 0.1m]
       - Current position: [1.5m, 0.2m, 0.1m]
       - Calculated velocity: [0.3m/s, -0.1m/s, 0m/s]
       - Prediction (200ms ahead): [1.56m, 0.18m, 0.1m]
    
    2. ERROR CALCULATION:
       The controller calculates three key errors:
       
       a) Distance Error: How far we are from desired distance
          - Desired distance to ball: 1.0m
          - Current distance: 1.5m
          - Distance error = 0.5m (need to move forward)
       
       b) Lateral Error: Side-to-side alignment error
          - Ball is 0.2m to left of robot's center
          - Lateral error = 0.2m (need to move left)
       
       c) Angular Error: How much we need to rotate to face ball
          - Using arctangent: atan2(y, x) = atan2(0.2, 1.5) ≈ 7.6 degrees
          - Angular error = 7.6 degrees (need to rotate left)
    
    3. STRATEGY SELECTION:
       Based on these errors, we categorize and select a movement strategy:
       
       Categorization:
       - Distance error (0.5m) = "medium" (0.3m-0.6m range)
       - Lateral error (0.2m) = "small" (0.1m-0.25m range)
       - Angular error (7.6°) = "small" (5°-10° range)
       
       Strategy lookup in the table with key ("medium", "small", "small"):
       → Selected strategy: "APPROACH_WITH_MINOR_ALIGNMENT"
       → Strategy parameters: 
          forward_scale=0.9, lateral_scale=0.2, angular_scale=0.4
       → Meaning: Prioritize forward movement, with minor lateral and angular corrections
    
    4. PID COMPUTATION:
       For each dimension, we compute PID control outputs:
       
       Forward (X) calculation with distance_error = 0.5m:
       - P term = Kp * error = 1.0 * 0.5m = 0.5m/s
       - I term = Ki * accumulated_error = 0.05 * 0.3m*s = 0.015m/s
         (accumulated over time to overcome friction)
       - D term = Kd * error_rate = 0.3 * 0.2m/s = 0.06m/s
         (error is decreasing at 0.2m/s, so we reduce deceleration)
       - Raw output = 0.5 + 0.015 + 0.06 = 0.575m/s
       
       Similar calculations for lateral and angular movements.
    
    5. COORDINATION AND SAFETY:
       Now we apply strategy factors and safety constraints:
       
       a) Apply strategy scaling factors:
          - Forward velocity = 0.575m/s * 0.9 = 0.5175m/s
          - Lateral velocity = 0.22m/s * 0.2 = 0.044m/s
          - Angular velocity = 0.15rad/s * 0.4 = 0.06rad/s
          
       b) Apply coordination between dimensions:
          - Angular and lateral movements have same sign (both left)
          - Reduce lateral movement to prevent over-correction
          - Lateral velocity = 0.044m/s * 0.8 = 0.0352m/s
          
       c) Apply safety limits:
          - Check against max velocities
          - Forward capped at 0.5m/s
          - Final command = [0.5m/s, 0.035m/s, 0.06rad/s]
    
    This entire process repeats 10-40 times every second, constantly adapting
    to the ball's movement to create smooth, responsive tracking behavior.
    
    What is a PID Controller?
    ------------------------
    A PID (Proportional-Integral-Derivative) controller is a feedback control system
    that calculates an output based on the error between a desired setpoint and the
    actual measured value. In robotics, PID controllers are widely used for precise
    movement control.
    
    The PID equation has three terms:
    1. Proportional (P): Responds proportionally to current error
       - Example: If you're 1 meter away from the ball, move forward at speed proportional to 1m
       - Acts like a spring pulling the robot toward the target
    
    2. Integral (I): Accumulates past errors to eliminate persistent errors
       - Example: If the robot keeps stopping short of the ball due to friction, 
         the integral term gradually increases to overcome this
       - Acts like a memory of past errors
    
    3. Derivative (D): Responds to the rate of change of error
       - Example: If approaching the ball quickly, start slowing down to prevent overshooting
       - Acts like a dampener to prevent oscillation
    
    For a robot with multiple movement dimensions, we use separate PID controllers for:
    - Forward/backward motion (X axis)
    - Side-to-side motion (Y axis) 
    - Rotational motion (angular Z axis)
    
    How This Node Works:
    ------------------
    This node:
    1. Receives position data from the sensor fusion node
    2. Calculates errors (distance, lateral position, angular orientation)
    3. Applies PID control to each error component
    4. Uses movement strategies to coordinate the three motion dimensions
    5. Adjusts behavior based on data freshness and system resources
    6. Publishes velocity commands to drive the robot
    
    The robot operates in states like:
    - Searching: Looking for the ball
    - Tracking: Following the ball once found
    - Stopped: At the target position
    - Recovery: Handling error conditions
    
    Features included:
    - Adaptive control rate based on CPU load
    - Coordinated movement for smooth trajectories
    - Data freshness monitoring for safety
    - Resource monitoring to prevent system overload
    
    Architecture:
    -----------
    The node uses a modular design with these key components:
    - StateManager: Tracks robot state
    - TargetTracker: Processes target position data
    - PID Controllers: Calculate raw motion commands
    - MovementStrategy: Coordinates movement between dimensions
    - VelocityControl: Applies limits and safety constraints
    - DiagnosticsPublisher: Monitors and reports performance
    
    This architecture separates concerns and allows each component 
    to be understood, tested, and improved independently.
    """
    def __init__(self):
        """
        Initialize the enhanced PID controller node with phased initialization.
        
        The initialization process is broken into phases to ensure dependencies
        are properly established before they're needed:
        
        Phase 1: Parameter initialization
        Phase 2: Core components (resources, timing)
        Phase 3: State management
        Phase 4: Dependent components (controllers, strategies)
        Phase 5: Communications (publishers, subscribers)
        
        This approach makes the initialization sequence easier to understand
        and debug, as each phase has clear responsibilities.
        """
        super().__init__('pid_controller')
        self.throttled_logger = ThrottledLogger(self.get_logger())
        try:
            # Phase 1: Parameter initialization (must come first)
            self._initialize_parameters()
            if self.parameter_manager.debug_level >= 2:
                self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)
            else:
                self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
            
            # Phase 2: Initialize core components
            self._initialize_core_components()
            
            # Phase 3: Initialize state manager and register as observer
            self._initialize_state_manager()
            
            # Phase 4: Initialize dependent components
            self._initialize_dependent_components()
            
            # Phase 5: Setup communications
            self._setup_publishers()
            self._setup_subscriptions()
            self._setup_timers()
            
            # Validate complete initialization
            self._validate_initialization()
            self.get_logger().info("Initialization complete - all components ready")
        except Exception as e:
            self.get_logger().error(f"Initialization failed: {str(e)}")
            raise RuntimeError(f"PID Controller initialization failed: {str(e)}")

    def _initialize_parameters(self):
        """Initialize and validate all parameters."""
        self.parameter_manager = ParameterManager(self)
        # No validation needed here - moved to _validate_initialization
        
    def _initialize_core_components(self):
        """Initialize core components with no dependencies."""
        self.callback_group = ReentrantCallbackGroup()
        self.fast_trig = FastTrigonometry()
        self._init_memory_pools()
        
        # Pre-allocated message objects
        self._cmd_vel_msg = Twist()
        
        # Pre-allocated objects for frequent operations
        self._key_tuple = ["none", "none", "none"]  # Use list instead of tuple for mutability
        
        # Pre-allocated error container
        self._current_errors = [0.0, 0.0, 0.0]  # distance, lateral, angular
        
        # Initialize transform check timer variable
        self.transform_check_timer = None
        
        # Create resource monitor
        self.resource_monitor = ResourceMonitor(throttled_logger, debug_level=self.parameter_manager.debug_level)
        
    def _initialize_state_manager(self):
        """Initialize state manager and register as observer."""
        self.state_manager = EnhancedStateManager()
        self.state_manager.register_observer(self)
        
    def _initialize_dependent_components(self):
        """
        Initialize components that depend on core components.
        
        This method initializes the specialized components that make up the PID controller
        system. Each component has a specific responsibility in the control pipeline:
        
        COMPONENT ARCHITECTURE EXPLAINED:
        --------------------------------
        
        1. TRANSFORM SYSTEM:
           Purpose: Manages coordinate transformations between different robot frames
           Responsibility: Converts position data between different reference frames
           Why it's needed: Position data from different sensors (camera, LIDAR) come in
                          different coordinate systems and need to be unified
        
        2. PID CONTROLLERS:
           Purpose: Calculate control outputs from error values
           Responsibility: Apply PID algorithm to each movement dimension
           Why it's needed: Converts position errors into smooth velocity commands
                          with proportional, integral, and derivative terms
        
        3. TARGET TRACKER:
           Purpose: Process, filter, and predict target position
           Responsibility: Smooth sensor data and predict target movement
           Why it's needed: Raw sensor data can be noisy and needs processing,
                          prediction compensates for system delays
        
        4. STRATEGY MODULE:
           Purpose: Select appropriate movement strategy based on errors
           Responsibility: Choose between different movement patterns
           Why it's needed: Different error patterns require different movement
                          approaches for natural, efficient motion
        
        5. VELOCITY CONTROL:
           Purpose: Apply safety limits and constraints to velocity commands
           Responsibility: Ensure velocity commands are safe and achievable
           Why it's needed: Prevents excessive speeds and ensures smooth acceleration
                          and deceleration for stable movement
        
        6. RECOVERY MODULE:
           Purpose: Handle error conditions and recovery behaviors
           Responsibility: Manage behavior when normal control is not possible
           Why it's needed: Provides safety behaviors when sensors fail or
                          exceptional conditions occur
        
        7. CONTROL STRATEGY FACTORY:
           Purpose: Create and select appropriate control strategies
           Responsibility: Determine which control approach to use based on conditions
           Why it's needed: Allows dynamic switching between different control
                          approaches based on system state and resources
        
        8. PERFORMANCE MONITOR:
           Purpose: Track system performance and adjust parameters
           Responsibility: Monitor CPU usage and adapt control rate
           Why it's needed: Ensures reliable operation on resource-constrained systems
                          by balancing control quality against resource usage
                          
        This architecture follows a modular design with clear separation of concerns,
        making the system maintainable, testable, and adaptable to different robots.
        """
        # 1. TRANSFORM SYSTEM:
        # Purpose: Manages coordinate frame transformations (e.g., camera frame → base frame)
        # This allows us to convert positions between different sensor reference frames
        self.tf_buffer = Buffer()  # Stores transform history
        self.tf_listener = TransformListener(self.tf_buffer, self)  # Listens for transform broadcasts
        
        # Create transform system that handles all coordinate transformations
        self.transform_system = TransformSystem(self, throttled_logger, self.tf_buffer)
        # We need transforms between robot base and IMU (for orientation data)
        self.transform_system.add_transform_dependency("base_link", "imu_link", required=True)
        # Start the transform discovery process
        if not self.transform_system.start_initialization():
            raise RuntimeError("Failed to start transform system initialization")
            
        # 2. PID CONTROLLERS:
        # Purpose: Calculate velocity commands using the PID control algorithm
        # There are separate controllers for forward, lateral, and rotational movement
        self._init_controllers()  # Defined in separate method for clarity
        
        # 3. TARGET TRACKER:
        # Purpose: Process and filter target position data, predict movement
        # Smooths noisy sensor data and compensates for system processing delays
        pm = self.parameter_manager
        self.target_tracker = TargetTrackingModule(
            throttled_logger,
            filter_buffer_size=pm.filter_buffer_size,  # How many position samples to keep
            prediction_horizon=pm.prediction_horizon,  # How far ahead to predict (seconds)
            debug_level=pm.debug_level                 # Logging verbosity
        )
        
        # 4. STRATEGY MODULE:
        # Purpose: Select movement patterns based on error patterns
        # Determines which type of movement is most appropriate for the current situation
        self.strategy_module = MovementStrategyModule(throttled_logger, pm.debug_level)
        
        # 5. VELOCITY CONTROL:
        # Purpose: Apply safety limits and constraints to velocity commands
        # Ensures robot doesn't move too fast and has smooth acceleration/deceleration
        self.velocity_control = VelocityControlModule(
            throttled_logger,
            max_velocity=[
                self.parameter_manager.linear_x_max,     # Maximum forward speed (m/s)
                self.parameter_manager.linear_y_max,     # Maximum lateral speed (m/s)
                self.parameter_manager.angular_max       # Maximum rotation speed (rad/s)
            ]
        )
        # Configure approach behavior - how robot slows down near target
        self.velocity_control.set_approach_parameters(
            pm.approach_distance,      # Distance at which to start slowing down (m)
            pm.min_approach_factor     # Minimum speed factor during final approach
        )
        
        # 6. RECOVERY MODULE:
        # Purpose: Handle error conditions with special behaviors
        # Takes over control during sensor failures or unexpected situations
        self.recovery_module = RecoveryBehaviorModule(throttled_logger)
        
        # 7. CONTROL STRATEGY FACTORY:
        # Purpose: Create and select appropriate control strategies
        # Chooses between full PID, simplified, or recovery control based on conditions
        self.strategy_factory = ControlStrategyFactory(
            self.pid_controllers,           # PID controllers for each dimension
            self.coordinated_controller,    # Coordinates lateral and angular movement
            self.parameter_manager,         # Access to configuration parameters
            self.distance_error_tracker,    # Tracks forward error history
            self.lateral_error_tracker,     # Tracks lateral error history
            self.angular_error_tracker,     # Tracks angular error history
            self.velocity_control,          # Applies safety limits
            self.recovery_module            # Handles error recovery
        )
        
        # 8. PERFORMANCE MONITOR:
        # Purpose: Adapt control behavior based on system performance
        # Adjusts control rate and complexity based on CPU load
        self.performance_monitor = EnhancedPerformanceMonitor(
            self.resource_monitor,          # Monitors CPU, memory usage
            self.parameter_manager,         # Access to configuration parameters
            self.get_logger(),              # Primary logger
            throttled_logger                # Rate-limited logger for high-frequency logs
        )
        
        if hasattr(self.resource_monitor, 'set_rate_limits'):
            self.resource_monitor.set_rate_limits(
                min_rate=pm.min_control_rate,
                max_rate=pm.max_control_rate,
                base_rate=pm.update_rate
            )
        if hasattr(self.resource_monitor, 'set_cpu_thresholds'):
            self.resource_monitor.set_cpu_thresholds(
                low_threshold=pm.cpu_low_threshold,
                high_threshold=pm.cpu_high_threshold
            )
        self.resource_monitor.start()
        
    def _init_memory_pools(self):
        """Setup memory pools and reusable objects for efficiency."""
        # Initialize object pool manager
        self.object_pool = ObjectPoolManager(max_twist=10, max_vector3=15)
        
    def _init_controllers(self):
        """
        Initialize the PID controllers with improved tuning for controlled velocity.
        
        PURPOSE:
        This method sets up the core PID controllers that drive the robot's movement.
        These controllers are the brain of the motion control system, converting
        position errors into smooth, stable velocity commands. This method initializes
        three separate controllers (one for each movement dimension) and connects them
        with error trackers to monitor error trends.
        
        THE THREE PID CONTROLLERS:
        1. Linear X Controller (Forward/Backward):
           - Controls the robot's approach to the target
           - Maintains desired distance from the ball
           - Parameters optimized for smooth acceleration/deceleration
           
        2. Linear Y Controller (Left/Right):
           - Controls the robot's side-to-side alignment with the target
           - Centers the robot on the ball
           - Parameters optimized for precise lateral positioning
           
        3. Angular Controller (Rotation):
           - Controls the robot's orientation to face the target
           - Keeps the ball centered in the robot's view
           - Parameters optimized for smooth rotation without oscillation
           
        COORDINATED CONTROLLER:
        Additionally, creates a special coordinated controller that links
        the lateral (Y) and angular movement. This coordination creates more
        natural movement patterns, similar to how a car turns - combining
        forward motion, steering, and turning in a coordinated way.
        
        ERROR TRACKERS:
        Each controller is paired with an error tracker that:
        - Monitors error trends (increasing, decreasing, oscillating)
        - Provides data for adaptive gain adjustments
        - Helps detect and respond to different movement scenarios
        
        HOW IT'S USED:
        - Called during initialization
        - Creates and configures all PID controllers
        - Sets up gain values and limits from parameters
        - Establishes the coordinated control system
        
        Returns:
            None
            
        Raises:
            InitializationError: If controller initialization fails
        """
        pm = self.parameter_manager
        try:
            self.pid_linear_x, self.distance_error_tracker = PIDControllers.create_controller_with_tracker(
                PIDControllers.ControllerType.LINEAR_X,
                pm.linear_x_kp, pm.linear_x_ki, pm.linear_x_kd,
                pm.linear_x_min, pm.linear_x_max,
                "distance", throttled_logger, max_history=8
            )
            self.pid_linear_y, self.lateral_error_tracker = PIDControllers.create_controller_with_tracker(
                PIDControllers.ControllerType.LINEAR_Y,
                pm.linear_y_kp, pm.linear_y_ki, pm.linear_y_kd,
                pm.linear_y_min, pm.linear_y_max,
                "lateral", throttled_logger, max_history=8
            )
            self.pid_angular, self.angular_error_tracker = PIDControllers.create_controller_with_tracker(
                PIDControllers.ControllerType.ANGULAR,
                pm.angular_kp, pm.angular_ki, pm.angular_kd,
                pm.angular_min, pm.angular_max,
                "angular", throttled_logger, max_history=8
            )
            self.pid_linear_x.validate_initialization()
            self.pid_linear_y.validate_initialization()
            self.pid_angular.validate_initialization()
            
            # Store controllers in a dictionary for easy access
            self.pid_controllers = {
                'linear_x': self.pid_linear_x,
                'linear_y': self.pid_linear_y,
                'angular': self.pid_angular
            }
            
            self.coordinated_controller = PIDControllers.CoordinatedController(
                self.pid_linear_y, 
                self.pid_angular,
                throttled_logger,
                {
                    'coupling_factor': pm.coordinated_coupling_factor,
                    'smoothing_factor': pm.coordinated_smoothing_factor,
                    'min_angle_for_reduction': pm.coordinated_min_angle_for_reduction,
                    'zero_angle_threshold': pm.coordinated_zero_angle_threshold,
                    'max_angle_factor': pm.coordinated_max_angle_factor,
                    'same_sign_scale': pm.coordinated_same_sign_scale,
                    'opposite_sign_scale': pm.coordinated_opposite_sign_scale,
                }
            )
            
            self.get_logger().info("PID controllers initialized successfully with error trackers")
        except Exception as e:
            handle_initialization_error(
                self.get_logger(),
                "Failed to initialize PID controllers",
                e
            )
            
    def _setup_publishers(self):
        """Set up all publishers for this node."""
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            TOPICS["output"]["cmd_vel"],
            10
        )
        
        # Initialize diagnostics publisher
        self.diagnostics_publisher = DiagnosticsPublisher(
            self,
            self.pid_controllers,
            self.target_tracker,
            self.strategy_module,
            self.velocity_control,
            self.resource_monitor,
            self.parameter_manager,
            self.get_logger(),
            throttled_logger
        )
        
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
        
    def _setup_timers(self):
        """Set up timer callbacks for periodic tasks with tiered frequencies."""
        # Main control loop timer
        self.timer = self.create_timer(1.0 / self.parameter_manager.update_rate, self.control_loop_callback)
        
        # Diagnostic timer
        self.diagnostic_timer = self.create_timer(
            1.0 / self.parameter_manager.diagnostics_rate, 
            lambda: self.diagnostics_publisher.publish_diagnostics(self.state_manager)
        )
        
        # Resource monitoring timer
        self.resource_timer = self.create_timer(
            self.parameter_manager.cpu_throttle_interval, 
            self._update_resource_monitoring
        )
    
    def _validate_initialization(self):
        """Validate that all components are properly initialized."""
        pm = self.parameter_manager
        
        # Validate parameters
        required_params = [
            'update_rate', 'min_control_rate', 'max_control_rate',
            'approach_distance', 'min_approach_factor',
            'distance_threshold', 'lateral_threshold', 'angular_threshold'
        ]
        for param in required_params:
            if not hasattr(pm, param) or getattr(pm, param) is None:
                raise ValueError(f"Required parameter '{param}' is not initialized")
                
        # Validate components
        required_components = [
            'resource_monitor', 'transform_system', 'target_tracker',
            'strategy_module', 'velocity_control', 'recovery_module',
            'diagnostics_publisher', 'performance_monitor', 'strategy_factory'
        ]
        for component in required_components:
            if not hasattr(self, component) or getattr(self, component) is None:
                raise RuntimeError(f"Required component '{component}' is not initialized")
    
    # StateObserver Implementation
    def on_state_change(self, old_state, new_state, reason=""):
        """Called when robot state changes."""
        if self.parameter_manager.debug_level >= 1:
            self.throttled_logger.info(
                f"STATE TRANSITION: {old_state} → {new_state} {reason}",
                throttle_duration_sec=LOG_THROTTLE_STATE,
                log_id='state_transition'
            )
        
        # Handle recovery state transitions
        if new_state == "recovery":
            stop_cmd = self.recovery_module.start_recovery()
            self.cmd_vel_pub.publish(stop_cmd)
        elif old_state == "recovery" and new_state != "recovery":
            self.recovery_module.reset()
            if self.parameter_manager.debug_level >= 1:
                self.throttled_logger.info("Exiting recovery mode", throttle_duration_sec=LOG_THROTTLE_STATE, log_id='recovery_exit')
            
        # Complete controller reset when transitioning between tracking and other states
        if new_state == "tracking" or old_state == "tracking":
            self._complete_controller_reset()
            if new_state == "tracking":
                self.state_manager.robot.force_target_reacquisition = True
                
        if new_state != "tracking" and new_state != "searching" and new_state != "lost_ball":
            stop_cmd = self.recovery_module.stop_robot()
            self.cmd_vel_pub.publish(stop_cmd)
    
    def on_freshness_change(self, freshness_level, data_age):
        """Called when data freshness level changes."""
        if freshness_level != "unknown" and self.parameter_manager.debug_level >= 1:
            self.throttled_logger.info(
                f"Data freshness changed: {freshness_level} "
                f"(age: {data_age:.3f}s)",
                throttle_duration_sec=1.0,
                log_id='freshness_change'
            )
            
        # Handle critical freshness
        if freshness_level == "critical" and not self.state_manager.movement.robot_stopped:
            self.get_logger().warning(f"CRITICAL DATA AGE: {data_age:.3f}s - Safety stop triggered")
            stop_cmd = self.recovery_module.stop_robot()
            self.cmd_vel_pub.publish(stop_cmd)
            
    def state_callback(self, msg):
        """Handle robot state updates with improved recovery behavior."""
        new_state = msg.data
        # If state changed, handle the transition using state manager
        if new_state != self.state_manager.robot.state:
            current_time = time.time()
            if self.parameter_manager.debug_level >= 2:
                time_in_state = current_time - self.state_manager.robot.last_state_change_time
                self.get_logger().info(f"Time in state '{self.state_manager.robot.state}': {time_in_state:.2f}s")
            
            # Use state manager to handle transition
            self.state_manager.transition_state(new_state)
    
    def orientation_callback(self, msg):
        """
        Handle orientation updates from the IMU with improved transform handling.
        
        ORIENTATION AND TRANSFORMS FOR BEGINNERS
        --------------------------------------
        
        WHAT IS ORIENTATION?
        Orientation means "which way is the robot facing?" It's usually measured as
        an angle (in radians) around the vertical axis:
        - 0 radians = robot facing forward
        - π/2 (~1.57) radians = robot facing left
        - π (~3.14) radians = robot facing backward
        - 3π/2 (~4.71) radians = robot facing right
        
        WHY DO WE NEED TRANSFORMS?
        Different sensors on the robot have different "viewpoints." For example:
        - The IMU (Inertial Measurement Unit) measures orientation from its position
        - The camera sees the world from where it's mounted
        - The LIDAR scans from its location
        
        If these sensors aren't at the exact center of the robot and perfectly aligned,
        we need to "transform" their measurements to a common reference frame (usually
        the robot's center, called "base_link").
        
        CONCEPTUAL UNDERSTANDING OF THE MATH:
        
        1. REFERENCE FRAMES:
           Think of each sensor having its own coordinate system (or "reference frame"):
           - IMU Frame: Coordinate system centered at the IMU
           - Base Frame: Coordinate system centered at the robot's center
           - Camera Frame: Coordinate system centered at the camera
           
           We need to convert between these frames!
        
        2. QUATERNIONS:
           Quaternions are a mathematical way to represent 3D rotations.
           They use four numbers (x,y,z,w) instead of angles like roll/pitch/yaw.
           
           Why use quaternions?
           - They avoid mathematical problems like "gimbal lock"
           - They make combining rotations easier and more stable
           - They're computationally efficient for 3D rotations
        
        3. THE TRANSFORM PROCESS:
           a) Create a "forward vector" in the original frame
              This is like drawing an arrow pointing forward from the sensor
              
           b) Apply the transform (rotation + translation)
              This is like moving and rotating the arrow to the new frame
              
           c) Calculate the new orientation from the transformed vector
              This is like measuring the angle of the arrow in the new frame
        
        SIMPLIFIED EXAMPLE:
        
        Imagine the IMU is mounted facing 10° to the right of the robot's forward direction:
        - IMU says: "I'm facing 0° (straight ahead)"
        - Transform says: "But the IMU is mounted 10° right of center"
        - After transform: "So the robot is actually facing 10° left"
        
        This process ensures all measurements use the same reference point,
        making sensor fusion and control calculations accurate.
        """
        # Extract yaw (z component) from the Vector3Stamped message
        # This is the raw orientation angle from the IMU (in radians)
        raw_orientation = msg.vector.z
        
        # Store timestamp for freshness checking
        # (We need to know how recent this measurement is for safety)
        self.state_manager.robot.last_orientation_time = time.time()
        
        # Check if transform system is ready before attempting transforms
        # (Transforms might not be available immediately at startup)
        if (not hasattr(self, 'transform_system') or 
            not self.transform_system.is_transform_system_ready()):
            # If transforms aren't ready, use raw orientation as fallback
            self.state_manager.robot.robot_orientation = raw_orientation
            return
        
        # Now we'll transform the orientation from IMU frame to robot's reference frame
        try:
            # Step 1: Get the transform between IMU frame and reference frame
            # This tells us how the IMU is positioned relative to the robot center
            transform = self.transform_system.get_transform_between_frames(
                self.transform_system.imu_frame,            # Source frame (IMU)
                self.transform_system.reference_frame       # Target frame (Robot center)
            )
            
            # Check if we got a valid transform with rotation information
            if (transform and hasattr(transform, 'transform') and 
                hasattr(transform.transform, 'rotation')):
                
                # Step 2: Extract the quaternion components from the transform
                # These four values represent the 3D rotation between frames
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w
                
                # Step 3: Create a "forward" unit vector in the IMU frame
                # This represents "forward" direction as seen by the IMU
                # Use optimized trigonometry if enabled for better performance
                if self.parameter_manager.use_fast_trigonometry:
                    forward_x = self.fast_trig.cos(raw_orientation)
                    forward_y = self.fast_trig.sin(raw_orientation)
                else:
                    forward_x = math.cos(raw_orientation)
                    forward_y = math.sin(raw_orientation)
                forward_z = 0.0  # No vertical component for yaw rotation
                
                # Step 4: Build a rotation matrix from the quaternion
                # This matrix will be used to rotate our vector
                # We pre-calculate common terms for efficiency
                xx = qx * qx
                xy = qx * qy
                xz = qx * qz
                xw = qx * qw
                yy = qy * qy
                yz = qy * qz
                yw = qy * qw
                zz = qz * qz
                zw = qz * qw
                
                # These values form the top two rows of the 3x3 rotation matrix
                # (We only need these rows since we're ignoring the z-component)
                r00 = 1.0 - 2.0 * (yy + zz)
                r01 = 2.0 * (xy - zw)
                r02 = 2.0 * (xz + yw)
                r10 = 2.0 * (xy + zw)
                r11 = 1.0 - 2.0 * (xx + zz)
                r12 = 2.0 * (yz - xw)
                
                # Step 5: Apply the rotation to transform our forward vector
                # This gives us the "forward" direction in the robot's frame
                tx = r00 * forward_x + r01 * forward_y + r02 * forward_z
                ty = r10 * forward_x + r11 * forward_y + r12 * forward_z
                
                # Step 6: Calculate the new orientation angle from the transformed vector
                # atan2 gives us the angle between the x-axis and our vector
                if self.parameter_manager.use_fast_trigonometry:
                    self.state_manager.robot.robot_orientation = self.fast_trig.atan2(ty, tx)
                else:
                    self.state_manager.robot.robot_orientation = math.atan2(ty, tx)
            else:
                # If transform information isn't complete, use raw orientation as fallback
                self.state_manager.robot.robot_orientation = raw_orientation
                
        except Exception as e:
            # In case of error, fall back to raw orientation
            # This ensures we always have some orientation value even if transform fails
            if self.parameter_manager.debug_level >= 2:
                self.get_logger().warning(f"Orientation transform error: {str(e)}")
            self.state_manager.robot.robot_orientation = raw_orientation
        
        # Log orientation updates at high debug level
        if self.parameter_manager.debug_level >= 3:
            # Use pre-computed table for degrees conversion if fast trig is enabled
            if self.parameter_manager.use_fast_trigonometry:
                orientation_degrees = self.state_manager.robot.robot_orientation * 57.29578  # 180/pi
            else:
                orientation_degrees = math.degrees(self.state_manager.robot.robot_orientation)
                            
            self.throttled_logger.info(f"Orientation update: yaw={orientation_degrees:.2f}°", 
                throttle_duration_sec=LOG_THROTTLE_CONTROL, log_id='Orientation_update')
    
    def _is_orientation_fresh(self):
        """
        Check if orientation data is fresh enough to use.
        
        PURPOSE:
        This method determines if the robot's orientation data (which direction it's facing)
        is recent enough to be trustworthy. Orientation data comes from the IMU (Inertial
        Measurement Unit) sensor, and like any sensor data, it becomes less reliable
        the older it gets. Using outdated orientation data could lead to incorrect
        movement decisions.
        
        WHY FRESHNESS MATTERS:
        - Outdated orientation can cause the robot to turn in the wrong direction
        - The robot needs current orientation to calculate angular errors accurately
        - If orientation data is stale, the system may need to use alternative strategies
        
        HOW IT'S USED:
        - Called before using orientation data in calculations
        - Helps determine if the robot can trust its sense of direction
        - Used as a safety check to prevent erroneous movements
        
        FRESHNESS THRESHOLD:
        - Orientation data older than 0.5 seconds is considered stale
        - This threshold is a balance between reliability and availability
        
        Returns:
            bool: True if orientation data is fresh enough to use,
                 False if it's too old or missing
        """
        if self.state_manager.robot.last_orientation_time is None:
            return False
            
        current_time = time.time()
        age = current_time - self.state_manager.robot.last_orientation_time
        
        # Consider orientation data older than 0.5 seconds as stale
        return age < 0.5
    
    def target_callback(self, msg):
        """
        Process incoming target message with event-based control capability.
        
        This enhanced callback not only updates target data but also
        can trigger immediate control cycle for improved responsiveness.
        """
        # Early exit if shutting down
        if self.state_manager.shutting_down:
            return
        
        # Record time for event tracking
        event_time = time.time()
        
        # Update target data in the target tracking module
        update_success = self.target_tracker.update_target(msg, self.parameter_manager.debug_level)
        
        if not update_success:
            return
        
        # Check for fusion rate updates
        if self.parameter_manager.enable_fusion_rate_detection:
            try:
                # Make sure the method exists and handle possible exceptions
                if hasattr(self.target_tracker, 'get_fusion_rate'):
                    fusion_rate, was_updated = self.target_tracker.get_fusion_rate()
                    
                    if was_updated and fusion_rate > 0:  # Ensure rate is positive
                        # Update performance monitor with fusion rate
                        self.performance_monitor.update_fusion_rate(fusion_rate)
                        
                        self.get_logger().info(f"Detected fusion rate: {fusion_rate:.2f}Hz")
            except Exception as e:
                self.get_logger().warning(f"Error getting fusion rate: {str(e)}")
        
        # Event-based control execution
        # Only trigger if we're in tracking mode and it's been long enough since last execution
        try:
            if (self.state_manager.robot.state == "tracking" and 
                (event_time - self.state_manager.perf.last_event_execution) > (1.0 / self.parameter_manager.max_control_rate) and
                (event_time - self.state_manager.perf.last_timer_execution) > 0.05):  # Minimum 50ms between executions
                
                # Execute control loop directly in response to new data
                self.execute_control_cycle(event_triggered=True)
                self.state_manager.perf.last_event_execution = event_time
                self.state_manager.perf.event_control_count += 1
        except Exception as e:
            self.get_logger().error(f"Error in event-based control: {str(e)}")
    
    def _update_resource_monitoring(self):
        """
        Update resource monitoring stats and adapt system behavior.
        
        PURPOSE:
        This method monitors system resources (primarily CPU) and adjusts the
        controller's behavior to prevent overloading the system. Think of it like
        a thermostat for your computer - when things get too hot (high CPU usage),
        it reduces the workload to cool things down and maintain system stability.
        
        RESOURCE MONITORING FEATURES:
        1. CPU Load Tracking: Monitors CPU usage percentage
        2. Cycle Skipping: Can skip processing cycles when CPU is extremely high
        3. Object Pool Monitoring: Tracks memory management efficiency
        4. Adaptive Behavior: Adjusts processing based on resource availability
        
        CYCLE SKIPPING MECHANISM:
        When CPU usage exceeds the max_cpu_skip_threshold (e.g., 90%):
        - The next control cycle is skipped entirely
        - A warning message is logged (limited to once every 2 seconds)
        - This provides an emergency pressure release valve for the CPU
        
        OBJECT POOL STATISTICS:
        In debug mode (debug_level >= 2), periodically logs:
        - Current pool sizes (how many objects are available)
        - Maximum usage (peak demand)
        - Miss count (times new objects had to be created)
        
        HOW IT'S USED:
        - Called periodically on a timer (typically every 0.5 seconds)
        - Works alongside the computation level system to manage resources
        - Provides an additional layer of protection against CPU overload
        - Helps diagnose memory management efficiency in debug mode
        
        Returns:
            None
        """
        try:
            # Update CPU stats if the method exists
            if hasattr(self.resource_monitor, 'update_cpu_stats'):
                self.resource_monitor.update_cpu_stats()
                
                # Check for high CPU - trigger cycle skipping if needed
                if (self.parameter_manager.enable_cycle_skipping and 
                    hasattr(self.resource_monitor, 'current_cpu_usage') and
                    self.resource_monitor.current_cpu_usage > self.parameter_manager.max_cpu_skip_threshold):
                    
                    self.state_manager.skip_next_cycle = True
                    
                    current_time = time.time()
                    if current_time - self.state_manager.perf.last_cpu_warning_time >= 2.0:
                        self.get_logger().warning(
                            f"HIGH CPU LOAD: {self.resource_monitor.current_cpu_usage:.1f}% - skipping next cycle"
                        )
                        self.state_manager.perf.last_cpu_warning_time = current_time
                else:
                    self.state_manager.skip_next_cycle = False
                
            # Log pool statistics periodically if in debug mode
            if self.parameter_manager.debug_level >= 2 and hasattr(self, 'object_pool'):
                pool_stats = self.object_pool.get_stats()
                current_time = time.time()
                if current_time - self.state_manager.perf.last_pool_log_time >= 10.0:
                    pool_msg = (
                        f"Object pool stats: "
                        f"Twist={pool_stats['Twist']['pool_size']}/{pool_stats['Twist']['max_usage']} "
                        f"(misses={pool_stats['Twist']['misses']}), "
                        f"Vector3={pool_stats['Vector3']['pool_size']}/{pool_stats['Vector3']['max_usage']} "
                        f"(misses={pool_stats['Vector3']['misses']}), "
                        f"Float32MultiArray={pool_stats['Float32MultiArray']['pool_size']}/{pool_stats['Float32MultiArray']['max_usage']} "
                        f"(misses={pool_stats['Float32MultiArray']['misses']})"
                    )
                    self.throttled_logger.info(pool_msg, throttle_duration_sec=10.0, log_id='pool_stats')
                    self.state_manager.perf.last_pool_log_time = current_time
        except Exception as e:
            if self.parameter_manager.debug_level >= 1:
                self.get_logger().error(f"Error in resource monitoring: {str(e)}")
    
    def _log_periodic_status(self):
        """
        Log periodic status updates about the controller's operation.
        
        PURPOSE:
        This method provides visibility into the inner workings of the control system
        by periodically logging key performance metrics and state information. This is
        like the dashboard in a car - showing you important information about how
        the system is running without overwhelming you with too much detail.
        
        STATUS INFORMATION LOGGED:
        1. Robot State: Current operational state (tracking, searching, etc.)
        2. Strategy: Current movement strategy being used
        3. CPU Usage: Average CPU utilization percentage
        4. Cycle Time: How long each control cycle takes (milliseconds)
        5. Update Rate: How frequently the controller is running (Hz)
        6. Control Mode: Whether using simplified or full control
        7. Data Freshness: How recent the sensor data is
        8. Event-Driven %: Percentage of control cycles triggered by events vs. timer
        9. Skip Count: Number of cycles that were skipped due to high CPU load
        
        WHY THIS MATTERS:
        - Helps diagnose performance issues during operation
        - Provides insight into the controller's decision-making
        - Allows monitoring resource usage and efficiency
        - Helps verify that the system is operating as expected
        
        HOW IT'S USED:
        - Called periodically during the control loop
        - Only logs when debug_level >= 1
        - Uses throttled logging to avoid flooding the log
        - Provides a summary view of the system's current state
        
        Returns:
            None
        """
        if self.parameter_manager.debug_level < 1:
            return
        perf_stats = self.resource_monitor.get_performance_stats()
        if not perf_stats or not all(k in perf_stats for k in ['cpu_avg', 'cycle_time_ms', 'update_rate']):
            perf_stats = {
                'cpu_avg': 0.0,
                'cycle_time_ms': 0.0,
                'update_rate': getattr(self.parameter_manager, 'update_rate', 3.0),
                'skips': 0
            }
        strategy_name = getattr(self.strategy_module, 'current_strategy', 'unknown')
        total_cycles = max(1, self.state_manager.perf.event_control_count + self.state_manager.perf.timer_control_count)
        event_ratio = self.state_manager.perf.event_control_count / total_cycles * 100.0
        status_msg = (
            f"Status: Robot state={self.state_manager.robot.state}, "
            f"Strategy={strategy_name}, "
            f"CPU={perf_stats['cpu_avg']:.1f}%, "
            f"Cycle time={perf_stats['cycle_time_ms']:.2f}ms, "
            f"Rate={perf_stats['update_rate']:.1f}Hz, "
            f"Simplified={self.state_manager.movement.using_simplified_control}, "
            f"Freshness={self.state_manager.freshness.level}, "
            f"Event-driven={event_ratio:.1f}%, "
            f"Skips={self.state_manager.perf.skipped_cycle_count}"
        )
        if self.parameter_manager.debug_level >= 1:
            self.throttled_logger.info(status_msg, throttle_duration_sec=LOG_THROTTLE_CONTROL, log_id='periodic_status')
    
    def control_loop_callback(self):
        """
        Regular control loop to calculate and publish velocity commands with CPU optimization.
        
        The Control Loop: Heart of the Robot
        ----------------------------------
        This function is the central "brain" of the robot's movement system, executing
        multiple times per second to continuously update velocity commands based on 
        the latest target information.
        
        Think of this loop as similar to how your brain constantly adjusts your walking
        speed and direction when chasing a ball - you don't consciously calculate
        each muscle movement, but continuously adapt based on the ball's position.
        
        Control Loop Flow:
        ----------------
        1. Check if we should execute this cycle
           - Skip if shutting down
           - Skip if waiting for transforms
           - Skip if CPU is overloaded (adaptive rate control)
        
        2. Execute the control cycle
           - Get current time and calculate time step (dt)
           - Check data freshness (is sensor data recent enough?)
           - Calculate errors (distance, lateral, angular)
           - Apply appropriate control strategy based on errors
           - Compute velocity commands using PID algorithms
           - Apply safety limits to velocity commands
           - Publish velocity commands to move the robot
           - Update performance metrics
        
        3. Handle any errors safely
           - Catch exceptions to prevent crashes
           - Log detailed error information
           - Stop the robot safely if an error occurs
        
        Performance Optimization:
        -----------------------
        This loop includes several optimizations for resource-constrained systems:
        - Adaptive execution rate based on CPU load
        - Cycle skipping when system is under heavy load
        - Simplified calculations when high precision isn't needed
        - Early returns to avoid unnecessary processing
        
        This control loop typically runs between 10-40Hz (10-40 times per second)
        depending on system load and configuration parameters.
        """
        try:
            # Skip if shutting down
            if self.state_manager.shutting_down:
                return
            
            # Skip if transform system is not initialized and required
            if (hasattr(self, 'transform_system') and 
                not self.transform_system.is_transform_system_ready()):
                
                # Log at most once per 20 cycles to avoid spamming
                if self.state_manager.robot.cycle_count % 20 == 0:
                    status = self.transform_system.get_status()
                    self.get_logger().warn(
                        f"Control loop waiting for transform initialization: {status['message']}"
                    )
                    
                    # Try to restart initialization if it's in error state
                    if status['status'] == TransformStatus.ERROR:
                        self.get_logger().warn("Attempting to restart transform initialization")
                        self.transform_system.start_initialization()
                
                # Still increment cycle count
                self.state_manager.robot.cycle_count += 1
                return
                    
            # Apply adaptive rate control if enabled
            if (self.parameter_manager.adaptive_control_rate and 
                hasattr(self.resource_monitor, 'current_cpu_usage') and 
                hasattr(self.parameter_manager, 'update_rate')):
                
                # Calculate adaptive rate
                adaptive_rate = self.performance_monitor.calculate_adaptive_rate(
                    self.state_manager,
                    self.parameter_manager.update_rate, 
                    self.resource_monitor.current_cpu_usage
                )
                
                # Check if we should skip this cycle
                if self.performance_monitor.should_skip_cycle(
                    self.resource_monitor.current_cpu_usage,
                    self.state_manager.last_control_time,
                    adaptive_rate
                ):
                    # Increment skipped cycle count
                    self.state_manager.perf.skipped_cycle_count += 1
                    return
            
            # Execute the control cycle
            self.execute_control_cycle(event_triggered=False)
                    
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.get_logger().error(f"Unexpected error in control_loop_callback: {str(e)}\nStack trace:\n{stack_trace}")
            
            # Try to safely stop the robot
            try:
                self.recovery_module.stop_robot()
            except Exception as stop_error:
                self.get_logger().error(f"Failed to stop robot after error: {str(stop_error)}")
    
    def execute_control_cycle(self, event_triggered=False):
        """
        Execute one complete control cycle with improved error handling and state management.
        
        This method contains the actual implementation of the PID control loop. Each cycle
        performs a complete sense-think-act process:
        
        SENSE → THINK → ACT Cycle:
        -------------------------
        
        SENSE: Gather and validate information
        - Check transform system readiness
        - Calculate time since last cycle
        - Check data freshness and quality
        - Get latest target position from fusion system
        
        THINK: Process information and make decisions
        - Calculate current errors (distance, lateral, angular)
        - Determine if robot should stop or move
        - Choose appropriate movement strategy
        - Run PID calculations for each movement dimension
        - Apply coordinated control between dimensions
        - Apply safety limits and constraints
        
        ACT: Execute the decided actions
        - Create velocity command message
        - Publish command to robot drive system
        - Update state information
        - Log diagnostic information
        
        Mathematical Overview:
        --------------------
        For each dimension (x, y, angular), we calculate:
        
        Error = TargetPosition - CurrentPosition
        
        P term = Kp * Error
        
        I term = Ki * ∫(Error dt)
           (Approximated as Ki * sum(Error * dt))
        
        D term = Kd * (Error - PreviousError)/dt
           (Rate of change of error)
        
        Output = P term + I term + D term
        
        Then outputs are coordinated between dimensions to create
        smooth, natural movement that can follow a moving target.
        
        Args:
            event_triggered: Boolean indicating if this cycle was triggered
                            by an event (vs. a timer), affects performance tracking
        """
        try:
            # Skip if shutting down
            if self.state_manager.shutting_down:
                return
            
            # Skip if transform system is not initialized and required
            if (hasattr(self, 'transform_system') and 
                not self.transform_system.is_transform_system_ready()):
                return
            
            # Track execution time source for metrics
            if event_triggered:
                self.state_manager.perf.last_event_execution = time.time()
            else:
                self.state_manager.perf.last_timer_execution = time.time()
                self.state_manager.perf.timer_control_count += 1
            
            # Mark cycle start for performance tracking
            cycle_start_time = time.time()
            
            # Calculate dt since last control
            current_time = time.time()
            dt = current_time - self.state_manager.last_control_time
            # Ensure dt is reasonable (protect against time jumps)
            if dt > 1.0:
                self.get_logger().warning(f"Large time step detected: {dt:.3f}s - capping at 0.1s")
                dt = 0.1  # Cap at 100ms to prevent instability
            elif dt <= 0.0:
                dt = 0.001  # Ensure positive dt to prevent division by zero
            
            self.state_manager.last_control_time = current_time
            
            # Increment cycle counter
            self.state_manager.robot.cycle_count += 1
            
            # Check data freshness - critical for safety
            is_fresh, freshness_level, data_age = self._check_data_freshness()
            
            # Update state manager with freshness info
            self.state_manager.update_freshness(freshness_level, data_age)
            
            # Log periodic status updates (once every 50 cycles)
            if self.state_manager.robot.cycle_count % 50 == 0:
                self._log_periodic_status()
            
            # Handle critical data freshness (prioritized over other state handling)
            if freshness_level == "critical":
                if not self.state_manager.movement.robot_stopped:
                    self.get_logger().warning(f"CRITICAL DATA AGE: {data_age:.3f}s - Safety stop triggered")
                    stop_cmd = self.recovery_module.stop_robot()
                    self.cmd_vel_pub.publish(stop_cmd)  # Ensure command is published
                    self.state_manager.movement.robot_stopped = True
                    self.state_manager.movement.stop_time = time.time()
                return  # Exit early when data is critically stale
                
            # Special handling for recovery mode (higher priority than other states)
            if self.state_manager.robot.state == "recovery":
                position_data = self.target_tracker.get_position_data()
                orientation_data = {'yaw': self.state_manager.robot.robot_orientation}
                cmd_vel, is_complete = self.recovery_module.handle_recovery(
                    time.time(), position_data, orientation_data
                )
                self.cmd_vel_pub.publish(cmd_vel)
                if is_complete:
                    self.recovery_module.reset()
                    self.get_logger().info("Recovery sequence completed")
                return
            
            # Handle non-tracking states (searching/lost_ball have their own handling)
            if self.state_manager.robot.state != "tracking":
                if self._handle_non_tracking_state():
                    return  # Exit after handling non-tracking state
            
            # Check if orientation data is fresh (prevents race conditions)
            if not self._is_orientation_fresh():
                if self.parameter_manager.debug_level >= 2:
                    self.get_logger().warning("Skipping control cycle - orientation data is stale")
                return
            
            # Determine computation level needed
            # The key CPU optimization - avoid expensive calculations when appropriate
            if self.parameter_manager.use_simplified_control_when_possible:
                computation_level = self._determine_computation_level()
                self.state_manager.movement.computation_level = computation_level
                
                # Create appropriate control strategy based on computation level
                current_strategy = self.strategy_factory.create_strategy(
                    self.state_manager.robot.state,
                    computation_level
                )
                
                if computation_level < 3:
                    self.state_manager.movement.using_simplified_control = True
                    self.state_manager.perf.simplified_control_count += 1
                else:
                    self.state_manager.movement.using_simplified_control = False
                    # Record that we did full computation
                    self.state_manager.movement.last_full_computation_time = time.time()
            else:
                # Standard control path
                self.state_manager.movement.using_simplified_control = False
                current_strategy = self.strategy_factory.create_strategy(
                    self.state_manager.robot.state,
                    3  # Full computation
                )
            
            # Calculate current errors
            distance, lateral, bearing, angular_degrees = self._calculate_errors()
            
            if self.parameter_manager.debug_level >= 2:
                debug_msg = (
                    f"PRE-STOP CHECK: distance={distance:.3f}m (target={self.parameter_manager.desired_distance:.3f}m), "
                    f"lateral={lateral:.3f}m, angular={angular_degrees:.2f}°, "
                    f"is_stopped={self.state_manager.movement.robot_stopped}"
                )
                self.get_logger().info(debug_msg)
            
            # Check stop conditions and handle if needed
            if self._handle_stop_conditions(distance, lateral, angular_degrees, dt):
                return  # Exit if stop conditions were met
            
            # Get position data for control strategy
            position_data = self.target_tracker.get_position_data()
            
            # Compute velocities using selected strategy
            velocities = current_strategy.compute_velocity_command(
                self._current_errors,
                position_data, 
                current_time,
                freshness_level
            )
            
            # Update strategy factory with last velocity
            self.strategy_factory.set_last_cmd_vel(velocities)
            
            # Guard against NaN values which can cause silent failures
            if any(math.isnan(v) for v in velocities):
                self.get_logger().error("NaN velocity detected - resetting to zero")
                velocities = [0.0, 0.0, 0.0]
            
            # Publish command
            cmd_vel_msg = self._cmd_vel_msg
            cmd_vel_msg.linear.x = float(velocities[0])
            cmd_vel_msg.linear.y = float(velocities[1])
            cmd_vel_msg.angular.z = float(velocities[2])
            self.cmd_vel_pub.publish(cmd_vel_msg)
            
            # Update last command velocity in state manager
            self.state_manager.movement.last_cmd_vel = (velocities[0], velocities[1], velocities[2])
            
            # Update performance stats
            cycle_duration = time.time() - cycle_start_time
            self.performance_monitor.update_performance_stats(cycle_duration)
                
        except Exception as e:
            stack_trace = traceback.format_exc()
            self.get_logger().error(f"Unexpected error in execute_control_cycle: {str(e)}\nStack trace:\n{stack_trace}")
            
            # Try to safely stop the robot
            try:
                stop_cmd = self.recovery_module.stop_robot()
                self.cmd_vel_pub.publish(stop_cmd)  # Ensure the stop command is published
            except Exception as stop_error:
                self.get_logger().error(f"Failed to stop robot after error: {str(stop_error)}")
    
    def _check_data_freshness(self):
        """
        Check the freshness of target data and update system state accordingly.
        
        PURPOSE:
        This method determines how recent and reliable our sensor data is. Data freshness
        is critical for safety - if our data is old or missing, we need to adjust our
        behavior or even stop the robot entirely. Think of it like checking the expiration
        date on food before eating it.
        
        DATA FRESHNESS LEVELS:
        - "fresh": Data is recent and reliable (full-speed operation allowed)
        - "stale": Data is getting old (reduced speed for safety)
        - "critical": Data is too old to trust (robot stops for safety)
        - "invalid": Data is missing or corrupted (robot stops for safety)
        
        HOW IT'S USED:
        - Called at the start of each control cycle
        - Affects velocity scaling in the control strategies
        - Triggers safety stops when data is critically old
        - Updates freshness state in the state manager
        
        Returns:
            tuple: (is_fresh, freshness_level, age)
                is_fresh: Boolean indicating if data is fresh enough to use
                freshness_level: String categorization ("fresh", "stale", "critical", "invalid")
                age: Float age of the data in seconds
        """
        try:
            # Get freshness information from the target tracker
            is_fresh, freshness_level, data_age = self.target_tracker.is_target_fresh(
                max_age=self.parameter_manager.fresh_data_timeout
            )
            
            # Validate returned data (guard against unexpected return values)
            if not isinstance(freshness_level, str):
                self.get_logger().warning(f"Invalid freshness level type: {type(freshness_level)}")
                freshness_level = "critical"  # Default to critical for safety
            if not isinstance(data_age, (int, float)):
                self.get_logger().warning(f"Invalid data age type: {type(data_age)}")
                data_age = 999.0  # Default to high age for safety
            
            return is_fresh, freshness_level, data_age            
        except Exception as e:
            self.get_logger().error(f"Error checking data freshness: {str(e)}")
            # Default to critical for safety in case of errors
            return False, "critical", 999.0
    
    def _calculate_errors(self):
        """
        Calculate tracking errors using filtered position values.
        
        PURPOSE:
        This method computes the three key errors that drive the PID controller:
        1. Distance Error: How far we are from the desired distance to the ball
           (positive = too far, negative = too close)
        2. Lateral Error: How far left/right we are from the ball's center
           (positive = ball is to the left, negative = ball is to the right)
        3. Angular Error: How far we need to rotate to face the ball
           (positive = need to turn left, negative = need to turn right)
        
        These errors are the foundation of the entire control system - they represent
        the difference between where we are and where we want to be.
        
        HOW IT'S USED:
        - Called during each control cycle
        - Provides input values to the PID controllers
        - Used to determine if the robot should stop or move
        - Stored in error trackers for trend analysis
        
        MATH EXPLANATION:
        - Distance Error = Current Distance - Desired Distance
          (We want to maintain a specific distance from the ball)
        - Lateral Error = Current Lateral Position - 0
          (We want to be directly centered on the ball)
        - Angular Error = Current Bearing
          (We want to face the ball directly, so bearing should be 0)
        
        Returns:
            tuple: (distance, lateral, bearing, angular_degrees)
                distance: Current distance to target (meters)
                lateral: Current lateral offset (meters)
                bearing: Current angular offset (radians)
                angular_degrees: Current angular offset (degrees, for logging)
        """
        # Get filtered position data from target tracker
        position_data = self.target_tracker.get_position_data()
        
        # Handle the case where position data is missing or incomplete
        if not position_data or not all(k in position_data for k in ['distance', 'lateral', 'bearing']):
            # Set default values
            return 0.0, 0.0, 0.0, 0.0
        
        # Extract position values directly to local variables to avoid repeated dict lookups
        distance = position_data['distance']
        lateral = position_data['lateral']
        bearing = position_data['bearing']
        
        # Calculate errors using pre-allocated array
        self._current_errors[0] = distance - self.parameter_manager.desired_distance  # distance_error
        self._current_errors[1] = lateral - 0.0                    # lateral_error 
        self._current_errors[2] = bearing                          # angular_error
        
        # Convert angular error to degrees for logging
        # Use optimized conversion if fast trigonometry is enabled
        if self.parameter_manager.use_fast_trigonometry:
            angular_degrees = bearing * 57.29578  # 180/pi - faster than math.degrees
        else:
            angular_degrees = math.degrees(bearing)
        
        return distance, lateral, bearing, angular_degrees
    
    def _handle_non_tracking_state(self):
        """
        Handle robot behavior when not in tracking mode.
        
        PURPOSE:
        This method defines what the robot should do when it's not actively tracking
        a ball. Different states require different behaviors - for example, when
        the robot is initializing or in an error state, it should stop moving entirely,
        but when searching for a ball, it might need to keep moving.
        
        STATE-SPECIFIC BEHAVIORS:
        - "searching" or "lost_ball": Allow movement (controlled by other nodes)
          The robot is actively looking for the ball, so it may need to move
          around to find it. This movement is typically controlled by a different
          part of the system.
          
        - All other non-tracking states: Stop the robot completely
          These include "initializing", "error", "recovery", etc. In these states,
          the robot should stop for safety and to avoid unexpected behavior.
        
        HOW IT'S USED:
        - Called at the beginning of each control cycle
        - Checks the current robot state before attempting tracking
        - Ensures appropriate behavior when not actively tracking
        - Prevents unintended movement in non-tracking states
        
        Returns:
            bool: True to indicate that the method handled the situation
                 (control loop should exit early after this)
        """
        # When not tracking, ensure robot is stopped (unless controlled by another node)
        if self.state_manager.robot.state not in ["searching", "lost_ball"]:
            stop_cmd = self.recovery_module.stop_robot()
            self.cmd_vel_pub.publish(stop_cmd)
        return True  # Indicate that the method handled the situation
    
    def _handle_stop_conditions(self, distance, lateral, angular_degrees, dt):
        """
        Check and handle stop conditions if the robot should stop moving.
        
        PURPOSE:
        This method determines whether the robot should stop moving (when it has reached
        the target position) or start moving (when it has drifted away from the target).
        It coordinates the stopping behavior and ensures smooth transitions between
        moving and stopped states.
        
        HOW IT WORKS:
        1. First checks if a stopped robot needs to start moving again
           (_reset_stopped_state_if_needed)
        2. If not, checks if a moving robot needs to stop
           (_evaluate_stop_conditions)
        3. If stopping is needed, sends stop command and updates state
        4. Regardless of stop/start decision, updates error trackers
        
        HOW IT'S USED:
        - Called during each control cycle
        - Manages the transition between moving and stopped states
        - Controls when to send stop commands to the motors
        - Updates error trackers with latest data
        
        Parameters:
            distance: float - Current distance to target (meters)
            lateral: float - Current lateral offset (meters)
            angular_degrees: float - Current angular offset (degrees)
            dt: float - Time since last cycle (seconds)
            
        Returns:
            bool: True if stop conditions were handled (robot stopped),
                 False if normal control should continue
        """
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
                distance, lateral, angular_degrees, self.state_manager.movement.robot_stopped
            )
            
            if should_stop:
                if not self.state_manager.movement.robot_stopped:
                    self.get_logger().info(stop_reason)
                    self.state_manager.movement.robot_stopped = True
                    self.state_manager.movement.stop_time = time.time()
                    # Generate stop command
                    stop_cmd = self.recovery_module.stop_robot()
                    # Publish stop command to actually stop the robot
                    self.cmd_vel_pub.publish(stop_cmd)
                return True  # Handled by stopping the robot
        
        # Update error trackers
        self.distance_error_tracker.update(self._current_errors[0], dt)
        self.lateral_error_tracker.update(self._current_errors[1], dt)
        self.angular_error_tracker.update(self._current_errors[2], dt)
        
        return False  # Not handled, continue with normal control
    
    def _evaluate_stop_conditions(self, distance, lateral, angular_degrees, is_stopped):
        """
        Evaluate if the robot should stop based on current errors.
        
        PURPOSE:
        This method determines if the robot has reached its target position accurately
        enough to stop moving. It implements a hysteresis system (different thresholds
        for stopping vs. starting) to prevent oscillation around the target position.
        Think of it like a thermostat that turns off at 72°F but doesn't turn on again
        until temperature drops to 68°F.
        
        ERROR THRESHOLDS:
        - Distance threshold: How close to target distance is "close enough"
        - Lateral threshold: How centered on the ball is "centered enough" 
        - Angular threshold: How directly facing the ball is "facing enough"
        
        HYSTERESIS SYSTEM:
        - When moving: Uses stricter thresholds to stop (precision)
        - When stopped: Uses more lenient thresholds to start (stability)
        - The longer the robot stays stopped, the larger the error needed to start moving
        
        HOW IT'S USED:
        - Called by _handle_stop_conditions during each control cycle
        - Determines when the robot has reached its target position
        - Prevents jittery movement when near the target
        - Provides human-readable reason for stopping or moving
        
        Parameters:
            distance: float - Current distance to target (meters)
            lateral: float - Current lateral offset (meters)
            angular_degrees: float - Current angular error in degrees
            is_stopped: bool - Whether the robot is currently stopped
            
        Returns:
            tuple: (should_stop, reason)
                should_stop: bool - True if robot should stop, False if it should move
                reason: str - Human-readable explanation of the decision
        """
        # Calculate error values
        distance_error = abs(distance - self.parameter_manager.desired_distance)
        lateral_error = abs(lateral)
        angular_error = abs(angular_degrees)
        
        # Start with base thresholds
        distance_threshold = self.parameter_manager.distance_threshold
        lateral_threshold = self.parameter_manager.lateral_threshold
        angular_threshold = self.parameter_manager.angular_threshold
        
        # Increase angular threshold when at target distance
        if distance_error < self.parameter_manager.distance_threshold * 1.5:
            angular_threshold *= self.parameter_manager.angular_at_target_factor
            if self.parameter_manager.debug_level >= 2:
                threshold_msg = f"At target distance: increased angular threshold to {angular_threshold:.2f}°"
                self.throttled_logger.info(threshold_msg, throttle_duration_sec=LOG_THROTTLE_DIAG, log_id='angular_thresholds')
        
        # Apply state-dependent hysteresis
        if is_stopped:
            # Higher thresholds to start moving (requires larger errors)
            hysteresis = 1.5 + self.state_manager.movement.movement_hysteresis
            distance_threshold = min(distance_threshold * hysteresis, 0.2)
            lateral_threshold = min(lateral_threshold * hysteresis, 0.15)
            angular_threshold = min(angular_threshold * hysteresis, 15.0)
        else:
            # Lower thresholds to stop (more precision when near target)
            hysteresis = 0.8
            distance_threshold *= hysteresis
            lateral_threshold *= hysteresis
            angular_threshold *= hysteresis
        
        # Log thresholds for debugging
        if self.parameter_manager.debug_level >= 2:
            debug_msg = (
                f"Stop thresholds: d={distance_error:.3f}/{distance_threshold:.3f}, "
                f"l={lateral_error:.3f}/{lateral_threshold:.3f}, "
                f"a={angular_error:.2f}/{angular_threshold:.2f}"
            )
            self.throttled_logger.info(debug_msg, throttle_duration_sec=LOG_THROTTLE_DIAG, log_id='stop_thresholds')
            
        
        # Check if any error exceeds thresholds
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
        
        # Accumulate hysteresis for sustained stops
        if not is_stopped:
            self.state_manager.movement.movement_hysteresis += 0.05
            self.state_manager.movement.movement_hysteresis = min(0.3, self.state_manager.movement.movement_hysteresis)  # Cap at 0.3
        
        return True, reason  # Return True to indicate robot SHOULD stop
    
    def _reset_stopped_state_if_needed(self, distance_error, lateral_error, angular_error):
        """
        Reset stopped state if significant movement is required.
        
        PURPOSE:
        This method checks if a stopped robot needs to start moving again because it has
        drifted too far from the target position. This can happen if the ball moves or
        if the robot is bumped or pushed. The method implements time-dependent hysteresis,
        where the longer the robot has been stopped, the more error is needed to trigger
        movement (preventing jitter from minor disturbances).
        
        KEY FEATURES:
        - First-movement boost: Provides easier first movement after startup
        - Time-dependent hysteresis: Higher thresholds the longer robot is stopped
        - Distance-aware angular threshold: More lenient angular errors at target distance
        - Different threshold scaling for each error type
        
        HOW IT'S USED:
        - Called by _handle_stop_conditions at the start of each control cycle
        - If the robot is already moving, does nothing
        - If the robot is stopped, checks if errors are large enough to start moving
        - Resets movement hysteresis when movement starts
        
        WHEN MOVEMENT STARTS:
        - Any error (distance, lateral, angular) exceeding its threshold will
          trigger movement
        - All thresholds are calculated with hysteresis based on stop duration
        - When moving starts, movement_hysteresis is reset to zero
        
        Parameters:
            distance_error: float - Error in distance (meters)
            lateral_error: float - Error in lateral position (meters)
            angular_error: float - Error in angular position (degrees)
            
        Returns:
            bool: True if stopped state was reset (robot started moving),
                 False otherwise (robot remains stopped)
        """
        # If already moving, no need to reset
        if not self.state_manager.movement.robot_stopped:
            return False
        
        # Handle initialization for first movement after startup
        if self.state_manager.movement.initial_movement_boost:
            hysteresis = 0.5  # Much lower hysteresis for first movement
            self.state_manager.movement.initial_movement_boost = False
        else:
            # Regular hysteresis calculation
            stop_duration = time.time() - self.state_manager.movement.stop_time
            hysteresis = min(1.1, 1.0 + stop_duration * 0.1)
        
        # Calculate movement thresholds with hysteresis
        distance_threshold = self.parameter_manager.distance_threshold * hysteresis * 0.7
        lateral_threshold = self.parameter_manager.lateral_threshold * hysteresis * 0.7
        
        # Adjust angular threshold based on distance to target
        if abs(distance_error) < self.parameter_manager.distance_threshold * 1.2:
            # More lenient when at target distance
            angular_threshold = self.parameter_manager.angular_threshold * self.parameter_manager.angular_at_target_factor * hysteresis
        else:
            # Standard threshold otherwise
            angular_threshold = self.parameter_manager.angular_threshold * hysteresis
        
        # Check if any error exceeds thresholds
        if (abs(distance_error) > distance_threshold or
            abs(lateral_error) > lateral_threshold or
            abs(angular_error) > angular_threshold):
            
            # Log the decision to exit stopped state
            log_msg = (
                f"Exiting stopped state - Movement required: "
                f"distance_error={distance_error:.3f}m(threshold={distance_threshold:.3f}), "
                f"lateral_error={lateral_error:.3f}m(threshold={lateral_threshold:.3f}), "
                f"angular_error={angular_error:.2f}°(threshold={angular_threshold:.2f})"
            )
            if self.parameter_manager.debug_level >= 1:
                self.throttled_logger.info(log_msg, throttle_duration_sec=2.0, log_id='exit_stopped')
            
            # Reset stopped state
            self.state_manager.movement.robot_stopped = False
            
            # Reset movement hysteresis
            self.state_manager.movement.movement_hysteresis = 0.0
            
            return True
        
        return False
    
    def _complete_controller_reset(self):
        """
        Complete reset of all controllers and error states.
        
        PURPOSE:
        This method performs a thorough reset of the entire control system, bringing
        it back to a clean initial state. This is like pressing a "reset" button on
        the robot's brain - clearing out all accumulated errors, trends, and temporary
        data to start fresh. This is important when changing states or when significant
        events occur that make previous calculations invalid.
        
        WHAT GETS RESET:
        1. All PID Controllers: Clears P, I, D terms and internal states
        2. Coordinated Controller: Resets coordination logic
        3. Error Trackers: Clears error history and trends
        4. Strategy Module: Resets movement strategy selection
        5. Velocity Control: Resets velocity limits and smoothing
        6. Target Tracker: Resets target position filtering and prediction
        7. Movement State: Resets hysteresis, computation levels, and stopped state
        8. Freshness State: Resets data freshness tracking
        
        HOW IT'S USED:
        - Called during state transitions (especially entering/exiting tracking state)
        - Called during recovery operations
        - Used when the control system needs a fresh start
        - Ensures the robot doesn't carry old state information into new situations
        
        Returns:
            None
        """
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
        # Reset strategy module
        self.strategy_module.reset()
        # Reset velocity control module (centralized state)
        self.velocity_control.reset()
        # Reset target tracker if needed
        if hasattr(self, 'target_tracker'):
            if hasattr(self.target_tracker, 'reset'):
                self.target_tracker.reset()
        # Reset movement hysteresis
        self.state_manager.movement.movement_hysteresis = 0.0
        # Set stopped state
        self.state_manager.movement.robot_stopped = True
        self.state_manager.movement.stop_time = time.time()
        # Reset computation tracking
        self.state_manager.movement.using_simplified_control = False
        self.state_manager.movement.last_full_computation_time = time.time()
        # Reset data freshness tracking
        self.state_manager.freshness.level = "unknown"
        self.get_logger().info("Complete controller reset performed")
    
    def _determine_computation_level(self):
        """
        Determine the level of computation needed for the current cycle.
        
        PURPOSE:
        This method dynamically adjusts how much computational complexity to use
        based on CPU load and tracking requirements. It allows the system to scale
        back processing when the CPU is overloaded, while ensuring critical operations
        still happen. Think of it like a car's engine management system that balances
        performance and efficiency based on current needs.
        
        COMPUTATION LEVELS:
        - Level 0: Minimal computation (emergency mode under extreme CPU load)
          Just basic damping of previous velocities, no PID calculation
        - Level 1: Basic computation (high CPU load)
          Simple proportional control without coordination or prediction
        - Level 2: Moderate computation (moderate CPU load)
          Basic PID with limited coordination and prediction
        - Level 3: Full computation (normal operation)
          Complete PID with full coordination, prediction, and optimization
        
        DECISION FACTORS:
        1. CPU Usage: Primary factor - higher usage means lower computation
           - >95% CPU: Level 0 (minimal)
           - >90% CPU: Max Level 1
           - >85% CPU: Max Level 2
           - ≤85% CPU: Level 3 (full)
        2. Time Since Full Computation: Ensures periodic full accuracy
           - Forces Level 3 at least every 0.5 seconds
        3. Error Magnitude: Adapts to tracking requirements
           - Larger errors can justify higher computation levels
        
        HOW IT'S USED:
        - Called during each control cycle
        - Determines which control strategy to use
        - Affects how much CPU time the control loop consumes
        - Helps ensure controller doesn't overload the CPU
        
        Returns:
            int: 0-3 indicating computation level (0=minimal, 3=full)
                Higher values = more computation/better accuracy
                Lower values = less computation/less CPU usage
        
        Notes:
            - Full computation (level 3) is now allowed up to 85% CPU usage.
            - Scaling down only starts above 85% CPU.
            - The system always returns to full computation periodically to
              maintain accuracy, even under high load.
        """
        computation_level = 3
        cpu_usage = self.resource_monitor.current_cpu_usage
        # Adjusted thresholds for high baseline CPU
        if cpu_usage > 95.0:
            return 0  # Minimal computation only at extreme load
        elif cpu_usage > 90.0:
            computation_level = min(computation_level, 1)
        elif cpu_usage > 85.0:
            computation_level = min(computation_level, 2)
        # Always perform full computation periodically to ensure accuracy
        time_since_full = time.time() - self.state_manager.movement.last_full_computation_time
        if time_since_full > 0.5:
            return 3
        position_data = self.target_tracker.get_position_data()
        if not position_data or not all(k in position_data for k in ['distance', 'lateral', 'bearing']):
            return computation_level
        distance_error = abs(position_data['distance'] - self.parameter_manager.desired_distance)
        lateral_error = abs(position_data['lateral'])
        angular_error = abs(position_data['bearing'])
        if self.parameter_manager.use_fast_trigonometry:
            angular_degrees = angular_error * 57.29578
        else:
            angular_degrees = math.degrees(angular_error)
        # Use velocity_control for velocity state
        velocity_stable = all(abs(v) < 0.05 for v in self.velocity_control.last_cmd_vel)
        if (distance_error < self.parameter_manager.distance_threshold * 0.3 and
            lateral_error < self.parameter_manager.lateral_threshold * 0.3 and
            angular_degrees < self.parameter_manager.angular_threshold * 0.3):
            computation_level = min(computation_level, 0)
        elif (distance_error < self.parameter_manager.distance_threshold * 0.7 and
              lateral_error < self.parameter_manager.lateral_threshold * 0.7 and
              angular_degrees < self.parameter_manager.angular_threshold * 0.7):
            computation_level = min(computation_level, 1)
        elif (distance_error < self.parameter_manager.distance_threshold * 1.5 and
              lateral_error < self.parameter_manager.lateral_threshold * 1.5 and
              angular_degrees < self.parameter_manager.angular_threshold * 1.5):
            computation_level = min(computation_level, 2)
        if velocity_stable and computation_level > 0:
            computation_level -= 1
        if (self.distance_error_tracker.get_trend() == "stable" and
            self.lateral_error_tracker.get_trend() == "stable" and
            self.angular_error_tracker.get_trend() == "stable"):
            computation_level = min(computation_level, 1)
        return computation_level
    
    def prepare_shutdown(self):
        """Prepare for node shutdown."""
        if self.state_manager.shutting_down:
            return  # Already shutting down
            
        print("Preparing for shutdown")  # Use print instead of ROS logger
        self.state_manager.shutting_down = True
        
        # Cancel all timers to prevent them from continuing to fire
        for timer in [self.timer, self.diagnostic_timer, self.resource_timer]:
            if hasattr(self, timer.__name__) and timer.__name__ is not None:
                try:
                    timer.cancel()
                except Exception:
                    pass
                    
        if hasattr(self, 'transform_check_timer') and self.transform_check_timer is not None:
            try:
                self.transform_check_timer.cancel()
            except Exception:
                pass
                    
        try:
            # Create a stop command directly
            stop_cmd = Twist()  # All values default to 0.0
            
            # Try to publish but don't rely on logging
            try:
                self.cmd_vel_pub.publish(stop_cmd)
                print("Robot motion stopped during shutdown")
            except Exception:
                print("Failed to publish stop command - context may be invalid")
                
            # Quick sleep to allow message to be sent if context still valid
            time.sleep(0.1)
        except Exception as e:
            print(f"Error stopping robot during shutdown: {str(e)}")


# Main function to start the ROS2 node and handle shutdown
# This is the entry point for the program
# It sets up signal handling for graceful shutdown and starts the node

def main(args=None):
    """Main function to initialize and run the PID Controller node."""
    rclpy.init(args=args)
    node = None
    
    # Flag to track shutdown state
    shutdown_initiated = False
    
    # Set up signal handler for graceful shutdown
    original_sigint_handler = signal.getsignal(signal.SIGINT)
    
    try:
        print("=================================================")
        print("Optimized PID Controller for Basketball Tracking Robot")
        print("=================================================")
        try:
            node = PIDControllerNode()
        except InitializationError as e:
            print(f"\nINITIALIZATION ERROR: {str(e)}")
            print("\nThe system cannot start due to critical initialization errors.")
            sys.exit(1)
            
        # Set up signal handler for graceful shutdown
        def signal_handler(sig, frame):
            nonlocal shutdown_initiated
            
            if shutdown_initiated:
                print("\nForce quitting (received multiple shutdown signals)")
                sys.exit(1)
                
            shutdown_initiated = True
            print("\nShutdown requested by user (Ctrl+C)")
            print("Stopping robot and shutting down...")
            
            # Stop the robot first, before rclpy.shutdown() invalidates the context
            if node is not None:
                # Use direct method to stop robot without logging
                stop_cmd = Twist()  # All values default to 0.0
                try:
                    node.cmd_vel_pub.publish(stop_cmd)
                    # Small delay to allow message to be sent
                    time.sleep(0.1)
                except Exception as e:
                    print(f"Error during emergency stop: {str(e)}")
                node.state_manager.shutting_down = True
                
            # Important: Raise KeyboardInterrupt to break out of rclpy.spin()
            raise KeyboardInterrupt
            
        # Register the signal handler
        signal.signal(signal.SIGINT, signal_handler)
        
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            # This will be raised by our signal handler
            print("Exiting main loop...")
        except Exception as e:
            print(f"\nRUNTIME ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
    finally:
        try:
            # Restore original signal handler before cleanup
            signal.signal(signal.SIGINT, original_sigint_handler)
            
            if node is not None:
                # No need to call prepare_shutdown() since signal handler does it
                node.destroy_node()
                print("Node destroyed successfully")
        except Exception as e:
            print(f"Error during shutdown: {str(e)}")
        
        print("Calling rclpy.shutdown()...")
        rclpy.shutdown()
        print("Shutdown complete")

if __name__ == '__main__':
    main()