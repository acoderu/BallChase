"""
Basketball Tracking Robot - PID Control System Package
=====================================================

WHAT IS THIS PACKAGE? >
---------------------
This package contains the core control systems that make the robot track
and follow a basketball intelligently. It's like the robot's "brain" that 
helps it decide how to move in response to what it sees.

WHAT IS PID CONTROL? <¯
--------------------
PID stands for Proportional-Integral-Derivative control, which is a powerful 
technique used in many real-world systems - from thermostats to cruise control 
to industrial robots!

IMAGINE THIS: =—
------------
Think about how you drive a car toward a parking spot:

1. PROPORTIONAL (P): How far away are you? The further away, the faster you drive.
   "I'm 10 meters away, so I'll press the gas pedal harder."

2. INTEGRAL (I): Have you been off target for a while? Adjust to fix persistent errors.
   "I've been slightly to the left of the spot for a while, I need to steer right a bit more."

3. DERIVATIVE (D): Are you approaching quickly? Slow down to avoid overshooting.
   "I'm getting close fast, better start braking to avoid driving past the spot!"

KEY MODULES:
----------
- pid_computation.py: Core PID control algorithms and strategy selection
- pid_helpers.py: Utility functions and performance optimizations  
- pid_target_filter.py: Filtering algorithms to handle sensor noise
- pid_target_tracking.py: Movement planning and trajectory control

These modules work together to create smooth, intelligent robot movement
that can track and follow a basketball in real-time!
"""

from ball_chase.pid.pid_computation import *
from ball_chase.pid.pid_helpers import *
from ball_chase.pid.pid_target_filter import *
from ball_chase.pid.pid_target_tracking import *