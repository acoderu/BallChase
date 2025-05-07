#!/usr/bin/env python3

"""
Optimized Basketball Chaser - State Management Node
==================================================

ARCHITECTURAL OVERVIEW
---------------------

                 ┌──────────────┐     ┌───────────────┐     ┌─────────────────┐
                 │ Sensor Nodes │     │ Fusion Node   │     │ State Manager   │     ┌───────────────┐
 Physical World  │ ------------ │     │ ------------- │     │ --------------- │     │ PID Controller │
 (Basketball) ──>│ - YOLO       │────>│ - Kalman      │────>│ - State Machine │────>│ - Motion      │──> Robot
                 │ - HSV        │     │   Filter      │     │ - Decision      │     │   Control     │    Movement
                 │ - LiDAR      │     │ - Uncertainty │     │   Logic         │     │               │
                 │ - Depth      │     │   Tracking    │     │ - Behavioral    │     │               │
                 └──────────────┘     └───────────────┘     │   Switching     │     └───────────────┘
                                                            └─────────────────┘

SYSTEM DESIGN RATIONALE
----------------------

The State Management Node serves as a critical intermediary between sensor fusion and motor control. 
This architectural design follows the principle of separation of concerns, creating a more robust and 
maintainable system.

Why State Management is Necessary:
---------------------------------
1. **Decision Making vs. Data Processing**: 
   - The Fusion Node focuses solely on combining sensor data optimally (WHAT is happening)
   - The State Manager interprets this data to decide robot behavior (WHAT to DO about it)
   - This separation allows each node to be specialized and optimized for its specific task

2. **Distinct State Handling**:
   - Different robot states (tracking, searching, stopped) require completely different behaviors
   - Direct Fusion→PID connection would require the fusion node to handle all these state variations
   - State Manager centralizes this decision-making, keeping other nodes simpler

3. **Resilience to Sensor Issues**:
   - Temporary sensor failures require special handling (RECOVERY state)
   - Complete loss of detection requires search patterns (SEARCHING state)
   - These responses are behavioral decisions, not sensor fusion problems

4. **Situational Awareness**:
   - The State Manager maintains context over time (e.g., how long in current state)
   - It can implement hysteresis to prevent oscillating behaviors
   - It handles transitions between fundamentally different robot operational modes

Why PID Controller is Not Directly Connected to Fusion:
-----------------------------------------------------
1. **Command Abstraction**:
   - The PID controller only needs to know target positions during active tracking
   - During searching or stopped states, different command generation is needed
   - State Manager provides appropriate commands based on current state

2. **Safety Constraints**:
   - Direct Fusion→PID connection could cause erratic robot movement with noisy sensor data
   - State Manager implements protective constraints before commanding movement
   - It can halt movement entirely in specific states regardless of fusion output

3. **Operational Mode Switching**:
   - PID control is only one of several possible control modes
   - State Manager can switch between different control strategies:
     * PID tracking for normal operation
     * Pattern-based searching when target is lost
     * Complete stop when appropriate

4. **Context-Based Parameter Tuning**:
   - Different situations require different PID parameters
   - State Manager can adjust control parameters based on detected conditions
   - This adaptive behavior would be difficult in a direct Fusion→PID architecture

This node represents the "brain" of the robot, making high-level decisions while the fusion node
acts as the "perception system" and the PID controller as the "motor system". This mimics the
layered decision-making architecture seen in both biological systems and advanced robotics.

Optimized for Raspberry Pi 5 performance while maintaining all functionality.
"""

# ROS 2 Core imports
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import String, Bool, Float32
import math
import json
import time
from enum import Enum, auto

# ----- OPTIMIZED DATA STRUCTURES -----

class RobotState(str, Enum):
    """Enumeration of robot operational states with string values for better readability
    
    🔍 BEGINNER'S GUIDE: Understanding Robot States
    --------------------------------------------
    
    Think of robot states like the different "modes" your phone can be in:
    - Locked (waiting for input)
    - Active (being used)
    - Power-saving (when not in use)
    - Charging (when plugged in)
    
    Our robot also has different modes or "states" that determine how it behaves.
    Each state has specific rules about:
    - What the robot should do in this state
    - When it should switch to a different state
    - What information it needs to make decisions
    
    ---
    Concept: Finite State Machine (FSM)
    -----------------------------------
    A state machine defines a set of states a system can be in and the rules for transitioning
    between those states. This is foundational in robotics for creating reliable, predictable behavior.
    
    🔹 Real-world example: Traffic Light
    A traffic light is a simple state machine with three states: Red, Yellow, and Green.
    - Each state has a specific meaning (Stop, Prepare to stop, Go)
    - The transitions happen in a specific order (Green → Yellow → Red → Green...)
    - Each state lasts for a predetermined time before transitioning
    - The light can only be in one state at a time
    
    Key aspects of our robot's state machine:
    - Each state has well-defined entry and exit conditions
    - Transitions occur when specific conditions are met
    - Hysteresis prevents rapid state fluctuation (more on this below)
    - Events trigger evaluation of transition conditions
    
    🔹 What is hysteresis?
    Hysteresis means requiring a stronger signal to enter a state than to stay in it. 
    
    For example:
    - To switch from TRACKING to LOST_BALL, we might require no detection for 1.5 seconds
    - But to switch back from LOST_BALL to TRACKING, we might need 6 consecutive detections
    
    This prevents the robot from rapidly switching states when conditions are borderline.
    It's like how your home thermostat doesn't turn on immediately when the temperature 
    drops 0.1 degrees below the setting - it waits until the change is significant.

State Transition Diagram:
┌───────────────────┐        Ball detected        ┌───────────────────┐
│                   │     with confidence >0.7    │                   │
│   INITIALIZING    │─────────────────────────────>     TRACKING      │
│                   │                             │                   │
└─────────┬─────────┘                             └─────────┬─────────┘
          │                                                 │
          │                                                 │
No ball   │                                                 │ Ball lost
after     │                                                 │ temporarily
timeout   │                                                 │
          │                                                 │
          ▼                                                 ▼
┌───────────────────┐      Search unsuccessful    ┌───────────────────┐
│                   │<────────────────────────────│                   │
│     LOST_BALL     │                             │     SEARCHING     │
│                   │                             │                   │
└─────────┬─────────┘                             └─────────┬─────────┘
          │                                                 │
          │                                                 │
Ball      │                                                 │ Ball 
found     └─────────────────┐              ┌───────────────┘ found
                            │              │               
                            ▼              ▼               
                     ┌───────────────────────┐      High uncertainty
                     │                       │───────────────────────┐
                     │       TRACKING        │                       │
                     │                       │<──────────────────────┘
                     └───────────┬───────────┘    Uncertainty reduced
                                 │
                                 │ Ball stationary
                                 │ and close
                                 ▼
                     ┌───────────────────────┐
                     │                       │
                     │       STOPPED         │
                     │                       │
                     └───────────────────────┘"""
    INITIALIZING = "initializing"  # Startup state, waiting for first reliable detection
    TRACKING = "tracking"          # Actively tracking the ball with reliable detections
    LOST_BALL = "lost_ball"        # Ball not found after extensive searching
    STOPPED = "stopped"            # Stationary state when ball is close and stationary
    SEARCHING = "searching"        # Actively searching for a lost ball
    RECOVERY = "recovery"          # Recovery mode during sensor gaps or high uncertainty


class OptimizedBuffer:
    """Memory-efficient fixed-size buffer with pre-allocated array.
    Eliminates memory fragmentation from growing lists.
    
    🔍 BEGINNER'S GUIDE: Why Do We Need Special Buffers?
    -------------------------------------------------
    
    Imagine you're taking notes in a small notebook with limited pages:
    
    Option 1: Regular List (Inefficient)
    - When you fill the notebook, you buy a bigger one
    - You copy all your old notes to the new notebook
    - You throw away the old notebook
    - Each time you need a bigger notebook, you waste time copying
    
    Option 2: Circular Buffer (Efficient)
    - You use one notebook of fixed size
    - When you fill it, you go back to the first page
    - You erase the oldest note and write your new note
    - You never need to buy a new notebook or copy anything
    
    In computer terms:
    - Regular lists grow indefinitely, requiring memory reallocation
    - Each reallocation is slow and causes memory fragmentation
    - Circular buffers have fixed size, so they're faster and more memory-efficient
    - They're perfect for robots where we only care about recent measurements
    
    ---
    Concept: Circular Buffer (Ring Buffer)
    -------------------------------------
    A circular buffer is a fixed-size data structure that stores the most recent N items. When the buffer is full, new items overwrite the oldest ones. This is very efficient for streaming data, like sensor readings, because it avoids memory allocation and keeps only the most relevant data.
    
    Mathematical intuition:
    - Think of the buffer as a circle with N slots. When you reach the end, you wrap around to the beginning.
    - This is like a queue that never grows, so it is very fast and memory-efficient.
    
    Circular Buffer Operation:
    
    Adding Elements (when buffer is already full):
    ────┬───┬───┬───┬───┬───┐
    Idx │ 0 │ 1 │ 2 │ 3 │ 4 │
    ────┼───┼───┼───┼───┼───┤
    Data │ D │ E │ A │ B │ C │
    ────┴───┴───┴───┴───┴───┘
                  ▲
                  │
               next_index
               
    Adding new element 'F':
    ────┬───┬───┬───┬───┬───┐
    Idx │ 0 │ 1 │ 2 │ 3 │ 4 │
    ────┼───┼───┼───┼───┼───┤
    Data │ D │ E │ F │ B │ C │
    ────┴───┴───┴───┴───┴───┘
                      ▲
                      │
                   next_index
    
    Reading Elements (getting all values):
    
    From index 3 to end: [B, C]
    From start to index 3: [D, E, F]
    Final result: [B, C, D, E, F]  (oldest to newest)
    
    The key to this implementation is using modulo arithmetic:
    next_index = (next_index + 1) % max_size
    
    This wrapping mechanism makes the buffer extremely efficient for robotics applications
    where we need to maintain a history of recent data without unbounded memory growth."""
    
    def __init__(self, max_size=10):
        """Initialize with fixed buffer size and pre-allocated array."""
        # Store the maximum capacity of our buffer (how many items it can hold)
        self.max_size = max_size
        
        # Pre-allocate the entire array with None values
        # This is like preparing a row of empty boxes before we put anything in them
        # It's more efficient than growing the array one element at a time
        self.data = [None] * max_size
        
        # Index where we'll put the next value (starts at position 0)
        # Think of this as a pointer showing where to place the next item
        self.next_index = 0
        
        # Current number of valid values in the buffer (starts at 0)
        # This helps us know if the buffer is partially filled or completely full
        self.size = 0
    
    def add(self, value):
        """Add value to buffer with optimized memory management.
        
        Mathematical idea:
        - We use modular arithmetic to wrap the index around when it reaches the end.
        - This is like counting on a clock: after 12 comes 1 again."""
        # Step 1: Store the new value at the current "next_index" position
        # If the buffer is already full, this overwrites the oldest value
        # (this is a key feature of circular buffers - they automatically discard old data)
        self.data[self.next_index] = value
        
        # Step 2: Move the index forward by 1, wrapping around if needed
        # The "% self.max_size" (modulo operation) is the magic that makes this circular
        # Example: if max_size is 5 and next_index is 4, then (4+1)%5 = 0, so we wrap to the beginning
        # It's exactly like how a clock wraps from 12 back to 1
        self.next_index = (self.next_index + 1) % self.max_size
        
        # Step 3: Update the count of values in our buffer
        # We use min() to make sure size never exceeds max_size
        # This only matters while the buffer is filling up; once full, size stays at max_size
        self.size = min(self.size + 1, self.max_size)
    
    def get_all(self):
        """Get all valid values as a list, optimized for performance.
        
        Mathematical idea:
        - If the buffer is not full, just return the first 'size' elements.
        - If the buffer is full, return the elements starting from 'next_index' to the end, then from the beginning to 'next_index'.
        - This is a classic use of modular arithmetic in data structures."""
        # Two different cases to handle when returning values:
        
        # CASE 1: Buffer is not yet full (simpler case)
        # If we haven't filled the buffer yet, the valid values are just in positions 0 through size-1
        # Example: If size=3 in a buffer of max_size=5, we just return the first 3 elements
        if self.size < self.max_size:
            # Slice the array to get just the first 'size' elements
            return self.data[:self.size]
            
        # CASE 2: Buffer is full (more complex case)
        # In a full buffer, the oldest value might not be at position 0
        # Instead, the oldest value is at 'next_index' (since that's where we'll write next)
        # We want the values in chronological order (oldest to newest)
        else:
            # This creates a list with two parts:
            # 1. data[next_index:] - from next_index to the end (the older values)
            # 2. data[:next_index] - from the beginning to next_index (the newer values)
            # The '+' operator joins these two lists together
            # Example: If next_index=3 in a buffer of size 5, we return [data[3], data[4], data[0], data[1], data[2]]
            return self.data[self.next_index:] + self.data[:self.next_index]
    
    def get_latest(self, count=1):
        """Get the most recent n values with optimized array handling.
        
        Mathematical idea:
        - To get the latest N items, we calculate the correct start index using modular arithmetic.
        - This is efficient and avoids copying unnecessary data."""
        if self.size == 0:
            return []
        
        count = min(count, self.size)
        if self.size < self.max_size:
            # Buffer not full yet - just return last n elements
            return self.data[max(0, self.size - count):self.size]
        else:
            # Buffer is full - calculate proper indices
            start_idx = (self.next_index - count) % self.max_size
            if start_idx < self.next_index:
                # No wrap-around needed
                return self.data[start_idx:self.next_index]
            else:
                # Wrap-around: concatenate end and start of buffer
                return self.data[start_idx:] + self.data[:self.next_index]
    
    def clear(self):
        """Clear the buffer without reallocating memory.
        
        Concept:
        - Instead of deleting the data, we just reset the indices. This is much faster and avoids memory fragmentation."""
        # Just reset indices without clearing data array
        self.next_index = 0
        self.size = 0
    
    def __len__(self):
        """Get current buffer size.
        
        Concept:
        - Returns how many items are currently stored in the buffer."""
        return self.size


class EfficientTrendAnalyzer:
    """Analyzes trends in time series data with pre-computed differences
    and optimized memory usage.
    
    🔍 BEGINNER'S GUIDE: Understanding Trends in Data
    ---------------------------------------------
    
    This class helps the robot understand how values are changing over time. 
    
    ## What is a Trend?
    
    A trend is the general direction that something is changing over time:
    - **Rising trend**: Values are going up (like a car speeding up)
    - **Falling trend**: Values are going down (like a car slowing down)
    - **Stable trend**: Values are staying about the same (like cruise control)
    
    ## Real-World Examples
    
    Think about checking your body temperature:
    - Normal: Around 98.6°F (37°C) - stable trend
    - Getting sick: 99°F → 100°F → 101°F - rising trend
    - Recovering: 101°F → 100°F → 99°F - falling trend
    
    Or tracking your bank account balance:
    - Spending more than you earn: Balance decreases each month - falling trend
    - Saving: Balance increases each month - rising trend
    - Breaking even: Balance stays about the same - stable trend
    
    ## Why Do We Need This?
    
    For our robot, tracking trends helps make important decisions:
    
    - If position uncertainty is rising, we need to be more careful
    - If ball speed is decreasing, we might prepare to stop soon
    - If detection confidence is falling, we might be losing track of the ball
    
    ---
    Concept: Trend Analysis in Time Series
    -------------------------------------
    - This class helps us understand if a value (like uncertainty or confidence) is increasing, decreasing, or stable over time.
    - It uses a sliding window (buffer) to keep recent values and calculates the rate of change.
    - This is like looking at a graph and asking: is the line going up, down, or flat?
    
    Mathematical intuition:
    - The rate of change is the difference between values divided by the time between them (slope).
    - Stability is measured by how much the values vary (standard deviation).
    
    Trend Analysis Visualization:
    
    Time Series Data and Trend Detection:
    
          │  *                      Rise Detected (+1)
          │   *                     ↗
    Value │    *  *               *
          │          *    *      * 
          │           *  *  *   *  
          │───────────────────────  Stable (0)
          │                     *  
          │                      * 
          │                       * Fall Detected (-1)
          │                        *↘
          └─────────────────────────
                     Time →
                     
    Algorithm Flow:
    
    ┌─────────────┐
    │ New value   │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐         ┌─────────────┐
    │ Calculate   │         │ Update      │
    │ difference  │────────>│ caches      │
    └──────┬──────┘         └─────────────┘
           │
           ▼
    ┌─────────────┐
    │ Calculate   │
    │ rate        │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Determine   │
    │ trend       │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │ Rising,     │
    │ Falling, or │
    │ Stable?     │
    └─────────────┘
    
    The trend is determined by examining the average rate of change:
    - Rising trend: average rate > threshold
    - Falling trend: average rate < -threshold  
    - Stable trend: |average rate| <= threshold"""
    
    def __init__(self, window_size=10):
        """Initialize with window size and pre-computed caches."""
        self.values = OptimizedBuffer(window_size)
        self.timestamps = OptimizedBuffer(window_size)
        # Cache for trend calculation to avoid recomputation
        self.diff_cache = OptimizedBuffer(window_size - 1)
        self.rate_cache = OptimizedBuffer(window_size - 1)
        self.cache_valid = False
        self.stability_score_cache = None
    
    def add(self, value, timestamp=None):
        """Add a value with optimized caching.
        
        Mathematical idea:
        - When a new value comes in, we calculate the difference from the previous value and the rate of change.
        - This is like calculating the slope between two points on a graph.
        - We store these differences and rates for fast trend analysis."""
        # If no timestamp is provided, use the current time
        # Timestamps are important to calculate how fast values are changing
        if timestamp is None:
            timestamp = time.time()  # Current time in seconds since epoch
        
        # We need at least one previous value to calculate a change/trend
        if len(self.values) > 0:
            # Get the most recent value and its timestamp
            prev_value = self.values.get_latest(1)[0]
            prev_time = self.timestamps.get_latest(1)[0]
            
            # STEP 1: Calculate how much the value changed (the difference)
            # Example: If previous value was 10 and new value is 12, diff = +2
            value_diff = value - prev_value
            
            # STEP 2: Calculate how much time passed between measurements
            # Example: If previous time was 100.0 and current is 101.5, diff = 1.5 seconds
            time_diff = timestamp - prev_time
            
            # STEP 3: Store the value difference in our cache for later analysis
            self.diff_cache.add(value_diff)
            
            # STEP 4: Calculate and store the rate of change (how fast it's changing)
            # Rate = Value change ÷ Time change (just like speed = distance ÷ time)
            # Example: If value changed by +2 over 1.5 seconds, rate = +1.33 units per second
            if time_diff > 0:  # Avoid division by zero
                self.rate_cache.add(value_diff / time_diff)
            else:
                # If timestamps are identical, we can't calculate a rate, so use 0
                self.rate_cache.add(0.0)
        
        # Add the new value and timestamp to our history buffers
        self.values.add(value)
        self.timestamps.add(timestamp)
        
        # The stability score is now outdated because we have new data
        # We'll recalculate it only when needed (this is an optimization technique called "lazy evaluation")
        self.stability_score_cache = None
    
    def get_trend(self, num_samples=None):
        """Calculate trend direction and rate with optimized cached calculations.
        
        Returns:
            tuple: (direction, rate) where direction is 1 (rising), -1 (falling), or 0 (stable)
                  and rate is the average change per second
        
        Mathematical idea:
        - The trend is the average rate of change over the window.
        - If the average rate is close to zero, the trend is stable.
        - This is like fitting a straight line to the recent data and looking at its slope."""
        # We need at least 2 values to detect a trend (can't have a trend with just 1 point)
        if len(self.values) < 2:
            # Return "no trend" (0) and zero rate if we don't have enough data
            return 0, 0.0
        
        # Determine how many recent samples to analyze
        # If not specified, use all available samples
        if num_samples is None or num_samples > len(self.values):
            num_samples = len(self.values)
        
        # Get the rate of change values from our cache
        # We need (num_samples-1) rates because each rate is calculated between 2 consecutive points
        # e.g., 5 points produce 4 rates of change
        rates = self.rate_cache.get_latest(num_samples - 1)
        
        # If we couldn't get any rates (shouldn't happen, but just in case)
        if not rates:
            return 0, 0.0
        
        # STEP 1: Calculate the average rate of change
        # This is just like taking the average speed over multiple segments of a journey
        # Example: If rates were [2.0, 3.0, -1.0, 4.0], the average is 2.0
        avg_rate = sum(rates) / len(rates)
        
        # STEP 2: Determine trend direction using a threshold
        # We use a small threshold (0.001) to decide if it's stable
        # This prevents tiny fluctuations from being classified as trends
        if abs(avg_rate) < 0.001:  # Threshold for stability
            return 0, 0.0
        
        direction = 1 if avg_rate > 0 else -1
        return direction, avg_rate
    
    def is_stable(self, threshold=0.05):
        """Check if values are stable with early exit optimization.
        
        Mathematical idea:
        - Stability means the values don't change much relative to their size.
        - We use the min and max to quickly check if the range is within a threshold.
        - This is a fast way to check for stability without calculating variance."""
        if len(self.values) < 2:
            return True
        
        values = self.values.get_all()
        if not values:
            return True
        
        # Calculate min-max range with early exit optimization
        min_val = max_val = values[0]  # Start with first value
        for val in values[1:]:
            if val < min_val:
                min_val = val
            elif val > max_val:
                max_val = val
                
            # Early exit if range already exceeds threshold
            reference = max(abs(min_val), abs(max_val), 0.01)
            if (max_val - min_val) / reference >= threshold:
                return False
        
        # Final check
        reference = max(abs(min_val), abs(max_val), 0.01)
        return (max_val - min_val) / reference < threshold
    
    def get_stability_score(self):
        """Get stability score with caching for performance.
        
        Mathematical idea:
        - The stability score is based on the standard deviation (how much values vary) divided by the mean (average value).
        - A high score means the values are very stable (low variation).
        - This is a normalized measure, so it works for different scales."""
        # Return cached value if available
        if self.stability_score_cache is not None:
            return self.stability_score_cache
            
        if len(self.values) < 2:
            self.stability_score_cache = 1.0
            return 1.0
        
        values = self.values.get_all()
        if not values:
            self.stability_score_cache = 1.0
            return 1.0
        
        # Calculate standard deviation with optimized algorithm
        n = len(values)
        if n <= 1:
            self.stability_score_cache = 1.0
            return 1.0
            
        # Use single-pass algorithm for better performance
        mean = sum(values) / n
        # Optimized variance calculation
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = math.sqrt(variance)
        
        # Calculate normalized stability score
        reference = max(abs(mean), 0.01)  # Avoid division by zero
        normalized_std = std_dev / reference
        
        # Convert to stability score (higher is more stable)
        stability = 1.0 / (1.0 + 10.0 * normalized_std)
        stability = max(0.0, min(1.0, stability))
        
        # Cache the result
        self.stability_score_cache = stability
        return stability


class OptimizedSystemHealthMonitor:
    """Monitors system health with optimized memory usage and computation.
    
    ---
    Concept: System Health Monitoring
    --------------------------------
    - This class keeps track of the health of different parts of the robot (tracking, fusion, sensors, etc.).
    - It uses trend analysis to detect problems early (like rising uncertainty or sensor gaps).
    - It combines different metrics into a single confidence score.
    
    Mathematical intuition:
    - Confidence is calculated as a weighted product of different factors (tracking, uncertainty, sensors).
    - Penalties are applied for warnings, reducing the overall confidence.

---
Concept: Quality and Reliability Metrics
---------------------------------------
Tracking reliability in robotics requires:
1. Quantitative measures of sensor/detection quality
2. Historical tracking to identify trends
3. Confidence intervals for decision making
4. Handling of intermittent failures

Mathematical approach:
- Moving averages for stability measurement
- Threshold-based qualification with hysteresis
- Time-weighted reliability scoring"""
    
    def __init__(self):
        """Initialize health monitor with pre-allocated data structures."""
        # Use bit flags for boolean states to reduce memory usage
        self.status_flags = 0
        
        # Components to monitor with simplified data structure
        self.components = {
            'tracking': [False, 0.0, 0.0],  # [status, confidence, last_update]
            'fusion': [False, float('inf'), 0.0],  # [status, uncertainty, last_update]
            'motion': ["unknown", 0.0, 0.0],  # [state, confidence, last_update]
            'sensors': [0, False, 0.0]  # [active_count, gap_detected, last_update]
        }
        
        # Trend analysis
        self.trends = {
            'uncertainty': EfficientTrendAnalyzer(20),
            'tracking_confidence': EfficientTrendAnalyzer(10),
            'sensor_count': EfficientTrendAnalyzer(5)
        }
        
        # Warning flags
        self.warnings = []
        self.warning_history = OptimizedBuffer(10)
        
        # System metrics
        self.system_confidence = 1.0
        self.message_counters = {}
        
        # Initialize logging helper with reduced memory overhead
        self._last_throttled_logs = {}
    # ===== State Transition Logic =====

    
    def update_tracking(self, reliable, confidence):
        """Update tracking status with minimal memory allocation.
        
        Concept:
        - Updates the tracking status and confidence, and records the time.
        - Adds the confidence value to the trend analyzer for later analysis."""
        current_time = time.time()
        
        self.components['tracking'][0] = reliable
        self.components['tracking'][1] = confidence
        self.components['tracking'][2] = current_time
        
        # Update trend
        self.trends['tracking_confidence'].add(confidence, current_time)
    
    def update_fusion(self, uncertainty):
        """Update fusion status with minimal computation.
        
        Concept:
        - Updates the uncertainty value and records the time.
        - Adds the uncertainty to the trend analyzer to detect if uncertainty is rising or falling.
        - Sets the fusion status to True if uncertainty is below a threshold."""
        current_time = time.time()
        
        self.components['fusion'][1] = uncertainty
        self.components['fusion'][2] = current_time
        
        # Auto-calculate status based on uncertainty thresholds
        self.components['fusion'][0] = uncertainty < 0.4
        
        # Update trend
        self.trends['uncertainty'].add(uncertainty, current_time)
    
    def update_motion(self, state, confidence=0.7):
        """Update motion state with minimal memory allocation.
        
        Concept:
        - Records the current motion state (e.g., stationary, moving) and confidence.
        - This helps the system adapt its behavior based on how the ball is moving."""
        self.components['motion'][0] = state
        self.components['motion'][1] = confidence
        self.components['motion'][2] = time.time()
    
    def update_sensors(self, active_count, gap_detected=False):
        """Update sensor status with minimal memory allocation.
        
        Concept:
        - Records the number of active sensors and whether a gap is detected.
        - Adds the sensor count to the trend analyzer to monitor sensor health over time."""
        self.components['sensors'][0] = active_count
        self.components['sensors'][1] = gap_detected
        self.components['sensors'][2] = time.time()
        
        # Update trend
        self.trends['sensor_count'].add(active_count)
    
    def evaluate_health(self):
        """Evaluate overall system health with optimized computation.
        
        Concept:
        - Checks for stale data, degraded tracking, high uncertainty, sensor gaps, and low sensor count.
        - Uses trend analysis to detect if uncertainty is rising (a sign of trouble).
        - Returns a list of new warnings for the system."""
        current_time = time.time()
        self.warnings = []
        
        # Check for stale data
        for component, data in self.components.items():
            age = current_time - data[2]  # Last update is at index 2
            if age > 2.0:
                self.warnings.append(f"{component}_stale_data")
        
        # Check for degraded tracking using direct indexing for performance
        if self.components['tracking'][1] < 0.4 and not self.components['tracking'][0]:
            self.warnings.append('tracking_degraded')
        
        # Check for high uncertainty
        if self.components['fusion'][1] > 0.5:
            # Check if uncertainty is rising
            direction, rate = self.trends['uncertainty'].get_trend(5)
            if direction > 0 and rate > 0.05:
                self.warnings.append('uncertainty_rising')
            else:
                self.warnings.append('high_uncertainty')
        
        # Check for sensor gaps during tracking
        if self.components['sensors'][1] and self.components['tracking'][0]:
            self.warnings.append('sensor_gap_during_tracking')
        
        # Check for low sensor count
        if self.components['sensors'][0] < 1:
            self.warnings.append('no_active_sensors')
        
        # Efficiently track new warnings
        new_warnings = []
        for warning in self.warnings:
            # Using a more efficient set operation would be better,
            # but we're maintaining the original logic
            if warning not in [w for w in self.warning_history.get_all() if w is not None]:
                new_warnings.append(warning)
                self.warning_history.add(warning)
        
        return new_warnings
    
    def calculate_system_confidence(self):
        """Calculate overall system confidence with optimized operations.
        
        Mathematical idea:
        - Combines tracking confidence, fusion uncertainty, and sensor count into a single score.
        - Uses weighted multiplication and applies penalties for warnings.
        - This is like a health score for the robot, where each part contributes to the total.
        
        Returns:
            The processed result after applying the calculation logic"""
        # Start with base confidence
        confidence = 1.0
        
        # Factor in tracking confidence with optimized weights
        tracking_weight = 0.4
        tracking_confidence = self.components['tracking'][1]
        confidence *= (tracking_weight * tracking_confidence + (1 - tracking_weight))
        
        # Factor in fusion uncertainty (invert to get confidence)
        uncertainty = self.components['fusion'][1]
        # Fast approximation using fewer operations
        uncertainty_factor = 1.0 / (1.0 + uncertainty * 2.0)
        uncertainty_weight = 0.3
        confidence *= (uncertainty_weight * uncertainty_factor + (1 - uncertainty_weight))
        
        # Factor in sensor count
        sensor_count = self.components['sensors'][0]
        sensor_factor = min(1.0, sensor_count / 2.0)  # 2+ sensors = full confidence
        sensor_weight = 0.2
        confidence *= (sensor_weight * sensor_factor + (1 - sensor_weight))
        
        # Apply penalties for warnings
        warning_penalty = 0.1 * len(self.warnings)
        confidence = max(0.1, confidence - warning_penalty)
        
        # Store and return
        self.system_confidence = confidence
        return confidence
    
    def get_diagnostic_data(self):
        """Get diagnostic data with minimal object creation.
        
        Concept:
        - Collects the current health, warnings, and trends into a dictionary for diagnostics.
        - This is used for monitoring and debugging the robot."""
        # Calculate confidence
        system_confidence = self.calculate_system_confidence()
        
        # Build diagnostic data
        data = {
            'system_confidence': round(system_confidence, 3),
            'components': {},
            'warnings': list(self.warnings),
            'trends': {}
        }
        
        # Add component data
        for component, info in self.components.items():
            # Clean up data - remove timestamps
            if component == 'tracking':
                clean_data = {'status': info[0], 'confidence': info[1]}
            elif component == 'fusion':
                clean_data = {'status': info[0], 'uncertainty': info[1]}
            elif component == 'motion':
                clean_data = {'state': info[0], 'confidence': info[1]}
            elif component == 'sensors':
                clean_data = {'active_count': info[0], 'gap_detected': info[1]}
            
            data['components'][component] = clean_data
        
        # Add trends
        for trend_name, analyzer in self.trends.items():
            if len(analyzer.values) >= 2:
                direction, rate = analyzer.get_trend()
                stability = analyzer.get_stability_score()
                
                data['trends'][trend_name] = {
                    'direction': direction,
                    'rate': round(rate, 5),
                    'stability': round(stability, 3)
                }
        
        return data
    
    def increment_message_counter(self, topic):
        """Track message counts with minimal overhead.
        
        Concept:
        - Keeps track of how many messages have been received for each topic.
        - Useful for diagnostics and debugging."""
        if topic not in self.message_counters:
            self.message_counters[topic] = 0
        self.message_counters[topic] += 1
        return self.message_counters[topic]
    
    def throttled_log(self, logger, message, key, min_interval=1.0, level="info"):
        """Log with throttling and cached message evaluation.
        
        Concept:
        - Only logs a message if enough time has passed since the last log with the same key.
        - Prevents flooding the log with repeated messages."""
        current_time = time.time()
        
        # Check if enough time has passed since last log
        if key in self._last_throttled_logs:
            elapsed = current_time - self._last_throttled_logs[key]
            if elapsed < min_interval:
                return
                
        # Update last log time
        self._last_throttled_logs[key] = current_time
        
        # Log with appropriate level
        if level == "error":
            logger.error(message)
        elif level == "warn":
            logger.warn(message)
        else:
            logger.info(message)


# ----- OPTIMIZED VECTOR OPERATIONS -----

def vec3_distance(pos1, pos2=None):
        """Calculate Euclidean distance with optimized operations."""
    if pos2 is None:
        # Distance from origin (0,0,0)
        return math.sqrt(pos1[0]**2 + pos1[1]**2 + pos1[2]**2)
    else:
        # Distance between two points
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        return math.sqrt(dx**2 + dy**2 + dz**2)

def vec2_distance(pos):
        """Calculate XY-plane distance from origin with optimized operations."""
    return math.sqrt(pos[0]**2 + pos[1]**2)


# ----- OPTIMIZED JSON ENCODER -----

class FastJSONEncoder(json.JSONEncoder):
    """JSON Encoder optimized for common data types."""
    def default(self, obj):
        # Direct handling of common types without isinstance overhead
        obj_type = type(obj)
        
        # Handle float types first (most common)
        if obj_type is float:
            return obj
        
        # Handle integer types
        if obj_type is int:
            return obj
        
        # Handle list types
        if obj_type is list:
            return obj
        
        # Handle bool types
        if obj_type is bool:
            return obj
            
        # Fall back to slower isinstance checks for other types
        try:
            if hasattr(obj, 'tolist'):  # For array-like objects
                return obj.tolist()
            return super(FastJSONEncoder, self).default(obj)
        except TypeError:
            return str(obj)  # Last resort - stringify


# ----- MAIN NODE IMPLEMENTATION -----

class OptimizedBallChaseStateManager(Node):
    """Optimized state management node for the basketball chasing robot.
    
    🔍 BEGINNER'S GUIDE: What is a State Manager?
    ------------------------------------------
    
    The State Manager is like the "brain" of our robot. It makes high-level decisions about what the
    robot should do at any given moment based on what's happening around it.
    
    ## What Does the State Manager Do?
    
    1. **Receives Information**:
       - Gets position data about the basketball (from sensor fusion)
       - Checks if the ball is moving or stationary
       - Monitors uncertainty (how confident we are in our measurements)
       - Tracks how long we've been in the current state
    
    2. **Makes Decisions**:
       - Should we follow the ball? (TRACKING state)
       - Should we stop because we've reached the ball? (STOPPED state)
       - Should we search for the ball if we lost it? (SEARCHING state)
       - Should we wait for better data if our measurements are unreliable? (RECOVERY state)
    
    3. **Sends Commands**:
       - Tells the robot's motor controller where to go (or to stay still)
       - Changes the controller's parameters based on the situation
       - Provides feedback about the robot's current state
    
    ## Real-World Analogy
    
    Think of the State Manager like a basketball coach:
    - The coach observes the game (receives sensor data)
    - Makes decisions based on the situation (transitions between states)
    - Calls plays and gives instructions to players (sends commands to controllers)
    - Adapts strategy based on what's working or not (learns and adjusts parameters)
    
    ---
    Concept: State Machine for Robot Behavior
    ----------------------------------------
    - This class manages the robot's high-level behavior using a state machine.
    - The robot transitions between states like INITIALIZING, TRACKING, LOST_BALL, STOPPED, SEARCHING, and RECOVERY.
    - Each state has its own logic for when to transition to another state, based on sensor data, tracking reliability, and uncertainty.
    
    Mathematical intuition:
    - A state machine is a mathematical model where the system is always in one of a finite number of states.
    - Transitions between states are triggered by events or conditions (like losing the ball or detecting high uncertainty).
    - This is a powerful way to organize complex robot behavior in a way that is easy to reason about and debug."""
    
    def __init__(self):
        """Initialize the state manager node with reduced overhead."""
        super().__init__('ball_chase_state_manager')
        
        # Create callback groups for concurrency control
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.subscription_cb_group = ReentrantCallbackGroup()
        
        # Start time for elapsed time tracking
        self.start_time = time.time()
        
        # Declare parameters with optimized grouping
        self._declare_parameters()
        
        # Load parameters
        self._load_parameters()
        
        # Initialize state variables with default values
        self._init_state_variables()
        
        # Initialize health monitoring
        self.health_monitor = OptimizedSystemHealthMonitor()
        
        # Set up subscriptions with staged startup
        self._setup_subscriptions()
        
        # Set up publishers
        self._setup_publishers()
        
        # Set up timers with optimized frequencies
        self._setup_timers()
        
        self.get_logger().info("Optimized Basketball Chaser State Manager initialized in INITIALIZING state")
        self.publish_state()
    
    def _declare_parameters(self):
        """Declare all parameters with optimized grouping."""
        # Define parameter groups for better performance
        timing_params = [
            ('lost_ball_timeout', 1.5),
            ('max_search_time', 30.0),
            ('stationary_time_threshold', 1.5),
            ('max_lost_ball_time', 5.0),
            ('max_recovery_time', 3.0),
        ]
        
        search_params = [
            ('search_rotation_speed', 0.5),
            ('max_rotation_time', 15.0),
        ]
        
        detection_params = [
            ('min_tracking_detections', 3),
            ('min_retracking_detections', 6),
            ('proximity_threshold', 0.5),
            ('stationary_threshold', 0.05),
        ]
        
        uncertainty_params = [
            ('position_uncertainty_threshold', 0.5),
            ('uncertainty_recovery_threshold', 0.35),
        ]
        
        hysteresis_params = [
            ('tracking_hysteresis_time', 1.0),
            ('lost_ball_hysteresis_time', 0.5),
            ('recovery_hysteresis_time', 0.3),
        ]
        
        adaptive_params = [
            ('adaptive_parameters_enabled', True),
            ('adaptive_factor_stationary', 1.5),
            ('adaptive_factor_moving', 0.8),
        ]
        
        gap_params = [
            ('gap_tolerance_time', 1.5),
            ('gap_stationary_multiplier', 2.0),
            ('gap_enabled', True),
        ]
        
        system_params = [
            ('health_confidence_threshold', 0.5),
            ('health_check_interval', 1.0),
            ('diagnostic_publish_rate', 1.0),
            ('full_diagnostic_rate', 5.0),
            ('resource_monitoring_enabled', True),
        ]
        
        # Combine all parameter groups
        all_params = (timing_params + search_params + detection_params + 
                     uncertainty_params + hysteresis_params + adaptive_params + 
                     gap_params + system_params)
        
        # Declare all parameters in a single batch for better performance
        self.declare_parameters(namespace='', parameters=all_params)
    
    def _load_parameters(self):
        """Load parameters with optimized batching."""
        # Timing thresholds
        self.lost_ball_timeout = self.get_parameter('lost_ball_timeout').value
        self.max_search_time = self.get_parameter('max_search_time').value
        self.stationary_time_threshold = self.get_parameter('stationary_time_threshold').value
        self.max_lost_ball_time = self.get_parameter('max_lost_ball_time').value
        self.max_recovery_time = self.get_parameter('max_recovery_time').value
        
        # Search pattern parameters
        self.search_rotation_speed = self.get_parameter('search_rotation_speed').value
        self.max_rotation_time = self.get_parameter('max_rotation_time').value
        
        # Detection thresholds
        self.min_tracking_detections = self.get_parameter('min_tracking_detections').value
        self.min_retracking_detections = self.get_parameter('min_retracking_detections').value
        self.proximity_threshold = self.get_parameter('proximity_threshold').value
        self.stationary_threshold = self.get_parameter('stationary_threshold').value
        
        # Uncertainty thresholds
        self.position_uncertainty_threshold = self.get_parameter('position_uncertainty_threshold').value
        self.uncertainty_recovery_threshold = self.get_parameter('uncertainty_recovery_threshold').value
        
        # Hysteresis parameters
        self.tracking_hysteresis_time = self.get_parameter('tracking_hysteresis_time').value
        self.lost_ball_hysteresis_time = self.get_parameter('lost_ball_hysteresis_time').value
        self.recovery_hysteresis_time = self.get_parameter('recovery_hysteresis_time').value
        
        # Adaptive parameters
        self.adaptive_parameters_enabled = self.get_parameter('adaptive_parameters_enabled').value
        self.adaptive_factor_stationary = self.get_parameter('adaptive_factor_stationary').value
        self.adaptive_factor_moving = self.get_parameter('adaptive_factor_moving').value
        
        # Gap tolerance parameters
        self.gap_tolerance_time = self.get_parameter('gap_tolerance_time').value
        self.gap_stationary_multiplier = self.get_parameter('gap_stationary_multiplier').value
        self.gap_enabled = self.get_parameter('gap_enabled').value
        
        # System health parameters
        self.health_confidence_threshold = self.get_parameter('health_confidence_threshold').value
        self.health_check_interval = self.get_parameter('health_check_interval').value
        
        # Resource management parameters - reduced frequencies for Pi optimization
        self.diagnostic_publish_rate = self.get_parameter('diagnostic_publish_rate').value
        self.full_diagnostic_rate = self.get_parameter('full_diagnostic_rate').value
        self.resource_monitoring_enabled = self.get_parameter('resource_monitoring_enabled').value
        
        # Store base parameters for adaptive scaling
        self.base_lost_ball_timeout = self.lost_ball_timeout
        self.base_stationary_threshold = self.stationary_threshold
        self.base_min_tracking_detections = self.min_tracking_detections
        self.base_min_retracking_detections = self.min_retracking_detections
    
    def _init_state_variables(self):
        """Initialize state tracking variables with optimized memory allocation."""
        # Current state
        self.current_state = RobotState.INITIALIZING
        self.state_start_time = time.time()
        self.previous_state = None
        
        # Ball tracking
        self.last_position = None
        self.last_detection_time = None
        self.consecutive_detections = 0
        self.tracking_reliable = False
        self.position_uncertainty = float('inf')
        self.tracking_confidence = 0.5  # Default value
        
        # Motion state tracking with defaults
        self.motion_state = "unknown"
        self.last_motion_state = None
        self.in_motion_transition = False
        
        # Position history for stationary detection - use optimized buffer
        self.position_history = OptimizedBuffer(10)
        
        # Ball proximity and movement detection
        self.ball_distance = float('inf')
        self.is_ball_close = False
        self.is_ball_stationary = False
        self.stationary_start_time = None
        
        # Gap tracking
        self.in_sensor_gap = False
        self.gap_start_time = None
        self.gap_duration = 0.0
        
        # Search variables
        self.search_direction = 1  # 1 for counter-clockwise, -1 for clockwise
        self.search_rotation_start_time = None
        self.search_angle_accumulated = 0.0  # Track rotation during search
        
        # Recovery variables
        self.recovery_reason = None
        self.recovery_attempt_count = 0
        
        # State protection variables - optimize with fixed-size buffer
        self.state_transition_history = OptimizedBuffer(10)
        self.transition_times = {}
        self.last_state_change_time = time.time()
        self.hysteresis_counts = {}  # Count of blocked transitions
        
        # Uncertainty tracking
        self.uncertainty_history = EfficientTrendAnalyzer(20)  # Track uncertainty trend
        self.uncertainty_history.add(self.position_uncertainty)
        
        # Diagnostic data
        self.diagnostic_data = {}
        self.last_full_diagnostic_time = 0.0
        
        # Message counters
        self.message_counts = {}
        
        # Rate limiting variables
        self._last_log_time = {}
        self._param_update_timer = 0
    
    def _setup_subscriptions(self):
        """Set up subscriptions to fusion node topics."""
        # Ball position
        self.position_sub = self.create_subscription(
            PointStamped,
            '/basketball/fused/position',
            self.position_callback,
            10,
            callback_group=self.subscription_cb_group
        )
        
        # Tracking status
        self.tracking_status_sub = self.create_subscription(
            Bool,
            '/basketball/fused/tracking_status',
            self.tracking_status_callback,
            10,
            callback_group=self.subscription_cb_group
        )
        
        # Position uncertainty
        self.uncertainty_sub = self.create_subscription(
            Float32,
            '/basketball/fused/position_uncertainty',
            self.uncertainty_callback,
            10,
            callback_group=self.subscription_cb_group
        )
        
        # Motion State Integration
        self.motion_state_sub = self.create_subscription(
            String,
            '/basketball/fused/motion_state',
            self.motion_state_callback,
            10,
            callback_group=self.subscription_cb_group
        )
        
        # Confidence-Based Decision Making
        self.confidence_sub = self.create_subscription(
            Float32,
            '/basketball/fused/tracking_confidence',
            self.tracking_confidence_callback,
            10,
            callback_group=self.subscription_cb_group
        )
        
        # Gap Tolerance Enhancement
        self.gap_detection_sub = self.create_subscription(
            Bool,
            '/basketball/fused/sensor_gap',
            self.sensor_gap_callback,
            10,
            callback_group=self.subscription_cb_group
        )
        
        # Fusion diagnostics 
        self.fusion_diagnostics_sub = self.create_subscription(
            String,
            '/basketball/fusion/diagnostics',
            self.fusion_diagnostics_callback,
            3,
            callback_group=self.subscription_cb_group
        )
    
    def _setup_publishers(self):
        """Set up publishers for commands and state information."""
        # Robot motion commands
        self.cmd_vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )
        
        # Current robot state
        self.state_publisher = self.create_publisher(
            String,
            '/robot/state',
            10
        )
        
        # System health status - reduced QoS for Pi optimization
        self.health_publisher = self.create_publisher(
            String,
            '/robot/health',
            5
        )
        
        # Enhanced diagnostics - reduced QoS for Pi optimization
        self.diagnostics_publisher = self.create_publisher(
            String,
            '/robot/diagnostics',
            3
        )
    
    def _setup_timers(self):
        """Set up all timer callbacks with optimized frequencies for Pi performance."""
        # Critical state management timer (5Hz instead of 10Hz)
        # This reduces CPU usage while maintaining responsiveness
        self.state_timer = self.create_timer(
            0.2,  # 5Hz instead of 10Hz 
            self.state_manager_callback,
            callback_group=self.timer_cb_group
        )
        
        # Health check timer (reduced frequency)
        self.health_timer = self.create_timer(
            max(self.health_check_interval, 1.0),  # Ensure minimum 1s interval
            self.health_check_callback,
            callback_group=self.timer_cb_group
        )
        
        # Periodic state republishing (0.25Hz instead of 0.5Hz)
        self.state_republish_timer = self.create_timer(
            4.0,  # 4s instead of 2s
            self.publish_state,
            callback_group=self.timer_cb_group
        )
        
        # Diagnostic publication (0.5Hz instead of 1Hz)
        self.diagnostic_timer = self.create_timer(
            2.0 / self.diagnostic_publish_rate,  # Halve the frequency
            self.publish_diagnostics,
            callback_group=self.timer_cb_group
        )
        
        self.get_logger().info("Timers set up with optimized frequencies for Raspberry Pi")
    
    def tracking_status_callback(self, msg):
    """Process tracking reliability flag with reduced logging.
        
        This method is called automatically by the ROS 2 framework when new messages arrive.
        It processes the incoming data and updates the node's internal state accordingly."""
        # Update health monitor
        self.health_monitor.increment_message_counter('tracking_status')
        
        # Store tracking status
        self.tracking_reliable = msg.data
        
        # Update health monitoring
        if hasattr(self, 'tracking_confidence'):
            self.health_monitor.update_tracking(self.tracking_reliable, self.tracking_confidence)
        else:
            self.health_monitor.update_tracking(self.tracking_reliable, 0.5)
        
        # Log with reduced frequency - 20x reduction compared to original
        msg_count = self.health_monitor.message_counters.get('tracking_status', 0)
        if msg_count % 20 == 0:  # Only log every 20th message
            self.get_logger().info(f"Fusion tracking status: reliable={self.tracking_reliable}")
    
    def uncertainty_callback(self, msg):
    """Process and analyze position uncertainty data to inform adaptive tracking behavior.
        
        This method receives and processes uncertainty values from the fusion node, which 
        represent the estimated error margins in the ball's position. These uncertainty 
        metrics enable the robot to make intelligent decisions about its tracking behavior.
        
        Data Transformations:
        1. Value Processing:
           Input: ROS Float32 message → Output: Calibrated uncertainty value
           - Extracts raw uncertainty value (in meters)
           - Applies bounds checking and validation
           - Updates internal uncertainty tracking variable
           
        2. Trend Analysis:
           Input: Uncertainty value → Output: Trend classification
           - Adds value to time-series analyzer (EfficientTrendAnalyzer)
           - Determines if uncertainty is rising, falling, or stable
           - Calculates rate of change in uncertainty
           
        3. System Health Integration:
           Input: Uncertainty value → Output: Health status update
           - Updates system health monitor with current uncertainty
           - Contributes to overall system confidence calculation
           
        4. Recovery Evaluation:
           Input: Uncertainty trend → Output: Recovery need assessment
           - Uses trend information to evaluate if recovery is needed
           - Rising uncertainty can trigger RECOVERY state transition
           
        Mathematical Basis - Uncertainty Propagation:
        In sensor fusion systems, uncertainty is propagated through Kalman filter covariance
        matrices. The scalar uncertainty values received here typically represent:
        
        - For position: spatial standard deviation in meters
        - For velocity: standard deviation in meters per second
        
        Rising uncertainty indicates the system is becoming less confident in its tracking,
        which can happen due to:
        - Sensor disagreement
        - Lack of recent measurements
        - Unpredictable ball movement
        - Sensor obstruction or failure"""
        # Update health monitor
        self.health_monitor.increment_message_counter('uncertainty')
        
        # Store uncertainty value
        old_uncertainty = self.position_uncertainty
        self.position_uncertainty = msg.data
        
        # Check for significant change
        significant_change = abs(self.position_uncertainty - old_uncertainty) > 0.05
        
        # Update trend analysis
        self.uncertainty_history.add(self.position_uncertainty)
        
        # Update health monitoring
        self.health_monitor.update_fusion(self.position_uncertainty)
        
        # Evaluate if we should enter recovery mode based on uncertainty
        self.evaluate_uncertainty_recovery()
        
        # Log with reduced frequency
        msg_count = self.health_monitor.message_counters.get('uncertainty', 0)
        if msg_count % 30 == 0 or significant_change:  # Reduced logging
            direction, rate = self.uncertainty_history.get_trend(5)
            trend_str = "stable"
            if direction > 0:
                trend_str = f"rising ({rate:.3f}/s)"
            elif direction < 0:
                trend_str = f"falling ({-rate:.3f}/s)"
                
            self.get_logger().info(
                f"Position uncertainty: {self.position_uncertainty:.3f}m, trend: {trend_str}"
            )
    
    def position_callback(self, msg):
    """Process incoming basketball position information and calculate distance metrics.
        
        Concept:
        - Converts the incoming ROS PointStamped message to a tuple for fast processing.
        
        Data Transformations:
        1. Coordinate Space Transformation: 
           Input: 3D position (x,y,z) → Output: Distance metrics (total distance, ground distance)
           - Converts Cartesian coordinates to scalar distance values using Euclidean distance formula:
             distance = √(x² + y² + z²)
           - Ground distance isolates horizontal components: ground_distance = √(x² + y²)
           
        2. Temporal Analysis: 
           Input: Current timestamp → Output: Detection intervals and timeout flags
           - Calculates time deltas between consecutive detections
           - Sets detection time markers used by state transition logic
           
        3. Motion Pattern Recognition: 
           Input: Sequence of positions → Output: Stationary determination
           - Updates circular buffer with position history
           - Computes movement magnitude between consecutive positions
           - Determines if ball is stationary using adaptive thresholds based on motion state
           
        4. Reliability Assessment:
           Input: Position and time data → Output: Tracking confidence flags
           - Increments consecutive detection counter (reset by gaps or timeouts)
           - Sets tracking_reliable flag when counter threshold is reached
           
        Key Performance Optimizations:
        - Uses pre-allocated circular buffer for position history
        - Implements early exit for invalid positions (0,0,0)
        - Reuses vector calculations across multiple operations
        
        Position Processing Pipeline:
        
        ┌───────────────┐
        │ PointStamped  │
        │ Message       │
        └───────┬───────┘
                │
                ▼
        ┌───────────────┐      ┌───────────────┐
        │ Extract       │      │ Validate      │
        │ Position Data │─────>│ Position      │
        └───────┬───────┘      └───────┬───────┘
                │                      │
                │       ┌──────────────┘
                │       │
                ▼       ▼
        ┌───────────────────────┐
        │ Calculate Distances   │
        │ - Total distance      │
        │ - Ground distance     │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐      ┌───────────────────────┐
        │ Update Position       │      │ Update Health &       │
        │ History Buffer        │─────>│ Diagnostic Data       │
        └───────────┬───────────┘      └───────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Check Ball Movement   │
        │ - Calculate movement  │
        │ - Update stationary   │
        │   detection flags     │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │ Update Detection      │
        │ Statistics            │
        │ - Increment counter   │
        │ - Update reliability  │
        └───────────────────────┘
        
        Mathematical idea:
        - The Euclidean distance formula is used to measure how much the ball has moved: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
        - Adaptive thresholds allow the system to be more or less sensitive depending on the context (e.g., more strict when stationary).
        
        This method is called automatically by the ROS 2 framework when new messages arrive.
        It processes the incoming data and updates the node's internal state accordingly."""
        current_time = time.time()
        
        # Update health monitor
        self.health_monitor.increment_message_counter('position')
        
        # Extract position - using tuple instead of numpy array for better performance
        position = (msg.point.x, msg.point.y, msg.point.z)
        
        # Update detection time
        self.last_detection_time = current_time
        
        # Update position history for stationary detection with timestamp
        self.position_history.add((position, current_time))
        
        # Calculate position change if we have a previous position
        if self.last_position is not None:
            # Optimized distance calculation (Euclidean distance in 3D)
            position_change = 0.0
            for i in range(3):
                diff = position[i] - self.last_position[i]
                position_change += diff * diff
            position_change = math.sqrt(position_change)
            
            # Count as valid detection with adaptive threshold based on motion state
            valid_change_threshold = 1.0  # Default threshold
            
            # Adjust threshold based on motion state
            if hasattr(self, 'motion_state'):
                if self.motion_state == "stationary":
                    valid_change_threshold = 0.5
                elif self.motion_state == "long_stationary":
                    valid_change_threshold = 0.3
                elif self.motion_state == "medium_fast":
                    valid_change_threshold = 1.5
            
            if position_change < valid_change_threshold:
                self.consecutive_detections += 1
            else:
                # Only flag large jumps in non-fast motion states
                if (self.motion_state != "medium_fast" and 
                    position_change > valid_change_threshold * 1.5):
                    # Throttled logging for performance
                    self.health_monitor.throttled_log(
                        self.get_logger(),
                        f"Large position jump: {position_change:.2f}m (threshold: {valid_change_threshold:.2f}m)",
                        "position_jump",
                        min_interval=1.0
                    )
                    # Don't reset counter completely, just decrement
                    self.consecutive_detections = max(0, self.consecutive_detections - 1)
                else:
                    # Normal increment for expected movement in fast state
                    self.consecutive_detections += 1
        else:
            # First detection
            self.consecutive_detections = 1
        
        # Calculate distance to ball - optimized to only consider XY plane
        self.ball_distance = math.sqrt(position[0]**2 + position[1]**2)
        
        # Update close ball detection with adaptive threshold
        adaptive_proximity = self.proximity_threshold
        if self.adaptive_parameters_enabled and hasattr(self, 'motion_state'):
            if self.motion_state in ["stationary", "long_stationary"]:
                adaptive_proximity *= 1.1  # Slightly larger threshold for stationary
            
        self.is_ball_close = self.ball_distance <= adaptive_proximity
        
        # Update stationary ball detection
        self.update_ball_stationary_status()
        
        # Store current position for next comparison
        self.last_position = position
        
        # Handle state transitions based on position updates
        self.handle_position_based_transitions(current_time)
        
        # Update sensor count for health monitoring
        # Assume at least one sensor active if we're receiving position updates
        if not hasattr(self, 'active_sensor_count'):
            self.active_sensor_count = 1
        self.health_monitor.update_sensors(self.active_sensor_count, self.in_sensor_gap)
    # ===== State Transition Logic =====

    
    def update_ball_stationary_status(self):
    """Check if the ball is stationary with optimized calculations.
        
        Concept:
        - Looks at the recent position history to see if the ball has moved more than a threshold.
        - Uses early exit optimization: if any movement exceeds the threshold, we know the ball is not stationary.
        - Adapts the threshold based on motion state and distance to the ball.
        
        Mathematical idea:
        - Uses the maximum movement in the buffer to decide if the ball is stationary.
        - The threshold is scaled based on context (e.g., further balls appear to move less in the camera frame)."""
        if len(self.position_history) < 3:  # Need multiple samples to determine
            self.is_ball_stationary = False
            return
        
        # Get position history with timestamps
        history = self.position_history.get_all()
        if not history:
            self.is_ball_stationary = False
            return
        
        # Get the most recent position
        latest_position, latest_time = history[-1]
        
        # Calculate maximum movement over history with early exit optimization
        max_movement = 0.0
        max_age = 0.0
        
        # Apply adaptive stationary threshold based on motion state and ball distance
        adaptive_threshold = self.stationary_threshold
        
        if self.adaptive_parameters_enabled:
            # Scale threshold based on motion state
            if hasattr(self, 'motion_state'):
                if self.motion_state == "stationary":
                    adaptive_threshold *= self.adaptive_factor_stationary
                elif self.motion_state == "long_stationary":
                    adaptive_threshold *= self.adaptive_factor_stationary * 1.2
                elif self.motion_state == "medium_fast":
                    adaptive_threshold *= self.adaptive_factor_moving
            
            # Scale threshold based on distance
            # Further balls appear to move less in camera frame
            if self.ball_distance > 2.0:
                adaptive_threshold *= 0.7
            elif self.ball_distance > 1.0:
                adaptive_threshold *= 0.9
        
        for pos, timestamp in history:
            # Calculate position change - optimized distance calculation
            movement = 0.0
            for i in range(3):
                diff = latest_position[i] - pos[i]
                movement += diff * diff
            movement = math.sqrt(movement)
            
            # Early exit optimization - if we exceed threshold, ball is not stationary
            if movement > adaptive_threshold:
                self.is_ball_stationary = False
                
                # Only log if status changed
                if hasattr(self, 'last_stationary_status') and self.last_stationary_status != False:
                    self.get_logger().info(
                        f"Ball is now moving: movement={movement:.3f}m, threshold={adaptive_threshold:.3f}m"
                    )
                    self.last_stationary_status = False
                return
            
            max_movement = max(max_movement, movement)
            
            # Track age of oldest sample
            age = latest_time - timestamp
            max_age = max(max_age, age)
        
        # Ball is stationary if we got here (max_movement <= adaptive_threshold)
        self.is_ball_stationary = True
        
        # Log changes in stationary status with reduced frequency
        if not hasattr(self, 'last_stationary_status') or self.last_stationary_status != self.is_ball_stationary:
            self.get_logger().info(
                f"Ball stationary status changed: {self.is_ball_stationary}, "
                f"movement={max_movement:.3f}m, threshold={adaptive_threshold:.3f}m"
            )
            self.last_stationary_status = self.is_ball_stationary
    
    def handle_position_based_transitions(self, current_time):
    """Handle state transitions with early-exit optimizations.
        
        Concept:
        - Decides when to transition between states based on position updates and time in state.
        - Uses early exit to avoid unnecessary checks.
        - Each state has its own handler for transitions.
        
        Mathematical idea:
        - State transitions are triggered by logical conditions (e.g., enough detections, timeouts, or changes in ball movement).
        - This is a practical application of finite state machines in robotics.
        
        Returns:
            bool: True if a state transition occurred, False otherwise"""
        # Calculate time in current state for hysteresis
        time_in_state = current_time - self.state_start_time
        
        # Apply state-specific handlers with early exits
        if self.current_state == RobotState.INITIALIZING:
            self._handle_initializing_transitions(time_in_state)
        elif self.current_state == RobotState.LOST_BALL:
            self._handle_lost_ball_transitions(time_in_state)
        elif self.current_state == RobotState.RECOVERY:
            self._handle_recovery_transitions(time_in_state)
        elif self.current_state == RobotState.SEARCHING:
            self._handle_searching_transitions(time_in_state)
        elif self.current_state == RobotState.TRACKING:
            self._handle_tracking_transitions(time_in_state, current_time)
        elif self.current_state == RobotState.STOPPED:
            self._handle_stopped_transitions()
    
    def _handle_initializing_transitions(self, time_in_state):
    """Handle transitions from INITIALIZING state with optimized checks.
        
        Returns:
            bool: True if a state transition occurred, False otherwise"""
        # Check for combination of reliability and detections
        tracking_confidence_sufficient = self.tracking_reliable
        
        # Add motion-based criteria
        motion_allows_tracking = True
        if hasattr(self, 'motion_state') and self.motion_state in ["stationary", "long_stationary"]:
            # For stationary balls, lower detection requirements
            detections_sufficient = self.consecutive_detections >= max(2, self.min_tracking_detections - 1)
        else:
            # Regular threshold for moving balls
            detections_sufficient = self.consecutive_detections >= self.min_tracking_detections
        
        # Transition if either criteria met
        if detections_sufficient and (tracking_confidence_sufficient or motion_allows_tracking):
            self.get_logger().info(
                f"Transitioning to TRACKING: consecutive_detections={self.consecutive_detections}, "
                f"reliable={self.tracking_reliable}, motion_state={self.motion_state}"
            )
            self.transition_to_state(RobotState.TRACKING)
    
    def _handle_lost_ball_transitions(self, time_in_state):
    """Handle transitions from LOST_BALL state with optimized checks.
        
        Returns:
            bool: True if a state transition occurred, False otherwise"""
        # Apply hysteresis to prevent rapid state transitions
        if time_in_state < self.lost_ball_hysteresis_time:
            return
            
        # Check for timeout in LOST_BALL state
        if time_in_state > self.max_lost_ball_time:
            self.get_logger().info(f"LOST_BALL timeout after {time_in_state:.1f}s - returning to TRACKING")
            self.transition_to_state(RobotState.TRACKING)
            return
            
        # Check criteria for returning to tracking
        # For stationary balls, we need fewer consecutive detections
        if hasattr(self, 'motion_state') and self.motion_state in ["stationary", "long_stationary"]:
            retracking_threshold = max(2, self.min_retracking_detections - 2)
        else:
            retracking_threshold = self.min_retracking_detections
        
        # Check if we have enough consecutive detections
        if self.consecutive_detections >= retracking_threshold:
            self.get_logger().info(
                f"Transitioning from LOST_BALL to TRACKING: {self.consecutive_detections} "
                f"consecutive detections (threshold: {retracking_threshold})"
            )
            self.transition_to_state(RobotState.TRACKING)
            return
        
        # Alternative criteria: reliable tracking and minimum detections
        if (self.consecutive_detections >= self.min_tracking_detections and 
            (self.tracking_reliable or 
             (hasattr(self, 'is_ball_stationary') and self.is_ball_stationary))):
            self.get_logger().info("Transitioning from LOST_BALL to TRACKING: reliable tracking resumed")
            self.transition_to_state(RobotState.TRACKING)
            return
    
    def _handle_recovery_transitions(self, time_in_state):
    """Handle transitions from RECOVERY state with optimized checks.
        
        Returns:
            bool: True if a state transition occurred, False otherwise"""
        # Apply hysteresis to prevent rapid state transitions
        if time_in_state < self.recovery_hysteresis_time:
            return
            
        # Check for timeout in RECOVERY state
        if time_in_state > self.max_recovery_time:
            self.get_logger().info(f"RECOVERY timeout after {time_in_state:.1f}s - returning to TRACKING")
            self.transition_to_state(RobotState.TRACKING)
            return
        
        # Check if uncertainty has improved
        if self.position_uncertainty < self.uncertainty_recovery_threshold:
            self.get_logger().info(
                f"Recovery successful - uncertainty reduced to {self.position_uncertainty:.3f}m"
            )
            self.transition_to_state(RobotState.TRACKING)
            return
        
        # Check if we have strong detections despite uncertainty
        if self.consecutive_detections >= self.min_retracking_detections + 2:
            self.get_logger().info(
                f"Recovery successful - strong detection sequence ({self.consecutive_detections} frames)"
            )
            self.transition_to_state(RobotState.TRACKING)
            return
    
    def _handle_searching_transitions(self, time_in_state):
    """Handle transitions from SEARCHING state with optimized checks.
        
        Returns:
            bool: True if a state transition occurred, False otherwise"""
        # Criteria for returning to tracking from search
        detection_threshold = self.min_tracking_detections
        if self.motion_state in ["stationary", "long_stationary"]:
            detection_threshold -= 1  # Easier to resume tracking stationary objects
        
        if self.consecutive_detections >= detection_threshold and time_in_state >= 1.0:
            self.get_logger().info("Ball found during search - returning to TRACKING")
            self.transition_to_state(RobotState.TRACKING)
            return
    
    def _handle_tracking_transitions(self, time_in_state, current_time):
    """Handle transitions from TRACKING state with optimized checks.
        
        🔍 BEGINNER'S GUIDE: How Our Robot Decides When to Stop
        ---------------------------------------------------
        
        This function decides when our robot should stop following the ball. It's like when
        you're playing fetch with a dog - if the ball stops moving and is close enough, you
        wait a moment to confirm it's really stopped, then you stop running.
        
        Our robot asks these questions:
        
        1. "Am I close enough to the ball?" (is_ball_close = True)
        2. "Has the ball stopped moving?" (is_ball_stationary = True)
        3. "Has it been still long enough?" (stationary_duration >= threshold)
        
        If all answers are YES, the robot transitions from TRACKING to STOPPED state.
        
        ## How the Code Works:
        
        1. First, we check if we've been tracking long enough to consider changing state
           (this prevents rapid flip-flopping between states)
        
        2. Then we check if the ball is both close and stationary:
           - If yes: We start a timer (or continue it if already started)
           - If no: We reset the timer
        
        3. Once the ball has been close and stationary for a certain time
           (stationary_time_threshold), we transition to STOPPED state
           
        4. We adjust how long we wait based on what we know about the ball:
           - If we know the ball has been still for a long time already,
             we don't wait as long before stopping
        
        Returns:
            bool: True if a state transition occurred, False otherwise"""
        # Apply hysteresis to prevent rapid state transitions
        if time_in_state < self.tracking_hysteresis_time:
            return
            
        if self.is_ball_close and self.is_ball_stationary:
            if self.stationary_start_time is None:
                # First time detecting stationary ball
                self.stationary_start_time = current_time
                return
            
            # Only stop if ball has been stationary for required time
            stationary_duration = current_time - self.stationary_start_time
            
            # Adjust threshold based on motion state for more responsive stopping
            adaptive_threshold = self.stationary_time_threshold
            if self.motion_state == "long_stationary":
                adaptive_threshold *= 0.7  # Shorter threshold for known stationary balls
            
            if stationary_duration >= adaptive_threshold:
                # Ball has been stationary and close for required time
                self.transition_to_state(RobotState.STOPPED)
                self.get_logger().info(
                    f"Ball is close ({self.ball_distance:.2f}m) and stationary "
                    f"for {stationary_duration:.1f}s - stopping"
                )
        else:
            # Reset stationary timer if conditions aren't met
            self.stationary_start_time = None
    
    def _handle_stopped_transitions(self):
    """Handle transitions from STOPPED state with optimized checks.
        
        Returns:
            bool: True if a state transition occurred, False otherwise"""
        # Handle transition based on ball movement or distance
        if not self.is_ball_close or not self.is_ball_stationary:
            reason = "moved away" if not self.is_ball_close else "started moving"
            self.get_logger().info(f"Ball has {reason} - resuming tracking")
            self.transition_to_state(RobotState.TRACKING)
            return
    
    def evaluate_uncertainty_recovery(self):
    """Evaluate if we should enter recovery mode based on uncertainty trends.
        
        Concept:
        - Monitors the trend of position uncertainty to decide if the robot should enter recovery mode.
        - Uses both the current value and the rate of change (trend) to make the decision.
        
        Mathematical idea:
        - If uncertainty is high and rising, it's a sign that the robot is losing track of the ball.
        - This is like watching a graph and noticing when the line starts going up quickly."""
        if self.current_state != RobotState.TRACKING:
            return
        
        # Check uncertainty against threshold with early exit
        if self.position_uncertainty < self.uncertainty_recovery_threshold:
            return
        
        # Check trend to see if it's improving or worsening
        if len(self.uncertainty_history.values) >= 5:
            direction, rate = self.uncertainty_history.get_trend(5)
            
            # Enter recovery if uncertainty is high and rising
            if direction > 0 and rate > 0.01:
                self.get_logger().info(
                    f"Entering RECOVERY state: Uncertainty high ({self.position_uncertainty:.3f}m) "
                    f"and rising ({rate:.3f}/s)"
                )
                self.recovery_reason = "rising_uncertainty"
                self.transition_to_state(RobotState.RECOVERY)
                return
            
            # Also enter recovery if uncertainty is very high even if stable
            if self.position_uncertainty > self.position_uncertainty_threshold:
                self.get_logger().info(
                    f"Entering RECOVERY state: Uncertainty exceeds threshold "
                    f"({self.position_uncertainty:.3f}m > {self.position_uncertainty_threshold:.3f}m)"
                )
                self.recovery_reason = "high_uncertainty"
                self.transition_to_state(RobotState.RECOVERY)
                return
    
    def motion_state_callback(self, msg):
    """Process ball motion state classifications to adapt robot tracking behavior.
        
        This method receives ball motion state information from the fusion node, which
        classifies the ball's movement pattern (e.g., "moving", "stationary"). This
        classification enables adaptive tracking behavior based on the ball's activity.
        
        Data Transformations:
        1. Categorical State Processing:
           Input: ROS String message → Output: Internal motion state variable
           - Parses string message to extract motion category
           - Compares with previous state to detect changes
           - Updates motion state tracking variables
           
        2. Parameter Adaptation:
           Input: Motion state → Output: Adjusted tracking parameters
           - Modifies tracking thresholds based on motion state
           - Adjusts required detection count for different motion patterns
           - Example: Requires fewer detections to maintain tracking during
             stationary periods compared to rapid movement
           
        3. Behavioral Response:
           Input: Motion state transition → Output: Robot behavior adjustments
           - Updates stationary detection flags for stop behavior
           - Enables transition to STOPPED state when appropriate
           - Influences search behavior during LOST_BALL state
           
        Motion State Categories and Their Effects:
        - "moving": Standard tracking parameters, full-speed pursuit
        - "slight_movement": Reduced speed, more sensitive tracking thresholds
        - "stationary": Enables STOPPED state transition, very sensitive thresholds
        - "long_stationary": Accelerated transition to STOPPED state, lowest thresholds
        - "unknown": Default parameters, cautious approach
        
        This dual state model (motion state + robot operational state) creates a more
        nuanced and responsive robot that can adapt to different ball behaviors while
        maintaining the simplicity of a discrete state machine architecture."""
        # Update health monitor
        self.health_monitor.increment_message_counter('motion_state')
        
        # Store previous and current state
        self.last_motion_state = self.motion_state
        self.motion_state = msg.data
        
        # Detect transitions
        motion_state_changed = self.last_motion_state != self.motion_state
        self.in_motion_transition = motion_state_changed
        
        # Update health monitoring (assume 0.7 confidence for now)
        self.health_monitor.update_motion(self.motion_state, 0.7)
        
        # Log state changes - only log the transitions
        if motion_state_changed:
            self.get_logger().info(f"Motion state changed: {self.last_motion_state} → {self.motion_state}")
            
            # During transitions, adjust parameters immediately
            self.adapt_parameters_to_motion_state()
            
            # Force state reevaluation after parameter changes
            if self.current_state in [RobotState.LOST_BALL, RobotState.TRACKING]:
                self.handle_position_based_transitions(time.time())
        
        # Log periodic updates with reduced frequency
        msg_count = self.health_monitor.message_counters.get('motion_state', 0)
        if msg_count % 40 == 0:  # Reduced from 20 to 40
            self.get_logger().info(f"Current motion state: {self.motion_state}")
    
    def adapt_parameters_to_motion_state(self):
    """Adapt tracking parameters with optimized update frequency.
        
        Concept:
        - Adjusts thresholds and timeouts based on the current motion state of the ball.
        - This allows the robot to be more responsive to fast-moving balls and more patient with stationary ones.
        
        Mathematical idea:
        - Parameter adaptation is a form of feedback control: the system changes its behavior based on what it observes.
        - This is a key idea in robotics and control theory."""
        if not self.adaptive_parameters_enabled:
            return
        
        # Throttle updates to reduce CPU usage
        current_time = time.time()
        if hasattr(self, '_last_param_update') and current_time - self._last_param_update < 1.0:
            return
        self._last_param_update = current_time
            
        # Reset parameters to base values first
        self.lost_ball_timeout = self.base_lost_ball_timeout
        self.stationary_threshold = self.base_stationary_threshold
        self.min_tracking_detections = self.base_min_tracking_detections
        self.min_retracking_detections = self.base_min_retracking_detections
        
        # Apply state-specific adjustments
        if self.motion_state == "stationary":
            # Stationary balls need longer timeouts, higher thresholds
            self.lost_ball_timeout *= self.adaptive_factor_stationary
            self.stationary_threshold *= self.adaptive_factor_stationary
            # Keep detection requirements the same
            
        elif self.motion_state == "long_stationary":
            # Long-stationary balls get even more relaxed parameters
            self.lost_ball_timeout *= self.adaptive_factor_stationary * 1.2
            self.stationary_threshold *= self.adaptive_factor_stationary * 1.2
            # Easier to maintain tracking
            self.min_tracking_detections = max(2, int(self.min_tracking_detections * 0.7))
            
        elif self.motion_state == "medium_fast":
            # Fast-moving balls need faster response
            self.lost_ball_timeout *= self.adaptive_factor_moving
            self.stationary_threshold *= self.adaptive_factor_moving
            # More detections to confirm tracking
            self.min_tracking_detections += 1
            self.min_retracking_detections += 2
        
        # Log parameter adaptation with reduced verbosity
        self.get_logger().info(
            f"Adapted parameters for {self.motion_state}: lost_ball_timeout={self.lost_ball_timeout:.2f}s"
        )
    
    def tracking_confidence_callback(self, msg):
    """Process confidence values with reduced logging.
        
        This method is called automatically by the ROS 2 framework when new messages arrive.
        It processes the incoming data and updates the node's internal state accordingly."""
        # Update health monitor
        self.health_monitor.increment_message_counter('tracking_confidence')
        
        # Store confidence value
        self.tracking_confidence = msg.data
        
        # Update health monitoring
        if hasattr(self, 'tracking_reliable'):
            self.health_monitor.update_tracking(self.tracking_reliable, self.tracking_confidence)
        
        # Use confidence as factor in state transitions
        if self.current_state == RobotState.TRACKING:
            # Low confidence can trigger recovery or search
            if self.tracking_confidence < 0.3 and self.consecutive_detections < 5:
                if self.position_uncertainty > self.uncertainty_recovery_threshold:
                    # High uncertainty with low confidence - enter recovery
                    self.get_logger().info(
                        f"Entering RECOVERY state: Low confidence ({self.tracking_confidence:.2f}) "
                        f"with high uncertainty ({self.position_uncertainty:.3f}m)"
                    )
                    self.recovery_reason = "low_confidence"
                    self.transition_to_state(RobotState.RECOVERY)
                else:
                    # Low confidence without high uncertainty - might be temporary
                    # Log but don't change state yet
                    self.health_monitor.throttled_log(
                        self.get_logger(),
                        f"Low tracking confidence: {self.tracking_confidence:.2f}",
                        "low_confidence",
                        min_interval=2.0,
                        level="warn"
                    )
        
        elif self.current_state == RobotState.RECOVERY:
            # High confidence can trigger exit from recovery
            if self.tracking_confidence > 0.7 and self.consecutive_detections >= self.min_tracking_detections:
                self.get_logger().info(
                    f"Exiting RECOVERY: Confidence improved to {self.tracking_confidence:.2f}"
                )
                self.transition_to_state(RobotState.TRACKING)
        
        # Log periodic updates with reduced frequency
        msg_count = self.health_monitor.message_counters.get('tracking_confidence', 0)
        if msg_count % 60 == 0:  # Reduced from 30 to 60
            self.get_logger().info(f"Tracking confidence: {self.tracking_confidence:.2f}")
    
    def sensor_gap_callback(self, msg):
    """Process sensor gap information with optimized handling.
        
        This method is called automatically by the ROS 2 framework when new messages arrive.
        It processes the incoming data and updates the node's internal state accordingly."""
        # Update health monitor
        self.health_monitor.increment_message_counter('sensor_gap')
        
        # Record previous value for change detection
        previous_gap = getattr(self, 'in_sensor_gap', False)
        
        # Store gap status
        self.in_sensor_gap = msg.data
        
        # Update health monitoring - sensor count not updated here
        self.health_monitor.update_sensors(
            getattr(self, 'active_sensor_count', 0), 
            self.in_sensor_gap
        )
        
        # Only update timing information on state changes
        if self.in_sensor_gap != previous_gap:
            if self.in_sensor_gap:
                # New gap started
                self.gap_start_time = time.time()
                self.get_logger().info("Sensor gap detected - entering gap tolerance mode")
            else:
                # Gap ended
                self.gap_duration = 0.0
                self.gap_start_time = None
                self.get_logger().info("Sensor gap ended")
        
        # Calculate gap duration if in a gap
        if self.in_sensor_gap and self.gap_start_time is not None:
            self.gap_duration = time.time() - self.gap_start_time
        
        # Handle gap appropriately based on current state
        self.handle_sensor_gap()
    
    def handle_sensor_gap(self):
    """Handle sensor gaps with early-exit optimization.
        
        Concept:
        - Decides how to handle sensor gaps based on the current state and motion state.
        - Uses adaptive tolerance times to allow for short gaps without changing state.
        
        Mathematical idea:
        - Adaptive tolerance times are based on the motion state (e.g., longer for stationary balls).
        - This is a practical application of adaptive control in robotics."""
        if not self.gap_enabled or not self.in_sensor_gap:
            return
        
        current_time = time.time()
        
        # Get time in gap
        gap_duration = 0.0
        if self.gap_start_time is not None:
            gap_duration = current_time - self.gap_start_time
        
        # Calculate adaptive tolerance time based on motion state
        tolerance_time = self.gap_tolerance_time
        if hasattr(self, 'motion_state'):
            if self.motion_state in ["stationary", "long_stationary"]:
                # Longer tolerance for stationary balls
                tolerance_time *= self.gap_stationary_multiplier
            elif self.motion_state == "small_movement":
                # Medium tolerance for slow movement
                tolerance_time *= 1.5
            elif self.motion_state == "medium_fast":
                # Lower tolerance for fast movement
                tolerance_time *= 0.8
        
        # Handle gap based on current state with early-exit optimization
        if self.current_state == RobotState.TRACKING:
            self._handle_tracking_gap(gap_duration, tolerance_time, current_time)
        elif self.current_state == RobotState.STOPPED:
            self._handle_stopped_gap(gap_duration, tolerance_time, current_time)
    
    def _handle_tracking_gap(self, gap_duration, tolerance_time, current_time):
    """Handle sensor gap in TRACKING state."""
        # Stay in TRACKING during short gaps with adaptive tolerance
        if gap_duration < tolerance_time:
            # Temporarily override the timeout logic
            # Set last detection time to keep within lost_ball_timeout
            self.last_detection_time = current_time - (self.lost_ball_timeout * 0.5)
            
            # Log with moderate frequency
            self.health_monitor.throttled_log(
                self.get_logger(),
                f"Gap tolerance active: {gap_duration:.1f}s/{tolerance_time:.1f}s in {self.motion_state} state",
                "gap_tolerance",
                min_interval=2.0  # Reduced frequency
            )
        else:
            # Gap too long - consider entering recovery mode
            if self.position_uncertainty < self.uncertainty_recovery_threshold:
                # Uncertainty still acceptable - stay in tracking
                self.health_monitor.throttled_log(
                    self.get_logger(),
                    f"Extended gap ({gap_duration:.1f}s) but uncertainty acceptable ({self.position_uncertainty:.3f}m)",
                    "extended_gap",
                    min_interval=2.0  # Reduced frequency
                )
            else:
                # Gap too long with rising uncertainty - enter recovery
                self.get_logger().info(
                    f"Entering RECOVERY state: Gap duration ({gap_duration:.1f}s) exceeds tolerance ({tolerance_time:.1f}s)"
                )
                self.recovery_reason = "sensor_gap"
                self.transition_to_state(RobotState.RECOVERY)
    
    def _handle_stopped_gap(self, gap_duration, tolerance_time, current_time):
    """Handle sensor gap in STOPPED state."""
        # Special protection for STOPPED state during gap
        if hasattr(self, 'motion_state') and self.motion_state in ["stationary", "long_stationary"]:
            # For stationary balls, stay in STOPPED state longer
            extended_tolerance = tolerance_time * 1.5
            if gap_duration < extended_tolerance:
                self.health_monitor.throttled_log(
                    self.get_logger(),
                    f"Protecting STOPPED state during gap: {gap_duration:.1f}s/{extended_tolerance:.1f}s",
                    "stopped_protection",
                    min_interval=4.0  # Reduced frequency
                )
                # Force detection time update to prevent state changes
                self.last_detection_time = current_time
    
    def fusion_diagnostics_callback(self, msg):
    """Process fusion node diagnostics with reduced parsing overhead.
        
        This method is called automatically by the ROS 2 framework when new messages arrive.
        It processes the incoming data and updates the node's internal state accordingly."""
        try:
            # Update health monitor
            self.health_monitor.increment_message_counter('fusion_diagnostics')
            
            # Parse diagnostic data - only when needed
            # Check message counter to reduce parsing frequency
            msg_count = self.health_monitor.message_counters.get('fusion_diagnostics', 0)
            if msg_count % 3 == 0:  # Only parse every 3rd message
                diag_data = json.loads(msg.data)
                
                # Only extract what we need - don't store the entire message
                if 'active_sensors' in diag_data:
                    self.active_sensor_count = len(diag_data['active_sensors'])
                    
                    # Update health monitoring
                    self.health_monitor.update_sensors(
                        self.active_sensor_count, 
                        getattr(self, 'in_sensor_gap', False)
                    )
            
            # Only log with very low frequency
            if msg_count % 60 == 0:  # Reduced from 30 to 60
                self.get_logger().info(f"Fusion diagnostics: {self.active_sensor_count} active sensors")
                
        except Exception as e:
            # Log errors but continue execution
            self.get_logger().error(f"Error processing fusion diagnostics: {str(e)}")
    
    def state_manager_callback(self):
    """Main control loop executed periodically (5Hz) to manage the robot's operational state.
        
        This callback serves as the central orchestration point for the entire state machine,
        coordinating all aspects of the robot's behavior. It performs a systematic sequence
        of operations:
        
        Data Flow and Transformations:
        1. Temporal Analysis Pipeline:
           Input: Current time, last detection time → Output: Time-based metrics
           - Computes time since last detection (for timeout handling)
           - Calculates time spent in current state (for state-specific behaviors)
           - Determines sensor gap durations (for reliability assessment)
           
        2. State-Specific Processing:
           Input: Current state, time metrics → Output: State-appropriate behaviors
           - Delegates to specialized handlers for each robot state
           - Each handler generates appropriate motion commands for that state
           - Handlers assess conditions specific to their state
        
        3. Transition Evaluation:
           Input: Current conditions → Output: State transition decisions
           - Evaluates if conditions warrant a state change
           - Applies hysteresis and protection logic
           - Executes transitions when appropriate
           
        4. Safety & Recovery Assessment:
           Input: System conditions → Output: Safety responses
           - Monitors for anomalous conditions regardless of state
           - Triggers recovery behavior when needed
           - Provides fallback behaviors for unexpected situations
           
        State Manager Execution Flow:
        
        ┌────────────────────────────────────────────────────────────┐
        │                       Timer Trigger                         │
        └──────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────────────────────────────────────────────┐
        │                     Calculate Timings                       │
        │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
        │  │ Time in state│  │Time since    │  │ Gap duration │      │
        │  └──────────────┘  │last detection│  └──────────────┘      │
        └──────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────────────────────────────────────────────┐
        │                  State-Specific Handling                    │
        │  ┌──────────────────────────────────────────────────────┐  │
        │  │ Switch (current_state)                               │  │
        │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
        │  │  │ INITIALIZING│  │ TRACKING    │  │ SEARCHING   │   │  │
        │  │  └─────────────┘  └─────────────┘  └─────────────┘   │  │
        │  │                                                      │  │
        │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │  │
        │  │  │ LOST_BALL   │  │ STOPPED     │  │ RECOVERY    │   │  │
        │  │  └─────────────┘  └─────────────┘  └─────────────┘   │  │
        │  └──────────────────────────────────────────────────────┘  │
        └──────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────────────────────────────────────────────┐
        │                 Transition Evaluation                       │
        │  ┌──────────────────────────────────────────────────────┐  │
        │  │ Switch (current_state)                               │  │
        │  │  ┌─────────────────────┐  ┌─────────────────────┐    │  │
        │  │  │ Check INITIALIZING  │  │ Check TRACKING      │    │  │
        │  │  │ transitions         │  │ transitions         │    │  │
        │  │  └─────────────────────┘  └─────────────────────┘    │  │
        │  │                                                      │  │
        │  │  ┌─────────────────────┐  ┌─────────────────────┐    │  │
        │  │  │ Check other state   │  │ Apply hysteresis &  │    │  │
        │  │  │ transitions         │  │ validation          │    │  │
        │  │  └─────────────────────┘  └─────────────────────┘    │  │
        │  └──────────────────────────────────────────────────────┘  │
        └──────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────────────────────────────────────────────┐
        │                 Safety Checks & Recovery                    │
        │  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐ │
        │  │ Check sensor   │  │ Evaluate       │  │ Monitor system │ │
        │  │ gaps           │  │ uncertainty    │  │ health         │ │
        │  └────────────────┘  └────────────────┘  └───────────────┘ │
        └──────────────────────────────┬─────────────────────────────┘
                                       │
                                       ▼
        ┌────────────────────────────────────────────────────────────┐
        │                  Command Generation                         │
        │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
        │  │ Generate     │  │ Apply safety │  │ Publish      │      │
        │  │ state-based  │  │ limits       │  │ command      │      │
        │  │ movement     │  │              │  │              │      │
        │  └──────────────┘  └──────────────┘  └──────────────┘      │
        └────────────────────────────────────────────────────────────┘
        
        This method is called automatically by the ROS 2 framework when new messages arrive.
        It processes the incoming data and updates the node's internal state accordingly."""
        current_time = time.time()
        
        # Update adaptive parameters periodically with reduced frequency
        if self.adaptive_parameters_enabled and hasattr(self, 'motion_state'):
            # Only update every ~2 seconds instead of 1 second
            if not hasattr(self, 'last_parameter_update') or current_time - self.last_parameter_update > 2.0:
                # Only update if not in transition (to avoid parameter fluctuations)
                if not getattr(self, 'in_motion_transition', False):
                    self.adapt_parameters_to_motion_state()
                    self.last_parameter_update = current_time
        
        # Calculate time since last detection
        time_since_detection = (current_time - self.last_detection_time 
                              if self.last_detection_time is not None else float('inf'))
        
        # Print state summary much less frequently (every 6 seconds instead of 3)
        if not hasattr(self, 'state_manager_call_count'):
            self.state_manager_call_count = 0
        self.state_manager_call_count += 1
        
        if self.state_manager_call_count % 30 == 0:  # 30 calls at 5Hz = ~6 seconds
            # Lazily evaluate warning string only when needed
            health_warnings = ""
            if hasattr(self, 'health_monitor') and len(getattr(self.health_monitor, 'warnings', [])) > 0:
                health_warnings = f", Warnings: {', '.join(self.health_monitor.warnings[:2])}"
                if len(self.health_monitor.warnings) > 2:
                    health_warnings += f"... (+{len(self.health_monitor.warnings) - 2})"
                
            # Use fewer string formatting operations
            self.get_logger().info(
                f"State: {self.current_state}, Detect: {self.consecutive_detections}, "
                f"Dist: {self.ball_distance:.2f}m, Uncert: {self.position_uncertainty:.3f}m{health_warnings}"
            )
        
        # Handle state-specific behaviors with optimized method calls
        getattr(self, f'_handle_{self.current_state.lower()}_state')(current_time, time_since_detection)
    
    def _handle_initializing_state(self, current_time, time_since_detection):
    """Handle behavior in INITIALIZING state."""
        # Check if we should timeout initialization
        time_in_state = current_time - self.state_start_time
        if time_in_state > 5.0:  # 5 seconds to initialize
            if self.consecutive_detections >= 2:  
                # If we have any detections, try tracking
                self.get_logger().info("Initialization timeout with detections - transitioning to TRACKING")
                self.transition_to_state(RobotState.TRACKING)
            else:
                # No detections - go to LOST_BALL to wait
                self.get_logger().info("Initialization timeout with no detections - transitioning to LOST_BALL")
                self.transition_to_state(RobotState.LOST_BALL)
    
    def _handle_tracking_state(self, current_time, time_since_detection):
    """Handle behavior in TRACKING state."""
        # STEP 1: Evaluate if tracking is still reliable
        # This checks if we should continue trying to track the ball or give up
        # Think of this as asking "Do we still have a good idea where the ball is?"
        reliability_check = self.evaluate_tracking_reliability(time_since_detection)
        
        # STEP 2: Handle the case where tracking has become unreliable
        # If tracking is no longer reliable, we need to change our behavior
        if not reliability_check:
            # IMPORTANT: Apply hysteresis to prevent rapid state changes
            # Hysteresis means "resistance to change" - we don't want to immediately
            # give up tracking just because of a brief unreliable moment
            # This is like waiting a moment to make sure the ball is really lost
            time_in_state = current_time - self.state_start_time
            if time_in_state < self.tracking_hysteresis_time:
                # We haven't been in TRACKING state long enough to make a decision
                # This prevents flickering between states when tracking is borderline
                return
            
            # STEP 3: Determine why tracking failed and transition to LOST_BALL
            # We want to know WHY tracking failed for diagnostics and debugging
            
            # There are two main reasons tracking can fail:
            reason = "unreliable tracking"  # Default: too much uncertainty or inconsistency
            
            # Check if we simply haven't seen the ball for too long
            # This is like saying "I haven't seen the ball in 5 seconds, so I must have lost it"
            if time_since_detection > self.lost_ball_timeout:
                reason = "detection timeout"
            
            # Log the reason for debugging and user information
            self.get_logger().info(f"Ball lost! Reason: {reason}")
            
            # STEP 4: Formally change the robot's state to LOST_BALL
            # This will trigger different motion behavior (usually stopping)
            self.transition_to_state(RobotState.LOST_BALL)
    
    def _handle_lost_ball_state(self, current_time, time_since_detection):
    """Handle behavior in LOST_BALL state."""
        # Already handled transitions in position_callback
        # Just stay in LOST_BALL state - we don't search for the ball
        # Keep the robot stationary during LOST_BALL
        self.stop_robot()
    
    def _handle_searching_state(self, current_time, time_since_detection):
    """Handle behavior in SEARCHING state."""
        # Execute search pattern
        self.execute_search_rotation()
        
        # Check timeout
        time_in_state = current_time - self.state_start_time
        if time_in_state > self.max_search_time:
            self.get_logger().info(f"Search timeout after {time_in_state:.1f}s - transitioning to LOST_BALL")
            self.transition_to_state(RobotState.LOST_BALL)
    
    def _handle_recovery_state(self, current_time, time_since_detection):
    """Handle behavior in RECOVERY state."""
        # Check timeout
        time_in_state = current_time - self.state_start_time
        if time_in_state > self.max_recovery_time:
            self.get_logger().info(f"Recovery timeout after {time_in_state:.1f}s - returning to TRACKING")
            self.transition_to_state(RobotState.TRACKING)
            return
        
        # Monitor uncertainty trends during recovery
        if len(self.uncertainty_history.values) >= 3:
            direction, rate = self.uncertainty_history.get_trend(3)
            
            # If uncertainty is decreasing significantly, consider ending recovery
            if direction < 0 and abs(rate) > 0.03:
                self.get_logger().info(
                    f"Exiting RECOVERY: Uncertainty decreasing at {abs(rate):.3f}/s "
                    f"({self.position_uncertainty:.3f}m)"
                )
                self.transition_to_state(RobotState.TRACKING)
                return
            
            # If uncertainty is still increasing despite recovery, try LOST_BALL
            elif direction > 0 and rate > 0.05 and time_in_state > 1.0:
                self.get_logger().info(
                    f"Recovery unsuccessful - uncertainty still rising at {rate:.3f}/s. "
                    f"Transitioning to LOST_BALL."
                )
                self.transition_to_state(RobotState.LOST_BALL)
                return
        
        # Stay in recovery mode - stop the robot
        self.stop_robot()
        
        # Log recovery status periodically with reduced frequency
        if self.state_manager_call_count % 30 == 0:  # Reduced from 20 to 30
            self.get_logger().info(
                f"In RECOVERY mode: reason={self.recovery_reason}, "
                f"duration={time_in_state:.1f}s, uncertainty={self.position_uncertainty:.3f}m"
            )
    
    def _handle_stopped_state(self, current_time, time_since_detection):
    """Handle behavior in STOPPED state."""
        # Already handling transitions in position_callback
        # Just ensure we're stopped
        if self.state_manager_call_count % 10 == 0:  # Only check periodically to reduce CPU
            self.stop_robot()
    
    def evaluate_tracking_reliability(self, time_since_detection):
    """Evaluate tracking reliability with optimized early-exit checks.
        
        Args:
            time_since_detection (float): Time since last detection
            
        Returns:
            bool: True if tracking is reliable, False otherwise"""
        # Get motion state for context
        motion_state = getattr(self, 'motion_state', 'unknown')
        
        # First check detection timeout - fastest check
        if time_since_detection > self.lost_ball_timeout:
            # Special case for stationary balls
            if (self.is_ball_stationary or motion_state in ["stationary", "long_stationary"]) and \
               time_since_detection < self.lost_ball_timeout * 1.5:
                return True
            return False
        
        # Early exit if tracking is reliable
        if self.tracking_reliable:
            return True
        
        # Check if we have consistent detections
        if self.consecutive_detections >= self.min_retracking_detections:
            return True
        
        # Check if we're in a temporary sensor gap
        if self.in_sensor_gap and self.gap_enabled:
            if self.gap_start_time is not None:
                gap_duration = time.time() - self.gap_start_time
                
                # Calculate adaptive tolerance time based on motion state
                tolerance_time = self.gap_tolerance_time
                if motion_state in ["stationary", "long_stationary"]:
                    tolerance_time *= self.gap_stationary_multiplier
                
                if gap_duration < tolerance_time:
                    return True
        
        # Check motion state for special cases
        if motion_state in ["stationary", "long_stationary"]:
            return True
            
        # If we get here, tracking is not reliable
        return False
    
    def execute_search_rotation(self):
    """Execute an optimized search rotation to find the ball."""
        # Initialize search start time if needed
        if self.search_rotation_start_time is None:
            self.search_rotation_start_time = time.time()
            self.search_angle_accumulated = 0.0
        
        # Calculate time in search and adjust direction if needed
        search_time = time.time() - self.search_rotation_start_time
        
        # Reverse direction after some time to avoid winding cables
        if search_time > self.max_rotation_time / 2 and self.search_direction > 0:
            self.search_direction = -1
            self.get_logger().info("Switching search direction to clockwise")
        
        # Calculate rotation command - reuse same Twist object to reduce allocations
        if not hasattr(self, '_search_twist'):
            self._search_twist = Twist()
        
        self._search_twist.angular.z = self.search_direction * self.search_rotation_speed
        
        # Update accumulated angle
        self.search_angle_accumulated += abs(self._search_twist.angular.z) * 0.2  # 5Hz updates
        
        # Publish command
        self.cmd_vel_publisher.publish(self._search_twist)
    
    def transition_to_state(self, new_state):
    """Handle state transitions with optimized protection logic.
        
        🔍 BEGINNER'S GUIDE: How State Transitions Work
        -------------------------------------------
        
        This function is responsible for changing the robot's state (for example, from TRACKING to SEARCHING).
        Think of it like changing gears in a car - there's a specific process to follow:
        
        1. **Check if we actually need to change** - If we're already in the target state, do nothing
        2. **Make sure we're not changing too frequently** - Prevents rapid back-and-forth switching
        3. **Clean up the old state** - Take care of anything that needs to be done before leaving
        4. **Initialize the new state** - Set up everything needed for the new state
        5. **Notify the system** - Tell other parts of the robot about the change
        
        ## Real-World Example
        
        Think about switching activities in your day:
        
        - You're reading a book (READING state)
        - You decide to go for a run (want to transition to RUNNING state)
        - You bookmark your page and put the book down (clean up old state)
        - You change into running clothes and shoes (initialize new state)
        - You tell your family you're going for a run (notify the system)
        - Now you're in the RUNNING state!
        
        ## Why is this function complex?
        
        Transitions need to be carefully managed because:
        - Changing states too frequently can make the robot seem "jittery"
        - Some states require special cleanup (like stopping motors)
        - The robot might need different parameters in different states
        - We need to track how long we spend in each state
        
        Concept:
        
        This method implements the core state transition mechanism for the robot's finite state machine.
        It coordinates all aspects of changing from one operational state to another, including:
        
        Data Transformations:
        1. State Validation Transform:
           Input: Target state → Output: Validated transition decision
           - Compares current and target states to prevent redundant transitions
           - Applies hysteresis control based on transition history and timing
           - Implements rate limiting to prevent oscillations between states
        
        2. State Cleanup Operations:
           Input: Current state → Output: Clean system state
           - Performs state-specific cleanup actions (e.g., stopping motors when exiting TRACKING)
           - Resets state-specific timers and counters
           - Generates appropriate notification messages
        
        3. State Initialization:
           Input: Target state → Output: Initialized new state
           - Sets up initial conditions for the new state
           - Configures state-specific parameters and behavior settings
           - Initializes timers and tracking variables
        
        4. Event Notification:
           Input: State change event → Output: ROS messages and logging
           - Creates and publishes state change messages for other nodes
           - Generates diagnostic logging with transition reasoning
           - Updates state history for system analysis
        
        Hysteresis Implementation:
        A critical concept in this method is "hysteresis" - the prevention of rapid back-and-forth
        transitions between states. This is achieved through:
        
        1. Transition Rate Limiting: 
           - Each state pair has a minimum transition interval (e.g., must wait 2s before
             returning to a state you just left)
           - Formula: current_time - last_transition_time >= min_interval
        
        2. Transition Counting:
           - Tracks how many times certain transitions have been attempted
           - Increases required transition evidence when oscillation is detected
           
        3. Adaptive Thresholds:
           - Detection requirements get stricter after multiple state changes
           - Prevents "bouncing" between states due to noisy data
        
        Mathematical Basis:
        The state machine implements a modified Mealy machine (finite automaton where transitions
        depend on both current state and input conditions) with:
        - States: Discrete robot operational modes (INITIALIZING, TRACKING, etc.)
        - Inputs: Sensor data, timing information, and confidence metrics
        - Transition Functions: Methods like _handle_tracking_transitions
        - Hysteresis: Added temporal component that considers transition history
        
        Returns:
            bool: True if transition succeeded, False if blocked by hysteresis or validation
        
        Engineering Rationale:
        Stable state management is critical in robotics because:
        1. Physical systems have inertia and cannot respond instantly to rapid command changes
        2. Sensor noise can trigger false transitions without adequate filtering
        3. User experience suffers when robot behavior appears erratic or indecisive
        4. Battery efficiency decreases with frequent motor direction changes
        
        This robustness comes at the cost of slightly delayed responses to genuine state
        change conditions, a necessary trade-off for reliable operation.
        
        Transition Logic Visualization:
        
                                ┌─────────────────────┐
                                │ transition_to_state │
                                └──────────┬──────────┘
                                           │
                                           ▼
                               ┌────────────────────────┐
                           No  │                        │ Yes
                        ┌──────┤  new_state != current  ├───────┐
                        │      │        state?          │       │
                        │      └────────────────────────┘       │
                        │                                       │
                        │                                       ▼
                        │                            ┌─────────────────────┐
                        │                            │  apply_state        │
                        │                            │  protection         │
                        │                            └─────────┬───────────┘
                        │                                      │
                        │                                      ▼
                        │                            ┌─────────────────────┐
                        │                            │   Protection        │ Yes
                        │                            │   blocked           ├───┐
                        │                            │   transition?       │   │
                        │                            └─────────┬───────────┘   │
                        │                                      │ No            │
                        │                                      ▼               │
                        │                             ┌──────────────────┐     │
                        │                             │ Record previous  │     │
                        │                             │ state            │     │
                        │                             └───────┬──────────┘     │
                        │                                     │                │
                        │                                     ▼                │
                        │                             ┌──────────────────┐     │
                        │                             │ Update timers &  │     │
                        │                             │ counters         │     │
                        │                             └───────┬──────────┘     │
                        │                                     │                │
                        │                                     ▼                │
                        │                             ┌──────────────────┐     │
                        │                             │ State-specific   │     │
                        │                             │ initialization   │     │
                        │                             └───────┬──────────┘     │
                        │                                     │                │
                        │                                     ▼                │
                        │                             ┌──────────────────┐     │
                        │                             │ Update motion    │     │
                        │                             │ commands         │     │
                        │                             └───────┬──────────┘     │
                        │                                     │                │
                        │                                     ▼                │
                        │                             ┌──────────────────┐     │
                        │                             │ Log and publish  │     │
                        │                             │ state change     │     │
                        │                             └───────┬──────────┘     │
                        │                                     │                │
                        ▼                                     ▼                │
        ┌──────────────────────────┐             ┌──────────────────────┐     │
        │ Return false (unchanged) │             │ Return true (changed)│     │
        └──────────────────────────┘             └──────────────────────┘     │
                                                            ▲                 │
                                                            └─────────────────┘
        Returns:
            bool: True if a state transition occurred, False otherwise"""
        # Apply state protection to prevent rapid oscillations
        new_state = self.apply_state_protection(new_state)
        
        if new_state == self.current_state:
            return
        
        # Log the transition
        time_in_prev_state = time.time() - self.state_start_time
        self.get_logger().info(
            f"State transition: {self.current_state} → {new_state} "
            f"(after {time_in_prev_state:.1f}s)"
        )
        
        # Store previous state for reference
        self.previous_state = self.current_state
        
        # Update state and reset state timer
        self.current_state = new_state
        self.state_start_time = time.time()
        self.last_state_change_time = time.time()
        
        # Reset state-specific variables with reduced logging
        if new_state == RobotState.TRACKING:
            self.get_logger().info("Ball tracking initiated")
        
        elif new_state == RobotState.LOST_BALL:
            self.get_logger().info("Ball lost. Entering wait mode.")
            self.stop_robot()
            self.search_rotation_start_time = None
        
        elif new_state == RobotState.STOPPED:
            self.get_logger().info("Ball is close and stationary - stopping robot")
            self.stop_robot()
        
        elif new_state == RobotState.SEARCHING:
            self.get_logger().info("Starting search pattern")
            self.search_rotation_start_time = time.time()
            self.search_angle_accumulated = 0.0
        
        elif new_state == RobotState.RECOVERY:
            # Increment recovery attempt counter
            self.recovery_attempt_count += 1
            self.get_logger().info(
                f"Entering recovery mode (attempt #{self.recovery_attempt_count}): "
                f"reason={self.recovery_reason}"
            )
            self.stop_robot()
        
        # Record transition for hysteresis tracking
        self.state_transition_history.add(self.current_state)
        self.transition_times[self.current_state] = time.time()
        
        # Publish the new state
        self.publish_state()
    
    def apply_state_protection(self, proposed_state):
    """Apply optimized protection against rapid state oscillations.
        
        Concept:
        - Checks if enough time has passed in the current state before allowing a transition.
        - Detects oscillating patterns in the state history and blocks them.
        - Special protection for the STOPPED state during sensor gaps.
        
        Mathematical idea:
        - Uses minimum time thresholds (hysteresis) and pattern detection in the state history.
        - This is a practical application of state machine protection in robotics."""
        # Get the current time to calculate how long we've been in this state
        current_time = time.time()
        
        # Calculate how long the robot has been in its current state
        # Example: If we entered TRACKING at time 100.0 and now it's 102.5, we've been in this state for 2.5 seconds
        time_in_state = current_time - self.state_start_time
        
        # SPECIAL CASE: We always allow transitions from INITIALIZING
        # This is because INITIALIZING is just a startup state, not a normal operational state
        if self.current_state == RobotState.INITIALIZING:
            return proposed_state  # No protection, allow any transition
        
        # HYSTERESIS IMPLEMENTATION:
        # Define the minimum time the robot must spend in each state before transitioning
        # This prevents the robot from rapidly switching between states (like flickering)
        # Think of this like "debouncing" a button press in electronics
        min_times = {
            # Must stay in TRACKING for a minimum time (set by parameter)
            RobotState.TRACKING: self.tracking_hysteresis_time,  
            
            # Must stay in LOST_BALL for a minimum time (set by parameter)
            # This prevents the robot from giving up searching too quickly
            RobotState.LOST_BALL: self.lost_ball_hysteresis_time,
            
            # Must search for at least 1.5 seconds before changing state
            # This gives the search pattern time to actually look around
            RobotState.SEARCHING: 1.5,
            
            # Don't immediately leave STOPPED state (prevents jitter when the ball is still)
            RobotState.STOPPED: 0.5,
            
            # Must stay in RECOVERY for a minimum time (set by parameter)
            # This gives time for sensor issues to resolve
            RobotState.RECOVERY: self.recovery_hysteresis_time
        }
            
        # Get the minimum time for our current state (default to 0 if not specified)
        # This tells us how long we must stay in this state before transitioning
        min_time = min_times.get(self.current_state, 0.0)
        
        if time_in_state < min_time:
            # Count blocked transitions
            transition_key = f"{self.current_state}->{proposed_state}"
            self.hysteresis_counts[transition_key] = self.hysteresis_counts.get(transition_key, 0) + 1
            
            # Only log significant blocked transitions
            if proposed_state != self.current_state:
                self.health_monitor.throttled_log(
                    self.get_logger(),
                    f"Blocked transition to {proposed_state}: too soon "
                    f"({time_in_state:.1f}s < {min_time:.1f}s required)",
                    f"blocked_{transition_key}",
                    min_interval=2.0
                )
            
            return self.current_state
            
        # Check for oscillating transitions with optimized history checking
        history = self.state_transition_history.get_all()
        if len(history) >= 4:
            # If we detect a pattern like A->B->A->B
            if (history[-1] == history[-3] and 
                history[-2] == history[-4] and
                proposed_state == history[-2]):
                
                # Check if these transitions happened in quick succession
                if current_time - self.transition_times.get(history[-4], 0) < 5.0:
                    # This is an oscillation - apply hysteresis by remaining in current state
                    self.get_logger().info(
                        f"Detected state oscillation pattern. "
                        f"Remaining in {self.current_state} for stability."
                    )
                    return self.current_state
        
        # Special case: Protect STOPPED state during gaps for stationary balls
        if (self.current_state == RobotState.STOPPED and
            proposed_state == RobotState.TRACKING and
            self.in_sensor_gap and
            self.motion_state in ["stationary", "long_stationary"]):
            
            self.get_logger().info(
                f"Protecting STOPPED state during sensor gap for {self.motion_state} ball"
            )
            return self.current_state
            
        # Allow the transition
        return proposed_state
    
    def stop_robot(self):
    """Send command to stop all robot motion with reduced overhead.
        
        Concept:
        - Publishes a zero-velocity command to stop the robot.
        - Uses a pre-allocated message for efficiency."""
        # Reuse the same Twist object to reduce allocations
        if not hasattr(self, '_stop_twist'):
            self._stop_twist = Twist()  # All fields initialize to 0
        self.cmd_vel_publisher.publish(self._stop_twist)
    
    def publish_state(self):
    """Publish current robot state for other nodes to consume.
        
        Concept:
        - Publishes the current state as a message so other parts of the system can react.
        - Uses a pre-allocated message for efficiency."""
        # Reuse the same message object to reduce allocations
        if not hasattr(self, '_state_msg'):
            self._state_msg = String()
        self._state_msg.data = self.current_state
        self.state_publisher.publish(self._state_msg)
    
    def health_check_callback(self):
    """Perform periodic health checks with reduced computational overhead.
        
        Concept:
        - Evaluates the health of the system and logs warnings if needed.
        - Adjusts tracking parameters if health is degraded.
        - Publishes health status for monitoring.
        
        Mathematical idea:
        - Health is a function of confidence, warnings, and system state.
        - This is a practical example of monitoring and feedback in robotics.
        
        Returns:
            bool: True if a state transition occurred, False otherwise"""
        # Evaluate overall health
        new_warnings = self.health_monitor.evaluate_health()
        
        # Calculate system confidence
        system_confidence = self.health_monitor.calculate_system_confidence()
        
        # Log new warnings
        for warning in new_warnings:
            self.get_logger().warn(f"Health warning: {warning}")
        
        # Adjust behavior based on health with reduced frequency
        if system_confidence < self.health_confidence_threshold:
            self.health_monitor.throttled_log(
                self.get_logger(),
                f"System health degraded: confidence={system_confidence:.2f}, "
                f"warnings={len(self.health_monitor.warnings)}",
                "degraded_health",
                min_interval=5.0,  # Increased from 3.0 to 5.0
                level="warn"
            )
            
            # Adjust tracking parameters based on health
            if self.adaptive_parameters_enabled:
                # More conservative tracking with degraded health
                self.lost_ball_timeout *= 0.7
                self.min_retracking_detections += 1
        
        # Publish health status
        if not hasattr(self, '_health_msg'):
            self._health_msg = String()
            
        health_data = {
            'system_confidence': round(system_confidence, 3),
            'warnings': self.health_monitor.warnings[:5],  # Limit to 5 warnings
            'state': self.current_state
        }
        self._health_msg.data = json.dumps(health_data, cls=FastJSONEncoder)
        self.health_publisher.publish(self._health_msg)
    
    def publish_diagnostics(self):
    """Publish diagnostics with optimized JSON serialization and reduced content.
        
        Concept:
        - Publishes a summary of the robot's state, tracking, and health for monitoring and debugging.
        - Includes trends and resource usage in full diagnostics."""
        current_time = time.time()
        
        # Get basic diagnostic data
        if self.last_detection_time is not None:
            time_since_detection = current_time - self.last_detection_time
        else:
            time_since_detection = float('inf')

        state_duration = current_time - self.state_start_time
        
        # Create position description only if needed
        position_info = {}
        if self.last_position is not None:
            # Optimize calculation of direction
            direction = math.degrees(math.atan2(self.last_position[1], self.last_position[0]))
            
            position_info = {
                "distance": round(self.ball_distance, 2),
                "direction": round(direction, 1),
                "coordinates": [
                    round(self.last_position[0], 2),
                    round(self.last_position[1], 2),
                    round(self.last_position[2], 2)
                ]
            }
        
        # Build diagnostic info with minimal content for regular updates
        diagnostic_info = {
            "state": self.current_state,
            "tracking": {
                "reliable": self.tracking_reliable,
                "consecutive_detections": self.consecutive_detections,
                "uncertainty": round(self.position_uncertainty, 3),
                "time_since_detection": round(time_since_detection, 2)
            },
            "ball": {
                "distance": round(self.ball_distance, 2),
                "is_close": self.is_ball_close,
                "is_stationary": self.is_ball_stationary
            }
        }
        
        # Every 5+ seconds, include full diagnostics
        full_diagnostics = False
        if current_time - self.last_full_diagnostic_time > self.full_diagnostic_rate:
            full_diagnostics = True
            self.last_full_diagnostic_time = current_time
            
            # Add position info only in full diagnostics
            if position_info:
                diagnostic_info["ball"]["position"] = position_info
                
            # Add state duration in full diagnostics
            diagnostic_info["state_duration"] = round(state_duration, 1)
            
            # Add motion state info
            if hasattr(self, 'motion_state'):
                diagnostic_info["motion_state"] = self.motion_state
            
            # Add sensor gap information
            if hasattr(self, 'in_sensor_gap') and self.in_sensor_gap:
                gap_duration = 0.0
                if hasattr(self, 'gap_start_time') and self.gap_start_time is not None:
                    gap_duration = current_time - self.gap_start_time
                    
                diagnostic_info["sensor_gap"] = {
                    "active": True,
                    "duration": round(gap_duration, 2),
                }
            
            # Add uncertainty trend analysis
            if hasattr(self, 'uncertainty_history') and len(self.uncertainty_history.values) >= 3:
                direction, rate = self.uncertainty_history.get_trend(5)
                
                trend_name = "stable"
                if direction > 0:
                    trend_name = "rising"
                elif direction < 0:
                    trend_name = "falling"
                
                diagnostic_info["uncertainty_trend"] = {
                    "trend": trend_name,
                    "rate": round(abs(rate), 3),
                }
            
            # Add system health information
            if hasattr(self, 'health_monitor'):
                diagnostic_info["system_health"] = {
                    "confidence": round(self.health_monitor.system_confidence, 2),
                    "warnings_count": len(self.health_monitor.warnings)
                }
        
        # Only include resource usage in full diagnostics
        if full_diagnostics and self.resource_monitoring_enabled:
            try:
                # Add memory usage
                import resource
                memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                diagnostic_info["resources"] = {
                    "memory_kb": memory_usage,
                    "uptime": round(current_time - self.start_time, 1)
                }
            except ImportError:
                pass
        
        # Publish diagnostic info
        if not hasattr(self, '_diag_msg'):
            self._diag_msg = String()
            
        self._diag_msg.data = json.dumps(diagnostic_info, cls=FastJSONEncoder)
        self.diagnostics_publisher.publish(self._diag_msg)
        
        # Log summarized diagnostics with reduced frequency
        if full_diagnostics:
            # Only log basic info to reduce string processing
            self.get_logger().info(
                f"Diagnostic: {self.current_state}, "
                f"Ball: {self.ball_distance:.2f}m "
                f"({'stationary' if self.is_ball_stationary else 'moving'}), "
                f"Health: {getattr(self.health_monitor, 'system_confidence', 0.0):.2f}"
            )


def main(args=None):
    """Main function to initialize and run the state manager node."""
    rclpy.init(args=args)
    
    # Welcome message - reduced content
    print("=================================================")
    print("Optimized Basketball Chaser - State Manager Node")
    print("=================================================")
    
    try:
        # Create node
        node = OptimizedBallChaseStateManager()
        
        # Use MultiThreadedExecutor with adjusted thread count for Pi 5
        from rclpy.executors import MultiThreadedExecutor
        # 2 threads for Pi 5 to avoid overloading
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        
        try:
            # Spin with executor
            executor.spin()
        except KeyboardInterrupt:
            node.get_logger().info("State Manager shutdown requested (Ctrl+C)")
        finally:
            # Cleanup
            executor.shutdown()
            
    except Exception as e:
        print(f"Error starting State Manager: {str(e)}")
    finally:
        # Clean shutdown
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()