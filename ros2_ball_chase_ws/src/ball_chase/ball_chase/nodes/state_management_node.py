#!/usr/bin/env python3

"""
Enhanced Basketball Chaser - State Management Node

This node determines the robot's behavior based on tracking reliability,
transitioning between states like initialization, tracking, lost ball, 
and stopped states.

Improvements:
- Enhanced motion state integration with adaptive parameters
- Uncertainty-based decision making with trend analysis
- Sophisticated sensor gap handling with recovery modes
- Complete diagnostic data integration with confidence metrics
- Resource-efficient implementation with optimized memory usage
"""

import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import PointStamped, TwistStamped, Twist
from std_msgs.msg import String, Bool, Float32
import numpy as np
import time
import math
import json
from collections import deque  # For tracking history

# Add a custom JSON encoder for NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    """JSON Encoder that can handle NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyJSONEncoder, self).default(obj)

class RobotState:
    """Enumeration of robot operational states"""
    INITIALIZING = "initializing"  # Startup state, waiting for first reliable detection
    TRACKING = "tracking"          # Actively tracking the ball with reliable detections
    LOST_BALL = "lost_ball"        # Ball not found after extensive searching
    STOPPED = "stopped"            # Stationary state when ball is close and stationary
    SEARCHING = "searching"        # Actively searching for a lost ball
    RECOVERY = "recovery"          # Recovery mode during sensor gaps or high uncertainty

class FixedSizeBuffer:
    """Efficient fixed-size buffer with circular array implementation."""
    
    def __init__(self, max_size=10):
        """Initialize with fixed buffer size."""
        self.data = []
        self.max_size = max_size
        self.next_index = 0
        self.is_full = False
    
    def add(self, value):
        """Add value to buffer with fixed memory allocation."""
        if len(self.data) < self.max_size:
            self.data.append(value)
        else:
            self.data[self.next_index] = value
            self.next_index = (self.next_index + 1) % self.max_size
            self.is_full = True
    
    def get_all(self):
        """Get all values as a list."""
        return self.data.copy()
    
    def get_latest(self, count=1):
        """Get the most recent n values."""
        if not self.data:
            return []
        if count >= len(self.data):
            return self.data.copy()
        
        if self.is_full:
            # Buffer is full - need to calculate indices
            start_idx = (self.next_index - count) % self.max_size
            if start_idx < self.next_index:
                return self.data[start_idx:self.next_index]
            else:
                return self.data[start_idx:] + self.data[:self.next_index]
        else:
            # Buffer not full yet - just return last n elements
            return self.data[-count:]
    
    def clear(self):
        """Clear the buffer."""
        self.data = []
        self.next_index = 0
        self.is_full = False
    
    def __len__(self):
        """Get current buffer size."""
        return len(self.data)

class TrendAnalyzer:
    """Analyzes trends in time series data with efficient memory usage."""
    
    def __init__(self, window_size=10):
        """Initialize with window size."""
        self.values = FixedSizeBuffer(window_size)
        self.timestamps = FixedSizeBuffer(window_size)
    
    def add(self, value, timestamp=None):
        """Add a value with optional timestamp."""
        if timestamp is None:
            timestamp = time.time()
        
        self.values.add(value)
        self.timestamps.add(timestamp)
    
    def get_trend(self, num_samples=None):
        """
        Calculate trend direction and rate.
        
        Returns:
            tuple: (direction, rate) where direction is 1 (rising), -1 (falling), or 0 (stable)
                  and rate is the average change per second
        """
        if len(self.values) < 2:
            return 0, 0.0
        
        # Get samples to analyze
        if num_samples is None or num_samples > len(self.values):
            num_samples = len(self.values)
        
        values = self.values.get_latest(num_samples)
        timestamps = self.timestamps.get_latest(num_samples)
        
        # Calculate differences
        diffs = []
        rates = []
        
        for i in range(1, len(values)):
            value_diff = values[i] - values[i-1]
            time_diff = timestamps[i] - timestamps[i-1]
            
            diffs.append(value_diff)
            if time_diff > 0:
                rates.append(value_diff / time_diff)
        
        # Calculate average rate
        if not rates:
            return 0, 0.0
        
        avg_rate = sum(rates) / len(rates)
        
        # Determine direction
        if abs(avg_rate) < 0.001:  # Threshold for stability
            return 0, 0.0
        
        direction = 1 if avg_rate > 0 else -1
        return direction, avg_rate
    
    def is_stable(self, threshold=0.05):
        """Check if values are stable (not changing significantly)."""
        if len(self.values) < 2:
            return True
        
        values = self.values.get_all()
        if not values:
            return True
        
        # Calculate min-max range
        min_val = min(values)
        max_val = max(values)
        reference = max(abs(min_val), abs(max_val), 0.01)  # Avoid division by zero
        
        # Check if range is less than threshold
        return (max_val - min_val) / reference < threshold
    
    def get_stability_score(self):
        """Get stability score (0-1, higher is more stable)."""
        if len(self.values) < 2:
            return 1.0
        
        values = self.values.get_all()
        if not values:
            return 1.0
        
        # Calculate standard deviation
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance)
        
        # Calculate normalized stability score
        reference = max(abs(mean), 0.01)  # Avoid division by zero
        normalized_std = std_dev / reference
        
        # Convert to stability score (higher is more stable)
        stability = 1.0 / (1.0 + 10.0 * normalized_std)
        return max(0.0, min(1.0, stability))

class SystemHealthMonitor:
    """Monitors system health metrics with diagnostic tracking."""
    
    def __init__(self):
        """Initialize health monitor."""
        # Components to monitor
        self.components = {
            'tracking': {
                'status': False,
                'confidence': 0.0,
                'last_update': 0.0
            },
            'fusion': {
                'status': False,
                'uncertainty': float('inf'),
                'last_update': 0.0
            },
            'motion': {
                'state': 'unknown',
                'confidence': 0.0,
                'last_update': 0.0
            },
            'sensors': {
                'active_count': 0,
                'gap_detected': False,
                'last_update': 0.0
            }
        }
        
        # Trend analysis
        self.trends = {
            'uncertainty': TrendAnalyzer(20),
            'tracking_confidence': TrendAnalyzer(10),
            'sensor_count': TrendAnalyzer(5)
        }
        
        # Warning flags
        self.warnings = []
        self.warning_history = deque(maxlen=10)
        
        # System metrics
        self.system_confidence = 1.0
        self.message_counters = {}
        
        # Initialize logging helper
        self._last_throttled_logs = {}
    
    def update_tracking(self, reliable, confidence):
        """Update tracking status."""
        current_time = time.time()
        
        self.components['tracking']['status'] = reliable
        self.components['tracking']['confidence'] = confidence
        self.components['tracking']['last_update'] = current_time
        
        # Update trend
        self.trends['tracking_confidence'].add(confidence, current_time)
    
    def update_fusion(self, uncertainty):
        """Update fusion status."""
        current_time = time.time()
        
        self.components['fusion']['uncertainty'] = uncertainty
        self.components['fusion']['last_update'] = current_time
        
        # Auto-calculate status based on uncertainty thresholds
        self.components['fusion']['status'] = uncertainty < 0.4
        
        # Update trend
        self.trends['uncertainty'].add(uncertainty, current_time)
    
    def update_motion(self, state, confidence=0.7):
        """Update motion state."""
        current_time = time.time()
        
        self.components['motion']['state'] = state
        self.components['motion']['confidence'] = confidence
        self.components['motion']['last_update'] = current_time
    
    def update_sensors(self, active_count, gap_detected=False):
        """Update sensor status."""
        current_time = time.time()
        
        self.components['sensors']['active_count'] = active_count
        self.components['sensors']['gap_detected'] = gap_detected
        self.components['sensors']['last_update'] = current_time
        
        # Update trend
        self.trends['sensor_count'].add(active_count, current_time)
    
    def evaluate_health(self):
        """Evaluate overall system health and generate warnings."""
        current_time = time.time()
        self.warnings = []
        
        # Check for stale data
        for component, data in self.components.items():
            age = current_time - data.get('last_update', 0)
            if age > 2.0:
                self.warnings.append(f"{component}_stale_data")
        
        # Check for degraded tracking
        if (self.components['tracking']['confidence'] < 0.4 and 
            not self.components['tracking']['status']):
            self.warnings.append('tracking_degraded')
        
        # Check for high uncertainty
        if self.components['fusion']['uncertainty'] > 0.5:
            # Check if uncertainty is rising
            direction, rate = self.trends['uncertainty'].get_trend(5)
            if direction > 0 and rate > 0.05:
                self.warnings.append('uncertainty_rising')
            else:
                self.warnings.append('high_uncertainty')
        
        # Check for sensor gaps during tracking
        if (self.components['sensors']['gap_detected'] and 
            self.components['tracking']['status']):
            self.warnings.append('sensor_gap_during_tracking')
        
        # Check for low sensor count
        if self.components['sensors']['active_count'] < 1:
            self.warnings.append('no_active_sensors')
        
        # Log new warnings
        new_warnings = [w for w in self.warnings if w not in self.warning_history]
        if new_warnings:
            self.warning_history.extend(new_warnings)
            return new_warnings
        
        return []
    
    def calculate_system_confidence(self):
        """Calculate overall system confidence metric (0-1)."""
        # Start with base confidence
        confidence = 1.0
        
        # Factor in tracking confidence
        tracking_weight = 0.4
        tracking_confidence = self.components['tracking']['confidence']
        confidence *= (tracking_weight * tracking_confidence + (1 - tracking_weight))
        
        # Factor in fusion uncertainty (invert to get confidence)
        uncertainty = self.components['fusion']['uncertainty']
        uncertainty_factor = 1.0 / (1.0 + uncertainty * 2.0)  # Higher uncertainty = lower factor
        uncertainty_weight = 0.3
        confidence *= (uncertainty_weight * uncertainty_factor + (1 - uncertainty_weight))
        
        # Factor in sensor count
        sensor_count = self.components['sensors']['active_count']
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
        """Get diagnostic data as JSON-compatible dictionary."""
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
            # Clean up data - remove timestamps and functions
            clean_data = {k: v for k, v in info.items() if k != 'last_update'}
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
        """Track message counts for diagnostics."""
        if topic not in self.message_counters:
            self.message_counters[topic] = 0
        self.message_counters[topic] += 1
        return self.message_counters[topic]
    
    def throttled_log(self, logger, message, key, min_interval=1.0, level="info"):
        """Log with throttling to reduce overhead."""
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

class EnhancedBallChaseStateManager(Node):
    """
    Enhanced state management node for the basketball chasing robot.
    
    Determines appropriate robot behavior based on tracking reliability,
    motion state, and uncertainty metrics with adaptive parameters.
    """
    
    def __init__(self):
        """Initialize the state manager node."""
        super().__init__('ball_chase_state_manager')
        
        # Create callback groups for concurrency control
        self.timer_cb_group = MutuallyExclusiveCallbackGroup()
        self.subscription_cb_group = ReentrantCallbackGroup()
        
        # Start time for elapsed time tracking
        self.start_time = time.time()
        
        # Declare parameters
        self._declare_parameters()
        
        # Load parameters
        self._load_parameters()
        
        # Initialize state variables with default values
        self._init_state_variables()
        
        # Initialize health monitoring
        self.health_monitor = SystemHealthMonitor()
        
        # Set up subscriptions with staged startup
        self._setup_subscriptions()
        
        # Set up publishers
        self._setup_publishers()
        
        # Set up timers
        self._setup_timers()
        
        self.get_logger().info("Enhanced Basketball Chaser State Manager initialized in INITIALIZING state")
        self.publish_state()
    
    def _declare_parameters(self):
        """Declare all parameters with default values."""
        self.declare_parameters(
            namespace='',
            parameters=[
                # Timing thresholds
                ('lost_ball_timeout', 1.5),          # Seconds without detection to consider ball lost
                ('max_search_time', 30.0),           # Seconds to search before giving up
                ('stationary_time_threshold', 1.5),  # Time ball needs to be stationary before stopping
                ('max_lost_ball_time', 5.0),         # Maximum time to stay in LOST_BALL state
                ('max_recovery_time', 3.0),          # Maximum time in RECOVERY state
                
                # Search pattern parameters
                ('search_rotation_speed', 0.5),      # Rotation speed during search (rad/s)
                ('max_rotation_time', 15.0),         # Maximum time to rotate before giving up
                
                # Detection thresholds
                ('min_tracking_detections', 3),      # Consecutive detections to confirm tracking
                ('min_retracking_detections', 6),    # Higher threshold for transitioning back to tracking
                ('proximity_threshold', 0.5),        # Distance to consider ball "close" (meters)
                ('stationary_threshold', 0.05),      # Max movement to consider ball stationary
                
                # Uncertainty thresholds
                ('position_uncertainty_threshold', 0.5),  # Maximum acceptable position uncertainty
                ('uncertainty_recovery_threshold', 0.35), # Uncertainty threshold for recovery
                
                # Hysteresis parameters
                ('tracking_hysteresis_time', 1.0),   # Minimum time in tracking state
                ('lost_ball_hysteresis_time', 0.5),  # Minimum time in lost ball state
                ('recovery_hysteresis_time', 0.3),   # Minimum time in recovery state
                
                # Adaptive parameters
                ('adaptive_parameters_enabled', True), # Enable adaptive parameters
                ('adaptive_factor_stationary', 1.5),  # Parameter scaling for stationary balls
                ('adaptive_factor_moving', 0.8),      # Parameter scaling for moving balls
                
                # Gap tolerance parameters
                ('gap_tolerance_time', 1.5),          # Maximum gap tolerance time
                ('gap_stationary_multiplier', 2.0),   # Gap tolerance multiplier for stationary balls
                ('gap_enabled', True),                # Enable gap tolerance
                
                # System health parameters
                ('health_confidence_threshold', 0.5), # System confidence threshold for normal operation
                ('health_check_interval', 1.0),       # Health check interval in seconds
                
                # Resource management parameters
                ('diagnostic_publish_rate', 1.0),     # Rate to publish diagnostics (Hz)
                ('full_diagnostic_rate', 5.0),        # Rate for full diagnostics (seconds)
                ('resource_monitoring_enabled', True) # Enable resource monitoring
            ]
        )
    
    def _load_parameters(self):
        """Load parameters from ROS parameter server."""
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
        
        # Resource management parameters
        self.diagnostic_publish_rate = self.get_parameter('diagnostic_publish_rate').value
        self.full_diagnostic_rate = self.get_parameter('full_diagnostic_rate').value
        self.resource_monitoring_enabled = self.get_parameter('resource_monitoring_enabled').value
        
        # Store base parameters for adaptive scaling
        self.base_lost_ball_timeout = self.lost_ball_timeout
        self.base_stationary_threshold = self.stationary_threshold
        self.base_min_tracking_detections = self.min_tracking_detections
        self.base_min_retracking_detections = self.min_retracking_detections
    
    def _init_state_variables(self):
        """Initialize state tracking variables."""
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
        
        # Position history for stationary detection
        self.position_history = FixedSizeBuffer(10)
        
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
        
        # State protection variables
        self.state_transition_history = deque(maxlen=10)
        self.transition_times = {}
        self.last_state_change_time = time.time()
        self.hysteresis_counts = {}  # Count of blocked transitions
        
        # Uncertainty tracking
        self.uncertainty_history = TrendAnalyzer(20)  # Track uncertainty trend
        self.uncertainty_history.add(self.position_uncertainty)
        
        # Diagnostic data
        self.diagnostic_data = {}
        self.last_full_diagnostic_time = 0.0
        
        # Message counters
        self.message_counts = {}
    
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
        
        # System health status
        self.health_publisher = self.create_publisher(
            String,
            '/robot/health',
            5
        )
        
        # Enhanced diagnostics
        self.diagnostics_publisher = self.create_publisher(
            String,
            '/robot/diagnostics',
            3
        )
    
    def _setup_timers(self):
        """Set up all timer callbacks with prioritized rates."""
        # Critical state management timer (10Hz)
        self.state_timer = self.create_timer(
            0.1, 
            self.state_manager_callback,
            callback_group=self.timer_cb_group
        )
        
        # Health check timer (adjustable 1-5Hz based on health)
        self.health_timer = self.create_timer(
            self.health_check_interval, 
            self.health_check_callback,
            callback_group=self.timer_cb_group
        )
        
        # Periodic state republishing (0.5Hz)
        self.state_republish_timer = self.create_timer(
            2.0, 
            self.publish_state,
            callback_group=self.timer_cb_group
        )
        
        # Diagnostic publication (1Hz)
        self.diagnostic_timer = self.create_timer(
            1.0 / self.diagnostic_publish_rate,
            self.publish_diagnostics,
            callback_group=self.timer_cb_group
        )
        
        self.get_logger().info("Timers set up with prioritized rates")
    
    def tracking_status_callback(self, msg):
        """
        Process tracking reliability flag from the fusion node.
        
        Args:
            msg (Bool): Whether tracking is reliable
        """
        # Update health monitor
        self.health_monitor.increment_message_counter('tracking_status')
        
        # Store tracking status
        self.tracking_reliable = msg.data
        
        # Update health monitoring
        if hasattr(self, 'tracking_confidence'):
            self.health_monitor.update_tracking(self.tracking_reliable, self.tracking_confidence)
        else:
            self.health_monitor.update_tracking(self.tracking_reliable, 0.5)
        
        # Log with reduced frequency
        msg_count = self.health_monitor.message_counters.get('tracking_status', 0)
        if msg_count % 10 == 0:
            self.get_logger().info(f"Fusion tracking status message #{msg_count}: reliable={self.tracking_reliable}")
    
    def uncertainty_callback(self, msg):
        """
        Process position uncertainty from the fusion node.
        
        Args:
            msg (Float32): Position uncertainty in meters
        """
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
        if msg_count % 10 == 0 or significant_change:
            direction, rate = self.uncertainty_history.get_trend(5)
            trend_str = "stable"
            if direction > 0:
                trend_str = f"rising ({rate:.3f}/s)"
            elif direction < 0:
                trend_str = f"falling ({-rate:.3f}/s)"
                
            self.get_logger().info(
                f"Position uncertainty: {self.position_uncertainty:.3f}m, "
                f"trend: {trend_str}"
            )
    
    def position_callback(self, msg):
        """
        Process ball position updates from the fusion node.
        
        Args:
            msg (PointStamped): 3D position of the ball
        """
        current_time = time.time()
        
        # Update health monitor
        self.health_monitor.increment_message_counter('position')
        
        # Extract position
        position = np.array([msg.point.x, msg.point.y, msg.point.z])
        
        # Update detection time
        self.last_detection_time = current_time
        
        # Update position history for stationary detection with timestamp
        self.position_history.add((position, current_time))
        
        # Calculate position change if we have a previous position
        if self.last_position is not None:
            position_change = np.linalg.norm(position - self.last_position)
            
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
                    self.get_logger().info(
                        f"Large position jump detected: {position_change:.2f}m "
                        f"(threshold: {valid_change_threshold:.2f}m)"
                    )
                    # Don't reset counter completely, just decrement
                    self.consecutive_detections = max(0, self.consecutive_detections - 1)
                else:
                    # Normal increment for expected movement in fast state
                    self.consecutive_detections += 1
        else:
            # First detection
            self.consecutive_detections = 1
        
        # Calculate distance to ball
        self.ball_distance = np.linalg.norm(position[:2])  # Only consider XY plane distance
        
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
    
    def update_ball_stationary_status(self):
        """Check if the ball hasn't moved significantly over recent history with adaptive thresholds."""
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
        
        # Calculate maximum movement over history
        max_movement = 0.0
        max_age = 0.0
        
        for pos, timestamp in history:
            # Calculate position change
            movement = np.linalg.norm(latest_position - pos)
            max_movement = max(max_movement, movement)
            
            # Track age of oldest sample
            age = latest_time - timestamp
            max_age = max(max_age, age)
        
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
        
        # Ball is stationary if maximum movement is below threshold
        self.is_ball_stationary = max_movement <= adaptive_threshold
        
        # Log changes in stationary status with reduced frequency
        if not hasattr(self, 'last_stationary_status') or self.last_stationary_status != self.is_ball_stationary:
            self.get_logger().info(
                f"Ball stationary status changed: {self.is_ball_stationary}, "
                f"movement={max_movement:.3f}m, threshold={adaptive_threshold:.3f}m"
            )
            self.last_stationary_status = self.is_ball_stationary
    
    def handle_position_based_transitions(self, current_time):
        """
        Handle state transitions based on new position information with motion awareness.
        
        Args:
            current_time (float): Current timestamp
        """
        # Calculate time in current state for hysteresis
        time_in_state = current_time - self.state_start_time
        
        # Transition from INITIALIZING to TRACKING
        if self.current_state == RobotState.INITIALIZING:
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
        
        # Transition from LOST_BALL to TRACKING
        elif self.current_state == RobotState.LOST_BALL:
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
        
        # Transition from RECOVERY to TRACKING
        elif self.current_state == RobotState.RECOVERY:
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
        
        # Transition from SEARCHING to TRACKING
        elif self.current_state == RobotState.SEARCHING:
            # Criteria for returning to tracking from search
            detection_threshold = self.min_tracking_detections
            if self.motion_state in ["stationary", "long_stationary"]:
                detection_threshold -= 1  # Easier to resume tracking stationary objects
            
            if self.consecutive_detections >= detection_threshold and time_in_state >= 1.0:
                self.get_logger().info("Ball found during search - returning to TRACKING")
                self.transition_to_state(RobotState.TRACKING)
                return
        
        # Handle transition to STOPPED when ball is close and stationary
        elif self.current_state == RobotState.TRACKING:
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
        
        # Handle transition back from STOPPED if ball moves or is no longer close
        elif self.current_state == RobotState.STOPPED:
            # Handle transition based on ball movement or distance
            if not self.is_ball_close or not self.is_ball_stationary:
                reason = "moved away" if not self.is_ball_close else "started moving"
                self.get_logger().info(f"Ball has {reason} - resuming tracking")
                self.transition_to_state(RobotState.TRACKING)
                return
    
    def evaluate_uncertainty_recovery(self):
        """
        Evaluate if we should enter recovery mode based on uncertainty trends.
        Only triggers during TRACKING state.
        """
        if self.current_state != RobotState.TRACKING:
            return
        
        # Check uncertainty against threshold
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
        """
        Process motion state updates with enhanced integration.
        
        Args:
            msg (String): Current motion state (stationary, long_stationary, small_movement, medium_fast)
        """
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
        
        # Log state changes
        if motion_state_changed:
            self.get_logger().info(f"Motion state changed: {self.last_motion_state} → {self.motion_state}")
            
            # During transitions, adjust parameters immediately
            self.adapt_parameters_to_motion_state()
            
            # Force state reevaluation after parameter changes
            if self.current_state in [RobotState.LOST_BALL, RobotState.TRACKING]:
                self.handle_position_based_transitions(time.time())
        
        # Log periodic updates with moderate frequency
        msg_count = self.health_monitor.message_counters.get('motion_state', 0)
        if msg_count % 20 == 0:
            self.get_logger().info(f"Current motion state: {self.motion_state}")
    
    def adapt_parameters_to_motion_state(self):
        """Adapt tracking parameters based on the current motion state."""
        if not self.adaptive_parameters_enabled:
            return
        
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
        
        # Log parameter adaptation
        self.get_logger().info(
            f"Adapted parameters for {self.motion_state}: "
            f"lost_ball_timeout={self.lost_ball_timeout:.2f}s, "
            f"stationary_threshold={self.stationary_threshold:.3f}m, "
            f"min_tracking={self.min_tracking_detections}, "
            f"min_retracking={self.min_retracking_detections}"
        )
    
    def tracking_confidence_callback(self, msg):
        """
        Process confidence values with enhanced decision making.
        
        Args:
            msg (Float32): Confidence level (0.0-1.0) of current tracking
        """
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
        
        # Log periodic updates with low frequency
        msg_count = self.health_monitor.message_counters.get('tracking_confidence', 0)
        if msg_count % 30 == 0:
            self.get_logger().info(f"Tracking confidence: {self.tracking_confidence:.2f}")
    
    def sensor_gap_callback(self, msg):
        """
        Process sensor gap information with adaptive gap tolerance.
        
        Args:
            msg (Bool): Whether the system is currently in a sensor gap
        """
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
        
        # Record gap start time for duration tracking
        if self.in_sensor_gap and not previous_gap:
            # New gap started
            self.gap_start_time = time.time()
            self.get_logger().info("Sensor gap detected - entering gap tolerance mode")
        elif not self.in_sensor_gap and previous_gap:
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
        """Handle sensor gaps with state-specific behaviors."""
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
        
        # Handle gap based on current state
        if self.current_state == RobotState.TRACKING:
            # Stay in TRACKING during short gaps with adaptive tolerance
            if gap_duration < tolerance_time:
                # Temporarily override the timeout logic
                # Set last detection time to keep within lost_ball_timeout
                self.last_detection_time = current_time - (self.lost_ball_timeout * 0.5)
                
                # Log with moderate frequency
                self.health_monitor.throttled_log(
                    self.get_logger(),
                    f"Gap tolerance active: {gap_duration:.1f}s/{tolerance_time:.1f}s "
                    f"in {self.motion_state} state",
                    "gap_tolerance",
                    min_interval=1.0
                )
            else:
                # Gap too long - consider entering recovery mode
                if self.position_uncertainty < self.uncertainty_recovery_threshold:
                    # Uncertainty still acceptable - stay in tracking
                    self.health_monitor.throttled_log(
                        self.get_logger(),
                        f"Extended gap ({gap_duration:.1f}s) but uncertainty acceptable "
                        f"({self.position_uncertainty:.3f}m)",
                        "extended_gap",
                        min_interval=1.0
                    )
                else:
                    # Gap too long with rising uncertainty - enter recovery
                    self.get_logger().info(
                        f"Entering RECOVERY state: Gap duration ({gap_duration:.1f}s) "
                        f"exceeds tolerance ({tolerance_time:.1f}s)"
                    )
                    self.recovery_reason = "sensor_gap"
                    self.transition_to_state(RobotState.RECOVERY)
                
        elif self.current_state == RobotState.STOPPED:
            # Special protection for STOPPED state during gap
            if hasattr(self, 'motion_state') and self.motion_state in ["stationary", "long_stationary"]:
                # For stationary balls, stay in STOPPED state longer
                extended_tolerance = tolerance_time * 1.5
                if gap_duration < extended_tolerance:
                    self.health_monitor.throttled_log(
                        self.get_logger(),
                        f"Protecting STOPPED state during gap: {gap_duration:.1f}s/{extended_tolerance:.1f}s",
                        "stopped_protection",
                        min_interval=2.0
                    )
                    # Force detection time update to prevent state changes
                    self.last_detection_time = current_time
    
    def fusion_diagnostics_callback(self, msg):
        """
        Process fusion node diagnostics for health monitoring.
        
        Args:
            msg (String): JSON-formatted diagnostic data
        """
        try:
            # Update health monitor
            self.health_monitor.increment_message_counter('fusion_diagnostics')
            
            # Parse diagnostic data
            diag_data = json.loads(msg.data)
            
            # Store for system health evaluation
            self.diagnostic_data = diag_data
            
            # Extract useful information
            if 'active_sensors' in diag_data:
                self.active_sensor_count = len(diag_data['active_sensors'])
                
                # Update health monitoring
                self.health_monitor.update_sensors(
                    self.active_sensor_count, 
                    getattr(self, 'in_sensor_gap', False)
                )
            
            # Only log with low frequency
            msg_count = self.health_monitor.message_counters.get('fusion_diagnostics', 0)
            if msg_count % 30 == 0:
                self.get_logger().info(f"Fusion diagnostics received: {self.active_sensor_count} active sensors")
                
        except Exception as e:
            self.get_logger().error(f"Error processing fusion diagnostics: {str(e)}")
    
    def state_manager_callback(self):
        """
        Regular timer callback for state management.
        
        Called at 10Hz to:
        1. Monitor tracking quality and perform state transitions
        2. Execute appropriate actions for the current state
        3. Handle timeouts and recovery actions
        """
        current_time = time.time()
        
        # Update adaptive parameters periodically
        if self.adaptive_parameters_enabled and hasattr(self, 'motion_state'):
            # Only update if not in transition (to avoid parameter fluctuations)
            if not getattr(self, 'in_motion_transition', False):
                # Update less frequently (every ~1 second) to avoid overhead
                if not hasattr(self, 'last_parameter_update') or current_time - self.last_parameter_update > 1.0:
                    self.adapt_parameters_to_motion_state()
                    self.last_parameter_update = current_time
        
        # Calculate time since last detection
        time_since_detection = (current_time - self.last_detection_time 
                              if self.last_detection_time is not None else float('inf'))
        
        # Print state summary every 30 calls (approximately 3 seconds)
        if not hasattr(self, 'state_manager_call_count'):
            self.state_manager_call_count = 0
        self.state_manager_call_count += 1
        
        if self.state_manager_call_count % 30 == 0:
            health_warnings = ""
            if hasattr(self, 'health_monitor') and getattr(self.health_monitor, 'warnings', []):
                health_warnings = f", Warnings: {', '.join(self.health_monitor.warnings)}"
                
            self.get_logger().info(
                f"State: {self.current_state}, Detections: {self.consecutive_detections}, "
                f"Reliable: {self.tracking_reliable}, Distance: {self.ball_distance:.2f}m, "
                f"Uncertainty: {self.position_uncertainty:.3f}m{health_warnings}"
            )
        
        # Handle state-specific behaviors
        if self.current_state == RobotState.TRACKING:
            # Determine when to transition from TRACKING to LOST_BALL
            reliability_check = self.evaluate_tracking_reliability(time_since_detection)
            
            if not reliability_check:
                # Apply hysteresis before state change
                time_in_state = current_time - self.state_start_time
                if time_in_state < self.tracking_hysteresis_time:
                    return
                
                # Transition to LOST_BALL with reason
                reason = "unreliable tracking"
                if time_since_detection > self.lost_ball_timeout:
                    reason = "detection timeout"
                
                self.get_logger().info(f"Ball lost! Reason: {reason}")
                self.transition_to_state(RobotState.LOST_BALL)
        
        elif self.current_state == RobotState.INITIALIZING:
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
        
        elif self.current_state == RobotState.LOST_BALL:
            # Check for transition back to TRACKING based on consecutive detections
            detection_threshold = self.min_retracking_detections
            
            # Apply adaptive threshold for stationary balls
            if hasattr(self, 'motion_state') and self.motion_state in ["stationary", "long_stationary"]:
                detection_threshold = max(3, detection_threshold - 2)
            
            if self.consecutive_detections >= detection_threshold:
                self.get_logger().info(
                    f"Detected {self.consecutive_detections} consecutive frames in LOST_BALL - "
                    f"returning to TRACKING"
                )
                self.transition_to_state(RobotState.TRACKING)
                return
            # Check for timeout in LOST_BALL state
            time_in_state = current_time - self.state_start_time
            if time_in_state > self.max_lost_ball_time:
                self.get_logger().info(f"LOST_BALL timeout after {time_in_state:.1f}s - returning to TRACKING")
                self.transition_to_state(RobotState.TRACKING)
                return
                
            # Just stay in LOST_BALL state - we don't search for the ball
            # Keep the robot stationary during LOST_BALL
            self.stop_robot()
        
        elif self.current_state == RobotState.SEARCHING:
            # Execute search pattern
            self.execute_search_rotation()
            
            # Check timeout
            time_in_state = current_time - self.state_start_time
            if time_in_state > self.max_search_time:
                self.get_logger().info(f"Search timeout after {time_in_state:.1f}s - transitioning to LOST_BALL")
                self.transition_to_state(RobotState.LOST_BALL)
        
        elif self.current_state == RobotState.RECOVERY:
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
            
            # Log recovery status periodically
            if self.state_manager_call_count % 20 == 0:
                self.get_logger().info(
                    f"In RECOVERY mode: reason={self.recovery_reason}, "
                    f"duration={time_in_state:.1f}s, uncertainty={self.position_uncertainty:.3f}m"
                )
    
    def evaluate_tracking_reliability(self, time_since_detection):
        """
        Evaluate tracking reliability with adaptive criteria.
        
        Args:
            time_since_detection (float): Time since last detection
            
        Returns:
            bool: True if tracking is reliable, False otherwise
        """
        # Get motion state for context
        motion_state = getattr(self, 'motion_state', 'unknown')
        
        # Don't transition to LOST_BALL if ball is stationary
        ignore_reliability = (
            self.is_ball_stationary or 
            motion_state in ["stationary", "long_stationary"]
        )
        
        # Extend timeout for stationary balls
        if ignore_reliability and time_since_detection < self.lost_ball_timeout * 1.5:
            return True
        
        # Check consecutive detections for stability assessment
        has_consistent_detections = self.consecutive_detections >= self.min_retracking_detections
        
        # Check if we're in a temporary sensor gap
        in_tolerated_gap = False
        if self.in_sensor_gap and self.gap_enabled:
            if self.gap_start_time is not None:
                gap_duration = time.time() - self.gap_start_time
                
                # Calculate adaptive tolerance time based on motion state
                tolerance_time = self.gap_tolerance_time
                if motion_state in ["stationary", "long_stationary"]:
                    tolerance_time *= self.gap_stationary_multiplier
                
                in_tolerated_gap = gap_duration < tolerance_time
        
        # Combine criteria
        # Keep tracking if:
        # 1. Tracking is reliable, OR
        # 2. We have consistent detections, OR
        # 3. We're in a tolerated gap, OR
        # 4. We're ignoring unreliability due to motion state
        reliability_ok = (
            self.tracking_reliable or 
            has_consistent_detections or 
            in_tolerated_gap or 
            ignore_reliability
        )
        
        # Check detection timeout
        timeout_ok = time_since_detection <= self.lost_ball_timeout
        
        # Return combined result
        return reliability_ok and timeout_ok
    
    def execute_search_rotation(self):
        """Execute a search rotation to find the ball."""
        # Initialize search start time if needed
        if self.search_rotation_start_time is None:
            self.search_rotation_start_time = time.time()
            self.search_angle_accumulated = 0.0
        
        # Calculate time in search and adjust direction if needed
        search_time = time.time() - self.search_rotation_start_time
        
        # Reverse direction after some time to avoid winding cables
        if search_time > self.max_rotation_time / 2:
            if self.search_direction > 0:  # Only switch once
                self.search_direction = -1
                self.get_logger().info("Switching search direction to clockwise")
        
        # Calculate rotation command
        twist = Twist()
        twist.angular.z = self.search_direction * self.search_rotation_speed
        
        # Update accumulated angle
        self.search_angle_accumulated += abs(twist.angular.z) * 0.1  # 10Hz updates
        
        # Publish command
        self.cmd_vel_publisher.publish(twist)
    
    def transition_to_state(self, new_state):
        """
        Handle state transitions with state protection.
        
        Args:
            new_state (str): The state to transition to
        """
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
        
        # Handle exit actions for current state
        if self.current_state == RobotState.LOST_BALL:
            # Record total time ball was lost
            self.total_lost_time = getattr(self, 'total_lost_time', 0) + time_in_prev_state
        
        # Store previous state for reference
        self.previous_state = self.current_state
        
        # Update state and reset state timer
        self.current_state = new_state
        self.state_start_time = time.time()
        self.last_state_change_time = time.time()
        
        # Reset state-specific variables
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
        self.state_transition_history.append(self.current_state)
        self.transition_times[self.current_state] = time.time()
        
        # Publish the new state
        self.publish_state()
    
    def apply_state_protection(self, proposed_state):
        """
        Apply protection against rapid state oscillations.
        
        Args:
            proposed_state (str): The proposed new state
            
        Returns:
            str: The actual state to transition to (may be different from proposed)
        """
        current_time = time.time()
        time_in_state = current_time - self.state_start_time
        
        # Always allow transitions from INITIALIZING
        if self.current_state == RobotState.INITIALIZING:
            return proposed_state
        
        # Define minimum times in each state
        min_time_in_state = {
            RobotState.TRACKING: self.tracking_hysteresis_time,
            RobotState.LOST_BALL: self.lost_ball_hysteresis_time,
            RobotState.SEARCHING: 1.5,  # At least 1.5 seconds in SEARCHING
            RobotState.STOPPED: 0.5,    # At least 0.5 second in STOPPED
            RobotState.RECOVERY: self.recovery_hysteresis_time
        }
            
        # Block transitions if not enough time in current state
        min_time = min_time_in_state.get(self.current_state, 0.0)
        
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
            
        # Check for oscillating transitions (ping-pong between states)
        if len(self.state_transition_history) >= 4:
            recent_states = list(self.state_transition_history)
            
            # If we detect a pattern like A->B->A->B
            if (recent_states[-1] == recent_states[-3] and 
                recent_states[-2] == recent_states[-4] and
                proposed_state == recent_states[-2]):
                
                # Check if these transitions happened in quick succession
                if current_time - self.transition_times.get(recent_states[-4], 0) < 5.0:
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
        """Send command to stop all robot motion immediately."""
        twist = Twist()  # All fields initialize to 0
        self.cmd_vel_publisher.publish(twist)
    
    def publish_state(self):
        """Publish current robot state for other nodes to consume."""
        msg = String()
        msg.data = self.current_state
        self.state_publisher.publish(msg)
    
    def health_check_callback(self):
        """
        Perform periodic health checks and adjust behaviors.
        
        This evaluates system confidence metrics and warns about potential issues.
        """
        # Evaluate overall health
        new_warnings = self.health_monitor.evaluate_health()
        
        # Calculate system confidence
        system_confidence = self.health_monitor.calculate_system_confidence()
        
        # Log new warnings
        for warning in new_warnings:
            self.get_logger().warn(f"Health warning: {warning}")
        
        # Adjust behavior based on health
        if system_confidence < self.health_confidence_threshold:
            self.health_monitor.throttled_log(
                self.get_logger(),
                f"System health degraded: confidence={system_confidence:.2f}, "
                f"warnings={len(self.health_monitor.warnings)}",
                "degraded_health",
                min_interval=3.0,
                level="warn"
            )
            
            # Adjust tracking parameters based on health
            if self.adaptive_parameters_enabled:
                # More conservative tracking with degraded health
                self.lost_ball_timeout *= 0.7
                self.min_retracking_detections += 1
        
        # Publish health status
        health_msg = String()
        health_data = {
            'system_confidence': round(system_confidence, 3),
            'warnings': self.health_monitor.warnings,
            'state': self.current_state
        }
        health_msg.data = json.dumps(health_data)
        self.health_publisher.publish(health_msg)
    
    def publish_diagnostics(self):
        """Publish enhanced diagnostic information."""
        current_time = time.time()
        
        # Get basic diagnostic data
        if self.last_detection_time is not None:
            time_since_detection = current_time - self.last_detection_time
        else:
            time_since_detection = float('inf')

        state_duration = current_time - self.state_start_time
        
        # Create position description
        position_info = {}
        if self.last_position is not None:
            distance = np.linalg.norm(self.last_position[:2])
            direction = math.degrees(math.atan2(self.last_position[1], self.last_position[0]))
            
            position_info = {
                "distance": f"{distance:.2f}m",
                "direction": f"{direction:.1f}°",
                "coordinates": f"({self.last_position[0]:.2f}, {self.last_position[1]:.2f}, {self.last_position[2]:.2f})"
            }
        
        # Build basic diagnostic info
        diagnostic_info = {
            "state": {
                "current": self.current_state,
                "previous": self.previous_state,
                "duration": f"{state_duration:.1f}s"
            },
            "tracking": {
                "reliable": self.tracking_reliable,
                "consecutive_detections": self.consecutive_detections,
                "uncertainty": f"{self.position_uncertainty:.3f}m",
                "time_since_detection": f"{time_since_detection:.2f}s",
                "confidence": getattr(self, 'tracking_confidence', 0.0)
            },
            "ball": {
                "distance": f"{self.ball_distance:.2f}m",
                "is_close": self.is_ball_close,
                "is_stationary": self.is_ball_stationary,
                "position": position_info
            }
        }
        
        # Every 5 seconds, include full diagnostics
        full_diagnostics = False
        if current_time - self.last_full_diagnostic_time > self.full_diagnostic_rate:
            full_diagnostics = True
            self.last_full_diagnostic_time = current_time
            
            # Add motion state info
            if hasattr(self, 'motion_state'):
                diagnostic_info["motion_state"] = {
                    "current": self.motion_state,
                    "previous": getattr(self, 'last_motion_state', "unknown"),
                    "in_transition": getattr(self, 'in_motion_transition', False)
                }
            
            # Add sensor gap information
            if hasattr(self, 'in_sensor_gap') and self.in_sensor_gap:
                gap_duration = 0.0
                if hasattr(self, 'gap_start_time') and self.gap_start_time is not None:
                    gap_duration = current_time - self.gap_start_time
                    
                diagnostic_info["sensor_gap"] = {
                    "active": True,
                    "duration": f"{gap_duration:.2f}s",
                }
            
            # Add uncertainty trend analysis
            if hasattr(self, 'uncertainty_history') and len(self.uncertainty_history.values) >= 3:
                direction, rate = self.uncertainty_history.get_trend(5)
                stability = self.uncertainty_history.get_stability_score()
                
                trend_name = "stable"
                if direction > 0:
                    trend_name = "rising"
                elif direction < 0:
                    trend_name = "falling"
                
                diagnostic_info["uncertainty_trend"] = {
                    "trend": trend_name,
                    "rate": f"{abs(rate):.3f}/s",
                    "stability": f"{stability:.2f}"
                }
            
            # Add system health information
            if hasattr(self, 'health_monitor'):
                health_data = self.health_monitor.get_diagnostic_data()
                diagnostic_info["system_health"] = health_data
        
        # Only include resource usage in full diagnostics
        if full_diagnostics and self.resource_monitoring_enabled:
            try:
                # Add memory usage
                import resource
                memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                diagnostic_info["resources"] = {
                    "memory_kb": memory_usage,
                    "uptime": f"{current_time - self.start_time:.1f}s"
                }
            except ImportError:
                pass
        
        # Publish diagnostic info
        msg = String()
        msg.data = json.dumps(diagnostic_info, cls=NumpyJSONEncoder)
        self.diagnostics_publisher.publish(msg)
        
        # Log summarized diagnostics with reduced frequency
        if full_diagnostics:
            # Create simplified summary
            summary = (
                f"State: {self.current_state} ({state_duration:.1f}s), "
                f"Ball: {self.ball_distance:.2f}m "
                f"({'stationary' if self.is_ball_stationary else 'moving'}), "
                f"Uncertainty: {self.position_uncertainty:.3f}m"
            )
            
            # Add health info if available
            if hasattr(self, 'health_monitor'):
                summary += f", Health: {self.health_monitor.system_confidence:.2f}"
                if self.health_monitor.warnings:
                    warnings_str = ", ".join(self.health_monitor.warnings[:2])
                    if len(self.health_monitor.warnings) > 2:
                        warnings_str += f" +{len(self.health_monitor.warnings) - 2} more"
                    summary += f", Warnings: {warnings_str}"
            
            self.get_logger().info(f"Diagnostic Summary: {summary}")


def main(args=None):
    """Main function to initialize and run the state manager node."""
    rclpy.init(args=args)
    
    # Welcome message
    print("=================================================")
    print("Enhanced Basketball Chaser - State Manager Node")
    print("=================================================")
    print("This node manages the robot's operational states:")
    print("- INITIALIZING: Startup, waiting for ball detection")
    print("- TRACKING: Following the basketball")
    print("- LOST_BALL: Ball not found, waiting for it to reappear")
    print("- STOPPED: Ball is close and stationary")
    print("- SEARCHING: Actively searching for a lost ball")
    print("- RECOVERY: Recovery mode during sensor gaps or high uncertainty")
    print("=================================================")
    
    try:
        # Create node
        node = EnhancedBallChaseStateManager()
        
        # Use MultiThreadedExecutor for better performance
        from rclpy.executors import MultiThreadedExecutor
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