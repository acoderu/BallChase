#!/usr/bin/env python3

"""
Basketball Tracking Robot - System Diagnostic Node
=================================================

This node monitors the overall health and synchronization of the basketball
tracking robot system by collecting and analyzing diagnostic data from all nodes:
- YOLO Detection Node
- LIDAR Detection Node
- State Management Node
- Fusion Node
- PID Controller Node

Features:
- Low resource utilization optimized for Raspberry Pi 5
- State synchronization monitoring
- Detection pipeline integrity checks
- System-wide event correlation
- Throttled console logging with severity indicators
- Periodic diagnostic summaries to file
- End-of-run system status reports
- Improved resource management and thread safety
- Enhanced error handling and recovery
- Configurable thresholds and parameters
- Performance optimizations for resource-constrained environments

Usage:
ros2 run ball_chase diagnostic_node

WHAT IS A DIAGNOSTIC NODE? 🩺
===========================

┌─────────────────────────────────────────────────────────────────────────┐
│                        THE ROBOT DOCTOR                                  │
└─────────────────────────────────────────────────────────────────────────┘

    Think of the diagnostic node as a doctor for your robot system!

    Just like a doctor monitors your:               The diagnostic node monitors:
    ┌───────────────────────┐                     ┌───────────────────────┐
    │ • Heart rate          │                     │ • Node heartbeats     │
    │ • Blood pressure      │                     │ • CPU & memory usage  │
    │ • Temperature         │                     │ • Detection accuracy  │
    │ • Reflexes            │                     │ • Response times      │
    │ • Overall health      │                     │ • System state        │
    └───────────────────────┘                     └───────────────────────┘

    And when something is wrong, the diagnostic node:
    
    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │  DETECTS    │ ---> │  LOGS THE   │ ---> │ CORRELATES  │ ---> │ HELPS WITH  │
    │  PROBLEMS   │      │   ISSUE     │      │   EVENTS    │      │  RECOVERY   │
    └─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘

HOW THE DIAGNOSTIC NODE WORKS 🔍
=============================

┌─────────────────────────────────────────────────────────────────────┐
│                    DIAGNOSTIC NODE ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────┘

                  Robot Nodes Send Status Updates
                              │
                              ▼
     ┌───────────────────────────────────────────────────┐
     │              DIAGNOSTIC NODE                       │
     │                                                   │
     │  ┌─────────────┐     ┌─────────────┐             │
     │  │ STATUS      │     │ PIPELINE    │             │
     │  │ TRACKING    │     │ MONITORING  │             │
     │  └──────┬──────┘     └──────┬──────┘             │
     │         │                   │                    │
     │         └────────┬──────────┘                    │
     │                  │                               │
     │         ┌────────▼─────────┐                     │
     │         │    ISSUE         │                     │
     │         │   DETECTION      │                     │
     │         └────────┬─────────┘                     │
     │                  │                               │
     │       ┌──────────┴──────────┐                    │
     │       │                     │                    │
     │  ┌────▼─────┐         ┌────▼─────┐              │
     │  │ LOGGING  │         │ ALERTING │              │
     │  └──────────┘         └──────────┘              │
     └───────────────────────────────────────────────────┘
                              │
                              ▼
                  Human-Readable Diagnostic Reports

REAL-WORLD EXAMPLES 🌟
===================

┌─────────────────────────────────────────────────────────────────────┐
│                    DIAGNOSTIC SCENARIOS                              │
└─────────────────────────────────────────────────────────────────────┘

  1️⃣ HEARTBEAT MONITORING           2️⃣ SENSOR CONSISTENCY CHECK
  
  ┌─────────────────────────┐      ┌─────────────────────────┐
  │ YOLO Node               │      │ Position from LIDAR:    │
  │ Last Update: 2.3s ago   │      │   (x=1.2, y=0.5, z=0.1) │
  │ Status: ✅ OK           │      │                         │
  │                         │      │ Position from YOLO:     │
  │ LIDAR Node              │      │   (x=3.7, y=0.6, z=0.1) │
  │ Last Update: 30.5s ago  │      │                         │
  │ Status: ❌ MISSING      │      │ ⚠️ INCONSISTENCY!       │
  │                         │      │ Distance: 2.5m          │
  │ Fusion Node             │      │ (Threshold: 1.0m)       │
  │ Last Update: 1.1s ago   │      │                         │
  │ Status: ✅ OK           │      │ → Log warning           │
  └─────────────────────────┘      └─────────────────────────┘
  
  3️⃣ SYSTEM RESOURCE MONITORING     4️⃣ EVENT CORRELATION
  
  ┌─────────────────────────┐      ┌─────────────────────────┐
  │ CPU Usage: 87%          │      │ Recent Events:          │
  │ Memory: 65%             │      │                         │
  │ Temperature: 72°C       │      │ 1. LIDAR node missing   │
  │                         │      │ 2. Position inconsistent│
  │ ⚠️ HIGH CPU USAGE!      │      │ 3. Controller error     │
  │                         │      │                         │
  │ → Adjust detection      │      │ CORRELATION DETECTED!   │
  │   processing rate       │      │ → Possibly related      │
  └─────────────────────────┘      └─────────────────────────┘

EVERYDAY ANALOGY 🏠
================

Think of your robot system like a car with multiple systems (engine, brakes, 
steering, etc.). The diagnostic node is like the car's dashboard and computer 
system that monitors everything and shows warning lights when something needs 
attention.

Just like your car dashboard might show:
- Check engine light
- Low fuel warning
- High temperature alert

The diagnostic node shows warnings about:
- Disconnected sensors
- Inconsistent readings
- High CPU usage
- System errors

It helps you identify and fix problems before they cause bigger issues!
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rcl_interfaces.msg import IntegerRange, FloatingPointRange
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String, Bool, Float32, Float32MultiArray
from geometry_msgs.msg import PointStamped, Twist
import json
import time
import os
import threading
import signal
import sys
import traceback
import atexit
from datetime import datetime
from collections import deque, OrderedDict
import math
import random
import gc
import weakref


class ColorPrinter:
    """Utility for colored console output with improved error handling."""
    
    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'bold': '\033[1m'
    }
    
    # Flag to disable color if terminal doesn't support it
    _color_enabled = True
    
    @classmethod
    def disable_color(cls):
        """Disable colored output."""
        cls._color_enabled = False
    
    @classmethod
    def enable_color(cls):
        """Enable colored output."""
        cls._color_enabled = True
    
    @classmethod
    def print(cls, text, color=None, bold=False):
        """
        Print colored text to console with robust error handling.
        
        Args:
            text: The text to print
            color: Color name (must be in COLORS dict)
            bold: Whether to make text bold
        """
        try:
            # Only apply color formatting if enabled
            if cls._color_enabled and color:
                color_code = ""
                if color and color in cls.COLORS:
                    color_code = cls.COLORS[color]
                if bold:
                    color_code += cls.COLORS['bold']
                    
                print(f"{color_code}{text}{cls.COLORS['reset']}")
            else:
                # Fallback to plain text
                print(text)
        except (IOError, BrokenPipeError):
            # Handle broken pipe (e.g., output redirection closed)
            try:
                # Try to restore stderr/stdout
                sys.stdout = os.fdopen(1, 'w')
                sys.stderr = os.fdopen(2, 'w')
            except Exception:
                pass
        except Exception:
            # Last resort fallback
            try:
                print(text, file=sys.stderr)
            except Exception:
                pass
    
    @classmethod
    def format(cls, text, color=None, bold=False):
        """
        Format text with color codes.
        
        Args:
            text: The text to format
            color: Color name (must be in COLORS dict)
            bold: Whether to make text bold
            
        Returns:
            str: Formatted text with color codes
        """
        try:
            # Only apply color formatting if enabled
            if cls._color_enabled and color:
                color_code = ""
                if color and color in cls.COLORS:
                    color_code = cls.COLORS[color]
                if bold:
                    color_code += cls.COLORS['bold']
                    
                return f"{color_code}{text}{cls.COLORS['reset']}"
            else:
                return text
        except Exception:
            # Fallback to unformatted text if formatting fails
            return text


class EventLogger:
    """
    Efficient event logging with severity levels, throttling, 
    and robust resource management.
    
    IMAGINE THIS: 📝
    ---------------
    Think of EventLogger as a careful note-taker who writes down everything 
    important that happens with your robot system. Just like a ship's captain keeps 
    a logbook recording the journey, weather conditions, and any unusual events,
    EventLogger keeps track of everything happening in your robot.
    
    HOW IT WORKS:
    ------------
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      EVENT LOGGER SYSTEM                             │
    └─────────────────────────────────────────────────────────────────────┘
    
        ┌─────────────────┐            ┌─────────────────┐
        │  EVENT HAPPENS  │            │ SEVERITY LEVELS │
        │  IN THE SYSTEM  │            │                 │
        │                 │            │  INFO           │ → "Battery at 90%"
        │  • Node starts  │            │  WARNING        │ → "Battery below 20%"
        │  • Error occurs │            │  ERROR          │ → "Connection lost"
        │  • Warning      │            │  CRITICAL       │ → "System overheating"
        │  • Status update│            │                 │
        └────────┬────────┘            └────────┬────────┘
                 │                              │
                 └──────────────┬──────────────┘
                                │
                                ▼
                      ┌─────────────────────┐
                      │                     │
                      │   THROTTLE CHECK    │  → Don't log the same warning
                      │                     │    too many times
                      └─────────┬───────────┘
                                │
                                ▼
             ┌─────────────────────────────────────┐
             │                                     │
             │      LOG OUTPUTS                    │
             │                                     │
             │  ┌─────────────┐  ┌─────────────┐   │
             │  │ COLORED     │  │ FILE        │   │
             │  │ CONSOLE     │  │ STORAGE     │   │
             │  │ OUTPUT      │  │             │   │
             │  │             │  │ timestamp | │   │
             │  │ [WARNING]   │  │ severity  | │   │
             │  │ Battery low │  │ message   | │   │
             │  │             │  │ data      | │   │
             │  └─────────────┘  └─────────────┘   │
             └─────────────────────────────────────┘
    
    EVERYDAY ANALOGY:
    ---------------
    It's like your phone keeping track of notifications. Some notifications are 
    just for information (new email), some are warnings (low battery), and some 
    are critical (emergency alerts). Your phone displays these messages with 
    different colors and importance levels, and it doesn't keep showing you the 
    same notification over and over.
    
    EventLogger works the same way - it organizes robot events by importance,
    displays them appropriately, saves them for later review, and makes sure
    you aren't overwhelmed with repeated messages about the same issue.
    """
    
    # Severity levels as class constants
    INFO = 'INFO'
    WARNING = 'WARNING'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'
    
    # Default throttle configuration
    DEFAULT_MAX_THROTTLE_ENTRIES = 500
    DEFAULT_THROTTLE_CLEANUP_INTERVAL = 60  # seconds
    
    # Helper class to wrap floats for WeakValueDictionary
    class TimeValue:
        """Simple wrapper class for timestamp values to allow weak references."""
        def __init__(self, value):
            self.value = value
            
        def __float__(self):
            return float(self.value)
            
        def __lt__(self, other):
            if isinstance(other, EventLogger.TimeValue):
                return self.value < other.value
            return self.value < other
            
        def __gt__(self, other):
            if isinstance(other, EventLogger.TimeValue):
                return self.value > other.value
            return self.value > other
    
    def __init__(self, log_dir="./diagnostic_logs", max_throttle_entries=None,
                 cleanup_interval=None, max_log_size_mb=10):
        """
        Initialize event logger with configurable limits.
        
        Args:
            log_dir: Directory for log files
            max_throttle_entries: Maximum number of throttle entries to keep
            cleanup_interval: Seconds between throttle cleanup runs
            max_log_size_mb: Maximum log file size in MB
        """
        # Configure limits with defaults
        self.max_throttle_entries = max_throttle_entries or self.DEFAULT_MAX_THROTTLE_ENTRIES
        self.cleanup_interval = cleanup_interval or self.DEFAULT_THROTTLE_CLEANUP_INTERVAL
        self.max_log_size_bytes = max_log_size_mb * 1024 * 1024
        
        # Use a weak value dictionary for improved memory management
        # Maps throttle_key -> TimeValue wrapper object
        self.throttle_times = weakref.WeakValueDictionary()
        
        # Track actual throttle entries with bounded size
        self._throttle_keys = OrderedDict()
        
        # Thread safety
        self.lock = threading.RLock()  # Reentrant lock for safety
        
        # Timestamps and counters
        self.start_time = time.time()
        self.last_cleanup_time = self.start_time
        self.current_log_size = 0
        
        # Ensure log directory exists
        self.log_dir = self._setup_log_directory(log_dir)
            
        # Create log file with timestamp
        self.log_file, self.file_handle = self._create_log_file()
        
        # Register cleanup handlers for graceful exit
        self._register_cleanup_handlers()
    
    def _setup_log_directory(self, log_dir):
        """Set up log directory with robust error handling."""
        try:
            os.makedirs(log_dir, exist_ok=True)
            return log_dir
        except (PermissionError, OSError) as e:
            print(f"WARNING: Could not create log directory {log_dir}: {str(e)}")
            try:
                # Try user's home directory as fallback
                home_dir = os.path.expanduser("~")
                fallback_dir = os.path.join(home_dir, "diagnostic_logs")
                os.makedirs(fallback_dir, exist_ok=True)
                print(f"Falling back to: {fallback_dir}")
                return fallback_dir
            except (PermissionError, OSError):
                # Last resort: current directory
                print("Falling back to current directory")
                return "./"
    
    def _create_log_file(self):
        """Create a new log file with error handling."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"diagnostic_{timestamp}.log")
        file_handle = None
        
        try:
            # Create file with automatic closure on exception
            file_handle = open(log_file, 'w')
            file_handle.write(f"# Basketball Robot Diagnostic Log - {timestamp}\n")
            file_handle.write("# timestamp | severity | category | message\n")
            file_handle.write("-" * 80 + "\n")
            file_handle.flush()
            self.current_log_size = file_handle.tell()
            return log_file, file_handle
        except (PermissionError, OSError) as e:
            print(f"WARNING: Could not create log file: {str(e)}")
            # Try to close file if it was partially opened
            if file_handle:
                try:
                    file_handle.close()
                except Exception:
                    pass
            return log_file, None
    
    def _register_cleanup_handlers(self):
        """Register multiple handlers to ensure clean shutdown."""
        # Register atexit handler
        atexit.register(self.close)
        
        # Register signal handlers for clean shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            # Use a separate handler function to avoid reference loops
            def handler(signum, frame, self=self):
                self.close()
                # Re-raise the signal with default handler
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
            
            try:
                original_handler = signal.getsignal(sig)
                signal.signal(sig, handler)
                
                # Store original handler to be able to chain them
                setattr(self, f"_original_{sig}_handler", original_handler)
            except (ValueError, OSError):
                # Signal may not be available on this platform
                pass
    
    def _cleanup_throttle_entries(self, force=False):
        """
        Clean up old throttle entries to prevent unbounded growth.
        
        Args:
            force: Force cleanup regardless of interval
        """
        current_time = time.time()
        
        # Only run cleanup if enough time has passed or forced
        if not force and (current_time - self.last_cleanup_time < self.cleanup_interval):
            return
            
        with self.lock:
            # Update cleanup timestamp
            self.last_cleanup_time = current_time
            
            # Check if we need cleanup
            if len(self._throttle_keys) <= self.max_throttle_entries:
                return
                
            # Calculate how many entries to remove
            excess = len(self._throttle_keys) - self.max_throttle_entries
            
            # Remove oldest entries (first items in OrderedDict)
            for _ in range(min(excess, len(self._throttle_keys))):
                try:
                    self._throttle_keys.popitem(last=False)  # Remove oldest entry
                except KeyError:
                    # Dictionary may have been modified by another thread
                    break
    
    def _check_log_rotation(self):
        """Check if log file needs rotation due to size."""
        if not self.file_handle:
            return False
            
        with self.lock:
            if self.current_log_size > self.max_log_size_bytes:
                # Close current log
                self.file_handle.flush()
                self.file_handle.close()
                
                # Create new log file
                self.log_file, self.file_handle = self._create_log_file()
                return True
                
        return False
    
    def log(self, severity, category, message, data=None, throttle_key=None, throttle_seconds=0):
        """
        Log an event with optional throttling.
        
        Args:
            severity: Severity level (INFO, WARNING, ERROR, CRITICAL)
            category: Event category for grouping
            message: Main log message
            data: Optional structured data for the log
            throttle_key: Key for throttling similar messages
            throttle_seconds: Minimum seconds between similar messages
        """
        # Check throttling early to avoid unnecessary work
        if throttle_key and throttle_seconds > 0:
            current_time = time.time()
            
            with self.lock:                # Get throttle time object (float wrapped in a simple object for WeakValueDictionary)
                throttle_time = self.throttle_times.get(throttle_key)
                
                if throttle_time is not None:
                    # Check if throttling applies
                    if current_time - throttle_time.value < throttle_seconds:
                        return  # Skip logging due to throttling
                
                # Update throttle time - use actual float for timestamp
                self.throttle_times[throttle_key] = self.TimeValue(current_time)
                
                # Also update ordered tracking dictionary
                self._throttle_keys[throttle_key] = current_time
        
        # Format timestamp outside lock
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        # Select colors based on severity and category
        if severity == self.ERROR or severity == self.CRITICAL:
            console_color = 'red'
            console_bold = True
        elif severity == self.WARNING:
            console_color = 'yellow'
            console_bold = False
        else:
            console_color = 'green' if category == 'STATE' else 'cyan'
            console_bold = False
        
        # Format JSON data (if any) outside lock
        json_data = self._format_json_data(data)
        
        # Print to console - do this outside lock to minimize lock time
        console_msg = f"[{timestamp}] {ColorPrinter.format(f'{severity:8s}', console_color, console_bold)} "
        console_msg += f"{ColorPrinter.format(f'[{category}]', 'blue')} {message}"
        ColorPrinter.print(console_msg)
        
        # Write to log file with lock protection
        with self.lock:
            try:
                if self.file_handle:
                    log_line = f"{timestamp} | {severity:8s} | {category:12s} | {message}"
                    if json_data:
                        log_line += f" | {json_data}"
                    log_line += "\n"
                    
                    # Track log size and check rotation before writing
                    self.current_log_size += len(log_line)
                    if self.current_log_size > self.max_log_size_bytes:
                        self._check_log_rotation()
                    
                    # Write to file
                    self.file_handle.write(log_line)
                    
                    # Flush periodically to ensure data is written
                    if severity in (self.ERROR, self.CRITICAL) or random.random() < 0.1:
                        self.file_handle.flush()
            except Exception as e:
                # Fallback to console if file logging fails
                print(f"ERROR: Could not write to log file: {str(e)}")
            
            # Periodically clean up throttle entries
            self._cleanup_throttle_entries()
    
    def _format_json_data(self, data):
        """
        Format data as JSON with careful error handling.
        
        Args:
            data: Data to format as JSON
            
        Returns:
            str: JSON-formatted string or error message
        """
        if not data:
            return ""
            
        try:
            if isinstance(data, dict):
                return json.dumps(data)
            elif isinstance(data, (list, tuple)):
                return json.dumps(data)
            else:
                return str(data)
        except Exception as e:
            return f"Error formatting data: {str(e)}"
    
    def info(self, category, message, data=None, throttle_key=None, throttle_seconds=0):
        """Log info level event."""
        self.log(self.INFO, category, message, data, throttle_key, throttle_seconds)
    
    def warning(self, category, message, data=None, throttle_key=None, throttle_seconds=0):
        """Log warning level event."""
        self.log(self.WARNING, category, message, data, throttle_key, throttle_seconds)
    
    def error(self, category, message, data=None, throttle_key=None, throttle_seconds=0):
        """Log error level event."""
        self.log(self.ERROR, category, message, data, throttle_key, throttle_seconds)
    
    def critical(self, category, message, data=None, throttle_key=None, throttle_seconds=0):
        """Log critical level event."""
        self.log(self.CRITICAL, category, message, data, throttle_key, throttle_seconds)
    
    def write_summary(self, summary_data):
        """Write a structured summary to the log file."""
        with self.lock:
            try:
                if self.file_handle:
                    # Prepare summary text
                    summary_text = "\n" + "=" * 80 + "\n"
                    summary_text += "SYSTEM SUMMARY\n"
                    summary_text += "=" * 80 + "\n"
                    
                    # Write summary data
                    for section, content in summary_data.items():
                        summary_text += f"\n## {section}\n"
                        
                        if isinstance(content, dict):
                            for key, value in content.items():
                                summary_text += f"  {key}: {value}\n"
                        elif isinstance(content, list):
                            for item in content:
                                summary_text += f"  - {item}\n"
                        else:
                            summary_text += f"  {content}\n"
                    
                    summary_text += "\n" + "=" * 80 + "\n"
                    
                    # Update log size and write
                    self.current_log_size += len(summary_text)
                    self.file_handle.write(summary_text)
                    self.file_handle.flush()
            except Exception as e:
                # Log to console if file logging fails
                print(f"ERROR: Could not write summary to log file: {str(e)}")
    
    def close(self):
        """Close the log file cleanly."""
        with self.lock:
            if self.file_handle:
                try:
                    self.file_handle.flush()
                    self.file_handle.close()
                    self.file_handle = None
                except Exception as e:
                    print(f"ERROR: Could not close log file: {str(e)}")
    
    def __del__(self):
        """Clean up resources when object is destroyed."""
        self.close()


class Position:
    """
    Structured class for position data with clear semantics.
    Includes improved validation and error handling.
    """
    
    def __init__(self, x=0.0, y=0.0, z=0.0, confidence=1.0):
        """
        Initialize position with coordinates and confidence.
        
        Args:
            x: X coordinate (meters, forward is positive)
            y: Y coordinate (meters, left is positive)
            z: Z coordinate (meters, up is positive)
            confidence: Detection confidence (0.0-1.0)
        """
        # Validate and convert input values - handle potential invalid inputs
        try:
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
            self.confidence = min(max(float(confidence), 0.0), 1.0)  # Clamp to [0.0, 1.0]
        except (ValueError, TypeError) as e:
            # Fallback to safe values if conversion fails
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0
            self.confidence = 0.0  # Zero confidence for invalid positions
            # Re-raise for caller to handle
            raise ValueError(f"Invalid position data: {str(e)}")
            
        # Check for NaN/Inf values
        if (math.isnan(self.x) or math.isnan(self.y) or math.isnan(self.z) or
            math.isinf(self.x) or math.isinf(self.y) or math.isinf(self.z)):
            raise ValueError("Position contains NaN or Inf values")
    
    @classmethod
    def from_tuple(cls, pos_tuple, confidence=1.0):
        """
        Create position from tuple representation.
        
        Args:
            pos_tuple: (x, y, z) tuple
            confidence: Detection confidence (0.0-1.0)
            
        Returns:
            Position: New position object
            
        Raises:
            ValueError: If input is invalid
        """
        # Validate input
        if pos_tuple is None:
            return cls()
            
        # Handle different tuple sizes
        if isinstance(pos_tuple, (list, tuple)):
            if len(pos_tuple) >= 3:
                try:
                    return cls(
                        x=float(pos_tuple[0]),
                        y=float(pos_tuple[1]),
                        z=float(pos_tuple[2]),
                        confidence=float(confidence)
                    )
                except (ValueError, TypeError):
                    # Try to recover what we can
                    x = 0.0
                    y = 0.0
                    z = 0.0
                    
                    # Extract what we can from the tuple
                    try:
                        if len(pos_tuple) > 0 and pos_tuple[0] is not None:
                            x = float(pos_tuple[0])
                    except (ValueError, TypeError):
                        pass
                        
                    try:
                        if len(pos_tuple) > 1 and pos_tuple[1] is not None:
                            y = float(pos_tuple[1])
                    except (ValueError, TypeError):
                        pass
                        
                    try:
                        if len(pos_tuple) > 2 and pos_tuple[2] is not None:
                            z = float(pos_tuple[2])
                    except (ValueError, TypeError):
                        pass
                    
                    return cls(x, y, z, 0.1)  # Low confidence for partial recovery
            elif len(pos_tuple) == 2:
                # Handle 2D positions
                try:
                    return cls(
                        x=float(pos_tuple[0]),
                        y=float(pos_tuple[1]),
                        z=0.0,
                        confidence=float(confidence)
                    )
                except (ValueError, TypeError):
                    return cls()  # Return default if conversion fails
            else:
                # Not enough dimensions
                return cls(confidence=0.1)  # Low confidence
        else:
            # Not a tuple or list
            return cls(confidence=0.0)  # Zero confidence
    
    def to_tuple(self):
        """
        Convert to tuple representation.
        
        Returns:
            tuple: (x, y, z) tuple
        """
        return (self.x, self.y, self.z)
    
    def distance_to(self, other):
        """
        Calculate Euclidean distance to another position with comprehensive error handling.
        
        Args:
            other: Another Position object
            
        Returns:
            float: Distance in meters
            
        Raises:
            ValueError: If input is invalid or math error occurs
            TypeError: If other is not a Position object
        """
        # Validate input
        if other is None:
            raise TypeError("Cannot calculate distance to None")
            
        if not isinstance(other, Position):
            raise TypeError(f"Expected Position object, got {type(other).__name__}")
            
        try:
            # Calculate 3D Euclidean distance
            dx = self.x - other.x
            dy = self.y - other.y
            dz = self.z - other.z
            
            # Check for NaN/Inf in computation
            if (math.isnan(dx) or math.isnan(dy) or math.isnan(dz) or
                math.isinf(dx) or math.isinf(dy) or math.isinf(dz)):
                raise ValueError("Distance calculation produced NaN or Inf values")
                
            # Calculate distance with overflow protection
            distance = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # Sanity check on result
            if math.isnan(distance) or math.isinf(distance) or distance < 0:
                raise ValueError(f"Invalid distance result: {distance}")
                
            return distance
        except ZeroDivisionError:
            # This shouldn't happen with Euclidean distance, but just in case
            return 0.0
        except (ValueError, OverflowError) as e:
            # Handle other math errors
            raise ValueError(f"Error calculating distance: {str(e)}")
    
    def get_confidence_weighted_position(self, other, min_confidence=0.1):
        """
        Get a new position that is weighted by confidence of two positions.
        
        Args:
            other: Another Position object
            min_confidence: Minimum confidence to consider
            
        Returns:
            Position: New position weighted by confidence
            
        Raises:
            TypeError: If other is not a Position object
        """
        if not isinstance(other, Position):
            raise TypeError(f"Expected Position object, got {type(other).__name__}")
            
        # Ensure confidences are at least min_confidence
        self_conf = max(self.confidence, min_confidence)
        other_conf = max(other.confidence, min_confidence)
        
        # Calculate weight factors
        total_conf = self_conf + other_conf
        if total_conf <= 0:
            # If both confidences are at minimum, return average
            return Position(
                (self.x + other.x) / 2.0,
                (self.y + other.y) / 2.0,
                (self.z + other.z) / 2.0,
                min_confidence
            )
        
        # Weight by confidence
        self_weight = self_conf / total_conf
        other_weight = other_conf / total_conf
        
        # Calculate weighted position
        x = self.x * self_weight + other.x * other_weight
        y = self.y * self_weight + other.y * other_weight
        z = self.z * self_weight + other.z * other_weight
        
        # Calculate combined confidence (higher if positions are close)
        try:
            distance = self.distance_to(other)
            proximity_factor = 1.0 / (1.0 + distance)  # Approaches 1 as distance approaches 0
            combined_confidence = (self_conf + other_conf) / 2.0 * proximity_factor
        except (ValueError, TypeError):
            combined_confidence = min(self_conf, other_conf)
        
        return Position(x, y, z, combined_confidence)
    
    def is_valid(self):
        """
        Check if position is valid (not NaN/Inf).
        
        Returns:
            bool: True if position is valid
        """
        return (not math.isnan(self.x) and not math.isnan(self.y) and not math.isnan(self.z) and
                not math.isinf(self.x) and not math.isinf(self.y) and not math.isinf(self.z))
    
    def __eq__(self, other):
        """
        Check if two positions are equal.
        
        Args:
            other: Another Position object
            
        Returns:
            bool: True if positions are equal
        """
        if not isinstance(other, Position):
            return False
            
        # Compare with small epsilon to handle floating point errors
        epsilon = 1e-6
        return (abs(self.x - other.x) < epsilon and
                abs(self.y - other.y) < epsilon and
                abs(self.z - other.z) < epsilon)
    
    def __str__(self):
        """String representation of position."""
        return f"Position(x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}, conf={self.confidence:.2f})"
    
    def __repr__(self):
        """Detailed string representation of position."""
        return f"Position({self.x}, {self.y}, {self.z}, {self.confidence})"


class TimedRingBuffer:
    """
    Efficient fixed-size buffer with time-based indexing and improved edge case handling.
    Thread-safe and optimized for resource-constrained environments.
    
    IMAGINE THIS: 🔄
    ---------------
    Think of TimedRingBuffer like a small circular notebook where you write down
    the last 10 things that happened, along with when they happened. Once the notebook
    is full, you start writing over the oldest entries. This way, you always have
    the most recent information, without using up endless paper.
    
    HOW IT WORKS:
    ------------
    
    ┌─────────────────────────────────────────────────────────────────────┐
    │                     TIMED RING BUFFER                                │
    └─────────────────────────────────────────────────────────────────────┘
    
                                NEW DATA
                                    │
                                    ▼
                            ┌───────────────┐
                            │ TIME-STAMPED  │
                            │ ENTRY         │
                            └───────┬───────┘
                                    │
                                    ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                                                                     │
    │   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐    │
    │   │ 1 │   │ 2 │   │ 3 │   │ 4 │   │ 5 │   │ 6 │   │ 7 │   │ 8 │    │
    │   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───┘    │
    │     ▲                                                               │
    │     │                                                               │
    │    Next                                                             │
    │  Position                FIXED-SIZE RING BUFFER                     │
    │     │                   (When full, overwrites                      │
    │     │                     oldest entries)                           │
    │     ▼                                                               │
    │   ┌───┐   ┌───┐                                                     │
    │   │ 9 │   │10 │                                                     │
    │   └───┘   └───┘                                                     │
    │                                                                     │
    └─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ RETRIEVAL FUNCTIONS  │
                         │                      │
                         │ • Latest entries     │
                         │ • Entries by time    │
                         │ • All entries        │
                         └──────────────────────┘
    
    The "ring" in the name refers to how it wraps around back to the beginning
    when it reaches the end. If your buffer size is 10 and you add an 11th item,
    it replaces the first item, and so on.
    
    EVERYDAY ANALOGY:
    ---------------
    It's similar to your phone's call history, which might store your last 100 calls.
    Once you make the 101st call, your oldest call drops off the list. Each call
    is stored with a timestamp, and you can look through this history to find:
    
    - Your most recent calls
    - Calls from a specific time period
    - All calls in chronological order
    
    The TimedRingBuffer does this same job but for robot sensor data, events,
    or any information that needs to be tracked with timestamps.
    """
    
    def __init__(self, max_size=10):
        """
        Initialize with fixed buffer size.
        
        Args:
            max_size: Maximum number of items to store (must be positive)
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        
        # Pre-allocate arrays for better memory efficiency
        self.data = [None] * max_size
        self.timestamps = [0.0] * max_size
        self.max_size = max_size
        self.next_index = 0
        self.is_full = False
        self.count = 0
        self.lock = threading.RLock()  # Thread safety for concurrent access
    
    def add(self, value, timestamp=None):
        """
        Add a value to the buffer with timestamp.
        
        Args:
            value: The data to store
            timestamp: Optional timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        with self.lock:    
            # Store data
            self.data[self.next_index] = value
            self.timestamps[self.next_index] = timestamp
            
            # Update indexing
            old_index = self.next_index
            self.next_index = (self.next_index + 1) % self.max_size
            
            # Critical transition: detect the exact moment we fill the buffer
            if not self.is_full:
                self.count += 1
                if self.count >= self.max_size:
                    self.is_full = True
                    self.count = self.max_size
            
            # Return the index where data was stored
            return old_index
    
    def get_latest(self, count=1):
        """
        Get the most recent entries.
        
        Args:
            count: Number of recent entries to retrieve
            
        Returns:
            list: List of (timestamp, value) tuples, most recent first
        """
        with self.lock:
            if self.count == 0:
                return []
                
            # Limit count to actual data available
            count = min(count, self.count)
            
            result = []
            
            if self.is_full:
                # Buffer is full - need to wrap around from current position
                start_idx = self.next_index - count
                if start_idx < 0:
                    start_idx += self.max_size  # Wrap around
                
                # Collect items in correct order
                for i in range(count):
                    idx = (start_idx + i) % self.max_size
                    if self.data[idx] is not None:  # Safety check
                        result.append((self.timestamps[idx], self.data[idx]))
            else:
                # Buffer not full yet - just take the last count items
                start_idx = max(0, self.count - count)
                for i in range(start_idx, self.count):
                    if self.data[i] is not None:  # Safety check
                        result.append((self.timestamps[i], self.data[i]))
            
            # Sort by timestamp, most recent first (stable sort)
            return sorted(result, key=lambda x: x[0], reverse=True)
    
    def get_all(self):
        """
        Get all values with timestamps, ordered by timestamp.
        
        Returns:
            list: List of (timestamp, value) tuples ordered by timestamp
        """
        with self.lock:
            if self.count == 0:
                return []
                
            result = []
            
            if self.is_full:
                # Need to collect all valid items
                for i in range(self.max_size):
                    if self.data[i] is not None:  # Safety check
                        result.append((self.timestamps[i], self.data[i]))
            else:
                # Just get the valid items
                for i in range(self.count):
                    if self.data[i] is not None:  # Safety check
                        result.append((self.timestamps[i], self.data[i]))
            
            # Sort by timestamp (stable sort)
            return sorted(result, key=lambda x: x[0])
    
    def get_within_timeframe(self, start_time, end_time):
        """
        Get values within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            list: List of (timestamp, value) tuples within the specified time range
        """
        # Ensure start_time <= end_time
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        
        with self.lock:
            result = []
            
            # Determine which entries to check
            if self.is_full:
                # Check all items
                indices = range(self.max_size)
            else:
                # Check just the valid items
                indices = range(self.count)
            
            # Filter entries within timeframe
            for i in indices:
                if (self.data[i] is not None and  # Safety check
                    start_time <= self.timestamps[i] <= end_time):
                    result.append((self.timestamps[i], self.data[i]))
            
            # Sort by timestamp
            return sorted(result, key=lambda x: x[0])
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            # Reset counters but keep pre-allocated arrays
            self.next_index = 0
            self.is_full = False
            self.count = 0
            
            # Optional: explicitly clear data for GC
            for i in range(self.max_size):
                self.data[i] = None
                self.timestamps[i] = 0.0


class LockManager:
    """
    Lock manager class to enforce consistent lock acquisition order
    and prevent deadlocks.
    """
    
    def __init__(self):
        """Initialize lock manager with predefined lock order."""
        # Create locks in a specific order
        self.state_lock = threading.RLock()
        self.diagnostic_lock = threading.RLock()
        self.position_lock = threading.RLock()
        self.statistics_lock = threading.RLock()
        
        # Store locks in order for tracking
        self._lock_order = {
            'state_lock': 0,
            'diagnostic_lock': 1,
            'position_lock': 2,
            'statistics_lock': 3
        }
        
        # Track currently held locks by this thread
        self._held_locks = threading.local()
        self._held_locks.locks = set()
        
        # Track lock acquisition with timeouts
        self.lock_timeouts = {
            'state_lock': 1.0,      # 1 second timeout
            'diagnostic_lock': 1.0,
            'position_lock': 1.0,
            'statistics_lock': 1.0
        }
    
    def acquire(self, lock_name, timeout=None):
        """
        Acquire a lock with proper ordering and deadlock detection.
        
        Args:
            lock_name: Name of the lock to acquire
            timeout: Maximum time to wait for lock acquisition
            
        Returns:
            bool: True if lock was acquired, False if timeout
            
        Raises:
            ValueError: If lock_name is invalid
            RuntimeError: If acquiring lock would create a deadlock
        """
        # Validate lock name
        if lock_name not in self._lock_order:
            raise ValueError(f"Invalid lock name: {lock_name}")
            
        # Get lock object
        lock = getattr(self, lock_name)
        
        # Check lock ordering
        thread_locks = getattr(self._held_locks, 'locks', set())
        for held_lock in thread_locks:
            if self._lock_order[held_lock] > self._lock_order[lock_name]:
                raise RuntimeError(
                    f"Lock ordering violation: Cannot acquire {lock_name} while holding {held_lock}"
                )
        
        # Use provided timeout or default
        actual_timeout = timeout if timeout is not None else self.lock_timeouts.get(lock_name, 1.0)
        
        # Try to acquire the lock with timeout
        acquired = lock.acquire(timeout=actual_timeout)
        if acquired:
            # Track this lock as held
            thread_locks.add(lock_name)
            self._held_locks.locks = thread_locks
        
        return acquired
    
    def release(self, lock_name):
        """
        Release a lock safely.
        
        Args:
            lock_name: Name of the lock to release
            
        Raises:
            ValueError: If lock_name is invalid
            RuntimeError: If lock wasn't held
        """
        # Validate lock name
        if lock_name not in self._lock_order:
            raise ValueError(f"Invalid lock name: {lock_name}")
            
        # Get lock object
        lock = getattr(self, lock_name)
        
        # Check if lock is actually held
        thread_locks = getattr(self._held_locks, 'locks', set())
        if lock_name not in thread_locks:
            raise RuntimeError(f"Attempted to release unheld lock: {lock_name}")
        
        # Release the lock
        lock.release()
        
        # Update held locks
        thread_locks.remove(lock_name)
        self._held_locks.locks = thread_locks
    
    def with_lock(self, lock_name, timeout=None):
        """
        Context manager for safely using locks.
        
        Args:
            lock_name: Name of lock to acquire
            timeout: Maximum time to wait for lock acquisition
            
        Returns:
            Context manager that acquires and releases the lock
            
        Usage:
            with lock_manager.with_lock('state_lock'):
                # Critical section
        """
        class LockContext:
            def __init__(self, manager, name, timeout):
                self.manager = manager
                self.name = name
                self.timeout = timeout
                self.acquired = False
                
            def __enter__(self):
                self.acquired = self.manager.acquire(self.name, self.timeout)
                return self.acquired
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.acquired:
                    self.manager.release(self.name)
                return False  # Don't suppress exceptions
        
        return LockContext(self, lock_name, timeout)


class RetryHandler:
    """
    Utility for handling retries of operations that may fail transiently.
    """
    
    def __init__(self, max_retries=3, base_delay=0.1, max_delay=1.0):
        """
        Initialize retry handler.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def execute(self, operation, *args, retry_on=Exception, **kwargs):
        """
        Execute an operation with retries on failure.
        
        Args:
            operation: Function to execute
            *args: Arguments for the function
            retry_on: Exception type to retry on (default: any Exception)
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the operation
            
        Raises:
            The last exception if all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except retry_on as e:
                last_error = e
                
                # Don't sleep on the last attempt
                if attempt < self.max_retries:
                    # Exponential backoff with jitter
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    jitter = random.uniform(0, 0.1 * delay)
                    time.sleep(delay + jitter)
        
        # If we get here, all retries failed
        raise last_error


class CircuitBreaker:
    """
    Circuit breaker pattern implementation to prevent repeated calls to failing components.
    """
    
    # Circuit states
    CLOSED = 'closed'      # Normal operation - requests flow through
    OPEN = 'open'          # Failing - requests immediately return with error
    HALF_OPEN = 'half_open'  # Testing - allowing limited requests to check recovery
    
    def __init__(self, name, failure_threshold=5, reset_timeout=60):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name (for logging)
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds before attempting reset (half-open)
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        
        # State tracking
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.last_success_time = time.time()
        
        # Thread safety
        self.lock = threading.RLock()
    
    def execute(self, operation, *args, **kwargs):
        """
        Execute an operation through the circuit breaker.
        
        Args:
            operation: Function to execute
            *args: Arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the operation
            
        Raises:
            CircuitOpenError: If circuit is open (too many prior failures)
            The underlying exception if operation fails
        """
        with self.lock:
            # Check if circuit is open
            if self.state == self.OPEN:
                # Check if reset timeout has expired
                if time.time() - self.last_failure_time >= self.reset_timeout:
                    # Transition to half-open to test service
                    self.state = self.HALF_OPEN
                else:
                    # Circuit still open - fast fail
                    raise RuntimeError(f"Circuit '{self.name}' is open due to previous failures")
        
        try:
            # Execute the operation
            result = operation(*args, **kwargs)
            
            # Operation succeeded - update state
            with self.lock:
                self.last_success_time = time.time()
                
                # If we were in half-open state, reset the circuit
                if self.state in (self.HALF_OPEN, self.OPEN):
                    self.state = self.CLOSED
                    self.failure_count = 0
            
            return result
            
        except Exception as e:
            # Operation failed - update state
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Check if we need to open the circuit
                if (self.state == self.CLOSED and self.failure_count >= self.failure_threshold) or \
                   (self.state == self.HALF_OPEN):
                    self.state = self.OPEN
            
            # Re-raise the original exception
            raise e


class SystemDiagnosticNode(Node):
    """
    Diagnostic node for tracking overall system health and synchronization.
    Optimized for low resource usage on Raspberry Pi 5.
    
    Improvements:
    - Enhanced thread safety with proper lock ordering using LockManager
    - Reduced memory usage with more efficient data structures
    - Improved error handling and recovery with RetryHandler and CircuitBreaker
    - Configurable thresholds via ROS parameters with dynamic updates
    - Better resource management with improved cleanup
    - Self-monitoring capabilities and adaptive behavior
    - State machine for monitoring system integrity
    
    IMAGINE THIS: 🏥👨‍⚕️
    ---------------
    Think of the SystemDiagnosticNode as a doctor who's constantly monitoring a patient
    (your robot system). The doctor regularly checks vital signs (CPU, memory), looks
    for symptoms of problems (missing heartbeats, inconsistent sensor data), and
    keeps detailed medical records (logs). If something goes wrong, the doctor can
    diagnose the issue and suggest a treatment plan.
    
    HOW IT WORKS (BEGINNER'S GUIDE):
    -------------------------------
    1. The node SUBSCRIBES (listens) to messages from all other robot nodes
       - This is like the doctor putting a stethoscope on different parts of the body
       - Each message tells the doctor that a part of the system is still working
    
    2. The node runs regular CHECK-UPS (timer callbacks)
       - Check if all nodes are still sending messages (heartbeats)
       - Check if different sensors agree with each other (consistency)
       - Check if the system has enough resources (CPU, memory)
    
    3. When problems are found, the node:
       - LOGS the issue (writes it down in the medical chart)
       - ANALYZES related issues (are multiple symptoms connected?)
       - Sends ALERTS if needed (like a doctor calling for a nurse)
    
    4. The node maintains HISTORY of system behavior
       - Keeps track of past issues (like medical history)
       - Looks for patterns in problems (like recurring symptoms)
       - Helps diagnose complex issues by connecting related events
    
    This diagnostic node helps keep your robot healthy and helps you understand
    what's wrong when problems occur - just like how a good doctor helps keep
    you healthy and diagnoses issues when you're sick!
    """
    
    # System states for the diagnostic node itself
    STATE_INITIALIZING = 'initializing'
    STATE_RUNNING = 'running'
    STATE_DEGRADED = 'degraded'
    STATE_ERROR = 'error'
    
    def __init__(self):
        """Initialize the diagnostic node with improved configuration and error handling."""
        super().__init__('system_diagnostics')
        
        # Initialize node state
        self.node_state = self.STATE_INITIALIZING
        
        # Create lock manager for thread safety
        self.locks = LockManager()
        
        # Create retry handler for operations that may fail transiently
        self.retry_handler = RetryHandler(
            max_retries=3,
            base_delay=0.1,
            max_delay=1.0
        )
        
        # Create circuit breakers for critical external components
        self.circuit_breakers = {
            'state_tracking': CircuitBreaker('state_tracking', failure_threshold=5, reset_timeout=30),
            'diagnostics': CircuitBreaker('diagnostics', failure_threshold=5, reset_timeout=30),
            'position_tracking': CircuitBreaker('position_tracking', failure_threshold=5, reset_timeout=30)
        }
        
        # Configuration parameters - declare with constraints and descriptions
        self._declare_parameters()
        
        # Load parameters with validation
        self._load_parameters()
        
        # Parameter subscriber for dynamic updates
        self.add_on_set_parameters_callback(self._on_parameter_change)
        
        # Initialize event logger
        self._init_logger()
        
        # Initialize state trackers
        self._init_state_tracking()
        
        # Initialize statistics tracking
        self._init_statistics()
        
        # Create QoS profiles for subscriptions
        self._create_qos_profiles()
        
        # Create subscriptions with error handling
        self._setup_subscriptions()
        
        # Create timers with staggered starts
        self._setup_timers()
        
        # Log successful initialization
        self.logger.info("INIT", "Initialized system diagnostic node with configured intervals: "
                        f"diagnostics={self.diagnostic_interval}s, "
                        f"summary={self.summary_interval}s")
        
        # Register shutdown callback for final summary
        self._add_on_shutdown_callback(self._shutdown_callback)
        
        # Update own resource usage
        self.last_resource_update = time.time()
        self.last_memory_check = time.time()
        
        # Periodic garbage collection configuration
        if hasattr(gc, 'enable'):
            gc.enable()
        self.gc_counter = 0
        self.adaptive_gc = True  # Enable adaptive garbage collection
        
        # Mark as running
        self.node_state = self.STATE_RUNNING
    
    def _declare_parameters(self):
        """
        Declare parameters with defaults, constraints, and descriptions.
        Improves configuration flexibility and documentation.
        
        Returns:
            None
        """
        # Helper function for declaring parameters with bounds
        def declare_bounded_param(name, default, description, lower_bound=None, upper_bound=None):
            """
            Declare a parameter with bounds.
            
            Args:
                name: Parameter name
                default: Default value
                description: Parameter description
                lower_bound: Lower bound for the parameter value
                upper_bound: Upper bound for the parameter value
                
            Returns:
                None
            """
            param_descriptor = ParameterDescriptor(description=description)
            
            if lower_bound is not None and upper_bound is not None:
                if isinstance(default, int):
                    param_descriptor.integer_range = [
                        IntegerRange(from_value=lower_bound, to_value=upper_bound)
                    ]
                else:
                    param_descriptor.floating_point_range = [
                        FloatingPointRange(from_value=lower_bound, to_value=upper_bound)
                    ]
            
            self.declare_parameter(name, default, param_descriptor)
        
        # Timing parameters
        declare_bounded_param('diagnostic_interval', 5.0, 
                             'Seconds between system diagnostic runs', 1.0, 60.0)
        declare_bounded_param('summary_interval', 60.0, 
                             'Seconds between summary logging', 10.0, 300.0)
        declare_bounded_param('state_check_interval', 1.0, 
                             'Seconds between state synchronization checks', 0.1, 10.0)
        declare_bounded_param('heartbeat_interval', 10.0, 
                             'Seconds between heartbeat logs', 1.0, 60.0)
        
        # Storage parameters
        declare_bounded_param('state_buffer_size', 20, 
                             'Size of state change buffer', 5, 100)
        declare_bounded_param('error_buffer_size', 10, 
                             'Size of error event buffer', 5, 50)
        declare_bounded_param('system_buffer_size', 30, 
                             'Size of system event buffer', 10, 100)
        
        # Feature flags
        self.declare_parameter(
            'sync_check_enabled', 
            True, 
            ParameterDescriptor(description='Enable state synchronization checks')
        )
        self.declare_parameter(
            'pipeline_check_enabled', 
            True, 
            ParameterDescriptor(description='Enable detection pipeline integrity checks')
        )
        self.declare_parameter(
            'resource_check_enabled', 
            True, 
            ParameterDescriptor(description='Enable system resource monitoring')
        )
        
        # Logging parameters
        self.declare_parameter(
            'log_directory', 
            './diagnostic_logs', 
            ParameterDescriptor(description='Directory for diagnostic logs')
        )
        declare_bounded_param('max_log_size_mb', 10, 
                             'Maximum log file size in megabytes', 1, 100)
        
        # Threshold parameters
        declare_bounded_param('position_difference_threshold', 0.5, 
                             'Position difference threshold (meters)', 0.1, 2.0)
        declare_bounded_param('detection_rate_threshold', 2.0, 
                             'Minimum acceptable detection rate (Hz)', 0.5, 10.0)
        declare_bounded_param('high_cpu_threshold', 85.0, 
                             'High CPU usage threshold (%)', 50.0, 99.0)
    
    def _load_parameters(self):
        """
        Load and validate parameters with robust error handling.
        Ensures valid configuration even with incorrect parameter values.
        
        Returns:
            dict: Dictionary of loaded parameters
        """
        # Initialize default values in case parameter loading fails
        default_values = {
            'diagnostic_interval': 5.0,
            'summary_interval': 60.0,
            'state_check_interval': 1.0,
            'heartbeat_interval': 10.0,
            'state_buffer_size': 20,
            'error_buffer_size': 10,
            'system_buffer_size': 30,
            'sync_check_enabled': True,
            'pipeline_check_enabled': True,
            'resource_check_enabled': True,
            'log_directory': './diagnostic_logs',
            'max_log_size_mb': 10,
            'position_difference_threshold': 0.5,
            'detection_rate_threshold': 2.0,
            'high_cpu_threshold': 85.0
        }
        
        # Dict to track loaded parameters
        loaded_params = {}
        
        try:
            # Timing parameters with validation
            self.diagnostic_interval = max(1.0, self.get_parameter('diagnostic_interval').value)
            loaded_params['diagnostic_interval'] = self.diagnostic_interval
            
            self.summary_interval = max(10.0, self.get_parameter('summary_interval').value)
            loaded_params['summary_interval'] = self.summary_interval
            
            self.state_check_interval = max(0.1, self.get_parameter('state_check_interval').value)
            loaded_params['state_check_interval'] = self.state_check_interval
            
            self.heartbeat_interval = max(1.0, self.get_parameter('heartbeat_interval').value)
            loaded_params['heartbeat_interval'] = self.heartbeat_interval
            
            # Storage parameters with validation
            self.state_buffer_size = max(5, self.get_parameter('state_buffer_size').value)
            loaded_params['state_buffer_size'] = self.state_buffer_size
            
            self.error_buffer_size = max(5, self.get_parameter('error_buffer_size').value)
            loaded_params['error_buffer_size'] = self.error_buffer_size
            
            self.system_buffer_size = max(10, self.get_parameter('system_buffer_size').value)
            loaded_params['system_buffer_size'] = self.system_buffer_size
            
            # Feature flags
            self.sync_check_enabled = bool(self.get_parameter('sync_check_enabled').value)
            loaded_params['sync_check_enabled'] = self.sync_check_enabled
            
            self.pipeline_check_enabled = bool(self.get_parameter('pipeline_check_enabled').value)
            loaded_params['pipeline_check_enabled'] = self.pipeline_check_enabled
            
            self.resource_check_enabled = bool(self.get_parameter('resource_check_enabled').value)
            loaded_params['resource_check_enabled'] = self.resource_check_enabled
            
            # Log configuration
            self.log_directory = str(self.get_parameter('log_directory').value)
            loaded_params['log_directory'] = self.log_directory
            
            self.max_log_size_mb = max(1, self.get_parameter('max_log_size_mb').value)
            loaded_params['max_log_size_mb'] = self.max_log_size_mb
            
            # Threshold parameters with validation
            self.position_difference_threshold = max(0.1, 
                self.get_parameter('position_difference_threshold').value)
            loaded_params['position_difference_threshold'] = self.position_difference_threshold
            
            self.detection_rate_threshold = max(0.5, 
                self.get_parameter('detection_rate_threshold').value)
            loaded_params['detection_rate_threshold'] = self.detection_rate_threshold
            
            self.high_cpu_threshold = min(99.0, max(50.0,
                self.get_parameter('high_cpu_threshold').value))
            loaded_params['high_cpu_threshold'] = self.high_cpu_threshold
            
        except Exception as e:
            # Log error and use default values
            print(f"ERROR: Failed to load parameters: {e}")
            print("Using default values for missing parameters")
            
            # Apply default values for any parameters that failed to load
            for param, default in default_values.items():
                if param not in loaded_params:
                    setattr(self, param, default)
                    loaded_params[param] = default
        
        # Store all parameters as a dict for summary reporting
        self.param_values = loaded_params
        return loaded_params
    
    def _init_logger(self):
        """Initialize event logger with configured parameters."""
        # Create logger with configured parameters
        self.logger = EventLogger(
            log_dir=self.log_directory,
            max_log_size_mb=self.max_log_size_mb
        )
        self.logger.info("INIT", "System Diagnostic Node starting")
        
        # Log parameters for reference
        self.logger.info("CONFIG", f"Loaded {len(self.param_values)} parameters", self.param_values)
    
    def _on_parameter_change(self, params):
        """
        Handle dynamic parameter updates.
        
        Args:
            params: List of changed parameters
            
        Returns:
            SetParametersResult: Result of parameter update
        """
        from rclpy.parameter import SetParametersResult
        
        result = SetParametersResult(successful=True)
        
        try:
            # Track which parameters actually changed
            changed_params = {}
            
            for param in params:
                name = param.name
                
                # Get the new value
                if param.type_ == Parameter.Type.DOUBLE:
                    new_value = param.value
                elif param.type_ == Parameter.Type.INTEGER:
                    new_value = param.value
                elif param.type_ == Parameter.Type.BOOL:
                    new_value = param.value
                elif param.type_ == Parameter.Type.STRING:
                    new_value = param.value
                else:
                    continue  # Skip other types
                
                # Handle specific parameters
                if name == 'diagnostic_interval':
                    # Validate
                    if new_value < 1.0:
                        result.successful = False
                        result.reason = "diagnostic_interval must be >= 1.0"
                        break
                    # Update if changed
                    if new_value != self.diagnostic_interval:
                        self.diagnostic_interval = new_value
                        changed_params[name] = new_value
                        # Update timer if it exists
                        if 'diagnostic' in self.timers:
                            self.timers['diagnostic'].cancel()
                            self.timers['diagnostic'] = self.create_timer(
                                self.diagnostic_interval, 
                                self._run_system_diagnostics
                            )
                
                elif name == 'summary_interval':
                    # Validate
                    if new_value < 10.0:
                        result.successful = False
                        result.reason = "summary_interval must be >= 10.0"
                        break
                    # Update if changed
                    if new_value != self.summary_interval:
                        self.summary_interval = new_value
                        changed_params[name] = new_value
                        # Update timer if it exists
                        if 'summary' in self.timers:
                            self.timers['summary'].cancel()
                            self.timers['summary'] = self.create_timer(
                                self.summary_interval, 
                                self._write_periodic_summary
                            )
                
                elif name == 'state_check_interval':
                    # Validate
                    if new_value < 0.1:
                        result.successful = False
                        result.reason = "state_check_interval must be >= 0.1"
                        break
                    # Update if changed
                    if new_value != self.state_check_interval:
                        self.state_check_interval = new_value
                        changed_params[name] = new_value
                        # Update timer if it exists
                        if 'sync' in self.timers:
                            self.timers['sync'].cancel()
                            self.timers['sync'] = self.create_timer(
                                self.state_check_interval, 
                                self._check_state_synchronization
                            )
                
                elif name == 'position_difference_threshold':
                    # Validate
                    if new_value <= 0.0:
                        result.successful = False
                        result.reason = "position_difference_threshold must be > 0.0"
                        break
                    # Update if changed
                    if new_value != self.position_difference_threshold:
                        self.position_difference_threshold = new_value
                        changed_params[name] = new_value
                
                elif name == 'detection_rate_threshold':
                    # Validate
                    if new_value < 0.1:
                        result.successful = False
                        result.reason = "detection_rate_threshold must be >= 0.1"
                        break
                    # Update if changed
                    if new_value != self.detection_rate_threshold:
                        self.detection_rate_threshold = new_value
                        changed_params[name] = new_value
                
                elif name == 'high_cpu_threshold':
                    # Validate
                    if new_value < 50.0 or new_value > 99.0:
                        result.successful = False
                        result.reason = "high_cpu_threshold must be between 50.0 and 99.0"
                        break
                    # Update if changed
                    if new_value != self.high_cpu_threshold:
                        self.high_cpu_threshold = new_value
                        changed_params[name] = new_value
                
                # Feature flags
                elif name == 'sync_check_enabled':
                    if new_value != self.sync_check_enabled:
                        self.sync_check_enabled = new_value
                        changed_params[name] = new_value
                        # Update timer if needed
                        if new_value and 'sync' not in self.timers:
                            self.timers['sync'] = self.create_timer(
                                self.state_check_interval, 
                                self._check_state_synchronization
                            )
                        elif not new_value and 'sync' in self.timers:
                            self.timers['sync'].cancel()
                            del self.timers['sync']
                
                elif name == 'pipeline_check_enabled':
                    if new_value != self.pipeline_check_enabled:
                        self.pipeline_check_enabled = new_value
                        changed_params[name] = new_value
                
                elif name == 'resource_check_enabled':
                    if new_value != self.resource_check_enabled:
                        self.resource_check_enabled = new_value
                        changed_params[name] = new_value
            
            # Log parameter changes if successful
            if result.successful and changed_params:
                self.logger.info("CONFIG", 
                               f"Parameters updated: {changed_params}",
                               data=changed_params)
                
                # Update param_values dictionary
                for k, v in changed_params.items():
                    self.param_values[k] = v
            
        except Exception as e:
            result.successful = False
            result.reason = f"Error updating parameters: {str(e)}"
            self.logger.error("CONFIG", f"Parameter update failed: {str(e)}")
        
        return result
    
    def _init_state_tracking(self):
        """Initialize state tracking with thread-safe access and data validation."""
        # State tracking data structures
        with self.locks.with_lock('state_lock'):
            self.node_states = {}  # Maps node_name -> state_dict
            self.node_heartbeats = {}  # Maps node_name -> ROS time message
            self.node_status = {}  # Maps node_name -> health status
            self.node_recovery_attempts = {}  # Maps node_name -> recovery count
        
        with self.locks.with_lock('diagnostic_lock'):
            self.node_diagnostics = {}  # Maps node_name -> diagnostic_dict
        
        with self.locks.with_lock('position_lock'):
            self.position_trackers = {}  # Maps source -> TimedRingBuffer of positions
        
        # Initialize event buffers with configured sizes
        self.state_changes = TimedRingBuffer(max_size=self.state_buffer_size)
        self.error_events = TimedRingBuffer(max_size=self.error_buffer_size)
        self.system_events = TimedRingBuffer(max_size=self.system_buffer_size)
        
        # Track motion status
        self.robot_stopped = True
        
        # Event correlation mapping
        self.event_correlations = {}  # Maps event_id -> related_events
        
        # Node failure tracking for recovery
        self.node_failures = {
            'lidar': {'count': 0, 'last_failure': 0, 'recovery_attempted': False},
            'yolo': {'count': 0, 'last_failure': 0, 'recovery_attempted': False},
            'fusion': {'count': 0, 'last_failure': 0, 'recovery_attempted': False},
            'state_manager': {'count': 0, 'last_failure': 0, 'recovery_attempted': False},
            'pid': {'count': 0, 'last_failure': 0, 'recovery_attempted': False}
        }
        
        # Validation counters for data integrity
        self.validation_stats = {
            'invalid_states': 0,
            'invalid_positions': 0,
            'invalid_diagnostics': 0,
            'recovered_errors': 0
        }
    
    def _init_statistics(self):
        """Initialize statistics tracking."""
        # Statistics tracking
        with self.locks.with_lock('statistics_lock'):
            self.statistics = {
                'start_time': time.time(),
                'events_processed': 0,
                'errors_detected': 0,
                'warnings_detected': 0,
                'state_transitions': 0,
                'sync_issues': 0,
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'node_uptime': 0.0,
                'last_gc_time': 0.0
            }
    
    def _create_qos_profiles(self):
        """Create QoS profiles for various topics with clear semantics."""        # Create QoS profile for critical state information
        self.state_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,  # Changed to match publisher's setting
            history=HistoryPolicy.KEEP_LAST,
            depth=5  # Keep last 5 messages
        )
        
        # Create QoS profile for diagnostic information
        self.diagnostic_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,  # No need to get old diagnostic data
            history=HistoryPolicy.KEEP_LAST,
            depth=1  # Only need the latest diagnostic data
        )
        
        # Create QoS profile for position data
        self.position_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Can miss some position updates
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=3  # Keep a few recent positions
        )
        
        # Create QoS profile for velocity commands
        self.velocity_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,  # Match publisher's setting
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
    
    def _setup_subscriptions(self):
        """Setup subscriptions to all node topics with appropriate QoS profiles and error handling."""
        try:
            # State management node state - use state QoS for reliability
            self.create_subscription(
                String,
                '/robot/state',
                self.state_callback,
                qos_profile=self.state_qos
            )
            
            # Node diagnostics - use diagnostic QoS
            diagnostic_topics = [
                ('/basketball/lidar/diagnostics', 'lidar'),
                ('/basketball/yolo/diagnostics', 'yolo'),
                ('/basketball/fusion/diagnostics', 'fusion'),
                ('/robot/diagnostics', 'state_manager'),
                ('/pid/diagnostics', 'pid')
            ]
            
            self.diagnostic_subscriptions = []
            for topic, source in diagnostic_topics:
                # Create closure that captures the source correctly
                callback = lambda msg, src=source: self.diagnostic_callback(msg, src)
                
                sub = self.create_subscription(
                    String,
                    topic,
                    callback,
                    qos_profile=self.diagnostic_qos
                )
                self.diagnostic_subscriptions.append(sub)
            
            # Position tracking (for detection pipeline analysis) - use position QoS
            position_topics = [
                ('/basketball/yolo/position', 'yolo'),
                ('/basketball/lidar/position', 'lidar'),
                ('/basketball/fused/position', 'fusion')
            ]
            
            self.position_subscriptions = []
            for topic, source in position_topics:
                # Create closure that captures the source correctly
                callback = lambda msg, src=source: self.position_callback(msg, src)
                
                sub = self.create_subscription(
                    PointStamped,
                    topic,
                    callback,
                    qos_profile=self.position_qos
                )
                self.position_subscriptions.append(sub)
              # Monitor velocity commands - need reliability for commands
            self.velocity_subscription = self.create_subscription(
                Twist,
                '/controller/cmd_vel',
                self.velocity_callback,
                qos_profile=self.velocity_qos  # Use velocity-specific QoS
            )
            
            self.logger.info("INIT", f"Established {len(diagnostic_topics)} diagnostic subscriptions, "
                            f"{len(position_topics)} position subscriptions")
        except Exception as e:
            self.logger.critical("INIT", f"Failed to set up subscriptions: {str(e)}")
            # Log traceback for debugging
            traceback_str = traceback.format_exc()
            self.logger.error("TRACE", f"Subscription setup failed: {traceback_str}")
    
    def _setup_timers(self):
        """
        Setup timers for periodic diagnostics with resource-sensitive scheduling and thread safety.
        Uses staggered starts to prevent CPU spikes and handles concurrent execution safely.
        
        Returns:
            dict: Dictionary of created timers
        """
        # Store timers for management
        self._timer_dict = {}
        
        # Track which timers are already running to prevent concurrent execution
        self.timer_running = {
            'sync': False,
            'diagnostic': False,
            'summary': False,
            'heartbeat': False,
            'resource': False,
            'gc': False
        }
        
        # Mutex for timer execution
        self.timer_mutex = threading.RLock()
        
        try:
            # Create wrapper for each timer callback to prevent concurrent execution
            def create_safe_callback(timer_name, original_callback):
                def safe_callback():
                    # Skip if already running
                    with self.timer_mutex:
                        if self.timer_running.get(timer_name, False):
                            self.logger.debug("TIMER", 
                                           f"Timer {timer_name} skipped - already running",
                                           throttle_key=f"timer_skip_{timer_name}", 
                                           throttle_seconds=30)
                            return
                        # Mark as running
                        self.timer_running[timer_name] = True
                    
                    try:
                        # Call the original callback
                        original_callback()
                    except Exception as e:
                        # Log any unhandled exceptions from the callback
                        self.logger.error("TIMER", 
                                        f"Error in timer {timer_name}: {str(e)}",
                                        throttle_key=f"timer_error_{timer_name}", 
                                        throttle_seconds=10)
                        # Log traceback for debugging
                        traceback_str = traceback.format_exc()
                        self.logger.error("TRACE", 
                                        f"Timer error trace: {traceback_str}",
                                        throttle_key="timer_trace", 
                                        throttle_seconds=300)
                    finally:
                        # Always mark as not running when done
                        with self.timer_mutex:
                            self.timer_running[timer_name] = False
                
                return safe_callback
            
            # State synchronization check timer
            if self.sync_check_enabled:
                self._timer_dict['sync'] = self.create_timer(
                    self.state_check_interval, 
                    create_safe_callback('sync', self._check_state_synchronization)
                )
            
            # Full system diagnostic check - stagger startup to avoid CPU spikes
            self._timer_dict['diagnostic'] = self.create_timer(
                self.diagnostic_interval + 2.0,  # Add 2 seconds to stagger startup
                create_safe_callback('diagnostic', self._run_system_diagnostics)
            )
            
            # Summary log timer - stagger startup further
            self._timer_dict['summary'] = self.create_timer(
                self.summary_interval + 5.0,  # Add 5 seconds to stagger startup 
                create_safe_callback('summary', self._write_periodic_summary)
            )
            
            # Heartbeat timer
            self._timer_dict['heartbeat'] = self.create_timer(
                self.heartbeat_interval + 1.0,  # Add 1 second to stagger startup
                create_safe_callback('heartbeat', self._log_system_heartbeat)
            )
            
            # Resource monitoring timer - runs last to enable adaptive behavior
            self._timer_dict['resource'] = self.create_timer(
                5.0,  # Check resources every 5 seconds
                create_safe_callback('resource', self._monitor_own_resources)
            )
            
            # Garbage collection timer
            self._timer_dict['gc'] = self.create_timer(
                30.0,  # Run garbage collection every 30 seconds
                create_safe_callback('gc', self._run_garbage_collection)
            )
            
            self.logger.info("INIT", 
                           f"Timers established with staggered intervals: "
                           f"diagnostic={self.diagnostic_interval + 2.0:.1f}s, "
                           f"summary={self.summary_interval + 5.0:.1f}s, "
                           f"heartbeat={self.heartbeat_interval + 1.0:.1f}s",
                           throttle_key="timers_setup", 
                           throttle_seconds=60)
            
            return self._timer_dict
        except Exception as e:
            self.logger.critical("INIT", f"Failed to set up timers: {str(e)}")
            # Log traceback for debugging
            traceback_str = traceback.format_exc()
            self.logger.error("TRACE", f"Timer setup failed: {traceback_str}")
            
            # Return what we have so far
            return self._timer_dict
    
    def _add_on_shutdown_callback(self, callback):
        """
        Register a callback for node shutdown with improved reliability.
        Uses weak references to prevent reference cycles.
        
        Args:
            callback: Function to call on shutdown
        """
        # Store original destroy_node method
        original_destroy = self.destroy_node
        
        # Create a weakref to self to avoid reference cycles
        weak_self = weakref.ref(self)
        
        # Define new destroy_node method that calls callback first
        def destroy_with_callback():
            try:
                # Get strong reference
                strong_self = weak_self()
                if strong_self is not None:
                    # Call the callback
                    callback()
            except Exception as e:
                print(f"Error in shutdown callback: {e}")
                # Log traceback for debugging
                if weak_self() is not None:
                    traceback_str = traceback.format_exc()
                    weak_self().get_logger().error(f"Shutdown callback failed: {traceback_str}")
            finally:
                # Always call original method
                original_destroy()
        
        # Replace the destroy_node method
        self.destroy_node = destroy_with_callback
    
    def _shutdown_callback(self):
        """Generate final summary when node is shutting down."""
        self.logger.info("SHUTDOWN", "Generating final system summary...")
        self._generate_final_summary()
        
        # Ensure all resources are properly released
        self._cleanup_resources()
    
    def _cleanup_resources(self):
        """Clean up all resources to prevent leaks."""
        # Cancel all timers
        for timer_name, timer in self.timers.items():
            try:
                timer.cancel()
            except Exception:
                pass
        
        # Clear all buffers
        try:
            self.state_changes.clear()
            self.error_events.clear()
            self.system_events.clear()
        except Exception:
            pass
        
        # Run garbage collection
        try:
            if hasattr(gc, 'collect'):
                gc.collect()
        except Exception:
            pass
    
    def _run_garbage_collection(self):
        """
        Adaptive garbage collection based on memory pressure and node activity.
        Avoids memory leaks in long-running sessions while minimizing latency impact.
        """
        try:
            # Check current time and conditions
            current_time = time.time()
            
            # Access statistics safely
            with self.locks.with_lock('statistics_lock'):
                last_gc_time = self.statistics['last_gc_time']
                uptime = current_time - self.statistics['start_time']
                memory_usage = self.statistics.get('memory_usage', 0.0)
                
                # Determine if GC should run based on conditions
                should_run_gc = False
                gc_reason = ""
                
                # Time-based collection (but adaptive)
                time_since_last_gc = current_time - last_gc_time
                
                # Adapt GC frequency based on memory pressure and uptime
                if memory_usage > 3000:  # High memory pressure
                    # Run more frequently under high memory
                    gc_interval = 15.0  # 15 seconds
                    should_run_gc = time_since_last_gc >= gc_interval
                    gc_reason = "high memory pressure"
                elif uptime < 300:  # First 5 minutes
                    # More frequent during startup
                    gc_interval = 30.0  # 30 seconds
                    should_run_gc = time_since_last_gc >= gc_interval
                    gc_reason = "startup phase"
                else:
                    # Normal operation - less frequent
                    gc_interval = 60.0  # 60 seconds
                    should_run_gc = time_since_last_gc >= gc_interval
                    gc_reason = "periodic maintenance"
                
                # Check memory increase since last check
                if current_time - self.last_memory_check > 10.0:
                    if hasattr(self, 'last_memory_value'):
                        memory_increase = memory_usage - self.last_memory_value
                        # Significant memory increase triggers collection
                        if memory_increase > 100:  # More than 100MB increase
                            should_run_gc = True
                            gc_reason = f"memory increase of {memory_increase:.1f}MB"
                    # Update memory tracking
                    self.last_memory_value = memory_usage
                    self.last_memory_check = current_time
                
                # Increment counter for generation-based collection
                self.gc_counter += 1
                if self.gc_counter >= 20:  # Every 20 cycles (regardless of time)
                    should_run_gc = True
                    gc_reason = "generation cycle"
                    self.gc_counter = 0
                
                # Only run if determined necessary and we have gc.collect
                if should_run_gc and hasattr(gc, 'collect'):
                    # Run garbage collection
                    collected = gc.collect()
                    self.statistics['last_gc_time'] = current_time
                    
                    # Log collection results conditionally
                    if collected > 0:
                        # Only log if objects were actually collected
                        self.logger.info("RESOURCE", 
                                      f"Garbage collection: {collected} objects collected (reason: {gc_reason})",
                                      throttle_key="gc_run", throttle_seconds=60)
                        
                        # Track memory after collection
                        if time_since_last_gc > 0:
                            # Calculate collection rate (objects/second)
                            collection_rate = collected / time_since_last_gc
                            # If collection rate is high, we might need more frequent GC
                            if collection_rate > 100 and self.adaptive_gc:
                                # Adjust next GC interval
                                self.logger.info("RESOURCE", 
                                               f"High collection rate detected: {collection_rate:.1f} objects/second",
                                               throttle_key="high_collection_rate", throttle_seconds=300)
        
        except MemoryError:
            # Critical memory error - force immediate collection
            if hasattr(gc, 'collect'):
                gc.collect()
            self.logger.critical("RESOURCE", "Memory error triggered emergency garbage collection",
                               throttle_key="emergency_gc", throttle_seconds=10)
        except Exception as e:
            self.logger.error("RESOURCE", f"Error during garbage collection: {str(e)}",
                             throttle_key="gc_error", throttle_seconds=30)
            
            # Log traceback for debugging
            traceback_str = traceback.format_exc()
            self.logger.error("TRACE", f"Garbage collection error: {traceback_str}",
                            throttle_key="gc_trace", throttle_seconds=300)
    
    def state_callback(self, msg):
        """
        Process state messages from the state manager.
        Tracks the primary system state with improved error handling and circuit breaker.
        
        Args:
            msg: String message containing the system state
        """
        # Use circuit breaker pattern to prevent repeated failures
        try:
            # Process state through circuit breaker
            self.circuit_breakers['state_tracking'].execute(
                self._process_state_message, msg
            )
        except Exception as e:
            if isinstance(e, RuntimeError) and "Circuit 'state_tracking' is open" in str(e):
                # Circuit is open - log once
                self.logger.error("CIRCUIT", 
                                "State tracking circuit breaker open - skipping processing",
                                throttle_key="state_circuit_open", 
                                throttle_seconds=30)
            else:
                # Other error
                self.logger.error("STATE", 
                                f"Unhandled error in state callback: {str(e)}",
                                throttle_key="state_callback_error", 
                                throttle_seconds=5)
                # Log traceback for debugging critical errors
                traceback_str = traceback.format_exc()
                self.logger.error("TRACE", 
                                f"State callback error: {traceback_str}",
                                throttle_key="state_callback_trace", 
                                throttle_seconds=30)
            
            # Update node state if we're experiencing persistent errors
            if self.node_state == self.STATE_RUNNING:
                self.node_state = self.STATE_DEGRADED
    
    def _process_state_message(self, msg):
        """
        Process state message with improved validation and error handling.
        Extracted from state_callback for better organization and circuit breaking.
        
        Args:
            msg: String message containing the system state
            
        Raises:
            ValueError: If state data is invalid
        """
        # Extract and validate state data
        if not hasattr(msg, 'data'):
            raise ValueError("Message has no 'data' attribute")
            
        new_state = msg.data
        
        # Validate state data
        if not isinstance(new_state, str):
            # Update validation stats
            self.validation_stats['invalid_states'] += 1
            
            # Log warning
            self.logger.warning("STATE", 
                            f"Invalid state data type: {type(new_state).__name__}",
                            throttle_key="invalid_state_type", 
                            throttle_seconds=10)
            
            # Try to recover
            if isinstance(new_state, (int, float, bool)):
                new_state = str(new_state)  # Convert to string
                self.logger.info("RECOVERY", 
                               f"Converted non-string state to string: {new_state}",
                               throttle_key="state_convert", 
                               throttle_seconds=30)
                self.validation_stats['recovered_errors'] += 1
            else:
                raise ValueError(f"Invalid state data type: {type(new_state).__name__}")
        
        # Use the lock manager for thread safety
        success = False
        try:
            # Try to acquire the lock with timeout
            success = self.locks.acquire('state_lock', timeout=1.0)
            if not success:
                self.logger.warning("LOCKS", 
                                  "Failed to acquire state lock in time - state update skipped",
                                  throttle_key="state_lock_timeout", 
                                  throttle_seconds=5)
                return
                
            # Get old state
            old_state = self.node_states.get('system', {}).get('state')
            
            # Update heartbeat
            self.node_heartbeats['state_manager'] = self.get_clock().now().to_msg()
            
            # Check for state change
            if old_state != new_state:
                if old_state is not None:  # Don't log the first state
                    # Prepare log message before adding to buffer (more efficient)
                    log_msg = f"System state change: {old_state} → {new_state}"
                    self.logger.info("STATE", log_msg)
                    
                    # Create state change event
                    event = {
                        'node': 'system',
                        'old_state': old_state,
                        'new_state': new_state,
                        'timestamp': time.time(),
                        'event_id': f"state_change_{int(time.time())}"
                    }
                    
                    # Record the state change
                    self.state_changes.add(event)
                    
                    # Update statistics safely
                    success2 = False
                    try:
                        success2 = self.locks.acquire('statistics_lock', timeout=0.5)
                        if success2:
                            self.statistics['state_transitions'] += 1
                    finally:
                        if success2:
                            self.locks.release('statistics_lock')
                
                # Update stored state
                if 'system' not in self.node_states:
                    self.node_states['system'] = {}
                    
                self.node_states['system'] = {
                    'state': new_state,
                    'previous': old_state,
                    'timestamp': time.time()
                }
                
                # Store for potential event correlation
                if old_state is not None:
                    event_id = f"state_change_{int(time.time())}"
                    self.event_correlations[event_id] = {
                        'type': 'state_change',
                        'time': time.time(),
                        'related_events': [],
                        'details': {
                            'node': 'system',
                            'old_state': old_state,
                            'new_state': new_state
                        }
                    }
                
        finally:
            # Always release the lock if acquired
            if success:
                self.locks.release('state_lock')
        
        # Check state synchronization immediately if enabled
        if self.sync_check_enabled:
            self._check_state_synchronization()

    def diagnostic_callback(self, msg, source):
        """
        Process diagnostic messages from individual nodes.
        Uses lazy parsing for better performance.
        
        Args:
            msg: Diagnostic message (JSON string)
            source: Source node identifier
        """
        try:
            # Update heartbeat with thread safety
            with self.locks.with_lock('state_lock'):
                self.node_heartbeats[source] = self.get_clock().now().to_msg()
            
            # Fast validation check - avoid parsing invalid data
            if not msg.data or not isinstance(msg.data, str) or not msg.data.strip():
                self.logger.error("PARSING", f"Empty or invalid diagnostic data from {source}", 
                                throttle_key=f"invalid_diag_{source}", throttle_seconds=10)
                return
            
            # Quick check for JSON format before full parsing
            first_char = msg.data.strip()[0]
            if first_char != '{':
                self.logger.error("PARSING", f"Invalid JSON format from {source}: does not start with '{{'", 
                                throttle_key=f"invalid_json_{source}", throttle_seconds=10)
                return
                
            # Parse diagnostic data - use partial parsing for efficiency
            try:
                # Parse JSON data
                data = json.loads(msg.data)
                
                # Extract key information
                node_name = data.get('node', source)
                status = data.get('status', 'unknown')
                timestamp = data.get('timestamp', time.time())
                
                # Determine if we need to process the full message
                process_full = (status in ['error', 'critical', 'warning'] or 
                               'state' in data)
                
                # Prepare minimal diagnostic data to save memory
                minimal_data = {
                    'node': node_name,
                    'status': status,
                    'timestamp': timestamp
                }
                      # Store diagnostic data with thread safety
                with self.locks.with_lock('diagnostic_lock'):
                    self.node_diagnostics[node_name] = {
                        'timestamp': time.time(),
                        'status': status,
                        # Only store full data if process_full is True
                        'data': data if process_full else minimal_data
                    }
                
                # Update node state if included
                self._update_node_state(node_name, data)
                
                # Check for errors or warnings and update statistics
                self._process_diagnostic_status(node_name, status, data)
                
                # Look for specific issues in the diagnostics data only if needed
                if process_full:
                    self._check_node_specific_diagnostics(node_name, data)
                  # Update statistics safely
                with self.locks.with_lock('statistics_lock'):
                    self.statistics['events_processed'] += 1
                
            except json.JSONDecodeError as e:
                # Detailed error reporting for JSON parsing issues
                self.logger.error("PARSING", 
                                f"JSON parse error from {source}: {str(e)}", 
                                throttle_key=f"parse_error_{source}", throttle_seconds=10)
                
        except Exception as e:
            self.logger.error("PROCESSING", 
                            f"Error processing {source} diagnostics: {str(e)}",
                            throttle_key=f"diag_error_{source}", 
                            throttle_seconds=10)
            # Log traceback for debugging critical errors
            traceback_str = traceback.format_exc()
            self.logger.error("TRACE", f"Diagnostic callback error: {traceback_str}",
                             throttle_key="diag_callback_trace", throttle_seconds=30)
    
    def _update_node_state(self, node_name, data):
        """
        Update node state information from diagnostic data.
        Extracted from diagnostic_callback for better organization.
        
        Args:
            node_name: Name of the node
            data: Diagnostic data dictionary
        """
        try:
            # Skip if no state data
            if 'state' not in data:
                return
                
            current_state = data.get('state', {})
            # Handle state field properly based on type
            if isinstance(current_state, dict):
                state_name = current_state.get('current', current_state.get('state', 'unknown'))
                prev_state = current_state.get('previous', None)
            else:
                # Handle string state representation
                state_name = str(current_state)
                with self.state_lock:
                    prev_state = self.node_states.get(node_name, {}).get('state')
            
            # Thread-safe state operations            with self.locks.with_lock('state_lock'):
                # Check for state change
                old_state = self.node_states.get(node_name, {}).get('state')
                if old_state != state_name and old_state is not None:
                    # Log state change - prepare message first
                    change_msg = f"Node {node_name} state change: {old_state} → {state_name}"
                    self.logger.info("STATE", change_msg)
                    
                    # Record the state change
                    self.state_changes.add({
                        'node': node_name,
                        'old_state': old_state,
                        'new_state': state_name
                    })
                      # Update statistics safely
                    with self.locks.with_lock('statistics_lock'):
                        self.statistics['state_transitions'] += 1
                
                # Update node state
                if node_name not in self.node_states:
                    self.node_states[node_name] = {}
                
                self.node_states[node_name] = {
                    'state': state_name,
                    'previous': prev_state,
                    'timestamp': time.time()
                }
        except Exception as e:
            self.logger.error("STATE", f"Error updating node state for {node_name}: {str(e)}",
                             throttle_key=f"update_state_error_{node_name}", throttle_seconds=10)
    
    def _process_diagnostic_status(self, node_name, status, data):
        """
        Process diagnostic status and update statistics.
        Extracted from diagnostic_callback for better organization.
        
        Args:
            node_name: Name of the node
            status: Status string ('error', 'warning', etc.)
            data: Diagnostic data dictionary
        """
        try:
            # Check for errors or warnings
            if status in ['error', 'critical']:
                # Prepare error data before logging
                error_message = data.get('message', 'No message provided')
                error_details = {
                    'node': node_name,
                    'status': status,
                    'message': error_message
                }
                
                # Log error and add to buffer
                self.logger.error("NODE", f"{node_name} reported ERROR status: {error_message}",
                                throttle_key=f"error_{node_name}", throttle_seconds=5)
                
                # Track error events
                self.error_events.add({
                    'node': node_name,
                    'status': status,
                    'message': error_message,
                    # Include only essential details to save memory
                    'details': {
                        'time': data.get('timestamp', time.time()),
                        'health': data.get('health', {}).get('overall', 0.0) 
                                if isinstance(data.get('health'), dict) else 0.0
                    }
                })
                  # Update statistics safely
                with self.locks.with_lock('statistics_lock'):
                    self.statistics['errors_detected'] += 1
                
            elif status == 'warning':
                self.logger.warning("NODE", 
                                  f"{node_name} reported WARNING status: {data.get('message', 'No details')}",
                                  throttle_key=f"warning_{node_name}", throttle_seconds=5)
                  # Update statistics safely
                with self.locks.with_lock('statistics_lock'):
                    self.statistics['warnings_detected'] += 1
        except Exception as e:
            self.logger.error("STATUS", f"Error processing diagnostic status for {node_name}: {str(e)}",
                             throttle_key=f"status_error_{node_name}", throttle_seconds=10)
    
    def position_callback(self, msg, source):
        """
        Track position detections from different nodes with improved validation and error handling.
        Uses structured Position class and circuit breaker for improved reliability.
        
        Args:
            msg: PointStamped message
            source: Source node identifier ('yolo', 'lidar', 'fusion')
        """
        # Use circuit breaker pattern to prevent repeated failures
        try:
            # Process position through circuit breaker
            self.circuit_breakers['position_tracking'].execute(
                self._process_position_message, msg, source
            )
        except Exception as e:
            if isinstance(e, RuntimeError) and "Circuit 'position_tracking' is open" in str(e):
                # Circuit is open - log once
                self.logger.error("CIRCUIT", 
                                f"Position tracking circuit breaker open for {source} - skipping processing",
                                throttle_key=f"position_circuit_open_{source}", 
                                throttle_seconds=30)
            else:
                # Other error
                self.logger.error("POSITION", 
                                f"Unhandled error in position callback for {source}: {str(e)}",
                                throttle_key=f"pos_error_{source}", 
                                throttle_seconds=10)
                # Log traceback for debugging critical errors
                if "division by zero" in str(e) or "float division" in str(e):
                    traceback_str = traceback.format_exc()
                    self.logger.error("TRACE", 
                                    f"Position callback error: {traceback_str}",
                                    throttle_key="pos_callback_trace", 
                                    throttle_seconds=30)
                
            # Update node failures count
            if source in self.node_failures:
                self.node_failures[source]['count'] += 1
                self.node_failures[source]['last_failure'] = time.time()
    
    def _process_position_message(self, msg, source):
        """
        Process position message with validation and error handling.
        Extracted from position_callback for better organization and circuit breaking.
        
        Args:
            msg: PointStamped message
            source: Source node identifier
            
        Raises:
            ValueError: If position data is invalid
        """
        # Extract and validate position
        if not hasattr(msg, 'point'):
            raise ValueError(f"Position message from {source} has no 'point' attribute")
        
        # Use retry handler for position extraction which might fail
        def extract_position():
            return Position(
                getattr(msg.point, 'x', 0.0),
                getattr(msg.point, 'y', 0.0),
                getattr(msg.point, 'z', 0.0)
            )
        
        try:
            pos = self.retry_handler.execute(
                extract_position,
                retry_on=(AttributeError, ValueError, TypeError)
            )
        except Exception as e:
            # Update validation stats
            self.validation_stats['invalid_positions'] += 1
            
            # Log error and try recovery
            self.logger.error("POSITION", 
                            f"Invalid position data from {source}: {str(e)}",
                            throttle_key=f"invalid_pos_{source}", 
                            throttle_seconds=10)
            
            # Try to recover with default values
            pos = Position(0.0, 0.0, 0.0, confidence=0.1)  # Low confidence for recovered positions
            self.validation_stats['recovered_errors'] += 1
        
        # Validate position values for NaN/infinity
        if (math.isnan(pos.x) or math.isnan(pos.y) or math.isnan(pos.z) or
            math.isinf(pos.x) or math.isinf(pos.y) or math.isinf(pos.z)):
            self.validation_stats['invalid_positions'] += 1
            self.logger.warning("POSITION", 
                              f"Position from {source} contains NaN/Inf values: {pos}",
                              throttle_key=f"nan_pos_{source}", 
                              throttle_seconds=10)
            
            # Fix NaN/Inf values
            pos.x = 0.0 if math.isnan(pos.x) or math.isinf(pos.x) else pos.x
            pos.y = 0.0 if math.isnan(pos.y) or math.isinf(pos.y) else pos.y
            pos.z = 0.0 if math.isnan(pos.z) or math.isinf(pos.z) else pos.z
            pos.confidence = 0.1  # Low confidence for recovered positions
            self.validation_stats['recovered_errors'] += 1
        
        # Extract timestamp carefully
        timestamp = time.time()
        if (hasattr(msg, 'header') and 
            hasattr(msg.header, 'stamp') and 
            hasattr(msg.header.stamp, 'sec')):
            try:
                ros_time = msg.header.stamp.sec + (msg.header.stamp.nanosec / 1e9)
                # If ROS time is reasonable (not too far from system time), use it
                if abs(ros_time - timestamp) < 10.0:  # Within 10 seconds
                    timestamp = ros_time
            except (AttributeError, TypeError) as e:
                self.logger.warning("POSITION", 
                                  f"Error extracting timestamp from {source} position: {str(e)}",
                                  throttle_key=f"timestamp_error_{source}", 
                                  throttle_seconds=30)
        
        # Store position safely - critically important to follow lock acquisition order:
        # First state_lock, then position_lock if needed
        state_lock_acquired = False
        position_lock_acquired = False
        
        try:
            # First acquire state_lock (earlier in acquisition order)
            state_lock_acquired = self.locks.acquire('state_lock', timeout=0.5)
            if not state_lock_acquired:
                self.logger.warning("LOCKS", 
                                  f"Failed to acquire state lock for {source} heartbeat - continuing with position update",
                                  throttle_key=f"state_lock_timeout_{source}", 
                                  throttle_seconds=5)
            else:
                # Update heartbeat while we have state_lock
                self.node_heartbeats[source] = self.get_clock().now().to_msg()
                
                # Update node failure tracking (successful message resets failure count)
                if source in self.node_failures:
                    self.node_failures[source]['count'] = 0
                    self.node_failures[source]['recovery_attempted'] = False
            
            # Now try to acquire position_lock
            position_lock_acquired = self.locks.acquire('position_lock', timeout=0.5)
            if not position_lock_acquired:
                self.logger.warning("LOCKS", 
                                  f"Failed to acquire position lock for {source} - position update skipped",
                                  throttle_key=f"pos_lock_timeout_{source}", 
                                  throttle_seconds=5)
                return
            
            # Initialize position tracker if needed
            if source not in self.position_trackers:
                self.position_trackers[source] = TimedRingBuffer(max_size=5)
            
            # Store position
            self.position_trackers[source].add(pos, timestamp)
            
        finally:
            # Always release locks in reverse order of acquisition
            if position_lock_acquired:
                self.locks.release('position_lock')
            if state_lock_acquired:
                self.locks.release('state_lock')
    
    def velocity_callback(self, msg):
        """
        Track velocity commands with improved zero-checking.
        
        Args:
            msg: Twist message
        """
        try:
            # Extract velocities with safe defaults if attributes are missing
            linear_x = getattr(msg.linear, 'x', 0.0)
            linear_y = getattr(msg.linear, 'y', 0.0)
            angular_z = getattr(msg.angular, 'z', 0.0)
            
            # Use epsilon for near-zero comparisons
            epsilon = 1e-5
            is_stopped = (abs(linear_x) < epsilon and 
                         abs(linear_y) < epsilon and 
                         abs(angular_z) < epsilon)
            
            # Check for robot state change
            if is_stopped:
                # Robot is stopped
                if not hasattr(self, 'robot_stopped') or not self.robot_stopped:
                    self.robot_stopped = True
                    self.logger.info("MOTION", "Robot stopped (zero velocity command)")
                    
                    # Add to system events
                    self.system_events.add({
                        'type': 'motion',
                        'event': 'stop',
                        'values': [linear_x, linear_y, angular_z]
                    })
            else:
                # Robot is moving
                if not hasattr(self, 'robot_stopped') or self.robot_stopped:
                    self.robot_stopped = False
                    self.logger.info("MOTION", 
                        f"Robot moving: linear=({linear_x:.2f}, {linear_y:.2f}), angular={angular_z:.2f}")
                    
                    # Add to system events
                    self.system_events.add({
                        'type': 'motion',
                        'event': 'move',
                        'values': [linear_x, linear_y, angular_z]
                    })
                      # Update heartbeat for PID controller
            with self.locks.with_lock('state_lock'):
                self.node_heartbeats['pid'] = self.get_clock().now().to_msg()
            
        except Exception as e:
            self.logger.error("VELOCITY", f"Error processing velocity command: {str(e)}",
                             throttle_key="vel_error", throttle_seconds=10)

    def _check_state_synchronization(self):
        """
        Check if all nodes agree on the current system state.
        Reports discrepancies between state manager and other nodes.
        """
        try:
            # Get the main system state with thread safety
            system_state = None
            transition_time = 0
            
            with self.locks.with_lock('state_lock'):
                system_state_data = self.node_states.get('system', {})
                system_state = system_state_data.get('state')
                transition_time = system_state_data.get('timestamp', 0)
                
                if not system_state:
                    return  # No system state available yet
                
                # For recent state changes, allow more time for sync
                current_time = time.time()
                is_recent_change = (transition_time > 0 and 
                                  current_time - transition_time < 2.0)
                
                # Nodes that should sync with system state
                observed_states = {}
                
                # Check state nodes
                for node, state_data in self.node_states.items():
                    if node == 'system':
                        continue
                        
                    # Get node's reported state
                    node_state = state_data.get('state', 'unknown')
                    observed_states[node] = node_state
                    
                    # Check for major discrepancies with configurable grace period
                    node_state_time = state_data.get('timestamp', 0)
                    grace_period = 2.0 if is_recent_change else 1.0
                    
                    # Skip if node hasn't had time to sync yet
                    if node_state_time > 0 and current_time - node_state_time < grace_period:
                        continue
                    
                    # Check if states are compatible
                    if not self._states_compatible(system_state, node_state):
                        # Log discrepancy
                        self.logger.warning("SYNC", 
                            f"State mismatch: System={system_state}, {node}={node_state}", 
                            throttle_key=f"{node}_state_mismatch", throttle_seconds=2)
                        
                        # Record the sync issue
                        self.system_events.add({
                            'type': 'sync',
                            'nodes': ['system', node],
                            'states': [system_state, node_state]
                        })
                        
                        # Update statistics
                        with self.statistics_lock:
                            self.statistics['sync_issues'] += 1
        except Exception as e:
            self.logger.error("SYNC", f"Error checking state synchronization: {str(e)}",
                             throttle_key="sync_check_error", throttle_seconds=10)
    
    def _states_compatible(self, state1, state2):
        """
        Check if two states are compatible (allowing for naming differences).
        Improved with more comprehensive state mapping and resilience to case variations.
        
        Args:
            state1: First state name
            state2: Second state name
            
        Returns:
            bool: True if states are compatible
        """
        # Direct match
        if state1 == state2:
            return True
            
        # Handle None/empty cases
        if not state1 or not state2:
            return False
            
        # Normalize states - lowercase and strip spaces
        try:
            state1_norm = state1.lower().strip()
            state2_norm = state2.lower().strip()
        except AttributeError:
            # Handle non-string states
            return False
        
        # Comprehensive state compatibility mapping
        compatibility_map = {
            # General states
            'initializing': ['init', 'starting', 'initialization', 'bootup', 'setup'],
            'tracking': ['track', 'following', 'chase', 'pursuit', 'seeking'],
            'lost_ball': ['lost', 'searching', 'no_ball', 'ball_lost', 'not_found'],
            'stopped': ['stop', 'halt', 'idle', 'stationary', 'waiting', 'paused'],
            'recovery': ['recover', 'recovering', 'correction', 'adjusting'],
            'searching': ['search', 'seeking', 'hunt', 'looking', 'scanning'],
            'error': ['fault', 'failure', 'problem', 'issue', 'crashed'],
            
            # Motion states
            'stationary': ['stopped', 'idle', 'halt', 'not_moving', 'static'],
            'slow_movement': ['slow', 'creeping', 'inching', 'careful_movement'],
            'fast_movement': ['fast', 'rapid', 'quick', 'swift', 'high_speed'],
            'turning': ['rotate', 'pivoting', 'spinning', 'rotation'],
        }
        
        # Check if state2 is compatible with state1
        for base_state, compatible_states in compatibility_map.items():
            if state1_norm == base_state or state1_norm in compatible_states:
                if state2_norm == base_state or state2_norm in compatible_states:
                    return True
        
        # Allow certain specific state combinations
        allowed_combinations = [
            ('searching', 'lost_ball'),       # These states often coincide
            ('recovery', 'stopped'),          # Recovery often involves stopping
            ('initializing', 'stopped'),      # Initialization often starts stopped
            ('tracking', 'slow_movement'),    # Tracking often involves movement
            ('error', 'stopped'),             # Errors often lead to stopping
            ('calibrating', 'stationary'),    # Calibration usually happens when stationary
        ]
        
        for s1, s2 in allowed_combinations:
            if ((state1_norm == s1 and state2_norm == s2) or
                (state1_norm == s2 and state2_norm == s1)):
                return True
                
        return False
    
    def _check_node_specific_diagnostics(self, node_name, data):
        """
        Check for node-specific issues in diagnostic data.
        Uses configurable thresholds from parameters.
        
        Args:
            node_name: Name of the node
            data: Diagnostic data dictionary
        """
        try:
            # Check detection rates for YOLO
            if node_name == 'yolo':
                metrics = data.get('metrics', {})
                if metrics and 'detection_rate' in metrics:
                    # Check for low detection rate
                    detection_rate = metrics.get('detection_rate', 0)
                    if detection_rate < self.detection_rate_threshold:
                        self.logger.warning("DETECTION", 
                            f"Low YOLO detection rate: {detection_rate:.1f}/sec (threshold: {self.detection_rate_threshold}/sec)",
                            throttle_key="yolo_low_rate", throttle_seconds=10)
            
            # Check for high CPU usage
            system_metrics = data.get('system', {})
            if system_metrics and 'cpu_load' in system_metrics:
                cpu_load = system_metrics.get('cpu_load', 0)
                if cpu_load > self.high_cpu_threshold:
                    self.logger.warning("RESOURCE", 
                        f"High CPU usage in {node_name}: {cpu_load:.1f}% (threshold: {self.high_cpu_threshold}%)",
                        throttle_key=f"{node_name}_high_cpu", throttle_seconds=10)
                        
            # Check for memory issues
            if system_metrics and 'memory_usage_mb' in system_metrics:
                memory_mb = system_metrics.get('memory_usage_mb', 0)
                # Alert if memory usage is very high (> 90% of 4GB for RPi)
                if memory_mb > 3686:  # ~90% of 4GB
                    self.logger.warning("RESOURCE", 
                        f"High memory usage in {node_name}: {memory_mb:.1f}MB",
                        throttle_key=f"{node_name}_high_memory", throttle_seconds=10)
        except Exception as e:
            self.logger.error("DIAG", f"Error in node-specific diagnostics for {node_name}: {str(e)}",
                             throttle_key=f"node_diag_error_{node_name}", throttle_seconds=10)
    
    def _check_detection_pipeline(self):
        """
        Check for issues in the detection-fusion-control pipeline with improved error handling.
        Look for mismatched timestamps, missing detections, and data inconsistencies.
        Implements correlation between pipeline issues for root cause analysis.
        """
        # Use our retry handler to make pipeline checks more reliable
        try:
            self.retry_handler.execute(
                self._execute_pipeline_check,
                retry_on=(ValueError, KeyError, IndexError, TypeError)
            )
        except Exception as e:
            self.logger.error("PIPELINE", 
                            f"Failed to check detection pipeline after retries: {str(e)}",
                            throttle_key="pipeline_check_failed", 
                            throttle_seconds=10)
            
            # Log more details for debugging if needed
            if "division by zero" in str(e) or "index out of range" in str(e):
                traceback_str = traceback.format_exc()
                self.logger.error("TRACE", 
                                f"Pipeline check error: {traceback_str}",
                                throttle_key="pipeline_trace", 
                                throttle_seconds=30)
    
    def _execute_pipeline_check(self):
        """
        Execute the actual pipeline check logic.
        Extracted for better error handling and retry capability.
        
        Raises:
            Various exceptions that can be retried
        """
        # Access position data with thread safety
        fusion_positions = None
        fusion_time = 0
        fusion_pos = None
        missing_sources = []
        
        # Use lock manager for better deadlock prevention
        position_lock_acquired = False
        
        try:
            # Try to acquire position lock with timeout
            position_lock_acquired = self.locks.acquire('position_lock', timeout=0.5)
            if not position_lock_acquired:
                self.logger.warning("LOCKS", 
                                  "Failed to acquire position lock for pipeline check - check skipped",
                                  throttle_key="pipeline_lock_timeout", 
                                  throttle_seconds=5)
                return
            
            # Check if we have recent position data from all sources
            current_time = time.time()
            fusion_positions = self.position_trackers.get('fusion', None)
            
            if fusion_positions is None or fusion_positions.count == 0:
                return  # No fusion data yet
                
            # Get latest fusion position timestamp
            latest_fusion = fusion_positions.get_latest(1)
            if not latest_fusion:
                return
                
            fusion_time = latest_fusion[0][0]  # [0] is first entry, [0] is timestamp
            fusion_pos = latest_fusion[0][1]   # [0] is first entry, [1] is position
            
            # Check if fusion data is too old
            if current_time - fusion_time > 5.0:  # More than 5 seconds old
                return  # Fusion data too old, skip check
            
            # Check if we have recent detections from other nodes
            detection_sources = ['lidar', 'yolo']
            
            for source in detection_sources:
                source_positions = self.position_trackers.get(source, None)
                if source_positions is None or source_positions.count == 0:
                    missing_sources.append(source)
                    continue
                    
                # Get latest detection
                latest = source_positions.get_latest(1)
                if not latest:
                    missing_sources.append(source)
                    continue
                    
                source_time = latest[0][0]
                
                # Check if detection is too old
                if current_time - source_time > 2.0:  # More than 2 seconds old
                    missing_sources.append(source)
        
        finally:
            # Always release the lock if acquired
            if position_lock_acquired:
                self.locks.release('position_lock')
        
        # Process findings outside the lock for better concurrency
        if missing_sources and fusion_positions and current_time - fusion_time < 1.0:
            # Create a unique event ID for correlation
            event_id = f"pipeline_issue_{int(current_time)}"
            
            self.logger.warning("PIPELINE", 
                              f"Fusion active but missing detections from: {', '.join(missing_sources)}",
                              throttle_key="missing_detections", 
                              throttle_seconds=5)
            
            # Record pipeline issue
            event = {
                'type': 'pipeline',
                'issue': 'missing_detections',
                'sources': missing_sources,
                'timestamp': current_time,
                'event_id': event_id
            }
            self.system_events.add(event)
            
            # Store for event correlation
            self.event_correlations[event_id] = {
                'type': 'pipeline_issue',
                'time': current_time,
                'related_events': [],
                'details': {
                    'missing_sources': missing_sources,
                    'fusion_time': fusion_time
                }
            }
            
            # Check for recovery needs
            for source in missing_sources:
                if source in self.node_failures:
                    failure_info = self.node_failures[source]
                    
                    # Increment failure count
                    failure_info['count'] += 1
                    failure_info['last_failure'] = current_time
                    
                    # Check if recovery should be attempted
                    if (failure_info['count'] > 3 and  # Multiple failures
                        not failure_info['recovery_attempted'] and  # Not attempted yet
                        current_time - failure_info['last_failure'] < 30.0):  # Recent failures
                        
                        # Log recovery attempt
                        self.logger.warning("RECOVERY", 
                                          f"Detection node {source} has {failure_info['count']} failures - "
                                          f"recovery may be needed",
                                          throttle_key=f"recovery_{source}", 
                                          throttle_seconds=60)
                        
                        # Mark as attempted
                        failure_info['recovery_attempted'] = True
        
        # Check position differences - look for large discrepancies
        # When two sensors strongly disagree on position
        self._check_position_consistency()
    
    def _check_position_consistency(self):
        """
        Check for consistency between position reports from different sensors.
        Uses configurable threshold for position differences.
        
        IMAGINE THIS: 🧩
        ---------------
        Imagine asking three friends to tell you where a basketball is on a court.
        If two friends point to completely different places, you know one of them
        must be wrong! This method does the same thing with the robot's sensors.
        
        The robot has multiple ways to detect the basketball (LIDAR, camera, etc.),
        and they should all report similar positions. If not, something is wrong.
        
        HOW IT WORKS (STEP-BY-STEP):
        ---------------------------
        1. Collect the latest position data from each sensor
           - Position from LIDAR: (x=1.2, y=0.5, z=0.1)
           - Position from camera: (x=1.3, y=0.6, z=0.1)
           - Position from fusion: (x=1.25, y=0.55, z=0.1)
        
        2. Compare the positions - are they close to each other?
           - Calculate distance between positions
           - If the distance is too large (e.g., > 1.0 meter), it's suspicious!
        
        3. Record and report inconsistencies
           - Log warning message about the large difference
           - Track the event for later analysis
           - Try to correlate with other issues
        
        This is a critical check because big position differences usually mean
        one of the sensors is giving bad data, which could make the robot behave
        incorrectly - like trying to catch a basketball that isn't really there!
        """
        try:
            # Use lock manager for better deadlock prevention
            position_lock_acquired = False
            
            try:
                # Try to acquire position lock with timeout
                position_lock_acquired = self.locks.acquire('position_lock', timeout=0.5)
                if not position_lock_acquired:
                    self.logger.warning("LOCKS", 
                                      "Failed to acquire position lock for consistency check - check skipped",
                                      throttle_key="consistency_lock_timeout", 
                                      throttle_seconds=5)
                    return
                
                # Get latest positions from each source
                positions = {}
                
                for source, buffer in self.position_trackers.items():
                    latest = buffer.get_latest(1)
                    if latest:
                        # Extract timestamp and position
                        timestamp, position = latest[0]
                        positions[source] = {
                            'timestamp': timestamp,
                            'position': position
                        }
            finally:
                # Always release the lock if acquired
                if position_lock_acquired:
                    self.locks.release('position_lock')
            
            # We need at least two sources to compare
            if len(positions) < 2:
                return
                
            # Compare fusion with detection sources
            if 'fusion' in positions:
                fusion_pos = positions['fusion']['position']
                fusion_time = positions['fusion']['timestamp']
                
                # Compare with detections - check each detection source
                # ignore yolo distance since it is in pixel and needs to be converted for better esimation...
                for source in ['lidar']: 
                    if source in positions:
                        source_pos = positions[source]['position']
                        source_time = positions[source]['timestamp']
                        
                        # Only compare if timestamps are reasonably close
                        if abs(fusion_time - source_time) < 1.0:
                            # Calculate distance between positions using safe method
                            try:
                                distance = fusion_pos.distance_to(source_pos)
                                
                                # Check if distance is unusually large (potential inconsistency)
                                if distance > self.position_difference_threshold:
                                    # Create a unique event ID for correlation
                                    event_id = f"position_inconsistency_{int(time.time())}"
                                    
                                    self.logger.warning("CONSISTENCY", 
                                        f"Large position difference between fusion and {source}: "
                                        f"{distance:.2f}m (threshold: {self.position_difference_threshold}m)",
                                        throttle_key=f"pos_diff_{source}", throttle_seconds=5)
                                    
                                    # Record inconsistency event
                                    event = {
                                        'type': 'inconsistency',
                                        'sources': ['fusion', source],
                                        'distance': distance,
                                        'positions': [
                                            str(fusion_pos),  # Use string representation for logging
                                            str(source_pos)
                                        ],
                                        'event_id': event_id,
                                        'timestamp': time.time()
                                    }
                                    self.system_events.add(event)
                                    
                                    # Store for event correlation
                                    self.event_correlations[event_id] = {
                                        'type': 'position_inconsistency',
                                        'time': time.time(),
                                        'related_events': [],
                                        'details': {
                                            'sources': ['fusion', source],
                                            'distance': distance
                                        }
                                    }
                                    
                                    # Check if this inconsistency correlates with other recent events
                                    self._correlate_events(event_id, 'position_inconsistency')
                            except (ValueError, TypeError, ZeroDivisionError) as err:
                                # Handle specific math errors
                                self.logger.error("CONSISTENCY", 
                                    f"Error calculating position distance for {source}: {str(err)}",
                                    throttle_key=f"pos_calc_error_{source}", throttle_seconds=10)
                                
                                # Increment validation error counter
                                self.validation_stats['invalid_positions'] += 1
        except Exception as e:
            self.logger.error("CONSISTENCY", f"Error checking position consistency: {str(e)}",
                             throttle_key="consistency_error", throttle_seconds=10)
    
    def _correlate_events(self, event_id, event_type):
        """
        Correlate events to detect related issues.
        
        IMAGINE THIS: 🕵️‍♀️
        ---------------
        Think of this like a detective connecting clues on a case board. When one
        strange thing happens in your robot system, it might be connected to other
        strange things that happened around the same time.
        
        For example, if the LIDAR sensor stops working AND the robot reports strange
        position data within a few seconds, these problems are probably related rather
        than separate coincidences!
        
        HOW IT WORKS (STEP-BY-STEP):
        ---------------------------
        1. When a new problem occurs (like a sensor error):
           - Note the time it happened
           - Create a unique ID for this problem
        
        2. Look for other recent problems (within 5 seconds)
           - Check which types of problems might be related
           - See if any recent problems match those types
        
        3. If related problems are found:
           - Connect them in both directions (like drawing a line between clues)
           - Log the connection so we know they might have the same root cause
        
        This is like how doctors look for patterns in symptoms to diagnose an illness.
        Rather than treating each symptom separately, they look for the underlying cause
        that explains multiple symptoms.
        
        Args:
            event_id: ID of the current event
            event_type: Type of the current event
        """
        try:
            current_time = time.time()
            
            # Look for events in the last 5 seconds
            time_window = 5.0
            
            # Get related event types based on current event
            related_types = []
            if event_type == 'position_inconsistency':
                related_types = ['pipeline_issue', 'state_change']
            elif event_type == 'pipeline_issue':
                related_types = ['position_inconsistency', 'state_change']
            elif event_type == 'state_change':
                related_types = ['position_inconsistency', 'pipeline_issue']
            
            # Check for related events
            for other_id, correlation in self.event_correlations.items():
                # Skip self-correlation
                if other_id == event_id:
                    continue
                    
                # Check if event is recent
                if current_time - correlation['time'] <= time_window:
                    # Check if event type is related
                    if correlation['type'] in related_types:
                        # Add bidirectional correlation
                        self.event_correlations[event_id]['related_events'].append(other_id)
                        correlation['related_events'].append(event_id)
                        
                        # Log correlation for diagnostics
                        self.logger.info("CORRELATION", 
                            f"Correlated events: {event_type} related to {correlation['type']}",
                            throttle_key="event_correlation", throttle_seconds=5)
        except Exception as e:
            self.logger.error("CORRELATION", f"Error correlating events: {str(e)}",
                           throttle_key="correlation_error", throttle_seconds=10)
    
    def _check_node_heartbeats(self):
        """
        Check for nodes that haven't reported recently with configurable thresholds.
        
        IMAGINE THIS: ❤️
        ---------------
        Think of this like checking if your friends are still active in a group chat.
        If someone hasn't sent a message in a long time, you might wonder if they're
        still there or if they've lost connection.
        
        This method checks if each part of the robot (like LIDAR, cameras, etc.) 
        has "checked in" recently. If not, it marks them as "missing" - which might
        mean they've crashed or are having problems.
        
        HOW IT WORKS (STEP-BY-STEP):
        ---------------------------
        1. Get current time (like looking at your watch)
        2. For each robot component (node):
           - Check when we last heard from it
           - Compare with a timeout threshold (LIDAR: 5 seconds, Camera: 5 seconds, etc.)
           - If it's been too long, add it to a "missing" list
        
        3. If nodes are missing:
           - Log warnings so we know something's wrong
           - Keep track of how long they've been missing
           - Potentially change system state to "DEGRADED" if important nodes are missing
        
        It's like a roll call in school - we're checking who's present and who's absent!
        
        Returns:
            list: List of missing node names
        """
        missing_nodes = []
        
        try:
            current_time = self.get_clock().now().to_msg()
            current_sec = current_time.sec + (current_time.nanosec / 1e9)
            
            # Thread-safe access to heartbeat data using lock manager
            state_lock_acquired = False
            
            try:
                # Try to acquire lock with timeout
                state_lock_acquired = self.locks.acquire('state_lock', timeout=1.0)
                if not state_lock_acquired:
                    self.logger.warning("LOCKS", 
                                      "Failed to acquire state lock for heartbeat check - check skipped",
                                      throttle_key="heartbeat_lock_timeout", 
                                      throttle_seconds=5)
                    return []
                
                # Nodes to check with their thresholds
                node_thresholds = {
                    'lidar': 5.0,          # Detection nodes need faster updates
                    'yolo': 5.0,
                    'fusion': 5.0,
                    'state_manager': 3.0,  # State manager is critical
                    'pid': 10.0            # Default threshold
                }
                
                # Check each node
                for node, threshold in node_thresholds.items():
                    if node not in self.node_heartbeats:
                        missing_nodes.append(node)
                        continue
                        
                    last_time = self.node_heartbeats[node]
                    if not hasattr(last_time, 'sec') or not hasattr(last_time, 'nanosec'):
                        # Invalid timestamp
                        missing_nodes.append(f"{node} (invalid timestamp)")
                        continue
                        
                    last_sec = last_time.sec + (last_time.nanosec / 1e9)
                    
                    # Check if node hasn't reported in a while (using its threshold)
                    if current_sec - last_sec > threshold:
                        missing_nodes.append(f"{node} ({current_sec - last_sec:.1f}s)")
            finally:
                # Always release the lock if acquired
                if state_lock_acquired:
                    self.locks.release('state_lock')
            
            # Log missing nodes if any were found
            if missing_nodes:
                self.logger.warning("HEARTBEAT", 
                    f"Missing heartbeats from nodes: {', '.join(missing_nodes)}",
                    throttle_key="missing_heartbeats", throttle_seconds=10)
                
                # Record heartbeat issue with unique event ID for correlation
                event_id = f"heartbeat_issue_{int(time.time())}"
                self.system_events.add({
                    'type': 'heartbeat',
                    'missing_nodes': missing_nodes,
                    'timestamp': time.time(),
                    'event_id': event_id
                })
                
                # Store for event correlation
                self.event_correlations[event_id] = {
                    'type': 'heartbeat_issue',
                    'time': time.time(),
                    'related_events': [],
                    'details': {
                        'missing_nodes': missing_nodes
                    }
                }
                
                # Check for node recovery needs
                for node_entry in missing_nodes:
                    # Extract node name from potentially annotated entry
                    node = node_entry.split(' ')[0] if ' ' in node_entry else node_entry
                    
                    if node in self.node_failures:
                        failure_info = self.node_failures[node]
                        
                        # Increment failure count
                        failure_info['count'] += 1
                        failure_info['last_failure'] = time.time()
                        
                        # Log persistent failure
                        if failure_info['count'] > 3:
                            self.logger.error("HEARTBEAT", 
                                            f"Node {node} has been missing for {failure_info['count']} checks",
                                            throttle_key=f"persistent_missing_{node}", 
                                            throttle_seconds=60)
                            
                            # Update node state if persistent failures
                            if self.node_state == self.STATE_RUNNING:
                                self.node_state = self.STATE_DEGRADED
                                self.logger.warning("STATE", 
                                                  f"Diagnostic node state changed to DEGRADED due to missing nodes",
                                                  throttle_key="degraded_state", 
                                                  throttle_seconds=60)
        except Exception as e:
            self.logger.error("HEARTBEAT", f"Error checking node heartbeats: {str(e)}",
                             throttle_key="heartbeat_error", throttle_seconds=10)
            missing_nodes = []  # Reset on error
        
        return missing_nodes
    
    def _run_system_diagnostics(self):
        """
        Run comprehensive system diagnostics with error isolation.
        Each check is run in a separate try-except block to isolate failures.
        
        IMAGINE THIS: 🔍
        ---------------
        Picture a thorough car inspection where a mechanic systematically checks
        the engine, brakes, tires, lights, and fluids - one system at a time.
        Even if there's a problem with the brakes, they still continue checking
        everything else to give you a complete picture of your car's health.
        
        This method does a complete "inspection" of your robot system, checking
        all the important components even if some checks fail.
        
        HOW IT WORKS (STEP-BY-STEP):
        ---------------------------
        1. Start a series of health checks:
           - Pipeline check: Is data flowing correctly between components?
           - Heartbeat check: Are all nodes actively running?
           - Resource check: Is there enough CPU, memory, and other resources?
        
        2. For each check:
           - Run it in a protected way (try-except) so one failure doesn't stop everything
           - Record if it succeeded or failed
           - Collect detailed results
        
        3. When all checks are done:
           - Generate a complete system health report
           - Log the results
           - Track how long the checks took
        
        It's like getting a full physical exam at the doctor - multiple tests
        that together give a complete picture of your health status!
        
        Returns:
            dict: Results of diagnostic checks with success/failure status
        """
        # Track overall success
        checks_run = 0
        checks_succeeded = 0
        results = {
            'pipeline_check': False,
            'heartbeat_check': False,
            'resource_check': False
        }
        
        # Get current time for performance tracking
        start_time = time.time()
        
        try:
            # Check if we're already running diagnostics (prevent concurrent execution)
            if hasattr(self, '_diagnostics_running') and self._diagnostics_running:
                self.logger.warning("DIAG", 
                                  "Diagnostic run skipped - previous run still in progress",
                                  throttle_key="diag_overlap", throttle_seconds=10)
                return results
            
            # Set flag to prevent concurrent execution
            self._diagnostics_running = True
            
            # Check detection pipeline
            if self.pipeline_check_enabled:
                try:
                    self._check_detection_pipeline()
                    checks_succeeded += 1
                    results['pipeline_check'] = True
                except Exception as e:
                    self.logger.error("DIAG", f"Pipeline check failed: {str(e)}",
                                    throttle_key="pipeline_check_error", throttle_seconds=30)
                finally:
                    checks_run += 1
            
            # Check node heartbeats
            try:
                missing_nodes = self._check_node_heartbeats()
                checks_succeeded += 1
                results['heartbeat_check'] = True
                results['missing_nodes'] = missing_nodes
            except Exception as e:
                self.logger.error("DIAG", f"Heartbeat check failed: {str(e)}",
                                throttle_key="heartbeat_check_error", throttle_seconds=30)
            finally:
                checks_run += 1
            
            # Check for resource issues
            if self.resource_check_enabled:
                try:
                    resource_stats = self._check_system_resources()
                    checks_succeeded += 1
                    results['resource_check'] = True
                    results['resource_stats'] = resource_stats
                except Exception as e:
                    self.logger.error("DIAG", f"Resource check failed: {str(e)}",
                                    throttle_key="resource_check_error", throttle_seconds=30)
                finally:
                    checks_run += 1
            
            # Thread-safe access to statistics
            statistics_lock_acquired = False
            try:
                statistics_lock_acquired = self.locks.acquire('statistics_lock', timeout=0.5)
                if statistics_lock_acquired:
                    # Update node uptime
                    self.statistics['node_uptime'] = time.time() - self.statistics['start_time']
                    
                    # Log diagnostic run
                    self.logger.info("DIAG", 
                                   f"System diagnostic complete: "
                                   f"{checks_succeeded}/{checks_run} checks passed, "
                                   f"{self.statistics['node_uptime']:.1f}s uptime, "
                                   f"{self.statistics['events_processed']} events",
                                   throttle_key="diag_complete", throttle_seconds=30)
                else:
                    # Log without statistics if lock acquisition failed
                    self.logger.info("DIAG", 
                                   f"System diagnostic complete: "
                                   f"{checks_succeeded}/{checks_run} checks passed",
                                   throttle_key="diag_complete", throttle_seconds=30)
            finally:
                if statistics_lock_acquired:
                    self.locks.release('statistics_lock')
            
            # Record execution time
            execution_time = time.time() - start_time
            results['execution_time'] = execution_time
            
            # Update node state based on check results
            if checks_run > 0:
                success_rate = checks_succeeded / checks_run
                if success_rate < 0.5:
                    # More than half of checks failed
                    if self.node_state != self.STATE_ERROR:
                        self.node_state = self.STATE_ERROR
                        self.logger.error("STATE", 
                                        "Diagnostic node state changed to ERROR due to multiple failed checks",
                                        throttle_key="error_state", throttle_seconds=60)
                elif success_rate < 1.0:
                    # Some checks failed
                    if self.node_state == self.STATE_RUNNING:
                        self.node_state = self.STATE_DEGRADED
                        self.logger.warning("STATE", 
                                          "Diagnostic node state changed to DEGRADED due to some failed checks",
                                          throttle_key="degraded_state", throttle_seconds=60)
                else:
                    # All checks succeeded
                    if self.node_state == self.STATE_DEGRADED:
                        # Recover from degraded state if all checks pass
                        self.node_state = self.STATE_RUNNING
                        self.logger.info("STATE", 
                                       "Diagnostic node state recovered to RUNNING",
                                       throttle_key="recovered_state", throttle_seconds=60)
            
        except Exception as e:
            self.logger.error("DIAG", f"Error running system diagnostics: {str(e)}",
                             throttle_key="diag_error", throttle_seconds=10)
            # Update node state on critical error
            if self.node_state != self.STATE_ERROR:
                self.node_state = self.STATE_ERROR
                self.logger.error("STATE", 
                                "Diagnostic node state changed to ERROR due to exception in diagnostics",
                                throttle_key="error_state", throttle_seconds=60)
        finally:
            # Always clear the running flag
            self._diagnostics_running = False
        
        return results
    
    def _check_system_resources(self):
        """
        Check for system resource issues.
        Improved to track memory usage as well as CPU.
        """
        try:
            # Thread-safe access to diagnostic data
            with self.locks.with_lock('diagnostic_lock'):
                # Check combined CPU usage from all nodes
                total_cpu = 0.0
                total_memory_mb = 0.0
                node_count = 0
                
                for node, diag in self.node_diagnostics.items():
                    data = diag.get('data', {})
                    system_data = data.get('system', {})
                    
                    if system_data:
                        # Get CPU usage
                        if 'cpu_load' in system_data:
                            cpu = float(system_data.get('cpu_load', 0))
                            total_cpu += cpu
                            node_count += 1
                        
                        # Get memory usage
                        if 'memory_usage_mb' in system_data:
                            memory_mb = float(system_data.get('memory_usage_mb', 0))
                            total_memory_mb += memory_mb
                
                # Calculate average CPU usage
                if node_count > 0:
                    avg_cpu = total_cpu / node_count
                    
                    # Check for high overall CPU usage
                    if avg_cpu > self.high_cpu_threshold:
                        self.logger.warning("RESOURCE", 
                            f"High system CPU usage: {avg_cpu:.1f}% average "
                            f"across {node_count} nodes (threshold: {self.high_cpu_threshold}%)",
                            throttle_key="high_system_cpu", throttle_seconds=10)
                    
                    # Update statistics safely
                    with self.statistics_lock:
                        self.statistics['cpu_usage'] = avg_cpu
                        self.statistics['memory_usage'] = total_memory_mb
                
                # Check total memory usage (estimate)
                if total_memory_mb > 3500:  # ~85% of 4GB (typical for RPi 5)
                    self.logger.warning("RESOURCE", 
                        f"High total memory usage: {total_memory_mb:.1f}MB",
                        throttle_key="high_system_memory", throttle_seconds=30)
                    
                    # Record resource issue
                    self.system_events.add({
                        'type': 'resource',
                        'issue': 'high_memory',
                        'value': total_memory_mb,
                        'threshold': 3500
                    })
                
        except Exception as e:
            self.logger.error("RESOURCE", 
                f"Error checking system resources: {str(e)}",
                throttle_key="resource_check_error", throttle_seconds=10)

    def _monitor_own_resources(self):
        """
        Monitor this node's resource usage and adjust behavior adaptively.
        Implements self-healing by adjusting diagnostic frequency.
        """
        try:
            # Track elapsed time
            current_time = time.time()
            elapsed = current_time - self.last_resource_update
            self.last_resource_update = current_time
            
            # Update node uptime in statistics
            with self.locks.with_lock('statistics_lock'):
                self.statistics['node_uptime'] = current_time - self.statistics['start_time']
            
            # Adaptive timer adjustments based on system load
            # If system is heavily loaded, reduce frequency of non-critical checks
            with self.locks.with_lock('statistics_lock'):
                system_cpu = self.statistics['cpu_usage']
                
                # Handle high CPU load
                if system_cpu > 90:
                    # System is heavily loaded, extend timer intervals
                    if self.diagnostic_interval < 10.0:
                        # Double the diagnostic interval (up to 10 seconds max)
                        new_interval = min(self.diagnostic_interval * 2, 10.0)
                        old_interval = self.diagnostic_interval
                        self.diagnostic_interval = new_interval
                        
                        # Reset timer with new interval
                        if 'diagnostic' in self.timers:
                            self.timers['diagnostic'].cancel()
                            self.timers['diagnostic'] = self.create_timer(
                                self.diagnostic_interval, 
                                self._run_system_diagnostics
                            )
                            
                            self.logger.info("ADAPTIVE", 
                                f"Reduced diagnostic frequency due to high CPU load: "
                                f"{old_interval:.1f}s → {new_interval:.1f}s",
                                throttle_key="adaptive_interval", throttle_seconds=30)
                
                # Handle normal or low CPU load - restore normal intervals
                elif system_cpu < 60 and self.diagnostic_interval > 5.0:
                    # System has available resources, decrease interval to normal
                    old_interval = self.diagnostic_interval
                    self.diagnostic_interval = 5.0
                    
                    # Reset timer with new interval
                    if 'diagnostic' in self.timers:
                        self.timers['diagnostic'].cancel()
                        self.timers['diagnostic'] = self.create_timer(
                            self.diagnostic_interval, 
                            self._run_system_diagnostics
                        )
                        
                        self.logger.info("ADAPTIVE", 
                            f"Restored normal diagnostic frequency: "
                            f"{old_interval:.1f}s → {self.diagnostic_interval:.1f}s",
                            throttle_key="adaptive_restore", throttle_seconds=30)
                
                # Check for log rotation need
                if hasattr(self.logger, '_check_log_rotation'):
                    self.logger._check_log_rotation()
                
                # Run throttle cleanup periodically
                if hasattr(self.logger, '_cleanup_throttle_entries'):
                    self.logger._cleanup_throttle_entries()
                
        except Exception as e:
            self.logger.error("RESOURCE", 
                f"Error monitoring own resources: {str(e)}",
                throttle_key="own_resource_error", throttle_seconds=10)
    
    def _log_system_heartbeat(self):
        """Log a system heartbeat with basic status and health score."""
        try:
            # Thread-safe access to state data
            with self.locks.with_lock('state_lock'):
                # Get system state
                system_state = self.node_states.get('system', {}).get('state', 'unknown')
            
            # Thread-safe access to diagnostic data
            with self.locks.with_lock('diagnostic_lock'):
                # Count active nodes
                active_nodes = 0
                error_nodes = 0
                
                for node, diag in self.node_diagnostics.items():
                    # Check if diagnostic data is recent (within 5 seconds)
                    if diag.get('timestamp', 0) > time.time() - 5.0:
                        active_nodes += 1
                        
                        # Count nodes in error state
                        if diag.get('status') in ['error', 'critical']:
                            error_nodes += 1
            
            # Calculate simple health score (0-100%)
            total_nodes = 5  # Expected nodes
            node_score = (active_nodes / total_nodes) * 100 if total_nodes > 0 else 0
            error_penalty = (error_nodes / total_nodes) * 50 if total_nodes > 0 else 0
            health_score = max(0, min(100, node_score - error_penalty))
              # Get current performance stats
            with self.locks.with_lock('statistics_lock'):
                elapsed = time.time() - self.statistics['start_time']
                events_per_sec = self.statistics['events_processed'] / elapsed if elapsed > 0 else 0
                
                # Log heartbeat with health score
                self.logger.info("HEARTBEAT", 
                                f"System health: {health_score:.0f}%, "
                                f"state={system_state}, "
                                f"{active_nodes}/{total_nodes} nodes reporting "
                                f"({error_nodes} with errors), "
                                f"{events_per_sec:.1f} events/sec", 
                                throttle_key="heartbeat", throttle_seconds=10)
        except Exception as e:
            self.logger.error("HEARTBEAT", 
                f"Error logging system heartbeat: {str(e)}",
                throttle_key="heartbeat_error", throttle_seconds=10)
    
    def _write_periodic_summary(self):
        """Write a periodic summary of system status to log file."""
        try:
            # Calculate uptime
            current_time = time.time()
            uptime = current_time - self.statistics['start_time']
            
            # Thread-safe access to state data
            with self.state_lock:
                # Get system state for summary
                system_state = self.node_states.get('system', {}).get('state', 'unknown')
                
                # Create state information
                node_states = {}
                for node, state_data in self.node_states.items():
                    state = state_data.get('state', 'unknown')
                    timestamp = state_data.get('timestamp', 0)
                    age = current_time - timestamp if timestamp > 0 else 0
                    
                    node_states[node] = f"{state} (age: {age:.1f}s)"
            
            # Thread-safe access to diagnostics data
            active_nodes = 0
            error_nodes = 0
            node_health = {}
            
            with self.diagnostic_lock:
                # Count active nodes for summary
                for node, diag in self.node_diagnostics.items():
                    if diag.get('timestamp', 0) > current_time - 5.0:
                        active_nodes += 1
                        
                        # Track node health
                        status = diag.get('status', 'unknown')
                        data = diag.get('data', {})
                        
                        if status in ['error', 'critical']:
                            error_nodes += 1
                        
                        # Extract health score if available
                        health_data = data.get('health', {})
                        if isinstance(health_data, dict) and 'overall' in health_data:
                            health_score = health_data.get('overall', 0.0) * 100.0  # Convert to percentage
                            node_health[node] = f"{status.upper()} (health: {health_score:.1f}%)"
                        else:
                            node_health[node] = status.upper()
            
            # Thread-safe access to statistics
            with self.statistics_lock:
                # Create summary data
                summary_data = {
                    "System Status": {
                        "Uptime": f"{uptime:.1f} seconds",
                        "Active Nodes": f"{active_nodes}/5 nodes",
                        "Nodes with Errors": error_nodes,
                        "System State": system_state,
                        "Events Processed": self.statistics['events_processed'],
                        "Events per Second": f"{self.statistics['events_processed'] / uptime:.1f}" if uptime > 0 else "0.0",
                        "Errors Detected": self.statistics['errors_detected'],
                        "Warnings Detected": self.statistics['warnings_detected'],
                        "State Transitions": self.statistics['state_transitions'],
                        "Sync Issues": self.statistics['sync_issues'],
                        "Average CPU Usage": f"{self.statistics['cpu_usage']:.1f}%",
                        "Total Memory Usage": f"{self.statistics['memory_usage']:.1f} MB"
                    },
                    "Node States": node_states,
                    "Node Health": node_health
                }
            
            # Get recent events
            recent_events = []
            
            # Get error events first (high priority)
            for timestamp, event in self.error_events.get_latest(3):
                event_time = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                node = event.get('node', 'unknown')
                message = event.get('message', 'No message')
                
                error_str = f"{event_time} - ERROR [{node}]: {message}"
                recent_events.append(error_str)
            
            # Get other system events
            for timestamp, event in self.system_events.get_latest(5):
                event_time = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                event_type = event.get('type', 'unknown')
                
                # Format based on event type
                if event_type == 'sync':
                    nodes = event.get('nodes', [])
                    states = event.get('states', [])
                    event_str = f"{event_time} - Sync issue: {'+'.join(nodes)} with states {'+'.join(states)}"
                elif event_type == 'inconsistency':
                    distance = event.get('distance', 0.0)
                    sources = event.get('sources', [])
                    event_str = f"{event_time} - Position inconsistency: {distance:.2f}m between {'+'.join(sources)}"
                elif event_type == 'motion':
                    motion_event = event.get('event', 'unknown')
                    event_str = f"{event_time} - Motion: {motion_event}"
                elif event_type == 'heartbeat':
                    missing = event.get('missing_nodes', [])
                    event_str = f"{event_time} - Missing heartbeats: {', '.join(missing)}"
                elif event_type == 'pipeline':
                    issue = event.get('issue', 'unknown')
                    event_str = f"{event_time} - Pipeline issue: {issue}"
                elif event_type == 'resource':
                    issue = event.get('issue', 'unknown')
                    value = event.get('value', 0)
                    event_str = f"{event_time} - Resource issue: {issue} ({value})"
                else:
                    event_str = f"{event_time} - {event_type}: {str(event)}"
                    
                recent_events.append(event_str)
            
            summary_data["Recent Events"] = recent_events
            
            # Add configuration settings
            summary_data["Configuration"] = {
                "diagnostic_interval": f"{self.diagnostic_interval:.1f}s",
                "position_difference_threshold": f"{self.position_difference_threshold:.2f}m",
                "detection_rate_threshold": f"{self.detection_rate_threshold:.1f}/sec",
                "high_cpu_threshold": f"{self.high_cpu_threshold:.1f}%"
            }
            
            # Write summary to log file
            self.logger.write_summary(summary_data)
            
            # Log to console
            self.logger.info("SUMMARY", f"Periodic summary written to log: {uptime:.1f}s uptime, "
                            f"{self.statistics['events_processed']} events, "
                            f"{self.statistics['errors_detected']} errors")
        except Exception as e:
            self.logger.error("SUMMARY", 
                f"Error writing periodic summary: {str(e)}",
                throttle_key="summary_error", throttle_seconds=10)
    
    def _count_active_nodes(self):
        """
        Count number of nodes that have reported recently.
        
        Returns:
            tuple: (active_count, error_count, total_count)
        """
        active_count = 0
        error_count = 0
        total_count = 5  # Expected total nodes
        
        # Define the expected nodes
        expected_nodes = ['lidar', 'yolo', 'fusion', 'state_manager', 'pid']
        
        # Get current time for age calculation
        current_time = time.time()
        
        # Safe access to diagnostics with lock manager
        diagnostic_lock_acquired = False
        
        try:
            # Try to acquire lock with timeout
            diagnostic_lock_acquired = self.locks.acquire('diagnostic_lock', timeout=0.5)
            if not diagnostic_lock_acquired:
                # If lock acquisition fails, return default values
                self.logger.debug("LOCKS", 
                                "Failed to acquire diagnostic lock for active node count",
                                throttle_key="count_lock_timeout", 
                                throttle_seconds=30)
                return (0, 0, total_count)
            
            # Count active and error nodes
            for node in expected_nodes:
                diag = self.node_diagnostics.get(node, None)
                if diag is not None:
                    # Check if diagnostic data is recent (within 5 seconds)
                    if diag.get('timestamp', 0) > current_time - 5.0:
                        active_count += 1
                        
                        # Check error status
                        if diag.get('status') in ['error', 'critical']:
                            error_count += 1
        finally:
            # Always release the lock if acquired
            if diagnostic_lock_acquired:
                self.locks.release('diagnostic_lock')
        
        return (active_count, error_count, total_count)
    
    def _generate_final_summary(self):
        """Generate final system summary for the log."""
        try:
            # Calculate final statistics
            current_time = time.time()
            uptime = current_time - self.statistics['start_time']
            
            # Thread-safe access to statistics
            with self.statistics_lock:
                events_per_sec = self.statistics['events_processed'] / uptime if uptime > 0 else 0
                
                # Create summary data
                summary_data = {
                    "Final System Status": {
                        "Uptime": f"{uptime:.1f} seconds",
                        "Events Processed": self.statistics['events_processed'],
                        "Processing Rate": f"{events_per_sec:.1f} events/sec",
                        "Errors Detected": self.statistics['errors_detected'],
                        "Warnings Detected": self.statistics['warnings_detected'],
                        "State Transitions": self.statistics['state_transitions'],
                        "Sync Issues": self.statistics['sync_issues'],
                        "Final CPU Usage": f"{self.statistics['cpu_usage']:.1f}%",
                        "Final Memory Usage": f"{self.statistics['memory_usage']:.1f} MB"
                    }
                }
            
            # Add node health information
            node_health = {}
            with self.diagnostic_lock:
                for node, diag in self.node_diagnostics.items():
                    data = diag.get('data', {})
                    status = diag.get('status', 'unknown')
                    last_update = diag.get('timestamp', 0)
                    age = current_time - last_update
                    
                    # Extract health score if available
                    health_data = data.get('health', {})
                    health_score = 0.0
                    
                    if isinstance(health_data, dict) and 'overall' in health_data:
                        health_score = health_data.get('overall', 0.0) * 100.0  # Convert to percentage
                    
                    node_health[node] = f"{status} (health: {health_score:.1f}%, last update: {age:.1f}s ago)"
            
            summary_data["Node Health"] = node_health
            
            # Add state transition summary
            state_changes_list = []
            for timestamp, change in self.state_changes.get_all():
                change_time = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                node = change.get('node', 'unknown')
                old_state = change.get('old_state', 'unknown')
                new_state = change.get('new_state', 'unknown')
                
                state_changes_list.append(
                    f"{change_time} - {node}: {old_state} → {new_state}"
                )
            
            summary_data["State Transitions"] = state_changes_list
            
            # Add error events
            error_list = []
            for timestamp, error in self.error_events.get_all():
                error_time = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                node = error.get('node', 'unknown')
                message = error.get('message', 'No message')
                
                error_list.append(f"{error_time} - {node}: {message}")
                
            summary_data["Errors"] = error_list
            
            # Write summary to log file
            self.logger.write_summary(summary_data)
            
            # Print final message to console
            ColorPrinter.print("=" * 80, 'cyan')
            ColorPrinter.print("SYSTEM DIAGNOSTIC SUMMARY", 'cyan', bold=True)
            ColorPrinter.print("=" * 80, 'cyan')
            ColorPrinter.print(f"Uptime: {uptime:.1f} seconds", 'white')
            ColorPrinter.print(f"Events: {self.statistics['events_processed']} ({events_per_sec:.1f}/sec)", 'white')
            ColorPrinter.print(f"Errors: {self.statistics['errors_detected']}", 
                             'red' if self.statistics['errors_detected'] > 0 else 'white')
            ColorPrinter.print(f"State Transitions: {self.statistics['state_transitions']}", 'white')
            ColorPrinter.print(f"Summary written to: {self.logger.log_file}", 'green')
            ColorPrinter.print("=" * 80, 'cyan')
        except Exception as e:
            self.logger.error("SUMMARY", 
                f"Error generating final summary: {str(e)}",
                throttle_key="final_summary_error", throttle_seconds=1)
            # Log traceback for debugging critical errors
            traceback_str = traceback.format_exc()
            self.logger.error("TRACE", f"Final summary error: {traceback_str}")


def main(args=None):
    """
    Main function for the diagnostic node with improved error handling,
    memory management, and graceful shutdown.
    """
    # Initialize ROS with exception handling
    try:
        rclpy.init(args=args)
    except Exception as e:
        print(f"ERROR: Failed to initialize ROS: {e}")
        return 1
    
    # Create node
    node = None
    exit_code = 0
    
    try:
        # Setup signal handlers for graceful interrupt
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        
        def signal_handler(sig, frame):
            nonlocal node
            print(f"\nReceived signal {sig}, shutting down gracefully...")
            if node is not None:
                try:
                    # Log shutdown
                    if hasattr(node, 'logger'):
                        node.logger.info("SHUTDOWN", 
                                       f"Received signal {sig}, initiating graceful shutdown")
                    
                    # Run cleanups
                    if hasattr(node, '_cleanup_resources'):
                        node._cleanup_resources()
                except Exception as e:
                    print(f"WARNING: Error during signal handler cleanup: {e}")
                finally:
                    # Restore original handler and re-raise
                    signal.signal(sig, original_sigint if sig == signal.SIGINT else original_sigterm)
                    os.kill(os.getpid(), sig)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Create node with memory monitoring
        gc_at_start = gc.collect() if hasattr(gc, 'collect') else 0
        node = SystemDiagnosticNode()
        
        # Configure enhanced memory monitoring if psutil is available
        try:
            import psutil
            process = psutil.Process(os.getpid())
            
            # Monitor function with psutil
            def monitor_memory():
                try:
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    if hasattr(node, 'logger'):
                        node.logger.info("MEMORY", 
                                        f"Memory usage: {memory_mb:.1f}MB",
                                        throttle_key="memory_monitor", 
                                        throttle_seconds=300)
                except Exception:
                    pass  # Ignore monitoring errors
            
            # Create memory monitor timer
            memory_timer = threading.Timer(60.0, monitor_memory)
            memory_timer.daemon = True
            memory_timer.start()
        except ImportError:
            # psutil not available, use simpler memory tracking
            print("Note: psutil not available - using basic memory tracking")
        
        # Create executor with error handling
        try:
            # Create a MultiThreadedExecutor with thread number control
            # This provides better responsiveness while still managing resources
            num_threads = min(4, os.cpu_count() or 4)  # Limit to 4 threads maximum
            executor = rclpy.executors.MultiThreadedExecutor(num_threads=num_threads)
            executor.add_node(node)
            
            # Log execution start
            if hasattr(node, 'logger'):
                node.logger.info("INIT", 
                               f"Starting execution with {num_threads} threads")
            
            # Run the executor with comprehensive error handling
            try:
                executor.spin()
            except KeyboardInterrupt:
                print("\nShutting down diagnostic node due to keyboard interrupt...")
                if hasattr(node, 'logger'):
                    node.logger.info("SHUTDOWN", "Keyboard interrupt received")
            except MemoryError:
                print("\nERROR: Out of memory - initiating emergency shutdown")
                if hasattr(node, 'logger'):
                    node.logger.critical("MEMORY", "Out of memory - forcing shutdown")
                    
                # Force garbage collection
                if hasattr(gc, 'collect'):
                    gc.collect()
                    
                exit_code = 2
            except Exception as e:
                print(f"\nERROR: Unhandled exception during execution: {e}")
                if hasattr(node, 'logger'):
                    node.logger.critical("EXECUTION", f"Unhandled exception: {str(e)}")
                    
                # Log detailed traceback
                tb = traceback.format_exc()
                print(tb)
                if hasattr(node, 'logger'):
                    node.logger.error("TRACE", f"Execution error trace: {tb}")
                    
                exit_code = 1
        except Exception as e:
            print(f"ERROR: Failed to create executor: {e}")
            exit_code = 1
    except Exception as e:
        print(f"ERROR: Failed to create diagnostic node: {e}")
        print(traceback.format_exc())
        exit_code = 1
    finally:
        # Ensure clean shutdown even if errors occurred
        if node is not None:
            try:
                # Log shutdown
                if hasattr(node, 'logger'):
                    node.logger.info("SHUTDOWN", 
                                   f"Diagnostic node shutting down with exit code {exit_code}")
                
                # Run final cleanup with timeout protection
                if hasattr(node, '_cleanup_resources'):
                    try:
                        # Use a timeout to prevent hanging during cleanup
                        cleanup_thread = threading.Thread(target=node._cleanup_resources)
                        cleanup_thread.daemon = True
                        cleanup_thread.start()
                        cleanup_thread.join(timeout=5.0)  # Wait up to 5 seconds
                        
                        if cleanup_thread.is_alive():
                            print("WARNING: Cleanup timed out - forcing shutdown")
                            if hasattr(node, 'logger'):
                                node.logger.warning("SHUTDOWN", 
                                                 "Cleanup timed out - forcing shutdown")
                    except Exception as e:
                        print(f"ERROR: Exception during resource cleanup: {e}")
                
                # Generate final report if possible
                if hasattr(node, '_generate_final_summary'):
                    try:
                        node._generate_final_summary()
                    except Exception as e:
                        print(f"ERROR: Exception during final summary: {e}")
                
                # Close logger if possible
                if hasattr(node, 'logger') and hasattr(node.logger, 'close'):
                    try:
                        node.logger.close()
                    except Exception:
                        pass
                
                # Destroy node
                node.destroy_node()
            except Exception as e:
                print(f"ERROR: Exception during node cleanup: {e}")
        
        # Shutdown ROS with timeout protection
        try:
            # Use a timeout to prevent hanging during ROS shutdown
            shutdown_thread = threading.Thread(target=rclpy.shutdown)
            shutdown_thread.daemon = True
            shutdown_thread.start()
            shutdown_thread.join(timeout=5.0)  # Wait up to 5 seconds
            
            if shutdown_thread.is_alive():
                print("WARNING: ROS shutdown timed out - forcing exit")
                # Force exit if ROS shutdown hangs
                os._exit(exit_code)
        except Exception as e:
            print(f"ERROR: Exception during ROS shutdown: {e}")
            # Force exit on shutdown errors
            os._exit(exit_code if exit_code != 0 else 1)
        
        # Final memory report if possible
        if hasattr(gc, 'collect'):
            gc_at_end = gc.collect()
            print(f"Memory cleanup: {gc_at_end} objects collected during final GC")
    
    return exit_code


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"CRITICAL ERROR in main: {e}")
        traceback.print_exc()
        sys.exit(1)