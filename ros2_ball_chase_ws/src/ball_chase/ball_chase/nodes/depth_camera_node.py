#!/usr/bin/env python3

"""
Raspberry Pi 5 Ultra-Optimized Basketball Tracking - Depth Camera Node
======================================================================

Highly efficient implementation for basketball tracking designed
specifically for the Raspberry Pi 5's resource constraints.
"""
# Standard library imports - only import what's needed
import os
import time
import sys  # For immediate log flushing
import math  # For 3D estimation calculations
import psutil  # For accurate CPU monitoring

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor

# ROS2 message types
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String, Float32MultiArray
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# Third-party libraries
import numpy as np
from cv_bridge import CvBridge

# Project utilities
from ball_chase.utilities.resource_monitor import ResourceMonitor
from ball_chase.utilities.time_utils import TimeUtils
from ball_chase.config.config_loader import ConfigLoader
from ball_chase.utilities.ground_position_filter import GroundPositionFilter
from ball_chase.utilities.performance_metrics import PerformanceMetrics

# Memory optimization classes
class LightweightBuffer:
    """
    Efficient fixed-size buffer implementation to replace deque.
    Maintains a circular buffer with O(1) append and access operations.
    """
    def __init__(self, maxlen, init_with_none=True):
        """
        Initialize a fixed-size buffer.
        
        Args:
            maxlen (int): Maximum length of the buffer
            init_with_none (bool): Whether to initialize with None values
        """
        self.maxlen = maxlen
        self.buffer = [None] * maxlen if init_with_none else []
        self.size = 0
        self.index = 0
    
    def append(self, item):
        """Add an item to the buffer, overwriting oldest if full."""
        if self.size < self.maxlen:
            if self.size < len(self.buffer):
                self.buffer[self.size] = item
            else:
                self.buffer.append(item)
            self.size += 1
        else:
            self.buffer[self.index] = item
        
        # Move index for next write
        self.index = (self.index + 1) % self.maxlen
    
    def clear(self):
        """Clear the buffer."""
        self.size = 0
        self.index = 0
    
    def __len__(self):
        """Return current buffer size."""
        return self.size
    
    def __getitem__(self, index):
        """Access item at specified index."""
        if index < 0:
            index = self.size + index
        
        if 0 <= index < self.size:
            idx = (self.index - self.size + index) % self.maxlen
            return self.buffer[idx]
        
        raise IndexError("Buffer index out of range")
    
    def __iter__(self):
        """Iterate through buffer elements in order."""
        if self.size == 0:
            return
        
        # Start from oldest entry
        start_idx = 0 if self.size < self.maxlen else self.index
        
        for i in range(self.size):
            yield self.buffer[(start_idx + i) % self.maxlen]

class TTLDict:
    """
    Dictionary with time-to-live functionality for entries.
    Automatically cleans up expired entries during access.
    """
    def __init__(self, default_ttl=60.0):
        """
        Initialize a TTL dictionary.
        
        Args:
            default_ttl (float): Default TTL in seconds for entries
        """
        self.data = {}
        self.timestamps = {}
        self.default_ttl = default_ttl
    
    def __setitem__(self, key, value, ttl=None):
        """Set an item with a specific TTL."""
        self.data[key] = value
        self.timestamps[key] = time.time()
        # Store TTL with the key if provided
        if ttl is not None:
            if not hasattr(self, 'ttls'):
                self.ttls = {}
            self.ttls[key] = ttl
    
    def set(self, key, value, ttl=None):
        """Set an item with an optional custom TTL."""
        self.__setitem__(key, value, ttl)
    
    def __getitem__(self, key):
        """Get an item, removing if expired."""
        self._cleanup(key)
        if key in self.data:
            return self.data[key]
        raise KeyError(key)
    
    def get(self, key, default=None):
        """Get an item with a default value if missing or expired."""
        self._cleanup(key)
        return self.data.get(key, default)
    
    def _cleanup(self, key=None):
        """Clean up expired entries."""
        current_time = time.time()
        
        # Clean specific key if provided
        if key is not None:
            if key in self.timestamps:
                ttl = self.ttls.get(key, self.default_ttl) if hasattr(self, 'ttls') else self.default_ttl
                if current_time - self.timestamps[key] > ttl:
                    del self.data[key]
                    del self.timestamps[key]
                    if hasattr(self, 'ttls') and key in self.ttls:
                        del self.ttls[key]
            return
        
        # Full cleanup (expensive, use sparingly)
        keys_to_remove = []
        for k, timestamp in self.timestamps.items():
            ttl = self.ttls.get(k, self.default_ttl) if hasattr(self, 'ttls') else self.default_ttl
            if current_time - timestamp > ttl:
                keys_to_remove.append(k)
        
        for k in keys_to_remove:
            del self.data[k]
            del self.timestamps[k]
            if hasattr(self, 'ttls') and k in self.ttls:
                del self.ttls[k]
    
    def cleanup_all(self):
        """Force cleanup of all expired entries."""
        self._cleanup()
    
    def __contains__(self, key):
        """Check if key exists and is not expired."""
        self._cleanup(key)
        return key in self.data
    
    def __delitem__(self, key):
        """Delete an item."""
        del self.data[key]
        del self.timestamps[key]
        if hasattr(self, 'ttls') and key in self.ttls:
            del self.ttls[key]
    
    def keys(self):
        """Return non-expired keys."""
        self._cleanup()
        return self.data.keys()
    
    def values(self):
        """Return values for non-expired keys."""
        self._cleanup()
        return self.data.values()
    
    def items(self):
        """Return items for non-expired keys."""
        self._cleanup()
        return self.data.items()

class Matrix4x4:
    """
    Efficient 4x4 matrix implementation for transform operations.
    Optimized for 3D transformations with minimal memory allocations.
    """
    def __init__(self):
        """Initialize identity matrix."""
        # Initialize as identity matrix (row-major order)
        self.data = np.eye(4, dtype=np.float32)
    
    @classmethod
    def from_tf_transform(cls, transform):
        """
        Create matrix from ROS transform.
        
        Args:
            transform: ROS Transform message
        
        Returns:
            Matrix4x4: New matrix with transform data
        """
        matrix = cls()
        
        # Extract quaternion
        qx = transform.transform.rotation.x
        qy = transform.transform.rotation.y
        qz = transform.transform.rotation.z
        qw = transform.transform.rotation.w
        
        # Convert quaternion to rotation matrix
        # Precompute common products
        xx = qx * qx
        xy = qx * qy
        xz = qx * qz
        xw = qx * qw
        yy = qy * qy
        yz = qy * qz
        yw = qy * qw
        zz = qz * qz
        zw = qz * qw
        
        # Fill rotation part (3x3 top-left)
        matrix.data[0, 0] = 1.0 - 2.0 * (yy + zz)
        matrix.data[0, 1] = 2.0 * (xy - zw)
        matrix.data[0, 2] = 2.0 * (xz + yw)
        
        matrix.data[1, 0] = 2.0 * (xy + zw)
        matrix.data[1, 1] = 1.0 - 2.0 * (xx + zz)
        matrix.data[1, 2] = 2.0 * (yz - xw)
        
        matrix.data[2, 0] = 2.0 * (xz - yw)
        matrix.data[2, 1] = 2.0 * (yz + xw)
        matrix.data[2, 2] = 1.0 - 2.0 * (xx + yy)
        
        # Fill translation part (right column)
        matrix.data[0, 3] = transform.transform.translation.x
        matrix.data[1, 3] = transform.transform.translation.y
        matrix.data[2, 3] = transform.transform.translation.z
        
        return matrix
    
    def transform_point(self, x, y, z):
        """
        Transform a 3D point using this matrix.
        
        Args:
            x, y, z (float): Point coordinates
        
        Returns:
            tuple: Transformed (x, y, z) coordinates
        """
        # Apply transformation
        tx = self.data[0, 0] * x + self.data[0, 1] * y + self.data[0, 2] * z + self.data[0, 3]
        ty = self.data[1, 0] * x + self.data[1, 1] * y + self.data[1, 2] * z + self.data[1, 3]
        tz = self.data[2, 0] * x + self.data[2, 1] * y + self.data[2, 2] * z + self.data[2, 3]
        
        return (tx, ty, tz)
    
    def transform_vector(self, x, y, z):
        """
        Transform a 3D vector using this matrix (no translation).
        
        Args:
            x, y, z (float): Vector components
        
        Returns:
            tuple: Transformed (x, y, z) vector
        """
        # Apply rotation only
        tx = self.data[0, 0] * x + self.data[0, 1] * y + self.data[0, 2] * z
        ty = self.data[1, 0] * x + self.data[1, 1] * y + self.data[1, 2] * z
        tz = self.data[2, 0] * x + self.data[2, 1] * y + self.data[2, 2] * z
        
        return (tx, ty, tz)

# Config loading - done once at module level
config_loader = ConfigLoader()
config = config_loader.load_yaml('depth_config.yaml')

# Configuration from config file - streamlined for performance
DEPTH_CONFIG = config.get('depth', {
    "scale": 0.001,                # Depth scale factor (converts raw depth to meters)
    "min_depth": 0.1,              # Minimum valid depth in meters
    "max_depth": 8.0,              # Maximum valid depth in meters
    "radius": 5,                   # Default radius for depth sampling
    "min_valid_points": 3,         # Minimum valid points for reliable depth
    "use_depth_history": True,     # Enable depth history for continuity
    "max_roi_size": 60,            # Maximum ROI size
    "min_roi_size": 15,            # Minimum ROI size
    "history_max_age": 5.0,        # Maximum age for depth history (seconds) - INCREASED from 3.0 to 5.0
    "temporal_blending": 0.7,      # Temporal blending weight
    "quality_preference": 0.7,     # Quality vs. speed preference
    "adaptive_frame_skip": True    # Enable adaptive frame skipping
})

# Topic configuration
TOPICS = config.get('topics', {
    "input": {
        "camera_info": "/ascamera/camera_publisher/depth0/camera_info",
        "depth_image": "/ascamera/camera_publisher/depth0/image_raw",
        "yolo_detection": "/basketball/yolo/position",
        "yolo_bbox": "/basketball/yolo/bbox"
    },
    "output": {
        "yolo_3d": "/basketball/yolo/position_3d",
        "combined": "/basketball/detected_position"
    }
})

# Common reference frame configuration
COMMON_REFERENCE_FRAME = config.get('frames', {
    "reference_frame": "base_link",  # Common reference frame for all sensors
    "transform_timeout": 0.1         # Timeout for transform lookups (seconds)
})

# Distance-tiered processing configuration
DISTANCE_TIERS = {
    "close": {
        "range": (0.0, 1.0),      # 0-1m range
        "roi_size": 15,           # Small ROI for close objects
        "sampling_radius": 3,     # Small sampling radius
        "min_points": 1,          # Even single point can be reliable
        "fallback_scale": 1.0     # No scaling for close objects
    },
    "medium": {
        "range": (1.0, 1.75),     # 1-1.75m range
        "roi_size": 20,           # Medium ROI
        "sampling_radius": 5,     # Medium sampling
        "min_points": 3,          # Need more points for confidence
        "fallback_scale": 1.0     # No scaling
    },
    "far": {
        "range": (1.75, 2.75),    # 1.75-2.75m (problematic range)
        "roi_size": 30,           # Larger ROI for challenging range
        "sampling_radius": 10,    # MORE sampling points (increased from 7)
        "min_points": 3,          # REDUCED from 5 to improve hit rate
        "fallback_scale": 0.95    # IMPROVED scale factor (from 0.9)
    },
    "very_far": {
        "range": (2.75, 8.0),     # 2.75m+ range
        "roi_size": 40,           # Very large ROI for distant objects
        "sampling_radius": 12,    # INCREASED from 9 for better far detection
        "min_points": 5,          # REDUCED from 7 to improve hit rate
        "fallback_scale": 0.85    # Scale considerably
    }
}

# Performance tier configuration
PERFORMANCE_TIERS = {
    "balanced": {
        "process_every_n_frames": 1,    # Process every frame
        "roi_size_scale": 1.0,          # Normal ROI size
        "cache_lifetime": 0.5,          # REDUCED from 5.0 to prevent overuse of cache
        "use_matrix_transforms": True,  # Use optimized transforms
        "min_cpu_target": 60.0,         # Target for adaptation
        "max_cpu_target": 80.0          # Max acceptable CPU
    },
    "performance": {
        "process_every_n_frames": 2,    # Skip every other frame
        "roi_size_scale": 0.8,          # Smaller ROI for speed
        "cache_lifetime": 0.3,          # REDUCED from 10.0
        "use_matrix_transforms": True,  # Use optimized transforms
        "min_cpu_target": 40.0,         # Lower target
        "max_cpu_target": 75.0          # Lower max CPU
    },
    "ultra_performance": {
        "process_every_n_frames": 3,    # Process every 3rd frame
        "roi_size_scale": 0.6,          # Much smaller ROI
        "cache_lifetime": 0.2,          # REDUCED from 20.0
        "use_matrix_transforms": True,  # Use optimized transforms
        "min_cpu_target": 30.0,         # Very low target
        "max_cpu_target": 60.0          # Very low max CPU
    }
}

class OptimizedPositionEstimator(Node):
    """
    A ROS2 node that converts 2D ball detections to 3D positions.
    Optimized for basketball tracking on Raspberry Pi 5.
    """
    
    def __init__(self):
        """Initialize the 3D position estimator node with all required components."""
        super().__init__('basketball_3d_position_estimator')
        
        # Initialize core attributes
        self._init_attributes()
        
        # Memory-related setup
        self._setup_memory_pools()
        
        # Setup in logical order
        self._setup_callback_group()
        self._init_camera_parameters()
        self._setup_tf2()
        self._setup_transform_cache()
        self._setup_subscriptions()
        self._setup_publishers()
        self._setup_resource_monitoring()
        
        # Performance timers
        self._setup_performance_timers()
        
        # Pre-allocate message objects for reuse
        self._preallocate_messages()
        
        # Finalize initialization
        self._log_initialization()

    def _init_attributes(self):
        """Initialize all attributes with default values."""
        # Performance settings
        self.current_tier = "balanced"  # Start with balanced tier
        self.process_every_n_frames = PERFORMANCE_TIERS[self.current_tier]["process_every_n_frames"]
        self.frame_counter = 0
        self.current_cpu_usage = 0.0
        self._scale_factor = float(DEPTH_CONFIG["scale"])
        self._min_valid_depth = float(DEPTH_CONFIG["min_depth"])
        self._max_valid_depth = float(DEPTH_CONFIG["max_depth"])
        
        # Error tracking
        self.detection_errors = LightweightBuffer(10)  # Buffer to track recent errors
        self.yolo_detections_failed = 0  # Count failed detections
        
        # Depth tracking structure
        self.last_reported_depth = {
            'raw': 0.0,
            'processed': 0.0,
            'source': None,
            'timestamp': 0.0
        }
        
        # Debug flags
        self.debug_mode = False
        self.debug_depth = False
        self.last_debug_log = 0
        
        # Performance tracking
        self.performance_metrics = PerformanceMetrics(window_size=30)
        self.start_time = TimeUtils.now_as_float()
        self.successful_conversions = 0
        self.fps_history = LightweightBuffer(5)  # INCREASED from 3 for better averaging
        self.current_fps = 0.0
        self.last_fps_update = 0
        
        # Transform verification
        self.verified_transform = False
        self.transform_not_verified_logged = False
        
        # Camera information
        self.camera_info_logged = False
        self.camera_info = None
        self.depth_array = None
        self.depth_header = None
        
        # Detection tracking - using TTLDict for automatic cleanup
        self.detection_cache = {
            'YOLO': {'detection_2d': None, 'position_3d': None, 'timestamp': 0}
        }
        self.detection_history = {
            'YOLO': {'latest_position': None, 'last_time': 0}
        }
        
        # Position tracking
        self.last_position = None
        self.last_position_time = 0
        self.position_filter_alpha = 0.8
        self.max_position_change = 4.0
        
        # Error tracking with TTL
        self.error_last_logged = TTLDict(default_ttl=20.0)
        
        # Timestamps
        self.last_resource_alert_time = 0
        self.last_cache_log_time = 0
        self.last_diag_log_time = 0
        self.last_detection_time = TimeUtils.now_as_float()
        
        # Cache statistics
        self.cache_hits = 0
        self.total_attempts = 0
        self.cache_attempts = 0  # Track actual cache attempts separately
        
        # Bridge for image conversion
        self.cv_bridge = CvBridge()
        
        # Initialize the ground position filter
        self.position_filter = GroundPositionFilter()
        
        # Ball position and direction tracking
        self.ball_position_log_interval = 5.0  # Log every 5 seconds
        self.most_recent_detection = {
            'position': None,
            'timestamp': 0.0,
            'source': None
        }
        self.last_ball_position_log = 0.0
        self.detection_log_frequency = 5  # Log every X successful detections
        self.detection_log_frequency = 5  # Log every X successful detections
        self.ball_position_log_min_interval = 0.5  # Minimum time between logs to prevent spam
        self.consecutive_successful_detections = 0  # Counter for successful detections
        
        # Depth-specific settings
        self.use_temporal_blending = DEPTH_CONFIG.get("temporal_blending", 0.7)
        self.use_dynamic_sampling = True
        self.use_neighbor_data = True
        self.historical_fallback_always = True
        self.min_points_threshold = 3
        
        # Region-based depth tracking
        self.region_grid_size = 20
        self.depth_history_max_age = DEPTH_CONFIG.get("history_max_age", 5.0)  # Now 5.0
        
        # Define ROI size limits
        self.max_roi_size = DEPTH_CONFIG.get("max_roi_size", 60)
        self.min_roi_size = DEPTH_CONFIG.get("min_roi_size", 15)
        self.roi_size = 20  # Default ROI size
        
        # Path statistics
        self.path_counts = {'direct': 0, 'circular': 0, 'roi': 0, 'fallback': 0}
        
        # Initialize quality tracking
        self.quality_preference = DEPTH_CONFIG.get("quality_preference", 0.7)
        
        # Frame IDs
        self.depth_camera_frame = "ascamera_camera_link_0"
        self.detection_camera_frame = "ascamera_color_0"
        
        # Bounding box data
        self.yolo_bbox_data = {
            'width': 0.0,
            'height': 0.0,
            'timestamp': 0.0
        }
        
        # Track incoming YOLO detections (NEW)
        self.yolo_detections_received = 0
        self.yolo_detections_processed = 0
        self.last_yolo_received_time = 0
        
        # Track detection locations for adaptive frame skipping
        self.detection_locations = {}
        
        # Track consecutive no-depth frames
        self.consecutive_no_depth_frames = 0
        self.last_frame_had_depth = False
    
    def _setup_memory_pools(self):
        """Set up memory pools for frequently created objects."""
        # Create a pool of PointStamped messages
        self.point_pool = []
        for _ in range(10):  # Pre-allocate 10 points
            self.point_pool.append(PointStamped())
        self.point_pool_index = 0
        
        # Create a pool of Matrix4x4 for transforms
        self.matrix_pool = []
        for _ in range(5):  # Pre-allocate 5 matrices
            self.matrix_pool.append(Matrix4x4())
        self.matrix_pool_index = 0
        
        # Setup TTL dictionaries for various caches
        self.transform_cache = TTLDict(default_ttl=1.0)  # REDUCED from 5.0 to 1.0
        self.depth_region_cache = TTLDict(default_ttl=0.5)  # REDUCED from 1.0 to 0.5
        self.depth_region_stats = TTLDict(default_ttl=30.0)  # 30 second default TTL
        self.depth_history = TTLDict(default_ttl=DEPTH_CONFIG.get("history_max_age", 5.0))  # Now 5.0
        self.depth_stability_map = TTLDict(default_ttl=30.0)
        
        # Setup sequence tracking with TTL
        self.depth_sequence_by_region = TTLDict(default_ttl=10.0)
        
        # Setup buffer for depth values
        self.valid_depths_buffer = LightweightBuffer(64, init_with_none=False)  # INCREASED from 32
    
    def get_point_from_pool(self):
        """Get a PointStamped from the pool, initializing more if needed."""
        if not self.point_pool:
            # Expand pool if empty
            self.point_pool.append(PointStamped())
        
        # Get point and remove from pool
        point = self.point_pool.pop()
        
        # Clear the point data
        point.header.frame_id = ""
        point.header.stamp.sec = 0
        point.header.stamp.nanosec = 0
        point.point.x = 0.0
        point.point.y = 0.0
        point.point.z = 0.0
        
        return point
    
    def return_point_to_pool(self, point):
        """Return a PointStamped to the pool."""
        # Only keep a reasonable number in the pool
        if len(self.point_pool) < 20:
            self.point_pool.append(point)
    
    def get_matrix_from_pool(self):
        """Get a Matrix4x4 from the pool, initializing more if needed."""
        if not self.matrix_pool:
            # Expand pool if empty
            self.matrix_pool.append(Matrix4x4())
        
        # Get matrix and remove from pool
        matrix = self.matrix_pool.pop()
        
        # Reset to identity matrix
        matrix.data = np.eye(4, dtype=np.float32)
        
        return matrix
    
    def return_matrix_to_pool(self, matrix):
        """Return a Matrix4x4 to the pool."""
        # Only keep a reasonable number in the pool
        if len(self.matrix_pool) < 10:
            self.matrix_pool.append(matrix)
    
    def _preallocate_messages(self):
        """Pre-allocate message objects to reduce memory allocations."""
        # Create reusable message objects
        self.reusable_point = PointStamped()
        self.reusable_yolo_point = PointStamped()
        self.reusable_diag = String()
        self._filtered_msg_reuse = PointStamped()
    
    def _setup_callback_group(self):
        """Set up callback group and QoS profile for subscriptions."""
        # Single reentrant callback group
        self.callback_group = ReentrantCallbackGroup()
        
        # QoS profile with minimal buffer sizes
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1  # Minimal buffering
        )
    
    def _init_camera_parameters(self):
        """Initialize camera and detection parameters."""
        # Camera intrinsics (will be updated from camera_info)
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        
        # Depth image resolution (default values)
        self.depth_width = 640
        self.depth_height = 480
        
        # Coordinate scaling factors for YOLO (320x320) to depth camera (640x480)
        self.x_scale = 2.0
        self.y_scale = 2.0
    
    def _setup_tf2(self):
        """Set up tf2 components for coordinate transformations."""
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Add common reference frame
        self.reference_frame = COMMON_REFERENCE_FRAME["reference_frame"]
        self.transform_timeout = COMMON_REFERENCE_FRAME["transform_timeout"]
        
        # Schedule a check to verify transform is properly registered
        self.transform_check_timer = self.create_timer(2.0, self._verify_transform)
    
    def _setup_transform_cache(self):
        """Set up matrix-based transform cache for optimized transforms."""
        # TTL Dict already set up in _setup_memory_pools
        
        # Add matrix-specific variables
        self.use_matrix_transforms = PERFORMANCE_TIERS[self.current_tier]["use_matrix_transforms"]
        self.matrix_cache = TTLDict(default_ttl=PERFORMANCE_TIERS[self.current_tier]["cache_lifetime"])
        
        # Pre-compute common transforms if possible
        self.common_transforms_computed = False
        self.transform_check_timer_matrix = self.create_timer(5.0, self._cache_common_transforms)
    
    def _cache_common_transforms(self):
        """Cache commonly used transforms to avoid frequent lookups."""
        # Only try if we already have verified regular transforms
        if not self.verified_transform:
            return
            
        try:
            # Look up transform from reference frame to depth camera frame
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                self.depth_camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            
            # Convert to matrix and cache
            ref_to_depth_matrix = Matrix4x4.from_tf_transform(transform)
            self.matrix_cache.set("ref_to_depth", ref_to_depth_matrix)
            
            # Look up inverse transform
            transform = self.tf_buffer.lookup_transform(
                self.depth_camera_frame,
                self.reference_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            
            # Convert to matrix and cache
            depth_to_ref_matrix = Matrix4x4.from_tf_transform(transform)
            self.matrix_cache.set("depth_to_ref", depth_to_ref_matrix)
            
            # Look up transform between camera frames
            transform = self.tf_buffer.lookup_transform(
                self.depth_camera_frame,
                self.detection_camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            
            # Convert to matrix and cache
            depth_to_detect_matrix = Matrix4x4.from_tf_transform(transform)
            self.matrix_cache.set("depth_to_detect", depth_to_detect_matrix)
            
            # Also cache the inverse
            transform = self.tf_buffer.lookup_transform(
                self.detection_camera_frame,
                self.depth_camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            
            # Convert to matrix and cache
            detect_to_depth_matrix = Matrix4x4.from_tf_transform(transform)
            self.matrix_cache.set("detect_to_depth", detect_to_depth_matrix)
            
            # Also add transform from detection to reference frame (NEW)
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                self.detection_camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.2)
            )
            
            # Convert to matrix and cache
            detect_to_ref_matrix = Matrix4x4.from_tf_transform(transform)
            self.matrix_cache.set("detect_to_ref", detect_to_ref_matrix)
            
            # Mark as computed and cancel timer
            self.common_transforms_computed = True
            self.transform_check_timer_matrix.cancel()
            
            self.get_logger().info("Cached common transforms as matrices")
            
        except Exception as e:
            # Just try again later
            pass
    
    def _verify_transform(self):
        """Verify transform is registered and cancel verification timer if successful."""
        try:
            # Check transform between reference frame and depth camera frame
            if self.tf_buffer.can_transform(
                self.reference_frame,
                self.depth_camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            ) and self.tf_buffer.can_transform(
                self.depth_camera_frame,
                self.detection_camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=0.1)
            ):
                self.verified_transform = True
                self.get_logger().info(f"Transform verification successful between {self.reference_frame}, {self.depth_camera_frame}, and {self.detection_camera_frame}")
                self.transform_check_timer.cancel()
                return
        except Exception:
            pass
            
        # If transform is not ready, log warning
        if not self.transform_not_verified_logged:
            self.get_logger().warning(f"Transform not yet available between required frames: {self.reference_frame}, {self.depth_camera_frame}, and {self.detection_camera_frame}")
            self.transform_not_verified_logged = True
    
    def _setup_subscriptions(self):
        """Set up all subscriptions for this node."""
        # Subscribe to camera calibration information
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            TOPICS["input"]["camera_info"],
            self.camera_info_callback,
            10,
            callback_group=self.callback_group
        )
        
        # Subscribe to depth image with optimized QoS
        self.depth_sub = self.create_subscription(
            Image,
            TOPICS["input"]["depth_image"],
            self.depth_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )
        
        # Subscribe to YOLO ball detections
        self.yolo_sub = self.create_subscription(
            PointStamped,
            TOPICS["input"]["yolo_detection"],
            self.yolo_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )
        
        # Subscribe to YOLO bounding box information (with fallback)
        bbox_topic = TOPICS["input"].get("yolo_bbox", "/basketball/yolo/bbox")
        self.yolo_bbox_subscription = self.create_subscription(
            Float32MultiArray,
            bbox_topic,
            self.yolo_bbox_callback,
            self.qos_profile,
            callback_group=self.callback_group
        )
    
    def _setup_publishers(self):
        """Set up all publishers for this node."""
        # YOLO 3D position publisher
        self.yolo_3d_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["yolo_3d"],
            10
        )
        
        # Combined publisher (for backward compatibility)
        self.position_publisher = self.create_publisher(
            PointStamped,
            TOPICS["output"]["combined"],
            10
        )
        
        # Diagnostics publisher
        self.system_diagnostics_publisher = self.create_publisher(
            String,
            "/basketball/depth_camera/diagnostics",
            10
        )
        
        # Error publisher
        self.error_publisher = self.create_publisher(
            String,
            "/basketball/depth_camera/errors",
            10
        )
    
    def _setup_resource_monitoring(self):
        """Set up resource monitoring with adaptive performance adjustment."""
        # Initialize the resource monitor with reduced frequency
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=45.0,  # 45 seconds between updates
            enable_temperature=False
        )
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.resource_monitor.start()
        
        # Start performance metrics
        self.performance_metrics.start_monitoring(interval=1.0)
    
    def _setup_performance_timers(self):
        """Set up timers for performance adjustment and diagnostics."""
        # Performance adjustment timer
        self.performance_timer = self.create_timer(15.0, self._adjust_performance)
        
        # Diagnostics timer
        self.diagnostics_timer = self.create_timer(30.0, self.publish_system_diagnostics)
        
        # Cache cleanup timer
        self.cache_cleanup_timer = self.create_timer(60.0, self._cleanup_all_caches)
    
    def _cleanup_all_caches(self):
        """Perform cleanup of all caches to prevent memory growth."""
        # Force cleanup of all TTL dictionaries
        self.transform_cache.cleanup_all()
        self.depth_region_cache.cleanup_all()
        self.depth_region_stats.cleanup_all()
        self.depth_history.cleanup_all()
        self.depth_stability_map.cleanup_all()
        self.depth_sequence_by_region.cleanup_all()
        self.error_last_logged.cleanup_all()
        
        # Report memory usage
        mem = psutil.virtual_memory()
        self.get_logger().info(f"Cache cleanup complete. Memory: {mem.percent}% used, {mem.available / (1024*1024):.1f} MB available")
    
    def _log_initialization(self):
        """Log initialization status."""
        self.get_logger().info("Pi-Optimized 3D Position Estimator initialized")
        self.get_logger().info(f"Performance tier: {self.current_tier}")
        self.get_logger().info(f"Processing 1 in {self.process_every_n_frames} frames")
        
        # Force flush logs
        sys.stdout.flush()

    """
    This section contains the core detection algorithms for the OptimizedPositionEstimator class.
    These algorithms handle depth processing and 3D position estimation.
    """

    def _process_detection(self, msg, source):
        """
        Ultra-optimized detection processing with distance-specific handling.
        
        Args:
            msg: PointStamped detection message
            source: Source of detection ('YOLO')
        """
        # Skip if we don't have depth data yet
        if self.depth_array is None or self.camera_info is None:
            if self.debug_mode:
                self.get_logger().warning("Skipping detection: No depth data or camera info available")
            return
            
        # Skip if transform not verified
        if not self.verified_transform:
            if self.debug_mode:
                self.get_logger().warning("Skipping detection: Transform not verified")
            return
            
        # Apply scaling to the detection coordinates from YOLO 320x320 to camera 640x480
        scaled_msg = msg
        if source == 'YOLO':
            # Get a point from pool
            scaled_msg = self.get_point_from_pool()
            # Apply scaling factors to convert from YOLO coordinates to camera coordinates
            scaled_msg.header = msg.header
            scaled_msg.point.x = msg.point.x * self.x_scale
            scaled_msg.point.y = msg.point.y * self.y_scale
            scaled_msg.point.z = msg.point.z
            
        # Track detection timestamp for adaptive processing
        current_time = time.time()
        self.last_detection_time = current_time
        
        # Track detection position and movement for adaptive frame skipping
        if hasattr(self, 'detection_locations'):
            current_pos = (msg.point.x, msg.point.y)
            
            if source in self.detection_locations:
                # Calculate movement magnitude
                prev_pos = self.detection_locations[source]['position']
                dx = current_pos[0] - prev_pos[0]
                dy = current_pos[1] - prev_pos[1]
                movement = (dx**2 + dy**2)**0.5
                
                # Update movement info
                self.detection_locations[source] = {
                    'position': current_pos,
                    'time': current_time,
                    'movement': movement
                }
            else:
                # First detection for this source
                self.detection_locations[source] = {
                    'position': current_pos,
                    'time': current_time,
                    'movement': 0.0
                }
        
        # Check cache first for performance
        if self._check_position_cache(scaled_msg, source):
            # Return scaled message to pool if we used it
            if source == 'YOLO' and scaled_msg != msg:
                self.return_point_to_pool(scaled_msg)
            
            # Record that we processed this detection, even if from cache
            if source == 'YOLO':
                self.yolo_detections_processed += 1
            
            return
            
        # Process detection with distance-optimized approach
        success = self._get_3d_position(scaled_msg, source)
        
        # Record processing result
        if source == 'YOLO':
            self.yolo_detections_processed += 1
            
            # Log if processing failed
            if not success:
                self.yolo_detections_failed += 1
                failure_msg = f"Failed to process YOLO detection at ({msg.point.x:.2f}, {msg.point.y:.2f})"
                self.log_error(failure_msg, is_warning=True, detection_error=True)
            
        # Return scaled message to pool if we used it
        if source == 'YOLO' and scaled_msg != msg:
            self.return_point_to_pool(scaled_msg)

    def _ultra_fast_depth(self, pixel_x, pixel_y):
        """
        Highly optimized depth processing with distance-specific strategies.
        Prioritizes direct and circular methods that are proven to work.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            
        Returns:
            tuple: (depth, valid_points)
        """
        try:
            # Reset valid_depths_buffer (more efficient than creating new list)
            valid_depths_buffer = self.valid_depths_buffer
            valid_depths_buffer.clear()
            
            # First try direct center pixel for closest ranges (0.00-1.00m)
            direct_depth = 0.0
            try:
                d = self.depth_array[pixel_y, pixel_x]
                if d > 0:
                    scaled = d * self._scale_factor
                    if self._min_valid_depth < scaled < self._max_valid_depth:
                        direct_depth = scaled
                        
                        # Update depth tracking for direct pixel case
                        self.last_reported_depth['raw'] = scaled
                        self.last_reported_depth['processed'] = scaled
                        self.last_reported_depth['source'] = 'direct'
                        self.last_reported_depth['timestamp'] = time.time()
                        
                        self.path_counts['direct'] += 1
                        self._store_depth_history(pixel_x, pixel_y, scaled, 1)
                        return scaled, 1
            except IndexError:
                # Handle edge case of invalid pixel coordinates
                pass
            
            # Get region key for this position
            region_key = self._get_region_key(pixel_x, pixel_y)
            
            # Get historical depth if available to determine sampling strategy
            historical_depth = None
            tier_name = "medium"  # Default tier
            sampling_radius = 5   # Default radius
            
            # Determine best sampling strategy based on history
            if region_key in self.depth_history:
                entry = self.depth_history.get(region_key)
                if entry and 'depth' in entry:
                    historical_depth = entry['depth']
                    
                    # Determine distance tier
                    for name, tier in DISTANCE_TIERS.items():
                        if tier['range'][0] <= historical_depth <= tier['range'][1]:
                            tier_name = name
                            sampling_radius = tier['sampling_radius']
                            break
            
            # Sample with adaptive pattern based on distance tier
            if tier_name == "close":
                # For close objects, use tight circular pattern
                return self._sample_close_range(pixel_x, pixel_y, sampling_radius, valid_depths_buffer)
            elif tier_name == "medium":
                # For medium range, use standard circular pattern
                return self._sample_medium_range(pixel_x, pixel_y, sampling_radius, valid_depths_buffer)
            elif tier_name == "far":
                # For far range (problematic zone), use enhanced sampling
                return self._sample_far_range(pixel_x, pixel_y, sampling_radius, valid_depths_buffer, historical_depth)
            else:  # very_far
                # For very far objects, use maximum sampling
                return self._sample_very_far_range(pixel_x, pixel_y, sampling_radius, valid_depths_buffer, historical_depth)
            
        except Exception as e:
            if self.debug_depth:
                self.get_logger().error(f"DEPTH DEBUG: Exception in depth processing: {str(e)}")
            return 1.5, 0  # Default on error

    def _sample_close_range(self, pixel_x, pixel_y, radius, valid_depths_buffer):
        """
        Sampling strategy optimized for close range (0-1m).
        Uses tight circular sampling with small radius.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            radius: Sampling radius
            valid_depths_buffer: Buffer to store valid depths
            
        Returns:
            tuple: (depth, valid_points)
        """
        # For close range, center pixel is usually reliable, but we'll sample a few more
        # Sample using tight circle pattern
        for r in range(1, radius+1):
            # Sample 4 cardinal directions for each radius
            for angle in [0, 90, 180, 270]:
                dx = int(r * math.cos(math.radians(angle)))
                dy = int(r * math.sin(math.radians(angle)))
                
                x = pixel_x + dx
                y = pixel_y + dy
                
                # Check bounds
                if 0 <= x < self.depth_array.shape[1] and 0 <= y < self.depth_array.shape[0]:
                    d = self.depth_array[y, x]
                    if d > 0:
                        scaled = d * self._scale_factor
                        if self._min_valid_depth < scaled < self._max_valid_depth:
                            valid_depths_buffer.append(scaled)
        
        # If we found enough valid depths, use them
        count = len(valid_depths_buffer)
        if count >= DISTANCE_TIERS["close"]["min_points"]:
            # Close range is very reliable, use median
            raw_depths = [d for d in valid_depths_buffer]
            depth = np.median(raw_depths)
            raw_depth = depth  # For close range, raw and processed are the same
            
            # Update depth tracking
            self.last_reported_depth['raw'] = raw_depth
            self.last_reported_depth['processed'] = depth
            self.last_reported_depth['source'] = 'close'
            self.last_reported_depth['timestamp'] = time.time()
            
            self.path_counts['circular'] += 1
            self._store_depth_history(pixel_x, pixel_y, depth, count)
            return depth, count
        
        # Try historical data or fallback
        return self._fallback_depth(pixel_x, pixel_y, valid_depths_buffer)

    def _sample_medium_range(self, pixel_x, pixel_y, radius, valid_depths_buffer):
        """
        Sampling strategy optimized for medium range (1-1.75m).
        Uses standard circular sampling pattern.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            radius: Sampling radius
            valid_depths_buffer: Buffer to store valid depths
            
        Returns:
            tuple: (depth, valid_points)
        """
        # Middle range uses standard circular pattern with 8 directions
        for r in range(1, radius+1):
            # Sample 8 points around the circle (45° increments)
            for angle in range(0, 360, 45):
                dx = int(r * math.cos(math.radians(angle)))
                dy = int(r * math.sin(math.radians(angle)))
                
                x = pixel_x + dx
                y = pixel_y + dy
                
                # Check bounds
                if 0 <= x < self.depth_array.shape[1] and 0 <= y < self.depth_array.shape[0]:
                    d = self.depth_array[y, x]
                    if d > 0:
                        scaled = d * self._scale_factor
                        if self._min_valid_depth < scaled < self._max_valid_depth:
                            valid_depths_buffer.append(scaled)
        
        # If we found enough valid depths, use them
        count = len(valid_depths_buffer)
        if count >= DISTANCE_TIERS["medium"]["min_points"]:
            # Medium range - use median for stability
            raw_depths = [d for d in valid_depths_buffer]
            depth = np.median(raw_depths)
            raw_depth = depth  # For medium range, raw and processed are typically the same
            
            # Update depth tracking
            self.last_reported_depth['raw'] = raw_depth
            self.last_reported_depth['processed'] = depth
            self.last_reported_depth['source'] = 'medium'
            self.last_reported_depth['timestamp'] = time.time()
            
            self.path_counts['circular'] += 1
            self._store_depth_history(pixel_x, pixel_y, depth, count)
            return depth, count
        
        # Try historical data or fallback
        return self._fallback_depth(pixel_x, pixel_y, valid_depths_buffer)

    def _sample_far_range(self, pixel_x, pixel_y, radius, valid_depths_buffer, historical_depth=None):
        """
        Sampling strategy optimized for far range (1.75-2.75m) - the problematic zone.
        Uses enhanced sampling with adaptive depth ranges.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            radius: Sampling radius
            valid_depths_buffer: Buffer to store valid depths
            historical_depth: Historical depth value if available
            
        Returns:
            tuple: (depth, valid_points)
        """
        # For far range (problematic zone), use enhanced sampling with bigger pattern
        # and more points
        
        # First standard circular - with denser angular sampling (IMPROVED)
        for r in range(1, radius+1):
            # Sample 12 points around the circle (30° increments) - INCREASED from 8 points
            for angle in range(0, 360, 30):  # Changed from 45 to 30 degrees
                dx = int(r * math.cos(math.radians(angle)))
                dy = int(r * math.sin(math.radians(angle)))
                
                x = pixel_x + dx
                y = pixel_y + dy
                
                # Check bounds
                if 0 <= x < self.depth_array.shape[1] and 0 <= y < self.depth_array.shape[0]:
                    d = self.depth_array[y, x]
                    if d > 0:
                        scaled = d * self._scale_factor
                        if self._min_valid_depth < scaled < self._max_valid_depth:
                            valid_depths_buffer.append(scaled)
        
        # For problematic range, add additional scan lines across the region
        # to increase chances of valid depth - with finer sampling (IMPROVED)
        for offset in range(-radius, radius+1, 1):  # Changed from 2 to 1 for finer sampling
            # Horizontal line
            y = pixel_y + offset
            if 0 <= y < self.depth_array.shape[0]:
                for x in range(max(0, pixel_x-radius), min(self.depth_array.shape[1], pixel_x+radius+1), 1):  # Changed from 2 to 1
                    d = self.depth_array[y, x]
                    if d > 0:
                        scaled = d * self._scale_factor
                        if self._min_valid_depth < scaled < self._max_valid_depth:
                            valid_depths_buffer.append(scaled)
            
            # Vertical line
            x = pixel_x + offset
            if 0 <= x < self.depth_array.shape[1]:
                for y in range(max(0, pixel_y-radius), min(self.depth_array.shape[0], pixel_y+radius+1), 1):  # Changed from 2 to 1
                    d = self.depth_array[y, x]
                    if d > 0:
                        scaled = d * self._scale_factor
                        if self._min_valid_depth < scaled < self._max_valid_depth:
                            valid_depths_buffer.append(scaled)
        
        # If we found enough valid depths, use them
        count = len(valid_depths_buffer)
        min_required = DISTANCE_TIERS["far"]["min_points"]
        
        if count >= min_required:
            # Far range is less reliable, filter outliers and use mean
            if count > min_required + 2:
                # If we have enough points, remove outliers before calculating
                depths = np.array([d for d in valid_depths_buffer])
                mean = np.mean(depths)
                std = np.std(depths)
                # Keep depths within 2 standard deviations
                filtered_depths = depths[np.abs(depths - mean) <= 2 * std]
                if len(filtered_depths) >= min_required:
                    raw_depth = np.mean(filtered_depths)
                else:
                    raw_depth = np.mean(depths)
            else:
                # If we have just enough, use all points
                raw_depth = np.mean([d for d in valid_depths_buffer])
            
            # Apply correction factor for this range
            depth = raw_depth * DISTANCE_TIERS["far"]["fallback_scale"]
            
            # Update depth tracking
            self.last_reported_depth['raw'] = raw_depth
            self.last_reported_depth['processed'] = depth
            self.last_reported_depth['source'] = 'far'
            self.last_reported_depth['timestamp'] = time.time()
            
            self.path_counts['circular'] += 1
            self._store_depth_history(pixel_x, pixel_y, depth, count)
            return depth, count
        
        # Historical fallback with strong bias toward historical data
        # for this problematic range
        if historical_depth is not None:
            # Return historical depth with slight adjustment
            adjusted_depth = historical_depth * DISTANCE_TIERS["far"]["fallback_scale"]
            if self.debug_depth:
                self.get_logger().info(
                    f"DEPTH DEBUG: Using adjusted historical depth: {adjusted_depth:.3f}m "
                    f"(original: {historical_depth:.3f}m) for problematic range"
                )
            
            # Update depth tracking
            self.last_reported_depth['raw'] = historical_depth
            self.last_reported_depth['processed'] = adjusted_depth
            self.last_reported_depth['source'] = 'historical_far'
            self.last_reported_depth['timestamp'] = time.time()
            
            self.path_counts['fallback'] += 1
            return adjusted_depth, max(1, count)
        
        # Standard fallback
        return self._fallback_depth(pixel_x, pixel_y, valid_depths_buffer)

    def _sample_very_far_range(self, pixel_x, pixel_y, radius, valid_depths_buffer, historical_depth=None):
        """
        Sampling strategy optimized for very far range (2.75m+).
        Uses maximum sampling with fallback to historical data.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            radius: Sampling radius
            valid_depths_buffer: Buffer to store valid depths
            historical_depth: Historical depth value if available
            
        Returns:
            tuple: (depth, valid_points)
        """
        # For very far range, use dense sampling with large radius
        # and apply special scaling
        
        # Use 16-point pattern for each radius
        for r in range(1, radius+1):
            # Sample 16 points around the circle (22.5° increments) for dense sampling
            for angle in range(0, 360, 22):
                dx = int(r * math.cos(math.radians(angle)))
                dy = int(r * math.sin(math.radians(angle)))
                
                x = pixel_x + dx
                y = pixel_y + dy
                
                # Check bounds
                if 0 <= x < self.depth_array.shape[1] and 0 <= y < self.depth_array.shape[0]:
                    d = self.depth_array[y, x]
                    if d > 0:
                        scaled = d * self._scale_factor
                        if self._min_valid_depth < scaled < self._max_valid_depth:
                            valid_depths_buffer.append(scaled)
        
        # For very far range, also sample a square region
        extended_radius = radius + 2  # Extended radius for very far objects
        for y in range(max(0, pixel_y-extended_radius), min(self.depth_array.shape[0], pixel_y+extended_radius+1), 1): # Changed from 2 to 1
            for x in range(max(0, pixel_x-extended_radius), min(self.depth_array.shape[1], pixel_x+extended_radius+1), 1): # Changed from 2 to 1
                d = self.depth_array[y, x]
                if d > 0:
                    scaled = d * self._scale_factor
                    if self._min_valid_depth < scaled < self._max_valid_depth:
                        valid_depths_buffer.append(scaled)
        
        # If we found enough valid depths, use them
        count = len(valid_depths_buffer)
        min_required = DISTANCE_TIERS["very_far"]["min_points"]
        
        if count >= min_required:
            # Far range is unreliable, use robust statistics
            depths = np.array([d for d in valid_depths_buffer])
            
            # For very far range, use median of trimmed array for stability
            depths = np.sort(depths)
            # Trim 10% from both ends if we have enough points
            if len(depths) > 15:
                trim_size = int(len(depths) * 0.1)
                depths = depths[trim_size:-trim_size]
            
            raw_depth = np.median(depths)
            
            # Apply correction factor for very far range
            depth = raw_depth * DISTANCE_TIERS["very_far"]["fallback_scale"]
            
            # Update depth tracking
            self.last_reported_depth['raw'] = raw_depth
            self.last_reported_depth['processed'] = depth
            self.last_reported_depth['source'] = 'very_far'
            self.last_reported_depth['timestamp'] = time.time()
            
            self.path_counts['circular'] += 1
            self._store_depth_history(pixel_x, pixel_y, depth, count)
            return depth, count
        
        # Historical fallback for very far range
        if historical_depth is not None:
            # Return historical depth with adjustment
            adjusted_depth = historical_depth * DISTANCE_TIERS["very_far"]["fallback_scale"]
            if self.debug_depth:
                self.get_logger().info(
                    f"DEPTH DEBUG: Using adjusted historical depth: {adjusted_depth:.3f}m "
                    f"(original: {historical_depth:.3f}m) for very far range"
                )
            self.path_counts['fallback'] += 1
            return adjusted_depth, max(1, count)
        
        # Last resort for very far objects - use larger default
        if count > 0:
            # Use average of what we have, even if insufficient
            depth = np.mean([d for d in valid_depths_buffer])
            depth *= DISTANCE_TIERS["very_far"]["fallback_scale"]
            return depth, count
        else:
            # Default value based on histogram analysis
            self.path_counts['fallback'] += 1
            return 3.0, 0  # Default depth for very long range

    def _fallback_depth(self, pixel_x, pixel_y, valid_depths_buffer):
        """
        Fallback depth estimation when primary methods fail.
        Tries historical data and reasonable defaults.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            valid_depths_buffer: Buffer with any valid depths found
            
        Returns:
            tuple: (depth, valid_points)
        """
        count = len(valid_depths_buffer)
        region_key = self._get_region_key(pixel_x, pixel_y)
        
        # Fallback 1: Use any valid depths we found, even if fewer than threshold
        if count > 0:
            depth = np.median([d for d in valid_depths_buffer])
            
            # Update depth tracking
            self.last_reported_depth['raw'] = depth
            self.last_reported_depth['processed'] = depth
            self.last_reported_depth['source'] = 'fallback_partial'
            self.last_reported_depth['timestamp'] = time.time()
            
            self.path_counts['fallback'] += 1
            self._store_depth_history(pixel_x, pixel_y, depth, count)
            return depth, count
        
        # Fallback 2: Use historical data for this region
        if region_key in self.depth_history:
            entry = self.depth_history.get(region_key)
            if entry:
                current_time = time.time()
                age = current_time - entry.get('timestamp', 0)
                
                if age < self.depth_history_max_age:
                    self.path_counts['fallback'] += 1
                    return entry.get('depth'), entry.get('valid_points', 1)
        
        # Fallback 3: Try neighboring regions
        historical_depth, historical_points = self._get_historical_depth_anywhere(pixel_x, pixel_y)
        if historical_depth is not None:
            self.path_counts['fallback'] += 1
            return historical_depth, historical_points
        
        # Last resort: Use default depth
        # For basketball tracking, provide reasonable default
        self.path_counts['fallback'] += 1
        return 1.75, 0  # Default depth is mid-range based on histogram analysis

    def _get_3d_position(self, msg, source):
        """
        Enhanced 3D position estimation with improved handling for all distance ranges.
        
        Args:
            msg: Detection message with 2D coordinates
            source: Detection source ('YOLO')
            
        Returns:
            bool: Success status
        """
        try:
            # Skip if we don't have depth data yet
            if self.depth_array is None or self.camera_info is None:
                return False
                
            # Skip if transform not verified
            if not self.verified_transform:
                return False
            
            # Verify we can transform between detection frame and depth frame
            if not self._transform_detection_to_depth_frame(msg):
                return False
                
            # Get 2D coordinates from detection (in detection frame)
            orig_x = float(msg.point.x)
            orig_y = float(msg.point.y)
            
            # Check if we have a YOLO bounding box for this detection
            estimated_3d_point = None
            if source == "YOLO" and hasattr(self, 'yolo_bbox_data') and self.yolo_bbox_data.get('timestamp', 0) > time.time() - 1.0:
                bbox_width = self.yolo_bbox_data.get('width', 0)
                bbox_height = self.yolo_bbox_data.get('height', 0)
                
                if bbox_width > 0 and bbox_height > 0:
                    estimated_3d_point = self._estimate_3d_from_2d(msg, bbox_width, bbox_height)
                    
                    if estimated_3d_point is not None and self.debug_mode:
                        self.get_logger().info(
                            f"Using estimated 3D position from YOLO 2D: "
                            f"({estimated_3d_point[0]:.2f}, {estimated_3d_point[1]:.2f}, {estimated_3d_point[2]:.2f})"
                        )
            
            # First transform these coordinates to 3D in the detection camera frame
            # using a standard depth or assumed Z value (1.0m)
            assumed_z = 1.0  # Assumed depth for creating initial 3D point
            
            # Use detection camera intrinsics
            detection_x = (orig_x - self.cx) * assumed_z / self.fx
            detection_y = (orig_y - self.cy) * assumed_z / self.fy
            detection_z = assumed_z
            
            # Create a point in 3D space using detection camera frame
            detection_3d = self.get_point_from_pool()
            detection_3d.header.frame_id = self.detection_camera_frame
            detection_3d.header.stamp = self.get_clock().now().to_msg()
            detection_3d.point.x = detection_x
            detection_3d.point.y = detection_y
            detection_3d.point.z = detection_z
            
            # Transform to depth camera frame
            depth_frame_point = self._transform_to_depth_frame(detection_3d)
            # Return detection_3d to pool
            self.return_point_to_pool(detection_3d)
            
            if depth_frame_point is None:
                return False
            
            # Project the 3D point in depth camera frame back to depth camera's 2D space
            # This gives us the correct pixel coordinates in the depth image
            proj_x = depth_frame_point.point.x / depth_frame_point.point.z * self.fx + self.cx
            proj_y = depth_frame_point.point.y / depth_frame_point.point.z * self.fy + self.cy
            
            # Return depth_frame_point to pool if we're done with it
            self.return_point_to_pool(depth_frame_point)
            
            # Round to integers for pixel lookup
            pixel_x = int(round(proj_x))
            pixel_y = int(round(proj_y))
            
            # Constrain to valid image bounds
            depth_height, depth_width = self.depth_array.shape
            margin = 10
            
            pixel_x = max(margin, min(pixel_x, depth_width - margin - 1))
            pixel_y = max(margin, min(pixel_y, depth_height - margin - 1))
            
            if self.debug_depth:
                self.get_logger().info(f"DEPTH DEBUG: Mapped to ({pixel_x}, {pixel_y}) in {depth_width}x{depth_height} image")
                sys.stdout.flush()
            
            # Get depth using distance-optimized estimation
            median_depth, valid_points = self._ultra_fast_depth(pixel_x, pixel_y)
            
            # Check if estimated 3D point is available from YOLO
            # Use it as a reference to validate or replace the depth measurement
            if estimated_3d_point is not None:
                # Calculate reference depth from estimated 3D point
                estimated_distance = np.linalg.norm(estimated_3d_point)
                
                # If depth measurement is poor quality or differs significantly from estimate
                # (IMPROVED threshold check)
                if valid_points < 3 or abs(median_depth - estimated_distance) > estimated_distance * 0.25:  # Reduced from 0.3 to 0.25
                    if self.debug_mode:
                        self.get_logger().info(
                            f"Using YOLO estimate instead of depth: {median_depth:.2f}m vs {estimated_distance:.2f}m "
                            f"(Valid points: {valid_points})"
                        )
                    # Use the estimated distance instead
                    median_depth = estimated_distance
                    valid_points = max(valid_points, 5)  # Ensure reasonable quality score
                    
                    # Create the 3D position message directly from the estimated point
                    reference_position_msg = self.get_point_from_pool()
                    reference_position_msg.header.stamp = self.get_clock().now().to_msg()
                    reference_position_msg.header.frame_id = self.reference_frame
                    reference_position_msg.point.x = float(estimated_3d_point[0])
                    reference_position_msg.point.y = float(estimated_3d_point[1]) 
                    reference_position_msg.point.z = float(estimated_3d_point[2])
                    
                    # Transform to depth camera frame
                    depth_frame_msg = self._transform_to_depth_frame(reference_position_msg)
                    # Return reference_position_msg to pool
                    self.return_point_to_pool(reference_position_msg)
                    
                    if depth_frame_msg is not None:
                        # Use this as our final position
                        position_msg = depth_frame_msg
                        
                        # Publish
                        source_msg = self.reusable_yolo_point if source == 'YOLO' else self.reusable_point
                        source_msg.header = position_msg.header
                        source_msg.point = position_msg.point
                        
                        # Publish to source-specific topic
                        if source == 'YOLO':
                            self.yolo_3d_publisher.publish(source_msg)
                        
                        # Also publish to combined topic
                        self.position_publisher.publish(source_msg)
                        
                        # Update cache and statistics
                        self.detection_cache[source]['timestamp'] = time.time()
                        if self.detection_cache[source]['detection_2d'] is None:
                            self.detection_cache[source]['detection_2d'] = self.get_point_from_pool()
                        self.detection_cache[source]['detection_2d'].header = msg.header
                        self.detection_cache[source]['detection_2d'].point = msg.point
                        
                        if self.detection_cache[source]['position_3d'] is None:
                            self.detection_cache[source]['position_3d'] = self.get_point_from_pool()
                        self.detection_cache[source]['position_3d'].header = position_msg.header
                        self.detection_cache[source]['position_3d'].point = position_msg.point
                        
                        # Return depth_frame_msg to pool
                        self.return_point_to_pool(position_msg)
                        
                        self.successful_conversions += 1
                        return True
            
            # Better handling of poor quality cases
            if valid_points == 0:
                # Look for cached depth values in any recent history
                historical_depth, historical_points = self._get_historical_depth_anywhere(pixel_x, pixel_y)
                if historical_depth is not None:
                    median_depth = historical_depth
                    valid_points = historical_points
                else:
                    # Last resort: use a reasonable default depth
                    median_depth = 2.3  # Based on frequently occurring value in logs
                    valid_points = 1
            
            # Record detection quality metrics
            detection_quality = "good" if valid_points >= 8 else "fair" if valid_points >= 3 else "poor"
            
            # Convert to 3D using the pinhole camera model
            x = float((pixel_x - self.cx) * median_depth / self.fx)
            y = float((pixel_y - self.cy) * median_depth / self.fy)
            z = float(median_depth)
            
            # Create the 3D position message in depth camera frame
            camera_position_msg = self.get_point_from_pool()
            camera_position_msg.header.stamp = self.get_clock().now().to_msg()
            camera_position_msg.header.frame_id = self.depth_camera_frame
            camera_position_msg.point.x = x
            camera_position_msg.point.y = y
            camera_position_msg.point.z = z
            
            # Transform position to common reference frame
            transformed_msg = self._fast_transform(camera_position_msg)
            # Return camera_position_msg to pool
            self.return_point_to_pool(camera_position_msg)
            
            if transformed_msg is None:
                return False
            
            # Apply position filtering using shared ground position filter
            position = (transformed_msg.point.x, transformed_msg.point.y, transformed_msg.point.z)
            filtered_position = self._filter_position(position)
            
            # Create message with filtered position (reusing message objects)
            filtered_msg = self._filtered_msg_reuse
            filtered_msg.header = transformed_msg.header
            filtered_msg.point.x = filtered_position[0]
            filtered_msg.point.y = filtered_position[1]
            filtered_msg.point.z = filtered_position[2]
            
            # Return transformed_msg to pool
            if transformed_msg != self.reusable_point and transformed_msg != self.reusable_yolo_point:
                self.return_point_to_pool(transformed_msg)
            
            # Transform back to depth camera frame for consistent output
            depth_frame_msg = self._transform_to_depth_frame(filtered_msg)
            if depth_frame_msg is None:
                return False
            
            # Publish using reusable messages
            source_msg = self.reusable_yolo_point if source == 'YOLO' else self.reusable_point
            source_msg.header = depth_frame_msg.header
            source_msg.point = depth_frame_msg.point
            
            # Publish to source-specific topic
            if source == 'YOLO':
                self.yolo_3d_publisher.publish(source_msg)
            
            # Also publish to combined topic
            self.position_publisher.publish(source_msg)
            
            # Return depth_frame_msg to pool if needed
            if depth_frame_msg != source_msg and depth_frame_msg != self._filtered_msg_reuse:
                self.return_point_to_pool(depth_frame_msg)
            
            # Update cache - store both 2D and 3D positions
            # IMPROVE: Only cache new positions if the current cache is empty or old
            current_time = time.time()
            if self.detection_cache[source]['timestamp'] == 0 or current_time - self.detection_cache[source]['timestamp'] > 0.1:  # Reduced from 0.5 to 0.1
                if self.detection_cache[source]['detection_2d'] is None:
                    self.detection_cache[source]['detection_2d'] = self.get_point_from_pool()
                
                # Copy data to existing object instead of creating new ones
                self.detection_cache[source]['detection_2d'].header = msg.header
                self.detection_cache[source]['detection_2d'].point = msg.point
                
                if self.detection_cache[source]['position_3d'] is None:
                    self.detection_cache[source]['position_3d'] = self.get_point_from_pool()
                
                # Copy data to existing object
                self.detection_cache[source]['position_3d'].header = source_msg.header
                self.detection_cache[source]['position_3d'].point = source_msg.point
                self.detection_cache[source]['timestamp'] = current_time
            
            # Count successful conversion
            self.successful_conversions += 1
            
            # Trigger position logging based on successful detections
            self.consecutive_successful_detections += 1
            if self.consecutive_successful_detections >= self.detection_log_frequency:
                current_time = time.time()
                if current_time - self.last_ball_position_log >= self.ball_position_log_min_interval:
                    self._log_ball_position_and_direction()
                    self.consecutive_successful_detections = 0
            
            # Log based on detection_log_frequency setting
            if self.successful_conversions % self.detection_log_frequency == 0:
                self._update_fps()
                # Use 0.05 interval for faster CPU check
                actual_cpu = psutil.cpu_percent(interval=0.05)
                
                # Log in more detail for debugging
                if self.debug_mode:
                    # Path statistics
                    path_stats = (f"direct:{self.path_counts['direct']} "
                                f"circular:{self.path_counts['circular']} "
                                f"fallback:{self.path_counts['fallback']}")
                    
                    self.get_logger().info(
                        f"3D position ({source}): "
                        f"({filtered_position[0]:.2f}, {filtered_position[1]:.2f}, {filtered_position[2]:.2f})m | "
                        f"Depth: {median_depth:.2f}m | "
                        f"FPS: {self.current_fps:.1f} | "
                        f"Quality: {detection_quality} ({valid_points} points) | "
                        f"Paths: {path_stats} | "
                        f"CPU: {actual_cpu:.1f}%"
                    )
                else:
                    # Get raw depth value for reporting
                    raw_depth_value = self.last_reported_depth.get('raw', median_depth)
                    processed_depth = self.last_reported_depth.get('processed', median_depth)
                    depth_source = self.last_reported_depth.get('source', 'unknown')
                    
                    # Simplified log with enhanced depth information
                    self.get_logger().info(
                        f"3D position ({source}): "
                        f"({filtered_position[0]:.2f}, {filtered_position[1]:.2f}, {filtered_position[2]:.2f})m | "
                        f"Depth: {median_depth:.2f}m (raw: {raw_depth_value:.2f}m, source: {depth_source}) | "
                        f"Quality: {detection_quality} ({valid_points} points)"
                    )
                
                sys.stdout.flush()  # Force flush
            
            return True
        except Exception as e:
            self.log_error(f"Error in 3D conversion: {str(e)}")
            return False

    def _estimate_3d_from_2d(self, detection_msg, bbox_width, bbox_height):
        """
        Improved 3D position estimation with better long-range handling.
        Estimates a 3D position from a 2D detection and bbox dimensions.
        
        Args:
            detection_msg: Detection message with 2D coordinates
            bbox_width: Width of bounding box in pixels
            bbox_height: Height of bounding box in pixels
                
        Returns:
            np.ndarray: Estimated 3D position [x, y, z] or None if estimation fails
        """
        try:
            # Known basketball diameter in meters (standard basketball is ~9 inches / 22.86 cm)
            basketball_diameter_meters = 0.2286
            
            # Calculate distance based on apparent size vs actual size
            focal_length_pixels = 345.58  # Calibrated focal length for camera
            
            # Enhanced distance estimation with safeguards for large distances
            if bbox_width < 5:  # Very small detection, likely far away
                # Use a more conservative approach for small bounding boxes
                estimated_distance = 3.0  # Default to 3.0m for very distant balls
            else:
                estimated_distance = (basketball_diameter_meters * focal_length_pixels) / bbox_width
                
                # Apply distance-specific correction factors
                if estimated_distance < 1.0:
                    # Close range - minimal correction needed
                    correction = 1.0
                elif estimated_distance < 1.75:
                    # Medium range - slight correction (5% per meter)
                    correction = 1.0 + (estimated_distance - 1.0) * 0.05
                elif estimated_distance < 2.75:
                    # Far range (problem zone) - stronger correction (10% per meter)
                    correction = 1.0375 + (estimated_distance - 1.75) * 0.1
                else:
                    # Very far range - significant correction (15% per meter)
                    correction = 1.0375 + 0.1 + (estimated_distance - 2.75) * 0.15
                
                # Apply correction
                estimated_distance = estimated_distance / correction
            
            # Cap maximum distance to realistic range based on logs
            estimated_distance = min(estimated_distance, 3.5)
            
            # Get camera frame
            camera_frame = detection_msg.header.frame_id or self.detection_camera_frame
            
            # Get camera to reference frame transform
            # Try matrix transform first for efficiency
            if self.use_matrix_transforms and "detect_to_ref" in self.matrix_cache:
                matrix = self.matrix_cache.get("detect_to_ref")
                
                # Calculate ray from camera through detection
                ray_x = (detection_msg.point.x - self.cx) / self.fx
                ray_y = (detection_msg.point.y - self.cy) / self.fy
                ray_z = 1.0  # Forward direction in camera frame
                
                # Normalize ray
                magnitude = math.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
                if magnitude > 0:
                    ray_x /= magnitude
                    ray_y /= magnitude
                    ray_z /= magnitude
                
                # Get camera position in reference frame
                # This is in the last column of the transformation matrix
                camera_pos_x = matrix.data[0, 3]
                camera_pos_y = matrix.data[1, 3]
                camera_pos_z = matrix.data[2, 3]
                
                # Transform ray direction to reference frame
                ref_dir_x, ref_dir_y, ref_dir_z = matrix.transform_vector(ray_x, ray_y, ray_z)
                
                # Calculate 3D position by extending ray by estimated distance
                est_x = camera_pos_x + estimated_distance * ref_dir_x
                est_y = camera_pos_y + estimated_distance * ref_dir_y
                est_z = camera_pos_z + estimated_distance * ref_dir_z
                
                # For basketball tracking, we know the ball is always on the ground
                # Override est_z with reasonable value for basketball center height (half diameter)
                est_z = 0.12  # Basketball radius (~12cm) above ground
            else:
                # Fallback to standard transform
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.reference_frame,
                        camera_frame,
                        rclpy.time.Time(),
                        rclpy.duration.Duration(seconds=0.2)
                    )
                    
                    # Cache as matrix for future use
                    if self.use_matrix_transforms:
                        matrix = Matrix4x4.from_tf_transform(transform)
                        self.matrix_cache.set("detect_to_ref", matrix)
                    
                    # Extract camera position in reference frame
                    camera_pos_x = transform.transform.translation.x
                    camera_pos_y = transform.transform.translation.y
                    camera_pos_z = transform.transform.translation.z
                    
                    # Calculate direction vector using camera intrinsics
                    ray_x = (detection_msg.point.x - self.cx) / self.fx
                    ray_y = (detection_msg.point.y - self.cy) / self.fy
                    ray_z = 1.0  # Forward direction in camera frame
                    
                    # Normalize the direction vector
                    magnitude = math.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
                    if magnitude > 0:
                        ray_x /= magnitude
                        ray_y /= magnitude
                        ray_z /= magnitude
                    
                    # Extract rotation quaternion from transform
                    qx = transform.transform.rotation.x
                    qy = transform.transform.rotation.y
                    qz = transform.transform.rotation.z
                    qw = transform.transform.rotation.w
                    
                    # Convert quaternion to rotation matrix to rotate the ray
                    # Precompute common products
                    xx = qx * qx
                    xy = qx * qy
                    xz = qx * qz
                    xw = qx * qw
                    yy = qy * qy
                    yz = qy * qz
                    yw = qy * qw
                    zz = qz * qz
                    zw = qz * qw
                    
                    # Rotation matrix elements
                    r00 = 1 - 2 * (yy + zz)
                    r01 = 2 * (xy - zw)
                    r02 = 2 * (xz + yw)
                    r10 = 2 * (xy + zw)
                    r11 = 1 - 2 * (xx + zz)
                    r12 = 2 * (yz - xw)
                    r20 = 2 * (xz - yw)
                    r21 = 2 * (yz + xw)
                    r22 = 1 - 2 * (xx + yy)
                    
                    # Apply rotation to camera ray
                    ref_dir_x = r00 * ray_x + r01 * ray_y + r02 * ray_z
                    ref_dir_y = r10 * ray_x + r11 * ray_y + r12 * ray_z
                    ref_dir_z = r20 * ray_x + r21 * ray_y + r22 * ray_z
                    
                    # Normalize the rotated direction vector
                    magnitude = math.sqrt(ref_dir_x**2 + ref_dir_y**2 + ref_dir_z**2)
                    if magnitude > 0:
                        ref_dir_x /= magnitude
                        ref_dir_y /= magnitude
                        ref_dir_z /= magnitude
                    
                    # Calculate 3D position by extending ray by estimated distance
                    est_x = camera_pos_x + estimated_distance * ref_dir_x
                    est_y = camera_pos_y + estimated_distance * ref_dir_y
                    est_z = camera_pos_z + estimated_distance * ref_dir_z
                    
                    # For basketball tracking, we know the ball is always on the ground
                    # Override est_z with reasonable value for basketball center height (half diameter)
                    est_z = 0.12  # Basketball radius (~12cm) above ground
                    
                except Exception as e:
                    self.log_error(f"Transform lookup error in 3D estimation: {str(e)}", is_warning=True)
                    return None
            
            if self.debug_mode:
                self.get_logger().info(
                    f"Estimated 3D from YOLO 2D: distance={estimated_distance:.2f}m, "
                    f"pos=({est_x:.2f}, {est_y:.2f}, {est_z:.2f})"
                )
                
            return np.array([est_x, est_y, est_z])
            
        except Exception as e:
            self.log_error(f"Error estimating 3D from 2D: {str(e)}", is_warning=True)
            return None
   
    def destroy_node(self):
        """Clean shutdown of the node."""
        # Clear any large stored data
        self.depth_array = None
        self.camera_info = None
        
        # Stop resource monitor
        if hasattr(self, 'resource_monitor') and self.resource_monitor:
            try:
                self.resource_monitor.stop()
            except:
                pass
                
        # Stop performance metrics
        if hasattr(self, 'performance_metrics'):
            try:
                self.performance_metrics.stop_monitoring()
            except:
                pass
                
        super().destroy_node()

    """
    This section contains all helper methods for the OptimizedPositionEstimator class.
    These methods provide support functions for the core algorithms.
    """

    def log_error(self, error_message, is_warning=False, detection_error=False):
        """Simplified error logging with rate limiting."""
        current_time = TimeUtils.now_as_float()
        
        # Rate-limited logging through TTLDict
        if error_message not in self.error_last_logged:
            if is_warning:
                self.get_logger().warning(f"DEPTH: {error_message}")
            else:
                self.get_logger().error(f"DEPTH: {error_message}")
            
            # Record this error was logged (auto-expires after TTL)
            self.error_last_logged.set(error_message, current_time)
            
            # Force flush stdout to reduce log delay
            sys.stdout.flush()
            
        # For detection errors, also store in detection error buffer
        if detection_error:
            # Add error with timestamp to detection errors buffer
            self.detection_errors.append({
                'message': error_message,
                'timestamp': current_time,
                'type': 'warning' if is_warning else 'error'
            })
            
            # Publish error for external monitoring
            err_msg = String()
            err_msg.data = f"{{'error': '{error_message}', 'timestamp': {current_time}, 'type': '{'warning' if is_warning else 'error'}'}}"
            self.error_publisher.publish(err_msg)

    def camera_info_callback(self, msg):
        """Process camera calibration information."""
        self.camera_info = msg
        
        # Cache intrinsics for faster access
        self.fx = float(msg.k[0])  # Focal length x
        self.fy = float(msg.k[4])  # Focal length y
        self.cx = float(msg.k[2])  # Principal point x (optical center)
        self.cy = float(msg.k[5])  # Principal point y (optical center)
        
        # Update image dimensions
        self.depth_width = msg.width
        self.depth_height = msg.height
        
        # Log camera info once (first time received)
        if not self.camera_info_logged:
            self.get_logger().info(f"Camera info received: {self.depth_width}x{self.depth_height}")
            self.get_logger().info(f"Intrinsics: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}")
            self.get_logger().info(f"Scaling factors: x_scale={self.x_scale}, y_scale={self.y_scale}")
            self.camera_info_logged = True

    def depth_callback(self, msg):
        """Process depth image with efficient frame skipping."""
        try:
            # Adaptive frame skipping based on movement
            if DEPTH_CONFIG.get("adaptive_frame_skip", True):
                skip = self._determine_frame_skip()
                
                self.frame_counter += 1
                if self.frame_counter % skip != 0:
                    return
            else:
                # Standard frame skipping
                self.frame_counter += 1
                if self.frame_counter % self.process_every_n_frames != 0:
                    return
            
            # Use direct imgmsg_to_cv2 for better performance
            # Avoid unnecessary copying by using 'passthrough' encoding
            self.depth_array = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.depth_header = msg.header
            
        except Exception as e:
            self.log_error(f"Depth processing error: {str(e)}")

    def yolo_callback(self, msg):
        """Handle YOLO detections by processing them."""
        # Track all incoming YOLO detections
        self.yolo_detections_received += 1
        self.last_yolo_received_time = time.time()
        
        # Log additional debugging for tracking detection flow
        if self.yolo_detections_received % 10 == 0:  # Every 10 detections
            self.get_logger().info(
                f"YOLO detection stats: Received: {self.yolo_detections_received}, "
                f"Processed: {self.yolo_detections_processed}, "
                f"Failed: {self.yolo_detections_failed}, "
                f"Success rate: {(self.yolo_detections_processed/max(1, self.yolo_detections_received))*100:.1f}%"
            )
        
        # Process the detection
        self._process_detection(msg, 'YOLO')

    def yolo_bbox_callback(self, msg):
        """
        Process bounding box information from YOLO detection.
        
        Args:
            msg (Float32MultiArray): Bounding box data formatted as [center_x, center_y, width, height, confidence]
        """
        try:
            # Handle Float32MultiArray format for YOLO
            if hasattr(msg, 'data') and len(msg.data) >= 4:
                # Format: [center_x, center_y, width, height, confidence]
                width = msg.data[2]   # width is the 3rd value (index 2)
                height = msg.data[3]  # height is the 4th value (index 3)
                
                # Store bounding box data with timestamp
                self.yolo_bbox_data = {
                    'width': width,
                    'height': height,
                    'timestamp': time.time()
                }
                
                if self.debug_mode:
                    self.get_logger().info(f"Received YOLO bbox: {width:.1f}x{height:.1f}")
                    
        except Exception as e:
            self.log_error(f"Error processing YOLO bbox: {str(e)}", is_warning=True)

    def _determine_frame_skip(self):
        """
        Determine optimal frame skip rate based on CPU usage and movement.
        Returns the number of frames to skip.
        """
        # Default: process all frames unless CPU is high
        if self.current_cpu_usage > PERFORMANCE_TIERS[self.current_tier]["max_cpu_target"]:
            return min(4, self.process_every_n_frames + 1)  # Increase skip rate when CPU is high
        elif self.current_cpu_usage < PERFORMANCE_TIERS[self.current_tier]["min_cpu_target"]:
            return max(1, self.process_every_n_frames - 1)  # Decrease skip rate when CPU is low
        
        # Use current performance tier setting
        return self.process_every_n_frames

    def _filter_position(self, position):
        """
        Apply position filtering using the shared GroundPositionFilter class.
        This ensures consistent ground movement tracking between both nodes.
        
        Args:
            position: (x, y, z) position tuple/list
        
        Returns:
            Filtered position as (x, y, z) tuple
        """
        current_time = time.time()
        filtered_position = self.position_filter.update(position, current_time)
        return filtered_position

    def _fast_transform(self, point_stamped):
        """
        Optimized transform with matrix-based calculations when possible.
        Falls back to ROS transforms when needed.
        
        Args:
            point_stamped: PointStamped to transform
            
        Returns:
            Transformed PointStamped or None on failure
        """
        # Unique key for this transform
        frame_key = f"{self.reference_frame}_{point_stamped.header.frame_id}"
        
        # Use matrix transform if enabled and key exists in cache
        if self.use_matrix_transforms and frame_key in self.matrix_cache:
            # Get cached matrix
            matrix = self.matrix_cache.get(frame_key)
            
            # Extract point data
            x = point_stamped.point.x
            y = point_stamped.point.y
            z = point_stamped.point.z
            
            # Transform using matrix
            tx, ty, tz = matrix.transform_point(x, y, z)
            
            # Create result using object pool
            result = self.get_point_from_pool()
            result.header.frame_id = self.reference_frame
            result.header.stamp = point_stamped.header.stamp
            result.point.x = tx
            result.point.y = ty
            result.point.z = tz
            
            return result
        
        # Standard ROS transform with caching
        try:
            if frame_key in self.transform_cache:
                # Use cached transform
                transform = self.transform_cache.get(frame_key)
                
                try:
                    transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
                    return transformed
                except Exception:
                    # Remove from cache if transform fails
                    if frame_key in self.transform_cache:
                        del self.transform_cache[frame_key]
            
            # Get new transform
            transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                point_stamped.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=self.transform_timeout)
            )
            
            # Cache the transform
            self.transform_cache.set(frame_key, transform)
            
            # Also cache as matrix for future use
            if self.use_matrix_transforms:
                matrix = Matrix4x4.from_tf_transform(transform)
                self.matrix_cache.set(frame_key, matrix)
            
            # Apply transform
            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed
            
        except Exception as e:
            if self.debug_mode:
                self.get_logger().error(f"Transform lookup error: {str(e)}")
            return None

    def _transform_detection_to_depth_frame(self, msg):
        """
        Transform a detection from detection frame to depth camera frame.
        
        Args:
            msg: PointStamped detection message
            
        Returns:
            bool: True if transform is available, False otherwise
        """
        # Quick check for cached matrix transform
        if self.use_matrix_transforms and "detect_to_depth" in self.matrix_cache:
            return True
            
        try:
            # Get transform from detection frame to depth frame
            transform = self.tf_buffer.lookup_transform(
                self.depth_camera_frame,
                self.detection_camera_frame,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=self.transform_timeout)
            )
            
            # Cache as matrix for future use
            if self.use_matrix_transforms:
                matrix = Matrix4x4.from_tf_transform(transform)
                self.matrix_cache.set("detect_to_depth", matrix)
            
            # For debugging transforms
            if self.debug_mode:
                self.get_logger().info(
                    f"Transform from {self.detection_camera_frame} to {self.depth_camera_frame}: "
                    f"Translation: ({transform.transform.translation.x:.3f}, "
                    f"{transform.transform.translation.y:.3f}, {transform.transform.translation.z:.3f})"
                )
                sys.stdout.flush()
            
            return True
        except Exception as e:
            if self.debug_mode:
                self.get_logger().warning(f"Transform detection error: {str(e)}")
            return False

    def _transform_to_depth_frame(self, point_stamped):
        """
        Transform a point from reference frame to depth camera frame.
        
        Args:
            point_stamped: PointStamped to transform
            
        Returns:
            Transformed PointStamped or None on failure
        """
        # Check for cached matrix transform
        if self.use_matrix_transforms:
            frame_key = f"{self.depth_camera_frame}_{point_stamped.header.frame_id}"
            
            if frame_key in self.matrix_cache:
                # Get cached matrix
                matrix = self.matrix_cache.get(frame_key)
                
                # Extract point data
                x = point_stamped.point.x
                y = point_stamped.point.y
                z = point_stamped.point.z
                
                # Transform using matrix
                tx, ty, tz = matrix.transform_point(x, y, z)
                
                # Create result
                result = self.get_point_from_pool()
                result.header.frame_id = self.depth_camera_frame
                result.header.stamp = point_stamped.header.stamp
                result.point.x = tx
                result.point.y = ty
                result.point.z = tz
                
                return result
        
        # Standard ROS transform
        try:
            transform = self.tf_buffer.lookup_transform(
                self.depth_camera_frame,
                point_stamped.header.frame_id,
                rclpy.time.Time(),
                rclpy.duration.Duration(seconds=self.transform_timeout)
            )
            
            # Cache as matrix for future use
            if self.use_matrix_transforms:
                frame_key = f"{self.depth_camera_frame}_{point_stamped.header.frame_id}"
                matrix = Matrix4x4.from_tf_transform(transform)
                self.matrix_cache.set(frame_key, matrix)
            
            transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
            return transformed
        except Exception as e:
            if self.debug_mode:
                self.get_logger().error(f"Transform to depth frame error: {str(e)}")
            return None

    def _get_region_key(self, x, y):
        """
        Get a key for a spatial region of the depth image.
        
        Args:
            x, y: Pixel coordinates
            
        Returns:
            str: Region key in format "region_x_region_y"
        """
        region_x = int(x // self.region_grid_size)
        region_y = int(y // self.region_grid_size)
        return f"{region_x}_{region_y}"

    def _calculate_roi_size(self, base_size, success_rate):
        """
        Calculate ROI size based on success rate - higher success = smaller ROI.
        
        Args:
            base_size: Base ROI size
            success_rate: Success rate of depth detection in this region
            
        Returns:
            int: Calculated ROI size
        """
        if success_rate > 0.8:
            # Very reliable region - use smaller ROI for efficiency
            return max(self.min_roi_size, int(base_size * 0.7))
        elif success_rate > 0.5:
            # Moderately reliable - use default size
            return base_size
        elif success_rate > 0.2:
            # Somewhat unreliable - use larger ROI
            return min(self.max_roi_size, int(base_size * 1.5))
        else:
            # Very unreliable - use much larger ROI
            return min(self.max_roi_size, int(base_size * 2.0))

    def _get_adaptive_roi_size(self, pixel_x, pixel_y):
        """
        Calculate appropriate ROI size based on depth reliability in this region.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            
        Returns:
            int: Adaptive ROI size
        """
        region_key = self._get_region_key(pixel_x, pixel_y)
        
        # Default ROI size based on current settings
        base_roi_size = self.roi_size
        
        # Scale ROI size based on performance tier
        base_roi_size = int(base_roi_size * PERFORMANCE_TIERS[self.current_tier]["roi_size_scale"])
        
        # FPS optimization: use even smaller ROIs under high load
        if self.current_cpu_usage > 90.0 and base_roi_size > 8:
            base_roi_size = 8  # Ultra-small ROI for high CPU
        
        # Check if we have statistics for this region
        if region_key in self.depth_region_stats:
            stats = self.depth_region_stats.get(region_key)
            # Only adapt if we have enough data
            if stats.get('total_attempts', 0) >= 3:
                success_rate = stats.get('success_rate', 0.5)
                roi_size = self._calculate_roi_size(base_roi_size, success_rate)
                
                # Get distance tier if available
                if 'avg_depth' in stats:
                    avg_depth = stats['avg_depth']
                    # Find appropriate distance tier
                    for tier_name, tier_config in DISTANCE_TIERS.items():
                        if tier_config['range'][0] <= avg_depth <= tier_config['range'][1]:
                            # Adjust ROI size based on tier
                            roi_size = max(roi_size, tier_config['roi_size'])
                            break
                
                if self.debug_depth:
                    self.get_logger().info(f"DEPTH DEBUG: Using ROI size {roi_size} for region {region_key} (success rate: {success_rate:.2f})")
                return roi_size
        
        # Default size with distance-based minimum
        # Check if we have depth history for this region
        if region_key in self.depth_history:
            entry = self.depth_history.get(region_key)
            if entry and 'depth' in entry:
                depth = entry['depth']
                # Find appropriate distance tier
                for tier_name, tier_config in DISTANCE_TIERS.items():
                    if tier_config['range'][0] <= depth <= tier_config['range'][1]:
                        # Ensure ROI size meets minimum for this distance
                        base_roi_size = max(base_roi_size, tier_config['roi_size'])
                        break
        
        return base_roi_size

    def _update_depth_stats(self, pixel_x, pixel_y, success, nonzero_count, depth=None):
        """
        Update statistics for depth measurements in this region.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            success: Whether depth detection was successful
            nonzero_count: Number of nonzero depth values
            depth: Detected depth value if available
        """
        region_key = self._get_region_key(pixel_x, pixel_y)
        current_time = time.time()
        
        # Initialize stats for this region if needed
        if region_key not in self.depth_region_stats:
            self.depth_region_stats.set(region_key, {
                'total_attempts': 0,
                'successful': 0,
                'success_rate': 0.0,
                'last_updated': current_time,
                'depths': [],
                'avg_depth': None
            })
        
        # Get current stats
        stats = self.depth_region_stats.get(region_key, {})
        
        # Update stats
        stats['total_attempts'] = stats.get('total_attempts', 0) + 1
        if success and nonzero_count > 0:
            stats['successful'] = stats.get('successful', 0) + 1
        
        # Calculate success rate
        if stats['total_attempts'] > 0:
            stats['success_rate'] = stats['successful'] / stats['total_attempts']
        
        # Update timestamp
        stats['last_updated'] = current_time
        
        # Update depth tracking if provided
        if depth is not None and depth > 0:
            # Initialize depths list if needed
            if 'depths' not in stats:
                stats['depths'] = []
            
            # Add depth to list (keeping last 5)
            depths = stats['depths']
            depths.append(depth)
            if len(depths) > 5:
                depths.pop(0)
            
            # Calculate average depth
            if depths:
                stats['avg_depth'] = sum(depths) / len(depths)
        
        # Store updated stats
        self.depth_region_stats.set(region_key, stats)

    def _store_depth_history(self, pixel_x, pixel_y, depth, valid_points):
        """
        Store depth history for a region for future reference.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            depth: Depth value
            valid_points: Number of valid points used for depth calculation
        """
        if not DEPTH_CONFIG.get("use_depth_history", True) or depth <= 0:
            return
            
        region_key = self._get_region_key(pixel_x, pixel_y)
        current_time = time.time()
        
        # Store minimum needed information
        self.depth_history.set(region_key, {
            'depth': depth,
            'timestamp': current_time,
            'valid_points': valid_points,
            'pixel_x': pixel_x,
            'pixel_y': pixel_y
        })
        
        # Simplified stability tracking
        if region_key not in self.depth_stability_map:
            self.depth_stability_map.set(region_key, {
                'success_count': 1,
                'stability_score': 0.5
            })
        else:
            stability_data = self.depth_stability_map.get(region_key, {})
            stability_data['success_count'] = stability_data.get('success_count', 0) + 1
            # Simple incremental update of stability score
            stability_data['stability_score'] = min(0.9, 
                stability_data.get('stability_score', 0.5) + 0.05)
            self.depth_stability_map.set(region_key, stability_data)
        
        # Update sequence tracking
        if region_key not in self.depth_sequence_by_region:
            self.depth_sequence_by_region.set(region_key, [])
        
        sequence = self.depth_sequence_by_region.get(region_key, [])
        
        # Store depth and timestamp tuple
        sequence.append((depth, current_time, valid_points))
        
        # Keep only recent values
        while len(sequence) > 5:
            sequence.pop(0)
        
        # Update sequence
        self.depth_sequence_by_region.set(region_key, sequence)
        
        # Reset global counter since we had success
        self.consecutive_no_depth_frames = 0 if hasattr(self, 'consecutive_no_depth_frames') else 0
        self.last_frame_had_depth = True

    def _get_historical_depth_anywhere(self, pixel_x, pixel_y):
        """
        Look for ANY valid historical depth data.
        First checks target region, then neighboring regions, then anywhere.
        
        Args:
            pixel_x, pixel_y: Pixel coordinates
            
        Returns:
            tuple: (depth, valid_points) or (None, 0) if not found
        """
        current_time = time.time()
        
        # First check direct region
        region_key = self._get_region_key(pixel_x, pixel_y)
        if region_key in self.depth_history:
            entry = self.depth_history.get(region_key)
            if entry:
                age = current_time - entry.get('timestamp', 0)
                if age < self.depth_history_max_age:
                    if self.debug_depth:
                        self.get_logger().info(
                            f"DEPTH DEBUG: Using direct region history: {entry['depth']:.3f}m from {age:.1f}s ago"
                        )
                        sys.stdout.flush()
                    return entry.get('depth'), entry.get('valid_points', 1)
        
        # Check neighboring regions
        if self.use_neighbor_data:
            region_x = int(pixel_x // self.region_grid_size)
            region_y = int(pixel_y // self.region_grid_size)
            
            # Search neighboring cells in spiral pattern
            for r in range(1, 3):  # Search up to 2 cells away
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        # Skip corners (further away)
                        if abs(dx) == r and abs(dy) == r:
                            continue
                        
                        # Skip center (already checked)
                        if dx == 0 and dy == 0:
                            continue
                        
                        neighbor_key = f"{region_x+dx}_{region_y+dy}"
                        if neighbor_key in self.depth_history:
                            entry = self.depth_history.get(neighbor_key)
                            if entry:
                                age = current_time - entry.get('timestamp', 0)
                                if age < self.depth_history_max_age:
                                    if self.debug_depth:
                                        self.get_logger().info(
                                            f"DEPTH DEBUG: Using neighbor region history: {entry['depth']:.3f}m from {age:.1f}s ago"
                                        )
                                        sys.stdout.flush()
                                    return entry.get('depth'), max(1, entry.get('valid_points', 1) // 2)  # Reduce quality score
        
        # Find ANY history entry, starting with newest
        if self.depth_history:
            candidates = []
            for key, entry in self.depth_history.items():
                if not entry:
                    continue
                age = current_time - entry.get('timestamp', 0)
                if age < self.depth_history_max_age:
                    candidates.append((entry, age))
            
            if candidates:
                # Sort by age (newest first)
                candidates.sort(key=lambda x: x[1])
                entry = candidates[0][0]
                if self.debug_depth:
                    self.get_logger().info(
                        f"DEPTH DEBUG: Using ANY available depth history: {entry['depth']:.3f}m (age: {candidates[0][1]:.1f}s)"
                    )
                    sys.stdout.flush()
                return entry.get('depth'), max(1, entry.get('valid_points', 1) // 2)  # Reduce quality score
        
        return None, 0

    def _check_position_cache(self, msg, source):
        """
        Improved cache checking with higher hit rates.
        
        Args:
            msg: Detection message
            source: Detection source ('YOLO')
            
        Returns:
            bool: True if cache hit, False otherwise
        """
        # Track cache attempt
        self.cache_attempts += 1
        
        # Skip cache check if no cached position
        if (source not in self.detection_cache or 
            self.detection_cache[source]['detection_2d'] is None):
            return False
            
        # Get current detection and cached 2D detection
        curr_x, curr_y = msg.point.x, msg.point.y
        cached_detection = self.detection_cache[source]['detection_2d']
        cached_x, cached_y = cached_detection.point.x, cached_detection.point.y
        
        # Calculate squared distance
        dx = curr_x - cached_x
        dy = curr_y - cached_y
        dist_sq = dx*dx + dy*dy
        
        # Cache timing check
        curr_time = time.time()
        cached_time = self.detection_cache[source]['timestamp']
        
        # Scale threshold based on performance tier
        # SIGNIFICANTLY REDUCED from original to avoid excessive caching
        movement_threshold = 0.5  # Base threshold (reduced from 2.0)
        cache_duration = PERFORMANCE_TIERS[self.current_tier]["cache_lifetime"]
        
        # Scale threshold based on CPU usage - more permissive when CPU is high
        if self.current_cpu_usage > 85.0:
            movement_threshold *= 1.2
            cache_duration *= 1.1
        
        # Use cache if position is similar and cache is fresh
        if dist_sq < movement_threshold and curr_time - cached_time < cache_duration:
            cached_3d = self.detection_cache[source]['position_3d']
            
            # Reuse message object for this source
            new_msg = self.reusable_yolo_point if source == 'YOLO' else self.reusable_point
            new_msg.header.frame_id = cached_3d.header.frame_id
            new_msg.header.stamp = self.get_clock().now().to_msg()
            new_msg.point = cached_3d.point
            
            # Publish directly
            if source == 'YOLO':
                self.yolo_3d_publisher.publish(new_msg)
            
            self.position_publisher.publish(new_msg)
            
            # Update counters
            self.cache_hits += 1
            self.successful_conversions += 1
            
            # Trigger position logging on cache hits too
            self.consecutive_successful_detections += 1
            if self.consecutive_successful_detections >= self.detection_log_frequency:
                current_time = time.time()
                if current_time - self.last_ball_position_log >= self.ball_position_log_min_interval:
                    self._log_ball_position_and_direction()
                    self.consecutive_successful_detections = 0
            
            return True
        
        # Update total attempts
        self.total_attempts += 1
        return False

    def _update_fps(self):
        """Update FPS calculation."""
        curr_time = time.time()
        # Only update every second
        if curr_time - self.last_fps_update > 1.0:
            elapsed = curr_time - self.start_time
            if elapsed > 0:
                self.current_fps = self.successful_conversions / elapsed
                self.fps_history.append(self.current_fps)
            self.last_fps_update = curr_time

    def _adjust_performance(self):
        """Adaptive performance adjustment based on CPU usage and FPS."""
        # Update FPS
        self._update_fps()
        
        # Get current CPU usage
        cpu = self.current_cpu_usage
        
        # Determine appropriate performance tier
        new_tier = None
        
        if cpu > 85.0:
            # High CPU - switch to ultra performance
            new_tier = "ultra_performance"
        elif cpu > 70.0:
            # Moderate CPU - switch to performance
            new_tier = "performance"
        elif cpu < 40.0 and self.current_fps < 5.0:
            # Low CPU and low FPS - switch to balanced
            new_tier = "balanced"
        else:
            # Maintain current tier
            new_tier = self.current_tier
        
        # Apply new tier if changed
        if new_tier != self.current_tier:
            old_tier = self.current_tier
            self.current_tier = new_tier
            
            # Update settings from tier
            self.process_every_n_frames = PERFORMANCE_TIERS[new_tier]["process_every_n_frames"]
            self.roi_size = int(20 * PERFORMANCE_TIERS[new_tier]["roi_size_scale"])
            
            # Log tier change
            self.get_logger().info(
                f"Performance tier changed: {old_tier} -> {new_tier} "
                f"(CPU: {cpu:.1f}%, FPS: {self.current_fps:.1f})"
            )
        
        # Fine-tune frame skipping within tier
        old_skip = self.process_every_n_frames
        
        if cpu > PERFORMANCE_TIERS[self.current_tier]["max_cpu_target"] + 10.0:
            # CPU much higher than target - increase skip
            target_skip = min(8, self.process_every_n_frames + 1)
        elif cpu > PERFORMANCE_TIERS[self.current_tier]["max_cpu_target"]:
            # CPU higher than target - slight adjustment
            target_skip = min(4, self.process_every_n_frames)
        elif cpu < PERFORMANCE_TIERS[self.current_tier]["min_cpu_target"] - 10.0:
            # CPU much lower than target - decrease skip
            target_skip = max(1, self.process_every_n_frames - 1)
        else:
            # CPU within target range - maintain
            target_skip = self.process_every_n_frames
        
        # Apply new skip rate if changed
        if target_skip != old_skip:
            self.process_every_n_frames = target_skip
            
            # Only log significant changes
            if abs(old_skip - target_skip) > 0:
                self.get_logger().info(
                    f"Frame skip adjusted: 1 in {old_skip} -> 1 in {target_skip} "
                    f"(CPU: {cpu:.1f}%, FPS: {self.current_fps:.1f})"
                )

    def _handle_resource_alert(self, resource_type, value):
        """Handle resource alerts from the resource monitor."""
        if resource_type == 'cpu':
            # Ensure we get non-zero values
            try:
                cpu_value = float(value)
                self.current_cpu_usage = max(0.1, cpu_value)
                
                # Log significant CPU changes
                current_time = time.time()
                if cpu_value > 90.0 and current_time - self.last_resource_alert_time > 30.0:
                    self.get_logger().info(f"High CPU usage: {cpu_value:.1f}%")
                    self.last_resource_alert_time = current_time
            except (ValueError, TypeError):
                # Default value if conversion fails
                self.current_cpu_usage = 50.0

    def publish_system_diagnostics(self):
        """Publish comprehensive system diagnostics with detection quality metrics."""
        # Only run at specified interval
        current_time = time.time()
        if current_time - self.last_diag_log_time < 2.0:
            return
        self.last_diag_log_time = current_time
        
        # Update FPS with accurate measurement
        self._update_fps()
        
        # Get accurate CPU usage
        actual_cpu = psutil.cpu_percent(interval=0.05)
        self.current_cpu_usage = actual_cpu
        
        # Calculate metrics (only if we have frames processed)
        if self.fps_history:
            # Calculate avg fps from buffer
            sum_fps = 0
            count = 0
            for fps in self.fps_history:
                sum_fps += fps
                count += 1
            
            avg_fps = sum_fps / count if count > 0 else 0
            
            # Calculate frame rate percentage and cache hit rate
            frame_rate_pct = 100.0 / self.process_every_n_frames
            # Fixed to ensure cache hit rate is 0-100%
            cache_hit_rate = 0.0
            if self.cache_attempts > 0:
                cache_hit_rate = min(100.0, (self.cache_hits / self.cache_attempts) * 100.0)
            
            # Calculate reliability metrics
            detection_age_yolo = 0
            if 'YOLO' in self.detection_cache and self.detection_cache['YOLO']['timestamp'] > 0:
                detection_age_yolo = current_time - self.detection_cache['YOLO']['timestamp']
            
            # Reliability score (0-100)
            reliability = 100.0
            if detection_age_yolo > 2.0:
                reliability = 75.0
            if detection_age_yolo > 5.0:
                reliability = 50.0
            
            # Log comprehensive status with immediate flush
            self.get_logger().info(
                f"Depth camera: {self.current_fps:.1f} FPS (avg: {avg_fps:.1f}), "
                f"CPU: {actual_cpu:.1f}%, "
                f"RAM: {psutil.virtual_memory().percent:.1f}%, "
                f"Reliability: {reliability:.1f}%, "
                f"Frames: 1:{self.process_every_n_frames}, "
                f"Cache: {cache_hit_rate:.1f}%, "
                f"Tier: {self.current_tier}"
            )
            sys.stdout.flush()
            
            # Publish detailed diagnostics
            diag_data = {
                "fps": self.current_fps,
                "avg_fps": avg_fps,
                "cpu": actual_cpu,
                "ram": psutil.virtual_memory().percent,
                "frame_skip": self.process_every_n_frames,
                "frame_rate_pct": frame_rate_pct,
                "cache_hit_rate": cache_hit_rate,
                "reliability": reliability,
                "detection_age_yolo": round(detection_age_yolo, 2),
                "tier": self.current_tier,
                "path_counts": self.path_counts,
                "yolo_received": self.yolo_detections_received,
                "yolo_processed": self.yolo_detections_processed,
                "timestamp": current_time,
            }
            self.reusable_diag.data = str(diag_data)
            self.system_diagnostics_publisher.publish(self.reusable_diag)

    def _log_ball_position_and_direction(self):
        """
        Periodically log the ball's distance and direction for monitoring.
        This provides more descriptive information about the ball's position.
        """
        current_time = time.time()
        
        # Check if we have any recent successful detections
        yolo_timestamp = 0
        if 'YOLO' in self.detection_cache:
            yolo_timestamp = self.detection_cache['YOLO']['timestamp']
        
        # Find the most recent detection - INCREASED recency window from 2.0 to 7.0
        position_3d = None
        source = None
        
        if yolo_timestamp > 0 and current_time - yolo_timestamp < 7.0:  # Increased from 2.0
            position_3d = self.detection_cache['YOLO']['position_3d']
            source = "YOLO"
        
        # Exit if no recent detection
        if position_3d is None:
            self.get_logger().info("Ball tracking: No recent ball detection to report")
            return
        
        # Calculate distance from camera to ball
        ball_x = position_3d.point.x
        ball_y = position_3d.point.y
        ball_z = position_3d.point.z
        
        # Get distance from origin (camera position)
        distance = math.sqrt(ball_x**2 + ball_y**2 + ball_z**2)
        
        # Calculate position in reference frame if needed
        ref_position = None
        if position_3d.header.frame_id != self.reference_frame:
            try:
                # Create a temporary point for transform
                temp_point = PointStamped()
                temp_point.header = position_3d.header
                temp_point.point = position_3d.point
                
                # Transform to reference frame
                ref_point = self._fast_transform(temp_point)
                if ref_point is not None:
                    ref_position = (ref_point.point.x, ref_point.point.y, ref_point.point.z)
                    # Return to pool
                    self.return_point_to_pool(ref_point)
            except Exception:
                pass
        else:
            ref_position = (ball_x, ball_y, ball_z)
        
        # Log the position, distance and direction
        if ref_position is not None:
            # Calculate ground distance (ignore height)
            ground_distance = math.sqrt(ref_position[0]**2 + ref_position[1]**2)
            
            # Calculate direction angles using reference frame coordinates
            # Azimuth (horizontal angle) - positive is right, negative is left
            azimuth = math.degrees(math.atan2(ref_position[1], ref_position[0]))
            
            # Elevation (vertical angle) - positive is up, negative is down
            elevation = math.degrees(math.atan2(ref_position[2], math.sqrt(ref_position[0]**2 + ref_position[1]**2)))
            
            # Direction in plain language based on reference frame coordinates
            horizontal_direction = "left" if ref_position[1] > 0 else "right"
            forward_backward = "in front" if ref_position[0] > 0 else "behind"
            
            # Descriptive direction from robot's perspective
            direction_desc = f"{horizontal_direction} and {forward_backward}"
            
            # Log detailed information
            self.get_logger().info(
                f"🏀 Ball tracking: Distance = {distance:.2f}m | Ground distance = {ground_distance:.2f}m | "
                f"Position = ({ref_position[0]:.2f}, {ref_position[1]:.2f}, {ref_position[2]:.2f})m | "
                f"Direction: {direction_desc} | "
                f"Azimuth = {azimuth:.1f}° | Elevation = {elevation:.1f}° | "
                f"Source: {source}"
            )
            
            # Update most recent detection data for other components
            self.most_recent_detection = {
                'position': ref_position,
                'timestamp': current_time,
                'source': source,
                'distance': distance,
                'ground_distance': ground_distance,
                'azimuth': azimuth,
                'elevation': elevation
            }
        else:
            # Fallback to simple log if reference position not available
            self.get_logger().info(
                f"🏀 Ball tracking: Distance = {distance:.2f}m | "
                f"Position (in {position_3d.header.frame_id}) = ({ball_x:.2f}, {ball_y:.2f}, {ball_z:.2f})m | "
                f"Source: {source}"
            )
        
        # Force flush logs
        sys.stdout.flush()

def main(args=None):
    """Main function to initialize and run the 3D position estimator node."""
    rclpy.init(args=args)
        
    # Set Raspberry Pi environment variable
    os.environ['RASPBERRY_PI'] = '1'
        
    # Enable depth debugging if requested via command line
    if args and '--debug-depth' in args:
        DEPTH_CONFIG['debug_depth'] = True
        print("Depth debugging enabled via command line")
        
    # Create and initialize the node
    node = OptimizedPositionEstimator()
        
    # Use a SingleThreadedExecutor instead of MultiThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
        
    print("=================================================")
    print("Ultra-Optimized Basketball Tracking")
    print("=================================================")
        
    try:
        sys.stdout.flush()  # Force flush logs
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Stopped by user.")
    except Exception as e:
        node.get_logger().error(f"Error: {str(e)}")
    finally:
        # Clean up
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()