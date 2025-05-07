#!/usr/bin/env python3

"""
Tennis Ball Tracking Robot - HSV Ball Detector Node
==================================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving tennis ball.
The system uses multiple sensing modalities for robust detection:
- YOLO neural network detection (more accurate but computationally intensive)
- HSV color-based detection (this node - fast and efficient)
- LIDAR for depth sensing
- Depth camera for additional depth information

This Node's Purpose:
------------------
This HSV detector node uses traditional computer vision techniques to detect tennis balls
based on their distinctive yellow-green color. It processes camera images, applies color
filtering in the HSV color space, and identifies circular objects of the right size and shape.

HSV color detection offers several advantages:
- Very fast processing compared to neural network approaches
- More resilient to changes in lighting conditions than RGB
- Can be fine-tuned for specific color targets

Data Pipeline:
-------------
1. Camera images are received from '/ascamera/camera_publisher/rgb0/image'
2. Images are processed to extract tennis ball position using HSV filtering
3. Detected positions are published to '/tennis_ball/hsv/position'
4. These positions are then used by:
   - Depth camera node for 3D position estimation
   - Sensor fusion node for combining with other detection methods
   - State manager for decision making
   - PID controller for motor control

Educational Visualization - How HSV Ball Detection Works:
--------------------------------------------------------

┌─────────────────────────────────────────────────────────────────────────────┐
│                        THE HSV COLOR SPACE EXPLAINED                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
         ┌─────────────────┐         │         ┌─────────────────┐
         │                 │         │         │                 │
         │      HUE        │◄────────┼────────►│    SATURATION   │
         │                 │         │         │                 │
         └─────────────────┘         │         └─────────────────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │                 │
                            │      VALUE      │
                            │                 │
                            └─────────────────┘

   HUE: The actual color             SATURATION: Color purity       VALUE: Brightness
   (0-180 in OpenCV)                 (0-255 in OpenCV)              (0-255 in OpenCV)
   
   Tennis ball yellow is              High values mean pure          Controls how bright
   typically between                  colors, low values             or dark the color
   hue values 25-45                   are more washed out            appears

┌─────────────────────────────────────────────────────────────────────────────┐
│                    HSV BALL DETECTION PROCESSING PIPELINE                    │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
  │  1. ORIGINAL  │         │  2. CONVERT   │         │ 3. HSV COLOR  │
  │     IMAGE     │ ───────►│  BGR TO HSV   │ ───────►│    FILTER     │
  │               │         │               │         │               │
  └───────────────┘         └───────────────┘         └───────────────┘
                                                              │
                                                              ▼
  ┌───────────────┐         ┌───────────────┐         ┌───────────────┐
  │  6. PUBLISH   │         │  5. VALIDATE  │         │ 4. MORPHOLOGY │
  │   POSITION    │ ◄─────── │  BALL CONTOUR │ ◄─────── │   OPERATIONS  │
  │               │         │               │         │               │
  └───────────────┘         └───────────────┘         └───────────────┘

1. Receive RGB camera image
2. Convert to HSV color space where yellow is easily isolated
3. Apply mask to keep only pixels in the tennis ball's color range
4. Clean up mask with erosion/dilation to remove noise
5. Find contours and filter by size and circularity
6. Publish 2D position when valid ball detected

┌─────────────────────────────────────────────────────────────────────────────┐
│                      BALL VALIDATION PROCESS EXPLAINED                       │
└─────────────────────────────────────────────────────────────────────────────┘

              ┌─────────────────┐
              │   YELLOW BLOB   │
              │  FROM HSV MASK  │
              └────────┬────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │      SIZE CHECK          │
        │  100 < area < 1500 px²   │──┐
        └──────────────────────────┘  │ If too small or
                       │              │ too large, reject
                       │ Pass         │
                       ▼              │
        ┌──────────────────────────┐  │
        │   CIRCULARITY CHECK      │  │
        │   0.5 < circ < 1.3       │──┘
        └──────────────────────────┘
                       │
                       │ Pass
                       ▼
        ┌──────────────────────────┐
        │    CONFIDENCE SCORE      │
        │  Based on how closely    │
        │  the contour matches     │
        │  ideal ball properties   │
        └──────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │    PUBLISH POSITION      │
        │   with confidence in Z   │
        └──────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE PROCESSING SYSTEM                            │
└─────────────────────────────────────────────────────────────────────────────┘

 CPU USAGE      RESOLUTION FACTOR    FRAME PROCESSING    IMAGE SIZE REDUCTION
 
 ▁▁▁▁▁▁▁▁▁█  →      FULL (1.0)     →    EVERY FRAME    →    ┌─────┐
 ≤ 60%                                                       │     │ 320×320
                                                             └─────┘
                                                             
 ▁▁▁▁▁▁▁███  →     MEDIUM (0.75)   →   EVERY OTHER     →    ┌───┐
 > 70%                                    FRAME              │   │ 240×240
                                                             └───┘
                                                            
 ███████████  →     SMALL (0.5)    →  1 IN 3 FRAMES    →    ┌──┐
 > 85%                                                       │  │ 160×160
                                                             └──┘

This adaptive system enables the node to continue functioning under high CPU load,
trading off some detection quality to maintain system responsiveness.
"""

import sys
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import psutil
import json
from collections import deque
from functools import lru_cache

# Add necessary paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import utility modules
from utilities.resource_monitor import ResourceMonitor
from utilities.time_utils import TimeUtils
from ball_chase.config.config_loader import ConfigLoader

# Configuration management
class ConfigManager:
    """
    Handles loading and processing of configuration settings.
    
    The ConfigManager centralizes all configuration loading and processing to make
    the system more maintainable. It provides structured access to all configuration
    parameters needed by other components.
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                          CONFIGURATION HIERARCHY                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    
                            ┌───────────────┐
                            │  YAML CONFIG  │
                            │    FILE       │
                            └───────┬───────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │ ConfigManager │
                            └───────┬───────┘
                                    │
                   ┌────────────────┼────────────────┐
                   │                │                │
                   ▼                ▼                ▼
        ┌────────────────┐ ┌──────────────┐ ┌─────────────────┐
        │ Ball Detection │ │    Topics    │ │ Performance     │
        │ Parameters     │ │ Configuration │ │ Configuration   │
        └────────────────┘ └──────────────┘ └─────────────────┘
    
    This hierarchical organization makes it easy to understand and modify 
    different aspects of the system's behavior.
    """
    
    def __init__(self):
        # Load configuration from file
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.load_yaml('hsv_config.yaml')
        self._init_configuration()
        
    def _init_configuration(self):
        """Initialize all configuration parameters from YAML file"""
        # Topic configuration
        self.topics = self.config.get('topics', {
            "input": {
                "camera": "/ascamera/camera_publisher/rgb0/image"
            },
            "output": {
                "position": "/tennis_ball/hsv/position"
            }
        })
        
        # HSV color range configuration
        hsv_lower = np.array(self.config.get('ball', {}).get('hsv_range', {}).get('lower', [27, 58, 77]), dtype=np.uint8)
        hsv_upper = np.array(self.config.get('ball', {}).get('hsv_range', {}).get('upper', [45, 255, 255]), dtype=np.uint8)
        
        # Ball detection configuration
        self.ball_config = {
            "hsv_range": {
                "lower": hsv_lower,  # Lower HSV boundary for tennis ball
                "upper": hsv_upper   # Upper HSV boundary for tennis ball
            },
            "size": self.config.get('ball', {}).get('size', {
                "min_area": 100,     # Minimum area in pixels for 320x320 image
                "max_area": 1500,    # Maximum area in pixels for 320x320 image
                "ideal_area": 600    # Ideal area for confidence calculation
            }),
            "shape": self.config.get('ball', {}).get('shape', {
                "min_circularity": 0.5,   # Minimum circularity (0.7 is a perfect circle)
                "max_circularity": 1.3,   # Maximum circularity
                "ideal_circularity": 0.7  # Ideal circularity for confidence calculation
            })
        }
        
        # Display configuration
        self.display_config = self.config.get('display', {
            "enable_visualization": False,  # Whether to show detection visualization
            "window_width": 800,            # Width of visualization window
            "window_height": 600            # Height of visualization window
        })
        
        # Diagnostic configuration
        self.diag_config = self.config.get('diagnostics', {
            "target_width": 320,           # Target width for processing 
            "target_height": 320,          # Target height for processing
            "debug_level": 1,              # 0=errors only, 1=info, 2=debug
            "log_interval": 10             # Log every N frames for performance stats
        })
        
        # Performance configuration
        self.perf_config = self.config.get('performance', {
            # CPU thresholds for reducing processing
            "cpu_high_threshold": 85.0,       # Above this threshold, reduce processing dramatically
            "cpu_medium_threshold": 70.0,     # Above this threshold, start reducing processing
            "cpu_low_threshold": 60.0,        # Below this threshold, process at full quality
            
            # Resolution downscaling factors for different CPU loads
            "high_load_scale": 0.5,           # Scale down to 50% resolution in high load
            "medium_load_scale": 0.75,        # Scale down to 75% resolution in medium load
            
            # Processing frequency control
            "min_processing_interval": 0.05,  # At least 50ms between frames in high load
            "cpu_check_interval": 1.0         # Check CPU usage every 1 second
        })


class ImageProcessor:
    """
    Handles image processing and ball detection logic.
    
    The ImageProcessor implements the computer vision pipeline for detecting
    tennis balls using HSV color filtering. It handles all stages from image
    preprocessing to ball detection and validation.
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                     HSV COLOR FILTERING EXPLAINED                            │
    └─────────────────────────────────────────────────────────────────────────────┘
    
        Original Image            HSV Color Space             HSV Mask
          (RGB/BGR)              Transformation
    
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
    │     ████      │          │    H: 30-45   │          │               │
    │   ████████    │  ======> │    S: 70-255  │  ======> │     ████      │
    │   ████████    │          │    V: 80-255  │          │    ██████     │
    │     ████      │          │               │          │               │
    └───────────────┘          └───────────────┘          └───────────────┘
         Tennis Ball              Color Range              Filtered Result
    
    After obtaining the mask, we apply morphological operations (erosion/dilation)
    to clean up noise, then find contours in the mask to identify potential tennis balls.
    
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                      BALL CONTOUR ANALYSIS                                   │
    └─────────────────────────────────────────────────────────────────────────────┘
    
        Contour Properties          Circle Approximation         Confidence Score
    
         Area: π × r²                 Minimum Enclosing            Closeness to
                                         Circle                   Ideal Properties
    
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
    │   ░░░░░░░     │          │       O       │          │  Size: 0.82   │
    │  ░░░░░░░░░    │  ======> │      ╱ ╲      │  ======> │  Shape: 0.91  │
    │  ░░░░░░░░░    │          │     │   │     │          │               │
    │   ░░░░░░░     │          │      ╲ ╱      │          │  TOTAL: 0.88  │
    └───────────────┘          └───────────────┘          └───────────────┘
    
    Circularity = Contour_Area / Circle_Area
    A perfect circle has circularity close to 1.0
    We accept values between 0.5-1.3 to account for partial occlusion and noise
    """
    
    def __init__(self, config, logger):
        """Initialize the image processor with configuration and logger"""
        self.config = config
        self.logger = logger
        
        # Extract parameters from config
        self.target_width = self.config.diag_config["target_width"]
        self.target_height = self.config.diag_config["target_height"]
        self.debug_level = self.config.diag_config["debug_level"]
        
        # Ball detection parameters
        self.lower_yellow = self.config.ball_config['hsv_range']['lower']
        self.upper_yellow = self.config.ball_config['hsv_range']['upper']
        self.min_ball_area = self.config.ball_config['size']['min_area']
        self.max_ball_area = self.config.ball_config['size']['max_area']
        self.ideal_area = self.config.ball_config['size']['ideal_area']
        self.min_circularity = self.config.ball_config['shape']['min_circularity']
        self.max_circularity = self.config.ball_config['shape']['max_circularity']
        self.ideal_circularity = self.config.ball_config['shape']['ideal_circularity']
        
        # Initialize image buffers
        self._init_image_buffers()
    
    def _init_image_buffers(self):
        """Pre-allocate memory for image operations to avoid frequent allocations"""
        # Pre-allocate buffers for different scales
        self.image_buffers = {}
        for scale in [1.0, self.config.perf_config.get('medium_load_scale', 0.75), 
                     self.config.perf_config.get('high_load_scale', 0.5)]:
            width = int(self.target_width * scale)
            height = int(self.target_height * scale)
            # Only create if sensible dimensions (at least 32x32)
            if width >= 32 and height >= 32:
                self.image_buffers[scale] = {
                    'bgr': np.zeros((height, width, 3), dtype=np.uint8),
                    'hsv': np.zeros((height, width, 3), dtype=np.uint8),
                    'mask': np.zeros((height, width), dtype=np.uint8)
                }
        
        # Create morphological kernels at startup to avoid runtime creation
        self.morph_kernels = {
            'small': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            'medium': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            'large': cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        }
    
    @lru_cache(maxsize=8)  # Cache results for better performance
    def calculate_scaled_thresholds(self, scale_factor):
        """Calculate area thresholds scaled by current resolution factor"""
        # Area scales with the square of linear dimensions
        area_scale = scale_factor * scale_factor
        return {
            'min_area': self.min_ball_area * area_scale,
            'max_area': self.max_ball_area * area_scale,
            'ideal_area': self.ideal_area * area_scale
        }
    
    def preprocess_image(self, frame, scale_factor):
        """
        Preprocess the image for ball detection.
        
        This method transforms the raw camera image into a binary mask
        highlighting regions that match the tennis ball's color profile.
        
        ┌─────────────────────────────────────────────────────────────────────────────┐
        │                    IMAGE PREPROCESSING PIPELINE                              │
        └─────────────────────────────────────────────────────────────────────────────┘
        
          Original               Resized                HSV                Binary
          Image                  Image               Conversion              Mask
          
        ┌─────────┐           ┌─────────┐         ┌─────────┐          ┌─────────┐
        │         │           │         │         │         │          │         │
        │  1280   │  resize   │   320   │  BGR    │   320   │  inRange │   320   │
        │   x    ─┼─────────► │    x    │ ─────► │    x    │ ─────────►│    x    │
        │  720    │  factor   │   320   │  to HSV │   320   │  H,S,V    │   320   │
        │         │           │         │         │         │  limits   │         │
        └─────────┘           └─────────┘         └─────────┘          └─────────┘
                                                                            │
                                                                            │
                                                                            ▼
                                                                       ┌─────────┐
                                   Applied to original image  ◄────────┤ Cleaned │
                                   to extract ball position            │  Mask   │
                                                                       └─────────┘
                                                                       erosion/
                                                                       dilation
        
        Args:
            frame: The input image frame in BGR format
            scale_factor: The scaling factor to apply
            
        Returns:
            tuple: The processed frame and HSV mask
        """
        # STEP 1: Resize based on scale factor
        width = int(self.target_width * scale_factor)
        height = int(self.target_height * scale_factor)
        
        # Use pre-allocated buffer if available
        if scale_factor in self.image_buffers:
            # Resize directly into pre-allocated buffer
            cv2.resize(frame, (width, height), dst=self.image_buffers[scale_factor]['bgr'])
            frame = self.image_buffers[scale_factor]['bgr']
        else:
            # Fallback - create a new buffer
            frame = cv2.resize(frame, (width, height))
        
        # STEP 2: Convert from BGR to HSV color space efficiently
        # Use pre-allocated buffer if available
        if scale_factor in self.image_buffers:
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV, dst=self.image_buffers[scale_factor]['hsv'])
            hsv = self.image_buffers[scale_factor]['hsv']
        else:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # STEP 3: Create a mask that only shows yellow pixels
        # Use pre-allocated buffer if available
        if scale_factor in self.image_buffers:
            cv2.inRange(hsv, self.lower_yellow, self.upper_yellow, 
                        dst=self.image_buffers[scale_factor]['mask'])
            mask = self.image_buffers[scale_factor]['mask']
        else:
            mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
        
        # STEP 4: Clean up the mask with morphological operations
        # Select kernel size based on current scale
        if scale_factor < 0.6:  # Very small images
            kernel = self.morph_kernels['small']
        else:
            kernel = self.morph_kernels['medium']
            
        # Apply morphology in-place to avoid allocations
        cv2.erode(mask, kernel, dst=mask, iterations=1)
        cv2.dilate(mask, kernel, dst=mask, iterations=2)
        
        return frame, mask
    
    def detect_ball(self, frame, mask, scale_factor, enhanced_detection=False, cpu_usage=0.0):
        """
        Detect a tennis ball in the frame using the HSV mask.
        
        IMAGINE THIS: 🔍
        ---------------
        Think of this method like a detective examining a crime scene. The HSV mask
        has highlighted all yellow objects (like using a special UV light that makes
        certain evidence glow). Now the detective needs to figure out which of those
        glowing spots is actually the evidence they're looking for (the tennis ball).
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. FINDING CONTOURS ("What yellow shapes do we see?")
           - The mask shows all yellow pixels as white, everything else as black
           - Contours trace the outlines of all these white blobs
           - This is like drawing an outline around each glowing spot
        
        2. FILTERING BY SIZE ("Is it the right size?")
           - A tennis ball can't be too small or too large in the image
           - We filter out tiny specs and huge blobs
           - This is like saying "the evidence should be about the size of a golf ball"
        
        3. CHECKING CIRCULARITY ("Is it round enough?")
           - A tennis ball is round, so we check how circular each shape is
           - Circularity = (Area of shape) / (Area of its smallest enclosing circle)
           - Perfect circles have circularity close to 1.0
           - We accept values from 0.5-1.3 to handle partially obscured balls
           - This is like saying "our evidence should be roughly circular"
        
        4. CALCULATING CONFIDENCE ("How sure are we?")
           - We calculate a confidence score based on how close the shape matches
             our ideal tennis ball properties
           - 70% of the score comes from shape (circularity)
           - 30% comes from size (area)
           - This is like saying "I'm 85% confident this is what we're looking for"
        
        5. ENHANCED DETECTION (When CPU allows)
           - For better results, we can use more advanced techniques
           - This includes Hough Circles to find perfect circles
           - This is like the detective bringing in special equipment
             for a more thorough analysis when there's time
        
        DAILY LIFE EXAMPLE: 🏐
        -------------------
        Imagine you're looking for a specific yellow ball in a playroom with many toys.
        You first notice all yellow things (HSV filter), then check each one:
        - Is it the right size? (Not a tiny yellow bead or huge yellow blanket)
        - Is it round? (Not a yellow block or banana)
        - Does it look like the ball we're looking for? (Confidence score)
        
        That's exactly what this method does, but with computer vision!
        
        Args:
            frame: The preprocessed image frame
            mask: The HSV color mask
            scale_factor: The current scaling factor
            enhanced_detection: Whether to use enhanced detection techniques
            cpu_usage: Current CPU usage percentage
            
        Returns:
            dict or None: Detection information or None if no ball found
        """
        # STEP 1: Find contours efficiently
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Skip processing if no significant contours
        if not contours or len(contours) == 0:
            return None, contours
        
        # STEP 2: Calculate scaled area thresholds
        area_scale = scale_factor * scale_factor
        min_area = self.min_ball_area * area_scale
        max_area = self.max_ball_area * area_scale
        ideal_area = self.ideal_area * area_scale
        
        # STEP 3: Pre-filter small contours
        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area * 0.5:  # Use 50% of min as pre-filter
                filtered_contours.append((cnt, area))
        
        # If no contours pass pre-filtering, return early
        if not filtered_contours:
            return None, contours
        
        # STEP 4: Find the best ball candidate
        best_contour = None
        best_radius = 0.0
        best_confidence = 0.0
        best_center = (0, 0)
        best_area = 0.0
        best_circularity = 0.0
        
        for cnt, area in filtered_contours:
            # Skip contours outside area range
            if area < min_area or area > max_area:
                continue
                
            # Find the smallest enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            
            # Calculate circularity
            circle_area = np.pi * (radius ** 2) if radius > 0 else 1
            circularity = area / circle_area
            
            # Skip if circularity is outside acceptable range
            if circularity < self.min_circularity or circularity > self.max_circularity:
                continue
            
            # Calculate confidence score
            circularity_score = 1.0 - min(abs(circularity - self.ideal_circularity) / 
                                         self.ideal_circularity, 1.0)
            size_score = 1.0 - min(abs(area - ideal_area) / ideal_area, 1.0)
            
            # Combined confidence score (weighted average)
            confidence = (circularity_score * 0.7) + (size_score * 0.3)
            
            # Keep the highest confidence match
            if confidence > best_confidence:
                best_contour = cnt
                best_radius = radius
                best_confidence = confidence
                best_center = (cx, cy)
                best_area = area
                best_circularity = circularity
        
        # STEP 5: Enhanced detection (optional)
        if enhanced_detection and cpu_usage < 70.0 and len(filtered_contours) > 0:
            self._apply_enhanced_detection(filtered_contours, mask, min_area, max_area,
                                          best_confidence, best_center, best_radius, best_area)
        
        # STEP 6: Return detection if a ball was found
        if best_contour is not None:
            # Scale coordinates back to original resolution for consistent reporting
            scale_back = 1.0 / scale_factor
            
            center_x, center_y = best_center
            center_x *= scale_back
            center_y *= scale_back
            
            # Return detection information
            return {
                'center': (center_x, center_y),
                'radius': best_radius * scale_back,
                'area': best_area * scale_back * scale_back,
                'circularity': best_circularity,
                'confidence': best_confidence,
                'contour': best_contour,
                'scale_factor': scale_factor
            }, contours
        
        return None, contours
    
    def _apply_enhanced_detection(self, filtered_contours, mask, min_area, max_area, 
                                 best_confidence, best_center, best_radius, best_area):
        """
        Apply advanced detection techniques when CPU allows
        
        This is called by detect_ball when enhanced detection is enabled
        """
        try:
            # Get the largest contour by area for enhanced processing
            largest_cnt, largest_area = max(filtered_contours, key=lambda x: x[1])
            
            # Only attempt circle detection if we have a significant contour
            if largest_area > min_area and largest_area < max_area:
                # Apply Hough Circle detection with adaptive parameters
                min_radius = int(np.sqrt(min_area/np.pi))
                max_radius = int(np.sqrt(max_area/np.pi))
                
                detected_circles = cv2.HoughCircles(
                    mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                    param1=50, param2=10, 
                    minRadius=min_radius,
                    maxRadius=max_radius
                )
                
                # If circles are found, incorporate into detection
                if detected_circles is not None:
                    # Convert circles to integer coordinates
                    detected_circles = np.round(detected_circles[0, :]).astype(int)
                    
                    # Find the best circle
                    for (x, y, r) in detected_circles:
                        # Calculate how well circle matches contour
                        circle_center = (float(x), float(y))
                        circle_radius = float(r)
                        circle_area = np.pi * r * r
                        
                        # Only consider if better than current best
                        if circle_area > min_area and circle_area < max_area:
                            # Check if this circle improves detection
                            if best_confidence < 0.7:  # Only replace if current confidence is low
                                circle_confidence = 0.8  # Default confidence from HoughCircles
                                
                                # Use the circle instead of contour if it's more reliable
                                best_center = circle_center
                                best_radius = circle_radius
                                best_area = circle_area
                                best_confidence = max(best_confidence, circle_confidence)
                                
                                if self.debug_level >= 2:
                                    self.logger.debug("Enhanced detection improved result")
                                break
        except Exception as e:
            # Ignore errors in enhanced detection - fall back to standard
            if self.debug_level >= 2:
                self.logger.debug(f"Enhanced detection error: {e}")


class PerformanceMonitor:
    """
    Manages performance monitoring and adaptive processing
    to optimize resource usage.
    
    The PerformanceMonitor continuously checks system resource usage (primarily CPU)
    and adapts processing quality and rate to maintain system stability. This adaptive
    approach allows the node to continue functioning even during high system load.
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    ADAPTIVE PROCESSING DECISION TREE                         │
    └─────────────────────────────────────────────────────────────────────────────┘
    
                       ┌─────────────────┐
                       │    Check CPU    │
                       │     Usage       │
                       └────────┬────────┘
                                │
                   ┌────────────┴───────────────┐
                   │                            │
                   ▼                            ▼
         ┌─────────────────┐         ┌──────────────────┐
         │   CPU ≤ 60%?    │ Yes     │   60% < CPU ≤ 70%│ Yes
         └────────┬────────┘─────────►   (Normal Load)  ├────────┐
                  │ No                └──────────────────┘        │
                  │                                               │
                  ▼                                               │
         ┌─────────────────┐                                      │
         │  70% < CPU ≤ 85%│ Yes                                  │
         │  (Medium Load)  ├─────┐                                │
         └────────┬────────┘     │                                │
                  │ No           │                                │
                  │              │                                │
                  ▼              ▼                                ▼
     ┌───────────────────┐  ┌────────────────┐           ┌─────────────────┐
     │    CPU > 85%      │  │ • Scale: 0.75x │           │  • Scale: 1.0x  │
     │   (High Load)     │  │ • Skip: 1/2    │           │  • Skip: None   │
     └─────────┬─────────┘  │ • Memory: Med  │           │  • Memory: Max  │
               │            └────────────────┘           └─────────────────┘
               ▼
     ┌─────────────────────┐
     │  • Scale: 0.5x      │
     │  • Skip: 2/3        │
     │  • Memory: Minimal  │
     └─────────────────────┘
    
    The system also includes an emergency mode for extreme CPU usage (>95%)
    that further reduces processing to ensure system stability.
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                        RESOURCE MONITORING CYCLE                             │
    └─────────────────────────────────────────────────────────────────────────────┘
    
     ┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
     │ Check System│      │  Analyze &  │      │   Adjust    │      │  Apply New  │
     │  Resources  │─────►│   Decide    │─────►│ Processing  │─────►│  Settings   │
     │             │      │             │      │  Parameters │      │             │
     └─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
           │                                                               │
           └───────────────────────────────────────────────────────────────┘
                                   Repeat cycle
    
    This continuous monitoring and adaptation cycle runs independently of the
    image processing pipeline to ensure responsive system behavior.
    """
    
    def __init__(self, config, logger):
        """Initialize with configuration and logger"""
        self.config = config
        self.logger = logger
        
        # Extract parameters from config
        self.cpu_high_threshold = self.config.perf_config.get('cpu_high_threshold', 85.0)
        self.cpu_medium_threshold = self.config.perf_config.get('cpu_medium_threshold', 70.0)
        self.cpu_low_threshold = self.config.perf_config.get('cpu_low_threshold', 60.0)
        self.high_load_scale = self.config.perf_config.get('high_load_scale', 0.5) 
        self.medium_load_scale = self.config.perf_config.get('medium_load_scale', 0.75)
        self.min_processing_interval = self.config.perf_config.get('min_processing_interval', 0.05)

        # Initialize metrics
        self.current_cpu_usage = 0.0
        self.cpu_history = deque(maxlen=30)  # Track last 30 seconds
        self.adaptation_history = deque(maxlen=20)  # Track adaptation changes
        
        # Configure optimized processing
        self._configure_optimizations()
        
    def _configure_optimizations(self):
        """Configure performance optimizations based on available system resources"""
        try:
            total_ram = psutil.virtual_memory().total / (1024 * 1024)  # MB
            
            # On Pi 5 with 16GB RAM, we can use more advanced options
            if total_ram >= 12000:  # At least 12GB
                # Enable more advanced detection features that use more RAM
                self.use_enhanced_detection = True
                self.default_scale_factor = 1.0
                
                self.logger.info(f"Using enhanced detection features (high RAM mode)")
            else:
                # Standard settings for lower memory systems
                self.use_enhanced_detection = False
                self.default_scale_factor = 0.75
                
                self.logger.info(f"Using standard detection features (limited RAM mode)")
        except Exception as e:
            # Default settings if we can't check memory
            self.use_enhanced_detection = False
            self.default_scale_factor = 0.75
            self.logger.warn(f"Could not determine system memory. Using default settings: {e}")
        
        # Initialize current scale to default
        self.current_scale_factor = self.default_scale_factor
        
        # Number of frames to skip in low power mode (0 means no skipping)
        self.low_power_skip_frames = 0
        self.frame_skip_counter = 0
        self.skip_count = 0
    
    def check_cpu_and_adjust_processing(self):
        """
        Check CPU usage and adjust processing quality/rate accordingly.
        
        IMAGINE THIS: 🏎️
        ---------------
        Think of this method like an automatic transmission in a car that shifts gears
        based on how hard the engine is working. When going uphill (high CPU), the car
        shifts to a lower gear (reduced quality) to prevent the engine from overheating.
        When on a flat road (low CPU), it shifts back to a higher gear (full quality)
        for better performance.
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. MEASURING SYSTEM LOAD ("How hard is the computer working?")
           - We check the CPU usage percentage every second
           - This is like checking the tachometer in a car
        
        2. DECISION MAKING ("Should we change our processing approach?")
           - We compare the current CPU usage against three thresholds:
             * Low (< 60%): Computer has plenty of capacity
             * Medium (60-85%): Computer is working hard
             * High (> 85%): Computer is struggling
        
        3. ADJUSTING PROCESSING ("Shifting gears")
           - For high CPU usage: Reduce image size to 50% and process only 1 in 3 frames
           - For medium CPU usage: Reduce image size to 75% and process every other frame
           - For low CPU usage: Use full-size images and process every frame
        
        4. RECORDING CHANGES ("Keeping a log")
           - We track all adaptive changes for diagnostics
           - This helps understand how the system behaves over time
        
        REAL-WORLD ANALOGY: 🎮
        -------------------
        This is similar to how video games adjust graphics quality based on framerate:
        - When the game is running smoothly, it maintains high-quality graphics
        - When frames start dropping, it reduces texture detail, shadows, etc.
        - When performance improves again, it gradually restores quality
        
        The goal is to maintain responsive, real-time ball tracking even when the
        system is under heavy load from other processes.
        
        Returns:
            tuple: (bool indicating if settings changed, current CPU usage)
        """
        settings_changed = False
        
        try:
            # Get current CPU usage
            cpu_usage = psutil.cpu_percent(interval=0.1)
            self.current_cpu_usage = cpu_usage
            
            # Add to history
            self.cpu_history.append((TimeUtils.now_as_float(), cpu_usage))
            
            # Current scale and skip settings before adjustments
            old_scale = self.current_scale_factor
            old_skip = self.low_power_skip_frames
            
            # Adjust processing based on CPU load
            if cpu_usage > self.cpu_high_threshold:
                # Very high CPU - dramatic reduction
                self.current_scale_factor = self.high_load_scale
                self.low_power_skip_frames = 2  # Process only 1 in 3 frames
                
                if old_scale != self.current_scale_factor or old_skip != self.low_power_skip_frames:
                    self.logger.warn(
                        f"CPU usage very high ({cpu_usage:.1f}%): reducing resolution to "
                        f"{int(100*self.current_scale_factor)}% and processing 1 in {self.low_power_skip_frames+1} frames"
                    )
                    settings_changed = True
                    
                    # Record adaptation
                    self.adaptation_history.append({
                        'time': TimeUtils.now_as_float(),
                        'cpu': cpu_usage,
                        'action': 'high_reduction',
                        'scale': self.current_scale_factor,
                        'skip': self.low_power_skip_frames
                    })
                    
            elif cpu_usage > self.cpu_medium_threshold:
                # Moderately high CPU - medium reduction
                self.current_scale_factor = self.medium_load_scale
                self.low_power_skip_frames = 1  # Process every other frame
                
                if old_scale != self.current_scale_factor or old_skip != self.low_power_skip_frames:
                    self.logger.info(
                        f"CPU usage high ({cpu_usage:.1f}%): reducing resolution to "
                        f"{int(100*self.current_scale_factor)}% and processing every other frame"
                    )
                    settings_changed = True
                    
                    # Record adaptation
                    self.adaptation_history.append({
                        'time': TimeUtils.now_as_float(),
                        'cpu': cpu_usage,
                        'action': 'medium_reduction',
                        'scale': self.current_scale_factor,
                        'skip': self.low_power_skip_frames
                    })
                    
            elif cpu_usage < self.cpu_low_threshold:
                # Low CPU - restore full processing if we were reducing
                if self.current_scale_factor < 1.0 or self.low_power_skip_frames > 0:
                    self.current_scale_factor = self.default_scale_factor
                    self.low_power_skip_frames = 0  # Process all frames
                    
                    self.logger.info(
                        f"CPU usage normal ({cpu_usage:.1f}%): restoring normal processing "
                        f"at {int(100*self.current_scale_factor)}% resolution"
                    )
                    settings_changed = True
                    
                    # Record adaptation
                    self.adaptation_history.append({
                        'time': TimeUtils.now_as_float(),
                        'cpu': cpu_usage,
                        'action': 'restore_normal',
                        'scale': self.current_scale_factor,
                        'skip': self.low_power_skip_frames
                    })
            
            return settings_changed, cpu_usage
            
        except Exception as e:
            self.logger.error(f"Error in CPU monitoring: {e}")
            return False, self.current_cpu_usage
    
    def should_skip_frame(self):
        """
        Check if the current frame should be skipped based on CPU load
        
        Returns:
            bool: True if frame should be skipped, False if it should be processed
        """
        if self.low_power_skip_frames > 0:
            self.frame_skip_counter += 1
            if (self.frame_skip_counter % (self.low_power_skip_frames + 1)) != 0:
                # Skip this frame
                self.skip_count += 1
                return True
        
        return False
    
    def handle_resource_alert(self, resource_type, value):
        """
        Handle resource alerts by adjusting processing behavior
        
        Args:
            resource_type: Type of resource (cpu, memory, etc)
            value: Current value of the resource
        
        Returns:
            bool: True if emergency measures were taken
        """
        self.logger.warn(f"Resource alert: {resource_type.upper()} at {value:.1f}%")
        
        # If CPU usage is critically high, implement more aggressive measures
        if resource_type == 'cpu' and value > 95.0:  # Extremely high CPU
            old_skip = self.low_power_skip_frames
            old_scale = self.current_scale_factor
            
            # Emergency measures - very low resolution and high frame skipping
            self.low_power_skip_frames = 3  # Skip 3 frames, process 1
            self.current_scale_factor = 0.4  # 40% of original resolution
            
            self.logger.warn(
                f"CRITICAL CPU USAGE: Emergency reduction to {int(100*self.current_scale_factor)}% "
                f"resolution and 1 in {self.low_power_skip_frames+1} frames"
            )
            
            # Record adaptation
            self.adaptation_history.append({
                'time': TimeUtils.now_as_float(),
                'resource_type': resource_type,
                'value': value,
                'action': 'emergency_reduction',
                'old_scale': old_scale,
                'new_scale': self.current_scale_factor,
                'old_skip': old_skip,
                'new_skip': self.low_power_skip_frames
            })
            
            return True
        
        return False


class HSVTennisBallTracker(Node):
    """
    A ROS2 node that uses HSV color filtering to detect a yellow tennis ball
    in camera images and publishes its position.
    
    HSV (Hue, Saturation, Value) color space is better for color detection than RGB
    because it separates color (hue) from intensity (value) and color purity (saturation).
    This makes it more robust to lighting changes.
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                          ROS2 NODE ARCHITECTURE                              │
    └─────────────────────────────────────────────────────────────────────────────┘
    
                           ┌────────────────────┐
                           │  HSVTennisBallNode │
                           └──────────┬─────────┘
                                      │
          ┌─────────────────┬─────────┼─────────┬─────────────────┐
          │                 │                   │                 │
          ▼                 ▼                   ▼                 ▼
    ┌──────────────┐ ┌─────────────┐   ┌───────────────┐  ┌─────────────┐
    │ ConfigManager│ │    Image    │   │  Performance  │  │  Resource   │
    │              │ │  Processor  │   │   Monitor     │  │  Monitor    │
    └──────────────┘ └─────────────┘   └───────────────┘  └─────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                             ROS2 TOPICS FLOW                                 │
    └─────────────────────────────────────────────────────────────────────────────┘
    
    INPUTS:                                               OUTPUTS:
    ┌────────────────┐                                   ┌────────────────┐
    │  Camera Image  │                                   │  Ball Position │
    │     Topic      │                                   │     Topic      │
    └───────┬────────┘                                   └────────┬───────┘
            │                                                     │
            ▼                                                     ▼
    ┌────────────────┐                                   ┌────────────────┐
    │   Subscribe    │                                   │    Publish     │
    └───────┬────────┘                                   └────────┬───────┘
            │                                                     │
            ▼                                                     ▼
    ┌────────────────┐      ┌───────────────┐           ┌────────────────┐
    │ Image Callback │─────►│  Process &    │──────────►│  PointStamped  │
    │    Function    │      │  Detect Ball  │           │    Message     │
    └────────────────┘      └───────────────┘           └────────────────┘
    
                            ┌───────────────┐
                            │  Diagnostics  │─────────► System monitoring
                            │   Publisher   │          & health tracking
                            └───────────────┘
    """
    
    def __init__(self):
        """
        Initialize the HSV tennis ball tracker node.
        
        IMAGINE THIS: 🏗️
        ---------------
        This initialization process is like setting up a factory assembly line.
        Each component needs to be created, connected, and calibrated before
        the production line can start running smoothly.
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. COMPONENT CREATION ("Building the machinery")
           - Create the ConfigManager to load and manage settings
           - Create the PerformanceMonitor to manage resource usage
           - Create the ImageProcessor to handle ball detection
           - Set up the ResourceMonitor to track system health
        
        2. COMMUNICATION SETUP ("Connecting the conveyor belts")
           - Subscribe to camera image topic to receive input
           - Create publishers for ball position and diagnostics
           - Initialize message conversion tools (CvBridge)
           - Set up timer for regular diagnostics
        
        3. STATE INITIALIZATION ("Setting the initial conditions")
           - Initialize counters and statistics trackers
           - Create data structures for metrics collection
           - Set up performance tracking deques
           - Configure visualization if enabled
        
        4. SYSTEM LOGGING ("Recording the startup")
           - Log important configuration parameters
           - Report detection thresholds and HSV values
           - Confirm successful initialization
        
        REAL-WORLD ANALOGY: 🚀
        -------------------
        This is similar to the pre-flight checklist for launching a rocket:
        - All systems must be properly initialized
        - Communication channels must be established
        - Initial conditions must be set correctly
        - Everything must be logged and verified
        
        Only when all these steps are complete can the node begin its core
        function of processing images and detecting tennis balls.
        """
        # Initialize ROS node
        super().__init__('hsv_tennis_ball_tracker')
        
        # Initialize configuration
        self.config_manager = ConfigManager()
        
        # Initialize performance monitor
        self.perf_monitor = PerformanceMonitor(self.config_manager, self.get_logger())
        
        # Initialize image processor
        self.image_processor = ImageProcessor(self.config_manager, self.get_logger())
        
        # Add resource monitoring
        self._setup_resource_monitoring()
        
        # Set up subscriptions and publishers
        self._setup_communication()
        
        # Initialize state variables
        self._init_state_variables()
        
        # Set up visualization if enabled
        self._setup_visualization()
        
        # Log startup information
        self._log_startup_info()
    
    def _setup_resource_monitoring(self):
        """Set up resource monitoring for adaptive processing"""
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(
            node=self,
            publish_interval=15.0,  # Less frequent to reduce overhead
            enable_temperature=True
        )
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        self.resource_monitor.start()
        
        # Add timer to check CPU and adjust processing rate/quality
        self.cpu_check_timer = self.create_timer(
            self.config_manager.perf_config.get('cpu_check_interval', 1.0),
            self._check_cpu_and_adjust_processing
        )
    
    def _setup_communication(self):
        """Set up all subscriptions and publishers"""
        # Extract topic names from config
        camera_topic = self.config_manager.topics["input"]["camera"]
        position_topic = self.config_manager.topics["output"]["position"]
        
        # Subscribe to the camera feed
        self.subscription = self.create_subscription(
            Image, 
            camera_topic, 
            self.image_callback, 
            10
        )
        
        # Create publishers
        self.ball_publisher = self.create_publisher(
            PointStamped, 
            position_topic, 
            10
        )
        
        self.system_diagnostics_publisher = self.create_publisher(
            String, 
            "/tennis_ball/hsv/diagnostics",  
            10
        )
        
        self.cpu_usage_publisher = self.create_publisher(
            Float32,
            '/system/resources/cpu_load',
            10
        )
        
        # CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Timer for publishing diagnostics
        self.diagnostics_timer = self.create_timer(2.0, self.publish_system_diagnostics)
    
    def _init_state_variables(self):
        """Initialize all state tracking variables"""
        # Get important config parameters
        self.target_width = self.config_manager.diag_config["target_width"]
        self.target_height = self.config_manager.diag_config["target_height"]
        self.enable_visualization = self.config_manager.display_config["enable_visualization"] 
        self.debug_level = self.config_manager.diag_config["debug_level"]
        self.log_interval = self.config_manager.diag_config["log_interval"]
        
        # Performance tracking
        self.start_time = TimeUtils.now_as_float()
        self.frame_count = 0
        self.no_detection_count = 0
        self.last_detection_time = None
        self.last_frame_time = 0.0
        
        # Detection statistics
        self.detection_count = 0
        self.detection_sizes = deque(maxlen=50)
        self.detection_confidences = deque(maxlen=50)
        
        # Performance metrics
        max_history = 100
        self.fps_history = deque(maxlen=max_history)
        self.processing_times = deque(maxlen=max_history)
        self.detection_history = deque(maxlen=max_history)
        
        # Error and warning tracking
        self.errors = deque(maxlen=50)
        self.warnings = deque(maxlen=50)
        
        # Diagnostic metrics
        self.diagnostic_metrics = {
            'fps_history': deque(maxlen=10),
            'processing_time_history': deque(maxlen=10),
            'detection_rate_history': deque(maxlen=10),
            'last_detection_position': None,
            'last_detection_time': 0.0,
            'total_frames': 0,
            'missed_frames': 0,
            'errors': deque(maxlen=10),
            'warnings': deque(maxlen=10),
            'adaptations': deque(maxlen=20)
        }
    
    def _setup_visualization(self):
        """Set up visualization windows if enabled"""
        if self.enable_visualization:
            cv2.namedWindow("Tennis Ball Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Tennis Ball Detector", 
                           self.config_manager.display_config["window_width"], 
                           self.config_manager.display_config["window_height"])
            self.get_logger().info("Visualization enabled - showing detection window")
    
    def _log_startup_info(self):
        """Log important information during startup"""
        self.get_logger().info("HSV Tennis Ball Tracker has started!")
        self.get_logger().info(f"Processing images at {self.target_width}x{self.target_height} to match YOLO")
        
        ball_config = self.config_manager.ball_config
        self.get_logger().info(
            f"Looking for balls with area between "
            f"{ball_config['size']['min_area']} and {ball_config['size']['max_area']} pixels"
        )
        self.get_logger().info(
            f"HSV color range: Lower={ball_config['hsv_range']['lower']}, "
            f"Upper={ball_config['hsv_range']['upper']}"
        )
        
        perf_config = self.config_manager.perf_config
        self.get_logger().info(
            f"CPU adaptivity enabled: high={perf_config.get('cpu_high_threshold')}%, "
            f"medium={perf_config.get('cpu_medium_threshold')}%"
        )
    
    def _check_cpu_and_adjust_processing(self):
        """Timer callback to check CPU and adjust processing parameters"""
        settings_changed, cpu_usage = self.perf_monitor.check_cpu_and_adjust_processing()
        
        # Publish CPU usage
        cpu_msg = Float32()
        cpu_msg.data = float(cpu_usage)
        self.cpu_usage_publisher.publish(cpu_msg)
        
        # Update diagnostic metrics with current CPU usage and adaptations
        if settings_changed and hasattr(self, 'diagnostic_metrics'):
            self.diagnostic_metrics['current_cpu'] = cpu_usage
            self.diagnostic_metrics['current_scale'] = self.perf_monitor.current_scale_factor
            self.diagnostic_metrics['frame_skip'] = self.perf_monitor.low_power_skip_frames
    
    def _handle_resource_alert(self, resource_type, value):
        """Handle resource alerts from the resource monitor"""
        emergency_taken = self.perf_monitor.handle_resource_alert(resource_type, value)
        
        # Update diagnostic metrics
        if emergency_taken and hasattr(self, 'diagnostic_metrics'):
            self.diagnostic_metrics['adaptations'].append({
                'timestamp': TimeUtils.now_as_float(),
                'resource_type': resource_type,
                'value': value,
                'action': 'emergency_reduction',
                'new_scale': self.perf_monitor.current_scale_factor,
                'new_skip': self.perf_monitor.low_power_skip_frames
            })
    
    def _generate_trace_id(self):
        """Generate a unique trace ID for debugging"""
        return f"hsv_{self.frame_count}_{int(time.time()*1000) % 10000}"
    
    def image_callback(self, msg):
        """
        Process each incoming camera image to detect tennis balls.
        
        IMAGINE THIS: 📷
        ---------------
        Think of this function like what happens in your brain when you're watching a
        tennis match. Every time your eyes receive a new "frame" of the match, your brain
        processes it to identify where the ball is, then updates your understanding of the game.
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. RECEIVING IMAGES ("Seeing frames of the tennis match")
           - A new image arrives from the camera via ROS message system
           - We get about 30 of these per second in ideal conditions
        
        2. ADAPTIVE PROCESSING ("Deciding which frames to analyze")
           - If the computer is very busy, we might skip some frames
           - This is like how your brain might ignore some visual information
             when you're overwhelmed with too many things to process
        
        3. RATE LIMITING ("Processing at a reasonable pace")
           - We enforce a minimum time between processing frames
           - This prevents flooding the system with too many computations
           - It's like how you might watch slow-motion replays for important moments
        
        4. IMAGE PROCESSING PIPELINE ("Spotting the tennis ball")
           - Convert the ROS message to an OpenCV image
           - Apply our image processing techniques to find the ball
           - This is similar to how your visual cortex processes what you see
        
        5. RESULT HANDLING ("Updating your understanding")
           - If a ball is found, publish its position
           - If no ball is found, log diagnostics
           - Either way, update performance metrics
        
        REAL-WORLD ANALOGY: 🎬
        -------------------
        This is similar to a video editor reviewing footage:
        - Each frame comes in as raw material
        - The editor decides which frames need processing based on available time
        - Important frames get analyzed in detail, others might be skipped
        - When something interesting is found, it's marked for others to see
        - The whole process repeats for the next frame
        
        This callback is the heart of the node - it's triggered every time a new
        image arrives from the camera, and coordinates the entire detection process.
        
        Args:
            msg (Image): The incoming camera image from ROS
        """
        # Check if we should skip this frame based on CPU load
        if self.perf_monitor.should_skip_frame():
            self.frame_count += 1  # Still count it for metrics
            return
        
        # Check minimum time between frames for rate limiting
        current_time = TimeUtils.now_as_float()
        time_since_last_frame = current_time - self.last_frame_time
        min_interval = self.config_manager.perf_config.get('min_processing_interval', 0.05)
        
        if time_since_last_frame < min_interval:
            # Too soon since last frame - enforce minimum interval
            self.perf_monitor.skip_count += 1
            return
            
        # Update last frame time
        self.last_frame_time = current_time
        
        # Start timing for performance metrics
        processing_start = TimeUtils.now_as_float()
        self.frame_count += 1
        self.diagnostic_metrics['total_frames'] += 1
        
        trace_id = self._generate_trace_id()
        if self.debug_level >= 2:  # Only log at debug level 2+
            self.get_logger().debug(f"Processing frame {self.frame_count} ({trace_id})")

        try:
            # Convert ROS image to OpenCV format
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Create a copy for visualization if enabled
            if self.enable_visualization:
                display_frame = frame.copy()
            
            # Process image using current scale factor from performance monitor
            scale_factor = self.perf_monitor.current_scale_factor
            frame, mask = self.image_processor.preprocess_image(frame, scale_factor)
            
            # Detect ball in the processed image
            enhanced_detection = self.perf_monitor.use_enhanced_detection
            cpu_usage = self.perf_monitor.current_cpu_usage
            detection, contours = self.image_processor.detect_ball(
                frame, mask, scale_factor, enhanced_detection, cpu_usage
            )
            
            # Handle detection (or lack thereof)
            if detection:
                self._handle_successful_detection(detection, msg.header)
            else:
                self._handle_failed_detection(contours)
            
            # Update visualization if enabled
            if self.enable_visualization:
                self._update_visualization(frame, detection, processing_start)
            
            # Record processing time
            processing_time = TimeUtils.now_as_float() - processing_start
            self.processing_times.append(processing_time)
            self.diagnostic_metrics['processing_time_history'].append(processing_time)
            
            # Log performance metrics occasionally
            if self.frame_count % self.log_interval == 0:
                self._log_performance_metrics(processing_start)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.errors.append((TimeUtils.now_as_float(), str(e)))
            if len(self.errors) > 3:
                # Multiple errors in a row, log traceback
                self.diagnostic_metrics['errors'].append({
                    'time': TimeUtils.now_as_float(),
                    'error': str(e),
                    'trace_id': trace_id
                })
    
    def _handle_successful_detection(self, detection, header):
        """
        Handle a successful ball detection.
        
        IMAGINE THIS: 🎯
        ---------------
        Think of this method like a sports announcer who just spotted an amazing play.
        The announcer needs to quickly tell everyone what they saw, update the scoreboard,
        and keep track of statistics - all while the game continues.
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. EXTRACTING DETAILS ("What exactly did we see?")
           - Get the ball's center position (x, y coordinates)
           - Get the confidence score (how sure are we it's a ball)
           - Get the ball's size and other properties
        
        2. LOGGING INFORMATION ("Commentating the play")
           - If appropriate, log information about what we found
           - When CPU is busy, we reduce logging to avoid slowing things down
        
        3. CREATING THE MESSAGE ("Updating the scoreboard")
           - Create a standardized PointStamped message containing:
             * The ball's 2D position (x, y coordinates)
             * The confidence score (stored in z coordinate)
             * The camera frame information
             * The timestamp for synchronization
        
        4. PUBLISHING POSITION ("Broadcasting to everyone")
           - Send the position message to the '/tennis_ball/hsv/position' topic
           - Other nodes (like depth camera, fusion, and control) use this information
        
        5. UPDATING STATISTICS ("Keeping track of the game")
           - Reset the "no detection" counter
           - Increment the detection counter
           - Store detection metrics for diagnostics
        
        REAL-WORLD EXAMPLE: 📊
        -------------------
        This is similar to a quality control system in a factory:
        - When a good product is detected on the conveyor belt, the system:
          * Records exactly where it was found
          * Logs the details for the supervisor
          * Signals other systems about the product
          * Updates statistics about production quality
        
        Args:
            detection: The detection information dictionary
            header: ROS message header from the original image
        """
        # Extract detection details
        center_x, center_y = detection['center']
        confidence = detection['confidence']
        area = detection['area']
        scale_factor = detection['scale_factor']
        
        # Log the detection (less frequently when CPU is high)
        log_this_detection = self.debug_level >= 1 and (
            self.perf_monitor.current_cpu_usage < self.config_manager.perf_config.get('cpu_medium_threshold', 70.0) or 
            self.frame_count % 10 == 0  # Less logging in high CPU
        )
        
        if log_this_detection:
            self.get_logger().info(
                f"FOUND BALL at ({center_x:.1f}, {center_y:.1f}) "
                f"radius: {detection['radius']:.1f}, area: {area:.1f}, "
                f"confidence: {confidence:.2f}, scale: {scale_factor:.2f}"
            )
        
        # Create and publish position message
        position_msg = PointStamped()
        
        # Use original image timestamp for synchronization
        if TimeUtils.is_timestamp_valid(header.stamp):
            position_msg.header.stamp = header.stamp
        else:
            position_msg.header.stamp = TimeUtils.now_as_ros_time()
            if self.debug_level >= 2:
                self.get_logger().debug("Using current time (invalid original timestamp)")
            
        position_msg.header.frame_id = "ascamera_color_0"  # Camera frame
        
        # Set position coordinates and confidence
        position_msg.point.x = float(center_x)
        position_msg.point.y = float(center_y)
        position_msg.point.z = float(confidence)  # Use z for confidence
        
        # Publish the ball position
        self.ball_publisher.publish(position_msg)
        
        # Reset no detection counter and update statistics
        self.no_detection_count = 0
        self.detection_count += 1
        self.last_detection_time = TimeUtils.now_as_float()
        
        # Store detection metrics for statistics
        self.detection_sizes.append(area)
        self.detection_confidences.append(confidence)
        
        # Store for diagnostics
        self.diagnostic_metrics['last_detection_position'] = (center_x, center_y)
        self.diagnostic_metrics['last_detection_time'] = TimeUtils.now_as_float()
    
    def _handle_failed_detection(self, contours):
        """
        Handle case where no ball was detected.
        
        IMAGINE THIS: 🔍
        ---------------
        This method is like a detective who didn't find what they were looking for,
        but still needs to document the search and provide reasons why the search
        might have failed.
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. TRACKING MISSES ("Recording the failed attempt")
           - Increment the counter for consecutive frames without detection
           - Update diagnostic metrics to reflect the missed detection
        
        2. ADAPTIVE LOGGING ("Deciding how much to report")
           - When CPU load is high, reduce logging frequency
           - When CPU load is normal, provide more detailed information
           - This balances diagnostic detail against performance impact
        
        3. DETAILED ANALYSIS ("Understanding why we didn't find it")
           - If yellow objects were detected but rejected, explain why
           - Report whether they were the wrong size, wrong shape, etc.
           - This helps diagnose configuration issues or false negatives
        
        REAL-WORLD EXAMPLE: 🏠
        -------------------
        This is similar to a home security system that didn't detect an intruder:
        - It records that no detection occurred
        - It tracks how long it's been since the last detection
        - It might explain "motion was detected but was too small to be a person"
        - After a certain period without detections, it might trigger a warning
        
        Args:
            contours: List of contours found in the image
        """
        # Increment no detection counter
        self.no_detection_count += 1
        
        # Track missed frames for diagnostics
        self.diagnostic_metrics['missed_frames'] += 1
        
        # Log "no ball found" at specified intervals (less frequently when CPU is high)
        cpu_load = self.perf_monitor.current_cpu_usage
        log_interval = self.log_interval * (1 + int(cpu_load > 70))
        
        if self.no_detection_count % log_interval == 0:
            self._log_no_detection_info(contours)
    
    def _log_no_detection_info(self, contours):
        """Log detailed information about why no ball was detected"""
        # Only log detailed info if CPU isn't too high
        if self.perf_monitor.current_cpu_usage > self.perf_monitor.cpu_high_threshold:
            # Just brief logging at high CPU
            if self.no_detection_count % 20 == 0:  # Very occasional logging
                self.get_logger().info(f"No ball detected for {self.no_detection_count} frames (high CPU mode)")
            return
            
        # Normal detailed logging
        self.get_logger().info(f"NO BALL FOUND (for {self.no_detection_count} consecutive frames)")
        
        # Get scaled area thresholds
        scale_factor = self.perf_monitor.current_scale_factor
        area_scale = scale_factor * scale_factor
        
        # Get ball detection parameters from image processor
        min_area = self.image_processor.min_ball_area * area_scale
        max_area = self.image_processor.max_ball_area * area_scale
        
        # If there were yellow objects, explain why they weren't detected as balls
        if contours and len(contours) > 0:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            
            if largest_area > 20 * area_scale:  # Only report significant blobs
                (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
                circle_area = np.pi * (radius ** 2) if radius > 0 else 1
                circularity = largest_area / circle_area
                
                # Explain why it was rejected
                reason = "unknown reason"
                if largest_area < min_area:
                    reason = f"too small (area={largest_area:.0f}, min={min_area:.0f})"
                elif largest_area > max_area:
                    reason = f"too large (area={largest_area:.0f}, max={max_area:.0f})"
                elif circularity < self.image_processor.min_circularity:
                    reason = f"not circular enough (circularity={circularity:.2f}, min={self.image_processor.min_circularity})"
                elif circularity > self.image_processor.max_circularity:
                    reason = f"too circular (circularity={circularity:.2f}, max={self.image_processor.max_circularity})"
                
                self.get_logger().info(f"Largest yellow object rejected because: {reason}")
    
    def _update_visualization(self, frame, detection, processing_start):
        """
        Update visualization window if enabled
        
        Args:
            frame: The processed image frame
            detection: Detection information or None
            processing_start: Start time for processing metrics
        """
        if not self.enable_visualization:
            return
            
        # Only update visualization every few frames in high CPU mode
        if self.perf_monitor.current_cpu_usage > 80 and self.frame_count % 3 != 0:
            return
        
        try:
            # Create copy for display
            display_frame = frame.copy()
            
            # Add detection visualization if ball was found
            if detection:
                # Scale coordinates to match current frame size
                scale_factor = detection['scale_factor']
                center = detection['center']
                radius = detection['radius']
                
                # Scale back to current display size
                current_x = int(center[0] * scale_factor)
                current_y = int(center[1] * scale_factor)
                current_radius = int(radius * scale_factor)
                
                # Draw the detection
                cv2.circle(display_frame, (current_x, current_y), current_radius, (0, 255, 0), 2)
                cv2.circle(display_frame, (current_x, current_y), 2, (0, 0, 255), -1)
                
                # Add confidence text
                confidence_text = f"Conf: {detection['confidence']:.2f}"
                cv2.putText(display_frame, confidence_text, (current_x - 30, current_y - current_radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add performance metrics
            elapsed = TimeUtils.now_as_float() - processing_start
            fps_text = f"FPS: {1.0/elapsed:.1f}" if elapsed > 0 else "FPS: --"
            cpu_text = f"CPU: {self.perf_monitor.current_cpu_usage:.1f}%"
            scale_text = f"Scale: {self.perf_monitor.current_scale_factor:.2f}"
            
            cv2.putText(display_frame, fps_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, cpu_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, scale_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the image
            cv2.imshow("Tennis Ball Detector", display_frame)
            cv2.waitKey(1)  # Update window
        
        except Exception as e:
            self.get_logger().warn(f"Visualization error: {e}")
    
    def _log_performance_metrics(self, processing_start):
        """Log performance metrics for monitoring"""
        elapsed = TimeUtils.now_as_float() - self.start_time
        processed_count = self.frame_count - self.perf_monitor.skip_count
        
        if elapsed > 0 and processed_count > 0:
            # Calculate key metrics
            fps = processed_count / elapsed
            detection_rate = self.detection_count / processed_count if processed_count > 0 else 0
            avg_processing_time = np.mean(list(self.processing_times)) if self.processing_times else 0
            
            # Update diagnostic metrics
            self.diagnostic_metrics['fps_history'].append(fps)
            self.diagnostic_metrics['detection_rate_history'].append(detection_rate)
            
            # Log to console (only detailed log in normal CPU mode)
            if self.perf_monitor.current_cpu_usage < 75:
                self.get_logger().info(
                    f"Performance: {fps:.1f} FPS, {detection_rate*100:.1f}% detection rate, "
                    f"{avg_processing_time*1000:.1f}ms/frame, processed {processed_count} frames "
                    f"(skipped {self.perf_monitor.skip_count})"
                )
    
    def publish_system_diagnostics(self):
        """
        Publish comprehensive system diagnostics including adaptive processing info.
        
        IMAGINE THIS: 🩺
        ---------------
        This method is like a doctor performing a comprehensive health check-up.
        It examines all aspects of the node's operation, collects vital signs,
        identifies any potential problems, and creates a detailed health report.
        
        HOW IT WORKS (STEP-BY-STEP):
        --------------------------
        1. COLLECTING METRICS ("Measuring vital signs")
           - Performance metrics: FPS, processing time, detection rate
           - Health indicators: Time since last detection, error counts
           - System resources: CPU usage, memory usage, temperature
           - Adaptive settings: Current scale factor, frame skipping
        
        2. ANALYZING HEALTH ("Diagnosing problems")
           - Identify potential issues like low FPS, high processing time
           - Check for detection problems like long periods without detection
           - Monitor system resource constraints
           - Generate warnings and errors as appropriate
        
        3. CREATING DIAGNOSTIC DATA ("Writing the medical report")
           - Create a structured JSON document containing all metrics
           - Organize information into categories for easy consumption
           - Include status indicators (active/warning/error)
           - Calculate overall health scores for different subsystems
        
        4. PUBLISHING DIAGNOSTICS ("Sharing the report")
           - Publish the complete diagnostic data to a ROS topic
           - Other monitoring systems can use this data to track node health
           - Log a condensed summary to the console for human operators
        
        REAL-WORLD ANALOGY: 🚗
        -------------------
        This is similar to the diagnostic system in modern cars:
        - Sensors throughout the car collect performance data
        - The computer analyzes this data for potential problems
        - It generates warnings when issues are detected (check engine light)
        - A technician can get a complete diagnostic report with all metrics
        - The system adapts to maintain performance despite issues
        
        This diagnostic process runs every 2 seconds, providing regular health
        updates without significantly impacting performance.
        """
        if not hasattr(self, 'diagnostic_metrics'):
            return  # Not enough data collected yet
            
        current_time = TimeUtils.now_as_float()
        elapsed_time = current_time - self.start_time
        
        # Calculate average metrics
        avg_fps = np.mean(list(self.diagnostic_metrics['fps_history'])) if self.diagnostic_metrics['fps_history'] else 0.0
        avg_processing_time = np.mean(list(self.diagnostic_metrics['processing_time_history'])) if self.diagnostic_metrics['processing_time_history'] else 0.0
        avg_detection_rate = np.mean(list(self.diagnostic_metrics['detection_rate_history'])) if self.diagnostic_metrics['detection_rate_history'] else 0.0
        
        # Time since last detection
        time_since_detection = 0
        if self.diagnostic_metrics['last_detection_time'] > 0:
            time_since_detection = current_time - self.diagnostic_metrics['last_detection_time']
        else:
            time_since_detection = float('inf')
        
        # Build warnings list
        warnings = []
        errors = []
        
        # Check for performance issues
        if avg_fps < 10.0 and elapsed_time > 10.0:
            warnings.append(f"Low FPS: {avg_fps:.1f}")
            
        if avg_processing_time > 50.0:  # 50ms is slow
            warnings.append(f"High processing time: {avg_processing_time:.1f}ms")
            
        # Check for detection issues
        if time_since_detection > 5.0 and elapsed_time > 10.0:
            warnings.append(f"No ball detected for {time_since_detection:.1f}s")
            
        if avg_detection_rate < 0.1 and elapsed_time > 10.0:  # Less than 10% detection rate
            errors.append(f"Very low detection rate: {avg_detection_rate*100:.1f}%")
        
        # System resources
        system_resources = {}
        try:
            system_resources = {
                'cpu_percent': self.perf_monitor.current_cpu_usage,  # Use stored value
                'memory_percent': psutil.virtual_memory().percent
            }
            
            # Check for high resource usage
            if system_resources['cpu_percent'] > 80.0:
                warnings.append(f"High CPU usage: {system_resources['cpu_percent']:.1f}%")
                
            # Add temperature if available
            if hasattr(psutil, 'sensors_temperatures'):
                try:
                    temps = psutil.sensors_temperatures()
                    if temps and 'cpu_thermal' in temps:
                        system_resources['temperature'] = temps['cpu_thermal'][0].current
                except:
                    # Temperature reading can fail silently
                    pass
        except Exception as e:
            # Handle any errors accessing system metrics
            self.get_logger().warn(f"Error getting system resources: {e}")
        
        # Build diagnostics data structure
        diag_data = {
            "node": "hsv",
            "timestamp": current_time,
            "uptime_seconds": elapsed_time,
            "status": "error" if errors else ("warning" if warnings else "active"),
            "health": {
                "camera_health": 1.0 - (len(warnings) * 0.1),
                "detection_health": avg_detection_rate if avg_detection_rate > 0 else 0.5,
                "processing_health": 1.0 - (avg_processing_time / 100.0) if avg_processing_time < 100.0 else 0.0,
                "overall": 1.0 - (len(errors) * 0.3) - (len(warnings) * 0.1)
            },
            "metrics": {
                "fps": avg_fps,
                "processing_time_ms": avg_processing_time,
                "total_frames": self.diagnostic_metrics['total_frames'],
                "missed_frames": self.diagnostic_metrics['missed_frames'],
                "detection_rate": avg_detection_rate
            },
            "detection": {
                "latest_position": self.diagnostic_metrics['last_detection_position'],
                "time_since_last_detection_s": time_since_detection,
                "currently_tracking": time_since_detection < 1.0
            },
            "configuration": {
                "hsv_range": {
                    "lower": self.image_processor.lower_yellow.tolist(),
                    "upper": self.image_processor.upper_yellow.tolist()
                },
                "area_range": [self.image_processor.min_ball_area, self.image_processor.max_ball_area],
                "circularity_range": [self.image_processor.min_circularity, self.image_processor.max_circularity]
            },
            "resources": system_resources,
            "adaptive_processing": {
                "current_scale_factor": self.perf_monitor.current_scale_factor,
                "frame_skip_count": self.perf_monitor.low_power_skip_frames,
                "skipped_frames": self.perf_monitor.skip_count,
                "cpu_thresholds": {
                    "high": self.perf_monitor.cpu_high_threshold,
                    "medium": self.perf_monitor.cpu_medium_threshold,
                    "low": self.perf_monitor.cpu_low_threshold
                }
            },
            "errors": errors,
            "warnings": warnings
        }
        
        # Publish as JSON
        msg = String()
        msg.data = json.dumps(diag_data)
        self.system_diagnostics_publisher.publish(msg)
        
        # Also log to console (condensed in high CPU)
        if self.perf_monitor.current_cpu_usage < self.perf_monitor.cpu_medium_threshold:
            # Normal detailed logging
            self.get_logger().info(
                f"HSV diagnostics: {avg_fps:.1f} FPS, {avg_detection_rate*100:.1f}% detection rate, "
                f"Status: {diag_data['status']}, Scale: {self.perf_monitor.current_scale_factor:.2f}, "
                f"Skip: {self.perf_monitor.low_power_skip_frames}"
            )
        else:
            # Condensed logging in high CPU
            self.get_logger().info(
                f"HSV status: {diag_data['status']}, CPU: {self.perf_monitor.current_cpu_usage:.1f}%, "
                f"Scale: {self.perf_monitor.current_scale_factor:.2f}"
            )
    
    def destroy_node(self):
        """Ensure proper cleanup of resources"""
        # Close OpenCV windows if enabled
        if hasattr(self, 'enable_visualization') and self.enable_visualization:
            try:
                cv2.destroyAllWindows()
            except Exception as e:
                self.get_logger().warn(f"Error closing OpenCV windows: {str(e)}")
        
        # Stop threads and timers
        if hasattr(self, 'resource_monitor') and self.resource_monitor:
            self.resource_monitor.stop()
            
        if hasattr(self, 'cpu_check_timer'):
            self.cpu_check_timer.cancel()
            
        # Clear cached data
        if hasattr(self, 'image_processor') and hasattr(self.image_processor, 'calculate_scaled_thresholds'):
            self.image_processor.calculate_scaled_thresholds.cache_clear()
            
        # Clear image buffers
        if hasattr(self, 'image_processor') and hasattr(self.image_processor, 'image_buffers'):
            self.image_processor.image_buffers.clear()
            
        super().destroy_node()


def main(args=None):
    """Main function to initialize and run the HSV Tennis Ball Tracker node."""
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create our HSV tennis ball tracker node
    node = HSVTennisBallTracker()
    
    # Print welcome message
    print("=================================================")
    print("Tennis Ball Tracking - HSV Ball Detector Node")
    print("=================================================")
    print("This node uses HSV color filtering to detect tennis balls.")
    print(f"Processing images at {node.target_width}x{node.target_height} to match YOLO")
    print("")
    print("Subscriptions:")
    print(f"  - Camera: {node.config_manager.topics['input']['camera']}")
    print("")
    print("Publications:")
    print(f"  - Ball position: {node.config_manager.topics['output']['position']}")
    print(f"  - CPU usage: /system/resources/cpu_load")
    print("")
    print("Performance Adaptation:")
    print(f"  - High CPU threshold: {node.perf_monitor.cpu_high_threshold}%")
    print(f"  - Medium CPU threshold: {node.perf_monitor.cpu_medium_threshold}%")
    print("")
    print("Press Ctrl+C to stop the program")
    print("=================================================")
    
    try:
        # On Pi 5, use process priority to balance with other nodes
        try:
            import os
            os.nice(5)  # Slightly lower priority than critical nodes
            print("Set HSV tracker to adjusted process priority")
        except:
            pass
        
        # Keep the node running until interrupted
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Stopping HSV tracker (Ctrl+C pressed)")
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())
    finally:
        # Clean shutdown
        if node.enable_visualization:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        print("HSV Tennis Ball Tracker has been shut down.")

if __name__ == '__main__':
    main()