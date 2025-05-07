"""
Basketball Tracking Robot - Optimized YOLO Detection Node
==========================================================

Project Overview:
----------------
This project involves a robotic car designed to autonomously track and follow a moving basketball.
The system uses multiple sensing modalities for robust detection:
- YOLO object detection (this node)
- HSV color-based detection
- LiDAR for depth sensing
- Depth camera for additional depth information

These different sensing modalities are combined through sensor fusion to provide reliable
tracking even when one sensing method may fail.

This Node's Purpose:
------------------
This specific node uses a lightweight YOLO neural network to detect basketballs in camera
images. The node processes raw camera frames, runs them through a pre-trained YOLO model,
and publishes the detected ball's position with a confidence score.

Optimizations Added:
------------------
- Memory reuse and pre-allocation for better performance
- Fixed-size buffer implementation for history tracking
- Reuse of message objects for publishing
- Improved logging system with throttling and categories
- Resource monitoring with adaptive behavior

Coordinate System:
----------------
- Input images are resized to 320x320 pixels
- Output coordinates are in the same 320x320 image space
- (0,0) represents the top-left corner of the image
- Published positions include:
  - x: horizontal position (0-320)
  - y: vertical position (0-320)
  - z: confidence score (0-1)

🧠 BEGINNER'S GUIDE: Neural Networks Explained
==============================================

What is a Neural Network?
------------------------
A neural network is a computing system inspired by the human brain. Just like our brains
have billions of interconnected neurons that process information, artificial neural networks
have digital "neurons" connected in layers that learn to recognize patterns.

Think of a neural network like this:
- Each artificial neuron receives multiple inputs
- It processes these inputs using weights (importance factors)
- It produces an output that gets passed to other neurons

Mathematically, a single neuron works like this:
1. It receives inputs (x₁, x₂, ..., xₙ)
2. Each input has a weight (w₁, w₂, ..., wₙ) - how important that input is
3. The neuron calculates the sum: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b (where b is a "bias")
4. It applies an "activation function" f(z) to produce the output

For example, with inputs [0.2, 0.5], weights [0.3, 0.8], and bias 0.1:
z = 0.2 × 0.3 + 0.5 × 0.8 + 0.1 = 0.06 + 0.4 + 0.1 = 0.56
Output with ReLU activation: f(0.56) = 0.56 (ReLU keeps positive values as-is)

Neural networks get their power from:
- Multiple layers of neurons (called "deep learning" when many layers)
- Training with many examples to adjust the weights
- The ability to learn complex patterns without explicit programming

    ┌─────────────┐       ┌─────────────┐       ┌─────────────┐       ┌──────────┐
    │             │       │  ●      ●   │       │  ●      ●   │       │          │
    │   Input     │       │    ●  ●     │       │    ●  ●     │       │  Output  │
    │   Layer     │──────▶│●        ●   │──────▶│●        ●   │──────▶│  Layer   │
    │  (image)    │       │  ●    ●     │       │  ●    ●     │       │(classes) │
    │             │       │    ●        │       │    ●        │       │          │
    └─────────────┘       └─────────────┘       └─────────────┘       └──────────┘
                            Hidden Layer 1        Hidden Layer 2

   Image pixels        Feature extraction     Higher-level features    Predictions
 (320×320×3 values)                                                  (80 class scores)

🔍 BEGINNER'S GUIDE: Convolutional Neural Networks (CNNs)
========================================================

What Makes CNNs Special?
----------------------
Convolutional Neural Networks (CNNs) are specialized neural networks designed for
processing grid-like data, especially images. They're inspired by how the visual cortex
in animals works.

Why are they perfect for images?
- Regular neural networks don't understand the spatial structure of images
- CNNs preserve and use this spatial information
- They're much more efficient for image processing

Key Components of CNNs:
---------------------

1. Convolutional Layers:
   - Instead of connecting to all inputs, neurons connect to small regions
   - They apply "filters" (or kernels) that slide across the image
   - Each filter detects a specific pattern (like edges, textures, or shapes)
   
   Example: A 3×3 filter applied to an image creates a new "feature map"
   
   Original Image Section:   Filter:           Result:
   ┌─────────────┐          ┌─────────┐
   │ 1  2  3  4  │          │ 1  0  1 │         
   │ 5  6  7  8  │    ×     │ 0  1  0 │   =   29 (at center position)
   │ 9  10 11 12 │          │ 1  0  1 │         
   │ 13 14 15 16 │          └─────────┘
   └─────────────┘
   
   Calculation: 1×1 + 0×2 + 1×3 + 0×5 + 1×6 + 0×7 + 1×9 + 0×10 + 1×11 = 29

2. Pooling Layers:
   - Reduce the size of feature maps to focus on important information
   - Common type: Max Pooling (take maximum value in each small region)
   - Helps make detection more robust to small image changes
   
   Max Pooling 2×2 example:
   Input:              Output:
   ┌─────────┐         ┌─────┐
   │ 1  5  3  8 │      │ 6  9 │
   │ 6  2  9  4 │  →   │ 7  5 │
   │ 3  7  1  2 │      └─────┘
   │ 5  4  5  3 │
   └─────────┘
   
   For each 2×2 region, we take the maximum value.

3. Fully Connected Layers:
   - Final layers that combine all features for classification
   - Work like traditional neural networks
   - Convert spatial features into class probabilities

How a CNN Processes an Image:
---------------------------
                 Input Image
                     ↓
    ┌─────────────────────────────┐
    │     Convolutional Layer     │  Multiple filters detect different features
    └──────────────┬──────────────┘  (edges, corners, textures)
                   ↓
    ┌─────────────────────────────┐
    │       Pooling Layer         │  Reduce size, keep important features
    └──────────────┬──────────────┘
                   ↓
    ┌─────────────────────────────┐
    │     Convolutional Layer     │  Detect more complex patterns
    └──────────────┬──────────────┘
                   ↓
                  ...                More conv & pooling layers
                   ↓
    ┌─────────────────────────────┐
    │    Fully Connected Layer    │  Combine all features
    └──────────────┬──────────────┘
                   ↓
    ┌─────────────────────────────┐
    │        Output Layer         │  Final classification results
    └─────────────────────────────┘

🚀 BEGINNER'S GUIDE: Understanding YOLO v12 (Latest)
===================================================

YOLO Evolution:
-------------
YOLO has evolved significantly since its first version. Our system uses YOLOv12,
which represents the latest advancements in real-time object detection.

How YOLO Works - From Basic to Advanced:
--------------------------------------

1. Grid-Based Approach:
   YOLO divides the image into a grid (for example, 8×8 cells).
   Each grid cell is responsible for detecting objects centered within it.
   
   ┌─┬─┬─┬─┬─┬─┬─┬─┐
   ├─┼─┼─┼─┼─┼─┼─┼─┤
   ├─┼─┼─┼─┼─┼─┼─┼─┤
   ├─┼─┼─┼─┼─┼─┼─┼─┤
   ├─┼─┼─┼─┼─┼─┼─┼─┤
   ├─┼─┼─┼─┼─┼─┼─┼─┤
   ├─┼─┼─┼─┼─┼─┼─┼─┤
   ├─┼─┼─┼─┼─┼─┼─┼─┤
   └─┴─┴─┴─┴─┴─┴─┴─┘

2. Predictions Per Cell:
   For each grid cell, YOLO predicts:
   - Bounding boxes (x, y, width, height)
   - Confidence scores (how likely the box contains an object)
   - Class probabilities (what object it might be)
   
   Mathematically, each grid cell predicts:
   - Box center: (bx, by) - relative to the cell (values 0-1)
   - Box dimensions: (bw, bh) - relative to the image (values 0-1)
   - Confidence: Pr(Object) × IOU(truth, pred) - how accurate the box is
   - Class probabilities: Pr(Class_i | Object) - for each possible class

3. Anchor Boxes:
   YOLOv12 uses predefined "anchor boxes" (template shapes) to better predict
   objects of different sizes and shapes.
   
   Instead of predicting arbitrary boxes, it predicts modifications to these templates:
   
   Anchor Box Templates:     Predictions:           Final Predictions:
   ┌───────┐                 - Offset from cell     ┌─────────┐
   │       │  Wide           - Scale adjustment     │         │ Adjusted
   └───────┘                 - Confidence           └─────────┘
   
   ┌───┐                     - Offset from cell     ┌─────┐
   │   │  Tall               - Scale adjustment     │     │ Adjusted
   └───┘                     - Confidence           └─────┘

YOLOv12 Specific Innovations:
---------------------------

1. Optimized Backbone:
   YOLOv12 uses an efficient feature extraction network optimized for mobile devices.
   
   Dense → Efficient MBConv → CSP Connection → Optimized Pyramid
   
   The "12s" in our model name indicates it's the "small" variant of YOLOv12,
   designed for resource-constrained environments like our Raspberry Pi.

2. Multi-Scale Detection:
   YOLOv12 detects objects at multiple scales using a Feature Pyramid Network (FPN).
   This helps it find both large and small objects in the same image.
   
   Large Feature Map  → Detects small objects
   Medium Feature Map → Detects medium objects
   Small Feature Map  → Detects large objects

3. Advanced Loss Function:
   YOLOv12 uses a specialized loss function that combines:
   - Box position loss (how far predicted boxes are from true boxes)
   - Confidence loss (how certain predictions should be)
   - Classification loss (how accurate object types are)
   
   Total Loss = λ₁·Coordinate_Loss + λ₂·Confidence_Loss + λ₃·Classification_Loss

4. Resource Efficiency:
   YOLOv12 is designed to run efficiently on limited hardware while maintaining accuracy:
   - MobileNetV3-inspired blocks for lightweight processing
   - Quantization-aware training for smaller model size
   - Channel pruning to remove redundant features

5. Efficiency Metrics:
   YOLOv12 320×320 (as used in our system):
   - Model size: ~3-5MB (versus 200+MB for heavier models)
   - Inference speed: 20-30 FPS on Raspberry Pi 4
   - Detection accuracy: 80-85% mAP50 (mean Average Precision at 50% IOU)

YOLOv12 vs. Previous Versions:
----------------------------
- Faster than YOLOv8-11 on edge devices due to architecture optimizations
- Better accuracy-to-size ratio than previous versions
- Improved small object detection (ideal for basketball tracking at a distance)
- Native quantization support for edge deployment
- Training enhancements for smaller datasets

🔍 BEGINNER'S GUIDE: Understanding YOLO in Practice
------------------------------------

YOLO (You Only Look Once) is a popular object detection algorithm that can find
different objects in images. Think of YOLO like a smart scanner that can quickly
spot specific items (like basketballs) in a picture.

How YOLO Works - A Simple Explanation:

1. Division: YOLO divides the image into a grid (like a checkerboard)
2. Prediction: For each grid cell, YOLO predicts:
   - Is there an object here?
   - What type of object is it?
   - Where exactly is the object (its bounding box)?
3. Confidence: YOLO gives each prediction a confidence score (how sure it is)
4. Selection: We keep only the predictions with high confidence

Visually, it works like this:

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │                 │     │ ┌─┬─┬─┬─┬─┬─┐ │     │     ┏━━━━┓      │
    │                 │     │ ├─┼─┼─┼─┼─┼─┤ │     │     ┃   ┃      │
    │    Original     │ → │ │ ├─┼─┼─┼─┼─┼─┤ │ → │ │     ┗━━━━┛      │
    │     Image       │     │ ├─┼─┼─┼─┼─┼─┤ │     │   Basketball   │
    │                 │     │ ├─┼─┼─┼─┼─┼─┤ │     │   Detection    │
    │                 │     │ └─┴─┴─┴─┴─┴─┘ │     │                 │
    └─────────────────┘     └─────────────────┘     └─────────────────┘
         Camera              Grid Analysis          Object Identification
         Input

The main steps in our YOLO detector pipeline are:

   Camera Image → Resize to 320×320 → YOLO Neural Network →
   → Find Basketball → Calculate Position & Confidence → Publish Results

YOLO is called "You Only Look Once" because unlike older methods that scan
an image multiple times looking for different objects, YOLO processes the
entire image just once - making it much faster!
"""
# ========================================================================
# 🔍 BEGINNER'S GUIDE: Understanding Python Imports
# ========================================================================
# Imports are like telling the computer which tools we need for our program.
# Think of them as gathering all the necessary equipment before starting a project.

# Basic Python tools we need (like scissors, tape, and glue)
import sys          # Helps interact with the Python system itself
import os           # Provides functions to work with files and folders
import json         # Helps work with JSON data (a way to store information)
import time         # Gives us ways to measure and work with time
from collections import deque  # A special list that's optimized for certain operations

# 🗺️ Setting up the Python path
# -----------------------------
# This is like creating a map so Python knows where to find our files.
# Imagine telling someone: "To find my house, first go to the mall, then drive north 2 miles"

# Add the parent directory of 'config' to the Python path
# This allows us to import configuration files and utilities from other folders in the project.
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Add the 'src' directory to the Python path
# This is needed so we can import our own modules easily.
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 🤖 ROS2 tools (Robot Operating System)
# -------------------------------------
# ROS is a framework that helps robots communicate between different parts.
# Think of this as importing special robot communication tools.
import rclpy                        # The main ROS2 Python library
from rclpy.node import Node         # Lets us create a ROS2 node (a program unit in ROS)
from sensor_msgs.msg import Image   # For working with camera images
from geometry_msgs.msg import PointStamped, Pose2D  # For position data
from std_msgs.msg import String, Float32MultiArray  # For basic message types
from vision_msgs.msg import BoundingBox2D           # For defining areas in images
from cv_bridge import CvBridge                      # Converts between ROS and OpenCV images

# 📷 Computer Vision and Machine Learning tools
# -------------------------------------------
# These tools help our robot "see" and understand images.
# Like giving the robot special glasses and a brain to interpret what it sees.
import MNN                # A lightweight neural network framework
import MNN.cv as mnn_cv2  # Computer vision tools within MNN
import MNN.numpy as mnn_np # Array processing within MNN
import cv2 as std_cv2     # OpenCV - a powerful computer vision library
import numpy as np        # For efficient array operations (like working with image pixels)

# 🖥️ System monitoring tools
# -------------------------
# Helps us check how hard the computer is working.
# Like gauges that show engine temperature and fuel level in a car.
import psutil            # Used to monitor CPU, memory usage, etc.

# 📦 Our own custom tools for this project
# --------------------------------------
# These are specialized tools we've built specifically for this robot.
# Like custom attachments for a power drill that serve specific purposes.
from ball_chase.utilities.resource_monitor import ResourceMonitor  # Monitors system resources
from ball_chase.utilities.time_utils import TimeUtils              # Helps with time measurements
from ball_chase.config.config_loader import ConfigLoader           # Loads configuration settings

# ========================================================================
# 🔍 BEGINNER'S GUIDE: Configuration and Settings
# ========================================================================
# Just like how a car has settings (radio volume, seat position, temperature),
# our robot code needs settings too. We store these in configuration files
# so we can change them without changing the code.

# 📝 Loading the robot's settings
# ------------------------------
# This is like reading an instruction manual before assembling furniture.
# We're getting all the settings we need to run our basketball detector.
config_loader = ConfigLoader()                   # Creates a tool to load settings
config = config_loader.load_yaml('yolo_config.yaml')  # Loads settings from a file

# 🧠 YOLO Neural Network Settings
# -----------------------------
# These settings tell our neural network how to operate - like setting the
# difficulty level and graphics quality in a video game.
#
# Think of these as adjusting the "basketball detection glasses" our robot wears:
MODEL_CONFIG = {
    "path": "yolo12s_320.mnn",    # Which YOLO model file to use (there are different versions)
    "input_width": 320,           # Width of images we'll feed to the neural network (in pixels)
    "input_height": 320,          # Height of images we'll feed to the neural network (in pixels)
    "precision": "medium",        # How precise should calculations be? (higher = more accurate but slower)
    "backend": "CPU",             # Use the CPU for calculations (not GPU)
    "thread_count": 1,            # How many CPU cores to use (more cores = faster but uses more power)
    "confidence_threshold": 0.2   # How confident must the model be to report a basketball? (0.2 = 20%)
}

# 🏀 Basketball Identifier
# ----------------------
# The YOLO model can detect many objects (people, cars, dogs, etc.)
# We need to tell it which object ID corresponds to basketballs.
#
# This is like telling our robot: "When looking at objects, we only care about item #32 - basketballs"
BASKETBALL_CLASS_ID = config.get('model', {}).get('basketball_class_id', 32)

# 📢 Communication Channels (Topics)
# --------------------------------
# Topics are like radio channels that different parts of the robot use to communicate.
# Each channel has a specific purpose - just like how emergency services use dedicated radio frequencies.
TOPICS = config.get('topics', {
    "input": {
        "camera": "/ascamera/camera_publisher/rgb0/image"  # Channel where camera images arrive
    },
    "output": {
        "position": "/basketball/yolo/position",  # Channel to broadcast basketball positions
        "bbox": "/basketball/yolo/bbox"           # Channel to broadcast basketball bounding boxes (the box around the ball)
    }
})

# 🔧 Diagnostic Settings
# --------------------
# These settings control how often we check and report the system's health.
# Like how often a car's dashboard updates its readings.
DIAG_CONFIG = config.get('diagnostics', {
    "log_interval": 15,                # How often to write status messages (in seconds)
    "performance_log_interval": 30     # How often to record performance metrics (in seconds)
})

# ========================================================================
# 🔍 BEGINNER'S GUIDE: Circular Buffers for Memory Efficiency
# ========================================================================
class LightweightBuffer:
    """
    Fixed-size buffer for storing recent detections with efficient memory usage.
    Uses pre-allocated memory to avoid dynamic allocations during runtime.
    
    Imagine this as a small circular notebook with exactly 10 pages:
    - When you fill up all 10 pages and need to write something new,
      you go back to page 1 and overwrite the oldest content
    - This way, you always have the 10 most recent entries
    - It's much faster than buying a new notebook every time!
    
    In computers, this is called a "circular buffer" or "ring buffer" and
    it's very efficient for storing recent data without using too much memory.
    """
    def __init__(self, max_size=10):
        """
        Initialize a fixed-size buffer.
        
        Args:
            max_size: Maximum number of items to store (default: 10)
        
        Think of this as preparing a notebook with exactly max_size pages:
        - We create empty slots (None) for each page
        - We start with page 0 as our next writing position
        - We track if we've gone all the way around the notebook (is_full)
        - We keep count of how many pages we've written on
        """
        self.data = [None] * max_size  # Pre-allocate list with empty spaces
        self.max_size = max_size       # How many items we can store
        self.next_index = 0            # Where to write the next item
        self.is_full = False           # Have we filled the buffer at least once?
        self.count = 0                 # How many items are currently stored
    
    def add(self, timestamp, value):
        """
        Add a timestamped value to the buffer.
        
        Args:
            timestamp: When the value was recorded
            value: The data to store
            
        This is like writing on the next available page in our notebook:
        - We write the timestamp and value at the current page
        - We move to the next page (or wrap around to page 0 if we reach the end)
        - We update our count of written pages
        - If we've gone through all pages once, we mark the notebook as "full"
        """
        try:
            # Store the data as a tuple of (timestamp, value)
            self.data[self.next_index] = (timestamp, value)
            
            # Move to the next position, wrapping around if needed
            # Example: If max_size is 10 and next_index is 9, (9+1)%10 = 0
            self.next_index = (self.next_index + 1) % self.max_size
            
            # Increase our count of stored items
            self.count += 1
            
            # If we've gone all the way around, mark as full and cap the count
            if self.count >= self.max_size:
                self.is_full = True
                self.count = self.max_size
        except Exception as e:
            # Silently handle errors to prevent crashes during data collection
            # This could happen if the buffer is accessed before initialization
            pass
    
    def get_latest(self, count=1):
        """
        Get the latest entries from the buffer.
        
        Args:
            count: How many recent items to retrieve (default: 1)
            
        Returns:
            A list of the most recent entries
            
        This is like reading the most recently written pages in our notebook:
        - If the notebook is empty, return an empty list
        - If asking for more pages than we've written, only return what we have
        - We need to figure out which pages to read based on where we last wrote
        - If we've wrapped around (notebook is full), we might need to read from
          both the end and the beginning of the notebook
        """
        # If the buffer is empty, return an empty list
        if self.count == 0:
            return []
        
        # Don't try to get more items than we actually have
        count = min(count, self.count)
        
        # If the buffer has wrapped around (is full)
        if self.is_full:
            # Calculate starting index for the range we want
            # Example: If next_index is 3 and we want 2 items, (3-2)%10 = 1
            start_idx = (self.next_index - count) % self.max_size
            
            # If the range doesn't wrap around the end of the array
            if start_idx < self.next_index:
                return self.data[start_idx:self.next_index]
            else:
                # The range wraps around, so we need to concatenate two parts
                # Example: If buffer is [5,6,7,8,9,0,1,2,3,4] and next_index is 5,
                # and we want 3 items, we return [2,3,4]
                return self.data[start_idx:] + self.data[:self.next_index]
        else:
            # Buffer hasn't wrapped around yet, just return a slice
            # Example: If next_index is 5, we've written to indices 0,1,2,3,4
            # If we want 3 items, we return data[2:5]
            return self.data[max(0, self.next_index-count):self.next_index]
    
    def clear(self):
        """
        Clear the buffer, removing all items.
        
        This is like ripping out all the written pages from our notebook
        so we can start fresh.
        """
        # Reset to initial state (but keep the pre-allocated memory)
        self.next_index = 0
        self.is_full = False
        self.count = 0

class OptimizedBasketballDetector(Node):
    """
    An optimized ROS2 node that uses a YOLO neural network to detect basketballs
    in camera images and publishes their positions with timestamp information.
    
    This node subscribes to camera images, performs basketball detection using
    a YOLO neural network model, and publishes the detected ball position
    along with confidence information.
    
    Published Topics:
    - /basketball/yolo/position (geometry_msgs/PointStamped): 
      The detected ball position with timestamp
    - /basketball/yolo/bbox (std_msgs/Float32MultiArray):
      The bounding box of the detected ball for distance estimation
      
    Subscribed Topics:
    - /ascamera/camera_publisher/rgb0/image (sensor_msgs.Image): 
      RGB camera feed
    """
    
    # 🔍 BEGINNER'S GUIDE: The YOLO Pipeline
    #
    # The YOLO neural network works in a pipeline, similar to an assembly
    # line in a factory. Each step transforms the data in a specific way:
    #
    #     ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
    #     │  Camera   │        │   Pre-    │        │   YOLO    │
    #     │   Image   │───────>│ processing│───────>│  Neural   │
    #     │  (Input)  │        │           │        │  Network  │
    #     └─────────────┘        └─────────────┘        └─────────┬─────┘
    #                                                      │
    #     ┌─────────────┐        ┌─────────────┐        ┌─────────▼─────┐
    #     │ Publishing│        │   Post-   │        │  Network  │
    #     │  Results  │<───────│ processing│<───────│  Output   │
    #     │           │        │           │        │           │
    #     └─────────────┘        └─────────────┘        └─────────────┘
    #
    # 1. We start with a camera image (RGB pixels)
    # 2. Pre-processing prepares the image for the neural network
    #    (resize, normalize values, change format)
    # 3. The YOLO neural network processes the image
    # 4. Post-processing extracts basketball detections from the network output
    # 5. Publishing sends the detected position to other robot components
    
    def __init__(self):
        """Initialize the basketball detector node."""
        # Call the parent class constructor to set up the ROS node
        super().__init__('optimized_basketball_detector')
        
        # Initialize performance tracking variables first - needed for logging
        # These help us keep track of how fast the node is running and how many images it has seen.
        self.start_time = TimeUtils.now_as_float()
        self.image_count = 0
        
        # Allocate throttle tracking dictionary early for _log function
        # This is used to prevent spamming the logs with repeated messages.
        self._throttle_times = {}
        
        # Initialize parameters with reasonable defaults
        self._init_parameters()
        
        # Initialize diagnostic metrics with pre-allocated buffers
        self._init_diagnostic_metrics()
        
        # Add Raspberry Pi resource monitoring
        self._setup_resource_monitoring()
        
        # Initialize publishers and subscribers
        self._init_ros_communication()
        
        # Bridge to convert between ROS images and OpenCV images
        self.bridge = CvBridge()
        
        # Initialize diagnostic metrics with pre-allocated buffers
        self._init_diagnostic_metrics()
        
        # Preallocate resources for reuse
        self._preallocate_resources()
        
        # Configure logging based on config
        log_config = config.get('logging', {
            'console_level': 'info',
            'publish_level': 'debug',
            'detection_log_rate': 10,
            'max_errors': 50,
            'performance_log_interval': 30
        })
        
        # Set log level
        self._configure_logging(log_config)
        
        # Initialize log file if configured
        self._init_log_file()
        
        # Log successful initialization
        self._log('info', 'INIT', f"Initialization complete, YOLO detector ready at {self.get_clock().now().to_msg().sec}.{self.get_clock().now().to_msg().nanosec//1000000:03d}s")

    def _init_parameters(self):
        """
        Initialize node parameters with reasonable defaults.
        
        This function sets up all the basic settings our basketball detector needs.
        Think of it like getting a car ready before a race - checking oil, 
        adjusting mirrors, and setting the right tire pressure.
        """
        # ⚙️ Configuration for the MNN neural network
        # ------------------------------------------
        # These settings control how our neural network will run.
        # Like selecting the right gear for a specific terrain:
        self.mnn_config = {
            "precision": MODEL_CONFIG["precision"],  # How precise calculations should be
            "backend": MODEL_CONFIG["backend"],      # Which processing unit to use (CPU)
            "numThread": MODEL_CONFIG["thread_count"], # How many CPU cores to use
        }
        
        # 🔋 Settings for power-efficient operation
        # ---------------------------------------
        # These settings help our code run better on less powerful computers
        # (like Raspberry Pi) by being more efficient.
        #
        # This is like having "eco mode" in a car to save fuel:
        
        # Check if we should run in low power mode (for Raspberry Pi)
        self.low_power_mode = config.get('raspberry_pi', {}).get('low_power_mode', False)
        
        # If in low power mode, process only every other frame to save CPU
        # (skipping frames means we look at fewer images per second)
        self.frame_skip_count = 1 if self.low_power_mode else 0
        
        # Set the minimum confidence value for accepting a detection
        # (higher values mean fewer false positives but might miss some basketballs)
        self.detection_threshold = MODEL_CONFIG["confidence_threshold"]
        
        # 🔢 Counters for tracking progress
        # -------------------------------
        # These variables keep track of how many frames we've processed
        # and help us assign numbers to our detections.
        #
        # Think of these like the odometer and trip counter in a car:
        self.frame_counter = 0   # Counts every frame we receive
        self.seq_counter = 0     # Counts every detection we publish
        
        # ⚠️ Error tracking variables
        # -------------------------
        # These variables help us know if something has gone wrong
        # and when the last error happened.
        #
        # Like warning lights on a car dashboard:
        self.model_load_failed = False  # Did we fail to load the neural network?
        self.last_error_time = 0        # When did we last encounter an error?
        
    def _setup_resource_monitoring(self):
        """
        Set up resource monitoring for the node.
        
        This function creates a system to watch the computer's resources
        (CPU, memory, temperature) to make sure our robot's brain doesn't
        overheat or get overloaded.
        """
        # 📊 BEGINNER'S GUIDE: Resource Monitoring
        # ---------------------------------------
        # Just like a car has gauges for speed, fuel, and temperature,
        # our robot needs to monitor its computer resources.
        #
        # This is important because:
        # 1. AI vision processing (YOLO) is very demanding on the computer
        # 2. Raspberry Pi computers have limited resources and can overheat
        # 3. If resources get too low, we need to adapt our behavior
        
        # Create a resource monitor that checks CPU, memory, and temperature
        self.resource_monitor = ResourceMonitor(
            node=self,                  # Tell the monitor which node to watch
            publish_interval=15.0,      # Check every 15 seconds (less frequent to save resources)
            enable_temperature=True     # Also monitor temperature (important for Raspberry Pi)
        )
        
        # Set up an alert system - when resources get too high, call our alert handler
        # This is like setting up a car alarm that goes off when the engine overheats
        self.resource_monitor.add_alert_callback(self._handle_resource_alert)
        
        # Configure warning thresholds - at what point should we take action?
        # These are high because YOLO naturally uses a lot of CPU
        self.resource_monitor.set_threshold('cpu', 90.0)      # Alert if CPU usage > 90%
        self.resource_monitor.set_threshold('memory', 90.0)   # Alert if memory usage > 90%
        
        # Start the monitor running in the background
        self.resource_monitor.start()
        
        # If we're in low power mode (for Raspberry Pi), log that information
        if self.low_power_mode:
            self._log('info', 'SYSTEM', "Running in low power mode for Raspberry Pi", throttle=0)

    def _init_ros_communication(self):
        """
        Initialize publishers and subscribers.
        
        This function sets up all the communication channels our node needs
        to send and receive information to other parts of the robot system.
        """
        # 📡 BEGINNER'S GUIDE: ROS Communication
        # --------------------------------------
        # In ROS (Robot Operating System), different parts of the robot
        # communicate through "topics" - which are like radio channels.
        # 
        # There are two ways to use topics:
        # 1. Subscribe - Listen for messages (like a radio receiver)
        # 2. Publish - Send out messages (like a radio transmitter)
        #
        # Think of it like this:
        #   - The camera node broadcasts images on its channel
        #   - We tune in to that channel to receive images
        #   - After processing, we broadcast results on our own channels
        #   - Other nodes (like the movement controller) tune in to our channels

        # 📥 Subscribe to incoming camera images
        # -------------------------------------
        # This is like tuning a radio to the station where images are broadcast.
        # Every time a new image arrives, our image_callback function will be called.
        self.subscription = self.create_subscription(
            Image,                      # The type of message (an image)
            TOPICS["input"]["camera"],  # The channel name to listen to
            self.image_callback,        # Function to call when a message arrives
            10                          # Queue size (how many messages to buffer)
        )  

        # 📤 Create publishers for our detection results
        # --------------------------------------------
        # These are like our own radio stations where we broadcast our findings.
        # Other nodes that need our results can tune in to these channels.

        # Publisher for basketball position (where we found it in the image)
        # PointStamped includes x, y coordinates and a timestamp
        self.ball_publisher = self.create_publisher(
            PointStamped,                 # Message type (position with timestamp)
            TOPICS["output"]["position"], # Channel name
            10                            # Queue size
        )  

        # Publisher for the bounding box (the rectangle around the basketball)
        # This helps other nodes know the size of the ball, not just its center
        self.bbox_publisher = self.create_publisher(
            Float32MultiArray,                            # Message type (array of numbers)
            TOPICS["output"].get("bbox", "/basketball/yolo/bbox"),  # Channel name
            10                                            # Queue size
        )

        # Publisher for system health information
        # This helps monitor how well our detector is working
        self.system_diagnostics_publisher = self.create_publisher(
            String,                        # Message type (text string)
            "/basketball/yolo/diagnostics", # Channel name
            10                             # Queue size
        )
        
        # ⏰ Create a timer for regular diagnostics
        # ---------------------------------------
        # This is like setting an alarm clock that goes off every 3 seconds,
        # reminding us to publish our diagnostic information
        self.diagnostics_timer = self.create_timer(
            3.0,                           # Time in seconds between calls
            self.publish_system_diagnostics # Function to call when timer triggers
        )

    def _configure_logging(self, log_config):
        """Configure logger levels and parameters."""
        # Map string log levels to rclpy.logging.LoggingSeverity values
        level_map = {
            'debug': rclpy.logging.LoggingSeverity.DEBUG,
            'info': rclpy.logging.LoggingSeverity.INFO,
            'warn': rclpy.logging.LoggingSeverity.WARN,
            'error': rclpy.logging.LoggingSeverity.ERROR
        }
        
        # Set logger level 
        log_level = level_map.get(log_config.get('console_level', 'info').lower(), 
                                  rclpy.logging.LoggingSeverity.INFO)
        self.get_logger().set_level(log_level)
        
        # Store log config
        self.log_config = log_config

    def _init_log_file(self):
        """Initialize log file if configured."""
        log_file = self.log_config.get('log_file')
        if not log_file:
            return
            
        import logging
        file_handler = logging.FileHandler(log_file, mode='a')
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Get the logger used by rclpy
        logger_name = self.get_name()
        logger = logging.getLogger(logger_name)
        logger.addHandler(file_handler)
        
        # Set file logging level
        file_level = self.log_config.get('file_level', 'debug').upper()
        file_handler.setLevel(getattr(logging, file_level))

    def _init_diagnostic_metrics(self):
        """Initialize metrics for diagnostic monitoring with fixed buffers."""
        # Use fixed-size buffers to avoid memory allocations
        buffer_size = 10  # Keep last 10 entries for each metric
        
        self.diagnostic_metrics = {
            # Use fixed-size buffers for time series data
            'fps_history': np.zeros(buffer_size, dtype=np.float32),
            'processing_time_history': np.zeros(buffer_size, dtype=np.float32),
            'inference_time_history': np.zeros(buffer_size, dtype=np.float32),
            'detection_rate_history': np.zeros(buffer_size, dtype=np.float32),
            'confidence_history': np.zeros(buffer_size, dtype=np.float32),
            
            # Indices for circular buffers
            'fps_idx': 0,
            'processing_idx': 0,
            'inference_idx': 0,
            'detection_rate_idx': 0,
            'confidence_idx': 0,
            
            # Initialize counters
            'total_frames': 0,
            'detected_frames': 0,
            'missed_frames': 0,
            
            # Initialize detection data
            'last_detection_position': (0.0, 0.0),
            'last_detection_time': 0.0,
            
            # Limited-size error tracking
            'errors': [],
            'warnings': []
        }
        
        # Pre-allocate error and warning arrays with max size
        self.diagnostic_metrics['errors'] = [None] * 10
        self.diagnostic_metrics['warnings'] = [None] * 10
        self.error_count = 0
        self.warning_count = 0
        
        # Use LightweightBuffer for recent detections
        self.recent_detections = LightweightBuffer(max_size=10)

    def _preallocate_resources(self):
        """Preallocate resources that will be reused during operation."""
        # Preallocate message objects for reuse
        self._position_msg = PointStamped()
        self._position_msg.header.frame_id = "ascamera_color_0"
        
        # Preallocate bbox message
        self._bbox_msg = Float32MultiArray()
        self._bbox_msg.data = [0.0, 0.0, 0.0, 0.0, 0.0]  # [center_x, center_y, width, height, confidence]
        
        # Preallocate tensor for reuse
        self._input_tensor = None
        
        # Preallocate arrays for detection processing
        self._best_box = np.zeros(4, dtype=np.float32)  # [x0, y0, x1, y1]

    def _log(self, level, category, message, data=None, throttle=0):
        """
        Unified logging with component tags, throttling, and optional data.
        This function helps us keep track of what the node is doing, and makes debugging easier.
        
        Args:
            level (str): Log level ('debug', 'info', 'warn', 'error')
            category (str): Log category ('YOLO', 'DETECT', 'STATE', etc.)
            message (str): Main log message
            data (dict): Optional data to include in debug logs
            throttle (int): Throttle interval in seconds (0 = no throttling)
        """
        # Check if throttled
        now = TimeUtils.now_as_float()
        throttle_key = f"{category}:{message}"
        
        if throttle > 0:
            # Skip if within throttle period
            if throttle_key in self._throttle_times:
                if now - self._throttle_times[throttle_key] < throttle:
                    return
            
            # Update last log time
            self._throttle_times[throttle_key] = now
        
        # Format timestamp as [T+Ns][ROS:timestamp] for precise timing
        ros_time = self.get_clock().now().to_msg()
        ros_time_str = f"{ros_time.sec}.{ros_time.nanosec//1000000:03d}"
        
        # For important messages, include both timestamps
        if category in ['DETECT', 'FUSION', 'STATE']:
            # Only calculate uptime if we need it
            uptime = now - self.start_time
            time_prefix = f"[T+{uptime:.1f}s][ROS:{ros_time_str}]"
        else:
            time_prefix = ""
        
        # Format the log message with component and optional timestamp
        if time_prefix:
            formatted_msg = f"{time_prefix}[{category}] {message}"
        else:
            formatted_msg = f"[{category}] {message}"
        
        # Add data as JSON if provided and in debug mode
        if data and level == 'debug':
            # Truncate data for console
            data_str = str(data)
            if len(data_str) > 100:
                data_str = data_str[:100] + "..."
            formatted_msg += f" | {data_str}"
        
        # Log with appropriate level
        if level == 'debug':
            self.get_logger().debug(formatted_msg)
        elif level == 'info':
            self.get_logger().info(formatted_msg)
        elif level == 'warn':
            self.get_logger().warn(formatted_msg)
        elif level == 'error':
            self.get_logger().error(formatted_msg)
            
        # Store in error/warning history if needed
        if level == 'error':
            self.diagnostic_metrics['errors'][self.error_count % 10] = {
                "time": now,
                "message": message
            }
            self.error_count += 1
        elif level == 'warn':
            self.diagnostic_metrics['warnings'][self.warning_count % 10] = {
                "time": now,
                "message": message
            }
            self.warning_count += 1

    def _add_to_metric_history(self, metric_name, value):
        """Add a value to a circular buffer metric history."""
        if metric_name + '_history' in self.diagnostic_metrics:
            idx_name = metric_name + '_idx'
            history_name = metric_name + '_history'
            
            # Ensure index exists
            if idx_name not in self.diagnostic_metrics:
                self.diagnostic_metrics[idx_name] = 0
                
            # Get current index
            current_idx = self.diagnostic_metrics[idx_name]
            
            # Validate value before storing (prevent NaN/inf)
            if isinstance(value, (int, float)) and np.isfinite(value):
                # Store value
                self.diagnostic_metrics[history_name][current_idx] = value
                
                # Update index
                self.diagnostic_metrics[idx_name] = (current_idx + 1) % len(self.diagnostic_metrics[history_name])
            else:
                # Log invalid value but don't store it
                self._log('warn', 'METRIC', f"Invalid value for {metric_name}: {value}", throttle=30)

    def _get_metric_average(self, metric_name):
        """Get the average of a metric history."""
        try:
            if metric_name + '_history' in self.diagnostic_metrics:
                values = self.diagnostic_metrics[metric_name + '_history']
                # Filter out zeros (uninitialized values) and non-finite values
                valid_values = values[(values > 0) & np.isfinite(values)]
                if len(valid_values) > 0:
                    return float(np.mean(valid_values))
        except Exception as e:
            # Quietly handle exceptions to prevent diagnostics from breaking the node
            self._log('debug', 'METRIC', f"Error getting metric average for {metric_name}: {str(e)}", throttle=30)
        return 0.0

    def _trace_start(self, operation):
        """Start timing an operation."""
        if not hasattr(self, 'trace_points'):
            self.trace_points = {}
        self.trace_points[operation] = TimeUtils.now_as_float()

    def _trace_end(self, operation):
        """End timing an operation and return elapsed time in ms."""
        if hasattr(self, 'trace_points') and operation in self.trace_points:
            elapsed = (TimeUtils.now_as_float() - self.trace_points[operation]) * 1000
            return elapsed
        return 0.0

    def load_model(self, config):
        """
        Load the YOLO model for tennis ball detection.
        This function loads the neural network that will help us find basketballs in images.
        
        Args:
            config (dict): Configuration parameters for the MNN runtime
        """
        
        # 🔍 BEGINNER'S GUIDE: Neural Network Models
        #
        # A neural network model is like a recipe book that the computer follows to
        # recognize objects. In our case, the model has been trained to recognize
        # basketballs. Here's what happens when we load a model:
        #
        #                  Model File (.mnn)
        #                 ┌───────────────┐
        #                 │     YOLO      │
        #                 │    Neural     │
        #                 │    Network    │
        #                 │    Weights    │
        #                 └───────┬───────┘
        #                         │
        #                         ▼
        #         ┌─────────────────────────────┐
        #         │        MNN Library         │
        #         │ (loads model into memory)  │
        #         └──────────┬────────────────┘
        #                      │
        #                      ▼
        #         ┌─────────────────────────────┐
        #         │ Ready-to-use Neural Network│
        #         │ (can now detect objects)   │
        #         └─────────────────────────────┘
        #
        # Think of it like loading a specialized tool - first we get the tool
        # from storage (the model file), then we set it up (load into memory),
        # and finally it's ready to use (for detecting basketballs).
        try:
            # Get model filename from the path in config
            model_filename = os.path.basename(MODEL_CONFIG["path"])
            
            # Use get_package_share_directory to find the model path
            from ament_index_python.packages import get_package_share_directory
            package_share_dir = get_package_share_directory('ball_chase')
            model_path = os.path.join(package_share_dir, 'models', model_filename)
            
            self._log('info', 'MODEL', f"Loading model from {model_path}...")
            
            # Check if model exists
            if not os.path.exists(model_path):
                self._log('error', 'MODEL', f"Model not found at {model_path}")
                self.model_load_failed = True
                self.net = None
                return
            
            # Initialize MNN runtime manager with our configuration
            self.runtime_manager = MNN.nn.create_runtime_manager((config,))
            
            # Load the YOLO model from file using the absolute path
            self.net = MNN.nn.load_module_from_file(
                model_path, [], [], runtime_manager=self.runtime_manager
            )
            
            # Create a test image to warm up the model
            dummy_image = np.zeros(
                (3, MODEL_CONFIG["input_height"], MODEL_CONFIG["input_width"]), 
                dtype=np.float32
            )
            dummy_tensor = MNN.expr.const(
                dummy_image, 
                [3, MODEL_CONFIG["input_height"], MODEL_CONFIG["input_width"]], 
                MNN.expr.NCHW
            )
            dummy_tensor = MNN.expr.convert(dummy_tensor, MNN.expr.NC4HW4)
            dummy_input = MNN.expr.reshape(
                dummy_tensor, 
                [1, 3, MODEL_CONFIG["input_height"], MODEL_CONFIG["input_width"]]
            )
            
            try:
                # Run the model once to warm it up
                self.net.forward(dummy_input)
                self._log('info', 'MODEL', "YOLO model loaded and warmed up successfully")
            except Exception as e:
                self._log('warn', 'MODEL', f"Model loaded but warmup failed: {str(e)}")
                # Continue even if warmup fails - the model might still work
                
        except Exception as e:
            self._log('error', 'MODEL', f"Failed to load model: {str(e)}")
            # Don't raise, continue with graceful degradation
            self.model_load_failed = True

    def preprocess_image(self, cv_image):
        """
        Preprocess the camera image for YOLO inference.
        This function prepares the image so the neural network can understand it.
        """
        
        # 🔍 BEGINNER'S GUIDE: Image Pre-processing
        #
        # Neural networks need images in a specific format to work correctly.
        # Pre-processing is like translating from human language to computer language.
        #
        # Here's what happens to our image:
        #
        #   Original Image        Resized Image        Normalized Data       Formatted Tensor
        #  ┌─────────────┐       ┌─────────┐           ┌─────────┐           ┌─────────┐
        #  │             │       │         │           │         │           │C        │
        #  │   Varies    │  →   │ 320x320 │    →     │ Values  │    →     │H Pixels │
        #  │   in size   │       │ pixels  │           │ 0 to 1  │           │W        │
        #  │             │       │         │           │         │           │         │
        #  └─────────────┘       └─────────┘           └─────────┘           └─────────┘
        #
        # 1. Resize: Make all images exactly 320x320 pixels
        # 2. Color Format: Convert from BGR (OpenCV) to RGB (YOLO)
        # 3. Normalize: Change pixel values from 0-255 to 0-1 (neural networks prefer this)
        # 4. Reformat: Change from height×width×channels to channels×height×width format
        # 5. Tensor Creation: Package the image data for the neural network
        #
        # This is similar to preparing ingredients before cooking - everything needs
        # to be the right size and format before the neural network can use it.
        # Resize the image to what our model expects (320x320)
        if (cv_image.shape[0] != MODEL_CONFIG["input_height"] or 
            cv_image.shape[1] != MODEL_CONFIG["input_width"]):
            cv_image = std_cv2.resize(
                cv_image, 
                (MODEL_CONFIG["input_width"], MODEL_CONFIG["input_height"])
            )
        # Convert from BGR (OpenCV format) to RGB (what our model expects)
        rgb_image = cv_image[..., ::-1]  # Swaps color channels
        # Normalize pixel values to [0,1] range for neural network input
        rgb_image = rgb_image.astype(np.float32) * (1.0/255.0)
        # Change image format from HWC (height, width, channels) to CHW (channels, height, width)
        chw_image = np.transpose(rgb_image, (2, 0, 1))
        # Create an MNN tensor from our image (for inference)
        input_tensor = MNN.expr.const(
            chw_image, 
            [3, MODEL_CONFIG["input_height"], MODEL_CONFIG["input_width"]], 
            MNN.expr.NCHW
        )
        # Convert to NC4HW4 format for MNN backend (optimized for speed)
        input_tensor = MNN.expr.convert(input_tensor, MNN.expr.NC4HW4)
        # Add batch dimension (1 image per batch)
        input_tensor = MNN.expr.reshape(
            input_tensor, 
            [1, 3, MODEL_CONFIG["input_height"], MODEL_CONFIG["input_width"]]
        )
        return input_tensor

    def process_detections(self, output_var):
        """
        Process YOLO output to extract basketball detections.
        This function takes the output from the neural network and figures out where the basketball is.
        """
        
        # 🔍 BEGINNER'S GUIDE: Understanding YOLO Output
        #
        # The neural network doesn't directly tell us "there's a basketball at (x,y)".
        # Instead, it gives us encoded data that we need to interpret.
        #
        # Here's what YOLO outputs and how we process it:
        #
        #   Neural Network Output                 Processing Steps                 Final Result
        #  ┌─────────────────────┐              ┌──────────────────┐               ┌─────────┐
        #  │                     │  Extract     │ Center: (cx,cy)  │  Convert to   │   ┏━━┓   │
        #  │  Complex encoded    │─────────────>│ Size: (w,h)      │─────────────>│   ┃ ┃   │
        #  │  detection data     │  coordinates │ Class scores     │  basketball  │   ┗━━┛   │
        #  │                     │              │ Confidence       │  detection    │         │
        #  └─────────────────────┘              └──────────────────┘               └─────────┘
        #
        # 1. First, we convert the network output to a usable format
        # 2. We extract the center coordinates (cx, cy) of detected objects
        # 3. We extract the width and height (w, h) of the detection boxes
        # 4. We extract probability scores for each possible object class
        # 5. We find the highest probability class and its confidence score
        # 6. We filter to keep only basketball detections (class ID 32)
        # 7. We keep only detections with confidence above our threshold
        # 8. If multiple basketballs are detected, we take the one with highest confidence
        # 9. We calculate the bounding box corners from the center and dimensions
        # 10. We adjust the confidence based on the aspect ratio (basketballs should be round)
        #
        # This is like a detective piecing together clues to figure out where
        # the basketball is in the image and how confident we are about it.
        # Convert model output to NCHW format and remove batch dimension
        output_var = MNN.expr.convert(output_var, MNN.expr.NCHW).squeeze()
        # Extract detection data: center x/y, width, height, and class probabilities
        cx, cy = output_var[0], output_var[1]  # Center coordinates
        w, h = output_var[2], output_var[3]    # Width and height
        probs = output_var[4:]                 # Class probabilities
        # Convert from center format to corner format (x0, y0, x1, y1)
        x0 = cx - w * 0.5  # Top-left x
        y0 = cy - h * 0.5  # Top-left y
        x1 = cx + w * 0.5  # Bottom-right x
        y1 = cy + h * 0.5  # Bottom-right y
        # Combine into boxes array
        boxes = mnn_np.stack([x0, y0, x1, y1], axis=1)
        # Get confidence scores and class IDs for each detection
        scores = mnn_np.max(probs, axis=0)         # Highest probability for any class
        class_ids = mnn_np.argmax(probs, axis=0)   # Which class has highest probability
        # Find all basketball detections with confidence above threshold
        basketball_indices = []
        for i in range(len(class_ids)):
            if scores[i] < 0.01:
                continue
            if (class_ids[i] == BASKETBALL_CLASS_ID and 
                scores[i] > MODEL_CONFIG["confidence_threshold"]):
                basketball_indices.append(i)
        # If no basketballs found, return None
        if not basketball_indices:
            return None, 0.0
        # If multiple basketballs detected, take the one with highest confidence
        best_idx = basketball_indices[0]
        for idx in basketball_indices:
            if scores[idx] > scores[best_idx]:
                best_idx = idx
        # Get the box coordinates for our best detection
        box = boxes[best_idx]
        x0_val, y0_val, x1_val, y1_val = box.read_as_tuple()
        # Calculate confidence adjustments
        base_confidence = scores[best_idx]
        # Adjust confidence based on aspect ratio (basketballs should be round)
        width = x1_val - x0_val
        height = y1_val - y0_val
        aspect_ratio = width / height if height > 0 else 1.0
        size_confidence = 1.0 - abs(1.0 - aspect_ratio) * 0.5
        # Final confidence combines model confidence and size confidence
        final_confidence = base_confidence * size_confidence
        best_box = [x0_val, y0_val, x1_val, y1_val]
        return best_box, final_confidence

    def image_callback(self, msg):
        """
        Process each incoming camera image to detect tennis balls.
        This function is called every time a new image arrives from the camera.
        It runs the neural network, finds the basketball, and publishes the result.
        """
        
        # 🔍 BEGINNER'S GUIDE: Camera Callbacks in Robotics
        #
        # This function is like a factory worker who processes images as they arrive
        # on a conveyor belt. Every time the camera captures a new image, this
        # function gets called automatically to process it.
        #
        # The image processing pipeline looks like this:
        #
        #              ┌─────────┐
        #              │ Camera  │
        #              │ Image   │
        #              └────┬────┘
        #                   │
        #                   ▼
        #         ┌──────────────────┐    Skip frame if    ┌─────────┐
        #         │ New Frame Arrived│─── low power mode ──>│  Done   │
        #         └────────┬─────────┘      is active       └─────────┘
        #                  │
        #                  ▼
        #         ┌──────────────────┐
        #         │ Convert to OpenCV│
        #         │ Image Format     │
        #         └────────┬─────────┘
        #                  │
        #                  ▼
        #         ┌──────────────────┐
        #         │  Pre-process     │
        #         │  Image           │
        #         └────────┬─────────┘
        #                  │
        #                  ▼
        #         ┌──────────────────┐
        #         │  Run YOLO Neural │
        #         │  Network         │
        #         └────────┬─────────┘
        #                  │
        #                  ▼
        #         ┌──────────────────┐      No ball      ┌─────────┐
        #         │  Process YOLO    │──── detected ─────>│  Done   │
        #         │  Output          │                   └─────────┘
        #         └────────┬─────────┘
        #                  │ Ball found!
        #                  ▼
        #         ┌──────────────────┐
        #         │ Calculate Center │
        #         │ & Confidence     │
        #         └────────┬─────────┘
        #                  │
        #                  ▼
        #         ┌──────────────────┐
        #         │ Publish Position │
        #         │ & Bounding Box   │
        #         └────────┬─────────┘
        #                  │
        #                  ▼
        #         ┌──────────────────┐
        #         │Update Performance│
        #         │Metrics & Log     │
        #         └──────────────────┘
        #
        # This callback is the heart of our detection system, taking raw camera
        # images and turning them into information the robot can use to track
        # the basketball.
        self._trace_start('overall')
        # Skip frames based on low power mode
        self.frame_counter += 1
        if self.frame_skip_count > 0 and (self.frame_counter % (self.frame_skip_count + 1)) != 0:
            return
        # Check if model failed to load or is not available
        if not hasattr(self, 'net') or self.net is None:
            now = TimeUtils.now_as_float()
            # Only log error every 10 seconds to avoid flooding
            if not hasattr(self, 'last_error_time') or now - self.last_error_time > 10:
                self._log('error', 'MODEL', "Cannot process image - model not available", throttle=10)
                self.last_error_time = now
            return
        # Update statistics
        inference_start = TimeUtils.now_as_float()
        self.last_callback_time = inference_start  # Track last callback time
        self.image_count += 1
        self.diagnostic_metrics['total_frames'] = self.image_count
        try:
            # Convert ROS image to OpenCV format (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Preprocess the image for model input (resize, normalize, format)
            self._trace_start('preprocess')
            input_tensor = self.preprocess_image(cv_image)
            preprocess_time = self._trace_end('preprocess')
            # Run the neural network to detect objects
            self._trace_start('inference')
            output_var = self.net.forward(input_tensor)
            inference_time = self._trace_end('inference')
            # Track inference time for diagnostics
            self._add_to_metric_history('inference_time', inference_time)
            # Process the model output to find tennis balls
            self._trace_start('postprocess')
            best_box, confidence = self.process_detections(output_var)
            postprocess_time = self._trace_end('postprocess')
            # If a tennis ball was detected, publish its position
            if best_box is not None:
                x0, y0, x1, y1 = best_box
                # Calculate center point of the tennis ball
                center_x = (x0 + x1) / 2
                center_y = (y0 + y1) / 2
                width = x1 - x0
                height = y1 - y0
                # Update diagnostic metrics for detection
                self.diagnostic_metrics['detected_frames'] += 1
                self.diagnostic_metrics['last_detection_position'] = (center_x, center_y)
                self.diagnostic_metrics['last_detection_time'] = TimeUtils.now_as_float()
                # Update metrics
                self._add_to_metric_history('confidence', confidence)
                # Calculate detection rate
                detection_rate = self.diagnostic_metrics['detected_frames'] / self.image_count
                self._add_to_metric_history('detection_rate', detection_rate)
                # Log detection with improved formatting similar to fusion node
                # Only log if confidence is above threshold or periodically
                should_log = (confidence > 0.3 or 
                             (self.image_count % self.log_config.get('detection_log_rate', 10) == 0))
                if should_log:
                    self._log('info', 'DETECT', 
                             f"Ball at ({center_x:.1f}, {center_y:.1f}), Size: {width:.1f}x{height:.1f}, Conf: {confidence:.2f}", 
                             throttle=0)  # No throttling for actual detections
                # Create and publish the position message with timestamp
                # Use pre-allocated message object
                # Validate the timestamp
                if TimeUtils.is_timestamp_valid(msg.header.stamp):
                    self._position_msg.header.stamp = msg.header.stamp
                else:
                    self._position_msg.header.stamp = TimeUtils.now_as_ros_time()
                # Increment sequence counter
                self.seq_counter += 1
                # Set position data (x, y = center, z = confidence)
                self._position_msg.point.x = float(center_x)
                self._position_msg.point.y = float(center_y)
                self._position_msg.point.z = float(confidence)  # Using z for confidence
                # Publish position
                self.ball_publisher.publish(self._position_msg)
                # Use pre-allocated bbox message
                # Format: [center_x, center_y, width, height, confidence]
                self._bbox_msg.data[0] = float(center_x)
                self._bbox_msg.data[1] = float(center_y)
                self._bbox_msg.data[2] = float(width)
                self._bbox_msg.data[3] = float(height)
                self._bbox_msg.data[4] = float(confidence)
                # Publish the bounding box
                self.bbox_publisher.publish(self._bbox_msg)
                # Store detailed data for high-confidence detections
                if confidence > 0.5:
                    # Use lightweight buffer instead of deque
                    self.recent_detections.add(TimeUtils.now_as_float(), {
                        "position": (float(center_x), float(center_y)),
                        "size": (float(width), float(height)),
                        "confidence": float(confidence),
                        "aspect_ratio": float(width/height) if height > 0 else 1.0,
                        "frame_id": self.seq_counter
                    })
            else:
                # Only log "no detection" occasionally to avoid flooding logs
                if self.image_count % 30 == 0:
                    self._log('debug', 'DETECT', "No basketball detected in recent frames", throttle=5)
            # Calculate and display performance metrics
            total_time = (TimeUtils.now_as_float() - inference_start) * 1000  # milliseconds
            elapsed_time = TimeUtils.now_as_float() - self.start_time
            fps = self.image_count / elapsed_time if elapsed_time > 0 else 0
            # Update metrics for diagnostics
            self._add_to_metric_history('fps', fps)
            self._add_to_metric_history('processing_time', total_time)
            # Log performance periodically (throttled)
            if self.image_count % DIAG_CONFIG["performance_log_interval"] == 0:
                self._log('info', 'PERF', 
                         f"FPS: {fps:.1f}, Processing: {total_time:.1f}ms, Inference: {inference_time:.1f}ms", 
                         throttle=DIAG_CONFIG["performance_log_interval"])
            overall_time = self._trace_end('overall')
            # Log detailed timing occasionally (highly throttled)
            if self.image_count % self.log_config.get('trace_log_interval', 100) == 0:
                detail = {
                    "preprocess_ms": preprocess_time,
                    "inference_ms": inference_time,
                    "postprocess_ms": postprocess_time,
                    "overhead_ms": overall_time - (preprocess_time + inference_time + postprocess_time)
                }
                self._log('debug', 'TRACE', "YOLO timing breakdown", data=detail, throttle=30)
        except Exception as e:
            self._log('error', 'ERROR', f"Error processing image: {str(e)}")

    def publish_system_diagnostics(self):
        """Publish comprehensive system diagnostics."""
        current_time = TimeUtils.now_as_float()
        elapsed_time = current_time - self.start_time
        ros_time = self.get_clock().now().to_msg()
        
        # Calculate average metrics
        avg_fps = self._get_metric_average('fps')
        avg_processing_time = self._get_metric_average('processing_time')
        avg_inference_time = self._get_metric_average('inference_time')
        avg_detection_rate = self._get_metric_average('detection_rate')
        avg_confidence = self._get_metric_average('confidence')
        
        # Time since last detection
        if self.diagnostic_metrics['last_detection_time'] > 0:
            time_since_detection = current_time - self.diagnostic_metrics['last_detection_time']
        else:
            time_since_detection = float('inf')
        
        # Build warnings list
        warnings = []
        errors = []
        
        # Check for performance issues
        if avg_fps < 5.0 and elapsed_time > 20.0:
            warnings.append(f"Low FPS: {avg_fps:.1f}")
            
        if avg_processing_time > 150.0:
            warnings.append(f"High processing time: {avg_processing_time:.1f}ms")
            
        # Check for detection issues
        if time_since_detection > 10.0 and elapsed_time > 20.0:
            warnings.append(f"No ball detected for {time_since_detection:.1f}s")
            
        if avg_detection_rate < 0.05 and elapsed_time > 20.0:
            errors.append(f"Very low detection rate: {avg_detection_rate*100:.1f}%")
        
        # Check model status
        if hasattr(self, 'model_load_failed') and self.model_load_failed:
            errors.append("Model failed to load")
        elif not hasattr(self, 'net') or self.net is None:
            errors.append("Model not available")
        
        # System resources
        system_resources = {
            'cpu_percent': psutil.cpu_percent(interval=None),
            'memory_percent': psutil.virtual_memory().percent,
            'threads': MODEL_CONFIG["thread_count"]
        }
        
        # Check for high resource usage
        if system_resources['cpu_percent'] > 90.0:
            warnings.append(f"Critical CPU usage: {system_resources['cpu_percent']:.1f}%")
            
        # Add temperature if available
        if hasattr(psutil, 'sensors_temperatures'):
            temps = psutil.sensors_temperatures()
            if temps and 'cpu_thermal' in temps:
                system_resources['temperature'] = temps['cpu_thermal'][0].current
                if system_resources['temperature'] > 80.0:
                    warnings.append(f"High CPU temperature: {system_resources['temperature']:.1f}°C")
        
        # Build diagnostics data structure
        diag_data = {
            "node": "yolo",
            "timestamp": current_time,
            "ros_time": f"{ros_time.sec}.{ros_time.nanosec//1000000:03d}",
            "uptime_seconds": elapsed_time,
            "status": "error" if errors else ("warning" if warnings else "active"),
            "health": {
                "model_health": 0.0 if (not hasattr(self, 'net') or self.net is None) else (avg_confidence * 0.8 if avg_confidence > 0 else 0.5),
                "detection_health": avg_detection_rate if avg_detection_rate > 0 else 0.0,
                "processing_health": 1.0 - (avg_processing_time / 200.0) if avg_processing_time < 200.0 else 0.0,
                "overall": 0.0 if (not hasattr(self, 'net') or self.net is None) else (1.0 - (len(errors) * 0.3) - (len(warnings) * 0.1))
            },
            "metrics": {
                "fps": avg_fps,
                "processing_time_ms": avg_processing_time,
                "inference_time_ms": avg_inference_time,
                "total_frames": self.diagnostic_metrics['total_frames'],
                "detected_frames": self.diagnostic_metrics['detected_frames'],
                "detection_rate": avg_detection_rate
            },
            "detection": {
                "latest_position": self.diagnostic_metrics['last_detection_position'],
                "time_since_last_detection_s": time_since_detection,
                "currently_tracking": time_since_detection < 2.0,
                "average_confidence": avg_confidence
            },
            "configuration": {
                "model": MODEL_CONFIG["path"],
                "precision": MODEL_CONFIG["precision"],
                "backend": MODEL_CONFIG["backend"],
                "confidence_threshold": MODEL_CONFIG["confidence_threshold"],
                "low_power_mode": self.low_power_mode
            },
            "resources": system_resources,
            "errors": errors,
            "warnings": warnings
        }

        # Add recent detections if available
        if hasattr(self, 'recent_detections'):
            recent_data = self.recent_detections.get_latest(5)  # Get last 5 detections
            if recent_data and len(recent_data) > 0 and recent_data[0] is not None:
                diag_data["detection"]["recent_detections"] = recent_data
        
        # Run health check
        health_issues = self.perform_health_check()
        if health_issues:
            for issue in health_issues:
                self._log('warn', 'HEALTH', issue, throttle=30)
            # Add to warnings
            warnings.extend(health_issues)

        # Publish as JSON
        msg = String()
        msg.data = json.dumps(diag_data)
        self.system_diagnostics_publisher.publish(msg)
        
        # Also log to console in a consistent format like the fusion node
        status_char = "✓" if diag_data['status'] == "active" else ("!" if diag_data['status'] == "warning" else "✗")
        self._log('info', 'DIAG', 
                 f"{status_char} {avg_fps:.1f} FPS, {avg_detection_rate*100:.1f}% detection rate, "
                 f"Avg inference: {avg_inference_time:.1f}ms, Status: {diag_data['status']}",
                 throttle=0)  # Don't throttle diagnostics

    def perform_health_check(self):
        """Run periodic health checks and self-diagnostics."""
        health_issues = []
        
        # Check model state - prioritize model checks
        if hasattr(self, 'model_load_failed') and self.model_load_failed:
            health_issues.append("YOLO model failed to load")
        elif not hasattr(self, 'net') or self.net is None:
            health_issues.append("YOLO model not loaded")
            
            # Try to reload the model if it's been a while since the last attempt
            current_time = TimeUtils.now_as_float()
            if (not hasattr(self, 'last_model_reload_attempt') or 
                current_time - self.last_model_reload_attempt > 60.0):  # Try once per minute
                self._log('info', 'MODEL', "Attempting to reload YOLO model...")
                self.last_model_reload_attempt = current_time
                self.load_model(self.mnn_config)
                
                # Check if reload was successful
                if hasattr(self, 'net') and self.net is not None:
                    self._log('info', 'MODEL', "Model reload successful!")


    def _handle_resource_alert(self, resource_type, value):
        """
        Handle high resource usage by adjusting detector behavior.
        
        This function is called when the computer's resources (CPU, memory)
        get too high. It can change the detector's behavior to use fewer 
        resources, like skipping some camera frames to reduce load.
        
        Args:
            resource_type: What resource is high ('cpu' or 'memory')
            value: How high the resource usage is (percentage)
        """
        # 🔄 BEGINNER'S GUIDE: Adaptive Behavior in Robotics
        # -------------------------------------------------
        # Smart robots need to adapt to their environment - including their
        # own internal environment (how hard their computer is working).
        #
        # Think of this like a car's automatic transmission:
        # - When going uphill (high CPU), it shifts to a lower gear (low power mode)
        # - This sacrifices some speed for reliability
        # - When resources return to normal, it can shift back to high performance
        #
        # This is called "feedback control" - the system monitors itself and
        # adjusts automatically to maintain reliable operation.
        
        # Log a warning message about high resource usage
        self._log('warn', 'SYSTEM', 
                 f"Resource alert: {resource_type} at {value:.1f}% - may affect performance",
                 throttle=30)  # Only log once per 30 seconds to avoid spamming
        
        # Record this warning in our diagnostic history for later analysis
        if hasattr(self, 'diagnostic_metrics') and 'warnings' in self.diagnostic_metrics:
            # Create the warning message
            warning_msg = f"High {resource_type}: {value:.1f}%"
            
            # Check if this exact warning is already recorded
            already_warned = False
            for w in self.diagnostic_metrics['warnings']:
                if w and w.get('message', '') == warning_msg:
                    already_warned = True
                    break
                    
            # If not already warned and we have space to store more warnings
            if not already_warned and self.warning_count < len(self.diagnostic_metrics['warnings']):
                # Add the warning to our history
                self.diagnostic_metrics['warnings'][self.warning_count] = {
                    "time": TimeUtils.now_as_float(),  # When it happened
                    "message": warning_msg             # What happened
                }
                self.warning_count += 1
        
        # If CPU usage is extremely high (95%+), take automatic action
        if resource_type == 'cpu' and value > 95.0 and not self.low_power_mode:
            # Log that we're entering low power mode
            self._log('warn', 'SYSTEM', 
                     "Enabling low power mode due to high CPU usage",
                     throttle=60)  # Log at most once per minute
                     
            # Enable low power mode
            self.low_power_mode = True
            
            # Skip every other frame to reduce CPU load by ~50%
            # This means we'll process 15 frames/second instead of 30 frames/second
            self.frame_skip_count = 1  # Skip every other frame

    def destroy_node(self):
        """
        Clean up YOLO model resources.
        
        Textbook Explanation:
        ---------------------
        When shutting down a robotics node, it's important to release all resources (memory, model objects, background processes) to avoid memory leaks and ensure a clean shutdown. This function deletes the neural network, releases memory, stops resource monitoring, and triggers garbage collection.
        
        Mathematical/Intuitive Link:
        - Resource management is like cleaning up after an experiment: you want to leave the system in a good state for the next run.
        - Proper cleanup prevents memory leaks, which can slow down or crash the robot over time.
        """
        self._log('info', 'SHUTDOWN', "Cleaning up resources...")
        
        # Release model resources
        if hasattr(self, 'net'):
            del self.net
        
        # Release pre-allocated tensors
        if hasattr(self, '_input_tensor'):
            self._input_tensor = None
        
        # Stop resource monitor
        if hasattr(self, 'resource_monitor'):
            self.resource_monitor.stop()
            
        # Force garbage collection to clean up ML resources
        import gc
        gc.collect()
        
        self._log('info', 'SHUTDOWN', "YOLO detector resources released")
        super().destroy_node()

# The main function is the entry point of the program. It sets up ROS, creates the node, and keeps it running.
def main(args=None):
    """
    Main function to initialize and run the basketball detector.
    
    Textbook Explanation:
    ---------------------
    This is the entry point of the program. It sets up the ROS system, creates the detector node, and keeps it running until interrupted. This is a standard pattern in robotics: initialize, run, and clean up on exit.
    
    Mathematical/Intuitive Link:
    - The main loop (rclpy.spin) keeps the node alive, processing messages and running callbacks.
    - Clean shutdown ensures all resources are released and the robot is ready for the next session.
    """
    
    # 🔍 BEGINNER'S GUIDE: Complete YOLO Object Detection Process
    #
    # To help visualize how YOLO detection works from start to finish,
    # here's the complete process in one overview diagram:
    #
    #   ┌───────────────────────────────────────────────────────────────────────────┐
    #   │                         YOLO Detection Process                         │
    #   └───────────────────────────────────────────────────────────────────────────┘
    #                                     │
    #                                     ▼
    #   ┌───────────────┐      ┌───────────────────┐      ┌───────────────────┐
    #   │Camera captures│      │  Image divided    │      │  Each grid cell   │
    #   │    image      │─────>│  into grid cells  │─────>│predicts potential │
    #   │ (320x320)     │      │    (e.g. 8x8)     │      │    objects        │
    #   └───────────────┘      └───────────────────┘      └───────────────────┘
    #                                                                 │
    #   ┌───────────────┐      ┌───────────────────┐                 │
    #   │ Basketball    │      │  Best detection   │                 │
    #   │ position      │<─────│  selected by      │<────────────────┘
    #   │ published     │      │  confidence       │
    #   └───────────────┘      └───────────────────┘
    #           │
    #           ▼
    #   ┌──────────────────────────────────────────┐
    #   │ Robot uses position to track & follow ball │
    #   └──────────────────────────────────────────┘
    #
    # Key Innovation of YOLO: Instead of scanning the image multiple times
    # at different scales (which is slow), YOLO processes the entire image
    # at once ("You Only Look Once"), making it much faster than previous
    # object detection methods - perfect for real-time robotics!
    # Initialize ROS
    rclpy.init(args=args)
    
    # Create our basketball detector node
    node = OptimizedBasketballDetector()
    
    print("Optimized YOLO Basketball Detector is now running! Press Ctrl+C to stop.")
    
    try:
        # Keep the node running until interrupted
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Stopping YOLO detector (Ctrl+C pressed)")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Clean shutdown
        node.destroy_node()
        rclpy.shutdown()
        print("YOLO Basketball Detector has been shut down.")

if __name__ == '__main__':
    main()