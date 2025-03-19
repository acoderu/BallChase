#!/bin/bash

# Start LIDAR node
ros2 run ball_tracking nodes/lidar_node &

# Start YOLO node
ros2 run ball_tracking nodes/yolo_ball_node &

# Start HSV node
ros2 run ball_tracking nodes/hsv_ball_node &

# Start Fusion node
ros2 run ball_tracking nodes/fusion_node &

# Start Diagnostic node
ros2 run ball_tracking nodes/diagnostics_node &

# Start Diagnostic Visualizer node
ros2 run ball_tracking nodes/diagnostics_visualizer_node &

# Wait for all background processes to finish
wait
