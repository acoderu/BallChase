#!/bin/bash
set -e

echo "[INFO] Starting BallChase real-time ROS2 nodes with CPU core pinning..."

# Paths (edit if your workspace or script paths change)
WS=~/dev/BallChase/ros2_ball_chase_ws
SRC=$WS/src/ball_chase/ball_chase/nodes

# Start YOLO node on Core 1
echo "[INFO] Launching YOLO node on Core 1..."
taskset -c 1 ros2 run ball_chase yolo_ball_node &
YOLO_PID=$!

# Start LIDAR node on Core 2
echo "[INFO] Launching LIDAR node on Core 2..."
taskset -c 2 ros2 run ball_chase lidar_node &
LIDAR_PID=$!

# Start Fusion node on Core 2
echo "[INFO] Launching Fusion node on Core 2..."
taskset -c 2 ros2 run ball_chase state_fusion_node &
FUSION_PID=$!

# Start State Manager on Core 3
echo "[INFO] Launching State Management node on Core 3..."
taskset -c 3 ros2 run ball_chase state_management_node &
STATE_PID=$!



# Allow nodes a few seconds to spin up before sending lifecycle events
sleep 5

echo "[INFO] Configuring and Activating optimized_fusion_node lifecycle..."
ros2 lifecycle set /optimized_fusion_node configure || echo "[WARN] Configure failed"
ros2 lifecycle set /optimized_fusion_node activate || echo "[WARN] Activate failed"


sleep 5

# Start PID controller on Core 3
echo "[INFO] Launching PID Controller on Core 3..."
taskset -c 3 python3 $SRC/pid_controller_node.py --ros-args -p debug_level:=3 &
PID_PID=$!

echo "[INFO] All nodes launched. PIDs: YOLO=$YOLO_PID, LIDAR=$LIDAR_PID, FUSION=$FUSION_PID, STATE=$STATE_PID, PID=$PID_PID"

# Wait for all background jobs to exit
wait
