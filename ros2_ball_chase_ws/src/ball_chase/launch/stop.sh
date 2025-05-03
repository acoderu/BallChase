#!/bin/bash
echo "[INFO] Stopping all BallChase ROS2 nodes..."

# Define node identifiers (use names or command patterns)
NODES=("yolo_ball_node" "lidar_node" "state_fusion_node" "state_management_node" "pid_controller_node.py")

for NODE in "${NODES[@]}"; do
    PIDS=$(pgrep -f "$NODE")
    if [ -z "$PIDS" ]; then
        echo "[INFO] $NODE not running."
    else
        echo "[INFO] Stopping $NODE (PID(s): $PIDS)..."
        kill $PIDS
    fi
done

# Wait for graceful shutdown
sleep 2

# Force kill if still running
for NODE in "${NODES[@]}"; do
    PIDS=$(pgrep -f "$NODE")
    if [ ! -z "$PIDS" ]; then
        echo "[WARN] $NODE still running, sending SIGKILL..."
        kill -9 $PIDS
    fi
done

echo "[INFO] All BallChase nodes stopped."
