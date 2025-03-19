#!/bin/bash

# Get a list of all running nodes
nodes=$(ros2 node list)

# Filter nodes that belong to the ball_tracking/nodes directory
for node in $nodes; do
    if [[ $node == */ball_tracking/nodes/* ]]; then
        # Gracefully shut down the node
        ros2 lifecycle set $node shutdown
    fi
done
