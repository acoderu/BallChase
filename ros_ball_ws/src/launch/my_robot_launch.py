from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ball_tracking',
            executable='nodes/lidar_node',
            name='lidar_node',
            output='screen'
        ),
        Node(
            package='ball_tracking',
            executable='nodes/yolo_ball_node',
            name='yolo_ball_node',
            output='screen'
        ),
        Node(
            package='ball_tracking',
            executable='nodes/hsv_ball_node',
            name='hsv_ball_node',
            output='screen'
        ),
        Node(
            package='ball_tracking',
            executable='nodes/fusion_node',
            name='fusion_node',
            output='screen'
        ),
        Node(
            package='ball_tracking',
            executable='nodes/diagnostics_node',
            name='diagnostics_node',
            output='screen'
        ),
        Node(
            package='ball_tracking',
            executable='nodes/diagnostics_visualizer_node',
            name='diagnostics_visualizer_node',
            output='screen'
        ),
    ])
