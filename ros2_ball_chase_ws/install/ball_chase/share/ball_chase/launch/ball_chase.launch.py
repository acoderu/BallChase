from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ball_chase',
            executable='lidar_node',
            name='lidar_node',
            output='screen'
        ),
        Node(
            package='ball_chase',
            executable='yolo_ball_node',
            name='yolo_ball_node',
            output='screen'
        ),
        Node(
            package='ball_chase',
            executable='hsv_ball_node',
            name='hsv_ball_node',
            output='screen'
        ),
        Node(
            package='ball_chase',
            executable='depth_camera_node',
            name='depth_camera_node',
            output='screen'
        ),
        Node(
            package='ball_chase',
            executable='fusion_node',
            name='fusion_node',
            output='screen'
        )
    ])
