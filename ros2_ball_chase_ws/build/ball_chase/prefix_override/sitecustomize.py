import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ubuntu/dev/BallChase/ros2_ball_chase_ws/install/ball_chase'
