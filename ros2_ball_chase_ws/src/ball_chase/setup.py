from setuptools import find_packages, setup
from glob import glob

package_name = 'ball_chase'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/models', glob('models/*.mnn')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='11306260+liangfuyuan@user.noreply.gitee.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],

    entry_points={
        'console_scripts': [
            'lidar_node = ball_chase.nodes.lidar_node:main',
            'yolo_ball_node = ball_chase.nodes.yolo_ball_node:main',
            'hsv_ball_node = ball_chase.nodes.hsv_ball_node:main',
            'depth_camera_node = ball_chase.nodes.depth_camera_node:main',
            'fusion_node = ball_chase.nodes.fusion_node:main',         
            'state_fusion_node = ball_chase.nodes.state_aware_fusion_node:main', 
            'base_link_fusion_node = ball_chase.nodes.base_link_fusion_node:main',
       ],
    },   
)

