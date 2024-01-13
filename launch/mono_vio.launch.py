#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

# ros2 launch bcr_bot gazebo.launch.py
# ros2 launch viro_mono_vio mono_vio.launch.py

def generate_launch_description():
    package_name = 'viro_mono_vio'

    return LaunchDescription([
        Node(
            package = package_name,
            executable='mono_vo_node.py',
            name='mono_vo_node',
            output='screen',
            parameters=[{
                'image_topic': '/kinect_camera/image_raw',
                'camera_info_topic': '/kinect_camera/camera_info'
                }]
        ),
        Node(
            package = package_name,
            executable='mono_vl_node.py',
            name='mono_vl_node',
            output='screen',
            parameters=[{
                'marker_size': 50.0,
                'image_topic': '/kinect_camera/image_raw',
                'depth_image_topic': '/kinect_camera/depth/image_raw',
                'camera_info_topic': '/kinect_camera/camera_info'
                }]
        ),
        Node(
            package = package_name,
            executable='mono_vlane_node.py',
            name='mono_vlane_node',
            output='screen',
            parameters=[{
                'image_raw_topic': '/kinect_camera/image_raw',
                'cmd_vel_topic': '/cmd_vel'
                }]
        ),
        # this node subscribes to, whatever is published at
        # localizer node topic -- vl
        # camera odom node topic -- vo
        # imu topic ------- determined by you.
        # sensor fusion will be done
        Node( 
            package = package_name,
            executable='mono_vimu_node.py',
            name='mono_vimu_node',
            output='screen',
            parameters=[{
                'imu_topic': '/imu',
                'output_pose_topic': '/mono_vo_pose_filtered'
                }]
        )
    ])
