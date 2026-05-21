""" Static transform publisher acquired via MoveIt 2 hand-eye calibration """
""" EYE-TO-HAND: world -> camera_color_optical_frame """
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nodes = [
        Node(
            package="goc_demo",
            executable="tf_tweaker",
            output="log",
            arguments=[
                "--parent",
                "world",
                "--frame",
                "camera_color_optical_frame",
                "--translation", "-0.7789", "0.9351", "0.3168",
                "--quaternion", "-0.0423", "0.7623", "-0.6446", "0.0388",
                # "--roll",
                # "1.73858",
                # "--pitch",
                # "0.113961",
                # "--yaw",
                # "-3.13419",
            ],
        ),
    ]
    return LaunchDescription(nodes)
