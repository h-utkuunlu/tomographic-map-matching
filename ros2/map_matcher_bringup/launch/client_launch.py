import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    unitree_id = LaunchConfiguration("unitree_id")

    unitree_id_arg = DeclareLaunchArgument(
        "unitree_id", default_value=EnvironmentVariable("UNITREE_ID")
    )

    matcher_server = Node(
        package="map_matcher",
        executable="map_matcher_client",
        name="matcher_client",
        namespace=unitree_id,
        output="screen",
        emulate_tty=True,
    )

    return launch.LaunchDescription(
        [
            unitree_id_arg,
            matcher_server,
        ]
    )
