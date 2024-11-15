import launch
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

    param_config = LaunchConfiguration("parameter_config")
    data_config = LaunchConfiguration("data_config")

    param_config_arg = DeclareLaunchArgument("parameter_config", default_value="")
    data_config_arg = DeclareLaunchArgument("data_config", default_value="")

    bandwidth_estimator_node = Node(
        package="map_matcher",
        executable="map_matcher_bandwidth",
        name="map_matcher_bandwidth",
        output="screen",
        emulate_tty=True,
        parameters=[{"parameter_config": param_config, "data_config": data_config}],
    )

    return launch.LaunchDescription(
        [
            bandwidth_estimator_node,
            param_config_arg,
            data_config_arg,
        ]
    )
