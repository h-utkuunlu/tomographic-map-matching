import launch

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    unitree_id = LaunchConfiguration("unitree_id")

    unitree_id_launch_arg = DeclareLaunchArgument("unitree_id", default_value="B1_154")

    container = ComposableNodeContainer(
        package="rclcpp_components",
        executable="component_container",
        name="matcher_server_container",
        namespace=unitree_id,
        composable_node_descriptions=[
            ComposableNode(
                package="map_matcher",
                plugin="map_matcher_ros::MatcherServer",
                name="matcher_server",
                namespace=unitree_id,
            )
        ],
        output="screen",
        emulate_tty=True,
    )

    return launch.LaunchDescription([unitree_id_launch_arg, container])
