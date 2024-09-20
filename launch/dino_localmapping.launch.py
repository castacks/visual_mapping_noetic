from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare the argument for the config file
    visual_mapping_config = LaunchConfiguration('visual_mapping_config')

    # Use PathJoinSubstitution to dynamically join the package path and the config file
    config_fp = PathJoinSubstitution([
        FindPackageShare('physics_atv_visual_mapping'),
        'config',
        'ros',
        visual_mapping_config
    ])

    return LaunchDescription([
        # Declare the launch argument with a default value
        DeclareLaunchArgument(
            'visual_mapping_config',
            default_value='wheelchair.yaml',
            description='Config file for visual mapping'
        ),

        # Node definition
        Node(
            package='physics_atv_visual_mapping',
            executable='dino_localmapping',
            name='visual_localmapping',
            output='screen',
            parameters=[{'config_fp': config_fp}]
        ),

        # Uncomment this node if needed in the future
        # Node(
        #     package='physics_atv_visual_mapping',
        #     executable='dino_cost',
        #     name='dino_cost'
        # ),
    ])
