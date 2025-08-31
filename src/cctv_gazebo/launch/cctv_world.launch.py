import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node



def generate_launch_description():
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_cctv_gazebo = get_package_share_directory('cctv_gazebo')

    world_file = os.path.join(pkg_cctv_gazebo, 'worlds', 'cctv_world.world')

    # 启动Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([pkg_gazebo_ros, '/launch/gazebo.launch.py']),
        launch_arguments={'world': world_file}.items(),
    )

    # 相机TF配置 - 使用你验证过的正确参数
    static_cam_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '5.0', '3.1415926', '3.1415926', '0', 'odom', 'overhead_camera_optical_frame'],
        output='screen',
        name='camera_tf_publisher'
    )
    # 在 cctv_world.launch.py 中，在 static_cam_tf 后面添加：
    static_map_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        output='screen',
        name='map_odom_tf_publisher'
    )

    # 颜色跟踪器 - 带完整参数配置
    color_tracker = Node(
        package='cctv_perception',
        executable='color_tracker',
        parameters=[{
            'world_frame': 'odom',
            'enable_perspective_correction': True,  # 启用透视校正
            'correction_method': 'advanced',  # 'simple' 或 'advanced'
        }],
        output='screen',
        name='color_tracker'
    )

    # 替换原来的 map_builder 节点
    map_builder = Node(
        package='cctv_mapping',
        executable='map_builder',  # 注意这里改名了
        parameters=[{
            'resolution': 0.05,
            'world_size': 8.0,
            'origin_x': -4.0,
            'origin_y': -4.0,
            'publish_period': 0.5,
            'detection_threshold': 1,
            'obstacle_radius_cells': 2,
            'robot_free_radius_cells': 4,
            'target_free_radius_cells': 2,
            'inflate_plan_m': 0.30,
            'inflate_nav_m': 0.00,
        }],
        output='screen'
    )

    navigation = Node(
        package='cctv_navigation',
        executable='astar_debug_navigation',
        parameters=[{
            # 速度参数 - 提高速度
            'max_linear_speed': 0.25,  # 提高到0.25 m/s (原来0.16)
            'max_angular_speed': 0.8,  # 降低角速度到0.8 (原来1.0)，减少摆动
            'min_crawl_speed': 0.08,  # 提高最小速度 (原来0.05)

            # 路径跟随参数 - 减少摆动
            'lookahead_distance': 0.6,  # 增加前瞻距离 (原来0.45)，更平滑
            'goal_tolerance': 0.25,  # 目标容忍度

            # 角度控制参数 - 关键改进
            'angle_lpf_alpha': 0.15,  # 降低角度滤波系数 (原来0.2)，更平滑
            'target_debounce': 0.05,  # 减少目标抖动 (原来0.10)

            # 重规划参数
            'replan_interval': 8.0,  # 增加重规划间隔 (原来6.0)
            'map_replan_cooldown': 1.5,  # 减少地图重规划冷却时间
            'obstacle_threshold': 50,

            # 线段检查参数 - 稍微放宽，减少频繁避障
            'segment_step': 0.03,  # 略增加步长 (原来0.02)
            'line_check_margin': 0.10,  # 减少边距 (原来0.12)
            'line_check_min_hits': 2,  # 减少连续命中要求 (原来3)
            'line_check_max_len': 0.7,  # 增加检查长度 (原来0.5)
        }],
        output='screen',
        name='navigation_controller'
    )

    # RViz2可视化 (可选)
    rviz_config_file = os.path.join(pkg_cctv_gazebo, 'config', 'cctv_navigation.rviz')
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file] if os.path.exists(rviz_config_file) else [],
        output='screen',
        name='rviz2'
    )

    return LaunchDescription([
        gazebo,
        static_cam_tf,
        static_map_tf,
        color_tracker,
        map_builder,
        navigation,
        # rviz,  # 取消注释以启用RViz
    ])