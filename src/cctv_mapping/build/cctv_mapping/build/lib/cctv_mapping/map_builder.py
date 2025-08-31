#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
import numpy as np
import cv2
from collections import defaultdict
import time


class MapBuilder(Node):
    def __init__(self):
        super().__init__('map_builder')

        # 地图参数
        self.map_width = 8.0  # 世界坐标中的地图宽度(米)
        self.map_height = 8.0  # 世界坐标中的地图高度(米)
        self.resolution = 0.05  # 每个栅格的大小(米/像素)
        self.origin_x = -4.0  # 地图原点x坐标
        self.origin_y = -4.0  # 地图原点y坐标

        # 计算栅格地图尺寸
        self.grid_width = int(self.map_width / self.resolution)
        self.grid_height = int(self.map_height / self.resolution)

        # 初始化地图数据
        self.occupancy_map = np.full((self.grid_height, self.grid_width), -1, dtype=np.int8)  # -1=未知, 0=自由, 100=占用
        self.confidence_map = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)  # 置信度地图

        # 物体位置存储
        self.robot_positions = []
        self.obstacle_positions = []
        self.target_position = None

        # 物体检测历史 (用于提高可靠性)
        self.obstacle_history = defaultdict(list)  # {(grid_x, grid_y): [时间戳列表]}
        self.detection_threshold = 3  # 需要检测到3次才确认为障碍物
        self.history_timeout = 5.0  # 5秒内的检测才有效

        # 路径规划相关
        self.robot_path = []  # 机器人历史路径
        self.planned_path = []  # 规划路径

        # ROS接口
        self.setup_ros_interfaces()

        # 定时器
        self.create_timer(0.1, self.update_map)  # 10Hz更新地图

        self.get_logger().info(
            f'地图构建器启动 - 分辨率: {self.resolution}m, 尺寸: {self.grid_width}x{self.grid_height}')

    def setup_ros_interfaces(self):
        """设置ROS订阅和发布"""
        # 订阅颜色追踪器的输出
        self.robot_sub = self.create_subscription(
            PointStamped, '/robot_position', self.robot_callback, 10)
        self.obstacle_sub = self.create_subscription(
            PointStamped, '/obstacle_positions', self.obstacle_callback, 10)
        self.target_sub = self.create_subscription(
            PointStamped, '/target_position', self.target_callback, 10)

        # 发布地图
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        # 发布可视化地图 (用于调试)
        from sensor_msgs.msg import Image
        from cv_bridge import CvBridge
        from nav_msgs.msg import Path
        self.bridge = CvBridge()
        self.viz_map_pub = self.create_publisher(Image, '/map_visualization', 10)

        # 订阅A*规划的路径用于可视化
        self.planned_path_sub = self.create_subscription(
            Path, '/planned_path', self.planned_path_callback, 10)
        self.astar_path = []

    def world_to_grid(self, world_x, world_y):
        """世界坐标转换为栅格坐标"""
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        """栅格坐标转换为世界坐标"""
        world_x = grid_x * self.resolution + self.origin_x
        world_y = grid_y * self.resolution + self.origin_y
        return world_x, world_y

    def is_valid_grid(self, grid_x, grid_y):
        """检查栅格坐标是否有效"""
        return 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height

    def robot_callback(self, msg: PointStamped):
        """机器人位置回调"""
        world_x, world_y = msg.point.x, msg.point.y
        grid_x, grid_y = self.world_to_grid(world_x, world_y)

        if self.is_valid_grid(grid_x, grid_y):
            # 记录机器人位置
            self.robot_positions.append((world_x, world_y, time.time()))

            # 更新机器人路径（保留最近100个位置）
            if len(self.robot_path) > 100:
                self.robot_path.pop(0)
            self.robot_path.append((grid_x, grid_y))

            # 机器人周围区域标记为自由空间
            self.mark_free_space(grid_x, grid_y, radius=2)

    def obstacle_callback(self, msg: PointStamped):
        """障碍物位置回调"""
        world_x, world_y = msg.point.x, msg.point.y
        grid_x, grid_y = self.world_to_grid(world_x, world_y)

        if self.is_valid_grid(grid_x, grid_y):
            current_time = time.time()

            # 添加到检测历史
            self.obstacle_history[(grid_x, grid_y)].append(current_time)

            # 清理过期的检测记录
            self.obstacle_history[(grid_x, grid_y)] = [
                t for t in self.obstacle_history[(grid_x, grid_y)]
                if current_time - t < self.history_timeout
            ]

            # 如果检测次数足够，标记为障碍物
            if len(self.obstacle_history[(grid_x, grid_y)]) >= self.detection_threshold:
                self.mark_obstacle(grid_x, grid_y, radius=1)
                self.obstacle_positions.append((world_x, world_y, current_time))

    def target_callback(self, msg: PointStamped):
        """目标位置回调"""
        world_x, world_y = msg.point.x, msg.point.y
        grid_x, grid_y = self.world_to_grid(world_x, world_y)

        if self.is_valid_grid(grid_x, grid_y):
            self.target_position = (world_x, world_y)
            # 目标周围标记为自由空间
            self.mark_free_space(grid_x, grid_y, radius=1)

    def planned_path_callback(self, msg: Path):
        """A*规划路径回调"""
        self.astar_path = []
        for pose in msg.poses:
            world_x = pose.pose.position.x
            world_y = pose.pose.position.y
            grid_x, grid_y = self.world_to_grid(world_x, world_y)
            if self.is_valid_grid(grid_x, grid_y):
                self.astar_path.append((grid_x, grid_y))

    def mark_free_space(self, center_x, center_y, radius=1):
        """标记自由空间"""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                grid_x, grid_y = center_x + dx, center_y + dy
                if self.is_valid_grid(grid_x, grid_y):
                    if self.occupancy_map[grid_y, grid_x] != 100:  # 不覆盖确认的障碍物
                        self.occupancy_map[grid_y, grid_x] = 0
                        self.confidence_map[grid_y, grid_x] = min(self.confidence_map[grid_y, grid_x] + 0.1, 1.0)

    def mark_obstacle(self, center_x, center_y, radius=1):
        """标记障碍物"""
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:  # 圆形障碍物
                    grid_x, grid_y = center_x + dx, center_y + dy
                    if self.is_valid_grid(grid_x, grid_y):
                        self.occupancy_map[grid_y, grid_x] = 100
                        self.confidence_map[grid_y, grid_x] = 1.0

    def simple_path_planning(self):
        """简单的A*路径规划"""
        if not self.robot_positions or not self.target_position:
            return []

        # 获取当前机器人位置和目标位置
        robot_pos = self.robot_positions[-1]
        start_grid = self.world_to_grid(robot_pos[0], robot_pos[1])
        goal_grid = self.world_to_grid(self.target_position[0], self.target_position[1])

        if not (self.is_valid_grid(start_grid[0], start_grid[1]) and
                self.is_valid_grid(goal_grid[0], goal_grid[1])):
            return []

        # 简化的A*算法
        from heapq import heappush, heappop

        open_set = [(0, start_grid, [])]
        closed_set = set()

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while open_set:
            cost, current, path = heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            path = path + [current]

            if current == goal_grid:
                return path

            if len(path) > 200:  # 防止路径过长
                continue

            for dx, dy in directions:
                next_pos = (current[0] + dx, current[1] + dy)

                if (next_pos not in closed_set and
                        self.is_valid_grid(next_pos[0], next_pos[1]) and
                        self.occupancy_map[next_pos[1], next_pos[0]] != 100):
                    # 计算代价 (距离 + 启发式)
                    g_cost = len(path)
                    h_cost = abs(next_pos[0] - goal_grid[0]) + abs(next_pos[1] - goal_grid[1])
                    f_cost = g_cost + h_cost

                    heappush(open_set, (f_cost, next_pos, path))

        return []  # 未找到路径

    def create_visualization_map(self):
        """创建可视化地图"""
        # 创建彩色地图
        viz_map = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

        # 根据占用状态着色
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                if self.occupancy_map[y, x] == 100:  # 障碍物 - 黑色
                    viz_map[y, x] = [0, 0, 0]
                elif self.occupancy_map[y, x] == 0:  # 自由空间 - 白色
                    viz_map[y, x] = [255, 255, 255]
                else:  # 未知空间 - 灰色
                    viz_map[y, x] = [128, 128, 128]

        # 绘制机器人路径 - 蓝色
        for grid_x, grid_y in self.robot_path:
            if self.is_valid_grid(grid_x, grid_y):
                viz_map[grid_y, grid_x] = [255, 0, 0]  # BGR格式

        # 绘制A*规划路径 - 绿色粗线
        for i, (grid_x, grid_y) in enumerate(self.astar_path):
            if self.is_valid_grid(grid_x, grid_y):
                # 绘制粗一点的线
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = grid_x + dx, grid_y + dy
                        if self.is_valid_grid(nx, ny):
                            viz_map[ny, nx] = [0, 255, 0]  # 绿色

        # 绘制旧的简单规划路径 - 青色（如果存在）
        for grid_x, grid_y in self.planned_path:
            if self.is_valid_grid(grid_x, grid_y):
                viz_map[grid_y, grid_x] = [255, 255, 0]  # 青色

        # 绘制当前机器人位置 - 红色圆点
        if self.robot_positions:
            robot_pos = self.robot_positions[-1]
            grid_x, grid_y = self.world_to_grid(robot_pos[0], robot_pos[1])
            if self.is_valid_grid(grid_x, grid_y):
                cv2.circle(viz_map, (grid_x, grid_y), 3, (0, 0, 255), -1)

        # 绘制目标位置 - 黄色方块
        if self.target_position:
            grid_x, grid_y = self.world_to_grid(self.target_position[0], self.target_position[1])
            if self.is_valid_grid(grid_x, grid_y):
                cv2.rectangle(viz_map, (grid_x - 2, grid_y - 2), (grid_x + 2, grid_y + 2), (0, 255, 255), -1)

        # 放大地图以便观看
        scale_factor = 4
        viz_map_scaled = cv2.resize(viz_map,
                                    (self.grid_width * scale_factor, self.grid_height * scale_factor),
                                    interpolation=cv2.INTER_NEAREST)

        return viz_map_scaled

    def update_map(self):
        """定时更新和发布地图"""
        # 执行路径规划
        self.planned_path = self.simple_path_planning()

        # 发布占用栅格地图
        map_msg = OccupancyGrid()
        map_msg.header = Header()
        map_msg.header.stamp = self.get_clock().now().to_msg()
        map_msg.header.frame_id = 'odom'

        map_msg.info.map_load_time = self.get_clock().now().to_msg()
        map_msg.info.resolution = self.resolution
        map_msg.info.width = self.grid_width
        map_msg.info.height = self.grid_height

        map_msg.info.origin.position.x = self.origin_x
        map_msg.info.origin.position.y = self.origin_y
        map_msg.info.origin.position.z = 0.0
        map_msg.info.origin.orientation.w = 1.0

        # 转换地图数据 (注意ROS地图的行列顺序)
        map_data = []
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                map_data.append(int(self.occupancy_map[y, x]))

        map_msg.data = map_data
        self.map_pub.publish(map_msg)

        # 发布可视化地图
        viz_map = self.create_visualization_map()
        viz_msg = self.bridge.cv2_to_imgmsg(viz_map, "bgr8")
        viz_msg.header = map_msg.header
        self.viz_map_pub.publish(viz_msg)

        # 日志输出
        if len(self.robot_positions) > 0:
            robot_pos = self.robot_positions[-1]
            obstacle_count = np.sum(self.occupancy_map == 100)
            free_count = np.sum(self.occupancy_map == 0)
            path_length = len(self.planned_path)

            self.get_logger().info(
                f'机器人: ({robot_pos[0]:.1f}, {robot_pos[1]:.1f}) | '
                f'障碍物: {obstacle_count} | 自由: {free_count} | 路径长度: {path_length}'
            )


def main(args=None):
    rclpy.init(args=args)
    map_builder = MapBuilder()
    try:
        rclpy.spin(map_builder)
    except KeyboardInterrupt:
        pass
    finally:
        map_builder.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()