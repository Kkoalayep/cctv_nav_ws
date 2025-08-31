#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PointStamped, PoseWithCovarianceStamped
from std_msgs.msg import Header
import numpy as np
import heapq
import math
from typing import List, Tuple, Optional


class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')

        # 参数配置
        self.allow_diagonal = self.declare_parameter('allow_diagonal', True).get_parameter_value().bool_value
        self.inflation_cells = self.declare_parameter('inflation_cells', 1).get_parameter_value().integer_value
        self.plan_rate = self.declare_parameter('plan_rate', 2.0).get_parameter_value().double_value

        # 地图相关
        self.occupancy_map: Optional[np.ndarray] = None
        self.map_info = None
        self.inflated_map: Optional[np.ndarray] = None

        # 起点和终点
        self.start_pos: Optional[Tuple[float, float]] = None
        self.goal_pos: Optional[Tuple[float, float]] = None

        # 路径规划状态
        self.last_plan_time = 0.0
        self.current_path: Optional[List[Tuple[int, int]]] = None

        # 日志控制
        self.last_log_time = 0.0
        self.log_interval = 5.0  # 5秒输出一次规划信息

        # 运动模型（8方向或4方向）
        if self.allow_diagonal:
            self.motions = [
                (0, 1, 1.0),  # 上
                (1, 0, 1.0),  # 右
                (0, -1, 1.0),  # 下
                (-1, 0, 1.0),  # 左
                (1, 1, 1.414),  # 右上
                (1, -1, 1.414),  # 右下
                (-1, -1, 1.414),  # 左下
                (-1, 1, 1.414),  # 左上
            ]
        else:
            self.motions = [
                (0, 1, 1.0),  # 上
                (1, 0, 1.0),  # 右
                (0, -1, 1.0),  # 下
                (-1, 0, 1.0),  # 左
            ]

        self.setup_ros_interfaces()

        # 创建规划定时器
        self.plan_timer = self.create_timer(1.0 / self.plan_rate, self.plan_callback)

        # 规划阈值 - 避免频繁重规划
        self.replan_distance_threshold = 0.5  # 机器人移动超过0.5m才重规划
        self.last_start_pos = None

        self.get_logger().info(f'A*路径规划器启动 - 对角线移动: {self.allow_diagonal}, 膨胀: {self.inflation_cells}格')

    def setup_ros_interfaces(self):
        """设置ROS接口"""
        # 订阅
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.robot_sub = self.create_subscription(
            PointStamped, '/robot_position', self.robot_callback, 10)
        self.goal_sub = self.create_subscription(
            PointStamped, '/target_position', self.goal_callback, 10)

        # 可选：订阅RViz的目标点设置
        self.rviz_goal_sub = self.create_subscription(
            PoseStamped, '/goal_pose', self.rviz_goal_callback, 10)

        # 发布
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

    def map_callback(self, msg: OccupancyGrid):
        """地图更新回调"""
        self.map_info = msg.info

        # 将地图数据转换为2D数组
        width = msg.info.width
        height = msg.info.height
        self.occupancy_map = np.array(msg.data).reshape((height, width))

        # 创建膨胀后的地图
        self.inflated_map = self.inflate_obstacles(self.occupancy_map)

        # 地图更新后重新规划
        if self.start_pos is not None and self.goal_pos is not None:
            self.request_replan()

    def robot_callback(self, msg: PointStamped):
        """机器人位置更新回调"""
        self.start_pos = (msg.point.x, msg.point.y)

    def goal_callback(self, msg: PointStamped):
        """目标位置更新回调"""
        self.goal_pos = (msg.point.x, msg.point.y)
        # 只在目标改变时输出一次
        if not hasattr(self, 'last_goal_pos') or self.last_goal_pos != self.goal_pos:
            self.get_logger().info(f'🎯 新目标设定: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})')
            self.last_goal_pos = self.goal_pos
        self.request_replan()

    def rviz_goal_callback(self, msg: PoseStamped):
        """RViz目标点回调"""
        self.goal_pos = (msg.pose.position.x, msg.pose.position.y)
        # 只在目标改变时输出一次
        if not hasattr(self, 'last_goal_pos') or self.last_goal_pos != self.goal_pos:
            self.get_logger().info(f'🎯 RViz目标设定: ({self.goal_pos[0]:.2f}, {self.goal_pos[1]:.2f})')
            self.last_goal_pos = self.goal_pos
        self.request_replan()

    def request_replan(self):
        """请求重新规划"""
        self.last_plan_time = 0.0  # 强制立即规划

    def inflate_obstacles(self, occupancy_map: np.ndarray) -> np.ndarray:
        """障碍物膨胀"""
        if self.inflation_cells <= 0:
            return occupancy_map.copy()

        inflated = occupancy_map.copy()
        height, width = occupancy_map.shape

        # 找到所有障碍物位置
        obstacles = np.where(occupancy_map == 100)

        for y, x in zip(obstacles[0], obstacles[1]):
            # 在每个障碍物周围膨胀
            for dy in range(-self.inflation_cells, self.inflation_cells + 1):
                for dx in range(-self.inflation_cells, self.inflation_cells + 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width:
                        # 使用欧几里得距离判断是否在膨胀范围内
                        distance = math.sqrt(dx * dx + dy * dy)
                        if distance <= self.inflation_cells:
                            inflated[ny, nx] = 100

        return inflated

    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """世界坐标转栅格坐标"""
        if self.map_info is None:
            raise ValueError("地图信息未初始化")

        grid_x = int((world_x - self.map_info.origin.position.x) / self.map_info.resolution)
        grid_y = int((world_y - self.map_info.origin.position.y) / self.map_info.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """栅格坐标转世界坐标"""
        if self.map_info is None:
            raise ValueError("地图信息未初始化")

        world_x = grid_x * self.map_info.resolution + self.map_info.origin.position.x
        world_y = grid_y * self.map_info.resolution + self.map_info.origin.position.y
        return world_x, world_y

    def is_valid_cell(self, x: int, y: int) -> bool:
        """检查栅格坐标是否有效且可通行"""
        if self.inflated_map is None:
            return False

        height, width = self.inflated_map.shape
        if not (0 <= x < width and 0 <= y < height):
            return False

        # 检查是否为障碍物 (100) 或未知区域 (-1)
        return self.inflated_map[y, x] == 0

    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """启发式函数 - 欧几里得距离"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return math.sqrt(dx * dx + dy * dy)

    def astar_search(self, start_grid: Tuple[int, int], goal_grid: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """A*搜索算法"""
        if not self.is_valid_cell(start_grid[0], start_grid[1]):
            self.get_logger().warn(f'起点不可通行: {start_grid}')
            return None

        if not self.is_valid_cell(goal_grid[0], goal_grid[1]):
            self.get_logger().warn(f'终点不可通行: {goal_grid}')
            return None

        # 初始化
        open_set = []
        heapq.heappush(open_set, (0, start_grid))

        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}

        closed_set = set()

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal_grid:
                # 重构路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_grid)
                return path[::-1]  # 反转得到从起点到终点的路径

            closed_set.add(current)

            # 探索邻居
            for dx, dy, cost in self.motions:
                neighbor = (current[0] + dx, current[1] + dy)

                if neighbor in closed_set:
                    continue

                if not self.is_valid_cell(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = g_score[current] + cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)

                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # 未找到路径

    def smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """路径平滑 - 简单的直线段优化"""
        if len(path) <= 2:
            return path

        smoothed = [path[0]]

        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            # 从终点向起点找最远的可直达点
            while j > i + 1:
                if self.is_line_clear(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    break
                j -= 1
            else:
                # 没找到可直达的点，只能走一步
                smoothed.append(path[i + 1])
                i += 1

        return smoothed

    def is_line_clear(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """检查两点之间的直线是否无障碍"""
        x0, y0 = start
        x1, y1 = end

        # Bresenham直线算法
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        x, y = x0, y0

        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1

        error = dx - dy

        while True:
            if not self.is_valid_cell(x, y):
                return False

            if x == x1 and y == y1:
                break

            if error * 2 > -dy:
                error -= dy
                x += x_inc

            if error * 2 < dx:
                error += dx
                y += y_inc

        return True

    def create_path_message(self, grid_path: List[Tuple[int, int]]) -> Path:
        """创建路径消息"""
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'odom'

        for grid_x, grid_y in grid_path:
            pose = PoseStamped()
            pose.header = path_msg.header

            world_x, world_y = self.grid_to_world(grid_x, grid_y)
            pose.pose.position.x = world_x
            pose.pose.position.y = world_y
            pose.pose.position.z = 0.0

            # 简单的朝向计算
            pose.pose.orientation.w = 1.0

            path_msg.poses.append(pose)

        return path_msg

    def plan_callback(self):
        """定时规划回调"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        # 检查是否需要规划
        if current_time - self.last_plan_time < 1.0 / self.plan_rate:
            return

        # 检查规划条件
        if (self.inflated_map is None or
                self.start_pos is None or
                self.goal_pos is None or
                self.map_info is None):
            return

        try:
            # 转换为栅格坐标
            start_grid = self.world_to_grid(self.start_pos[0], self.start_pos[1])
            goal_grid = self.world_to_grid(self.goal_pos[0], self.goal_pos[1])

            # 执行A*搜索
            raw_path = self.astar_search(start_grid, goal_grid)

            if raw_path is None:
                self.get_logger().warn('未找到可行路径')
                # 发布空路径
                empty_path = Path()
                empty_path.header.frame_id = 'odom'
                empty_path.header.stamp = self.get_clock().now().to_msg()
                self.path_pub.publish(empty_path)
                return

            # 路径平滑
            smooth_path = self.smooth_path(raw_path)
            self.current_path = smooth_path

            # 创建并发布路径消息
            path_msg = self.create_path_message(smooth_path)
            self.path_pub.publish(path_msg)

            # 计算路径长度
            path_length = len(smooth_path)
            world_distance = 0.0
            for i in range(len(smooth_path) - 1):
                x1, y1 = self.grid_to_world(smooth_path[i][0], smooth_path[i][1])
                x2, y2 = self.grid_to_world(smooth_path[i + 1][0], smooth_path[i + 1][1])
                world_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # 控制日志输出频率
            current_log_time = self.get_clock().now().nanoseconds / 1e9
            if current_log_time - self.last_log_time >= self.log_interval:
                self.get_logger().info(
                    f'🛤️ 路径规划: {path_length}节点, 长度{world_distance:.2f}m'
                )
                self.last_log_time = current_log_time

            self.last_plan_time = current_time

        except Exception as e:
            self.get_logger().error(f'路径规划失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    planner = AStarPlanner()
    try:
        rclpy.spin(planner)
    except KeyboardInterrupt:
        pass
    finally:
        planner.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()