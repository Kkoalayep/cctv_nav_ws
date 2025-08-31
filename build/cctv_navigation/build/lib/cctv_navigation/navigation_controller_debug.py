#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import math
from enum import Enum
from heapq import heappush, heappop
import time


class NavigationState(Enum):
    IDLE = 0
    PLANNING = 1
    NAVIGATING = 2
    REACHED = 3
    STUCK = 4
    NO_PATH = 5


class DebugAStarPlanner:
    """A*路径规划器 - 带调试信息"""
    
    def __init__(self, map_data, map_info, logger):
        self.map_data = map_data
        self.map_info = map_info
        self.width = map_info.width
        self.height = map_info.height
        self.resolution = map_info.resolution
        self.origin_x = map_info.origin.position.x
        self.origin_y = map_info.origin.position.y
        self.logger = logger
        
        # A*参数
        self.obstacle_threshold = 50
        
        # 处理地图数据
        if len(map_data) != self.width * self.height:
            self.logger.error(f"地图数据长度不匹配！期望: {self.width * self.height}, 实际: {len(map_data)}")
            self.grid_map = np.full((self.height, self.width), -1, dtype=np.int8)
        else:
            self.grid_map = np.array(map_data).reshape(self.height, self.width)
        
        self.logger.info(f"地图信息: {self.width}x{self.height}, 分辨率: {self.resolution}")
        self.logger.info(f"原点: ({self.origin_x}, {self.origin_y})")
        
        # 统计地图内容
        unknown_count = np.sum(self.grid_map == -1)
        free_count = np.sum(self.grid_map == 0)
        obstacle_count = np.sum(self.grid_map > self.obstacle_threshold)
        self.logger.info(f"地图统计: 未知={unknown_count}, 自由={free_count}, 障碍物={obstacle_count}")
    
    def world_to_grid(self, world_x, world_y):
        grid_x = int((world_x - self.origin_x) / self.resolution)
        grid_y = int((world_y - self.origin_y) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        world_x = grid_x * self.resolution + self.origin_x + self.resolution / 2
        world_y = grid_y * self.resolution + self.origin_y + self.resolution / 2
        return world_x, world_y
    
    def is_valid(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        return self.grid_map[y, x] != -1 and self.grid_map[y, x] < self.obstacle_threshold
    
    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        
        # 8个方向
        directions = [
            (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
            (0, -1, 1.0),                   (0, 1, 1.0),
            (1, -1, 1.414),  (1, 0, 1.0),  (1, 1, 1.414)
        ]
        
        for dx, dy, cost in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append(((nx, ny), cost))
        
        return neighbors
    
    def plan(self, start_world, goal_world):
        start_grid = self.world_to_grid(start_world[0], start_world[1])
        goal_grid = self.world_to_grid(goal_world[0], goal_world[1])
        
        self.logger.info(f"规划路径: 起点世界坐标{start_world} -> 栅格{start_grid}")
        self.logger.info(f"规划路径: 终点世界坐标{goal_world} -> 栅格{goal_grid}")
        
        # 检查起点和终点
        if not self.is_valid(start_grid[0], start_grid[1]):
            start_value = self.grid_map[start_grid[1], start_grid[0]] if 0 <= start_grid[0] < self.width and 0 <= start_grid[1] < self.height else "越界"
            self.logger.error(f"起点无效: 栅格{start_grid}, 值={start_value}")
            return []
        
        if not self.is_valid(goal_grid[0], goal_grid[1]):
            goal_value = self.grid_map[goal_grid[1], goal_grid[0]] if 0 <= goal_grid[0] < self.width and 0 <= goal_grid[1] < self.height else "越界"
            self.logger.error(f"终点无效: 栅格{goal_grid}, 值={goal_value}")
            return []
        
        self.logger.info("起点和终点都有效，开始A*搜索...")
        
        # A*算法
        open_set = []
        heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        closed_set = set()
        iterations = 0
        max_iterations = 10000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current = heappop(open_set)[1]
            
            if current == goal_grid:
                self.logger.info(f"找到路径！A*迭代次数: {iterations}")
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_grid)
                path.reverse()
                
                # 转换为世界坐标
                world_path = [self.grid_to_world(x, y) for x, y in path]
                self.logger.info(f"路径长度: {len(world_path)} 点")
                return world_path
            
            closed_set.add(current)
            
            neighbors = self.get_neighbors(current)
            if iterations == 1:
                self.logger.info(f"起点邻居数量: {len(neighbors)}")
            
            for neighbor, move_cost in neighbors:
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                    
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        self.logger.error(f"A*搜索失败！迭代次数: {iterations}, 开放集大小: {len(open_set)}")
        return []


class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')
        
        # 导航参数
        self.max_linear_speed = 0.15
        self.max_angular_speed = 0.8
        self.goal_tolerance = 0.15
        
        # 状态变量
        self.current_state = NavigationState.IDLE
        self.robot_position = None
        self.target_position = None
        self.current_map = None
        self.planner = None
        
        # 路径相关
        self.current_path = []
        self.path_index = 0
        
        # ROS接口
        self.setup_ros_interfaces()
        
        # 控制循环定时器
        self.create_timer(0.5, self.control_loop)
        
        self.get_logger().info('A*导航控制器启动 - 调试版')

    def setup_ros_interfaces(self):
        # 订阅
        self.robot_sub = self.create_subscription(
            PointStamped, '/robot_position', self.robot_position_callback, 10)
        self.target_sub = self.create_subscription(
            PointStamped, '/target_position', self.target_position_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        
        # 发布
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/planned_path', 10)

    def robot_position_callback(self, msg):
        self.robot_position = (msg.point.x, msg.point.y)

    def target_position_callback(self, msg):
        new_target = (msg.point.x, msg.point.y)
        if self.target_position != new_target:
            self.target_position = new_target
            self.current_state = NavigationState.PLANNING
            self.get_logger().info(f'新目标: ({new_target[0]:.2f}, {new_target[1]:.2f})')

    def map_callback(self, msg):
        self.current_map = msg
        if msg.data and len(msg.data) > 0:
            self.planner = DebugAStarPlanner(msg.data, msg.info, self.get_logger())
        else:
            self.get_logger().warn("接收到空地图数据")

    def plan_path(self):
        if not self.robot_position or not self.target_position or not self.planner:
            self.get_logger().warn(f"规划条件不满足: 机器人={self.robot_position is not None}, 目标={self.target_position is not None}, 规划器={self.planner is not None}")
            return False
        
        self.get_logger().info("开始路径规划...")
        start_time = time.time()
        path = self.planner.plan(self.robot_position, self.target_position)
        plan_time = time.time() - start_time
        
        if path:
            self.current_path = path
            self.path_index = 0
            self.publish_path(path)
            self.get_logger().info(f'路径规划成功! 路径长度: {len(path)}, 规划时间: {plan_time:.3f}s')
            return True
        else:
            self.get_logger().error('A*路径规划失败 - 未找到可行路径')
            return False

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = 'odom'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        
        for point in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        
        self.path_pub.publish(path_msg)

    def control_loop(self):
        cmd = Twist()
        
        if self.current_state == NavigationState.IDLE:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        elif self.current_state == NavigationState.PLANNING:
            if self.plan_path():
                self.current_state = NavigationState.NAVIGATING
            else:
                self.current_state = NavigationState.NO_PATH
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        elif self.current_state == NavigationState.NAVIGATING:
            if not self.robot_position or not self.target_position:
                cmd.linear.x = 0.0
                cmd.angular.z = 0.0
            else:
                # 简单的点到点导航
                robot_x, robot_y = self.robot_position
                target_x, target_y = self.target_position
                distance_to_goal = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
                
                if distance_to_goal < self.goal_tolerance:
                    self.current_state = NavigationState.REACHED
                    cmd.linear.x = 0.0
                    cmd.angular.z = 0.0
                    self.get_logger().info('成功到达目标！')
                else:
                    # 简单比例控制
                    dx = target_x - robot_x
                    dy = target_y - robot_y
                    target_angle = math.atan2(dy, dx)
                    
                    cmd.linear.x = min(self.max_linear_speed, distance_to_goal * 0.5)
                    cmd.angular.z = max(-self.max_angular_speed, 
                                       min(self.max_angular_speed, target_angle * 2.0))
                    
        elif self.current_state == NavigationState.REACHED:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            
        elif self.current_state == NavigationState.NO_PATH:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        
        # 发布控制命令
        self.cmd_vel_pub.publish(cmd)
        
        # 状态日志
        if self.robot_position and self.target_position:
            robot_x, robot_y = self.robot_position
            target_x, target_y = self.target_position
            distance = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
            
            self.get_logger().info(
                f'状态: {self.current_state.name} | 距离目标: {distance:.2f}m | 路径点: {len(self.current_path)}'
            )


def main(args=None):
    rclpy.init(args=args)
    controller = NavigationController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
