#!/usr/bin/env python3

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


class AStarPlanner:
    """完整的A*路径规划器"""
    
    def __init__(self, map_data, map_info):
        self.map_data = map_data
        self.map_info = map_info
        self.width = map_info.width
        self.height = map_info.height
        self.resolution = map_info.resolution
        self.origin_x = map_info.origin.position.x
        self.origin_y = map_info.origin.position.y
        
        # 创建地图数组
        self.grid_map = np.array(map_data).reshape(self.height, self.width)
        
        # A*参数
        self.obstacle_threshold = 50
    
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
        return self.grid_map[y, x] < self.obstacle_threshold
    
    def heuristic(self, a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def get_neighbors(self, node):
        x, y = node
        neighbors = []
        
        # 8方向移动
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
    
    def smooth_path(self, path):
        """路径平滑化"""
        if len(path) < 3:
            return path
        
        smoothed = [path[0]]
        i = 0
        
        while i < len(path) - 1:
            j = len(path) - 1
            
            # 找到从当前点能直接到达的最远点
            while j > i + 1:
                if self._line_of_sight(path[i], path[j]):
                    break
                j -= 1
            
            smoothed.append(path[j])
            i = j
        
        return smoothed
    
    def _line_of_sight(self, start, end):
        """检查两点间是否有直线视线"""
        x0, y0 = start
        x1, y1 = end
        
        # Bresenham直线算法
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while True:
            if not self.is_valid(x0, y0):
                return False
            
            if x0 == x1 and y0 == y1:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return True
    
    def plan(self, start_world, goal_world):
        """A*路径规划主函数"""
        start_grid = self.world_to_grid(start_world[0], start_world[1])
        goal_grid = self.world_to_grid(goal_world[0], goal_world[1])
        
        # 检查起点和终点
        if not self.is_valid(start_grid[0], start_grid[1]):
            return []
        if not self.is_valid(goal_grid[0], goal_grid[1]):
            return []
        
        # A*算法
        open_set = []
        heappush(open_set, (0, start_grid))
        
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}
        
        closed_set = set()
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal_grid:
                # 重建路径
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_grid)
                path.reverse()
                
                # 转换为世界坐标并平滑化
                world_path = [self.grid_to_world(x, y) for x, y in path]
                grid_path_smoothed = self.smooth_path(path)
                world_path_smoothed = [self.grid_to_world(x, y) for x, y in grid_path_smoothed]
                
                return world_path_smoothed
            
            closed_set.add(current)
            
            for neighbor, move_cost in self.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal_grid)
                    
                    heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # 未找到路径


class NavigationController(Node):
    def __init__(self):
        super().__init__('astar_navigation_controller')
        
        # 导航参数
        self.max_linear_speed = 0.12
        self.max_angular_speed = 0.6
        self.goal_tolerance = 0.15
        self.path_tolerance = 0.1
        
        # Pure Pursuit参数
        self.lookahead_distance = 0.3
        self.min_lookahead = 0.2
        self.max_lookahead = 0.5
        
        # 状态变量
        self.current_state = NavigationState.IDLE
        self.robot_position = None
        self.robot_yaw = 0.0
        self.last_robot_position = None
        self.target_position = None
        self.current_map = None
        self.planner = None
        
        # 路径相关
        self.current_path = []
        self.path_index = 0
        self.last_plan_time = 0
        self.replan_interval = 5.0
        
        # 卡住检测
        self.stuck_counter = 0
        self.max_stuck_count = 100
        self.stuck_threshold = 0.02
        
        # ROS接口
        self.setup_ros_interfaces()
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('🌟 A*导航控制器启动')

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
        new_position = (msg.point.x, msg.point.y)
        
        # 估算机器人朝向
        if self.last_robot_position:
            dx = new_position[0] - self.last_robot_position[0]
            dy = new_position[1] - self.last_robot_position[1]
            movement = math.sqrt(dx*dx + dy*dy)
            
            if movement > 0.01:
                self.robot_yaw = math.atan2(dy, dx)
        
        self.last_robot_position = self.robot_position
        self.robot_position = new_position

    def target_position_callback(self, msg):
        new_target = (msg.point.x, msg.point.y)
        if self.target_position != new_target:
            self.target_position = new_target
            self.current_state = NavigationState.PLANNING
            self.stuck_counter = 0
            self.get_logger().info(f'🎯 新目标: ({new_target[0]:.2f}, {new_target[1]:.2f})')

    def map_callback(self, msg):
        self.current_map = msg
        if msg.data:
            self.planner = AStarPlanner(msg.data, msg.info)

    def plan_path(self):
        """使用A*算法规划路径"""
        if not self.robot_position or not self.target_position or not self.planner:
            return False
        
        start_time = time.time()
        path = self.planner.plan(self.robot_position, self.target_position)
        plan_time = time.time() - start_time
        
        if path:
            self.current_path = path
            self.path_index = 0
            self.last_plan_time = time.time()
            
            # 发布路径用于可视化
            self.publish_path(path)
            
            self.get_logger().info(
                f'✅ A*路径规划成功! 路径长度: {len(path)}, 规划时间: {plan_time:.3f}s')
            return True
        else:
            self.get_logger().warn('❌ A*路径规划失败!')
            return False

    def publish_path(self, path):
        """发布路径用于RViz可视化"""
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

    def get_lookahead_point(self):
        """Pure Pursuit: 获取前瞻点"""
        if not self.current_path or not self.robot_position:
            return None
        
        robot_x, robot_y = self.robot_position
        
        # 找到最近的路径点
        min_dist = float('inf')
        closest_index = self.path_index
        
        for i in range(self.path_index, len(self.current_path)):
            path_x, path_y = self.current_path[i]
            dist = math.sqrt((robot_x - path_x)**2 + (robot_y - path_y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        
        self.path_index = closest_index
        
        # 找前瞻点
        for i in range(closest_index, len(self.current_path)):
            path_x, path_y = self.current_path[i]
            dist = math.sqrt((robot_x - path_x)**2 + (robot_y - path_y)**2)
            
            if dist >= self.lookahead_distance:
                return (path_x, path_y)
        
        # 返回路径终点
        return self.current_path[-1] if self.current_path else None

    def pure_pursuit_control(self, lookahead_point):
        """Pure Pursuit控制算法"""
        if not lookahead_point or not self.robot_position:
            return 0.0, 0.0
        
        robot_x, robot_y = self.robot_position
        target_x, target_y = lookahead_point
        
        # 计算目标角度
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        
        # 角度误差
        angle_error = target_angle - self.robot_yaw
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi
        
        # 控制律
        linear_velocity = min(self.max_linear_speed, distance * 0.8)
        angular_velocity = max(-self.max_angular_speed, 
                              min(self.max_angular_speed, angle_error * 2.0))
        
        # 角度误差大时减速
        if abs(angle_error) > 0.5:
            linear_velocity *= 0.5
        
        return linear_velocity, angular_velocity

    def check_if_stuck(self):
        """检查是否卡住"""
        if not self.robot_position or not self.last_robot_position:
            return False
        
        dx = self.robot_position[0] - self.last_robot_position[0]
        dy = self.robot_position[1] - self.last_robot_position[1]
        movement = math.sqrt(dx*dx + dy*dy)
        
        if movement < self.stuck_threshold:
            self.stuck_counter += 1
        else:
            self.stuck_counter = max(0, self.stuck_counter - 2)
        
        return self.stuck_counter > self.max_stuck_count

    def control_loop(self):
        """主控制循环"""
        cmd = Twist()
        
        if self.current_state == NavigationState.IDLE:
            pass
            
        elif self.current_state == NavigationState.PLANNING:
            if self.plan_path():
                self.current_state = NavigationState.NAVIGATING
            else:
                self.current_state = NavigationState.NO_PATH
                
        elif self.current_state == NavigationState.NAVIGATING:
            if not self.robot_position or not self.target_position:
                pass
            else:
                # 检查是否到达目标
                robot_x, robot_y = self.robot_position
                target_x, target_y = self.target_position
                distance_to_goal = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
                
                if distance_to_goal < self.goal_tolerance:
                    self.current_state = NavigationState.REACHED
                    self.get_logger().info('🎯 成功到达目标！')
                else:
                    # 检查是否需要重新规划
                    current_time = time.time()
                    if current_time - self.last_plan_time > self.replan_interval:
                        self.get_logger().info('🔄 定期重新规划路径...')
                        self.current_state = NavigationState.PLANNING
                    
                    # Pure Pursuit控制
                    lookahead_point = self.get_lookahead_point()
                    if lookahead_point:
                        linear_vel, angular_vel = self.pure_pursuit_control(lookahead_point)
                        cmd.linear.x = linear_vel
                        cmd.angular.z = angular_vel
                        
                        # 检查是否卡住
                        if self.check_if_stuck():
                            self.current_state = NavigationState.STUCK
                            self.get_logger().warn('🚫 机器人卡住！')
                            
        elif self.current_state == NavigationState.REACHED:
            pass
            
        elif self.current_state == NavigationState.STUCK:
            # 简单的脱困：后退并重新规划
            cmd.linear.x = -0.05
            cmd.angular.z = 0.2
            if self.stuck_counter % 50 == 0:  # 每5秒重试
                self.current_state = NavigationState.PLANNING
                self.stuck_counter = 0
                
        elif self.current_state == NavigationState.NO_PATH:
            pass
        
        # 发布控制命令
        self.cmd_vel_pub.publish(cmd)
        
        # 状态日志
        if self.robot_position and self.target_position and time.time() % 2 < 0.1:
            robot_x, robot_y = self.robot_position
            target_x, target_y = self.target_position
            distance = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
            
            status_emoji = {
                NavigationState.IDLE: "⏸️",
                NavigationState.PLANNING: "🧭",
                NavigationState.NAVIGATING: "🚀",
                NavigationState.REACHED: "🎯",
                NavigationState.STUCK: "🚫",
                NavigationState.NO_PATH: "❌"
            }
            
            self.get_logger().info(
                f'{status_emoji.get(self.current_state, "🤖")} 状态: {self.current_state.name} | '
                f'距离目标: {distance:.2f}m | 路径点: {self.path_index}/{len(self.current_path)}'
            )


def main():
    rclpy.init()
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
