#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Path
import numpy as np
import math
from typing import List, Tuple, Optional


class NavigationController(Node):
    def __init__(self):
        super().__init__('navigation_controller')

        # 控制参数
        self.max_linear_speed = self.declare_parameter('max_linear_speed', 0.2).get_parameter_value().double_value
        self.max_angular_speed = self.declare_parameter('max_angular_speed', 0.5).get_parameter_value().double_value
        self.goal_tolerance = self.declare_parameter('goal_tolerance', 0.1).get_parameter_value().double_value
        self.obstacle_avoidance_distance = self.declare_parameter('obstacle_avoidance_distance',
                                                                  0.3).get_parameter_value().double_value

        # PID参数
        self.kp_linear = self.declare_parameter('kp_linear', 1.0).get_parameter_value().double_value
        self.kp_angular = self.declare_parameter('kp_angular', 2.0).get_parameter_value().double_value
        self.ki_linear = self.declare_parameter('ki_linear', 0.1).get_parameter_value().double_value
        self.ki_angular = self.declare_parameter('ki_angular', 0.1).get_parameter_value().double_value
        self.kd_linear = self.declare_parameter('kd_linear', 0.05).get_parameter_value().double_value
        self.kd_angular = self.declare_parameter('kd_angular', 0.05).get_parameter_value().double_value

        # 路径跟踪参数
        self.lookahead_distance = 0.3  # 前瞻距离
        self.min_lookahead_distance = 0.15
        self.max_lookahead_distance = 0.6

        # 状态变量
        self.current_pos: Optional[Tuple[float, float]] = None
        self.current_yaw: float = 0.0
        self.planned_path: List[Tuple[float, float]] = []
        self.current_goal_index = 0
        self.obstacle_positions: List[Tuple[float, float]] = []

        # PID积分和微分项
        self.linear_integral = 0.0
        self.angular_integral = 0.0
        self.last_linear_error = 0.0
        self.last_angular_error = 0.0
        self.last_time = self.get_clock().now()

        # 控制状态
        self.is_navigating = False
        self.goal_reached = False

        # 日志控制
        self.last_nav_log_time = 0.0
        self.nav_log_interval = 3.0  # 3秒输出一次导航状态

        self.setup_ros_interfaces()

        # 控制循环
        self.control_timer = self.create_timer(0.1, self.control_callback)  # 10Hz控制频率

        self.get_logger().info(
            f'导航控制器启动 - 最大线速度: {self.max_linear_speed}m/s, 最大角速度: {self.max_angular_speed}rad/s')

    def setup_ros_interfaces(self):
        """设置ROS接口"""
        # 订阅
        self.robot_sub = self.create_subscription(
            PointStamped, '/robot_position', self.robot_callback, 10)
        self.path_sub = self.create_subscription(
            Path, '/planned_path', self.path_callback, 10)
        self.obstacle_sub = self.create_subscription(
            PointStamped, '/obstacle_positions', self.obstacle_callback, 10)

        # 发布
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def robot_callback(self, msg: PointStamped):
        """机器人位置更新"""
        self.current_pos = (msg.point.x, msg.point.y)

        # 简单的朝向估计（基于运动方向）
        # 在实际应用中，您可能需要IMU或里程计来获取精确朝向
        if hasattr(self, 'last_pos') and self.last_pos is not None:
            dx = self.current_pos[0] - self.last_pos[0]
            dy = self.current_pos[1] - self.last_pos[1]
            if abs(dx) > 0.01 or abs(dy) > 0.01:  # 只有在有明显移动时才更新朝向
                self.current_yaw = math.atan2(dy, dx)

        self.last_pos = self.current_pos

    def path_callback(self, msg: Path):
        """路径更新回调"""
        if not msg.poses:
            self.planned_path = []
            self.is_navigating = False
            self.get_logger().info('❌ 收到空路径，停止导航')
            return

        # 检查路径是否真的改变了，避免频繁重启导航
        new_path = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            new_path.append((x, y))

        # 如果路径没有显著变化，就不重启导航
        if (self.planned_path and len(new_path) == len(self.planned_path) and
                self.is_navigating and not self.goal_reached):
            path_similar = True
            for i, (new_point, old_point) in enumerate(zip(new_path, self.planned_path)):
                if abs(new_point[0] - old_point[0]) > 0.1 or abs(new_point[1] - old_point[1]) > 0.1:
                    path_similar = False
                    break

            if path_similar:
                return  # 路径没有显著变化，继续当前导航

        # 路径有显著变化，更新导航
        self.planned_path = new_path
        self.current_goal_index = 0
        self.is_navigating = True
        self.goal_reached = False

        # 重置PID
        self.linear_integral = 0.0
        self.angular_integral = 0.0
        self.last_linear_error = 0.0
        self.last_angular_error = 0.0

        # 只在路径真正改变时输出日志
        self.get_logger().info(f'🚀 新路径导航: {len(self.planned_path)}个路径点')

    def obstacle_callback(self, msg: PointStamped):
        """障碍物位置更新"""
        obstacle_pos = (msg.point.x, msg.point.y)

        # 简单的障碍物管理：保持最近的几个障碍物
        self.obstacle_positions.append(obstacle_pos)
        if len(self.obstacle_positions) > 10:  # 只保留最近10个障碍物
            self.obstacle_positions.pop(0)

    def distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """计算两点距离"""
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def find_lookahead_point(self) -> Optional[Tuple[float, float]]:
        """寻找前瞻点"""
        if not self.planned_path or self.current_pos is None:
            return None

        # 动态调整前瞻距离（根据速度）
        # 这里简化为固定值，实际可以根据当前速度调整
        lookahead_dist = self.lookahead_distance

        # 从当前目标点开始搜索
        for i in range(self.current_goal_index, len(self.planned_path)):
            point = self.planned_path[i]
            dist = self.distance(self.current_pos, point)

            if dist >= lookahead_dist:
                # 找到合适的前瞻点
                if i > self.current_goal_index:
                    self.current_goal_index = i - 1
                return point

        # 如果没找到合适的前瞻点，返回路径终点
        return self.planned_path[-1]

    def check_obstacle_collision(self, target_point: Tuple[float, float]) -> bool:
        """检查是否会与障碍物碰撞"""
        if self.current_pos is None:
            return False

        for obs_pos in self.obstacle_positions:
            # 检查目标点是否太接近障碍物
            if self.distance(target_point, obs_pos) < self.obstacle_avoidance_distance:
                return True

            # 检查路径是否穿过障碍物
            if self.line_obstacle_intersection(self.current_pos, target_point, obs_pos):
                return True

        return False

    def line_obstacle_intersection(self, start: Tuple[float, float],
                                   end: Tuple[float, float],
                                   obstacle: Tuple[float, float]) -> bool:
        """检查线段是否与障碍物相交"""
        # 计算点到线段的最短距离
        A = np.array(start)
        B = np.array(end)
        P = np.array(obstacle)

        AB = B - A
        AP = P - A

        if np.dot(AB, AB) == 0:  # 起点终点相同
            return np.linalg.norm(AP) < self.obstacle_avoidance_distance

        t = np.dot(AP, AB) / np.dot(AB, AB)
        t = max(0, min(1, t))  # 限制在线段上

        closest_point = A + t * AB
        distance = np.linalg.norm(P - closest_point)

        return distance < self.obstacle_avoidance_distance

    def avoid_obstacles(self, target_point: Tuple[float, float]) -> Tuple[float, float]:
        """简单的障碍物规避"""
        if not self.check_obstacle_collision(target_point):
            return target_point

        # 简单的规避策略：尝试左右偏移
        avoidance_distance = 0.5
        current_to_target = math.atan2(
            target_point[1] - self.current_pos[1],
            target_point[0] - self.current_pos[0]
        )

        # 尝试左右两个方向
        for angle_offset in [math.pi / 2, -math.pi / 2]:
            avoid_angle = current_to_target + angle_offset
            avoid_x = self.current_pos[0] + avoidance_distance * math.cos(avoid_angle)
            avoid_y = self.current_pos[1] + avoidance_distance * math.sin(avoid_angle)
            avoid_point = (avoid_x, avoid_y)

            if not self.check_obstacle_collision(avoid_point):
                # 只在首次规避时输出日志
                if not hasattr(self, 'is_avoiding') or not self.is_avoiding:
                    self.get_logger().info('⚠️ 执行障碍物规避')
                    self.is_avoiding = True
                return avoid_point

        # 重置规避状态
        self.is_avoiding = False
        # 如果都不行，就停止
        self.get_logger().warn('⛔ 无法找到安全的规避路径')
        return self.current_pos

    def pure_pursuit_control(self, target_point: Tuple[float, float]) -> Tuple[float, float]:
        """Pure Pursuit路径跟踪控制"""
        if self.current_pos is None:
            return 0.0, 0.0

        # 计算到目标点的距离和角度
        dx = target_point[0] - self.current_pos[0]
        dy = target_point[1] - self.current_pos[1]
        distance = math.sqrt(dx * dx + dy * dy)
        target_angle = math.atan2(dy, dx)

        # 角度误差
        angle_error = target_angle - self.current_yaw

        # 归一化角度到[-pi, pi]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi

        return distance, angle_error

    def pid_control(self, linear_error: float, angular_error: float, dt: float) -> Tuple[float, float]:
        """PID控制器"""
        # 线速度PID
        self.linear_integral += linear_error * dt
        linear_derivative = (linear_error - self.last_linear_error) / dt if dt > 0 else 0
        linear_output = (self.kp_linear * linear_error +
                         self.ki_linear * self.linear_integral +
                         self.kd_linear * linear_derivative)

        # 角速度PID
        self.angular_integral += angular_error * dt
        angular_derivative = (angular_error - self.last_angular_error) / dt if dt > 0 else 0
        angular_output = (self.kp_angular * angular_error +
                          self.ki_angular * self.angular_integral +
                          self.kd_angular * angular_derivative)

        # 更新上次误差
        self.last_linear_error = linear_error
        self.last_angular_error = angular_error

        return linear_output, angular_output

    def control_callback(self):
        """主控制循环"""
        if not self.is_navigating or self.goal_reached or not self.planned_path or self.current_pos is None:
            # 发布停止命令
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            return

        # 检查是否到达最终目标
        final_goal = self.planned_path[-1]
        final_distance = self.distance(self.current_pos, final_goal)

        if final_distance < self.goal_tolerance:
            self.goal_reached = True
            self.is_navigating = False
            cmd = Twist()
            self.cmd_vel_pub.publish(cmd)
            self.get_logger().info(f'🎉 目标到达！最终距离: {final_distance:.3f}m')
            return

        # 寻找前瞻点
        target_point = self.find_lookahead_point()
        if target_point is None:
            return

        # 障碍物规避
        safe_target = self.avoid_obstacles(target_point)

        # Pure Pursuit控制
        distance_error, angle_error = self.pure_pursuit_control(safe_target)

        # 计算时间差
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        self.last_time = current_time

        if dt > 0.2:  # 如果时间间隔太大，重置PID
            self.linear_integral = 0.0
            self.angular_integral = 0.0
            dt = 0.1

        # PID控制
        linear_cmd, angular_cmd = self.pid_control(distance_error, angle_error, dt)

        # 限制速度
        linear_cmd = max(-self.max_linear_speed, min(self.max_linear_speed, linear_cmd))
        angular_cmd = max(-self.max_angular_speed, min(self.max_angular_speed, angular_cmd))

        # 如果角度误差太大，降低线速度
        if abs(angle_error) > math.pi / 3:  # 60度
            linear_cmd *= 0.3
        elif abs(angle_error) > math.pi / 6:  # 30度
            linear_cmd *= 0.6

        # 如果距离很近，降低速度
        if distance_error < 0.3:
            linear_cmd *= 0.5

        # 发布控制命令
        cmd = Twist()
        cmd.linear.x = linear_cmd
        cmd.angular.z = angular_cmd
        self.cmd_vel_pub.publish(cmd)

        # 定期输出状态信息
        current_log_time = current_time.nanoseconds / 1e9
        if current_log_time - self.last_nav_log_time >= self.nav_log_interval:
            self.get_logger().info(
                f'🚗 导航中 - 距目标{final_distance:.2f}m | 速度[{linear_cmd:.2f}, {angular_cmd:.2f}] | 进度{self.current_goal_index + 1}/{len(self.planned_path)}'
            )
            self.last_nav_log_time = current_log_time


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