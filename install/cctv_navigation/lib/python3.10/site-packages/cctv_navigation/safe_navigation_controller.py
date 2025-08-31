#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
import numpy as np
import math
import time
from enum import Enum


class NavigationState(Enum):
    IDLE = 0
    NAVIGATING = 1
    REACHED = 2
    AVOIDING = 3


class SafeNavigationController(Node):
    def __init__(self):
        super().__init__('safe_navigation_controller')
        
        # 安全导航参数
        self.max_linear_speed = 0.08  # 降低速度
        self.max_angular_speed = 0.4
        self.goal_tolerance = 0.15
        self.safety_distance = 0.3  # 安全距离30cm
        
        # 状态变量
        self.current_state = NavigationState.IDLE
        self.robot_position = None
        self.target_position = None
        self.current_map = None
        
        # ROS接口
        self.setup_ros_interfaces()
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info('安全导航控制器启动 - 保守策略')

    def setup_ros_interfaces(self):
        self.robot_sub = self.create_subscription(
            PointStamped, '/robot_position', self.robot_position_callback, 10)
        self.target_sub = self.create_subscription(
            PointStamped, '/target_position', self.target_position_callback, 10)
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def robot_position_callback(self, msg):
        self.robot_position = (msg.point.x, msg.point.y)

    def target_position_callback(self, msg):
        self.target_position = (msg.point.x, msg.point.y)
        if self.target_position:
            self.current_state = NavigationState.NAVIGATING
            self.get_logger().info(f'新目标: {self.target_position}')

    def map_callback(self, msg):
        self.current_map = msg

    def world_to_grid(self, world_x, world_y):
        """世界坐标转栅格坐标"""
        if not self.current_map:
            return None, None
        
        origin_x = self.current_map.info.origin.position.x
        origin_y = self.current_map.info.origin.position.y
        resolution = self.current_map.info.resolution
        
        grid_x = int((world_x - origin_x) / resolution)
        grid_y = int((world_y - origin_y) / resolution)
        return grid_x, grid_y

    def is_obstacle_in_direction(self, angle, check_distance=0.4):
        """检查指定方向是否有障碍物"""
        if not self.robot_position or not self.current_map:
            return False
        
        robot_x, robot_y = self.robot_position
        
        # 检查多个点
        num_checks = int(check_distance / 0.05)  # 每5cm检查一次
        for i in range(1, num_checks + 1):
            check_x = robot_x + math.cos(angle) * (i * 0.05)
            check_y = robot_y + math.sin(angle) * (i * 0.05)
            
            grid_x, grid_y = self.world_to_grid(check_x, check_y)
            if grid_x is None or grid_y is None:
                return True
            
            # 检查边界
            if (grid_x < 0 or grid_x >= self.current_map.info.width or
                grid_y < 0 or grid_y >= self.current_map.info.height):
                return True
            
            # 检查障碍物
            index = grid_y * self.current_map.info.width + grid_x
            if index < len(self.current_map.data):
                if self.current_map.data[index] > 50:  # 障碍物
                    return True
        
        return False

    def find_safe_direction(self, target_angle):
        """找到一个安全的行进方向"""
        # 首先检查目标方向是否安全
        if not self.is_obstacle_in_direction(target_angle):
            return target_angle
        
        # 如果目标方向不安全，尝试左右偏移
        for offset in [0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2]:
            safe_angle = target_angle + offset
            if not self.is_obstacle_in_direction(safe_angle):
                self.get_logger().info(f'避障：偏移{offset:.1f}弧度')
                return safe_angle
        
        # 如果都不安全，停止
        self.get_logger().warn('所有方向都不安全，停止')
        return None

    def control_loop(self):
        """主控制循环"""
        cmd = Twist()
        
        if self.current_state == NavigationState.IDLE:
            pass
            
        elif self.current_state == NavigationState.NAVIGATING:
            if not self.robot_position or not self.target_position:
                return
            
            robot_x, robot_y = self.robot_position
            target_x, target_y = self.target_position
            
            # 计算距离和目标角度
            dx = target_x - robot_x
            dy = target_y - robot_y
            distance = math.sqrt(dx*dx + dy*dy)
            target_angle = math.atan2(dy, dx)
            
            if distance < self.goal_tolerance:
                self.current_state = NavigationState.REACHED
                self.get_logger().info('🎯 安全到达目标！')
                return
            
            # 寻找安全方向
            safe_angle = self.find_safe_direction(target_angle)
            
            if safe_angle is not None:
                # 安全前进
                self.current_state = NavigationState.NAVIGATING
                
                # 计算角度误差
                angle_error = safe_angle - 0  # 假设机器人朝向为0
                while angle_error > math.pi:
                    angle_error -= 2 * math.pi
                while angle_error < -math.pi:
                    angle_error += 2 * math.pi
                
                # 保守的速度控制
                if abs(angle_error) > 0.3:
                    # 主要转向
                    cmd.angular.z = max(-self.max_angular_speed, 
                                       min(self.max_angular_speed, angle_error * 1.5))
                    cmd.linear.x = 0.02  # 转向时很慢前进
                else:
                    # 小心前进
                    cmd.linear.x = min(self.max_linear_speed, distance * 0.3)
                    cmd.angular.z = angle_error * 0.8
                
                # 最后安全检查：如果前方有障碍物，立即停止
                if self.is_obstacle_in_direction(safe_angle, 0.25):
                    cmd.linear.x = 0.0
                    self.current_state = NavigationState.AVOIDING
                    self.get_logger().warn('⚠️ 紧急停止：前方检测到障碍物')
                
            else:
                # 没有安全方向，停止并尝试后退
                self.current_state = NavigationState.AVOIDING
                cmd.linear.x = -0.03  # 慢速后退
                cmd.angular.z = 0.2   # 轻微转向
                
        elif self.current_state == NavigationState.AVOIDING:
            # 避障状态：慢速后退并转向
            cmd.linear.x = -0.02
            cmd.angular.z = 0.3
            
            # 2秒后重新尝试导航
            time.sleep(0.1)
            if time.time() % 20 < 0.1:  # 每2秒重试一次
                self.current_state = NavigationState.NAVIGATING
                self.get_logger().info('重新尝试导航')
                
        elif self.current_state == NavigationState.REACHED:
            pass
        
        # 发布控制命令
        self.cmd_vel_pub.publish(cmd)
        
        # 调试信息
        if (self.robot_position and self.target_position and 
            time.time() % 1 < 0.1):  # 每秒打印一次
            
            robot_x, robot_y = self.robot_position
            target_x, target_y = self.target_position
            distance = math.sqrt((target_x - robot_x)**2 + (target_y - robot_y)**2)
            
            self.get_logger().info(
                f'🤖 状态: {self.current_state.name} | '
                f'距离: {distance:.2f}m | 速度: ({cmd.linear.x:.2f}, {cmd.angular.z:.2f})'
            )


def main():
    rclpy.init()
    controller = SafeNavigationController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
