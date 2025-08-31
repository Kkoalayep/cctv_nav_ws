#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
import math


class DebugNavigation(Node):
    def __init__(self):
        super().__init__('debug_navigation')
        
        self.robot_position = None
        self.target_position = None
        
        # 订阅
        self.robot_sub = self.create_subscription(
            PointStamped, '/robot_position', self.robot_callback, 10)
        self.target_sub = self.create_subscription(
            PointStamped, '/target_position', self.target_callback, 10)
        
        # 发布
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 定时器
        self.create_timer(0.5, self.control_loop)
        
        self.get_logger().info('🔍 调试导航启动 - 超慢速度')

    def robot_callback(self, msg):
        self.robot_position = (msg.point.x, msg.point.y)
        self.get_logger().info(f'🤖 机器人位置: ({msg.point.x:.2f}, {msg.point.y:.2f})')

    def target_callback(self, msg):
        self.target_position = (msg.point.x, msg.point.y)
        self.get_logger().info(f'🎯 目标位置: ({msg.point.x:.2f}, {msg.point.y:.2f})')

    def control_loop(self):
        cmd = Twist()
        
        if not self.robot_position or not self.target_position:
            self.get_logger().info('⏸️ 等待位置信息...')
            self.cmd_pub.publish(cmd)
            return
            
        robot_x, robot_y = self.robot_position
        target_x, target_y = self.target_position
        
        # 计算距离和方向
        dx = target_x - robot_x
        dy = target_y - robot_y
        distance = math.sqrt(dx*dx + dy*dy)
        target_angle = math.atan2(dy, dx)
        
        self.get_logger().info(
            f'📍 位置差: dx={dx:.2f}, dy={dy:.2f}'
        )
        self.get_logger().info(
            f'📏 距离: {distance:.2f}m, 角度: {target_angle:.2f}弧度({math.degrees(target_angle):.1f}度)'
        )
        
        if distance < 0.2:
            self.get_logger().info('🎉 到达目标！')
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            # 超级简单的控制：先转向，再前进
            if abs(target_angle) > 0.2:  # 如果角度偏差大于0.2弧度
                self.get_logger().info(f'🔄 转向中... 目标角度: {math.degrees(target_angle):.1f}度')
                cmd.linear.x = 0.0
                cmd.angular.z = 0.1 if target_angle > 0 else -0.1  # 超慢转向
            else:
                self.get_logger().info(f'➡️ 前进中... 距离: {distance:.2f}m')
                cmd.linear.x = 0.03  # 超慢前进 3cm/s
                cmd.angular.z = 0.0
        
        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = DebugNavigation()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
