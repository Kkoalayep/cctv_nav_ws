#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf2_ros
from tf2_ros import TransformException


def quat_to_rot_matrix(qx, qy, qz, qw):
    """四元数转旋转矩阵"""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]
    ], dtype=np.float64)


class ColorTracker(Node):
    def __init__(self):
        super().__init__('color_tracker')
        self.bridge = CvBridge()

        # 参数
        self.world_frame = self.declare_parameter('world_frame', 'odom').get_parameter_value().string_value
        self.tf_timeout = 0.1
        self.camera_height = 5.0

        # 物体高度（用于透视校正）
        self.heights = {'robot': 0.1, 'obstacle': 0.3, 'target': 0.25}

        # 颜色范围
        self.color_ranges = {
            'robot': (np.array([0, 0, 0]), np.array([180, 255, 80])),
            'obstacle': (np.array([40, 50, 50]), np.array([80, 255, 255])),
            'target': [(np.array([0, 50, 50]), np.array([10, 255, 255])),
                       (np.array([170, 50, 50]), np.array([180, 255, 255]))]
        }

        self.min_areas = {'robot': 200, 'obstacle': 300, 'target': 150}
        self.colors = {'robot': (0, 0, 255), 'obstacle': (0, 255, 0), 'target': (255, 0, 0)}

        # 相机参数
        self.fx = self.fy = self.cx = self.cy = None
        self.last_image_frame = None

        # 日志控制
        self.last_robot_log_time = 0.0
        self.robot_log_interval = 2.0  # 机器人2秒一次
        self.logged_obstacles = set()  # 已记录的障碍物位置
        self.logged_targets = set()  # 已记录的目标位置

        # TF和ROS接口
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.setup_ros_interfaces()

        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.get_logger().info('颜色追踪器启动')

    def setup_ros_interfaces(self):
        """设置ROS接口"""
        self.image_sub = self.create_subscription(Image, '/overhead/overhead_camera/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/overhead/overhead_camera/camera_info',
                                                        self.camera_info_callback, 10)

        # 创建发布者
        self.robot_pub = self.create_publisher(PointStamped, '/robot_position', 10)
        self.obstacle_pub = self.create_publisher(PointStamped, '/obstacle_positions', 10)
        self.target_pub = self.create_publisher(PointStamped, '/target_position', 10)
        self.debug_image_pub = self.create_publisher(Image, '/debug_image', 10)

    def get_publisher(self, obj_type):
        """获取对应类型的发布者"""
        if obj_type == 'robot':
            return self.robot_pub
        elif obj_type == 'obstacle':
            return self.obstacle_pub
        elif obj_type == 'target':
            return self.target_pub
        else:
            raise ValueError(f"Unknown object type: {obj_type}")

    def should_log(self, obj_type, world_pos):
        """判断是否应该记录日志"""
        current_time = self.get_clock().now().nanoseconds / 1e9

        if obj_type == 'robot':
            # 机器人每2秒记录一次
            if current_time - self.last_robot_log_time >= self.robot_log_interval:
                self.last_robot_log_time = current_time
                return True
            return False

        elif obj_type == 'obstacle':
            # 障碍物只记录一次（基于位置）
            pos_key = (round(world_pos[0], 1), round(world_pos[1], 1))
            if pos_key not in self.logged_obstacles:
                self.logged_obstacles.add(pos_key)
                return True
            return False

        elif obj_type == 'target':
            # 目标只记录一次（基于位置）
            pos_key = (round(world_pos[0], 1), round(world_pos[1], 1))
            if pos_key not in self.logged_targets:
                self.logged_targets.add(pos_key)
                return True
            return False

        return False

    def camera_info_callback(self, msg: CameraInfo):
        """相机参数回调"""
        if self.fx is None:  # 只记录一次
            self.fx, self.fy = msg.k[0], msg.k[4]
            self.cx, self.cy = msg.k[2], msg.k[5]
            self.get_logger().info(f'相机参数: {msg.width}x{msg.height}, fx={self.fx:.1f}, fy={self.fy:.1f}')

    def perspective_correction(self, pixel_x, pixel_y, object_height, stamp):
        """透视校正：高精度方法"""
        try:
            # 原始像素->世界坐标
            wx_orig, wy_orig = self.pixel_to_world(pixel_x, pixel_y, stamp)

            # 几何校正
            distance = np.sqrt(wx_orig * wx_orig + wy_orig * wy_orig)
            if distance > 1e-6:
                ratio = (self.camera_height - object_height) / self.camera_height
                wx_corr = wx_orig * ratio
                wy_corr = wy_orig * ratio

                # 校正后世界坐标->像素坐标
                px_corr, py_corr = self.world_to_pixel(wx_corr, wy_corr, stamp)
                return px_corr, py_corr, (wx_orig, wy_orig), (wx_corr, wy_corr)

        except Exception as e:
            self.get_logger().warn(f'透视校正失败: {e}')

        # 失败时返回原始值
        wx_orig, wy_orig = self.pixel_to_world(pixel_x, pixel_y, stamp)
        return pixel_x, pixel_y, (wx_orig, wy_orig), (wx_orig, wy_orig)

    def pixel_to_world(self, pixel_x, pixel_y, stamp):
        """像素转世界坐标"""
        # 像素->射线
        x_cam = (pixel_x - self.cx) / self.fx
        y_cam = (pixel_y - self.cy) / self.fy
        ray_cam = np.array([x_cam, y_cam, 1.0]) / np.sqrt(x_cam * x_cam + y_cam * y_cam + 1.0)

        # TF变换
        tf_msg = self.tf_buffer.lookup_transform(self.world_frame, self.last_image_frame, stamp,
                                                 rclpy.duration.Duration(seconds=self.tf_timeout))
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        t_world = np.array([t.x, t.y, t.z])
        R_wc = quat_to_rot_matrix(q.x, q.y, q.z, q.w)

        # 射线与地面交点
        ray_world = R_wc @ ray_cam
        s = -t_world[2] / ray_world[2]
        if s < 0:
            raise RuntimeError(f'交点在相机后方 (s={s:.3f})')

        p_world = t_world + s * ray_world
        return float(p_world[0]), float(p_world[1])

    def world_to_pixel(self, world_x, world_y, stamp):
        """世界坐标转像素"""
        tf_msg = self.tf_buffer.lookup_transform(self.last_image_frame, self.world_frame, stamp,
                                                 rclpy.duration.Duration(seconds=self.tf_timeout))
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        t_cam = np.array([t.x, t.y, t.z])
        R_cw = quat_to_rot_matrix(q.x, q.y, q.z, q.w)

        p_world = np.array([world_x, world_y, 0.0])
        p_cam = R_cw @ p_world + t_cam

        pixel_x = (p_cam[0] / p_cam[2]) * self.fx + self.cx
        pixel_y = (p_cam[1] / p_cam[2]) * self.fy + self.cy
        return pixel_x, pixel_y

    def detect_objects(self, hsv_image, obj_type):
        """通用物体检测"""
        if obj_type == 'target':
            # 红色需要两段HSV
            mask1 = cv2.inRange(hsv_image, self.color_ranges[obj_type][0][0], self.color_ranges[obj_type][0][1])
            mask2 = cv2.inRange(hsv_image, self.color_ranges[obj_type][1][0], self.color_ranges[obj_type][1][1])
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_image, self.color_ranges[obj_type][0], self.color_ranges[obj_type][1])

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_areas[obj_type]:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    results.append((contour, (cx, cy)))
                    if obj_type in ['robot', 'target']:  # 只返回最大的一个
                        break

        return results

    def draw_detection_info(self, debug_image, contour, center, obj_type, world_orig, world_corr, px_corr=None):
        """绘制检测信息"""
        cx, cy = center
        color = self.colors[obj_type]

        # 绘制边界框和中心
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(debug_image, (x, y), (x + w, y + h), color, 2)
        cv2.circle(debug_image, (cx, cy), 8, color, -1)

        # 绘制校正线
        if px_corr:
            cv2.circle(debug_image, (int(px_corr[0]), int(px_corr[1])), 5, (255, 255, 0), 2)
            cv2.line(debug_image, (cx, cy), (int(px_corr[0]), int(px_corr[1])), (255, 255, 0), 2)

        # 文本信息
        info_lines = [
            obj_type.capitalize(),
            f'Original: ({round(world_orig[0], 1)}, {round(world_orig[1], 1)})',
            f'Corrected: ({round(world_corr[0], 1)}, {round(world_corr[1], 1)})'
        ]

        # 绘制文本
        text_y = y - 10
        for line in info_lines:
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            bg_y = text_y - text_size[1] - 5
            cv2.rectangle(debug_image, (x, bg_y), (x + text_size[0] + 10, text_y + 5), (0, 0, 0), -1)
            cv2.putText(debug_image, line, (x + 5, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            text_y -= text_size[1] + 8

    def process_object_type(self, hsv_image, debug_image, obj_type, stamp):
        """处理特定类型的物体"""
        detections = self.detect_objects(hsv_image, obj_type)

        for contour, center in detections:
            try:
                px, py = center
                px_corr, py_corr, world_orig, world_corr = self.perspective_correction(
                    px, py, self.heights[obj_type], stamp)

                # 发布校正后的坐标
                ps = PointStamped()
                ps.header.stamp = stamp.to_msg()
                ps.header.frame_id = self.world_frame
                ps.point.x, ps.point.y, ps.point.z = float(world_corr[0]), float(world_corr[1]), 0.0
                self.get_publisher(obj_type).publish(ps)

                # 绘制检测信息
                self.draw_detection_info(debug_image, contour, center, obj_type, world_orig, world_corr,
                                         (px_corr, py_corr))

                # 日志输出（控制频率）
                if self.should_log(obj_type, world_corr):
                    wx_orig_rounded = (round(world_orig[0], 1), round(world_orig[1], 1))
                    wx_corr_rounded = (round(world_corr[0], 1), round(world_corr[1], 1))
                    self.get_logger().info(f'{obj_type}: 原始坐标{wx_orig_rounded} -> 修正坐标{wx_corr_rounded}')

            except Exception as e:
                self.get_logger().warn(f'{obj_type}处理失败: {e}')

    def image_callback(self, msg: Image):
        """图像处理主回调"""
        if self.fx is None:
            return

        try:
            self.last_image_frame = msg.header.frame_id
            stamp = Time.from_msg(msg.header.stamp)

            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            debug_image = cv_image.copy()
            hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

            # 添加校正方法标识
            cv2.putText(debug_image, "Correction: Advanced", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                        2)

            # 处理各类物体
            for obj_type in ['robot', 'obstacle', 'target']:
                self.process_object_type(hsv_image, debug_image, obj_type, stamp)

            # 发布调试图像
            debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
            debug_msg.header = msg.header
            self.debug_image_pub.publish(debug_msg)

        except Exception as e:
            self.get_logger().error(f'图像处理错误: {e}')


def main(args=None):
    rclpy.init(args=args)
    tracker = ColorTracker()
    try:
        rclpy.spin(tracker)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()