#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
from collections import defaultdict

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import Image
from std_msgs.msg import Header
import cv2
from cv_bridge import CvBridge


class MapBuilder(Node):
    """
    构建基础障碍图，并发布两张地图：
      - /map_planning   ：较厚膨胀（规划更保守）
      - /map_navigation ：较薄或不膨胀（导航期碰撞检查更宽松，减少"空气墙"）
    话题：
      订阅：/robot_position, /target_position, /obstacle_positions (PointStamped)
      发布：/map_planning, /map_navigation (OccupancyGrid), /map_visualization (Image)
    """

    def __init__(self):
        super().__init__('map_builder')

        # ------ 地图参数 ------
        self.resolution = self.declare_parameter('resolution', 0.05).value  # m/cell
        self.world_size = self.declare_parameter('world_size', 8.0).value  # m，正方形
        self.origin_x = self.declare_parameter('origin_x', -4.0).value
        self.origin_y = self.declare_parameter('origin_y', -4.0).value

        self.grid_w = int(self.world_size / self.resolution)
        self.grid_h = int(self.world_size / self.resolution)

        # 基础占用图（0=自由, 100=障碍）
        self.base_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.int16)
        self._add_border_walls(thickness=2)

        # ------ 发布频率与"仅变化时发布" ------
        self.publish_period = self.declare_parameter('publish_period', 0.5).value  # 2Hz
        self.force_pub_after = self.declare_parameter('force_pub_after', 3.0).value
        self.last_pub_time = 0.0
        self.last_hash_plan = None
        self.last_hash_nav = None

        # ------ 障碍检测去抖 ------
        self.history_timeout = self.declare_parameter('history_timeout', 5.0).value
        self.detection_threshold = self.declare_parameter('detection_threshold', 1).value  # 命中次数阈值
        self.obstacle_radius_cells = self.declare_parameter('obstacle_radius_cells', 2).value  # 基础障碍打点半径（格）
        self.obstacle_hits = defaultdict(list)  # {(gx,gy): [t1,t2,...]}

        # ------ 膨胀（米） ------
        self.inflate_plan_m = self.declare_parameter('inflate_plan_m', 0.20).value  # 规划用
        self.inflate_nav_m = self.declare_parameter('inflate_nav_m', 0.00).value  # 导航用（0=不膨胀）

        # ------ 清自由的范围（避免把新障碍擦掉，适当缩小） ------
        self.robot_free_radius = self.declare_parameter('robot_free_radius_cells', 4).value
        self.target_free_radius = self.declare_parameter('target_free_radius_cells', 2).value

        # ------ 新增：可视化数据 ------
        self.robot_path = []  # 机器人走过的路径
        self.target_position = None  # 目标点位置
        self.planned_path = []  # 规划的路径

        # ROS I/O
        self.map_plan_pub = self.create_publisher(OccupancyGrid, '/map_planning', 10)
        self.map_nav_pub = self.create_publisher(OccupancyGrid, '/map_navigation', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)  # 新增：为RViz兼容
        self.viz_pub = self.create_publisher(Image, '/map_visualization', 10)
        self.bridge = CvBridge()

        self.create_subscription(PointStamped, '/robot_position', self.robot_cb, 30)
        self.create_subscription(PointStamped, '/target_position', self.target_cb, 10)
        self.create_subscription(PointStamped, '/obstacle_positions', self.obstacle_cb, 30)
        self.create_subscription(Path, '/planned_path', self.path_cb, 10)  # 新增：订阅规划路径

        # 定时发布
        self.create_timer(self.publish_period, self._maybe_publish)

        self.get_logger().info(
            f'MapBuilder: {self.grid_w}x{self.grid_h} @ {self.resolution} m/px, '
            f'plan inflate={self.inflate_plan_m}m, nav inflate={self.inflate_nav_m}m'
        )

    # ---------- 回调 ----------
    def robot_cb(self, msg: PointStamped):
        gx, gy = self.world_to_grid(msg.point.x, msg.point.y)
        if self.in_bounds(gx, gy):
            # 记录路径
            self.robot_path.append((gx, gy))
            if len(self.robot_path) > 400:
                self.robot_path.pop(0)
            # 附近清自由（不要覆盖已是障碍的格）
            self._mark_free(gx, gy, self.robot_free_radius)

    def target_cb(self, msg: PointStamped):
        # 保存目标点世界坐标
        self.target_position = (msg.point.x, msg.point.y)
        gx, gy = self.world_to_grid(msg.point.x, msg.point.y)
        if self.in_bounds(gx, gy):
            self._mark_free(gx, gy, self.target_free_radius)

    def obstacle_cb(self, msg: PointStamped):
        gx, gy = self.world_to_grid(msg.point.x, msg.point.y)
        if not self.in_bounds(gx, gy):
            return
        now = time.time()
        lst = self.obstacle_hits[(gx, gy)]
        lst.append(now)
        # 保留时间窗口内的命中
        self.obstacle_hits[(gx, gy)] = [t for t in lst if now - t < self.history_timeout]
        if len(self.obstacle_hits[(gx, gy)]) >= int(self.detection_threshold):
            self._mark_obstacle(gx, gy, self.obstacle_radius_cells)

    def path_cb(self, msg: Path):
        """新增：接收规划路径"""
        self.planned_path = []
        for pose in msg.poses:
            x = pose.pose.position.x
            y = pose.pose.position.y
            self.planned_path.append((x, y))

    # ---------- 核心：按需发布两张地图 ----------
    def _maybe_publish(self):
        plan = self._inflate_grid(self.base_grid, self.inflate_plan_m)
        nav = self._inflate_grid(self.base_grid, self.inflate_nav_m)

        h_plan = hash(plan.tobytes())
        h_nav = hash(nav.tobytes())
        now = time.time()
        changed = (self.last_hash_plan != h_plan) or (self.last_hash_nav != h_nav)
        force = (now - self.last_pub_time) > float(self.force_pub_after)

        if changed or force:
            self._publish_grid('/map_planning', plan, self.map_plan_pub)
            self._publish_grid('/map_navigation', nav, self.map_nav_pub)
            self._publish_grid('/map', plan, self.map_pub)  # 新增：为RViz发布标准地图话题
            self._publish_viz(plan, nav)
            self.last_hash_plan = h_plan
            self.last_hash_nav = h_nav
            self.last_pub_time = now

    # ---------- 地图写入 ----------
    def _mark_free(self, cx, cy, r):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                gx, gy = cx + dx, cy + dy
                if self.in_bounds(gx, gy) and self.base_grid[gy, gx] != 100:
                    self.base_grid[gy, gx] = 0

    def _mark_obstacle(self, cx, cy, r_cells):
        r2 = r_cells * r_cells
        for dy in range(-r_cells, r_cells + 1):
            for dx in range(-r_cells, r_cells + 1):
                if dx * dx + dy * dy <= r2:
                    gx, gy = cx + dx, cy + dy
                    if self.in_bounds(gx, gy):
                        self.base_grid[gy, gx] = 100

    # ---------- 膨胀 ----------
    def _inflate_grid(self, grid: np.ndarray, radius_m: float) -> np.ndarray:
        """正方邻域朴素膨胀：radius_m / resolution 个格子"""
        obst = (grid >= 50).astype(np.uint8)
        cells = max(0, int(round(float(radius_m) / max(self.resolution, 1e-6))))
        if cells == 0 or obst.sum() == 0:
            return np.where(obst > 0, 100, 0).astype(np.int16)
        h, w = obst.shape
        mask = obst.copy()
        for dy in range(-cells, cells + 1):
            for dx in range(-cells, cells + 1):
                yy = np.clip(np.arange(h)[:, None] + dy, 0, h - 1)
                xx = np.clip(np.arange(w)[None, :] + dx, 0, w - 1)
                mask = np.maximum(mask, obst[yy, xx])
        return np.where(mask > 0, 100, 0).astype(np.int16)

    # ---------- 发布 ----------
    def _publish_grid(self, frame: str, grid: np.ndarray, pub):
        msg = OccupancyGrid()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'  # 统一用 map 框架
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_w
        msg.info.height = self.grid_h
        msg.info.origin.position.x = self.origin_x
        msg.info.origin.position.y = self.origin_y
        msg.info.origin.orientation.w = 1.0
        msg.data = grid.reshape(-1).astype(int).tolist()
        pub.publish(msg)

    def _publish_viz(self, plan: np.ndarray, nav: np.ndarray):
        """发布可视化图像：只显示目标点、走过的路线、规划的路线"""
        # 白色背景
        img = np.ones((self.grid_h, self.grid_w, 3), dtype=np.uint8) * 255

        # 关键改进：只显示基础障碍物（未膨胀的），而不是膨胀后的
        img[self.base_grid == 100] = (0, 0, 0)  # 只显示原始障碍物大小

        # 1. 绘制机器人走过的路径（蓝色）
        for gx, gy in self.robot_path[-200:]:  # 只显示最近200个点
            if self.in_bounds(gx, gy):
                img[self.grid_h - 1 - gy, gx] = (255, 0, 0)  # BGR格式，蓝色

        # 2. 绘制规划路径（绿色线条）
        if len(self.planned_path) >= 2:
            for i in range(len(self.planned_path) - 1):
                x1, y1 = self.planned_path[i]
                x2, y2 = self.planned_path[i + 1]
                gx1, gy1 = self.world_to_grid(x1, y1)
                gx2, gy2 = self.world_to_grid(x2, y2)

                if self.in_bounds(gx1, gy1) and self.in_bounds(gx2, gy2):
                    self._draw_line(img, gx1, gy1, gx2, gy2, (0, 255, 0))  # 绿色

        # 3. 绘制目标点（红色X）
        if self.target_position:
            tx, ty = self.target_position
            gx, gy = self.world_to_grid(tx, ty)
            if self.in_bounds(gx, gy):
                self._draw_x(img, gx, gy, 4, (0, 0, 255))  # 红色X

        # 放大显示
        scale = 3
        vis = cv2.resize(img, (self.grid_w * scale, self.grid_h * scale), interpolation=cv2.INTER_NEAREST)

        # 发布
        msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'  # 改为odom
        self.viz_pub.publish(msg)

    # ---------- 绘图辅助函数 ----------
    def _draw_line(self, img, x1, y1, x2, y2, color):
        """在网格图像上绘制线段"""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        x_inc = 1 if x1 < x2 else -1
        y_inc = 1 if y1 < y2 else -1
        error = dx - dy

        for _ in range(dx + dy + 1):
            if self.in_bounds(x, y):
                img[self.grid_h - 1 - y, x] = color
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

    def _draw_x(self, img, cx, cy, size, color):
        """绘制X标记"""
        for i in range(-size, size + 1):
            # 主对角线 (\)
            if self.in_bounds(cx + i, cy + i):
                img[self.grid_h - 1 - (cy + i), cx + i] = color
            # 副对角线 (/)
            if self.in_bounds(cx + i, cy - i):
                img[self.grid_h - 1 - (cy - i), cx + i] = color

    # ---------- 工具 ----------
    def world_to_grid(self, x: float, y: float):
        gx = int((x - self.origin_x) / self.resolution)
        gy = int((y - self.origin_y) / self.resolution)
        return gx, gy

    def in_bounds(self, gx: int, gy: int) -> bool:
        return 0 <= gx < self.grid_w and 0 <= gy < self.grid_h

    def _add_border_walls(self, thickness=2):
        self.base_grid[0:thickness, :] = 100
        self.base_grid[-thickness:, :] = 100
        self.base_grid[:, 0:thickness] = 100
        self.base_grid[:, -thickness:] = 100


def main(args=None):
    rclpy.init(args=args)
    node = MapBuilder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()