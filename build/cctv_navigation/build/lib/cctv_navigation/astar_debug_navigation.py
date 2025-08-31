#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 纯视觉 A* 导航：规划与控制用不同膨胀地图
# - 订阅 /map_planning（厚膨胀）做 A*，/map_navigation（薄/不膨胀）做线段碰撞检查
# - 仅用视觉 /robot_position /target_position；无 /odom
# - lookahead 跟随 + 滞回 + 角误差低通
# - 目标去抖、地图冷却、定时重规划

import math, time, heapq, numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from enum import IntEnum
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Header


def wrap_to_pi(a: float) -> float:
    while a > math.pi: a -= 2.0*math.pi
    while a < -math.pi: a += 2.0*math.pi
    return a


class NavigationState(IntEnum):
    IDLE=0; PLANNING=1; NAVIGATING=2; NO_PATH=3; REACHED=4


class YawEstimator:
    """用 (x,y,t) 序列估计朝向 yaw = atan2(vy,vx)"""
    def __init__(self, window_sec=0.4, min_dt=0.15, min_disp=0.01):
        self.w=float(window_sec); self.min_dt=float(min_dt); self.min_disp=float(min_disp)
        self.buf=deque(); self.yaw=None
    def update(self,x,y,t):
        self.buf.append((t,x,y))
        t0=t-self.w
        while len(self.buf)>=2 and self.buf[0][0]<t0: self.buf.popleft()
        if len(self.buf)<2: return self.yaw
        t_old,x_old,y_old=self.buf[0]; dt=t-t_old; dx=x-x_old; dy=y-y_old
        if dt>=self.min_dt and math.hypot(dx,dy)>=self.min_disp: self.yaw=math.atan2(dy,dx)
        return self.yaw


class DualMapPlanner:
    """同时持有规划图(plan)与控制图(nav)，A* 用 plan，线段检查用 nav。"""
    def __init__(self, map_plan: np.ndarray, map_nav: np.ndarray,
                 resolution: float, origin_xy, obstacle_threshold=50):
        self.map_plan = map_plan.astype(np.int16)
        self.map_nav  = map_nav.astype(np.int16)
        self.res = float(resolution)
        self.ox, self.oy = origin_xy
        self.th = int(obstacle_threshold)
        self.h, self.w = self.map_plan.shape

    # 坐标转换
    def world_to_grid(self,x,y):
        return int(math.floor((x-self.ox)/self.res)), int(math.floor((y-self.oy)/self.res))
    def grid_to_world(self,gx,gy):
        return self.ox+(gx+0.5)*self.res, self.oy+(gy+0.5)*self.res

    # 查询
    def in_bounds(self,gx,gy): return 0<=gx<self.w and 0<=gy<self.h
    def free_plan(self,gx,gy): return self.map_plan[gy,gx] < self.th
    def free_nav(self,gx,gy):  return self.map_nav [gy,gx] < self.th
    def valid_plan(self,gx,gy):return self.in_bounds(gx,gy) and self.free_plan(gx,gy)

    # 邻居（A*）
    def neighbors(self,node):
        x,y=node; res=[]
        dirs=[(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
              (-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),(1,1,math.sqrt(2))]
        for dx,dy,c in dirs:
            nx,ny=x+dx,y+dy
            if not self.valid_plan(nx,ny): continue
            if dx!=0 and dy!=0:
                if not (self.valid_plan(x+dx,y) and self.valid_plan(x,y+dy)): continue
            res.append(((nx,ny),c))
        return res

    @staticmethod
    def heuristic(a,b):
        dx,dy=abs(a[0]-b[0]),abs(a[1]-b[1])
        dmin,dmax=min(dx,dy),max(dx,dy)
        return math.sqrt(2)*dmin+(dmax-dmin)

    def plan(self, start_xy, goal_xy):
        sx,sy=self.world_to_grid(*start_xy); gx,gy=self.world_to_grid(*goal_xy)
        if not self.valid_plan(sx,sy) or not self.valid_plan(gx,gy): return []
        start,goal=(sx,sy),(gx,gy)
        openh=[(0.0,start)]; came={start:None}; gc={start:0.0}
        while openh:
            _,cur=heapq.heappop(openh)
            if cur==goal: break
            for (nx,ny),step in self.neighbors(cur):
                ng=gc[cur]+step
                if (nx,ny) not in gc or ng<gc[(nx,ny)]:
                    gc[(nx,ny)]=ng
                    heapq.heappush(openh,(ng+self.heuristic((nx,ny),goal),(nx,ny)))
                    came[(nx,ny)]=cur
        if goal not in came: return []
        cells=[]; c=goal
        while c is not None: cells.append(c); c=came[c]
        cells.reverse()
        return [self.grid_to_world(cx,cy) for (cx,cy) in cells]

    # 线段占用检查：对“导航图”（薄/不膨胀）检查
    def segment_is_free_world(self, x0, y0, x1, y1,
                              step_m=0.03, margin_m=0.08,
                              min_consecutive_hits=2, max_len_m=0.6):
        dx,dy=x1-x0,y1-y0
        L=math.hypot(dx,dy)
        if L<1e-6: return True
        L=min(L,max_len_m)
        inv=1.0/(math.hypot(dx,dy)+1e-9)
        nx,ny=dx*inv,dy*inv
        n=max(2,int(L/max(step_m,1e-3)))
        start_i=int(margin_m/max(step_m,1e-3)); end_i=n-start_i
        consec=0
        for i in range(start_i,end_i+1):
            t=(i/n)*L; x=x0+nx*t; y=y0+ny*t
            gx,gy=self.world_to_grid(x,y)
            if not self.in_bounds(gx,gy): return False
            if not self.free_nav(gx,gy):
                consec+=1
                if consec>=int(min_consecutive_hits): return False
            else:
                consec=0
        return True


class NavigationController(Node):
    def __init__(self):
        super().__init__('astar_debug_navigation_dualmap')

        # 运动/控制参数
        self.max_linear_speed   = self.declare_parameter('max_linear_speed',   0.16).value
        self.max_angular_speed  = self.declare_parameter('max_angular_speed',  1.0).value
        self.goal_tolerance     = self.declare_parameter('goal_tolerance',     0.18).value
        self.lookahead_distance = self.declare_parameter('lookahead_distance', 0.45).value
        self.min_crawl_speed    = self.declare_parameter('min_crawl_speed',    0.05).value

        # 规划/重规划
        self.replan_interval    = self.declare_parameter('replan_interval',    6.0).value
        self.map_replan_cooldown= self.declare_parameter('map_replan_cooldown',2.0).value
        self.obstacle_threshold = self.declare_parameter('obstacle_threshold', 50).value

        # 目标与角度滤波
        self.target_debounce    = self.declare_parameter('target_debounce',    0.10).value
        self.angle_lpf_alpha    = self.declare_parameter('angle_lpf_alpha',    0.2).value

        # 线段检查灵敏度
        self.segment_step       = self.declare_parameter('segment_step',       0.03).value
        self.line_check_margin  = self.declare_parameter('line_check_margin',  0.08).value
        self.line_check_min_hits= self.declare_parameter('line_check_min_hits',2).value
        self.line_check_max_len = self.declare_parameter('line_check_max_len', 0.6).value

        # 状态
        self.state = NavigationState.IDLE
        self.robot_xy = None
        self.target_xy = None

        self.map_plan = None
        self.map_nav = None
        self.map_res = None
        self.map_origin = (0.0, 0.0)
        self.planner: DualMapPlanner = None

        self.path = []
        self.path_idx = 0
        self.last_plan_time = 0.0

        self.yaw_est = YawEstimator()
        self.yaw = None
        self.angle_err_filt = 0.0

        # ROS I/O
        self.cmd_pub  = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path,  '/planned_path', 10)
        self.create_subscription(OccupancyGrid, '/map_planning',  self.map_plan_cb, 10)
        self.create_subscription(OccupancyGrid, '/map_navigation', self.map_nav_cb, 10)
        self.create_subscription(PointStamped, '/robot_position', self.robot_cb, 30)
        self.create_subscription(PointStamped, '/target_position', self.target_cb, 30)

        self.create_timer(0.05, self.control_loop)  # 20Hz
        self.get_logger().info('Dual-Map A* 导航已启动（规划厚图 / 控制薄图）')

    # ---------- 地图回调 ----------
    def _update_planner_if_ready(self):
        if self.map_plan is not None and self.map_nav is not None and self.map_res is not None:
            self.planner = DualMapPlanner(self.map_plan, self.map_nav, self.map_res,
                                          self.map_origin, obstacle_threshold=self.obstacle_threshold)

    def map_plan_cb(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        grid = np.array(msg.data, dtype=np.int16).reshape(h, w)
        grid = np.where(grid < 0, 100, grid)
        self.map_plan = grid
        self.map_res = float(msg.info.resolution)
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self._update_planner_if_ready()
        if self.state in (NavigationState.NAVIGATING, NavigationState.NO_PATH) and \
           time.time() - self.last_plan_time > float(self.map_replan_cooldown):
            self.state = NavigationState.PLANNING

    def map_nav_cb(self, msg: OccupancyGrid):
        w, h = msg.info.width, msg.info.height
        grid = np.array(msg.data, dtype=np.int16).reshape(h, w)
        grid = np.where(grid < 0, 100, grid)
        self.map_nav = grid
        self.map_res = float(msg.info.resolution)
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self._update_planner_if_ready()

    # ---------- 位置/目标 ----------
    def robot_cb(self, msg: PointStamped):
        self.robot_xy = (float(msg.point.x), float(msg.point.y))
        self.yaw = self.yaw_est.update(self.robot_xy[0], self.robot_xy[1], time.time())
        if self.state == NavigationState.IDLE and self.target_xy and self.planner:
            self.state = NavigationState.PLANNING

    def target_cb(self, msg: PointStamped):
        new_target = (round(float(msg.point.x), 3), round(float(msg.point.y), 3))
        if (self.target_xy is None or
            math.hypot(new_target[0] - (self.target_xy[0] if self.target_xy else 1e9),
                       new_target[1] - (self.target_xy[1] if self.target_xy else 1e9)) > float(self.target_debounce)):
            self.target_xy = new_target
            if self.planner and self.robot_xy:
                self.state = NavigationState.PLANNING

    # ---------- 规划 ----------
    def plan_path(self):
        if not (self.planner and self.robot_xy and self.target_xy):
            return False
        path = self.planner.plan(self.robot_xy, self.target_xy)
        if not path or len(path) < 2:
            self.get_logger().warn('❌ A* 未找到可行路径')
            self.path = []
            return False

        # 简单平滑（3点均值）
        if len(path) >= 3:
            sm = []
            for i in range(len(path)):
                xs, ys, c = 0.0, 0.0, 0
                for j in range(max(0, i-1), min(len(path), i+2)):
                    xs += path[j][0]; ys += path[j][1]; c += 1
                sm.append((xs/c, ys/c))
            path = sm

        self.path = path
        self.path_idx = 0
        self.last_plan_time = time.time()

        header = Header(); header.stamp = self.get_clock().now().to_msg(); header.frame_id = 'map'
        pm = Path(); pm.header = header
        for (x, y) in path:
            ps = PoseStamped(); ps.header = header
            ps.pose.position.x = x; ps.pose.position.y = y; ps.pose.orientation.w = 1.0
            pm.poses.append(ps)
        self.path_pub.publish(pm)
        self.get_logger().info(f' 规划成功：{len(path)} 个路点')
        return True

    # ---------- 控制回环 ----------
    def control_loop(self):
        cmd = Twist()

        if self.state == NavigationState.IDLE:
            pass

        elif self.state == NavigationState.PLANNING:
            ok = self.plan_path()
            self.state = NavigationState.NAVIGATING if ok else NavigationState.NO_PATH

        elif self.state == NavigationState.NO_PATH:
            pass

        elif self.state == NavigationState.REACHED:
            pass

        elif self.state == NavigationState.NAVIGATING:
            if not (self.robot_xy and self.path):
                self.state = NavigationState.PLANNING
            else:
                rx, ry = self.robot_xy
                gx, gy = self.path[-1]
                if math.hypot(gx - rx, gy - ry) < self.goal_tolerance:
                    self.state = NavigationState.REACHED
                    self.get_logger().info('已到达目标')
                else:
                    # 选择前瞻点 + 滞回
                    target_idx = self.path_idx
                    while target_idx < len(self.path) - 1:
                        tx_, ty_ = self.path[target_idx]
                        if math.hypot(tx_ - rx, ty_ - ry) >= self.lookahead_distance:
                            break
                        target_idx += 1
                    target_idx = min(target_idx, len(self.path) - 1)
                    cur_tx, cur_ty = self.path[self.path_idx]
                    if math.hypot(cur_tx - rx, cur_ty - ry) < 0.5 * float(self.lookahead_distance):
                        self.path_idx = max(self.path_idx, target_idx)
                    tx, ty = self.path[self.path_idx]

                    # 线段占用（用“导航图”）——不通过则尝试更近一点，否则重规划
                    if not self.planner.segment_is_free_world(
                        rx, ry, tx, ty,
                        step_m=float(self.segment_step),
                        margin_m=float(self.line_check_margin),
                        min_consecutive_hits=int(self.line_check_min_hits),
                        max_len_m=float(self.line_check_max_len)
                    ):
                        # 退一步试近点
                        near_idx = max(0, self.path_idx - 1)
                        ntx, nty = self.path[near_idx]
                        if self.planner.segment_is_free_world(
                            rx, ry, ntx, nty,
                            step_m=float(self.segment_step),
                            margin_m=float(self.line_check_margin),
                            min_consecutive_hits=int(self.line_check_min_hits),
                            max_len_m=float(self.line_check_max_len) * 0.6
                        ):
                            tx, ty = ntx, nty
                        else:
                            self.state = NavigationState.PLANNING
                            self.cmd_pub.publish(Twist())
                            return

                    # 控制量（视觉 yaw + 低通）
                    bearing = math.atan2(ty - ry, tx - rx)
                    yaw = self.yaw if self.yaw is not None else 0.0
                    angle_err = wrap_to_pi(bearing - yaw)
                    # 角误差低通
                    delta = wrap_to_pi(angle_err - self.angle_err_filt)
                    self.angle_err_filt = wrap_to_pi(self.angle_err_filt + float(self.angle_lpf_alpha) * delta)
                    angle_err = self.angle_err_filt

                    dist = math.hypot(tx - rx, ty - ry)
                    v_des = min(self.max_linear_speed, 0.9 * dist)
                    lin_scale = max(0.3, math.cos(min(abs(angle_err), 1.2)))
                    cmd.linear.x = max(self.min_crawl_speed, v_des * lin_scale)

                    kp_ang = 0.9
                    cmd.angular.z = max(-self.max_angular_speed,
                                        min(self.max_angular_speed,
                                            kp_ang * angle_err * (0.6 + 0.4 * (1.0 - lin_scale))))
                    # 定时重规划
                    if time.time() - self.last_plan_time > float(self.replan_interval):
                        self.state = NavigationState.PLANNING

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = NavigationController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
