#!/bin/bash
echo "清理Gazebo和ROS2进程..."
pkill -f gazebo
pkill -f gzserver  
pkill -f gzclient
pkill -f ros2
pkill -f color_tracker
sleep 2
echo "清理完成！"
