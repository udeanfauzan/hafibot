cmake_minimum_required(VERSION 3.0.2)
project(fall_detection_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES fall_detection_ros
#  CATKIN_DEPENDS rospy sensor_msgs std_msgs cv_bridge
#  DEPENDS system_lib
)

catkin_install_python(PROGRAMS
  scripts/fall_detection_node.py
  scripts/fall_alarm_node.py
  scripts/human_following_node.py
  scripts/flask_stream_node.py
  scripts/camera_node.py
  scripts/telegram_notifier_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
