cmake_minimum_required(VERSION 3.8)
project(harvesting_robot_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(kdl_parser REQUIRED)
find_package(orocos_kdl REQUIRED)

# This package depends on a custom OMPL workspace that provides HyRRT
# and HybridStateSpace extensions. Make sure the OMPL workspace has been
# built and sourced before compiling this package.
find_package(ompl REQUIRED)

set(CUSTOM_OMPL_INCLUDE_SRC /home/francisco/workspace/ompl_ws/src/ompl/src)
set(CUSTOM_OMPL_INCLUDE_INSTALL /home/francisco/workspace/ompl_ws/install/ompl/include)
set(CUSTOM_OMPL_LIBRARY /home/francisco/workspace/ompl_ws/install/ompl/lib/libompl.so)

add_executable(mode2_trajectory_node
  src/mode2_trajectory_node.cpp
)

ament_target_dependencies(mode2_trajectory_node
  ament_index_cpp
  rclcpp
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  visualization_msgs
  kdl_parser
  orocos_kdl
)

# Keep these include directories for compatibility with the current
# custom OMPL workspace layout used by this project.
target_include_directories(mode2_trajectory_node PRIVATE
  ${CUSTOM_OMPL_INCLUDE_SRC}
  ${CUSTOM_OMPL_INCLUDE_INSTALL}
)

# Force linking against the custom OMPL library that contains
# HyRRT and HybridStateSpace, instead of the system ROS OMPL.
target_link_libraries(mode2_trajectory_node
  ${CUSTOM_OMPL_LIBRARY}
  orocos-kdl
)

install(TARGETS mode2_trajectory_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()