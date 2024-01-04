cmake_minimum_required(VERSION 3.8)
project(sensor_fusion)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)


# 如果你的opencv找不到，可以自己指定目錄
set(OpenCV_DIR   "/usr")


find_package(OpenCV REQUIRED)
find_package(autoware_cmake REQUIRED)
autoware_package()

###########
## Build ##
###########


include_directories(
  ${PROJECT_SOURCE_DIR}/include
)


set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -O0 -Wfatal-errors -pthread -w -g")


### plugins ###
ament_auto_add_library(sensor_fusion_nodelet SHARED
  src/sensor_fusion.cpp
)
target_link_libraries(sensor_fusion_nodelet stdc++ dl ${OpenCV_LIBS})
target_link_libraries(sensor_fusion_nodelet
  ${OpenCV_LIBS}
)

rclcpp_components_register_node(sensor_fusion_nodelet
  PLUGIN "object_recognition::SensorFusionNodelet"
  EXECUTABLE sensor_fusion_node
)
#############
## Install ##
#############
ament_auto_package(INSTALL_TO_SHARE
  result
  launch
)

install(
  TARGETS
    
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

