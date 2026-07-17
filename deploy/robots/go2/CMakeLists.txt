cmake_minimum_required(VERSION 3.12)
project(go2_controller)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

set(UNITREE_SDK_ROOT "/opt/unitree_robotics" CACHE PATH "Path to the installed unitree_sdk2")

include_directories(
  /usr/include/eigen3
  ${UNITREE_SDK_ROOT}/include
  ${UNITREE_SDK_ROOT}/include/ddscxx # dds
  ${UNITREE_SDK_ROOT}/include/iceoryx/v2.0.2 # dds
  include
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_directories(${UNITREE_SDK_ROOT}/lib)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0
)

file(GLOB_RECURSE ADD_SRC_LIST
  src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_ctrl main.cpp)
add_executable(go2_lowstate_logger lowstate_logger.cpp)
