cmake_minimum_required(VERSION 3.16)
project(go1_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ONNXRUNTIME_ROOT "/home/chenyog/data/unitree_rl_mjlab/deploy/thirdparty/onnxruntime-linux-x64-1.22.0" CACHE PATH "ONNX Runtime root directory")

find_package(Threads REQUIRED)
find_package(yaml-cpp REQUIRED)

find_library(ONNXRUNTIME_LIB
  NAMES onnxruntime libonnxruntime.so.1.22.0
  PATHS "${ONNXRUNTIME_ROOT}/lib"
  NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_LIB)
  message(FATAL_ERROR "Could not find ONNX Runtime library under ${ONNXRUNTIME_ROOT}/lib")
endif()

add_executable(go1_deploy
  src/main.cpp
  src/go1_controller.cpp
  src/onnx_policy.cpp
)

target_include_directories(go1_deploy PRIVATE
  include
  "${ONNXRUNTIME_ROOT}/include"
)

target_link_libraries(go1_deploy PRIVATE
  Threads::Threads
  yaml-cpp
  "${ONNXRUNTIME_LIB}"
)

set_target_properties(go1_deploy PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
