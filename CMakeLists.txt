# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.10)
project(point_pillars_ros)
find_package(CUDA REQUIRED)

# if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
#   set(CMAKE_C_COMPILER /usr/bin/gcc)
#   set(CMAKE_CXX_COMPILER /usr/bin/g++)
#   set(CUDA_INSTALL_TARGET_DIR targets/x86_64-linux)
# endif()

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/${CUDA_INSTALL_TARGET_DIR}/include)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-Wno-deprecated-declarations -O2")
add_compile_options(-W)
add_compile_options(-std=c++11)

set( SMS 30 32 35 37 50 52 53 60 61 62 70 72 75 87)
foreach(sm ${SMS})
	set(GENCODE ${GENCODE} -gencode arch=compute_${sm},code=sm_${sm})
endforeach()
set(HIGHEST_SM 87)
set(GENCODE ${GENCODE} -gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    -ccbin ${CMAKE_CXX_COMPILER}
    -Xcompiler -DWIN_INTERFACE_CUSTOM
    -Xlinker -lsocket
)

if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
  message("Using Debug Mode")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G --ptxas-options=-v)
endif()

SET(TENSORRT_PATH /home/xiaohu/3D/TensorRT-8.4.0.6/)
SET(TENSORRT_LIB_PATH /home/xiaohu/3D/TensorRT-8.4.0.6/lib)



find_package(catkin REQUIRED COMPONENTS
   roscpp
   pcl_ros
   autoware_msgs
)

find_package(PCL 1.8 REQUIRED)

catkin_package(
   CATKIN_DEPENDS
   roscpp
   pcl_ros
   autoware_msgs
)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    include
    ${TENSORRT_PATH}/include
)

file(GLOB_RECURSE SOURCE_FILES
    src/*.cu
    src/*.cpp
)

# 生成可执行文件,ros启动就不要改默认路径
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../) # 设置可执行目标文件的输出目录
cuda_add_executable(${PROJECT_NAME} main.cpp ${SOURCE_FILES})

file(GLOB TENSORRT_LIBS "${TENSORRT_LIB_PATH}/*.so")
target_link_libraries(${PROJECT_NAME}
    ${TENSORRT_LIBS}
    ${PCL_LIBRARIES}
    ${catkin_LIBRARIES}
)
