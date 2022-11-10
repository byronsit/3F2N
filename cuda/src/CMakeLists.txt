cmake_minimum_required(VERSION 3.8)

SET(CMAKE_C_COMPILER "/usr/bin/gcc-6")
SET(CMAKE_CXX_COMPILER "/usr/bin/g++-6")

add_definitions(-D_USE_MATH_DEFINES)
find_package(OpenCV REQUIRED )
find_package(CUDA)
include(FindCUDA)

cuda_add_executable(csne_cuda main.cu stdafx.h)
target_link_libraries(csne_cuda ${OpenCV_LIBS} )

set(CUDA_NVCC_FLAGS; -arch=sm_21 )#cmake






