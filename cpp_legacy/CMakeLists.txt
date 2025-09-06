cmake_minimum_required(VERSION 3.10)
project(mppi)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 3.4 REQUIRED)

include_directories(
        ${EIGEN3_INCLUDE_DIRS}
        ./include
)

add_executable(mppi src/main.cpp src/mppi.cpp include/mppi.h)