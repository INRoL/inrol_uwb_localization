cmake_minimum_required(VERSION 3.10)
project(inrol-viu-calibration)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
set(EIGEN_INCLUDE_DIR ${EIGEN3_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(yaml-cpp REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(glog REQUIRED libglog)
include_directories(${glog_INCLUDE_DIRS})

message(STATUS "======================================")
message(STATUS "Found Eigen headers in: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Found Ceres headers in: ${CERES_INCLUDE_DIR}")
message(STATUS "Found Ceres libraries as: ${CERES_LIBRARIES}")
message(STATUS "Found glog libraries as: ${glog_LIBRARIES}")
message(STATUS "Found yaml-cpp libraries as: yaml-cpp")
message(STATUS "======================================")

set(common_include_directory ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(app)
add_subdirectory(libinroluwb)
