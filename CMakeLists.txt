
cmake_minimum_required(VERSION 2.8)
project(yolt)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/")

find_package(Caffe REQUIRED)
#find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS system filesystem regex REQUIRED)
find_package(OpenCV REQUIRED)

option(-std "std" c++11)

# COMPILE LIBS
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
message("Caffe_INCLUDE_DIRS is ${Caffe_INCLUDE_DIRS}")	

#include_directories(${CUDA_INCLUDE_DIRS})
#message("CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")

# ADD OUR OWN EXECUTABLE FILES
set(GLOG_LIB glog)

add_executable(yolt src/yolt.cpp)

target_link_libraries(yolt ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB}  ${OpenCV_LIBS})

