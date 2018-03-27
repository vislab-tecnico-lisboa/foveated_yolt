
cmake_minimum_required(VERSION 2.8)
project(yolt)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/")

### Set libraries' locations
# Set Caffe
set(Caffe_DIR "/home/cristina/caffe/build")
set(Caffe_INCLUDE_DIRS "/home/cristina/caffe/distribute/include")
set(Caffe_LIBRARIES "/home/cristina/caffe/build/lib/libcaffe.so")

find_package(Caffe REQUIRED)
#find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS system filesystem regex REQUIRED)
find_package(OpenCV REQUIRED)

#set(CAFFE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/caffe)
#set(TRACKER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/tracker/GOTURN)
#set(DETECTOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/detector)

option(-std "std" c++11)

# COMPILE LIBS
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY

message("Caffe_INCLUDE_DIRS is ${Caffe_INCLUDE_DIRS}")	


#include_directories(${CUDA_INCLUDE_DIRS})
#message("CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")
#include(${DETECTOR_DIR}/CMakeLists.txt)
#include(${TRACKER_DIR}/CMakeLists.txt)

# ADD HEADERS DIRECTORIES SO OUR SOURCE FILES CAN FIND THEM
#include_directories(${TRACKER_DIR}/src)
#include_directories(${TRACKER_DIR}/src/native)

#SET(CMAKE_C_COMPILER /usr/bin/gcc)
#SET(CMAKE_CXX_COMPILER /usr/bin/g++)

# ADD OUR OWN EXECUTABLE FILES
set(GLOG_LIB glog)

#add_executable(main src/main.cpp)
add_executable(yolt src/yolt_exp.cpp)

target_link_libraries(yolt ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB}  ${OpenCV_LIBS})

