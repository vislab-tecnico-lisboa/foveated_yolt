cmake_minimum_required(VERSION 2.8)
project(foveated_yolt)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/")


find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
)

SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -std=c++11")
# Include package config, boost-python
#find_package(PkgConfig REQUIRED)
#include(cmake/boost-python.cmake)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
)

find_package(Caffe REQUIRED)
#  find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS system filesystem regex python REQUIRED)
find_package(OpenCV REQUIRED)

option(-std "std" c++11)

include_directories(
	include
	${Caffe_INCLUDE_DIRS}
	${PYTHON_INCLUDE_DIR}
	${PYTHON_NUMPY_INCLUDE_DIR}
  	${catkin_INCLUDE_DIRS}
)

add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
add_definitions(-DCPU_ONLY)

# ADD OUR OWN EXECUTABLE FILES
set(GLOG_LIB glog)

add_executable(
	yolt 
	src/yolt.cpp
	src/laplacian_foveation.cpp
	src/network_classes.cpp
)

target_link_libraries(
	yolt
	${Boost_LIBRARIES} 
	${Caffe_LIBRARIES} 
	${PYTHON_LIBRARIES}
	${GLOG_LIB}  
	${OpenCV_LIBS}
)

add_executable(
	yolt_ros
	src/ros/yolt_ros.cpp
	src/laplacian_foveation.cpp
	src/network_classes.cpp
)

target_link_libraries(
	yolt_ros
	${Boost_LIBRARIES} 
	${Caffe_LIBRARIES} 
	${PYTHON_LIBRARIES}
	${GLOG_LIB}  
	${OpenCV_LIBS}
  	${catkin_LIBRARIES}
)



# FOVEATION PYTHON BINDINGS

# boost_python_module(np_opencv_converter src/python_bindings/np_opencv_converter.cpp src/python_bindings/conversion.cpp)
# target_link_libraries(np_opencv_converter boost_system boost_python ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

# include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# boost_python_module(np_opencv_module src/python_bindings/np_opencv_module.cpp)
# target_link_libraries(np_opencv_module boost_system boost_python np_opencv_converter ${OpenCV_LIBS})

# add_library(yolt_python SHARED src/python_bindings/yolt_python.cpp)
# starget_link_libraries(yolt_python -lboost_python -lboost_system ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

#  set_target_properties(yolt_python PROPERTIES SUFFIX .so)
#  set_target_properties(yolt_python PROPERTIES PREFIX "")



