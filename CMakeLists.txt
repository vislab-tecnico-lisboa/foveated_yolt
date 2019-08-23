cmake_minimum_required(VERSION 2.8.3)
project(foveated_yolt)

set(ROS FALSE)
set(CLASSIFICATION 	FALSE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/")

SET(CMAKE_CXX_FLAGS_RELEASE "-Wall -O3 -Wno-deprecated-declarations -std=c++11")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")

# Include package config, boost-python
find_package(PkgConfig REQUIRED)
#include(cmake/boost-python.cmake)

#  add_definitions(-DOPENCV_3)

if(${CLASSIFICATION})
  include(cmake/Modules/FindCaffe.cmake)
  find_package(Caffe REQUIRED)
  include_directories(	${Caffe_INCLUDE_DIRS})
endif(${CLASSIFICATION})

find_package(Boost COMPONENTS system filesystem regex REQUIRED)
find_package(OpenCV 3 REQUIRED)
#find_package(Python COMPONENTS Interpreter Development)
#find_package(PythonInterp REQUIRED)
#find_package(NumPy REQUIRED)
#find_package(PythonLibs REQUIRED)

option(-std "std" c++11)

include_directories(
	include
	${PYTHON_INCLUDE_DIR}
	${PYTHON_NUMPY_INCLUDE_DIR}
)

set(GLOG_LIB glog)

if(${ROS})
  find_package(catkin REQUIRED COMPONENTS
    roscpp
    cv_bridge
    image_transport
    dynamic_reconfigure
    actionlib
    actionlib_msgs
  )

  generate_dynamic_reconfigure_options(
    cfg/fovea.cfg
  )

  add_action_files(
    DIRECTORY action
    FILES Eye.action Task.action
  )

  generate_messages(
    DEPENDENCIES actionlib_msgs
  )

  catkin_package(
     INCLUDE_DIRS include
     CATKIN_DEPENDS roscpp actionlib actionlib_msgs message_runtime
  )

  include_directories(
    include
    ${catkin_INCLUDE_DIRS}
  )

add_executable (foveation_ros src/ros/foveation_ros_node.cpp src/ros/foveation_ros.cpp src/laplacian_foveation.cpp)
target_link_libraries (foveation_ros ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB} ${OpenCV_LIBS} ${catkin_LIBRARIES})
add_dependencies(foveation_ros ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_EXPORTED_TARGETS)

add_executable (yolt_ros src/ros/yolt_ros_node.cpp src/ros/yolt_ros.cpp src/network_classes.cpp)
target_link_libraries (yolt_ros ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB} ${OpenCV_LIBS} ${catkin_LIBRARIES})
add_dependencies(yolt_ros ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_EXPORTED_TARGETS)

# FOVEATION PYTHON BINDINGS
endif(${ROS})

add_definitions(${Caffe_DEFINITIONS})

if(${CLASSIFICATION})
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
	${GLOG_LIB}  
	${OpenCV_LIBS}
  )
endif(${CLASSIFICATION})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#boost_python_module(np_opencv_converter src/python_bindings/np_opencv_converter.cpp src/python_bindings/conversion.cpp)
#target_link_libraries(np_opencv_converter boost_system boost_python ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

#boost_python_module(np_opencv_module src/python_bindings/np_opencv_module.cpp)
#target_link_libraries(np_opencv_module boost_system boost_python-py35 np_opencv_converter ${OpenCV_LIBS} -fPIC)

#boost_python_module(yolt_python src/python_bindings/yolt_python.cpp src/laplacian_foveation.cpp)
#target_link_libraries(yolt_python -lboost_python-py35 -lboost_system ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES} -fPIC)

#set_target_properties(yolt_python PROPERTIES SUFFIX .so)
#set_target_properties(yolt_python PROPERTIES PREFIX "")

add_subdirectory(pybind11)
#add_subdirectory(src/python_bindings/)

set(PYTHON_FILES
#	pyopenface.py
#	__init__.py
	smooth_foveation.cpp)

pybind11_add_module(pysmooth_foveation src/laplacian_foveation.cpp src/python_bindings/smooth_foveation.cpp src/python_bindings/conversions.cpp)
target_link_libraries(pysmooth_foveation PRIVATE pybind11::module ${OpenCV_LIBS})
set_target_properties(pysmooth_foveation PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}" SUFFIX "${PYTHON_MODULE_EXTENSION}") 
configure_file(src/python_bindings/__init__.py src/python_bindings/__init__.py)






