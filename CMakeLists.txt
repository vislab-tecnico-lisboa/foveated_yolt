cmake_minimum_required(VERSION 2.8.3)
project(foveated_yolt)

set(ROS FALSE)
set(CLASSIFICATION TRUE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/")

# Include package config, boost-python
find_package(PkgConfig REQUIRED)
include(cmake/boost-python.cmake)
include(cmake/Modules/FindCaffe.cmake)
#  add_definitions(-DOPENCV_3)

if(${CLASSIFICATION})
  find_package(Caffe REQUIRED)
endif(${CLASSIFICATION})

find_package(Boost COMPONENTS system filesystem regex python REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(NumPy REQUIRED)
find_package(PythonLibs REQUIRED)

option(-std "std" c++11)

include_directories(
	include
	${Caffe_INCLUDE_DIRS}
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

add_executable (foveation_ros src/ros/foveation_ros.cpp src/laplacian_foveation.cpp)
target_link_libraries (foveation_ros ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB} ${OpenCV_LIBS} ${catkin_LIBRARIES})
add_dependencies(foveation_ros ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_EXPORTED_TARGETS)

add_executable (yolt_ros src/ros/yolt_ros.cpp src/network_classes.cpp)
target_link_libraries (yolt_ros ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB} ${OpenCV_LIBS} ${catkin_LIBRARIES})
add_dependencies(yolt_ros ${PROJECT_NAME}_gencfg ${PROJECT_NAME}_EXPORTED_TARGETS)

# FOVEATION PYTHON BINDINGS
endif(${ROS})

add_definitions(${Caffe_DEFINITIONS})

# ADD OUR OWN EXECUTABLE FILES


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


boost_python_module(np_opencv_converter src/python_bindings/np_opencv_converter.cpp src/python_bindings/conversion.cpp)
target_link_libraries(np_opencv_converter boost_system boost_python ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
boost_python_module(np_opencv_module src/python_bindings/np_opencv_module.cpp)
target_link_libraries(np_opencv_module boost_system boost_python np_opencv_converter ${OpenCV_LIBS})

boost_python_module(yolt_python src/python_bindings/yolt_python.cpp src/laplacian_foveation.cpp)
target_link_libraries(yolt_python -lboost_python -lboost_system ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

set_target_properties(yolt_python PROPERTIES SUFFIX .so)
set_target_properties(yolt_python PROPERTIES PREFIX "")




