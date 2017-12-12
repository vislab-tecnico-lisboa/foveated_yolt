cmake_minimum_required(VERSION 2.8)
project(yolt)
#find_package(CUDA 8.0 REQUIRED)
#find_package(CAFFE REQUIRED )
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules/")


# Include package config, boost-python
find_package(PkgConfig REQUIRED)
#include(cmake/boost-python.cmake)



#find_package(Caffe REQUIRED)
find_package(CUDA REQUIRED)
find_package(Boost COMPONENTS system filesystem regex python REQUIRED)
find_package( OpenCV 2.4.9 REQUIRED )
  find_package(PythonInterp REQUIRED)
  find_package(PythonLibs REQUIRED)
#include_directories(${Caffe_INCLUDE_DIRS})
#add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
#message("Caffe_INCLUDE_DIRS is ${Caffe_INCLUDE_DIRS}")	

#set(CAFFE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/caffe)
#set(TRACKER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/tracker/GOTURN)
#set(DETECTOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/detector)


option(-std "std" c++11)

# COMPILE LIBS
#include_directories(${Caffe_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
message("CUDA_INCLUDE_DIRS is ${CUDA_INCLUDE_DIRS}")
link_directories(${Boost_LIBRARY_DIR})


# ADD HEADERS DIRECTORIES SO OUR SOURCE FILES CAN FIND THEM
#include_directories(${TRACKER_DIR}/src)
#include_directories(${TRACKER_DIR}/src/native)



#SET(CMAKE_C_COMPILER /usr/bin/gcc)
#SET(CMAKE_CXX_COMPILER /usr/bin/g++)

#add_executable(show_detector src/show_detector.cpp)
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Caffe_LIBRARIES} ${Boost_LIBRARIES} ${GLOG_LIB})
#target_link_libraries(show_detector ${PROJECT_NAME} pedestrian_detector)

# ADD OUR OWN EXECUTABLE FILES
set(GLOG_LIB glog)

#add_executable(main src/main.cpp)
#add_executable(yolt src/yolt.cpp)

#add_executable(box src/volume_box.cpp)
#add_executable(derived src/derived_classes.cpp)
#add_executable(multiple src/multiple_derived_classes.cpp)
#target_link_libraries(main  ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB}  ${OpenCV_LIBS}) 
#target_link_libraries(yolt ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB}  ${OpenCV_LIBS})


add_library(np_opencv_converter SHARED src/np_opencv_converter.cpp src/utils/conversion.cpp)
#boost_python_module(np_opencv_converter src/np_opencv_converter.cpp src/utils/conversion.cpp)
target_link_libraries(np_opencv_converter boost_system boost_python ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})



#add_library(np_opencv_converter SHARED src/np_opencv_converter.cpp src/utils/conversion.cpp)
#target_link_libraries(np_opencv_converter -lboost_python -lboost_system ${OpenCV_LIBS} ${PYTHON_LIBRARIES})
#set_target_properties(np_opencv_converter PROPERTIES SUFFIX .so)
#set_target_properties(np_opencv_converter PROPERTIES PREFIX "")

add_library(yolt_python SHARED src/yolt_python.cpp)
target_link_libraries(yolt_python -lboost_python -lboost_system ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

set_target_properties(yolt_python PROPERTIES SUFFIX .so)
set_target_properties(yolt_python PROPERTIES PREFIX "")

#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
#target_link_libraries (show_tracker_mot ${PROJECT_NAME})

#add_executable(test_selective_search src/test_selective_search.cpp)
