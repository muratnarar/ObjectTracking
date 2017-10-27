cmake_minimum_required(VERSION 2.8)
project(ObjectTracking)
set (CMAKE_CXX_STANDARD 11)

find_package( OpenCV REQUIRED )
FIND_PACKAGE( Boost 1.40 COMPONENTS system filesystem REQUIRED )
find_package(dlib REQUIRED)

#Bring the headers into the project
include_directories(include)
#include_directories(/usr/local/dlib-19.7/dlib)
#include(/usr/local/dlib-19.7/dlib)

#Can manually add the sources using the set command as follows:
set(SOURCES src/main.cpp src/ModifiedTrackerMedianFlow.cpp include/ModifiedTrackerMedianFlow.hpp)

#However, the file(GLOB...) allows for wildcard additions:
#file(GLOB SOURCES "src/*.cpp")

add_executable(ObjectTracking ${SOURCES})
target_link_libraries( ObjectTracking ${OpenCV_LIBS} ${Boost_LIBRARIES} ${dlib_LIBS})
