cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
project(zjuCV-hw1)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES main.cpp)
add_executable(zjuCV-hw1 ${SOURCE_FILES})
target_link_libraries (zjuCV-hw1 ${OpenCV_LIBS})