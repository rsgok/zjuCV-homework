cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
project(mytrain)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES main.cpp)
add_executable(mytrain ${SOURCE_FILES})
target_link_libraries (mytrain ${OpenCV_LIBS})