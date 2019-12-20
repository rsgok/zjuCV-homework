cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
project(part2)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES example_19-01.cpp)
add_executable(example_19-01 ${SOURCE_FILES})
target_link_libraries (example_19-01 ${OpenCV_LIBS})