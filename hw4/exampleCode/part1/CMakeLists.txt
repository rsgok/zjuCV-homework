cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
project(part1)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(SOURCE_FILES example_18-01_from_disk.cpp)
add_executable(example_18-01_from_disk ${SOURCE_FILES})
target_link_libraries (example_18-01_from_disk ${OpenCV_LIBS})