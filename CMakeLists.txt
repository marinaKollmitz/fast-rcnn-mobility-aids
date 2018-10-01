cmake_minimum_required(VERSION 2.6)
project(fast-rcnn-mobility-aids)

find_package(PCL 1.3 REQUIRED COMPONENTS common filters segmentation visualization)
find_package(OpenCV)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(generate_proposals src/generate_proposals.cpp)

target_link_libraries(generate_proposals ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
