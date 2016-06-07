cmake_minimum_required(VERSION 2.8 FATAL_ERROR) 
project(pcl_ground_based_rgbd_people_detection)

find_package(OpenCV REQUIRED)  
find_package(PCL 1.7 REQUIRED) 
include_directories(${PCL_INCLUDE_DIRS}) 
link_directories(${PCL_LIBRARY_DIRS}) 
add_definitions(${PCL_DEFINITIONS}) 

add_executable(people_detection src/main.cpp) 
target_link_libraries(people_detection ${PCL_LIBRARIES} ${OpenCV_LIBS})
