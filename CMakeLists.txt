
cmake_minimum_required(VERSION 3.19 )

project("demo")
# set(CMAKE_BUILD_TYPE Release)  
# set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++14 -Ofast")
set(CMAKE_BUILD_TYPE Debug)  # 没有这句话vscode无法单步调试
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++14 -O0")
# SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

# OpenCV
find_package(OpenCV 3.2 REQUIRED) # 版本设置
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIBS:${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS:${OpenCV_INCLUDE_DIRS}")

add_executable(monocalib 01_mono_calib.cc)
add_executable(calib_undistort 02_apply_mono_calib.cc) 
add_executable(calib_pro 03_apply_mono_calib_2.cc) 
add_executable(calib_headpose 04_apply_mono_calib_3.cc) 
add_executable(calib_dist 05_pnp_calc_distance.cc) 

target_link_libraries(calib_pro ${OpenCV_LIBS} pthread)
target_link_libraries(monocalib ${OpenCV_LIBS} pthread)
target_link_libraries(calib_undistort ${OpenCV_LIBS} pthread)
target_link_libraries(calib_undistort ${OpenCV_LIBS} pthread)
target_link_libraries(calib_headpose ${OpenCV_LIBS} pthread)
target_link_libraries(calib_dist ${OpenCV_LIBS} pthread)