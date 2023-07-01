cmake_minimum_required(VERSION 3.25)
project(SobelEdgeDetectorCPU)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 17)

add_executable(SobelEdgeDetectorCPU main.cpp reader.cpp saver.cpp brighter.cpp sobel.cpp)
target_link_libraries(SobelEdgeDetectorCPU ${OpenCV_LIBS})