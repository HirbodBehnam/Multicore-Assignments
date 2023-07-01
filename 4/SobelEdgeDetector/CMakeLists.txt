cmake_minimum_required(VERSION 3.25)
project(SobelEdgeDetector CUDA)
find_package(OpenCV REQUIRED)
set(CMAKE_CUDA_STANDARD 17)
#set(CUDA_NVCC_FLAGS "-g -G")

add_executable(SobelEdgeDetector main.cu reader.cu brighter.cu saver.cu sobel.cu)

set_target_properties(SobelEdgeDetector PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(SobelEdgeDetector ${OpenCV_LIBS})