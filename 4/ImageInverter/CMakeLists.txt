cmake_minimum_required(VERSION 3.25)
project(ImageInverter CUDA)
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CUDA_STANDARD 17)

add_executable(ImageInverter main.cu)

set_target_properties(ImageInverter PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ImageInverter ${OpenCV_LIBS})