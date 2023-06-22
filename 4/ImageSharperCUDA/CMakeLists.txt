cmake_minimum_required(VERSION 3.25)
project(ImageSharperCUDA CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(ImageSharperCUDA main.cu bmp.cu)

set_target_properties(ImageSharperCUDA PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
