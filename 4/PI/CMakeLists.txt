cmake_minimum_required(VERSION 3.25)
project(PI CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(PI main.cu)

set_target_properties(PI PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
