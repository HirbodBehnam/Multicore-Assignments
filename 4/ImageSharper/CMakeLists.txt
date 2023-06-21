cmake_minimum_required(VERSION 3.25)
project(ImageSharper C)

set(CMAKE_C_STANDARD 11)

add_executable(ImageSharper main.c bmp.c)
target_link_libraries(ImageSharper m)
