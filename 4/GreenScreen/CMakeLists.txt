cmake_minimum_required(VERSION 3.25)
project(GreenScreen C)

set(CMAKE_C_STANDARD 11)

add_executable(GreenScreen main.c)
target_link_libraries(GreenScreen m)