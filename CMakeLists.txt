cmake_minimum_required(VERSION 3.11)

project(DDMPartition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MPI REQUIRED)

add_subdirectory(src)
add_subdirectory(test)