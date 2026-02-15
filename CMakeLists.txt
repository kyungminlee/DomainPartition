cmake_minimum_required(VERSION 3.11)

project(DDMPartition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(ddmpartition src/DDMPartition.cc)
target_include_directories(ddmpartition PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_executable(DDMPartition_test src/DDMPartition_test.cc)
target_link_libraries(DDMPartition_test PRIVATE ddmpartition)