add_library(ddmpartition ddm/DDMPartition.cc)
target_include_directories(ddmpartition PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/util
  ${CMAKE_CURRENT_SOURCE_DIR}/ddm
)
target_link_libraries(ddmpartition PUBLIC MPI::MPI_CXX)

add_executable(DDMPartition_test ddm/DDMPartition_test.cc)
target_link_libraries(DDMPartition_test PRIVATE ddmpartition)
