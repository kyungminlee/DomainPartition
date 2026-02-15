find_package(MPI REQUIRED)


add_executable(mpitest mpitest.cc)
target_link_libraries(mpitest PRIVATE ddmpartition MPI::MPI_CXX)