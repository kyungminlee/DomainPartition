#pragma once

#include <mpi.h>
#include <vector>
#include "anyprint.hh"
#include "MPITraits.hh"

class __attribute__((visibility("default"))) MPIError : public std::runtime_error {
public:
  MPIError(char const * name, int ierr) : std::runtime_error(name), _errorcode(ierr) {}
  ~MPIError() = default;
  int errorcode() const { return _errorcode; }
public:
  int _errorcode = MPI_SUCCESS;
};

class MPIHelper {
public:
  MPIHelper(MPI_Comm comm) : _comm(comm) {
    int ierr;
    ierr = MPI_Comm_size(comm, &_size);
    if (ierr != MPI_SUCCESS) { throw MPIError("MPI_Comm_size failed", ierr); }
    ierr = MPI_Comm_rank(comm, &_rank);
    if (ierr != MPI_SUCCESS) { throw MPIError("MPI_Comm_rank failed", ierr); }
  }
  MPIHelper(MPIHelper &&) = default;
  MPIHelper(MPIHelper const &) = default;
  MPIHelper & operator=(MPIHelper &&) = default;
  MPIHelper & operator=(MPIHelper const &) = default;

  template <typename T1, typename T2, typename A2, typename A3>
  void allgatherv(
    T1 const * sendbuf,
    std::size_t count,
    std::vector<T2, A2> & recvbuf,
    std::vector<int, A3> & displs
  ) const {
    int sendcount = static_cast<int>(count);
    std::vector<int> recvcounts(_size);
    int ierr = MPI_Allgather(&sendcount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, _comm);
    if (ierr != MPI_SUCCESS) { throw MPIError("MPI_Allgather failed", ierr); }
    displs.resize(_size+1);
    displs[0] = 0;
    for (int i = 1; i <= _size; ++i) {
      displs[i] = displs[i-1] + recvcounts[i-1];
    }
    recvbuf.resize(displs.back());
    ierr = MPI_Allgatherv(sendbuf, sendcount, MPITypeTrait<T1>::datatype(),
                      recvbuf.data(), recvcounts.data(), displs.data(), MPITypeTrait<T2>::datatype(),
                      _comm);
    if (ierr != MPI_SUCCESS) { throw MPIError("MPI_Allgatherv failed", ierr); }
  }
  
  template <typename T>
  MPI_Request isend(T const * data, std::size_t count, int dest, int tag) const {
    MPI_Request request;
    MPI_Datatype datatype = MPITypeTrait<T>::datatype();
    int ierr = MPI_Isend(
        static_cast<void const *>(data), static_cast<int>(count), datatype,
        dest, tag, _comm, &request);
    if (ierr != MPI_SUCCESS) { throw MPIError("MPI_Send failed", ierr); }
    return request;
  }

  template <typename T>
  MPI_Request irecv(T * data, std::size_t count, int source, int tag) const {
    MPI_Request request;
    MPI_Datatype datatype = MPITypeTrait<T>::datatype();
    int ierr = MPI_Irecv(
        static_cast<void *>(data), static_cast<int>(count), datatype,
        source, tag, _comm, &request);
    if (ierr != MPI_SUCCESS) { throw MPIError("MPI_Recv failed", ierr); }
    return request;
  }

  MPI_Comm comm() const { return _comm; }
  int rank() const { return _rank; }
  int size() const { return _size; }

  template <typename ... Ts>
  void print(Ts && ... args) const {
    for (int i = 0; i < _size; i++) {
      if (_rank == i) {
        anyprint::print("[", _rank, "] ", std::forward<Ts>(args)...);
      }
      barrier();
    }
  }

  template <typename T>
  void syncrun(T && func) const {
    for (int i = 0; i < _size; i++) {
      if (_rank == i) {
        func();
      }
      barrier();
    }
  }

  void barrier() const {
    int ierr = MPI_Barrier(_comm);
    if (ierr != MPI_SUCCESS) {
      throw MPIError("MPI_Barrier failed", ierr);
    }
  }
private:
  MPI_Comm _comm = MPI_COMM_NULL;
  int _rank = -1;
  int _size = -1;
};
