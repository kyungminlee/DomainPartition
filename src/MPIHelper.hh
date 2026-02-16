#pragma once

#include <mpi.h>
#include <vector>
#include "anyprint.hh"

template <typename T>
struct MPITypeTrait;

#define DEFINE_MPI_TYPE_TRAIT(CPP_TYPE, MPI_TYPE) \
template <> \
struct MPITypeTrait<CPP_TYPE> { \
  static MPI_Datatype datatype() { return MPI_TYPE; } \
}

DEFINE_MPI_TYPE_TRAIT(double, MPI_DOUBLE);
DEFINE_MPI_TYPE_TRAIT(int, MPI_INT);
DEFINE_MPI_TYPE_TRAIT(unsigned int, MPI_UNSIGNED);
DEFINE_MPI_TYPE_TRAIT(char, MPI_CHAR);
DEFINE_MPI_TYPE_TRAIT(signed char, MPI_SIGNED_CHAR);
DEFINE_MPI_TYPE_TRAIT(unsigned char, MPI_UNSIGNED_CHAR);
DEFINE_MPI_TYPE_TRAIT(short, MPI_SHORT);
DEFINE_MPI_TYPE_TRAIT(unsigned short, MPI_UNSIGNED_SHORT);
DEFINE_MPI_TYPE_TRAIT(long, MPI_LONG);
DEFINE_MPI_TYPE_TRAIT(unsigned long, MPI_UNSIGNED_LONG);
DEFINE_MPI_TYPE_TRAIT(long long, MPI_LONG_LONG);
DEFINE_MPI_TYPE_TRAIT(unsigned long long, MPI_UNSIGNED_LONG_LONG);
DEFINE_MPI_TYPE_TRAIT(float, MPI_FLOAT);
DEFINE_MPI_TYPE_TRAIT(long double, MPI_LONG_DOUBLE);
DEFINE_MPI_TYPE_TRAIT(bool, MPI_CXX_BOOL);

#undef DEFINE_MPI_TYPE_TRAIT

class MPIError : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class MPIHelper {
public:
  MPIHelper(MPI_Comm comm)
  : _comm(comm) {

    if (MPI_Comm_size(comm, &_size) != MPI_SUCCESS) {
      throw MPIError("MPI_Comm_size failed");
    }
    if (MPI_Comm_rank(comm, &_rank) != MPI_SUCCESS) {
      throw MPIError("MPI_Comm_rank failed");
    }
  }
  
  template <typename T1, typename T2, typename A2, typename A3>
  void allgatherv(T1 const * data, std::size_t count, std::vector<T2, A2> & recvbuf, std::vector<int, A3> & displs) const;
  
  template <typename T>
  MPI_Request isend(T const * data, std::size_t count, int dest, int tag) const;

  template <typename T>
  MPI_Request irecv(T * data, std::size_t count, int source, int tag) const;

  MPI_Comm getComm() const { return _comm; }
  int getRank() const { return _rank; }
  int getSize() const { return _size; }


  template <typename ... Ts>
  void print(Ts && ... args) const {
    for (int i = 0; i < _size; i++) {
      if (_rank == i) {
        anyprint::print("[", _rank, "] ", std::forward<Ts>(args)...);
      }
      MPI_Barrier(MPI_COMM_WORLD); 
    }
  }

  template <typename T>
  void syncrun(T && func) const {
    for (int i = 0; i < _size; i++) {
      if (_rank == i) {
        func();
      }
      MPI_Barrier(MPI_COMM_WORLD); 
    }
  }

  void barrier() const {
    MPI_Barrier(_comm);
  }

private:

  MPI_Comm _comm = MPI_COMM_NULL;
  int _rank = -1;
  int _size = -1;
};


template <typename T1, typename T2, typename A2, typename A3>
void MPIHelper::allgatherv(T1 const * data, std::size_t count, std::vector<T2, A2> & recvbuf, std::vector<int, A3> & displs) const {
  int sendcount = static_cast<int>(count);
  std::vector<int> recvcounts(_size);
  if (MPI_Allgather(&sendcount, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, _comm) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Allgather failed");
  }

  displs.resize(_size+1);
  displs[0] = 0;
  for (int i = 1; i <= _size; ++i) {
    displs[i] = displs[i-1] + recvcounts[i-1];
  }

  recvbuf.resize(displs.back());

  if (MPI_Allgatherv(data, sendcount, MPITypeTrait<T1>::datatype(),
                     recvbuf.data(), recvcounts.data(), displs.data(), MPITypeTrait<T2>::datatype(),
                     _comm) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Allgatherv failed");
  }
}


template <typename T>
MPI_Request MPIHelper::isend(T const * data, std::size_t count, int dest, int tag) const {
  MPI_Request request;
  if (MPI_Isend(
      static_cast<void const *>(data),
      static_cast<int>(count),
      MPITypeTrait<T>::datatype(),
      dest,
      tag,
      _comm,
      &request) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Send failed");
  }
  return request;
}


template <typename T>
MPI_Request MPIHelper::irecv(T * data, std::size_t count, int source, int tag) const {
  MPI_Request request;
  if (MPI_Irecv(
    static_cast<void *>(data), 
    static_cast<int>(count), 
    MPITypeTrait<T>::datatype(),
    source,
    tag,
    _comm,
    &request) != MPI_SUCCESS
  ) {
    throw std::runtime_error("MPI_Recv failed");
  }
  return request;
}