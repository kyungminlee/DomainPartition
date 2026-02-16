#pragma once

#include <mpi.h>
#include <vector>
#include "anyprint.hh"
#include "MPITraits.hh"

/// Exception thrown when an MPI call returns an error code.
class __attribute__((visibility("default"))) MPIError : public std::runtime_error {
public:
  MPIError(char const * name, int ierr) : std::runtime_error(name), _errorcode(ierr) {}
  ~MPIError() = default;
  int errorcode() const { return _errorcode; }
public:
  int _errorcode = MPI_SUCCESS;
};

/// Lightweight wrapper around an MPI communicator providing common
/// collective and point-to-point operations with error checking.
class MPIHelper {
public:
  /// Construct from an existing MPI communicator.
  /// Queries and caches the communicator's size and rank.
  /// @throws MPIError if MPI_Comm_size or MPI_Comm_rank fails.
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

  /// Gather variable-length data from all ranks into all ranks.
  ///
  /// First performs an MPI_Allgather to exchange per-rank send counts, then
  /// computes displacements and resizes @p recvbuf accordingly, and finally
  /// calls MPI_Allgatherv to collect the data.
  ///
  /// @tparam T1  Send buffer element type (must have an MPITypeTrait specialization).
  /// @tparam T2  Receive buffer element type (must have an MPITypeTrait specialization).
  /// @param sendbuf  Pointer to the local send buffer.
  /// @param count    Number of elements to send from this rank.
  /// @param[out] recvbuf  Resized to hold all gathered data from every rank.
  /// @param[out] displs   Resized to size()+1; displs[i] is the offset into
  ///                      recvbuf where rank i's data begins.
  /// @throws MPIError if any MPI call fails.
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
  
  /// Initiate a non-blocking send (MPI_Isend).
  ///
  /// @tparam T  Element type (must have an MPITypeTrait specialization).
  /// @param data   Pointer to the data to send.
  /// @param count  Number of elements to send.
  /// @param dest   Rank of the destination process.
  /// @param tag    Message tag.
  /// @return The MPI_Request handle for the pending send.
  /// @throws MPIError if MPI_Isend fails.
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

  /// Initiate a non-blocking receive (MPI_Irecv).
  ///
  /// @tparam T  Element type (must have an MPITypeTrait specialization).
  /// @param data    Pointer to the buffer where received data will be stored.
  /// @param count   Maximum number of elements to receive.
  /// @param source  Rank of the source process.
  /// @param tag     Message tag.
  /// @return The MPI_Request handle for the pending receive.
  /// @throws MPIError if MPI_Irecv fails.
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

  /// Print a message from each rank in rank order.
  ///
  /// Iterates through all ranks with a barrier between each, so that rank 0
  /// prints first, then rank 1, etc. Each line is prefixed with "[rank] ".
  template <typename ... Ts>
  void print(Ts && ... args) const {
    for (int i = 0; i < _size; i++) {
      if (_rank == i) {
        anyprint::print("[", _rank, "] ", std::forward<Ts>(args)...);
      }
      barrier();
    }
  }

  /// Execute a callable on each rank sequentially in rank order.
  ///
  /// Each rank calls @p func one at a time (rank 0 first, then rank 1, etc.),
  /// with a barrier between each invocation. Useful for serialized I/O or
  /// debugging output.
  template <typename T>
  void syncrun(T && func) const {
    for (int i = 0; i < _size; i++) {
      if (_rank == i) {
        func();
      }
      barrier();
    }
  }

  /// Synchronize all ranks in the communicator (MPI_Barrier).
  /// @throws MPIError if MPI_Barrier fails.
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
