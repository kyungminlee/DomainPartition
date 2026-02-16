#pragma once

#include "DataEntry.hh"
#include "DDMPartition.hh"
#include "MPIHelper.hh"


namespace NSPC_DDM {

template <typename T>
class DDMMPISynchronizer;

template <>
class DDMMPISynchronizer<RealNodeScalar> {
public:
  /// @brief Synchronize shared node data across all domains via MPI_Allgatherv.
  /// Replaces each interface node's value with the average across all owning domains.
  /// @param data  Node scalar data indexed by global node ID, modified in place.
  void fullSynchronize(
    MPI_Comm comm,
    RealNodeScalar & data,
    DDMPartition const & partition
  ) const;

  /// @brief Exchange shared node data with neighboring domains via non-blocking point-to-point.
  /// Averages each node's value with contributions from adjacent domains.
  /// @param inData   Input node data (local indexing), read-only.
  /// @param outData  Output node data (local indexing), written with averaged values.
  void neighborExchange(
    MPI_Comm comm,
    RealNodeScalar const & inData,
    RealNodeScalar & outData,
    DDMPartition const & partition,
    DDMNeighbor const & neighbor
  ) const;
};

} // namespace NSPC_DDM