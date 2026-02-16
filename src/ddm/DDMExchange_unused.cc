#include "DDMExchange.hh"

using namespace NSPC_DDM;

/// @brief Synchronize node data across all domains using collective communication.
///
/// Each MPI rank corresponds to one domain. This method ensures that shared
/// (interface) nodes have consistent values across all domains by averaging.
///
/// The algorithm proceeds in three steps:
///   1. **Pack**: For the calling rank's domain, extract the values of its local
///      nodes from the globally-indexed @p data array into a contiguous send buffer,
///      ordered by local node index.
///   2. **Allgatherv**: Collect every domain's local node data into a single receive
///      buffer via MPI_Allgatherv. After this call, each rank holds the local node
///      values of all domains, with displacements indicating where each domain's
///      segment begins.
///   3. **Average**: For each local node, look up all domains that share it
///      (via DDMPartition::getDomains). For each such domain, retrieve the node's
///      value from the receive buffer using its local index within that domain
///      (via DDMPartition::getLocalNodeIndex). The node's value in @p data is then
///      replaced with the arithmetic mean across all owning domains.
///
/// Interior nodes (owned by exactly one domain) are unchanged. Interface nodes
/// (shared by multiple domains) converge toward the average of their domain-local
/// values.
///
/// @param comm       MPI communicator (rank == domain ID).
/// @param data       Node scalar data indexed by global node ID. Modified in place.
/// @param partition  The DDM partition describing node-to-domain relationships.
void DDMMPISynchronizer<RealNodeScalar>::fullSynchronize(
  MPI_Comm comm,
  RealNodeScalar & data,
  DDMPartition const & partition
) const {
  MPIHelper mpi(comm);
  int myDomain = mpi.rank();

  // Pack local data from global array into send buffer
  int myNodes = partition.getNodeCount(myDomain);
  std::vector<double> sendBuf(myNodes);
  for (int iLocal = 0; iLocal < myNodes; ++iLocal) {
    int globalNode = partition.getNode(myDomain, iLocal);
    sendBuf[iLocal] = data[globalNode];
  }

  // Gather local node data from all domains
  std::vector<double> recvBuf;
  std::vector<int> displs;
  mpi.allgatherv(sendBuf.data(), sendBuf.size(), recvBuf, displs);

  // Update each local node by averaging across all domains that own it
  for (int iLocal = 0; iLocal < myNodes; ++iLocal) {
    int globalNode = partition.getNode(myDomain, iLocal);
    auto const & domains = partition.getDomains(globalNode);
    double sum = 0.0;
    for (int dom : domains) {
      int localIdx = partition.getLocalNodeIndex(globalNode, dom);
      sum += recvBuf[displs[dom] + localIdx];
    }
    data[globalNode] = sum / domains.size();
  }
}


/// @brief Exchange node data with neighboring domains using point-to-point communication.
///
/// Unlike fullSynchronize, this method only communicates with adjacent domains
/// (those sharing at least one interface node), making it more efficient for
/// large domain counts.
///
/// The algorithm proceeds as follows:
///   1. **Post receives**: For each neighboring domain, post a non-blocking
///      MPI_Irecv into the corresponding segment of the receive buffer. The
///      DDMNeighbor::displacements array defines the offset and count for each
///      neighbor's data within the flat buffer.
///   2. **Pack and send**: Copy interface node values from @p inData (indexed by
///      local node ID within DDMNeighbor::nodes) into a send buffer, then post
///      non-blocking MPI_Isend to each neighbor using the same displacement layout.
///   3. **Wait and accumulate**: After all receives complete, accumulate the
///      received values into a per-local-node sum (outAgg) and count (outCount),
///      mapping each received value back to its local node index via
///      DDMNeighbor::nodes.
///   4. **Average**: For each local node, compute the final value as the average
///      of the node's own value (@p inData) and all received neighbor values:
///      @code
///        outData[i] = (inData[i] + outAgg[i]) / (outCount[i] + 1)
///      @endcode
///      The "+1" accounts for the node's own contribution. Interior nodes
///      (outCount == 0) simply copy inData unchanged.
///   5. **Wait for sends**: Ensure all sends have completed before returning.
///
/// @param comm       MPI communicator (rank == domain ID).
/// @param inData     Input node scalar data (local indexing), read-only.
/// @param outData    Output node scalar data (local indexing), written with averaged values.
/// @param partition  The DDM partition describing node-to-domain relationships.
/// @param neighbor   Precomputed neighbor structure for the calling rank's domain.
void DDMMPISynchronizer<RealNodeScalar>::neighborExchange(
  MPI_Comm comm,
  RealNodeScalar const & inData,
  RealNodeScalar & outData,
  DDMPartition const & partition,
  DDMNeighbor const & neighbor
) const {
  MPIHelper mpi(comm);

  std::vector<double> recvBuf(neighbor.nodes.size());
  std::vector<MPI_Request> recvReqs(neighbor.neighborDomain.size());
  // post receives
  for (int i = 0; i < neighbor.neighborDomain.size(); ++i) {
    int neighborID = neighbor.neighborDomain[i];
    int displacement = neighbor.displacements[i];
    int count = neighbor.displacements[i+1] - displacement;
    recvReqs[i] = mpi.irecv(&recvBuf[displacement], count, neighborID, 0);
  }

  std::vector<double> sendBuf(neighbor.nodes.size());
  for (int i = 0; i < neighbor.nodes.size(); ++i) {
    sendBuf[i] = inData[neighbor.nodes[i]];
  }
  std::vector<MPI_Request> sendReqs(neighbor.neighborDomain.size());
  for (int i = 0; i < neighbor.neighborDomain.size(); ++i) {
    int neighborID = neighbor.neighborDomain[i];
    int displacement = neighbor.displacements[i];
    int count = neighbor.displacements[i+1] - displacement;
    sendReqs[i] = mpi.isend(&sendBuf[displacement], count, neighborID, 0);
  }

  int myNodes = partition.getNodeCount(mpi.rank());
  std::vector<double> outAgg(myNodes, 0.0);
  std::vector<int> outCount(myNodes, 0);

  if (MPI_Waitall(recvReqs.size(), recvReqs.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Waitall failed");
  }
  for (size_t i = 0; i < recvBuf.size(); ++i) {
    outAgg[neighbor.nodes[i]] += recvBuf[i];
    ++outCount[neighbor.nodes[i]];
  }
  for (int i = 0; i < myNodes; ++i) {
    outData[i] = (inData[i] + outAgg[i]) / (outCount[i] + 1);
  }
  if (MPI_Waitall(sendReqs.size(), sendReqs.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Waitall failed");
  }
}