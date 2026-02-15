#include <mpi.h>
#include <stdexcept>
#include "MPIHelper.hh"
#include "DDMPartition.hh"
#include <random>

using namespace anyprint;

DDMPartition makeTestPartition(int nDomain) {
  int nNode = 4*nDomain;
  DDMPartition partition(nNode, nDomain);

  for (int i = 0; i < nDomain; ++i) {
    std::vector<int> nodes;
    for (int j = 0; j < 4; ++j) {
      nodes.push_back(4 * i + j);
    }
    if (i > 0) {
      nodes.push_back(4 * i - 1);
    }
    if (i < nDomain - 1) {
      nodes.push_back(4 * i + 4);
    }
    partition.setNodesOfDomain(i, nodes);
  }
  partition.normalize();
  return partition;
}



template <typename Out>
void exchange(
  double const * inData,
  Out && outData,
  DDMPartition const & partition,
  DDMNeighbor const & neighbor,
  MPI_Comm comm
) {
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

  int myNodes = partition.getNodeCount(mpi.getRank());
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
    // outData[i] = (outData[i] + outAgg[i]) / (outCount[i] + 1);
    if (outCount[i] > 0) {
      outData[i] = outAgg[i] / outCount[i];
    } else {
      print("[", mpi.getRank(), "] why is outCount[", i, "] = ", outCount[i], " ?");
    }
  }
  if (MPI_Waitall(sendReqs.size(), sendReqs.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Waitall failed");
  }
}



void test_mpi(MPI_Comm comm) {
  MPIHelper mpi(comm);  

  DDMPartition partition = makeTestPartition(mpi.getSize());
  DDMNeighbor neighbor = getDDMNeighbor(mpi.getRank(), partition);

  mpi.syncrun([&]() {
    print("- rank: ", mpi.getRank());
    print("  partition:");
    dump(partition, indentation(4));
    print("  neighbor:");
    dump(neighbor, indentation(4));
  });

  int nNode = partition.getNodeCount();
  int nDomain = partition.getDomainCount();

  std::vector<double> globalData(nNode);

  // std::mt19937 rng(0);
  // std::uniform_real_distribution<double> dist(0.0, 1.0);
  for (int i = 0; i < nNode; ++i) {
    // globalData[i] = dist(rng);
    globalData[i] = i;
  }
  std::vector<double> localData(partition.getNodeCount(mpi.getRank()));
  for (int i = 0; i < localData.size(); ++i) {
    localData[i] = globalData[partition.getNode(mpi.getRank(), i)] + 100 * (mpi.getRank() + 1);
  }

  mpi.barrier();

  mpi.print("Starting exchange");

  exchange(localData.data(), localData.data(), neighbor, comm);

  mpi.barrier();
  mpi.print("Finished exchange");
  
  mpi.print("local data", localData);



}



int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  try {
    test_mpi(MPI_COMM_WORLD);
  } catch(...) {
    MPI_Abort(MPI_COMM_WORLD, -1);
  }


  MPI_Finalize();
}