#include <mpi.h>
#include <stdexcept>
#include "MPIHelper.hh"
#include "DDMPartition.hh"
#include "DDMExchange.hh"
#include <random>
#include <thread>
#include "Expr.hh"

using namespace anyprint;
using namespace NSPC_DDM;

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



// template <typename Out>
// void exchange(
//   double const * inData,
//   Out && outData,
//   DDMPartition const & partition,
//   DDMNeighbor const & neighbor,
//   MPI_Comm comm
// ) {
//   MPIHelper mpi(comm);
  
//   std::vector<double> recvBuf(neighbor.nodes.size());
//   std::vector<MPI_Request> recvReqs(neighbor.neighborDomain.size());
//   // post receives
//   for (int i = 0; i < neighbor.neighborDomain.size(); ++i) {
//     int neighborID = neighbor.neighborDomain[i];
//     int displacement = neighbor.displacements[i];
//     int count = neighbor.displacements[i+1] - displacement;
//     recvReqs[i] = mpi.irecv(&recvBuf[displacement], count, neighborID, 0);
//   }

//   std::vector<double> sendBuf(neighbor.nodes.size());
//   for (int i = 0; i < neighbor.nodes.size(); ++i) {
//     sendBuf[i] = inData[neighbor.nodes[i]];
//   }
//   std::vector<MPI_Request> sendReqs(neighbor.neighborDomain.size());
//   for (int i = 0; i < neighbor.neighborDomain.size(); ++i) {
//     int neighborID = neighbor.neighborDomain[i];
//     int displacement = neighbor.displacements[i];
//     int count = neighbor.displacements[i+1] - displacement;
//     sendReqs[i] = mpi.isend(&sendBuf[displacement], count, neighborID, 0);
//   }

//   int myNodes = partition.getNodeCount(mpi.rank());
//   std::vector<double> outAgg(myNodes, 0.0);
//   std::vector<int> outCount(myNodes, 0);

//   if (MPI_Waitall(recvReqs.size(), recvReqs.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
//     throw std::runtime_error("MPI_Waitall failed");
//   }
//   for (size_t i = 0; i < recvBuf.size(); ++i) {
//     outAgg[neighbor.nodes[i]] += recvBuf[i];
//     ++outCount[neighbor.nodes[i]];
//   }
//   for (int i = 0; i < myNodes; ++i) {
//     outData[i] = (inData[i] + outAgg[i]) / (outCount[i] + 1);
//   }
//   if (MPI_Waitall(sendReqs.size(), sendReqs.data(), MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
//     throw std::runtime_error("MPI_Waitall failed");
//   }
// }



void testNeighborExchange(MPI_Comm comm) {
  MPIHelper mpi(comm);  

  DDMPartition partition = makeTestPartition(mpi.size());
  DDMNeighbor neighbor = getDDMNeighbor(mpi.rank(), partition);

  mpi.syncrun([&]() {
    print("- rank: ", mpi.rank());
    print("  partition:");
    dump(partition, indentation(4));
    print("  neighbor:");
    dump(neighbor, indentation(4));
  });

  int nNodeGlobal = partition.getNodeCount();
  int nDomain = partition.getDomainCount();

  using T = Expr;
  RealNodeValue<T> globalData(nNodeGlobal);

  for (int i = 0; i < nNodeGlobal; ++i) {
    globalData[i] = i+1;
  }
  RealNodeValue<T> localData(partition.getNodeCount(mpi.rank()));
  for (int i = 0; i < localData.size(); ++i) {
    localData[i] = globalData[partition.getNode(mpi.rank(), i)]
                   + 100 * (mpi.rank() + 1);
  }

  mpi.barrier();
  mpi.print("Starting exchange");
  DDMMPISynchronizer<RealNodeValue<T>> sync;
  sync.neighborExchange(comm, localData, localData, partition, neighbor);
  mpi.barrier();
  mpi.print("Finished exchange");
  mpi.print("local data", localData);
}


void testFullSynchronize(MPI_Comm comm) {
  MPIHelper mpi(comm);

  DDMPartition partition = makeTestPartition(mpi.size());

  // mpi.syncrun([&]() {
  //   print("- rank: ", mpi.rank());
  //   print("  partition:");
  //   dump(partition, indentation(4));
  // });

  int nNodeGlobal = partition.getNodeCount();

  using T = Expr;
  RealNodeValue<T> data(nNodeGlobal);

  // Each rank sets its own domain's nodes to (globalNodeID + 1) * (rank + 1)
  int myDomain = mpi.rank();
  int myNodes = partition.getNodeCount(myDomain);
  
  for (int iNode = 0; iNode < nNodeGlobal ; ++iNode) {
    data[iNode] = (myDomain + 1) * 100 + iNode;
  }
  

  // for (int iLocal = 0; iLocal < myNodes; ++iLocal) {
  //   int globalNode = partition.getNode(myDomain, iLocal);
  //   data[globalNode] = (globalNode + 1) * (myDomain + 1);
  // }

  mpi.barrier();
  mpi.print("Before sync: ", data);
  mpi.barrier();

  DDMMPISynchronizer<RealNodeValue<T>> sync;
  sync.fullSynchronize(comm, data, partition);

  std::this_thread::sleep_for(std::chrono::seconds(1));

  mpi.barrier();
  mpi.print("After sync: ", data);
  mpi.barrier();
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  try {
    testFullSynchronize(MPI_COMM_WORLD);
  } catch(MPIError const & error) {
    std::cerr << "MPIError: " << error.what() << " (errorcode=" << error.code() << ")" << std::endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
  } catch(...) {
    MPI_Abort(MPI_COMM_WORLD, -1);
  }


  MPI_Finalize();
}