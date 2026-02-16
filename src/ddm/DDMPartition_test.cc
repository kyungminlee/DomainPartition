#include "DDMPartition.hh"
#include <algorithm>
#include <cassert>
#include <iostream>

#include "anyprint.hh"

using namespace NSPC_DDM;

DDMPartition createTestDDMPartition() {
  int nNode = 10;
  int nDomain = 3;
  DDMPartition partition(nNode, nDomain);

  // Domain 0: [0, 1, 2, 3]
  partition.setNodesOfDomain(0, {0, 1, 2, 3});
  // Domain 1: [2, 3, 4, 5, 6] (overlaps with 0 at 2,3)
  partition.setNodesOfDomain(1, {2, 3, 4, 5, 6});
  // Domain 2: [5, 6, 7, 8, 9] (overlaps with 1 at 5,6)
  partition.setNodesOfDomain(2, {5, 6, 7, 8, 9});

  partition.normalize();
  return partition;
}

DDMPartition createTestDDMPartition2() {
  int nNode = 20;
  int nDomain = 5;
  DDMPartition partition(nNode, nDomain);

  partition.setNodesOfDomain(0, {0, 1, 2, 3, 4, 5});
  // Overlap D0: {5} (1 node)
  partition.setNodesOfDomain(1, {5, 6, 7, 8, 9, 10});
  // Overlap D1: {8, 9, 10} (3 nodes)
  partition.setNodesOfDomain(2, {8, 9, 10, 11, 12, 13});
  // Overlap D2: {12, 13} (2 nodes)
  partition.setNodesOfDomain(3, {12, 13, 14, 15, 16});
  // Overlap D1: {6, 7} (2 nodes), Overlap D3: {16} (1 node)
  partition.setNodesOfDomain(4, {6, 7, 16, 17, 18, 19});

  partition.normalize();
  return partition;
}

void testDDMPartition(DDMPartition const & partition) {
  // Check consistency: Nodes -> Domains
  for (int iDomain = 0; iDomain < partition.getDomainCount(); ++iDomain) {
    std::vector<int> const & nodes = partition.getNodes(iDomain);
    for (int i = 0; i < (int)nodes.size(); ++i) {
      int iNode = nodes[i];
      assert(partition.isNodeOwnedBy(iNode, iDomain));
      assert(partition.getLocalNodeIndex(iNode, iDomain) == i);
    }
  }

  // Check consistency: Domains -> Nodes
  for (int iNode = 0; iNode < partition.getNodeCount(); ++iNode) {
    std::vector<int> const & domains = partition.getDomains(iNode);
    for (int iDomain : domains) {
      std::vector<int> const & domainNodes = partition.getNodes(iDomain);
      assert(std::find(domainNodes.begin(), domainNodes.end(), iNode) != domainNodes.end());
    }
  }
  std::cout << "testDDMPartition passed." << std::endl;
}

// void testDDMNeighbor(DDMPartition const & partition) {
//   // Test neighbors for Domain 1
//   // Domain 1 has nodes {2, 3, 4, 5, 6}
//   // Neighbors:
//   // Domain 0 shares {2, 3} -> local indices {0, 1} inside Domain 1
//   // Domain 2 shares {5, 6} -> local indices {3, 4} inside Domain 1

//   DDMNeighbor neighbor = getDDMNeighbor(1, partition);

//   // Check neighbor domains
//   assert(neighbor.neighborDomain.size() == 2);
//   assert(neighbor.neighborDomain[0] == 0);
//   assert(neighbor.neighborDomain[1] == 2);

//   // Check displacements
//   assert(neighbor.displacements.size() == 3);
//   assert(neighbor.displacements[0] == 0);
//   assert(neighbor.displacements[1] == 2);
//   assert(neighbor.displacements[2] == 4);

//   // Check shared node indices (local indices in Domain 1)
//   assert(neighbor.nodes.size() == 4);
//   std::vector<int> expectedNodes = {0, 1, 3, 4};
//   assert(neighbor.nodes == expectedNodes);

//   // using namespace anyprint;
//   // print("neighborDomain: ", neighbor.neighborDomain);
//   // print("displacements: ", neighbor.displacements);
//   // print("nodes: ", neighbor.nodes);
//   std::cout << "testDDMNeighbor passed." << std::endl;
// }

void debugDDMNeighbor(DDMPartition const & partition) {
  using namespace anyprint;
  for (int iDomain = 0; iDomain < partition.getDomainCount(); ++iDomain) {
    DDMNeighbor neighbor = getDDMNeighbor(iDomain, partition);
    print("- domain: ", iDomain);
    print("  neighborDomain: ", neighbor.neighborDomain);
    write(std::cout, "  globalNodes: [");
    {
      char const * sep = "";
      for (int x : neighbor.nodes) {
        std::cout << sep; sep = ", ";
        std::cout << partition.getNodes(iDomain)[x];
      }
      std::cout << "]\n";
    }
    print("  localNodes: ", neighbor.nodes);
    print("  displacements: ", neighbor.displacements);
  }
}


int main() {
  DDMPartition partition = createTestDDMPartition2();

  testDDMPartition(partition);
  // testDDMNeighbor(partition);
  debugDDMNeighbor(partition);
}