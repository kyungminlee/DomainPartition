#include "DDMPartition.hh"
#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <numeric>
#include "anyprint.hh"

namespace NSPC_DDM {

DDMPartition::DDMPartition(int nNode, int nDomain)
  : _nodesOfDomain(nDomain)
  , _domainsOfNode(nNode)
{
}

void DDMPartition::normalize() {
  for (auto & nodes: _nodesOfDomain) {
    std::sort(nodes.begin(), nodes.end());
    nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  }
  for (auto & domains: _domainsOfNode) {
    std::sort(domains.begin(), domains.end());
    domains.erase(std::unique(domains.begin(), domains.end()), domains.end());
  }
}

void DDMPartition::setNodesOfDomain(int iDomain, std::vector<int> const & nodeMapping) {
  assert(iDomain < _nodesOfDomain.size());
  _nodesOfDomain[iDomain] = nodeMapping;
  for (auto iNode: nodeMapping) {
    _domainsOfNode[iNode].push_back(iDomain);
  }
}

std::vector<int> const & DDMPartition::getNodes(int iDomain) const {
  assert(iDomain < _nodesOfDomain.size());
  return _nodesOfDomain[iDomain];
}

std::vector<int> const & DDMPartition::getDomains(int iNode) const {
  assert(iNode < _domainsOfNode.size());
  return _domainsOfNode[iNode];
}

bool DDMPartition::isNodeOwnedBy(int iNode, int iDomain) const {
  auto const & domains = _domainsOfNode[iNode];
  return std::find(domains.begin(), domains.end(), iDomain) != domains.end();
}

int DDMPartition::getLocalNodeIndex(int iNode, int iDomain) const {
  std::vector<int> const & nodes = _nodesOfDomain[iDomain];
  auto iter = std::find(nodes.begin(), nodes.end(), iNode);
  assert(iter != nodes.end());
  return std::distance(nodes.begin(), iter);
}

DDMNeighbor getDDMNeighbor(int iDomain, DDMPartition const & partition) {
  DDMNeighbor neighbor;
  std::map<int, std::set<int>> neighborMap;

  std::vector<int> const & nodes = partition.getNodes(iDomain);
  for (int i = 0; i < (int)nodes.size(); ++i) {
    int iNode = nodes[i];
    for (int neighborID : partition.getDomains(iNode)) {
      if (neighborID != iDomain) {
        neighborMap[neighborID].insert(i);
      }
    }
  }

  size_t totalNodes = 0;
  for (auto const & kv : neighborMap) {
    totalNodes += kv.second.size();
  }
  neighbor.neighborDomain.reserve(neighborMap.size());
  neighbor.displacements.reserve(neighborMap.size() + 1);
  neighbor.nodes.reserve(totalNodes);

  neighbor.displacements.push_back(0);
  for (auto const & kv : neighborMap) {
    neighbor.neighborDomain.push_back(kv.first);
    neighbor.nodes.insert(neighbor.nodes.end(), kv.second.begin(), kv.second.end());
    neighbor.displacements.push_back(neighbor.nodes.size());
  }
  return neighbor;
}

namespace {
  struct DSU {
    std::vector<int> parent;
    DSU(int n) : parent(n) {
      std::iota(parent.begin(), parent.end(), 0);
    }
    int find(int i) {
      if (parent[i] == i) return i;
      return parent[i] = find(parent[i]);
    }
    void unite(int i, int j) {
      int root_i = find(i);
      int root_j = find(j);
      if (root_i != root_j) {
        parent[root_i] = root_j;
      }
    }
  };
}

DDMPartition reconstructDDMPartition(
    std::vector<int> const & nNodesPerDomain,
    std::vector<DDMNeighbor> const & neighbors)
{
  int nDomain = nNodesPerDomain.size();
  assert(neighbors.size() == (size_t)nDomain);

  // 1. Calculate offsets for linear indexing of all local nodes
  std::vector<int> offsets(nDomain + 1, 0);
  for (int i = 0; i < nDomain; ++i) {
    offsets[i + 1] = offsets[i] + nNodesPerDomain[i];
  }
  int totalLocalNodes = offsets[nDomain];

  // 2. Use DSU to merge shared nodes based on neighbor info
  DSU dsu(totalLocalNodes);

  for (int i = 0; i < nDomain; ++i) {
    auto const & myNeighbor = neighbors[i];
    int nNeighbors = myNeighbor.neighborDomain.size();

    for (int k = 0; k < nNeighbors; ++k) {
      int neighborID = myNeighbor.neighborDomain[k];
      
      // Process each pair of domains only once
      if (neighborID < i) continue;

      // Identify shared nodes in Domain i
      int start_i = myNeighbor.displacements[k];
      int count_i = myNeighbor.displacements[k+1] - start_i;
      
      // Identify corresponding shared nodes in Domain neighborID
      auto const & otherNeighbor = neighbors[neighborID];
      auto it = std::find(otherNeighbor.neighborDomain.begin(), 
                          otherNeighbor.neighborDomain.end(), i);
      assert(it != otherNeighbor.neighborDomain.end());
      
      int k_other = std::distance(otherNeighbor.neighborDomain.begin(), it);
      int start_j = otherNeighbor.displacements[k_other];
      int count_j = otherNeighbor.displacements[k_other+1] - start_j;

      assert(count_i == count_j);

      // Unite the corresponding nodes
      // Assumption: The shared nodes are listed in the same relative order (Global ID order)
      for (int m = 0; m < count_i; ++m) {
        int u = offsets[i] + myNeighbor.nodes[start_i + m];
        int v = offsets[neighborID] + otherNeighbor.nodes[start_j + m];
        dsu.unite(u, v);
      }
    }
  }

  // 3. Assign new Global IDs to the disjoint sets
  std::vector<int> globalID(totalLocalNodes);
  std::vector<int> rootToGlobal(totalLocalNodes, -1);
  int currentGlobalID = 0;

  for (int i = 0; i < totalLocalNodes; ++i) {
    int root = dsu.find(i);
    if (rootToGlobal[root] == -1) {
      rootToGlobal[root] = currentGlobalID++;
    }
    globalID[i] = rootToGlobal[root];
  }

  // 4. Construct the Partition
  DDMPartition partition(currentGlobalID, nDomain);
  for (int i = 0; i < nDomain; ++i) {
    std::vector<int> domainNodes;
    domainNodes.reserve(nNodesPerDomain[i]);
    for (int local = 0; local < nNodesPerDomain[i]; ++local) {
      domainNodes.push_back(globalID[offsets[i] + local]);
    }
    partition.setNodesOfDomain(i, domainNodes);
  }
  
  partition.normalize();
  return partition;
}

void dump(DDMNeighbor const & neighbor, anyprint::indentation const & indent) {
  using namespace anyprint;
  print(indent, "neighborDomain: ", neighbor.neighborDomain);
  print(indent, "nodes: ", neighbor.nodes);
  print(indent, "displacements: ", neighbor.displacements);
}

void dump(DDMPartition const & partition, anyprint::indentation const & indent) {
  using namespace anyprint;
  print(indent, "nodesOfDomain:");
  for (int iDomain = 0; iDomain < partition.getDomainCount(); ++iDomain) {
    print(indent, "  domain: ", iDomain);
    print(indent, "  nodes: ", partition.getNodes(iDomain));
  }
  print(indent, "domainsOfNode:");
  for (int iNode = 0; iNode < partition.getNodeCount(); ++iNode) {
    print(indent, "  node: ", iNode);
    print(indent, "  domains: ", partition.getDomains(iNode));
  }
}

}