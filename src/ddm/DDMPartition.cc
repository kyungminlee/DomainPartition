#include "DDMPartition.hh"
#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <numeric>
#include "anyprint.hh"

namespace NSPC_DDM {

/// @brief Construct a partition with @p nNode global nodes and @p nDomain domains.
///
/// Allocates empty node lists for each domain and empty domain lists for each
/// node. Use setNodesOfDomain() to populate the associations afterward.
DDMPartition::DDMPartition(int nNode, int nDomain)
  : _nodesOfDomain(nDomain)
  , _domainsOfNode(nNode)
{
}

/// @brief Sort and deduplicate all node-domain association lists.
///
/// For every domain, sorts its node list and removes duplicates.
/// For every node, sorts its domain list and removes duplicates.
/// Should be called after all setNodesOfDomain() calls are complete to ensure
/// consistent, canonical ordering of the internal data structures.
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

/// @brief Assign a list of global node indices to a domain.
///
/// Replaces the node list for domain @p iDomain with @p nodeMapping, and
/// appends @p iDomain to the domain list of each referenced node (the reverse
/// mapping). Note that this does not clear previous reverse-mapping entries,
/// so calling this multiple times for the same domain without resetting will
/// accumulate duplicates. Call normalize() after all domains have been
/// populated to remove any duplicates.
///
/// @param iDomain    The domain index (must be < getDomainCount()).
/// @param nodeMapping  The global node indices belonging to this domain.
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

/// @brief Check whether a global node belongs to a given domain.
///
/// Performs a linear search through the domain list of @p iNode.
/// For normalized partitions the domain lists are sorted, but this
/// implementation does not exploit that (linear scan).
///
/// @param iNode    Global node index.
/// @param iDomain  Domain index to test membership against.
/// @return True if @p iNode is listed among the nodes of @p iDomain.
bool DDMPartition::isNodeOwnedBy(int iNode, int iDomain) const {
  auto const & domains = _domainsOfNode[iNode];
  return std::find(domains.begin(), domains.end(), iDomain) != domains.end();
}

/// @brief Find the local index of a global node within a domain's node list.
///
/// Searches the node list of @p iDomain for @p iNode and returns its
/// zero-based position. Asserts that the node is found; calling this with
/// a node that does not belong to the domain is a programming error.
///
/// @param iNode    Global node index to look up.
/// @param iDomain  Domain whose local ordering is queried.
/// @return The local index (position) of @p iNode in the domain's node list.
int DDMPartition::getLocalNodeIndex(int iNode, int iDomain) const {
  std::vector<int> const & nodes = _nodesOfDomain[iDomain];
  auto iter = std::find(nodes.begin(), nodes.end(), iNode);
  assert(iter != nodes.end());
  return std::distance(nodes.begin(), iter);
}

/// @brief Build the neighbor information for a single domain.
///
/// Iterates over all nodes owned by @p iDomain.  For each node, examines
/// every other domain that also owns that node and records the node's
/// *local* index (within @p iDomain) as a shared node with that neighbor.
///
/// The result is packed into a CSR-like DDMNeighbor structure:
///   - neighborDomain[k]: the k-th neighboring domain ID (sorted by domain ID
///     since std::map is used internally).
///   - nodes[displacements[k] .. displacements[k+1]): the local node indices
///     shared with neighborDomain[k], in sorted order (std::set).
///   - displacements: prefix-sum offsets into the nodes array, with
///     displacements[0] == 0.
///
/// @param iDomain    The domain for which to compute neighbor information.
/// @param partition  The global partition containing node-domain associations.
/// @return A DDMNeighbor describing all neighbor domains and shared nodes.
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
  /// @brief Disjoint-set (union-find) with path compression.
  ///
  /// Used internally by reconstructDDMPartition() to merge local node indices
  /// that refer to the same global node across domain boundaries.
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

/// @brief Reconstruct a DDMPartition from per-domain node counts and neighbor data.
///
/// Given only the number of nodes in each domain and the neighbor connectivity
/// (which local nodes are shared between which domains), this function recovers
/// a full DDMPartition with consistent global node IDs.
///
/// Algorithm:
///   1. Assign each local node a unique linear index by concatenating domains
///      (domain i's local node j gets index offsets[i] + j).
///   2. For every pair of neighboring domains, use the DDMNeighbor data to
///      identify which local nodes correspond to the same physical node, and
///      merge them via union-find (DSU).  Each pair is processed only once
///      (neighborID > i) to avoid double-merging.
///   3. Flatten the DSU into contiguous global IDs (0, 1, 2, ...) by mapping
///      each DSU root to a new sequential ID.
///   4. Build the DDMPartition using the computed global IDs and normalize it.
///
/// @pre The shared nodes between any two neighboring domains must be listed in
///      the same relative order in both DDMNeighbor structures (i.e., the k-th
///      shared node from domain A's perspective corresponds to the k-th shared
///      node from domain B's perspective).
///
/// @param nNodesPerDomain  Number of local nodes in each domain.
/// @param neighbors        Neighbor information for each domain (one per domain).
/// @return A fully populated and normalized DDMPartition.
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

/// @brief Print the contents of a DDMNeighbor for debugging.
void dump(DDMNeighbor const & neighbor, anyprint::indentation const & indent) {
  using namespace anyprint;
  print(indent, "neighborDomain: ", neighbor.neighborDomain);
  print(indent, "nodes: ", neighbor.nodes);
  print(indent, "displacements: ", neighbor.displacements);
}

/// @brief Print the contents of a DDMPartition for debugging.
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