#pragma once

#include <vector>
#include "anyprint.hh"

namespace NSPC_DDM {

class DDMPartition {
public:
  DDMPartition() = default;

  /// Construct a partition with the given number of global nodes and domains.
  /// The node-domain associations are initially empty and must be populated
  /// via setNodesOfDomain().
  DDMPartition(int nNode, int nDomain);
  DDMPartition(DDMPartition &&) = default;
  DDMPartition(DDMPartition const &) = default;
  DDMPartition & operator=(DDMPartition &&) = default;
  DDMPartition & operator=(DDMPartition const &) = default;

  /// Sort and deduplicate the node and domain lists for all entries.
  void normalize();

  /// Assign the list of global node indices to a domain, and update the
  /// reverse mapping (domainsOfNode) accordingly. Does not remove previous
  /// associations—call normalize() after all domains are set to clean up
  /// duplicates.
  void setNodesOfDomain(int iDomain, std::vector<int> const & nodeMapping);

  std::vector<int> const & getNodes(int iDomain) const;
  std::vector<int> const & getDomains(int iNode) const;

  int getNodeCount() const { return _domainsOfNode.size(); }
  int getDomainCount() const { return _nodesOfDomain.size(); }

  int getNodeCount(int iDomain) const { return _nodesOfDomain[iDomain].size(); }
  int getDomainCount(int iNode) const { return _domainsOfNode[iNode].size(); }
  int getNode(int iDomain, int iLocalNode) const { return _nodesOfDomain[iDomain][iLocalNode]; }

  /// Return true if global node iNode belongs to domain iDomain.
  bool isNodeOwnedBy(int iNode, int iDomain) const;

  /// Return the local index of global node iNode within domain iDomain.
  /// The node must belong to the domain (asserts on failure).
  int getLocalNodeIndex(int iNode, int iDomain) const;

public:
  std::vector<std::vector<int>> _nodesOfDomain;
  std::vector<std::vector<int>> _domainsOfNode;
};

/// Neighbor information for a single domain, stored in a CSR-like layout.
/// neighborDomain[k] is the k-th neighboring domain ID.
/// nodes[displacements[k] .. displacements[k+1]) are the local node indices
/// shared with neighborDomain[k].
struct DDMNeighbor {
  std::vector<int> neighborDomain;
  std::vector<int> displacements;
  std::vector<int> nodes;
};

/// Build the neighbor information for a given domain from the partition.
/// For each neighboring domain, records the local node indices (within
/// iDomain) that are shared, stored in a CSR-like layout using
/// displacements and nodes arrays.
DDMNeighbor getDDMNeighbor(int iDomain, DDMPartition const & partition);

/// Reconstruct a DDMPartition from per-domain node counts and neighbor
/// information alone (without prior knowledge of global node IDs).
/// Uses a disjoint-set (union-find) to identify shared nodes across
/// domain boundaries and assigns new global node IDs.
/// Assumes shared nodes are listed in the same relative order in both
/// neighboring domains.
DDMPartition reconstructDDMPartition(
    std::vector<int> const & nNodesPerDomain,
    std::vector<DDMNeighbor> const & neighbors);

void dump(DDMNeighbor const & neighbor, anyprint::indentation const & indent=anyprint::indentation(0));
void dump(DDMPartition const & partition, anyprint::indentation const & indent=anyprint::indentation(0));

}