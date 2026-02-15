#pragma once

#include <vector>
#include "anyprint.hh"

class DDMPartition {
public:
  DDMPartition() = default;
  DDMPartition(int nNode, int nDomain);
  DDMPartition(DDMPartition &&) = default;
  DDMPartition(DDMPartition const &) = default;
  DDMPartition & operator=(DDMPartition &&) = default;
  DDMPartition & operator=(DDMPartition const &) = default;

  void normalize();
  void setNodesOfDomain(int iDomain, std::vector<int> const & nodeMapping);
  std::vector<int> const & getNodes(int iDomain) const;
  std::vector<int> const & getDomains(int iNode) const;

  int getNodeCount() const { return _domainsOfNode.size(); }
  int getDomainCount() const { return _nodesOfDomain.size(); }

  int getNodeCount(int iDomain) const { return _nodesOfDomain[iDomain].size(); }
  int getDomainCount(int iNode) const { return _domainsOfNode[iNode].size(); }
  int getNode(int iDomain, int iLocalNode) const { return _nodesOfDomain[iDomain][iLocalNode]; }

  bool isNodeOwnedBy(int iNode, int iDomain) const;
  int getLocalNodeIndex(int iNode, int iDomain) const;

public:
  std::vector<std::vector<int>> _nodesOfDomain;
  std::vector<std::vector<int>> _domainsOfNode;
};

struct DDMNeighbor {
  std::vector<int> neighborDomain;
  std::vector<int> displacements;
  std::vector<int> nodes;
};

DDMNeighbor getDDMNeighbor(int iDomain, DDMPartition const & partition);

DDMPartition reconstructDDMPartition(
    std::vector<int> const & nNodesPerDomain,
    std::vector<DDMNeighbor> const & neighbors);

void dump(DDMNeighbor const & neighbor, anyprint::indentation const & indent);
void dump(DDMPartition const & partition, anyprint::indentation const & indent);