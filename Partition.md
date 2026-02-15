# DDMPartition Class Overview

The `DDMPartition` class is responsible for managing the topology of a Domain Decomposition Method (DDM). It acts as a bidirectional mapping system between **Global Nodes** (mesh vertices) and **Domains** (partitions).

## 1. Core Data Storage
The class maintains the partition state using two synchronized adjacency lists (vectors of vectors). This allows for efficient lookups in both directions.

| Member Variable | Type | Description |
| :--- | :--- | :--- |
| **`_nodesOfDomain`** | `vector<vector<int>>` | **Forward Map**: Given a `Domain ID`, retrieves the list of Global Node IDs inside it. |
| **`_domainsOfNode`** | `vector<vector<int>>` | **Reverse Map**: Given a `Node ID`, retrieves the list of Domains that share/own this node. |

## 2. Key Workflows

### Initialization
You must construct the object with the total counts upfront. This pre-allocates the primary vectors to avoid resizing overhead later.
```cpp
DDMPartition partition(nNode, nDomain);
```

### Population (`setNodesOfDomain`)
This is the primary setter. You provide the list of nodes for a specific domain.
*   **Automatic Reverse Mapping**: When you call `setNodesOfDomain`, the class **automatically** iterates through the provided nodes and updates `_domainsOfNode`. You do not need to manually populate the node-to-domain relationship.

### Normalization (`normalize`)
This method is a "finalize" step. It iterates through all internal lists to:
1.  **Sort** the IDs (nodes within a domain, and domains within a node).
2.  **Remove duplicates** (`std::unique`).
*   *Why it matters*: It ensures deterministic ordering for `getLocalNodeIndex` and cleaner data for ownership checks.

## 3. Querying the Partition

### Global-to-Local Mapping
*   **`getLocalNodeIndex(int iNode, int iDomain)`**:
    *   Converts a Global Node ID to a local index (0 to $N_{local}-1$) specific to that domain.
    *   **Performance Note**: This performs a linear search (`std::find`) over the domain's node list. It is $O(N_{domain\_size})$.

### Connectivity & Ownership
*   **`getDomains(int iNode)`**: Returns which domains share a specific node (useful for identifying interface nodes).
*   **`isNodeOwnedBy(int iNode, int iDomain)`**: Boolean check to see if a node exists in a specific domain.

## 4. Usage Example

```cpp
// 1. Setup: 100 global nodes, split into 4 domains
DDMPartition ddm(100, 4);

// 2. Populate: Define nodes for Domain 0
// This automatically registers Domain 0 in _domainsOfNode for nodes 10, 11, and 12.
ddm.setNodesOfDomain(0, {10, 11, 12});

// 3. Finalize: Sorts internal vectors
ddm.normalize();

// 4. Usage
// Get all domains touching node 11
std::vector<int> const& owners = ddm.getDomains(11);

// Find where node 12 sits in Domain 0's local vector
int localIndex = ddm.getLocalNodeIndex(12, 0);
```