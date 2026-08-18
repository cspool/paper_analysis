# *B. BALD Algorithm*

Our proposed Balanced Allocation with Load and Distance awareness (BALD) algorithm consists of three main steps: path profiling, path scheduling, and heuristic backtracking. The algorithm optimizes link allocation for multiple point-topoint and multicast tasks while remaining topology-agnostic and fault-tolerant.

*1) Path Profiling:* For a given topology, profiling node connectivity in advance reduces the search overhead of subsequent path planning and provides an efficient initial state. We use Dijkstra's shortest-path algorithm [22] to traverse the topology and obtain link connectivity and shortest communication distances between all node pairs, naturally handling asymmetric node degrees and heterogeneous edge weights.

The profiling process computes all-pairs shortest paths on a topology T = (N, E), where N denotes the set of nodes and E represents the set of edges. For each source node s ∈ N, Dijkstra's algorithm is invoked to obtain distances to all other nodes, and the resulting shortest-path distances are stored in S, as shown in Lines 2–20 of Algorithm 1.

For a 2D mesh topology, multiple shortest paths may exist between two nodes; for example, ((8, 9), (9, 5)) and ((8, 4), (4, 5)) are both the shortest paths from node 8 to node 5 in Fig. 7. To capture this information, we also maintain a unique path map U that records the unique paths for each node pair, as shown in Lines 16–18 and Lines 21–27 of Algorithm 1. During Dijkstra's algorithm, if a newly discovered path is shorter than the recorded one, the distance and path in U are updated; if it has the same length, the path count is incremented and the pair is marked non-unique.

