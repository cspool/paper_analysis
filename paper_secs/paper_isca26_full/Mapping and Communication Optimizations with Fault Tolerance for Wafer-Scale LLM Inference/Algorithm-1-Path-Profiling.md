# Algorithm 1 Path Profiling

```
Require: Topology T = (N, E)
Ensure: Shortest distance map S, Unique path map U
 1: Initialize S ← ∅, U ← ∅
 2: for each node s ∈ N do
 3: For all v ∈ N: set d[v] ← ∞, prev[v] ← None, ways[v] ←
      0, uniq[v] ← True
 4: Set d[s] ← 0, ways[s] ← 1, Queue Q ← [(0, s)]
 5: while Q not empty do
 6: Pop (dist, u) from Q
 7: if dist > d[u] then
 8: continue
 9: end if
10: for each neighbor v of u do
11: alt ← d[u] + weight(u, v)
12: if alt < d[v] then
13: d[v] ← alt, prev[v] ← u, ways[v] ← ways[u],
            uniq[v] ← True, Push (alt, v) into Q
14: else if alt = d[v] then
15: ways[v] += ways[u], uniq[v] ← False
16: end if
17: end for
18: end while
19: for each node t ∈ N do
20: if uniq[t] and d[t] < ∞ then
21: U[(s, t)] ← path from s to t via prev
22: end if
23: end for
24: S[s] ← d
25: end for
26: return (S, U)
```

*2) Path Scheduling:* Path Scheduling is the core of the BALD algorithm, which aims to allocate links for multiple multicast tasks while considering both link occupancy and communication distance. It allocates link resources for each task based on the profiling results and the current network state. The algorithm iteratively selects tasks and allocates links, ensuring balanced link load across the network. The scheduling process is shown in Algorithm 2.

A communication task C is represented as a set of pairs P(s, D), where s is the source node and D is the set of destination nodes. For each task, the algorithm iterates over the current frontier nodes as branches and selects available neighbors as candidates for the next allocated link. For each neighbor, the BALD algorithm computes a priority score based on three factors: branch cost, link load, and neighbor distance. Branch cost represents the cost of the current branch, i.e., the earliest time when it can be scheduled for a new task. Link load indicates the current occupancy of the link, which is typically the main bottleneck of collective communication on mesh topologies. Neighbor distance reflects the distance to the nearest destination and is precomputed in the path profiling step. The priority is computed as a weighted sum of these factors, enabling flexible tuning of the algorithm's behavior to balance communication distance and link load. The weights  $\alpha,\,\beta,$  and  $\gamma$  can be adjusted to emphasize different aspects of the scheduling process.

#### **Algorithm 2** Path Scheduling

```
Require: Communication C = (P(s, D)), where P is the set of
    multicast pairs with source s, destinations D
Require: Topology T = (N, E)
Require: Parameters \alpha, \beta, \gamma
Ensure: Communication Link Allocation A
 1: while P is not empty do
       for each p \in P do
 3:
          Set current_priority \leftarrow \infty
          for each branch in path[p] do
 5.
             for each neighbor in available_neighbors do
                Compute priority = \alpha \times branch\_cost + \beta \times link\_load
 6:
                +\gamma \times neighbor_distance
 7:
                if priority < current_priority then
                   Update
 8:
                                current_priority,
                                                        current branch,
                   current_candidate
 9.
                end if
             end for
10:
11:
          end for
          Set path[p][(branch, candidate)]
12:
13:
          Update branch_cost, link_load
          if t is candidate then
14:
             Remove t from p
15:
16:
          end if
17:
          A[p] \leftarrow path[p]
18:
       end for
19: end while
```

3) Heuristic Backtracking: Although the proposed path scheduling algorithm can effectively allocate links for multiple multicast tasks, it may not always yield the best allocation due to the complexity of the problem and the random order in which tasks are processed. In the example above, if (7, 4) is not occupied by task (7, 5), nodes 6 and 4 receive the same priority and the algorithm may pick one at random. Several such random choices produce link contention and degraded performance. We introduce a heuristic backtracking mechanism shown in Algorithm 3 that allows the algorithm to explore alternative paths and improve the overall link allocation.

The heuristic backtracking algorithm iteratively selects tasks whose allocated path uses the most heavily loaded link and then attempts to reallocate them. The algorithm maintains a tabu forbidden list to avoid revisiting previously explored paths, and a tabu candidate list to store tasks that have been

successfully backtracked. Based on a probability threshold  $\rho$ , the algorithm randomly selects tasks either from the tabu candidate list or from tasks not in the tabu forbidden list. We observe an interesting property: if a task is reallocated and the result improves overall allocation, that task is likely to remain a profitable target for further reallocation. Thus, the algorithm keeps track of tasks that have been successfully backtracked and adds them to tabu candidates, as shown in Lines 18–24 in Algorithm 3. If the total load of the overloaded links decreases after backtracking, the tasks are added as tabu candidates; otherwise, they are added to the tabu forbidden list. This process continues for a preset number of iterations I or until no further improvements can be made.

