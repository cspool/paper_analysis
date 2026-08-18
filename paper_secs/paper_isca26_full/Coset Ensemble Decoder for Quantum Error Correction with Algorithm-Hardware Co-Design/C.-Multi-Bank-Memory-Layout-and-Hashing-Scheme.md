# *C. Multi-Bank Memory Layout and Hashing Scheme*

To handle highly concurrent local memory accesses during cluster growth, we design a customized multi-bank memory with conflict-free hashing. The hashing maps vertex and edge memory accesses to different memory banks, supporting single-cycle access for distances up to 15[1](#page-5-2) .

Our memory system must satisfy a critical requirement: for any input lattice coordinate (x, y, z), we need to simultaneously access (i) the center vertex and its axis-aligned neighbors, and (ii) all incident edges. The core design principle is to ensure that these concurrent accesses land in distinct banks as shown in Fig. [6.](#page-6-2)

We represent the 3D lattice using integer coordinates (x, y, z) For edges, we use a consistent convention: each edge is identified by the coordinate of its *positive (forward) endpoint*. Specifically:

$$\pm \mathbf{e}_i : (x, y, z) \leftrightarrow (x, y, z) \pm \mathbf{e}_i, \quad i \in \{x, y, z\}$$
 (13)

are all represented by (x, y, z). Our implementation supports lattices with code distance up to 15, with smaller configurations also supported.

<span id="page-5-2"></span><sup>1</sup>Longer distances incur slightly higher clock cycle overhead.

![](_page_6_Figure_0.jpeg)

<span id="page-6-0"></span>Fig. 5. Two-stage hardware architecture: a fully pipelined clustering engine (outside the light-yellow region) feeds the post-clustering modules (inside the light-yellow region), comprising K parallel Ensemble Forest Exploration (EFE) instances and a Voting module that aggregates the K candidate corrections into the final logical-error estimate.

![](_page_6_Figure_2.jpeg)

<span id="page-6-2"></span>Fig. 6. Multi-bank hashing distributes 7-vertex neighborhood to distinct banks.

To distribute vertex data uniformly across banks, we employ a linear congruential hash function that maps each vertex coordinate to a bank index  $b_n$ :

$$b_v(x, y, z) = (\alpha x + \beta y + \gamma z) \bmod M, \tag{14}$$

where  $\alpha=1, \beta=3, \gamma=5$ , and M=22. A key property of these coefficients is that the resulting bank indices for the center (x,y,z) and its axis-aligned neighbors are *pairwise distinct by construction*, guaranteeing conflict-free concurrent accesses.

To support this bank distribution, vertices belonging to the same bank are densely packed using a lexicographic traversal order (i, j, k) with i outermost, then j, then k, where  $i = 0, \ldots, L-1, j = 0, \ldots, L-1, k = 0, \ldots, R-1$ . The bank-internal address  $a_v$  of a vertex is the rank of (x, y, z) among all triples that hash to the same bank:

$$a_v(x, y, z) = \sum_{\substack{0 \le i, j < L \\ 0 \le k < R}} [[(i + 3j + 5k) \bmod 22 = b_v(x, y, z)]] \cdot$$

$$[[(i,j,k) \prec_{\text{lex}} (x,y,z)]] \tag{1}$$

where  $[[\cdot]]$  is the Iverson bracket and  $(i, j, k) \prec_{lex} (x, y, z)$  denotes lexicographic precedence.

#### <span id="page-6-1"></span>D. Hierarchical ID Mapping for Cluster Merging

Another source of memory conflicts happens during cluster merging. It features higher concurrency and weaker spatial locality. Consider the conventional merging process on a 3D grid with  $O(N^3)$  points, where each coordinate stores a CID.

During cluster growth, only a small, spatially contiguous set of points changes its CID. Our existing multi-bank buffer with hash-based placement handles these localized updates efficiently. In contrast, merges between concurrently growing clusters induce many logically simultaneous CID updates that may be scattered across the volume. Directly rewriting the "VID→CID" store for tens of randomly located VIDs per merge scales poorly in hardware, amplifying both bank conflicts and write bandwidth demands. Fig. 7 (a) presents a straightforward way of changing storage mapping relationships. Each thin line represents a storage mapping relationship. Under the straightforward approach, all storage cells whose CID was originally 3, 6, or 7 must be remapped to 1 during cluster merging, resulting in a total of 15 storage cells that need to be updated.

We introduce an intermediate representation that decouples high-fan-out, poorly localized merge updates from the coordinate address space. VIDs are first mapped to RIDs in the multibank, hash-partitioned buffer (optimized for growth). A compact memory then holds an "RID-CID" indirection, where CID is the post-merge cluster identifier. Merge operations update only this "RID \rightarrow CID" mapping: all elements formerly addressed by the merged RIDs are logically relabeled by modifying a small number of RID entries rather than rewriting the many coordinates that reference them. Because "VID-RID" is already a many-to-one mapping during growth, the merge stage's write fan-out collapses from "number of touched VIDs" to "number of touched RIDs," enabling single-cycle remaps in the specialized memory and sharply reducing peak concurrent traffic. In Fig. 7 (b), we only update the memory mapping from RID to CID. In this example, the memory cells storing mapping relationships of RID 3, 6, and 7 are updated. Compared with the straightforward method, the concurrent memory access pressure has been relieved.

#### V. EXPERIMENTAL METHODOLOGY

# *C. Multi-Bank Memory Layout and Hashing Scheme*

To handle highly concurrent local memory accesses during cluster growth, we design a customized multi-bank memory with conflict-free hashing. The hashing maps vertex and edge memory accesses to different memory banks, supporting single-cycle access for distances up to 15[1](#page-5-2) .

Our memory system must satisfy a critical requirement: for any input lattice coordinate (x, y, z), we need to simultaneously access (i) the center vertex and its axis-aligned neighbors, and (ii) all incident edges. The core design principle is to ensure that these concurrent accesses land in distinct banks as shown in Fig. [6.](#page-6-2)

We represent the 3D lattice using integer coordinates (x, y, z) For edges, we use a consistent convention: each edge is identified by the coordinate of its *positive (forward) endpoint*. Specifically:

$$\pm \mathbf{e}_i : (x, y, z) \leftrightarrow (x, y, z) \pm \mathbf{e}_i, \quad i \in \{x, y, z\}$$
 (13)

are all represented by (x, y, z). Our implementation supports lattices with code distance up to 15, with smaller configurations also supported.

<span id="page-5-2"></span><sup>1</sup>Longer distances incur slightly higher clock cycle overhead.

![](_page_6_Figure_0.jpeg)

<span id="page-6-0"></span>Fig. 5. Two-stage hardware architecture: a fully pipelined clustering engine (outside the light-yellow region) feeds the post-clustering modules (inside the light-yellow region), comprising K parallel Ensemble Forest Exploration (EFE) instances and a Voting module that aggregates the K candidate corrections into the final logical-error estimate.

![](_page_6_Figure_2.jpeg)

<span id="page-6-2"></span>Fig. 6. Multi-bank hashing distributes 7-vertex neighborhood to distinct banks.

To distribute vertex data uniformly across banks, we employ a linear congruential hash function that maps each vertex coordinate to a bank index  $b_n$ :

$$b_v(x, y, z) = (\alpha x + \beta y + \gamma z) \bmod M, \tag{14}$$

where  $\alpha=1, \beta=3, \gamma=5$ , and M=22. A key property of these coefficients is that the resulting bank indices for the center (x,y,z) and its axis-aligned neighbors are *pairwise distinct by construction*, guaranteeing conflict-free concurrent accesses.

To support this bank distribution, vertices belonging to the same bank are densely packed using a lexicographic traversal order (i, j, k) with i outermost, then j, then k, where  $i = 0, \ldots, L-1, j = 0, \ldots, L-1, k = 0, \ldots, R-1$ . The bank-internal address  $a_v$  of a vertex is the rank of (x, y, z) among all triples that hash to the same bank:

$$a_v(x, y, z) = \sum_{\substack{0 \le i, j < L \\ 0 \le k < R}} [[(i + 3j + 5k) \bmod 22 = b_v(x, y, z)]] \cdot$$

$$[[(i,j,k) \prec_{\text{lex}} (x,y,z)]] \tag{1}$$

where  $[[\cdot]]$  is the Iverson bracket and  $(i, j, k) \prec_{lex} (x, y, z)$  denotes lexicographic precedence.

#### <span id="page-6-1"></span>D. Hierarchical ID Mapping for Cluster Merging

Another source of memory conflicts happens during cluster merging. It features higher concurrency and weaker spatial locality. Consider the conventional merging process on a 3D grid with  $O(N^3)$  points, where each coordinate stores a CID.

During cluster growth, only a small, spatially contiguous set of points changes its CID. Our existing multi-bank buffer with hash-based placement handles these localized updates efficiently. In contrast, merges between concurrently growing clusters induce many logically simultaneous CID updates that may be scattered across the volume. Directly rewriting the "VID→CID" store for tens of randomly located VIDs per merge scales poorly in hardware, amplifying both bank conflicts and write bandwidth demands. Fig. 7 (a) presents a straightforward way of changing storage mapping relationships. Each thin line represents a storage mapping relationship. Under the straightforward approach, all storage cells whose CID was originally 3, 6, or 7 must be remapped to 1 during cluster merging, resulting in a total of 15 storage cells that need to be updated.

We introduce an intermediate representation that decouples high-fan-out, poorly localized merge updates from the coordinate address space. VIDs are first mapped to RIDs in the multibank, hash-partitioned buffer (optimized for growth). A compact memory then holds an "RID-CID" indirection, where CID is the post-merge cluster identifier. Merge operations update only this "RID \rightarrow CID" mapping: all elements formerly addressed by the merged RIDs are logically relabeled by modifying a small number of RID entries rather than rewriting the many coordinates that reference them. Because "VID-RID" is already a many-to-one mapping during growth, the merge stage's write fan-out collapses from "number of touched VIDs" to "number of touched RIDs," enabling single-cycle remaps in the specialized memory and sharply reducing peak concurrent traffic. In Fig. 7 (b), we only update the memory mapping from RID to CID. In this example, the memory cells storing mapping relationships of RID 3, 6, and 7 are updated. Compared with the straightforward method, the concurrent memory access pressure has been relieved.

#### V. EXPERIMENTAL METHODOLOGY

