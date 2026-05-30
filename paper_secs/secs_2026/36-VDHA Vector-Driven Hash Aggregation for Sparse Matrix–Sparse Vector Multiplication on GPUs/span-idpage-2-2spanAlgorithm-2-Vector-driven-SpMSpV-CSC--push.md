# <span id="page-2-2"></span>Algorithm 2 Vector-driven SpMSpV (CSC / push)

**Input:** *A* in CSC: (col\_ptr, indices, values); *x* in sparse format: (idx, val); vector length *n* 

**Output:** Result vector *y* 

```
1: for all i \leftarrow 0 to n in parallel do
2: col \leftarrow idx[i], v\_val \leftarrow val[i]
3: start \leftarrow col\_ptr[col], end \leftarrow col\_ptr[col + 1]
4: ind\_list, val\_list = fetch(indices, values, i)
5: for j \leftarrow 0 to len(ind\_list) do
6: ind \leftarrow ind\_list[j], mat\_val \leftarrow val\_list[j]
7: write\_back(y[ind], mat\_val \otimes v\_val)
```

#### Different kinds of work-balance methods (fetch stage).

In the fetch phase, each nonzero in the vector indexes a column of the CSC matrix and loads the corresponding *indices* and *values* into *ind\_list* and *val\_list*. Since column lengths are highly irregular, different load-balancing strategies are used to assign these column workloads to CTAs. We highlight three representative methods:

- (1) Direct-mapped. Each active column is assigned to one CTA, with no prefix-scan overhead. This simple strategy is commonly used in implementations without fine-grained load balancing, such as NaiveSpMSpV [30].
- (2) **Block-mapped.** Multiple short columns are grouped for a CTA, and a block-level prefix scan computes their combined nonzero count, as used in graph frameworks like Gunrock [34].
- (3) Global-mapped. For each nonzero in the input vector, the corresponding column length is first recorded. A global inclusive scan over these column lengths yields the total number of matrix entries to be accessed, which is then evenly partitioned so that each CTA processes a segment of similar size, following methods used in merge-based SpMV [24, 31].

![](_page_3_Figure_5.jpeg)

**Figure 2.** Illustration of three load-balancing strategies in fetch step (matrix in CSC format). Cells with the same color represent column segments assigned to the same CTA.

Different kinds of write-back (write-back stage). In column-major SpMSpV, partial products ( $v\_val \otimes m\_val$ ) are accumulated into the output y[ind] during the write-back stage, which naturally leads to many-to-one updates. Several strategies are commonly used to realize this write-back:

- **(1) Atomic write-back.** Directly accumulates each partial product into the output using global atomics.
- **(2) Sort-based write-back.** Buffers (*row*, *val*) pairs, sorts them by row index, and reduces duplicates so that each row is written once, thereby avoiding global atomics.
- (3) Hash-based aggregation. Hash aggregation is a light-weight accumulation strategy widely used in sparse kernels such as hash-based SpGEMM [12, 13, 25, 28, 36]. Instead of emitting every update to global memory, partial products are inserted into a small shared-memory hash table. Each update is stored as a (key, value) pair, where the key is the row index; collisions are resolved using simple schemes such

as linear probing (i.e., probing the next slot until an empty or matching entry is found).

In hash-based SpGEMM, each output column maintains its own small hash table, and its size can be estimated through lightweight precomputation. In contrast, SpMSpV produces a *single* output vector, so all intermediate updates converge to one logical output column. This substantially increases the required aggregation capacity, necessitating a different design from conventional per-column hash accumulators.

#### <span id="page-3-0"></span>3 Motivation

The write-back phase is one of the central bottlenecks of GPU-based SpMSpV. Existing strategies fall into two main categories, both with substantial drawbacks:

Atomic write-back suffers from severe address contention (many-to-one updates) and uncoalesced stores, which prevent effective utilization of global bandwidth. Sort-based write-back requires costly global sorts and large temporary buffers, resulting in prohibitively high overhead.

We benchmarked both strategies on an NVIDIA A100 (peak bandwidth 1555 GB/s). At input sparsity 0.1 on *it-2004*, atomic write-back sustains about 270 GB/s, while sort-and-reduce reaches only 43.3 GB/s during the sorting stage. Under global load balancing, these write-back stages dominate overall runtime: atomic write-back accounts for more than 30% of execution time, whereas sort-based write-back consumes over 70%. As input density increases, the bandwidth of atomic write-back drops further (e.g., 251 GB/s at sparsity 0.2), while sort-based remains nearly constant (~45 GB/s).

These limitations motivate us to explore a hash-based write-back scheme, inspired by hash based SpGEMM methods [12, 13, 25, 28, 36]. However, as noted in Section 1, SpM-SpV lacks SpGEMM's row partitioning: hashing alleviates only local conflicts, with remaining ones still handled by global atomics. Therefore, for hashing to be effective, the matrix must exhibit sufficient locality, and the hash computation must be lightweight enough to offset its overhead.

To examine these conditions, we conduct a case study on the *it-2004* matrix, a large-scale web graph. We analyze its locality, evaluate processing strategies that enhance it, and discuss techniques to amortize the extra cost of hash computation.

