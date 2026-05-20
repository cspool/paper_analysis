**==> picture [117 x 57] intentionally omitted <==**

# **VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs** 

## Yuchen Li 

Department of Computer Science and Technology, Tsinghua University 

Beijing, China liyuchen24@mails.tsinghua.edu.cn 

## Peng Qu 

Department of Computer Science and Technology, Beijing National Research Center for Information Science and Technology, Tsinghua University Beijing, China qp2018@mail.tsinghua.edu.cn 

## **Abstract** 

Sparse matrix-sparse vector multiplication (SpMSpV) is a core primitive in graph analytics and scientific computing, also arising in spiking neural networks for event-driven spike propagation. On GPUs, the performance of the prevalent and efficient SpMSpV paradigm is often bottlenecked by the write-back phase of accumulating non-zero multiply–accumulate results; its many-to-one index scatter pattern causes severe conflicts and poor bandwidth utilization on GPUs. We present VDHA, a GPU-based weighted SpMSpV kernel that leverages block-private hash tables for local aggregation, substantially reducing write conflicts and improving memory coalescing. To further amplify this benefit, we incorporate column splitting with lightweight reordering to expose more locality, and employ a fetch–compute– writeback pipeline to overlap hash computation with memory accesses. Extensive evaluation on over 300 matrices with more than 5 million nonzeros, including web-scale graphs (Konect/LAW) and scientific workloads (SuiteSparse), shows that VDHA consistently outperforms state-of-the-art baselines. On web graphs, it achieves a 1.41× geometric-mean speedup (up to 3.42×), while on SuiteSparse it delivers 1.13× (up to 2.55×). We also provide a lightweight predictive model that identifies matrices favorable to VDHA with 91.3% accuracy. 

This work is licensed under a Creative Commons Attribution 4.0 International License. _PPoPP ’26, Sydney, NSW, Australia_ 

© 2026 Copyright held by the owner/author(s). ACM ISBN 979-8-4007-2310-0/2026/01 https://doi.org/10.1145/3774934.3786447 

## Zhe Pan 

Department of Computer Science and Technology, Tsinghua University Beijing, China pz@mail.tsinghua.edu.cn 

## Youhui Zhang 

Department of Computer Science and Technology, Beijing National Research Center for Information Science and Technology, Tsinghua University Zhongguancun Laboratory Beijing, China zyh02@mail.tsinghua.edu.cn 

_**CCS Concepts:**_ • **Computing methodologies** → **Shared memory algorithms** ; • **Computer systems organization** → **Single instruction, multiple data** . 

## _**Keywords:**_ SpMSpV, sparse matrix, GPU, Hashing, SNN 

## **ACM Reference Format:** 

Yuchen Li, Zhe Pan, Peng Qu, and Youhui Zhang. 2026. VDHA: Vector-Driven Hash Aggregation for Sparse Matrix-Sparse Vector Multiplication on GPUs. In _Proceedings of the 31st ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming (PPoPP ’26), January 31 – February 4, 2026, Sydney, NSW, Australia._ ACM, New York, NY, USA, 14 pages. https://doi.org/10.1145/3774 934.3786447 

## **1 Introduction** 

The sparse matrix-sparse vector multiplication (SpMSpV) computes y = Ax with both the matrix A and the input vector x being sparse. SpMSpV is frequently used in machine learning [8] and scientific computing [3, 5, 32, 41]. It also serves as a fundamental primitive in graph analytics, underlying core algorithms such as breadth-first search (BFS), PageRank and personalized PageRank, and it is the algebraic backbone of many graph frameworks including GraphBLAS [15], Gunrock [34], GraphBLAST [39], GraphMat [33]. 

Beyond these domains, SpMSpV also appears in eventdriven workloads such as spiking neural networks (SNNs), where spike delivery can be naturally formulated as sparse matrix-sparse vector multiplication [9]. Moreover, both brain-inspired neural models and real-world graphs (e.g., social networks) are known to exhibit highly clustered, smallworld connectivity patterns [35], which create opportunities for exploiting locality in SpMSpV execution. 

SpMSpV can be implemented under two execution paradigms: Row-major methods traverse CSR rows and are 

259 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Yuchen Li, Zhe Pan, Peng Qu, and Youhui Zhang 

naturally aligned with CSR-based SpMV. Some implementations can be regarded as direct extensions of SpMV, obtained by adding value validation [33]. Alternatively, methods such as tileSpMSpV [18] and BerryBees [26] adopt bitmapcompressed frontiers together with masking of visited nodes, and are specifically optimized for unweighted BFS-style traversals. However, for _weighted_ SpMSpV, row-major traversal still scans all matrix rows regardless of vector sparsity, and the bitmask cannot avoid loading matrix indices. As a result, row-major methods cannot fully exploit vector sparsity. 

Column-major SpMSpV, in contrast, follows a vectordriven paradigm: the computation consists of a _fetch phase_ , which gathers matrix columns corresponding to nonzeros in the vector, and a _write-back phase_ , which uses column indices to update the result vector, essentially an index-scatter where multiple entries may map to the same position. 

On CPUs, representative studies include fgSpMSpV [10], work-efficient SpMSpV [2], and HAM-SpMSpV [37]. On GPUs, prior work has explored different _write-back_ methods and kernel selection: graph analytics frameworks (e.g., Gunrock [34]) use atomic instructions to directly handle write conflicts; FastSpMSpV [40] adopts a _sort–reduce_ approach to avoid conflicts, and Adaptive SpMSpV [20] selects write-back strategies (atomic vs. sort) based on matrix characteristics, while also adapting load-balancing granularity and switching to row-major SpMSpV or SpMV under dense vectors. 

However, we observe that in some cases the two prevalent write-back strategies—atomic updates and sort-based updates—both fail to achieve satisfactory bandwidth utilization: the former suffers from scattered index updates and frequent conflicts, while the latter relies on costly global sorting (see Section 3). 

Similar challenges arise in SpGEMM, where hashing has been used effectively to aggregate partial products and eliminate intra-row conflicts [12, 13, 25, 28, 36]. However, SpMSpV lacks the natural row partitioning of SpGEMM: instead of resolving conflicts only within a single row, it must handle all intermediate updates across the matrix. As a result, a hash table can only eliminate a portion of the write conflicts, while the remaining updates still require global atomic writes. This leads to two key questions: whether SpMSpV provides sufficient locality for hash aggregation, and whether the benefit of fewer write conflicts can outweigh the overhead of the hash table. 

To address these challenges, we propose a vector-driven hash-aggregation (VDHA) algorithm for _weighted_ SpMSpV (both the matrix and the input vector contain general weights) on GPUs. VDHA reduces write-back conflicts via local aggregation in shared memory, enhances locality through column decomposition with reordering, and reduces hash cost by pipelining computation with memory access. Concretely, we propose **VDHA** : 

- **Shared-memory hash aggregation.** Intermediate results are first accumulated in a shared-memory hash table and flushed only when the table becomes sufficiently full, reducing the write-back conflicts and promoting coalesced writes. 

- **Short/long-column decomposition with reordering.** We first classify columns by their length (the number of nonzeros) into short and long categories. Long columns are further split into smaller segments and reordered to improve locality and raise aggregation density, thereby maximizing the benefit of shared-memory accumulation. 

- **Overlapping memory and computation.** We design a pipeline that overlaps irregular global memory accesses with hash computation, effectively hiding hash computation latency behind memory stalls and making aggregation nearly free. 

To systematically evaluate VDHA, we consider two benchmarks. The first consists of over 100 large-scale web graphs from the Konect [19] and LAW [6, 7] collections, which are representative of _graph analytics workloads_ where weighted SpMSpV is most critical (e.g., PageRank and Personalized PageRank on web graphs). The second includes over 200 matrices from the SuiteSparse [11] collection, a widely used benchmark that covers diverse domains such as scientific computing, engineering, and optimization. Both benchmarks contain only matrices with at least 5 million nonzeros. Together, these datasets allow us to assess both the practical impact on real graph workloads and the generality across broader application scenarios. 

Across four vector sparsity levels (0.01, 0.05, 0.10, 0.20; defined as the fraction of nonzeros in the input vector), VDHA outperforms the _best-of-seven_ baselines (including cuSPARSE, two row-major SpMSpV kernels using value validation [30, 31], and the four representative column-major SpMSpV kernels from [20, 34, 40]), achieving geometricmean speedups of **1.41** × on Konect/LAW (up to **3.42** ×) and **1.13** × on SuiteSparse (up to **2.55** ×). 

**Contributions.** This paper makes the following contributions: 

- **VDHA algorithm.** By enhancing locality and reducing hashing overhead, we realize a practical and efficient hashbased solution for weighted SpMSpV on GPUs 

- **Systematic comparison.** We conduct a comprehensive evaluation against SOTA baselines, across over 100 realworld network graphs and over 200 scientific graphs with a wide range of vector sparsities, demonstrating consistent speedups. 

- **Lightweight Performance Prediction.** We provide a lightweight analysis method to quickly assess whether a matrix benefits from VDHA, facilitating its integration into adaptive frameworks. 

260 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

VDHA: Vector-Driven Hash Aggregation for SpMSpV on GPUs 

## **2 Background:** 

## **2.1 General-Purpose Graphics Processing Units:** 

_**Programming and execution model.**_ A GPU kernel is launched as a grid of thread blocks (CTAs), each consisting of multiple warps (typically 32 threads). Thread blocks are scheduled onto streaming multiprocessors (SMs), where per-block register and shared-memory usage determine the maximum number of resident warps (occupancy). 

_**Memory system.**_ Global memory provides high bandwidth but also long latency (hundreds of cycles). To mitigate this, each SM has a private L1 cache and a configurable partition of on-chip SRAM ( _shared memory_ ) that can serve either as additional cache capacity or as an explicitly managed scratchpad, backed by a large device-wide L2 cache. Equally important is _memory coalescing_ : global accesses are issued in cache-line transactions (e.g., 128 B), so contiguous, wellaligned warp accesses are merged into fewer transactions, while scattered addresses fragment into many small ones, wasting bandwidth and exposing latency. 

_**Latency hiding.**_ SMs keep many warps resident simultaneously; when one warp stalls on memory, the scheduler issues instructions from another ready warp. This fine-grained multithreading allows latency to be hidden as long as sufficient parallelism and occupancy are maintained. 

## **2.2 SpMSpV** 

SpMSpV can be organized under two paradigms: _matrixdriven (row-major)_ and _vector-driven (column-major)_ . Figure 1 illustrates the differences in their computation flows and work patterns. 

**==> picture [242 x 102] intentionally omitted <==**

**----- Start of picture text -----**<br>
1 0 1 1 bitmask<br>x0 x2 x3 x0 * x2 x3 value<br>×<br>a00 a01 a03 a00x0 a22x2 a03x3 ... a00 a01 a03 a00x0+a01·0+a03x3 ...<br>a11 a13 + + a13x3 = ... a11 a13 a11·0+a13x3 = ...<br>a22 a22x2 ... a22 a22x2 ...<br>a30 a33 a30x0 a33x3 ... a30 a33 a30x0+a33x3 ...<br>a) Col-major SpMSpV b) Row-major SpMSpV<br>**----- End of picture text -----**<br>


**Figure 1.** Comparison of SpMSpV paradigms. (a) Columnmajor: iterate over nonzero entries of the input vector, fetch columns and accumulate partial products into the output. (b) Row-major: iterate over matrix rows and use a bitmask to skip inactive vector entries. 

Matrix-driven SpMSpV approaches iterate over matrix rows (CSR), treating the input vector as dense. Algorithm 1 illustrates this matrix-driven paradigm. In this scheme, each nonzero in a row first checks the corresponding position in a bitmap to confirm whether the vector entry is active(line 6); 

if the entry is active, the value is then loaded(lines 7,8). BFSlike unweighted SpMSpV systems such as TileSpMSpV [18] and BerryBees [26] further incorporate an _output mask_ to avoid unnecessary updates to inactive result entries, thus reducing redundant work during traversal. 

## **Algorithm 1** Matrix-driven SpMSpV (CSR / pull) 

**Input:** _𝐴_ in CSR: (row_ptr, indices, values) with _𝑁_ rows; dense vector _𝑥_ with value array _𝑥_ and bitmask _𝑏𝑚_ ; **Output:** Result vector _𝑦_ 

- 1: **for all** _𝑟_ ← 0 **to** _𝑁_ **in parallel do** 2: _𝑠𝑡𝑎𝑟𝑡_ ← _𝑟𝑜𝑤_  𝑝𝑡𝑟_ [ _𝑟_ ] _,𝑒𝑛𝑑_ ← _𝑟𝑜𝑤_  𝑝𝑡𝑟_ [ _𝑟_ + 1] 3: _𝑟𝑒𝑠_ ← 0 4: **for** _𝑗_ ← _𝑠𝑡𝑎𝑟𝑡_ **to** _𝑒𝑛𝑑_ **do** 5: _𝑐𝑜𝑙_ ← _𝑖𝑛𝑑𝑖𝑐𝑒𝑠_ [ _𝑗_ ] 6: _𝑚𝑎𝑠𝑘_ ← _𝑏𝑚_ [ _𝑐𝑜𝑙_ ] 7: **if** mask **then** 8: _𝑟𝑒𝑠_ ← _𝑟𝑒𝑠_ ⊕( _𝑣𝑎𝑙𝑢𝑒𝑠_ [ _𝑗_ ] ⊗ _𝑥_ [ _𝑐𝑜𝑙_ ]) 9: _𝑦_ [ _𝑟_ ] ← _𝑟𝑒𝑠_ 

In contrast, the vector-driven paradigm iterates only over the nonzeros of x. Algorithm 2 illustrates the vector-driven paradigm of SpMSpV. For each active entry in x, the corresponding column of the CSC matrix is fetched (line 4, the fetch step), and the partial products (i.e., _𝑚𝑎𝑡_  𝑣𝑎𝑙_ ⊗ _𝑣_  𝑣𝑎𝑙_ ) are generated toward the result vector according to the row indices of the column. These partial results are then written back to y (line 7, the write_back step). Representative column-major SpMSpV approaches include FastSpMSpV [40] and GPU graph frameworks such as Gunrock [34]. Hybrid methods such as Adaptive SpMSpV [20] combine both paradigms depending on vector sparsity. 

## **Algorithm 2** Vector-driven SpMSpV (CSC / push) 

**Input:** _𝐴_ in CSC: (col_ptr, indices, values); _𝑥_ in sparse format: (idx, val); vector length _𝑛_ **Output:** Result vector _𝑦_ 

- 1: **for all** _𝑖_ ← 0 **to** _𝑛_ **in parallel do** 2: _𝑐𝑜𝑙_ ← _𝑖𝑑𝑥_ [ _𝑖_ ], _𝑣_  𝑣𝑎𝑙_ ← _𝑣𝑎𝑙_ [ _𝑖_ ] 3: _𝑠𝑡𝑎𝑟𝑡_ ← col_ptr[ _𝑐𝑜𝑙_ ], _𝑒𝑛𝑑_ ← col_ptr[ _𝑐𝑜𝑙_ + 1] 4: _𝑖𝑛𝑑_  𝑙𝑖𝑠𝑡_ , _𝑣𝑎𝑙_  𝑙𝑖𝑠𝑡_ = fetch( _𝑖𝑛𝑑𝑖𝑐𝑒𝑠, 𝑣𝑎𝑙𝑢𝑒𝑠,𝑖_ ) 5: **for** _𝑗_ ← 0 **to** _𝑙𝑒𝑛_ ( _𝑖𝑛𝑑_  𝑙𝑖𝑠𝑡_ ) **do** 6: _𝑖𝑛𝑑_ ← _𝑖𝑛𝑑_  𝑙𝑖𝑠𝑡_ [ _𝑗_ ], _𝑚𝑎𝑡_  𝑣𝑎𝑙_ ← _𝑣𝑎𝑙_  𝑙𝑖𝑠𝑡_ [ _𝑗_ ] 7: write_back( _𝑦_ [ _𝑖𝑛𝑑_ ] _,𝑚𝑎𝑡_  𝑣𝑎𝑙_ ⊗ _𝑣_  𝑣𝑎𝑙_ ) 

## _**Different kinds of work-balance methods (fetch stage).**_ 

In the fetch phase, each nonzero in the vector indexes a column of the CSC matrix and loads the corresponding _𝑖𝑛𝑑𝑖𝑐𝑒𝑠_ and _𝑣𝑎𝑙𝑢𝑒𝑠_ into _𝑖𝑛𝑑_  𝑙𝑖𝑠𝑡_ and _𝑣𝑎𝑙_  𝑙𝑖𝑠𝑡_ . Since column lengths are highly irregular, different load-balancing strategies are used to assign these column workloads to CTAs. We highlight three representative methods: 

261 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Yuchen Li, Zhe Pan, Peng Qu, and Youhui Zhang 

**(1) Direct-mapped.** Each active column is assigned to one CTA, with no prefix-scan overhead. This simple strategy is commonly used in implementations without fine-grained load balancing, such as NaiveSpMSpV [30]. 

**(2) Block-mapped.** Multiple short columns are grouped for a CTA, and a block-level prefix scan computes their combined nonzero count, as used in graph frameworks like Gunrock [34]. 

**(3) Global-mapped.** For each nonzero in the input vector, the corresponding column length is first recorded. A global inclusive scan over these column lengths yields the total number of matrix entries to be accessed, which is then evenly partitioned so that each CTA processes a segment of similar size, following methods used in merge-based SpMV [24, 31]. 

**==> picture [228 x 194] intentionally omitted <==**

**----- Start of picture text -----**<br>
Direct Map Block Map<br>Global Map<br>Block 0 Block 1<br>Block 2 Block 3<br>**----- End of picture text -----**<br>


**Figure 2.** Illustration of three load-balancing strategies in fetch step (matrix in CSC format). Cells with the same color represent column segments assigned to the same CTA. 

_**Different kinds of write-back (write-back stage).**_ In column-major SpMSpV, partial products ( _𝑣_  𝑣𝑎𝑙_ ⊗ _𝑚_  𝑣𝑎𝑙_ ) are accumulated into the output _𝑦_ [ _𝑖𝑛𝑑_ ] during the write-back stage, which naturally leads to many-to-one updates. Several strategies are commonly used to realize this write-back: 

**(1) Atomic write-back.** Directly accumulates each partial product into the output using global atomics. 

**(2) Sort-based write-back.** Buffers ( _row, val_ ) pairs, sorts them by row index, and reduces duplicates so that each row is written once, thereby avoiding global atomics. 

**(3) Hash-based aggregation.** Hash aggregation is a lightweight accumulation strategy widely used in sparse kernels such as hash-based SpGEMM [12, 13, 25, 28, 36]. Instead of emitting every update to global memory, partial products are inserted into a small shared-memory hash table. Each update is stored as a _(key, value)_ pair, where the key is the row index; collisions are resolved using simple schemes such 

as linear probing (i.e., probing the next slot until an empty or matching entry is found). 

In hash-based SpGEMM, each output column maintains its own small hash table, and its size can be estimated through lightweight precomputation. In contrast, SpMSpV produces a _single_ output vector, so all intermediate updates converge to one logical output column. This substantially increases the required aggregation capacity, necessitating a different design from conventional per-column hash accumulators. 

## **3 Motivation** 

The write-back phase is one of the central bottlenecks of GPU-based SpMSpV. Existing strategies fall into two main categories, both with substantial drawbacks: 

**Atomic write-back** suffers from severe address contention (many-to-one updates) and uncoalesced stores, which prevent effective utilization of global bandwidth. **Sort-based write-back** requires costly global sorts and large temporary buffers, resulting in prohibitively high overhead. 

We benchmarked both strategies on an NVIDIA A100 (peak bandwidth 1555 GB/s). At input sparsity 0 _._ 1 on _it-2004_ , atomic write-back sustains about 270 GB/s, while sort-andreduce reaches only 43.3 GB/s during the sorting stage. Under global load balancing, these write-back stages dominate overall runtime: atomic write-back accounts for more than 30% of execution time, whereas sort-based write-back consumes over 70%. As input density increases, the bandwidth of atomic write-back drops further (e.g., 251 GB/s at sparsity ∼ 0.2), while sort-based remains nearly constant ( 45 GB/s). 

These limitations motivate us to explore a hash-based write-back scheme, inspired by hash based SpGEMM methods [12, 13, 25, 28, 36]. However, as noted in Section 1, SpMSpV lacks SpGEMM’s row partitioning: hashing alleviates only local conflicts, with remaining ones still handled by global atomics. Therefore, for hashing to be effective, the matrix must exhibit sufficient locality, and the hash computation must be lightweight enough to offset its overhead. 

To examine these conditions, we conduct a case study on the _it-2004_ matrix, a large-scale web graph. We analyze its locality, evaluate processing strategies that enhance it, and discuss techniques to amortize the extra cost of hash computation. 

## **3.1 Write-back Locality** 

To quantify the exploitable write-back locality in practice, we introduce two complementary metrics: 

_**Local overlap ratio** 𝜌_ ( _𝑇_ ) _**.**_ With a hash table of capacity _𝑇_ , each partial product either accumulates into an existing entry or inserts a new one; once the table is full, a flush is triggered. Let _𝐹_ ( _𝑇_ ) denote the total number of flushed entries and _𝑁_ the total number of intermediate products. We define _𝜌_ ( _𝑇_ ) = 1 − _[𝐹]_[(] _[𝑇]_[)][[.]] 

_𝑁_[[.]] 

262 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

VDHA: Vector-Driven Hash Aggregation for SpMSpV on GPUs 

Intuitively, _𝜌_ ( _𝑇_ ) measures the fraction of updates that can be temporally aggregated before being written back. A higher _𝜌_ ( _𝑇_ ) indicates stronger temporal locality and fewer global write transactions. 

_**Coalescing factor** 𝛾_ _**.**_ On GPUs, memory accesses from threads within a warp are merged into aligned memory sections (e.g., 128B). Suppose _𝐴_ denotes the memory bytes to be written back, and _𝑀_ the number of 128 B transactions = _𝐴_ required to serve the warp-level loads. We define _𝛾 𝑀_ ·128[.] 

Intuitively, _𝛾_ measures the degree of spatial coalescing: it is the fraction of useful payload bytes within all memory transactions. A higher _𝛾_ indicates that consecutive threads tend to access consecutive memory locations, leading to fewer wasted bytes per transaction and higher effective bandwidth utilization. 

On _it-2004_ , we observe that with _𝑇_ = 2048, _𝜌_ reaches 51.0%, 35.0%, and 14.8% at vector density = 100%, 10%, and 1%, respectively. Meanwhile, _𝛾_ reaches 0.744, 0.499, and 0.280. 

Taken together, _𝜌_ and _𝛾_ provide a comprehensive view of locality: _𝜌_ quantifies temporal aggregation, while _𝛾_ quantifies spatial coalescing. 

## **3.2 Skewed Nonzero Distributions** 

Many real-world graphs (e.g., social networks) exhibit highly skewed length distributions: a small fraction of columns are extremely long, while the majority remain short. For example, on _it-2004_ matrix, only 1.4% of its columns (≥256 nonzeros) account for more than 70% of the total nonzeros, with an average length of 1403 compared to the overall mean of just 27.9. 

This extreme skew indicates that a small fraction of ultralong columns dominate the workload. Moreover, within each long column, row indices never overlap, meaning that hash aggregation provides no benefit when the column is larger than the hash table size. 

Existing SpMSpV kernels address irregular column lengths generally in two ways. A block-mapped strategy assigns equal numbers of columns to each CTA, while global balancing distributes all nonzeros evenly across CTAs using prefix sums and searches. Neither approach exploits the limited locality within long columns, so both fall short of improving hash effectiveness. 

Similar skew patterns have also been reported in RoDe [27], which alleviates imbalance by decomposing long rows into smaller segments. Building on this insight, we adopt a **split-and-reorder** strategy. We first split long columns into segments so that each can be processed within a block. In addition, we introduce a lightweight reordering step that aligns neighboring segments to enhance adjacency and improve hash reuse. Short columns are still handled directly with the block-mapped method. This design not only balances the workload across blocks but also strengthens locality, which is crucial for effective hash aggregation. 

On _it-2004_ , after decomposition and reordering, the local overlap ratio _𝜌_ ( _𝑇_ ) improves markedly: with _𝑇_ = 2048, _𝜌_ reaches 89.8%, 65.3%, and 23.8% at vector densities of 100%, 10%, and 1%, respectively; _𝛾_ also increases to 2.607, 0.863, and 0.294 in these three cases. Note that _𝛾_ may exceed 1, as hashing reduces total memory access in write-back; with good memory coalescing, the effective utilization per transaction can be greater than 100% 

These results demonstrate that splitting long columns and reordering their segments is effective in enhancing locality, making hash-based aggregation substantially more beneficial than with naive processing. Evaluating on benchmarks shows that the atomic-unit utilization decreases from 22.99% to 12.82% after applying these optimizations, indicating that far fewer global atomic conflicts reach the write-back stage. 

## **3.3 Irregular Memory Accesses** 

The conventional fetch–writeback flow of GPU SpMSpV is dominated by irregular global-memory behavior. On _it-2004_ , our NCU profiling shows that the atomic write-back stage accounts for over 45% of stall cycles, and nearly 90% of these stalls are long scoreboard waits on global memory. A long scoreboard wait occurs when a warp is blocked until a pending global-memory load or store completes, meaning that even high occupancy cannot hide the latency of uncoalesced loads and scattered stores. 

To address this, we restructure the execution into a threestage **fetch–compute–writeback** pipeline. Compared with the conventional two-stage design, the additional compute stage performs hash aggregation while the next tile of data is being fetched asynchronously (e.g., via cp.async). In this way, memory latency that would otherwise cause stalls is overlapped with useful hash computation, reducing the apparent cost of aggregation, which includes computing hash indices and resolving collisions within shared memory through linear probing. 

This design proves effective in practice. With the additional compute phase, the stall ratio drops from over 45% to about 15%, still dominated by long scoreboard stalls (68.9%), but with significantly fewer total stall cycles. These results confirm that irregular memory latency can be successfully exploited to mask the cost of hashing, making hash-based write-back practical with minimal cost. Consistently, we observe that the fetch–compute–writeback pipeline reduces the hash computation cost from 16.7% to 12.3%. 

In summary, through the above analysis, we address the two key questions posed in Section 1. First, vector processing increases graph locality, thereby improving the effectiveness of hash aggregation. Second, the cost of hashing can be largely hidden through pipelining with memory access. Taken together, these results indicate that hash-based aggregation is a promising and practical direction for SpMSpV on modern GPUs. 

263 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Yuchen Li, Zhe Pan, Peng Qu, and Youhui Zhang 

## **4 Method** 

We target a vector-driven SpMSpV with the matrix stored in CSC format and a sparse input vector. Figure 3 illustrates our overall workflow: (①) long columns are split into smaller segments, while short columns are directly mapped; (②) large segments are reordered to enhance locality; and (③) all segments are block-mapped to GPU SMs and aggregated in shared memory before flushing. 

Based on this workflow, our design consists of two coordinated stages, with an additional pipelined execution scheme to reduce hash computation costs: 

**(1) Vector processing.** The input vector is scanned to identify active columns, which are classified as short or long. Long columns are further split and reordered, producing segments that are then assigned to GPU blocks. 

**(2) Block-level aggregation.** Each block processes its assigned segment using a private hash table in shared memory. Partial products are accumulated locally and then flushed in partial order, reducing global write conflicts and improving coalescing efficiency. 

**(3) Fetch–compute–writeback pipeline.** To exploit the memory stalls observed in atomic write-back, we restructure the execution into a three-stage pipeline. During hash insertions and aggregations on the current tile, asynchronous copy instructions (e.g., cp.async) are issued to fetch the next tile from global memory. 

The following subsections provide detailed descriptions of these three stages. 

## **4.1 Vector Processing** 

In a vector-driven SpMSpV, each nonzero ( _idx, val_ ) in the input vector selects a column _𝑐_ of the CSC-stored matrix _𝐴_ . Its adjacency range is given by offset[c] to offset[c+1], so the column length is simply _len_ = offset[c+1] − offset[c], and the associated vector value is _𝑥𝑐_ . We therefore represent this column by a tuple ( _start, len,𝑥𝑐_ ). 

Columns shorter than LEN_THRES are treated as _small columns_ , while longer ones are _long columns_ . Each long column is further split into segments of at most SPLIT_SIZE nonzeros. For the _𝑖_ -th segment, the starting position is _start_i_ = _start_ + _𝑖_ · SPLIT_SIZE, and its length is _len_i_ = min( _len_ − _𝑖_ · SPLIT_SIZE _,_ SPLIT_SIZE), and the vector value remains _𝑥𝑐_ , where _𝑖_ is the segment index. For efficient processing, long-column segments are first aggregated at the CTA level and then appended contiguously to a global queue. 

After splitting, the segments from the same column remain consecutive and will be mapped to the same SM. However, these intra-column segments contain disjoint row indices and therefore provide no opportunity for aggregation. In contrast, segments from different columns may share overlapping rows. To expose this cross-column locality, we sort all segments by the row index of their first nonzero, so that segments covering nearby index ranges are placed together. This 

increases the likelihood of overlap across adjacent segments and substantially improves the effectiveness of hash-based aggregation, as illustrated in Figure 4. 

This reordering is lightweight: it sorts only the segment metadata rather than the nonzeros inside each segment. If the number of segments is _𝑆_ , the sort cost is _𝑂_ ( _𝑆_ log _𝑆_ ) with _𝑆_ ∼ nnzlong/SPLIT_SIZE, which is much smaller than sorting all nonzeros. We do not reorder small columns, since their number can be on the same order as the input vector length, and sorting them would incur disproportionate overhead. 

## **4.2 Hash aggregation** 

After vector processing, column segments are block-mapped to GPU SMs, where each CTA maintains a private sharedmemory hash table. The purpose of this table is twofold: (1) _Local aggregation_ — accumulate partial products for identical row indices within the CTA, eliminating intra-block conflicts and producing coalesced writes; (2) _Partially ordered flush_ — the hash table reorganizes scattered row updates into a more sequential layout; entries are written in the bucket order defined by the hash function, which improves spatial locality and reduces memory transactions during flushing. 

Unlike SpGEMM, SpMSpV cannot eliminate all conflicts because multiple CTAs inevitably update the same rows. Thus global atomics remain necessary, but local aggregation reduces their frequency substantially. To keep the hash efficient, we employ atomicCAS with linear probing. Unbounded probing may lead to long collision chains and warp divergence; therefore we cap the number of probes by a threshold FALLBACK_ITER. Once this limit is exceeded, the update falls back to a global atomic operation. 

Although these conflicts require atomic instructions when writing back to global memory, they also reduce the relative cost of hash fallback. This allows us to make a better trade-off between probing in the hash table and the fallback strategy. 

## **Algorithm 3** Hash-based insertion with fallback 

|1:|**function**Insert(_𝐻,𝑖𝑛𝑑, val_)|
|---|---|
|2:|_ℎ_←hash_func(_𝑖𝑛𝑑_)%_𝑇𝐴𝐵𝐿𝐸_𝑆𝐼𝑍𝐸_|
|3:|_cnt_ ←0|
|4:|**while**_cnt < 𝐹𝐴𝐿𝐿𝐵𝐴𝐶𝐾_𝐼𝑇𝐸𝑅_**do**|
|5:|_𝑜𝑙𝑑_←atomicCAS(&_𝐻.𝑘𝑒𝑦_[_ℎ_]_,_−1_,𝑖𝑛𝑑_)|
|6:|**if**_𝑜𝑙𝑑_==−1 **or** _𝑜𝑙𝑑_==_𝑖𝑛𝑑_**then**|
|7:|UpdateHash(_𝐻.𝑣𝑎𝑙_[_ℎ_]_, val_)|
|8:|**return**|
|9:|_ℎ_←next_hash(_ℎ_)|
|10:|_cnt_ ←_cnt_+1|
|11:|Fallback(_𝑖𝑛𝑑, val_)|



Alg. 3 illustrates the pseudocode for hash-based insertion, where _𝐻_ is the shared memory hash table and each update is a key–value pair ( _𝑖𝑛𝑑, 𝑣𝑎𝑙_ ). The starting hash position is computed using hash_func. Each thread attempts 

264 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

VDHA: Vector-Driven Hash Aggregation for SpMSpV on GPUs 

**==> picture [480 x 434] intentionally omitted <==**

**Figure 3.** Overview of the VDHA design. Columns are first classified by length; long columns are split into smaller segments (①) and further reordered to enhance locality (②). Both the reordered long segments and short columns are then block-mapped to GPU SMs for hash aggregation, and the aggregated results are finally written back atomically (③). 

to claim the slot via atomicCAS, where -1 denotes an empty entry. If atomicCAS returns -1 or the target index, the insertion succeeds and UpdateHash updates the value. Otherwise, next_hash continues probing. Once the probe count exceeds FALLBACK_ITER, the update falls back to a direct global atomic: the pair ( _𝑖𝑛𝑑, val_ ) is added to result[ _𝑖𝑛𝑑_ ] using atomicAdd. 

## **4.3 Compute pipeline** 

To further reduce the cost of hash computation, we implement the fetch–compute–writeback pipeline that overlaps global-memory fetches with local aggregation. The key idea 

is to use the stall time of irregular memory accesses to perform hash updates, thereby hiding much of the hashing cost. Algorithm 4 illustrates the execution flow with double buffering. The kernel executes the following steps: 

1. **Asynchronous fetch.** The next segment of indices and values is fetched from global memory into shared memory using cp.async, which proceeds in the background (line 5). 

2. **Hash aggregation.** While the next segment is being fetched, threads aggregate the current segment into the shared-memory hash table (line 7). 

3. **Synchronization and flush.** After the hash computation of the current segment finishes, threads issue cp.async. 

265 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Yuchen Li, Zhe Pan, Peng Qu, and Youhui Zhang 

**==> picture [228 x 180] intentionally omitted <==**

**Figure 4.** Column decomposition and reorder. Columns are transposed for visualization: each bar denotes a column segment, where the left edge marks the starting row index and the width indicates its nonzero count. Long columns are split into shorter segments (①) and then reordered by starting row index to improve locality (②). 

**Algorithm 4** Fetch–Compute–Flush pipeline 

**Input:** Column segments _𝑠𝑒𝑔_ ; two shared-memory buffers _𝑏𝑢𝑓_ [0] _,𝑏𝑢𝑓_ [1]; shared-memory hash table _𝐻_ **Output:** Result vector _𝑦_ 1: Fetch( _𝑠𝑒𝑔_ 0, _𝑏𝑢𝑓_ [0]) 2: Sync() 3: **for** _𝑖_ = 0 to _𝑁_ **do** segs 4: **if** _𝑖_ ≠ _𝑁𝑠𝑒𝑔𝑠_ − 1 **then** 5: Fetch( _𝑠𝑒𝑔𝑖_ +1, _𝑏𝑢𝑓_ [( _𝑖_ + 1)%2]) 6: _𝑖𝑛𝑑, 𝑣𝑎𝑙_ ← _𝑏𝑢𝑓_ [ _𝑖_ %2] 7: insert( _𝐻,𝑖𝑛𝑑, 𝑣𝑎𝑙_ ) 8: Sync() 9: **if** hash table full **or** _𝑖_ == _𝑁_ − 1 **then** _𝑠𝑒𝑔𝑠_ 10: Flush( _𝐻,𝑦_ ) 

wait_group to ensure that the next segment has been fully loaded into shared memory (line 2,8). Aggregated entries remain in the shared-memory hash table until it approaches capacity, at which point they are flushed to global memory in bulk before processing continues (line 10). 

This design enables a steady overlap: while segment _𝑡_ is being computed, segment _𝑡_ +1 is already in fetch, effectively reducing the cost of hash computation. 

## **5 Implementation** 

## **5.1 Hash Table Design and Operations** 

**Hash function and probing strategy:** We adopt a simple modulo hash ( _𝑖_ %table_size), which preserves the loworder bits of indices. Compared with multiplicative hashing, 

this design better aligns with locally clustered indices, improving memory coalescing when results are written back to global memory. To reduce collisions, we use linear probing with a fixed stride ( _ℎ_ + _𝐶_ )%table_size instead of the usual ( _ℎ_ +1) scheme. This reduces collision probability when nonzeros are locally distributed. 

**Hash table update policy:** If all items belong to the same column in one step (e.g., long-column segment), updates can be performed directly with H_val[idx] += val, avoiding atomics. Otherwise, atomic operations (atomicAdd) are necessary to ensure correctness. 

## **5.2 Parameter Choices** 

**Hash table size:** 2048 entries, sufficient for typical CTA aggregation. The difference between _𝜌_ (2048) and _𝜌_ (4096) is minor (for vector densities of 100%, 10%, and 1%, _𝜌_ (4096) is 91.6%, 67.8%, and 26.2%, while _𝜌_ (2048) reaches 89.8%, 65.3%, and 23.8%), a smaller table size facilitates higher occupancy. 

**Launch configuration:** On NVIDIA A100 (SM80), each SM provides up to 168 KB shared memory and can schedule up to 64 warps. A 2048-entry hash table consumes 16 KB (4B key + 4B value per entry), and double-buffering requires additional shared memory. Under these constraints, an SM can host about ∼8 CTAs concurrently. To ensure full warp utilization, each CTA must contain at least 64/8 = 8 warps, which corresponds to 256 threads per block. This configuration balances occupancy with per-CTA memory demands. 

**Column splitting and thresholds:** SPLIT_SIZE = 256, ensuring long columns are cut into segments that fit exactly within a block. LEN_THRES = 128, which ensures that long columns are handled with dedicated CTAs for balanced workload, while avoiding an excessive number of segments that would otherwise increase processing overhead. 

## **6 Evaluation** 

## **6.1 Experimental Setup** 

We evaluate our method on two categories of datasets: 

**SuiteSparse Matrix Collection:** This set includes sparse matrices from diverse domains such as scientific computing, engineering, and optimization. We selected square matrices with more than 5 million nonzeros, resulting in over 200 test cases. 

**Web-scale Network Graphs:** We further included large real-world graphs from Konect and LAW. Similarly, we only considered square matrices with more than 5 million nonzeros, yielding over 100 test cases. These datasets primarily represent social networks and web graphs. 

To provide a comprehensive evaluation, we compare our method against several representative baselines: 

**Row-major SpMV:** We adopt NVIDIA cuSPARSE as the representative implementation. 

**Row-major SpMSpV:** Following the Adaptive SpMSpV work [20], we implement value-validation variants on top of 

266 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

VDHA: Vector-Driven Hash Aggregation for SpMSpV on GPUs 

HolaSpMV [31] and NaiveSpMV [30], which are open-source. Each accessed entry of _𝑥_ is checked using a bit mask [33]. Other row-major designs such as tileSpMSpV [18] and BerryBees [26] are tailored for unweighted BFS with visited mask and are thus not applicable in our weighted setting. 

**Column-major SpMSpV:** We faithfully implement four column-major kernels described in Adaptive-SpMSpV. These cover both atomic and sort-based write-back, each with balanced and unbalanced scheduling variants. For clarity, we refer to them as _BlockSort_ , _BlockAtomic_ , _GlobalSort_ , and _GlobalAtomic_ in the following tables and discussion. Among them, _GlobalSort_ corresponds to the sort–reduce strategy of FastSpMSpV [40], while _BlockAtomic_ closely resembles the atomic-based implementation in Gunrock [34]. 

Together, these baselines cover both state-of-the-art rowmajor SpMV (cuSPARSE), row-major SpMSpV, and columnmajor SpMSpV, ensuring a broad and fair comparison. 

All experiments were conducted on a server equipped with an NVIDIA A100 GPU (40GB HBM2e memory) and an AMD EPYC 7742 CPU. We compiled all kernels and baselines with CUDA nvcc 12.5, enabling the -O3 optimization flag. Both our method and baselines were executed in the same software and hardware environment to ensure fairness. 

## **6.2 SpMSpV Performance** 

We evaluate the performance of our method with random input vectors at four sparsity levels (0 _._ 01, 0 _._ 05, 0 _._ 1, and 0 _._ 2). 

Figure 5 presents results on both **Web-scale graphs** (left) and **SuiteSparse matrices** (right). Each row corresponds to one sparsity level, thus yielding eight subfigures in total. 

To summarize across datasets, Table 1 and 2 report the geometric mean (GeoMean) speedup of our method over other baselines on Konect & LAW and SuiteSparse benchmark, as well as the maximum observed speedup and the fraction of cases where our method achieves the best performance. 

Across the four sparsity levels (0.01, 0.05, 0.1, 0.2), our method achieves an average speedup of 1.41× on Konect (up to 3.42×) and 1.13× on SuiteSparse (up to 2.55×). While the least favorable cases naturally yield lower gains due to dataset-specific structure, the minimum speedups observed are 0.57× on Konect and 0.54× on SuiteSparse. In addition, it attains the best performance in about 90.3% of Konect cases and 59.5% of SuiteSparse cases overall, confirming that the advantage is consistent across sparsity levels. A more fine-grained analysis is provided in Section 6.4. 

When examining individual baselines, we find that our advantage over cuSPARSE and row-major SpMSpV methods is more pronounced at low input sparsity. That is because both approaches perform redundant work proportional to the number of matrix nonzeros, which leads to inefficiency when the input vector is sparse. cuSPARSE executes computations insensitive to input sparsity, while row-major methods must traverse the entire matrix and access auxiliary mask 

**Table 1.** Speedup and best fraction on **Konect** and **LAW** datasets. 

|Baseline \ Sparsity|0.01|0.05|0.10|0.20|
|---|---|---|---|---|
|cuSPARSE|G-mean: 8.93<br>Max: 33.40|4.24<br>8.89|2.85<br>5.00|1.91<br>7.08|
|NaiveSpMSpV|G-mean: 91.66<br>Max: 3015.74|53.07<br>1464.87|38.90<br>990.00|26.93<br>658.16|
|HolaSpMSpV|G-mean: 7.41<br>Max: 27.05|3.69<br>8.43|2.49<br>5.01|1.64<br>3.01|
|BlockSort|G-mean: 8.61<br>Max: 56.23|8.86<br>53.65|8.44<br>37.48|8.03<br>35.81|
|GlobalSort|G-mean: 4.42<br>Max: 8.25|4.16<br>7.33|4.24<br>6.91|4.46<br>7.94|
|BlockAtomic|G-mean: 5.06<br>Max: 63.37|5.86<br>67.02|5.73<br>48.48|5.56<br>60.68|
|GlobalAtomic|G-mean: 1.47<br>Max: 2.85|1.48<br>4.46|1.53<br>5.35|1.63<br>6.11|
|Best of 7|G-mean: 1.38<br>Max: 2.85|1.42<br>3.42|1.44<br>3.17|1.41<br>2.37|
|Best%|92.9%|91.3%|90.5%|86.5%|



arrays to validate activity. In contrast, column-major SpMSpV methods scale more directly with the number of active nonzeros, so our relative speedup against these baselines remains roughly stable across sparsity levels. 

We further note that the impact of load balancing differs across strategies. Block-level methods (e.g., BlockAtomic, BlockSort) are sensitive to skewed nonzero distributions: some CTAs receive little work while others are heavily loaded, which explains why our method achieves very high maximum speedups over them. NaiveSpMSpV, which directly maps threads to rows without explicit balancing, suffers even more from workload skew and thus shows particularly poor performance. By contrast, global strategies (GlobalAtomic, GlobalSort) distribute work more evenly across CTAs and therefore avoid such extreme slowdowns. 

At the same time, we also observe that the gains vary significantly between web graphs and scientific matrices. To understand this gap, we next analyze how locality metrics affect VDHA’s performance and present a lightweight predictive model to estimate whether a matrix benefits from our method. 

## **6.3 Ablation and Sensitivity Studies** 

We conducted an ablation study and a sensitivity analysis to quantify the contribution of each optimization stage and evaluate parameter choices. 

As shown in Table 4, we compare hash, hash+split, and hash+split+reorder, with normalized performance of 0.689×, 0.947×, and 1.000×, respectively. The split step improves 

267 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Yuchen Li, Zhe Pan, Peng Qu, and Youhui Zhang 

**==> picture [464 x 423] intentionally omitted <==**

**Figure 5.** Efficient performance of our method versus baselines at different sparsity. Efficient performance is defined as _efficient NNZ_ divided by runtime, where _efficient NNZ_ denotes the number of matrix nonzeros multiplied by the input vector sparsity. Left: web-scale graphs (Konect & LAW). Right: SuiteSparse matrices. 

load balance for long columns. Without splitting, the hashonly variant suffers from severe workload skew: threads processing extremely long columns generate disproportionately many intermediate updates, which largely exceed the capacity of the per-CTA hash table and result in poor performance. The reorder step improves the reuse of long-column segments within the hash table. Without reordering, segments from a single long column do not overlap in the hash table, limiting the effectiveness of shared-memory aggregation. Reordering these segments increases cross-column overlap and enhances aggregation efficiency, as discussed in Section 3.2. 

We then examine the sensitivity of VDHA to the hashtable size and split size. Table 5 reports the normalized performance across all tested configurations. VDHA achieves its best performance at HASH_TABLE_SIZE = 2048 and SPLIT_SIZE = 256, while all tested configurations remain within 0.8215×–1.000× of peak performance. This parameter choice balances two factors: (i) providing enough sharedmemory capacity to maximize the benefits of aggregation, and (ii) avoiding excessive shared-memory usage that would reduce occupancy, as discussed in Section 5.2. 

268 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

VDHA: Vector-Driven Hash Aggregation for SpMSpV on GPUs 

**Table 2.** Speedup and best fraction on **SuiteSparse** datasets. 

|Baseline \ Sparsity|0.01|0.05|0.10|0.20|
|---|---|---|---|---|
|cuSPARSE|G-mean: 6.53<br>Max: 41.61|3.68<br>10.47|2.60<br>6.27|1.80<br>9.54|
|NaiveSpMSpV|G-mean: 10.80<br>Max: 11817.54|7.81<br>5613.71|6.28<br>3835.80|5.51<br>5200.53|
|HolaSpMSpV|G-mean: 5.51<br>Max: 31.88|3.18<br>9.31|2.26<br>6.29|1.80<br>10.14|
|BlockSort|G-mean: 6.41<br>Max: 55.39|6.05<br>57.59|5.84<br>53.56|5.71<br>73.18|
|GlobalSort|G-mean: 5.41<br>Max: 9.07|5.13<br>8.59|4.50<br>8.77|5.24<br>8.55|
|BlockAtomic|G-mean: 1.99<br>Max: 53.94|2.04<br>57.67|1.95<br>60.35|1.85<br>109.23|
|GlobalAtomic|G-mean: 1.64<br>Max: 4.81|1.64<br>8.42|1.66<br>9.92|1.73<br>11.01|
|Best of 7|G-mean: 1.20<br>Max: 2.55|1.17<br>2.29|1.11<br>2.11|1.05<br>2.15|
|Best%|68.9%|63.6%|56.5%|48.8%|



## **6.4 Performance Characterization and Predictive Modeling** 

To better understand when VDHA performs well—and why it may underperform on certain matrices—we conduct a matrix-level performance characterization on representative matrices. Based on these observations, we further develop a lightweight predictive model to quickly determine whether or not VDHA is beneficial for a given input. Table 3 summarizes the selected datasets, including their dimensions, number of nonzeros (nnz), and structural characteristics( _𝜌_ and _𝛾_ ). 

_**Impact of Locality and Coalescing.**_ We jointly analyze the role of the local aggregation rate _𝜌_ and the coalescing factor _𝛾_ . A larger _𝜌_ means more updates can be absorbed into the shared-memory hash table, reducing the number of global writes. At the same time, a larger _𝛾_ reflects more coalesced memory transactions, lowering bandwidth waste. Together, high _𝜌_ and high _𝛾_ mean that many updates are aggregated and written back in contiguous sections, which directly translates into higher effective bandwidth. 

Table 3 shows representative cases. Web graphs such as _it-2004_ and _sk-2005_ exhibit strong temporal locality ( _𝜌_ ) and irregular accesses that are greatly improved after reordering (better _𝛾_ ), leading to substantial speedups. 

In contrast, some matrices exhibit near-diagonal nonzero structures, such as _atmosmodl_ and _G3_circuit_ . For these matrices, VDHA expansion generates few overlapping updates, resulting in low _𝜌_ and limited improvement in _𝛾_ . Moreover, when nonzeros are distributed in a highly regular fashion, 

our vector processing introduces little benefit but adds overhead, making VDHA less effective. Therefore, our approach is less advantageous on diagonal-like structure. 

_**Toward Fast Predictors.**_ While _𝜌_ and _𝛾_ explain performance behavior well, they are runtime metrics and cannot be obtained without executing the kernel. To enable lightweight prediction, we introduce two simple structural statistics that can be computed directly from the matrix. 

We denote the two structural indicators as (i) _bandwidth index 𝐵_ , defined as the average distance between the topmost and bottommost nonzeros per column in the CSC matrix, and (ii) _variance index 𝑉_ , defined as the column-wise variance of nonzeros. Together with the number of rows, the total number of nonzeros, and the input vector sparsity, we train a simple decision tree classifier following the methodology of Adaptive SpMSpV [20]. We use 70% of matrices for training and 30% for testing. On the test set, the decision tree achieves **91.3%** accuracy (measured by F1 score). 

Furthermore, if we fall back to a naive implementation (BlockAtomic) when the predictor estimates VDHA to be suboptimal, the geometric-mean speedup on the SuiteSparse dataset across all four vector sparsities improves from **1.13** × to **1.16** ×. If fallback to the best among all seven baselines (best-of-7), the adaptive scheme further achieves **1.22** × speedup. 

Our predictor uses five lightweight structural features (num_rows, num_nnzs, vector sparsity, bandwidth index _𝐵_ , and variance index _𝑉_ ) and achieves good accuracy. Adding more features (e.g., Adaptive SpMSpV uses 13 matrix and vector features) may improve accuracy further but increases extraction overhead, presenting a trade-off for future work. 

## **7 Related Works** 

The optimization of SpMSpV has been driven primarily by graph analytics workloads. GPU frameworks such as Gunrock [34], GraphBLAST [39], push-pull library [38], GraphLab [22], MultiGraph [17], Graphpad [1], Ligra [29], and GSwitch [23] incorporate SpMSpV as a core primitive to accelerate fundamental applications including BFS, PageRank, and personalized PageRank. Beyond frameworks, dedicated GPU kernels have also been proposed. Approaches such as TileSpMSpV [18] and BerryBees [26] specifically target unweighted BFS by employing a tiled format with output masking, in which frontier vectors are binary and results are accumulated using atomicOr. BerryBees further exploits bit-level tensor cores on recent GPUs to accelerate these operations. 

Other works developed more general-purpose kernels: FastSpMSpV [40] introduced a reduce-based method to avoid atomics via global reducing, while Adaptive SpMSpV [20] selected among multiple kernels (row/col-major, atomic- or 

269 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

Yuchen Li, Zhe Pan, Peng Qu, and Youhui Zhang 

**Table 3.** Statistics and locality/coalescing metrics of representative datasets. 

|Metric|it-2004|sk-2005|mycielskian19|inline_1|delaunay_n24|roadNet-CA|atmosmodl|G3_circuit|
|---|---|---|---|---|---|---|---|---|
|Group|LAW|LAW|Mycielski|GHS_psdef|DIMACS10|SNAP|Bourchtein|AMD|
|Rows|41.2M|50.6M|393K|503K|16.7M|1.97M|1.49M|1.59M|
|NNZs|1.15B|1.95B|903M|36.8M|101M|5.53M|10.3M|7.66M|
|_𝜌_0_._1(2048)|0.665|0.664|0.491|0.472|0.137|0.085|0.126|0.131|
|_𝛾_0_._1(2048)|0.793|0.617|0.399|0.490|0.199|0.149|0.196|0.189|
|Speedup|1.69×|1.92×|1.59×|1.74×|0.91×|0.68×|0.65×|0.65×|
|Thumbnail|||||||||



**Table 4.** Ablation study of individual optimization components. Performance is normalized to the full VDHA pipeline (hash + split + reorder). 

|Method|Normalized Performance|
|---|---|
|Hash only|0.689×|
|Hash + split|0.947×|
|Hash + split + reorder|1.000×|



**Table 5.** Parameter sensitivity of VDHA under different hash-table and split sizes (normalized average speedup). 

RoDe [27] highlighted that real-world matrices often exhibit highly skewed nonzero distributions, and addressed this by decomposing matrices into a regular part and a residual part processed separately. 

A large body of SpGEMM research has focused on handling the many-to-one write-back of intermediate products. Hash-based methods [12, 13, 25, 28, 36] use shared-memory hash tables to temporarily store and combine intermediate products before writing them back. Sort-based [4, 21] or merge-based approaches [16] first generate candidate triples and then sort or merge them to accumulate results in order. 

|**Split size**|**Hash-table size**|
|---|---|
||**1024**<br>**2048**<br>**3072**<br>**4096**|
|64<br>128<br>256<br>512|0.9779<br>0.9346<br>0.8831<br>0.8215<br>0.9254<br>0.9449<br>0.9109<br>0.8818<br>0.9784<br>**1.0000**<br>0.9848<br>0.9704<br>0.8953<br>0.9225<br>0.9031<br>0.9329|



sort-based, different load-balancing strategies) using heuristics on matrix statistics. These efforts highlight the challenges of avoiding write conflicts and balancing workloads. 

On CPUs, related efforts such as HAM-SpMSpV [37], workefficient SpMSpV [2], and Regu2D-SpMV [14] demonstrate efficient sparse computations by leveraging cache locality, vectorization, and work-efficient load balancing. 

Besides being studied directly, SpMSpV can also be viewed as a special case of SpMV, SpMM, or SpGEMM. GraphMat [33] observed that SpMSpV can be implemented on top of SpMV by adding a lightweight bitmask to validate vector entries before reading values, thereby reducing some unnecessary memory accesses. HOLA-SpMV [31] and Naive-SpMV [30] employ two fundamentally different loadbalancing strategies. HOLA-SpMV uses lightweight global balancing to equalize CTA workloads, whereas Naive-SpMV avoids balancing entirely, incurring no overhead but relying on GPU parallelism to mask imbalance. 

## **8 Conclusion** 

This paper presents VDHA, a GPU-based SpMSpV algorithm targeting the costly write-back problem. VDHA combines long-column decomposition with reordering, sharedmemory hash aggregation, and a fetch–compute–writeback pipeline to improve locality, reduce conflicts, and reduce hash costs. Experiments on over 300 SuiteSparse and web-scale matrices with more than 5 million nonzeros show consistent gains over state-of-the-art baselines, with up to 3.42× speedup (1.41× on average) on web graphs and up to 2.55× (1.13× on average) on scientific matrices. We further propose a lightweight analysis method to predict when VDHA is beneficial, achieving 91.3% accuracy. 

## **Acknowledgments** 

This work was supported by the National Natural Science Foundation of China under Grant No. 62250006, the National Key Research and Development Program of China under Grant 2025YFB3003200, The Tsinghua University Initiative Scientific Research Program, under Grant No.2022Z11ZRB002, The Suzhou-Tsinghua Innovation Leadership Program, under Grant No.20222002100, The Jiangsu Provincial Science and Technology Program, Grant No. BE2023005-3. 

270 

PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

VDHA: Vector-Driven Hash Aggregation for SpMSpV on GPUs 

## **References** 

- [1] Michael J Anderson, Narayanan Sundaram, Nadathur Satish, Md Mostofa Ali Patwary, Theodore L Willke, and Pradeep Dubey. 2016. Graphpad: Optimized graph primitives for parallel and distributed platforms. In _2016 IEEE International Parallel and Distributed Processing Symposium (IPDPS)_ . IEEE, 313–322. doi:10.1109/IPDPS.2016.86 

- [2] Ariful Azad and Aydin Buluç. 2017. A work-efficient parallel sparse matrix-sparse vector multiplication algorithm. In _2017 IEEE International Parallel and Distributed Processing Symposium (IPDPS)_ . IEEE, 688–697. doi:10.1109/IPDPS.2017.76 

- [3] Ariful Azad, Mathias Jacquelin, Aydin Buluç, and Esmond G. Ng. 2017. The Reverse Cuthill-McKee Algorithm in Distributed-Memory. In _2017 IEEE International Parallel and Distributed Processing Symposium (IPDPS)_ . 22–31. doi:10.1109/IPDPS.2017.85 

- [4] Nathan Bell and Michael Garland. 2012. Cusp: Generic parallel algorithms for sparse matrix and graph computations. _Version 0.3. 0_ 35 (2012). 

- [5] Thierry P. Berger, Julien Francq, Marine Minier, and Gaël Thomas. 2016. Extended Generalized Feistel Networks Using Matrix Representation to Propose a New Lightweight Block Cipher: Lilliput. _IEEE Trans. Comput._ 65, 7 (2016), 2074–2089. doi:10.1109/TC.2015.2468218 

- [6] Paolo Boldi, Marco Rosa, Massimo Santini, and Sebastiano Vigna. 2011. Layered Label Propagation: A MultiResolution Coordinate-Free Ordering for Compressing Social Networks. In _Proceedings of the 20th international conference on World Wide Web_ , Sadagopan Srinivasan, Krithi Ramamritham, Arun Kumar, M. P. Ravindra, Elisa Bertino, and Ravi Kumar (Eds.). ACM Press, 587–596. doi:10.1145/1963405.1963488 

- [7] Paolo Boldi and Sebastiano Vigna. 2004. The WebGraph Framework I: Compression Techniques. In _Proc. of the Thirteenth International World Wide Web Conference (WWW 2004)_ . ACM Press, Manhattan, USA, 595–601. doi:10.1145/988672.988752 

- [8] Chih-Chung Chang and Chih-Jen Lin. 2011. LIBSVM: A library for support vector machines. _ACM Trans. Intell. Syst. Technol._ 2, 3, Article 27 (May 2011), 27 pages. doi:10.1145/1961189.1961199 

- [9] Jiajie Chen, Le Yang, and Youhui Zhang. 2022. GaBAN: a generic and flexibly programmable vector neuro-processor on FPGA. In _Proceedings of the 59th ACM/IEEE Design Automation Conference_ (San Francisco, California) _(DAC ’22)_ . Association for Computing Machinery, New York, NY, USA, 931–936. doi:10.1145/3489517.3530561 

- [10] Yuedan Chen, Guoqing Xiao, Kenli Li, Francesco Piccialli, and Albert Y Zomaya. 2022. fgSpMSpV: A fine-grained parallel SpMSpV framework on HPC platforms. _ACM Transactions on Parallel Computing_ 9, 2 (2022), 1–29. doi:10.1145/3512770 

- [11] Timothy A Davis. 2019. Algorithm 1000: SuiteSparse: GraphBLAS: Graph algorithms in the language of sparse linear algebra. _ACM Transactions on Mathematical Software (TOMS)_ 45, 4 (2019), 1–25. doi:10 .1145/3322125 

- [12] Julien Demouth. 2012. Sparse matrix-matrix multiplication on the gpu. In _Proceedings of the GPU technology conference_ , Vol. 3. 

- [13] Zhaoyang Du, Yijin Guan, Tianchan Guan, Dimin Niu, Linyong Huang, Hongzhong Zheng, and Yuan Xie. 2022. OpSparse: a highly optimized framework for sparse general matrix multiplication on GPUs. _IEEE Access_ 10 (2022), 85960–85974. doi:10.1109/ACCESS.2022.3196940 

- [14] Xiang Fei and Youhui Zhang. 2021. Regu2D: Accelerating Vectorization of SpMV on Intel Processors through 2D-partitioning and Regular Arrangement. In _Proceedings of the 50th International Conference on Parallel Processing_ (Lemont, IL, USA) _(ICPP ’21)_ . Association for Computing Machinery, New York, NY, USA, Article 77, 11 pages. doi:10.1145/3472456.3472479 

- [15] John R Gilbert, Steve Reinhardt, and Viral B Shah. 2006. Highperformance graph algorithms from parallel sparse matrices. In _International Workshop on Applied Parallel Computing_ . Springer, 260–269. 

- [16] Felix Gremse, Andreas Hofter, Lars Ole Schwen, Fabian Kiessling, and Uwe Naumann. 2015. GPU-accelerated sparse matrix-matrix 

   - multiplication by iterative row merging. _SIAM Journal on Scientific Computing_ 37, 1 (2015), C54–C71. doi:10.1137/130948811 

- [17] Changwan Hong, Aravind Sukumaran-Rajam, Jinsung Kim, and P Sadayappan. 2017. MultiGraph: Efficient graph processing on GPUs. In _2017 26th International Conference on Parallel Architectures and Compilation Techniques (PACT)_ . IEEE, 27–40. doi:10.1109/PACT.2017.48 

- [18] Haonan Ji, Huimin Song, Shibo Lu, Zhou Jin, Guangming Tan, and Weifeng Liu. 2023. TileSpMSpV: A Tiled Algorithm for Sparse MatrixSparse Vector Multiplication on GPUs. In _Proceedings of the 51st International Conference on Parallel Processing_ (Bordeaux, France) _(ICPP ’22)_ . Association for Computing Machinery, New York, NY, USA, Article 9, 11 pages. doi:10.1145/3545008.3545028 

- [19] Jérôme Kunegis. 2013. Konect: the koblenz network collection. In _Proceedings of the 22nd international conference on world wide web_ . 1343–1350. doi:10.1145/2487788.2488173 

- [20] Min Li, Yulong Ao, and Chao Yang. 2021. Adaptive SpMV/SpMSpV on GPUs for Input Vectors of Varied Sparsity. _IEEE Transactions on Parallel and Distributed Systems_ 32, 7 (2021), 1842–1853. doi:10.1109/ TPDS.2020.3040150 

- [21] Weifeng Liu and Brian Vinter. 2014. An efficient GPU general sparse matrix-matrix multiplication for irregular data. In _2014 IEEE 28th international parallel and distributed processing symposium_ . IEEE, 370–381. doi:10.1109/IPDPS.2014.47 

- [22] Yucheng Low, Danny Bickson, Joseph Gonzalez, Carlos Guestrin, Aapo Kyrola, and Joseph M. Hellerstein. 2012. Distributed GraphLab: a framework for machine learning and data mining in the cloud. _Proc. VLDB Endow._ 5, 8 (April 2012), 716–727. doi:10.14778/2212351.2212354 

- [23] Ke Meng, Jiajia Li, Guangming Tan, and Ninghui Sun. 2019. A pattern based algorithmic autotuner for graph processing on GPUs. In _Proceedings of the 24th Symposium on Principles and Practice of Parallel Programming_ . 201–213. doi:10.1145/3293883.3295716 

- [24] Duane Merrill and Michael Garland. 2016. Merge-based sparse matrixvector multiplication (spmv) using the csr storage format. _Acm Sigplan Notices_ 51, 8 (2016), 1–2. doi:10.1145/3016078.2851190 

- [25] Yusuke Nagasaka, Akira Nukada, and Satoshi Matsuoka. 2017. Highperformance and memory-saving sparse general matrix-matrix multiplication for nvidia pascal gpu. In _2017 46th International Conference on Parallel Processing (ICPP)_ . IEEE, 101–110. doi:10.1109/ICPP.2017.19 

- [26] Yuyao Niu and Marc Casas. 2025. BerryBees: Breadth first search by bit-tensor-cores. In _Proceedings of the 30th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming_ . 339–354. doi:10.1145/3710848.3710859 

- [27] Meng Pang, Xiang Fei, Peng Qu, Youhui Zhang, and Zhaolin Li. 2024. A row decomposition-based approach for sparse matrix multiplication on GPUs. In _Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming_ . 377–389. doi:10.114 5/3627535.3638470 

- [28] Mathias Parger, Martin Winter, Daniel Mlakar, and Markus Steinberger. 2020. Speck: Accelerating gpu sparse matrix-matrix multiplication through lightweight analysis. In _Proceedings of the 25th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming_ . 362–375. doi:10.1145/3332466.3374521 

- [29] Julian Shun and Guy E Blelloch. 2013. Ligra: a lightweight graph processing framework for shared memory. In _Proceedings of the 18th ACM SIGPLAN symposium on Principles and practice of parallel programming_ . 135–146. doi:10.1145/2517327.2442530 

- [30] Markus Steinberger, Andreas Derlery, Rhaleb Zayer, and Hans-Peter Seidel. 2016. How naive is naive SpMV on the GPU?. In _2016 IEEE High Performance Extreme Computing Conference (HPEC)_ . IEEE, 1–8. doi:10.1109/HPEC.2016.7761634 

- [31] Markus Steinberger, Rhaleb Zayer, and Hans-Peter Seidel. 2017. Globally homogeneous, locally adaptive sparse matrix-vector multiplication on the GPU. In _Proceedings of the International Conference on Supercomputing_ . 1–11. doi:10.1145/3079079.3079086 

271 

Yuchen Li, Zhe Pan, Peng Qu, and Youhui Zhang 

## PPoPP ’26, January 31 – February 4, 2026, Sydney, NSW, Australia 

- [32] Liang Sun, Shuiwang Ji, and Jieping Ye. 2009. A least squares formulation for a class of generalized eigenvalue problems in machine learning. In _Proceedings of the 26th Annual International Conference on Machine Learning_ (Montreal, Quebec, Canada) _(ICML ’09)_ . Association for Computing Machinery, New York, NY, USA, 977–984. doi:10.1145/1553374.1553499 

- [33] Narayanan Sundaram, Nadathur Satish, Md Mostofa Ali Patwary, Subramanya R. Dulloor, Michael J. Anderson, Satya Gautam Vadlamudi, Dipankar Das, and Pradeep Dubey. 2015. GraphMat: high performance graph analytics made productive. _Proc. VLDB Endow._ 8, 11 (July 2015), 1214–1225. doi:10.14778/2809974.2809983 

- [34] Yangzihao Wang, Andrew Davidson, Yuechao Pan, Yuduo Wu, Andy Riffel, and John D Owens. 2016. Gunrock: A high-performance graph processing library on the GPU. In _Proceedings of the 21st ACM SIGPLAN symposium on principles and practice of parallel programming_ . 1–12. doi:10.1145/2851141.2851145 

- [35] Duncan J. Watts and Steven H. Strogatz. 1998. Collective dynamics of ‘small-world’ networks. _Nature_ 393, 6684 (01 Jun 1998), 440–442. doi:10.1038/30918 

- [36] Min Wu, Huizhang Luo, Fenfang Li, Yiran Zhang, Zhuo Tang, Kenli Li, Jeff Zhang, and Chubo Liu. 2025. HSMU-SpGEMM: Achieving High Shared Memory Utilization for Parallel Sparse General Matrix-Matrix Multiplication on Modern GPUs. In _2025 IEEE International Symposium on High Performance Computer Architecture (HPCA)_ . IEEE, 1452–1466. 

doi:10.1109/HPCA61900.2025.00109 

- [37] Lei Xu, Haipeng Jia, Yunquan Zhang, Luhan Wang, and Xianmeng Jiang. 2024. HAM-SpMSpV: an Optimized Parallel Algorithm for Masked Sparse Matrix-Sparse Vector Multiplications on multi-core CPUs. In _Proceedings of the 33rd International Symposium on HighPerformance Parallel and Distributed Computing_ (Pisa, Italy) _(HPDC ’24)_ . Association for Computing Machinery, New York, NY, USA, 160–173. doi:10.1145/3625549.3658680 

- [38] Carl Yang, Aydın Buluç, and John D Owens. 2018. Implementing pushpull efficiently in GraphBLAS. In _Proceedings of the 47th International Conference on Parallel Processing_ . 1–11. doi:10.1145/3225058.3225122 

- [39] Carl Yang, Aydın Buluç, and John D Owens. 2022. GraphBLAST: A high-performance linear algebra-based graph framework on the GPU. _ACM Transactions on Mathematical Software (TOMS)_ 48, 1 (2022), 1–51. doi:10.1145/3466795 

- [40] Carl Yang, Yangzihao Wang, and John D Owens. 2015. Fast sparse matrix and sparse vector multiplication algorithm on the GPU. In _2015 IEEE International Parallel and Distributed Processing Symposium Workshop_ . IEEE, 841–847. doi:10.1109/IPDPSW.2015.77 

- [41] Alwin Zulehner and Robert Wille. 2019. Matrix-Vector vs. MatrixMatrix Multiplication: Potential in DD-based Simulation of Quantum Computations. In _2019 Design, Automation & Test in Europe Conference & Exhibition (DATE)_ . 90–95. doi:10.23919/DATE.2019.8714836 

Received 2025-09-01; accepted 2025-11-10 

272 

