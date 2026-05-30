# A. Heterogeneous Architectures

In this section, we analyze the three heterogeneous architectures used in our evaluation. They include PEs from the SPADE and Sextans SpMM accelerators, and from PIUMA, which can support SpMM and other kernels. In total, we use four different PE pipelines, with diverse characteristics (e.g., out-of-order, in-order, SIMD, and scalar). Table III describes the PEs, which we call workers. The different workers access sparse and dense structures like in Figure 2.

We evaluate both on-chip and off-chip PEs, as well as different mechanisms to avoid data races. We selected three architectures to emphasize the generality of our approach across multiple settings. Figure 9 illustrates our three heterogeneous architectures. We describe them next.

TABLE III: Heterogeneous workers.

| Worker    | Worker<br>Type | Sparse<br>Format | Din<br>Reuse           | Dout<br>Reuse          |
|-----------|----------------|------------------|------------------------|------------------------|
| SPADE PE  | Cold           | COO-like         | None                   | Inter-tile             |
| Sextans   | Hot            | COO-like         | Intra-tile<br>(stream) | Inter-tile             |
| PIUMA MTP | Cold           | CSR-like         | None                   | Inter-tile             |
| PIUMA STP | Hot            | CSR-like         | Intra-tile<br>(stream) | Intra-tile<br>(demand) |

(a) SPADE-Sextans: SPADE (cold) and Sextans (hot) PEs are integrated in the same die and share the same memory controllers (Figure 9(a)). For simplicity, we do not equip the SPADE PEs with a 3-level cache subsystem as in [24]; we

only include a private L1 and a BBF. The SPADE PEs access the sparse input and dense output through the BBF, while they access the dense input through the L1. The SPADE PEs traverse the sparse matrix using an untiled row-ordered traversal (Figure 6(a)), while Sextans uses a tiled row-ordered traversal (Figure 6(b)). The SPADE PEs and the Sextans PEs can either operate in parallel or serially. To avoid data races when the SPADE and Sextans PEs are operating in parallel, we use two output buffers, one for each worker type. A Merger module is responsible for merging the output buffers at the end of execution. Note that in the serial operation mode, the Merger module is not used. To avoid data races between SPADE PEs, we closely follow the assignment and padding requirements discussed in the SPADE paper. For example, cold tiles in the same row panel are always assigned to the same SPADE PE.

- **(b) SPADE-Sextans+PCIe:** SPADE was proposed as an accelerator that is integrated in the same die as the CPU host. On the other hand, Sextans was proposed as a PCIe-attached accelerator. For our second architecture, we use on-chip SPADE PEs and an off-chip Sextans that accesses main memory through PCIe. We additionally assume that Sextans has enhanced computational throughput compared to the one in SPADE-Sextans.
- (c) PIUMA: This architecture consists of MTPs, STPs, DMA engines, and an Atomic engine, all sharing the same memory subsystem. The cold workers are the MTPs, which are latency tolerant and access memory on-demand. The hot workers are the STPs, which we equip with scratchpads and with DMA engines to increase their ability to exploit memory-level parallelism. The STPs access the sparse input on-demand and issue DMA descriptors for *Din* and *Dout*. *Din* is stored in the scratchpad. The Atomic engine enables MTPs and STPs to perform read-modify-write operations on the same memory location without suffering data races. As a result, the MTPs and STPs always operate in parallel, and we only employ the *MinTime Parallel* and *MinByte Parallel* heuristics (Figure 8).

#### B. Framework Implementation Details

HotTiles is implemented as a framework that is fully integrated into the software pipeline that generates the sparse matrix format for each accelerator. The HotTiles software is executed on the host of the heterogeneous architecture. It reads a sparse matrix from disk in MatrixMarket [12] file format. It then analyzes it and partitions it into hot and cold tiles transparently to the users. Finally, it generates the sparse

matrix formats. These formats, in combination with Din, can be directly accessed by the heterogeneous workers to execute the SpMM kernel. Alternatively, they can be stored for later use—e.g., they can be generated and used during GNN training and then saved and reused during GNN inference.

For the model, the users need to set the following traits of the heterogeneous architecture: the maximum computational throughput of each worker type (in GFLOP/s); the number of workers of each type; the size of the workers' scratchpads; the shared main memory bandwidth (in GB/s); the Din and Dout reuse types and the sparse matrix formats (Table I); and the way that each worker overlaps the SpMM tasks (Section IV-B).

The *visible latency per byte* (vis lat) for each worker type is hard to determine analytically. For this reason, a small number of profiling runs are executed using a set of small test matrices. In each run, only one worker type is used. After these runs, vis lat is automatically determined using a simple search method. The search objective is to minimize the error between the execution times predicted by our framework and the real execution times of the profiling runs. Note that this tuning process only needs to be done once when the *HotTiles* framework is first installed on a particular machine. The derived vis lat values can be reused for SpMM runs with different sparse matrices and thus do not contribute to the preprocessing overhead. In future work, we aim at employing sophisticated search or machine learning techniques (e.g., similar to [38]) to automatically set the values of vis lat and all the other heterogeneous architecture traits.

