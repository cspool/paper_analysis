# *A. Architectures and simulation*

We evaluate our architectures through simulation. For SPADE-Sextans, we use SST [57] and DRAMSim3 [46]. For PIUMA, we use an in-house simulator based on Sniper [16], [17]. We preprocess the sparse matrices to generate the sparse matrix formats on a dual-socket 48-core Intel Xeon Platinum 8260M CPU [35].

For the SPADE-Sextans archiecture, we test different system scales. We keep the PE frequency and the memory bandwidth constant, but we change: (1) the number of SPADE PEs and (2) the computational throughput and the scratchpad size of Sextans. The memory bandwidth is kept constant to evaluate our approach under different worker to memory bandwidth ratios.

Table IV displays the parameters for the different system scales. We use the system scale 4 as the baseline scale. In all scales, the cache line size is 64 bytes, the PE frequency is 0.8 GHz, and the main memory bandwidth is 205 GB/s. Note that this memory bandwidth is the maximum theoretical value that the memory controllers can provide. The maximum observed bandwidth for the heterogeneous architecture is 161 GB/s at system scale 8. For the SPADE PE out-of-order pipeline, we use the parameters in the original paper. Although SPADE supports different execution strategies, for simplicity, we set the SPADE PEs to execute an untiled sparse matrix traversal (Figure 6(a)), with each PE operating on a chunk of 64 continuous sparse matrix rows at a time. SPADE PEs use an untiled COO format, while Sextans operates on a tiled COO format. Sparse and dense matrix values are stored using single-precision floating point.

TABLE IV: Architectural parameters for different SPADE-Sextans system scales. System scale 4 is the base one.

|                 | SPADE      |                        |                    | Sextans    |                        |                      |
|-----------------|------------|------------------------|--------------------|------------|------------------------|----------------------|
| System<br>Scale | Num<br>PEs | SIMD<br>MACs/<br>Cycle | L1<br>Size<br>(kB) | Num<br>PEs | SIMD<br>MACs/<br>Cycle | SPAD<br>Size<br>(MB) |
| 1               | 4          | 1                      | 32                 | 1          | 5                      | 0.5                  |
| 2               | 8          | 1                      | 32                 | 1          | 10                     | 1                    |
| 4               | 16         | 1                      | 32                 | 1          | 20                     | 2                    |
| 8               | 32         | 1                      | 32                 | 1          | 40                     | 4                    |

For the SPADE-Sextans+PCIe architecture, we assume that Sextans is connected to the on-chip memory bus through PCIe, which has a maximum bandwidth of 32GB/s. We use this architecture to evaluate gSpMM versions with higher arithmetic intensity. As we increase the kernel arithmetic intensity, more SIMD operations are required per nonzero. As a result, the SPADE PEs require more cycles to execute all the arithmetic operations per nonzero. However, we assume that the off-chip Sextans increases its computational power beyond the numbers in Table IV. Specifically, for system scale 4, Sextans can now process 20 nonzeros per cycle, regardless of the arithmetic intensity of the kernel. All the other system parameters are kept the same.

For the PIUMA architecture, we use 4 MTPs and 2 STPs. The STPs use a tiled CSR-like format, while the MTPs use untiled CSR. We choose to store the sparse and dense matrix values using double-precision floating point to show that the method generalizes to various data sizes and types.

