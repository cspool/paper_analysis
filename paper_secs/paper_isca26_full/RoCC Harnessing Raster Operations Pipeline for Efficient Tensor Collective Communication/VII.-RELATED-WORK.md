# VII. RELATED WORK

1) Repurposing under-utilized GPU hardware components: GPGPU hardware's resource underutilization has attracted extensive research. Fung et al. [13] proposed to modify the textual unit for in-warp transactional memory to expand atomic compute capacity beyond the ROP. Jooybar et al. [21] proposed to repurpose the ROP unit as the deterministic commit unit. Kim et al. [25] repurposed idle registers for interim results; subsequent works [18], [20], [23], [24], [43], [58] leveraged register underutilization to reduce the area and power consumption of GPUs; Recent a few papers [5], [11], [15] repurposed RTAs for tree traversal and page-table

![](_page_12_Figure_9.jpeg)

Fig. 32: Speedup on diverse GPU architectures.

works; Lee et al. [28] employed stencil-test hardware for early termination in neural rendering. To the best of our knowledge, RoCC is the first work that repurposes ROP units for non-atomic, general-purpose collective communication.

- 2) CC Overlapping and Optimization: There have been extensive studies to optimize CC processing. Klenk et al. [26] integrated an in-switch aggregator for in-network reductions. Rashidi et al. [48] proposed an accelerator to offload AllReduce from NPU. Cho et al. [8] overlap tree-based reduction and broadcast with forward computation. Qin et al. [47] proposed to fuse adjacent communication operators in hybrid-parallel LLMs. Pati et al. [44] used DMA and PIM to offload reduction. RoCC provides a lightweight (leveraging existing hardware) and orthogonal (targeting intra-node speedup that can produce synergistic speedup with in-network reduction engines and algorithmic optimizations) solution.
- 3) Symmetric Memory Allocation: Similar concepts to our proposed symmetric tensor allocation have been used by commercial and academic solutions, such as OpenSHMEM [7], nvshmem [27], and BarreChord [12]. Unlike these, RoCC introduces symmetric memory to simplify the routing between ROPs in GPUs, without requiring virtual address-level symmetry.

