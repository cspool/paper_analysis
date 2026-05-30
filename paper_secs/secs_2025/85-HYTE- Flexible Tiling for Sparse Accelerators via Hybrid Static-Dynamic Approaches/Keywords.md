# Keywords

sparse tensor algebra, hardware acceleration, tiling

#### **ACM Reference Format:**

Xintong Li, Zhiyao Li, and Mingyu Gao. 2025. HYTE: Flexible Tiling for Sparse Accelerators via Hybrid Static-Dynamic Approaches. In *Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA* 

![](_page_0_Picture_17.jpeg)

This work is licensed under a Creative Commons Attribution 4.0 International License. *ISCA '25, Tokyo, Japan*© 2025 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-1261-6/25/06
https://doi.org/10.1145/3695053.3731044

## 1 Introduction

Sparse tensor data are prominently used in many domains including graph processing, high-performance computing, and machine learning. Due to their irregular data distributions, sparse tensor computations are usually inefficient on general-purpose processors, causing numerous random data accesses with little locality in the memory hierarchy, as well as severe load imbalance among parallel computing cores. Consequently, special-purpose sparse tensor accelerators have been proposed to optimize critical sparse kernels such as sparse-sparse matrix multiplications [4, 12, 14, 17, 20, 22, 24, 26, 31, 40, 41]. These accelerators typically contain an array of multiply-accumulate processing elements and a hierarchy of SRAM buffers. They use dedicated dataflow schemes that correspond to various iteration orders among tensor dimensions, such as Inner Product (IP), Outer Product (OP), and Gustavson's.

For large sparse tensors, the on-chip buffer in the accelerator may be insufficient to fit all data, and there would still be substantial random data accesses to the expensive off-chip memory. In such cases, *tiling* becomes an attractive solution, where the tensor is split into multiple smaller tiles that each fit in the buffer and are maximally reused on-chip before moving to the next tile. However, the irregular distribution of sparse data makes it difficult to identify the optimal tile shapes and sizes. A large tile with many non-zero elements may overflow the SRAM buffer and sacrifice data reuse, while a small tile with few non-zero elements would underutilize the buffer space and lead to many tiles which cause unnecessary refetches of the other operand tensors.

State-of-the-art sparse accelerators try to address this difficulty through either dynamic runtime tiling that flexibly changes the tile size [19, 25], or using static heuristics to slightly overbook the buffer space to improve utilization [38]. Unfortunately, purely dynamic tiling has to limit its tiling decisions to a small number of choices due to high implementation complexity, and purely static tiling is usually less efficient when data sparsity varies significantly. In addition, we find that these prior designs have not thoroughly explored the full design space of tiling. Many of their design parameters, including the tile shape, the inter-tile iteration order, and the relative space of SRAM buffers allocated among different operand tensors, are fixed and sub-optimal, especially when the tensors have diverse sparse patterns. Moreover, the metadata to support tiling,

e.g., the begin and end locations of the compressed non-zero data in a tile, may also become a significant overhead and require careful management by the hardware accelerator.

In this paper, we take a holistic approach to study the tiling strategies of sparse tensor accelerators and propose HYTE, a *hybrid static-dynamic* framework for *flexible and efficient* sparse tiling. HYTE supports a rich set of flexible tiling parameters, including the tile size (number of non-zero elements within a tile), the tile shape (coordinate range along each tensor dimension), the iteration order of dimensions across adjacent tiles, and the SRAM buffer allocation policies. At the static offline phase, HYTE relies on a scheduler to analyze the sparsity patterns of the operand tensors, using *effective yet lightweight sampling* approaches to estimate several key metrics. With the help of a performance model, the scheduler then generates a near-optimal tiling scheme with initial values for the above parameters. Our sampling method is more comprehensive than previous static heuristics [38], and gives more efficient tiling results with only minor offline overheads.

With the initial tiling scheme, the HYTE hardware further applies dynamic tuning, which shrinks or extends the tile size to always ensure maximum buffer utilization even with highly varying local data sparsity patterns. Because the statically scheduled scheme is near-optimal, dynamic tuning can be much simplified. Besides, HYTE efficiently manages the metadata in both the off-chip memory (for inter-tile execution) and the on-chip buffer (for intra-tile execution), and flexibly shares the buffer space between data and metadata to alleviate the metadata complexity.

We evaluate HYTE by comparing it with the state-of-the-art sparse accelerators [19, 25, 38] on a diverse range of sparse datasets. On the representative sparse-sparse matrix multiplication kernel with the Gustavson's hardware dataflow, HYTE is on average 3.3× to 6.2× faster than the baselines, and performs very close to the exhaustively searched static optimal schemes. Most of the benefits are enabled by the flexible tiling parameter choices and the effective static scheduling, while our dynamic features in hardware can also boost performance for certain pathological cases when the static scheduler fails to find a good scheme. We also show the performance gains of HYTE are consistent across various sparse computation kernels and different hardware dataflows. The offline scheduling cost is minor even though it executes on the CPU, thanks to our effective sampling method.

We make the following contributions in this paper.

- We demonstrate that existing sparse accelerators have not extensively explored the full design space of tiling, including the tile size, tile shape, inter-tile iteration order, and buffer allocation policies.
- We propose a static offline scheduler for sparse accelerators, which uses lightweight sampling to adaptively identify nearoptimal tiling schemes for various sparsity patterns.
- We design a hardware architecture for sparse accelerators, which supports dynamic tuning on the tile shape to ensure high buffer utilization, and efficiently manages the tiling metadata in both the off-chip memory and the on-chip buffer.
- We integrate the above techniques into a hybrid static-dynamic framework, which enables flexible and efficient tiling on sparse accelerators, and significantly outperforms previous approaches on diverse sparse matrices.

