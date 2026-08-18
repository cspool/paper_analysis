# <span id="page-10-0"></span>D. Sensitivity Studies

We study SegFold's sensitivity along two axes: *hardware* parameters and *input characteristics*.

1) Hardware Parameter Sensitivity: To understand how hardware parameters shape SegFold's performance, we perform two sensitivity studies across three matrix sizes  $\{256, 512, 1024\}$  and two density levels  $\{0.05, 0.1\}$ : (i) varying the bandwidth of the vector multicast network that delivers B rows, and (ii) varying the active-window size.

**Vector Multicast Bandwidth.** As shown in Fig. 5(a), Seg-Fold's global network routes B rows from memory to PE rows through a vector multicast network. To observe sensitivity, we vary the network multicast width from 1 to 16 B rows

<span id="page-11-0"></span>![](_page_11_Figure_0.jpeg)

Fig. 13: Per-simulation efficiency (cycles per MAC, lower is better) on synthetic square matrices across densities, comparing Flexagon (OP and Gustavson), Spada, and SegFold.

per cycle, keeping all other parameters fixed. As shown in Fig. 12, performance improves noticeably from 1 to 4 rows per cycle across all matrix sizes and densities, but the marginal benefit beyond 4 rows per cycle quickly diminishes. At higher density (d=0.1), the sensitivity is more pronounced for larger matrices, as increased nonzero density creates more contention on the network. Based on this trend and the associated area/wiring cost, we choose a bandwidth of 4 B rows per cycle.

Window Size of Active B Rows. As discussed in  $\S$ IV-B, the active window over B rows controls how many k values and B rows can be reordered and scheduled in parallel. We sweep window sizes from 1 to 64, and report the normalized cycles in Fig. 12. Performance improves substantially as the window grows up to 32, but shows little further gain beyond 32, consistently across both density levels. At higher density (d=0.1), the benefit of increasing window size is slightly more pronounced, as denser matrices expose more reordering opportunities. We therefore adopt a window size of 32 as our default configuration, as it offers a good trade-off between benefit and metadata/storage overhead.

2) Input Sensitivity: To understand how input characteristics affect SegFold's performance, we perform two sensitivity studies with the synthetic matrices: (i) varying the density of the two input matrices from highly sparse to entirely dense; and (ii) varying the relative sparsity of the two input matrices to create asymmetric sparsity patterns.

**Density Sweep.** We test synthetic matrices of sizes 256 to 1024 with densities ranging from 0.05 to 1.0. Figure 13 reports the average cycles per MAC across the three accelerators. As density increases, SegFold's per-MAC efficiency stays roughly flat through mid densities and then drops sharply at the fully dense end, while Spada's degrades sharply once density exceeds 0.4. This transition reflects bandwidth saturation in Spada's row-sequential 16-channel memory. Flexagon's outerproduct (OP) dataflow shows the biggest improvement with increasing density, as its static dataflow can better exploit reuse opportunities when more nonzeros are present, and the fixed control overhead is amortized. This also explains why SegFold's speedup over Flexagon is relatively large on the SuiteSparse matrices, which are generally sparser than the synthetic ones. Notably, in the fully dense case, SegFold outperforms all the baselines: its 2D array natively supports

<span id="page-11-1"></span>![](_page_11_Figure_6.jpeg)

Fig. 14: Asymmetric-sparsity sensitivity at K = 1024.

dense GEMM, and the dynamic-mapping overhead is diminished when the matrix is dense, since the  $\mathcal V$  space mapping is essentially static—every position is occupied, leaving no shifts or skip decisions for the merge network to make, while baselines have the static overhead introduced by isolating the multiplication and reduction phases. For sparsity levels beyond the figure's range, SegFold's speedup over Spada increases.

Asymmetric Sparsity. SegFold treats A and B asymmetrically, raising the question of whether sparsity differences between A and B influence whether  $A \times B$  or  $A^{\top} \times B^{\top}$  is faster. To understand this, we sweep  $(d_A, d_B)$  pairs over synthetic matrices with size 1024 and report the swap ratio  $\operatorname{cyc}(d_A, d_B)/\operatorname{cyc}(d_B, d_A)$  in Fig. 14: ratios < 1 (blue) favor placing the sparser matrix as operand A, while ratios > 1 (red) favor placing the denser one there. Since the upper triangle is the reciprocal of the lower, we focus on the lower-right half  $(d_A \leq d_B)$ . Most of this region is blue: having the sparser matrix as operand A is faster because SELECTA's fine-grained scheduling on A leverages high sparsity better than the coarse-grained loader on B can.

However, in the red corner where the disparity in density between A and B becomes very large, having the denser matrix as the first operand becomes faster. When A's sparsity is extreme, a substantial fraction of A's rows contain no nonzeros, yet each row still triggers a SELECTA iteration with its associated scheduling and pipeline overhead; the finegrained selection that benefits sparse A in the common case now operates over many empty rows and produces no useful work. Swapping the operands places the denser matrix in operand A—every row is occupied, so no iteration is wasted while the very sparse matrix moves to operand B, where the coarse-grained, demand-driven loader simply skips empty rows at no cost. The crossover from blue to red occurs once the density ratio  $d_B/d_A$  grows large enough (in our experiments, around  $32\times$  to  $64\times$ ) that the per-row SELECTA overhead saved by the swap exceeds the fine-grained sparsity benefit lost on A.

