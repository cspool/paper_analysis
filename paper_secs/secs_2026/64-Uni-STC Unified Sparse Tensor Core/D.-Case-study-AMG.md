# D. Case study: AMG

This experiment adapts an existing AMG solver [14], [53], a key tool in scientific computing, by substituting its original FP64 dense Tensor Core calculations with our STC designs. We then quantitatively evaluate the speedup achieved on the SpMV and SpGEMM kernels, using the DS-STC as the baseline.

As illustrated in Fig. 21, Uni-STC demonstrates superior performance. In contrast, other STCs—while effective on random matrices—are hampered by the irregularity of real-world sparse patterns, such as elements being concentrated on the diagonal or within specific rows and columns. For SpMV, architectural limitations in MAC arrays constrain DS-STC, SIGMA, GAMMA, and RM-STC, impeding effective acceleration. For SpGEMM, the absence of fine-grained task partitioning restricts gains for DS-STC, GAMMA, and RM-STC. Similarly, SIGMA achieves only marginal SpGEMM improvements; despite its focus on data reuse, it suffers from suboptimal MAC utilisation. Finally, although Trapezoid achieves a  $4.15 \times$  SpMV speedup via dot-product acceleration,

TABLE IX: Area breakdown of the core modules in Uni-STC. The percentage represents the total area for a projected deployment of 432 Uni-STCs (4 per SM  $\times$  108 SMs) on an NVIDIA A100 GPU, relative to its 826  $mm^2$  die area.

| Module Name             | Area (mm <sup>2</sup> ) | Percentage (%) |
|-------------------------|-------------------------|----------------|
| Benes & MUX networks    | 0.002                   | 0.1            |
| TMS & DPG               | 0.012                   | 0.6            |
| Extra adders in SDPU    | 0.018                   | 0.94           |
| Meta data buffer (144B) | 0.0005                  | 0.03           |
| Accumulate buffer (1KB) | 0.003                   | 0.15           |
| Matrix A buffer (2KB)   | 0.007                   | 0.3            |
| Total Overhead          | 0.0425                  | 2.12           |

real-world irregularity exacerbates load imbalances across its PE rows, limiting it to a modest  $1.06\times$  speedup for SpGEMM. Conversely, Uni-STC effectively mitigates these irregularities, delivering notable speedups of  $4.84\times$  for SpMV and  $2.46\times$  for SpGEMM.

## E. Energy Efficiency Density

We introduce the Energy Efficiency Density (EED) metric to holistically evaluate Uni-STC and guide the determination of the optimal number of DPGs. This metric quantifies the trade-offs among performance, energy consumption, and area, and is defined as the normalized energy efficiency per area:  $EED = \frac{\text{Speedup} \times \text{Energy Reduction}}{\text{Area Overhead}}. \text{ A higher EED value signifies greater energy efficiency achieved per unit of area.}$ 

Fig. 22 presents a detailed comparative analysis of the EED for the three STCs, revealing that Uni-STC consistently outperforms both DS-STC and RM-STC across the evaluated workloads. The analysis shows contrasting trends as the number of DPGs increases from 4 to 16: the EED for SpMV and SpMSpV gradually decreases, while for SpMM and SpGEMM, it conversely exhibits an upward trend. This tradeoff analysis identifies DPG=8 as an balanced configuration. At this setting, the EED for SpMM and SpGEMM nearly matches that of the DPG=16 configuration, representing a significant 1.37× improvement over DPG=4. Concurrently, the EED reduction for SpMV and SpMSpV is minimal—only 1.1× lower than at DPG=4. Based on this evidence, we establish DPG=8 as the default configuration for Uni-STC.

## F. Area Analysis and Time Budget

We synthesize the Uni-STC@FP64 (configured with 8 DPGs) using Yosys [79] and the FreePDK45 library [62]. Synthesis results indicate that the critical path lies within the "Execution & Write C" stage, which satisfies the 1.5 GHz timing constraint. Regarding area estimation, the buffers in Uni-STC are modeled using CACTI 7 [3] at 45 nm and scaled to 7 nm technology. For logic area, we aggregate the TMS and DPG due to their structural similarities. Furthermore, since the SDPU is derived from the original Tensor Core with additional adders, we only account for its incremental overhead. Table IX details the area breakdown of these specific modules. Ideally, the total area overhead for 432 Uni-STC units is approximately 2.12% of the 826 mm<sup>2</sup> die area of an NVIDIA A100 GPU [60].

#### VII. RELATED WORK

#### *A. Sparse Kernel Acceleration*

Prior works have adopted diverse strategies to accelerate SpMV. On heterogeneous multi-core CPU and GPU platforms, Speculative Segmented Sum [49] achieves higher throughput using speculative computation, CSR5 [47] fosters greater parallelism with tiling, HASpMV [42] leverages heterogeneity-aware formats to improve memory access, TileSpMV [57] promotes data locality through tiled processing, TileSpMSpV [34] uses adaptive kernels on GPUs, and DASP [52] utilises regularized tensor core for acceleration. For HBM-equipped FPGAs, Serpens [72] and Cuper [89] mitigates memory access conflicts through customized dataflows and reordering. In distributed environments, Dist-SpMV Balanced [55] reconciles computation and communication by means of graph partitioning.

To accelerate SpMM, researchers have pursued several optimization directions. The first is data-centric, VEGETA [33] and TB-STC [44] support hybrid sparsity formats, ASADI [41] applies diagonal compression, and Avalanche [6] improves access patterns via data reordering. The second direction optimizes the computation flow, SPADE [20] and HotTiles [19] reduce data transfer and adapt to sparsity, while GROW [31] balances data locality with parallelism. The third direction leverages specific hardware features, Eureka [22] utilises tensor cores, and Leda [88] optimizes dataflows on FPGAs.

For SpGEMM, various approaches have been proposed to improve performance. In hardware architecture and dataflows, OuterSPACE [63] pioneered the outer-product approach. Trapezoid [87] designs specialized dataflows, NeuraChip [69] uses hash-based decoupling, and TaskFusion [16] enhances data sharing. Other targeted improvements include SGCN [90] enhancing format support, HIRAC [67] improving locality, S2TA [50] supporting dual-sided sparsity, and GoSPA [11] applying intersection computation. Pattern-based and tilelevel optimisations represent another significant direction, explored in works such as SPAGHETTI [29], SpArch [94], DRT [61], GAMMA [93], HARP [39], Tailors [84], DS-STC [78], and RM-STC [30]. For sparse ML workloads, many works have investigated adaptive dataflow strategies, including FEATHER [73], Sparseloop [80], Flexagon [59], ACES [51], FEASTA [95], SPADA [43], CANDLES [24], Sparse Tensor Core [96], SparTen [21], and ExTensor [28]. Acceleration on CPUs and GPUs has also been extensively studied. Liu et al. [46], [48] proposed a foundational four-stage framework. HASpGEMM [8] improves load balancing on heterogeneous cores, while GPU-specific works exploit hardware registers [45]. In particular, TileSpGEMM [58] adopts tiled execution to enhance locality and alleviate load imbalance. Furthermore, approaches like IA-SpGEMM [82], [83] focus on input-aware method selection to adapt to matrix sparsity.

Several studies propose a unified design to accelerate multiple kernels. Early efforts combine pairs of sparse kernels, where VIA [65] improves index matching for SpMV and SpMM, and PruneGNN [25] includes units for both SpMM and SpGEMM. Griffin [68] later expands this scope by optimizing resource reuse across dense and sparse matrices. Building on this trend, KAMI [76] unifies dense GEMM with sparse SpMM and SpGEMM, and Siracusa et al. [70] propose a versatile multi-lane architecture.

Bitmap-based compression reduces indexing and bandwidth overhead. This technique is used by SMASH [37] to compress metadata and by Buluc¸ et al. [5] to cut bandwidth. More recent works adapt it for modern hardware, SpInfer [15] and BerryBees [56] design Tensor Core aware encodings, while AmgT [53] uses a bitmap driven format to accelerate both SpMV and SpGEMM.

