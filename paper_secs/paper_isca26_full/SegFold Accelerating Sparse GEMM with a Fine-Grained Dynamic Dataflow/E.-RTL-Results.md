# E. RTL Results

Table IV reports the post-synthesis RTL estimates of power and area for SegFold. We synthesize all modules using the

<span id="page-12-0"></span>TABLE IV: Power and area of SegFold. Left: ASAP7 7 nm synthesis at 1 GHz. Right: estimated 28 nm scaling at 1 GHz (with pipeline modifications).

|                         | ASAP7 7 nm                              |               | Est. 28 nm                                              |               |
|-------------------------|-----------------------------------------|---------------|---------------------------------------------------------|---------------|
| Component               | Area [μm²]                              | Power<br>[mW] | Area $[\mu \text{m}^2]$                                 | Power<br>[mW] |
| PE (×256)               | 26,304                                  | 59.1          | ~231,400                                                | ~342          |
| Switch ( $\times 256$ ) | 10,967                                  | 27.1          | $\sim 87,700$                                           | ~156          |
| FIFO Buffer (×256)      | 105,600                                 | 263.4         | $\sim 887,000$                                          | $\sim 1,574$  |
| Scratchpad (×16)        | 16,025                                  | 34.3          | $\sim 134,600$                                          | $\sim 208$    |
| Mem Controller (×1)     | 1,603                                   | 1.8           | ~13,470                                                 | ~11           |
| Total                   | <b>160,499</b> (0.160 mm <sup>2</sup> ) | 385.7         | $\sim$ <b>1,354,200</b> ( $\sim$ 1.35 mm <sup>2</sup> ) | ~2,290        |

ASAP7 7 nm standard-cell library [2] at 1 GHz. For the 2D mesh, we synthesize a single PE, switch, and FIFO buffer, and scale their results to a  $16 \times 16$  grid. Each PE row includes a spad for overflow C entries; these are synthesized per bank and scaled by 16 rows. The global memory controller is synthesized separately. At 7 nm, the total design occupies  $0.160 \, \text{mm}^2$  and consumes  $385.7 \, \text{mW}$ . To enable comparison with prior work synthesized at older technology nodes, we also provide estimated  $28 \, \text{nm}$  numbers by applying standard area ( $\sim 8 \times$ ) and power ( $\sim 5.5 \times$ ) scaling factors, with additional pipeline stages inserted to meet 1 GHz timing at  $28 \, \text{nm}$ .

For comparison, Flexagon [28] reports a total area (without cache) of  $1.35\,\mathrm{mm}^2$  and power consumption of  $856\,\mathrm{mW}$  at  $28\,\mathrm{nm}$ . When scaled to the same  $28\,\mathrm{nm}$  technology node, SegFold's estimated area of  $\sim 1.35\,\mathrm{mm}^2$  is comparable, while providing the additional hardware for dynamic scheduling (merge network, scratchpads) that enables the performance gains shown above. The estimated  $28\,\mathrm{nm}$  power of  $\sim 2.3\,\mathrm{W}$  assumes conservative vectorless 50% switching activity; real sparse workloads exhibit lower activity factors, which would substantially reduce dynamic power.

#### VII. ADDITIONAL RELATED WORK

**Reconfigurable Architectures.** Reconfigurable architectures [13], [35], [37] are inherently programmable and can support different dataflows. Prior architectures have explored specialization for sparsity, such as SPU [5] for sparse inner-product joins, and Capstan [18], [40] for parallel indirect memory access in Gustavson. To our knowledge, dynamic dataflow optimizations have not been explored on a reconfigurable architecture, and they would likely be costly given coarse-grain reconfigurable primitives.

**Dynamic Prediction.** Dynaflow [48] predicts the best dataflow for dataflow-flexible accelerators like those above, using ML techniques trained on dataset sparsity patterns. SparseAdapt [34] is a runtime system for a flexible reconfigurable design called Transmuter [35], which can change microarchitectural policies depending on sparse data patterns.

**Tile-Ordering Strategies.** Inter-tile ordering strategies such as Hilbert or Z-order traversals—used in CUBE [52] and Mosaic [27]—and the sliding-window tiling in Spahet [19] optimize the *order* in which tiles are visited to improve crosstile cache locality, but leave the scheduling of work *within* 

a tile static. SegFold's dynamic scheduling is orthogonal: it operates at sub-tile granularity, reordering operations inside a tile based on runtime sparsity, and could therefore compose with any of these inter-tile ordering schemes.

Flexibility and Dynamism in Related Domains. EIE [15] is a SpMV accelerator that mitigates load imbalance with large input queues. Cerebrus [20] is a SpMV accelerator that can support multiple dataflow patterns. ZeNa [22] is a sparse CNN accelerator that uses work-stealing queues to perform load balancing at a tile level. Graph processing accelerators face similar sparsity patterns and encounter the same loadbalance and reuse challenges. PolyGraph [4] uses offline preprocessing, which aims to improve both load balance and locality. HATS [29] uses bounded depth-first search ordering to improve locality. AWB-GCN [12] is a graph convolution accelerator, which dynamically load balances rows among PEs, and further splits pathologically long rows. These graph accelerators all exploit locality in settings where the output structure is known ahead of time; SegFold faces the additional challenge of discovering the output nonzero pattern on-thefly during SpGEMM execution, requiring runtime locality decisions rather than offline preprocessing.

Comparison with Preprocessing-Based Approaches. Building on the Gustavson-style preprocessing designs introduced in §II, we contrast SegFold's runtime approach with offline preprocessing in more detail. Gamma [51] and Zed [6] use offline preprocessing to reorder rows of the stationary matrix, grouping rows with similar sparsity patterns to improve streamingmatrix reuse under a Gustavson dataflow. This preprocessing must be performed once per matrix and amortized across repeated SpGEMM invocations. In contrast, SegFold achieves similar reuse benefits—grouping A columns with overlapping nonzero patterns via SELECTA—entirely at runtime, with no preprocessing step. This makes SegFold particularly advantageous for workloads where the sparse matrices change frequently (e.g., dynamic graphs or iterative solvers), as the cost of preprocessing cannot be amortized. For static matrices used repeatedly, preprocessing-based approaches may offer complementary benefits, and combining offline reordering with SegFold's runtime scheduling is a promising direction for future work.

#### VIII. CONCLUSION

This work introduces Segment, a novel dynamic dataflow for SpGEMM that simultaneously optimizes data reuse and load balance, overcoming limitations of conventional static dataflows. Through the co-designed SegFold accelerator, we demonstrate that fine-grained dynamic scheduling and remapping of work across PEs can significantly improve performance and utilization across diverse sparsity patterns and matrix sizes. Our results—a  $1.95\times$  geometric-mean speedup over the best dynamic baseline and  $5.3\times$  over the best static baseline—highlight the value of incorporating dynamism into sparse-matrix accelerators, providing a new pathway for efficient execution of irregular workloads.

