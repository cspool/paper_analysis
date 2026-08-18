# <span id="page-7-1"></span>*D. Folding: Mapping Segments to PEs in Space and Time*

The final V space representation of C will exhibit highly irregular row lengths, and each virtual position may require a different number of MAC operations. To maximize utilization and bandwidth, we do not constrain the V space row length to be less than the physical length of a PE row. Instead, we introduce two complementary techniques—*spatial folding* and *temporal folding*—that effectively map an irregular V space onto a regular 2D PE array.

Spatial folding addresses the mismatch between the irregular, variable-length rows in V space and the fixed physical width of the PE array. A long virtual row of C may extend beyond the capacity of a single PE row, while a short row may leave many PEs idle. To avoid underutilization, SegFold allows each logical row of outputs to *fold* across multiple PE rows.

Consider a PE array with R rows and P columns, indexed by coordinates (r, p) for 0 ≤ r < R, 0 ≤ p < P. Each router maintains a direction configuration, which specifies which of the four output ports will be used to place the next logical column of the V space row. When an on-the-fly intersection generates a new virtual column j in an occupied physical position (r, p), the router at (r, p) examines its four neighboring PEs in the following priority: {right, up, down, and left}. Right is always the first priority to first realize the default merge network configurations.

Formally, it selects the first coordinate

$$(r', p') \in \{(r, p+1), (r-1, p), (r+1, p), (r, p-1)\}$$
 (6)

whose occupancy bit satisfies O<sup>r</sup> ,p′ = 0. If such a neighbor exists, the router sets O<sup>r</sup> ,p′ ← 1, forwards the virtual column j to that PE, and updates its direction configuration to point to (r ′ , p′ ) for the next placement. In the example of Fig. [4\(](#page-5-0)c), the fourth element with index 3 in the bottom row is folded to the upper PE with an updated router configuration. Because each

TABLE II: SegFold Hardware Configuration

<span id="page-8-0"></span>

| Parameter                            | Value                        |  |  |
|--------------------------------------|------------------------------|--|--|
| PE array                             | 16 × 16                      |  |  |
| Per-row hardware (×16 rows)          | shifter, spad bank, LUT bank |  |  |
| Active B window size                 | 32                           |  |  |
| Cache size, associativity, line size | 1.5 MiB, 16-way, 128 B       |  |  |
| DRAM                                 | HBM2-8Gb, 2 Gbps             |  |  |

router exposes only its highest-priority free neighbor as the active direction, the spatial footprint of the V space row grows smoothly: horizontally when possible, and folding vertically when the row exceeds the local PE-row capacity. Through this mechanism, spatial folding enables irregular, long V space rows to occupy additional PE rows when necessary, while allowing shorter rows to leave unused PEs available for other work, thereby improving overall array utilization.

Temporal folding handles overflow and imbalance in the C reductions. While spatial folding effectively manages irregular row lengths and maps them onto the regular PE mesh, it does not guarantee that the number of nonzeros in a virtual row is always bounded by the total PE-array capacity. To support rows that exceed this capacity, each PE row is equipped with a spad that stores overflow virtual rows.

When a PE needs to accommodate a new partial sum, it can overwrite its local c<sup>i</sup> register in place and spill the previous partial sum into the spad. This mechanism not only supports rows that are longer than the physical array, but also helps mitigate reduction imbalance: partial sums that are likely to complete early can be offloaded to the spad, allowing PEs to be reused for other work while preserving correctness.

