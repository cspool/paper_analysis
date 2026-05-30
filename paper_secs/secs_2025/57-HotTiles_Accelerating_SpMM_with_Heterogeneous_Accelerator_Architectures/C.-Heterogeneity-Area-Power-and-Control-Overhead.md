# *C. Heterogeneity Area, Power, and Control Overhead*

Since PIUMA includes heterogeneous PEs by design, Hot-Tiles can be supported without additional hardware overhead. This is not the case for the SPADE-Sextans architecture, where HotTiles adds some overheads associated with combining and controlling different PEs. Specifically, HotTiles needs some control logic to orchestrate the two subaccelerators. Since when the two subaccelerators are operating in parallel they are

TABLE V: Benchmark sparse matrices used.

| Benchmark                    | Short | Domain               | Rows<br>(Mill) | NNZ<br>(Mill) | Density              |
|------------------------------|-------|----------------------|----------------|---------------|----------------------|
| as-Skitter                   | ski   | Internet<br>topology | 1.7            | 22            | 8 * 10 <sup>-6</sup> |
| coPapersCiteseer             | pap   | Citation<br>network  | 0.4            | 32            | $2*10^{-4}$          |
| delaunay_n22                 | del   | Geometry<br>problem  | 4.2            | 25            | $1*10^{-6}$          |
| dgreen                       | dgr   | VLSI                 | 1.2            | 27            | $2*10^{-5}$          |
| kron g500-logn19             | kro   | Synthetic graph      | 0.5            | 44            | $2*10^{-4}$          |
| mycielskian17                | myc   | Math.                | 0.1            | 100           | $1*10^{-2}$          |
| packing-500<br>x100x100-b050 | pac   | Numerical simulation | 2.1            | 35            | 8 * 10 <sup>-6</sup> |
| Serena                       | ser   | Environ.<br>science  | 1.4            | 64            | $3*10^{-5}$          |
| soc-Pokec                    | pok   | Social<br>network    | 1.6            | 31            | $1*10^{-5}$          |
| wiki-topcats                 | wik   | Web graph            | 1.8            | 29            | $9*10^{-6}$          |

writing to private output buffers, all that is needed from the new control logic is: (1) to signal the beginning of operation of each PE, (2) to monitor the termination of operation of each PE, and (3) to signal the beginning of operation of the Merger module. This control functionality can be accomplished in software by utilizing the SPADE Control Processing Element (CPE) [24], which is a general-purpose core that can write to memory-mapped registers.

The only additional hardware module that needs to be integrated into the heterogeneous SPADE-Sextans architecture is the Merger module. It includes a SIMD ADD module and some registers. We estimate its area and power using CACTI [9] for the memory structures and the numbers from [22] for the SIMD arithmetic. Similar to the SPADE paper, we scale area and power to 10 nm using the scaling factors from [60]. We compare them to the area and power of a SPADE PE, including PE pipeline, L1 cache, and BBF. Our results suggest that the Merger module has very small overheads. It accounts for less than 20% of the area and power of a single SPADE PE.

#### VIII. EVALUATION

The evaluation is organized in three parts. Subsection VIII-A compares the performance of heterogeneous execution with *HotTiles* against other execution environments. Then, subsection VIII-B investigates the effectiveness of using *HotTiles* for architecture exploration. Finally, subsection VIII-C discusses the preprocessing cost.

