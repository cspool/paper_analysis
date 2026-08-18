# *I. Knee-of-the-Curve: CXL Bottleneck Threshold*

The previous sections demonstrate that production workloads perform well with CXL. To identify the *threshold* at which DDR4-enabled CXL becomes a performance bottleneck, we run stress-test experiments that synthetically vary the fraction of the memory footprint that is actively (hotly) accessed. We use a memory-intensive benchmark on the Mem-Server platform (768 GB local DDR5 + 256 GB CXL DDR4) with a configurable hot-page fraction: a tunable percentage of the total memory footprint is accessed frequently, while the remainder is accessed infrequently. The kernel's TPP mechanism transparently manages page placement throughout, promoting hot pages to local DDR5 and demoting cold pages to CXL-attached DDR4. Table X summarizes the results.

TABLE X KNEE-OF-THE-CURVE STRESS TEST. LATENCY OVERHEAD IS RELATIVE TO THE 10% BASELINE (273 NS).

| Hot Footprint<br>(%) | Latency<br>(ns) | Local BW<br>(GBps) | CXL BW<br>(GBps) | Latency<br>Overhead |
|----------------------|-----------------|--------------------|------------------|---------------------|
| 10                   | 258             | 134                | 4.9              | - %                 |
| 20                   | 261             | 134                | 5.1              | +1%                 |
| 40                   | 261             | 134                | 4.9              | +1%                 |
| 60                   | 261             | 134                | 4.9              | +1%                 |
| 70                   | 264             | 132                | 6.9              | +2%                 |
| 75                   | 269             | 124                | 12.4             | +4%                 |
| 80                   | 279             | 108                | 21               | +8%                 |
| 100                  | 315             | 72                 | 34               | +22%                |

Stable regime (under 75% hot footprint). When the hot fraction is under 75%, the system operates in a flat, stable regime: average memory access latency remains under 264 ns, local DRAM bandwidth is high around 130 GBps, and CXL bandwidth stays at under 7 GBps. In this regime, TPP successfully accommodates the entire hot working set in local DRAM, and CXL traffic is limited to infrequent cold-page accesses.

The knee (75% hot footprint). The inflection point appears around 75%. Beyond 75% hot footprint, latency rises to 269 ns (+4%), local bandwidth drops to 124 GBps, and CXL bandwidth jumps to 12 GBps—upto 2.4× increase over the stable regime—as the hot working set begins to exceed what TPP can fully accommodate in local DRAM and promotion/demotion traffic intensifies.

Degradation regime (>75% hot footprint). Beyond 75%, performance degrades rapidly. At 80%, latency rises to 279 ns (+8%), local bandwidth falls to 108 GBps, and CXL bandwidth reaches 21 GBps. At 100%, where the entire footprint is hot, latency reaches 315 ns (+22%) while drawing CXL bandwidth as high as 34 GBps.

Datacenter workloads operate below the knee. All services evaluated in this paper have large cold memory fractions: Table I shows that at least 75% of pages remain idle for over 4 seconds across all workloads, and Table VIII confirms that CXL bandwidth in the fleet is 10–100× lower than local bandwidth. These workloads sit firmly in the flat regime, well below the ≈60% knee, confirming that DDR4-enabled CXL memory expansion is robust for the capacity-bound workloads typical of hyperscale deployments. For workloads that approach or exceed this threshold, higher-bandwidth CXL links (e.g., PCIe x16) or next-generation DDR5-based CXL devices would be needed.

