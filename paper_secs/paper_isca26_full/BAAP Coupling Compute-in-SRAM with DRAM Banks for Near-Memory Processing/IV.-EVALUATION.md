# IV. EVALUATION

<span id="page-7-0"></span>In this section, we evaluate BAAP experimentally, comparing it against several reference architectures running a variety of workloads.

## *A. Methodology*

Near-memory baseline. As discussed in Section [III,](#page-2-4) we demonstrate the potential of BAAP by extending the UPMEM PIM architecture [\[11\]](#page-13-4), which we use as our near-memory areaequivalent baseline. We simulate it with uPIMulator [\[29\]](#page-13-15), a cycle-approximate simulator that was recently released and validated against real UPMEM hardware running the PrIM benchmarks [\[21\]](#page-13-5), which we also employ for our design space exploration and the first part of our evaluation. As of this writing, the simulator supports all documented APIs and 62% of the ISA (it is purely based on publicly available information). In our experience, this suffices to run 12 out of the 16 PrIM benchmarks reliably. For PrIM benchmarks that offer multiple variants (e.g., SCAN-SSA, SCAN-RSS), we report, for each benchmark, the UPMEM baseline corresponding to the best-performing variant on UPMEM and use that same variant when comparing against BAAP.

Host-side baseline. For reference, we have provided a hostside baseline in multiple figures. Specifically, a server-grade Intel Xeon W-1390P CPU core (Rocket Lake architecture) clocked at 3.5 GHz, described in more detail in Table [V.](#page-8-0) For our final end-to-end evaluation, we employ all 16 cores.

System modeling and performance evaluation. We model BAAP itself using the gem5 simulator [\[39\]](#page-13-29). The memory backend parameters follow the latency and bandwidth figures measured experimentally by previous work on a real UPMEM product [\[21\]](#page-13-5), and are the same ones used to evaluate the UPMEM baseline with uPIMulator [\[29\]](#page-13-15). The frontend models an SRAM-based AP driven by a tiny in-order control CPU, closely following CAPE [\[8\]](#page-12-7) with additional optimizations [\[52\]](#page-14-9), [\[57\]](#page-14-12). The latency of each vector instruction supported by BAAP depends on the CAM searches and updates required to implement it, as summarized by Table [IV.](#page-4-0)

Low-level modeling. Since BAAP extends the scratchpad with additional functionality, we reduce its capacity for a fair comparison against a plain-scratchpad configuration. We obtain this figure as follows: First, we derive the area and energy overhead of BAAP's AP-endowed storage relative to a plain scratchpad of equal capacity in conventional CMOS logic, by obtaining a baseline for the SRAM scratchpad with FN-CACTI [\[47\]](#page-14-13) (7 nm) and comparing it to the RTL synthesis and layout results of an AP chain (Figure [4\)](#page-5-0) generated with Synopsys Design Compiler on the ASAP 7 nm PDK [\[10\]](#page-13-30). The latter were informed by SPICE simulations of 32x36 subarrays of the 6T push-rule SRAM bitcells, including peripheral circuitry [\[30\]](#page-13-23), and the tagging, accumulation, reduction, and intermediate-result propagation logic that enables AP functionality [\[8\]](#page-12-7), shown in Figure [4.](#page-5-0) The overhead works out to be 1.2281×. Although these area values are for conventional CMOS logic, rather than in-DRAM logic, it is the relative overhead we are interested in, as we reasonably assume that it would be a similar ratio for the in-DRAM implementations, especially since both designs are fundamentally SRAM arrays (BAAP, like CAPE [\[8\]](#page-12-7), uses Jeloka's 6T push-rule cell [\[30\]](#page-13-23)). Thus, when in an experiment we repurpose p% of the scratchpad to be AP-enabled, we assume AP storage of c · p/122.81, where c is the total scratchpad capacity of the near-memory baseline (64 KB for UPMEM). This primarily affects BAAP's vector lengths. As for the clock speed, we

TABLE V: O3CPU baseline specifications

<span id="page-8-0"></span>

| Core configuration | Intel Xeon W-1390P (Rocket Lake architecture) @ 3.5 GHz |  |  |  |
|--------------------|---------------------------------------------------------|--|--|--|
|                    | AVX-512 (SIMD), 5-issue, 352 ROB, 12 backend ports      |  |  |  |
| L1I cache          | 48 KiB/core, 12-way set associative                     |  |  |  |
| L1D cache          | 32 KiB/core, 8-way set associative                      |  |  |  |
| L2 cache           | 512 KiB/core, 8-way set associative                     |  |  |  |
| L3 cache           | 2 MiB/core 16-way set associative                       |  |  |  |
| Memory             | 4x 32GiB DIMM DDR4 @ 3200 MHz                           |  |  |  |

TABLE VI: AP configurations evaluated across the memory hierarchy

<span id="page-8-2"></span>

|                                  | Vector Length<br>VL        | Vector registers<br>(Bitwidth) | Scratchpad Capacity (excl. BAAP) | Memory                                                                     | Clock Freq.<br>(Logic) |
|----------------------------------|----------------------------|--------------------------------|----------------------------------|----------------------------------------------------------------------------|------------------------|
| Bank-level BAAP<br>(25% of WRAM) | 96 elements<br>x128 banks  | 32<br>(32 bits)                | ≈ 48 KB                          | DDR4-2400 [11]<br>BW: 1 GBps / bank<br>Aggregated: 128 GBps                | 350 MHz                |
| Bank-level BAAP<br>(50% of WRAM) | 224 elements<br>x128 banks | 32<br>(32 bits)                | ≈ 32 KB                          | DDR4-2400 [11]<br>BW: 1 GBps / bank<br>Aggregated: 128 GBps                | 350 MHz                |
| Bank-level BAAP<br>(75% of WRAM) | 320 elements<br>x128 banks | 32<br>(32 bits)                | ≈ 16 KB                          | DDR4-2400 [11]<br>BW: 1 GBps / bank<br>Aggregated: 128 GBps                | 350 MHz                |
| Bank-level BAAP<br>(88% of WRAM) | 384 elements<br>x128 banks | 32<br>(32 bits)                | ≈ 8 KB                           | DDR4-2400 [11]<br>BW: 1 GBps / bank<br>Aggregated: 128 GBps                | 350 MHz                |
| Host-side AP<br>(Based on [8])   | 49,152 elements            | 32<br>(32 bits)                | N/A                              | 4x 4H HBM<br>w/ 16 channels<br>(128 pseudo-channels)<br>Aggr. BW: 256 GBps | 2.72 GHz               |

<span id="page-8-1"></span>![](_page_8_Figure_4.jpeg)

Fig. 5: UPMEM scratchpad sensitivity study.

assume a reduced BAAP frequency of 350 MHz with respect to UPMEM's 500 MHz to conservatively account for any additional effects of DRAM process technology limitations. Moreover, in Section IV-D, we sweep a wider range of frequencies and vector lengths.

# IV. EVALUATION

<span id="page-7-0"></span>In this section, we evaluate BAAP experimentally, comparing it against several reference architectures running a variety of workloads.

## *A. Methodology*

Near-memory baseline. As discussed in Section [III,](#page-2-4) we demonstrate the potential of BAAP by extending the UPMEM PIM architecture [\[11\]](#page-13-4), which we use as our near-memory areaequivalent baseline. We simulate it with uPIMulator [\[29\]](#page-13-15), a cycle-approximate simulator that was recently released and validated against real UPMEM hardware running the PrIM benchmarks [\[21\]](#page-13-5), which we also employ for our design space exploration and the first part of our evaluation. As of this writing, the simulator supports all documented APIs and 62% of the ISA (it is purely based on publicly available information). In our experience, this suffices to run 12 out of the 16 PrIM benchmarks reliably. For PrIM benchmarks that offer multiple variants (e.g., SCAN-SSA, SCAN-RSS), we report, for each benchmark, the UPMEM baseline corresponding to the best-performing variant on UPMEM and use that same variant when comparing against BAAP.

Host-side baseline. For reference, we have provided a hostside baseline in multiple figures. Specifically, a server-grade Intel Xeon W-1390P CPU core (Rocket Lake architecture) clocked at 3.5 GHz, described in more detail in Table [V.](#page-8-0) For our final end-to-end evaluation, we employ all 16 cores.

System modeling and performance evaluation. We model BAAP itself using the gem5 simulator [\[39\]](#page-13-29). The memory backend parameters follow the latency and bandwidth figures measured experimentally by previous work on a real UPMEM product [\[21\]](#page-13-5), and are the same ones used to evaluate the UPMEM baseline with uPIMulator [\[29\]](#page-13-15). The frontend models an SRAM-based AP driven by a tiny in-order control CPU, closely following CAPE [\[8\]](#page-12-7) with additional optimizations [\[52\]](#page-14-9), [\[57\]](#page-14-12). The latency of each vector instruction supported by BAAP depends on the CAM searches and updates required to implement it, as summarized by Table [IV.](#page-4-0)

Low-level modeling. Since BAAP extends the scratchpad with additional functionality, we reduce its capacity for a fair comparison against a plain-scratchpad configuration. We obtain this figure as follows: First, we derive the area and energy overhead of BAAP's AP-endowed storage relative to a plain scratchpad of equal capacity in conventional CMOS logic, by obtaining a baseline for the SRAM scratchpad with FN-CACTI [\[47\]](#page-14-13) (7 nm) and comparing it to the RTL synthesis and layout results of an AP chain (Figure [4\)](#page-5-0) generated with Synopsys Design Compiler on the ASAP 7 nm PDK [\[10\]](#page-13-30). The latter were informed by SPICE simulations of 32x36 subarrays of the 6T push-rule SRAM bitcells, including peripheral circuitry [\[30\]](#page-13-23), and the tagging, accumulation, reduction, and intermediate-result propagation logic that enables AP functionality [\[8\]](#page-12-7), shown in Figure [4.](#page-5-0) The overhead works out to be 1.2281×. Although these area values are for conventional CMOS logic, rather than in-DRAM logic, it is the relative overhead we are interested in, as we reasonably assume that it would be a similar ratio for the in-DRAM implementations, especially since both designs are fundamentally SRAM arrays (BAAP, like CAPE [\[8\]](#page-12-7), uses Jeloka's 6T push-rule cell [\[30\]](#page-13-23)). Thus, when in an experiment we repurpose p% of the scratchpad to be AP-enabled, we assume AP storage of c · p/122.81, where c is the total scratchpad capacity of the near-memory baseline (64 KB for UPMEM). This primarily affects BAAP's vector lengths. As for the clock speed, we

TABLE V: O3CPU baseline specifications

<span id="page-8-0"></span>

| Core configuration | Intel Xeon W-1390P (Rocket Lake architecture) @ 3.5 GHz |  |  |  |
|--------------------|---------------------------------------------------------|--|--|--|
|                    | AVX-512 (SIMD), 5-issue, 352 ROB, 12 backend ports      |  |  |  |
| L1I cache          | 48 KiB/core, 12-way set associative                     |  |  |  |
| L1D cache          | 32 KiB/core, 8-way set associative                      |  |  |  |
| L2 cache           | 512 KiB/core, 8-way set associative                     |  |  |  |
| L3 cache           | 2 MiB/core 16-way set associative                       |  |  |  |
| Memory             | 4x 32GiB DIMM DDR4 @ 3200 MHz                           |  |  |  |

TABLE VI: AP configurations evaluated across the memory hierarchy

<span id="page-8-2"></span>

|                                  | Vector Length<br>VL        | Vector registers<br>(Bitwidth) | Scratchpad Capacity (excl. BAAP) | Memory                                                                     | Clock Freq.<br>(Logic) |
|----------------------------------|----------------------------|--------------------------------|----------------------------------|----------------------------------------------------------------------------|------------------------|
| Bank-level BAAP<br>(25% of WRAM) | 96 elements<br>x128 banks  | 32<br>(32 bits)                | ≈ 48 KB                          | DDR4-2400 [11]<br>BW: 1 GBps / bank<br>Aggregated: 128 GBps                | 350 MHz                |
| Bank-level BAAP<br>(50% of WRAM) | 224 elements<br>x128 banks | 32<br>(32 bits)                | ≈ 32 KB                          | DDR4-2400 [11]<br>BW: 1 GBps / bank<br>Aggregated: 128 GBps                | 350 MHz                |
| Bank-level BAAP<br>(75% of WRAM) | 320 elements<br>x128 banks | 32<br>(32 bits)                | ≈ 16 KB                          | DDR4-2400 [11]<br>BW: 1 GBps / bank<br>Aggregated: 128 GBps                | 350 MHz                |
| Bank-level BAAP<br>(88% of WRAM) | 384 elements<br>x128 banks | 32<br>(32 bits)                | ≈ 8 KB                           | DDR4-2400 [11]<br>BW: 1 GBps / bank<br>Aggregated: 128 GBps                | 350 MHz                |
| Host-side AP<br>(Based on [8])   | 49,152 elements            | 32<br>(32 bits)                | N/A                              | 4x 4H HBM<br>w/ 16 channels<br>(128 pseudo-channels)<br>Aggr. BW: 256 GBps | 2.72 GHz               |

<span id="page-8-1"></span>![](_page_8_Figure_4.jpeg)

Fig. 5: UPMEM scratchpad sensitivity study.

assume a reduced BAAP frequency of 350 MHz with respect to UPMEM's 500 MHz to conservatively account for any additional effects of DRAM process technology limitations. Moreover, in Section IV-D, we sweep a wider range of frequencies and vector lengths.

