# C. System Infidelity Comparison

Fig. 12 compares our decoder with three baseline designs in terms of system infidelity, a more appropriate metric introduced in Sec. V-B. For small code distances d=3,

our decoder achieves essentially the same system infidelity as Micro-Blossom, and significantly outperforms both UF-based decoders. Although MWPM yields intrinsically lower LER than UF variants, Micro-Blossom's decoding latency exceeds the threshold at  $d\!\geq\! 5$  (Fig. 10), so  $\hat{C}(R)\!>\! 0$  penalizes its system infidelity; it eventually becomes worse than that of Helios. In contrast, our decoder maintains low infidelity thanks to its better scalability, achieving higher accuracy than the UF-based decoders while maintaining low latency. At d=11, our decoder reduces the system infidelity by up to 74.3% compared to Micro-Blossom and by 51.7% compared to Helios.

#### VII. HARDWARE PERFORMANCE ANALYSES

#### A. Matched/Normalized Hardware Resource Comparison

Table I summarizes the hardware cost of our design and three representative baseline decoders: Micro-Blossom [8], Helios [9], and QUEKUF [23]. To make a fair comparison, we further normalize prior results to a common code distance using the reported resources together with the scaling complexity described in the original papers. Specifically, we estimate the resource cost at the target distance by fitting/interpolating from the reported results under the stated scaling trend. For latency comparisons, the clock frequency of each baseline is taken as published from its single design point, and is applied uniformly across all evaluated code distances; this matches our own setup, in which a single RTL design is used across all distances. In terms of logic utilization, our architecture requires only 108k LUTs, which is about 8.0× fewer than Micro-Blossom, and roughly  $4.3\times$  fewer than QUEKUF. A similar trend holds for flip-flops (FFs): our design uses 43k FFs, i.e.,  $2.9 \times$  fewer than Helios and  $14.7 \times$  fewer than QUEKUF. Regarding on-chip memory, our design uses about half the BRAM of QUEKUF while supporting nearly twice the maximum code distance, which demonstrates a better efficiency. In terms of achievable frequency, our decoder runs at 163 MHz, which is 3.7× higher than Micro-Blossom and  $2.1 \times$  higher than Helios.

We also quantify the dynamic-power overhead of ensemble parallelism using the Vivado power report. Each EFE branch contributes about 50 mW of dynamic power, so the  $K{=}24$  parallel branches together account for approximately 1.2 W of dynamic power. Increasing K therefore scales only the branch

![](_page_10_Figure_0.jpeg)

<span id="page-10-1"></span>Fig. 12. System infidelity comparison with SOTA decoders.

![](_page_10_Figure_2.jpeg)

<span id="page-10-2"></span>Fig. 13. FPGA resource usage vs. code distance d. Filled markers denote full Vivado synthesis (d=3, 9, 15); open markers are estimates. The shaded region indicates extrapolation beyond measured data.

term linearly while leaving the shared clustering engine and voting part unchanged.

# C. System Infidelity Comparison

Fig. 12 compares our decoder with three baseline designs in terms of system infidelity, a more appropriate metric introduced in Sec. V-B. For small code distances d=3,

our decoder achieves essentially the same system infidelity as Micro-Blossom, and significantly outperforms both UF-based decoders. Although MWPM yields intrinsically lower LER than UF variants, Micro-Blossom's decoding latency exceeds the threshold at  $d\!\geq\! 5$  (Fig. 10), so  $\hat{C}(R)\!>\! 0$  penalizes its system infidelity; it eventually becomes worse than that of Helios. In contrast, our decoder maintains low infidelity thanks to its better scalability, achieving higher accuracy than the UF-based decoders while maintaining low latency. At d=11, our decoder reduces the system infidelity by up to 74.3% compared to Micro-Blossom and by 51.7% compared to Helios.

#### VII. HARDWARE PERFORMANCE ANALYSES

#### A. Matched/Normalized Hardware Resource Comparison

Table I summarizes the hardware cost of our design and three representative baseline decoders: Micro-Blossom [8], Helios [9], and QUEKUF [23]. To make a fair comparison, we further normalize prior results to a common code distance using the reported resources together with the scaling complexity described in the original papers. Specifically, we estimate the resource cost at the target distance by fitting/interpolating from the reported results under the stated scaling trend. For latency comparisons, the clock frequency of each baseline is taken as published from its single design point, and is applied uniformly across all evaluated code distances; this matches our own setup, in which a single RTL design is used across all distances. In terms of logic utilization, our architecture requires only 108k LUTs, which is about 8.0× fewer than Micro-Blossom, and roughly  $4.3\times$  fewer than QUEKUF. A similar trend holds for flip-flops (FFs): our design uses 43k FFs, i.e.,  $2.9 \times$  fewer than Helios and  $14.7 \times$  fewer than QUEKUF. Regarding on-chip memory, our design uses about half the BRAM of QUEKUF while supporting nearly twice the maximum code distance, which demonstrates a better efficiency. In terms of achievable frequency, our decoder runs at 163 MHz, which is 3.7× higher than Micro-Blossom and  $2.1 \times$  higher than Helios.

We also quantify the dynamic-power overhead of ensemble parallelism using the Vivado power report. Each EFE branch contributes about 50 mW of dynamic power, so the  $K{=}24$  parallel branches together account for approximately 1.2 W of dynamic power. Increasing K therefore scales only the branch

![](_page_10_Figure_0.jpeg)

<span id="page-10-1"></span>Fig. 12. System infidelity comparison with SOTA decoders.

![](_page_10_Figure_2.jpeg)

<span id="page-10-2"></span>Fig. 13. FPGA resource usage vs. code distance d. Filled markers denote full Vivado synthesis (d=3, 9, 15); open markers are estimates. The shaded region indicates extrapolation beyond measured data.

term linearly while leaving the shared clustering engine and voting part unchanged.

