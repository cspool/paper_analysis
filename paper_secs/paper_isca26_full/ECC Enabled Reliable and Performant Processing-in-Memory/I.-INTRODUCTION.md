# I. INTRODUCTION

Processing-in-Memory (PIM) architectures integrate compute units near or within memory to reduce data movement. PIM architectures have been widely explored by both academia and industry [4]–[6], [30], [33], [57], demonstrating substantial performance and energy efficiency gains. In particular, performance gains are maximized when compute units access local data utilizing high internal DRAM bandwidth. Thus, bank-PIMs [5], [7], [11], [23], [28], [33], [43], [59], which position compute units near individual memory banks, deliver higher throughput (8× or more) compared to rank-PIMs [33], [72], which place compute units within a memory rank. However, meeting datacenter reliability requirements with a bank-PIM remains an unaddressed challenge.

We present the *reliable bank-PIM* architecture that maintains the performance advantage of bank-local computation while simultaneously approaching the reliability of current datacenter rank-level ECC (Figure 1 plots quantitative results from section VI). We show that naive approaches to PIM reliability are insufficient. Bank-PIM ECC approaches that rely solely on near-bank ECC (i.e., on-die ECC alone [14], [22],

![](_page_0_Figure_9.jpeg)

Fig. 1: Modeled reliability vs. theoretical max speedup for standard rank-PIM and bank-PIM in a one channel, 4 rank system; reliable\_PIM uses DDR5 on-die ECC redundancy, and reliable\_PIM\* aligns with HBM3 on-die ECC redundancy.

[26]) can lead to unacceptably high failure rates that are more than two orders of magnitude worse than our reliable bank-PIM. This remains the case even when considering the greater on-die ECC redundancy of HBM3 [22]. Switching to a rank-PIM architecture solves the reliability challenge by harnessing strong rank-level ECC, but sacrifices up to  $8\times$  performance.

These general results hold true considering end-to-end long-sequence LLM decoding. LLM decoding is an important application that is a good match for PIM execution and has been the focus of recent PIM efforts [13], [31], [33], [60]. Our reliable bank-PIM prevents a 30-50% inference accuracy drop from memory errors expected with purely on-die ECC, while still improving end-to-end latency and throughput by  $2-4\times$  over a GPU and a rank-PIM.

The reliable bank-PIM comprises two main components and is tailored for "all-bank" style PIMs where control is retained by the host [43], [48] (see section IV). The first component is a two-tiered ECC [10], [21], [56], [81] that combines strong error detection within each bank while leveraging "chipkill"-level ECC for correction at the rank level (e.g., [15], [20], [32], [34], [56]). The second is a novel masking of scaling-induced VRT errors at the bank level to avoid triggering frequent, performance-sapping rank-level corrections.

The first ECC tier is applied near the bank and focuses on detection. This maintains the performance of bank-local processing while providing high coverage for multi-bit errors

<sup>\*</sup>Both authors contributed equally to this research.

caused by operational faults. In contrast to traditional rank-PIMs that trade bank-local computation for reliability, the reliable bank-PIM only requires accessing the second-tier rank-level ECC when correcting errors or performing writes. The read-mostly nature of our target PIM applications allows this approach to avoid frequent cross-chip re-encoding and write amplification (host writes are naturally at the rank level).

An important performance challenge is keeping the rate of rank-level corrections low. This is straightforward for operational faults, which are rare and can be detected and mapped out (e.g., page retirement, subsection IV-B). Scalinginduced VRT errors are much more problematic due to their random and variable nature. On-die ECC typically tolerates VRT errors, but the detection-focused first-tier ECC does not.

We introduce a novel *Codeword Flip* mechanism to mask VRT errors rather than correct them. Codeword Flip exploits the VRT error behavior to mask repeated VRT errors and prevents re-sensitization of previously failing cells. It stores the corrected codeword in flipped form after a rank-level correction, without any metadata to track the flip state (subsection IV-C). Our results show that Codeword Flip keeps correction overhead at under 2% even at extreme VRT error rates (subsection VI-B).

In summary, our main contributions are:

- We show that bank-PIM architectures cannot attain high reliability with purely bank-level protection.
- We introduce *Codeword Flip* to minimize rank-level VRT error correction overheads, preserving bank-PIM performance, while continuing to detect multi-bit errors.
- We propose a two-tier ECC tailored for host-controlled bank-PIMs that operates with minimal changes to existing interfaces, provides rank-level chipkill ECC, and achieves reliability approaching that of rank-level ECC.
- We evaluate the reliability and performance of our reliable bank-PIM and demonstrate 400× better SDC rate with < 2.1% performance degradation compared to a DDR5 bank-PIM baseline at equal redundancy.
- We conduct an end-to-end performance evaluation on long-sequence generative LLMs, confirming that reliable bank-PIM not only enhances reliability but also preserves the internal bandwidth advantages of bank-PIM, leading to 2 − 3× improved latency and throughput compared to rank-PIM or GPU alone.

