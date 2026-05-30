# *B. Comparison of Sister Platforms*

ASADI is a hardware-software co-designed sparse attention accelerator, where the hardware design and software design complement each other and are indispensable. This section conducts experiments to compare ASADI with two sister systems, i.e., DIA-PIM and CSR-ASADI to evaluate the contributions of hardware and software optimizations in isolation. The average performance of each model on each dataset is shown in Figure 20, with all performance metrics normalized to the speedups of the PIM baseline.

ASADI vs. DIA-PIM. Figure 20 demonstrates that DIA-PIM offers an average 1.3× speedup when compared to the PIM-baseline. This speedup can be attributed to the DIA format's superior data locality when performing *S*×*V* operation. However, it is worth noting that the DIA format involves both compression and decompression phases. While DIA-PIM benefits from the compression phase, it does not gain any advantage from the decompression phase, as it requires the same number of cross-bank transfers as the CSR format. The crossbank transfer of DIA-PIM increases with sequence length, similar to the PIM-baseline, resulting in similar performance when processing various datasets.

ASADI vs. CSR-ASADI. Figure 20 presents the speedups of CSR-ASADI compared to the PIM baseline. However, CSR-ASADI underperforms compared to the PIM baseline when processing GLUE and SQuAD datasets due to limited in-situ parallelism with short sequence length and poor row locality of sparse attention. As shown in Figure 9 (b) and Figure 11 (b), the CSR-based SDDMM and SpMM computation paradigms involve many bubbles when performing insitu computing and severely impact the overall parallelism. In contrast, ASADI utilizes the DIA computation paradigm utilizing the inherent diagonal locality to reduce the number

![](_page_10_Figure_0.jpeg)

Fig. 21. Speedups of ASADI and modern sparse attention accelerators (Normalized to PIM baseline)

of bubbles significantly, resulting in better speedups than CSR-ASADI. Moreover, as the in-situ computing parallelism increases with sequence length, CSR-ASADI shows improved speedups when processing longer sequence datasets.

We can draw four conclusions from the above comparisons. First, in-situ computing is more effective in processing long sequences than PIM architecture. Secondly, the current PIM architecture cannot fully utilize the diagonal locality of the DIA format without fine-grained software-hardware co-design. Thirdly, compared to CSR-based computation paradigm, our proposed DIA-based computation paradigm effectively reduces the bubbles and enhances the parallelism of in-situ computing. Finally, our DIA-based in-situ architecture can expose the superiority of the proposed DIA computation paradigm for sparse attention.

