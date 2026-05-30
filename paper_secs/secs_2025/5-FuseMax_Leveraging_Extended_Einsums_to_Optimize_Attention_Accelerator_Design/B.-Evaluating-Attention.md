# B. Evaluating Attention

We now evaluate FuseMax to demonstrate the benefits it provides on the attention kernel by comparing it to the two baselines.

Utilization. Figure 6a shows the utilization of the 1D PE array when performing attention. FLAT's utilization drops for sequence lengths  $\geq 256 {\rm K}$ —it becomes memory bandwidth limited because it must spill the QK and A tensors to memory. By using a 1-pass cascade (+Cascade), FuseMax's utilization becomes independent of sequence length. We also note that without the FuseMax binding (+Architecture), the 1D array is forced to stall and utilization drops. Adding in this binding (+Binding) enables FuseMax to fully utilize the 1D array again.

Similarly, Figure 6b shows the utilization of the 2D array. Because of the large amount of compute required for the softmax, most configurations achieve poor utilization of this array. In fact, because the 1-pass cascade increases the compute required, +Cascade's 2D array utilization is lower than FLAT's at short sequence lengths. On the other hand, FuseMax (+Binding) achieves high utilization across the board and, at long sequence lengths, reaches almost 100% utilization. Both baselines achieve slightly higher utilization on XLM, which can be attributed to the higher intensity caused by a larger embedding dimension (E/F).

Figure 7 explores this phenomenon in more detail, breaking down the utilization by Einsum. FuseMax effectively hides both the costs of the memory traffic and softmax compute, allowing it to achieve high 2D array utilization while spending most of the cycles on the tensor products.

**Speedup.** Figure 8 shows that FuseMax achieves an average speedup of  $10\times$  over the unfused baseline and  $6.7\times$  over FLAT. We note FuseMax achieves lower speedup on XLM only because the baselines are able to achieve higher utilization of the 2D array on this transformer (Figure 6b).

![](_page_13_Figure_0.jpeg)

Fig. 12: Pareto-optimal curves for FuseMax at sequence length 256K.

**Energy.** Figure 9 shows that FuseMax uses 77% the energy of the unfused baseline and 79% the energy of FLAT.<sup>6</sup> The energy use of the unfused baseline and FLAT are dominated by the DRAM access energy, the global buffer access energy, and the QK and AV (Einsums 22 and 24) compute energy. FuseMax achieves its energy savings by significantly reducing the DRAM and global buffer access energies. In fact,  $\geq 95\%$  of the energy used by FuseMax across all models and sequence lengths goes to the compute performed by the MACC functional units in the 2D array.

