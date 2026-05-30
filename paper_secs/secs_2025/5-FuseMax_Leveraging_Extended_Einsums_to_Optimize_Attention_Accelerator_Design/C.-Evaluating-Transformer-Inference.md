# C. Evaluating Transformer Inference

To evaluate the benefits of FuseMax on end-to-end transformer inference, we include the other required linear layers (Section IV-A). We use Timeloop to search for optimal mappings for these linear layers and use the same mappings for all three accelerator configurations. The attention modeling remains the same as Section VI-B.

**Speedup.** Figure 10 shows the performance improvement achieved by FuseMax. Across the sequence lengths tested, FuseMax achieves an average speedup of  $7.6\times$  over the unfused baseline and  $5.3\times$  over FLAT. As discussed in Section IV-A, as sequence length grows, attention becomes a larger fraction of the total required compute. Therefore, at 1M tokens, FuseMax achieves an average  $10\times$  speedup over the unfused baseline and  $7.5\times$  speedup over FLAT.

**Energy.** Figure 11 shows the energy reduction achieved by FuseMax. Here, we see similar results: as attention becomes a larger fraction of the kernel, the energy reduction increases. FuseMax uses 82% of the unfused baseline and 83% of FLAT's energy during end-to-end inference.

