# *A. Experimental Setup*

*1) Hardware Setup:* We implement UNICORE in Spinal-HDL [\[10\]](#page-14-32) and synthesize the generated RTL using Synopsys Design Compiler targeting a TSMC 28 nm process and a 1 GHz clock frequency. This setup is used consistently across all accelerator comparisons. To assess system-level behavior, we extend the BitMoD cycle-accurate simulator [\[7\]](#page-14-30) with timing and energy models derived from our post-synthesis RTL. All accelerators are evaluated under the same area constraint and equipped with a 512 KB activation buffer and a 512 KB weight buffer, both modeled using CACTI [\[32\]](#page-14-33) followed the configurations of BitMoD to ensure a controlled comparison. Off-chip memory energy is estimated using the DDR4 configuration with a 25.6 GB/s bandwidth in DRAMSim3 [\[23\]](#page-14-34) for the prefill phase. For the decode phase evaluation, we include an additional HBM2 setting with a 256 GB/s bandwidth.

*2) Baselines:* We compare UNICORE against five representative GEMM accelerators for LLMs. For hardware evaluation, we use three bit-parallel composable-multiplier designs: OliVe [\[15\]](#page-14-10), Tender [\[22\]](#page-14-12), and M-ANT [\[18\]](#page-14-11). To cover fixedwidth mixed-precision designs, we additionally include two state-of-the-art accelerators BitMoD [\[7\]](#page-14-30) and AxCore [\[50\]](#page-15-4).

<span id="page-9-0"></span>![](_page_9_Figure_0.jpeg)

Fig. 14: Normalized area breakdown of C-PE and GEMM.

3) Quantization Settings: Our evaluation is based on posttraining quantization (PTQ). We compare the accuracy in four W/A/KV bit configurations: 4/4/16, 4/8/16, 3/8/16, and 4/4/4. We further include AxCore in its native 4/16/16 fixed-width configuration for comparison. We extend BitMoD and M-ANT to support integer activation quantization at different bit-widths, implementing their quantization schemes within a unified evaluation framework. To ensure a fair and consistent evaluation, we apply the calibration-free mode of BitMoD to M-ANT and AxCore, aligning all baselines with the evaluation protocol of UNICORE. All 8-bit settings use per-channel weight quantization and per-token activation quantization. In 4-bit and 3-bit quantization scenarios, for accelerators that natively support fine-grained group quantization (INT, MXFP4, M-ANT, BitMoD, AxCore, UNICORE), we fix the group size at 32. For architectures that do not support fine-grained quantization (OliVe and Tender), we employ channel-wise or tensor-wise quantization. UNICORE quantizes activations using standard floating-point formats together with an INT8 scale. When enabling K/V quantization, we also quantize Q and the softmax output P using the same group size and bitwidth as the activation, while only K and V perform online format selection (Section VII-D). All quantized models are obtained from BF16 checkpoints on Hugging Face [44].

### B. Area Efficiency

- 1) PE Level: Figure 14a compares the normalized area breakdown of composable PE (C-PE) of each design. The breakdown includes composable multiplication (Composable Mult), accumulators (Accum), and other logic; registers are excluded because different systolic dataflows require different buffering. Across multiplier-based baselines, the Composable Mult block dominates PE area (up to 83.6%) in some designs, due to the quadratic cost of partial-product generation. Instead, UNICORE yields a 13.6%–43% reduction in PE area relative to prior designs. The larger Others component—mainly the shifting logic for FP-to-sign-magnitude conversion—avoids a full FP adder and is reused across all bit-width modes.
- 2) GEMM Level: Figure 14b extends this comparison to the full GEMM unit, showing that UNICORE attains the smallest normalized GEMM area (0.51), corresponding to a 19%–49% area reduction over all baselines. Both the combinational logic

<span id="page-9-1"></span>![](_page_9_Figure_6.jpeg)

Fig. 15: Normalized compute density of the C-PE across different implementations and precisions.

<span id="page-9-2"></span>![](_page_9_Figure_8.jpeg)

Fig. 16: Normalized compute density (TOPS/mm<sup>2</sup>) of the GEMM array across different precisions.

and register overhead of UNICORE are the lowest among all designs, indicating that the S-FPMA-based pipeline reduces both the compute core and the surrounding buffering cost. To evaluate scalability up to 16-bit precision, we extend Tender to support W16A16. As shown in Figure 14b, 16-bit support slightly increases accumulator width but introduces only modest overhead; UNICORE still maintains a 24% smaller GEMM area than Tender. This demonstrates that the S-FPMA pipeline remains compact and preserves its area advantage even as precision increases.

