# B.1 MULTI-ASPECTS METRICS EVALUATION

This evaluation suite includes absolute quality of videos and relative difference metrics that quantify the difference between FP16 generation.

**Absolute Quality.** Consistent with prior quantization works (Zhao et al., 2024; Feng et al., 2025c), we adopt CLIPSIM, VQA, and FlowScore to measure text-video alignment, quality, and temporal consistency, respectively.

**Relative Difference Metrics.** Following prior sparse attention works (Xi et al., 2025; Yuan et al., 2024; Ren et al., 2025; Zhang et al., 2025d), we adopt Peak Signal-to-Noise Ratio (PSNR), Structural Similarity Index Measure (SSIM), and Learned Perceptual Image Patch Similarity (LPIPS) for pixel-space differences, structural similarity, and high-level patch similarity, respectively.

All the evaluations are conducted on high-resolution generation tasks. Due to the computational overhead, we use the OpenSORA prompt sets used in (Zhao et al., 2024; Feng et al., 2025c) for video generation.

