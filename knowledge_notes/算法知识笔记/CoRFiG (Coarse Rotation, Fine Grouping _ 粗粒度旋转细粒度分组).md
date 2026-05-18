## CoRFiG (Coarse Rotation, Fine Grouping / 粗粒度旋转细粒度分组)

术语是什么？

CoRFiG是GyRot提出的旋转与分组量化协同策略。它将Hadamard rotation的scope R限制在coarse粒度（如R=1024），同时保持group quantization的group size G在fine粒度（如G=32），满足R=2^g·G关系（g为正整数）。这解耦了rotation的outlier dispersion scope与group quantization的localized scaling granularity。

从算法pipeline角度拆解术语：

全局rotation将outlier分散到所有channel，与fine group quantization（G=32）的localized scaling冲突——小G下group quantization自身即可捕获local variance，全局rotation反而引入inter-group interference。CoRFiG的pipeline（以LLaMA-3-8B, R=1024, G=32, g=5为例）：

```
// CoRFiG量化pipeline
Offline 权重旋转:
  for each weight matrix W with Nch channels:
    // 分割为Nch/R个rotation blocks
    for each block b of size R=1024:
      H_block = Hadamard(R)  // R×R Hadamard matrix
      W_rot[b] = H_block^T · W[b]  // 局部旋转，仅影响R channels内
    // HAP permutation: outlier channels → harmonic rows (可fuse进weight)
    W_rot = HAP_permute(W_rot)
    // Group量化: G=32 per group
    for each group g of size G=32:
      W_Q[g] = INT4_quantize(W_rot[g], SW[g])  // SW: INT8 scale

Online 推理 (每token):
  // Step 1: Online rotation (仅在非线性层后)
  if layer_has_nonlinear_before:
    for each rotation block of size R:
      X_rot = FHT(X)  // Fast Hadamard Transform, O(R log R)
    // 量化: G=32 per group
    for each group g:
      zx = ceil(-min(X_rot[g]))  // 直接unscaled domain计算 (reformulated asym)
      sx = (max(X_rot[g]) + zx) / (2^b - 1)
      X_Q[g] = clip((X_rot[g] + zx) / sx, 0, 2^b-1)  // INT4

  // Step 2: INT4 GEMM + Integer Dequantization
  for each group g:
    partial_sum = X_Q[g] · W_Q[g]  // INT4 dot product
    y[g] = SW[g] * (SX[g] * partial_sum - ZX[g] * WSUM[g])  // 全整数dequant

// 关键对比:
// global rotation (Quarot): X_rot = H_global · X → outlier disperse Nch-wide → G=32下PPL=7.04
// CoRFiG (GyRot): X_rot_block = H_R · X_block → outlier disperse R-wide → G=32下PPL=6.91
// No rotation + G32: PPL=7.40 (RTN) → CoRFiG achieves synergy
```

术语一般如何实现？如何使用？

CoRFiG的关键参数选择：(1) R选择：GyRot的FHT硬件支持2的幂次rotation scope up to 1024，实验显示R=1024时PPL饱和（Table III: R=1024 PPL=6.91 vs R=512 PPL=6.89→接近收敛）。(2) G选择：G=32为GyRot PE的最小支持粒度（32-way INT4 dot product），G=32配合R=1024实现最佳accuracy-hardware tradeoff。(3) R=2^g·G约束：保证rotation block内group对齐，使HAP harmonic row alignment的benefit均匀覆盖所有group。GyRot对比LightRot (R=G=128) 的关键区别在于CoRFiG解耦允许更fine的G而不损失rotation benefit。

涉及论文标题：
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

