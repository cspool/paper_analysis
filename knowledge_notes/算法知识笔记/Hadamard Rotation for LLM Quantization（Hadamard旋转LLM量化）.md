## Hadamard Rotation for LLM Quantization（Hadamard旋转LLM量化）

术语是什么？通过联网搜索让回答具体和精准。

Hadamard Rotation是在LLM weight-activation量化中使用Hadamard矩阵对activation进行正交变换以平滑outlier的技术。Hadamard矩阵H是元素仅为+1和-1的正交矩阵（H·H^T = I）。当H乘到activation矩阵X上时，它将每个channel的值重新分布到所有channel的线性组合中，从而平滑集中在特定channel的极端值（channel-wise outliers）。为保持数学等价性，weight矩阵需在offline左乘H^T。该技术由QuaRot首次提出用于4-bit uniform量化，RoMeo将其扩展为rotation+mixed precision的两阶段方案。

从算法pipeline角度拆解术语：

Hadamard rotation在LLM量化pipeline中的计算流程（以RoMeo为例）：

```
// 原始Linear层: Y = X · W
// 插入Hadamard rotation后: Y = (X·H) · (H^T·W)

Offline阶段（weight预处理）:
  W_rotated = H^T · W  // 离线完成，固定cost

Online阶段（每个token推理）:
  // Step 1: Activation Hadamard Rotation
  X_rotated = Fast_Walsh_Hadamard_Transform(X)  // O(MK log K)
  // 效果: peak activation从1272降至58.5（RoMeo实测）
  
  // Step 2: Token-wise Mixed Precision Quantization
  // 旋转后outlier呈纯token-wise分布
  for t in 0..M:
      if is_outlier_token(t):
          X_Q[t,:] = INT8_quantize(X_rotated[t,:])
      else:
          X_Q[t,:] = INT4_quantize(X_rotated[t,:])
  
  // Step 3: Cross-precision GEMM
  Y_low = X_Q_int4 · W_rotated_Q_int4  // W4A4
  Y_high = X_Q_int8 · W_rotated_Q_int8  // W8A8 (等)
  Y = combine(Y_low, Y_high)
```

Hadamard矩阵的递归结构（以H_4为例）：
```
H_1 = [1]
H_2 = [1  1; 1 -1]
H_4 = [H_2  H_2; H_2 -H_2]
     = [1  1  1  1; 1 -1  1 -1; 1  1 -1 -1; 1 -1 -1  1]
```

乘法可通过Fast Walsh-Hadamard Transform (FWT)在O(MK log K)复杂度内实现（vs naive O(MK^2)），相比Linear层的O(MKN) GEMM可忽略。RoMEo使用HadaCore库实现FWT。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现库：HadaCore（Tensor Core accelerated Hadamard transform kernel，K. Agarwal et al., 2024），QuaRot提供了参考PyTorch实现。
- 在线FWT的开销：RoMeo实测Hadamard transformation仅占layer latency的~4%（batch=64时），显著低于GEMM主导成本。
- 注意事项：
  (1) Hadamard matrix要求维度为2的幂，非2的幂需padding或truncate。
  (2) 旋转在模型中不同位置的应用策略影响性能——QuaRot在Qwen3-14B上因在attention heads间插入旋转导致性能下降（40 heads→inefficient transformation），RoMeo在heads的hidden dimension上应用旋转避免此问题。
  (3) 旋转非无损——它将model weight的数值分布改变，可能影响某些精度敏感层的quality，但正交性保证数学等价（X·W = X·H·H^T·W）。
- 旋转可与其他quantization技术正交组合：如SpinQuant（learned rotations）、DuQuant（dual transformation）、FlatQuant等优化旋转矩阵的方法可与RTMPQ的mixed precision方案相结合。
- **CoRFiG (Coarse Rotation, Fine Grouping)**：GyRot提出将rotation scope R与group quantization粒度G解耦——R=2^g·G, R≤1024。全局rotation导致outlier dispersed across all channels → 与group quantization的localized scaling冲突（小G下RTN PPL从7.40升至30.12 at G=32）。CoRFiG限制rotation scope到R=1024，既保留distribution flattening benefit又保留group-level local variance→G=32+R1024 PPL=6.91。配合HAP（Harmonic-Aligned Permutation，利用Hadamard harmonic rows对齐outlier channels）进一步tighten per-group range→INT8 SF可达FP16 SF parity。

涉及论文标题：
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization
- GyRot: Leveraging Hidden Synergy between Rotation and Fine-grained Group Quantization for Low-bit LLM Inference

