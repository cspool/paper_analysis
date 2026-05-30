## Two-Level Quantization for Attention Map

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Two-Level Quantization 是 SageAttention3 提出的针对 attention map P（softmax 输出，值域 [0, 1]）的专用量化策略。由于 P 的值域极窄（[0, 1]），直接做 FP4 microscaling quantization 时，scale factor 的范围仅为 [0, 0.167]，导致 scale factor 在 E4M3 FP8 格式下仅能使用 280 个（35×8）有效量化输出值。Two-Level Quantization 通过两级量化将有效输出扩大至 1016 个（127×8）：(1) Level 1 — per-token 将 P 归一化到 [0, 448×6]，在 FP32 中无损；(2) Level 2 — 对归一化后的 P 做标准 FP4 microscaling quantization，此时 scale factor 充分利用 E4M3 的 127 个有效值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Level 1: Per-token normalization (FP32, lossless)
s_P1 = rowmax(P̃) / (448 × 6)    // shape: [B_q], FP32
P̃_2 = P̃ / s_P1                   // P̃_2 ∈ [0, 448×6], FP32

// Level 2: Standard FP4 microscaling quantization
s_P2, P̂ = φ(P̃_2)                 // s_P2 ∈ E4M3, range [0, 448]
                                  // P̂ ∈ E2M1 (NVFP4)

// Dequantization for PV MatMul
P̃ ≈ P̂ × s_P2 × s_P1              // Three-factor dequant
O = FP4MM(P̂, s_P2, V̂, s_V) × s_P1 // s_P1 applied as post-scaling
```
选择 448×6 是因为 448 是 E4M3 的最大 representable 正值，6 是 NVFP4 (E2M1) 的最大正数值（max(|FP4|) = 6）。rowmax(P̃)/(448×6) 保证 P̃/s_P1 的最大值恰好为 448×6，从而 s_P2 的最大值为 448。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：在 FlashAttention 的 inner loop 中，online softmax 计算 P̃ 后立即执行两级量化。Level 1 的 rowmax 可利用 online softmax 已计算的 rowmax 值（softmax 的 S 矩阵 rowmax），通过 max reduction over 16 个连续元素匹配 NVFP4 的 1×16 block 粒度。Level 2 复用 SageAttention3 已有的 FP4 microscaling 量化 kernel，融合到 softmax epilogue 中。仅增加一次 per-token element-wise division 和一次 scalar-vector multiplication，几乎无额外开销。使用场景：所有需要对 softmax 输出做极低比特量化的 attention 实现，特别是 FP4 等极低比特格式。

涉及论文标题：
- SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training
