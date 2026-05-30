## Multi-Scale Salient Attention Distillation (MSAD / 多尺度显著注意力蒸馏)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MSAD 是 QuantSparse 论文提出的在校准阶段用于对齐量化 attention 与 FP attention 的高效蒸馏框架。它解决 video DiT PTQ 中朴素联合量化和稀疏注意力产生的 "amplified attention shift" 问题——量化噪声与稀疏 mask 相互增强，导致 attention 分布严重偏移。MSAD 的核心思路是避免直接存储和蒸馏完整 O(L²) attention 矩阵（对 HunyuanVideo-13B 的 L>10⁴ tokens, 单层 ~6.82GB 不可承受），转而通过两个互补的蒸馏分支以极少内存开销监督 attention 对齐：(1) Global Guidance——对 Q 和 K 做 average pooling 下采样（stride s=128），在低分辨率上计算 attention 并 MSE 蒸馏 FP 与 quantized 版本，捕捉全局结构拓扑（内存 O(L̃²), s=128 时仅 ~0.14GB）；(2) Local Guidance——利用 attention saliency 的重尾分布特性（<10% tokens 占据大部分 attention mass），仅对 FP 模型识别出的 top-k=256 salient queries 做高分辨率 attention 蒸馏（内存 O(kL)），保留细粒度关键细节。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MSAD 伪代码（校准阶段，每 transformer block）：

```
Input: X ∈ R^{L×d_in}, FP weights W_q,W_k,W_v, calibration data
Output: Quantized weights with optimized {s,z}

// FP forward
Q_fp = X·W_q^T, K_fp = X·W_k^T, V_fp = X·W_v^T  // FP16
A_fp = softmax(Q_fp·K_fp^T / √d_k)               // ∈ R^{h×L×L}

// Compute token saliency (Eq. 7)
s_j = Σ_h Σ_i A_fp[h,i,j]                         // aggregate attention received
I = top-k({s_j})                                   // select k salient queries

// Quantized forward
Q_q = Q(X)·Q(W_q)^T, K_q = Q(X)·Q(W_k)^T          // W4A8 quantized matmul
// Q(·) = s·(clip(⌊X/s⌋+z, 0, 2^b-1) - z)

// Global Guidance (Eq. 6)
Q̃_fp = AvgPool(Q_fp, s), K̃_fp = AvgPool(K_fp, s)  // stride s=128
Q̃_q = AvgPool(Q_q, s), K̃_q = AvgPool(K_q, s)
A_global_fp = softmax(Q̃_fp·K̃_fp^T / √d_k)
A_global_q = softmax(Q̃_q·K̃_q^T / √d_k)
L_global = MSE(A_global_fp || A_global_q)

// Local Guidance (Eq. 8)
A_local_fp = softmax(Q_fp[I,:]·K_fp^T / √d_k)     // only for salient queries
A_local_q = softmax(Q_q[I,:]·K_q^T / √d_k)
L_local = MSE(A_local_fp || A_local_q)

// Total loss (Eq. 9)
L_total = L_quant + λ_global·L_global + λ_local·L_local
// λ_global=1e-4 for Wan2.1; λ_global=1.0 for HunyuanVideo
// λ_local=1e-4 for Wan2.1; λ_local=1e2 for HunyuanVideo

// Optimize quant params {s,z}, channel-wise scale, rotation matrix
s,z ← AdamW(L_total, lr_scale=5e-2, lr_others=5e-3)
```

张量形状：L≈10400 tokens (720×1280, 60 frames → H=720/8=90, W=1280/8=160, T=60/4=15 → ~90×160×15/L≈21600, 经 compression to L≈10⁴); L̃=L/s²≈10400/128²≈0.64→1 tokens; k=256。MSAD 内存开销仅 +0.8% GPU memory, 校准时间 +1.6%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MSAD 在 QuantSparse 校准阶段与 block-wise PTQ 结合使用：逐 block 加载 → FP forward 计算 saliency + global/local attention targets → 量化 forward 计算对应 attention → MSE 蒸馏 → AdamW 优化量化参数（15 epochs, cosine LR）。全局和局部分支均以极小开销运行（Global 下采样 s=128 将 L² 降至 L²/16384, Local 仅 kL 计算）。MSAD 有效缓解量化+稀疏化的 attention shift, 使 W4A8+15% density 下 PSNR 从 14.35 提升至 18.72 (Wan2.1-14B 消融)。代码: https://github.com/wlfeng0509/QuantSparse（待发布）。

涉及论文标题：
- QuantSparse Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

---
