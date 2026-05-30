## Vector Quantization (VQ) for LLM Weight Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Vector Quantization（向量量化）是一种将高维浮点向量映射到有限个质心（centroid）的非均匀量化方法。与均匀量化（uniform quantization）将每个标量映射到等间隔固定点不同，VQ 将 d 个连续浮点值作为一个 d 维向量整体量化，映射到预先学习的 codebook C = {c_1, c_2, ..., c_k} 中的一个质心。每个质心 c_i ∈ R^d 是一个 d 维向量，存储 d 个浮点值。原始权重矩阵被划分为多个 group，每个 group 共享一个 codebook。存储时，每个 d 维向量仅需 ⌈log₂(k)⌉ 比特索引（加上 codebook 和 scale 的开销）。VQ 的核心优势：质心可以自由分布在 d 维空间中任意位置（非均匀 grid），能更灵活地匹配权重分布，在相同 bit budget 下可获得比均匀量化更高的 signal-to-quantization-noise ratio (SQNR)。对 LLM 推理，VQ 作为 storage data type（存储格式），推理时需先解码到 native data type 再参与计算。

GPTVQ 论文证实：d 越高（1D→2D→4D），SQNR 越高。但 codebook 大小随 d 指数增长（k = 2^{b·d}，b 为 bits/dimension），因此需在 codebook 大小、解码开销和精度之间权衡。GPTVQ 选择 2D VQ + 6-bit index（3 bits/dim）作为移动端最佳平衡点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
GPTVQ 中 2D VQ 的编码/解码流程：

**编码（量化）阶段**（离线，Algorithm 2/VQ_quant）：
```
# 输入: 权重矩阵 W[:, P] (r × d, d=2), codebook C_g (d × k)
# 输出: 量化权重 Q[:, P] + 索引矩阵 I
for each d-dim vector x = W[row, P]:
    # 找最优质心（E-step 公式 5）
    c_idx = argmin_m (x - c_m)^T D (x - c_m)
    # D = diag(1/H^{-1}_{11}, ..., 1/H^{-1}_{dd}) (Hessian 加权)
    Q[row, P] = C_g[:, c_idx]
    I[row] = c_idx  # 存储 6-bit index
```

**解码（推理）阶段**（在线，移动 CPU）：
```
# 输入: 6-bit index i, LUT C_g (64 entries × 8-bit), scale s
v1 = TBL(C_g_dim0, i)  # 查表: dimension 0 的 8-bit 值
v2 = TBL(C_g_dim1, i)  # 查表: dimension 1 的 8-bit 值
w_fp = s * (v1 + v2)   # 合并两维 + 反量化
```

格式总 bpv 计算：bpv = log₂(k)/d + k·d·b_c/l，其中 k=质心数，d=VQ 维度，b_c=codebook bit-width，l=group size。GPTVQ 默认 k=64, d=2, b_c=8, l=8192，得 bpv = 6/2 + 64·2·8/8192 = 3 + 0.125 = 3.125。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VQ for LLM 的代表性方法：AQLM (Egiazarian et al. 2024) 使用 d=8 + 16-bit indices + block fine-tuning，适合云端 GPU；GPTVQ (van Baalen et al. 2024) 使用 d=2 + 6-bit indices 移动 CPU TBL 指令，适合移动端；QuIP# (Tseng et al. 2024) 使用 lattice codebook + Hadamard 旋转。实现框架：GPTVQ 基于 PyTorch（量化）+ 自研 C 推理引擎（CPU 解码）。关键实现考量：(1) codebook 大小受硬件 LUT 指令限制（移动 CPU TBL 仅支持 5-6 bit index）；(2) d 越大 SQNR 越高但解码越慢（需更多 TBL 调用）；(3) codebook 本身也需量化（INT8/INT4）以减少 overhead。

**VQ for MoE LLMs（KBVQ-MoE 的扩展）**：KBVQ-MoE (ICLR 2026) 将 VQ 专门适配到 MoE 架构。核心区别：(1) 不直接对所有 expert 权重做 VQ，而是先通过 KLT 引导的 SVD（IDRE）提取共享低秩分量保持全精度，仅对 expert-specific 残差做 VQ；(2) VQ 后通过 per-channel affine compensation（BCOS）校正量化输出的 mean/variance 偏移。KBVQ-MoE 使用 K-means++ 初始化 codebook（100 iterations），子向量长度 d=4，在 Qwen1.5-MoE-A2.7B 上 2-bit VQ 实现 87% 压缩率（27.9GB→4.3GB），3-bit 下 Avg Acc 67.99 接近 FP16 的 68.07。

涉及论文标题：
- GPTVQ: The Blessing of Dimensionality for LLM Quantization
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance
- KBVQ-MoE KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models

---
