## Learnable Importance Scorer (LIS / 可学习重要性评分器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learnable Importance Scorer (LIS) 是 VisionSelector 中用于评估视觉 token 全局重要性的轻量级模块（仅 12.85M 参数 / Qwen2.5-VL-7B）。LIS 解耦于 MLLM backbone，通过两层线性投影计算每个 token 的全局上下文感知重要性得分：给定视觉 token V ∈ R^{N×D}（N 个 token，D 维特征），投影为 Query Q = VW_q 和 Key K = VW_k（W_q, W_k ∈ R^{D×d}，d 为投影维度默认 1792 = D/2），计算简化自注意力矩阵 A = QK^T/√d ∈ R^{N×N}，每个 token 的重要性得分 s_i = (1/N)·Σ_{j=1}^{N} A_{ij}（全局平均池化）。该设计使 LIS 能同时感知所有 token 间的全局交互关系，而非仅依赖局部特征或预训练 attention map。训练时仅更新 LIS 参数（W_q, W_k），冻结 MLLM 全部参数，使用 near-zero initialization 确保初始训练稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === LIS 前向计算 ===
# 输入: V ∈ R^{N×D}, 来自 Vision Encoder + Projector
# 参数: W_q, W_k ∈ R^{D×d}, d << D (d=D/2)

V_norm = LayerNorm(V)           # 输入归一化
Q = V_norm @ W_q                # R^{N×d}
K = V_norm @ W_k                # R^{N×d}
A = Q @ K.T / sqrt(d)           # R^{N×N}, 全局 token 交互矩阵
s_i = mean(A[i, :])             # R^{N}, per-token 重要性得分

# 计算复杂度: O(N²·d), 远小于 LLM self-attention 的 O(N²·D)
# N ≈ 2000 visual tokens, d = 1792, D = 3584 (Qwen2.5-VL-7B)
# LIS FLOPs / LLM self-attn FLOPs ≈ d/D = 0.5 (单层对比)

# === 与 FastV/VisionZip 的关键区别 ===
# FastV: s_i = mean(text→vision attention scores)  (依赖 LLM 内部预训练 attn)
# VisionZip: s_i = mean(末层 vision encoder attn map)  (text-agnostic)
# LIS: s_i = mean(QK^T)  (独立于 backbone, 端到端学习)
```

Annotations: LIS 使用 Qwen2.5-VL-7B hidden_dim D=3584 的一半 (d=1792) 作为投影维度。在 Qwen2.5-VL-3B 上 d=1024 (D=2048/2)。在 LLaVA-OneVision-1.5-8B 上 d=2048。Near-zero initialization: W_q, W_k 初始化为接近零的小值，确保训练初期 s_i 接近均匀分布，LIS 不干扰 MLLM backbone 的预训练知识。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
LIS 作为 plug-and-play 模块部署在 modality interface 与 LLM 之间：Vision Encoder → PatchMerger → Projector → **LIS** → DiffTopK/Mask → LLM。训练时与 MLLM backbone 完全解耦：LR=5e-5, AdamW, cosine annealing, 1 epoch (144K samples), 8 A800 GPUs, DeepSpeed ZeRO-3, 约 40 分钟完成训练。推理时 LIS 计算仅增加极小的开销（2×矩阵乘+QK^T+mean），与 FlashAttention 完全兼容。LIS 的全局交互设计使其能消除 attention sink 偏差：因为得分是从 LIS 自己学习的交互矩阵计算，而非依赖预训练 attention map 中首 token 的虚假高 attention。工作原理解释：LIS 在训练中通过下游 CE loss 学习识别对任务回答最关键的视觉 token，而非像 FastV 那样依赖可能存在 bias 的预训练 attention。

涉及论文标题：
- VisionSelector__End-to-End_Learnable_Visual_Token_Compression_for_Efficient_Multimodal_LLMs
