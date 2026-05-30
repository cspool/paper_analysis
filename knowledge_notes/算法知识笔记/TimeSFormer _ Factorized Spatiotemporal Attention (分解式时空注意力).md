## TimeSFormer / Factorized Spatiotemporal Attention (分解式时空注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TimeSFormer (Time-Space Transformer) 是 Facebook Research (Bertasius et al., ICML 2021) 提出的纯 Transformer 视频分类架构，核心创新是 Factorized (Divided) Spatiotemporal Attention：将 3D 联合时空注意力分解为两个独立的顺序操作——spatial self-attention（同帧内不同 patch 间）→ temporal self-attention（同 spatial position 跨帧间）。这种分解将计算复杂度从 O((T·N)²)（联合时空）降至 O(T·N² + N·T²) ≈ O(N² + T²)，且 counterintuitively 比联合注意力更准确（分离强加了有用的 inductive bias）。HORNet 使用 TimeSFormer-Tiny 作为 video encoder（patch_size=16, T=32, D=768），提取 per-frame spatiotemporal features 供 frame selection policy 使用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === TimeSFormer in HORNet ===
# 输入: V (T=32 frames, 288×288×3), P=16, D=768

# Step 1: Patchify
patches = conv2d(V, kernel=16, stride=16)  # (T, 18, 18, 768)

# Step 2: Spatial Self-Attention (per frame)
for t in 1..T:
    x_t = patches[t].flatten()          # (324, 768)
    x_t = FlashAttention(Q(x_t), K(x_t), V(x_t))  # intra-frame

# Step 3: Temporal Self-Attention (per patch position)
for (i,j) in grid(18,18):
    x_ij = [x_1[i,j], ..., x_T[i,j]]   # (T, 768)
    x_ij = FlashAttention(Q(x_ij), K(x_ij), V(x_ij))  # cross-frame

# Step 4: Spatial Average Pooling → per-frame descriptors
F = avg_pool_2d(x)                      # (T, 768)
```

复杂度：Divided Space-Time = O(TN² + NT²) vs Joint Space-Time = O(T²N²)。T=32, N=324 时，Divided ≈ 105K + 1K token pairs vs Joint ≈ 110M pairs。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HuggingFace Transformers 提供 `TimesformerModel`（`attention_type='divided_space_time'`）。HORNet 使用 TimeSFormer-Tiny 变体从预训练权重初始化，在 GRPO 训练中与 MLP 联合微调。Spatial self-attention 用标准 `nn.MultiheadAttention` per-frame batch，temporal self-attention 用 `einops.rearrange` 重排维度后 batch attention。兼容 FlashAttention。HORNet 中 encoder + policy 共 <1M trainable params。

涉及论文标题：
- HORNet__Task-Guided_Frame_Selection_for_Video_Question_Answering_with_Vision-Language_Models
