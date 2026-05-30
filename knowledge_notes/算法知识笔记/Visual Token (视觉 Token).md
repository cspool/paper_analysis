## Visual Token (视觉 Token)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Visual Token 是多模态 LLM 中由视觉编码器（如 CLIP-ViT）从图像/视频帧中提取的 embedding 向量序列。每张图像被划分为 patch（如 14×14 像素），每个 patch 经 ViT 编码为一个 D 维向量，构成一个 visual token。Video LLM 中对多帧采样，total visual token 数量 = 帧数 × 每帧 patch 数 × 每 patch token 数。

AIM 论文揭示的关键发现：multi-modal LLM 中 visual tokens 存在极高冗余——仅需 ~25% 的 visual tokens 即可维持 video 推理性能，FLOPs 却降低 77%。冗余原因：(1) 相邻 patch 高度相似；(2) 视频帧间大量重叠；(3) 许多 visual tokens 不携带对推理有贡献的信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Visual Token 的生命周期（MLLM 中的张量流转）**：

```
// 输入：Image I ∈ R^{H×W×3} 或 Video V ∈ R^{T×H×W×3}

// 1. 编码阶段
patches = PatchEmbed(I)  // [N_patches, patch_dim]
v = ViT(patches)          // [N_patches, D_vis]
// 例：CLIP-ViT-L/14：336×336 → 576 patches → 576 visual tokens × 1024 dims

// 2. 投影到 LLM 空间
v_llm = MLP_Adapter(v)    // [N_patches, D_llm]
// 例：576 × 4096 (Vicuna-7B)

// 3. 视频场景：多帧拼接
v_all = Concat([v_llm_frame1, v_llm_frame2, ..., v_llm_frameT])
// 例：32 frames × 576 = 18432 tokens → Adaptive Pooling → ~2304 tokens

// 4. 进入 LLM
x = Concat([v_all; text_tokens])  // [2304 + M, 4096]
// FLOPs ∝ (N_v + M)² —— visual tokens 主导计算量
```

**AIM 中的 Visual Token 压缩**：
- Token Merging（LLM 前）：N_v → N_v × 0.25，仅保留 25%
- Token Pruning（LLM 内部）：l₁~l₂ 层间 N_v 线性递减 → l₂ 层后 N_v = 0
- 文本 token 不受影响

术语一般如何实现？如何使用？

Visual tokens 由视觉编码器（CLIP-ViT、SigLIP 等）自动生成。AIM 无需修改编码器，在编码器输出后插入 Token Merging，在 LLM 层间插入 Token Pruning。Visual token 的数量控制是 AIM 自适应推理的核心杠杆。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

---
