## Hierarchical Attention in Vision Encoders（视觉编码器中的分层注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Attention in Vision Encoders 是 HiPrune 论文揭示的视觉编码器（Vision Transformer, ViT）内部注意力分布规律：不同深度的 ViT 层对不同语义层次的图像信息产生差异化关注——浅层注意力分散、中间层聚焦物体区域（object-centric）、深层编码全局上下文（global representation）。具体表现为三个阶段的渐进过渡：(1) 浅层（Layer 1~L/3）：注意力分布相对均匀，token 间注意力排名差异小，embedding 空间中高注意力 token 分布分散；(2) 中间层（Layer L/3~2L/3）：注意力向图像中的 main object 集中，top-10% 高注意力 token 与 COCO segmentation mask 的 IoU 达最大值（CLIP-L: 1× 在 L/2，SigLIP: 1× 在 L/2，DeiT: 1× 在 L/2，V-JEPA2: 1× 在 L/2）；(3) 深层（Layer 2L/3~L）：注意力从 object cluster 扩散至全图均匀分布，编码全局上下文信息，可作为有限 token budget 下的理想全局指标。该模式跨 CLIP、SigLIP、SigLIP2、DeiT、V-JEPA2 五种架构一致存在，与预训练数据或模型架构设计无关。t-SNE 投影显示注意力排名在相邻层之间呈现连续轨迹，证明注意力转移是渐进有序的。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 token pruning 中使用分层注意力：
```
# === Hierarchical Attention for Token Selection ===
# 给定 ViT 各层 attention maps: all_attns = [A_1, ..., A_L]
# A_l: (H, N+1, N+1), H=num_heads

# 1. 从 object layer l (中间层) 提取 object-centric attention
mid_attn = all_attns[l][:, 1:, 1:].mean(0).sum(0)  # (N,)
# a_i^{[l]} = mean_h sum_n A_h[n, i]  每个 token 收到总关注度
anchor_idx = topk(mid_attn, k=N_a)               # 选物体区域 token

# 2. 从输出层提取 global attention  
deep_attn = all_attns[-1][:, 1:, 1:].mean(0).sum(0)
register_idx = topk(deep_attn, k=N_r)            # 选全局信息 token

# 3. 组合: Anchor (细节) + Buffer (空间邻居) + Register (全局)
retained = [anchor_idx, buffer(anchor_idx), register_idx]
```

Object layer l 的选择方法（dispersion-based searching）：
```
# 各候选层计算 top-k 高注意力 token 的平均 pairwise 距离
for l in candidate_layers:
    top_tokens = embeddings[l][topk(attention[l], k=K)]
    pairwise_dist[l] = mean(||t_i - t_j||_2 for all pairs)
# 选择 pairwise_dist 变化最剧烈的"临界点"层作为 object layer
# CLIP-L/14: l=9 (共 24 层), pairwise_dist 在 layer 9 处明显跃变
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
分层注意力模式的使用：(1) 无需额外训练——直接从 ViT forward pass 中获取 attention map。(2) Object layer 通过 dispersion-based searching 确定（计算 top-K token pairwise distance 的拐点），一次确定后固定使用。(3) HiPrune 在 LLaVA-1.5 (CLIP-L/14, 24 layers) 中使用 layer 9，在 LLaVA-NeXT (CLIP-L/14) 中使用 layer 9，在 Qwen2.5-VL (SigLIP, 27 layers) 中使用 layer 16。(4) 该模式验证跨 CLIP-L/B、SigLIP、SigLIP2、DeiT、V-JEPA2 五类编码器，通过 COCO val2017 的 segmentation mask IoU 定量验证（Table 1: 中间层 top-10% token IoU 归一化值均为 1×，浅层/深层仅 0.26×~0.82×）。(5) 与 VAR-Turbo 中的 Learning Region/Inert Region 分区（knowledge_notes: score 44.8）形成补充——两者都揭示了 Transformer 层的 attention 行为随深度系统性地变化，但 HiPrune 聚焦于"attention 关注什么语义内容"，VAR-Turbo 聚焦于"attention 是否还有学习能力（高频信息保留程度）"。

涉及论文标题：
- HiPrune__Training-Free_Visual_Token_Pruning_via_Hierarchical_Attention_in_Vision-Language_Models
