## Token Recycling via Importance-Weighted Merging（基于重要性加权的Token回收/合并）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Recycling 是 VFlowOpt 提出的剪枝 token 信息保留机制。在标准 token 剪枝中，被丢弃的 token 信息永久丢失（不可逆操作）。Token Recycling 将 pruned tokens 按空间位置分组融合为 compact representations，重新纳入保留集合。流程：(1) 将图像平面划分为 a×a 正方形网格；(2) 各网格 cell 内的 pruned tokens 按重要性加权平均融合为单个 token：t_merged = Σ I_i · t_i / Σ I_i；(3) 融合 token 替换该网格内最高重要性 pruned token 的位置，归入 retained set。效果：在减少 token 数量的同时，将低重要性区域的视觉特征"压缩"保留而非丢弃。

与 ToMe (Token Merging, Bolya et al. 2023) 的关键区别：(1) ToMe 基于余弦相似度合并任意相似 token 对（无空间约束），Token Recycling 按空间网格分组（保持空间结构）；(2) ToMe 在 ViT 内部逐层执行以减少总 token 数，Token Recycling 仅在剪枝阶段执行（与保留集合并）；(3) ToMe 使用等权重平均，Token Recycling 使用重要性加权平均（高 token 贡献更多）；(4) Recycling 是剪枝的补充机制，仅在 token 被标记为 prune 后才触发。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Token Recycling in VFlowOpt
# Input: pruned tokens P, their features F_P, importance I_P, spatial coords C_P
#        grid size a, retained tokens R
# Output: augmented retained set R' (R + recycled tokens)

grid_cells = {}                                         # Map (p,q) -> list of (idx, feature, importance)
for each pruned token t_i with (x_i, y_i):
    p, q = floor(x_i / a), floor(y_i / a)              # Grid cell assignment
    grid_cells[(p,q)].append((i, F_P[i], I_P[i]))

for (p,q), cell_tokens in grid_cells:
    if len(cell_tokens) == 0: continue
    # Importance-weighted merging
    I_sum = sum(I_j for _, _, I_j in cell_tokens)
    t_merged = sum(I_j * F_j for _, F_j, I_j in cell_tokens) / I_sum
    # Replace position of highest-importance token in this cell
    i_max = argmax_j(I_j for _, _, I_j in cell_tokens)
    R[spatial_pos[i_max]] = t_merged                     # 归入保留集

# Total retained = R (top-k) + sum(len>0 for all cells) (recycled)
```

Annotations: 网格大小 a 由 Bayesian Optimization 搜索——a 越大则每 cell 覆盖更多 token（更多 token 融合为 1 个，更激进压缩），a 越小则保留更多细粒度空间信息。VFlowOpt 仅在 LLM 前的第一阶段剪枝后执行 Recycling（深层不再执行，因为 token 数已大幅减少后融合收益有限）。消融：移除 Recycling 导致 POPE 从 89.1→86.8（retain 25%），说明 Recycling 对 preserving coarse-grained semantics 关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在 PyTorch 中为每个 grid cell 执行 `torch.sum(I.unsqueeze(-1) * F, dim=0) / I.sum()` 即可完成加权平均融合。Position IDs 在剪枝/回收后保持不变，保留原始空间结构。使用场景：(1) 任何 token 剪枝框架中作为信息保留的补充机制；(2) 对高分辨率场景（如 LLaVA-OneVision 7290 tokens）尤其有效——大量低重要性背景 token 通过 Recycling 压缩为少数代表性 token 而非全部丢弃。开源实现见 https://github.com/sihany077/VFlowOpt。

涉及论文标题：
- VFlowOpt__A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_Flow-Guided_Optimization
