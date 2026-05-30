## Structural Redundancy-Aware Pruning (SRAP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Structural Redundancy-Aware Pruning (SRAP) 是 QuantCache 论文提出的在线（runtime）层剪枝机制，针对 Diffusion Transformers (DiTs) 在单个 timestep 内的结构冗余进行自适应剪枝。核心观察：DiT 的某些层在同一个 timestep 内表现出显著的 representational overlap（表示重叠），意味着某些层的计算可以被剪枝而不损失信息。SRAP 包含两个关键组件：(1) **Layer-wise Cosine Similarity Pruning**：在 timestep t 内计算相邻层 l 和 l+1 的 feature cosine similarity：S_t^(l,l+1) = ⟨p_t^(l), p_t^(l+1)⟩ / (||p_t^(l)|| · ||p_t^(l+1)||)。当 S_t^(l,l+1) > τ_high → 完全跳过 layer l+1（P_prune=1）；当 τ_low ≤ S ≤ τ_high → 以概率 P_base 随机剪枝；当 S < τ_low → 不剪枝（P_prune=0）。(2) **Adaptive Temporal Pruning Rate**：跟踪跨 timestep 的累积 feature variation V_t = Σ_{i=0}^k ||p_t - p_{t-i}||_1。当 V_t < δ_low（扩散过程处于精细 refine 阶段）→ 增加全局剪枝概率；当 V_t > δ_high（剧烈内容变换阶段）→ 减少剪枝以维持信息流。SRAP 联合 HLC 和 AIGQ 形成三层次计算优化，最终实现 6.72× total speedup。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SRAP 在每个 timestep 内的执行流程：
```python
# SRAP: Structural Redundancy-Aware Pruning (per-timestep)
# 计算全局时序累积变化 V_t，决定当前 timestep 的整体剪枝激进程度
V_t = sum(norm(p_t[l] - p_{t-i}[l], 1) for i in range(k) for l in layers)
if V_t < delta_low:       # 精细 refine 阶段 → 激进剪枝
    global_prune_scale = 1.5
elif V_t > delta_high:    # 剧烈变化阶段 → 保守剪枝
    global_prune_scale = 0.5
else:
    global_prune_scale = 1.0

# 逐层剪枝决策
for l in range(num_layers - 1):
    S = cosine_similarity(p_t[l], p_t[l+1])  # Eq. 9
    if S > tau_high:
        P_prune = 1.0         # 高度冗余 → 必剪
    elif S >= tau_low:
        P_prune = P_base * global_prune_scale  # 中等冗余 → 概率剪枝
    else:
        P_prune = 0.0         # 低冗余 → 不剪
    if random() < P_prune:
        skip_layer(l+1)       # 跳过该层；输出直接复用上一层 feature
        p_t[l+1] = p_t[l]     # feature copy forward
```
SRAP 区别于传统的 static layer pruning（预定义固定子集剪枝）——它在运行时根据实时 feature similarity 动态决策，使剪枝行为与当前生成内容的自适应匹配。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SRAP 完全在推理时在线执行，无需训练或校准：(1) 在每个 timestep 内对每对相邻层计算 cosine similarity（基于 FP16/量化后的 feature）；(2) 通过预设阈值 τ_high、τ_low 和 P_base 进行决策；(3) 剪枝的 overhead 极小（仅 cosine similarity 计算 + 随机数），远小于被跳过的 full layer computation。SRAP 最适用于 DiT-based 视频生成模型，因为这些模型的相邻层在 denoising 中后期（高相似度）表现出显著冗余。SRAP 有效补充了 HLC（跨 timestep 缓存）和 AIGQ（精度降维），形成时间-层-精度三维联合优化。开源实现见 https://github.com/JunyiWuCode/QuantCache。

涉及论文标题：
- QuantCache Adaptive Importance-Guided Quantization with Hierarchical Latent and Layer Caching for Video Generation

---
