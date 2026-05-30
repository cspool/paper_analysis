## Progressive Token Dropping（渐进式Token丢弃）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Progressive Token Dropping 是一种多阶段 token 剪枝策略，在 LLM decoder 的多个 depth（浅层→中层→深层）逐步减少 visual token 数量，形成递减的 token 金字塔结构（M → K_a → K_b → K_c），而非在单层一次性丢弃大量 token。V2Drop 采用三层渐进式剪枝：shallow layer（如 layer 3）执行首次筛选，middle layer（如 layer 17）进一步求精，deep layer（如 layer 22）最终压缩到目标数量。渐进式策略的核心优势：早期层的粗筛保留足够 token 供后续层细化选择，避免一次性丢弃过多可能重要的 token；每层基于该层的 variation 信息重新评分，利用更深层的语义理解进行更精准的筛选。

Ablation 结果（V2Drop, LLaVA-1.5-7B, retain 192 tokens）：progressive dropping 相比 one-time dropping 在 POPE 上提升 9.3%、在 MME 上提升 5.9%，证明渐进式策略有效保留关键视觉信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Progressive Token Dropping Schedule
# Input:  M vision tokens, pruning layers L = [l_a, l_b, l_c]
# Output: K_c retained tokens

M_curr = M  # e.g., 576 for LLaVA-1.5
for l in LLM_layers:
    h = TransformerLayer_l(h)
    if l == l_a:   # shallow: aggressive first cut
        M_curr = retain_top_k_by_variation(h, K_a)  # e.g., K_a = 288 (50%)
    elif l == l_b: # middle: further refinement
        M_curr = retain_top_k_by_variation(h, K_b)  # e.g., K_b = 173 (30%)
    elif l == l_c: # deep: final selection
        M_curr = retain_top_k_by_variation(h, K_c)  # e.g., K_c = 128 (22%)

# One-time dropping (for comparison):
# Drop all at single layer: M → K_c in one step
```

关键设计选择：(1) 剪枝层位置——V2Drop 的 ablation（Table 6）显示 (3, 17, 22) 组合最优（97.6% 原性能），但 (3, 15, 27) 等也达到 97.0%+，对层选择鲁棒；(2) 每层压缩比分配——由 performance-efficiency trade-off 调控。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在每层剪枝后重组 token sequence（丢弃的 visual token 位置被删除，缩短后续层的序列长度），直接减少后续所有 attention 和 FFN 的 FLOPs。使用场景：(a) 需要细粒度控制 compression ratio vs. performance 的部署场景；(b) 视频 LLM 场景，长 visual token sequence 需要多阶段筛选；(c) 与 variation/attention/duplication 等任意评分信号配合。PDrop (CVPR 2025) 和 V2Drop (CVPR 2026) 均采用此策略。开源实现见 https://github.com/xuyang-liu16/V2Drop 和 https://github.com/XingLuan/PyramidDrop。

**VFlowOpt 的三阶段渐进式剪枝**（来自 VFlowOpt 论文）：
VFlowOpt 将 LMM 均分为三个阶段，每阶段开始按阶段特定保留率 R=[R1, R2, R3] 保留高重要性 token。剪枝点位置取决于模型层数：LLaVA-OneVision-7B 在 LLM 前、第 9 层后、第 18 层后；LLaVA-NeXT-7B 在 LLM 前、第 10 层后、第 20 层后。每阶段先计算重要性得分，按 R_current 保留 top-k token，剩余 token 进入 recycling（按 a×a 网格加权平均融合后替代最高重要性 token 位置归入保留集合）。整体平均保留率公式：R̄ = (R1·L1 + R1·R2·L2 + R1·R2·R3·L3)/L，其中 L1/L2/L3 为三个阶段各含层数。与 V2Drop/PDrop 的关键区别：(1) 三阶段位置由均匀划分决定（非启发式选择特定层号）；(2) 每阶段执行 importance re-scoring（非仅首次评分后传递）；(3) pruned token 进入 recycling 而非直接丢弃。

涉及论文标题：
- V2Drop__Variation-aware_Vision_Token_Dropping_for_Faster_Large_Vision-Language_Models
- VFlowOpt__A_Token_Pruning_Framework_for_LMMs_with_Visual_Information_Flow-Guided_Optimization
