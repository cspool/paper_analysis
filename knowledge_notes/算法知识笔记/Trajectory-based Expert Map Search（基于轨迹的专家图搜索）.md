## Trajectory-based Expert Map Search（基于轨迹的专家图搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Trajectory-based Expert Map Search 是 FineMoE 的第二种 expert map 检索方式，利用已观察到的前 (l-d) 层 expert probability distributions（称为 "expert trajectory"）与 Expert Map Store 中历史 expert maps 对应层的 cosine similarity，检索最匹配的 historical expert map。Expert trajectory 定义为 "从 Layer 1 到当前 visible layer 的 gate network probability distributions 序列"。用于第 l ∈ [d+1, L] 层（prefetch distance 之后，已有足够的 trajectory history）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Trajectory-based Expert Map Search 流程：

Input: current_trajectory[tokens] ∈ R^{(l-d)×J} (前 l-d 层的 gate probability)
        Expert Map Store historical maps ∈ R^{C×L×J}
        target_layer l ∈ [d+1, L]

# 对于每次 inference iteration 的每个 layer l:
for l in range(d+1, L+1):
    # Step 1: 收集前 (l-d) 层的 expert trajectory
    traj_new = concat([P_1, ..., P_{l-d}])  # R^{(l-d)×J} flattened to R^{(l-d)·J}

    # Step 2: pairwise cosine similarity with historical
    traj_old = map_old[:, :(l-d), :].reshape(C, -1)  # R^{C×(l-d)·J}
    score_traj ∈ R^{B×C} = cos_sim(traj_new, traj_old)

    # Step 3: 选择最相似 historical iteration
    best_iter = argmax(score_traj, dim=-1)

    # Step 4: 提取该 iteration 的第 l 层 expert map
    P_l = map_old[best_iter, l, :]  # R^{J}
    
    # Step 5: similarity-aware expert prefetching for layer l
    prefetch_experts_with_similarity_aware_selection(P_l, score_traj)

# 特点：随着 l 增大，(l-d) 增大 → trajectory 信息量增加 → prediction 更准确
# 例：l=d+1 时仅用 1 层 trajectory →; l=L 时用 L-d 层 trajectory（最多信息）
```

与 Semantic Search 的协同关系：
- Semantic search: 适应初始层（无 trajectory history），利用 prompt 全局语义
- Trajectory search: 适应后续层（轨迹越长越准），利用 expert selection 的序列依赖性
- 两者通过 unified redundancy score 统一为单一 map store

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch native cosine_similarity 计算。traj_new 按需拼接前序层的 gate outputs（每层 gate 输出被缓存）。Pearson correlation analysis 表明 trajectory similarity 与 expert hit rate 正相关（所有 model-dataset 组合）。随着 l 增大（trajectory 信息增加），trajectory-based prediction 准确度持续提高，弥补了 semantic-based 在后期层的不足。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
