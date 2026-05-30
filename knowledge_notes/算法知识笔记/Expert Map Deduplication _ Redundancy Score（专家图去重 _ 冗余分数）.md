## Expert Map Deduplication / Redundancy Score（专家图去重 / 冗余分数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Map Deduplication 是 FineMoE Expert Map Store 的容量管理策略：当 Expert Map Store 达到容量上限 C（默认 1K）时，通过计算新 iteration data 与历史 data 的 pairwise redundancy score 来判断哪些 historical expert maps 是冗余的（即新 data 已能覆盖其 expert selection pattern 空间），并剔除冗余 maps 以维持 store 的 pattern diversity。Redundancy score 统一了 semantic similarity 和 trajectory similarity：

RDY_{x,y} = (d/L) × score^{sem}_{x,y} + ((L-d)/L) × score^{traj}_{x,y}

其中 d 是 prefetch distance，L 是总层数，x 是新 batch iteration index，y 是历史 iteration index。权重 (d/L) 和 ((L-d)/L) 对应 semantic search 和 trajectory search 在 overall matching 中的贡献比例。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Expert Map Deduplication 流程：

Input: new_batch_context (B 个新 iterations 的 semantic + trajectory data)
        Expert Map Store with C historical maps (at capacity)
Output: Updated Expert Map Store (≤ C maps, 去重后)

# Step 1: 计算所有 pairwise redundancy scores
for x in range(B):           # 新 iterations
    for y in range(C):       # 历史 iterations
        score_sem[x,y] = cos_sim(sem_new[x], sem_old[y])
        score_traj[x,y] = cos_sim(traj_new[x], traj_old[y])
        RDY[x,y] = (d/L) × score_sem[x,y] + ((L-d)/L) × score_traj[x,y]

# Step 2: 对于每个新 iteration x，找到与之最冗余的历史 iteration y
for x in range(B):
    best_y = argmax(RDY[x, :])  # 最低 redundancy → 最不相似 → 最值得保留
    
# Step 3: 用新 iteration 替换与之最相似（冗余）的历史 iteration
# 注意：保留最少 redundancy 的历史 maps → 维持 pattern diversity

# 理论保证 (Minimum Sphere Covering):
#   保持 2LJ expert maps → ≥75% similarity lower bound (任意新 iteration 可找到 ≥75% 相似的 map)
#   保持 (1/2)LJ·ln(LJ) maps → ≥98% similarity lower bound
#   对于现代 MoE: L∈[8,128], J∈[24,96] → 需求 < 50K maps → < 200MB
```

理论分析：expert map deduplication 可被形式化为 Minimum Sphere Covering Problem（每个 expert map 是向量空间中的一个点，要去重后仍能覆盖尽可能多的 pattern 空间）。Dumer (2007) 和 Rankin (1947) 给出覆盖球面的 number-of-centers lower bound。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 中以 PyTorch pairwise cosine similarity 计算 redundancy scores。去重在每次 Expert Map Store 满容时触发（而非每次 iteration）。权重比例 (d/L) vs ((L-d)/L) 直接映射 semantic search 和 trajectory search 在整体 expert map matching 中的贡献。实验表明 C=1K 已足够（similarity scores 在 >1K 后 quickly diminishing returns），对应 ≤50MB 内存开销。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading
