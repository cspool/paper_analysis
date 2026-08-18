## Neighbor Shuffling（邻居洗牌，BFS 随机化启发式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Neighbor Shuffling 是 SATIC 的子问题形成类启发式：BFS 遍历 VIG 时，在每个 BFS 层随机置换当前节点的邻接表顺序，避免"同层多个等资格候选变量时默认选择顺序引入偏差"。默认 SATIC 从随机根做确定性 BFS，相邻变量因强相关被同组；但同层多候选时固定顺序会造成系统性偏差（部分结构总被优先选入/排除）。洗牌在不同迭代间改变选择，引入受控多样性。单独使用（无 Limited Neighbors）可能有害——系统已经高度随机（B0 vs B3 中洗牌反而变差）；与 Limited Neighbors 配合时效果最好（C1 vs C2 提升）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 SATIC++ 编译框架的迭代循环内（每迭代执行）：
```
# 每迭代一次 BFS 遍历
root ← randint(1, max_var)
queue ← [root]; visited ← {}
while queue:
    u ← pop(queue)
    adj ← VIG.adj[u]
    if shuffle_enabled: adj ← random.shuffle(adj)   # Neighbor Shuffling
    for v in adj:
        if v not in visited: visited.add(v); push(queue, v)
# 输出：不同迭代生成不同的变量序 var_list → 不同子问题 → 探索多样结构
```
评估（Batch-4-100-1000，10K 迭代）：Neighbor Shuffling 把 solved 从 67（基础 SATIC）提到 85——子问题选择的结构多样性显著提升；但受限于度数 80 vs 子问题 20 变量的差距，单纯洗牌仍近似随机选点，需 Limited Neighbors 进一步约束。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：在 BFS 出队扩展时对邻接表做随机置换（Python random.shuffle 级别）；复杂度 O(V+E)（洗牌摊销）。使用：作为子问题多样性机制与其他 tricks 组合（评估显示与 Limited Neighbors 协同最佳）；避免单独在高度随机系统上使用（可能劣化）。

涉及论文标题：
- SATIC: An Optimizing Ising Compiler for SAT(isfiability)
