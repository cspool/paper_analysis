## Layer-Adaptive KV Selection / Budget Allocation（层自适应KV选择/预算分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Layer-Adaptive KV Selection 是 StreamKV 提出的统一 KV 选择模块，用于将 KV 压缩和检索统一为同一框架。它将选择问题形式化为：在给定总预算 $N$（需保留的 KV blocks 总数）下，跨 $L$ 个 transformer 层自适应分配每层的选择数量 $K_l$，使得保留的 KV blocks 总量 $= N$。与 uniform allocation（每层选择相同数量）不同，自适应策略根据每层相似度分布的集中程度分配预算——信息更集中的层获得更多预算，在总预算不变的情况下最大化保留的信息量。核心机制：(1) 每层计算候选 representative key vectors 与 selection criterion 的 cosine similarity；(2) Softmax 归一化得到概率分布；(3) 通过 binary search 确定全局 cumulative score threshold $p$，使跨层累积达到 $N$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

层自适应选择的三步算法流程：

```
# 输入: {R_l, c^l}_{l=1}^L (每层候选 + 选择标准), 总预算 N

# Step 1: Cosine Similarity per layer
for each layer l:
    for each candidate j in R_l:
        Sim_l(j) = cos_sim(r_j^l, c^l)

# Step 2: Softmax Normalization + Descending Sort
for each layer l:
    ~Sim_l(j) = softmax(Sim_l)(j)  # Eq.(6)
    priority_l = sort_descending(~Sim_l)  # s_l(j) = index of j-th largest

# Step 3: Binary Search for Global Threshold p (Algorithm 1)
p_1, p_2 = 0, 1
while p_2 - p_1 > ε:
    p = (p_1 + p_2) / 2
    for each layer l:
        K_l^p = min{k | Σ_{j=1}^k ~Sim_l(s_l(j)) ≥ p}  # Eq.(7)
    if Σ_l K_l^p == N:
        return p, {K_l^p}
    elif Σ_l K_l^p < N:
        p_1 = p  # p 太小，提高阈值以选更多
    else:
        p_2 = p  # p 太大，降低阈值

# 输出: {I_l}_{l=1}^L, 其中 I_l = top-K_l 候选的索引集合
```

Binary search 的正确性：cumulative sum function $f(p) = \sum_l K_l^p$ 关于 $p$ 单调递减（p 越大，K_l^p 越小），因此 binary search 可找到使 f(p)=N 的 p。

术语一般如何实现？如何使用？

实现方式：统一模块同时用于压缩（criterion = guidance prompt vector）和检索（criterion = question vector）。复杂度：binary search O(log(1/ε)) ≈ 常数次迭代，每次计算 O(L log|R|)（排序后取前缀和）。适用场景：(1) 任何需要跨层预算分配的 KV 选择问题；(2) 可推广到其他需要自适应保留重要 token/feature 的任务（如 token pruning、expert selection in MoE）。消融实验（Table 4）验证：Ada.+Ada.（压缩和检索均自适应）优于 Uni.+Uni.（全 uniform），如 50% 压缩率 59.07% vs 58.12%。

涉及论文标题：
- StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression
