## Contribution-Aware Expert Pruning (CAEP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Contribution-Aware Expert Pruning (CAEP) 是一种基于专家协作模式的 MoE 模型剪枝算法。与传统的独立专家剪枝方法（SEER-MoE 基于路由分数截断、GEM 基于输出影响力排序）不同，CAEP 利用 HSDL 提取的字典矩阵 D 和稀疏编码 R 计算每个专家的综合贡献分数，考虑专家在协作模式中的结构角色，而非仅按个体指标排序。核心理念：删除专家时应考虑其所属的协作模式是否完整——如果一个专家是关键协作模式中的必要成员，即使其个体路由分数不高也应保留。

算法流程：(1) 从 R 计算模式级贡献 $R_{\text{sum}} = \sum_{j} R_{:,j}$，结合 D 计算专家的总贡献分数 $e$，降序排序；(2) 以 $k_1$-分位数为阈值生成初始 mask；(3) 迭代：找出贡献最小的模式并移除，重算贡献分数并更新 mask，直到保留专家数达到目标 $(1-k_2) \cdot N_e$。实验效果：在 DeepSeek-MoE-16B 上，CAEP 剪枝 25% 专家后平均 accuracy 为 0.612，优于 SEER-MoE (0.5872) 和 GEM (0.5870)，在 OBQA 上从 0.420 提升至 0.473。

从算法pipeline角度拆解术语：

```
输入: D in R^{N_e x N_p}, R in R^{N_p x N_s}, k_1, k_2
输出: expert mask m in {0,1}^{N_e}

# Step 1: 计算贡献分数
R_sum = sum_{j=1}^{N_s} R[:,j]         # 每个 pattern 的样本级总激活
D_sum = D @ R_sum^T                     # 专家-模式贡献矩阵
e = sum_{i=1}^{N_p} D_sum[:,i]          # 每个专家的总贡献分数

# Step 2: 初始阈值 mask
e_sorted = sort_descending(e)
threshold = e_sorted[ceil(k_1 * N_e)]
m = (e >= threshold)

# Step 3: 迭代剪枝
while count_ones(m) > (1 - k_2) * N_e:
    i* = argmin_i R_sum[i]              # 最少使用的协作模式
    D = delete_column(D, i*)
    R = delete_row(R, i*)
    recompute R_sum, D_sum, e
    m = (e > threshold)

return m
```

术语一般如何实现？如何使用？

基于 PyTorch/NumPy 实现。剪枝后参数量（DeepSeek-MoE-16B）：仅剪枝 normal experts，保留 shared experts，新参数量 = 16.4 - 14.7 x k_2 B（式 10）。适用场景：(1) MoE 部署压缩；(2) 领域特化剪枝——针对特定领域保留相关协作模式；(3) 替代独立评估的剪枝方法。

涉及论文标题：
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models
