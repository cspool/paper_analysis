## Evolutionary Strategy for Model Compression (进化策略模型压缩)

术语是什么？
进化策略（Evolutionary Strategy, ES）是一种基于种群的随机优化算法，不需要梯度信息。在模型压缩中，ES 通过维护一组候选压缩方案（个体），每代通过选择（Selection）、交叉（Crossover）和变异（Mutation）操作迭代优化，在无梯度约束下搜索最优的剪枝/合并配置。

在 EEP 中，ES 被用于搜索最优的 Expert Pruning 和 Expert Merging 配置：
- **个体表示**：每个个体是在所有 MoE 层上的 Router Mapping 矩阵集合 W={W^l_RM, W^l_EM}_{l=1..L}（或 Pruning Phase 中 W_RM = W_EM）
- **Fitness Function**：F(W) = 模型在训练子集上的下游任务准确率（generation-based evaluation），仅需做推理即可评估
- **选择（Selection）**：按 fitness 排名，前 M_CP 个个体进入候选父代集 CP（elitism selection）
- **交叉（Crossover）**：随机从 CP 中采样两个父代 W_f 和 W_m，随机组合两者的 merging coefficients（沿 retained expert 维度交叉），或以一定概率直接选择单一父代的全部矩阵
- **变异（Mutation）**：Pruning Phase 随机替换 pruned expert；Merging Phase 对 merging coefficients 逐元素加入 Gaussian noise N(0, σ²)
- **世代更替**：每代将变异后代 NG 加入种群 P ← P ∪ NG，不淘汰旧个体

从算法pipeline角度拆解术语。
```
# EEP 进化搜索伪代码
Input: Θ = all expert weights, F = evaluator, Epochs, M_CP, Iters
Output: optimal W*

1: P ← {random one-hot initialization W_init with F(W_init)}
2: for phase in {Pruning, Merging}:
3:   for t = 1..Iters:
4:     NG ← ∅
5:     for i = 1..Epochs:
6:       CP ← {W_i | F(W_i·Θ) ranks top min(M_CP, |P|)}
7:       W_f, W_m ← RandomSample(CP)           # 选择
8:       W_new ← Mutate(Crossover(W_f, W_m))    # 交叉+变异
9:       NG ← NG ∪ {(W_new, F(W_new))}          # 评估
10:    P ← P ∪ NG                                # 更新种群
11: return argmin_W F(W)
```

为什么需要？/解决什么痛点？
- LLM 上的梯度计算需要大量 GPU 显存（至少 2× 模型大小），基于梯度的 fine-tuning 对大多数用户不可行
- ES 只需推理即可评估 fitness，可在推理设备上运行，无需反向传播的额外显存
- ES 适用于离散搜索空间（expert 选择和 one-hot 约束），梯度方法难以直接处理离散决策
- ES 天然支持并行评估（种群中所有个体可并行推理），可利用多 GPU/多节点

术语一般如何实现？如何使用？
- EEP 的 ES 超参数：Pruning Phase 40 iterations, Merging Phase 160 iterations, population size 取决于评估预算
- 为减少搜索参数，expert weights 按深度分组（4 groups 或 32 groups），组内共享 merging coefficients
- 适用场景：任何需要搜索离散/连续混合优化空间且梯度不可得或代价过高的模型压缩任务
- 局限性：搜索过程需要大量推理调用（每代每个个体一次完整模型推理），搜索成本随种群规模线性增长
- 相关方法：Model Soup (uniform averaging), Evolutionary Model Merging (Akiba et al. 2024)
- 代码：https://github.com/imagination-research/EEP

涉及论文标题：
- Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs
