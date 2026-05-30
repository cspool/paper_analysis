## Evolutionary Search for MoE Top-k Allocation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Evolutionary Search for Top-k Allocation 是 LExI 提出的用于在 MoE 模型中搜索逐层最优 active expert 数量的优化算法。给定每层在不同 top-k 下的 Frobenius 范数扰动损失 D_j(k) 作为 proxy，目标是在总 active expert budget B 约束下找到最小化总敏感损失的分配方案 k* = (k_1, ..., k_L)。由于搜索空间为 {k_min, ..., k_max}^L，穷举不可行，进化搜索在 G_max 代内通过选择、交叉、变异迭代逼近全局最优。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

LExI 的进化搜索是一种离线优化方法，作用于模型推理前的配置搜索阶段：

```
# LExI Stage 2: Evolutionary Search Flow

1. 输入Proxy数据 (编译期静态信息)
   D[layer][k] = Frobenius perturbation loss  // 来自Stage 1 profiling
   B = total active expert budget
   k_min, k_max per layer

2. 种群初始化 (Population Initialization)
   population = []
   for i in range(N_pop):
       k_i = 随机生成满足 Σk_j = B 的可行解

3. 迭代进化 (Generation Loop)
   for generation in range(G_max):
       # 评估适应度
       for each k in population:
           φ(k) = Σ_j D[j][k_j]  // 总敏感损失
       
       # 锦标赛选择 (Tournament Selection)
       p1, p2 = select_parents(population)  // 选 φ 最小的
       
       # 均匀交叉 (Uniform Crossover)
       offspring = []
       for layer_j in range(L):
           α_j ~ Bernoulli(0.5)
           offspring[j] = α_j * p1[j] + (1-α_j) * p2[j]
       
       # 变异 (Mutation) - 保持 budget 不变
       Δ_j ∈ {-1, 0, +1}, ΣΔ_j = 0
       offspring[j] += Δ_j
       
       # 投影到可行域
       offspring = project(offspring, B, k_min, k_max)
       
       # 更新种群
       population.append(offspring)

4. 输出最优解
   k* = argmin_{k in population} φ(k)
```

与梯度方法相比，进化搜索不需要加载模型或计算梯度，仅需查表 D[layer][k]，搜索 O(N_pop × G_max) 次 fitness evaluation。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 LExI 中，进化搜索使用 Python 实现（numpy/random），N_pop ≈ 50-100，G_max ≈ 200-500。总 active expert budget B 是用户可控参数（如 B=100 对应平均每层约 3 个 active expert），$k_{min} = 1$，$k_{max} = k_{base}$（预训练 top-k）。变异操作确保 ΣΔ = 0（某层 +1 必须有另一层 -1 补偿）。投影步骤将超出 [k_min, k_max] 的值 clip 到边界并调整其他层以维持 budget B。进化搜索是 offline 执行（一次 profiling 后可为任意 B 搜索），不增加推理 latency。

涉及论文标题：
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference
