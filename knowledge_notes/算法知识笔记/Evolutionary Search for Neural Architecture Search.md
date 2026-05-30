## Evolutionary Search for Neural Architecture Search

术语解释
Evolutionary Search（演化搜索）是 NAS 中基于生物进化原理的搜索算法，通过种群初始化、选择、突变、交叉等操作迭代优化架构种群，以找到满足约束条件的最优架构。AutoMoE 采用演化搜索在异构 MoE 搜索空间中寻找 Pareto 最优架构。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
演化搜索的核心流程（Algorithm 1 in AutoMoE/HAT）：
1. **初始化**：随机采样 num_population 个架构
2. **评估**：对每个架构评估 validation loss（通过 Supernet 快速估计）和 latency（在目标设备上实测）
3. **选择**：选出 top num_parents 个架构作为父代
4. **突变**：对种群中随机选择的架构以 mutate_prob 概率随机修改一个搜索维度，生成 num_mutations 个子代（需满足 latency constraint）
5. **交叉**：随机选择两个架构交换部分维度，生成 num_crossover 个子代（需满足 latency constraint）
6. **新一代**：population = parents ∪ mutations ∪ crossovers
7. 重复 2-6 直到迭代结束，返回最优架构

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# AutoMoE 演化搜索（Algorithm 1）
popu = random_sample(search_space, n=125)  # 125 个随机架构

for iter in range(15):
    # Step 1: 评估所有架构
    for arch in popu:
        arch.val_loss = supernet.evaluate(arch)    # Supernet 快速估计
        arch.latency = measure_latency(arch, device=CPU, passes=100)  # partial gold
    
    # Step 2: 选择父代
    parents = top_k(popu, key=val_loss, k=25)
    
    # Step 3: Mutation (50 offsprings)
    mutations = []
    for _ in range(50):
        parent = random_choice(popu)
        child = mutate(parent, prob=0.3)  # 随机修改 1 个维度
        if child.latency <= latency_constraint:
            mutations.append(child)
    
    # Step 4: Crossover (50 offsprings)
    crossovers = []
    for _ in range(50):
        p1, p2 = random_choice(popu, 2)
        child = crossover(p1, p2)  # 交换部分维度
        if child.latency <= latency_constraint:
            crossovers.append(child)
    
    # Step 5: 新一代
    popu = parents + mutations + crossovers  # 125 个

# 返回最优
return top_1(popu, key=val_loss)
```

Mutation 操作实例：随机选择以下维度之一进行修改：
- 某层的 expert 数量：random(1, M)
- 某 expert 的 FFN 尺寸：random_choice([1024, 2048, 3072])
- Decoder 层数：random_choice([1-6])
- Attention heads：random_choice([4, 8])

术语一般如何实现？如何使用？
- AutoMoE 搜索参数：population=125, parents=25, mutation=50 (prob=0.3), crossover=50, 迭代=15
- 搜索耗时：~224 GPU-hours（含 Supernet 训练），远低于 Evolved Transformer 的 2,192,000 GPU-hours
- 搜索空间大小：M^L × NML 种可能配置（极大的搜索空间），演化搜索通过 guided exploration 而非穷举
- 关键优化：Partially gold latency（100 passes vs 300 passes 的 gold latency）加速搜索过程中的评估
- HAT (Wang et al., 2020) 首次将演化搜索用于 NLP NAS，AutoMoE 将其扩展到 MoE 空间

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

---
