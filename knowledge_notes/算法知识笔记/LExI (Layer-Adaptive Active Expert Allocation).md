## LExI (Layer-Adaptive Active Expert Allocation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LExI 是一种 data-free 的 post-training MoE 推理优化技术，核心思想是为预训练 MoE 模型的每一层静态分配不同的 active expert 数量（top-k_j），替代传统所有层统一的 top-k。LExI 通过两阶段 pipeline 实现：(1) Monte Carlo 敏感性分析：使用随机 Gaussian 输入计算每层在不同 top-k 下的 Frobenius 范数输出扰动；(2) 进化搜索：以扰动损失为 proxy，在总 active expert budget B 约束下搜索全局最优的逐层 k_j 分配。LExI 不删除任何 expert 参数，仅通过减少低敏感层的 active expert 数量来减少 FFN 计算量、inter-GPU 通信和 memory bandwidth 使用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# LExI 两阶段 Pipeline

# === Stage 1: Per-Layer Sensitivity Profiling ===
输入: pretrained MoE model M, target top-k list T = [1, 2, ..., k_base]
输出: D[layer][k] = average Frobenius norm perturbation

for layer in range(L):
    for k in T:
        perturbations = []
        for iter in range(N_iter):
            X = randn(B, L_seq, H)  # ~ N(0,1)
            set_topk(M, k_base)
            Y_base = moe_forward(M, X)   # 只计算当前层
            set_topk(M, k)
            Y_k = moe_forward(M, X)
            Δ = ||Y_k - Y_base||_F       # Frobenius norm
            perturbations.append(Δ)
        D[layer][k] = mean(perturbations)

# === Stage 2: Evolutionary Search ===
输入: D, budget B, k_min, k_max
输出: k* = (k_1, ..., k_L)  # 每层最优 top-k

population = rand_feasible(N_pop, L, B)  # 满足 Σk_j = B
for gen in range(G_max):
    p1, p2 = tournament_select(population)  # min Σ D_j(k_j)
    offspring = uniform_crossover(p1, p2)   # 每层随机选父代
    offspring = mutate(offspring)            # ±1, ΣΔ = 0
    offspring = project(offspring, B)        # 保证 budget 约束
    population.append(offspring)
k* = argmin_{k in population} Σ_j D_j(k_j)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LExI 在 vLLM 推理框架上使用：加载预训练模型 → 运行一次 LExI profiling（Stage 1）+ search（Stage 2）→ 得到 k* → `set_topk(layer_j, k_j)` 修改每层路由参数 → 正常 vLLM 推理。LExI 是 data-free 的（仅用随机噪声 + 模型权重），不需要任何 calibration 数据集或微调。Budget B 是可控参数：B 越小吞吐越高但精度越低，B 越大越接近 baseline 精度。LExI 不减少模型显存占用，但可与 expert pruning 方法结合以实现 memory + computation 的联合优化。限制：(1) 不减少 memory footprint；(2) 对 k_base=1 的模型（如 Llama-4）不适用。

涉及论文标题：
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference
