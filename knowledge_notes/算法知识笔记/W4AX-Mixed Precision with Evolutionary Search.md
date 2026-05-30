## W4AX-Mixed Precision with Evolutionary Search

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
W4AX-mixed 是 Quamba2 提出的混合精度方案：在 SSM 模型的不同 block 中动态选择 W4A8（权重 4-bit+激活 8-bit）或 W4A16（权重 4-bit+激活 16-bit），以在 prefill 延迟和 MMLU 泛化性之间取得最佳平衡。Full W4A8 最大化了 prefill 加速（TTFT 140.78ms vs FP16 197.80ms）但 MMLU 5-shot 下降 5.8%（41.2% vs 47.0% FP16）。Full W4A16 保持较好泛化（45.3% MMLU）但 prefill 延迟增加（209.19ms）。W4AX-mixed 通过进化搜索（evolutionary search, population=40, generations=5）自动识别对量化敏感的 block 分配 W4A16，其余用 W4A8，最终在 MMLU 提升 2.9%（达 44.0%）同时仅增加 10% prefill 延迟（158.36ms）。手工设计的混合精度（前/后 N 层用 W4A16，命名为 HC-first/HC-last）效果远差于自动搜索。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Evolutionary search for mixed precision
population = [random_bitwidth_config() for _ in range(40)]   # 40 个随机配置
for gen in range(5):
    fitness = [eval_accuracy(cfg) for cfg in population]     # 评估准确率
    population = top_k(population, fitness, k=20)            # 保留 top 50%
    new_pop = []
    for _ in range(10):                                      # 10 crossover
        p1, p2 = random_pair(population)
        child = crossover(p1, p2)
        new_pop.append(child)
    for _ in range(10):                                      # 10 mutation
        p = random(population)
        mutant = mutate(p)
        new_pop.append(mutant)
    population = population + new_pop

best_config = population[argmax(fitness)]
# best_config: e.g. [W4A8, W4A8, W4A16, W4A8, ...] (per-block)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
搜索的搜索空间：每层独立选择 W4A8/W4A16，对 56 层模型有 $2^{56}$ 种配置。进化搜索在 5 代内收敛到 Pareto 前沿。适合需要同时优化延迟和准确率的场景（如云服务需要平衡 TTFT 和精度）。论文发现设计的混合精度模型在 Pareto 前沿上优于手工规则（HC-first/HC-last），说明 SSM block 的量化敏感度分布不均匀且无简单规律（并非"首/尾层更敏感"）。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Meta Tokens 是 Hymba 提出的 128 个可学习 embedding，预训练期间 prepend 到所有输入，与模型参数联合优化。推理时固定，K/V/SSM 状态离线预计算，等效 learned cache initialization。三重功能：(1) 缓解 attention sink——吸收 >50% 本该流向 BOS 的 attention；(2) 封装压缩世界知识——不同 domain 激活不同 meta tokens；(3) 作为初始 cache 调制后续 token attention 分布。消融：recall acc +3.75%（48.04%→51.79%），attention map entropy 整体下降。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Offline（仅一次）: 预计算 meta tokens 状态
K_meta, V_meta = W^K @ R, W^V @ R
h_meta = SSM_scan(R)   # 128步 scan 后的 state

# Online: 预计算状态 + 用户输入状态拼接
K = [K_meta; K_online], V = [V_meta; V_online]
h_init = h_meta         # SSM 从预计算 state 继续
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与 ViT register tokens (Darcet et al. 2023)、prefix tuning (Lester et al. 2021)、StreamingLLM 类似。推理开销极低（仅 cache 多 128 位置）。局限：任务特定 meta tokens 未探索。

涉及论文标题：
- Hymba: A Hybrid-head Architecture for Small Language Models
