## Regularized Evolutionary Search for Transformer Architecture (正则化演化架构搜索)

术语解释
Regularized Evolutionary Search 是 Brainformers (Zhou et al., ICML 2023) 提出的自动化 Transformer 架构搜索方法，在固定训练时间预算和 inference step time 约束下，通过演化算法联合搜索层类型序列、层宽度、gating 机制、routing 策略和激活函数，发现非均匀的 MoE Transformer block 架构。

术语是什么？
与标准 NAS（Neural Architecture Search）不同，Regularized Evolutionary Search 的关键特征：
1. **Block-wise 搜索空间**：不搜索单个算子，而是搜索整个 block 的 sub-layer 序列（F_attn, F_moe, F_ffn）以及各层的维度配置
2. **Training time constrained**：每个搜索 trial 固定 wall clock time（而非固定 training steps），更快收敛的模型自动获得更多 training steps
3. **Early stopping regularization**：在 25% max training steps 时检查，违反 perplexity 或 inference time 约束的模型提前淘汰（R=-1）
4. **Proxy training + scale-up evaluation**：小规模 proxy model（100M, block 堆叠 3 次）搜索 block → ScaleModelDim (2x/4x) + StackNTimes 扩展到目标规模（1B/8B）

从算法pipeline角度拆解术语。
```
# Regularized Evolutionary Search Algorithm
population_size = p

for generation t = 1 to T0:
    for each block_arch B^(i) in SamplePopulation(B, p):
        # Proxy model: stack block 3 times → ~12 sub-layers, 100M scale
        G^(i) = StackThreeTimes(B^(i))
        
        if EarlyStopping(G^(i)):     # at 25% T_max
            R^(i) = -1               # prune: poor perplexity or slow step time
        else:
            A_i, T_i = Train(G^(i), fixed_wall_clock_time)
            R^(i) = f(A_i, T_i)      # reward = accuracy + step time

# Top-k architectures evaluated at target scales
G_topk = TopK({G^(i), R^(i)})
for G^(i) in G_topk:
    G^(i) = ScaleModelDim(G^(i))    # 2x/4x dim scaling
    G^(i) = StackNTimes(G^(i))      # N = target_activated / activated_per_block
    A_i, T_i = Train(G^(i))         # full-scale evaluation
```

术语一般如何实现？如何使用？
- 基于 GLaM 框架（Google 内部），512 TPU V4 运行 1 周完成搜索
- 搜索在 500 trials 内即能发现显著优于 GLaM baseline 的架构
- 相比 Evolved Transformer（2,192,000 GPU-hours），计算开销大幅降低
- 可用于优化 MoE 模型（gating 策略、expert 数量、routing 粒度、层类型分布）
- 局限：大模型规模搜索仍昂贵；block structure 在不同尺度上的可迁移性未被充分验证

涉及论文标题：
- Brainformers Trading Simplicity for Efficiency

---
