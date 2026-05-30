## Effective Parameter Multiplier (EPM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Effective Parameter Multiplier (EPM) 是一种用于衡量模型改进效果的方法论指标。给定一个基线的 scaling law f(N)（将参数数量 N 映射到任务精度），对于一个经过某种优化（如 LatentMoE、量化、剪枝）的模型，EPM 定义为达到相同精度所需的 baseline 参数量与优化模型物理参数量的比值：

$$N_{eff} = f^{-1}(S_{treat})$$

$$\lambda = \frac{N_{eff}}{N_{treat}}$$

其中 S_treat 是优化模型在目标 benchmark 上的得分，f 是 baseline 模型的 scaling law（将参数映射到精度），f^{-1} 是其反函数。λ > 1 表示优化模型相当于拥有更多有效参数。

EPM 的核心作用是建立 iso-accuracy baseline，用于公平比较不同架构的 inference efficiency。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
EPM Construction Pipeline (LatentMoE, Section 4.3.1):

1. Establish Baseline Scaling Law:
   Use Qwen-3-Dense model family (0.6B, 1.7B, 4B, 8B, 14B, 32B)
   Measure MMLU accuracy for each size
   Fit: f(N) = a·log(N) + b  (log-linear fit)

2. Measure Treated Model:
   Kimi-K2-1T-LatentMoE MMLU score = S_treat

3. Compute Effective Parameters:
   N_eff = f^{-1}(S_treat) = exp((S_treat - b) / a)

4. Compute EPM:
   λ = N_eff / N_treat = exp((S_treat - b)/a) / 1.0T
   For Kimi-K2-1T-LatentMoE: λ ≈ 1.35

5. Construct Iso-Accuracy Baseline:
   N_iso = λ · N_treat = 1.35T
   → Kimi-K2-1.35T: 61 layers → 80 layers (d=4096, standard MoE)

6. Compare Inference Efficiency:
   Kimi-K2-1T-LatentMoE (1T physical params) 
   vs Kimi-K2-1.35T (1.35T, iso-accuracy)
   Result: 1.24×-3.46× speedup for LatentMoE
```

EPM 基于 Frantar et al. (2025) 的 "Effective Parameter Count" 框架（Compression Scaling Laws: Unifying Sparsity and Quantization）。原始框架用于量化/稀疏模型的压缩效率评估，LatentMoE 将其推广到架构改进的效率评估。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实践使用步骤：
1. 选择一组同架构、不同规模的 baseline 模型（如 Dense 或 Standard MoE）
2. 在目标 benchmark 上测量各 scale 的精度
3. 拟合 scaling law f(N)（通常 log-linear）
4. 测量优化模型在相同 benchmark 上的精度
5. 反推 N_eff 并计算 λ

局限性：
- 依赖于 scaling law 的拟合质量（需要足够多的 baseline scale points）
- 外推超出 baseline 训练范围可能不准确
- λ 是 benchmark-dependent（不同 benchmark 可能得出不同的 λ）
- 假设 baseline 和 treated model 在 scaling behavior 上可比

涉及论文标题：
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
- Compression Scaling Laws: Unifying Sparsity and Quantization (Frantar et al., 2025)
