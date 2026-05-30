## Frobenius Norm Perturbation Profiling (Monte Carlo Sensitivity Analysis)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Frobenius Norm Perturbation Profiling 是 LExI 提出的用于评估 MoE 每层对 top-k 变化敏感度的 data-free 方法。核心原理：对于某一层，用不同的 top-k 值计算同一批随机 Gaussian 输入的输出，用 Frobenius 范数 ||Y_k - Y_base||_F 量化输出偏差。偏差越大，说明该层对该 k 值越敏感（减少 expert 会导致输出变化大）。重复 N_iter 次取平均以获得统计稳健的估计。整个过程仅需模型权重，无需真实数据。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Frobenius Norm Perturbation Profiling
# 对每个 MoE layer 独立执行

# 输入生成
X = randn(32, 128, 4096)  # [Batch=32, Seq=128, Hidden=4096]
                           # 标准正态分布 N(0,1)

# Baseline 输出 (pretrained k_base = 2)
set_topk(this_layer, 2)
Y_base = moe_forward(X)    # [32, 128, 4096]

# 扰动输出 (k=1)
set_topk(this_layer, 1)
Y_k1 = moe_forward(X)

# Frobenius 范数计算
# ||A||_F = sqrt(Σ_{i,j} A_{i,j}²)
Δ_k1 = ||Y_k1 - Y_base||_F
      = sqrt(sum((Y_k1[b,s,h] - Y_base[b,s,h])² 
                 for b in 0..31, s in 0..127, h in 0..4095))
# Δ_k1 是一个标量，反映 top-1 vs top-2 的输出总偏差

# 重复 N_iter > 1M 次，取平均获得稳健估计
D[layer][1] = mean(Δ_k1 over N_iter iterations)
```

LExI 实验显示：Mixtral-8x7B 浅层在 k=1 vs k=2 时扰动小（低敏感），深层扰动大（高敏感）；Qwen1.5-MoE 呈现相反模式，浅层更敏感；OLMoE 和 DeepSeekV2 呈钟形曲线（中间层最稳定）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 PyTorch 中实现：`torch.norm(Y_k - Y_base, p='fro')`。选择 Frobenius 范数而非其他度量（如 L1/L∞/cosine similarity）的原因：Frobenius 范数在欧几里得空间中精确捕捉高维输出的 magnitude shift，Monte Carlo 采样确保对 diverse input 的泛化性。计算开销：仅需前向传播（无反向传播），且每层独立执行可并行化。对于 Mixtral-8x7B（32 MoE layers × ~8 k-values），profiling 在单 H100 上 <30 分钟完成。

涉及论文标题：
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference
