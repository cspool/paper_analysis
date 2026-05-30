## Entropy-Guided Quantization Optimization (Maximum Output Entropy)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Entropy-Guided Quantization Optimization（熵引导量化优化）是 Squat (EdgeQAT) 论文提出的量化感知训练技术。其理论基础来自 Messerschmitt (1971)：对高斯分布而言，最大化量化输出熵（Maximum Output Entropy, MOE）的量化器与最小化平均误差（Minimum Average Error, MAE）的量化器近似等价（仅差乘法常数）。因此，在QAT中最大化量化后query/key的信息熵等价于最小化量化误差。具体实现为损失函数 `L_E = -log(Σ_l Σ_h log(1 + σ_q²·σ_k²))`，其中σ_q²和σ_k²为各层各头query和key的方差。因为高斯分布的熵 H(q) = ½log(2πeσ_q²) ∝ σ_q²，通过增大方差来增大熵，同时log(1+σ_q²·σ_k²)的对数缩放防止梯度爆炸。该术语也可广义理解为基于信息论准则（如熵最大化）指导量化器设计的优化方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Squat中熵损失计算的流程（以LLaMA-58M单步训练为例）：
```
def entropy_loss(all_queries, all_keys, L_layers, H_heads):
    total_log_entropy = 0.0
    for l in range(L_layers):
        for h in range(H_heads):
            q = all_queries[l][h]  # [B, N, d_h]
            k = all_keys[l][h]
            var_q = q.var()         # σ_q²
            var_k = k.var()         # σ_k²
            total_log_entropy += log(1 + var_q * var_k)
    L_E = -log(total_log_entropy)
    return L_E
```
理论基础：H(q) = -Σ p(q_i)·log p(q_i) = ½log(2πeσ_q²)，最大化H(q) ∝ 最大化σ_q²。MOE ≈ MAE准则下最大化熵 = 最小化量化误差。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在QAT训练循环中，熵损失作为额外正则项：`L_total = L_distill + r_E·L_E + r_D·L_D`。Squat中r_E=0.5。实现注意事项：(1)方差在batch维度计算；(2)内层log(1+x)防止数值问题；(3)外层log确保与CE loss尺度匹配。该技术依赖query/key近似高斯的观测（SmoothQuant/Agile-Quant也发现类似特性），对非高斯分布等价性不严格成立但仍可作为正则化。

涉及论文标题：
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

---
