## Time-Invariant vs Time-Variant Inference Costs（时间无关 vs 时间相关的推理成本）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Time-Invariant vs Time-Variant Inference Costs 是本文提出的 GQA Transformer 推理成本分解框架，将每 token 的推理 FLOPs 和 Memory 按是否随上下文长度 T 增长分为两类：

- **Time-Invariant Costs（时间无关成本）**：与 T 无关的固定开销。FLOPs 方面：模型参数相关的线性投影（QKVO projection + FFN），C_const = 2N（N 为参数量）。Memory 方面：存储模型参数本身，M_const = N。
- **Time-Variant Costs（时间相关成本）**：随 T 线性增长的开销。FLOPs 方面：attention softmax 计算 C_att(T) = 4TL·d_h·n_h。Memory 方面：KV cache N_kv(T) = 2TL·d_h·n_kv。

核心洞察：T 很大时（如 128K），time-variant costs 主导。例如 1.2B 模型在 128K 下，~90% memory 被 KV cache 占用，仅有 ~10% 用于模型参数。因此长上下文下应通过减少 n_h 和 n_kv（降低 time-variant cost）而非减少 N（降低 time-invariant cost）来优化。

从算法pipeline角度拆解术语：

**成本分解公式**：
```
C_infer(T) = 2N               + 4TL·d_h·n_h
           = 时间无关 FLOPs     + 时间相关 FLOPs (attention softmax)

M_infer(T) = N                + 2TL·d_h·n_kv
           = 时间无关 Memory    + 时间相关 Memory (KV cache)
```

**长上下文下的资源分配优化**：
```
若 T=128K, d_h=64, L=36:
  C_var = 4 × 128K × 36 × 64 × n_h = 1.18G × n_h FLOPs/token
  C_const = 2N

若 N=1.2B, C_const = 2.4G
  当 n_h=32: C_var = 37.8G → attention 占 94% FLOPs
  当 n_h=8:  C_var = 9.4G  → attention 占 80% FLOPs
  → 减少 n_h 大幅节省 FLOPs，而适度增加 N（补偿 loss）仅小幅增加 C_const

若 n_kv=8:  M_var = 2 × 128K × 36 × 64 × 8 = 4.7B floats = 9.4GB (BF16)
若 n_kv=1:  M_var = 2 × 128K × 36 × 64 × 1 = 0.59B floats = 1.18GB (BF16)
  → 减少 n_kv 大幅节省 Memory
```

术语一般如何实现？如何使用？

该框架用于指导 cost-optimal GQA 配置搜索：通过解耦 n_h 与 d（Change 1）自由控制 time-variant FLOPs，通过联合优化 N 与 (n_h, n_kv)（Change 2）平衡 time-invariant 与 time-variant 资源分配。通过三步搜索找到给定 (T, L*) 下硬件感知成本 Z 最小的配置。实验验证该框架在 T=128K 时可节省 >50% memory 和 FLOPs vs Llama-3 GQA，无 loss 损失。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs

---
