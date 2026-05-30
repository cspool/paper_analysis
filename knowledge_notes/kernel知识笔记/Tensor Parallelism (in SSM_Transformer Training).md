## Tensor Parallelism (in SSM/Transformer Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Tensor Parallelism (TP) 由Shoeybi et al.(2019)在Megatron-LM中提出，是一种模型并行策略：将单层内的权重矩阵按列或行切分到多个GPU上，每层计算后通过all-reduce同步部分结果。对于Transformer的self-attention层，TP沿attention heads维度切分Q/K/V权重；对于MLP层，沿列切分第一个线性层，沿行切分第二个线性层。每层仅需1次all-reduce（在MLP output或attention output之后）。论文中所有模型使用TP size=4进行训练。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mamba与Mamba-2的TP通信差异：
```
// Transformer TP (1 all-reduce per layer):
// Attention: Q,K,V按head切分 → 各GPU独立计算attention → all-reduce(output)
// MLP: column-parallel Linear1 → GELU → row-parallel Linear2 → all-reduce

// Mamba TP (2 all-reduces per layer):
// - 第1次all-reduce: 在input projection (Linear_proj) 之后
// - 第2次all-reduce: 在SSM scan & gating之后, output projection之前
// 原因: Mamba的SSM scan需要完整的hidden dim进行计算

// Mamba-2 TP (1 all-reduce per layer):
// - 利用SSD的多头结构, 类似attention沿head维度切分
// - 仅需在output projection后1次all-reduce
// 约束: 必须使用GroupNorm(而非LayerNorm)作为内部归一化
// GroupNorm: 沿hidden dim分组归一化, 各组独立计算
// 要求group_size > 256以保证统计量精度
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在Megatron-LM中通过--tensor-model-parallel-size参数设置。论文训练使用TP=4, DP=256, 共1024 H100 GPUs。TP的核心trade-off：减少单GPU显存需求，但引入通信开销（all-reduce延迟）。Mamba-2通过SSD的多头结构将TP通信降至1次all-reduce，与Transformer持平，而Mamba需要2次，增加了通信开销。

涉及论文标题：
- An_Empirical_Study_of_Mamba-based_Language_Models

---
