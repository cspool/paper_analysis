## Gated DeltaNet

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Gated DeltaNet 是将 delta rule（增量学习规则）与门控机制结合的线性序列模型，是 MoM 论文使用的默认 memory update 机制。其 memory update rule 为：

$$M_t = a_t (I - k_t^T k_t) M_{t-1} + b_t k_t^T v_t$$

其中 a_t, b_t ∈ (0,1) 是数据依赖的标量门控参数，I 是 d×d 单位矩阵。与标准 DeltaNet 的区别在于增加了额外的输入门 a_t。

Delta Rule 的核心思想：(I - k_t^T k_t) M_{t-1} 部分从当前 memory 中"减去"与当前 key k_t 相关的旧信息（类似 online linear regression 的更新），b_t k_t^T v_t 部分将当前 token 的 kv 关联写入 memory。这种设计使模型能更精确地选择性更新 memory，而非像简单 Linear Attention 那样无差别累加。

从算法pipeline角度拆解术语。

**Gated DeltaNet 在 MoM 中的应用**：

```
# 单一 memory 版本的 Gated DeltaNet:
for t in 1..T:
  k_t = x_t @ W_k           # [d]
  v_t = x_t @ W_v           # [d]
  a_t = sigmoid(x_t @ W_a)  # 数据依赖的门控
  b_t = sigmoid(x_t @ W_b)

  # Delta rule update (核心):
  decay = I - k_t^T k_t     # [d, d]
  M_t = a_t * decay @ M_{t-1}  # 每行减 k_t @ (k_t^T @ M_{t-1})
  M_t = M_t + b_t * k_t^T v_t

  q_t = x_t @ W_q
  o_t = q_t @ M_t

# MoM 多 memory 版本:
for each activated memory m:
  k_t^m = x_t @ W_k^m       # memory-specific
  v_t^m = x_t @ W_v^m
  a_t^m = sigmoid(x_t @ W_a^m)
  b_t^m = sigmoid(x_t @ W_b^m)
  M_t^m = a_t^m (I - k_t^{m,T} k_t^m) M_{t-1}^m + b_t^m k_t^{m,T} v_t^m
```

术语一般如何实现？如何使用？

Gated DeltaNet 通过 Triton chunk-wise parallel scan kernel 实现。由于 (I - k_t^T k_t) 的矩阵运算涉及 d×d 的外积和矩阵乘法，实现上使用 chunk 内并行处理来隐藏延迟。在 MoM 中每个 memory 独立执行 Gated DeltaNet update，通过 varlen kernel 处理不同 memory 各自的 token 子序列。代码开源：https://github.com/OpenSparseLLMs/MoM。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories
- Gated Delta Networks: Improving Mamba2 with Delta Rule

---
