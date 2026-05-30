## Time-Uniform Micro-Batching / Token Buffer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Time-Uniform Micro-Batching（时间均匀微批次划分）是 FOLDMOE 解决 causal attention 微批次计算不均衡问题的策略。传统的 token-uniform 切分（每个 micro-batch 包含相同数量的 token）在 causal attention 场景下导致后序微批次计算量更大（因为它们需要 attend 到更长的前缀 KV cache），造成 pipeline 中 attention 阶段的计算时间不均衡。

FOLDMOE 通过两部分解决：
1. **Token Buffer**：在 attention 层和 MoE 层之间插入 FIFO 缓冲区，解耦二者的微批次划分——attention 可以按时间均匀切片（不按 token 数均匀），MoE 层仍按 token 数量均匀切片
2. **Quick-Start Time-Uniform Slicing Algorithm**：基于 attention FLOPs 模型确定切片方案

$$FLOPs(l, c) = (4H + 3h) \cdot l \cdot c + 8H^2 \cdot l$$

其中 l 是微批次 token 数，c 是累积上下文长度，H 是 d_model，h 是注意力头数。算法先分配 minimal quick-start slice（ceil(L/d)），然后迭代确定后续切片边界使每个微批次的 FLOPs 接近理想值 t̂。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

以 8-token 序列、d=4 为例：

```
Token-uniform (问题):    attention 时间 [2, 3, 4, 5] ms → pipeline 不均衡
Time-uniform (FOLDMOE): attention 时间 [4, 4, 4, 4] ms → 饱和阶段完美 overlap

切片方案: S = {4, 2, 1, 1} (early slices 更多 token，later slices 更少 token)
Token Buffer 操作:
  Attn 产出 Z_{1:4} → Buffer: [Z1,Z2,Z3,Z4] → emit Z_{1:2} to MoE (size=2)
  Attn 产出 Z_{5:6} → Buffer: [Z3,Z4,Z5,Z6] → emit Z_{3:4} to MoE (size=2)
  Attn 产出 Z_{7:7} → Buffer: [Z5,Z6,Z7]    → not enough, wait
  Attn 产出 Z_{8:8} → Buffer: [Z5,Z6,Z7,Z8] → emit Z_{5:6} to MoE (size=2)
  Buffer: [Z7,Z8] → emit Z_{7:8} to MoE (size=2)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Token Buffer 是 FOLDMOE 框架内部的软件数据结构（FIFO queue），位于 attention 层输出和 A2A dispatch 之间
- 切片算法（Algorithm 1）在训练开始前根据 seqlen、模型配置、overlap degree 预先计算切片方案，时间复杂度 O(L)
- 算法约束确保 token buffer 始终有足够 token 满足 MoE 侧的最小微批次需求：∑_{i=1}^j l_i ≥ (j/d)·L

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
