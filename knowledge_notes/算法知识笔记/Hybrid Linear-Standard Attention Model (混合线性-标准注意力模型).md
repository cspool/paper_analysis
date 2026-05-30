## Hybrid Linear-Standard Attention Model (混合线性-标准注意力模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hybrid model 是将线性注意力层与标准 softmax 注意力层混合的 Transformer 架构。纯线性注意力模型虽训练推理高效，但在 recall-intensive 任务（in-context learning、Needle-in-a-Haystack）上表现不佳。Hybrid model 在部分层保留标准 softmax attention（提供 recall 能力），其余层使用线性注意力（提供长序列高效处理），实现吞吐量与模型能力的平衡。

典型 Hybrid 配置：每 4 层用 1 层标准 attention + 3 层线性 attention（1/4 hybrid），或 1/8、1/2 hybrid。LASP-2 消融实验（Table 4）表明，更高 hybrid ratio 通常带来更好 convergence，但标准 attention 层会增加 quadratic 计算和通信开销。

从算法pipeline角度拆解术语。

**1/4 Hybrid Linear-Llama3 架构 + LASP-2H SP**：

```
Layer 1: Linear Attention  → SP: AllGather M_t (d×d)
Layer 2: Linear Attention  → SP: AllGather M_t
Layer 3: Linear Attention  → SP: AllGather M_t
Layer 4: Standard Attention → SP: AllGather K_t, V_t (C×d) + Softmax(QK^T/√d)V
...每 4 层循环...

// 统一 AllGather-based 通信范式:
//   Linear layer: 通信 M_t [B, H, d, d] — 与序列长度无关
//   Standard layer: 通信 K_t, V_t [B, H, C, d] — 与 chunk 长度有关
```

术语一般如何实现？如何使用？

Hybrid model 通过修改模型配置实现（指定层类型）。LASP-2H 为两类层提供统一的 AllGather-based SP。修改 Llama3 源码将指定层的 `LlamaAttention` 替换为 `LinearAttention`（Triton kernel 实现），其余层保持不变。代码开源：https://github.com/OpenSparseLLMs/Linear-MoE。

涉及论文标题：
- LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid

---
