## Layer-dependent Context Dynamics (层依赖的上下文动态)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Layer-dependent Context Dynamics 是 FastKV 论文通过实验揭示的一种 LLM 内部注意力动态现象：transformer decoder 的不同层对关键 token 的注意力焦点表现出显著不同的行为模式。通过喂入 128K token 输入到 LLaMA-3.1-8B-Instruct，在每层收集获得最高平均注意力质量（across heads）的 top-512 critical tokens，计算层间 critical token 索引的平均重叠率随层距离的变化曲线（Figure 1a），发现：(1) 早期层（≤15）：重叠率随层距离增大而急剧下降，表明各层的 critical token 集合频繁变化——注意力高度不稳定；(2) 后期层（>15）：重叠率衰减显著减缓，表明同一 token 子集在多个连续层中保持一致性重要——注意力趋于稳定。这一动力学解释了为何 GemFilter（单层 token 选择应用于所有层）和 PyramidInfer（从首层即开始剪枝）会导致显著的准确率损失——它们在注意力稳定之前就丢弃了后续层仍需要的 token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**分析层依赖上下文动态的实验方法：**

```
# 输入：128K token 序列 X
# 输出：每层的 top-512 critical token indices

for l = 0 to L-1:
    Att_l = layer_l(X).attention_weights      # shape: (H, N_I, N_I)
    for h = 0 to H-1:
        # 跨 query 维度平均，得到每个 key token 的全局被关注度
        token_attn_lh[i] = mean_over_queries(Att_l[h, :, i])
    token_attn_l = mean_over_heads(token_attn_lh)
    top512_l = TopK(token_attn_l, 512)         # 当前层的 critical tokens

# 计算层间重叠率（Overlap Ratio）
for delta = 1 to L-1:
    for l1 = 0 to L-delta-1:
        l2 = l1 + delta
        overlap = |top512_l1 ∩ top512_l2| / 512
        avg_overlap[delta] += overlap / (L - delta)
```

**该动力学对算法设计的关键启示：**
- 早期层（≤15）必须处理完整上下文——确保每层可自由关注其偏好的 token 子集，即使这些子集跨层差异大。
- 后期层（>15）可以安全地对上下文进行激进剪枝——各层关注的 token 子集高度重叠，剪枝引入的信息损失最小。
- TSP 层应当选在上下文稳定点之后（LLaMA-3.1-8B 的 layer 15），之前层保留完整上下文，之后层仅传播关键 token。

术语一般如何实现？如何使用？

该观察直接指导了 FastKV 的 TSP 层选择策略（Eq 3：argmin L2 distance）。在实际应用中：(1) 对不同模型需重新分析层依赖动态以选择合适的 TSP 层（LLaMA-3.1-8B 选 layer 15，Ministral-8B 选 layer 17，Mistral-Nemo-12B 选 layer 19）；(2) 也可通过 Eq 3 的自动标定方法，在少量样本上最小化 TSP 输出与完整上下文输出的 hidden state L2 距离来选层；(3) 该分析使用完整 attention map，在生产部署中不需要重复，仅在设计阶段进行一次离线分析即可。

涉及论文标题：
- FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation

---
