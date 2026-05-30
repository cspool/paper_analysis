## Channel Order Preserving

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Channel order preserving 是 Mamba2 SSD 计算的一个结构属性：SSD scan 的线性递归是**逐 channel 独立**计算的，即 $h_t[c] = A_{t}[c] \cdot h_{t-1}[c] + B_{t}[c] \cdot x_t[c]$，因此输出的 channel 顺序与输入的 channel 顺序完全一致。这不同于 Transformer 的 self-attention（跨 channel 混合），是 SSM 独有的计算特性。Quamba2 利用该属性来实现 sort-and-cluster 量化：由于输出 channel 顺序被保留，通过 offline 重排输入投影的**列**、causal conv1d 的**channel**、normalization 权重和输出投影的**行**，可以保持整个 block 的 compute-invariance——量化模型的输出与未重排的 FP16 模型完全等价。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 证明 sketch（以简化 SSD 为例）
# 输入 x 重排：x' = P @ x（P 是 permutation matrix）
# SSD scan 是 channel-wise：h[c] = f(h[c], x[c])
# 因此 y'[c] = f(h[c], x'[c]) = f(h[c], x[perm(c)])
# 输出 y 被同样 permutate：y = P^T @ y'
# 通过同时重排下游的 W_out 行，恢复正确输出：
# W_out_reordered = P @ W_out（对行做逆排列）
# 最终输出不变：W_out_reordered @ y' = P@W_out @ P^T @ y' = W_out @ y（当 y'=P@y）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
仅在 SSD/selective scan 这种 channel-wise 计算中成立。Mamba1 的同类特性也可被利用（因其 selective scan 也是 per-channel 的），但论文未明确讨论 Mamba1 中的实现细节。该属性使 offline weight reordering 成为可能，是 sort-and-cluster 量化可行性的理论前提。没有该属性，sort 后的激活需要通过网上 reorder 恢复，增加延迟。

涉及论文标题：
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models
