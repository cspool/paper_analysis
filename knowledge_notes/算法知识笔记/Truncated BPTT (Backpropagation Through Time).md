## Truncated BPTT (Backpropagation Through Time)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Truncated BPTT 是将完整 BPTT（沿整个序列反向传播梯度）截断为固定长度 K 的技巧，通过在每个截断边界 detach 隐藏状态来阻止梯度继续向更早的时间步流动。Stuffed Mamba 论文使用此技术训练长上下文 Mamba-2：将 12 个序列拼接，每个序列处理后 detach 隐藏状态，下一个序列从 detach 的状态继续前向。这等价于 concatenation + 在序列边界截断梯度。目的：(1) 使状态初始值分布更多样化（非始终零初始化）；(2) 降低内存成本——只需缓存 K 步的激活值，而非完整序列。GLA 论文（Yang et al., 2024a）先提出此方法用于扩展 RNN 的上下文长度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Truncated BPTT with 12 sequences per sample
h = zeros(N, P)  # 初始状态
total_loss = 0

for seq_idx in range(12):  # 12 个拼接序列
    # 前向传播（状态连续）
    for t in range(len(seq)):
        h = update(h, seq[t])       # 使用上一序列的最终状态
        total_loss += CE(head(h), seq[t+1])

    # 梯度截断边界
    h = h.detach()  # 停止梯度向更早序列传播

    # 反向传播仅到当前序列
    total_loss.backward()  # 梯度传播范围: 当前序列
    optimizer.step()
    total_loss = 0
```
关键区别：状态 h 在推理时连续流动（保存上下文信息），但梯度在序列边界截断（节省内存）。每序列内的 BPTT 长度 = 序列长度，总共 12 个独立反向传播段。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：在每个序列边界调用 `h = h.detach()`。Stuffed Mamba 使用 12 序列拼接（≈ 12 × T_train 总长），结合 WSD LR scheduler 和 0.5M tokens/batch。该技术已被 GLA、RWKV、Mamba-2 等 RNN 训练广泛采用。优势：内存 O(K) vs 完整 BPTT 的 O(T)，且状态初始化条件更丰富（非始终零初始化）。局限：无法学习跨 K 步的长距离依赖——但论文中的 12×8K=96K 已足够捕获大多数依赖。在长序列训练中，推荐与 gradient clipping（论文用 1.0）配合使用。

涉及论文标题：
- Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

---
