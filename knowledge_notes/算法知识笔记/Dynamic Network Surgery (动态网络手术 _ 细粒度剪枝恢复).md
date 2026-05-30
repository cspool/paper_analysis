## Dynamic Network Surgery (动态网络手术 / 细粒度剪枝恢复)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Network Surgery (DNS) 是由 Guo et al. (NIPS 2016) 提出的细粒度剪枝方法。与传统的"剪完即弃"（prune and discard）不同，DNS 在剪枝后持续训练中允许被剪掉的连接**恢复**（surgery 操作）。具体地：DNS 维护一个 binary mask T_k 指示连接是否被剪枝。在训练迭代中：(1) 若 `|w_k * T_k|` 低于阈值 `a_k`，则 mask 被设为 0（剪枝）；(2) 若 `|w_k * (1 - T_k)|` 高于阈值 `b_k`，则 mask 被恢复为 1（surgery，恢复连接）。阈值 `a_k` 和 `b_k` 通过每层的均值和标准差自适应调整。

这种"可逆剪枝"设计使 DNS 能探索 sparser 的结构而不永久丢失有用的连接，最终达到比静态剪枝更高的压缩率和更好的精度。FQ 论文使用 DNS 作为其压缩 pipeline 的剪枝阶段，使 ResNet-50 达到 82.70% 稀疏度（仅 17.3% 连接保留）且精度损失仅 0.48% (Top-1)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
DNS 剪枝训练循环伪代码：

```
# Initialize: W (weights), T (mask, all 1s)

for each training iteration:
    # Forward/backward with masked weights
    W_masked = W ⊙ T  # element-wise multiply with mask
    loss = forward(W_masked, x)
    backward(loss)

    # Update weights (all weights updated, even pruned ones)
    W = W - lr * ∇W

    # Dynamic pruning and surgery (per layer)
    for each layer l:
        # Compute layer-wise thresholds
        a_k = μ_l + c_a * σ_l  # pruning threshold
        b_k = μ_l + c_b * σ_l  # surgery (recovery) threshold
        # Note: c_b < c_a, so b_k < a_k

        for each weight w at index i:
            if T[i] == 1 and |W[i]| < a_k:
                T[i] = 0  # prune
            elif T[i] == 0 and |W[i]| > b_k:
                T[i] = 1  # surgery: restore connection

    # Periodically increase sparsity target
    if should_increase_sparsity():
        increase_threshold(c_a, c_b)  # tighten thresholds
```

**Annotations**: `c_a > c_b` 保证剪枝门槛高于恢复门槛，形成 hysteresis 防止连接反复切换；sparsity 通过逐步收紧阈值增量式提升；剪掉的权重仍然接收梯度更新（通过 1-T 通道），使其有机会在后续训练中恢复到有意义的值从而被"手术恢复"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
DNS 的实现需要修改标准训练循环，在每个 optimizer step 后插入 pruning/surgery 逻辑。实现要点：(1) 维护 per-layer mask；(2) 每层的 μ_l 和 σ_l 从当前权重统计中计算（通常对非零权重）；(3) sparsity 通过 schedule 渐进增加（如 cosine annealing of sparsity target）。DNS 特别适合与后续量化和编码步骤组合成完整的压缩 pipeline（如 FQ 的 FC pipeline）。Mayo 框架 (https://github.com/deep-fry/mayo) 包含 DNS 实现。

涉及论文标题：
- Focused Quantization for Sparse CNNs
