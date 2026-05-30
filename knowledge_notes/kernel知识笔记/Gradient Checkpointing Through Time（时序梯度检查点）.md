## Gradient Checkpointing Through Time（时序梯度检查点）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gradient checkpointing through time 是将标准梯度检查点（gradient checkpointing / activation recomputation）技术应用于时间维度的内存优化方法。标准梯度检查点（Chen et al., 2016）在训练深度网络时，选择性地不保存某些层的中间激活，在反向传播时重新计算它们，以计算时间换取内存。TTT 层的内循环会产生 T 个中间隐藏状态 W_1,...,W_T（每个 token 对应一个 d×d 矩阵），直接全部保存会消耗不可接受的内存。使用 mini-batch TTT 和 dual form 后，仅需保存每个 mini-batch 结束时的 W（共 T/b 个，而非 T 个），但仍可能过多。Gradient checkpointing through time 在此之上进一步减少内存：仅保存部分 mini-batch 边界的 W，反向传播时从最近的检查点重新计算内循环的 forward pass。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TTT 训练中的 gradient checkpointing 执行流程：

```
# ===== Forward pass (with checkpointing) =====
W_0 = θ_init
checkpoints = []          # 仅保存部分 W

for m in 0, 1, ..., T/b - 1:    # 每个 mini-batch
    if m % checkpoint_interval == 0:
        checkpoints.append((m, W_{m*b}))   # 保存检查点

    # dual form: 从 W_{m*b} 计算 Z_block 和 W_{(m+1)*b}
    Z_block, W_next = dual_form(X_block, W_{m*b}, η)

    # 不保存中间激活（如 Z_block 计算中的中间 matmul 结果）
    # 仅保存 mini-batch 边界的 W 作为检查点

# ===== Backward pass =====
# 从最后一个检查点开始，重新计算 forward 以获取中间激活
# 标准做法：反向遍历检查点，对每段重新执行 forward + backward
for m in reversed(range(num_checkpoints)):
    # rematerialize: 从检查点重新执行该段的 forward
    # 这次保存所有中间激活用于梯度计算
    # backward: 计算梯度并传播
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实践中：
- JAX 和 PyTorch 都提供内置的梯度检查点 API（`jax.checkpoint` / `torch.utils.checkpoint`）
- 标准梯度检查点是按层应用的（每一层可独立重计算），而 TTT 需要按时间步应用（跨多个 mini-batch）
- 论文未详细说明检查点间隔的配置，但指出这是标准技术的直接应用
- 使用 mini-batch TTT (b=16) 后，T/b 个检查点已经远少于 T 个，在 2k 上下文中为 128 个检查点（可行），在 32k 上下文中为 2000 个（需要进一步减少）

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States
