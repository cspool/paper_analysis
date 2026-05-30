## Mini-batch TTT（小批量测试时训练）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mini-batch TTT 是 TTT 层中用于并行化内循环梯度计算的技术。标准的 online gradient descent（每个 token 后立即更新 W）无法并行化，因为 W_t 依赖于 W_{t-1}。Mini-batch TTT 将输入序列分成大小为 b 的 mini-batch，在 mini-batch 内部共享起始权重 W_{t'}（其中 t' = t - mod(t, b) 为上一个 mini-batch 的结束时间步），从而可以在 mini-batch 内并行计算 b 个梯度。更新公式：G_t = ∇ℓ(W_{t'}; x_t)，W_t = W_{t-1} - η G_t。b=1 等价于 online GD，b=T 等价于 batch GD。论文选择 b=16 作为 quality-speed 的最优折中。从消融实验（Table 1）看，从 batch GD (b=T=2048) 切换到 mini-batch GD (b=16) 是将 linear attention 转变为 TTT-Linear 的最大单一改进（PPL 从 14.05 降至 12.35，-1.70）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mini-batch TTT 的数学描述：

```
设序列长度 T，mini-batch size b（假设 T 整除 b）

W_0 = θ_init

for m = 0, 1, ..., T/b - 1:       # m 为 mini-batch 索引
    t_start = m * b
    W_ref = W_{t_start}             # mini-batch 内共享的参考权重

    # parallel: 计算 b 个梯度（均对 W_ref 求导）
    for i = 1, ..., b:
        t = t_start + i
        G_t = ∇ℓ(W_ref; x_t)       # 对 W_ref 求导，而非 W_{t-1}

    # sequential: 逐 token 累积更新
    for i = 1, ..., b:
        t = t_start + i
        W_t = W_{t-1} - η · G_t    # 从 W_{t-1} 开始梯度步

# 信息传播的两个通道：
# 1. cumsum 通道（始终活跃）：W_t = W_0 - η Σ_{s=1}^t G_s
# 2. 梯度通道（仅在 mini-batch 边界活跃）：G_t 对 W_{t'} 求导
```

直观理解：b 控制搜索空间大小 vs. 并行度的权衡。较小的 b 意味着更多的梯度步（更大的有效搜索空间），但更少的并行机会；较大的 b 意味着更高的并行度但更少的梯度步。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在实践中，mini-batch TTT 与 dual form 结合使用。对于每个 mini-batch：
1. 使用 dual form 一次性计算出 W_b（mini-batch 结束时的权重）和 z_1,...,z_b（mini-batch 内所有输出 token）
2. W_b 作为下一个 mini-batch 的参考权重 W_ref
3. 论文设定 b=16（所有实验），η_base=1.0 (TTT-Linear) 或 0.1 (TTT-MLP)

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States
