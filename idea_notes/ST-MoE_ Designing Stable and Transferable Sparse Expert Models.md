## ST-MoE: Designing Stable and Transferable Sparse Expert Models

- baseline方法是什么？
  - Baseline 方法：(1) **Dense T5 模型**（T5-Large, T5-XXL）：标准 Transformer encoder-decoder，所有 token 经过相同的 FFN 层，无稀疏 MoE 路由。全栈执行例子：输入 token → embedding lookup → 27 层 encoder（每层 self-attention → RMSNorm → FFN_{GEGLU} → dropout → residual add），24 层 decoder（含 enc-dec attention），输出 logits → softmax 预测。硬件为 TPU，Mesh Tensorflow 分布式训练，bfloat16 mixed precision，Adafactor optimizer。(2) **Switch Transformer**（Fedus et al., 2021）：token-based top-1 routing，单 expert 路由，expert 容量限制，训练不稳定且微调质量低于密集模型。全栈执行例子：token → router 计算 softmax → 选择 top-1 expert → 若 expert 容量未满则 dispatch via all2all → GEGLU FFN 计算 → combine via all2all → residual add（若 token 被 drop 则跳过计算直接 residual）。核心缺陷：训练 1/3 运行不稳定、微调质量落后于密集模型（尤其在 SuperGLUE 等小任务上）、训练时无有效的稳定化机制。
  - 全栈执行例子（Baseline: Switch Transformer / Dense T5, TPU, single token）：
    ```
    # 算法层：标准 MoE routing，token 进入 → router softmax → top-1 dispatch → GEGLU FFN
    router_logits = W_r @ x  # [d_model, num_experts]
    gate = softmax(router_logits)  # float32 selective precision
    expert_id = argmax(gate)  # top-1 routing
    # 无稳定化 loss，logits 可能非常大 → bfloat16 roundoff error 累积
    
    # 系统框架层（Mesh Tensorflow）：
    # Data parallelism over rows, Model parallelism over columns
    # einsum one-hot tensor dispatch/combine, group-based load balancing
    group_size = batch / num_groups  # 通常 2-8 sequences/group
    token_capacity = CF * group_size / num_experts  # train CF=1.25
    
    # 编译框架层：Mesh Tensorflow 编译为 TPU XLA，einsum → all2all → matmul → all2all
    # Kernel 调度层：bfloat16 matmul on TPU MXU (systolic array)，float32 allreduce
    # 硬件架构层：TPU v3/v4 cores，HBM，interconnect for all2all
    
    # === 缺陷 ===
    # 1. 训练不稳定 → 大 logits 进入 router softmax → bfloat16 roundoff errors
    # 2. 微调过拟合 → 稀疏模型收敛快但在小任务(CB/WSC)上验证质量差
    # 3. 微调超参数 → 使用密集模型最优超参数(batch_size/learning_rate)会掩盖稀疏改进
    ```

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：**ST-MoE（Stable and Transferable Mixture-of-Experts）**通过三个维度解决 baseline 缺陷：
    - **(A) Router Z-Loss 稳定训练**：引入 $L_z(x) = \frac{1}{B} \sum_i (\log \sum_j e^{x_j^{(i)}})^2$ 作为辅助损失，直接约束路由器输出 logits 的绝对值大小。这解决了 routing 时 bfloat16 roundoff errors 导致的训练不稳定问题（大 logits 在 softmax 中的微小扰动会巨大改变输出）。Router z-loss 在不损害模型质量的前提下使训练 100% 稳定，甚至轻微提升质量（neg log perp 从 -1.755 → -1.741）。相比之下，其他稳定方法（移除 GEGLU、加 noise、tight update clipping）都严重损害质量。
    - **(B) 差异化微调协议**：发现稀疏模型需要与密集模型不同的微调超参数——更小的 batch size（65k vs 1M）和更高的 learning rate。提出仅更新非 MoE 参数子集的微调策略以对抗过拟合。这解决了稀疏模型微调质量差的问题，使得 ST-MoE-32B 在 SuperGLUE 上达到 91.2 的 SOTA（超过人类 baseline）。
    - **(C) 架构设计原则**：sparse-dense stacking（每个 MoE 层前后加 dense FFN 层，改善 -0.014 neg log perp）、multiplicative bias（expert FFN 中加逐元素乘性偏置，4% 收敛加速）、实际 capacity factor 考量（train CF=1.25 在 Pareto 效率上优于 CF=2.0，后者导致 +14% step time）、以及 BPR routing for low CF。
  - 全栈执行例子（ST-MoE-32B, TPU, single token）：
    ```
    # === 算法层 ===
    x = embedding(token_id)  # d_model=5120
    
    # Router Z-Loss (training only, not inference)
    router_logits = W_r @ x  # [5120, 64]
    z_loss = mean(logsumexp(router_logits)**2)  # 惩罚大logits
    total_loss = CE_loss + 0.01 * load_balance_loss + 0.001 * z_loss
    
    # Top-2 routing with capacity factor and relative threshold
    gate = softmax(router_logits)  # cast to float32
    top2_gates, top2_indices = top_k(gate, k=2)
    top2_gates = top2_gates / sum(top2_gates)  # renormalize
    # 2nd expert routed if gate2/gate1 > 0.2, else stochastic
    
    # Sparse-Dense Stacked FFN
    x = DenseFFN(x)                    # 先经过 dense FFN
    # Multiplicative Bias in Expert
    for expert_i in top2_indices:
      expert_out = [GELU(x @ W_11) * (x @ W_12) * B_scale] @ W_2
      # B_scale: learnable [1, d_ff] initialized to ones, 逐元素乘
    y = DenseFFN(sum(gate_i * expert_out_i))  # 后接 dense FFN
    
    # === 系统框架层（Mesh Tensorflow）===
    # 2D mesh: d x m (data x model parallelism)
    # 64 experts, expert layer frequency = 1/4
    # train CF=1.25 (Pareto optimal), eval CF=2.0
    # all2all dispatch/combine, allreduce for model parallelism
    # 3D mesh when num_experts < data_parallel_rows
    
    # === 微调策略 ===
    # 仅更新 Non-MoE 参数 (Attention + Non-MoE FFN params, ~20% of total)
    # batch_size=65k, learning_rate=1e-3 (sparse 偏好更小 batch, 更高 lr)
    # dropout=0.1 global, 额外 expert dropout
    # aux load balancing loss 在微调时可关闭（不影响质量）
    
    # === 编译框架层 ===
    # Mesh Tensorflow → XLA → TPU (bfloat16 matmul, float32 accumulate)
    # === Kernel 调度层 ===
    # TPU MXU: bfloat16 matrix multiply, GELU via lookup table
    # === 硬件架构层 ===
    # Google TPU v3/v4, HBM, inter-chip interconnect (ICI)
    ```

  - 解决效果对比：
    | 缺陷 | Baseline | ST-MoE 解决方案 | 效果 |
    |------|----------|-----------------|------|
    | 训练不稳定 | 1/3 runs diverge | router z-loss | 100% stable, 质量略微提升 |
    | 微调质量差 | 密集模型更好 (小任务) | 差异化微调协议 + 参数子集更新 | SOTA on SuperGLUE (91.2) |
    | 微调过拟合 | CB: 100% train acc, 低 validation | 更小 batch size, 更高 lr | ST-MoE-L 在所有 task 上超越 Dense-L |
    | CF 过大低效 | CF=2.0 慢 14% | train CF=1.25 Pareto 选择 | 几乎相同的质量，显著降低计算成本 |
    | 小规模结论不迁移 | top-1 vs top-2 结论反转 | 更大规模 (8x) 实验验证 | top-2 在更大规模下优于 top-1 |
