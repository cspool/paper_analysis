## Heterogeneous Load Balance Loss (异构负载均衡损失)

术语是什么？
Heterogeneous Load Balance Loss 是 MoE++ 为异构专家（FFN vs. 零计算专家）设计的负载均衡损失函数。标准 MoE 负载均衡损失将所有专家视为等价，但 MoE++ 中 FFN 专家和零计算专家的参数量和计算量差异巨大，统一分配 token 不合理。该损失引入超参数 τ 控制零计算专家与 FFN 专家的 token 分配比例：L_b = Σ η_i·f_i·P_i，其中 f_i 为专家 i 被选中频率，P_i 为平均路由概率，η_i=1（FFN 专家）或 τ（零计算专家）。较小的 τ 将更多 token 分配给零计算专家（更高 throughput），较大的 τ 将更多 token 分配给 FFN 专家（通常更高性能）。

从算法pipeline角度拆解术语：
```
# 异构负载均衡损失计算
f_i = mean(Indicator(token selects expert i))  # [N], 每个专家的选中频率
P_i = mean(Softmax(G(x))_i)                     # [N], 每个专家的平均路由概率
eta_i = 1.0 if is_ffn[i] else tau               # 权重：FFN=1, ZC=τ
L_b = sum(eta_i * f_i * P_i for i in range(N))
L_total = L_ce + beta * L_b  # beta=0.01
```

配套的异构专家容量公式：C_i_FFN = γ·τ·T/(τ·N_FFN+N_ZC)，C_i_ZC = γ·T/(τ·N_FFN+N_ZC)。当 τ=1 时退化为标准均匀分配；τ<1 时零计算专家获得更高容量。默认 τ=0.75，capacity factor γ=1.1。

术语一般如何实现？如何使用？
- 在 Megatron 训练代码的 MoE 层中对 load balance loss 计算做修改：按专家类型分组统计 f_i 和 P_i，分别乘 η_i
- τ 的选择是 throughput-accuracy trade-off：τ=0.10 时 expert forward throughput 提升最高（164.5%）但 accuracy 下降；τ=0.75 时平衡（~25% throughput 提升，accuracy 持平或略优于 baseline）
- 预训练时 τ 可以保持固定，未来工作可探索自适应 τ 策略（如训练早期大 τ 保证学习、后期小 τ 加速）

涉及论文标题：
- MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts
