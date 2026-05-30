## Expert Activation Priority Detection in MoE

术语解释
Priority Detection 是 HD-MoE 动态调度中的预测组件，利用 MoE 相邻层之间 expert 激活的时间局部性（temporal locality），为每个节点上的每个 expert 计算优先级分数 prio_ic = 2·P_ic·f̂_i·IS/comp，识别将成为下一层计算瓶颈的热点 expert。

术语是什么？
Expert Activation Priority Detection 是一种基于预测的运行时负载估计机制。其核心假设是 MoE 模型中相邻 transformer layers 的 gating 输出具有高度相似性——这是由于 residual connections（skip connections）使得相邻层的 hidden state 差异较小，gate 网络在这些相似输入上的输出也相似。HD-MoE 论文（Figure 3d）展示了后续层的 gate 函数对当前层 expert activation 的高预测精度，验证了这一假设。Priority score 综合考虑了三个因素：(1) expert 在节点上的分配比例 P_ic，(2) 预测的激活频率 f̂_i，(3) expert 的计算量 IS/comp。

从系统架构角度拆解术语
Priority Detection 的执行流程：
```
# 在 Layer l 的 expert computation 期间执行
# 利用 Layer l 的实际 gating 输出预测 Layer l+1 的热点

for each node c in parallel:
    # 从 Layer l 的 gate 输出获取实际 expert activation
    actual_gate_l = gate_output[l]  # shape: (B, E)
    
    # 预测 Layer l+1 的 expert activation
    # 利用 temporal locality: gate_{l+1}(x_{l+1}) ≈ gate_{l+1}(x_l)
    # 其中 x_{l+1} = x_l + FFN(x_l) + Attn(x_l)，residual connection 保证相似性
    predicted_activation_lplus1 = predict_next_layer(actual_gate_l)
    
    for each expert i with P_ic > 0:
        f̂_i = predicted_activation_lplus1[i] / B  # 归一化激活频率
        prio_ic = 2 · P_ic · f̂_i · IS / comp       # 预估该 expert 的计算时间

# 全局选择最热点 expert
bottleneck_node = argmax_c(Σ_i prio_ic)
hot_experts = top_k({prio_i,bottleneck_node}, k=available_broadcast_slots)
```

术语一般如何实现？如何使用？
预测模型可以使用简单的 "last-value" 预测（即 f̂_i[l+1] = f_i[l]，假设相邻层完全相同）或更复杂的滑动平均/指数平滑。HD-MoE 论文（Figure 3d）的实际预测精度已经足够高，说明 "last-value" 或简单移动平均即可满足需求。更复杂的实现可使用小型神经网络预测器，但 overhead 可能抵消收益。该技术也可推广到 GPU 集群上的 MoE serving——预测热点 expert 并提前加载到 GPU memory 或执行 expert replication。

涉及论文标题：
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing
