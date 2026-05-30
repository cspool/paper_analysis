## AdaRound (Adaptive Rounding, 自适应舍入)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AdaRound（Adaptive Rounding）是一种后训练量化中的权值舍入策略，由 Nagel et al. (ICML 2020) 提出。标准量化使用 round-to-nearest（就近舍入），但 AdaRound 发现：对于最小化任务损失，学习一个是否向上或向下舍入的决策变量比始终就近舍入更好。具体实现：对每个权值 w_i 学习一个连续变量 v_i ∈ [0,1]（通过 sigmoid 约束），最终舍入方向由 round(w_i/s + σ(v_i)-0.5) 决定，其中 σ 为 sigmoid 函数；v_i 通过重建损失（MSE 量化输出 vs 全精度输出）优化。QuEST 将 AdaRound 作为其 baseline 方法的量化策略（表 6 中 "Baseline" = 直接量化 + AdaRound），并在此基础上叠加 TLA、CMA 和 L_G 展示递进式改善。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
AdaRound 核心伪代码：
```
# 标准 Round-to-Nearest
W_int = clamp(round(W / s) + Z, qmin, qmax)

# AdaRound: 学习舍入方向
V = randn_like(W) * 0.01               # 可学习变量
for step in range(adapt_round_steps):
    # 前向：软舍入
    W_soft = clamp(floor(W / s) + σ(V) + Z, qmin, qmax)  # σ(V) ∈ [0,1]
    W_deq = (W_soft - Z) * s
    output_q = layer(W_deq, x_calib)    # 量化后输出
    loss = MSE(output_fp, output_q)     # 重建损失
    loss.backward()                     # 仅更新 V
# 推理时固定舍入决策
W_int_final = clamp(floor(W / s) + round(σ(V_final)) + Z, qmin, qmax)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AdaRound 通常作为 PTQ pipeline 的最后一步，在校准集上逐层优化舍入方向变量。在 QuEST 的上下文中，AdaRound 是 baseline 的量化策略但不作为论文的核心贡献——论文的重点是通过选择性微调超越 AdaRound 的效果。实际使用：(1) AdaRound 不需要反向传播通过量化器（仅优化 V，梯度通过 soft rounding）；(2) 适用于权重和激活；(3) 在 8-bit 效果较好，在 4-bit 效果有限（舍入误差远超 AdaRound 的调整空间），这正是 QuEST 需要权重微调来补充的原因。

涉及论文标题：
- QuEST Low-bit Diffusion Model Quantization via Efficient Selective Finetuning
