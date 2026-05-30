## Bayesian Optimization for Auto-Tuning Communication Parameters

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlowMoE 使用轻量级贝叶斯优化 (Bayesian Optimization, BO) 自动搜索分布式 MoE 训练中的最优 all-reduce chunk 大小 S_p。目标函数 f(S_p) = per-iteration training time（平均 10 次迭代）。BO 使用高斯过程 (GP) 拟合 f(S_p) 的 posterior distribution，通过 Expected Improvement (EI) 采集函数选择下一个采样点。仅需约 8 次采样即可收敛到近优值，BO 开销 < 1% 迭代时间。当硬件环境变化（GPU 型号、网络带宽、模型配置）时，BO 重新执行以自动适应新环境。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
// FlowMoE BO 调优流程:
// 目标: 找到最优 S_p 使 per-iteration training time 最小

// Step 1: 初始采样
S_p_candidates = [1MB, 4MB, 16MB, 64MB, 256MB]
for S_p in S_p_candidates:
    time[S_p] = measure_avg_iteration_time(model, S_p, warmup=10, iters=10)

// Step 2-6: 贝叶斯优化迭代
for iter in range(num_bo_iters):  // 通常 3-5 次即可收敛
    // GP 拟合
    gp.fit(S_p_sampled, time_sampled)
    // 采集: Expected Improvement
    S_p_next = argmax(EI(S_p | gp))
    time_next = measure_avg_iteration_time(model, S_p_next, 10, 10)
    S_p_sampled.append(S_p_next)
    time_sampled.append(time_next)

// Step 7: 返回最优
S_p_opt = S_p_sampled[argmin(time_sampled)]
// 示例: BERT-Large-MoE 上 S_p_opt ≈ 2.5MB
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 使用 scikit-optimize 或 BoTorch (PyTorch-based BO 库) 实现
- 目标函数评估开销：10 轮迭代 × per-iteration time（GPT2-Tiny-MoE ~100ms → 评估约 1 秒, DeepSeek-V2-S ~3.2s → 评估约 32 秒）
- 约 8 次采样 → 总 profiling 开销 < 1 分钟（小模型）或 < 5 分钟（大模型），低于单次训练 epoch 的典型时长
- S_p 仅在硬件环境或模型配置变化时需要重新搜索
- FlowMoE BO 的独特之处：优化的不是模型超参数（学习率、batch size），而是系统参数（all-reduce chunk size），这使 BO 从"训练前一次性搜索"变为"环境感知的自适应调优"

涉及论文标题：
- FlowMoE: A Scalable Pipeline Scheduling Framework for Distributed Mixture-of-Experts Training
