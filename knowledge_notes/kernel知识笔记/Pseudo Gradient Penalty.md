## Pseudo Gradient Penalty

术语解释
EDiT 的核心组件，一种防止 Local SGD 训练中 loss spike 的梯度质量控制策略。通过三步级联：异常 worker 排除→加权平均→梯度裁剪。

术语是什么？
大规模异构集群中，部分 worker 可能因硬件故障、数据噪声等产生异常梯度。该策略使用 pseudo gradient（当前参数与上一同步步参数的差值除以学习率）替代真实梯度进行 worker 间质量评估：(1) EMA 追踪每个 worker 的 pseudo gradient norm，偏离阈值则排除该 worker；(2) 剩余 worker 按 pseudo gradient norm 反比加权（norm 越小权重越高，表示更稳定的更新）；(3) 超过全局阈值的 fused gradient 被裁剪。

术语一般如何实现？如何使用？
- 集成在 EDiT/DRLover 中
- pseudo gradient 比真实 gradient 更稳定（不受单步数据波动影响）
- EMA 阈值需根据训练规模调整

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---
