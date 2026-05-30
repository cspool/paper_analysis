## EDiT (Elastic Distributed Training)

术语解释
EDiT 是基于 Local SGD 的高效异步分布式训练方法，由 Ling 团队采用并贡献到 ICLR 2025 (Cheng et al.)。结合 layer-wise synchronization、pseudo gradient penalty 和 time-based synchronization 三个机制，解决传统同步 All-Reduce 训练在大规模集群中的 straggler 问题。

术语是什么？
传统同步分布式训练（All-Reduce）面临四个挑战：(1) 高通信开销；(2) straggler 节点瓶颈所有节点；(3) 弹性训练困难；(4) 对数据噪声敏感。EDiT 通过三个核心机制解决：

1. **Layer-wise synchronization**：逐层同步参数（非全局 barrier），prefetch 机制将下一层通信与当前层计算重叠。
2. **Pseudo gradient penalty**：(a) EMA 追踪 pseudo gradient 检测异常 worker 并排除；(b) 按 pseudo gradient norm 加权平均剩余 worker 的梯度；(c) 统一梯度裁剪防止发散。
3. **Time-based synchronization**：按时间阈值而非固定步数触发同步——快节点可执行更多局部更新，动态适应异构环境。

在理想环境中加速比可达 66.1%（baseline 速度降至 5.49e-2 step/s 时）。

从kernel调度角度拆解术语（EDiT 通信-计算 Overlap）：
```
Workers W_0..W_3, layers L_0..L_N:

for step in training:
    for layer in model:
        # Forward + Backward (各 worker 独立)
        hidden = layer.forward(hidden)
        grad = layer.backward(loss)

        # Layer-wise sync + prefetch
        if layer % sync_interval == 0:
            async_broadcast_layer_weights(layer)  # 非阻塞
            # 下一层 forward 与此层 broadcast 并行

    # Pseudo Gradient Penalty（同步时）
    pseudo_grad = (curr_params - prev_params) / lr
    if |norm(pseudo_grad) - EMA(norm)| > threshold:
        exclude_worker(i)                       # 异常排除
    fused = weighted_avg(valid_pseudo_grads)    # 加权平均
    fused = clip(fused, threshold)              # 梯度裁剪

    # Time-based sync
    if elapsed > sync_deadline: sync_all()
```

术语一般如何实现？如何使用？
- 论文链接：https://openreview.net/forum?id=xtlMtbVfWu
- Ling 团队在异构加速器集群上使用 EDiT 训练 300B MoE 模型
- 与 DLRover 框架集成
- 在 straggler 严重的异构环境中加速效果更显著

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---
