## Expert-wise CPU Optimization (专家级CPU优化器 / Streaming CPU Optimizer)

术语是什么？
Expert-wise CPU Optimization 是 ES-MoE 提出的一种 optimizer 调度策略，将 CPU-based optimizer 的执行粒度从传统 layer 级或 model 级细化到 expert 级。在 MoE 训练中，当 expert 参数和 optimizer states 被 offload 到 CPU 后，CPU 上的 Adam optimizer 更新比 GPU 慢约 31×。传统方案（如 ZeRO-Offload）在整 layer 所有 expert 的 backward 完成后再触发 optimizer，或使用 delayed update（延迟梯度更新到下一 iteration，但引入 staleness 影响精度）。ES-MoE 的方案是：每个 expert 完成 backward pass 后**立即**触发其 CPU Adam step，而不等待同一 layer 的其他 experts——靠近 output 的 layers 的 expert optimizer 执行时间被靠近 input 的 layers 的 GPU backward 计算隐藏。

从kernel调度角度拆解：
Expert-wise CPU Optimization 与 GPU backward 的 overlap 时序（gantt 视角）：

```
Timeline (MoE-L, 24 layers, 4 GPUs, 16 experts):
================================================================================
Layer 23 (closest to output):
  GPU: [E0 backward][E1 bwd][E2 bwd][E3 bwd]
  CPU:              [E0 Adam][E1 Adam][E2 Adam][E3 Adam]  ← 与 GPU 其他 layers 重叠
                          ↓ CPU Adam 与 GPU Layer 22 backward 并行 ↓
Layer 22:
  GPU: [E0 backward][E1 bwd][E2 bwd][E3 bwd]
  CPU:              [E0 Adam][E1 Adam][E2 Adam][E3 Adam]
                          ↓ 与 GPU Layer 21 backward 并行 ↓
...
Layer 0 (closest to input):
  GPU: [E0 backward][E1 bwd][E2 bwd][E3 bwd]
  CPU:              [E0 Adam][E1 Adam][E2 Adam][E3 Adam]
================================================================================
End-to-end: CPU Adam latency ~hidden by GPU backward of preceding layers
```

伪代码：
```python
def expert_wise_cpu_optimization(layer_backward_graph):
    for layer in reversed(model.layers):  # backward: output → input
        for expert in layer.experts:
            # GPU: compute expert gradients
            expert_grads = expert.backward(activations)  # on GPU
            # Immediately trigger CPU Adam (async, non-blocking)
            async_cpu_adam_update(
                expert.params_cpu,      # offloaded params in CPU RAM
                expert_grads.cpu(),     # download gradients GPU→CPU
                expert.opt_states,      # Adam m, v on CPU
                lr, beta1, beta2
            )
            # GPU continues with next expert / next layer
            # CPU Adam runs concurrently
```

与 ZeRO-Offload 的关键区别：ZeRO-Offload 使用 **delayed update**（将 optimizer 延迟到下一 iteration 以隐藏延迟），但引入 staleness——参数更新使用的是上一 iteration 的梯度，可能影响模型精度。ES-MoE 的 expert-wise optimization 使用**当前 iteration 的梯度**即时更新，保持数学等价性。Ablation 结果显示，expert-wise optimizer overlapping 贡献了 8.7% 的 throughput 提升，且不影响精度。

术语一般如何实现？如何使用？
- 实现依赖 CPU 多线程 + GPU streams 的异步执行
- DeepSpeed CPU Adam 基于 AVX2/AVX-512 向量指令优化，单 socket 可达数十 GFLOPS
- 关键挑战：CPU optimizer 的吞吐量受 CPU 核心数限制（ES-MoE 使用 32-core EPYC 7543）
- 当 number of layers 较少或 experts 较少时，overlap 效果减弱（CPU optimizer 可能暴露在 critical path 上）

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
