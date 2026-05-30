## Micro-Batch Pipeline Parallelism for MoE Training

术语是什么？
Micro-Batch Pipeline Parallelism for MoE 是将 GPipe 风格的微批次流水线引入 MoE 训练层的技术。将 MoE 层的三个阶段（All-to-All Dispatch S → Expert 计算 C → All-to-All Collect R）类比为 GPipe 的模型层，将 mini-batch 沿 batch 维度切分为 n 个 micro-batch，使不同 micro-batch 的三个阶段在多个 CUDA stream 中并行执行。与 FasterMoE 沿 device 维度切分不同，MPMoE 沿 batch 维度切分：(1) 保留 NCCL All-to-All 的集体通信优化；(2) pipeline granularity n 不受 device 数限制；(3) 交替调度 S 和 R stage 增强内存访问局部性。

从算法pipeline角度拆解术语：
MPMoE 的 pipeline 调度（以 n=4 为例）：

```
时间轴 →
Stream_comm:  S(0)---|R(0)---|S(2)---|R(2)---|
Stream_comp:         |C(0)---|C(1)---|C(2)---|C(3)---|
// S(i): dispatch, C(i): expert FFN, R(i): collect
// S 和 R 交替调度（利用 NCCL 双向通信，增强内存局部性）
```

Pipeline granularity n 的最优选择（MPMoE Figure 14）：B < 8k 时 n=2，8k ≤ B ≤ 22k 时 n=4，B > 22k 时 n=8。n 随 B 单调递增——过粗的 pipeline 导致 insufficient overlap，过细的 pipeline 导致 kernel launch overhead 和 GPU under-utilization。

术语一般如何实现？如何使用？
- 实现要点：(a) 沿 batch 维度切分：`torch.split(T_I, B/n, dim=1)`；(b) 每个 micro-batch 的 stage 在不同 CUDA stream 上异步执行；(c) 自适应 granularity：MPMoE-pb 通过 profile-based search（Algorithm 1），MPMoE-pm 通过性能模型。
- 适用场景：MoE 训练中 batch size 较大（>256 tokens/GPU）且通信/计算比高的场景。
- 局限性：n>8 时 kernel launch overhead 超过 overlap 收益；需要足够 GPU 资源支持多 stream 并发。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

---
