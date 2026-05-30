## CUDA Stream Overlapping for Computation-Communication Overlap

术语是什么？
CUDA Stream Overlapping 是利用 CUDA 的异步执行模型，将计算（computation kernel）、通信（NCCL collective）、内存拷贝（cudaMemcpyAsync）分配到不同的 CUDA stream 中并行执行，使 GPU 的 SM（计算单元）、copy engine（内存拷贝引擎）和 NIC（网络接口卡）同时工作。在 MoE 训练中具体体现为：All-to-All dispatch/collect（通信）与 Expert FFN 计算（计算）的 overlap，以及 CPU offload 场景下 D2H/H2D 内存拷贝（mem）与前两者的 overlap。

从kernel调度角度拆解术语：
MPMoE 定义了 α(y, x) slowdown 因子量化并行操作间的干扰（Section 2.3），将操作分为三类流：comm（通信）、comp（计算）、mem（内存拷贝）：

```
// 基准：单独执行各操作
W_comp(B) // 处理 B 个 tokens 的计算时间
W_comm(B) // All-to-All B 个 tokens 的通信时间
W_mem(B)  // D2H/H2D B 个 tokens 的拷贝时间

// 并行执行时的 slowdown（α > 1 表示变慢）
实际 comp 时间 = W_comp / α(comp, comm)  // 通信对计算的干扰
实际 comm 时间 = W_comm / α(comm, comp)  // 计算对通信的干扰
实际 mem 时间  = W_mem / α(mem, comm)    // 通信对内存拷贝的干扰

// MPMoE micro-benchmark 观测（Figure 3）:
// α(comp, x) ≈ 1.0（计算几乎不受影响，因为SM和copy engine/NIC独立）
// α(comm, mem) < 1.0（通信和内存拷贝共享 memory bandwidth，互相干扰严重）
// α(comm, comp) 需 > 0.5 才能获得正向 overlap 收益
```

MPMoE 的性能模型（Section 4.2）利用这些 α 因子，在 3 种 pipeline paradigm（图 8）中估算各阶段（P0-P4）的真实执行时间。每个 paradigm 的瓶颈由最大瓶颈 CUDA stream 决定：
- Paradigm 1（仅 comp+comm，适用 S4）：`T_P2 = max((t_S+t_R)/α(comm,comp), t_C/α(comp,comm))`
- Paradigm 2（前向+mem copy，适用 S1/S2/S3 前向）：M 依赖 S 和 C 的输出
- Paradigm 3（后向+mem copy，适用 S1/S2/S3 后向）：C 依赖 M 的输入

术语一般如何实现？如何使用？
- 实现要点：(1) 使用非默认 CUDA stream（`cudaStreamCreate`），避免 default stream 的隐式同步；(2) `cudaMemcpyAsync` 需 pinned memory（`cudaMallocHost`）；(3) NCCL 通信通过 `ncclGroupStart/End` 在指定 stream 上执行；(4) 深度优先 issue order（先提交一个 stream 的所有操作，再提交另一个 stream）通常比广度优先更能实现良好 overlap。
- MPMoE 中的应用：S1/S2 策略需要 3 个 CUDA stream（comp、comm、mem）；S4 仅需 2 个（comp、comm）。S2 在 N 大时（如 64 GPU）性能恶化，因为 mem copy 和 comm 共享 memory bandwidth（α(comm,mem) 显著小于 1）。
- 局限性：(a) kernel 资源饱和时无法 overlap（如 SM 全占满时计算 kernel 已用完 GPU）；(b) 小数据量时 kernel launch overhead 抵消 overlap 收益；(c) 不同 GPU 架构的 copy engine 数量不同（A100 有 1 个 copy engine 但支持双向并发）。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

---
