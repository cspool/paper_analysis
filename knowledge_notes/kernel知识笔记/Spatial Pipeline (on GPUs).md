## Spatial Pipeline (on GPUs)

术语是什么？
Spatial Pipeline 是 Kitsune 提出的 GPU kernel 执行抽象：将 DL 计算图的不同算子映射为 pipeline 的不同 stage，各 stage 对应一组 CTA，stage 间通过片上 queue 传递 tile 级数据。不同 stage 的 CTA **在空间上并发**（不同 SM 或同一 SM 的不同执行单元），实现 operator 级的流水线并行。通过 cudaPipeline API 暴露给开发者，语义类似 CUDA Graphs 但要求所有 kernel co-resident on GPU。

从kernel调度角度拆解术语：
Spatial pipeline 的调度结构：

```
sf-node: {stage_0, stage_1, ..., stage_n}
Queue: {queue_0, queue_1, ..., queue_{n-1}}  // stage_i → queue_i → stage_{i+1}

ILP最优CTA分配 (Algorithm 2):
  maximize Throughput
  subject to:
    Throughput < t_i × ResourceScale(a_i) × Speedup(a_i)  // 每stage性能约束
    Throughput × DRAM_Bytes < DRAM_Bandwidth               // 内存带宽约束
    Throughput × L2_Bytes < L2_Bandwidth                   // L2带宽约束
    1 ≤ a_i ≤ #SMs
    Σ IsSimt_i × a_i = #SMs     // SIMT和Tensor类CTA重叠SM分配
    Σ IsTensor_i × a_i = #SMs   // 利用不同execution unit的独立性
```

以 MeshGraphNets MLP forward pass 为例：
```
sf-node = {Linear_1 (256×1024 GEMM), ReLU, Linear_2 (1024×256 GEMM)}
Queue_0: Linear_1 → ReLU (payload: 64-256KB tiles)
Queue_1: ReLU → Linear_2

ILP求解: a_Linear1=64, a_ReLU=44, a_Linear2=44
  → 64+44=108 SM for stage 0+1 (SIMT/Tensor overlap: ReLU用SIMT, Linear1用Tensor)
  → 44 SM for stage 2 (Linear2用Tensor, 可能与其他stage重叠)
  → 152 CTAs 压缩到108 SM预算内（通过类型互补）
```

与垂直融合的关键对比：(a) Spatial pipeline 中不同 stage 分布在**不同 CTA**（空间并行），垂直融合中不同 operator 在**同一 CTA**内 temporal multiplex（时间复用）；(b) Spatial pipeline 通过 queue 在 CTA 间传递数据（L2 resident），垂直融合通过 shared memory 在 CTA 内部传递；(c) Spatial pipeline 支持隐藏维度/归约维度的并行（通过 queue 的多对一拓扑），垂直融合不支持。

术语一般如何实现？如何使用？
Kitsune compiler 自动 lowering：Subgraph Selection → Pipeline Design → Load Balance (ILP) → CUDA kernel 改写（每个 kernel 约 8 人时手动改写，10-40 LOC）。Modified grid scheduler 的 cudaPipeline API 带 kernel type metadata（SIMT/TENSOR），双 arbiter 实现异构 CTA pairing。

涉及论文标题：
- Kitsune: Enabling Dataflow Execution on GPUs

---
