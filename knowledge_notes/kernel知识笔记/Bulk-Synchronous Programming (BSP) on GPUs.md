## Bulk-Synchronous Programming (BSP) on GPUs

术语是什么？
Bulk-Synchronous Programming (BSP) 是 GPU 默认的执行模型：每个 DL 算子映射为单个 CUDA kernel，一个 kernel 独占 GPU，所有 CTA 完成后 global barrier，再 launch 下一个 kernel。源自 Leslie Valiant 的 BSP 并行计算模型（1990），GPU 采用简化版：每个 kernel 的 CTA 独立执行到完成 → implicit global barrier → 下一个 kernel。现代 GPU 虽通过 CUDA Streams 支持有限的多 kernel 并发，但 grid scheduler 的 FIFO 设计使 kernel 间几乎没有执行重叠——新 kernel 需等当前 kernel 全部 CTA dispatch 后才开始 dispatch。

从kernel调度角度拆解术语：
BSP 模型在 GPU 上的执行流程：

```
// PyTorch eager 执行 MLP: Linear→ReLU→Linear
kernel_Linear1<<<grid, block>>>(input, intermediate1);
// implicit barrier: GPU等待所有CTA完成
// intermediate1 写入DRAM (non-resident on-chip)

kernel_ReLU<<<grid, block>>>(intermediate1, intermediate2);
// barrier: 等待所有CTA完成
// intermediate2 写入DRAM

kernel_Linear2<<<grid, block>>>(intermediate2, output);
// barrier: 等待所有CTA完成
```

BSP 的三大缺陷（Kitsune 论文分析）：
1. **资源闲置**：单 kernel 执行时 TensorCore 或 SIMT core 之一空闲。论文实测 inference 中 20-25% runtime、training 中 37-67% runtime 中 SM 和 DRAM 利用率均 <33% 峰值。
2. **大 intermediate 溢写 DRAM**：MLP hidden dim ≥ 768（A100 192KB shared memory 约束）时 intermediate tile 超出 SM 片上容量，必须 round-trip DRAM（A100 latency ≈ 409ns/572 cycles）。
3. **无法利用 reduction/hidden 维度并行**：如 back-propagation 中 batch 维度 gradient reduction，仅少数 CTA 执行 reduce，大多数 SM 空闲。

Kitsune 通过 Synchronous Dataflow Execution 解决以上三个问题：(1) heterogeneous CTA co-location 解决资源闲置；(2) L2-resident queue 消除 DRAM spill；(3) parallel reduction tree via queue 解决 reduction 并行不足。

术语一般如何实现？如何使用？
BSP 是 GPU 的默认执行模型，由 CUDA driver + grid scheduler 硬件强制执行。开发者通过 CUDA kernel launch 使用，无需额外编程。垂直融合（Vertical Fusion）和 Kitsune dataflow 是在 BSP 上的不同突破：Vertical Fusion 在单个 CTA 内 temporal multiplex 多个算子避免 barrier；Kitsune 改用 spatial pipeline 实现真正的并发。

涉及论文标题：
- Kitsune: Enabling Dataflow Execution on GPUs

---
