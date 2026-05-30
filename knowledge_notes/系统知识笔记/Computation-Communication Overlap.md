## Computation-Communication Overlap

术语是什么？

Computation-Communication Overlap（计算-通信重叠）是多 GPU 分布式计算中通过异步执行计算和通信操作来隐藏通信延迟的优化技术。核心思想是利用 GPU 的异构硬件资源——Tensor Core 执行 GEMM 计算，NVLink/PCIe 等互联硬件执行通信——通过 CUDA stream 并发使得两者在时间上重叠，从而降低端到端延迟。在生成式模型的 multi-GPU 部署中，TP/DP/EP 等并行策略引入的 AllReduce、ReduceScatter、All-to-All 等集合通信操作成为主要瓶颈，computation-communication overlap 是缓解该瓶颈的关键技术。

从系统架构角度拆解术语：

多 GPU 训练/推理中三类 overlap 方案的系统级对比（以 TP=4 GEMM+AllReduce 为例）:

```
(1) Sequential (no overlap):
GPU0: |--- GEMM ---|         |--- AllReduce ---|
Total latency = T_GEMM + T_AR

(2) Decomposition-based: 沿一维分解 GEMM output 为子 tensor
GPU0: |-- GEMM(sub0) --|-- AR(sub0) --|-- GEMM(sub1) --|-- AR(sub1) --|
限制: 只能沿一维分解（否则地址不连续）；碎片化 GEMM 可能 GPU 利用率不足

(3) Fusion-based: 通信原语嵌入 GEMM kernel 内部
单 kernel 内 tile 级交织执行 compute 和 communicate
限制: 需手动实现通信原语（不可复用 NCCL）；不同通信原语需独立实现

(4) Signaling-based (FlashOverlap):
Stream A: |------ GEMM (完整, 不碎片化) ------|
Stream B: | spin-wait |-- AR(G1) --| spin-wait |-- AR(G2) --|
优势: 不碎片化 GEMM + 直接调用 NCCL + tile-wise 粒度重叠
```

术语一般如何实现？如何使用？

PyTorch DDP/FSDP 通过 gradient bucketing 实现 overlap；Megatron-LM 和 DeepSpeed 在 pipeline parallelism 中实现。FlashOverlap 在 tensor parallelism 等 data-dependent 场景实现 overlap。衡量效率的指标为 achieved speedup / theoretical speedup 比值——FlashOverlap 在大多数场景达到 >80% 理论加速比。

涉及论文标题：
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering
