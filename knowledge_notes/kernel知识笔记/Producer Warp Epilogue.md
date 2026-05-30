## Producer Warp Epilogue

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Producer Warp Epilogue 是 SageAttention3 中针对 FPGA 约束下的 warp-specialized kernel 提出的一种新型 warp 调度策略。传统 warp-specialized GEMM kernel（如 CUTLASS 高效 GEMM）中，consumer warp 负责 Tensor Core MatMul + accumulator store to global memory，producer warp 仅负责从 global memory 加载数据到 shared memory，consumer 之间做 ping-pong 重叠（while one consumer computes, another consumer stores）。然而 FP4 attention kernel 的寄存器压力极高（因 FP4MMA 的 register fragment layout 复杂 + online softmax 状态 + two-level quantization 中间值），consumer warp 同时承担 MatMul 和 store 会导致寄存器溢出（register spilling），严重拖慢性能。Producer Warp Epilogue 将 store 职责从 consumer 移交给 producer：使用两个 producer warp 做 ping-pong — 一个 producer 加载下一轮输入数据时，另一个 producer 存储上一轮 MatMul 输出到 global memory。Consumer warp 仅负责将 FP4MMA 结果从寄存器搬运到 shared memory（低开销，寄存器需求少）。此设计在受限寄存器条件下实现 MatMul 与 global store 的 overlap。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 传统 warp-specialized kernel (consumer-producer)
producer_warp: while has_tile:
    load Q/K/V tile → shared_memory
    signal consumer

consumer_warp_0, consumer_warp_1 (ping-pong):
    while has_tile:
        wait producer signal
        FP4MMA QK^T  // consumer does MatMul
        softmax + two-level quantize P
        FP4MMA PV     // consumer does MatMul
        store O to global_memory  // consumer does store ← register pressure!

// Producer Warp Epilogue (SageAttention3)
producer_warp_0, producer_warp_1 (ping-pong):
    while has_tile:
        // Phase A: Load next tile
        load Q/K/V tile → shared_memory  // producer_warp_0
        // Phase B: Store previous output (overlapped with consumer MatMul)
        store O_prev (in shared_memory) → global_memory  // producer_warp_1
        signal consumer

consumer_warp:
    while has_tile:
        wait producer signal
        FP4MMA QK^T           // consumer only does MatMul
        softmax + quantize P
        FP4MMA PV
        move O to shared_memory  // consumer only moves to SMEM (lightweight)
        signal producer
```
关键区别：consumer 不再直接 store 到 global memory，而是将结果放入 shared memory；producer 在加载下一批数据的同时执行 store。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：在 CUDA C++ 中使用协作组（cooperative groups）或显式 warp 级别同步（`__syncwarp()`）实现。Producer warp_0 和 producer_warp_1 通过 shared memory flag 同步，consumer warp 通过 producer-consumer barrier 同步。Shared memory 用作 producer-consumer 之间的 output buffer。此优化带来的 kernel 加速约 10%（与 reuse shuffle 一起）。适用于所有寄存器压力大、传统 consumer-producer 分工无法容纳完整 pipeline stage 的 warp-specialized kernel。不适用于寄存器充裕的简单 kernel（此时传统方案更简单高效）。

涉及论文标题：
- SageAttention3: Microscaling FP4 Attention for Inference and An Exploration of 8-Bit Training
