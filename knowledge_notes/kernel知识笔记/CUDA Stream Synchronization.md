## CUDA Stream Synchronization

术语是什么？
CUDA Stream Synchronization 是 CUDA 编程模型中的一种重量级同步机制。一个 CUDA stream 是 GPU 操作的序列，同一 stream 上的操作严格按照发射顺序执行。当两个有依赖关系的 CUDA kernel 发射到同一 stream 时，CUDA runtime 保证 consumer kernel 的任何 thread block 都不会在 producer kernel 的所有 thread block 完成之前开始执行。这种同步是隐式的——两个 kernel 之间的 kernel launch boundary 充当了全局 barrier。本文将此称为 stream synchronization。

从kernel调度角度拆解术语：
Stream synchronization 的 kernel 调度伪代码：
```
cudaStream_t stream;
// Both kernels on same stream
producer_kernel<<<grid1, block1, 0, stream>>>(...);  // 所有thread block完成前
consumer_kernel<<<grid2, block2, 0, stream>>>(...);  // 不能开始任何thread block

// GPU SM调度时序（假设80 SM, producer有192 TB, consumer有192 TB）:
// Wave 1: SM[0..79] 执行 producer TB[0..79]    → 80 SM busy
// Wave 2: SM[0..79] 执行 producer TB[80..159]  → 80 SM busy
// Wave 3: SM[0..31] 执行 producer TB[160..191] → 48 SM idle (60% utilization)
// --- stream barrier: consumer不能在此之前开始 ---
// Wave 4: SM[0..79] 执行 consumer TB[0..79]    → 80 SM busy
// Wave 5: SM[0..79] 执行 consumer TB[80..159]  → 80 SM busy
// Wave 6: SM[0..31] 执行 consumer TB[160..191] → 48 SM idle (60% utilization)
// 总计: 6 waves, 平均利用率 = (3*80+3*32)/(6*80) = 70%
```
关键问题：当 thread block 数量不是 SM 数×occupancy 的整数倍时，每个 kernel 的最后一波（partial wave）会产生 SM 空闲。stream synchronization 将这个问题放大——两个依赖 kernel 的 partial wave 串行执行，空闲 SM 无法被另一个 kernel 利用。

术语一般如何实现？如何使用？
CUDA stream synchronization 是 CUDA runtime 的内置行为，无需额外实现。程序员通过将 kernel 发射到同一 stream（默认 stream 0 或显式创建的 stream）来使用。涉及 API：`cudaStreamCreate`、`cudaStreamSynchronize`、`cudaDeviceSynchronize`。典型使用场景：前后有数据依赖的 kernel（如 MLP 的两个 GeMM），PyTorch 等框架默认将同一模型的操作发射到默认 stream。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels
