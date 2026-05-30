## Double Buffering for CPU-GPU KV Cache Transfer (CPU-GPU KV Cache 传输的双缓冲)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Double Buffering for CPU-GPU KV Cache Transfer 是 TailorKV 中用于隐藏 PCIe 数据传输延迟的系统技术。GPU 上维护两个相同大小的 key cache buffer（读 buffer + 写 buffer），在 decoding 阶段交替使用：layer l-1 计算时，异步将 layer l 的 critical key cache 从 CPU prefetch 到写 buffer；layer l 开始计算时，切换读写 buffer（读 buffer 变为下一层的写 buffer），从读 buffer 读取已预取完成的 data。该设计与异步 CUDA stream 配合，使 GPU 计算（layer l-1 的 attention/FFN）与 CPU→GPU 数据传输（layer l 的 critical key prefetch）完全重叠。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// Double Buffering 生命周期（两个相邻 layer）

// === 初始化 ===
Buffer A = cudaMalloc(critical_key_size)  // 读 buffer
Buffer B = cudaMalloc(critical_key_size)  // 写 buffer

// === Layer l-1 执行期间 ===
// Stream 0 (Compute):  layer l-1 attention + FFN
// Stream 1 (Transfer): cudaMemcpyAsync(B, CPU_K_critical[l], H2D, stream1)
// Wait: layer l-1 完成后 Stream0 等待 Stream1 完成（B 就绪）

// === Layer l 执行期间（B 已就绪，A 被释放）===
swap(A, B)                                   // A 变为写 buffer, B 变为读 buffer
// Stream 0 (Compute):  layer l attention, 从 B 读取 K_critical
// Stream 1 (Transfer): cudaMemcpyAsync(A, CPU_K_critical[l+1], H2D, stream1)
// ...

// Timeline（Figure 5 的时间线图示）:
// Layer l-1:  |── Compute ──|── Wait ──|
//                   ↕ overlap
// Prefetch l:      |── PCIe H2D ──|
// Layer l:                        |── Compute (use prefetched) ──|── Wait ──|
//                                        ↕ overlap
// Prefetch l+1:                          |── PCIe H2D ──|
```

术语一般如何实现？如何使用？

实现依赖：(1) `cudaMemcpyAsync` 配合 non-default CUDA stream 实现异步传输（源必须是 pinned memory / `cudaMallocHost` 分配）；(2) 两个 `cudaMalloc` 分配的 GPU buffer，通过简单的指针交换（`std::swap`）实现读/写角色切换；(3) `cudaEventRecord` + `cudaStreamWaitEvent` 用于跨 stream 同步——compute stream 在开始当前层 attention 前等待 transfer stream 完成。

TailorKV 中 double buffering 仅用于 critical key cache 传输（d_s 个 channel × n 个 token，数据量小），Top-K 完整 token 的 fetch 不使用 double buffering（因需要当前层 query 才能确定哪些 token 需要 fetch，无法提前 prefetch）。这个不可 overlap 的 Top-K fetch 是 TailorKV pipeline 中唯一的串行瓶颈（Figure 5）。

涉及论文标题：
- TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization
