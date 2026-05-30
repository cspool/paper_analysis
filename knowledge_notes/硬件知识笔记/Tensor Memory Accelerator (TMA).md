## Tensor Memory Accelerator (TMA)

术语解释
NVIDIA Hopper架构引入的异步数据搬运单元，用于在GPU全局内存（HBM）和流多处理器（SM）的共享内存之间执行高效的异步批量数据传输。TMA替代了传统CUDA编程中由线程显式执行的内存拷贝指令（如cp.async），将数据搬运负载从CUDA Core转移到专用硬件单元。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TMA（Tensor Memory Accelerator）是NVIDIA H100/H800/H200 GPU（Hopper架构）中的专用固定功能硬件单元，位于每个SM内部。它的核心能力是：(1) 异步执行：单个线程（通常为DMA warp中的线程0）发出TMA指令后，TMA硬件独立于CUDA Core/Tensor Core异步完成数据传输，线程可立即继续执行其他工作；(2) 批量传输：TMA支持burst copy模式，可高效地传输大块数据（tile级，典型大小64-256KB），利用硬件预取和批量事务最大化内存带宽利用率；(3) 多维描述符：TMA使用内存描述符（descriptor）而非线性指针描述传输，支持多维tensor-like传输模式（如2D/3D tile）；(4) Multicast：TMA支持将同一数据同时传输到多个SM的shared memory（用于多播/广播场景如attention）；(5) Completion signaling：TMA传输完成可通过shared memory barriers自动通知consumer，无需consumer轮询或spin-wait；(6) Shared memory barriers：TMA可直接写入shared memory barriers，触发consumer warp的等待条件，实现零延迟producer-consumer同步。

Cypress论文（Yadav et al. 2025）在Figure 1b中展示了Hopper GEMM中TMA的关键角色：DMA warp通过TMA异步加载A/B矩阵tiles到shared memory（TMA_load指令），completion由prod barriers自动触发，compute warpgroup随后消费这些tiles。相比Ampere架构中所有线程参与`copy(tile(gA), sA_next)`的显式数据搬运，TMA将这一负载完全卸载到专用硬件。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TMA在Hopper架构中的运转流程（以GEMM为例，每个SM内部）：

```
1. DMA Warp（32 threads，仅thread 0操作TMA）：
   TMA_load(completion_barrier=prod[k%PIPE],
            src=gA_tile_in_HBM,
            dst=sA[..., k%PIPE],
            descriptor=tile_descriptor_2D)
   
2. TMA硬件独立执行：
   - 读取内存描述符（存储在constant memory或寄存器）
   - 向L2 cache/Memory Controller发出批量读请求
   - 将返回数据写入目标SM的shared memory
   - 复用L2 cache line以最大化带宽
   - 完成后自动arrive(prod[k%PIPE])——硬件写入shared memory barrier

3. Compute Warpgroup（128 threads）：
   wait(prod[k%PIPE])  ← 等待TMA completion barrier
   此时TMA传输已完成，数据在shared memory中就绪
   warpgroup_sync()     ← 128线程对齐
   wgmma(..., sA[:,:,k%PIPE], sB[:,:,k%PIPE])  ← 消费数据
```

TMA的关键硬件特性：
- 硬件端到端完成：TMA从发出请求到数据到达SMEM全由硬件管理，不需CUDA Core参与
- Shared memory barrier integration：TMA的completion event直接写入shared memory barrier（mbarrier），consumer通过wait(mbarrier)高效等待
- Multicast到多个SM：TMA可单次操作将相同tile广播到多个SM的shared memory（用于flash attention等需要同一数据被多个SM消费的场景）
- 带宽利用率：Hopper TMA在仅2KB消息大小时就能达到74%峰值NVLink带宽（vs Copy Engine需256MB）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在CUDA编程中使用TMA：
1. 通过CUDA 12.0+的`cuda::memcpy_async`或PTX `cp.async.bulk`指令
2. CUTLASS 3.x通过CuTe库封装TMA操作（`cute::copy` with TMA backend）
3. 需要预创建TMA描述符（`cudaTensorMap`或`cutlass::TensorMap`），描述传输的张量形状、数据类型、layout
4. TMA描述符常驻于constant memory或寄存器（每个128字节），由host端`cuTensorMapEncodeTiled`创建
5. 在多CTA场景下，TMA需配合`mbarrier`使用以实现completion同步
6. TMA传输需注意shared memory bank对齐以避免bank conflicts

涉及论文标题：
- Task-Based Tensor Computations on Modern GPUs
