## Asynchronous Memory Copy (async-copy)

术语是什么？

Asynchronous Memory Copy (async-copy) 是 NVIDIA Ampere 架构（A100, Compute Capability 8.0）引入的硬件加速异步数据搬移机制，底层 PTX 指令为 `cp.async`（硬件实现为 LDGSTS）。该机制允许 GPU thread 发起 global memory → shared memory 的数据传输后立即继续执行后续指令，数据搬移在后台由专用硬件（类似 SM 内的 DMA engine）异步完成。支持 4/8/16 字节的 copy 粒度，16 字节 copy 绕过 L1 cache（L1 BYPASS 模式）以避免缓存污染。数据搬移完成后需通过 shared memory barrier（`__pipeline_wait_prior` 或 `cuda::barrier`）同步。

从硬件架构角度拆解术语：

async-copy 在 SM 内的工作流程（以 Drawloom FillSMEM 阶段为例）：
1. **发起**：warp 内线程调用 `__pipeline_memcpy_async` 或 PTX `cp.async`，将 sparse matrix A 的 tile 和 column index 从 GMEM 拷贝到 SMEM
2. **流水线提交**：`__pipeline_commit()` 将发起的传输批次提交给 async-copy 硬件单元
3. **线程继续**：warp 不做 stall 等待，直接转入 FillREG 阶段（从 SMEM 加载数据到寄存器）
4. **同步**：在 Comp 阶段前调用 `__pipeline_wait_prior(N)` 确保 N 阶段前的 SMEM 数据已就绪
5. **双缓冲**：设置 delaySMEM=1 实现 double buffering——一个 buffer 被 async-copy 填充的同时，另一 buffer 被计算使用

术语一般如何实现？如何使用？

CUDA 编程通过三种抽象层使用 async-copy：(1) `<cuda/pipeline>` 的 `__pipeline_memcpy_async`+`__pipeline_commit`+`__pipeline_wait_prior`（最低层，保证使用 cp.async PTX）；(2) `<cuda/barrier>` 的 `cuda::memcpy_async`+`cuda::barrier`；(3) `<cooperative_groups/memcpy_async.h>` 的 block 级 collective copy。在 Drawloom 中，Multi-stage Register Pipeline 使用 async-copy 将稀疏矩阵 A（FS）和列索引（FCid）异步拷贝到 SMEM，通过 delaySMEM 参数控制 GMEM→SMEM 流水线与后续 FillREG+Comp 阶段的 overlap——这是解决 DASP 串行 memory-compute 导致的 warp stall 的关键硬件特性。在 RoMeo 中，cross-precision GEMM kernel 使用 cp.async 实现 software pipeline：pipeline fill 阶段发射 N_stage 个异步 copy→steady state 每 iteration 等待 oldest copy 完成→Tensor Core mma 计算→发射新 copy→drain 阶段完成剩余，从而将 GMEM→SMEM 数据传输与 mma 计算重叠。Hopper (H100) 进一步扩展为 Tensor Memory Accelerator (TMA) 支持 bulk tensor transfer。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores
- RoMeo: Mitigating Dual-dimensional Outliers with Rotated Mixed Precision Quantization

---

