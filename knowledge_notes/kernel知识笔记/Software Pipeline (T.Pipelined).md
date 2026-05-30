## Software Pipeline (T.Pipelined)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Software Pipeline（软件流水线）是 TileLang 中的 T.Pipelined 调度原语，用于将循环体内的数据搬移（如 global→shared memory copy）与计算（如 GEMM/attention）重叠执行，通过异步硬件机制隐藏内存延迟。TileLang 的 Pipeline 机制自动分析循环体内语句的 buffer 使用依赖关系，生成结构化的 interleaved schedule：将前后相邻 iteration 的 Copy 和 GEMM 交错执行（Copy(i+1) 与 GEMM(i) 重叠）。在不同 GPU 架构上自动选择最优硬件路径：(1) Ampere (A100) — cp.async 异步 global→shared copy + cp.async.commit_group + cp.async.wait_group；(2) Hopper (H100) — TMA（Tensor Memory Accelerator）硬件单元 + wgmma.mma_async + warp specialization + mbarrier 同步；(3) AMD CDNA — s_waitcnt lgkmcnt + buffer_load_dword lds 指令。

从 kernel 调度角度拆解术语，比如术语所在 kernel 调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

T.Pipelined 在不同架构上的调度伪代码：

```
// === Ampere (A100) Pipeline (cp.async) ===
// T.Pipelined(K // block_K, num_stages=2) 自动展开为:

// Prologue: 预取前 2 个 stage
cp.async(A[0], A_shared[0])       // stage 0 load A
cp.async(B[0], B_shared[0])       // stage 0 load B
cp.async.commit_group()
cp.async(A[1], A_shared[1])       // stage 1 load A
cp.async(B[1], B_shared[1])       // stage 1 load B
cp.async.commit_group()

// Main Loop (steady state):
for k in range(2, K//block_K):
  cp.async.wait_group<0>()        // 等待 stage (k-2) 完成
  __syncthreads()
  gemm(A_shared[(k-2)%2], B_shared[(k-2)%2], C_local)  // compute stage (k-2)
  cp.async(A[k], A_shared[k%2])   // async load stage k (overlapped)
  cp.async(B[k], B_shared[k%2])
  cp.async.commit_group()

// Epilogue: 完成最后 2 个 stage 的计算
cp.async.wait_group<0>(); __syncthreads()
gemm(A_shared[(K-2)%2], B_shared[(K-2)%2], C_local)
cp.async.wait_group<0>(); __syncthreads()
gemm(A_shared[(K-1)%2], B_shared[(K-1)%2], C_local)


// === Hopper (H100) Pipeline (TMA + Warp Specialization) ===
// T.Pipelined 自动推导 warp specialization:

// Producer Threads (by threadIdx.x):
for k in range(K // block_K):
  cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier(
    A_shared[k%2], &desc_A, [by*block_M, k*block_K], mbar_prod)
  cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier(
    B_shared[k%2], &desc_B, [k*block_K, bx*block_N], mbar_prod)
  mbarrier.arrive(mbar_prod)     // signal data ready

// Consumer Threads (by threadIdx.x):
for k in range(K // block_K):
  mbarrier.try_wait(mbar_prod)   // wait for producer
  wgmma.fence()
  wgmma.commit_group()
  wgmma.mma_async(A_shared[k%2], B_shared[k%2], C_local)
  wgmma.wait_group<0>()
  // ... producer 已在加载下一 tile (overlapped)


// === AMD CDNA Pipeline ===
s_waitcnt lgkmcnt(0)             // wait for LDS writes
buffer_load_dword lds, ...       // async global→LDS load
s_waitcnt lgkmcnt(0)
// compute ...
```

关键：T.Pipelined 比 Triton 的 num_stages 提供更灵活的 pipeline 控制 — 用户可通过显式标注 producer/consumer 顺序实现自定义 pipeline pattern（如 FlashAttention-3 级别的复杂 warp specialization pipeline）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 TileLang 中，用户只需在循环上添加 `T.Pipelined(K // block_K, num_stages=N)` annotation。编译器自动：(1) 分析 loop body 中各 buffer 的读/写关系 → 确定 prod/cons 角色；(2) 插入对应架构的异步指令序列；(3) Live Variable Analysis → 确定同步点 → 插入 barrier；(4) Hopper 架构上自动应用 warp specialization。对于专家用户，可通过显式 pipeline API（论文未详述）自定义同步策略。

涉及论文标题：
- TileLang: A Composable Tiled Programming Model for AI Systems

---
