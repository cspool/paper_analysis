## Megakernel / Fused Persistent Kernel for Distributed MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Megakernel（Fused Persistent Kernel）是将原本由多个独立 kernel 和 CPU-coordinated collective 组成的复杂分布式计算融合为单个 CUDA kernel 的设计范式。该 kernel 以 persistent 方式运行——一次 launch 后持续执行，在 kernel 内完成全部计算和通信，直到所有工作完成才返回。与传统的 "CPU launch kernel → GPU execute → return → CPU launch next kernel → ..." 模式相比，megakernel 将 CPU 从 control plane 移除，所有调度、同步、通信在 kernel 内由 GPU thread/warp 自主完成。FlashMoE 的 megakernel 融合 MoE layer 全部操作: Gate → Dispatch → Expert FFN (GEMM0+GELU+GEMM1) → Combine → 跨 GPU 通信，仅 1 次 kernel launch vs baseline 33–550 次 (Table 1)。类似工作: Mirage Persistent Kernel (MPK, Dec 2025) 自动将多 GPU 模型推理编译为单一 megakernel。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FlashMoE megakernel 执行: `__global__ void flashmoe_megakernel(A, X, O, L, N)` —— Grid=N blocks/GPU。Phase 1: 所有 block 执行 FusedGate(A) → 得 routing table T_φ 和 affinity scores G_φ。Phase 2: blockId < N-1 → Processor role (while(!interrupt) loop 等待 Scheduler 分配 task)，blockId == N-1 (OS block) 中 warp 0 → Scheduler, warp 1-3 → Subscriber。Processor 内 switch(task.type): GEMM0 (CUTLASS device-side fused GEMM + GELU) → notify → GEMM1 → NVSHMEM put to remote combine buffer。Subscriber poll flags → decode tile → enqueue task → doorbell Scheduler。Scheduler sweep doorbells → WarpInclusiveSum → assign ready tasks to idle Processors。全 done 后 interrupt → kernel return。1 kernel launch vs Megatron-LM+DeepEP 的 432 次。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现关键: (1) 轻量级依赖管理——用 shared memory doorbell 和 global memory flag 替代 CPU-GPU sync；(2) SM occupancy——tile=(128,64), block_size=128, registers=255/thread, max 2 blocks/SM, 0 spill；(3) Device-side BLAS——CUTLASS device-side API 在 persistent kernel 内执行 GEMM；(4) Device-side communication——NVSHMEM kernel 内 API。FlashMoE: 6820 行 CUDA/C++, kernel stack 0B, SMEM 46KB/block, binary 29MB。

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
