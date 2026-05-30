## CUDA Graph in MoE Serving（MoE服务中的CUDA图）

术语是什么？
CUDA Graph 是 NVIDIA CUDA 12.x 提供的 GPU 工作流图机制，将一系列 CUDA kernel launch 预先录制为静态计算图并重放执行。在 LLM serving 中，decoding phase 的 kernel 序列固定（每个 token 执行相同操作），CUDA Graph 消除了 CPU-side kernel launch overhead，实现 kernel 级 back-to-back 执行。PROBE 面临的挑战：动态 expert replication 和 P2P 传输产生变量控制流，与 CUDA Graph 的静态图要求冲突。

从kernel调度角度拆解术语：
```
// CUDA Graph 录制（静态配置）：
cudaStreamBeginCapture(stream)
  // 录制一系列 kernel launch:
  router_kernel<<<...>>>(...)
  alltoall_dispatch_kernel<<<...>>>(...)
  grouped_gemm_kernel<<<...>>>(...)
  alltoall_combine_kernel<<<...>>>(...)
cudaStreamEndCapture(stream, &graph)

// 推理时重放（避免 per-kernel launch overhead）：
cudaGraphLaunch(graph_exec, stream)  // 单次调用执行全部 kernel
```
PROBE 解决 Graph 兼容性的策略：(1) Planner 运行在 GPU 上（单 SM kernel），消除 host-device sync；(2) Prefetch 传输通过 CUDA event 控制，不引入 host-side 条件分支；(3) Expert slot 通过双缓冲管理，地址在 graph capture 时固定；(4) 动态 routing assignment 通过预分配的 device buffer 传递。

术语一般如何实现？如何使用？
vLLM 和 SGLang 在 decoding phase 使用 CUDA Graph（prefill 因 batch size 变化不使用）。通过 `cuda.graph()` 上下文管理器录制，`graph.replay()` 重放。PROBE 的 planner 和 prefetch 在 graph capture 前预先录制为可重放的 sub-graph，运行时通过 CUDA graph update 或 node-level 参数更新机制适应动态配置。

涉及论文标题：
- PROBE: Co-Balancing Computation and Communication in MoE Inference via Real-Time Predictive Prefetching
