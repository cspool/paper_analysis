## CUDA Graph for LLM Inference

术语解释
CUDA Graph 是 NVIDIA CUDA 提供的机制，将一系列 CUDA kernel launches 和 memory operations 预录制为一个 graph，后续通过单次 launch 重放，消除重复 kernel launch overhead。在 LLM 推理中，decode 阶段每 step 执行相同的 kernel 序列（仅数据不同），非常适合 CUDA Graph 优化。

术语是什么？
传统 CUDA 编程中，每个 kernel launch 都需 CPU 向 GPU 提交 work，kernel launch overhead 在 decode 阶段（每 step 的计算量非常小）占比显著。CUDA Graph 将一次 decode step 的全过程——attention kernel、top-k、all-to-all dispatch、expert FFN kernels、all-to-all combine 等——录制为一个 graph，后续 decode steps 通过 `cudaGraphLaunch()` 单次提交整个 graph，大幅减少 CPU-GPU 同步和 kernel launch 开销。vLLM 的 compilation framework 支持将 decode phase 编译为 CUDA Graphs。

从kernel调度角度拆解术语：
CUDA Graph 在 METRO vLLM 集成中的应用：

```
=== vLLM CUDA Graph Compilation for Decode ===

编译阶段 (one-time):
  for batch_size in [1, 2, 4, 8, 16, 32]:  // power-of-two
    cudaStreamBeginCapture(stream)
    // 录制以下 kernel 序列:
    Attention_kernel(batch_size, ...)
    AllGather_kernel(tokens)
    TopK_kernel(all_tokens)
    METRO_Routing_kernel(N, G, A, T)    // 单 SM, Algorithm 1
    for each activated expert:
      FFN_GEMM_kernel(expert_weight, tokens)
    AllToAll_Combine_kernel(outputs)
    cudaStreamEndCapture(stream, &graph)
    cudaGraphInstantiate(&graph_exec[batch_size], graph)
    
  // 存储 graph_exec[1,2,4,8,16,32] 供运行时使用

运行时 Decode (each step):
  batch_size = min(next_power_of_two(num_ready_tokens), 32)
  if num_ready_tokens != power_of_two:
    pad_tokens_to(batch_size)  // padding to reuse graph
  update_input_pointers(graph_exec[batch_size])
  cudaGraphLaunch(graph_exec[batch_size], stream)
  // 单次 launch 执行所有 kernel，无中间 CPU-GPU 同步
```

术语一般如何实现？如何使用？
- CUDA Graph 适用于 kernel 序列固定、仅输入数据变化的工作负载——LLM decode 是典型场景
- 限制：(a) 图结构编译后不可变——需要预编译多个 batch size 版本；(b) 不支持动态控制流（kernel 内部的分支可以，但 kernel 选择和 kernel 数量不可变）；(c) 内存地址在录制时固定——需在 relaunch 前更新指针
- vLLM 的 CUDA Graph 集成：预编译 power-of-two batch sizes 的 graph，非 power-of-two 通过 padding 复用最近更大的 graph
- CUDA Graph 不能替代所有 kernel launch overhead——仍存在首次 graph launch 的初始化开销
- METRO 将 routing kernel 嵌入 CUDA Graph 后，Decode 阶段无额外的 kernel launch overhead

涉及论文标题：
- Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens
