## Tutel Adaptive Mixture-of-Experts at Scale

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：(1) **Fast Encode/Decode GPU Kernels（K0, K1, K2）**——将 MoE dispatch 的 encode（生成 All-to-All 输入）和 combine 的 decode（从 All-to-All 输出恢复 MoE 层输出）从稠密 einsum（O(T·E·C_g·D)）替换为稀疏 SIMT-efficient 实现（O(T·k·D)）。基于 CUDA warp-level programming：每个 warp 处理单个 token 沿 M 维度的计算，利用 warp shuffling、Blelloch scan、half2 向量化等优化。(2) **Flexible All-to-All**——将 All-to-All 输出 layout 从 (W, E_g, C_g, D) 优化为 (E_g, C, D)，消除 C_g 对 world size W 的依赖，保证任意规模下 expert matmul 的输入 shape 一致。(3) **2DH (Two-Dimensional Hierarchical) All-to-All**——利用 stride memory copy 对齐非连续 chunks，将小消息聚合为大消息后再进行节点间 All-to-All，避免小消息导致的 InfiniBand 带宽利用不足。支持 MSCCL DSL 编译优化和 LL128 协议。(4) **自适应多流流水线调度**——将 token 沿 capacity 维度分区，在独立 CUDA stream 上异步执行 dispatch All-to-All → Expert FFN → combine All-to-All，实现通信与计算重叠。
  - 实验比较：(a) Kernel computation breakdown: TUTEL (Fast Encode/Decode) vs Fairseq/DeepSpeed MoE 各阶段的 kernel 耗时对比（Figure 15）；(b) Flexible A2A vs A2A layout 的 expert computation throughput（Figure 11）；(c) 2DH vs Linear All-to-All latency 对比（1MiB~256MiB, 64~4096 GPUs），使用 nccl-tests alltoall_perf benchmark（Figures 18, 19）；(d) GPU 内存节省：TUTEL (Fast Encode/Decode) vs Fairseq MoE，20%~90% 节省（Table 5/9）；(e) 自适应流水线在不同 capacity factor f 下的提升（Figure 13）。

- 后端平台是什么，配置是什么。
  - NVIDIA A100 SXM 80GB GPUs（Azure ND96amsr_A100_v4），节点内 NVLink 3.0 + NVSwitch，节点间 HDR InfiniBand (200 Gbps × 8 = 1,600 Gbps non-blocking)。NCCL 2.10.3-1 + NCCL RDMA SHARP plugin。最大 2,048 GPUs。

- 评估性能的软件/脚本是什么。修改了什么。
  - NCCL 2.10.3-1 + nccl-tests（alltoall_perf benchmark）：用于评估 2DH vs Linear All-to-All 延迟。修改：在 nccl-tests 中实现了 2DH All-to-All 算法（Algorithm 2）替代原生 Linear All-to-All。
  - MSCCL (Microsoft Collective Communication Language)：用 DSL 描述 2DH 算法并通过编译器优化生成 NCCL kernel，支持 LL128 协议。
  - PyTorch 1.8.0 + Fairseq moe branch（baseline）+ TUTEL MoE 集成。
  - 修改内容：(1) Fast Encode/Decode：CUDA kernel 实现 K0/K1/K2（Figure 21），替代 Fairseq 中的 einsum dense 实现；(2) Flexible All-to-All：自定义 layout 变换替代标准 NCCL All-to-All 的 (W, E_g, C_g, D) → (E_g, C, D)；(3) 2DH All-to-All：4-phase stride-memcpy + 节点内/节点间 All-to-All；(4) Adaptive Pipelining：token capacity 维度拆分 + 多 CUDA stream 异步调度。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/microsoft/tutel，集成于 Fairseq 和 DeepSpeed。
  - Fast Encode Kernel（K0/K1/K2）输入到输出全过程：
    1. **输入**：moe_input (T, M)，logits (T, E) — T 个 token 的隐藏特征和 gate logits
    2. **K0 (Gate Processing)**：Softmax(logits) → gate_probs；TopK(gate_probs) → idxs (T, k), scores (T, k)；compute_location(idxs) → locations (T, k)。每个 warp 处理一个 token，warp-level 并行。
    3. **K1 (Sparse Encode/Decode)**：对每个 token t，执行 `dispatch_input[idxs[t]][locations[t]] = bool(scores[t]) * moe_input[t]`。使用 warp shuffling + Blelloch scan 实现稀疏写入，复杂度从 O(T·E·C_g·D) 降至 O(T·k·D)。
    4. **K2 (All-to-All 输入/输出变换)**：Flexible All-to-All layout 变换 (E, C_g, D) → (E_g, C, D)，inline 完成无需额外 memory copy。
    5. **输出**：dispatch_input (E, C_g, M) → All-to-All dispatch → Expert FFN (matmul) → All-to-All combine → decode → moe_output (T, M)。
  - 2DH All-to-All 评估原理：
    1. **输入**：S bytes 数据，分布于 n GPUs（m GPUs/node, n/m nodes）
    2. **Phase 1 (Stride Memcpy)**：重排节点内 chunks 使同一本地目标 GPU 的数据连续
    3. **Phase 2 (Intra-node A2A)**：节点内 m GPUs 交换 S/m 大小的 chunks
    4. **Phase 3 (Stride Memcpy)**：重排使同一远程目标节点的数据连续
    5. **Phase 4 (Inter-node A2A)**：节点间 n/m nodes 交换合并后的大 chunks（S/n × m）
    6. **输出**：All-to-All 完成后的重排数据。通过 nccl-tests alltoall_perf 测量端到端延迟，对比 Linear（Algorithm 1）vs 2DH（Algorithm 2）。
