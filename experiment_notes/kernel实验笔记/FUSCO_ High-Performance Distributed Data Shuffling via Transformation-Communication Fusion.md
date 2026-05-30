## FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - FUSCO 的核心 kernel 调度组件是 **Data-Fused Communication Engine (dComm)**，约 2000 行 C++/CUDA 实现。dComm 通过以下机制实现 runtime 级别的 fused data+communication：
    1. **Segment Descriptor 驱动的 Gather/Scatter Kernel**：发送端 GPU kernel 读取 descriptor 数组（连续存放的 {addr, size} 对），根据累计已传输字节数定位当前 segment，从非连续内存中 gather 数据到 NIC ring buffer，在此过程中完成 layout transformation（将 expert-major layout 转换为 device-major layout），无需额外的 permute kernel。
    2. **Pipelined Slice 传输**：将多个 logical segments 打包为 slice（远大于单个 segment），GPU producer kernel 将 slice 写入 ring buffer，NIC consumer（RDMA）从 ring buffer 读取并发送。由于 RDMA 传输时间通常超过 GPU slice 准备时间，GPU memory copy 和 NIC 传输完全重叠。
    3. **GPUDirect P2P 节点内 Kernel**：对于 intra-node 传输，dComm 使用 GPUDirect P2P 直接 GPU-to-GPU copy，在 copy 路径中集成 descriptor 解释逻辑，inline 完成 layout transformation。
  - 实验比较 FUSCO 与 NCCL（使用 `index_select` 等 PyTorch 算子做显式重排+通信）和 DeepEP（基于 NVSHMEM 的 warp-specialized kernel + IBGDA）在三种流量模式下的通信微基准性能，以及 per-component 消融实验。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA H100 80GB HBM3（每节点 8 张，共 64 张）
  - 节点内互联：NVLink，每 GPU 18 条 link，理论聚合带宽约 480 GB/s per GPU
  - 节点间互联：Mellanox ConnectX-7 400 Gbps NIC × 10 per node（RoCE），理论跨节点带宽约 50 GB/s
  - CUDA 12.9，NCCL 2.26.3
  - 通信微基准参数：EP=64，hidden_dim=7168，top-k=8，num_experts=256（与 DeepSeek-V3 一致）

- 评估性能的软件/脚本是什么。修改了什么。
  - **通信微基准测试**：自建 benchmark 将 MoE 通信分为三阶段测量——preprocessing（路由结果转换为通信调度）、rearrangement（token 重排以对齐通信或 expert layout）、communication（all-to-all dispatch + combine）。
  - **宏基准测试**：
    - 训练：Megatron-LM，per-iteration training time
    - 推理：SGLang + prefill-decode disaggregation，time-to-first-token (TTFT)
  - **消融实验**：分别禁用 dComm（回退到 NCCL + 显式重排）、Planner（回退到默认 all-to-all，每个 token 独立发送）、Balancer（回退到同 index 的静态分组），测量性能退化。
  - **修改内容**：FUSCO 的 dComm runtime 作为独立 collective primitive（类似 send/recv/allgather）暴露，通过扩展的 PyTorch distributed backend 调用。Communication Planner 和 Online Balancer 约 1000 行 Python，使用 PyTorch GPU operators（sum, argsort, gather, scatter）构建 descriptor。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源情况**：论文声明 "Our code and data will be made publicly available"，截至分析时未找到公开代码仓库。FUSCO 基于 NCCL（https://developer.nvidia.com/nccl）和 PyTorch（https://pytorch.org）构建。
  - **dComm Kernel 执行原理全过程**：
    ```
    ┌── Kernel Input ──────────────────────────────────────┐
    │ descriptor_list = [                                    │
    │   {addr: 0x7f00, size: 8192},   // token₀ 在 GPU mem │
    │   {addr: 0x8a00, size: 8192},   // token₁              │
    │   {addr: 0x9100, size: 14336},  // token₂ (大 token)   │
    │   ...                                                 │
    │ ]  // 连续存放在 GPU global memory                      │
    │ ring_buffer[NIC_RING_SIZE]  // NIC 可见的环形缓冲区     │
    │ total_bytes_to_send = sum(descriptor_list[i].size)     │
    └───────────────────────────────────────────────────────┘
    
    ┌── GPU Producer Kernel (每个 slice 一次 launch) ──────┐
    │ slice_size = max_slice_bytes (如 1MB)                  │
    │ slice_start = slice_id * slice_size                    │
    │                                                        │
    │ // 定位当前 segment                                    │
    │ cumsum = 0                                             │
    │ for desc in descriptor_list:                           │
    │     if cumsum + desc.size > slice_start:               │
    │         // 当前 segment 跨越 slice 边界                │
    │         offset_in_seg = slice_start - cumsum           │
    │         bytes_to_copy = min(                           │
    │             desc.size - offset_in_seg,                 │
    │             slice_size - bytes_copied                  │
    │         )                                              │
    │         cudaMemcpyAsync(                               │
    │             ring_buffer + bytes_copied,                │
    │             desc.addr + offset_in_seg,                 │
    │             bytes_to_copy,                             │
    │             cudaMemcpyDeviceToDevice                   │
    │         )                                              │
    │         // ↑ 此 copy 完成了 layout transformation:     │
    │         //   从 expert-major 非连续布局 gather 到       │
    │         //   连续的 device-major ring buffer            │
    │         bytes_copied += bytes_to_copy                  │
    │         if bytes_copied >= slice_size: break           │
    │     cumsum += desc.size                                │
    │                                                        │
    │ // Signal NIC: slice 已就绪                             │
    │ __threadfence_system()                                │
    │ *slice_ready_flag = 1                                  │
    └───────────────────────────────────────────────────────┘
              │
              │ NIC 读 ring_buffer，RDMA Write 到远端 GPU
              │ (GPU 继续处理下一个 slice)
              ▼
    ┌── Receiver GPU Kernel ───────────────────────────────┐
    │ // 镜像逻辑：接收端 descriptor 数组指定 scatter 目标    │
    │ for desc in receiver_descriptor_list:                 │
    │     // desc.addr = 该 segment 应在 expert activation   │
    │     //              tensor 中的最终位置                │
    │     cudaMemcpyAsync(                                   │
    │         desc.addr,  // 直接写入 expert 计算所需的 layout│
    │         recv_buffer + bytes_received,                 │
    │         desc.size,                                     │
    │         cudaMemcpyDeviceToDevice                       │
    │     )                                                  │
    │     bytes_received += desc.size                        │
    └───────────────────────────────────────────────────────┘
    ```

    **NCCL Baseline 的 kernel 执行过程（对比）**：
    ```
    // Baseline (NCCL + PyTorch): 3 步，5 次 memory pass
    // Step 1: 显式重排 (2 memory passes: read + write)
    permuted = torch.index_select(tokens, dim=0, index=rank_indices)  
    
    // Step 2: All-to-all 通信 (implicit memory pass via NCCL)
    exchanged = nccl_all_to_all(permuted)
    
    // Step 3: 再次重排 (2 memory passes: read + write)
    expert_input = torch.index_select(exchanged, dim=0, index=expert_indices)
    
    // 总计: 5 次 memory pass（含重排）+ 1 次网络通信
    // FUSCO: 1 次 memory pass（gather→ring buffer）+ 1 次网络通信（pipelined）
    ```

    **Pipelined 时序图**：
    ```
    Time →
    GPU: |== Slice₀ Gather ==|== Slice₁ Gather ==|== Slice₂ Gather ==|
    NIC:                      |== RDMA Slice₀ ===|== RDMA Slice₁ ===|
    ```
    GPU gather 时间 < NIC RDMA 时间，因此 GPU 操作完全隐藏在通信延迟中。

    **三种流量模式下的 kernel 性能数据**（16k seqlen，单位 ms，FUSCO vs NCCL vs DeepEP）：
    - Real-world traffic: 86.84 vs 144.30 vs 119.48（1.66× / 1.38× speedup）
    - Single-node routed: 40.99 vs 157.28 vs 82.17（3.84× / 2.01× speedup）
    - Load-imbalanced: 151.30 vs 338.99 vs 213.74（2.24× / 1.42× speedup）

    **消融实验**（Table 3，16k seqlen，real-world traffic，单位 ms）：
    | Configuration | Latency | Degradation |
    |--------------|---------|-------------|
    | FUSCO (full) | 86.84 | - |
    | dComm-off | 119.48 | -27.32% |
    | Planner-off | 124.35 | -30.17% |
    | Balancer-off | 95.13 | -8.72% |
