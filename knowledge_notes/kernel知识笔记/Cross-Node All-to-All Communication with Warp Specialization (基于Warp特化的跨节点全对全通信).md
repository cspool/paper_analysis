## Cross-Node All-to-All Communication with Warp Specialization (基于Warp特化的跨节点全对全通信)

术语解释
Cross-Node All-to-All Communication with Warp Specialization 是 DeepSeek-V3 为 MoE 训练中 expert dispatch/combine 设计的高效通信 kernel。使用 warp specialization 技术（Bauer et al. 2014），将 20 SMs 划分为 10 个通信通道，每个通道内由不同 warp 分别处理 IB send、IB-to-NVLink forward、NVLink receive（dispatch）或 NVLink send、NVLink-to-IB forward+accumulate、IB receive+accumulate（combine）。定制 PTX 指令 + auto-tuned chunk size 最小化 L2 cache 污染。

术语是什么？
通信 kernel 的设计基于 H800 集群拓扑：节点内 NVLink 160 GB/s ≈ 3.2× IB 50 GB/s。策略：(1) token 先通过 IB 传输到目标节点上同 in-node index 的 GPU，再通过 NVLink 转发到持有目标 expert 的 GPU，IB 和 NVLink 传输完全流水线重叠；(2) 每 token 限制最多 4 个节点（M=4），平均每节点选 3.2 experts，实际 K_r=8，理论上可扩展到 13 experts 而不增加通信开销；(3) 仅 20/132 SMs 即可跑满 IB+NVLink 带宽。

从kernel调度角度拆解术语：
```
=== Cross-Node All-to-All Dispatch Kernel ===

// 20 SMs, 10 communication channels, warp specialization
// 每个 channel 处理一组 token 的 dispatch

Channel[k] (k=0..9):
  // Warp 0: IB Send
  for token in channel_tokens:
    if token_has_remote_experts:
      data = load_FP8_activation(token)            // HBM → registers
      post_IB_send(data, target_node, target_gpu)  // RDMA write to remote

  // Warp 1: IB-to-NVLink Forward (on target node)
  for incoming_ib_token:
    data = IB_recv_buffer → shared_memory
    forward_via_NVLink(data, target_expert_gpu)    // intra-node NVLink

  // Warp 2: NVLink Receive (on target GPU)
  for incoming_nvlink_token:
    data = NVLink_recv_buffer → HBM (expert_input_buffer)

// Dynamic warp allocation: 根据实际 workload 调整各 task 的 warp 数
// L2 cache 优化: PTX ld.global/st.global with cache eviction hints (cg/evict)
// Chunk size auto-tuning: 平衡 throughput vs L2 interference

=== Combine Kernel (reverse direction) ===
Channel[k]:
  // Warp 0: NVLink Send
  expert_output → NVLink send to aggregation GPU within node

  // Warp 1: NVLink-to-IB Forward + FP32 Accumulation
  NVLink_recv → accumulate in shared_memory (FP32) → IB send

  // Warp 2: IB Receive + FP32 Accumulation (on source node)
  IB_recv → FP32 accumulate → HBM (final output)
```

术语一般如何实现？如何使用？
Warp specialization 的关键：每个 warp 独立执行一个通信子任务，通过 shared memory 进行 warp 间数据交换。PTX 指令优化：使用带 cache bypass hint 的 load/store 指令（绕过 L2 cache 避免污染计算 SMs 的数据）。与 computation stream 重叠：dispatch/combine kernel 在独立 CUDA stream 上执行，与 attention/MLP 计算并行。此设计也应用于推理阶段的 all-to-all 通信（prefill 和 decode），但 decode 阶段使用 direct P2P IB（IBGDA）替代 warp specialization pipeline。

涉及论文标题：
- DeepSeek-V3 Technical Report
