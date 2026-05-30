## FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion

- 属于Serving调度的实现是什么？实验比较什么？
  - FUSCO 实现了一个 MoE-friendly 通信库，通过融合数据变换（data transformation）与通信（communication），替代框架中原有的 NCCL all-to-all 操作。核心实现包括三部分：
    1. **Data-Fused Communication Engine (dComm)**：引入 Segment Descriptor 抽象，将 MoE token 建模为逻辑 segments，在通信路径上直接完成数据重排（rearrangement），消除通信前后的显式 permute/repack 步骤。
    2. **Communication Planner**：基于 MoE router 的 token-expert 分配结果，构建两级 descriptor（Node-Level Forwarding + Expert-Level Distribution），实现层次化路由（hierarchical routing），在目的节点指定一个 forwarder GPU 接收跨节点数据后经 intra-node 链路分发，消除重复跨节点传输（token deduplication）。
    3. **Online Load Balancer**：将各节点的 GPU 按负载排序后贪心分组为 communication groups，每组包含每个节点的一个 GPU，组内 GPU 互为 forwarding endpoints，通过 circular shift 使高负载 GPU 分布到不同组，缓解跨节点流量倾斜。
  - 实验比较 FUSCO 与 NCCL（通用集合通信库）和 DeepEP（SOTA MoE 通信库，基于 NVSHMEM）在三种流量模式下的通信延迟，以及在 Megatron-LM 训练和 SGLang 推理上的端到端性能。

- 硬件平台是什么，配置是什么。
  - 8 节点集群，每节点配置：
    - CPU：2x Intel Xeon Platinum 8558（48 核/socket，192 线程/节点）
    - GPU：8x NVIDIA H100 80GB HBM3
    - 节点内互联：NVLink（每 GPU 18 条 NVLink link，理论聚合带宽约 480 GB/s per GPU）
    - 节点间互联：10x 400 Gbps Mellanox ConnectX-7 NIC（RoCE）
    - NIC-GPU 互联：PCIe 桥接
  - 软件环境：Linux kernel 5.15.0, Ubuntu 24.04, NVIDIA driver 535.183.06, CUDA 12.9, NCCL 2.26.3, PyTorch 2.7.0

- 开源Serving框架是什么。修改了什么。
  - **训练框架**：Megatron-LM —— 将 FUSCO 替换 Megatron-LM 中 MoE 层的 all-to-all 操作，通过扩展的 PyTorch distributed backend 调用 dComm primitive。
  - **推理框架**：SGLang —— 使用 prefill-decode disaggregation 配置，在 MoE 模型的 prefill 阶段用 FUSCO 替换 all-to-all 操作。
  - **修改内容**：约 500 行 Python 适配层（thin adaptation layer），桥接框架的 token-routing 路径与 FUSCO 的 planner 和 dComm primitive，无需修改模型逻辑或 expert kernel。FUSCO 本身在 NCCL transport layer 之上实现（约 2000 行 C++/CUDA），复用 NCCL 的设备注册、连接管理和 transport 层。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - **开源情况**：论文声明 "Our code and data will be made publicly available"，截至分析时未在 web search 中发现公开代码仓库。FUSCO 基于 NCCL（https://developer.nvidia.com/nccl）构建，复用其 transport 层。
  - **Serving框架执行全过程（以 SGLang 推理为例，EP=64，seqlen=16k）**：
    ```
    ┌─────────────────────────────────────────────────────┐
    │ 1. SGLang Prefill 阶段                              │
    │    输入：用户 prompt tokens [T₁, T₂, ..., T₁₆ₖ]     │
    │    MoE Router 计算 token→expert 分配 (top-k=8)       │
    │           ↓                                          │
    │ 2. FUSCO Communication Planner                      │
    │    读取 token-expert 矩阵 A (T×K)                      │
    │    构建两级 descriptor：                               │
    │      - Node-Level: 每个 destination node 仅一份拷贝   │
    │      - Expert-Level: node 内各 GPU→expert 的精确偏移  │
    │    Online Balancer 按 greedy circular-shift 分组      │
    │           ↓                                          │
    │ 3. FUSCO dComm Engine 执行                           │
    │    ┌─ Sender GPU ─────────────────────────────┐      │
    │    │ Slice₀: [desc→gather segments→ring buf]  │      │
    │    │ Slice₁: [desc→gather segments→ring buf]  │      │
    │    │ ...   ← GPU memory copy + layout transform│     │
    │    └──────────────────────────────────────────┘      │
    │              ↓ RDMA (RoCE, 400Gbps)                   │
    │    ┌─ Receiver (Forwarder) GPU ───────────────┐      │
    │    │ desc→scatter to receive buffer            │      │
    │    │ NVLink P2P → distribute to expert GPUs    │      │
    │    └──────────────────────────────────────────┘      │
    │           ↓                                          │
    │ 4. Expert FFN Computation                           │
    │    各 GPU 上的 expert 直接消费已排列好的 token buffer  │
    │           ↓                                          │
    │ 5. dComm 反向 (Combine)                              │
    │    对称的 descriptor 驱动的 gather+all-to-all         │
    │           ↓                                          │
    │ 6. SGLang 继续 decode                                │
    │    输出：first-token generation (TTFT)               │
    └─────────────────────────────────────────────────────┘
    ```
    
    **训练全过程（Megatron-LM，EP=64）**：
    ```
    Forward:
    Input tokens → Attention → MoE Gate (top-k routing)
      → FUSCO dispatch (dComm: segment descriptor → pipelined GPU-to-ringbuf→RDMA)
      → Expert FFN (各GPU直接计算，无额外重排)
      → FUSCO combine (dComm: 反向 descriptor 驱动)
      → Output
    
    Backward:
    Gradient → FUSCO combine (反向 dispatch) → Expert backward
      → FUSCO dispatch (反向 combine) → Attention backward
    ```
    
    FUSCO 在框架中作为 `send/recv/allgather` 级别的 primitive 暴露，调用方式类似：
    ```python
    # 伪代码：在 Megatron-LM MoE 层中使用 FUSCO
    # 传统 NCCL 方式:
    # tokens = permute(tokens, routing_indices)    # 显式重排
    # tokens = all_to_all(tokens)                   # 通信
    # tokens = permute(tokens, expert_indices)      # 再次重排
    # expert_output = expert_ffn(tokens)
    # ... 对称的反向操作
    
    # FUSCO 方式:
    descriptors = fusco_planner.build_plan(token_expert_matrix)
    tokens = fusco_dcomm.dispatch_with_fusion(tokens, descriptors)  # 一步完成
    expert_output = expert_ffn(tokens)
    tokens = fusco_dcomm.combine_with_fusion(expert_output, descriptors)
    ```
