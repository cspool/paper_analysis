## FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion

- baseline方法是什么？
  - **Baseline 1: NCCL（通用集合通信库）**：PyTorch 等框架默认使用的多 GPU 通信库。在 MoE 数据 shuffling 中，NCCL 要求数据以 device-major layout 排列，而 MoE 模型要求 expert-major layout。这导致每次 all-to-all 通信前后都需要显式的数据重排（permute/repack），使用 `torch.index_select` 等算子（或等价 CUDA kernel）扫描并修改整个 token buffer。
  - **Baseline 2: DeepEP（SOTA MoE 通信库）**：基于 NVSHMEM 的 MoE 专用通信库，使用 warp specialization 和 IBGDA 实现高效的跨节点通信。但其 token deduplication 是局部和静态的，优化与特定硬件（InfiniBand、NVLink、IBGDA）紧密耦合。
  - 全栈执行例子（以 DeepSeek-V3 MoE、EP=64、H100、8-node cluster、seqlen=16k 为例）：
    - **模型推理算法层**：DeepSeek-V3 MoE，top-k=8 routing，256 experts，hidden_dim=7168。Router 为每个 token 选择 top-8 experts，产生 T×8 的 token-expert 分配矩阵。
    - **系统框架层**：Megatron-LM（训练）/ SGLang（推理）使用 NCCL 或 DeepEP 作为通信后端。MoE 层的典型执行流程为：① `index_select(tokens, rank_indices)` 按 destination rank 重排 → ② `all_to_all(permuted_tokens)` 跨设备交换 → ③ `index_select(received, expert_indices)` 按 expert layout 再次重排 → ④ expert FFN 计算 → ⑤-⑥ 对称的反向 permute + all-to-all。步骤①和③是 memory-bound 的 permutation 操作，每次扫描整个 token buffer。
    - **编译框架层**：论文未明确说明。NCCL 和 DeepEP 均直接编译为 CUDA kernel，无中间编译框架。
    - **kernel调度层**：NCCL 使用高度优化的 all-to-all collective kernel。DeepEP 使用 NVSHMEM 的 one-sided put/get 操作 + warp-specialized kernel 实现低延迟通信。但二者均**不感知**数据的 logical segment 结构和 routing 语义——NCCL 的 all-to-all 将数据视为无结构的字节流；DeepEP 的 deduplication 限于特定硬件特性。数据 layout transformation 作为独立的 kernel launch 在通信前后执行。
    - **硬件架构层**：8 节点 × 8×NVIDIA H100 80GB，节点内 NVLink（480 GB/s per GPU），节点间 10×400Gbps RoCE（约 50 GB/s）。
  - **Baseline 痛点**：
    1. **冗余数据复制与重排**（核心痛点）：通信前后各需一次 memory-bound 的 permute/repack。Profiling 显示 rearrangement 占 intra-node 总延迟的 68.8%，占 inter-node 总延迟的 25%。这是因为 all-to-all 要求 device-major layout，而模型需要 expert-major layout，每次通信都产生一对对称的逆排列。
    2. **冗余数据通信**：当同一 token 被路由到同一节点上的多个 expert 时，NCCL 会通过跨节点网络多次发送完全相同的 token payload。DeepEP 有一定 deduplication 但限于局部和静态优化。
    3. **通信负载不均衡**：token routing 的偏斜分布导致各 GPU 跨节点流量不均，产生网络热点和带宽利用不足。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **FUSCO 方法**：基于 "fusing data transformation and communication" 原则，将数据重排嵌入通信操作内部，消除显式的通信前后 permute 步骤。三项核心设计：
    1. **Data-Fused Communication Engine (dComm)**（解决痛点 1）：引入 Segment Descriptor 抽象（{addr, size} 对数组），将 token 建模为 logical segments。发送端 GPU kernel 在 gather 数据到 NIC ring buffer 的过程中 inline 完成 expert-major→device-major 的 layout transformation；接收端 kernel 在 scatter 数据时直接写入 expert activation tensor 的最终位置。使用 pipelined 设计：GPU 准备 slice 与 NIC RDMA 传输完全重叠。
    2. **Communication Planner + Hierarchical Routing**（解决痛点 2）：构建两级 descriptor——Node-Level Forwarding（每个目的节点仅发送一份 token 拷贝给 forwarder GPU）和 Expert-Level Distribution（forwarder 经 intra-node NVLink 分发给各 expert GPU）。这利用节点内高带宽（480 GB/s）替代跨节点重传，显著减少跨节点流量。
    3. **Online Load Balancer**（解决痛点 3）：贪心算法——各节点内按跨节点负载降序排列 GPU，circular shift by node index，构成 communication groups（每组含每节点一个 GPU，组内互为 forwarding endpoints），使高负载 GPU 分散到不同组，利用独立物理通道（多 NIC）并行执行各组通信。

  - 全栈执行例子（与 baseline 同配置，EP=64，H100，8-node，seqlen=16k）：
    - **模型推理算法层**：与 baseline 相同（DeepSeek-V3 MoE，top-k=8，256 experts），不改变模型架构、router 逻辑或收敛性。
    - **系统框架层**：Megatron-LM（训练）和 SGLang（推理）通过约 500 行 Python 适配层替换原有 all-to-all 为 FUSCO。Communication Planner（约 1000 行 Python，使用 PyTorch GPU operators）基于 MoE router 输出构建两级 descriptor plan；dComm（约 2000 行 C++/CUDA）作为独立 collective primitive 执行 fused 通信。**流程从 5 步（permute→all-to-all→permute→compute→reverse）简化为 3 步（FUSCO dispatch→compute→FUSCO combine）**，消除所有通信前后的显式重排。
    - **编译框架层**：论文未明确说明。FUSCO 基于 NCCL transport 层，复用其设备注册和连接管理，无编译框架依赖。
    - **kernel调度层**：dComm 的 GPU kernel 采用 producer-consumer 模式。GPU producer kernel 根据 descriptor 从非连续内存 gather segments 到 contiguous ring buffer（inline layout transformation）。NIC consumer 通过 RDMA 从 ring buffer 发送 slice。由于 slice 的 RDMA 传输时间 > GPU gather 时间，GPU 操作完全被 NIC 掩盖。Intra-node 使用 GPUDirect P2P + inline descriptor 解释。对比 baseline 的 5 次 memory pass（index_select 读+写 ×2 + NCCL 内部拷贝），FUSCO 仅需 1 次 GPU memory pass（descriptor-driven gather to ring buffer）+ 1 次 NIC 传输（pipelined）。
    - **硬件架构层**：与 baseline 相同（8×H100 per node，NVLink + RoCE）。FUSCO 的关键硬件利用策略：① 利用节点内 NVLink 高带宽（480 GB/s vs 跨节点 50 GB/s）做 hierarchical routing 的 expert-level 分发；② 利用多 NIC（10×400Gbps per node）配合 Online Balancer 的 communication groups 实现并行跨节点传输。
    
    关键性能对比（16k seqlen，real-world traffic）：
    - FUSCO vs NCCL: 1.66× communication speedup，训练 1.17-1.39× speedup，推理 TTFT 1.09-1.25× speedup
    - FUSCO vs DeepEP: 1.38× communication speedup，训练 1.10-1.19× speedup，推理 TTFT 1.06-1.16× speedup
    - 消融：dComm 贡献约 27-33%，Planner（含 deduplication）贡献约 27-67%（single-node routed 下最高），Balancer 贡献约 3-17%
