## MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoESys 在 kernel/运行时层面做了三项优化：（1）**Hierarchical AlltoAll Communication**——针对 MoE expert parallelism 中的 AlltoAll 通信，利用网络拓扑层次（intra-node NVSwitch + inter-node NIC/switch），先 intra-node AlltoAll 通过 NVSwitch 收集数据，再按相同 rank 的 GPU 分组做 inter-node AlltoAll，避免跨不同 rank 的 NIC 通道产生交换机路由冲突；（2）**Custom H2D/D2H Kernels**——使用 CUDA Pinned Memory 优化 Host-to-Device/Device-to-Host 数据传输，减少层间传输延迟；（3）**Fused Multi-head Attention Kernel**——来自 NVIDIA BERT MLPerf 1.1 实现的 Fused MHA kernel，减少 kernel launch 次数。
  - 实验比较：（1）Hierarchical AlltoAll vs baseline AlltoAll 的通信时间占比和 end-to-end training time breakdown（FWD/BWD/OPT/Comm），在 2 nodes(16 GPUs) 和 4 nodes(32 GPUs) 下对比不同参数规模的 MoE 模型；（2）整体 training throughput（tokens/s）提升；（3）Cross-wise comparison 中各 kernel/通信优化的 peak memory 和 computation speed 贡献。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA A100 80GB，单节点 8 GPU 通过 NVSwitch 互联（NVLink 900 GB/s），节点间通过 NIC（100G Mellanox ConnectX 系列）+ leaf/spine switch 互联。
  - 网络拓扑：m 个 cluster，每个 cluster 含 p 个 GPU node。Leaf switch 按 rank 分组，同一 rank 的 NIC 直连同一 leaf switch，跨 rank 需经过 spine switch（带宽低于 leaf switch）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 框架：PaddlePaddle / PaddleFleetX 分布式训练框架。
  - 修改的通信原语：将标准 AlltoAll 替换为 Hierarchical AlltoAll——在 NCCL 或 PaddlePaddle 通信层实现两阶段 AlltoAll（intra-node via NVSwitch → inter-node via NIC grouped by rank）。
  - CUDA kernel 修改：H2D/D2H 使用 cudaHostAlloc 分配 pinned memory + 异步 cudaMemcpyAsync；Fused MHA 集成自 NVIDIA MLPerf BERT 实现。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - MoESys 基于开源 PaddlePaddle/PaddleFleetX。论文称代码将发布于 PaddlePaddle GitHub，截至搜索未找到独立 MoESys 仓库。
  - Hierarchical AlltoAll 的执行原理与全流程：
    1. **输入**：MoE Gate 网络已确定每个 token 的 expert 分配，每个 GPU 需要将 tokens 发送到对应 expert 所在的 GPU。
    2. **阶段一——Intra-node AlltoAll via NVSwitch**：
       - 8 GPU per node，每个 GPU 通过 NVSwitch 以 900 GB/s 带宽做全交换。
       - 目标：将跨 rank 的数据先在本节点内通过 NVLink 搬运到对应 rank 的 GPU。
       - 例如 GPU0 (rank 0) → GPU7 (rank 7) 的数据：通过 NVLink 从 GPU0 搬到 GPU7。
    3. **阶段二——Inter-node AlltoAll via NIC (grouped by rank)**：
       - 按 rank 分组——所有 node 的 GPU0 (rank 0) 组成通信组，通过 NIC + leaf switch（不走 spine switch）做 AlltoAll。
       - 所有 node 的 GPU7 (rank 7) 同理。
       - 优势：同 rank 的 NIC 接入同一 leaf switch，不经过 spine switch（低带宽瓶颈），避免跨 rank 通信的 spine switch 路由开销。
    4. **对比 Baseline AlltoAll**（如 GPU0 of Node1 in Cluster A ↔ GPU7 of Node2 in Cluster B）：
       - Baseline 路径：NIC1 → LE1 → SPq → LE1 → NICn（经过 spine switch，高延迟 + 带宽竞争）
       - Hierarchical 路径：GPU0 → NVLink → GPU7（intra-node）→ NICn → NICn（inter-node，同一 leaf switch）
    5. **性能输出**：通信时间占比显著下降。以 80.7B MoE model / 4 nodes 32 GPUs 为例，end-to-end training 性能提升 10.3%，通信阶段 speedup 15.5%。peer-to-peer 通信效率提升 p 倍（p=单节点 GPU 数）。
  - Custom H2D/D2H Kernel 原理：
    1. 使用 `cudaHostAlloc` 分配 pinned (page-locked) host memory，避免默认 pageable memory 的额外 copy。
    2. `cudaMemcpyAsync` 在独立 CUDA stream 上异步执行 H2D/D2H transfer，与 GPU computation kernel 在 default stream 上的执行重叠。
    3. 在 MoE layer 切换时，将下一层 expert 参数从 CPU pinned memory 异步传输到 GPU global memory，同时 GPU 执行当前 layer 的计算。
