## HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

- baseline方法是什么？
  - **Megatron-LM 标准 AlltoAll（HD1-AlltoAll）**：MoE 训练中，expert parallelism 要求通过 AlltoAll collective 将 token dispatch 到对应 expert 所在的 GPU，计算完成后再 AlltoAll combine 回来。标准 AlltoAll 不利用 GPU 集群的分层拓扑结构（Node/IB → QPI → NVLink → Intra-GPU），所有 GPU 之间平等通信，导致低带宽链路（如 InfiniBand inter-node、QPI）成为瓶颈。在 MoE 场景下，由于 E/G > 1（每 GPU 持有多个 expert），同一 GPU 上的多个 expert 可能被同一 token 选中（top-K），导致 token 在 AlltoAll 中被重复传输（重复率可达 55%，见表 II），进一步放大通信开销。通信占训练总时间的 30-60%。
  - **Tutel-2DH（二维分层 AlltoAll）**：将 AlltoAll 分解为 Inter-Node + Intra-Node 两层，利用 intra-node 高带宽。但仅支持二维分层，无法适应更复杂的拓扑（如四层：Node/QPI/NVLink/Intra-NVLink），且不进行 token 去重。
  - **SmartMoE（expert placement 优化）**：通过动态调整 expert 在各 GPU 间的分布来平衡负载，但不考虑 token 去重，也不适配分层拓扑的带宽差异。在分层去重 AlltoAll 场景下，其 expert swap 策略反而可能增加通信量。
  - 全栈执行例子（Baseline Megatron-LM + 标准 AlltoAll, 4 nodes × 8 A6000 GPUs, DeepSeek-V3, K=8, E=256）：
    - **算法层**：MoE gate 做 Top-8 token-to-expert routing → mask I_route。标准 AlltoAll：每个 GPU 向所有 31 个其他 GPU 发送分配给其 experts 的 tokens。由于每 GPU 持有 E/G=8 个 experts，同一 token 可能选中同一 GPU 上的多个 experts → token 被重复发送 8 次至该 GPU。以 K=8, R=4（按 nodes 分组）为例，每 group 去重前最多 8 条相同 token → 重复率 55%（表 II）。
    - **系统框架层**：Megatron-LM → NCCL AlltoAll collective。NCCL 内部使用 Ring 或 Tree 算法，不感知分层拓扑——inter-node IB 链路 (200Gb/s) 和 intra-node NVLink (112.5GB/s) 被平等对待，低带宽 IB 链路的传输量决定整体通信延迟。
    - **编译框架层**：论文未明确说明。PyTorch eager mode → NCCL → CUDA。
    - **kernel 调度层**：NCCL AlltoAll kernel（GPU SM 上执行 send/recv）。每个 GPU 发送 max(p) ≈ T/G 的 token embeddings（M 维 × FP16=2 字节），总通信量 ≈ G · max(p) · M · 2 字节。由于 token 重复，55% 的流量是冗余的。
    - **硬件架构层**：4 nodes × 8 A6000-48G。NVLink 112.5GB/s，PCIe 4.0 x16，IB 200Gb/s。Inter-node AlltoAll 通过 IB NIC → PCIe → GPU memory，带宽受限于 IB ~25GB/s（远低于 NVLink）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HierMoE 方法**：通过两个拓扑感知的算法联合优化 MoE 训练的 AlltoAll 通信。核心洞察是：利用 GPU 集群的分层拓扑，在高层（低带宽链路）消除 token 重复以减少通信量，通过 expert 交换平衡各层级负载分布。
  - 两大设计对应解决 baseline 缺陷：
    1. **HierD-AlltoAll 解决 token 重复传输和拓扑不适配问题**：
       - Baseline 缺陷：标准 AlltoAll 无视分层拓扑，低带宽链路拖累全局；token 重复传输（55% 重复率）浪费带宽。Tutel-2DH 仅支持二维。
       - HierMoE 方法：将 AlltoAll 分解为 D 维，每层按 expert group 进行 token 去重。通过线性性能模型自动选择最优 d*。关键权衡：高层（小 group 数）去重收益大但传输量大；低层（大 group 数）去重收益小但带宽高。所有 7 种 AlltoAll 线性模型 r² > 0.997。
    2. **HierD-ES 解决去重 AlltoAll 下的负载不均衡问题**：
       - Baseline 缺陷：SmartMoE 的 expert swap 不考虑 token 去重和分层拓扑，在去重 AlltoAll 上反而降低性能（HD2-MoE-Smart < HD2-MoE）。
       - HierMoE 方法：为 HierD-AlltoAll 设计的分层 expert swap，统计去重后 token 分布变化（四种 case），增量更新通信时间估计矩阵 Q_d*（O(D·T·K·E)），选择最小化通信时间的 expert pair。smooth-max (γ=10) 平滑优化。Expert 交换 ~1% 时间开销。
  - 关键实验结果：
    - AlltoAll 通信加速比：HierMoE vs Megatron-LM 1.99×-2.72×, vs Tutel-2DH 2.34×-3.32×
    - 端到端训练加速比：HierMoE vs Megatron-LM 1.18×-1.27×
    - HierD-AlltoAll (HD-MoE) vs HD2-MoE: 1.37×-1.45× 加速
    - HierD-ES vs HD-MoE: 额外 1.13×-1.17× 加速
