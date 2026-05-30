## HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission

- baseline方法是什么？
  - **标准 Expert Parallelism (EP) + MoE 训练系统**（Tutel, FasterMoE, SmartMoE）：在单 DC 高性能环境下，通过 All-to-All (A2A) 通信在 GPU 间交换 token data 和 expert 输出。核心流程：Gate network 计算 routing → A2A dispatch（将 token 发送到对应 expert 所在 GPU）→ Expert FFN 本地计算 → A2A combine（将 expert 输出合并回原 GPU）。现有优化方法集中在对计算和通信重叠（FasterMoE 的 overlap scheduling、Tutel 的 2D 分层 A2A、SmartMoE 的 pipeline 调度），本质是在高带宽 DC 内部通过隐藏通信延迟来提升吞吐。但在跨 DC 场景下，低带宽（如 10Gbps Ethernet vs PCIe 128Gbps）导致通信时间远远超过计算时间，重叠策略失效——因为无论怎么重叠，通信时间本身无法被消除或缩减。此时 EP 占训练总时间的 50%-90%（Figure 2b），成为核心瓶颈。
  - **全栈执行例子（Baseline Tutel on Cluster-L: 4 DCs × 8 GPUs, MoE with E=32 experts, K=2 activated experts）**：
    - **模型推理/训练算法层**：Standard MoE layer: token embedding → Attention → Gate(top-K routing) → A2A dispatch(token data 跨 DC 传输) → Expert FFN(2×GeMM with SiLU on GPU Tensor Cores) → A2A combine(output 跨 DC 传输) → next layer。每层 2 次全局 A2A，每次传输 data size = D·(G-1)/G per GPU。当 G=32 时，跨 DC 低带宽链路的 A2A 传输时间主导了整个 iteration。
    - **系统框架层**：PyTorch v1.12.1 + Tutel。Tutel 使用 NCCL All-to-All collective 原语执行 token dispatch/combine。训练使用 Adam optimizer + DDP (All-Reduce 同步梯度)。跨 DC 通信经过 10Gbps Ethernet，与 intra-DC 的 128Gbps PCIe 形成巨大带宽差距。
    - **编译框架层**：论文未明确说明。PyTorch eager mode → CUDA compiler → cuBLAS/NCCL backend。
    - **kernel 调度层**：标准 cuBLAS GEMM (expert FFN) + NCCL All-to-All collective。无自定义 kernel。通信与计算通过 Tutel 的 pipeline 调度尝试重叠，但在 10Gbps 下 A2A 通信时间（数十 ms）远超 expert 计算时间（<1ms），重叠收益极低。
    - **硬件架构层**：NVIDIA A800 GPU (PCIe 3.0 x16 128Gbps 节点内) + Ethernet 10Gbps 跨节点。EP 的 A2A 通信经过低带宽跨 DC 链路，成为系统瓶颈。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **HybridEP 方法**：通过三部分设计将 EP 从纯 A2A 通信改造为数据-专家混合传输，结构化消除跨 DC 通信瓶颈：
    1. **Stream-Based Modeling（流建模，§III）**：将 MoE 训练解耦为计算流和通信流，建模 A2A（传输 data）和 AG（传输 expert）的延迟特性——A2A 延迟 O(1)（与 GPU 数无关），AG 延迟 O(n)（随 GPU 数线性增长）。通过推导 A2A→AG 转换关系：减少 $\frac{D}{G}$ 的 A2A 流量换取 $P_E$ 的 AG 流量，求解最优混合比例 p。关键洞察：当 $2D \geq G \cdot P_E$ 时，仅使用 AG（p=0）即可消除所有跨 DC 的 token data 传输，用更适合压缩和异步的 expert 传输替代。
    2. **Domain-Based Partition（域分区，§IV-A）**：将 modeling 输出的 p 值映射到 GPU 级通信拓扑。定义 Expert Domain（域内 AG，域间 A2A），通过 Multilevel Description → Location Renumbering → Topology Construction 三步在处理复杂层级硬件架构（DC→Node→GPU）的同时保持通信模式清晰。
    3. **Parameter-Efficient Migration（参数高效迁移，§IV-B）**：利用 expert 的两大优势——(i) 可压缩性（expert weight 分布紧凑，残差稀疏，可通过 shared+residual Top-k 实现 50× 压缩），(ii) 异步潜力（expert 仅间歇参与计算，可提前传输）。通过 SR-Based Expert Compression 减少传输流量，Asynchronous Communicator 实现 expert 通信与 pre-expert computation 的完全重叠。
  - **对应解决 Baseline 缺陷**：
    - Baseline 的 A2A 通信时间在低带宽下无法消除 → HybridEP 通过 expert 迁移将 A2A 转换为 AG，用更适合压缩和异步传输的 expert 替代 data 传输，从根本上减少跨 DC 链路上的通信量。
    - Baseline 的重叠策略在低带宽下失效 → HybridEP 的 AG 通信可以与 pre-expert computation 重叠（expert 不需要 token data 作为输入，可独立传输），且 AG 的异步特性和高可压缩性使其 overhead 远小于 A2A。
    - Baseline 通信流量随 token 数线性增长 → HybridEP 的 AG 流量与 token 数无关（仅与 expert 大小相关），提供固定上界的通信流量，使系统性能更可预测（Figure 16）。
    - Baseline 的 A2A 频率为 O(G²) → HybridEP 通过扩展 expert domain 将 A2A 频率降低并转换为 AG（Table VII），减少跨域通信次数。
  - **全栈执行例子（HybridEP on Cluster-L: 4 DCs × 8 GPUs, p=0, AG-only case, Mistral-Small, E=32, K=2）**：
    - **模型推理/训练算法层**：
      - Token batch → Embedding → Attention → Gate(top-K routing)
      - **关键差异**：不再执行跨 DC 的 A2A dispatch/combine
      - 每个 Expert Domain 内（假设 S_ED=8，每 DC 的 8 GPU 为一个 domain），所有 expert 提前通过 AG 收集到域内每 GPU
      - Expert FFN 全部在本地 GPU 执行（因为所有 expert 参数已通过 AG 在域内可用）
      - 无需跨域数据传输，仅需域内 AG 同步 expert 参数
      - SR-Based Expert Compression 将 expert 传输量压缩 50×：原始 P_E=4.7MB → 压缩后 ~0.094MB per expert
    - **系统框架层**：
      - 修改后的 Tutel + PyTorch v1.12.1
      - 训练前：Stream-Based Modeling 根据 G=32, B_inter=10Gbps, P_E=4.7MB, D=3MB → 判断 2×3MB - 32×4.7MB < 0 → Case 2.1 或转换判断 → 若转为 Case 2.2 则 p=0 (仅 AG)
      - Domain-Based Partition：构建 2 层 topology (S_ED^0=4, S_ED^1=8)
      - SREncode 与 optimizer step 融合：将每 expert 压缩为 value-index 稀疏格式存入 Send Queue
      - Asyn-comm：AG 通信在 Attention 计算时并发执行，NCCL All-Gather 从 Send Queue 收集域内其他 GPU 的压缩 expert
      - SRDecode 与 expert FFN 融合：解码恢复完整 expert 参数并立即执行 FFN
    - **编译框架层**：论文未明确说明。PyTorch eager mode。
    - **kernel 调度层**：NCCL All-Gather (expert 参数域内收集) 替代了 NCCL All-to-All (token data 跨域传输)。Top-k 压缩/解压通过 custom CUDA kernel 实现。AG 通信通过 CUDA stream 与 Attention 的 GEMM kernel 并行。
    - **硬件架构层**：4 × 8 × NVIDIA A800。intra-DC PCIe 3.0 128Gbps, inter-DC Ethernet 10Gbps。当 p=0 时，跨 DC 的 A2A token 传输完全消除，仅保留域内的 AG expert 传输，而 AG 的流量被 SR 压缩减少 50×，并通过异步通信隐藏延迟。端到端 speedup：up to 5.6× (192MB data, Cluster-L) vs Tutel/FasterMoE/SmartMoE。
