## NVL72 System Topology（NVL72 系统拓扑）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVL72 是 NVIDIA 在 Blackwell 架构（GB200）中提出的多 GPU 系统拓扑，将 72 个 B200 GPU 通过 NVLink 5th Gen 和 NVSwitch 全互联成单一的高带宽域。每个 GB200 Superchip 包含 1 个 Grace CPU + 2 个 B200 GPU，36 个 Superchip 组成一个 NVL72 rack。NVL72 内部，所有 72 个 GPU 之间通过 NVLink 5th Gen 提供 1.8 TB/s 的双向带宽（每 GPU），实现 all-to-all 全互联。这 72 个 GPU 可作为统一的 EP（Expert Parallelism）group 或 DP（Data Parallelism）group 使用。论文默认使用 NVL72 拓扑作为 32 B200 GPU 系统的互联模型。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
NVL72 对 MoE 推理的影响通过论文的通信分析体现：

- **NVL72 内部 (NVLink 5th Gen, 1.8 TB/s)**：MoE all-to-all dispatch/combine 通信延迟极低（B=128 时 ~17.65 µs）。这使得 MoE 通信不是瓶颈，系统可以达到更高的 $B_{\text{SLO}}$。
- **跨 NVL72 (InfiniBand XDR, 100 GB/s)**：通信延迟大幅增加（B=128 时 ~151.8 µs，8.6× 更高）。通信成为主导延迟来源，$B_{\text{SLO}}$ 显著降低。

论文的关键部署决策分析：使用一个大规模 EP group（如 256 GPU 跨多个 NVL72）还是一个 NVL72 内的紧密耦合 group（32 GPU）。NVL72 内的 32 GPU 配置在大多数场景下更优：(1) 避免了跨 rack 的低带宽通信；(2) 每 GPU 管理 8 个 expert 天然缓解 expert 负载偏斜。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
NVL72 是 Blackwell 平台的旗舰配置，物理实现采用 NVSwitch 芯片和铜缆背板互联。与上一代 H100 的 8-GPU NVSwitch 域相比，NVL72 将高带宽域扩大了 9×（8→72 GPU）。对于 MoE 部署，建议将 EP group 保持在一个 NVL72 域内（32-72 GPU），避免跨 InfiniBand 的 all-to-all 通信。如果 expert 数量超过单域可容纳的范围（如 256 experts），使用多 NVL72 实例（32 GPU×8）而非一个大 EP group（256 GPU）。

涉及论文标题：
- Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts
