## NVLink / NVSwitch（GPU 高速互联）

术语是什么？
NVLink 是 NVIDIA 开发的高带宽 GPU-to-GPU 互联技术。第 4 代 NVLink（用于 H100）提供每条链路 450 GB/s 单向带宽，通过 NVSwitch 芯片实现节点内 8 GPU 的全互联（all-to-all）。NVLink 域是 MoE 训练和推理中最宝贵的通信资源——其带宽远高于节点间 InfiniBand（450 GB/s vs 50 GB/s per link）。MoE Parallel Folding 的核心优化之一就是将 EP/CP 的 All-to-All 通信限制在 NVLink 域内，避免跨节点通信。

从硬件架构角度拆解术语：
在 DGX H100 节点中，NVLink/NVSwitch 的拓扑布局：
- 每 H100 GPU 有 18 条 NVLink 4.0 链路连接到 4 个 NVSwitch 芯片
- NVSwitch 是 8×8 全交叉无阻塞交换机——8 GPU 间任意 pair 都可同时以 450 GB/s 全速通信
- 总聚合带宽：8 × 18 × 50 GB/s = 7200 GB/s（双向）
- MoE Parallel Folding 的策略：通过控制 EP×TP×CP ≤ 8，使这些并行维度在 NVLink 域内完成通信
  - Mixtral 8x22B, 128 GPU: EP=8, TP=2, CP=1 → EP×TP=8 即 8 GPU，恰好在一个 DGX 节点内
  - 当 EP×CP > 8（跨越 NVLink 域）时，A2A 走 InfiniBand，延迟显著增加（Figure 6）

术语一般如何实现？如何使用？
- NVLink 通过 NCCL 的高性能 SHARP 优化实现通信（NCCL 2.27+ 支持 NVLink SHARP，将 AG/RS 的 SM 占用从 16 降至 6）
- 在训练框架中，通过控制并行度将通信限制在 NVLink 域内是获取高 MFU 的关键策略
- MoE Parallel Folding 通过生成异构并行映射（Attention/MoE 分层独立），自动将 EP 通信折叠到 NVLink 域

涉及论文标题：
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core
- NetMoE: Accelerating MoE Training through Dynamic Sample Placement
