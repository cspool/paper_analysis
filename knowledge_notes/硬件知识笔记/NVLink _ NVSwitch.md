## NVLink / NVSwitch

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

NVLink 是 NVIDIA 开发的高带宽 GPU 直连互联技术，为同一节点内多个 GPU 之间提供远高于 PCIe 的点对点通信带宽。NVSwitch 是连接多个 NVLink 端口的交换芯片，使节点内所有 GPU 可通过全互联拓扑（而非 ring 或 tree）直接通信。FUSCO 评估中每 H100 GPU 配置 18 条 NVLink link，理论聚合带宽约 480 GB/s per GPU。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

FUSCO 利用 NVLink 的高带宽实现 Hierarchical Routing 的第二级（Expert-Level Distribution）。在 FUSCO 的设计中，跨节点通信通过 RoCE (50 GB/s) 仅发送一份 token 拷贝给 forwarder GPU，forwarder 再通过 NVLink P2P (480 GB/s) 在同一节点内的多 GPU 间分发。这种设计的核心硬件依据是节点内/节点间带宽的巨大不对称性（480 vs 50 GB/s，约 9.6×）。

```
# NVLink 在 FUSCO Hierarchical Routing 中的角色
# Inter-node (RoCE 400Gbps × 10 网卡, ~50 GB/s):
#   sender → forwarder: 发送一份 token 拷贝 (cross-node, slow)
# Intra-node (NVLink 480 GB/s):
#   forwarder → expert GPU₁: P2P copy via cudaMemcpy (fast)
#   forwarder → expert GPU₂: P2P copy via cudaMemcpy (fast)
#   ...
# 效果: top-k 次跨节点传输 → 1 次跨节点 + k 次节点内传输
```

H100 的 NVLink 4.0 提供 900 GB/s 双向带宽（每条 link 50 GB/s × 18 links）。dComm 的 intra-node 路径利用 GPUDirect P2P over NVLink，在 GPU-to-GPU copy 路径中集成 descriptor 解释逻辑，inline 完成 layout transformation。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- NVLink 带宽远高于 PCIe 和网络带宽（H100 NVLink 900 GB/s vs PCIe 5.0 x16 64 GB/s vs 400Gbps RoCE 50 GB/s），是节点内 GPU 通信的最优路径
- 在 MoE 训练中，NVLink 通常用于 TP（tensor parallelism）的 all-reduce 和 ESP 的 AllGather/ReduceScatter
- FUSCO 创新性地将 NVLink 用于 Hierarchical Routing 的 expert-level 分发阶段，利用 P2P 直接 copy，而非集合通信
- NCCL 在 intra-node 场景自动选择 NVLink 传输路径（通过拓扑检测），GPUDirect P2P 使用 NVLink 实现 peer-to-peer memory access

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- LongCat-Flash Technical Report
