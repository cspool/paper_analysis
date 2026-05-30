## NVL72 (NVIDIA GB200 NVL72)

术语解释
NVL72 是 NVIDIA 基于 GB200 Grace Blackwell Superchip 的机架级全互联系统，在单个机架内通过 NVLink Switch 连接 72 个 Blackwell GPU，提供 130 TB/s 的统一 GPU-to-GPU 双向带宽，形成一个逻辑上的巨型 GPU。

术语是什么？
NVL72 的本质是将 72 个 Blackwell GPU 通过 NVLink 5.0（1.8 TB/s per GPU）全部连接到 NVSwitch fabric，实现机架内任意 GPU 之间的 peer-to-peer direct communication（无 PCIe 中转）。其 chip-level architecture：每个 GB200 Superchip 含 1×Grace CPU + 2×Blackwell GPU；9 个 NVSwitch tray 提供全互联交换；总 36 个 GB200 × 2 GPU = 72 GPU。所有 GPU 共享 coherent NVLink domain → 在 MoE EP 部署中可以视为 homogeneous all-to-all 互联，EP 通信不再受 inter-node vs intra-node 带宽差异限制。

从芯片设计角度拆解：
```
=== NVL72 Chip-Level Topology ===
72 Blackwell GPUs connected via NVSwitch Fabric

   GB200_0          GB200_1          ...  GB200_35
   GPU0  GPU1       GPU2  GPU3            GPU70 GPU71
     \    /           \    /                \    /
    NVSwitch_1       NVSwitch_2     ...   NVSwitch_9
         \              |                /
          \-------------+---------------/
                 NVLink 5.0 Fabric
          (1.8 TB/s per GPU, 130 TB/s total)

All-to-all bandwidth: homogeneous (no intra/inter-node drop)
vs Traditional: NVLink intra-node 900 GB/s → IB inter-node 50 GB/s (18× drop!)
```

在 DualSparse-MoE 的 ASTRA-SIM 模拟中，NVL72 (EP=9, TP=8) 的 homogeneous fabric 使 S-ETP 的通信优化效果特别显著（10.2-80.4% bandwidth improvement），因消除了 ETP 在异质互联下的多轮 collective bottleneck。

术语一般如何实现？如何使用？
- NVIDIA GB200 NVL72 物理配置：36 个 GB200 Superchip → 72 Blackwell GPU + 36 Grace CPU + 9 NVSwitch tray
- NVLink 5.0: 1.8 TB/s 双向/GPU，总聚合 130 TB/s
- 对 MoE EP：全互联消除 intra/inter-node 通信差异 → EP scaling 不受网络拓扑限制 → 更细粒度 EP (如 EP=72) 可行
- S-ETP 优势场景：NVL72 的 homogeneous fabric 使减少 collective rounds (ETP→S-ETP) 的收益最大化
- CloudMatrix384 (CM384): 类似 NVL72 的全互联 homogeneous 系统但规模为 384 加速器，同样使 S-ETP 获益 (9.9-28.3%)

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
