## GPU Cluster Hierarchical Topology for MoE Communication

术语解释
GPU 集群的分层互联拓扑是 HierMoE 算法设计的核心硬件背景。四层拓扑从高到低为：Inter-Node (InfiniBand, ~200Gb/s) → Inter-QPI → Inter-NVLink → Intra-NVLink (~112.5GB/s)，带宽逐层递增约 4-500×。HierD-AlltoAll 利用这种分层带宽差异，通过 token 去重将通信量从低带宽的高层"推"向高带宽的低层。

术语是什么？
GPU 集群中多节点多 GPU 之间的互联具有天然的层次化带宽结构。以 HierMoE 使用的 4 节点 × 8 GPU 集群为例：(1) Inter-Node 层：节点间通过 InfiniBand (Mellanox MT28908, 200Gb/s) 互联；(2) Inter-QPI 层：同一节点内不同 QPI domain 间通过 QPI 互联；(3) Inter-NVLink 层：同一 QPI domain 内不同 GPU 间通过 NVLink (112.5GB/s per 4× link) 互联；(4) Intra-NVLink 层：GPU 内部通过共享内存访问。每一层参与 AlltoAll 通信的 GPU 数量不同（U[1]=4 nodes, U[2]=8 QPI groups, U[3]=16 NVLink groups, U[4]=G=32），导致不同层的 expert group 大小不同，token 去重收益各异。

从硬件架构角度拆解术语，给出具体例子。

```
四层拓扑 AlltoAll 示例 (4 nodes × 8 GPUs = 32 GPUs):
┌─────────────────────────────────────────────────────────────┐
│ Node 0                        Node 1                        │
│ ┌─────────┐ ┌─────────┐      ┌─────────┐ ┌─────────┐      │
│ │QPI 0    │ │QPI 1    │      │QPI 0    │ │QPI 1    │      │
│ │GPU0 GPU1│ │GPU2 GPU3│ ...  │GPU8 GPU9│ │GPU10 11│ ...  │
│ │ NVLink  │ │ NVLink  │      │ NVLink  │ │ NVLink  │      │
│ └─────────┘ └─────────┘      └─────────┘ └─────────┘      │
│           ↑ QPI              ↑ IB (200Gb/s) → Node 2,3     │
└─────────────────────────────────────────────────────────────┘

层级带宽比 (A6000 集群):
  Intra-NVLink:   112.5 GB/s  (baseline, 1×)
  Inter-NVLink:   ~112.5 GB/s  (1×)
  Inter-QPI:      ~PCIe 4.0 x16 ≈ 32 GB/s (0.28×)
  Inter-Node IB:  200 Gb/s ≈ 25 GB/s  (0.22×)

HierD-AlltoAll d* 选择权衡:
  d=1 (标准AlltoAll): 所有32 GPU平等通信，IB瓶颈
  d=2 (2D-AlltoAll):  4节点间→节点内8GPU，IB传4组tokens
  d=3 (3D-AlltoAll):  4节点→8 QPI域→域内2 GPU，IB传4组
  d=4 (4D-AlltoAll):  4节点→8 QPI→16 NVLink→GPU内
  → d* = argmin over 4 estimated times based on actual routing
```

术语一般如何实现？如何使用？

- 通过 nccl-tests 中的 ncclAlltoAll 变体测量各层级的 α (启动延迟) 和 β (每字节传输时间)
- 4 节点 32 GPU 集群需要测量 7 种 AlltoAll 变体（1 标准 + 3 Inter-level + 3 Intra-level）
- 参数拟合在集群启动时一次性完成（<300s），训练期间不需要重新测量
- 论文使用的 A6000 集群不是最高端的训练配置，但 HierMoE 在更复杂的拓扑（如 H100 NVSwitch + NVLink Network）上的理论收益更大，因为更多层级意味着更多去重机会

涉及论文标题：
- HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap
