## Huawei Ascend 910C NPU / CloudMatrix384 Supernode

术语是什么？

Huawei Ascend 910C 是华为自研的 AI 加速器 NPU (Neural Processing Unit)，每颗配备 64 GB HBM。基于 Ascend 910C 构建的 CloudMatrix384 是华为的超节点推理平台，集成 384 颗 Ascend 910C 和 192 颗 Kunpeng 920 CPU（每节点 16×Ascend 910C + 4×Kunpeng 920，1.5 TB 系统 RAM），分布在 24 个节点。所有 NPU 和 CPU 通过 Unified Bus (UB) 互联——一种超高带宽 P2P 架构，提供 non-blocking all-to-all 连接，节点内和节点间通信延迟接近均匀。这使得 CloudMatrix384 可作为单个大规模逻辑节点运行，支持 TP 和 EP 等细粒度并行策略。

从硬件架构角度拆解术语：

```
CloudMatrix384 节点拓扑：

单个节点（共 24 个）:
├── 16 × Ascend 910C NPU
│   └── 每颗 64 GB HBM, HCCS/Ascend UB 互联
├── 4 × Kunpeng 920 CPU
│   └── 1.5 TB 系统 RAM
└── 互联: Unified Bus (UB)
    ├── Intra-node: 超高带宽, non-blocking all-to-all
    └── Inter-node: RDMA-capable links

全系统:
├── 384 Ascend 910C NPU (24 nodes × 16)
├── 192 Kunpeng 920 CPU
├── 24.576 TB 总 HBM (384 × 64 GB)
└── 36 TB 总系统 RAM (24 × 1.5 TB)
```

Unified Bus 是 CloudMatrix384 的关键架构特性：与传统的 PCIe/NVLink 分级拓扑（intra-node NVLink 快，inter-node InfiniBand 慢）不同，UB 提供 "near-uniform" 的节点内/节点间通信性能。这意味着 MoE 的 all-to-all 通信在跨节点 expert dispatch 时不会出现显著的带宽悬崖，使 EP 可以跨越更多 NPU 而通信开销仍可控。

术语一般如何实现？如何使用？

CloudMatrix384 是华为云的硬件平台，通过 CANN 软件栈暴露 NPU 计算和通信能力。在 ElasticMoE 中的使用：HMM 数据面通过 CANN API 管理 NPU HBM 和 IPC，HCCL 通过 UB 执行 P2P 传输和集合通信。与 NVIDIA 生态的对比：Ascend 910C 大致对应 A100/H100 级别 AI 加速器，CloudMatrix384 的 UB 架构类比 NVIDIA NVSwitch 全互联 fabric，但规模更大（384 NPU vs 最多 256 GPU with NVSwitch/NVL72）。

涉及论文标题：
- ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

---
