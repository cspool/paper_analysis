## Bandwidth-Constrained Fine-Tuning (带宽受限微调)

术语解释
Bandwidth-Constrained Fine-Tuning 是指在 GPU 间仅有低带宽互联（如 PCIe Switch，无 NVLink/InfiniBand）且 GPU 数量有限的"经济型"集群上对大模型进行微调的场景。这种硬件配置下，GPU 间通信和 host-to-GPU 数据搬移成为主要瓶颈，传统的数据并行或张量并行因频繁的 collective communication 而效率低下。

术语是什么？
带宽受限 GPU 节点的典型特征：
- GPU 数量少（4-8 卡/节点）
- GPU 间通过 PCIe Switch 互联（带宽 ~32 GB/s，远低于 NVLink 的 900 GB/s）
- 无跨节点高速互联（或仅有基础 Ethernet）
- 每 GPU 显存有限（如 A800 40GB）
- 追求高性价比（低成本硬件 + 大模型能力）

APTMoE 针对此场景选择 Pipeline Parallelism 而非 Tensor/Expert Parallelism 的原因：Pipeline Parallelism 仅需异步 P2P 通信（点对点传输 activations/gradients），通信量远小于 All-to-All（EP）或 All-Reduce（DP），更适合低带宽环境。

从系统架构角度拆解术语。
```
# 带宽受限环境下的并行策略选择
# Tensor Parallelism: 每层都需要 All-Reduce → 通信量大 → 不适合 PCIe
# Expert Parallelism: 每层需要 All-to-All Dispatch+Combine → 通信量大 → 不适合 PCIe
# Data Parallelism: 每 step 需要 All-Reduce 梯度 → 通信量大 → 不适合 PCIe
# Pipeline Parallelism: 仅 P2P 传输 activations → 通信量小 → 适合 PCIe ✓

# APTMoE 的总数据搬移量对比
Mobius:     每个 stage 全量 expert 参数 + MHA + Gate → Host↔GPU
APTMoE:     仅高热度 expert 参数 + MHA + Gate → Host↔GPU
            低热度 expert → 在 CPU 就地计算（0 数据搬移）
```

术语一般如何实现？如何使用？
- 硬件典型配置：4-8 GPU 同节点 PCIe Switch、多节点通过 InfiniBand HDR 100Gbps 互联
- 软件栈：Ubuntu + PyTorch + NCCL（限制在 PCIe 带宽内）
- Pipeline stage 数 > GPU 数，相邻 stage 映射到不同 GPU（cross-mapping 减少 contention）
- Micro-batch 数 = GPU 数（平衡流水线气泡和内存）
- CPU 参与计算可额外提升吞吐 15%-33%（取决于 expert 规模和热度偏斜程度）

涉及论文标题：
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes

---
