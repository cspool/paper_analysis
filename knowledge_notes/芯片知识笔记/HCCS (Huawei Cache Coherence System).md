## HCCS (Huawei Cache Coherence System)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HCCS (Huawei Cache Coherence System) 是华为 Ascend NPU 的 intra-node 高速互联协议，功能类似 NVIDIA NVLink。它实现同一 Atlas server 内多个 Ascend NPU 之间的全互联（fully-connected），提供远超 PCIe 的带宽和 cache coherence 能力。在 MixServe 的 Ascend 910B 集群中，每台 Atlas 800T A2 server 内 8 个 Ascend 910B NPU 通过 HCCS 全互联，单链路带宽最高 480 Gbps（约 60 GB/s 双向）。

从芯片设计角度拆解术语，比如术语如何在芯片设计中发挥作用，给出术语在芯片设计中运转流程的具体例子。通过联网搜索让回答具体和精准。
HCCS 的芯片级实现：
- 物理层：基于华为自研的 HCCS link 协议，使用差分信号对通过 PCB trace 或 bridge 连接 Ascend NPU die。HCCS 控制器集成在 Ascend NPU die 内部，通过 HCCS I/O 模块直接访问 NPU 的 on-chip buffer 和 HBM memory。
- 拓扑：Atlas 800T A2 server 内 8 个 Ascend 910B 通过 HCCS 全互联（fully-connected mesh），每对 NPU 之间有专用 HCCS 通信链路。
- 缓存一致性：HCCS 支持 NPU 间 cache coherence，允许 NPU 直接访问远程 NPU 的 HBM 内存，实现 load/store 语义（类似 NVLink P2P memory access）。

在 MixServe 中的角色：
- HCCS 构成 intra-node 高带宽通信域，用于 TP group 内的 RS/AG 通信。
- HCCS 带宽（~60 GB/s per link）显著高于 inter-node RoCE（200 Gbps ≈ 25 GB/s），约 2.4× 差距，是 hybrid TP-EP 设计的硬件基础。
- MixServe 的 profiling 数据（Fig. 3）显示 d≤8（纯 intra-node HCCS）时通信开销低，d>8（涉及 inter-node RoCE）时开销激增。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 华为 Ascend 生态：HCCS 是华为全栈 AI 解决方案的组成部分，与 CANN、Ascend NPU、PyTorch 协同。HCCL（Huawei Collective Communication Library，类似 NCCL）能感知 HCCS 拓扑并优化 collective 通信。
- 与 NVLink 对比：NVLink 4.0 总带宽 900 GB/s（18 lanes × 50 GB/s），HCCS 480 Gbps ≈ 60 GB/s per link。关键差异在 lane 数量。
- 使用方式：开发者通过 CANN + PyTorch 分布式接口间接使用，HCCS/HCCL 自动被用于 intra-node collective 通信。

涉及论文标题：
- MixServe: An Automatic Distributed Serving System for MoE Models with Hybrid Parallelism Based on Fused Communication Algorithm
