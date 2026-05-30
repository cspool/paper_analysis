## Ascend 910A NPU

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Ascend 910A 是华为设计的神经网络处理单元（NPU），专为 AI 训练和推理设计。每芯片集成 32 个 AI Core（达芬奇架构计算单元），支持 FP16 半精度 320 TFLOPS 和 INT8 整数精度 640 TOPS。最大内存容量 2TB，最大内存带宽 1.07TB/s。SoC 采用 Mesh NoC 架构提供统一可扩展通信网络，实现 256GB/s 片上带宽。

在 Ascend 910A 服务器（Atlas 800 9000 型号）中，每 8 个 NPU 分为两组，组内通过 HCCS（Huawei Cache Coherence System）高速互联。多节点间通过两级 Fat-tree 网络拓扑互联，每 Leaf 交换机连接 4 个 NPU 服务器，使用 RoCE 实现节点间通信。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

LocMoE 论文中使用 3 组 Ascend 910A 集群配置：
- 64N：8 节点 × 8 NPU = 64 NPUs
- 128N：16 节点 × 8 NPU = 128 NPUs
- 256N：32 节点 × 8 NPU = 256 NPUs

在 MoE 训练中，每个 NPU 负责计算一部分 expert 的 FFN。数据流：token embeddings 通过 Fat-tree 网络进行 All-to-All 通信（HCCL 执行）分配到各 NPU 上的 expert → 各 NPU 的 AI Core 执行 expert FFN 计算（GeLU 激活的两层线性变换）→ All-to-All 将结果返回。HCCS 提供节点内 NPU 间 256GB/s 的高带宽，TP 域利用 HCCS 分担 EP 域的跨节点通信压力（Group-wise All-to-All）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Ascend 910A 通过 CANN（Compute Architecture for Neural Networks）异构计算架构提供编程接口，支持 MindSpore、PyTorch、TensorFlow 等 AI 框架。HCCL 提供集合通信原语（All-to-All, All-Gather, All-Reduce 等），支持 ring/mesh/HD/ring+HD/mesh+HD 等多种通信算法。LocMoE 使用 CANN 5.1.RC2.1 (toolkit 1.84, driver 23.0.rc2) 和 MindSpore 2.0.0 进行训练。

涉及论文标题：
- LocMoE: A Low-overhead MoE for Large Language Model Training
