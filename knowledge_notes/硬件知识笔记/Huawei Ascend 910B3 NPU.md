## Huawei Ascend 910B3 NPU

术语解释
Ascend 910B3 是华为于 2024 年推出的 AI 训练/推理加速器 (NPU)，属于华为 Ascend 9xx 系列 AI 处理器。ETR 论文的全部实验在该 NPU 集群上完成。

术语是什么？
Ascend 910B3 单颗 NPU 规格：20 AI Cores @ 1.8GHz 主频，fp16 理论算力 313 TFLOPS，HBM 64GB @ 1.6GHz 频率，HBM 带宽 1.6 TB/s。每 8 颗 NPU 安装在同一 Atlas 800T A2 服务器内，服务器内部采用全 mesh 互联（任意两 NPU 可直连通信）。计算单元分工：AI CORE 负责矩阵乘法和卷积（Cube 计算单元），AI VECTOR CORE 负责并行向量计算，MIX AIC 负责异构算子融合，AI CPU 负责专用 AI 指令执行。

从硬件架构角度拆解：
ETR 论文使用三组集群规模：
- 32N: 4 台 Atlas 800T A2 (每台 8 NPU), TP=4, PP=4, DP=2, EP=2
- 64N: 8 台服务器, TP=8, PP=4, DP=2, EP=2
- 256N: 32 台服务器, TP=8, PP=8, DP=4, EP=2

并行策略与硬件映射关系：
- TP (Tensor Parallelism): 单层内权重切分至 TP 组内 NPU，利用服务器内全 mesh 互联高带宽
- PP (Pipeline Parallelism): 不同层分配至不同 PP stage，利用服务器间互联
- DP (Data Parallelism): 不同 micro-batch 分配至不同 DP 组
- EP (Expert Parallelism): expert 分布在不同 NPU，通过 All-to-All 通信完成 token dispatch/combine

术语一般如何实现？如何使用？
通过 Huawei CANN (Compute Architecture for Neural Networks) 软件栈暴露 NPU 计算和通信能力。Ascend 910B3 定位与 NVIDIA A100 80GB 大致同级，但使用华为自研 Da Vinci 架构 (非 CUDA)。软件生态包括 MindSpore 和 MindSpeed-LLM (基于 Megatron-LM 改造) 训练框架。相比 Ascend 910C（CloudMatrix384 Supernode 的 384 NPU fabric），910B3 的服务器间互联规模较小，但仍可通过 Atlas 800T A2 的全 mesh 实现高效的节点内通信。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection
