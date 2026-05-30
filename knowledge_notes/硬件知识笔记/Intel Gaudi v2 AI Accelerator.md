## Intel Gaudi v2 AI Accelerator

术语解释
Intel Gaudi v2 是 Intel 的 AI 训练和推理加速器（AI Accelerator），基于 Habana Labs 架构，面向生成式 AI 和大语言模型工作负载设计。通过 SynapseAI 软件栈暴露计算和通信能力，提供与 NVIDIA GPU 不同的生态选择。

术语是什么？
Gaudi v2 的核心特性：
- **计算架构**：包含矩阵乘法引擎（Matrix Multiplication Engine, MME）和张量处理器核心（Tensor Processor Core, TPC），支持 FP32/FP16/BF16/INT8 等精度
- **内存**：96 GB HBM2e 显存，带宽约 2.45 TB/s
- **互联**：24 × 100 Gbps RoCE (RDMA over Converged Ethernet) 端口，支持 scale-out 集群通信
- **软件栈**：SynapseAI 框架提供 PyTorch/TensorFlow 集成，自动图编译和 kernel 优化
- **编程模型**：通过 `habana_frameworks.torch` 提供 PyTorch 兼容 API，`hpu` 设备类型

核心差异化：Gaudi 系列使用 RoCE/RDMA 而非 NVLink/NVSwitch 进行多卡互联，强调以太网 scale-out 能力。Gaudi 3 对比 Gaudi 2 有 2× FP8 算力和 1.5× HBM 带宽提升。

从硬件架构角度拆解术语：
```
Gaudi v2 Hardware Architecture:
┌─────────────────────────────────────────────────┐
│                   Gaudi v2 Chip                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ MME      │  │ TPC x N  │  │ DMA Eng  │      │
│  │(Matrix)  │  │(Tensor)  │  │          │      │
│  └──────────┘  └──────────┘  └──────────┘      │
│  ┌──────────────────────────────────────┐       │
│  │         96 GB HBM2e (4 stacks)       │       │
│  └──────────────────────────────────────┘       │
│  ┌──────────────────────────────────────┐       │
│  │   24×100Gbps RoCE (RDMA) Ports       │──────>│──> Scale-out
│  └──────────────────────────────────────┘       │    集群互联
└─────────────────────────────────────────────────┘
         │
    SynapseAI Graph Compiler
         │
    PyTorch/TF → HPU Graph → 执行
```

在 Kim et al. (2025) 的 MoE KD 实验中，使用 4 卡 Intel Gaudi v2 + SynapseAI 1.18.0 进行训练，batch_size=16，LR=1e-5，epochs=10。Gaudi v2 的多卡通信通过 RoCE 进行 router + expert 的协作。

术语一般如何实现？如何使用？
- SynapseAI 通过 `habana_frameworks.torch` 提供与 PyTorch 兼容的 HPU (Habana Processing Unit) 设备
- `hpu` 设备类型替代 `cuda` 设备类型：`model.to('hpu')`, `tensor.to('hpu')`
- 训练和推理缩放：通过 Gaudi 集群以太网 scale-out，支持数据并行和模型并行
- Intel Developer Cloud 提供 Gaudi v2 实例用于开发和评测
- 相比于同代 NVIDIA H100，Gaudi v2 在性价比和以太网生态集成上有优势，但在软件生态成熟度和绝对算力上有差距

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models
