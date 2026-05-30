## Ascend 910 AI Accelerator（昇腾910 AI加速器）

术语是什么？
Ascend 910 是华为推出的 AI 训练加速器（NPU），基于自研达芬奇（Da Vinci）架构。910B 系列为升级版，910B2-64GB 配备 64GB HBM。PreMoE 在 64×Ascend 910B2-64GB NPU 集群上完成实验。部署效率随 sparsity 变化：0% sparsity（256 experts/layer）→ 64 NPUs, latency 115.35ms/tok；50% sparsity（128/layer）→ 32 NPUs, 93.72ms/tok, +23% throughput；75% sparsity（64/layer）→ 16 NPUs, 73.20ms/tok, +58% throughput。Ascend 910B2 的 Tile-based architecture（粗粒度 Tile 架构）区别于 NVIDIA GPU 的 SIMT，采用多级 on-chip buffer 加向量/矩阵计算单元。EfficientMoE 论文在 4 节点共 32 块 Ascend 910（32GB 版本）集群上实验。

从硬件架构角度拆解术语：
在 PreMoE 的硬件配置中：64×Ascend 910B2-64GB NPU 通过 HCCS (Huawei Cache Coherence System) 进行节点内互联，节点间通过 RoCE 网络互联。每块 NPU 64GB HBM 需容纳 expert shards。PreMoE 的 Expert Pruning 减少每 NPU 的 expert 存储量：50% sparsity 时每 NPU expert 参数减半，所需 NPU 从 64 降至 32，但单 NPU 内存压力相同（因为 NPU 数也减半）。关键：all-to-all 通信涉及的 NPU 数从 64 降至 32（50% sparsity）或 16（75% sparsity），减少通信跳数和拥塞。

EfficientMoE 论文对比：V100 节点内 NVLink 300 GB/s vs Ascend 910 节点间 RoCE 100 GB/s——带宽差距解释了通信优化在不同硬件上的性能差异。

术语一般如何实现？如何使用？
Ascend 910 系列已演进至 910B/910B2/910C，搭配 CANN (Compute Architecture for Neural Networks) 软件栈和 MindSpore/PyTorch 框架。编程模型类似 CUDA，通过 Ascend C 编写自定义算子。主要使用场景：华为云/私有化部署的大模型训练和推理、MoE 模型的 Expert Pruning 部署（PreMoE on 910B2）、MindSpore 生态模型开发。910B2-64GB 相比 910-32GB 主要改进：翻倍的 HBM 容量、更高的内存带宽、更大的 on-chip buffer。

涉及论文标题：
- EfficientMoE: Optimizing Mixture-of-Experts Model Training With Adaptive Load Balance
- PreMoE: Proactive Inference for Efficient Mixture-of-Experts
