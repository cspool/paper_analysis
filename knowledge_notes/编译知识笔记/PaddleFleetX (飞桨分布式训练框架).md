## PaddleFleetX (飞桨分布式训练框架)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PaddleFleetX 是百度 PaddlePaddle 生态中的一站式大模型分布式训练框架（GitHub: https://github.com/PaddlePaddle/PaddleFleetX），支持 Data Parallelism、Model Parallelism（Tensor/Pipeline/Sharding）、Expert Parallelism、Hybrid Parallelism 等多种分布式策略的组合使用。MoESys 以 PaddleFleetX 为基础架构实现，在其上增加了 Hierarchical Storage、2D Prefetch、Elastic MoE Training、Hierarchical AlltoAll 等优化。根据 MoESys 论文 Table 1 的基准测试数据（截至 2023 年 3 月），PaddleFleetX 在标准 dense GPT 模型上的 training throughput 比 Megatron-LM 高 0.4%-14.2%（模型越大差距越小），且 GPU TFLOPS utilization 更接近理论峰值。

从编译框架角度拆解术语：
PaddleFleetX 在 MoESys 中的角色层次：
```
┌─────────────────────────────────────┐
│          MoESys (上层优化)            │
│  Hierarchical Storage, 2D Prefetch,  │
│  Elastic Training, Ring Memory...    │
├─────────────────────────────────────┤
│       PaddleFleetX (分布式框架)       │
│  Data/Expert/Tensor/Pipeline Parallel │
│  分布式训练调度 + 参数管理             │
├─────────────────────────────────────┤
│   PaddlePaddle (深度学习框架)         │
│  动态图执行 + JIT 静态图编译 + Kernel  │
├─────────────────────────────────────┤
│   CUDA + cuBLAS + cuDNN + NCCL       │
│   GPU Kernel + 通信原语               │
└─────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PaddleFleetX 提供了统一的分布式配置接口，用户只需在 YAML config 中指定 parallel strategy（如 `dp: 1, mp: 1, pp: 4, sharding: 4`），框架自动处理通信 op 插入和张量切分。
- 支持 GPT、BERT、ViT、Wide&Deep 等模型的分布式训练示例。
- 截至 2023 年 1 月最新 release 为 v2.4.1，项目活跃度中等（480 stars, 165 forks）。论文称 MoESys 代码将发布于 PaddlePaddle GitHub，但截至搜索未找到独立的 MoESys 仓库。
- MoESys 对 PaddleFleetX 的修改主要在：通信层（Hierarchical AlltoAll 替换标准 AlltoAll）、存储管理层（增加 Hierarchical Storage 和 CPU cache 管理）、训练调度层（增加 Elastic Training 的节点动态分配逻辑）、推理优化层（增加 Graph Optimization Pipeline 和 Ring Memory Offloading）。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
