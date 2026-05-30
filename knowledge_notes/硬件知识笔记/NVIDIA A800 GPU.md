## NVIDIA A800 GPU

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NVIDIA A800 是 NVIDIA 面向中国市场推出的数据中心 GPU，基于 Ampere 架构（GA100），是 A100 的出口管制合规变体。A800-SXM4-40GB 版本配备 40 GB HBM2e 显存，内存带宽约 1,555 GB/s。与 A100 的主要区别在于 NVLink 带宽被限制为 400 GB/s（A100 为 600 GB/s），其他规格基本一致。NetMoE 在 4 节点 × 8 A800 GPU 集群上（共 32 GPUs）运行所有实验。

从硬件架构角度拆解术语：
A800 在 NetMoE 集群中的关键架构参数：
- 每节点 8 张 A800-SXM4-40GB，通过 NVLink 全互联（400 GB/s per GPU pair）
- 节点间通过 InfiniBand 互联（100 GB/s）
- HBM2e 带宽 ~1,555 GB/s（intra-device 内存拷贝约 2 TB/s，在 NetMoE 通信建模中被忽略）
- FP16/BF16 峰值算力 ≈ 312 TFLOPS（与 A100 相同）
- 每 GPU 40 GB 显存限制了大模型单卡部署，需要 expert parallelism 将 expert 分布到多 GPU
- NetMoE 测试的模型从 MoE-GPT-S（H=768）到 MoE-GPT-XXL（H=4096），均可在 40 GB 内完成单 expert 部分的计算

术语一般如何实现？如何使用？
- A800 与 A100 软件生态完全兼容（CUDA、cuDNN、NCCL、PyTorch），不需要特殊适配
- 在分布式训练中，A800 的限制主要在 NVLink 带宽（400 vs 600 GB/s），这意味着在相同 expert 分布下，A800 集群的 intra-node All-to-All 通信比 A100 慢 ~1.5×
- NetMoE 的实验结果（1.67× end-to-end speedup）是在 A800 上获得的，在 A100/H100 上可能因 NVLink 带宽更高而收益略有不同

涉及论文标题：
- NetMoE: Accelerating MoE Training through Dynamic Sample Placement
