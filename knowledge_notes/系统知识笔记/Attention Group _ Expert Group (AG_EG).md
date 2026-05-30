## Attention Group / Expert Group (AG/EG)

术语解释
Attention Group (AG) 和 Expert Group (EG) 是 Disaggregated Expert Parallelism 中 GPU 资源的两个专用分区。

术语是什么？
AG 负责存储和处理 Transformer block 中需要所有 tokens 参与计算的标准组件：Self-Attention layers（MLA 或 MHA）和 Shared Expert（如有）。AG 内参数全复制到每个 GPU，每个 AG 设备独立处理其分配的 micro-batch tokens，无需 group 内 collective 通信。EG 负责存储和处理 sparse MoE routed experts，E 个 experts 分布到 eg 个设备上（每个设备持有 E/eg 个 experts）。由于 token-to-expert routing 的稀疏性，每个 token 仅访问 top-k 个 expert，这些 expert 天然位于单一设备上（每个 expert 不跨设备切分），因此 EG 内也无需通信。

从系统架构角度拆解：
AG/EG 的大小分配直接影响推理吞吐：(1) ag 越大，attention 计算越快（更多 GPU 并行），但留给 EG 的设备越少，每个 EG 设备持有更多 experts；(2) eg 越大，expert 计算越快，但 AG 资源减少。最优 ag:eg 比值取决于模型的 attention-to-expert 计算比和硬件特性。DeepSeek-V2（shared expert 模型）推荐 (ag=3, eg=5) 或 (ag=8, eg=24)；Qwen3-MoE（无 shared expert）推荐 (ag=4, eg=4)。AG 内复制 attention+shared expert 参数的显存开销需满足 GPU memory 约束。

术语一般如何实现？如何使用？
实现方式：(1) 使用 PyTorch Distributed 的进程组 API 创建两个独立的 NCCL communicator group——AG communicator（ag 个 rank）和 EG communicator（eg 个 rank）；(2) 模型加载时，所有 AG rank 加载完整的 attention+shared expert 参数，每个 EG rank 仅加载其分配的 E/eg 个 expert 参数；(3) A2E/E2A 通信跨越两个 communicator group，使用 inter-group NCCL P2P 操作。

涉及论文标题：
- Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism

---
