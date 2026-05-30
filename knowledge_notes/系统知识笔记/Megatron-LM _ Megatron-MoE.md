## Megatron-LM / Megatron-MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Megatron-LM（Shoeybi et al., 2019）是 NVIDIA 开发的大规模语言模型分布式训练框架，核心贡献是提出了高效的模型并行策略——Tensor Parallelism (TP) 将单层 Transformer 的算子切分到多 GPU。后续版本扩展支持 Pipeline Parallelism (PP)、Data Parallelism (DP)、Sequence Parallelism (SP) 和 Expert Parallelism (EP)。Megatron-MoE 是 Megatron-LM 的 MoE 扩展版本，支持将 MoE 层的专家分布到多 GPU 进行专家并行训练。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

FOLDMOE 使用 Megatron-MoE (core_r0.9.0) 作为 baseline（无 overlapping 的 vanilla 实现）和底层训练框架：

```
Megatron-LM 并行策略组合 (FOLDMOE 使用):
┌─────────────────────────────────────────────────┐
│ Attention Layer: DP=2 × TP/SP=8                 │
│ MoE Layer:       EP=16 (每 GPU 1 expert)       │
│ Framework:       Megatron-LM + FOLDMOE mods     │
│ Communication:   NCCL 2.21.5 (A2A, all-reduce) │
│ Attention:       FlashAttention (fused kernel)  │
└─────────────────────────────────────────────────┘
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Megatron-LM 是工业界广泛使用的 LLM 训练框架：
- GitHub: https://github.com/NVIDIA/Megatron-LM
- 提供完整的 GPT、BERT、T5 等模型实现及混合并行策略
- 支持 FP16/BF16 混合精度训练
- 在 FOLDMOE 评估中，Megatron-MoE (core_r0.9.0) 作为 non-overlapping baseline（不做通信-计算重叠），FOLDMOE 在其基础上修改 Transformer block 的 forward 逻辑

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models

FasterMoE 基于 Megatron-LM 作为 baseline——修改其 MLP 模块用于 MoE 训练（替换 dense FC 层为 MoE MLP 层），并在其上集成 FastMoE 的 expert parallelism 实现。

FarSkip-Collective 在 Megatron-LM 中做了两项关键修改以支持异步通信重叠：(1) 使用 `torch.dist.all_to_all(async_op=True)` 替代同步 A2A，配合 CUDA Stream 分离通信与计算；(2) 实现 Stateful Async All-to-All Autograd Function——在 forward 和 backward 均使用 async_op，通过 backward hook 和 PyTorch autograd Sequence Number hijacking 实现反向传播的通信重叠。在 MI325X 8GPU 单节点上实现 88.4% A2A 重叠率（DeepSeek-V2 Lite）和 88.9%（DeepSeek-V3 L=6），端到端训练加速 1.11x 和 1.04x。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Tutel（Hwang et al., 2022）是 Microsoft 开发的 Adaptive Mixture-of-Experts 训练系统，发表在 PPoPP '22。核心特性包括：(1) MoE-only token-level overlapping——在 MoE 层内将 token 微批次的 A2A 通信与 expert 计算重叠；(2) 自适应 expert capacity 和 load balancing；(3) 与 Megatron-LM 兼容。Tutel 是 FOLDMOE 的主要对比 baseline，代表 SOTA token-level overlapping 方法。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Tutel 的 token-level overlapping（FOLDMOE 与其对比）：

```
Tutel 流水线 (MoE-only):
Stream 0: [Expert(b0)] [Expert(b1)] [Expert(b2)] [Expert(b3)]
Stream 1: [A2A-d(b0)] [A2A-c(b0)] [A2A-d(b1)] [A2A-c(b1)] ...
           ↑ 仅 MoE 层内的 overlapping

FOLDMOE 流水线 (Attention-MoE):
Stream 0: [Attn(b0)] [Attn(b1)] [Exp(b0)] [Attn(b2)] [Exp(b1)] [Attn(b3)] [Exp(b2)] [Exp(b3)]
Stream 1:            [A2A(b0) ..............] [A2A(b1) ..............] [A2A(b2) ...] [A2A(b3)]
           ↑ 整个 Transformer block 的 overlapping

关键差异: FOLDMOE 的 Stream 0 计算量大得多 (attention O(n²) + expert)，能更充分掩盖 Stream 1 的 A2A 通信
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Tutel GitHub: https://github.com/microsoft/tutel
- FOLDMOE 使用 Tutel v0.3.2 作为对比 baseline
- 实验设置中 Tutel 的 overlap degree d 从 {2,4,8,16} 搜索最优值
- Tutel 的 overlapping 受限于 expert computation 太小（32K seqlen 下仅占 21%），FOLDMOE 在 32K seqlen 上取得 1.17x 加速（GPT-MoE-M）
- FSMoE 对比 Tutel (w/ PipeMoE) 在 1458 配置 MoE 层上获得 1.18×–1.22× 加速，在真实模型（GPT2-XL MoE, Mixtral-7B, Mixtral-22B）上获得 1.19×–3.01× 加速。FSMoE 通过模块化支持 4 种路由函数，灵活的调度器支持 DP+MP+EP+ESP 混合并行下的全场景优化。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
