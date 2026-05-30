## Megatron-LM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Megatron-LM是NVIDIA开发的大规模语言模型训练框架，由Shoeybi et al.(2019)首次提出，经过Narayanan et al.(2021)增加pipeline parallelism和Korthikanti et al.(2022)增加sequence parallelism持续演进。它是一个基于PyTorch的分布式训练库，核心提供三种模型并行策略的高效实现：Tensor Parallelism（层内权重切分）、Pipeline Parallelism（层间流水线）、Sequence Parallelism（序列维度切分，通常与TP组合使用）。框架内部优化了通信模式（如all-reduce与计算的overlap）、activation checkpointing（选择性重计算）、以及混合精度训练（BF16/FP16）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
Megatron-LM的SSM支持（论文贡献）：
```
用户在Megatron-LM中训练Mamba-2-Hybrid的流程：
1. 配置模型 (model config YAML):
   - num_layers: 56
   - hidden_size: 4096
   - mamba_state_dim: 128
   - mamba_head_dim: 64
   - mamba_num_groups: 8
   - hybrid_attention_ratio: 0.071
   - hybrid_mlp_ratio: 0.50
   - use_rotary_position_embeddings: false

2. 配置并行策略:
   - tensor_model_parallel_size: 4
   - pipeline_model_parallel_size: 1
   - sequence_parallel: true

3. 启动训练:
   torchrun --nproc_per_node=8 pretrain_gpt.py \
     --tensor-model-parallel-size 4 \
     --sequence-parallel \
     --use-mamba \
     --mamba-version 2 \
     --hybrid-layer-pattern ...

4. 框架内部处理:
   - 构建Hybrid层pattern（Algorithm 1层分配）
   - 为Mamba-2层选择GroupNorm（而非LayerNorm）
   - TP通信：Mamba-2 1次all-reduce, Mamba 2次all-reduce
   - Mamba-2不支持sequence parallelism（仅支持TP+PP）
```

Megatron-LM本质上是一个模型库+训练循环框架（非编译器），负责分布式执行的正确性、通信编排和训练循环优化。

术语一般如何实现？如何使用？
代码：https://github.com/NVIDIA/Megatron-LM。论文的SSM分支：https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba。安装后通过pretrain_gpt.py脚本启动训练，使用--use-mamba, --mamba-version, --hybrid-layer-pattern等新参数。支持H100/A100 GPU集群，通过NCCL进行GPU间通信，支持BF16混合精度。

涉及论文标题：
- An_Empirical_Study_of_Mamba-based_Language_Models

---
