## Megatron-Core（NVIDIA 大模型训练框架）

术语是什么？
Megatron-Core 是 NVIDIA 开发的用于大规模语言模型训练的分布式框架，是 Megatron-LM 项目的核心组件。它支持 TP（Tensor Parallelism）、PP（Pipeline Parallelism）、DP（Data Parallelism）、CP（Context Parallelism）和 EP（Expert Parallelism）的组合，并提供分布式优化器（Distributed Optimizer）、混合精度训练（FP8/BF16）等基础设施。Megatron-Core 已被用于训练 Nemotron、LLaMA 系列等工业级大模型。

从编译框架角度拆解术语：
Megatron-Core 作为训练框架/编译框架的角色：
1. **并行策略定义**：用户通过配置指定并行度 (tp_size, pp_size, cp_size, ep_size)，框架自动生成并行通信组并将模型参数分片到对应设备。
2. **模型图构建**：Megatron-Core 提供预构建的 Transformer Layer（含 Attention + MoE FFN），自动处理层内的并行通信（TP 的 AllGather/ReduceScatter、EP 的 All-to-All 等）。
3. **Pipeline Schedule**：支持 1F1B（one-forward-one-backward）和 interleaved pipeline scheduling，通过 micro-batch 实现计算-通信重叠。
4. **MoE Parallel Folding**：Megatron-Core 的核心创新之一，生成异构并行映射使 Attention 和 MoE 层可以独立配置并行策略。

术语一般如何实现？如何使用？
```python
# 使用 Megatron-Core 训练 MoE 模型
from megatron.core import parallel_state
# 初始化 5D 并行：TP=2, CP=1, EP=8, DP=8, PP=8
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=2,
    context_parallel_size=1,
    expert_model_parallel_size=8,
    pipeline_model_parallel_size=8,
)
# MoE Parallel Folding 自动处理 Attention/MoE 的异构映射
```
代码开源在 https://github.com/NVIDIA/Megatron-LM，MoE Parallel Folding 已于 2025 年初合入 main 分支。

涉及论文标题：
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core

---
