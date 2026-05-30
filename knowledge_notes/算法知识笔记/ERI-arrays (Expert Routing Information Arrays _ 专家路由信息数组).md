## ERI-arrays (Expert Routing Information Arrays / 专家路由信息数组)

术语是什么？

ERI-arrays 是 PFT 数据结构中的元数据组件，包含四个数组，用于在 padding-free MoE pipeline 中追踪每个 token 的路由信息，使得 dispatch、MLP 和 combine 各阶段可以在无 zero-padding 的情况下正确操作。

四个 ERI-array：
1. **token_ids** [B]：token 在原始输入序列中的位置索引，用于 gather/scatter 操作
2. **expert_ids** [B]：token 被路由到的目标 expert 编号
3. **tokens_per_expert** [E]：每个 expert 分配到的有效 token 数量，驱动 uneven alltoall 和 Sequential GeMM 的切片
4. **combine_weights** [B]：gating 输出的概率权重，在 combine 阶段缩放 expert 输出

在 RBD 中还有扩展的 pilot/local replica ERI-arrays 和 s1_mapping_indices。

术语一般如何实现？

ERI-arrays 在 PFT construction 阶段生成（gating 之后、dispatch 之前），随后贯穿整个 MoE layer forward pass：dispatch 用 token_ids + expert_ids 做 gather → alltoall 用 tokens_per_expert 确定传输量 → MLP 用 tokens_per_expert 切片 → combine 用 token_ids + combine_weights 做 scatter。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
