## MoE Token Dispatcher（MoE Token 分发器）

术语是什么？
MoE Token Dispatcher 是负责在 EP/ETP 并行组之间路由 token 的运行时组件。在 MoE 层中，Router 为每个 token 分配 expert 后，Dispatcher 负责：(1) 将 token 从当前 rank 发送到持有对应 expert 的 rank，(2) 处理 ETP 组内的 activation 同步，(3) 计算完成后将输出 token 送回原 rank。MoE Parallel Folding 中的 Dispatcher 统一处理 ETP 和 EP 的任意组合，支持 token-dropping 和 token-dropless 两种训练范式。

从 kernel 调度角度拆解术语：
MoE Token Dispatcher 的前向计算流程（以 ETP=2, EP=2, 4 GPU 为例）：

```
Forward Pass 伪代码:
1. router_probs, router_indices = Router(local_input)  # 本地计算 gating
2. if token_dropping:
3.     # Sub-sequence dropping: 仅基于本地 logits 决策（零额外通信）
4.     capacity = CF * total_tokens / num_experts
5.     expert_counts = count_tokens_per_expert(router_indices)
6.     exceeded = expert_counts > capacity  # 标记超容量 expert
7.     # 丢弃超出容量的 token
8. 
9. permuted_tokens, permuted_indices = Permute(local_input, router_indices)
10. # EP 组内 All-to-All: 每个 rank 发送/接收 token
11. dispatched_tokens = AlltoAllV(permuted_tokens, EP_group)
12. # ETP 组内 AllGather: 确保 ETP rank 间 activation 一致
13. gathered_tokens = AllGatherV(dispatched_tokens, ETP_group)
14. # Expert GEMM 计算
15. expert_output = ExpertFFN(gathered_tokens)
16. # ETP 组内 ReduceScatter: 聚合切分的输出
17. scattered_output = ReduceScatterV(expert_output, ETP_group)
18. # EP 组内反向 All-to-All: token 返回原 rank
19. returned_tokens = AlltoAllV(scattered_output, EP_group)
20. output = Unpermute(returned_tokens, permuted_indices)
```

关键设计：
- **Sub-sequence dropping**：仅基于本地 sub-sequence 的 logits 做 token dropping，无需跨 rank AllGather 收集全局 logits（论文验证不影响模型收敛）
- **统一接口**：无论 Attention 层使用 TP/CP/DP 何种组合，Dispatcher 的输入始终是 token batch（通过 reshape 统一）
- **动态 tensor shape**：支持 EP 和 ETP 任意组合下的可变 token 数量

术语一般如何实现？如何使用？
- 在 Megatron-Core 中作为 MoE layer 的内部组件实现
- 使用 NCCL All-to-All-V（可变长度 all-to-all）、AllGather-V、ReduceScatter-V
- 配置方式：通过 capacity_factor 控制 token dropping（CF=1 用于 benchmark，dropless 模式用于训练）
- Backward pass 中 AG/RS 操作与 Forward 互换

涉及论文标题：
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core
- MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

**MoEBlaze 补充**：MoEBlaze 采用完全不同的 token dispatch 范式——**基于索引的轻量级路由**，不依赖 All-to-All 通信，也不需要 per-expert materialized token buffer。核心思路：Gate → TopK 选择后，构建四组轻量级 int32 索引数据结构替代 materialized buffer：(1) expert_token_indices[L×K]——按 expert 拼接的 token ID 列表；(2) expert_token_offsets[E+1]——每个 expert 的 token 起止位置；(3) token_expert_indices[L×K]——按 token ID 排列的 expert ID；(4) token_index_map[L×K]——每个 token 在 expert_token_indices 中的位置。Forward 中通过 on-the-fly gather（expert_token_indices 索引原输入张量）和 on-the-fly reduction（token_index_map 索引中间结果）完成专家计算。反向传播通过相同的逆向索引 scatter 操作将梯度直接映射，无需传统的 (L,d)→(L×k,d) 中间展开步骤。总内存开销仅 4×L×K×4 bytes（int32 索引）vs 传统方法的 L×K×d×2 bytes（bf16 materialized buffer），在 DeepSeek 规模下从约 94GB 降至约 16MB。此方法适用于单 GPU 训练场景（论文实验聚焦 H100 单卡），扩展至多 GPU 需处理跨设备索引映射。
