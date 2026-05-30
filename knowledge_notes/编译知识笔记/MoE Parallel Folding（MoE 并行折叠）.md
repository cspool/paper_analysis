## MoE Parallel Folding（MoE 并行折叠）

术语是什么？
MoE Parallel Folding 是 NVIDIA 在 Megatron-Core 中提出的一种异构混合并行映射策略，核心思想是将 Transformer 模型中 Attention 层和 MoE 层的并行策略**解耦**，允许每层独立选择最优并行配置，然后通过"折叠"(Folding) 将通信密集型并行维度映射到高带宽节点内网络（如 NVLink），减少跨节点通信开销。

在 MoE Parallel Folding 中：
- Attention 层使用四维并行组：TP × CP × DP × PP
- MoE 层使用另一套四维并行组：TP(ETP) × EP × DP(EDP) × PP
- 唯一约束：Attention 和 MoE 层的 PP group 数量和成员必须一致
- MoE 层的 EP 可以"折叠"到 Attention 层的任意子组中（如 TP/CP/DP），使得 EP 的 All-to-All 通信尽可能在 NVLink 域内完成

从编译框架角度拆解术语：
MoE Parallel Folding 在 Megatron-Core 框架中通过**并行组生成器**（parallel group generator）实现。其核心流程：

1. **并行组生成（generate_mappings）**：输入 world_size, tp, cp, ep, etp, pp 参数，为 Attention 和 MoE 分别生成两套独立的 rank 布局和通信子组：
   ```python
   attn_dp = world_size // tp // cp // pp
   moe_dp = world_size // etp // ep // pp
   attn_ranks = ranks.reshape(attn_dp, pp, cp, tp)
   moe_ranks = ranks.reshape(moe_dp, pp, ep, tp)
   ```

2. **层间转换**：从 Attention 到 MoE 层的转换仅需 reshape 操作——将 sequence/subsequence 展平为 batch of tokens，**无显式通信开销**。

3. **通信折叠**：由于 MoE 层的 EP 通信子组被折叠到 Attention 层的 TP/DP 子组内，当 EP × TP ≤ 节点内 GPU 数（如 8）时，所有 All-to-All 通信限制在 NVLink 内（450 GB/s），而非跨节点 InfiniBand（400 Gbps）。

4. **Token Dispatcher 集成**：框架中的 token dispatcher 接收 Attention 层输出（可能来自 DP/CP/TP 任意组合），统一处理为 token batch 并执行 EP/ETP 的分发和计算。

术语一般如何实现？如何使用？
在 Megatron-Core 中使用 MoE Parallel Folding 的方式：
- 通过配置文件指定 Attention 层的 (tp, cp, dp, pp) 和 MoE 层的 (etp, ep, dp, pp)
- 最优配置由性能实验确定：例如 Mixtral 8x22B 在 128 GPU 上使用 Attention TP=2, CP=1 和 MoE EP=8, ETP=1（即 MoE 层不做 TP，8 个完整 expert 分布到 8 GPU，最大化 GEMM 效率）
- 对 fine-grained MoE（如 G8T8），使用 EP 替代 ETP 可将通信占比从 >70% 大幅降低

涉及论文标题：
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core
