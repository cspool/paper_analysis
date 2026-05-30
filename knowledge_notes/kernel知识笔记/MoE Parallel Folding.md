## MoE Parallel Folding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Parallel Folding 是 NVIDIA 在 Megatron-Core 中提出的异构混合并行策略，核心思想是**解耦 Transformer 中 Attention 层和 MoE 层的并行映射**。传统分布式训练中，所有层共享同一套并行配置（如 TP=2, PP=4），但实际上 Attention 层和 MoE 层的最优并行策略不同：Attention 层受益于高 TP/CP 处理密集序列计算，MoE 层受益于高 EP 处理稀疏 expert 计算。Parallel Folding 允许 Attention 层使用独立的 TP×CP×DP×PP 四维并行映射，MoE 层使用 Expert-TP×EP×Expert-DP×PP 四维并行映射。通过将通信密集型并行操作"折叠"到 NVLink 高带宽域内，减少跨节点通信开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Parallel Folding 的关键在于通信域的折叠映射：

```
# Attention 层并行配置: TP=2, CP=2
# → Attention 的 TP×CP group = 4 GPUs (同一节点 NVLink 域内)

# MoE 层并行配置: EP=8
# → MoE 的 EP group = 8 GPUs (1 节点或跨节点)

# Folding: Attention 的 TP×CP group (4 GPUs) "折叠"到
#           MoE 的 EP group (8 GPUs) 中
# → 将 Attention TP/CP 通信限定在 NVLink 高带宽域
# → 避免跨节点通信扩大到 Attention 层
```

实际训练配置示例（128 H100, 46.8% MFU）：

```
# 最优配置:
Attention: TP=1, CP=2  (TP×CP=2 GPUs, NVLink 域内)
MoE:       EP=8         (EP=8, 单节点 8 GPU 的 NVLink 内)
PP=4, VPP=8, DP 自动

# 效果:
- TP=1 避免 Attention 层的跨节点 TP 通信
- EP=8 最大化 expert 间并行
- CP=2 分担长序列内存压力
- PP=4 跨 4 个 pipeline stage
```

配置搜索调优实践（论文总结）：
1. TP 和 EP 保持在 NVLink 域内 —— TP/EP 每层都有通信，NVLink 带宽远超 InfiniBand
2. MoE 层 EP 性能优于 TP —— expert 独立计算，EP 仅需 All-to-All token dispatch
3. AllToAll-based token dispatcher 对 TopK=1-4 更高效（vs AllGather-based）
4. CP 配合 GQA 可重叠通信与计算，降低 KV cache 通信量
5. 跨节点扩展用 PP+DP，VPP 减少 pipeline bubble size
6. 早期训练阶段对 MoE 层启用 recomputation 缓解负载不均 OOM

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现于 Megatron-Core (https://github.com/NVIDIA/Megatron-LM)，通过 NeMo 框架调用：
- 为 Attention 和 MoE 层分别创建独立的 process groups
- 通信域映射：Attention TP/CP groups 被映射为 MoE EP group 的子集
- 传统限制 EP ≤ DP 被打破，允许 EP 独立设置
- 需要 nccl 支持灵活的子通信域创建

已知性能：Mixtral 8x22B 达 49.3% MFU, Qwen2-57B-A14B 达 39.0% MFU, Llama 3-E8T2 达 46.8% MFU (128 H100)。扩展至 1024 GPUs, 128K 序列长度。

涉及论文标题：
- Llama 3 Meets MoE: Efficient Upcycling
