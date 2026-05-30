## 5-D Hybrid Parallelism

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

5-D Hybrid Parallelism 是 Megatron-Core 中用于大规模 MoE 模型训练的**五种并行策略的任意组合**，具体包括：

1. **Tensor Parallelism (TP)**：将单层内的权重张量沿 hidden/column 维度切分到多个 GPU，每 GPU 计算部分结果后通过 AllReduce 合并。适合 Attention 层的 QKV 投影和 FFN 权重矩阵
2. **Expert Parallelism (EP)**：将不同 expert 的权重放置到不同 GPU 上，token 通过 All-to-All 通信路由到对应 expert 所在 GPU 计算后再返回。适合 MoE 层
3. **Pipeline Parallelism (PP)**：将模型按层切分为多个 stage，每个 stage 放置在不同 GPU 上，通过 micro-batch pipeline 流水线执行。配合 Virtual Pipeline Parallelism (VPP) 减少 pipeline bubble
4. **Context Parallelism (CP)**：将长序列沿序列维度切分到多个 GPU，减少单 GPU 的激活内存。配合 Ring Attention 或 Blockwise Transformers
5. **Data Parallelism (DP) with ZeRO-1**：每个 DP rank 持有完整模型副本但分片 optimizer states，处理不同 batch 数据后梯度 allreduce

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

五种并行的通信模式和切分方式：

```
# TP: 沿 hidden/column 维度切分
W [d_model, d_ffn] → shard along d_ffn → W_0 [d_model, d_ffn/tp], W_1 [...]
Forward: y_partial = x @ W_i  →  AllReduce(y_partial)
Backward: 同上，梯度沿相同维度 reduce

# EP: 按 expert 分配到不同 GPU
experts {E_0,...,E_7}, EP=4 → GPU0: {E_0,E_1}, GPU1: {E_2,E_3}, ...
Forward: AllToAll_Scatter(tokens) → ExpertCompute → AllToAll_Gather
通信量: 2 × total_tokens × d_model × sizeof(dtype)

# PP: 按层切分 stage
Layers {0..63}, PP=4 → Stage0: {0..15}, Stage1: {16..31}, ...
Forward: GPU0→GPU1→GPU2→GPU3 (send/recv activations)
VPP: 每 GPU 交替执行多个 virtual stage，填充 bubble

# CP: 沿序列长度切分
SeqLen=8192, CP=2 → GPU0: tokens[0:4096], GPU1: tokens[4096:8192]
Attention: RingAttention 交换 KV 块完成跨段 attention

# DP: 复制模型权重，独立 batch
B_total = dp_size × micro_batch_size
ZeRO-1: optimizer states 分片，梯度 AllReduce 后各 rank 更新自己的分片
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现于 Megatron-Core + NeMo：
- 用户指定 (TP, EP, PP, CP, DP) 和 global batch size
- Megatron-Core 自动建立对应的 NCCL process groups 和通信拓扑
- 关键约束：EP ≤ DP (传统)，但 MoE Parallel Folding 可打破此约束
- 实际使用以 TP×CP 不跨节点、EP 保持 NVLink 域内、PP 跨节点、DP 跨所有节点为最佳实践
- 已知性能：46.8% MFU (128 H100, Llama 3-E8T2), 49.3% MFU (Mixtral 8x22B)

涉及论文标题：
- Llama 3 Meets MoE: Efficient Upcycling
