## Ring Attention（环注意力，含 zigzag 交错 / striped 分区变体）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ring Attention 是 Ring-style（环形）序列并行注意力 kernel/通信模式：把一条长序列按 token 均分到多个 GPU，GPU 排成逻辑环，每个 GPU 持有本地 token 的 Q 分片与 K/V 分片；每一步计算本地 Q 与当前 K/V 的局部 attention（含局部 softmax 统计量），然后把 K/V 传给下一个 rank、从上一个 rank 接收新 K/V，经过 H-1 轮（H=GPU 数）后每个 token 与完整序列的 K/V 交互完毕。计算与通信可重叠（计算当前块时异步发送前一块）。关键变体：(1) Striped Attention——把序列按等间隔 stripe 轮转分给各实例（每实例可对每个 KV shard 计算，缓解因果掩码负载不均）；(2) Zigzag Ring Attention（zigzag 交错）——把序列切成 2N 个 shard S0..S2N-1，实例 i 分配 (Si, S2N-i-1)，使各实例计算量相等（负载均衡）。Tetris（ISCA'26）在 prefill 阶段扩展 Flash Attention 支持 zigzag ring attention，并用 NVSHMEM 降低 ring 通信开销。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Zigzag Ring Attention prefill（N 个 SP 实例，序列分 2N shard）
rank i 本地持有 shard S_i 与 S_{2N-1-i} 的 Q/K/V
for step = 0 to N-1:
    o_i, lse_i = FlashAttention_zigzag(Q_local, K_cur, V_cur)   # 本地 partial attention（在线 softmax）
    Send(K_cur, V_cur) -> GPU_{(i+1)%N}; Recv(K_cur, V_cur) <- GPU_{(i-1)%N}   # P2P ring
    # 通信与下一轮计算重叠（NVSHMEM async put / NCCL P2P）
# 结束时各 GPU 已 attend 全部 K/V，合并 partial softmax 统计量得到精确注意力输出
```
Annotations: Q_local 为本地 token 的 query；K_cur/V_cur 为当前轮持有的 KV shard；o_i/lse_i 为 partial output 与 log-sum-exp（online softmax 用）；ring 步数=N；因果掩码下 zigzag 交错使各实例工作量相等。
Tetris 中的跨 chunk 扩展：每个 chunk 内用 zigzag 交错保证均衡；计算新 chunk 前把前序 chunk 的 KV cache 在 zigzag 布局下均匀重分布到当前实例组（每个 Pi 只需把后半段 KV 传给 P4+i），复用 ring communicator 与下一层 prefill 跨层重叠（cache balancing 隐藏）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：Ring Attention（Li et al. ACL'23）、ring-flash-attention 开源库、Striped Attention（arXiv:2311.09431）、ZigZag Ring Attention；Tetris 在 vLLM/PyTorch 后端扩展 Flash Attention 的 zigzag 变体 + NVSHMEM one-sided put 做 ring 传输（NVLink intra-node / InfiniBand inter-node），decoding 阶段 ring 只传 Q（体积小，减少通信）。使用场景：长上下文 prefill/decode 跨多 GPU 的上下文并行（CP/SP）；局限：ring 步数随 GPU 数线性增长、所有实例须同步开始（单实例延迟会拖累整环），SP 大小变化产生资源碎片（Tetris 的 CDSP 正是为解决此点）。Web 证据：Ring Attention 论文与 ZigZag 说明（上下文并行常见实现）。

涉及论文标题：
- Tetris: Efficient Long-context LLM Serving with Chunkwise Dynamic Sequence Parallelism
