## MoE Data Shuffling (Token Dispatch-Combine)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Data Shuffling 是 Expert Parallelism 中由 token-to-expert routing 触发的全局数据重排过程。在 EP 下专家分布在不同 GPU 上，每个 token 经 router 分配给 top-k 专家后经历完整的 shuffle pipeline：token 按 destination rank 重排（permute）→ all-to-all 跨设备交换（dispatch）→ 按 expert layout 再次重排 → expert FFN 计算 → 对称的反向 all-to-all（combine）→ 恢复原始 token order。FUSCO 的 profiling 显示 shuffling 占端到端运行时的 22%–61%（随 EP degree 增长），其中 rearrangement（permute/repack）占 intra-node 延迟的 68.8%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 传统 MoE Data Shuffling (NCCL baseline，per MoE layer):
# Stage 1: permute by rank → Stage 2: A2A dispatch
# → Stage 3: permute by expert → Stage 4: FFN
# → Stage 5: inverse permute → Stage 6: A2A combine → Stage 7: inverse permute
# 共 5 次 memory passes（permute 各 read+write） + 2 次网络传输

# FUSCO fused approach:
descriptors = planner.build(token_expert_matrix)  # 一次性构建两级 descriptor
dispatched = dcomm.dispatch(tokens, descriptors)  # gather→ringbuf→RDMA，一步完成
expert_outputs = experts(dispatched)              # 直接消费已排列好的数据
output = dcomm.combine(expert_outputs, descriptors)  # 对称反向 fused 操作
# 仅 1 次 memory pass + 1 次 pipelined 网络传输（per dispatch/combine）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Expert-major layout（模型执行所需）和 device-major layout（通信操作所需）的 mismatch 是 root cause——每次 all-to-all 都需要一对对称的逆排列。Token size 4-14KB，足够大以 amortize per-unit transformation cost。FUSCO 通过 fused approach 消除所有显式 permute，在 16K seqlen real-world traffic 下比 NCCL 快 1.66×。

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
