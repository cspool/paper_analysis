## Parameter-Efficient Migration

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Parameter-Efficient Migration 是 HybridEP 中用于使 expert 传输更轻量级、从而支持更大 Expert Domain 和更好通信效率的优化技术组合。它由两部分组成：(1) SR-Based Expert Compression——将 expert 参数分解为 shared expert（所有 GPU 共享的公共知识）+ residual（每个 expert 特有的差异），仅传输压缩后的 residual（Top-k 稀疏格式），压缩比最高 50× 不损失精度；(2) Asynchronous Communicator——两阶段异步通信机制（Initialization 阶段：SREncode 与上一 iteration 的 optimizer step 融合；Asyn-comm 阶段：AG 通信与 pre-expert computation 重叠）。核心效果是减少 $P_E$ 的有效大小，使更多训练配置从 $2D < G \cdot P_E$（需混合 A2A+AG）转换为 $2D \ge G \cdot P_E$（纯 AG），从而扩大 Expert Domain 并实现更好的加速。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Asynchronous Communicator 的两阶段调度流程（以模型 = [(pre-expert, expert) pairs × N layers] 建模）：

```
┌── Initialization Stage (与 optimizer.step 融合) ──────────────┐
│  时间点: 上一 iteration 结束, optimizer 更新参数后              │
│                                                                │
│  for each MoE layer l (sequentially):                          │
│      for each expert in layer l:                               │
│          residual = expert - shared_expert                     │
│          compressed = TopK(residual, k = P_E/CR)  # SREncode  │
│          send_queue.push(compressed)                           │
│  注意: SREncode 与 optimizer.step() 融合, overhead 减少 ~30%   │
└────────────────────────────────────────────────────────────────┘

┌── Asyn-comm Stage (与 pre-expert computation 重叠) ────────────┐
│  时间点: 当前 iteration 的 forward pass 开始                    │
│                                                                │
│  stream_comm (与 stream_compute 并行):                          │
│      for each MoE layer l:                                     │
│          for each compressed_expert in send_queue[l]:           │
│              NCCL_AllGather(compressed_expert, domain_group)    │
│              recv_queue[l].push(all_compressed_experts)         │
│                                                                │
│  stream_compute (同时执行):                                     │
│      for each transformer block:                               │
│          if block is non-MoE:                                  │
│              attention + FFN (pre-expert computation)           │
│          if block is MoE:                                      │
│              # 同步: 等待 AG 完成                                │
│              experts = [SRDecode(c, shared) for c in recv_q]   │
│              tokens → gate → experts (no cross-DC A2A!)         │
│                                                                │
│  注意: SRDecode 与 expert FFN 融合, overhead 减少 ~45%          │
└────────────────────────────────────────────────────────────────┘
```

关键设计权衡：(1) Migration 不是免费的——编码/解码有计算开销（SREncode overhead < 专家大小相关，SRDecode overhead 见图 15），但通过融合可大幅降低；(2) 压缩比的 trade-off——50× CR 下 loss 不受影响（Figure 14），但更高压缩可能损害精度；(3) 存储开销——compressed residual 占用少，shared expert 需额外 GPU memory，可通过 CPU offloading 解决（ZeRO-Offload 兼容）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Expert compression 的核心机制是 residual decomposition——shared_expert = mean(all experts)，每个 expert 的 residual = expert - shared_expert。Residual 的分布更集中（Figure 9a "res" suffix），因为 expert 间的主要差异仅集中在少数参数上，这被 DeepSeekMoE 等 prior work 证实。
- Top-k 压缩保留 |residual| 最大的 k 个元素（含符号），以 value-index 稀疏格式传输。解码时通过 scatter(indices, values) 恢复残差，与 shared_expert 相加得到完整 expert。
- 异步通信器的 Send Queue/Recv Queue 类似于 pipeline buffer，管理压缩 expert 的流式传输。Send Queue 在优化器步骤后预先填充（Initialization），Recv Queue 在通信完成后填充（Asyn-comm）。
- 与其他 expert 传输优化：Janus 的 expert prefetching 仅优化 A2A 内的 expert 预取；HybridEP 的 AG-based expert migration 通过改变通信本质（AG 替代 A2A）+ 压缩 + 异步三管齐下实现更高效的 expert 传输。

涉及论文标题：
- HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission
