## Expert Replication (in MoE Serving)

术语解释
Expert Replication 是在 Expert Parallelism 部署中为热门 expert 创建多个副本（replicas）的策略。通过将同一 expert 的多个副本放置在不同 GPU 上，避免热门 expert 所在 GPU 成为瓶颈，实现 token 负载的分散。

术语是什么？
在 MoE 推理中，不同 expert 被选中的频率差异很大——少数"热门 expert"（如 shared expert 或通用专家）处理了大部分 token。当 EP 将每个 expert 完整放在单一 GPU 上时，持有热门 expert 的 GPU 会因处理过多 token 而成为瓶颈。Expert Replication 通过创建热门 expert 的额外副本并分布到负载较轻的 GPU 上来缓解这一问题。EPLB 按比例复制：expert 处理的 token 越多，获得越多的 replicas。Replication 因子（如 1.5x）表示总 replicas 数 / 总 experts 数。

从系统架构角度拆解术语：
Expert replication 在 workload 下的效果：

```
=== No Replication (1.0x): 8 GPUs, 128 experts ===
每个 expert 仅在一个 GPU 上
问题: Expert 0 处理 30% tokens → GPU 持有 Expert 0 处理 token 数远超其他 GPU
→ EP load imbalance → 快 GPU 等待慢 GPU

=== 1.5x Replication: 8 GPUs, 192 replicas ===
热门 experts 有 2-3 个 replicas，冷门保持 1 个
热门 Expert 0 有 3 replicas → 分布到 GPU 0, 3, 5
token routing 将 Expert 0 的 tokens 分配到 3 个 replicas 上
→ compute-bound prefill 性能提升（-17% TTFT at 1.5x）

但副作用（METRO 揭示）:
→ memory-bound decode 下 activated experts 增加 ~30% at 1.5x
→ HBM → Tensor Core weight 加载量增加
→ decode latency 退化 +14% at 1.5x
```

术语一般如何实现？如何使用？
- EPLB 实现：根据滑动窗口内各 expert 的 token 计数确定 replica 数，容量约束为总 GPU 内存可容纳的总 expert 数
- 在 memory-bound decode 阶段，高 replication 因子反而有害（METRO 发现），需要在 prefill（受益于 replication）和 decode（受损害于 replication）之间权衡
- METRO 的策略：保持 replication 以优化 prefill，但通过 expert-minimizing token routing 消除 replication 对 decode 的副作用
- Replication 也增加内存压力——每个 replica 占用额外 GPU 内存，限制了可为 prefill 分配的 KV cache 空间

涉及论文标题：
- Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

---
