## Virtual Group Initialization (虚拟组初始化)

术语是什么？

Virtual Group Initialization 是 NVIDIA 为 fine-grained MoE upcycling 提出的 Router 和 Expert 初始化策略。在 fine-grained MoE 中，每个 expert 的 hidden size 被缩小为原始 FFN 的 1/G（G 为 granularity），需要 T 个 expert 的加权输出才能重建完整 dense MLP 的功能。普通的随机 Router 初始化无法保证初始的 top-T 选择恰好覆盖所有 G 个 shard，导致 upcycling 初始阶段 loss 极高且无法收敛。Virtual Group Init 通过将 Router 权重分组复制，确保每一个 expert group 内部初始权重相同，从而保证 top-T 的初始选择均匀分布在 G 个 group 上。

核心原理——两个保证：
1. **Router 分组**：N = E×G 个 expert 分为 G 个 group，每组 E 个 expert 都是同一个 dense MLP shard 的副本。Router 权重在组内初始化相同，组间可以不同。
2. **Top-T 均匀覆盖**：因为每组内 Router 权重相同，top-T 操作自然从不同组中选择 expert（当 T ≥ G 时），保证每个 shard 都被选到至少一次。

从算法pipeline角度拆解：

以 E2G2T2（4 experts, 2 shard, top-2）为例：

```
# 稠密 FFN: y = FFN_0(x) + FFN_1(x)  (分成 2 个 shard)

# === 错误做法 (Naive) ===
FFN = [FFN_0, FFN_1, FFN_0, FFN_1]  # 复制
router = random_init([0.4, 0.2, 0.3, 0.1])
router_top2 = [0.4, 0.0, 0.3, 0.0]  # 选 expert 0 和 2 (都是 FFN_0!)
# MoE 输出 = 0.4*FFN_0 + 0.3*FFN_0 != FFN(x)  ← 出错!

# === Virtual Group Init (正确做法) ===
# Group 0 (experts 0,1): 都是 FFN_0
# Group 1 (experts 2,3): 都是 FFN_1
router = [0.3, 0.3, 0.2, 0.2]  # 组内相同
router_top2 = [0.3, 0.3, 0.0, 0.0]  # 选 expert 0 (Group 0) 和 expert 1 (Group 0)
# 但这里只选了 Group 0! topK=2 不足以覆盖 G=2 组 → 需要 topK >= G

# === 实际使用情况 (E8G8T8) ===
# G=8 个 shard, topK=8 → 恰好每个 group 选 1 个
# 初始 MoE 输出 = (1/(E*G)) * (T/G) * dense_output ≈ dense_output / (E*G)
# Weight Scaling 补偿该缩放因子
```

术语一般如何实现？

在 Megatron-LM upcycling 模块中实现。先按 intermediate dimension 切分 dense FFN 权重 → 复制 shards → 构建 Virtual Group Router（每个 group 内复制相同权重）→ 应用 Weight Scaling。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts
