## Expert Piggybacking / Opportunistic Piggybacking (专家搭便车 / 机会式共享)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Piggybacking 是 OEA 框架的 Phase 2 核心机制。在 Phase 1 确定了 batch 中所有 token 的关键 expert 集合 S_base 后（这些 expert 的权重必须从 HBM 加载到 SRAM），Piggybacking 允许每个 token 将自己的次要 expert 选择（top-k0 之后排名较低的 expert）重新指向 S_base 中已存在的 expert——即"搭便车"使用已被其他 token 激活的 expert。这在不增加 T（唯一激活 expert 数）的前提下恢复了因 Phase 1 只使用 k0 < k 个 expert 而损失的模型质量。其本质是将"加载 expert 权重"的固定成本（b 项）在 batch 内 token 间摊销，但通过路由而非计算层面实现。

从系统架构角度拆解术语：
Piggybacking 的工作流程（以 Qwen3，k0=5，k=8，B=16 为例）：
```
Phase 1 后: S_base 约含35个expert (16个token各贡献5个，部分重叠)
Token i 的top-5: [e3, e17, e42, e88, e105]
Token i 的6-8位: [e22, e7, e91]

Piggybacking检查:
  e22 ∈ S_base? 是 → 附加到token i (免费, 不增加T)
  e7  ∈ S_base? 是 → 附加到token i (免费, 不增加T)
  e91 ∈ S_base? 否 → 跳过 (会增加T, 不附加)
  |S_i|已达7, 继续遍历token i的9位expert

最终token i激活: [e3, e17, e42, e88, e105, e22, e7]
T仍为 |S_base| = 35 (vanilla为48)
延迟从175.7μs降至136.0μs (k0=5时降低23%)
```

术语一般如何实现？如何使用？
- 关键约束（来自消融实验）：(1) k_max = k（piggybacking 不超过原始 expert 配额，超过 k 反而因过度分散而有害）；(2) maxP = N（不限制可 piggyback 的 expert 排名范围——即使是排名靠后、"out-of-policy"的 expert 也可能有用）；(3) 仅在 decode 阶段使用（prefill 已有足够的 token 数使 MoE 层 compute-bound）。
- 与 expert 并行扩展：在 expert parallelism 下，延迟由每台机器上最大 T 决定。可在每台机器独立做 piggybacking。
- 局限性：当 batch 内 token 来自相似分布时（如相同 benchmark prompt），S_base 较小 → piggybacking 可恢复的质量有限；当 token 分布多样化时（如 cross-entropy 的随机文本），S_base 较大 → piggybacking 恢复效果更好。

涉及论文标题：
- Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining
