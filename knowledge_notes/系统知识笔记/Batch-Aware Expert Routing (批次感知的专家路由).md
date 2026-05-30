## Batch-Aware Expert Routing (批次感知的专家路由)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Batch-Aware Expert Routing 是一类在 MoE 模型推理阶段根据整个 batch 的 token 分布动态调整每个 token 的 expert 选择的路由策略。与传统的 token-centric top-k routing（每个 token 独立选择 k 个 expert，不考虑其他 token 的选择）不同，batch-aware routing 利用 batch 内 token 间的 expert 重叠性来减少需要加载的唯一 expert 总数 T，从而在 memory-bound decode 阶段降低延迟。关键洞察：若某 expert 已被 batch 中至少一个 token 确认为关键（需加载其权重到 SRAM），则其他 token 可以"免费"使用该 expert，因为其权重已在 SRAM 中。

从系统架构角度拆解术语：
OEA 的 batch-aware routing 两阶段流程：
```
Phase 1 - Baseline Selection:
  对每个token i:
    选择其top-k0个最关键expert → S_i_base
  得到S_base = ∪_i S_i_base  (所有必须加载的expert)

Phase 2 - Piggybacking:
  对每个token i:
    对每个低优先级expert e_{i,j} (j > k0):
      若 e_{i,j} ∈ S_base (已在SRAM中):
        将e_{i,j}附加到token i (免费)
      直到 |S_i| = k (token i达到expert容量上限)
  
最终: T = |S_base| (仅Phase 1决定T大小)
     每个token仍激活≈k个expert (质量保持)
```

术语一般如何实现？如何使用？
- 实现：在 serving 框架（如 SGLang）的 MoE layer forward 中插入 batch-aware routing 逻辑。在 router 计算完所有 token 的 scores 后、实际 dispatch 前执行。
- 适用条件：batch size 中等（8-64），decode 阶段 memory-bound。prefill 阶段因有效 token 数大（prompt length × batch）已接近 compute-bound，不需此优化。
- 相关方法对比：Lynx (Gupta et al., 2024) 采用"减法"策略（先取所有 expert 的 union，再删除最不流行的），但可能误删对单个 token 关键的 expert。OEA 采用"加法"策略（先保证 baseline，再追加共享 expert），更安全。

涉及论文标题：
- Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining
