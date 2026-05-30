## MoE Lottery Subnetworks (Iterative Estimate-Prune-Finetune)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Lottery Subnetworks 是 Jaiswal et al. (2025) 提出的 expert-level sparsification 方法，受 Lottery Ticket Hypothesis (Frankle & Carbin, 2018) 启发，将传统 one-shot expert pruning 改进为迭代 **estimate-prune-finetune** 三阶段循环。核心创新：(1) 用 k 轮迭代替代 one-shot pruning，每轮重估 MC-Suite 准则；(2) 每轮 pruning 后插入 task-agnostic budget finetuning（next-token prediction on C4, 仅需 ~1M training tokens），校正 expert 丢弃导致的 sub-optimal 状态（负载偏斜 + 性能骤降）。

从算法pipeline角度拆解：

```
def moe_lottery(M, s, k, criterion, X_calib):
    droprate = s / k           # e.g., 50%/4=12.5%
    tokens = 0.2M              # finetuning budget round 1
    for r in range(k):
        # ESTIMATE: per-layer importance scoring
        drop_sets = {}
        for l in M.moe_layers:
            scores = [criterion(M, l, e, X) for e in remaining[l]]
            n_drop = int(n_experts * droprate)
            drop_sets[l] = argsort(scores)[:n_drop]
        
        # PRUNE: remove from router + delete weights
        for l in M.moe_layers:
            W_G.keep_mask[drop_sets[l]] = False
            for e in drop_sets[l]: del experts[e]
        
        # FINETUNE: task-agnostic next-token prediction
        opt = AdamW(M.parameters(), lr=1e-6)
        for batch in X_calib:
            if tokens_used >= tokens: break
            loss = cross_entropy(M(batch.inp), batch.lbl)
            loss.backward(); opt.step()
        tokens *= 2  # progressive: 0.2M→0.4M→0.8M→1.6M
    return M
```

**性能对比** (Mixtral-8×7B Base, C4 pp, full=7.44):

| % Dropped | One-shot Min-EAN | Iterative Min-EAN | MoE Lottery Min-EAN |
|-----------|-----------------|-------------------|---------------------|
| 12.5% | 7.95 | 7.90 | 7.89 |
| 50.0% | 14.74 | 10.44 | **9.76** |
| 75.0% | 30.59 | 17.39 | **13.05** |

关键发现：(a) MoE Lottery ≥ One-shot 3× 更优；(b) Finetuning 收益在 ~1M tokens 后饱和（Table 5）；(c) Finetuning 不显著改变 expert 选择（Figure 5b，MoE Lottery 与 Iterative 选择高度一致），但重调 router weights 实现负载 rebalance（Figure 6）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 配置：AdamW, cosine LR, max lr=1e-6, batch=8, 8×A100
- Progressive token schedule 最小化总计算；每轮重置 optimizer
- 建议在 Base model 上做 MoE Lottery pruning 再 instruction tuning（Instruct 上 finetuning 收益更小）
- 不修改 Serving 框架，pruned 模型可直接推理

涉及论文标题：
- Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations
