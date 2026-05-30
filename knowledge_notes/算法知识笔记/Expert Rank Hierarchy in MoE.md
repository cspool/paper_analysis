## Expert Rank Hierarchy in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Rank Hierarchy 是 LYNX 通过实验揭示的 MoE router top-k 选择中不同 rank 位置的 expert 对输出质量的贡献不对称性。核心发现：top-1 (rank-0) expert 对输出质量具有决定性影响——deny top-1 expert 会导致 catastrophic accuracy drop；而 lower-ranked experts (rank 1, 2, ..., k-1) 高度冗余——deny 它们仅造成 minimal accuracy degradation。此外，当 cumulatively restore experts（先 restore top-1, 再加 top-2, ...）时，恢复 3-4 个 expert 后 accuracy 迅速接近 baseline，之后 diminishing returns。

这一层级结构是 LYNX 在设计时保留所有 token 的 top-1 expert（as "anchor"）、仅重映射 lower-ranked experts 的理论依据。发现该 hierarchy 在 GSM8K 和 HumanEval 两个性质完全不同的 task 上一致成立，表明它是 MoE computation 的结构属性而非 task-specific artifact。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LYNX Expert Rank Hierarchy 的验证实验（以 Qwen2-57B, k=8 为例）：

```
# 实验 1: Deny expert by rank（Figure 7）
for rank in range(k):  # rank 0..7
    # 强制将 batch 中所有 token 的 rank-r expert 替换为下一个候选
    for token in batch:
        deny expert at position rank in sorted top-k
        replace with next-best expert
    measure accuracy drop

结果: rank-0 (top-1) deny → catastrophic drop (>50% accuracy loss)
      rank-1 to rank-7 deny → <5% drop each

# 实验 2: Cumulative restore（Figure 8）
for n_keep in range(1, k+1):
    for token in batch:
        keep top n_keep experts, remap rest
    measure accuracy recovery

结果: 保持 3-4 experts → 恢复 ~95% baseline accuracy
      保持 5+ experts → 几乎完全恢复
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LYNX 利用 Expert Rank Hierarchy 设计 Adaptive Expert Scorer：高置信度 token 始终保留其 top-1 expert（确保 minimal accuracy impact），low-confidence token 的 lower-ranked experts 被安全地 remap 到 batch 的 reduced active expert set。这比 naive voting schemes（对所有 top-k 选择等权重投票）更有效地减少 active expert 数量而不损失 accuracy。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection
