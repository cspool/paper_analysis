## Batch Pruning for MoE Decoder (MoE解码器批量剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Batch Pruning（批量剪枝）是 MoE decoder 自回归推理中的一种动态优化技术：在 beam search 过程中，当某个句子生成 EOS（结束符）完成翻译后，立即将其从后续 decoder step 的计算中移除（prune），避免为已完成句子浪费 expert computation 和 weight loading。这是 MoE 特有的优化——在 dense 模型中，已完成句子仍会参与 batched GEMM 但开销可控（权重共享）；而在 MoE 中，每个 expert 的权重加载是 memory-bound 操作，为已完成句子加载 expert weights 是纯浪费。

论文 "Who Says Elephants Can't Run" 在 CUB radix sort token routing 的基础上实现 batch pruning：将已完成句子的 expert_idx 设为 INT_MAX（极大值），routing sort 自动将其排列到激活矩阵末尾，后续仅处理前 active_tokens 行。实现简单且高效，1.14× 加速。

从系统架构角度拆解术语：

Batch Pruning 在 MoE Decoder Beam Search 中的流程：
```
# 每个 decoder step (beam search iteration)
# batch_size=B, beam_size=K, 每个 beam 状态: active/done

for t in 1..max_output_len:
    # === Batch Pruning in Gating ===
    for i in 0..B*K:
        if beam[i].finished:                  # 已生成 EOS
            expert_idx[i] = INT_MAX            # 故意设极大值
            expert_scale[i] = 0
        else:
            expert_idx[i], expert_scale[i] = router_top1(hidden[i])
    
    # === Token Routing (CUB radix sort) ===
    # 排序后已完成句子自动排列到末尾
    sorted_order = radix_sort_by_expert_idx(expert_idx)
    hidden_perm = permute(hidden, sorted_order)
    
    # 仅对 active tokens 执行 expert GEMM
    active_tokens = count(not finished)       # 有效 token 数
    expert_offsets = compute_offsets(expert_idx[:active_tokens])
    # grouped GEMM 仅处理前 active_tokens 行
    for expert e with tokens:
        out_e = GEMM(hidden_perm[expert_offsets[e]:next_offset], W_e)
    
    hidden = unpermute(out, reverse_order)     # 恢复原始顺序
    # 已完成句子不参与输出采样（由 decoder 主循环处理）
```

为什么有效：MoE GEMM 是 memory-bound，weight loading 时间是主要开销。当 1/4 句子已完成，将节省约 1/4 的 weight loading bandwidth（对于活跃 experts）。加速效果随 batch 增大和句子完成时间分散而增加。论文实验 1.14× 加速（相对已高度优化的 baseline）。

术语一般如何实现？如何使用？

在 FasterTransformer 或类似推理框架中实现。关键点：(1) 需要与 token routing 机制配合（本文用 radix sort，也可以在 all-to-all 之前 filter）；(2) 需要对 MoE grouped GEMM 支持 varlen-M（可变 token 数）；(3) 需要正确处理未完成句子的 attention mask 和 beam search 的 beam 管理。通用化到 decoder-only LLM 时，可在 prefill 后对每个 request 维护 active flag，每次 decode step 检查并动态移除已完成的 requests。

涉及论文标题：
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production
