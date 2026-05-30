## Computation Reordering for MoE Verification（MoE验证阶段计算重排序）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Computation Reordering for MoE Verification 是 MoE-SpeQ Execution Engine 在 verify 阶段的内存局部性优化：在 target model 并行验证 k 个草稿 tokens 前，分析 ELB 按 expert_id 将 tokens 重新分组排序，使分配给同一 expert 的 tokens 连续计算。Expert weights 加载到 GPU cache 后由所有需要该 expert 的 tokens 连续消费，最大化 L1/L2 cache 复用率，减少 global memory traffic。

从kernel调度角度拆解术语：
```
# Naive: token-by-token, expert weights cache thrashing
for t in 1..k:
    for e in router(h[t]).topk():
        load(W_e) → compute FFN(h[t], W_e)
# expert weights 在 cache 中被频繁置换

# Reordered: expert-by-expert, maximal cache reuse
token_groups = group_by_expert(ELB, all_tokens)
for e, tokens in token_groups.items():
    load(W_e) → 一次 cache fill
    for t in tokens: compute FFN(h[t], W_e)
# L1/L2 cache hit rate 显著提升
```
Reordering 在 CPU 端完成（基于 ELB），按 expert 分组序列发射 GPU kernels。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
与 Fused MoE kernel 正交：fused kernel 解决 draft 阶段 kernel launch overhead；computation reordering 解决 verify 阶段 memory locality。论文消融未单独测量 reordering 收益。

涉及论文标题：
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts
