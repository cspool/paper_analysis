## Stabilizers (in KV Cache Eviction) / 稳定器

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Stabilizers 是 LOCRET 中用于缓解 chunked prefill KV cache eviction 导致的上下文不连续性的机制。在每次 chunked prefill 步骤中，当前 chunk 的最后 $n_s$ 个 token 的 CIS 被强制设为 $+\infty$，确保它们永不被 evict。这些保留的 token 作为 "稳定器"，在下一 chunk 的处理中提供局部、连续的上下文锚点。

其设计动机：KV cache eviction 造成 tokens 在位置上的不连续（某些位置的 cache unit 被移除），导致后续 token 接受不连续的上下文。这种不连续性导致 retaining head 的 CIS 预测不稳定——因为 retaining head 的输入 [Q, K, V] 受到不连续 KV cache 的影响——进而导致 eviction 错误被放大，最终使 hidden states 产生大量误差。Stabilizers 通过保证最近的 $n_s$ 个 token 始终存在，阻止了这一错误传播链。

从算法pipeline角度拆解术语。

**Stabilizers 在 LOCRET 中的伪代码（Algorithm 1）**：

```
for chunk in chunk_positions:
    K_chunk, V_chunk, score_chunk = M(x[begin:end], K_cache, V_cache)
    K_cache = Concat(K_cache, K_chunk)
    V_cache = Concat(V_cache, V_chunk)
    score_cache = Concat(score_cache, score_chunk)
    
    if chunk is not the last chunk:
        // === Stabilizers 机制 ===
        score_cache[score_cache.length - n_s : score_cache.length] = +inf
        // 最后 n_s 个 token 的 CIS 被设为无穷大，永不被 evict
    
    indices = top-b(score_cache).indices  // top-b 中必然包含 stabilizers
    K_cache, V_cache, score_cache = K_cache[indices], V_cache[indices], score_cache[indices]
```

**Stabilizers 的消融实验**（LOCRET Figure 3）：
- $n_s = 0$（无 stabilizers）：R.Number 准确率 0%，模型完全失败
- $n_s$ 较小时（如 500-1000）：严重性能退化
- $n_s = 2500$（默认值）：准确率恢复正常
- 原因（Figure 3b-c）：短 stabilizers 或无 stabilizers 导致最后 hidden state 的最大绝对误差和各层 CIS 预测的 mean absolute error 显著增大

术语一般如何实现？如何使用？

Stabilizers 是 eviction 策略中的一个简单机制——将固定数量最近 token 的 score 设为极大值。它不需要额外计算，仅修改 score_cache 的部分值。在 LOCRET 中默认 $n_s = 2500$，同时另有 $n_{loc} = 100$ 个 local token 在最后处理且永不被 evict。Stabilizers 的数量需在 "保持上下文连续性" 和 "留给其他重要 token 的 budget 空间" 之间平衡（太大会压缩可用 budget 导致性能退化，Figure 5b）。

涉及论文标题：
- LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices

---
