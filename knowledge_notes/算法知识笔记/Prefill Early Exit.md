## Prefill Early Exit

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Prefill Early Exit 是 YOCO 利用其 decoder-decoder 架构的计算依赖特性实现的 prefill 加速策略。在标准 Transformer 中，prefill 必须执行全部 L 层的前向计算。YOCO 的关键洞察是：Cross-Decoder 的 cross-attention 仅依赖 Self-Decoder 的输出 K̂,V̂，而 K̂,V̂ 在 Self-Decoder 完成后即可计算。因此 prefill 阶段只需执行 Self-Decoder（L/2 层）+ 生成 K̂,V̂，然后**提前退出**，无需执行 Cross-Decoder。生成的第一个 token 仍然是正确的，因为 Cross-Decoder 在 decode 阶段会逐步执行。这一特性质来自架构设计而非工程优化。结合 Self-Decoder 的高效 attention（线性复杂度），prefill 延迟从 O(LN²d) 降至 O(LNd/2)，在 512K context 上从 180s 降至 <6s。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Prefill Early Exit 的执行流程对比：

```
=== Standard Transformer Prefill === 
Input: x[1..N] tokens
for l in 1..L:
    X^l = DecoderLayer_l(X^{l-1})     # ALL L layers
    cache K^l, V^l for decode
Return: X^L (used to predict 1st token)
Latency: O(L * N² * d)  ← quadratic in N

=== YOCO Prefill with Early Exit ===
Input: x[1..N] tokens
for l in 1..L/2:
    X^l = SelfDecoderLayer_l(X^{l-1})  # ONLY L/2 layers
K̂, V̂ = proj_KV(X^{L/2})                # generate global KV cache
# === EXIT HERE ===
# Cross-Decoder NOT executed during prefill
# First token prediction uses Cross-Decoder in decode phase
Latency: O(L/2 * N * d) for gated retention  ← linear in N
         or O(L/2 * N * C * d) for sliding-window
```

**Annotations**: Prefill Early Exit 是安全的（sound），因为 Cross-Decoder 第 1 个 token 的 cross-attention 仅依赖位置 1 的全局 K̂,V̂（causal mask），而 K̂,V̂ 的位置 1 已在 Self-Decoder prefill 中完全计算。因此 prefill 退出时生成的第一个 token 与执行完整 prefill 的结果完全相同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Prefill Early Exit 适用于：(1) 长上下文首次 token 延迟（TTFT）优化——1M context 从 380s 降至约 5s；(2) 流式场景——快速响应用户的第一个 token；(3) 批处理 prefill——减少 per-request 的 prefill 计算量使 batch size 增大。实现上只需在推理引擎的 prefill 阶段跳过 Cross-Decoder layers。限制：仅适用于 YOCO 类架构（Self-Decoder 输出可直接生成全局 KV cache）；对短上下文加速比相对较小（32K 时约 2.87×）；decode 阶段仍需执行 Cross-Decoder（但仅对单 token 做 cross-attention，开销较小）。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)
