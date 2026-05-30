## KV Cache Error Accumulation (KV Cache 误差累积)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Cache Error Accumulation 指在稀疏解码过程中，由于每个 decode step 使用近似稀疏注意力而非精确 dense attention，生成 token 及其 KV cache 条目包含近似误差，这些误差被写入 KV cache 后随 decoding 步数持续累积的现象。传统 KV cache 假设每步由精确 attention 计算——稀疏解码打破这一假设：t₀ 时刻的近似 attention 产生带误差的 token₀，token₀ 的 K₀/V₀ 写入 cache；t₁ 时刻的 attention 基于不精确的 K₀/V₀ 做近似计算，产生更大误差的 token₁ 和 K₁/V₁，形成"误差累积"正反馈闭环。

这解释了为何 sparse decoding（如 Quest、InfLLM）的性能随解码长度增长而下降（ReSA Figure 1）：短解码时仅有少量 token 经过稀疏 attention，误差小；长解码时绝大多数 token 都经过稀疏 attention，误差逐级放大。

从算法pipeline角度拆解术语：

```
// KV Cache 误差累积的数学刻画
t=0 (prefill):  K_0 = K_dense, V_0 = V_dense   (e_0 = 0)
t=1:            token_1 = SparseAttn(q_1, K_0, V_0) + ε_1
                K_1 = K_0 ∪ {k_1 + ε_k1}
t=2:            token_2 = SparseAttn(q_2, K_1, V_1) + ε_2  (|ε_2| > |ε_1|)
t=T:            token_T = SparseAttn(q_T, K_{T-1}, V_{T-1}) + ε_T
                累积误差 ≈ Σ ε_i (单调增长)

// ReSA 方案: 每 f 步 dense rectification 限制误差窗口
if t % f == 0:
    K_t, V_t = DenseAttn(tokens[t-f:t], K_{t-f}, V_{t-f})
    max_error ≤ f · avg(|ε_i|)
```

术语一般如何实现？如何使用？

检测方法：对比 sparse vs dense decoding 在相同 prompt 下生成质量随 decode length 变化曲线。缓解方法：(a) ReSA 的 periodic dense rectification；(b) TriForce/MagicDec 的 self-speculation（sparse KV drafting + dense KV verification）；(c) Quest 的跳过前两层策略，但 ReSA 实验表明该方法改善有限。

涉及论文标题：
- Rectified Sparse Attention
