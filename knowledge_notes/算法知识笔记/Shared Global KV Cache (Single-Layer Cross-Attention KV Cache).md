## Shared Global KV Cache (Single-Layer Cross-Attention KV Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shared Global KV Cache 是 YOCO 的核心创新组件。在传统 decoder-only Transformer 中，每层 decoder 独立计算并存储自己的 KV cache。YOCO 将全局 KV cache 的生成和使用分离：Self-Decoder 的最终输出 M=X^{L/2} 经过一次线性变换生成全局的 K̂=LN(M)W_K 和 V̂=LN(M)W_V，然后所有 Cross-Decoder 层（L/2 层）的 cross-attention 共享这组 KV cache。这意味着 KV cache 总量从 O(L×N×d) 降至 O(N×d+L×C×d) ≈ O(N×d)（C 为 Self-Decoder 的常量 memory），约节省 L 倍。全局 KV cache 与 GQA 兼容（K̂,V̂ 使用较少的 KV heads），可进一步压缩。该设计将"缓存一次"（cache once）的概念实体化为架构组件。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Shared Global KV Cache 的生成和使用流程：

```python
# === Generation (once per sequence) ===
M = SelfDecoder.forward(X_0)              # output of last self-decoder layer
M_norm = RMSNorm(M)                       # pre-normalization
K_hat = M_norm @ W_K                      # global key cache, [N, d_k]
V_hat = M_norm @ W_V                      # global value cache, [N, d_v]

# === Cross-Decoder Usage (all L/2 layers reuse) ===
for l in range(L//2):
    X_l_norm = RMSNorm(X_l)
    Q_l = X_l_norm @ W_Q_l                # per-layer fresh query
    # Standard cross-attention with SHARED K_hat, V_hat
    A = softmax(Q_l @ K_hat.T / sqrt(d_k) + causal_mask)  
    O = A @ V_hat
    X_{l+1} = SwiGLU(RMSNorm(O + X_l)) + (O + X_l)
```

**Annotations**: K̂,V̂ 的维度：d_k = d_v = d_head × h_kv（GQA 时可减少 h_kv）。causal_mask 确保 cross-attention 也是因果的（t 位置只能关注 ≤t 的全局 context）。K̂,V̂ 的生成仅需一次 O(N×d²) 的矩阵乘法，相比 L 层的 KV cache 生成节省 L×计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Shared Global KV Cache 的实现和使用场景：(1) 长上下文推理——单层 cache 使 65B 模型 128K token 仅需 1GB KV cache（vs Transformer 需数十 GB）；(2) Pre-caching for RAG——可以预先计算并缓存文档的 K̂,V̂，查询时直接复用；(3) 与 GQA 结合——减少 K̂,V̂ 的 head 数进一步节省（YOCO-3B 使用 h_kv=8 vs h_q=24）；(4) 分布式推理——K̂,V̂ 可以存储在一台 GPU 上，Cross-Decoder 分布到多 GPU 各自计算 Q_l。限制：K̂,V̂ 本质上是 Self-Decoder 的输出投影，其表达能力受限于 Self-Decoder 的质量和层数。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)
