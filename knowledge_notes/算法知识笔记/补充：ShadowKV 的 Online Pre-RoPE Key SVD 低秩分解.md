## 补充：ShadowKV 的 Online Pre-RoPE Key SVD 低秩分解

ShadowKV 文献 Low-Rank Decomposition for KV Cache Compression 增加了 key cache 在线 SVD 分解的新方式——与传统方法对权重矩阵做离线分解不同，ShadowKV 对 **pre-RoPE key cache**（而非权重 $W_k$）做在线、prompt-dependent 的 SVD 截断分解。

具体发现：pre-RoPE keys 的奇异值衰减最快（比 post-RoPE keys、values、权重矩阵都要低秩），同一序列内 key 的低秩子空间高度共享（内序列相似度 ~0.8-1.0），不同序列间低秩子空间不同（跨序列相似度 ~0.2-0.4）。因此，对 pre-RoPE key cache 直接逐序列做 SVD（rank r=160 for d=128）比 data-independent 的 weight decomposition 更精准，实现 6× 压缩而无精度损失。

```
// ShadowKV Pre-RoPE Key SVD（online, per-sequence）
K = X @ W_k^T                    // pre-RoPE key, shape [s, d]
A, B = SVD(K, rank=r)            // A: [s, r], B: [h_kv, r, d]
// 低秩存储替代完整 K：
// GPU 存储: A [s, r] + B [h_kv, r, d]
// 解码时按需重建: K_selected = Gather(A, I) @ B → [k*c, d]
// 重建仅针对选中的 top-k chunk（~1.56% tokens）
```

ShadowKV 的低秩分解在 pre-filling 阶段完成，SVD 开销占比随序列长度递减（64K: 6.65%, 128K: 3.25%, 256K: 1.75%, 512K: 0.97%），因为 attention 计算为 $O(S^2d)$ 而 SVD 为 $O(Sdr)$。

涉及论文标题：
- ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference
- xKV: Cross-Layer SVD for KV-Cache Compression

**xKV 的扩展——从单层到跨层在线 SVD**：xKV 将 ShadowKV 的单层 pre-RoPE key SVD 扩展到**跨层**维度。与 ShadowKV 对每层独立做 SVD（Single SVD）不同，xKV 将多个相邻层的 pre-RoPE KV-Cache 水平拼接后做一次统一的跨层 SVD：concat([K_ℓ1, ..., K_ℓ{∣G∣}]) = U S V^T，提取跨层共享的左奇异向量作为共享基。xKV 同样在 prefill 阶段按请求在线执行，SVD 开销在 128K context 下 <10% prefill time。跨层 SVD 的关键优势：由于层间主导奇异向量高度对齐（由 CKA 验证），共享基比每层独立 SVD 更高效——相同压缩比下跨层 SVD 保留更多信息，相同 rank 下跨层 SVD 压缩率更高（≈ G× vs per-layer SVD）。xKV 还对 keys 和 values 分配不同 rank ratio（1:1.5），并对 pre-RoPE states 分解后重新施加 RoPE。
