## Flash-Decoding / Split-KV（分KV解码并行化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Flash-Decoding（或 Split-KV）是一种针对 LLM 推理解码阶段的 attention 并行化策略。解码阶段 query sequence length 极短（通常1个token），而 KV cache 极长（数千tokens），attention 变为 memory-bound——瓶颈不是 tensor core 计算吞吐而是 KV cache 的 HBM 加载带宽。此外，FlashAttention-2/3 的默认算法沿 query sequence length 维度并行化（不同 CTA 处理不同 Q tiles），解码时 query length=1 导致 parallelism 不足。Flash-Decoding 的核心思想：将 KV cache sequence length 维度分割为多个 segments，不同的 threadblocks 各自加载同一个 Q tile 和不同 KV segments，独立计算局部 attention output O_k 和 log-sum-exp lse_k，最后通过 separate reduction kernel 合并：O = Σ_k O_k × exp(lse_k - lse_max) / Σ_k exp(lse_k - lse_max)。

从系统架构角度拆解术语：
FlashAttention-3 中 Flash-Decoding 的执行流程（inference, query_len=1, KV_len=8192）：
```
// Split parameter n (heuristic at launch, e.g. n=4)
// Phase 1: Split attention kernel (n threadblocks)
for k in 0..n-1:
    Threadblock k loads:
      - Q tile: same for all k (query_len=1, B_r=1)
      - KV segment k: K/V[k*B_c_seg : (k+1)*B_c_seg]
    Compute local attention:
      S_k = Q × K_k^T
      P_k = softmax(S_k)  // local softmax, may not be globally correct
      O_k = P_k × V_k
      lse_k = rowmax(S_k) + log(rowsum(exp(S_k - rowmax)))
    Write O_k, lse_k to HBM

// Phase 2: Reduction kernel (1 threadblock)
lse_max = max(lse_0, ..., lse_{n-1})
O = Σ_k O_k × exp(lse_k - lse_max)
O = O / Σ_k exp(lse_k - lse_max)
write O to HBM
```
FlashAttention-3 的 enhanced Flash-Decoding 还支持 early exit——如果某 KV segment 中所有 score 低于阈值（如 causal mask 导致），该 threadblock 写 lse = -∞ 并跳过后续计算。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Flash-Decoding 最早在 PyTorch blog (Dao et al., 2023) 中描述，已集成到 FlashAttention 2.6+ 和 vLLM。FlashAttention-3 的 inference 实现在此基础上增加：(1) GQA packing——将多个 query heads 打包到同一 threadblock 的 Q tile 中（利用 WGMMA first operand width=64 的大粒度），实现 N× speedup（N=GQA ratio）而无需修改 kernel loop 结构；(2) PagedAttention with TMA——使用自定义 SM90_TMA_LOAD_PAGED_OP class 和基于 virtual shape 的 tensor map descriptor，block table 通过额外参数传入 TMA copy method。Flash-Decoding 的 split parameter n 由 heuristic 在 kernel launch 时确定——取决于 KV cache length 和 GPU 的 SM 数量。

涉及论文标题：
- FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision
