## Bi-Branch KV Cache (双分支 KV 缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Bi-Branch KV Cache 是 CSKV 提出的兼顾压缩效率和局部精度的 KV Cache 管理策略。将 KV Cache 分成两个分支：(1) 压缩分支（Compressed Cache）：存储全部历史 token 的低维压缩特征 $X A^K$，维度 $h_{comp}$；(2) 完整精度分支（Full-Precision Cache）：仅保留最近 $m$ 个 token 的完整精度 Key/Value，维度 $h_{out}$（默认 m=32）。Attention 计算时，历史 token 从压缩分支通过 $B^K$ 重建，近期 token 使用完整精度值，利用"近期 token 对下一 token 预测影响最大"的观察。

从算法pipeline角度拆解术语。

```
// Prefilling
K_full = X @ W_K              // (n, hout)
K_compressed = X @ A_K        // (n, hcomp)，→ Compressed Cache
K_local = K_full[-m:, :]      // (m, hout)，→ Full Cache

// Decoding
k = x @ W_K; k_comp = x @ A_K
Compressed_Cache ← [K_compressed; k_comp]  // (n+1, hcomp)
Full_Cache ← [K_local; k]                  // (m+1, hout)

// 重建
K_hat = Compressed_Cache[:(n-m), :] @ B_K  // 旧 token 重建
K_for_attn = concat([K_hat, Full_Cache])
// 维护: Full_Cache 移除最旧 token 保持 m
```

术语一般如何实现？如何使用？

窗口大小 m=32 为默认值（消融：m>32 后收益递减，m=32 Avg.Acc=0.92 vs m=4096 Avg.Acc=0.96）。更大 m 使更多 token 以完整精度存储（降低压缩效果），需权衡 m 与内存预算。可与量化叠加使用。

涉及论文标题：
- CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios
