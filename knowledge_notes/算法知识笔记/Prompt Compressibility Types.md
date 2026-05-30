## Prompt Compressibility Types

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Prompt Compressibility 是 SPECPREFILL 通过实验发现的 prompt 对 token dropping 的不同响应模式分类。queries 分为三类：(1) Information-dense — 信息密集短 prompts，token dropping 效果差；(2) Compressible — 含大量冗余，删除大部分 token 后质量不下降；(3) Noisy — 删除部分噪声 token 后性能反而提升。

从算法pipeline角度拆解：

通过质量-保持率曲线分类：
```
compressible: quality(10%) ≈ quality(100%)     // 质量稳定
info_dense:   quality(10%) << quality(100%)    // 显著下降
noisy:        quality(50%) > quality(100%)     // 先升后降
```

术语一般如何实现？如何使用？

论文未给出自动分类算法（列为 future work）。实践中可让用户根据延迟/质量权衡决定保持率，或开发自适应策略动态调整。固定保持率已在 LongBench 多数类别中有效。

涉及论文标题：
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

---
