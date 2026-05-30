## Column-Group Bitmap (CGB)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CGB 是 ARB-LLM 提出的权重分区精炼策略。BiLLM 使用 column bitmap C_s 标记 salient 列和 group bitmap G 标记 non-salient 权重的 sparse/concentrated 分组，但 salient 列的 G 区域闲置。CGB 将 salient 列也纳入 group bitmap：G_s = 1_n C_s^T ⊙ G，G_ns = 1_n C_ns^T ⊙ G，产生四个 zone：salient-sparse、salient-concentrated、non-salient-sparse、non-salient-concentrated，每组独立二值化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Hessian 敏感度 → C_s (salient column bitmap) → C_ns = ¬C_s
# G = (|W| > threshold)  (magnitude-based group bitmap)
# 四个 zone masks:
G_s  = repeat(C_s, n, 1) ⊙ G          # salient-sparse & salient-concentrated
G_ns = repeat(C_ns, n, 1) ⊙ G         # non-salient-sparse & non-salient-concentrated
# 对每个 zone 独立执行 ARB: zone_mask → ARB(W, zone_mask, T)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
扩展 group 从 2→4 进一步改善性能（ARB-LLM_X ppl 6.55），但额外存储 ~0.8GB/7B 模型。CGB 是 ARB-LLM_X 和 ARB-LLM_RC 的共同组件，相比 BiLLM bitmap 有 consistent 提升。

涉及论文标题：
- ARB-LLM Alternating Refined Binarizations for Large Language Models

---
