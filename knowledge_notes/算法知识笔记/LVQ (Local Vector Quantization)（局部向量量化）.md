## LVQ (Local Vector Quantization)（局部向量量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LVQ 是 SQ 的改进变体，由 Aguerrebere et al. (VLDB 2023) 提出。不同于 SQ 使用全局范围，LVQ 为每向量独立确定 [v_l,v_r]，在各自范围内均匀量化。优势：(1) 对异常值更鲁棒——一个向量的极端值不影响其他向量精度；(2) 同 bit 数下精度优于 SQ。代价：每向量额外存储 2 个 FP32 (v_l, v_r)，对 D>300 的向量可忽略。在 Extended RaBitQ 论文中，LVQ 是最具竞争力的 baseline，但 Extended RaBitQ 在 B>6 时 error 仍比 LVQ 小 1.3x-3.1x。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Index
for each vector:
    v_l = min(o_r), v_r = max(o_r)
    step = (v_r - v_l) / (2^B - 1)
    for d = 1..D:
        code[d] = round((o_r[d] - v_l) / step)
    store: v_l, v_r, code[1..D]

# Query: o_approx[d] = v_l + code[d] * step
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现与 SQ 几乎一致，距离计算通过 B-bit 整数×浮点 SIMD 内积完成。与 IVF 索引结合使用。详见原 LVQ 论文（VLDB 2023）。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
