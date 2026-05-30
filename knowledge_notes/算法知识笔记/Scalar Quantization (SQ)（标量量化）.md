## Scalar Quantization (SQ)（标量量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Scalar Quantization (SQ) 是向量量化的最简形式：对每个维度独立量化。收集所有向量的全局 min v_l 和 max v_r，范围 [v_l,v_r] 均匀分为 2^B-1 区间，每浮点数取整到最近区间边界，存为 B-bit 无符号整数。压缩率 32/B（B=8→4x, B=4→8x）。距离计算通过 B-bit 整数与浮点向量做内积（SIMD），无需查表，效率显著优于 PQ。缺点：(1) B<4 时精度极差；(2) 全局范围对异常值敏感。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Index 阶段
v_l = min(all vectors all dims), v_r = max(all vectors all dims)
step = (v_r - v_l) / (2^B - 1)
for each vector:
    for d = 1..D:
        code[d] = round((o_r[d] - v_l) / step)

# Query 阶段
# 通过 SIMD 计算 ⟨code, q⟩ 内积（整数×浮点）直接估计距离
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
广泛部署于 Faiss (`IndexScalarQuantizer`)、Milvus、SPANN 等，常与 IVF 结合，B=8 为典型配置。AVX512 SIMD 批量计算。是中等压缩率最常用的方法。

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
