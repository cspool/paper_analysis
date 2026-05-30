## Extended RaBitQ (Multi-bit RaBitQ / B-bit RaBitQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Extended RaBitQ 是对原始 RaBitQ（仅支持 B=1 bit/dim）的扩展，支持任意 B bits/dim 的中等压缩率（B=4→8x, B=5→6.4x, B=8→4x）。核心创新是在 D 维空间（而非填充到 B·D 维）中构造码本：G = {-(2^B-1)/2 + u | u=0,...,2^B-1}^D，归一化后做随机正交旋转得 G_r = {P·y/||y|| | y∈G}。该码本继承 RaBitQ 的无偏估计器和误差界，误差随 B 指数衰减（经验公式 ε < 5.75·2^{-B}/√D，>99.9% 置信度）。理论证明仅需 B = Θ(log(ε^{-2}·log(1/δ)/D)) bits 达到误差界 ε，达渐近最优。编码算法通过枚举至多 D·2^{B-1} 个临界值找到最近码本向量，复杂度 O(2^B·D log D)。开源: https://github.com/VectorDB-NTU/Extended-RaBitQ

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Index 阶段: 码本 G_r = {P·y/||y|| | y∈G}
P = random_orthogonal_matrix(D)
for each o_r:
    o = (o_r - c) / ||o_r - c||, o' = P^{-1} o
    # Algorithm 1: 枚举 critical values 找最近码本向量
    t=0, v_max=0, 初始化 y_cur, ⟨y_cur,o'⟩, ||y_cur||
    while 存在未枚举的 critical value:
        t = next_min_critical_value()  # minheap, O(log D)
        更新 y_cur（仅一维变化, O(1)）, ⟨y_cur,o'⟩, ||y_cur||
        if ⟨y_cur,o'⟩/||y_cur|| > v_max:
            v_max = ⟨y_cur,o'⟩/||y_cur||, t_max = t
    ȳ = round(t_max · o')
    ȳ_u = ȳ + (2^B-1)/2 · 1_D  # B-bit 无符号整数向量
    # 拆分: ȳ_u = 2^{B-1}·ȳ₀ + ȳ_last（MSB=RaBitQ 二进制码）

# Query 阶段: 两阶段距离比较
q' = P^{-1} q, s = Σ_i q'[i]
# Stage 1: 仅用 MSB ȳ₀ (RaBitQ 二进制码)
⟨ȳ₀,q'⟩ = FastScan_SIMD(ȳ₀, q')
if lower_bound(dist_rough) > best: continue  # 剪枝
# Stage 2: 增量计算
⟨ȳ_u,q'⟩ = 2^{B-1}·⟨ȳ₀,q'⟩ + ⟨ȳ_last,q'⟩  # 复用 Stage 1 结果
⟨ō,q⟩ = (1/||ȳ||)·(⟨ȳ_u,q'⟩ - (2^B-1)/2 · s)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
C++ 实现，与 IVF 索引结合：(1) 数据集聚为 4,096 个聚类（百万级）；(2) 每聚类用本地质心中心化；(3) Algorithm 1 量化编码；(4) 分离存储 ȳ₀（MSB）和 ȳ_last。B=4 或 8 时复用现有 4/8-bit 整数内积实现；其他 B 通过拆分实现。B=5 时 >95% recall (6.4x 压缩)，B=7 时 >99% recall (4.5x 压缩)，均无需 re-ranking。代码: https://github.com/VectorDB-NTU/Extended-RaBitQ

涉及论文标题：
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
