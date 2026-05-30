## Quantization Codebook（量化码本）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
量化码本是预定义的向量集合，每个数据向量映射到码本中最近向量（量化向量），最近向量的索引即为量化码。码本大小 K 决定压缩率：log₂(K) bits 编码。关键约束：(1) 覆盖数据分布；(2) 结构支持高效最近搜索和距离计算。Extended RaBitQ 码本：G_r = {P·y/||y|| | y = (-(2^B-1)/2+u), u∈{0,...,2^B-1}}^D，大小 2^{B·D}，由 P 唯一确定，仅需存储 P。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
不同方法的码本对比：
- PQ: C₁×...×C_M 笛卡尔积，|C|=256^M
- SQ: {v_l+i·step}^D 网格，|C|=(2^B)^D
- RaBitQ (B=1): P·{±1/√D}^D, |C|=2^D
- Extended RaBitQ: P·{y/||y|| | y∈网格}, |C|=2^{B·D}
Extended RaBitQ 码本独特性：(1) D 维空间非填充；(2) 归一化保无偏性；(3) 随机旋转破对齐；(4) 整数网格支持高效内积。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RaBitQ/Extended RaBitQ 仅需存储 P，码本按需计算，节省指数级存空间。代码: https://github.com/VectorDB-NTU/Extended-RaBitQ

涉及论文标题：
- RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
- Practical and Asymptotically Optimal Quantization of High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor Search

---
