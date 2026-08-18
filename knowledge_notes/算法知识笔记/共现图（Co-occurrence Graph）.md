## 共现图（Co-occurrence Graph）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
共现图把稀疏高阶张量变换为加权无向图 CoG(V,E)：每个顶点是一个张量索引（r 阶张量 $A_{I,J,K}$ 有 |V|=I+J+K 个顶点），任意两个索引若在同一非零元素（超边）中共现则连边，边权等于共现次数 $W((i,j))=nnz(A[..;i;..;j;..])$（式 2）。完整子图（clique）对应一个非零张量元素；每个非零指示一次类似 SpMM 的向量乘操作。它是超图的 pairwise 投影（数学等价但显式暴露索引重叠），用 CSR 存储足迹 $M_{CSR}=(\sum|r|+1)\times4+E\times4$ 字节（FP32，E 为不重叠边数，$E \ll N\binom{r}{2}$）。基于它可量化张量收缩的复用（式 3）：contraction 模式共享顶点→输入复用（B[K,:] 被多个输出目标复用），free 模式共享顶点→输出复用（同一 C[I,J,:] 聚合多个输入），把 tiling 变成可求解的图划分问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
构建与使用 pipeline：
```
# 构造
for (idx, val) in nonzero_coords:       # 每个非零
    for (u,v) in pairs(idx):            # 全索引对
        W[u,v] += 1                     # 共现计数(去重)
G = CSR(V, E, W)
# 使用: 复用量化 + 划分 + 数据流
Reuse = sum(D(v))/|f1| + sum(D(v))/|c| - ...   # 式3, D=加权度
P = CoGTP_partition(G, N)               # 式6 PCST 式划分
# 执行: 沿图遍历 push/pull
```
例子：$2\times2\times2$ 张量生成 6 顶点图，$W(I_0,K_1)=2$ 表示两索引在 2 个非零中共同出现；三元组 (I0,J0,K0) 构成 clique ⟺ A[I0,J0,K0] 非零。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现（TensorPrism CoG Scheduler 的权重计算单元）：Coordinate Parser 提取索引对、Dimension Pair Selector 选维度对、Hash-based Engine 去重、Index Pair Buffer（FIFO，可配深度如 256）缓冲、图生成器构造 CSR 共现图、成本分析器评估式 6 划分质量。使用场景：任何高阶稀疏张量收缩（LLM 注意力、科学计算、推荐、量子模拟），密度跨 14 个数量级。局限性：边权只统计共现次数不区分输入/输出模式；图构建有预处理开销（较 SPADE/HotTiles/GSpTC 增 8.0%/6.7%/4.2%，远低于 TCP 25.4%）；存储较超图平均增 3.0%（非瓶颈）。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
