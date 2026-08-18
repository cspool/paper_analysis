## 超图（Hypergraph，张量表示）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
超图是普通图的推广：一条超边可连接任意数量顶点。在稀疏张量场景中，每个非零元素用一条超边编码其全索引集（如 $A_{i,j,k}\ne0$ ⟺ 超边 {i,j,k}），顶点集=全部模式索引，从而完整保留张量的高阶稀疏结构（哪些索引组合非零）。划分工具如 KaHyPar（k-way hypergraph partitioning）在其上做最小边割+顶点数均衡划分。TensorPrism 指出超图作为张量表示的缺陷：(1) 坐标重叠（索引重叠=数据复用指示）不直接暴露——超边列出非零但无法直接回答"索引 i 与 j 共现几次"；(2) 传统超图划分只均衡顶点数与边割，而边数（=非零元素=关联计算量）代表实际工作量，工作量失衡；(3) 复杂度随张量阶数快速上升（量子模拟 90% 复用机会丢失）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
超图表示→划分→执行的 pipeline（HyperSB baseline 路线）：
```
edges = [set(idx) for (idx, val) in nonzero_coords]   # 每非零一条超边
part = KaHyPar_partition(vertices=all_indices, hyperedges=edges,
                         k=16, objective=min_edgecut)
# 各分区独立执行收缩; 缺陷: 工作量(边数)未均衡, 复用不在目标函数
```
超图与共现图的数学等价性：任意顶点对若共现则连边、边权=共现超边数；TensorPrism 用共现图保留超边语义（可恢复原非零元素）的同时暴露重叠。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：COO 坐标→超边集→超图划分库（KaHyPar，https://github.com/kahypar/kahypar）。使用场景：张量原生稀疏收缩划分（Gündüz et al.、HyperSB）。TensorPrism 以其为 baseline（HyperSB 平均 1.49× 更慢），并说明其仍优于 GSpTC/TCP 的原因：直接表示高维顺序、消除展开导致的中间数据膨胀。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
