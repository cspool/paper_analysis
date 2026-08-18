## CoGTP（共现图张量划分算法，Co-occurrence Graph Tensor Partitioning）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CoGTP 是把稀疏高阶张量收缩的 tiling 形式化为共现图上修改版 PCST 划分的算法，三项目标并行优化：(1) 图内复用——同分区高权值边=高共现=高时间局部性（第一项 α·ΣW(e)）；(2) 跨分区通信——cut 边惩罚（λ_cut·ΣW(e_cross)）；(3) 负载均衡——加权度 D(v)=ΣW(u,v) 估计工作量、二次偏差惩罚（λ_b·RMS 项）。产出 N 个分区（N=PE 数），决定张量 A/B/C 的 tiling 因子。流程：构造共现图→BFS 多种子初始化→Kernighan-Lin 式迭代细化（只迁移边界顶点、算 ΔF、保留正增益、ΔF<ε 收敛），每轮 $O(\sqrt{|V|}d)$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Algorithm 1 概要
G = build_cograph(T)                      # 边权 w(u,v)=nnz(T[..;u;..;v;..])
D(v) = sum_u w(u,v)                       # 加权度=工作量估计
P = BFS_seed_clustering(G, N)             # 初始化
F = objective(P)                          # 式6
while True:
    best = None; max_gain = 0
    for v in boundary_vertices(P):        # 只在分区边界找候选
        for j in candidate_partitions(neighbors(v)):
            gain = delta_objective(v, i->j)
            if gain > max_gain: best=(v,i,j); max_gain=gain
    if best is None: break
    apply(best)                           # 单顶点迁移
    if objective(P) - F < eps: break
    F = objective(P)
```
例：尝试迁移 K1 被丢弃（ΔF 负）、迁移 K0 被保留（ΔF 正）（Fig.6）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
硬件实现：TensorPrism 共现图调度器的成本分析器（占调度器面积 62.9%）按式 6 评估分区质量，加法器汇总、分区索引缓冲存 PE 映射；λ_b 还作为运行时 DMUX 重分配（把局部 buffer 条目分给欠利用 MAC）的算法级前导（双级负载均衡：静态 CoGTP + 运行时 DMUX）。使用场景：FROSTT 8 数据集（密度 1e-14~1e-2）+ LLaMA 注意力张量。效果：2.22×/2.40×/1.71×/1.76×/1.49× 加速（vs SPADE/HotTiles/GSpTC/TCP/HyperSB），DRAM 访问降为 1/2.18/2.11/1.27/1.53，复用效率 67.86%（高 23.7%~57.4%）。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
