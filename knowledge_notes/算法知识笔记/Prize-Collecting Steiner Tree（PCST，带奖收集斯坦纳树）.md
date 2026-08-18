## Prize-Collecting Steiner Tree（PCST，带奖收集斯坦纳树）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PCST 是斯坦纳树问题的带奖变体：给定无向图 G=(V,E)，每条边带非负代价 $c_e$、每个顶点带非负奖 $\pi_v$，目标是找连通子图（树）$T=(V_T,E_T)$ 使 $\min_T(\sum_{e\in E_T}c_e+\sum_{v\notin V_T}\pi_v)$——最小化"连进来的边代价 + 未收集顶点的奖"。与必须连接给定终端的经典斯坦纳树不同，PCST 把"哪些顶点必须连"放宽为"每个顶点可选且给奖"，在连边代价与放弃顶点奖之间取平衡。它是大规模图系统组合优化的经典问题，广泛用于网络设计、设施选址等领域。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
TensorPrism 把张量 tiling 重述为"修改版 PCST"（CoGTP）：连通子图=一个分区，边权 W(e)（共现次数）对应"奖"（收集=复用收益），跨分区边（cut）对应"代价"，λ_b 负载均衡为额外约束。目标（式 6）：$\max\{\alpha\sum_i\sum_{e\in P_i}W(e)-\lambda_{cut}\sum_{e_{cross}}W(e)-\lambda_b\sqrt{\sum_i(\sum_{v\in P_i}(D(v)-W/N)^2)}\}$，α=2.0/λ_cut=1.0/λ_b=1.0。求解：BFS 多种子初始化（先收集高连通区）→ Kernighan-Lin 式边界单顶点迁移迭代（每步算 ΔF、保留正增益、ΔF<ε 收敛），每轮复杂度 $O(\sqrt{|V|}d)$（d=平均度 1-10），近线性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现（Algorithm 1）：输入共现图 G + 分区数 N，输出 N 个分区。三步：构造共现图（顶点=索引、边权=共现次数）→ BFS 初始化 → 迭代细化。参数选择依据：α=2.0 因高度数顶点提供 O(k²) 复用、O(k) 存储（2:1 收益比）；λ_cut=1.0/λ_b=1.0 匹配单位通信成本与 power-law 图负载分布系数。敏感性（Flickr）：α≤3 稳定、α>3 崩塌；λ_cut>1.5 后性能骤降；λ_b 在 1.0-1.5 达峰；(α=2,λ_cut=1,λ_b=1) 达 93.8% 峰值性能为保守最优。应用：FROSTT 8 数据集 + LLaMA 注意力张量的分区，N=16（PE 数）。

涉及论文标题：
- TensorPrism: Rethinking Sparse High-order Tensor Acceleration via Co-occurrence Graph
