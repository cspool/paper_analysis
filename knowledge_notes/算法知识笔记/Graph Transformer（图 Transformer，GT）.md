## Graph Transformer（图 Transformer，GT）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Graph Transformer 是把 Transformer 架构（Vaswani et al., 2017）应用到图结构数据的模型范式：把顶点视为输入 token，用全局（all-to-all）自注意力捕获长程依赖与复杂结构交互，突破传统 GNN message-passing 的局部感受野限制（后者受 over-smoothing、over-squashing、表达力有限与扩展性差困扰）。与标准 Transformer 的区别：(1) 输入是无序顶点集（非固定顺序 token 序列），顶点处理顺序可灵活重排；(2) 顶点特征维度高（D_feat=100–1000）；(3) 常把图结构编码（structural encoding）融入输入特征与注意力矩阵。
- 论文给出的通用更新规则（式1）：对顶点 v，H^v = concat({h_u^l | u ∈ N(v)})（N(v) 为 v 的注意力邻居集，全连接注意力下即全图顶点）；h̄_v^{l+1} = softmax(h_v^l W_Q (H^v W_K)^T / √d_K) · H^v W_V；h_v^{l+1} = FFN(h̄_v^{l+1}) + h̄_v^{l+1}。以 Graphormer 为例：初始嵌入 h_v^{(0)} = x_v + z^-_{deg^-(v)} + z^+_{deg^+(v)}（入/出度可学习嵌入），注意力系数加最短路径距离偏置 bias_{φ(v,u)}。全局注意力使 GT 表达力优于 GNN，但代价是 O(N²) 计算与中间数据。
- 论文评估的 GT 模型（Table III）：Graph Transformer [Dwivedi & Bresson 2020]（4 层、hidden 128、12 head）、Graphormer [Ying et al. 2021]（4 层、hidden 768、8 head）、UGformer [Nguyen et al. 2022]（4 层、hidden 384、4 head）、Edge Transformer / EGformer [Bergen et al. 2021]（8 层、hidden 200、4 head）。profiling（Fig.2）显示 attention 占 67.08%、FFN 占 24.53% 执行时间（合计 91.61%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一个图 Transformer 层的推理 pipeline：① 输入顶点特征 x_v + 结构编码（如度嵌入/最短路径偏置）→ ② Q/K/V 投影（h_v W_Q、H^v W_K、H^v W_V）→ ③ 全对注意力分数 QK^T/√d_K（+ 结构偏置）→ ④ softmax 归一化 → ⑤ value 加权聚合（H^v W_V）→ ⑥ FFN + 残差 → 下一层。全连接注意力下第③步是 N×N 稠密矩阵，是 O(N²) 复杂度与 O(N²) 中间注意力矩阵的来源。
- 与 GNN 的关键差异（Fig.1）：(1) 局部 message-passing vs 全局注意力；(2) 隐式拓扑传播 vs 显式结构编码。因此现有 GNN 优化/加速器（SpMM、CSR 遍历）不能直接支持 GT。
- KV caching 不适用于 GT（论文 Q2 分析）：GT 处理无序顶点集的动态全对交互，无固定顺序的"过去 token"依赖，无法套用 LLM 的 KV Cache；但 TDS 的确定性稀疏依赖结构可做缓存/复用（TDS-CSR）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 软件实现：通用框架如 DGL（保留 O(N²) 全局注意力）、TorchGT（dual-interleaved 稀疏注意力，依赖 Hamiltonian path）、PyG；TAGT-S 用 TDS 稀疏化修改 DGL。评估数据集（Table II）：Yelp（716,847 顶点/13.9M 边/300 维/100 类）、Reddit（232,965/114.6M/602 维/41 类）、Ogbn-Arxiv（169,343/1.17M/128 维/40 类）、Ogbn-Products（2,449,029/61.9M/100 维/47 类）、Ogbn-Papers100M（111M/1.6B/128 维/172 类），任务均为节点分类。
- 硬件实现：TAGT 加速器（Alveo U280 FPGA）以 TDS 为原生执行表示，TDL/TCU 实时构造 TDS、FAU 流式注意力、SCU 块级异步 softmax（见 硬件架构 层条目）。
- 参考实现：Graphormer 开源（https://github.com/microsoft/Graphormer）；本论文未开源。

涉及论文标题：
- TAGT: An Efficient Graph Transformer Accelerator with Topology-aware Sparsification and Merging
