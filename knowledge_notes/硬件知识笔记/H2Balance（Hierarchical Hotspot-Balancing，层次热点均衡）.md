## H2Balance（Hierarchical Hotspot-Balancing，层次热点均衡）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
H2Balance 是 ParetoES 的多核（32 ACPE）负载均衡算法，解决簇访问幂律（power-law）分布导致的核间负载不均（长尾/straggler）。簇访问遵循重尾分布：少量热点簇主导查询负载，静态按簇 ID 线性分配使单核负载可差 4.64×（κ=W_max/W_min，CV_W=34.25%）。H2Balance 分三阶段（Algorithm 2）：(1) Hotspot Profiling——用 100 个查询构建热点矩阵 H∈R^{100×C}（H[t,c]=查询 t 对簇 c 的向量访问数），累计 Q[c]=Σ_t H[t,c] 得全局优先排列 perm=argsort_desc(Q)；(2) Hotspot Anchoring——top-32 热点簇静态分配给 32 个专用核（各核锚定一个热点簇，均匀分散主导负载）；(3) Greedy Reconciliation——其余簇按 perm 顺序贪心分配给当前最轻负载核（load[p]=Σ 已分配簇的 Q）。效果（Sp.Baidu 100 查询）：CV 从 34.25%→0.01%、κ≈1.0001（静态分配后重测）；对 100 个新查询 CV 34.25%→5.16%、κ 4.64→1.24，吞吐 +37.52%；对分布漂移（+0.2 正向偏置）CV 36.27%→12.25%、κ 4.48→1.63，峰值负载 -28.74%。均衡度量：CV_W=σ_W/μ_W×100%（W_i 为核 i 负载）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
H2Balance 的映射流程（host 离线执行、静态配置，硬件运行时无同步）：
```
Input: H (100xC 热点矩阵), P=32 核
Q[c] = sum_t H[t,c] for all c          # Stage1: 热点 profiling
perm = argsort_desc(Q)                  # 全局优先级
load[0..31] = 0; M = {}
for i in 0..min(31, C-1):               # Stage2: 热点锚定
    c = perm[i]; M[c] = i; load[i] += Q[c]
for i in 32..C-1:                       # Stage3: 贪心调和
    c = perm[i]; p* = argmin_p load[p]; M[c] = p*; load[p*] += Q[c]
return M                                # cluster-to-core 映射 -> 静态下发 ACPE
```
硬件侧执行：各 ACPE 只处理映射给自己的簇集合，Mem Map LUT 按 sub_nprobe 在本核簇集内定位；H2Balance 是"静态映射 + 运行时无均衡动作"的设计（对比动态任务窃取），因此零核间通信、维持 32 通道全隔离。架构角色：把系统级负载分布对齐到硬件多核拓扑，消除 straggler 对流水吞吐的拖累。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：host 侧（CPU/GPU）以少量样本查询（100 个）离线 profiling 后运行上述三阶段贪心，产出 cluster-to-core 映射表 M，随 bitstream/初始化静态写入 FPGA；换数据集/分布时重新 profiling 与映射（不重综合硬件）。使用注意：极端分布漂移下均衡最优性下降（论文 +0.2 bias 实验），但对新查询与温和漂移保持强鲁棒（5.16%/12.25% CV）。它属于"离线热点感知的静态均衡"策略（类似服务器调度的热点锚定+贪心，但应用于硬件多核簇分配）。论文未开源。

涉及论文标题：
- ParetoES Hardware-Accelerated Sparse Embedding Similarity via Pareto-Optimal Pruning
