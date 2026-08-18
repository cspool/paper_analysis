## HERO（Hoisting-Enhanced DFG Optimization，PKB 融合 DFG 优化框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- HERO（Hoisting-Enhanced DFG Optimization）是 HE² 提出的离线 DFG（数据流图）优化框架：把 FHE 编译器（EVA/CHET/ResiBM 等）生成的 CKKS 程序 DFG 作为输入，通过 PKB 识别、度数最小化扩展、PKB 融合与 BSGS 配置探索，最大化 hoisting 的通信/计算削减潜力，再把重构后的 DFG 按 PKB 并行度映射到异构 xPU-xMU 加速器的 IRF/EVF 数据流上。它不是新算法内核，而是"程序图级重组 + 成本模型驱动融合决策"的编译器式优化流水线。
- 核心组件：(1) PKB identifying——从 DFG 输入遍历、按路径顺序给 keyswitch 分层、同层分组为 PKB；(2) Degree-minimized PKB expanding——用交换律算子（EWO/Autom 与 ModUp/ModDown 可交换）贪心扩展 PKB 压低入/出度；(3) PKB Fusing（首次提出）——用旋转可加性 Rot(Rot(ct,s),t)=Rot(ct,s+t) 与 EWO 后移把串行 PKB 融合成 O(n1·n2) 并行的大 PKB（逆 BSGS），把低并行（<10）碎片 PKB 合并为高并行（>30）PKB；(4) Fusion evaluator——FuseScore(i,j) 在融合后 evk 超存储容量时判无效、否则表示计算+通信最大联合节省（n_i·n_j 乘积不变约束下），全局最优融合方案由 DP 递推式 DP[i][j]=max_{j'}DP[i][j']+DP[j'+1][j]+FuseScore(j',j'+1) 求出；(5) BSGS 配置探索——内存足够时禁用 BSGS、受限时偏好 bs 与 gs 差距大的配置；(6) 数据流映射——按 PKB 的 IP 并行度选择 IRF（高并行）或 EVF（单 keyswitch）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- HERO 的编译流程（输入 DFG → 输出映射方案）：
```
# 输入：FHE 编译器（EVA/CHET/ResiBM）生成的 CKKS DFG（节点=算子，边=数据依赖）
DFG → PKB identifying（分层分组）
    → degree-minimized expanding（每个 PKB 入/出度最小化，为 hoisting 铺垫）
    → 两两 PKB 融合候选（逆 BSGS：Rot(Rot(ct,s),t)=Rot(ct,s+t) 合并旋转、EWO 后移）
    → Fusion evaluator（FuseScore 计算各融合对的计算+通信节省，evk 超容则判无效）
    → 全局 DP 求解最优融合方案（式 5，8 GB HBM evk 工作集约束）
    → BSGS 参数选定（禁用 / bs-gs 差距）
    → 按 PKB 并行度映射 IRF/EVF（IP 并行>1 用 IRF，否则 EVF）
# 输出：重构后的 DFG（高并行 PKB 序列）+ 每 PKB 的数据流映射
```
- Annotations：优化目标 = 计算量 + 通信量 + evk 工作集三者的联合权衡（与单体 EVF 加速器的"计算 + off-chip 访存"模型不同，见论文 Sec. III-D 的异构专用性能模型）；效果：HERO 相比 baseline 平均削减 1.64× 计算与 3.27× 通信，相比直接 hoisting 再多削减 2.25× 计算与 2.42× 通信；融合案例 ConvBN（9/8/8 并行 PKB）：原始 25 个 ModUp/ModDown，融合后提取到 8 条并行路径首尾（Fig. 9）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为离线编译器式 pass 集成在 FHE 编译流水线中（HE² 论文未开源实现，arXiv 2605.31004 无 artifact 链接，联网未找到仓库）；FuseScore 的成本模型输入 = 相对 IP 数、中间结果尺寸、所需 evk 数与存储容量。使用：任何 CKKS 应用的 DFG（bootstrapping C2S/S2C、HELR、ResNet-20/56、BERT 的矩阵×密文乘法）在映射到异构硬件前先过 HERO；它决定每个 PKB 走 IRF 还是 EVF、融合哪些 PKB、BSGS 参数取多少——是"算法优化（hoisting）与硬件数据流（IRF/EVF）之间的胶水层"。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
