## PKB（Parallel Keyswitch Block，并行密钥切换块）与 PKB 融合

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PKB（Parallel Keyswitch Block）是 HE² 对 CKKS 数据流图（DFG）的核心抽象：一组并行的 keyswitch（同一输入密文 ct 经不同旋转步长 s_i 后各自做 keyswitch），在明文矩阵×密文向量乘法与 bootstrapping 的 C2S/S2C 阶段占据主导。状态最先进实现用并行 keyswitch（不同旋转步长）+ PMul/CAdd 线性组合完成；旋转步长在 PKB 内通常构成算术级数。HE² 的 HERO 框架先遍历 DFG、按路径顺序给 keyswitch 分层并把同层分组为 PKB（PKB identifying），再用交换律算子贪心扩展 PKB 压低入/出度（degree-minimized PKB expanding），为 hoisting 创造最大削减空间。
- PKB 融合（PKB Fusing，HE² 首次提出）：利用旋转可加性 Rot(Rot(ct,s),t)=Rot(ct,s+t) 与 EWO 后移（Rot(PMul(ct,pt))=PMul(Rot(ct),Autom(pt))），把两个串行 PKB（n1 与 n2 条旋转路径）融合为 O(n1·n2) 条并行旋转的大 PKB（逆 BSGS 变换），从而把 CKKS 程序中大量低并行（<10）的碎片 PKB 合并成高并行（>30）PKB，让 hoisting 的 ModUp/ModDown 共享潜力被完全释放。代价：evk 数量（按非重复旋转步长子集计）、IP 数与中间 MemOps 计算量上升。融合收益由 Fusion evaluator 的 FuseScore 量化（融合后 evk 超存储容量判无效），全局最优融合方案由 DP 递推式 DP[i][j]=max_{j'}DP[i][j']+DP[j'+1][j]+FuseScore(j',j'+1) 求出。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 两串行 PKB 融合（HE² 论文式 (4)，n1=n2=2 简化例）：
```
# 融合前：PKB1（n1=2 并行旋转）→ EWO F → PKB2（n2=2 并行旋转）
PKB1: { Rot(ct, s1), Rot(ct, s2) }
F:    求和/线性组合（EWO）
PKB2: { Rot(·, s1'), Rot(·, s2') }
# 融合：EWO 沿各路径后移 + 旋转步长相加
Fused: { F_i'( Rot(ct, s1+s1'), Rot(ct, s1+s2'),
                 Rot(ct, s2+s1'), Rot(ct, s2+s2') ) }   # 4 条并行旋转（3 条不同步长）
# 融合后 hoisting：4 条并行 ModUp → 1 次共享 ModUp；输出端线性组合后 1 次 ModDown
```
- Annotations：s_j+s_i' 出现重复步长时可去重减少 evk 数；Fusion evaluator 依据相对 IP 数、中间结果尺寸与所需 evk 数评估"省下的 ModUp/ModDown 通信 vs 增加的 evk 存储与 MemOp 计算"，在 8 GB HBM 存储约束下用 DP 选全局最优；案例 ConvBN DFG（3 个 9/8/8 并行 PKB）：原始 25 个 ModUp/ModDown，直接 hoisting 只优化 PKB1，融合后 PKB2+PKB3 的 ModUp/ModDown 可提取到 8 条并行路径首尾（Fig. 9）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为离线程序变换在 FHE 编译器（EVA/CHET/ResiBM 生成 DFG）之后、映射到硬件之前执行；HE² 的 HERO 框架完整流程 = PKB 识别 → 度数最小化扩展 → DP 融合评估 → BSGS 配置选定 → 按 PKB 并行度映射 IRF/EVF 数据流。作用：把 hoisting 的通信削减从"程序天然并行"中挖出（相比直接 hoisting 再多削 2.25× 计算/2.42× 通信），并让融合后的 MemOps（IP/PMul）整块卸到近存 xMU，省掉 EVF 所需的巨大片上 evk 存储（SHARP 180+18 MB → HE² 44/84 MB）。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
