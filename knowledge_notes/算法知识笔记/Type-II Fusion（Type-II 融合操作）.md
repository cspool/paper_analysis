## Type-II Fusion（Type-II 融合操作）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Type-II fusion 是光子量子计算中把两个较小的图态合并为更大图态的线性光学操作，是图态生成（graph state generation）中最关键的操作，因为它使图态的资源高效并行生成成为可能。与 Type-I fusion（两个输入顶点合并为一个，继承双方边）不同，Type-II fusion 更复杂：两个输入顶点都被移除，一个顶点的邻居连接到另一个顶点的邻居（若先前未连接）或与之断开（若先前已连接）。Type-I 和 Type-II fusion 都可通过线性光学概率性实现（HWP 半波片 + PBS 偏振分束器）。论文聚焦 Type-II fusion，因为它支持 heralded photon loss（光子丢失可被 heralded/预警），这是设计擦除容错的前提。Type-II fusion 有三个结果（如图 1 所示）：两个融合量子比特被探测器捕获在不同侧 → 融合成功；被捕获在同一侧 → 融合失败（failed qubits 被有效测量在 Z 基并断开，图结构对编译器已知）；其中一个量子比特未被捕获（光子丢失）→ 融合擦除（输出图态结构未知）。融合成功率理论上限 50%（可通过额外光学硬件提升至 75% 或更高）。论文的融合错误模型：1-p_fail=0.75（无擦除时的成功率，需额外干涉装置 [18][22][49]），σ_fus=(1+V_HOM)/2=99.75%（HOM 可见度 99.5%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Type-II fusion 在论文 pipeline 中的具体计算过程（树编码融合的一个分支）：
```
# 输入：图态 A 的叶子量子比特 q_i^c(A)、图态 B 的叶子量子比特 q_i^c(B)
# 线性光学：HWP + PBS，双光子 Hong-Ou-Mandel 干涉
def type2_fusion(qA, qB):
    # 两个光子模式经过 PBS：水平/垂直偏振分束
    det1, det2 = PBS_interference(qA, qB)
    if det1 != det2:            # 两个光子在不同探测器侧
        return SUCCESS           # 融合成功：邻居互联，图结构确定
    elif det1 == det2:           # 两个光子在同一侧
        return FAILURE           # 融合失败：两 qubit 被 Z 基测量移除，图结构已知
    else:                        # 某个光子未被捕获（丢失）
        return ERASURE           # 融合擦除：图输出不确定，需间接测量保护
```
图论效果：成功时——两个输入顶点 v1、v2 被移除，v1 的邻居集合 N(v1) 与 v2 的邻居集合 N(v2) 直接相连（或断开重叠边）；失败时——v1、v2 被移除且无新连接（等同于 Z 测量）；擦除时——结果未知。在树编码中，融合失败时 q_i^c 被测量掉而 q_i^a/q_i^b 保留（Z 测量移除 q_i^b 留备份 q_i^a）；擦除时用间接 Z 测量恢复。论文真实硬件实验中用 Perceval 搭建 fusion 电路：双轨编码（dual-rail）下融合电路 = 两 qubit 光子模式的置换 + 相移 + 两个分束器。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：线性光学元件（HWP、PBS、探测器）构成干涉仪。文献中的实现细节存在差异（是否在第一个 PBS 前插入 HWP 导致成功时的局部酉校正与失败时有效测量不同），论文采用 [25] 的方案。成功率增强：(1) 冗余编码（m 次融合尝试，P_fail=p_fail^m）；(2) repeat-until-success（成功后终止，需 ancilla 光子）；(3) 本论文的树编码（b 个分支 + 间接 Z 测量，S_tree=1-(1-(1-p_eras)^2+p_fail)^b）。额外干涉测量装置（ancilla-assisted Bell measurement）可将成功率提升至 75%（Grice 2011；Ewert & van Loock 2014）。使用场景：Type-II fusion 是三类 PQC 架构（all-photonic、emitter-based、spin memory）共同的图态拼接原语；论文在 Quandela 云平台用 Perceval 实现真实 fusion 电路（实测 HOM 不可区分度 92.0%、透射率 5.16%）。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
