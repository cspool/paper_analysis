## SWAP Mirroring（SWAP 吸收/镜像合并）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SWAP mirroring 是 CANOPUS 的核心路由机制：当插入的 SWAP 与前一个已映射的 2Q 门 U 作用在同一个物理 qubit 对时，不把 SWAP 当作独立固定成本的门，而是把它们合并为一个复合酉 U' = SWAP·U（在规范表示中仍是单个 2Q 门），其边际合成成本 c_g = COST(SWAP·U) − COST(U)。由于复合酉的规范坐标往往落在"更便宜"的多胞形，c_g 通常低于独立 SWAP 的朴素成本 c_swap，甚至可以取负值。论文例子：CX 基下 SWAP·iSWAP ~ Can(1/2,0,0) 只需 1 个 CX 合成，而 iSWAP 本身需 2 个 CX，故 c_g = c_cx − 2c_cx = −c_cx（负成本！）；√iSWAP 基下 SWAP·√iSWAP 的 c_g = 0。门镜像（mirror gate，如 pSWAP 族、ECP、CX 的镜像）天然利于这种吸收，因为镜像组合能覆盖 Weyl chamber 更大区域。
- 别名：SWAP absorption、piggyback（论文对 qLDPC 场景的描述："把 SWAP 插入 piggyback 到 CX 上而不增加额外 2Q 门数"——CX·SWAP 复合等价于 iSWAP，门数与成本均不变）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译框架内运转流程：① 维护 last mapped layer L（当前 DAG 中无后继交互的 2Q 门集合，即每个物理 qubit 对上的"最后映射层"）；② 枚举候选 SWAP 时，若候选 SWAP 的物理 qubit 对 (q0,q1) ∈ L，则取出 U=L[(q0,q1)]，计算复合酉 U'=SWAP·U 的规范坐标（KAK），查 monodromy 多胞形得 COST(SWAP·U)，进而 c_g = COST(SWAP·U) − COST(U)；③ 把 c_g 与深度增量 Δdepth、差分拓扑距离一起代入统一启发式 H = w_g·c_g + w_d·Δdepth + (ΔAvg{dist}_E + k_E·ΔAvg{dist}_E)·c_swap 评分，选最小 H 的 SWAP 插入；④ 插入后按 Algorithm 1/2 更新 L、D（wire duration）、C（交换对）。Fig.6(a) 示例：第一步 SWAP 吸收进前驱 Can(0.5,0,0) 形成镜像门 Can(0.5,0.5,0)，CX 基下 c_g = 1×c_cx、√iSWAP 基下 c_g = 0×c_√iSWAP——ISA 感知的成本评估让路由器主动选择能"免费吸收"SWAP 的选路。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：体现在 CANOPUS 开源实现（https://github.com/Youngcius/canopus，CanopusMapping pass）中 L/D/C 数据结构的维护（Algorithm 1、Algorithm 2，全部 O(1) 哈希操作）与启发式评分。qLDPC case study 展示了其价值：CX-iSWAP 组合 ISA 下，稳定子测量电路中有大量"把 SWAP 吸收进 CX"的机会，CANOPUS 相对 SABRE 使逻辑错误率 square 拓扑再降 52.6%（vs CX ISA 的 49.4%）。
- 效果量化：SWAP mirroring 使 CANOPUS 在所有 216 个 case 中取得最低 routing overhead，且换更强 ISA（含镜像门）收益放大（chain 上 CX 1.88× → ZZPhase_ 1.39×，-26%）。

涉及论文标题：
- Unifying Qubit Routing Across Diverse Quantum ISAs via Canonical Representation
