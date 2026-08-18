## MM-sc 列切分 tiling（Column-wise Weight-Membrane Tiling）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- MM-sc 的 tiling 决定权重矩阵与膜电位矩阵如何在多个 PE（或神经核）间划分。传统加速器按块切分（block-wise，行×列块都切），需要跨 PE 的 spike 归约/部分和同步。ELSA 采用列切分（column-wise）：把权重矩阵的列与膜电位的列按列分组分配到不同 PE（如 Fig.10d：第 1、2 列给 PE1，第 3、4 列给 PE2），而 spike 广播给所有 PE。
- 该切分的收益：① spike 只需广播（所有 PE 收到同一批 spike），消除跨 PE 的部分和归约（Local Input Reducer 的 spike reduction 不再需要）；② 每个 PE 只存自己的权重/膜分片，无重叠，面积利用率高；③ 与 Gustavson 行式累加天然契合——同一 spike 触发各 PE 各自累加自己的权重列分片。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 列切分下的执行（2 个 PE、权重 4 列）：
```
# 广播 spike (x=0, y∈{1,3}) 到 PE1、PE2
PE1 持有 W 列 1-2（行 2、4 的对应片段）→ 累加膜行 V[0] 的列 1-2
PE2 持有 W 列 3-4 → 累加膜行 V[0] 的列 3-4
# 无跨 PE 归约：每个输出列只在一个 PE 内完成
```
- ELSA 论文 Fig.10d 明确："column-wise divide the synaptic weight and membrane（1st and 2nd column to PE1 and 3rd and 4th column to PE2）rather than dividing them block-wise in traditional accelerators"。配套的映射原则：partition 阶段优先层内（layer-wise）划分——把整层放进同一神经核，从而"spike broadcast in tiling strategy and spike reduction between PEs in Local Input Reducer can be avoided"。
- Annotations：列切分维度 = 输出特征维（膜列）；spike 广播 = 所有 PE 收到相同输入脉冲集；归约消除 = 每输出元素由唯一 PE 产出，无部分和合并。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：ELSA 的 PE 控制模块按 spike 的行/列编码从本地权重 buffer 取对应列分片；贪心分区算法（Algorithm 2）决定哪些层放进同一神经核（约束：核内存 A 与核神经元电路数 D），层内 MM-sc 再用列切分在 4 个 PE 间展开。效果：Tab.IV 中 ELSA 面积效率 41.26 GOPS/mm² 为弹性 SNN 加速器最高（TrueNorth 0.134、PAICORE 19.78），部分归功于无重叠、无归约的列切分；Fig.14 显示映射三阶段（partition→mapping→routing）的目标之一就是"minimize NoC traffic"（层内划分直接省掉 spike 广播跨核流量）。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
