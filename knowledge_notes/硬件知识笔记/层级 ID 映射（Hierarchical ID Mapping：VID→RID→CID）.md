## 层级 ID 映射（Hierarchical ID Mapping：VID→RID→CID）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
聚类合并（merge）时的两层间接寻址：顶点 ID（VID）先映射到根 ID（RID），再由紧凑的 RID→CID 表映射到合并后簇 ID（CID）。直接法在 merge 时要把所有被并簇的存储单元逐一重写 CID（Fig.7a：CID 3/6/7 → 1 需更新 15 个存储单元）；层级法只更新 RID→CID 表中少量条目（Fig.7b：只改 RID 3/6/7 三条映射）。写扇出从"触及 VID 数"塌缩为"触及 RID 数"（VID→RID 在增长期本就是多对一），实现单周期 remap，化解 merge 阶段"高并发、弱空间局部"的 CID 重写风暴。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
```
增长期: VID ->(多对一) RID   存于多 bank 哈希缓冲(局部更新友好)
merge 期: 仅改 RID->CID 小表:
  merged_RIDs = {3,6,7} -> CID 1
  写 3 条映射(而非 15 个 VID 存储单元)   # 写扇出塌缩
  后续 VID 解析 CID = TABLE[RID(VID)]   # 读取时走两级间接
```
两级间接的分工：哈希多 bank 缓冲服务增长期的局部 VID→RID 更新，紧凑 RID→CID 表服务 merge 期的分散重标号——与多 bank 哈希互补（前者解局部并发、后者解全局扇出）。消融：单独 1.03×（d=11、p=0.0015），主要价值在缓解峰值写带宽而非纯延迟；与其余优化协同达 3.24×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：小容量片上 SRAM/寄存器堆存 RID→CID 表（容量 ∝ RID 数，远小于 VID 数）+ 哈希多 bank 存 VID→RID。类似思想即虚拟化/间接层（页表、标签重映射、fat tree 逻辑端口映射），本处针对聚类 merge 的"高扇出重标号"语义定制。使用场景：任何需要频繁"批量逻辑重标号"的流式图/并查集加速器；代价是多一次间接读取（读取路径 latency +1），换来写侧带宽与冲突的指数级下降。

涉及论文标题：
- Coset Ensemble Decoder for Quantum Error Correction with Algorithm-Hardware Co-Design
