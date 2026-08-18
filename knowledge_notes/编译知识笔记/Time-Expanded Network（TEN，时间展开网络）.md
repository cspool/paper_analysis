## Time-Expanded Network（TEN，时间展开网络）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Time-Expanded Network（时间展开网络）是 TACOS（MICRO'24，Topology-Aware Collective Algorithm Synthesizer）提出的集合通信综合建模技术：把物理网络拓扑沿时间轴复制展开为分层图，每个时间步一层网络快照，节点/链路按时间分层表示"数据在特定时间步位于特定位置"，从而把"随时间演化的数据传输调度"转化为在展开图上的组合匹配问题——TACOS 在此图上做随机+贪心的 link-chunk 匹配算法（每次利用所有可用通道做一次数据传输）。PipeComm 在 motivation 中直接对比：TACOS 的贪心启发式常使关键路径被慢速链路主导（图 2c 的 AllGather 例中 TACOS 与 MultiTree 平均链路利用率仅 67%），且其 chunk 分区复杂度 Θ(c²n²)（c=chunk 数）限制其只能少量分区、无显式 pipeline 模型、分区后甚至性能变差。PipeComm 也借用了 TACOS 的 switch unfolding 方法（把 switch 网络展开为固定点对点连接）用于 switch 拓扑评估。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
TEN 在综合框架中的运转流程（对比 PipeComm）：
```
TACOS（TEN + 贪心 link-chunk 匹配）:
  输入: 拓扑 + 通信规格 → 时间展开网络（每时步一层快照）
  → 每时步贪心匹配"可用链路 × 待传 chunk"（随机化 + 贪心）
  → 输出单轮 schedule（无跨迭代流水模型）
  问题: 关键路径被慢速链路主导; 每时步只匹配少量剩余传输 → 末期通道空闲
        chunk 分区 → 复杂度 Θ(c²n²) 爆炸 → 实际只敢用 1~4 chunk → 重叠不足

PipeComm（两阶段合成，替代 TEN 的贪心匹配）:
  阶段一 MILP/增量构造 pattern（II 容量约束 Σx≤II/w 显式防跨迭代拥塞）
  阶段二 Modulo-II Reservation Table 调度（每传输分配无冲突时隙）
  → 显式流水线模型：稳态每 II 步完成一次迭代，所有链路持续满占用（利用率 >80%）
  结果: vs TACOS 1.53×（Pipe-Sol）、TACOS(4 chunk) 1.39×
```
本质区别：TEN 是"时间步维度的展开搜索空间"，PipeComm 是"II 模数的稳态流水抽象"——后者天然支持任意多 chunk 的连续重叠且复杂度与 chunk 数无关。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：TACOS 在展开图上做贪心 link-chunk 匹配（开源：https://github.com/abejeb/TACOS 论文发布版）；PipeComm 作为 baseline 复现时按 1 和 4 个 chunk 评估（论文 A.-Methodology：TACOS 支持分区但复杂度随 chunk 平方增长，故评估 chunk=1/4）。使用场景：交换机拓扑的展开（PipeComm 用 TACOS 的 unfolding 方法把 switch 网络转成固定点对点连接再评估）；以及在合成框架中作为"顺序调度/贪心启发式"baseline 与 PipeComm 的"流水线最优合成"对照。局限：无显式 pipeline 模型、贪心次优、复杂度随 chunk 爆炸。

涉及论文标题：
- PipeComm Maximizing Link Utilization through Pipeline-Aware Collective Communication Synthesis
