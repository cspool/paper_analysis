## 光子 PQC 前馈控制（Feed-Forward Control）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
前馈控制（feed-forward control）是光子 PQC（尤其是 MBQC）中的经典控制机制：量子测量结果（光子探测）实时决定后续操作——包括后续测量基的选择（adaptive measurement，MBQC 的本质特性）与图态生成流水线中失败子图的延迟调度。论文在 MemTree 中识别出两类需要 feed-forward 的场景：(1) 树编码融合的容错恢复——融合结果（成功/失败/擦除）触发对辅助量子比特 q_i^a/q_i^b 的条件测量（X 或 Z 基）；该前馈不在光学关键路径上：融合结果检测后受影响 qubit 可保持为 dangling qubit，控制器只需记录稍后应用的恢复模式，校正测量不必立即触发，只需在 dangling 分支被后续图态测量消耗前或程序最终测量前同步（遵循 MBQC 的 adaptive-measurement 模型 [68]）。(2) 图生成流水线的失败延迟——融合失败后决定兄弟子图是否延迟到下一时间步。控制路径组成：SNSPD 光子探测（延迟 <50 ps）→ 小型组合逻辑块（b 输入 AND/OR 网络，判断 b 个融合分支是否逻辑成功/是否需要 stall）→ 时间延迟模块。总经典 feed-forward 延迟估计 <5 ns，远低于 spin memory 硬件的单个发射时间步（30 ns），因此在下一发射层开始前可同步更新测量模式与子图调度。论文用 Perceval 的 FFCircuitProvider 实现该逻辑。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
feed-forward 控制在硬件中的运转流程（树编码融合分支，时序）：
```
# 时刻 t: Type-II 融合执行 (融合电路: 光子模式置换 + 相移 + 分束器)
# t+50ps : SNSPD 探测, 判定融合结果 (成功/失败/擦除)
# t+Δt   : 组合逻辑 (b 输入 AND/OR 网络) 汇总 b 个分支结果
#          -> 逻辑融合成功? 需延迟兄弟子图? (Δt 使总延迟 < 5 ns)
# t+Δt   : 控制器记录恢复模式 (dangling qubit 保留, 无需立即校正)
# 下一发射层前: 同步恢复测量模式 (X/Z 测量 q_i^a/q_i^b) 与子图调度
# 场景1: 融合擦除 -> 记录 "间接 Z 测量模式" (X(q_i^b)+Z(q_i^a))
# 场景2: 融合失败 -> 记录 "Z 测量 + 备份保留" (Z(q_i^b), 留 q_i^a)
# 场景3: 流水线失败 -> 触发时间延迟模块, 兄弟子图下移一个时间步
```
关键性质：校正测量可延迟（dangling qubit 机制）→ 前馈不阻塞光学关键路径，只需在消费前同步——这是 feed-forward 延迟可放宽到 <5 ns 的硬件基础。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 硬件——SNSPD（超导纳米线单光子探测器，延迟 <50 ps，Marsili 等 Nat. Photonics 7, 210, 2013）+ 组合逻辑 + 时间延迟模块；(2) 软件——Perceval 的 FFCircuitProvider（https://github.com/Quandela/Perceval）实现条件前馈电路，论文真实硬件实验用它搭建融合 + 容错恢复电路。使用场景：所有 MBQC 光子平台（测量基自适应 + 生成流水线失败处理）；也是量子网络/中继器中的标准机制。论文未开源（模拟器）；Perceval 开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
