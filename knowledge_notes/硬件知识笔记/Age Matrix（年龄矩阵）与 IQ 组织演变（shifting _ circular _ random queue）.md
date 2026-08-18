## Age Matrix（年龄矩阵）与 IQ 组织演变（shifting / circular / random queue）

术语解释
年龄矩阵（age matrix）记录 IQ 内指令的相对年龄顺序，供 select 逻辑按"最老就绪优先"仲裁发射；它是现代 random queue 组织（指令乱序插入 issue 空位、不保年龄序）的配套结构，是 IQ 从 shifting queue → circular queue → random queue 演变的结果。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链（论文 II-C，引用 [6]-[9]）：(1) shifting queue——早期 FIFO 型 IQ，issue 后移位补空保持年龄序，IPC 高但移位逻辑复杂、功耗大；(2) circular queue——免移位降复杂度功耗，但不能把新指令插入 issue 空位，容量效率低、IPC 退化；(3) random queue——指令直接插入 issue 留下的空位（无年龄顺序），容量高效但破坏年龄序、单凭随机发射 IPC 显著退化；(4) 解决方案——加 age matrix 标识最老就绪指令并优先发射（虽只保证最老就绪者被优先、其余按年龄随机，仍大幅恢复 IPC），现代设计（AMD Bulldozer [4]、IBM POWER8 [5]、Pentium 4 [8]）均采用该组合；(5) 本论文在 random queue + age matrix 的基线上实现 HWL——HSD 需要"在乱序队列中把依赖指令派到同 segment"，这与 shifting/circular 队列（有天然年龄/位置约束）不同，是 HWL 区别于 prior 层级方案（H-SW、narrowing 均假设 circular IQ）的关键适配点。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（论文 II-A）：wakeup 完成后 select 触发；select 用 age matrix 在就绪指令中仲裁，输出最老就绪者优先授予 grant；grant 除发射外还反馈回 wakeup 矩阵 wordline。年龄比较不直接参与 wakeup 矩阵（其 cell 只表示数据依赖），而是独立于唤醒矩阵维护。论文在 SimpleScalar 自建模拟器中以"random queue + age matrix"实现基线 IQ（替换原 RUU），并在此之上叠加 HWL 的分段与派发逻辑。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：age matrix 为窗口尺寸×窗口尺寸的位矩阵，dispatch/issue 时维护相对年龄位；select 侧与唤醒矩阵并列使用。使用：在 random queue 组织的现代乱序核中恢复 age-order 发射；也是评估"无年龄顺序"IQ 方案的基线（narrowing 与 H-SW 需从 circular IQ 适配到 random IQ 才能与本论文公平对比，论文在 V-G1 做了该适配并指出其 L1 容量管理缺失导致 3.2%/3.6% IPC 退化）。

涉及论文标题：
- Hierarchical Wakeup Logic of the Issue Queue for High Scalability
