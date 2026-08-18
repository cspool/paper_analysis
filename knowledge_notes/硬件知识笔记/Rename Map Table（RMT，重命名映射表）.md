## Rename Map Table（RMT，重命名映射表）

术语解释
RMT 在 rename 阶段把逻辑寄存器映射到物理寄存器并记录生产者的关键元数据（IQ 行号、本论文扩展的 segment 号/segment offset/长延迟 flag），是设置 wakeup matrix cell 与 HSD 派发决策的信息源。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) rename 把逻辑目的寄存器重命名为物理寄存器，同时为每条源寄存器查其 producer（写该逻辑寄存器的最新指令）在 IQ 中的行号；(2) 该行号用于 dispatch 时在 wakeup matrix 相应列写 cell（论文 II-B：cell 值在指令派发到行时写入，producer 的 IQ entry 号存于 RMT）；(3) 本论文扩展 RMT 三项字段：producer 所在 segment 号（决定目标 segment）、segment offset 号（决定 L1 矩阵列号，与全 IQ 行号并存）、"is long latency" flag（生产者执行延迟是否 > L2 额外流水深度，供 HSD 切边判定）；(4) rename 占用两周期（第 1 周期读 RMT、第 2 周期写 RMT，参照 Pentium 4 [27] 的假设）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
运转流程（论文 IV-A/IV-D）：rename 读 RMT 得每个源的 producer segment 号与 busy 状态 → HSD 判定 chunk 归属与目标 segment → dispatch 写入该 segment 的 L1（列=producer 的 segment offset）或 L2 → 写 RMT 登记新 producer 的 IQ 行号/segment 号/offset/长延迟 flag。bundle 内并行重命名时 producer 信息由 DCL（寄存器号比较器+优先级编码器）在 bundle 内解析，先于/并行于 RMT 读，避免串行依赖拖慢 rename 第二级（其延迟测得为 sparse-tree adder 的 88%，6-wide 默认；10-wide 为 1.59×，可再流水化）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：现代实现为多端口表（每周期支持 bundle 宽度个读/写）+ busy bit（物理寄存器是否已分配/就绪）；条目数与逻辑寄存器数（ISA 寄存器文件大小）匹配，物理寄存器堆（本文 512 int + 512 fp）决定重命名代数的上限。使用：提供唤醒与派发所需的 producer 追踪；扩展字段（segment 元数据）是 HWL 的"派发侧"信息基础设施。论文未开源。

涉及论文标题：
- Hierarchical Wakeup Logic of the Issue Queue for High Scalability
