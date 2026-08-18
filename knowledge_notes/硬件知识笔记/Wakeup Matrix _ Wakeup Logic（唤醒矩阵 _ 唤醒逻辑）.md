## Wakeup Matrix / Wakeup Logic（唤醒矩阵 / 唤醒逻辑）

术语解释
唤醒逻辑把"生产者结果就绪"传播给 IQ 中等待的消费者；RAM-based 形式称为 wakeup matrix——一个二维依赖数组，行=消费者、列=生产者，producer grant 经 wordline 广播、消费者经 bitline 读出 ready，无需 CAM 式 tag 比较。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 两类实现——CAM-based 用 tag 广播+每项比较器，RAM-based 用依赖矩阵（wakeup matrix）[3]，两者均在商业处理器中使用 [4][5]；(2) wakeup matrix 中 cell(i,j)=1 表示指令 i 依赖指令 j，dispatch 时按 RMT 给出的 producer 行号写入；(3) producer 发射时 grant 驱动其列 wordline 广播，cell=1 的行经 bitline 读出 ready 信号（论文 Fig.2 例子：insn1 依赖 insn0 → cell(1,0)=1；grt0 拉高 → wordline0 广播 → bitline1 读出 rdy1）；(4) 矩阵规模随 IQ 尺寸平方增长（如 200×200），延迟随之上升，这是本论文要打破的瓶颈。Web 证据（Goshima 等 MICRO-34）：dependence matrix 把唤醒变成"被发射指令列的 bitwise-OR"（矩阵广播），免除关联搜索；narrowing 利用 ~90% 依赖距离 ≤32 的观察。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在 HWL 中唤醒矩阵被拆成两级（论文 Fig.3-6）：每 segment 一个物理独立的 L1 矩阵（列=segment offset 号 0..S-1，论文 Fig.5 中 i2 依赖 i1 时写 cell(2,1)），其后全尺寸 L2 矩阵 3 级流水化；消费者 ready = L1 ready OR L2 ready（Fig.6 单行 OR 门）。L1 情形单周期完成 wakeup-select（Fig.4 ①），L2 情形第 1-2 周期唤醒、第 3 周期 select（②，多 2 周期）。dispatch 决定哪些 cell 写 L1（同 segment 依赖）或 L2（跨 segment 依赖）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SRAM-like 位单元矩阵 + wordline/bitline 驱动；延迟由列扇出（wordline 驱动能力）与 bitline 长度决定。降延迟手段：narrowing（只留对角线附近 D 个 cell [3]）、列压缩（MS-rel 减少无消费者列 [9]）、以及本论文的层级化（小 L1 + 流水化 L2）。评估：论文用 HSPICE 22nm PTM 测 200×200 基线矩阵+select 延迟并归一化，L1=25 项时 HWL IQ 周期降到基线的 47%（缩短 53%）。本论文未开源实现。

涉及论文标题：
- Hierarchical Wakeup Logic of the Issue Queue for High Scalability
