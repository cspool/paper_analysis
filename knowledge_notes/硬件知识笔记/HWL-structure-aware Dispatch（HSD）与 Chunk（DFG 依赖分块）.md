## HWL-structure-aware Dispatch（HSD）与 Chunk（DFG 依赖分块）

术语解释
HSD 是本论文的派发方案：按 dataflow graph（DFG）把指令切成小依赖子图（chunk），尽量把 chunk 内节点派发到同一 producer segment，使唤醒尽量走 L1 单周期；chunk 通过三条切边规则形成，并用 last-ready 预测决定保留哪条入边。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 目标——L1 容量有限，无法把所有依赖指令放同 segment，需挑选哪些指令"必须"放 producer segment（chunk 内）以最小化 L2 唤醒惩罚；(2) 切边规则（论文 Fig.7）——① 节点有两入边时用 LRP 预测 last-ready 父节点，切掉先就绪父边（最终就绪由 last-ready 父决定）；② 生产者执行已完成（busy-bit 查）则切边（不再在 IQ 内唤醒消费者），该节点成 chunk 叶子；③ 生产者执行延迟 > L2 额外流水深度则切边（执行延迟隐藏 L2 唤醒延迟），成叶子；④ 无目的寄存器节点天然是叶子；(3) chunk 尺寸分布（Fig.8）——平均 91% 动态指令属于 ≤16 项 chunk，故 L1 可小至 25；(4) 例外程序 deepsjeng/xalancbmk/xz/fotonik3d 的 chunk 偏大（<85% 属于 <16），需要额外混合派发模式。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件实现（论文 IV-B/IV-D）：RMT 每项加"长延迟 flag"（生产者执行延迟是否 > L2 额外深度）；rename 时对 LRP 预测的 last-ready 源寄存器查该 flag 与 busy-bit 完成标志，两者均假则该指令属于 producer 的 chunk、目标 segment = producer segment。bundle 内并行重命名时：DCL（寄存器号比较器+优先级编码器）识别 bundle 内 producer，LBMUX 从 bundle 内更老指令或 RMT 选"is long latency/register busy"信号，SMUX1 选 producer segment、SMUX2+SMCL 依 LRP/PHT 与 flag 定最终目标 segment（Fig.9，关键路径橙色）；默认 6-wide bundle 下第二 rename 级延迟为 sparse-tree adder 的 88%，10-wide 为 1.59×（可再流水化）。noHSD 对比：不用 LRP（随机选未就绪源）、不检查长延迟 → producer-segment 派发失败率显著更高（Fig.14），fotonik3d（no-stall）与 nab（stall）等程序 IPC 改善明显（Fig.13）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在模拟器中作为派发策略开关实现（HSD on/off × no-stall/stall/hybrid）；HSD 显著降低 DFR 从而降低 L2 唤醒频率。论文未开源；切边思想可复用于任何"把依赖集中到局部快速结构"的层级调度设计（类似 cache 的局部性管理）。评估：Fig.15 中 HSD-hybrid 平均 IPC 退化 0.9%（vs noHSD 场景与 HSD-nostall 1.7%、HSD-stall 4.6%）。

涉及论文标题：
- Hierarchical Wakeup Logic of the Issue Queue for High Scalability
