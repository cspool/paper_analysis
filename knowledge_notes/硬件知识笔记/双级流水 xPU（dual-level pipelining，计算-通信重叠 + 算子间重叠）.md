## 双级流水 xPU（dual-level pipelining，计算-通信重叠 + 算子间重叠）

术语解释
HE² xPU 的微架构核心：第一级"计算-通信重叠"利用 ModUp/ModDown 的多 dnum 组并行，把某组的 HBM 传输与其它组的计算流水交错；第二级"算子间重叠"通过 NTTU/BConvU 吞吐匹配把单组内 INTT→BConv→NTT 的依赖链并行化。联合效果是在不提高单算子并行度的前提下隐藏 IRF 关键路径上的 xPU-xMU 通信延迟。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 动机：ModUp/ModDown 处理多个分解（dnum）密文组，每组走 INTT→BConv→NTT 流水；组间并行天然允许"一组传输、其它组计算"的重叠。但既有 ASIC（SHARP/ARK/BTS/F1）堆叠高并行低延迟计算单元，ModUp 输出很快算完、随后长时间等 HBM 传输（通信瓶颈、计算单元闲置，Fig. 11(b)）；简单降低计算并行度也无法消除 stall（Fig. 11(c)）。
- 双级流水：(1) 计算-通信重叠——IRF 下 ModUp 输出边算边经 1 TB/s HBM 流式写到 xMU、ModDown 收到输入即开始，用组间流水交错隐藏每次传输延迟；(2) 算子间重叠——把每组的 INTT→BConv→NTT 三段经 NTTU/BConvU 吞吐匹配并行化（见"NTTU/BConvU"条目）。额外机制：INTT-Resident 密文格式管理把关键路径 INTT→BConv→NTT 拆成并行 BConv→NTT 与 NTT 两条路径（NTT-Resident 用于含 PMul/CMul 子图、INTT-Resident 用于其余），NTTU allocator 随密文 level 动态平衡两路负载；INTT-Resident 引入额外 MemOp 域变换开销，但由 xPU 并行度提升与 xMU 近存带宽抵消。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 3 组 ModUp 的双级流水时间线（对比 Fig. 11(b)/(d)）：
```
时间轴 →   t0        t1        t2        t3        t4
组0:     [INTT BConv NTT]→[传输→xMU]
组1:         [INTT BConv NTT]→[传输→xMU]     # 组1 计算与组0 传输重叠（第1级）
组2:             [INTT BConv NTT]→[传输→xMU]
          └ 组内 INTT/BConv/NTT 也重叠（第2级）┘
```
- Annotations：第 1 级重叠受"计算吞吐 vs 传输带宽"的差距决定，44 MB scratchpad 即可维持全流水（不需缓冲全量结果）；第 2 级重叠需要 NTTU/BConvU 吞吐匹配（NTTU 输出跨 limb 吞吐 ≥ BConvU 输入需求）；INTT-Resident 把单条 INTT→BConv→NTT 串行链变两条并行链、以额外 MemOp 换并行。效果：HE²-SM（IRF）与 SHARP-xMU 性能相近但 stall 大幅减少，通信 stall 占比 68.2%→6.67%、通信能耗 6.60%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Verilog RTL（TSMC 7nm，xPU 面积 47.4/55.7 mm²、功耗 74.5/79.7 W，SM/LM 版本）；CACTI-6.0 建模 scratchpad/布线。使用：所有 IRF 数据流的 ModUp/ModDown 流水；与 HERO 正交——HERO 降低通信频率（每次传输更少）、双级流水隐藏单次传输延迟；hybrid 数据流的 EVF 区主要由计算延迟限制，INTT-Resident 的并行收益主要作用于 EVF 区。评估：cycle-accurate 模拟器（HE²-SM/LM vs SHARP、SHARP-xMU 消融，Fig. 14）。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
