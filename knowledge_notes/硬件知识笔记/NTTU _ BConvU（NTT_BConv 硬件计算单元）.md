## NTTU / BConvU（NTT/BConv 硬件计算单元）

术语解释
HE² xPU 内的两类计算密集型算子单元：可配置迭代 radix-2 NTTU（NTT/INTT 动态共享，平均吞吐 768 w/ns）与 tree-based BConvU（每周期从分解组所有 limb 各收 1 系数做流水树归约，吞吐 672 w/ns）。二者吞吐匹配使 INTT→BConv→NTT 键切换流水实现算子间重叠，用比 SHARP 更低的并行度追平其 IRF 关键路径性能。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NTT/INTT 与 BConv 的并行模式错配是 xPU 内部无法重叠的根源：NTT 蝶形访问同一多项式内的不同系数（高多项式内并行），BConv 同时处理来自多个多项式的系数（高多项式间并行）。既有设计（SHARP/ARK/F1 等）NTTU 提供高多项式内但有限的多项式间并行，其跨 limb 输出吞吐不足以喂满 BConvU 输入 → 算子间无法重叠。
- HE² 的解法：(1) 可配置迭代 radix-2 NTTU（Mu et al. 设计）：每个 dnum 组的 NTTU 均匀分布到 BConv 所需的所有 limb 上，保证并行供数与全流水（BConvU 每分解组通常 <15 个 limb，NTTU 并行足够）；NTTU 在 NTT/INTT 间动态共享以对齐 BConvU I/O 需求。(2) tree-based BConvU：每周期从分解组所有 limb 各收 1 个系数、做流水化树归约。通过自适应吞吐匹配与灵活调度，xPU 实现 INTT-BConv-NTT 重叠（Fig. 11(d)），以 NTT 768 w/ns + BConv 672 w/ns 的较低吞吐追平 SHARP（1024/16384 w/ns）在 IRF 关键路径上的性能。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 一次 ModUp 的 INTT→BConv→NTT 流水（dnum=3 组，组间计算-通信重叠）：
```
对每个 dnum 组 g（组间流水交错）：
    INTT: NTTU 在 g 的各 limb 上做逆变换（系数域→BConv 需要的形式）
    BConv: BConvU 每周期收各 limb 1 系数 → 树归约 → 目标基结果（Q→PQ·dnum 域）
    NTT: NTTU 转回系数域，输出边算边流式写往 xMU（IRF）
组 g 的 NTT 与组 g+1 的 INTT/BConv 重叠（组级流水）+ 同组内 INTT-BConv-NTT 算子间重叠
```
- Annotations：NTTU 吞吐按 keyswitch 各级平均 768 w/ns（NTT/INTT 共享）；BConvU 树归约深度决定延迟、limb 数决定每周期输入宽度；双级流水（计算-通信重叠 + 算子间重叠）的联合效果把 IRF 通信 stall 从 68.2% 压到 6.67%（见"双级流水 xPU"条目）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：Verilog RTL + TSMC 7nm PDK 综合；面积 96×NTTU 2.05 mm²/8.71 W、672×BConvU 5.32 mm²/22.6 W（Table III，注意 672 是 BConvU 单元数、吞吐为 672 w/ns）。使用：ModUp/ModDown 的 INTT→BConv→NTT 流水核心；NTTU allocator 在 INTT-Resident 流水的并行 BConv→NTT 与 NTT 两路间动态分配负载（随密文 level 变化）；对比 SHARP 的高并行堆叠（1024/16384 w/ns）证明"吞吐匹配 + 重叠"优于"盲目堆并行"。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
