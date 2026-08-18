## xPU-xMU 异构架构（ASIC-NMP 异构 FHE 加速器）

术语解释
首个 xPU（ASIC）-xMU（NMP/HBM 近存）异构 CKKS FHE 加速器（HE²）：计算密集型算子（ComOps：NTT/BConv/ModUp/ModDown）由高性价比 ASIC 模块 xPU 加速，内存密集型算子（MemOps：IP/PMul/CAdd/Autom）由高带宽 HBM 近存模块 xMU 加速，规避单体 ASIC 的大片上 SRAM 开销与单体 NMP 的大逻辑集成开销。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 单体 ASIC 加速器（SHARP/ARK/BTS/CLake/UFC/FAST）用定制模块加速 ComOps，但 MemOps（IP/PMul）需 180+18 MB 级片上 SRAM（占面积功耗近半）；单体 NMP（FHENDI）用近存带宽加速 MemOps，但需在 DRAM 内集成大规模复杂计算核心（DRAM 工艺逻辑密度比 CMOS 低 10×、速度慢 3×，面积预算受限 25%，功耗与散热不可行）。xPU-xMU 异构把两者互补：xPU（7nm，1 GHz）承担复杂 ComOps，xMU（HBM2 内 bank-level PE，12nm，450 MHz）承担轻量 SIMD 风格 MemOps。分工依据 = 算子算术强度（Table I：ComOps AI 0.89–3.38 ops/byte，MemOps AI 0.07–0.12）。端到端比 SHARP 快 1.66×、EDAP 降 9.23×、通信 stall 仅 6.67%；xMU PE 占 HBM 面积 11.1%，xPU 47.4–55.7 mm²（HE²-SM/LM），整机 71.9/80.2 mm²。
- 核心挑战：keyswitch（占 CKKS 80% 计算）的数据流是交替 ComOps/MemOps（ModUp→IP→ModDown）。若 IP 卸到 xMU（IRF 数据流），ModUp 输出与 IP 结果（单次最高 144 MB）经 1 TB/s HBM 往返、落在关键路径上——原始 heterogeneous 组合（SHARP-xMU）通信 stall 占 bootstrapping/ResNet-20 的 68.2%/68.7%。HE² 用 HERO（DFG 级 PKB 融合放大 hoisting，降通信频率）+ 双级流水 xPU（隐藏通信延迟）+ hybrid 数据流（低并行 PKB 回退 EVF）解决。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 一次 keyswitch（ModUp-IP-ModDown）在 xPU-xMU 上的运转流程（IRF 数据流）：
```
xPU:  ct → ModUp（逐 dnum 组走 INTT→BConv→NTT 流水，边算边流式写出）
        └─ 1 TB/s HBM ─►  xMU: IP/PMul（bank-level PE 从 row buffer 取 256-bit，
                                    MemOp fusion 省 row-switch）→ 结果回传
        ◄── 1 TB/s HBM ─┘
xPU:  ModDown（收到输入即算，不缓冲全量）→ 加回原密文
```
- Annotations：xPU 双级流水使 ModUp 输出的组间计算-通信重叠（一组传输与另一组计算并行）；xMU 只做向量化 MemOps、由 host（xPU 侧）控制；hybrid 方案对单 keyswitch PKB 改走 EVF（evk 预载 84 MB）避免中间结果往返。xPU 微架构：迭代 radix-2 NTTU（768 w/ns）+ tree BConvU（672 w/ns）+ EWEU（512 w/ns）+ NTTU allocator + INTT/NTT-Resident 密文格式管理；xMU：256-bit/PE 局部缓冲、row-major 布局、in-DRAM automorphism（global row buffer 2048 coeff/cycle 站内、bank I/O 128 coeff/cycle 站间、GBus 32 coeff/cycle 组间）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：xPU 用 Verilog RTL + TSMC 7nm PDK 综合（面积/功耗 CACTI-6.0 建模 SRAM 与布线，含 OF-Twist twiddle 生成、KSKGen evk 生成）；xMU 把 PE 集成进两个 HBM2 栈（8 GB、1 TB/s）的 column decoder，遵循 AiM/Newton 类 bank-level 近存先例，12nm PDK 综合，RTL 校验时序与 HBM bank I/O 兼容。评估：自研 cycle-accurate 模拟器建模 SHARP、HE²-SM（44 MB 仅 IRF）、HE²-LM（84 MB hybrid）、SHARP-xMU 四种架构（SHARP 复现与原文平均差 1.20%）。用途：CKKS bootstrapping/HELR/ResNet-20/56/BERT 端到端加速；对比 SHARP/FAST/FHENDI/Anaheim/BTS/CLake/ARK/UFC。

涉及论文标题：
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
