## HMUX Chiplet（HC，HMUX 芯粒）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- HC（HMUX Chiplet）是 CASCADE 多芯粒 TFHE 加速器的基本计算芯粒：每个 HC 实现一个流式 HMUX 数据通路（完整遍历 = 一个 HMUX），包含 Rotation Unit（negacyclic rotation + 多项式减法）、Decomposition Unit（位分解，(k+1) 多项式 → (k+1)×l 多项式）、FFT Unit、VMA（Vector Multiplication-Add）Unit、IFFT Unit 五个专用功能单元，全部深度流水；另含输入/输出 double buffer（隐藏 D2D 时延）、BSK SRAM（10.5 MB）、局部 buffer（768 KB）与 D2D PHY。12 个 HC 按 4×3 网格、环形拓扑经 UCIe D2D 链路互连；HC0 额外集成 VPU（Vector Processing Unit）做 key-switching 等轻量操作（12 MB SRAM）。单 HC 面积 92.5 mm²、TDP 29.91 W（TSMC 28nm、1.2 GHz），HC0 加 VPU 额外 60.1 mm²/13.8 W，全系统 1170.1 mm²/372.72 W。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- HC 的运转流程（一次 HMUX 的数据通路）：ACC_{i-1}（RLWE 多项式流）从输入 buffer 进入 → Rotation Unit 做 X^(a_i) 旋转与减法 → Decomposition Unit 按位分解 → FFT Unit（BU 个并行 butterfly、conflict-free 地址生成）转到频域 → VMA Unit 把 BSK 多项式与 ACC 多项式逐系数乘加（外积）→ IFFT Unit 转回时域 → 输出经 double buffer 由 D2D 传给下游 HC。所有单元以多项式系数粒度（PCG）流水，系数算完即流入下一单元。融合组内：HMUX 输出回馈本地输入再执行下一 HMUX；inter-HC batching：多个 RLWE 密文交错注入避免气泡。IP=256 表示 VMA 单元硬件并行度（同时取 256 个 BSK 系数 = 内部 BSK SRAM 带宽）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：RTL 用 Synopsys Design Compiler + TSMC 28nm 库综合（1.2 GHz），D2D PHY 按 UCIe Advanced 建模（16 GT/s、64-bit → 1024 Gbps）；性能用自研 cycle-accurate 模拟器（基于 F1 [32] 方法）逐 cycle 建模各功能单元与 D2D 通信。使用：作为多芯粒系统的复制单元（全部 HC 架构相同、仅 HC0 加 VPU），支持按 n 与 C 的关系处理任意长 HMUX 链（n>C 时 RLWE 在环上多次循环）；设计空间探索中单 die 面积约束 50-150 mm²、C∈[4,32]，最优 C=12（C>12 时 D2D PHY 面积税使性能/面积下降）。

涉及论文标题：
- Unlocking Pipeline Parallelism for Bootstrapping: A Pipelined Multi-Chiplet TFHE Accelerator
