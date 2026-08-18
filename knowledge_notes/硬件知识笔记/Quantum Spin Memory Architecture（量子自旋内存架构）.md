## Quantum Spin Memory Architecture（量子自旋内存架构）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
量子自旋内存架构（quantum spin memory architecture）是一种光子量子计算机（PQC）硬件架构，基于半导体量子点（QD）发射器（如 InGaAs QD 腔体），其核心机制是用"自旋内存"存储量子比特并受控地将其发射为光子图态（相关方案见 Gliniasty 等的 Spin-Optical Quantum Computing Architecture, Quantum 8, 1423, 2024；Huet 等的单固态量子发射器确定性可重构图态生成, Nat. Commun. 16, 4337, 2025）。它是三大 PQC 图态生成架构之一：(1) all-photonic——SPDC 贝尔对 + 线性光学融合（PsiQuantum）；(2) emitter-based——相互作用的量子发射器（理论确定性，emitter-CZ 未实验演示）；(3) spin memory——QD 发射器 + 线性光学融合硬件，实验已演示。论文采用 spin memory 架构，理由：(1) caterpillar 态可由该架构确定性、可配置地生成（小规模已实验验证 [29]），其灵活性支持树编码容错方案的集成；(2) 平台已可通过云访问（Quandela 24-photon modes PQC [2]），是 near-term 可验证的架构。论文以该架构为目标的编译器（MemTree）与融合方案（树编码）均面向其硬件特性设计：caterpillar 态是图态生成的资源态，融合用线性光学完成。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
spin memory 架构的硬件运转流程（图态生成，Fig.2(c) 红框）：
```
# 硬件组成: InGaAs QD 腔体 + LA 激发脉冲源 + OSRP 脉冲源 + 线性光学融合硬件 + SNSPD 探测器
# 1) caterpillar 态发射（自旋内存核心机制）
for each main_path_qubit:
    LA(pi/2)        # 纵向声学激发脉冲 -> 发射光子（与 QD 自旋纠缠）
    OSRP(pi)        # 光学自旋旋转脉冲 -> 自旋旋转, 定义叶分支结构
#    每个 caterpillar 态: 12 ns 初始化 + 0.6 ns/qubit 发射, 上限 30 qubit
# 2) 图态拼接（线性光学）
fusion(caterpillar_A, caterpillar_B)   # Type-II 融合 (HWP+PBS), σ_fus=99.75%
# 3) 测量执行（探测）
SNSPD 单光子探测 (<50 ps 延迟) -> 融合结果判定 -> feed-forward
# 硬件时序参数（论文模拟器配置）:
#   t_cycle = 30 ns/层; 1-p_fail = 0.75; T2 = 2.34 μs; p_eras ≈ 10% (真实硬件)
```
架构关键特性：发射器互相隔离、无相互作用 → 光子源间无 crosstalk → CZ（融合）操作并行度高、电路执行时间短（论文在真实硬件上相对超导 QPU 验证此优势）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：(1) 硬件——InGaAs 半导体量子点腔体 + LA 激发 + OSRP 光学自旋旋转，实验演示见 Huet 等（Nat. Commun. 16, 4337, 2025）；(2) 云平台——Quandela 提供 24-photon modes PQC 云访问（其 whitepaper 见 quandela.com），论文真实硬件实验在其上完成，光学电路用 Perceval 工具包构建；(3) 模拟——论文自研 realistic error-aware 模拟器按上述配置模拟（未开源）。使用场景：near-term 光子 MBQC 计算（室温运行、长退相干、天然适配量子网络集成）；论文用它验证"解决融合擦除后 PQC 相对超导硬件的性能优势"（QAOA 6~12 qubit，PST/IST 指标优于 RUS+photonic 与 Qiskit+IBM Torino）。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
