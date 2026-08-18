## Caterpillar 态生成与光学自旋旋转脉冲（OSRP）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Caterpillar 态是 spin memory 架构生成的分支链图态（主路径线性链 + 每主路径顶点挂叶量子比特），其生成机制由两类脉冲驱动（Fig.2(c)）：(1) 纵向声学（longitudinal-acoustic, LA）激发脉冲 LA(π/2)——施加于 QD 腔体，迭代发射线性纠缠的光子（每个 qubit 一个激发脉冲）；(2) 光学自旋旋转脉冲（optical spin rotation pulse, OSRP(φ), φ=π）——插入到激发脉冲之间，通过旋转 QD 自旋改变发射光子的纠缠结构，从而在纯线性链之外产生叶分支，得到 caterpillar 结构。生成的 caterpillar 态可作为通用图态生成的资源态（定义见 Pettersson 等 PRX Quantum 6, 010305, 2025；物理制备见 Huet 等 Nat. Commun. 16, 4337, 2025）。硬件参数：初始化 12 ns + 每 qubit 发射 0.6 ns、单个 caterpillar 上限 30 qubit、OSRP fidelity 99%。论文用 caterpillar 态：(1) 作为 MemTree 生成计划的原子资源态（程序无关、结构按需确定）；(2) 在其上组装树编码逻辑量子比特（主路径 q_root + 叶 qubit + 4-qubit 线性图融合），形成容错融合的基础结构。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
caterpillar 态生成在硬件中的运转流程（时序）：
```
# QD 腔体上的脉冲序列（生成 caterpillar 态）
time 0    : initialization (12 ns)
t+0.6 ns  : LA(pi/2)        # 发射 q1 (主路径起点, 与自旋纠缠)
t+1.2 ns  : OSRP(pi)        # 自旋旋转 -> 准备分支
t+1.8 ns  : LA(pi/2)        # 发射 q2 (叶分支或主路径下一 qubit, 取决于自旋态)
t+2.4 ns  : OSRP(pi)
...
# 输出: caterpillar 态 = {主路径链 m1-m2-...-mk} + {每顶点挂 b 个叶 qubit}
# 约束: 单 caterpillar ≤ 30 qubit (near-term 硬件上限)
# 后续: 叶子经 Z 测量分离出的 4-qubit 线性图融合组装成树编码逻辑量子比特
```
硬件配置：InGaAs QD、LA(π/2) 与 OSRP(φ=π) 交替脉冲、QD-cavity（微腔增强）；论文模拟器按此配置模拟（12 ns + 0.6 ns/qubit、30-qubit 上限、OSRP 99%）。制备参数 b_prep=6（>b=4）：同 timestep 并行尝试分支制备，成功分支 <b 则下 timestep 重试（真实硬件 83.3% 单步成功率、97.1% 两步内）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：spin memory 硬件的脉冲级控制——LA 激发脉冲由激光驱动 QD 发射光子，OSRP 由光学脉冲旋转 QD 自旋，两者交替组成发射序列；实验实现见 Huet 等（Nat. Commun. 16, 4337, 2025，单固态量子发射器确定性可重构图态生成）；模拟实现见论文自研模拟器。使用场景：任何 QD 发射器类 PQC 的图态生成；caterpillar 态是 MemTree"程序无关、按需确定结构"的光子源利用策略的物理基础（光子利用率 ~10% vs OneAdapt 0.03%）。论文未开源。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
