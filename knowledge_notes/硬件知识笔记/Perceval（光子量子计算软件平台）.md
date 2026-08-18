## Perceval（光子量子计算软件平台）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Perceval 是 Quandela 开发的离散变量光子量子计算（Discrete Variable Photonic Quantum Computing）开源软件平台（Heurtel 等, Quantum 7, 931, 2023；https://github.com/Quandela/Perceval）：提供量子程序描述、光子线路构建与仿真、以及向真实光子 QPU（Quandela 云平台）提交执行的完整工具链。核心组件：pc.Processor（定义光学处理器）、pc.Circuit（组合线性光学元件：分束器、相移器、波导）、pc.Source（光子源模型）、探测器模型、FFCircuitProvider（前馈控制电路，支持条件操作——正是论文实现融合容错恢复测量所需的组件）、以及模拟/云执行接口（execute()、pc.remote）。论文对 Perceval 的使用：(1) 真实硬件实验——用 Perceval 构建树编码融合的光学硬件电路（双轨 dual-rail 编码：H、V 偏振两个光子模式编码一个 qubit；融合电路 = 两 qubit 光子模式置换 + 相移 + 两个分束器），实测硬件特征 HOM 不可区分度 92.0%、透射率 5.16%、g^(2)=2.0%；(2) feed-forward 控制——FFCircuitProvider 实现融合结果触发的条件 X/Z 测量；(3) 模拟器基础——论文引用的 PQC 模拟器（[24]，Motivation 中的 Max-Cut QAOA 模拟 Fig.3）。论文的融合错误率等参数也部分取自 Perceval 生态（如 HOM 可见度、双轨编码惯例）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
Perceval 在硬件验证中的运转流程（树编码融合电路，Fig.13）：
```
# 输入: 双轨编码 qubit (H/V 偏振两个光子模式)
# 1) 构建融合电路 (pc.Circuit)
fusion = pc.Circuit(4)               # 4 个光子模式 (两 qubit 各 2 模式)
fusion.add(permutation)              # 模式置换
fusion.add(phase_shift)              # 相移
fusion.add(BS, BS, ...)              # 两个分束器 (beam splitters)
# 2) 定义处理器与探测器
proc = pc.Processor("slos", fusion)  # 线性光学处理器
proc.add_detector(...)               # 光子探测器 (SNSPD 模型)
# 3) 前馈控制 (FFCircuitProvider)
ff = pc.FF.FFCircuitProvider(proc)
ff.add_conditional_measure(...)      # 融合结果 -> 条件 X/Z 测量 q_i^a/q_i^b
# 4) 执行
results = proc.execute(shots=N)      # 本地仿真
remote_results = pc.remote(...)      # 提交 Quandela 云 QPU (真实硬件)
# 输出: 融合结果分布 -> PST / IST 指标计算
```
该流程对应论文真实硬件实验：6~12 qubit QAOA 程序经 MemTree 编译 → Perceval 电路 → Quandela 云硬件执行 → 与 RUS+photonic、Qiskit+IBM Torino 对比。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现/使用：Perceval 开源（https://github.com/Quandela/Perceval，pip 安装 perceval-quandela；Quantum 7, 931, 2023 论文）；支持本地方真与 Quandela 云 QPU 远程提交（Quandela 提供 24-photon modes 平台）；组件包括 Processor/Circuit/Source/FFCircuitProvider 等。使用场景：(1) 光子线路设计与仿真验证；(2) 向真实光子硬件提交实验（本论文）；(3) 作为 PQC 算法/编译研究的模拟环境（论文 Motivation 的 QAOA 模拟即用它）。注意：论文自研的 spin memory 误差感知模拟器与 Perceval 是不同工具（自研模拟器未开源），Perceval 用于真实硬件电路与 feed-forward。

涉及论文标题：
- Photonic Quantum Computing on Spin Memory Architecture with Tree-Encoded Fusion
