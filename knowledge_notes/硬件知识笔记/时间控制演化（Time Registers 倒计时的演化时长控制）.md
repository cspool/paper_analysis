## 时间控制演化（Time Registers 倒计时的演化时长控制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DS-ISA 用"指定时长"而非"收敛检测"来控制 DSU 演化终点：N_EVOLVE/C_EVOLVE 的 Time 信号放上 Time Bus、写入每个参与组的 Time Registers 并启动倒计时，演化持续该时长；计时归零后选择逻辑切回 Idle Registers，重新施加默认锁定信号，把组件锁存在演化后的状态。选时依据是先前 DSU 文献的物理时间尺度（推断 ~100ns 级 [InstaTrain]，训练 µs 级，DE 求解/优化 µs 级 [DS-TIDE/BRIM]），200 MHz 控制器下取 10ns 为最小演化时长、更长相位靠重复演化循环构建。论文明确把收敛检测（如用空闲 ADC 做 variation-detection 模块）留作扩展——时间控制更基础，能覆盖不要求平衡态的负载，接口干净且确定。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
硬件流程（一次 inference，100ns 演化）：N_EVOLVE 解码 → GM 选中输出组 → 选择逻辑从 Idle Registers 切到 NLM Registers（解锁演化节点）→ 同一拍 Time 信号广播到 Time Bus → 各参与组 Time Registers 写入 100ns 对应计数值并开始每周期递减 → 演化期间控制器可并行发绿放行指令（如为下一任务 N_LOAD 数据）→ 倒计时到零 → 选择逻辑切回 Idle Registers → 输出节点被锁定在演化终态 → N_STORE 经 ADC 读回。节点控制与耦合控制共享该 timer 机制（C_EVOLVE 同样按 GM 选定耦合子矩阵 + Time 倒计时）。评估中 optimization/DE/training 各用 10ns × 100 次循环模拟更长相位与迭代结构。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：每参与组一个 Time Register + 共享 Time Bus + 与选择逻辑联动的计数归零信号；时长是 E-Type 指令的 2B 立即数（Imm_time）。使用方式：应用按收敛尺度设定演化时长（推理短演化、训练/优化用循环近似长时间演化）；需要精确收敛的应用可未来加收敛检测扩展。局限：固定时长不保证达到平衡态，论文评估刻意固定时长以隔离控制架构开销。

涉及论文标题：
- DS-ISA: Instruction Set Architecture for Dynamical System Units
