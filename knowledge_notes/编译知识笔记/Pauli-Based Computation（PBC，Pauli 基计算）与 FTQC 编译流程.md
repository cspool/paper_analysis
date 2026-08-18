## Pauli-Based Computation（PBC，Pauli 基计算）与 FTQC 编译流程

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- PBC 是容错量子计算（FTQC）的编译执行模型（Litinski "A Game of Surface Codes" 系统化）：把 Clifford+T 线路转译为 Pauli product rotations（PPR）序列，再以 Pauli product measurements（PPM）+ magic state teleportation 执行。逻辑链：①任意 Rz(θ) 经 Solovay-Kitaev/GridSynth 分解为 Clifford+T；②Pauli 乘积旋转 P_θ=exp(−iPθ)，其中 S=Z_{π/4}、T=Z_{π/8}，标准分解 H=Z_{π/4}X_{π/4}Z_{π/4}、CNOT=(Z⊗X)_{π/4}(I⊗X)_{−π/4}(Z⊗I)_{−π/4}；③Clifford 门可吸收进最终测量（Clifford 把 Pauli 映射到 Pauli），Pauli 算子按交换/反交换规则化简（若 P,P' 交换则 P_{π/4} 可越过 P'_θ；若反对易则 P'_θ 变成 (iPP')_θ）；④转译后只剩 PPM 序列，在表面码上以 lattice surgery 执行。O3LS 的编译流水线（Fig.5）：高层量子算法 QASM → ①Clifford+T 分解（GridSynth，误差容限 10⁻⁵）→ ②Pauli-based transpilation（PBC）→ ③表面码级映射与调度（O3LS 四模块）。
- 从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 编译流程例子（O3LS 输入到输出全过程）：输入 = adder_28 / ising_26 / swap_test 等 QASM（MQT Bench [41] / FTCircuitBench [21]，后者部分源自 QASMBench）→ GridSynth 把任意旋转分解到 Clifford+T（容限 10⁻⁵）→ Pauli-based transpilation 转成 PPR 序列（Pauli DAG 表达依赖，Clifford 吸收进测量）→ O3LS Module 1 自动布局搜索（squeezed 布局）→ Module 2 Y-synthesis（Y 门分解 + 算子抵消）→ Module 4 边感知初始映射 → Module 3 松散调度 → 输出可执行 lattice surgery 指令序列（时间步 + 每时间片布局 + ancilla 路径）。开源工具：Qiskit 的 LitinskiTransformation pass（GitHub PR #15217）、PennyLane Pauli Product Rotations 编译插件均实现 PBC 转译。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现为编译器的前端转译 pass + 后端调度 pass。前端：读取 QASM、GridSynth（Ross-Selinger，qiskit-gridsynth-plugin）合成、Pauli 化简（交换/反对易规则、Clifford 吸收）。后端：O3LS 的布局/合成/映射/调度四模块，输出时间步与 LER。评估：STIM（d=9、p=10⁻³）表征原子操作 + PyMatching 2 解码 + 分层 LER 模型。论文未声明开源（arXiv:2604.15099，GitHub 仓库未能定位）。
- 补充（TACO 论文）：TACO 把 PBC 作为主要 baseline 并量化其核心缺陷——把 Clifford 门（尤其 CNOT）穿过非 Clifford 门换到末尾会生成高权重 multi-qubit π/4 旋转：20 比特 QFT 经 PBC 后操作权重最高达 16 比特（图 5），18 比特 QFT 的 40,777 个门深度 39,055、每层仅约 1.08 个门（图 2c），门级并行坍塌。PBC 的硬件代价：Compact 架构（1.5n+3 tile、每 qubit 单边暴露）执行 Y-edge 操作需 3 个额外 ancilla 与最多 9 cycles；Fast 架构（2n+√(8n)+1 tile）面积大；且因每层单操作、无路由优化空间（LSQECC 无法作用于 PBC）。TACO 对比结论：保持并行（操作权重 ≤2）的 Clifford 消除 + 资源局部性架构，使 18 比特 QFT 深度降到 6,598（比 PBC 浅 5.9×）、执行时间最高 21.9×（几何平均 4.4×）优于 PBC。
- 涉及论文标题：
- O3LS: Optimizing Lattice Surgery via Automatic Layout Searching and Loose Scheduling
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing
