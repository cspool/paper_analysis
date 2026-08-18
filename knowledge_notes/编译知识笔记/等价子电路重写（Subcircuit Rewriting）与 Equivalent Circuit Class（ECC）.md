## 等价子电路重写（Subcircuit Rewriting）与 Equivalent Circuit Class（ECC）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 等价子电路重写是量子线路优化的主流方法：预先生成/验证小规模"等价电路类"（ECC，Equivalent Circuit Class——功能等价的子电路集合），运行时在线路中匹配某个子电路并用类内更优者替换。代表性系统：Quartz（PLDI'22，自动搜索+形式化验证生成 ECC，成本搜索应用；https://github.com/quantum-compiler/quartz）、QUESO（PLDI'23，"Synthesizing Quantum-Circuit Optimizers"，用相位多项式恒等式过滤 PIF 增强 ECC 生成，Java 实现；https://github.com/qqq-wisc/queso，已并入 qqq-wisc/guoq）、Quanto 等。
- 核心局限（本文 Motivation）：ECC 模式受规模限制——实践仅覆盖 3 qubit/6 门窗口，构造更大等价类计算不可行；因此重写只能做局部（short-range）优化，捕获长程结构需要大量重写且随电路规模迅速退化。
- 与 PhasePoly 关系：PhasePoly 正交于重写框架——前者用结构化 parity 矩阵推理捕获长程 CNOT/Rz 结构，后者擅长小电路局部化简；组合管线"PhasePoly→重写"收益最大（重写后接 PhasePoly 再增 ≈6–13%，反序仅 0.75–1.25%）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（Quartz 为例）：① 离线 ECC 生成：对给定门集（IBM/Rigetti/Nam 等），用 SAT/搜索枚举给定大小内所有子电路、验证等价性（Quartz 用自动验证），形成 ECC 集；② 在线优化：把输入线路切窗（3-qubit/6-gate），匹配 ECC，用成本搜索（cost-based backtracking）选择替换序列使总门数最小；重复直至无改善。QUESO 在 ① 中用相位多项式恒等式过滤快速判定子电路等价，生成更多/更准的 ECC。
- 实验对比（本文 Q2/Q3）：Quartz 平均总门减 22.17%、CNOT 16.88%；QUESO 27.83%/20.70%；PhasePoly 34.70%/26.83%。电路规模分组：<200 门小电路上重写可打平/超过 PhasePoly（10 中 2 个）；200–500 门 10 中 3 个打平；>500 门除 1 个 QAOA 外全部显著落后。大电路族（MCX/Adder/HWB）上 Quartz/QUESO 在 2 小时预算内饱和或失败，PhasePoly 线性增长（Fig.5/Fig.14）。
- 作用：量化说明"固定窗口局部重写"与"结构化全局推理"的分界线，是论文论证相位多项式应作一等编译阶段的证据。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：Quartz C++ 编译器（OpenQASM 2.0/3.0 输入，支持符号参数，gate-set context shift）；QUESO Java（Maven 构建，OpenQASM，支持 IBM/Rigetti/Ion trap/Nam 门集；Docker artifact Zenodo 10.5281/zenodo.7809285）。评估设置：推荐 3-qubit/6-gate 子电路、每电路 2 小时预算（本文遵循）。使用场景：量子逻辑优化流水线的局部化简阶段；与全局优化（PhasePoly）串联。
- 与本文关系：作为 baseline 与组合对象；结论是"专门的相位多项式 pass + 局部重写"优于"把相位信息散进重写框架"。

涉及论文标题：
- Leveraging Phase Polynomials for Quantum Circuit Optimization
