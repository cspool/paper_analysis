## NWQEC（PNNL 容错量子电路转译框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- NWQEC 是 Pacific Northwest National Laboratory（PNNL）开源的 fault-tolerant quantum circuit（FTQC）转译工具包（https://github.com/pnnl/nwqec，MIT 许可，NWQWorkflow 生态成员），C++17/Python 实现：解析 OpenQASM 2.0 为内部电路表示，提供一系列 FTQC 编译 pass——Clifford+T 转换（内嵌高性能 C++ gridsynth 后端，约 20× 加速）、Pauli-Based Circuit（PBC）生成、Tfuse（T-count 优化，最多 -30%）、以及 TACO 的 Clifford reduction（保持电路并行的 Clifford 门消除）。提供 `nwqec-cli` 与 `gridsynth` 两个 C++ CLI，`pip install nwqec` 有预编译 wheel（Linux x86_64/ARM64、macOS Intel/Apple Silicon）。构建：CMake ≥3.16 + C++17，GMP/MPFR 自动管理（缺省下载预编译二进制）。
- 论文用途（TACO）：TACO 作为 NWQEC 的一个 Clifford reduction pass 集成；论文 artifact（Zenodo DOI 10.5281/zenodo.19449157）含集成 TACO 的 NWQEC 自包含副本与 benchmark 电路、figure 脚本。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 在 NWQEC 中的运转流程（TACO 转译，nwqec-cli 输入到输出）：
```
输入: 算法级 QASM（如 QASMBench 18 比特 qft，783 门）
① 解析 OpenQASM 2.0 → 内部电路表示
② 动态分解 pass（FTQC 导向：逐门选低 FTQC 代价分解、Rz(π)→Z、合并单比特门）
③ Clifford+T 合成 pass（内嵌 gridsynth，ε=10⁻¹⁰）
④ Clifford reduction pass（TACO：MA Normal Form + Toffoli Clifford 相消，
   见对应条目）→ 输出 Clifford 减少后的 Clifford+T 电路
⑤ 可选 PBC 生成 / Tfuse 优化（nwqec-cli 子命令）
输出: 优化后 Clifford+T 电路（QFT：Clifford 降 98.6%、T 门 9,529、深度 6,598）
```
- 构建/使用：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`；artifact 的 `./run_all.sh` 一键复现 Fig.5/14/17/20/21/22，单图脚本 `./plot_fig_5.sh`（--force-collect 强制重跑），输出 results/*.csv 与 figures/*.pdf，全部实验 CPU-only、数分钟内完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：模块化 pass 式转译系统（C++17 核心 + Python 绑定/脚本），Tableau 基 IR 支持高性能 PBC pass；依赖 GMP/MPFR（合成后端）。使用：`pip install nwqec` 或源码 CMake 构建；CLI `nwqec-cli <in.qasm>` 输出转译结果。作用：把"算法电路 → 容错 Clifford+T/Pauli 电路"的全流程（分解、合成、Clifford 消除、T 优化）做成可复现开源工具，TACO 的 Clifford reduction 是其中面向"Clifford 瓶颈"的最新 pass。相关：同生态 NWQASM、QASMTrans、NWQSim、QASMBench。

涉及论文标题：
- Transpiler-Architecture Co-Design to Curb Clifford Costs in Fault-Tolerant Quantum Computing
