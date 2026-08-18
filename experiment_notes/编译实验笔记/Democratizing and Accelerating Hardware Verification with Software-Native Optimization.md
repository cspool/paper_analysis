## Democratizing and Accelerating Hardware Verification with Software-Native Optimization

- 属于编译框架的实现是什么？实验比较什么？
  - 实现为 UCV (UnityChip Verification) 的 RTL 打包/代码生成流水线，前端项目生成器为 Picker（https://github.com/XS-MLVP/picker ）。打包流程：Verilog/SystemVerilog DUT(+可选 UVM VIP) 经目标模拟器编译为动态库 → 链接由 RTL 源码自动生成的 backend adapter（提供事件控制、数据类型封装的胶水逻辑，隔离 Verilator/GHDL/VCS 差异）→ SWIG 生成多语言绑定（Python/C++/Java/Go/Scala）→ 与各语言原生实现的 platform runtime 组合成可 import、可版本化的软件包。非侵入式内省也属编译侧：对可分析后端（Verilator、GHDL）编译期从模拟器发射的 C++/IR 工件提取寄存器指针与"优化映射逆变换"，运行时加载为小体积 symbol-pointer 数据库（MemD 调试路径），替代 DPI/VPI 导出信号（其导致二进制最多膨胀 4×、每寄存器更新加检查分支、禁用优化合并）。
  - 实验比较：Q1 调试性能——XData 三模式 MemD/VPI/DPI vs VPI 与 cocotb 基线（Fig.11）；Q4 软件时序开销——Picker 生成的多语言 wrapper（C++/Python/Java/Go）相对裸 Verilator 的吞吐/内存开销（Fig.13）；Q3 另报告 UCV+ vs UVM 的编译时间（Table IV）。
- 硬件平台是什么，配置是什么。
  - 论文 Table III：2× AMD EPYC 7773X 64 核、16×64GB DDR4 3200 MT/s、Ubuntu 20.04.1（内核 5.15.0-127-generic）、GCC 11.4.0、OpenJDK 17；Verilator v5.034、CIRCT 1.66、SWIG 4.2.1、Python 3.8.10、Golang 1.23.4。artifact：Ubuntu 22.04.5 LTS、Verilator v5.026、Picker v0.9.0 master、cocotb v1.9.2、Python 3.10.12、OpenJDK 17.0.18、Golang v1.25.1；最低 8 核/64GB 内存/20GB 磁盘。
- 开源编译框架是什么。修改了什么。
  - 不修改开源模拟器本体：Verilator（v5.034/v5.026）为可插拔后端，GHDL 同样支持；VCS 默认走 VPI/DPI，实验性指针模式需 hack 构建中间工件。新增/修改的是编译流水线外围：自动生成 backend adapter 与 SWIG 绑定、编译期指针与逆映射提取、namespace 级符号隔离（解决多实例全局符号冲突）。全开源：平台 https://github.com/XS-MLVP/picker ；实验 artifact https://github.com/Makiras/UnityChipExp （MIT，Zenodo 10.5281/zenodo.19447034，Docker ghcr.io/makiras/unitychipexp:latest）。
- 开源情况。基于开源文档和论文，使用例子解释编译框架如何使用？作用是什么？至少具体到编译框架输入到输出的全过程。
  - 输入到输出全过程：输入 DUT Verilog/SystemVerilog（示例：XiangShan BPU 的 RTL）→ Picker 调 Verilator 把 DUT(+可选 VIP) 编译为 C++ 动态库 → 从 RTL 源码/模拟器发射工件生成 backend adapter（事件控制、类型封装、信号与内部结构绑定元数据）→ SWIG 生成语言绑定 → 与 platform runtime（三层：HLL 用户 API / HLL 平台 runtime / backend adapter）链接成软件包 → 用户 import 该包、注册到语言异步运行时（asyncio/Boost.Asio），用 pytest/JUnit 写测试。作用：把"运行测试依赖模拟器工程构建"解耦为"RTL 是普通软件依赖"；编译期指针数据库支撑 MemD 调试（比 VPI 快 4.8×–17.5×、比 cocotb 快 16.3×–25.2×，峰值内存较 VPI 降 13%–77%、较 cocotb 降 46%–77%）；多语言事件驱动 wrapper 在 XiangShan 上吞吐损失 ≤3%（vs 裸 Verilator），CoupledL2/RocketChip 上 14%–55%。
