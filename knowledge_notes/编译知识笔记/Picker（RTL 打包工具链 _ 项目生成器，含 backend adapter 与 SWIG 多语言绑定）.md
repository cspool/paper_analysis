## Picker（RTL 打包工具链 / 项目生成器，含 backend adapter 与 SWIG 多语言绑定）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Picker（https://github.com/XS-MLVP/picker ）是 UCV 平台的前端项目生成器，把 RTL 设计模块（.v/.sv/.scala）封装成二进制软件库（.so）并以引脚级（Pin-Level）操作暴露给多种高级语言（C++、Python、Java、Scala、Golang、Lua），口号 "Pick your favorite language to verify your chip"。两个子命令：`export` 把 RTL 项目源码导出为软件库；`pack` 把 UVM 事务打包为 UVM agent + Python 类（UVM↔Python 通信）。依赖 xspcomm 基础类型库（XS-MLVP/xcomm）、slang、fmt、verible；Python 绑定经 SWIG（≥4.2.0）生成，可打包 pip wheel（xspcomm + picker）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
本文的打包流水线：输入 DUT Verilog/SystemVerilog（+可选 UVM VIP）→ 目标模拟器（Verilator/VCS）把 DUT 编译为动态库 → 从 RTL 源码自动生成 backend adapter（提供事件控制、数据类型封装的胶水逻辑，隔离各模拟器差异，是 UCV 三层结构"用户 API / 平台 runtime / backend adapter"的最底层）→ SWIG 生成多语言绑定 → 与各语言原生实现的 platform runtime 组合成可 import、可版本化的软件包。用法示例：`picker export -f top.sv --sim verilator --lang python -T top` 生成 Python 包，随后可直接用 pytest 驱动 DUT——把"运行测试依赖模拟器工程构建"解耦为"RTL 是普通软件依赖"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
构建：`make init && make && sudo -E make install`（CMake ≥3.11、GCC C++20、Python 3.8+、Verilator ≥4.218）。典型场景：以 .so 交付 DUT 使验证方不必接触 RTL 源码；同组织 Toffee（https://github.com/XS-MLVP/toffee ）是基于 Picker 把 UVM 方法学引入 Python 的验证框架。本文场景：XiangShan/RocketChip/CoupledL2 的 DUT 打包与 C++/Python/Java/Go wrapper 生成，Fig.13 评估各语言事件驱动 wrapper 相对裸 Verilator 的吞吐/内存开销。SWIG（http://www.swig.org/ ）本身是通用 C/C++↔脚本语言绑定生成器，在 Picker 中承担 backend adapter→HLL 的绑定生成。

涉及论文标题：
- Democratizing and Accelerating Hardware Verification with Software-Native Optimization
