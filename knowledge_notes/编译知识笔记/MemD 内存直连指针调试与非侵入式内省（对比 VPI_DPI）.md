## MemD 内存直连指针调试与非侵入式内省（对比 VPI/DPI）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MemD（memory direct）是 UCV 非侵入式内省层（introspection）的调试数据通路：不经过模拟器的 VPI/DPI 调试接口，而是编译期从模拟器发射的 C++/IR 工件提取内部寄存器/状态的指针与"优化映射逆变换"，运行时加载为小体积 symbol-pointer 数据库；调试时按需重算信号值（对优化合并的计算节点做逆映射恢复原始值），既不为调试保留额外中间态、也不依赖模拟器调试接口导出信号。同一机制支撑热补丁（利用寄存器只在采样沿生效：周期精确模型在采样前覆写寄存器值；涉及组合路径时经指针级访问安装函数指针钩子包裹/替换生成的组合更新例程）与单进程多实例（namespace 级符号隔离，解决多实例全局符号名冲突）。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
传统调试路径（DPI/VPI）的代价链：为每个导出信号在模拟器内生成支持代码 → 二进制膨胀最多 4×（坏 cache 局部性）+ 可写/可锁 VPI 路径在每个寄存器更新前插检查分支 + 外部可控寄存器禁用计算节点合并优化 → 三者叠加使 Verilator 开 VPI 损失 70% 性能、程序翻倍。MemD 路径：对可分析后端（Verilator、GHDL）编译期提取指针与逆映射（VCS 默认走 VPI/DPI，实验性指针模式需 hack 构建中间工件）→ 运行时软件层按需计算调试视图，完全避开 per-access 调用路径。论文用"cocotb 以手工匹配的 -O3 编译后性能追平 XData 的 VPI 模式"验证：调用路径（而非编译优化设置）是核心瓶颈。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
性能数据（本文 Fig.11，AMD EPYC 7773X、Verilator 8 线程）：MemD 比 VPI 快 4.8×–17.5×、比 cocotb（内部走 VPI）快 16.3×–25.2×；峰值内存较 VPI 降 13%–77%、较 cocotb 降 46%–77%。DPI 在小信号集下运行时接近 MemD，但只暴露代码选定的固定信号集、成本随信号范围增长直至接近 VPI。典型使用：BPU 验证中把 SRAM 预测器表以 MemD 派生表视图逐项检查（而非从波形反推表语义）；GDB-for-QEMU 风格的 SoC 指令级 RTL 调试器。Web 证据：DPI-C 调用开销低于 VPI、VPI 对象为大型句柄结构且上下文切换代价高（https://community.cadence.com/cadence_technology_forums/f/functional-verification/37006/performance-difference-between-api-pli-vpi-dpi-tcl-verilog-functions-w-r-t-object-browsing-read 、https://bbs.eetop.cn/thread-888714-1-62.html ）；Verilator `/*verilator public*/` 原生指针访问比 DPI 导出更快但可移植性差（https://man.sourcentral.org/debian-wheezy/1+verilator ）。

涉及论文标题：
- Democratizing and Accelerating Hardware Verification with Software-Native Optimization
