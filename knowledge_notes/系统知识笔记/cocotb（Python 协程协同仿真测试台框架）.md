## cocotb（Python 协程协同仿真测试台框架）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
cocotb（COroutine-based COsimulation TestBench）是开源 Python 验证框架：测试台用 Python async/await 协程编写，运行在 RTL 模拟器进程内，经分层软件栈（Python API → PyGPI C 扩展 → GPI Core → 各模拟器接口库 libcocotbvpi/vhpi/fli → IEEE VPI/VHPI 或 FLI）把 Python 调度器挂到模拟器回调上。模拟器按内核事件调度在回调点调用 cocotb，每次调用给软件一个短暂执行窗；cocotb 的 dispatcher 把内核通知映射为 trigger 就绪、恢复对应协程、经模拟器外部接口读写信号。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
事件往返流：Python 协程注册 Trigger（如 RisingEdge(clk)）→ PyGPI 经 GPI 注册模拟器回调（vhpi_register_cb）→ 模拟器到事件点调用 GPI C 处理器 → PyGPI handle_gpi_callback → Python 调度器 react() 对等待该 trigger 的协程 next() → 协程读写 dut.signal.value。关键语义：cocotb 不拥有时间推进、不定义观测边界——时序/顺序/可观测性语义全部继承自模拟器内核，软件执行被切成回调界定的短切片；信号访问为动态、按名字查找（层次路径 → handle），需人工掌握 RTL 结构。本文定位的两个缺陷：① 回调可能在信号稳定前触发（组合路径 b=f(a)，a 变化后立即回调读到 b 旧值，cocotb Issue#3110）；② 模拟器拥有的回调循环与软件异步运行时（asyncio）不兼容（cocotb 讨论 #3994）——"simulator-centric"的代表。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
安装：pip install cocotb（本文 artifact 用 v1.9.2）；测试用 @cocotb.test() 装饰器 + `await Timer()/RisingEdge()`；支持 Verilator、Icarus、VCS、Questa 等模拟器。本文将其作为 Python 事件驱动验证基线：UCV MemD 比 cocotb 快 16.3×–25.2×、峰值内存低 46%–77%（cocotb 内部走 VPI；-O3 编译后性能追平 XData 的 VPI 模式，证明调用路径是瓶颈）。Web 证据：DeepWiki cocotb 架构（https://deepwiki.com/cocotb/cocotb/1.1-architecture ）、模拟器集成（https://deepwiki.com/cocotb/cocotb/7-simulator-integration ）、timing triggers（https://deepwiki.com/cocotb/cocotb/4.1-timing-triggers ）。

涉及论文标题：
- Democratizing and Accelerating Hardware Verification with Software-Native Optimization
