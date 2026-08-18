## UVM（Universal Verification Methodology，通用验证方法学）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
UVM 是 Accellera 标准化的 SystemVerilog 验证方法学（IEEE 1800.2），为可复用测试台定义标准化组件与类库：driver（把抽象事务按时间对齐驱动到 DUT 接口）、monitor（信号传播稳定后采样并重构事务）、scoreboard（DUT 实际输出与参考模型预测比对判定 pass/fail）、sequencer/sequence（事务流生成）、reference model（预期行为模型）、coverage（功能覆盖 covergroup/coverpoint 引导回归）。它是工业界主导的验证框架，但门槛高、与 RTL 模拟器事件调度强耦合。本文把 UVM 作为传统验证架构基线（Table I），并把"复用既有 UVM 验证组件（VIP）"作为 UCV 需解决的组合难题（§III-B）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
UVM 事务流（sequence → DUT → scoreboard）：sequence 生成事务 → sequencer 经 seq_item_port.get_next_item() 交给 driver → driver 经 virtual interface 在采样沿前驱动信号波形（time-aligned driving）→ DUT 执行 → monitor 在传播稳定后采样接口信号（post-stable sampling）、经 uvm_analysis_port 广播重构事务 → scoreboard 收 monitor 的实际事务与 reference model 预测比对。调度纪律由模拟器事件相位保证（driver 采样沿前写、monitor 稳定后读）。本文视角下的架构缺陷：UVM 组件依赖模拟器的事件驱动控制流与事务传输，而软件侧框架（cocotb 等）假设验证逻辑全在软件执行、缺乏对应机制，跨域复用因此需要事件同步（UCV 用 XEvent 注册表）与事务调度（XSocket 线程池）两项补充；传统软件-UV M集成用共享内存 relay 的进程级同步带来等待/序列化开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：SystemVerilog 类库（uvm_pkg），工厂/配置/phase 机制（build/run/cleanup），Synopsys VCS、Cadence Xcelium、Siemens Questa 原生支持。VIP（Verification IP）是把 driver/monitor/RM 等打包的可复用组件（Accellera VIP 推荐实践 https://www.accellera.org/images/downloads/standards/uvm/VIP_1.0.pdf ）。本文用法：UCV+（UCV 启用 UVM 支持）在 NoC/ICache 验证中对比纯 UVM 共享内存 relay（Table IV）；社区研究中 UVM 经验参与者 16 人、达成 100% 行覆盖。Web 证据：组件数据流与 scoreboard 同步（https://verificationacademy.com/forums/t/scoreboard-evaluating-before-monitor-updates-in-systemverilog-testbench-dff/51535 ）、Mehta《ASIC/SoC Functional Design Verification》层级结构。同名缩写注意区分：CUDA 的 Unified Virtual Memory（见知识库_硬件架构.md 的 UVM 条目，与本条无关联）。

涉及论文标题：
- Democratizing and Accelerating Hardware Verification with Software-Native Optimization
