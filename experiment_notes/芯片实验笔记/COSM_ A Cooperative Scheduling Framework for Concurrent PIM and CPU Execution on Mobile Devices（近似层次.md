## COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices（近似层次匹配：LPDDR5 命令接口与 PIM 单元片上缓冲的 DRAM 芯片级扩展，核心调度硬件见 实验_硬件架构.md）

- 属于芯片设计的实现是什么？实验比较什么？
  - 近似匹配（论文主体是内存控制器调度硬件，此处覆盖其 DRAM 芯片结构改动）：实现 = (1) 扩展 LPDDR5 命令集：PIM_Exec（可抢占执行，bank 级、自主逐列执行 nPTL）、PIM_Pause（列边界原子暂停/恢复）、PIM_RdBuf/PIM_WrBuf（外部总线）与 PIM_LdBuf/PIM_StBuf（内部带宽）两阶段传输命令、PIM_Barrier，并配套 bank 级/通道级时序约束（表 1）；(2) PIM 单元内新增 PEE（PIM Execution Engine：CSC/CR/PC 寄存器状态机，按 tCCD 自主推进列地址并支持冻结/恢复）；(3) 每 DRAM bank 旁新增 SRAM buffer 段（1kB，容量 = 单条 nPTL 长度 PIM_LdBuf/StBuf 最大传输量）。实验比较：面积——Synopsys Design Compiler + TSMC 90nm @2.4GHz 综合控制器模块 0.069 mm²（PIM scheduler 0.014 + IWE 0.0085 + Command Arbiter 0.0054 mm²），占 Snapdragon 8 级 LPDDR5 控制器（约 0.93 mm²，据公开 die 照片测得）的 7.4%，并按公开缩放数据折算 5nm 保守估计；能量——每 token 能耗较 AsyncDIMM-Bank/Chopim 降 1.34×/1.61×（细粒度 CPU-PIM 交错虽增加行切换开销，但消除长 active-idle 周期）。
- 模拟器名，模拟器链接（web search），或论文修改的模拟器。
  - 面积评估无性能模拟器（Synopsys Design Compiler 综合 + 5nm 缩放）；性能模拟器 = 修改版 Ramulator2（LPDDR5-PIM 模块，src/dram/impl/LPDDR5-PIM.cpp，基座 https://github.com/CMU-SAFARI/ramulator2）；DRAM 功耗用 DRAMPower。
- 模拟器模拟什么的性能，修改了什么。
  - 模拟 LPDDR5-PIM 芯片的新命令行为：PIM_Exec 按 tCCD 自主逐列执行与 PIM_Pause 的列边界冻结/恢复、PIM_RdBuf/WrBuf（外部总线）与 PIM_LdBuf/StBuf（内部带宽）的两阶段解耦传输、bank buffer 就绪与 tBL 时序，并统一 PIM_LdBuf/StBuf 与 PIM_Exec 的执行模型（同为 nPTL 长度、可抢占）。修改：LPDDR5 模块新增上述命令与表 1 时序约束、PEE 寄存器状态机与 bank buffer 模型；DRAMPower 核算 PIM 单元计算 + bank 访问 + CPU-mediated 传输的能量。
- 开源情况。基于开源文档和论文，使用例子解释模拟器如何使用？作用是什么？
  - 开源情况：Zenodo artifact（https://doi.org/10.5281/zenodo.19660293）含修改版 Ramulator2（含 LPDDR5-PIM 模型）、trace 与脚本。使用例子：`./build.sh` 编译后 `./run_script.sh figure8`，在 LPDDR5-6400（2ch×2rank、16 banks/rank）配置下回放 LLM 与移动应用 trace，模拟器逐 cycle 推演 PIM 执行/暂停/两阶段传输与 CPU 访问，输出吞吐与能量；面积侧用 DC 综合控制器模块并折算 5nm，评估把低干扰 PIM 命令接口与 bank buffer 引入 DRAM 芯片的 7.4% 面积代价。
