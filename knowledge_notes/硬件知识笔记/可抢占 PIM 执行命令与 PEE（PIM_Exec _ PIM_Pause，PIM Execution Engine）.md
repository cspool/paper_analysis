## 可抢占 PIM 执行命令与 PEE（PIM_Exec / PIM_Pause，PIM Execution Engine）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
PIM_Exec 让 PIM 单元自主连续执行 nPTL 个周期（执行长度由 DRAM 配置寄存器定义），无需逐列触发命令，命令总线占用降 nPTL/tCCD 倍；PIM_Pause 在列边界原子暂停 PIM 执行，让位于高优先级 CPU 请求。PEE（PIM Execution Engine）是 PIM 单元内的执行状态机：CSC（列状态计数器，推进列地址）、CR（命令寄存器，记录命令类型）、PC（PIM 计数器，记录进度）。仲裁器以 PC_inf=(clk−clk1)/tCCD 推断 PC，与 PIM 单元同步进度而无需额外信号。背景矛盾：命令长 ≥64（2rank×16banks）才能支撑全 bank 并行、≥128 才使命令总线占用 <40%，但短命令才能保证 CPU 低延迟——可抢占命令同时拿到"长命令的吞吐"与"短命令的响应"。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
COSM Fig.5 时序：clk1 发 PIM_Exec(起始列 64) → PEE 状态 1→2（CSC=64、CR=命令类型、PC=0）；每 tCCD 周期 PEE 自主 CSC++/PC++ 并执行 CR 命令；clk2=clk1+tCCD+1 CPU 读该 bank → 仲裁器发 PIM_Pause → PEE 完成当前列操作（t1=2tCCD，列 64-65 完成）、冻结 CSC、释放总线；CPU 访问后仲裁器重发 PIM_Exec，从 CSC 冻结列（66）恢复，剩余执行时间 t2=nPTL−2tCCD。三条语义保证：列原子性（仅在列边界暂停，进度确定可跟踪）、状态保存（中间数据与 PIM 缓冲/寄存器冻结）、状态恢复（重开原行 + 用冻结 PC 精确续点）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：PIM 单元内 PEE 硬件 + DRAM 配置寄存器定义 nPTL；存储型 PIM_Exec(St) 暂停后须等数据稳定（PRE 需 tCCD+tWR）而载入型只需 tRTP → 鼓励中间结果复用、减少 store。使用：nPTL=128 为 COSM 权衡点（过短命令总线饱和，过长 PIM_Pause 注入频繁）；与空闲感知调度（IWE）配合使用；论文实测较固定 32-cycle 命令 PIM 性能 2.02×（CPU 干扰 <5%）。局限：列级暂停不保证更细粒度；SIMD PIM 的扩展未探索。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices
