## M²NDP（Memory-Mapped Near-Data Processing，内存映射近数据处理架构与模拟器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
M²NDP 是 POSTECH 提出的通用型 CXL 近数据处理架构（MICRO 2024，Low-Overhead General-Purpose Near-Data Processing in CXL Memory Expanders，arXiv:2404.19381），由两部分组成：M²func——用重定义的 CXL.mem 报文实现低开销主机-设备通信（单条 store 指令启动 kernel、返回结果，免去 CXL.io 环形缓冲的多轮往返）；M²μthread——基于扩展 RISC-V 向量指令的细粒度多线程执行模型，隐藏内存延迟并最大化并行度。相比 CPU/GPU 主机 + 被动 CXL 内存，M²NDP 最高 128× 加速、整体 14.5×，能耗最高降 87.9%。配套开源 cycle-level 模拟器（https://github.com/PSAL-POSTECH/M2NDP-public），由 Ramulator（内存时序）+ BookSim2（互连/NoC）组合而成。AXLE 以 M²NDP 为 SOTA 基线与实现底座：M²NDP 原生只支持 bulk synchronous（BS）卸载流，AXLE 在其上新增 RP 模型与异步背流机制。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
M²NDP 的卸载路径（BS 流）：主机对映射到特定地址范围（uncacheable 内存映射函数区）发单条 CXL.mem store（携带 kernel 信息）→ CXL 内存控制器上的自定义 packet filter 识别 kernel 启动命令 → CCM 调度器把任务切块分派给 µthreads 执行（每处理单元 16 µthreads 快速切换隐藏访存延迟）→ 同步 store 响应即 kernel 完成信号，主机靠 memory barrier 阻塞后续访存直到结果可取。AXLE 沿用其 µthread 执行模型与 packet filter 机制，仅替换控制/数据面协议。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：开源模拟器 M2NDP-public（cycle-level，Ramulator + BookSim2）作为研究平台；AXLE 论文用其做全部端到端评估（配置：主机 3GHz/32 处理单元/2 µthreads/DDR5-4800 16 通道，CCM 2GHz/16 处理单元/16 µthreads）。使用方式：评估 CXL 近数据处理卸载机制（kernel 启动开销、µthread 调度、带宽利用）；扩展方式是在其上加新协议模型（如 AXLE 的 RP/背流/DMA executor）。局限：BS 流导致主机同步 stall（AXLE 动机）。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
