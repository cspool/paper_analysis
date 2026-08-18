## µthread（微线程 / Fine-Grained Multithreading）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
µthread 是 M²NDP/AXLE CCM 处理单元内的细粒度多线程执行模型：每个处理单元集成多个（16 个）µthreads，以极细粒度（逐周期/逐指令级）快速切换执行，保证持续有指令可取，从而隐藏内存访问延迟、最大化并行度——本质是 barrel 式细粒度多线程。M²NDP 以扩展 RISC-V 向量指令实现 µthread。主机侧为模拟通用多线程核，配置 2 µthreads/处理单元（模拟超线程）。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
执行流程：CCM 调度器把一个卸载 kernel 划分为固定大小输入向量任务 → 每个 µthread 领取一个输入向量块执行 RISC-V 指令 → 某 µthread 发生访存 miss 时立即切换其他 µthread 的指令流（隐藏 DDR5 访问延迟）→ 各 µthread 把结果写入设备内存后由调度器按策略（RR/FIFO）继续分派。设计要点：调度策略需平衡负载并最大化 CXL 内存带宽利用率；AXLE 的 OoO 流式支持 µthread 乱序完成（结果顺序 {2,0,1} 不等同物理 buffer 顺序 {0,1,2}），由 metadata 记录 payload slot ID 解耦。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：硬件线程槽 + 切换逻辑（寄存器文件分区、无缓存共享或小缓存）；M²NDP 以 RISC-V 向量扩展为基础实现。使用方式：近内存计算设备（CCM/PIM）隐藏访存延迟、提升带宽利用率；主机侧以 2 µthread/处理单元模拟超线程。评估中每处理单元 16 µthreads 使 CCM 在 2GHz/16 单元配置下获得高带宽利用。

涉及论文标题：
- AXLE: Coordinated Offloading with Asynchronous Back-Streaming in Computational Memory Systems
