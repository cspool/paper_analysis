## 带宽解耦 CPU-mediated 数据传输（PIM_RdBuf/PIM_WrBuf + PIM_LdBuf/PIM_StBuf）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CPU-mediated transfer = 主机 CPU 为 PIM 任务搬运输入/收集结果的必要传输（写权重/激活进 bank、读结果出），与 CPU 访问共享外部总线。问题（COSM §3.3）：传输呈突发高行缓冲局部性，在 FR-FCFS 下独占控制器、饿死需行冲突的 CPU 请求——CPU 任务降速 >80%；传输占解码注意力推理时间 50–60%（KV 长度 64 时 >60%、4k 时约 50%），折合整体 CPU 性能 -40%。带宽解耦 = 把一次传输拆成两段命令：外部总线段 PIM_WrBuf（控制器→bank buffer）/PIM_RdBuf（buffer→控制器）+ 内部带宽段 PIM_StBuf（buffer→bank）/PIM_LdBuf（bank→buffer），配每 bank 1kB SRAM buffer；两段可分别利用 bus/bank 空闲窗口，不再要求内/外带宽同时空闲。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一次写传输的调度流程（写序列 = PIM_WrBuf & PIM_StBuf，读序列 = PIM_LdBuf & PIM_RdBuf）：
```
CPU 发起 PIM 写 → 编译器分解为命令对 (PIM_WrBuf, PIM_StBuf) 入 PRWQ
PIM scheduler 每周期按优先级取命令：
    1) PIM_RdBuf/PIM_WrBuf   # 独占外部总线（瓶颈），优先利用 window_bus 前的 bus 空闲
    2) PIM_LdBuf/PIM_StBuf   # 清空/填充 buffer，防阻塞后续总线命令
    3) PIM_Exec              # 匹配当前 bank 空闲窗口
```
Annotations：内存序——同一传输派生的命令对必须按程序序执行（调度器对同 bank PRWQ 保到达序），其他命令（PIM 执行/CPU 访问）不与传输流共享专用 buffer 区域、可任意交错。时序：RdBuf/WrBuf 与同通道 Read/Write 间 tBL；LdBuf/StBuf 与 PIM_Exec(Ld/St) 镜像同一执行模型（nPTL、可抢占）→ 统一调度逻辑。buffer 容量只需覆盖单条 nPTL 长度命令的传输量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译器辅助翻译（读写序列自动分解为命令对，无需改用户程序与 OS）+ 内存控制器 PRWQ 队列。效果（COSM 消融）：Overlapped+Decoupled 使 CPU 性能较 Base +11.5%，叠加 IWE（All 配置）后全 workload CPU 降速 <5%。使用：PIM 系统中所有主机侧数据搬运；把"外部总线"与"内部带宽"解耦是消除传输干扰的关键（对比：baseline 全部用标准读/写命令，无专门调度）。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices
