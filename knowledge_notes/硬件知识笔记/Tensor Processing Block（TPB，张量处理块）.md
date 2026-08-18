## Tensor Processing Block（TPB，张量处理块）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- TPB 是 M100 NPU 的基本张量计算单元（M100 NPU = 1 个 CCB + 14 个 TPB cluster × 4 TPB = 56 TPB）。它负责 tensor 计算与变换，由若干专用功能单元组成：2MB High-Bandwidth Shared Memory（HBSM，数据存储与通信枢纽）、Tensor Computing Unit（TCU，卷积/矩阵乘）、Configurable Vector Unit（CVU，pooling/softmax/layernorm 等向量任务）、Data Transformation DMA Unit（DTDU，搬运/转置）、Gather-Scatter DMA Unit（GSDU，不规则搬运）、CPU Starter Unit（CSU，触发 cluster CPU 协助）、Synchronization Unit（SU，同步计数）、Tensor Walker Unit（TWU，地址生成）。功能单元经 HBSM 预定义地址区间 + 同步计数交换数据，无需专用 datapath。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程：固件经 ICB 把 TPB 指令（数千 bit，含 tensor shape 与通信需求元数据）下发到 cluster 指令队列 → 指令在数据就绪且同步条件满足时派发到对应功能单元（TCU/CVU/DTDU 等）→ TWU 生成嵌套地址序列流式读写 HBSM → SU 维护生产者-消费者计数协调各单元流水 → 输出写回 HBSM 或经 Mesh/DRB 送其他 TPB。TPB 指令触发 CPU 任务时由 CSU 保存参数、触发中断，cluster RISC-V CPU 经 VCIX 接口访问 TPB 数据与设备后标记指令完成。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：真实芯片硬件模块（M100 SoC，TSMC N5A），功能单元专用化设计，共享 HBSM 与 SU 实现单元间数据流。使用：由编译器/固件生成的 TPB 指令驱动，软件负责同步与调度（orchestrated dataflow）。未开源（论文未提供 RTL）。

涉及论文标题：
- M100: An Orchestrated Dataflow Architecture Powering General AI Computing
