## Issue Queue（IQ，指令发射队列）

术语解释
指令发射队列（Issue Queue / instruction scheduler）是乱序（OoO）超标量处理器中保存已重命名但尚未发射的指令、跟踪其源操作数就绪状态并仲裁发射的硬件结构；它容纳"指令窗口"（window），是 wakeup-select 关键路径的载体，其容量与发射宽度直接决定可挖掘的指令级并行（ILP）。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
逻辑链：(1) 指令经 rename 后写入 IQ，等待两个源操作数全部就绪；(2) IQ 的唤醒逻辑（wakeup logic）负责把"生产者已发射/结果就绪"传播到等待的消费者；(3) select 逻辑从就绪指令中按优先级（通常最老优先）仲裁出可发射者，授予 grant；(4) grant 同时反馈回唤醒逻辑触发下一级依赖的就绪更新——构成 wakeup-select 闭环；(5) 为支撑更高 IPC，处理器需增大 IQ 尺寸与发射宽度，但这使唤醒矩阵（二维依赖数组）规模平方增长、wordline 扇出与 bitline 长度上升，延迟成为关键路径。本论文基线为 12-issue、200×200 matrix scheduler IQ（Table I），并指出资源翻倍本可带来 +36% IPC 却被 IQ 时序瓶颈限制。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
在硬件架构中 IQ 的运转流程（论文 Fig.1/2）：dispatch 时指令写入 IQ 某行，RMT 记录其目的逻辑寄存器对应的 producer 行号；wakeup matrix 的 cell(i,j)=1 表示指令 i 依赖 j（双源则两张矩阵）；当 producer 的 grant 信号拉高时驱动该列 wordline 向所有行广播，依赖 cell 经 bitline 读出使消费者 ready；两条源都 ready 后发 issue request 给 select；select 按资源可用性仲裁（随机队列中用 age matrix 选最老就绪者）后输出 grant，指令从 payload RAM 发射到功能单元，grant 再经 wordline 反馈触发下一代唤醒。整个 wakeup-select 必须单周期完成（否则依赖指令无法背靠背发射），故是处理器关键路径 [1][2]；整体占两个流水级：wakeup-select 与 payload RAM 读。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现分两类（论文引用 [3]）：CAM-based（tag 广播+每项比较器，如 AMD Bulldozer [4]）与 RAM-based/wakeup matrix（SRAM-like 二维依赖数组，如 IBM POWER8 [5]）；现代处理器多采用 random queue + age matrix 组织（无年龄顺序、乱序插入空位、用 age matrix 恢复最老优先）。Web 证据（Sohi 等 Complexity-Effective Superscalar Processors 延迟模型）：wakeup 延迟至少随窗口尺寸线性增长、发射宽度影响更大，广播长线是主要开销来源。使用上，IQ 容量（如本文 200 项、1.5× scaling 的 300 项、2× 的 400 项）与发射宽度（12-issue）是评估配置的核心参数；针对其延迟瓶颈的研究方案包括本论文 HWL、hierarchical scheduling window、wakeup matrix narrowing、matrix scheduler reloaded 等。

涉及论文标题：
- Hierarchical Wakeup Logic of the Issue Queue for High Scalability
