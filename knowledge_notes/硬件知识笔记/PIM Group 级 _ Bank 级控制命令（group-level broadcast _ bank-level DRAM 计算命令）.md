## PIM Group 级 / Bank 级控制命令（group-level broadcast / bank-level DRAM 计算命令）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
near-bank PIM 中 Host 内存控制器驱动 PIM core 计算的两级命令机制。group 级（group-level）指令：一条命令经内存总线广播到同一 PIM group（同 channel/rank）内的全部 core，各 core 用自己的本地数据执行同一操作，把命令流量从"每 core 一条"压到"每组一条"；bank 级（bank-level）指令：逐 core/逐 bank 独立下发，粒度细、灵活但命令流量大。PIM core 之间通常不能直接通信，跨 core 数据交换经 Host 内存总线中转。HBM-PIM [25]、GDDR6-AiM [26] 等后端均支持 SIMD 风格 group 广播命令（如 AttAcc 模拟器中的 PIM_MAC_AB 全 bank GEMV 命令）；DRAM 命令格式约束 group 级命令对组内所有 core 必须使用相同地址偏移，否则无法广播。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
DCC 的分级命令生成策略（§4.3 II）：优先 group 级——编译器把组内各 core 的张量数据 padding 到相同大小，使局部地址布局一致，内存控制器即可一条计算命令驱动整组（消除"全 PIM 并行时控制器命令远超常规 DRAM"的控制瓶颈）；当后端无 group 支持、或 core 需要访问不同地址偏移时退化为并行 bank 级命令，保持细粒度 bank 并行。执行例子（GEMV）：Host 控制器发一条 group 级 MAC 广播命令 → 组内 16 个 bank 的 GEMV/FPU 同时以相同行偏移读取各自本地数据并计算 → 结果写回各 bank；随后仅对需要归约的部分结果发 bank 级读命令取回。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：DRAM 计算命令作为标准 DRAM 命令序列的扩展（如 AttAcc 的 PIM_MAC_AB/SB/PB），由内存控制器调度器选择下发；group/bank 划分与地址映射绑定（HBM3-PIM 的 Ch→Pch→Ra→BG→Ba 映射）。使用：编译器后端决定命令粒度（broadcast vs per-bank），padding 对齐是启用 group 广播的代价（少量冗余数据换取命令带宽）；DCC 实测该分级策略在 HBM-PIM 与 AttAcc 上均能减少命令流量并提升 end-to-end 性能（数据见本库 PIM/DRAM-PIM/AttAcc 条目）。Web 证据（Hot Chips 33）：Aquabolt-XL 处理器以 SIMD 方式执行同一指令，与 group 广播语义一致。

涉及论文标题：
- DCC: Data-Centric Compilation of Machine Learning Kernels for Processing-In-Memory Architectures
