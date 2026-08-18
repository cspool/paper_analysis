## 重叠调度与 PIM_Barrier（Overlapped Scheduling，tile 级传输/执行重叠）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
把 PIM workload 切成无数据依赖的 tile（如矩阵乘子矩阵，reduction 由 CPU 做），跨 tile 并行发不同阶段的命令——收集 tile T_i 结果的同时执行 T_{i+1}，打破传统 CPU 软件串行控制的"输入传输→PIM 执行→结果收集"三阶段（Fig.7a：串行使内部/外部带宽交替空闲）。PIM_Barrier 命令插在重叠阶段边界，保证上一阶段全部操作完成后才进入下一阶段。与 GPU 双缓冲/软件流水同思想，但作用在内存通道的 PIM 命令流层面，且 CPU 仍是 reduction 的执行者。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
重叠执行伪代码（Fig.7b 时间轴：阶段内同时存在"收集 T_i 结果 + 执行 T_{i+1} + 写入 T_{i+2} 输入"）：
```
for phase in 全部 tile:
    并行发：PRWQ[PIM_RdBuf(T_{i-1} 收集) / PIM_WrBuf(T_{i+1} 输入)]   # 外部总线
         与  PEQ[PIM_Exec(T_i 计算)]                                  # 内部带宽
    PIM_Barrier    # 本 phase 全部操作完成后才进入下一 phase
```
Annotations：跨 tile 无依赖（reduction 归 CPU），故传输/执行可任意交错；PIM_Barrier 由编译器在 tile 边界自动插入。效果（COSM 消融）：Overlapped 使 CPU 性能较 Base +3.7%——但高行命中率的 PolyBench 内核因激进重叠引入额外行切换而受损；须与带宽解耦（+11.5%）和 IWE（保持 <5%）组合使用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：编译器插入 PIM_Barrier + 调度器维护 PEQ/PRWQ 双队列跨 tile 取命令（优先级：总线命令 > buffer 命令 > 执行命令）。使用：PIM 上任何"传输-计算-收集"型 workload（GEMV、attention 层）；与 GPU 侧软件流水（如 TMA+num_stages 多级流水）的区别是重叠发生在内存控制器命令级、由 IWE 窗口驱动。局限：重叠带来额外行切换开销，需按 row-hit rate 权衡。

涉及论文标题：
- COSM: A Cooperative Scheduling Framework for Concurrent PIM and CPU Execution on Mobile Devices
