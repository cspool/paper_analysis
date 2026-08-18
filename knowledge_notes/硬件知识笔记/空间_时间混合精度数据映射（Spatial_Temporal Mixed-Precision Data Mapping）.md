## 空间/时间混合精度数据映射（Spatial/Temporal Mixed-Precision Data Mapping）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 空间/时间混合精度数据映射是 SingularBit Tensor Core 执行 rank 级混合精度矩阵乘时采用的两种并行/调度策略，解决"不同 rank 位宽不同 → 位串行延迟不同 → 负载不均衡"的问题（Fig.9）。空间混合精度（用于 U 计算）：精度沿 rank 维变化，不同 rank 分给不同 core，每个 core 以该 rank 对应的位宽整 rank 处理，多个 core 并行执行不同精度；因为位串行延迟 ∝ 位宽，core scheduler 把下一 rank 分配给最早空闲的 core，实现动态负载均衡。时间混合精度（用于 V^T 计算）：精度沿 rank（归约）维变化，每个 core 顺序处理输出块、沿 rank 精度转变处顺序累加部分和；由于所有 core 沿归约维经历相同精度序列，它们同步结束，无需动态调度。
- 从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
- 运转流程（LLaMA 一个线性层，$W^T=U\cdot S\cdot V^T$，4 个精度区 4/3/2/1-bit）：U 侧（空间）——rank 0..r1 为 4-bit、r1..r2 为 3-bit、r2..r3 为 2-bit、r3..R 为 1-bit，共 4 个 core 各取一个精度区并行计算 $X\hat{U}_{[区]}$（4-bit core 延迟 ≈ 4× 1-bit core）；FSM/core scheduler 按完成顺序把下一 rank 分给空闲 core（如 1-bit core 先完成、立即接新 rank），各 core 忙闲由运行期状态驱动；U 结果按 rank 汇总后与 S 相乘。V^T 侧（时间）——同一 core 内沿归约维（rank 序）处理输出块：先算 4-bit 区的部分和、再 3-bit、2-bit、1-bit 区，inter-group 累加器把各精度区部分和按 rank 序累加；所有 core 走相同精度序列、同步结束。两策略的取舍：空间并行度高但需动态调度器，时间同步简单但单核串行位宽延迟更长。
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：由 SingularBit-W Compute FSM（提供当前 bit 位置、驱动 activation/weight loader）与 core scheduler（数据移动、buffer 操作、rank→core 分配）协同实现，无需软件重配置；与传统多精度核（每精度独立数据通路或运行期重配置 [49][50]）相比免去重配置开销。论文在 SVD 分解语义上的映射依据：U 是行空间基（各 rank 独立、可跨核并行），V^T 是输出投影（沿归约维累加、核内顺序）。论文数据：该映射使 tensor core 直接执行 rank-aware 混合精度而无精度切换惩罚，支撑 5.27×/5.32× reasoning speedup 与 4.64×/4.78× 能耗节省。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference
