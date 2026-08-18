## SM Sub-partition（SMSP）与异构指令流水线（Tensor/FMA/XU/MIO 单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Ampere 及之后的 NVIDIA SM 典型划分为 4 个 SM Sub-partition（SMSP）与 1 个 Memory I/O（MIO）单元。每个 SMSP 含一个 Warp Scheduler（逐周期指令发射）、Register File 和一组专用 math pipeline：固定延迟的 FMA pipeline（FP32 加/乘/FMA）、Tensor pipeline（MMA 指令，如 HMMA/WGMMA，各精度 FP8/FP16/BF16）、变延迟的 XU pipeline（特殊函数，如 MUFU.EX2 对应 base-2 指数）。MIO 单元负责数据移动：L1 cache、Shared Memory（SMEM）与 LSU（执行 LDG/STS 等访存指令）。这套跨 Ampere~Blackwell 稳定的一致 SM 组织，是 PIPEWEAVE 实现跨架构泛化的微架构抽象基础——它把 kernel 建模为对这些异构 pipeline 的 demand 向量。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
PIPEWEAVE 的 Feature Analyzer 按 pipeline 分别建模（对应表 III 各 pipeline 的主要操作）：
```
# 每 task 的 math pipeline demand（表 III）：
Tensor: N_ops = α · tile_M · tile_N · tile_K   # α=2 GEMM, α=4 FlashAttention(两次MMA)
FMA:    对 FP32 add/mul/FMA 逐表达式与循环迭代空间计数
XU:     对 exp/log/rsqrt/sin/cos 等近似特殊函数计数
C_p = N_ops / Th_p        # Th_p 来自硬件规格 S（Tensor 512-4096 ops/clk/SM 等）
# MIO pipeline：按字节计 B_i（global/L2/SMEM 各层），C_mem = B / BW_mem
# 再按 task → SM → GPU 三级聚合，SM 级保留 Max SM 特征捕捉负载不均
```
关键设计取舍：不强行解析建模 pipeline 间的指令级并发（如 Tensor 与 FMA 并行）与架构专属机制（如 Hopper TMA），而是把这些 raw pipeline demand 作为独立特征交给 MLP 学习非线性交互——换取跨代通用性（各 pipeline 吞吐、带宽都是表中参数，换 GPU 只需换 S）。论文以 A100 上 FlashAttention-2 为例，绘制执行效率 vs 各 pipeline demand 的独立饱和曲线验证"每 pipeline 一个 roof"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
微架构上：MMA 由 Tensor Core 执行（warp 级，每个 warp 的 MMA 由 warp scheduler 按指令窗口发射），FMA 由 CUDA core 的 FP32 单元执行，XU（SFU）处理特殊函数，MIO 内的 LSU 执行访存（Hopper 起 TMA 可单线程发起 bulk 异步拷贝）。建模/工具层面：NVIDIA Nsight Compute（NCU）可测各 pipeline 的 op 计数与吞吐，PIPEWEAVE 用 NCU 在 A100/H100 上验证解析 op 计数（total op 误差 ≤0.5%，per-SM max ≤6.3%）。使用：对 LLM 推理 kernel，Tensor/FMA/XU 三 pipeline 已覆盖绝大多数计算需求（ALU 等因利用率低且难以解析计数被省略）；这套按 pipeline 拆分的建模方法原理上可扩展到 AMD GPU 等其它加速器。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction
