## ROP（Raster Operations Pipeline，光栅化管线单元）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ROP（Raster Operations Pipeline，光栅化管线/光栅操作单元）是 GPU 中每个 Memory Partition Unit（MPU，内存分区单元：L2 cache slice + DRAM controller + ROP）内的固定功能单元，原始职责是图形管线最后阶段——像素数据写回帧缓冲前做颜色混合（blending）、抗锯齿（antialiasing）、深度测试（depth test）等。现代 GPU 为把 ROP 的近距离访存能力扩展到通用计算，扩展其可编程性以支持原子操作：ROP 内 ALU 可对 L2 缓存数据执行 read-modify-write 原子运算（如 SASS RED.E.ADD），成为 GPU 原子指令的实际执行位置。关键物理特性：每个 ROP 只能访问同 MPU 的 L2 cache slice；ROP 靠近内存/L2 而远离 SM。RoCC 论文经反向工程（SASS 微基准 + SR_CLOCKLO 计时）揭示 ROP 微架构：原子操作对 SM 异步执行（指令发出即返回）、V100 ROP 数据通路总延迟 ≈28 cycle（A100 22 cycle）、单流水级 ≈3 cycle（A100 1 cycle）、一个 ROP 含 4 个可并行 32-bit 标量执行单元（与 PTX 最大 atomic 宽度 4×FP32 一致）。AI 专用 GPU（H100/H200/B200）保留 ROP 供原子操作使用——像素吞吐从 A100 225.6 GPixel/s 降到 H200 47.16 GPixel/s，而 atomic 带宽从 A100 422.51 GOp/s 升到 B200 729.783 GOp/s，说明 ROP 在 AI GPU 上持续存在且原子能力随内存带宽增强。no note evidence（omnisearch 未覆盖 vault 目录）；web 补充：NVIDIA 白皮书（A100/Hopper 架构）与 TechPowerUp 规格库。

从硬件架构角度拆解术语，比如术语如何在硬件架构中发挥作用，给出术语在硬件架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
ROP 硬件运转流程（一次原子操作，RoCC 论文 Figure 4）：① SM 发原子操作请求包 → ② MPU 仲裁器识别为原子操作（区别于普通 L2 请求）→ ③ 请求进入 atomic command buffer → ④ 每 cycle 命令生成器取一个原子命令 → ⑤ 读 ROP 内部 cache 取操作数（miss 则读同 MPU 的 L2 slice）→ ⑥ ROP ALU 计算结果写回 ROP cache 并送 result sequencer 保证原子序列 → ⑦ 结果发往 MPU results buffer、L2 slice 与请求 SM。一次 ROP 原子流水总延迟 V100 ≈28 cycle、每级 3 cycle；4 个执行单元并行（4-way 32-bit 标量）。RoCC 在此基础扩展：在 ROP 内新增 doorbell manager、collective decoder + primitive decoder、collective command buffer 等小逻辑，把 CC 操作译码成 ROP μOp 序列（ReadDoorbell/Write/DepBarrier/Add/RingDoorbell）并跨 GPU 接力，使 ROP 从"原子运算执行单元"升级为"近内存的集体通信引擎"——这是本术语在本文中的核心作用。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：ROP 为固定硬件单元，位于 MPU 内，由 ISA 原子指令（PTX RED / SASS RED.E.ADD）触发；NVIDIA/AMD 的原子指令数据通路即面向 ROP 设计。使用方式：① 图形：像素 blend/depth/写帧缓冲；② 通用计算：atomicAdd/atomicCAS 等（CUDA/PTX 直接使用，ROP 隐藏执行）；③ RoCC 新用法：新增 ROP_AR/ROP_AG/ROP_A2A 指令 + rocc_allreduce 等 intrinsic，ROP 异步执行 CC 不占 SM，实现 GEMM 与 CC 细粒度重叠。论文实测 LLM 训练中 ROP 高度空闲（Qwen2-7B 训练 11476 个 kernel 仅 141 个发过 atomic 指令、GEMM 内 atomic 占比 ≈0.0183%），是 CC 复用的容量依据。

涉及论文标题：
- RoCC Harnessing Raster Operations Pipeline for Efficient Tensor Collective Communication
