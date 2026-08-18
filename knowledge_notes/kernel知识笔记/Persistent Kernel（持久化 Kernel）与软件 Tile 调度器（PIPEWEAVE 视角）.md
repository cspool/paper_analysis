## Persistent Kernel（持久化 Kernel）与软件 Tile 调度器（PIPEWEAVE 视角）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Persistent Kernel 是让少数长生命周期 CTA 在整个 kernel 执行期间驻留 SM、反复从全局 work queue 取小粒度工作单元的 GPU kernel 编程范式（如 CUTLASS Ping-Pong GEMM、FlashAttention-3、Stream-K）。区别于传统"一 CTA 一任务、硬件调度器动态派发"的模型，persistent kernel 中 CTA 只 launch 一次，硬件调度器（GigaThread Engine）的角色退居其次，任务分配（哪个 tile 给哪个 SM 的哪个 worker）由软件 tile scheduler 决定。PIPEWEAVE 明确指出：此时"基本调度单元 task"不再是 CTA 而是 resident CTA 每次取回的 tile，且调度语义必须显式建模软件调度器（如 FA3 的 MinHeap 调度逻辑）才能准确预测 per-SM 负载分布。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 FlashAttention-3（Hopper，persistent）为例的软件 tile 调度模拟：
```
# PIPEWEAVE Scheduling Simulator 复刻 FA3 的 MinHeap 调度（约 40 行）
# N_SM 个持久 CTA 各占一个 SM；每 CTA 从共享 work queue 取 tile
queue = MinHeap(tiles, key=按序列/头优先顺序)   # 或 work-stealing
while queue 非空:
    t = queue.pop_min()
    sm = 负载最小的空闲 CTA      # 软件决定去向，而非硬件 RR
    sm.task_list.append(t)
# 输出 task 分布 {T_1..T_N_SM}，供 Feature Analyzer 求每 SM 的 pipeline demand
```
对硬件调度范式（FA2、RMSNorm 等），则模拟 RR 策略：先给每个 SM 至少一个 CTA，资源够再第二轮，直到 SM 饱和，之后新 CTA 在旧 CTA 完成时补位。论文验证（Nsight Compute）：FA3（persistent、确定性调度可显式模拟）的 per-SM op 计数误差仅 0.45%，而 FA2（动态硬件调度）为 6.34%——persistent kernel 的可确定性正是其可建模性来源。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现上：CUTLASS/Triton 提供 persistent 编程支持（Triton `@triton.jit` 内 while 循环 + 原子指针取 tile、CUTLASS 的 tile scheduler 模板），开源库 FlashInfer 的 FA3、cuBLAS 的 Hopper persistent GEMM 均属此类。PIPEWEAVE 中调度范式由表驱动（Table V：GEMM/Attention 等 HW/SW 双范式、RMSNorm/SiLU&Mul/Fused MoE 为 HW）。使用意义：persistent kernel 消除反复 launch 开销、减少 tail 效应，但其性能强依赖软件调度质量——这正是 PIPEWEAVE"beyond simulation"用 P80 性能上限模型诊断 Fused MoE（SGLang Triton、persistent）在 A40 上 921 个 underperforming points 并 autotune（BLOCK_SIZE/num_stages/num_warps）提速 1.61× 的前提。

涉及论文标题：
- PIPEWEAVE: Synergizing Analytical and Learning Models for Unified GPU Performance Prediction
