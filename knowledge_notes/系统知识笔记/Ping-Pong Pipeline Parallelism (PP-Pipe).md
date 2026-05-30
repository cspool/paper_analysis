## Ping-Pong Pipeline Parallelism (PP-Pipe)

术语解释
Ping-Pong Pipeline Parallelism (PP-Pipe) 是 MegaScale-Infer 提出的 DEP 架构下的 micro-batch 调度算法，通过将输入 mini-batch 划分为多个 micro-batch 实现 AG 和 EG 的并发执行。

术语是什么？
PP-Pipe 的核心思想是将 DEP 中 AG 和 EG 的串行依赖关系转化为流水线并行：将 mini-batch 沿 batch 维度切分为 r1 个 micro-batch，每个 micro-batch 在 AG 处理后通过 A2E 发送给 EG，同时 AG 开始处理下一个 micro-batch。这样 AG 处理 micro-batch i+1 时，EG 正在处理 micro-batch i，形成 "ping-pong" 交替执行模式。PP-Pipe 不处理 Shared Expert（假设其不存在或与 Attention 串行），且每个 micro-batch 内的 A2E/E2A 通信与 EG 计算串行，仅在 micro-batch 之间有重叠。

从系统架构角度拆解：
PP-Pipe 在两 GPU group 间建立流水线：
- AG 端：处理 micro-batch i → A2E(micro-batch i) → 处理 micro-batch i+1 → A2E(micro-batch i+1) → ...
- EG 端：等待 A2E(micro-batch 0) → 计算 expert(micro-batch 0) → E2A(micro-batch 0) → 等待 A2E(micro-batch 1) → ...
关键约束是 r1=1 时退化为 Naive DEP（完全串行），r1 增大使流水线更深但增加 kernel launch 开销。PP-Pipe 的局限：(1) 不支持 shared expert 与 A2E 的并行；(2) micro-batch 内通信无法与 expert 计算重叠；(3) 粗粒度调度下通信仍是瓶颈（如表征为 DeepSeek-V2 S=4096 的 528.94ms 非重叠通信时间）。

术语一般如何实现？如何使用？
PP-Pipe 在 MegaScale-Infer 中实现，需要：(1) 将 mini-batch 按 batch 维度均分为 r1 个 micro-batch；(2) 为每个 micro-batch 建立 CUDA stream 或独立的 kernel launch 序列；(3) 使用 NCCL 的异步通信 API（ncclSend/ncclRecv + cudaStream）实现通信与计算的 overlap。PP-Pipe 是 FinDEP 的 baseline，FinDEP 在此基础上通过 r2 fine-grained 维度和 shared expert 感知调度实现了 1.02-1.61× 提速。

涉及论文标题：
- Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism

---
