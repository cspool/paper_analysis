# MPK tGraph Mega-Kernel化

- 导出时间：2026-06-04 10:56 CST
- 来源：当前 Codex session 可用对话上下文
- 保存路径：human_notes/MPK-tGraph-Mega-Kernel化.md
- 范围：仅用户输入与 Codex 最终输出

## 对话记录

### 001 User
# Context from my IDE setup:

## Active file: draft/review_draft.md

## Active selection of the file:
| Mega-Kernel化 | MPK tGraph* | SM级tGraph替代kernel级DAG；per-task-pair event (352 bytes/task metadata, 全部GPU device memory连续数组)；**128 Workers + 4 Scheduler-SMs物理分区；Cross-task pipelining** (TMA prefetch next weight tile during current compute) | SM级表示暴露kernel barrier遮蔽的细粒度并行；Hybrid JIT+AOT：attention JIT dynamic load balance + MatMul AOT消除dispatch overhead | Q2.3, vault: MPK (59.2) |
## Open tabs:
- review_draft.md: draft/review_draft.md
- L6_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L6_horizon_summary.md
- L5_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L5_horizon_summary.md
- L4_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L4_horizon_summary.md
- L3_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L3_horizon_summary.md

## My request for Codex:
教我:'| Mega-Kernel化 | MPK tGraph* | SM级tGraph替代kernel级DAG；per-task-pair event (352 bytes/task metadata, 全部GPU device memory连续数组)；**128 Workers + 4 Scheduler-SMs物理分区；Cross-task pipelining** (TMA prefetch next weight tile during current compute) | SM级表示暴露kernel barrier遮蔽的细粒度并行；Hybrid JIT+AOT：attention JIT dynamic load balance + MatMul AOT消除dispatch overhead | Q2.3, vault: MPK (59.2) |'

### 002 Codex Final
可以，把这行当成一张“方法卡片”来读：

> **MPK tGraph**：把原来“一个算子一个 kernel”的执行方式，编译成一个长期驻留的 **mega-kernel**；在这个 mega-kernel 内部，用 **SM 级 task/event 图**调度整段模型推理。

**语义切分**

| 段 | 关键词 | 这段在说什么 |
|---|---|---|
| S1 | Mega-Kernel化, MPK, tGraph | 方法身份：全模型融合成一个 persistent mega-kernel |
| S2 | SM级 tGraph, kernel级 DAG, per-task-pair event | 表示方式：从 kernel 粒度下沉到 SM/task 粒度 |
| S3 | 352 bytes/task metadata, GPU device memory连续数组 | 数据结构：task/event 元数据紧凑存在 GPU 上 |
| S4 | 128 Workers + 4 Scheduler-SMs | 运行时：H100 上 SM 物理分区 |
| S5 | Cross-task pipelining, TMA prefetch | 流水线：当前 task 计算时预取下一个 task 的数据 |
| S6 | Hybrid JIT+AOT | 调度策略：attention 动态分配，MatMul 静态预分配 |

**一句话版本**

传统 vLLM/SGLang 类系统是：

```text
Q_proj kernel -> barrier -> Attention kernel -> barrier -> MLP kernel -> barrier
```

MPK 变成：

```text
single mega-kernel:
  scheduler SMs 管 event
  worker SMs 执行 task
  task 之间用 event 精确同步
  当前 task compute 时，TMA 预取下一个 task 的 weight tile
```

**核心概念**

`tGraph` 不是普通 CUDA Graph。CUDA Graph 主要记录“kernel launch 的 DAG”；MPK 的 `tGraph` 记录的是“SM 级 task 和 event 的 DAG”。一个 task 是某个算子切出来的一小块工作，通常在一个 SM 上执行；event 是 task 间同步点。

最关键的依赖规则是：

$$
event(t_i, t_j)\ \text{is inserted iff}\ OutputRegion(t_i)\cap InputRegion(t_j)\neq \emptyset
$$

Annotations:
- `t_i` 是 producer task，比如某个 MatMul tile。
- `t_j` 是 consumer task，比如对应 AllReduce tile。
- 只有输出区域和输入区域重叠，才需要 event。
- 所以 AllReduce 不必等整个 MatMul kernel 完成；某个 MatMul task 完成后，对应 AllReduce task 就可以开始。

**128 Workers + 4 Scheduler-SMs**

这是 H100 的具体配置：H100 有 132 个 SM，MPK 把其中 128 个当 worker SM，4 个当 scheduler SM。worker 负责真正计算，scheduler 负责 event queue 和 task dispatch。

```text
H100: 132 SMs
  128 Worker SMs:
    dequeue task -> execute -> notify event
  4 Scheduler SMs:
    poll event -> dispatch ready tasks
```

**352 bytes/task metadata**

这不是 event 大小，而是每个 task 描述符大小。它记录 task 的输入/输出 tensor 指针、配置参数、dependent event、trigger event 等。MPK 把 task/event 元数据放在 GPU device memory 的连续数组里，worker 会把即将执行的 task description 预取到 shared memory，减少队列访问开销。

**Cross-task pipelining**

传统 kernel 边界会制造 bubble：一个 kernel 完了，下一个 kernel 才能开始预取数据。MPK 把 task 拆成 preload 和 compute 两段：

```text
Task T1:
  preload W1 by TMA
  compute W1 by Tensor Core

Task T2:
  preload W2 by TMA can overlap with T1 compute
```

Annotations:
- TMA 负责异步把 HBM 中的 tile 搬到 shared memory。
- Tensor Core 正在算当前 tile 时，TMA 可以提前搬下一个 task 的 weight tile。
- 前提是 T1 已经发完自己的数据搬运指令，并且 shared memory page 够用。
- 这就是 row 里说的 “TMA prefetch next weight tile during current compute”。

**Hybrid JIT+AOT**

MPK 不把所有 task 都动态调度，也不把所有 task 都静态绑定，而是混合：

| 模式 | 用在哪里 | 为什么 |
|---|---|---|
| JIT | Attention | sequence length 不同，执行时间不均衡，需要动态 load balance |
| AOT | MatMul/MLP | 执行时间更确定，提前分配可减少 dispatch overhead |

所以 row 里的 “attention JIT dynamic load balance + MatMul AOT 消除 dispatch overhead” 可以理解为：

```text
Attention: 谁先算完，谁继续拿后续 task，减少长短序列负载不均
MatMul: 提前放进 worker queue，event 一激活就执行，少走 scheduler
```

**怎么记**

MPK 的主线是：

```text
kernel级 DAG
  -> SM级 tGraph
  -> task-pair event
  -> in-kernel worker/scheduler runtime
  -> TMA cross-task pipeline
  -> JIT+AOT 混合发射
```

这行的真正含义是：MPK 把“算子级调度问题”降到“SM 级 task 调度问题”，于是原来被 kernel barrier 遮住的并行性、预取机会、compute/communication overlap 都露出来了。

**笔记证据**

- [2.2-Kernel-Fusion.md](</data3/paper_analysis/paper_secs/secs_multimodal_kernel/Mirage Persistent Kernel A Compiler and Runtime for Mega-Kernelizing Tensor Programs/span-idpage-2-0span2.2-Kernel-Fusion.md>): tGraph、task/event、per-task dependency。
- [5-In-Kernel-Parallel-Runtime.md](</data3/paper_analysis/paper_secs/secs_multimodal_kernel/Mirage Persistent Kernel A Compiler and Runtime for Mega-Kernelizing Tensor Programs/5-In-Kernel-Parallel-Runtime.md>): worker/scheduler runtime。
- [5.2-Hybrid-Task-Launch.md](</data3/paper_analysis/paper_secs/secs_multimodal_kernel/Mirage Persistent Kernel A Compiler and Runtime for Mega-Kernelizing Tensor Programs/span-idpage-7-0span5.2-Hybrid-Task-Launch.md>): JIT/AOT tradeoff。
- [MPK kernel实验笔记](</data3/paper_analysis/experiment_notes/kernel实验笔记/Mirage Persistent Kernel_ A Compiler and Runtime for Mega-Kernelizing Tensor Programs.md>): 128 workers、4 schedulers、352 bytes/task、TMA pipeline。

### 003 User
# Context from my IDE setup:

## Active file: draft/review_draft.md

## Active selection of the file:
**确定SM分区（GreenContext CUDA 12.4+，重分区~4.1μs**
## Open tabs:
- review_draft.md: draft/review_draft.md
- L6_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L6_horizon_summary.md
- L5_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L5_horizon_summary.md
- L4_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L4_horizon_summary.md
- L3_horizon_summary.md: learning_outputs/20260529_185938__MoEDiT多模态Video__单GPUNPU加速器__多算子微算子并发__硬件体系结构/L3_horizon_summary.md

## My request for Codex:
[$export-conversation-notes](/data3/paper_analysis/.codex/skills/export-conversation-notes/SKILL.md) 
