## Virtual Queue Scheduling for MoE (MoE虚拟队列调度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Virtual Queue Scheduling 是 MoE-Prism 的 Quality-Constrained Throughput Scheduler 使用的核心调度数据结构：维护 M 个虚拟队列（M = 可能的 k_active 值数量，如 9,10,...,32），每个请求 R_i 不是放入单一队列，而是加入所有满足其质量约束的虚拟队列（所有 m ≥ k_min_i）。调度器并行评估所有 M 个虚拟队列的效用 U_m = Σ tokens(R_i) / C(|Q_m|, m)，选择最高效用的队列发射。发射时，批次内请求被从所有虚拟队列中原子移除。这使调度器能够同时考虑 M 种不同的 (k_active, batch_composition) 组合，打破传统的"先组批再定配置"或"先定配置再组批"的循环依赖。

从系统架构角度拆解术语：
```
Virtual Queues state (example with M=4, k_active ∈ {1,2,3,4}):

Requests pending: R_A(k_min=1,tokens=100), R_B(k_min=2,tokens=50), R_C(k_min=3,tokens=200)

Q_1 = [R_A]                          # only R_A qualifies
Q_2 = [R_A, R_B]                     # R_A+R_B qualify
Q_3 = [R_A, R_B, R_C]                # all qualify
Q_4 = [R_A, R_B, R_C]                # all qualify

Utilities (假设C(n,k) = (n*k)*单位延迟):
U_1 = 100/C(1,1) = simple throughput
U_2 = 150/C(2,2) = batch of 2, k=2
U_3 = 350/C(3,3) = batch of 3, k=3  ← likely highest utility
U_4 = 350/C(3,4) = batch of 3, k=4  ← more quality, same batch, higher cost

选择 Q_3 → 所有3个请求一起发射(k_active=3) → 原子移除R_A,R_B,R_C from all Q_1..Q_4
```
关键性质：(1) 非独立调度——同一个请求可出现在多个候选批次中；(2) 原子性——发射后所有虚拟队列状态一致更新；(3) 等价于在每个调度周期枚举指数级组合空间的一个高效近似。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoE-Prism 中 M 与 refactored model 的 sub-expert 配置空间大小相同（如 Deepseek 的 k 范围 8→24，M=17）。性能模型 C(k_active) 通过一次性 offline benchmark 构建为 lookup table。
- 对比传统单队列调度：FIFO 对所有请求用同个队列+批次内最高 k_min → 粗粒度 QoS；FullBatch 等满 B_max → 无 QoS 概念。虚拟队列方法提供 fine-grained per-configuration utility comparison。
- 概念上类似 networking 中的 Virtual Output Queuing (VOQ) 和 multi-queue weighted fair queuing，但用于 MoE LLM serving 的异构质量约束场景。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

---
