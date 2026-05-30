## Expert Data Parallelism (EDP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Data Parallelism (EDP) 是 Expert Parallelism 与 Data Parallelism 结合产生的特殊并行模式。当 DP degree > EP degree 时，系统必须跨多个 EP group 使用 DP。EDP group 定义为共享同一 EP rank 的多个设备（不同 EP group 中同一位置的 GPU）。EDP group 内的设备持有相同 expert 的 replica（参数完全一致），各自处理来自不同 EP group 的不同 token。这些 replica 之间通过标准 DP 机制同步参数和梯度（all-reduce）。EDP 是 FineEP 实现 token scheduling 的关键基础——因为同一 expert 在多个 GPU 上有 replica，token 可以选择任一 replica 计算，从而创造了"调度空间"。

从算法pipeline角度拆解术语：
以 DP=8, EP=4 为例：
- 8 GPU 分为 2 个 EP group（各 4 GPU）。
- EP group 0: GPU{0,1,2,3}，EP group 1: GPU{4,5,6,7}。
- Expert 0 的 replica 在 GPU 0 和 GPU 4（EDP group of expert 0 = {0,4}）。
- 各 GPU 0 和 GPU 4 持有 expert 0 的相同参数，但处理不同 EP group 的不同 tokens。
- 传统 EP：token 只能在 assigned EP group 内的 GPU 0（或 GPU 4）计算 expert 0。
- FineEP：合并 EP groups 后，token 可在 EDP group {0,4} 中任一 GPU 计算 expert 0。

术语一般如何实现？如何使用？
- 参数同步：EDP group 内通过 DP all-reduce 同步 expert replica 的 gradients。
- 在 Megatron-LM 中，EDP 由 DP 和 EP 的配置自动形成，无需显式设置。
- FineEP 利用 EDP 创建 token 调度空间：每个 expert 在 |G_FineEP|/EP_degree 个 GPU 上有 replica。
- 约束：所有 replica 必须具有相同的 local expert index（确保 DP synchronization 一致性，避免 deadlock）。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

---
