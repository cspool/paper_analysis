## OptSche (Optimal MoE Task Scheduling Algorithm)

术语是什么？
OptSche 是 ScheMoE (EuroSys '24) 提出的 MoE 训练最优任务调度算法。其核心目标是在给定输入 tensor 分区度 r 的条件下，找到 MoE layer 中 compute tasks（compression、decompression、expert computation）和 communication tasks（A2A dispatch、A2A combine）的最优执行顺序，使得通信时间被计算时间最大化隐藏。

ScheMoE 将 MoE layer 的 forward propagation 形式化为 7 类任务：C₁（第一次 compress）、A₁（第一次 A2A dispatch）、D₁（第一次 decompress）、E（expert fflayer compute）、C₂（第二次 compress，combine 前）、A₂（第二次 A2A combine）、D₂（第二次 decompress，combine 后）。当输入 token tensor 被均匀划分为 r 份时，总任务数为 7r。OptSche 在数据依赖约束下数学证明了 r=2 时 CompTask 的最优执行顺序：

$$(\mathsf{C}_1^1\mathsf{C}_1^2)(\mathsf{D}_1^1\mathsf{E}^1\mathsf{C}_2^1)(\mathsf{D}_1^2\mathsf{E}^2\mathsf{C}_2^2)(\mathsf{D}_2^1\mathsf{D}_2^2)$$

CommTask（A₁ 和 A₂）在前置 CompTask 完成后立即启动。这一顺序保证了任何满足依赖约束的合法调度都无法超越。最优性证明（定理 1）基于四项交换引理，每项证明交换任意两个 CompTask 的顺序不会使总执行时间更短。

从系统架构角度拆解术语：
ScheMoE 中 OptSche 的调度流程：

```
// MoE Layer Forward (r=2), 14 个 sub-tasks 入队
tasks = [C1_1, C1_2, A1_1, A1_2, D1_1, D1_2, E1, E2, C2_1, C2_2, A2_1, A2_2, D2_1, D2_2]

// Profiler 预热 (初次运行或配置变化时):
for each task_type in [C, A, D, E]:
    time_model[task_type] = profile_on_current_hardware(task_type)

// Scheduler 按 OptSche 最优顺序调度 CompTasks + CommTasks 异步启动:
// CompTask顺序: (C_1^1→C_1^2)→(D_1^1→E^1→C_2^1)→(D_1^2→E^2→C_2^2)→(D_2^1→D_2^2)
// A1_1 等待 C1_1 完成后在 Intra/Inter stream 启动
// A1_2 等待 max(C1_2完成, A1_1完成) 后启动
// A2_1 等待 max(C2_1完成, A1_2完成) 后启动
// A2_2 等待 max(C2_2完成, A2_1完成) 后启动
```

OptSche 相比 Tutel/Faster-MoE 的固定 schedule 的改进：(1) 数学最优性保证——无论硬件/模型配置如何变化，schedule 始终最优；(2) Profiler 自动适配——通过预热阶段测量实际各 task 耗时，适应不同硬件和 A2A 算法；(3) 可扩展——用户可继承 Scheduler 接口实现自定义调度算法。消融实验表明 OptSche 在 +ZFP+Pipe-A2A 基础上额外贡献 9% 加速。局限：当 A2A 通信占比很小时（< 5%），OptSche 仅带来 marginal 提升；当 EP 与 TP 混合时（expert 层使用 tensor parallel），任务依赖关系改变，当前 OptSche 不再适用。

术语一般如何实现？如何使用？
OptSche 作为 ScheMoE Scheduler 模块的核心实现（C++/CUDA，~1200 行核心代码），用户无需手动配置——系统在第一次 iteration 自动 profile 各 task 耗时并构建性能模型，随后按 OptSche 顺序调度。

涉及论文标题：
- ScheMoE: An Extensible Mixture-of-Experts Distributed Training System with Tasks Scheduling
