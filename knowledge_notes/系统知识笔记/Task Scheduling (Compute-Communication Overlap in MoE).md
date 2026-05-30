## Task Scheduling (Compute-Communication Overlap in MoE)

术语解释
MoE任务调度是将MoE推理中的计算任务（expert FFN、attention）和通信任务（All-to-All）进行流水线化或重叠调度，以隐藏通信延迟，减少端到端延迟。

术语是什么？
MoE层的执行天然包含两个阶段：计算和通信。通信延迟（All-to-All）往往是主要瓶颈。任务调度的核心是将计算和通信重叠执行：
- **ScMoE**：使用shortcut-connected MoE架构——来自前一层的top-1 MoE（shortcut）+ 当前层的shared expert，解耦通信顺序，实现通信与计算100%重叠
- **HiDup**：将每台设备的输入数据拆分为两个无依赖microbatch，一个microbatch的计算与另一个microbatch的通信并行
- **MoESys**：弹性MoE训练策略，2D预取 + 分层存储上的融合通信，计算与参数就绪状态重叠
- **ScheMoE**：模块化计算任务（数据压缩、expert计算）和通信任务（集合通信），设计自适应最优调度算法流水线化这些模块化操作
- **PipeMoE**：性能模型预测计算和通信成本 → 最优多项式时间解决方案pipeline任务以隐藏通信延迟
- **EPS-MoE**：动态选择最优kernel实现，自适应重叠FFN计算与All-to-All通信

从系统架构角度拆解术语。
以PipeMoE的任务调度为例：
```
# 阶段1：性能建模
T_comp[l] = estimate_expert_compute_time(experts_on_device, tokens)
T_comm[l] = estimate_alltoall_time(tokens_cross_device, bandwidth)

# 阶段2：最优调度
# 目标：min端到端时间 → pipeline通信与计算
# PipeMoE的多项式时间解：
schedule = []
for each microbatch:
    # 通信开始 → 计算开始的时间偏移
    Δ[l] = optimal_overlap(T_comp[l], T_comm[l], 
                           dependency_graph)
    launch_async_comm(microbatch, offset=0)
    launch_async_compute(microbatch, offset=Δ[l])
```

术语一般如何实现？如何使用？
- 需要异步执行能力（CUDA streams、多线程）
- 关键挑战：正确建模通信和计算时间以确定最优偏移量
- 适用场景：分布式MoE训练/推理（多GPU场景）
- 效果：训练加速1.25x-1.69x，推理加速2.2x（ExFlow）

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models

---
