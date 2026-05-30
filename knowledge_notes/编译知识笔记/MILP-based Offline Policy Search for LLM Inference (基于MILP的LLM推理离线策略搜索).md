## MILP-based Offline Policy Search for LLM Inference (基于MILP的LLM推理离线策略搜索)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MILP (Mixed Integer Linear Programming) based Policy Search 是 MoE-Lightning 提出的离线最优推理策略搜索方法。给定硬件配置 H（GPU/CPU memory、bandwidth、FLOPS）、模型配置 M（layers、hidden dims、experts count、top-k）和工作负载配置 W（avg prompt length、generation length），MILP 求解器搜索使 per-layer decode latency T 最小的 6 元组策略 P = (N, μ, A_g, F_g, r_w, r_c)，同时满足 CPU/GPU memory 约束。优化目标式 (12)：T = max(comm^{cpu_to_gpu}, T_cpu, T_gpu)，其中 T_gpu = T_attn^g + T_ffn^g，每个 T_x = max(comm_x, comp_x)。搜索空间包含：2 个整数变量（N 批量大小、μ 微批次大小）、2 个二元变量（A_g CPU/GPU attention 指示、F_g CPU/GPU FFN 指示）、2 个连续变量（r_w GPU static weights 比例、r_c GPU KV cache 比例）。MILP 搜索时间 < 1 分钟（offline），无需像 FlexGen 那样进行数小时的 data fitting。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
MILP Policy Search 流程：
```
Input: H (hardware specs), M (model specs), W (workload specs)

1. For each (N, μ) candidate:
    a. Compute T_attn^g, T_ffn^g, T_attn^c, T_ffn^c using HRM:
       T_x = max(comm_x, comp_x)
       comm_x = bytes_x / b_g (or b_c), comp_x = flops_x / p_g (or p_c)
    b. Compute comm^{cpu_to_gpu} = total_bytes_H2D / b_cg
    c. T = max(comm^{cpu_to_gpu}, T_cpu, T_gpu)
    d. Check constraints: GPU mem usage <= m_g, CPU mem usage <= m_c
    e. If feasible and T < best_T: update best policy

2. Output: P* = (N*, μ*, A_g*, F_g*, r_w*, r_c*)
```

关键约束：
- GPU memory: weight_buffer (2 × per_layer_weights × (1-r_w)) + activations (μ × hidden_size × 2) + KV cache if A_g=1
- CPU memory: model_weights (for offloading) + KV cache for all batches (N × s × n_kv × d)
- A_g = 0 时不需要传输 KV cache H2D（CPU attention 替代）；A_g = 1 时采用 S4 schedule
- 典型结果（Mixtral 8x7B on T4）：A_g=0, F_g=1, N=504, μ=36, r_w=0, r_c=0

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：MoE-Lightning 使用开源 MILP solver（论文未指定，可能使用 Python PuLP + CBC 或 OR-Tools）。搜索空间较小（N 和 μ 受限于 memory 物理上限，二元变量 2 个，连续变量 2 个），exhaustive grid search 也可行。
- 输入来源：硬件参数通过 profiling 获取（peak FLOPS, peak BW——cudaMemcpy 测 PCIe BW，STREAM benchmark 测 CPU BW）；模型参数从 config.json 读取；工作负载参数由用户指定。
- 与 FlexGen 对比：FlexGen 需要数小时的 offline data fitting（运行大量不同 policy 的实测来建立延迟模型），而 MoE-Lightning 的 HRM MILP 仅需理论与实践峰值参数，搜索时间 < 1 分钟。代价是精度略低（但足以区分不同策略的相对效果）。
- 离线特性：Policy 在推理服务启动前一次性生成，不随在线请求动态调整。MoE-Lightning 的 request batching（Algorithm 2）确保在线执行接近最优 micro-batch 填充。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
