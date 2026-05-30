## Dynamic Expert Placement (动态专家放置 / Greedy Expert Scheduling)

术语是什么？
Dynamic Expert Placement 是 ES-MoE 的核心调度算法，在每 iteration 根据 gating network 的输出（各 expert 的 token 数量）动态决定 expert→GPU 的映射，目标是最小化各 GPU 间处理时间的差异（minimize makespan）。这与传统 expert parallelism 中 experts 到 GPU 的**静态映射**根本不同——传统 EP 中 expert 固定在指定 GPU 上，token 负载不均直接导致 GPU 间计算量不均。ES-MoE 借助 expert offloading（experts 从 CPU 按需加载），使 placement 决策在每 iteration 可自由调整，从而将负载均衡决策与 token routing 决策解耦。

从kernel调度角度拆解：
Dynamic Expert Placement 算法（Greedy Scheduling, Graham 1969, 4/3-approximation）：

```python
def greedy_expert_placement(expert_loads, num_gpus):
    """
    Input:
      expert_loads: List[(expert_id, token_count)], 所有 expert 的 token 数量
      num_gpus: 可用 GPU 数量
    Output:
      gpu_assignments: Dict[gpu_id -> List[expert_id]], 每个 GPU 分配的 experts
    
    Algorithm: Greedy Minimum Makespan Scheduling
    Complexity: O(m * log n + m * log m) where m=#experts, n=#GPUs
    Actual runtime: <2.69 us (on CPU), negligible vs ms-scale expert compute
    """
    # Step 1: Model each expert's processing cost
    for expert_id, token_count in expert_loads:
        compute_time = token_count * FLOPs_per_token / GPU_TFLOPS
        upload_time = expert_param_size / PCIe_bandwidth
        processing_time[expert_id] = max(compute_time, upload_time)
    
    # Step 2: Sort experts by processing time (descending)
    sorted_experts = sort_by_processing_time(expert_loads, descending=True)
    
    # Step 3: Greedy assignment
    gpu_loads = [0] * num_gpus
    gpu_assignments = {gpu_id: [] for gpu_id in range(num_gpus)}
    
    for expert_id, _ in sorted_experts:
        # Assign to GPU with minimum accumulated load
        target_gpu = argmin(gpu_loads)
        gpu_assignments[target_gpu].append(expert_id)
        gpu_loads[target_gpu] += processing_time[expert_id]
    
    return gpu_assignments
```

效果（ES-MoE 论文 Figure 6）：MoE-M 64 experts，传统静态 placement 下 GPU 间 token 数差异达 102%（max/min ratio），动态 placement 将差异降至 15%。同时完全消除 zero-padding（不再需要统一 batch size 的 batched MM）。

术语一般如何实现？如何使用？
- 算法运行在 CPU 上（gating network 输出后、token permutation 前），每次 iteration 调用
- 处理时间建模需考虑 expert 参数传输时间 + 计算时间，传输时间取决于 PCIe 带宽和 expert 参数量
- 在 cloud VM 场景下，算法可扩展考虑异构 GPU 能力（不同 GPU model 有不同 TFLOPS）
- 局限性：当 expert offloading 不可用时（GPU only 模式），动态 placement 无法使用

涉及论文标题：
- Scaling Beyond the GPU Memory Limit for Large Mixture-of-Experts Model Training
