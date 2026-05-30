## Towards MoE Deployment Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：论文提出三项 MoE 推理阶段的算法优化：(1) **Dynamic Gating（动态门控）**：提出基于 argsort 的可变大小 token 分发算法替代传统基于 static capacity + dispatch mask 的方案，核心计算为 argsort(O(S log S)) + bin-count(O(S)) + index(O(SD))，避免空 token placeholder 传输和 token dropping；(2) **Expert Buffering（专家缓冲）**：基于观察到的 MoE 推理中 expert 激活高度稀疏但具有强时序局部性（temporal locality），设计 GPU expert cache + CPU offload + LIFO 淘汰策略，利用两个关键观察——(a) expert 负载高度不平衡（存在高频 hot experts），(b) 同一 expert 在连续 batch 中持续活跃（时序局部性）；(3) **Load Balancing（负载均衡）**：将 expert-to-GPU 分配建模为 multi-way number partitioning 问题（NP-hard），提出 Greedy Balancing（基于独立激活的贪心分配）和 Anti-correlation Balancing（考虑 expert 间 Pearson 相关性，修改负载公式为 $\sum_{m} P_{mn} (\tilde{A}_{m} + 0.5 * S_{am})$）两种近似算法。
  - 实验比较：(a) Dynamic gating vs Static gating vs Tutel gating 在不同 batch size 和节点数下的吞吐量（LM: 6.21-11.23×、MT Encoder: 5.75-10.98×、MT Decoder: 2.58-5.71× 吞吐提升，Figure 9）；(b) 不同 gating 策略的内存消耗（动态内存+静态内存分解，Figure 10），dynamic gating 使 LM batchsize=8 的激活内存从 6.29GB 降至 1.28GB（79.6% 减少）；(c) Expert Buffering 下不同 cache 大小的 cache miss rate 与 Belady's MIN 对比（Figure 12）；(d) Load Balancing 对负载分布的影响（Max Load 和 Avg Max Load 指标，Figure 14）；(e) 三种优化组合的吞吐量和内存（Figures 9, 10）。

- 硬件平台是什么，配置是什么。
  - CPU: 2×Intel Xeon E5-2698 v4 @ 2.2GHz，700GB DDR4
  - CPU-GPU: PCIe 3.0 ×16，实测带宽饱和在约 12GB/s
  - GPU: 8×NVIDIA Tesla V100 (Volta)，32GB HBM2 @ 900GB/s，NVLink 300GB/s（单节点内）；多节点通过 InfiniBand 互联（带宽论文未明确说明）
  - 单节点(8 GPU)到四节点(32 GPU)的扩展实验

- 模型是什么。数据集和bench分别是什么。
  - 模型: (a) **Language Modeling MoE**：52B 参数，E=512 experts，24 layers，TD=1024，HD=4096，MF=2（每 2 层中 1 层为 MoE），C=0.05，top-2 gating，vocab=51200；Dense baseline: 355M 参数，相同层数和隐藏维度；(b) **Machine Translation MoE**：54.5B 参数，E=128 experts，48 layers（encoder+decoder），TD=2048，HD=8192，MF=4，C=1，top-2 gating，vocab=256206；Dense baseline: 3.3B 参数
  - 数据集: (a) LM: PILE [8] validation set，选取 Wikipedia、PubMed、Github 三个子域分析不同数据对 expert 激活模式的影响；(b) MT: NLLB-200 [22] validation set，English→French/Japanese/Asturian 三个目标语言
  - Benchmark 指标: Throughput (tokens/s)、Latency (ms per batch)、Memory Usage (peak GPU memory, static vs dynamic breakdown)、Cache Miss Rate、Max Load / Avg Max Load

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供独立开源仓库链接。论文明确说明实现基于 fairseq [23]（https://github.com/facebookresearch/fairseq），未提供论文修改版本的 fork/branch 链接。
  - 算法pipeline 伪代码级解释：
    **Dynamic Gating 核心算法**（在 MoE 层的 gating 函数之后执行）：
    ```
    输入: gate_scores = (B*S, E)  # gating 对每个 token 的打分
          capacity_factor = C     # 静态方案中的容量参数，动态方案不再需要
          n_experts = E
          top_k = 2               # top-2 gating

    # 1. 计算 routing decisions
    expert_weights, expert_indices = topk(gate_scores, top_k)  # (B*S, 2), (B*S, 2)
    # expert_indices[i] = [e1, e2], 第 i 个 token 被路由到专家 e1 和 e2

    # 2. 展平并排序（动态门控核心）
    flat_expert_indices = expert_indices.reshape(-1)  # (B*S*top_k,)
    sort_order = argsort(flat_expert_indices)          # O(n log n), n = B*S*top_k
    sorted_indices = flat_expert_indices[sort_order]

    # 3. 统计每个专家分配的 token 数量
    expert_counts = bincount(sorted_indices, minlength=E)  # O(n), shape (E,)

    # 4. 按 expert_id -> GPU_id 映射，聚合每 GPU 的接收量
    # expert_to_gpu[e] = device_id
    gpu_sizes = [sum(expert_counts[e] for e in experts_on_gpu[g]) for g in range(n_gpus)]

    # 5. 两阶段 all-to-all
    # Phase 1: 交换每个 GPU 的 token 接收量（极小消息）
    all_to_all_size(gpu_sizes)
    # Phase 2: 按 sort_order 重排 tokens 后分片发送（实际数据传输）
    reordered_tokens = tokens[sort_order]  # index-based, O(n*D)
    all_to_all_data(reordered_tokens, split_sizes=gpu_sizes)

    # 6. 各 GPU 按 expert 执行 FFN
    for expert_id in local_experts:
        if expert_counts[expert_id] > 0:
            expert_outputs = expert_ffn(expert_inputs[expert_id])
    ```
    与 Static Gating 对比（原方案）：
    ```
    # 原方案: 创建 dispatch mask, 大小 (E, S, S*C)
    dispatch_mask = zeros(E, S, S*C)           # 大量零值
    # 填 mask 过程中检查容量，超出的 token 被丢弃
    dispatched = einsum('ij,ijk->ik', tokens, dispatch_mask)  # 巨大稀疏矩阵乘
    # 即使用不到的空 capacity 也会传输 placeholder (零向量)
    ```
    **Expert Buffering 核心算法**：
    ```
    输入: active_experts = [e for e in local if expert_counts[e] > 0]
          gpu_cache = {expert_id: parameters}  # 大小可配置，如 10 experts/GPU
          cpu_memory = {expert_id: parameters} # 所有 experts 的完整参数

    for expert_id in sorted(active_experts):  # 按 expert_id 升序串行执行
        if expert_id in gpu_cache:
            params = gpu_cache[expert_id]      # Cache hit
        else:
            if gpu_cache.is_full():
                # 淘汰策略：优先淘汰非当前 batch 活跃的 expert
                inactive_in_cache = [e for e in gpu_cache if e not in active_experts]
                evict(inactive_in_cache[0])   # LIFO (最后加入的先淘汰)
            params = memcopy(cpu_memory[expert_id] -> gpu_cache)  # 与 all-to-all 并行
        output[expert_id] = expert_ffn(inputs[expert_id], params)
    ```
    **Load Balancing (Greedy Balancing) 核心算法**：
    ```
    输入: historical_activations A_mb, shape (E, B)  # expert m 在 batch b 的负载比例
    输出: expert_to_gpu[e] = device_id              # E/gpu 每设备

    avg_load = mean(A_mb, axis=1)                    # 每个 expert 的历史平均负载 (E,)
    sorted_experts = argsort_descending(avg_load)     # 按负载从高到低排序
    gpu_loads = zeros(n_gpus)                        # 各 GPU 累积负载
    gpu_capacity = E // n_gpus                       # 每 GPU 的 expert 数量上限

    assignment = [-1] * E
    for expert_id in sorted_experts:
        # 选当前负载最小的 GPU（已满的不可选）
        candidates = [g for g in range(n_gpus) if gpu_count[g] < gpu_capacity]
        best_gpu = argmin(gpu_loads[g] for g in candidates)
        assignment[expert_id] = best_gpu
        gpu_loads[best_gpu] += avg_load[expert_id]
        gpu_count[best_gpu] += 1
    ```
