## DAG-based Batching Strategy Search (DAG + DP for Critical Path Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DAG-based Batching Strategy Search 是 MoE-GEN 提出的离线推理配置优化方法。将 MoE 模型的单层 forward pass 抽象为有向无环图（DAG），其中节点为 computation（attention pre/post、expert、CPU attention）或 memory copy（HtoD weight prefetch、HtoD KV-cache copy、DtoH KV-cache update），边为依赖关系。目标是在给定硬件配置下，从搜索空间（B, b_a, b_e, ω, S_Expert, S_Params）中枚举候选配置，对每个配置用动态规划（DP）计算 DAG 的 critical path（最长路径），选择执行时间 T 最短的配置。这本质上是一个离线 scheduling 优化问题：将 memory copy 和 computation 作为可调度的 job，通过控制 batch sizes 影响每个 job 的执行时间，最大化 pipeline overlap。

从编译框架角度拆解术语，比如术语如何在编译框架中发挥作用，给出术语在编译框架中运转流程的具体例子。通过联网搜索让回答具体和精准。
MoE-GEN 的 DAG-based 搜索流程（类似编译器的 auto-tuning 过程）：

```
输入: Hardware profile (GPU/CPU compute latency per module, PCIe bandwidth, memory)
      Model architecture (num_layers, num_experts, expert_size, attention_size, kv_dim)
      User task (prompt_length, decoding_length, host_memory_capacity)

离线 Profiling 阶段 (before runtime):
  for each module m in {attention_pre, attention_self_gpu, attention_self_cpu,
                         attention_post, expert_ffn, kv_cache_copy, weight_copy}:
    for each batch_size in {1, 2, 4, ..., max_batch}:
      latency[m][batch_size] = measure(module_m, batch_size)
      peak_memory[m][batch_size] = measure_peak_memory(module_m, batch_size)
  PCIe_bandwidth = timed_cudaMemcpy(large_buffer) / time

搜索阶段:
  best_config = None, best_throughput = 0
  for B in candidate_B_range:          // 受 host memory 约束
    for b_a in candidate_ba_range:     // attention micro-batch
      for b_e in candidate_be_range:   // expert micro-batch
        for ω in {0, 0.1, ..., 1.0}:  // CPU split ratio
          for S_Expert in candidate_expert_buffer:
            // 检查 memory constraint (Eq 2, 3)
            if violates_memory_constraint(B, b_a, b_e, S_Expert):
              continue
            
            // 构建 DAG 并求解 Critical Path
            T = solve_dag_critical_path(B, b_a, b_e, ω, S_Expert, profiles)
            throughput = B / T
            
            if throughput > best_throughput:
              best_throughput = throughput
              best_config = (B, b_a, b_e, ω, S_Expert)

  return best_config
```

DP 求解 DAG Critical Path（公式 4）：
```
function solve_dag_critical_path(B, b_a, b_e, ω, S_Expert, profiles):
    nodes = build_dag(B, b_a, b_e, ω)  // 根据配置构造 DAG
    topo_order = topological_sort(nodes)
    
    dp[node] = 0 for all nodes
    for v in topo_order:
        dp[v] = max(dp[u] for u in predecessors(v)) + cost(v)
        // cost(v) = latency lookup from profiling data
    
    return dp[exit_node]  // critical path length = total layer time
```

MoE offloading DAG (解码阶段单层，图 6)：
```
[Copy Attn Weights] ──→ [Pre-Attention GPU] ──→ [Copy KV-Cache] ──→ [GPU Self-Attn] ──┐
                                                    │                                     │
                                                    └──→ [CPU Self-Attn] ────────────────→ [Post-Attention] 
                                                                                               │
                                                                                               ↓
                                                                               [Copy Expert1 Weights] ──→ [Expert1 FFN] ──┐
                                                                                       │                                  │
                                                                                       └──→ [Copy Expert2 Weights] ──→ [Expert2 FFN] ──→ ... ──→ [Output]
```

DAG 的依赖关系：
- Post-Attention 依赖 GPU Self-Attn（需要等 KV-cache copy 完成）和 CPU Self-Attn（可直接读 host KV-cache）两者的结果 concatenate。
- 第一个 Expert 依赖 Router（在 Post-Attention 内部），后续 Expert 顺序依赖。
- Expert weights copy 可与前一个 Expert compute 重叠（但需在前一个 Expert 开始后才启动）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **MoE-GEN 实现**：搜索在 runtime 前离线完成，结果（最优配置）传给 MoE-GEN Engine。Profiling 使用 CUDA events 测量延迟，torch memory stats API 测量峰值内存。搜索空间枚举后对每个配置调用 DP solver。
- **与 ML 编译器 auto-tuning 的关系**：类似 TVM AutoScheduler 或 Triton 的 auto-tuning，但 MoE-GEN 调的是 batch size / buffer size（系统级 hyperparameters）而非 kernel tile size / loop order（算子级）。本质上是将 scheduling 优化建模为 combinatorial search over configuration space + latency predictor（profiling-based）。
- **局限性**：
  - Profiling 开销：需对每个 (module, batch_size, seq_len) 组合预先 profiling，组合数较多。但 profiling 是一次性开销，可用于所有后续任务。
  - 搜索精度：依赖 profiling 数据的准确性。若实际运行时因 GPU 频率波动或 PCIe contention 导致 latency 偏移，最优配置可能不再是全局最优。
  - Search space discretization：连续变量（如 batch size）被离散化为候选值集合，可能错过理论最优值。
- **类似工作**：
  - **FlexGen**：使用线性规划搜索最优的 GPU/CPU/disk offloading 策略。
  - **Alpa**：使用整数线性规划自动搜索 intra/inter-operator parallelism 配置。
  - **MoE-Lightning**：使用 profiling-based 方法搜索最优 memory movement schedule。

涉及论文标题：
- MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

从编译框架角度拆解术语：
IP 在位宽分配中作为"离线编译器"的角色：将 expert 重要性分析的结果（ϕᵢ, wᵢ, εᵢⱼ）作为输入，输出每个 expert 的最优位宽配置，等价于编译器中资源分配优化的 register allocation 或 instruction scheduling 问题。具体流程：
```
输入: 每个 MoE layer 的 N 个 expert，目标平均位宽 k
// Step 1: 计算损失矩阵
for each expert i, bit j ∈ {1,2,3}:
    ε_{i,j} = ||F(θ) - F(θ[e_i → Q(e_i, j)])||_F  // 量化重构 F-norm
    c_{i,j} = ϕ_i^α · w_i^β · ε_{i,j}^γ              // 加权损失

// Step 2: 构建并求解 IP
// 二元变量 x_{i,j}: expert i 是否分配 j bit
MIN  Σ_i Σ_j c_{i,j} · x_{i,j}
s.t. Σ_i Σ_j j · x_{i,j} = N · k          // 全局平均位宽
     Σ_j x_{i,j} = 1, ∀i                  // 唯一分配
     Σ_i x_{i,3} ≥ 1, Σ_i x_{i,2} ≥ 1    // 至少一个 3/2-bit
     x_{i,j} ∈ {0,1}

// Step 3: 输出配置
B_i = argmax_j x_{i,j}  // 每个 expert 的位宽
```
求解可使用 Gurobi、OR-Tools、PuLP 等标准 IP solver，或暴力枚举（24 变量规模下可行）。

术语一般如何实现？如何使用？
- IP 在位宽分配中的使用场景：(1) MoE-LLM 的 expert-wise 混合精度量化（MC-MoE）；(2) Dense LLM 的 layer-wise/block-wise 混合精度（如 HAWQ-V2 使用 Hessian 作为损失度量）；(3) 通用神经网络中按 filter/channel 分配位宽。
- 实现依赖于：(a) 准确的损失度量（F-norm/Hessian/PPL 等）；(b) 合理的约束设计（平均位宽、最小/最大位宽、保护约束）；(c) 高效的求解器（IP 在小规模下秒级求解，大变量需启发式或贪心替代）。
- 优势：全局最优（在给定损失函数下）、约束灵活可扩展、无需梯度。局限：损失度量的准确性直接影响分配质量；大规模实例下 IP 求解可能指数爆炸，需用 LP 松弛或贪心替代。
- 在 MC-MoE 中，PMQ 的 IP 约束还包括"至少一个 expert 为 3-bit 和 2-bit"，防止所有 expert 坍塌到同一极端位宽。

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---
