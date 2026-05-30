## FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 FineEP，一种基于线性规划的 token scheduling 策略实现 MoE 训练中的细粒度 GPU 负载均衡。核心算法包括：
  1. **LPP-based Load Balancing（§5.1）**：将每 micro-batch 的负载均衡建模为线性规划问题。变量 `x_e^g`（expert e 在 GPU g 上的 replica load），约束 `Σ_g x_e^g = load_e`（每个 expert 的总 load 分配到其 replicas），目标 `min max_g Σ_e x_e^g`（最小化最大 GPU load）。使用 HiGHs 求解器在 CPU 单线程求解，利用 warm-start 跨 micro-batch 复用。
  2. **Locality-Aware Token Routing（§5.2, Algorithm 1）**：贪婪路由策略——优先将 GPU g 上的 tokens 路由到同在 GPU g 的 local replica（减少通信），再路由到 remote replica。
  3. **Graph-Theoretic Expert Placement（§6）**：
     - Symmetric Placement（§6.2）：无先验 load 知识时，用 Cayley graphs 构造对称 expert placement（保证图密度最小化 max induced subgraph density）。
     - Asymmetric Placement（§6.3）：已知 load 分布时，greedy 确定 replica counts + Monte Carlo sampling 确定 placement graph（选 max induced subgraph density 最小的图）。
     - Adaptive Replacement（§6.4）：后台监控 load 分布 → 时间序列预测 → Equation 3 评估 → 触发 placement 更新。
  4. **Communication-Aware Scheduling（Appendix A.1）**：扩展 LPP 目标函数为 `min comp + α·comm`，区分 intra-node (α₁) 和 inter-node (α₂) 通信权重。
  5. **Pipelining（Appendix A.2）**：将 tokens 拆分为 EP（前者）和 FineEP（后者）两部分，用 EP 的 all-to-all 通信覆盖 FineEP 的调度时间。
  实验比较 FineMoE vs Megatron-LM/SmartMoE/FlexMoE/DeepSpeed 的端到端吞吐量、负载均衡（Zipfian skewness s∈[0,2]）、执行时间分解、调度开销（vs experts/GPUs scaling）、ablation（warm solving/locality-aware routing/overlapping）。

- 硬件平台是什么，配置是什么。
  4 节点，每节点 8×NVIDIA H100 80GB SXM GPU（共 32 GPU），900 GB/s NVLink intra-node，2×400 Gbps InfiniBand NIC per node。BF16 精度。PP=节点数（仅 inter-node），DP=8, EP=4, d=2。禁用 TP。Selective activation recomputation（仅 MoE FFN）。Distributed optimizers（类 ZeRO-1）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - GPT 32×1.3B: 24 layers, h=2048, FFN_h=8192, 32 experts, top-2
    - GPT 16×3.2B: 16 layers, h=4096, FFN_h=16384, 16 experts, top-2
    - GPT 8×6.7B: 32 layers, h=4096, FFN_h=16384, 8 experts, top-2
    - Mixtral 16×2B: 32 layers, h=2048, FFN_h=8192, 16 experts, top-2
    - Mixtral 8×7B: 32 layers, h=4096, FFN_h=14336, 8 experts, top-2
  - 数据集：Wikipedia（预训练）。
  - Benchmark：端到端训练吞吐量(tokens/s)、max GPU load / avg GPU load（负载均衡指标）、dispatch time（通信性能）、调度时间（overhead）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未声明开源，经 web search 未发现公开仓库。基于 Megatron-LM（github.com/NVIDIA/Megatron-LM）实现。

  **FineEP Token Scheduling 算法伪代码**：

```
输入: {input_e^g} (GPU g 上分配给 expert e 的 token 数), expert placement {G_EDP^e}
输出: token-to-(GPU, replica) mapping

// === Step 1: LPP Solving (per micro-batch) ===
// Solve: min max_{g in G_FineEP} sum_{e: g in G_EDP^e} x_e^g
// s.t.   sum_{g in G_EDP^e} x_e^g = load_e, ∀e
//        x_e^g >= 0
// where load_e = sum_g input_e^g
// Warm-start: reuse previous solution as initial simplex state
{x_e^g} = HiGHs_solve_warmstart(LPP, {load_e}, prev_solution)

// === Step 2: Locality-Aware Token Routing (Algorithm 1) ===
{remain_input_e^g} = {input_e^g}
{remain_x_e^g} = {x_e^g}

for each expert e in E:
    // Phase A: Route local tokens to local replicas (reduce all-to-all)
    for each GPU g in G_EDP^e:
        y = min(remain_input_e^g, remain_x_e^g)
        route next y tokens of expert e from GPU g to local replica on GPU g
        remain_input_e^g -= y
        remain_x_e^g -= y

    // Phase B: Route remaining tokens to remote replicas
    for each GPU g in G_FineEP:
        for each GPU g' in G_EDP^e:
            y = min(remain_input_e^g, remain_x_e^{g'})
            route next y tokens of expert e from GPU g to replica on GPU g'
            remain_input_e^g -= y
            remain_x_e^{g'} -= y

// === Step 3: Distributed Execution ===
// All GPUs execute all-gather({input_e^g})
// Each GPU independently runs Steps 1-2 (deterministic algorithm)
// Each GPU produces identical token-to-replica dispatching plan

// === Step 4: Optional Communication-Aware Variant ===
// Extended LPP 4:
// minimize comp + α·comm
// where comp = max_g sum_e x_e^g
//       comm = max_g max(send_g, recv_g)
//       send_g = (sum_e input_e^g) - local_g
//       recv_g = (sum_e x_e^g) - local_g
//       local_g = sum_e min(x_e^g, input_e^g)
// For topology-aware: split α into α₁ (intra-node) and α₂ (inter-node)
```

  **Graph-Theoretic Expert Placement 算法**：

```
// === Symmetric Placement (no prior load knowledge, §6.2) ===
// Use Cayley graphs: group = (Z_{2^p}, +) or product groups
// Example: 8 GPUs, 8 experts → (Z_8, +), generators {1, -1} → cycle graph
// Example: 16 GPUs, 32 experts → (Z_4×Z_4, +), generators {(0,1),(0,-1),(1,0),(-1,0)} → 4x4 toroidal grid
// Property: all edges (experts) are uniformly distributed, minimizing max induced subgraph density

// === Asymmetric Placement (known loads, §6.3) ===
// Step 1: Greedy replica count allocation
heap = max-heap of (expert e, load_e / replica_count_e)
while remaining_replicas > 0:
    (e, max_load_per_replica) = heap.pop()
    replica_count_e += 1
    heap.push(e, load_e / replica_count_e)

// Step 2: Monte Carlo placement sampling
best_placement = nil, best_score = inf
for iter in 1..M:  // M Monte Carlo iterations
    placement = random_assign_experts_to_gpus(replica_counts)
    // Equation 3: compute max density
    m = max_{G_max subset G} (1/|G_max| * sum_{e: G_EDP^e subset G_max} load_e)
    if m < best_score:
        best_placement = placement
        best_score = m

// === Adaptive Replacement (§6.4) ===
// Every ~50 iterations:
predicted_loads = moving_average(historical_loads)
future_m = Equation3_simulate(current_placement, predicted_loads)
if future_m > threshold:
    new_placement = asymmetric_placement(predicted_loads)
    reinitialize_model_states(new_placement)  // migrate expert params + optimizer states
```

  **关键超参数与结果**：
  - FineEP d=2 (DP_degree/EP_degree = 8/4)
  - HiGHs solver: 单 CPU thread, LP 变量数 O(|E|d), 约束数 O(|E|+|G|)
  - Scheduling overhead: ~100 μs (min) 到 <1 ms (64 GPUs, 256 experts)
  - Warm-start LPP solving: 进一步减少求解时间
  - Adaptive replacement interval: 50 iterations（训练初期），数百 iterations（训练后期）
  - 端到端加速：最多 47.6% vs Megatron-LM，平均 36.9%，超 FlexMoE 13.9%
  - 负载均衡：s<1 时 FineMoE (w/o AR) 完美均衡；s>1 时 FineMoE (with AR) 借助 asymmetric placement 保持完美均衡
  - 调度额外开销：仅 0.4 ms dispatch time vs vanilla Megatron-LM
