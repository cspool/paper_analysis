## Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

- baseline方法是什么？
  Baseline 是 EPLB (Expert Parallelism Load Balancer) 的 token routing 算法——将每个 expert 的 token 均匀分配到其所有 replicas 上（"token-balancing"）。EPLB 的完整流程包含两步：(1) expert replication：按上一时间窗口各 expert 处理 token 数的比例创建 replicas；(2) expert placement：将 replicas 放置在 GPU 上以平衡各 GPU 期望处理的 token 数；(3) token routing：将每个 expert 的 token 均匀分配到其 replicas 上。Baseline 的核心假设是"GPU runtime 与处理的 token 数成正比"，这在 compute-bound 场景（prefill）下成立，但在 memory-bound 场景（decode）下不成立。

  **Baseline 全栈执行例子（以 Qwen3-30B decode batch=32 tokens/GPU, 8×A100, 1.5× replication 为例）**：
  - **算法层**：MoE router 为每个 token 计算 top-k experts；EPLB token routing 将每个 expert 的 token 均匀分配到所有 replicas——若 expert e 有 3 个 replicas 和 9 个 token，则每个 replica 分到 3 个 token
  - **系统框架层**：vLLM EP——Attention 层 DP，MoE FFN 层 EP 分布在 8 个 GPU 上。EPLB 的 expert placement/replication 根据历史 token 分布周期性更新
  - **编译框架层**：vLLM CUDA Graph compilation——prefill 和 decode 的 MoE 层计算被编译为 CUDA Graphs（论文未修改编译框架）
  - **Kernel 调度层**：decode 阶段每个 MoE layer 执行：local top-k → all-to-all dispatch → FFN (cutlass GroupGemm) → all-to-all combine。由于令牌均匀分布，每个 GPU 上可能激活更多 expert replicas（每个 expert 的 token 分散到多个 replica），导致加载 expert weight 的内存流量增加
  - **硬件架构层**：8×NVIDIA A100 40GB, 600 GB/s NVLink。Memory-bound 的 decode 阶段，FFN runtime 由 HBM → Tensor Core 的 weight 加载带宽决定，而非 Tensor Core 计算。更多 activated experts = 更多 weight 需加载 = 更长的 memory traffic = 更高的 decode latency

  **Baseline 核心痛点**：
  1. EPLB token-balancing 在 memory-bound decode 阶段错误地增加了 activated expert replicas 数量（1.5× replication 下 activated experts 增加 ~30% vs no-replication），因为均匀分配迫使更多 replica 被激活
  2. 增加的 activated experts 导致更多 expert weights 需从 HBM 加载到 Tensor Cores，memory traffic 增加，decode latency 恶化（1.5× replication 下 +14% TPOT）
  3. 虽然 replication 能改善 compute-bound prefill 性能（-17% TTFT），但对 memory-bound decode 的退化反而导致 overall token throughput 下降（-10% at 1.5× replication）
  4. 换言之，EPLB 强制 prefill 和 decode 使用同一套 load-balancing 策略，无法为 memory-bound decode 阶段做针对性优化

- 论文方法是什么？如何对应解决Baseline的缺陷？
  METRO 提出将 token routing 目标从 "balance tokens across GPUs" 改为 "minimize activated experts across GPUs"，因为 memory-bound regime 下 GPU runtime ∝ activated experts 数量而非 token 数量。具体设计包括三个组件：

  1. **MIN-EXP-ROUTING 问题形式化**：将最小化 activated experts 建模为 ILP 问题，通过 Lemma 1 证明只需将每个 expert 的所有 token 路由到单个 replica 即可——这从根本上避免了 token-balancing 导致的 replica inflation
  2. **GPU-native 贪心近似算法**：由于 ILP 最优解计算开销过大（31%-104% FFN time），METRO 使用 O(|A|) 复杂度的 greedy 算法——每个 expert i 并行选择当前 activated experts 最少的 GPU g*，全序加锁避免死锁，运行在单 SM 上仅需 17-26us
  3. **All-gather dispatch 替换 all-to-all**：使每个 GPU 获得全局 top-k 知识 T[1..N]，作为 Algorithm 1 的输入。在 memory-bound 小 batch 下的开销可忽略（~3us bandwidth vs ~100us NCCL launch latency）

  **论文方法全栈执行例子（以同样 Qwen3-30B decode batch=32 tokens/GPU, 8×A100, 1.5× replication 为例）**：
  - **算法层**：MoE router top-k 不变。METRO routing：对于每个有 token 的 expert e，查 placement matrix 获取其候选 GPU 集合，选择当前 activated expert 计数最少的 GPU 进行路由（greedy min-L），该 expert 所有 token 路由到同一 GPU。例如 expert e 有 3 个 replicas (GPU 0, 3, 5)，当前 L=[2,1,1,1,2,0,1,1]，则选 GPU 5 (L[5]=0)，e 的所有 token 全部路由到 GPU 5，仅激活一个 replica
  - **系统框架层**：vLLM EP + METRO——Attention 层 DP 不变。METRO 仅替换 decode phase 的 token routing 逻辑，prefill phase 继续用 EPLB routing。METRO 与 EPLB 共用 expert placement/replication 策略，不干扰 prefill
  - **编译框架层**：vLLM CUDA Graph compilation——METRO routing kernel 被编译进 decode phase CUDA Graphs，power-of-two batch sizes 预编译（论文未修改编译框架本身，仅添加 kernel 到 graph）
  - **Kernel 调度层**：decode 阶段每个 MoE layer 执行：
    1. all-gather tokens（替换 all-to-all dispatch）→ 每个 GPU 获得全局 ~256 tokens
    2. 每个 GPU 在全局 token 集上计算 top-k → 构建 T[1..128]
    3. METRO CUDA kernel（单 SM）greedy 路由 → 每个 expert 匹配到单一 GPU
    4. Expert FFN 仅计算分配给本 GPU 的 activated experts → 仅加载被激活 expert 的 weight
    5. all-to-all combine 返回结果
    ⚡ 关键改进：activated experts 减少 up to 42.3% → memory traffic 减少 → FFN 时间减少 up to 81us/layer → end-to-end decode latency 降低 11%-22%
  - **硬件架构层**：8×NVIDIA A100 40GB, 600 GB/s NVLink。METRO 不改变硬件使用方式，但通过减少 activated experts 数量直接降低了 HBM → Tensor Core 的 weight 加载量。在 memory-bound regime 下，weight 加载是 runtime bottleneck，减少 activated experts = 减少 memory traffic = 降低 latency

  **痛点映射**：
  | Baseline 痛点 | METRO 解决方案 |
  |---|---|
  | Token-balancing inflate activated experts (+30%) | MIN-EXP-ROUTING 最小化 activated experts per GPU (-42.3% vs EPLB) |
  | 更多 activated experts → 更多 weight 内存流量 → decode latency 恶化 | Greedy routing 使每个 expert 的 token 集中到单一 replica，减少 activated experts → 减少 memory traffic → decode latency 降低 11%-22% |
  | EPLB 强制 prefill/decode 用同一 routing 策略 | METRO 仅应用于 memory-bound decode phase，prefill 继续用 EPLB token-balancing |
  | ILP 最优解计算开销过大（31%-104% FFN time） | O(|A|) greedy algorithm on single SM：17-26us，near-optimal (within 10.9% of optimal) |
  | All-to-all dispatch 无法提供全局 top-k 信息 | All-gather dispatch 使各 GPU 获得全局 T[1..N]，overhead 在 memory-bound 小 batch 下可忽略 |

  实验效果：(a) decode latency 降低 11%-22%；(b) total token throughput 提升 3%-21%（co-deployed prefill+decode）；(c) 在固定 SLO 下 decode throughput 达 EPLB 的 1.98x-4.11x；(d) 这些增益在 Qwen3-30B、Qwen3-235B、DeepSeek-V3 等多种模型和 InstructCoder、NuminaMath、Humaneval、GSM8K 等多种 workload 上一致。
