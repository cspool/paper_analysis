## JANUS: Disaggregating Attention and Experts for Scalable MoE Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - JANUS 提出三个算法层面的 MoE 推理优化：
    1. **Activated-Expert-Balanced Scheduling (AEBS, Algorithm 1)**：将 MoE 层激活 expert 调度形式化为负载均衡问题——收集 batch 中所有 token 的 top-K routing 结果获得激活 expert 集合 → 单副本 expert 分配至唯一持有实例 → 多副本 expert 贪心分配至当前激活 expert 数最少的实例 → 重写每个 token 的路由结果为物理副本 ID。目标是 minimize a_max = max_i(activated experts on instance i)，因为 MoE 层延迟主要由 straggler（a_max 最大的实例）决定。每 MoE 实例独立运行相同 deterministic 算法，实现 synchronization-free。
    2. **Monte Carlo a_max Estimator + Theoretical Bound（Section 3.5, Appendix A）**：将 expert 激活建模为 balls-into-bins process，推导 closed-form upper bound: a_max ≤ min(C, ā_max + sqrt(2·ā_max·ln n_e)) + 1 (Eq. 5)。利用 recent activation trace 构建 Monte Carlo estimator â_max(n_e, B) 查找表——对每个候选 (n_e, B) 从 trace 采样 B tokens → 应用 AEBS 策略 → 记录结果 â_max。表格周期性重建以适应当前 workload。
    3. **Fine-Grained SLO-Aware Resource Scaling（Section 3.5, Algorithm 2 + Eq. 1-3）**：基于 Roofline + Little's Law 构建 TPOT 性能模型。Attention latency (Eq. 1b) 遵循 Roofline: memory-bound plateau c_a + computation/KV-cache access αb + c_kv·b·S_ctx。MoE latency (Eq. 1c): β·a_max + c_e。稳态 batch size B* 由 Little's Law: B* = λ·TPOT(B*) 求解 (bounded binary search)。枚举 (n_a, n_e) 搜索空间 → 求解 B* → 检查 SLO + memory feasibility → 选择 min(n_a+n_e)。Activation-Aware Replica Placement (Algorithm 3, Appendix B): min-max 优化 co-activation load I(g) = Σ a(e,e')，贪心放置 + bounded swap 解决。
  - 实验比较：
    - AEBS vs EPLB：a_max reduction (Fig. 13), MoE-layer latency (Fig. 14), scheduling overhead (Fig. 15)
    - Full JANUS vs ablations：2PC+EGate+AEBS vs 1PC+EGate vs 2PC+AGate (Fig. 12)
    - Scaling quality：搜索空间可视化 (Fig. 16)，验证 JANUS 选择的资源高效配置
    - Resource cost：24h production trace 下 GPU-hour 节省 39% vs SGLang, 16% vs MegaScale-Infer (Fig. 11)
    - Monte Carlo bound validation：Analytical bound vs â_max across n_e ∈ {6,8,12,16} (Fig. 17)

- 硬件平台是什么，配置是什么。
  - 4 节点 × 8× NVIDIA H100 80GB (共 32 GPU)
  - Intra-node: NVLink 900 GB/s, Inter-node: IB 400 Gbps
  - 模型：DeepSeek-V2, Qwen3-MoE, Scaled-DS variants (top-k=8, 160/200 experts)
  - 所有参数 KV 缓存 BF16 格式

- 模型是什么。数据集和bench分别是什么。
  - **模型**：
    - DeepSeek-V2：MoE, top-k routing, 共享+路由 expert 架构
    - Qwen3-MoE (235B)：MoE, 含共享 expert
    - Scaled-DS-1：top-k=8, 160 experts/layer, expert intermediate=1024
    - Scaled-DS-2：top-k=8, 200 experts/layer, expert intermediate=1536
  - **数据集**：
    - ShareGPT：avg input 16 tokens + avg output 256 tokens，用于端到端 TPOT/TPG 测量
    - BurstGPT：合成动态到达 trace，模拟生产 LLM 服务负载
    - Production trace (24h)：真实 LLM 服务 trace，用于 scaling 行为评估
  - **Benchmark**：TPOT SLO 满足率，per-GPU throughput (TPG)，GPU-hour 消耗

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确给出 JANUS 公开开源仓库。基于 SGLang (https://github.com/sgl-project/sglang) 实现。
  - **AEBS 算法伪代码（Algorithm 1）**：

    ```
    Algorithm 1: Activated-Expert-Balanced Scheduling (AEBS)
    Input:
      T: number of tokens, n_e: number of MoE instances
      k: activated experts per token
      L(i,j): logical expert ID for token i, expert j
      R(e): number of replicas for expert e
      G(e): set of instances hosting replicas of expert e
      P(e,g): physical replica ID of expert e on instance g
    Output:
      O(i,j): physical replica ID for token i, expert j

    1. E ← ∪_{i=1}^{T} ∪_{j=1}^{k} {L(i,j)}  // 收集所有激活 experts
    2. Initialize actRep[e] ← -1 for all e ∈ E
    3. Initialize load[g] ← 0 for all g ∈ {1, ..., n_e}

    // Assign single-replica experts
    4. for all e ∈ E where R(e) == 1 do
    5.     g ← unique instance in G(e)
    6.     actRep[e] ← P(e,g)
    7.     load[g] ← load[g] + 1

    // Assign multi-replica experts via greedy load balancing
    8. for all e ∈ E where R(e) > 1 do
    9.     g* ← argmin_{g ∈ G(e)} load[g]  // 最少负载实例
    10.    actRep[e] ← P(e, g*)
    11.    load[g*] ← load[g*] + 1

    // Map token routing to physical replicas
    12. for i = 1 to T do
    13.     for j = 1 to k do
    14.         O(i,j) ← actRep[L(i,j)]

    // Synchronization-free: 每个 MoE instance 独立运行相同 AEBS
    // 产生相同 O(i,j)，通过确定性算法保证一致性
    ```

  - **SLO-Aware Scaling 算法伪代码（Algorithm 2）**：

    ```
    Algorithm 2: Fine-Grained, SLO-Aware Resource Scaling
    Input:
      n_max: upper bound of instance sizes
      n_e^min: lower bound of MoE instance sizes (= ⌈E/C⌉)
      B_max: upper bound of batch sizes (GPU memory budget)
    Output: (n_a*, n_e*, B*): optimal configuration

    1. opt ← ⊥; J* ← ∞
    2. for (n_a, n_e) ∈ {1,...,n_max} × {n_e^min,...,n_max} do
    3.     B* ← solve B = λ · TPOT(B, n_a, n_e, S_ctx) via binary search in [1, B_max]
    4.     if B* == ⊥ then continue  // 无可行解
    5.     T ← TPOT(B*, n_a, n_e, S_ctx)  // Eq. (1)
    6.     if T > SLO or not MemoryFeasible then continue
    7.     if n_a + n_e < J* then
    8.         opt ← (n_a, n_e, B*); J* ← n_a + n_e
    9. return opt
    ```

  - **TPOT 性能模型（Eq. 1）张量公式**：

    ```
    TPOT = Σ_{ℓ=1}^{L} [T_attn^(ℓ) + T_moe^(ℓ) + T_comm^(ℓ)]

    T_attn^(ℓ) = max(c_a^(ℓ), α^(ℓ)·b + c_kv^(ℓ)·b·S_ctx)
      // Roofline: memory-bound plateau vs computation+KV-cache
      // b = B/n_a (per-instance batch), S_ctx = avg context length

    T_moe^(ℓ) = β^(ℓ) · a_max^(ℓ)(n_e, B) + c_e^(ℓ)
      // Linear dependence on max activated expert count
      // a_max estimated via Monte Carlo from recent trace

    T_comm^(ℓ) = profiled cost of two-phase communication
    ```

  - **a_max Theoretical Bound（Appendix A, Eq. 5）**：

    ```
    Uniform activation: p_e = K/E
    E[a_g] ≤ C · [1 - (1 - K/E)^B]  // expected activated experts per instance
    ā_max = max_g E[a_g]              // bottleneck instance

    Tail bound (Bernstein + union):
    a_max ≤ min(C, ā_max + sqrt(2·ā_max·ln n_e)) + 1

    Two regimes:
    - Small B: ā_max << C, a_max grows with B → T_moe increases
    - Large B: ā_max → C, a_max plateaus → T_moe capped, T_attn dominates
    ```

  - **Monte Carlo â_max Estimator 使用原理**：
    1. 从最近的 activation trace 采样 B tokens（按 empirical distribution）
    2. 对每个 MoE layer ℓ，应用当前 AEBS 策略 + 候选配置 (n_e, B)
    3. 记录 â_max^(ℓ)(n_e, B) = 各 MoE instance 中 max distinct activated experts
    4. 构建 lookup table [n_e][B] → â_max^(ℓ)
    5. 周期性重建（如每 15 min）以跟踪 workload 变化
    6. 在 Algorithm 2 的 TPOT 评估中 constant-time 查表

  - **关键结果**：
    - AEBS vs EPLB: a_max 降低 2-5 experts (Fig. 13), MoE-layer latency 降低 up to 30% (Fig. 14)
    - AEBS overhead: batch=64 → <20μs, batch=4096 → <90μs (Fig. 15)
    - Full JANUS (2PC+EGate+AEBS): per-GPU throughput 4.7× SGLang, 2.2× MegaScale-Infer, 3.3× xDeepServe
    - Resource cost: 39% GPU-hour saving vs SGLang, 16% vs MegaScale-Infer (24h trace)
