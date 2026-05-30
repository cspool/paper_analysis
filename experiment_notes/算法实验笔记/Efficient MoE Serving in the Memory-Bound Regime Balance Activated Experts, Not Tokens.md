## Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 METRO（Minimum Expert Token ROuting），一种面向 memory-bound 状态下 Expert Parallelism (EP) MoE serving 的 token-routing 算法。核心思想：在 memory-bound 的 decode 阶段，GPU 的 MoE 层延迟由 "activated expert replicas 数量" 而非 "处理 token 数量" 决定；而现有 token-balancing 路由算法（如 EPLB）会 inflate activated experts，导致 decode 性能退化。METRO 将 token routing 建模为 MIN-EXP-ROUTING ILP 问题——给定 N 个 expert、G 个 GPU、expert–GPU placement matrix A、每个 expert 在当前 batch 中的 token 数 T[1..N]，目标是最小化各 GPU 上 activated expert 数量的最大值 λ。Lemma 1 证明任何可行解可约化为"每个 expert 的所有 token 仅路由到一个 replica"。由于 ILP 最优解（二分搜索 + bipartite matching / Dinic max-flow）在 CPU 上的计算开销达 FFN 层时间的 31.4%–41.3%，GPU 上达 86.4%–103.8%，METRO 提出 GPU-native 贪心近似算法：并行遍历每个有 token 的 expert i，获取候选 GPU 集合 G_i，按 GPU ID 全序加锁避免死锁，选择当前 activated expert 计数器 L[g] 最小的 GPU g* 进行分配，复杂度 O(|A|)。同时 METRO 将 EP 的 all-to-all dispatch 替换为 all-gather dispatch，使每个 GPU 获得全局 top-k 信息作为算法输入。

  实验比较：(a) METRO vs EPLB token routing 在真实系统（vLLM, 8×A100）上的 decode latency (TPOT) 和 total token throughput（prefill+decode co-deployed）；(b) METRO vs EPLB 在专有模拟器（8-16×B200）上的相同指标；(c) METRO vs Optimal（二分搜索+max-flow）的 routing quality（max activated experts per GPU per decode batch）；(d) METRO 的 latency breakdown（greedy algorithm overhead, top-k overhead, communication overhead vs FFN reduction）；(e) decode throughput-latency Pareto-optimality 分析：变 batch size 和 TP/EP 配置下的 Pareto 前沿比较，METRO 在固定 SLO 下实现 1.98×–4.11× decode throughput 提升。

- 硬件平台是什么，配置是什么。
  真实系统：Google Cloud a2-highgpu-8g VM，8×NVIDIA A100 40GB GPU，600 GB/s NVLink（全部 GPU 在同一 NVLink domain）。模拟器：专有工业级 multi-GPU 性能模拟器（proprietary analytical roofline model），建模 8×B200 192GB（Qwen3-235B 实验）和 16×B200 192GB（DeepSeek-V3 实验），900 GB/s NVLink。模拟器支持 register、shared memory、compute、L2、HBM、network 多级硬件建模，含 TP/EP 并行策略映射，account for workload imbalance by estimating runtime based on the most bottlenecked GPU。

- 模型是什么。数据集和bench分别是什么。
  真实系统模型：Qwen3-30B-A3B（128 experts, fine-grained MoE）。模拟器模型：Qwen3-235B-A22B（128 experts）、DeepSeek-V3-671B（256 experts）。数据集：(a) 真实系统——InstructCoder（~114K code-editing instruction triplets, decode-heavy）、NuminaMath-1.5（~900K competition-level math problems with chain-of-thought, decode-heavy）；(b) 模拟器——Humaneval（164 Python programming problems, decode-heavy）、GSM8K（8,500 grade-school math word problems, prefill-heavy）。Context length：8K（真实系统），1K input + 2K output（模拟器）。Metrics：Total Token Throughput（prefill+decode co-deployed）、Decode Latency (TPOT)、Max activated experts per GPU per decode batch。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文作者包含 NVIDIA（N. Oswald, Q. Huang, H. Linsenmaier, C. Mei, R. Zhao, R. Borkar, B. D. Rouhani, D. Nellans, R. Krashinsky）和 Yale/Princeton/CMU 学术机构。**论文未明确提供开源代码仓库**（截至查询时未找到公开 GitHub 链接）。已集成 vLLM 的实现细节在 §V 中描述。算法核心流程如下：

  **=== MIN-EXP-ROUTING ILP 问题形式化 ===**
  ```
  输入: N experts, G GPUs, placement matrix A in {0,1}^{N×G}, token counts T[1..N]
  决策变量:
    x_{i,g} >= 0  : expert i 在 GPU g 上处理的 token 数
    y_{i,g} in {0,1}: expert i 是否在 GPU g 上被激活
    lambda >= 0     : 所有 GPU 中最大 activated experts 数
  目标: min lambda
  约束:
    (1) Sum_{i=1..N} y_{i,g} <= lambda, for all g   # 每 GPU activated experts <= lambda
    (2) Sum_{g=1..G} x_{i,g} = T[i], for all i      # 所有 token 必须被路由
    (3) x_{i,g} = y_{i,g} = 0 if A_{i,g}=0          # 路由遵守 placement matrix
    (4) x_{i,g} <= T[i] * y_{i,g}                   # token 仅路由到 activated expert
  ```

  **Lemma 1**: 任何可行解要么已将所有 token 路由到各 expert 的单个 replica，要么可映射到满足此性质的解而不增加目标值。

  **=== METRO 贪心近似算法（CUDA kernel, 单 SM）===**
  ```
  输入: N, G, A in {0,1}^{N×G}, T[1..N]
  输出: y_{i,g}
  初始化: L[g] <- 0, lock l_g for each g=1..G; y_{i,g} <- 0
  For each expert i = 1 to N in parallel:   // 并行度 = N (128-256)
      if T[i] > 0:
          G_i <- {g | A_{i,g} = 1}          // 查 placement matrix 获取候选 GPU
          acquire all locks {l_g | g in G_i} in GPU ID total order  // 全序加锁防死锁
          g* <- argmin_{g in G_i} L[g]      // 选 activated experts 最少的 GPU
          y_{i,g*} <- 1; L[g*] <- L[g*] + 1
          release all locks {l_g | g in G_i}
  lambda = max_g Sum_i y_{i,g}
  x_{i,g} = T[i] if y_{i,g}=1 else 0        // Lemma 1: 所有 token -> 单个 replica
  ```
  复杂度: O(|A|) vs 最优解 O((N+G)^2 * (|A|/G+N+G) * log(|A|/G))

  **=== METRO all-gather dispatch 流程 ===**
  ```
  For each MoE layer:
    1. All-gather: 每个 GPU 将本地 tokens 广播到所有 GPU
       (替换传统 all-to-all dispatch，获得全局 token 集合)
    2. Top-K: 每个 GPU 在全局 token 集合上独立计算 top-k
       -> 构建全局 T[1..N]（每个 expert 的总 token 数）
    3. METRO Routing: 每个 GPU 独立执行 Algorithm 1
       -> 确定每个 expert 在哪个 GPU 上激活
    4. FFN Compute: 每个 GPU 仅计算分配给自己的 expert FFN
    5. All-to-all Combine: 将 expert 输出 embedding 返回原 GPU
  ```

  关键性能数据：(a) Algorithm 1 开销最多 26us (1.5x replication)，但 FFN 时间减少最多 81us；(b) all-gather redundant top-k 额外开销 <3us (<1% 层时间)；(c) all-gather 通信在 memory-bound 小 batch 下（2MB/GPU on 600 GB/s NVLink ~ 3us）远低于 NCCL launch 固定开销（~100us）；(d) routing quality 在最优解的 10.9% 以内，比 EPLB 降低 up to 42.3% activated experts；(e) decode latency 降低 11%-22%, total token throughput 提升 3%-21%（co-deployed prefill+decode）。
