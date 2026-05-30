## Profiling-guided Memory Planning（Profiling引导的内存规划）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Profiling-guided Memory Planning 是 SwapMoE 的离线优化方法，在给定 hardware 和 memory budget 下自动搜索最优 Virtual Experts 配置。配置 $config = \{frequency, [k_1, ..., k_L]\}$（每层 VE 大小 + 更新频率）。搜索空间极大（12 层 SwitchT-16 为 $12^{16}$），使用 Genetic Algorithm 在 profiling-based performance models 上搜索。三个 model：(1) $E_{memory}(config)$：per-expert memory profiling 线性累加；(2) $E_{latency}(config)$：per-expert latency + IO loading time 累加；(3) $E_{accuracy}(config)$：小型 DNN（2 FC + ReLU）从 profiling 数据学习 config→accuracy 映射。优化：maximize accuracy + minimize latency, s.t. memory ≤ budget。

从系统架构角度拆解术语：

```
Phase 1: Fine-grained Profiling (~20 min, 一次性)
  For each MoE layer:
    measure: expert_param_size, expert_inference_time, expert_loading_time
    measure: IO bandwidth (GPU↔CPU PCIe, CPU↔SSD)
  Collect labeled profiling samples from deployment environment
  Train E_accuracy DNN: config_features → accuracy (converges in minutes)

Phase 2: Genetic Algorithm Search (~5 sec, per budget)
  population = [random_config() for _ in range(P)]
  For generation in range(G):
    # Mutation: randomly change one parameter
    # Crossover: swap or average two configs
    # Fitness: E_accuracy - λ*E_latency (if E_memory ≤ budget, else -inf)
    # Selection: tournament select top performers
  return config*
```

发现遗传算法能自动学到：中间 MoE layers 分配更多 VE（对逻辑处理影响更大），浅层和深层相对少。这一 insight 验证了搜索算法的有效性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) E_accuracy DNN 为 2-layer MLP + ReLU，训练数据通过 profiling dataset 上以随机配置运行 SwapMoE 收集，预测误差 <1%；(2) E_memory 和 E_latency 为确定性累加模型——将 per-expert profiling 数据按配置中的 k_l 求和；(3) Genetic algorithm 参数论文未完全公开（population size、generations），但总搜索时间仅 ~5 秒表明规模适中；(4) 搜索出的 config 在运行时固定使用——VE size 和 frequency 不变，但具体哪些 experts 进入 VE 由 runtime importance scores 动态决定。适用场景：任何需要在部署前确定最优资源配置的 memory-constrained MoE serving 系统。

涉及论文标题：
- SwapMoE: Serving Off-the-shelf MoE-based Large Language Models with Tunable Memory Budget
