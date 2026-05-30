## a_max (Maximum Activated Expert Count)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

a_max 是 JANUS 论文定义的核心 MoE 性能度量，表示一次 MoE layer forward pass 中，所有 MoE instance (GPU) 上 distinct activated expert 数量的最大值：

$$
a_{\max} = \max_{i \in \{1,\dots,n_e\}} a_i
$$

其中 a_i 是 MoE instance i 在本次 layer forward 中被分配到的 distinct expert 数量（注意：是 distinct expert count，而非 token count）。JANUS 的核心发现是：在在线 decode 场景下，MoE 层是 memory-bound 的，其延迟主要由 a_max 决定（即最慢的 instance 决定整层延迟），而非 total token count 或 routing probabilities。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

a_max 与 MoE layer latency 的关系 (JANUS Eq. 1c, Roofline 推导):

```
MoE layer latency model:
  T_moe = β · a_max + c_e

Roofline analysis:
  Expert arithmetic intensity: I_e ≈ 2b·d_h·d_e / 2d_e·d_h = b
  (b = per-expert batch size, d_h = hidden dim, d_e = expert intermediate dim)

  Compute-bound condition: I_e ≥ π/β
  For H100 (π=989 TFLOPs, β=3.35 TB/s):
    B ≥ π·n/(β·k) ≈ 989×256/(3.35×8) ≈ 9.4k tokens per layer
  For A100 (π=312 TFLOPs, β=2.0 TB/s):
    B ≥ 312×256/(2.0×8) ≈ 5k tokens

  Online decode: per-instance batch size typically < 100
  → MoE layers are MEMORY-BOUND in online serving
  → Latency ∝ number of expert weights to load = ∝ distinct activated experts

  Therefore: T_moe ∝ a_max (not ∝ total_token_count)
```

a_max 的 theoretical bound (JANUS Appendix A, balls-into-bins model):
```
Uniform activation: p_e = K/E (K=top-k, E=total experts)
Expected activated experts per instance (容量C的instance):
  E[a_g] ≤ C · [1 - (1 - K/E)^B]

Bottleneck instance: ā_max = max_g E[a_g]

Tail bound (Bernstein + union bound over n_e instances):
  a_max ≤ min(C, ā_max + sqrt(2·ā_max·ln n_e)) + 1

Three regimes (Fig. 17):
  Sparse (B ≲ 10): â_max ≤ 4, insensitive to placement
  High-Leverage (B ∈ [10,100]): steepest slope, 30-60% of C
    → Online decode operates here → placement + scheduling matter most
  Saturation (B ≥ 100): â_max plateaus near min(C, E/n_e)
    → Structural ceiling, no scheduling policy can push below it
```

a_max Monte Carlo estimation:
```
For each candidate (n_e, B):
  Sample B tokens from recent activation trace
  Apply AEBS scheduling strategy
  Record â_max = max_i(distinct experts on instance i)
  Build lookup table â_max^(ℓ)(n_e, B) for each MoE layer ℓ
Rebuild periodically (e.g., every 15 min)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- a_max 是分析指标，不在系统中直接显式计算
- Monte Carlo estimator 用于 JANUS 的 SLO-aware scaling 决策（Algorithm 2 TPOT evaluation）
- AEBS scheduling 的目标函数是 minimize a_max（greedy heuristic）
- a_max bound 公式用于快速 pruning 明显 infeasible 的 (n_e, B) 配置
- 可以推广到任何 expert-count-balanced 而非 token-count-balanced 的 MoE 系统

涉及论文标题：
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference
