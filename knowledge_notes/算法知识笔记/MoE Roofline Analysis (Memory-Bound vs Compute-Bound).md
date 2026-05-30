## MoE Roofline Analysis (Memory-Bound vs Compute-Bound)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Roofline Analysis 是 JANUS 用来分析 MoE 层性能瓶颈的方法论。通过将单个 expert 的 arithmetic intensity（算术强度）I_e 与硬件 roofline 拐点 π/β 比较，判断 MoE 层在当前 workload 下是 memory-bound 还是 compute-bound。

单个 expert（含 2 个 GEMM）的 arithmetic intensity：
$$
I_e \approx \frac{2b \cdot d_h \cdot d_e}{2 \cdot d_e \cdot d_h} = b
$$

即 expert 的算术强度近似等于其 batch size b = B·k/n（B = layer-wise batch size, k = top-k, n = experts per GPU）。在在线 decode 场景下，b 通常远小于使 expert compute-bound 所需的阈值，因此 MoE 层是 memory-bound 的，延迟由需要从 HBM 加载的 expert weight 数量决定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Roofline Analysis for MoE Layer:

Given:
  π: peak FLOPs (e.g., H100: 989 TFLOPs/s, A100: 312 TFLOPs/s)
  β: memory bandwidth (e.g., H100: 3.35 TB/s, A100: 2.0 TB/s)
  Roofline ridge point: π/β

For single expert with batch size b:
  FLOPs = 2 · b · d_h · d_e  (2 GEMMs: gate+up projection, down projection)
  Bytes = 2 · d_h · d_e  (expert weight loading, assuming weights loaded once)
  Arithmetic Intensity I_e = FLOPs / Bytes ≈ b

For MoE layer with n experts per GPU, top-k routing:
  Per-expert batch size b = B · k / n  (expected, uniform routing)
  Minimum B for compute-bound: B ≥ π·n/(β·k)

Numerical Examples (JANUS):
  DeepSeek-V3 on H100 (n=256, k=8):
    B_min = 989 × 256 / (3.35 × 8) ≈ 9,400 tokens
  DeepSeek-V3 on A100 (n=256, k=8):
    B_min = 312 × 256 / (2.0 × 8) ≈ 5,000 tokens

  Online decode: per-instance B typically < 100
  → MoE layers are firmly MEMORY-BOUND in online serving

Implication for latency modeling (JANUS Eq. 1c):
  T_moe = β · a_max + c_e
  (linear in distinct activated expert count, not in token count)

Validation (JANUS Fig. 2 right, Fig. 3):
  Fix B=64, vary activated expert count → latency ~linear
  Fix B, vary activation distribution (uniform vs skewed) → nearly identical latency
  Vary B from 64 to 512 → latency changes marginally
```

JANUS TPOT Model (Eq. 1, integrating roofline):

```
TPOT = Σ_{ℓ=1}^{L} [T_attn^(ℓ) + T_moe^(ℓ) + T_comm^(ℓ)]

T_attn^(ℓ) = max(c_a^(ℓ), α^(ℓ)·b + c_kv^(ℓ)·b·S_ctx)
  // Attention follows roofline: memory-bound plateau (c_a) dominates at small b
  // then transitions to compute+KV-cache regime at large b

T_moe^(ℓ) = β^(ℓ) · a_max^(ℓ)(n_e, B) + c_e^(ℓ)
  // MoE is memory-bound in online serving → linear in a_max

T_comm^(ℓ) = profiled two-phase communication cost
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- HPC 标准方法：roofline model 由 Williams, Waterman, Patterson (2009) 提出
- JANUS 使用 roofline 指导：(1) TPOT 性能模型结构选择；(2) 解释为什么 AEBS minimize a_max 而非 token count；(3) 辨识 high-leverage batch size range [10, 100]
- 系数 (α, β, c_a, c_kv, c_e) 通过一次性 offline profiling 获得
- 可推广到任何需要在不同 batch size regime 下分析 MoE 层性能的方法

涉及论文标题：
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

**LatentMoE 的 Roofline 扩展（Section 2.1-2.2）**：

LatentMoE 在 JANUS 的基础上扩展了 roofline analysis 到两个维度：(1) Memory bandwidth analysis for latency-critical serving（考虑完整 memory traffic: weights + inputs + intermediate activations）；(2) Communication-computation ratio analysis for throughput-oriented serving。

Memory BW Roofline (LatentMoE, GB200, FP4):
- Ridge point: F/BW_HBM = 10 PFLOPs / 8 TB/s = 1250 FLOPs/byte
- Per-expert compute: C_exp = 2·t_exp·d·m
- Per-expert memory traffic: M_exp = d·m + t_exp·(d+m)
- Arithmetic intensity: I = 2·t_exp·d·m / [d·m + t_exp·(d+m)]
- For Qwen3-235B (d=4096, m=1536): t_exp ≥ 1418 for compute-bound
- Typical latency-critical: t_exp ~ hundreds → firmly memory-bound

Communication Roofline (LatentMoE, GB200 NVL72):
- All-to-All volume per GPU: M_comm = 2.5·(N/EP)·t_exp·d (mixed precision: FP4+BF16)
- Compute time: t_comp = 4·t_exp·d·m/F
- Communication time: t_comm = 5·t_exp·d/BW_NVL
- Ratio t_comm/t_comp = 5·F/(4·m·BW_NVL) ≈ 9 for Qwen3-235B
- → Communication is ~9× more expensive than computation in throughput regime

Key insight from LatentMoE roofline: d (hidden dimension) is the only parameter that affects BOTH memory BW (via d·m) and communication cost (via K·d), while m (intermediate dim) only affects memory BW. This motivates compressing d→ℓ as the primary optimization target.
