## Draft KV Budget Selection（草稿KV预算选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Draft KV Budget K 是 MagicDec 中用于控制压缩 KV cache 大小的关键超参数，决定 draft model 可访问的历史 token 数量。K 的选择直接影响 SD 加速比的三要素：(1) draft cost T_D(B,K) — K 越小，draft attention 计算和 KV loading 越少；(2) token 接受率 α(K) — K 越大，draft 看到的上下文越多，接受率越高；(3) 若使用 dynamic KV selection (如 PQCache)，搜索 cost T_select(B,S,K) 也受 K 影响。

MagicDec 的选择框架：对于给定的 batch size B 和 sequence length S，计算不同 K 下的 (draft cost, acceptance rate) 曲线，通过公式 (4) 找到最小化 T_Avg^SD/T_T 的 K*。实操中：B=32 时 SnapKV self-speculation 需要 K≥512 才能达到 speedup > 1（Figure 5c），因为接受率 α 必须超过最小阈值 α_min 才能使 Ω(γ,α) > 1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Draft KV Budget 最优选择（MagicDec 公式 4 简化版）
def select_optimal_K(B, S, model, hardware, alpha_curve):
    best_speedup = 1.0
    for K in candidate_budgets:  # [256, 512, 1024, 2049, ...]
        T_D = measure_draft_cost(B, K)       # 压缩 KV 的 draft 时间
        T_V = measure_verify_cost(B, S)      # 完整 KV 的验证时间
        T_T = measure_target_cost(B, S)      # AR 解码时间
        alpha = alpha_curve[K]                # K 对应的接受率
        for gamma in [2..12]:
            omega = (1 - alpha^(gamma+1)) / (1 - alpha)
            # 公式 (2): speedup = Ω(γ,α) * T_T / (γ*T_D + T_V)
            speedup = omega * T_T / (gamma * T_D + T_V)
            if speedup > best_speedup:
                best_speedup, best_K, best_gamma = speedup, K, gamma
    return best_K, best_gamma, best_speedup
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

K 的选择需通过 offline profiling 预先测量不同 (B, S, K) 组合下的 T_D, T_V, T_T 和 α。MagicDec 论文中 StreamingLLM-based self-speculation 使用 K=256 和 K=512 两种 budget（Figure 6），SnapKV-based 使用 K=2049（Table 6）。大 batch + 长序列时大 K 更优（memory-bound 主导，大 K 提高接受率而 draft cost 增加可忽略）。Batch 中不同序列可使用不同 K（heterogeneous batch），MagicDec 根据各序列所需的最小接受率推荐"可接受 K 预算"（Figure 5c ticked budgets）。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
