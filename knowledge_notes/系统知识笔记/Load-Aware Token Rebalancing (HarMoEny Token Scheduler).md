## Load-Aware Token Rebalancing (HarMoEny Token Scheduler)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Load-Aware Token Rebalancing 是 HarMoEny 中用于解决 MoE 多 GPU 推理中动态负载均衡问题的 online scheduling 算法（Algorithm 2）。在传统 expert parallelism 中，token 被 all-to-all dispatch 到持有指定 expert 的 GPU——若某些 expert 更热门（接收更多 token），其所在 GPU 计算负载远超其他 GPU，导致 GPU idle time 高达 82-86%。HarMoEny 的 rebalancing 在每个 batch 中动态将 token 从过载 GPU 重路由到欠载 GPU，目标是最小化各 GPU 的 token 处理量差异。

从系统架构角度拆解术语：

```
# HarMoEny Algorithm 2: Token Rebalancing (per MoE layer)
# Input: S_initial[g_from, expert, g_to] — naive token-to-GPU schedule
# Output: S — rebalanced schedule

REBALANCE(S_initial):
    S = S_initial
    t_avg = floor(S.sum() / |G|)  # target balanced tokens per GPU
    t_g = S.sum(dim=(0,1))        # current tokens per GPU

    while any(t_g > t_avg):
        g_max = Argmax(t_g)                          # most overloaded GPU
        g_from = Argmax(sum(S[:,:,g_max], dim=1))    # GPU sending most to g_max
        e_max = Argmax(S[g_from, :, g_max])          # expert sending most
        t_move = S[g_from, e_max, g_max]             # tokens available to move

        if t_move < q:                                # q = token threshold
            return S  # insufficient tokens to amortize expert transfer

        g_min = Argmin(t_g)                           # least loaded GPU
        if g_min == g_max or t_g[g_min] + q > t_avg:
            return S  # no feasible transfer

        t_s = min(t_move, t_avg - t_g[g_min])        # actual tokens to transfer
        S[g_from, e_max, g_max] -= t_s
        S[g_from, e_max, g_min] += t_s               # re-route to g_min
        t_g[g_max] -= t_s
        t_g[g_min] += t_s

    return S
```

关键设计：
- **Deterministic**: 所有 GPU 在 metadata exchange（Step 2）后拥有相同的 m_all → REBALANCE 输入相同 → 输出 schedule S 相同 → 无需额外同步
- **Overhead**: Scheduler overhead 为 30.8%（Switch128, 128 experts）/ 20.3%（Qwen, 60 experts）of layer latency
- **效果**: GPU idle time 从 82.6% → 2.6%（Switch128, 90% skew），total layer latency 从 289ms → 149.5ms（-48.3%）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HarMoEny 用 1115 行 PyTorch 代码实现。S 为 3D tensor（可视为 numpy array），rebalancing 是纯 CPU 侧的整数算术操作。Algorithm 2 的贪心复杂度 O(|G|²·|E|)，因 |G|≤8 和 |E|≤128，overhead 仅 20-31% of latency 且被整体延迟降低所补偿。token threshold q 由硬件规格静态确定（q > φ·d_type/(2β)，Section 4.4），无需 per-model tuning。

涉及论文标题：
- HarMoEny: Efficient Multi-GPU Inference of MoE Models
