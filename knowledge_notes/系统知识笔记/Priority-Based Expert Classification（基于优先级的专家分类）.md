## Priority-Based Expert Classification（基于优先级的专家分类）

术语是什么？
Priority-Based Expert Classification 是 Diff-MoE 提出的一种运行时 expert 分类机制。它为每个 expert 赋予一个在 [0, MaxP] 范围内的优先级分数，并根据分数将 experts 分为 globally hot（永久缓存）、locally hot（动态缓存）和 cold（按需加载后即驱逐）三类。分数在离线阶段初始化，在线推理中根据 expert 是否被当前 gating 激活以及是否在 GPU 缓存中动态增减。

从系统架构角度拆解术语：
优先级更新规则（Equation 1）以当前 MoE layer i 为单位执行：

```
if E_k^i ∈ A:                              // 被当前 gating 激活
    p_k^i = clip(p_k^i + Δ_inc, 0, MaxP)   //   +1
elif E_k^i ∉ A and E_k^i ∈ C:             // 未被激活但在 GPU 缓存中
    p_k^i = clip(p_k^i - Δ_dec_in, 0, MaxP) //   -0.4
else:                                       // 未被激活且不在 GPU
    p_k^i = clip(p_k^i - Δ_dec_out, 0, MaxP) //   -0.2
```

分类阈值与参数设定：
- **Globally Hot**：离线确定，p = MaxP = 2，固定不变。由微调阶段统计的 top-N 高频激活 experts 获得。
- **Locally Hot**：p ≥ threshold_hot = 1（即至少被激活 1 次的 non-global expert）。有资格进入 MPCi。
- **Cold**：p < 1。不会被缓存，用后即驱逐。

参数不对称设计的动机（Δ_inc > Δ_dec_in > Δ_dec_out）：
- Δ_inc = 1：使专家一次激活后即可晋升 locally hot，捕获短窗口内复用（1.13×–2.40× reactivation in 3–6 iterations）。
- Δ_dec_in = 0.4：加速驱逐占用 GPU 但不活跃的 experts，为近期激活的 locally hot 腾出 MPCi 空间。
- Δ_dec_out = 0.2：对未驻留 GPU 的 experts 缓慢降级，保留未来复用的可能性，避免过早遗忘偶尔激活的 experts。
- MaxP = 2 × threshold_hot = 2：限制分数上界，防止长期活跃 experts 分数过载，保证对新活跃 experts 的响应速度（Diff = MaxP - threshold_hot = 1 提供适中的 margin）。

术语一般如何实现？如何使用？
在 Diff-MoE 实现中，优先级管理通过 per-layer 的 PriorityScore 数据结构（Python dict 或 fixed-size array mapping expert index → float score）完成。每次 MoE layer 执行时，先由 gating network 返回 A，然后遍历所有 experts 按规则更新分数。MaxP=2, threshold_hot=1, Δ_inc=1 是论文的经验最优配置。与传统的 LFU（Least Frequently Used）不同，优先级分数对"被缓存但不活跃"的 experts 施加更大的惩罚，从而保护缓存空间用于近期活跃的 locally hot experts。

涉及论文标题：
- Diff-MoE: Efficient Batched MoE Inference with Priority-Driven Differential Expert Caching
