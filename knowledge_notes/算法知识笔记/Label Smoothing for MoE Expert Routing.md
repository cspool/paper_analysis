## Label Smoothing for MoE Expert Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Label Smoothing for MoE Expert Routing 是 LEGO 在 LLM Judge 选择 expert 后应用的软性权重分配策略。传统做法：LLM 选中的 expert 权重=1，其他 expert 权重=0（硬路由），但这会导致：(1) 错误选择累积——若 LLM 判断错误，整个预测完全依赖错误 expert；(2) 训练不稳定——梯度仅流向被选中的 expert，其他 expert 无信号更新。LEGO 的 label smoothing 方案（Eq. 7）：选中 expert 的权重为 α ∈ (0,1)，其余 (K-1) 个 expert 平分剩余权重 (1-α)/(K-1)。这种软性分配使所有 expert 都接收到一定梯度信号，同时保留选中 expert 的主导地位。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 硬路由 (hard routing)
if LLM_chosen == k:
    weights = [0, ..., 0, 1, 0, ..., 0]    // one-hot, 仅 expert k 激活
else:
    // 其他 expert 无梯度信号

// Label Smoothing 软路由 (LEGO, Eq.7)
α = 0.8  // 选中 expert 的主导权重
chosen = LLM_choice
for k in 1..K:
    if k == chosen:
        ω(k) = α
    else:
        ω(k) = (1-α) / (K-1)              // 其余 expert 共享

X̂⁽ᵗ⁾ = Σ_k ω(k) · Decoder(h_i^k)         // Eq.8: 软性组合

// 梯度流到所有 expert：∂ℒ/∂θ^k ∝ ω(k)，所有 expert 都被更新
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 理论基础：Labels Smoothing（Müller et al. 2019）原本用于分类任务（将 one-hot label 平滑为 soft label 防止过拟合）。LEGO 将其应用于 MoE routing weights 的 slot
- α 的选择：α 接近 1 → 接近硬路由（更依赖 LLM 判断）；α 接近 1/K → 均匀路由（忽略 LLM 判断）。论文未明确给出 α 的具体值，由实验调参确定
- 作用机制：(a) 缓解 LLM 偶然判断错误的影响（错误 expert 仍有少量权重，不至于完全错误）；(b) 防止 expert collapse（仅部分 expert 持续被更新，其他 expert 停滞）；(c) 与 diversity loss 协同（diversity loss 促进 expert 分化，label smoothing 确保所有 expert 被更新）
- 与 top-k routing 的对比：LLM MoE 常用 top-2 routing（选 2 个 expert 各给部分权重）→ LEGO 的 smoothing 选择所有 expert（K=5），但主导 expert 权重远大于其他
- 局限：α 作为超参数需调优；α 过大则 smoothing 效果弱，α 过小则丢失 LLM Judge 的选择信号

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---
