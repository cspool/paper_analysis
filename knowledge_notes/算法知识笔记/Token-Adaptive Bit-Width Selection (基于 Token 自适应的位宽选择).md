## Token-Adaptive Bit-Width Selection (基于 Token 自适应的位宽选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token-Adaptive Bit-Width Selection 是 D2MoE 提出的动态量化 router 训练方法，使每个 MoE expert 可以根据当前输入 token 的表示动态选择最合适的量化 bit-width。它基于 Observation：不同 token 对同一 expert 的量化敏感度不同（例如 expert 4 layer 1 量化到 INT1 在样本 1 上损失 0.5% 精度，在样本 10 上损失 0.2%）。

核心设计包括两个机制：
1. **Quantized Expert Capacity**：为每个 bit-width expert 设定 token 容量上限 c_k·T（如 D2MoE-V1 中 {0.3, 0.4, 0.3} 对应 INT2/3/4），超限 token 随机丢弃，防止训练时 bit-width router 坍塌到某一固定 bit-width
2. **Dynamic Bit-Width Selection Loss**：Loss = (1/T) Σ [CE(p(x), q(x)) + (α/L) Σ p_k^l(x) · b_k]，其中 CE 项保持精度（倾向高 bit-width），正则项 p_k^l(x)·b_k 促选低 bit-width（b_k 越小越好），α 平衡精度与效率

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-MoE-3.5B 推理时的 bit-width 选择流程为例：

```
=== Token-Adaptive Bit-Width Selection (Inference) ===
输入: 一个 transformer block 内 T 个 token 的 hidden states {h_t}，
      每 expert j 的 bit-width router R_j (已微调)，
      候选 bit-width {b_1=2, b_2=3, b_3=4}，容量 {c_1=0.3, c_2=0.4, c_3=0.3}
      Top-2 expert gating 结果: 每个 token t 选择 2 个 experts

for each expert j that has tokens routed to it:
    tokens_to_this_expert = {t | expert_j selected for token t}
    for each token t in tokens_to_this_expert:
        # Router 输出 K 个 bit-width 的 logits
        logits = R_j(h_t)  # 轻量化 MLP, 输入 hidden_dim, 输出 K
        probs = softmax(logits)  # p_k^l(x): 第 k 个 bit-width 的概率
        selected_bitwidth = top1(probs)
    
    # 容量约束：如果某 bit-width 超出 c_k·T，超限 token 跳过该 expert
    for each bitwidth k:
        if count(selected_bitwidth == k) > c_k * T:
            randomly drop excess tokens (skip expert computation)

输出: 每 token 每 expert 的 selected_bitwidth
      → 用于后续 MWQ 反量化 + GEMM
```

**训练时的 Dynamic Bit-Width Selection Loss**：
```
Loss = (1/T) Σ_t [CE(p_t, q_t) + (α/L) Σ_l Σ_k p_k^l(x_t) · b_k]

其中:
  p_t, q_t: D2MoE 模型和 FP16 基准模型的 logits (after LM head)
  p_k^l(x_t): token t 在 layer l 选中 bit-width k 的概率
  b_k: 第 k 个 bit-width 的数值（如 2, 3, 4）
  CE 项→ 保证精度（本质促选高 bit-width）
  正则项→ 促选低 bit-width（概率分配越小越好）
  α 控制精度-效率权衡
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
D2MoE 中，Bit-Width Router 是一个轻量化 MLP（参数占比 <0.5%），放置在每个 expert 之前。使用 C4 通用数据集（2048 random 2048-token segments）微调，batch_size=64。LLaMA-MoE-3.5B 微调耗时 ~2 小时（2×A6000），Mixtral 8×7B 耗时 ~4 小时。运行时额外开销：计算 <0.28%，内存 <0.53%，延迟 <1.67%（主要是 router 中 softmax 操作）。

对比 EdgeMoE（离线 calibration 固定 bit-width）和 MC-MoE（固定 activation frequency 分配），Token-Adaptive Bit-Width Selection 可以随 token 动态调整，在相同精度下节省 33%-53% 峰值内存。

涉及论文标题：
- D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving
