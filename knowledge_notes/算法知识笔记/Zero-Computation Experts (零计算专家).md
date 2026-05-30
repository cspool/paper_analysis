## Zero-Computation Experts (零计算专家)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Zero-Computation Experts 是 LongCat-Flash / MoE++ 提出的一种动态计算资源分配机制。在标准 MoE 的 FFN expert pool 中额外引入 Z 个"零计算专家"（Zero-Computation Experts），其输出定义为恒等映射 `E_i(x_t) = x_t`（输入直接通过，不经过 FFN 计算）。Router 从 N+Z 个 experts 中选 top-K 个，如果 token 选中 zero-computation expert，则该 expert 不引入额外 FLOPs。这样，模型可根据 token 的上下文重要性自适应分配计算资源：简单 token（如标点、虚词、冠词）更多激活 zero-computation experts 节省计算，困难 token（如语义关键词）更多激活 FFN experts 获取更强表达能力。

公式：$$\begin{aligned} \operatorname{MoE}(x_t) &= \sum_{i=1}^{N+Z} g_i E_i(x_t), \\ E_i(x_t) &= \begin{cases} \operatorname{FFN}_i(x_t), & 1 \leq i \leq N \\ x_t, & N < i \leq N+Z \end{cases} \end{aligned}$$ 其中 $g_i$ 为 router 输出（softmax + top-K selection），K 为每 token 选中的总 expert 数，实际激活的 FFN expert 数在 [0, K] 之间动态变化。

LongCat-Flash 配置：N=512 FFN experts, Z=256 zero-computation experts, K=12, 期望 FFN 激活数 $K_e=8$，实际激活参数范围 18.6B-31.3B（平均 27B）。训练后观察：平均 FFN 激活数在 20B tokens 后收敛到 ~8（波动 <1%），而标准差持续在 ~3，表明 token 间计算资源分配存在显著差异。浅层（Layer 1）中 function words、标点、数字持续获得较少计算资源；深层（Layer 28）资源分配更动态。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Zero-Computation Experts Forward Pass (per token, per MoE layer)

输入: x_t [d_model]  # MoE layer input
参数: FFN_experts = [FFN_0, ..., FFN_{N-1}]  # N 个标准 FFN experts
      zero_experts = [Identity, ..., Identity]  # Z 个零计算 experts (恒等映射)
      router: Linear(d_model, N+Z)
      expert_bias: [N+Z]  # PID-controlled bias

# Step 1: Router 计算
logits = router(x_t)  # [N+Z]
probs = softmax(logits + expert_bias)  # 加上 expert bias 后 softmax

# Step 2: Top-K 选择
scores, indices = topk(probs, k=K)  # 从 N+Z 个中选 K 个

# Step 3: Expert 聚合
output = zeros_like(x_t)
for g, idx in zip(scores, indices):
    if idx < N:
        output += g * FFN_experts[idx](x_t)  # FFN expert: 实际 FLOPs
    else:
        output += g * x_t  # Zero-comp expert: 零 FLOPs (identity)

# Step 4: PID Bias Update (训练时, 仅更新 FFN experts)
# Δb_i = μ * (K_e/K * 1/N - T_i/(K*T_all))  for i in [0, N-1]
# b_i = b_i + Δb_i

输出: output [d_model]
```

Forward pass 的计算复杂度由实际激活的 FFN expert 数量决定，而非固定的 K。Training loss 实验显示：动态激活 4.2B-7.0B 参数（平均 6B）的 zero-expert 变体，validation loss 持续低于固定 top-k=8（固定 6B）的 baseline（Figure 3a）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. **Zero-comp expert 不参与任何矩阵乘法**：在 MoE GEMM kernel 中直接跳过，返回输入引用。在 Distributed EP 场景下，zero-comp expert 的 dispatch/combine 通信也可省略（DeepEP 修改支持）。
2. **PID Controller 控制平均负载**：仅对 FFN experts 更新 bias（Eq. 2），zero-comp experts 的 bias 固定为零。PID 确保 $K_e/F$ 比例的 token 流向 FFN experts。在 LongCat-Flash 中，$K_e=8, F=K=12$，意味着平均 8/12=66.7% 的选中 expert 为 FFN expert。
3. **Load Balance 适应**：在 device-level load balance loss 中，将所有 zero-comp experts 划入一个单独的 group (D+1)，其 coefficient 保证 loss 收敛时 FFN:zero 比例为 $K_e:(K-K_e)$。
4. **Kernel 集成**：MoE permute/unpermute kernels 需集成 zero-comp expert 处理逻辑——识别选中 zero-comp expert 的 token，对这些 token 跳过通信/expert GEMM，直接累加 gating weight × input。

与相关工作对比：AdaMoE [Zeng et al., 2024] 提出 "null experts"，与 MoE++ [Jin et al., 2024] 同期独立提出类似概念。LongCat-Flash 在此基础上增加了 PID 控制器的精细计算预算控制和 EP-aware load balance。

涉及论文标题：
- LongCat-Flash Technical Report
- MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts
- AdaMoE: Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models
