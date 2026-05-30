## Dynamic Expert Skipping in MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic Expert Skipping（动态专家跳过）是在 MoE LLM 推理时，对每个 token 动态决定是否跳过 routing weight 较小的 expert，从而减少实际激活的 expert 数量以加速推理。与 Expert Pruning（永久删除 expert 权重）不同，Dynamic Skipping 不修改模型参数，仅在推理时基于 routing weight 比例做在线决策。

核心机制（top-2 场景，k=2）：对于每个 token x，Router 计算 routing weights w = {w_{e0}, w_{e1}}（w_{e0} ≥ w_{e1}）。如果次要 expert 的权重远小于主要 expert，即 w_{e1} < β · w_{e0}，则跳过 e1，仅使用 e0 计算输出。β 是每层独立的超参数，通过在校准集上前向推理并取该层所有 token 的 w_{e1}/w_{e0} 的中位数来确定（使跳过概率约 50%）。

理论依据（Sec. A.2）：假设不同 expert 输出向量间的 L2 距离 D 近似恒定，则跳过次要 expert(s) 的重建损失上界为：

$$\mathcal{L} \leq \frac{\sum_{m=i+1}^{k} w_m}{\sum_{m=1}^{k} w_m} \cdot D$$

在 top-2 特例下，条件简化为 w_{e1} ≤ β · w_{e0}（β = H/(D−H)），其中 H 为允许的损失上界。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// ============ Calibration: 确定每层 β ============
for each MoE_layer l:
    ratios = []
    for x in calibration_set:
        w = Softmax(Router_l(x))            // routing weights
        {e0, e1} = TopK(w, k=2)            // w_e0 ≥ w_e1
        ratios.append(w_e1 / w_e0)
    beta[l] = median(ratios)               // 每层独立中位数

// ============ Inference: 动态跳过 ============
for each token x in input_sequence:
    for each MoE_layer l:
        w = Softmax(Router_l(x))
        {e0, e1} = TopK(w, k=2)
        if w_e1 < beta[l] * w_e0:
            // 跳过 e1，仅使用 top-1 expert
            z = E_{e0}(x)                   // 未归一化，因仅一个 expert
        else:
            // 正常 top-2
            w̃_e0 = w_e0 / (w_e0 + w_e1)
            w̃_e1 = w_e1 / (w_e0 + w_e1)
            z = w̃_e0 * E_{e0}(x) + w̃_e1 * E_{e1}(x)
```

**推广到 top-k 场景（k > 2）**：
```
// 保留 top-i* expert，其中 i* = min i 满足：
// Σ_{m=i+1}^k w_m ≤ β · Σ_{m=1}^k w_m
i_star = 1
cumsum = w_e1 + w_e2 + ...   // 从第2大开始累加
total = w_e0 + w_e1 + ... + w_{k-1}
while i_star < k and cumsum > beta * total:
    cumsum -= w_{e_{i_star}}
    i_star += 1
// 使用 top-i_star expert 计算加权输出
```

Annotations:
- β 确定逻辑：取中位数使 calibration 集上约 50% token 跳过次要 expert
- 当 w_{e1} ≈ 0（次要 expert 几乎无贡献）时跳过收益最大，w_{e0} ≈ w_{e1}（两 expert 同等重要）时不跳过
- 跳过不影响内存使用（expert 权重仍在 GPU 显存中），但减少计算 FLOPs
- 与 Expert Pruning 正交叠加：r=6 pruning + skipping 可得 1.23× 加速 + 62.91 LM-eval；而 r=4 pruning alone 仅 59.57

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现方式**：在 HuggingFace Transformers 的 MoE layer forward 中插入条件判断，无需修改模型权重
- **β 校准**：对 Mixtral 8x7B + C4 校准集，第一层 β 值从 0.402 到最后一层 0.535 不等（见 Sec. A.7），各层差异显著，说明 expert 选择倾向在不同深度变化
- **适用场景**：(a) 与 Expert Pruning 组合使用以进一步加速；(b) 单独用于 8-expert 原模型：仅 skipping 可得 1.08× 加速，LM-eval 从 67.58 降至 66.37
- **性能 trade-off**（Tab. 5）：Mixtral 8x7B Instruct + r=4 pruning + skipping → 1.33× speedup, LM-eval 62.33（原模型 69.98 的 ~89%）
- **领域特化**（Tab. 8）：数学任务中 skipping 导致稍大性能下降（β 用 MATH 校准），但 r=6 pruning + skipping 仍优于 r=4 pruning alone
- **与动态剪枝的区别**：MC-MoE 的 ODP 也使用 w₁/w₀ < μ 跳过，但额外引入 token protection（保护 top 2% 重要 token）。Dynamic Expert Skipping 更简洁，无保护机制，与 permanent pruning 组合是核心贡献

涉及论文标题：
- Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models
