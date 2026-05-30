## Adaptive Expert Skipping for MoE Inference

术语解释
Adaptive Expert Skipping 是一种在 MoE 推理时动态跳过冗余专家计算以加速推理的技术。不同于永久剪枝（permanent pruning），skipping 在每 token 级别按需决定是否跳过某些已激活专家。DiEP 论文提出基于 routing weight ratio 和 CKA similarity 的自适应跳过机制：对于 Top-2 routing 选择的专家 e₀（高权重）和 e₁（低权重），若 w_e1 < γ·w_e0，则跳过 e₁ 的 FFN 计算。γ = γ₁ × γ₂，其中 γ₁ 为 calibration data 中 ratio w_e1/w_e0 的中位数（per-layer），γ₂ 为两专家输出 CKA 相似度与层平均 CKA 相似度的比值。

术语是什么？
MoE 推理中每个 token 需要经过 top-k 个专家的 FFN 计算，但并非所有被激活的专家对最终输出的贡献均等。NAEE（Lu et al. 2024）发现许多 token 的次要专家贡献很小。Adaptive Expert Skipping 利用这一观察，在推理时动态做出 per-token 的 skip 决策：

1. Router 产生 routing weights w = {w_e0, w_e1}（假设 k=2）
2. 若 w_e1 < γ·w_e0，跳过 expert e1，仅用 e0 的输出
3. γ 分解为两部分：
   - γ₁ = median(w_e1/w_e0) over calibration data（per-layer），捕获 layer-specific routing 分布特性
   - γ₂ = ρ(y_e0, y_e1) / mean(ρ(y_ei, y_ej))（per-layer），捕获两个特定专家的输出相似度
4. γ 为 per-layer 常量（推理时不变），仅 skip 决策是动态的

DiEP 结果：Mixtral 8×7B 50% pruning + skipping: 1.28× speedup, 48% GPU memory reduction, 保留 ~92% 性能。

从系统架构角度拆解术语：
```
# Adaptive Expert Skipping in MoE Inference

# Pre-computation (calibration phase, done once per layer)
for layer l in MoE layers:
    samples = []
    for token in calibration_data:
        w0, w1 = top2(routing_weights[l][token])
        samples.append(w1 / w0)
    γ1[l] = median(samples)
    
    # CKA similarity
    cka_matrix[l] = compute_expert_pairwise_cka(layer l, calibration_data)
    for each expert pair (e0, e1):
        γ2[l][e0][e1] = cka_matrix[l][e0][e1] / mean(cka_matrix[l])
    
    γ[l] = γ1[l] × γ2[l]  # stored per-layer

# Inference (per token)
def moe_layer_forward(x, layer l):
    w = router[l](x)  # routing weights for all experts
    e0, e1 = top2_indices(w)
    w0, w1 = w[e0], w[e1]
    
    if w1 < γ[l] * w0:  # adaptive skip condition
        # Skip e1, only use e0
        output = w0 * FFN_e0(x)
    else:
        # Use both experts (standard MoE)
        output = w0 * FFN_e0(x) + w1 * FFN_e1(x)
    
    return output
```

关键设计选择：
- γ 是 per-layer 而非 per-expert-pair（简化实现）
- 论文描述 γ₂ 为 "ratio of CKA similarity ρ(y_e0,y_e1) to mean CKA similarity across all data samples in layer l"
- Skipping 仅影响 FFN 计算，不改变 router 行为

术语一般如何实现？如何使用？
- 无需训练：γ 参数在 calibration data 上一次性计算，推理时直接使用
- 与 pruning 正交：DiEP 中 pruning（permanent removal）+ skipping（dynamic）可叠加，50% pruning + skipping 获得 1.28× speedup（vs 1.26× pruning-only）
- 兼容性：可与量化、merging 等其他压缩方法组合
- 实现简单：在标准 MoE layer forward 中添加条件判断即可，无需 custom kernel 或框架修改
- 局限性：(1) speedup 受限于 skipping rate（主要节省次要专家的 FFN 计算）；(2) 对 top-1 routing 的 MoE 不适用；(3) per-token 条件分支可能影响 GPU 的 SIMT 效率

涉及论文标题：
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning
