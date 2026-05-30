## Expert Significance in MoE (ϕ·w Multi-Factor Importance)

术语是什么？
Expert Significance（专家重要性）是 MC-MoE 提出的衡量 MoE 中每个 expert 对模型输出贡献程度的多维评估指标。包含三维因子：(1) 访问频率 ϕᵢ = nᵢ/N：expert i 被 Router 选入 Top-K 的频率，反映通用性；(2) 激活权重和 wᵢ = Σσᵢʲ/N：expert i 的 routing score 累计值，反映每次激活的贡献强度——低频但高权 expert 对特定 token 可能极为关键；(3) 量化重构误差 εᵢⱼ = ‖F(θ) − F(θ[eᵢ→Q(eᵢ,j)])‖_F：expert i 被单独量化到 j-bit 后输出 activation 的 F-norm 偏差。三者以 ϕᵢᵅ·wᵢᵝ·εᵢⱼᵞ 组合构成 Integer Programming 损失函数核心项（MC-MoE 消融确定 α=β=1, γ=2），解决"均匀量化忽略重要 expert"和"仅看频率忽略低频高权 expert"两大缺陷。

从算法pipeline角度拆解术语：
```
// 离线: 在 FP16 MoE 模型上用校准数据 C4 做一次前向推理
for each token t in C4:
    routing = softmax(W_gate @ t)
    top_k = TopK(routing, k=2)
    for each selected expert e_i:
        ϕ_i++, w_i += routing[e_i]
ϕ_i /= total_tokens, w_i /= total_tokens

// 量化误差: 单独量化每个 expert 并测输出 F-norm
for each expert e_i, bit j ∈ {1,2,3}:
    ε_{i,j} = ||F(θ) - F(θ[e_i→Q(e_i,j)])||_F

// 综合重要性 = ϕ_i^α · w_i^β · ε_{i,j}^γ
```

术语一般如何实现？如何使用？
- 离线计算：仅需一次 FP16 前向推理（无梯度），计算开销极小
- 应用：(a) expert 位宽分配（MC-MoE PMQ）；(b) expert 静态剪枝（永久移除不重要的 expert）；(c) expert 卸载决策（低频低权 expert 卸载到 CPU/SSD）
- MC-MoE 的发现：ϕ 和 w 的分布可能不一致甚至相反（如 expert[1,3] 低权但高频），验证了多因素评估的必要性
- 局限：重要性依赖校准数据分布，分布外任务可能导致不同的重要性排序

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---
