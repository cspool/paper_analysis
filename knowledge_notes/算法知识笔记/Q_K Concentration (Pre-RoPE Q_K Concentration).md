## Q/K Concentration (Pre-RoPE Q/K Concentration)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Q/K Concentration 是 TriAttention（Mao et al., 2025）发现的 pre-RoPE 空间中的注意力现象：在 Rotary Position Embedding (RoPE) 旋转之前，大量 attention head 的 Query 和 Key 向量高度聚集于固定的非零中心周围。该聚集使用 Mean Resultant Length (MRL) $R = \|\mathbb{E}[q]\|/\mathbb{E}[\|q\|]$ 量化：R→1 表示完美聚集，R→0 表示均匀分散。在 Qwen3-8B 全 1152 个 attention head 中，约 90% 的 head 的 R > 0.95，证实聚集具有普遍性。

关键特性：(1) 跨位置稳定——同一 head 中不同位置 tokens 的 Q/K 向量围绕同一中心聚集；(2) 跨上下文稳定——Math/Coding/Chat 三种不同领域的 MRL 几乎相同（0.977-0.980）；(3) 跨架构普遍——GQA (Qwen3, Llama3) 和 MLA (GLM-4.7-Flash) 均存在，MLA 中 96.6% heads 的 R > 0.95；(4) 模型内在属性——校准数据质量（HTML vs Chat）和数据量（50K-960K tokens）几乎不影响聚集度量。

从算法pipeline角度拆解术语：

Q/K Concentration 使 attention logit 变为可预测的三角函数级数：
```
# 无聚集时（标准 RoPE Attention）：
logit(q, k) = Σ_f ‖q_f‖·‖k_f‖·cos(ω_f·Δ + (arg(q_f)-arg(k_f)))
# q_f, k_f 随 token 位置和内容变化 → 需要实时计算

# 有聚集时：q_f ≈ q̄_f, k_f ≈ k̄_f（近似为常数中心）
logit(Δ) ≈ Σ_f ‖q̄_f‖·‖k̄_f‖·cos(ω_f·Δ + φ̄_f)
# 其中 φ̄_f = arg(q̄_f) - arg(k̄_f) 是固定相位差
# 结果：attention 退化为仅依赖距离 Δ 的三角函数级数
```

术语一般如何实现？如何使用？

实现：离线校准阶段收集少量 tokens（50K 即可）的 pre-RoPE Q/K 向量。对每个 head 的每个频段 f，计算：(1) Q 中心 E[q_f]（复数均值）；(2) 期望 Q 范数 E[‖q_f‖]；(3) Mean Resultant Length R_f = ‖E[q_f]‖/E[‖q_f‖]。统计量在推理前离线计算一次，以 JSON/numpy 格式存储，推理时直接使用。

核心用途：(1) Q 中心用作"通用 proxy query"——通过三角函数级数预测任意位置的 key 会收到多少 attention（S_trig 评分）；(2) R_f 用作 S_trig 和 S_norm 的自适应加权因子；(3) 诊断工具——识别哪些 head 有强距离偏好。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---
