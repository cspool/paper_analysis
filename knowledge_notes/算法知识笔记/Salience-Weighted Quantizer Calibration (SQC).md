## Salience-Weighted Quantizer Calibration (SQC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SQC 是 SliM-LLM (ICML 2025) 提出的量化器校准方法，用于增强 group 内局部 salient 权重的表达能力。动机：即使 SBA 给高 salience group 分配高 bit-width，group 内部仍有约 1% 的离散 salient 元素与非 salient 元素共享同一套量化器参数 (scale/z)。传统量化器以所有元素的最小均方误差为目标，非 salient 元素（占~99%）主导优化方向，导致 salient 信息退化。SQC 通过 3-σ 规则筛选 salient 权重 (w_s = {w | w < μ-3σ ∪ w > μ+3σ})，引入 calibration 参数 τ 扩展量化器感知区间，对 scale 和 zero point 在 [1-λ, 1+λ] 区间搜索（λ=0.1, 50 candidates），优化加权目标 argmin_τ (||w_s - τs·Q(w_s, τs, τz)||² + ||w_us - τs·Q(w_us, τs, τz)||²)。关键设计：w_s 和 w_us 共享同一套 (τs, τz)，无需额外存储，保持推理效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: group权重 w^b (n×128), bit-width g_b
# 输出: 最优量化参数 Δ*, z*

# Step 1: 3-σ规则筛选salient权重
μ, σ = mean(w^b), std(w^b)
w_s = {w | w < μ-3σ or w > μ+3σ}  # 约占1%
w_us = w^b - w_s                    # 剩余~99%

# Step 2: 搜索最优τ
w_max, w_min = max(w^b), min(w^b)
best_loss = INF; λ = 0.1; n = 50
for τ in linspace(1-λ, 1+λ, 2n):
    Δ = τ * (w_max - w_min) / (2^g_b - 1)
    z = -⌊τ * w_min / Δ⌋
    ŵ_s = fakequant(w_s, g_b, Δ, z)
    ŵ_us = fakequant(w_us, g_b, Δ, z)
    loss = ||w_s - ŵ_s||²₂ + ||w_us - ŵ_us||²₂
    if loss < best_loss: best_loss, Δ*, z* = loss, Δ, z

# Step 3: 使用最优参数量化整个group
ŵ_q^b = fakequant(w^b, g_b, Δ*, z*)
```
效果：OPT-1.3B 某 channel 绝对误差从 0.0055（vanilla quantizer）降至 0.0039（SQC），salient 权重区域误差显著降低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SQC 作为 GPTQ 流程中 fakequant 步骤的替代实现：GPTQ 逐列量化时，每列调用 SQC（替代默认 MinMax 量化器）确定最优 τ、Δ、z。SQC 仅改变量化器参数搜索方式，不引入额外推理参数（τ* 融入最终 Δ* 和 z*）。SQC 可与 SBA 独立使用：SBA 提供 group 级混合精度优化，SQC 在统一精度场景也可独立提升量化质量。

涉及论文标题：
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

---
