## Residual Vector Quantization (RVQ) in PTQ

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Residual Vector Quantization (RVQ，残差向量量化) 是一种多阶段向量量化方法，最早由 Juang & Gray (1982) 在语音编码领域提出。在 LLM PTQ 中，RVQ 用于将低比特 VQ 方法扩展到更高比特：给定一个目标向量 x 和目标总位宽 p，使用一组 q_i-bit 的码书逐残差量化——第一阶段用 q_0-bit 码书量化 x 得 δ_0，第二阶段用 q_1-bit 码书量化残差 (x - δ_0)/s_1 得 δ_1，依此类推，最终 Ŵ = Σ δ_i · s_i。QuIP# 使用 RVQ 将 2-bit E8P 扩展到 3-bit 和 4-bit：(a) 4-bit = 2× E8P 2-bit RVQ（ρ₁≈1.03, ρ₂≈3.45）；(b) 3-bit = E8P 2-bit（ρ≈0.98）+ E8 1-bit（范数≤2 的 E8 元素 + 15 个范数 4 元素，ρ≈2.04）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
QuIP# 中 4-bit RVQ 的量化流程（在 BlockLDLQ 内部，每块 8 维）：
```
# w ∈ R^8 (待量化的 8 维权重块)
# RVQ(w, p=4, q=[2,2])

# Stage 1: 2-bit E8P 量化, scale ρ₁
delta_1 = e8p_quantize(w / rho_1) * rho_1

# Stage 2: 量化残差
residual = (w - delta_1) / rho_2
delta_2 = e8p_quantize(residual) * rho_2

# 最终量化值
w_hat = delta_1 + delta_2

# 输出: 2 个 16-bit 码字 → 4 bits/weight (平均)
```
与标量量化（SQ）对比的优势：RVQ 的每个阶段使用高维 VQ（8D），保留了跨维度形状信息。SQ 在 4-bit 下退化为 1D 16-level 均匀量化，无法捕获多维分布的形状；RVQ 的 2-stage E8P 用 2×2-bit = 4-bit 实现多维球状码书，更匹配 RHT 变换后的高斯权重分布。论文提到更高级的多码书量化方法（如 Additive Quantization）可能进一步改进，但 RVQ "已足够"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
RVQ 的实现要点：(1) 各阶段缩放因子 ρ_i 通过最小化高斯→RVQ 量化误差确定（数值搜索），各模型略有不同（ρ 因 incoherence processing 不完全产生精确高斯）；(2) 推理时每个阶段独立解码——stage 1 E8P 码字 + stage 2 E8P 码字 → 两个 8D 向量加权求和 → 送入 MMA；(3) RVQ 的计算开销近似正比于阶段数，但所有阶段可融合在同一 kernel 内完成；(4) RVQ 不限于 E8P——任何低比特 VQ 码书均可作为 RVQ 的子阶段（如 GPTVQ 使用 2×2D VQ 做 4-bit 量化）。

涉及论文标题：
- QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

---
