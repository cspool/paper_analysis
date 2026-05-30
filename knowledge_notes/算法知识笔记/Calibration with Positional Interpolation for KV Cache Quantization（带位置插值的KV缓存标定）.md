## Calibration with Positional Interpolation for KV Cache Quantization（带位置插值的KV缓存标定）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Calibration with Positional Interpolation 是 PM-KVQ 提出的 KV Cache 量化标定策略，解决长 CoT 场景下短标定数据无法捕获长上下文数据分布的问题。核心思路：在标定阶段对 RoPE 的位置索引 m 施加缩放因子 s（即 m → s×m），将短序列（如 2048 tokens）的 RoPE 有效扩展到长上下文（如 8192 tokens），使标定过程中计算出的通道重参数化因子 λ_i 能够覆盖 RoPE 低频通道的完整周期分布。这在不增加标定计算和内存开销的前提（避免了 O(N²) 注意力复杂度）下，使短标定数据能够近似长上下文数据分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**问题背景**：RoPE 低频通道的周期可达 ~54K tokens（DeepSeek-R1-Distill-Qwen-7B, θ_{d/2} = θ^{-1} ≈ 1/10000^{2/d·d/2} = 1/10000）, 标定数据仅 512~2048 tokens → 低频通道仅覆盖正弦波的很小一段 → λ_i 标定不准确 → 通道重参数化失效（outlier 未正确迁移到 Query）。

**标定流程**：

```
// === 传统标定（短上下文问题） ===
calibration_len = 2048
data = load_calibration_data(calibration_len)  // RedPajama arXiv subset

// RoPE: θ_i = θ^{-2i/d}
for each channel i in [0..d-1]:
    freq_i = theta ** (-2 * i / d)  // i 接近 d/2 时频率极低（周期 > 54K tokens）
    pos[m] = m * freq_i  // 仅 2048 个位置 → 低频通道分布不完整
    K_rope[m,i] = K[m,i] * cos(pos[m]) - K[m,i+d/2] * sin(pos[m])

// 通道重参数化因子 λ_i = (max_m K_rope[m,i])^α
// 在 2048 tokens 上: max_m 仅覆盖正弦波约 1/26 周期 → λ_i 不准确

// === PM-KVQ标定（带位置插值） ===
s = 4  // 位置缩放因子
calibration_len = 2048
effective_len = s * calibration_len = 8192  // 嵌入8192上下文信息

for each channel i in [0..d-1]:
    // 关键修改：位置索引乘以缩放因子
    pos_scaled[m] = s * m * freq_i  // m=0..2047 → s*m=0,4,8,...,8188
    K_rope[m,i] = K[m,i] * cos(pos_scaled[m]) - K[m,i+d/2] * sin(pos_scaled[m])

// 通道重参数化因子更准确：
// λ_i(s=4) = (max_{m∈[0,2047]} |K_rope[m,i] at scaled positions|)^α
// 近似覆盖了 8192 tokens 的有效位置范围
// α 通过网格搜索在 [0,1] 区间寻优 (grid_size=20), 最小化自注意力重建损失

// 标定完成后得到的 Λ = diag(λ_i)
// 推理时通道重参数化: P = (Q·Λ)·Q((K·Λ^{-1})^T)
```

**消融实验结果**（DeepSeek-LLaMA-8B on AIME-2024-I）：
- Calibration len=2048, s=1 (no PI): pass@1=46.67%, Voting=60.00%
- Calibration len=2048, s=4 (8192 effective): pass@1=48.33%, Voting=60.00% (+1.66%)
- Calibration len=2048, s=16 (32768 effective): pass@1=46.67%, Voting=53.33% (过插值退化)
- Calibration len=8192, s=1 (真长上下文): pass@1=48.33%, Voting=60.00%

s=4 达到与真长上下文标定相同的效果，s=16 则过插值导致性能退化。

术语一般如何实现？如何使用？

实现关键：(1) s 值需根据目标上下文长度选择——论文对 32K 目标使用 s=4（8192 effective）即可；(2) 位置插值仅在标定阶段修改 RoPE，推理时恢复正常 RoPE；(3) α 网格搜索需要最小化 self-attention 算子的重建损失（非端到端 loss）；(4) 该技术独立于量化方式，可与任何基于通道重参数化的 KV Cache 量化方案配合。

适用场景：所有使用 RoPE 的长上下文 LLM 进行 KV Cache 量化标定，尤其在标定数据长度受限于 O(N²) 注意力复杂度时。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---
