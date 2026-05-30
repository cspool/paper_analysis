## Information Calibration Quantization (ICQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ICQ 是 IR-QLoRA 提出的基于信息熵最大化的量化校准技术。传统的对称量化（如 QLoRA 的 NormalFloat）零点固定为零，量化权重熵未经优化，导致与原始权重的互信息（mutual information）不足。ICQ 引入可学习的 calibration constant τ，将量化公式从 ŵ = NFk(w/s) 改为 ŵ = NFk((w-τ)/s)，并通过搜索最大化量化权重信息熵 H(ŵ) = -Σ P(q_i) log₂ P(q_i) 来确定最优 τ*。由于 PTQ 中原始权重 w 固定，最大互信息等价于最大熵。ICQ 能有效提升量化权重的信息表示能力，4-bit LLaMA-7B 的权重熵从 3.67 提升到 3.74，无需微调即可在 MMLU 上带来 0.5% 的精度提升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
ICQ 对每个权重块的搜索过程：
```
Input: Block weight w ∈ ℝ^64, λ=0.1, n=100, σ=1
Output: τ₁^FP8, τ₂^FP16

1. τ₀ = quantile_0.5(w)                     # 用中位数初始化（正态分布对称轴）
2. H* = 0
3. for τ in Linspace(τ₀ - λσ, τ₀ + λσ, 2n+1):  # 201 个候选值均匀采样
4.     w̃ = w - τ
5.     s = absmax(w̃)                           # scale factor
6.     ŵ = NF4(w̃ / s)                          # NormalFloat 4-bit 量化
7.     Calculate P(q_i) = count(ŵ == q_i) / len(w̃) for i=0..15
8.     H = -Σ_{i=0}^{15} P(q_i) · log₂ P(q_i)
9.     if H > H*:
10.        τ* = τ, H* = H
11. τ₁^FP8 = FP8(τ* / absmax(τ*))              # double-quantize τ*
12. τ₂^FP16 = absmax(τ*)
```
τ₀ 使用中位数而非均值初始化：正态分布在对称轴附近密度最高，中位数使更多数据落入量化 bin 的中心区域。搜索范围 [τ₀-0.1σ, τ₀+0.1σ] 覆盖 ~95% 情况下的最优解。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ICQ 在 PTQ 阶段作为一次性预处理步骤执行，结果可缓存用于后续多次微调。对每个量化 block（默认 block_size=64）独立执行搜索，τ* 与 scale factor s 一起执行 double quantization 以控制存储开销（仅增加约 2% 参数）。ICQ 兼容 NormalFloat 和 Integer 量化：在 Integer 量化中，τ 相当于 zero point，可与原有 zero point 合并实现零开销。在 IR-QLoRA 代码中，ICQ 集成在模型加载后的 `prepare_model_for_kbit_training` 阶段。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention

---
