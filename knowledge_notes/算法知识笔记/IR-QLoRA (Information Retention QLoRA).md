## IR-QLoRA (Information Retention QLoRA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IR-QLoRA 是一种面向 LoRA-finetuning 量化的精确 LLM 压缩框架，从统一的信息保留视角提出两个核心技术：(1) **ICQ (Information Calibration Quantization)**：通过最大化量化权重信息熵的统计校准量化，使量化后的 LLM 权重尽可能保留原始参数信息；(2) **IEC (Information Elastic Connection)**：通过参数自由的弹性连接增强 LoRA 的表征多样性，使 LoRA 能够直接利用原始输入信息。IR-QLoRA 在 LLaMA/LLaMA2 系列的 2-4 bit 量化下均实现显著精度提升，且仅增加 ~0.31-0.46% 的训练时间开销。IEC 的额外参数可完全合并入 LoRA 矩阵，无推理开销。框架兼容 NormalFloat 和 Integer 两种量化格式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
IR-QLoRA 的完整 forward pass（以 NF4 量化的 linear 层为例）：
```
# === 离线 PTQ 阶段（执行一次，结果可缓存）===
for each block in weights with blocksize B=64:
    τ₀ = median(w_block)                               # 初始化 calibration constant
    H* = 0
    for τ in Linspace(τ₀ - 0.1, τ₀ + 0.1, step=0.001): # 200 候选
        w' = w_block - τ
        ŵ = NF4(w' / absmax(w'))                       # NormalFloat 4-bit 量化
        H = -Σ P(q_i) · log₂(P(q_i))                   # 信息熵
        if H > H*: τ* = τ; H* = H                       # 选最大熵
    # Double-quantize τ* 和 scale factor s
    τ₁^FP8 = FP8(τ* / absmax(τ*)), τ₂^FP16 = absmax(τ*)
    s₁^FP8  = FP8(s / absmax(s)),  s₂^FP16  = absmax(s)

# === 推理/微调阶段 ===
# ICQ 量化前向
ŵ^FP16 = NF4((w - τ*) / absmax(w - τ*)) · s₁ · s₂ + τ₁ · τ₂
y'_ICQ = x · ŵ^FP16

# IEC 增强的 LoRA 前向
# U₁: 输入 x (h维) → 低秩中间表示 (r维)
x₁ = x · ℓ₁                                              # 标准 LoRA 左矩阵
x₂ = β₁ · (r/h) · mean_pool(x, segments=r)               # IEC: 输入分组平均
x' = x₁ + x₂                                              # 弹性连接融合
# U₂: 低秩中间表示 (r维) → 输出 (o维)
y₁ = x' · ℓ₂                                             # 标准 LoRA 右矩阵
y₂ = β₂ · repeat(x', o/r times)                           # IEC: 重复拼接
U_IEC(x) = y₁ + y₂

# 最终输出
y = y'_ICQ + α · U_IEC(x)
```
其中 IEC 参数 β₁, β₂ 可在推理时通过矩阵数学合并入 ℓ₁, ℓ₂ 以消除全部推理开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/htqin/ir-qlora。基于 QLoRA 官方代码（HuggingFace Transformers + PEFT）修改。ICQ 在模型的 `prepare_model_for_kbit_training` 阶段执行搜索，IEC 通过修改 PEFT 中 LoRA 层的 `forward` 函数实现。使用时在 QLoRA 或 QA-LoRA 的标准微调流程前插入 ICQ 搜索步骤即可。ICQ 搜索 [τ₀-0.1, τ₀+0.1] 区间 200 个候选值，选择信息熵最大的 τ*。β₁, β₂ 作为可训练参数与 LoRA 权重一起被 AdamW 优化。默认超参数：λ=0.1（搜索范围系数）, n=100（搜索密度）, LoRA r=64, α=16, dropout=0.1, lr=2e-4, batch_size=16。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention

---
