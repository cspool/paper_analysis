## Information Elastic Connection (IEC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
IEC 是 IR-QLoRA 提出的增强 LoRA 信息表征能力的参数自由连接技术。标准 LoRA 中 ℓ₂ 矩阵仅能使用 ℓ₁ 的低秩变换结果，无法直接访问原始输入 x 的信息，且变换形式局限于矩阵乘法（同质化）。IEC 通过两个参数自由操作解决：(1) **U₁ 弹性下采样**：将原始输入 x 按 (r/h) 比例分组平均后加到 ℓ₁ 输出，使低秩变换能融合原始输入信息；(2) **U₂ 弹性上采样**：将中间表示 x' 重复拼接 (o/r) 次后加到 ℓ₂ 输出，引入参数无关的多样化变换。IEC 每层仅引入 2 个可学习标量 β₁, β₂，且可通过矩阵数学合并入 ℓ₁, ℓ₂ 消除推理开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 h=4096, r=64, o=4096 为例（LLaMA-7B attention 投影层）：
```
# U₁: 输入 x ∈ ℝ^{b×4096} → 中间表示 x' ∈ ℝ^{b×64}
# 步骤 1: 标准 LoRA 左矩阵变换
x₁ = x · ℓ₁                          # [b, 4096] × [4096, 64] = [b, 64]

# 步骤 2: IEC 分组平均（parameter-free）
# 将 x 分成 r=64 组，每组 4096/64 = 64 维，按组求均值
x_split = reshape(x, [b, 64, 64])    # 64 groups of 64 dims
x_pool = mean(x_split, dim=-1)       # [b, 64], group-wise average
x₂ = β₁ · (64/4096) · x_pool         # [b, 64], scaled by learnable β₁

# 步骤 3: 弹性融合
x' = x₁ + x₂                         # [b, 64], IEC-enhanced intermediate

# U₂: 中间表示 x' ∈ ℝ^{b×64} → 输出 ∈ ℝ^{b×4096}
# 步骤 4: 标准 LoRA 右矩阵变换
y₁ = x' · ℓ₂                         # [b, 64] × [64, 4096] = [b, 4096]

# 步骤 5: IEC 重复拼接（parameter-free）
y₂ = β₂ · repeat(x', times=64)       # [b, 64] × 64 concat = [b, 4096]

# 步骤 6: 弹性融合
U_IEC(x) = y₁ + y₂

# 推理时合并（消除开销）：
# ℓ̃₁_{i,j} = ℓ₁_{i,j} + β₁·gcd(h,r)/h  if floor(i/(h/gcd)) == floor(j/(r/gcd))
# ℓ̃₂_{i,j} = ℓ₂_{i,j} + β₂·gcd(o,r)/r  if floor(i/(r/gcd)) == floor(j/(o/gcd))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
IEC 通过修改 PEFT 库中 LoRA 层的 `forward` 函数实现，对每个使用 LoRA 的线性层自动附加 IEC 连接。β₁, β₂ 初始化为小值（约 0.01），在微调过程中与 LoRA 权重一同训练。推理时执行矩阵合并（Eq. 16-17），使得 IEC 完全零开销。IEC 独立于量化方法，可与 QLoRA、QA-LoRA 等任意 LoRA-finetuning 量化框架直接结合。消融实验表明 IEC 单独带来 1.8% MMLU 提升（4-bit LLaMA-7B），其中 U₁ 贡献 1.0%，U₂ 贡献 1.3%。

涉及论文标题：
- Accurate LoRA-Finetuning Quantization of LLMs via Information Retention

---
