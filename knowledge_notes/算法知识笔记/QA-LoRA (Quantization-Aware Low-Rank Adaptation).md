## QA-LoRA (Quantization-Aware Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
QA-LoRA（Quantization-Aware Low-Rank Adaptation）由 Xu et al. (2023, Huawei) 提出，是一种将量化感知引入 LoRA 低秩适配的联合微调-部署方法。核心创新在于引入分组操作（group-wise operators）来平衡量化与适应的自由度（degrees of freedom）：一方面增加量化的自由度（每列分为 L 组，每组独立 α_{l,j}、β_{l,j}，替代 per-column 量化），另一方面降低适应的自由度（输入 x 经组内求和聚合，LoRA 矩阵 A 维度从 D_in×D_int 缩减为 L×D_int，L << D_in）。QA-LoRA 解决了 QLoRA 的核心痛点：(1) QLoRA 微调后合并权重恢复为 FP16，推理时若需 INT4 则需额外 PTQ 导致精度损失；(2) QA-LoRA 通过数学变换，将 LoRA 适配器权重仅合并到零点矩阵 β' = β - s·(BA)⊘α，保持 Ŵ 和 α 不变，使合并后模型仍为 INT 格式，无需 PTQ 直接 INT 推理。因此 QA-LoRA 同时获得微调效率（INT 格式训练）和推理效率（INT 格式部署），在 INT2 极端低位宽下优势尤为显著。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# QA-LoRA 核心 Pipeline (基于 LLaMA-7B, INT4, group_size=32)

# Step 1: GPTQ 分组量化（离线，一次性）
W = [4096, 4096]  # FP16 weight, D_in=D_out=4096
g = 32            # group size
L = D_in // g = 128  # number of groups
# 对每列 j 的每 l 组:
for j in range(D_out):
    for l in range(L):
        w_group = W[l*g:(l+1)*g, j]
        α[l, j] = (max(w_group) - min(w_group)) / (2^N - 1)
        β[l, j] = min(w_group)
        W_hat[l*g:(l+1)*g, j] = round((w_group - β[l,j]) / α[l,j])
# 量化权重: W_tilde = α * (W_hat - β)  (group-wise 反量化形式)

# Step 2: QA-LoRA 初始化
# A = Parameter(L, D_int)  — 相比 QLoRA 的 (D_in, D_int) 减少 D_in/L = 32 倍
# B = Parameter(D_int, D_out)
QA = AvgPool1d(g)  # 组内求和聚合: 4096 → 128

# Step 3: 微调前向
def forward(x, W_tilde, A, B):
    # x: [batch, D_in]
    y_base = x @ W_tilde.T                     # INT4 矩阵乘
    x_agg = QA(x) * g                          # [batch, D_in] → [batch, L] 组内求和
    y_lora = (x_agg @ A.T) @ B.T * s           # s 为 adapt 系数
    return y_base + y_lora

# Step 4: 合并推理（无损 INT 格式）
# 仅更新零点矩阵 β:
β_new[l, j] = β[l, j] - s * (B @ A)[j, l] / α[l, j]
# W_hat 和 α 不变，合并后模型仍为 INT 格式，直接 INT 推理
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/yuhuixu1993/qa-lora。基于 HuggingFace Transformers + PEFT + GPTQ。核心修改：(1) 用 AvgPool1d（group_size=D_in/L）替换原始 linear 的输入，将 D_in 降维到 L；(2) 用分组量化（per-group α、β）替代 per-column 或 per-tensor 量化；(3) merge 时仅更新 β 矩阵。关键超参：L（组数）= D_in // group_size，常用 group_size=32（即 L=128 for 4096-d）；D_int（LoRA 中间维）。训练效率：INT4 算子由 CUDA 优化（vs QLoRA 的 NF4 无算子优化），训练时间比 QLoRA 减少 35-65%（LLaMA-7B: 21.5h vs 40.0h on V100）。推理效率：比 QLoRA FP16 推理快 >50%。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation
- QA-LoRA Quantization-Aware Low-Rank Adaptation of Large Language Models

---
