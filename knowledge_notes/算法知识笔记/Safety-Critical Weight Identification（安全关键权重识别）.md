## Safety-Critical Weight Identification（安全关键权重识别）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
安全关键权重识别是 Q-resafe 框架的核心机制之一，指在量化 LLM 的众多权重中识别出对模型安全能力最为关键的一小部分权重（top-τ%），以便仅对这些权重进行定向修补而不扰动其余权重。其理论基础：(1) LLM 的能力（包括安全能力）集中在少部分权重中；(2) 量化主要损害效用相关的权重，安全相关的权重子空间在量化过程中因缺乏专门保护而严重受损；(3) 通过 SNIP score I(W_ij, x) = |W_ij · ∇_{W_ij} L(x)| 度量每个权重对安全相关损失（条件负对数似然）的敏感性，排序后选 top-τ% 作为安全关键权重。τ 的选取平衡安全恢复与效用保持：τ=1（全部更新）ASR 降至 1.6% 但 GPU-hours=2.1；τ=0.6 ASR=1.8% GPU-hours=1.2；τ=0.2 ASR=13.9% GPU-hours=0.5。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```python
# 周期性安全关键权重识别（每 K=1000 步执行一次）
if step % K == 0:
    # Step 1: 计算当前模型 Q^t 的每层 SafeScore
    for each layer l with weight matrix Q_l:
        SafeScore_l = zeros_like(Q_l)
        for each prompt x in D_calib:
            y = sample(model, x)  # 或使用 ground truth
            loss_l = -log p(y|x; Q_l)
            loss_l.backward()
            SafeScore_l += abs(Q_l * Q_l.grad)  # element-wise SNIP
        SafeScore_l /= len(D_calib)

    # Step 2: 全局 top-τ 选择
    all_scores = flatten([SafeScore_l for l in layers])
    threshold = top_k_percentile(all_scores, τ)

    # Step 3: 构建掩码矩阵
    for each layer l:
        M_Q[l] = (SafeScore_l >= threshold).float()  # ∈ {0,1}

    # Step 4: 映射到 LoRA 维度
    # M_Q ∈ {0,1}^{d_in × d_out} → (M_A ∈ {0,1}^{d_in × r}, M_B ∈ {0,1}^{r × d_out})
    for each layer l:
        row_mask = any(M_Q[l], dim=1)  # 有安全关键权重的行
        col_mask = any(M_Q[l], dim=0)  # 有安全关键权重的列
        M_A[l] = row_mask.unsqueeze(1).expand(d_in, r)  # 整行标记
        M_B[l] = col_mask.unsqueeze(0).expand(r, d_out)  # 整列标记
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现时：(1) 校准数据集 D_calib 用于计算 SNIP score，Q-resafe 使用 Alpaca-cleaned 或 UltraChat；(2) 周期 K 的选择平衡识别准确性和计算开销——K=1000 为论文默认值，每 1000 步重新评估一次哪些权重是安全关键的；(3) τ 的选取通过消融实验确定（论文推荐 τ=0.6）；(4) 对于无微调量化方法（AWQ），Q-resafe 不执行 DPO 训练，而是在全精度预训练模型上直接计算 SNIP score，将 top-τ% 安全关键权重保留为 FP16，其余量化为 INT4。该技术的核心优势：仅修改极小部分权重即可恢复安全能力，避免了全量微调的计算开销（1.2 GPU-hours vs SFT 8.4h/DPO 9.6h）和效用损失。

涉及论文标题：
- Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models
