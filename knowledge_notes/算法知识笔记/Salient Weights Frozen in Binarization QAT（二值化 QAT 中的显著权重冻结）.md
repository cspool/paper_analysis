## Salient Weights Frozen in Binarization QAT（二值化 QAT 中的显著权重冻结）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
显著权重冻结（Salient Weights Frozen）是 PB-LLM QAT 训练框架的核心策略。在量化感知训练开始前，按权重大小（magnitude）排序选出 top-k%（如 2%-30%）的显著权重，这些权重在训练全过程保持冻结（不参与梯度更新），仅对剩余的二值化权重的 FP latent 进行优化。设计动机：(1) 显著权重承载了 LLM 的关键语言能力——即使只保留 2% 的权重不解冻，训练也能更快收敛（图 5 训练曲线）；(2) 冻结减少可训练参数量，降低优化难度——PB-LLM 仅需 1-10K iterations 即可恢复量化模型性能，而 LLM-QAT 等全参数量化训练方法需要 100K iterations；(3) 反直觉的是，仅凭 Salient Frozen + Optimal Scaling 两个机制，无需任何训练就能使部分二值化 LLM 保持一定语言能力（图 6：50% salient 的 OPT-1.3B PPL ~20，非完全崩溃）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 训练前（仅执行一次）
W = pretrained_linear_layer.weight  # ∈ R^{d_o × d_i}
salient_frac = 0.1  # 保留 10% 为显著权重
salient_mask = zeros_like(W)
threshold = quantile(|W|.flatten(), 1 - salient_frac)
salient_mask[|W| >= threshold] = 1

W_salient = W * salient_mask        # Frozen, INT8 quantized
W_F_unsalient = W * (1 - salient_mask)  # Trainable FP latent

# 训练循环
for step in 1..10000:
    # 前向: salient 部分冻结，unsalient 部分二值化
    Ŵ_salient = MinMaxQuant(W_salient, bit=8)     # 固定不变
    Ŵ_unsalient = α * sign(W_F_unsalient)          # 每步更新，α = ||w_F||₁/n
    y = Ŵ_salient @ x + Ŵ_unsalient @ x

    # 反向: 仅更新 W_F_unsalient
    ∂L/∂W_F_unsalient = STE(∂L/∂Ŵ_unsalient)  # STE 穿过 sign()
    W_F_unsalient -= lr * ∂L/∂W_F_unsalient
    # W_salient 不更新 (frozen)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PB-LLM 实现中，salient weight 检测使用 element-wise magnitude 排序（而非 AWQ 的 per-channel activation-based 方法）。选择 magnitude 准则的原因：PB-LLM 实验表明，在 PTQ 场景下 magnitude 和 Hessian 检测的效果差异不大（Table 1），而 magnitude 更简单、无需校准数据。显著权重比例（salient fraction）是关键超参数：30% salient（等效 ~3.7-bit）可接近 FP 性能（LLaMA-7B Avg 66.9 vs FP 68.7），10% salient（等效 ~1.7-bit）仍有合理性能（Avg 60.6）。低于 5% salient 时性能急剧下降。显著权重的 element-wise 分布呈均匀随机散射（图 3），无明显的列聚集模式，因此 column-wise 选择方法不适合二值化场景。

涉及论文标题：
- PB-LLM Partially Binarized Large Language Models
