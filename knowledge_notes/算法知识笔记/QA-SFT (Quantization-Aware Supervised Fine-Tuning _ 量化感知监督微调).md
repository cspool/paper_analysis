## QA-SFT (Quantization-Aware Supervised Fine-Tuning / 量化感知监督微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization-Aware Supervised Fine-Tuning (QA-SFT) 是 RoSTE 论文（ICML 2025）提出的新范式，将量化感知训练（QAT）与监督微调（SFT）合并为单一训练阶段，直接输出量化后的微调模型。传统两阶段 pipeline（先 SFT 后 PTQ）先训练全精度模型再量化，导致量化误差无法在训练中被补偿，性能次优。QA-SFT 的核心公式为：`min_{W,R} L_SFT(m_Q(·; W, R)) s.t. R R^T = I`，同时优化量化权重矩阵 W 和旋转矩阵 R。QA-SFT 区别于 QA-PEFT（如 QLoRA）的关键点在于：(1) 不引入额外适配器参数（如 LoRA），直接优化原始权重；(2) 使用 4-bit 权重量化、激活量化和 KV cache 量化（W4A4KV4），而非仅量化权重；(3) 结合 incoherence processing（旋转矩阵消除 outlier），使低比特激活量化成为可能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RoSTE 的完整 QA-SFT pipeline 伪代码：

```
# 输入: 预训练模型权重 {W_i^pt}_{i=0}^{ℓ-1}, SFT 数据集 D_sft, 校准样本 D_cal (n=128)
# 输出: 量化微调模型 m_Q(·; W^{KT}, R^{K-1})

# Phase 1: 修改 normalization layers
for each norm_layer in model:
    if isinstance(norm_layer, LayerNorm):
        absorb mean subtraction into prev weight matrix
        absorb scale/bias into next weight matrix
    if isinstance(norm_layer, RMSNorm):
        absorb RMSNorm scale into next weight matrix

# Phase 2: 初始化
W^0 = {W_i^pt}_{i=0}^{ℓ-1}  # 从预训练权重初始化

# Phase 3: RoSTE 训练循环 (外层 K=1, 内层 T steps)
for k = 0, ..., K-1:
    # -- Lower Level: Rotation Configuration --
    # 全配置量化误差
    E_no_rotation = compute_E(W^{kT}, {I}_{i=0}^{ℓ-1}, D_cal)    # 公式 (12)
    E_all_rotation = compute_E(W^{kT}, {H}_{i=0}^{ℓ-1}, D_cal)

    # 逐层自适应选择
    for i = 0, ..., ℓ-1:
        err_no_rot = compute_layer_error(W_i, I, D_cal)
        err_rot = compute_layer_error(W_i, H, D_cal)
        R_i^k = I if err_no_rot < err_rot else H   # 选择误差更低的配置

    # -- Upper Level: QAT via Rotation-aware STE --
    for t = 0, ..., T-1:
        mini_batch = sample(D_sft)

        # Forward (每层):
        for each layer i:
            X_rot = Q_x(X · R_i)           # 激活量化（含在线旋转）
            W_rot = Q_w(R_i^T · W_i)       # 权重量化（含旋转）
            output = X_rot · W_rot          # INT4 矩阵乘法

        loss = SFT_loss(model_output, labels)  # CE loss on tokens

        # Backward (STE with rotation):
        # ∂L/∂W_i ≈ R_i · (grad from upper layer)  # STE 旋转修正
        # ∂L/∂X ≈ grad_output @ (W_rot)^T · R_i^T
        W_i -= lr * ∂L/∂W_i

# Phase 4: 推理时合并离线旋转
merge R_1, R_1^T, R_2, R_2^T, R_4^T into adjacent weight matrices
keep R_3, R_3^T, R_4 as online fast Hadamard CUDA kernel
```

QA-SFT 中的量化误差函数 E (公式 12) 用于 rotation selection：
```
E({W_i}, {R_i}) = Σ_i ||Q_w(R_i^T W_i) - R_i^T W_i||^2   # 权重量化误差
                + (1/n) Σ_i Σ_j ||Q_x(X_{i,j} R_i) - X_{i,j} R_i||^2  # 激活量化误差
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
QA-SFT 的实现要点：(1) 量化方案：非对称均匀量化（asymmetric uniform quantizer），per-token activation quantization（沿 token 维度分组），per-channel weight quantization（沿输出通道维度分组），clipping factor c ∈ {1, 0.95, 0.9}；(2) 训练配置：AdamW optimizer，learning rate sweep（不同模型大小使用不同 LR 范围），cosine/linear LR schedule，gradient accumulation for large models（如 Llama 3.1 8B 用 gradient_accumulation=16）；(3) 旋转矩阵实现：使用 fast Hadamard CUDA kernel（继承自 QuaRot/QuIP# 开源实现），离线可合并旋转预先吸收到权重矩阵中减少推理开销；(4) 代码开源：https://github.com/OptimAI-Lab/RoSTE。训练成本：Qwen2.5 7B W4A4KV4 的 RoSTE 训练时间 2.8h（8×A100），比 SFT→QuaRot 两阶段 2.1→0h 略多但精度显著提升（ROUGE Avg 25.10 vs QuaRot 4.79）。

涉及论文标题：
- RoSTE: An Efficient Quantization-Aware Supervised Fine-Tuning Approach for Large Language Models
