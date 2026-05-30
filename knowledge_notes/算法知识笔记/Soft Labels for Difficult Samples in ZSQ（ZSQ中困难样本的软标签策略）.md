## Soft Labels for Difficult Samples in ZSQ（ZSQ中困难样本的软标签策略）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Soft Labels for Difficult Samples 是 SynQ 论文提出的解决 ZSQ 中"错误硬标签误导微调"问题的策略。在 ZSQ 中，合成数据集的标签在初始化时被随机分配，然后通过最小化 Inception Loss 使预训练模型对合成样本预测出对应标签。然而，预训练模型对高难度样本的预测常出错（错误率随难度增加而显著上升，见图3），导致这些样本的硬标签不可靠。SynQ 提出：对难度 δ = 1 - q_{y_i}(x_i; θ) 超过阈值 τ（通常 τ=0.5）的困难样本，训练时完全跳过交叉熵（硬标签）损失，仅使用 KL 散度（软标签/知识蒸馏），以避免错误硬标签对量化模型微调的误导。对于容易样本（δ ≤ τ），则同时使用 KL 散度和交叉熵损失。这一策略通过指示函数 1_{δ(x_i,θ) ≤ τ} 动态决定 CE 损失的施加与否。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def compute_SynQ_loss(model_fp, model_q, x_i, y_i, τ=0.5):
    """SynQ总损失，困难样本仅用软标签"""
    # x_i: 滤波后的合成样本
    # y_i: 硬标签（one-hot）
    with torch.no_grad():
        prob_fp = F.softmax(model_fp(x_i), dim=-1)    # q(x_i; θ)
        difficulty = 1 - prob_fp[y_i.argmax()]          # δ = 1 - q_{y_i}(x_i; θ)

    # KL散度（始终计算，作为软标签的知识蒸馏）
    prob_q = F.softmax(model_q(x_i), dim=-1)
    L_KL = F.kl_div(F.log_softmax(model_q(x_i)), prob_fp, reduction='batchmean')

    # CAM对齐损失（始终计算）
    L_CAM = compute_CAM_loss(model_fp, model_q, x_i, y_i)

    # 交叉熵损失：仅对容易样本施加
    total_loss = L_KL + λ_CAM * L_CAM
    if difficulty <= τ:                                # 容易样本
        L_CE = F.cross_entropy(model_q(x_i), y_i)
        total_loss += λ_CE * L_CE
    # 困难样本: 跳过硬标签，仅使用KL+CAM

    return total_loss
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 难度根据预训练模型对真实标签的预测概率定义；(2) 阈值 τ 通常设为 0.5（最优 trade-off），在 {0.5, 0.55, 0.6, 0.65, 0.7} 中搜索；(3) 对于 CIFAR-10，由于预训练模型错误率在更高难度（约 0.65）才开始上升，τ=0.7 为最优——说明不同数据集/模型的最佳 τ 取决于预训练模型的错误率-难度曲线（图3）；(4) 该策略无需额外计算开销（仅需在微调时判断 δ 是否 ≤ τ 决定是否施加 CE）；(5) 该策略的理论基础是：KL 散度蒸馏让量化模型模仿预训练模型的完整输出分布（含不确定性），而非仅拟合一个可能错误的硬标签。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

---
