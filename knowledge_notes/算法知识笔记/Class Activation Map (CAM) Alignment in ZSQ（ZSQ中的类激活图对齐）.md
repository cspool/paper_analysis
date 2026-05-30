## Class Activation Map (CAM) Alignment in ZSQ（ZSQ中的类激活图对齐）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Class Activation Map (CAM) Alignment 是 SynQ 论文提出的解决 ZSQ 中量化模型"基于错误图像区域预测（off-target patterns）"问题的技术。核心思想：在微调量化模型时，除了标准的 KL 散度（知识蒸馏）和交叉熵损失外，额外加入 CAM 对齐损失 L_CAM = ||S^θ(x_i) - S^θ^q(x_i)||_F²，其中 S^θ 和 S^θ^q 分别为预训练模型和量化模型的 Grad-CAM 显著性图。通过最小化两者之间的 Frobenius 范数（等价于 MSE），强制量化模型关注与预训练模型相同的图像判别区域，从而将目标定位知识从预训练模型蒸馏到量化模型。与 HAST 的特征对齐（Feature Alignment，对齐中间层激活图）相比，CAM 对齐直接对齐"与预测结果因果相关的区域"，更精确地解决了"off-target prediction"问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def compute_CAM_loss(model_fp, model_q, x, target_class):
    """计算预训练模型与量化模型之间的CAM对齐损失"""
    # 1. 获取目标层的激活和梯度
    # 以ResNet为例，选择最后一层卷积层（layer4）
    activation_fp = get_layer_activation(model_fp, 'layer4', x)  # A^{k;θ}(x)
    activation_q  = get_layer_activation(model_q,  'layer4', x)  # A^{k;θ^q}(x)

    # 2. 计算预训练模型的Grad-CAM
    score_fp = model_fp(x)[target_class]   # y^{y_i}
    grad_fp = autograd.grad(score_fp, activation_fp)[0]  # ∂y/∂A
    alpha_fp = grad_fp.mean(dim=(2,3))     # 全局平均池化 → 通道权重
    S_fp = torch.relu((alpha_fp.view(-1,1,1) * activation_fp).sum(dim=1))
    S_fp = S_fp / (S_fp.max() + 1e-8)      # 归一化到[0,1]

    # 3. 计算量化模型的Grad-CAM（同理）
    score_q = model_q(x)[target_class]
    grad_q = autograd.grad(score_q, activation_q)[0]
    alpha_q = grad_q.mean(dim=(2,3))
    S_q = torch.relu((alpha_q.view(-1,1,1) * activation_q).sum(dim=1))
    S_q = S_q / (S_q.max() + 1e-8)

    # 4. MSE对齐
    L_CAM = torch.norm(S_fp - S_q, p='fro')**2
    return L_CAM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 SynQ 实现中：(1) CAM 对齐选择 Grad-CAM 作为显著性图技术（优于 CAM 和 Grad-CAM++，见 Section 5.4）；(2) 平衡超参数 λ_CAM 在 {20, 50, 100, 200, 300, 500, 2000} 中 grid search；(3) CAM 对齐的时间复杂度为 O(NLT_θ)，每次迭代需执行预训练模型和量化模型各一次前向传播和 L 层反向传播（用于计算梯度），微调时间开销约增加 17.81%；(4) CAM 对齐可无缝集成到任意使用合成数据集的 ZSQ 方法中。SynQ 实验验证 CAM 对齐显著优于特征对齐（表6：CAM 48.26% vs FA 46.77%），且两者训练时间几乎相同。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

---
