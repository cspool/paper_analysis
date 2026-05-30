## Grad-CAM (Gradient-weighted Class Activation Mapping / 梯度加权类激活映射)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Grad-CAM（Gradient-weighted Class Activation Mapping）是 Selvaraju et al. (ICCV 2017) 提出的可视化 CNN 决策依据的技术。Grad-CAM 生成一张与输入图像同尺寸的热力图（saliency map），高亮区域表示模型在做出特定类别预测时"关注"的图像部分。其核心公式为：S^θ(x_i) = ReLU(Σ_k α_k · A^{k;θ}(x_i))，其中 A^{k;θ}(x_i) 为目标卷积层第 k 通道的激活图，α_k = (1/(W_k H_k)) Σ_{w,h} ∂y^{y_i}/∂A^{k;θ}_{wh}(x_i) 为第 k 通道对目标类别预测分数的平均梯度权重（即通道重要度），ReLU 过滤掉负贡献区域只保留对预测有正面影响的区域。Grad-CAM 是 CAM (Zhou et al., CVPR 2016) 的直接推广——CAM 要求模型末尾有全局平均池化层（GAP），Grad-CAM 通过梯度反向传播消除了此限制，可应用于任意 CNN 架构。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 SynQ 论文中，Grad-CAM 作为 CAM 对齐损失 L_CAM 的显著性图生成方法，其计算流程为：
```
输入: 预训练模型θ, 输入图像x, 真实类别y_i, 目标层layer4
输出: 显著性图 S^θ(x) (尺寸: H×W)

1. 前向传播: output = θ(x), 记录layer4的激活A ∈ R^{K×W'×H'}
2. 获取预测分数: y_score = output[y_i]           // 真实类别的logit
3. 反向传播梯度: grad = ∂(y_score) / ∂A           // K×W'×H'
4. 全局平均池化梯度: α_k = (1/(W'H')) Σ_{w,h} grad[k,w,h]  // K维向量
5. 加权组合: S_raw = Σ_k α_k · A[k,:,:]           // W'×H'
6. ReLU过滤: S = ReLU(S_raw)                      // 丢弃负贡献
7. 上采样: S = interpolate(S, (H,W))              // 恢复到原图尺寸
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Grad-CAM 在 PyTorch 中的实现：(1) 使用 register_forward_hook 捕获目标层的激活 A；(2) 使用 register_full_backward_hook 或 torch.autograd.grad 获取梯度；(3) 在指定类别（通常为真实标签类别）上反向传播获取通道权重。SynQ 对比了三种 CAM 技术（W3A3 ResNet-18）：Grad-CAM > CAM > Grad-CAM++。Grad-CAM++ 专为多目标定位设计，对于单目标分类任务不如 Grad-CAM。Grad-CAM 优于 CAM 的另一个优势是：CAM 仅适用于末尾有 GAP 层的模型（如 ResNet），Grad-CAM 无此限制。Grad-CAM 还可用于模型可解释性分析、弱监督目标定位、以及作为知识蒸馏中的注意力转移目标。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

---
