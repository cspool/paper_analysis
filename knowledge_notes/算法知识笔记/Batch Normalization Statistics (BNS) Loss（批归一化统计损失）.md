## Batch Normalization Statistics (BNS) Loss（批归一化统计损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Batch Normalization Statistics (BNS) Loss 是 Zero-shot Quantization 中用于引导合成数据集生成的损失函数。其核心思想是：预训练模型的 Batch Normalization 层在训练过程中累积了原始训练数据的 running mean μ^l(θ) 和 running standard deviation σ^l(θ)，这些统计量本质上编码了原始数据在各层的分布特征。BNS Loss 通过最小化合成数据在预训练模型上计算出的统计量与原始统计量之间的 L2 距离，迫使合成数据的分布逼近真实训练数据的分布。公式为 L_BNS = (1/L) Σ_{l=1}^L (||μ^l(θ) - μ^l(θ,{x_i})||² + ||σ^l(θ) - σ^l(θ,{x_i})||²)，其中 L 为 BN 层总数，μ^l(θ,{x_i}) 和 σ^l(θ,{x_i}) 是当前合成样本 batch 在前向传播时在 BN 层 l 处计算出的均值和标准差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def compute_BNS_loss(model, synthetic_batch):
    """计算所有BN层的统计量匹配损失"""
    loss = 0.0
    bn_layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            # module.running_mean: 训练期间累积的原始数据均值 (frozen)
            # module.running_var:  训练期间累积的原始数据方差 (frozen)
            # 前向传播中计算的当前batch统计量
            with torch.no_grad():
                model(synthetic_batch)  # 触发BN层更新当前batch的统计
            # 获取当前batch在BN层的mean和var（需要hook或forward hook提取）
            current_mean = get_current_batch_mean(module)  # μ^l(θ, {x_i})
            current_var  = get_current_batch_var(module)   # σ²
            target_mean  = module.running_mean             # μ^l(θ)
            target_var   = module.running_var              # σ²(θ)
            loss += torch.norm(current_mean - target_mean, p=2)**2
            loss += torch.norm(torch.sqrt(current_var) - torch.sqrt(target_var), p=2)**2
            bn_layer_count += 1
    return loss / bn_layer_count
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中，BN 层的 running_mean 和 running_var 在模型训练后即被冻结（model.eval() 模式下不更新）。实现 BNS Loss 需要：(1) 通过 register_forward_hook 或直接访问 model.modules() 获取每层 BN 的统计量；(2) 在前向传播时使用 model.train() 模式使 BN 层计算当前 batch 的统计（而非使用 running 统计），部分实现要求临时冻结 BN 的 affine 参数；(3) 计算 L2 距离平方作为损失。BNS Loss 是 ZSQ 合成数据集生成的最基础损失项，几乎所有 Noise-optimization ZSQ 方法（ZeroQ, IntraQ, HAST, TexQ, SynQ）均使用。平衡超参数 alpha 通常设为 0.01-1.0。

在 Task-Specific ZSQ for Object Detection 中，BNS Loss 作为 L_prior 用于 CNN-backbone 模型（YOLOv5、CNN-based Mask R-CNN），但与标准 ZSQ 不同——BNS Loss 与 task-specific 检测损失 L_detect 联合使用：L_total = alpha_prior*L_BNS + alpha_detect*L_detect + L_reg，其中 L_detect = L_category + L_box + L_conf，使得合成图像不仅匹配 BN 统计分布，还重建目标检测任务需要的类别和边界框信息。YOLOv5 超参 {alpha_detect, alpha_BN, alpha_TV, alpha_l2} = {0.5, 0.01, 0, 5e-4}。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection

---
