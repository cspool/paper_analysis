## Zero-shot Quantization (ZSQ / 零样本量化 / Data-Free Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Zero-shot Quantization（ZSQ），也称为 Data-Free Quantization（DFQ），是一种在无法访问原始训练数据的情况下对预训练模型进行量化的技术。ZSQ 的核心流程分为两步：(1) **合成数据集生成（Step 1）**：利用预训练模型的内部统计信息（如 Batch Normalization 层的 running mean 和 standard deviation、分类层预测分布、特征纹理等）反向优化随机噪声或训练一个生成器，生成与原始训练数据分布相似的合成样本；(2) **量化模型微调（Step 2）**：使用 Round-To-Nearest（RTN）初始化量化模型，然后用合成数据集微调量化模型，最小化 KL 散度（知识蒸馏）和交叉熵损失。ZSQ 在数据因隐私、安全或法规原因不可用的真实场景（如医疗数据、商业机密数据）中至关重要。

ZSQ 方法按合成数据集生成方式分为三类：
- **Synthesis-free ZSQ**：无需生成数据，仅基于模型参数属性（如 weight equalization、bias correction）校准量化参数，但在极低位宽（3-bit/4-bit）下性能急剧退化。
- **Generator-based ZSQ**：训练一个额外的生成器网络（如 GAN）产生合成样本。代表方法：GDFQ（首次使用 BN 统计引导的生成器）、ARC/AutoReCon（NAS-based 图像重建）、Qimera（叠加潜在嵌入生成边界支持样本）、AdaSG（将 ZSQ 建模为零和博弈）、AdaDFQ（自适应调节合成样本难度）。
- **Noise-optimization-based ZSQ**：直接从随机高斯噪声出发，通过梯度下降迭代优化噪声以匹配预训练模型统计信息，无需训练额外生成器。代表方法：IntraQ（保留类内异质性）、HAST（困难样本生成与难度提升）、TexQ（纹理特征分布校准）、PSAQ-ViT（ViT 的 patch similarity 引导）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 SynQ 论文的 Noise-optimization ZSQ pipeline（ResNet-18 W3A3 on ImageNet）为例：
```
// === Step 1: 合成数据集生成（Noise Optimization） ===
Initialize {x_i}_{i=1}^{5120} ~ N(0, 1)          // 5120张高斯噪声图像
Randomly assign labels {y_i}_{i=1}^{5120}         // 随机类别标签
for iter in 1..1000:
    // 最小化两项损失
    L_IL = (1/N) Σ CE(q(x_i; θ), y_i)             // Inception Loss: 预训练模型预测对齐标签
    L_BNS = (1/L) Σ ||μ^l(θ) - μ^l(θ,{x_i})||²   // BN Statistics Loss: 匹配BN层统计
           + ||σ^l(θ) - σ^l(θ,{x_i})||²
    L_total = L_IL + α * L_BNS
    x_i = x_i - η * ∇_{x_i} L_total              // 更新合成样本（不更新模型参数）
    if loss plateau for 50 iters: η *= 0.1       // 学习率衰减

// === Step 2: 量化与微调 ===
θ^q = RTN_quantize(θ, bit=3)                      // 使用RTN初始量化（Min-max）
for epoch in 1..100:
    for each x_i in {x_i}:
        loss = KL(q(x_i; θ) || q(x_i; θ^q))        // 知识蒸馏（始终应用）
             + λ_CE * CE(q(x_i; θ^q), y_i)         // 交叉熵（硬标签）
    θ^q = θ^q - η * ∇_{θ^q} loss                   // SGD 更新 θ^q
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ZSQ 的通用实现方式：(1) 使用 PyTorch 加载预训练模型，提取 BN 层 running_mean 和 running_var；(2) 初始化随机噪声张量作为合成样本，在某些方法中还需初始化生成器网络；(3) 使用 Adam 优化器迭代优化合成样本（或训练生成器），损失函数通常包括 BNS loss（L2 距离匹配 BN 统计量）和 CE loss/IL（促使预训练模型对合成样本做出指定类别的预测）；(4) 使用 torch.quantization 或自定义 STE 伪量化模块执行 RTN 量化初始化和微调；(5) 微调时通常使用 SGD + momentum=0.9 优化器，学习率在 {1e-4, 1e-5, 1e-6} 范围。SynQ 开源地址：https://github.com/snudm-starlab/SynQ。ZSQ Survey (IJCAI 2025)：https://github.com/snudm-starlab/ZSQ-Survey。

ZSQ 用于目标检测的 Task-Specific 扩展：Task-Specific ZSQ for Object Detection 指出，现有 ZSQ 方法用于检测任务时存在根本性问题——分类 ZSQ 仅需随机采样类别标签（如 U(0,1000) for ImageNet），而检测任务需要同时合成边界框坐标和类别标签。task-agnostic ZSQ（如 PSAQ-ViT V2、MimiQ、CLAMP-ViT）仅使用 BNS 或 PSE 先验损失生成无任务的通用图像，放弃检测训练损失，导致合成图像缺乏目标位置、尺寸和类别分布的 task-specific 信息，性能次优。task-specific 方案通过 Adaptive Label Sampling（利用预训练检测网络自动重建 label）和 task-specific QAT（同时使用 KL 蒸馏 + 特征蒸馏 + L_detect 检测损失）解决了这一问题，在 W6A6 YOLOv5-l 上超越 full-data LSQ +1.8% mAP，仅用 1/60 训练数据。开源代码：https://github.com/DFQ-Dojo/dfq-toolkit。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning
- Task-Specific Zero-shot Quantization-Aware Training for Object Detection

---
