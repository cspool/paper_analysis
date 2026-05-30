## QT-DoG: Quantization-aware Training for Domain Generalization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QT-DoG 将量化感知训练（QAT）作为隐式正则化器用于域泛化（Domain Generalization, DG）。核心思想是：权重量化会引入均匀分布的量化噪声 Δ（量化误差范围 [−s/2, +s/2]），该噪声作为隐式正则项，推动优化过程趋向损失景观中的平坦极小值（flatter minima），从而减少对源域的过拟合并提升 OOD 泛化能力。量化公式：w̄ = ⌊clip(w/s, −Q_N, Q_P)⌉，w_q = w̄ × s。采用 LSQ (Learned Step Size Quantization) 作为主要量化方法，每通道独立学习 scaling factor s，所有层除最后一层外均量化至低比特。在训练进行到一定步数后（DomainNet: 8000 步，其余数据集: 2000 步）启动量化。EoQ (Ensemble of Quantization) 训练 5 个独立初始化的量化模型，通过 bagging 方式（平均 softmax 输出）集成预测：ŷ = argmax_k Softmax((1/E) Σ f(x; w_q^i))。
  - 实验比较：(1) QT-DoG (单模型 7-bit) vs ERM, IRM, Group DRO, Mixup, MLDG, CORAL, MMD, Fish, Fishr, SWAD, MIRO, CCFP, ARM, VREx, RSC, Mixstyle, SagNet 等 DG 方法——在 DomainBed 五大数据集上比较 OOD 准确率；(2) EoQ (5 模型集成) vs ERM Ensemble, DiWA, EoA, DART；(3) QT-DoG + CORAL / + MixStyle 组合实验；(4) 不同量化方法消融：LSQ vs INQ (QAT) vs OBC (PTQ)；(5) WILDS 数据集实验 (Amazon, Camelyon)；(6) ViT 实验 (DeiT-Small) 和 CLIP 实验 (ViT-B/16)；(7) Bit precision 分析 (8/7/6/5/4/3/2-bit)；(8) ResNeXt-50-32x4d 更大预训练数据集实验；(9) 量化步数消融 (1000/2000/3000/4000)；(10) Channelwise vs Layerwise scaling factor 消融；(11) 统一噪声注入消融。

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA A100 GPU，Python 3.8.16，PyTorch 1.10.0，Torchvision 0.11.0，CUDA 12.1。CPU 推理延迟测试使用 AMD EPYC 7302 处理器（全精度 34.28ms vs INT8 21.02ms for ResNet-50）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：ResNet-50（25M 参数，ImageNet 预训练，主要骨干）；ResNeXt-50-32x4d（25M 参数，Instagram 1B 图像预训练）；DeiT-Small（ViT，Domain Generalization 实验）；CLIP ViT-B/16（CLIP-based DG 实验）；DistilBERT（WILDS text 任务）。
  - 数据集：DomainBed 基准——PACS (4 domains, 9991 samples, 7-class)、VLCS (4 domains, 10729 samples, 5-class)、OfficeHome (4 domains, 15588 samples, 65-class)、TerraIncognita (4 domains, 24788 samples, 10-class)、DomainNet (6 domains, 586575 samples, 345-class)；WILDS 基准——Amazon (review 分类, 10th percentile acc)、Camelyon17 (病理切片分类, average acc)。
  - Benchmark metric：out-of-domain accuracy（每个 domain 轮流作为目标域，其余为源域，三次独立运行取平均 ± 标准误），使用 DomainBed 训练-域验证协议（Gulrajani & Lopez-Paz, 2021）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://saqibjaved1.github.io/QT_DoG/（论文声称将开源代码、环境配置、复现指令）。
  - 软件依赖：DomainBed 框架（Gulrajani & Lopez-Paz, 2021），PyTorch，LSQ (Esser et al., 2020)，GradCAM 可视化使用 pytorch-grad-cam。
  - 算法 pipeline 核心流程（QT-DoG with LSQ, ResNet-50, 7-bit）：
    1. 训练设置：采用 DomainBed 默认超参——batch_size=32（per-domain），Adam optimizer，lr=5e-5，无 weight decay。ImageNet 预训练 ResNet-50 初始化。每个域轮流作为目标域，其余为源域，20% 源域样本用作验证集。
    2. 前 2000 步（DomainNet 为 8000 步）：正常全精度 ERM 训练。
    3. 第 2000 步起：对除最后一层外的所有层启用 LSQ 量化——
       伪代码：
       ```
       for each layer with weight W:
         s = learnable_parameter(init_value)  # per-channel scaling factor
         W_bar = round(clip(W / s, -Q_N, Q_P))  # Q_N=2^(b-1), Q_P=2^(b-1)-1 for signed b-bit
         W_q = W_bar * s                          # quantized weight
         # Forward: y = x @ W_q
         # Backward: STE (Straight-Through Estimator) through round()
       ```
    4. 每 300 步在源域验证集上评估，选择最佳模型。
    5. 推理时使用量化权重 W_q（INT7 格式），模型体积压缩约 4.6x（25M FP32 → ~5.4M INT7 等效）。
    6. EoQ：独立训练 5 个模型（不同随机种子），集成输出为各模型 softmax 概率的平均：
       ```
       y_hat = argmax(mean([softmax(f(x; W_q^i)) for i in 1..E], dim=0))
       ```
       5 个 7-bit 量化模型总参数量 ≈ 1.1x 全精度单模型，显著低于 DiWA（60 个全精度模型）和 EoA（6 个全精度模型）。
  - 关键发现：7-bit 为最优比特精度（PACS 87.8% vs ERM 84.7%）；QAT (LSQ/INQ) 有效而 PTQ (OBC) 无效（无训练过程无法寻找平坦极小值）；量化在 2000 步时引入效果最佳。
