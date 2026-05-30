## SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

- 属于算法pipeline的实现是什么？实验比较什么？
  提出SYNQ，一个面向Zero-shot Quantization（ZSQ）的合成感知微调框架，无需任何真实训练数据即可对预训练模型进行量化。核心包含三个创新组件：(1) **高斯低通滤波器（Low-pass Filter）**：对合成数据集在频域应用高斯低通滤波去除高频噪声，使合成样本的幅度分布更接近真实图像；(2) **类激活图对齐（CAM Alignment）**：通过MSE损失对齐预训练模型与量化模型的Grad-CAM显著性图，确保量化模型基于正确图像区域进行预测；(3) **困难样本仅用软标签（Soft Labels for Difficult Samples）**：根据预训练模型预测概率定义样本难度δ(x_i,θ)=1-q(x_i;θ)，对难度超过阈值τ的样本仅使用KL散度（软标签），不施加交叉熵损失（硬标签），防止错误标签误导训练。实验比较W4A4和W3A3量化下的Top-1准确率，对比方法包括GDFQ、ARC、Qimera、ARC+AIT、IntraQ、AdaSG、AdaDFQ、HAST、TexQ、PLF。

- 硬件平台是什么，配置是什么。
  所有实验在配备Intel Xeon Silver 4214和NVIDIA RTX 3090的工作站上完成。实现基于PyTorch和TorchVision库，Python语言。

- 模型是什么。数据集和bench分别是什么。
  模型：ResNet-20（用于CIFAR-10和CIFAR-100）、ResNet-18、ResNet-50、MobileNetV2（用于ImageNet）；ViT模型包括DeiT-Tiny、DeiT-Small、Swin-Tiny、Swin-Small（均预训练于ImageNet）。数据集：CIFAR-10、CIFAR-100、ImageNet（ILSVRC 2012），仅用于评估（不参与训练）。评估指标：Top-1准确率。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/snudm-starlab/SynQ
  
  算法Pipeline（以ResNet-18 W3A3 ImageNet为例）：
  1. **合成数据集生成（Step 1）**：初始化5120个服从高斯噪声的样本{x_i}及随机类别标签{y_i}。迭代优化最小化Inception Loss L_IL（交叉熵）与Batch Normalization Statistics Loss L_BNS（BN层running mean和std的L2距离），生成与原始分布相似的合成样本。baseline进一步集成TexQ的calibration center synthesis和HAST的hard sample generation/promotion。
  2. **低通滤波（Idea 1）**：对每个合成样本x_i应用高斯低通滤波：x_i^F = F^{-1}(G ⊙ F(x_i))，其中G_{uv} = exp(-D(u,v)²/(2D₀²))，D_0为截止频率超参数（搜索范围{20,40,60,80,100}）。F和F^{-1}分别为FFT和逆FFT。
  3. **量化初始化**：使用RTN（Round-To-Nearest）方案将全精度模型θ量化为θ^q。
  4. **微调量化模型（Step 2，100 epochs）**：对每个滤波后样本x_i^F，计算三项损失：
     - KL散度：KL(q(x_i^F;θ) || q(x_i^F;θ^q))（知识蒸馏，始终应用）
     - CAM对齐损失：L_CAM = ||S^θ(x_i^F) - S^θ^q(x_i^F)||_F²，其中S^θ为Grad-CAM生成的显著性图（Idea 2）
     - 交叉熵损失：仅当δ(x_i^F;θ) ≤ τ时（τ搜索范围{0.5,0.55,0.6,0.65,0.7}）施加λ_CE·CE(q(x_i^F;θ^q), y_i)（Idea 3）
     总损失：L_SYNQ = KL + 1_{δ≤τ}·λ_CE·CE + λ_CAM·L_CAM
  5. **优化器**：SGD with momentum=0.9, weight decay=1e-4。Batch size：CIFAR-10/100为256，ImageNet为16。初始学习率在{1e-4,1e-5,1e-6}中搜索，每epoch衰减0.1。
  
  Grad-CAM显著性图计算：S^θ(x_i) = ReLU(Σ_k α_k·A^{k;θ}(x_i))，其中α_k = (1/(W_k H_k)) Σ_{w,h} ∂y^{y_i}/∂A^{k;θ}_{wh}(x_i)为第k通道激活A^{k;θ}对真实类别预测分数的平均梯度权重。
