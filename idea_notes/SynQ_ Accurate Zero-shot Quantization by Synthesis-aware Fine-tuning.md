## SynQ: Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

- baseline方法是什么？
  Baseline方法为HAST + TexQ的合成数据集生成部分（calibration center synthesis + hard sample generation + sample difficulty promotion），微调阶段使用标准ZSQ损失函数 L_ZSQ = KL(q(x_i;θ) || q(x_i;θ^q)) + λ_CE·CE(q(x_i;θ^q), y_i)，即对所有合成样本同时施加KL散度知识蒸馏和交叉熵硬标签损失。全栈执行例子（ResNet-18 W3A3 on ImageNet）：
  - 算法Pipeline：随机初始化高斯噪声样本 → 最小化L_IL + α·L_BNS优化合成样本 → RTN量化预训练模型 → 对所有样本统一使用KL+CE损失微调量化模型 → 输出W3A3量化权重和激活
  - 系统框架：PyTorch + TorchVision，单卡RTX 3090训练
  - 编译框架：论文未明确说明
  - Kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明
  Baseline存在三个核心缺陷：1）合成数据集含高频噪声，频域分布与真实图像差异显著；2）量化模型基于错误图像区域（off-target patterns）做预测；3）困难样本的硬标签常错误，误导微调。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  SYNQ在baseline合成数据集生成后，引入三个合成感知的微调创新改进：
  - Idea 1（低通滤波）：对合成样本x_i在频域应用高斯低通滤波器G，去除高频噪声 → 解决缺陷1。x_i^F = F^{-1}(G ⊙ F(x_i))，G为2D高斯核。滤波后样本的幅度分布更接近真实图像（ImageNet主要集中低频），使量化模型微调更高效。
  - Idea 2（CAM对齐）：在微调损失中加入L_CAM = ||S^θ(x_i^F) - S^θ^q(x_i^F)||_F²，强制量化模型的Grad-CAM显著性图对齐预训练模型 → 解决缺陷2。Grad-CAM通过激活梯度加权定位对预测贡献最大的图像区域，ReLU保留正贡献区域，MSE对齐确保量化模型关注与预训练模型相同的目标区域。
  - Idea 3（困难样本软标签）：定义样本难度δ = 1 - q_{y_i}(x_i;θ)，当δ > τ时仅使用KL散度（软标签），跳过硬标签交叉熵 → 解决缺陷3。避免预训练模型对困难样本的误分类标签误导量化模型。
  总损失：L_SYNQ = (1/N) Σ [KL + 1_{δ≤τ}·λ_CE·CE + λ_CAM·L_CAM]，其中低通滤波在微调前一次性完成。
  
  全栈执行例子（ResNet-18 W3A3 on ImageNet）：
  - 算法Pipeline：随机初始化高斯噪声样本 → 最小化L_IL + α·L_BNS生成合成样本 → FFT → 逐元素乘高斯低通核G → IFFT得到滤波样本x_i^F → RTN量化初始化θ^q → for each epoch/每个x_i^F: 前向传播θ和θ^q → 反向传播计算Grad-CAM梯度 → 计算L_CAM对齐 → 计算δ判定是否施加CE → 聚合KL + 条件CE + L_CAM → 更新θ^q（SGD, momentum=0.9, wd=1e-4, 100 epochs） → 输出W3A3量化模型
  - 系统框架：PyTorch + TorchVision，单卡RTX 3090，5120张合成样本，batch size=16（ImageNet），合成阶段Adam优化器（lr=0.5, 衰减0.1/50步）
  - 编译框架：论文未明确说明
  - Kernel调度：论文未明确说明
  - 硬件架构：论文未明确说明
  
  Ablation验证（ResNet-18 W3A3）：Baseline 43.63% → +I1 49.43% (+5.80pp) → +I2 48.26% (+4.63pp) → +I3 46.42% (+2.79pp) → 全组合52.02%。低通滤波贡献最大，三项互补。SYNQ兼容任意ZSQ方法（GDFQ、Qimera、AdaDFQ、IntraQ、HAST、TexQ、Genie-PTQ），平均微调时间开销仅17.81%。
