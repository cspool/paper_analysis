## Thinking in Granularity Dynamic Quantization for Image Super-Resolution by Intriguing Multi-Granularity Clues

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Granular-DQ，一种 patch-wise、layer-invariant 的动态量化方法，包含两个顺序步骤：(1) Granularity-Bit Controller (GBC) 构建粗到细的多粒度层次表示，按每个 patch 对整张图像的贡献比例分配 bit-width；(2) Entropy-to-Bit (E2B) 机制基于像素熵统计对高 bit patch 进行细粒度 bit-width 自适应调整，配合 Adaptive Threshold Calibration (ATC) 利用 EMA 动态校准熵阈值。实验比较基线：与全精度模型及 PAMS、CADyQ、CABM、AdaBM、RefQSR 对比 PSNR/SSIM 和 FAB (Feature Average Bit-width)；消融研究 GBC、E2B、ATC 各自贡献及候选 bit 配置和阈值数量的影响。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 4090 GPUs，PyTorch 框架实现。训练时 LR RGB patch 随机裁剪至 48×48（CNN）或 64×64（Transformer），batch size 16，300K iterations，初始 lr=2×10^{-4}，250K iterations 后减半。

- 模型是什么。数据集和bench分别是什么。
  CNN 模型：SRResNet、EDSR、IDN；Transformer 模型：SwinIR-light、HAT-S。训练集：DIV2K（800 样本，×2 和 ×4 SR）。评估 benchmark：Urban100、Test2K、Test4K（源自 DIV8K 经 bicubic 下采样）。指标：PSNR、SSIM（重建精度）、FAB（量化效率）、BitOPs（计算复杂度）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源链接：https://github.com/MmmingS/Granular-DQ.git。算法流程：
  1. 输入图像 X → GBC 编码器 E 提取 D 层多粒度特征 Z={Z_1,...,Z_D}（分辨率递减，Z_1 最细粒度，Z_D 最粗粒度）。
  2. 所有粒度特征 GroupNorm + 平均池化到 Z_D 分辨率 → concat → GAP → 通道统计量 S。
  3. 线性层 W_g ∈ R^{(N×D)×N} 作用于 S 生成门控 logits G，对每个 patch X_i 使用 Gumbel-Softmax 采样门控索引 θ_i = argmax_n(g_{i,n} + σ_n)，计算门控分数 p_i（patch 贡献概率），映射到候选 bit code b_n ∈ {4,6,8}。
  4. E2B：对训练集所有 LR patch 计算像素熵 H（基于 Gaussian 加权核密度估计），按升序排序得到 H。插入分位数阈值 t1=0.5, t2=0.9 将 H 划分为 3 个子区间，对应 bit codes [4,5,8]。对 GBC 分配高 bit 的 patch，据其熵 E 落入区间决定适配 bit-width。
  5. ATC：训练首 epoch 用 EMA (γ=0.9997) 动态校准阈值 t^(j) = t^(j-1)·γ + Norm(E)·(1-γ)。
  6. 量化器：QuantSR 作为候选量化方案，权重统一 8-bit 线性量化。仅使用 L1 loss 训练。Transformer 的 attention block 保持全精度。
