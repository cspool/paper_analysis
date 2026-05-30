## Diffusion Model Binarization（扩散模型二值化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
扩散模型二值化是将扩散模型（Diffusion Model）的 UNet 噪声估计网络中的权重和激活从 32-bit 全精度量化为 1-bit 的最极端压缩技术。与 LLM 二值化不同，扩散模型二值化面临三个独特挑战：(1) 模型结构——UNet 的 encoder 逐层下采样（H×W 减半、C 翻倍）、decoder 逐层上采样，维度不断变化，使得 identity shortcut 无法使用，切断了全精度信息流（二值化模型严重依赖 shortcut 补偿 1-bit 信息损失）；(2) 多步迭代——扩散模型需要 T 步（如 T=2000）迭代去噪，每步的激活分布不同，静态的二值化参数（bias、激活函数）无法适配所有 timestep；(3) 任务特性——SR 任务基于像素级重建，对精度极度敏感，直接套用 BNN/ReActNet 等分类任务二值化方法导致性能崩溃。BI-DiffSR 是首个专门为扩散模型 SR 设计的二值化方法，从结构和激活两个维度解决问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 BI-DiffSR 的 ×2 SR 推理 pipeline 为例，扩散模型二值化的完整流程（DDIM 50 步）：
```
x_T ~ N(0, I)                      # 初始高斯噪声
y = bicubic_upsample(LR_image)     # LR 上采样到 HR 分辨率

for timestep_t in reversed(selected_50_timesteps):
    input_6ch = Concat(x_t, y)                    # [H,W,6]
    F_s = FP_Conv(input_6ch)                      # 仅首层 FP
    t_em = SinusoidalPE(t)
    
    # Encoder: 4层，每层 ResBlock×2 + CP-Down
    for level in 1..4:
        for i in 1..2:
            i_grp = floor(K * t / T)              # K=5, 选 timestep 组
            x_shifted = x + bias[i_grp]            # TaR
            x_bin = Sign(x_shifted)                # 1-bit 激活
            w_bin = (||w||_1/n) * Sign(w)          # 1-bit 权重
            x_conv = bit_count(XNOR(x_bin, w_bin)) # 1-bit 卷积
            x_act = RPReLU[i_grp](x_conv)         # TaA
            x = x_act + x_input                    # identity shortcut
        x = CP_Down(x)                             # H/2, C*2
    
    # Decoder: 4层，每层 ResBlock×3 + CS-Fusion + CP-Up
    for level in 1..4:
        x = CS_Fusion(x, encoder_skip[level])
        for i in 1..3:
            # BI-Conv block (同上)
        x = CP_Up(x)                               # H*2, C/2
    
    epsilon_t = FP_Conv(F_d)
    x_{t-1} = DDIM_step(x_t, epsilon_t, t)

HR_image = x_0
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 训练实现：每个 BI-Conv block 中 Sign 用 STE 回传梯度，RPReLU 为 learnable parameter，Adam 优化器，L1 loss。训练 1M iterations（2×A100-80G），batch=16，lr=1e-4。推理时理论加速基于 XNOR+bit-count 替代 FP MAC（32× 内存节省，64× 计算节省），但 BI-DiffSR 未实现定制 CUDA kernel。开源代码：https://github.com/zhengchen1999/BI-DiffSR。模型参数量 4.58M（含折算），单步 OPs 36.67G，比全精度 SR3 节省 79.2% OPs。

涉及论文标题：
- Binarized Diffusion Model for Image Super-Resolution
- BinaryDM Accurate Weight Binarization for Efficient Diffusion Models

BinaryDM 从不同于 BI-DiffSR 的角度解决 DM 二值化问题：不改变 UNet 架构（无 CP-Down/CP-Up 等结构修改），而是通过 QAT 训练策略优化二值化过程。EBB 解决权重表征能力坍塌，LRM 解决优化方向模糊。BinaryDM 适用于 DDIM（pixel-space）和 LDM（latent-space）两种 DM 架构，无需像 BI-DiffSR 那样修改网络结构。与 BI-DiffSR 的 PTQ 策略不同，BinaryDM 是 QAT 方法，需完整训练流程但获得更好的精度-效率平衡（W1A4 FID=7.74 vs EfficientDM W4A4 FID=10.60）。

---
