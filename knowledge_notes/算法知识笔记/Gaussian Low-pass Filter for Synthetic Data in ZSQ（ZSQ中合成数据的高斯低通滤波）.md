## Gaussian Low-pass Filter for Synthetic Data in ZSQ（ZSQ中合成数据的高斯低通滤波）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gaussian Low-pass Filter for Synthetic Data 是 SynQ 论文提出的解决 ZSQ 合成数据集高频噪声问题的技术。合成数据集由噪声优化生成（起始于高斯噪声），其频域特征与真实图像显著不同——合成样本的幅度谱均匀分布在高频和低频区域，而真实图像（如 ImageNet）的能量主要集中于低频区域。SynQ 通过傅里叶变换将合成样本 x_i 转换到频域，逐元素乘以 2D 高斯低通核 G 进行滤波，再应用逆傅里叶变换得到滤波后的样本 x_i^F。高斯核 G_{uv} = exp(-D(u,v)²/(2D₀²))，其中 D(u,v) 为频域坐标 (u,v) 到中心的距离，D₀ 为截止频率超参数。滤波后合成样本的幅度分布显著接近真实图像，缓解了噪声导致微调效率低下的问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def gaussian_low_pass_filter(x, D0):
    """对合成样本x应用高斯低通滤波器"""
    # x: (C, H, W) 或 (N, C, H, W) 张量
    # D0: 截止频率（ImageNet推荐值: 40-60）
    C, H, W = x.shape[-3], x.shape[-2], x.shape[-1]

    # 1. 构建2D高斯低通核
    u = torch.arange(W).float() - W/2
    v = torch.arange(H).float() - H/2
    U, V = torch.meshgrid(u, v, indexing='xy')
    D = torch.sqrt(U**2 + V**2)                    # 到频域中心的距离
    G = torch.exp(-D**2 / (2 * D0**2))              # 高斯低通核 H×W
    G = G.unsqueeze(0).unsqueeze(0)                 # (1, 1, H, W)

    # 2. 逐通道应用频域滤波
    x_filtered = torch.zeros_like(x)
    for c in range(C):
        F_x = torch.fft.fft2(x[c])                  # 2D FFT
        F_x_shifted = torch.fft.fftshift(F_x)       # 零频移到中心
        F_filtered = F_x_shifted * G.squeeze()       # 逐元素乘高斯核
        F_ishifted = torch.fft.ifftshift(F_filtered)
        x_filtered[c] = torch.fft.ifft2(F_ishifted).real  # 逆FFT取实部
    return x_filtered

# 在SynQ pipeline中的使用位置（生成合成数据集后、微调前）
synthetic_dataset = generate_synthetic_samples(model, N=5120)
filtered_dataset = [gaussian_low_pass_filter(x_i, D0=50) for x_i in synthetic_dataset]
# filtered_dataset随后用于微调量化模型
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：(1) 使用 PyTorch 的 torch.fft.fft2 / torch.fft.ifft2 进行 2D 快速傅里叶变换，时间复杂度 O(Z log Z)（Z=HW）；(2) 对 CIFAR 数据集使用较小 D₀（约 8-20），对 ImageNet 使用较大 D₀（约 40-60）；(3) 滤波在合成数据集生成完成后、量化模型微调开始前一次性离线完成，不增加微调循环的计算开销；(4) 过小的 D₀ 会导致过度平滑丢失关键信息，过大的 D₀ 则滤波效果不足。SynQ 消融实验证明低通滤波器是三项贡献中影响最大的（ResNet-18 W3A3：基线 43.63% → +I1 49.43%，+5.80pp）。该技术可直接应用于任意 ZSQ 方法生成的合成数据集。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning

---
