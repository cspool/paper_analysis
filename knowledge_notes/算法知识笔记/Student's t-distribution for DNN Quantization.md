## Student's t-distribution for DNN Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在 DNN 量化上下文中，Student's t-distribution 用于建模神经网络权重和激活的统计分布。t-distribution 由自由度参数 ν 控制，其概率密度函数为：S(t;ν) = Γ((ν+1)/2) / (√(νπ)·Γ(ν/2)) · (1 + t²/ν)^{-(ν+1)/2}。ν 越小，分布的峰值越尖（leptokurtic）、尾部越厚（heavy-tailed）；ν→∞ 时收敛到标准正态分布 N(0,1)。本论文对 30+ DNN（含 LLM、BERT 类 Transformer、CNN、ViT）的权重和激活进行大规模 profiling，发现：(1) 大多数分布的自由度 ν 在个位数（约 2-7）；(2) Kolmogorov-Smirnov (KS) 检验证明 t-distribution 的拟合度优于正态分布（KS-Δ 大多为正值）；(3) ν ≈ 10 可视为正态分布的近似分界线。这一发现颠覆了量化领域长期依赖的正态分布假设（NF4 的基础）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
t-distribution profiling pipeline：

```
# 逐模型、逐层、逐 tensor 的统计分析
for model in [LLMs, BERT-variants, CNNs, ViTs]:
    for layer in model.modules:
        if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            W = layer.weight.detach().flatten()
            if W.numel() > N_max:
                W = random_downsample(W)  # 大张量下采样
    
            # 对每个 tensor 拟合 t-distribution 和 normal distribution
            ν_W, μ_W, σ_W = fit_t_distribution(W)  # MLE 估计
            μ_N, σ_N = fit_normal(W)
    
            # KS 检验比较拟合优度
            KS_t = ks_test(W, t_distribution(ν_W, μ_W, σ_W))
            KS_n = ks_test(W, normal(μ_N, σ_N))
            KS_Δ = KS_n - KS_t  # 正值 → t-distribution 更优

# 激活 profiling: 用随机生成输入（与模型适配的 shape）前向传播
# 收集每层的激活张量，重复上述统计
```

该 profiling 的直接输出：大多数模型的 ν 均值在 5 附近。这成为 SF4 固定 ν=5 的经验依据。Table 1 列出主要模型的 ν 和 KS-Δ 值：Mistral-7B (ν=1.66), LLaMA2-7B (ν=6.78), OPT-1B (ν=6.68), BLOOM-7B (ν=10.13, 接近正态边界)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现基于 HuggingFace Transformers、PyTorch torchvision 和 timm 库加载模型。对 nn.Linear、nn.Conv1D、nn.Conv2D 层的权重逐一 profiling；大张量（数亿参数）随机下采样（经验显示不影响结果）。激活 profiling 使用随机生成输入。t-distribution 参数通过最大似然估计（MLE）拟合。Profiling 结果直接指导：(a) SF4 的 ν 选择；(b) 判断哪些模型适合基于正态分布的量化方法（ν>10 的模型如 FLAN-T5, BLOOM-7B 可能与 NF4 兼容）；(c) 设计硬件高效数据类型（E2M1 形状分段逼近 SF4，因 SF4 的形状反映了 t-distribution 的概率密度结构）。

涉及论文标题：
- Learning from Students: Applying t-Distributions to Explore Accurate and Efficient Formats for LLMs
