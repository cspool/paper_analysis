## Synthesis-aware Fine-tuning（合成感知微调）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Synthesis-aware Fine-tuning 是 SynQ 论文提出的 Zero-shot Quantization 微调范式。与传统的 ZSQ 微调（对所有合成样本统一使用 KL 散度 + 交叉熵损失）不同，"合成感知"意味着微调过程充分认识到合成数据集与真实数据的三个关键差异并针对性解决：(1) 合成数据集含有高频噪声（频域分布与真实图像不同）→ 通过低通滤波去除噪声；(2) 合成数据集导致量化模型学习错误图像区域（off-target patterns）→ 通过 CAM 对齐蒸馏定位知识；(3) 合成数据集的硬标签对困难样本不可靠 → 困难样本仅用软标签。这三个策略共同构成了 synthesis-aware 的微调框架，使得量化模型在合成数据上的微调效果更接近在真实数据上的微调。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SynQ 的 Synthesis-aware Fine-tuning 完整流程（Algorithm 1）：
```
输入: 预训练模型θ, 超参数 n_ep, D0, λ_CAM, λ_CE, τ
输出: 量化模型参数 θ^q

// Step 1: 生成合成数据集（任意ZSQ方法均可）
{x_i}_{i=1}^N = generate_synthetic_dataset(θ, N=5120)

// Step 2: 合成感知微调
θ^q = RTN_quantize(θ)                                  // 量化初始化
{x_i^F} = gaussian_low_pass_filter({x_i}, D0)          // Idea 1: 低通滤波
for epoch in 1..n_ep:
    for x_i^F, y_i in {x_i^F, y_i}:
        // 计算KL、CAM损失（始终应用）
        L = KL(q(x_i^F; θ) || q(x_i^F; θ^q)) + λ_CAM * L_CAM(x_i^F; θ, θ^q)
        // 条件交叉熵（Idea 3）
        if δ(x_i^F, θ) ≤ τ:
            L += λ_CE * CE(q(x_i^F; θ^q), y_i)
        L.backward()
    optimizer.step()
return θ^q
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Synthesis-aware Fine-tuning 作为一个框架可集成到任意使用合成数据集的 ZSQ 方法中。SynQ 论文验证了其在 6 种 baseline（GDFQ, Qimera, AdaDFQ, IntraQ, HAST, TexQ）上的兼容性：在 Qimera W3A3 ResNet-18 上提升高达 31.17%p（1.17% → 32.34%），在 PTQ 方法 Genie 上平均提升 0.66%p。三个组件的贡献排序（ResNet-18 W3A3）：低通滤波（+5.80pp）> CAM 对齐（+4.63pp）> 困难样本软标签（+2.79pp）。三者组合达到最佳（基线 43.63% → 52.02%，+8.39pp）。微调时间开销仅 17.81%，且性能随合成数据集大小增长而提高。

涉及论文标题：
- SynQ Accurate Zero-shot Quantization by Synthesis-aware Fine-tuning
