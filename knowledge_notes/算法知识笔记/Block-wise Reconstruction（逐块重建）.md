## Block-wise Reconstruction（逐块重建）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-wise Reconstruction 是 BRECQ (ICLR 2021) 提出的后训练量化框架，将模型按 block 单元逐个进行量化重建。流程：(1) 将模型拆为若干 block；(2) 用校准集获取 block 原始 FP 输出 O；(3) 量化 block 内权重/激活得到 Ô；(4) 最小化 Ô 与 O 的重建损失（MSE 或 Hessian guided loss）；(5) 使用 AdaRound 学习舍入策略。Block-wise 相比 layer-wise 考虑了跨层依赖，相比全局重建计算开销可控。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for each block B in M.blocks:
    O_fp = B.forward(D)
    H_bar = compute_APH(O_fp, M.rest, D)       # (APHQ-ViT)
    if B has MLP:
        replace GELU → ReLU; reconstruct MLP   # MR (APHQ-ViT)
    for iter in range(max_iter):
        O_hat = B_quantized.forward(D)          # QDrop + AdaRound
        L = sum((O_hat - O_fp)^2 * H_bar)       # APH loss
        L.backward(); update(AdaRound_weights)
    M.replace_block(B, B_quantized)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 中通过 hook 机制截取 block 输入/输出。典型超参：batch_size=32, max_iter=20000, lr_weight=1e-3, lr_act=4e-5。校准集：ImageNet 1024 张无标签图。BRECQ 用 MSE 或 FIM 近似 Hessian loss；QDrop 添加随机 activation dropout；APHQ-ViT 用 APH loss + MR。

I&S-ViT 使用标准 block-wise reconstruction L_l = ||X_l - X̄_l||_2 作为学习目标，只向后更新第 l 个 transformer block 的权重。在 SOS 三阶段策略中：Stage 1 用全精度权重 + channel-wise 激活量化优化 block-wise loss；Stage 2 通过 scale reparameterization 无损转换；Stage 3 在量化权重和 layer-wise 量化下再优化 block-wise loss。Adam 优化器，lr=4e-5，cosine 衰减，ImageNet batch_size=64，6-bit 200 iterations，其他 1000 iterations。

在 EfficientQAT 的 Block-AP 中，block-wise reconstruction 首次被扩展为真正的 QAT——直接训练所有权重(W)和量化参数(s, z)，而非仅训练辅助参数（rounding/clipping/step sizes）。Block-AP 实验证明（Table 5）：全训练(s,z,W) PPL=8.53 vs 仅训练rounding PPL=15.50 vs 仅训练clipping PPL=11.28，且全训练显存(8.5GB)反而低于rounding训练(8.6GB，因需额外存储rounding参数副本)。这表明无需复杂的可训练参数设计即可实现最优block-wise重建。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- ARB-LLM Alternating Refined Binarizations for Large Language Models
- AffineQuant Affine Transformation Quantization for Large Language Models
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization
- OmniQuant Omnidirectionally Calibrated Quantization for Large Language Models
- AnyBCQ Hardware Efficient Flexible Binary-Coded Quantization for Multi-Precision LLMs
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs
- D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models
- EfficientQAT Efficient Quantization-Aware Training for Large Language Models
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
- PT²-LLM Post-Training Ternarization for Large Language Models
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers

aespa 论文提出了一种介于 layer-wise 和 block-wise 之间的折中方案：逐层单独量化以保持效率（layer-wise granularity），但以 attention 输出为重构目标（block-wise-like reconstruction target）。与 BRECQ 的全 block 联合量化不同，aespa 将 W_Q、W_K、W_V 逐个量化，但每层的损失函数都指向 attention 输出误差最小化。该策略的复杂度为 O(d_h d^2)，远低于传统 block-wise 的 O(B d_h L·max{d,L})。实验表明该折中方案在 INT3/4 下性能几乎不损失，INT2 下仅有轻微退化（vs BRECQ），但量化速度提升 10× 以上。，在其基础上额外收集量化输出-噪声对 (ε̂, Δε)_t 用于估计时间步感知的联合高斯分布参数。

BiLLM 使用 GPTQ/OBC 的 block-wise 二阶误差补偿（block_size=128），但移除了 column-wise 补偿以提升 PTQ 效率。补偿过程：对每个 128 列 block 完成量化后，计算误差 E = (W - B) / H_chol_inv，将 E 乘以 Hessian 逆对应行补偿到后续未量化列中（W_{:,b+β:} -= E · H_chol_inv_{b:b+β, b+β:}），此方法在二值化场景同样有效。

PT²-LLM 在 block-wise ternarization（block_size=128）中使用 SSR（Structural Similarity-based Reordering）替代固定顺序或 Hessian 重排序。SSR 的核心动机：三值化对块内权重分布极为敏感——离群列和散乱分布会扭曲三值网格 {−α+μ, μ, α+μ}，导致大量权重被错误映射。SSR 在每次选块时，计算残差矩阵列间余弦相似度，选 top-k 最相似列组成量化块，使块内分布更紧凑、块内方差更小、离群列因彼此相似而不再是离群值。
