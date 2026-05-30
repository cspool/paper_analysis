## Pre-Quantization Fine-tuning with Fused Rectifier (PQFR)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pre-Quantization Fine-tuning with Fused Rectifier (PQFR) 是 SPR²Q 提出的量化预处理机制，其核心思想是在量化前将可训练的低秩 rectifier 模块（ΔW = BA）的权重增量融入骨干网络，使得模型在量化前主动吸收补偿信息。与普通 PTQ 仅优化量化器参数不同，PQFR 修改权重使其对量化更鲁棒。具体流程：(1) 对每个需要量化的权重矩阵 W，引入低秩增量 ΔW = BA（A∈ℝ^{r×d_in}, B∈ℝ^{d_out×r}, r=8）；(2) 前向计算 W' = W + ΔW，然后伪量化 W_q' = Q_{a,b}(W')；(3) 使用 STE 反向传播，联合优化 rectifier 参数 (A,B) 和量化器裁剪界 (a,b)；(4) 损失函数为 L_pixel + λ·L_feature（像素级 L1 重建 + 逐块特征 L2 对齐）；(5) 训练完成后 fusion：W_final ← W + ΔW。与 QLoRA 区别：QLoRA 将 LoRA 适配器添加到量化后的模型用于下游任务微调，PQFR 将 rectifier 融入量化前的模型以改善量化质量。与 PTQ1.61 的 Restorative LoRA 类似但 PQFR 扩展为 rectifier group + 静态路由。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# PQFR: 对单个 Mamba 模块的量化预处理
W = module.weight  # 冻结的原始 FP32 权重
A = nn.Parameter(torch.randn(r, d_in))  # 可训练低秩矩阵
B = nn.Parameter(torch.randn(d_out, r))
a, b = nn.Parameter(l_init), nn.Parameter(u_init)  # 可训练裁剪界

for iter in range(12000):  # Adam, lr=1e-2, Cosine Annealing
    X = get_batch(DF2K)
    delta_W = B @ A  # 低秩增量
    W_prime = W + delta_W  # 补偿后的权重
    
    W_clipped = clamp(W_prime, a, b)
    s = (b - a) / (2^4 - 1)  # 4-bit 量化步长
    W_q = round((W_clipped - a) / s) * s + a  # 量化-反量化
    
    Y_q = X @ W_q  # 量化模型前向
    
    L_pixel = L1(Y_q, Y_fp)  # 像素级重建
    L_feature = sum(L2(φ_l(Y_q), φ_l(Y_fp)) for l in all_blocks)
    loss = L_pixel + λ * L_feature
    
    loss.backward()  # STE 近似 round 导数
    update(A, B, a, b)
```
仅使用 PQFR（无 RGT/OSRC）：Set5 PSNR 从 37.20 提升至 37.44 (+0.24 dB)，Urban100 从 30.69 提升至 31.25 (+0.56 dB)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
基于 PaddlePaddle 框架实现，训练 12,000 iterations (batch=8)，优化器 Adam (lr=1e-2, Cosine Annealing)，rectifier rank r=8。训练完成后 rectifier 被融合到 FP32 权重中——无额外推理参数、无额外 FLOPs、无动态结构修改。

涉及论文标题：
- SPR²Q: Static Priority-based Rectifier Routing Quantization for Image Super-Resolution
- PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization (Restorative LoRA 同类预处理范式)

---
