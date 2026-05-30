## Learnable Logarithmic Rounding (LLR, 可学习对数舍入)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learnable Logarithmic Rounding (LLR) 是 LOGART 提出的首个对对数域 PTQ 实现可学习舍入的技术。现有对数 PTQ 均使用 RTN（⌊·⌋ 或 ⌊·⌉）直接舍入到最近量化级别。LLR 将舍入决策参数化：用 floor ⌊·⌋ 作为下界，引入可学习变量 R（每个 weight 一个），通过 sigmoid σ(R) ∈ (0,1) 控制每个 weight 是向下还是向上舍入。Quant: Q_W = clamp(⌊-log_2(|W|/s)⌋ + σ(R), 0, 2^{N-1}-1)，Dequant: Ŵ = s · sign(W) ⊙ 2^{-Q_W}。优化目标：min_R E[||ΔW·X||_F²] + λ·Σ(1-|2σ(R)-1|^β)，正则项鼓励 σ(R) 逼近 0 或 1（hard rounding）。LLR 梯度：∂L/∂R = 2s·ln2 · M_c ⊙ 2^{-Q_W} ⊙ sign(W) ⊙ [(WX-ŴX)X^T] ⊙ σ'(R) + λ·∂f_reg/∂R。与线性可学习舍入（AdaRound）的关键区别：梯度包含指数项 2^{-Q_W}，小幅值 weight 梯度小、大幅值 weight 梯度大，与对数分布密度结构一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
# LLR (Learnable Logarithmic Rounding)
R = nn.Parameter(torch.zeros_like(W))  # per-weight learnable variable
opt = Adam([R], lr=0.05→0.015, CosineAnnealingLR)
for iter in range(500):  # LLM, 2000 for Vision
    # Soft quantize (HAF noise injected in forward pass)
    Q_W = clamp(floor(-log_B(|W|/(s_of·S))) + σ(R), l_a, U)
    Ŵ = S * sign(W) * B^{-Q_W}
    # Loss: reconstruction + regularization
    L_recon = ||(W - Ŵ) @ X||_F²
    L_reg = λ * Σ(1 - |2σ(R)-1|^β)
    L = L_recon + L_reg
    L.backward()  # auto-diff through quant chain
    opt.step()
# Hard round after convergence
Q_final = clamp(floor(-log_B(|W|/(s_of·S))) + round(σ(R)), l_a, U)
```

术语一般如何实现？如何使用？
LLR 在 PyTorch 中实现：R 为 nn.Parameter，σ(R) 用 torch.sigmoid(R)。前向传播 soft quantize，loss 计算 Frobenius 范数重建误差。Adam + CosineAnnealingLR。LOGART 开源: https://github.com/logart-lab/logart。LLR 需反向传播因此比 RTN 慢，但离线 PTQ 阶段完成一次后推理使用 hard-quantized 权重。LLR 与 OHS 有强协同：OHS 先建立优质网格，LLR 在其上收敛更快更优（OHS+LLR 500 iters > 纯 LLR 2000 iters）。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---
