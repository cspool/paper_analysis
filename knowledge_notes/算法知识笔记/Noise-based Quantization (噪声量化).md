## Noise-based Quantization (噪声量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Noise-based Quantization（噪声量化）是一种将量化误差建模为随机噪声的量化模拟方法，训练时向权重添加采样噪声（w+ε）替代确定性量化值（ŵ），在反向传播中通过 STE 或直接对噪声更新梯度。其数学模型：给定权重 w 和量化步长 Δ，量化误差 ε = w − clamp(⌊w/Δ⌉, −2^{b-1}, 2^{b-1}−1)·Δ 服从 U[-Δ/2, Δ/2] 均匀分布。使用 w+ε 的前向传播，损失函数的期望值为：E[L(w+ε)] ≈ E[L(w) + ε·∇_w L(w) + ½·εᵀ·∇²_w L(w)·ε] ≈ E[L(w) + ½·εᵀ·∇²_w L(w)·ε]，因为 E[ε]=0 消除一阶项。结果：损失隐式惩罚 ∇²_w L(w)（Hessian 迹），驱动权重收敛到更平坦的损失区域。该技术最早由 NICE (Baskin et al. 2021) 和 DiffQ (Défossez et al. 2022) 提出，Bit-Shrinking (Lin et al. 2023) 将其引入 PTQ 结合 sharpness-aware scheduling。HDRQ 是首个将噪声量化引入模型合并框架的方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
HDRQ 中噪声量化的 block-wise reconstruction 过程：
```
for iter in range(1, 20001):
    # 1. 确定性量化
    w_hat = clamp(round(w/Δ), -2^{b-1}, 2^{b-1}-1) * Δ
    
    # 2. 噪声采样（前 16500 iter 使用噪声模拟）
    if iter <= 16500:
        ε = w - w_hat                     # U[-Δ/2, Δ/2]
        w_train = w + ε                   # 噪声版本权重
    else:
        w_train = w_hat                   # 切换 fake quantization
    
    # 3. 前向 + 损失
    O_hat = block_forward(x, w_train)
    L_rec = ||O_hat - O_fp||₂²            # 重建损失
    L_dist = λ * ||w_src - w_train||₂²    # 距离正则 (λ=5e-2)
    L = L_rec + L_dist
    
    # 4. 更新权重 w
    optimizer.step()                       # Adam, LR=0.001, cos annealing
```
关键设计：(1) 噪声量化在绝大多数迭代（16500/20000）中生效，确保充分的 Hessian 正则化；(2) 最后 3500 迭代切换到确定性 fake quantization，此时学习率已很小，不会破坏正则化效果；(3) 配合距离正则化项（w_train 到源权重的 ℓ₂ 距离）同时控制权重 divergence。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
噪声量化在 PyTorch 中通过自定义 autograd function 或 fake_quantize 模块实现。典型实现：(1) 在 forward 中计算 deterministic quantized value ŵ；(2) 计算 ε = w − ŵ；(3) 使用 straight-through estimator (STE) 在 backward 中将梯度从 w+ε 回传到 w。HDRQ 的噪声量化与标准噪声量化方法的区别在于额外增加了距离正则项和切换到 fake quantization 的阶段性策略。噪声量化的优势：不需要 STE 近似梯度（如果噪声是加性的而非乘性的）、天然支持平坦极小值搜索、理论上有 Hessian 平滑保证。局限性：(1) 训练过程中的噪声方差影响收敛速度；(2) 噪声模拟和真实量化之间存在 gap（通过最后切换到 fake quantization 缓解）。

涉及论文标题：
- Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation
