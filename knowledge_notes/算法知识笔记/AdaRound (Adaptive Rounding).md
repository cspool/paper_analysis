## AdaRound (Adaptive Rounding)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AdaRound (ICML 2020) 是后训练量化权重舍入优化方法。将舍入建模为二值优化：每个权重决定"向上舍入"还是"向下舍入"，通过连续软舍入变量 V（sigmoid→[0,1]）最小化重建损失，用 STE 梯度回传优化。V 初始化为 0（对应 RTN），经微调获得更好舍入决策。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
W_int = floor((W - z) / s)
h = sigmoid(V)                                  # [0,1] 软舍入
W_hat = s * (W_int + h) + z                     # 软量化
L = APH_loss(layer(W_hat, X), O_fp)
dL/dV = dL/dW_hat * s * sigmoid'(V)             # STE 梯度
V = V - lr * dL/dV
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
AdaRound 在 BRECQ/QDrop/APHQ-ViT 中作为 block 重建核心优化变量。典型配置：lr_weight=1e-3，Adam 优化器。不改变量化参数（scale/zp），仅优化舍入方向，不增加推理开销。

涉及论文标题：
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- D2-DPM Dual Denoising for Quantized Diffusion Probabilistic Models

D2-DPM 使用 AdaRound 作为扩散模型 PTQ 的权重量化器，固定首尾层为 8-bit，其余层用 AdaRound 量化至目标位宽。

aespa 将 AdaRound 的 layer-wise 重构目标替换为 attention-wise 重构目标：对 W_Q 使用 `loss = tr(E[K^TK]·ΔW_Q·E[XX^T]·ΔW_Q^T)`、W_K 使用 `loss = tr(E[Q^TQ]·ΔW_K·E[XX^T]·ΔW_K^T)`、W_V 使用 `loss = tr(ΔW_V·E[XA^TAX^T]·ΔW_V^T)`，替代原始的 `loss = ||WX - W̃X||^2`。参数设置：2000 iterations, lr=0.015, rounding loss weight λ=1.5。
