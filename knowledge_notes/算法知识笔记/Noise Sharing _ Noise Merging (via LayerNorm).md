## Noise Sharing / Noise Merging (via LayerNorm)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Noise Sharing (Noise Merging) 是 QeRL 提出的零开销噪声注入技术。将 AQN 的 additive Gaussian noise Z_noisy 等价融入 RMSNorm scale w：w_noise=Z_noise+w。数学等价：X·(Z_noisy+Ŵ) = X·Z_noisy+X·Ŵ，通过 RMSNorm 变换得 X_norm·((Z_noisy/w+I)^T⊙Ŵ)，即 additive noise → row-wise multiplicative noise on weight。优势：(1) 零参数开销；(2) 零额外 flops；(3) 不破坏 Marlin NVFP4×BF16 kernel 兼容性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Noise Sharing 操作
w_noise = w + Z_noisy                  # [d], zero overhead
x_norm = x / sqrt(mean(x²)+ε)
x_scaled = w_noise ⊙ x_norm           # 等效乘法噪声
output = x_scaled·\hat{W}^T + LoRA    # Marlin kernel accelerated
# 噪声共享：Q/K/V → 同 RMSNorm；gate/up → 同 RMSNorm
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/NVlabs/QeRL。仅适用于 Pre-LN Transformer (RMSNorm→Linear)。不创建独立噪声向量，修改已有 RMSNorm weight——在训练每步采 Z_noisy 并更新 w_noise，前向完成后恢复 w_orig。核兼容：噪声不写入量化权重，保持 packed 4-bit layout 不变。

涉及论文标题：
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs
