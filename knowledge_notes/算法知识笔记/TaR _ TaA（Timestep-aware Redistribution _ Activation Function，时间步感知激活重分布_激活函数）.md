## TaR / TaA（Timestep-aware Redistribution / Activation Function，时间步感知激活重分布/激活函数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
TaR 和 TaA 是 BI-DiffSR 提出的扩散模型时序感知二值化参数。扩散模型的 T 步迭代去噪中，激活分布随 timestep 剧烈变化，静态的 bias 和 RPReLU 无法适配所有 timestep。TaR/TaA 受 MoE 启发，设置 K 对 (bias^(i), RPReLU^(i))，将总 timestep T 均分为 K 组，每 timestep 仅激活对应组参数：`i = floor(K*t/T)`。这等价于将长时序分割为短区间，降低了单组参数的适配难度。BI-DiffSR 中 K=5, T=2000。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: x_input ∈ [H,W,C], timestep t, T=2000, K=5
i = floor(K * t / T)                 # i ∈ {0,1,2,3,4}
x_shifted = x_input + bias[i]         # TaR: 仅第 i 组 bias
x_bin = Sign(x_shifted)               # 1-bit 激活
w_bin = (||w||_1/n) * Sign(w)         # 1-bit 权重
x_conv = bit_count(XNOR(x_bin, w_bin)) # 1-bit 卷积
x_act = RPReLU[i](x_conv)             # TaA: 仅第 i 组 RPReLU
x_out = x_act + x_input               # shortcut
```
消融：同时用 TaR+TaA PSNR=32.66dB，单独任一反降低（仅 TaR=29.27dB, 仅 TaA=29.13dB vs 无=31.99dB）。K=5 已足够（PSNR 从 K=1=31.99→K=2=32.42→K=5=32.66）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 实现：`bias ∈ R^{K×C}` 和 `RPReLU ∈ R^{K×C}` 为 learnable Parameter，推理时 `i=int(K*t/T)` 索引选择。总参数量仅 K×C×2=640 vs 4.58M。设计灵感源于 MoE 稀疏激活——多组参数，每步仅选 1 组。适用于任何多步迭代模型（扩散、流模型）中分布随时间变化的场景。

涉及论文标题：
- Binarized Diffusion Model for Image Super-Resolution

---
