## Shift-Uniform-Log2 Quantizer (SULQ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shift-Uniform-Log2 Quantizer (SULQ) 是 I&S-ViT 提出的专门用于 ViT post-Softmax 激活的量化器，旨在解决标准 log2 量化器的"量化低效"(quantization inefficiency)问题。SULQ 的量化公式为：X_q = UQ(-log₂(X + η), b)，即先在 log2 函数的输入上添加 shift bias η，将激活值平移后做 log2 变换，再对输出应用均匀量化器 (UQ)。反量化过程为：X̄ = 2^{-round(D-UQ(X_q))} - η，其中对去量化输出做 round 确保整数输出，使得推理时可用硬件友好的 bit-shift 操作。相比标准 log2 量化器，SULQ 仅增加一次 round 操作和两次加法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SULQ 在 ViT block 中的量化流程（以 3-bit 为例）：
```
输入: X ∈ R^{N×N} (post-Softmax attention map, 值域 ~[10^{-8}, 1])

# Step 1: 确定 η 参数
η_best = argmin_η MSE(X, D-SULQ(SULQ(X, b, η), b, η))
# 通过 grid search 从候选取使量化误差最小的 η

# Step 2: SULQ 量化 (推理时)
Y = -log2(X + η)          # shift + log2 变换
Y_q = UQ(Y, b)            # 均匀量化到 b-bit 整数 [0, 2^b-1]

# Step 3: SULQ 反量化
Y_fp = D-UQ(Y_q)          # 均匀反量化到浮点
Y_int = round(Y_fp)       # round 确保整数,用于 bit-shift
X_hat = 2^{-Y_int} - η    # 2 的负指数幂 + 反 shift
```
关键性质：(1) SULQ 通过均匀量化器完整覆盖输入域，不会像 log2 量化器那样有大量值被 clamp 到远端；(2) SULQ 保持对接近零区域的细粒度 bit 分配，同时对接近 1 的区域分配稀疏 bit，匹配 post-Softmax 的长尾分布；(3) η 参数可调节量化点的分布模式。I&S-ViT 实验：DeiT-S W3A3 时 SULQ 单独贡献 +17.34% 准确率提升（从 3.36% 到 20.70%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 PyTorch 中定义 `sulq_quantize(x, b, eta)` 函数，内部调用 `torch.log2(x + eta)` 后接标准均匀量化。η 在优化前通过 grid search 选取使 MSE 最小的值。推理时 SULQ 通过 bit-shift 操作执行 2^{·} 计算，与 log2 量化器硬件效率相同。I&S-ViT 将 SULQ 专门用于 post-Softmax 激活。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---
