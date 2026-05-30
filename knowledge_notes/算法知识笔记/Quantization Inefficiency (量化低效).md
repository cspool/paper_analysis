## Quantization Inefficiency (量化低效)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Quantization Inefficiency 是 I&S-ViT 命名的问题，指 log2 量化器在量化和 post-Softmax 激活时，代表性范围无法覆盖完整输入域。例如 post-Softmax 激活范围 [1.08e-8, 0.868]，-log₂ 输出 [0, 26]，但 3-bit 仅覆盖 [0, 7]，导致 [8, 26] 段全部 clamp 到 7。大量远离零的值被强制映射到同一远端级别，失去区分度。由于 post-Softmax 中大量接近零的值，此问题严重影响模型精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 3-bit log2量化器
X ∈ [1.08e-8, 0.868], s = 0.868/7 ≈ 0.124
Y = -log2(X/s) ∈ [0, 26]  # 26 >> 7
Y_q = clamp(round(Y), 0, 7)
# [8,26]→7, X_hat=9.69e-4, 所有X<9.69e-4映射到同一值

# SULQ方案
Y = -log2(X+η), η=0.01 → Y∈[0,19]
Y_q = UQ(Y, 3)  # 8级别均匀覆盖[0,19]，无clamp损失
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
检测方法：统计 -log₂(X/s) 的输出范围，若 max > 2^b-1 则存在量化低效。SULQ 通过在 log2 输入前添加 shift bias η 压缩输出范围，再通过均匀量化器替代 clamp 实现完整覆盖。η 通过 grid search 最小化 MSE 来确定。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---
