## MLP Kernel for Linear Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MLP Kernel 是线性注意力中替代固定非线性特征图（如 ELU+1）的可学习特征映射。SUPRA 中定义为 φ(x)=ReLU(Wx+b)，W∈R^{D×D} 在 queries 和 keys 间共享。比传统固定 kernel 更强大——MLP 可学习适合特定任务的特征表示。T2R 也使用类似 MLP kernel，但追求近似 softmax；SUPRA 直接替换，学到与 softmax 完全不同的 attention 模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 固定 kernel (Katharopoulos 2020):
phi(x) = ELU(x) + 1, sim(q,k) = phi(q)·phi(k)

# MLP kernel (SUPRA):
phi(x) = ReLU(W @ x + b)  # W 共享于 Q/K
phi_q = RoPE(phi(q)), phi_k = RoPE(phi(k))
sim(q,k) = phi_q · phi_k
```

参数开销：每 head 的 W∈R^{d_h×d_h}，总约 D²/h per layer。7B 模型（D=4096, h=32, d_h=128）：128²×32×32≈16M 总参数（~0.2%）。W 初始化为接近小随机值，使初始 φ(x)≈ReLU(x)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现于 OpenLM fork (https://github.com/TRI-ML/linear_open_lm)。Square matrices with biases，保持 Q/K 特征维度不变。共享 W 保证相似度的对称性。

涉及论文标题：
- Linearizing_Large_Language_Models

---
