## Post-Softmax Activation Quantization in ViTs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Post-Softmax 激活指 ViT MHSA 中经过 Softmax 的注意力权重矩阵 A∈R^{N×N}。特性：(1) 值域 [0,1] 严格非负；(2) 呈长尾分布——大部分值集中于近零区域，少量值近 1。标准均匀量化器在近零区域量化级别稀疏，导致小注意力值被量化为零，破坏注意力机制的信息传递。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ViT Attention pipeline中的post-Softmax量化位置
Q,K,V = Linear_qkv(LayerNorm(X))
Scores = Q·K^T/√d_k
A = Softmax(Scores)              # post-Softmax激活
A_q = SULQ(A, b)                 # I&S-ViT专用量化器
Output = A_q · V                  # 量化注意力权重×V
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
现有方法：(1) FQ-ViT: log2 量化器，对数变换展平长尾分布；(2) PTQ4ViT: twin uniform quantizer；(3) I&S-ViT: SULQ (shift-uniform-log2)，在 log2 前加 shift bias 后接均匀量化，解决 log2 的量化低效问题。所有 log2 系列量化器推理时通过 bit-shift 执行。I&S-ViT 实验：DeiT-S W3A3 下 SULQ 比 log2 量化器精度高 +3.18%（55.78% vs 52.60% UQ baseline）。

涉及论文标题：
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization

---
