## T2R (Finetuning Pretrained Transformers into RNNs)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
T2R 由 Kasai et al. (EMNLP 2021) 提出，是最早将预训练 softmax Transformer 转换为 RNN 的方法。核心思路：用可学习 MLP 线性注意力 φ(x)=ReLU(Wx+b)（Q/K 共享 W）替换 softmax，追求近似原始 attention 矩阵。SUPRA 对 T2R 进行了三项关键改进：(1) 分母除法 → GroupNorm；(2) 无位置编码 → RoPE；(3) 无衰减 → 固定 γ。T2R 仅在 ~100M 模型上验证，需约 20% 预训练 tokens，而在 1B 规模 uptraining 时性能崩溃（Table 3）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
T2R: sim(q,k)=φ(q)·φ(k), φ(x)=ReLU(Wx+b)
v'_i = φ(q_i)^T Σ φ(k_j)v_j / φ(q_i)^T Σ φ(k_j)
# 问题: 分母不稳定 + 无位置编码 + 无衰减

SUPRA 改进:
v'_i = GroupNorm(Σ γ^{i-j}·RoPE(φ(q_i))·RoPE(φ(k_j))·v_j)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
T2R 原始实现：https://github.com/jungokasai/T2R（fairseq）。其近似 softmax 的策略在理论上更优雅但实践中受限（需比较完整 attention 矩阵，计算昂贵，不可扩展）。SUPRA 验证直接替换优于近似。

涉及论文标题：
- Linearizing_Large_Language_Models

---
