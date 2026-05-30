## LogN Scaling / SSMax (Scalable Softmax / LogN Trick)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LogN Scaling（SSMax / Scalable Softmax / LogN trick）是对 attention logits 施加与序列长度 $N$ 相关的全局缩放的技术：$L_t = s \log N \cdot S_t$，其中 $s$ 为可学习标量。最早由 Jianlin (2021) 从熵不变性角度提出，后经 Chiang & Cholak (2022)、Nakanishi (2025) 系统研究。动机：随 $N$ 增长，标准 softmax 分布趋于均匀（熵增高），注意力分散到过多无关 token。LogN 通过 $\log N$ 因子放大 logits 使 softmax 更尖锐，在长上下文时保持聚焦。

从算法pipeline角度拆解术语，给出具体例子。

```
# LogN attention forward pass
N = seq_len
scale = s * log(N)     # s 为可学习参数
S = Q @ K^T / sqrt(d)  # attention scores
L = scale * S           # LogN 缩放
A = softmax(L)          # 更尖锐的分布
output = A @ V
```

**核心缺陷**（Scale-invariant Attention 论文指出）：LogN 是位置无关的全局缩放——对近处 token（$t=1-100$）和远处 token（$t=10000+$）施加相同缩放因子。这导致：(1) 局部上下文的总注意力随 $N$ 增长快速衰减；(2) 无法实现"局部稠密 + 全局稀疏"的理想模式。实验显示：LogN+RoPE 在 @64k 时 Val loss=3.378，而 Scale-invariant p-RoPE=3.247。

术语一般如何实现？如何使用？

实现极为简单——softmax 前乘以 $s \log N$。$s$ 通过梯度下降学习或手工设定。与 p-RoPE 组合优于与 RoPE 组合。适用于不需要精细局部注意力控制的场景。

涉及论文标题：
- Scale-invariant Attention

---
