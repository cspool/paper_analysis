## Relative Positional Bias (Transformer-XL Style)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Relative Positional Bias 是 Transformer 中的一种位置编码方法（Dai et al. 2019, Transformer-XL），在 attention score 计算中直接添加一个仅依赖于相对位置 (i-j) 的可学习偏置项，替代绝对位置编码嵌入到 input 中的方式：

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}} + B_{rel}\right)V$$

其中 $B_{rel}[i, j] = b_{i-j}$，$b$ 是一个可学习的 bias table。相对距离通常 clip 到 [-k, k] 范围。

在 GLaM 中，每层维护独立的 per-layer Relative Positional Bias，替代标准绝对位置编码。这使模型能更好地处理变长序列和长距离依赖。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GLaM Attention + Relative Positional Bias
# 输入: x [B, S, M=8192], nheads=128, dhead=128
# bias_table: [2k+1, nheads] 可学习参数

scores = Q @ K^T / sqrt(dhead)  # [B, nheads, S, S]

# 为每个 (i, j) 查相对位置 bias
for i, j in range(S):
    dist = clip(i - j, -k, k)
    bias[i, j] = bias_table[dist + k]  # [nheads]

scores += bias
output = softmax(scores) @ V
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：预计算相对距离矩阵 [S, S]，查 bias table 后 broadcast 到 attention scores。优势是天然支持任意长度外推（超出范围的相对距离被 clip）。T5 使用简化的 bucket 版本（32 个 bucket），GLaM 使用 per-layer 独立 table。在现代框架中通常通过 `torch.nn.Embedding` 存储 bias table。

涉及论文标题：
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
