## TUPE (Transformer with Untied Positional Encoding / 解耦位置编码Transformer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

TUPE 是 Ke et al. (2020) 提出的位置编码方案，将 self-attention 中的 content 相关性和 position 相关性解耦为两个独立项。传统 Transformer 将 positional encoding 与 word embedding 相加后一同输入 attention（$A = (X+P)W_Q W_K^T (X+P)^T$），导致 content-content、content-position、position-content、position-position 四种交互混合。TUPE 将 attention 分解为：$A = \text{softmax}(X W_Q W_K^T X^T + P U_Q U_K^T P^T)$，content 和 position 使用各自独立的 Q/K 投影矩阵（$W_Q, W_K$ vs $U_Q, U_K$），去除交叉项噪声。

论文 5B MoE 模型使用 TUPE attention，来自 Kim et al. (2021) 的 DeepSpeed MoE 配置。

从算法pipeline角度拆解术语：

TUPE 在 MoE Transformer 层中的计算：
```
输入: x ∈ R^{S×D}, position_ids ∈ Z^S

# Content attention
Q_c, K_c = x @ W_Q, x @ W_K           # 标准 Q/K 投影
A_c = Q_c @ K_c^T / sqrt(d)            # content-content score

# Position attention（独立参数）
pos = learnable_pos_embed[position_ids]
Q_p, K_p = pos @ U_Q, pos @ U_K       # position 专属 Q/K
A_p = Q_p @ K_p^T / sqrt(d)            # position-position score

# 组合（无交叉项）
A = softmax(A_c + A_p)                 # 仅 content-content + position-position
output = A @ (x @ W_V)                 # value 投影
```

优势：(1) 去除 content-position 交叉项噪声，attention 更专注于语义内容；(2) position 学习独立的 Q/K 参数，更好捕获绝对和相对位置关系；(3) 对不同长度序列更鲁棒；(4) 可分别处理 content 和 position 的 Q/K。

术语一般如何实现？如何使用？

Ke et al. (2020) 开源实现在 GitHub（https://github.com/guolinke/TUPE）。Kim et al. (2021) 将其集成到 DeepSpeed MoE 训练框架。在 PyTorch 中修改标准 `nn.MultiheadAttention`：分别计算 content attention 和 position attention，softmax 前相加。

涉及论文标题：
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production
