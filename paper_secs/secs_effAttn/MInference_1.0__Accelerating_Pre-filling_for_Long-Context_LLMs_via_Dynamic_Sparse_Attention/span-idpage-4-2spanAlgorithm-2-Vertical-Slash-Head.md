# <span id="page-4-2"></span>Algorithm 2 Vertical-Slash Head

 $y \leftarrow \operatorname{sparse}(AV, i_{vs})$ 

return y

```
Input: Q, K, V \in \mathbb{R}^{S \times d_h}, k_v, k_s \in \mathbb{N}

# Approximate vertical and slash pattern (last_q = 64)
\widehat{A} \leftarrow \operatorname{softmax}\left(Q_{[-\operatorname{last\_q:}]}K^{\top}/\sqrt{d} + m_{\operatorname{casual}}\right)

# Indices of top k_v vertical line, sum in vertical i_v \leftarrow \operatorname{argtopk}\left(\operatorname{sum}_v(\widehat{A}), k_v\right)

# Indices of top k_s slash line, sum in slash i_s \leftarrow \operatorname{argtopk}\left(\operatorname{sum}_s(\widehat{A}), k_s\right)

# Build sparse attention index i_{vs} \leftarrow \operatorname{sparseformat}(i_v, i_s)

# Final dynamic sparse attention scores (only index block)

A \leftarrow \operatorname{softmax}\left(\operatorname{sparse}(QK^{\top}, i_{vs})/\sqrt{d}\right)

# Sparse mixed scores and values
```

