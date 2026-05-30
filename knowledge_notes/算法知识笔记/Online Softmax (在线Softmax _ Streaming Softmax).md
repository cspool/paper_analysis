## Online Softmax (在线Softmax / Streaming Softmax)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Online Softmax 是允许不一次性访问全部输入即可精确计算 softmax 的增量算法，源于 Milakov & Gimelshein (2018)，由 FlashAttention 引入深度学习 attention。核心递推: `m_new = max(m_old, rowmax(S_i))`, `l_new = e^{m_old-m_new}*l_old + rowsum(exp(S_i-m_new))`, `O_new = diag(e^{m_old-m_new})*O_old + exp(S_i-m_new)@V_i`。数学正确性: `exp(S_old-m_new) = exp(S_old-m_old)*exp(m_old-m_new)`。SageAttention 沿用 FlashAttention-2 online softmax，S 计算用 INT8 Matmul，其余保持 FP16。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for i in num_Q_tiles:
    m = -inf, l = 0, O = 0
    for j in num_KV_tiles:
        S = Q̂_i @ K̂_j^T × scales   # INT8 + dequant → FP16
        m_new = max(m, rowmax(S)); P̃ = exp(S - m_new)
        l = exp(m - m_new)*l + rowsum(P̃)
        O = diag(exp(m - m_new))@O + P̃@V_j; m = m_new
    O = diag(1/l) @ O
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CUDA/Triton 中通过 register 维护 running state (m,l,O)。FlashAttention-2 中 l,m 以 log-space 维护避免 overflow。SageAttention 所有中间计算保持 FP16。主流 GPU attention kernel（FlashAttention-2/3, FlashInfer, xformers, SageAttention）均使用 online softmax。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization
