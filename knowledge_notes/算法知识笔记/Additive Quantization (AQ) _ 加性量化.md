## Additive Quantization (AQ) / 加性量化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

加性量化（Additive Quantization, AQ）由 Babenko & Lempitsky (CVPR 2014) 提出，是一种将高维向量压缩为多个码本中码字之和的向量量化方法。给定 d 维向量 $x \in \mathbb{R}^d$，AQ 使用 M 个码本 $\{C_1, C_2, \dots, C_M\}$（每个码本 $C_i \in \mathbb{R}^{d \times h}$ 含 h 个 d 维码字），将 x 近似为每个码本中选出一个码字的和：$x \approx \sum_{i=1}^M C_i b_i$，其中 $b_i$ 是 one-hot 向量从第 i 个码本中选出一个码字。由于码本不要求两两正交（不同于 Product Quantization 需要维度划分），AQ 通常获得比 PQ 更低的量化误差。但编码复杂度为 NP-hard（等价于全连接成对 Markov Random Field 的 MAP 推断），实际操作中常用 beam search 近似编码。

在 CommVQ 中，AQ 被适配用于 KV cache 压缩：(1) 使用学到的编码器（线性层 + 激活函数 + Gumbel-Softmax）替代传统 beam search 编码，实现端到端可微训练；(2) 码本通过梯度下降优化，最小化原始 KV 向量与解码向量间的 MSE loss；(3) 解码简单高效：$\hat{t}_i = s_i C$（二进制序列 s_i 与码本 C 的矩阵乘法）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**CommVQ 中 AQ 的 pipeline 流程**：

```
// 离线训练阶段
calibration_set = collect_kv_cache(FineWeb-Edu, model)
encoder = Linear(d, hidden) + Activation + Linear(hidden, N_c) + GumbelSoftmax
codebook = Parameter(N_c, d)  // 可学习码本

for epoch in range(epochs):
    for k_vec in calibration_set:  // k_vec: [d]
        s = encoder(k_vec)          // s: [N_c], 二进制序列
        k_hat = s @ codebook        // k_hat: [d], 解码重建
        loss = MSE(k_vec, k_hat)
        loss.backward()  // 同时优化 encoder 和 codebook

// 推理阶段 - Prefill
K_prefill, V_prefill = QKV_proj(X_prompt)  // [N, d]
S_K[i] = encoder_K(K_prefill[i])            // 每 token 独立编码
S_V[i] = encoder_V(V_prefill[i])
store(S_K, S_V)  // 替代 FP16 KV cache
```

术语一般如何实现？如何使用？

AQ 码本容量为 $h^M$（指数级），远超 PQ 的 $Mh$。在 CommVQ 中，M=1（单码本），$N_c$（码本行数，即 $h$ 维度）控制压缩率：Avg. bit = $N_c/d$。LLaMA-3.1-8B (d=1024)：$N_c=1024$ 对应 1-bit，$N_c=2048$ 对应 2-bit。在通用场景中，AQ 用于近似最近邻搜索、图像分类（压缩 SIFT/GIST 特征）、向量数据库索引。对于 LLM 推理中的 KV cache 压缩，AQ 对 Value cache 使用标准形式的加法量化，对 Key cache 通过 RoPE-可交换码本变体以高效融入 self-attention。

涉及论文标题：
- CommVQ: Commutative Vector Quantization for KV Cache Compression

---
