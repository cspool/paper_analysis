## δ-approximate Compressor（δ-近似压缩器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
δ-approximate compressor 是分布式优化理论中一类压缩算子的形式化定义：一个算子 $\mathcal{C}: \mathbb{R}^d \to \mathbb{R}^d$ 是 δ-近似压缩器（$\delta \in [0,1]$），如果对任意 $v \in \mathbb{R}^d$ 满足 $\mathbb{E}\|\mathcal{C}(v)-v\|^2 \leq (1-\delta)\|v\|^2$。直观上，δ 越大压缩越精确（δ=1 为无损）。该定义比 unbiased κ-approximate compressor 更广泛（含 biased compressor），可涵盖 top-k sparsifier、top-k low-rank compressor、随机量化（含随机取整 biased variant）等。Remark 4.1 指出任意 κ-approximate unbiased compressor $\mathcal{U}$ 可转换为 $\frac{1}{1+\kappa}$-approximate biased compressor：$\mathcal{C}(v) = \frac{1}{1+\kappa}\mathcal{U}(v)$。SDP4Bit 的 Theorem 4.1 使用 δ-approximate compressor 分析 weight difference compression 的收敛性，放宽了 QSDP 对特定 quantizer 和 Polyak-Łojasiewicz condition 的限制。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SDP4Bit 使用 δ-approximate compressor 分析 weight diff compression 的收敛（Algorithm 4 中的 $\mathcal{C}_w$）：
```
# SDP4Bit SGD with gradient + weight diff compression
for t in range(T):
    # Gradient: unbiased compressor (κ-approximate)
    g_tilde = U_g(g_t)           # E[U_g(v)] = v
    # Weight difference: arbitrary δ-approximate compressor
    w_t = w_{t-1} - η * g_tilde
    delta_tilde = C_w(w_t - w_tilde_{t-1})  # C_w ∈ class of δ-approx compressors
    w_tilde_t = w_tilde_{t-1} + delta_tilde
```
关键：$\mathcal{C}_w$ 可以是 biased 的（δ < 1），因为 weight diff compression 的误差传播受权重差值 $\|\delta w\|$ 而非权值本身 $\|w\|$ 控制，使得误差累积的范围更小（proof via $e_t = w_t - \tilde{w}_t$ 的递推收缩界）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
δ-approximate compressor 概念主要用于理论收敛分析，实践中的实现通常是 symmetric INT quantizer（如 SDP4Bit 的 INT4 group-wise 量化）、top-k sparsifier、或 low-rank compressor。在分析时，需推导具体量化方案对应的 δ 值（如 INT4 对称量化下 $\delta \approx 1 - \frac{1}{12 \cdot 2^{2k}}$），然后将 δ 代入 Theorem 4.1 得到收敛界。SDP4Bit 的实证验证表明 qWD 配合 INT4 group-wise 量化（group_size=2048）实际表现为 δ 充分接近 1 使得训练准确率与全精度几乎无异。

涉及论文标题：
- SDP4Bit: Toward 4-bit Communication Quantization in Sharded Data Parallelism for LLM Training
