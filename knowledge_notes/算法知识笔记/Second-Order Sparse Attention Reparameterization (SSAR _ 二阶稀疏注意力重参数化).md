## Second-Order Sparse Attention Reparameterization (SSAR / 二阶稀疏注意力重参数化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SSAR 是 QuantSparse 论文提出的在推理阶段恢复稀疏注意力信息丢失的技术。背景：稀疏注意力 mask 删除了低值但非零的 attention connections，定义了全注意力与稀疏注意力之差为**一阶残差** Δ^(t)=A_full^(t)−A_sparse^(t)。先前的 cache-based 方法（如 DiTFastAttn）假设 Δ^(t) 在 diffusion 时间步间不变（Δ^(t')≈Δ^(t) ∀t,t'），在参考步缓存后复用。但量化引入噪声 ϵ^(t) 后，一阶残差变化 Δ_quant^(t)=Δ^(t)+ϵ^(t) 的 temporal stability 被破坏（量化噪声使 Δ_quant^(t) 在时间步间不守恒）。SSAR 的核心洞察：**二阶残差** Δ̃_quant^(t)=Δ_quant^(t)−Δ_quant^(t-1) 具有显著更高时间稳定性，因为量化噪声 ϵ^(t) 在 diffusion 过程中呈缓变随机过程，相邻步噪声分布相似 → ϵ^(t)−ϵ^(t-1) 近似平稳。SSAR 缓存一阶+二阶残差，在推理时叠加于稀疏注意力输出之上求近似全注意力，并可选 SVD 投影到 top-r 主成分进一步抑制时间方差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SSAR 推理伪代码（结合 W4A8 量化+稀疏 mask M, cache interval τ=5, SVD rank r=16）：

```
Input: M_quant, prompt P, denoising steps T, interval τ
Output: Generated video Y

Load M_quant, input P
Initialize cache: Δ_cache = None, t_ref = -1

for t = 0 to T-1:
    // Quantized sparse attention forward
    Q_q = Q(X_t)·Q(W_q)^T, K_q = Q(X_t)·Q(W_k)^T, V_q = Q(X_t)·Q(W_v)^T
    A_sq = softmax(Q_q·K_q^T/√d_k ⊙ M)  // Eq. 3, M = SVG spatial-temporal mask

    if t - t_ref ≤ τ and cache is valid:
        // Reuse cached residuals (no full attention recomputation)
        A_approx = A_sq + Δ_cache                         // one mat-add overhead
    else:
        // Refresh: recompute full attention for this step
        A_full = softmax(Q_q·K_q^T/√d_k)                 // no mask
        Δ_q_t = A_full - A_sq                            // first-order residual

        if t_ref ≥ 0:  // have previous reference step
            // Compute and cache second-order residual (Eq. 14)
            Δ̃_q = Δ_q_t - Δ_q_prev                            // second-order
            SVD(Δ̃_q) = S·U·V^T                               // decompose
            Δ̃_q_proj = S_{:,:r} · U_{:r,:r} · V^T_{:,:r}      // project to top-r
            Δ_cache = Δ_q_t + Δ̃_q_proj                        // first+second (Eq. 16)
        else:
            Δ_cache = Δ_q_t                                  // first step: only first-order

        t_ref = t
        Δ_q_prev = Δ_q_t
        A_approx = A_sq + Δ_cache

    // Compute attention output
    Out = A_approx @ V_q
    // → subsequent transformer blocks → denoise step
```

关键参数：cache interval τ=5（每 5 步重新计算全 attention 刷新缓存）, SVD rank r=16。开销：仅额外一次矩阵加法（+0.2% DiT time）, 缓存存储 2× attention output（一阶+二阶合并）, +8-11% GPU memory。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SSAR 在推理阶段与量化稀疏 attention 共同使用。优势：(1) 二阶残差缓存与一阶残差合并存储（额外开销可忽略）；(2) SVD 投影选取 top-r 主成分抑制时间方差，几乎无额外开销；(3) 与 MSAD 互补——MSAD 在 calibration 阶段对齐 attention map, SSAR 在 inference 阶段通过残差校正恢复稀疏化损失；(4) cache interval 可调，interval=3→最高质量, interval=6→最高速度。SSAR 在 Wan2.1-14B W4A8 下将 PSNR 从 14.16 (no reparam) → 17.08 (first-order) → 18.68 (second-order) → 18.72 (SSAR with SVD, top-16)。代码: https://github.com/wlfeng0509/QuantSparse（待发布）。

涉及论文标题：
- QuantSparse Comprehensively Compressing Video Diffusion Transformer with Model Quantization and Attention Sparsification

---
