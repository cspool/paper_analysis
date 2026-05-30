## Parameter Salience in LLM PTQ (δ)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Parameter Salience（参数显著性 / δ）是 LLM 后训练压缩中的权重重要性度量，定义为量化某个权重元素对层输出产生的二阶误差。基于 SparseGPT (Frantar & Alistarh, 2023) 的形式化：在二次近似 loss 下 L(Ŵ) = tr((Ŵ-W)H(Ŵ-W)^T)，消除 (i,j) 位置元素引入的误差为 δ_{i,j} = w_{i,j}² / [H⁻¹]_{j,j}²。其中 H = (1/P) Σ x^{[k]T} x^{[k]} 是校准数据激活的 Hessian 近似，H⁻¹ 通过 Cholesky 分解高效计算。δ 结合了权重幅值和激活二阶信息：||w|| 大的元素不一定显著（若对应 channel 激活小），而激活 outlier channel 上的权重即使很小也可能高 δ。SliM-LLM 揭示由于 activation outlier channels 始终出现在固定位置（token sink 现象），权重 δ 沿 channel 方向呈现 spatial clustering，这正是 group-wise 混合精度的理论基础。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: 权重W (n×m), 校准激活x (t×m)
# 输出: 每个元素的salience δ_{i,j}

H = (1/t) * x^T · x              # Hessian近似 (m×m)
H_inv = Cholesky((H + λI)^(-1))  # 逆Hessian Cholesky因子
inv_diag = diag(H_inv)^2          # 对角元平方

for j in range(m):
    δ[:,j] = W[:,j]² / inv_diag[j]²

# 简化近似 (λ极小时): δ_{i,j} ≈ (W_{i,j} · ||x_j||²₂)²
```
Theorem 1 (SliM-LLM)：激活 outier channel p 使 ||x_p||₂  >> ||x_j||₂, ∀j≠p，导致 H_{p,p} >> H_{j,j}，在 λ ≤ e^{-1} 下 δ_{:,p} >> δ_{:,k}。定理连接了 activation outliers → Hessian 对角 dominance → channel-wise salience clustering 的逻辑链。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
δ 在 LLM 压缩中的主要用途：(1) SparseGPT 稀疏化——按 δ 排序选保留权重；(2) PB-LLM/SpQR 元素级混合精度——按 δ 阈值分离高/低精度权重；(3) SliM-LLM group 级混合精度——group 平均 δ 决定 bit-width；(4) BiLLM 显著权重识别——δ 指导结构化二值分割。计算开销主要在 Cholesky 分解 O(m³)，但 per-layer 仅一次。使用 128 个 2048-token 校准样本即可获稳定 Hessian 估计。

涉及论文标题：
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models
- BiLLM Pushing the Limit of Post-Training Quantization for LLMs

---
