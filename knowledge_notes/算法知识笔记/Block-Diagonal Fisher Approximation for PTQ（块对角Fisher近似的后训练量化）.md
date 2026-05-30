## Block-Diagonal Fisher Approximation for PTQ（块对角Fisher近似的后训练量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-Diagonal Fisher Approximation 是 GuidedQuant 提出的 Hessian 矩阵近似策略，用于后训练量化中更准确地估计量化对 end loss 的影响。传统 layer-wise output-based PTQ（如 GPTQ）将所有 output features 视为同等重要，使用 `||XW - XŴ||²` 作为量化目标。SqueezeLLM 使用 diagonal Fisher 近似 `(ŵ-w)ᵀdiag(F)(ŵ-w)`，但忽略所有 off-diagonal 的 cross-weight interactions。GuidedQuant 提出 block-diagonal 近似：保留 Fisher Information Matrix 中每个 output channel 对应的 d_in×d_in 对角块 `F_j = (1/n) Σ(∂ℓ_i/∂w_j)(∂ℓ_i/∂w_j)ᵀ`，忽略跨 channel 和跨层的 off-diagonal 项。这等价于二阶 Taylor 展开中假设 Hessian block-diagonal，捕获了同一 output channel 内权重之间的相互作用，同时保持计算可处理性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Block-diagonal Fisher 在 GuidedQuant 中的计算流程：
```
# 输入：校准数据 X (n × d_in)，权重 W (d_in × d_out)
# 输出：逐层量化权重 Ŵ

# Step 1: 单次 backward pass 计算 end loss 对各层输出的梯度
for each layer l:
    Z_l = X_l @ W_l                          # 层输出 (n × d_out)
    G_l = ∂ℓ/∂Z_l                             # end loss 梯度 (n × d_out)

# Step 2: 计算 block-diagonal Fisher（每 output channel j 一个 d_in×d_in 块）
for each layer l:
    for each output channel j in [1, d_out]:
        # F_j 捕获第 j 个 output channel 内的 cross-weight 交互
        F_j = (1/n) * X_lᵀ @ Diag(G_l[:,j]²) @ X_l    # d_in × d_in
        
# Step 3: 量化目标（等价于 block-diagonal 二次近似）
# min Σ_j (w_j - ŵ_j)ᵀ F_j (w_j - ŵ_j)
```
注意：直接计算和存储所有 d_out 个 d_in×d_in 的 F_j 矩阵对于 LLM 不可行（Llama-2-7B 需 >110TB）。GuidedQuant 通过 averaging approximation（见单独术语）将 d_out 个矩阵按 g 组平均为 g 个共享矩阵来解决。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block-diagonal Fisher 近似在 LLM 压缩中首次被 GuidedQuant (ICML 2025) 扩展到现代 LLM 规模。此前 WoodFisher (NeurIPS 2020) 使用任意大小 B×B 对角块用于 CNN 剪枝，Optimal BERT Surgeon (EMNLP 2022) 用于 BERT 剪枝（B=50），BRECQ (ICLR 2021) 使用 residual block 对应块用于 CNN 量化。GuidedQuant 的关键创新：(1) 块大小 = d_in×d_in（对应 output channel），保留 channel 内全部权重交互；(2) averaging approximation 使存储从 Θ(d_in² d_out) 降至 Θ(d_in² g)，g 为分组数（通常 2-4）；(3) 与任意 layer-wise output-based PTQ 方法（LNQ、QTIP、SpinQuant）可直接集成。

涉及论文标题：
- GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance

---
