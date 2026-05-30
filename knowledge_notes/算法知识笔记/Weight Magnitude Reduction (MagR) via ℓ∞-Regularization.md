## Weight Magnitude Reduction (MagR) via ℓ∞-Regularization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Magnitude Reduction (MagR) 是一种基于 ℓ∞-regularized 最小二乘优化的权重预处理技术，在 PTQ 量化之前应用，目的是缩小预训练权重的 channel-wise 最大绝对值（ℓ∞ 范数），从而降低量化步长 δ = (max(w)−min(w))/(2^b−1)，减少量化误差。MagR 的核心优化问题为：min_w ½‖Xw − Xŵ‖² + α‖w‖∞，其中 X 是校准数据的特征矩阵，ŵ 是预训练权重，α 是惩罚参数。MagR 的关键洞察是：LLM 各层的特征矩阵 X 是近似秩亏的（fraction rank 均值 70-84%，最低仅 0.1%），因此 X 的核空间非平凡，存在许多 w 满足 Xw ≈ Xŵ 但 ‖w‖∞ 远小于 ‖ŵ‖∞。MagR 利用这一自由度在核空间中寻找 ℓ∞ 范数最小的解。不同于 AWQ、OmniQuant、QuIP 等线性变换方法需要推理时对特征施加逆变换 T⁻¹ 产生推理开销，MagR 是非线性变换，直接替换权重，推理时零开销。MagR 被 NeurIPS 2024 接收。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MagR 在 overall PTQ pipeline 中的位置：

```
# ====== 离线预处理（一次性，每个 linear layer） ======
输入: 预训练权重 Ŵ ∈ R^{m×n}, Hessian H = XᵀX ∈ R^{m×m}
超参数: K=150, α=10⁻³ (per-channel) / 10⁻⁴ (per-group), η = 1/λ_max(H)

W⁰ = Ŵ
for k = 0 to K-1:
    V^k = W^k - η · H · (W^k - Ŵ)                    # 梯度下降步
    # Proximal step via Moreau decomposition:
    # prox_{ηα‖·‖∞}(V^k) = V^k - ηα · proj_{‖·‖₁≤1}(V^k/(ηα))
    for j = 1 to n:                                   # 每列独立
        v_j = V^k[:, j]
        u = sort(|v_j|, descending)                   # O(m log m)
        ρ = max{i: u_i > (Σ_{r=1}^i u_r - ηα) / i}    # 找阈值索引
        θ = (Σ_{r=1}^ρ u_r - ηα) / ρ                  # 软阈值
        W^{k+1}[:, j] = sign(v_j) ⊙ max(|v_j| - θ, 0)
输出: 预处理权重 W' = W^K

# ====== 量化（与标准 PTQ 相同） ======
δ = β · (max(W'_col) - min(W'_col)) / (2^b - 1)   # β ∈ [0.80, 0.95]
W_q = δ · clamp(round(W'/δ) - z, 0, 2^b-1) + z·δ

# ====== 推理 ======
Y = X @ dequant(W_q)    # 零额外开销
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MagR 基于 PyTorch 实现，开源地址：https://github.com/AozhongZhang/MagR。基于 OPTQ 仓库构建，采用 block-wise 加载（每次 7 个 linear layer 到 GPU）。校准数据：WikiText2 128 个 2048-token 序列。关键参数：迭代数 K=150，α=10⁻³（per-channel）/ 10⁻⁴（per-group），步长 η=1/λ_max(H) 保证收敛，β ∈ [0.80, 0.95]（与 bit-width 正相关）。预处理时间：LLaMA2-7B ~15 min，13B ~30 min，70B ~3.5 hr（单 A100 80GB）。预处理后的权重可直接用于任何标准 PTQ 方法（RTN、OPTQ、QuIP），无需修改推理代码。per-group 变体：将 V ∈ R^{m×n} reshape 为 R^{d×(m·n/d)} 后独立做 ℓ₁-ball 投影，梯度步不变。

涉及论文标题：
- MagR: Weight Magnitude Reduction for Enhancing Post-Training Quantization

---
