## Dynamic Base Quantization (DLog, 动态基量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Base Quantization (DLog) 是混合基对数量化方案：大幅值 weight 用 base-√2（细粒度），小幅值 weight 用 base-2（硬件友好）。码本含 n₁ 个 base-√2 码字和 n₂ 个 base-2 码字（n₁+n₂=2^{N-1}-1）。阈值 t 分割区域：|W_ij| ≥ t → base-√2, scale=√2^m, U=n₁-1；|W_ij| < t → base-2, scale=2^{⌊(m-n₁)/2⌋}, U=n₂-1。其中 m = ⌊log_√2(max(|W|))⌉。元素级量化/反量化：Q_W = clamp(⌊-log_B(|W|/S)⌋, 0, U)，Ŵ = S·sign(W) ⊙ B^{-Q_W}，B ∈ {2, √2} per-element。

从算法pipeline角度拆解术语：
```
# DLog Quant/Dequant
m = round(log_sqrt2(max(|W|)))
t = sqrt2^{(m-n1+1)/2 + floor((m-n1)/2)}
for each (i,j):
    if |W[i,j]| >= t:
        B[i,j]=sqrt2; S[i,j]=sqrt2^m; U[i,j]=n1-1
    else:
        B[i,j]=2; S[i,j]=2^{floor((m-n1)/2)}; U[i,j]=n2-1
    Q_W[i,j] = clamp(floor(-log_{B[i,j]}(|W[i,j]|/S[i,j])), 0, U[i,j])
Ŵ = S ⊙ sign(W) ⊙ B^{-Q_W}  # element-wise dequant
```
n₁:n₂ 由 OHS 中 DBS (Dynamic Base Search) 以 block-wise 重建误差最小化为目标搜索。

术语一般如何实现？如何使用？
在 PyTorch 中：逐元素选择 base（torch.where），分别计算 log_2 和 log_√2 对数域索引。DBS 使用离散 grid search 遍历 n₁ 整数值，每次评估块级 Frobenius 重建损失。DLog 在 LOGART 消融中 alone 将 PPL 从 170.64 (pure Log2 RTN) 降至 66.63 (OPT-125M 3-bit)，为最有效的单组件。

涉及论文标题：
- LOGART: PUSHING THE LIMIT OF EFFICIENT LOGARITHMIC POST-TRAINING QUANTIZATION

---
