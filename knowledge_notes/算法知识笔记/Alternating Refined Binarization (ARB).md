## Alternating Refined Binarization (ARB)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ARB（Alternating Refined Binarization）是 ARB-LLM（ICLR 2025）提出的迭代二值化精炼算法。传统二值化一次性计算 μ, α, B 后不再调整，导致二值化权重与全精度权重存在分布偏移（残差 R = W - Ŵ 的均值非零）。ARB 通过交替迭代更新三个参数来逐步对齐分布：每轮先更新 μ_refine = μ + mean(R)（修正均值偏移），再解析更新 α = diag(B^T(W-μ))/(Σ(B⊙M)²)（∂L₁/∂α=0 解），最后更新 B = sign(W-μ)。理论保证（Theorem 1）：每轮 L₁^τ = L₁⁰ - m((α^τ)²-(α⁰)²-(μ^τ-μ⁰)²) ≤ L₁⁰。ARB 有两个扩展：ARB-X（引入 calibration data 优化 L₂）和 ARB-RC（双轴缩放消除 μ）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
一阶 ARB 伪代码：
```
ARB(W, M, T):
    μ = mean(W⊙M, dim=1); α = mean(|W-μ|⊙M, dim=1); B = sign(W-μ)
    for iter in 1..T:
        R = W - (α·B + μ); δμ = mean(R⊙M, dim=1); μ += δμ
        α = Σ_j (B·,j⊙M·,j)·(W·,j-μ) / Σ_j (B·,j⊙M·,j)²    # ∂L/∂α=0
        B = sign(W-μ)
    return α·B + μ
```
二阶 ARB（salient weights）：Ŵ = α₁B₁+α₂B₂+μ，更新 α₁,α₂ 后对 {±α₁±α₂} 四候选 binary search 最近邻确定 B₁,B₂。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/ZHITENGLI/ARB-LLM。T=15 参数充分收敛，block_size=128。嵌入 BiLLM 框架：salient 用二阶 ARB，non-salient 用一阶 ARB。LLaMA-7B 上 ARB-RC 1 轮迭代即 ppl=15.23（BiLLM: 49.79），15 轮 ppl=14.03。

涉及论文标题：
- ARB-LLM Alternating Refined Binarizations for Large Language Models

---
