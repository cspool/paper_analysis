## Discretization in State Space Models (Zero-Order Hold / ZOH)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Discretization（离散化）将连续时间 SSM 参数 (Δ, A, B) 转为离散参数 (Ā, B̄)。连续 SSM: h'(t) = Ah(t) + Bx(t), y(t) = Ch(t)。Zero-Order Hold (ZOH)：假设每步内 x(t) 恒定：Ā = exp(ΔA), B̄ = (ΔA)^{-1}(exp(ΔA)-I)·ΔB。欧拉近似：Ā = I + ΔA, B̄ = ΔB（实践中常用简化）。离散化作用：(1) 连续→离散映射；(2) 保证模型归一化；(3) 赋予分辨率不变性；(4) 在选择机制中，Δ 的输入依赖性通过离散化传递到 Ā, B̄，使递归动态整体成为内容感知。Theorem 1 揭示离散化是 RNN gating 的原则性基础：N=1, A=-1, B=1 时 ZOH 精确给出 gated RNN 形式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# ZOH 离散化 (per timestep, per channel):
Ā_td = exp(Δ_td · A_d)                       # exp(标量×向量), element-wise
B̄_td = (Δ_td·A_d)^{-1}·(exp(Δ_td·A_d)-1)·Δ_td·B_t

# 实践中通常简化为:
Ā_td ≈ exp(Δ_td · A_d)                        # 精确 exp
B̄_td ≈ Δ_td · B_t                            # 一阶 Taylor: lim_{x→0} (e^x-1)/x = 1
```
Mamba 的选择性 SSM 中，离散化是关键桥梁：Δ 的输入依赖性 → Ā = exp(Δ·A) → 整个递归因 Ā 而变化。因此虽然 A 本身不是选择性的，但通过 Δ 的离散化传递，Ā 变得时间可变。Ablation 显示 A 的选择性不是必需的——Δ 的选择性足够。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 Mamba 硬件感知实现中，离散化融合进单一 CUDA kernel，在 SRAM 内完成：加载 (Δ, A, B) → SRAM 计算 Ā=exp(ΔA), B̄=ΔB → 传给 parallel scan → 输出 y。离散化不作为独立步骤物化 Ā, B̄ 到 HBM，节省 O(BLDN) 内存 IO。其他 SSM 变体 (S5, DSS) 使用不同的离散化策略（有时直接参数化 Ā, B̄ 跳过离散化步骤）。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
