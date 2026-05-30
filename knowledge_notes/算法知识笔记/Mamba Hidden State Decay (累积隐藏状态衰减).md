## Mamba Hidden State Decay (累积隐藏状态衰减)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mamba隐藏状态衰减是指Mamba SSM中隐藏状态H_t随时间步的指数衰减机制。Mamba的隐藏状态更新公式为 H_t = Ā_t ⊙ H_{t-1} + B̄_t ⊙ X_t，其中 Ā_t = exp(Δ_t ⊙ A) ∈ (0,1)^{d_s×d_e} 是per-channel的衰减因子，A ∈ R_{<0}^{d_s×d_e} 是负的学习矩阵，保证 Ā_t 始终小于1。这意味着每个时间步的历史隐藏状态都会被衰减。在一段序列结束后，初始隐藏状态H_0对最终H_L的贡献被衰减为 ∏_{k=1}^L Ā_k = exp((Σ_{k=1}^L Δ_k) ⊙ A)。由于A<0且Δ_t>0，累加和ΣΔ_k随L增大而增大，使得指数项向零衰减。LongMamba的核心贡献之一就是识别了这一衰减效应作为Mamba长上下文性能瓶颈的根源。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Mamba SSM的衰减机制（Eq.4, 7, 12）
Δ_t = Softplus(X_t) ∈ R_{>0}^{d_e}          # per-channel正步长
A ∈ R_{<0}^{d_s×d_e}                          # 负的可学习矩阵
Ā_t = exp(Δ_t ⊙ A) ∈ (0,1)^{d_s×d_e}         # ⊙: 广播element-wise乘

H_t = Ā_t ⊙ H_{t-1} + B̄_t ⊙ X_t              # 由于Ā_t<1，旧H逐步缩小
∏_{k=1}^L Ā_k = exp((Σ_{k=1}^L Δ_k) ⊙ A)    # A<0 → L变大 → 趋近于0
```
LongMamba分析：全局通道累积衰减足够小（>θ）使其能在训练长度内保持全局信息；但当S≫L时，衰减累积使全局通道无法捕获早期token信息。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
A矩阵初始化为HiPPO或其他结构化矩阵，训练中通过梯度更新。Δ_t通过Linear_proj + Softplus动态生成，是Mamba选择性机制的关键。LongMamba通过token filtering（跳过Δ_t<g的token，设Ā'_t=1）使有效衰减步数≈L而非S。代码：https://github.com/GATECH-EIC/LongMamba。

涉及论文标题：
- LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

---
