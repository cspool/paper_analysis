## HIPPO Theory / S4D Initialization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
HiPPO（High-order Polynomial Projection Operators, Gu, Dao, et al. 2020）是一种在线信号压缩理论：将输入信号 f(t) 的最优多项式近似投影到正交多项式基上，得到 recurrent 形式的系数更新方程 h'(t) = Ah(t) + Bf(t)。其中 A 矩阵的构造依赖于所选的正交基——最著名的是 HiPPO-LegS，使用指数扭曲的 Legendre 多项式，产生特定结构的 A 矩阵。S4D（Diagonal State Space Models, Gu, Gupta, et al. 2022）在此基础上将 A 矩阵限制为对角形式，并发展了多种初始化方案：S4D-Lin（A_n = -1/2 + iπn，线性间距频率，源于傅里叶基）、S4D-Real（A_n = -(n+1)，纯实数值）、S4D-Inv（A_n = -1/2 + i·N/π·(N/(2n+1)-1)，反比频率）。实部的 -1/2 或负值确保系统稳定（basis 函数被 e^{-t/2} 包络限定），虚部控制振荡频率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Mamba 使用 S4D-Real 初始化（默认）：
```
# S4D-Real: 纯实数值对角初始化
A_n = -(n + 1)  # for n = 0, 1, ..., N-1
# A ∈ R^{D×N}, 所有 N 个元素都是负实数

# S4D-Lin (备选，用于 complex-valued SSM):
A_n = -1/2 + i·π·n  # for n = 0, 1, ..., N-1
# 实部固定 -1/2 (保证稳定性), 虚部线性增长 (振荡频率)
```

Mamba 论文中的 ablation（Table 8, 350M LM）：
- S4D-Lin (complex): perplexity 9.16
- S4D-Real (real): 8.85
- Random init (real, with S4D-Real parameterization): 8.71
- Random init (real, with original Mamba setup): 8.71

结论：在语言建模（离散模态）上，实数值 S4D-Real 和随机初始化均优于传统的复数值 S4D-Lin。这与早期 SSM 工作在音频等连续模态上需要复数值的发现互补——Mamba hypothesis：complex 适合 continuous modalities，real 适合 discrete modalities。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HiPPO 初始化和 S4D 变体在 Mamba 代码库中实现（https://github.com/state-spaces/mamba）。对于大多数 language modeling 应用，S4D-Real 或随机初始化足够。对于音频等连续信号任务，S4D-Lin 或 complex-valued 变体可能更好。S4 和 S4D 的完整理论细节见 Gu, Goel, and Ré (2022) 和 Gu, Gupta, et al. (2022)。S4D-Real 中 A 矩阵作为可学习参数，初始化后通过训练进一步优化。

涉及论文标题：
- Mamba__Linear-Time_Sequence_Modeling_with_Selective_State_Spaces

---
