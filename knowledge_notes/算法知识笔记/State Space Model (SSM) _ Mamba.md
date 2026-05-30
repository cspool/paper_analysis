## State Space Model (SSM) / Mamba

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SSM 是序列建模框架，通过 $h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t$ 递归编码历史信息。Mamba (Gu & Dao, 2023) 引入输入依赖性选择，$B_t = W_B x_t$, $C_t = W_C x_t$, $\Delta_t = \text{Softplus}(W_\Delta x_t)$，使模型选择性传播或遗忘信息。Mamba-2 (Dao & Gu, 2024) 统一 SSM 与 attention。核心优势：训练用 parallel scan（O(N)），推理用 recurrent（O(1) 内存）。关键局限：常量 size state 导致 recall 弱（300M recall acc 19.23% vs Transformer 39.98%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Mamba 推理（recurrent mode, O(1) 内存）
h = zeros(d_state)
for t in 1..N:
    Δ_t = softplus(W_Δ @ x_t)      # 输入依赖步长
    Ā_t = exp(Δ_t * A)             # 离散化
    B̄_t = Δ_t * (W_B @ x_t)        # 输入投影
    h = Ā_t * h + B̄_t * x_t        # state update
    y_t = (W_C @ x_t) @ h           # output
# 仅需维护 h ∈ R^{d_state}，无需存储 per-token 状态
```

在 Hymba 中，Mamba 作为 SSM heads 提供全局 context 摘要，与 attention heads 并行。SSM 的高效推理特性使得可以激进地用 SWA 替代 global attention（仅 3 层保留），因为 SSM 已总结全局信息。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：https://github.com/state-spaces/mamba, mamba.py, FLA。适合：(1) 长序列（1M+ tokens, O(1) 推理内存）；(2) hybrid 架构的全局上下文组件（Hymba, Jamba, Samba, Zamba）；(3) 端侧高效推理。

涉及论文标题：
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Hymba: A Hybrid-head Architecture for Small Language Models
- Quamba2: A Robust and Scalable Post-training Quantization Framework for Selective State Space Models
