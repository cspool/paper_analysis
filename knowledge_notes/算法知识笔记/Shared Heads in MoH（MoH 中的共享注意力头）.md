## Shared Heads in MoH（MoH 中的共享注意力头）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shared Heads 是 MoH 架构中始终激活的注意力头子集。在 MoH 中，h 个注意力头被分为两类：前 h_s 个为共享头（shared heads），始终对所有 token 激活；剩余 h − h_s 个为路由头（routed heads），由 router 动态选择 Top-K 激活。共享头的设计动机是：某些注意力头可能捕获跨上下文的通用知识（如语言中的语法规则、视觉中的基础纹理特征），使这些 head 始终激活可减少其他路由头之间的冗余。论文消融实验（Tab.5）表明，添加共享头能将 ViT Acc 从 75.6% 提升到 78.3%（75% 激活，100 epoch）。

从算法pipeline角度拆解术语：
```
# 共享头路由分数计算
s_s = Softmax(W_s @ x_t)          # W_s ∈ R^{h_s×d_in}
# 共享头始终激活，routing score 非零
for i in 1..h_s:
    g_i = α_1 * s_s[i]            # α_1 来自两阶段路由的 head-type 系数

# 共享头计算 attention
for i in 1..h_s:
    H^i = Attention(X @ W_Q^i, X' @ W_K^i, X' @ W_V^i)

# 共享头输出参与加权求和
output += Σ_{i=1}^{h_s} g_i · H^i · W_O^i
```
共享头可视为 Soft MoE (Puigcerver et al., 2024) 的一种形式——所有 token 都经过这些 head，但通过 routing score 加权。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 共享头比例：论文消融（Tab.6）表明，共享头占激活头比例在 13.9%~74.0% 范围内性能稳定（ViT-S, 75% 激活, 100 epoch, Acc 在 78.4%~78.6% 之间），推荐使用较高比例（>40%）。
- LLaMA3-8B Continue-Tuning 中：简单选择每层前 16 个注意力头作为共享头。
- 共享头与 DeepSeekMoE 的 shared experts 概念类似但应用领域不同：DeepSeekMoE 的 shared experts 在 FFN 层，MoH 的 shared heads 在 attention 层。

涉及论文标题：
- MoH: Multi-Head Attention as Mixture-of-Head Attention
