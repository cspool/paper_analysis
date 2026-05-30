## Two-Stage Routing in MoH（MoH 中的两阶段路由）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Two-Stage Routing 是 MoH 中用于动态平衡共享头和路由头权重的路由策略。与标准 MoE 的单一 router 不同，MoH 的路由分两阶段：(1) Stage 1——分别计算每个 head 的 token 级分数：共享头用 W_s 投影，路由头用 W_r 投影；(2) Stage 2——通过 W_h 投影产生 α_1 和 α_2 两个 head-type 级别的系数，动态调整共享头和路由头对最终输出的贡献比例。消融实验（Tab.5）表明，两阶段路由在共享头基础上进一步提升性能（ViT Acc 78.3%→78.6%, DiT FID 69.54→69.42）。

从算法pipeline角度拆解术语：
```
# Input: token x_t ∈ R^{d_in}

# Stage 1: 逐头分数
s_s = Softmax(W_s @ x_t)          # 共享头分数 [h_s]
s_r = Softmax(W_r @ x_t)          # 路由头分数 [h-h_s]

# Stage 2: Head-type 系数
[α_1, α_2] = Softmax(W_h @ x_t)  # W_h ∈ R^{2×d_in}, α_1+α_2=1

# 最终 routing score
for i in 1..h_s:     g_i = α_1 * s_s[i]          # 共享头
for i in h_s+1..h:   g_i = (Top-K选中的i) ? α_2 * s_r[i-h_s] : 0
```
两阶段路由的设计直觉：α_1/α_2 让模型根据 token 内容决定是更多依赖共享头（通用知识）还是路由头（专用知识）。例如，对于简单/常见 token，α_1 可能更大；对于需要特定领域知识的 token，α_2 可能更大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- W_h 仅为 2×d_in 的投影矩阵，参数量极小。
- 两阶段路由与标准 MoE routing 的主要区别：标准 MoE 仅有一个 router 产生所有 expert 的分数；MoH 的两阶段路由显式分离了"head 选择"和"head-type 权重平衡"两个决策层次。
- 论文未提供 α_1/α_2 在不同任务/类别下的详细分析（仅在 Appendix D 中可视化了 routing score 分布，指出共享头的 routing score 在不同类别间变化更大）。

涉及论文标题：
- MoH: Multi-Head Attention as Mixture-of-Head Attention
