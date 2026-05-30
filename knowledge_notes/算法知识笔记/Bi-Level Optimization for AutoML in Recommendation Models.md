## Bi-Level Optimization for AutoML in Recommendation Models

术语是什么？
Bi-Level Optimization（双层优化）是一种优化框架，将模型参数和超参数/架构参数分为两个层级交替优化。在 M3oE 的语境中：外层优化（upper-level）在给定融合权重 α, β 的情况下更新模型参数 W = argmin_W L(W, α, β)；内层优化（lower-level）在模型参数更新后，基于一个 mini-batch 的数据优化融合权重 α, β = argmin_{α,β} L(W* , α, β)。这一方法源自 DARTS（Differentiable Architecture Search, Liu et al. 2018），区别在于 M3oE 搜索的不是网络结构（如卷积核大小、层数），而是融合权重——将 α_d, α_t, β_d, β_t 参数化为可训练标量经 Sigmoid 激活，使其可直接通过梯度下降优化。

从算法pipeline角度拆解术语：
M3oE 的 Bi-Level Optimization 训练流程：
```
初始化: 模型参数 W, 可训练标量 e_αd, e_αt, e_βd, e_βt

for epoch in 1..E:
    // Step 1: 外层更新 (模型参数)
    前向传播计算预测 ŷ_{d,t}
    计算 Loss L = Σ_{d,t} BCE(ŷ_{d,t}, y_{d,t})
    反向传播更新 W (固定 α, β)

    // Step 2: 内层更新 (融合权重)
    取一个 mini-batch 数据
    计算当前 W 下的 Loss
    反向传播更新 e_αd, e_αt, e_βd, e_βt (固定 W)
    更新 α_d = Sigmoid(e_αd), β_d = Sigmoid(e_βd), ...
```
内层更新的计算量很小（仅 4 个标量参数），因此额外开销"trivial"（论文原文）。消融实验（Table 3, w/o AutoML）表明：将融合权重固定为 0.5（等价于无差异化融合）会导致 MovieLens AUC 从 77.02 降至 76.37，KuaiRand-Pure AUC 从 66.37 降至 65.41，验证了自适应权重学习的必要性。

术语一般如何实现？如何使用？
在推荐模型中使用 Bi-Level Optimization 通常涉及以下实现细节：(1) 融合权重不参与常规 optimizer 的更新步骤，而是单独用一个 optimizer（如 Adam）在验证 loss 上优化；(2) 由于架构参数少，内层优化通常在一个 mini-batch 上完成即可（不需要完整 epoch）；(3) 实际部署时权重在训练完毕后固定，推理时无额外开销。此方法不仅适用于 M3oE 的融合权重，也可扩展到其他需要自适应权衡的超参数场景（如多任务 loss 权重）。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework
- MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning (用于优化 Rational Activation 参数 Θ vs LoRA/Router 参数 Ω，inner: Ω=argmin L(D_train,Ω,Θ) lr=1e-4，outer: min L(D_val,Ω*,Θ) lr=1e-6，交替优化；Θ 仅为每层 ~12 scalars)

---
