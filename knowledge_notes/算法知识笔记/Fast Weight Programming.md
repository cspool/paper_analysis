## Fast Weight Programming

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fast Weight Programming (Schmidhuber, 1992; Irie et al., 2021/2022) 是一种区分 "slow weights"（传统梯度下降学习的参数）和 "fast weights"（推理时动态更新的参数）的框架。在 DeltaNet/Gated DeltaNet 中，隐藏状态 S_t 被解释为 fast weight matrix，每步通过 delta rule（等价于在线回归的 SGD 更新）修改：S_{t+1} = S_t - β_t ∇L(S_t)，L(S_t) = 1/2 ||S_t k_t - v_t||²。α_t（forget gate）等价于 adaptive weight decay，β_t 等价于 adaptive learning rate。Gated DeltaNet 的每次前向传播被理解为对 fast weight 执行一步含 weight decay 的 SGD 更新。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Loss(S_t) = 1/2 ||S_t k_t - v_t||²           // 在线回归目标
∇L = (S_t k_t - v_t) k_t^T                    // 回归梯度
S_{t+1} = α_t S_t - β_t ∇L                    // α_t: weight decay, β_t: LR
        = S_t(α_t I - α_t β_t k_t k_t^T) + β_t v_t k_t^T
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该视角的价值在于理论统一（将 Mamba2/DeltaNet/GatedDeltaNet/Longhorn/TTT/Titans 统一在 online learning 下）和设计指导（可通过改进优化器系统地设计新架构）。TTT 和 Titans 在此基础上探索了非线性回归和多步更新。

涉及论文标题：
- Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

---
