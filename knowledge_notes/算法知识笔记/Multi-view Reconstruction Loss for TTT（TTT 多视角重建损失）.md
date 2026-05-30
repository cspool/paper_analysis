## Multi-view Reconstruction Loss for TTT（TTT 多视角重建损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-view Reconstruction Loss 是 TTT 层内循环使用的自监督损失函数。基本形式为 ℓ(W; x_t) = ||f(θ_K x_t; W) - θ_V x_t||²，其中 θ_K x_t 是 training view（被破坏/降维的输入），θ_V x_t 是 label view（重建目标），f 是隐藏状态模型。与传统的 denoising autoencoder 不同，TTT 中的 training view 和 label view 投影矩阵 θ_K 和 θ_V 不是手工设计的，而是通过外循环学习得到的可训练参数。这种设计使自监督任务能端到端地优化为最终的下一个 token 预测目标服务。类似于对比学习中的多视角（multi-view）框架：training view 提供部分信息，模型需要发现维度间的相关性来重建 label view。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Multi-view reconstruction 的计算流程：

```
# 输入 token x_t ∈ R^d

# Step 1: 生成三个视角（learned projections）
train_view = θ_K @ x_t    # training view，通常低秩投影（降维）
label_view = θ_V @ x_t    # label view，重建目标
test_view  = θ_Q @ x_t    # test view，用于生成输出

# Step 2: 计算自监督损失
# 模型 f 从 training view 重建 label view
pred = f(train_view; W)   # f 的当前参数为 W
loss = ||pred - label_view||²  # MSE reconstruction loss

# Step 3: 梯度更新 W
# W_new = W - η · ∇_W loss

# Step 4: 输出 token（使用 test view）
output = f(test_view; W_new)
```

维度关系：training_view 和 label_view 的维度 d' 小于 x_t 的维度 d（低秩投影），test_view 通常保持 d' 维度。θ_K, θ_V, θ_Q ∈ R^{d'×d} 是外循环参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在实际 TTT 层实现中：
- θ_K, θ_V, θ_Q 类似于 self-attention 中的 Key, Value, Query 投影矩阵，但语义不同
- 这些投影通过外循环的 next-token prediction loss 反向传播学习
- TTT-Linear 中 d' = d（全维度投影），TTT-MLP 中类似
- self-supervised loss 的选择（MSE reconstruction）是初步的；论文指出未来工作可能探索更复杂的 self-supervised 任务族

涉及论文标题：
- Learning to (Learn at Test Time): RNNs with Expressive Hidden States
