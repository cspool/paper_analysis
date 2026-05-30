## RMoE (Layerwise Recurrent Router for MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

RMoE (Recurrent Router for Mixture-of-Experts) 是一种在 MoE 路由过程中引入跨层循环依赖的 router 设计。核心思想：当前 MoE 中不同层的 router 独立工作，各自仅基于本层 hidden state 做出路由决策，缺乏跨层协调。RMoE 在每层 router 前插入一个跨层共享的轻量级 Gated Recurrent Unit (GRU)，将各层路由决策串联为一个序列——第 i 层的路由结果依赖于第 i-1 层及之前所有层的 GRU 隐状态：

1. **逐层投影**：x_i' = Proj_i(x_i)，将 hidden state x_i ∈ R^h 降维到 GRU 状态维度 R^p（p=128），每层使用独立的 Proj_i（因为不同层的 hidden state norm 和方差差异大）
2. **跨层 GRU**：h_i = GRU(x_i', h_{i-1})，其中 GRU 跨层共享参数，h_i 携带前 i-1 层路由决策的历史信息
3. **路由决策**：score_i = softmax(h_i @ G_i)，top-k gating 选择 experts
4. **标准 MoE 计算**：y_i = sum_n g_n(h_i; G_i, k) * E_n(x_i)

RMoE 的四个核心特性：(1) 跨层信息共享——GRU 显式传递历史路由决策，使当前层 router 知道 token 在之前层被分配到哪些 experts；(2) 额外梯度路径——GRU 提供跨层反向传播的 Recurrent Gradient，辅助 router 优化；(3) 正交兼容——GRU 路由作为一个新计算阶段，可与 XMoE、DeepSeekMoE 等现有方法组合；(4) 开销可忽略——对 0.91B 模型仅增加 ~3.5M 参数，训练速度仅降低 0.4%，GPU 内存仅增加 1.4%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

RMoE 在 decoder-only transformer 中的完整前向流程：

```
# 初始化：跨层共享 GRU 参数 W_s, U_s, W_z, U_z, W_h
# 每层独立：Proj_i (linear, h→p), Router G_i (linear, p→N)
h_0 = zeros(p)  # p=128

for layer i in 1..L:
    # Step 1: Attention (标准)
    x_i = Attention_i(LayerNorm_attn(x_{i-1})) + x_{i-1}

    # Step 2: RMoE (替代标准 FFN/MoE)
    x_i_norm = LayerNorm_moe(x_i)

    # 2a: 逐层投影降维
    x_i_prime = Proj_i(x_i_norm)  # [B, L, h] → [B, L, p]

    # 2b: GRU 跨层循环
    s_i = sigmoid(W_s @ x_i_prime + U_s @ h_{i-1})   # reset gate
    z_i = sigmoid(W_z @ x_i_prime + U_z @ h_{i-1})   # update gate
    h_tilde = tanh(W_h @ x_i_prime + s_i * (W_h @ h_{i-1}))
    h_i = (1 - z_i) * h_tilde + z_i * h_{i-1}       # [B, L, p]

    # 2c: 基于 GRU 输出的 routing
    gating_scores = softmax(h_i @ G_i)               # [B, L, N]
    topk_val, topk_idx = topk(gating_scores, k)

    # 2d: 稀疏 Expert 计算
    y_i = zeros_like(x_i_norm)
    for n in topk_idx:
        y_i += topk_val[n] * Expert_n(x_i_norm)

    # Step 3: 残差连接
    x_i = y_i + x_i
```

**消融变体**：
- **RMoE + NP (Not Passing)**：将 h_i = GRU(x_i', h_0)，取消跨层 recurrence（GRU 变为 stateless），与 RMoE 参数量相同但性能大幅下降（Enwiki8 BPC 1.141→1.150），验证跨层 recurrence 是主要贡献者
- **RMoE + detach h_{i-1}**：detach h_{i-1} 阻止其梯度计算，仅保留前向信息共享。性能比 RMoE-NP 更差（1.159 vs 1.150），证明 Recurrent Gradient（反向梯度流）比前向信息共享更重要
- **RMoE + NP + r-α**：将上一层的 routing logits 作为残差加到当前层：g_i = softmax(h_i @ G_i) + α * softmax(h_{i-1} @ G_{i-1})。性能接近 RMoE-NP，不能提供有效的跨层信息。detach 后性能大幅下降，再次验证额外梯度的重要性

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **代码开源**：https://github.com/qiuzh20/RMoE
- **小模型实现**：基于 PyTorch 原生，8 层 decoder-only transformer (hidden=352, 16 experts top-2)，1×A100 约 21 小时训练
- **大模型实现**：基于 Megablocks 框架（block-sparse kernel），24 层 Llama-style (hidden=1280, RoPE + SwiGLU + RMSNorm, 16 experts top-4 fine-grained)，8×A100 约 5 天 pre-training
- **关键超参数**：GRU hidden dim p=128（p=256/512 在大规模设置下性能下降）；使用逐层独立 Proj_i 而非共享投影器（共享 Proj 性能更差，因为不同层 hidden state 分布差异大）；GRU 优于 RNN 和 LSTM
- **SFT 时冻结策略**：冻结 GRU 和线性 router 层，或仅冻结 router
- **Load Balance Loss 权重**：0.01（与标准 SMoE 相同）

涉及论文标题：
- Layerwise Recurrent Router for Mixture-of-Experts
