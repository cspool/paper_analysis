## AWQ (Activation-aware Weight Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

AWQ (Lin et al., 2023, MLSys 2024 Best Paper) 是一种基于激活感知的 LLM 权重量化方法。核心观察：LLM 权重中并非所有通道同等重要——对应较大激活幅度的权重通道对模型输出贡献更大。AWQ 通过分析校准数据的激活分布，识别重要权重通道，并对其施加更小的量化步长（等价于 per-channel scaling），同时将保护这些通道的代价以数学等效的方式转移到其他通道。

AWQ 的核心技技术：（1）计算激活的 per-channel 幅值（如 L2 norm 或 max），识别显著通道；（2）为显著通道搜索最优缩放因子 s（放大权重、缩小对应激活，或反之），目标是 min_{s} ||Q(W·diag(s)) · (diag(s)^{-1}·X) - WX||_F；（3）基于量化误差的 grid search 确定 s；（4）量化后不保存 s 的副本——s 被融合到量化前后的权重中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# AWQ (simplified)
# W ∈ R^{o×c}, X ∈ R^{b×c} (calibration activations)

# Step 1: Compute per-channel activation statistics
for j in range(c):
    act_norm[j] = mean_abs(X[:,j])  # average activation magnitude

# Step 2: Identify salient channels (top 1%)
salient_mask = act_norm > percentile(act_norm, 99)

# Step 3: Search optimal scaling factors
for j in range(c):
    if salient_mask[j]:
        # Grid search for optimal scale s
        best_s = 1.0; best_loss = inf
        for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            W_scaled = W * s  # scale weight
            X_scaled = X / s  # inverse scale activation
            loss = ||Q(W_scaled) @ X_scaled - W @ X||_F
            if loss < best_loss:
                best_s = s; best_loss = loss
        
        W[:,j] *= best_s  # apply scaling to weight
        # (activations are scaled inversely at runtime)

# Step 4: Standard per-channel quantization on scaled weights
W_q = per_channel_symmetric_quantize(W, bits=4)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

AWQ 开源实现在 https://github.com/mit-han-lab/llm-awq。默认使用 Pile 数据集中的 128 条序列作为校准集。推理时配合 TinyChat/tinychat 或 vLLM AWQ kernel 使用。在 MoEQuant 论文中，AWQ 作为第二个核心 baseline 使用。AWQ 也可被 AGQ 增强——原始 AWQ 对所有 token 等权计算损失，AGQ 引入 gating coefficient c_i 使量化损失变为 L = Σ_i c_i · ||W x_i - W_hat x_i||_F^2。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance
