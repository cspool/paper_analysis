## Block-wise Post-Training Quantization (逐块后训练量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block-wise Post-Training Quantization（逐块后训练量化）是一种 PTQ 优化策略，将模型按 Transformer Block（或 Layer）为单位分解，逐块进行量化参数校准和优化，而非一次性量化整个模型。其核心动机：全模型端到端 PTQ 的显存开销随模型规模线性增长（需同时存储 FP 和量化模型的所有中间激活），而 block-wise 方式每块仅需当前 block 的输入/输出，显存可控。流程为：对每个 Transformer Block（Attention + FFN），使用少量校准数据作为输入，以该 block 的 FP 输出为教师（target），优化量化 block 的量化参数（scale、rotation matrix、clipping threshold 等）以最小化输出 MSE。优化完成后将 block 的量化参数吸收（fold）到权重中，再处理下一 block。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 S²Q-VDiT 的 block-wise PTQ 流程为例：
```
# Block-wise PTQ Pipeline (基于 GPTQ weight quantizer)
calib_data = select_calibration_samples(model, N=40)  # SDS 筛选

for block_idx, block in enumerate(model.blocks):
    # 1. 收集当前 block 的 FP 输入激活
    X_fp = collect_inputs(calib_data, model, block_idx)  # X_fp ∈ R^{N×n×d}

    # 2. 预计算 attention map（用于 STD token 重加权）
    A = block.attention(X_fp)  # A ∈ R^{H×n×n}

    # 3. 计算 token 重要性权重
    for j in range(n):
        S_j = sum(A[h, i, j] for all h, i)
    λ_j = normalize(S_j, λ_min=0.5, λ_max=1.0)

    # 4. 逐 block 优化量化参数（30 样本，15 epochs）
    for epoch in range(15):
        for x in random_sample(X_fp, 30):
            y_fp = block_fp(x)     # FP block 输出
            y_q = block_q(x)       # 量化 block 输出
            # 重加权的 token-wise MSE
            L = (1/n) * sum(λ_j * ||y_fp[j] - y_q[j]||² for j in range(n))
            L.backward()
            # 更新量化参数（不同参数不同 lr）
            update(scale, lr=5e-3)
            update(rotation_matrix, lr=5e-3)
            update(clip_threshold, lr=5e-2)
    optimizer.step()

    # 5. 吸收量化参数到权重（weight folding），推理时无额外开销
    fold_quant_params_to_weight(block_q)

# 激活量化：推理时 online dynamic per-token quantization
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block-wise PTQ 是当前扩散模型 PTQ 的主流范式（Q-Diffusion, PTQ4DiT, ViDiT-Q, S²Q-VDiT 均采用）。实现上在 PyTorch 中逐 block 注册 hook 收集中间激活，用 AdamW + cosine LR scheduler 优化可学习量化参数。S²Q-VDiT 中 block-wise 优化使用 30 个校准样本（从 40 个 SDS 筛选样本中随机选取），每 block 训练 15 epochs。

涉及论文标题：
- S²Q-VDiT Accurate Quantized Video Diffusion Transformer with Salient Data and Sparse Token Distillation

---
