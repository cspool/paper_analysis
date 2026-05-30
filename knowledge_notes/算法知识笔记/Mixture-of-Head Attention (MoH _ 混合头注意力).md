## Mixture-of-Head Attention (MoH / 混合头注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixture-of-Head Attention (MoH) 是一种将 Mixture-of-Experts (MoE) 机制引入 multi-head attention 的新型注意力架构。MoH 将每个 attention head 视为 MoE 框架中的 expert，通过一个可学习的 router 为每个输入 token 动态选择 Top-K 个 attention head 进行激活，并将标准 MHA 中的等权求和替换为加权求和。MoH 不增加注意力头数量，参数总量与标准 MHA 可比（router 带来的额外参数量极小，约为 O(h·d_in)）。MoH 在 ViT（图像分类）、DiT（图像生成）和 LLM 上均验证有效，可在仅激活 50%~90% 注意力头的情况下达到或超越标准 MHA 的性能。代码开源：https://github.com/SkyworkAI/MoH（Apache 2.0），发表于 ICML 2025。

从算法pipeline角度拆解术语：
MoH layer 的计算流程（h 个 head，h_s 个共享 head，K 个路由 head 激活）：
```
# Input: X ∈ R^{T×d_in}, X' ∈ R^{T'×d_in}
# 参数: W_Q^i, W_K^i, W_V^i, W_O^i (per head i)
# Router: W_s ∈ R^{h_s×d_in}, W_r ∈ R^{(h-h_s)×d_in}, W_h ∈ R^{2×d_in}

# Step 1: Router 计算共享head分数
s_s = Softmax(W_s @ x_t)           # [h_s] per token

# Step 2: Router 计算路由head分数
s_r = Softmax(W_r @ x_t)           # [h-h_s] per token

# Step 3: Top-K 选择路由head
topk_indices = TopK(s_r, K)        # 选择分数最高的K个路由head

# Step 4: 两阶段系数
[α_1, α_2] = Softmax(W_h @ x_t)   # 平衡共享/路由head贡献

# Step 5: 组装routing score g_i
for i in 1..h_s:     g_i = α_1 * s_s[i]                    # 共享head
for i in h_s+1..h:   g_i = (i ∈ topk_indices) ? α_2 * s_r[i-h_s] : 0  # 路由head

# Step 6: 仅激活g_i≠0的head计算attention
for i where g_i ≠ 0:
    Q_i = X @ W_Q^i, K_i = X' @ W_K^i, V_i = X' @ W_V^i
    H^i = Softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i

# Step 7: 加权求和输出
MoH(X, X') = Σ_{i=1}^{h} g_i · H^i · W_O^i

# Step 8: Load Balance Loss（仅对路由head）
P_i = mean(s_r[i-h_s])              # token选择head i的平均概率
f_i = mean(1[token选择head i])      # head i被选择的实际比例
L_b = Σ_{i=h_s+1}^{h} P_i · f_i     # 鼓励均匀路由

# 总loss: L = L_task + 0.01 * L_b
```

关键参数：
- h: 总注意力头数
- h_s: 共享头数（始终激活）
- K: 每 token 激活的路由头数（Top-K）
- 激活比例 = (h_s + K) / h
- 激活预算在各层可不均匀分布：浅层激活较少 head，深层激活较多 head（论文中 TransNeXt 设置）
- β = 0.01（load balance loss 权重）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 训练方式：(1) 从头训练——MoH 可直接替换标准 MHA 进行训练，router 与模型其他参数同时学习，需添加 Load Balance Loss 防止 routing collapse；(2) Continue-Tuning——预训练 MHA 模型（如 LLaMA3-8B）可转换为 MoH 模型，关键技巧包括：参数无关 router（用 ℓ₂ norm of query 作为 routing score）、straight-through estimator 量化 routing score 保持输出分布稳定、两阶段训练（第一阶段适配数据分布，第二阶段切换为 MoH）。
- 推理加速：将 Q/K/V 特征通过 router mask 转为稀疏矩阵，用稀疏矩阵乘法替代 dense 矩阵乘法。序列越长优势越大（seq=512 时 50% head 激活比 MHA 快 37.3%）。
- 代码开源：https://github.com/SkyworkAI/MoH，基于 Skywork-MoE 训练框架。预训练模型在 HuggingFace：Chat-UniVi/MoH-ViT-*、Chat-UniVi/MoH-DiT-*、Chat-UniVi/MoH-LLaMA3-8B。
- 与 MoA（Mixture of Attention Heads, Zhang et al. 2022）的区别：MoH 不增加参数、引入 shared heads 和 two-stage routing、支持 continue-tuning、在 ViT/DiT/LLM 多框架验证。

涉及论文标题：
- MoH: Multi-Head Attention as Mixture-of-Head Attention
