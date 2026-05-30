## Visual Token Sampling (VTS) / Query-Guided Visual Token Sampling（查询引导的视觉Token采样）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Token Sampling (VTS) 是 GroundVTS (CVPR 2026) 提出的核心模块，用于在 Vid-LLM pipeline 中对 visual tokens 进行 query-guided 细粒度采样。VTS 位于 visual encoder + multimodal projector 之后、LLM 输入之前，通过计算每个 visual token 与 text query 的语义相关性，动态选择最 informative 的 visual tokens 送入 LLM。与 uniform frame sampling（对所有帧平等分配 token 配额）和 frame-level query selection（基于外部编码器粗粒度选帧）不同，VTS 在 token 级别进行选择：同一帧内不同空间位置的 token 可因与 query 相关性不同而获得不同的保留权重。VTS 包含两个子操作：Query-Guided Token Scoring（相关性估计）和 Differentiable Top-K Selection（基于 Gumbel-Softmax STE 的可微分选择）。输出非均匀的 visual token 分布——高 query 相关性区域 token 密度高，低相关性区域 token 稀疏或为零。GroundVTS 证明，以一半的 token 预算（ρ=0.5），VTS 超越了全量 uniform baseline（Charades-STA R1@0.7: 34.2 vs 30.5）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
VTS 在 Vid-LLM pipeline 中的完整执行流程（GroundVTS-Q, ρ=0.5, 2 FPS）：
```
# === VTS Pipeline ===
# 输入: 视频帧 {F_t}, 文本查询 text_query
# 参数: W_v, W_q (可学习投影矩阵), τ, τ_g (温度), D_r (隐藏维度)

# 前处理
H_v = VisionEncoder({F_t})      # T frames → N_v visual tokens
V = Projector(H_v)              # MLP → R^{N_v × D}
Q = TextTokenizer(text_query)   # → R^{N_t × D}

# VTS Step 1: Query-Guided Token Scoring
V' = W_v @ V                    # W_v ∈ R^{D × D_r}
q' = W_q @ mean(Q, dim=0)      # W_q ∈ R^{D × D_r}
w = softmax(V' @ q'^T / τ)     # ∈ R^{N_v}, token-query 相关性

# VTS Step 2: Differentiable Top-K Selection
K = ceil(ρ * N_v)               # 保留 K 个 token
g_i ~ Gumbel(0, 1)
z = softmax((log w + g) / τ_g)  # Gumbel-Softmax 松弛
I_K = TopK_indices(w, K)
z_hard = 1[i ∈ I_K]
z_tilde = z_hard + z - stopgrad(z)  # STE

# Weighted Re-encoding
w_hat = exp(w/τ') * z_tilde / sum(exp(w/τ') * z_tilde)
V_selected = w_hat * MLP(V)

# 保留原始位置编码 + 送入 LLM
input = concat([V_selected + PE[I_K], Q])
answer = LLM.generate(input)
```

关键超参数: ρ=0.5 (保留 50% tokens), D_r=512(GroundVTS-Q) / 128(GroundVTS-I)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VTS 作为可训练模块嵌入 Vid-LLM pipeline, 参数约 29-35M (W_v, W_q, MLP_vts)。使用三阶段训练：(1) Stage 1: VTS Warm-up (冻结 LLM, 仅训练 VTS) → (2) Stage 2: Joint LoRA Adaptation (VTS + Projector + LoRA(LLM), LLaVA-Video-178K) → (3) Stage 3: Grounding Fine-tuning (Grounding-FT 70K, VTG 任务)。Gumbel-Softmax + STE 通过 PyTorch 原生 F.gumbel_softmax(hard=True) 实现, 兼容 FlashAttention, 无需自定义 kernel。代码开源: https://github.com/Florence365/GroundVTS。

涉及论文标题：
- GroundVTS__Visual_Token_Sampling_in_Multimodal_Large_Language_Models_for_Video_Temporal_Grounding
