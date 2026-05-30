## Video DiT (Video Diffusion Transformer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Video DiT（Video Diffusion Transformer）是将Diffusion Transformer架构应用于视频生成的模型范式。核心流程：(1) VAE编码器将原始视频clip压缩为latent表示（如stability-ai VAE做8×8空间下采样），得到shape为[F, H, W]的latent token grid，F/H/W分别为frames、高度、宽度方向的token数；(2) 对latent表示加入扩散噪声（DDPM或flow matching范式），带噪latent与conditioning（timestep、text prompt等）一起输入DiT模型；(3) DiT模型由多个DiT block堆叠而成，每个block包含self-attention模块（捕获视频token间的时空依赖）和cross-attention模块（对齐text prompt）。Self-attention有两种范式：interleaved spatial-temporal attention（交替在spatial和temporal维度做attention，计算高效但信息捕获不足）和3D full attention（所有token间全对全attention，质量最佳但O(S²)复杂度）。跨模态对齐通过cross-attention实现，其中Q来自视频token（S个），K/V来自text prompt（<120 tokens，复杂度远低于self-attention）。

从算法pipeline角度拆解，Video DiT训练的pipeline：
```
# Video DiT training pipeline (flow matching variant)
for each training step:
    # 1. VAE encoding
    latent = VAE.encode(video_clip)  # [F, H, W] latent grid
    
    # 2. Noise injection (flow matching)
    t ~ Uniform(0, 1)
    noise ~ N(0, I)
    z_t = t * latent + (1-t) * noise  # 线性插值路径
    
    # 3. Text encoding
    text_emb = TextEncoder(text_prompt)  # ~120 tokens
    
    # 4. DiT forward
    h = z_t + timestep_embedding(t)
    for block in DiT_blocks:
        # Self-attention (3D full attention)
        h = SelfAttn(LN(h)) + h           # O(S²d), S = F*H*W
        # Cross-attention (text conditioning)
        h = CrossAttn(LN(h), text_emb) + h  # O(S·T·d), T < 120
        # FFN
        h = FFN(LN(h)) + h
    velocity = output_proj(h)
    
    # 5. Flow matching loss
    loss = MSE(velocity, latent - noise)  # predict velocity field
```

典型配置：0.8B (28层, 12头, head size 96)、2.7B (32层, 16头, head size 128)、30B (42层, 24头, head size 256)。SOTA模型如Meta MovieGen、HunyuanVideo、CogVideoX均采用类似架构。self-attention在长序列下占>90%训练时间（200K tokens时forward 92%、backward 93%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现：基于LLM的transformer架构改造，主要在attention范式（spatial-temporal vs full 3D）和conditioning方式（timestep embedding + text cross-attention）上与LLM不同。主要框架包括HunyuanVideo（Tencent, GitHub开源）、OpenSora（潽方AI, GitHub开源）、CogVideoX（智谱, GitHub开源）、MovieGen（Meta, 闭源）。训练使用Adam optimizer, lr=1e-4, gradient checkpointing, flow matching或DDPM范式。开源实现通常基于PyTorch + FSDP/DeepSpeed分布式训练。

涉及论文标题：
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training

## Flow Matching

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Flow Matching（流匹配）是一种生成建模范式，替代传统的DDPM（Denoising Diffusion Probabilistic Models）用于训练扩散模型。与DDPM学习从噪声中预测原始数据或噪声不同，Flow Matching学习一个连续的velocity field（速度场）v(x, t)，该速度场定义了从简单分布（如标准高斯噪声）到数据分布的连续归一化流（Continuous Normalizing Flow, CNF）。训练时沿noise→data的线性/最优传输路径采样中间状态z_t = (1-t)·noise + t·data，模型学习预测速度v(z_t, t) = data - noise。推理时从噪声x_0 ~ N(0,I)开始，用ODE solver（如Euler method）沿学习的速度场逐步积分到达数据点：x_{t+Δt} = x_t + v(x_t, t)·Δt。Flow Matching的核心优势：(1) 比DDPM更简单的训练目标（直接预测速度而非噪声/原始数据，虽然数学上等价）；(2) 更灵活的前向过程（可用最优传输路径而非固定高斯扩散过程）；(3) 结合rectified flow可在少量ODE步骤（如<10步）内实现高质量采样。

从算法pipeline角度拆解，Flow Matching在Video DiT中的使用：
```
# Flow Matching training (DSV paper)
# 给定: latent video z ~ p_data, noise ε ~ N(0,I)

# Forward: 定义线性概率路径
t ~ Uniform(0, 1)
z_t = (1-t) * z + t * ε              # 最优传输路径
target_velocity = ε - z               # 速度场目标

# DiT预测速度
v_pred = DiT(z_t, t, text_emb)        # 模型输出=预测速度场

# Loss
L = MSE(v_pred, target_velocity)      # 简单MSE，无需noise schedule
```

Flow matching loss直接反映模型能力和训练进展（DSV论文和MovieGen论文均使用此特性评估收敛）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Flow Matching论文（Lipman et al., 2023, ICLR 2023; Tong et al., 2023, NeurIPS 2023）提供了理论基础。主流实现：Stable Diffusion 3 (Esser et al., 2024)使用rectified flow matching实现4-8步高质量采样；FLUX.1系列模型基于flow matching；Meta MovieGen (Polyak et al., 2024)使用flow matching训练视频DiT；SD3.5系列。PyTorch实现：扩散模型中简单的训练范式切换——将DDPM的noise prediction loss替换为velocity prediction loss，无需架构修改。推理时需ODE solver（如Euler, RK4, DPM-Solver）替代DDPM的DDIM采样。

涉及论文标题：
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training

## Critical KV Pairs (Dynamic Attention Sparsity)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Critical KV Pairs指在attention计算中对给定query贡献最大的key-value pairs子集。定义：对query q和所有KV pairs S_q = {(k_i, v_i)}，attention score A(q, k_i) = softmax(q·k_i)，critical KV pairs I_q ⊆ S_q为那些A(q, k_i)超过θ-百分位阈值的pairs。DSV论文使用cumulative sum threshold θ=90%，即top KV pairs的attention score之和占总score的90%。整个attention head的sparsity定义为所有queries上非critical KV pairs的平均比例：E_{q~Q}[|S_q\I_q| / |S_q|]。

从算法pipeline角度拆解critical KV pairs的识别和使用：
```
# Critical KV identification (DSV's concept)
# 给定: Q, K, V ∈ R^{S×d_k}

# Step 1: 计算attention scores (conceptual - DSV用低秩近似替代)
scores = Q @ K^T / sqrt(d_k)        # [S, S] - DSV避免物化此矩阵
attn = softmax(scores, dim=-1)       # [S, S]

# Step 2: 识别critical KV pairs per query
for q in range(S):
    sorted_scores, indices = sort(attn[q], descending=True)
    cumsum = cumsum(sorted_scores)
    k = argmin(cumsum >= 0.9)         # θ=90% cumulative sum
    critical_indices[q] = indices[:k]  # top-k KV indices

# Step 3: Sparse attention - 仅对critical KV计算
for q in range(S):
    K_crit = K[critical_indices[q]]   # gather critical KV
    V_crit = V[critical_indices[q]]
    O[q] = softmax(Q[q] @ K_crit^T / sqrt(d_k)) @ V_crit
```

DSV发现Video DiT中的关键特性：(1) attention scores服从power-law分布（少数KV贡献大部分score）；(2) critical KV pairs不具局部性（仅15.1%在5-token半径内），无法用固定窗口模式近似；(3) 稀疏度在attention heads间和training steps间高度异质且动态变化；(4) 相邻token的critical KV pairs高度重叠（2×2×2 3D cube内>92.4%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

识别critical KV的方法分为：(1) 预训练静态方法——LLM推理中用window+token sink（StreamingLLM）、Heavy-Hitter Oracle (H2O)；Minference使用预定义的profile-based attention patterns；(2) 在线预测方法——DSV使用低秩sparsity predictor在线预测（训练阶段）；(3) block-based方法——BLASST的block-level sparsity via softmax thresholding。预定义方法适用于LLM推理（有明确的局部性模式），在线预测方法适用于Video DiT训练（critical KV无局部性模式）。DSV方法的冗余损失极小（>98% attention score被预测的critical KV覆盖）。

涉及论文标题：
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training

## Sparsity Predictor (Low-Rank Attention Score Approximation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparsity Predictor是DSV中用于在线预测attention score分布并识别critical KV pairs的轻量级组件。每个self-attention模块配备两个额外的低秩可训练矩阵W_Q^lr和W_K^lr（形状d × d_lr，其中d_lr ≪ d_k，如d_k=128时d_lr=16），将输入投影到远小于原始Q/K的低维空间：Q_lr = H @ W_Q^lr, K_lr = H @ W_K^lr。用低秩乘积Q_lr·K_lr^T近似原始attention score分布QK^T（注意：近似pre-softmax的QK^T而非softmax后的值，因为softmax单调，pre-softmax的相对顺序和softmax后的相对顺序一致）。预测器参数量极小（3B模型<10M参数）。

从算法pipeline角度拆解，predictor的训练和使用：
```
# Predictor 结构
W_Q_lr ∈ R^{d × d_lr}    # low-rank query projection, d_lr << d_k
W_K_lr ∈ R^{d × d_lr}    # low-rank key projection

# Forward: predict attention scores
Q_lr = H @ W_Q_lr        # [S, d_lr]
K_lr = H @ W_K_lr        # [S, d_lr]
approx_scores = Q_lr @ K_lr^T   # [S, S] - 近似QK^T分布

# Predictor training loss (detached from main graph!)
L_approx = 0.95 * CosLoss(approx_scores, Q @ K^T) \
         + 0.05 * NormLoss(approx_scores, Q @ K^T)
# CosLoss: cosine similarity, 保持相对大小关系
# NormLoss: L2 norm difference, 保持整体scale一致
# 从主计算图detached → predictor gradient不影响DiT参数

# Stage 2: 使用predictor识别critical KV
K_per_query = ceil((1 - sparsity_head) * S)
crit_indices = FusedTopK(approx_scores, k=K_per_query)  # [H, S, K]
O = SparseAttention(Q, K, V, crit_indices)
```

设计动机：(1) 避免物化完整attention score矩阵（O(S²)内存）；(2) 在fused kernel中完成低秩MM和top-K选择，中间不物化完整矩阵；(3) 不影响fused attention kernel（FlashAttention）的优化路径；(4) 低秩近似带来了约O(S·d_lr)的额外计算开销，远小于O(S²)的attention计算本身。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现：每个attention module增加两个nn.Linear(d, d_lr)（无bias），predictor参数独立于主模型参数（不同步梯度），需手动管理predictor的parameter replication和gradient synchronization（因为FSDP可能切分不同参数）。训练策略采用sample-based方法减少query计算量（随机采样部分query进行loss计算）。Cosine loss + Norm loss组合相比MSE对attention score的scale变化更robust。训练到avg(L_approx) < 0.01后进入Stage 2。

涉及论文标题：
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training

## Speculative Decoding（投机解码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Speculative Decoding 是一种加速 LLM 自回归推理的算法，不改变输出分布，保证与原始大模型完全一致的生成结果。其核心流程：(1) 使用一个计算成本低的小模型（draft model $M_q$）自回归生成 $\gamma$ 个候选 token；(2) 将 prefix $\sigma$ 与 $\gamma$ 个候选 token 拼接，送入大模型 $M_p$ 进行一次 forward pass；(3) 对比 $M_q$ 和 $M_p$ 在每个 token 位置的 logits，按某种准则（通常为 rejection sampling）接受或拒绝候选 token；(4) 若某个 token 被拒绝，从该位置起用 $M_p$ 重新采样。整个过程仅需一次大模型 forward pass 即可验证 $\gamma$ 个 token，而标准的自回归解码需要 $\gamma$ 次 forward pass。若接受率 $\alpha$ 高，则实际加速比接近 $\gamma$·$\alpha$。典型实现可达到 2-3× 加速（对 T5X/Chinchilla 70B）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Speculative Decoding Algorithm
def speculative_decode(prefix, M_q, M_p, gamma):
    # Stage 1: Draft phase
    draft_tokens = []
    current_prefix = prefix
    for i in range(gamma):
        token = M_q.autoregressive_step(current_prefix)
        draft_tokens.append(token)
        current_prefix = current_prefix + [token]
    
    # Stage 2: Verify phase (single forward pass)
    full_seq = prefix + draft_tokens
    logits_p = M_p.forward(full_seq)     # [len(full_seq), vocab_size]
    logits_q = M_q.forward(full_seq)     # [len(full_seq), vocab_size]
    
    # Stage 3: Accept/Reject
    accepted = []
    for i in range(gamma):
        pos = len(prefix) + i
        p_dist = softmax(logits_p[pos])
        q_dist = softmax(logits_q[pos])
        # Rejection sampling
        if random() < min(1, p_dist[draft_tokens[i]] / q_dist[draft_tokens[i]]):
            accepted.append(draft_tokens[i])
        else:
            # Rejected: resample from adjusted distribution
            adjusted = max(0, p_dist - q_dist)
            adjusted = adjusted / sum(adjusted)
            bonus_token = sample(adjusted)
            accepted.append(bonus_token)
            break
    return prefix + accepted
```

加速比分析：若大模型单步推理时间为 $T_p$，小模型为 $T_q$（$T_q \ll T_p$），接受率为 $\alpha$，则每轮期望生成 token 数为 $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$，理论加速比为 $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma T_q + T_p)}$。

典型变体：Medusa（无需辅助小模型，通过预训练多个预测头同时预测多个 token）、Draft & Verify（跳过中间层替代独立小模型）、SpecTr（扩展候选 token 数量）、SpecInfer（云端多 draft model）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源实现广泛集成于主流框架：TensorRT-LLM 支持 speculative decoding、HuggingFace TGI 提供原生支持、vLLM 通过 draft model API 支持。使用时需准备一个与 target model 同 vocab 的小 draft model（如 LLaMA-68M 搭配 LLaMA-7B）。SpecExec 进一步将 speculative decoding 应用于消费级设备，通过将大模型参数 offload 到 RAM/SSD，在 4-bit 量化下运行 50B+ 模型达 4-6 tok/s。Apple 的 Speculative Streaming 将 drafting 融入 target model 本身（修改微调目标从 next-token prediction 到 future n-gram prediction），消除对独立 draft model 的需求。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models

## LoRA (Low-Rank Adaptation / 低秩适应)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LoRA 是一种参数高效微调（PEFT）方法，由 Hu et al.（2021）提出。核心思想：冻结预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，注入可训练的低秩分解矩阵 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$（$r \ll \min(d, k)$），将权重更新约束为低秩形式：

$$h = W_0 x + \frac{\alpha}{r} BA x$$

其中 $\alpha$ 为缩放超参数，$r$ 为秩（通常 2-64）。仅 $A$ 和 $B$ 可训练，参数量从 $d \times k$ 降至 $r(d+k)$，在 $r=8, d=k=4096$ 时可减少 >99% 可训练参数。推理时 $BA$ 可融合回原权重：$W = W_0 + \frac{\alpha}{r} BA$，无额外推理延迟。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# LoRA Forward Pass (per linear layer)
def lora_forward(x, W_0, A, B, alpha, r):
    # W_0 frozen, A and B trainable
    # A init: Kaiming uniform, B init: zeros
    h_base = W_0 @ x              # frozen pretrained pathway
    h_lora = (alpha / r) * (B @ A) @ x  # low-rank adaptation pathway
    return h_base + h_lora

# Training: only A and B receive gradients
# Inference: fuse B@A into W_0, then just W @ x

# QLoRA variant: add 4-bit quantization
def qlora_forward(x, W_0_quantized, A_16bit, B_16bit, alpha, r):
    W_0_dequant = dequant(W_0_quantized)  # NF4 → BF16
    h_base = W_0_dequant @ x
    h_lora = (alpha / r) * (B_16bit @ A_16bit) @ x
    return h_base + h_lora
    # Key: double quantization of quantization constants for further memory saving
```

关键变体：QLoRA（将 W_0 量化为 4-bit NF4，再在 BF16 下训练 LoRA 参数，使 65B 模型可在单 48GB GPU 上微调）、DoRA（将预训练权重分解为 magnitude 和 direction 分量分别微调）、PiSSA（使用 SVD 初始化 A/B，将大奇异值分配给可训练矩阵加速收敛）、LoRA+（为 B 和 A 设置不同学习率）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HuggingFace PEFT 库（https://github.com/huggingface/peft）提供了 LoRA 标准实现，支持对 Transformer 的 Q/K/V/O 投影和 FFN 层注入 adapter。典型用法：`LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])`。秩选择经验：r=8 适用于大多数任务，r=64 接近 full fine-tuning 性能。QLoRA 由 bitsandbytes 库提供 4-bit 量化后端。Punica 和 S-LoRA 系统支持多租户 LoRA serving（多个 adapter 共享同一 base model）。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models

## Mixture-of-Experts (MoE / 混合专家模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixture-of-Experts (MoE) 是一种稀疏激活的神经网络架构，将 FFN 层替换为多个"专家"子网络（experts），通过一个可学习的路由机制（router/gate）为每个输入 token 选择 top-k 个专家进行计算。核心特性：(1) 总参数量巨大（可达万亿级别），但每个 token 仅激活少量参数（稀疏性），推理计算量与激活参数量成正比而非总参数量；(2) 路由函数通常为简单的 softmax 门控：$g(x) = \text{softmax}(W_r x)$，选择 top-k（通常 k=1 或 2）个专家；(3) 需添加负载均衡损失（load balancing loss）防止所有 token 都路由到同一专家。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# MoE Layer Forward Pass
def moe_forward(x, router_W, experts, top_k=2):
    # x: [batch_size, seq_len, d_model]
    # router_W: [d_model, num_experts]
    
    # Step 1: Routing
    router_logits = x @ router_W          # [B, S, E]
    router_probs = softmax(router_logits)
    top_k_weights, top_k_indices = topk(router_probs, top_k)  # [B, S, k]
    
    # Step 2: Dispatch tokens to selected experts
    for expert_id in range(num_experts):
        mask = (top_k_indices == expert_id)  # tokens routed to this expert
        expert_input = x[mask]
        expert_output = experts[expert_id](expert_input)  # FFN per expert
        # Step 3: Combine (weighted sum)
        output[mask] += top_k_weights[mask] * expert_output
    
    # Load balancing loss
    fraction_tokens_routed = mean of router_probs  # [E]
    L_balance = num_experts * sum(fraction_tokens_routed * fraction_tokens_routed)
    return output, L_balance
```

代表性 MoE 模型：Switch Transformer（1.6T 参数，top-1 路由，2048 experts）、GLaM（1.2T 参数，top-2 路由，训练成本仅为 GPT-3 的 1/3）、Mixtral 8×7B（8 experts，每次激活 2 个，总 46.7B 参数，激活 12.9B，性能超 LLaMA2-70B）、DeepSeek-V2（Multi-head Latent Attention + MoE）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

训练框架：DeepSpeed-MoE、Megatron-LM（支持 expert parallelism + tensor parallelism 混合）、Tutel（动态自适应并行和流水线策略）。推理框架：vLLM 支持 expert parallelism、EdgeMoe（端侧通过 expert-wise bit-width adaptation 减少加载时间）、PC-MoE（利用 expert 激活的时间局部性，维护参数委员会减少资源消耗）。MoE 模型可通过 Sparse Upcycling 从 dense checkpoint 初始化（使用约 50% 的原始预训练成本，性能显著超过 dense 对应模型）。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models

## Post-Training Quantization (PTQ / 训练后量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Post-Training Quantization (PTQ) 是不需要重新训练即可将全精度模型转换为低精度模型的技术。对于 LLM，PTQ 分为两大类别：(i) Weight-Only Quantization——仅量化权重（W4A16、W3A16），激活保持 FP16，推理时动态反量化到 FP16 进行 MatMul；(ii) Weight-Activation Co-Quantization——同时量化权重和激活（W8A8、W4A8），可直接利用整数计算单元加速。

标准量化公式：

$$X^{\text{Int}N} = \text{Round}\left(\frac{2^N}{\text{absmax}(X^{\text{FP32}})} \times X^{\text{FP32}}\right) = \text{Round}(c^{\text{FP32}} \times X^{\text{FP32}})$$

$$X^{\text{FP32}} = \text{dequantize}(c^{\text{FP32}}, X^{\text{Int}N}) = \frac{X^{\text{Int}N}}{c^{\text{FP32}}}$$

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

GPTQ（Weight-Only PTQ 核心方法）的逐层量化流程：

```
# GPTQ: Layer-wise quantization with Hessian-based error compensation
for layer in model.layers:
    W = layer.weight                               # [d_out, d_in] FP16
    H = inverse_hessian(W, calibration_data)        # [d_in, d_in]
    # H captures weight importance correlations
    
    for col in range(d_in):
        # Quantize column 'col'
        w_q[:,col] = quantize(W[:,col])  # INT3/INT4
        error = w_q[:,col] - W[:,col]
        # Compensate: update remaining columns using Hessian
        W[:,col+1:] -= error * H[col, col+1:] / H[col, col]
    # Result: W_int4 with compensation minimizing output error
```

AWQ（Activation-aware Weight Quantization）的关键改进：观察 activation 分布而非 weight 大小来决定哪些权重重要。通过 per-channel scaling factor $s$ 保护~1% salient weights：

$$s^* = \arg\min_s \|\| Q(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} X - WX \|\|$$

SmoothQuant（Weight-Activation Co-Quantization）：利用 activation 不同 channel 的相似性，通过 per-channel scaling 变换将量化难度从 activation 转移到 weight：$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

主流实现：GPTQ（GitHub: IST-DASLab/gptq）、AWQ（GitHub: mit-han-lab/llm-awq）、llama.cpp（K-quant 量化 2-8bit，CPU 3-4× 加速）、bitsandbytes（LLM.int8() 8-bit 量化 GPU 推理）、TensorRT-LLM（集成 GPTQ/AWQ/SmoothQuant）。QuaRot 和 SpinQuant 利用随机旋转矩阵消除 outlier 提高量化友好度。VPTQ 引入 Vector Quantization 替代标量量化，在 2-bit 下相比 GPTQ/AWQ 提升 up to 4.41 perplexity。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models

## Prompt Compression（提示压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Prompt Compression 是在 LLM 推理前减少输入 prompt 长度以降低计算量的技术。由于 attention 的计算复杂度为 $O(T^2 d)$，prompt 长度 $T$ 减少 $k$ 倍可带来约 $k^2$ 倍的 attention 计算节省。核心方法包括：(1) Token Pruning——在推理过程中逐步移除不重要的 token（如 PoWER-BERT、DynamicViT）；(2) Prompt Summarization——使用小模型或基于熵的方法压缩 prompt 文本（如 LLMLingua）；(3) Soft Prompt Compression——训练 autoencoder 将长 prompt 压缩为少量 summary vectors（如 AutoCompressors、ICAE）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LLMLingua 的 token 级迭代压缩：

```
# LLMLingua: Coarse-to-Fine Prompt Compression
def compress_prompt(prompt, target_ratio, budget_controller):
    segments = split_into_segments(prompt)
    
    # Stage 1: Coarse-grained (segment level)
    segment_importance = []
    for seg in segments:
        entropy = compute_entropy(LLM(seg))  # per-token entropy from LLM
        segment_importance.append(mean(entropy))
    
    # Budget controller allocates compression budget per segment
    budgets = budget_controller(segment_importance, target_ratio)
    
    # Stage 2: Fine-grained (token level, iterative)
    for seg, budget in zip(segments, budgets):
        tokens = tokenize(seg)
        while len(tokens) / len(original_tokens) > budget:
            # Remove token with lowest conditional perplexity increase
            scores = [perplexity_increase(seg, i) for i in range(len(tokens))]
            tokens.pop(argmin(scores))
    
    # Stage 3: Distribution alignment (instruction tuning)
    compressed = detokenize(concatenate(all_tokens))
    return compressed
```

LLMLingua 可实现 20× 压缩比。LLMLingua-2（ACL 2024）将压缩重新定义为 token 分类问题，使用双向 encoder 替代单向 LLM，3-6× 更快且 1.6-2.9× 端到端延迟改善。LongLLMLingua 针对长上下文场景增加了位置偏差处理。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

压缩可发生在预处理或运行时：LLMLingua 系列在推理前压缩 prompt；Token pruning（如 Deja Vu、PuMer）在推理过程中动态移除 token。AutoCompressors 和 ICAE 训练专门的压缩模型将长上下文映射为少量 soft prompt token。对于 ViT，DynamicViT 和 A-ViT 根据输入复杂度自适应选择保留的 patch token 数量。使用场景：RAG 检索长文档、多轮对话压缩历史、代码补全长上下文。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models

## KV Cache (Key-Value Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

KV Cache 是 Transformer decoder 自回归推理中为避免重复计算而缓存 attention 的 Key 和 Value 中间状态的机制。由于自回归解码每步仅生成一个新 token，先前 token 的 K/V 已在之前步骤计算过——直接缓存复用可避免每步重新计算所有历史 token 的 K/V，将单步计算复杂度从 $O(T^2 d)$ 降至 $O(T d)$。内存占用：$2 \times B \times S \times D \times L \times 4$ bytes（FP32，B=batch_size, S=seq_len, D=hidden_dim, L=layers, 2=K+V, 4=4 bytes/FP32）。对于 LLaMA-7B（D=4096, L=32），当 B=1, S=2048 时 KV cache 约需 2GB。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Decoder Inference with KV Cache
def decoder_step_with_cache(x_new, K_cache, V_cache, layer):
    # x_new: [B, 1, D]  -- only the new token
    # K_cache, V_cache: lists of [B, prev_len, D] per layer
    
    Q_new = layer.W_Q @ x_new           # [B, 1, D]
    K_new = layer.W_K @ x_new           # [B, 1, D]
    V_new = layer.W_V @ x_new           # [B, 1, D]
    
    # Update cache (append new K/V)
    K = concat([K_cache[layer], K_new])  # [B, prev_len+1, D]
    V = concat([V_cache[layer], V_new])  # [B, prev_len+1, D]
    K_cache[layer] = K
    V_cache[layer] = V
    
    # Attention: Q_new attends to ALL cached K,V
    scores = Q_new @ K^T / sqrt(d_head)  # [B, 1, prev_len+1]
    attn = softmax(scores)
    output = attn @ V                     # [B, 1, D]
    return output
```

优化变体：(i) Multi-Query Attention (MQA)——所有 head 共享同一 K/V，减少 cache 大小 $h$ 倍；(ii) Grouped-Query Attention (GQA)——将 heads 分组共享 K/V，平衡质量和效率；(iii) Multi-head Latent Attention (MLA, DeepSeek-V2)——将 K/V 压缩到低维 latent space，推理时通过矩阵 up-projection 恢复（up-projection matrix 可被吸收到 $W^K, W^V$ 中）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

KV cache 管理是 LLM serving 系统的核心挑战。vLLM 的 PagedAttention 将 KV cache 按 block 管理（类似 OS 虚拟内存），消除内存碎片。H2O 基于 attention scores 淘汰低重要性 token 的 KV cache（up to 5× 内存节省）。FastGen 为每个 attention head 自适应选择最优稀疏模式。CacheGen 将 KV cache 作为流式数据编码（增量标记 + 算术编码）。Prompt Cache 预计算并缓存常见文本段的注意力状态（GPU 8×、CPU 60× 加速）。vAttention 直接依赖 OS/CUDA 做物理内存重分配，在 vLLM 基础上进一步提升 1.29× 吞吐。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models

## Hybrid Transformer-Mamba Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hybrid Transformer-Mamba 模型是一种将传统 Transformer 注意力层与 SSM（State-Space Model，具体为 Mamba-2）层**交替交错排列**的新型语言模型架构。其核心思想是结合两者的互补优势：Transformer 注意力层的强大语言建模能力（尤其在 recall 和 in-context learning 任务中，注意力机制的 quadratic pairing 让每对 token 直接交互）弥补 Mamba 的劣势（选择性压缩导致信息随时间衰减）；Mamba 的线性计算复杂度和恒定推理内存（无需 KV Cache，因递归结构只需固定大小 hidden state $h_t \in \mathbb{R}^{s}$）弥补 Attention 的 quadratic 复杂度瓶颈（当 sequence length 增加时 $O(L^2)$ 计算和 $O(L)$ KV cache 导致爆炸）。具体来说，Hybrid 模型按一定比例交替排列 attention layers 和 Mamba-2 layers（如 Hybrid-2.7B 为 6 attention layers + 58 Mamba-2 layers），每层有独立的前向线性投影、归一化和残差连接。在输入处理上，Mamba-2 层将分离的 attention 和 FFN 合并为统一层（RMSNorm → input projection 生成 dt/xBC/z → conv1D + SiLU → SSD → z-gating → RMSNorm → output projection），而 attention 层保持标准 multi-head attention 流程。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Hybrid 模型的推理 pipeline 交替执行两种 layer：

```
# Hybrid Model Inference Pipeline
for layer in model.layers:
    x = RMSNorm(x)
    if layer.is_attention:
        # 标准 multi-head attention
        Q, K, V = proj_qkv(x)  # [b, h, l, d_head]
        O = FlashAttention(Q, K, V)  # QK^T → softmax → PV
        x = O_proj(O) + x  # residual
    else:  # Mamba-2 layer
        z, dt, xBC = input_proj(x)  # z: gating, dt/xBC: SSM inputs
        xBC = conv1D(xBC)
        xBC = SiLU(xBC)
        x, B, C = split(xBC)
        dt = softplus(dt + dt_bias)  # 确保 dt > 0
        # SSD 计算
        Y = SSD(dt, A, x, B, C)  # chunked semiseparable matrix
        Y = Y * z  # z-gating (element-wise)
        x = RMSNorm(Y)
        x = output_proj(x) + residual
```

关键计算：SSD 的 block decomposition（见 SSD 条目）将 SSM 的半可分矩阵分为 diagonal blocks（独立并行 MatMul）和 off-diagonal blocks（通过 right/center/left 因子传递状态信息）。在 Mamba-2 论文的 Hybrid-2.7B 配置中，attention 层为 30 head × d_head=128，SSD 为 80 head × d_head=64，d_state=128，block_size=256。

优点：在 256K sequence length 下，Hybrid 模型比 Mistral/Llama-3.1 8B/Mixtral 8×7B 推理快 2.5×，KV Cache 内存仅需 1/8。劣势：两种 kernel（FA-2 和 SSD）的异构计算模式导致随 sequence length/batch size 变化的性能瓶颈转移——短序列下 Mamba-2 层主导延迟（数量多），长序列下 attention 层的 quadratic 复杂度主导延迟。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源实现：Mamba-2 仓库 (https://github.com/state-spaces/mamba) 提供 Hybrid-2.7B (Mamba2attn-2.7B) 的 GPU 优化 CUDA kernel（FA-2 + SSD 5-kernel 实现）。使用时通过 PyTorch 加载模型，FA-2 通过 fused CUDA kernel 执行，SSD 的 5 kernel（chunk cumsum → chunk state → state passing → BMM chunk → chunk scan）逐个 launch。Variants 包括 Jamba (MoE Hybrid)、Samba (shared attention block)、Zamba 等，通过调整 attention/Mamba-2 比例和结构实现不同 trade-off。

涉及论文标题：
- HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models

## State-Space Duality (SSD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

State-Space Duality (SSD) 是 Mamba-2 中提出的硬件高效 SSM 并行处理算法。其核心洞察：SSM 的计算可以表示为**半可分矩阵**（semiseparable matrix），因而可同时进行两种计算：(i) **线性（递归）计算**——逐时间步更新 hidden state，$O(L)$ 复杂度，适合推理；(ii) **二次（注意力式）计算**——通过 MatMul 并行处理整个序列，利用 GPU/Tensor Core 的 MatMul 算力。SSD 使用 block decomposition 策略将序列分段：diagonal blocks 内独立并行计算局部 SSM 输出（通过 MatMul），off-diagonal 部分分解为 right factor（block 内状态汇总）、center factor（block 间累积乘法，1-semiseparable multiplication，传递全局状态信息）、left factor（将累积全局状态投影到每个 block 的输出），最终 $Y = Y_{Diag} + Y_{Off}$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

SSD 的 block decomposition 伪代码（来自 Mamba-2/HLX 论文 Fig. 6）：

```
Input: dt:[b,n,l], A:[n], B:[b,s,l], C:[b,s,l], x:[b,h,l]
Output: state_Final:[b,n,h,s], Y_Final:[b,n,h,l]

# 0. Block decomposition: l → [c, cl] (c chunks of size cl)
# 1. Chunk Cumsum kernel
sdt = softplus(dt + dt_bias)                    # [b, n, c, cl]
dA_CS = cumsum(sdt × A)                          # cumulative decay

# 2. Chunk State kernel (right factor)
decay_states = exp(dA_CS[:,:,:,-1:] - dA_CS)   # time decay within block
states = einsum(B, decay_states, sdt, x)         # (right factor) [b,n,h,s,c]

# 3. State Passing kernel (center factor)
dA_chunkCS = cumsum(zero_padding(dA_CS[:,:,:,-1]))  # inter-chunk decay
decay_chunk = causal_mask(exp(dA_chunkCS[:,:,:,None] - dA_chunkCS[:,:,None,:]))
states_int = einsum(decay_chunk, states)         # (propagated states)

# 4. BMM Chunk kernel
CB_T = einsum(C, B^T)                            # [b, c, cl, cl]

# 5. Chunk Scan kernel
L = causal_mask(exp(dA_CS[:,:,:,:,None] - dA_CS[:,:,:,None,:]))
Y_Diag = einsum(CB_T, L, sdt, x)                # diagonal output
state_decay_out = exp(dA_CS)
Y_Off = einsum(C, states_int × state_decay_out)  # off-diagonal output
Y_Final = Y_Diag + Y_Off
```

其中 `h` = head dim, `n` = num heads, `s` = state dim, `l` = seq len, `c` = num chunks, `cl` = chunk len。

对比：SSD 相比 Mamba-1 的关键改进是增加了 MatMul 操作的可并行性——通过 tile 分解将递归操作转为可并行的 MatMul blocks。但 SSD 仍有大量 element-wise 操作和 Einsum 多维张量运算，导致 memory-bound 特征和低 compute utilization（GPU 上约 26.9% on A100, 38% on H100）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPU 实现：SSD 在 GPU 上分为 5 个 CUDA kernel 执行（chunk cumsum, chunk state, state passing, BMM chunk, chunk scan），每个 kernel 之间中间数据通过 DRAM 传递。PyTorch 参考实现位于 mamba 仓库 (https://github.com/state-spaces/mamba)。由于 5 kernel 分离执行导致大量 DRAM 流量和低数据重用，HLX 提出 PipeSSD 将其融合为单 kernel 三阶段流水线（详见 PipeSSD 条目）。Fused SSD 虽然最大化数据重用，但在 GPU 上不可行——中间数据 642KB/block 超过 SM 寄存器+共享内存容量（A100: 256KB RF + 164KB SMEM, H100: 256KB RF + 224KB SMEM），导致 register spilling 和 occupancy 下降，延迟反而恶化 1.74×。

涉及论文标题：
- HLX: A Unified Pipelined Architecture for Optimized Performance of Hybrid Transformer-Mamba Language Models

## Attention State / Attention Composition (⊕ operator)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention State 是 FlashInfer 定义的 attention 计算的标准输出格式，基于 Block-Parallel Transformer (BPT, Liu & Abbeel, 2023) 的 observation：attention outputs for the same query and different keys/values can be composed。Attention State 是一个 tuple $\begin{bmatrix} \mathbf{O}(\mathcal{I}) \\ \mathbf{LSE}(\mathcal{I}) \end{bmatrix}$，其中 $\mathbf{O}(\mathcal{I})$ 是 query 对 index set $\mathcal{I}$ 的 attention output，$\mathbf{LSE}(\mathcal{I}) = \log \sum_{i \in \mathcal{I}} \exp(\mathbf{q} \cdot \mathbf{k}_i)$ 是 attention scale（log-sum-exp of attention scores）。Attention State 的关键性质：$\oplus$ operator 是 **associative and commutative** 的，即 $\text{State}(\mathcal{I} \cup \mathcal{J}) = \text{State}(\mathcal{I}) \oplus \text{State}(\mathcal{J})$，且合并顺序任意。这意味着多个 partial attention computation 的结果可在任意顺序下合并为正确 final output。⊕ operator 定义为：

$$\begin{bmatrix} \mathbf{O}(\mathcal{I} \cup \mathcal{J}) \\ \mathbf{LSE}(\mathcal{I} \cup \mathcal{J}) \end{bmatrix} = \begin{bmatrix} \frac{\exp(\mathbf{LSE}(\mathcal{I}))\mathbf{O}(\mathcal{I}) + \exp(\mathbf{LSE}(\mathcal{J}))\mathbf{O}(\mathcal{J})}{\exp(\mathbf{LSE}(\mathcal{I})) + \exp(\mathbf{LSE}(\mathcal{J}))} \\ \log(\exp(\mathbf{LSE}(\mathcal{I})) + \exp(\mathbf{LSE}(\mathcal{J}))) \end{bmatrix}$$

在 FlashInfer 中，Attention State 用作 partial attention computation 的 canonical output，⊕ 用作 standard reduction operator（类比 GEMM 中的 summation）。这使得：(1) Load-balanced kernel 可将长 KV-cache 拆分为多个 chunks，由不同 CTAs 并行处理，各 CTA 输出 partial AttentionState；(2) Contraction kernel 用 ⊕ compose 所有 partial states 为 final output；(3) 合并顺序无关，允许 deterministic fixed-order aggregation（vs Stream-K 的 atomic non-deterministic aggregation）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Attention State 在 FlashInfer 中的完整 pipeline：

```
// ===== Attention Computation with Attention State =====
// Input: Q ∈ R^{T_q × d}, K ∈ R^{L × d}, V ∈ R^{L × d}
// Output: O ∈ R^{T_q × d}

// Step 1: Split K/V into chunks (for load-balanced scheduling)
chunks = [(K_1, V_1, L_1), (K_2, V_2, L_2), ..., (K_n, V_n, L_n)]
// 其中 Σ L_i = L

// Step 2: Parallel per-chunk attention computation
for each chunk_i in parallel:  // different CTAs
    // Standard FlashAttention for this chunk
    O_i = 0; m_i = -inf; l_i = 0
    for each KV tile in chunk_i:
        S = Q × K_tile^T
        m_new = max(m_i, rowmax(S))
        P = exp(S - m_new)
        l_new = l_i * exp(m_i - m_new) + rowsum(P)
        O_i = O_i * exp(m_i - m_new) + P × V_tile
        m_i = m_new
        l_i = l_new
    
    // Convert to AttentionState: canonical output format
    // O_i is already accumulated attention output
    // LSE_i = log(l_i) + m_i  (recover log-sum-exp from running stats)
    partial_state[i] = AttentionState(
        O = O_i / l_i,           // normalize
        LSE = log(l_i) + m_i     // attention scale
    )

// Step 3: Merge all partial states via ⊕ (contraction)
O_final = zeros(T_q, d)
LSE_final = -inf  // log(0) equivalent
for each partial_state[i]:
    O_final, LSE_final = (O_final, LSE_final) ⊕ partial_state[i]
    // ⊕ expansion:
    // weight_final = exp(LSE_final)
    // weight_i = exp(partial_state[i].LSE)
    // O_final = (weight_final * O_final + weight_i * partial_state[i].O) 
    //         / (weight_final + weight_i)
    // LSE_final = log(weight_final + weight_i)

// ===== Associativity Proof Sketch =====
// ⊕ is associative because:
//   (A ⊕ B) ⊕ C = A ⊕ (B ⊕ C)
// This follows from the associativity of addition and the monotonicity of exp/log
// ⊕ is commutative because:
//   A ⊕ B = B ⊕ A
// This follows from the commutativity of addition

// ===== Key Insight =====
// The same ⊕ operation works for:
// 1. Merging parallel chunks from load-balanced scheduling
// 2. Merging prefix and suffix in composable formats
// 3. Merging speculative decoding tree branches
// All three use cases share the same attention composition operator
```

与相关概念的比较：
- **FlashDecoding (Dao et al., 2023)**：使用 Split-K 将 KV split 为 chunks，各 chunk 输出 partial softmax → final reduction。本质上是 Attention State + ⊕ 的特例
- **Ring-Attention (Liu et al., 2023)**：利用 ⊕ 的 associative 性质将 attention 分布到多设备，ring communication 传递 partial states
- **FlashInfer**：将 ⊕ 标准化为 attention kernel 的 canonical reduction operator，使 load-balanced scheduling、composable formats、tree attention 统一使用同一 merge 原语

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashInfer 中 Attention State 的实现：
- 在 CUDA 层面，Attention State 用两个 per-query-row floats 表示：(O_i ∈ R^d, LSE_i ∈ R)
- Contraction kernel 接收多个 partial Attention States → GPU 上用 fast math (exp/log on CUDA MUFU) 执行 ⊕ compose
- Implementation 注意数值稳定性：weight = exp(LSE - LSE_max) 避免 overflow（与 online softmax 中的 rescaling 类似）
- 多种使用场景复用同一 contraction kernel：load-balanced scheduling（同 batch 内 chunks）、composable formats（prefix + suffix）、speculative tree attention（tree branches）

涉及论文标题：
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving

## Structural Symmetry between FFN and Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Structural Symmetry between FFN and Attention 是 Geva et al. (2020) 提出的 FFN 与单头注意力之间的结构等价性观察。核心等价推导：FFN(X) = φ(X·W_1^T)·W_2 可重新解释为 "X attends over W_1（作为 keys）to retrieve values from W_2（作为 values）"。具体来说：(1) 在 Attention(Q,K,V) = softmax(QK^T/√d_k)·V 中，将 Q 替换为 X，K 替换为 W_1，V 替换为 W_2^T；(2) 将 softmax 替换为 element-wise 非线性 φ(·)；(3) 两者形式上完全一致。因此 FFN 可被理解为 "对长度为 d_ff 的参数序列的注意力"——X query 通过 key W_1 访问 value W_2 中的存储知识（Geva et al. 的 "key-value memory" 解释）。对于 gated SwiGLU 变体，SwiGLU(X) = (SiLU(X·W_gate) ⊙ (X·W_up))·W_down，可定义 φ_s(Q,K) = SiLU(Q·K^{(g)T}) ⊙ Q·K^{(u)T}，同样重写为 φ_s(Q,K)·V，保持注意力类的结构。FlashMHF 论文基于此对称性提出 multi-head FFN 概念——正如 multi-head attention 从不同子空间并行学习，multi-head FFN 也应当从多个独立的参数子空间并行处理，增强表达力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# 单头Attention 与 FFN 的结构对称性证明（FlashMHF 论文公式1-4）：

# 1. 标准单头 Attention:
def single_head_attention(Q, K, V):
    # Q, K, V ∈ R^{L × d_k}
    scores = Q @ K.T / sqrt(d_k)     # [L×d_k] × [d_k×L] → [L×L]
    weights = softmax(scores, dim=-1)  # row-wise softmax: 每行之间归一化
    output = weights @ V              # [L×L] × [L×d_k] → [L×d_k]
    return output

# 2. 标准 FFN (vanilla):
def ffn(X, W1, W2):
    # X ∈ R^{L × d_model}, W1 ∈ R^{d_ff × d_model}, W2 ∈ R^{d_model × d_ff}
    hidden = activation(X @ W1.T)    # [L×d_model] × [d_model×d_ff] → [L×d_ff]
    output = hidden @ W2.T           # [L×d_ff] × [d_ff×d_model] → [L×d_model]
    return output

# 3. 对称性证明——用 Attention "模板" 表达 FFN:
#    令 Q = X, K = W1, V = W2^T, softmax → element-wise φ
#    → FFN = "X attends over W1 parameters to retrieve from W2 values"

# 4. Gated SwiGLU 的注意形式重写（FlashMHF 公式3-4）:
def swiglu_as_attention(X, K_gate, K_up, V):
    # K_gate, K_up ∈ R^{d_ff × d_model}, K = [K_gate, K_up]
    # V ∈ R^{d_ff × d_model} (即 W_down)
    Q = X                           # query = input
    gate = SiLU(Q @ K_gate.T)       # [L×d_model] × [d_model×d_ff] → [L×d_ff]
    up   = Q @ K_up.T               # [L×d_model] × [d_model×d_ff] → [L×d_ff]
    output = (gate ⊙ up) @ V.T      # element-wise gate × up, then project
    return output
    # 定义 φ_s(Q,K) = SiLU(Q·K^{(g)T}) ⊙ Q·K^{(u)T}
    # 则 output = φ_s(Q,K)·V，证明 SwiGLU 是广义注意力的一个实例

# 5. Multi-Head 推广（FlashMHF 核心思路）:
#    标准 attention 有 multi-head → FFN 也应有 multi-head
#    Q = split_H(X · W_in) ∈ R^{L×H×d_h}
#    每 head h 独立执行 FFÑ(Q[:,h,:]; K^h, U^h, V^h)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该对称性在 Transformer 研究中有多个系统的应用：(1) Geva et al. (2020) 利用此对称性分析 FFN 的知识存储能力——将 W_2 的每一行解释为一个 "learned pattern"，FFN 通过 softmax-like 选择检索最相关的 pattern；(2) Tokenformer (Wang et al., 2024) 将所有线性投影替换为 Token-Parameter Attention（PAttention），彻底 operationalize 此对称性；(3) FlashMHF 通过此对称性设计 multi-head FFN——正如 MHA 的 H 个头从 H 个不同子空间处理 Q 以丰富表示，MH-FFN 的 H 个头也从 H 个不同子空间处理 X 以增强表达力；(4) MLP-Mixer 和 DaViT 利用此对称性在 token-mixing（attention-like over tokens）和 feature-mixing（FFN-like over features/channels）之间建立对称操作。实际使用中，此对称性是 rethinking FFN architecture 从 "通用近似函数" 到 "结构化参数注意力" 的概念转变基石。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
- Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2020)
- Tokenformer (Wang et al., 2024)

## SwiGLU FFN

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SwiGLU（Swish-Gated Linear Unit）是现代 LLM（LLaMA、LLaMA-2/3、Qwen、Mistral 等）中普遍采用的 gated FFN 变体。计算流程：SwiGLU(X) = (SiLU(X·W_gate^T) ⊙ (X·W_up^T)) · W_down。首先输入 X ∈ R^{L×d_model} 分别通过两个独立投影 W_gate（gate分支）和 W_up（up分支）映射到 d_ff 维（d_ff ≈ 8/3·d_model 是经验最优 ratio）；然后 gate 分支经 SiLU（Sigmoid Linear Unit = x·σ(x)）激活后与 up 分支做 element-wise 乘法 ⊙；最后经 W_down 投影回 d_model。SwiGLU 是 GLU family（Gated Linear Unit）的一种，其激活函数为 Swish/SiLU。相比原始 GELU-gated FFN、ReLU FFN 或标准 non-gated FFN，SwiGLU 在 scaling law 实验中和实际预训练中展现出更好的 perplexity 和收敛速度（Touvron et al., 2023; Shazeer, 2020）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# SwiGLU FFN 完整 forward（PyTorch 风格，LLaMA-like）:
# 参数: W_gate, W_up ∈ R^{d_ff × d_model}, W_down ∈ R^{d_model × d_ff}
# 输入: X ∈ R^{L × d_model}

def swiglu_ffn(X, W_gate, W_up, W_down):
    # Step 1: Gate branch — 计算 gating signal
    gate_logits = X @ W_gate.T        # [L, d_model] × [d_model, d_ff] → [L, d_ff]
    gate = F.silu(gate_logits)        # SiLU(x) = x * sigmoid(x), element-wise
    
    # Step 2: Up branch — 计算 value stream
    up = X @ W_up.T                   # [L, d_model] × [d_model, d_ff] → [L, d_ff]
    
    # Step 3: Gating — element-wise multiply
    hidden = gate * up               # [L, d_ff] ⊙ [L, d_ff] → [L, d_ff]
    # 关键: gate ∈ (0, ~d_ff) 范围（SiLU 无上界），
    #       对 up 的每个 channel 做软性 important/unimportant 选择
    
    # Step 4: Output projection
    output = hidden @ W_down.T        # [L, d_ff] × [d_ff, d_model] → [L, d_model]
    return output

# 张量形状示例（LLaMA-7B: d_model=4096, d_ff=11008, L=2048）:
# gate_logits: [2048, 4096] × [4096, 11008] → [2048, 11008]  (约90M elements/bf16)
# hidden: [2048, 11008]  ≈ 45MB in bf16 → 必须写入HBM
# output: [2048, 11008] × [11008, 4096] → [2048, 4096]   (约17MB)
```

与标准 FFN 的关键区别：(1) gate 分支引入 channel-wise multiplicative gating——不同于简单 activation 的逐元素非线性，gating 机制允许 FFN 动态选择信息流中的哪些 channel 特征被保留/抑制；(2) 双投影设计（gate + up）使得参数量略多于标准 FFN（d_ff × d_model × 2 + d_ff × d_model = 3·d_ff·d_model vs 标准 FFN 的 2·d_ff·d_model），但这被证明是 parameter-efficient 的；(3) SiLU 的非单调性（在 x<0 时轻微负激活）相比 ReLU 的 hard-zero 提供更丰富的梯度流。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SwiGLU 在现代 GPU 上的实现主要由 cuBLAS GEMM 执行：三次矩阵乘法（gate/proj + up/proj + down/proj）通过高度优化的 tiled GEMM kernel 完成，分别由独立的 cublasGemmEx 调用。SiLU 激活和 element-wise multiply 在后续的 pointwise kernel/fused elementwise kernel 中完成。优化策略：(1) torch.compile 可将三次 GEMM + SiLU + multiply + down GEMM 融合为单个 fused kernel（部分场景），减少 kernel launch overhead；(2) 利用 NVIDIA cuBLASLt 的 fused epilogue 将 SiLU 激活与 gate GEMM 的 output write 融合；(3) FlashMHF 等工作正在探索更彻底的 I/O-aware fusion——通过 multi-head 设计分解大中间激活，再通过 SRAM-resident blockwise 计算消除 HBM round-trip。当 batch size 较小时（inference 典型 bs=1-8），中间激活 hidden ∈ R^{L×d_ff} 虽然远小于 attention 的 QK^T ∈ R^{L×L}，但仍占显著 HBM 带宽——LLaMA-7B 单层 SwiGLU 的 hidden tensor ≈ L·11008·2 bytes，L=4096 时约 90MB。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
- LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators (for GEMM-based Operator Chain)

## Multi-Head Feed-Forward Network (MH-FFN)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-Head Feed-Forward Network (MH-FFN) 是将 multi-head attention 的思想直接应用于 FFN 的朴素设计（FlashMHF 论文定义）。核心操作：(1) 输入 X ∈ R^{L×d_model} 先经 W_in 线性投影再通过 split_H 沿 d_model 维度切分为 H 个 head query，每 head 维度 d_h = d_model/H；(2) 每 head h 独立执行 key-value 形式的 FFN：FFÑ(Q_h; K^h, U^h, V^h) = (SiLU(Q_h·K^{hT}) ⊙ (Q_h·U^{hT}))·V^h，其中 K^h, U^h, V^h ∈ R^{d_ff×d_h} 是每 head 的私有参数；(3) 所有 head 的输出 concat 后经 W_out 投影回 d_model。这个设计直接从 MHA 的 split/parallel/compute/concat 范式迁移而来，但遇到两个关键挑战：(1) Memory Pressure——H 个 head 各自 materialize 中间激活 ∈ R^{L×d_ff}，总内存 O((L·H + d_model)·d_ff)，随 H 线性增长；(2) Scaling Imbalance——模型 scale up 时 d_ff 增长（因模型总参数增长需求）但 d_h 固定（如 128，继承自 MHA 设计），d_ff/d_h ratio 从 128M 的 16 膨胀到 1.3B 的 45，远超过最优范围。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# Naïve Multi-Head FFN (MH-FFN) forward pass:
# 参数: W_in ∈ R^{d_model×d_model}, W_out ∈ R^{d_model×d_model}
#       For h=1..H: K^h, U^h ∈ R^{d_ff×d_h}, V^h ∈ R^{d_ff×d_h}
# 输入: X ∈ R^{L×d_model}

def naive_mh_ffn(X, W_in, W_out, per_head_params):
    H = len(per_head_params)  # number of heads
    d_h = d_model // H         # per-head dimension
    
    # Step 1: Project and split into heads
    Q = split_H(X @ W_in)     # [L, d_model] → [L, H, d_h]
    
    # Step 2: Per-head independent FFN computation
    S = []
    for h in range(H):
        K_h, U_h, V_h = per_head_params[h]  # each: [d_ff, d_h]
        Q_h = Q[:, h, :]                     # [L, d_h]
        
        # head-wise SwiGLU-style key-value FFN:
        gate = SiLU(Q_h @ K_h.T)   # [L, d_h] × [d_h, d_ff] → [L, d_ff]
        up   = Q_h @ U_h.T         # [L, d_h] × [d_h, d_ff] → [L, d_ff]
        out_h = (gate * up) @ V_h  # [L, d_ff] × [d_ff, d_h] → [L, d_h]
        
        S.append(out_h)            # H × [L, d_h] → 总激活 H·L·d_ff
    
    # Step 3: Concatenate heads and output projection
    O = concat_H(S) @ W_out        # [L, d_model] × [d_model, d_model]
    return O

# 问题演示（370M scale: d_model=1024, H=8, d_h=128, d_ff=2752, L=4096）:
# 每 head 激活: [4096, 2752] ≈ 22.5MB (bf16)
# 总 head 激活: H × 22.5 = 180MB (bf16) —— 相比标准 SwiGLU 的 22.5MB 为 8×
# d_ff/d_h = 2752/128 = 21.5 → 显著超过 optimal range ~8/3
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MH-FFN 的实用化受限于两个核心问题：(1) 内存消耗随 H 线性膨胀，即使在小模型 (128M) 下激活内存也是标准 FFN 的 H 倍，成为训练和推理的 bottleneck；(2) FLOPs 相同但 scaling imbalance 导致性能退化——FlashMHF 实验证实 MH-FFN 在 128M 优于 baseline 但在 370M 已失效。已有近似工作：(1) MH-MoE (Wu et al., 2024) 探索了多 FFN head + MoE sparse routing，但所有 head 共享 expert parameters，在 dense 模式下比 FlashMHF 需要 H 倍 FLOPs（公平对比不可行），且 memory 同样随 H 线性增长；(2) Tokenformer 将 FFN 替换为 Token-Parameter Attention，使用 learnable token 而非 fixed weight matrix 作为 key-value store，可视为一种隐式的 multi-head FFN 实现。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
- Multi-Head Mixture-of-Experts (Wu et al., 2024)

## FlashMHF / Parallel FFN Sub-Networks

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlashMHF（Flash Multi-Head Feed-Forward）是 MH-FFN 的实用化改进，其核心创新是 Parallel FFN Sub-Networks 设计——用于解决 naïve MH-FFN 的 Scaling Imbalance 问题。传统 MH-FFN 的 d_ff/d_h ratio 随模型 scale 膨胀（128M: 16, 370M: 21, 1.3B: 45），因为 d_ff 增长而 d_h 固定。FlashMHF 的解决方案：(1) 将每 head 的 d_ff 维计算分解为 E 个 parallel sub-network，每 sub-network 的 internal dimension d_e ≈ 8/3·d_h（维持 SwiGLU 的最优 expansion ratio），总 d_ff = E·d_e；(2) 引入 learned gating：每 head h 有 gating matrix W^h ∈ R^{d_h×E}，计算 per-token sub-network weights R^h = sigmoid(Q_h·W^h) / Σ sigmoid（soft normalization），然后用 R^h 加权聚合所有 sub-network 的输出；(3) 最终 concat 所有 head 输出并做 W_out 投影。这个设计本质上类似 dense MoE——每 token 的所有 E 个 "expert"（sub-network）都参与计算（无 sparse top-k routing），以微小 gating 开销换取平衡的 internal ratio 和丰富的 representational diversity。相比标准 SwiGLU（单路径 "greedy search"），FlashMHF 的 H×E 个 parallel pathway 可类比为 implicit thinking 的 "beam search"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# FlashMHF Forward Pass (以 370M: d_model=1024, H=8, d_h=128, E=7, d_e≈342):
def flashmhf_forward(X):
    # 参数:
    # W_in ∈ R^{d_model × d_model}
    # For h=1..H, e=1..E: K_e^h, U_e^h, V_e^h ∈ R^{d_e × d_h}  (每个342×128)
    # For h=1..H:          W^h ∈ R^{d_h × E}                   (每头128×7)
    # W_out ∈ R^{d_model × d_model}
    
    # === Step 1: Head-wise split ===
    Q = split_H(X @ W_in)           # [L,1024]→[L,8,128]
    
    # === Step 2: Per-head gating + sub-network aggregation ===
    S = []  # 每head的输出
    for h in range(H):              # H=8
        Q_h = Q[:, h, :]            # [L, 128]
        
        # 2a: Gating — 学习每token对E个子网络的权重
        P = Q_h @ W[h]             # [L,128] × [128,7] → [L,7]   (logits)
        R = sigmoid(P) / (sigmoid(P).sum(dim=1, keepdim=True) + 1e-8)  # [L,7]
        # R[:,e] ∈ (0,1), Σ_e R[:,e] ≈ 1
        
        # 2b: Sub-network computation & weighted aggregation
        S_h = zeros([L, d_h])       # [L, 128]
        for e in range(E):          # E=7
            K_e, U_e, V_e = params_K[h][e], params_U[h][e], params_V[h][e]
            # 每个 [d_e, d_h] = [342, 128]
            
            # FFÑ sub-computation (SwiGLU-style key-value):
            gate = SiLU(Q_h @ K_e.T)       # [L,128]×[128,342]→[L,342]
            up   = Q_h @ U_e.T             # [L,128]×[128,342]→[L,342]
            out_e = (gate * up) @ V_e      # [L,342]×[342,128]→[L,128]
            
            S_h += R[:, e:e+1] * out_e     # gated aggregation
        
        S.append(S_h)                # [L, 128]
    
    # === Step 3: Concat & output ===
    O = concat_H(S) @ W_out         # [L,1024]×[1024,1024]→[L,1024]
    return O, Q, R

# 关键设计参数推导:
# 标准 SwiGLU: d_ff ≈ 8/3·d_model = 8/3·1024 ≈ 2731 (round to 2752)
# FlashMHF:   d_e ≈ 8/3·d_h = 8/3·128 ≈ 341 (round to 342 in multiples of 64)
#             E = floor(d_ff / d_e) = 2752/342 ≈ 8, 论文用 7
#             d_ff_actual = E × d_e = 7 × 342 = 2394 (需调整 layers 保持总参数 ≈ baseline)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FlashMHF 的实现：
1. **参数组织**：K/U/V 参数组织为 [H, E, d_e, d_h] 的 4D tensor 以利于 kernel 访问。Gating weights W^h ∈ R^{d_h×E} 为可训练参数，通过 sigmoid + normalization 得到 per-token per-sub-network 权重。相比 MoE 的 softmax-gated sparse routing，FlashMHF 的 sigmoid gate 避免了 top-k selection 的 load imbalance 和 token dropping 问题，且所有 sub-network 都参与计算（dense activation）。
2. **训练**：标准 PyTorch training loop，FlashMHF module 替换标准 SwiGLU FFN module。使用 AdamW optimizer，training hyperparameters 与 baseline 完全一致。128M/370M: 60B tokens (Pile), 1.3B: 100B tokens。单 GPU 训练（pretraining_tp=1）。
3. **推理优化**：SRAMFFN kernel（Triton/ThunderKittens 实现）用于高效 fused 计算——将 Step 2b 的 inner loop 和 Step 2a 全部融合为单个 I/O-aware kernel，避免中间 gate/up ∈ R^{L×d_e} 写入 HBM。
4. **配置灵活性**：d_h 可调（64/128/256），通过改变 H 和 E 适配不同模型 scale。380M 实验显示 d_h=128 为 sweet spot——d_h=64 每 head 容量不足（representational bottleneck），d_h=256 减少 head 数降低 diversity。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network

## FFN Scaling Imbalance (d_ff/d_h Ratio)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FFN Scaling Imbalance 是 FlashMHF 论文识别的 naïve Multi-Head FFN 在大模型 scale 上失效的根本原因。问题本质：MH-FFN 继承 MHA 的设计惯例——d_h 固定（如 128）而 d_ff 随模型 scale 增长（因模型参数总量增长要求）。这导致 d_ff/d_h ratio 随模型变大而失衡：128M 模型 d_ff/d_h = 2048/128 = 16；370M 模型 = 2752/128 = 21.5；1.3B 模型 = 5760/128 = 45。根据 Kaplan et al. (2020) 的 scaling law，FFN 的 d_ff/d_model ratio 存在最优范围（经验值约 8/3），偏离此范围会导致 parameter efficiency 下降。在 MH-FFN 语境下，d_ff/d_h 的角色等同于标准 FFN 的 d_ff/d_model——每 head 的 internal capacity (d_ff) 与 input dimension (d_h) 之比若过大，单个 head 的参数利用效率低，部分 d_ff 维的 capacity 被浪费或未被有效利用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Scaling Imbalance 的量化分析（FlashMHF Table 4 数据）:

模型 Scale   d_model   d_ff    H    d_h    d_ff/d_h   问题严重度
─────────────────────────────────────────────────────────
128M         892       2048    6    128    16.0       中等失衡（MH-FFN 仍优于 baseline）
370M         1024      2752    8    128    21.5       显著失衡（MH-FFN = baseline）
1.3B         2048      5760    16   128    45.0       严重失衡（MH-FFN << baseline 预期）

# 此时标准 SwiGLU 的 d_ff/d_model ratio:
# 128M: 2048/892  ≈ 2.30  ≈ 8/3.5  (接近最优)
# 370M: 2752/1024 ≈ 2.69  ≈ 8/3    (接近最优)
# 1.3B: 5760/2048 ≈ 2.81  ≈ 8/3    (接近最优)

# FlashMHF 的解决方案——引入 E 个子网络:
# 每子网络 internal dim d_e ≈ 8/3·d_h = 8/3·128 ≈ 342
# 128M: E=8, d_ff_total = 8×342 = 2736, d_ff/d_h = d_e/d_h = 342/128 ≈ 2.67 ≈ 8/3 ✓
# 370M: E=7, d_ff_total = 7×342 = 2394, d_ff/d_h = d_e/d_h = 342/128 ≈ 2.67 ≈ 8/3 ✓
# 1.3B: E=15, d_ff_total = 15×342 = 5130, d_ff/d_h = d_e/d_h = 342/128 ≈ 2.67 ≈ 8/3 ✓

# 关键洞察: FlashMHF 通过 sub-network 分解将 ratio 锁定在 d_e/d_h ≈ 8/3,
#           而非 d_ff/d_h。每 sub-network 内 ratio 平衡，多个 sub-network 并联
#           提供足够的 total capacity 和 representational diversity。
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该术语的实际使用：(1) 诊断工具——当设计新的 FFN 变体时，检查 d_ff/d_h（或等效等效扩展比例）是否在 2-4 范围内。若超出此范围，预期 parameter efficiency 下降；(2) 解决策略——FlashMHF 采用 parallel sub-network decomposition（dense MoE）将单一路径拆解为 E 个 balanced-ratio 的 sub-path，其他可能的策略包括增加 d_h（但减少 H 降低 diversity）、调整 d_ff 增长策略、或使用 non-uniform head sizes；(3) 跨模型 scale 的行为预测——此 ratio 可用于预测新 FFN 架构在不同模型 size 下的 scalability；(4) 实验验证——FlashMHF 的 128M vs 370M 消融实验提供了 direct evidence：MH-FFN 从 128M 的 gain 到 370M 的 failure，唯一的变化就是 d_ff/d_h 从 16→21（ratio 恶化），而加入 parallel sub-network 后恢复 gain。

涉及论文标题：
- Flash Multi-Head Feed-Forward Network
- Scaling Laws for Neural Language Models (Kaplan et al., 2020)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-Level Attention Sparsity via Softmax Thresholding（BLASST）是一种训练无关的动态稀疏注意力方法。在FlashAttention的block-wise online softmax过程中，对每个KV block计算local maximum score m̃_i^{(j)}，并与running maximum m_i^{(j)}比较。当 m̃_i^{(j)} - m_i^{(j)} < ln(λ) 时（即block的局部最大值远小于已见最大值），跳过该block的后续计算。推导逻辑：(1) softmax中每个score的指数exp(s_ij)都会除以全局exp最大值做数值稳定，(2) 因此block中所有score的贡献被exp(m̃_i^{(j)} - m_i^{(j)}) < λ上界限制，(3) 当λ足够小时，block对最终输出的贡献可忽略。跳过三项操作：softmax指数计算（CUDA core MUFU.EX2）、attention-value矩阵乘法（tensor core MMA）、Value block的HBM加载（仅decode kernel）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

BLASST的算法pipeline（基于FlashAttention的tiled online softmax改造）：

```
# Input: Q∈R^{L×d}, K∈R^{L×d}, V∈R^{L×d}, threshold λ
# Tiling: Q→T_r blocks of B_r, K/V→T_c blocks of B_c

for i in 1..T_r:                    # 遍历query blocks
    m = -∞, O = 0, l = 0            # 初始化online softmax状态
    for j in 1..T_c:                # 遍历KV blocks
        S_ij = Q_i × K_j^T          # [B_r×B_c] QK^T, tensor core BMM1
        m_local = rowmax(S_ij)      # block local maximum
        m_new = max(m, m_local)     # 更新running maximum
        
        if m_local - m_new < ln(λ): # BLASST核心: 跳过检查
            continue                # 跳过softmax + PV乘法 + V加载
        else:
            P_tilde = exp(S_ij - m_new)           # softmax (MUFU.EX2)
            l = exp(m - m_new)*l + rowsum(P_tilde) # 更新归一化因子
            O = exp(m - m_new)*O + P_tilde × V_j   # PV matmul (BMM2)
            m = m_new
    O_i = O / l                      # final renormalization
return {O_i}
```

关键特性：(1) skip decision使用已在FlashAttention中计算好的统计量（local max, running max），零额外overhead；(2) exp(m_local - m_new) ≤ λ保证了被跳过block的输出误差有理论上界（Appendix B）；(3) 同一阈值λ适用于所有attention head和layer，自动适应不同head的稀疏度分布。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

BLASST已集成到TensorRT-LLM（https://github.com/NVIDIA/TensorRT-LLM）和FlashInfer中。使用时仅需在attention接口传入一个scalar threshold λ。λ通过校准自动确定：在校准数据集上sweep不同threshold，记录(λ·L, sparsity)数据点，拟合 λ·L = α·exp(β·s)。推理时给定目标sparsity S和context length L，直接用 λ = α·exp(β·S)/L。Sparsity-aware training变体在fine-tuning的forward pass中应用BLASST，backward中被跳过block自然不收梯度，迫使模型将重要信息集中到高attention score block。

涉及论文标题：
- BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

## Sparsity-Aware Training for Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparsity-Aware Training for Attention是一种fine-tuning策略，在训练/微调过程中应用稀疏注意力模式（而非dense attention），使模型学会将重要信息集中在被保留的attention block中，从而在推理时应用稀疏attention时更robust。BLASST的实现方式：在fine-tuning的forward pass中应用BLASST的threshold-based block skipping（根据m̃-m < ln(λ)条件跳过block），backward pass中被跳过的block自然不接收梯度（因其forward未计算），无需auxiliary loss或architecture change。模型在训练中学会将关键attention信息集中在高score block中，使得推理时即使aggressive pruning（70-90% sparsity）也能维持准确率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

训练pipeline与标准fine-tuning的唯一区别在于attention forward pass：

```
# 标准fine-tuning forward:
O = FlashAttention(Q, K, V)           # dense attention

# Sparsity-aware training forward:
O = BLASST(Q, K, V, λ)               # sparse attention with threshold λ
# Backward: 自动通过O的computational graph反向传播
# 被跳过block无compute node → 自然无梯度回传
```

梯度仅流过实际参与计算的attention block。采用ProLong论文中的curriculum training策略逐步增加sparsity。效果（BLASST论文Figure 6）：在RULER benchmark上，sparsity-aware trained模型在50-75% sparsity区间的accuracy退化降低至training-free的1/1.7。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Sparsity-aware training的实现需要：(1) 可微的sparse attention kernel（如BLASST，forward中有skip decision但仅对non-skipped block构造compute graph）；(2) 稀疏模式在训练中保持固定（BLASST使用固定threshold λ，而非动态变化的sparsity pattern）；(3) curriculum scheduling——gradually降低threshold以增加sparsity。与pruning-aware training、dropout-based sparsity training等方法属于同一思路的不同实现。

涉及论文标题：
- BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding
- DSV: Exploiting Dynamic Sparsity to Accelerate Large-Scale Video DiT Training

DSV的两阶段训练是另一种sparsity-aware training范式，与BLASST的关键区别：(1) DSV使用可训练的sparsity predictor（低秩矩阵W_Q^lr, W_K^lr）来预测critical KV pairs，而非使用固定threshold；(2) DSV在Stage 1训练predictor、Stage 2激活稀疏计算，BLASST在fine-tuning中直接应用threshold；(3) DSV的稀疏模式是per-query动态的（每个query有不同critical KV set），BLASST是per-block固定的。DSV的predictor训练loss：L_approx = 0.95·CosLoss(QK_lr, QK_main) + 0.05·NormLoss(QK_lr, QK_main)，predictor的计算图从主模型detached。

## Threshold Calibration for Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Threshold Calibration for Sparse Attention是BLASST提出的自动阈值选择方法。由于固定阈值在不同context length下会产生不一致的sparsity（如λ=1e-3在4K context下sparsity仅23%，在64K下达到75%），需要动态适配阈值。校准过程（Algorithm 2）：(1) 在校准数据集D上做一次forward pass，计算所有attention scores；(2) 对每个候选threshold λ_j和每个样本(x_i, L_i)，从同一次attention scores中提取该λ_j下的achieved sparsity s_ij；(3) 记录数据点(λ_j·L_i, s_ij)；(4) 拟合指数模型 λ·L = α·exp(β·s)。一次forward pass即可覆盖所有候选threshold（因sparsity可从相同的attention scores离线计算不同λ的结果）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

校准算法pipeline：

```
# Input: calibration dataset D, threshold set Λ, sparsity bounds [s_min, s_max]
# Output: calibration parameters α, β

P = []  # data points
for (x_i, L_i) in D:
    # 单次forward pass → 获取所有attention scores
    all_scores = forward_pass(x_i)  # 使用dense FlashAttention
    
    for λ_j in Λ:
        # 从scores离线统计sparsity（不做重复forward pass）
        s_ij = measure_sparsity(all_scores, λ_j)  # 统计满足m̃-m < ln(λ)的block比例
        if s_min ≤ s_ij ≤ s_max:
            P.append((λ_j * L_i, s_ij))

# 拟合指数模型: λ·L = α·exp(β·s)
α, β = fit_exponential_model(P)

# 推理时: target sparsity S, context length L → λ = α·exp(β·S)/L
```

发现：λ与L成反比关系 λ = a/L，其中a = α·exp(β·S)。理论依据：attention scores按行归一化到和为1，更长序列中每个token的平均score更低，需要更小的threshold才能实现相同的sparsity。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实际部署流程：(1) 用~1000条RULER样本在不同context length（4K-64K）上执行一次dense forward pass，(2) sweep λ∈[1e-6, 1e-1]范围计算每个λ下的sparsity，(3) 拟合α,β参数，(4) 推理时仅需指定target sparsity S（如50%或75%），系统自动按λ=α·exp(β·S)/L设置threshold。校准后sparsity偏差仅~1.2%（Table 6），远优于固定threshold的~27%偏差。校准参数a在不同数据集间表现一致（Table 12），无需per-task retuning。

涉及论文标题：
- BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

## Multi-Head Latent Attention (MLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-Head Latent Attention (MLA) 是DeepSeek-V2/V3/R1系列模型提出的注意力机制变体，通过低秩联合压缩Key和Value进入共享latent空间来大幅减少KV cache内存占用。与MHA（Multi-Head Attention）和GQA（Grouped Query Attention）不同，MLA不直接存储每个head的K和V，而是将K/V投影到一个低秩latent表示 c_KV ∈ R^{d_latent}（d_latent << num_heads × head_dim），推理时仅缓存latent vector而非完整的KV cache。decoding时从latent vector上投影回各head的K和V。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MLA的算法pipeline（简化）：

```
# 压缩阶段（prefill，每个token）:
c_KV = x @ W_down     # x∈R^d, c_KV∈R^{d_latent}，latent压缩
# 缓存 c_KV 而非完整 K/V

# Decompression（decode，每个query token）:
K_i = c_KV @ W_UK_i + x @ W_KR_i   # 第i个head的K，from latent + rope部分
V_i = c_KV @ W_UV_i                 # 第i个head的V，from latent

# Attention计算（仍使用online softmax）:
S_i = Q_i × K_i^T / sqrt(d_k)
O_i = softmax(S_i) × V_i
```

MLA的核心trade-off：decode从memory-bound变为compute-bound（因latent decompression引入额外计算，但减少了HBM KV cache读取量）。BLASST paper验证了其在MLA上的兼容性：DeepSeek-R1使用MLA + BLASST，在60% sparsity下GPQA Diamond/Mmlu Pro/LiveCodeBench准确率几乎无退化（Table 11）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MLA已在DeepSeek-V2、DeepSeek-V3、DeepSeek-R1等模型中实现，开源代码在DeepSeek的HuggingFace模型仓库中。MetaAttention框架将MLA作为一种RowNorm-based attention variant支持。从实现角度看，MLA的关键是latent space的维度选择（d_latent通常为512或576），以及rope positional encoding与latent-compressed部分和non-compressed部分的组合方式。

涉及论文标题：
- BLASST: Dynamic BLocked Attention Sparsity via Softmax Thresholding

## Dynamic Neural Networks (Dynamic DNNs)

术语是什么？
Dynamic Neural Networks（动态神经网络）是一类在执行路径或架构配置上依赖于运行时输入的深度神经网络。与静态 DNN（所有输入沿同一计算图执行）不同，动态 DNN 根据不同输入选择不同的执行路径、激活不同的子网络或调整网络宽度/深度。典型例子包括：(1) InstaNAS——根据输入图像由 controller 网络动态选择最优的子架构路径（如跳过某些卷积层、使用不同 kernel 大小）；(2) Dynamic Routing Networks——根据输入特征自适应选择从不同路径的语义分割专家网络；(3) Conditional Convolution——根据输入动态计算卷积权重（Mixture of Experts 风格，多个 expert 的权重由 gating 网络按输入加权）。动态 DNN 的设计目标是减少 FLOPs 和推理延迟（特别适合边缘设备），但其 input-dependent 的计算图给 GPU 并发调度带来挑战。

从算法pipeline角度拆解术语：
InstaNAS（实例感知神经架构搜索）动态推理的伪代码：
```
Algorithm: InstaNAS Dynamic Inference
Input: image x, supernet with N possible paths
Output: class prediction

// Phase 1: Controller selects architecture based on input
architecture_config = ControllerNet(x)  
// config: {layer1_skip: True, layer3_filters: 64, layer5_path: "B", ...}

// Phase 2: Execute selected sub-graph
for each layer in supernet:
    if architecture_config[layer].skip:
        continue  // 跳过该层
    
    if architecture_config[layer].is_dynamic:
        // 动态选择路径 (如不同kernel size)
        path = architecture_config[layer].path
        x = execute_path[layer][path](x)
    else:
        x = execute_static[layer](x)

return classifier(x)
```

在 GPU 上执行时，每个 controller 选择的子图对应不同的 kernel 序列（不同的 kernel 类型、不同的 kernel 大小），且每个 input image 产生不同的序列。这导致：(1) 无法提前构建全局 kernel DAG；(2) 大量小 kernel（Conv 2D 被分成多个小的 tile-based kernel，每层可能有多个 kernel）；(3) GPU occupancy 低（InstaNAS-A 在 RTX 3060 上仅 39%）。

术语一般如何实现？如何使用？
动态 DNN 的实现框架：PyTorch（通过 `if/else` 控制流和动态图机制天然支持）、TensorFlow（需要 `tf.cond` 等动态控制算子）。常见动态机制：(1) early exit（提前退出，如 BranchyNet，当中间层置信度足够高时提前输出预测）；(2) layer skipping（如 SkipNet，用轻量 gate 决定跳过哪些层）；(3) adaptive width/depth（如动态选择 channel 数或 block 数）。ACS 论文评估了 InstaNAS（CIFAR10, InstaNAS-A 架构）、Dynamic Routing（Cityscapes, Dynamic-A 16-layer）、Conditional Convolution（4 experts, EfficientNet-B4 backbone, ImageNet）三类动态 DNN。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs

## Brax (Physics Simulation Engine for Deep RL)

术语是什么？
Brax 是 Google 开发的基于 JAX 的可微分物理仿真引擎，专门用于大规模刚体仿真以支持深度强化学习（Deep RL）训练。Brax 将物理仿真（刚体碰撞检测、接触力计算、关节约束求解等）映射为 GPU kernel，通过 JAX 的 XLA 编译器生成 CUDA kernel 在 GPU 上并行执行。Brax 支持多种仿真环境（Ant、Humanoid、Grasp、Cheetah、Walker2d 等 MuJoCo 兼容环境），每个环境包含：刚体定义（link 质量、惯性、几何形状）、关节约束（joint 类型、运动范围）、actuator 定义（驱动力矩）、碰撞检测和接触动力学。在 Deep RL 训练中，Brax 用于数据采集阶段——agent 的策略网络（DNN）根据当前环境状态输出动作，Brax 在 GPU 上并行仿真多个环境实例，产生下一状态和奖励。

从算法pipeline角度拆解术语：
Brax 中一次 Deep RL 训练步骤的 pipeline：
```
Algorithm: Deep RL Training Step with Brax
Input: policy_net (DNN), env (Brax simulation environment)
Output: training_batch

// Phase 1: Data Collection (在GPU上)
// 并行模拟 N=4096 个环境实例
states = env.reset(batch_size=4096)   // (4096, state_dim) on GPU

for step in range(rollout_length):     // 如 rollout_length=10
    // 1. 策略推理: 所有环境实例并行
    actions = policy_net(states)        // 大量小kernel → GPU occupancy低
    
    // 2. 物理仿真步进: 每步包含数百个小kernel
    //    - collision_detection: 检测刚体对之间的碰撞
    //    - contact_forces: 计算接触力
    //    - joint_forces: 计算关节约束力
    //    - forward_dynamics: 从力计算加速度
    //    - integrate: 更新位置和速度
    next_states = env.step(states, actions)  
    // 小kernel问题: 每次碰撞检测根据input不同有不同的kernel路径
    // (哪些刚体对接触取决于当前状态 → input-dependent graph)
    
    states = next_states
    trajectory.append((states, actions, rewards))

// Phase 2: Policy Update (在GPU上)
// 从trajectory采样batch训练policy_net
loss = policy_update(trajectory)  // 标准DNN训练，计算量通常可饱含GPU
```

数据采集阶段占 Deep RL 训练时间的 30-70%（取决于环境复杂度），而该阶段 GPU occupancy 仅约 34%（平均），原因正是 Brax 产生的大量小 kernel——每个仿真步骤需要数百个 kernel launch，每个 kernel 仅有少量 CTA（中位数 < 200 CTA），无法填满 GPU（RTX 3060 28 SM）。

术语一般如何实现？如何使用？
Brax 开源（github.com/google/brax），基于 JAX（jax.readthedocs.io）。安装：`pip install brax`。使用流程：(1) 选择/自定义仿真环境（`brax.envs.create("ant")`）；(2) 调用 `env.reset()` 初始化状态；(3) 调用 `env.step(actions)` 推进仿真；(4) JAX 的 `vmap` 自动并行化多个环境实例。Brax 训练 pipeline 集成在 `brax.training` 中，包括 PPO、SAC 等 RL 算法。ACS 论文使用 Brax 的 5 个 MuJoCo 环境（Ant, Grasp, Humanoid, Cheetah, Walker2d）评估 ACS 的效果，在 ACS-HW 下实现 Deep RL 训练端到端加速 1.42×（平均）。

涉及论文标题：
- ACS Concurrent Kernel Execution on Irregular, Input-Dependent Computational Graphs

## Implicit GEMM Convolution Algorithm

术语是什么？
Implicit GEMM Convolution Algorithm 是一种将 2D 卷积（Conv2D）操作映射为矩阵乘法（GEMM）而在 GPU 上高效执行的算法，由 NVIDIA CUTLASS 库实现。与传统的 im2col 方法不同（im2col 先将输入图像显式展开为大型卷积矩阵，再调用 GEMM），Implicit GEMM 在从 global memory 加载数据到 shared memory 时**即时**构造卷积矩阵的 tile，避免了 im2col 矩阵的额外内存分配和带宽开销。映射关系：输入 image tensor x(NHWC) 展开为矩阵 A[NHW × RSC]，filter tensor w(KRSC) 视为矩阵 B[RSC × K]，输出 y(NPQK) 对应矩阵 C[NPQ × K] = A × B。

从算法pipeline角度拆解术语：
Implicit GEMM 的计算映射过程：
```
// 2D Convolution参数:
//   Input:  x[N, H, W, C]      batch × height × width × in_channels
//   Filter: w[K, R, S, C]      out_channels × kernel_h × kernel_w × in_channels
//   Output: y[N, P, Q, K]      batch × out_h × out_w × out_channels
//   (P, Q由padding和stride决定)

// 映射为GEMM: C = A × B
//   A矩阵: [N*P*Q, C*R*S]  — 每个输出像素对应一行，每行含感受野所有输入channel元素
//   B矩阵: [C*R*S, K]      — 每个filter展开为一列
//   C矩阵: [N*P*Q, K]      — 输出

// 传统im2col方法:
float* A_col = im2col(x, N, H, W, C, R, S, P, Q);  // 显式构造，内存开销大
gemm(A_col, w_reshaped, y_reshaped, N*P*Q, K, C*R*S);

// Implicit GEMM方法 (CUTLASS):
// 不显式构造A_col，而是在tile加载时计算A的索引映射
// 对每个tile (m_tile, k_tile):
//   A[m_start..m_start+M_tile, k_start..k_start+K_tile]
//   通过反向索引计算源自x的哪个 (n, h, w, c) 位置:
//     n = m / (P * Q)
//     p = (m % (P * Q)) / Q
//     q = (m % (P * Q)) % Q
//     对filter的 (r, s, c) → k索引, 取x[n, p*stride+r-pad, q*stride+s-pad, c]
```
cuSync 中，两个依赖 Conv2D 经 Implicit GEMM 后的依赖关系：第二个 Conv2D 的 Implicit GEMM 的每个 consumer tile 依赖第一个 Conv2D 的 Implicit GEMM 的所有 column tile。这通过 DSL 描述为 `Dep dep({g2, Tile(x,y)}, {g1, Tile(x/(R*S), y)})`，cuSyncGen 据此生成 RowSync（每行 row 一个 semaphore）和 Conv2DTileSync（每 tile 一个 semaphore）。

术语一般如何实现？如何使用？
CUTLASS 中 Implicit GEMM 的实现使用专门的 iterator（`conv2d_fprop_activation_tile_access_iterator` 和 `conv2d_fprop_filter_tile_access_iterator`）在 tile 加载时计算地址偏移。推荐配置：所有 tensor 128-bit 对齐的 NHWC 布局，channel 数 C 和 K 为 32 的倍数，使用 `kOptimized` iterator 模式预计算指针增量。cuSync 通过修改 CUTLASS Conv2D kernel 的 tile 加载循环添加 wait/post 同步点来支持 Implicit GEMM 的细粒度同步，修改量约 22 行（0.6% 的 CUTLASS Conv2D 代码）。实验显示，对 ResNet-38 和 VGG-19，cuSync 同步 Conv2D kernel 后最多减少 22% 推理时间。

涉及论文标题：
- A Framework for Fine-Grained Synchronization of Dependent GPU Kernels

## Any-to-Any (A2A) Multimodal Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Any-to-Any（A2A）多模态模型是一类新兴的多模态模型，能够接受文本和多种模态数据（图像、视频、音频）的任意组合作为输入，并生成任意组合的模态输出。截至 2026 年 3 月，Hugging Face 上有超过 11,000 个 A2A 模型变体。代表性模型包括：Qwen Omni 系列（接受 T/I/V/A 输入，生成 T/A 输出）、InternVL 3（T/I/V→T）、DeepSeek Janus（T/I→T/I）、LTX-2（T/I→V/A）、Qwen Image（T→I）。传统 text-only LLM 或仅生成图像/视频的 Diffusion 模型是 A2A 的特例——所有请求沿同一线性 pipeline 遍历所有 component。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

A2A 模型的计算图结构（以 Qwen 3 Omni 为例）：

```
Component Graph (DAG):
  E_img (Vision Encoder) ──┐
  E_vid (Video Encoder)  ──┼──► L_th (Thinker LLM) ──► L_ta (Talker LLM) ──► G_aud (Vocoder)
  E_aud (Audio Encoder)  ──┘         │                        │                    │
                                      ▼                        ▼                    ▼
                                   text output             audio tokens          audio waveform

Request Types (不同输入/输出组合遍历不同子图):
  ① T+I → T:     E_img → L_th → text output
  ② T+I+V → T:   E_img → E_vid → L_th → text output
  ③ T+I → A:     E_img → L_th → L_ta → G_aud → audio output
  ④ T+I+V+A → A: E_img → E_vid → E_aud → L_th → L_ta → G_aud → audio output

关键特性:
  - 不同 request type 遍历不同子图 → 各 component 面临不同 request rate
  - 不同 component 的计算特性差异极大:
    Qwen 3 Omni on A100: E_aud 21.43 req/s vs G_aud 0.12 req/s (178× 差异)
    Thinker LLM 2.15 req/s vs Talker LLM 0.12 req/s (18× 差异)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

A2A 模型的实现通常基于：(1) 预训练的 modality-specific encoder（Vision Transformer for image/video, Whisper-style encoder for audio），将多模态输入编码为 unified embedding；(2) 核心 LLM（如 Qwen 系列）进行跨模态理解和推理；(3) modality-specific generator（如 Diffusion Transformer for image, autoregressive + vocoder for audio）。Serving 时，不同 executor type 处理不同的 component 集合：encoder executor（处理多模态输入→embedding）、LLM executor（autoregressive 生成）、DiT executor（扩散去噪）、vocoder executor（token→waveform）。vLLM-Omni 和 SGLang-Omni 提供通用的 component-wise disaggregation 机制，Cornfigurator 在此基础上增加自动规划。

涉及论文标题：
- Cornserve Efficiently Serving Any-to-Any Multimodal Models

## Component Heterogeneity in A2A Multimodal Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Component Heterogeneity（组件异构性）在 A2A 模型中表现为两个维度：(1) Request Type Heterogeneity——不同 request type（不同输入/输出模态组合）遍历模型 component graph 的不同子图，导致每个 component 面临不同的 request rate；(2) Computational Heterogeneity——不同 component 有巨大差异的资源需求和计算特性。Cornfigurator 论文的 Table 2 量化了这一点：Qwen 3 Omni 在 A100-80GB 上，audio encoder 的吞吐是 21.43 req/s 而 vocoder 仅 0.12 req/s（178× 差异），thinker LLM 2.15 req/s vs talker LLM 0.12 req/s（18× 差异）。两种异构性叠加导致各 component 负载极度不均衡，使固定部署策略在不同 workload 下性能差异显著。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Component heterogeneity 对 Serving 的影响（Qwen 3 Omni 例子）：

```
给定 workload: π_text=2/3, π_audio=1/3
  各 request type 的 component 调用:
    text-output types (①-④): 需要 E_img/E_vid/E_aud + L_th
    audio-output types (⑤-⑧): 需要 E_img/E_vid/E_aud + L_th + L_ta + G_aud

Per-component request rate:
  E_img: 100% of requests (所有 type 都含 image input)
  L_th:  100% of requests (所有 type 都经过 thinker)
  L_ta:  33% of requests (仅 audio-output types)
  G_aud: 33% of requests (仅 audio-output types)

瓶颈分析 (假设各 component 独立部署):
  L_th 吞吐 = 2.15 req/s  →  100% load → 需要 1/2.15 ≈ 0.47 GPU-seconds/req
  L_ta 吞吐 = 0.12 req/s  →   33% load → 需要 0.33/0.12 ≈ 2.75 GPU-seconds/req
  G_aud 吞吐 = 0.12 req/s →   33% load → 需要 0.33/0.12 ≈ 2.75 GPU-seconds/req

→ L_ta 和 G_aud 是瓶颈 (每 req 消耗的 GPU 资源远多于 L_th)
→ 最优部署: 大量 GPU 分配给 L_ta+G_aud, 少量给 L_th+encoders
  Cornfigurator 16GPU plan: 4×(E_img+E_vid+L_th) + 11×(L_ta+G_aud)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Component heterogeneity 是 Cornfigurator 规划的核心动机。传统方法（monolithic 或 fixed disaggregation）无法自动适应 heterogeneity——monolithic 使 slowest component 成为全部模型的瓶颈；fixed disaggregation 可能将低负载 component（如图像 encoder 在 audio-heavy workload 下）分配到过多 GPU 导致资源浪费。Cornfigurator 通过 per-request-type reasoning 和计划枚举自动找到匹配 heterogeneity pattern 的最优 colocation/disaggregation 组合。

涉及论文标题：
- Cornserve Efficiently Serving Any-to-Any Multimodal Models

## Multimodal Component Graph (A2A Model Architecture)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multimodal Component Graph 是 A2A 模型的计算结构抽象——一个由异构 component 节点和有向数据依赖边组成的有向无环图（DAG）。每个节点代表一个处理特定模态的模型组件（如 Vision Encoder, Thinker LLM, Talker LLM, Vocoder），每条边代表组件间的数据流（如 encoder 输出的 embedding 流入 LLM）。不同 request type 沿不同路径遍历该图——每条路径对应一种输入/输出模态组合。在 Cornfigurator 中，Model Definition 就是这个 component graph，作为 planner 的输入之一。Cornfigurator 支持将某些边标记为 colocatable（对应的两个 component 可被 MERGE 到同一 executor 或 KEEP 分离），planner 枚举所有 colocatable edge 的 Keep/Merge 组合来探索部署策略空间。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Component graph 的抽象和 request type 路径映射：

```
Model Definition Graph G = (C, E):
  C = {E_img, E_vid, E_aud, L_th, L_ta, G_aud}
  E = {(E_img→L_th), (E_vid→L_th), (E_aud→L_th), (L_th→L_ta), (L_ta→G_aud)}
  Colocatable edges E_c: 论文未明确说明完整 E_c，runtime 确定哪些边可 colocate

Request Type → Subgraph 映射:
  f: (input_modalities, output_modality) → path ⊆ G
  例: f(T+I, T) = [E_img, L_th]
      f(T+I+V, A) = [E_img, E_vid, L_th, L_ta, G_aud]

Component 计算量 (以 Qwen 3 Omni on A100 为例):
  E_img: 5.43 req/s   (较快的视觉编码)
  E_vid: 2.93 req/s   (视频编码，多帧处理)
  E_aud: 21.43 req/s  (音频编码最快)
  L_th:  2.15 req/s   (thinker LLM, 自回归)
  L_ta:  0.12 req/s   (talker LLM, 最慢——生成 audio tokens)
  G_aud: 0.12 req/s   (vocoder, token→waveform)

Planner 枚举: 对每条 e∈E_c 选择 KEEP 或 MERGE
  Fully disaggregated: all KEEP → 6 nodes, 每个可独立配置
  Monolithic: all MERGE → 1 node, 所有 component 共享配置
  Cornfigurator 最优: 部分 KEEP, 部分 MERGE — 例 audio encoder 分离,
    thinker+encoder colocated, talker+vocoder colocated
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Component graph 的 formalization 是 Cornfigurator 能够处理通用 A2A 模型（而非仅 MLLM 或 Diffusion 等特例）的关键——只要将模型表达为 component DAG + colocatable edges，planner 就能自动搜索部署方案。对于实际 A2A 模型，component 数量通常 ≤ 10（Qwen Omni 有 6 个），使得枚举空间可管理。Graph 定义需包含：节点（component 名称和类型）、边（数据依赖）、colocatable edges 标记、以及每个节点支持的 executor types。论文未明确说明 graph definition 的具体格式/API。

涉及论文标题：
- Cornserve Efficiently Serving Any-to-Any Multimodal Models
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend

## Discrete Diffusion Language Model (DLM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

离散扩散语言模型（Discrete Diffusion Language Model, DLM）是一类将文本生成建模为离散token空间上迭代去噪过程的生成模型。与自回归模型逐token从左到右生成不同，DLM初始化整个序列为[MASK] token（absorbing state），然后通过多步反向扩散过程逐步预测和填充token。核心数学框架：(1) 前向过程：$q(x_t|x_0) = \alpha_t x_0 + (1-\alpha_t) \mathbf{m}$，其中$\alpha_t = \prod_{i=1}^t(1-\beta_i)$，$\mathbf{m}$为[MASK]的one-hot表示；(2) 反向过程：$p_{\theta}(x_{t-1}|x_t)$通过神经网络学习近似反向transition，使用bidirectional attention同时建模所有位置；(3) 训练损失：reweighted cross-entropy $\mathcal{L}_D = \mathbb{E}_t[\frac{1}{t}\mathbb{E}_{q(x_t|x_0)}[-\sum_n \delta_{x_t^n,\mathbf{m}}(x_0^n)^\top \log f_{\theta}(x_t)^n]]$，仅对masked位置计算loss。代表模型：MDLM (NeurIPS 2024)、Dream 7B（从Qwen2.5-7B微调，580B tokens训练）、LLaDA、Mercury、Gemini Diffusion。Dimple使用Dream作为DLM backbone。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

DLM的训练和推理pipeline（以absorbing state DLM为例）：

```
# === 训练阶段 ===
Input: 文本序列 x_0, 词汇表 V (含[MASK] token)
Forward pass:
  1. 采样时间步 t ~ Uniform(0, 1]
  2. 计算mask概率: p_mask = 1 - α_t
  3. 对每个token独立以概率p_mask替换为[MASK]: x_t = mask(x_0, p_mask)
  4. 将x_t输入bidirectional Transformer (full attention over all positions)
  5. 对所有被masked的位置输出logits预测原始token: f_θ(x_t) ∈ R^{L×V}
  6. 计算损失（仅masked位置）: L = -Σ_{n: x_t^n=[MASK]} log softmax(f_θ(x_t)^n)[x_0^n] / t

# === 推理阶段（以MaskGIT为例）===
Input: 目标序列长度 L, 解码步数 T
初始化: x_T = [[MASK], ..., [MASK]]  (L个[MASK])
For t = T down to 1:
  1. z_t = f_θ(x_t)  # bidirectional forward
  2. p_t = softmax(z_t)
  3. confidence c^(i) = max(p_t^(i)) for i in masked positions
  4. 选择K = ceil(L * t/T) 个最高置信度位置
  5. 对选中位置采样: x_{t-1}^(i) ~ Categorical(p_t^(i))
  6. 其余位置保持[MASK]
Output: x_0（所有位置已去mask的token序列）
```

Annotations: $x_0$: 无噪声token序列; $x_t$: 时间步$t$的噪声序列; $\alpha_t$: 信号保留率; $\beta_i$: 每步噪声率; $f_\theta$: bidirectional Transformer; $T$: 总解码步数（8-64步）; $L$: 目标序列长度; 关键区别：DLM使用bidirectional attention（vs AR的causal），所有位置同时预测（vs AR逐位置）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DLM通常基于预训练AR LLM初始化：从AR LLM checkpoint加载权重，将causal attention替换为bidirectional，使用masked language modeling loss在大规模数据上训练。Dream (https://github.com/DreamLM/Dream) 从Qwen2.5-7B初始化，使用580B tokens训练。推理使用迭代解码策略（MaskGIT或变体），通过confidence-based selection逐步去mask。优势：并行解码（每步可同时预测多个token）、bidirectional context利于planning/infilling、可控性（精确输出长度和结构）。

**Fast-dLLM的加速贡献**：Fast-dLLM针对DLM推理提出了训练无关的加速方法：(1) Block-wise近似KV Cache利用双向注意力相邻步KV激活高余弦相似度的特性，在分块解码中缓存和复用prefix/suffix的K/V矩阵，减少重复的全注意力计算；(2) Confidence-Aware Parallel Decoding通过理论保证的安全并行解码减少总解码步数。两者结合在LLaDA 8-shot gen_len=1024上实现27.6×端到端加速。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

## Discrete Diffusion Multimodal Large Language Model (DMLLM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Discrete Diffusion Multimodal Large Language Model (DMLLM) 是将视觉编码器与离散扩散语言模型（DLM）结合的多模态大模型。与标准MLLM（如LLaVA）使用自回归生成不同，DMLLM使用扩散过程生成文本回答。架构组成：(1) Vision Encoder（如Qwen2.5-VL ViT，冻结）编码图像为visual tokens；(2) Projector（2层MLP）将visual tokens映射到LLM embedding空间；(3) DLM Backbone（如Dream）处理拼接后的visual+text tokens，通过bidirectional attention和迭代去噪生成回答。Dimple是首个公开的DMLLM，证明DMLLM在相似训练预算下可达到与自回归MLLM相当的性能（13个benchmark平均62.4% vs LLaVA-NEXT 58.5%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

DMLLM推理pipeline（Dimple为例）：

```
Input: 图像 I, 文本问题 Q, 预定义生成长度 L_answer

1. Vision Encoding:
   visual_tokens = VisionEncoder(I)  # 冻结ViT, N_v个token
   visual_emb = Projector(visual_tokens)  # 2层MLP → LLM dim

2. Input Construction:
   prompt = [BOS] + visual_emb + text_emb(Q) + [EOS]
   answer_init = [[MASK]] * L_answer  # 全部初始化为[MASK]
   x_T = concat(prompt, answer_init)  # 总长度 L = L_prompt + L_answer

3. Structure Prior（可选）:
   预置特定位置token（如"Thus, the answer is \box{"），标记为"已确定"

4. Iterative Diffusion Decoding:
   For step t (直到所有[MASK]被填充):
     a. z_t = f_θ(x_t)  # bidirectional forward（可能使用Prefilling）
     b. p_t = softmax(z_t)  # pre-revision probabilities
     c. For each masked i: c^(i) = max(p_t^(i))  # confidence
     d. 选择c^(i) >= γ 的位置一次性批量更新；若无则fallback随机选择K个

5. Output: 去除[Padding] tokens，提取有效文本
```

Annotations: L_answer由response_length参数预设（Dimple使用4/8/16/64取决于benchmark）；N_v取决于图像分辨率和patch size；[Padding]是Dream tokenizer中的特殊token，用于填充answer长度不足的部分。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DMLLM训练使用Autoregressive-then-Diffusion策略：(1) Phase I AR Alignment & Tuning: causal attention + next-token prediction进行视觉-语言对齐和instruction tuning；(2) Phase II Diffusion Tuning: 恢复bidirectional attention + masked LM loss，仅mask answer部分，复用相同instruction数据。此策略解决纯扩散训练的两个低效——监督覆盖率低、每样本仅一个timestep监督。推理时Confident Decoding可将迭代数压缩到response_length/3左右。已开源：https://github.com/yu-rp/Dimple。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding

## MaskGIT Decoding Algorithm

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MaskGIT（Masked Generative Image Transformer, CVPR 2022）是一种基于置信度的迭代并行解码算法，最初设计用于图像生成，后被广泛应用于离散扩散语言模型。核心思想：将所有位置初始化为[MASK]，每步通过bidirectional Transformer预测所有masked位置的token分布，按置信度排序选择最高置信度的K个位置解码，其余保持[MASK]，迭代直到完成。与自回归解码相比：每步可并行解码多个token，总步数远小于序列长度。MaskGIT使用cosine mask schedule $\gamma(t/T)$ 决定每步解码的token数量：$\gamma(r) = \cos(\pi r/2)$，从$\gamma(0)=1$（全mask）到$\gamma(1)=0$（全解码）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Input: 序列长度 L, 总步数 T, 温度 τ
初始化: x_T = [[MASK], ..., [MASK]]

For step t = T, T-1, ..., 1:
  z_t = f_θ(x_t)  # bidirectional parallel predict
  p_t = softmax(z_t / τ)
  For each masked i: c^(i) = max(p_t^(i))  # confidence
  n_masked = count([MASK])
  K = ceil(n_masked * cos(π * (t/T) / 2))  # cosine schedule
  或固定策略: K = ceil(L * t/T)
  I_t = TopK({c^(i)}, K)
  For i in I_t: x_{t-1}^(i) ~ Categorical(p_t^(i))
  其余保持[MASK]

Return: x_0
```

Annotations: T典型值8-12（图像）或等于L（文本）；τ=0时退化为greedy；Dimple评估时设τ=0、每步1 token以保证确定性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MaskGIT最初在google-research/maskgit (JAX)实现，后被移植到PyTorch。在DLM中，Dream使用MaskGIT作为默认解码算法。实现要点：(1) confidence使用pre-revision概率（在temperature/top-p调整前），避免revision后概率退化；(2) temperature调度——初始高temperature增加多样性，最终低temperature确保质量。优势：并行解码2-64x speedup vs AR。局限性：需预定义序列长度；对mask schedule敏感。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding

## Confident Decoding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Confident Decoding是Dimple提出的动态token选择策略，用于改进离散扩散模型的推理效率。与标准MaskGIT每步解码固定数量token（由schedule决定）不同，Confident Decoding基于绝对置信度阈值$\gamma \in (0,1)$动态决定每步解码的token数量。核心motivation：文本不同位置的token可预测性差异大——固定短语很早就可高置信预测，而复杂推理位置需要更多上下文。Fixed schedule忽略这种异质性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Input: x_t, logits z_t, τ, γ, fallback K

p_t = softmax(z_t)              # pre-revision (用于confidence)
p̃_t = softmax(z_t / τ)          # post-revision (用于采样)

For each masked i:
  c^(i) = max(p_t^(i))          # confidence
  x̃^(i) ~ Categorical(p̃_t^(i))  # 候选token

If ∃i, c^(i) ≥ γ:
  I = {i | c^(i) ≥ γ}           # 批量更新高置信位置
  For i in I: x_{t+1}^(i) = x̃^(i)
Else:
  I = RandomSample({1..N}, K)   # fallback: 随机选K个
  For i in I: x_{t+1}^(i) = x̃^(i)

Return x_{t+1}
```

Annotations: γ=0.7为Dimple经验值；confidence使用pre-revision概率（不受temperature影响，保留位置间相对关系）；Fallback保证即使无高置信度位置也能推进生成。典型效果：22 token仅需7次迭代完成（~1/3 response_length）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Confidence计算使用pre-revision概率——因为temperature/top-p等revision以position-wise方式应用，破坏跨位置relative confidence ranking。Confidence函数可选max probability（最简单）、entropy、或概率margin。阈值γ需调优：太高→频繁fallback（低效）；太低→低质量token过早解码（质量差）。可与任何decoding algorithm组合（MaskGIT、随机选择等）。预期加速：Dimple上将迭代数压缩到response_length的1/3至1/2。

**Fast-dLLM的理论扩展**：Fast-dLLM对confidence-aware parallel decoding进行了严格的理论分析（Theorem 1）。当n个token的边际置信度均满足p_j(X_{i_j}=x_{i_j}|E) > 1-ε，且(n+1)ε ≤ 1时，greedy parallel decoding（乘积边际分布的argmax）等价于greedy sequential decoding（真实联合分布的argmax）。该定理同时给出了L_p距离上界D_TV < (3n-1)ε/2和前向KL散度上界D_KL < (n-1)[H_b(ε) + ε·ln(|V|-1)]，量化了乘积分布对真实联合分布的逼近程度。

基于此定理，Fast-dLLM提出两种实用策略：(1) **Threshold策略**：仅解码c_i > τ的token，始终保底解码max confidence token以避免死循环；(2) **Factor策略**：排序置信度后找最大n使(n+1)(1-c^(n)) < f，动态控制并行度。Factor策略通常比threshold策略提供1.4-1.5×更高吞吐量（代价约1-3%准确率）。Fast-dLLM在LLaDA上实现置信度感知并行解码单独加速13.3×（8-shot, gen_len=1024），与KV Cache结合达27.6×。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

## Structure Prior

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Structure Prior是Dimple提出的离散扩散模型输出控制机制。由于扩散模型在生成前已知完整序列长度，且任意位置token可独立预测，可在初始化时预置特定位置的token值，这些位置在后续迭代解码中始终保持不变（不被mask、不参与更新）。Structure Prior实现：(1) 精确输出格式控制（JSON/LaTeX），不依赖instruction prompt间接引导；(2) 推理步骤结构控制（如强制先描述image1再image2）；(3) 精确长度和结束位置控制。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Structure Prior使用流程（以Structured Reasoning为例）：

```
Response Length: 64 tokens

1. 定义Structure Priors:
   Prior 1: position[0:5] = "In the first image, there "
   Prior 2: position[20:25] = "In the second image, there "
   Prior 3: position[50:58] = "The common item in the two images is"

2. 初始化:
   x_T[0:5] = tokenize("In the first image, there ")    # 固定
   x_T[20:25] = tokenize("In the second image, there ")  # 固定
   x_T[50:58] = tokenize("The common item...")           # 固定
   x_T[其他] = [MASK]  # 正常参与扩散

3. 迭代去噪: 每步仅对[MASK]位置预测/更新；固定位置永不被mask
   结果: 第10步解码"scissors"（最终答案）→ 答案先于完整推理步骤出现

4. Length Control变体:
   在position[L-12:L-4]预置 "Thus, the answer is \box{"
   强制模型在此位置输出最终答案，自动调整前序推理跨度
```

Annotations: Prior token值通过tokenizer映射后直接写入序列；固定位置在attention中正常参与（可被attend），但不被更新；Prior可放在任意位置。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

扩散模型bidirectional generation的独特能力——AR模型无法实现（从左到右生成，无法在生成前指定后序token）。实现：初始化时将prior位置设为目标token IDs，解码循环跳过这些位置。用途：格式控制（JSON/XML/LaTeX）、推理引导（三段式分析）、长度控制。局限性：需预知总序列长度和prior绝对位置；prior可能与模型实际推理冲突。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding

## Autoregressive-then-Diffusion Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Autoregressive-then-Diffusion (AR-then-Diffusion) 是Dimple提出的混合训练范式，将DLM高效转化为DMLLM。先用AR训练建立多模态能力（监督信号覆盖率高），再用Diffusion训练恢复并行解码能力。解决纯扩散训练两个低效：(1) Masked LM每个样本仅对masked token计算loss，监督覆盖率低于next-token prediction；(2) 每个样本仅提供一个timestep的扩散监督（vs AR的causal attention确保每个生成步骤都被监督）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Phase 1a: AR Alignment (causal attention, lr=0.001, batch=256, data=LLaVA-CC3M 559k)
  L_AR = -Σ log p_θ(token_i | prompt, token_{<i})
  作用: 视觉-语言对齐（训练projector）

Phase 1b: AR Instruction Tuning (causal attention, lr=2e-5/5e-6, batch=128, data=LLaVA-NEXT 739k)
  作用: instruction following能力

Phase 2: Diffusion Tuning (bidirectional attention, lr=5e-7, batch=128, data=LLaVA-NEXT 739k复用)
  数据预处理: [EOS]→随机n个[Padding]; t~Uniform(0,1]; 仅mask answer部分
  L_D = (1/t) * Σ_{i: x_t^i=[MASK]} -log softmax(f_θ(x_t)^i)[x_0^i]
  作用: 恢复bidirectional attention + 扩散生成能力
```

Annotations: 三阶段总计~100 H100 GPU hours。关键: DLM (Dream) 从AR LLM微调而来，AR阶段causal attention不引入严重inductive bias。仅mask answer部分（prompt始终可见）。[EOS]→[Padding]替换因为扩散模型不依赖[EOS]终止。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

验证（Dimple Table 2）：纯扩散训练在9个benchmark上全面劣于AR+DT；AR alone有训练-推理gap；AT+DT在所有benchmark最优；显著缓解Length Bias（ChartQA accuracy从42.7%→8.6%变为稳定）。策略有效性基于DLM与AR LLM的数学统一性（吸收态扩散与AR均可描述为扩散过程，区别仅在transition matrix构造）。未来方向：更高效的Phase II训练策略以降低训练成本。

涉及论文标题：
- Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding

## Multimodal Inference Pipeline Stages (Encode / Prefill / Decode)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multimodal Inference Pipeline 是 EPD-Serve 定义的 MLLM 推理三阶段划分：(1) **Encode 阶段**：多模态编码器 E（Vision Transformer / Audio Encoder 等）将原始多模态输入 I_m 转换为高维特征向量序列 V_m ∈ R^{n×d}，作为 Prefill 阶段的输入特征；(2) **Prefill 阶段**：文本提示 I_t 编码为 V_t，拼接多模态特征 V_m + V_t 输入 LLM Decoder，执行首次前向传播生成首 token O_1 并构建全层 KVCache KV1；(3) **Decode 阶段**：基于 KVCache 和上一 token，LLM 自回归迭代生成后续 token O_i+1，直至 <eos> 或 max_length。三阶段具有显著的计算异质性：Encode 为 compute-heavy（ViT 参数 0.7-6B）、Prefill 为 memory+compute 混合（KV Cache 构建）、Decode 为 memory-bound（逐 token GEMV）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

EPD-Serve MLLM 推理 pipeline 的数学形式：

```
Encode 阶段:
  V_m = E(I_m)                    // I_m: 图像/音频/视频
  V_m ∈ R^{n×d}                   // n: visual tokens, d: feature dim
                                  // 例 openPangu-7B-VL: ViT 0.7B

Prefill 阶段:
  V_t = TokenEmbed(I_t)           // I_t: 文本 prompt
  O_1, KV_1 = LLM(V_m, V_t)      // 首 token + 全层 KVCache
                                  // attention: softmax(QK^T/√d_k)

Decode 阶段 (自回归循环):
  for i = 1 to max_length:
    O_{i+1}, KV_{i+1} = LLM(O_i, KV_i)  // 基于历史 KVCache
    if O_{i+1} == <eos>: break
```

三阶段计算特征对比（openPangu-7B-VL）：

| 阶段 | 模块 | 参数量 | 计算特征 | 瓶颈 |
|------|------|--------|----------|------|
| Encode | ViT | 0.7B | Compute-heavy (ViT forward) | AI Core utilization |
| Prefill | LLM(7B) | 7B | Memory+Compute (KVCache build) | Seq length quadratic |
| Decode | LLM(7B) | 7B | Memory-bound (GEMV/token) | HBM bandwidth |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

三阶段划分是实现 EPD Disaggregation 的基础——每个阶段被映射为独立可调度的实例进程。实例间通过：(1) E-P 异步特征预取（仅传 hash，从 MM Store 检索特征向量）；(2) P-D 分层分组 KV Cache 传输（按 Transformer 层打包，延迟调度对齐通信与计算）。三阶段的并行策略可按需独立配置：Encode 偏好数据/序列并行、Prefill 可根据序列长度选择 pipelining、Decode 偏好张量并行降低延迟。EPD-Serve 在论文中使用的模型为 openPangu-7B-VL (ViT 0.7B + LLM 7B) 和 Qwen3-VL-8B (ViT 0.6B + LLM 8B)，表明 pipeline 阶段划分适用于典型 MLLM 架构。Encode 阶段因 Attention 复杂度随序列长度平方增长，在某些场景下编码延迟可超过 LLM Prefill 时间，是该阶段的根本性能瓶颈。

涉及论文标题：
- EPD-Serve A Flexible Multimodal EPD Disaggregation Inference Serving System On Ascend

## Block-wise Approximate KV Cache for Diffusion LLM

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-wise Approximate KV Cache是Fast-dLLM提出的针对双向注意力扩散LLM的KV缓存机制。自回归模型使用causal attention mask，已生成的token不会受未来token影响，因此KV Cache可以精确复用。但扩散LLM使用full bidirectional attention——任意token的计算依赖所有其他token，每步生成后所有token的注意力分布都可能改变，导致标准KV Cache不可用。Fast-dLLM观察到相邻推理步之间的KV激活余弦相似度接近1（Figure 3, red boxed region），意味着在块内（block）解码的多个步中，前缀token的Key和Value几乎不变，可以安全近似复用缓存。基于此，Fast-dLLM采用分块生成策略：将输出序列分为K个块（每块B个token，默认B=32），块内多步解码复用cache的prefix K/V，块完成后全序列forward更新cache再进入下一块。

块大小B是关键的精度-速度trade-off：B太小→频繁cache更新增加开销；B太大→缓存失配精度下降（Figure 4，B=32最佳）。

两种变体：(1) **PrefixCache**：仅缓存prefix（prompt+已生成块）的K/V；(2) **DualCache**：额外缓存suffix（全[MASK]末尾块）的K/V，进一步减少attention计算量。DualCache在长序列上加速更强（8-shot gen_len=1024: DualCache 27.6× vs PrefixCache 18.6×）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

PrefixCache单块解码流程（DualCache在step 2额外缓存suffix K/V）：

```
Input: x (全序列: prompt + [MASK]×L), block k, block_size B, steps T
   s = |prompt| + (k-1)*B         # 当前块起始索引
   e = |prompt| + k*B              # 当前块结束索引
   
   # Step 1: Cache初始化（仅首次调用）
   K_prefix, V_prefix = compute_KV(x[0:|prompt|])    # 缓存prompt的K/V
   
   # Step 2: 块内迭代解码（复用cache）
   for t = 1 to T:
       Q = x[s:e] * W_Q                             # 仅当前块作为query
       K_rest = x[s:] * W_K                          # 剩余部分（当前块+suffix）的K
       V_rest = x[s:] * W_V                          
       
       S_prefix = Q * K_prefix^T                     # (B, |p|)
       S_rest   = Q * K_rest^T                       # (B, L_rest)
       S = concat([S_prefix, S_rest])
       P = softmax(S)
       
       O = P_prefix * V_prefix + P_rest * V_rest     # attention输出
       # ... 后续FFN层 ...
       
       # confidence计算 + 解码 ...
       if all_unmasked(x[s:e]): break
   
   # Step 3: Cache更新（块完成后，与forward融合无额外开销）
   K_full = compute_KV(x[:])                          # 全序列KV重算
   K_prefix = K_full[:e]                              # 扩展prefix cache到当前块结束
   V_prefix = V_full[:e]
```

计算量对比（单attention step）：
- 无Cache: QK^T 需要 (B, |p|+L) × (|p|+L, d)^T → O(B·(|p|+L)·d)
- PrefixCache: 仅计算Q×K_rest^T，O(B·L_rest·d)，省去prefix部分（大比例）
- DualCache: 仅计算Q×K_block^T，O(B²·d)，省去prefix+suffix

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Fast-dLLM在PyTorch eager模式实现：forward pass中，使用kvcache标志位控制是否跳过prefix/suffix的attention计算。cache存储在Python dictionary中（key: layer_id, value: (K_tensor, V_tensor)）。block size通过--block_size参数控制（默认32），使用lm-eval框架进行评估。DualCache需额外存储suffix位置的K/V，实现上将后缀token的position标记并在attention中对后缀部分使用null op。开源代码：https://github.com/NVlabs/Fast-dLLM（v1目录）。

涉及论文标题：
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

## τ-leaping for Discrete Diffusion Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

τ-leaping（tau-leaping）最初来自化学动力学中的Gillespie算法扩展，用于加速离散状态随机过程的模拟。在离散扩散模型中，τ-leaping是一种近似加速策略：精确反向扩散过程每步仅修改1个token（逐个token去噪），需要L步完成长度L的序列——极其低效。τ-leaping允许在单步中同时更新多个[MASK]位置的token，大幅减少所需步数。具体地，对于吸收态（masked）扩散的反向过程，τ-leaping近似为：

$$q_{s|t}(x_s^i | x_t) = \begin{cases} 1, & x_t^i \neq [\text{MASK}], x_s^i = x_t^i \\ \frac{s}{t}, & x_t^i = [\text{MASK}], x_s^i = [\text{MASK}] \\ \frac{t-s}{t} q_{0|t}(x_s^i | x_t), & x_t^i = [\text{MASK}], x_s^i \neq [\text{MASK}] \end{cases}$$

其中t和s为扩散时间步（s < t），x_t为当前带噪声序列，q_{0|t}为模型预测的干净数据分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LLaDA中τ-leaping推理（baseline，无KV Cache）：

```
Input: prompt p0, total_length L, num_steps T (e.g. 128)
   x = [p0; [MASK] * (L - |p0|)]       # 全[MASK]初始化
   timesteps = linspace(1, 0, T+1)      # T=128等分
   
   for step in range(T):
       t = timesteps[step]              # 当前噪声水平
       s = timesteps[step+1]            # 目标噪声水平
       
       logits = model(x)                # full bidirectional forward
       probs = softmax(logits)
       
       for each masked position i:
           if x[i] == [MASK]:
               # 以 (t-s)/t 概率解码为预测token，否则保持[MASK]
               with prob (t-s)/t:
                   x[i] = argmax(probs[i])
               else:
                   x[i] = [MASK]        # 概率 s/t
```

核心问题：多个[MASK] token在单步中并行解码时，采样假设条件独立——product of marginals: q(X|E) = Π_j p_j(X_{i_j}|E)——但真实数据分布p(X|E)包含token间条件依赖。当每步解码大量token时，独立性假设的偏离会导致质量下降。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

τ-leaping在LLaDA和Dream的官方实现中通过schedule控制每步解码的token数量。LLaDA默认最优策略是每步1 token（非并行），因为τ-leaping越激进（每步越多token）质量越差。Fast-dLLM通过confidence过滤改进τ-leaping：不是随机解码掩码token，而是先计算置信度，仅在高置信度时并行解码多个token，从而在加速的同时保持质量。τ-leaping步数T的选择：T越大→每步修改越少token→质量好但慢；T越小→每步修改越多token→快但质量差。LLaDA默认T=128，Fast-dLLM通过减少实际NFE（number of function evaluations）加速推理。

涉及论文标题：
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

## Conditional Independence Problem in MDM Parallel Decoding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

条件独立性问题（Conditional Independence Problem）是掩码扩散模型（MDM）中并行解码面临的根本性质量问题。在MDM的τ-leaping推理中，多个[MASK]位置同时解码时，每个位置的token从其边际分布独立采样：p_j(X_{i_j}|E)。但真实的联合概率包含token间依赖：p(X_{i_1},...,X_{i_n}|E) = p(X_{i_1}|E)·p(X_{i_2}|X_{i_1},E)·...·p(X_{i_n}|X_{i_1},...,X_{i_{n-1}},E)。乘积边际分布忽略了条件依赖项，可能产生统计上不合理（但在各边际高概率）的token组合。Fast-dLLM论文给出了一个经典例子："The list of poker hands that consist of two English words are: _ _ " → 正确答案如"high card"或"two pair"，但独立采样可能产生"high house"——两个词分别高概率但组合不合理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

问题数学形式化：

```
# 假设解码位置i1和i2
# 边际分布独立采样:
x_i1 ~ p(X_i1 | x_context)        # p(X_i1="high")=0.4
x_i2 ~ p(X_i2 | x_context)        # p(X_i2="house")=0.3
# 乘积概率: q("high","house") = 0.4 × 0.3 = 0.12

# 真实联合分布:
# p(X_i1="full", X_i2="house") > 0.5   (full house常见)
# p(X_i1="high", X_i2="house") ≈ 0     (不存在的组合)
```

Fast-dLLM Theorem 1量化了条件独立假设与真实联合分布的偏差：

- 当每个token边际置信度 > 1-ε 且 (n+1)ε ≤ 1时，argmax的乘积分布 = argmax的真实分布（等价）
- 一般情况：L_p距离上界为 ((n-1)^p + 2n)^(1/p)·ε，TV距离 < (3n-1)ε/2
- 前向KL散度：D_KL(p||q) < (n-1)[H_b(ε) + ε·ln(|V|-1)]

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

缓解策略：(1) Fast-dLLM: 仅在高置信度token上并行解码（阈值τ≈0.7-0.9），低置信度token保持[MASK]留待后续步骤；(2) 降低每步并行token数（trade-off speed）；(3) 使用辅助模型显式建模token间依赖（如Discrete Copula Diffusion [Liu et al. 2024]）；(4) Block Diffusion: 通过块内自回归+块间扩散的半自回归方式，在块内保留token依赖。Fast-dLLM的factor策略使用理论绑定量(n+1)(1-c^(n)) < f动态选择安全并行token数，在速度和质量间取得最优平衡。

涉及论文标题：
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding

## Complementary Masking (互补掩码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Complementary Masking（互补掩码）是Fast-dLLM v2提出的block diffusion训练策略。对于每个训练样本，采样一个随机binary mask m ∈ {0,1}^D（D为block size），其中m_i=1表示位置i被替换为[MASK] token。同时生成互补mask m̄ = 1 - m，两个view（m和m̄）放入同一个batch中训练。这确保每个token既在masked上下文（被mask时从其他可见token预测自己）又在unmasked上下文（可见时帮助预测其他masked token）中被训练。由于互补性，m和m̄的masked token集合完全不重叠，两个view的loss覆盖了序列中所有L个位置——使得无需在loss中除以mask比例（无需归一化系数1/t），总监督信号量恒定为L。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Block size D=32, training sample x_0 of length L
# 每个block独立采样mask

m = random_binary_mask(shape=[D])      # 每位置以p=0.5概率为1（mask）
m_complement = 1 - m                   # 互补mask

# View 1: 对x_0应用m → masked位置替换为[MASK]
x_t^1 = x_0.copy()
x_t^1[m == 1] = [MASK]

# View 2: 对x_0应用m_complement → 互补的masked位置替换为[MASK]
x_t^2 = x_0.copy()
x_t^2[m_complement == 1] = [MASK]

# 两个view放入同一batch，model同时处理
# Noised x_t和clean x_0沿sequence维度拼接（总长2L）
# 使用block-wise attention mask [[M_BD, M_OBC], [0, M_BC]]

# Loss（无需1/t归一化，因两个view覆盖所有L个位置）:
L = -Σ 1[x_t^1_i=[MASK]]·log p(x_0^1_i|x_t^1)  # view 1的masked位置
  + -Σ 1[x_t^2_i=[MASK]]·log p(x_0^2_i|x_t^2)  # view 2的masked位置
# 每个样本总贡献L个token的loss（完整序列监督）
```

消融实验（Table 2）证明：+pad+CM（complementary masking）比naive token shift提升+3.7 avg accuracy，是训练配方中最关键的组件。互补掩码还与token shift协同：masked位置使用i-1的hidden state预测token i，保留AR模型的representation quality。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 训练时将两个view构造为同一batch中的两个sample；(2) 使用flex-attention实现自定义block-wise attention mask同时处理noised和clean序列；(3) 适用于从预训练AR模型微调为block diffusion模型的场景。与标准masked language modeling（MLM）的区别：MLM通常mask 15% token，每个样本仅mask一次；CM mask ~50% token两次（互补），监督信号更密集。

涉及论文标题：
- Fast-dLLM v2: Efficient Block-Diffusion LLM

## Token Shift / Shifted-Label Strategy

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Shift（或称Shifted-Label Strategy）是Fast-dLLM v2用于在block diffusion训练中保留预训练AR模型representation quality的技术。在标准masked diffusion中，每个masked位置i使用自身的hidden state h_i来预测token x^i。而Token Shift改用前一个位置i-1的hidden state h_{i-1}来预测token x^i：logit用于预测x^i的位置是i-1而非i。这使得预测的计算路径与AR模型的next-token prediction保持一致（AR模型中position i-1预测position i），让dLLM在支持block内bidirectional attention的同时维持AR-like的temporal representation。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 标准masked diffusion预测（无token shift）:
for each masked position i:
    h_i = transformer_output[i]          # 使用位置i的hidden state
    logit_i = lm_head(h_i)               # 预测位置i的token
    
# Token Shift预测（Fast-dLLM v2）:
for each masked position i:
    h_{i-1} = transformer_output[i-1]    # 使用位置i-1的hidden state
    logit_i = lm_head(h_{i-1})           # 预测位置i的token（shifted）
    
# 效果：position i-1的hidden state负责预测position i
# 与AR next-token prediction的形式一致: p(x_i | h_{i-1})
```

Token Shift与complementary masking协同工作：masked位置i使用i-1的hidden state（i-1在complementary view中可能是可见的），使得模型能利用完整的prefix context进行预测。消融实验（Table 2）中"naive token shift"即为仅使用token shift但无complementary masking和padding的baseline（avg=41.3）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：在计算loss时对logits做offset索引——对每个masked位置i，取logits[i-1]而非logits[i]来计算cross-entropy。这与Dream的预训练方法（Ye et al., 2025b）一致。适用范围：从AR模型（使用next-token prediction训练）微调为diffusion模型时，token shift是保持AR预训练质量的关键技术。论文未明确说明此技术的原创来源，Dream论文中已有类似设计。

涉及论文标题：
- Fast-dLLM v2: Efficient Block-Diffusion LLM

## Block-wise Attention Mask for Diffusion dLLM (M_BD / M_OBC / M_BC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-wise Attention Mask是Fast-dLLM v2中用于block diffusion训练的自定义注意力掩码，将noised序列x_t和clean序列x_0沿sequence维度拼接（总长2L），应用hybrid attention pattern M_full ∈ {0,1}^{2L×2L}，分解为四个子掩码区域：

M_full = [[M_BD, M_OBC], [0, M_BC]]

其中：
- **M_BD (Block-diagonal mask)**：x_t内部的块内双向自注意力。同一block内token互相可见，支持block内的masked token refinement。矩阵仅对角block为1。
- **M_OBC (Offset block-causal mask)**：x_t → x_0的跨序列注意力。每个noised token可attend到前面block的clean token，保持块间causal conditioning。矩阵为上三角block结构。
- **M_BC (Block-causal mask)**：x_0内部的自回归式注意力。clean token可attend到同block及之前block的token，保持AR-like progression。
- **0 (左下角)**：x_0不能attend到x_t（clean不应看到noise），保证训练的信息流向正确。

推理时简化为：已解码block（x_0的前缀）作为cached prefix只读，当前block x_t^b双向自注意力+对prefix的causal attention。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 训练时序列拼接
Input = concat([x_t, x_0])           # 总长2L
# x_t: noised sequence (部分[MASK])
# x_0: clean sequence (原始token)

# Attention mask M_full ∈ {0,1}^{2L×2L}
M_BD[i][j] = 1 iff block(i)==block(j)  # x_t内部块内双向
M_OBC[i][j] = 1 iff block(j) < block(i) # x_t看clean历史块
M_BC[i][j] = 1 iff block(j) <= block(i)  # x_0内部块因果

# 推理时简化mask:
# 已解码块作为prefix → 缓存K/V → 仅当前noised block计算
# 当前block: bidirection自注意力 + causal attend to prefix
```

使用PyTorch flex-attention实现，避免手动构造完整2L×2L的mask矩阵（内存O(L²)），而是通过自定义score_mod函数在attention计算时动态决定哪些position pair可见。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现基于PyTorch的`torch.nn.attention.flex_attention.flex_attention` API（或旧版`F.scaled_dot_product_attention`的custom mask参数）。自定义`score_mod`函数实现四种mask逻辑。key insight：这个mask设计与AR模型的causal attention高度接近（仅将同一block内从causal改为bidirectional），因此预训练AR模型只需少量微调即可适应——这是Fast-dLLM v2仅需~1B tokens微调的关键原因。

涉及论文标题：
- Fast-dLLM v2: Efficient Block-Diffusion LLM

## Grouped-Query Attention (GQA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Grouped-Query Attention (GQA, 分组查询注意力) 是介于 Multi-Head Attention (MHA) 和 Multi-Query Attention (MQA) 之间的一种 attention 变体。在 MHA 中，每个 query head 有独立的 KV head(GQA_ratio=1)；在 MQA 中，所有 query heads 共享单一 KV head（extreme sharing）。GQA 将 query heads 分组，同组内的多个 query heads 共享一个 KV head。GQA ratio (如 1, 4, 8, 16) 定义了 query heads 与 KV heads 的比例——ratio=1 等价于 MHA，ratio=num_heads 等价于 MQA。GQA 由 Ainslie et al. (2023) 提出，用于减少 KV cache 内存占用和 attention 计算量，同时保持比 MQA 更好的模型质量。在 LLM serving 中，GQA 对 attention kernel 性能有直接影响：高 GQA ratio 意味着更大的 Q 矩阵（更多 query heads 共享同一 K/V），decode 阶段每个 query head 的 K/V 共享度更高，缓解了 per-query GEMV 的 tensor core underutilization 问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

GQA 在 attention 计算中的实现（以 ratio=4, num_q_heads=32, num_kv_heads=8 为例）：

```
// MHA (ratio=1): 32 Q heads, 32 KV heads
for head in 1..32:
    Q_h = X @ W_q[h]              // (seq_len, d)
    K_h = X @ W_k[h]              // (seq_len, d)
    V_h = X @ W_v[h]              // (seq_len, d)
    O_h = softmax(Q_h @ K_h^T / sqrt(d)) @ V_h
O = concat(O_1..O_32) @ W_o

// GQA (ratio=4): 32 Q heads, 8 KV heads
// KV heads shared: Q_1..Q_4 use KV_1; Q_5..Q_8 use KV_2; ...
for group in 1..8:
    KV_h = group
    for q_head in 4*(group-1)+1 .. 4*group:
        Q_h = X @ W_q[q_head]
        O_h = softmax(Q_h @ K_{KV_h}^T / sqrt(d)) @ V_{KV_h}
O = concat(all O_h) @ W_o

// Key difference in KV cache:
// MHA: KV cache size = 2 × 32 × L × d
// GQA-4: KV cache size = 2 × 8 × L × d     (4× smaller)
// GQA-16: KV cache size = 2 × 2 × L × d    (16× smaller, Llama-2-70B style)
```

GQA 对 attention kernel 性能的影响（decode 阶段）：
```
GQA ratio=1 (MHA):
  Q: (1, d) vector  → GEMV  → tensor core 利用率低

GQA ratio=4:
  Q: (4, d) matrix  → small GEMM  → tensor core partial utilization

GQA ratio=16:
  Q: (16, d) matrix → medium GEMM  → tensor core near-full utilization
  → FlashAttention/FlashInfer 在 GQA-16 时 decode 性能大幅改善
```

在 FastTree 的 kernel benchmark 评估中（Figure 9），GQA ratio=1 时 FastTree 对 FlashAttention/FlashInfer 的 speedup 最高（tensor core underutilization 最严重），GQA ratio=16 时 speedup 缩小但仍显著（因 FastTree 的 query aggregation further increases effective batch size beyond GQA grouping + KV reuse via shared memory）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GQA 在模型训练时通过修改 attention 的 KV projection 实现：将 K 和 V 的 weight 矩阵大小从 (d_model, num_heads × d_k) 改为 (d_model, num_kv_heads × d_k)。推理时，各 attention kernel 库（FlashAttention、FlashInfer、FastTree）通过 GQA ratio 参数确定 K/V head 与 Q head 的映射关系。FastTree 的 attention kernel 在 grouping plan 中不区分 head——context-queries grouping 在 batch（request）维度聚合 queries，而 GQA 在 head 内聚合。两者正交互补：GQA 增加每个 query 的有效 Q matrix size，FastTree 增加每个 group 的有效 Q matrix size（通过跨 request 的 query aggregation）。因此在 GQA-16 + FastTree grouping 下，tensor core 利用率达到最高。

涉及论文标题：
- FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving

## Block Diffusion

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block Diffusion是一种介于全扩散（bidirectional, 全并行）和全自回归（causal, 全顺序）之间的生成范式。核心思想：将序列划分为固定大小的块（blocks），块内使用bidirectional attention + masked token prediction（扩散范式），块间使用causal conditioning（自回归范式）。Fast-dLLM v2首次将block diffusion扩展到现代LLM规模（7B），提出完整的训练+推理recipe：训练时使用block-wise attention mask（M_BD+M_OBC+M_BC）+ complementary masking + token shift；推理时使用block级KV cache（跨block复用已解码上下文）+ sub-block DualCache（块内高效并行refinement）+ confidence-aware parallel decoding。仅需~1B tokens微调即可将预训练AR模型（Qwen2.5-Instruct）转化为block diffusion模型，相比Dream的580B tokens减少500×。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fast-dLLM v2的完整Block Diffusion pipeline：

```
# === 训练阶段 ===
Input: 预训练AR模型 θ_AR, 训练数据D, block size D=32
1: 将序列padding到D的整数倍（padding token不参与loss）
2: Packing: 多个样本拼接至context length L
3: 对每个block b采样random mask m_b和互补mask m̄_b
4: 两个view放入同一batch
5: Attention mask M_full = [[M_BD, M_OBC], [0, M_BC]]
6: Token shift: 位置i-1的hidden state预测位置i的token
7: Loss: masked-token-only cross-entropy

# === 推理阶段 ===
Input: prompt p, target_len L, block_size B=32, sub_block_size S=8
1: x ← [p; [MASK]×L]
2: for k in 1..⌈L/B⌉:                       # 逐block
3:     复用block级KV cache（已解码block的K/V）
4:     for each sub-block in current block:
5:         bidirectional attention within sub-block
6:         confidence > τ token并行解码（threshold=0.9）
7:         DualCache复用sub-block prefix/suffix K/V
8:     end
9:     刷新block级KV cache
10: end
```

Block Diffusion的关键优势：(1) 块内bidirectional attention提供更丰富的context modeling；(2) 块间causal conditioning保证全局语义连贯性；(3) block级KV cache实现与AR模型类似的cache复用；(4) 训练与AR模型高度兼容（仅需1B tokens微调）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Fast-dLLM v1（训练无关的block diffusion加速）已开源：https://github.com/NVlabs/Fast-dLLM。Fast-dLLM v2代码和模型待发布。训练使用64×A100 GPU + DeepSpeed Zero-3，1.5B模型约8小时，7B模型约12小时。推理时block size=32, sub-block size=8, threshold=0.9实现2.5×加速（vs AR baseline）。Block diffusion的block size是关键的accuracy-efficiency trade-off参数：训练和推理block size应保持一致（mismatch导致显著性能退化，Table 4）；sub-block size提供推理粒度的灵活调节（Table 3，size=8最优）。

涉及论文标题：
- Fast-dLLM Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding
- Fast-dLLM v2: Efficient Block-Diffusion LLM

## IO-Awareness / IO-Aware Algorithm

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

IO-Awareness（IO感知）是一种算法设计原则，要求算法明确考虑不同层级内存之间的读写（IO）开销，而不仅仅是算术运算（FLOPs）数量。该概念源于Aggarwal & Vitter (1988)的IO复杂度理论，FlashAttention论文将其引入深度学习attention计算领域。核心观察：现代GPU上计算速度已远超内存速度（A100 SRAM带宽~19TB/s vs HBM带宽~1.5-2.0TB/s，~10×差距），大多数Transformer操作是memory-bound而非compute-bound。IO-aware算法的目标是通过reorganizing computation来减少慢速内存（HBM）的访问次数，即使这意味着增加FLOPs，因为HBM带宽才是真正的瓶颈。FlashAttention通过tiling将$N \times N$ attention矩阵的HBM读写从$\Theta(N^2)$降至$\Theta(N^2d^2M^{-1})$（M为SRAM大小），实测HBM读写减少8×，整体加速3×，同时FLOPs反而从66.6增至75.2 GFLOPs。这一反直觉结果（FLOPs增但速度更快）正是IO-awareness的核心理念：memory access > compute。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

IO-awareness在attention计算中的应用（以FlashAttention为例）：
```python
# IO-unaware (standard attention): 3次独立操作，每次经HBM
S = Q @ K.T          # Step 1: GEMM, write S[N,N] to HBM
P = softmax(S)       # Step 2: softmax, read S from HBM, write P[N,N] to HBM
O = P @ V            # Step 3: GEMM, read P from HBM, write O to HBM
# HBM traffic = 2*Nd (Q,K input) + 2*N² (S write+read) + 2*N² (P write+read) + Nd (V input) + Nd (O output)
# ≈ 3Nd + 4N² elements

# IO-aware (FlashAttention): 单kernel，block-wise计算，无N×N矩阵在HBM
# Block sizes: B_c ≈ M/(4d), B_r = min(B_c, d)
for j in range(T_c):                    # outer loop: KV blocks in SRAM
    load K_j[ B_c x d ], V_j[ B_c x d ] from HBM to SRAM
    for i in range(T_r):                # inner loop: Q blocks
        load Q_i[ B_r x d ] from HBM to SRAM
        S_ij = Q_i @ K_j.T              # in SRAM: B_r x B_c
        m_new = max(m_i, rowmax(S_ij))  # online softmax stats
        l_new = exp(m_i-m_new)*l_i + sum(exp(S_ij - m_new))
        O_i = (l_i*exp(m_i-m_new)*O_i + exp(S_ij-m_new) @ V_j) / l_new
        save m_i, l_i, O_i to HBM       # only O(N) per write, NOT O(N²)
# HBM traffic = O(N²d²/M) << O(N²)
```
关键：中间S_ij和P_ij仅驻留SRAM，逻辑流程中的每个步骤都设计为最小化HBM交互。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

IO-awareness的实现方式：(1) **Tiling/Blocking**：将大数据分解为fit in fast memory的小块，分批处理；(2) **Kernel fusion**：将多个操作合并为单个kernel，消除中间结果的slow memory round-trip；(3) **Recomputation**：用compute换取memory——不存储中间结果而是重新计算；(4) **Memory hierarchy-aware scheduling**：根据各层内存的带宽/容量特性安排数据驻留位置。在FlashAttention中，这些技术组合使用：tiling确保每block fit in SRAM，kernel fusion消除kernel间HBM传输，recomputation消除backward的O(N²)存储需求。实际使用：`flash_attn_func(q, k, v)`作为PyTorch中标准attention的直接替代，自动在kernel内部应用所有IO-aware优化。IO-awareness理念已扩展到FFN（SRAMFFN/FlashMHF）、通信重叠（FlashOverlap）等场景。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness

## Online Softmax / Tiled Softmax Computation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Online Softmax（在线softmax，又称tiled softmax或streaming softmax）是一种允许在不一次性访问全部输入数据的情况下精确计算softmax的增量算法。源于Milakov & Gimelshein (2018)的"Online normalizer calculation for softmax"。标准safe softmax需要两次遍历：第一次找全局最大值m，第二次计算$\exp(x_i-m)$并求和，再归一化。Online softmax通过维护running state $(m, \ell)$（running max和running sum）在一次遍历中完成计算，每接收一个data block时更新状态：$m' = \max(m_A, m_B)$，$\ell' = \ell_A \cdot \exp(m_A-m') + \sum \exp(x_B - m')$。关键在于softmax对输入平移不变（$x - c$不改变结果），online版本不断将"参考坐标系"（最大值）重设并施加代数修正。最终结果与标准两次遍历完全等价，非近似。FlashAttention利用online softmax实现block-wise attention计算：每个(i,j) block pair的softmax计算仅需维护当前row的$(m_i, \ell_i)$状态，使得N×N attention矩阵永远不需要整体存在。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Online softmax在FlashAttention的tiled attention中的核心流程（per query row）：
```
# 状态初始化（每query row i维护）
m_i = -inf           # running max of attention scores
l_i = 0              # running sum of exp(scores - running max)
O_i = 0              # running weighted sum of V (output accumulator)

# 对每个KV block j（按序处理）：
S_ij_block = Q_i @ K_j.T           # [B_r, B_c] attention scores
m_ij = rowmax(S_ij_block)          # block local max per query row
m_new = max(m_i, m_ij)             # 更新global max
# 重缩放旧累加值（补偿max变化）：
l_i = l_i * exp(m_i - m_new)       # rescale old exp-sum
O_i = O_i * exp(m_i - m_new)       # rescale old output
# 添加新block的贡献：
P_ij = exp(S_ij_block - m_new)     # [B_r, B_c] unnormalized softmax
l_i += rowsum(P_ij)                # update exp-sum
O_i += P_ij @ V_j                  # accumulate weighted V
m_i = m_new                        # update running max
# 处理完所有KV blocks后：
O_i = O_i / l_i                    # 最终归一化得到exact softmax output
```
该算法的正确性保证了最终O_i与标准两次遍历softmax完全一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Online softmax已在FlashAttention系列（v1-v4）、FlashInfer、xFormers等GPU attention库中广泛实现。实现细节：(1) 使用base-2 scaling（exp2替代exp）以利用硬件MUFU.EX2指令和与FFMA（fused multiply-add）的编译器融合；(2) 处理全mask行：当rowmax为-inf时替换为0避免NaN；(3) 反向传播时，存储前向的LogSumExp $L_i = m_i + \ln(\ell_i)$（每query row一个scalar），反向在SRAM中重计算$P_{ij} = \exp(S_{ij} - L_i)$来求梯度。Flash-D (2025)进一步提出用sigmoid替代softmax division，完全消除max subtraction步骤。Online softmax的流式处理模式不仅限于softmax——LayerNorm/RMSNorm、running statistics (Adam/RMSProp)等也可用类似模式实现tiled/streaming计算。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
- FlashAttention-2 Faster Attention with Better Parallelism and Work Partitioning
- UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels
- FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
- FlashInfer Efficient and Customizable Attention Engine for LLM Inference Serving
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

**MetaAttention 的 RowNorm Online 泛化**：MetaAttention 将 online softmax 思想泛化为通用的 RowNorm Online 接口，支持任意 row-wise normalization（softmax、sigmoid、ReLU norm、RetNet reduceAbsSum norm 等），而非仅局限于 softmax。该接口定义为三段式：(1) online_prologue——初始化归一化状态变量（如 row_max=-inf, row_sum=0 或 row_sum_wo_clamp=0）；(2) online_forward——对每个 block 更新归一化状态，计算 rescale factor 传给 aggregation 阶段用于修正已累积输出；(3) online_epilogue——最终归一化。在 MetaAttention 的 scheduling 中，RowNorm Online 产生的中间状态变量（row_max, row_sum 等）作为 IntermediateTensor 纳入调度（通常分配在 register），elementwise/scaling 操作被 SIMT fused，reduce 操作使用 intra-warp reduction。这使得 MetaAttention 能在一个框架内同时支持 parallel pattern（如 FlashAttention-like online softmax）和 recurrent pattern（如 chunk-parallel state update）的 online normalization。

**FlashAttention-2 的算法改进**：FlashAttention-2对online softmax做了两项关键tweak来减少non-matmul FLOPs：
1. **Un-scaled output maintenance**：FlashAttention v1在每次内迭代都做`O_i = diag(ℓ)^{-1} @ O_tilde` rescale。FlashAttention-2改为维护un-scaled output $\tilde{\mathbf{O}}^{(j)} = \operatorname{diag}(e^{m^{(j-1)}-m^{(j)}})\tilde{\mathbf{O}}^{(j-1)} + e^{\mathbf{S}^{(j)}-m^{(j)}}\mathbf{V}^{(j)}$，仅在所有KV blocks处理后一次性做`diag(ℓ)^{-1}` rescale得到最终O。消除每次迭代对已累积output的elementwise rescale（non-matmul operation）。伪代码对比：
```
# FlashAttention v1 (每次迭代rescale):
O_i = diag(ℓ)^{-1} @ (diag(exp(m_old-m_new)) @ (diag(ℓ_old) @ O_old) + P_tilde @ V_j)

# FlashAttention-2 (维护un-scaled, 最终rescale):
O_tilde = diag(exp(m_old-m_new)) @ O_tilde + P_tilde @ V_j
# ... 循环结束后:
O_i = diag(ℓ)^{-1} @ O_tilde
```
2. **仅存LogSumExp L**：FlashAttention-2反向仅需`L = m + log(ℓ)`（每行一个scalar），替代FlashAttention v1的(m, ℓ) pair。反向从L重建softmax denominator：$P_{ij} = \exp(S_{ij} - L_i)$。减少register压力和对non-matmul计算的需求。

## Surrogate Maximum (X-Row Tile Maximum) for Tensorized Softmax

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Surrogate Maximum（替代最大值，记作m̂[i]）是FlashAttention-T (PPoPP'26) 在Tensorized Online Softmax算法中提出的关键概念。在fused attention的online softmax中，attention output rescaling操作 `O = exp(m_old - m)·O` 要求scaling factor `exp(m_old[i]-m[i])` 对每行独立计算。然而，repurposed tensor MMA scaling instruction要求scaling factor α在所有行上uniform（X行共享同一个α值，X=16 for HMMA.1688, X=64 for HGMMA.64x8x8）。为满足这一约束，Surrogate Maximum定义为attention logit矩阵S的第i行所在X-row tile的最大值：
$$hat{m}[i] = \max({S[i',j']: j' \in [0,s), i' \in [X \cdot \lfloor i/X \rfloor, X \cdot \lfloor i/X \rfloor + X)})$$

m̂[i]在X行内保持uniform，使scaling factor `exp(m̂_old - m̂)` 满足tensor MMA uniform scaling constraint。

Numerical safety guarantees:
1. **No overflow**（严格保证）：m̂[i] ≥ m[i]（tile max ≥ row max），故 exp(S[i,j]-m̂[i]) ≤ exp(S[i,j]-m[i]) ≤ 1，永不超F_max
2. **Negligible all-underflow probability**（高概率）：更大m̂增大single exponent underflow概率，但整行所有exponent同时underflow的联合概率在典型分布（Gaussian等）下asymptotically small
3. **Fallback机制**：极端分布触发all-underflow时，选择性fallback到vectorized rescaling（跳过surrogate，保持其他primitives tensorized）

与FlashDecoding++的static maximum不同，X-row tile surrogate动态适应局部行分布。

从算法pipeline角度拆解术语：

Tensorized Online Softmax (Algorithm 1)的核心流程：
```
// Input: S∈R^{n×s}, O∈R^{n×d}, m_old∈R^n, l∈R^n, surrogate tile size X
// Step 1: Compute X-row tile maxima (warp REDUX, 2 instructions)
m̂ ← tilemax(S, X)                    // m̂ ∈ R^{⌈n/X⌉}
// Step 2: Get old surrogate maximums (broadcast)
m̂_old ← m_old[X·i] for i ∈ [0, ⌈n/X⌉)
// Step 3: Tensorized O rescaling
//   scaling factor exp(m̂_old[k] - m̂[k]) uniform ∀ rows in tile k
//   → satisfies tensor MMA uniform scaling constraint ✓
for k in 0..⌈n/X⌉-1:
    O[kX:(k+1)X,:] ← exp(m̂_old[k] - m̂[k]) · O[kX:(k+1)X,:]
// Step 4: Assign surrogate to per-row m
m[i] ← m̂[⌊i/X⌋] for i ∈ [0, n)
// Step 5: Tensorized S rescaling (constant log₂(e) → always uniform)
Z ← log₂(e) · S - (log₂(e) · m)
// Step 6: Vector exp₂
P̃ ← exp₂(Z)                          // MUFU.EX2, stay vectorized
// Step 7: Tensorized row-sum reduction
l ← exp(m_old - m)·l + rowsum(P̃)
// Step 8: return P̃, O, m, l
```

对比Baseline per-row maximum:
```
// Baseline: m[i] = max(S[i,:]) — per-row, non-uniform
// O[i,:] ← exp(m_old[i] - m[i]) · O[i,:]
//   scaling factor varies per row → CANNOT use tensor MMA scaling ✗

// FlashAttention-T: m̂[k] = max(16/64 consecutive rows) — tile-uniform
// O[kX:(k+1)X,:] ← exp(m̂_old[k] - m̂[k]) · O[kX:(k+1)X,:]
//   scaling factor uniform in tile → CAN use tensor MMA scaling ✓
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Surrogate maximum的实现：(1) warp all-reduce REDUX in 2 instructions（vs baseline逐行SHFL-based max需多次warp shuffle）；(2) FA2+Max16 ablation（仅加surrogate maximum，无tensorization）即带来1-3% speedup，因REDUX > SHFL in throughput；(3) Hopper TLP实现中surrogate未被使用（仅tensorize P̃ row-summation无需scaling factor），数值稳定性与baseline一致。surrogate maximum概念可推广到任何需在特定粒度approximate per-element operations以对齐hardware-aligned computation的场景（如block-level normalization with uniform statistics）。

涉及论文标题：
- FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism

## Block-Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-Sparse Attention是FlashAttention的扩展变体，通过在预定义的block级稀疏mask约束下跳过零值block的attention计算来加速推理。给定block sparsity mask $\mathbf{M} \in \{0,1\}^{N/B_r \times N/B_c}$（其中$B_r, B_c$为block sizes），block-sparse attention仅计算$M_{ij}=1$的(i,j) block对：$\mathbf{S}_{ij} = \mathbf{Q}_i \mathbf{K}_j^T$仅在$M_{ij}=1$时计算，softmax和$\tilde{\mathbf{P}}_{ij}\mathbf{V}_j$同理。其IO复杂度为$\Theta(Nd + N^2d^2M^{-1}s)$（s为non-zero block比例），比dense FlashAttention减少sparsity倍。与一般稀疏attention不同，block-sparse要求稀疏模式在block边界对齐——这一约束恰好与FlashAttention的tiling自然吻合，因为tiling本身就在block粒度上操作。论文使用固定butterfly sparsity pattern（Dao et al., 2022），这种模式被证明可以逼近任意稀疏矩阵。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Block-sparse FlashAttention (Algorithm 5)的核心计算流程：
```
# 与FlashAttention Algorithm 1的区别仅在于内循环条件：
for j = 1 to T_c:
    load K_j, V_j from HBM to SRAM
    for i = 1 to T_r:
        if M[i][j] == 0:           # ← 唯一区别：跳过零值block
            continue                  # 省softmax + PV计算 + V加载
        # 其余完全同FlashAttention:
        load Q_i from HBM to SRAM
        S_ij = Q_i @ K_j.T         # BMM1（compute-bound，可能仍需计算...实际上FlashAttention的block-sparse同样跳过S_ij计算）
        # online softmax...
        # accumulate O_i...
```
注意：在FlashAttention的block-sparse实现中，即使$M_{ij}=1$的block在BMM1（Q_i @ K_j.T）步骤也不计算——算法直接跳过整个内循环迭代，从而实现与sparsity s成比例的runtime减少。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Block-sparse attention在FlashAttention代码库（https://github.com/HazyResearch/flash-attention）中以`BlockSparseAttention`接口提供。使用：指定block sparsity mask作为`(N/B_r, N/B_c)`的二进制矩阵。butterfly pattern是常用选择：对序列中相距较远的token pair赋予1（长程依赖），对相距近的token pair也赋予1。在LRA benchmark上，block-sparse FlashAttention达到2.8× speedup（vs dense，seq length 1K-4K），同时accuracy与dense attention持平（LRA平均59.6 vs 59.8）。在Path-256（seq length 64K）上，block-sparse使Transformer首次达到63.1%准确率（dense FlashAttention因memory限制无法扩展到64K）。block-sparse的sparsity pattern选择对accuracy影响较大——论文使用预定义的butterfly pattern而非learned sparsity。

涉及论文标题：
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness
- Flex Attention: A Programming Model for Generating Optimized Attention Kernels

## Block Quantization（块量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Quantization（块量化）是针对FP8等低精度格式的一种量化策略：将tensor划分为多个block（如$B_r \times d$或$B_c \times d$大小的子矩阵），每个block独立计算并保存一个scaling factor（通常为block内元素绝对值的最大值），量化时block内所有元素除以该block的scaling factor后映射到FP8表示范围，反量化时乘以对应的scaling factor恢复。与per-tensor quantization（整个tensor共享一个scalar scale）相比，block quantization提供更细粒度的动态范围适配——每个block独立伸缩，大幅减小outlier elements对量化精度的破坏。FlashAttention-3中，Q、K、V分别在进入attention kernel前进行block quantization，scaling factor可以fuse到前序操作（如rotary embedding，本身是memory-bound，无额外开销）。由于FlashAttention的tiled算法自然按block操作，block-wise scaling可以在$S_{ij}=Q_i K_j^T$计算时以零成本整合：$\tilde{S}_{ij} = \text{scale}_Q(i) \cdot \text{scale}_K(j) \cdot (Q_i K_j^T)$，scale因子仅需逐block相乘一次。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FP8 FlashAttention-3前向的block quantization pipeline：
```
输入: Q, K, V ∈ R^{N×d}, block sizes B_r, B_c
1. 将Q按行划分为T_r个blocks Q_i ∈ R^{B_r×d}
   将K按行划分为T_c个blocks K_j ∈ R^{B_c×d}
   将V按行划分为T_c个blocks V_j ∈ R^{B_c×d}
2. 对每个Q_i计算scale: scale_Q[i] = max(|Q_i|) / max_FP8
   对每个K_j计算scale: scale_K[j] = max(|K_j|) / max_FP8
   对每个V_j计算scale: scale_V[j] = max(|V_j|) / max_FP8
3. 量化（可fuse到rotary embedding）:
   Q_i_FP8 = quantize_FP8(Q_i / scale_Q[i])
   K_j_FP8 = quantize_FP8(K_j / scale_K[j])
   V_j_FP8 = quantize_FP8(V_j / scale_V[j])
4. Tiled attention主循环:
   for j in 0..T_c-1:
       S_ij = FP8_GEMM(Q_i_FP8, K_j_FP8^T)       // FP8 tensor core
       S_ij *= scale_Q[i] * scale_K[j]             // rescale before softmax
       P_ij = softmax(S_ij)
       O_i = FP8_GEMM(P_ij, V_j_FP8) * scale_V[j] // FP8 tensor core
```
关键：scale因子仅在softmax前和PV累加后各乘一次，不引入每元素scale开销——因为scale对于整个block是常数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Block quantization在FlashAttention-3中通过CUTLASS的FP8 WGMMA primitives实现。量化本身在kernel外部或fuse到rotary embedding kernel中完成（rotary embedding是memory-bound，fuse不增加延迟）。Kernel内部通过WGMMA的FP8模式直接使用量化后的Q_i_FP8, K_j_FP8, V_j_FP8。Scale因子存储为per-block FP32标量数组，在kernel内通过寄存器传递。Block quantization同样适用于KV cache quantization（如KIVi, KVQuant）和weight quantization（如LLM.int8()），是一种通用的细粒度量化策略。

涉及论文标题：
- FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision

## Incoherent Processing（非相干处理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Incoherent Processing（非相干处理）是一种量化预处理技术，通过在量化前对数据施加随机正交变换来"均匀化"数据分布，减少outlier对量化精度的破坏。给定矩阵$\mathbf{Q}$，将其乘以随机正交矩阵$\mathbf{M}$（满足$\mathbf{M}\mathbf{M}^\top = \mathbf{I}$）得到$\mathbf{Q}' = \mathbf{Q}\mathbf{M}$。由于$\mathbf{M}$是正交矩阵，$(\mathbf{Q}\mathbf{M})(\mathbf{K}\mathbf{M})^\top = \mathbf{Q}\mathbf{K}^\top$，即attention的数学结果不变。但关键在于：$\mathbf{Q}\mathbf{M}$的每个元素是$\mathbf{Q}$一行中各元素的随机加权和（权重来自$\mathbf{M}$的列），这使原本集中在少数维度的outlier被"分散"到所有维度中——每个元素的大小趋于均匀（由中心极限定理），大幅降低量化时的动态范围差异。FlashAttention-3采用Chee et al. (QuIP) 和Tseng et al. (QuIP#) 的方法，取$\mathbf{M}$为Hadamard矩阵$\mathbf{H}$与随机对角符号矩阵$\mathbf{D}$的乘积：$\mathbf{M} = \mathbf{H}\mathbf{D}$，计算复杂度从$O(d^2)$降至$O(d \log d)$（由于Hadamard变换可用Fast Walsh-Hadamard Transform加速）。

从算法pipeline角度拆解术语：
Incoherent processing在FP8 FlashAttention-3中的位置（fuse到rotary embedding，零额外开销）：
```
// 原始Q, K (FP16/BF16)
1. Apply Rotary Position Embedding: Q_rope, K_rope = RoPE(Q), RoPE(K)
   // rotary embedding is memory-bound, fuse incoherent processing here
2. Multiply by random orthogonal matrix M = H × D:
   Q' = Q_rope × M    // = Q_rope × H × D, O(d log d) via Fast Walsh-Hadamard
   K' = K_rope × M    // = K_rope × H × D
   // Q'K'^T = (Q_rope M)(K_rope M)^T = Q_rope (M M^T) K_rope^T = Q_rope K_rope^T
3. Block quantize Q', K' to FP8 e4m3
4. Proceed with FP8 FlashAttention-3 on Q'_FP8, K'_FP8
```
数学验证：$\mathbf{M}$由随机对角矩阵$\mathbf{D}$（对角元为±1随机取值）和Hadamard矩阵$\mathbf{H}$组成。$\mathbf{H}$满足$\mathbf{H}\mathbf{H}^\top = d\mathbf{I}$（Hadamard矩阵是正交的，但scale by $\sqrt{d}$），归一化后为正交。$\mathbf{D}$是对角符号矩阵，满足$\mathbf{D}\mathbf{D}^\top = \mathbf{I}$。乘积$\mathbf{M}=\mathbf{H}\mathbf{D}$仍是正交矩阵。FlashAttention-3的数值实验验证：FP8 with block quant + incoherent processing 的RMSE比per-tensor FP8 baseline低2.6×。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Incoherent processing源自QuIP (Chee et al., NeurIPS 2023) 和QuIP# (Tseng et al., 2024) 的LLM weight quantization方法。FlashAttention-3将其adapt到attention activation quantization场景。实现采用Fast Walsh-Hadamard Transform (FWHT)：`y = FWHT(x)`迭代式地将输入向量通过log2(d)层butterfly操作，每层O(d)，总O(d log d)。随机符号矩阵$\mathbf{D}$的生成使用固定seed的PRNG（每个head独立），存储开销仅O(d)而非O(d²)。在FlashAttention-3的implementation中，incoherent processing与rotary embedding融合在同一个preprocessing kernel中，不引入额外kernel launch开销。

涉及论文标题：
- FlashAttention-3 Fast and Accurate Attention with Asynchrony and Low-precision

## Mixture of Experts (MoE) Architecture

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixture of Experts (MoE) 是一种神经网络架构模式，将传统 Transformer 中的单一 FFN（Feed-Forward Network）替换为多个（E 个）规模相同的 FFN（称为 "experts"），并通过一个可训练的 gating network 动态选择每个输入 token 对应的 top-k 个 experts（通常 k=1 或 2）进行稀疏激活。MoE 的核心价值是"条件计算"（conditional computation）——增加模型总参数量（更多 experts）而不等比增加计算量（每个 token 仅激活少量 experts），从而实现 sublinear scaling of compute cost with model size。MoE 架构最早由 Shazeer et al. (2017, "Outrageously Large Neural Networks") 引入深度学习，后在 GShard (Lepikhin et al., 2021)、Switch Transformer (Fedus et al., 2022)、DeepSeek-V3 (2024)、Mixtral 8x7B (2024) 等模型中广泛采用。

MoE layer 执行流程：
1. **Gate**: 对输入 token x，gate function G(x) = softmax(x·W_g) 计算所有 E 个 experts 的 affinity scores
2. **Top-K selection**: 从 affinity scores 中选 top-k (k=2)，得 selected_experts 和对应 weights g
3. **Dispatch**: 将 token 送到选中的 experts 所在的设备（本地或远端 GPU）
4. **Expert FFN**: 各 expert 对被路由到的 token 执行 FFN(x) = W_2·φ(xW_1 + b_1) + b_2，其中 φ 为 GELU/ReLU/SiLU 等激活函数
5. **Combine**: 若 k>1，将多个 expert 的输出按 gating weights 加权合并：h_i = Σ_j (g_{i,e_j}/C_i)·h_i^j，其中 C_i = Σ_j g_{i,e_j} 为归一化因子

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# ===== MoE Layer Forward Pass (Algorithm) =====
# Input:  x ∈ R^{S×H}  (S tokens, H hidden dim)
# Output: y ∈ R^{S×H}
# Experts: E FFNs, each W1[e]∈R^{H×D}, W2[e]∈R^{D×H}
# Gate: W_g ∈ R^{H×E}

def moe_forward(x, experts, gate, k=2, capacity_factor=1.0):
    S, H = x.shape
    E = len(experts)
    C = int(S * k * capacity_factor / E)  # expert capacity
    
    # Step 1: Gate — 计算所有 token 对所有 expert 的 affinity
    gate_logits = x @ W_g  # [S, E]  (或 softmax(x·W_g))
    gate_scores = softmax(gate_logits, dim=-1)
    
    # Step 2: Top-K selection — 每个 token 选 top-2 experts
    topk_weights, topk_indices = topk(gate_scores, k, dim=-1)
    # topk_weights: [S, k], topk_indices: [S, k]  (哪些 expert)
    
    # Step 3: Dispatch — 按 expert 分组 token
    expert_inputs = {e: [] for e in range(E)}
    expert_weights = {e: [] for e in range(E)}
    for i in range(S):
        for j in range(k):
            e = topk_indices[i, j]
            if len(expert_inputs[e]) < C:  # capacity check
                expert_inputs[e].append(x[i])
                expert_weights[e].append(topk_weights[i, j])
    
    # Step 4: Expert FFN — 各 expert 独立处理
    expert_outputs = {}
    for e in range(E):
        if len(expert_inputs[e]) > 0:
            batch_e = stack(expert_inputs[e])  # [n_e, H]
            # FFN: two linear layers with activation
            h1 = batch_e @ W1[e]  # [n_e, D]
            h1 = gelu(h1)         # activation
            h2 = h1 @ W2[e]       # [n_e, H]
            expert_outputs[e] = h2
        else:
            expert_outputs[e] = None
    
    # Step 5: Combine — weighted sum of expert outputs
    y = zeros(S, H)
    combine_norm = zeros(S)
    for i in range(S):
        for j in range(k):
            e = topk_indices[i, j]
            w = topk_weights[i, j]
            if expert_outputs[e] is not None:
                # need to find which row in expert_outputs[e] is token i
                y[i] += w * expert_outputs[e][token_idx_in_expert[i, j]]
        y[i] /= combine_norm[i] if combine_norm[i] > 0 else 1.0
    
    return y
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MoE 实现涉及几个关键设计选择：(1) **Expert capacity C**——通过 capacity factor 控制，C = (tokens × top_k × cf) / num_experts，cf < 1 时可能丢弃 token（需 auxiliary load balancing loss 鼓励 uniform routing）；(2) **Load balancing**——MoE 需要 auxiliary loss 鼓励 token 均匀分布到各 experts，常见公式: L_aux = E·Σ_i f_i·P_i，其中 f_i = expert i 处理的 token 比例，P_i = gate 分配给 expert i 的平均概率；(3) **分布式部署**——experts 分布在多 GPU 上时需 cross-GPU AlltoAll 通信（dispatch + combine），通信开销可占总运行时间 68%；(4) **Auxiliary loss**——除 load balancing loss 外，还有 z-loss（防止 logits 过大导致数值不稳定）。

代表模型: GShard (E=2048, k=2), Switch Transformer (E=2048, k=1), DeepSeek-V3 (E=256, k=8, 685B total params), Mixtral 8x7B (E=8, k=2), Qwen3-235B-A22B (E=128)。

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel

## Top-K Gating / Sparse Token Routing in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Top-K Gating (Token Routing) 是 MoE 架构中决定每个 token 由哪些 experts 处理的核心机制。给定输入 token x ∈ R^H，gate function 计算该 token 对所有 E 个 experts 的 affinity scores（通过线性投影 x·W_g + optional noise），然后选择 top-k 个最高分的 experts 作为该 token 的"激活 experts"。未被选中的 experts 不参与该 token 的计算——这就是"稀疏激活"（sparse activation）的来源。

形式化 (FlashMoE 使用 top-2 routing, capacity factor=1.0):
- Gate logits: $l = xW_g \in \mathbb{R}^E$
- Affinity scores: $g_i = \text{softmax}(l)_i$ 或直接 $g_i = \text{softmax}(\text{topk}(l, k))_i$（仅对 top-k 做 softmax，其他为 0）
- Top-K indices: $E_i = \{e_1, e_2, ..., e_k\}$ 其中 $g_{e_1} \ge g_{e_2} \ge ... \ge g_{e_k}$ 且对任意 $e \notin E_i$, $g_e = 0$
- Expert capacity 限制: 若某 expert 已收到 C 个 token，则此后即使被选为 top-k 也跳过该 token（token 被"丢弃"）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# ===== Top-K Gating Pseudocode (标准 MoE) =====
def topk_gating(x, W_g, k=2, capacity_factor=1.0, noise_std=0.0):
    """
    x: [S, H] input tokens
    W_g: [H, E] gate weight matrix
    Returns: routing_table mapping expert→[(token_idx, weight)]
    """
    S, H = x.shape
    E = W_g.shape[1]
    C = int(S * k * capacity_factor / E)
    
    # 1. Gate logits
    logits = x @ W_g  # [S, E]
    
    # 2. Optional: add noise (for exploration during training)
    if noise_std > 0:
        noise = randn(S, E) * noise_std
        logits = logits + noise
    
    # 3. Softmax to get affinity scores
    gate_scores = softmax(logits, dim=-1)  # [S, E]
    
    # 4. Top-K selection: 每 token 选 k 个最高分 expert
    # Using topk: returns top k values and indices along dim=-1
    topk_scores, topk_experts = topk(gate_scores, k, dim=-1)
    # topk_scores:   [S, k] → 归一化的 gating weights
    # topk_experts:  [S, k] → expert indices
    
    # 5. Build routing table T_φ
    # T_φ[e][c] = (token_idx, combine_weight) 或 (token_idx, weight)
    T_phi = {e: [] for e in range(E)}
    
    for token_idx in range(S):
        for j in range(k):
            e = topk_experts[token_idx, j]
            w = topk_scores[token_idx, j]
            if len(T_phi[e]) < C:  # Expert not yet full
                T_phi[e].append((token_idx, w))
            # else: token overflow → dropped for this expert
            # (auxiliary loss encourages balanced routing to minimize drops)
    
    return T_phi, gate_scores
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Top-K Gating 的变体：(1) **Top-1 routing** (Switch Transformer)——仅选 1 个 expert，最稀疏但需更多 experts 保持模型质量；(2) **Top-2 routing** (GShard, FlashMoE)——最广泛使用，平衡稀疏性和模型容量；(3) **Top-8 routing** (DeepSeek-V3)——极高稀疏度 (8/256) 需配合 shared experts 保持质量；(4) **Expert choice routing**——反向：expert 选 top tokens 而非 token 选 experts，保证 load balance 无需 auxiliary loss。

Auxiliary Load Balancing Loss: $L_{aux} = E \cdot \sum_{e=1}^E f_e \cdot P_e$，其中 $f_e = \frac{1}{S} \sum_{i=1}^S \mathbb{1}[e \in E_i]$ 为 expert e 的实际 token 比例，$P_e = \frac{1}{S} \sum_{i=1}^S g_{i,e}$ 为 gate 平均分配给 expert e 的概率。当 f_e 和 P_e 均为 1/E 时 loss 最小（完全均匀）。实际训练中 this loss 乘以 small coefficient (0.01) 添加到主 loss。

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel

## KV Cache (Key-Value Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

KV Cache（Key-Value Cache）是Transformer自回归推理中用于消除冗余计算的核心技术。在自回归生成（autoregressive generation）中，模型逐token生成输出。第t步生成时，需要计算当前token对之前所有t-1个历史token的attention。若每步都重新计算所有历史token的Key和Value投影，则第t步需O(t·d²)新计算。KV Cache的核心思想：每步生成token后，将其Key向量k_t = x_t·W^K和Value向量v_t = x_t·W^V存储（缓存）在GPU内存中。下一步attention计算时，query q_{t+1}只需与已缓存的K矩阵（形状[t, d]）做attention，无需重新投影历史token。这减少计算量从O(t²·d²)到O(t·d²)。代价是存储L层×2（K和V）×序列长度×head_dim×num_heads×precision bytes的缓存，在长序列和大模型中成为内存瓶颈。对于MLLM，visual tokens（来自ViT patch embeddings，通常数百到数千个）显著增加了KV cache的序列长度压力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# KV Cache在自回归生成中的工作流程：
# L层transformer, 每层有独立的KV cache

# === Prompt Encoding Phase (Prefill) ===
X = concat(prompt_embeddings)  # [L_p, d]
for layer l in 1..L:
    K_0^l = X · W_K^l          # [L_p, d] × [d, d] → [L_p, d]  (full projection)
    V_0^l = X · W_V^l          # [L_p, d] × [d, d] → [L_p, d]
    缓存 K_0^l, V_0^l           # 存储到GPU memory

# === Generation Phase (Decode) ===
for step t = 1, 2, ...:
    x_t = embedding(token_{t-1}) # 上一个生成的token
    for layer l in 1..L:
        k_t^l = x_t · W_K^l      # [1, d]  (仅计算新token的K/V)
        v_t^l = x_t · W_V^l
        K_t^l = [K_{t-1}^l; k_t^l]  # 追加到已有cache: [L_p+t, d]
        V_t^l = [V_{t-1}^l; v_t^l]
        o_t^l = Softmax(q_t^l · (K_t^l)^T / √d) · V_t^l  # attention
    生成 next_token = argmax(output_projection(o_t^L))

# 内存占用示例（Qwen2.5-VL-7B: visual tokens=1024, text tokens=256, 28 layers, 28 heads, head_dim=128, BF16）:
# 单层K cache: (1024+256) × 28 × 128 × 2 bytes = 9.2 MB
# 单层V cache: 同上 = 9.2 MB
# 总KV Cache: 28 layers × 2 × 9.2 MB ≈ 515 MB
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

KV Cache在PyTorch中的实现：每层维护两个tensor缓冲区，prefill时批量写入所有prompt token的K/V，decode时每次追加一个token（torch.cat操作）。HuggingFace Transformers的`DynamicCache`类管理动态增长的KV cache。推理框架如vLLM使用PagedAttention将KV cache按block（page）管理以减少内存碎片。KV cache压缩技术分为三类：(1) eviction——丢弃低重要性token（H2O, SnapKV）；(2) quantization——降低KV精度（KIVI, GEAR, MiKV）；(3) merging——将低重要性token合并到保留token（KVMerge, CaM, FlowMM）。

涉及论文标题：
- FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference

## KV Cache Merging (合并式KV Cache压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

KV Cache Merging是一种KV cache压缩范式，与eviction（丢弃）和quantization（量化）并列。核心思想：在保留KV cache预算有限时，不直接丢弃低重要性token的KV状态，而是将其**合并（merge）**到被保留的高重要性token的KV状态中，从而以紧凑表示保留更丰富的上下文信息。形式化：给定KV cache K_t, V_t ∈ R^{L×d}和目标压缩比B，首先选出top-B个pivot tokens的K^p, V^p（保留完整信息），然后将剩余的non-pivot tokens K^n, V^n按相似度合并到pivot tokens中：K^{merged}, V^{merged} = f_merge(K_t, S), g_merge(V_t, S)，其中S ∈ R^{L×L}为token间的相似度矩阵。典型合并操作使用余弦相似度匹配最近邻后做加权平均。Merging相比eviction的优势：即使被合并token的信息被压缩，其内容仍部分保留在pivot token中（而非完全丢失），减少context fragmentation和hallucination。挑战：(1) 不准确的相似度计算可能将无关token合并导致语义混淆；(2) 在多模态场景下，不同模态token的分布偏移（distributional divergence）使简单的余弦相似度合并不可靠。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 通用KV Cache Merging流程：
Input: K ∈ R^{L×d}, V ∈ R^{L×d}, budget B (保留比例)
Output: K^{merged} ∈ R^{B·L×d}, V^{merged} ∈ R^{B·L×d}

# Step 1: Token重要性评估（基于累积attention scores）
importance[i] = Σ_h Σ_{j∈recent_tokens} α_{j→i}^h

# Step 2: 选出pivot set和non-pivot set
sorted_indices = argsort(importance, descending=True)
pivot_indices = sorted_indices[:B·L]        # top-B 保留
non_pivot_indices = sorted_indices[B·L:]    # 其余将被合并
K^p = K[pivot_indices];    V^p = V[pivot_indices]

# Step 3: 基于相似度的最近邻合并
for each i in non_pivot_indices:
    similarities = cosine_similarity(K[i], K^p)  # [B·L]
    j_star = argmax(similarities)                # 最近邻pivot
    weight = attention_based_weight(i, j_star)
    K^p[j_star] = weight · K^p[j_star] + (1-weight) · K[i]
    V^p[j_star] = weight · V^p[j_star] + (1-weight) · V[i]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

KV Cache Merging的典型实现：在每层attention计算后，对KV cache做后处理合并。PyTorch实现使用torch.cosine_similarity和加权平均操作。常用变体：(1) KVMerge（Wang et al., 2024）——基于模型自身指示的合并位置决策；(2) CaM（Zhang et al., 2024）——将eviction候选合并到保留状态中；(3) MiniCache（Liu et al., 2024a）——利用层间KV相似度做intra-layer压缩；(4) LOOK-M（Wan et al., 2024b）——multimodal-specific的KV cache合并方法。FlowMM在此基础上引入了跨模态信息流引导的层自适应合并策略和敏感度自适应的token匹配，解决多模态场景下统一合并策略的不足。

涉及论文标题：
- FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference

## Cross-Modal Information Flow in MLLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cross-Modal Information Flow（跨模态信息流）指MLLM的Transformer层中不同模态token（visual和text）之间的attention交互模式和强度。在MLLM的每一层attention计算中，QK^T attention矩阵包含四个block：visual→visual、visual→text、text→visual、text→text。Cross-modal attention scores A_{v→t}（visual→text）和A_{t→v}（text→visual）共同构成跨模态信息流。FlowMM发现这一信息流在MLLM不同层中存在显著分化：浅层以intra-modal交互为主（cross-modal attention比例低），负责低层单模态特征提取；深层跨模态交互显著增强（cross-modal attention比例高），负责跨模态融合和高层语义抽象。此pattern在ALFRED/MMCoQA/TextNeedle三个不同任务上一致。这一发现是FlowMM层自适应合并策略的理论基础。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Cross-Modal Information Flow量化（FlowMM公式6-7）：
for head h in 1..H:
    A_{v→t}^{l,h} = Σ_{v∈V} Σ_{t∈T} α_{v→t}^{l,h}   # visual→text attention
    A_{t→v}^{l,h} = Σ_{t∈T} Σ_{v∈V} α_{t→v}^{l,h}   # text→visual attention

ρ^l = (1/H) · Σ_{h=1}^{H} (A_{v→t}^{l,h} + A_{t→v}^{l,h}) / A^{l,h}

# ρ^l ∈ [0, 1]: 
#   → 0: 几乎纯intra-modal交互
#   → 1: 几乎纯cross-modal交互
# FlowMM在Qwen2.5-VL-7B上的发现:
#   浅层(layers 1-12): ρ^l < 0.2, intra-modal主导
#   深层(layers 13-28): ρ^l > 0.2, cross-modal显著增加
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Cross-Modal Information Flow分析通过一次校准前向传播即可完成：在校准样本上执行inference，从每层attention矩阵中提取visual↔text的attention scores，计算ρ^l。通常需少量（数十到数百个）校准样本取平均以得到稳定pattern。FlowMM将此用于指导KV cache合并策略——若ρ^l ≥ θ（阈值最优值0.2-0.3），执行跨模态合并；若ρ^l < θ，执行模态内合并。阈值过低（<0.1）导致浅层过早跨模态合并→模态信息混淆（modal confusion）；过高（>0.4）限制深层跨模态融合→跨模态语义理解不足。

涉及论文标题：
- FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference

## Token Sensitivity-Aware KV Cache Management

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Sensitivity-Aware KV Cache Management（token敏感度感知的KV cache管理）是一种识别和保护高敏感度token的KV cache压缩增强策略。核心思想：并非所有token对模型输出质量贡献相同——某些token包含任务关键信息（如问题中的实体名、特定指令token），其KV状态在合并或丢弃时会对生成质量造成显著负面影响。这些token被定义为"高敏感度token"。FlowMM将sensitivity定义为token对模型输出保真度的贡献度——若合并某token的KV状态导致后续生成准确度/相关性显著下降，则该token为高敏感度。直接通过逐一扰动测试测量敏感度计算成本过高，因此FlowMM使用attention scores作为sensitivity的零开销近似：attention scores直接量化token对当前生成步骤的影响，可在正常attention计算中免费获得。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Sensitivity-Adaptive Token Matching (FlowMM公式9-10):
# Step 1: 计算token间余弦相似度
for token i in K^n (non-pivot set):
    for token j in K^p (pivot set):
        u_{i,j} = (k_i^T · k_j) / (||k_i|| · ||k_j||)

# Step 2: Sensitivity-gated nearest neighbor matching
for token i in K^n:
    # 仅在低敏感度pivot (I_j ≤ τ) 中搜索最近邻
    j* = Argmax_{j∈K^p, I_j ≤ τ}(u_{i,j})
    # 高敏感度pivot (I_j > τ) 被保护，不接受任何合并
    merge(K_i, V_i) → K_{j*}, V_{j*}

# 设计逻辑：
# - 低敏感度pivot: 可接受合并 → 信息聚合点
# - 高敏感度pivot: 不接受合并 → 信息保护
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Sensitivity evaluation使用proxy tokens方法（FlowMM公式8）：选择prompt末尾少量token作为proxy（这些token通常capture任务特定上下文信息），对每个token i计算其从所有proxy tokens收到的attention scores之和：I^{l,h}(i) = Σ_{j∈P} α_{j→i}^{l,h}。相比使用全局累积attention（可能biased），proxy token方法提供更公平的token重要性估计。敏感度阈值τ需校准：过高→保护过多pivot→合并候选不足→压缩效率降低；过低→高敏感度token未受保护→任务性能下降。FlowMM消融实验（Table 4）：移除sensitivity protection在TextNeedle任务上性能下降最显著（-3.68%），因为该任务需要精确保留特定文本token信息。该策略可与其他KV cache压缩方法组合使用。

涉及论文标题：
- FlowMM Cross-Modal Information Flow Guided KV Cache Merging for Efficient Multimodal Context Inference

## SAM2 (Segment Anything Model 2)

术语是什么？
SAM2（Segment Anything Model 2）是Meta AI于2024年7月发布的promptable视觉分割基础模型，继承自SAM但扩展到视频领域。SAM2使用Hiera-based hierarchical vision transformer作为图像编码器（四个尺寸：Tiny 38.9M、Small 46M、Base+ 80.8M、Large 224.4M），核心创新是**streaming memory module**——存储过去帧的object-aware memory context来condition当前帧预测，实现实时视频处理而无需回溯所有历史帧。输入可以是单张图像或视频帧序列，支持point/box/mask三种prompt方式指定分割目标。输出是跨帧的masklet（时空一致性分割mask序列）。2024年9月发布SAM2.1改进版，2024年12月更新支持更好的multi-object tracking和torch.compile加速。

从算法pipeline角度拆解术语：
在FoundationMotion的Object Detection & Tracking阶段（Sec 3.2.3），SAM2作为时序tracking backbone：
```
# SAM2 Two-Stage Tracking in FoundationMotion
M_0 = SAM2VideoPredictor.init_state(video[0], prompts=B_init)
# B_init = 初始帧检测的所有person + object bboxes作为prompts

for t in 1..T:
    M_t = SAM2VideoPredictor.propagate_in_video(M_{t-1})
    # SAM2内部：memory encoder计算当前帧特征 → memory bank存储
    # → memory attention跨帧condition → mask decoder输出当前帧mask
    
    if t % 5 == 0:  # keyframe refinement
        B_new = Hands23.detect(video[t])  # 重新检测手部
        SAM2VideoPredictor.add_new_prompts(B_new)  # 注入新prompts纠正drift
```

ID分配：persons ID∈[0,99]，left_hand=ID×10+1，right_hand=ID×10+4，objects ID≥1000。

术语一般如何实现？如何使用？
通过`SAM2VideoPredictor`类使用。初始化时调用`init_state(video_frame, prompts)`，prompts包含正负点坐标、bbox坐标或mask。之后循环调用`propagate_in_video()`获取每帧mask。支持中途调用`add_new_prompts()`或`remove_objects()`动态增删跟踪目标。官方仓库：https://github.com/facebookresearch/sam2，支持torch.compile加速VOS。

涉及论文标题：
- FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos

## Grounded-DINO

术语是什么？
Grounded-DINO是将DINO（DETR with Improved Denoising Anchor Boxes）transformer检测器与grounded pre-training结合的开放词汇目标检测器，发表于ECCV 2024。核心设计是**三阶段tight fusion**：(1) Feature Enhancer阶段做deep early fusion（deformable self-attention + image-to-text/text-to-image cross-attention）；(2) Language-Guided Query Selection阶段选择与文本最相关的top-Nq图像特征作为decoder queries；(3) Cross-Modality Decoder阶段每层做self-attention→image cross-attention→text cross-attention→FFN。支持任意文本类别名作为输入，输出检测bbox和对应类别标签。Grounding DINO 1.5 Pro使用ViT-L backbone在20M+ grounding images上训练达到SOTA零样本检测性能。

从算法pipeline角度拆解术语：
在FoundationMotion的Open-Vocabulary Object Detection阶段（Sec 3.2.1）：
```
# Step 1: Qwen2.5-VL-7B生成场景中的object categories
O = Qwen2.5-VL.scene_analysis(video_frame[0])  # → {o1, o2, ..., on}

# Step 2: Grounded-DINO逐类别检测（非拼接所有类别）
for each category o_i in O:
    bboxes_i = GroundedDINO(image=video_frame[0], text_prompt=o_i)
    # text_prompt单独query每个类别 → one-to-one box-label alignment

# 输出: B_obj = {(bbox, class_label)_i} 定位到具体像素坐标
```

论文的ablation关键发现：使用Grounded-DINO per-class query（而非concat所有类）可强制one-to-one bbox-label对齐，提升检测质量。

术语一般如何实现？如何使用？
通过HuggingFace Transformers或官方GitHub仓库使用。输入图像+文本prompt（如"a red car. a person."），输出detection bboxes和对应的text-matched类别。支持batch推理。官方：https://github.com/IDEA-Research/GroundingDINO；HuggingFace: `grounding-dino` model。也支持TensorRT部署（Grounding DINO 1.5 Edge在NVIDIA Orin NX上达75.2 FPS）。

涉及论文标题：
- FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos

## VGGT (Visual Geometry Grounded Transformer)

术语是什么？
VGGT（Visual Geometry Grounded Transformer）是Oxford VGG + Meta AI开发的feed-forward神经网络，获CVPR 2025 Best Paper Award。它从1张至数百张任意视角图像中，在数秒内直接推断场景的全部3D属性：相机位姿（外参+内参）、深度图、点云图和3D点轨迹，无需任何后处理优化（如bundle adjustment）。使用transformer-based aggregator处理多视图图像，然后专门的camera_head预测pose encoding并解码为旋转+平移+内参矩阵（OpenCV convention）。与DUSt3R/MASt3R需要visual geometry optimization不同，VGGT直接输出准确参数。

从算法pipeline角度拆解术语：
在FoundationMotion的Video Preprocessing阶段（Sec 3.1），VGGT用于**相机运动过滤**：
```
# FoundationMotion中VGGT的使用
frames = sample_frames(video_clip, stride=5)  # 采样关键帧
poses = VGGT.infer_poses(frames)  # 推断每帧相机位姿：R_t, T_t

# 计算相机运动分数
delta_t = mean(||T_{i+1} - T_i||)   # 平均位移变化
delta_r = mean(||R_{i+1} - R_i||)   # 平均旋转变化
motion_score = alpha*delta_t + beta*delta_r + gamma*max(delta_t) + delta*max(delta_r)

if motion_score > 0.3:  # 阈值过滤
    skip_video()  # 相机运动过大→tracking质量差→丢弃
```

作用：过滤相机大幅运动的视频，因为此时物体运动+camera motion耦合使人类都难以描述其真实运动轨迹，tracking和标注精度严重退化。

术语一般如何实现？如何使用？
通过官方GitHub仓库使用，输入多视图图像，输出camera poses、depth maps、point maps。适用于3D重建、pose estimation、multi-view geometry等场景。也支持COLMAP格式bundle adjustment（2025年6月后）。官方：https://github.com/chengwei920412/vggt-3dgs。

涉及论文标题：
- FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos

## Motion Understanding / Motion Reasoning in VLMs

术语是什么？
Motion Understanding（运动理解）指视觉语言模型对视频中物体运动、空间关系变化和时间动态的理解能力，区别于传统VLM的"what"理解（物体识别、场景分类、事件检测），Motion Understanding聚焦于"how"——物体如何运动（方向、速度、轨迹）、运动之间的空间关系（相对位置变化、几何约束）和时间顺序（哪个动作先发生）。现有benchmarks如MotionBench涵盖6类motion task（action recognition, temporal ordering, motion attribute等），但缺乏spatial reasoning维度（如何交互、相对轨迹、几何约束）。FoundationMotion通过5类QA覆盖：Motion Recognition、Action Order、Motion-related Objects、Location-based Motion、Repetition Count。

从算法pipeline角度拆解术语：
Motion Understanding的训练pipeline（FoundationMotion方式）：
```
# 数据生成端
video → [detection + tracking] → bbox_trajectories_JSON 
       → GPT-4o-mini(frames, bbox_json, overlay) → motion_caption (7维度)
       → GPT-4o-mini(frames, caption) → 5-type QAs

# 模型训练端
VLM_base (NVILA/Qwen-VL) + 467K motion QAs → SFT fine-tuning
# 评估：模型在motion benchmarks上的QA准确率
# e.g. MotionBench: 45.7% → 46.7% (NVILA-Video-15B → +FT)
# e.g. AV-Car: 84.4% → 91.5% (NVILA-Video-15B → +FT, +7.1%)
```

关键发现：fine-tuning 46.7K videos (467K QAs)即可显著提升motion understanding，证明高质量motion数据比模型规模更重要（15B fine-tuned超越72B base和Gemini-2.5-Flash）。

术语一般如何实现？如何使用？
通过motion-centric数据fine-tuning实现。方法包括：(1)构建motion QA数据集（人工标注或自动pipeline）；(2)SFT fine-tuning开源VLM（使用llamafactory或官方training code）；(3)在MotionBench、VLM4D等benchmarks上评估。训练配置：cosine annealing LR schedule、Adam optimizer、无weight decay。评估指标：多选QA准确率（4选项随机分布）。

涉及论文标题：
- FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos

## Multi-Object Tracking (MOT) in Videos

术语是什么？
Multi-Object Tracking（MOT，多目标跟踪）是计算机视觉中的核心任务，指在视频序列中同时检测、识别并持续跟踪多个目标物体（如行人、车辆、手部等）的运动轨迹。与Single Object Tracking（SOT）不同，MOT需要处理目标间的遮挡、身份切换（ID switch）、目标进出画面等复杂情况。FoundationMotion中的MOT pipeline使用**hierarchical multi-stage detection + two-stage tracking**策略：预先通过多个专用检测器（Grounded-DINO for open-vocab objects、Cascade Mask R-CNN for persons、Hands23 for hands）获得初始检测，然后使用SAM2进行cross-frame propagation和keyframe refinement。

从算法pipeline角度拆解术语：
FoundationMotion的MOT pipeline：
```
# Hierarchical Detection
objects  = GroundedDINO(frame_0, categories_from_QwenVL)  # open-vocab
persons  = CascadeMaskRCNN_ViTDetH(frame_0, tau=0.8)      # person (high conf)
for each detected person:
    keypoints = ViTPose+(person_region)                    # 42 hand kpts
    hands     = Hands23(expand_region(hand_kpts, 1.5x))   # left/right hand
    # Hands23 output: (bbox, side{L/R}, contact_state, object_bbox)

# ID Assignment (hierarchical encoding)
person_id in [0, 99]
left_hand_id  = person_id * 10 + 1
right_hand_id = person_id * 10 + 4
object_id >= 1000  # non-person objects

# Two-Stage Tracking with SAM2
M_0 = SAM2.init(frame_0, prompts=all_detections)  # Stage 1: init
for t in 1..T:
    M_t = SAM2.propagate(M_{t-1})                  # Stage 1: per-frame propagation
    if t % 5 == 0:                                 # Stage 2: keyframe refine
        B_new = Hands23.detect(frame_t)            # re-detect hands
        M_t = SAM2.propagate(M_{t-1}, B_new)       # inject new prompts

# Trajectory Output
for each tracked object:
    trajectory[obj] = {
        bbox: [[l/width, t/height, r/width, b/height]_t for t in 0..T],
        object_type: str,
        interactions: [neighbor_obj_ids_at_t for t in 0..T]
    }
```

实际输出用于GPT-4o-mini生成motion captions的JSON格式：每个object有跨所有帧的bbox序列（归一化坐标）、object_type和interactions（记录每帧与其他object的空间关系）。

术语一般如何实现？如何使用？
Video MOT通常通过tracking-by-detection范式实现：先用detector获取每帧检测，再用association算法（如Kalman filter + Hungarian matching in SORT/DeepSORT）或learned tracker（如SAM2的memory-based propagation）跨帧关联。FoundationMotion选择SAM2-based方法因为其memory module对遮挡和外观变化更鲁棒。两阶段设计（全帧propagation + keyframe re-detection）平衡了效率（全帧SAM2 propagation计算量大）和精度（纯propagation会drift）。

涉及论文标题：
- FoundationMotion: Auto-Labeling and Reasoning about Spatial Movement in Videos

## Unified Attention Abstraction (Relevance Scoring + Aggregation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Unified Attention Abstraction（统一注意力抽象）是 MetaAttention 提出的核心抽象，将各类 attention 机制的共同本质归结为两个基本操作：(1) **Relevance Scoring（相关性评分）**——计算输入 tokens 之间的成对相似度或交互，通常通过内积或其他相似度度量实现，形成 token-to-token 关系的数学描述；(2) **Aggregation（聚合）**——利用 relevance scores 将上下文信息整合为每个 token 的表示，即加权求和 Value vectors。这两个操作捕获了所有 attention 变体的共同骨架：先计算 token 间相关性，再用相关性加权聚合信息。

该抽象的关键在于其**完备性**——能够表达 Softmax Attention、Sigmoid Attention、ReLU Attention、Linear Attention（Mamba2）、RetNet、Multi-head Latent Attention (MLA)、Sliding Window Attention、Sparse Attention (SeerAttention)、Gated Retention 等十余种 attention 变体。每个变体的差异被归约为：(1) relevance scoring 的具体计算方式（如 matmul vs chunk-wise matmul vs state-based matmul）；(2) 中间 tensor 的自定义变换（masking、scaling、normalization）；(3) aggregation 的具体方式（全局 vs 增量压缩 state）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

抽象在 attention pipeline 中的体现——所有 attention 变体共享同一高层结构：
```
# 统一的 attention pipeline（MetaAttention unified template）
def attention(Q, K, V, customizable_functions):
    state = init_state()         # 初始状态（并行模式: None; 循环模式: zeros）
    for segment in sequence:
        # Step 1: Relevance Scoring（固定，不可自定义）
        scores = relevance_scoring(Q[segment], K, state)
        
        # Step 2: Customizable Score Transformation
        scores = scores_Mod(scores)         # 元素级变换（mask/scale）
        weights = scores_RowNorm(scores)    # 行归一化（softmax/sigmoid/L2）
        
        # Step 3: Aggregation（固定，不可自定义）
        output = aggregate(weights, V, state)
        
        # Step 4: Customizable Output Transformation
        output = output_Mod(output)         # 最终输出变换
    return output
```

两种实例化模式：
- **Parallel Pattern**: `relevance_scoring = matmul(Q, K^T)`，`aggregate = matmul(weights, V)`。适用需要全局上下文的 attention（Softmax/Sigmoid/MLA/RetNet Parallel）
- **Recurrent Pattern**: `relevance_scoring = matmul(Q, h)`（h 为压缩 hidden state），`aggregate: h = h + matmul(K[i]^T, V[i])`。适用 stateful attention（Mamba2/RetNet Recurrent/Gated Retention）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MetaAttention 在 7.3k 行 C++/Python 中实现该抽象。用户仅需声明 pattern 类型（Parallel/Recurrent）、定义 input tensor shapes、编写 customizable functions（Mod 和 RowNorm），框架自动完成 scheduling、code generation 和 multi-backend execution。实现的关键技术：RowNorm Online 接口泛化 online softmax 到任意 row-wise normalization；IntermediateTensor scheduling 自动传播 tile shape 和 memory placement；TMA+Tensor Core (NVIDIA) 或 Matrix Core+async copy (AMD) 双 backend 支持。详见论文 Section 3 (Programming with MetaAttention)。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

## Attention Parallel Pattern / Recurrent Pattern

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention Parallel Pattern 和 Recurrent Pattern 是 MetaAttention 从统一 attention 抽象中推导出的两种计算模式，分别对应需要**全局上下文**和可以**压缩为固定大小状态**的 attention 机制。

**Parallel Pattern**（并行模式）：attention 需要在完整 KV sequence 上计算全局上下文。Relevance scoring 实现为并行矩阵乘法 `scores = matmul(Q, K)`（Q 的每个 token query 与所有 K token 做内积，O(N²) 复杂度）。Aggregation 实现为 `output = matmul(weights, V)`（每个 query 聚合所有 V token 的信息）。适用于 Softmax Attention、Sigmoid Attention、RetNet Parallel、MLA、Sliding Window Attention、Sparse Attention 等。关键优化：online block-wise normalization（online softmax/sigmoid 等）避免物化 N×N score matrix。

**Recurrent Pattern**（循环模式）：attention 将上下文压缩为固定大小的 hidden state h，迭代遍历 sequence。Relevance scoring 实现为 `output = matmul(Q, h)`（仅需与压缩 state 做 matmul，O(d²) 复杂度）。Aggregation 实现为 `h = h + matmul(K[i]^T, V[i])`（增量更新 hidden state）。适用于 Mamba2 SSM、RetNet Recurrent、Gated Retention 等 state space model 类 attention。关键优化：chunk parallelism——将 sequence 分块并行处理，块内用 recurrent 更新 state，块间传递 state。

从算法pipeline角度拆解：

Parallel Pattern 伪代码（以在线归一化为例）：
```
def parallel_attention(Q, K, V):
    O = zeros(B, H, S, d_v)
    for q_block in Q.split(B_r):            # 沿 seq_len 并行
        m, l, O_acc = -inf, 0, 0
        for kv_block in (K, V).split(B_c):  # 沿 KV seq_len 串行迭代
            S = q_block @ kv_block.K^T       # [B_r, B_c] relevance scoring
            S = scores_Mod(S)                # mask/scale
            m_new = max(m, rowmax(S))
            l = l * exp(m - m_new) + rowsum(exp(S - m_new))
            O_acc = O_acc * exp(m - m_new) + softmax(S) @ kv_block.V
            m = m_new
        O[q_block] = O_acc / l
    return O
```

Recurrent Pattern 伪代码（以 Mamba2 chunk parallelism 为例）：
```
def recurrent_attention_chunked(Q, K, V, chunk_size):
    h = zeros(B, H, d_state)          # 初始压缩 state
    for chunk in sequence.split(chunk_size):
        # Chunk 内并行计算 relevance + aggregation
        O[chunk] = matmul(Q[chunk], h)   # relevance scoring from state
        # 块内并行更新 state（每个位置增量贡献）
        h = h + sum(matmul(K[i]^T, V[i]) for i in chunk)
    return O
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MetaAttention 将两种 pattern 实现为固定的 kernel 模板。Parallel pattern 模板包含 online normalization mainloop（外层沿 KV seq_len 分 tile，内层 TMA load + wgmma QK^T + SIMT customizable functions + wgmma PV + rescale）；Recurrent pattern 模板包含 chunk-based mainloop（外层沿 chunk 迭代，内层并行 matmul + state update）。用户选择 pattern 后，customizable functions 通过 code inlining 注入模板固定位置，无需修改 scheduling logic。同一 pattern 可应用于多个 attention 变体（如 Parallel pattern 支持 Softmax/Sigmoid/ReLU/MLA/RetNetParallel/Sparse GQA）。

涉及论文标题：
- MetaAttention: A Unified and Performant Attention Framework across Hardware Backends

## Block Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block Sparse Attention（块稀疏注意力）是一种通过block粒度稀疏化attention score矩阵来降低$O(n^2)$计算复杂度的方法。与element-wise sparse attention不同，block sparse attention将Q和KV分别划分为blocks（如block size=128），根据先验知识定义哪些(Q-block, KV-block) pair需要计算（FB=Full）、哪些部分计算（CB=Causal，需逐元素mask）、哪些完全跳过（EB=Empty）。Block粒度保留了GPU Tensor Core友好的dense tile数据布局，避免了per-element sparse indexing的control divergence overhead。常见pattern：Causal Attention（下三角mask）、Strided Attention（对角线banded pattern）、Global+Local Attention（Longformer/BigBird，全局+局部窗口）、Star Attention（anchor block+其余causal）、Streaming Attention（attention sink + 最近token窗口）。

从算法pipeline角度拆解，block sparse attention在tiled attention kernel中的流程：
```
for each (q_block, kv_block) in attention mask:
    if (q_block, kv_block) in EB: continue  # 完全跳过
    Q_tile = load_to_sram(Q[q_block])
    K_tile = load_to_sram(K[kv_block])
    scores = Q_tile @ K_tile.T  # Tensor Core MMA
    if (q_block, kv_block) in CB:
        scores = mask_apply(scores, causal_mask)  # 仅CB blocks需要
    # online softmax with running (m, l, O)
    m_new = max(m_old, rowmax(scores))
    O_acc = O_acc * exp(m_old - m_new) + exp(scores - m_new) @ V_tile
    l = l * exp(m_old - m_new) + rowsum(exp(scores - m_new))
    m_old = m_new
O = O_acc / l  # final normalization
```
**Annotations**: EB blocks完全跳过（节省$block\_size^2$ FLOPs），CB blocks内部仍需逐元素mask（因下三角跨越block边界），FB blocks零mask overhead。

术语一般如何实现？如何使用？单GPU kernel：FlexAttention (PyTorch)通过create_block_mask生成BlockMask（kv_num_blocks + kv_indices紧凑编码），FlashInfer用BSR格式支持block sparse。分布式系统：UltraAttn首次提出通用CP系统支持block sparse attention，将FB/CB/EB集合编码为ILP约束自动解决分布式负载均衡。适用场景：long-context LLM training (Longformer, BigBird, SAMBA)、video generation (VideoGPT strided)、vision (Swin Transformer shifted window)、inference with attention sink (Star Attention, StreamingLLM)。

涉及论文标题：
- UltraAttn: Efficiently Parallelizing Attention through Hierarchical Context-Tiling
- FlashAttention Fast and Memory-Efficient Exact Attention with IO-Awareness

## GEMM+AR / GEMM+RS / GEMM+A2A Communication Patterns

术语是什么？

GEMM+AR、GEMM+RS、GEMM+A2A 是分布式模型训练和推理中三种典型的"通用矩阵乘法 + 数据依赖的集合通信"组合模式。AR = AllReduce，RS = ReduceScatter，A2A = All-to-All。这些 pattern 广泛存在于 multi-GPU 并行策略中，且通信开销往往是端到端延迟的主要瓶颈（FlashOverlap profiling 显示 GEMM+AR 在 LLM serving 中占 31.6-42.2%、GEMM+RS 在 Llama2-7B training 中占 ~30%、GEMM+A2A 在 Mixtral-8x7B training 中占 >40%）。

从算法pipeline角度拆解术语：

三种 pattern 在模型训练/推理 pipeline 中的位置：

```
(1) GEMM+AR (AllReduce):
发生位置: Tensor Parallelism (TP) 和 Data Parallelism (DP)
流程:
  TP: 每个 GPU 计算 GEMM 的部分结果 (partial sum)
      → AllReduce 将所有 GPU 的 partial sum 求和并广播
      → 每个 GPU 获得完整结果
  DP: 每个 GPU 独立计算 gradient
      → AllReduce 求和所有 GPU 的 gradients
      → 每个 GPU 获得平均 gradient

典型场景: Llama3-70B TP=8, attention proj + FFN 后的 AllReduce
通信量: 2×(N-1)/N × data_size (Ring AllReduce)

(2) GEMM+RS (ReduceScatter):
发生位置: TP training (AllReduce 分解为 RS+AG) + FSDP backward
流程:
  TP training: GEMM partial results → ReduceScatter (沿 row 维 reduce 并 scatter)
              → AllGather (聚合完整结果)
  FSDP: weight gradient GEMM → ReduceScatter → 每个 GPU 持有部分 reduced gradient

典型场景: Llama2-7B FSDP training
通信量: (N-1)/N × data_size

(3) GEMM+A2A (All-to-All):
发生位置: Expert Parallelism (EP) in MoE models
流程:
  每个 GPU 计算其 local experts 的 FFN (GEMM)
  → All-to-All: 每个 GPU 将其计算的 token 发送到 token 原始 GPU
  → 每个 GPU 接收来自所有 GPU 的 token，形成完整 batch

典型场景: Mixtral-8x7B EP=4
特点: 动态 routing 导致 GPU 间 workload imbalance → 通信开销加剧
```

**Annotations**: FlashOverlap 的 GEMM+AR 加速在 RTX 4090 上达 1.02-1.65×、A800 上达 1.30×。GEMM+RS 在 A800 pairwise NVLink 上加速 1.07-1.31×。GEMM+A2A 在 MoE 场景因 workload imbalance 需要 predictor 取所有 GPU max 延迟。

术语一般如何实现？如何使用？

三种 pattern 均通过 NCCL 集合通信 API 实现。FlashOverlap 通过统一的 signaling + reordering 机制支持全部三种 pattern——仅 reordering 粒度不同（tile/subtile/subtoken level），signaling 和 counting table 机制完全复用。在 PyTorch 分布式训练中，TP 使用 `torch.distributed.all_reduce`、FSDP 使用 `torch.distributed.reduce_scatter`、MoE 使用 `torch.distributed.all_to_all`。FlashOverlap 替换这些调用为带 overlap 的实现。

涉及论文标题：
- Efficient and Adaptable Overlapping for Computation and Communication via Signaling and Reordering
