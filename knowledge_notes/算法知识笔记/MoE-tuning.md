## MoE-tuning

术语解释
一种将 dense MLLM 逐步转换为 sparse MoE 架构的三阶段微调框架：第一阶段预训练对齐跨模态表示，第二阶段扩展为 MoE 结构（复制 FFN + 训练 router），第三阶段 instruction tuning。由 MoE-LLaVA [25] 首次提出，EvoMoE 在此基础上改进 Stage II/III。

术语是什么？
MoE-tuning 解决的核心问题是：直接将 dense LLM 同时转换为 vision-language model 和 sparse MoE 会导致显著的性能下降。因此采用分阶段策略：

- **Stage I (Pre-training + Alignment)**：仅训练 MLP Projector，将视觉 token 映射至 LLM 的语义空间。使用混合多模态数据集（MIMIC-IT、LRV、SViT、LVIS），建立基础的视觉-语言理解能力。
- **Stage II (MoE Initialization)**：将 LLM 中 alternating decoder layer 的 FFN 替换为 MoE 层。传统方式是将原始 FFN 复制 N 份作为 N 个 expert 的初始化，训练线性 router（top-2 selection）和所有参数。EvoMoE 改进为 Expert Evolution（仅训练 1 个 expert + 演化）。
- **Stage III (Instruction Tuning)**：使用 LLaVA-mix-665k 进行指令微调，进一步提升多模态任务表现。EvoMoE 改进为训练 DTR 替代线性 router。

MoE-tuning 的损失函数：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{regressive}} + \alpha \cdot \mathcal{L}_{\text{aux}}, \quad \alpha = 0.001$$

其中 L_aux 为负载均衡损失：
$$\mathcal{L}_{\text{aux}} = E \cdot \sum_{i=1}^{E} \mathcal{F}_i \cdot \mathcal{G}_i$$

F_i 为每个 expert 处理的 token 比例，G_i 为每个 expert 的平均路由概率。

从算法pipeline角度拆解术语：
```
# MoE-tuning 三阶段（MoE-LLaVA 原始版本）
# Stage I: Pretraining
for epoch in pretrain_epochs:
    img_tokens = VisionEncoder(image)        # CLIP-L
    proj_tokens = MLP_Projector(img_tokens)  # MLP with GeLU
    text_tokens = Tokenizer(text)
    all_tokens = concat(proj_tokens, text_tokens)
    loss = LM_Head(LLM(all_tokens))
    update(MLP_Projector.params)

# Stage II: MoE Initialization
# 将选定的 FFN layers 替换为 MoE：
for layer in MoE_layers:
    layer.experts = [copy(layer.FFN) for _ in range(N)]  # 复制初始化
    layer.router = Linear(hidden_dim, N)                  # 线性 router
# 训练所有参数（除 vision encoder）
for data in LLaVA_mix_665k:
    router_logits = layer.router(token_hidden)
    top_k_indices, top_k_probs = topk(softmax(router_logits), k=2)
    expert_outputs = sum(prob * layer.experts[idx](token) for ...)
    loss = LM_loss + 0.001 * load_balance_loss(L_aux)
    update(all_params)

# Stage III: Instruction Tuning
# 同上，使用更大的数据量进行最终的指令微调
```

术语一般如何实现？如何使用？
- 基于 LLaVA 1.5 代码库实现（https://github.com/haotian-liu/LLaVA）
- MoE-LLaVA 开源仓库：https://github.com/PKU-YuanGroup/MoE-LLaVA
- 使用 DeepSpeed ZeRO-2 进行分布式训练
- MoE layer 采用 alternating placement（每隔若干层放一个 MoE layer）而非全部层替换
- 支持多种 LLM backbone：Qwen 系列、StableLM、Phi-2、OpenChat 等
- 训练硬件：8x A100-80G，bf16 精度
- MoE-tuning 的局限性（由 EvoMoE 揭示并解决）：复制初始化导致 Expert Uniformity、线性 Router 导致 Router Rigidity

涉及论文标题：
- EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models

---
