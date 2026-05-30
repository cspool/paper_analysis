## Expert Capacity / Capacity Factor

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Capacity（专家容量）是 MoE 训练中控制每个专家每步最多处理 token 数量的约束机制。定义为：

$$Expert\ Capacity = CF \cdot \frac{B \cdot L}{|\mathcal{E}|}$$

其中 CF 是 Capacity Factor（容量因子），B 是 batch size，L 是序列长度，|E| 是专家总数。CF=1.0 表示每个专家容量等于均匀分配时的期望 token 数。CF>1 提供冗余容量以减少 token dropping，但增加内存和计算开销。超出容量的 token 被丢弃（token dropping），即那些 token 跳过该 MoE 层的专家计算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Expert Capacity 在 MoE 层 forward 中的作用：

```
def moe_forward_with_capacity(x, gate, experts, capacity_factor=1.0):
    B, L, D = x.shape
    num_experts = len(experts)
    capacity = int(capacity_factor * B * L / num_experts)

    gate_scores = gate(x)           # [B*L, num_experts]
    expert_idx = argmax(gate_scores, dim=-1)  # top-1

    # 每个专家的 token 计数器
    expert_counts = zeros(num_experts)
    output = zeros_like(x)

    for t in range(B * L):
        e = expert_idx[t]
        if expert_counts[e] < capacity:
            output[t] = experts[e](x[t]) * gate_scores[t, e]
            expert_counts[e] += 1
        # else: token dropped (output[t] remains 0, or use residual)

    return output
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- CF 的选择是 memory vs. quality 的 trade-off：CF=1.0 节省内存但可能有较多 token dropping（尤其当 gate 路由不均衡时）；CF=1.25-2.0 更安全但增加计算和通信
- **Auxiliary load balancing loss**: 通常与 capacity 协同使用——loss 惩罚路由不均衡，capacity 作为硬件约束的硬上限
- FOLDMOE 使用 CF=1.0，配合 EP=16（每 GPU 1 个专家），在保证训练收敛（通过 Figure 12 验证 loss curve 一致性的同时）最小化内存
- Joint MoE Scaling Laws 在评估阶段使用 dropless 模式（CF→∞ / 移除 capacity 限制），确保所有 token 均被处理以避免 capacity-induced dropping 影响 loss 评估

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
- Llama 3 Meets MoE: Efficient Upcycling
- LocMoE: A Low-overhead MoE for Large Language Model Training
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping

**Lancet 的不规则 Expert Capacity**（Lancet, MLSys 2024）：

Lancet 提出了 expert capacity 在 micro-batch 分区场景下的不规则使用机制。当沿 batch 维度将 MoE layer 的输入分为 k 个 micro-batch 时，直接等比例缩小每个 micro-batch 的 expert capacity（C/k）会导致额外 token dropping——因为 token 分布不均匀（如第一个 micro-batch 有 3/4 C 个 token 指向某 expert，超过了 C/k）。Lancet 实现 special gating operator 在各 partition 间传递容量信息：第一个 partition 使用多少容量后，调整剩余 capacity 传给后续 partition。保证所有 partition 的 token-to-expert mapping 和 token dropping 与不分区的原版完全一致（数学等价性）。但这导致每个 partition 可以向每个 expert 发送 0 到 C 之间任意数量的 token，引入不规则 all-to-all 通信（Irregular All-to-Allv）。

**LocMoE 的 Expert Capacity 下界理论**：

LocMoE 首次将 pMoE (Chowdhury et al., 2023) 在 CV 领域的 expert capacity 下界结论推广到 NLP 领域，并结合网络结构分析：

前提假设：
1. Gating weight 范数 ‖ω_i‖ 对所有 expert 等价
2. Token 均匀分布在高维单位球面 (‖x_m‖ = 1)
3. GrAP 的正交 gating weight 满足 Lemma 2：各 expert 被等概率选择 P{i_j = i'} = 1/n

基于高维球面几何推导：
- token 应分配给 expert i 的概率 $p_\delta = 1 - I_{\delta^2}(1/2, (d-1)/2)$，其中 $\delta = \cos(\theta)$ 为 token 与 gating weight 夹角余弦
- 当 d 很大且 $\delta = \Theta(1/\sqrt{d})$ 时，$p_\delta \approx 0.3$
- 当 $\delta$ 增大（token 与 expert 更匹配，夹角变小），$p_\delta$ 快速衰减至 0——仅少量 token 为 class-discriminative

由此得到 expert capacity 下界：

$$ec_{min} \ge \frac{1}{n \cdot \operatorname{erfc}(\sqrt{\frac{\delta^2 d}{2 - \delta^2}})} > \frac{1}{n} \exp(\frac{\delta^2 d}{2 - \delta^2})$$

实验测得 δ ≈ 0.03，可据此计算安全的 expert capacity 下界，在保证模型精度前提下降低 capacity 以减少冗余计算。

**Llama 3 Meets MoE 对 Capacity Factor 的扩展分析**：

论文通过 CF∈{1, 2, 4, Dropless} 消融实验（Table 4）量化了 CF 对 training MFU 和 downstream accuracy 的 trade-off：
- CF=1: MFU=46.8%（最高），MMLU 0-shot=63.7
- CF=2: MFU=39.2%，MMLU 0-shot=64.0（最高 accuracy）
- CF=4: MFU=39.4%，MMLU 0-shot=63.5（最终主实验选择）
- Dropless (CF=∞): MFU=39.6%，MMLU 0-shot=63.3

关键发现：(1) CF 隐式引入了正则化——Dropless 的 MMLU 准确率反而不及 CF=2/4，因为缺少 token dropping 的正则化效果；(2) CF=1 时 MFU 显著高于高 CF，因为更少 token 被处理且内存 footprint 更小，允许更小的模型并行度；(3) CF=4 为 accuracy-MFU 最佳平衡点。

训练配置补充：主实验使用 100B tokens, bfloat16, 512 H100 GPU, CF=4。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FO-MoE（Fairness-Oriented Mixture of Experts）是 Fair-MoE 论文提出的面向公平性的 MoE 架构变体，专为医疗 Vision-Language Model 设计，用于过滤偏置 patch embedding 并提取公平的任务相关特征。FO-MoE 包含两级 MoE：

1. **Embedding-based MoE**：替换图像和文本 encoder 最后一个 attention block 中的 MLP 层。输入为所有 patch embeddings I^1 ∈ R^(N+1)×D（N 个 patch + 1 个 [CLS] token）。Gate 输出 W^1 = softmax(G^1(I^1))，然后通过两级稀疏化：Ŵ^1 = Top_c(Top_r(W^1, k^1), α)。Top_r 保留每行（每个 patch）权重最高的 k^1 个 expert；Top_c 通过 expert capacity C 限制每列（每个 expert）可处理的 patch 数，仅保留 α = C(N+1)k^1/M^1 个最高权重。被清零的权重对应的 expert 输出被丢弃，实现**偏置 patch 的主动过滤**——包含敏感属性信息（如肤色、性别特征）的 patch 对应的 expert 输出权重被清零。

2. **Feature-based MoE**：放置在 encoder 之后，取 [CLS] token 对应的特征向量 I^2_0 ∈ R^D 作为输入，通过 M^2 个 experts 做进一步 sparse gating：W^2 = Top_r(softmax(G^2(I^2_0)), k^2)。最终公平特征 I^3 = Σ_{b=0}^{M^2-1} Ŵ^2_b · E^2_b(I^2_0)。

Expert 结构为标准两层 MLP：E_b(x) = T̃_b · σ(W̃_b · x)，其中 σ 为激活函数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

在 Fair-MoE 的 CLIP-based pipeline 中，FO-MoE 的图像侧前向流程：

```
# 图像侧 FO-MoE 流程
I_image = ViT_patch_embed(fundus_image)       # (N+1)×D
I_enc = attention_blocks[0..K-2](I_image)     # 前 K-1 个 block

# === Embedding-based MoE (替换最后一个 attention block 的 MLP) ===
I^1 = I_enc                                    # 输入
W^1 = softmax(G^1(I^1))                       # Gate: R^{(N+1)×D} → R^{(N+1)×M^1}
Ŵ^1 = Top_r(W^1, k^1)                         # 保留每行 top-k^1 权重
Ŵ^1 = Top_c(Ŵ^1, α)                           # capacity filtering: 每列仅保留 α 个
# α = C·(N+1)·k^1 / M^1, C 为 expert capacity
I^2_a = Σ_{b=0}^{M^1-1} Ŵ^1_{a,b} · E^1_b(I^1_a)  # 加权聚合各 expert 输出

# === [CLS] token 作为特征向量 ===
I_feat = I^2_0                                # R^D

# === Feature-based MoE (encoder 之后) ===
W^2 = Top_r(softmax(G^2(I_feat)), k^2)        # R^{M^2}, 保留 top-k^2
I^3 = Σ_{b=0}^{M^2-1} Ŵ^2_b · E^2_b(I_feat)   # Fair image feature

# 文本侧对称执行相同流程 → T^3 (Fair text feature)

# 对比学习损失
similarity = cosine(I^3, T^3)
L = contrastive_loss(similarity) + FOL
```

**与标准 Sparse MoE 的关键区别**：
- 标准 MoE 的 Top_c 仅用于 load balancing（防止个别 expert 过载），Fair-MoE 的 Top_c 用于**公平性过滤**——通过 capacity 约束使偏置 patch 的 expert 权重被清零
- 两级 MoE 设计：patch 级（过滤空间偏置）+ feature 级（过滤语义偏置）
- 同时应用于图像和文本两侧 encoder

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FO-MoE 基于标准 CLIP (ViT-B/16 或 ViT-L/14) 架构修改。实现方式：在 PyTorch 中替换 CLIP encoder 最后一个 Transformer block 的 MLP 为 embedding-based MoE，并在 encoder 输出后插入 feature-based MoE。Gate 使用标准线性层 + Softmax，sparse 操作通过 mask tensor 实现（单 GPU 训练，无 EP 通信开销）。论文代码开源在 https://github.com/LinjieT/Fair-MoE-Medical-Fairness-Oriented-Mixture-of-Experts-in-Vision-Language-Models。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
