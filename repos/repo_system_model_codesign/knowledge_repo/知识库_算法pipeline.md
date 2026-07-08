# 知识库_算法pipeline

## Context Sparsity (Inference-Time Context Sparsity, 推理时上下文稀疏性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
推理时上下文稀疏性（Inference-Time Context Sparsity）是指在 LLM 推理（尤其是 decode 阶段）时，对注意力机制沿 context（序列长度）维度施加稀疏化：每个 query token 不再对所有历史 token 的 KV cache 计算 attention，而是仅选取其中少量（k 个，k << N）最相关的 token 参与 scaled dot-product attention 计算。稀疏度（sparsity ratio）定义为 S× = N/k，表示仅使用 1/S 比例的 context token。核心原理基于两点：(1) 理论上的 Embedding Bottleneck——当 hidden dimension d << context 长度 N 时，dense attention 的输出 V^T·a 在 attention simplex 上不是单射（多个不同 attention 分布映射到同一 hidden 表示），因此完全密集的 attention 在长上下文下本来就是不可能的；(2) 实证发现——新一代模型（尤其是 hybrid 架构和大规模模型）对推理时的 aggressive context 稀疏化表现出显著的鲁棒性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 LLM decode pipeline 中，context sparsity 作用于每个 decoder layer 的 attention 模块：
```
# 标准 dense decode (baseline)
def dense_attention_decode(Q, K_cache, V_cache):
    # Q: (Hq, d), K_cache: (N, Hkv, d), V_cache: (N, Hkv, d)
    scores = Q @ K_cache.transpose(-1,-2) / sqrt(d)  # (Hq, N)
    attn_weights = softmax(scores, dim=-1)              # (Hq, N)
    output = attn_weights @ V_cache                     # (Hq, d)
    return output

# 推理时 context sparsity decode
def sparse_attention_decode(Q, K_cache, V_cache, sparsity_ratio, indexer):
    N = K_cache.shape[0]
    k = N // sparsity_ratio                            # 选中的 token 数

    # Step 1: 构建基础 mask (sink + local window)
    mask_indices = set(range(0, 128))                   # Sink tokens
    mask_indices |= set(range(pos-128, pos))            # Local window

    # Step 2: 索引器选择 top-k
    topk_indices = indexer.select_topk(Q, K_cache, k - len(mask_indices))
    mask_indices |= set(topk_indices)

    # Step 3: 仅对选中 token 计算 attention
    K_sparse = K_cache[list(mask_indices)]             # (k', Hkv, d)
    V_sparse = V_cache[list(mask_indices)]             # (k', Hkv, d)
    scores = Q @ K_sparse.transpose(-1,-2) / sqrt(d)   # (Hq, k')
    attn_weights = softmax(scores, dim=-1)
    output = attn_weights @ V_sparse                    # (Hq, d)
    return output
```
关键设计选择：
- 稀疏 mask 由三部分 additive 组合：sink tokens（前 128 个固定保留）+ local window（当前 token 附近 128 个）+ 索引器选出的 top-k。
- 稀疏模式为 per-token、per-query、per-head 级别——每个 query head 可以有不同的选中 token 集合，不强制块结构（block sparsity）。
- 稀疏度选择：5×（20% tokens）、50×（2%）、100×（1%）、500×（0.2%）等。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现分为两层：(1) 索引器层——用于快速从完整 KV cache 中选出 top-k 相关 token，可用方法包括 Oracle top-k（精确但昂贵，仅用于上界研究）、vAttention 随机采样、Double Sparsity 量化索引器、HashAttention 语义哈希、PQCache 乘积量化等。(2) Kernel 层——稀疏 decode kernel 根据索引从 paged KV cache 中 gather 选中 token 并执行分块 attention。开源实现：https://github.com/skylight-org/sparse-attention-hub（基于 HuggingFace Transformers + Ray Tune，Apache 2.0）。使用场景：长上下文 LLM serving（>32K tokens）、agentic 工作负载（多轮对话累积长历史）、长文档 QA、仓库级代码理解等。实证表明 Qwen3.5-27B 在 50× 稀疏下 RULER-HARD 和 AIME2025 质量与 dense 持平，SWE-Bench 上仅差 ~2pp。

涉及论文标题：
- Inference Time Context Sparsity

## Oracle Top-K Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Oracle Top-K 稀疏注意力是一种理想化的稀疏 attention 上界方法：在 decode 的每一步，对每个 query head，计算其与所有 KV cache token 的精确 attention score（即完整的 QK^T），然后仅保留 score 最高的 k 个 token 参与后续的 softmax 和加权求和。称为 "Oracle" 是因为它使用精确（而非近似的）attention score 来做出选择决策，因此代表了任何基于 top-k 选择的稀疏化方法在该稀疏度下的理论上界。它的作用不是作为实际部署方案（因为仍需计算完整 QK^T），而是作为实验对照，消除近似索引器的质量混淆因素，从而纯粹评估"模型在仅看到 k 个最相关 token 时能多好地工作"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def oracle_topk_attention(Q, K_cache, V_cache, k):
    # Q: (Hq, d), shape of K_cache: (N, Hkv, d)
    N = K_cache.shape[0]

    # Step 1: 计算完整 attention score（这是 "oracle" 的代价）
    full_scores = Q @ K_cache.transpose(-1,-2) / sqrt(d)  # (Hq, N)

    # Step 2: 选 top-k 索引
    _, topk_indices = torch.topk(full_scores, k, dim=-1)   # (Hq, k)

    # Step 3: 仅对选中的 token 计算完整 softmax + 加权求和
    K_sparse = K_cache[topk_indices]                       # (Hq, k, d) after gather
    V_sparse = V_cache[topk_indices]
    scores_sparse = (Q.unsqueeze(1) * K_sparse).sum(-1) / sqrt(d)
    attn_weights = softmax(scores_sparse, dim=-1)           # (Hq, k)
    output = (attn_weights.unsqueeze(-1) * V_sparse).sum(-2)  # (Hq, d)
    return output
```
Oracle top-k 的核心限制：虽然最终仅对 k 个 token 做 attention，但 Step 1 仍需计算完整 QK^T（O(N·d) 计算和 O(N·d) 内存读取），因此不节省计算或带宽。它仅作为评估稀疏化对模型质量影响的上界工具。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 "Inference Time Context Sparsity" 论文中，Oracle top-k 被用作所有稀疏化实验的主要评估机制，以消除近似索引器的混淆因素。论文在 RULER-HARD、LOFT、AIME2025 和 SWE-Bench 上使用 Oracle top-k 评估了 20 个模型在多种稀疏度下的表现。关键发现：大模型（如 Qwen2.5-72B）和 hybrid 架构（如 Qwen3.5-27B、Gemma3-27B）在 50× 稀疏下 oracle top-k 质量与 dense 接近甚至持平；小标准模型（如 Qwen2.5-1.5B）的 oracle top-k 退化严重，但可用 vAttention 随机索引恢复。

涉及论文标题：
- Inference Time Context Sparsity

## vAttention (Verified Sparse Attention via Sampling, 基于采样的验证稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
vAttention 是一种基于随机采样的稀疏注意力方法，发表于 ICLR 2026。其核心思想是：不对 attention 使用确定性的 top-k 选择（后者在小模型上因 attention 分数分散而退化），而是将每个 KV cache token 的选中概率设为与其 attention score 成比例，然后通过随机采样选出 k 个 token 参与 attention 计算。由于采样是无偏的，整个稀疏 attention 在期望意义上等价于 dense attention（经过重要性加权校正）。"Verified" 指的是可以通过统计检验验证稀疏化后的 attention output 与 dense output 的误差在可控范围内。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
def vattention_sparse(Q, K_cache, V_cache, k):
    N = K_cache.shape[0]

    # Step 1: 计算采样概率（与 attention score 成比例）
    scores = Q @ K_cache.transpose(-1,-2) / sqrt(d)  # (Hq, N)
    probs = softmax(scores, dim=-1)                    # (Hq, N)

    # Step 2: 按概率随机采样 k 个 token（不放回）
    sampled_indices = torch.multinomial(probs, k, replacement=False)  # (Hq, k)

    # Step 3: 对采样到的 token 计算 attention，并用采样概率做重要性加权
    K_sampled = K_cache[sampled_indices]               # (Hq, k, d)
    V_sampled = V_cache[sampled_indices]
    sampled_probs = probs.gather(-1, sampled_indices)  # (Hq, k)

    scores_sparse = (Q.unsqueeze(1) * K_sampled).sum(-1) / sqrt(d)
    # 重要性加权校正
    weights = softmax(scores_sparse - log(sampled_probs), dim=-1)
    output = (weights.unsqueeze(-1) * V_sampled).sum(-2)
    return output
```
关键特性：因为是随机采样而非确定性 top-k，vAttention 避免了 attention 分数均匀分散时的退化问题——即使没有明显的高分 token，采样仍能无偏地代表完整分布。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 "Inference Time Context Sparsity" 论文中，vAttention 被用作 oracle top-k 的对比方法。在 RULER-HARD-32K 上，当小标准模型（如 Qwen2.5-1.5B）的 oracle top-k 在 50× 稀疏下退化到 dense 性能的 ~63% 时，vAttention 能几乎恢复到 dense parity（~100%）。论文认为这表明"稀疏化的核心限制不在于稀疏本身，而在于确定性选择机制的局限性"。vAttention 的实现包含在 https://github.com/skylight-org/sparse-attention-hub 中。

涉及论文标题：
- Inference Time Context Sparsity

## Double Sparsity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Double Sparsity 是一种训练后稀疏注意力方法（Post-Training Sparse Attention），发表于 arXiv 2024。其核心思想是使用双重稀疏化：(1) 第一层稀疏——对 query 和 key 向量进行量化压缩（使用少量比特通道，如 8 通道 16-bit 精度），在量化空间中快速计算近似的 query-key 匹配分数；(2) 第二层稀疏——基于近似分数选出 top-k 个 token，然后在完整精度下仅对这 k 个 token 计算精确 attention。这种"粗筛+精选"的两阶段设计使得索引器本身的开销极低，适合在线 decode 场景。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Double Sparsity 索引器 (8-channel, 16-bit precision)
def double_sparsity_indexer(Q, K_cache, num_channels=8):
    # Q: (Hq, d), K_cache: (N, Hkv, d)
    # Step 1: 量化为低精度通道
    Q_quant = quantize_to_channels(Q, num_channels, bits=16)  # (Hq, num_channels)
    K_quant = quantize_to_channels(K_cache, num_channels, bits=16)  # (N, Hkv, num_channels)

    # Step 2: 在量化空间中计算近似分数
    # 使用低精度内积，计算和带宽开销极低
    approx_scores = einsum('hd,nkd->hnk', Q_quant, K_quant).sum(-1)  # (Hq, N)
    # 由于量化通道数少、精度低，该操作极快

    # Step 3: 基于近似分数选出 top-k
    _, topk_indices = torch.topk(approx_scores, k, dim=-1)

    # Step 4: 在完整精度下计算精确 attention（仅 k 个 token）
    # ...（标准稀疏 attention 计算）
    return topk_indices
```
在 "Inference Time Context Sparsity" 论文的 kernel benchmark（Table 2）中，Double Sparsity 被用作索引器的代表，模拟 8 通道 16-bit 精度配置。加入索引器开销后，MHA 下 2× 稀疏度即 break-even（超过 FlashInfer dense baseline），100× 稀疏度达 4.17× 加速；GQA 下 10-20× 稀疏度 break-even。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Double Sparsity 在论文中的使用场景是作为实际可部署的在线稀疏索引器，替代昂贵的 oracle top-k。其量化通道设计使其可以高效地在 GPU 上实现——低精度内积的计算和带宽开销远低于完整精度 QK^T。更轻量的变体（如 HashAttention、PQCache 或更低精度 Double Sparsity）预期进一步降低索引器开销，扩大稀疏化的净收益。

涉及论文标题：
- Inference Time Context Sparsity

## Attention Sink (注意力沉池 / Sink Token)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Attention Sink（注意力沉池）是 LLM 注意力机制中观察到的一个现象：序列开头的几个初始 token（通常前 4-128 个）会不成比例地吸引大量注意力分数，即使这些 token 在语义上并不重要。这一现象最早在 "Efficient Streaming Language Models with Attention Sinks" 等工作中被系统研究。由于这些初始 token 充当了注意力分数的"倾泻池"（sink），在稀疏注意力中必须始终保留它们作为 mask 的一部分，否则 softmax 的归一化分母会被扭曲，导致模型质量急剧下降。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 context sparsity 的 mask 构建中，sink tokens 是 additive mask 的第一个组件：
```
def build_sparse_mask(N, pos, k, sink_size=128, local_window=128):
    mask = set()

    # Component 1: Sink tokens —— 前 sink_size 个 token 永远保留
    mask |= set(range(0, sink_size))          # indices 0..127

    # Component 2: Local window —— 当前 token 附近的 local_window 个 token
    mask |= set(range(max(0, pos - local_window), pos))

    # Component 3: Top-k 稀疏选择 —— 索引器选出剩余 token
    remaining_budget = k - len(mask)
    topk_from_rest = indexer.select_topk_excluding(Q, K_cache, mask, remaining_budget)
    mask |= set(topk_from_rest)

    return sorted(mask)
```
在 "Inference Time Context Sparsity" 论文中，sink token 数量固定为 128。即使极端稀疏度下（如仅用 16-32 个 token），128 个 sink token 也始终保留。论文 Figure 2(c) 展示了 Qwen3.5 家族在极端 top-K（K ∈ {1, 4, 16, 64, 128}，sink=64 固定）下的饱和曲线——更大模型更早达到 dense parity。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Attention sink 不需要训练或修改模型——它是对注意力分数分布的经验观察的直接应用。实现时只需在稀疏 mask 构建阶段固定保留前几个 token。sink token 数量是一个超参数：过少可能导致 softmax 归一化不稳定，过多则浪费稀疏预算。常见设置是 4-128 个。

涉及论文标题：
- Inference Time Context Sparsity

## Hybrid Architecture LLM (混合架构大语言模型, Linear Attention + SDPA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
混合架构 LLM 是指将标准 scaled dot-product attention (SDPA) 层与线性注意力（Linear Attention）层或状态空间模型（SSM）层在同一个 Transformer 模型中交替使用的架构设计。代表模型包括 Qwen3.5（线性注意力 + SDPA 交替）和 Gemma3（滑动窗口注意力 + SDPA 交替）。设计动机：纯 SDPA 模型在长上下文下受 O(N²) 计算和 O(N) KV cache 内存瓶颈限制；纯线性/SSM 模型虽然高效但表达力受限，在复杂推理任务上不及 SDPA。混合架构试图取两者之长——线性/SSM 层提供高效的长程上下文处理，SDPA 层保留精确的 token 级交互能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Qwen3.5 风格的 Hybrid Decoder Layer 交替
def hybrid_decoder_layer(x, layer_idx):
    if layer_idx % 2 == 0:
        # 偶数层: Linear Attention (O(N·d²), 无 KV cache, 常数内存)
        output = linear_attention(x)   # 维护固定大小的循环状态 S ∈ R^{d×d}
    else:
        # 奇数层: 标准 SDPA (O(N·d), KV cache 增长)
        output = sdpa_attention(x)     # 标准 softmax(QK^T/√d)V
    output = ffn(layernorm(output)) + output
    return output
```
"上下文稀疏性"论文的关键发现：hybrid 架构模型（Qwen3.5、Gemma3）对推理时 context 稀疏化的鲁棒性显著优于纯 SDPA 模型。在 RULER-HARD 上，hybrid 模型在 50× 稀疏下性能几乎不变，且这种鲁棒性与模型规模几乎无关（小 hybrid 模型同样鲁棒，而小标准模型退化严重）。论文推测：线性/SSM 层的存在使模型学会将长程信息编码到循环状态中，减少了对密集 KV cache 访问的依赖。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
当前主流 hybrid 架构实现：
- Qwen3.5 (2026): 0.8B/2B/4B/9B/27B，线性注意力 + SDPA 交替
- Gemma3 (2025): 1B/4B/12B/27B，滑动窗口 + SDPA
- GLM-5 (2026): 同样采用 hybrid 设计

在 serving 场景下，hybrid 架构天然适合与 context sparsity 结合——线性层无需稀疏化（已是 O(1) 内存），SDPA 层施加 aggressive 稀疏化。这解释了为何 hybrid 模型能在极高稀疏度下保持质量。

涉及论文标题：
- Inference Time Context Sparsity

## Embedding Bottleneck in Attention (注意力嵌入瓶颈)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
注意力嵌入瓶颈（Embedding Bottleneck）是 "Inference Time Context Sparsity" 论文提出的一个理论概念，由 Theorem 1 和 Corollary 2 形式化。核心论证：当 hidden dimension d 远小于 context 长度 N（这是所有实际 LLM 的常态，如 d=128 per head, N 可达 128K+），dense attention 输出 o = V^T·a 的映射 a → V^T·a 在 attention simplex 上不是单射（injective）。即存在两个不同的 attention 分布 a ≠ a'，但产生相同的 attention output。这意味着即使模型"看到"了所有 N 个 token，d 维 hidden 向量也无法编码和传递 O(N) 个 attention score 中的所有细粒度变化。Corollary 2 指出：要无损保留所有可能的 dense attention 分布差异，需要 d ≥ N-1——对于百万 token 上下文需要百万维 hidden width，这在实践中不可能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Theorem 1 的构造性证明流程（伪代码）：
```
# 证明存在 a ≠ a' 且 V^T·a = V^T·a'（当 d < N-1）
# 输入: V ∈ R^{N×d}, d < N-1
# 输出: 两个不同的有效 attention 分布 a, a'

# Step 1: 构造 null space 与 zero-sum hyperplane 的交集
# N(T) = {z: V^T·z = 0}, dim(N(T)) ≥ N-d
# H = {z: 1^T·z = 0}, dim(H) = N-1
# dim(N(T) ∩ H) ≥ (N-d) + (N-1) - N = N-d-1 > 0

# Step 2: 选择非零 z ∈ N(T) ∩ H，缩放使 ||z||_∞ = 1

# Step 3: 构造两个有效的 attention 分布
a_0 = (1/N) * ones(N)                  # 均匀分布
a   = a_0 + (β/N) * z                  # 正扰动, β∈(0,1)
a'  = a_0 - (β/N) * z                  # 负扰动
# a, a' 均满足: sum=1, 所有元素>0, a≠a'
# 且 V^T·a = V^T·a_0 + (β/N)·V^T·z = V^T·a_0 = V^T·a'
# 因此 a ≠ a' 但产生相同 output
```
实际意义：这个理论结果说明 dense attention 在长上下文下不是"信息无损"的——d 维 hidden vector 本身就构成了信息瓶颈。因此稀疏 attention 并非信息的"损失"，而是显式选择了哪些 token 关系通过瓶颈，可能是比 dense attention 更优的信息路由方式。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Embedding bottleneck 不是需要"实现"的组件，而是对 attention 本质局限的理论刻画。其工程启示是：(1) 不应将 dense attention 视为"完美"的上界——它在长上下文下本身就有根本性信息损失；(2) 极端稀疏化应被视为原则性目标而非近似妥协；(3) 训练时应显式考虑 context sparsity，让模型学会在有限的注意预算下最大化信息路由效率。论文还引用了 Bhattamishra et al. (NeurIPS 2024) 的补充结果：有限精度循环模型（RNN/LSTM/SSM/线性注意力）需要 hidden state size Ω(N) 才能从 N-token 序列中恢复任意 token。

涉及论文标题：
- Inference Time Context Sparsity

## Einsum Notation in Sparse Tensor Algebra (稀疏张量代数中的爱因斯坦求和记号)

术语是什么？

Einstein Summation (Einsum) 是一种**用索引变量表示张量代数运算的数学记号**，在稀疏张量代数编译器中作为标准的输入/中间表示。Einsum 记号的约定是：重复出现的索引变量表示在该维度上进行归约（求和），单次出现的索引变量表示该维度的广播或保留。

例如（FuseFlow/SAM 上下文）：
- $T_i = B_{ij} C_j$：索引 $j$ 在 $B_{ij}$ 和 $C_j$ 中重复出现 → 对 $j$ 做内积（归约求和），$i$ 仅出现在 LHS 和 B 中 → 广播维度。
- $\forall_{ikj} T_{ij}^0 = A_{ik} X_{kj}$：$\forall_{ikj}$ 表示显式的 dataflow order（遍历顺序为 $i \to k \to j$），上标 $^0$ 表示张量的第一个版本。

在 FuseFlow 中，Einsum 表达式是编译器的**输入 IR**——PyTorch 模型通过 Torch-MLIR/MPACT 被 lowering 到 MLIR Linalg + SparseTensor dialects，每个张量操作被表示为 Einsum 表达式。稀疏格式信息（CSR、DCSR、COO 等）编码在 SparseTensor dialect 的 `sparse_tensor.encoding` 属性中。

从算法pipeline角度拆解术语：

以 GCN 2-layer 前向传播为例，说明 Einsum 如何表示整个 ML pipeline：

```
Layer 1:
  E1: H1'_{ij}  = A_{il} X_{lj}          // 稀疏-稠密 matmul (A CSR sparse adjacency, X dense features)
  E2: H1_{ik}   = H1'_{ij} W1_{jk}       // 稠密-稠密 matmul
  E3: O1_{ik}   = ReLU(H1_{ik})          // 逐元素非线性 (非 Einsum 操作，在 SAMML 中单独处理)

Layer 2:
  E4: H2'_{ij}  = A_{il} O1_{lj}         // 稀疏-稠密 matmul
  E5: H2_{ik}   = H2'_{ij} W2_{jk}       // 稠密-稠密 matmul
  E6: O2_{ik}   = softmax(H2_{ik})       // 分类输出

FuseFlow EKF 将每个 Fuse{} region 内的 Einsum 融合:
  Fuse{Layer 1}: E1 + E2 + E3 → 融合 subgraph (一个 Fused Einsum)
  Fuse{Layer 2}: E4 + E5 + E6 → 融合 subgraph (一个 Fused Einsum)
```

**Einsum 在融合中的角色**：Einsum 的索引变量直接对应 dataflow 的坐标处理。当 EKF 通过 index substitution 连接 producer 和 consumer 时：
- Producer 的输出索引变量被替换为 consumer 的输入索引变量
- 归约索引（如 $l$, $j$, $k$）需要被重命名以避免冲突
- POG 追踪所有索引变量的 outer-to-inner 关系

**与稠密编译器中的 Einsum 的区别**：在稠密编译器（如 XLA、TVM）中，Einsum 主要用于描述操作的数学语义，循环嵌套顺序由后续的 schedule 决定。在稀疏数据流编译器中，Einsum 的索引变量直接映射到坐标 streams——索引变量的遍历顺序决定了哪些坐标先被处理，直接影响 intersect/union 操作的效率和内存访问模式。

术语一般如何实现？如何使用？

Einsum 在 FuseFlow 中通过 MLIR Linalg dialect 的 `linalg.einsum` 操作表示。每个 `linalg.einsum` 包含：
- 索引表达式（如 `"ij, jk -> ik"`）
- 输入/输出张量
- 可选的 affine maps（指定 dataflow order）
- 稀疏编码属性（SparseTensor dialect）

类似地，TACO [Kjolstad et al., OOPSLA 2017] 使用张量索引记号（tensor index notation）作为输入，Custard [Hsu et al., ASPLOS 2023] 使用类似 Einsum 的记号。TeAAL [Nayak et al., MICRO 2023] 使用 cascaded Einsums 描述稀疏张量加速器。

涉及论文标题：
- FuseFlow

---

## Fibertree Representation (纤维树表示)

术语是什么？

Fibertree 是 SAM 和 FuseFlow 中用于**表示多维稀疏张量的层级嵌套数据结构**。Fibertree 将 n 阶张量分解为 n 个层级（level），每层包含：
- **pos 数组**：指向下一层级的起始位置（用于定位某一坐标的内部结构）
- **crd 数组**：该层级的坐标值
- **可选的 val 数组**：最内层的值

Fibertree 的关键思想：张量的坐标信息与值信息分离存储和传输，坐标用于稀疏迭代（intersect/union 判断），值仅在坐标匹配时才被获取（fetch）和计算。

从算法pipeline角度拆解术语：

CSR 矩阵 $B_{ij}$ 的 fibertree 表示：
```
Level 1 (i-dim, outer):
  pos: [0, 2, 3, 5]          // 第 i 行的非零元素从 Level 2 的哪个位置开始
  crd: [0, 1, 2]             // 行坐标 i = {0, 1, 2}
  (ref 指向 Level 2 的对应区间)

Level 2 (j-dim, inner):
  pos: [0, 2, 3, 5, 7]      // 第 j 列的位置
  crd: [0, 2, 1, 0, 2]      // 列坐标 j: row0→{0,2}, row1→{1}, row2→{0,2}
  
Value:
  val: [B[0,0], B[0,2], B[1,1], B[2,0], B[2,2]]

转译为 standard CSR:
  row_ptr = [0, 2, 3, 5]     // Level 1 pos
  col_idx = [0, 2, 1, 0, 2]  // Level 2 crd
  values  = [B[0,0], B[0,2], B[1,1], B[2,0], B[2,2]]

在 SAM dataflow 中:
  LS(B_i) 输出 crd stream: {0, 1, 2}, ref stream: {ptr→row0, ptr→row1, ptr→row2}
  LS(B_j) 接收 ref stream, 输出 crd stream: {0, 2, 1, 0, 2} (对齐 ref 确定的区间)
  Val(B)  输出 val stream: {B[0,0], B[0,2], B[1,1], B[2,0], B[2,2]}
```

Fibertree 的通用性：通过组合不同的 per-level 存储格式（dense、compressed、singleton），fibertree 可以表示任意稀疏格式：
- **Dense level**：固定长度的连续坐标，无 pos 数组（直接计算偏移）
- **Compressed level**：变长区间，需要 pos 数组定位
- **Coordinate level**（用于 COO 格式）：每个坐标单独存储

FuseFlow 的 sparsity-source agnostic 设计（Section 4.1）直接源自 fibertree 的格式抽象能力——无论稀疏来自 graph adjacency（lossless）还是 weight pruning（lossy），只要格式在编译前确定，fibertree 表示是相同的。

术语一般如何实现？如何使用？

Fibertree 最初由 Sze et al. [Wu et al., 2022] 在 SparseLoop 分析模型中形式化，SAM [Hsu et al., ASPLOS 2023] 将其作为核心张量表示。在 FuseFlow 中，fibertree 体现在：
- MLIR SparseTensor dialect 的 `sparse_tensor.encoding` 指定 per-level 格式
- SAMML 图中 LS primitives 按 fibertree 层级产生 crd/ref/val streams
- Comal 模拟器的 token 生成逻辑基于 fibertree 的层级遍历

涉及论文标题：
- FuseFlow

---

## Quantization-Aware Training (QAT)

术语是什么？

量化感知训练（Quantization-Aware Training, QAT）是一种在训练过程中模拟量化效应的神经网络训练方法。与训练后量化（Post-Training Quantization, PTQ）不同，QAT 在训练的前向传播中使用量化后的权重和/或激活值 `Q(x)` 计算损失 `f(Q(x))`，但在反向传播中通过 Straight-Through Estimator (STE) 将梯度从量化值传递回全精度 master weight `x`，使得模型在训练过程中"感知"并适应量化引入的精度损失。QAT 的优化目标是在量化约束下最小化任务损失，即 `min_x f(Q(x))`，其中 `Q` 是量化算子（将连续值映射到离散量化网格），`x` 是浮点 master weight。CAGE 论文将 QAT 重新定义为多目标优化问题，其 Pareto 最优条件为 `∇f(x*) + λ(x* - Q(x*)) = 0`。

从算法pipeline角度拆解术语：

QAT 在算法 pipeline 中的执行流程如下：

```
# QAT 训练一次迭代的伪代码
输入: master权重 x (FP32), 量化器 Q, 优化器 Opt, 输入数据 batch

# 1. 量化权重
x_q = Q(x)                    # 将FP32权重映射到INT4/INT8等离散值

# 2. 量化前向传播
y = forward(batch, x_q)       # 使用量化权重计算输出
loss = criterion(y, target)   # 计算损失

# 3. 反向传播 (STE)
grad_x_q = ∇loss              # 损失对量化权重的梯度
grad_x = grad_x_q             # STE: 直通，∂Q(x)/∂x ≈ I
                              # CAGE改进: grad_x = grad_x_q + λ*(x - x_q)

# 4. 更新master权重
x = Opt.update(x, grad_x)     # 使用全精度梯度更新FP32 master权重
```

关键特征：
- Master weight `x` 始终保持 FP32 精度，仅在每次前向传播时被量化
- STE 是 QAT 的核心梯度估计技术：反向传播时将量化器视为恒等映射
- QAT 比 PTQ 精度更高但训练开销更大，因为需要完整的训练流程
- CAGE 扩展了 STE，增加了曲率感知的量化误差修正项

术语一般如何实现？如何使用？

- **PyTorch 实现**：通过自定义 `autograd.Function` 实现量化算子的前向（实际量化）和反向（STE 直通梯度），或在 `torch.ao.quantization` 中使用内置 QAT 模块
- **典型量化器**：均匀量化（INT4/INT8）、NF4（NormalFloat4）、QuEST（带 Hadamard 变换的 INT 量化）、GPTQ 等
- **使用场景**：(1) LLM 压缩部署：将 8B-70B 模型量化为 W4A16 或 W4A4 以减少内存和加速推理；(2) 边缘设备部署：在资源受限设备上运行量化模型；(3) 预训练量化：从头训练量化模型（如 CAGE 在 Llama 系列上的 W3A3/W4A4 预训练）
- **CAGE 开源实现**：https://github.com/IST-DASLab/CAGE，基于 PyTorch 2.6 + CUDA 12.6，通过 `--w-bits` 和 `--a-bits` 控制量化位宽

涉及论文标题：
- CAGE: Curvature-Aware Gradient Estimation for Quantization-Aware Training

---


**SLA2 论文的 QAT 用法（低比特 Attention QAT）**：

SLA2 将 QAT 应用于 attention 计算的量化加速，方式独特：
- **前向传播**：对 Q、K、P（softmax 后概率）、V 四个张量执行 INT8/FP8 量化再反量化，量化方案遵循 SageAttention2++：`Q̂, s_Q = quant(Q); K̂, s_K = quant(K)` → `S = dequant(Q̂K̂ᵀ/√d, s_Q, s_K)` → `P = softmax(S⊙M)` → `P̂, s_P = quant(P); V̂, s_V = quant(V)` → `O_s = dequant(P̂V̂, s_P, s_V)`。这意味着整个 attention 的矩阵乘法在低比特精度下进行（Tensor Core 加速）。
- **反向传播**：全程 FP16 精度，使用原始 FP16 的 Q、K、V 和前向输出 O_s 计算梯度 `dQ, dK, dV = backward(dO_s, O_s, Q, K, V)`。这与传统 QAT 不同——传统 QAT 反向也使用 STE 通过量化器传梯度，而 SLA2 的反向完全不涉及量化，利用的是 SageAttention2++ 的数学等价性（低比特 forward + FP16 backward 在精度上可接受）。
- **训练流程**：在 Stage 2 端到端微调扩散模型时启用 QAT 前向。Ablation 显示去掉 QAT 后生成质量明显下降，验证了 QAT 对低比特 attention 的必要性。量化带来约 1.3× kernel 加速。


- SLA2: Sparse-Linear Attention with Learnable Routing and QAT

## Straight-Through Estimator (STE)

术语是什么？

Straight-Through Estimator (STE) 是量化感知训练（QAT）中用于处理量化算子不可微问题的一种梯度估计技术。量化算子 `Q: R → {q_1, q_2, ..., q_k}` 将连续值映射到离散量化网格，其导数在几乎所有位置为零（阶跃函数），在量化边界处未定义。STE 通过将量化器的雅可比矩阵近似为单位矩阵来解决这个问题：`∂Q(x)/∂x ≈ I`，即梯度"直通"量化算子传播到 master weight。数学上，STE 定义 `∂L/∂x = ∂L/∂Q(x) · I = ∂L/∂Q(x)`，即量化输入的梯度直接作为量化前参数的梯度。这使得基于梯度的优化器（如 AdamW、SGD）能够正常更新浮点 master weight。

从算法pipeline角度拆解术语：

STE 在 QAT 训练迭代中的计算流程：

```
# STE 的前向和反向传播
前向传播:
  x_q = round(x / scale) * scale    # 量化: FP32 → INTk → FP32 (反量化)
  loss = f(x_q)                      # 使用量化值计算损失

反向传播 (STE):
  ∂loss/∂x_q = autograd(loss)        # 正常自动微分得到对x_q的梯度
  ∂loss/∂x = ∂loss/∂x_q * ∂x_q/∂x   # 链式法则
            ≈ ∂loss/∂x_q * I         # STE: ∂x_q/∂x ≈ I (忽略量化舍入的梯度)
            = ∂loss/∂x_q             # 梯度直接穿过量化算子

  # CAGE 改进: ∂loss/∂x = ∂loss/∂x_q + λ*(x - x_q)
  # 增加了量化误差作为曲率修正信号
```

STE 的局限性：
- 忽略了量化操作引起的局部曲率变化：当 `x - Q(x)` 较大时，`f(Q(x))` 在 x 处的梯度可能与直通梯度有显著偏差
- 梯度估计存在偏差：STE 是有偏估计，因为 `∂Q(x)/∂x` 在大多数点实际为 0
- 无法显式优化量化误差：标准 STE 仅优化 `f(Q(x))`，不控制 `||x - Q(x)||`

术语一般如何实现？如何使用？

- **PyTorch 实现**：自定义 `torch.autograd.Function`，在 `forward` 中执行实际量化/反量化，在 `backward` 中返回输入梯度（identity gradient）
- **变体**：Clipped STE（将梯度截断在量化范围内）、Scaled STE（按量化步长缩放梯度）
- **CAGE 扩展**：在 STE 梯度上增加 `λ_t * (x_t - Q(x_t))` 修正项，实现曲率感知
- **使用场景**：所有 QAT 方法的基础组件；也用于二值化神经网络（BNN）、离散表示学习、Gumbel-Softmax 的替代方案

涉及论文标题：
- CAGE: Curvature-Aware Gradient Estimation for Quantization-Aware Training

---

## CAGE (Curvature-Aware Gradient Estimation)

术语是什么？

CAGE（Curvature-Aware Gradient Estimation，曲率感知梯度估计）是 ISTA DASLab 在 MLSys 2025 提出的 QAT 梯度估计方法。CAGE 将 QAT 重新框架化为多目标优化问题：同时最小化任务损失 `f(Q(x))` 和量化误差 `||x - Q(x)||`。其核心创新是在 STE 梯度基础上增加一个曲率感知的修正项：`g_t = ∇̃f(x_t) + λ_t · (x_t - Q(x_t))`，其中 `λ_t` 是动态调节系数，`(x_t - Q(x_t))` 是量化误差（无梯度，作为纯信号而非可微项）。CAGE 理论保证收敛到 Pareto 最优状态：`∇f(x*) + λ(x* - Q(x*)) = 0`，收敛速度为 `O(1/√T)`。

CAGE 的两个关键机制：
1. **Silence Period（沉默期）**：训练前 `s` 比例步数中 `λ_t = 0`（等同于标准 STE QAT），让模型先学会好的表示，之后再逐步引入量化约束
2. **Linear Ramp-up**：沉默期后 `λ_t = λ · (r_t - s)/(1 - s)`，线性增加到最终值 λ
3. **Coupled vs Decoupled**：Coupled CAGE 将修正加到梯度上（`g_t = ∇̃f + λ_t·e_t`），由优化器二阶矩归一化后生效；Decoupled CAGE 将修正直接加到 AdamW 的更新量上（`Δ_t = AdamW_update + λ_t·e_t`），避免二阶矩削弱修正信号

CAGE 是 **optimizer-agnostic**（可配合 AdamW、Muon、Shampoo 等任何优化器）和 **quantizer-agnostic**（可配合任何量化器），且计算开销极低（≈ 一次逐元素减法 + 标量乘法，per-iteration overhead 几乎为零）。

从算法pipeline角度拆解术语：

```
# CAGE (Decoupled) + AdamW 完整算法
输入: x_0 (FP32 master权重), T (总步数), Q (量化器), λ (CAGE系数), s (沉默比例)
     α (学习率), β1,β2 (AdamW动量系数), ω (weight decay)
初始化: m=0, v=0

for t = 0, 1, ..., T-1:
    r_t = (t+1)/T                           # 训练进度
    
    # Step 1: 动态λ调度
    if r_t <= s:
        λ_t = 0                              # 沉默期
    else:
        λ_t = λ * (r_t - s) / (1 - s)        # 线性ramp-up
    
    # Step 2: 量化前向传播
    x_q = Q(x_t)                              # 量化权重
    loss = f(x_q)                             # 前向计算损失
    
    # Step 3: 反向传播 + CAGE修正
    g_ste = ∇̃f(x_t)                          # STE梯度
    e_t = x_t - x_q                          # 量化误差（无梯度）
    g_t = g_ste + λ_t * e_t                  # Coupled: 修正加到梯度
    
    # Step 4: Decoupled weight decay
    x_t = (1 - α*ω) * x_t
    
    # Step 5: AdamW更新
    m = β1*m + (1-β1)*g_t
    v = β2*v + (1-β2)*g_t²
    m_hat = m / (1-β1^t)
    v_hat = v / (1-β2^t)
    
    # Step 6: Decoupled CAGE + 参数更新
    Δ_t = m_hat / (√v_hat + ε) + λ_t * e_t   # Decoupled: 修正直接作用于Δ
    x_{t+1} = x_t - α * Δ_t
```

关键实验结果：
- W3A3 CAGE 训练的 validation loss 低于 QuEST W4A4
- Llama 3.1-8B W4A16 fine-tuning: CAGE RULER 73.2 vs QuEST 68.7 vs GPTQ 65.1
- 将 QAT fine-tuning 的压缩精度损失减半

术语一般如何实现？如何使用？

- **开源**：https://github.com/IST-DASLab/CAGE (MIT License)
- **依赖**：Python 3.11, CUDA 12.6, PyTorch 2.6
- **核心超参**：`--cage-lambda`（默认10，修正强度）、`--cage-silence-ratio`（默认0.8，前80%步数不施加修正）、`--cage-schedule`（linear_ramp 或 constant）
- **使用**：`bash train.sh --model-size-prefix=200M --w-bits=4 --a-bits=4 --cage-lambda=15 --cage-silence-ratio=0.7`
- **适用场景**：LLM QAT 预训练（30M-3.2B Llama 模型）、QAT fine-tuning（Llama 3.1-8B, Llama-3.2-3B）
- **硬件**：单张 NVIDIA H100 GPU
- **量化配置**：支持 W4A4, W3A3, W4A16, W8A4, W2A2 等任意位宽组合

涉及论文标题：
- CAGE: Curvature-Aware Gradient Estimation for Quantization-Aware Training

---

## QuEST (Quantization with Hadamard Transform)

术语是什么？

QuEST（Quantization with Stable Training）是 ISTA DASLab 在 ICML 2025 提出的 LLM 低精度 QAT 方法，支持 1-bit 权重和激活的稳定训练。QuEST 的核心贡献是引入 Hadamard 变换对权重和激活进行预处理，消除异常值（outlier）对量化过程的干扰，使得极低精度（如 W1A1 或 W2A2）下的 QAT 训练能够稳定收敛。在 CAGE 论文中，QuEST 作为 baseline 量化器被使用（具体为其 INT 变体，即 QuEST INT），CAGE 在其基础上进一步改进了梯度估计。

从算法pipeline角度拆解术语：

QuEST 在 QAT pipeline 中的角色是**量化算子 Q** 的实现。其核心流程：

```
# QuEST 量化器的一次前向传播
输入: x (FP32 master权重矩阵), bit_width (目标位宽)

# Step 1: Hadamard 变换（消除outlier）
x_transformed = H @ x @ H            # 对权重矩阵做双边Hadamard变换
                                     # H 是 Hadamard 矩阵，将outlier能量分散到所有维度

# Step 2: 均匀量化
scale = max(|x_transformed|) / (2^{bit_width-1} - 1)
x_q_transformed = round(x_transformed / scale) * scale

# Step 3: 逆Hadamard变换（还原到原始空间）
x_q = H^T @ x_q_transformed @ H^T     # Hadamard矩阵是对称正交矩阵，H=H^T

返回: x_q (量化+反量化后的权重，用于前向传播)
```

在 CAGE 中，QuEST 作为 `Q` 算子被调用：`e_t = x_t - QuEST(x_t)` 计算量化误差，用于 CAGE 修正。

术语一般如何实现？如何使用？

- **开源**：QuEST 论文开源代码（与 CAGE 同一实验室，IST-DASLab）
- **作用**：作为 QAT 的量化算子，替代标准均匀量化或 NF4 量化
- **关键优势**：Hadamard 变换使量化对 outlier 不敏感，支持 1-bit 极低精度训练
- **在 CAGE 中的使用**：`--quantizer=quest` 或作为默认量化器，CAGE 的修正项 `λ_t * (x_t - QuEST(x_t))` 不依赖具体量化器实现
- **计算开销**：Hadamard 变换可通过 Fast Walsh-Hadamard Transform (FWHT) 高效实现，复杂度 O(d log d)

涉及论文标题：
- CAGE: Curvature-Aware Gradient Estimation for Quantization-Aware Training

---

## Multi-Objective Optimization for QAT

术语是什么？

多目标优化视角下的 QAT 是 CAGE 论文提出的理论框架。传统 QAT 仅优化单一目标 `min_x f(Q(x))`（最小化量化后模型的任务损失）。CAGE 将其重新定义为双目标优化问题：同时最小化 (1) 任务损失 `f(Q(x))` 和 (2) 量化误差 `||x - Q(x)||`。在这个框架下，Pareto 最优状态满足一阶条件：`∇f(x*) + λ(x* - Q(x*)) = 0`，其中 λ 是平衡两个目标的权重。CAGE 的梯度修正项 `λ_t · (x_t - Q(x_t))` 正是对这个 Pareto 最优条件的随机梯度近似。论文给出了非凸光滑假设下的收敛保证：`E[||∇_{λP} f(Q(x̂_T))||²] = O(1/√T)`。

从算法pipeline角度拆解术语：

多目标优化在 CAGE 训练中的具体体现：

```
# QAT 的 Pareto 前沿与 CAGE 的优化轨迹
优化空间中的两个目标:
  L1(x) = f(Q(x))           # 任务损失（越小越好）
  L2(x) = ||x - Q(x)||²     # 量化误差（越小越好）

# 传统 STE QAT: 仅在L1方向上优化
x_{t+1} = x_t - α * ∇̃f(Q(x_t))     # 仅沿任务损失梯度方向移动
→ 可能停留在L2较大的区域（量化误差未受控）

# CAGE: 在L1和L2的Pareto前沿上优化
x_{t+1} = x_t - α * (∇̃f(Q(x_t)) + λ_t * (x_t - Q(x_t)))
→ λ_t*(x_t-Q(x_t)) 是 ∂L2/∂x = 2(x_t - Q(x_t)) 的方向
→ 优化器同时沿减小任务损失和减小量化误差的方向移动
→ 收敛到 Pareto 最优：∇f(x*) + λ(x* - Q(x*)) = 0
```

Silence Period 的作用也可以从多目标优化角度理解：在训练早期（前 s 比例步数），`λ_t = 0`，优化器仅最小化任务损失 `L1`，让模型先找到好的损失景观区域；之后逐步增大 `λ_t`，引入量化约束 `L2`，引导模型沿 Pareto 前沿向低量化误差区域移动。

术语一般如何实现？如何使用？

- **理论框架**：将任何 QAT 方法视为在 `(f(Q(x)), ||x-Q(x)||)` 空间中的优化轨迹
- **CAGE 实现**：通过 `λ_t` 的动态调度（silence + linear ramp-up）实现对两个目标的渐进平衡
- **评估指标**：(1) Validation loss ↓（任务质量）；(2) Quantization error ↓（量化保真度）；(3) 两者在 Pareto 前沿上的位置
- **泛化性**：该框架不限于特定量化器或优化器，任何 QAT 方法都可以从这个视角分析其优化行为
- **局限性**：论文仅在 smooth non-convex 假设下给出收敛保证，实际深度神经网络的损失景观更复杂

涉及论文标题：
- CAGE: Curvature-Aware Gradient Estimation for Quantization-Aware Training

---

## Online Softmax (Tiled Softmax with Running Statistics)

术语是什么？

Online Softmax 是一种在分块（tiled）计算 attention 时保持数值稳定性的算法。由于 FlashAttention 沿序列长度维度分块处理 Q 和 KV，每次只能看到部分 KV tile，无法一次性获得完整的 row-wise max。Online Softmax 通过维护 running statistics——running max $m$（逐行最大值）和 running normalizer $\ell$（指数和）——在每次处理新 KV tile 时增量更新，确保最终结果与完整 softmax 数学等价。FlashAttention-4 在此基础上增加了条件 rescaling：仅当 $m_j - m_{j-1} > \tau$（$\tau = \log_2(256) = 8.0$）时才执行 rescaling，减少逐元素乘加操作。

从算法pipeline角度拆解术语：

```
# Online Softmax 在 FlashAttention 分块循环中的执行
输入: Q tile (M×d), KV tiles 逐个流式进入, τ = log₂(256) = 8.0

初始化: m = -∞ (size M), ℓ = 0 (size M), O = 0 (M×d)

for each KV tile j:
    S_j = Q @ K_j^T                         # M × N_j MMA (tensor core)
    m_new = rowmax(S_j)
    
    # FA4 条件 rescaling:
    for each row i:
        if m_new[i] - m[i] > τ:
            scale = 2^(m[i] - m_new[i])      # = exp(m_old - m_new)
            O[i,:] = scale * O[i,:]           # rescale 之前的输出
            P[i,:] = exp(S_j[i,:] - m_new[i])
        else:
            P[i,:] = exp(S_j[i,:] - m[i])     # 跳过 rescaling!
        m[i] = max(m[i], m_new[i])
    
    # 更新 normalizer
    if rescaling_occurred:
        ℓ = scale * ℓ + rowsum(P)
    else:
        ℓ = ℓ + rowsum(P)
    
    O = O + P @ V_j                           # P (TMEM) × V (SMEM)

O_final = diag(1/ℓ) @ O                       # 最终归一化
```

术语一般如何实现？如何使用？

- 所有 tiled attention 实现（FlashAttention-1/2/3/4、FlashInfer、FlexAttention）均依赖 online softmax 保证 tiled 计算的数值稳定性
- FlashAttention-4 的条件 rescaling 变体通过修改 rescaling 触发条件实现，最终归一化 `O/ℓ` 消去所有累积漂移，保证数学等价性
- 性能收益：跳过一次 rescaling = 省去 head_dim 次逐元素 FMA 操作；在实际 attention 计算中，行最大值快速收敛后大部分 KV tile 迭代可跳过 rescaling
- warp divergence 处理：当 warp 内任何线程需要 rescaling 时，整个 warp 统一执行（避免分支发散）

涉及论文标题：
- FlashAttention-4
- MAC-Attention Match-Amend-Complete Attention for Efficient Long-Context Inference

**MAC-Attention 中的 Online Attention Update (⊕)**：MAC-Attention 将 online softmax 用于 attention 输出的增量更新。⊕ 操作定义为 `A_new = A_old ⊕ Attn(Q, K_{new}, V_{new})`，其实现使用 running max 和 running normalizer 的 incremental update，等价于重新计算完整 attention 但仅访问新增的 KV 段。类似地，⊖ (downdate) 操作用于从已有 attention 输出中"减去"特定 KV 段的贡献。这两个操作使 MAC-Attention 的 Amend 阶段能够高效修正复用的 attention 输出。

---

## Conditional Softmax Rescaling

术语是什么？

条件 softmax rescaling 是 FlashAttention-4 提出的优化在线 softmax 中逐元素 rescaling 操作的算法。标准在线 softmax 中每次遇到更大的行最大值时需要对累积输出 $O$ 和 normalizer $\ell$ 乘以 rescaling factor $e^{m_{\text{old}} - m_{\text{new}}}$。FA4 设置阈值 $\tau = \log_2(256) = 8.0$（对应 rescaling factor 256），仅当 $m_{\text{new}} - m_{\text{old}} > \tau$ 时才触发 rescaling，否则跳过并使用旧 max 继续计算。最终通过 $m_{\text{final}}$ 和 $\ell_{\text{final}}$ 统一归一化消去累积的"漂移"，保证数学正确性。

从算法pipeline角度拆解术语：

```
条件 rescaling (FA4) vs 标准 online softmax:

标准做法 (FA1-FA3): 每次新 max 都 rescale
  m_new = rowmax(S_j)
  scale = exp(m_old - m_new)                   # 总是计算
  O = scale * O + exp(S_j - m_new) @ V_j       # 总是 rescale O
  ℓ = scale * ℓ + rowsum(exp(S_j - m_new))     # 总是 rescale ℓ

FA4 条件做法: 仅当 Δm > τ 时 rescale
  if m_new - m_old > τ = log₂(256) = 8.0:
      scale = exp(m_old - m_new)
      O = scale * O + exp(S_j - m_new) @ V_j
      ℓ = scale * ℓ + rowsum(exp(S_j - m_new))
      m_old = m_new
  else:
      O = O + exp(S_j - m_old) @ V_j            # 无逐元素乘法!
      ℓ = ℓ + rowsum(exp(S_j - m_old))
      # m_old 保持不变 (延迟更新)
```

关键数值分析：跳过 rescaling 引入的数值偏差 ≤ τ = 8.0（即 max 偏差 ≤ 8，对应 rescaling factor 误差 ≤ 256），在 BF16 的动态范围（~3.4×10³⁸）内安全。最终 `O/ℓ` 归一化消去所有累积漂移。

术语一般如何实现？如何使用？

- 阈值 τ 是经验选择：更大值跳过更多 rescaling 但数值风险更高；`log₂(256) = 8.0` 在 FA4 实验中表现良好
- 适用所有 tiled attention + online softmax 的实现，不限于特定 GPU 架构
- 在实际 attention 中，行最大值从初始 -∞ 快速上升到稳定值，因此大部分后续 KV tile 迭代中 rescaling 可被跳过
- 在 CUDA/CuTe kernel 层面通过条件分支 + warp-level 一致性检查实现

涉及论文标题：
- FlashAttention-4

---

## Software-Emulated Exponential (Polynomial Approximation of 2^x on GPU)

术语是什么？

软件指数函数模拟是 FlashAttention-4 提出的通过多项式近似 + IEEE 754 位操作在 GPU FMA 单元上计算 $2^x$ 的方法。核心分解：$2^x = 2^{\lfloor x \rfloor} \cdot 2^{x - \lfloor x \rfloor}$。整数部分 $2^{\lfloor x \rfloor}$ 通过 IEEE 754 exponent field 的位操作高效计算；分数部分 $2^{x_{\text{frac}}}$（$x_{\text{frac}} \in [0,1)$）通过 degree-3 多项式（Sollya 优化系数，Horner 方法在 FMA 单元求值）近似。BF16 精度下 degree-3 与硬件 MUFU.EX2 误差无差异（量化误差 $3.9 \times 10^{-3}$ 主导）。FA4 混合策略：每行 10-25% 条目用软件模拟（FMA），其余用硬件 MUFU.EX2，使指数吞吐量分散到 MUFU + FMA 两个单元并行处理。

从算法pipeline角度拆解术语：

```
软件 2^x 计算 (Cody-Waite range reduction + degree-3 polynomial):

输入: x (FP32/BF16)
输出: 2^x (近似值)

Step 1: x = max(x, -127.0)                    # clamp 防 underflow

Step 2: floor(x) via IEEE 754 trick
  biased = x + (2^23 + 2^22)                   # 将小数位移入 mantissa 低位
  x_floor = biased - (2^23 + 2^22)             # round-down mode

Step 3: x_frac = x - x_floor                   # ∈ [0, 1)

Step 4: 多项式求值 (Horner, 3 FMAs)
  # 系数 p0=1.0, p1,p2,p3 由 Sollya 优化
  t1 = fma(p3, x_frac, p2)
  t2 = fma(t1, x_frac, p1)
  result_frac = fma(t2, x_frac, 1.0)           # = 1 + p1*x + p2*x² + p3*x³

Step 5: 组合
  # 将 floor(x) 移入 exponent field + mantissa of result_frac
  result = bitwise_combine(x_floor, result_frac)
```

精度 (Table 2, FA4):
| 方法                 | FP32 Max Rel Err | BF16 Max Rel Err |
|---------------------|------------------|------------------|
| Hardware MUFU.EX2   | 1.41×10⁻⁷        | 3.89×10⁻³        |
| Degree-3 Polynomial | 8.77×10⁻⁵        | 3.90×10⁻³        | ← 与硬件BF16无差异|

混合策略：对每行 10-25% 条目用软件模拟，其余用 MUFU.EX2（MUFU 和 FMA 可并行执行）

术语一般如何实现？如何使用？

- **动机**：Blackwell B200 MUFU.EX2 仅 16 ops/clock/SM，远低于 tensor core 的 8192 ops/clock/SM；$T_{\text{exp}} = MN/16$ cycles 在 forward 中与 MMA 耗时相当，成为瓶颈
- **实现**：CUDA intrinsic `__fmaf_rn()` + `__float2int_rn()` + PTX 位操作；在 CuTe-DSL 中通过 PTX escape hatch 实现
- **何时使用**：当 $T_{\text{exp}}$ 接近或超过 $T_{\text{MMA}}$ 时（如 Blackwell forward pass）。若指数单元吞吐足够（如 B300/GB300 的 32 ops/clock/SM），收益减小
- **degree 选择**：BF16 场景 degree-3 足够（量化误差主导）；FP32 可能需要 degree-4/5

涉及论文标题：
- FlashAttention-4

---

## FlashAttention

术语是什么？

FlashAttention 是由 Dao et al. (NeurIPS 2022) 提出的一种 IO-aware 精确 attention 算法，通过**分块（tiling）**和**kernel 融合**将 attention 计算的 HBM 访问量从 O(N²d) 降至 O(Nd)。核心思想是将 Q、K、V 矩阵沿序列长度维度分块（tile），在快速的片上 SRAM 中完成 QK^T → softmax → PV 的完整计算，仅将最终输出 O 写回 HBM，避免物化大型中间张量 S（attention scores, N×N）和 P（attention probabilities, N×N）。通过 online softmax 算法保证 tiled 计算的数值稳定性——维护 running max `m`、running normalizer `ℓ` 和 running output `O`，增量更新以确保结果与完整 softmax 数学等价。

FlashAttention 是 Flashlight 要自动生成的 kernel 的**目标形态**。FlashAttention 本身是手写 CUDA/Triton kernel（或其后续版本 FA2/FA3/FA4），而 Flashlight 的目标是通过编译器从标准 PyTorch 代码自动生成等价的融合 kernel，无需手写。

从算法pipeline角度拆解术语：

```
FlashAttention 算法 (forward pass, standard scaled dot-product attention):

输入: Q, K, V ∈ R^{N×d}, scale = 1/√d
参数: tile_M (Q tile size), tile_N (KV tile size)

初始化:
  O = zeros(N, d)       # output accumulator (in HBM)
  m = -inf * ones(N)     # running max (in HBM)
  ℓ = zeros(N)           # running normalizer (in HBM)

# 外循环: 遍历 Q tile
for Q_block in tiles(Q, tile_M):
    Q_tile = load(Q_block)                         # (tile_M, d) HBM→SRAM
    m_tile = load(m[Q_block])                      # (tile_M,)
    ℓ_tile = load(ℓ[Q_block])                      # (tile_M,)
    O_tile = load(O[Q_block])                      # (tile_M, d)
    
    # 内循环: 遍历 KV tile
    for KV_block in tiles(K, tile_N):
        K_tile = load(K[KV_block])                 # (tile_N, d) HBM→SRAM
        V_tile = load(V[KV_block])                 # (tile_N, d)
        
        # Step 1: QK^T (on-chip MMA)
        S = Q_tile @ K_tile^T                      # (tile_M, tile_N)
        
        # Step 2: Online softmax (on-chip)
        m_new = max(m_tile, rowmax(S))
        P = exp(S - m_new)                          # rescale by new max
        ℓ_new = ℓ_tile * exp(m_tile - m_new) + rowsum(P)
        O_tile = diag(exp(m_tile - m_new)) @ O_tile + P @ V_tile
        
        m_tile, ℓ_tile = m_new, ℓ_new
    
    O[Q_block] = diag(1/ℓ_tile) @ O_tile           # final normalization
    m[Q_block], ℓ[Q_block] = m_tile, ℓ_tile        # write back for backward

返回: O
```

**IO 复杂度分析**：
- 标准 attention: O(N²d) HBM 访问（S 和 P 各需写入和读取 HBM）
- FlashAttention: O(N²d² / M) HBM 访问，其中 M 是 SRAM 大小
- 当 d²/M < 1 时（通常成立），实际加速约 2-4×，显存节省 10-20×

**Flashlight 与 FlashAttention 的关系**：
- FlashAttention 是行为目标：Flashlight 生成的 kernel 执行与 FlashAttention 相同的 tiled + fused 算法
- FlashAttention 是手写 kernel：需要为每种 attention 变体编写特定实现
- Flashlight 是编译器：从标准 PyTorch 代码自动生成等价 kernel，无需为每种变体手写

术语一般如何实现？如何使用？

- **FA1 (NeurIPS 2022)**：手写 CUDA kernel，仅支持 standard scaled dot-product attention（forward + backward）
- **FA2 (2023)**：C++ 模板 + CUDA，支持 ALiBi, MQA/GQA
- **FA3 (2024)**：Hopper H100 优化（warp specialization, FP8, TMA, ping-pong）
- **FA4 (2025)**：Blackwell B200 优化（TMEM, 2-CTA MMA, CuTe-DSL）
- **使用**：`from flash_attn import flash_attn_func; O = flash_attn_func(Q, K, V, causal=True)`
- **PyTorch 集成**：`torch.nn.functional.scaled_dot_product_attention` 在支持条件下自动使用 FlashAttention 后端
- **开源**：https://github.com/Dao-AILab/flash-attention (BSD License)

涉及论文标题：
- FlashAttention (NeurIPS 2022)
- FlashAttention-2, FlashAttention-3, FlashAttention-4
- Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants

---



**SLA2 的 Block-wise Sparse FlashAttention 扩展**：

SLA2 在 FlashAttention 的分块框架之上实现了稀疏-线性混合 attention kernel。核心修改：
- 新增 mask 路由判断：根据压缩路由 M_c[i,j] 的值，每个 block pair 分叉到稀疏路径（M_c=1）或线性路径（M_c=0）。
- 稀疏路径：quant(Q_i) → quant(K_j) → dequant(INT8/FP8 matmul) → softmax（online rescaling 与 FlashAttention 的 m,ℓ 状态合并） → quant(P_ij) → quant(V_j) → dequant(matmul) → O_s 累加。
- 线性路径：跳过 QKᵀ 和 PV matmul，改为累加局部 KᵀV（h_j = (K_j^φ)ᵀV_j）和归一化因子 z_j = rowsum((K_j^φ)ᵀ)，将 O(N²d) 降至 O(Nd²)。
- 最终组合：O = α⊙O^s + (1-α)⊙O^l，其中 α 为可学习的每行比例向量。
- RTX 5090 上 97% sparsity 时达到 FlashAttn2 的 18.7× 加速。


- SLA2: Sparse-Linear Attention with Learnable Routing and QAT

## Differential Attention

术语是什么？

Differential Attention（差分注意力）是一种 attention 变体，通过两组独立的 QK pair 计算 attention score 的差分（`A = A1 - λ·A2`），旨在减少 attention 中的噪声并提升稀疏性。与 standard attention 的单组 `Q@K^T` 不同，differential attention 使用两组 query 和 key：`(Q1, K1)` 和 `(Q2, K2)`，计算 `A1 = Q1@K1^T` 和 `A2 = Q2@K2^T`，最终 attention score 为 `A = A1 - λ·A2`，其中 λ 是可学习的标量系数。Differential attention 属于 **data-dependent attention**（attention 计算依赖运行时数据而非仅是静态 mask 模式），因此超出了 FlexAttention 模板的表达能力。

从算法pipeline角度拆解术语：

```
Differential Attention 算法:

输入: Q1, Q2, K1, K2, V ∈ R^{N×d}
      λ (标量或由 λ_q1·λ_k1 / λ_q2·λ_k2 动态计算)
输出: O ∈ R^{N×d}

A1 = Q1 @ K1^T * (1/√d)         # 第一组 attention score (N×N)
A2 = Q2 @ K2^T * (1/√d)         # 第二组 attention score (N×N)
A = A1 - λ * A2                  # 差分: element-wise subtraction
A = GroupNorm(A)                 # 可选: per-head group normalization
P = softmax(A, dim=-1)           # (N×N)
O = P @ V                        # (N×d)

# FlexAttention 无法表达的原因:
# - score_mod(score, b, h, q_idx, kv_idx) 只接收单个 score
# - Differential attention 需要两组独立的 QK pair + λ 差值
# - λ 是 data-dependent (依赖 Q1, Q2, K1, K2 的实际值)
```

术语一般如何实现？如何使用？

- **使用场景**：减少 attention 噪声、提升长上下文性能（论文报告在长上下文任务上优于 standard attention）
- **实现方式**：自定义 CUDA/Triton kernel（双 matmul 分支）或使用 Flashlight 从标准 PyTorch 代码自动生成融合 kernel
- **Flashlight 实现**：用户写标准 PyTorch 差分 attention 代码，Flashlight 将两个 QK^T matmul、差分组合、softmax、PV matmul 全部融合为一个 kernel
- **性能**：Flashlight 生成的 diffattn kernel 相比 `torch.compile` 默认（3+ kernel）有显著加速

涉及论文标题：
- Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants

---

## Evoformer Gated Self-Attention (AlphaFold Attention)

术语是什么？

Evoformer 是 AlphaFold2 蛋白质结构预测模型的核心模块，其 attention 机制包含两种独特的**门控自注意力（gated self-attention）**变体：row-wise 和 column-wise gated self-attention。与 standard attention 不同，Evoformer 的 attention 在 softmax 之后、与 V 相乘之前，需要额外乘以一个门控值：`O = sigmoid(gate) ⊙ (softmax(QK^T + bias) @ V)`，其中 gate 和 bias 都来自 pair representation 的线性变换。这种 multi-branch、data-dependent 的计算模式使得 Evoformer attention 是 Flashlight 展示其超越 FlexAttention 能力的核心案例——Flashlight 在该变体上取得了 5× 加速。

从算法pipeline角度拆解术语：

```
Evoformer Row-wise Gated Self-Attention:

输入:
  Q, K, V ∈ R^{N×d}           # MSA representation
  gate ∈ R^{N×N×1}            # gating values (from pair representation)
  bias ∈ R^{1×N×N}            # attention bias (from pair representation)

计算:
  S = Q @ K^T * (1/√d)                       # attention score (N×N)
  S = S + bias                                # add pair bias
  P = softmax(S, dim=-1)                      # (N×N)
  g = sigmoid(gate)                           # gating values (data-dependent!)
  P_gated = g ⊙ P                             # element-wise gate modulation
  O = P_gated @ V                             # (N×d)

# 与 standard attention 的区别:
# (1) bias 从 pair representation 注入 (data-dependent, 非静态位置编码)
# (2) gating: P 在乘以 V 之前经过 sigmoid gate 调制
# (3) gate 和 bias 都是 data-dependent (依赖 MSA + pair 特征)
```

**Flashlight 的 5× 加速来源**：
- `torch.compile` 默认将 Evoformer attention 拆分为 5+ 个 kernel：QK^T matmul (CUBLAS) → bias add (Triton) → softmax (Triton) → sigmoid gate (Triton) → gating multiply (Triton) → PV matmul (CUBLAS)，每个中间结果写回 HBM
- Flashlight 通过 Unified Reduction IR + Semantic Fusion + Structural Fusion + Twin Matmul 将所有操作融合为一个 kernel，S, P, gated_P 全部在片上即时消费
- AlphaFold2 端到端推理延迟改善 6-9%（Evoformer 是主要耗时模块，但整体包含其他模块如 IPA）

术语一般如何实现？如何使用？

- **应用场景**：AlphaFold2/OpenFold 蛋白质结构预测（48 层 Evoformer, 序列长度 ~256-1024）
- **原始实现**：JAX/Haiku（AlphaFold2），attention 未作特殊优化
- **PyTorch 实现**：OpenFold 使用 PyTorch 重写，可通过 `torch.compile(model, enable_flashlight=True)` 启用 Flashlight
- **Flashlight 优势**：无需修改模型代码，编译器自动识别 gated attention 模式并生成融合 kernel

涉及论文标题：
- Flashlight: PyTorch Compiler Extensions to Accelerate Attention Variants
- AlphaFold2 (Jumper et al., Nature 2021)

## MAC-Attention (Match-Amend-Complete Attention, 匹配-修正-补全注意力)

术语是什么？

MAC-Attention 是一种训练无关（training-free）、模型无关的 LLM 长上下文解码加速方案（MLSys 2026）。核心思想是将 decode 阶段的 attention 计算从每次从零开始重新计算 O(N·d) KV cache，转变为"缓存 + 增量修正"范式：复用语义相似历史 query 的 prior attention output，仅对不匹配区域进行局部修正和补全。三个阶段的含义：(1) **Match**：在滑动窗口内用 pre-RoPE L2 距离找到与当前 query 最相似的历史 query，复用其缓存的 attention 输出；(2) **Amend**：在匹配边界附近的小 band 上重新计算 attention，修正 softmax 质量集中在 decode 光标附近导致的位置偏差；(3) **Complete**：用数值稳定的 log-sum-exp merge 将修正后的 prefix attention 与 KV tail 上的 fresh attention 融合。匹配命中时计算和带宽复杂度为常数级别（仅处理 O(r+tail) 而非 O(N)），与 context 长度无关。

从算法pipeline角度拆解术语：

```
# MAC-Attention decode 算法流程
输入: Q_n (新 query, pre-RoPE & post-RoPE), K_cache, V_cache (paged KV)
全局状态: Q_ring[κ], A_ring[κ] (滑动窗口缓存, κ=512)

# === Stage 1: Match ===
for i in range(len(Q_ring)):
    dist[i] = ||Q_n_pre_rope - Q_ring[i]||_2  # pre-RoPE L2 距离
best_idx = argmin(dist)
if dist[best_idx] > τ (τ=0.45):
    return full_attention(Q_n, K_{0~n}, V_{0~n})  # miss → fallback

Q_m = Q_ring[best_idx]
A_m = A_ring[best_idx]
m = n - κ + best_idx  # 匹配 token 在序列中的绝对位置

# === Stage 2: Amend ===
j = max(m - r, 0)  # rectification band 起始位置, r=256
# 从 A_m 中移除旧 band attention, 加入新 band attention
A_prefix = A_m ⊖ Attn(Q_m, K_{j~m}, V_{j~m})   # downdate (移除)
         ⊕ Attn(Q_n, K_{j~n}, V_{j~n})          # update (加入)
# ⊖ 和 ⊕ 均为 online softmax update 操作

# === Stage 3: Complete ===
A_tail = Attn(Q_n, K_{m~n}, V_{m~n})  # fresh attention on KV tail
A_n = logsumexp_merge(A_prefix, A_tail)  # 数值稳定融合

# === Cache Update ===
Q_ring.append(Q_n_pre_rope); A_ring.append(A_n)
if len(Q_ring) > κ: Q_ring.pop(0); A_ring.pop(0)
return A_n
```

KV skip 比例：r_skip = (N − r)/N，当 N 很大时接近 100%（120K context 时 ~98.9% KV 被跳过）。

术语一般如何实现？如何使用？

- **匹配窗口设计**：κ=512 的滑动窗口，仅缓存最近 token 的 Q 和 A，保证 attention 始终计算 ≤ κ 个 token。匹配和检索仅在此窗口内发生。这同时解决了"无法物理缓存全部 query/attention"和"匹配到太早的 query 无法节省足够 KV 计算"的问题。
- **匹配相似度度量**：使用 pre-RoPE（应用 Rotary Position Embedding 之前）query 的 L2 距离，因为 RoPE 编码了位置信息，pre-RoPE query 更适合语义相似度比较。
- **Per-head 独立匹配**：每个 query head 独立进行匹配（不同 head 可匹配到不同位置），配合 load balancing planning 确保各 head 的计算负载均衡。
- **兼容性**：与 chunked prefill、continuous batching、speculative decoding、PD disaggregation、MHA/GQA 均兼容。对 Llama 3.1 家族验证，Qwen3-30B-A3B-Instruct 在 LongBench v2 上质量与 full attention 持平。
- **阈值 τ 选择**：τ=0.45 时匹配命中率 ~99.5%，KV skip 率 ~98.9%。τ 越大 → 匹配越严格 → 命中率越低但精度越高。
- **开源**：https://github.com/YJHMITWEB/MAC-Attention.git，ACM AE Badge 认证，MLSys 2026。

涉及论文标题：
- MAC-Attention Match-Amend-Complete Attention for Efficient Long-Context Inference

---

## K-smoothing (Attention Key Smoothing for Quantization, 注意力Key平滑量化预处理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

K-smoothing 是 SageAttention 系列（Zhang et al., 2025d;a;c）提出的 attention 量化预处理技术：在对 Key 矩阵 K 进行 INT8 量化之前，先减去 K 的**逐行均值（token-wise mean）**，即 K^{sm} = K − mean_row(K)，其中 mean_row(K) ∈ R^{1×d} 对序列维度（N 个 token）取平均。核心动机是：Q 和 K 矩阵在通道（head dimension）维度上存在显著的异常值（outliers）——某些通道的值可比其他通道大 20× 以上，这些异常值主导了 INT8 量化的 scale = max(|X|)/127，导致大量正常值被粗量化。减去列均值后，K^{sm} 的分布以零为中心，动态范围大幅压缩，INT8 量化步长减小，量化精度提高。

在 SageBwd 中的关键发现：K-smoothing 在训练中是**必要且无额外开销**的。由于 softmax 梯度 dS 的每行求和为 0（数学性质），dQ = dS·K = dS·(K − 1·mean_row(K)⊤) = dS·K^{sm}，即反向 pass 无需任何修正项。这使得 K-smoothing 成为"仅在 kernel 入口处修改 K、无需改动反向逻辑"的零额外开销技术。

相比之下，Q-smoothing（Q^{sm} = Q − mean_row(Q)）在训练中**不一定受益**。因为 Q-smoothing 将 logits 分解为 S = Q^{sm}K⊤ + 1(μ_QK⊤)，反向时 dK 不能仅由 centered branch 计算：dK = dS⊤Q = dS⊤Q^{sm} + (dS⊤1)μ_Q⊤ = dK_center + dK_bias。额外的 bias 分支 dK_bias 引入新的量化噪声路径，可能抵消 Q-smoothing 减少激活动态范围带来的收益。SageBwd 实验证实 Q-smoothing 在预训练中无持续改进，甚至可能略微降低梯度保真度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

K-smoothing 在 SageBwd INT8 attention 训练 pipeline 中的位置：

```
输入: Q, K, V ∈ R^{N×d} (FP16/BF16)

# === K-smoothing (在 kernel 入口执行) ===
K_sm = K - mean_row(K)            # R^{N×d}: 每列减该列的 N 个 token 均值

# === Per-block INT8 量化 Q 和 K_sm (而非原始 K) ===
for each FlashAttention tile Q_i (B_q×d), K_j (B_kv×d):
    s_Q = max(|Q_i|) / 127,  Q̂_i = round(Q_i / s_Q)
    s_K = max(|K_j_sm|) / 127, K̂_j = round(K_j_sm / s_K)  # 量化 smoothed K

# === 前向: Q̂_i×K̂_j 在 INT8 Tensor Core 上执行 ===
S_ij = Q̂_i @ K̂_jᵀ × s_Q × s_K     # dequant 恢复为 FP16
# → online softmax → P̂_ij×V̂_j → O

# === 反向: dQ 计算直接使用 K_sm ===
dQ_i += dŜ_ij @ K̂_j_sm × s_dS × s_K
# 无需修正: dS 每行和=0 ⇒ dS·(1·mean_row(K)⊤) = 0 ⇒ dQ = dS·K = dS·K_sm ✓

# === 反向: dK 计算 ===
dK_i += dŜ_ijᵀ @ Q̂_i × s_dS × s_Q
# 注意: dK 使用的是原始 Q（若仅做 K-smoothing）= 若做 Q-smoothing 需额外 bias 项
```

数学证明（K-smoothing 的零开销性）：
- dS 每行和 = 0（softmax 梯度的性质：∂L/∂S 的行均值在 softmax + cross-entropy 下为 0）
- dQ = dS·K = dS·(K^{sm} + 1·μ_K⊤) = dS·K^{sm} + (dS·1)·μ_K⊤ = dS·K^{sm} + 0·μ_K⊤ = dS·K^{sm}
- 因此 dQ 可直接用 K^{sm} 计算，无需存储或恢复原始 K

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现位置**：在 FlashAttention-style kernel 的入口处，K tile 从 HBM 加载到 SRAM 后、INT8 量化前，使用 CUDA/Triton 的 reduce 操作计算 per-tile column mean 并原地减去。
- **mean_row(K) 的计算**：对序列维度 N 求平均，得到 R^{1×d} 向量。在 tiled kernel 中可近似为 per-tile 或使用全局 moving average。SageAttention 使用全局 K 均值（整个 K 矩阵的 token-wise mean）。
- **与 Q-smoothing 的取舍**：SageBwd 默认仅用 K-smoothing。Q-smoothing 的 bias 分支引入额外量化噪声，在预训练中未观察到收益。推理场景（SageAttention/SageAttention2）中 Q-smoothing 可能有用，因为推理无需反向 pass。
- **与其他平滑技术的对比**：(1) SmoothQuant (Xiao et al., 2023) 将量化难度从 activation 迁移到 weight，通过 per-channel scaling 实现；(2) K-smoothing 仅对 K 做 token-wise mean subtraction，更轻量级。
- **开源**：https://github.com/thu-ml/SageAttention

涉及论文标题：
- SageBwd

## QK-norm (Query-Key Normalization, 查询-键归一化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

QK-norm（Query-Key Normalization）是由 Henry et al. (2020, EMNLP Findings) 提出的 Transformer 训练稳定技术：在计算 attention logits S = QK⊤/√d 之前，先对 Q 和 K 的每个 token 应用 RMSNorm（或 ℓ2-norm），即 Q_norm = RMSNorm(Q), K_norm = RMSNorm(K)，再计算 S = Q_norm·K_norm⊤。原始论文使用 ℓ2-norm + 可学习温度参数 g：`softmax(g · Q̂K̂⊤)·V`，其中 Q̂ = Q/‖Q‖₂, K̂ = K/‖K‖₂。后续工作（Dehghani et al., 2023; Gemma2, 2024）常用 RMSNorm 替代 ℓ2-norm，效果相当但计算更高效。

QK-norm 的本质是将 scaled dot-product attention 转变为**缩放余弦相似度 attention**：归一化后的 Q 和 K 的模长为 1（或 RMS=1），logits 值域被严格控制，避免因 Q 和 K 范数增长导致的 softmax 饱和（梯度消失）或数值溢出。

在 SageBwd 中的关键发现：QK-norm 在大 tokens-per-step（TPS=2.1M）下是**防止训练崩溃的必要条件**。原因：(1) 未归一化的 Q 和 K 在预训练中范数持续增长（RMSNorm 的 learnable scale γ 逐渐增大），导致 INT8 量化步长增大，绝对量化噪声增加；(2) 量化噪声沿 dS 路径（RMS 仅~10⁻⁷）传播，小绝对误差产生大相对误差；(3) 大 TPS 下梯度噪声低，系统性量化偏差不会被噪声掩盖，导致训练发散。QK-norm 通过将 Q 和 K 的动态范围压缩到可控区间，使 INT8 量化步长保持较小，从而控制量化误差在可接受水平。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

QK-norm 在 SageBwd pipeline 中的使用：

```
# 标准 attention（无 QK-norm）
S = Q @ Kᵀ / √d              # Q,K ∈ R^{N×d}, S 值域不受控
P = softmax(S)                # softmax 可能饱和

# QK-norm attention（Henry et al., 2020 原始 + RMSNorm 变体）
Q_norm = RMSNorm(Q)           # per-token RMSNorm: Q_i = Q_i / RMS(Q_i) * γ_Q
K_norm = RMSNorm(K)           # γ_Q, γ_K 为可学习 scale 向量 ∈ R^{d}
S = Q_norm @ K_normᵀ          # 等价于缩放余弦相似度, 值域[-γ², γ²]
P = softmax(S)                # softmax 稳定, 梯度良好

# 或使用 ℓ2-norm + 温度参数 (Henry et al., 2020 原始)
Q_hat = Q / ||Q||₂            # per-token ℓ2 normalization
K_hat = K / ||K||₂
S = g * Q_hat @ K_hatᵀ       # g 为可学习标量温度参数, 初始化 = 1/√d 或 log(N)
```

QK-norm 训练稳定性对比（830M 参数实验, from "Methods for improving LLM training stability", 2024）：

| Method | Max LR without divergence |
|--------|--------------------------|
| Baseline (BF16) | 6e-3 |
| Soft capping | 40e-3 |
| QK-norm | 40e-3 |
| QKV-norm | 60e-3 |
| QK-norm + soft capping | 60e-3 |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现方式**：在 PyTorch 中，QK-norm 通常实现为在 attention 计算前对 Q 和 K 调用 `F.rms_norm` 或自定义 RMSNorm 层。注意 QK-norm 的 RMSNorm **不共享** LayerNorm 的参数——Q 和 K 各有独立的 RMSNorm 权重。
- **RMSNorm vs ℓ2-norm**：RMSNorm 计算 RMS = √(mean(x²))，计算复杂度 O(d)；ℓ2-norm 计算 √(Σx²)，同样 O(d) 但多一步 mean vs sum。RMSNorm 的梯度更稳定（避免除零），现代实现偏好 RMSNorm。
- **γ 参数的漂移**：SageBwd 特别关注 RMSNorm 的 learnable scale γ 在预训练中的增长——γ 逐渐增大意味着有效 σ_Q, σ_K 增大，即使 QK-norm 仍可能最终使 INT8 量化误差超标。这是大 TPS 下 SageBwd 仍需要额外谨慎的原因。
- **与 MLA (Multi-Latent Attention) 的关系**：MLA（DeepSeek-V2/V3）中 Q 和 K 经过低秩压缩后 latent space 不直接对应 head dim，QK-norm 不可直接应用。替代方案包括 QuacK (Anson & Aitchison, 2025) 和 QK Clip (Kimi, 2025)。
- **开源**：原始 QK-norm 代码 https://github.com/CyndxAI/QKNorm

涉及论文标题：
- SageBwd

## Per-block INT8 Quantization for Attention (注意力 Per-block INT8 量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Per-block INT8 Quantization 是一种量化粒度策略：将矩阵 X ∈ R^{m×n} 按行或按块（block）划分为子矩阵，每个子矩阵独立计算量化 scale 因子 δ = max(|X_block|)/127，该 block 内所有元素共享同一个 scale。在 SageBwd 中，"block" 对应 FlashAttention 的分块 tile（Q tile: B_q×d, K tile: B_kv×d），因此 per-block quantization 与 FlashAttention 的内存 tiling 天然对齐——Q_i、K_j 和 V_j 在从 HBM 加载到 SRAM 后、执行 MatMul 前就地量化。

与 per-tensor quantization（整个矩阵共享一个 scale）相比，per-block quantization 的粒度更细，能更好捕获局部数值分布特征，减少异常值对量化精度的全局影响。与 per-channel quantization 相比，per-block 在序列维度上更细粒度，特别适合 attention 中 Q 和 K 的数值分布沿序列维度变化剧烈（不同 token 的 Q 范数可能相差数倍）的场景。

SageBwd 中使用了三种量化粒度的混合方案：(1) QKᵀ 的 Q 和 K 使用 per-block INT8（FlashAttention tile 粒度）；(2) PV 的 P 使用 per-token INT8（每行一个 scale），V 使用 per-block INT8；(3) 反向中 dS、dO、P 均使用 per-block INT8。这种混合设计在量化精度和实现复杂度之间取得平衡。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Per-block INT8 量化在 SageBwd 前向 QKᵀ 计算中的流程：

```
# Q_i ∈ R^{B_q×d}: FlashAttention Q tile
# K_j ∈ R^{B_kv×d}: FlashAttention K tile

# Step 1: Per-block 量化 Q_i
δ_Q = max(|Q_i|) / 127              # 标量, tile 内所有元素共享
Q̂_i = round(Q_i / δ_Q)               # INT8, 值域 [-127, 127]
# 存储开销: 1 个 FP32 scale + B_q×d 个 INT8 = 4 + B_q×d bytes (vs FP16 的 2×B_q×d)

# Step 2: Per-block 量化 K_j (转置后量化)
δ_K = max(|K_j|) / 127
K̂_j = round(K_j / δ_K)               # INT8, 转置布局: d×B_kv → B_kv×d 量化

# Step 3: INT8 MatMul on Tensor Core
Ŝ_ij = Q̂_i @ K̂_jᵀ                    # INT8 matmul → INT32 accumulator

# Step 4: Dequantize
S_ij = Ŝ_ij × δ_Q × δ_K              # INT32 → FP32, 恢复为 attention logits
```

per-tensor vs per-block vs per-channel 对比：

| 粒度 | Scale 数量 | 行-列共享 | 适用场景 |
|------|-----------|----------|---------|
| Per-tensor | 1 | 全矩阵共享 | 分布均匀的 weight |
| Per-channel (row) | m | 每行独立 | channel-wise 异常值（常见于 activation） |
| Per-block | m×n/(B_q×B_kv) | tile 内共享 | FlashAttention tiling 天然对齐 |
| Per-token (row) | m | 每 token 独立 | softmax 后的 P（每行和为1） |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现**：在 Triton kernel 中使用 `tl.max(tl.abs(x))` 计算 per-tile scale，然后 `tl.round(x / scale)` 量化。Triton 自动将 `@` 操作映射到 GPU Tensor Core 指令（通过 `tl.dot` 的 `input_precision` 参数指定 INT8）。
- **量化公式**：对称均匀量化（symmetric uniform quantization）：`x̂ = clamp(round(x/δ), -127, 127), δ = max(|x|)/127`。无 zero point（对称），简化 dequant 为纯乘法。
- **SageBwd 中 dP 不量化的理由**：dP = dO·V⊤ 保持 FP16 是 SageBwd 最关键的精度保留设计——per-block INT8 量化的绝对误差约 δ/2（~0.4% of max(|X|)），对 dP 来说可容忍，但 dP 的量化误差会通过 dS = P∘(dP−δ) 被 P 的极小值（~5×10⁻³）缩小后进入 dS（~10⁻⁷），产生巨大的相对误差。
- **与其他量化方案的对比**：(1) FlashAttention3 FP8 使用 per-tensor FP8（e4m3/e5m2），需硬件 FP8 Tensor Core（H100+）；(2) SageAttention2++ 使用 per-thread INT4，更细粒度；(3) SmoothQuant 使用 per-channel 对称 INT8 + per-token 动态量化。

涉及论文标题：
- SageBwd

## dS (Softmax Gradient) Sensitivity in Low-bit Attention Training (低比特注意力训练中的Softmax梯度敏感性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

dS（Softmax Gradient）是 attention 反向传播中 softmax 操作的梯度张量，定义为 dS = P∘(dP − δ1⊤)，其中 P = softmax(QK⊤/√d) 是 attention 概率矩阵，dP = dO·V⊤ 是 loss 对 P 的梯度，δ = rowsum(dO∘O) 是每行的修正项。dS 的物理意义：将上游梯度 dP 通过 softmax 的 Jacobian 变换后，得到对 logits S 的梯度。dS 是反向 pass 的关键中间张量——dQ = dS·K 和 dK = dS⊤·Q 都依赖 dS。

SageBwd 的核心发现：dS 是 INT8 量化注意力训练的**瓶颈张量**。在 N=4096 序列长度下，实测 RMS(dS) ≈ 1×10⁻⁷，而 RMS(P) ≈ 5×10⁻³、RMS(dP) ≈ 5×10⁻⁵。dS 的极小量级源于双重压缩：(1) P 的每行是概率分布（和为 1），最大单元素约 1/N；(2) dP−δ 的每行均值被 δ 吸收。理论分析（SageBwd Appendix B）证明 RMS(dS) ≤ (1/√N)·max_i‖dP_i − δ_i·1‖_∞——序列越长，dS 越小。

INT8 量化引入的是近似**固定的绝对噪声**（约 scale/2 ≈ max(|dS|)/254），当信号本身量级为 10⁻⁷ 时，相对误差可达 ~50%+。这与其他张量（O、dV 等 RMS 约 0.1-1.0）形成鲜明对比——同样的 INT8 绝对噪声对"大信号"可忽略，对 dS 则是灾难性的。SageBwd 的实验验证：dS 的 cosine similarity（vs FPA）仅 0.9789，相对 ℓ² error 达 0.2045，是除 dQ/dK（被 dS 误差传播放大）外最差的张量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

dS 计算与误差传播路径：

```
# === dS 的产生 ===
# 前向: O = softmax(QKᵀ/√d) @ V
# 反向输入: dO ∈ R^{N×d} (loss 对 O 的梯度)

# Step 1: 计算 δ (用前向 O)
δ = rowsum(dO ∘ O)                 # R^{N}: dO 和 O 逐元素乘 → 行求和
                                    # 物理意义: softmax Jacobian 的对角修正项

# Step 2: 计算 dP (FP16, 不量化 —— SageBwd 关键!)
dP = dO @ Vᵀ                        # R^{N×N}: 若量化此步 → 误差连锁放大

# Step 3: 计算 dS (量化瓶颈)
dS = P ∘ (dP - δ·1ᵀ)               # R^{N×N}: element-wise
                                    # P ≈ 5×10⁻³, (dP-δ) ≈ 5×10⁻⁵ → dS ≈ 1×10⁻⁷

# Step 4: dS INT8 量化 (误差源!)
s_dS = max(|dS|) / 127
dŜ = round(dS / s_dS)              # 量化噪声 ≈ s_dS/2 ≈ max(|dS|)/254

# Step 5: 误差传播到 dQ 和 dK —— 被 Q 和 K 的范数放大!
dQ = dŜ @ K × s_dS                  # K 的范数 ~1-10 → 误差放大 1-10×
dK = dŜᵀ @ Q × s_dS                 # Q 的范数 ~1-10 → 误差放大 1-10×
```

dS 的 magnitude bound 推导（SageBwd Appendix B）：

$$
\mathrm{RMS}(\mathbf{dS}) \leq \frac{1}{\sqrt{N}} \max_{i} \|\mathbf{dP}_{i} - \boldsymbol{\delta}_{i} \mathbf{1}\|_{\infty}
$$

证明链：
1. RMS(dS_i)² = (1/N)·Σⱼ P_{i,j}²·(dP_{i,j} − δ_i)²
2. ≤ ‖dP_i − δ_i·1‖_∞² · (1/N)·Σⱼ P_{i,j}²
3. = ‖dP_i − δ_i·1‖_∞² · RMS(P_i)²
4. ≤ ‖dP_i − δ_i·1‖_∞² / N （因为 P_i 是概率分布, RMS(P_i) ≤ 1/√N）
5. 全局: RMS(dS) ≤ (1/√N)·max_i‖dP_i − δ_i·1‖_∞

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **FlashAttention 中 dS 不物化**：FlashAttention kernel 中 dS 在 SRAM 内即时计算和消费——从 dO、V、P 和 δ 在线计算 dS，立即用于 dQ 和 dK 累加，不写回 HBM。这意味着 dS 的量化噪声观察是间接的——通过 dQ 和 dK 的精度损失体现。
- **SageBwd 的缓解策略**：(1) dP 保持 FP16——切断 dS 误差的最上游来源；(2) QK-norm 降低 Q 和 K 范数——减少误差传播放大倍数；(3) K-smoothing + per-block 量化——最小化 dS 的量化步长；(4) 降低 TPS——利用梯度噪声掩蔽量化偏差。
- **实测数据**（N=4096, SageBwd with QK-norm, 78B tokens）：RMS(P) ≈ 5×10⁻³, RMS(dP) ≈ 5×10⁻⁵, RMS(dS) ≈ 1×10⁻⁷。dS 比 dP 小约 500×（理论 bound 仅约 64×——说明 dP−δ 的分布比最坏情况更集中）。
- **与 general INT8 training 的关系**：dS 的脆弱性不仅是 attention 特有的——任何含 softmax 的模块在低比特训练中都可能遇到类似问题。JPQI (Hu et al., 2025) 的 "post-quantization integral" 从信息论角度分析了类似现象。

涉及论文标题：
- SageBwd

## Trainable Low-bit Attention (SageBwd, 可训练低比特注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Trainable Low-bit Attention (SageBwd) 是 Zhang et al. (2025c, SageAttention3) 提出、并在本文中系统分析的可训练 INT8 注意力机制。与仅支持推理的低比特注意力（SageAttention/SageAttention2/FlashAttention3 FP8）不同，SageBwd 同时支持前向和反向 pass 中的 INT8 量化——将 attention 的 7 个矩阵乘法中的 6 个量化为 INT8，仅保留 dP = dO·V⊤ 为 FP16。在 325M Llama 模型预训练（78B tokens）中，SageBwd 在 260K TPS 下匹配 FPA 性能（loss: 2.561 vs 2.563），在 RTX 4090 上达到 FlashAttention2 1.67× 的端到端吞吐加速。

SageBwd 的核心设计决策树：
1. **前向量化**：QKᵀ 用 per-block INT8（Q 和 K 各自 per-tile scale）+ K-smoothing；PV 用 per-token INT8（P）+ per-block INT8（V）
2. **反向选择量化**：dP = dO·V⊤ 保持 FP16（最关键——阻断 dS 误差传播链）；dV = Pᵀ·dO、dQ = dS·K、dK = dSᵀ·Q 以及 dS 本身均使用 per-block INT8
3. **数值稳定**：QK-norm（RMSNorm on Q,K）+ K-smoothing（仅 K，无 Q-smoothing）控制量化误差
4. **训练策略**：TPS 不宜过大（推荐 ≤260K tokens/step），利用梯度噪声掩蔽量化偏差

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

SageBwd 完整前向+反向 pipeline（单头 attention, N=4096, d=128）：

```
=== 前向 pass ===
输入: Q, K, V ∈ R^{N×d} (BF16/FP16)

1. QK-norm: Q = RMSNorm(Q), K = RMSNorm(K)     # 控制动态范围
2. K-smoothing: K = K - mean_row(K)              # 消除通道异常值
3. 分块: {Q_i} (T_m blocks of B_q×d), {K_j}, {V_j} (T_n blocks of B_kv×d)
4. Per-block INT8 量化 Q_i, K_j, V_j:
     Q̂_i = round(Q_i / max(|Q_i|)*127), s_Q_i = max(|Q_i|)/127
     K̂_j = round(K_j / max(|K_j|)*127), s_K_j = max(|K_j|)/127
     V̂_j = round(V_j / max(|V_j|)*127), s_V_j = max(|V_j|)/127
5. Tiled QKᵀ + online softmax + PV (FlashAttention 框架):
     for i in 1..T_m:
         O_i=0, m_i=-inf, l_i=0
         for j in 1..T_n:
             S_ij = Q̂_i @ K̂_jᵀ × s_Q_i × s_K_j    # INT8 MatMul → dequant
             m_new = max(m_i, rowmax(S_ij))
             P̃_ij = exp(S_ij - m_new)
             s_P = exp(rowmax(S_ij)-m_new)/127     # per-token scale
             P̂_ij = round(P̃_ij / s_P)               # per-token INT8 quant P
             O_i = diag(e^{m_i-m_new})×O_i + P̂_ij @ V̂_j × s_P × s_V_j
             m_i,l_i = m_new, l_new·e^{m_i-m_new}+rowsum(P̃_ij)
         O_i = O_i / l_i                            # final rescale
6. 返回: O = {O_i}, L = {m_i + log(l_i)} (for backward)

=== 反向 pass ===
输入: dO ∈ R^{N×d}, 前向量化参数 {s_Q,Q̂}, {s_K,K̂}, {s_V,V̂}, {L_i}

1. D = rowsum(dO ∘ O)                              # δ = rowsum(dO∘O)
2. for j in 1..T_n:
       for i in 1..T_m:
           # Recompute S, P (from forward quantized Q,K)
           S_ij = Q̂_i @ K̂_jᵀ × s_Q_i × s_K_j
           P_ij = exp(S_ij - L_i)                   # recompute P from forward L

           # dV = PᵀdO (INT8)
           s_P, P̂_ij = INT8_quant(P_ij)             # per-block
           s_dO, dÔ_i = INT8_quant(dO_i)            # per-block
           dV_j += P̂_ijᵀ @ dÔ_i × s_P × s_dO        # INT8 MatMul

           # dP = dOVᵀ (FP16 —— 不量化!)
           dP_ij = dO_i @ V_jᵀ                       # FP16 MatMul

           # dS = P∘(dP-δ) → INT8 quant
           dS_ij = P_ij ∘ (dP_ij - D_i)             # softmax gradient
           s_dS, dŜ_ij = INT8_quant(dS_ij)          # per-block

           # dQ = dS×K, dK = dSᵀ×Q (INT8)
           dQ_i += dŜ_ij @ K̂_j × s_dS × s_K_j       # INT8 MatMul
           dK_i += dŜ_ijᵀ @ Q̂_i × s_dS × s_Q_i      # INT8 MatMul

3. 返回: dQ, dK, dV
```

量化统计：7 个 MatMul → 6 个 INT8 + 1 个 FP16 (dP)
- Forward:  QKᵀ (INT8), PV (INT8) → 2/2 INT8
- Backward: dV=PᵀdO (INT8), dP=dOVᵀ (FP16), dQ=dSK (INT8), dK=dSᵀQ (INT8), + dS compute → 4/5 INT8

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现框架**：OpenAI Triton（https://github.com/triton-lang/triton）。Triton 自动将 `tl.dot` 映射到 GPU Tensor Core 指令，`input_precision="int8"` 启用 INT8 Tensor Core。
- **使用条件**：(1) 需要 QK-norm（RMSNorm on Q,K）；(2) 推荐 TPS ≤ 260K；(3) K-smoothing 必需；(4) Q-smoothing 可省略（SageBwd 默认关闭）
- **预训练 vs 微调**：SageBwd 在微调中天然零性能损失（SageAttention3 原论文）；预训练需满足上述 TPS 和 QK-norm 条件。
- **kernel 性能**：RTX 4090 上 head dim=64/128 达到 FA2 1.67× 吞吐。当前实现优先正确性和稳定性而非激进 kernel 融合。
- **开源**：https://github.com/thu-ml/SageAttention（预计 2025 年 7 月 15 日开源 SageAttention3/SageBwd 代码）。
- **局限**：(1) 大 TPS（>2M tokens/step）下 SageBwd 仍与 FPA 有差距（loss 2.640 vs 2.586），需要进一步缓解 dS 量化误差；(2) 未验证 >1B 参数模型和更长序列。

涉及论文标题：
- SageBwd

## Pre-RoPE L2 Query Matching (应用RoPE前的L2查询匹配)

术语是什么？

Pre-RoPE L2 Query Matching 是 MAC-Attention 中 Match 阶段使用的 query 相似度搜索机制。在 Transformer 中，Rotary Position Embedding (RoPE) 通过旋转变换将位置信息编码到 query 和 key 向量中——post-RoPE query 包含了位置相关分量，不适合直接用于语义相似度比较。MAC-Attention 在应用 RoPE 之前保存 query 向量（pre-RoPE Q），用 L2 距离在滑动窗口内搜索最近邻。L2 距离度量的是向量空间的欧氏距离，相比余弦相似度更适合捕捉 attention 分布的全局缩放差异。

从算法pipeline角度拆解术语：

```
# Pre-RoPE vs Post-RoPE 在 query matching 中的角色
原始 query: q ∈ R^d

# Pre-RoPE: 纯语义表示
q_pre = q  # 未施加位置编码, 适合语义匹配

# Post-RoPE: 语义+位置混合
q_post = RoPE(q, pos_n)  # 施加了位置 n 的旋转变换, 适合 attention 计算

# MAC-Attention 的使用方式:
#   Match 阶段: 用 q_pre 在 Q_ring 中做 L2 最近邻搜索
#   Compute 阶段: 用 q_post 做 attention 计算 (与 K_cache 交互)
```

术语一般如何实现？如何使用？

- **缓存开销**：每个 decode token 需额外保存 pre-RoPE Q（d 维 BF16 向量 × Hq heads），仅在滑动窗口 κ=512 内保存，内存开销可忽略（~Hq × κ × d × 2 bytes）。
- **搜索效率**：在 κ=512 的窗口内做 brute-force L2 搜索，每次搜索的复杂度为 O(κ·Hq·d)。MAC-Attention 将此搜索放在 fused persistent kernel 内部（in-kernel matching），利用共享内存/L2 cache 完成，避免 HBM 往返。
- **为什么不用余弦相似度**：L2 距离对向量的幅值（magnitude）敏感，而 pre-RoPE query 的幅值信息包含了 attention 分布的"集中度"信号（幅值大的 query 通常 attention 分布更集中），这对匹配质量有帮助。
- **为什么不用 post-RoPE**：post-RoPE query 包含了位置旋转分量，两个位置不同的 token 即使语义完全相同，post-RoPE query 也会因旋转角度不同而距离变大，破坏语义匹配的有效性。

涉及论文标题：
- MAC-Attention Match-Amend-Complete Attention for Efficient Long-Context Inference

## Sparse Expert Merging (稀疏专家合并 / Element-wise Expert Merging)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sparse Expert Merging（稀疏专家合并）是一种 MoE（Mixture-of-Experts）模型压缩技术，与传统的 **coarse-grained expert merging（粗粒度专家合并）** 不同，它不在整个 expert 层面合并权重，而是在**逐权重元素（element-wise）粒度**上选择性合并。核心思想是：两个 expert 的权重矩阵 W_i 和 W_j ∈ R^{d×h} 在部分维度上有共享知识（如通用语言建模的权重模式），在另一部分维度上则保留各自的专家专业化知识（如处理特定领域/语言模式的参数）。Sparse Expert Merging 通过构建 binary masks 在逐元素级别区分"可安全合并"和"必须保留"的权重条目——相似条目取平均（或加权合并），独有重要条目保留原 expert 的值。这种 fine-grained merging 同时保留了跨 expert 的共享知识和每个 expert 的独特专业化能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Sparse Expert Merging 的算法 pipeline（以 PuzzleMoE pairwise 合并为例）：
```
输入: W_i, W_j ∈ R^{d×h}  # 两个 expert 的权重矩阵
      X_i, X_j            # 校准集上的输入激活
      τ_sim = 0.4          # 相似阈值

# Step 1: 计算逐元素相似性（对称百分比差异）
Δ = ||W_i| - |W_j|| / (|W_i| + |W_j|)      # ∈ [0,1]，越小越相似
M^{sim} = 1_{Δ ≤ τ_sim}                      # similarity mask

# Step 2: 计算 saliency-based 互补掩码（Wanda 指标）
A_i = |W_i| ⊙ ‖X_i‖₂                        # 逐元素重要性
A_j = |W_j| ⊙ ‖X_j‖₂
M_i^{sal} = 1_{A_i ≥ A_j}                   # expert i 更重要的位置
M_j^{sal} = 1 − M_i^{sal}                   # expert j 更重要的位置

# Step 3: Dual-mask 构建与稀疏合并
M_i = M_i^{sal} ∨ M^{sim}                   # expert i 的最终 mask
M_j = M_j^{sal} ∨ M^{sim}                   # expert j 的最终 mask
W_{merged} = M^{sim} ⊙ (|W_i|+|W_j|)/2      # 相似条目取平均
           + (1−M^{sim}) ⊙ (M_i^{sal}⊙|W_i| + M_j^{sal}⊙|W_j|)  # 不相似条目取更 salient 的

# Step 4: 存储符号（推理时恢复方向）
S_i = 1_{W_i < 0},  S_j = 1_{W_j < 0}
# 存储: W_{merged}, M_i, M_j, S_i, S_j
# 推理重建: Ŵ_i = (-1)^{S_i} ⊙ M_i ⊙ W_{merged}
```

关键设计理由：
1. **相似性用对称百分比差异而非直接差值**：|W_i|-|W_j| 会因符号相反产生伪罚分——Δ 仅度量幅度相似性，符号单独存储和恢复。
2. **Pairwise 而非 k≥3 合并**：k 个 expert 联合合并时每个位置有 (2^k−1) 种选择，组合爆炸不可行。Pairwise 合并有 O(1) 闭式解和线性时间复杂度。
3. **随机分组足够好**：随机分组与搜索分组的平均准确率仅差 0.3pp，因为力量来自逐元素稀疏合并本身而非分组策略。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
1. **Offline 压缩阶段**：在 calibration dataset（如 C4 128 samples）上运行单次前向 pass，收集每个 expert 的输入激活。然后对每对 expert 计算 saliency + similarity masks，执行逐元素合并。PuzzleMoE 对 Mixtral-8x7B 的压缩仅需 2 分钟（vs D2 55min, HC-SMoE ~210min）。
2. **合并后存储**：合并后的 W_{merged} 替代原始两个 expert 权重，masks 和 signs 通过 bit-packing 嵌入 Bfloat16 未使用指数位（零额外存储）。
3. **推理时使用**：MoE routing gate 照常选择 top-k experts。对每个激活 expert，从 packed W_{merged} 中 on-the-fly 解码 mask 和 sign，重建对应 expert 的有效权重 Ŵ_i。
4. **跨模型适用性**：已验证 Mixtral-8x7B（8 experts）、Deepseek-MoE（64 experts）、Qwen1.5-MoE-A2.7B、Qwen3-MoE-30B-A3B。在 50% sparsity 下，Mixtral MMLU 仅下降 2.2pp（vs HC-SMoE 下降 18.9pp），Deepseek-MoE 下降 0.5pp（vs HC-SMoE 下降 20pp+）。
5. **与量化结合**：50% merging + 3-bit AWQ → 4.8× 总压缩（3.35 bit/weight），精度仅下降 1.7%。

涉及论文标题：
- PuzzleMoE

## Dual-Mask for MoE Expert Merging (双掩码专家合并：Similarity Mask + Saliency Mask)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual-Mask 是 PuzzleMoE 提出的 MoE 专家合并中的核心掩码设计，由两个互补的 binary mask 组成：(1) **Similarity Mask (M^{sim})**——基于逐元素权重幅度相似性，标识两个 expert 间"共享可合并"的权重条目；(2) **Saliency Mask (M_i^{sal}, M_j^{sal})**——基于 activation-aware 权重重要性，标识每个 expert 独有的、必须保留的关键权重。两个 mask 通过 OR 操作合并为每个 expert 的最终 mask（M_i = M_i^{sal} ∨ M^{sim}），实现"相似的部分取平均共享，不同的部分保留各自的专家知识"的选择性合并决策。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Dual-Mask 的决策逻辑可以理解为对每个权重条目 w_i[k,l] 和 w_j[k,l] 的三类处理：

**情况 1：相似且非关键差异**（Δ ≤ τ_sim）→ 两个 mask 都包含该条目 → 取幅度平均 (|w_i|+|w_j|)/2，推理时根据各自符号恢复方向。

**情况 2：不相似，expert i 更重要**（Δ > τ_sim 且 A_i ≥ A_j）→ M_i 包含、M_j 不包含 → W_{merged} 取 |w_i| 的值。推理时 expert i 激活该权重、expert j 该位置为 0（被 mask 掉）。

**情况 3：不相似，expert j 更重要**（Δ > τ_sim 且 A_j > A_i）→ M_j 包含、M_i 不包含 → 对称处理。

数学表达：
```
M^{sim} = 1_{Δ ≤ 0.4}                          # similarity: ≈40% 条目被标记为相似
M_i^{sal} = 1_{|W_i|⊙‖X_i‖₂ ≥ |W_j|⊙‖X_j‖₂}   # saliency: 约各 50%
M_i = M_i^{sal} ∨ M^{sim}                       # dual-mask: ~70% 条目对 expert i 有效
```

关键设计要点：
- **τ_sim=0.4 的选取**：0.3-0.5 范围内 PPL 最优。太小欠用相似性（合并不够）、太大过度合并（伤害专业化）。
- **符号分离**：相似性仅基于幅度（|W|），符号单独存储和恢复——避免正负抵消。
- **Saliency 用 Wanda 指标**：|W| ⊙ ‖X‖₂ 同时考虑权重幅度和激活范数，优于纯 magnitude-based 评分。
- **压缩后的 mask 存储**：通过 bit-packing 将 M_i, M_j, S_i, S_j 的 4 个 bit 嵌入 W_{merged} 的 Bfloat16 未使用指数位中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
1. **Calibration 阶段**：在 C4 128 samples（seq_len=2048）上运行单次前向 pass，收集每个 expert 输入激活的 L2 norm ‖X‖₂。
2. **Saliency 计算**：对每对 expert (E_i, E_j)，计算 A_i = |W_i| ⊙ ‖X_i‖₂ 和 A_j = |W_j| ⊙ ‖X_j‖₂，逐元素比较得到 M_i^{sal}, M_j^{sal}。
3. **Similarity 计算**：计算 Δ = ||W_i|−|W_j||/(|W_i|+|W_j|)，阈值 τ_sim=0.4 得到 M^{sim}。
4. **合并执行**：按上述公式逐元素合并得到 W_{merged}。合并复杂度 O(d×h) per pair。
5. **打包存储**：M_i, M_j, S_i, S_j 通过 bit-packing 嵌入 W_{merged} Bfloat16 指数位。
6. **推理重建**：CUDA kernel 中 on-the-fly 提取 mask/sign bits，重建 Ŵ_i = (-1)^{S_i} ⊙ M_i ⊙ W_{merged}。
7. **超参鲁棒性**：τ_sim=0.4 跨模型固定，校准集 C4 vs MATH 仅影响 0.1pp Avg Accuracy。

涉及论文标题：
- PuzzleMoE

## Wanda Metric (Activation-Aware Weight Importance, 激活感知权重重要性评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wanda（Pruning by **W**eights **a**n**d** **a**ctivations）是 Sun et al. (ICLR 2024) 提出的 LLM 后训练剪枝方法中的权重重要性评分指标：`Importance(W_{ij}) = |W_{ij}| × ‖X_{:,j}‖₂`，即每个权重的重要性等于其自身的绝对值 × 该权重对应输入通道的激活 L2 范数。核心洞察是：仅靠权重幅度（magnitude pruning）忽略了一个关键事实——LLM 中存在 emergent large-magnitude features，某些通道的激活值可比其他通道大 20× 以上。一个很小的权重如果乘以巨大的激活值，对输出的贡献可能远超一个大权重乘以小激活值。Wanda 通过乘积形式同时编码了权重的"局部重要性"和激活的"全局通道重要性"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Wanda 的使用流程（在 PuzzleMoE saliency mask 构造中）：
```
输入: W ∈ R^{d×h}, calibration dataset D (128 samples, seq_len=2048)

# Step 1: 收集激活统计量（单次前向 pass）
for each batch x in D:
    前向传播，记录每个线性层输入激活 X ∈ R^{N×d}
‖X‖₂ = mean over N samples of L2 norm per input channel  # shape: (d,)

# Step 2: 计算 per-weight importance
A = |W| ⊙ ‖X‖₂  # shape: (d×h)，逐元素乘积
# A[i,j] = |W[i,j]| × ‖X[i,:]‖₂  # i 通道的所有权重共享同一激活范数

# Step 3: 用于 pruning/saliency 决策
# 在 PuzzleMoE 中：比较两个 expert 的 A_i 和 A_j
M_i^{sal} = 1_{A_i ≥ A_j}  # expert i 更重要的位置
```

Wanda 的核心优势：
1. **速度**：仅需单次前向 pass（无需反向传播或权重更新），LLaMA-65B 50% sparsity pruning 仅需 5.6 秒（vs SparseGPT 1353 秒，~240× 加速）。
2. **无需微调**：pruning 后直接可用，不需要 fine-tuning 恢复精度。
3. **效果**：LLaMA-65B 50% sparsity 下 PPL 4.57（与 SparseGPT 的 4.57 持平），远超 magnitude pruning 的 5.90。
4. **在 PuzzleMoE 中的角色**：Wanda 指标用于构造 saliency mask——决定每个位置上哪个 expert 的权重更重要、应被保留。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
1. **Calibration data**：通常使用 C4 或 Wikitext2 的 128 samples（seq_len=2048），与下游任务无关。
2. **激活收集**：在 FP16/BF16 模型上运行前向 pass，记录每个线性层的输入激活。不使用梯度。
3. **Pruning 执行**：对每个权重矩阵，逐行（per output neuron）比较 Wanda 分数，保留每行 top-k%（如 50%）最重要的权重，其余置零。
4. **结构支持**：原生支持 unstructured sparsity（任意位置）和 N:M semi-structured sparsity（如 2:4，即每 4 个连续权重中最多 2 个非零）。
5. **PuzzleMoE 的扩展使用**：不是用于 pruning，而是用于 expert 间 saliency 比较——逐元素比较两个 expert 的 Wanda 分数，决定合并时保留谁的权重。
6. **开源**：https://github.com/locuslab/wanda

涉及论文标题：
- PuzzleMoE

## Global Output Channel-wise Mixed-Precision Quantization（全局输出通道混合精度量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Global Output Channel-wise Mixed-Precision Quantization 是 MixLLM（Zheng et al., MLSys 2026）提出的后训练量化策略：在所有线性层的所有 output channel 上全局统一排序 salience（该通道量化后对最终 loss 的贡献度），将 bit 预算集中分配到全局最敏感的 ~10% output channel（8-bit symmetric），其余 ~90% channel 使用 4-bit asymmetric 量化，同时激活固定 8-bit symmetric。这与 per-layer（各层独立决定精度比例）和 per-channel local fraction（各层内固定比例但不同层之间不可比较）的根本区别在于：MixLLM 的 salience 度量是跨层可比较的——"一个层的'重要'通道可能对最终 loss 的影响远小于另一个层的'不重要'通道"。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MixLLM 全局混合精度量化在 LLM 推理 pipeline 中的全流程（以 Llama 3.1 70B, W4.4A8 配置为例）：

```
# === Phase 1: 离线校准 — 全局 Salience 搜索 ===
Input: 所有线性层权重 {W_ℓ}, 校准集 D (Wikitext2 128 samples, seq_len=2048)
Output: 全局 precision map P: (ℓ, c) → {INT4, INT8}

for each sample d in D:
    # FP16 前向 + 反向，捕获 per-channel 梯度
    for each linear layer ℓ:
        g_ℓ,c = gradient of loss w.r.t. W_ℓ[c,:]  # per-channel grad
        # Fisher Information 的二阶 Taylor 近似累积
        # S_c = 1/|D| * Σ|g^T·Δ + 0.5·(g^T·Δ)²|
        # Δ = c_q - c_0 是量化误差向量
        S[ℓ,c] += |g_ℓ,c^T * Δ_c + 0.5 * (g_ℓ,c^T * Δ_c)^2| / |D|

# 全局排序：所有层的所有 output channel 按 S 降序统一排列
all_channels = [(ℓ, c, S[ℓ,c]) for all ℓ, c]
sort_descending(all_channels, key=salience)
top_k = int(len(all_channels) * 10 / 100)  # W4.4A8: 10% INT8

# === Phase 2: 量化执行 ===
for each ℓ, c:
    if rank((ℓ,c)) < top_k:
        # 8-bit symmetric per-group (group=128)
        quantize(W_ℓ[c,:], bits=8, symmetric=true, group=128)
    else:
        # 4-bit asymmetric per-group (group=128)
        quantize(W_ℓ[c,:], bits=4, symmetric=false, group=128)

# Activation: 8-bit symmetric per-group (group=128), 所有通道统一
```

**Salience 分布特征**（MixLLM Table 4）：v_proj 层 71.22% 通道 8-bit、down_proj 53.82%、gate_proj 仅 0.73%——bit 预算自动集中在最需要精度的层类型。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **搜索效率**：单 pass 方法 7B/8B 模型约 7 分钟，70B 模型约 55 分钟（单 A100）。迭代方法成本高但效果与单 pass 相当。
- **Output channel 维度的系统优势**：每个 output channel 子问题独立，INT4/INT8 kernel 通过 CUDA Graph 完全并行，fused epilogue scatter 开销 "basically costless"。
- **Calibration 数据**：Wikitext2 128 样本 2048 seq_len，小样本量即可获得稳定 salience 排序。
- **为什么 activation 固定 8-bit**：MatMul 计算受 weight 张量（大）约束大于 activation 张量（小）——weight 8→4 bit 提升 ~80% 计算强度，activation 8→4 bit 仅提升 ~5.88% 但精度损失大得多。

涉及论文标题：
- MixLLM: LLM Quantization with Global Mixed-Precision between Output and Embeddings

## Fisher Information Matrix for Quantization Salience（Fisher 信息矩阵量化敏感度估计）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fisher Information Matrix（FIM）在 MixLLM 中被用作量化敏感度估计工具：通过 FIM 近似 Hessian 矩阵，在不显式计算二阶导数的前提下，估计每个 output channel 量化后对模型 loss 的贡献。推导链：(1) Salience S_c = |l(c_q) - l(c_0)|，即该通道量化前后 loss 的绝对变化量；(2) Taylor 展开：l(c) ≈ l(c_0) + g^T·Δ + 0.5·Δ^T·H·Δ；(3) 用经验 Fisher F = 1/|D|·Σ g_d·g_d^T 近似 H（在局部最小值附近 H ≈ F）；(4) 代入得 S_c = 1/|D|·Σ|g_d^T·Δ + 0.5·(g_d^T·Δ)²|。MixLLM 的关键 insight：保留一阶项 g_d^T·Δ——因为量化后的 residual 分布不是零均值的，一阶项捕获了量化引入的偏向（bias）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Input: FP16 模型, 校准集 D
for each sample in D:
    # FP16 前向传播
    output = model.forward(x)
    loss = CrossEntropy(output, y)
    loss.backward()  # 获取 per-weight 梯度

    for each linear layer ℓ:
        g_ℓ = W_ℓ.grad  # [out_c, in_c] per-element 梯度
        for each output channel c:
            W_q = simulate_4bit(W_ℓ[c,:])  # 模拟 4-bit 量化
            Δ_c = W_q - W_ℓ[c,:]           # 量化误差向量 [in_c]

            # 一阶项: g^T·Δ (标量)
            term1 = dot(g_ℓ[c,:], Δ_c)

            # 二阶 Fisher 对角近似: (g^T·Δ)² (标量)
            term2 = term1^2

            # 累积 |一阶 + 0.5·二阶|
            S[ℓ,c] += abs(term1 + 0.5 * term2) / len(D)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **为何用 Fisher 而非真实 Hessian**：真实 Hessian 的存储和计算开销为 O((parameters)²)，对 LLM 不可行。Fisher `F = E[gg^T]` 仅需一阶梯度，在局部最小值附近是 Hessian 的良好近似（Lecun et al. "Optimal Brain Damage", Hassibi & Stork "Optimal Brain Surgeon"）。MixLLM 使用对角近似进一步压缩：`H_{ii} ≈ g_i²`，忽略 off-diagonal 跨元素交互。
- **PyTorch 实现**：通过 backward hook (`W_ℓ.register_hook`) 捕获梯度，前向时计算量化误差并累积 salience。小 batch 样本足够——仅需 128 样本。
- **与其他方法对比**：(1) AWQ 仅用 activation magnitude（不看梯度→不感知 loss impact）；(2) SpQR 的 Optimal Brain Surgeon 逐权重计算（更精细但更慢）；(3) MixLLM 的 Fisher 在开销（单 pass）和精度（考虑 loss）间取得实用平衡。
- **单 pass 的实证有效性**：MixLLM 验证单 pass Fisher 搜索与迭代（逐层重算真实 loss change）效果相当，证明全局 ordering 对一次近似已有足够信噪比。

涉及论文标题：
- MixLLM: LLM Quantization with Global Mixed-Precision between Output and Embeddings

---

## Activation Recomputation（激活重计算 / Gradient Checkpointing）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Activation Recomputation（也称 Gradient Checkpointing）是 Chen et al. (2016) 提出的 GPU 内存优化技术：forward pass 中不保存所有 activation tensors（O(L) 内存），仅保存部分 checkpoint layers；backward 需要 activation 时从最近 checkpoint 重新 forward 计算。这是时间换空间的 tradeoff——额外计算 ≈ 1× forward pass，但 activation 内存从 O(L) 降至 O(√L)。在 LLM 训练中几乎是标配（如 Megatron-LM `--recompute-activations`），因为 70B+ 模型的 activations 可达数百 GB 不 recompute 无法容纳。

从算法pipeline角度拆解，给出具体例子：

```
标准训练（无 recomputation）:
  Forward: 逐层执行，存储所有 A1, A2, ..., AN
  Backward: 从后向前读取 activations 计算梯度

Recomputation（checkpoint = every layer boundary）:
  Forward:
    A0 = X  (checkpoint)
    执行 layer_1.forward() → 得到 A1, 不存储（立即释放）
    执行 layer_2.forward() → 得到 A2, 不存储
    ...仅在 boundary 保存 checkpoint
  
  Backward:
    需要 A_{N-1} for layer_N.backward()
    → 从 checkpoint A_{N-2} recompute_forward(layer_{N-1}) → 得到 A_{N-1}
    → 执行 layer_N.backward(A_N, A_{N-1}, dAN)
    → A_{N-1} 用后释放
    → 需要 A_{N-2} → 从 checkpoint A_{N-3} recompute_forward(layer_{N-2}) → ...
```

**对内存碎片的影响**（STAlloc 关键关注）：
- 无 recomputation：activations 为 scoped tensors（forward alloc → backward free, LIFO），allocator 容易管理
- 有 recomputation：activations 变为 transient tensors（forward alloc → 立即 free → backward 时短暂 re-alloc → 立即 free），大量快速 alloc-free 产生碎片化空洞

STAlloc 的 temporal regularity 分类对此仍然有效：recomputation 将部分 scoped tensors 转为 transient tensors，但两种类型的 lifespan pattern 仍是 iteration-consistent 的。STAlloc 测得 PyTorch caching allocator 在 recomputation 下 memory efficiency 降至 ~60%（vs 无 recomputation 的 90%），但 STAlloc 通过 offline planning 仍保持 >95%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **PyTorch**：`torch.utils.checkpoint.checkpoint(fn, *args)` 
- **Megatron-LM**：`--recompute-activations` + `--recompute-granularity`（full/selective）
- **STAlloc 中的效果**：recomputation 使 dynamic allocator 的 reusable space 效果更好（static/dynamic peak 时间分离更明显），fallback 量下降 24.9%

涉及论文标题：
- Reducing GPU Memory Fragmentation via Spatio-Temporal Allocation Planning (STAlloc)

---

## Sparse-Linear Attention (SLA/SLA2，稀疏-线性混合注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparse-Linear Attention (SLA) 是一种用于加速 Diffusion Transformer (DiT) 中 attention 计算的混合方法，由 Zhang et al. (2025c) 提出。其核心思想是将全量 attention 矩阵 P = softmax(QKᵀ/√d) ∈ R^{N×N} 分解为两部分：P = P₁ + P₂，其中 P₁ = P⊙M（稀疏分支处理的 mask 选中位置）和 P₂ = P⊙(1-M)（线性分支处理的其余位置）。SLA 使用稀疏 softmax attention 计算 P₁（仅对 M=1 位置），使用线性 attention 近似 P₂（对 M=0 位置），输出 O = O_s + Proj(O_l)。

**SLA2 的改进**（本论文）：SLA2 识别出 SLA 的两个核心缺陷：(1) 稀疏分支产生的是行归一化后的 P_s = P₁/α（α = P₁·1），但实际需要的是 P₁V = α⊙O_s，导致缩放失配——SLA 通过 Proj(O_l) ≈ P₂V + (α-1)⊙O_s 来补偿，但线性分支负担过重；(2) 路由基于启发式 Top-k(pool(Q)pool(K)ᵀ) 而非优化目标。

SLA2 的解决方案：(I) 使用 O = α⊙O_s + (1-α)⊙O_l 直接匹配稀疏+线性分解，α ∈ R^{N×1} 可学习；(II) 可学习路由器 R(Q,K) = Top-k(k%, proj_q(Q̄)proj_k(K̄)ᵀ) 替代启发式选择，通过最小化 MSE(FullAttn, SLA2) 训练；(III) 在稀疏分支上叠加 QAT（低比特 attention）进一步加速。

在 Wan2.1-1.3B 和 Wan2.1-14B 视频 DiT 模型上，97% sparsity 达到 18.6× attention 加速，视频质量超越 90% sparsity 的 baseline。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

SLA2 算法 pipeline（单头 attention 前向）：

```
输入: Q, K, V ∈ R^{N×d}, 可学习参数 proj_q, proj_k ∈ R^{d×d}, α ∈ R^{N/b_q×1}
参数: b_q, b_k (block sizes), k% (sparsity)

# Step 1: 压缩与路由
Q̄ = pool(Q)                       # mean pooling: R^{N×d} → R^{N/b_q×d}
K̄ = pool(K)                       # R^{N×d} → R^{N/b_k×d}
P_c = softmax(proj_q(Q̄) @ proj_k(K̄)ᵀ / √d)   # 可学习路由分数
M_c = Top-k(k%, P_c)               # 每行 top-k% → 1, 其余 → 0

# Step 2: 稀疏 attention (M_c[i,j] = 1 的 block pair)
for i in 1..T_m (Q blocks), j in 1..T_n (KV blocks):
    if M_c[i,j] == 1:
        # QAT: 低比特量化 matmul
        Q̂_i, s_Q = quant(Q_i)       # FP16 → INT8/FP8
        K̂_j, s_K = quant(K_j)
        S_ij = dequant(Q̂_i @ K̂_jᵀ, s_Q, s_K) / √d
        P_ij = softmax(S_ij)        # online rescaling with m,l
        P̂_ij, s_P = quant(P_ij)
        V̂_j, s_V = quant(V_j)
        O_i^s += dequant(P̂_ij @ V̂_j, s_P, s_V)

# Step 3: 线性 attention (M_c[i,j] = 0)
for i in 1..T_m, j in 1..T_n:
    if M_c[i,j] == 0:
        H_i += (K_j^φ)ᵀ @ V_j     # 局部 KᵀV 累加 (O(Nd²))
        Z_i += rowsum((K_j^φ)ᵀ)   # 归一化因子累加
O_i^l = Q_i^φ @ H_i / (Q_i^φ @ Z_i)  # 线性 attention 输出

# Step 4: α-组合
O = α ⊙ O^s + (1-α) ⊙ O^l
```

训练两阶段：
- Stage 1: MSE(FullAttn(Q,K,V), SLA2(Q,K,V)) 训练 R 和 α，使用 SoftTop-k 替代硬 Top-k 保证梯度可传
- Stage 2: 端到端扩散 loss 微调，R 固定，α 继续优化，使用硬 Top-k（与推理一致）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现**：基于 FlashAttention 的 block-wise CUDA kernel。稀疏路径在 Tensor Core 上执行低比特 matmul（INT8/FP8），线性路径用累加方式避免 O(N²) 的 QKᵀ。反向手动推导 Q,K,V,Q^φ,K^φ 的梯度（Algorithm 3），其余参数用 PyTorch autograd。
- **使用**：替换扩散模型中所有 attention 层为 SLA2。推理时传入训练好的 α 和路由器参数 R，对每个 attention 层在线计算 M_c 和执行 SLA2 前向。
- **开源**：SLA v1 代码在 https://github.com/thu-ml/SLA，SLA2 代码尚未公开。
- **适用场景**：视频/图像 DiT 模型的 attention 加速。sparsity 越高加速越大（90%→97%）。需要微调（500 steps），不能零样本使用。

涉及论文标题：
- SLA2: Sparse-Linear Attention with Learnable Routing and QAT

---

## Learnable Router for Attention Routing（注意力可学习路由器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Learnable Router for Attention Routing 是 SLA2 提出的用于动态决定 attention 矩阵中每个位置应走稀疏分支还是线性分支的可训练模块 R(Q,K)。路由器以 Q 和 K 为输入（V 不影响 attention 权重的形成，因此不需要），输出二值 mask M ∈ {0,1}^{N×N}：M=1 表示走稀疏 attention（精确计算），M=0 表示走线性 attention（低秩近似）。

路由器的设计包含三个关键要素：
1. **压缩**：对 Q、K 沿序列维度做 mean pooling（block size b_q, b_k），将 N×d 压缩到 (N/b_q)×d 和 (N/b_k)×d，将路由计算复杂度从 O(N²) 降至 O(N²/(b_q·b_k))。利用 DiT 中相邻 token 分布相似的特性（smooth QKᵀ 变化）。
2. **可学习投影**：proj_q, proj_k ∈ R^{d×d} 对压缩后的 Q̄、K̄ 做线性变换，学习任务自适应的表示空间。设置 proj_q = proj_k = I 可恢复为 SLA 的启发式路由。
3. **Top-k 选择**：对 P_c = softmax(proj_q(Q̄)proj_k(K̄)ᵀ/√d) 每行取 top-k% 位置为 M=1。训练时用 SoftTop-k 保证梯度可传，推理时用硬 Top-k。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

路由器 R 的计算流程：

```
输入: Q, K ∈ R^{N×d}, proj_q, proj_k ∈ R^{d×d}
参数: b_q, b_k (pooling block sizes), k% (top-k ratio)

# 1. 压缩
Q̄ = pool_b_q(Q)          # R^{N×d} → R^{N/b_q×d}, 每 b_q 个 token 取均值
K̄ = pool_b_k(K)          # R^{N×d} → R^{N/b_k×d}

# 2. 可学习投影 + 路由分数
P_c = softmax(proj_q(Q̄) @ proj_k(K̄)ᵀ / √d)   # R^{N/b_q × N/b_k}

# 3. Top-k (训练用 SoftTop-k, 推理用硬 Top-k)
# 训练:
M_c = SoftTop-k(k%, P_c)  # σ(P_c/τ + λ_i), λ_i 通过二分搜索确保每行 sum = k%×N/b_k

# 推理:
M_c = HardTop-k(k%, P_c)  # 每行取 top-k% 设为 1, 其余为 0

# 4. 展开
M = expand(M_c)           # R^{N/b_q×N/b_k} → R^{N×N}
```

路由器仅依赖 Q、K（不依赖 V），因为路由决定的是 P = softmax(QKᵀ/√d) 中哪些位置贡献大，而 P 完全由 QKᵀ 决定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **训练**：Stage 1 用各层的 Q,K,V 构造数据集 D，最小化 L = MSE(FullAttn(Q,K,V), SLA2(Q,K,V,k%,R,α)) 训练 R 参数（proj_q, proj_k）和 α。使用 SoftTop-k 保证梯度可传。
- **推理**：R 参数固定，使用硬 Top-k 在线计算 M_c。在 FlashAttention block-wise kernel 中，每个 block pair 根据 M_c[i,j] 的值选择稀疏或线性执行路径。
- **设计动机**：相比于 SLA 的启发式 Top-k(pool(Q)pool(K)ᵀ)，可学习投影允许路由器学习哪些 Q-K 交互模式更适合稀疏计算——例如，将某些权重从 P₁ 移到 P₂ 可能不增加 P₂ 的秩（因为 P₂ 已有相似模式），但能提高 P₁ 的稀疏度。

涉及论文标题：
- SLA2: Sparse-Linear Attention with Learnable Routing and QAT

---

## SoftTop-k Operator（可微分 Top-k 算子）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SoftTop-k 是 Top-k 选择算子的可微分近似，由 Ding et al. (2024) 在 SMART Pruner 中提出。标准的 Top-k 操作不可微（离散选择阻断梯度传播），SoftTop-k 通过 sigmoid + 温度参数 + 二分搜索约束将离散选择松弛为连续软选择，使得梯度可以通过 Top-k 传递。

SoftTop-k 的定义（SLA2 中的使用）：

$$\text{SoftTop-k}(k\%, P_c)_{ij} = \sigma\left(\frac{(P_c)_{ij}}{\tau} + \lambda_i\right)$$

其中：
- σ 是 sigmoid 函数，将值映射到 (0,1)
- τ 是温度参数（SLA2 使用 τ=0.1），越小越接近硬 Top-k
- λ_i 是通过二分搜索确定的每行偏置，确保每行的 SoftTop-k 输出之和恰好为 k% × N/b_k（保留 Top-k 的行约束特性）
- 梯度通过重参数化技巧（reparameterization trick）传播

SoftTop-k 输出是 [0,1] 连续值，在 SLA2 中用作注意力 mask 的软版本——值接近 1 的位置走稀疏分支贡献大，接近 0 的走线性分支。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
输入: P_c ∈ R^{B×N}, k%, τ=0.1
输出: M_soft ∈ R^{B×N}, 每行和 = k% × N

# 对每行 i:
target_sum = k% × N
λ_low, λ_high = -100, 100            # 二分搜索边界
while λ_high - λ_low > 1e-6:
    λ_mid = (λ_low + λ_high) / 2
    row_sum = sum(σ(P_c[i,:]/τ + λ_mid))
    if row_sum > target_sum:
        λ_low = λ_mid                 # 降低 λ 减少激活
    else:
        λ_high = λ_mid                # 提高 λ 增加激活

M_soft[i,:] = σ(P_c[i,:]/τ + λ_low)
```

Forward: M_soft[i,j] ∈ (0,1)，梯度通过 σ'(x) = σ(x)(1-σ(x)) 传播。
Backward: ∂L/∂(P_c)_{ij} = ∂L/∂M_soft_{ij} · σ'(·)/τ。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **使用场景**：SLA2 的 Stage 1 训练中替代硬 Top-k，使路由器参数 proj_q、proj_k 能接受梯度。Stage 2 和推理时换回硬 Top-k。
- **温度 τ 的作用**：τ→0 时 SoftTop-k → HardTop-k（但梯度消失）；τ 大时软选择平滑但偏离 Top-k 行为。SLA2 使用 τ=0.1 平衡。
- **参考**：Ding et al., "Separate, Dynamic and Differentiable (SMART) Pruner", 2024, https://arxiv.org/abs/2403.19969

涉及论文标题：
- SLA2: Sparse-Linear Attention with Learnable Routing and QAT

---

## Linear Attention（线性注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Linear Attention 是一类将标准 attention 的 O(N²d) 复杂度降至 O(Nd²) 的注意力机制。标准 attention 计算 O = softmax(QKᵀ/√d)V，需要先计算 N×N 的注意力分数矩阵（QKᵀ），再与 V 相乘——两步的复杂度分别为 O(N²d) 和 O(N²d)。Linear Attention 通过引入 kernel 函数 φ(·) 将 softmax attention 近似为：

$$O = \frac{\phi(Q)(\phi(K)ᵀV)}{\phi(Q)(\phi(K)ᵀ\mathbf{1})}$$

利用矩阵乘法结合律，先计算 KᵀV ∈ R^{d×d}（O(Nd²)），再左乘 Q（O(Nd²)），避免了显式的 N×N 注意力矩阵。主要 linear attention 变体包括：Performer（φ=exp 的随机 Fourier 特征近似）、Linear Transformer（φ=elu+1）、Linformer（低秩投影）等。

在 SLA2 中，linear attention 用作稀疏 attention 的补充——对 mask M=0 的位置用 linear attention 近似，避免丢失这些位置的全局信息。SLA2 使用 φ = softmax（不近似的准确 kernel 函数），归一化为 norm(φ(Q)φ(K)ᵀ⊙(1-M)) 确保行和为 1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

SLA2 中 linear attention 的计算（仅处理 M=0 的位置）：

```
输入: Q, K, V ∈ R^{N×d}, M ∈ {0,1}^{N×N} (sparse mask)
Q^φ = softmax(Q)                  # kernel 函数
K^φ = softmax(K)

# 利用结合律: 先算 KᵀV (O(Nd²)), 再乘 Q (O(Nd²))
H = (K^φ)ᵀ @ ((1-M) ⊙ V)         # R^{d×d}: 加权 KᵀV, 仅 M=0 的 V 参与
Z = (K^φ)ᵀ @ (1-M) @ 1           # R^{d×1}: 归一化因子, 1 为全 1 向量
O_l = (Q^φ @ H) / (Q^φ @ Z)       # R^{N×d}: 线性 attention 输出
```

在 SLA2 的 block-wise kernel 中，线性路径不计算完整的 H 和 Z，而是累加局部块：
```
for each KV block j where M_c[i,j] == 0:
    h_j = (K_j^φ)ᵀ @ V_j           # 局部 KᵀV
    z_j = rowsum((K_j^φ)ᵀ)         # 局部归一化
    H_i += h_j; Z_i += z_j          # 累加
O_i^l = Q_i^φ @ H_i / (Q_i^φ @ Z_i)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **优势**：训练和推理均为 O(Nd²)，适合长序列（N >> d）。但近似精度通常不如 softmax attention，尤其在需要精确位置关系时。
- **SLA2 的使用**：linear attention 不与 sparse attention 竞争，而是互补——sparse attention 负责高权重的 key 位置（精确计算），linear attention 负责低权重的全局上下文（近似捕获）。这种分工使 sparsity 可达 97% 而质量不降。
- **局限性**：单独使用 linear attention 在视频生成中通常质量不足（SANA、Dig 仅对图像生成有效）。SLA2 的稀疏+线性混合克服了这一限制。

涉及论文标题：
- SLA2: Sparse-Linear Attention with Learnable Routing and QAT

---

## Diffusion Transformer (DiT)（扩散 Transformer）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Diffusion Transformer (DiT) 是将 Transformer 架构用作扩散模型（Diffusion Model）去噪骨干网络的生成模型架构。传统扩散模型（如 Stable Diffusion）使用 U-Net 作为去噪网络，而 DiT 用纯 Transformer 替代 U-Net——将噪声图像/视频的 latent patch 序列化为 token 序列，通过多层 Transformer block（self-attention + FFN + adaptive layer norm）逐 timestep 去噪。

DiT 的核心组件：
- **Patch embedding**：将 latent z ∈ R^{C×H×W}（或视频 z ∈ R^{C×T×H×W}）分 patch 后线性投影为 token 序列
- **Timestep conditioning**：扩散 timestep t 通过 adaptive layer norm（AdaLN）注入每层——scale + shift 参数由 t 的 MLP 嵌入生成
- **Self-attention**：标准 scaled dot-product attention（视频 DiT 通常加 spatial-temporal 分解或 3D full attention）
- **FFN**：标准 position-wise MLP

SLA2 的目标模型是 Wan2.1-T2V（Text-to-Video），一个基于 DiT 的视频生成模型。Wan2.1 使用 3D full attention（所有 spatial-temporal token 互相 attend），序列长度 N 极大（480P 约 1560 tokens, 720P 更大），attention 成为计算瓶颈。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

DiT  video 生成 pipeline（以 Wan2.1-T2V 为例）：

```
输入: text prompt, 随机噪声 z_T ∈ R^{C×T×H×W}, timesteps T..1

# 文本编码
c = T5-XXL(prompt)                               # text embedding

for t in T, T-1, ..., 1:
    # 1. Patch + position embedding
    x = patch_embed(z_t)                           # R^{N×d}, N = T×H×W / patch_size²
    
    # 2. DiT blocks (重复 L 层)
    for layer in 1..L:
        # Adaptive layer norm (timestep conditioning)
        scale, shift = MLP(t_embedding)            # 从 timestep t 生成
        x_norm = AdaLN(x, scale, shift)
        
        # Self-attention (所有 N 个 token 互相 attend) ← SLA2 加速目标
        x_attn = SLA2(x_norm)                      # 替代标准 attention
        
        # FFN
        x = x_attn + FFN(x_norm)
    
    # 3. Unpatch + 去噪一步
    z_{t-1} = denoise_step(unpatch(x), z_t, t)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现**：基于 PyTorch + FlashAttention（标准 attention）或 SLA2 kernel（稀疏 attention）。Wan2.1 的 DiT 层使用 3D full attention 在 spatial-temporal token 上，序列长度 N = (T/p_t)×(H/p_s)×(W/p_s)。
- **训练**：扩散 loss（噪声预测 MSE），在大规模视频数据集上预训练+在高质量子集上微调。SLA2 的微调在此基础上做 500 steps，batch 64 (1.3B) / 15 (14B)。
- **推理**：多步去噪（通常 50 步），每步调用 DiT 前向。SLA2 加速 attention 后，1.3B 模型端到端延迟从 97s 降至 7s（13.9×），14B 模型延迟降低 4.35×。
- **开源模型**：Wan2.1-1.3B/14B 开源（Apache 2.0），https://github.com/Wan-Video/Wan2.1

涉及论文标题：
- SLA2: Sparse-Linear Attention with Learnable Routing and QAT

---

## α-Combination (Learnable Ratio for Sparse + Linear Attention Fusion，可学习稀疏-线性注意力融合比例)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

α-Combination 是 SLA2 提出的稀疏 attention 和线性 attention 输出的可学习融合方式：O = α⊙O_s + (1-α)⊙O_l，其中 α ∈ R^{N×1}（0 < α < 1）是每行可学习的融合比例向量。这一公式直接匹配 full attention 的稀疏-线性分解 P ≈ α⊙P_s + (1-α)⊙P_l：α⊙P_s 对应 P₁（mask 选中部分），(1-α)⊙P_l 对应 P₂（其余部分），并且 α⊙P_s + (1-α)⊙P_l 自动行归一化。

与 SLA 的对比：
- **SLA**: O = O_s + Proj(O_l)，其中 Proj ∈ R^{d×d} 需要同时补偿 (α-1)⊙O_s（稀疏分支缩放失配）和 P₂V（线性分支近似误差）
- **SLA2**: O = α⊙O_s + (1-α)⊙O_l，α 直接补偿缩放失配，线性分支仅需近似 P₂V

α 在 Stage 1 通过 MSE(FullAttn, SLA2) 初始化，在 Stage 2 与扩散模型端到端微调。α 的每行独立使不同 query token 可以有不同比例的稀疏/线性混合。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# SLA2 output fusion (Algorithm 2, lines 26-27)
O^s = {O_i^s}                     # sparse attention output, N×d
O^l = {O_i^l}                     # linear attention output, N×d

# α 对每 b_q 行的 block 共享一个标量值
# 展开到 N×1
α_expanded = repeat(α, b_q)       # R^{N/b_q×1} → R^{N×1}

# 逐行 weighted sum
O = α_expanded ⊙ O^s + (1 - α_expanded) ⊙ O^l
```

数学上等价于：
$$O_i = \alpha_i \cdot O_i^s + (1-\alpha_i) \cdot O_i^l, \quad \forall i \in [1, N]$$

其中 O_i^s 和 O_i^l 均已行归一化（每行和为 1），所以 O_i 也自动行归一化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- α 的维度为 R^{N/b_q×1}（而非 R^{N×1}），因为 α 的粒度与路由 mask M_c 的粒度一致（block-wise），展开后逐行应用到完整序列。
- α 值在 [0,1] 之间（由 sigmoid 或 clamp 确保）。α → 1 表示更依赖稀疏 attention（更精确但更贵），α → 0 表示更依赖线性 attention（更高效但更近似）。
- 训练中 α 是可学习的——模型自适应决定每个 query 需要多少精确计算。

涉及论文标题：
- SLA2: Sparse-Linear Attention with Learnable Routing and QAT

## Mixture-of-Experts (MoE / 混合专家模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixture-of-Experts（MoE）是一种神经网络架构范式，它将传统的单一前馈网络（FFN）替换为多个独立的"专家"子网络和一个门控网络（gating network/router）。与 dense LLM 中所有 token 经过同一个 FFN 处理不同，MoE 模型对每个 token 只激活一小部分专家（top-k gating），从而在增加总参数容量的同时保持较低的逐 token 计算量。MoE 层由三部分组成：(1) 多个 expert FFN（每个 expert 是一个独立的前馈子网络，典型为 SwiGLU 结构）；(2) 一个 gating network（小型线性层 + softmax），为每个 token 计算 expert 选择分数；(3) 一个聚合单元，将各 expert 的输出加权求和得到 MoE 层最终输出。MoE 的核心思想是条件计算（conditional computation）：不同 token 使用不同的参数子集，从而实现参数容量与计算成本的解耦。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MoE layer 的算法 pipeline 如下（以 top-k gating, k=4 为例）：

```
输入: x ∈ R^{B×d}                        # B 个 token, 维度 d
      W_gate ∈ R^{d×E}                   # gating 权重, E 个 experts
      experts = [FFN_1, ..., FFN_E]      # E 个 expert FFN

# Step 1: Gating (Router)
scores = x @ W_gate                       # [B, E], raw logits
topk_vals, topk_idx = topk(softmax(scores), k=4)  # 每个 token 选 top-4

# Step 2: Expert Dispatch (token-to-expert assignment)
for e in 1..E:
    tokens_e = {x_i | e in topk_idx[i]}   # 分配给 expert e 的 token 集合
    N_e = len(tokens_e)
    if N_e == 1:   # GEMV: [1, d] x [d, d_ff] → [1, d_ff]
        out_e = GEMV(tokens_e, W1_e) → gelu → GEMV(intermediate, W2_e)
    elif N_e > 1:  # GEMM: [N_e, d] x [d, d_ff] → [N_e, d_ff]
        out_e = tokens_e @ W1_e → gelu → @ W2_e   # grouped GEMM

# Step 3: Aggregation
for i in 1..B:
    output[i] = sum(topk_vals[i][j] * out_{topk_idx[i][j]} for j in 1..k)
```

关键特征：(1) 每个 token 的计算量 ≈ k × FFN_cost（非 E × FFN_cost）；(2) token-to-expert 分配由 gating 动态决定，是 input-dependent 的稀疏激活；(3) shared expert 接收所有 token（如 Qwen3-Next），总是 compute-bound。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **Gating 方式**：top-k gating（最常见，k=1~8）通过 softmax 后取 top-k；auxiliary load balancing loss 防止 expert collapse。DeepSeekMoE 使用 shared + routed expert 混合设计，shared expert 处理所有 token。
- **Expert FFN 结构**：典型为 SwiGLU（d → d_ff → d_ff → d），其中 d_ff 通常为 4d 或 8d/3（SwiGLU activation）。
- **推理优化**：grouped GEMM 将多个 expert 的 MatMul 合并为一个 kernel；all-to-all communication 用于跨 GPU token dispatch（expert parallelism）；异构 expert placement（如 Sieve 将 expert 动态分到 GPU/PIM）。
- **主要模型**：Mixtral-8x7B/8x22B、DeepSeek-V2/V3/R1、Qwen3/Qwen3-Next、GPT-OSS-120B、Gemini 2.5。

涉及论文标题：
- Sieve

## Arithmetic Intensity (算术强度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Arithmetic Intensity（算术强度，AI）定义为执行一个计算操作时每字节内存传输所对应的浮点运算次数（FLOPs/Byte），即 AI = Total FLOPs / Total Bytes Transferred。它是 Roofline Model 的核心参数，用于判断一个操作是 compute-bound（高 AI）还是 memory-bound（低 AI）。在 MoE 模型 PIM offloading 场景中，Sieve 论文指出：每个 expert 的算术强度由分配给它的 token 数 N 决定——1 token → GEMV（AI 极低，memory-bound），多 token → GEMM（AI 高，compute-bound）。B200 GPU 的 compute/memory ridge point 约在 ~250 ops/byte，而 HBM-PIM 的 ridge point 约在 ~10-20 ops/byte（因内部带宽极高但计算能力有限）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 MoE expert FFN 为例计算算术强度（W1 ∈ R^{d×d_ff}, W2 ∈ R^{d_ff×d}, N 个 token, 忽略激活函数和中间激活）：
```
total_FLOPs = 2*N*d*d_ff + 2*N*d_ff*d ≈ 4*N*d*d_ff
total_bytes = (N*d + d*d_ff) + (d_ff*d + N*d_ff)  # input + weights
            ≈ 2*d*d_ff + N*(d + d_ff)
AI(N) = 4*N*d*d_ff / (2*d*d_ff + N*(d+d_ff))
```
N=1 → AI ≈ 2（仅 2 FLOPs/byte, memory-bound on GPU）
N=64 → AI ≈ 100+（compute-bound on GPU）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Sieve 用算术强度指导专家 placement：运行时根据每个 expert 的 token count N 计算 AI，低 AI（N≤2, GEMV）→ PIM（利用 PIM 高内部带宽），高 AI（N≥5, GEMM）→ GPU（利用 GPU Tensor Core 高吞吐）。注意：roofline 模型忽略 DRAM 时序开销（tRCD, tRP, tRFC, bank contention），可能高估 PIM 上 GEMV 性能 1.8–4.2×。Sieve 使用运行时 cost table（实测 GEMV time exponential moving average）而非纯 roofline 估计。

涉及论文标题：
- Sieve

## Bimodal Expert Distribution (双峰专家分布)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Bimodal Expert Distribution（双峰专家分布）是 Sieve 论文识别的现代 MoE 模型关键特征：在一次 batch 推理中，token-to-expert 分配呈极端双峰形态——少数"热门"专家被大量 token 选中（形成 GEMM），大量"冷门"专家仅被 1 个或极少数 token 选中（退化为 GEMV）。量化数据（batch=64）：Qwen3-Next 中 44.2% experts 仅收 1 token（GEMV expert），89.3% 收 ≤4 tokens；GPT-OSS 中 32.6% experts 仅收 1 token。形成原因：(1) load balancing loss 不能完全消除分配不均；(2) 某些 expert 学到的模式更通用；(3) 现代 MoE 趋势（更多 expert 总数 + 更少激活 expert 数）加剧尾部分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# token-to-expert 分配后统计每个 expert 的 token 数
for each expert e in 1..E:
    N_e = count(tokens assigned to e)

# 双峰分布特征
popular_experts   = {e | N_e >= 5}   # ~10-20%, GEMM (高算术强度)
unpopular_experts = {e | N_e <= 4}   # ~80-90%, GEMV/skinny GEMM (低算术强度)
GEMV_experts      = {e | N_e == 1}   # ~20-44% (取决于模型和 batch size)

# Sieve 调度: N_e 小的 → PIM, N_e 大的 → GPU
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
双峰分布是 Sieve Scheduler 的核心理念基础：将 N_e 小的 unpopular experts（memory-bound GEMV）offload 到 PIM，N_e 大的 popular experts（compute-bound GEMM）留在 GPU。早期 MoE（Mixtral-8x7B, batch≥16）在大 batch 下 expert 分布趋于均匀，不存在明显双峰分布。双峰分布是现代更稀疏 MoE（Qwen3、GPT-OSS、DeepSeek-V3 在 batch≤256）的特征。Sieve 将双峰分布从"需要被 load balancing 消除的问题"重新定义为"可被异构硬件分工利用的机会"。

涉及论文标题：
- Sieve

## Arbitrary Bit-Width Low-Precision Data Types for GPU Computation (任意位宽低精度数据类型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Arbitrary Bit-Width Low-Precision Data Types 是指在 GPU 上高效支持**1-8 bit 任意位宽**的低精度数据类型，包括有符号整数、无符号整数和浮点数格式。Tilus 是首个在单一框架内覆盖此全谱的编程语言/编译器，支持的位宽为：

- **无符号整数**：uint1, uint2, uint3, uint4, uint5, uint6, uint7, uint8
- **有符号整数**：int2, int3, int4, int5, int6, int7, int8
- **浮点数**：float3, float4, float5, float6, float7, float8——支持任意 exponent/mantissa 分布，如 e4m3 (1 sign + 4 exp + 3 mantissa = 8 bit), e3m3, e3m2, e2m2, e2m1, e1m1 等。

这填补了 LLM 量化中的一个关键空白：4-bit 量化（如 int4）在某些模型上仍有精度损失，而 8-bit（如 int8）又浪费带宽。5-7 bit 量化（如 eXmY [3], QuantLLM [60]）可以在精度和效率之间找到更好的平衡点，但此前缺乏高效的 GPU kernel 支持。

从算法pipeline角度拆解术语：

这些低精度数据类型在 LLM 推理 pipeline 中的使用方式：

```
# LLM Decode 阶段的一个 Transformer Layer
输入: hidden_states ∈ FP16[B, D]  # B=batch_size, D=hidden_dim

# ---- Linear 1: QKV Projection (使用低精度权重) ----
W_qkv ∈ INT4[B, D, 3D]  # 权重以 4-bit 存储, 减少 4× DRAM 占用
# Step 1: Weight Loading
W_tile = LoadLowPrecision(W_qkv, dtype=INT4)  # 从 global memory 加载
# Step 2: Dequantize + Cast (在寄存器内完成)
W_f16 = FastCast(W_tile, FP16)  # PRMT/LOP3 向量化转换
# Step 3: MatMul on Tensor Core
QKV = MatMul(hidden_states, W_f16)  # FP16 × FP16 MMA

# ---- Linear 2: Attention Output Projection ----
W_o ∈ INT6[B, D, D]  # 使用 6-bit 精度恢复更多信息
# ... (同上流程, 但 casting 路径为 INT6→FP16)

# ---- Linear 3: FFN Up Projection ----
W_up ∈ FP6_E3M2[B, D, 4D]  # 6-bit float (e3m2), 更好的动态范围
# ... (同上, 但 casting 路径不同——需要处理浮点格式的 exponent/mantissa)
```

**为什么不同层可以使用不同位宽**：attention 的 QKV/O 层对精度更敏感（影响 attention score 分布），可能使用更高位宽（如 int6）；FFN 的 up/gate 层相对不敏感，可以使用更低精度（如 int4）。Tilus 的单一 program template 通过参数化 tile sizes 统一覆盖所有这些组合。

**Compact Storage 原理**：亚字节数据（bit_width < 8）紧凑打包在 u8 字节数组中——元素之间无 padding bit，单个元素可能跨越两个连续 u8 字节的边界。例如 6 个 int6 值占用 6×6÷8 = 4.5 ≈ 5 bytes（最后 4 bit 为 padding）。访问时通过 bitwise 操作（AND 提取 + SHIFT 对齐 + OR 拼接）还原。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Tilus 的任意位宽支持通过以下机制实现：
1. **Weight Layout Transformation**：在 kernel 启动前将权重从原始 dtype 变换为 u8 紧凑格式。通用规则：每线程 n bytes → reinterpret 为 `u8, local(n₂).spatial(T).local(n₁)`，其中 n₁=gcd(n,16), n₂=n/gcd(n₁,16)。
2. **Register Tensor Reinterpretation**：通过 `View()` 指令在寄存器中将 u8 张量无代价 reinterpret 为目标低精度 dtype（利用布局代数的除法性质）。
3. **Vectorized Casting**：PRMT + LOP3 + bitwise 指令将低精度值转换为 FP16/BF16/INT8 标准类型。
4. **Fallback Mechanism**：对于极端情况（如 uint1 的 bit-level packing），使用通用 bitwise 路径作为后备方案。

涉及论文标题：
- Tilus: A Tile-Level GPGPU Programming Language for Low-Precision Computation

---

## Commutativity Hypothesis in LLM Inference（LLM 推理中的交换性假设）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Commutativity Hypothesis（交换性假设）是 Span Query 论文的核心理论前提：**若两个 token span 的语义不依赖于它们在序列中的相对顺序，则它们的 KV cache pages 可以任意排列和复用，且不影响模型输出的正确性**。形式化表达为：若 span A 和 span B commute（AB=BA），则物理 page 中 A 的 KV 值可被任意不同 context 的请求以不同 virtual position 复用——因为"order doesn't matter"。这是对传统 LLM serving 中"position=physical layout"假定的根本性挑战。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
交换性假设的判定与利用流程：
```
输入: Expression Tree T（含 span 节点和依赖关系）

判定 Commutativity:
  for each pair of sibling spans (S_i, S_j):
      if S_i 的输出不是 S_j 的输入 AND S_j 的输出不是 S_i 的输入:
          S_i 和 S_j commute → 可标记为 ++(S_i, S_j)

典型场景:
  - Chat messages: SYS + USER + ASSISTANT → 不 commute（消息顺序含语义）
  - RAG documents: ++(doc1, doc2, ..., docN) → commute（文档独立，检索排序无关）
  - Multi-agent generators: ++(G(p1), G(p2)) → commute（并行生成，输出独立）
  - Judge-Generator: J(G(p)) → J 依赖 G 的输出 → 不 commute

利用 Commutativity 进行 KV cache 复用:
  请求1: R(query1, ++(docA, docB))
    → docA pages 在 physical addr P1, docB pages 在 physical addr P2
  
  请求2: R(query2, ++(docA, docB))  // 不同 query，相同 docs
    → Baseline (linear prefix): query2 ≠ query1 → prefix mismatch → cache miss for docs
    → Span Query: docA 和 docB 的 physical pages (P1, P2) 可直接复用
      因为 RoPE on read + span hash chaining 使 page 与 position 解耦
      query token 从新位置开始（仅需新分配 query pages）
  
  Cache 复用率: docs 部分 = 100%（vs baseline 中 ~0% 当 query 不同时）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现需要两个关键机制配合：(1) **RoPE on read**——若 RoPE 在 KV cache 写入时施加，同一 page 被不同 context 复用时 position encoding 错误。RoPE on read 使 position 在 attention 计算时动态注入，page 本身 position-free。(2) **Prefix hash chaining selective disable**——vLLM 的 prefix caching 依赖 block hash（hash(token_content + prior_block_hash)）。在 span 边界暂停 hash accumulation（以 `(` 开头的 block），使 span 内 pages 的 hash 不依赖前序 span——从而使不同 query 后的 docs 仍被识别为"相同 content"并 cache hit。Commutativity Hypothesis 将 LLM serving 的 cache 复用从"prefix-exact-match"升级为"commutative-subtree-reuse"，是 Span Query 获得 10-20× TTFT 加速的理论基础。

涉及论文标题：
- Using Span Queries to Optimize Cache and Attention Locality

## RoPE on Read（读取时 RoPE / 解耦位置编码）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RoPE on Read 是 Span Query 论文对 vLLM 中 Rotary Position Embedding（RoPE）施加时机的关键修改：将 RoPE 从 KV cache **写入时**（传统 vLLM 做法）移到 KV cache **读取时/attention 计算时**。这使 KV cache page 本身不含 position encoding，position 在 attention 计算阶段动态根据该 page 在当前 span context 中的 virtual position 注入。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
传统 RoPE on Write vs Span Query RoPE on Read：
```
# === 传统 vLLM (RoPE on Write) ===
def write_kv_cache(token_embedding, position):
    K = apply_rope(K_linear(token_embedding), position)  # RoPE 在写入时施加
    V = V_linear(token_embedding)
    kv_cache[page][offset] = (K, V)  # K 已含 position info

def read_kv_cache(page, offset):
    K, V = kv_cache[page][offset]
    return K, V  # K 已含 position，直接使用

# 问题: 同一 page 被不同 context 复用时 position 固定
#   docA 在请求1的 position=[16,32)，在请求2的 position=[48,64)
#   → K 中编码的 position 错误


# === Span Query (RoPE on Read) ===
def write_kv_cache(token_embedding):
    K_pre_rope = K_linear(token_embedding)  # 不施加 RoPE!
    V = V_linear(token_embedding)
    kv_cache[page][offset] = (K_pre_rope, V)  # K path 不含 position info

def read_kv_cache(page, offset, virtual_position):
    K_pre_rope, V = kv_cache[page][offset]
    K = apply_rope(K_pre_rope, virtual_position)  # RoPE 在读取时动态注入
    return K, V

# 优势: 同一 page 可被不同 span context 以不同 position 复用
#   docA 在请求1被读为 position=[16,32)，在请求2被读为 position=[48,64)
#   → 每次读取时根据 Span Table 中的 virtual position 动态注入正确 position
```

RoPE 的数学原理（背景）：RoPE 通过旋转变换将位置信息编码到 Q 和 K 向量中：`Q_pos = Q · R(pos)`, `K_pos = K · R(pos)`，其中 `R(pos)` 是旋转矩阵。Attention score `Q_pos^T · K_pos = Q^T · R(pos_q - pos_k) · K`——仅依赖相对位置差。RoPE on Read 利用这一性质：只要 attention 计算时 Q 和 K 的 position 一致，K 是否在写入时已含 position 无关。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 Span Query 的 vLLM 修改中（gpu_model_runner.py，232 行修改），RoPE on Read 的实现：(1) KV cache 写入路径移除 RoPE application（K 以 pre-RoPE 形式存储）；(2) Attention kernel 在从 KV cache load K 后、计算 QK^T 前，根据 Span Table 提供的 virtual position 施加 RoPE；(3) RoPE 的计算开销在 attention 的 memory-bound 背景下可被 HBM 延迟隐藏。该修改与 CIDRA 算法协同：CIDRA 需要将同一 physical block reposition 到不同 virtual position，RoPE on Write 会使 reposition 后的 KV 值 position 错误，而 RoPE on Read 天然兼容任意 reposition。

涉及论文标题：
- Using Span Queries to Optimize Cache and Attention Locality

## Dual-Output Paradox（双输出悖论）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dual-Output Paradox（双输出悖论）是 Span Query 论文识别的一个 LLM serving 中的结构性问题：**模型服务向 client 输出一个 token stream，同时向 KV cache 写入一个（可能不同的）token 序列**。当查询包含 commutative join（如 multi-agent 场景中 `++(G(p1), G(p2))`——两个 generator 并行输出）时，client 期望按某种顺序接收输出（如先 G(p1) 后 G(p2)），但 KV cache 中这些 spans 的顺序可能因 commutativity 优化而不同——产生了"输出给 client 的东西"和"写入 KV cache 的东西"不一致的结构性问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
```
场景: Judge-Generator with 2 generators
  Client 期望: 先收到 G1 输出，再收到 G2 输出
  Expression: J(G(p1), G(p2))  // Judge 依赖两个 generator 的输出

Direct execution:
  Step 1: G(p1) → 输出 token1_1, token1_2, ..., token1_m → 写入 KV cache
  Step 2: G(p2) → 输出 token2_1, token2_2, ..., token2_n → 写入 KV cache
  Step 3: J(outputs) → 读取 G1+G2 的 KV cache → 输出判断

问题（Dual-Output Paradox）:
  - G1 和 G2 是 independent → commute → 可并行执行
  - 但 client 看到的 output stream 必须按某种顺序（先 G1 后 G2）
  - 而 KV cache 中 G1 和 G2 的顺序可能被优化器重排（最快完成者先写）
  → 输出 stream 的顺序 ≠ KV cache 中 token 的顺序

Span Query 的解决方案（Plus Distribution Rule）:
  原始: G(p) + ⨝(r1, r2)   // generator + commutative join
  重写: ⨝(G(p)+r1, G(p)+r2)  // 将 join 分布到 generator 输出路径
  含义: 将 "generator 先执行，输出再被 join" 
        重写为 "generator 的输出分别进入 join 的不同分支"
        每个分支独立向 client 输出 + 独立写入 KV cache
        join 发生在 client 侧或 server 端 scheduler 层
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Plus Distribution 是 Span Query high-level optimizer（固定点树重写）的四条规则之一。它确保 commutative join 的每个 child subtree 独立管理其 output stream 和 KV cache layout，消除了 client output 和 KV cache 之间的顺序冲突。在 4-way Judge-Generator 场景中，该规则将单次 8-way judge 优化为 3 层 2-way judge 的树状结构，每层 judge 仅需看 2 个 generator 的输出，避免了 lost-in-the-middle 且显著减少 TTFT。

涉及论文标题：
- Using Span Queries to Optimize Cache and Attention Locality
