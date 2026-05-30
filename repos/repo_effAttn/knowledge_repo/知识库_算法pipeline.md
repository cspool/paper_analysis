## Semantic Segment Partitioning for Video Streams（视频流语义分段划分）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Semantic Segment Partitioning for Video Streams 是一种基于视觉语义边界动态划分视频流的方法。与传统的均匀分段（每固定帧数一段）不同，该方法按视频内容的语义变化自适应确定段边界。核心流程：(1) 使用 ViT 编码器提取每帧的 patch 级 embedding $f_t \in \mathbb{R}^{P^2 \times D}$；(2) 计算相邻帧 embedding 的 cosine similarity $s_t = \frac{f_{t-1} \cdot f_t}{\|f_{t-1}\| \|f_t\|}$；(3) 将相似度低于阈值（如 0.99）的帧标记为语义边界；(4) 应用 exclusion window（最小段长 $m$）避免过短段；(5) 若段长超过上限 $M$，通过 segment merging 合并段内余弦相似度最高的相邻帧对（利用视频的时间冗余）。输出为语义段序列 $[\mathbf{S}^i]$，每段 $\mathbf{S}^i := [f_t^i]_{t=1}^{T_i}$ 满足 $T_i \in [m, M]$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

语义分段划分是 Streaming Video QA pipeline 的第一阶段，位于视频帧编码之前。它接收原始视频帧序列，输出语义边界标记的段划分结果。具体计算过程：

```
# 输入: 视频帧序列，ViT encoder
for t = 1 to T:
    f_t = ViT(frame_t)  # ∈ R^{P²×D}

# Step 1: 计算相邻帧相似度
for t = 2 to T:
    s_t = cos_sim(f_{t-1}, f_t)  # Eq.(1)
    if s_t < threshold:  # e.g., 0.99
        boundaries.append(t)

# Step 2: Exclusion Window 过滤
# 确保任意两个 boundary 之间距离 ≥ m（如 m=4）
boundaries = filter_by_window(boundaries, window_size=m)

# Step 3: Segment Merging
for each segment S^i:
    if len(S^i) > M:  # M = 64
        while len(S^i) > M:
            # 找到段内最相似的相邻帧对并合并
            (t1, t2) = argmax(cos_sim(S^i[t], S^i[t+1]))
            S^i[t1] = mean(S^i[t1], S^i[t2])  # 合并
            remove(S^i[t2])

# 输出: 语义段序列 [S^1, S^2, ...]
```

术语一般如何实现？如何使用？

实现方式：基于 ViT 编码器的 embedding 输出进行帧间相似度计算，不需要额外训练。阈值 $m$、$M$ 和 similarity threshold 是超参数（StreamKV 使用 m=4, M=64, threshold=0.99）。Segment merging 通过贪心合并最高相似度相邻帧对实现。适用场景：(1) 流式视频理解中需要在编码前确定段边界；(2) 避免均匀分段破坏语义连续性的任何长视频处理任务；(3) 可推广到其他需要内容感知分段的场景。与均匀分段相比，论文实验表明语义分段在所有压缩率下均获得更高准确率（Table 2: 50% 压缩率下 59.07% vs 57.32%）。

涉及论文标题：
- StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression

## Guidance Prompt for KV Cache Compression（引导提示驱动的KV缓存压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Guidance Prompt for KV Cache Compression 是 StreamKV 提出的一种面向流式视频场景的 KV cache 压缩方法。与现有的基于用户问题（question-dependent）的 KV 压缩方法（如 FastV、SparseVLM、SnapKV）不同，该方法不依赖具体的用户问题，而是引入一个 guidance prompt 来捕获视频段内的关键语义元素，以此作为压缩的选择依据。Guidance prompt 覆盖五类语义元素：(1) salient entities（人物、物体、场景、关键视觉概念）；(2) key events and actions（发生了什么、何时、何地）；(3) temporal and causal relationships（事件时序和因果链）；(4) contextual cues（场景切换、对话、叙事变化）；(5) important numerical or factual details（计数、摘要、事实类信息）。压缩在每段编码完成后立即执行（非解码阶段离线压缩），使用 guidance prompt 的平均 query vector $\mathbf{g}^l = \frac{1}{N_g}\sum_{k} \mathbf{g}_k^l$ 作为层自适应 KV 选择模块的 selection criterion。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

Guidance Prompt KV 压缩的 pipeline 流程：

```
# 输入: 语义段 S^i 的 KV blocks B_l^i, 压缩率 θ, guidance prompt G

# Step 1: 将 guidance prompt G 送入 LLM，提取每层的 query vectors
for each layer l:
    g^l = (1/N_g) Σ_k g_k^l  # guidance prompt 平均 query vector

# Step 2: 计算总压缩预算
N = ⌈(1-θ) × T_i⌉ × L  # T_i 段帧数, L 层数

# Step 3: 层自适应选择
{I_l^i}_{l=1}^L = SelectKV({R_l^i, g^l}_{l=1}^L, N)  # Eq.(9)

# Step 4: 保留选中的 KV blocks + summary KV block
~B_l^i = [b_m^{i,l} | m ∈ I_l^i]  # 压缩后 KV blocks
B_l ← [B_l, ~B_l^i, b_s^{i,l}]    # 存入 KV Bank

# 关键: b_s^{i,l} (summary KV block) 不参与压缩，始终保留
```

与 question-dependent 压缩的对比：现有方法（FastV、SparseVLM）需要已知用户问题才能压缩，不适合 StreamingVQA 场景（问题未知、多轮对话）。Guidance prompt 使压缩聚焦于视频语义本身而非特定问题，更鲁棒。

术语一般如何实现？如何使用？

实现方式：guidance prompt 是一个预定义的文本模板（见论文 Appendix A），如 "Please identify the key semantic elements in this video segment, including salient entities, key events, temporal relationships, contextual cues, and important factual details"。将 guidance prompt 送入 Video-LLM，提取其在各 transformer 层的 query vectors 作为 selection criterion。适用场景：(1) 流式视频处理中未知用户问题的 KV 压缩；(2) 多轮对话场景下需要保留通用语义信息；(3) 任何需要 problem-agnostic 压缩的长上下文 LLM 推理场景。论文实验表明：60% 压缩率下 Overall 准确率 58.9%，甚至优于无压缩的 ReKV（53.5%），证明压缩不仅减少显存还通过去除冗余提升精度。

涉及论文标题：
- StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression

## Layer-Adaptive KV Selection / Budget Allocation（层自适应KV选择/预算分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Layer-Adaptive KV Selection 是 StreamKV 提出的统一 KV 选择模块，用于将 KV 压缩和检索统一为同一框架。它将选择问题形式化为：在给定总预算 $N$（需保留的 KV blocks 总数）下，跨 $L$ 个 transformer 层自适应分配每层的选择数量 $K_l$，使得保留的 KV blocks 总量 $= N$。与 uniform allocation（每层选择相同数量）不同，自适应策略根据每层相似度分布的集中程度分配预算——信息更集中的层获得更多预算，在总预算不变的情况下最大化保留的信息量。核心机制：(1) 每层计算候选 representative key vectors 与 selection criterion 的 cosine similarity；(2) Softmax 归一化得到概率分布；(3) 通过 binary search 确定全局 cumulative score threshold $p$，使跨层累积达到 $N$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

层自适应选择的三步算法流程：

```
# 输入: {R_l, c^l}_{l=1}^L (每层候选 + 选择标准), 总预算 N

# Step 1: Cosine Similarity per layer
for each layer l:
    for each candidate j in R_l:
        Sim_l(j) = cos_sim(r_j^l, c^l)

# Step 2: Softmax Normalization + Descending Sort
for each layer l:
    ~Sim_l(j) = softmax(Sim_l)(j)  # Eq.(6)
    priority_l = sort_descending(~Sim_l)  # s_l(j) = index of j-th largest

# Step 3: Binary Search for Global Threshold p (Algorithm 1)
p_1, p_2 = 0, 1
while p_2 - p_1 > ε:
    p = (p_1 + p_2) / 2
    for each layer l:
        K_l^p = min{k | Σ_{j=1}^k ~Sim_l(s_l(j)) ≥ p}  # Eq.(7)
    if Σ_l K_l^p == N:
        return p, {K_l^p}
    elif Σ_l K_l^p < N:
        p_1 = p  # p 太小，提高阈值以选更多
    else:
        p_2 = p  # p 太大，降低阈值

# 输出: {I_l}_{l=1}^L, 其中 I_l = top-K_l 候选的索引集合
```

Binary search 的正确性：cumulative sum function $f(p) = \sum_l K_l^p$ 关于 $p$ 单调递减（p 越大，K_l^p 越小），因此 binary search 可找到使 f(p)=N 的 p。

术语一般如何实现？如何使用？

实现方式：统一模块同时用于压缩（criterion = guidance prompt vector）和检索（criterion = question vector）。复杂度：binary search O(log(1/ε)) ≈ 常数次迭代，每次计算 O(L log|R|)（排序后取前缀和）。适用场景：(1) 任何需要跨层预算分配的 KV 选择问题；(2) 可推广到其他需要自适应保留重要 token/feature 的任务（如 token pruning、expert selection in MoE）。消融实验（Table 4）验证：Ada.+Ada.（压缩和检索均自适应）优于 Uni.+Uni.（全 uniform），如 50% 压缩率 59.07% vs 58.12%。

涉及论文标题：
- StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression

## Representative Key Vector for KV Cache Similarity Retrieval（KV缓存相似度检索的代表键向量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Representative Key Vector 是 StreamKV 中用于高效 KV 相似度计算的每帧/每块的聚合特征向量。对于第 i 段的第 m 帧，其 per-patch key vectors 为 $[\mathbf{k}_{m,p}^i]_{p=1}^{P^2}$（$P^2$ 为 ViT patch 数），representative key 定义为所有 patch-wise key 的均值：$\mathbf{r}_m^i = \frac{1}{P^2} \sum_{p=1}^{P^2} \mathbf{k}_{m,p}^i \in \mathbb{R}^{D'}$。其中 $D'$ 为不区分 attention heads 的拼接维度（将所有 head 的 key 维度拼接为单个向量）。Representative key 的用途：(1) 作为 KV 压缩/检索中 cosine similarity 计算的输入（替代完整的 multi-head key tensor）；(2) 存入 KV Bank 的索引结构中，与对应 KV blocks 一一映射，实现快速相似度检索而不需要加载完整 KV blocks。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

Representative key 在 pipeline 中的计算和使用：

```
# Per-frame KV block 定义
# b_m^i = [(k_{m,p}^i, v_{m,p}^i)]_{p=1}^{P²}  # 所有 patch 的 KV

# Representative key 计算 (Eq.3)
r_m^i = (1/P²) Σ_{p=1}^{P²} k_{m,p}^i  ∈ R^{D'}
# 不区分 attention heads: D' = num_heads × head_dim
# 所有 head 的 key 拼接为一个长向量后取 patch 平均

# KV Bank 存储结构（per layer l）
B_l = [b_1, b_2, ..., b_n]     # KV blocks (存储完整 K, V)
R_l = [r_1, r_2, ..., r_n]     # Representative keys (用于检索)

# 检索时: 使用 R_l 做相似度计算，用 B_l 查表获取完整 KV
{sim_j} = cos_sim(R_l, criterion)  # 仅需轻量 D' 维向量比较
I = top_K(sim_j, K)                 # 选出 top-K 索引
{P} = [B_l[j] | j ∈ I]             # 从 Bank 获取完整 KV blocks
```

关键设计权衡：representative key 是 patch-mean pooled 的一维向量，丢弃了 patch 级空间信息但保留了帧级语义特征，使相似度检索的计算量从 O(P² × D') 降至 O(D')。

术语一般如何实现？如何使用？

实现方式：在每段编码完成后，对每帧的 key tensor（形状 [num_heads, P², head_dim]）做 reshape 到 [P², D'] 后在 P² 维度上取平均。Representative key 与对应的完整 KV blocks 成对存储。适用场景：(1) 需要快速遍历大型 KV 库进行相似度检索的场景；(2) 可推广到任何需要轻量索引来组织压缩后 KV cache 的系统；(3) 相似度计算使用 cosine similarity（方向性）而非 L2 距离，因为 attention 机制天然对向量方向敏感。论文未独立评估 representative key 的消融，但其是层自适应选择模块的必要输入。

涉及论文标题：
- StreamKV: Streaming Video Question-Answering with Segment-based KV Cache Retrieval and Compression

## Quantization Range Narrowing for FP16 Accumulator Safety

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Quantization Range Narrowing for FP16 Accumulator Safety（量化范围收缩以保证 FP16 累加器安全）是 SageAttention2++ 提出的一种量化策略。与传统的"最大范围量化"（如 FP8 E4M3 格式将值量化到其完整表示范围 [-448, 448]）不同，该策略有意将量化 scale factor 调大（即缩小量化后的数值范围），使量化后张量的值域远小于 FP8 格式的上界。目的是确保在 mma.m16n8k32 指令的 FP16 累加器中累积 32 个乘积项后，累加结果不会超出 FP16 的最大可表示值 65504。数学约束为：

$$P_r \times V_r \leq 2047 \quad \text{（其中 } P_r = \max(|\tilde{P}|)/\delta_P,\; V_r = \max(|V|)/\delta_V \text{）}$$

即量化后 P 和 V 的元素上界乘积不超过 2047（65504/32）。若启用 Delayed FP32 Buffering（两次 MMA 结果在 FP16 中累加），约束变为 $P_r \times V_r \leq 1023.5$。

从算法pipeline角度拆解术语：

该策略位于 attention 量化 pipeline 的 P×V 阶段。标准 SageAttention2 pipeline 中，P 和 V 分别量化到 FP8 E4M3 完整范围（$P_r=448, V_r=448$）。SageAttention2++ 修改 scale factor 计算：

原 SageAttention2：
```
δ_P = max(|P̃|) / 448       # E4M3 完整范围
δ_V = colmax(|V|) / 448     # E4M3 完整范围
P̂ = round(P̃ / δ_P)          # P̂ ∈ [-448, 448]
V̂ = round(V / δ_V)          # V̂ ∈ [-448, 448]
# 使用 mma.f32.f8.f8.f32 (FP32 acc, 不怕溢出)
O = P̂V̂ * δ_P * δ_V
```

SageAttention2++ narrowing：
```
δ_P = max(|P̃|) / 224        # 缩小范围，P̂ ∈ [-224, 224]
δ_V = colmax(|V|) / 4.5     # 缩小范围，V̂ ∈ [-4.5, 4.5]
P̂ = round(P̃ / δ_P)          # 每个元素 |P̂| ≤ 224
V̂ = round(V / δ_V)          # 每个元素 |V̂| ≤ 4.5
# 使用 mma.f16.f8.f8.f16 (FP16 acc, 需要范围安全)
# 验证: |32 × P̂ × V̂| ≤ 32 × 224 × 4.5 = 32256 ≤ 65504 ✓
O = P̂V̂ * δ_P * δ_V
```

关键设计权衡：缩小 V 的量化范围（V_r=4.5 << 448）会导致 V 的量化精度下降，但由于 P 的范围相应扩大（P_r=224，仍小于 448），两者的乘积 $P_r \times V_r$ 保持不变（精度等价）。实验（Table 2）表明 (P_r=224, V_r=4.5) 与 (P_r=448, V_r=448) 的 CosSim 均为 99.97%、L1 误差一致，证明该"置换"几乎无损。

术语一般如何实现？如何使用？

该策略的实现方式是修改量化 kernel 中 scale factor 的计算逻辑，将除数从 FP8 格式最大值（448 for E4M3）改为自定义的 $P_r$ 和 $V_r$。$P_r$ 和 $V_r$ 作为超参数由实验确定，选择满足精度约束和累加器安全约束的最优对。SageAttention2++ 通过 Table 2 的网格搜索确定了 (224, 4.5) 为最优参数：在 CosSim=99.97%、L1=0.01862 的条件下最大化性能。

该策略适用于任何使用低精度累加器（如 FP16, BF16）进行高精度 Matmul（如 FP8, INT8 输入）的场景——只要累加器的表示范围小于操作数乘积的最大可能值，就需要缩小操作数的量化范围。典型场景包括 FP8/INT8 MMA with FP16/BF16 accumulator 的 GEMM kernel、attention kernel、FFN kernel 等。

涉及论文标题：
- SageAttention2++: A More Efficient Implementation of SageAttention2

## Dynamic Sparse Attention (动态稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

动态稀疏注意力（Dynamic Sparse Attention）是一种推理时（inference-time）技术，根据当前输入动态决定注意力矩阵中哪些位置需要计算、哪些可以跳过，从而减少注意力计算的 FLOPs。与静态稀疏注意力（如 Longformer 的固定 sliding window + global attention pattern）不同，动态稀疏注意力不预设固定的稀疏掩码位置，而是根据每个具体 prompt 的内容在线估计重要的 token/block 位置。

MInference 论文将动态稀疏注意力形式化为：$A(M) = \text{Softmax}(QK^T/\sqrt{d} - c(1-M))$，其中 $M_{i,j} \in \{0,1\}$ 是动态稀疏掩码，$c$ 是大常数（如 1e5），使 $M_{i,j}=0$ 的位置的注意力权重接近零。优化目标是最小化稀疏输出与 dense 输出的差异（$\min |A(M) - A_{\text{dense}}|$）以及总延迟（$\min t_{\text{sparse}}(M) + t_{\text{overhead}}(M)$）。

核心挑战在于：(1) 注意力分布高度动态——同一 token 位置在不同 prompt 下关注的 token 完全不同（MInference 验证：对 128K context 取 top-4K 列，在另一 prompt 上 recall 从 96.8% 降至 83.7%）；(2) 但注意力模式的类型（pattern type）在同一 head 上跨 prompt 保持一致性——即 head 总是表现为 A-shape/Vertical-Slash/Block-Sparse 中的某一种；(3) 在线估计必须低开销，否则估计开销抵消稀疏计算的收益。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

动态稀疏注意力在 MInference 的三步 pipeline 中执行：

```
# Step 1: 离线 Pattern Assignment（一次性）
for each attention head h:
    p_best[h] = KernelAwareSearch(Q_ref, K_ref, V_ref, target_FLOPs)
    # 为每个 head 分配 A-shape / Vertical-Slash / Block-Sparse 模式

# Step 2: 在线动态稀疏索引构建（每个 prompt）
for each attention head h:
    if p_best[h] == "A-shape":
        # 静态掩码：固定保留 1K global + 4K local，零开销
        M[h] = StaticMask(global=1024, local=4096)
    elif p_best[h] == "Vertical-Slash":
        Â = softmax(Q_{[-64:]} @ K^T / √d)    # 仅用最后 64 个 query
        i_v = argtopk(sum_v(Â), k_v)           # top-k 垂直列
        i_s = argtopk(sum_s(Â), k_s)           # top-k 斜线
        M[h] = SparseFormat(i_v, i_s)
    elif p_best[h] == "Block-Sparse":
        Q̂ = MeanPool(Q, 64); K̂ = MeanPool(K, 64)
        Â = softmax(Q̂ @ K̂^T / √d)             # block-level 近似
        i_b = argtopk(Â, k_b)                  # top-k blocks
        M[h] = SparseFormat(i_b)

# Step 3: 稀疏注意力计算
for each attention head h:
    y[h] = SparseAttention(Q, K, V, M[h])  # 仅计算 M[h] 标记的位置
```

**具体例子**（LLaMA-3-8B, 128K context）：
- Full attention: $QK^T$ 矩阵 $131072 \times 131072$，约 $2.2 \times 10^{11}$ FLOPs
- Dynamic Sparse Attention: 仅计算 ~4% 的 attention 位置（~96% sparsity），FLOPs 降为 $\sim 9 \times 10^9$
- 开销：Vertical-Slash head 的估计开销 <15%，Block-Sparse head 的估计开销 <25%

术语一般如何实现？如何使用？

动态稀疏注意力的实现通常包含三个关键组件：

1. **模式识别器（Pattern Identifier）**：离线分析 attention head 的稀疏模式类型。可以是基于启发式的（如 MInference 的 kernel-aware search）或基于统计的（如观察 attention map 的空间分布特征：初始 token 集中度、垂直条纹、块状聚集等）。

2. **在线估计器（Online Estimator）**：用极低的计算代价预测当前输入的稀疏分布。常见方法：
   - MInference VS head: 仅使用最后 $\text{last}_q$ 个 query（默认 64）做 matmul
   - MInference BS head: mean pooling + block-level matmul
   - Quest: 基于 query-aware 的 chunk-based importance scoring
   - SparQ Attention: 使用 low-rank hidden states 近似注意力

3. **稀疏计算 kernel**：执行带动态稀疏掩码的高效注意力计算。需要在 GPU 上支持非规则内存访问模式，通常基于 FlashAttention 的 tiling 框架修改。

使用场景：长上下文 LLM 推理的 pre-filling 阶段（prompt >32K tokens 时收益显著），尤其适用于 retrieval、summarization、long-document QA 等需要全局上下文的场景。在 short context（<10K）下动态索引构建开销占比可能达到 30%，收益有限。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

## A-shape Attention Pattern (A形注意力模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

A-shape 注意力模式是 MInference 论文识别的三种长上下文注意力稀疏模式之一。其名称来源于注意力矩阵的视觉形状：attention weights 集中在**初始 token（attention sink / global tokens）**和**局部窗口（local window / recent tokens）**，在注意力矩阵热力图上形成类似字母 "A" 的形状——两侧有高注意力值（左侧=初始 token，右侧对角线=局部窗口），中间区域几乎为零。

A-shape 模式的特征：(1) **空间分布**：Static structured——无论输入内容如何变化，重要 token 的位置始终是初始若干 token + 末尾局部窗口；(2) **GPU 延迟**：Low——因为稀疏模式是结构化的、固定的，可以直接使用 FlashAttention 仅计算对应区域；(3) **索引构建时间**：Zero——完全静态，无需在线估计。

A-shape 模式的典型 attention head 负责处理局部语法结构或对初始 token（如 BOS token 或系统 prompt）的持续关注。StreamingLLM 论文（Xiao et al., 2024）首次系统性地识别了这一模式并用于 decoding 阶段的 KV cache 压缩。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

A-shape 模式的计算极其简单——使用固定的因果掩码变体：

```
# A-shape 稀疏掩码定义
S = 131072  # 序列长度
M[i, j] = 1 if (j < GLOBAL) or (j >= i - LOCAL) else 0
# GLOBAL = 1024 (初始 global tokens)
# LOCAL = 4096 (局部 window tokens)

# 稀疏注意力计算
A = softmax(Q @ K^T / √d - c * (1 - M))  # c=1e5, 非M区域强制为0
y = A @ V
# 等效于在 FlashAttention 中仅遍历 global + local 区域
```

**具体执行**（LLaMA-3-8B, 128K context, A-shape head）：
- 仅计算: row 0-1023（global）的所有列 + row 1024-131071 的列 j∈[0,1024)∪[i-4096,i]
- FLOPs: ~1K × 128K + 127K × 5K ≈ $1.9 \times 10^9$（vs dense 的 $2.2 \times 10^{11}$）
- 稀疏率: ~99%

术语一般如何实现？如何使用？

实现方式：在 FlashAttention kernel 中，将 Q 的分块循环限制在 global token block 和 local window block 范围内，跳过中间所有 token blocks。由于模式完全固定，可以在 kernel 编译期就确定 loop range。

使用场景：A-shape 模式适合负责局部语法处理的 attention head，如相邻 token 的依存关系、局部上下文理解。不适合需要全局检索或多跳推理的 head（如 retrieval head）。MInference 论文的搜索结果显示，A-shape 模式主要在模型的中间层出现，占比较少（<<10%）。

主要局限：当关键信息位于 global window 和 local window 之间时（如中间位置的 passkey），A-shape head 完全无法捕获，导致 retrieval 类任务准确率崩溃。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

## Vertical-Slash Attention Pattern (垂直-斜线注意力模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Vertical-Slash（VS）注意力模式是 MInference 论文识别的三种长上下文注意力稀疏模式之一，也是占比最高的模式（>90% 的 attention heads 被分配为 VS 模式）。VS 模式的注意力权重集中在：(1) **垂直列（Vertical lines）**——某些特定 token 被几乎所有位置的 query token 广泛关注（类似 "attention sink" 但不仅限于初始 token，可出现在序列中的任意位置）；(2) **斜线（Slash lines）**——token 以固定间隔关注序列中其他位置的 token，在注意力矩阵上形成对角线/斜线模式，是 RoPE 位置编码在长上下文下的典型表现。

VS 模式的关键特征：(1) **空间分布**：Dynamic structured——垂直列和斜线的**具体位置**随输入内容动态变化，但**模式类型**（即总是垂直+斜线组合）在同一个 head 上保持一致；(2) **GPU 延迟**：Medium——需要混合 block-level（斜线用 64×64 blocks）和 column-level（垂直线用 1×64 columns）两种稀疏格式；(3) **索引构建时间**：Small——仅使用最后 64 个 query 向量做估计，占 5-15% 总时间。

与 A-shape 的关键区别：VS 模式的垂直列可以出现在序列中任意位置（不仅是初始 token），因此能捕获分布在 prompt 中间位置的重要信息（如长文档中间的 key-value pairs、中间章节的主题句等）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Vertical-Slash Head 的动态稀疏索引构建与计算（Algorithm 2）
输入: Q, K, V ∈ R^{S×d_h}, k_v=30, k_s=2000, last_q=64

# Step 1: 估计注意力分布（仅用最后 64 个 query）
Â = softmax(Q[-64:] @ K^T / √d + m_causal)  # [64, S]

# Step 2: 提取垂直列索引（沿 query 维度求和→每列的全局重要性）
score_v = Â.sum(dim=0)                        # [S]
i_v = argtopk(score_v, k_v)                   # [30] — top-30 垂直列

# Step 3: 提取斜线索引（沿斜线方向求和→每条斜线的全局重要性）
score_s = Â 沿斜线方向求和                       # [S] 每条斜线 score
i_s = argtopk(score_s, k_s)                   # [2000] — top-2000 斜线

# Step 4: 构建混合稀疏格式
# 斜线用 64×64 blocks（slashes 在 block level 连续）
# 垂直线用 1×64 columns（vertical 是细粒度列级）
i_vs = PointRangeTwoWayMerge(i_v, i_s, block_size=64)

# Step 5: 稀疏注意力计算
A = softmax(sparse(Q @ K^T, i_vs) / √d)       # 仅计算 i_vs 索引区域
y = sparse(A @ V, i_vs)
```

**具体例子**（LLaMA-3-8B, 128K context, VS head, k_v=30, k_s=2000）：
- 计算量：~30 × 128K (垂直列) + 2000 × 64 × 64 (斜线 blocks) + 64 × 128K (估计)
- FLOPs: ~$2.0 \times 10^9$（vs dense 的 $2.2 \times 10^{11}$）
- 稀疏率: ~99%
- 1M context 下 kernel 级加速：13× vs FlashAttention

术语一般如何实现？如何使用？

实现需要两个定制 GPU kernel：
1. **Vertical-Slash Index Kernel**：使用 point-range two-way merge 算法——垂直列视为 points、斜线转换为每行对应的 column ranges，合并后输出两部分：block indexes（斜线的 64×64 blocks）+ column indexes（垂直的 1×64 columns）。GPU 上按行并行（N = S/B 行），每行时间复杂度 O(k_v + k_s)。

2. **Vertical-Slash FlashAttention Kernel**：混合 kernel——前半部分使用 Block-Sparse FlashAttention 处理斜线 blocks（标准 FlashAttention tiling），后半部分使用 PIT（Permutation Invariant Transformation）将非连续的 column data 加载到 dense compute blocks 处理垂直列。

使用场景：VS 模式是最通用的稀疏注意力模式，适用于绝大多数 attention head。能有效处理 retrieval（垂直列捕获关键 value token）、summarization（斜线捕获周期性结构）、QA（两者结合）等各类任务。需要注意的是 k_v 和 k_s 的配置需要通过 Kernel-Aware Search 离线确定以匹配 target FLOPs。

**Sparse Frontier 论文的补充发现**：VS 模式在 retrieval 任务（Low Scope, Low Dispersion）上表现优异，但需要根据任务类型选择近似窗口大小——retrieval-heavy 任务（Ruler NIAH、Story Retrieval）用 512 tokens 窗口，其他任务用 256 tokens。在 128K tokens 序列上，0.93 sparsity (1/15 budget) 的 VS pattern 仍保持在 Pareto 前沿上。FlexPrefill 在此基础上添加了 threshold-based 动态 budget 分配（由 coverage α 和 min_budget 控制），但论文发现在高压缩比下动态分配无效（回退到 α=0 均匀分配）。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
- XAttention: Block Sparse Attention with Antidiagonal Scoring

## Antidiagonal Scoring (反对角线评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Antidiagonal Scoring（反对角线评分）是 XAttention 论文提出的 block 重要性估计方法。核心洞察：注意力矩阵的**反对角线值之和**（antidiagonal sum）可以作为 block 重要性的高效代理指标。反对角线指从矩阵左下到右上的对角线方向（即方向为 $(-1, +1)$ 的线），与标准主对角线正交。

与现有方法的对比：(1) MInference/FlexPrefill 使用 mean/sum pooling 估计 block 重要性——但 pooling 在 block 内仅有少量显著垂直/斜线模式时会严重低估重要性；(2) 反对角线天然交叉 block 内所有可能的垂直列和斜线模式（见 Vertical-Slash Attention Pattern），确保不遗漏任何关键模式；(3) 每个 token 至少参与一条反对角线，保证信息完整性。

Strided 变体（Strided Antidiagonal Scoring）：以步长 S 在 B×B 的 block 内沿反对角线采样，将 Q reshape 为 S 组、K reshape 为 S 组，计算 S×S 的近似注意力矩阵。计算复杂度仅为完整注意力的 $1/S^2$。

从算法pipeline角度拆解术语：

```
# Strided Antidiagonal Scoring（block size B, stride S）
Input: Q, K ∈ R^{L×d}, block_size B, stride S
Output: block importance scores

# 对每个 B×B 的 attention block
for b = 0 to N_B - 1:
    # Step 1: Q anti-diagonal reshape [B, d] -> [S, B//S, d]
    Q_reshaped = []
    for i = S-1 down to 0:  # 从下到上遍历反对角线
        Q_reshaped.append(Q[b*B:(b+1)*B, :][i::S, :])

    # Step 2: K stride reshape [L, d] -> [S, L//S, d]
    K_reshaped = []
    for i = 0 to S-1:
        K_reshaped.append(K[i::S, :])

    # Step 3: 近似注意力分数
    A_approx = Softmax(Q_reshaped @ K_reshaped^T / sqrt(d_h) / S)

    # Step 4: 反对角线分数 = block 重要性
    score[b] = sum of antidiagonal values in A_approx
```

关键性质：反对角线交叉每个 block 内所有可能的垂直和斜线模式（见 XAttention Figure 2）。即使 block 内仅有一条垂直列或斜线，反对角线也必定与之相交。

消融实验（Table 6）：同等计算量下，antidiagonal 模式比 random 和 diagonal 模式密度更低且准确率更高——S=8 时 antidiagonal avg 88.47 (density 20.97%) vs random 82.48 (27.57%) vs diagonal 81.06 (24.47%)。

术语一般如何实现？如何使用？

基于 FlashInfer 框架实现。反对角线 scoring 作为 prefill attention 的预处理步骤——在 FlashAttention kernel 调用前，先执行轻量级 Q/K reshape + 小矩阵乘法来估计 block 重要性。计算开销极低（O(L×d/S²)），仅占总 prefill 时间的很小比例。

使用场景：适用于任意 Transformer 模型的 attention 模块（causal 和 non-causal 均支持），作为 block-sparse attention 的 block selection 指导。已在 Llama-3.1-8B（文本）、Qwen2-VL-7B（视频理解）、HunyuanVideo（视频生成/DiT）上验证。Stride S 的选择需权衡——S 越大越稀疏但可能漏检斜线模式（S=64 时 avg 降至 81.21）；推荐 S=8 或 S=16。

涉及论文标题：
- XAttention: Block Sparse Attention with Antidiagonal Scoring

## Block-Sparse Attention Pattern (块稀疏注意力模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Block-Sparse（BS）注意力模式是 MInference 论文识别的三种长上下文注意力稀疏模式中**最动态**的一种。其注意力权重在空间上没有明显的垂直条纹或斜线规律，而是呈现出分散的块状聚集（spatial clustering）——重要 token 以群体形式出现，但群体的位置高度依赖输入内容。

BS 模式的关键特征：(1) **空间分布**：Dynamic structured——注意力权重分散但存在块级空间聚集（block-level spatial clustering），相邻 token 的重要性往往相近；(2) **GPU 延迟**：Low——64×64 block-level 的 top-K 选择可以直接使用 Block-Sparse FlashAttention kernel，延迟与 block 数量线性相关；(3) **索引构建时间**：Small——使用 mean pooling 降采样后的 block-level matmul 进行近似，开销约占总时间的 25%（高于 VS 的 5-15%，因为需要额外的 pooling + block-level matmul）。

BS 模式的 motivation：MInference 分析发现，即使在注意力最分散的 head 中，非零注意力权重与其最近邻非零权重的距离仍然集中在 ~5 个 token 以内（在 128K context 下），证明了块级空间聚集的存在。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Block-Sparse Head 的动态稀疏索引构建与计算（Algorithm 3）
输入: Q, K, V ∈ R^{S×d_h}, k_b=100, block_size=64

# Step 1: Mean pooling 降采样（block_size=64）
Q̂ = MeanPool(Q, block_size)    # [S/64=2048, d_h] — 沿 seq 维度每 64 行取平均
K̂ = MeanPool(K, block_size)    # [2048, d_h]

# Step 2: Block-level 注意力近似
# 关键性质: MeanPool(Q)·MeanPool(K)^T ≈ MeanPool(QK^T)
# 即 pooling+matmul 近似等价于 matmul+pooling
Â = softmax(Q̂ @ K̂^T / √d + m_causal)  # [2048, 2048]

# Step 3: 提取 top-k_b blocks（每行取 top-k）
i_b = argtopk(Â, k_b, dim=1)           # [2048, 100] — 每行 top-100 blocks

# Step 4: 构建稀疏格式
i_b = sparseformat(i_b)                # 每行 block index list

# Step 5: 稀疏注意力计算
A = softmax(sparse(Q @ K^T, i_b) / √d)
y = sparse(A @ V, i_b)
```

**具体例子**（LLaMA-3-8B, 128K context, BS head, k_b=100, block_size=64）：
- 估计阶段：Q̂ @ K̂^T → [2048, 2048]，仅 $2048^2 \times 128$ FLOPs vs $131072^2 \times 128$
- 稀疏计算：每行仅计算 top-100 blocks（100 × 64 × 64 tokens），共 $2048 \times 100 \times 64^2 \times 128$ FLOPs
- 总 FLOPs: ~$5.4 \times 10^9$（vs dense 的 $2.2 \times 10^{11}$）
- 稀疏率: ~97.5%
- 1M context 下 kernel 级加速：30× vs FlashAttention（三种模式中最快）

术语一般如何实现？如何使用？

实现基于 Triton 版 FlashAttention kernel 修改：以 selected block index 为额外输入，每个 thread block 不再遍历所有 K/V blocks，而是仅遍历每行的 top-K blocks。速度比公式为 $s_p = S / (2B \times k_b)$，其中 B=64 为 block size，$k_b$ 为每行保留的 top blocks。

BS 模式主要分布在模型的 intermediate-to-late layers。适合处理高度动态的注意力需求（如 KV retrieval、multi-hop tracing），但单独使用效果不佳（仅 BS 模式在 InfiniteBench 上平均 18.7 vs Full 38.2），需要与 A-shape 和 VS 模式组合使用。其优势是 kernel 速度最快（30×），且 block-level 的结构化稀疏在 GPU 上非常高效。

**Sparse Frontier 论文的补充发现**：Block-Sparse 使用更小的 block size (16×16，而非 MInference 的 64×64)，因消融实验显示更小 block 始终产生更好性能。Block-Sparse 在 High Scope 或 High Dispersion 任务（如 Ruler VT、Story Filtering）上优于 Vertical-Slash——因为为每个 query block 选择不同的 key token blocks，支持处理多个独立语义片段。Paper 中 block-sparse 始终保留 attention sinks（第一个 key block）和局部上下文（对角线 block）。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
- XAttention: Block Sparse Attention with Antidiagonal Scoring

## Kernel-Aware Optimal Sparse Pattern Search (核感知最优稀疏模式搜索)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Kernel-Aware Optimal Sparse Pattern Search 是 MInference 论文提出的离线（offline）搜索方法，用于为每个 attention head 确定最优的稀疏模式（A-shape / Vertical-Slash / Block-Sparse 之一）及其具体参数配置（如 VS 的 k_v 和 k_s 数量、BS 的 k_b 数量）。它是 MInference 三步 pipeline 的第一步。

"Kernel-Aware"（核感知）的含义是：搜索空间中的 FLOPs 使用**真实 GPU kernel 中的 FLOPs**（而非概念上的稀疏 token 数），确保搜索出的最优配置在实际 GPU 执行时确实能达到预期的加速效果。例如，一个 $64 \times 64$ 的 block 块计算在 GPU 上的实际 FLOPs 与 $1 \times 64$ 的 column 不同，虽然它们覆盖的 token 数量相同。

搜索优化目标：最小化稀疏 attention 输出与 dense attention 输出的差异（$\argmin |y_i - y|$），而非仅最小化 attention score 的差异。这使用 FlashAttention 进行计算（降低 GPU 内存），并包含了 V 矩阵的信息，实现了 end-to-end 的最优模式选择。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Algorithm 1: Kernel-Aware Sparse Pattern Search
输入: Q, K, V ∈ R^{S×d_h}, patterns p ∈ {A-shape, VS, BS}, target FLOPs t

# Phase 1: 构建 Kernel-Aware 搜索空间
ρ = ∅  # 候选配置集合
for each pattern candidate σ_i:
    t_i = FLOPs_in_kernel(σ_i)     # GPU kernel 真实 FLOPs
    while |t_i - t| > ε:           # 调整参数使 FLOPs 逼近目标
        σ_i = ChangeSpace(σ_i, p_i)  # 微调参数（step=50）
        t_i = FLOPs_in_kernel(σ_i)
    ρ = ρ ∪ {σ_i}                  # 加入候选集

# Phase 2: 基于 Recall 的最优模式选择
y = FlashAttention(Q, K, V)       # Dense attention 输出作为 ground truth
for each candidate σ_i in ρ:
    y_i = SparseAttention(Q, K, V, σ_i)  # 候选配置的稀疏输出
p_best = argmin(|y_i - y|)        # 选择输出差异最小的配置
```

**搜索空间配置（MInference 论文）**：
| Pattern | Search Space |
|---------|-------------|
| A-shape | {(1024, 4096)} — 1K global + 4K local |
| Vertical-Slash | {(30, 2048), (100, 1800), (500, 1500), (3000, 200)} |
| Block-Sparse | {100} — top-100 blocks |

搜索使用一条 30K tokens 的 KV retrieval 合成样本，约 15 分钟在单 A100 上完成。同一模型的不同 context 长度版本（如 262K 和 1M）可复用相同的最优配置，展示了搜索结果的泛化性。

术语一般如何实现？如何使用？

实现步骤：
1. 选取一条代表性的 reference sample（不需要与下游任务完全一致，论文验证了 KV retrieval 合成数据的泛化性）
2. 运行 FlashAttention 获取 dense attention output（所有 query 的 attention output 作为 ground truth）
3. 对每个候选模式配置，执行对应的稀疏 attention 计算
4. 计算 $|y_i - y|$（L2 distance）并选择最小差异的配置
5. 记录最优配置到配置文件中，推理时直接读取

使用时注意事项：
- 需要确保 target FLOPs 与目标加速比匹配——更高 target FLOPs 意味着更高的准确率但更低加速比
- search space 的 ChangeSpace step 太小会导致搜索时间过长，太大可能跳过最优配置
- 搜索结果可以在模型的不同 context 版本间转移（论文验证了从 262K 模型迁移到 1M 模型的有效性）

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

## StreamingLLM (流式LLM注意力机制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

StreamingLLM（Xiao et al., 2024）是一种免训练的 KV cache 压缩方法，核心发现是 LLM 的注意力矩阵中存在 **attention sink** 现象——初始若干个 token 会持续获得不成比例的高注意力分数（即使它们在语义上不重要）。基于此提出：在 decoding 阶段，KV cache 仅保留 attention sink（初始 4 个 token）+ 最近局部窗口（W 个 token），丢弃中间所有 token 的 KV。

在 MInference 论文中，StreamingLLM 被用作 baseline（对应 A-shape 模式），参数为 1K global tokens + 4K local window。在 pre-filling 阶段的评测中，StreamingLLM 在 retrieval 类任务上表现极差（InfiniBench Retr.KV: 0.8 vs Full Attention 14.4, RULER 有效 context 仅 4K vs Full 16K），因为一旦关键信息超出 local window 范围，模型无法访问。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# StreamingLLM 的 KV cache 管理（decoding 阶段）
GLOBAL = 4    # attention sink tokens
LOCAL = W     # 局部窗口大小

# 每次 decode step:
KV_cache = KV_cache[:GLOBAL] + KV_cache[-LOCAL:]  # 丢弃中间 token
# 新 token 的 KV 始终追加到局部窗口末尾

# 注意力计算（所有 head 统一）
M[i, j] = 1 if (j < GLOBAL) or (j >= i - LOCAL) else 0
A = softmax(Q @ K^T / √d - c * (1 - M))
```

在 MInference 的 pre-filling 场景中：
- Global tokens 扩展到 1K（而非 4），因为 pre-filling 需要更多 global context
- Local window 设为 4K
- Decoding 阶段保持 dense 计算（不做稀疏）

术语一般如何实现？如何使用？

StreamingLLM 实现简单——在 PyTorch 中可通过修改 attention mask 实现：
```python
mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
mask[:, GLOBAL:-LOCAL] = False  # 屏蔽中间 token
```

使用场景：适合对局部上下文依赖强的任务（如 language modeling、对话），但不适合 retrieval、multi-hop QA 等需要全局上下文的任务。在 MInference 的分类中，StreamingLLM 等同于 "Ours w/ only A-shape"，代表了仅依赖静态局部模式的稀疏注意力方法的性能上限。

核心局限：无法处理需要非局部信息的任务（KV retrieval 准确率接近 0），因为关键信息的 token 可能位于 global window 和 local window 之间。

涉及论文标题：
- MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

## SnapKV (基于注意力重要性的KV选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SnapKV（Li et al., 2024）是一种静态 KV cache 压缩算法，在 prefill 阶段一次性选择重要的 KV pairs 供后续所有 decode step 复用。核心思想：利用 prefill 最后一层的 attention weight 来评估每个 KV pair 的重要性——一个 KV pair 如果被最后一层的 query 赋予高 attention score，说明它对后续生成也很重要。SnapKV 的处理流程：(1) 在 prefill 阶段计算最后一层的 attention weights；(2) 对 attention weights 沿 sequence length 维度做 average pooling（kernel_size=5）以平滑噪声；(3) 强制保留 observation window（最近 32 个 token 的 KV）；(4) 在剩余位置中选 Top-K 最高 attention score 的位置；(5) 将所有层的 KV cache 按选中的位置索引 gather 到压缩 KV cache 中。在 decode 阶段，draft model 仅对压缩 KV cache 做注意力计算。

在 MagicDec 论文中，SnapKV 被用作 self-speculation 的 draft KV 压缩算法，与 StreamingLLM 对比。SnapKV 的接受率远高于 StreamingLLM（Figure 4c），原因是 SnapKV 基于最后一层真实 attention 做重要性选择，而非 StreamingLLM 的固定 sink+window 策略。SnapKV 最佳 KV budget 为 2049（vs StreamingLLM 的 512），更大的 budget 带来更高接受率（>85%），且作为静态方法无搜索开销，使 SD speedup 达到 2.51x。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# SnapKV 的 KV 选择流程（prefill 阶段，一次完成）
输入: K_full, V_full ∈ [B, S, n_layers, n_heads, d_head]
       最后一层的 attention weights W_last ∈ [B, n_heads, S, S]

# Step 1: 沿序列维度 average pooling（平滑噪声）
W_pooled = AvgPool1d(W_last, kernel_size=5)  # [B, n_heads, S, S]

# Step 2: 取最后一行（最后一个 query 对所有 key 的 attention）
importance = W_pooled[:, :, -1, :]  # [B, n_heads, S]

# Step 3: 跨 head 聚合重要性（sum 或 max）
importance_agg = importance.sum(dim=1)  # [B, S]

# Step 4: 保留 observation window + Top-K 其余位置
obs_window = 32
obs_indices = [S-obs_window : S]  # 最近 32 token 必须保留
remaining = importance_agg[:, :S-obs_window]
top_k_indices = TopK(remaining, K - obs_window)

# Step 5: 合并索引并 gather 压缩 KV
draft_indices = sort(obs_indices ∪ top_k_indices)  # |draft_indices| = K
K_draft = gather(K_full, draft_indices, dim=2)  # [B, K, n_layers, n_heads, d_head]
V_draft = gather(V_full, draft_indices, dim=2)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SnapKV 开源：https://github.com/FasterDecoding/SnapKV。实现依赖 HuggingFace Transformers，在模型 forward pass 中插入 KV selection 逻辑。使用方式：设置 `window_size=32`（observation window）、`kernel_size=5`（pooling 核大小）、`max_capacity_prompt=K`（目标 KV budget）。适用于长上下文 LLM 推理（32K+ tokens），尤其适合 retrieval 和 QA 任务（因基于真实 attention score，能保留关键信息 token）。MagicDec 中使用 SnapKV 作为 static KV selection 的推荐方案——接受率高且无 decode 阶段搜索开销。

**原始 SnapKV 论文的核心机制**（Li et al., 2024, NeurIPS）：

SnapKV 的核心创新在于两个关键发现：(1) prompt 末尾一个 "observation window" 内的 queries 对 prefix keys 的注意力分配模式与生成阶段高度一致（Fig. 2，overlap rates 验证）；(2) 这一注意力模式在生成过程中保持稳定（Fig. 3）。基于此，SnapKV 无需依赖最后一层特殊处理——每层独立使用自身的 observation window queries 计算对 prefix keys 的注意力权重，沿 query 维度求和得到投票分数 C_h = Σ_i W_obs[h, i, :]，再通过 1D pooling（kernel_size 可配）聚类保留上下文完整性，最后 TopK 选择保留的 prefix KV 位置。

与后续工作（MagicDec/R-KV）中描述的关键差异：(a) 原始 SnapKV 每层独立投票，而非仅依赖最后一层；(b) observation window 通常使用 prompt 末尾的直接 query tokens，而非专门训练的 head；(c) 投票后保留 observation window 的完整 KV（不做压缩），仅压缩 prefix 部分。

**SnapKV 压缩流程（原始论文全栈）**：
```
# Prefill 阶段，每层 attention 计算完成后：
Q_obs = Q[:, :, -L_obs:, :]                    # observation window queries
attn_weights = Q_obs @ K_prefix^T / sqrt(d)     # [H, L_obs, L_prefix]
vote = attn_weights.sum(dim=-2)                 # [H, L_prefix] 沿 query 维求和
pool_vote = MaxPool1d(vote, kernel_size, stride=1, padding=k//2)
k = max_capacity - L_obs
indices = TopK(pool_vote, k, dim=-1)            # 每 head 独立选 TopK
K_compress = K_prefix.gather(indices)            # 压缩后的 prefix KV
V_compress = V_prefix.gather(indices)
K_new = cat([K_compress, K_obs])                # 拼接完整 observation window
V_new = cat([V_compress, V_obs])
# Decode 阶段：直接使用 K_new, V_new，KV cache 大小恒定
```

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding
- R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

**Sparse Frontier 论文中的 SnapKV 实现与评估**：使用 kernel_size=21（1D average pooling）平滑，observation window=128 tokens（原论文为 32），始终保留前 4 个 prefix token。近似窗口为 256 tokens（无显著任务依赖）。Paper 还评估了 Ada-SnapKV 变体——使用 max-aggregation（而非 mean）跨 query positions 和 heads 进行分数计算，结合动态 token budget 分配（每 head 最低 budget 20%）。在全量评测中，eviction-based 解码方法（SnapKV/Ada-SnapKV）在高稀疏度下普遍弱于 Quest 的 full-cache 方法，但 Ada-SnapKV（adaptive budget）始终优于 uniform SnapKV，尤其 multi-query 任务。

**R-KV 对 SnapKV 的扩展与改进**：R-KV 将 SnapKV 从 prefilling 阶段移植到 decoding 阶段，每 B_buffer=128 步触发一次压缩（而非仅 prefill 阶段一次性选择）。关键改进：(1) GQA 聚合方式从 mean-pooling 改为 max-pooling——R-KV 实验发现 max-pooling 更好地保留每个 query head 中最关键的 attention 信号；(2) 除 importance scoring 外额外引入 redundancy estimation（key vector 余弦相似度），通过 joint selection score Z = λ·I − (1−λ)·R 同时平衡重要性和去冗余性，解决 SnapKV 在推理模型（DeepSeek-R1）长 CoT 输出中因重复内容获得高 attention 而 over-retain 冗余 token 的问题。R-KV 使用的 λ=0.1 使 redundancy 项权重 (1−λ)=0.9 足以有效抑制冗余。在 AIME24 上，R-KV 以 10% KV cache budget 达到 lossless 压缩（SnapKV 同 budget 仅 ~20% pass@1）。

## Speculative Decoding for Long Context (长上下文投机解码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Speculative Decoding（SD，投机解码）是一种 lossless 的 LLM 解码加速算法：使用轻量 draft model 快速生成 γ 个候选 token，target model 通过一次并行 forward pass 验证所有候选，通过 greedy matching 或概率接受保证输出与 target model 原生 AR 解码完全一致。标准 SD 的加速比公式为：

$$\frac{T_{Avg}^{SD}}{T_T} = \frac{1}{\Omega(\gamma, \alpha)} \left( \frac{\gamma \cdot T_D}{T_T} + \frac{T_V(\gamma)}{T_T} \right)$$

其中 $\Omega(\gamma,\alpha) = \frac{1 - \alpha^{\gamma + 1}}{1 - \alpha}$ 为每步验证的期望生成 token 数，$\alpha \in [0,1]$ 为 draft token 接受率，$T_D, T_T, T_V$ 分别为 draft、target 解码、验证时间。

MagicDec 的核心贡献是打破了 "SD 仅对小 batch 有效" 的传统认知。通过在长上下文 + 大 batch 场景下识别 KV cache 瓶颈转移（Section 3.2），MagicDec 证明了当 $S \ge S_{\text{inflection}}$ 时 SD 对大 batch 仍然有效甚至 speedup 随 batch 增大而提升。关键机制：(1) 长序列下 KV cache loading 成为主导瓶颈（memory-bound），验证与解码共享 KV budget → $T_V/T_T \approx 1$；(2) 使用压缩 KV 的 self-speculation 使 $T_D/T_T \to 0$；(3) KV 压缩获得更高接受率（>90%）→ 降低 costly verification 次数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# MagicDec 长上下文 SD 的 decode 循环
while not all_done:
    # Phase 1: Draft（使用压缩 KV cache，生成 γ 个候选 token）
    for i in 1..γ:
        q_new = W_q @ embed(last_token)           # query
        s = q_new @ K_draft^T / sqrt(d_head)       # 仅对 K_draft (size K<<S) 计算
        a = Softmax(s)
        o = a @ V_draft
        next_token = LMHead(FFN(o))
        draft_tokens.append(next_token)
        # 更新 draft KV（追加新 token）
        update(K_draft, V_draft, next_token)
    
    # Phase 2: Verify（使用完整 KV cache，一次 forward 验证全部候选）
    # 拼接 last_token + draft_tokens 的 γ+1 个查询位置
    q_all = W_q @ embed([last_token] + draft_tokens)
    s_full = q_all @ K_full^T / sqrt(d_head)       # 对完整 KV cache
    logits = LMHead(FFN(Softmax(s_full) @ V_full))
    
    # Phase 3: Greedy matching 确定接受
    accepted = []
    for i, draft_tok in enumerate(draft_tokens):
        if draft_tok == argmax(logits[i]):
            accepted.append(draft_tok)
        else:
            accepted.append(argmax(logits[i]))       # 不匹配则取 target token
            break
    output.extend(accepted)
    # 更新完整 KV cache（追加所有新 token 的 KV）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

长上下文 SD 的部署要点：draft model 使用 target model 自身 + 压缩 KV（self-speculation），或小模型 + 压缩 KV。KV 压缩可选 static（SnapKV 无搜索开销但接受率上限 ~85-90%）或 dynamic（PQCache/TopK 接受率 >95% 但 batch-size 相关的 search cost T_select）。MagicDec 开源 https://github.com/Infini-AI-Lab/MagicDec 提供完整实现，基于 GPT-Fast + FlashInfer + torch.compile + CUDA graphs。适用场景：长上下文 LLM serving（S > 4000 tokens），batch size 32-256，speedup 1.2x-2.51x。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding

## Critical Sequence Length - S_inflection（临界序列长度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

$S_{\text{inflection}}$ 是 MagicDec 提出的临界序列长度概念：对于给定的模型-硬件组合，当输入 context length $S \ge S_{\text{inflection}}$ 时，speculative decoding（SD）在高 batch size 下开始有效（speedup > 1），且 speedup 随 batch size 增大而提升。当 $S < S_{\text{inflection}}$ 时，SD 在大 batch 下失效（speedup < 1），因为推理过程 compute-bound 导致 $T_V/T_T$ 过高。

$S_{\text{inflection}}$ 由两个因素决定：
1. **模型 FLOPS-to-memory ratio**：GQA 模型（如 LLaMA-3.1-8B）有更高的 FLOPS-to-memory 比（因为 KV head 更少），需更长序列才能达到 memory-bound 状态，因此 $S_{\text{inflection}}$ 更高。非 GQA 模型（如 LLaMA-2-7B）的 $S_{\text{inflection}}$ 更低。
2. **GPU FLOPS-to-bandwidth ratio**：H100（高 FLOPS/带宽比）的 $S_{\text{inflection}}$ 低于 A100 和 L40，意味着在 H100 上 SD 更早开始有效。

MagicDec 实验测定 LLaMA-3.1-8B 在 8×A100 上的 $S_{\text{inflection}} \approx 4000$ tokens（Figure 2c）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

S_inflection 的推导基于 roofline 模型。当 batch size 增大时：

- $S < S_{\text{inflection}}$：推理 compute-bound，线性层（MLP/Attention 投影）的计算成为瓶颈。验证 cost $T_V/T_T$ 随 batch 增大显著上升（Figure 2b），因为验证需对所有候选 token 做完整计算。SD 失效，speedup 随 batch 增大而下降。
- $S \ge S_{\text{inflection}}$：推理 memory-bound，KV cache loading 成为主导瓶颈。$T_V/T_T \approx 1$（验证与解码共享 KV）。同时 draft 使用压缩 KV（budget K << S）→ $T_D/T_T \to 0$ → speedup = $\Omega(\gamma,\alpha) > 1$。

```
# S_inflection 的判断逻辑
if S < S_inflection:
    # compute-bound: T_V/T_T 随 B 增大显著上升
    # SD speedup 随 B 增大下降 → 大 batch 应禁用 SD
else:
    # memory-bound: KV bottleneck, T_V/T_T ≈ 1
    # 压缩 KV draft 使 T_D/T_T → 0
    # SD speedup 随 B 增大反而提升 → 大 batch 使用 SD 有利
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

使用方式：在部署长上下文 LLM serving 时，根据模型和 GPU 类型估计 $S_{\text{inflection}}$，决定是否对当前请求启用 SD。对于 LLaMA-3.1-8B + A100，S > 4000 时启用 SD 且不限制 batch size；S < 4000 时仅在小 batch（< 32）启用 SD。H100 的 S_inflection 更低（~2000-3000），L40 更高。MagicDec 框架自动根据 profile 数据确定 S_inflection 并选择最优 drafting 策略。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding

## Cross-Head Unified Sparse Attention (CUSA) / 跨Head统一稀疏注意力

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

CUSA 是 LessIsMore 论文提出的免训练稀疏注意力机制的核心组件。它解决长期困扰稀疏注意力方法的核心问题：推理（reasoning）任务中 token 重要性是全局属性，而非每个 attention head 的局部属性。CUSA 的工作流程：(1) 每个 attention head 基于精确 attention score 独立提案自己的 top-k 候选 token；(2) 通过 UnionFlatten 将所有 head 的候选聚合为一个统一候选集；(3) 全局排序后保留最高分的 K·(1-r) 个 token（r 为近邻保留比例）；(4) 所有后续 attention head 和 layer 共享这个统一的 token 索引集 ρ 进行稀疏注意力计算。关键设计在于聚合步骤——CUSA 不假设 head 功能完全相同（常规假设是对不同 head 不同 token subset），而是利用 LessIsMore 论文中观察到的跨 head 空间局部性（cross-head spatial locality）来消除个别 head 的噪声选择，降低 selection variance。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

CUSA 在 LessIsMore 的 decoding pipeline 中的具体执行过程：

**伪代码（基于 Algorithm 1）**：
```
Input: hidden state h, KV cache C, token budget K, static ratio r
for each decoder layer i:
    q, k, v = f(W_qkv, h)           # 标准 QKV 投影
    C.append(k, v)                   # 更新 KV cache
    if i is Full Attention Layer:    # 仅前 2 层
        o = FullAttention(q, C[:])   # 完整注意力
    elif i is Token Selection Layer: # 仅 1 层（如 Layer 12）
        o = FullAttention(q, C[:])
        P = q @ C.K^T                # [H, 1, L_kv]，注意力分数矩阵
        # 各 head 独立提案
        rho_head = TopKIndices(P[:, :-(K·r)], k=K·(1-r))
        # 跨 head 统一聚合
        rho_unified = UnionFlatten(rho_head)
        rho_recent = Recent(K·r)
        rho = rho_unified[:K·(1-r)] ∪ rho_recent
    else:                            # 其余所有层
        o = SparseAttention(q, C[rho])  # 仅对 rho 中的 token 计算注意力
    h = FFN(o)
return lm_head(h)
```

**GQA 下的 CUSA 张量计算**（以 DeepSeek-R1-8B 为例，hq=32, hkv=8, r=4）：
1. P = q @ C.K^T  # [32, 1, L_kv]（Token Selection Layer 中计算）
2. For each KV group g (0..7):
     P_g = P[4g:4g+4, :, :]        # [4, 1, L_kv]，4 query heads 共享 1 KV head
     对每个 query head h in [0..3]:
       idx_g_h = TopK(P_g[h], k=K·0.75)
     聚合 4 个 head 的提案: idx_g = union(idx_g_0, idx_g_1, idx_g_2, idx_g_3)
3. 全局统一: idx_all = unique(flatten([idx_g for g in 0..7]))
4. 按原始 attention score 排序: idx_sorted = sort_by_score(idx_all)
5. 保留历史 token: idx_hist = idx_sorted[:K·0.75]
6. 保留近邻 token: idx_recent = [L_kv-K·0.25, ..., L_kv-1]
7. rho = idx_hist ∪ idx_recent  # 所有 32 query heads 和后续 layer 共享

术语一般如何实现？如何使用？

CUSA 在 LessIsMore 中实例化为 TidalDecode 的上层机制，通过 2 个 Full Attention Layer + 1 个 CUSA Token Selection Layer + 剩余 Sparse Attention Layers 实现。CUSA 的关键属性：
- **低频重选**：token 选择仅在 1-2 层执行（token selection layer），产生的 ρ 跨后续层复用。图 4 验证：仅 Layer 2 选择 vs 每层都选，attention recall 几乎相同（~95% vs ~96%），说明全局 token 重要性跨层稳定。
- **GQA kernel 友好**：所有 query heads 共享统一 ρ，单次 global-to-shared memory 加载即可服务整个 KV group 的 query heads，消除 per-head 独立选择导致的冗余 KV loading（G2S 从 2.34MB 降至 1.04MB, Table 4）。
- **免训练**：不需要任何模型权重修改或 fine-tuning，纯推理时机制。
- **架构无关**：可集成到任何 selection-based 稀疏注意力框架中，兼容 GQA 和 MHA。

代码开源：https://github.com/DerrickYLJ/LessIsMore

涉及论文标题：
- Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

## Attention Recall / 注意力召回率

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention Recall 是量化稀疏注意力质量的诊断指标，定义为稀疏注意力所选 token 子集捕获的真实注意力分布的占比。公式：$R_i = \frac{\sum (\text{softmax}(W_i)[\rho])}{\sum (\text{softmax}(W_i))}$，其中 $W_i = \frac{Q_i K_i^T}{\sqrt{d}}$ 为 head i 的 pre-softmax attention scores，ρ 为稀疏注意力选择的 token 子集（|ρ| = k < L_kv）。R_i 取值范围 [0, 1]，越高表示选中的 token 越准确捕获了 full attention 的信息。在推理任务（reasoning）中，高 attention recall 是保持准确率的必要条件（非充分条件），因为即使小选择误差在数千步 decoding 中也会累积为逻辑不一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

Attention Recall 的计算过程（以单 head 单 decoding step 为例）：
```
输入: Q_i (1 x d), K_i (L_kv x d), V_i (L_kv x d), 选中索引 rho (k 个)
1. 计算完整 attention scores: W_i = Q_i @ K_i^T / sqrt(d)  # [1, L_kv]
2. 计算完整 attention weights: A_i = softmax(W_i)           # [1, L_kv]
3. 计算选中部分的 attention mass: mass_retained = sum(A_i[rho])  # 标量
4. 计算全部 attention mass: mass_total = sum(A_i)               # = 1.0 (softmax)
5. Attention Recall: R_i = mass_retained / mass_total
```

LessIsMore 论文中用 Running Average Attention Recall 追踪长程 decoding 中的累积质量：每 N 步采样一次 R_i，对全部 head 取平均。Running average 可平滑单步波动，显示 recall 随 generation length 的增长趋势。论文图 1a 显示：StreamingLLM 和 TidalDecode 在 32K decoding 过程中 recall 从 ~90% 分别退化至 ~65% 和 ~75%，而 LessIsMore 稳定在 ~90%。

术语一般如何实现？如何使用？

Attention Recall 是离线分析工具，不参与在线推理。主要用于：(1) 对比不同稀疏注意力方法的 token 选择质量；(2) 诊断推理过程中的 selection error 累积（recall 退化 vs generation length 的曲线）；(3) 验证设计选择的合理性（如 CUSA 的跨头统一选择 vs per-head 独立的 recall 对比，图 4）。LessIsMore 论文利用 attention recall 的关键发现：即使在推理任务（AIME）和检索任务（NiTH）上使用相同 token budget，稀疏注意力在推理上的 recall 远低于检索（图 1b），因为推理需要更多 decoding step，selection error 累积更大。

涉及论文标题：
- Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

## Cross-Head Spatial Locality / 跨Head空间局部性

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cross-Head Spatial Locality 是 LessIsMore 论文在推理模型注意力模式中发现的第一个稳定局部性结构：在长程推理（long-horizon reasoning）的每个 decoding step 中，不同 attention head 的 top-k 重要 token 排名存在显著重叠。具体表现为：在同一 K-V group（GQA 模型）内，多个 query head 的 top-4K token 集合高度重叠（黄色区域），且跨所有 attention head 也存在大量共同关注 token（红色区域，图 2）。这一现象与传统假设——不同 attention head 功能高度特异化、需要独立的 token subset——直接矛盾。Cross-head spatial locality 在模型的各层和各 decoding step 中持续存在（论文附录 A.6，图 9-10）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Cross-Head Spatial Locality 的定量刻画**（基于 Qwen3-8B, AIME-24）：
1. 在某一 decoding step（如第 20K 步）和某一层（如 Layer 4）：
   a. 对每个 head i，计算其 ground-truth top-4K token 集合 T_i
   b. 计算 pairwise Jaccard similarity: J_{ij} = |T_i ∩ T_j| / |T_i ∪ T_j|
   c. 对同一 KV group 内的 4 个 heads 计算平均 Jaccard：~0.6-0.8（黄色区域，图 2）
   d. 对所有 32 heads 计算全局平均 Jaccard：~0.4-0.6（红色区域，图 2）
2. 追踪不同 decoding step（10K/15K/20K/25K 步）：重叠区域保持稳定（附录图 9）。
3. 追踪不同层：重叠模式从早期层到后期层持续存在。
4. 追踪不同任务（AIME-24/25, GPQA）：重叠模式跨任务一致。

**与传统假设的对比**：
- 传统假设：head i 关注 token A/B/C, head j 关注 token X/Y/Z → 需要独立 token 子集
- 实际观察：head i 和 head j 的关注 token 高度重叠 → 全局统一 token 子集同样有效
- 关键推论：按 head 独立 top-k 选择不仅冗余（相同 token 被多个 head 重复选择浪费 budget），还会引入 head 特定噪声（个别 head 的"错误"选择不被全局一致性纠正）

术语一般如何实现？如何使用？

Cross-Head Spatial Locality 被 LessIsMore 直接用于推导 CUSA 的跨 head 统一 token 选择机制：(1) 各 head 仍独立提案（保留 head 之间的细微差异），但 (2) 通过 UnionFlatten 聚合后全局排名，使得被多数 head 认同的 token 优先保留，个别 head 的噪声选择被自然淘汰。跨 head 空间局部性也是 CUSA 低频重选可行的理论基础——因为 token 重要性是全局一致的，早期层的选择可在后期层复用而不退化。

涉及论文标题：
- Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

## Temporal Recency Locality / 时间近邻局部性

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Temporal Recency Locality 是 LessIsMore 论文在推理模型注意力模式中发现的第二个稳定局部性结构：在长程推理的每个 decoding step 中，最近生成的 token（recently generated tokens）始终获得异常高的 attention score，且注意力分布中分配给近邻 token 的比例在整个 decoding 过程中保持稳定。具体表现为：(1) 在任意 decoding step 的 attention 分布中，最后 ~25% 的 token 获得约 25% 的总 attention mass；(2) 这个比例在不同 token budget（2K-8K）、不同 decoding step（1K-32K）、不同推理任务（AIME-24/25, GPQA）中保持稳定（论文附录 A.6，图 9-10）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Temporal Recency Locality 的定量刻画**（基于 Qwen3-8B, 多种 token budget）：
1. 在任意 decoding step t，对所有 32 heads 计算 attention distribution A_i[t]：
   $$A_i[t] = \text{softmax}(q_t K_{1:t}^T / \sqrt{d}) \in \mathbb{R}^{1 \times t}$$
2. 定义 recency ratio r_obs(t) = sum of A_i[t] on last W tokens / total attention mass
   其中 W = t（随序列增长递增）
3. 关键发现：r_obs(t) 在各 decoding step 中维持在 ~0.2-0.3（约 20-30% attention mass 分配给最近的约 25% tokens）。
4. 与 token budget K 的关联：近邻 token 占总关键 token 的比例保持恒定（recency ratio），不受 budget 绝对大小影响（图 8）。

**与 StreamingLLM 固定窗口的关键区别**：
- StreamingLLM：固定大小 sliding window（如最近 256 tokens），不随 token budget 变化
- LessIsMore Stable Recency Window：固定比例 r=0.25，即 K·r 个 token，随 budget K 自适应缩放
- 原因：Temporal recency locality 显示比例关系恒定（~25%），而非绝对大小恒定

术语一般如何实现？如何使用？

Temporal Recency Locality 直接导致 LessIsMore 的 Stable Recency Window 设计：在 CUSA 的 token 选择中，固定比例 r=0.25 的 token budget 专门分配给最近 K·r 个 token。这个设计确保：(1) 无论使用多少 token budget，近邻上下文始终占固定比例；(2) 历史 token 和近邻 token 之间的资源分配是动态平衡的，反映推理中"逐步构建在前一步基础上"的增量性质。消融实验（附录 A.1.2，图 8）验证：仅 25% 近邻 + 75% cross-head 选择的组合达到最高 attention recall 并成功解题；纯近邻（100%）因丢弃长程上下文而 recall 最低；纯 cross-head 但 0% 近邻也无法解题——证明推理同时依赖长程依赖和逐步近邻推理。

涉及论文标题：
- Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

## KV Cache (Key-Value Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Cache 是 Transformer Decoder 自回归推理中用于缓存已生成 token 的 Key 和 Value 张量的内存结构。在自回归生成中，每次生成新 token 时需对所有历史 token 重新计算 Attention。为避免重复计算，将每层每个 Attention Head 中已生成 token 的 Key 投影 $K = X \cdot W_K$ 和 Value 投影 $V = X \cdot W_V$ 存储在 GPU 显存中。第 n 步生成时，仅需计算当前 token 的 Q、K、V，然后将 Q 与缓存的全部历史 K 做 Attention，再与缓存的全部历史 V 加权求和。

KV Cache 的显存占用为 $2 \times B \times L \times H \times D \times \mathrm{bytes\_per\_param}$，随序列长度 L 线性增长。长序列（如 128K tokens）下，KV Cache 可能超过 GPU 显存容量，成为内存瓶颈。反复从 HBM 加载 KV Cache 也会受限于显存带宽。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**KV Cache 在 Decoder 推理中的伪代码流程**：

```
// 初始化：KV Cache 为空
K_cache = []   // 每层每头
V_cache = []

for n in 1..max_new_tokens:
    // Step 1: 仅计算当前 token 的 Q, K, V
    x_n = embedding(token_n)
    Q_n = x_n @ W_Q
    K_n = x_n @ W_K
    V_n = x_n @ W_V

    // Step 2: 将新的 K, V 追加到 Cache
    K_cache.append(K_n)
    V_cache.append(V_n)

    // Step 3: 用 Q_n 与完整 K_cache 计算 Attention
    scores = Q_n @ K_cache^T / sqrt(d_k)
    scores = causal_mask(scores)      // 上三角置 -inf
    attn_weights = softmax(scores)    // [1, n]
    output = attn_weights @ V_cache   // [1, d_v]

    // Step 4: 生成下一个 token
    token_{n+1} = argmax(lm_head(output))
```

**KV Cache 压缩/剪枝（如 H2O、A2SF）**：
```
// 每步计算 A2S 或 A2SF 分数
for n in 1..N:
    scores = Q_n @ K_cache^T / sqrt(d_k)
    // 更新每个 key token k 的累积重要性
    for k in 1..n:
        A_k += alpha^(n - generation_step(q)) * softmax(scores)[k]
    // 保留分数最高的 K 个 token
    K_cache = K_cache[top_k_indices(A, K)]
    V_cache = V_cache[top_k_indices(A, K)]
```

术语一般如何实现？如何使用？

在实际推理框架中，KV Cache 由 HuggingFace Transformers 的 `DynamicCache` 或 vLLM 的 PagedAttention 实现。vLLM 将 KV Cache 组织为固定大小的 Page（block），实现类似操作系统的虚拟内存管理：不连续物理存储 + 连续逻辑地址，消除碎片。KV Cache 压缩方法有五条主线：(1) Token Pruning（H2O、StreamingLLM）通过重要性评分选择性逐出 token；(2) Quantization（KIVI、Oaken）将 FP16 KV Cache 量化为 INT4/INT8；(3) Channel Shrinking（CSKV、MLA）对 W_K/W_V 做低秩分解，存储低维中间特征而非完整 Key/Value，从通道维度压缩。方法 (1)(2) 通常以即插即用方式集成到 HuggingFace Transformers，方法 (3) 需要轻微微调。(4) Layer-wise KV Cache Sharing（KVSharer）通过跨层共享 KV Cache 实现层间压缩——不计算某些层的 KV cache 而直接复用其他层的 KV cache。方法 (4) 也是即插即用，无需训练。(5) Temporal Compression（MTLA）——在低秩 latent 压缩基础上进一步沿时间维合并相邻 KV cache vectors，将序列长度从 T 降至 T/s。per-token cache = 9d_h·l/(2s)（s=2 时接近 MQA 水平），per-token decoding complexity 从 O(T) 降至 O(T/s)。需要从 scratch 训练或 fine-tune。

(4) Vector Quantization（CommVQ、VQLLM）：将每个 token 的 key/value 向量作为整体而非逐个标量进行量化。使用学到的码本（codebook）将 d 维向量编码为低 bit 表示（如 1-bit 或 2-bit），解码时通过码本矩阵乘法重建。CommVQ 通过设计 RoPE-可交换码本将解码融入 self-attention 计算，大幅降低解码开销。方法 (4) 需要校准数据离线训练编码器和码本。

**VLM 场景下的 KV Cache Pruning 特殊性（Cross-Self Pruning 论文揭示）**：

多模态 VLM（如 LLaVA）中，视觉 token 和文本 token 混合构成 KV Cache。现有方法（SnapKV、H2O）在整个混合序列上统一使用 self-attention scores 做 token 重要性估计，忽略了多模态场景下的分布差异：(1) self-attention scores（同一模态内）和 cross-attention scores（跨模态间）具有显著不同的分布——文本 token 的 self-attention scores 通常大于视觉 token，若统一对待会导致视觉 token 被过度剪枝，破坏跨模态交互；(2) Jensen-Shannon (JS) Divergence 分析显示 self-attention 和 cross-attention 的分布在不同 VLM 层间有大幅变化。解决方法：将 attention 矩阵分解为 intra-modality 和 inter-modality 两部分，独立进行 token 选择后取交集（M = M^s ∧ M^c），确保 token 在两个维度上都被判定为重要才保留。

**KV-Distill 中的 KV Cache 压缩方法**：KV-Distill 在序列长度维度压缩 KV Cache，通过可训练的 Token Importance Scorer (FFN) 在 prefill 阶段选出 top-k 重要 token，仅保留这些 token 的 KV 表示。被选中 token 在 encoding 阶段通过条件计算路由（LoRA-adapted W^Q/W^O）从被丢弃 token 处聚合信息。因此压缩后的 KV Cache 不仅在数量上减少（序列长度维度），在质量上也经过了增强。压缩在 prefill 完成后完成，decode 阶段直接使用压缩后的 KV Cache，零额外开销。

**NACL 的 KV Cache Eviction 策略**：
NACL 提出在 encoding 阶段（而非 generation 阶段）一次性执行全局最优 KV Cache 淘汰。核心策略：(1) PROXY-TOKENS EVICTION：选取输入末尾 ~10% token（对应用户问题）作为 proxy tokens，仅用 proxy tokens 的 attention scores 求和评估 token 重要性，避免 H2O 的全量累加引入冗余信息和 attention bias；(2) RANDOM EVICTION：将评分归一化后作为概率分布，每个 attention head 独立随机采样保留 token，确保信息跨 head 多样化保留。budget=20% 时，LLaMA-7B 32层×32头下 token 在至少一个 head 保留概率达 99.92%。KV cache 可压缩至 5×（20% budget），long-text 性能仅 -0.7%（vs H2O -2.9%）。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache
- CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs
- CommVQ: Commutative Vector Quantization for KV Cache Compression
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs
- Cross-Self KV Cache Pruning for Efficient Vision-Language Inference
- FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation
- HATA__Trainable_and_Hardware-Efficient_Hash-Aware_Top-k_Attention_for_Scalable_Large_Model_Inference
- KV-Distill: Nearly Lossless Learnable Context Compression for LLMs
- KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing
- LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation
- Multi-head_Temporal_Latent_Attention
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time
- Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving
- LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation
- TokenButler: Token Importance is Predictable
- TransMLA: Multi-Head Latent Attention Is All You Need
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs
- dKV-Cache: The Cache for Diffusion Language Models
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

**KV Cache in Diffusion Language Models (dKV-Cache 论文贡献)**：
传统 KV Cache 依赖因果注意力掩码（causal attention mask）和固定左到右解码顺序。Diffusion Language Models (DLMs) 使用双向注意力（bidirectional attention）和非顺序解码（non-sequential decoding），因此标准 KV Cache 不兼容——(1) 每步所有 token 互相 attend，K/V 状态随时间步变化；(2) 无法预知下一步解码哪个位置。dKV-Cache 揭示了 DLM 中 token 表征动态：已解码 token 的 K/V 在后续步趋于稳定，[MASK] token 持续波动。基于此提出延迟缓存（delayed caching）：仅缓存已解码 token 的 K/V，掩码 token 每步重新计算；且引入一步延迟避免刚解码 token 的表征剧变被过早缓存。这证明了缓存机制不仅限于 AR 模型。

**Q-Filters 对 KV Cache 压缩的贡献**：Q-Filters 提出了一类新的 token pruning 方法——基于 QK 几何特性的重要性估计。与 H2O（累积注意力分数）、StreamingLLM（固定 attention sink + sliding window）不同，Q-Filters 通过离线 SVD 找到每个注意力头 Query 分布的主方向（Q-Filter），推理时用 $\langle K_t^h, v_1^+ \rangle$ 估计 KV pair 重要性。该方法训练无关、FlashAttention 兼容、校准成本极低（<3 分钟），且 Q-Filters 对校准数据域不敏感（跨域余弦相似度 > 0.9）。

**LightTransfer 对层级别 KV Cache 缩减的贡献**：LightTransfer 引入层级 KV cache 缩减的新维度——不同 Transformer 层具有不同功能角色：某些层是"懒惰层"（注意力集中在 sink + recent tokens），可将其 full attention 替换为 streaming attention（仅保留 sink + recent KV cache），非懒惰层保留 full attention。识别方法：利用 FlashAttention LSE 值在 prefilling 阶段在线计算 lazy ratio $r_i$，使用优先队列动态选择懒惰层。这种 layer-wise hybrid 策略在 50% 层替换时实现 2.17× 吞吐提升、LongBench 仅下降 <1.5%，且无需任何训练。与 token-level pruning（H2O, StreamingLLM）和 inter-layer sharing（MiniCache, SqueezeAttention）正交互补，可与 SnapKV 等 intra-layer 方法组合使用。

**Quest 对 KV Cache 的选择而非驱逐方法**：Quest 提出 query-aware KV cache sparsity——不驱逐任何 token，完整保留 KV cache，但在每步 decode 时基于当前 query 动态选择 Top-K 关键 page 加载到 attention。核心机制：维护 per-page per-channel Key 的 min/max 值作为元数据（大小 ~12.5% of KV cache），用 Query 向量计算 attention score 的数学上界估计每 page 关键性。与 token pruning/eviction 方法的根本区别——(a) 保留所有 token 信息，不造成不可逆信息丢失；(b) 基于当前 query 动态选择，而非历史注意力累积。在 passkey retrieval 任务中，eviction 方法准确率 0-4%，Quest 64-1024 token budget 达 100%。开源：https://github.com/mit-han-lab/Quest。

---

## Low-bit KV Cache Dequantization in Autoregressive Decoding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Low-bit KV Cache Dequantization 是在自回归解码过程中将低比特（INT4/INT2）量化存储的 Key-Value cache 在 attention 计算前恢复为 FP16 精度以参与混合精度矩阵乘法的过程。与 low-bit weight 的 dequantization 不同，weight 是静态的、可离线预处理的——Marlin/Ladder 等 kernel 可在模型加载时完成所有 layout transformation。而 KV cache 是动态生成的：每个 decode step 都产生新的 K/V 并需要在线 quantize→pack→store，下一次 step 又需要 load→dequant→compute。这种"在线量化+在线解量化"的循环特性使 CUDA kernel 设计极具挑战——dequantization 成为 attention 计算的 critical path 而非一次性开销。

从算法pipeline角度拆解术语。

BitDecoding 中低比特 KV cache 的自回归解码全流程：

```
// Prefill 后
KV_cache_fp16 = [FP16 K/V of all prompt tokens]    // shape: L×d
Partition: X_pack = quantize_and_pack(KV[:L-N_r])   // → low-bit packed
           X_res  = KV[L-N_r:]                      // → FP16 residual

// Decode step t (autoregressive loop)
while not EOS:
    // Step 1: 新 token embedding
    x_t = embedding(token_t)

    // Step 2: QKV Projection (FP16 GEMM on TC)
    Q_t, K_t, V_t = x_t @ W_Q, x_t @ W_K, x_t @ W_V

    // Step 3: Dequant + Attention on packed KV cache
    for each tile in X_pack:
        K_fp16_tile = dequant(load_packed_K(tile), load_K_params(tile))
        V_fp16_tile = dequant(load_packed_V(tile), load_V_params(tile))
        S += Q_t @ K_fp16_tile^T         // mixed-precision GEMM
    S = softmax(S / sqrt(d))
    O = S @ V_fp16  (over same tiles)

    // Step 4: Residual KV attention (标准 FP16)
    O += FlashAttention(Q_t, X_res_K, X_res_V)

    // Step 5: Append new K/V to residual
    X_res_K.append(K_t)
    X_res_V.append(V_t)
    if len(X_res_K) == N_r:
        quantize_and_pack(X_res_K, X_res_V) → append to X_pack
        X_res_K, X_res_V = [], []

    // Step 6: FFN + LM head → next token
    token_{t+1} = argmax(lm_head(FFN(O)))
```

术语一般如何实现？如何使用？

BitDecoding 通过 Residual Kernel（在线量化+pack）和 Packing Kernel（在线 dequant+compute）实现。典型 4-bit 配置下 dequant 开销 <15% kernel time（vs QServe ~50%）。2-bit 下 <35%（dequant 更昂贵但 memory savings 更大）。开源：https://github.com/OpenBitSys/BitDecoding。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

---

## Query Transformation for Tensor Cores (MHA/MQA/GQA Unification)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Query Transformation 是 BitDecoding 中使 MHA/MQA/GQA 在 Tensor Cores 上高效执行的统一方法。Decode 阶段 Q length 仅为 1 token，M 维度极小（低 arithmetic intensity），直接 QK^T 会严重 underfill Tensor Cores。利用 GQA/MQA 的 KV head sharing 特性：GQA 下 gq = hq/hkv 个 query heads 共享同一组 KV head，将 [1, gq, hkv] 的 Q tensor reshape 为 [gq, hkv]（gq 个 query heads 被当作一个更大的 GEMM 块并行处理），饱满 Tensor Core mma fragment，提升 warp occupancy 和吞吐。

从算法pipeline角度拆解术语。

```
// 原始 decode Q layout（underfill TC）
Q: [batch=1, num_heads=hq, head_dim=d]
// QK^T: M=1×hq → small M → underfill TC tile T_m
// 对于 standard attention, batch=1, seq_q=1 → M=hq 但对于每个 KV head 仅 gq queries

// BitDecoding Query Transformation:
// 对于 GQA (gq = hq/hkv > 1):
Q_reshaped = Q.view(1, hkv, gq, d)  // 根据 KV head 分组
           = Q.view(hkv, gq, d)      // 每个 KV head 有 gq 个 queries
// 现在对于每组 KV head:
//   M_effective = gq (e.g., LLaMA-3.1-8B: gq=4, d=128 → GEMM [4,128]×[128,L])
//   Tile 填充率提升 gq× → arithmetic intensity 提升 → TC efficiency 提升

// Attention computation per KV head group:
for each KV head group i:
    Q_i = Q_reshaped[i]            // [gq, d]
    K_i = K_cache[i]               // [L, d]
    S_i = Q_i @ K_i^T / sqrt(d)    // [gq, L] — larger M → TC efficient
    A_i = softmax(S_i)
    O_i = A_i @ V_cache[i]         // [gq, d]
O = concat and reshape O_i back to [1, hq, d]
```

术语一般如何实现？如何使用？

实现在 BitDecoding 的 kernel launch 前。仅需一次 PyTorch view/reshape（零开销）。对于 MHA（gq=1），transformation 无效果；对 GQA（gq≈4-8）效果显著；对 MQA（gq=hq=32-64）效果最大。在 RTX 4090 上 BitDecoding GQA 3× speedup vs QServe 仅 1.4×——QServe CUDA Core-only 无法利用 GQA 带来的 compute intensity 提升。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

---

## Accumulative Attention Score (A2S)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Accumulative Attention Score (A2S) 是一种通过累加 softmax 后注意力分数来评估 token 重要性的方法。其核心直觉：在 Attention 操作中，少数 token（如主语、核心动词）持续获得高注意力分数，而多数 token（如介词、冠词）分数很低。A2S 将每个 token 在所有 query 下的 Attention Score 累加，作为区分重要与不重要 token 的指标。

在 Encoder 模型中（SpAtten）：$A_k^l = \sum_{i=1}^l \sum_{h=1}^H \sum_{q=1}^N S_{q,k}^{i,h}$，跨层累积。

在 Decoder 模型中（H2O）：$A_{n,k}^{l,h} = \sum_{q=k}^{n} S_{q,k}^{l,h}$，沿 Generation Step 累积。由于 Causal Mask 的存在（$S_{q,k}^{l,h} = 0, \forall q < k$），第 k 个 token 只累积 n-k 次分数，导致早期 token 天然拥有更多累积次数，形成不公平比较。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**H2O 中 A2S 的计算流程**：

```
// 初始化 A2S 数组
A = zeros(N)  // N: 当前序列长度

// 每个 Generation Step 更新 A2S
for n in 1..max_gen:
    S_n = softmax(Q_n @ K^T / sqrt(d_k))  // [1, N], 下三角非零
    for k in 1..n:  // k <= n (causal mask)
        A[k] += S_n[k]

    // 保留 A 值最高的 K 个 token
    keep_indices = top_k(A, K)
    K_cache = K_cache[keep_indices]
    V_cache = V_cache[keep_indices]
    A = A[keep_indices]

    // 注：H2O 还保留一半预算用于 local cache (最近 token)
```

**A2S 的核心问题**：
第 k 个 token 被累积 n-k+1 次（第 k 步到第 n 步），第 n 个 token 仅被累积 1 次。Softmax 值恒非负，累积次数越多 A2S 越大→早期 token"虚胖"，近期重要 token 被误杀。

术语一般如何实现？如何使用？

A2S 以即插即用方式集成到推理流程。对每层每个 head 维护一个长度为当前序列的 A2S 向量。每步 Attention 计算后更新 A2S 并做 top-k 选择。H2O 的实现开源，A2SF 作为其改进版本也在 https://github.com/Dirac-Notation/A2SF 开源。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder
- SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

---

## Causal Mask (Masked Self-Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Causal Mask 是 Transformer Decoder 中自注意力操作的遮蔽机制。它在 Attention Score 矩阵 S = QK^T / sqrt(d_k) 上施加一个上三角为 -inf（或极小值）的 mask，使得 Softmax 后对应位置概率为 0，实现"当前 token 只能看到它自身和之前生成的 token，不能看到未来 token"的自回归约束。

数学形式：$S'_{i,j} = S_{i,j} + M_{i,j}$，其中 $M_{i,j} = -\infty$ if $i < j$ else $0$。Softmax 后 $\text{softmax}(-\infty) = 0$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**带 Causal Mask 的 Self-Attention**：

```
Q = x @ W_Q  // [1, N, d_k] 或 [N, d_k]
K = x @ W_K  // [1, N, d_k]
V = x @ W_V  // [1, N, d_v]

scores = Q @ K^T / sqrt(d_k)  // [N, N]
// 应用 Causal Mask
for i in 0..N:
    for j in i+1..N:
        scores[i][j] = -inf  // 上三角置 -inf

attn = softmax(scores)  // 下三角非零，上三角为 0
output = attn @ V        // [N, d_v]
```

**Causal Mask 对 A2S 的影响**：
因上三角为 0，第 k 个 token 在第 q 步（q < k）时 $S_{q,k} = 0$，只有在第 k 步及之后才产生非零分数。这导致：
- Token 1：累积 N 次
- Token 2：累积 N-1 次
- Token k：累积 N-k+1 次

A2S 值天然按 token 位置排序——早期 token 的累积优势掩盖了真实重要性。

术语一般如何实现？如何使用？

Causal Mask 是所有自回归 Transformer（GPT、LLaMA、OPT 等）的标准组件。在 PyTorch 中通常用 `torch.nn.Transformer.generate_square_subsequent_mask()` 或 `torch.triu()` 生成。FlashAttention 等优化 kernel 将其融入计算流程，避免显式构建 N×N mask 矩阵。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder

---

## Attention Sink

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention Sink 是 StreamingLLM（Xiao et al., ICLR 2024）发现的 LLM 注意力现象：首 token（bos token 或序列第一个 token）在所有 generation step 中持续获得异常高的 Attention Score，即使该 token 语义上并不重要。其根本原因在于 Softmax 归一化必须让概率之和为 1——深层 head 中大量 token 的注意力分数趋近于 0，模型将"多余的注意力质量"倾泻到首 token 这个始终可用的"接收槽"上。

在 KV Cache 压缩中，Attention Sink 具有双重意义：(1) 必须保留首 token，否则准确率急剧下降；(2) 好的 token 剪枝方法应当自然地保留 Attention Sink token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Attention Sink 在 token 剪枝中的角色**：

```
// StreamingLLM 策略（固定框架）
cache = [sink_tokens] + [recent_tokens]  // 始终保留前 4 个 token + 最近 W 个 token

// A2SF 中 Attention Sink 被自然保留
// 即使 α 衰减历史分数，sink token 每步都获得极大 Score
// Σ α^{n-q} × S_q,0 的衰减被连续高分抵消 → sink token 保持高分
```

**A2SF 论文验证（Section 4.3）**：
A2SF 即使施加 Forgetting Factor，Sink Token 仍然被选中且保持高分——"这是因为即使施加遗忘因子，Sink Token 每步都输出较大的值，相比其他 token 保持较高值"。

术语一般如何实现？如何使用？

在 KV Cache 管理策略中，最简单的是始终将首 token 加入"不可逐出"列表。StreamingLLM 固定保留前 4 个 token。H2O 和 A2SF 中，由于首 token 的 A2S/A2SF 分数天然很高，通常无需特殊处理即被自然保留。这已被 A2SF 的实验验证（A2SF 下 Attention Sink token 被正常选中）。

在 HISA 中，Attention Sink 现象被用于层级索引的 block 选择策略：HISA 的 block-level 粗过滤中，**首 block 和尾 block 被强制包含**在候选集 C_t 中（C_t = TopK(J_{t,:}, m) ∪ {first block, last block}）。原因是首 block 包含 attention sink tokens（模型将"多余注意力"倾泻至此），尾 block 包含局部上下文（query 自身附近的 tokens 通常有高 attention）。这一强制包含确保 HISA 的粗过滤不会因 block-level 近似而丢失这两类关键信息。同样的策略也见于 MoBA (Lu et al., 2025)。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder
- DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs
- HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation
- LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important
- MagicPIG: LSH Sampling for Efficient LLM Generation
- PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference
- Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation
- Star_Attention__Efficient_LLM_Inference_over_Long_Sequences (Star Attention leverages attention sink phenomenon: anchor blocks shift block-local attention sinks to a single sink, enabling block-local attention to approximate global attention)
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs
- TokenButler: Token Importance is Predictable

**Q-Filters 论文中的 Attention Sink 处理**：Q-Filters 在 NIAH 实验中不对前两层进行 KV Cache 压缩（"we do not compress key-value pairs in the first two layers of the models"），这与 Attention Sink 通常在浅层表现更显著的现象一致。由于 Q-Filters 通过 Key 在 Query 主方向上的投影来估计重要性，而 Attention Sink token 的 Key 通常在该方向上有显著投影（因其高注意力），因此 Q-Filters 自然倾向于保留 Attention Sink token。

**MagicPIG贡献的几何解释**（Section 3）：MagicPIG对Attention Sink提供了三个关键几何发现——(1) k_sink（首token的key state）在不同输入token下朝向几乎不变（相似度>0.99）；(2) k_avg（所有key的均值向量）在不同输入句下朝向稳定（相似度>0.9）；(3) k_sink和k_avg几乎相反（余弦相似度-0.9~-0.8）。这三者形成了Figure 2c的几何：query接近k_0方向，key集中在相反方向的窄锥中（除sink token外）。该几何解释了为什么TopK搜索困难（q和k分布方向相反→NN搜索效果差），以及为什么LSH需要centering（否则几乎所有key与q的碰撞概率接近0）。

**SPECPREFILL 中的 Attention Sink 缓解**：SPECPREFILL 识别到 Attention Sink 现象会扭曲 token 重要性估计（首几个 token 倾向于获得过高的注意力权重），通过 **Look-ahead Decoding**（向前解码 N=8 步，聚合 N 个解码 token 的注意力而非仅依赖最后一个 token 的注意力）来缓解这一偏差。这种策略与仅使用最后 token 注意力的方法相比，显著减少了对 sink token 的过拟合选择。

---

## Λ-like Attention Mask (Lambda-shaped Attention Mask / Streaming Attention Mask)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Λ-like Attention Mask 是一种特殊的因果注意力遮罩模式，最早由 LM-Infinite（Han et al., 2023）和 StreamingLLM（Xiao et al., 2023b）提出，因形状类似希腊字母 Λ 而得名。该 mask 使每个 query token 仅 attend 到：(1) 序列开头的固定数量 token（attention sinks，构成 Λ 的左分支）；(2) 自身之前的最近 W 个 token（sliding window，构成 Λ 的右分支）。序列中间的 token 被完全跳过（mask 置为 -∞），不参与 attention 计算。

数学表达：M_streaming[i,j] = 0 if (j ≤ S) or (i - j ≤ W and j ≤ i) else -∞，其中 S 为 sink token 数量，W 为 recent window 大小。

从算法pipeline角度拆解术语。

```
# Λ-like Mask 的 attention 计算
def streaming_attention(Q, K, V, S, W):
    """
    Q, K, V: [batch, heads, seq_len, dim]
    S: number of attention sinks (initial tokens)
    W: recent window size
    """
    seq_len = Q.shape[2]

    # 构建 Λ-like mask
    mask = torch.full((seq_len, seq_len), float('-inf'))

    # 每个 query i 可以 attend 到:
    for i in range(seq_len):
        # (1) attention sinks: 前 S 个 token
        mask[i, :min(S, i+1)] = 0
        # (2) recent window: 最近 W 个 token
        start = max(0, i - W + 1)
        mask[i, start:i+1] = 0

    scores = (Q @ K.transpose(-1, -2)) / sqrt(d)
    scores = scores + mask  # -∞ → softmax 后概率为 0
    attn_weights = softmax(scores, dim=-1)
    return attn_weights @ V
```

**在 DuoAttention 中的应用**：Λ-like mask 是 streaming heads 的 attention 计算核心，替代 causal mask。在 deployment 中 streaming heads 仅使用该 mask 计算 attention：attn_streaming = softmax(QK^T ⊙ M_streaming)V。DuoAttention 使用 S=64（sink tokens）和 W=256（recent tokens），KV cache 保持 constant O(S+W) 大小。在 chunked pre-filling 中，每个 chunk 的 KV 计算完毕后立即 prune streaming heads 的 KV cache（仅保留 sink + recent），下一 chunk 的 attention 仅需处理 constant 数量历史 token。

术语一般如何实现？如何使用？

StreamingLLM 首次系统化使用该 mask 实现 infinite-length 流式推理（无需重新预训练即可处理超预训练长度的输入）。DuoAttention 将其限定在 streaming heads 类别上使用（retrieval heads 使用完整 causal mask）。实现方式：修改标准 FlashAttention kernel 的 mask 输入（将 Λ-like mask 作为 block-wise attention mask），或使用 FlashInfer 的 block-sparse attention 模块（Guo et al., 2024）。DuoAttention 训练阶段用 block-sparse approximation 加速 Λ-like attention 计算。

涉及论文标题：
- DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Forgetting Factor α（0 < α < 1）是 A2SF 方法的核心创新。它是一个指数衰减系数，在累积 Attention Score 时反复乘以历史分数，使得越早产生的 Attention Score 收敛至 0，消除 Causal Mask 导致的 token 位置偏差。

公式：$A_{n,k}^h = \sum_{q=1}^n \alpha^{n-q} \times S_{q,k}^h$

展开形式：$A_{n,k}^{h} = S_{n,k}^{h} + \alpha \cdot S_{n-1,k}^{h} + \alpha^{2} \cdot S_{n-2,k}^{h} + \dots + \alpha^{N-k} \cdot S_{k,k}^{h}$

α 的含义：
- α = 1.0：等价于 H2O 的原始 A2S，无衰减，全量历史累积
- α = 0.0：完全忽略历史，仅用当前步 Attention Score 决定重要性
- α ∈ [0.1, 0.3]：论文实验发现的最优范围，主要考虑近期历史
- α → 0：快速收敛，仅看近期趋势——适合区分度高的数据集
- α → 1：缓慢收敛，长历史仍影响——适合需记忆早期关键信息的数据集

该设计受人类遗忘曲线（Ebbinghaus Forgetting Curve）启发——其简化形式为指数型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**A2SF 伪代码**：

```
alpha = 0.2  // Forgetting Factor, [0.1, 0.3] 为最优范围

// 初始化
A = zeros(N)  // 每个 token 的 A2SF 分数

for n in 1..max_gen:
    // Step 1: 衰减所有已有分数
    A *= alpha  // 所有历史分数 × α（等效于 α^{new_q - old_q}）

    // Step 2: 加入当前步的 Attention Score
    S_n = softmax(Q_n @ K^T / sqrt(d_k))  // [1, n]
    for k in 1..n:
        A[k] += S_n[k]  // 注意：S_n[k] 是 q=n, key=k 的分数

    // Step 3: 按 A2SF 分数选择保留 token
    keep_indices = top_k(A, K)
    evict_indices = rest

    // A2SF 不分配 local cache，全量 budget 用于 selective
```

**关键运算细节**：
- 每步先对全局 A 做 `A *= alpha`，一次乘法即实现所有历史分数多一次衰减
- 再累加当前步分数：`A[1:n] += S_n[1:n]`
- 时间复杂度 O(n)，与 H2O 的 O(n) 相同，无额外开销

术语一般如何实现？如何使用？

A2SF 以即插即用方式集成到 HuggingFace Transformers 推理流程。每次 Attention 计算后，调用 `cache.evict_by_a2sf(k, alpha)` 完成选择。用户只需设置 α 和 cache_ratio 两个参数。代码开源：https://github.com/Dirac-Notation/A2SF。A2SF 可与后续处理不重要 token 的技术（No Token Left Behind 的量化、Get More with LESS 的低秩分解、Keyformer 的 Gumbel-Softmax）兼容叠加。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder

---

## Token Merging (Token 合并) in Multi-modal LLMs

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token Merging 是一种通过合并相似 token 来减少多模态 LLM 输入 token 数量的训练无关技术。在 AIM 论文中，Token Merging 发生在 Visual Encoder 之后、LLM 输入之前：将 N⁰ 个视觉 token embedding 按余弦相似度配对，每对最相似的 token 取均值合并，迭代多轮直至达到目标保留率。与 ToMe（Token Merging for ViT）在 Vision Encoder 每层内做合并不同，AIM 的 Token Merging 在 Encoder 输出后一次性执行，对 Encoder 架构无侵入，即插即用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Token Merging 伪代码（AIM 风格）**：

```
// 输入：visual tokens v ∈ R^{N×D}，目标保留率 r_merge
// 输出：合并后的 tokens v' ∈ R^{(N×r_merge)×D}

def token_merging(v, r_merge):
    current = v
    N = len(current)
    I = ceil(log2(1/r_merge))  // 所需迭代轮数
    
    for iter in 1..I:
        // 将 tokens 分成 A（偶数位置）和 B（奇数位置）
        A = current[0::2]  // 偶数索引
        B = current[1::2]  // 奇数索引
        
        // 计算 A 和 B 之间的余弦相似度矩阵
        sim = cosine_similarity(A, B)  // [N/2, N/2]
        
        // 对 A 中每个 token，找到 B 中最相似的 token
        for i in 0..len(A):
            j_star = argmax(sim[i])      // B 中最匹配的索引
            merged = (A[i] + B[j_star]) / 2.0  // 取平均合并
            current.append(merged)
        
        N = len(current)
    
    return current  // 共 N⁰ × (1/2)^I = N⁰ × r_merge 个 token
```

**视频场景的特殊处理**：合并仅在单帧内（spatial）进行，不跨帧（temporal）合并。消融实验表明跨帧合并在低保留率下显著损害性能（如 r=3.1% 时 temporal merging 47.4 vs spatial 52.3 on VideoMME），因为跨帧合并破坏 token 的时序顺序。

术语一般如何实现？如何使用？

在 AIM 实现中，Token Merging 以函数形式插入到 LLaVA 推理流程的 Visual Encoder 输出与 LLM 输入之间。默认配置：video LLM 保留 25%（迭代 2 轮），image LLM 保留 12.5%（迭代 3 轮）。额外计算开销极小：video 场景 88.25 GFLOPs，仅占 Qwen2-7B 推理的 0.6%。代码开源：https://github.com/LaVi-Lab/AIM。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

---

## Visual Token Pruning in Multi-modal LLM (多模态 LLM 视觉 Token 剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Visual Token Pruning in MLLM 是一种在多模态 LLM 推理过程中，基于重要性度量从 LLM 各层中逐步移除视觉 token，以减少 Attention 和 FFN 计算量的训练无关技术。与纯文本 Decoder 的 KV Cache token pruning（为节省显存）不同，MLLM 的视觉 token pruning 主要目标是减少计算量——因为视觉 token 数量（数千个）远超文本 token。AIM 的核心发现：LLM 早期层做跨模态融合需要 visual tokens，后期层专注文本推理，可以大幅剪除 visual tokens。基于此，AIM 设计分层剪枝策略：早期层全保留，中期层线性递减，后期层全部移除，同时文本 token 始终不剪。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**AIM Visual Token Pruning 伪代码**：

```
// 超参数：l1（开始剪枝层）, l2（完全移除层）, L（总层数）
// 保留率调度：r^l = 1 if l<l1; 1 - (l-l1)/(l2-l1) if l1≤l≤l2; 0 if l>l2

for l in 1..L:
    // Step 1: 标准 Self-Attention（当前层输入含 visual + text tokens）
    x = [v^{l-1}; t]  // visual + text tokens
    A = softmax(Q @ K^T / sqrt(d_k))  // Attention 权重矩阵
    
    // Step 2: PageRank 重要性计算
    s = ones(N_v + N_t) / (N_v + N_t)  // 均匀初始化
    for _ in range(num_iterations):
        s = (A @ s) / (N_v + N_t)  // PageRank 迭代（公式 1）
    
    // Step 3: 仅对 visual tokens 按 s 排序剪枝
    v_scores = s[:N_v]  // 仅取 visual token 分数
    k = int(N_v * r^l)  // 当前层保留数量
    keep_idx = top_k(v_scores, k)
    
    // Step 4: 移除被剪枝的 visual tokens
    v^l = v^{l-1}[keep_idx]  // 保留的 visual tokens
    // text tokens 始终全部保留
```

**与文本 Decoder Token Pruning 的关键差异**：
| 维度 | 文本 Decoder (H2O/A2SF) | MLLM (AIM) |
|------|------------------------|-----------|
| 剪枝对象 | KV Cache 中的 token | 当前层输入中的 visual token |
| 目标 | 减少显存 + 计算 | 减少计算（FLOPs） |
| 重要性度量 | A2S/A2SF（累积 Attention Score） | PageRank（Attention 图稳态分布） |
| 文本 token | 可剪（选择性保留关键 token） | 不剪（文本始终全保留） |
| 层级策略 | 每层独立决策 | 全局 Scheduler 控制逐层保留率 |

术语一般如何实现？如何使用？

AIM 实现中，Token Pruning 以 hook 方式插入到 Qwen2/Vicuna LLM 的每层 Attention 后。默认配置：video（Qwen2-7B, 28 layers）l₁=14, l₂=22；image（Vicuna-1.5-7B, 32 layers）l₁=13, l₂=21。额外开销极小：video 场景 Token Pruning 仅 4.18 GFLOPs（<0.03% LLM FLOPs）。代码开源：https://github.com/LaVi-Lab/AIM。

注意：AIM 的 Token Pruning 与 FlashAttention 不兼容（需要显式 Attention 矩阵计算 PageRank），但与量化（quantization）和稀疏注意力（sparse attention）兼容。Dynamic-LLaVA 的 Vision-Language Context Sparsification 与 AIM 的关键差异：Dynamic-LLaVA 同时稀疏化 vision 和 language 上下文（而非仅 vision），使用可学习 predictor（而非 PageRank 启发式），且适用于 decoding w/ KV cache 的在线 KV 压缩场景。

**TransPrune 的 Token Pruning 方法**：TransPrune 从与 attention/similarity 完全不同的视角出发——利用 token 表征在 LLM 层间传播时的变化（Token Transition）来反映 token 重要性。核心包括：(1) **TTV（Token Transition Variation）**：测量每个 token 在 self-attention 和 FFN 模块中表征的幅度变化（L2 norm 比率）和方向变化（cosine similarity），仅依赖 token 自身的输入→输出变化（无需 inter-token 依赖），天然避免 attention 位置偏差；(2) **IGA（Instruction-Guided Attention）**：计算 instruction tokens 对 image tokens 的单向 attention 权重，引入任务相关语义监督；(3) **Accumulation**：跨中间层（7-12）累积 TTV 值，稳定剪枝决策。TransPrune 的 TTV 计算仅需模块输入/输出 tensor（兼容 FlashAttention），IGA 仅计算 instruction→image 的单向 attention（非完整 N×N attention map）。在 LLaVA-v1.5-7B 上降低 TFLOPs 至 40.8% 时性能几乎无损。代码将开源于 https://github.com/liaolea/TransPrune。与 AIM 关键差异：TransPrune 为 training-free（无需学习），使用 token 自身 transition 信号（非 attention 图结构），且与 FlashAttention 完全兼容。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification
- TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model

---

## PageRank-based Token Importance Scoring

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PageRank-based Token Importance Scoring 是一种基于图中心性算法评估 token 在 Attention 图中重要性的方法。将 Self-Attention 权重矩阵 A^l ∈ R^{(N+M)×(N+M)} 视为有向图的邻接矩阵，每个 token 是图中的一个节点，A_{i,j} 表示 token j 对 token i 的"投票"权重。在此基础上运行 PageRank 算法——稳态分布中得分高的节点即"被众多重要节点所关注的节点"——得分者即为重要的 token。

与 A2S（简单累加 Attention Score）不同，PageRank 递归考虑"被谁关注"：一个 token 若被高分 token 关注，其自身得分也高。这更准确反映 Attention 图中的信息流结构。

公式（AIM 公式 1）：

$$s_i^l = \frac{1}{N^l + M^l} \sum_{j=1}^{N^l + M^l} \mathbf{A}_{i,j}^l \cdot s_j^l$$

其中 $\mathbf{A}^l$ 为 softmax 归一化的 Attention 权重矩阵，$s$ 初始均匀分布。AIM 仅对 visual tokens 按 PageRank 分数排序剪枝，text tokens 始终保留。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**PageRank 用于 Token 重要性评估的完整流程**：

```
def pagerank_token_importance(A, num_visual, num_text, tol=1e-6):
    """
    A: Attention 权重矩阵 [N_v+N_t, N_v+N_t]
    num_visual: 视觉 token 数量
    num_text: 文本 token 数量
    返回: visual tokens 的 PageRank 重要性分数 [N_v]
    """
    N = num_visual + num_text
    
    // 初始化：均匀分布
    s = ones(N) / N
    
    // 幂迭代（Power Iteration）直到收敛
    for _ in range(max_iter):
        s_new = A.T @ s / N
        if norm(s_new - s) < tol:
            break
        s = s_new
    
    // 仅返回 visual token 的分数
    return s[:num_visual]
```

**为什么使用 PageRank 而非直接累加 Attention Score**：
- A2S：只看"∑_{query} 对 key 的关注总量"——与 token 位置相关，早期 token 累积次数多
- PageRank：考虑 Attention 图的全局结构——"被重要 token 关注的 token 更可能是重要的"——递归纠正位置偏差

术语一般如何实现？如何使用？

在 AIM 实现中，每层 Attention 计算后取 softmax 后的 A 矩阵（不用 FlashAttention），运行少量迭代的幂迭代法（矩阵-向量乘法 O(N²)）。因为仅对 visual tokens 排序，且 visual token 数量随层递减，实际开销很小（4.18 GFLOPs for Qwen2-7B）。该技术与 Zero-TPrune（Wang et al., CVPR 2024）中的 PageRank-based token pruning 一脉相承，AIM 将其扩展到多模态场景。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

---

## Adaptive Inference (自适应推理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Adaptive Inference 是一种动态调整模型推理计算量的范式：根据当前资源约束（FLOP budget、延迟要求）、输入复杂度或期望准确率，动态选择推理配置（模型深度、token 数量、精度等），而非始终以最大计算量推理。Han et al. (TPAMI 2021) 将 Adaptive Inference 定义为"动态神经网络"的核心类别。AIM 将其引入多模态 LLM 领域：通过调节 Token Merging 的保留率 r_merge 和 Token Pruning 的 Scheduler 参数 (l₁, l₂)，实现从 2.5% 到 100% FLOPs 的连续可调范围，仅损失 <13% 准确率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**AIM Adaptive Inference 配置空间**：

```
# 自适应推理配置映射
configurations = {
    "extreme_efficiency": {"r_merge": 0.016, "l1": 14, "l2": 22},  # 2.51 FLOPs, 50.9 acc
    "high_efficiency":   {"r_merge": 0.031, "l1": 14, "l2": 22},  # 3.72 FLOPs, 52.3 acc
    "efficiency":        {"r_merge": 0.063, "l1": 14, "l2": 22},  # 6.17 FLOPs, 53.6 acc
    "balanced":          {"r_merge": 0.125, "l1": 14, "l2": 22},  # 11.14 FLOPs, 56.4 acc
    "default":           {"r_merge": 0.25,  "l1": 14, "l2": 22},  # 14.76 FLOPs, 58.2 acc
    "high_quality":      {"r_merge": 0.50,  "l1": None,"l2": None}, # 46.48 FLOPs, 58.5 acc
    "base_model":        {"r_merge": 1.0,   "l1": None,"l2": None}, # 99.63 FLOPs, 58.2 acc
}

def adaptive_inference(image_or_video, flop_budget):
    // 1. 根据 FLOP budget 选择最接近的配置
    config = select_config(flop_budget, configurations)
    
    // 2. Token Merging
    visual_tokens = merge_by_cosine_sim(visual_tokens, config.r_merge)
    
    // 3. Token Pruning with Scheduler
    for l in 1..L:
        visual_tokens, text_tokens = forward_layer(visual_tokens, text_tokens, l)
        if config.l1 and l >= config.l1:
            k = len(visual_tokens) * retention_ratio(l, config.l1, config.l2)
            visual_tokens = prune_by_pagerank(visual_tokens, k)
```

**自适应范围（Video, LLaVA-OV-7B）**：
- FLOPs span: 2.51 TB ~ 99.63 TB（40× 范围）
- Accuracy range: 50.9 ~ 58.5 VideoMME（<13% 降幅）
- Prefill time: 10.12 ms ~ 439.58 ms

术语一般如何实现？如何使用？

AIM 的自适应推理不需要修改模型权重，所有配置共享同一预训练模型。部署时根据目标设备（AR 眼镜、手机、PC、机器人）的计算资源选择配置。配置空间可预先采样并制表，运行时查表即可。代码开源：https://github.com/LaVi-Lab/AIM。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

---

## Multi-modal LLM (MLLM / 多模态大语言模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Multi-modal LLM (MLLM) 是一种将预训练 LLM 与视觉编码器结合，使其能理解图像/视频等多模态输入的模型架构。典型 pipeline：Visual Encoder (如 CLIP-ViT) → Projection/Adapter (如 MLP) → LLM (如 LLaMA/Qwen/Vicuna) → Text Decoder。视觉数据经 Encoder 转换为 visual token embeddings，经 Adapter 投影到 LLM 的 embedding 空间，与 text tokens 拼接后送入 LLM 做多模态推理。

AIM 论文使用的基座模型：LLaVA-OneVision-7B（video, Qwen2-7B backbone）、LLaVA-1.5-7B（image, Vicuna-v1.5-7B backbone）。video LLM 从视频均匀采样 32~192 帧，每帧经 ViT 编码为数百 tokens，总计可达数千 visual tokens。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**MLLM 标准推理 Pipeline（以 LLaVA-OV-7B 为例）**：

```
// 1. 视觉编码阶段
frames = sample_frames(video, num_frames=32)  // 均匀采样 32 帧
for frame in frames:
    v_tokens_per_frame = ViT(frame)  // [H×W/patch², D_vis]
// 总计：32 × (336/14)² = 32 × 576 ≈ 18432 tokens → Adaptive Pooling → ~2304 tokens

// 2. 投影阶段
v = MLP_Adapter(v_tokens)  // D_vis → D_llm (embedding 对齐)

// 3. 多模态推理阶段
text_tokens = Tokenizer(prompt)
x = Concat([v; text_tokens])  // [N_v + N_t, D_llm]
for l in 1..L:
    x = TransformerLayer_l(x)  // Self-Attn + FFN + Residual

// 4. 文本生成
output = lm_head(x[-1])  // 取最后一个 token 的 logits
```

**AIM 在此 pipeline 中的插桩位置**：
- Token Merging：在 Step 2 和 Step 3 之间（Adapter 输出→Token Merging→LLM 输入）
- Token Pruning：在 Step 3 的每个 TransformerLayer 的 Self-Attention 之后

术语一般如何实现？如何使用？

主流 MLLM 实现包括 LLaVA 系列（https://github.com/haotian-liu/LLaVA）、Qwen-VL（https://github.com/QwenLM/Qwen-VL）等。AIM 以即插即用方式集成到 LLaVA 推理流程中，无需修改模型权重。安装依赖：PyTorch 2.3.1, CUDA 12.1, 修改版 transformers/lmms-eval/qwen-vl-utils。

Dynamic-LLaVA 在 LLaVA-1.5 标准 pipeline 第 l=2 层 decoder 后插入两个轻量 predictor，分别对 vision 和 language token 做 keep/discard 决策，决策共享至所有后续层。训练时冻结 Vision Encoder 和 Projector，仅更新 LLM 和 Predictor 参数（LLM lr=5e-6, Predictor lr=2e-4）。额外的 MLLM 效率问题：prefill 仅执行一次，image token 减少的收益在 decoding 阶段随 output token 数量增长而逐渐湮没，因此需同时稀疏化 vision 和 language 上下文。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification

---

## Visual Token (视觉 Token)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Visual Token 是多模态 LLM 中由视觉编码器（如 CLIP-ViT）从图像/视频帧中提取的 embedding 向量序列。每张图像被划分为 patch（如 14×14 像素），每个 patch 经 ViT 编码为一个 D 维向量，构成一个 visual token。Video LLM 中对多帧采样，total visual token 数量 = 帧数 × 每帧 patch 数 × 每 patch token 数。

AIM 论文揭示的关键发现：multi-modal LLM 中 visual tokens 存在极高冗余——仅需 ~25% 的 visual tokens 即可维持 video 推理性能，FLOPs 却降低 77%。冗余原因：(1) 相邻 patch 高度相似；(2) 视频帧间大量重叠；(3) 许多 visual tokens 不携带对推理有贡献的信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Visual Token 的生命周期（MLLM 中的张量流转）**：

```
// 输入：Image I ∈ R^{H×W×3} 或 Video V ∈ R^{T×H×W×3}

// 1. 编码阶段
patches = PatchEmbed(I)  // [N_patches, patch_dim]
v = ViT(patches)          // [N_patches, D_vis]
// 例：CLIP-ViT-L/14：336×336 → 576 patches → 576 visual tokens × 1024 dims

// 2. 投影到 LLM 空间
v_llm = MLP_Adapter(v)    // [N_patches, D_llm]
// 例：576 × 4096 (Vicuna-7B)

// 3. 视频场景：多帧拼接
v_all = Concat([v_llm_frame1, v_llm_frame2, ..., v_llm_frameT])
// 例：32 frames × 576 = 18432 tokens → Adaptive Pooling → ~2304 tokens

// 4. 进入 LLM
x = Concat([v_all; text_tokens])  // [2304 + M, 4096]
// FLOPs ∝ (N_v + M)² —— visual tokens 主导计算量
```

**AIM 中的 Visual Token 压缩**：
- Token Merging（LLM 前）：N_v → N_v × 0.25，仅保留 25%
- Token Pruning（LLM 内部）：l₁~l₂ 层间 N_v 线性递减 → l₂ 层后 N_v = 0
- 文本 token 不受影响

术语一般如何实现？如何使用？

Visual tokens 由视觉编码器（CLIP-ViT、SigLIP 等）自动生成。AIM 无需修改编码器，在编码器输出后插入 Token Merging，在 LLM 层间插入 Token Pruning。Visual token 的数量控制是 AIM 自适应推理的核心杠杆。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

---

## Pruning Scheduler (剪枝调度器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Pruning Scheduler 是 AIM 中控制 LLM 各层 visual token 保留率 r^l 的分段线性函数。它决定了 visual token 在哪层开始被剪枝（l₁）、在哪层被完全移除（l₂）、以及中间层如何递减。设计依据：消融实验发现——visual tokens 在早期 LLM 层做跨模态融合时必须全保留（第 8 层剪枝导致 58.0→41.9 的性能崩溃），但在晚期层可全部移除（第 22 层剪枝几乎无影响：58.0→58.1）。

公式（AIM 公式 2）：

$$r^l = \begin{cases} 1, & \text{if } l < l_1 \\ 1 - k(l - l_1), & \text{if } l_1 \le l \le l_2 \\ 0, & \text{if } l > l_2 \end{cases}$$

其中 $k = \frac{1}{l_2 - l_1}$ 为递减斜率。l₁ 和 l₂ 是用户可调的 Scheduler 参数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**AIM Scheduler 伪代码**：

```
def pruning_scheduler(l, l1, l2, L):
    """
    l: 当前层索引 (1-indexed)
    l1: 开始剪枝的层
    l2: visual token 完全移除的层
    返回: 当前层 visual token 保留率 r^l
    """
    if l < l1:
        return 1.0                        // 早期层：全保留（跨模态融合阶段）
    elif l1 <= l <= l2:
        return 1.0 - (l - l1) / (l2 - l1)  // 中期层：线性递减
    else:  # l > l2
        return 0.0                        // 晚期层：全部移除（纯文本推理阶段）

// 实际使用：
for l in 1..L:
    r_l = pruning_scheduler(l, l1=14, l2=22, L=28)
    num_keep = int(len(visual_tokens) * r_l)
    visual_tokens = prune_by_pagerank(visual_tokens, num_keep)
```

**不同 (l₁, l₂) 配置的实验结果（VideoMME）**：
| l₁ | l₂ | FLOPs (TB) | VideoMME | 说明 |
|----|----|-----------|----------|------|
| 28 | 29 | 22.90 | 58.0 | 仅 Merging，不剪枝 |
| 14 | 22 | 14.76 | 58.2 | 默认配置（最优 trade-off） |
| 7  | 22 | 12.01 | 56.8 | 更快剪枝，轻微降性能 |
| 14 | 15 | 12.10 | 54.3 | 更快完成剪枝，显著降性能 |
| 7  | 8  | 6.71  | 41.9 | 极早剪枝，性能崩溃 |

**关键发现**：l₂ 比 l₁ 更关键——只要 l₂ ≥ 22（层总数为 28），即使 l₁ 提前到 7，性能仅从 58.2 降至 56.8。但若 l₂ 提前到 15，性能锐降至 54.3。说明晚期层（>22）的 visual tokens 几乎无贡献。

术语一般如何实现？如何使用？

Scheduler 是纯算术函数，无额外计算开销。在 AIM 实现中，Scheduler 参数 (l₁, l₂) 作为超参数在推理前指定。用户可根据目标 FLOP budget 从预标定的配置表中选择合适的 (l₁, l₂) 组合。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

---

## Cross-modal Fusion in LLM Layers (LLM 层的跨模态融合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cross-modal Fusion in LLM Layers 是多模态 LLM 中 visual tokens 和 text tokens 在 Transformer 层中通过 Self-Attention 进行信息交互的过程。AIM 通过消融实验发现：LLM 的不同层对 cross-modal fusion 的需求差异显著——早期层依赖 visual tokens 建立视觉-文本对齐（跨模态融合阶段），后期层主要进行 text-only reasoning（文本推理阶段），visual tokens 可被安全移除。

这一发现直接指导了 AIM 的 Scheduler 设计：l₁~l₂ 之间的层是 fusion→text-only 的过渡带。默认配置 l₁=14, l₂=22 对应 Qwen2-7B 的 28 层中前 50% 全保留 visual tokens，50%~79% 线性递减，79% 后全部移除。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Cross-modal Fusion 在各层的角色变化**：

```
LLM Layer 1~13 (l < l₁): Cross-modal Fusion 阶段
    x = [v; t]  // visual + text tokens 拼接
    A = softmax(Q@K^T / sqrt(d_k))  // Attention 包含 visual↔text 交互
    // visual tokens 通过 Attention 将视觉信息传递给 text tokens
    // 结论：此阶段 visual tokens 必须全保留

LLM Layer 14~21 (l₁ ≤ l ≤ l₂): 过渡阶段
    // visual↔text 交互逐渐减少
    // 每次剪除 1/(l₂-l₁) 比例的 visual tokens
    // 关键 visual tokens 在 PageRank 中得分高，被保留到最后

LLM Layer 22~28 (l > l₂): Text-only 推理阶段
    x = [t]  // 仅 text tokens
    // Self-Attention 和 FFN 全部在 text tokens 上
    // visual tokens = 0，计算量大幅降低
```

**消融证据**：
- l₂=8（第 8 层后移除全部 visual tokens）：VideoMME 从 58.0 暴跌至 41.9
- l₂=15（第 15 层后移除）：54.3（部分恢复，但仍低）
- l₂=22（第 22 层后移除）：58.1（几乎无损）
- l₂=29（不移除）：58.0

结论：前 14 层（50%）需要 visual tokens 做 fusion；22 层（79%）后 visual tokens 完全无用。

术语一般如何实现？如何使用？

Cross-modal fusion 是 MLLM 架构的内在属性，无需额外实现。AIM 通过分析各层的 Attention 行为（PageRank 分数分布、visual↔text attention 比例）和消融实验（在不同层剪枝 visual tokens 观察性能影响）来量化 fusion→text-only 的转变点。这些发现可用于指导其他 MLLM 的效率优化。

涉及论文标题：
- AIM: Adaptive Inference of Multi-Modal LLMs via Token Merging and Pruning

---

## Token Pruning in Transformer Decoder

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token Pruning 是一种在 Transformer Decoder 推理过程中，基于某种重要性度量从 KV Cache 中移除不重要 token 的 KV 项，以减少显存占用和 Attention 计算量的技术。不同于 Encoder 模型的 token pruning（仅减少计算量），Decoder 中的剪枝同时减少显存和计算。剪枝粒度可以是 layer-wise（同一层所有 head 使用相同剪枝 mask）、head-wise（H2O 提出，每 head 独立维护分数和选择）或 token-wise。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Decoder Token Pruning 通用框架**：

```
// 超参数
cache_ratio = 0.2  // 保留 token 比例
K = int(N * cache_ratio)

// 每层每头独立维护
for layer in 1..L:
    for head in 1..H:
        // Step 1: 计算 Attention（可能加速）
        scores = flash_attn(Q, K_cache, V_cache)

        // Step 2: 更新重要性分数（不同方法的核心差异）
        importance = update_importance(scores, history, method)
        // method ∈ {"A2S" (H2O), "A2SF" (本文), "Gumbel-Softmax" (Keyformer), ...}

        // Step 3: 选择保留 token 并逐出
        keep_idx = top_k(importance, K)  // 强制保留 attention sink
        K_cache = K_cache[keep_idx]
        V_cache = V_cache[keep_idx]
        importance = importance[keep_idx]
```

**主要方法对比**：
| 方法 | 重要性度量 | 分配策略 |
|------|----------|---------|
| Local Attention | 位置（最近 token） | 全量 W 个最近 |
| H2O | A2S | 50% local + 50% A2S selection |
| A2SF | A2S + 遗忘因子 α | 100% selective |
| Keyformer | Gumbel-Softmax A2S | selective |
| StreamingLLM | 位置锚定 | 4 sink + W recent |

术语一般如何实现？如何使用？

H2O 是首个将 A2S-based token pruning 应用于 Decoder 的工作（NeurIPS 2024），代码开源。A2SF 基于 H2O 框架改进，也开源。在实际集成中，需修改 HuggingFace Transformers 的 Attention 层，在 `forward()` 后追加缓存管理逻辑。所有方法都是训练无关的即插即用方案，不增加 latency 开销。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder

---

## Heavy-Hitter (H2) Token

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Heavy-Hitter (H2) Token 是 H2O 论文提出的概念，指在 Attention 运算中贡献大部分 Attention Score 累积值的少数 token。H2O 的实验显示，约 20% 的 token（H2 token）贡献了绝大部分注意力分数，若将其移除则准确率急剧下降。H2 token 构成了 KV Cache Token Pruning 的理论基础：只需保留 H2 token 即可保持模型性能。

H2 的识别依赖 A2S：累积较高的 token 即为 H2 token。然而 A2SF 论文指出，由于 Causal Mask 的偏差，H2O 识别的 H2 往往偏向早期 token——不一定真正重要。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**H2 Token 在 A2SF 的 re-interpretation**：

H2O 假设：A2S 高 → 重要（语言现象）
A2SF 揭示的偏差：A2S 高 ← 位置靠前 + 累积次数多（数学必然）

因此 A2SF 不否定 H2 概念本身，而是修正识别 H2 的方式——用带衰减的 A2SF 替换原始 A2S，使真正的 H2 token（每步都输出高分而非仅靠位置优势）脱颖而出。

术语一般如何实现？如何使用？

H2 Token 选择是 A2S/A2SF 方法的输出步骤。在实现中，`top_k(A, K)` 返回的 K 个 token 即为 H2 token。H2O 为每个 head 独立维护 H2 列表，允许不同 head 关注不同重要 token。A2SF 继承了这一 head-wise 设计。

**KV-Distill 对 H2 方法的分析**：KV-Distill 将 H2O 分为两个范式——H2A（问题感知）：将问题和上下文拼接后计算累积注意力，利用问题信号扫描上下文中的关键信息；H2I（问题无关）：仅在上下文内部计算累积注意力。KV-Distill 实证发现：H2I 在问题无关范式下性能急剧下降——例如 LLAMA-3 SQuAD 上 H2I 25% retention 准确率仅 56.6%（vs uncompressed 87.6%，H2A 84.0%）。这说明 H2 的注意力累积机制在没有问题信号引导时无法有效识别对未知问题重要的 token。

**NACL 对 H2O/H2 的分析**：NACL 揭示了 H2O 的 attention bias 问题——H2 token 的高 A2S 分数部分源于位置偏差（初始和最近 token 天然高 attention），而非真正的语义重要性。在 LongBench passkey retrieval 中，H2O 30% budget 仅 PR-Zh=3.7/PR-En=5.0（vs Full 8.0/10.1），证明 H2 token 选择遗漏了中间位置的关键信息。NACL 通过 proxy tokens（仅用末尾 ~10% token 的 attention）替代全量累加，配合 head-wise RANDOM EVICTION，将 passkey retrieval 提升至 NACL 30%=6.8/9.0。

涉及论文标题：
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder
- KV-Distill: Nearly Lossless Learnable Context Compression for LLMs
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

---

## Approximate Attention (近似注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Approximate Attention 是一类通过计算注意力矩阵的稀疏子集而非完整 N×N 矩阵来加速 Transformer 推理的技术。与 FULLATTN（精确注意力，如 FLASHATTN/RINGATTN/ULYSSES）保留完整计算结果不同，approximate attention 的核心思想是：注意力矩阵中大部分元素的 Softmax 值趋近于 0，仅少数 token pair 贡献有意义的 attention score，因此可以跳过无关计算，以微小精度损失换取显著加速。

Approximate attention 可大致分为三类：(1) 静态稀疏模式——如 MINFERENCE 为每个 attention head 预先分配固定的稀疏模式（如 diagonal, vertical+slash），仅计算模式指定的 attention score；(2) 基于 KV cache 压缩的——如 H2O、SNAPKV、LOCRET，通过选择保留重要的 KV pair 减少后续 attention 计算量；(3) 基于 anchor 的分布式近似——如 STARATTN 和 APB，在序列并行框架中通过 anchor block 和/或 passing blocks 近似全局 attention。

从算法pipeline角度拆解术语。

**APB 中的 Approximate Attention 流程（4-stage pipeline）**：

```
Stage 1: Context Splitting
  文档 d 按 H 个 host 均分，每 host 持有 B_h
  Anchor block A = {query, d[0:l_a]}（远小于 STARATTN）

Stage 2: Block Compression
  每 host 独立打分，选 Top-l_p KV pair:
  s = R([Q_h, K_h, V_h])           // retaining heads MLP 推理
  B_h^C = top_k(KV_h, l_p, by=s)   // 仅保留 l_p 个最重要 KV

Stage 3: Communication  
  (K^C_{1:H}, V^C_{1:H}) = AllGather(K_h^C, V_h^C)
  P_h = concat(K^C_{1:h-1}, V^C_{1:h-1})  // passing block

Stage 4: Computation
  K = [K_a, K_p^C, K_h], V = [V_a, V_p^C, V_h]
  A = flash_attn_with_mask(Q, K, V, M')   // 修改后 mask
  P_h 在 attention 后丢弃，不进入 FFN
```

**与 FULLATTN 的计算量对比**：
APB FLOPs/forward 远小于 FULLATTN（Table 9），因为：(a) 每 host 仅处理 l_b = n/H 长度，而非完整 n；(b) passing block 仅 l_p 长度（默认 l_p = l_b/8）；(c) anchor block 仅 l_a 长度（l_a = l_b/4 或 l_b/8）。

术语一般如何实现？如何使用？

APB 的 approximate attention 通过定制 FLASHATTN kernel（修改 attention mask）+ retaining heads（LOCRET 训练的 MLP）+ AllGather 通信实现。开源：https://github.com/thunlp/APB。STARATTN 通过大 anchor block + 无通信实现 approximate attention（https://github.com/NVIDIA/Star-Attention）。MINFERENCE 通过静态 head-specific 稀疏模式实现。Quest 通过 query-aware page-level selection + upper-bound criticality estimation 实现 approximate attention：先加载 per-page min/max Key metadata 计算 criticality score，Top-K 选择关键 page，仅对选中 page 执行 FlashAttention（https://github.com/mit-han-lab/Quest）。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

---

## Retaining Heads / KV Cache Compression with Learned Importance (LOCRET)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Retaining Heads 是 LOCRET（Huang et al., 2024）提出的可训练 KV cache 重要性评分模块。它在每层 transformer 注入一个小型 MLP（两层线性变换，中间维度 $d_{\mathbf{R}}=1024$），接收当前层 $[\mathbf{Q}, \mathbf{K}, \mathbf{V}]$ 的拼接作为输入，输出每个 token 在每个 KV head 的 **Causal Importance Score (CIS)** $\tilde{\mathbf{S}}[k] \in \mathbb{R}^{h/g}$。CIS 反映该 KV cache unit 对未来 token 理解的重要性——高分 token 被保留在 KV cache budget 中，低分 token 在 chunked prefill 过程中被 evict。

核心设计：$\tilde{\mathbf{S}} = \mathbf{R}([\mathbf{Q}, \mathbf{K}, \mathbf{V}]) = \sigma([\mathbf{Q}, \mathbf{K}, \mathbf{V}]\mathbf{W}_1)\mathbf{W}_2$，其中 $\mathbf{W}_1 \in \mathbb{R}^{(d_m + 2d_{kv}) \times d_{\mathbf{R}}}$ 和 $\mathbf{W}_2 \in \mathbb{R}^{d_{\mathbf{R}} \times h/g}$。对 GQA 模型，重要性估计跨 head 联合进行但输出 per-head 分数。训练时 LLM backbone 完全冻结，仅训练 retaining head 参数（占模型总参数 2.5%-8%），训练开销 < 1 GPU 小时。

Ground truth CIS 定义：$\mathbf{S}[k]_j := \max_p (\mathbf{Q}_j \mathbf{K}_j^T)_{p,k}$，即所有 answer token 对该 prefix token k 的最大 pre-softmax attention score。训练 Loss：$\text{Smooth-}\mathcal{L}_1(\tilde{\mathbf{S}}, \mathbf{S}) + \alpha \mathcal{L}_2(\tilde{\mathbf{S}}[k], \tilde{\mathbf{S}}[k+1])$，后者为相邻 token 平滑正则化。训练数据为任意 long-context QA SFT 数据集（LongAlpaca/LongAlign/Anti-Haystack 均可，性能差异极小）。

与 H2O/A2S/SNAPKV 不同，retaining heads 的 CIS 评分是 **causal** 的——仅依赖当前及之前 token，不依赖后续 token。这使其在 chunked prefill 中评分始终准确（无 local-global discrepancy），且与 FlashAttention 完全兼容（不需要 materialize attention matrix）。推理开销极低（Table 20: w/ R 19153 tok/s vs w/o R 20304 tok/s at 4096 ctx，差距来自系统波动）。

从算法pipeline角度拆解术语。

**LOCRET 原始论文中的 Retaining Head 训练与推理完整流程**：

```
// ============ 训练阶段 ============
// 1. 注入 retaining head
for layer i in 1..L:
    R_i = MLP(d_R=1024)   // W1 ∈ R^{(d_m+2d_kv)×d_R}, W2 ∈ R^{d_R×h/g}

// 2. 前向传播收集 ground truth CIS
Q_i, K_i, V_i = layer_i(H_{i-1})   // 正常 attention 前向
// 对每个 prefix token k:
S_i[k]_j = max_p (Q_i_j @ K_i_j^T)_{p,k}
// p 遍历所有 answer token

// 3. Retaining head 预测
Ŝ_i = R_i([Q_i, K_i, V_i])          // 同时输入所有 head 的信息

// 4. 训练（backbone 冻结）
Loss = Smooth-L1(Ŝ_i, S_i) + α * L2(Ŝ_i[k], Ŝ_i[k+1])
// 仅更新 W1, W2

// ============ 推理阶段（chunked prefill + eviction）============
// Hyperparameters: b (budget), B (chunk size), n_s (stabilizers), n_loc (local)
for chunk in chunks(0, L - n_loc, B):
    K_chunk, V_chunk, score_chunk = forward_with_retaining_heads(chunk)
    K_cache = concat(K_cache, K_chunk)
    V_cache = concat(V_cache, V_chunk)
    score_cache = concat(score_cache, score_chunk)
    if not last_chunk:
        score_cache[-n_s:] = +inf     // stabilizers protection
    indices = topk(score_cache, b)    // keep highest CIS
    K_cache, V_cache, score_cache = K_cache[indices], ...

// 处理最后 n_loc tokens（保证不被 evict）
K_cache, V_cache = forward_final(local_tokens, K_cache, V_cache)
output = model.generate(K_cache, V_cache)
```

**LOCRET-Q 变体（query-aware）**：训练时将 query 最后 $l_a$ 个 token 前置到训练序列首部，使 CIS labels 感知 query。推理时 query 在序列首部，确保所有 eviction 感知 query。这使得 LOCRET-Q 在 RULER 等 query-driven benchmark 上可用（75.54% vs LOCRET 34.33%）。

**训练配置**（LOCRET 原始）：
- 数据：LongAlpaca（默认），LongAlign/Anti-Haystack 也可
- 步数：3000 steps, batch_size=1, max_seq_len=10240
- 优化器：AdamW (lr=5e-4, linear scheduler, warmup=2000)
- Loss: Smooth-L1 + α·L2（α=0.0025）
- 训练开销：Phi-3-mini-128K 0.47h, Llama-3.1-8B 0.80h（单 A800）
- Retaining head 参数占比：8% (Phi-3-mini) / 2.5% (Llama-3.1-8B)

术语一般如何实现？如何使用？

Retaining heads 以即插即用方式注入每层 transformer 的 attention block 之后。训练时 backbone 冻结，仅 MLP 参数更新。推理时 retaining head 在每个 chunked prefill step 执行一次 MLP forward，开销可忽略。与 FlashAttention 完全兼容。保留 pre-RoPE KV cache 并从起始位置重新分配连续 position embedding 以增强上下文连续性。支持 MHA 和 GQA 架构。开源：LOCRET 原始 https://github.com/huangyuxiang03/Locret；APB 复用实现 https://github.com/thunlp/APB。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs
- LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices

---

## Anchor Block (in Distributed Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Anchor Block 是分布式近似注意力（STARATTN/APB）中的一个关键设计：在每 host 的 local context block 前 prepend 一段包含输入序列开始部分 token 的"锚点块"，使每个 host 的注意力计算能感知文档起始位置的上下文。其设计动机源于 Attention Sink 现象——模型需要起始 token 作为注意力质量的"倾泻槽"，否则每个独立 block 会自行产生 attention sink 导致注意力分布失真。

STARATTN 使用 l_a = l_b 的大 anchor block（anchor 与 local block 等长），APB 将其减小到 l_a = l_b/4 或 l_b/8。APB 还在 anchor block 中嵌入 query token，使 retaining heads 感知查询相关信息。

从算法pipeline角度拆解术语。

**Anchor Block 在 APB 中的作用**：

```
// 构造
A = {q_1, ..., q_{l_q}, d_1, ..., d_{l_a}}  // query + 文档开头 l_a 个 token
position_ids(A) = 0, 1, ..., l_q+l_a-1      // 从 0 开始的位置编码

// 每 host 的 context layout (host h > 1)
context = [A, P_h, B_h]  // anchor → passing → local

// Attention 计算
Q = [Q_a, Q_h]
K = [K_a, K_p^C, K_h]    // anchor KV + compressed passing KV + local KV
V = [V_a, V_p^C, V_h]
// anchor block 的 KV 参与所有 token 的 attention 计算

// Anchor block hidden states 通过 FFN
H_a^out = FFN(A_a)       // anchor 的输出被保留
```

**STARATTN vs APB anchor block 差异**：
| 维度 | STARATTN | APB |
|------|----------|-----|
| Anchor 长度 l_a | l_b (与 local block 等大) | l_b/4 或 l_b/8 |
| Anchor 内容 | 文档开头 l_a token | query + 文档开头 l_a token |
| FFN 开销 | 大（l_a = 16K → 大量重复计算） | 小（l_a = 4K） |
| 是否嵌入 query | 否 | 是 |

消融实验（Table 3）证明 anchor block 是最关键组件——移除后 E.MC 从 72 降至 28。

术语一般如何实现？如何使用？

Anchor block 在 tokenization 和 embedding 阶段构造——将 query token 和文档开头 token 拼接后分配连续 position IDs，然后以 prepend 方式与 local context block 合并送入 Transformer。实现无额外复杂度，仅需在输入预处理阶段调整 token 排列。APB 开源：https://github.com/thunlp/APB。

**Star Attention 的 Anchor Block 设计差异**：与 APB 相比，Star Attention 使用 l_a = l_b 的大 anchor block（anchor 与 local block 等长），内容仅为文档开头的 l_a 个 token（不含 query），位置编码保持原始位置。STARATTN 的消融实验（Table 4）证明：(1) anchor block 的内容至关重要——使用常量 token（空格/标点）时准确率降为 0%，使用随机 token 时降 10%；(2) anchor block 的位置编码影响较小——即使随机化位置 ID，准确率仅降约 2%。此外，Star Attention 的 anchor block KV 在阶段一后被丢弃（仅保留 context block 部分的 KV），而 APB 保留了 anchor block 的 hidden states 通过 FFN。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs
- Star_Attention__Efficient_LLM_Inference_over_Long_Sequences

---

## Passing Block (传递块)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Passing Block 是 APB 特有的设计：在每 host 上，由前序所有 host 的压缩 KV cache 拼接而成的上下文块。它在 attention 计算中作为 local context block 和 anchor block 之间的中间层，使当前 host 能"看到"前序 host 中最关键的 KV pair，从而弥补序列并行中 context 不可见的问题。Passing block 在 attention 计算后被丢弃，不参与 FFN 计算，也不持久化存储。

构造方式：Host h 通过 AllGather 获取所有 host 的压缩 KV cache (K^C_{1:H}, V^C_{1:H})，然后取前序 host 的压缩块拼接：P_h = (K_p^C, V_p^C) = (K^C_{1:h-1}, V^C_{1:h-1})。后续 host 的压缩块被忽略。

从算法pipeline角度拆解术语。

**Passing Block 在 APB 层的生命周期**：

```
// Step 1: 本地压缩
K_h^C, V_h^C = compress(K_h, V_h, l_p)    // 每 host 独立压缩

// Step 2: 全局收集
K_all = AllGather(K_h^C)                  // [H*l_p, d]
V_all = AllGather(V_h^C)

// Step 3: 构造 passing block（仅取前序 host）
K_p = K_all[0:(h-1)*l_p]                  // host 1 的 P_1 为空
V_p = V_all[0:(h-1)*l_p]

// Step 4: Attention 计算
K = [K_a; K_p; K_h]                       // anchor + passing + local
V = [V_a; V_p; V_h]
A = flash_attn(Q, K, V, mask=M')

// Step 5: 丢弃 passing block
// P_h 不进入 FFN，不缓存
H_a^out, H_h^out = FFN(A_a, A_h)          // 仅 anchor 和 local 通过 FFN
```

**Passing Block 的作用**（消融实验 Table 3）：
- 有 passing block (No.0)：E.MC = 72.00
- 无 passing block (No.4)：E.MC = 64.00（-8%）
- 有 passing block 但随机选择 KV (No.2)：E.MC = 66.00（retaining heads 关键）

**多 host 扩展性**（Table 4）：有了 passing block，APB 在 32K 输入下 H=2→8 性能稳定在 92-94；而 STARATTN（无 passing block）在 H=8 时降至 84，因 middle context 不可见。

术语一般如何实现？如何使用？

Passing block 通过 NCCL AllGather 通信实现。通信开销极小（0.62ms/block，<1% total），因为只传输 l_p=2K 个 token 的 KV cache（vs l_b=16K 的完整 local context）。AllGather 后本地拼接 KV tensors，送入修改的 FLASHATTN kernel。开源：https://github.com/thunlp/APB。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

---

## Online Softmax for Distributed Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Online Softmax 是一种数值稳定的 Softmax 计算方法，在分布式注意力中用于将各设备上独立计算的部分注意力结果合并为全局精确结果。传统 Softmax 需要两次遍历（一次求 max 用于数值稳定，一次求和归一化），Online Softmax 通过维护 running max 和 running sum 实现单次遍历，使各设备可独立计算部分结果后通过通信合并。

在 RINGATTN 中，online softmax 用于 ring-style 通信：每个 host 计算当前块的 max 和 sum，传递给下一个 host，后者用接收的 (m, l) 修正自己的计算。在 STARATTN/APB 的 decoding 阶段（stage-2），online softmax 用于 Gather+MergeScore：各 host 独立计算 attention，然后通过 Gather 收集各 host 的 (A_h, lse_h)，用 lse（log-sum-exp）合并为全局 attention。

从算法pipeline角度拆解术语。

**APB Decoding 中的 Online Softmax MergeScore**：

```
// 每 host 独立计算（Algorithm 3）
Q, K, V = qkv_proj(H)

if h < H:
    A_h, lse_h = Attention(Q, K_cache[h], V_cache[h])   // 对本地 KV cache
else:  // 最后一个 host
    A_h, lse_h = Attention(Q, [K_cache[H], K], [V_cache[H], V])

// Gather 所有 host 的部分结果
A_1..A_H, lse_1..lse_H = Gather(A_h, lse_h)

// MergeScore: 利用 online softmax 合并
// A_global = Σ_h A_h * exp(lse_h - lse_max) / Σ_h exp(lse_h - lse_max)
lse_max = max(lse_1, ..., lse_H)
weights = [exp(lse - lse_max) for lse in lse_1..lse_H]
A_global = sum(w_h * A_h for w_h, A_h in zip(weights, A_1..A_H)) / sum(weights)

H_out = FFN(A_global)
```

术语一般如何实现？如何使用？

Online Softmax 以 kernel 形式实现在 FLASHATTN 和各类分布式注意力框架中。APB 复用 STARATTN stage-2 的 online softmax 解码方案（Algorithm 3）。在 HuggingFace Transformers 中，通过自定义 attention forward 函数集成。FLASHATTN（Dao, 2024）是 PyTorch 中最广泛使用的 online softmax attention 实现。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

---

## STARATTN (Star Attention / 星形注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

STARATTN（Star Attention，NVIDIA, 2024）是一种结合序列并行与近似注意力的两阶段分布式推理方法。Phase 1（Context Encoding）：将长上下文划分为连续 block，每 block prepend 一个与 block 等大的 anchor block（文档首部），各 host 独立计算 anchor↔local 之间的 block-sparse attention，无跨 host 通信。Phase 2（Query Processing）：query 广播到所有 host，各 host 对本地 KV cache 计算 partial attention，通过 distributed softmax 合并各 host 结果获得全局精确 attention。

APB 相比于 STARATTN 的改进：(1) 使用更小的 anchor block（l_a = l_b/4 vs l_b），减少 FFN 开销；(2) 引入 passing block（压缩前序 host KV cache）弥补 STARATTN 中 middle context 不可见的问题；(3) 使用 retaining heads 而非 anchor-only 做 KV 选择。

从算法pipeline角度拆解术语。

**STARATTN Phase 1（Context Encoding）**：

```
// 序列划分
blocks = split(doc, H)  // [B_1, ..., B_H], each of size l_b
// 每 host（除 host 1）prepend anchor block
for h in 2..H:
    context[h] = [A, B_h]  // A = doc[0:l_b]，与 block 等大

// 每 host 独立计算，无通信
for h in 1..H:
    Q = qkv_proj(context[h])[0]
    K, V = qkv_proj(context[h])[1:]  // anchor + local 的 KV
    A_h = flash_attn(Q, K, V)
    H_h = FFN(A_h)
    // 仅保留 B_h 的 KV cache，A 的 KV cache 丢弃
```

**STARATTN vs APB 对比**：
| 维度 | STARATTN | APB |
|------|----------|-----|
| Anchor 大小 | l_a = l_b (16K) | l_a = l_b/4 (4K) |
| 跨 host 通信 | Phase 1 无 | AllGather (K^C, V^C) |
| Passing Block | 无 | 有（前序 host 的压缩 KV） |
| 多 host 扩展 | 退化（middle context 不可见） | 稳定（passing block 补偿） |
| 速度 (128K, 8 hosts) | 29,600 tok/s | 37,575 tok/s |
| 开源 | https://github.com/NVIDIA/Star-Attention | https://github.com/thunlp/APB |

术语一般如何实现？如何使用？

STARATTN 以两阶段方式集成到 HuggingFace Transformers 的 `model.generate()` 中：prefill 阶段替换为 Phase 1 block-sparse attention，decode 阶段替换为 Phase 2 distributed softmax。APB 在 Phase 1 基础上增加了 block compression（retaining heads）、AllGather 通信和 passing block 构造。STARATTN 开源：https://github.com/NVIDIA/Star-Attention (ICML 2025)。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

## IO Similarity (Input-Output Similarity / 输入输出相似度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

IO Similarity 是衡量 Transformer 模型中某个模块（layer、attention sublayer 或 FFN sublayer）对前向传播贡献程度重要性的度量。其直观含义：如果某个 Transformer 模块的输入向量经过该模块处理后，输出向量与输入向量高度相似（余弦相似度接近 1），说明该模块对信息几乎没有变换——输入"穿过"该模块后变化极小，因此该模块在当前上下文中的重要性较低，可以被跳过（skip）以节省计算。

数学定义：给定两个 n 维向量 $\vec{a}$（输入）和 $\vec{b}$（输出），余弦相似度为：

$$Similarity(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \sqrt{\sum_{i=1}^{n} b_i^2}}$$

AdaSkip 论文通过实验验证了 IO Similarity 与模块重要性之间的强相关性：采用 LeastSkip 策略（跳过 IO Similarity 最低的层，即最重要的层）在仅跳 1 层时 GPT score 即降至 1.0 以下；而 MostSkip 策略（跳过 IO Similarity 最高的层，即最不重要的层）在跳 1/3/5 层时 GPT score 分别为 8.9/6.1/4.2，明显更优。

在长上下文推理中，不同模型（LLaMA3.1-8B-128k、InternLM-7B-8k、Vicuna-v1.5-7B-16k）的 IO Similarity 分布差异极大：InternLM 的高 Similarity 层集中在中部（如 layers 12-14），而 LLaMA3.1 的高 Similarity 层集中在尾部（layers 25-29），且曲线近似单调递增。这种差异性正是 AdaSkip 需要 per-model 自适应学习 IO Similarity 分布的动机。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

在 AdaSkip 的算法 pipeline 中，IO Similarity 是核心决策依据，贯穿 Prefilling 和 Decoding 两个阶段：

**阶段 1：Offline Importance Learning（Prefilling）**
```
// 在 N 个历史样本上累积各 sublayer 的平均 IO Similarity
for each sample i with |T_i| tokens:
    for each sublayer j in attention_sublayers ∪ ffn_sublayers:  // 共 2M 个
        for each token t in Prefilling phase:
            Simi_j += cosine_sim(a_it^j, b_it^j)  // 输入向量 vs 输出向量

Simi_j = Simi_j / sum(|T_i|)  // 归一化为平均 IO Similarity
```

**阶段 2：Sublayer 排序与选择**
```
sorted = sort_by(Simi_j, descending=True)  // Similarity 越高 → 越不重要 → 越应跳过
m = M - M/α  // 根据加速比 α 确定跳过的 layer 数
skipped = sorted[0:2m]  // 跳过 Similarity 最高的前 2m 个 sublayer
```

**阶段 3：Online Importance Learning（Decoding）**
```
// 用前 P 个 decoded token 计算当前上下文的 IO Similarity
for each FFN sublayer j not in skipped:
    for token t in 1..P:  // online learning window
        Simi_j^P += cosine_sim(a_t^j, b_t^j)
    Simi_j^P /= P

// 用阈值 β 筛选额外可跳过的 FFN
β = min(Simi_j for j in skipped)  // skipped set 中的最低 Similarity
for each FFN j not in skipped:
    if Simi_j^P > β:
        skipped^P += j  // 当前上下文也高 Similarity → 也可跳过
```

**IO Similarity 跨任务泛化性**（关键发现）：
AdaSkip 发现 offline 学习的 IO Similarity 在不同数据集间具有高 hit rate（如 TriviaQA → MFieldQA: ATTN top-10 hit 9.31/10, FFN top-10 hit 9.56/10），说明 IO Similarity 分布是模型内在特性而非任务特定特征。

**IO Similarity 的 Phase 特性**（Observation 3）：
Attention sublayer 和 FFN sublayer 在 prefill 和 decoding 阶段有相似的趋势但波动程度不同。特别是 FFN sublayer 在 decoding 阶段的 IO Similarity 高于 prefill 阶段——这意味着更多 FFN 可在 decoding 阶段额外跳过。

术语一般如何实现？如何使用？

IO Similarity 的计算可通过对推理框架的前向传播进行 hook（拦截）来实现。具体做法：
1. 在 HuggingFace Transformers 的每个 attention 和 FFN 模块前后注册 forward hook，捕获输入和输出 hidden states
2. 推理过程中实时计算 `cosine_sim(a, b) = (a·b)/(|a||b|)`，利用 PyTorch `torch.nn.functional.cosine_similarity` 高效实现
3. 对多 token 的 Similarity 取平均，将 per-sublayer 统计量存储为元数据
4. 后续推理的每个 sublayer 入口处检查是否 ∈ skipped set，如是则执行 identity shortcut
5. 跳过的 sublayer 输出用 Scale_j * a 近似补偿

AdaSkip 开源：https://github.com/ASISys/AdaSkip

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference

## Layer-wise Skipping (逐层跳过)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Layer-wise Skipping 是一种在 LLM 推理过程中跳过 Transformer Layer 执行的加速策略，通过省略特定位置层的 Self-Attention 和 FFN 计算（两模块同时跳过）来减少计算量和延迟。该方法基于关键观察：并非所有 Transformer 层对生成质量同等重要，某些层的输出与其输入高度相似（高 IO Similarity），可以被安全跳过。

AdaSkip 论文将现有 Layer-wise Skipping 策略分为三类：
1. **Early Skipping (SkipDecode)**：始终跳过模型前几层（除第一层外）。支持 batching 操作但可能跳过重要层。
2. **Periodic Skipping (Unified Skipping)**：在中间层按固定频率周期性跳层（如每 N 层跳 1 层）。支持 batching 但无法捕捉不同层的重要性差异。
3. **Early Exit**：在每层计算后判断条件（如置信度），一旦满足条件立即退出，跳过后续所有层。可能忽略后面更重要的层，且通常需要额外训练 classifier 或微调模型。

Layer-wise Skipping 的核心限制：按整层（attention + FFN）粒度进行 skip，忽略了 attention sublayer 和 FFN sublayer 各自独立的重要性分布特征。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Early Skipping (SkipDecode) 伪代码**：
```
for layer l in 1..L:
    if l <= K and l != 1:
        skip  // identity shortcut
    else:
        x = Attention(LayerNorm(x)) + x
        x = FFN(LayerNorm(x)) + x
```

**Periodic Skipping (Unified Skipping) 伪代码**：
```
for layer l in 1..L:
    if l > 1 and l < L and (l - 1) % N == 0:
        skip  // identity shortcut
    else:
        x = Attention(LayerNorm(x)) + x
        x = FFN(LayerNorm(x)) + x
```

**Early Exit 伪代码**：
```
for layer l in 1..L:
    x = Attention(LayerNorm(x)) + x
    x = FFN(LayerNorm(x)) + x
    if confidence_score(x) > threshold:
        return lm_head(x)
return lm_head(x)
```

术语一般如何实现？如何使用？

Layer-wise Skipping 通常以 hook 或 model patching 方式集成到 HuggingFace Transformers 推理流程中。实现可以是：替换指定层的 forward 方法为 identity function（直接返回输入），或在 forward 入口处判断是否属于 skip set。Early Exit 变体需要额外训练 confidence classifier 或微调模型参数以补偿信息损失。

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference

## Sublayer-wise Skipping (子层粒度跳过)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Sublayer-wise Skipping 是 AdaSkip 提出的核心创新——在 Transformer 推理中按 sublayer 粒度（独立评估 Attention sublayer 和 FFN sublayer）而非整层粒度进行选择性跳过。其关键观察（Observation 2）：Attention 和 FFN sublayer 的 IO Similarity 分布独立且特征不同——在长上下文推理中，Attention sublayer 的平均 IO Similarity 更高且更集中（如 LLaMA3.1-8B-128k 最后 11 层 attention Similarity 稳定在 ~0.97，FFN 仅 ~0.95），意味着更多 attention 可被跳过，且跳过 attention 还能节省 KV cache。

整层跳过每次 skip 同时省略 2 个 sublayer（attention + FFN），而 sublayer-wise skipping 每次 skip 1 个 sublayer。由于 IO Similarity 分布不同，sublayer-wise 有更多加速机会——尤其在长上下文场景下 attention 的 O(n²) 开销远大于 FFN 的 O(n)，优先跳过更多 attention 能获得更大加速。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 整层跳过（baseline）：每次 skip 同时跳过 attention + FFN
for layer l in 1..L:
    if l in layer_skip_set:
        x = x  // identity
    else:
        x = Attention(LayerNorm(x)) + x
        x = FFN(LayerNorm(x)) + x

// Sublayer-wise 跳过（AdaSkip）：独立判断 attention 和 FFN
for layer l in 1..L:
    if attn_idx[l] in skipped_set:
        x = x  // skip attention
    else:
        x = Attention(LayerNorm(x)) + x
    if ffn_idx[l] in skipped_set:
        x = x  // skip FFN
    else:
        x = FFN(LayerNorm(x)) + x
```

**Sublayer 排序与选择**（AdaSkip 核心）：
```
all_sublayers = [(Simi_attn[1], 'attn', 1), ..., (Simi_ffn[M], 'ffn', M)]
sorted = sort(all_sublayers, by=Simi, descending=True)
skipped = sorted[0:2m]  // 2M 个子层统一按 Similarity 排序，取前 2m 个
```

术语一般如何实现？如何使用？

需要对 HuggingFace Transformer 做细粒度 hook：每个 attention 和 FFN 子模块分配唯一的 sublayer index（index ∈ [0, 2M-1]），分别捕获输入/输出 hidden states 计算各自的 IO Similarity，并在 forward 中为 attention 和 FFN 分别在入口处插入 skip 判断条件。Skip 时用 Scale_j 补偿。

AdaSkip 开源：https://github.com/ASISys/AdaSkip

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference

## Offline Importance Learning (离线重要性学习)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Offline Importance Learning 是 AdaSkip 在 Prefilling 阶段使用的 sublayer 重要性学习方法。由于 Prefilling 阶段前无任何可用信息指导 sublayer skipping 决策，AdaSkip 利用历史推理任务中累积的 IO Similarity 统计量为新任务的 skipping 提供依据。

核心 Insight：历史 Prefilling 特征与新的 Prefilling 特征之间具有高度相关性——跨数据集 top-K hit rate 实验（Table 1）验证：如用 TriviaQA 学习的 ATTN similarity 在 MultiFieldQA 上 top-10 hit rate 达 9.31/10，FFN 跨数据集 hit rate 也在 9.38-9.56/10。说明 IO Similarity 分布是模型内在特性，可跨任务共享。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Input: N 个历史推理样本，M 个 Transformer Layer，共 2M 个 sublayer

// Step 1: 累积各 sublayer 的 IO Similarity 和 Scale Factor
for sublayer j in 1..2M:
    Simi_j = 0.0; Scale_j = 0.0; total_tokens = 0
    for sample i in 1..N:
        for token t in 1..|T_i|:
            a = get_input_hidden(sublayer=j, token=t)
            b = get_output_hidden(sublayer=j, token=t)
            Simi_j += (a·b)/(|a||b|)        // 公式(2)
            Scale_j += |b|/|a|               // 公式(3)
            total_tokens += 1

// Step 2: 归一化
Simi_j /= total_tokens; Scale_j /= total_tokens

// Step 3: 排序——Similarity 越高越应跳过
sorted_all = sort(zip(Simi, range(2M)), by=Simi, descending=True)

// Step 4: 根据加速比 α 确定跳过数量
m_skip = M - M/α; num_skip = 2 * m_skip
skipped_set = sorted_all[0 : num_skip]
// 在 Prefilling 中：j ∈ skipped_set → output = Scale_j * input
```

术语一般如何实现？如何使用？

一次性 profiling 过程：选定代表性历史数据集（如 AdaSkip 使用的 TriviaQA、MultiFieldQA-en、2WikiMQA），运行一次完整 Prefilling 推理并 hook 各 sublayer 输入/输出 hidden states，计算 average IO Similarity 和 Scale Factor 存储为 per-model metadata。后续对该模型的所有推理任务复用此 metadata。

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference

## Online Importance Learning (在线重要性学习)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Online Importance Learning 是 AdaSkip 在 Decoding 阶段用于发现额外 FFN sublayer 跳过机会的方法。基于 Observation 3（FFN 在 decoding 阶段 IO Similarity 高于 prefill），利用当前上下文前 P 个 decoded token（online learning window）的 IO Similarity 信息，动态识别出高于离线学习阈值 β 的额外 FFN sublayer，加入 skip set 以进一步加速 decoding。

核心 Insight（Table 2）：仅用初始少量 decoded token 就能高命中率预测后续 decoding 中不重要 sublayer。Window size 从 5→20 时 hit rate 显著提升，20→40 趋稳，P≈20 即可。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Input: offline 学习的 skipped_set, P=online learning window size
// Step 1: 前 P 个 token 全 sublayer 执行
for token t in 1..P:
    for sublayer j in all sublayers:
        output = forward_sublayer(j, input)
        if sublayer_type_j == FFN:
            Simi_j^P += cosine_sim(input_j, output_j)

// Step 2: 平均
for FFN j: Simi_j^P /= P  // 公式(5)

// Step 3: 阈值 β
β = min(Simi_j for j in skipped_set)

// Step 4: 找出当前上下文高于 β 的额外 FFN
for FFN j not in skipped_set:
    if Simi_j^P > β:
        extra_skipped.append(j)

// Step 5: 合并 skip set
skipped^P = skipped_set ∪ extra_skipped
// 第 P+1 token 起用 skipped^P 跳过
```

术语一般如何实现？如何使用？

实现要点：P≈20 即可获得稳定 hit rate；仅对 FFN 做 online learning（attention skipping 完全由 offline 确定）；online overhead 仅 P 个 token 的全量 forward。额外 FFN skip 同样使用 offline 学习的 Scale_j 补偿。

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference

## Scale Factor Compensation (比例因子补偿)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Scale Factor Compensation 是 AdaSkip 中用于补偿 sublayer 跳过造成信息损失的技术。当 sublayer 被跳过时，输出直接等于输入（identity shortcut），但会丢失 sublayer 对向量模长的变换。Scale Factor 补偿通过将输入按历史平均模长比缩放，使近似输出更接近原始输出。

$$Scale_{j} = \frac{\sum_{i=1}^{N} \sum_{t=1}^{|T_{i}|} \frac{\|\vec{b}_{it}^{j}\|}{\|\vec{a}_{it}^{j}\|}}{\sum_{i=1}^{N} |T_{i}|}$$

跳过时近似输出：$\vec{b}_{it}^{\hat{j}} = Scale_j \cdot \vec{a}_{it}^j$

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 在 Offline Importance Learning 中同时累积
Scale_j = average(||output|| / ||input||) across all historical samples

// 推理时 skip sublayer 应用补偿
if j in skipped_set:
    b_hat = Scale_j * a    // 跳过 + 模长补偿
else:
    b = forward_sublayer(j, a)
```

有效性前提：Residual connection 使输入输出模长变化微小，当夹角不大时（高 Similarity），仅模长缩放即可有效补偿。高 Similarity = 小夹角 = Scale Factor 补偿有效。

术语一般如何实现？如何使用？

Scale Factor 与 IO Similarity 一起在 Offline Importance Learning 阶段被计算和存储。每个 sublayer 对应一个 Scale_j。使用时仅需 bypassed sublayer 的输入乘以 Scale_j 作为近似输出——O(1) 乘法 per skipped sublayer per token，额外开销可忽略。

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference

## Early Exit (提前退出 / 早停策略)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Early Exit 是一种 Layer-wise Skipping 策略，在每层 Transformer 计算后评估条件（如置信度 score、entropy、或 classifier 输出），一旦条件满足立即退出推理，跳过后续所有层，用当前 hidden states 直接生成输出。代表工作包括 CALM（Schuster et al. 2022）、LITE（Varshney et al. 2023）、EE-LLM（Chen et al. 2024）。

与 AdaSkip 的 Offline/Online Importance Learning 不同，Early Exit 是条件式动态跳过——跳过的层数不固定。缺点是：(1) 可能跳过后面更重要的层；(2) 通常需要额外训练 classifier 或微调模型；(3) 在长上下文 decoding 中错误逐层累积导致质量显著退化（AdaSkip 实验中 Vicuna Rouge-L 在两个 summarization 任务上降至 <4.0）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Confidence-based Early Exit
for layer l in 1..L:
    x = Attention(LayerNorm(x)) + x
    x = FFN(LayerNorm(x)) + x
    logits_l = lm_head(x)
    confidence = max(softmax(logits_l))
    if confidence > threshold:
        return logits_l
return lm_head(x)

// Classifier-based Early Exit
for layer l in 1..L:
    x = TransformerLayer_l(x)
    exit_prob = classifier_l(x)  // trained per-layer classifier
    if exit_prob > 0.5:
        return lm_head_l(x)
return lm_head(x)
```

术语一般如何实现？如何使用？

实现分类：Confidence-based（无需训练，用 entropy/logit margin 判断，但长文本质量差）；Classifier-based（训练 per-layer classifier，需额外数据）；Fine-tuning-based（如 LITE 的 instruction tuning 使中间层也能生成好输出）。

涉及论文标题：
- AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference

## α-entmax (Alpha-entmax Sparse Activation Function)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

α-entmax 是 softmax 的稀疏泛化，由 Peters et al. (2019, ACL) 在《Sparse Sequence-to-Sequence Models》中提出。它基于 Tsallis α-entropy 构建，通过参数 α > 1 控制输出概率分布的稀疏程度。核心公式：
$$\alpha\text{-entmax}(\mathbf{s})_i = [(\alpha - 1)s_i - \tau]_{+}^{1/(\alpha - 1)}$$
其中 [x]_+ = max(x, 0) 即 ReLU 截断，τ ∈ R 是归一化常数（阈值），保证输出和为 1。

α 值与稀疏度关系：α → 1：逼近 softmax（稠密，全非零）；α = 1.5：适中稀疏（实践中 ~95% 注意力权重为零）；α = 2.0：sparsemax（高度稀疏 ~99% 零）；α → ∞：逼近 argmax。核心性质：可微分稀疏性——输出天然含有精确零，且整个过程可微支持端到端梯度训练；数据依赖性——稀疏模式由输入 logits 自适应决定，无需预定义固定 mask。

从算法pipeline角度拆解术语。

**α-entmax 替代 softmax 的 Attention Pipeline**：
```
S = QK^T / √d                     // n×n (不物化)
// 对每行 i: 用 Halley-Bisection 求解 τ_i (3 次迭代)
P_i = [(α-1)S_i - τ_i]_+^{1/(α-1)} // S_ij < τ_i/(α-1) → 0
O = PV                              // 零权重不贡献
```
**α 退火训练**（AdaSplash）：从 α=1.0 (dense) 线性增至目标值 (1.5/2.0)，over 1B tokens，避免直接稀疏训练的不稳定性。

术语一般如何实现？如何使用？

核心难点是求 τ：Sorting-based（α=1.5/2.0 精确，O(n log n)，GPU 效率低）；Bisection（线性收敛，~23 次迭代）；Halley-Bisection（三次收敛，~3 次迭代）。GPU 实现：利用 α-entmax Jacobian 的稀疏性加速反向传播；块方式累积 f/f'/f'' 避免物化 S。使用方式：`pip install adasplash`，`output = adasplash(Q, K, V, alpha=1.5)`。

涉及论文标题：
- AdaSplash: Adaptive Sparse Flash Attention

## Halley-Bisection Algorithm

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Halley-Bisection 是 AdaSplash 提出的混合求根算法，快速求解 α-entmax 中的归一化阈值 τ。结合 Halley 方法的三次收敛与 Bisection 的保证收敛：$$\tau_H = \tau - \frac{2 f(\tau) f'(\tau)}{2 f'(\tau)^2 - f(\tau) f''(\tau)}$$ 其中 f/f'/f'' 为 α-entmax 阈值方程及其一/二阶导。Fail-safe：当 Halley 更新超出 [τ_lo, τ_hi] 回退 bisection。结果：3 次迭代到 machine precision（vs. bisection 23 次），~15× runtime 加速（2.38ms vs 36.67ms at n=8192），1.75× 内存节省。

从算法pipeline角度拆解术语。

```
Input: s ∈ R^n, α, T
1. τ_lo = max(s)-1, τ_hi = max(s)-n^{1-α}, τ = (τ_lo+τ_hi)/2
2. for t=1..T:
     // Block-wise 累积 f,f',f'' (GPU SRAM)
     for each block j: S_blk = Q_i@K_j^T
         f   += Σ [(α-1)S_blk-τ]_+^{1/(α-1)} - 1
         f'  += -1/(α-1) Σ [(α-1)S_blk-τ]_+^{1/(α-1)-1}
         f'' += (2-α)/(α-1)² Σ [(α-1)S_blk-τ]_+^{1/(α-1)-2}
     if f<0: τ_lo=τ else: τ_hi=τ
     τ_H = τ - 2·f·f'/(2·(f')²-f·f'')
     τ = τ_H if τ_H∈[τ_lo, τ_hi] else (τ_lo+τ_hi)/2
3. Return τ
```

术语一般如何实现？如何使用？

Triton kernel 将 f/f'/f'' 累积分布到多个 K block，SRAM 中增量累加不写入 HBM。每 Q_i block 独立运行 Halley-Bisection。适用于任何需快速求解 α-entmax τ 的场景。

涉及论文标题：
- AdaSplash: Adaptive Sparse Flash Attention

## Adaptive Sparsity (Transformer 自适应稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Adaptive Sparsity 指注意力权重的稀疏模式由输入数据动态决定（data-dependent），区别于预定义固定模式的静态稀疏方法（sliding window、block-sparse）。α-entmax 是实现自适应稀疏性的典型方法——threshold τ 根据 logits 分布动态计算，低于 τ/(α-1) 的注意力分数被精确置零。

对比：Windowed/BigBird（固定窗口，低灵活性）；Top-k（固定预算 k，不可微）；α-entmax（数据驱动动态稀疏，精确可微）。在 ModernBERT 实验中 α=1.5 产生 ~95% 整体稀疏度，α=2.0 产生 ~99%。

从算法pipeline角度拆解术语。

α-entmax 的自适应性：对每个 attention row，τ 依赖于该行所有 logits 的分布。高方差 logits → 高 τ（高稀疏），低方差 → 低 τ（低稀疏）。训练策略：α 从 1.0 线性退火到目标值确保 smooth transition。GPU 加速：动态 block mask M 检测非零 P block，pointer-increment lookup tables 跳过 null blocks。

术语一般如何实现？如何使用？

训练：α 退火 + continuous pretraining。推理：利用动态稀疏性跳过不相关 KV block，节省 HBM 带宽和 GEMM 计算。适用于长文档分类、检索和需要选择性忽略无关上下文的场景。

涉及论文标题：
- AdaSplash: Adaptive Sparse Flash Attention

## KV Cache Channel Shrinking (KV Cache 通道收缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Channel Shrinking for KV Cache 是一种从通道维度（channel/head dimension）而非 token 维度压缩 KV Cache 的方法。核心思路：对 Key 投影矩阵 $W^K \in \mathbb{R}^{h_{in} \times h_{out}}$ 和 Value 投影矩阵 $W^V$ 做低秩分解 $W^K \approx A^K B^K$，其中 $A^K \in \mathbb{R}^{h_{in} \times h_{comp}}$，$B^K \in \mathbb{R}^{h_{comp} \times h_{out}}$，$h_{comp} \ll h_{out}$。推理时存储压缩特征 $X A^K \in \mathbb{R}^{n \times h_{comp}}$（而非完整 Key $X W^K \in \mathbb{R}^{n \times h_{out}}$），内存从 $O(n \times h_{out})$ 降至 $O(n \times h_{comp})$。

与 Token Pruning（从 token 维度丢弃 token）不同，Channel Shrinking 从通道/特征维度压缩，保留所有 token 的信息但以低维近似表示。CSKV 论文通过 SVD 分析发现 KV Cache 的奇异值呈显著长尾分布——仅保留最大的 50% 奇异值导致 MMLU 精度下降 <1%，证明通道维度存在大量冗余。

从算法pipeline角度拆解术语。

```
// Channel Shrinking 核心流程
// 低秩分解
A_K ∈ R^{hin × hcomp}  // 降维投影矩阵
B_K ∈ R^{hcomp × hout} // 升维重建矩阵
// 压缩率 = (hout - hcomp) / hout

// Prefilling: 输入序列 X ∈ R^{n×hin}
K_full = X @ W_K          // 完整 Key (n, hout)，用于 attention
K_compressed = X @ A_K    // 压缩 Key (n, hcomp)，存入 KV Cache

// Decoding: 需要完整 Key 时重建
K_reconstructed = K_compressed @ B_K  // (n, hcomp)@(hcomp, hout) → (n, hout)
```

术语一般如何实现？如何使用？

CSKV 在 HuggingFace Transformers 中实现：修改 attention 层的 key/value 投影逻辑，增加 A_K/B_K/A_V/B_V 低秩权重。训练通过 ASVD 初始化 + 逐层 MSE 重建损失微调（单 A100 90 分钟）。适用于长上下文 32K+ tokens，可与量化叠加达到 95% 总压缩率。

涉及论文标题：
- CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios

## Low-Rank Decomposition for KV Cache Compression (KV Cache 低秩分解压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

低秩分解压缩 KV Cache 是通过对 Key/Value 权重矩阵 $W^K, W^V$ 做矩阵分解来减少 KV Cache 存储开销的技术。将 $W^K \in \mathbb{R}^{h_{in} \times h_{out}}$ 近似为两个更小矩阵的乘积 $W^K \approx A^K B^K$，其中 $A^K \in \mathbb{R}^{h_{in} \times h_{comp}}$，$B^K \in \mathbb{R}^{h_{comp} \times h_{out}}$。推理时存储 $X A^K$（维度 $h_{comp}$）而非 $X W^K$（维度 $h_{out}$），实现 $h_{out} / h_{comp}$ 倍压缩。

CSKV 发现 KV Cache 的奇异值呈长尾分布——大量小奇异值可移除而不显著影响性能。直接使用标准 SVD 分解 $W = U \Sigma V^T$ 保留前 k 个奇异值存在局限：它不考虑激活值分布。ASVD（Activation-aware SVD）通过缩放矩阵 S 使分解对高激活值维度更敏感。

从算法pipeline角度拆解术语。

```
// 标准 SVD 低秩分解
W ∈ R^{hin × hout}
U, Σ, V_T = SVD(W)
A = U[:, :hcomp] @ sqrt(Σ[:hcomp, :hcomp])  // (hin, hcomp)
B = sqrt(Σ[:hcomp, :hcomp]) @ V_T[:hcomp, :] // (hcomp, hout)

// ASVD 变体
S = diag(mean(|X|, dim=0)^α)  // 缩放矩阵, α=0.5
W_s = W @ S
U_s, Σ_s, _ = SVD(W_s)
A = inv(S) @ U_s[:, :hcomp] @ sqrt(Σ_s[:hcomp, :hcomp])
B = sqrt(Σ_s[:hcomp, :hcomp]) @ V_s_T[:hcomp, :] @ inv(S)
```

术语一般如何实现？如何使用？

PyTorch 中通过 `torch.linalg.svd()` 实现。ASVD 初始化需从标定数据集采样 256 样本，收集每层 Key/Value 激活，计算 Absolute Mean Value 作为 S（α=0.5）。初始化后的 A/B 作为可训练参数，通过逐层 MSE 损失微调。适用于 LLaMA、Mistral 等标准 Transformer 架构。

ReCalKV 将低秩 KV Cache 压缩进一步细化为不对称的 Key/Value 策略：(1) 对于 Keys，使用 HSR（Head-wise Similarity-aware Reordering）先通过 CKA 相似度将结构相似的 head 分组，再对每组做 grouped SVD；(2) 对于 Values，使用 OVC（Offline Value Calibration）用标定数据对 SVD 分解后的 L_v 和 R_v 做闭式校准，最小化 ||L_v R_v X - W_v X||_F^2，然后通过 Matrix Fusion 将 R_v 融合进 output projection W_o 消除推理时重建开销。ReCalKV 还引入 Fisher Information 引导的逐层压缩率分配，使重要层保留更多 rank。经 256 个 WikiText-2 标定样本在单张 A800 GPU 上完成离线压缩后，推理时 50% 压缩率下零样本 QA 仅降 ~2%。

涉及论文标题：
- CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration
- TransMLA: Multi-Head Latent Attention Is All You Need
- xKV: Cross-Layer SVD for KV-Cache Compression

**xKV 直接对 KV-Cache 做跨层低秩分解**：与上述方法对权重矩阵离线分解不同，xKV 直接对 prefill 阶段产生的 **KV-Cache（而非权重 W_K/W_V）** 做在线 SVD，且引入了**跨层维度**：将多层的 KV-Cache 水平拼接后做统一 SVD，提取跨层共享基。xKV 发现 CKA 分析表明跨层主导左奇异向量高度对齐（但 token-wise cosine similarity 很低），因此跨层 SVD 比单层 SVD 更高效——相同 rank 保留更多跨层共享信息，相同压缩比下精度更高（8× 压缩下 xKV avg=87.8% vs Single SVD avg=35.3%）。此外 xKV 还兼容 MLA 架构（对 non-RoPE latent representations 做跨层 SVD）。

TransMLA 提出 BKV-PCA——对 K_nope 和 V 做联合低秩压缩前，先计算平衡因子 α = E[||K_nope||₂]/E[||V||₂] 缩放 K 使两者 norm 对齐，避免 key 主导 PCA 主成分方向导致 value 信息丢失。与 weight-based SVD 相比，activation-based PCA（在标定数据激活值上做 PCA 而非在权重矩阵上做 SVD）在 TransMLA 实验中显著降低压缩损失（Figure 4b）。BKV-PCA 联合压缩 [K_nope; V]（(2g-1)d 维）到 r_kv 维 latent 空间，推理时仅缓存 r_kv 维 latent vector，而非 (2g-1)d 维的完整 NoPE key + value。

## Bi-Branch KV Cache (双分支 KV 缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Bi-Branch KV Cache 是 CSKV 提出的兼顾压缩效率和局部精度的 KV Cache 管理策略。将 KV Cache 分成两个分支：(1) 压缩分支（Compressed Cache）：存储全部历史 token 的低维压缩特征 $X A^K$，维度 $h_{comp}$；(2) 完整精度分支（Full-Precision Cache）：仅保留最近 $m$ 个 token 的完整精度 Key/Value，维度 $h_{out}$（默认 m=32）。Attention 计算时，历史 token 从压缩分支通过 $B^K$ 重建，近期 token 使用完整精度值，利用"近期 token 对下一 token 预测影响最大"的观察。

从算法pipeline角度拆解术语。

```
// Prefilling
K_full = X @ W_K              // (n, hout)
K_compressed = X @ A_K        // (n, hcomp)，→ Compressed Cache
K_local = K_full[-m:, :]      // (m, hout)，→ Full Cache

// Decoding
k = x @ W_K; k_comp = x @ A_K
Compressed_Cache ← [K_compressed; k_comp]  // (n+1, hcomp)
Full_Cache ← [K_local; k]                  // (m+1, hout)

// 重建
K_hat = Compressed_Cache[:(n-m), :] @ B_K  // 旧 token 重建
K_for_attn = concat([K_hat, Full_Cache])
// 维护: Full_Cache 移除最旧 token 保持 m
```

术语一般如何实现？如何使用？

窗口大小 m=32 为默认值（消融：m>32 后收益递减，m=32 Avg.Acc=0.92 vs m=4096 Avg.Acc=0.96）。更大 m 使更多 token 以完整精度存储（降低压缩效果），需权衡 m 与内存预算。可与量化叠加使用。

涉及论文标题：
- CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios

## ASVD (Activation-aware Singular Value Decomposition, 激活感知奇异值分解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

ASVD 是标准 SVD 低秩分解的激活感知扩展。标准 SVD 仅基于权重矩阵 $W$ 的奇异值做低秩近似，忽略激活值分布——某些输出通道激活值远大于其他通道，在 MSE 中贡献更大。ASVD 引入缩放矩阵 $S = \operatorname{diag}(|\bar{X}|^\alpha)$（$\bar{X}$ 为标定数据平均激活值，$\alpha=0.5$），对缩放后权重 $W_s = W S$ 做 SVD，使低秩分解重点保留高激活值通道的信息。CSKV 中使用 Absolute Mean Value 方法计算 S，从 256 个标定样本收集激活值。

从算法pipeline角度拆解术语。

```
// 1. 收集标定激活 (256 samples)
calib_X = collect_activations(model, calib_dataset)
// 2. 计算 S
S_diag = mean(|calib_X|, dim=0)^α  // α=0.5
S = diag(S_diag)
// 3. 对缩放后权重 SVD
U_s, Σ_s, V_s_T = SVD(W @ S)
// 4. 构造低秩分解
A_K = inv(S) @ U_s[:, :hcomp] @ sqrt(Σ_s[:hcomp, :hcomp])
B_K = sqrt(Σ_s[:hcomp, :hcomp]) @ V_s_T[:hcomp, :] @ inv(S)
```

术语一般如何实现？如何使用？

CSKV 消融实验证明 SVD-based 初始化的必要性：随机初始化训练完全无法收敛（Avg.Acc=0.00），ASVD 初始化后 Loss 从 ~5.5 迅速收敛到 ~4.0。ASVD 初始化在各压缩率下均优于标准 SVD（80% 压缩：ASVD 0.92 vs SVD 0.87 vs Random 0.00）。仅对 W_K, W_V 做 ASVD（非所有权重）。

涉及论文标题：
- CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios

## Layer-wise Reconstruction Loss for KV Cache (逐层重建损失)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

逐层重建损失是 CSKV 用于训练低秩 KV Cache 压缩矩阵的损失函数。传统端到端 fine-tuning 需在整个模型上做前向/反向传播；逐层重建损失在每层内部独立最小化 MSE：$L_K = \operatorname{MSE}(K, \hat{K})$，其中 $K = X W^K$，$\hat{K} = X A^K B^K$。总损失 $\mathcal{L}_{all} = \sum_{j=1}^{n_l} (L_{K,j} + L_{V,j})$。各层独立训练，可并行。仅需 90 分钟/单 A100-80G（AdamW, lr=5e-5, epoch=1, batch_size=1）。

从算法pipeline角度拆解术语。

```
for layer in model.layers:
    W_K, W_V = layer.self_attn.k_proj.weight, layer.self_attn.v_proj.weight
    A_K, B_K = ASVD_init(W_K, calib)  // SVD-based init
    A_V, B_V = ASVD_init(W_V, calib)
    for X in train_loader:
        K = X @ W_K.T; K_hat = X @ A_K.T @ B_K.T
        V = X @ W_V.T; V_hat = X @ A_V.T @ B_V.T
        loss = MSELoss(K, K_hat) + MSELoss(V, V_hat)
        loss.backward()  // 仅更新 A_K, B_K, A_V, B_V
        optimizer.step()
```

术语一般如何实现？如何使用？

适用于需压缩 KV Cache 但无法承受完整 retraining 的场景。逐层训练避免梯度跨层传播开销。数据量小（scaled-down Pile, epoch=1），泛化性好。适用于 LLaMA、Mistral 等标准 Transformer 架构。

涉及论文标题：
- CSKV: Training-Efficient Channel Shrinking for KV Cache in Long-Context Scenarios

---

## KV Footprint (KV足迹) / Critical KV Footprint

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Footprint 是 PruLong 论文提出的衡量 KV cache 内存使用效率的统一指标，定义为所有推理时间步上未被逐出（un-evicted）的 KV entries 数量的时间积分（time-aggregated sum），归一化至完整 causal attention 对应值。与 KV cache size（瞬时指标）不同，KV Footprint 捕获 KV cache 在 pre-filling 和 decoding 全过程的累积内存使用，包括每个 KV entry 的生命周期。理想 KV eviction 方法应尽早逐出 KV 以最小化 KV Footprint。Critical KV Footprint 定义为保持 ≥90% full attention 性能的最小 KV Footprint。

从算法pipeline角度拆解术语。

**KV Footprint 计算**：
```
KV_footprint = 0
for t in prefill_steps + decode_steps:
    un_evicted[t] = count(active_KV[t]) + count(inactive_KV[t])  // 未被 evict 的 KV
    KV_footprint += un_evicted[t]
KV_footprint /= total_full_attention_entries  // 归一化百分比

// Critical KV Footprint
critical = min footprint subject to: score(footprint) >= 0.9 × score(full_attn)
```

**Footprint 示例（N=6 tokens, prefill 2 chunks, decode 4）**：
- Full causal attention：36 KV-query pairs → footprint 100%
- Step 3 后 evict 部分 KV：26/36 → 72.2%
- Step 1 后即刻 evict：更低 footprint

术语一般如何实现？如何使用？

KV Footprint 是分析性指标。论文附录验证了 KV Footprint 与真实硬件指标的相关性：PruLong 较低 KV footprint 对应较低 peak GPU memory（26.3 GiB vs PyramidKV 33.7 GiB）和较高 throughput。KV Footprint 作为理想化指标可忽略 CUDA kernel 差异、PyTorch GC 延迟等实现细节，实现跨方法公平比较。代码开源：https://github.com/princeton-pli/PruLong

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---

## Chunked Eviction (分块逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Chunked Eviction 是将 post-fill eviction 方法适配到 chunked pre-filling 场景的技术：在每个 pre-fill chunk 处理后立即执行 KV eviction，而非等所有 chunk 处理完毕。两种变体：Naive（chunk 自身末尾 k token 计算重要性）和 Patched（拼接 prompt tail 提供全局重要性信号）。Chunked Eviction 显著降低 KV Footprint——因 KV 在 pre-fill 过程中即被提前移除。

从算法pipeline角度拆解术语。

**Naive Chunked Eviction**：
```
for each chunk c:
    K_c, V_c = forward(chunk_tokens[c])
    scores = attention_score(last_k_tokens_of_chunk, K_c)  // 局部信号
    keep_idx = top_k(smoothed_scores, p × len(K_c))
    K_cache.append(K_c[keep_idx])  // 非 keep 的 KV 被 evict
```

**Patched Chunked Eviction**：
```
for each chunk c:
    X_patched = concat([chunk_tokens[c], prompt[-k:]])
    K_patched, V_patched = forward(X_patched)
    scores = attention_score(prompt_tail, K_patched)  // 全局信号
    keep_idx = top_k(smoothed_scores, p × len(K_c))
    K_cache.append(K_c[keep_idx])
    // 丢弃 patched token 的 KV（最后 chunk 除外）
```

术语一般如何实现？如何使用？

GQA 场景下需在 KV group 内 mean-pool attention 后统一选择，避免为每个 query head 独立选择 → 8× 内存节省（Llama-3.1-8B GQA）。Patched PyramidKV + mean-pool 在 RAG（<34% KV footprint）和 LongQA（<35%）上取得所有方法中最优。代码开源：https://github.com/princeton-pli/PruLong

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---

## Post-fill Eviction (后填充逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Post-fill Eviction 是 KV cache 压缩方法的一类：在 pre-fill 阶段完全结束后才基于 attention scores 启发式一次性选择并逐出 KV。代表工作：H2O（累积 attention score）、PyramidKV（pyramidal budget）、SnapKV（最后 token attention）、FastGen。根本缺陷：pre-fill 阶段保留所有 KV → 高峰值内存 → KV Footprint 几乎无 reduction。在 chunked pre-filling 成为标准实践（SGLang 默认 8192 chunks）后问题更突出。PruLong 论文通过 Chunked Eviction 解决了这一缺陷。

从算法pipeline角度拆解术语。

```
// 标准 Post-fill Eviction 流程
for token in prompt:  // Pre-fill: 全部保留
    K_cache.append(W_K(token)); V_cache.append(W_V(token))
// Pre-fill 后一次性 evict
scores = moving_average(Σ last_k_queries softmax(Q @ K^T / √d))
keep_idx = top_k(scores, budget)
K_cache = K_cache[keep_idx]; V_cache = V_cache[keep_idx]
// Decoding: 使用压缩后的 cache
while not EOS: ...
```

术语一般如何实现？如何使用？

Hook 方式集成到 HuggingFace Transformers attention 层。PyramidKV: https://github.com/FYYFU/PyramidKV；SnapKV: https://github.com/FasterDecoding/SnapKV。论文将其适配为 Chunked Eviction 后大幅降低 KV Footprint。

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---

## Recency Eviction (近因逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Recency Eviction 是基于时间近因的 KV cache 压缩策略：仅保留最近 W 个 token（local window）和前 S 个 attention sink token，其余 KV 全部 evict。KV cache 大小固定为 W+S（不随序列长度增长）。代表工作：StreamingLLM（发现 attention sink + streaming heads）、DuoAttention（仅部分 heads 设为 streaming）、MoA（natural text 训练 head specialization）、PruLong（改进的 head specialization）。优势：decoding 阶段 memory 恒定、与 chunked pre-filling 天然兼容、显著降低 KV Footprint。代价：可能遗忘远处重要信息。

从算法pipeline角度拆解术语。

```
// 纯 Recency Eviction（StreamingLLM 风格）
K_cache, V_cache = [], []  // 固定最大 W+S
for token t:
    k, v = W_K(t), W_V(t)
    K_cache.append(k); V_cache.append(v)
    if len > W+S:
        K_cache = concat([K_cache[:S], K_cache[-W:]])  // 保留 sinks + local
        V_cache = concat([V_cache[:S], V_cache[-W:]])

// 混合模式（PruLong/DuoAttention 风格，仅 streaming heads）
if z_lh == 0:  // streaming head
    K_attn = concat([K_cache[:S], K_cache[-W:]])
else:  // retrieval head: full cache
    K_attn = K_cache
```

术语一般如何实现？如何使用？

Attention mask 修改或 KV cache 截断实现。PruLong 默认 W=1024, S=128（对 128K 上下文有效）。StreamingLLM：https://github.com/mit-han-lab/streaming-llm。PruLong：https://github.com/princeton-pli/PruLong。

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---

## Streaming Attention Heads / Retrieval Heads (流式注意力头 / 检索头)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Transformer attention heads 的二分分类：Streaming Heads 仅关注最近 W token + 前 S attention sinks，KV cache 固定 W+S 大小，负责局部语法结构；Retrieval Heads 保留完整 KV cache，从远处检索关键信息，对 recall/RAG 任务至关重要。PruLong/DuoAttention 通过训练学习各 head 的二值分类 mask z_{i,j} ∈ {0,1}。

从算法pipeline角度拆解术语。

```
// 混合 Attention
if z_lh == 1:  // retrieval: full causal attention
    attn = FlashAttention(Q, K_full, V_full)
else:  // streaming: local + sinks
    K_attn = concat([K[:S], K[-W:]])
    attn = FlashAttention(Q, K_attn, V_attn)

// PruLong 训练后离散化：top k% log_α → z=1, 其余 z=0
```

术语一般如何实现？如何使用？

DuoAttention（Xiao et al., 2025）首次提出系统性的 retrieval/streaming head 二分分类方法和基于优化的识别方法。核心发现：retrieval heads 仅占总 head 的少数（MHA: ~25%, GQA: ~50%），但对其做 KV cache 压缩会显著损害长上下文能力；streaming heads 占多数，压缩其 KV cache 几乎无性能影响。识别方法：基于优化的 gate value training（合成 passkey retrieval 数据 + L2 distillation loss + L1 regularization），直接测量输出偏差而非依赖 attention score profiling。消融实验证明该方法优于 attention profiling-based 方法（FastGen, RazorAttention）和 language modeling-based 方法。实现代码：https://github.com/mit-han-lab/duo-attention。

StreamingLLM（Xiao et al., ICLR 2024）首次发现 streaming heads（attention sink）现象。PruLong 在 Llama-3.1-8B-Instruct 上 70% streaming heads 可在 recall 上保持 ≥90% 性能，critical KV footprint 约 30%。不同 task 对 retrieval/streaming 最优比例不同。代码：https://github.com/princeton-pli/PruLong

CompressKV（Lin et al., 2025）进一步细化了 Retrieval Head 的子类型：传统 Retrieval Head 识别标准要求 head 的 top-1 attention 精确落在正确答案 token 上（仅捕捉 copy-paste 行为），而 **Semantic Retrieval Head** 聚合 head 在整个 answer span 上的 attention scores 来评估语义检索能力。公式：SemanticRetrievalScore(h) = Σ_{t} I[y_t ∈ A] Σ_{j∈A} a_{t,j}^h。这种方法能捕捉到对答案周边语义相关 token（如 "eat", "a thing" 围绕 "sandwich"）有高 attention 的 head——这些 head 即使 top-1 attention 不落在正确答案 token 上，仍具有语义检索能力。在 GQA-based LLM 中，使用 Semantic Retrieval Head（而非全部 head 或传统 Retrieval Head）进行 KV cache eviction 的 token 选择，可避免 Streaming Head 主导 eviction 导致仅保留首尾 token 的问题。

涉及论文标题：
- DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs
- CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation
- Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

**Elastic Attention 的新贡献**：将 head 分类从静态（训练后固定）变为动态（test-time adaptive）。通过 Attention Router（每层轻量 MLP，0.27M 参数）在推理时根据输入 hidden states 实时决定每个 head 使用 FA 还是 SA 模式，而非像 DuoAttention/PruLong 那样训练后 head 分配固定不变。Router 使用 Gumbel-Sigmoid + STE 训练，backbone 冻结，仅训练 Router 参数（12h on 8×A800）。

---

## Hard Concrete Distribution / L0 Regularization for Pruning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hard Concrete Distribution（Louizos et al., ICLR 2018）是将离散二值 mask 优化连续化的方法。通过 Gumbel-Softmax 重参数化 + hard sigmoid gate，使 Bernoulli 采样可微且支持 {0,1} 端点的非零概率质量。PruLong 用它学习 attention head 的 retrieval/streaming 分类 mask，消除 DuoAttention continuous gating 的 train-test rounding gap。

从算法pipeline角度拆解术语。

```
// Hard Concrete 重参数化（PruLong）
u ~ Uniform(1e-6, 1-1e-6)
s = σ( (2/3) × log(u/(1-u)) + log_α )
g̃ = -0.1 + 1.1 × s
z̃ = clamp(g̃, 0, 1)  // hard gate: support {0,1}

// 期望 L0 稀疏度（闭式）
P(z > 0) = σ(log_α + log(10))
s(π) = 1 - (1/(L×H)) × Σ P(z_{i,j} > 0)

// Lagrangian: L_reg = λ1(s - t) + λ2(s - t)²，λ1,λ2 gradient ascent

// 最终：max_{λ1,λ2} min_{log_α} E[NTP_loss] + L_reg
```

术语一般如何实现？如何使用？

PruLong 训练：1000 steps, 1M token batch, seq_len 131K, LR=1.0 for log_α/λ, sparsity warmup 0→t over 800 steps, 不更新 model weights。训练后取 top k% log_α = +∞ (z=1), 其余 -∞ (z=0)。原始：https://github.com/AMLab-Amsterdam/L0_regularization。PruLong：https://github.com/princeton-pli/PruLong。

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---

## PruLong

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PruLong 是 Princeton PLI 提出的端到端 KV cache eviction 方法。核心：将 attention heads 二分类为 retrieval（full KV）和 streaming（local W=1024 + sinks S=128）。vs DuoAttention 的三项创新：(1) next-token prediction loss 替代 L2 reconstruction；(2) Hard Concrete Bernoulli masks 消除 train-test gap；(3) natural long-context data 替代 synthetic passkey。冻结模型权重，仅训练 mask parameters，1000 steps 收敛。Recall critical KV footprint 46% vs DuoAttention 58%（-12 points）。

从算法pipeline角度拆解术语。

**PruLong 训练伪代码**：
```
# 参数：log_α_{i,j} (每层每头), λ1, λ2 (Lagrange), τ=2/3, l=-0.1, r=1.0
for step in 1..1000:
    t = min(target, target × step/800)  # sparsity warmup
    for each head (i,j):
        z̃ = HardConcrete(log_α, τ, l, r)  # Bernoulli 采样
        attn = z̃ × Attn_full + (1-z̃) × Attn_streaming
    L_ntp = cross_entropy(logits, labels)
    s = 1 - mean(σ(log_α + log(10)))  # expected sparsity
    L = L_ntp + λ1(s-t) + λ2(s-t)²
    log_α -= ∇L; λ1 += ∇L; λ2 += ∇L

# 离散化：top k% log_α → z=1 (retrieval), 其余 z=0 (streaming)
```

**PruLong vs DuoAttention@70% sparsity**：
| 维度 | DuoAttention | PruLong |
|------|-------------|---------|
| 训练目标 | L2 reconstruction | NTP loss |
| Mask 类型 | Continuous z∈[0,1] | Hard Concrete Bernoulli |
| 训练数据 | Synthetic passkey | Natural long data |
| Recall score | 38.6 | 91.4 |

术语一般如何实现？如何使用？

PyTorch + HuggingFace Transformers，在 Llama-3.1-8B-Instruct 和 ProLong-8B 上评估。训练配置：batch 1,048,576 tokens, seq_len 131,072, model weights frozen。代码：https://github.com/princeton-pli/PruLong

涉及论文标题：
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---

## DuoAttention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

DuoAttention（Xiao et al., 2025, MIT & NVIDIA）是一个将 LLM 的 attention head 分类为 Retrieval Heads 和 Streaming Heads 的 KV cache 压缩框架。核心观察：仅少数 attention head（Retrieval Heads）对长上下文处理至关重要，需要 full attention across all tokens；多数 head（Streaming Heads）主要关注 attention sink（首 token）和最近 token，可以使用 constant-length KV cache（仅保留 sink + recent tokens）。

DuoAttention 包含三个阶段：
1. **Retrieval Head Identification（训练）**：为每个 KV head 分配可训练 gate value α_{i,j} ∈ [0,1]，前向 pass 中混合 full attention 和 streaming attention 输出：attn = α·full_attn + (1-α)·streaming_attn。在合成 passkey retrieval 数据集（BookSum + 10×32-word passkeys）上以 L2 distillation loss（最后 hidden state 偏差）+ L1 regularization（λ=0.05）优化 gate values。仅数千可训参数，所有模型权重冻结，2,000 steps on 8×A100 完成。
2. **Head Binarization & Reordering**：按 sparsity quantile threshold τ 将 α 二值化为 {retrieval, streaming}，高于 τ 为 retrieval head。预处理时重排 Q/K/V 投影的输出通道，将两类 head 分为连续簇，推理时用 slicing/concat 替代 scatter/gather。
3. **Deployment（双 KV Cache）**：每层两个 KV cache——retrieval heads 使用 full KV cache（all tokens），streaming heads 使用 constant KV cache（仅 sink 64 + recent 256 tokens）。Chunked pre-filling 中 streaming heads 每 chunk 后立即 prune KV 仅保留 sink+recent，下一 chunk 仅 attend 到 constant 数量的历史 token，pre-filling 复杂度从 O(L²) 降至 O(LK)。

与完全依赖 attention score profiling 的方法（FastGen, RazorAttention）不同，DuoAttention 直接测量 output deviation（压缩 KV cache 后的输出偏差），可以识别 attention scores 中并不明显、但对 long-context 至关重要的 retrieval head。

从算法pipeline角度拆解术语。

**Phase 1: Gate Value Training**
```
# 初始化：α_{i,j} = 1.0 (所有 head 初始假设为 retrieval)
# 合成数据：BookSum + 10×32-word passkeys, 50 长度区间 (1K→max_len)

for step in 1..2000:
    # 前向 (per KV head j in layer i)
    full_out = softmax(Q @ K^T ⊙ M_causal) @ V
    stream_out = softmax(Q @ K^T ⊙ M_streaming) @ V  # Λ-like mask
    attn_{i,j} = α_{i,j} · full_out + (1-α_{i,j}) · stream_out

    # Loss (仅最后 l 个 passkey tokens)
    L_distill = (1/N) Σ (H_full_last - H_mixed_last)²  # L2 on hidden states
    L_reg = Σ_i Σ_j |α_{i,j}|                            # L1 sparsity
    L = L_distill + 0.05 · L_reg

    # AdamW: lr=0.02 warmup(400 steps 0.002→0.02)→decay(400 steps 0.02→0.002)
    # 仅更新 α_{i,j}，模型权重冻结
```

**Phase 2: Binarization & Head Reordering**
```
# 按 sparsity quantile 确定阈值 τ
for each head (i,j):
    type = "retrieval" if α_{i,j} > τ else "streaming"

# Head reordering: 重排 W_Q, W_K, W_V 的输出通道
# retrieval heads → 连续簇 0..R-1, streaming heads → 连续簇 R..H-1
```

**Phase 3: Dual KV Cache Decoding**
```
# Per layer forward:
Q_ret, Q_str = split(Q, head_dim)        # 沿 head 维度切分
K_ret, V_ret = full_kv_cache              # 全量历史 (retrieval heads)
K_str, V_str = sink_and_recent_kv_cache   # 仅 sink + recent (streaming heads)

out_ret = FlashAttention(Q_ret, K_ret, V_ret)
out_str = FlashAttention(Q_str, K_str, V_str)
output = concat([out_ret, out_str], head_dim) @ W_O
```

**Phase 4: Chunked Pre-filling (streaming heads)**
```
for each chunk of K tokens:
    K_chunk, V_chunk = compute_KV(chunk)
    # streaming heads: 立即 prune，仅保留 sink + recent
    K_str = prune_to_sink_and_recent(K_str, K_chunk)
    V_str = prune_to_sink_and_recent(V_str, V_chunk)
# Streaming heads 复杂度: time O(LK) [vs O(L²)], memory O(K) [vs O(L)]
```

**配置与性能**：
| 模型 | Attention | Retrieval Ratio | Memory Reduction | Latency Reduction (Decode/Pre-fill) |
|------|-----------|----------------|------------------|-------------------------------------|
| Llama-2-7B | MHA (32 heads) | 25% | up to 2.55× | 2.18× / 1.73× |
| Llama-3-8B | GQA (8 KV heads) | 50% | up to 1.67× | 1.50× / 1.63× |

GQA 模型的 retrieval ratio 更高（50% vs 25%），因为 GQA 中 per-group gate value 绑定多个 query head，必须保守压缩。MHA 中每个 head 独立 gate，可更激进压缩。

**结合量化**：DuoAttention + QServe (W8A8KV4) → Llama-3-8B 单 A100 容纳 3.3M tokens（6.4× capacity vs full attention BF16, 仅需 0.52 GB per token）。

术语一般如何实现？如何使用？

开源：https://github.com/mit-han-lab/duo-attention。基于 PyTorch + FlashInfer (RoPE/RMSNorm kernels) + FlashAttention-2。训练用 FSDP2 + DeepSpeed Ulysses sequence parallelism 支持长序列。Deployment 默认 sink=64, recent=256, pre-fill chunk=32K。NIAH 上 25% retrieval ratio (MHA) / 50% (GQA) 即可保持 full attention 级别准确率，LongBench 上同样保持接近 full attention 的性能。所有 baseline（H2O, TOVA, StreamingLLM, FastGen）在 NIAH 上均失败（无法在不同深度正确检索）。

与 PruLong 对比：DuoAttention 使用 L2 reconstruction loss + synthetic passkey + continuous gating（有 train-test rounding gap）；PruLong 使用 NTP loss + Hard Concrete Bernoulli + natural long-context data，在 natural data recall 上更优（91.4 vs 38.6 at 70% sparsity）。

涉及论文标题：
- DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

---

## PyramidKV

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PyramidKV（Cai et al., 2024）是首个提出跨层不均匀 KV cache budget 分配的 KV cache 压缩方法。核心创新源于 Pyramidal Information Funneling 的发现——LLM 底层注意力分散（broad-spectrum），中层逐步收窄（localized），顶层集中在少量关键 token（massive attention）。基于此，PyramidKV 提出：(1) 动态跨层 budget 分配——底层多分配 cache、顶层少分配 cache，按算术序列递减；(2) 基于 instruction tokens（最后 α=8 个 token）的 attention score 进行 token 选择（与 SnapKV 共享基本重要性评估框架）。

原始论文 budget 公式：k^{m-1} = k^{total}/(β·m)（顶层），k^0 = 2·k^{total}/m - k^{m-1}（底层），中间层 k^l = k^0 - (k^0 - k^{m-1})/(m-1) × l（arithmetic sequence）。超参 β=20 控制金字塔陡峭程度，α=8 为各层固定保留的 instruction token 数。该公式确保 Σ_l k^l = k^{total}，各层 budget 严格单调递减。

与后续工作在 PruLong 中的变体——使用 γ 参数化的 budget[l] = base_budget × (1 - l/L)^γ——不同，原始 PyramidKV 的算术序列由 total budget 和 β 解析确定，无需 base_budget 比例。

从算法pipeline角度拆解术语。

**原始 PyramidKV（Cai et al., 2024）Budget Allocation + Selection 流程**：
```
// === Step 1: Budget Allocation (预计算，一次性) ===
// m = 总层数, k_total = 总 KV cache budget
// β = 20 (超参，控制金字塔形状), α = 8 (instruction tokens)

k_top = k_total / (β * m)             // 顶层 budget (最少)
k_bottom = 2 * k_total / m - k_top    // 底层 budget (最多)
Δ = (k_bottom - k_top) / (m - 1)      // 层间递减步长

for l in 0..m-1:
    k_l = k_bottom - Δ * l            // arithmetic sequence
    // e.g., L=32, k_total=2048: k_0≈100, k_31≈10 (不含instruction tokens)

// === Step 2: Attention Score Calculation ===
// Prefill 阶段，对每层每 head:
A = softmax(Q @ K.T / sqrt(d_k))       // [seq_len, seq_len]
for h in 0..H-1:
    s_h = sum(A_h[-α:, :], dim=0)      // 最后α个token对各key的attention sum
    // s_h[i] = Σ_{j ∈ [n-α, n]} A_ij^h

// === Step 3: KV Selection ===
for l in 0..m-1:
    retain instruction tokens (最后α个)
    remaining_budget = k_l - α
    for h in 0..H-1:
        top_indices = topk(s_h, remaining_budget)
        K_selected[l,h] = K[l,h, cat([instruction_indices, top_indices])]
        V_selected[l,h] = V[l,h, cat([instruction_indices, top_indices])]

// === vLLM集成 (Appendix R) ===
// Per-layer block table: 每个sequence的block table扩展为每层独立
// 解决uniform eviction在小budget下的fragmentation
```

**Ablation: 算术 vs 几何 vs 指数衰退**（Table 4, LLaMa-3-8B, KV size=64）：
- Linear (PyramidKV): LongBench avg 34.76
- Geometric decay: 34.36
- Exponential decay: 34.23
- Entropy-based adaptive: 32.71
- Gini-based adaptive: 32.58
结论：线性算术序列最匹配观察到的注意力渐进收窄模式，且计算开销最小。

术语一般如何实现？如何使用？

开源：https://github.com/Zefan-Cai/PyramidKV（官方实现），支持 Flash Attention v2 和 SDPA attention，包含 PyramidKV、SnapKV、H2O、StreamingLLM 四种方法的统一实现。支持 LLaMA-3-8B/70B-Instruct、Mistral-7B 等模型。

使用方法（from official repo README）：`python -m longbench.pred --model llama3-8b --method pyramidkv --env_conf config/llama3-8b/pyramidkv.json`。配置文件指定 KV cache size、α (window size)、β (pyramid steepness)。

PruLong 论文后续实现了其 Chunked Eviction 变体——Patched PyramidKV + mean-pool 在 RAG（<34% KV footprint）上取得最优结果。该变体使用 (1 - l/L)^γ 参数化（而非原始算术序列），always-retained window=64。

WindowKV（Zuo et al., 2025）将 PyramidKV 的逐层 budget 分配扩展到**组级别（group-level）**：将 m 层分为 H=m/γ 组，使用相同算术序列公式跨组分配 budget，组内各层均匀共享。同时将 token 级选择替换为 window 级选择，并引入任务自适应分类器决定每窗口中保留的 token 比例 p（localization: p=ω；aggregation: p<ω）。

涉及论文标题：
- PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---

## Sliced Shapley Value (SSV / 切片 Shapley 值)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Sliced Shapley Value (SSV) 是 CoKV 提出的一种高效近似 Shapley Value 的方法，用于在 LLM 中评估每个 attention head 的合作贡献。传统 Shapley Value 需要枚举所有 coalition size j ∈ {1,...,n} 的期望 marginal contribution，计算复杂度为 O(n·T/ε²)（n 为 head 总数，T 为单次验证集推理时间）。SSV 的洞察是：在 LLM 中，不同 coalition size j 下 head 的 expected complementary contribution 分布高度相关（附录 B.3 实验验证），因此只需计算少数代表 coalition size H ⊆ {1,...,n} 的 complementary contribution 即可准确捕获 head 的相对重要性。

公式定义：$\mathcal{SSV}_i^{\mathcal{H}} = \frac{1}{|\mathcal{H}|} \sum_{j \in \mathcal{H}} \mathcal{SV}_{i,j}$，其中 $\mathcal{SV}_{i,j}$ 是 (i,j)-coalitions 的 expected complementary contribution。CoKV 使用 H={32,64,96,128}（n=256），计算复杂度降为 O(|H|·T/ε²)。

SSV 的关键优势：(1) 利用 complementary contribution U(S)-U(N\S) 可同时更新 coalition S 中所有 head 的估计值，一次采样助攻多个 head；(2) 只需 |H|≪n 个 coalition size 即可稳定估计，理论保证为 (ε,δ)-approximation；(3) 分布对称性（coalition size s 和 n-s 的分布几乎相同）使只需计算 s<n/2 的 coalition size。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**SSV 计算的伪代码（Algorithm 1）**：

```
Input: Heads N = {h_1,...,h_n}, 采样次数 M, coalition sizes H={j_1,...,j_k}
Output: SSV_i^H for each head h_i

// 初始化累计矩阵
SV_{i,j} = 0, m_{i,j} = 0  for i,j in 1..n

for k = 1 to M:
    π^k = random_permutation(1..n)        // 随机排列 heads
    j = random_choice(H)                   // 随机选 coalition size
    S = {π^k(1), ..., π^k(j)}             // 前 j 个 heads 构成 coalition
    
    // 计算 complementary contribution
    U_S = model_accuracy(mask N\S)         // mask S 以外的 heads
    U_NS = model_accuracy(mask S)          // mask S 中的 heads
    u = U_S - U_NS
    
    // 更新 coalition S 中所有 head 的估计
    for t = 1 to j:
        SV_{π^k(t), j} += u
        m_{π^k(t), j} += 1

// 平均得到 SSV
for i = 1 to n:
    SSV_i^H = (1/|H|) * sum_{j in H} (SV_{i,j} / m_{i,j})

return {SSV_1^H, ..., SSV_n^H}
```

**SSV 在推理中的使用（Budget Allocation）**：
```
// Hyperparameters
α = #heads with zero extra budget (hyperparam, {1,5,10,15,20,30,40})
B = shared budget, s = local window size

// Step 1: Min-max normalize SSV
min_α = α-th smallest SSV
max_SSV = max(SSV)
NSV_i = 0  for α smallest SSV heads
NSV_i = (SSV_i - min_α) / (max_SSV - min_α)  for remaining n-α heads

// Step 2: Proportional allocation
c_i = B * (NSV_i / sum(NSV_j)) + s    // cache size for head h_i
```

术语一般如何实现？如何使用？

SSV 计算是 offline 预计算过程：在验证集（随机划分 15% 数据）上运行。CoKV 使用 8×RTX 3090 GPU 并行计算不同 coalition size。250 samples/coalition size 时 MAE<1/256（约 20.93 小时），满足精度要求。推荐进行两次独立采样，MAE<1/n 时取平均作为最终 SSV。SSV 具有任务特异性——不同 task 的 SSV 分布差异显著，但同 task 类型内泛化性好（附录 B.4 交叉验证）。推理时根据用户所选 task 加载对应的 SSV 分数表。代码开源：https://github.com/nawei1010/CoKV。

涉及论文标题：
- CoKV: Optimizing KV Cache Allocation via Cooperative Game

---

## Shapley Value in LLM Attention Head Evaluation (Shapley 值在 LLM 注意力头评估中的应用)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Shapley Value (Shapley, 1953) 是合作博弈论中的经典公平分配解。在 LLM 语境下，CoKV 将每个 attention head 视为博弈中的"玩家"，定义效用函数 U(S) 为 coalition S 中 head 未被 mask、其余 head 被 mask 时模型在验证集上的准确率。Shapley Value SV_i 衡量 head h_i 在所有可能 coalition 中的期望 marginal contribution：

$$SV_i = \frac{1}{n} \sum_{S \subset N \setminus \{h_i\}} \frac{U(S \cup \{h_i\}) - U(S)}{\binom{n-1}{|S|}}$$

直接计算 Shapley Value 需要评估指数级数量的 coalition（#P-hard），在 LLM 中完全不可行——Llama-3-8B-Instruct 有 256 个 KV groups（via GQA），枚举所有 coalition 需要 2^256 次模型推理。

CoKV 利用 complementary contribution 形式重写 Shapley Value：

$$SV_i = \frac{1}{n} \sum_{S \subset N \setminus \{h_i\}} \frac{U(S) - U(N \setminus S)}{\binom{n-1}{|S|}}$$

这使一次采样可同时更新 coalition S 中所有 head 的估计值，而非仅更新一个 head（如传统 marginal contribution 形式）。CoKV 进一步提出 Sliced Shapley Value (SSV)，基于"不同 coalition size 下分布高度相关"的实证观察，仅计算少数 coalition size 的贡献。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Shapley Value 在 LLM 推理中的语义**：

- U(S) = accuracy when heads in S are active (retain all KV), heads in N\S are masked (retain only local window KV)
- SV_i > 0: head h_i 在合作中对模型性能有正贡献，其 KV 应被保留更多
- SV_i ≈ 0: head h_i 对合作贡献微小，可分配较少 cache
- SV_i < 0: head h_i 对合作有负贡献（removing it improves performance），可不分配额外 cache

CoKV 将效用函数 U 定义为模型在特定 task 验证集上的准确率，这意味着 Shapley Value 是 task-specific 的——同一 head 在不同 task 上的重要性不同。这与 HeadKV 等 baseline 的 task-agnostic 评估形成鲜明对比。

术语一般如何实现？如何使用？

Shapley Value 在 LLM 中的直接计算不可行，需通过采样近似。CoKV 的 SSV 是当前该场景下的 SOTA 近似方法。CoKV 推荐在 8 卡 GPU 上并行计算，250 samples/coalition size 时精度满足要求（MAE<1/n）。SSV 预计算后存储为 per-task 的分数表，推理时查表加载即可，不增加推理延迟。代码开源：https://github.com/nawei1010/CoKV。

涉及论文标题：
- CoKV: Optimizing KV Cache Allocation via Cooperative Game

---

## Complementary Contribution (互补贡献) in Cooperative Game Theory

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Complementary Contribution（互补贡献）是 Shapley Value 计算中的一种高效采样形式。给定 coalition S ⊆ N（N 为所有玩家集合），互补贡献定义为 U(S) - U(N\S)。与传统的 marginal contribution U(S∪{h_i}) - U(S) 不同，互补贡献的一次采样结果可同时用于更新 S 中所有玩家的 Shapley Value 估计：

对每个 h_i ∈ S，$SV_i = \mathbb{E}[\frac{U(S) - U(N \setminus S)}{|S|}]$

在 LLM 语境下，U(S) 是 coalition S 中 head 活跃（保留完整 KV cache）、N\S 中 head 被 mask（仅保留 local window KV）时的模型准确率。关键优势：marginal contribution 的每次采样（加入/不加入一人）只能更新一个 head 的估计值，而 complementary contribution 每次采样（对比 S vs N\S）可更新 |S| 个 heads。当 |S|≈n/2 时，效率提升约 n/2 倍。

CoKV 采用 complementary contribution 而非 traditional marginal contribution 正是利用了这一效率优势，使得在 8×3090 GPU 上约 21 小时即可完成 256 个 groups 的准确评估。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Complementary Contribution 的一次采样**：

```
// 一次 Complementary Contribution 采样
π = random_permutation([h_1, ..., h_n])
j = random_choice(H)                     // H = {32,64,96,128}
S = {π(1), ..., π(j)}                    // 前 j 个 heads

// 两次前向推理
acc_active = forward(model, mask=N\S)    // S 中 heads 活跃，其余 mask
acc_masked = forward(model, mask=S)      // S 中 heads mask，其余活跃

// 互补贡献
u = acc_active - acc_masked

// 同时更新 S 中所有 |S|=j 个 heads
for h in S:
    SV_{h, j}.accumulate(u)
    count_{h, j} += 1

// 对比：Traditional Marginal Contribution（一次仅更新 1 个 head）
// h_i = random_choice(N)
// S = random_subset(N \ {h_i})
// u = U(S ∪ {h_i}) - U(S)
// SV_i.accumulate(u)    ← 仅 1 个 head 受益
```

**效率对比**：
- Marginal Contribution：M 次采样 → 每个 head 期望 M/n 次更新
- Complementary Contribution：M 次采样 → 每个 coalition 平均 n/2 大小 → 每个 head 期望 M·|H|/2n 次更新 → 约 |H|/2 ≈ 2 倍加速（|H|=4, n=256）

术语一般如何实现？如何使用？

Complementary Contribution 源自 Zhang et al. (SIGMOD 2023) 和 Sun et al. (TKDE 2024) 的 Shapley Value 近似理论。在 LLM 场景中，每次采样需要 2 次模型前向推理（一次算 U(S)，一次算 U(N\S)）。CoKV 按 coalition size 并行化：每个 coalition size j∈H 分配独立 GPU 计算，8 卡服务器上 4 个 coalition size 各用 2 卡做独立采样。采样结果取平均后 MAE<1/n 即为收敛。代码开源：https://github.com/nawei1010/CoKV。

涉及论文标题：
- CoKV: Optimizing KV Cache Allocation via Cooperative Game

---

## Head-Level KV Cache Budget Allocation (注意力头级别的 KV Cache 预算分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Head-Level KV Cache Budget Allocation 是一种在 Transformer 推理中将有限的 KV cache 总预算按 attention head 的重要性非均匀分配的策略。与 token-level 压缩（如 H2O、A2SF 在每个 head 内独立选择保留哪些 token）和 layer-level 分配（如 PyramidKV 深层分配更少 cache）不同，head-level 分配认识到不同 attention head 对模型性能的贡献差异显著，因此应为重要 head 保留更多 KV pairs，不重要 head 保留更少或仅保留 local window。

CoKV 的预算分配公式：

$$c_i = B \cdot \frac{\mathcal{NSV}_i}{\sum_{j=1}^n \mathcal{NSV}_j} + s$$

其中 B 为共享预算总数（总 KV pairs 减去所有 head 的 local window 固定部分），NSV_i 为 head h_i 的归一化 SSV 分数（α 个最低分 head 的 NSV=0，仅保留 local window），s=8 为 local window 大小。最终每个 head 的 cache size c_i = 按 SSV 分数比例分配 + 固定 local window。

CoKV 实验发现：当平均 cache size 达 512 tokens/group（约 6.4% of full cache for 8K context），CoKV 平均准确率超越 Full KV Cache（说明 CoKV 成功识别并限制了有负面贡献的 heads）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Head-Level Budget Allocation 的完整流程（CoKV）**：

```
// === 预计算阶段（Offline） ===
// 对每个 task，计算 SSV 分数
SSV[task_id] = compute_SSV(model, validation_set, H={32,64,96,128}, M=250)

// === 推理阶段（Online） ===
B = total_shared_budget  // e.g., (avg_cache_size - s) * num_groups
s = 8                    // local window
α = optimal_alpha[task]  // 从验证集选取

// 加载指定 task 的 SSV
ssv = load_SSV(task_id)  // shape: [num_groups=256]

// Step 1: Min-max normalize with α-low cutoff
sorted_ssv = sort(ssv)
min_α = sorted_ssv[α]
max_ssv = sorted_ssv[-1]
nsv = zeros_like(ssv)
for i where ssv[i] > min_α:
    nsv[i] = (ssv[i] - min_α) / (max_ssv - min_α)

// Step 2: Proportional allocation
total_nsv = sum(nsv)
for each group i:
    if nsv[i] == 0:
        c_i = s              // only local window
    else:
        c_i = B * (nsv[i] / total_nsv) + s

// Step 3: Per-group token selection (SnapKV mechanism)
for each group i:
    A_win = softmax(Q_win @ K_prefix^T / sqrt(d_h))
    token_scores = A_win.max(dim=1).mean(dim=0)
    keep_idx = topk(token_scores, c_i)
    K_retain = K_prefix[keep_idx]
    V_retain = V_prefix[keep_idx]
    K_cache = cat([K_retain, K_local])
    V_cache = cat([V_retain, V_local])
```

**与其他分配策略的对比**：

| 策略 | 粒度 | 重要性依据 | 代表方法 |
|------|------|-----------|---------|
| Uniform | per-head | 无（均分） | SnapKV |
| Layer-level | per-layer | 层深度（金字塔形） | PyramidKV |
| Head-level (独立) | per-head | 个体 retrieval-reasoning 分 | HeadKV-R2 |
| Head-level (稀疏度) | per-head | 个体 concentration degree | Ada-SnapKV |
| Head-level (合作) | per-head | 合作博弈 Shapley Value | CoKV |

术语一般如何实现？如何使用？

Head-level 分配在推理前计算各 head 的 cache size c_i，推理时在每个 Transformer 层 prefill 完毕后按照各自的 c_i 独立执行 token eviction。与 GQA 兼容：GQA 中一组 query heads 共享同一 KV cache，CoKV 将每个 KV group 作为合作博弈的玩家，评估 group-level SSV 后按 group 分配 budget。CoKV 在 Mistral-7B 和 Llama-3-8B 上验证，128 tokens/group 时保留 Full KV 97.29% 的性能。代码开源：https://github.com/nawei1010/CoKV。

与 CoKV 的 head-level 分配不同，CompressKV 采用 **Error-Aware 层级自适应分配（Layer-Level）**：离线在 LongBench 上模拟极端压缩（每层仅保留 ≈32 tokens），计算每层 attention output 的 Frobenius norm 重建误差 e^(l) = Σ_t ||O_comp^l - O_full^l||_F / ||O_full^l||_F，跨数据集归一化平均后得到层级重要性分数。在线推理时按分数比例分配 cache budget 给各 layer（而非各 head），设置 per-layer 上下界 [m=32, M=3×B_per-layer]。优势：(a) 离线计算无在线开销；(b) 基于真实压缩误差而非 attention 统计量（entropy/variance），跨模型泛化性更好。

涉及论文标题：
- CoKV: Optimizing KV Cache Allocation via Cooperative Game
- CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

注：KVzip 采用不同的 non-uniform allocation 策略——不是按 head 各自分配固定 budget，而是在所有 head 的所有 KV pairs 中取全局 top r% 最高得分进行保留。这天然导致重要性高的 head 保留更多 KV pairs，无需显式计算 per-head budget。该方法比 CoKV 的 Shapley-value 分配更简单，在 KVzip 的实验中优于 uniform allocation（Figure 16）。

---

## Additive Quantization (AQ) / 加性量化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

加性量化（Additive Quantization, AQ）由 Babenko & Lempitsky (CVPR 2014) 提出，是一种将高维向量压缩为多个码本中码字之和的向量量化方法。给定 d 维向量 $x \in \mathbb{R}^d$，AQ 使用 M 个码本 $\{C_1, C_2, \dots, C_M\}$（每个码本 $C_i \in \mathbb{R}^{d \times h}$ 含 h 个 d 维码字），将 x 近似为每个码本中选出一个码字的和：$x \approx \sum_{i=1}^M C_i b_i$，其中 $b_i$ 是 one-hot 向量从第 i 个码本中选出一个码字。由于码本不要求两两正交（不同于 Product Quantization 需要维度划分），AQ 通常获得比 PQ 更低的量化误差。但编码复杂度为 NP-hard（等价于全连接成对 Markov Random Field 的 MAP 推断），实际操作中常用 beam search 近似编码。

在 CommVQ 中，AQ 被适配用于 KV cache 压缩：(1) 使用学到的编码器（线性层 + 激活函数 + Gumbel-Softmax）替代传统 beam search 编码，实现端到端可微训练；(2) 码本通过梯度下降优化，最小化原始 KV 向量与解码向量间的 MSE loss；(3) 解码简单高效：$\hat{t}_i = s_i C$（二进制序列 s_i 与码本 C 的矩阵乘法）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**CommVQ 中 AQ 的 pipeline 流程**：

```
// 离线训练阶段
calibration_set = collect_kv_cache(FineWeb-Edu, model)
encoder = Linear(d, hidden) + Activation + Linear(hidden, N_c) + GumbelSoftmax
codebook = Parameter(N_c, d)  // 可学习码本

for epoch in range(epochs):
    for k_vec in calibration_set:  // k_vec: [d]
        s = encoder(k_vec)          // s: [N_c], 二进制序列
        k_hat = s @ codebook        // k_hat: [d], 解码重建
        loss = MSE(k_vec, k_hat)
        loss.backward()  // 同时优化 encoder 和 codebook

// 推理阶段 - Prefill
K_prefill, V_prefill = QKV_proj(X_prompt)  // [N, d]
S_K[i] = encoder_K(K_prefill[i])            // 每 token 独立编码
S_V[i] = encoder_V(V_prefill[i])
store(S_K, S_V)  // 替代 FP16 KV cache
```

术语一般如何实现？如何使用？

AQ 码本容量为 $h^M$（指数级），远超 PQ 的 $Mh$。在 CommVQ 中，M=1（单码本），$N_c$（码本行数，即 $h$ 维度）控制压缩率：Avg. bit = $N_c/d$。LLaMA-3.1-8B (d=1024)：$N_c=1024$ 对应 1-bit，$N_c=2048$ 对应 2-bit。在通用场景中，AQ 用于近似最近邻搜索、图像分类（压缩 SIFT/GIST 特征）、向量数据库索引。对于 LLM 推理中的 KV cache 压缩，AQ 对 Value cache 使用标准形式的加法量化，对 Key cache 通过 RoPE-可交换码本变体以高效融入 self-attention。

涉及论文标题：
- CommVQ: Commutative Vector Quantization for KV Cache Compression

---

## RoPE-Commutative Codebook (RoPE-可交换码本)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

RoPE-可交换码本是 CommVQ 中针对 Key Cache 设计的特殊码本结构。由于 Key 向量在 self-attention 中会经过 RoPE 旋转，标准 AQ 解码后还需单独应用 RoPE，导致解码开销与 self-attention 叠加。RoPE-可交换码本利用 RoPE 矩阵的 2x2 块对角特性：一个 $2 \times 2$ 矩阵 $C = \begin{pmatrix} x & y \\ -y & x \end{pmatrix}$ 与 RoPE 旋转矩阵 $R_m^i = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}$ 满足交换律 $R_m^i C = C R_m^i$。通过在 2D 子空间中设计满足该形式的子码本 $C_K^{jl}$，使 key-query attention 计算 $\alpha_i = q R_t (s_i C_K R_i)^T$ 可改写为 $\alpha_i = \sum_{j,l} (q^j R_t^j) C_K^{jlT} R_i^{jT} [s_i^j=l]^T$，其中 $(q^j R_t^j) C_K^{jlT}$ 对所有 token i 仅需计算一次，解码从 $O(d N_c N)$ 降为与 self-attention 同量级的开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**RoPE-可交换码本的 decoding 流程（单层单头 decoding step）**：

```
// 输入：当前 query q [1, d]，量化 key cache S_K [N, d/2]，2D 子码本 C_K^j_l [2, 2]
// 预计算：对所有 j, l 计算 (q^j R_t^j) C_K^{jlT}，仅需一次
q_proj = q @ W_Q              // [1, d]
q_rope = apply_rope(q_proj)   // 按 2D 子空间旋转

// 预计算复用部分
for j in 0..d/2:              // 遍历 d/2 个 2D 子空间
    q_j = q_rope[2*j:2*j+2]   // [2]
    for l in 0..N_c'-1:       // 遍历每个量化级别
        precomp[j][l] = q_j @ C_K^j_l^T  // [2] @ [2,2]^T -> [2]

// 逐 token 计算 attention score
for i in 0..N-1:
    alpha[i] = 0
    for j in 0..d/2:
        s_val = S_K[i][j]      // 量化索引，每维度 ∈ [0, N_c'-1]
        alpha[i] += dot(precomp[j][s_val[0]], rope_rotate_i(s_val[0]))
                  + dot(precomp[j][s_val[1]], rope_rotate_i(s_val[1]))
```

**聚类中心定义**：码本的 $N_{c'}^2$ 个聚类中心定义为：
$$c_{a,b} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} C_K^j[a] + \begin{bmatrix} 0 \\ 1 \end{bmatrix} C_K^j[b]$$

其中 $C_K^j[a]$ 和 $C_K^j[b]$ 是第 a 和第 b 个 $2 \times 2$ 可交换子码本。每个 2D 子向量被分配到最近聚类中心，用索引对 {a, b} 作为量化表示。

术语一般如何实现？如何使用？

子码本通过 EM 算法在 FineWeb-Edu 校准集上训练（含 soft clustering assignment + temperature annealing 稳定训练）。为提升压缩率，将连续 g 个 2D 子空间分为一组共享量化向量 s，Avg. bit = $R \cdot \log_2(N_{c'}) / g$（R 为残差迭代次数）。CommVQ 配置：$N_{c'}=64$, $g=64$，1-bit 时 R=11，2-bit 时 R=21。Key 码本总大小仅 2.75 MB (1-bit) / 5.25 MB (2-bit)，与 token 数无关。

涉及论文标题：
- CommVQ: Commutative Vector Quantization for KV Cache Compression

---

## Gumbel-Softmax for Differentiable Quantization Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Gumbel-Softmax（Jang et al., 2016）是一种将离散分类采样重参数化为可微操作的技术。给定类别概率 $\pi_1, \dots, \pi_k$，标准 Softmax 输出概率分布，Gumbel-Softmax 通过引入 Gumbel 噪声 $g_i \sim \text{Gumbel}(0,1)$ 并使用温度参数 $\tau$ 控制输出的连续松弛：$y_i = \frac{\exp((\log \pi_i + g_i) / \tau)}{\sum_j \exp((\log \pi_j + g_j) / \tau)}$。当 $\tau \to 0$ 时，$y$ 趋近 one-hot 向量；当 $\tau$ 较大时，$y$ 是平滑的连续向量。这使得离散量化过程可端到端梯度优化。

在 CommVQ 中，Gumbel-Softmax 用于编码器输出层，使编码器将连续的 KV 向量映射到离散的二进制序列 s_i ∈ {0,1}^{N_c} 的过程保持可微。训练时 τ 较大（平滑梯度），推理时 τ → 0（硬量化）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**CommVQ 中 Gumbel-Softmax 编码器训练流程**：

```
encoder = Sequential(
    Linear(d, hidden_dim),
    ReLU(),
    Linear(hidden_dim, N_c),
    GumbelSoftmax(tau=1.0, hard=False)  // 训练模式：连续松弛
)

for t_i in kv_cache_batch:  // t_i: [d]
    s_i = encoder(t_i)      // s_i: [N_c], 连续值 ∈ (0,1)
    t_hat_i = s_i @ C       // C: [N_c, d] 码本
    loss = MSE(t_i, t_hat_i)
    loss.backward()         // 同时更新 encoder 参数和 codebook C

// 推理阶段：切换到 hard mode
encoder[-1].hard = True     // τ → 0, 输出近 one-hot
s_i = encoder(t_i)          // s_i: [N_c], 精确 {0,1}
store(s_i)                  // 每维 1 bit
```

术语一般如何实现？如何使用？

PyTorch 实现：`F.gumbel_softmax(logits, tau=1.0, hard=False)`。训练时通常使用温度退火（temperature annealing）：初始 τ 较高（1.0-5.0）以保证梯度平滑流动，随训练逐步降低 τ 使输出接近离散。在 VQ-VAE、DALL-E（码本学习）、CommVQ（KV cache 编码器）中广泛使用。CommVQ 的编码器为每层每 token 独立运行，prefill 阶段一次性编码，不增加 decoding 阶段开销。

Dynamic-LLaVA 将 Gumbel-Softmax + STE 用于 token pruning：两个 predictor（Image/Output predictor）各自输出一个 [N, 2] 维的决策矩阵 D，沿第二维做 Gumbel-Softmax 得到 D†（连续松弛），forward 时 argmax 生成离散 mask M，backward 时 STE 将 ∂L/∂M 直接传递给 ∂L/∂D†，绕过 argmax 的不可微问题。τ 从 1 指数衰减至 0.1。与 CommVQ 的差异：CommVQ 用于连续向量→离散码本映射（量化），Dynamic-LLaVA 用于 token keep/discard 二分类决策（剪枝）。

涉及论文标题：
- CommVQ: Commutative Vector Quantization for KV Cache Compression
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification for Codebook Learning (码本学习的EM算法)
- Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

**Elastic Attention 中的 Gumbel-Sigmoid for Attention Routing**：不同于 CommVQ 的 Gumbel-Softmax（多类别码本映射）和 Dynamic-LLaVA 的 Gumbel-Softmax（token keep/discard 二分类），Elastic Attention 使用 Gumbel-Sigmoid（二分类 Gumbel-Softmax）对每个 head 做 FA vs SA 二选一路由。具体：z ∈ R^{H×2} 经 Gumbel-Sigmoid 得 r_soft ∈ R^{H×2}（连续松弛），argmax 得 r_hard（离散路由），STE 传导梯度。温度 τ 按 τ(t) = max(τ_min, τ_init · exp(-r·p)) 退火（r=0.6）。早期高 τ 鼓励探索，后期低 τ 逼近离散 Bernoulli。与 Lagrange 乘子协同优化。

## Lagrangian Constraint Training for Sparsity (拉格朗日约束稀疏训练)

术语是什么？

Lagrangian Constraint Training for Sparsity 是一种通过可训练 Lagrange 乘子（λ1, λ2）在 min-max 框架下自动平衡语言建模损失与稀疏度约束的训练方法。与固定惩罚系数不同，λ 通过 gradient ascent 动态调整，使不同任务能容忍不同的 sparsity-vs-performance gap。Elastic Attention（Tang et al., 2025）和 PruLong（Bhaskar et al., 2025）均采用此方法。

从算法pipeline角度拆解术语。

```
# 训练目标（Elastic Attention）
# 外层: max_{λ1, λ2}, 内层: min_{router params}
Ω_MSR = (1/(L·H)) · Σ_l Σ_h I[r_hard[l,h] == SA]
L_diff = Ω_MSR - t        # t = target sparsity
L = L_language + λ1·L_diff + λ2·L_diff²

# 梯度下降更新 router 参数
θ_router -= lr_router · ∂L/∂θ_router

# 梯度上升更新 Lagrange 乘子
λ1 += lr_λ · L_diff
λ2 += lr_λ · (2·λ2·L_diff)
```

术语一般如何实现？如何使用？

Elastic Attention：sparsity-sensitive tasks t=0.7（更多 FA），sparsity-robust tasks t=1.0（全部 SA）。λ 随机初始化，`lr_λ=1e-3`（高于 router lr=5e-4）。训练时不同 task 的 λ 收敛到不同值——敏感任务 λ 更大（更强约束），鲁棒任务 λ 更小（宽松），自动实现 task-dependent sparsity。PruLong 中 t warmup 0→0.7（800 steps），λ 同样可训练。两种方法都使用 non-tight constraint（不强制精确满足 t）。

涉及论文标题：
- Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs

## Attention Router (注意力路由器)

术语是什么？

Attention Router 是 Elastic Attention 提出的轻量级模块，以 MoE 风格的 gating 在推理时动态决定每个 attention head 使用 FA 还是 SA 计算模式，实现 test-time 自适应稀疏度分配。每层仅 0.27M 参数（head_dim=128），由 Task MLP + Router MLP 组成。输入 Key hidden states，输出 head-wise 二值路由决策。

从算法pipeline角度拆解术语。

```
# Attention Router 前向（per layer）
Input: x_K ∈ R^{s×H×d'}  # Key hidden states

# Step 1: Boundary Pooling（首100 + 尾100 tokens）
x_K' = Pool(x_K[:100] ∪ x_K[-100:])  # [H, d']

# Step 2: Task MLP → task-specific features
z_task = SiLU(W_task1 @ x_K')        # intermediate = 4×d'
z_task = W_task2 @ z_task            # [H, d']

# Step 3: Router MLP → routing logits
z = SiLU(W_router1 @ z_task)
z = W_router2 @ z                    # [H, 2]

# Step 4: Gumbel-Sigmoid → hard routing
g = -log(-log(u + ε) + ε)           # Gumbel noise
r_soft[:, 1] = σ((z[:, 1] - z[:, 0] + g) / τ)  # SA prob
r_hard = argmax(r_soft, dim=-1)      # 0=FA, 1=SA
r_hard = r_hard + (r_soft - detach(r_soft))  # STE
```

术语一般如何实现？如何使用？

训练：backbone 冻结，仅优化 Router（12h on 8×A800，300 steps，seq_len=65536），decoupled LR（router=5e-4, λ=1e-3），训练数据 0.74B tokens 五源混合。推理：Router 仅 ~0.196ms/call，延迟不随 seq_len 增长（Boundary Pooling 固定 200 tokens）。SA 模式可选 SSA（sink+local window）或 XA（block sparse），训练时同时学习。代码：https://github.com/LCM-Lab/Elastic-Attention。

涉及论文标题：
- Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

## Model Sparsity Ratio (Ω_MSR) / Effective Sparsity Ratio (Ω_ESR) (模型稀疏率 / 有效稀疏率)

术语是什么？

Ω_MSR 和 Ω_ESR 是 Elastic Attention 提出的形式化稀疏度量。Ω_MSR = 使用 SA 的 head 比例（不考虑 per-head 内部剪枝率），Ω_ESR = 综合 head 比例和 per-head token 剪枝率的实际 attention 覆盖比例。Ω_ESR 用于 fair comparison——不同 SA 方法（SSA vs XA）的 per-head pruning ratio 不同，Ω_ESR 折算到"实际被 attention 覆盖的 token 比例"。

从算法pipeline角度拆解术语。

```
Ω_MSR = (1/(H·L)) · Σ_h Σ_l I[π^{(l,h)} = SA]

Ω_ESR = (1/(H·L)) · Σ_h Σ_l ρ^{(l,h)}
# ρ: FA head=0, SA head=ρ_SA (e.g., SSA prunes 90% → ρ=0.9)

# Example: 50% heads SA with 80% token pruning
# Ω_MSR = 0.5, Ω_ESR = 0.5 × 0.8 = 0.4
```

术语一般如何实现？如何使用？

Ω_MSR 用于训练约束（L_diff = Ω_MSR - t）。Ω_ESR 用于推理效率对比——RULER 实验中 Elastic Attention 在长序列下 Ω_ESR 低于同类方法（更少 token 被 attention 覆盖），证明 adaptive sparsity 比 static assignment 更 effective。Figure 8 使用 Ω_ESR 做 performance-vs-sparsity Pareto frontier 对比。

涉及论文标题：
- Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers（Expectation-Maximization, EM）算法来学习 RoPE-可交换码本。由于可交换码本需满足特定的 $2 \times 2$ 矩阵形式约束（$C = [[x, y], [-y, x]]$），且码本训练本质上是聚类问题，EM 比梯度下降更直接有效。EM 算法的目标：将 N 个 2D 校准向量 $k \in \mathbb{R}^{N \times 2}$ 分配到 $N_{c'}^2$ 个聚类中心（由 $N_{c'}$ 个 $2 \times 2$ 可交换子码本组合而成），最小化 MSE loss。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**CommVQ 码本学习的 EM 算法**：

```
# 输入：校准集 K ∈ R^{N×2}（每个 2D 子空间的 key 向量）
# 参数：码本 C_K^j = {C_K^{j0}, ..., C_K^{j(N_c'-1)}}
#      每个 C_K^{jl} = [[x_l, y_l], [-y_l, x_l]]

# 构建 N_c'^2 个聚类中心
for a in 0..N_c'-1:
    for b in 0..N_c'-1:
        c_ab = [1,0] @ C_K^j[a] + [0,1] @ C_K^j[b]  # [2]

while not converged:
    # E Step: 固定码本，soft assignment + temperature
    D = L2_distance_matrix(K, cluster_centers)  # [N, N_c'^2]
    W = softmax(-D / T)                          # soft assignment 权重
    m = W^T @ K                                  # 加权均值
    N_counts = sum(W, dim=0)                     # 每中心分配数

    # M Step: 固定分配，闭式解更新码本参数
    phi = (T^T @ S @ T)^{-1} @ T^T @ S @ m      # 闭式解

    T = T * decay_rate  # 温度退火
```

**闭式解的矩阵形式**：
$$\phi^* = (T^T S T)^{-1} T^T S m$$

其中 $T \in \{-1,0,1\}^{(2N_{c'}^2) \times (2N_{c'})}$ 是编码聚类中心与码本关系的常数矩阵，$S = \operatorname{diag}(N_{ij})$ 是分配计数的对角矩阵。

术语一般如何实现？如何使用？

为稳定训练，CommVQ 采用两项关键技术：(1) **Soft clustering assignment**：不用 hard assignment，而是根据距离对每个数据点到所有中心赋权重 $W_{ij} = e^{-D_{ij}/T} / \sum_k e^{-D_{ik}/T}$，防止死聚类中心；(2) **Temperature annealing**：温度 T 从高到低指数衰减。对于 1-bit 量化（$N_{c'}=64$）有 4096 个聚类中心，soft assignment 是关键。R 轮迭代式残差量化（每轮在上轮误差上拟合新码本）。训练在 FineWeb-Edu 校准集上进行，每层独立训练。

涉及论文标题：
- CommVQ: Commutative Vector Quantization for KV Cache Compression

---

## Residual Vector Quantization (RVQ) / 残差向量量化

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

残差向量量化（Residual Vector Quantization, RVQ）是一种多级向量量化方法。对输入向量 x，RVQ 使用 K 个级联码本：第一级找到码本 C_1 中最接近 x 的码字 $\hat{x}_1$，计算残差 $r_1 = x - \hat{x}_1$；第二级在 C_2 中找最接近 r_1 的码字 $\hat{r}_2$，累积近似 $\hat{x} = \hat{x}_1 + \hat{r}_2$；重复 K 次。最终 $\hat{x} = \sum_{k=1}^K \hat{c}_k$（K 个码本各选一个码字的和）。RVQ 的思想类似于梯度提升或残差学习。

在 VQLLM（Kumar 2024）中，RVQ 被用于 KV cache 压缩：将 key/value 向量按 channel group（d̂=32）分组，每组用独立的 K=8 级 RVQ 量化，每级 C=2048 个码字。码本通过 EMA（指数移动平均）在线更新。CommVQ 也采用了类似的"残差迭代"思想——通过 R 轮 EM 算法在上轮误差上拟合新码本（R=11 for 1-bit, R=21 for 2-bit）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**RVQ 标准编码/解码流程**：

```
def rvq_encode(x, codebooks):  # codebooks: list of K codebooks
    indices = []
    residual = x
    for k in range(K):
        idx = argmin(||residual - codebooks[k][j]|| for j in 0..C-1)
        indices.append(idx)
        residual = residual - codebooks[k][idx]
    return indices

def rvq_decode(indices, codebooks):
    x_hat = zero_vector
    for k in range(K):
        x_hat += codebooks[k][indices[k]]
    return x_hat
```

术语一般如何实现？如何使用？

RVQ 广泛应用于音频压缩（SoundStream、EnCodec）、图像生成（VQ-VAE-2）和 KV cache 压缩（VQLLM）。VQLLM 的 RVQ 实现使用非连续 channel grouping（对 Key 取间隔 d/d̂ 的通道），并在 Triton kernel 中融合 K 级查找/累加。RVQ 优势是编码质量随 K 增加而单调提升，但计算和存储开销也随 K 线性增长。CommVQ 的"残差迭代"与 RVQ 类似但实现不同：CommVQ 每轮使用完整 EM 聚类（而非单次最近邻查找），且所有 R 轮共享 g=64 的分组结构。

涉及论文标题：
- CommVQ: Commutative Vector Quantization for KV Cache Compression

---

## Modular Hierarchical Token Pruning (模块化层次化Token剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Modular Hierarchical Token Pruning 是 InfiniteHiP 提出的免训练长上下文 token 剪枝算法。核心思想：通过堆叠多个剪枝模块（pruning stage），每个模块基于当前 query block 动态评估 key chunk 的重要性，逐步将候选 key token 从全量缩减到常数级别（~2K-4K），最终生成 block sparse attention mask。与 HiP Attention 的迭代式 top-k 不同，每个剪枝模块使用 per-chunk top-1 代表 token 选择（而非全局 top-k），消除了全局 thread synchronization，实现 key sequence dimension 上的高并行度。

每个剪枝 stage S^(i) = (b_q^(i), l_c^(i), k^(i)) 包含三个参数：query block size b_q、chunk size l_c、保留 token 数 k。Stage 间数据流：全量 key → Stage 0: 分 chunk(l_c=256)→每 chunk 选代表 token→保留 top K chunk→Stage 1: 分 chunk(l_c=32)→选代表→保留 top K chunk→Stage 2: 分 chunk(l_c=8)→选代表→保留 top K chunk→输出 ~2K-4K key indices 用于 BSA。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**三阶段模块化剪枝 pipeline（3K preset, T_kv=1M tokens）**：

```
Input: Q ∈ R^(H×T_q×d), K ∈ R^(H×T_kv×d), n_sink=256, n_stream=1024
Output: Sparse key indices I^(3) for each query block

// Stage 0: 全量 key → 32K tokens
I^(0) = [n_sink, ..., T_kv - n_stream]  // 排除 sink/streaming
S^(0) = (b_q=64, l_c=256, k=32K)
For each query block m:
  C_j = chunk(I^(0), l_c=256)  // 约 4000 chunks
  For each chunk j:
    r_j = SelectRep(q_m, C_j, K)  // O(log₂ 256)=8 次点积
    s_j = max_{h,t} (q_{h,t}^T · k_{h, r_j})  // chunk 注意力分数估计
  T = argtop_{125}(s)  // 保留 top 125 chunks (125×256≈32K)
  I'^(0) = ∪_{j∈T} C_j
// Stage 1: 32K → 8K
S^(1) = (b_q=64, l_c=32, k=8K)
For each query block m:
  C_j = chunk(I'^(0), l_c=32)  // 约 1000 chunks
  r_j = SelectRep(q_m, C_j, K)
  s_j = max_{h,t} (q_{h,t}^T · k_{h, r_j})
  T = argtop_{250}(s)  // 保留 top 250 chunks (250×32=8K)
  I'^(1) = ∪_{j∈T} C_j
// Stage 2: 8K → ~3K (2K for layers >3, 4K for layers ≤3)
S^(2) = (b_q=64, l_c=8, k=2048|4096)
For each query block m:
  C_j = chunk(I'^(1), l_c=8)  // 约 256-512 chunks
  r_j = SelectRep(q_m, C_j, K)
  s_j = max_{h,t} (q_{h,t}^T · k_{h, r_j})
  T = argtop_{K}(s)
  I^(2) = ∪_{j∈T} C_j

// Final: BSA with I^(2) ≈ 2K-4K selected keys + sink + streaming
```

术语一般如何实现？如何使用？

InfiniteHiP 使用单个参数化的 Triton kernel 实现所有剪枝 stage，通过不同 (b_q, l_c, k) 参数区分。SelectRep 算法每次迭代仅访问 2 个 token（左右分支首 token），因此无需全局同步——这与 HiP Attention 的迭代式 top-k（需要全局同步）形成关键差异。剪枝 module 的数量 N=3 经实验确定的延迟-性能最优组合。

涉及论文标题：
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

## Chunk Sparsity of Attention (注意力的Chunk稀疏性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Chunk Sparsity of Attention 是 InfiniteHiP 发现并利用的 LLM 注意力分布规律：在长上下文中，top-k 高注意力 token 高度集中在极少数 chunk 中，而非均匀分布在整个序列上。具体观察（Llama 3.1 8B, 128K context）：(1) 不到 2% 的 chunk 包含了超过 12.5% 的 top-2K token；(2) 约 75% 的 64-token chunk 不包含任何 top-2K token。这一观察构成了 InfiniteHiP 模块化剪枝算法的设计基础——通过选择包含 top-k token 的少数 chunk 而非逐个选择 top-k token，可在极低成本下获得良好的 top-k 近似。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Chunk Sparsity 驱动的 token 选择策略**：

```
// 传统 top-k 选择：O(T_kv) 逐个比较
topk_indices = argtop_k(attn_scores[:, :])  // 对每个 query 评估所有 key

// Chunk Sparsity 驱动的选择：O(n_chunks + k log n_chunks)
n_chunks = T_kv / l_c  // 如 128K/256 = 500 chunks
For each chunk j:
  r_j = SelectRep(q, chunk_j)  // 每 chunk 仅 1 次 O(log l_c) 操作
  s_j = estimate_score(q, k[r_j])  // 估计该 chunk 的最高注意力分数
top_chunks = argtop_K(s)  // 仅对 n_chunks 个分数排序
selected_keys = ∪_{j∈top_chunks} chunk_j  // 展开 chunk 得到约 K×l_c 个 key
```

与 InfLLM 的预选固定代表 token 不同，InfiniteHiP 每个 query block 都动态重选代表 token，使 select 精度更高（recall 比 InfLLM 高 1.57%、比 HiP 高 4.72%）。

术语一般如何实现？如何使用？

Chunk Sparsity 的分析方法：(1) 在给定上下文中运行 dense attention 获取完整 attention matrix；(2) 对每个 query position 取 top-k key indices；(3) 将 key 序列划分为固定大小的 chunk；(4) 统计每个 chunk 包含的 top-k key 数量；(5) 绘制直方图（chunk 频率 vs 包含的 top-k key 百分比）。这一方法可推广到其他模型以评估剪枝方法的适用性。

涉及论文标题：
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

## Dynamic RoPE Adjustment for Out-of-Length (OOL) Generalization (动态RoPE外推)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dynamic RoPE Adjustment 是 InfiniteHiP 提出的训练无关的长上下文外推策略。核心观察：LLM 的不同层具有不同的 attention pattern——早期层（layer ≤ 5）呈现 dynamic sliding window-like attention（依赖相对位置信息），后期层依赖语义内容。基于此，InfiniteHiP 在不同层和不同计算阶段使用不同的 RoPE position ID 策略：(1) 前 3 层剪枝阶段使用 Chunk-indexed RoPE；(2) 第 4 层起剪枝阶段使用 Relative-style RoPE；(3) BSA 阶段使用 StreamingLLM-style RoPE。

从算法pipeline角度拆解术语：

**三种 RoPE 策略的具体实现**（ApplyRopeQ 和 ApplyRopeK，layer index l）：

```
// Chunk-indexed RoPE (l ≤ 3, 在剪枝阶段)
ApplyRopeQ_l(q) = ApplyRope(q, p[min(i_orig, l_c + n_stream)])
  // i_orig: q 的原始 position
  // 将 position 钳制在 chunk 级粒度，引导滑窗式 mask
ApplyRopeK_lj(k) = ApplyRope(k, p[c_orig])
  // c_orig: k 所在 chunk 的索引
  // 同 chunk 内所有 key 共享同一 position ID

// Relative-style RoPE (l > 3, 在剪枝阶段)
ApplyRopeQ_l(q) = ApplyRope(q, p[n_stream + 1])
  // 对 query 使用统一的相对偏移
ApplyRopeK_lj(k) = ApplyRope(k, p[j-1])
  // j∈{1,2}: SelectRep 中左分支(j=1)或右分支(j=2)
  // 两分支获得不同 position offset，实现层次化相对编码

// StreamingLLM-style RoPE (BSA 阶段，所有层)
  // 选中的 key（含 sink+streaming）按原始顺序排列
  // 最近 token 获得与当前 query 相同的 position ID
  // 等效于在原始 RoPE 空间中重新索引选中的 token
```

**消融实验结果（∞Bench En.MC, Llama 3.1 8B, T=128K）**：
- Chunk-indexed in pruning + StreamingLLM in BSA: 67.69%
- Relative in pruning + StreamingLLM in BSA: 70.31%（best for BSA）
- 混合（前3层 Chunk-indexed + 后续 Relative）+ StreamingLLM BSA: 74.23%（best overall）

术语一般如何实现？如何使用？

实现关键：(1) 在 Triton kernel 中根据 layer index l 选择 ApplyRopeQ/ApplyRopeK 的分支；(2) 预计算多组 cos/sin 向量以支持不同 position offset（增加额外 memory read overhead，使 prefill 慢约 1.6×）；(3) 可动态开关——当不需要 OOL generalization 时（如上下文在预训练长度内），可禁用动态 RoPE 以消除 overhead。Chunk-indexed RoPE 仅在 context pruning 的前 3 层使用，不要在全部层使用（全层 Chunk-indexed 导致显著性能退化）。

涉及论文标题：
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

## SelectRep / Hierarchical Top-1 Representative Token Selection (层次化Top-1代表Token选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

SelectRep 是 InfiniteHiP 和 HiP Attention 中用于在 key chunk 内快速估计 top-1 注意力 token 位置的层次化二分搜索算法。给定 query block q 和 key chunk C（大小 l_c），SelectRep 通过 O(log₂ l_c) 次点积操作收敛到 chunk 内与 q 注意力分数最高的 token 的近似位置，而无需评估 chunk 内所有 l_c 个 token。这是实现高效 chunk-level 剪枝的关键组件。

从算法pipeline角度拆解术语：

```
Input: Query block q ∈ R^(b_q×d), key chunk indices C ∈ N^(l_c), Keys K
Output: Representative token index r ∈ C

1: q̃ = ApplyRopeQ_l(q)
2: n_iter = ⌈log₂(l_c)⌉  // 如 l_c=256 → 8 次迭代
3: (n_first, n_last) = (1, l_c)
4: For i = 1 .. n_iter:
5:   m = ⌊(n_first + n_last) / 2⌋  // 二分中点
6:   B₁ = [n_first, m-1], B₂ = [m, n_last]  // 左右分支
7:   For j ∈ {1, 2}:
8:     r_j = B_j[0]  // 取分支首 token 作为代表
9:     k̃_j = ApplyRopeK_lj(K[r_j])  // 位置编码
10:    σ_j = max_t (q̃_t^T · k̃_j)  // 分支分数
11:  t = argmax_j σ_j  // 选择高分分支
12:  (n_first, n_last) = B_t  // 更新搜索范围
13: r = n_first  // 收敛到单个 token
```

关键性质：(1) 每次迭代仅需 2 次 token-level 点积（与 chunk size l_c 无关），因此整个 SelectRep 仅需 2·log₂(l_c) 次点积；(2) 层次化二分搜索避免了 HiP Attention 原始实现中的全局 top-k 同步；(3) 利用 attention locality（邻近 token 的注意力分数相似）保证估计质量。

术语一般如何实现？如何使用？

InfiniteHiP 将 SelectRep 实现为单个 Triton kernel 的一部分（与 chunk score estimation + top-K chunk selection 融合），利用 GPU 的 key sequence dimension 并行度（类似 FlashDecoding 的 split-KV）。SelectRep 的左右分支分别使用不同的 RoPE position offset（j=1 偏移 n_stream+1，j=2 偏移 n_stream），实现层次化相对位置编码。

涉及论文标题：
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU
- HiP Attention: A Training-free Sub-quadratic Cost Transformer Model Serving Framework With Hierarchically Pruned Attention

## Sparse Attention Mask Caching (稀疏注意力Mask缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Sparse Attention Mask Caching 是 InfiniteHiP 用于降低 decoding 阶段剪枝开销的优化策略。核心观察：在连续 decoding step 中，相邻 query token 的 attention pattern 高度相似（temporal locality），因此不需要每步都重新计算所有剪枝 stage 的稀疏 mask。通过为每个剪枝 stage 独立维护 mask 缓存和 refresh interval（n_refresh^(i)），仅周期性地更新 mask，可大幅降低 decoding 延迟。

从算法pipeline角度拆解术语：

```
// Decoding loop with mask caching
c^(i) = 0 for i = 1..N  // stage counters

For each decoding step:
  For each layer l = 1..L:
    For each stage i = 1..N:
      if c^(i) % n_refresh^(i) == 0:
        I^(l,i) = RunPruningStage(q_l, K, I^(l,i-1))  // 重新计算 mask
        // 记录 cache miss → 从 CPU UVM 加载缺失 key
      // else: 复用缓存的 I^(l,i)（跳过剪枝计算）
    O_l = BlockSparseAttention(q_l, K, V, I^(l,N))  // BSA with cached mask
  c^(i) = (c^(i) + 1) % n_refresh^(i)  // 更新所有 counter
```

**三种 refresh 配置及其效果（256K context decoding latency per token）**：
- Default: n_refresh = (16, 8, 4) → All cached: 110 µs/token → mask hit ratio Stage1 71.67%, Stage1&2 98.75%
- Fast: n_refresh = (32, 16, 8) → lower refresh frequency, 更低的平均延迟
- Flash: n_refresh = (96, 24, 8) → Stage1 几乎从不重算 → 最高 throughput（3M context 23.8 tok/s on L40S）

术语一般如何实现？如何使用？

实现要点：(1) 每个 stage 维护独立的 mask indices I^(l,i) 和 counter c^(i)；(2) 第一 stage（最昂贵，O(T_kv)）的 refresh interval 最大（16/32/96），因为其 mask 变化最慢；(3) 后续 stage（更便宜，O(constant)）的 refresh interval 较小；(4) 可在解码速度（更大的 interval）和 mask 精度（更小的 interval）间 trade off——论文显示增大 interval 对 NLU 性能影响极小（LongBench/∞Bench 中 3K-fast 和 3K 差异 <1%）。

涉及论文标题：
- InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

## Semantic Retrieval Head (语义检索注意力头)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Semantic Retrieval Head 是 CompressKV（Lin et al., 2025）提出的注意力头分类概念，是对传统 Retrieval Head 的扩展。传统 Retrieval Head 识别标准要求 head 的 top-1 attention 精确落在正确答案 token 上（仅捕捉 copy-paste 行为，如 Wu et al., ICLR 2025 的定义）。Semantic Retrieval Head 的识别标准不要求精确 top-k 命中——而是聚合 head 在整个 answer span A 上的 attention scores 来评估语义检索能力，公式为：

$$\text{SemanticRetrievalScore}(h) = \sum_{t=1}^{N} \mathbb{I}[y_t \in A] \sum_{j \in A} a_{t,j}^h$$

其中 y_t 是第 t 步生成的 token，A 是 answer span，a_{t,j}^h 是 head h 在 token j 上的 attention weight。得分越高，说明该 head 越能捕捉语义信息（包括 copy-paste 行为和更深的语义依赖），而非仅 copy-paste。

核心 insight：在 long-context 场景下，attention distribution 极其稀疏，大量 attention 分配给 initial/final tokens（attention sink）。传统 top-1/top-k 标准的 hit rate 极低（大部分 head 得分为零），且仅捕捉 copy-paste 模式，忽略语义依赖。例如生成 "sandwich" 时，模型不仅 attend "sandwich"，还 attend 周围语义相关 token（如 "eat", "a thing"）——传统标准不认可这些 head，但 Semantic Retrieval Head 标准能捕获。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**SRH 识别与 Token 选择过程**：

```
// === 离线阶段：SRH 识别 ===
// 在验证集（如 LongBench）上运行完整模型
for each layer l in 0..L-1:
    for each head h in 0..H-1:
        SRScore[l][h] = 0
        for each decoding step t where y_t in answer_span A:
            a_t = attention_weights[l][h][t]  // [seq_len]
            SRScore[l][h] += sum(a_t[j] for j in A)
    // L1 normalize within layer
    SRScore[l] = SRScore[l] / sum(SRScore[l])

// 取每层 top-k SRH（默认 k=4）
topk_SRH[l] = argsort(SRScore[l])[-k:]

// === 在线 Prefill 阶段：SRH 驱动的 Token 选择 ===
// observation window W = 8, pooling kernel size = 5
for each layer l:
    selected_heads = topk_SRH[l]
    S = zeros(seq_len)
    for h in selected_heads:
        A_h = attention_scores[l][h][:, -W:]  // [seq_len, W]
        S_h = sum(A_h, dim=-1)                 // [seq_len]
        S_h = avg_pool1d(S_h, kernel_size=5)  // [seq_len]
        S += S_h
    S = S / len(selected_heads)  // average
    keep_indices = topk(S, N)    // select top-N tokens
    K_cache = K[keep_indices]
    V_cache = V[keep_indices]
```

术语一般如何实现？如何使用？

SRH 识别离线完成，在 LongBench 或类似验证集上运行完整模型一次即可。对于 Llama-3.1-8B-Instruct（32 层），每层仅需 4 个 SRH 即可达到最佳性能——增加至 6/12/24 个 head 无进一步收益。在 Mistral-7B 和 Llama-3.1-8B 上，SRH 识别结果与传统 Retrieval Head 显著不同：传统方法中 layer 0 和 1 的所有 head 得分为零，而 SRH 方法能识别出低但有效的语义重要性 head。Masking top-30 SRH 导致 NIAH 准确率下降 ~74%（vs 传统 Retrieval Head 仅下降 ~12%）。代码开源：https://github.com/TUDa-HWAI/CompressKV.git。

涉及论文标题：
- CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation

---

## Error-Aware Layer-Adaptive Cache Allocation (误差感知层级自适应缓存分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Error-Aware Layer-Adaptive Cache Allocation 是 CompressKV 提出的层级 KV cache 预算分配策略。与依赖 attention 统计量（entropy/variance，如 PyramidKV/CAKE）不同，该方法直接量化 KV cache 压缩对每层 attention output 造成的重建误差，以此作为层级重要性的代理指标。核心思想：对压缩敏感的层（误差大）分配更多 cache budget，对压缩不敏感的层（误差小）分配更少 budget。

在离线阶段，模拟极端压缩场景（每层仅保留 m=32 tokens，约 0.3% 全量），计算每层压缩前后的 attention-block output 之间的 Frobenius norm 重建误差：

$$e^{(l)} = \sum_{t=1}^{T} \frac{\|\mathbf{O}_{\text{comp},t}^{(l)} - \mathbf{O}_{\text{full},t}^{(l)}\|_F}{\|\mathbf{O}_{\text{full},t}^{(l)}\|_F + \epsilon}$$

其中 O_full 使用完整 KV cache 的 attention output（含 output projection W_O），O_comp 使用压缩后 KV cache 的 attention output，ε=10^{-6} 防止除零。跨数据集 L1 归一化后平均，得到最终层级重要性分数 ẽ^{(l)}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Error-Aware 分配的完整算法**：

```
# === 离线阶段：误差分数计算 ===
for each dataset d in LongBench:
    for each layer l:
        # 模拟压缩：每层仅保留 32 tokens
        K_comp^l, V_comp^l = retain_top(K_full^l, V_full^l, 32)
        for each decoding step t:
            O_full = Attention(Q_t, K_full^l, V_full^l) @ W_O^l
            O_comp = Attention(Q_t, K_comp^l, V_comp^l) @ W_O^l
            e_d^l += ||O_comp - O_full||_F / (||O_full||_F + 1e-6)
    e_hat_d^l = e_d^l / sum(e_d^k for all k)     # L1 norm within dataset

e_bar^l = mean(e_hat_d^l for all d in datasets)   # cross-dataset average
e_tilde^l = e_bar^l / sum(e_bar^k for all k)      # final importance

# === 在线阶段：预算分配 (Algorithm 1) ===
B_i = m  for all layers i                         # m = 32 minimum
R = B_total - sum(B_i)                            # remaining
B_i = clip(B_i + round(e_tilde_i * R), m, M)     # M = 3 * B_per_layer
delta = B_total - sum(B_i)
while delta != 0:
    if delta > 0:
        j = argmax(e_tilde_i for i where B_i < M)
        B_j += 1; delta -= 1
    else:
        j = argmin(e_tilde_i for i where B_i > m)
        B_j -= 1; delta += 1
return B
```

术语一般如何实现？如何使用？

离线误差计算在 LongBench 全部 16 个数据集上进行，取平均值确保不依赖特定 task。上下界 m=32 和 M=3×B_per-layer 通过实验调优。在 Mistral-7B 和 Llama-3.1-8B 上，不同模型的误差分布显著不同，验证了 error-aware 方法捕捉到了模型特定的层级差异。与 CAKE/PyramidKV 的关键差异：(a) 离线计算无在线开销；(b) 基于真实压缩误差而非 attention 统计量代理指标，跨模型泛化性更好。代码开源：https://github.com/TUDa-HWAI/CompressKV.git。实现包含 `longbench/get_avg.py` 用于跨数据集平均误差分数。

涉及论文标题：
- CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation

---

## KV Cache Eviction (KV 缓存逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Cache Eviction 是一类 KV cache 压缩方法，通过选择性地逐出不重要 token 的 KV cache pairs 来减少显存占用，无需额外训练或微调。与量化（减少每个值的 bit 宽度）和低秩分解（从通道维度压缩）不同，eviction 方法的压缩来自减少保留的 token 数量。

Eviction 方法的核心在于两个决策：(1) Token 选择——如何判断哪些 token 的 KV pairs 重要/不重要；(2) 预算分配——每层/每 head 保留多少 token。Eviction 方法按评分依赖可分为两大类：(a) **Query-Aware**——重要性评分依赖当前 query 信息（如 SnapKV 的 observation window），压缩 cache 对初始 query 过拟合，多查询复用性能退化；(b) **Query-Agnostic**——评分仅依赖 context 自身（如 KVzip 的上下文重建），压缩后的 cache 可跨任意 query 复用。

所有 eviction 在 prefill 阶段完成后、decoding 阶段开始前执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**KV Cache Eviction 通用流程**：

```
# === Prefill 阶段 ===
Q, K, V = X @ W_Q, X @ W_K, X @ W_V
O_full = FlashAttention(Q, K, V)
K_cache = K; V_cache = V

# === Eviction 阶段（prefill 后） ===
importance_scores = compute_importance(Q, K, V)  # 各方法核心差异
budget = allocate_budget(layer_id, total_budget)  # 各方法核心差异
keep_indices = topk(importance_scores, budget)
K_cache = K_cache[keep_indices]
V_cache = V_cache[keep_indices]

# === Decoding 阶段 ===
for each new token:
    Q_new, K_new, V_new = x_new @ W_Q, x_new @ W_K, x_new @ W_V
    O_new = FlashAttention(Q_new, cat([K_cache, K_new]),
                           cat([V_cache, V_new]))
    K_cache.append(K_new); V_cache.append(V_new)
```

**各 Eviction 方法的 Token 重要性评估方式对比**：

| 方法 | 重要性评估 | Query依赖 | 预算分配 |
|------|-----------|----------|---------|
| StreamingLLM | 仅首+尾 token | Query-Agnostic | Uniform |
| H2O | 累积 attention 分数 | Query-Aware | Uniform |
| SnapKV | observation window attention clustering | Query-Aware | Uniform |
| PyramidKV | SnapKV attention clustering | Query-Aware | 金字塔形 |
| KVzip | 上下文重建 cross-attention max score | Query-Agnostic | Non-uniform head-budget |
| CAKE | attention entropy + variance + SnapKV | Query-Aware | attention 统计量 |
| CompressKV | SRH attention aggregation + pooling | Query-Aware | Error-aware (layer-level) |
| CoKV | SnapKV attention pooling | Query-Aware | Head-level Shapley |
| GemFilter | Filter layer last-query-key scores + pooling | Query-Aware | 单索引集（全局 uniform） |
| LOOK-M | 累积 attention + Text-Prior (文本优先) + KV pairs merging | Query-Aware (prefill attention) | Uniform (M+N recent+important) |
| StreamingLLM | Attention sink + sliding window (位置固定) | Query-Agnostic | Uniform |
| LaCache | Ladder-shaped cross-layer pattern (位置固定) | Query-Agnostic | Ladder (跨层错位) |

| TreeKV | 循环淘汰范围内相邻两 token 的平均 attention weight 比较 | Query-Aware (但淘汰范围均匀分布) | Uniform |
| LagKV | Channel-wise std after lag-normalize of K+V | Query-Agnostic | Uniform |
| LOCRET | Trained retaining head MLP predicts CIS | Causal (neither QA nor QG) | Uniform per-head |

注0：LOCRET 的评分既非 Query-Aware 也非 Query-Agnostic——它是 **causal** 的：CIS 仅依赖当前及之前 token 的 [Q, K, V]，不需要任何 query（prefill 阶段）也不需要完整序列。训练时 CIS 基于 answer tokens 的 attention score 为 ground truth，但推理时 retaining head 仅需 local context。LOCRET 是首个将 **trained** eviction scoring 与 chunked prefill 结合的方法，其 retaining head 开销 < 2% inference time。

注1：GemFilter 与传统 eviction 方法有本质区别——它在 prompt computation 阶段仅运行前 r 层处理全部 n 个 token（而非全部 m 层），因此 prompt 计算量从 Θ(mhn²d) 降至 Θ(rhn²d)。其余方法在 prompt computation 阶段仍处理全部 m 层。
注2：KVzip 是首个明确提出 query-agnostic 作为核心贡献的 eviction 方法。其重要性评分通过让 LLM 模拟重建原始上下文（"Repeat the previous context:" prompt + context forward pass），取每个 KV pair 在重建过程中收到的最大 cross-attention score。该评分与下游任务的 attention 模式高度重叠（Figure 6），证明重建关键 KV pairs 对各任务均重要。此外 KVzip 还支持 context-independent 模式：预计算 static head-level score，部署时零开销。

术语一般如何实现？如何使用？

Eviction 方法无需训练或微调，以即插即用方式集成到 HuggingFace Transformers 推理 pipeline——在每层 attention 计算后添加 eviction 步骤。所有 eviction 方法与 FlashAttention 兼容。CompressKV 额外包含一个自定义 CUDA kernel (`adakv`)。KVzip 通过 chunked scoring（m=2K chunk）将评分复杂度从 O(n_c²) 降至 O(m·n_c)，压缩开销约 2x prefill（仅执行一次）。Eviction 的压缩率取决于保留 token 数：如在 128K context 下仅保留 256 tokens（0.07% 容量），NIAH 准确率仍可达 90%（CompressKV 结果）。KVzip 在 30% budget（淘汰 70%）下保持接近无损性能，结合 4-bit 量化可将 16-bit 124K-token cache 从 16.3GB 降至 1.2GB。代码开源：https://github.com/TUDa-HWAI/CompressKV.git 和 https://github.com/snu-mllab/KVzip。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs
- Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction
- Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction
- LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference
- LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models
- LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important
- LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices
- CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression （TriAttention 提出 pre-RoPE eviction：评分不依赖 attention scores 而依赖 Q/K 中心的三角函数级数预测距离偏好 + 自适应范数补充。区分于所有 post-RoPE 方法——TriAttention 回到 pre-RoPE 空间，利用 Q/K Concentration 跨位置稳定性绕过观察窗口限制。校准一次离线完成，推理时无需计算 attention。）
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference （WindowKV 提出 window 级 eviction——将 context 切分为 review windows，以 window 为粒度选择保留，而非逐 token eviction。引入任务自适应分类器决定每窗口内保留的 token 比例，解决 token 级 eviction 破坏语义连贯性的问题。同时提出 intra-group layer KV cache indices sharing 减少 window selection 开销。）

注：LOCRET 引入训练式 causal eviction——不同于所有其他方法依赖启发式统计量（attention weight/entropy/channel std），LOCRET 使用小型训练 MLP（retaining head）预测 Causal Importance Score (CIS)。训练开销 < 1 GPU hour，保留全模型权重冻结。CIS 为每个 (layer, head, token) 三元组独立打分，eviction 跨 head 独立进行。支持 MHA 和 GQA 架构。在 NVIDIA 4090 消费级 GPU 上实现 128K+ 长上下文推理，压缩比 up to 20×（<10% 性能损失），10M token 上下文评估（1747.6× 压缩比）达 100% 准确率。

注：LOOK-M 是首个专门针对多模态长上下文场景的 KV cache eviction 方法。其核心创新在于：(a) Text-Prior——在累积 attention score 基础上为文本 token 显式加 T_p = Max(A_s) 确保文本 KV pair 优先保留；(b) 对被 evicted 的 KV pair 不直接丢弃，而是通过 nearest-neighbor matching + 三种合并策略（averaged/pivotal/weighted）将其信息融入 conserved token。在多模态场景下，该 text-prior + merging 组合在 Needle-in-a-Haystack 任务上显著超越纯文本 eviction 方法（H2O/SnapKV/RoCo）。

注：LaCache 是首个提出跨层异质 KV 存储的 eviction 方法。其两点创新：(a) Ladder-Shaped Pattern——不同层缓存不同位置 token 的 KV cache（浅层存储早期 token、深层存储近期 token），在相同总 budget 下覆盖更长上下文。该 pattern 通过两个超参数控制——Span S（同一 token 被保留的连续层数）和 Overlap O（每层保留的 token 数），且经消融验证位于 PPL-cache size Pareto 最优边界（1500+ 随机 pattern 对比）；(b) Iterative Compaction——ladder pattern 可递归应用于已压缩 cache，实现 O(1) 内存复杂度的无限连续生成。LaCache 的一个重要设计决策是故意不使用 attention maps 进行 token 重要性评估（与 H2O/SnapKV 不同），因此与 FlashAttention 天然兼容，在 H200 实测中取得 score-throughput Pareto 最优。在 PG19 数据集上支持 10M+ token 连续生成（Full KV 在 160K token 即 OOM），NIAH 50% budget 下准确率 99.16% vs StreamingLLM 54.54%。

注：Exploiting Sparsity 论文提供了一种区别于 eviction 的方案——不做 token eviction，而是将完整 KV cache 存放在 CPU 内存中，通过 Faiss ANN search 在每次 decoding step 动态检索 top-k 个最相关的 KV pair 传输到 GPU。该方法在 1M token NIAH 测试中 k=1 即可 100% 成功，而 StreamingLLM（eviction-based）完全失败（被 evict 的 token 无法恢复）。这说明对于某些长上下文任务，per-query sparse retrieval 从根本上优于 static eviction。

注：Quest（ICML 2024）提供了另一种非 eviction 方法——完整保留 KV cache，不做任何 token 驱逐，而是在每步 decode 时基于当前 query 动态选择关键 page 加载到 attention。相比 eviction 方法的核心优势：(a) 不丢失信息——所有 token 始终保留，未来 query 始终可以访问任意 token；(b) query-aware 选择——同一 token 对 query="is" 关键对 query="D" 不关键（Fig. 2），eviction 无法处理这种动态性；(c) Passkey retrieval 证明——H2O/TOVA/StreamingLLM 在 10K-100K 下准确率 0-4%，Quest 64-1024 token budget 即 100%（Tab. 1）。代价是需额外存储 per-page metadata（~12.5% of KV cache）且不减少显存占用（仅减少 memory movement）。

涉及论文标题：
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

---

## Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Grouped-Query Attention (GQA, Ainslie et al., 2023) 和 Multi-Query Attention (MQA, Shazeer, 2019) 是 Transformer decoder 中减少 KV Cache 内存占用的注意力机制变体。在标准 Multi-Head Attention (MHA) 中，每个 query head 拥有独立的 K/V head（即 hq = hkv，如 32 query heads 对应 32 KV heads），每层需要 32 组独立的 K/V 投影矩阵和 KV cache 存储。MQA 将共享推向极致——所有 query heads 共享同一组 K/V head（hkv = 1），每层仅需 2 个 K/V 投影矩阵（而非 MHA 的 2×hq 个），KV cache 内存降至 MHA 的 1/hq。GQA 是折中方案——将 query heads 分为若干组，每组共享一组 K/V head。共享度由 gq = hq/hkv 度量：gq=1 即 MHA，gq>1 为 GQA，hkv=1 (gq=hq) 为 MQA。

现代 LLM 广泛采用 GQA：Llama-3 (hq=32, hkv=8, gq=4)、Qwen3 (hq:hkv=4:1)、Llama-2-70B (hq=64, hkv=8, gq=8)。关键 trade-off：MQA 最大化 memory saving 但约束 attention 表达能力→GQA 在 memory 和 quality 间取平衡。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**GQA Attention Computation（nh=32, nkv=8 为例）**：
```
// 输入: Q ∈ R^{1×32×d_h}, K_cache, V_cache ∈ R^{T×8×d_h}
// gq = 32/8 = 4 query heads per KV head

// Step 1: Query projection (全部 32 个 Q heads)
Q = X @ W_Q  // shape: [1, 32*d_h]

// Step 2: K/V projection — 仅需 8 个 heads
K_new = X @ W_K  // shape: [1, 8*d_h]
V_new = X @ W_V  // shape: [1, 8*d_h]

// Step 3: Grouped attention — 每组 4 Q heads 共享 1 K/V
for kv_idx in range(8):
    q_group = Q[:, kv_idx*4 : (kv_idx+1)*4]  // [1, 4, d_h]
    K = K_cache[:, kv_idx, :]                 // [T, d_h]
    V = V_cache[:, kv_idx, :]                 // [T, d_h]
    // QK^T: 4 个 queries 联合与同一 K 做 GEMM → 更高 arithmetic intensity
    scores = q_group @ K^T / sqrt(d_h)       // [1, 4, T]
    attn = softmax(scores)                    // [1, 4, T]
    out[kv_idx*4:(kv_idx+1)*4] = attn @ V   // [1, 4, d_h]

// Step 4: Output projection
output = concat(all_outputs) @ W_O  // [1, 32*d_h]
```

**KV Cache 内存对比**（BF16，T=128K，L=36，d_h=64）：
| 变体 | hkv | KV Cache 大小 | 相对 MHA |
|------|-----|--------------|---------|
| MHA | 32 | 2×36×128K×64×32 ≈ 18.9GB | 1× |
| GQA (Llama-3) | 8 | 2×36×128K×64×8 ≈ 4.7GB | 0.25× |
| MQA | 1 | 2×36×128K×64×1 ≈ 0.59GB | 0.031× |

**论文发现（Cost-Optimal GQA）**：长上下文下进一步减少 head 数可以显著降低成本。T=128K 时，H=(8,1)（退化为 MQA）比 Llama-3 GQA H=(32,8) 的 KV cache 减少 87.5%，attention FLOPs 减少 75%，同时通过增大模型 N（1.8B vs 1.2B）补偿 loss。

术语一般如何实现？如何使用？

GQA/MQA 在训练阶段通过修改 attention layer 的 K/V projection 实现——将独立的 hq 个 K/V projection 矩阵合并/复用为 hkv 个。从 MHA checkpoint 转换为 GQA 可通过 mean pooling 已有 K/V heads 或 up-training。推理时无需特殊改动（KV cache 自动减少），与 FlashAttention、KV cache quantization、PagedAttention 等正交优化叠加。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

**TransMLA 的理论贡献**：在 Appendix A 中严格证明了 GQA < MLA < MQA 的表达能力层级（相同 KV cache 大小下）。GQA 可表示为 MLA 的特例（W^{UK} 必须是 block-selector 稀疏矩阵，仅能产生 g 个独立 key/value 重复 h/g 次），而 MLA 的 dense W^{UK} 允许任意跨 head 混合，拥有严格更强的表达能力。这为从 GQA 迁移到 MLA 提供了理论基础。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs
- GTA__Grouped-head_latenT_Attention
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation
- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention
- Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression
- TransMLA: Multi-Head Latent Attention Is All You Need
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

**Q-Filters 论文中的 GQA 处理**：Q-Filters 对 GQA 的处理方式为——对每组共享同一 KV head 的 Query heads，将其 Q-Filters（即各 Query head 的 SVD 第一右奇异向量）取平均，得到该 KV head 的统一 Q-Filter。推理时，用该平均 Q-Filter 对共享的 Key 向量做投影评估。论文验证该方法在 Llama-3.1-8B/70B（GQA, gq=4）上有效。

---

## Native Sparse Attention (NSA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Native Sparse Attention (NSA) 是 DeepSeek-AI 提出的一种可原生训练的稀疏注意力机制，通过动态分层稀疏策略替代 Transformer 中标准 Full Attention 的 O(t²) 密集计算。NSA 将每个 query 的 key/value 序列按时间分块（temporal blocks），经三条并行注意力路径处理：(1) **Token Compression（粗粒度压缩）**：通过可学习 MLP φ（含 intra-block position encoding）将连续 key/value block 压缩为块级紧凑表示 $\tilde{K}_t^{\text{cmp}}, \tilde{V}_t^{\text{cmp}}$（block length l=32, stride d=16），捕获全局高层语义，计算成本 O(t·l/d²)≈O(t/16)；(2) **Blockwise Token Selection（细粒度选择）**：利用压缩注意力的中间 softmax 分数 $\mathbf{p}_t^{\text{cmp}}$（免费获得）推导 selection block（l'=64）的重要性分数，经 Top-n（n=16）选出最重要的连续 token block 保留精细信息；(3) **Sliding Window（局部窗口）**：独立 512 token 局部窗口分支隔离局部模式学习，防止 shortcuts 压制全局路径。三条路径输出通过输入依赖的可学习门控 $g_t^c = \text{Sigmoid}(\text{MLP}_g(\mathbf{q}_t))$ 融合，且使用独立 K, V 投影矩阵（共 3 组而非 1 组）防止跨路径梯度干扰。

关键特性：(a) **全生命周期覆盖**——训练/prefilling/decoding 三个阶段均支持稀疏加速，不同于 H2O/Quest 等方法仅支持推理稀疏；(b) **端到端可训练**——所有算子可微，Top-n selection 在 forward 做离散选择、backward 仅对选中 block 的非零 attention 传梯度，形成隐式 straight-through estimation；(c) **硬件对齐**——blockwise 连续内存访问匹配 Tensor Core 需求，GQA group 内跨 head 共享 KV block 选择消除冗余传输。论文报告在 27B GQA+MoE 模型上：pretrain loss 低于 Full Attention，9 个通用 benchmark 中 7 个超越 Full Attention，LongBench 平均分 0.469（超 Full Attention +0.032），64k Needle-in-a-Haystack 完美检索，AIME 推理 +0.054~0.075。Kernel 层面：64k forward 9.0×/backward 6.0× speedup（Triton vs FA2 Triton），解码预期 11.6× speedup。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**NSA 完整 Attention 计算流程（per query token t）**：

```
输入: q_t ∈ R^{d}, K_cache, V_cache ∈ R^{t×d}  (t 为历史序列长度)
超参: l=32 (compression block), d=16 (stride), l'=64 (selection block), n=16, w=512

// === Branch 1: Token Compression ===
num_comp_blocks = floor((t - l) / d) + 1
for i in range(num_comp_blocks):
    K_block = K_cache[i*d : i*d+l]  // [l, d_k]
    K_cmp[i] = phi(K_block)          // [1, d_k], phi 为 MLP + intra-block PE
// 同理得 V_cmp ∈ R^{num_comp_blocks × d_v}
p_cmp = softmax(q_t @ K_cmp^T / sqrt(d_k))         // [1, num_comp_blocks]
o_cmp = p_cmp @ V_cmp                               // [1, d_v]
// ← p_cmp 被免费复用于 Step 2 的 block 重要性推导

// === Branch 2: Blockwise Token Selection ===
// 由 p_cmp 推导 selection block 重要性（共享 blocking scheme 时 p_slc = p_cmp）
p_slc = aggregate(p_cmp, stride=l'/d)               // [1, t/l']
// GQA: 跨 group 内所有 heads 聚合
p_slc_shared = sum_{h=1..H} p_slc^(h)              // [1, t/l']
I_t = topk_indices(p_slc_shared, n)                // 选 n=16 个 block
K_sel = concat([K_cache[i*l' : (i+1)*l'] for i in I_t])  // [nl', d_k]
V_sel = concat([V_cache[i*l' : (i+1)*l'] for i in I_t])
s_sel = q_t @ K_sel^T / sqrt(d_k)                   // [1, nl']
o_sel = softmax(s_sel) @ V_sel                      // [1, d_v]

// === Branch 3: Sliding Window ===
K_win = K_cache[t-w:t]  // [w, d_k]
V_win = V_cache[t-w:t]  // [w, d_v]
o_win = softmax(q_t @ K_win^T / sqrt(d_k)) @ V_win  // [1, d_v]

// === Gated Fusion ===
g_cmp, g_sel, g_win = sigmoid(MLP_gate(q_t))  // 各 ∈ [0,1]
o_t = g_cmp * o_cmp + g_sel * o_sel + g_win * o_win
// 总 KV 访问量 ≈ t/16 + 1024 + 512 ≪ t (长序列)
```

**Prefill 阶段**（训练时并行处理所有 t）：所有 query positions 共享同一套 compression K/V（仅需计算一次），selection block 索引 per-position 不同（需各算各的），window 天然 per-position。压缩和 window 分支复用 FlashAttention-2 kernel，selection 分支用 NSA 专用 group-centric kernel。

**解码阶段**（自回归）：每步只需加载 ~t/16 + nl' + w 个等效 token 量的 KV cache（64k 时 ≈5632 vs Full Attention 65536），memory access 量降 11.6×。

术语一般如何实现？如何使用？

NSA 在 Triton 上实现。Compression attention 和 window attention 直接复用 FlashAttention-2 kernel。Selection attention 使用 NSA 专用 group-centric kernel（详见 kernel调度 分层）。训练时 selection 的 Top-n 操作 forward 做离散 mask（仅计算选中 block 的 attention），backward 时因非零 attention score 梯度自然传播（隐式 STE），无需额外辅助 loss。压缩 MLP φ 与 backbone 联合训练。门控 MLP_g 输出经 sigmoid 约束在 [0,1]。

论文开源情况：DeepSeek-AI 出品，arXiv:2502.11089（ACL 2025 Best Paper），已在 DeepSeek V3.2-Exp 中采用。实现与标准 Transformer 训练流程兼容，可替换现有 attention 层。

涉及论文标题：
- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

---

## Token Compression in Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token Compression in Sparse Attention 是一种将连续 key/value token block 压缩为紧凑表示以减少注意力计算量的技术。不同于 token pruning（直接丢弃低分 token）或 KV cache eviction（推理时选择性淘汰），token compression 通过**可学习的参数化映射**将多个 token 的信息聚合为单一压缩表示，保留高层语义信息的同时大幅降低计算复杂度。

NSA 论文中的具体实现：给定 key 序列 $\mathbf{k}_{:t}$，以 block length l 和 stride d 滑动窗口（d < l 以缓解信息碎片化），每个长度为 l 的 block 经过含 intra-block position encoding 的 MLP φ 映射为单个压缩 key：
$$\tilde{K}_t^{\text{cmp}} = \{\varphi(\mathbf{k}_{id+1:id+l}) \mid 0 \le i \le \lfloor\frac{t-l}{d}\rfloor\}$$
同理生成压缩 value $\tilde{V}_t^{\text{cmp}}$。压缩后的 KV 长度从 t 降至 ~t/d（d=16 时约 t/16）。φ 在训练中与 backbone 联合学习最优压缩策略，不同于固定 pooling（mean/max）或哈希方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Token Compression 模块
// 输入: K_cache ∈ R^{t×d_k}, 超参 l=32, d=16

num_comp_blocks = (t - l) // d + 1
K_cmp = zeros(num_comp_blocks, d_k)
for i in range(num_comp_blocks):
    // 取连续 block: K_cache[i*d : i*d+l] ∈ R^{l×d_k}
    // Step 1: 添加 intra-block position encoding
    K_block_pe = K_cache[i*d:i*d+l] + PE_intra_block  // PE ∈ R^{l×d_k}
    // Step 2: MLP 压缩 (l×d_k → 1×d_k)
    // φ 为 2 层 MLP: Linear(d_k→4d_k) → ReLU → Linear(4d_k→d_k)
    // 先 reshape: [l, d_k] → [l*d_k]
    K_flat = K_block_pe.reshape(l * d_k)
    K_cmp[i] = phi(K_flat)  // [d_k]
// 输出: K_cmp ∈ R^{num_comp_blocks × d_k}

// 压缩 attention 计算:
scores = q_t @ K_cmp^T / sqrt(d_k)  // [1, num_comp_blocks]
attn = softmax(scores)
output = attn @ V_cmp  // V_cmp 同理生成
```

术语一般如何实现？如何使用？

Token compression 通常以两种方式实现：(a) 可学习 MLP-based（NSA 的方法），训练中学习压缩映射，表达能力最强但需额外参数和训练成本；(b) 固定 pooling-based（如 mean pooling over block），零额外参数但压缩质量受限于 pooling 策略。

实际使用中，压缩分支通常与 fine-grained selection 分支配合——压缩负责低成本的全局扫描（粗召回），selection 负责高精度的局部检索（精排序）。这种「粗召回+精排序」的二阶段设计在信息检索领域有广泛先例，NSA 将其内化到单个 attention 层中。

涉及论文标题：
- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

---

## Blockwise Token Selection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Blockwise Token Selection 是一种在稀疏注意力中以**连续 token block 为粒度**选择重要 KV 子集参与注意力计算的方法。与 token-level selection（逐 token 选择，如 HashAttention）相比，blockwise selection 的核心优势在于：(1) 连续内存访问匹配 GPU 的 coalesced HBM 读取模式，blockwise 加载吞吐远高于 scatter/gather 的随机索引读取；(2) blockwise 计算兼容 Tensor Core 的矩阵乘法 tile 要求（16/32/64/128 block sizes）；(3) 注意力分数在空间上往往呈块状聚集（blockwise clustering），相邻 token 重要性相似。

NSA 的 blockwise selection 采用「免费重要性分数」策略：利用 Token Compression 分支中已计算的压缩注意力分数 $\mathbf{p}_t^{\text{cmp}}$ 来推导 selection block 的重要性。当 compression block 与 selection block 共享 blocking scheme 时（l=32, l'=64, d=16 均整除），可通过空间对应关系的加权求和直接得到 selection block 分数。GQA 架构下跨 head 聚合确保 group 内 KV block 选择一致，解码时一次加载服务所有 head。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Blockwise Selection (复用了 Compression 的 p_cmp)
// p_cmp ∈ R^{num_comp_blocks} 已在上一步计算

// Step 1: 映射 compression block 分数到 selection block
// compression: l=32, d=16; selection: l'=64
// 每个 selection block (l'=64) 覆盖 4 个 compression strides
p_slc = zeros(t // l')
for j in range(len(p_slc)):
    start_cmp = (j * l') // d
    // 聚合该 selection block 覆盖的所有 compression block 分数
    for m in range(l' // d):   // l'/d = 4
        for n in range(l // d): // l/d = 2 (块内 stride 覆盖)
            idx = start_cmp - m - n
            if 0 <= idx < len(p_cmp):
                p_slc[j] += p_cmp[idx]

// Step 2: GQA 跨 head 聚合 (H=16 heads per group)
p_slc_shared = sum(p_slc_h for h in range(H))  // [t/l']

// Step 3: Top-n 选择 (n=16, 含 1 个初始块 + 2 个局部块)
I_t = topk_indices(p_slc_shared, n=16)  // sorted by importance
// 将连续 token block 拼接到 K_sel, V_sel
K_sel = concat(K_cache[i*l' : (i+1)*l'] for i in sorted(I_t))  // [nl', d_k]
V_sel = concat(V_cache[i*l' : (i+1)*l'] for i in sorted(I_t))

// Step 4: 精细 attention（仅对选中 block）
scores = q_t @ K_sel^T / sqrt(d_k)  // [1, nl'], nl'=1024
output = softmax(scores) @ V_sel
```

术语一般如何实现？如何使用？

Blockwise selection 的重要性分数计算有三种典型方式：(a) NSA 的「免费复用」——利用已有 compression attention 分数的空间聚合，零额外计算开销，可端到端训练；(b) 辅助 loss-based（如 SeerAttention）——训练单独的 block 重要性预测网络，用 KL 散度监督，增加额外参数和训练复杂度；(c) Heuristic-based（如 Quest 的 min-max chunk product）——无参数启发式计算，无需训练但召回率较低。

实现要点：selection block size l' 必须是 Tensor Core tile size 的倍数（通常 64 或 128）。GQA/MQA 场景必须跨 head 聚合分数再统一选择，否则每个 head 独立选择导致 KV block 加载并集远大于交集。Top-n 选择数 n 体现 sparsity-quality trade-off（NSA n=16 @ l'=64 → 1024 tokens，平均约 2560 tokens 含压缩和窗口）。

涉及论文标题：
- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

---

## Query-Group Compression (GQA-aware KV Cache Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Query-Group Compression 是 KV-Compress 提出的针对 GQA 模型的 KV cache 压缩策略。传统 KV cache 压缩方法（SnapKV、PyramidKV、H2O 等）在 GQA 模型上先将 KV cache repeat 到 query head 数量（例如从 8 KV heads repeat 到 32 query heads），再在重复后的 cache 上执行压缩。这导致：(1) cache 中 3/4（r-1/r）的 KVs 是重复数据；(2) 压缩率需超过 query-to-KV-head ratio r 才能带来超过 GQA 本身的额外压缩效果。

Query-Group Compression 直接在非 repeat 的 GQA KV cache（shape H_kv × L × d）上执行压缩。关键修改是将 eviction metric 的聚合范围改为每个 key 所属的 query group 内的所有 queries：$M_{h_k,j} = \sum_{h \in H_k} \sum_i (A_{h,i,j})^2$，其中 $H_k = \{h: r \cdot h_k \le h < r \cdot (h_k + 1)\}$。这对应 Equation 9-10 的推广。

对于 Llama-3/Mistral 模型 (r=4)，同样 max-cache-size C 下，KV-Compress 实际持有 1/4 的 KVs，相当于 4x 额外有效压缩率。

从算法pipeline角度拆解术语：

**GQA Query-Group Compression 的计算流程**：
```
输入：GQA model with H_kv KV heads, r query heads per KV head
参数：observation window w, pooling size p（或 excluded query window v for full mode）

for each layer and each KV head h_k in 1..H_kv:
    # 定义该 KV head 对应的 query group
    H_k = {h : r*h_k <= h < r*(h_k + 1)}  # r 个 query heads

    # 聚合该 group 内所有 query heads 的 squared attention
    for each query head h in H_k:
        for each query i in observation range:
            for each key j in causal range (j <= i):
                M_{h_k, j} += (A_{h, i, j})^2

    # 可选 max-pooling（KVC-w 变体）
    M_{h_k, j} = max_{t in [j-p/2, j+p/2]} M_{h_k, t}

# 在非 repeat 的 KV cache 上执行 eviction：
# 排序 M_{h_k, :} → 选择 top-C KVs per head → 释放其余
```

**与 baseline 方法的区别**：
```
# Baseline (SnapKV/PyramidKV naive GQA):
K_cache_raw = K_cache[:, :H_kv, :]          # [L, 8, d]
K_cache_repeat = repeat(K_cache_raw, r)       # [L, 32, d] — 3/4 duplicates!
metrics = compute_attention_scores(K_cache_repeat)  # on repeated cache
evict_KVs(K_cache_repeat, metrics)            # compresses repeated data

# Query-Group Compression (KV-Compress):
K_cache_raw = K_cache[:, :H_kv, :]          # [L, 8, d]
for h_kv in 0..H_kv-1:
    metrics[h_kv] = sum_{h in H_kv_group} attention_scores[h]
evict_KVs(K_cache_raw, metrics)              # compresses non-repeated data
# Same max-cache-size C: 4x fewer KVs stored
```

术语一般如何实现？如何使用？

实现方式：(1) 在 prefill 阶段计算完整 attention 后，按 query group 聚合 attention scores；(2) 聚合操作可以累积到 per-KV-head 的 metric tensor 中（而非 repeat 后的 per-query-head tensor）；(3) 后续 eviction 操作基于 M_{h_k, j} 在非 repeat cache 上执行。该方法与 FlashAttention 兼容——attention scores 在 FA 计算中获取（eager mode）或从 observation window queries 单独计算。

适用于所有 GQA 模型（Llama-3, Mistral, Qwen 等）的 KV cache 压缩场景。r 越大（query heads per KV head 越多），query-group compression 相对于 baseline repeat+compress 的优势越显著。在 KV-Compress 中，LongBench C=128 下以 1/4 KVs 达到 state-of-the-art（Mistral-7B: 37.64 vs Ada-SnapKV 36.71; Llama-3.1-8B: 46.26）。

涉及论文标题：
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head

---

## Variable-Head-Rate KV Eviction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Variable-Head-Rate KV Eviction 是一种 KV cache 压缩策略，允许每个 attention head 以不同的压缩率（eviction rate）保留不同数量的 KVs，而非所有 head 统一 evict 相同数量的 KVs。该概念由 Ada-SnapKV (Feng et al., 2024) 首次从算法角度提出——按 per-head eviction metric 跨 head 进行 eviction 选择（cross-head eviction），在 LongBench 上证明了可变 per-head 压缩率相比 uniform compression 的精度优势。

然而，Ada-SnapKV 的方案在现有推理框架中只能增加 cache 碎片化而无法实际减少物理内存占用：因为现有 PagedAttention 中所有 heads 的 KVs 打包在同一 cache block 中，evict 一个 head 的 KVs 而不 evict 其他 heads 的 KVs 不释放整个 block。

KV-Compress 通过修改 PagedAttention 的 block 布局（per-head per-layer 独立 block）使 variable-head-rate eviction 可以实际释放物理内存——每个 head 有独立的 block table，evict 某 head 的 block 可直接释放该 block 的物理内存。同时扩展到 variable per-layer rate。

从算法pipeline角度拆解术语：

**Variable-Head-Rate Eviction 的 eviction 选择过程**：
```
输入：M ∈ R^{H_kv × L}（per-head per-token eviction metrics）
参数：target total KVs after compression T（等价于 max-cache-size C）

# Step 1: 跨 head 展平 metrics
M_flat = M.reshape(-1)  # [H_kv * L]
# 每个元素对应一个特定 (head, token_position) 的 KV

# Step 2: 全局排序（variable-head-rate 的关键）
sorted_idx = argsort(M_flat)  # 按 metric 升序排列
evict_idx = sorted_idx[:H_kv*L - T]  # 最低 metric 的 KVs

# Step 3: per-head eviction count 自动由全局排序决定
for h in 0..H_kv-1:
    evict_count[h] = count(evict_idx where head == h)
    # 高 attention 集中 head → 少 evict → 保留更多 KVs
    # 低 attention 分散 head → 多 evict → 保留更少 KVs
```

**Uniform vs Variable-Head-Rate 对比**：
```
# Uniform rate (H2O, SnapKV, PyramidKV):
for h in 0..H_kv-1:
    evict_count[h] = E / H_kv  # 所有 head 相同

# Variable-head-rate (Ada-SnapKV, KV-Compress):
# evict_count 由全局排序自动分配
# 高 attention 集中的 head 自然保留更多 KVs
```

术语一般如何实现？如何使用？

在 KV-Compress 中，variable-head-rate eviction 通过两步排序实现：(1) 先在每个 head 内排序 metrics 获得 per-head per-eviction-block 的最大 metric；(2) 再跨 head 排序候选 block evictions。最终按总 evict block 预算 E_s 选择跨 head 的 block eviction 方案。跨 head 的 eviction 分配完全由 metric 值驱动，不需要手动设定 per-head 预算。

适用场景：需要高压缩率的长上下文推理场景。当不同 attention head 的功能差异显著时（一些 head 关注全局语义检索，一些关注局部 token 关系），variable-head-rate 相比 uniform 的优势更大。KV-Compress 在 Llama-3.1-8B LongBench 上以 1/4 KVs 超越所有 uniform-rate baselines。

涉及论文标题：
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head

---

## Squared Attention Metric (L2 Eviction Metric)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Squared Attention Metric 是 KV-Compress 提出的 KV eviction metric 计算方法，使用 attention scores 的平方和 $\sum (A_{h,i,j})^2$ 替代传统的 attention scores 直接求和 $\sum A_{h,i,j}$（L1 aggregation）。前者等价于最小化未来 attention 的 L2 误差，后者等价于最小化 L1 误差。

数学上：对于 key j 和 query i，标准 attention weight $A_{h,i,j} = \text{softmax}(Q_i K_j^T / \sqrt{d})$。L1 metric $M_{h,j}^{(L1)} = \sum_i A_{h,i,j}$。L2 metric $M_{h,j}^{(L2)} = \sum_i (A_{h,i,j})^2$。L2 metric 对高 attention 的 key 更敏感（平方惩罚放大差异），使得 eviction 更倾向于保留高 attention 的 KVs 并更激进地舍弃低 attention 的 KVs。

KV-Compress 实验验证 L2 在所有变体（KVC-w, KVC-full）、所有 max-cache-size（C=128/256/512/1024）和两个模型（Mistral-7B, Llama-3.1-8B）上一致优于 L1。

从算法pipeline角度拆解术语：

```
# L1 vs L2 对比计算
# 假设 attention weight distribution: [0.5, 0.3, 0.15, 0.04, 0.01]

# L1 (standard):
M1 = [0.5, 0.3, 0.15, 0.04, 0.01]  # 直接求和，差异小
# 如果 evict 最后两个: 丢失 0.05 attention mass → L1 error ≤ 0.05

# L2 (squared):
M2 = [0.25, 0.09, 0.0225, 0.0016, 0.0001]  # 平方求和，差异扩大
# 高 attention keys 的 metric 被放大 (0.25 vs 0.09, gap 0.16)
# 低 attention keys 的 metric 被压缩 (0.0016 vs 0.0001, gap 0.0015)
# 排序更确定，eviction 决策更准确

# KVC-w8-L2 computation (Equation with squared attention):
for h in H_k:  # query heads in key's query group
    for i = L-w to L:  # observation window
        for j = 1 to i:  # causal key range
            M_{h_k, j} += (A_{h, i, j})^2  # squared!
# then optional max-pooling
```

术语一般如何实现？如何使用？

实现简单：在 attention 计算的 metric 累积步骤中，将 `M += A` 改为 `M += A*A`（逐元素平方）。不影响其他算法组件（sort, top-k, block eviction selection）。与 GQA query-group aggregation 和 continual compression 正交叠加。

适用场景：所有基于 attention score 聚合的 KV eviction 方法通用。KV-Compress 中默认使用 L2（论文中标为 KVC-w8-L2），除非显式标注 L1（KVC-w8-L1）。

涉及论文标题：
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head

---

## Continual KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Continual KV Cache Compression 是在 LLM 推理的 decoding 阶段持续执行 KV cache 压缩的策略。传统方法仅在 prefill 完成后压缩一次，后续 decoding 生成的 KVs 不再压缩。Continual compression 在每个（或每隔若干）decoding step 后，累积新生成 token 的 attention 到已有 eviction metrics 中，当需要（如 preemption 即将发生）时基于更新后的 metrics 再次压缩。

数学上（KV-Compress Equation 20）：$M_{h_k,j}^{(cc)} = M_{h_k,j}^{(pool)} + \sum_{i=L_c}^{L_c+t} \sum_{h \in H_k} (A_{h,i,j})^2$，其中 $M^{(pool)}$ 为 prefill 阶段计算的初始 metric，$L_c$ 为 input context 长度，$t$ 为当前 decoding step。每次 decoding step 后，新 query 的 squared attention 被累积到对应 key 的 metric 中。

从算法pipeline角度拆解术语：

```
# Continual Compression 流程
输入：prefill 后的初始 metrics M_init, initial compressed KV cache
参数：compression rate r, block size b

for each decoding step t:
    # Step 1: 正常 decode
    Q = compute_query(x_t)
    A = attention(Q, K_cache, V_cache)  # 计算 attention
    next_token = sample(A @ V_cache)

    # Step 2: 累积新 attention 到 metrics
    for each key head h in H_kv:
        for each query head h_q in group H_k:
            for each cached key j:
                M[h, j] += (A[h_q, new_query, j])^2  # accumulate

    # Step 3: 检查是否触发重压缩
    if would_need_preemption() or compression_interval_reached():
        # 基于更新后 metrics 重新选择 eviction
        sort(M) → select E_s blocks to evict
        MoveCache(K, V, M) → free blocks
        # 重压缩可能 evict 不同 KVs（早期高 attention 后期低 attention 的 key 可能被新选中 evict）

    # Step 4: 追加当前 token KVs 到 cache
    append(K_new, V_new, to_cache)
```

术语一般如何实现？如何使用？

KV-Compress 中 continual compression 的触发条件是：(a) prefill 后立即压缩；(b) 当 preemption 即将发生时压缩（即 free blocks 不足时触发压缩以释放空间）。不使用固定间隔压缩（方案 1 和 2 被测试但不如方案 3+4 有效）。

适用场景：长文本生成（long output）或 high-concurrency serving 场景——decoding 过程中积累的 KVs 若不持续压缩，可能导致后期 KV cache 膨胀并触发 preemption。Continual compression 确保 KV cache 在 decoding 全程保持在压缩后的大小。

涉及论文标题：
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head

---

## Observation Window in KV Cache Eviction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Observation Window 是 KV cache eviction 方法中用于计算 eviction metrics 的 queries 范围。在 LLM 推理中，prefill 阶段处理完整 input prompt 后生成 KV cache。为了决定 evict 哪些 KVs，需要评估每个 KV 的"重要性"——这通过聚合该 KV 收到的 attention 来实现。Observation Window 限制了聚合哪些 queries 的 attention：是聚合全部 queries（从第一个到最后一个 prompt token），还是仅聚合最后的 w 个 queries。

两种主要策略：(1) **Full observation**（H2O, KVC-full）：聚合所有过去 queries，$\sum_{i=1}^L A_{h,i,j}$，O(L²) 计算复杂度；(2) **Limited observation window**（SnapKV, KVC-w）：仅聚合最后 w 个 queries，$\sum_{i=L-w}^L A_{h,i,j}$，O(L) 计算复杂度，且可避免写完整 attention matrix 到 global memory（兼容 FlashAttention）。

KV-Compress 设计了两个变体——KVC-full（全部 queries，排除 local window v=10）和 KVC-w（window w=8 + max-pooling p=7），发现：(1) KVC-full 在多数 subtask 上表现最好，但计算开销大（quadratic scaling），且在某些任务（SAMSum）上严重退化；(2) KVC-w-8 在整体上是更实用的选择。

从算法pipeline角度拆解术语：

```
# Full observation (H2O-style, KVC-full):
for query i in 1..L:
    for key j in 1..i:  # causal
        M[j] += A[i,j]  # 所有 queries

# KVC-full with excluded queries (Equation 19):
for query i in (j+v)..L:  # skip v local queries after key j
    M[j] += (A[i,j])^2

# Limited observation window (SnapKV-style, KVC-w):
for query i in (L-w)..L:  # only last w queries
    for key j in 1..i:
        M[j] += A[i,j]
# then max-pool (window along key dim, size p)

# KV-Compress KVC-w-8 (Equation 10 with squared attention):
for query i in L-8..L:
    for key j in 1..i:
        M[j] += (A[i,j])^2
M = max_pool(M, kernel_size=7)
```

术语一般如何实现？如何使用？

Observation window 策略的选择影响三个维度：(a) 计算开销——full observation O(L²) vs limited O(L)；(b) 指标质量——full observation 理论上信息更全，但近期 queries 的 attention pattern 可能更好预测 decoding 阶段的 attention；(c) 与 FlashAttention 的兼容性——limited window 可避免物化完整 attention matrix。

KV-Compress 实验表明：w=8 优于 w=32（Mistral 实验），较小的 window 更聚焦于与 decoding 行为相似的结尾 queries。max-pooling（p=7）沿 key 维度平滑 metric，保留 heavy-hitter 附近的 context KVs。

适用场景：所有基于 attention score 聚合的 KV eviction 方法。对于 prompt 结构为 "长文档 + 短问题" 的场景，结尾 queries（问题区域的 attention）能很好预测 decoding 阶段的 attention pattern——此时 limited window 有效。对于需要全局信息检索的任务，KVC-full 可能更优，但计算开销大。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation
- KV-Compress__Paged_KV-Cache_Compression_with_Variable_Compression_Rates_per_Attention_Head
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression （TriAttention 批判了 post-RoPE 下的 Observation Window 的根本局限：因 RoPE 旋转使 query 朝向随位置变化，仅最近约 25 个 query 有效，窗口无法通过增大改善。TriAttention 回到 pre-RoPE 空间，利用 Q/K 聚集（Q/K Concentration）和三角函数级数预测 attention，完全绕过观察窗口）

---

## Time-Invariant vs Time-Variant Inference Costs（时间无关 vs 时间相关的推理成本）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Time-Invariant vs Time-Variant Inference Costs 是本文提出的 GQA Transformer 推理成本分解框架，将每 token 的推理 FLOPs 和 Memory 按是否随上下文长度 T 增长分为两类：

- **Time-Invariant Costs（时间无关成本）**：与 T 无关的固定开销。FLOPs 方面：模型参数相关的线性投影（QKVO projection + FFN），C_const = 2N（N 为参数量）。Memory 方面：存储模型参数本身，M_const = N。
- **Time-Variant Costs（时间相关成本）**：随 T 线性增长的开销。FLOPs 方面：attention softmax 计算 C_att(T) = 4TL·d_h·n_h。Memory 方面：KV cache N_kv(T) = 2TL·d_h·n_kv。

核心洞察：T 很大时（如 128K），time-variant costs 主导。例如 1.2B 模型在 128K 下，~90% memory 被 KV cache 占用，仅有 ~10% 用于模型参数。因此长上下文下应通过减少 n_h 和 n_kv（降低 time-variant cost）而非减少 N（降低 time-invariant cost）来优化。

从算法pipeline角度拆解术语：

**成本分解公式**：
```
C_infer(T) = 2N               + 4TL·d_h·n_h
           = 时间无关 FLOPs     + 时间相关 FLOPs (attention softmax)

M_infer(T) = N                + 2TL·d_h·n_kv
           = 时间无关 Memory    + 时间相关 Memory (KV cache)
```

**长上下文下的资源分配优化**：
```
若 T=128K, d_h=64, L=36:
  C_var = 4 × 128K × 36 × 64 × n_h = 1.18G × n_h FLOPs/token
  C_const = 2N

若 N=1.2B, C_const = 2.4G
  当 n_h=32: C_var = 37.8G → attention 占 94% FLOPs
  当 n_h=8:  C_var = 9.4G  → attention 占 80% FLOPs
  → 减少 n_h 大幅节省 FLOPs，而适度增加 N（补偿 loss）仅小幅增加 C_const

若 n_kv=8:  M_var = 2 × 128K × 36 × 64 × 8 = 4.7B floats = 9.4GB (BF16)
若 n_kv=1:  M_var = 2 × 128K × 36 × 64 × 1 = 0.59B floats = 1.18GB (BF16)
  → 减少 n_kv 大幅节省 Memory
```

术语一般如何实现？如何使用？

该框架用于指导 cost-optimal GQA 配置搜索：通过解耦 n_h 与 d（Change 1）自由控制 time-variant FLOPs，通过联合优化 N 与 (n_h, n_kv)（Change 2）平衡 time-invariant 与 time-variant 资源分配。通过三步搜索找到给定 (T, L*) 下硬件感知成本 Z 最小的配置。实验验证该框架在 T=128K 时可节省 >50% memory 和 FLOPs vs Llama-3 GQA，无 loss 损失。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs

---

## Cost-Optimal GQA Configuration Search（成本最优GQA配置搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cost-Optimal GQA Configuration Search 是本文提出的三步搜索过程，用于在给定目标 loss L* 和推理上下文长度 T 的条件下，找到推理成本最低的 GQA 配置 (n_h, n_kv, N)。不同于传统方法将 n_h 由 d 唯一确定、n_kv 固定为 8、N 独立选择，该方法联合搜索这三个超参数以最小化硬件感知推理成本。

从算法pipeline角度拆解术语：

**三步搜索过程**：

```
// Step 1: Candidate Selection
// 定义候选 GQA 配置集
max_d = 1536  // 最大模型的 hidden size
H_cand = []
for nh in {1, 2, 4, 8, 16, 32}:
    for nkv in {1, 2, 4, 8, 16, 32}:
        if nkv <= nh:
            H_cand.append((nh, nkv))
// |H_cand| = 21（max(d)/d_h = 32, d_h=64）

// Step 2: Scaling Curves Fitting（T=8K 短上下文训练）
for each H in H_cand:
    for N in [3M, 19M, 85M, 150M, 200M, 470M, 680M, 1.2B]:
        model = build(N, H.nh, H.nkv)
        loss = train(model, SlimPajama, ratio=20:1 tokens/param)
    // 拟合 power-plus-constant 函数
    L(N; H) = (a_H / N)^{b_H} + E
    // R² > 0.999
    // E 为语言自然熵，跨配置共享

// Step 3: Cost Minimization
Input: target loss L*, context length T
for each H in H_cand:
    // 从 scaling curve 反求满足 L* 的最小 N
    N*(H) = a_H / (L* - E)^{1/b_H}
    // 计算推理成本
    C_infer = 2N* + 4TL·d_h·H.nh
    M_infer = N* + 2TL·d_h·H.nkv
    // 硬件感知综合成本 (λ=0.9 偏重 memory)
    Z(H) = 0.9 · M_infer^{1/2} + 0.1 · C_infer^{1/3}

H* = argmin Z(H)
return (N*(H*), H*.nh, H*.nkv)
// N* 为连续值，通过线性插值确定 (L,d)，实际部署取最接近整数配置
```

**为什么可以用 T=8K 外推至 T=128K**：
实验验证 T 对 loss 的影响与 N 和 H 相独立（Section 5.7）——相对 loss 差异 ΔL(T) 在 T>8K 后波动 <1%。因此 Step 2 仅需在 T=8K 下训练小模型，Step 3 将 T 代入成本公式即可外推。

**核心发现**：
- 长上下文下应使用更少的 head + 更大的模型（T=128K, L*=2.615 → H*=(8,1), N*=1.8B）
- Llama-3 GQA (d/dh, 8) 仅对特定 (L*, T) 最优，多数情况下 suboptimal
- n_h 比 n_kv 对 loss 更重要（相同参数增量下 n_h 增加带来更大 loss 降低）
- 对齐训练 FLOPs 时，用更少 head 可获更多训练数据，优势更大

术语一般如何实现？如何使用？

实际部署步骤：(1) 离线运行 Step 1-3 获得 (N*, n_h*, n_kv*)；(2) 选择与 N* 最接近的实际配置（通过 Table 7 的预定义 aspect ratio 插值）；(3) 用该配置从头训练模型（或从已有模型 up-training）。该方法与现有 serving 系统完全兼容——仅改变模型配置，无需修改框架或 kernel。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs

---

## Hardware-Aware Cost Function（硬件感知成本函数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hardware-Aware Cost Function（Z）是本文定义的统一推理成本指标，将 memory（M_infer）和 computational（C_infer）两种不同量纲的成本合并为单一数值，用于比较不同 GQA 配置的成本效率。公式为：

$$Z = \lambda \cdot M_{\text{infer}}^{\alpha} + (1-\lambda) \cdot C_{\text{infer}}^{\beta}$$

其中 λ 控制 memory vs compute 的相对重要性，α 和 β 控制成本增长的非线性惩罚。默认参数 λ=0.9, α=1/2, β=1/3 由作者环境的硬件利用率测试确定，反映 memory 通常是长上下文推理的主要瓶颈（偏重 memory）。

从算法pipeline角度拆解术语：

**参数含义**：
- λ=0.9：memory 占 90% 权重，反映长上下文下 memory bandwidth 为主要瓶颈
- α=1/2：memory 成本以平方根增长——边际成本递减（DDR 带宽利用率在大 memory footprint 时更高效）
- β=1/3：compute 成本以立方根增长——计算可被 Tensor Cores 更好地并行化
- λ=0 最小化纯 FLOPs；λ=1 最小化纯 memory

**在 Cost-Optimal GQA Search 中的使用**：
```
for each candidate H=(nh, nkv):
    N* = solve L(N;H) = L*
    C = 2N* + 4TL·d_h·nh
    M = N* + 2TL·d_h·nkv
    Z = 0.9 * sqrt(M) + 0.1 * cbrt(C)
H* = argmin Z
```

术语一般如何实现？如何使用？

参数 (λ, α, β) 可根据具体部署硬件调整——memory bandwidth 瓶颈更严重的硬件（如 edge devices）应增大 λ；compute-bound 场景（如 prefill 为主的 serving）可减小 λ。通过调整 λ 可实现 Pareto-optimal 的 memory-compute tradeoff。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs

---

## Power-Plus-Constant Scaling Law for GQA（GQA的幂加常数缩放定律）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Power-Plus-Constant Scaling Law 是本文发现的经验规律：(1) LM loss 与模型大小 N 之间满足 L(N) = (a/N)^b + E（用于 Step 2 拟合 scaling curves，R² > 0.999）；(2) LM loss 与 query head 数 n_h 之间也满足 L(n_h) = a·n_h^b + c（附录 I，R² > 0.999）。

关键特性：
- E/c 为"语言自然熵"——当 N→∞ 或 n_h→∞ 时 loss 收敛到的下界。在 470M 模型上拟合 c=1.53
- b < 0 → loss 随 N 或 n_h 增加而下降，但呈 diminishing returns
- n_h 的 power law 与 model size 和 context length 独立（在不同 N 和 T 下均成立）

从算法pipeline角度拆解术语：

**Step 2 Scaling Curve 计算**：
```
// 对每个 H=(nh, nkv)，训练 3M→1.2B 模型，拟合：
L(N) = (a/non_emb_params)^b + E

// 具体拟合值（以 H=(32,8) 为例）：
L(N) = (1.2×10^8 / N)^{0.12} + 2.615
// N=1.2B → L=2.615（与 Llama-3.2-1B 一致）

// 从 scaling curve 反求 N*:
N*(H) = a_H / (L* - E)^{1/b_H}
```

**n_h Scaling Law（附录 I）**：
```
// 470M model, T=1K:
L(n_h) = 0.579 · n_h^{-0.124} + 2.473    (R²>0.999)
// 680M model, T=1K:
L(n_h) = 0.398 · n_h^{-0.177} + 2.583    (R²>0.999)
// 1.2B model, T=1K:
L(n_h) = 0.301 · n_h^{-0.227} + 2.622    (R²>0.999)

// 不同 context length (470M):
L(n_h, T=1K)  = 1.513 · n_h^{-0.039} + 1.53
L(n_h, T=2K)  = 1.436 · n_h^{-0.041} + 1.53
L(n_h, T=8K)  = 1.356 · n_h^{-0.044} + 1.53
// 随 T 增大，c 收敛到相同值（1.53）→ n_h→∞ 时 loss 由 E 决定
```

术语一般如何实现？如何使用？

在 Step 2 中，对每个候选 GQA 配置 H 用 5-8 个不同 N 的训练数据进行非线性最小二乘拟合，获得 (a_H, b_H, E)。虽然 E 理论上由数据决定（跨 H 共享），但实践中为每个 H 独立拟合 E 可提升精度（因小模型下数据与模型间可能有交互）。该 law 经验上可外推至少一个数量级（如 3M→1.2B），类似 Llama-3 从 16B 预测 405B loss 的做法。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs

---

## WSD Learning Rate Scheduler（Warmup-Stable-Decay 学习率调度器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

WSD (Warmup-Stable-Decay) 学习率调度器是 MiniCPM (Hu et al., 2024) 提出的三阶段学习率策略：**Warmup 阶段**（前 10% steps，LR 从 0 线性增至 max_lr）→ **Stable 阶段**（中间 ~70% steps，LR 保持恒定为 max_lr）→ **Decay 阶段**（后 20% steps，LR 通过 cosine annealing 衰减至 0.1×max_lr）。与 cosine decay（全程衰减）相比，WSD 在 stable 阶段允许模型在最高学习率下持续学习，有利于 LLM 预训练中的充分优化。

从算法pipeline角度拆解术语：

```
// WSD Scheduler 伪代码
max_lr = sweep({1,2,5}×10^{-3,-4,-5}) on MHA baseline
total_steps = D_train / (batch_tokens)  // e.g. 20B / 512K
warmup_steps = 0.10 × total_steps
decay_steps  = 0.20 × total_steps
stable_steps = total_steps - warmup_steps - decay_steps

for step in range(total_steps):
    if step < warmup_steps:
        lr = max_lr × (step / warmup_steps)         // linear warmup
    elif step < warmup_steps + stable_steps:
        lr = max_lr                                   // stable plateau
    else:
        progress = (step - warmup_steps - stable_steps) / decay_steps
        lr = min_lr + 0.5 × (max_lr - min_lr) × (1 + cos(π × progress))
        // cosine decay to min_lr = 0.1 × max_lr
```

术语一般如何实现？如何使用？

本文在 GQA scaling experiments 中使用 WSD：10% warmup, 20% decay, 搭配 AdamW (β1=0.9, β2=0.95, weight_decay=0.1, grad_clip=1.0)。max_lr 对每个模型大小在 MHA baseline 上 grid search {1,2,5}×10^{-3,-4,-5} 获得，跨 GQA 配置复用。长上下文 adaption 阶段（T=4K→128K）使用更低的 max_lr=1e-5 + 新 optimizer state 防止 catastrophic forgetting。WSD 在 20:1 的 Chinchilla-optimal 数据比例下运行。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs

---

## Cross-Self Pruning (CSP) / Cross-Self Attention Decomposition for VLM KV Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cross-Self Pruning (CSP) 是一种 training-free 的 KV Cache 剪枝方法，专为多模态视觉语言模型（VLM）设计。核心创新是将原始注意力矩阵 A ∈ [0,1]^{L×L}（L = L_t + L_v，文本+视觉 token 长度）分解为四个子区域：(a) A^{st} ∈ [0,1]^{L_t×L_t}：文本→文本 self-attention；(b) A^{sv} ∈ [0,1]^{L_v×L_v}：视觉→视觉 self-attention；(c) A^{ct} ∈ [0,1]^{L_v×L_t}：视觉→文本 cross-attention；(d) A^{cv} ∈ [0,1]^{L_t×L_v}：文本→视觉 cross-attention。然后在 intra-modality（A^s = Σ_query A^{st} ⊕ Σ_query A^{sv}）和 inter-modality（A^c = Σ_query A^{ct} ⊕ Σ_query A^{cv}）两个维度上独立进行 top-K 选择，分别得到 binary mask M^s 和 M^c。最终保留的 token 必须在两个维度上都被判定为重要：M = M^s ∧ M^c（取交集）。

该设计解决了现有多模态 KV Cache 剪枝方法的核心缺陷：文本 token 的 self-attention scores 通常大于视觉 token，统一对待会导致关键视觉 token 被过度剪枝，破坏跨模态交互。通过独立评估 intra- 和 inter- 两个维度，CSP 确保视觉 token 即使在 self-attention 中得分较低，若在 cross-attention 中被文本 token 高度关注（说明跨模态信息重要），仍会被保留。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**CSP 在 VLM 推理中的伪代码流程**：

```
// 超参数：budget T, 最近窗口 R, 观察窗口 O
for each decoding iteration:
    if L_k < T: return K, V
    A = n-Softmax(O[-O:, :-R])      // n=1 平滑恢复
    A^{st}, A^{sv}, A^{ct}, A^{cv} = decompose(A)
    A^s = sum(A^{st}, axis=q) ⊕ sum(A^{sv}, axis=q)
    A^c = sum(A^{ct}, axis=q) ⊕ sum(A^{cv}, axis=q)
    M^s = TopK(A^s, K^s); M^c = TopK(A^c, K^c)
    M = M^s ∧ M^c                  // 双维度交集
    K = (K ⊙ M) ⊕ K[-R:]; V = (V ⊙ M) ⊕ V[-R:]
```

术语一般如何实现？如何使用？

CSP 以即插即用方式集成到 LLaVA 等 VLM 推理流程，仅修改 Attention 层的 token selection。默认 n=1, cross_ratio=0.5。在 MileBench 上 LLaVA-v1.5-13b 的 IR +9.6%、T-3 +8.3%。代码开源：https://github.com/TerryPei/CSP。

涉及论文标题：
- Cross-Self KV Cache Pruning for Efficient Vision-Language Inference

---

## n-Softmax (Smoothness Recovery for Pruned Attention Distribution)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

n-Softmax 是 CSP 提出的用于恢复 KV Cache 剪枝后注意力分布平滑性的技术。剪枝后 softmax 的分母从 Σ_{j∈I^+ ∪ I^-} e^{O_j} 变为 Σ_{j∈I^+} e^{O_j}，导致注意力分数被放大、分布变尖锐。n-Softmax 在分母中引入偏置 n：A_i = e^{O_i} / (n + Σ_{j∈I^+} e^{O_j})，相当于添加"虚拟 token"的贡献来模拟被剪枝 token 的归一化效应，恢复原始平滑性。论文固定 n=1。与标准 softmax（Σ A_i = 1）不同，n-Softmax 是放松归一化（Σ A_i < 1），额外的概率质量被 n 吸收。在 CSP 算法中，n-Softmax 作为 attention score 计算的第一步，为后续的 Cross-Self 分解和 top-K 选择提供更好的分数基础。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 标准 softmax（剪枝前）
A_i = e^{O_i} / Σ_{j=1}^{L} e^{O_j}          // 完整分母，平滑分布

// 标准 softmax（剪枝后 — 问题所在）
A_i = e^{O_i} / Σ_{j∈I^+} e^{O_j}            // 分母变小 → A_i 变大 → 分布尖锐

// n-Softmax（解决方案）
A_i = e^{O_i} / (n + Σ_{j∈I^+} e^{O_j})      // n=1, 恢复平滑性

// CSP 整体流程中的使用
A = n-Softmax(Q @ K^T / sqrt(d))             // attention logits → smoothed weights
M^s, M^c = CrossSelfDecomposeAndSelect(A)     // 后续 Cross-Self 分解
```

**消融效果**：在 ALFRED 数据集上，n-Softmax 配合 Cross-Self 分解带来一致且轻微的性能提升，在需要时间连贯性和细粒度特征保留的任务上尤为有效。

术语一般如何实现？如何使用？

实现仅需在 softmax 分母中加 n（一行代码修改）。n=1 在所有实验中使用。不独立使用，必须与 Cross-Self Attention Decomposition 组合。代码开源：https://github.com/TerryPei/CSP。

涉及论文标题：
- Cross-Self KV Cache Pruning for Efficient Vision-Language Inference

---

## Intra-modality / Inter-modality Attention Decomposition in VLM KV Cache Pruning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Intra-modality（模态内）和 Inter-modality（跨模态）注意力分解是将多模态 VLM 的注意力矩阵按 token 模态归属进行结构化拆分的技术。给定包含 L_t 个文本 token 和 L_v 个视觉 token 的序列，注意力矩阵 A ∈ R^{L×L} 分解为：

- **Intra-modality**：A^{st}（text→text）捕捉文本内语义关系；A^{sv}（visual→visual）捕捉图像内空间关系
- **Inter-modality**：A^{ct}（visual→text）表示视觉信息对文本理解的影响；A^{cv}（text→visual）表示文本查询对图像区域的聚焦

CSP 通过 Kernel Density Estimation (KDE) 和 Jensen-Shannon (JS) Divergence 定量发现：self-attention 和 cross-attention 分布在 VLM 中显著不同且不重叠，不同层间 JS divergence 大幅变化。统一使用原始 self-attention scores 做剪枝会导致文本 token（通常 self-attention score 更大）被系统性地保留更多，而关键视觉 token 被过度剪枝。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 矩阵分解
A_st = A[0:L_t, 0:L_t]          // text→text [L_t, L_t]
A_sv = A[L_t:L, L_t:L]          // visual→visual [L_v, L_v]
A_ct = A[L_t:L, 0:L_t]          // visual→text [L_v, L_t]
A_cv = A[0:L_t, L_t:L]          // text→visual [L_t, L_v]

// 沿 query 轴求和 → 各 key token 的重要性
A_s = sum(A_st, q) ⊕ sum(A_sv, q)    // intra-importance [L]
A_c = sum(A_ct, q) ⊕ sum(A_cv, q)    // inter-importance [L]

// K^s/K^c 比率根据数据集调整
// 多数数据集: cross_ratio=0.5 (平衡)
// IR/Spot-the-Diff: cross_ratio=0.9 (偏 cross-attention)
// ActionPrediction: cross_ratio=0.0 (仅 self-attention)
```

**KDE/JS 分布分析揭示的数据集特异性**：
- CLEVR-Change: cross-attention 峰值集中且主导 → 强跨模态依赖
- DocVQA: self-attention 更分散 → 强模态内依赖
- ActionPrediction: 高 JS divergence → 完全依赖 intra-modal attention

术语一般如何实现？如何使用？

该分解是 CSP 的预处理步骤，在每次剪枝时对 multi-head 平均后的 A 矩阵执行。K^s/K^c 比率是仅有的可调超参数。代码开源：https://github.com/TerryPei/CSP。

涉及论文标题：
- Cross-Self KV Cache Pruning for Efficient Vision-Language Inference

---

## SwiGLU FFN（SwiGLU 前馈网络）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

SwiGLU FFN 是现代 LLM（Llama, Qwen, GPT-oss 等）中替代传统 ReLU FFN 的激活门控前馈网络结构。与标准两层 FFN（Y = σ(XW_up)W_down）不同，SwiGLU 使用三个投影矩阵和一个门控机制：Y = (SiLU(XW_gate) ⊙ XW_up)W_down，其中 SiLU(x) = x·σ(x) 是 Swish 激活函数，⊙ 为逐元素乘法。中间维度 d_ff 通常设为 8d/3（而非传统 FFN 的 4d）。两个 up-projection 矩阵（W_gate, W_up）的参数在训练中可视为 2×d_ff 的总扩展，但因 gating 机制提供的非线性，在同等参数量下比 ReLU FFN 表现更好。

从算法pipeline角度拆解术语：

```
// SwiGLU FFN 前向计算
输入: h ∈ R^d (attention output)
// Step 1: Gate path — 带 SiLU 激活
x_gate = h @ W_gate^T    // [d] → [d_ff], W_gate ∈ R^{d_ff×d}
g = SiLU(x_gate)          // SiLU(x) = x / (1 + e^{-x})
// Step 2: Up path — 无激活
x_up = h @ W_up^T         // [d] → [d_ff], W_up ∈ R^{d_ff×d}
// Step 3: Element-wise 门控
y_inter = g ⊙ x_up        // [d_ff], gate × up
// Step 4: Down projection
output = y_inter @ W_down^T  // [d_ff] → [d], W_down ∈ R^{d×d_ff}
// 总参数: d × d_ff × 3 (W_gate, W_up, W_down)
// vs ReLU FFN: d × d_ff × 2 (W_up, W_down) + d_ff ≈ 4d
```

**SwiGLU 在 prefill 阶段的内存瓶颈**：在 Llama-3 风格模型中，中间维度 I = 4d（因为 W_gate 和 W_up 各输出 d_ff ≈ 8d/3，但中间激活 I_up 和 I_gate 都约为 2d_ff ≈ 4d 量级）。Prefill 阶段处理完整序列 S 时，峰值中间激活内存 = S × I ≈ S × 4d，远大于 attention 优化后的内存（FlashAttention 中 attention 峰值内存约 S × d）。因此 SwiGLU MLP 成为 prefill 阶段峰值内存的主导因素（MOM 论文的观察）。

术语一般如何实现？如何使用？

本文使用 SwiGLU FFN（遵循 Llama-3 惯例），d_ff ≈ 8d/3（而非 ReLU FFN 的 4d），round 至最近 32 的倍数。SwiGLU 的参数计数计入 time-invariant cost（不随 T 增长），因此优化模型时 N 的 scaling 同时影响 SwiGLU 参数量。与 RMSNorm 和 RoPE 一起构成现代 Llama-style 模型的标准组件。在 GPU 上通常用 cuBLAS GEMM 实现三个矩阵乘法，SiLU 和 element-wise 乘用 CUDA kernel。SwiGLU 的中间激活内存占 LLM prefill 阶段峰值内存的主要部分，是长上下文推理的内存瓶颈（MOM 和 MST 均基于此观察提出优化）。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs
- MOM: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models

---

## Mini-Sequence Inference（Mini-Sequence 推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Mini-Sequence Inference 是一种 LLM 推理阶段的 MLP 内存优化算法，将 MLP 层的输入序列沿 token 维度划分为多个较小的 "mini-sequences"（每个大小 N ≈ S/M），逐个通过 MLP 计算以降低峰值中间激活内存。其核心原理基于：MLP 层对每个 token 的计算是独立的（无 token 间信息交互），因此可以分批处理而不改变输出结果（数学等价性）。此算法是 Mini-Sequence Transformer (MST, NeurIPS 2024) 在推理场景的适配——MST 的训练版本需要梯度累加来支持 backward pass，而 Mini-Sequence Inference 只需要前向 pass，因此更简单、更高效。

从算法pipeline角度拆解术语：

```
// Mini-Sequence Inference for MLP layers (MOM Algorithm 1)
输入: A ∈ R^{S×d} (attention output, S=sequence_length)
超参: C (mini-sequence size), M = ceil(S/C)

// 非最后 MLP 层: 完整 mini-sequence 处理
if not last_mlp_layer:
    Partition A into {A_i}_{i=1}^M, A_i ∈ R^{B×N×d}, N ≈ C
    O = []  // 输出列表
    for i = 1 to M:
        // SwiGLU MLP:
        // gate = SiLU(A_i @ W_gate^T)       [N, I]
        // up   = A_i @ W_up^T                [N, I]
        // hidden = gate ⊙ up                 [N, I]
        // O_i = hidden @ W_down^T            [N, d]
        O.append(O_i)
        // 释放 I_up_i, I_gate_i, hidden_i 的中间内存
    return concat(O)  // [S, d]
// 最后 MLP 层: 仅处理最后一个 token
else:
    A_last = A[-1, :]        // [1, d]
    O_last = MLP(A_last)     // [1, d]
    logits = LM_Head(O_last) // [1, vocab_size]
    return logits
```

**内存节省分析**：
```
Standard: M_intermediate = S × I, I ≈ 4d ≈ 16384 (Llama-3-8B)
MOM:      M_intermediate = N × I = (S/M) × I, M = S/C

举例 (S=128K, C=8192, d=4096, I=16384, bf16):
Standard: 128K × 16384 × 2B = 4.2 GB per MLP layer
MOM:      8K × 16384 × 2B = 262 MB per mini-sequence (16× reduction)
```

**与 Chunked Prefill 的关键差异**：
- Chunked Prefill：将整个 transformer block（attention + MLP + LM Head）切分为多个 chunk，每个 chunk 串行执行完整 forward → 导致 attention 重复计算和 KV cache 重载
- Mini-Sequence：仅切分 MLP 层，attention 保持完整序列处理 → 无 attention 重复计算，单次 forward pass 完成

**Decode 阶段的化简**：decode 时每步仅 1 个 token，MLP 中间激活 = 1 × I，远非瓶颈。因此 Mini-Sequence 只作用于 prefill 阶段。

术语一般如何实现？如何使用？

HuggingFace Transformers 中实现：(1) 识别所有 MLP 层（通常为模型中的 SwiGLU 模块）；(2) 非最后一层的 MLP 层：将输入分块，循环执行 `SiLU(X_chunk @ W_gate^T) ⊙ (X_chunk @ W_up^T) @ W_down^T`，拼接输出；(3) 最后一层 MLP + LM Head：仅取 X[-1:] 进行投影。代码改动极小，仅修改 MLP 的 forward 方法。

兼容性：(a) 与 FlashAttention 完全兼容（attention 层不变）；(b) 与 GQA/MQA 完全兼容（attention 结构不变）；(c) 与 KV cache offloading 完全兼容（MOM 的核心贡献即是将两者结合）。与 HuggingFace 的 OffloadedCache 直接集成。由于 mini-sequence 尺寸更小，可更好地适配 GPU L2 cache，使 mini-sequence only 模式甚至能略微提升吞吐量（MOM 论文观察）。

涉及论文标题：
- MOM: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models

---

## Mini-Sequence Transformer (MST)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Mini-Sequence Transformer (MST, NeurIPS 2024) 是一种 LLM 训练优化方法，通过将输入序列在 MLP 和 LM-Head 层内部划分为 mini-sequences 来降低训练时的峰值中间激活内存。与 Mini-Sequence Inference（推理版本）不同，MST 需要处理 backward pass 中的梯度累加。MST 结合 gradient checkpointing 和 gradient accumulation：forward 时逐 mini-sequence 计算并释放中间激活，backward 时 recompute 激活并累加梯度。MST 在 LLM 训练中实现 12-24× 的序列长度扩展（如 Llama3-8B 从 5K 扩展到 60K on single A100）。

从算法pipeline角度拆解术语：

```
// MST Training (simplified)
输入: X ∈ R^{S×d}
// Forward pass (chunked)
M = S / C  // 划分 mini-sequences
for i = 1 to M:
    // 仅保留 attention output
    A_i = Attention(X_i)        // 使用 FlashAttention，forward 后仅保留 output
    // MLP mini-sequence
    O_i = MLP(A_i)              // forward only，不保留中间激活（类似 gradient checkpointing）
    // 释放 A_i, MLP 中间激活

// Loss computation
loss = CrossEntropy(LM_Head(O_M[-1]), target)

// Backward pass (recompute + gradient accumulation)
for i = M down to 1:
    A_i = recompute_Attention(X_i)
    recompute MLP forward with grad
    accumulate gradients into W_gate, W_up, W_down
```

中间激活内存节省：标准训练 $I_{mem} = S \times I$，MST $I_{mem} = S \times I / M$（M 为 mini-sequence 数量）。MST 也可与 activation recomputation 正交叠加：两者结合时中间内存进一步降至 $I_{mem} = S \times I / (M \times checkpoint\_segments)$。

术语一般如何实现？如何使用？

基于 PyTorch，开源代码 https://github.com/wdlctc/mini-s (MIT license)。实现为 HuggingFace 模型的替换 MLP/LM-Head forward 方法。关键实现细节：(1) chunk size C 通常设为 hidden dimension d，当 S < C 时不拆分（短序列无 overhead）；(2) 与 DeepSpeed-Ulysses 序列并行兼容（attention 用 all-to-all，MLP 用 mini-sequence）；(3) 支持 LoRA 等 PEFT 方法。训练吞吐量几乎无损失（因为长序列下 compute/IO 主导项 SI 和 SV 不变）。

涉及论文标题：
- MOM: Memory-Efficient Offloaded Mini-Sequence Inference for Long Context Language Models

## Dynamic Attention Mask (DAM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dynamic Attention Mask (DAM) 是一种免微调的动态稀疏注意力机制，为 Transformer 模型中每个 layer 和 attention head 的 attention map 分配自适应的稀疏 mask。DAM 通过两阶段流程运作：(1) **离线 mask 生成阶段**：冻结的预训练模型在 Pattern Capture Length (PCL) 范围内处理输入序列提取完整 attention map → Box-Cox 变换放大中小注意力值 → 全局归一化后以阈值 τ 二值化生成 "true mask" → 通过结构模式匹配（对角线 + 垂直模式）识别 pattern → 对超 PCL 长度外推生成 "extended mask"；(2) **推理应用阶段**：将 mask 在 softmax 前以 Hadamard product 应用于 attention score，mask 位置设为 -∞。FLOPs 从 O(L²) 降至 O(sL)（s 为每 query 平均保留 key 数，s ≪ L）。与 MoA 的预定义静态 mask 不同，DAM 从真实 attention 分布中捕获每层每头的异构模式，无需 fine-tuning。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// === Stage 1: Offline Mask Generation ===
// Input: calibration dataset, frozen LLaMA model, PCL L
// Output: per-layer per-head binary masks M_{ℓ,h}

for each sequence in dataset:
    L_eff = min(len(sequence), L)
    attn_maps = frozen_model(sequence[:L_eff])  // extract full attention

// Accumulate mean attention across dataset
for each layer ℓ, head h, positions (i,j):
    A_mean = A_accumulated / (C_count + ε)  // ε = 10^{-8}

// Box-Cox feature amplification (λ=0.5)
X = max(A_mean, ε)
B = (X^{0.5} - 1) / 0.5  // Box-Cox with λ=0.5
A_tilde = B - min_all(B)   // global normalization

// True mask generation (τ=0.3)
M_true[i,j] = 1 if A_tilde[i,j] >= τ else 0

// Structural pattern matching (μ=0.8)
Pattern pool P = {P_diag,r: j=i-r} ∪ {P_vert,c: j=c, i≥c}
for each P_k in P:
    γ_k = sum(M_true * P_k) / sum(P_k)
    if γ_k >= μ: add P_k to matched

// Build extended mask for length S > L
M_ext[i,j] = 1 if any matched P_k has P_k_ext[i,j] == 1 else 0

// === Stage 2: Online Inference ===
attention_scores = Q @ K^T / sqrt(d_k)
attention_scores = attention_scores ⊙ M    // Hadamard product
attention_scores[M == 0] = -∞              // discard masked
O = softmax(attention_scores) @ V
```

术语一般如何实现？如何使用？

基于 PyTorch + HuggingFace Transformers 实现。Stage 1 离线运行一次在 Multi-News 上，生成 {layer, head} → mask mapping 文件；Stage 2 加载 mask 替代标准 attention mask。PCL=512 由 A100 40GB 的显存约束决定。mask 与 tile-based GPU 执行兼容且可与 FlashAttention 融合。开源：https://github.com/HanzhiZhang-Ulrica/DAM。

涉及论文标题：
- DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration

## Box-Cox Transformation for Attention Feature Amplification

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Box-Cox 变换是一种幂变换统计方法，在 DAM 用于 attention map 的特征放大。原始 attention 分布高度偏斜（少数大值主导，大量中小值被淹没），使直接阈值化难以区分重要的中等连接。Box-Cox 变换 $B(x) = (x^{\lambda} - 1) / \lambda$ 在 λ=0.5 时既放大中小 attention 值又不改变大值的相对顺序。DAM 比较了 9 种变换方法，Box-Cox 产生最紧凑的值范围（max≈2.0, mean≈0.27, std≈0.35），而 Square Root 产生极端值（max≈150），不利于阈值化。

从算法pipeline角度拆解术语：

```
// DAMPipeline_BoxCox_Step:
// Input: mean_attention A_mean (accumulated over dataset)
// Output: normalized_attention A_tilde (ready for thresholding)

ε = 1e-8
X = max(A_mean, ε)                // ensure positive
B = (X^{0.5} - 1) / 0.5           // λ=0.5: 放大小值, 保留大值尺度
A_tilde = B - min_all_layers_heads(B)  // shift to non-negative

// 对比: Square Root 变换 A_tilde_sqrt = sqrt(X) 产生 max≈150, std≈22
```

术语一般如何实现？如何使用？

λ=0.5 是经验选择且在 DAM 中固定。变换后全局减去最小值确保非负，然后直接进入 τ 阈值化步骤。跨 Multi-News 数据集累积的 mean attention 作为输入。论文对比 9 种方法后选择 Box-Cox，因其紧凑分布最便于统一阈值化。

涉及论文标题：
- DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration

## Pattern Capture Length (PCL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Pattern Capture Length (PCL)，记作 L，是 DAM 中提取 attention pattern 的最大序列长度。$L = \min(S, L_{\max})$，$L_{\max}$ 为 full attention 不 OOM 的最长序列。DAM 在 A100 40GB + LLaMA 3.2 3B 上设定 PCL=512，平衡了捕捉 attention pattern 的完整性和计算可行性。关键理念：从硬件最大支持长度开始，按需向下调整（"top-down" 调参），而非从短到长试探。论文图 7 证实短序列中观察到的结构模式（对角线、垂直条带）可外推至更长序列。

从算法pipeline角度拆解术语：

```
L = min(S, L_max)  // L_max from hardware constraint

// Stage 1: extract attention map only within PCL
if S <= L:
    masks = full_attention_masks(frozen_model, seq[:S])
else:
    partial_attn = full_attention_masks(frozen_model, seq[:L])  // L×L only
    true_mask = threshold(partial_attn, τ)                      // L×L binary
    matched_patterns = structural_match(true_mask, μ)
    extended_mask = extrapolate(matched_patterns, target_len=S)  // S×S
```

术语一般如何实现？如何使用？

PCL 仅在 Stage 1（离线）使用。值由硬件决定——原 LLaMA 3.2 3B 在 A100 40GB 上 >4K tokens OOM，PCL=512 提供安全裕量。更大的 GPU 允许更大的 PCL。选值策略简单：从最大可支持长度开始，仅在需要时下调。

涉及论文标题：
- DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration

## True Mask Generation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

True mask $M_{\ell,h} \in \{0,1\}^{L \times L}$ 是 DAM Stage 1 中对归一化 attention map $\tilde{A}_{\ell,h}$ 以阈值 τ 二值化生成的二进制矩阵。每个元素 $m_{i,j}$ 通过比较 $\tilde{A}_{\ell,h,i,j}$ 与 τ 决定：≥τ=1（保留连接），<τ=0（丢弃连接）。τ=0.3 通过 attention 稀疏性分析确定。True mask 为后续结构模式匹配提供输入，用于识别可外推的规律性 pattern。

从算法pipeline角度拆解术语：

```
for each (ℓ,h), for each (i,j) in [0..L-1]×[0..L-1]:
    M_true[ℓ,h,i,j] = (A_tilde[ℓ,h,i,j] >= τ) ? 1 : 0
// τ=0.3: 仅保留归一化 attention 前 ~70% 的有效连接
```

术语一般如何实现？如何使用？

seq_len ≤ PCL 时直接使用 true mask 做推理；seq_len > PCL 时 true mask 覆盖前 L×L 区域，其余由 extended mask 补充。τ 控制 mask 密度——更高值产生更稀疏的 mask，减少更多计算但也可能丢弃重要连接。论文在 LongEval 上验证 τ=0.3 时 DAM 平均精度 0.7966 vs Full Attention 0.8011。

涉及论文标题：
- DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration

## Structural Pattern Matching for Attention Masks

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Structural Pattern Matching 是 DAM 从 true mask 中识别标准化 attention 结构的方法。模式池 $\mathcal{P}$ 含对角线模式 $P_{\text{diag},r}$（$j=i-r$）和垂直模式 $P_{\text{vert},c}$（$j=c, i \geq c$），共 2L 个候选。匹配分数 $\gamma_k = \frac{\sum M \cdot P_k}{\sum P_k}$，$\gamma_k \geq \mu$（μ=0.8）则模式被匹配。匹配到的模式可直接外推至任意长度，使 DAM 处理超 PCL 序列无需重算 full attention。

从算法pipeline角度拆解术语：

```
// 输入: true_mask M[L×L], threshold μ=0.8
// 输出: extended mask M_ext[S×S] for any S

// Pattern pool: 2L patterns total
P = {P_diag,r: p[i,j]=1 iff j=i-r, r=0..L-1}
  ∪ {P_vert,c: p[i,j]=1 iff j=c and i≥c, c=0..L-1}

for each P_k in P:
    γ_k = (M ⊙ P_k).sum() / P_k.sum()
    if γ_k >= μ: matched.append(P_k)

// extrapolate matched patterns to length S
M_ext = sum(P_k_extrapolated(S) for P_k in matched)
M_ext = (M_ext >= 1).astype(int)  // binarize
```

术语一般如何实现？如何使用？

仅使用对角线和垂直两种模式（基于 LLaMA 3.2 3B attention 观察）。μ=0.8 在 0.7~1.0 范围内表现鲁棒。模式定义使外推极其简单——对角线条件 $j=i-r$ 和垂直条件 $j=c, i \geq c$ 对任意长度均成立。模式池可扩展（增加水平条带、块模式等）以提升精度。

涉及论文标题：
- DAM: Dynamic Attention Mask for Long-Context Large Language Model Inference Acceleration

## Filter Layer / GemFilter (Early-Layer Token Filtering for Long-Context LLMs)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Filter Layer（过滤层）是 GemFilter 论文提出的核心概念：利用 LLM 的早期 Transformer 层作为输入 token 重要性过滤器，在仅运行前 r 层（而非全部 m 层）后即可识别与 query 相关的重要 token，从而将长上下文输入从 n 个 token 压缩至 k 个 token（如 128K→1024，约 1000× 压缩率），再将压缩后的 token 子序列送入完整 LLM 进行生成。

核心发现：LLM 在早期层（如 LLaMA 3.1 8B 的第 13 层）的 attention 矩阵中就能定位 answer-related tokens——即模型在生成答案之前就已"知道"哪些输入 token 对回答是重要的。这一发现将 prompt computation 的计算量从 Θ(mhn²d) 降至 Θ(rhn²d)（r << m），同时将第二遍推理的序列长度从 n 降至 k，全流程显著加速。

GemFilter 是一个 training-free 方法，不需要任何微调或额外的模型参数，与任何 Transformer LLM 兼容。其算法流程由两次前向传递组成：(1) **第一遍（Filter Pass）**：仅运行前 r 层，在第 r 层提取所有 attention head 的最后一 query token 对全部 key token 的 attention scores，跨 head 求和后取 top-k 最高分的 token 索引；(2) **第二遍（Generation Pass）**：将选中的 k 个 token（按原始顺序排列）送入完整 m 层 LLM 做标准 generation。

与 SnapKV/H2O 的关键差异：GemFilter 在 prompt computation 阶段就减少了计算量（仅运行前 r 层），而 SnapKV/H2O 在 prompt computation 阶段仍需处理全部 m 层和全部 n 个 token。此外 GemFilter 使用单一的全局 token 索引集 J（可打印供人类审查），而 SnapKV/H2O 使用 m·h 套索引。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**GemFilter 两遍推理伪代码（对应 Algorithm 1）**：

```
// ====== Input ======
// tokens T ∈ V^n (n=128K), filter layer index r=13, topk k=1024
// F_{1:m}: m-layer transformer (m=32 for LLaMA 3.1 8B)

// ====== First Pass: Filter (仅前 r 层, Prompt Computation) ======
F_{1:r}(T) → 获取第 r 层输出

// 提取第 r 层的 query 和 key（多 head）
Q^{(r)} = [Q^{(r,1)}, ..., Q^{(r,h)}]   // 每 head: [n, d_h]
K^{(r)} = [K^{(r,1)}, ..., K^{(r,h)}]   // 每 head: [n, d_h]

// 取最后一 query token Q_n^{(r)} ∈ R^{h×d_h}
// 计算对全部 key token 的 attention scores（跨所有 head 求和）
for j in 1..h:
    scores_j = Q_n^{(r,j)} @ K^{(r,j)^T}   // [1, n] per head

total_scores = sum(scores_j for j in 1..h)   // [1, n]
total_scores = avg_pool1d(total_scores, kernel=5)  // 聚类平滑

J = topk_index(total_scores, k)
J = sort(J)   // 按原始顺序：确保 <bos> 在最前

// ====== Second Pass: Full Generation (完整 m 层) ======
T_J = T[J]   // 仅 k=1024 个 token（vs 原始 128K）
output = Gen(F_{1:m}, T_J)   // 标准 greedy generation
// RoPE 位置编码重新计算，最大距离 = k（而非 n）
```

**时间复杂度对比（Theorem 3.3）**：

| Phase | Standard Attention | SnapKV/H2O | GemFilter |
|-------|-------------------|-----------|-----------|
| Prompt Computation | Θ(mhn²d) | Θ(mhn²d) | **Θ(rhn²d)** |
| Iterative Generation | Θ(mh(nt+t²)d) | Θ(mh(kt+t²)d) | Θ(mh(k²+t²)d) |

**GPU 内存对比（Theorem 3.3）**：

| Phase | Standard | SnapKV/H2O | GemFilter |
|-------|----------|-----------|-----------|
| Prompt Comp | mw + 2mhnd | mw + 2hnd + 2mhkd | **rw + 2hnd** |
| Iterative Gen | mw + 2mh(n+t)d | mw + 2mh(k+t)d | mw + 2mh(k+t)d |

n=128K, k=t=1024, r=13, m=32 时：Prompt Time = Standard:SnapKV:GemFilter = 32:32:13 → ~60% 减少；Prompt Memory = mw+2mhnd : mw+2hnd+2mhkd : rw+2hnd。

术语一般如何实现？如何使用？

基于 HuggingFace Transformers v4.43 PyTorch 实现，仅需在 attention forward 中添加 `find_context` 调用。核心函数：(1) `find_context()`: 在 filter layer 提取 last-query-key scores → topk → sort；(2) `top_index()`: Q_n^T K 跨 head 求和 + avg_pool1d + topk。依赖 `transformers==4.43.3` 和 `flash-attn==2.6.3`。

使用示例：`python needle_eval.py --model <hf_id> --modified gemfilter --topk 1024 --ctx_len 32000`

Filter Layer 选择：LLaMA 3.1 8B (32 layers): r=13; Mistral Nemo 12B (40 layers): r=19; Phi 3.5 Mini 3.8B (32 layers): r=19。消融显示 layer 13-25 之间性能鲁棒。

核心性能（LLaMA 3.1 8B, H100-80GB）：2.4× speedup vs SnapKV，GPU 内存 -30% vs SnapKV / -70% vs Standard。Needle in a Haystack (128K) 上显著优于 Standard 和 SnapKV。LongBench 上与 SnapKV/H2O 可比。

代码开源：https://github.com/SalesforceAIResearch/GemFilter

涉及论文标题：
- Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction

---

## Vision-Language Context Sparsification (视觉-语言上下文稀疏化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Vision-Language Context Sparsification 是 Dynamic-LLaVA 提出的同时稀疏化多模态 LLM 推理中两种上下文（视觉 token 和语言 token）的框架。与仅稀疏化视觉 token 的方法（如 FastV）不同，Dynamic-LLaVA 使用两个可学习预测器在 prefill 阶段减少图像 token，在 decoding 阶段减少输出文本 token，实现整个 MLLM 生成过程的一致性高效推理。

核心动机（Eq. 4）：prefill 仅执行一次，image token 减少的收益在 decoding 阶段逐渐湮没——当输出文本 token 数量 |S_l^{OT}| → ∞ 时，Computation(Decoding_w/o_cache)_l ∝ |S_l^{OT}|，Memory(Decoding_w/cache)_l ∝ |S_l^{OT}|。仅减少 image token 无法在长生成中持续受益。Dynamic-LLaVA 是首个同时稀疏化 vision 和 language 上下文的 MLLM 高效推理框架。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Dynamic-LLaVA 三模式稀疏化推理 Pipeline**：

```
超参数: l=2, r^I=0.2, r^OT=0.5

=== Prefill 阶段 (Eq. 5)——仅 image token 稀疏化 ===
S_l^I, S_l^T = LLM_layers_1_to_l(image_tokens, text_tokens)
D^I = P^I(S_l^I)                          // Image Predictor: [N_l^I, d] → [N_l^I, 2]
M^I = argmax_j(D^I)                        // j=0 discard, j=1 keep
S_l^{I*} = {S_{l,i}^I | M_i^I == 1}       // 保留 r^I≈20% image tokens (~115/576)
S_l^{P*} = S_l^{I*} ∪ S_l^T               // 后续 L-l 层用缩减后 token 集

=== Decoding w/o KV Cache (Eq. 2 modified)——vision + language 同时稀疏化 ===
D^{OT} = P^{OT}(S_l^{OT})                 // Output Predictor: [N_l^{OT}, d] → [N_l^{OT}, 2]
M^{OT} = argmax_j(D^{OT})                 // M^{OT}_{N^{OT}} 强制=1（最后token始终保留）
S_l^{OT*} = {S_{l,i}^{OT} | M_i^{OT} == 1}  // 保留 r^OT≈50% 输出文本 token
S_{l+1} = LLM_layers_l+1_to_L(S_l^{P*} ∪ S_l^{OT*})  // 计算量减半

=== Decoding w/ KV Cache (Eq. 6)——在线 KV 压缩 ===
Q,K,V = W^{Q,K,V} · S_{l,N^{OT}}^{OT}
M^{OT}_{N^{OT}} = argmax(P^{OT}(S_{l,N^{OT}}^{OT}))  // 对当前token单点决策
O = W^O · Attention(Q, S_l^K ∪ K, S_l^V ∪ V)
if M^{OT}_{N^{OT}} == 1: S_l^K ∪= K, S_l^V ∪= V  // 保留 KV
else:                    S_l^K ∪= ∅, S_l^V ∪= ∅  // 丢弃 KV
S_{l+1,N^{OT}}^{OT} = FFN(O)
// 决策共享至所有后续层
```

术语一般如何实现？如何使用？

Predictor 架构：Image predictor 含 2 个 ViT blocks + MLP(512→256→128→2)；Output predictor 仅 MLP(512→256→128→2)。参数极小（<1% 总计算量）。一层预测，多层复用。训练时使用 MaskedSoftmax + Gumbel-Softmax(τ: 1.0→0.1) + STE 端到端优化。训练数据：LLaVA-1.5 656K Mixture（仅含图像样本）。代码开源：https://github.com/Osilly/dynamic_llava。

涉及论文标题：
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification

---

## MaskedSoftmax (掩码Softmax)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

MaskedSoftmax 是 Dynamic-LLaVA 在训练阶段使用的带二值 mask 矩阵的 Softmax 变体。在端到端训练 token pruning predictor 时，标准做法是将非必要 token 的 value 设为零向量，但这会破坏自回归过程——丢弃 output text token 的 value 意味着该 token 无法用于预测下一个 token，使 language modeling loss 计算断裂。MaskedSoftmax 将 mask 应用于 attention score 矩阵（而非 value 向量），既隔离了非必要 token 的影响，又保持了完整的自回归训练结构。

公式（Eq. 7）：

$$\text{MaskedSoftmax}(\mathbb{X}_{i,j}, \mathbb{G}) = \frac{\exp(\mathbb{X}_{i,j})\mathbb{G}_{i,j}}{\sum_{k=1}^{N_l} \exp(\mathbb{X}_{i,k})\mathbb{G}_{i,k}}$$

其中 $\mathbb{X} \in \mathbb{R}^{N_l \times N_l}$ 是 QK^T/√d_k，$\mathbb{G} \in \{0,1\}^{N_l \times N_l}$ 由 predictor mask $\mathcal{M} = \mathcal{M}^I \cup \{1\}^{N^T} \cup \mathcal{M}^{OT}$ 构造，且 $\operatorname{diag}(\mathbb{G}) = 1$（每个 token 始终能 attend 到自己）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 构造 mask 矩阵
M = M^I ∪ {1}^{N^T} ∪ M^{OT}      // [N_l] 全 token 集合 mask
G = {M}^{N_l}                      // [N_l, N_l], 每行 = M
diag(G) = 1                        // 对角线强制为 1

// 修改 Attention 计算
scores = Q @ K^T / sqrt(d_k)       // [N_l, N_l]
// Causal mask 仍正常应用: scores[future] = -inf
// MaskedSoftmax: mask 矩阵通过乘法隔离非必要 token
attn_weights = MaskedSoftmax(scores, G)
// = exp(scores) * G / Σ exp(scores) * G  (element-wise)
O = attn_weights @ V
```

术语一般如何实现？如何使用？

与标准 causal attention mask 的对比：causal mask 是将未来位置设为 -∞（加法操作），MaskedSoftmax 是在 softmax 分子和分母中通过乘法引入二值 mask。两者可联合使用。Dynamic-LLaVA 训练时 causal mask 保证每个 token 仅基于前文特征做决策（与 inference 一致），MaskedSoftmax 隔离非必要 token 的 attention 影响（与 inference 时的 token 移除等价）。消融实验（Tab. 7）：w/o MaskedSoftmax 导致 VQAv2 下降 1.1%（77.8→76.7）、GQA 下降 1.5%（61.3→59.8）。

涉及论文标题：
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification

## Top-k Attention with ANN Retrieval for Long-Context LLM (基于ANN检索的Top-k注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Top-k Attention with ANN Retrieval 是一种将标准 dense attention 替换为仅对 top-k 个最相关 key-value pair 进行注意力计算的方法，通过 Approximate Nearest Neighbor (ANN) search 在 CPU 端的完整 KV cache 中检索与当前 query 最相关的 k 个 key，仅传输对应的 value 到 GPU 参与 attention 计算。核心逻辑链：(1) Attention scores 天然具有稀疏性——现代 LLM 中仅极少数 token 贡献了绝大多数 attention mass（图 3: 深层 layer 仅需少数 token 覆盖 75% 注意力质量）；(2) inner product attention score q·K^T 可直接用作向量相似度度量（dot product metric），因此 ANN search 可以代理 attention score computation；(3) 将完整 KV cache 存放在 CPU 内存中（便宜且充裕），仅将 k 个 value 向量从 CPU 传输到 GPU，将 GPU attention 计算的复杂度从 O(N·D) 降至 O(k·D)；(4) k 可以极小——2% of N 足以恢复 95% dense attention 性能，k=0.001% 即可完成 Needle In A Haystack 任务。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Top-k Attention Decoding Pipeline（Llama-3-8B, 1M token context, commodity GPU ~16GB）**：

```
# === Stage 1: Prefill & KV Cache Construction (一次性，高算力) ===
# 在 H100 GPU 上使用 FlashAttention 逐 chunk prefill
K_cache = []  # 存储在 CPU host memory
V_cache = []  # 存储在 CPU host memory
for chunk in split_into_chunks(input_tokens, chunk_size):
    Q, K, V = model.chunk_forward(chunk)
    K_cache.append(K.cpu())
    V_cache.append(V.cpu())

# === Stage 2: Build ANN Index (CPU side) ===
K_index = []  # Faiss indexes, one per head per layer
for ℓ in 1..L:
    K_index[ℓ] = []
    for h in 1..H:
        idx = faiss.IndexFlatIP(d_k)  # dot product = attention score
        idx.add(K_cache[ℓ][h])        # (N, d_k) -> index
        K_index[ℓ].append(idx)

# === Stage 3: Top-k Decoding Loop ===
K_gen = [[] * L]   # GPU-side: recently generated token keys
V_gen = [[] * L]   # GPU-side: recently generated token values

for step in 1..max_new_tokens:
    for ℓ in 1..L:
        # GPU: QKV projection
        q = x @ W_Q[ℓ]  # (1, d_k)
        k = x @ W_K[ℓ]; v = x @ W_V[ℓ]

        # CPU: ANN search for top-k context keys
        vals, I = K_index[ℓ][h].search(q.cpu(), k_per_head)
        # vals, I in R^k: top-k inner product scores + indices

        # GPU: Transfer selected V + scores
        V_sel = V_cache[ℓ][h][I].to_gpu()  # (k, d_v)
        vals_gpu = vals.to_gpu()

        # GPU: Attention over context (top-k)
        attn_ctx = softmax(vals_gpu / sqrt(d_k)) @ V_sel  # (1, d_v)

        # GPU: Attention over recent generated tokens (windowed)
        attn_gen = softmax(q @ K_gen[ℓ][h]^T / sqrt(d_k)) @ V_gen[ℓ][h]

        # Merge and continue
        attn_out = attn_ctx + attn_gen

        # Update GPU window cache
        K_gen[ℓ][h].append(k); V_gen[ℓ][h].append(v)

    x_new = sample(lm_head(attn_out))
```

术语一般如何实现？如何使用？

实现：(1) Faiss (Facebook AI Similarity Search) 作为核心向量检索引擎，支持 IndexFlatIP（exact inner product search）或 IndexHNSWFlat（approximate HNSW graph search）；(2) Prefill 阶段可使用 FlashAttention（单卡 H100）或 Ring Attention（分布式）构建完整 KV cache；(3) Decoding 阶段 GPU 侧维护一个小窗口的近期生成 token KV cache（windowed attention），CPU 侧维护完整 context KV cache 和 Faiss index；(4) CPU-GPU 数据传输仅涉及 k 个 value 向量（k ≪ N），避免 FlexGen 式全量 KV cache 往返搬移。

使用建议：(1) k 按 context length 的百分比设置——k = 2% of N 实现 >95% dense attention 性能，k = 0.001% 足以完成 NIAH；(2) 支持 layer-wise adaptive k budget——给定固定总 budget Σk_ℓ，按 linear increasing from first to last layer 分配获得更好的性能；(3) 适用于 "rent cloud for prefill once, query locally many times" 的使用模式；(4) 开源实现：https://github.com/ryansynk/topk-decoding。

涉及论文标题：
- Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs

---

## Attention Entropy (注意力熵)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention Entropy 是衡量单个 query 的 attention score 分布在所有 key 上集中程度的信息论度量。定义为对 softmax 后的 attention scores 计算 Shannon entropy：
$$E = -\sum_{i=1}^{N} a_i \log(a_i)$$
其中 $(a_1, \ldots, a_N) = \operatorname{Softmax}(qK^T/\sqrt{d_k})$ 是 attention score 分布。Attention entropy 越低表示 attention 越集中（sparse——少数 token 占据绝大部分 attention mass），越高表示 attention 越均匀分散（dense——所有 token 贡献大致相等）。最大熵发生在 uniform 分布时：$E_{\max} = -\log(1/N) = \log(N)$。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Attention Entropy 计算流程**：
```
# 输入: attention scores S = qK^T ∈ R^{N}
# 输出: entropy E ∈ R (标量)

def compute_attention_entropy(S):
    a = softmax(S / sqrt(d_k))     # (N,) 概率分布
    E = -sum(a_i * log(a_i) for i in 1..N)
    return E

# E ∈ [0, log(N)]
# E → 0: 极度集中 (one-hot-like, 所有 mass 在单个 token)
# E → log(N): 完全均匀 (所有 token 同等重要)
```

**跨层 Attention Entropy 分析**（Exploiting Sparsity 论文 Fig.5）：
```
for each layer ℓ in 1..L:
    # 对 50 个 1000-token samples 的最后一个 token 计算
    E_layer[ℓ] = mean_over_samples(compute_attention_entropy(S_last))
    # 聚合 all heads 的 attention score

# 观察: Layer 1 的 entropy 显著高于后续 layers
# Layer 2-32 的 entropy 迅速下降并保持低位
# → 深层 attention 天然更稀疏，可以更激进地压缩
```

术语一般如何实现？如何使用？

Attention entropy 在长上下文推理中有三个核心用途：(1) **稀疏度预测**——低 entropy 表示高稀疏度，可以用较少 k 恢复 dense attention 性能；(2) **任务难度预估**——不同的下游 task 的 attention entropy 有显著差异（Needle In A Haystack: entropy 1.93 vs Word Counting: entropy 2.68），高 entropy 任务需更多 k；(3) **跨层 budget 分配指导**——第一层 entropy 最高，后续层迅速下降，指导 layer-wise adaptive k 分配策略（给第一层更多 k budget，后续层逐渐减少）。

Exploiting Sparsity 论文发现：attention entropy 与 "达到 95% dense attention 性能所需的最小 k%" 之间的 Pearson correlation 达到 0.847，表明 entropy 是预测 k 需求的可靠指标。

涉及论文标题：
- Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs

---

## Needle In A Haystack (NIAH) / 大海捞针测试

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Needle In A Haystack (NIAH) 是评估 LLM 长上下文检索能力的最广泛使用的 benchmark 之一。测试方法：在长文档（"haystack"——通常由填充文本如 Paul Graham essays 组成）中的特定位置插入一条特定信息（"needle"——如 "The magic number is 12345"），要求模型在给定长文档的情况下正确回答关于 needle 的问题（如 "What is the magic number?"）。评估维度包括：(1) context length（从 1K 到 1M+ tokens）；(2) needle depth（needle 在 context 中的位置，如 0%/25%/50%/75%/100%）；(3) 不同难度的变体（single NIAH, multi NIAH, multi-needle with distractors 等）。

RULER benchmark（Hsieh et al., 2024）将 NIAH 扩展到 13 个 tasks，包含：(a) 原始 NIAH（single needle）；(b) Multi-keys NIAH（多个不同标签的 needle）；(c) Multi-values NIAH（同一标签对应多个 needle）；(d) Multi-queries NIAH（多个问题）；(e) Variable Tracking（追踪变量赋值链）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**NIAH 测试 Pipeline**：
```
# Needle = 关键信息（短文本）
# Haystack = 长填充文本（"The grass is green..." × N times）
# Depth = Needle 插入位置占 context length 的比例 (0-100%)

def evaluate_niah(model, context_length, depth):
    # 1. 构造测试序列
    haystack_prefix = repeat_filler_text(depth * context_length / 100)
    haystack_suffix = repeat_filler_text((100-depth) * context_length / 100)
    prompt = haystack_prefix + NEEDLE + haystack_suffix + QUESTION

    # 2. 推理
    response = model.generate(prompt)

    # 3. 评估
    return contains_needle_info(response, NEEDLE_ANSWER)
    # 返回 0 (失败) 或 1 (成功)
```

术语一般如何实现？如何使用？

NIAH 由于其低 entropy (1.93) 和单点检索特性，是最容易用 sparse attention 完成的任务——Exploiting Sparsity 论文显示 k=1 即可在 1M token context 上达到 100% 成功率。相比之下，Word Counting（需遍历全部文本统计词频）的 attention entropy 为 2.68，需要 8.87% of N 的 k 才能达到 95% dense attention 性能。NIAH 的简单性使其成为验证 long-context retrieval 方法最低可行性的标准，但不应作为长上下文能力的唯一评估依据。

涉及论文标题：
- Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs

---

## Layer-wise Adaptive k Budget (逐层自适应k预算)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Layer-wise Adaptive k Budget 是一种在 fixed total compute budget（总 k 值固定为 Σk_ℓ）条件下，通过非均匀分配每层 attention 的 k 值来最大化模型性能的技术。核心观察：(1) 不同层的 attention 稀疏度不同——Layer 1 entropy 最高（attention 最分散），后续层 entropy 迅速下降；(2) 后续层的 attention 更集中、更 sparse，意味着可以用更小的 k 而不损失性能；(3) 在 fixed total budget 约束下，将 k budget 从前层向后层线性递增（而非 uniform 分配），可以在不增加总计算量的前提下获得 non-trivial performance boost。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Uniform vs Adaptive k Budget 对比**：
```
# Total budget: K_total = Σ k_ℓ (固定，如 total k per layer sum = 32*128 = 4096)

# Strategy 1: Uniform
for ℓ in 1..L:
    k_ℓ = K_total / L      # 每层相同，如 128

# Strategy 2: Linear Increasing (Exploiting Sparsity)
for ℓ in 1..L:
    k_ℓ = k_base + (ℓ/L) * k_slope  # 第一层最小，最后一层最大
    # k_base + k_slope/2 = K_total/L (保持总 budget 不变)

# Decoding 时使用对应的 k_ℓ
for ℓ in 1..L:
    vals, I = K_index[ℓ].search(q.cpu(), k_ℓ)  # 逐层不同的 k
    attn_out = softmax(vals/sqrt(d_k)) @ V_ℓ[I]
```

术语一般如何实现？如何使用？

实现思路：(1) Offline profiling——在不同 task 上分析每层的 attention entropy 分布，确定各层相对 k 需求；(2) 基于 entropy 的自动分配——Pearson correlation 0.847 表明 entropy 是预测 k 需求的有效 proxy，可按 entropy 比例分配 k；(3) 使用方式简单——仅需在 Faiss search 阶段传入不同的 k 参数，无需修改其他 pipeline 组件。

Exploiting Sparsity 论文（Fig.9）在 RULER benchmark 上的实验显示：在相同 total k budget 下，linear increasing 策略优于 uniform 策略，对于某些 k budget 可获得 ~2% 的 RULER score 提升。

涉及论文标题：
- Exploiting Sparsity for Long Context Inference: Million Token Contexts on Commodity GPUs

---

## Token-Selective Propagation (TSP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token-Selective Propagation (TSP) 是 FastKV 论文提出的两阶段预填充策略的核心机制。TSP 在 transformer decoder 的中间某层（TSP Layer，如 LLaMA-3.1-8B 的 layer 15），基于最近 window tokens（默认 8 个）的注意力权重计算每个输入 token 的 saliency score，仅将得分最高的 top-R_TSP（默认 20%）token 的 hidden states 传播到后续层。TSP 之前的层保持完整上下文计算，之后的层仅在压缩后的 hidden states 子集上计算注意力。TSP 的动机来自 Layer-dependent Context Dynamics 的观察：早期层的注意力焦点高度不稳定（不同层的 critical token 集合差异大），若过早剪枝会不可逆地丢弃后续层需要的 token；后期层的注意力则趋于稳定（可以安全剪枝）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**TSP 在 FastKV 两阶段预填充中的伪代码：**

```
# 阶段一：完整上下文预填充（Layer 0 到 L_TSP）
for l = 0 to L_TSP:
    X, Att_l, K_X, V_X = layer_l(X)
    K, V = KV_Compress(K_X, V_X, Att_l, R_KV)       # 独立于 TSP 的 KV 压缩
    if l == L_TSP:
        # TSP: 计算 saliency score
        for i in 0..N_I-1:                            # N_I = 输入 token 数
            for h in 0..H-1:                           # H = head 数
                # Eq(1): window tokens 作为 query 的注意力聚合
                S_i^{l,h} = MaxPooling( Σ_{n=0}^{N_obs} Att_l[h, N_I-1-n, i+m] )
            # Eq(2): 跨 head 平均得到 layer-level saliency
            S_i^{TSP_layer} = (1/H) * Σ_h S_i^{TSP_layer,h}
        # 选取 top-R_TSP 关键 token + 所有 window tokens
        I_TSP = TopK(S^{TSP_layer}, N_I * R_TSP)
        I_TSP = I_TSP ∪ {N_I - N_obs, ..., N_I - 1}    # window tokens 强制保留
        x = X[I_TSP]                                   # 仅传播选中的 hidden states

# 阶段二：压缩上下文预填充（TSP Layer+1 到 Last Layer）
for l = L_TSP+1 to L-1:
    x, Att_l, K_x, V_x = layer_l(x)                   # 仅在 x（压缩后）上计算
    K, V = KV_Compress(K_x, V_x, Att_l, R_KV)
```

**TSP 与 GemFilter 的关键差异：** GemFilter 在 filter layer 选择 token 后，从 layer 0 重新开始预填充——早期层被迫使用同一 token 子集。TSP 保留早期层完整上下文，仅在后层（注意力稳定后）剪枝。被 TSP 丢弃的 token 已在早期层的注意力中将语义融合到保留 token 中（Figure 7）。

**TSP Layer 自动选择（Eq 3）：**
```
L_TSP = argmin_{L ≤ L_max} (1/N) Σ_{i=1}^{N} ||H_i - H'_{L,i}||₂²
```
其中 H_i 为完整上下文下最终层 hidden state，H'_{L,i} 为在 L 层应用 TSP 后的 hidden state。通过少量标定数据最小化输出偏差选择最优 TSP 层。

术语一般如何实现？如何使用？

实现集成在 HuggingFace Transformers 的 self-attention 层中，与 FlashAttention-2 兼容。关键实现要点：(1) TSP 的 saliency scoring 仅基于 N_obs=8 个 window token 的注意力行，计算开销极小（128K上下文仅 0.15s，占总预填充 0.88%）；(2) MaxPooling kernel_size=7 用于平滑时间维度的注意力分数；(3) TSP rate 与 KV retention rate 完全解耦独立配置——TSP rate 控制预填充计算量（等于 1 - Σ_{l>TSP}(1-R_TSP)，LLaMA-3.1-8B 下约 60%），KV retention rate 独立控制解码时每层保留的 KV cache 比例（10% 或 20%）；(4) window tokens 强制保留机制确保最新上下文永远可用。

涉及论文标题：
- FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation

---

## Layer-dependent Context Dynamics (层依赖的上下文动态)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Layer-dependent Context Dynamics 是 FastKV 论文通过实验揭示的一种 LLM 内部注意力动态现象：transformer decoder 的不同层对关键 token 的注意力焦点表现出显著不同的行为模式。通过喂入 128K token 输入到 LLaMA-3.1-8B-Instruct，在每层收集获得最高平均注意力质量（across heads）的 top-512 critical tokens，计算层间 critical token 索引的平均重叠率随层距离的变化曲线（Figure 1a），发现：(1) 早期层（≤15）：重叠率随层距离增大而急剧下降，表明各层的 critical token 集合频繁变化——注意力高度不稳定；(2) 后期层（>15）：重叠率衰减显著减缓，表明同一 token 子集在多个连续层中保持一致性重要——注意力趋于稳定。这一动力学解释了为何 GemFilter（单层 token 选择应用于所有层）和 PyramidInfer（从首层即开始剪枝）会导致显著的准确率损失——它们在注意力稳定之前就丢弃了后续层仍需要的 token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**分析层依赖上下文动态的实验方法：**

```
# 输入：128K token 序列 X
# 输出：每层的 top-512 critical token indices

for l = 0 to L-1:
    Att_l = layer_l(X).attention_weights      # shape: (H, N_I, N_I)
    for h = 0 to H-1:
        # 跨 query 维度平均，得到每个 key token 的全局被关注度
        token_attn_lh[i] = mean_over_queries(Att_l[h, :, i])
    token_attn_l = mean_over_heads(token_attn_lh)
    top512_l = TopK(token_attn_l, 512)         # 当前层的 critical tokens

# 计算层间重叠率（Overlap Ratio）
for delta = 1 to L-1:
    for l1 = 0 to L-delta-1:
        l2 = l1 + delta
        overlap = |top512_l1 ∩ top512_l2| / 512
        avg_overlap[delta] += overlap / (L - delta)
```

**该动力学对算法设计的关键启示：**
- 早期层（≤15）必须处理完整上下文——确保每层可自由关注其偏好的 token 子集，即使这些子集跨层差异大。
- 后期层（>15）可以安全地对上下文进行激进剪枝——各层关注的 token 子集高度重叠，剪枝引入的信息损失最小。
- TSP 层应当选在上下文稳定点之后（LLaMA-3.1-8B 的 layer 15），之前层保留完整上下文，之后层仅传播关键 token。

术语一般如何实现？如何使用？

该观察直接指导了 FastKV 的 TSP 层选择策略（Eq 3：argmin L2 distance）。在实际应用中：(1) 对不同模型需重新分析层依赖动态以选择合适的 TSP 层（LLaMA-3.1-8B 选 layer 15，Ministral-8B 选 layer 17，Mistral-Nemo-12B 选 layer 19）；(2) 也可通过 Eq 3 的自动标定方法，在少量样本上最小化 TSP 输出与完整上下文输出的 hidden state L2 距离来选层；(3) 该分析使用完整 attention map，在生产部署中不需要重复，仅在设计阶段进行一次离线分析即可。

涉及论文标题：
- FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation

---

## Grouped-head latenT Attention (GTA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Grouped-head latenT Attention (GTA) 是一种结合了 **shared attention map（共享注意力矩阵）** 和 **nonlinear value decoder（非线性值解码器）** 的高效注意力机制，由 Sun 等人于 2025 年提出。其核心思想是利用注意力计算中的冗余性——不同 head 的 attention map 高度相似，且 KV cache 可显著压缩——来同时减少预填充 FLOPs 和解码阶段的 KV cache 大小。

GTA 包含两个关键组件：

**(1) Shared Attention Map（共享注意力矩阵）**：将 query heads 分为 n_q 组、key heads 分为 n_k 组（n_q, n_k << n_h）。每个 head i 通过映射函数 q(i) 和 k(i) 分别分配到 Q group 和 K group，同一 Q group 内的 heads 共享同一套 QK^T 注意力计算。这从 MHA 的每 head 独立计算 n_h 次降至 n_q 次（n_q 为 Q group 数，<< n_h）。Key cache 仅需存储 n_k × d_h 维而不是 n_h × d_h 维。

**(2) Nonlinear Value Decoder（非线性值解码器）**：将 value cache 压缩为 latent space：引入 C = XW_C ∈ R^{N × n_c × d_l}（共享 latent value），每个 head 的 value V_i 由 V_i = (C_{c(i)}W_{P,i}) ⊙ Sigmoid(x_tW_{G,i}) 动态生成，而不是存储独立的 V_i。Latent dimension d_l ≥ d_h 以保证全秩投影不损失信息。Sigmoid gate 提供 context-adaptive 的非线性调制，增广 value 表示的有效秩。

这两个机制结合的效果是：KV cache 从 MHA 的 2n_h d_h N 降至 (n_k d_h + n_c d_l)N（1B 模型下仅 30% of GQA）；attention FLOPs 从 2n_h d_h N^2 降至 n_q(d_h + d_l)N^2（1B 模型下仅 37.5% of GQA）。

GTA 的预填充计算复杂度为 O(2NH^2 + (n_q d_h + n_k d_h + n_c d_h + d_l)NH + n_q(d_h + d_l)N^2)。解码时每步生成复杂度为 O(2H^2 + (n_q d_h + n_k d_h + n_c d_h + d_l)H + 2n_h d_h N)。关键效率优势来自 Eq 8 的 reformulation：将 attention 计算放在 latent space 上执行，decode 时无需从 latent vector 为每个 token 重新解压完整 value。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**GTA 张量计算流程（以 500M GTA4 配置为例：n_h=20, n_q=10, n_k=1, n_c=2, d_h=64, d_l=256, H=1280）：**

```
# === Prefill Phase (N tokens) ===

# 1. 输入投影 (Eq 5)
Q = X @ W_Q            # W_Q ∈ R^{1280×640},  Q ∈ R^{N×640}  (10 Q groups × 64)
K = X @ W_K            # W_K ∈ R^{1280×64},   K ∈ R^{N×64}   (1 K group × 64)
C = X @ W_C            # W_C ∈ R^{1280×512},  C ∈ R^{N×512}  (2 C groups × 256)

# 2. Head-to-Group 映射
# q(i): head 0→Q0, head 1→Q0, ..., head 9→Q9, head 10→Q0, ...
# k(i): all heads → K0  (n_k=1)
# c(i): heads 0-9 → C0, heads 10-19 → C1  (n_c=2)

# 3. 分组计算 attention (仅 n_q=10 次，非 20 次)
for g in 0..9:
    Q_g = Q[:, g*64:(g+1)*64]       # (N, 64)
    S_g = Q_g @ K^T / sqrt(64)       # (N, N)  attention scores
    A_g = softmax(S_g)               # (N, N)  attention weights
    
    for each head i where q(i) == g:
        c_idx = c(i)                 # 0 或 1
        C_ci = C[:, c_idx*256:(c_idx+1)*256]  # (N, 256)
        
        # 4. 非线性 Value Decoder (Eq 6 → Eq 8 reformulation)
        # Latent-space attention: 直接对 latent C 做加权
        O_i_latent = A_g @ C_ci      # (N, 256)
        
        # Head-specific 投影 + context-adaptive gate
        O_i = (O_i_latent @ W_{P,i}) ⊙ sigmoid(x_t @ W_{G,i})
        # W_{P,i} ∈ R^{256×64}, W_{G,i} ∈ R^{1280×64}
        
        # 5. 输出投影
        O_i = O_i @ W_{O,i}          # (N, 1280)

# 6. 合并所有 heads
O = sum(O_i for all heads i)        # (N, 1280)

# === KV Cache 写入 ===
# 仅存储 K (64 dims/token) 和 C (512 dims/token)
# 共计 576 dims/token/layer
# vs MHA: 2560 dims/token/layer = 22.5%
# vs GQA: 512 dims/token/layer (8×64) = 112.5% 反而更大
# 但实际 GTA4 的 n_k=1 让 K=64 极小，整体仍远小于 MHA

# === Decode Phase (1 new token) ===
# 追加 K_new (1,64) 和 C_new (1,512) 到 cache
# 对每组重新计算 score: S_g_new = Q_g_new @ K_all^T / 8
# A_g_new = softmax(S_g_new)  # 仅对 1 行 query 做
# O_i_latent_new = A_g_new @ C_all  # latent-space attention
# O_i_new = (O_i_latent_new @ W_{P,i}) ⊙ sigmoid(x_t @ W_{G,i})
```

**GTA 配置变体（论文 Table 5）：**

| 配置 | n_q | n_k | n_c | d_l | KV cache dims | vs MHA |
|------|-----|-----|-----|-----|---------------|--------|
| GTA1 (160M) | 3 | 1 | 1 | 128 | 192 (64+128) | 12.5% |
| GTA2 (160M) | 6 | 1 | 1 | 128 | 192 (64+128) | 12.5% |
| GTA3 (500M) | 5 | 1 | 1 | 128 | 192 (64+128) | 7.5% |
| GTA4 (500M) | 10 | 1 | 1 | 256 | 320 (64+256) | 12.5% |
| GTA-1B | 5 | 1 | 1 | 128 | 192 (64+128) | 7.5% |

术语一般如何实现？如何使用？

实现方式（基于论文和 GitHub repo https://github.com/plm-team/GTA）：

1. **训练实现**：在 PyTorch 中替换标准 MultiHeadAttention 模块。关键实现点：(a) W_Q/W_K/W_C 投影——W_Q 输出 n_q×d_h 维、W_K 输出 n_k×d_h 维、W_C 输出 n_c×d_l 维；(b) Head-to-group 映射表维护；(c) Eq 8 reformulation 实现——在 latent space 计算 attention，避免 decode 时重复解压；(d) Gate 生成 ——对每个 head 维护 W_{G,i}，gate 仅在当前 token x_t 上计算（与序列长度无关）；(e) RoPE 应用于 Q 和 K 的投影后添加。

2. **训练配置**：AdamW optimizer，cosine LR scheduler，global batch size 800-2048，训练于 4 节点 32×A800 GPU。160M 和 500M 模型使用 C4 数据集（1 epoch），1B 模型使用 smollm-corpus（220B tokens）。

3. **SFT 微调**：使用 LlamaFactory [39] 框架和 tulu3-sft-mixture 数据集。

4. **推理部署**：使用 HuggingFace Transformers v4.36.0。支持 DynamicCache（标准）和 OffloadedStaticCache（缓存卸载）。FP16/BF16 和 FP32 均支持。

5. **评价**：使用 lm-evaluation-harness [25] 在 PIQA、HellaSwag、ARC、Winogrande、BoolQ、MathQA、TruthfulQA 等 benchmark 上评估。使用 LLM-Viewer [38] 做 roofline 模拟评估预填充/解码时延。

适用场景：资源受限设备上的 LLM 部署（NVIDIA H100/A800/RTX 3060/Apple M2/BCM2712），尤其是长上下文生成（论文测试至 4096 tokens）和需要同时优化预填充+解码延迟的场景。论文坦承缺乏工程级 kernel 优化（"The limitation stems from our lack of engineering-focused optimization efforts"），理论效率增益的上限尚未达到，未来结合自定义 GPU kernel（如 FlashAttention 风格融合 kernel）可进一步提升。

涉及论文标题：
- GTA__Grouped-head_latenT_Attention

---

## Grouped-Value Attention (GVA) / Grouped-Head Attention (GHA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

GVA 和 GHA 是 GTA 论文在方法论演进中引入的两种中间注意力变体，作为从 MHA 到 GTA 的"进化步骤"（evolving patterns of attention mechanisms）。

**Grouped-Value Attention (GVA)**：将 attention weights（从 Q 和 K 计算得出）在 heads 组内共享。同一 group 内的多个 heads 使用相同的 attention distribution（softmax(QK^T)），但各自应用不同的 value 投影。这复用了 attention weight 计算（减少 QK^T 次数），但保留了每个 head 独立的 value transformation。KV cache 仍为 (H + n_k d_h)N，内存节省来自 K 的减少而非 V。

**Grouped-Head Attention (GHA)**：进一步压缩——在 heads 组内共享 Q 和 K 表示（同一 group 内使用相同的 query、key），values 从共享源分别计算。这显著降低了 KV cache 至 (n_k d_h + n_v d_h)N。代价是 reduced diversity in Q/K representations（共享导致 query/key 的多样性下降），限制了模型在复杂任务上捕获细粒度上下文依赖的能力。

GVA 和 GHA 共同阐释了 attention 机制中 **efficiency-expressivity trade-off**：GVA 在较少牺牲表达力的情况下减少计算冗余；GHA 以表达力损失换取更大的内存/计算节省。GTA 通过在 GHA 基础上引入 nonlinear value decoder（从 latent 动态生成 head-specific values），在保持 GHA 级内存效率的同时恢复表达力。

从算法pipeline角度拆解，给出具体例子。

**GVA 计算流程**：
```
# 假设 n_h=12, n_q=3, n_k=3, n_v=12 (MHA 的 V group 数=12)
for g in 0..2:
    Q_g = head_group_queries[g]       # 4 heads 共享
    K_g = K[:, g*64:(g+1)*64]
    A_g = softmax(Q_g @ K_g^T / 8)    # (N,N) → 4 heads 共享
    
    for each head i in group g:
        V_i = head_values[i]           # 独立 V
        O_i = A_g @ V_i                # 共享 attention × 独立 value
```

**GHA 计算流程**：
```
# 假设 n_h=12, n_q=3, n_k=3, n_v=3
for g in 0..2:
    Q_g = Q[:, g*64:(g+1)*64]          # Q 按 group 共享
    K_g = K[:, g*64:(g+1)*64]          # K 按 group 共享
    C_g = V_shared[:, g*64:(g+1)*64]   # 共享 value source
    
    A_g = softmax(Q_g @ K_g^T / 8)     # (N,N)
    
    for each head i in group g:
        V_i = C_g @ W_{i}               # 从共享源派生 head-specific V
        O_i = A_g @ V_i
```

**GTA 在 GHA 上的关键改进（对应 Eq 6）：**
```
# GHA: V_i = C_g @ W_i                  # 线性投影，仅依赖 C_g
# GTA: V_i = C_g @ W_{P,i} ⊙ σ(x_t W_{G,i})  # 非线性 + context-adaptive
# σ 为 Sigmoid，x_t 为当前 token
```

术语一般如何实现？如何使用？

GVA 和 GHA 在论文中主要作为**分析性"跳板"**出现——它们阐明了 attention 效率设计空间中的关键权衡维度（attention sharing vs value sharing vs KV compression），为 GTA 的设计提供了动机。论文未将这些变体作为独立方法发表或开源；实际使用中，工程师可根据具体设备的 memory/compute 瓶颈选择 GVA（memory-rich 场景，V 可保持独立）、GHA（memory-constrained 场景，但 attention 表达力有损）或 GTA（平衡方案，引入 latent + nonlinear decoder）。

涉及论文标题：
- GTA__Grouped-head_latenT_Attention

---

## Multi-head Latent Attention (MLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Multi-head Latent Attention (MLA) 是 DeepSeek-V2/V3 中提出的一种低秩 KV cache 压缩注意力机制。其核心思想是将 Multi-Head Attention 中的 Key 和 Value 投影到低维 latent space 进行压缩存储，在 attention 计算时通过 up-projection 解压回每个 head 的 K/V。

MLA 的具体流程：(1) **Low-rank joint compression**：输入 hidden state h_t 通过下投影矩阵 W_{DKV} ∈ R^{d_c×H} 压缩为低维 joint latent vector c_t^{KV} ∈ R^{d_c}（如 d_c=512），仅存储该 latent vector 作为 KV cache；(2) **Decompressed Key**：对每个 head i，通过 up-projection k_{t,i}^C = W_{UK,i} c_t^{KV} 解压出 no-positional 部分的 key；(3) **Decoupled RoPE**：额外生成共享的 RoPE key k_t^R = RoPE(W_{KR} h_t)，与内容 key 拼接 k_{t,i} = [k_{t,i}^C; k_t^R]；(4) **Decompressed Value**：v_{t,i}^C = W_{UV,i} c_t^{KV}；(5) **注意力计算**：每个 head 独立的 QK^T 和 V 加权求和，与 MHA 相同（无 attention sharing）。MLC 的 KV cache 仅需存储 (d_c + d_{rope}) 维/token/layer，相比 MHA 的 2n_h d_h 大幅压缩（DeepSeek-V3 约 85× 压缩）。

MLA 与 GTA 的关键区别：(a) MLA 仍为每个 head 计算独立 attention scores（n_h 次 QK^T），而 GTA 共享 attention map（n_q 次 QK^T）；(b) MLA 的 value 解压是纯线性（up-projection from c^{KV}），而 GTA 使用非线性 sigmoid gate 调制；(c) MLA 的 latent vector 同时压缩 K 和 V，GTA 的 latent 仅压缩 V，key 通过共享机制压缩；(d) MLA 有额外的 decode 时 up-projection 开销（per-head 从 latent 解压 K/V），GTA 通过 Eq 8 reformulation 将这部分消去。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**MLA 张量计算流程（以 DeepSeek-V3 典型配置为例：H=7168, n_h=128, d_c=512, d_nope=128, d_rope=64, d_h=d_nope+d_rope=192）：**

```
# === Prefill (N tokens) ===

# 1. Low-rank KV compression
c_KV = X @ W_DKV^T              # W_DKV ∈ R^{512×7168}, c_KV ∈ R^{N×512}

# 2. RoPE Key (shared across heads)
K_R = RoPE(X @ W_KR^T)          # W_KR ∈ R^{64×7168}, K_R ∈ R^{N×64}

# 3. 逐 head 解压 Q/K/V
for i in 0..127:
    # Query (split into nope + rope)
    Q_i_C = X @ W_{Q,i}^T         # W_Q ∈ R^{128×7168}
    Q_i_R = RoPE(X @ W_{QR,i}^T)  # W_QR ∈ R^{64×7168}
    Q_i = concat(Q_i_C, Q_i_R)    # (N, 192)
    
    # Key (nope from latent, rope shared)
    K_i_C = c_KV @ W_{UK,i}^T     # W_UK ∈ R^{128×512}
    K_i = concat(K_i_C, K_R)      # (N, 192)
    
    # Value (from latent)
    V_i = c_KV @ W_{UV,i}^T       # W_UV ∈ R^{128×512}

# 4. 每个 head 独立 attention (n_h=128 次)
for i in 0..127:
    S_i = Q_i @ K_i^T / sqrt(192) # (N, N)
    A_i = softmax(S_i)
    O_i = A_i @ V_i               # 128 heads 各自独立计算

# === KV Cache ===
# 仅存储: c_KV (512 dims/token) + K_R (64 dims/token)
# 共计 576 dims/token/layer
# vs MHA: 2 × 128 × 192 = 49152 dims/token
# 压缩比: ~85×

# === Decode (1 new token) ===
# 追加 c_KV_new (1,512) 和 K_R_new (1,64) 到 cache
# 读取 full cache c_KV_all + K_R_all
# 解压 all keys (per head): K_i_C_all = c_KV_all @ W_{UK,i}^T  # O(n_h d_c d_nope N) per decode
# 解压 all values (per head): V_i_all = c_KV_all @ W_{UV,i}^T  # 同上
# 每个 head 独立计算 QK^T 和 attention output
```

**MLA vs GTA 计算复杂度对比（论文 Table 4）：**

| 指标 | MLA | GTA |
|------|-----|-----|
| KV cache | (d_c + d_{rope})N | (n_k d_h + n_c d_l)N |
| Attention FLOPs | n_h(d_{rope} + 2d_{nope})N^2 | n_q(d_h + d_l)N^2 |
| Prefill 线性项 | (d_c+d_{rope}+n_h(d_{nope}+d_{rope}))NH + 2n_h d_c d_{nope}N + H^2 N | 2NH^2 + (n_q d_h + n_k d_h + n_c d_h + d_l)NH |
| Decode 额外开销 | 从 latent 解压所有 K/V（O(n_h d_c d_{nope} N)） | 无解压步骤（Eq 8 fusion） |

术语一般如何实现？如何使用？

MLA 的开源实现主要见于：(a) DeepSeek-V2/V3 官方代码；(b) vLLM/SGLang 中支持 MLA 的推理优化 kernel；(c) FlashMLA（针对 MLA 优化的 FlashAttention 变体）；(d) TransMLA 项目（https://github.com/MuLabPKU/TransMLA）可将 GQA-based 模型转换为 MLA 格式。

MLA 适用于长上下文大模型部署（DeepSeek-V3 使用 MLA 支持 128K 上下文），特别适合显存受限但计算能力充裕的硬件平台。MLA 的 decode 阶段需要 per-step 解压所有历史 KV，在超长上下文（>128K）时该线性开销可能成为瓶颈——这是 GTA 试图解决的 MLA 核心弱点之一。

涉及论文标题：
- GTA__Grouped-head_latenT_Attention

涉及论文标题：
- FastKV: KV Cache Compression for Fast Long-Context Processing with Token-Selective Propagation

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding

**补充（来自 Hardware-Efficient Attention for Fast Decoding）**：MLA 的一个关键架构缺陷是 KV cache 在 Tensor Parallelism (TP) 下的**全设备复制问题**。由于 MLA 缓存单头 latent c^{KV}（d_c=4d_h），而 up-projection 矩阵 W^{UK}, W^{UV} ∈ R^{(4d_h)×(h_q d_h)} 按列并行切分到 TP rank，每个 rank 需要完整 latent 来重建其负责 head 的 K/V。因此 TP=2 时每 device KV cache 仍为 4d_h（与单卡相同），TP=4 仍为 4d_h——TP 完全无法减少 MLA 的 per-device 内存。相比之下，GQA（h_kv=8）在 TP=2 时从 16d_h 降至 8d_h，GLA（h_c=2）从 4d_h 降至 2d_h。MLA 通过混合 TP+DP 缓解此问题（attention 子模块跨 DP group 复制），但引入 DP barrier straggler 效应。

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding
- Multi-head_Temporal_Latent_Attention

**补充（来自 X-EcoMLA）**：MLA 可通过 Attention Upcycling 从预训练 MHA/GQA 模型后训练转换而来，无需从零预训练。X-EcoMLA 证明了：(1) 通过 SVD 初始化 MLA 权重（从预训练 W^Q、W^K、W^V 提取低秩结构），训练开始时已接近原始性能；(2) 通过知识蒸馏（KL 散度）+ DPO 两阶段训练，仅需 3.6B-7B tokens 即可实现同等或更好的性能；(3) 统一共享 RoPE Key 设计（所有 head 共享一个 K^R 向量）相比 per-head RoPE 在固定维度预算下提供 n_h× 的位置编码容量；(4) 在 Llama3.2-1B 上使用 8B teacher 可实现 6.4× KV 压缩（15.6% KV size）且零精度损失，或 10.6× 压缩（9.4% KV size）仅 <0.1% 平均分下降。训练成本仅 70-140 GPU hours on AMD MI300，相比预训练节省 >5000×。

涉及论文标题：
- X-EcoMLA__Upcycling_Pre-Trained_Attention_into_MLA_for_Efficient_and_Extreme_KV_Compression

---

## Learning-to-Hash for Top-k Attention in LLMs (面向LLM Top-k注意力的学习哈希)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Learning-to-Hash for Top-k Attention 是一种将学习式哈希（learning-to-hash）技术集成到LLM top-k attention过程中的方法。与传统的Locality-Sensitive Hashing（LSH，基于随机投影）不同，learning-to-hash通过可训练的hash函数h(x)=2·Sigmoid(σ·xW_H)-1将高维query/key向量映射为紧凑的二进制hash codes（如128-bit），使得相似向量被赋予Hamming距离小的hash codes。

核心逻辑链：(1) Top-k attention只需知道哪些keys与当前query最相关（序数比较），而非精确的qk score数值；(2) learning-to-hash通过优化min Σs_i||h(q)-h(k_i)||² + balance/uncorrelation约束，将连续向量空间的相似性保持映射到Hamming空间；(3) 在推理时，仅需bitwise_xor+popc计算Hamming距离（O(s×rbit/32)），选出top-k最近keys进行sparse attention。

HATA（ACL 2025 Findings）是该方法的代表性工作，训练数据由prefill阶段的qk pairs采样构建（top 10%为正样本，线性衰减标签[1,20]；90%为负样本，标签-1），每attention head独立训练hash权重W_H∈R^{d×rbit}。与MagicPIG（LSH-based, 1500-bit）相比，HATA仅需128-bit的learned hash codes即可达到near-lossless精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Learning-to-Hash Training Pipeline（HATA）：**

```
# === Training Data Construction ===
for each sequence in calibration set:
    Q, K = prefill(sequence)                  # [n, d]
    for head in 1..H:
        m = random(n/2, n)                    # sample query position
        Score = Q[m] @ K[:m]^T                # causal qk scores
        top10 = TopK(Score, 0.1*m)
        labels = -1 * ones(m)                  # negative samples
        labels[top10] = linspace(20, 1)        # positive: 20→1 decay
        store(q_m, K[:m], labels)

# === Per-Head Hash Training (SGD) ===
W_H = init(d, 128)                             # rbit=128
for epoch in 1..15:
    for batch in training_chunks(32K):
        h_q = 2 * Sigmoid(0.1 * q @ W_H) - 1  # relaxed hash
        h_k = 2 * Sigmoid(0.1 * k_batch @ W_H) - 1
        loss = 0.01 * Σ s_i * ||h_q - h_k_i||²    # similarity
             + 2.0 * ||Σ h_k_i||²                  # bits balance
             + 1.0 * ||W_H^T @ W_H - I||            # uncorrelation
        W_H = SGD(lr=0.1, momentum=0.9)(loss)
```

**推理时的Hash-based Key Retrieval：**
```
# HATA Decode (per head)
Q_H = BitPack(Sign(Q @ W_H))                 # [1, 4] INT32
S = bitcount(bitwise_xor(Q_H, K_H_cache))     # Hamming distances [1, s]
Idx = TopK(-S, N)                             # smallest distances = most similar
O = FlashAttention(Q, K_cache[Idx], V_cache[Idx])
```

术语一般如何实现？如何使用？

HATA开源在https://github.com/gpzlx1/HATA。Hash权重W_H离线训练后作为固定模型参数加载（训练集150K-300K qk pairs）。每head独立训练W_H，支持MHA（32 heads → 32个W_H）和GQA（8 KV heads → 8个W_H per layer）。适用于长上下文（≥32K）或大batch LLM推理加速，与KVCache compression/offloading方法正交可组合。前两层保留vanilla attention（attention outlier layers）。

涉及论文标题：
- HATA__Trainable_and_Hardware-Efficient_Hash-Aware_Top-k_Attention_for_Scalable_Large_Model_Inference

---

## HashEncode for LLM Attention (面向LLM注意力的哈希编码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

HashEncode是HATA中定义的哈希编码函数，将d维query或key向量编码为rbit-bit的二进制hash code，并以packed integer格式存储。其计算过程为：V_H = BitPack(Sign(V @ W_H))，其中W_H∈R^{d×rbit}是经过训练得到的hash权重矩阵。

HashEncode的三个步骤：(1) MatMul：V @ W_H将输入向量从d维投影到rbit维；(2) Sign：对投影结果逐元素取符号函数sign(x)∈{-1,1}，得到rbit个二进制位；(3) BitPack：将rbit个二进制位打包为rbit/32个INT32整数（如128-bit → 4 INT32），便于在GPU上高效存储和计算。

HashEncode的复杂度为O(s×d×rbit)，其中s是序列长度。由于rbit=128远小于s（典型值s≥32K），额外prefill overhead<1%。

从算法pipeline角度拆解术语：

```
# HashEncode (Algorithm 2 in HATA)
Input:  V ∈ R^{s×d}         # s tokens, d-dim each
Param:  W_H ∈ R^{d×rbit}    # trained hash weights, rbit=128

Step 1: V_H = V @ W_H       # [s, d] × [d, 128] → [s, 128], float
Step 2: V_H = Sign(V_H)     # [s, 128], values ∈ {-1, 1}
Step 3: V_H = BitPack(V_H)  # [s, 4], 128 bits → 4 INT32
Output: V_H ∈ N^{s×4}       # compact hash codes

# BitPack detail (128-bit → 4 INT32):
# binary_code = [b0, b1, ..., b127] where b_i = 1 if V_H[i] > 0 else 0
# packed[0] = Σ_{i=0}^{31} b_i * 2^i     (INT32, bits 0-31)
# packed[1] = Σ_{i=32}^{63} b_i * 2^{i-32} (INT32, bits 32-63)
# packed[2] = Σ_{i=64}^{95} b_i * 2^{i-64} (INT32, bits 64-95)
# packed[3] = Σ_{i=96}^{127} b_i * 2^{i-96} (INT32, bits 96-127)
```

HashEncode在prefill阶段编码所有keys并缓存K_H_cache；在decode阶段每步编码新query和新key，仅需O(d×rbit)计算。

术语一般如何实现？如何使用？

在HATA实现中（https://github.com/gpzlx1/HATA），HashEncode的MatMul在GPU Tensor Cores上执行（cuBLAS或custom CUDA kernel），Sign和BitPack通过fused CUDA kernel完成（kernelfusion for hash encoding）。推理时W_H作为固定权重加载。与FlashAttention-2和FlashInfer框架兼容。适用场景：任何需要快速query-key相似度比较的长上下文LLM推理任务。

涉及论文标题：
- HATA__Trainable_and_Hardware-Efficient_Hash-Aware_Top-k_Attention_for_Scalable_Large_Model_Inference

## Token-wise Sparse Indexer (DSA Indexer / DeepSeek Sparse Attention Indexer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token-wise Sparse Indexer 是 DeepSeek Sparse Attention (DSA) 中的核心组件，用于在 long-context 场景下为每个 query token 从全前缀中选择 top-k 个最相关的 key tokens。DSA 首次在 DeepSeek-V3.2 (DeepSeek-AI, 2025) 中被采用，后续也被 GLM-5 等模型采用。

Indexer 维持轻量级 indexing keys k_s^I ∈ R^d、indexing queries q_{t,j}^I ∈ R^d（共 H^I 个 indexing heads）和 per-head gating weights w_{t,j}^I。对于 query position t 和 key position s，relevance score 定义：

$$I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}^I \cdot \text{ReLU}\left(\mathbf{q}_{t,j}^I \cdot \mathbf{k}_s^I\right)$$

Indexer 对全前缀 L 个 token 逐一打分后，取 top-k token 索引集 T_t = TopK(I_{t,:}, k)，送入下游 Sparse MLA 执行稀疏注意力计算。

核心矛盾：下游 Sparse MLA 仅需在 k 个 token 上计算 attention（O(Lk)），但 indexer 需扫描全前缀 L 个 token 打分（per-query O(L)，per-layer O(L²)）。在超长上下文（128K-1M tokens）下，indexer 从可忽略开销变为主导瓶颈。

从算法pipeline角度拆解术语。

**DSA Indexer + Sparse MLA 完整 pipeline**：

```
输入: 第 l 层 hidden states h ∈ R^{L×d}
      轻量 indexing weights W_Q^I, W_K^I, w_gate

// Step 1: 计算 indexing queries 和 keys
q_{t,j}^I = h_t @ W_Q^I[:, j, :]          // [L, H^I, d_head]
k_s^I = h_s @ W_K^I                        // [L, d_head]（所有 query heads 共享）

// Step 2: Token-wise scoring（瓶颈）
for t = 1 to L:                            // 每个 query position
    for s = 1 to t:                        // causal: 仅前向 token
        I_{t,s} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)
    T_t = TopK(I_{t,:}, k)                 // 选出 top-k token 索引

// Step 3: Sparse MLA（仅在 T_t 中的 token 上计算）
for t = 1 to L:
    c_selected = {c_s | s ∈ T_t}           // 从 KV latent cache 中 gather
    u_t = Attn(h_t, c_selected)            // 稀疏注意力
```

复杂度：Step 2 per-layer O(L²)，Step 3 per-layer O(Lk)。当 L=64K, k=2048 时，indexer 占主导（~5.6 ms vs Sparse MLA ~1.6 ms at A100）。

术语一般如何实现？如何使用？

Indexer 使用独立的轻量 indexing heads（通常 H^I 远小于主 attention heads），通过 TileLang 或 CUDA kernel 实现高效的 token-level matmul + ReLU + gating + TopK。DeepSeek-V3.2 的开源参考实现在 TileLang 仓库（https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32），高性能 CUDA kernel 在 DeepGEMM 和 FlashMLA 中。

在 HISA 论文中，Token-wise Sparse Indexer 被 HISA 层级索引器替换——HISA 将 flat token scan 改写为 block-level 粗过滤 + token-level 精筛两阶段，保留相同的 token-level scoring 公式但仅在候选 block 内执行。

涉及论文标题：
- HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

---

## Sparse MLA (Sparse Multi-Head Latent Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Sparse MLA 是 DeepSeek-V3.2 中采用的稀疏注意力算子，是 MLA (Multi-Head Latent Attention, DeepSeek-V2, 2024) 的稀疏变体。MLA 的核心设计是将传统 MHA 的 Key 和 Value 投影压缩为低维 latent 向量 c_s ∈ R^{d_c}（d_c ≪ d，通过低秩分解 W^{KV} = W^{UK} W^{DK} 实现），大幅减少 KV cache 内存占用。

Sparse MLA 采用 MQA (Multi-Query Attention) mode：每个 token 存储单个 latent key-value entry c_s，由所有 query heads 共享。给定 indexer 选择的 token 集 T_t（|T_t| = k），Sparse MLA 仅在这些 token 上计算 attention：

$$\mathbf{u}_t = \operatorname{Attn}(\mathbf{h}_t, \{\mathbf{c}_s \mid s \in \mathcal{T}_t\})$$

从算法pipeline角度拆解术语。

```
// Sparse MLA 计算流程（per layer, per token t）
输入: h_t ∈ R^d（当前 token hidden state）
      latent KV cache C = {c_s ∈ R^{d_c} | s = 1..L}
      selected indices T_t = {i_1, ..., i_k}

// Step 1: Query projection（标准流程）
q_t = h_t @ W_Q                              // [d] → [H, d_head]
q_t = RoPE(q_t)                              // 位置编码

// Step 2: 从 latent cache 中 gather 选中的 KV
C_selected = gather(C, T_t)                  // [k, d_c]

// Step 3: Up-projection 到完整维度（MLA 特有）
K_selected = C_selected @ W_UK               // [k, d_c] → [k, d]
V_selected = C_selected @ W_UV               // [k, d_c] → [k, d]

// Step 4: Sparse attention（仅 k 个 token）
scores = q_t @ K_selected^T / sqrt(d_head)   // [H, k]
weights = softmax(scores, dim=-1)            // [H, k]
o_t = weights @ V_selected                   // [H, d_head]
u_t = o_t @ W_O                               // [H·d_head] → [d]
```

与 Dense MLA 的区别：Dense MLA 对所有 L 个 token 计算 attention（O(L)），Sparse MLA 仅在 T_t 中的 k 个 token 上计算（O(k)）。当 k ≪ L 时（如 k=2048 vs L=64K），Sparse MLA 大幅节省 attention 计算。

术语一般如何实现？如何使用？

Sparse MLA 是 DeepSeek-V3.2 和 GLM-5 等模型的标准 attention 机制。在 vLLM 等 serving 框架中，MLA 的 latent KV cache 以 FP8 精度存储以减少内存占用。Sparse MLA 的输出接口是 token 索引集 T_t——只要 indexer 产生正确格式的 T_t，Sparse MLA 无需任何修改。这是 HISA 能够作为"即插即用" indexer 替代品的关键：HISA 产生的 T_t 与 DSA indexer 完全同构，下游 Sparse MLA 保持不变。

涉及论文标题：
- HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

---

## Hierarchical Indexed Sparse Attention (HISA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

HISA 是一种免训练的即插即用式层级索引策略，用于替代 DSA 中 O(L²) 复杂度的 flat token scan indexer。HISA 将索引搜索路径从"全前缀逐 token 扫描"改写为"block 粗过滤 → token 精筛"两阶段层级过程：

- **Stage 1 (Block-level Coarse Filtering)**：将前缀分为 M = ⌈L/B⌉ 个连续 block，每 block 用 mean pooling 生成代表向量 k̃_b^I。Query 对所有 M 个 block 代表打分，选 top-m blocks。复杂度 O(L/B)。
- **Stage 2 (Token-level Refinement)**：在候选 block 内的至多 mB 个 token 上，使用与 DSA 相同的 token-level scoring 公式逐 token 打分，选最终 top-k tokens。复杂度 O(mB)。

总复杂度 per-query: O(L/B + mB)，per-layer: O(L²/B + LmB)。当 m ≪ M 且 B ≪ L 时（超长上下文 + 选择性粗过滤），缩减显著。

HISA 的三个关键设计决策：(1) **首尾 block 强制保留**：处理 attention sink 和局部上下文；(2) **候选池过采样**：mB > k（如 mB=8192, k=2048），保证精筛质量；(3) **输出同构**：T_t 格式与 DSA indexer 完全相同，Sparse MLA 无需任何修改。

从算法pipeline角度拆解术语。

**HISA 两阶段层级索引（Algorithm 1）**：

```
输入: indexing queries q_{t,j}^I, gating weights w_{t,j}^I,
      token indexing keys {k_s^I}_{s=1}^L, block size B, block budget m, token budget k

// Stage 0: Block 划分与 Pooling（增量维护在 KV cache 旁）
M = ceil(L / B)
for b = 1 to M:
    k̃_b^I = MeanPool({k_s^I | s ∈ B_b})

// 对每个 query position t
for t = 1 to L:
    // Stage 1: Block-level 粗过滤
    for b = 1 to M such that B_b precedes t:     // causal
        J_{t,b} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k̃_b^I)
    C_t = TopK(J_{t,:}, m) ∪ {first block, last block}
    Ω_t = ∪_{b ∈ C_t} B_b                         // |Ω_t| ≤ mB

    // Stage 2: Token-level 精筛（与 DSA 公式(1)相同）
    for s ∈ Ω_t:
        I_{t,s} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)
    T_t^H = TopK({I_{t,s} | s ∈ Ω_t}, k)          // 同构输出

// T_t^H 直接送入 Sparse MLA（与 DSA 完全相同）
```

**三 regime 边界行为**：
- t ≤ k: 等价 dense attention（全选）
- k < t ≤ mB: 等价 DSA（粗过滤全选，精筛至 k）
- t > mB: HISA 层级优势激活（非平凡 block 剪枝）

默认超参数：B=128, m=64 (candidate 8192), k=2048。64K context 下 kernel 加速 2.16×-3.75× vs DSA indexer。

术语一般如何实现？如何使用？

HISA 作为 DSA indexer 的 drop-in replacement，直接替换 DeepSeek-V3.2 和 GLM-5 的 indexer 模块：
1. Block pooled keys 增量维护在 KV cache 旁，额外开销可忽略
2. Stage 1 在 TileLang 上实现为 block-level matmul kernel（M ≪ L）
3. Stage 2 在 TileLang 上实现为 token-level matmul kernel（仅在 Ω_t 上）
4. 输出 T_t^H 送入原始的 Sparse MLA 算子

代码仓库：https://github.com/MuLabPKU/TransArch（截至当前 HISA 代码标记为待发布）。

涉及论文标题：
- HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

## Grouped-Tied Attention (GTA / 分组绑定注意力) [Zadouri et al., 2025]

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Grouped-Tied Attention (GTA) 是 Zadouri、Strauss 和 Dao（Princeton University, 2025）提出的一种硬件高效注意力变体。核心思想是将 GQA（Grouped-Query Attention）中的独立 Key 和 Value 投影**绑定为单一状态**（tied KV state），将 KV cache 大小减半、算术强度翻倍，同时保持与 GQA 相当或更优的模型质量。

三个关键设计要素：(1) **KV Tying**：单一投影矩阵 W^{KV} 替代独立的 W^K 和 W^V，输出 shared *tied KV* 向量；(2) **Partial RoPE**：仅前半 head 维度（d_h/2）作为无位置编码的 content key（K_NoPE），后半来自独立的单头 RoPE 投影（K_RoPE，跨所有 KV head 广播）；(3) **Full value dimension**：value 路径使用 tied KV 的完整 d_h 维，保证 value 表达力不受损。

KV cache per token：hkv × 1.5 × d_h（含 d_h/2 广播 RoPE），vs GQA 的 hkv × 2 × d_h。算术强度约 2gq vs GQA 的 ~gq。

**注意**：本 GTA 与 Sun et al., 2025 的 "Grouped-head latenT Attention (GTA)" 是**不同方法**，仅共享缩写。Sun 的 GTA 使用共享 attention map + 非线性 value decoder；本 GTA 使用 KV 绑定 + 部分 RoPE。

从算法pipeline角度拆解术语，给出具体例子。

```
# GTA-4: hq=16, hkv=4, gq=4, dh=128, d_R=64
# 输入: X ∈ R^{B×L×d}

# 1. Q 投影（标准）
Q = X @ W^Q              # [B, L, 16, 128]

# 2. Tied KV 投影 —— 单一投影替代 W^K + W^V
KV = X @ W^{KV}          # [B, L, 4, 128] —— tied state

# 3. 构造 K 和 V
V = KV                   # value 用完整维度
K_NoPE = KV[:,:,:,:64]  # 前半维度，不加 RoPE
K_RoPE = apply_rope(X @ W^{K_RoPE})  # [B, L, 1, 64]
K_RoPE = broadcast(K_RoPE, 4)        # 广播到 4 个 KV head
K = concat([K_NoPE, K_RoPE], dim=-1)  # [B, L, 4, 128]

# 4. GQA-style attention
for g in 0..3:
    Q_g = Q[:, :, g*4:g*4+4, :]      # 4 query heads
    attn[g] = softmax(Q_g @ K_g^T / 8) @ V_g
```

**KV cache 对比（XL 1.471B, hq=16, hkv=4, dh=128, BF16）**：

| 方法 | bytes/token TP=1 | bytes/token TP=2 | bytes/token TP=4 |
|------|------------------|------------------|------------------|
| GQA-4 | 2048 | 1024 | 512 |
| GTA-4 | 1152 | 640 | 384 |
| MLA | 1152 | 1152 | 1152 |
| GLA-2 | 1152 | 640 | 640 |

GTA 在低 TP 度下比 GQA 节省最多（TP=2: 640 vs 1024），且随 TP 增加持续缩小 per-device cache。

术语一般如何实现？如何使用？

在 PyTorch 中替换标准 MHA 模块的 W_K 和 W_V 为 W^{KV}（输出 hkv×dh）+ W^{K_RoPE}（输出 1×dh/2）。训练配置与 baseline 一致（AdamW, cosine LR, FineWeb-Edu-100B），各 variant 通过加宽 FFN 匹配 MHA 参数总量。推理 kernel 开源：https://github.com/Dao-AILab/grouped-latent-attention，包含 warp specialization + software pipelining 优化。

适用场景：需要比 GQA 更小 KV cache、但保持分片能力的低 TP 度分布式推理。

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding

---

## Grouped Latent Attention (GLA / 分组潜在注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Grouped Latent Attention (GLA) 是 Zadouri、Strauss 和 Dao（Princeton, 2025）提出的并行友好 latent attention 变体，是对 MLA 的核心改进。GLA 将 MLA 的单头 latent compression 扩展为**多头** latent compression（h_c 个 latent head，每 head d_c=2d_h），使 latent head 可在 TP rank 间分片，解决 MLA 的 KV cache 跨设备全复制问题。

MLA 的单头 latent（d_c=4d_h）因 TP 按 head dimension 切分 up-projection 矩阵，每 rank 必须持有完整 latent——TP 不减少 MLA 的 per-device KV cache。GLA 通过多 latent head 分组设计：每 latent head 仅服务于一组 query head，该组的上投影矩阵完整存于对应 rank，无需复制 latent。

以 GLA-2（h_c=2, d_c=2d_h）为例：总 KV cache = 4d_h（与 MLA 相同），但 TP=2 时每 device 仅 2d_h（MLA 仍为 4d_h）。算术强度约 2gq（双倍于 GQA），与 MQA 的 h_q 相当但质量远超。

从算法pipeline角度拆解，给出具体例子。

```
# GLA-2: hq=16, h_c=2, gq=8, dh=128, d_c=256, d_R=32

# === 训练时: Down + Up ===
c_0^{KV} = X @ W^{DKV}_0   # [B, L, 256]
c_1^{KV} = X @ W^{DKV}_1   # [B, L, 256]
K_0 = c_0^{KV} @ W^{UK}_0   # per-group K
V_0 = c_0^{KV} @ W^{UV}_0   # per-group V

# === 解码时: Weight Absorption ===
# W^{UK} 吸收进 W^Q, W^{UV} 吸收进 W^O
Q_0 = X @ W^Q_absorbed_0   # [B, 1, 8, 256]
Q_1 = X @ W^Q_absorbed_1   # [B, 1, 8, 256]
O_0 = softmax(Q_0 @ (c_0^{KV})^T / 16) @ c_0^{KV}
O_1 = softmax(Q_1 @ (c_1^{KV})^T / 16) @ c_1^{KV}

# === 分布式 (TP=2) ===
# Rank 0: c_0^{KV}, Q_0, W^{VO}_0 → O_0 @ W^{VO}_0
# Rank 1: c_1^{KV}, Q_1, W^{VO}_1 → O_1 @ W^{VO}_1
O = AllReduce(O_0 @ W^{VO}_0 + O_1 @ W^{VO}_1)
```

**Serving benchmark 关键结果（DeepSeek-Coder-V2 236B FP8, 8×H100）**：
- GLA-8 TP=8 vs MLA TP=8（64 并发, 8K/4K）：throughput 1461 vs 859 tok/s（+70%），E2E latency 179s vs 381s（-53%）
- GLA-8 TP=8 vs MLA TP=2+DP=4（131K/4K 不平衡负载）：throughput 100 vs 37 tok/s（+2.7×）
- GLA kernel L_q=2（推测解码）：2× faster than FlashMLA

术语一般如何实现？如何使用？

开源实现：https://github.com/Dao-AILab/grouped-latent-attention。包含完整 CUDA kernel（warp specialization + software pipelining + distributed offset calculation）、PyTorch 模型定义和 SGLang serving 集成。

适用场景：(a) 大规模 distributed inference（TP≥2），每 device KV cache 是核心瓶颈；(b) 推测解码（L_q>1），GLA 算术强度更高；(c) 混合负载（不均匀序列长度），避免 DP straggler。

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding

---

## Arithmetic Intensity in Attention Decoding (注意力解码中的算术强度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

算术强度（Arithmetic Intensity, Williams et al., 2009）定义为每字节内存访问所执行的浮点运算次数（FLOPs/Byte），是 Roofline 性能模型的核心度量——判断 workload 是 memory-bound（算术强度低于硬件 roofline 拐点）还是 compute-bound（算术强度高于拐点）。

在 LLM 解码中，标准 MHA 的算术强度极低（约 1 FLOP/byte）：每从 HBM 加载的 BF16 KV 元素（2 bytes）仅执行 1 次 MAC（2 FLOPs），而 H100 的临界算术强度为 989 TFLOPS / 3350 GB/s ≈ 295 FLOPs/byte——MHA 解码时 GPU 利用率可低至 7%（Recasens et al., 2025）。

该论文推导了通用算术强度公式：
$$\text{AI} \approx \frac{2 \cdot L}{2 + \frac{m_{kv}}{g_q} \cdot L} \approx \frac{2g_q}{m_{kv}} \quad (L \gg h_q)$$

其中 g_q = h_q / h_{kv}（group size），m_{kv} ∈ {1,2}（1=共享 KV state，2=分离 K/V），L 为序列长度。

从算法pipeline角度拆解，给出具体例子。

**提升算术强度的三个独立维度**：

1. **增加 g_q**（更大 group size）：更多 query head 共享一个 KV head → 同一 KB 加载服务更多 FLOPs
   - MQA: g_q = h_q, AI ≈ h_q
   - GQA-4: g_q = 4, AI ≈ 4
2. **减少 m_kv**（KV tying, m_kv: 2→1）：分离 K/V → 共享 state → 内存加载减半 → AI 翻倍
   - GQA: m_kv=2, AI ≈ g_q
   - GTA: m_kv=1, AI ≈ 2g_q
3. **增加 h_q**（更多 query heads）：MLA/GLA 通过低秩参数再分配在保持 latent 维度不变的前提下增加 query head 数
   - MLA: h_q=128, AI ≈ 256（接近 H100 compute roof）

**Roofline 分析（H100 BF16, L_q=1 vs L_q=2）：**

| 方法 | AI (L_q=1) | AI (L_q=2) | 状态 (L_q=1) | 状态 (L_q=2) |
|------|-----------|-----------|-------------|-------------|
| MHA (hq=128) | ~1 | ~1 | Memory-bound | Memory-bound |
| GQA-4 (hq=128) | ~4 | ~4 | Memory-bound | Memory-bound |
| MQA (hq=128) | ~128 | ~128 | Bandwidth roof | Bandwidth roof |
| GLA-2 (hq=128) | ~128 | ~128 | Bandwidth roof | **Compute inflection** |
| MLA (hq=128) | ~256 | ~256 | Near compute roof | Beyond compute roof |

术语一般如何实现？如何使用？

实践中指导 attention 变体选择：目标 hardware roofline 拐点决定了理想算术强度。H100 上 L_q=1 解码时，MLA（AI≈256）接近 compute roof→更高效的 compute 利用；GLA-2（AI≈128）在 bandwidth roof→内存带宽利用更高（93% vs FlashMLA 72%）。L_q=2（推测解码）时，GLA-2 达 compute inflection point→2× faster than FlashMLA。算术强度分析是 GTA 和 GLA 设计的核心指导原则。

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding

---

## Decoupled Rotary Position Encoding (解耦旋转位置编码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

解耦 RoPE（Decoupled Rotary Position Encoding）是 MLA 和 GLA 中处理位置信息的关键技术。传统 RoPE（Su et al., 2023）直接对每个 head 的完整 K/Q 维度做旋转变换。但 MLA/GLA 的 latent compression 导致 K 被压缩为低维 latent vector——若直接在 latent 上加 RoPE，则 weight absorption 技巧（将 up-projection 矩阵吸收进 Q/O 投影）失效，因 RoPE 旋转矩阵与 up-projection 矩阵不满足交换律。

解耦方案：(1) Key 拆分为内容部分（来自 latent compression，不加 RoPE）和位置部分（额外单独投影 + RoPE，维度 d_R 通常远小于 d_h）；(2) Q 同样拆分；(3) Attention score 由两部分点积求和：content QK^T + positional Q_R K_R^T。位置部分 K_R 通常跨所有 head 共享（单头），进一步减少 KV cache。

从算法pipeline角度拆解，给出具体例子。

```
# MLA/GLA 的解耦 RoPE 张量计算
X ∈ R^{B×L×d}

# Content Key（来自 latent，不加 RoPE，可做 weight absorption）
K_C = c^{KV} @ W^{UK}         # [B, L, h_kv, d_nope]

# Positional Key（额外投影 + RoPE，通常单头跨 head 共享）
K_R = X @ W^{KR}              # [B, L, 1, d_R]
K_R = apply_rope(K_R)

# Query 同样拆分
Q_C = X @ W^{QC}              # [B, L, h_q, d_nope]
Q_R = X @ W^{QR}              # [B, L, h_q, d_R]
Q_R = apply_rope(Q_R)

# 分开计算后求和
attn = (Q_C @ K_C^T + Q_R @ K_R^T) / sqrt(d_nope + d_R)
```

**GTA 的变体应用**：GTA 不使用 latent compression，也不需 weight absorption，但采用类似的 partial RoPE——仅 half head dimension（d_h/2）加 RoPE 作为位置 key，剩余 half 作为无位置编码的内容 key（K_NoPE），后者与 value 共享 tied KV state。

术语一般如何实现？如何使用？

FlashMLA 和 GLA kernel 中，Positional（RoPE）和 Content（non-RoPE）部分的 attention 分别在 Tensor Cores 上计算后通过 FMA 合并。d_R 典型取 32（约 25-50% d_h），在位置信息保真度和 KV cache 开销间平衡。Cohere 的 Command-R 和 Llama 4 进一步通过仅在部分层应用 RoPE 来减少 d_R 的 KV cache 开销。

**MTLA 的 Temporal Compression of RoPE Keys**：MTLA 进一步将 decoupled RoPE keys 沿 temporal 维压缩——对每 s 个相邻 token 仅保留最新的 RoPE key 到 cache。推理时 j-th slot 的 RoPE key 缓存更新策略：若 i%s==1 则追加 k_i^R；否则用 k_i^R 覆盖当前 slot（ĥ_j^R = k_i^R）。训练时原始 K^R 配合 stride-aware causal mask 直接参与 attention 计算。此压缩不增加额外参数量，进一步减少 64/layer（per d_R=32, s=2, bf16）的 KV cache 开销。

涉及论文标题：
- Hardware-Efficient_Attention_for_Fast_Decoding
- Multi-head_Temporal_Latent_Attention

**补充（来自 X-EcoMLA）**：X-EcoMLA 采用统一共享 RoPE Key 设计（类似 DeepSeek MLA），即所有 attention head 共享单一 K^R ∈ R^{d_r} 向量（而非 per-head 各分配 d_r/n_h 维）。这在不同 head 数下提供 n_h× 的位置编码容量优势——以 8-head 模型为例，per-head RoPE 设计下每 head 仅获 32/8=4 维位置编码，而 X-EcoMLA 的共享设计每 head 获完整 32 维。X-EcoMLA 通过对比 MHA2MLA（per-head RoPE）展示了这一设计对极端 KV 压缩下性能保持的关键作用。K^R 的初始化使用所有 KV head 的 W^K 的平均值（W_K_avg = mean(W_K.view(d, n_kv, d_h), dim=1)），取最后 d_r 列。

涉及论文标题：
- X-EcoMLA__Upcycling_Pre-Trained_Attention_into_MLA_for_Efficient_and_Extreme_KV_Compression

---

## KL Divergence Distillation for KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KL Divergence Distillation for KV Cache Compression 是 KV-Distill 论文提出的核心训练目标，通过 KL 散度直接匹配压缩前后 KV cache 产生的 next-token 概率分布。与传统 auto-encoding loss（重建被压缩 token 本身）不同，KL 散度蒸馏在 token 预测分布层面进行优化，消除了 pretraining-inference mismatch。使用加权组合 L(θ) = λ·D_KL(p||q_θ) + (1-λ)·D_KL(q_θ||p)，其中 p=完整 KV cache 的 next-token 分布（teacher），q_θ=压缩后分布（student）。Forward KL 为 mean-seeking（覆盖所有高概率输出），Reverse KL 为 mode-seeking（集中在高概率区域）。λ=0.6 偏 forward KL，因为 reverse KL 的 L1 梯度主导。

从算法pipeline角度拆解：

```
# Teacher (frozen LM, 完整 KV cache)
p = softmax(LM.decode(X_full))
# Student (LoRA-adapted LM_θ, 压缩 KV cache ˜X)
q_θ = softmax(LM_θ.decode(˜X))
# Loss
L = 0.6 * Σ p·log(p/q_θ) + 0.4 * Σ q_θ·log(q_θ/p)
```

LLAMA-3 SQuAD 20% retention: weighted KL 86.0% vs forward-only 83.4% vs reverse-only 82.7% vs AE+CE 79.1%。

术语一般如何实现？如何使用？

teacher forward 在 torch.no_grad() 下执行，仅需一次完整 KV cache 编码。反向传播仅更新 150M 参数（LoRA adapter + scorer）。训练时随机采样 retention ratio（0.1%-80%），单一模型支持任意压缩率推理。

涉及论文标题：
- KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

**补充（来自 X-EcoMLA）**：X-EcoMLA 使用 KL 散度蒸馏将更大 teacher 模型的知识传递给 upcycle 后的 MLA student 模型，公式为 L_θ = Σ_{t=1}^T KL[p(·|y_{1:t}, x, θ_T) || p(·|y_{1:t}, x, θ)]。与 KV-Distill 的"压缩前后同模型分布匹配"不同，X-EcoMLA 的蒸馏是"跨模型架构的知识迁移"——teacher 可以是与 base model 不同的更大模型（如 8B teacher 指导 1B student）。消融实验（Table 12）表明纯 CE loss 导致性能大幅退化（48.54 vs 52.77），而纯 KL 蒸馏（50.84）或 KL+CE 混合（50.93-50.98）均远优于纯 CE，验证了 teacher dark knowledge 对 MLA upcycling 的关键作用。训练数据为 OpenHermes-2.5 + GenQA + Infinity-Instruct（6.8B tokens），AdamW optimizer with β=(0.9, 0.98)，batch size=96。

涉及论文标题：
- X-EcoMLA__Upcycling_Pre-Trained_Attention_into_MLA_for_Efficient_and_Extreme_KV_Compression

---

## Question-Independent KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Question-Independent KV Cache Compression 是 KV-Distill 定义的两大 KV Cache 压缩范式之一（另一为 Question-Aware）。在问题无关范式中，上下文在不知道具体问题时被压缩为紧凑表示，供后续任意多轮问答复用。问题感知范式可利用问题信号（问题→上下文的 attention）定位关键信息。问题无关压缩更难：必须在不了解未来查询时预测哪些信息值得保留。核心应用场景：固定知识库/长文档压缩一次供反复查询（如 RAG 文档索引压缩）。

从算法pipeline角度拆解：

```
// H2I (问题无关, 无训练): 仅 context self-attention
scores = Σ softmax(Q_ctx @ K_ctx^T)  // context 内部
// LLAMA-3 SQuAD 25%: 56.6%

// KV-Distill (问题无关, 可训练): scorer 学会预测重要性
s = FFN_scorer(hidden_η)  // 从 hidden states 学习
top_k = argtopk(s, k)
X_comp = LM_θ.encode(context, selected=top_k)
// LLAMA-3 SQuAD 25%: 86.6% (vs H2I 56.6%, H2A 84.0%)
```

术语一般如何实现？如何使用？

训练数据拆分为 (Context, Instruction, Answer) 三元组，压缩仅应用于 Context 部分。评估时，context 压缩后与 question 拼接送入模型。长上下文使用 folding 技巧（pad→reshape batch→分别压缩→unfold）。

涉及论文标题：
- KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

---

## Token Importance Scorer for KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token Importance Scorer 是 KV-Distill 中的可训练 FFN，在 prefill 阶段预测每个 token 对后续推理的重要性。输入为 layer η=6 的 hidden states X'_η ∈ R^{N×d}，输出为重要性分数 s ∈ R^N。取 top-k 作为保留 token。top-k 不可微，梯度通过 attention 衰减路径传播：α' = sigmoid(s) ⊙ α，被选 token sigmoid≈1 不变，未选 token sigmoid≈0 被衰减，梯度通过此路径传至 scorer。

从算法pipeline角度拆解：

```
s = FFN_scorer(LM.layer_6_output(context))  # d → d/4 → 1
indices = torch.topk(s, k).indices          # 不可微，梯度通过以下路径:
α' = sigmoid(s) ⊙ α                         # attention weights 衰减
# ∂L/∂s = (∂L/∂α') * α * sigmoid'(s)       # 梯度传播
```

术语一般如何实现？如何使用？

2 层 FFN（中间维度 d/4），约 1-2M 参数。与 LoRA adapter 联合优化（总 150M 参数）。推理时 scorer 仅执行一次，开销可忽略。若 scorer 错误评分，重要 token 信息在压缩 KV cache 中永久丢失。

涉及论文标题：
- KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

---

## Layer-wise KV Cache Sharing (层间KV缓存共享)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Layer-wise KV Cache Sharing 是一种跨 Transformer 层的 KV cache 压缩方法，通过在某些层的推理过程中跳过本层 KV cache 的计算，直接复用（拷贝）其他已计算层的 KV cache，从而减少 KV cache 的计算量和存储占用。与 intra-layer 压缩（在单层内通过 token pruning、量化或 channel shrinking 减少 KV cache）正交，layer-wise 共享从"层间冗余"的角度出发，利用了不同层 KV cache 之间可能存在的可替代性。

KVSharer 论文首次提出了一种无需额外训练的即插即用 layer-wise KV cache 共享方法。该方法基于一个反直觉的发现：共享不相似（dissimilar）的 KV cache 比共享相似的 KV cache 能更好地保持模型性能。这与传统参数共享/注意力共享中"相似度越高共享效果越好"的直觉相反。

从算法pipeline角度拆解术语：

**KVSharer 的 Layer-wise KV Cache Sharing 流程**：

```
// 阶段一：离线策略搜索（Algorithm 1，约60秒/模型）
输入: 预训练LLM M, 目标共享层数 C, 校准数据集 D, 相似度阈值 T
输出: 共享策略 Z

1. 在 D 上运行 M，保存每层 KV cache
2. 将每层 keys 和 values 分别 flatten → 1D向量，取平均作为该层 KV cache 表示
3. 计算任意两层之间的欧氏距离并按降序排列
4. Z ← ∅, P ← 0
5. for each (src, dst) in 降序排列:
       Z ← Z ∪ {(dst层 KV cache ← src层 KV cache)}
       // 输出端被输入端替换（输入端更敏感，不可反向）
       M_tmp ← 应用 Z 的模型
       s ← AvgCosineSim(M_tmp.last_hidden_state, M.last_hidden_state, D)
       if s ≤ T (0.5):  Z ← Z \ {(dst, src)}
       else:            P ← P + 1
       if P == C: return Z

// 阶段二：在线推理
for l in 1..num_layers:
    if l 是被替换层:
        K_cache[l] = K_cache[src_layer]  // 直接拷贝
        V_cache[l] = V_cache[src_layer]
    else:
        K_cache[l], V_cache[l] = compute_KV(l, x)
    output = attention(Q, K_cache[l], V_cache[l])
    output = FFN(output)
```

术语一般如何实现？如何使用？

KVSharer（https://github.com/yangyifei729/KVSharer）以即插即用方式实现，在 HuggingFace Transformers 的 forward pass 中根据预搜索策略 Z 修改每层的 KV cache 计算逻辑。与 FlashAttention 和 GQA/MHA 兼容。压缩率通过 C 控制：12.5%~37.5%。25% 压缩率下保持 >90% 性能。策略搜索仅需约 60 秒（4×A100），一次搜索可通用于所有下游任务。

涉及论文标题：
- KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing
- xKV: Cross-Layer SVD for KV-Cache Compression

**xKV 的 SVD-based Layer-wise KV Cache Sharing**：与 KVSharer 直接拷贝整层 KV-Cache（全等共享）不同，xKV 通过**跨层 SVD** 实现更精细的层间共享——不直接复用某一层的完整 KV-Cache，而是从多层的拼接 KV-Cache 中提取**共享的低秩基 A**，每层保留独立的低秩重构矩阵 B_ℓ_i。这种方式比全等共享更灵活（每层保留其特定信息），比单层独立压缩更高效（共享跨层公共子空间）。xKV 在 8× 压缩比下通过 xKV-4（4 层一组）实现 87.8% avg accuracy（vs KVSharer 类方法通常在 1.2x 压缩就出现明显退化）。

---

## KV Cache Dissimilarity-based Sharing (基于不相似度的KV缓存共享)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Cache Dissimilarity-based Sharing 是 KVSharer 论文提出的反直觉层间共享策略。传统方法基于"共享相似的"直觉，但 KVSharer 发现共享不相似的 KV cache 效果更好。具体通过计算任意两层 KV cache 的欧氏距离（分别 flatten keys 和 values 为 1D 向量后取平均作为该层表示），按距离降序排列优先尝试最不相似的层对。消融实验（Figure 6）证明 dissimilarity-based 的 PPL 显著低于 similarity-based（低 2 倍以上）。

从算法pipeline角度拆解术语：

**不相似度计算**：
```
for l in 1..L:
    // 在校准数据集 D 上收集各层 KV cache
    K_avg[l] = mean_{x∈D}(flatten(K_l(x)))
    V_avg[l] = mean_{x∈D}(flatten(V_l(x)))
    KV_repr[l] = concat(mean(K_avg[l]), mean(V_avg[l]))

// 距离矩阵
for i, j in 1..L:
    S[i][j] = ||KV_repr[i] - KV_repr[j]||_2  // Euclidean

R = argsort_descending(S)  // 距离大 = 不相似 = 优先尝试
```

术语一般如何实现？如何使用？

不相似度计算是策略搜索的前置步骤。校准数据集仅需 30 句 64-token 句子（Wikipedia）。距离矩阵 S 规模为 L×L。后续贪心搜索按 R 的顺序尝试替换并验证 hidden-state 相似度。该策略的关键 insight：不相似的 KV cache 捕获了不同的注意力信息，共享后信息多样性得以保留，而相似的层共享可能导致某些注意力模式完全丢失。

涉及论文标题：
- KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing

---

## LoRA (Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LoRA (Low-Rank Adaptation, Hu et al. ICLR 2022) 是参数高效微调方法。冻结预训练权重 W_0 ∈ R^{d×k}，注入可训练低秩分解 ΔW = BA（B ∈ R^{d×r}, A ∈ R^{r×k}，r << d,k）。前向 h = W_0 x + B A x。推理时融合 ΔW 到 W_0 实现零额外延迟。KV-Distill 使用 rsLoRA (Kalajdzievski, 2023) 变体改善大 rank 稳定性，并引入条件计算路由：被选 token 使用 LoRA W^Q/W^O，未选 token 使用冻结原始权重。

从算法pipeline角度拆解：

```
// KV-Distill 条件路由 LoRA
if token in selected_indices:
    Q = x @ (W_Q + B_Q @ A_Q)   # LoRA-adapted, 聚合丢弃 token 信息
    O = attn @ (W_O + B_O @ A_O)
else:
    Q = x @ W_Q                  # frozen, 其 KV 参与 attention 但最终丢弃
    O = attn @ W_O
K = x @ (W_K + B_K @ A_K)       # 所有 token 的 KV 使用 LoRA
```

术语一般如何实现？如何使用？

HuggingFace PEFT: LoraConfig(r=128, target_modules=["q_proj","k_proj","v_proj","o_proj"])。KV-Distill 额外实现 conditional routing forward 逻辑。推理时可 merge_and_unload() 融合 LoRA 到 base model。

涉及论文标题：
- KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

---

## Query-Agnostic KV Cache Eviction (查询无关的 KV Cache 淘汰)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Query-Agnostic KV Cache Eviction 是一类在 prefill 完成后不依赖任何 query 信息即可决定哪些 KV pairs 保留/淘汰的压缩范式。与 query-aware 方法（如 SnapKV、PyramidKV，利用 trailing window 中的 query token 计算 attention-based 重要性分数）相反，query-agnostic 方法的评分仅基于 context 自身，压缩后的 KV cache 可跨任意 query 复用，无需重复 prefill。

KVzip 是该范式的代表性方法，核心 insight：Transformer 天然作为 encoder-decoder——将 context 编码进 KV pairs（类比 Zip 压缩）；使 LLM 模拟重建原始上下文时，接收高 attention 的 KV pairs 恰好也是下游任务所需的关键信息。DuoAttention 的 context-independent head-level eviction 也属于 query-agnostic 范畴，但 DuoAttention 需数小时 8-GPU 优化 head scores，KVzip 仅需数次 forward pass 一分钟内完成。

从算法pipeline角度拆解术语：

**Query-Agnostic vs Query-Aware 对比**：

```
// === Query-Aware (SnapKV) ===
KV_c = Prefill(context || query_window)  // query 参与 prefill
scores = pool(softmax(Q_query_window @ K_context^T))
KV_compressed = topk_filter(KV_c, scores, budget)
// 问题：压缩 cache 对当前 query 过拟合，新 query 需重新 prefill

// === Query-Agnostic (KVzip) ===
KV_c = Prefill(context)                  // 仅 context，不含 query
input = "Repeat the previous context:" + context
scores = max_cross_attn(Forward(input, use_cache=KV_c))
KV_compressed = topk_filter(KV_c, scores, budget)
// 结果：压缩 cache 可跨任意 query 复用，单次 prefill 服务所有 query
```

术语一般如何实现？如何使用？

适用于 KV cache 可离线准备的场景：个性化对话代理（保留用户指令和对话历史）、企业级预计算文档 KV cache 检索、固定知识库多轮问答等。与 FlashAttention-2 兼容，通过 chunked scoring 扩展到长上下文（O(m·n_c) 线性复杂度）。支持与 KV cache 量化（QServe W8A8KV4）正交集成。代码开源：https://github.com/snu-mllab/KVzip。

LagKV 也是 query-agnostic 范式的一种实现，但其评分机制完全不同：不依赖上下文重建的反向 attention，而是利用 token-wise locality 和 lag-relative 归一化——用下一分区的 K/V 统计量归一化当前分区后计算 channel-wise std 作为重要性分数。这种方法进一步消除了 KVzip 仍需多次 forward pass 的成本，仅需一次 forward 即可完成压缩。

涉及论文标题：
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction
- LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important

---

## Context Reconstruction for KV Importance Scoring (基于上下文重建的 KV 重要性评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Context Reconstruction for KV Importance Scoring 是 KVzip 提出的 KV pair 重要性评估机制。核心思想：让 LLM 模拟重建原始上下文（teacher-forced decoding），观察每个 KV pair 在此过程中接收到的 attention 大小，以此判断其重要性——被重建"需要"的 KV pairs 为关键，应保留；几乎不被关注的为冗余，可淘汰。

该机制的关键发现：(1) 重建过程的 cross-attention 比 prefill self-attention 显著稀疏（Figure 5 直方图：大部分 KV pairs 收到极低 attention），因为模型可以高效利用 KV_c 中的高层表示+自身权重中的知识；(2) 重建所需的 KV pairs 与 QA、摘要、推理等下游任务的注意力模式高度重叠（Figure 6 2D histogram 下三角区域集中），证明重建作为 proxy task 能泛化到多种下游任务（类似 BERT/MAE 的自监督学习范式）。

从算法pipeline角度拆解术语：

**评分计算流程**：

```
输入: f_LM (LLM), context c (n_c tokens), chunk_size m=2048
输出: importance scores S ∈ R^{L×H×n_c}

1. KV_c = Prefill(c)

2. 将 c 分为 T = ceil(n_c/m) 个 chunk

3. for t = 1..T:
     if t == 1:
         input = "Repeat the previous context:" + c_1
     else:
         input = "Repeat the previous context starting with "
                 + c_{t-1}[-8:] + ":" + c_t
     
     通过 f_LM forward，使用 KV_c 作为 cache
     for l = 1..L, h = 1..H:
         Q = query_proj(hidden)           // G×n_in×d
         K_sub = subsample(KV_c, chunk_t) // (m+n_in)×d
         A = Softmax(Q @ K_sub^T)         // G×n_in×(m+n_in)
         A_sliced = A[:,:,:m]             // KV_c 部分
         S_chunk = max_{g,i} A_sliced     // H×m

4. S = concat([S_chunk_1, ..., S_chunk_T])
5. 淘汰: keep_indices = topk(S, r × n_c) across all heads
```

术语一般如何实现？如何使用？

评分使用标准 FlashAttention-2 forward pass，无需修改 attention kernel（除 softmax-free 变体外）。chunked scoring 使复杂度 O(m·n_c)，峰值内存恒定 O(m²)。repeat prompt 具体措辞影响极小（Table 2: 原始/改写/无 prompt 准确率差异 <0.2%）。评分开销约 2x 标准 prefill，但仅执行一次可被多查询摊还。代码开源：https://github.com/snu-mllab/KVzip。

涉及论文标题：
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

---

## Chunked Scoring for Long-Context KV Importance (长上下文 KV 重要性的分块评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Chunked Scoring 是 KVzip 将上下文重建评分扩展到长上下文（>100K tokens）的关键技术。直接计算完整 context 的 cross-attention 矩阵需要 O(n_c²) 内存和计算，不可行。Chunked scoring 将 context 分为固定大小 chunk（m=2K tokens），逐 chunk 独立计算重要性得分，复杂度降至 O(m·n_c)，峰值内存恒定 O(m²)。

该技术的关键设计：(1) 每 chunk 独立 forward，仅 subsample KV_c 中该 chunk 对应的 keys，形成 m+n_in 长度的 attention 计算（而非全量 n_c+n_in）；(2) chunk 间通过"前一 chunk 最后 8 tokens"的衔接 prompt 保持上下文连续性；(3) 各 chunk 得分直接拼接聚合，无需跨 chunk 归一化。

从算法pipeline角度拆解术语：

**计算流程**：

```
固定: m = 2048, T = ceil(n_c / m)

for t = 1..T:
    // Key subsampling: 仅取出当前 chunk 对应的 KV_c keys
    K_sub = KV_c.keys[:, (t-1)*m : t*m]   // H × m × d
    
    // Forward: input length = n_prompt + m
    // FlashAttention: n_in × (m + n_in) attention
    Q, K_full = forward_layer(input, KV_c)
    A = FlashAttention(Q, cat([K_sub, K_input_keys]), V)
    
    // 取 query 维度 max
    S_chunk_t = max_{query_dim} A[:,:,:m]  // H × m

// 聚合: T 个 chunk 拼接为完整得分
S = concat([S_chunk_1, ..., S_chunk_T])    // L × H × n_c
```

**复杂度**：
- Per-chunk FLOPs: O(m²)，总 FLOPs: O(m·n_c)，线性于 n_c
- 总 overhead: O(n_c² + n_c·m/2)，约 2x 标准 prefill O(n_c²/2)
- 峰值内存: O(m²)，恒定（vs O(n_c²) 全量计算）

术语一般如何实现？如何使用？

Chunked scoring 通过标准 FlashAttention-2 实现，无需修改 attention kernel。chunk size m=2K 在计算效率与 token position index 限制间取得平衡，Section C.1 消融验证不同 chunk size 间性能差异 <2%。对于 context-independent eviction 模式，chunked scoring 仅在预计算阶段执行一次。代码开源：https://github.com/snu-mllab/KVzip。

涉及论文标题：
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

---

## Context-Independent KV Eviction (上下文无关的 KV 淘汰)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Context-Independent KV Eviction 是 KVzip 支持的一种部署时零压缩开销的模式。与 context-dependent 模式（每个新 context 都需执行评分+淘汰，但有 ~2x prefill 开销）不同，context-independent 模式在部署前预计算 static head-level importance scores，推理时直接按 head 重要性分配固定 KV 容量。

该方法借用 DuoAttention 的 head-level eviction 框架（重要 head 保留 full context，不重要 head 使用 sliding window），但评分方式完全不同：DuoAttention 通过检索合成 passkey 优化 head scores（需数小时 8-GPU 优化），KVzip 通过上下文重建（更通用的 proxy task）在单个自然语言样本上计算 head scores（数次 forward pass，一分钟内完成），且性能更优（Figure 11）。

从算法pipeline角度拆解术语：

**预计算与部署流程**：

```
// === 预计算阶段（仅一次，per model） ===
context = single_book_sample   // 88K tokens 英文书（En.QA）
S = compute_scores(context)     // L×H×n_c，chunked scoring
S_head[l,h] = max_i S[l,h,i]   // L×H head-level scores

// 部署后策略
sorted_heads = argsort_desc(S_head)
for head in top_k(sorted_heads):
    // 保留更多 KV pairs（e.g., full context）
    budget[head] = high
for head in bottom_heads:
    // sliding window attention (e.g., 1K tokens)
    budget[head] = low

// 推理时：按固定 budget 执行 head-level eviction，零评分开销
```

术语一般如何实现？如何使用？

Head-level scores 使用通用文本样本预计算（KVzip 使用 SCBench En.QA 的英文书样本，88K tokens）。Figure 24 可视化显示 KVzip 的 head-score 分布比 DuoAttention 更均匀（因使用自然语言重建而非合成 passkey 检索），跨不同数据源（En.QA、En.MultiChoice、Retr.KV）的 head-score 模式高度重叠。部署后零评分开销，压缩比下限约 0.32-0.4（部分 head 仍需 sliding window）。适用于对压缩比要求适中但延迟敏感的场景。代码开源：https://github.com/snu-mllab/KVzip。

涉及论文标题：
- KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction

---

## Linear Attention (线性注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Linear Attention 是一种将标准 softmax attention 的 O(N²d) 计算复杂度降至 O(Nd²) 的注意力机制变体。其核心思想是：(1) 移除 softmax 操作（或用 kernel feature map 替代），(2) 利用矩阵乘法结合律，将 (Q K^T) V 的计算顺序改为 Q (K^T V)（即 right-product kernel trick）。

标准 softmax attention 计算 O = Softmax(Q K^T) V，必须先物化 N×N 的 attention score 矩阵（O(N²d) 空间和时间）。Linear Attention 改为 O = Q (K^T V)，先计算 K^T V（d×d 矩阵），再乘以 Q。由于 K^T V 的大小 d×d 与序列长度 N 无关，复杂度由 O(N²d) 降至 O(Nd²)。

Linear Attention 的递推形式揭示了其本质：
```
M_s = M_{s-1} + k_s^T v_s     // 递推更新 memory state
o_s = q_s M_s                  // 查询 memory state
```
其中 M_s ∈ R^{d×d} 是累积的 memory state。该形式等同于带有矩阵值隐藏状态的线性 RNN，因此线性注意力支持常量内存推理（无需 KV cache）和线性时间训练。

实际应用中，线性注意力有多种变体：Basic Linear Attention（恒等 kernel）、Lightning Attention（IO 优化）、Retention（chunk-wise recurrent）、GLA（Gated Linear Attention，带门控）、Based（混合 linear + sliding window attention）、Rebased（可学习 kernel 函数）。注意：线性注意力在 recall-intensive 任务（如 in-context learning、Needle-in-a-Haystack）上通常弱于标准 softmax attention，因此 hybrid 架构（混合 linear + standard）是常见折中。

从算法pipeline角度拆解术语。

**Linear Attention 的两种计算模式**：

Parallel form（训练，无 causal mask）:
```
Q, K, V = X @ W_Q, X @ W_K, X @ W_V    # 全部 [N, d]
M = K^T @ V                              # [d, d] — right-product first
O = Q @ M                                # [N, d]
```

Recurrent form（推理，逐 token）:
```
M_0 = zeros(d, d)
for s in 1..N:
    q_s, k_s, v_s = x_s @ W_Q, x_s @ W_K, x_s @ W_V
    M_s = M_{s-1} + k_s^T @ v_s         # O(d²) per token, constant memory
    o_s = q_s @ M_s                     # O(d²)
```

术语一般如何实现？如何使用？

Linear Attention 通过 Triton kernel 实现（如 LASP-2 使用 Triton 2.3.1），也可通过 Lightning Attention-2 的 left-product GPU kernel 优化。在分布式训练中，LASP-2 利用 memory state M_t ∈ R^{d×d} 与序列长度无关的特点，通过 AllGather M_t 实现高效序列并行。开源实现见 https://github.com/OpenSparseLLMs/Linear-MoE。

**Linear Attention 的根本局限——全局上下文坍缩（Global Context Collapse）**：Zhang et al. (MHLA, ICLR 2026) 系统揭示了 Linear Attention 的一个内在瓶颈：所有 token 被压缩进一个共享的全局 KV summary G = Σ_j φ(K_j)^T V_j ∈ R^{d×d}，导致：(1) **Rank 受限**：attention 矩阵 A_lin = Q̃ K̃^T 的 rank ≤ min(rank(Q̃), rank(K̃)) ≤ d，无论序列长度 N 多大，表达能力被严格限制在 head dimension d_h（通常 ≤ 72）；(2) **稀疏性丧失**：随 N 增长，每个 token 对全局 summary 的贡献趋于微不足道，注意力分布趋向均匀（高熵），模型无法选择性聚焦于信息量高的 token。这两点共同构成"全局上下文坍缩"，是 Linear Attention 在长序列任务上性能严重下降的根源。缓解方案包括：Focused Linear Attention（加 DW-Conv）、GLA（门控）、Mamba2（SSM）、以及本文的 MHLA（token 维度多头分组 + 可学习混合）。

涉及论文标题：
- LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---

## Mixture-of-Memories (MoM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Mixture-of-Memories (MoM) 是一种在 Linear Attention 框架中引入多个独立 memory state 的架构，受生物神经元 theta-gamma 振荡机制和 Mixture-of-Experts 思想的启发。核心设计：用 M 个独立的 memory state M^m ∈ R^{d×d} 替代传统线性模型中单一固定大小的 memory state，通过 Router 网络将每个输入 token 路由到 top-k 个 memory state 进行选择性更新，最后加权混合得到输出。

MoM 与 MoE 的关键区别：(1) Purpose: MoE 是为了在不显著增加计算量的情况下扩大参数量，MoM 是为了扩大线性模型的 memory capacity 并消除 memory interference；(2) Structure: MoE 的 experts 是 FFN 内的独立网络（channel mixing），MoM 的每个 memory 是 RNN state 配独立 K/V projection weights（token mixing）。

MoM 与 Gating Mechanism 的区别：Gating（如 forget gate a_t）是通过数据依赖的衰减系数选择性"遗忘"旧信息来减少干扰；MoM 是通过将不同信息写入不同 memory state 来实现"分离存储"，从根本上避免不同信息之间的覆盖（而非衰减后再覆盖）。

从算法pipeline角度拆解术语。

**MoM 层前向计算流程**：

```
输入: X ∈ R^{T×d}, 参数: W_g ∈ R^{d×M}, W_k^m, W_v^m ∈ R^{d×d} for m=1..M

Step 1 - Router (token assignment):
  for each token t:
    scores_t = TopK(softmax(x_t @ W_g), k)   # 选 top-k 个 memory
    g_t = scores_t / sum(scores_t)             # 归一化 importance weights

Step 2 - Memory-specific K/V projection:
  for each activated memory m:
    k_t^m = x_t @ W_k^m    # memory-specific key
    v_t^m = x_t @ W_v^m    # memory-specific value

Step 3 - Memory update (仅对激活的 memory):
  M_t^m = UpdateRule(M_{t-1}^m, k_t^m, v_t^m)
  # 非激活 memory: M_t^m = M_{t-1}^m (保持不变)

Step 4 - Shared memory update (始终激活):
  M_t^shared = UpdateRule(M_{t-1}^shared, k_t^shared, v_t^shared)

Step 5 - Memory mixing:
  M̃_t = Σ_{m in activated} g_t^{(m)} · M_t^m + M_t^shared

Step 6 - Output:
  o_t = q_t @ M̃_t
  o_t = activation(norm(o_t)) @ W_o
```

默认配置：M=4 memories + 1 shared memory，top-k=2 激活（activation ratio=0.5）。

复杂度：training 保持 O(n)（每个 memory 处理对应 subsequence，总计算量仍与总 token 数成线性），inference O(1)（每个 memory 维护固定大小的 d×d state，与序列长度无关）。

术语一般如何实现？如何使用？

MoM 的硬件高效实现通过 Triton varlen kernel 实现：① 将 tokens 按 routing 结果分组到各自 memory bucket；② 同 bucket tokens concat 为 varlen 序列；③ Triton kernel 对每个 segment 独立并行计算（chunk-wise parallel scan）；④ 输出加权混合。开源代码：https://github.com/OpenSparseLLMs/MoM 和 https://github.com/OpenSparseLLMs/Linear-MoE。MoM 使用 Gated DeltaNet 作为默认 memory update 方法，替换 K/V projection 为 memory-specific 版本，并施加 auxiliary loss（参考 Switch Transformer 的 load balancing loss）确保 memory 路由均衡。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---

## Right-Product Kernel Trick (右乘核心技巧)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Right-product kernel trick 是线性注意力实现线性复杂度的核心计算技巧。利用矩阵乘法结合律，将标准 attention 的 (Q K^T) V 计算顺序改为 Q (K^T V)，避免物化 N×N 的中间矩阵。

标准 attention（left-product）: O = (Q K^T) V
- Step 1: S = Q K^T → [N, N] 矩阵，O(N²d)
- Step 2: O = S V → [N, d]

Linear attention（right-product）: O = Q (K^T V)
- Step 1: M = K^T V → [d, d] 矩阵，O(Nd²)
- Step 2: O = Q M → [N, d]，O(Nd²)

关键差异：right-product 的中间结果 M 是 d×d 矩阵，其大小与序列长度 N 无关。这使得：(1) 训练复杂度从 O(N²d) 降至 O(Nd²)；(2) 分布式通信量（传输 M）与序列长度无关，对长序列 SP 极其有利。

从算法pipeline角度拆解术语：

```
// Standard Attention (left-product)
Q, K, V = [N, d] each
S = Q @ K^T           // [N, N] — 必须物化，O(N²d)
A = Softmax(S)        // [N, N]
O = A @ V             // [N, d]

// Linear Attention (right-product kernel trick)
Q, K, V = [N, d] each
M = K^T @ V           // [d, d] — 与 N 无关！ O(Nd²)
O = Q @ M             // [N, d], O(Nd²)
```

术语一般如何实现？如何使用？

Right-product kernel trick 通过 GPU kernel 实现，可用 Triton 或 CUDA。当 d 较小时（如 d=64 per head），d²=4096，K^T V 计算量很小。当模型使用大 hidden dim 时（如 d=2048），需考虑 TP 沿 d 维度切分。代码开源：https://github.com/OpenSparseLLMs/Linear-MoE。

涉及论文标题：
- LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid

---

## Memory State in Linear Attention (线性注意力中的记忆状态)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Memory state M ∈ R^{d×d} 是线性注意力递推形式中的核心状态。定义：M_s = Σ_{i=1}^s k_i^T v_i，即前 s 个 token 的 key-value 外积的累积和。M_s 编码了从序列开始到位置 s 的所有历史信息，使得第 s 个 token 的输出可以直接通过 o_s = q_s M_s 计算，无需访问任何之前的 key 或 value。

关键特性：
- M_s 大小固定为 d×d，与序列长度或 token 位置无关
- 递推更新：M_s = M_{s-1} + k_s^T v_s，开销为 O(d²)
- 推理时仅需存储一个 M_s（而非整个 KV cache），实现常量内存推理
- 在分布式 SP 中，每个 chunk t 的 local memory state M_t = K_t^T V_t 也是 d×d 大小，AllGather 通信量仅为 BHd²，与序列长度无关

从算法pipeline角度拆解术语。

**Memory state 在 LASP-2 分布式训练中的使用**：

```
// Chunk t 在设备上计算 local memory state
Q_t, K_t, V_t = X_t @ W_Q, X_t @ W_K, X_t @ W_V   // [C, d]
M_t = K_t^T @ V_t                                    // [d, d]

// AllGather 全局同步
[M_1, ..., M_T] = AllGather([M_1, ..., M_T])

// 全局累积
M_{1:t} = M_{1:t-1} + M_t    // 缓存到 HBM 用于 backward
O_t = Q_t @ M_{1:T}           // 使用全局 memory state
```

术语一般如何实现？如何使用？

Memory state 以 FP16/BF16 存储，形状为 [B, H, d, d]。对于 Linear-Llama3-1B (H=16, d=2048, B=1)，单个 M_t 约 1.07B 参数（~2.14GB FP16）。在 LASP-2 中，M_{1:T} 被缓存到 HBM 以避免 backward 时的重复计算。代码开源：https://github.com/OpenSparseLLMs/Linear-MoE。

涉及论文标题：
- LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---

## Memory Update Rules in Linear Sequence Models (统一视角下的记忆更新规则)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Memory Update Rules 是从递归形式统一描述各种线性序列模型的记忆更新公式。所有线性模型都可以从"memory state 更新"的角度统一表达为 M_t = f(M_{t-1}, k_t, v_t) 的形式。不同方法差异在于：(1) 是否有数据依赖的门控参数；(2) 门控是标量还是向量；(3) 更新是基于外积（k_t^T v_t）还是基于梯度。

MoM 论文 Table 1 提供了完整的统一视角：

| Method | Memory Update Rule |
|--------|-------------------|
| Linear Attn | M_t = M_{t-1} + k_t^T v_t |
| RetNet | M_t = γ M_{t-1} + k_t^T v_t |
| GLA | M_t = (a_t^T 1) M_{t-1} + k_t^T v_t |
| DeltaNet | M_t = (I - k_t^T k_t) M_{t-1} + b_t k_t^T v_t |
| G-DeltaNet | M_t = a_t (I - k_t^T k_t) M_{t-1} + b_t k_t^T v_t |
| TTT | M_t = M_{t-1} + b_t ∇l(M_{t-1}; k_t, v_t) |
| Titans | M_t = a_t M_{t-1} + b_t ∇_M l(M_{t-1}; k_t, v_t) |
| Mamba2 | M_t = a_t M_{t-1} + b_t k_t^T v_t |
| HGRN2 | M_t = (a_t^T 1) M_{t-1} + (1 - a_t)^T v_t |
| RWKV6 | M_t = a_t M_{t-1} + k_t^T v_t |
| RWKV7 | M_t = (a_t^T 1) M_{t-1} + b_t ∇l(M_{t-1}; k_t, v_t) |

其中 a_t, b_t ∈ (0,1) 通常是数据依赖的标量门控参数，γ 是数据无关常量。

关键演进趋势：(1) 早期方法数据无关（Linear Attn, RetNet 的 γ）；(2) 中期引入标量数据依赖门控（Mamba2, RWKV6）；(3) 近期引入向量门控（GLA, HGRN2）或 Delta Rule 自适应更新（G-DeltaNet）；(4) 最前沿方法引入 test-time regression 的梯度更新（TTT, Titans, RWKV7）。

从算法pipeline角度拆解术语。

所有 update rule 都可以融入 MoM 框架：将单一 memory M_t-1 替换为 M 个独立 memory M_t^m，每个 memory 独立执行 update rule，最终通过 router 权重混合。

```
通用 MoM 更新流程:
  for each activated memory m:
    k_t^m = x_t @ W_k^m           # memory-specific key
    v_t^m = x_t @ W_v^m           # memory-specific value
    # 任选 update rule:
    M_t^m = a_t^m · M_{t-1}^m + b_t^m · k_t^{m,T} v_t^m  # 示例: Mamba2 风格
```

术语一般如何实现？如何使用？

Memory update rules 通过 Triton chunk-wise parallel scan kernel 实现：将序列切分为 chunks，chunk 内并行矩阵运算（intra-chunk），chunk 间以 recurrent 方式传递 memory state（inter-chunk）。MoM 在此基础上增加了 token reordering（按 routing 结果分组）和 varlen 支持。代码开源：https://github.com/OpenSparseLLMs/Linear-MoE。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---

## Computation Decomposition in Linear Attention (线性注意力中的计算分解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Computation Decomposition 是将带 causal mask 的线性注意力计算分解为 intra-chunk（块内）和 inter-chunk（块间）两部分的技术。当存在 causal mask Ψ（下三角矩阵）时，right-product kernel trick 无法直接全局应用（token s 不能 attend token s+1..N）。将输出分解为：
- **Intra-chunk**: O_{t,intra} = [(Q_t K_t^T) ⊙ Ψ] V_t — 仅涉及 chunk 内部，使用 left-product（quadratic），可跨设备并行
- **Inter-chunk**: O_{t,inter} = Q_t M_{1:t-1} — 与之前所有 chunk 的 attention，使用 right-product（linear），PrefixSum 累积 memory states

这种分解使 intra-chunk 计算各设备完全并行（无通信依赖），inter-chunk 通信仅传输 memory state M_t（d×d，与序列长度无关），且 AllGather 可与 intra-chunk 计算 overlap。

从算法pipeline角度拆解术语。

**LASP-2 with Masking 流程**：

```
// 并行阶段
for chunk t in 1..T in parallel:
    Q_t, K_t, V_t = X_t @ W_Q, X_t @ W_K, X_t @ W_V
    M_t = K_t^T @ V_t                           // [d, d]

    // AllGather 与 intra 计算 overlap（不同 CUDA stream）
    [M_1, ..., M_T] = AllGather([M_1, ..., M_T])  ||  O_{t,intra} = [(Q_t @ K_t^T) ⊙ Ψ_t] @ V_t

    // Inter-chunk: recursive PrefixSum
    M_{1:t-1} = M_{1:t-2} + M_{t-1}             // 缓存到 HBM
    O_{t,inter} = Q_t @ M_{1:t-1}

    O_t = O_{t,intra} + O_{t,inter}
```

术语一般如何实现？如何使用？

Computation Decomposition 最早由 Yang et al. (2023) 在 GLA 中提出，Sun et al. (2024a) 在 LASP-1 中将其应用于分布式 SP。LASP-2 在此基础上将 ring P2P 改为单次 AllGather，利用 CUDA stream 实现通信-计算 overlap。Intra-chunk left-product 使用 Triton kernel。代码开源：https://github.com/OpenSparseLLMs/Linear-MoE。

涉及论文标题：
- LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid

---

## Hybrid Linear-Standard Attention Model (混合线性-标准注意力模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hybrid model 是将线性注意力层与标准 softmax 注意力层混合的 Transformer 架构。纯线性注意力模型虽训练推理高效，但在 recall-intensive 任务（in-context learning、Needle-in-a-Haystack）上表现不佳。Hybrid model 在部分层保留标准 softmax attention（提供 recall 能力），其余层使用线性注意力（提供长序列高效处理），实现吞吐量与模型能力的平衡。

典型 Hybrid 配置：每 4 层用 1 层标准 attention + 3 层线性 attention（1/4 hybrid），或 1/8、1/2 hybrid。LASP-2 消融实验（Table 4）表明，更高 hybrid ratio 通常带来更好 convergence，但标准 attention 层会增加 quadratic 计算和通信开销。

从算法pipeline角度拆解术语。

**1/4 Hybrid Linear-Llama3 架构 + LASP-2H SP**：

```
Layer 1: Linear Attention  → SP: AllGather M_t (d×d)
Layer 2: Linear Attention  → SP: AllGather M_t
Layer 3: Linear Attention  → SP: AllGather M_t
Layer 4: Standard Attention → SP: AllGather K_t, V_t (C×d) + Softmax(QK^T/√d)V
...每 4 层循环...

// 统一 AllGather-based 通信范式:
//   Linear layer: 通信 M_t [B, H, d, d] — 与序列长度无关
//   Standard layer: 通信 K_t, V_t [B, H, C, d] — 与 chunk 长度有关
```

术语一般如何实现？如何使用？

Hybrid model 通过修改模型配置实现（指定层类型）。LASP-2H 为两类层提供统一的 AllGather-based SP。修改 Llama3 源码将指定层的 `LlamaAttention` 替换为 `LinearAttention`（Triton kernel 实现），其余层保持不变。代码开源：https://github.com/OpenSparseLLMs/Linear-MoE。

涉及论文标题：
- LASP-2: Rethinking Sequence Parallelism for Linear Attention and Its Hybrid

---

## KV Pairs Merging in KV Cache (Averaged / Pivotal / Weighted Merging)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Pairs Merging 是 LOOK-M 提出的一种 KV cache 压缩辅助机制，将对被 evicted（淘汰）的 KV pair 不直接丢弃，而是通过 many-to-one nearest-neighbor matching 将其信息合并到 conserved（保留）的 KV pair 中。该机制解决纯 eviction 方法的核心缺陷：被淘汰的 token 中可能包含关键上下文信息（尤其是视觉细节），直接丢弃会导致 hallucination 和 contextual inconsistencies。

LOOK-M 提出三种合并策略：(a) **Averaged Merging (A-Merge)**——将 evicted token 与其最近匹配的 conserved token 直接求均值，公式为 k_c = 1/(L_sim+1) × (k_c + Σ k_sim[i])；(b) **Pivotal Merging (P-Merge)**——先对每个 evicted token 与其 closest conserved token 做平均融合产生 "pivotal token"，再将所有 pivotal tokens 与 conserved token 平均，强调 conserved token 的权重比例，公式为 k_c = 1/(L_sim+1) × {k_c + 0.5 × Σ(k_sim[i] + k_closest)}；(c) **Weighted Merging (W-Merge)**——基于 similarity matrix S 中的相似度值动态分配权重，而非静态平均，公式为 k_c = 1/(L_sim+1) × {k_c + Σ(k_sim[i] × S[x][y])}。三种策略中 TP + P-Merge（text-prior + pivotal merging）在多模态长上下文任务上取得最佳效果。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**KV Pairs Merging 伪代码（LOOK-M，基于 Section 3.3）**：

```
# 输入：conserved K_c [N+M, D], evicted K_e [L-(N+M), D]
# 输出：merged conserved K_c' [N+M, D]

# Step 1: 计算 similarity matrix between K_e and K_c
S = zeros(len(K_e), len(K_c))
for i in range(len(K_e)):
    for j in range(len(K_c)):
        S[i,j] = cosine_similarity(K_e[i], K_c[j])   # (a^T b)/(||a|| ||b||)

# Step 2: 对每个 conserved token j，收集其 maximum similarity set
match_map = defaultdict(list)
for i in range(len(K_e)):
    j_star = argmax(S[i,:])          # 最相似的 conserved token
    match_map[j_star].append(i)      # many-to-one 关系

# Step 3: 合并策略（三选一）

# (a) Averaged Merging
for j in match_map:
    indices = match_map[j]
    K_c[j] = (K_c[j] + sum(K_e[i] for i in indices)) / (len(indices) + 1)

# (b) Pivotal Merging
for j in match_map:
    indices = match_map[j]
    pivots = [(K_e[i] + K_c[j]) / 2 for i in indices]  # pivotal tokens
    K_c[j] = (K_c[j] + sum(pivots)) / (len(indices) + 1)

# (c) Weighted Merging
for j in match_map:
    indices = match_map[j]
    weighted_sum = sum(K_e[i] * S[i,j] for i in indices)  # similarity-weighted
    K_c[j] = (K_c[j] + weighted_sum) / (len(indices) + 1)

# Value 合并使用 Key 的相同 similarity matrix 和加权权重（KV-pair alignment property）
```

**与 Token Merging (ToMe/AIM) 的关键差异**：

| 维度 | Token Merging (AIM/ToMe) | KV Pairs Merging (LOOK-M) |
|------|-------------------------|--------------------------|
| 合并对象 | 输入 token embedding（visual tokens） | KV cache 中的 Key/Value 张量 |
| 合并阶段 | LLM 输入前 / ViT 层间 | Prompt prefill 完成后、decode 前 |
| 合并目的 | 减少 LLM 输入 token 数 → 减少 FLOPs | 保留 evicted token 的上下文信息 → 补偿 eviction 精度损失 |
| 相似度度量 | 余弦相似度（embedding 间） | 余弦相似度（Key 向量间） |
| 合并粒度 | 二对一成对合并（每次减半迭代） | many-to-one nearest-neighbor 匹配 |

术语一般如何实现？如何使用？

在 LOOK-M 实现中，KV Pairs Merging 在每层 Transformer 的 prefill 之后、eviction 之后执行。合并基于 Key 向量的余弦相似度矩阵，相似度矩阵和加权权重在 Key 和 Value 之间共享（alignment property）。无需任何训练或微调，以即插即用方式集成到 HuggingFace Transformers 推理流程。三种合并策略通过配置开关选择，TP + P-Merge 为默认最优配置。代码开源：https://github.com/SUSTechBruce/LOOK-M。

**MEDA 的差异化使用**：MEDA 采用 A-Merge（平均合并）策略，但与 LOOK-M 的关键不同在于：(1) MEDA 的合并发生在 entropy-guided dynamic layer-wise budget allocation 之后——每层根据跨模态注意力熵获得不同的 KV cache budget；(2) MEDA 使用 text-prior 累积注意力分数进行 KV pair 选择——在合并前先通过加 max(A_s) 偏置确保关键文本 token 不被 evict；(3) MEDA 仅需 A-Merge 即可取得最佳效果，消融实验验证移除 merging 会导致 CLEVR-Change ROUGE-L 从 18.9 降至 18.2。MEDA 的合并与熵引导分配互补——前者保留被淘汰 token 的信息，后者确保不同层获得与实际注意力密度匹配的 KV cache 大小。

涉及论文标题：
- LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference
- MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference
- Multi-head_Temporal_Latent_Attention

**MTLA 的 Hyper-network Dynamic Weighted Merging**：MTLA 提出第四种 KV merging 策略——通过 hyper-network 以输入内容为条件动态生成 per-position merge weight，而非基于相似度的静态合并。区别于 LOOK-M/MEDA 的 post-eviction merging（先逐出再合并），MTLA 的 merging 是架构内嵌的（architecture-level）——在 attention 计算前按固定 stride s 将 consecutively adjacent latent vectors 合并，无需 eviction 步骤。权重生成：w_i = Sigmoid(Linear(c_i) · Linear(pe_j))，其中 c_i 为第 i 个 token 的 latent vector，pe_j 为位置嵌入。Sigmoid gate 而非相似度，使合并策略 learnable 且 data-driven。该策略是 MTLA 的核心创新，使 KV cache 序列长度从 T 缩减至 T/s。

---

## Cross-Modal Attention Entropy (跨模态注意力熵)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cross-Modal Attention Entropy 是 MEDA 提出的用于量化多模态 LLM 每层跨模态注意力分布特性的度量指标。它通过同时计算文本→视觉（T→V）和视觉→文本（V→T）两个方向的注意力熵，捕捉该层对跨模态 token pair 的注意力集中/分散程度。公式为：

$$E_{CM}^l = -(E_{TV}^l + E_{VT}^l)$$

其中：
$$E_{TV}^l = \frac{1}{|T|} \sum_{i=1}^{n_T} \sum_{j=1}^{n_V} A_{TV}^l[i,j] \log A_{TV}^l[i,j]$$
$$E_{VT}^l = \frac{1}{|V|} \sum_{i=1}^{n_V} \sum_{j=1}^{n_T} A_{VT}^l[i,j] \log A_{VT}^l[i,j]$$

A_TV 和 A_VT 分别是文本 query 对视觉 key 和视觉 query 对文本 key 的跨模态注意力矩阵（公式 4）。该度量源于信息论中的 Shannon 熵，被观察到能有效反映不同层注意力密度的差异：早期层（如 Layer 1）注意力分散、熵较高；深层（如 Layer 24）注意力集中于少数关键跨模态 token 对、熵较低（Figure 2）。这种层级差异指导后续的动态 KV cache 分配策略。
从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**跨模态注意力熵计算流程**：
```
# 输入：多模态 prompt X（含 text tokens X_T, visual tokens X_V）
# 对每层 l:

# Step 1: 标准 QKV 投影
Q^l = X W_Q^l      # [L, D]
K^l = X W_K^l      # [L, D]

# Step 2: 按 modality index 分离 text 和 visual
Q_T^l = Q^l[text_indices]      # [n_T, D]
K_T^l = K^l[text_indices]      # [n_T, D]
Q_V^l = Q^l[visual_indices]    # [n_V, D]
K_V^l = K^l[visual_indices]    # [n_V, D]

# Step 3: 计算跨模态注意力矩阵
A_TV^l = Softmax(Q_T^l · (K_V^l)^T / √D)    # [n_T, n_V]
A_VT^l = Softmax(Q_V^l · (K_T^l)^T / √D)    # [n_V, n_T]

# Step 4: 计算跨模态注意力熵
E_TV^l = -(1/|T|) Σ_i Σ_j A_TV^l[i,j] · log(A_TV^l[i,j])
E_VT^l = -(1/|V|) Σ_i Σ_j A_VT^l[i,j] · log(A_VT^l[i,j])
E_CM^l = E_TV^l + E_VT^l   # 注意：公式 (6) 最终带负号，但用于分配时直接用 exp(E_CM^l)

# Step 5: 用于动态 KV cache 分配
α_l = exp(E_CM^l) / Σ_k exp(E_CM^k) · L · ρ
S_l = α_l · S
```

**直觉**：低熵 → 注意力集中于少数关键 token pair → 层已完成跨模态信息聚焦 → 分配更少 KV cache。高熵 → 注意力分散于大量 token pair → 层仍在广泛处理跨模态交互 → 分配更多 KV cache。

术语一般如何实现？如何使用？

跨模态注意力熵在 prefill 阶段计算一次（O(n_T · n_V) per layer，相比 O(L²) 的 self-attention 可忽略），产生 per-layer 熵值向量后用于确定各层的 KV cache budget S_l。由于只在 prefill 执行一次且不需要训练或调优参数，它与任何 MLLM（LLaVA、InternVL、LLaVA-Video 等）和量化/稀疏技术兼容。代码开源：https://github.com/AIoT-MLSys-Lab/MEDA。

涉及论文标题：
- MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference

---

## Dynamic Layer-wise KV Cache Allocation (动态层间KV缓存分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dynamic Layer-wise KV Cache Allocation 是 MEDA 提出的根据每层注意力特性动态而非均匀或静态线性分配 KV cache 大小的方法。核心创新在于使用跨模态注意力熵 E_CM^l 作为分配权重，通过 inverse entropy softmax 公式确定每层 cache 比例 α_l：

$$\alpha_l = \frac{\exp(E_{CM}^l)}{\sum_{k=1}^{L} \exp(E_{CM}^k)} \cdot L \cdot \rho$$

$$S_l = \alpha_l \cdot S$$

其中 L 为层数，ρ 为总压缩比（如 0.1 即总 cache 缩减到原来的 10%），S 为总 KV cache budget。因子 L 使 Σ_l (α_l / L) = ρ，确保各层 α_l 之和为 L·ρ。该公式确保：高熵层（注意力分散）获得较大的 α_l（更多 KV cache），低熵层（注意力集中）获得较小的 α_l（更少 KV cache）。

该设计的核心洞察来自 Figure 2 的实证观察：MLLM 的不同层的跨模态注意力密度存在显著差异——早期层注意力分散需要更多 cache 捕捉广泛的跨模态交互，深层注意力已收敛到关键 token 对需要较少的 cache。这与 PyramidKV 的静态线性递减（前层多后层少但无关实际注意力分布）形成鲜明对比。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**动态分配流程**：
```
# 输入：所有 L 层的跨模态注意力熵 [E_CM^1, ..., E_CM^L]
# 参数：总压缩比 ρ，总 KV cache budget S

# Step 1: Softmax 归一化
softmax_weights = softmax([E_CM^1, ..., E_CM^L])    # [L]，sum = 1

# Step 2: 计算每层分配比例
for l in 1..L:
    α_l = softmax_weights[l] · L · ρ                # 确保 sum(α_l) = L·ρ

# Step 3: 计算每层实际 KV cache 大小
for l in 1..L:
    S_l = α_l · S                                   # 若 ρ=0.1，平均每层保留 10%

# Step 4: 每层独立执行 KV pair selection + merging
for l in 1..L:
    N_l = floor(S_l / (1 + M_ratio))                # 按 β1:β2 = 3:1 分配
    K_c[l], V_c[l] = select_and_merge(K[l], V[l], budget=N_l)
```

**与 Uniform/Static Allocation 的对比**：

| 维度 | Uniform (H2O/SnapKV/LOOK-M) | Static Progressive (PyramidKV) | Dynamic (MEDA) |
|------|---------------------------|-------------------------------|----------------|
| 分配依据 | 无，所有层相同 | 固定线性递减（前多后少） | 实时跨模态注意力熵 |
| 是否感知层间差异 | 否 | 不感知实际差异 | 是，自适应 |
| 参数 | ρ 仅控制总 budget | ρ 控制总 budget + 线性递减率 | ρ 控制总 budget，α_l 自动计算 |

术语一般如何实现？如何使用？

实现为 prefill 后的单次分配步骤，不与特定硬件或框架绑定。计算开销 O(L) 可忽略。在 MEDA 中与 text-prior KV pair selection + average merging 组合使用，三者共同形成完整的即插即用 KV cache 压缩 pipeline。消融实验 (Table 5) 验证移除 Dynamic Allocation 导致 CLEVR-Change ROUGE-L 从 18.9 降至 17.8、Spot-the-Diff 从 18.2 降至 17.5。代码开源：https://github.com/AIoT-MLSys-Lab/MEDA。

涉及论文标题：
- PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling（先驱工作，首次提出跨层不均匀 KV cache 分配——基于 Pyramidal Information Funneling 的静态算术序列递减）
- MEDA: Dynamic KV Cache Allocation for Efficient Multimodal Long-Context Inference（将静态分配扩展为基于跨模态注意力熵的动态分配）

---

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

MileBench 是首个专门设计用于评估多模态大语言模型（MLLM）长上下文能力的综合 benchmark，由 Song et al. (2024) 在 COLM 2024 发表。包含 6,440 个多模态长文本样本，来自 29 个数据集（21 个已有 + 8 个自建），平均每个样本含 15.2 张图像和 422.3 个词。

MileBench 分为两大评估集合：(1) **Realistic Evaluation**——测试 MLLM 在多模态长上下文场景下的理解和推理能力，包括 Temporal Multi-image Tasks（T-1 到 T-4：动作理解与预测、物体与场景理解、视觉导航与空间定位、反事实推理与状态变化）和 Semantic Multi-image Tasks（S-1 到 S-5：知识 QA、富文本图像 QA、视觉关系推理、对话、空间理解）；(2) **Diagnostic Evaluation**——测试 MLLM 的长距离信息检索和干扰排除能力，包括 Needle in a Haystack Tasks（N-1 Text Needle、N-2 Image Needle）和 Image Retrieval（I-1）。

评估指标包括 Accuracy 和 ROUGE-L，按子任务内各数据集的平均计算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**MileBench 评估流程**：

```
# 每类任务的典型输入格式
# T-2 (Object & Scene): 多张时序图像 + "Is the mug still on the counter?"
# N-1 (Text Needle): 多张图像 + 长文本背景 + 插入特定文本片段 + "What was the needle?"

# 评估伪代码
for each sample in MileBench:
    images = load_images(sample.image_paths)       # 平均 15.2 张
    prompt = sample.question                        # 平均 422.3 词
    visual_tokens = visual_encoder(images)          # CLIP ViT-L → 576 tokens/image
    input_seq = interleave(visual_tokens, text_tokens)
    kv_cache, output = mllm.prefill(input_seq)      # 多模态 KV cache 构建
    answer = mllm.decode(kv_cache, max_new_tokens)
    score = metric(answer, sample.ground_truth)     # Accuracy 或 ROUGE-L
```

**MileBench 子任务分类（完整 Taxonomy）**：

| 类别 | 任务 | 数据集数 | 评估指标 |
|------|------|---------|---------|
| T-1: Action Understanding | Action Localization/Prediction/Sequence | 3 | Accuracy |
| T-2: Object & Scene | Object Existence/Interaction/Moving/Shuffle | 4 | Accuracy |
| T-3: Visual Navigation | Egocentric Navigation/Moving Direction | 2 | Accuracy |
| T-4: Counterfactual & State | Counterfactual Inference/State Change/Character Order/Scene Transition | 4 | Accuracy |
| S-1: Knowledge QA | Webpage/Textbook/Complex Multimodal/Long Text QA | 4 | Accuracy |
| S-2: Text-Rich QA | Slide QA/OCR QA/Document QA | 3 | Accuracy |
| S-3: Visual Relation | Visual Change Captioning/Relationship Expressing | 2 | ROUGE-L |
| S-4: Dialogue | Multimodal Dialogue/Conversational Embodied Dialogue | 2 | Accuracy/ROUGE-L |
| S-5: Space Understanding | Space Understanding | 1 | Accuracy |
| N-1: Text Needle | Text Needle In A Haystack | 1 | Accuracy |
| N-2: Image Needle | Image Needle In A Haystack | 1 | Accuracy |
| I-1: Image Retrieval | Image Retrieval | 1 | Accuracy |

术语一般如何实现？如何使用？

MileBench 数据集可从 HuggingFace 和百度网盘下载。评估框架开源在 https://github.com/MileBench。使用时配置 MLLM 的推理接口（支持 LLaVA、InternVL、MobileVLM 等架构），设置 batch_size=1（多数数据集），按子任务分别评估后汇总。MileBench 被认为是评估多模态长上下文 MLLM 的事实标准 benchmark，被 LOOK-M、Cross-Self KV Cache Pruning 等多篇论文采用。代码开源：https://github.com/MileBench。

涉及论文标题：
- LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference
- LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models

---

## Ladder-Shaped KV Cache Pattern (阶梯状 KV 缓存模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Ladder-Shaped KV Cache Pattern 是 LaCache (Shi et al., ICML 2025) 提出的一种跨层异质 KV cache 存储模式。其核心思想是：不同于 StreamingLLM 在所有层缓存同一组 token 的 KV cache，Ladder Pattern 让不同层存储不同位置 token 的 KV cache——浅层保留早期 token 的 KV 状态，深层逐步将焦点转移到更近期的 token，形成阶梯状（ladder-shaped）的二维存储结构。

该 pattern 由两个关键超参数控制：
- **Span S**：同一 token 的 KV 状态被保留的连续层数。S 越大 → 每个 token 被更多层覆盖 → 信息保留下界越高 → 存储成本越大。
- **Overlap O**：每层保留的 token 数量。O 越大 → 每层保留更多 KV 状态 → 语义连续性越好 → 存储效率越低。

每层缓存范围的递推公式：第 l 层保留 [start_l, end_l) 范围的 token，其中 start_l = (l-1) × (S-O)，end_l = start_l + O。相邻层间有 O-(S-O) 的 token 重合。

该 pattern 的两个理论依据：(1) 均匀覆盖提升全部 token 的信息保留下界——最坏情况下重要 token 出现在覆盖最少的层，均匀分布最小化此风险；(2) 相邻 token 语义关联性高，ladder 的跨层平滑过渡实现旧 token 的 smooth fade-out 而非 abrupt eviction。经 1500+ 随机 pattern 的 PPL-cache size Pareto 验证，ladder pattern 位于最优边界。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Ladder Pattern Eviction
for layer l in 1..L:
    start = (l - 1) * (S - O)
    end = start + O
    K_cache[l] = K_full[l, :, start:end, :]   # [H, O, d]
    V_cache[l] = V_full[l, :, start:end, :]   # [H, O, d]
```

关键维度：Full KV: L×H×T×d → Ladder compressed: L×H×O×d，压缩比 = T/O。

术语一般如何实现？如何使用？

Training-free，仅需在 prefill 后根据层索引计算保留范围并裁剪 KV cache。与 StreamingLLM 同属基于位置的静态 eviction，天然兼容 FlashAttention。LongBench 理解任务设 S ≈ num_layers × compression_ratio（均匀压缩），语言建模任务设 S = L/4（消融最优）。代码开源：https://github.com/GATECH-EIC/LaCache。

涉及论文标题：
- LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models

---

## Attention-Free KV Cache Eviction (注意力无关的 KV 缓存逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention-Free KV Cache Eviction 是一类不依赖 attention maps（softmax(QK^T) 输出）来识别重要 token 的 KV cache 压缩方法。与 Attention-Based 方法（H2O、SnapKV、TOVA 等需 prefill 阶段完整 attention scores）不同，Attention-Free 方法仅使用 token 位置或跨层结构来决定保留哪些 KV pairs。代表方法：StreamingLLM（attention sink + sliding window）、LaCache（ladder-shaped 跨层位置模式）。

核心优势：与 FlashAttention 完全兼容。FlashAttention 的 IO-aware tiling + online softmax 不物化完整 S ∈ R^{n×n}，因此 Attention-Based 方法要么放弃 FlashAttention（降速），要么在内核中额外输出 scores（增加 overhead）。Attention-Free 方法完全规避此问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Attention-Free (LaCache/StreamingLLM) — 无需 S 矩阵
for layer l in 1..L:
    keep_range = compute_range_from_position(l, S, O)
    K_cache[l] = K[l, keep_range]     # 仅依赖 token 位置
    V_cache[l] = V[l, keep_range]
# FlashAttention 正常加载压缩后的 KV tiles

# Attention-Based (H2O/SnapKV) — 需要完整 S 矩阵
S = Softmax(Q @ K^T / sqrt(d))        # 需要 materialize S
importance = aggregate(S)
keep_idx = topk(importance, budget)
# 问题：FlashAttention 不产出 S
```

术语一般如何实现？如何使用？

即插即用集成到 HuggingFace Transformers attention 层，无训练/模型修改，计算 overhead 极低（仅 tensor indexing）。LaCache 在 H200 上实现 score-throughput Pareto 最优 (LongBench)，超越所有 Attention-Based baselines。代码：StreamingLLM https://github.com/mit-han-lab/streaming-llm，LaCache https://github.com/GATECH-EIC/LaCache。

涉及论文标题：
- LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models
- Efficient Streaming Language Models with Attention Sinks (StreamingLLM)
- LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important

---

## Iterative KV Cache Compaction (迭代式 KV 缓存压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Iterative KV Cache Compaction 是 LaCache 提出的支持无限长度连续生成的 KV cache 管理策略。当 KV cache 达预设容量上限后，对已压缩 cache 再次应用压缩算法，逐次释放空间。随迭代次数增加，老 token 经历更多轮压缩（被更激进淘汰），新 token 保留更多。内存复杂度 O(1)（constant cache size）。

与 StreamingLLM sliding window 的关键区别：不是简单丢弃最早 token，而是通过 cascaded 压缩实现渐进信息衰减——老 token 在多轮 ladder eviction 中逐步失去各层覆盖，而非一次性消失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
for each decode step:
    append new KV to cache
    if cache_size >= budget:
        for layer l in 1..L:
            # 对已压缩 cache 再 apply ladder pattern
            K_cache[l] = ladder_evict(K_cache[l], S, O)
            V_cache[l] = ladder_evict(V_cache[l], S, O)
            # 老 token 在 ladder 左端被淘汰，新 token 在右端保留
    output = attention(Q_new, K_cache, V_cache)
```

术语一般如何实现？如何使用？

与 ladder-shaped pattern 联合部署，每步 decode 后检查 cache size 并触发。PG19 实验：LaCache 连续生成 10M+ tokens 且 PPL 保持稳定；Full KV cache 在 160K tokens 即 OOM。实现极简——仅在 HuggingFace Transformers attention 层中增加一个 cache size check 和 ladder eviction 触发。代码开源：https://github.com/GATECH-EIC/LaCache。

涉及论文标题：
- LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models

---

## LagKV (Lag-Relative KV Cache Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LagKV 是一种完全不依赖注意力权重（attention-weight-free）的 KV Cache 压缩/逐出方法。其名称中的 "Lag" 指代滞后参考（lag-relative）——使用下一个相邻分区的 KV 统计量作为参考来归一化和评分当前分区的 token。核心洞察基于两点：(1) 自回归模型中 token-wise locality——临近位置的 token 具有更相似的 K/V 张量值；(2) K/V 张量的 per-channel 分布特性（K 的 channel-wise variance 一致，V 的 token-wise variance 显著），使得 channel-wise 标准差成为有效的 token 重要性指标。

评分流程：(a) 将 KV cache 按 lag size L 递归分区；(b) 对每个分区 p，使用分区 p+1 的 token-wise max/min 对分区 p 进行归一化（消除 token-wise locality 导致的 channel 偏移）；(c) 计算归一化后 K/V 的 channel-wise 标准差；(d) softmax 转化为概率分布；(e) 对 K 和 V 的分数求和；(f) top-K 选择保留 rL 个 token。同时保留 attention sink（前 S 个 token，默认 S=16）和最后一个分区作为滑动窗口。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# === LagKV 递归压缩 (每层每 head 独立执行) ===
def lagkv_compress(K, V, S=16, L=128, r=0.5):
    # K, V: (num_heads, seq_len, head_dim)
    # Step 1: 保留 attention sink
    compressed_K = [K[:, :S, :]]
    compressed_V = [V[:, :S, :]]
    remaining = K[:, S:, :]  # length = n - S
    
    # Step 2: 若剩余长度 < 2L，不压缩
    if remaining.shape[1] < 2*L:
        return cat(compressed_K + [remaining]), cat(compressed_V + [remaining])
    
    # Step 3: 按 L 分区
    n_partitions = remaining.shape[1] // L
    K_parts = split(remaining, n_partitions, dim=1)
    V_parts = split(V[:, S:, :], n_partitions, dim=1)
    
    # Step 4: 递归压缩 (最后一个分区 = 滑动窗口，保留不压缩)
    for p in range(n_partitions - 1):
        K_cur, K_ref = K_parts[p], K_parts[p+1]  # K_ref 是 "lag chunk"
        V_cur, V_ref = V_parts[p], V_parts[p+1]
        
        # 4a. 参考 chunk 的 token-wise min/max (沿 seq 维度)
        min_K = K_ref.min(dim=1)  # (heads, head_dim)
        max_K = K_ref.max(dim=1)  # (heads, head_dim)
        min_V = V_ref.min(dim=1)
        max_V = V_ref.max(dim=1)
        
        # 4b. Max-min 归一化
        K_norm = (K_cur - min_K.unsqueeze(1)) / (max_K - min_K).unsqueeze(1)
        V_norm = (V_cur - min_V.unsqueeze(1)) / (max_V - min_V).unsqueeze(1)
        
        # 4c. Channel-wise 标准差 + Softmax
        score_K = softmax(K_norm.std(dim=-1), dim=1)  # (heads, L)
        score_V = softmax(V_norm.std(dim=-1), dim=1)  # (heads, L)
        
        # 4d. 求和得到最终 token score
        score = score_K + score_V  # (heads, L)
        
        # 4e. Top-K 选择
        k = int(r * L)
        keep_idx = topk(score, k, dim=1)
        compressed_K.append(gather(K_cur, keep_idx, dim=1))
        compressed_V.append(gather(V_cur, keep_idx, dim=1))
    
    # Step 5: 加上滑动窗口
    compressed_K.append(K_parts[-1])
    compressed_V.append(V_parts[-1])
    
    return cat(compressed_K, dim=1), cat(compressed_V, dim=1)

# === 压缩比计算 ===
# L_R = S + r*L*(floor((L_s-S)/L) - 1) + L + Mod(L_s-S, L)
# C = 1 - L_R/L_s
```

**关键数学公式**：
$$
\min_i^{p,Z} = \min_{\text{seq}}(Z_i^{p+1}), \quad \max_i^{p,Z} = \max_{\text{seq}}(Z_i^{p+1})
$$

$$
\bar{Z}_i^p = \frac{Z_i^p - \min_i^{p,Z}}{\max_i^{p,Z} - \min_i^{p,Z}}, \quad \text{score}(Z_i) = \operatorname{Softmax}(\operatorname{Std}(\bar{Z}_i))
$$

$$
\text{score}_i = \text{score}(K_i) + \text{score}(V_i)
$$

**Chunk-by-Chunk Prefill 变体**：将 prefill 也拆分为 chunk-by-chunk，每个 L-token chunk prefilled 后进行压缩。这使 hidden states 受压缩影响，但消除了 prefill 一次性全部 forward 的需求。实验显示 FGT 准确率从 100% 降至 ~80%（r=8×），但对序列长度和 needle depth 无强依赖。

术语一般如何实现？如何使用？

集成于 NVIDIA KVPress 框架 (https://github.com/NVIDIA/kvpress)，通过 `KVPressTextGenerationPipeline` 包装 HuggingFace model，在 `generate()` 过程中 hook `past_key_values` 应用压缩。与 FlashAttention 完全兼容——不依赖 attention weight 矩阵。在 Llama-3.1-8B-Instruct 和 Qwen2.5-7B-Instruct 上验证，RULER 16K 上超越 SnapKV 和 StreamingLLM 所有压缩比。64-digit passkey retrieval（L=1024, r=4×）exact match 89%（H2O 仅 35%）。代码开源：https://github.com/AI-Lab-China-Merchants-Bank/LagKV。

涉及论文标题：
- LagKV: Lag-Relative Information of the KV Cache Tells Which Tokens Are Important

## Causal Importance Score (CIS) / 因果重要性分数

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Causal Importance Score (CIS) 是 LOCRET 论文提出的 KV cache unit 重要性度量指标。其核心思想是：在 chunked prefill 的每步中，为每个 KV cache unit（单个 token 在单个 attention head 的 KV 向量对）分配一个分数，该分数**仅依赖当前及之前的 token**（causal），反映该 cache unit 对理解后续上下文的重要性。

CIS 的 ground truth 定义：对于训练样本 $d$，在第 $i$ 层第 $j$ 个 attention head，token $k$ 的 CIS 为所有 answer token 对该 prefix token 的最大 pre-softmax attention score：$\mathbf{S}[k]_j^{(i)} := \max_p (\mathbf{Q}_j^{(i)}\mathbf{K}_j^{(i)T})_{p,k}$，其中 $p \in [n_q(d)+1, n_q(d)+n_a(d)]$（answer token 范围），$k \in [1, n_q(d)]$（prefix token 范围）。对 GQA 模型，取同一 group 内不同 query head 的最大值。

CIS 的**因果性**（causality）是其区别于 H2O/SNAPKV 等 non-causal 评分的核心特征：CIS 在 token 出现时即可计算，不需要等待后续 token 的 attention score，因此天然兼容 chunked prefill。Non-causal 评分（如 H2O 的 A2S、SNAPKV 的 voting）需要完整序列的 attention scores 才能准确评估，在 chunked prefill 中只能看到当前 chunk 导致严重低估某些 token 的重要性（local-global discrepancy）。

从算法pipeline角度拆解术语。

**CIS 在 LOCRET 中的使用流程**：

```
// ============ Training: CIS ground truth 收集 ============
for layer l in 1..L:
    Q_l, K_l = qkv_proj(H_{l-1})
    // 计算 pre-softmax attention scores
    A_l = Q_l @ K_l^T / sqrt(d_k)        // [h, n_seq, n_seq]
    for head j in 1..h:
        for prefix_token k in 1..n_q:
            // 所有 answer token 对该 prefix token 的最大 pre-softmax score
            S[k]_j = max(A_l[j, n_q+1:n_q+n_a, k])

// ============ Inference: Retaining head 预测 CIS ============
for each chunked prefill step:
    Q_chunk, K_chunk, V_chunk = forward_attention(chunk, K_cache, V_cache)
    // Retaining head 预测 CIS（仅依赖当前及之前 token）
    score_chunk = R([Q_chunk, K_chunk, V_chunk])   // MLP forward
    score_cache = concat(score_cache, score_chunk)
    // CIS-based eviction: 保留 top-b 最高分
    indices = topk(score_cache, b)
    K_cache, V_cache = K_cache[indices], V_cache[indices]
```

**CIS 的数学保证**（LOCRET Theorem N.4）：用 Top-b CIS 选择 cache unit 等价于一个 **cache problem**（有预算限制的因果计算问题）。具体而言：若选择函数 $\text{Sel}(c_1, \dots, c_i) = \{c_{p_1}, \dots, c_{p_{b'}}\}$ 其中 $s_{p_1}, \dots, s_{p_{b'}} \in \text{Top-}b(s_1, \dots, s_i)$，则 $(f, b, \{c_i\})$ 是一个 cache problem。这意味着 CIS 可以形式化为一个在线缓存问题的最优解。

术语一般如何实现？如何使用？

CIS 的预测由 retaining head（小型 MLP）完成，推理开销可忽略。CIS 评分在 chunked prefill 的每个 chunk 步后执行一次 TopK eviction。CIS 一旦计算即不变（causal），无需重新评估。LOCRET 将 stabilizers（最后 $n_s$ 个 token）的 CIS 强制设为 $+\infty$ 以防止被 evict，缓解上下文不连续性。LOCRET-Q 变体训练时将 query token 前置使 CIS labels 感知 query，推理时将 query 置于序列首部。开源：https://github.com/huangyuxiang03/Locret。

涉及论文标题：
- LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices

---

## Stabilizers (in KV Cache Eviction) / 稳定器

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Stabilizers 是 LOCRET 中用于缓解 chunked prefill KV cache eviction 导致的上下文不连续性的机制。在每次 chunked prefill 步骤中，当前 chunk 的最后 $n_s$ 个 token 的 CIS 被强制设为 $+\infty$，确保它们永不被 evict。这些保留的 token 作为 "稳定器"，在下一 chunk 的处理中提供局部、连续的上下文锚点。

其设计动机：KV cache eviction 造成 tokens 在位置上的不连续（某些位置的 cache unit 被移除），导致后续 token 接受不连续的上下文。这种不连续性导致 retaining head 的 CIS 预测不稳定——因为 retaining head 的输入 [Q, K, V] 受到不连续 KV cache 的影响——进而导致 eviction 错误被放大，最终使 hidden states 产生大量误差。Stabilizers 通过保证最近的 $n_s$ 个 token 始终存在，阻止了这一错误传播链。

从算法pipeline角度拆解术语。

**Stabilizers 在 LOCRET 中的伪代码（Algorithm 1）**：

```
for chunk in chunk_positions:
    K_chunk, V_chunk, score_chunk = M(x[begin:end], K_cache, V_cache)
    K_cache = Concat(K_cache, K_chunk)
    V_cache = Concat(V_cache, V_chunk)
    score_cache = Concat(score_cache, score_chunk)
    
    if chunk is not the last chunk:
        // === Stabilizers 机制 ===
        score_cache[score_cache.length - n_s : score_cache.length] = +inf
        // 最后 n_s 个 token 的 CIS 被设为无穷大，永不被 evict
    
    indices = top-b(score_cache).indices  // top-b 中必然包含 stabilizers
    K_cache, V_cache, score_cache = K_cache[indices], V_cache[indices], score_cache[indices]
```

**Stabilizers 的消融实验**（LOCRET Figure 3）：
- $n_s = 0$（无 stabilizers）：R.Number 准确率 0%，模型完全失败
- $n_s$ 较小时（如 500-1000）：严重性能退化
- $n_s = 2500$（默认值）：准确率恢复正常
- 原因（Figure 3b-c）：短 stabilizers 或无 stabilizers 导致最后 hidden state 的最大绝对误差和各层 CIS 预测的 mean absolute error 显著增大

术语一般如何实现？如何使用？

Stabilizers 是 eviction 策略中的一个简单机制——将固定数量最近 token 的 score 设为极大值。它不需要额外计算，仅修改 score_cache 的部分值。在 LOCRET 中默认 $n_s = 2500$，同时另有 $n_{loc} = 100$ 个 local token 在最后处理且永不被 evict。Stabilizers 的数量需在 "保持上下文连续性" 和 "留给其他重要 token 的 budget 空间" 之间平衡（太大会压缩可用 budget 导致性能退化，Figure 5b）。

涉及论文标题：
- LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices

---

## Log-Distributed Token Selection for KV Cache (对数分布Token选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Log-Distributed Token Selection 是 LogQuant (Chen et al., 2024) 提出的 KV Cache token 保留策略，利用 base-2 对数分布选择哪些 token 保留为全精度（BF16），哪些量化为 INT2。核心观察是：LLM 中 attention spikes（高注意力分数的位置）遵循对数分布——距离当前位置越远的 token，其 attention spikes 的密度越稀疏。基于这一观察，LogQuant 以几何递减的密度保留 token：最新 W 个 token 密度 p，次新 W 个 token 密度 p/2，再次新 W 个 token 密度 p/4……

这与 KiVi 的"均匀最近窗"（仅保留最近 R 个 token）形成对比：均匀窗在远处硬截断会丢失关键的远距离 token，而对数窗在远处仍有稀疏保留。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Log-Distributed Token Selection 算法伪代码（Algorithm 1 from LogQuant）**：
```
Input: A (list of full-precision tokens), a* (new token), W (window length)
Output: A (updated list of tokens)

procedure APPENDTOKEN(A, a*, W):
  if length(A) < 3W:               // Cache未满：直接追加
    A ← concat(A, a*)
  else:                            // Cache已满：压缩 + 追加
    A ← concat(A[0:2W:2], A[2W:3W])  // 前2W个token步长=2子采样（密度减半）
    A ← concat(A, a*)                // 追加新token
  end if
  return A
end procedure
```

**具体执行流程**（W=42，以 Llama3.1-8B 为例）：
```
Step 1-125: 直接追加全精度token，cache长度递增
Step 126:   cache长度=126=3W，触发压缩
            A[0:84:2] → 保留42个（前84个中隔1取1）
            A[84:126] → 保留42个（全保留）
            追加新token → cache长度=85
Step 127-168: 直接追加，cache长度增至3W
Step 169:   再次压缩：A[0:84:2] → 效果：Window_0密度p/4，Window_1密度p/2
...
```

密度演化：最初 3W 个 token 均为全精度 → 第一次压缩后，旧 W 个保留密度 1/2，新 W 个密度 1 → 第二次压缩后，最旧 W 个密度 1/4，次旧 W 个密度 1/2，最新 W 个密度 1。自然形成 log₂ 递减密度。

Token Coverage 评估（公式 1）：Coverage = Σ(所选 token 的 attention score) / (3W)。该指标衡量选择方案捕获注意力质量的能力——越高表示保留的高注意力 token 越多。实验（Figure 4）显示 LogQuant 的对数选择在 Llama3-8B、Qwen1.5-7B、Phi3-mini 上均优于 KiVi（均匀窗）、StreamingLLM 和 H2O。

术语一般如何实现？如何使用？

在 LogQuant 实现中，对数分布选择通过 HuggingFace transformers 的 Cache 派生类完成。W = ⌊KiVi_R/3⌋（确保全精度 token 数不超过 KiVi 的 R）。对于 R=128，W=42，LogQuant 最多保留 126 个全精度 token（< KiVi 的 128）。未被对数选择保留的 token 被量化为 INT2（通过 Quanto 的 Key-per-channel 量化，group_size=64）。

使用方式：(1) 替代 KiVi 的均匀窗——直接替换 Cache 类；(2) 与 compression-aware 量化后端（Quanto/HQQ）结合；(3) 与 position-agnostic 重排结合——选择保留的 token 被连续存储以改善内存局部性。

涉及论文标题：
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation

---

## Position-Agnostic Attention Computation (位置无关注意力计算)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Position-Agnostic Attention Computation 是 LogQuant 证明并利用的一个 Attention 属性：在解码阶段的 Scaled Dot-Product Attention 中，Key 和 Value 矩阵中 token 的排列顺序不影响最终输出结果。数学上：对于任意置换 P（{1,…,N} 的重排），有 A·V = A_P·V_P。其中 A = softmax(QK^T)，A_P 是 K 经置换 P 后的注意力分布，V_P 是 V 经置换 P 后的 Value 矩阵。

这一属性源于：softmax 对每个 token 独立计算后归一化，而最终的 A·V 是对所有 token 的 Value 加权求和。加权求和的交换律保证顺序无关。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**置换不变性证明**：
```
原始 Attention:  O = A · V
                = Σ_i a_i · v_i        // a_i = softmax(q·k_i^T/√d)_i
                                        // 对 i 的求和，交换律保证顺序无关

经置换 P 后:
  K_P = K[P]                           // 按 P 重排 Key
  V_P = V[P]                           // 按 P 重排 Value
  A_P = softmax(Q · K_P^T)             // 注意力分布对应变化
  O_P = A_P · V_P
      = Σ_i a_{P(i)} · v_{P(i)}        // 由于求和遍历全部 token
      = Σ_j a_j · v_j                  // 每个 token 恰好贡献一次
      = O                              // 与原始输出相同
```

**在 LogQuant 中的应用**：
```
// 传统方式（KiVi）：KV Cache 按原始位置存储
// [BF16_token_1, INT2_token_2, ..., BF16_token_R, ..., INT2_token_N]
// → 全精度和量化 token 交错存储，内存碎片化

// LogQuant 方式：利用置换不变性重排
// [INT2_token_1, INT2_token_2, ..., INT2_token_{N-R}, BF16_token_1, ..., BF16_token_R]
// → 全精度和量化 token 分别连续存储 → 更好的内存局部性
```

术语一般如何实现？如何使用？

在 LogQuant 实现中，position-agnostic 属性通过 concat 操作体现：在 Cache 类中，全精度 token 被连续存储在 cache 的一端，量化 token 被连续存储在另一端（或反之），中间无交错。这通过继承 HuggingFace Cache 类时修改 K/V 的存储布局实现——无需修改 attention 计算本身。

使用方式：(1) 任何需要将 KV Cache 按不同精度/格式分组存储的场景均可利用此属性；(2) 可与 fused dequantization-attention kernel 结合——连续的全精度 K/V 段避免了 gather/scatter 操作；(3) 论文注明"未来的 operator fusion 优化将在此属性基础上直接在量化 cache 上计算 attention"。

注意：此属性仅适用于解码阶段（每次仅 1 个 query token）。预填阶段（prefill）因 softmax 的 causal mask 依赖 token 顺序，不适用此属性。

涉及论文标题：
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation

---

## Asymmetric KV Cache Quantization (非对称KV缓存量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Asymmetric KV Cache Quantization 是指对 KV Cache 中不同位置的 token 应用不同精度量化的策略。在 LogQuant 和 KiVi 等 training-free 方法中，核心思想是：(1) 部分"重要"token 保留为原始精度（BF16/FP16），(2) 其余 token 被量化到低精度（如 INT2）。这种"非对称"体现在时间/位置维度——不是所有 token 被同等对待——不同于传统对称量化（所有值统一量化到同一精度）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**非对称量化 vs 对称量化 vs 逐出**：
```
对称量化（如 GEAR, QAQ）：
  for each token t in KV_cache:
    K_quant[t] = quantize(K[t], bits=2)   // 所有token统一2-bit
    V_quant[t] = quantize(V[t], bits=2)

非对称量化（KiVi, LogQuant, JanusQuant）：
  for each token t in KV_cache:
    if t in selected_tokens:              // 选择性保留
      K_store[t] = K[t]                   // BF16全精度
    else:
      K_store[t] = quantize(K[t], bits=2) // INT2量化

逐出（H2O, StreamingLLM）：
  for each token t in KV_cache:
    if t in selected_tokens:
      K_store[t] = K[t]                   // 保留
    else:
      delete K[t]                         // 彻底删除
```

**LogQuant 的非对称设计**：
- 全精度 token 数量：2W~3W（W=⌊R/3⌋），例如 R=128 时 W=42，保留 84~126 个全精度 token
- 量化 token：其余所有 token 量化为 INT2（group_size=64, key-per-channel）
- 压缩率: 16L / (2(L-2W) + 16×2W)  ≈ 16L / (2L + 28W)

**为什么非对称优于逐出**（LogQuant Section 2.3）：
在相同的对数选择方案下，量化（降低精度）比逐出（移除 token）保留更多信息。L1 Attention 误差：
- LogQuant (2-bit quantization): 432.50
- KiVi (2-bit quantization): 556.10
- LogQuant (Eviction): 1076.70
- KiVi (Eviction): 1612.56

量化误差比逐出误差小 2-3×。原因：softmax 归一化下逐出 token 会重新分配 attention 权重，导致更大的分布偏差。

术语一般如何实现？如何使用？

在 HuggingFace transformers 中，非对称量化通过继承 Cache 类实现：(1) 维护 selected_indices 标记哪些 token 是全精度；(2) 全精度 token 直接存储在 self.key_cache[layer] 和 self.value_cache[layer]；(3) 量化 token 存储为 INT2 packed format；(4) 每次 attention 前，量化 token 经 dequantize() 恢复为 BF16 后与全精度 token 拼接。

非对称量化的适用场景：(1) 长上下文推理内存受限场景；(2) batch inference where KV cache dominates memory；(3) 与 GQA/MQA 结合——非对称量化进一步减少已缩减的 KV cache 内存。

涉及论文标题：
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation
- KiVi: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---

## Key-per-Channel KV Cache Quantization (逐Key通道量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Key-per-Channel Quantization 是 KV Cache 量化中的一种分组策略：沿 Key tensor 的 channel 维度（hidden dimension）独立计算每个 channel 的 scale factor 和 zero-point，而非跨 token 或跨整个 tensor。在 LogQuant 中，采用 "Key-per-channel strategy" 作为 Quanto 后端的量化配置，对 Key 矩阵的每个 channel 独立量化。

与 per-token 量化（沿序列长度 L 维度分组：每个 token 有独立 scale）和 per-tensor 量化（整个 K tensor 共用一个 scale）相比，Key-per-channel 在 K cache 的 channel 维度上提供更精细的量化粒度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**三种分组策略对比**（K ∈ R^{L×d}）：
```
Per-Tensor:         scale ∈ R, zero ∈ R                     // 全局1组参数
Per-Token:          scale ∈ R^L, zero ∈ R^L                 // 每组1个token的d维度值
Key-per-Channel:    scale ∈ R^d, zero ∈ R^d                 // 每组1个channel的L个值
Group-wise (G=64):  scale ∈ R^{L×ceil(d/64)}, ...           // 每组64个channel的值
```

**LogQuant 采用 Key-per-Channel 的原因**：
K cache 中不同 channel 的值分布差异大（outlier channel 问题）——少数 channel 的值幅度可达正常 channel 的 6 倍以上。Key-per-channel 量化将 outlier channel 隔离在自己的组内，避免其极值放大同组正常 channel 的量化误差。而 V cache 通常不存在明显的 channel-wise outlier 模式（LogQuant paper, Section 2.1; JanusQuant paper, Section 2.3）。

**Key-per-Channel 量化流程**：
```
// K ∈ R^{L×d}, target bits=2
for c in 1..d:
    scale[c] = max(|K[:,c]|) / (2^{bits-1} - 1)     // 每channel独立scale
    zero[c] = 0                                      // 对称量化zero=0
    K_quant[:,c] = round(K[:,c] / scale[c])
    K_quant[:,c] = clamp(K_quant[:,c], -2^{bits-1}, 2^{bits-1}-1)

// 解量化
K_deq[:,c] = K_quant[:,c] * scale[c]
```

术语一般如何实现？如何使用？

在 Quanto 和 HQQ 等量化后端中，Key-per-Channel 通过 `quantize(tensor, axis=1)` 实现（axis 沿 hidden dim）。在 HuggingFace 集成中，LogQuant 将 Quanto 的 `qtype` 设置为 per-channel 量化模式。

适用场景：(1) 当 K cache 存在 channel-wise outlier 时必须使用 per-channel（否则 outlier 会严重损害量化精度）；(2) 与 per-token 互补——V cache 可用 per-token（无显著 channel outlier），K cache 用 per-channel；(3) JanusQuant 的 RtSmooth 在 Key-per-Channel 之前先对 K 做 per-token 平滑变换，使 outlier channel 的值更均匀，再用 Key-per-Channel 量化获得更好精度。

涉及论文标题：
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation

---

## Training-free KV Cache Compression (免训练KV缓存压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Training-free KV Cache Compression 是一类无需模型重训练、微调或校准数据即可应用的 KV Cache 压缩方法。与 training-required 方法（如 MLA、GQA 需要重新训练注意力结构）不同，training-free 方法直接对预训练模型的 KV Cache 运行时的存储和访问进行优化。主要分为两类：(1) Eviction（逐出）——选择性删除不重要的 token，如 H2O、StreamingLLM、SnapKV；(2) Quantization（量化）——降低不重要 token 的数值精度，如 KiVi、LogQuant、QAQ。

Training-free 的核心优势：即插即用，无需访问训练数据和 GPU 训练资源，适用于任何预训练模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Training-free Eviction 流程（H2O/StreamingLLM 模式）**：
```
// Prefill: 正常计算，所有 token 存入 KV Cache
// Decoding: 每步或每 k 步触发 eviction
if len(K_cache) > budget:
    scores = compute_importance(K_cache, V_cache, Q_current)  // H2O用A2S分数
    keep_indices = top_k(scores, budget)                      // StreamingLLM仅保留sink+recent
    K_cache = K_cache[keep_indices]
    V_cache = V_cache[keep_indices]
```

**Training-free Quantization 流程（LogQuant/KiVi 模式）**：
```
// 在选择保留全精度token后
for token in non_selected_tokens:
    K_quant[token] = quantize(K[token], bits=2, per_channel)
    V_quant[token] = quantize(V[token], bits=2, per_token)

// Decoding 时 dequantize
K_deq = dequantize(K_quant)
V_deq = dequantize(V_quant)
attention = softmax(Q @ concat(K_deq, K_fp).T / sqrt(d))
```

术语一般如何实现？如何使用？

Training-free 方法通常通过以下方式集成：(1) HuggingFace transformers 的 Cache 类派生——LogQuant、KiVi 均采用此方式；(2) vLLM/SGLang 等 serving 框架的 KV Cache 管理模块——通过修改 PagedAttention 的内存管理逻辑；(3) monkey-patch 模型的 forward 方法——在 attention 层插入 eviction/quantization 逻辑。

LogQuant 选择方法 (1)：继承 `transformers.Cache`，在 `update()` 方法中调用 `APPENDTOKEN` 算法，利用 Quanto 后端量化非保留 token。优势是与现有 HF 推理 pipeline 无缝兼容。

涉及论文标题：
- LogQuant: Log-Distributed 2-Bit Quantization of KV Cache with Superior Accuracy Preservation
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

**Sparse Frontier 论文对 training-free sparse attention 的系统化**：该论文是迄今最大规模的 training-free 稀疏注意力实证研究（7065 配置，3 模型家族，9 任务，sparsity up to 0.95）。提出四轴分类体系：(1) Unit of Sparsification（blocks/pages vs verticals/slashes），(2) Importance Estimation（fixed vs content-aware），(3) Budget Allocation（uniform vs adaptive/threshold-based），(4) KV Cache Management（eviction vs full cache retention）。基于此选择 6 种代表性方法进行 harmonized 实现并在 vLLM 框架内统一评估。

---

## Multi-Head Linear Attention (MHLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Multi-Head Linear Attention (MHLA) 是一种新型线性注意力机制，由 Zhang et al. (ICLR 2026) 提出。核心创新在于：沿 token 维度（而非 channel/head 维度）将序列划分为 M 个 non-overlapping blocks（"token-level heads"），为每个 block 独立计算局部 KV summary，再通过可学习的系数矩阵 Mc 使每个 query block 生成专属的混合 summary，从而恢复 standard linear attention 中丧失的 query-conditioned 选择性和 token 级别多样性。

MHLA 的关键操作流程：
1. 输入 X ∈ R^(N×d)，投影得到 Q, K, V
2. 应用 kernelized feature map φ(·)：Q̃ = φ(Q), K̃ = φ(K)
3. 将序列沿 spatial（2D）或 spatiotemporal（3D）维度分为 M 个 blocks，每 block 含 N_b 个 token
4. 每 block b 计算局部 KV summary：S_b = Σ_{j∈b} K̃_j^T V_j ∈ R^(d×d)，z_b = Σ_{j∈b} K̃_j ∈ R^d
5. 通过可学习系数矩阵 Mc ∈ R^(M×M)，query block i 的混合 summary：S̃_i = Σ_{b=1}^M m_{i,b} S_b，z̃_i = Σ_{b=1}^M m_{i,b} z_b
6. 输出：o = (q̃^T S̃_i) / (q̃^T z̃_i) = Σ_{b} m_{i,b} (q̃^T S_b) / Σ_{b} m_{i,b} (q̃^T z_b)

复杂度 O(Nd² + M²d²)。当 M² ≤ N 时，主导项 O(Nd²) 与 standard linear attention 相同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**MHLA forward pass 伪代码**：

```
def mhla_forward(X, W_Q, W_K, W_V, M_c, block_ids):
    # X: [N, d], M_c: [M, M] learnable coefficient matrix
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V                # [N, d]
    Q_tilde = phi(Q)                                     # kernelized, e.g., ReLU
    K_tilde = phi(K)
    
    # Step 1: Compute local KV summaries per block
    S = []  # list of [d, d] summaries
    z = []  # list of [d] normalizers
    for b in range(M):
        mask_b = (block_ids == b)                        # [N_b]
        S_b = K_tilde[mask_b].T @ V[mask_b]              # [d, d]
        z_b = K_tilde[mask_b].sum(dim=0)                 # [d]
        S.append(S_b)
        z.append(z_b)
    
    # Step 2: Multi-Head Mixing — query-conditioned per block
    O = zeros(N, d)
    for i in range(M):
        mask_i = (block_ids == i)
        q_i = Q_tilde[mask_i]                            # [N_i, d]
        S_mixed_i = sum(M_c[i, b] * S[b] for b in range(M))  # [d, d]
        z_mixed_i = sum(M_c[i, b] * z[b] for b in range(M))  # [d]
        O[mask_i] = (q_i @ S_mixed_i) / (q_i @ z_mixed_i)    # [N_i, d]
    return O
```

**实际实现优化**：所有 block summaries 堆叠为 [M, d, d] tensor，通过 batched GEMM 一次性计算所有混合 summaries：S_all = einsum('ij,jkl->ikl', M_c, S_stacked)。

术语一般如何实现？如何使用？

MHLA 通过标准 PyTorch 和 GEMM 操作实现，无自定义 CUDA kernel，可直接替换任何 Transformer 架构中的 attention 模块。初始化策略：Mc 按 locality-biased 初始化 m_{i,j}^(0) ∝ 1 - dist(i,j)/max_k(dist(i,k))，训练过程中 clip 到 (0,1) 保持非负。开源实现：https://github.com/DAGroup-PKU/MHLA（MIT license），含五个子项目覆盖图像分类（DeiT/VLT）、图像生成（DiT/DiG）、T2I（SANA）、视频生成（Wan2.1）、NLP。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---

## Global Context Collapse (全局上下文坍缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Global Context Collapse 是 Zhang et al. (ICLR 2026) 在对 linear attention 进行系统分析时识别出的核心失败模式。其本质是：standard linear attention 将所有 token 压缩进一个固定的 d×d 全局 KV summary（G = Σ_j φ(K_j)^T V_j ∈ R^(d×d)），所有 query 共享这一 summary，导致当序列长度 N 增长时，信息量超出 d×d 矩阵的承载能力，模型的表示多样性和注意力选择性发生坍缩。

该现象可通过两个互补指标量化：
1. **Rank 限制**：A_lin = Q̃ K̃^T 的 rank ≤ min(rank(Q̃), rank(K̃)) ≤ d。当 N >> d 时，attention 矩阵严重秩亏，模型无法捕获多样的 query-conditioned 注意力模式。实测显示 linear-attention 模型 attention score 的 rank 始终被 head dimension（d_h ≤ 72）限制。
2. **稀疏性丧失/熵升高**：随 N 增大，每个 token 对全局 summary 的贡献趋于无穷小，注意力分布趋向均匀分布。实测 linear attention 的注意力熵显著高于 softmax attention，表明模型无法聚焦于少量信息量高的 token。

MHLA 论文通过 Imagenet DeiT 和 Wan2.1 视频生成（N=31500 tokens）的实验验证了该现象：视频生成中 vanilla linear attention 几乎无法训练（loss 平台高），而 MHLA 恢复了正常收敛。

从算法pipeline角度拆解术语。

**Global Context Collapse 在 attention pipeline 中的表现**：

```
// Standard Linear Attention — 共享全局 summary
G = zeros(d, d)
z = zeros(d)
for j in 1..N:
    G += phi(k_j)^T @ v_j       // 所有 token 信息混合进 [d, d]
    z += phi(k_j)
for i in 1..N:
    o_i = (phi(q_i)^T @ G) / (phi(q_i)^T @ z)  // 所有 query 共用同一 G

// 问题：当 N >> d 时，G 的信息容量饱和（rank ≤ d）
//   → 不同 query 获得的 context 几乎没有差异
//   → 注意力分布趋于 uniform（高熵）
//   → 模型失去聚焦能力
```

术语一般如何实现？如何使用？

该概念用于诊断 linear attention 的性能瓶颈，而非直接实现。缓解策略包括：(a) MHLA——将单个全局 summary 拆为 M 个局部 summary + 可学习混合；(b) Focused Linear Attention——添加 DW-Conv 注入局部信息；(c) GLA——门控机制；(d) Mamba——state space 模型。理解该概念有助于在长序列场景下选择或设计合适的 attention 变体。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---

## Multi-Head Mixing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Multi-Head Mixing 是 MHLA 的核心机制，通过一个可学习的系数矩阵 Mc ∈ R^(M×M) 实现 query-conditioned 的 block 级 KV summary 混合。矩阵元素 m_{i,j} 表示 query block i 对 key-value block j 的 affinity（亲和度），第 i 行 m_i 指定 query block i 如何将 M 个局部 summary 线性组合成该 block 专属的全局 summary。

与 standard multi-head attention 沿 channel 维度分头不同，MHLA 的 "multi-head" 指沿 token（spatial）维度的分组。每个 token-level head 独立计算其 local context，再通过可学习的跨 head 混合恢复全局信息。

初始化策略：locality-biased——m_{i,j}^(0) ∝ 1 - dist(i,j)/max_k(dist(i,k))，其中 dist(i,j) 是 block i 和 j 中心在 2D/3D 网格上的欧氏距离。该初始化编码了空间近邻优先的先验，提供更稳定快速的收敛；训练过程中 Mc 完全可学习，并通过 clip 到 (0,1) 保持非负。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Multi-Head Mixing 的 token 级效应**：

```
# block b(t) = token t 所属的 block
# S_j = Σ_{t in block j} K̃_t^T V_t  (local summary)
# Mixed summary for query block i:
S̃_i = Σ_{b=1}^M m_{i,b} S_b = Σ_{t=1}^N m_{i,b(t)} K̃_t^T V_t

# Token-level contribution: query q_i attends to token t via:
# m_{i,b(t)} × (q̃_i^T K̃_t) × V_t^T
#  ^^^^^^^^     ^^^^^^^^^^^^    ^^^^
#  block选择    token内重加权    值
#  (query-      (kernel inner
#   cond.)       product)
```

两阶段权重机制：(1) block 级选择 m_{i,b(t)}——query-conditioned（不同 query block 可获得不同 block 权重）；(2) block 内 token 重加权 q̃_i^T K̃_t——传统 kernel 相似度。两者结合恢复 query-conditioned 的 token 级多样性。

术语一般如何实现？如何使用？

Mc 初始化为 [M, M] 的 float tensor，作为模型参数参与端到端训练。每行 m_i 归一化到和为 1。训练中 Mc 随其他参数一同优化，每步更新后 clip 到 (0,1)。在 chunkwise parallel form 中，Mc 的上三角被 mask 以满足 causality。由于 M 通常较小（≤ sqrt(N)），Mc 的额外存储和计算开销可忽略。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---

## Query-Conditioned Selectivity (查询条件选择性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Query-Conditioned Selectivity 是 softmax attention 的一个关键特性：每个 query q_i 产生自己专属的注意力分布 {α_ij}_j，使不同 query 可以从同一序列中提取不同的上下文信息。在 standard softmax attention 中，α_ij = exp(q_i^T k_j/√d) / Σ_t exp(q_i^T k_t/√d)，权重同时依赖 query 和 key，实现了完全的 query-conditioned 逐 token 权重分配。

Standard linear attention 丧失了该特性：所有 query 共享同一个全局 summary G，导致 o_i = q̃_i^T G / q̃_i^T z 中唯一的 query 依赖性来自 q̃_i 本身，而 token 级别的贡献（k_j v_j）已在 G 中不可区分地融合，不同 query 获得几乎相同的 context vector。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**选择性对比（三个 attention 机制）**：

```
// Softmax Attention — 完全 query-conditioned，O(N²)
for each query i:
    scores_i = [exp(q_i @ k_j / sqrt(d)) for j in 1..N]  // 每个 query 独立计算
    alpha_i = softmax(scores_i)                            // 不同的分布
    o_i = sum(alpha_ij * v_j for j in 1..N)

// Standard Linear Attention — 丧失 selectivity，O(Nd²)
G = sum(phi(k_j)^T @ v_j for j in 1..N)  // 所有 token 融合
for each query i:
    o_i = (phi(q_i)^T @ G) / (phi(q_i)^T @ z)
    // q_i 不同，但 G 中 token 贡献不可分 → 选择性弱

// MHLA — 恢复 selectivity via 两阶段，O(Nd² + M²d²)
// Stage 1: query-conditioned block 选择
// Stage 2: block 内 token 级 kernel reweighting
S̃_i = Σ_b m_{i,b} S_b     // query block i 专属的混合 summary
o_i = (q̃_i^T @ S̃_i) / (q̃_i^T @ z̃_i)
     = Σ_t m_{i,b(t)} (q̃_i^T @ K̃_t) @ V_t^T
// 两阶段: block 级 m_{i,b(t)} + token 级 q̃_i^T K̃_t
```

术语一般如何实现？如何使用？

Query-conditioned selectivity 是评估注意力机制表达能力的重要维度。MHLA 论文通过注意力矩阵的 rank 和熵来量化该特性：rank 越高表示注意力空间越多样（更多不同的注意力模式），熵越低表示注意力越集中（更强的选择性）。在模型设计中，选择注意力变体时需要权衡该特性和计算效率。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---

## KV Summary vs Hidden States

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Summary 是 MHLA 论文引入的术语，用于严格区分 MHLA 中的 key-value summary 与传统 linear attention 中的 Hidden State。虽然二者在符号上可能相似（均为 d×d 矩阵），但底层计算和依赖图有本质区别：

1. **依赖关系**：传统 linear attention 的 Hidden State h_t 通过严格递推链 h_t = f(h_{t-1}, k_t, v_t) 更新，h_t 依赖于 h_{t-1}，存在状态传播。MHLA 的每个全局 KV Summary S_g 独立计算：S_g = Σ_b m_{g,b} S_b，各局部 summary S_b 相互独立并行计算，无状态传播。

2. **聚合模式**：传统 Hidden State 是一对一递推更新（one-to-one）；MHLA 的 KV Summary 是多对一聚合（many-to-one）——每个 S_g 由所有局部 S_b 通过特定混合系数 m_{g,b} 聚合而成。

这种设计避免了 Hidden State 中历史信息的刚性继承，使 MHLA 的 summary 具有更高的表达能力和灵活性。

从算法pipeline角度拆解术语。

```
// Traditional Hidden State (recurrent chain)
h_0 = 0
for t in 1..N:
    h_t = h_{t-1} + phi(k_t)^T @ v_t    // 严格递推，h_t ← h_{t-1}
    o_t = phi(q_t)^T @ h_t

// MHLA KV Summary (independent + mixture)
for b in 1..M:  // 并行
    S_b = sum(phi(k_j)^T @ v_j for j in block b)  // 独立计算
for i in 1..M:
    S̃_i = sum(m_{i,b} * S_b for b in 1..M)       // 多对一聚合
    o_i_block = q_i_block @ S̃_i
```

术语一般如何实现？如何使用？

该术语区分本身不涉及具体实现，但其概念影响架构设计：由于 MHLA 的 KV summary 独立计算，各 block 可完全并行化（训练时），且推理时可增量更新（causal inference 中仅更新当前 block 的 S_b 并重新计算受影响的 S̃_i）。这种设计更适合 GPU 的大规模并行计算。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---

## Kernelized Feature Map (核化特征映射)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Kernelized Feature Map φ(·) 是 linear attention 中将 softmax kernel 替换为正定特征映射的数学工具。Softmax attention 中 Sim(Q_i, K_j) = exp(Q_i K_j^T/√d) 不可分解为独立 feature 的内积。Linear attention 通过选取合适的 φ(·) 使得 Sim(Q_i, K_j) ≈ φ(Q_i) φ(K_j)^T，从而可用结合律将计算顺序从 (Q K^T) V 改为 Q (K^T V)，将复杂度从 O(N²d) 降至 O(Nd²)。

常用的 φ(·) 包括：
- φ(x) = elu(x) + 1（保证正值，最常用）
- φ(x) = ReLU(x)（MHLA 论文中在图像任务上使用）
- φ(x) = 1 + tanh(x)（某些变体）
- 可学习 kernel（如 Rebased attention）

从算法pipeline角度拆解术语。

```
// Kernelized feature map 在 linear attention 中的作用
Q, K = X @ W_Q, X @ W_K          // [N, d]
Q_tilde = phi(Q)                  // 例如: Q_tilde = ReLU(Q)
K_tilde = phi(K)                  // K_tilde = ReLU(K)

// 此时: Q_tilde @ K_tilde^T 近似 exp(Q @ K^T/sqrt(d))

// Right-product trick:
// O = (Q_tilde @ K_tilde^T) @ V   [N, N] @ [N, d]  ← O(N²d)
//   = Q_tilde @ (K_tilde^T @ V)   [N, d] @ [d, d]  ← O(Nd²)
```

关键约束：φ(·) 必须输出正值以保证注意力权重的非负性，避免除零问题。MHLA 论文中，NLP 任务上推荐省略 normalizer 项（q̃^T z̃_i）以提高长序列下的训练稳定性。

术语一般如何实现？如何使用？

纯 PyTorch 操作，直接应用激活函数于 Q 和 K 张量。在 GPU 上无额外开销。选择具体 φ(·) 影响模型性能，一般推荐 elu(x)+1（平衡正值性和梯度流）。代码中通常实现为：`Q_tilde = F.elu(Q) + 1; K_tilde = F.elu(K) + 1` 或 `Q_tilde = F.relu(Q); K_tilde = F.relu(K)`。

涉及论文标题：
- MHLA: Restoring Expressivity of Linear Attention via Token-Level Multi-Head

---


## Self-Speculation with Compressed KV Cache（压缩KV Cache自推测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Self-Speculation with Compressed KV Cache 是 MagicDec 提出的核心 SD drafting 策略：使用 target model 自身（而非独立的小 draft model）作为 draft model，配合稀疏化的 KV cache 进行 token 推测。与传统 SD 使用独立小模型作为 draft 不同，self-speculation 不额外加载任何模型权重（draft 与 target 共享），仅维护一份压缩 KV cache（budget K << full sequence length S）。

核心优势：(1) **低 draft cost**：draft 无额外参数加载，仅 KV loading cost = B × K × model_dim（vs target 的 B × S × model_dim），当 S > S_inflection 时 T_D/T_T → 0；(2) **高接受率**：KV 压缩比模型压缩更容易达到 >90% 的 token 接受率——因为 target model 看到的是自己"实际会关注的" KV 子集，而非一个能力更弱的小模型的预测；(3) **lossless**：验证阶段仍使用完整 KV cache，输出与 target model AR 解码完全一致。

在 MagicDec 中，self-speculation 在小 batch + 短序列时性能不如小 draft model（因为参数加载占主导，小模型的参数更少），但在大 batch + 长序列时超越小模型（KV bottleneck 主导，self-speculation 的高接受率优势体现，Figure 7c）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Self-Speculation 与传统 SD 的对比

# 传统 SD: 小 draft model + 完整 KV
draft_params = Theta_draft   # 独立加载（如 LLaMA-3.2-1B ~2.5 GB）
K_draft = K_full             # 完整 KV（~25 GB for B=128, S=32K）
T_D = Load(Theta_draft) + Attend(K_draft)  # 两项都大

# Self-Speculation: target 自身 + 压缩 KV
draft_params = Theta_target  # 复用 target 已在 GPU 上的权重
K_draft = Select(K_full, budget=K)  # 压缩 KV（~1.6 GB for B=128, K=2049）
T_D = Attend(K_draft)                # 仅 KV attention，无额外参数开销

# MagicDec 的 self-speculation decode 循环
# 每步 draft phase 使用压缩 KV:
q = W_q @ embed(last_token)
s = q @ K_draft^T / sqrt(d_head)     # K_draft size: [B, K, n_heads, d_head], K << S
a = Softmax(s)
o = a @ V_draft
# ... FFN + LM Head → next draft token

# 新 token 的 KV 追加到 K_draft, V_draft（而非重新压缩）
# 保证 K_draft 始终保持 K 核心 + 最近新增 token
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Self-speculation 在 MagicDec 中通过 GPT-Fast 后端实现。预填充阶段通过 SnapKV/StreamingLLM 选择压缩 KV 索引 → 构建 K_draft, V_draft。解码阶段 draft step 复用 target model weights + 压缩 KV attention，verify step 使用完整 KV + FlashInfer attention。KV 预算 K 由 MagicDec 框架根据模型/硬件/任务通过公式 (4) 优化选择。开源：https://github.com/Infini-AI-Lab/MagicDec。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding

## Draft KV Budget Selection（草稿KV预算选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Draft KV Budget K 是 MagicDec 中用于控制压缩 KV cache 大小的关键超参数，决定 draft model 可访问的历史 token 数量。K 的选择直接影响 SD 加速比的三要素：(1) draft cost T_D(B,K) — K 越小，draft attention 计算和 KV loading 越少；(2) token 接受率 α(K) — K 越大，draft 看到的上下文越多，接受率越高；(3) 若使用 dynamic KV selection (如 PQCache)，搜索 cost T_select(B,S,K) 也受 K 影响。

MagicDec 的选择框架：对于给定的 batch size B 和 sequence length S，计算不同 K 下的 (draft cost, acceptance rate) 曲线，通过公式 (4) 找到最小化 T_Avg^SD/T_T 的 K*。实操中：B=32 时 SnapKV self-speculation 需要 K≥512 才能达到 speedup > 1（Figure 5c），因为接受率 α 必须超过最小阈值 α_min 才能使 Ω(γ,α) > 1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Draft KV Budget 最优选择（MagicDec 公式 4 简化版）
def select_optimal_K(B, S, model, hardware, alpha_curve):
    best_speedup = 1.0
    for K in candidate_budgets:  # [256, 512, 1024, 2049, ...]
        T_D = measure_draft_cost(B, K)       # 压缩 KV 的 draft 时间
        T_V = measure_verify_cost(B, S)      # 完整 KV 的验证时间
        T_T = measure_target_cost(B, S)      # AR 解码时间
        alpha = alpha_curve[K]                # K 对应的接受率
        for gamma in [2..12]:
            omega = (1 - alpha^(gamma+1)) / (1 - alpha)
            # 公式 (2): speedup = Ω(γ,α) * T_T / (γ*T_D + T_V)
            speedup = omega * T_T / (gamma * T_D + T_V)
            if speedup > best_speedup:
                best_speedup, best_K, best_gamma = speedup, K, gamma
    return best_K, best_gamma, best_speedup
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

K 的选择需通过 offline profiling 预先测量不同 (B, S, K) 组合下的 T_D, T_V, T_T 和 α。MagicDec 论文中 StreamingLLM-based self-speculation 使用 K=256 和 K=512 两种 budget（Figure 6），SnapKV-based 使用 K=2049（Table 6）。大 batch + 长序列时大 K 更优（memory-bound 主导，大 K 提高接受率而 draft cost 增加可忽略）。Batch 中不同序列可使用不同 K（heterogeneous batch），MagicDec 根据各序列所需的最小接受率推荐"可接受 K 预算"（Figure 5c ticked budgets）。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding

## KV Cache Bottleneck & Bottleneck Shift in LLM Inference（LLM推理中的KV缓存瓶颈与瓶颈转移）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

KV Cache Bottleneck 指在长上下文 LLM 推理中，KV cache 的内存占用和内存带宽成为限制性能的主要因素。随着 sequence length S 和 batch size B 同时增大，KV cache 总大小 = B × S × n_layers × 2 × n_kv_heads × d_head（bf16 格式），KV cache 的加载时间随 B 和 S 线性增长，最终超过模型参数加载和计算时间，使推理从 compute-bound 转向 memory-bound。

Bottleneck Shift 描述了推理瓶颈随 (B, S) 变化的转移过程：短序列时，线性层（MLP + query/key/value projection）计算量随 B 增大 → compute-bound；长序列时，KV cache size 膨胀超过 parameter size → memory-bound。MagicDec 利用这一瓶颈转移来解释为何 SD 在长序列大 batch 下重新有效——当 KV loading 是瓶颈时，验证成本 T_V 的主要部分（KV loading）与目标解码 T_T 相同，T_V/T_T ≈ 1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Bottleneck 判定（基于 roofline model）
# 对每次 decode step:

# Compute time:
FLOPs_per_step = B * (2 * d_model^2 + 2 * S * d_model)  # MLP + Attention FLOPs
compute_time = FLOPs_per_step / GPU_peak_FLOPS

# Memory time:
# Model params loading: n_layers * (4 * d_model^2) bytes (bf16)
# KV cache loading: B * S * n_layers * 2 * n_kv_heads * d_head * 2 bytes (bf16)
total_bytes = param_bytes + kv_cache_bytes
memory_time = total_bytes / GPU_memory_BW

if memory_time > compute_time:
    bottleneck = "memory-bound"   # KV cache 是瓶颈
    # T_V/T_T ≈ 1, SD 有效
else:
    bottleneck = "compute-bound"   # 线性层计算是瓶颈
    # T_V/T_T > 1, SD 大 batch 下失效

# S_inflection: 对于给定 B, 使得 memory_time == compute_time 的 S
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Roofline model 判断 bottleneck：计算 arithmetic intensity = FLOPs / bytes_transferred，若低于 GPU 的 FLOPS/BW 拐点则为 memory-bound。MagicDec Figure 3c 展示了 LLaMA-2-7B 和 LLaMA-3.1-8B 在不同 S 下的 arithmetic intensity。实操：通过 profiling 测量不同 (B, S) 下的 compute time vs memory loading time 比例（Figure 1a），判断是否进入 memory-bound regime。MagicDec 利用 bottleneck shift 决定何时对当前 (B, S) 启用 SD——仅在 memory-bound（S > S_inflection）时启用 SD 可实现 speedup > 1。

涉及论文标题：
- MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding


## Locality Sensitive Hashing (LSH)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Locality Sensitive Hashing (LSH) 是一族哈希函数，其核心性质是：相似的输入向量以更高的概率被映射到相同的哈希码，而不相似的向量以更低的概率碰撞。LSH使用两个超参数(K, L)：L张哈希表独立构建，每张表使用K个独立的随机哈希函数将高维向量投影到整数哈希码。LSH最初用于近似最近邻搜索(ANN)，能够在亚线性时间内检索与查询向量相似的数据点。MagicPIG首次将LSH用于decoder-only LLM的self-attention采样估计。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**LSH在MagicPIG attention估计中的pipeline**：

```
// 预处理阶段：为每个attention head构建L张哈希表
For each head:
  K_cache_centered = K - mean(K)  // centering
  For table l in 1..L:
    For each key k_i in K_cache_centered:
      hash_code = Sign(k_i @ W_l)  // W_l ∈ R^{d×K}, K-bit hash
      HT[l][hash_code].append(i)   // 存储key索引

// 解码阶段：每步基于LSH采样估计attention output
Input: q ∈ R^{1×d}, W ∈ R^{d×(K×L)}, HT (L hash tables)
Output: attention output estimate ō

// Step 1: GPU计算query哈希码
q_code = Sign(q @ W)  // K×L bit

// Step 2: CPU查询哈希表，收集采样集合S
S = {}
For each table l in 1..L:
  candidates = HT[l][q_code[l*K:(l+1)*K]]
  For each idx in candidates:
    collision_count[idx] += 1
For each idx where collision_count[idx] >= 2:  // 至少2表碰撞
  S.insert(idx)

// Step 3: 计算采样概率（基于SimHash碰撞概率）
For each i in S:
  p_i = 1 - (1/π) * arccos(q·k_i / (|q|·|k_i|))
  u_i = 1 - (1-p_i^K)^L - L·p_i^K·(1-p_i^K)^{L-1}

// Step 4: Self-normalized Importance Sampling估计
w_S = q @ K[S]^T / √d
ō = Softmax(w_S - log(u)) @ V[S]
```

术语一般如何实现？如何使用？

LSH的典型超参数选择：K=8~10（手动ablation确定），L基于目标计算预算调整（如K=10时L=150对应2%计算量）。K控制空间划分精度——K太小则采样过多不相关key（增加计算），K太大则碰撞概率低。L增加可以弥补K较大时碰撞概率低的问题，但增加DRAM开销（哈希表内存随L线性增长，如Llama-3.1-8B 96K context下(10,150)配置需14GB）。SimHash是目前最常用的余弦LSH家族；更高级的LSH如Cross-polytope hash可进一步减少哈希表大小。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---

## SimHash

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

SimHash (Charikar, 2002) 是基于余弦相似度的LSH家族。对于向量x∈R^d，SimHash生成随机超平面w（从标准正态分布采样），返回Sign(w^T x)。两个向量x,y共享相同符号当且仅当随机投影不落在它们之间，概率为p = 1 - θ/π，其中θ = arccos(x·y/(|x|·|y|))。如果使用L张哈希表，每张K个随机哈希函数，y被查询x检索到的概率为1 - (1-p^K)^L。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**SimHash在LSH采样中的概率计算**：

```
// 给定: q∈R^d, k_i∈R^d, 随机投影矩阵W∈R^{d×(K×L)}
// q和k_i共享K-bit哈希码的碰撞概率:
cos_sim = (q·k_i) / (|q| * |k_i|)  // 余弦相似度
p_i = 1 - arccos(cos_sim) / π       // 单hash函数碰撞概率
// 至少2张哈希表碰撞的采样概率:
u_i = 1 - (1-p_i^K)^L - L·p_i^K·(1-p_i^K)^{L-1}
```

**关键性质**：u_i随q与k_i的余弦相似度单调递增——越相似的key越容易被采样，符合importance sampling要求。

术语一般如何实现？如何使用？

MagicPIG中，GPU侧对所有attention head共享随机投影矩阵W，K×L个随机向量，内存开销400KB~825KB（K=10,L=150时384KB）。SimHash的哈希函数计算（Sign(q@W)）是矩阵乘法和符号运算，适合GPU并行。数据预处理需要进行key centering——因为LLM中key向量集中在与query向量几乎相反的方向（attention sink几何），不centering则几乎所有key的碰撞概率都接近0。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---

## Self-normalized Importance Sampling (for Attention Estimation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Self-normalized Importance Sampling (SNIS) 是从提议分布u中采样来估计未知目标分布w期望的统计方法。在attention估计中，目标是对w = Softmax(qK^T/√d)下value的期望o = E_{i~w}[v_i]，但由于计算w需要所有qk_i^T内积（计算量O(nd)），无法直接获得。SNIS允许从提议分布u_i中采样B个索引i_1,...,i_B，然后用$\bar{o} = \sum_{j=1}^B (\tilde{w}_{i_j}/u_{i_j}) v_{i_j} / \sum_{j=1}^B (\tilde{w}_{i_j}/u_{i_j})$ 估计attention output，其中$\tilde{w}_i = \exp(qk_i^T/\sqrt{d})$是未归一化的attention score。该估计器有性质P[lim_{B→∞} X^{IS} = o] = 1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**SNIS在MagicPIG attention中的实现**：

```
// Self-normalized Importance Sampling for attention
// 提议分布u来自LSH (SimHash碰撞概率)
S = LSH_Sample(q, K, V, HT)  // 采样得到的key索引集合

unormalized_w = []  // 未归一化权重
sample_prob = []    // 采样概率
for idx in S:
  unormalized_w.append(exp(q @ k_idx / sqrt(d)))
  sample_prob.append(u_idx)  // u_i = LSH采样概率

// SNIS估计器（公式9的变体）
weighted_sum_V = 0
weight_sum = 0
for j in range(|S|):
  weight = unormalized_w[j] / sample_prob[j]
  weighted_sum_V += weight * v_S[j]
  weight_sum += weight

o_hat = weighted_sum_V / weight_sum  // Self-normalized
```

术语一般如何实现？如何使用？

在MagicPIG中，SNIS的提议分布u来自LSH SimHash碰撞概率u_i = 1 - (1-p_i^K)^L - L·p_i^K·(1-p_i^K)^{L-1}。由于u_i与qk_i^T/√d（在centering和范数归一化后等价于余弦相似度）单调相关，u近似满足最小方差条件u ∝ w_i|v_i-o|（因log|v_i-o|波动远小于qk_i^T/√d）。"至少2表碰撞"机制（而非标准SimHash的≥1表）极大提升了采样质量——降低了对低相似度key的采样概率。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---

## Oracle Sampling Estimation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Oracle Sampling Estimation是将attention output视为从attention score分布w中独立同分布采样的value期望值o = E_{i~w}[v_i]，然后通过Monte Carlo估计ô = (1/B) Σ_{j=1}^B v_{i_j}。称为"Oracle"是因为它假定了attention score分布w是已知的——在实际稀疏attention中w需要计算所有qk_i^T才能获得，因此oracle sampling不能直接实用化（只能节省wV计算，不能节省qK^T计算，最多2× wall-clock加速）。尽管有重复采样（Theorem 3.3保证实际计算量|S| ≤ 1+B·ε），Oracle采样理论上无偏（Theorem 3.2）且比TopK减少最多4×估计误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Oracle Sampling Estimation (预先知道w)
// w = Softmax(qK^T/√d)  // "Oracle"已知
o_hat = 0
S = {}  // 去重后的采样集合
counts = {}  // 重复计数
for j in 1..B:  // B = 采样预算
  i = CategoricalSample(w)  // 从w中采样
  counts[i] = counts.get(i, 0) + 1
  S.add(i)

for i in S:
  o_hat += (counts[i] / B) * v_i  // Equation 5

// Theorem 3.2: E[ô] = o (无偏)
// Theorem 3.3: E[|S|] ≤ 1 + B·(1-max_i w_i)
//   当w峰值明显时，实际计算量远小于B
```

术语一般如何实现？如何使用？

Oracle Sampling在MagicPIG中作为理论motivation而非实用方法。论文通过它证明：(1) 采样估计可以超越TopK的准确率上限；(2) 即使采样预算B很小（如0.002%上下文），oracle sampling仍能保持高准确率。MagicPIG通过LSH近似oracle sampling：用SimHash碰撞概率构造提议分布u，逼近w的分布形状，实现Self-normalized Importance Sampling。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---

## TopK Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

TopK attention是一种稀疏注意力近似方法：仅选择attention scores最高的K个key-value对参与加权平均计算，丢弃其余token的贡献。数学上，设w_{r_1} > ... > w_{r_K} > ... > w_{r_n}为排序后的attention scores，则TopK attention的计算为o^{TopK} = Σ_{i=1}^K w_{r_i} v_{r_i} / Σ_{i=1}^K w_{r_i}。Quest、Loki等方法是TopK attention的搜索近似（用近似搜索替代精确TopK排序以降低检索开销）。TopK attention是有偏估计——丢弃低score tokens的系统性偏差无法通过增加K来消除（除非K=n）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**TopK attention的缺陷分析**（MagicPIG Section 3）：

1. **长尾分布问题**：在许多层，Top20% tokens仅覆盖70~80% attention scores（Figure 2a），丢弃的30~20% scores导致不可忽略的估计误差（15-20%，Figure 4）。
2. **Attention Sink误导稀疏性**：首token（attention sink）吸收了大部分attention mass，使分布看起来稀疏，但剩余token间分布更均匀（Figure 2b）。
3. **搜索开销大**：IVF等搜索方法需要访问>30%的key states才能获得精确TopK（Liu et al., 2024a）。

**下游任务退化**：在聚合任务（Common/ Frequent Word Extraction）中，即使exact TopK也严重退化（Figure 1, Figure 9b-c）。检索任务（Needle-in-a-Haystack）中TopK表现可接受，因为所需信息集中在少数token上。

术语一般如何实现？如何使用？

Quest (Tang et al., 2024) 是TopK搜索的代表实现：将KV cache按page_size分页，计算q与每页summary的内积近似估计该页的重要性，TopK页被选中参与attention。page_size=16时检索开销Cost_1=1/16=6.25%。但Quest在lm-eval-harness中期上下文任务（GSM8K, COQA, MMLU）上准确率显著低于MagicPIG（Table 1）。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---

## Key Centering (for LSH-based Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Key Centering是MagicPIG中在LSH哈希表构建前对key向量进行的预处理操作：对每个attention head的K cache进行中心化，即k̄_i = k_i - (1/n)Σ_{j=1}^n k_j。由于Softmax对输入同时加常数具有平移不变性（Softmax(q·(K + c)/√d) = Softmax(qK^T/√d + constant) = Softmax(qK^T/√d)），centering不改变attention计算的数学结果。该操作的必要性源于LLM中key向量的几何特性——key平均方向k_avg与attention sink的key方向k_sink几乎相反（余弦相似度-0.9~-0.8），且k_sink的朝向在不同输入下几乎不变（相似度>0.99）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Key Centering预处理
// K = [k_1, k_2, ..., k_n] ∈ R^{n×d}
k_mean = (1/n) * Σ_{i=1}^n k_i
K_centered = K - k_mean  // 每行减去均值

// 不centering的后果（Figure 9a）：
//   - q和k方向几乎相反 → 随机投影无法区分key
//   - <0.1%的key能被query采样
//   - 检索任务准确率降至接近0
//   - 聚合任务准确率降至65%

// centering后的效果：
//   - key分布在query周围，随机投影能有效区分
//   - 准确率恢复到接近全注意力水平
```

术语一般如何实现？如何使用？

Centering是MagicPIG成功的关键ablated组件。论文Ablation（Section 5.3, Figure 9a）验证：不centering时准确率在检索(NIAH)中降至接近0，FWE降至65%。Centering后key向量不再集中在query的相反方向，SimHash投影能有效区分不同key的相似度。该操作在所有attention head上独立执行，计算量为O(nd)，在KV cache构建时一次性完成，不影响解码效率。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

## Mixture of Sparse Attention (MoSA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Mixture of Sparse Attention (MoSA) 是一种受 Mixture of Experts (MoE) 中 Expert-Choice Routing 启发的可学习内容感知稀疏注意力方法。每个 attention head 配有一个可学习的路由权重矩阵 W^r ∈ R^h，通过 sigmoid 函数 σ(x)=1/(1+e^{-x}) 计算每个 token 的选择得分 r=σ(XW^r) ∈ R^T，然后用 TopK 选择得分最高的 k 个 token，仅对这些被选 token 计算 Q、K、V 投影和 attention 矩阵。未被选中的 token 在该 head 的输出中填 0。所有 head 的输出求和，构成 MoSA 层的最终输出。

与标准 dense attention 的 O(T²) 复杂度相比，MoSA 将每 head 的复杂度降至 O(k²+T)：投影成本从 8hh'T 降至 8hh'k，attention 从 4h'T² 降至 4h'k²，额外的 routing overhead 为 2hT + h'k。由于 k << T，节省的 FLOPs 用于增加注意力头数（从 9 增至数百），实现更细粒度的 head 专业化。

核心设计要点：(1) router 输出 r_topk 在 attention 之后通过 diag(r_topk)·A 乘到输出上，使路由决策可通过梯度下降端到端学习；(2) causal mask 基于 token 原始位置索引而非子集位置：M_{a,b}=0 iff I_a≥I_b else -∞；(3) RoPE 旋转角度同样基于原始位置索引，保证位置编码的一致性；(4) Expert-Choice 路由天然保证完美负载均衡——每个 head 恰好处理 k 个 token，无需 auxiliary load-balancing loss；(5) 混合架构：保留 4 个 dense head 提供全局信息流和训练稳定性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# MoSA 单层前向
Input: X ∈ R^{T×h}

for each head i ∈ {1..H}:
  # Step 1: Token Selection via Router
  r = σ(X @ W^r_i)              # r ∈ R^T
  r_topk, I = TopK(r, k)        # I ∈ {0..T-1}^k

  # Step 2: Gather selected tokens
  X^s = X[I]                     # X^s ∈ R^{k×h}

  # Step 3: Q/K/V projections (only on k selected tokens)
  Q = X^s @ W^Q_i               # Q ∈ R^{k×h'}
  K = X^s @ W^K_i               # K ∈ R^{k×h'}
  V = X^s @ W^V_i               # V ∈ R^{k×h'}

  # Step 4: Causal mask based on original positions
  M[a,b] = 0 if I[a] >= I[b] else -∞

  # Step 5: Sparse Attention
  A = softmax(Q @ K^T / √h' + M) @ V  # A ∈ R^{k×h'}

  # Step 6: Router gating + output projection
  X^o = diag(r_topk) @ A @ W^O_i

  # Step 7: Scatter back to full sequence
  Y[j] = X^o[idx] if j == I[idx] else 0

Output: Y = Σ_{i=1..H} Y_i
```

**FLOPs 成本模型**：
- Dense head: FLOP = 8hh'T + 4h'T²
- MoSA head: FLOP = 8hh'k + 4h'k² + 2hT + h'k
- 当 T=1024, k=32, h=1024, h'=64: dense head ≈ 0.805 GFLOPs, MoSA head ≈ 0.019 GFLOPs (42x reduction)

术语一般如何实现？如何使用？

MoSA 使用纯 PyTorch 实现（einsum/scatter/gather），无需专用 CUDA kernel。开源代码：https://github.com/piotrpiekos/MoSA。Router 权重 W^r 与 Q/K/V/O 投影共同通过语言模型目标优化。IsoFLOP 实验中，首个 token 始终被所有 head 选中（attention sink 效应）。下游短序列任务中，自适应调整 k = max(floor(T/ρ), 2)。论文用 C4 数据集训练，T=1024，sparsity ρ=T/k 从 1 到 256。最佳 perplexity 在 ρ≈64 处取得，Small 模型（113M）perplexity 从 16.01 (dense) 降至 12.85 (-19.7%)。KV-cache 在 perplexity-matched 设定下减少 51-70%。

涉及论文标题：
- Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing

## Expert-Choice Routing for Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Expert-Choice Routing 最初由 Zhou et al. (NeurIPS 2022) 在 Mixture of Experts 中提出，核心思想是反转传统 token-choice routing：不是让每个 token 选择 top-k 个专家，而是让每个专家从 batch 中选择自己偏好的 top-k 个 token。这天然保证每个专家处理恰好 k 个 token，实现完美负载均衡，无需 auxiliary load-balancing loss。

MoSA 将此范式移植到 attention 机制：每个 attention head 作为一个"专家"，通过可学习的 per-head router 从输入序列中选择自己需要处理的 k 个 token。与标准 MoE 中 Expert-Choice 的关键区别：MoSA 在每个序列内独立选择 token（而非跨 batch），且选择基于 per-head scoring function 而非跨 head 共享的 gating network。

与传统 token-choice routing（如 Switch Transformer）相比：(1) 完美负载均衡——每个 head 恰好 k 个 token；(2) 动态计算分配——重要 token 可被多 head 选中获得更多计算；(3) 避免 expert collapse。代价是某些 token 可能不被任何 head 选中（需 dense head 兜底）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Expert-Choice vs Token-Choice Routing in Attention

# Token-Choice (Switch Transformer style):
#   gate = softmax(X @ W_gate)          # [T, E]
#   selected = topk(gate, k_experts)     # per-token
#   问题: 某些 expert 可能被大量 token 选中 → collapse
#   需要 auxiliary loss: L_balance = α·E·Σ_i f_i·P_i

# Expert-Choice (MoSA style):
#   for each head i:
#     scores = σ(X @ W^r_i)              # [T], sigmoid
#     tokens_i = topk(scores, k)          # head selects k tokens
#   优势: |tokens_i| ≡ k → 完美均衡
#   代价: ∃j 可能不被任何 head 选中

# 对比: Routing Transformer (online K-means):
#   centers_i = EMA of similar tokens    # 慢收敛
#   tokens_i = argmin dist(Q_j, centers)  # 需要先算全部 Q
#   FLOP 远高于 MoSA（投影 T 级 vs k 级）
```

术语一般如何实现？如何使用？

在 MoSA 中，Expert-Choice Routing 通过 per-head W^r 实现，sigmoid 激活（非竞争，遵循 σ-MoE 发现），避免 softmax 造成的 token 间竞争。Router 输出 r_topk 在 attention 后 gating 输出使梯度可反向传播。训练中 teacher-forcing 使 TopK 可看到全部 token；推理中需像 Mixture-of-Depths 学习自回归 router 预测 token 被选中的概率（论文列为 future work）。MoSA 不需要 auxiliary load-balancing loss。

涉及论文标题：
- Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing

## Hybrid Sparse-Dense Attention Architecture

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hybrid Sparse-Dense Attention Architecture 是一种将少量 dense attention head 与大量 sparse attention head 组合同一层中的架构设计。在 MoSA 中，hybrid 模型保留少量 dense head（实验确定最优为 4 个），其余 head 替换为 MoSA sparse head。Dense head 计算全部 T 个 token 的标准 attention（T×T），sparse head 仅处理 k << T 个 token（k×k）。

核心动机：纯 sparse attention 存在 router-attention 联合训练的稳定性问题——训练初期 router 随机选择，attention 学不到有用模式，导致 router 无梯度信号，形成恶性循环。少量 dense head 提供稳定的全局梯度流和语义信息，帮助 router 收敛到有意义的 token selection。

ablated 结论：(1) 0 dense head → 性能崩溃（Tiny 模型 ρ=16 时 ppl 从 22.46 升至 29.76）；(2) optimal dense head count = 4，与 sparsity ρ 无关；(3) >4 dense head → 占用 FLOP budget 导致可用的 MoSA head 减少，perplexity 回升。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Hybrid MoSA Layer
Input: X ∈ R^{T×h}

# Dense Heads (H_dense = 4, standard MHA)
Y_dense = 0
for i in 1..H_dense:
  Q, K, V = X @ W^Q_i, X @ W^K_i, X @ W^V_i  # all T tokens
  A = softmax(Q @ K^T / √h' + M_causal) @ V
  Y_dense += A @ W^O_i

# MoSA Sparse Heads (H_mosa, k = T/ρ tokens per head)
Y_sparse = 0
for i in 1..H_mosa:
  r = σ(X @ W^r_i); r_topk, I = TopK(r, k)
  X^s = X[I]
  Q, K, V = X^s @ W^Q_i, X^s @ W^K_i, X^s @ W^V_i
  M[a,b] = 0 if I[a] >= I[b] else -∞
  A = softmax(Q @ K^T / √h' + M) @ V
  X^o = diag(r_topk) @ A @ W^O_i
  Y_sparse = scatter_add(Y_sparse, X^o, I)

Output: Y = Y_dense + Y_sparse
```

**FLOP-matching 规则**：max H_mosa s.t. H_dense·FLOP_dense + H_mosa·FLOP_mosa ≤ H_baseline·FLOP_dense

术语一般如何实现？如何使用？

实现：修改 Transformer layer，同时实例化 dense heads（标准 MHA 逻辑）和 MoSA heads（router + sparse attention）。Optimal dense head count 通过 ablating ρ=4 和 ρ=16 在不同 dense head count (0-9) 下的 perplexity 确定——结果一致为 4，说明稳定化效果与 sparsity 无关。KV-cache: KV_total = T·H_dense + k·H_mosa，在 perplexity-matched 设定下 KV-cache 减少 51-70%。训练：与标准 transformer 相同（Adam, lr=0.00025, gradient clipping 0.25, warmup 4k steps）。

涉及论文标题：
- Mixture of Sparse Attention: Content-Based Learnable Sparse Attention via Expert-Choice Routing

## Mixture of Block Attention (MoBA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

MoBA（Mixture of Block Attention）是一种将 Mixture of Experts (MoE) 的专家路由原理从 FFN 层迁移到 attention 层的稀疏注意力架构。核心思想：将长上下文的 Key-Value 序列划分为等大小的 blocks，每个 query token 通过 parameter-free gating（query 与 mean-pooled K per block 的内积）计算与每个 block 的 affinity score，再用 top-k gating 选择最相关的 k 个历史 blocks 进行 attention 计算。MoBA 与 full attention **参数等价**（0 参数增量），支持训练中 MoBA↔Full Attention 无缝切换。

从算法pipeline角度拆解术语：MoBA 位于 Transformer 的 attention 层，直接替换标准 scaled dot-product attention，不影响其他层（FFN、LayerNorm）。计算流程为 Block Partitioning → Mean Pooling Key Representation → Gating Scores → Causal Top-k → Block-wise FlashAttention Varlen → Online Softmax Combining。

具体例子（1M context prefill, B=4096, k=12）：
```
输入：Q, K, V ∈ R^{N×h×d}, N=1M, h=32, d=128
n = N/B = 1M/4096 ≈ 244 blocks

# Step 1-2: Block partition + mean pool
K̄ = mean_pool(K.reshape(n, B, h, d), dim=1)  # [n, h, d] = [244, 32, 128]

# Step 3: Gating scores
S = einsum('nhd,mhd->nhm', Q, K̄)  # [1M, 32, 244], O(N·n·d) ≈ 31G FLOPs

# Step 4: Causal mask + topk
M[pos, :, i] = -inf if pos < i*B  # mask future blocks
G = topk(S + M, k=12, dim=-1)    # [1M, 32, 12] sparse indices

# Step 5: Block-wise varlen FA
# Current block (always causal): 1 block × 4096 tokens = 4K tokens
# Selected history blocks: 12 blocks × 4096 tokens = 49K tokens
# Total attention tokens = 53K per query (5.3% of 1M)

# Step 6: Online softmax combine
lse_s, lse_m = logsumexp from each partial attention
O = (exp(lse_s-lse_total)·O^s + exp(lse_m-lse_total)·O^m)
```
复杂度从 O(N²·d) 降至 O(k·B·N·d)，sub-quadratic。Sparsity = 1-(k+1)B/N = 94.7%。

术语一般如何实现？如何使用？

基于 PyTorch + FlashAttention + DeepSpeed-MoE 实现。开源：https://github.com/MoonshotAI/MoBA。核心 CUDA-level 优化包括：block-split + index_select（MoE-style token dispatch）、varlen FlashAttention（不同 block 的 query 数量不同）、online softmax combine（tiling 保证数值等价）。已部署于 Kimi 长上下文请求。适用于 continued pre-training 扩展已有模型 context length（如 Llama-8B→1M）。典型超参：block_size=4096, top-k=12, layer-wise hybrid（最后 3 层 full attention）。

涉及论文标题：
- MoBA: Mixture of Block Attention for Long-Context LLMs

## Top-k Gating for Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Top-k Gating for Attention 是将 MoE 中 top-k gating 机制应用于 attention 的 block 选择技术。每个 query token 计算与所有 KV blocks 的 affinity score 后，通过 top-k 选择最相关的 blocks 进行 attention。与 MoE 中 expert routing 的区别：这里路由的是 attention context（KV blocks），而非 FFN experts；gating 是 parameter-free 的（使用 query-key 内积而非可学习权重矩阵）。

从算法pipeline角度拆解术语：
```
s_i = ⟨q, mean_pool(K[I_i])⟩  # affinity score, query-to-block relevance
g_i = 1 if s_i ∈ Topk({s_j | j∈[n]}, k) else 0  # binary gating
I = ∪_{g_i>0} I_i  # selected KV indices for this query
```
Critical design elements:
- **Causality enforcement**: s_i = -∞ for future blocks (pos(q) < i·B)
- **Current block as shared expert**: 强制 g_i = 1 for query's own block, 类比 MoE shared expert
- **Parameter-free**: 不引入可训练参数，gating 仅依赖 Q 和 block-mean-pooled K

MoBA 证明 sliding window attention 和 attention sink 都是 top-k gating 的特例——gating 固定选择最近 blocks（SWA）或首尾 blocks（Sink）。

术语一般如何实现？如何使用？

实现为矩阵操作：\(S = Q @ K̄^T\)（batch matmul），\(G = \operatorname{topk}(S + M, k)\)（GPU topk）。在 MoBA 中使用，block_size 和 top-k 是主要超参。典型配置：训练用 B=512/k=3（short context），推理用 B=4096/k=12（1M context）。额外计算开销 <1% attention FLOPs。

涉及论文标题：
- MoBA: Mixture of Block Attention for Long-Context LLMs

## Hybrid Attention Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hybrid Attention Training 是利用 MoBA 与 full attention 参数等价性，在训练中在稀疏（MoBA）和稠密（Full）attention 模式间切换的策略。核心优势：MoBA 提供训练效率，full attention 恢复质量，两者共享同一套模型参数无需转换。

从算法pipeline角度拆解术语：
两种子策略：

**1. Two-stage MoBA/Full Hybrid**：
```
Stage 1: train(MoBA, 90% tokens)  → 高效长上下文训练
Stage 2: train(Full, 10% tokens)  → 恢复完整 attention 能力
结果：position-wise LM loss ≈ pure Full Attention
无 loss spike during switching
```

**2. Layer-wise Hybrid**：
```
前 L-N 层：MoBA（稀疏）
后 N 层：  Full Attention（稠密）
```
动机：SFT 中 prompt tokens 的 loss 被 mask，导致 MoBA 的稀疏梯度无法有效 backprop。最后几层 full attention 提供 dense gradient path。实验（Figure 5b）显示 SFT loss 随 full attention 层数增加而单调下降。

术语一般如何实现？如何使用？

训练脚本中通过 schedule 控制 attention_mode 切换。推理时 prefill 用 MoBA（加速），decoding 用 full attention（保证质量）。典型配置：Llama-8B 从 128K→1M continual pre-training，block_size=4096, top-k=12, 最后 3 层 full attention。

涉及论文标题：
- MoBA: Mixture of Block Attention for Long-Context LLMs

## Fine-Grained Block Segmentation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Fine-Grained Block Segmentation 是 MoBA 中借鉴 MoE fine-grained expert segmentation 的设计选择：将 context 划分为更细粒度的 blocks（更多但更小的 blocks），同时按比例增加 top-k 选择数量以保持相同的 sparsity。MoBA 实验证明更细的 block 粒度显著提升性能——32K context 从 8 blocks (B=4096) 细分到 128 blocks (B=256)，维持 sparsity=75%，LM loss 降低 ~0.01。

从算法pipeline角度拆解术语：
```
粗粒度：8 blocks × 4096 tokens/block, top-k=2 → 关注 3×4096=12K tokens
细粒度：128 blocks × 256 tokens/block, top-k=32 → 关注 33×256=8.4K tokens
两者 sparsity 相同（75%），但细粒度允许 gating 更精准地选择相关信息
```
类似 MoE 中 fine-grained experts 允许更灵活的 expert 组合，细分 blocks 允许 query 更精准地挑选相关的 context 子区间。

术语一般如何实现？如何使用？

通过调整 block_size 和 top-k 超参控制。平衡点：block_size 太小会增加 gating 计算开销（n=N/B 变大，S ∈ R^{N×n} 变大）；block_size 太大则选择粒度粗。MoBA 实验建议 B=512-4096 范围，取决于 context length 和 GPU memory。与 FlashAttention tiling 兼容——block_size 应为 FlashAttention tile size 的倍数以最大化 kernel efficiency。

涉及论文标题：
- MoBA: Mixture of Block Attention for Long-Context LLMs

---

## Gated DeltaNet

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Gated DeltaNet 是将 delta rule（增量学习规则）与门控机制结合的线性序列模型，是 MoM 论文使用的默认 memory update 机制。其 memory update rule 为：

$$M_t = a_t (I - k_t^T k_t) M_{t-1} + b_t k_t^T v_t$$

其中 a_t, b_t ∈ (0,1) 是数据依赖的标量门控参数，I 是 d×d 单位矩阵。与标准 DeltaNet 的区别在于增加了额外的输入门 a_t。

Delta Rule 的核心思想：(I - k_t^T k_t) M_{t-1} 部分从当前 memory 中"减去"与当前 key k_t 相关的旧信息（类似 online linear regression 的更新），b_t k_t^T v_t 部分将当前 token 的 kv 关联写入 memory。这种设计使模型能更精确地选择性更新 memory，而非像简单 Linear Attention 那样无差别累加。

从算法pipeline角度拆解术语。

**Gated DeltaNet 在 MoM 中的应用**：

```
# 单一 memory 版本的 Gated DeltaNet:
for t in 1..T:
  k_t = x_t @ W_k           # [d]
  v_t = x_t @ W_v           # [d]
  a_t = sigmoid(x_t @ W_a)  # 数据依赖的门控
  b_t = sigmoid(x_t @ W_b)

  # Delta rule update (核心):
  decay = I - k_t^T k_t     # [d, d]
  M_t = a_t * decay @ M_{t-1}  # 每行减 k_t @ (k_t^T @ M_{t-1})
  M_t = M_t + b_t * k_t^T v_t

  q_t = x_t @ W_q
  o_t = q_t @ M_t

# MoM 多 memory 版本:
for each activated memory m:
  k_t^m = x_t @ W_k^m       # memory-specific
  v_t^m = x_t @ W_v^m
  a_t^m = sigmoid(x_t @ W_a^m)
  b_t^m = sigmoid(x_t @ W_b^m)
  M_t^m = a_t^m (I - k_t^{m,T} k_t^m) M_{t-1}^m + b_t^m k_t^{m,T} v_t^m
```

术语一般如何实现？如何使用？

Gated DeltaNet 通过 Triton chunk-wise parallel scan kernel 实现。由于 (I - k_t^T k_t) 的矩阵运算涉及 d×d 的外积和矩阵乘法，实现上使用 chunk 内并行处理来隐藏延迟。在 MoM 中每个 memory 独立执行 Gated DeltaNet update，通过 varlen kernel 处理不同 memory 各自的 token 子序列。代码开源：https://github.com/OpenSparseLLMs/MoM。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories
- Gated Delta Networks: Improving Mamba2 with Delta Rule

---

## Memory Interference in Linear Sequence Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Memory Interference（记忆干扰）是线性序列模型的核心瓶颈：当序列中所有信息被压缩到单一固定大小的 memory state M ∈ R^{d×d} 时，新 token 的 K/V 写入会覆盖或衰减之前存储的信息表示。这导致模型在 recall-intensive 任务上的表现远不如为每个 token 维护独立 KV cache 的 Transformer。

数学模型：在线性注意力中，M_t = M_{t-1} + k_t^T v_t。当新的 k_t^T v_t 包含与已存储信息正交或冲突的内容时，无法将两者同时保留——它们被"搅拌"到同一个 memory state 中。即使有 forget gate a_t，也只是整体衰减，无法选择性保留特定信息。

MoM 通过多 memory 分离解决此问题：token 被路由到不同 memory，非激活 memory 保持 M_t^m = M_{t-1}^m 不变。这实现了"信息隔离"——不同类型的信息存储在不同的 memory 中，互不干扰。MoM 论文实验（Table 5）验证了各 memory 确实形成了专业化：Memory-1 偏好基础词/动词，Memory-2 偏好专有名词/科技术语，Memory-3 偏好技术术语/形容词，Memory-4 偏好疑问词/不完整名词。

从算法pipeline角度拆解术语。

```
# 单一 memory (有记忆干扰):
M_t = a_t * M_{t-1} + k_t^T v_t
# 问题: k_t^T v_t 会与 M_{t-1} 中任意行产生交互，
# M_{t-1} 中与 k_t 相似的信息被加强，相异的被稀释

# MoM 多 memory (无干扰):
if token t routed to memories {2, 4}:
  M_t^2 = GatedDeltaNet(M_{t-1}^2, k_t^2, v_t^2)   # 仅更新 Memory-2
  M_t^4 = GatedDeltaNet(M_{t-1}^4, k_t^4, v_t^4)   # 仅更新 Memory-4
  M_t^1 = M_{t-1}^1  # 未激活，保持不变——无干扰！
  M_t^3 = M_{t-1}^3  # 未激活
```

术语一般如何实现？如何使用？

Memory Interference 的缓解方法分两类：(1) Gating-based: 通过 forget gate / input gate 控制信息衰减（GLA, HGRN2, G-DeltaNet 等）；(2) Separation-based: 通过多 memory 分离不同信息（MoM）。两种方法互补——MoM 在 Gated DeltaNet 基础上叠加多 memory 分离。Router 的 auxiliary loss 确保负载均衡，防止某些 memory 成为 bottleneck。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---

## Shared Memory (in MoM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Shared Memory 是 MoM 架构中的一个特殊 memory state，始终被所有 token 激活（不走 Router 的 top-k 筛选），用于捕获全局序列信息。设计动机：多 memory 分离虽然消除了 memory interference，但也导致每个 memory 只能看到部分序列。Shared memory 获取完整序列上下文，弥补分离式 memory 可能丢失的跨 memory 长程依赖。

Shared memory 的灵感来自 DeepSeek-MoE 中的 shared experts——捕获跨不同上下文的共性知识。在 MoM 中，shared memory 作为所有 memory 的"背景知识库"，输出时与 top-k activated memories 的输出一起做加权混合。

从算法pipeline角度拆解术语。

```
# MoM 含 Shared Memory 的前向流程:
M_shared = 0                                    # d×d
M_1...M_M = 0                                   # M 个 d×d

for t in 1..T:
  # Router: 选 top-k 个 memory
  scores = TopK(softmax(x_t @ W_g), k)

  # Shared memory: 始终更新
  k_t^shared = x_t @ W_k^shared
  v_t^shared = x_t @ W_v^shared
  M_shared = GatedDeltaNet(M_shared, k_t^shared, v_t^shared)

  # Top-k memories: 选择性更新
  for m in topk_indices:
    M_m = GatedDeltaNet(M_m, k_t^m, v_t^m)

  # 混合输出:
  M̃_t = Σ g_t^{(m)} · M_t^m + M_t^shared
  o_t = q_t @ M̃_t
```

术语一般如何实现？如何使用？

Shared memory 使用独立的 K/V projection weights（W_k^shared, W_v^shared），不与其他 memory 共享。MoM 实验（Table 6 ablation）证实 shared memory 对 performance 有显著增益：w/ shared memory → Recall avg 28.16, w/o shared memory → 26.06。代码开源：https://github.com/OpenSparseLLMs/MoM。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---

## Auxiliary Loss for Memory Load Balancing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Auxiliary Loss for Memory Load Balancing 是 MoM 中用于确保 Router 将 token 均匀分配到各 memory state 的辅助训练损失函数。类似 Switch Transformer 中的 load balancing loss，目标是防止某些 memory 被过度激活（"hot memory"问题）而其他 memory 空闲。

MoM 的 auxiliary loss 公式：L_aux = α · Σ_m f_m · P_m，其中 f_m 是路由到 memory m 的 token 比例，P_m 是分配给 memory m 的平均 routing probability，α 是 auxiliary loss 的 scale 系数。最小化该损失鼓励均匀路由。

从算法pipeline角度拆解术语。

MoM 实验（Table 6）测试了不同 α 值的效果：
- α = 1e-2: Recall avg 27.59
- α = 1e-3: Recall avg 28.16 (best)
- α = 0: Recall avg 27.23

结果显示合适的 auxiliary loss weight 能提升性能（过大干扰主任务学习，过小导致负载不均衡）。

术语一般如何实现？如何使用？

实现为训练循环中的额外损失项：total_loss = language_modeling_loss + α · L_aux。MoM 的 Fig 5 热力图验证了施加 auxiliary loss 后各 memory 在各层的路由分布近乎均匀。代码开源：https://github.com/OpenSparseLLMs/MoM。

涉及论文标题：
- MoM: Linear Sequence Modeling with Mixture-of-Memories

---

## Multi-head Temporal Latent Attention (MTLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Multi-head Temporal Latent Attention (MTLA) 是在 MLA 的低秩 latent KV 压缩基础上，进一步沿 temporal（时间）维度压缩 KV cache 的注意力机制。MTLA 是首个在时序维度压缩 KV cache 的自注意力变体。核心流程：

(1) **低秩 latent 压缩**（继承 MLA）：输入 X ∈ R^{T×d} 经 W_r ∈ R^{d×r} 投影为低维 latent vector C ∈ R^{T×r}（r ≪ d），LayerNorm 稳定训练；(2) **Temporal 压缩 via hyper-network**：以 hyper-network 对每 s 个相邻 latent vector 动态生成 merge weight w_i = Sigmoid(Linear(c_i) · Linear(pe_j))，合并为 ĉ_j = Σ w_i·c_i，将 KV cache 序列长度从 T 降为 t = ⌈T/s⌉；(3) **Stride-aware causal mask**（训练时）：解决 training 时 compressed cache 长度与 sequence length 不匹配的问题；(4) **Absorbed attention**：利用矩阵乘法结合律将 W_K 吸收进 W_Q、W_V 吸收进 W_O，避免显式计算完整 K/V 矩阵；(5) **Decoupled RoPE temporal compression**：RoPE key 同样沿 temporal 维压缩，每 s 个 token 仅保留最新的 RoPE key。

MTLA 的 per-token KV cache 大小 = 9d_h·l/(2s)（s=2 时为 2.25d_h·l，接近 MQA 的 2d_h·l）。Per-token 解码复杂度从 O(T) 降至 O(T/s)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**MTLA 训练前向（s=2, r=256, d=512, n_h=8）**：

```
输入: X ∈ R^{T×d}

# Step 1: Query（标准 MHA）
Q = X @ W_Q               # W_Q ∈ R^{d×(n_h·d_h)}, Q ∈ R^{T×512}

# Step 2: Low-rank latent
C = LayerNorm(X @ W_r)   # W_r ∈ R^{d×r}, C ∈ R^{T×256}

# Step 3: Hyper-network 批量生成 merge weights
PE = (pe_1,...,pe_1, ..., pe_t,...,pe_t)  # 每个 pe_j 重复 s 次, 总长 T
W = Sigmoid(Linear(PE) × Linear(C))       # W ∈ R^{T×T}
W = chunk_mask(W)        # 仅保留对角线附近 chunk
Ĉ' = W @ C               # extended compressed sequence ∈ R^{T×256}

# Step 4: Absorbed attention with stride-aware mask
scores = (X @ (W_Q @ W_K^T)) @ Ĉ'^T / sqrt(d_h)   # W_K absorbed into W_Q
# Stride-aware mask: mask[n,m] = 0 if n==m or (m<n and m%2==0) else -∞
attn = softmax(scores + mask)
output = attn @ (Ĉ' @ (W_V @ W_O))                 # W_V absorbed into W_O
```

**MTLA 推理前向（s=2，incremental decoding）**：

```
输入: x_i ∈ R^{1×d}

# Step 1: Query
q_i = x_i @ W_Q

# Step 2: Latent vector
c_i = LayerNorm(x_i @ W_r)  # c_i ∈ R^{1×256}

# Step 3: Hyper-network 生成 merge weight
j = ceil(i/2)
pe_j: positional embedding at step j
w_i = Sigmoid(Linear(c_i) · Linear(pe_j))

# Step 4: 更新 compressed KV cache
if i % 2 == 1:   # 新 slot
    Ĉ = Concat(Ĉ, w_i ⊙ c_i)
else:             # 合并到当前 slot
    Ĉ_j = Ĉ_j + w_i ⊙ c_i  # 动态融合（可覆盖之前临时值 Ĉ_j'）

# Step 5: Absorbed attention
output = softmax((x_i @ W_Q_absorbed) @ Ĉ^T / sqrt(d_h)) @ (Ĉ @ W_V_absorbed)
```

术语一般如何实现？如何使用？

MTLA 在 Fairseq 框架上实现，作为可替换的 self-attention 模块。无需修改模型其他组件（FFN、LayerNorm 等）。开源代码（含 extended FlashAttention-2 CUDA kernel）：https://github.com/D-Keqi/mtla。

MTLA 特别适用于长序列任务（speech translation/recognition/understanding、text summarisation），因为 temporal compression 在长序列场景下收益最大。s=2 时已接近 MQA 的 KV cache 水平（144l vs 128l per-token elements），同时保持 MHA 级别的质量。s=4 时 per-token cache 降至 72l elements。但 s 过大导致性能下降（s=4 BLEU 23.05 vs s=2 23.28）。

涉及论文标题：
- Multi-head_Temporal_Latent_Attention

---

## Stride-aware Causal Mask

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Stride-aware Causal Mask 是 MTLA 为实现在 parallel training 下匹配 incremental inference attention pattern 而设计的一种特殊 causal mask。MTLA 在推理时每 s 个 token 共享一组 compressed KV cache vector（temporal compression ratio=s），导致 attention mask 不是标准的三角因果掩码。若训练时简单 pre-downsample KV cache，则 query 在训练时看到的 KV 信息与推理时不一致（覆盖了尚未合并的 incomplete vectors）。Stride-aware mask 通过限制 query 仅能 attend stride 边界上的历史位置，使训练-推理 attention pattern 一致。

Stride-aware causal mask 定义（row m, col n, stride s）：mask[m, n] = 0 iff (n == m) or (n < m and n % s == 0)，否则 mask[m, n] = -∞。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**stride-aware mask 可视化（T=6, s=2）**：

```
# mask ∈ R^{6×6}, 0=允许attend, -∞=masked
#   col: 1  2  3  4  5  6
# row 1:  0 -∞ -∞ -∞ -∞ -∞   (仅自身)
# row 2:  0  0 -∞ -∞ -∞ -∞   (自身 + col 2=stride boundary)
# row 3:  0 -∞  0 -∞ -∞ -∞   (自身 + col 2)
# row 4:  0  0 -∞  0 -∞ -∞   (自身 + cols 2,4)
# row 5:  0 -∞  0 -∞  0 -∞   (自身 + cols 2,4)
# row 6:  0  0 -∞  0 -∞  0   (自身 + cols 2,4,6)
```

构造伪代码：
```
mask = full(T, T, -inf)
for m in 1..T:
    for n in 1..T:
        if n == m or (n < m and n % s == 0):
            mask[m, n] = 0
```

术语一般如何实现？如何使用？

GPU 上通过 `torch.where` 或 attention kernel 内联条件分支实现。开销与标准 causal mask 同阶 O(T²)，可 fused 进 softmax kernel。仅训练时使用；推理时 incremental cache update 自然满足 stride pattern。超参数：temporal compression ratio s。

涉及论文标题：
- Multi-head_Temporal_Latent_Attention

---

## Hyper-network for Dynamic Temporal KV Cache Merging

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hyper-network for Dynamic Temporal KV Cache Merging 是 MTLA 中用于动态生成 temporal merge weight 的轻量网络。背景：不同输入序列长度和内容各异，fixed-weight merging（如 static averaging）无法自适应。Hyper-network 以 latent vector C 为条件，通过 Sigmoid gate 输出 per-position merge weight w_i，使合并策略数据驱动。

结构（training batch）：W = Sigmoid(Linear(PE) × Linear(C)) ∈ R^{T×T}，chunk mask 后乘 C 得 Ĉ'。Inference single-token：w_i = Sigmoid(Linear(c_i) · Linear(pe_j))。两个 Linear 层各将 256-dim 映射到 64-dim。Sigmoid（非 Softmax）：不同 position 间无需归一化竞争——每个 w_i 仅控制对应 c_i 对 merged vector 的贡献比。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Training batch 伪代码（s=2, r=256）**：

```
# 输入: C ∈ R^{T×256}, pe ∈ R^{t×64}
# 参数: W_pe: 256×64, W_c: 256×64

pe_proj = pe.repeat_interleave(s, dim=0) @ W_pe    # R^{T×64}
c_proj = C @ W_c                                    # R^{T×64}
W_raw = pe_proj @ c_proj.T                          # R^{T×T}
W = Sigmoid(W_raw)
W = chunk_mask(W)  # 仅保留 chunk 内连接
Ĉ' = W @ C         # R^{T×256}
```

**Inference single-token 伪代码**：
```
c_i ∈ R^{1×256}
j = ceil(i/s)
w_i = Sigmoid(Linear_pe(pe_j) @ Linear_c(c_i).T)   # scalar weight
```

术语一般如何实现？如何使用？

超参数：两个 Linear 层 256×64 = 32768 params（<0.04% 模型总量）。训练时与主模型共同训练，无额外损失。推理每次新增 token 仅两次 Linear + Sigmoid。Chunk mask 维持 streaming 属性，避免全局依赖导致训练不稳定。Hyper-network 输出 w_i 决定 c_i 对当前 slot merged vector ĉ_j 的贡献；当 i%s==0 时 ĉ_j 固化。

涉及论文标题：
- Multi-head_Temporal_Latent_Attention

---

## Absorbed Attention (Weight Absorption in Self-Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Absorbed Attention 是利用矩阵乘法结合律，将 KV cache 的 up-projection 矩阵 W_K/W_V 分别吸收进 query projection W_Q 和 output projection W_O，避免推理时显式计算完整 K、V 矩阵的技术。由 MLA 首次引入，MTLA 继承并适配。

核心变换（MTLA 版）：
```
标准: K = Ĉ @ W_K, V = Ĉ @ W_V  → output = softmax(Q@K^T/√d) @ V @ W_O
吸收: W_Q_absorbed = W_Q @ W_K^T, W_V_absorbed = W_V @ W_O
     output = softmax(X @ W_Q_absorbed @ Ĉ^T/√d) @ Ĉ @ W_V_absorbed
```

好处：(1) 避免 up-project Ĉ(r维)→K(n_h·d_h维) 再 down-project 的冗余；(2) Ĉ 直接参与 attention，计算量减少（r vs n_h·d_h）；(3) 内存带宽节省。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**对比：r=256, n_h·d_h=512, d=512, t=T/2**：

```
# 吸收前 K^T 计算量: t·r·(n_h·d_h) = t·256·512 = 131072t FLOPs
# 吸收后 K^T 计算量: N·d·r = N·512·256 = 131072N FLOPs
# 当 N=t (decode) 时两者相同，但免去存储中间 K 的内存和带宽
# 训练时 N=T > t, 吸收后计算量稍增但内存显著减少

# 吸收前 V@W_O 计算量: t·(n_h·d_h)·d + t·r·(n_h·d_h) = t·512·512 + t·256·512
# 吸收后 V@W_O 计算量: t·r·d = t·256·512
# 节省了显式 V 生成的 t·512·512 FLOPs（一半计算量）
```

术语一般如何实现？如何使用？

Weight absorption 前提：位置编码（RoPE）不能直接施加在 latent vector 上（旋转矩阵与 up-projection 矩阵不满足交换律）。因此必须使用 decoupled RoPE——位置部分单独计算 Q_RK_R^T 后加法合并到 content attention score。吸收后权重在 training 后预计算，直接 baked into FlashAttention-2 自定义 CUDA kernel 参数。

涉及论文标题：
- Multi-head_Temporal_Latent_Attention
- TransMLA: Multi-Head Latent Attention Is All You Need

TransMLA 通过 Absorb 操作实现 GQA→MLA 转换后的推理加速。转换后，W^{UK}（NoPE 部分，移除 RoPE 后）被吸收进 query projection：q̂_{t,i} = [(W_i^{UK})^T q_{t,i}^C; q_{t,i}^R]，所有 head 共享一个 latent KV head k̂_t = [c_t^{KV}; k_t^R]，仅需缓存 r_kv 维而非 2gd 维的 KV cache。TransMLA 的 RoRoPE 技术确保转换后的 W^{UK} 不含 RoPE（位置信息集中在第一 head 独立处理），满足 Absorb 操作的前提条件。

## Unstructured KV Cache Pruning (非结构化KV缓存剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Unstructured KV Cache Pruning 是一种对 LLM 推理中 Key-Value 缓存进行压缩的技术：移除 KV cache 矩阵中任意位置的单个元素（标量），而非按 channel、token 或 block 等结构化单元剪枝。与结构化剪枝（如 ThinK 的 per-channel Key cache 剪枝，每次移除整个 channel）不同，非结构化剪枝不对稀疏 pattern 施加任何几何约束——每个元素独立判断是否保留。

Mustafar 论文核心发现：Key cache 非结构化剪枝在 70% 稀疏度下精度（LongBench avg 41.55）远优于 ThinK 50% 结构化剪枝（38.53）。Value cache 非结构化剪枝突破结构化剪枝 30% 稀疏度上限，在 70% 稀疏度下保持精度（42.78 vs Dense 43.19）。

非结构化优于结构化的原因：(1) Key cache 虽有 channel-wise outliers，但 outlier channel 内部并非所有元素都有用——结构化整 channel 丢弃损失有价值元素，整 channel 保留携带冗余；(2) Value cache 元素分布均匀无 channel outliers，结构化剪枝无法识别有效 channel 导致 30% 实用上限；(3) 元素级剪枝实现比 channel 级更精确的取舍。

与 2:4 Semi-structured Sparsity 对比：2:4 是 NVIDIA Sparse Tensor Core 的模式——每连续 4 元素恰好保留 2 个，全局 50% 稀疏度。Mustafar 实验显示同等 50% 稀疏度下非结构化精度优于 2:4（LongBench avg 42.65 vs 40.89），因为 2:4 对局部 pattern 仍施加约束。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# 非结构化KV Cache剪枝pipeline
# 输入: K_cache, V_cache ∈ R^{Tx d}, 稀疏度 s, 局部窗口 W=32

# Step 1: Per-token magnitude-based pruning
for i = 1..T-W:
    abs_K = |K_cache[i, :]|          # shape [d]
    thresh_K = np.partition(abs_K, d*s)[d*s]
    mask_K[i] = (abs_K >= thresh_K)

# Step 2: Value cache同理
for i = 1..T-W:
    abs_V = |V_cache[i, :]|
    thresh_V = np.partition(abs_V, d*s)[d*s]
    mask_V[i] = (abs_V >= thresh_V)

# Step 3: 最近W个token保持dense (mask = all 1)

# Step 4: Apply及bitmap压缩
K_sparse = K_cache * mask_K
K_compressed = bitmap_compress(K_sparse, mask_K)  # tile=1x64
```

术语一般如何实现？如何使用？

实现需求：(1) 逐元素排序/top-k选择，计算开销 O(Td)，在 prefill 结束后批量执行；(2) 稀疏存储需 bitmap 或 CSR/COO 格式——irregular pattern 无法通过简单减少 matrix dimension 压缩；(3) 需配合特殊 kernel（Mustafar SpMV、FlashLLM SpMM）将内存节省转为计算加速；(4) 50% 稀疏度实际压缩比 65%（15% metadata overhead），70% 为 45%。适用场景：长上下文 LLM decode（memory-bound），prefill 仍用 FlashAttention。开源：https://github.com/dhjoo98/mustafar。

涉及论文标题：
- Mustafar__Promoting_Unstructured_Sparsity_for_KV_Cache_Pruning_in_LLM_Inference

## Output-aware Pruning for KV Cache (输出感知的KV缓存剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Output-aware Pruning 是一种使用 scoring metric 作为 proxy 估计 KV cache 中每个元素对 attention output 贡献的剪枝技术。核心思想：不单独看元素本身的 magnitude，而是将元素与对应输入（query 或 attention score）相乘，利用矩阵乘法链式关系评估对最终 output 的实际贡献。

Mustafar 中的两种形式：

1. **Key cache output-aware**: S = |K| ⊙ broadcast(Σ_{t} |Q_t|)，累加 32 个 query 的 L1 绝对值与 Key element-wise 乘积。Q×K attention 中每个 K 元素贡献正比于 |K_j| × |Q_j|。

2. **Value cache output-aware**: 
   - Per-channel: S = |V| ⊙ broadcast(Σ|α_t|)，需 attention scores（与 FlashAttention 不兼容——FA 不物化完整 attention matrix）。
   - Per-token: 等价于 magnitude-based——同 token 内所有 V 元素乘以同一 α_i，排序不变。

从算法pipeline角度拆解：

```
# Key cache output-aware pruning score
Q_accum = sum(|Q_t| for t in 0..31)     # shape [d]
for i in 1..T-W:
    score_K[i] = |K_cache[i]| * Q_accum  # element-wise
    mask_K[i] = topk_mask(score_K[i], d*(1-s))

# Value cache per-channel output-aware (需attention scores)
Attn_accum = sum(|attn_scores[t]| for t in 0..31)  # shape [T]
for c in 1..d:
    score_V[:,c] = |V_cache[:,c]| * Attn_accum     # per-channel
```

术语一般如何实现？如何使用？

Key cache 推荐 magnitude-only（精度已足够）；Value cache 推荐 per-token magnitude（自动等价 output-aware，无需 attention scores 且与 FlashAttention 兼容）。计算开销约 O(Td) per head，通常小于 2% of attention compute。

涉及论文标题：
- Mustafar__Promoting_Unstructured_Sparsity_for_KV_Cache_Pruning_in_LLM_Inference

---

## PROXY-TOKENS EVICTION (代理Token淘汰)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PROXY-TOKENS EVICTION 是 NACL 提出的 KV Cache 淘汰评分策略。在 attention matrix A ∈ R^{p×p} 中，仅使用输入末尾的一小部分 token（proxy tokens P，~10% tokens，对应用户问题部分）的 attention scores 来评估所有 prefix token 的重要性，而非 H2O 的累加全部 token 的 attention 或 MSRNN 的仅用当前 token。

评分函数：F_score = Σ_{x_p∈P} Softmax(A(x_p, *))。F_score[j] 表示 token j 对所有 proxy tokens 的综合重要性。淘汰建模为组合优化：S_t = argmax_{S⊂R} Σ_{x∈S} F_score(A, C_p) ∪ P，其中 R = x_prompt \ P，proxy tokens P 默认保留。

直觉：proxy tokens（用户问题）的 attention pattern 反映了"哪些 prefix token 对回答当前问题有用"，比全量累加更精准。0% proxy = MSRNN（信息不足），100% = H2O（冗余干扰），~10% 最优。

从算法pipeline角度拆解术语：

```
输入: A ∈ R^{p×p}, P = {p*0.9,...,p-1}, C_p

Step 1: F_score = Σ_{x_p∈P} Softmax(A[x_p, :])     # column-wise sum over proxy rows
Step 2: R = {0,...,p-1} \ P                          # exclude proxy tokens
Step 3: S_proxy = TopK(F_score[R], C_p)              # top-C_p non-proxy tokens
Step 4: S_keep = S_proxy ∪ P                         # proxy tokens always kept
```

术语一般如何实现？如何使用？

NACL 开源实现基于 PyTorch + FlashAttention-2。encoding 阶段收集 proxy tokens 的 attention scores（通过 FlashAttention-2 backward 方式重算或仅对 proxy tokens 重算），column-wise sum 后 top-k。默认 proxy tokens 设在输入末尾（用户问题处），无法区分时默认末尾 token。

涉及论文标题：
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

---

## RANDOM EVICTION (随机淘汰)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

RANDOM EVICTION 是 NACL 引入的基于概率采样的 KV Cache 淘汰策略。将 PROXY-TOKENS EVICTION 的 F_score 经 Softmax 归一化为概率分布 P_prompt = Softmax(F_score)，从该分布中采样 C_r 个 token 保留。每个 attention head 和每层使用不同随机种子实现 head-wise/layer-wise 多样化。

直觉：确定性 top-K 淘汰一旦丢弃关键 token 无法恢复。head-wise 随机采样确保每个 token 在多个 head 中有独立保留机会。LLaMA-7B 32层×32头 budget=20% 时保留概率 > 99.92%。

从算法pipeline角度拆解术语：

```
输入: F_score ∈ R^{p}, C_r, seed_h

Step 1: P_prompt = Softmax(F_score)                   # attention-guided prob distribution
Step 2: for head h: S_random^h ~ Multinomial(P_prompt, C_r, seed=seed_h)
Step 3: S_keep^h = S_proxy ∪ S_random^h               # total C = C_p + C_r
```

消融：移除 RANDOM EVICTION → short-text -1.2%, long-text -9.2%。Uniform 采样替代 → long-text -1.1%。随机 budget 10%→70% 性能提升 2.25%→8.17%，>90% 时下降（需 attention 引导）。

术语一般如何实现？如何使用？

PyTorch `torch.multinomial`，每 head 用 `head_idx * layer_idx * large_prime` 作为 seed。budget 分配：total 20% = 6% proxy + 12% random + 2% protect proxy（Table 4）。

涉及论文标题：
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

---

## Encoding Phase One-Eviction (编码阶段一次性淘汰)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Encoding Phase One-Eviction 是 NACL 的 KV Cache 淘汰范式创新。传统方法（H2O、MSRNN）在 generation 阶段每步贪心淘汰 O(p+T)。NACL 将淘汰移至 encoding 阶段一次性完成：利用完整 attention matrix A ∈ R^{p×p} 做全局最优淘汰，压缩 KV cache 用于全部 generation。Generation 阶段仅每 m 步轻量维护。Long-context 下 T ≪ p，复杂度从 O(p+T) 降至 O(1)。

从算法pipeline角度拆解术语：

传统 step-by-step（H2O）:
```
for each token t: K_cache.append(K_t,V_t); scores += attn_scores; K_cache = topK(scores, C)
复杂度: O(p+T) per eviction
```

NACL one-eviction:
```
Encoding: S = F_score(A_full, C); K_cache = K_prompt[S]
Generation: for each t, if t%m==0: light eviction    # 轻量维护
复杂度: O(1) (T ≪ p)
```

消融：移除 global eviction → -1.3% short-text, -1.5% long-text。

术语一般如何实现？如何使用？

在 prefill 完成后、generation 前插入一次性淘汰 hook。需 access encoding 阶段的 attention matrix（通过 FlashAttention-2 LSE 重算或仅 proxy tokens 重算）。NACL Algorithm 1 描述完整两阶段流程。

涉及论文标题：
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

---

## Attention Bias Problem in KV Cache Eviction (KV缓存淘汰中的注意力偏差)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention Bias Problem 是 NACL 揭示的 KV Cache 淘汰策略中的系统性偏差。表现为 attention scores 高度集中在 initial tokens 和 recent tokens，对中间 token attention 显著偏低（Fig. 2）。随序列增长，attention 分布趋于扁平化（Fig. 2d），基于 attention 的评分更难区分关键 token。导致 H2O（全量累加）和 MSRNN（当前 token）均在 long-context 中误淘汰中间的 task-critical token（如 passkey）。

从算法pipeline角度拆解术语：

NACL 的两种对抗机制：(1) proxy tokens 的 attention 天然更均衡——proxy tokens（用户问题）在语义上与 prefix 关键信息相关，非仅位置相关；(2) RANDOM EVICTION 的 head-wise 采样为中间 token 提供额外保留机会。

术语一般如何实现？如何使用？

检测：可视化不同 query position 的 attention score 分布 heatmap。对抗方案：proxy tokens 替代全量累加（NACL）、随机采样补充（NACL）、衰减因子降低位置优势（A2SF forgetting factor）。

涉及论文标题：
- NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time
- A2SF: Accumulative Attention Scoring with Forgetting Factor for Token Pruning in Transformer Decoder

## Progressive KV Cache Quantization（渐进式KV缓存量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Progressive KV Cache Quantization 是 PM-KVQ 为长 CoT LLM 提出的一种逐步降低 KV Cache 位宽的量化策略。与传统的每步解码直接量化到目标位宽不同，渐进式量化在推理初期以高精度（FP16/INT16）存储 KV Cache，当内存预算被完全占用后，通过位宽收缩（bit-width shrinking）逐步将已存储的 KV Cache 降级到更低位宽（16→8→4→2 bit），为新 token 腾出内存空间。核心思想是"以时间换精度"：前期内存未满时保持零量化误差，后期再有损压缩早期 token，充分利用目标硬件内存预算，从而在相同总内存约束下降低累积量化误差。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**渐进式量化在长 CoT 推理中的执行流程（以 Fbit=2-bit, max_len=32K 为例）：**

```
// === Prefill 阶段 ===
K_cache = []  // 空缓存
V_cache = []
memory_used = 0
memory_budget = max_len * 2 * hidden_dim * 2_bytes  // 按Fbit=2计算内存预算
current_bitwidth = 16  // 初始以INT16精度存储

// === Decoding 阶段（逐步生成 token） ===
for t in 1..max_len:
    // Step 1: 计算当前 token 的 KV
    K_new, V_new = self_attention_layer.compute_kv(hidden[t])

    // Step 2: 检查内存——是否需要收缩？
    required_mem = memory_used + 2 * hidden_dim * current_bitwidth/8
    while required_mem > memory_budget:
        // 位宽收缩：16→8, 8→4, 4→2
        old_bitwidth = current_bitwidth
        current_bitwidth = current_bitwidth / 2  // 依次16→8, 8→4, 4→2

        // Equivalent Right Shift: X_b = ((2^{2b}-2^b+1)(X_{2b}+2^{b-1})) >> 3b
        // 例如 8bit→4bit: X_4 = ((2^8-2^4+1)(X_8+2^3)) >> 12
        for each stored_token in K_cache, V_cache:
            K_cache[stored_token] = eq_right_shift(K_cache[stored_token], current_bitwidth)
            V_cache[stored_token] = eq_right_shift(V_cache[stored_token], current_bitwidth)

        memory_used = len(K_cache) * 2 * hidden_dim * current_bitwidth/8
        required_mem = memory_used + 2 * hidden_dim * current_bitwidth/8

    // Step 3: 以当前位宽存储新 token
    K_cache.append(quantize(K_new, bits=current_bitwidth))
    V_cache.append(quantize(V_new, bits=current_bitwidth))
    memory_used = len(K_cache) * 2 * hidden_dim * current_bitwidth/8

    // Step 4: 带解量化的 attention 计算
    K_deq = dequantize(K_cache, bits=current_bitwidth)  // 含保留的首token和recent 128 tokens为INT16
    V_deq = dequantize(V_cache, bits=current_bitwidth)
    output[t] = attention(Q[t], K_deq, V_deq)
```

**位宽收缩阶段图（Gantt-style 描述）：**

以 32K 最大输出长度为目标，实际 token 生成为时间轴：
- Phase 1 (token 1 ~ ~8K): current_bitwidth=16, 零量化误差
- Phase 2 触发 (~8K): 内存预算耗尽，收缩到 8-bit → Equivalent Right Shift 压缩早期 token
- Phase 3 (~16K): 再次耗尽，收缩到 4-bit
- Phase 4 (~24K): 最终收缩到 Fbit=2-bit

与传统方案（每步都 2-bit）对比：前期~8K token 以 16 倍精度存储，累积误差显著降低。

术语一般如何实现？如何使用？

实现关键：(1) 位宽选择遵循 2 的幂（16/8/4/2），确保整数移位操作高效；(2) Equivalent Right Shift 等价于反量化→再量化，但仅通过整数加法和移位实现，避免浮点转换开销；(3) 保留首 token 为 INT16（attention sink 效应），最近 128 tokens 用滑动窗口保留 INT16（继承 KIVI/SKVQ 设计）；(4) 量化粒度为非对称分组量化 group_size=128。适用场景为长 CoT 推理（max output >8K tokens），尤其在 2-bit 极低精度下有显著收益。局限：(1) 需要已知目标内存预算和最大输出长度来规划位宽收缩节点；(2) 未覆盖 MLA 注意力机制。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---

## Block-wise Memory Allocation for KV Cache Quantization（KV缓存块级内存分配）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Block-wise Memory Allocation 是 PM-KVQ 提出来解决不同 Transformer block 的 KV Cache 对量化的敏感度差异问题。核心思想是：不为所有 block 分配统一位宽，而是通过一阶泰勒近似评估每个 block 的 KV Cache 敏感度，将内存分配建模为整数规划问题，用 CVXPY 求解器在几秒内给出每个 block 的最优位宽配置。敏感 block（深层 block + 第一层）获得更多内存（更高位宽），不敏感 block 使用更低位宽，在相同总内存预算下最大化模型精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// === 预处理阶段：Block-wise Sensitivity Profiling ===
// 校准数据：RedPajama arXiv subset, 512 samples × 2048 tokens

for each calibration_sample:
    for each transformer_block i in [0..N-1]:
        // 前向传播获取 KV Cache 和梯度
        K_i, V_i = block.compute_kv(hidden_states)

        for each candidate_bitwidth b in B:
            // 量化 KV Cache
            K_i_q = asymmetric_group_quantize(K_i, bits=b, group_size=128)
            V_i_q = asymmetric_group_quantize(V_i, bits=b, group_size=128)

            // 一阶泰勒敏感度估计
            G_Ki = grad(loss, K_i)  // 损失对 Key Cache 的梯度
            G_Vi = grad(loss, V_i)

            s_{i,b} = ||G_Ki ⊙ (K_i - K_i_q)||_1 + ||G_Vi ⊙ (V_i - V_i_q)||_1

// === 整数规划求解 ===
variables: x_{i,b} ∈ {0,1}  // 每个 block 选一个位宽

Objective:
min Σ_i Σ_b x_{i,b} · s_{i,b}

Constraints:
Σ_b x_{i,b} = 1, ∀i                       // 每个 block 恰好一个位宽
Σ_i Σ_b x_{i,b} · Mem(Q_b(K_i)+Q_b(V_i)) ≤ M  // 总内存不超过预算

solved_by: CVXPY (几秒内求解, Diamond & Boyd 2016)

// === 推理时使用 ===
// 求解结果为每个block i 分配 Fbit_i
for each decoded_token:
    for each block i:
        target_bitwidth_i = solved_x_{i,b}

        // Progressive Quantization with per-block Fbit
        if block i 使用 progressive quantization:
            从16bit开始逐步降到 Fbit_i
        else:
            直接量化到 Fbit_i
```

从 Paper Figure 3/4 的敏感度分析可知：
- 深层 block 对量化更敏感 → 分配更高位宽（如 4-bit）
- Qwen 模型第一层 block 异常敏感 → 获得最高位宽
- LLaMA 模型各层敏感度相对平滑 → 位宽分配更均匀
- B = {2,4}（7B<模型）或 {4,8}（7B 模型），即每个 block 获得 2-bit 或 4-bit（或 4-bit/8-bit）

术语一般如何实现？如何使用？

实现关键：(1) 一阶泰勒近似需要在校准数据上做一次前向传播记录梯度，对 7B-70B 模型可在数分钟内完成；(2) CVXPY 求解整数规划仅需几秒；(3) 位宽可选集合 B 一般设 2 个候选值（{2,4} 或 {4,8}），限制解空间并保证求解速度；(4) 该策略与渐进式量化正交——每个 block 的 Fbit 由整数规划确定，但推理时仍可渐进降低到 Fbit。

适用场景：(1) 当可用内存不足以将所有 block 统一升级到更高位宽时（如 batch size 减小后单样本内存增加，但不足以统一 4-bit）；(2) 与均匀位宽配合——在大 batch 时使用均匀位宽，小 batch 时切换为 block-wise 分配以利用额外内存。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---

## Calibration with Positional Interpolation for KV Cache Quantization（带位置插值的KV缓存标定）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Calibration with Positional Interpolation 是 PM-KVQ 提出的 KV Cache 量化标定策略，解决长 CoT 场景下短标定数据无法捕获长上下文数据分布的问题。核心思路：在标定阶段对 RoPE 的位置索引 m 施加缩放因子 s（即 m → s×m），将短序列（如 2048 tokens）的 RoPE 有效扩展到长上下文（如 8192 tokens），使标定过程中计算出的通道重参数化因子 λ_i 能够覆盖 RoPE 低频通道的完整周期分布。这在不增加标定计算和内存开销的前提（避免了 O(N²) 注意力复杂度）下，使短标定数据能够近似长上下文数据分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**问题背景**：RoPE 低频通道的周期可达 ~54K tokens（DeepSeek-R1-Distill-Qwen-7B, θ_{d/2} = θ^{-1} ≈ 1/10000^{2/d·d/2} = 1/10000）, 标定数据仅 512~2048 tokens → 低频通道仅覆盖正弦波的很小一段 → λ_i 标定不准确 → 通道重参数化失效（outlier 未正确迁移到 Query）。

**标定流程**：

```
// === 传统标定（短上下文问题） ===
calibration_len = 2048
data = load_calibration_data(calibration_len)  // RedPajama arXiv subset

// RoPE: θ_i = θ^{-2i/d}
for each channel i in [0..d-1]:
    freq_i = theta ** (-2 * i / d)  // i 接近 d/2 时频率极低（周期 > 54K tokens）
    pos[m] = m * freq_i  // 仅 2048 个位置 → 低频通道分布不完整
    K_rope[m,i] = K[m,i] * cos(pos[m]) - K[m,i+d/2] * sin(pos[m])

// 通道重参数化因子 λ_i = (max_m K_rope[m,i])^α
// 在 2048 tokens 上: max_m 仅覆盖正弦波约 1/26 周期 → λ_i 不准确

// === PM-KVQ标定（带位置插值） ===
s = 4  // 位置缩放因子
calibration_len = 2048
effective_len = s * calibration_len = 8192  // 嵌入8192上下文信息

for each channel i in [0..d-1]:
    // 关键修改：位置索引乘以缩放因子
    pos_scaled[m] = s * m * freq_i  // m=0..2047 → s*m=0,4,8,...,8188
    K_rope[m,i] = K[m,i] * cos(pos_scaled[m]) - K[m,i+d/2] * sin(pos_scaled[m])

// 通道重参数化因子更准确：
// λ_i(s=4) = (max_{m∈[0,2047]} |K_rope[m,i] at scaled positions|)^α
// 近似覆盖了 8192 tokens 的有效位置范围
// α 通过网格搜索在 [0,1] 区间寻优 (grid_size=20), 最小化自注意力重建损失

// 标定完成后得到的 Λ = diag(λ_i)
// 推理时通道重参数化: P = (Q·Λ)·Q((K·Λ^{-1})^T)
```

**消融实验结果**（DeepSeek-LLaMA-8B on AIME-2024-I）：
- Calibration len=2048, s=1 (no PI): pass@1=46.67%, Voting=60.00%
- Calibration len=2048, s=4 (8192 effective): pass@1=48.33%, Voting=60.00% (+1.66%)
- Calibration len=2048, s=16 (32768 effective): pass@1=46.67%, Voting=53.33% (过插值退化)
- Calibration len=8192, s=1 (真长上下文): pass@1=48.33%, Voting=60.00%

s=4 达到与真长上下文标定相同的效果，s=16 则过插值导致性能退化。

术语一般如何实现？如何使用？

实现关键：(1) s 值需根据目标上下文长度选择——论文对 32K 目标使用 s=4（8192 effective）即可；(2) 位置插值仅在标定阶段修改 RoPE，推理时恢复正常 RoPE；(3) α 网格搜索需要最小化 self-attention 算子的重建损失（非端到端 loss）；(4) 该技术独立于量化方式，可与任何基于通道重参数化的 KV Cache 量化方案配合。

适用场景：所有使用 RoPE 的长上下文 LLM 进行 KV Cache 量化标定，尤其在标定数据长度受限于 O(N²) 注意力复杂度时。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---

## Equivalent Right Shift for KV Cache Bit-width Shrinking（KV缓存的等价位宽收缩右移）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Equivalent Right Shift 是 PM-KVQ 为渐进式量化中的位宽收缩步骤设计的移位策略。当 KV Cache 需要从 2b-bit 降级到 b-bit 时（如 8→4 bit），该策略通过整数加法和移位操作实现与"先反量化为浮点→再量化到更低精度"等价的效果，避免浮点计算开销。核心公式：X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b。

从算法pipeline角度拆解术语：

**三种位宽收缩策略对比（以 4-bit→2-bit 为例）**：

```
假设 4-bit 量化值 {0,1,...,15} → 压缩到 2-bit {0,1,2,3}

原始 4-bit 量化参数：
  S_4, Z_4  // 缩放因子和零点
  X_4 = quantize(X_fp16, bits=4, S=S_4, Z=Z_4)

// === (a) Direct Right Shift ===
// 直接右移 2 bits，仅保留高位
X_2 = X_4 >> 2  // {0..3}→0, {4..7}→1, {8..11}→2, {12..15}→3
Z_2 = Z_4
S_2 = (2^2 + 1) * S_4 = 5 * S_4
// 问题：信息丢失严重，pass@1 从 44.17% 降至 12.08%

// === (b) Modified Right Shift ===
// 修改零点和缩放因子以保持均值映射
X_2 = X_4 >> 2
Z_2 = Z_4 - (2^{2b} / 2) * S_4  // 调整零点
S_2 = 2^2 * S_4 = 4 * S_4
// 问题：仍丢失信息，pass@1 = 28.75%

// === (c) Equivalent Right Shift (PM-KVQ) ===
// 等价于 dequantize→requantize，但全整数操作
// X_b = ((2^{2b} - 2^b + 1)(X_{2b} + 2^{b-1})) >> 3b
b = 2, 2b = 4:
X_2 = ((2^4 - 2^2 + 1)(X_4 + 2^{1})) >> 12
    = ((16 - 4 + 1)(X_4 + 2)) >> 12
    = (13 * (X_4 + 2)) >> 12

// 对应等价浮点操作：
// 1. 反量化到 FP16:  X_fp16 = X_4 * S_4 + Z_4
// 2. 重新量化到 2-bit: X_2_new = clamp(round((X_fp16 - Z_2) / S_2), 0, 3)
// 3. Equivalent Right Shift 在整数域产生相同的量化值
Z_2 = Z_4  // 零点不变
S_2 = (2^b + 1) * S_4  // 例如 b=2: S_2 = 5*S_4
// 效果：pass@1 = 38.33%（vs Direct的12.08%, Modified的28.75%），Voting lossless
```

**公式推导（2b→b bit, b∈{2,4,8}）**：
- 16-bit→8-bit: b=8, 2b=16 → X_8 = ((2^{16}-2^8+1)(X_{16}+2^7)) >> 24
- 8-bit→4-bit: b=4, 2b=8 → X_4 = ((2^8-2^4+1)(X_8+2^3)) >> 12
- 4-bit→2-bit: b=2, 2b=4 → X_2 = ((2^4-2^2+1)(X_4+2^1)) >> 6

术语一般如何实现？如何使用？

实现关键：(1) 仅需整数乘法和移位，在 GPU/CPU 上均可高效执行（single cycle per element）；(2) 零点不变 (Z_b = Z_{2b}) 简化实现；(3) 缩放因子放大 (S_b = (2^b+1)S_{2b}) 补偿位宽降低后的动态范围损失；(4) 渐进式量化中仅在位宽收缩节点执行，不产生每步开销。

局限性：该策略设计用于 2 的幂位宽。对于非 2 的幂位宽（如 3-bit, 6-bit），需要不同的收缩策略。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

---

## Long Chain-of-Thought (Long-CoT) LLM Inference（长链式思考LLM推理）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Long Chain-of-Thought (Long-CoT) LLM 指经过训练后能够在推理时生成超长推理链（可达 32K-128K tokens）的大语言模型，如 OpenAI-o1、DeepSeek-R1 和 QwQ-32B。其核心特征是模型通过生成多步推理过程（多次反思、多角度验证、分步骤论证）来解决复杂数学、科学推理和多跳问答任务。与普通 LLM 的简短输出不同，Long-CoT 模型的解码阶段极其漫长，导致 KV Cache 内存开销巨大（可达模型权重的数倍），使得 KV Cache 压缩（特别是量化）成为 Long-CoT 推理的关键瓶颈。

从算法pipeline角度拆解术语：

**Long-CoT 推理的内存特征**（以 DeepSeek-LLaMA-8B 为例，batch_size=16）：

```
// 标准推理 (短 CoT, ~1K tokens)
Weights: 16 GB
KV Cache: batch_size × seq_len × 2 × num_layers × num_kv_heads × head_dim × 2_bytes
        = 16 × 1024 × 2 × 32 × 8 × 128 × 2 ≈ 2 GB  // 可控

// Long-CoT 推理 (32K tokens)
Weights: 16 GB
KV Cache: 16 × 32768 × 2 × 32 × 8 × 128 × 2 ≈ 64 GB  // 超模型权重4×
Total Memory: 80 GB  // 超出单卡 A100-80G
```

**Long-CoT 给 KV Cache 量化带来的两个独特挑战**：

1. **大累积量化误差**：每步解码时对 KV Cache 量化引入误差，在 32K 步后累积效应显著。KIVI (2-bit) 在 DeepSeek-Qwen-7B AIME-2024 上 pass@1=32.08%（vs FP16: 44.17%），损失 ~12%。
2. **短标定 vs 长上下文分布 mismatch**：RoPE 低频通道（周期可达 54K+ tokens）在 512-token 标定中分布不完整 → 通道重参数化因子 λ_i 不准确 → 量化误差放大。

**PM-KVQ 针对 Long-CoT 的设计决策**：
- 渐进量化：前期低误差（16-bit）→后期有损压缩（利用内存预算未被充分利用的"空隙"）
- 位置插值标定：用 s=4 将 2048-token 标定嵌入 8192-token 有效长度 → 覆盖更多 RoPE 低频通道周期
- Block-wise 内存分配：为对累积误差更敏感的深层 block 多分配内存

术语一般如何实现？如何使用？

Long-CoT LLM 推理部署的关键考量：(1) 使用 GQA/MQA 降低 KV Cache 原始尺寸（DeepSeek-V2 使用 MLA 进一步压缩）；(2) KV Cache 量化是长 CoT 场景下内存瓶颈的核心缓解手段——4-bit 可压缩 4×, 2-bit 可压缩 8×；(3) 评测需使用数学推理 benchmark（AIME, CMIMC）和代码生成 benchmark（LiveCodeBench），而非仅使用 perplexity——长 CoT 场景下 PPL 与端到端推理表现可能不一致；(4) 最大输出长度需设置到 32K-128K tokens 才能充分发挥 Long-CoT 的推理能力。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs

## Static Sparse Attention (静态稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

静态稀疏注意力（Static Sparse Attention）是一种在推理全过程中使用预定义的、固定的注意力掩码模式的稀疏注意力机制。与动态稀疏注意力不同，静态模式不根据输入内容在线调整注意力掩码，而是在推理前就已确定每个 token 可关注哪些位置的 token。PowerAttention 论文将静态稀疏注意力的设计问题形式化为：在 DAG（有向无环图）中找到最优边集，使得在固定出度约束（sparsity constraint）下，多步可达节点数最大化。

常见的静态稀疏注意力模式包括：(1) Sliding Window——每个 token 仅关注前 W 个 token；(2) Stride Slash——在 sliding window 基础上按等间距放置 slash token；(3) Dilated Attention——使用膨胀滑动窗口（如每隔一个位置跳过）；(4) LongNet——多 mask 叠加，以几何增长的 segment length 和 dilation ratio 构建；(5) PowerAttention——每个 token 关注距离为 2 的幂次的位置。静态模式的共同优势：训练阶段可进行效率优化（mask 固定可预编译），解码阶段对新 token 处理更高效（无需重新计算 mask），且实现简洁。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**静态稀疏注意力的 DAG 建模（PowerAttention 论文 Section 3.1）**：

静态稀疏注意力掩码可以自然地解释为 DAG 的邻接矩阵。对 d 层 LLM：
- 节点：每个 token 在特定位置
- 边：token i 可以 attend token j（j < i，因果性）
- 单层感受野（out-degree）：token i 直接关注的所有 token 集合 A_i
- 多层感受野（d-step reachability）：经过 d 层信息传播后，token i 能间接访问的所有 token

```
# Static Sparse Attention 的一般形式（伪代码）
# q_idx [M, 1], kv_idx [1, N]
block_size = 256  # CUDA block size

# 1. Sink tokens（序列开头的注意力汇标记）
mask_sink = kv_idx < block_size

# 2. Local window（局部上下文窗口）
blk_qk = q_idx // block_size - kv_idx // block_size
mask_window = blk_qk < window_size

# 3. Pattern-specific mask（各静态模式的核心差异）
# Sliding Window: 无额外 mask
# Stride Slash: mask_slash = blk_qk % stride_size == 0
# Dilated: mask_dilated = (blk_qk & 1 == 0) & (blk_qk < window_size)
# PowerAttention: mask_power = (blk_qk & (blk_qk - 1)) == 0

# 4. 因果性 + 组合
causal = q_idx >= kv_idx
mask = causal & (mask_window | mask_pattern | mask_sink)
```

**DAG 可达性分析**（各静态模式的路径复杂度，到达距离 N 的 token）：

$$
\begin{aligned}
\text{Sliding Window:} &\quad O(N) \text{ layers} \quad \text{(线性扩展)} \\
\text{Stride Slash:} &\quad O(\sqrt{N}) \text{ layers} \quad \text{(平方根扩展，有覆盖间隙)} \\
\text{Dilated:} &\quad O(N) \text{ layers, } \sim 50\% \text{ 覆盖率} \quad \text{(奇数距离不可达)} \\
\text{LongNet:} &\quad O(\log N) \text{ layers} \quad \text{(有覆盖盲区)} \\
\text{PowerAttention:} &\quad O(\log N) \text{ layers, } 100\% \text{ 覆盖率} \quad \text{(指数扩展)}
\end{aligned}
$$

术语一般如何实现？如何使用？

静态稀疏注意力通常通过定义固定的 attention mask 实现。在 PyTorch 中可直接使用 `torch.nn.functional.scaled_dot_product_attention` 的 `attn_mask` 参数，或使用 FlexAttention 的 `score_mod` 函数定义 mask。实现时通常采用 block-sparse 策略（block_size=64~256 tokens）以对齐 GPU memory access 模式。静态模式特别适用于：(1) 需要训练阶段效率优化的场景——mask 预编译为 block-sparse kernel；(2) 高稀疏度场景（>90% sparsity）——避免动态估计开销大于稀疏计算收益；(3) streaming/continuous batching 场景——新 token 无需重新估计 mask。

PowerAttention 论文的关键发现：尽管相同稀疏度下各静态模式的单层感受野大小相同（out-degree 相同），但多层信息传播后可达节点数差异巨大——设计良好的模式（如 PowerAttention）在 6 层后可覆盖全部 32K token，而 Sliding Window 仅覆盖约 2304×6≈14K token 的范围（且最后 token 无法访问序列开头的 token）。

涉及论文标题：
- PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

---

## Receptive Field in LLMs (LLM中的感受野)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

感受野（Receptive Field）在 LLM 语境下由 PowerAttention 论文首次系统定义：**模型在生成输出 token 时可利用的最大上下文 token 集合**。不同于 CV 中的空间感受野概念，LLM 中的感受野通过跨层信息传播建模——在层 l，token i 通过自注意力从单层感受野 A_i 中的 token 接收信息，经 FFN 处理后传播到下一层。在多层 LLM 中，token x 的感受野不仅包括直接关注的 token，还扩展到所有从 x 出发通过 DAG 可达的节点。形式化定义：给定 DAG G=(V,E)（V 为 token 节点，E 为注意力边集），token i 在 k 步内的感受野 R_k(i) = {j ∈ V | 从 i 到 j 的最短路径长度 ≤ k}。

感受野的两个关键属性：(1) 完整性（Completeness）——感受野是否能覆盖所有位置的 token；(2) 扩展效率（Expansion Efficiency）——感受野随层数的增长速度。PowerAttention 论文证明：即使不同稀疏模式在相同稀疏度下有相同的单层感受野大小（out-degree），设计良好的模式可以在多层传播后实现指数级增长的感受野。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**感受野的理论建模（PowerAttention 定理 B.1）**：

对于有向无环图 G，顶点标签 1..n，边集 E = {(i, j) | i-j = 2^k, k ∈ Z*}：
- 性质 1：任意顶点 i 的出度 < log n（每个 k 最多一条出边）
- 性质 2：任意顶点对 (i, j)，j < i，距离 ≤ log n

证明构造：距离 d = i-j 的二进制表示中为 1 的位为 k_1, k_2, ..., k_m（最多 log n 个），路径为：

$$i \to (i - 2^{k_1}) \to (i - 2^{k_1} - 2^{k_2}) \to \cdots \to j$$

路径长度 = popcount(d) ≤ log n。

**感受野的量化评估方法（Appendix A）**：

```
# 理论精度计算
B_k = 从最后 block 出发 k 步内可达的 block 集合
hat_alpha_k = |B_k| / 总 block 数  # 理论精度上界

# 实验精度计算（通过 passkey retrieval）
# 在 block 的范围内均匀采样 passkey 位置
alpha_k = B_k 内成功检索的样本数 / 总样本数  # 实验精度

# 关系：hat_alpha_k 是 alpha_k 的最小上界
```

**各模式感受野对比（Qwen2-7B, 32K context, 28 layers）**：

| 模式 | 6 层覆盖率 | 全覆盖所需层数 | 完整性 |
|------|-----------|---------------|--------|
| Sliding Window | ~14K / 32K | O(N) ≈ 14 layers | 完整 |
| Stride Slash | ~32K / 32K | O(√N) ≈ 6 layers | 完整 |
| Dilated | ~16K / 32K | N/A | ~50%（奇数位不可达） |
| LongNet | ~32K / 32K | O(log N) ≈ 5 layers | 不完整（段尾盲区） |
| PowerAttention | ~32K / 32K | O(log N) ≤ 5 layers | 完整 |

术语一般如何实现？如何使用？

感受野概念用于指导稀疏注意力模式的设计和评估：(1) 设计阶段——以最大化多层可达性为目标，在固定 out-degree（sparsity）约束下构造边集；(2) 评估阶段——通过 passkey retrieval 实验验证理论感受野与实际信息检索能力的一致性（PowerAttention Figure 1b 展示了理论和实验感受野的高度吻合）；(3) 调试阶段——通过 probing 分析（每层每位置训练 logistic classifier）可视化信息流，定位感受野覆盖盲区。PowerAttention 论文的 probing 实验揭示：即使 Full Attention 理论上单步可达所有 token，实际注意力头仍展示空间局部性——不仅检索 passkey 原始位置，还聚合相邻位置积累的信息。

涉及论文标题：
- PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

---

## PowerAttention (Power-of-2 Sparse Attention / 指数感受野稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PowerAttention 是一种新型静态稀疏注意力模式，核心思想是让每个 token 仅关注 block 距离为 2 的幂次的位置（power-of-2 distances），配合局部滑动窗口和 attention sink tokens。其 mask 定义的核心操作为 `(blk_qk & (blk_qk - 1)) == 0`，即仅保留 block 索引差值为 2 的幂次的注意力连接（差值为 1, 2, 4, 8, 16, 32, ...）。

理论保证（定理 B.1）：在 d 层 LLM 中，每个 token 可访问距离 ≤ 2^d 的所有 token，同时每个 token 的出度 ≤ log n。这同时实现了：(1) 指数级感受野增长（最优扩展效率）；(2) 完整 token 覆盖率（无盲区）；(3) 亚线性出度（高稀疏度）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**PowerAttention Mask 构造（Algorithm 1）**：

```python
# q_idx [M, 1]: query token 索引
# kv_idx [1, N]: key-value token 索引
# block_size = 256 (CUDA block size)
# window_size = 5 (sliding window block 数)

# 1. Sink token mask（序列开头的初始 token）
mask_sink = kv_idx < block_size  # [1, N]

# 2. Sliding window mask（局部上下文）
blk_qk = q_idx // block_size - kv_idx // block_size  # [M, N]
mask_window = blk_qk < window_size  # [M, N]

# 3. PowerAttention mask（核心创新）
# 位运算技巧: x & (x-1) == 0 当且仅当 x 是 2 的幂
mask_power = (blk_qk & (blk_qk - 1)) == 0  # [M, N]

# 4. 因果性
causal = q_idx >= kv_idx  # [M, N]

# 5. 组合所有 mask
mask = causal & (mask_window | mask_power | mask_sink)  # [M, N]
```

**PowerAttention 的指数感受野扩展原理（Theorem B.1 路径构造）**：

```
给定 query token i 和 key token j（j < i）:
  距离 d = i - j（在 binary 表示中最多有 log n 个 1）
  设 k₁, k₂, ..., k_m 为 d 二进制中 1 的位置（m = popcount(d)）
  路径: i → (i-2^{k₁}) → (i-2^{k₁}-2^{k₂}) → ... → j
  路径长度 = m ≤ log n

  例如 d = 13 = 0b1101 → k = {0, 2, 3}
  路径: i → (i-1) → (i-1-4) → (i-5-8) = j
```

**配置参数（PowerAttention 论文 4.1 节）**：
- window_size = 5 blocks (5 × 256 = 1280 tokens)
- sink_size = 1 block (256 tokens)
- power blocks ≈ 4 个典型的 power-of-2 位置（取决于序列长度）
- 总计每 token 最多关注 ~10 blocks = 2560 tokens
- 在 32K context (128 blocks) 下稀疏度 ≈ (128-10)/128 ≈ 92%
- 在 128K context (512 blocks) 下稀疏度 ≈ (512-10)/512 ≈ 98%

**时间复杂度**：O(N log^2 N)。每个 query 需要处理的 power-of-2 KV blocks 数为 O(log n)，window blocks 为常数，sink blocks 为常数，总 KV blocks = O(log n)。最终注意力计算量 = N × O(log n) = O(N log^2 N)，接近 sliding window 的 O(N)。

术语一般如何实现？如何使用？

PowerAttention 使用 PyTorch FlexAttention 实现 mask 定义，结合 Triton 进行序列并行训练（RingAttention）。在推理时，mask 预编译为 block-sparse kernel，利用 FlexAttention 自动将 mask 映射到 GPU tiling 策略。训练策略：先在 SlimPajama (1B tokens) 上继续预训练，再用 ChatQA 2 fine-tuning（含跨窗口 long-range dependencies），使模型学会利用指数感受野进行信息检索。

实际应用采用 Hybrid Architecture：每 7 层保留 2 层 Full Attention（保证 sink token 和复杂语义处理），其余 5 层使用 PowerAttention（最大化稀疏效率收益）。在 128K context 下，PowerAttention prefilling 比 Full Attention 快 3.0×，解码仅需 58% 的时间；kernel 开销比 Full Attention 快 21.6×。

涉及论文标题：
- PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

---

## Information Flow Across Transformer Layers (跨 Transformer 层信息流)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

跨 Transformer 层信息流（Inter-layer Information Flow）是指 token 的隐藏状态表示在 LLM 层间传播时逐步聚合来自其他 token 信息的过程。在层 l，token i 通过自注意力从单层感受野中的 token 接收信息，经 FFN 转换后，更新后的表示 o_i^(l) 传播到层 l+1。这一过程使 token x 的表示不仅编码自身信息，还编码从其他 token "中继"（relay）来的信息。

PowerAttention 论文通过 probing 实验验证了信息流在 LLM 中的三个关键性质：(1) 信息流**本质上存在**——即使未训练的 Full Attention 也展示空间局部性和渐进式信息扩散；(2) 稀疏注意力**放大**了信息流的层次化特征——Sliding Window 展示线性扩展，PowerAttention 展示 phase transition 式跳跃扩展；(3) 通过 Continue Pretraining + Fine-tuning 可**激活**稀疏模式的信息流机制——训练后 PowerAttention 的 probing 精度从 56% 提升至 100%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**信息流 Probing 方法（PowerAttention Appendix C）**：

```
输入: 长序列（N=16K），包含随机 6 类 passkey（apple/banana/cherry/grape/kiwi/lemon）
      passkey 固定在 10% 位置，其余为无关文本

对每层 l ∈ {1, ..., 28} 和每 block b ∈ {1, ..., 64}:
    1. 收集所有样本在层 l、block b 位置的 hidden state h_{l,b}
    2. 对 h_{l,b} 做 average pooling（等间距采样）
    3. 训练 logistic regression classifier: h_{l,b} → 6-class passkey
    4. 计算 classification accuracy = 正确分类比例 / 总样本数
    // 如果 accuracy ≈ 1/6（random），则该位置不包含 passkey 信息
    // 如果 accuracy > 1/6，则该位置的 hidden state 编码了 passkey 信息

总计：28 layers × 64 blocks = 1792 个独立 classifier 训练
```

**信息流的数学建模（PowerAttention Section 3.1）**：

```
// 层 l 的信息聚合
o_i^(l) = Σ_{j ∈ A_i^(l)} softmax(q_i^(l) · k_j^(l) / √dk) · v_j^(l)

// 其中 A_i^(l) 是 token i 在层 l 的单层感受野
// h_j^(l-1) 已编码了前 l-1 层传播的聚合信息
// o_i^(l) 通过 FFN 后变为 h_i^(l)，继续传播到层 l+1

// 多层感受野 = 所有从 token i 出发，经过 ≤ d 步 DAG 路径可达的 token
R_d(i) = {j | 存在从 i 到 j 的路径，路径长度 ≤ d}
```

**信息流 probing 的关键发现**：

PowerAttention Figure 5 的可视化揭示了信息流的层间演变模式：
- Full Attention（未训练）：信息从 passkey 位置逐步向周围扩散，后期层覆盖全部位置
- Sliding Window：信息以线性速率逐层向前推进（每层约扩展 window_size 个 block）
- PowerAttention（未训练）：信息在特定层出现跳跃式扩展（phase transition），但精度仅 ~56%
- PowerAttention（训练后）：信息流边界更清晰聚焦，最终 token 的 probing 精度达 100%，展示了训练对信息流机制的激活效果

术语一般如何实现？如何使用？

信息流概念在 PowerAttention 中有三个核心用途：(1) 设计指导——以最大化多层可达性为目标设计稀疏注意力边集（DAG 最优化问题）；(2) 诊断工具——通过 probing 分析检测特定稀疏模式的覆盖盲区或信息传播瓶颈；(3) 训练验证——通过对比训练前后的 probing 精度，验证继续预训练+fine-tuning 是否成功激活了稀疏模式的信息流机制。

在更广泛的 LLM 研究中，信息流分析与 mechanistic interpretability（机制可解释性）和行为分析密切相关。具体分析手段包括：(1) Probing classifiers（线性/非线性分类器检测隐藏状态中的特定信息）；(2) Activation patching（替换特定位置的激活值，观察输出变化）；(3) Attention pattern visualization（可视化不同层的注意力分布热力图）。PowerAttention 的 probing 方法与这些技术互补，专注于量化感受野而非解释具体的注意力模式。

涉及论文标题：
- PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

## Pyramidal Information Funneling (金字塔形信息漏斗汇聚)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Pyramidal Information Funneling 是 PyramidKV（Cai et al., 2024）通过系统性分析 LLM 中跨层注意力模式发现的一种信息聚合现象。该现象描述了 LLM 处理长上下文输入时，注意力机制的信息流呈现"金字塔形漏斗汇聚"模式：

- **底层（Lower Layers, e.g., layer 0-5）**：注意力近似均匀分布（broad-spectrum mode），模型从全局所有可用内容中聚合信息，不优先关注特定输入片段。注意力分数覆盖几乎全部 token。
- **中层（Middle Layers, e.g., layer 6-18）**：注意力逐步收窄到局部区域（localized attention），每个文档/段落内部 token 之间的注意力显著增强（可视化为沿对角线的红色三角形状），信息在各个上下文内部被精细化聚合。
- **顶层（Upper Layers, e.g., layer 24-30）**：出现 "Massive Attention" 现象——绝大多数注意力集中在极少量关键 token 上（concentrated attention bars），这些 token 承载了聚合后的核心信息，用于最终答案生成。

这种从"全局广播→局部聚拢→关键 token 集中"的递进模式在 LLaMA、Mistral、Mixtral 等多个模型家族中均被验证（Appendix D），表明其跨模型架构的普适性。该发现超越了过去孤立记录的 "Massive Activation"（Sun et al., 2024，仅关注个别层的大激活值）和 "Attention Sink"（Xiao et al., 2023，仅关注首 token 的注意力异常），提供了跨层信息流动态的全景视角。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

Pyramidal Information Funneling 的核心机制通过注意力分数的跨层演变来量化：

**注意力分布分析流程**：
```
Input: Long-context sequence of n tokens, LLM with L layers, H heads per layer
Output: Per-layer attention pattern (average over heads)

for l in 0..L-1:
    // 获取第 l 层所有 head 的注意力矩阵
    A_l = []  // [H, n, n]
    for h in 0..H-1:
        Q_lh = W_Q[l,h] @ x_l    // query projection
        K_lh = W_K[l,h] @ x_l    // key projection
        A_lh = softmax(Q_lh @ K_lh.T / sqrt(d_k))  // [n, n], causal masked
        A_l.append(A_lh)
    
    // 跨 head 平均得到层注意力模式
    A_avg_l = mean(A_l, dim=0)   // [n, n]
    
    // 分析注意力分散度
    // 底层 (l=0):   A_avg 近似均匀分布，entropy 高
    // 中层 (l=12):  A_avg 沿对角线集中，块状结构明显
    // 顶层 (l=30):  A_avg 在少数列上有极高值（massive attention columns）

// 观察结论：
// entropy(A_avg_0) ≈ log(n)         → 信息分散（广播模式）
// entropy(A_avg_12) ≈ log(n/4)      → 信息局部化（聚类模式）
// entropy(A_avg_30) << log(n)       → 信息集中在极少数 token（massive attention）
```

**金字塔形信息流的量化指标**：

对于第 l 层，定义注意力集中度：
```
// 每列（key token）的平均注意力
col_attn_l[j] = mean(A_avg_l[:, j])   // token j 收到的平均注意力

// 注意力集中度 = top-k columns 占有的注意力比例
concentration_l(k) = sum(top_k(col_attn_l, k)) / sum(col_attn_l)

// 金字塔性质：
// concentration_0(10) ≈ 10/n       (底层——均匀)
// concentration_15(10) ≈ 0.3       (中层——部分集中)
// concentration_30(10) ≈ 0.8       (顶层——高度集中，massive attention)
```

术语一般如何实现？如何使用？

Pyramidal Information Funneling 作为观察到的现象（非算法），其价值在于指导算法设计：

1. **KV Cache 压缩设计**（PyramidKV 的核心应用）：
   - 底层注意力分散 → 需要更多 KV cache budget 覆盖全局信息
   - 顶层注意力集中 → 仅需少量 KV cache budget 保留关键 token
   - 实现为算术序列递减的 budget 分配：k^l = k^0 - (k^0 - k^{m-1})/(m-1) × l

2. **Token Selection 策略**：
   - 由于顶层 massive attention 集中在特定位置（不限于首 token），token 选择应基于实际注意力分数（attention score-based selection），而非仅依赖位置启发式（如仅保留首尾 token）

3. **验证方法**：
   - 在多文档 QA 任务上可视化每层平均注意力矩阵（Figure 2）
   - 通过 LongBench 17 个数据集验证基于该现象的 KV cache 压缩策略有效性
   - 通过 Needle-in-a-Haystack 验证长上下文信息检索能力保持

4. **跨模型泛化**（Appendix D）：
   - LLaMA 系列（dense）：所有层显示清晰的金字塔形信息汇聚
   - Mistral-7B（dense）：同样显示该模式，注意力收窄略有提前
   - Mixtral-8x7B（MoE）：尽管有专家路由，仍显示一致的注意力收窄趋势

涉及论文标题：
- PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

## Q-Filters

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Q-Filters 是一种训练无关（training-free）的 KV Cache 压缩方法，由 Godey et al. (2025) 提出，核心思想是利用 Query 和 Key 向量的几何特性——特别是 Query 分布的各向异性（anisotropy）——来估计 KV pair 的重要性，而无需访问注意力权重矩阵。方法分为两个阶段：

**离线校准阶段**：(1) 从校准数据集（如 Pile 子集，~3000 样本）前向传播，收集各层各注意力头的 Query 表示 $Q^h$；(2) 对每个头的 Query 矩阵进行 SVD 分解 $Q^h = U \Sigma V^\top$；(3) 取第一右奇异向量 $v_1$ 作为该头的 Q-Filter，并进行符号规范化 $v_1^+ = \operatorname{sgn}(\mathbf{1}u_1^\top)v_1$ 以确保正期望投影。对于 GQA，对每组共享 KV head 的 Query head 的 Q-Filters 取平均。

**推理阶段**：对每个注意力头，计算所有已存储 Key 向量在 Q-Filter 上的投影 $\langle K_t^h, v_1^+ \rangle$ 作为重要性得分，保留得分最高的 KV pairs，丢弃得分最低的。

理论基础是论文的定理 3.3：$\mathbb{E}_{Q_i^h}(\langle Q_i^h, K_j^h \rangle) \approx \kappa^h \langle K_j^h, u^h \rangle$，其中 $u^h$ 是 Query 分布的主方向（即 Q-Filter 方向），$\kappa^h > 0$ 为常数。该定理表明 Key 在 Query 主方向上的投影可近似期望注意力 logits，因此可作为 KV pair 重要性的有效估计。与 K-Norm（仅用 L2 范数）相比，Q-Filters 额外捕捉了 Key 向量在 Query 主方向上的角度分量 $\cos(K_j^h, u^h)$，Spearman 相关性显著更高。

关键特性：(1) 训练无关——无需参数更新；(2) FlashAttention 兼容——不访问注意力权重矩阵，仅需一次标量积投影；(3) 上下文无关——Q-Filters 仅依赖模型固有几何特性，不同校准数据集的 Q-Filters 高度一致（余弦相似度 > 0.9）；(4) 校准成本极低——Llama-3.2-70B 上 < 3 分钟（2×A100-80GB GPU）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Q-Filters 完整 pipeline 伪代码**：

```
// ========== 离线校准阶段（仅执行一次） ==========
// 输入：预训练模型 M，校准数据集 D（如 Pile 子集, ~3000 样本）
// 输出：每层每头的 Q-Filter v_1^+

Q_bank = {}  // Q_bank[layer][head] = list of Query activations

for batch in D:  // 前向传播，收集 Query 表示
    for layer in M.layers:
        for head in layer.heads:
            Q_activations = get_query_activations(layer, head, batch)
            // Q_activations: [batch_size, seq_len, d_head]
            Q_bank[layer][head].append(Q_activations)

q_filters = {}
for layer in M.layers:
    for head in layer.heads:
        // Step 1: 随机采样并拼接
        Q_samples = random_sample(Q_bank[layer][head], 3000)
        Q_matrix = flatten(Q_samples)  // [N*samples, d_head]

        // Step 2: SVD 分解
        U, S, Vt = SVD(Q_matrix, full_matrices=False)
        v1 = Vt[0, :]  // 第一右奇异向量, shape: [d_head]

        // Step 3: 符号规范化（保证正期望投影）
        sign = sign(mean(U[:, 0]))  // 基于第一左奇异向量的均值符号
        q_filters[layer][head] = sign * v1  // v_1^+

// GQA 处理
if model_uses_GQA:
    for kv_head in range(num_kv_heads):
        q_start = kv_head * heads_per_group
        q_end = q_start + heads_per_group
        // 组内 Q-Filters 取平均
        q_filters[kv_head] = mean(q_filters[q_start:q_end])

// ========== 推理阶段 ==========
// 输入：KV Cache，最大容量 max_size，Q-Filters
// 输出：压缩后的 KV Cache

def q_filters_compress(kv_cache, max_size):
    for layer in M.layers:
        for head in layer.heads:
            K = kv_cache[layer][head].keys  // [seq_len, d_head]
            V = kv_cache[layer][head].values  // [seq_len, d_head]

            // 标量积投影：计算每个 Key 的重要性得分
            scores = K @ q_filters[layer][head]  // [seq_len]

            // Top-k 选择：保留得分最高的 KV pairs
            if seq_len > max_size:
                keep_indices = topk_indices(scores, max_size)
                kv_cache[layer][head] = (K[keep_indices], V[keep_indices])
```

**张量计算流程**：
给定 Key 矩阵 $K^h \in \mathbb{R}^{L \times d_H}$ 和 Q-Filter $v_1^+ \in \mathbb{R}^{d_H}$，重要性得分 $s = K^h \cdot v_1^+ \in \mathbb{R}^L$。保留 $s$ 最大的 $k$ 个 KV pairs。该操作仅涉及一次矩阵-向量乘法和一次 top-k 选择，计算复杂度 $O(L \times d_H)$，与 FlashAttention 完全兼容（无需物化注意力矩阵）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源实现：https://github.com/NathanGodey/qfilters ，基于 KVPress 库（https://github.com/kvpress）和 HuggingFace Transformers。使用方式：(1) 调用 KVPress 的 Q-Filters compressor 加载预计算的 Q-Filters；(2) 在 HuggingFace 推理 pipeline 中插入 compressor hook，每次 KV Cache 更新后自动执行 top-k 筛选。校准数据集推荐使用 Pile 或多域混合数据（Q-Filters 对数据域不敏感，跨域余弦相似度 > 0.9）。校准样本数建议 ~3000（边际收益在 1000 样本后递减）。压缩比支持 2× 到 64×，在 32× 压缩比下 NIAH 仍保持 99% 准确率。已知局限：对使用 QK-normalization（如 Olmo-2）或 attention bias（如 Qwen-2.5）的模型效果减弱，需适配分析。

涉及论文标题：
- Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression

---

## Joint Anisotropy of Query-Key Representations（Query-Key 表示的联合各向异性）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Joint Anisotropy of QK 是 Q-Filters 论文（Godey et al., 2025）提出的关于 Transformer 自注意力中 Query 和 Key 表示分布的关键几何发现。该发现整合并深化了两个先前的独立观察：

**先验一（Devoto et al., 2024）**：Key 向量的 L2 范数与平均注意力权重之间存在负相关——低范数 Key 对应高平均注意力，利用此现象可实现 KV Cache 压缩（K-Norm 方法）。

**先验二（Godey et al., 2024）**：Q 和 K 的分布具有各向异性（anisotropic）——它们并非均匀占据 $\mathbb{R}^{d_H}$，而是沿着某个共同方向"漂移"远离原点，且不同头的漂移方向可能相反（$\epsilon = \pm 1$）。

**Q-Filters 论文的核心发现**：上述两个现象可以通过"联合各向异性"统一解释。具体而言：
- **Observation 3.1（联合各向异性）**：存在 $u^h \in \mathbb{S}^{d_H-1}$ 和 $\epsilon = \pm 1$ 使得 $\mathbb{E}(\langle Q_i^h, u^h\rangle) > 0$ 且 $\mathbb{E}(\langle K_j^h, \epsilon u^h\rangle) > 0$。即 Q 和 K 分布在同一个方向 $u^h$ 上有非零均值投影（符号可能相反）。
- **Observation 3.2（单方向性）**：令 $u^h = \arg\max_{u} \mathbb{E}(\langle Q_i^h, u \rangle)$，则对于所有与 $u^h$ 正交的方向 $u_m$（$m \geq 2$），$\mathbb{E}(\langle Q_i^h, u_m \rangle) \approx 0$。即 Q 的各向异性集中在单一方向上。
- **Theorem 3.3（注意力近似）**：$\mathbb{E}_{Q_i^h}(\langle Q_i^h, K_j^h \rangle) \approx \kappa^h \langle K_j^h, u^h \rangle$，其中 $\kappa^h = \mathbb{E}(\langle Q_i^h, u^h \rangle) > 0$。

**推论**：在实践中，大多数因果 LM 中 $\epsilon = -1$（Q 和 K 在 $u^h$ 上投影符号相反），因此 $\mathbb{E}(\langle Q_i^h, K_j^h \rangle) \approx -\kappa^h |\mathbb{E}(\cos(K_j^h, u^h))| \cdot ||K_j^h||_2$。这解释了为何 K-Norm（仅用 L2 范数）有效——范数是乘积中的一项，但忽略了角度分量 $\cos(K_j^h, u^h)$。Q-Filters 直接使用 $\langle K_j^h, u^h \rangle$，同时捕捉范数和角度信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**几何验证流程**：

```
// ===== Observation 3.1 验证：联合各向异性 =====
// 输入：预训练模型 M，多个序列 X_1, X_2, ..., X_N
// 输出：主方向 u^h 和各向异性程度的验证

for each head h:
    // Step 1: 收集 Q 和 K 表示
    Q_all = []  // 所有 token 位置的 Q^h
    K_all = []  // 所有 token 位置的 K^h
    for X in [X_1, ..., X_N]:
        Q_all.append(M.forward_get_queries(X, head=h))
        K_all.append(M.forward_get_keys(X, head=h))

    // Step 2: SVD 找主方向
    Q_matrix = stack(Q_all)  // [total_tokens, d_head]
    U, S, Vt = SVD(Q_matrix)
    u_h = Vt[0, :]  // 第一右奇异向量 = 主各向异性方向
    v_2 = Vt[1, :]  // 第二右奇异向量 = 正交方向（用于对比）

    // Step 3: 验证 Observation 3.1
    proj_Q_on_u = Q_matrix @ u_h  // Q 在主方向上的投影
    proj_K_on_u = K_matrix @ u_h  // K 在主方向上的投影

    mean_Q_proj = mean(proj_Q_on_u)  // 应 > 0
    mean_K_proj = mean(proj_K_on_u)  // 可能 < 0（ε = -1）
    // 实践中 ε = -1 在大多数头中成立

    // Step 4: 验证 Observation 3.2（单方向性）
    proj_Q_on_v2 = Q_matrix @ v_2  // Q 在正交方向上的投影
    mean_proj_v2 = mean(proj_Q_on_v2)  // 应 ≈ 0
    // 确认只有 v_1 方向携带各向异性信息

    // Step 5: 验证 Spearman 相关性
    attn_scores = compute_attention_maps(M, X)  // 真实注意力
    S = average_attention_per_position(attn_scores)  // 公式 (2)

    q_filter_scores = K_matrix @ u_h  // Q-Filters 重要性估计
    knorm_scores = ||K_matrix||_2     // K-Norm 重要性估计

    corr_qfilter = spearman_correlation(S, q_filter_scores)
    corr_knorm = spearman_correlation(S, knorm_scores)
    // Q-Filters 相关性 > K-Norm 相关性（大多数头）
```

**几何直观解释**（Figure 3）：
将 $Q^h$ 和 $K^h$ 投影到 SVD 的前两个右奇异向量 $(v_1, v_2)$ 上：K 在 $v_1$ 上的投影颜色编码了该位置的平均注意力——投影值越极端（正向或负向），注意力越高。而在 $v_2$ 上的投影则显示近似零均值的对称分布，与注意力无关。这直观验证了"仅第一主方向编码了注意力选择信息"的结论。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该几何发现的实际用途是构建 Q-Filters：通过 SVD 计算 Query 表示的主方向 $v_1$ 作为 Q-Filter，在推理时用 $\langle K_t^h, v_1^+ \rangle$ 估计 KV pair 重要性。该分析适用于标准 MHA 和 GQA（GQA 需对组内 Q-Filters 取平均），但不适用于使用 QK-normalization 或 attention bias 的模型（因几何特性被修改）。实现不依赖特定框架——只需能提取模型中间激活并执行 SVD（如 NumPy/PyTorch 的 `torch.linalg.svd`）。

涉及论文标题：
- Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression

---

## Query-Aware KV Cache Sparsity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Query-Aware KV Cache Sparsity 是一种动态稀疏注意力策略，核心思想是：KV cache 中 token 的关键性取决于当前 query token，因此不能预先静态决定哪些 token 重要/不重要，而必须在每步 decode 时根据当前 query 动态选择关键 token。Quest（ICML 2024, MIT HAN Lab）首次系统性地提出并验证了这一概念：传统 KV cache 驱逐方法（H2O、TOVA、StreamingLLM）基于历史 attention 或固定窗口预判 token 重要性（query-agnostic），会丢弃对将来 query 可能关键的 token，导致 passkey retrieval 等长依赖任务准确率近乎 0%。Quest 不驱逐任何 token，而是在每步 decode 评估所有 KV cache page 对当前 query 的关键性，仅加载 Top-K 关键 page 参与 attention。

关键洞察（Quest Fig. 2）：对于 prompt "A is B. C is D. A is"，token "B" 在 query="is" 时 attention score 很高（因为是正确答案），但在之前的 query（"C", "is", "D"）中 attention score 很低。因此同一 token 的关键性随 query 变化而剧烈变化，query-agnostic 方法会错误地丢弃它。Quest Fig. 4 量化了这一效果：H2O（历史注意力累积）的 recall rate 远低于 100%，而 Quest 基于当前 query 的 recall rate 接近 full attention。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Quest 的 Query-Aware 两阶段稀疏注意力流程**：

```
// === 预处理（Prefill 后，每个 KV cache page） ===
for each page p (含 S=16 tokens):
    for each channel i in 1..d_head:
        M_i^p = max(M_i^p, k_i)  // per-page channel-wise max Key
        m_i^p = min(m_i^p, k_i)  // per-page channel-wise min Key
// 元数据大小: 2 × num_pages × d_head × 2 bytes (FP16)

// === Decode 阶段每步 ===
// Stage 1: Criticality Estimation
Input: Q ∈ R^{d_head}, all {M^p, m^p} for p=1..num_pages
for each page p:
    score_p = 0
    for each channel i in 1..d_head:
        U_i = max(Q_i * m_i^p, Q_i * M_i^p)  // 保证 U_i ≥ Q_i * K_i^(t) ∀t∈p
        score_p += U_i                        // page attention score 上界
top_k_indices = TopK({score_p}, k=K)         // K = token_budget / page_size

// Stage 2: Approximate Attention (仅加载 Top-K pages)
K_selected = load_K_pages(top_k_indices)   // K×S × d_head
V_selected = load_V_pages(top_k_indices)
S = Q @ K_selected^T / sqrt(d_head)        // 仅计算选中 tokens 的 attention
A = softmax(S)
O = A @ V_selected

// 内存加载量: 完整 KV cache 的 (1/PageSize + K/PageNum)
// 例: page_size=16, 64K context (4096 pages), K=256 → ~12.5% of full KV cache
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Quest 基于 FlashInfer kernel 库实现（开源：https://github.com/mit-han-lab/Quest）。核心实现组件：(1) 在 FlashInfer 中新增 per-page metadata buffer（channel-wise min/max Key values），插入新 token 时 O(d_head) 更新；(2) Criticality estimation CUDA kernel：element-wise max(Q_i*m_i^p, Q_i*M_i^p) + reduce-sum，计算 per-page upper-bound score；(3) Top-K filtering 使用 RAFT（RAPIDS）batched Top-K CUDA operator，延迟仅 5-10 µs；(4) Approximate attention 利用 FlashInfer 的 PageAttention 接口，传入 sparse page indices 执行仅选中 page 的 FlashAttention。前两层保持 full attention（因观察到稀疏度 <10%），其余层使用 Quest。支持 Llama-3.1、Mistral-v0.3 等模型家族。

涉及论文标题：
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

**Sparse Frontier 论文中的 Quest 实现与评估**：使用 page_size=16（消融实验确定），始终包含当前 token 所在 page。在 Sparse Frontier 的全量评测中，Quest 是 decoding 阶段整体最佳方法。Quest 在 0.95 sparsity (1/20 budget) 下仍可优于更小的 dense 模型。但 Quest 在合成数据（Ruler NIAH）上表现退化——随机符号序列导致 key representations 区分度下降，page-level 粗粒度放大这一效应。相反在自然语言 retrieval（Story Retrieval）上 Quest 优于 Ada-SnapKV。

---

## Page-Level KV Cache Criticality Estimation (基于 Min/Max Key 元数据的 Page 级关键性估计)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Page-Level KV Cache Criticality Estimation 是 Quest 的核心算法组件：利用 KV cache 每个 page 内 Key 向量的 per-channel 最小值 m_i 和最大值 M_i 作为紧凑元数据，结合当前 Query 向量 Q 计算每 page attention score 的数学上界，以此估计该 page 对当前 query 的关键性。关键数学保证：对 page 内任意 token t，$U_i = \max(Q_i \cdot m_i, Q_i \cdot M_i) \geq Q_i \cdot K_i^{(t)}$（因为 $m_i \leq K_i^{(t)} \leq M_i$），因此 $\sum_i U_i$ 是该 page 内最高可能的 pre-softmax attention score 的上界。选择上界最高的 K 个 page 等价于"不会遗漏任何可能得到高 attention score 的 page"。

这一设计的精妙之处：(a) 元数据大小仅 2/PageSize of KV cache（page_size=16 时 ~12.5%），criticality estimation 的内存加载远小于完整 KV cache；(b) 上界保证了选择的"安全性"——top-K pages by upper bound 一定包含真正高 attention 的 token；(c) 计算极简——仅需 per-channel max + reduce-sum，无矩阵乘法，因而 criticality estimation 开销极小。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Quest Criticality Estimation 的详细张量计算**：

```
输入：Q ∈ R^{d_head}, M ∈ R^{num_pages × d_head}, m ∈ R^{num_pages × d_head}
输出：top_k_indices ∈ Z^K

// Vectorized computation (CUDA kernel):
// Step 1: per-channel upper bound (element-wise, parallel over pages)
// U[p][i] = max(Q[i] * m[p][i], Q[i] * M[p][i])
// 等价于: sign-preserving max selection per channel

// Step 2: reduce-sum over channels (warp-level reduce)
// score[p] = sum_{i=1}^{d_head} U[p][i]
// This is the upper bound of max_{t∈p} (Q · K_t)

// Step 3: Top-K selection (RAFT batched Top-K)
// top_k_indices = argsort(-score)[:K]

// 数学正确性证明:
// For any token t in page p:
//   For each channel i: M_i^p ≥ K_i^{(t)} ≥ m_i^p
//   → max(Q_i · m_i^p, Q_i · M_i^p) ≥ Q_i · K_i^{(t)}
//   → sum_i max(Q_i · m_i^p, Q_i · M_i^p) ≥ sum_i Q_i · K_i^{(t)} = Q · K_t
//   → score_p ≥ max_{t∈p} (Q · K_t)
// Therefore score_p 是 page p 内最高 token attention score (pre-softmax) 的上界

// 复杂度: O(num_pages × d_head) ≈ O((seq_len/page_size) × d_head)
// vs full attention: O(seq_len × d_head)
// 节省因子: page_size × (因仅加载 metadata，非完整 K cache)
```

**论文 Fig. 3 验证**：Quest 的 query-aware sparsity（基于上界估计选择 page）与 oracle sparsity（基于真实 attention score 的 top-K）高度对齐，证明了上界估计的有效性。除前两层外（稀疏度 <10%），其余层的 Quest sparsity 与 oracle 几乎重合。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Quest 开源实现（https://github.com/mit-han-lab/Quest）：在 FlashInfer 中实现为 custom CUDA kernel。(a) Per-page metadata 存储在 dedicated GPU buffer，每 page 的 (M_i, m_i) 为 2 × d_head × FP16 元素，page_size=16 时 metadata overhead = 2 × d_head / 16 = d_head/8 per token；(b) Criticality estimation kernel 使用 grid-stride loop over pages，每个 thread block 处理多个 pages，利用 warp-level reduce 做 channel sum；(c) NVBench micro-benchmark 显示 criticality estimation latency 随 seq_len 增长而趋近 1/PageSize of FlashInfer full attention。Token budget (K × page_size) 是可调超参数：PG19 perplexity 中用 4096 (~1/8 of 32K)，LongBench 中 1K budget 即达 full cache 可比性能，Passkey retrieval 中 64-1024 budget 即 100% 准确率。

涉及论文标题：
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

---

## Token Budget in Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Budget（token 预算）是稀疏注意力方法中控制稀疏程度的核心超参数，表示参与 attention 计算的 token 数量上限。在 Quest 中，token budget = K × page_size，其中 K 是选中的关键 page 数量，page_size 是每 page 的 token 数（默认 16）。Token budget 直接决定了 memory load reduction 比例：加载量 = 完整 KV cache 的 (token_budget/seq_len + 1/page_size)。例如 32K context 下 token budget=2048 → 稀疏度 ~93.75%，memory load 减少 ~16×。

Token budget 是 accuracy-efficiency trade-off 的调控旋钮：较小 budget → 更高稀疏度/更快速度 but 可能遗漏关键 token → 精度下降；较大 budget → 更接近 full attention → 精度高 but 加速少。Quest 的实验显示：LongBench 六数据集上 1K budget 即达 full cache 可比性能，PG19 上 4096 budget (~1/8 of 32K) perplexity 与 full cache 几乎一致，Passkey retrieval 中 64-token budget (10K context) 即可 100% 准确。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Token Budget 在不同稀疏注意力方法中的含义:

// Quest (page-level selection, non-eviction):
K_pages = token_budget / page_size  // e.g. 2048/16 = 128 pages
top_k_indices = TopK(criticality_scores, k=K_pages)
selected_tokens = gather(KV_cache[top_k_indices])  // 最多 token_budget 个 tokens
// KV cache 完整保留，仅本次 attention 不加载所有 tokens

// H2O/TOVA (token-level eviction):
K_tokens = token_budget
keep_indices = TopK(importance_scores, k=K_tokens)
KV_cache = KV_cache[keep_indices]  // 永久驱逐其余 tokens

// 内存加载量对比 (per decode step, per layer per head):
// Quest:   2 × d_head × (seq_len/page_size + token_budget) bytes
// Eviction: 2 × d_head × token_budget bytes
// Full:    2 × d_head × seq_len bytes
// Quest 比 eviction 多 metadata load，但保留了所有 token 的信息
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Token budget 的选择需要根据任务类型和 context length 确定。Quest 论文的指导：(a) 短依赖任务（语言建模）：可使用较大 budget（如 1/8 of context）；(b) 长依赖任务（passkey retrieval）：仅需极小 budget（64-1024 tokens for 10K-100K context）；(c) 通用 long-context benchmark（LongBench）：1K budget 通常足够。实际部署时，token budget 作为 serving 配置参数，根据 latency SLO 和 accuracy 要求动态调整。与 KV cache 量化正交——Quest 兼容 weight quantization（4-bit），两者可叠加。

在 SeerAttention-R 中，token budget 从 token 级别转换为 block 级别：block_budget = token_budget / block_size（block_size=64 为默认）。与 Quest 不同，SeerAttention-R 的 token budget 还对应两种 sparsification 策略对比：(1) Top-K 方法（固定 token budget）：对 AttnGate 输出的块分数排序取 top-k，保证每步计算量可控；(2) Threshold 方法（自适应）：分数超过阈值的块被激活，不同 head/step 可能有不同的稀疏比。Threshold 方法在实现上更简单（无需排序），且在 high sparsity 区域精度略优。SeerAttention-R 实验中 token budget 范围：AIME 用 2k-8k，MATH-500/GPQA 用 1k-6k。

涉及论文标题：
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models
- SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

---

## Passkey Retrieval Task (长上下文关键信息检索评估)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Passkey Retrieval 是 Yarn（Peng et al., 2023）提出的评估 LLM 长距离依赖处理能力的 benchmark 任务。任务构造：在一大段无意义文本（"needle in a haystack"）中隐藏一个简单的 passkey（如数字串 "12345"），然后在文本末尾提问 "What is the passkey?"。模型需要在极长上下文中定位并提取该信息。关键变量：(a) passkey 在文本中的深度位置（如 0%, 25%, 50%, 75%, 100%）；(b) 总上下文长度（如 10K、100K tokens）。该任务在 Quest 论文中的特殊价值：对于 query-agnostic KV cache 驱逐方法（H2O/TOVA/StreamingLLM），passkey 在 question 之前出现，可能在 decode 阶段被提前驱逐，导致准确率 0-4%；而 Quest 不驱逐任何 token 且基于 query 动态选择，64-1024 token budget 即达 100% 准确率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// Passkey Retrieval 评估流程 (Quest 论文设置)
// 构造输入:
haystack = "The grass is green. The sky is blue. " × N   // 无意义填充文本
passkey = "The pass key is 12345."
question = "What is the pass key?"

// 测试条件:
depth_ratio ∈ {0%, 25%, 50%, 75%, 100%}  // passkey 插入位置
total_length ∈ {10K, 100K}               // 总 token 数

// Quest 论文的特殊设置（模拟多轮对话）:
// question 被逐 token feed 到 decode 阶段（而非 prefill）
// 因此 H2O/TOVA 可能在 decode 期间驱逐包含 passkey 的 token
// Quest 保留所有 token，靠 query-aware 选择在需要时加载

for each (depth, length) combination:
    prompt = haystack[:pos] + passkey + haystack[pos:]
    prefill(prompt)                       // FlashAttention + full cache
    for each question_token:              // 逐 token decode
        Q = embed(question_token) @ W_Q
        critical_pages = Quest.estimate(Q, page_metadata)
        O = SparseAttention(Q, KV_cache[critical_pages])
        // H2O/TOVA 在这里可能已驱逐 passkey
    accuracy = (generated_answer contains "12345")
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Passkey Retrieval 被广泛用于长上下文 LLM 评估。扩展形式包括 RULER benchmark 中的 Needle In A Haystack (NIAH) 变体（多针、多值、多查询）。Passkey retrieval 与 language modeling perplexity 互补——前者测量长距离精确检索能力，后者测量局部语言建模能力。Quest 论文的实验表明，对 query-agnostic 方法最具挑战性的正是 passkey retrieval（因 token 在 question 之前被驱逐），而对 query-aware 方法恰好是优势所在。

涉及论文标题：
- Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference
- Yarn: Efficient Context Window Extension of Large Language Models

## Redundancy-aware KV Cache Compression for Reasoning Models (R-KV)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

R-KV 是一种面向推理模型（如 DeepSeek-R1）、训练无关的 KV Cache 压缩方法。核心思路：现有 attention-based KV cache eviction 方法（如 SnapKV）仅依赖 attention score 判断 token 重要性，但推理模型的长 CoT 输出中存在大量重复的自反射和自我验证内容，这些冗余 token 因"self-attend 到自己"同样获得高 attention score，导致关键推理 token 被错误淘汰。R-KV 显式引入 redundancy estimation——通过 key vector 余弦相似度测量 token 间的语义冗余——并用 joint selection score Z = λ·I − (1−λ)·R 同时平衡重要性（I）和去冗余性（R）。

R-KV 三组件：(1) Importance Scoring：基于最后 α 个 observation tokens 的 attention weight，对 GQA 使用 max-pooling (而非 SnapKV 的 mean-pooling) 聚合 query head attention（§3.2）；(2) Redundancy Estimation：对 key vectors 做 L2 归一化后计算余弦相似度矩阵 S = K̄·K̄^T，对角线置零，保留最近 β 个高相似 token（不被标记冗余），剩余高相似 token 通过 softmax 归一化获得 redundancy score R_i^h（§3.3）；(3) Joint Selection：Z_i^h = λ·I_i^h − (1−λ)·R_i^h，λ=0.1（§3.4）。

R-KV 是 decoding-time 压缩：每 B_buffer=128 tokens 触发一次压缩，始终保留最后 α 个 observation tokens。在 AIME24 上，R-KV 以 10% KV cache budget 达到与 FullKV 持平（lossless compression），16% budget 时 even surpass FullKV by 5%（R1-Llama-8B）。进行固定 budget 分析：1024 budget @16K generation → 13.4× larger batch size, 9.2× throughput。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# R-KV decoding-time compression pipeline
# 每生成 B_buffer=128 tokens 后触发一次压缩

def R_KV_compress(K_full, V_full, L_full, B_budget, B_buffer, alpha=8, lambda_=0.1):
    L_budget = len(K_cache)  # 当前保留的KV cache长度
    
    # 检查是否触发压缩
    if L_full - L_budget < B_buffer:
        return K_full, V_full  # 不足一次压缩周期，不压缩
    
    # Step 1: 提取observation tokens + candidate tokens
    K_obs, V_obs = last_alpha(K_full, V_full)  # α=8，始终保留
    K_cand, V_cand = first_N(K_full, V_full, L_full - alpha)
    N_c = L_full - alpha  # 候选token数
    
    if N_c <= B_budget:
        return K_full, V_full  # 候选不足，不压缩
    
    # Step 2: Per-head Importance Scoring
    for h in range(H):  # H个attention heads
        Q_obs_h = Q_obs[:, h, :]  # [α, d_head]
        K_cand_h = K_cand[:, h, :]   # [N_c, d_head]
        
        # GQA: 同组query heads各自计算attention后max-pooling聚合
        A_h = softmax(Q_obs_h @ K_cand_h.T / sqrt(d))  # [α, N_c], Eq.(1)-(3)
        
        # 稳定化：滑动窗口max-pooling (窗口2W)
        A_tilde = sliding_window_maxpool(A_h, window=2W)
        
        # Per-token importance: 沿query维度取均值
        I_h[k] = mean(A_tilde[:, k]) for k in 0..N_c-1  # Eq.(4)
    
    # Step 3: Per-head Redundancy Estimation
    for h in range(H):
        # L2归一化key vectors
        K_norm_h = K_cand_h / (norm(K_cand_h, dim=-1) + 1e-8)  # [N_c, d]
        
        # 余弦相似度矩阵
        S_h = K_norm_h @ K_norm_h.T  # [N_c, N_c], Eq.(5)
        diag(S_h).fill_(0)  # 抑制自相似
        
        # 保留最近β个高相似token（largest indices）
        for i in range(N_c):
            similar_j = where(S_h[:, i] > T)  # 相似度阈值T
            recent_beta = similar_j.topk(k=beta, largest=True)
            S_h[recent_beta, i] = 0  # 不标记为冗余
        
        # 平均相似度 → softmax归一化 → redundancy score
        S_bar_h[i] = mean(S_h[:, i])  # Eq.(6)
        R_h = softmax(S_bar_h)  # [N_c], Eq.(6)
    
    # Step 4: Joint Selection + 跨head聚合
    for h in range(H):
        for k in range(N_c):
            Score_h[k] = lambda_ * I_h[k] - (1-lambda_) * R_h[k]  # Eq.(7)
    
    AggScore[k] = mean_h(Score_h[k])  # 跨head均值聚合
    
    # Step 5: Top-B_budget选择 + 拼接observation tokens
    top_idx = argmax(AggScore, k=B_budget)
    K_comp = cat([K_cand[top_idx], K_obs])
    V_comp = cat([V_cand[top_idx], V_obs])
    
    return K_comp, V_comp  # 压缩后长度 = B_budget + α
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

R-KV 开源：https://github.com/Zefan-Cai/R-KV。PyTorch 实现，基于 HuggingFace Transformers，在模型 forward pass 中插入 compression 逻辑。使用时设置超参：B_budget（KV cache budget）、B_buffer=128（压缩周期）、α=8（observation tokens）、λ=0.1（importance vs redundancy 权重）、T（similarity threshold，论文未明确给出值）、β（最近保留的高相似token数，论文未明确给出值）。R-KV 是 training-free 和 model-agnostic，可直接适用于任何使用 MHA/GQA 的 LLM。局限性：当前不支持 paged attention，且在没有 KV compression 专用接口的 serving 框架中需要 reallocate memory 引入开销（Appendix D）。

涉及论文标题：
- R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration

## Joint Selection Strategy for KV Cache Eviction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Joint Selection Strategy 是 R-KV 提出的 KV cache token 保留策略，通过线性组合 importance score（I_i^h）和 redundancy score（R_i^h）来决定每个 token 是否保留：Z_i^h = λ·I_i^h − (1−λ)·R_i^h。这是 R-KV 区别于纯 attention-based 方法（如 SnapKV，仅使用 I_i^h）的核心机制。λ∈[0,1] 控制两项目标的权衡：(a) 高 I_i^h → token 对后续解码重要，应保留；(b) 高 R_i^h → token 语义与大量其他 token 相似（冗余），应淘汰。两者通过相减在 joint score 中融合：当 token 重要性高且冗余性低时 Z_i^h 最高，最优先保留；当 token 重要性低且冗余性高时 Z_i^h 最低，最优先淘汰。

λ 的选择关键：R-KV 消融实验（§5.1）发现 I_i^h 分布极度稀疏（少数 outlier 主导），而 R_i^h（经 softmax 归一化）分布相对均匀。λ=0.1 时 redundancy 项的权重 (1−λ)=0.9 足以有效抑制冗余，同时 λ=0.1 的 importance 项保证 attention sink/初始 token 不被错误淘汰（λ=0 时初始四个 token 不保证保留，会严重损害生成能力，如 Fig. 5 所示）。λ≥0.5 后 selection 退化为近似纯 attention-based。最优 λ∈[0.01, 0.1]，论文所有实验使用 λ=0.1。

从算法pipeline角度拆解：

```
# Joint Selection 跨head聚合流程
# 输入: I ∈ R^{H × N_c} (per-head importance scores)
#       R ∈ R^{H × N_c} (per-head redundancy scores)
# 参数: λ=0.1

for h in range(H):  # 每个attention head
    for k in range(N_c):  # 每个候选token
        # 线性组合
        Z[h][k] = λ * I[h][k] - (1-λ) * R[h][k]

# 跨head聚合 → 均值
AggScore[k] = (1/H) * Σ_h Z[h][k]  for k in 0..N_c-1

# Top-B_budget选择
selected_indices = ArgSort(AggScore, descending=True)[:B_budget]
```

R-KV 的跨 head 聚合使用 mean（而非 max 或 sum），确保每个 head 的 joint score 对最终选择有均等贡献（因不同 head 可能关注不同类型的 token——某些 head 关注语法、某些关注语义、某些关注 attention sink）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Joint Selection 的核心实现是纯 PyTorch tensor 操作：per-head Z = λ*I - (1-λ)*R 用 element-wise arithmetic；跨 head 聚合用 tensor.mean(dim=head_dim)；Top-K 用 torch.topk。计算开销为每 compression step O(H·N_c) 的标量操作，相比 attention 计算 O(α·N_c·d) 和 similarity matrix O(N_c²·d) 可忽略不计。Joint Selection 的 λ 选择需针对不同模型/数据集做 sensitivity analysis——不同模型、不同数据集的 attention score 稀疏度和 redundancy 分布可能不同。R-KV 建议从 λ=0.1 出发做 grid search over {0.01, 0.05, 0.1, 0.5, 1.0}。

涉及论文标题：
- R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration

## Key Vector Cosine Similarity for Redundancy Estimation in KV Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

R-KV 提出的 redundancy estimation 机制：通过对 key vectors 做 L2 归一化后计算余弦相似度矩阵 S = K̄·K̄^T∈R^{n×n}（Eq.5），显式测量 token 间的语义冗余。核心洞察：冗余 token 的 key vectors 在向量空间中高度聚集（因为解码时 model 对重复内容产生相似的 key 表示），通过余弦相似度可以在向量空间层面（而非 token 表面）捕捉语义冗余，无需分析文本内容本身。

具体流程（Eq. 5-6）：(1) L2 归一化：K̄_i = K_i / (||K_i||₂ + ε)，去量纲化；(2) 相似度矩阵：S = K̄·K̄^T，S_{i,j}∈[-1,1] 表示 token i 和 j 的 key vector 方向夹角余弦；(3) 对角线置零：S_{i,i}=0，防止 token 被标记为"与自身冗余"；(4) 保留最近 β 个高相似 token：对每个 token i，找到 S_{:,i} > T（T 为相似度阈值）的高相似 token 集合 I_i，保留其中 β 个最新位置（largest indices）的 token 不被标记冗余——因为即使内容重复，最新出现的变体离当前解码位置最近，contextual relevance 最高；(5) 平均相似度：S̄_i = (1/n)·Σ_j S_{j,i}，衡量 token i 与多少其他 token 相似；(6) Softmax 归一化：R = softmax(S̄)，得到 per-token redundancy score ∈ [0,1]，总和为 1。

计算复杂度：O(n²·d) for similarity matrix computation（n 个 key vectors 两两内积），O(n²) for similarity matrix 处理。总 overhead 为 O(B_budget²)，在 B_budget=1536 时约 2.4M 元素相似度矩阵，相比 attention 计算 O(B_budget·B_buffer·d) 量级仍较小。

从算法pipeline角度拆解：

```python
def redundancy_estimation(K_cand, T, beta, eps=1e-8):
    """
    K_cand: [n, d_head] (n个候选token的key vectors)
    T: 相似度阈值 (论文未明确给出具体值)
    beta: 最近保留的高相似token数 (论文未明确给出具体值)
    """
    n, d = K_cand.shape
    
    # Step 1: L2归一化
    K_norm = K_cand / (K_cand.norm(dim=-1, keepdim=True) + eps)  # [n, d]
    
    # Step 2: 余弦相似度矩阵
    S = K_norm @ K_norm.T  # [n, n], S_ij = cos(k_i, k_j)
    S.fill_diagonal_(0)    # 抑制自相似
    
    # Step 3: 标记高相似pair并保留最近β个
    B = (S > T).float()  # [n, n], 二值化
    for i in range(n):
        similar_j = B[:, i].nonzero().squeeze(-1)  # 与i高相似的token索引
        if len(similar_j) <= beta:
            continue  # 不够β个，全保留
        
        # 保留最近β个（largest indices → 最新的token）
        recent_beta = similar_j.topk(k=beta, largest=True).values
        S[recent_beta, i] = 0  # 不标记为冗余
    
    # Step 4: 平均相似度
    S_bar = S.mean(dim=0)  # [n], 每个token被多少token"相似于"
    
    # Step 5: Softmax归一化
    R = torch.softmax(S_bar, dim=0)  # [n], Σ R_i = 1
    
    return R  # 高R_i → token更冗余
```

关键参数：T（similarity threshold）——太低会导致几乎所有 token 对被视为相似，太高会导致无 token 被标记冗余。R-KV 论文未明确给出 T 的具体值（仅说明为"fixed hyperparameter"）。β——控制即使 token 高度重复，最新出现的变体仍被保留。论文也未给出具体值。实际使用时可能需要 calibration。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该 redundancy estimation 是纯 PyTorch 实现，无需特殊硬件或 kernel。核心操作为矩阵乘法（K_norm @ K_norm.T），与标准 attention 的 Q@K^T 类似但 size 为 n×n（n 为 candidate token 数，如 B_budget=1536）。该操作在 GPU 上高效，n=1536 时 ~2.4M 元素矩阵乘法仅需 ~10ms（A100）。主要使用场景：针对推理模型（如 DeepSeek-R1）的长 CoT 解码——这些模型产生的输出含大量重复内容，redundancy estimation 是识别并淘汰冗余 token 的关键。尤其适合数学推理任务（如 MATH-500, AIME）中常见的反复自我验证模式。

局限性：(1) 对 key vector 高度依赖——若模型训练时未产生明显的 key vector 聚集（如短输出任务），redundancy estimation 可能不必要；(2) similarity threshold T 和 beta 的选择影响性能，需针对不同模型/任务 calibrate；(3) O(n²) 的相似度矩阵计算在极端大的 B_budget（如 >10K）时可能成为 bottleneck。

涉及论文标题：
- R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration

## Decoding-Time KV Cache Compression (解码阶段KV Cache压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Decoding-Time KV Cache Compression 是 R-KV 提出的压缩时机范式：与现有方法（SnapKV, PyramidKV 等）在 prefill 阶段一次性 select 并固定 KV cache 不同，R-KV 在 autoregressive decoding 过程中周期性（每 B_buffer tokens）触发压缩。这一设计源于推理模型的特性——generation output 远长于 input prompt（R1-Llama-8B 的 AIME24 平均生成 ~15.5K tokens vs prompt 仅 ~300 tokens），因此主要的 KV cache 增长和冗余发生在 decoding 阶段而非 prefill 阶段。

机制：(1) 分配固定大小的 cache budget B_budget 和 buffer B_buffer（用于存储新生成 tokens）；(2) 每生成 B_buffer 个 token 后触发压缩，将现有 cache（B_budget tokens）+ buffer（前 B_buffer−α tokens）合并为 n = B_budget + B_buffer − α 个候选 KV tokens；(3) 通过 joint selection score（Z = λ·I − (1−λ)·R）选出 top B_budget tokens 保留；(4) 始终保留最后 α 个 observation tokens。

从算法pipeline角度拆解：

```
# Decoding-Time Compression 时序
# 超参: B_budget=1536, B_buffer=128, alpha=8, lambda=0.1

Timeline:
t=0:    生成 [prompt 处理] → KV_cache ← prefill阶段的完整KV
t=128:  触发压缩#1: cache=1536 + buffer=128 → 选top 1536 + α=8 = 1544
t=256:  触发压缩#2: 再次压缩(1544 → 1536 + α)
t=384:  触发压缩#3: ...
...
压缩周期: 每B_buffer=128 tokens一次
压缩操作: n=B_budget+B_buffer-α=1656候选 → B_budget=1536保留 + α=8 obs
```

与 prefill-time compression 的对比：
- Prefill-time (SnapKV): prefill 阶段一次性 select → decode 阶段复用固定 KV，无运行时压缩 → 缺点：prefill 时 unknown 后续 generation 的注意力分布，可能错误淘汰将在 long CoT 中关键的 token
- Decoding-time (R-KV): decode 过程中周期性 re-evaluate → 每次压缩基于最新 observation tokens 的 attention → 动态适应生成过程中的注意力变化

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Decoding-time compression 的实现核心是内存管理：需预先分配 B_budget + B_buffer 的 KV cache 内存（而非 FullKV 的动态增长），避免频繁的 GPU memory allocation。在 PyTorch/HuggingFace 中通过 pre-allocate fixed-size tensors 实现。当前局限性：R-KV 的 decoding-time compression 与 paged attention（vLLM 的 core KV cache 管理机制）不兼容，因为 paged attention 的物理 page 分配是动态的且无 compression 专用接口（Appendix D）。需 serving framework 提供 dedicated KV compression API 支持 efficient memory reallocation。与 training-time compression（如 LoRA-based KV reduction、RL 训练产出更短 CoT）正交，可叠加使用（R-KV 作为 inference-time 加速 + training 产出更少冗余）。

涉及论文标题：
- R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration

---

## Head-wise Similarity-aware Reordering (HSR) for KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

HSR（Head-wise Similarity-aware Reordering）是 ReCalKV 提出的针对 Key 投影矩阵低秩压缩的 attention head 重排序策略。在 grouped SVD 压缩 KV Cache 时，head 的分组方式直接影响近似误差——将具有相似 left singular subspace 的 head 分为一组，可使 SVD 更好地捕获共享子空间结构，从而降低低秩近似误差。HSR 通过三步实现：(1) 计算所有 head 之间的 CKA 相似度矩阵 S ∈ R^{h×h}；(2) 贪心地将 CKA 相似度最高的 head 对分配到同一组（每组大小 s=4）；(3) 剩余 head 填入有空位的组。推理时需对 Key 执行在线 inverse reordering 恢复原始 head 顺序以保证解码等价性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// HSR 流程 (LLaMA-2-7B, h=32 heads, group_size=4)
// Step 1: CKA 计算
W_k ∈ R^{d_model × (h·d_k)}  // Key projection
for i,j in 0..h-1:
    W_i = W_k[:, i*d_k:(i+1)*d_k], W_j = W_k[:, j*d_k:(j+1)*d_k]
    G_i_c = center(W_i @ W_i.T); G_j_c = center(W_j @ W_j.T)
    S[i,j] = Tr(G_i_c @ G_j_c) / sqrt(Tr(G_i_c²)·Tr(G_j_c²))

// Step 2: 贪心分组
groups = [[] * 8]; remaining = set(range(32))
while remaining:
    i,j = argmax_{i,j in remaining} S[i,j]
    assign i,j to non-full group; remaining -= {i,j}

// Step 3: Group SVD + 推理 inverse reordering
order = flatten(groups)
for g in 0..7:
    W_g = concat(W_k heads in group g); L[g],R[g] = SVD_lowrank(W_g, r_g)
// 推理: z_g=x@L[g]; y=z_g@R[g]; inverse_reorder(y) 恢复原始顺序; apply RoPE
```

LLaMA-2-7B, 80% 压缩率：HSR alone 将 WikiText2 PPL 从 9.34 降至 9.01。可视化确认重排序后相邻 head 呈现更高 CKA 相似度。

术语一般如何实现？如何使用？

HSR 完全 offline 执行，PyTorch `torch.linalg.svd()` + 自定义 CKA 计算。group_size=4（32 heads→8 groups）。head permutation 索引需保存用于推理时 inverse 操作，若与 Triton fused kernel 集成则作为 kernel 内在线操作。仅应用于 Key 投影矩阵（Value 投影用 OVC）。

涉及论文标题：
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration

---

## Offline Value Calibration (OVC) for Value Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

OVC 是 ReCalKV 针对 Value 投影矩阵低秩压缩的后校准策略。标准 SVD 低秩分解 W_v ≈ L_v·R_v 不保证最小化在激活分布 X 上的重建误差 E = ||L_v R_v X - W_v X||_F^2。OVC 通过闭式解分别校准 L_v 和 R_v 来直接最小化该误差。Fisher Information 分析显示 Value 投影的 Fisher 显著高于 Key 投影，校准对保持 Value 精度尤为重要。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// 1. Initial SVD: W_v ≈ L_v·R_v
// 2. Calibrate L_v: dE/dL_v=0 →
L_v = W_v·X·X^T·R_v^T·(R_v·X·X^T·R_v^T)^{-1}
// 3. Calibrate R_v: dE/dR_v=0 →
R_v = (L_v^T·L_v)^{-1}·L_v^T·W_v
// 4. Matrix Fusion: W_o_fused = R_v·W_o
// 推理: output = Attention(Q, K, X@L_v) @ W_o_fused
```

LLaMA-2-7B, 80% 压缩率：OVC alone 将 WikiText2 PPL 从 9.34 降至 8.91，LongBench 从 9.01% 升至 13.09%。

术语一般如何实现？如何使用？

256 个 WikiText-2 样本做标定数据，PyTorch 矩阵运算 + `torch.linalg.inv()`，纯代数闭式解（无训练/无梯度）。R_v 通过 Matrix Fusion 与 W_o 合并，推理时零额外开销。r 较小（~64-256），每层校准仅需数秒。

涉及论文标题：
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration

---

## Centered Kernel Alignment (CKA) for Attention Head Similarity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

CKA 是 Kornblith et al. (NeurIPS 2019) 提出的表征相似度度量。给定两个 head 的 Key 投影子矩阵 W_i, W_j ∈ R^{d_model×d_k}，通过线性核 Gram 矩阵 G_i=W_i·W_i^T, G_j=W_j·W_j^T，centering 后计算 CKA(i,j) = HSIC(G_i_c, G_j_c)/√(HSIC(G_i_c,G_i_c)·HSIC(G_j_c,G_j_c))，其中 HSIC(A,B)=Tr(A·B)。CKA ∈ [0,1]，值越高表示 head 子空间结构越相似。ReCalKV 用 CKA 指导 HSR 的 head 分组。

从算法pipeline角度拆解术语：

```
// CKA for head similarity
G_i = W_i @ W_i.T  // [d_model, d_model]
G_i_c = G_i - G_i.mean(0) - G_i.mean(1) + G_i.mean()  // centering
CKA(i,j) = Tr(G_i_c @ G_j_c) / sqrt(Tr(G_i_c²)·Tr(G_j_c²))
```

术语一般如何实现？如何使用？

PyTorch: `torch.mm()` + `torch.trace()`。ReCalKV 在 offline 阶段对所有 head 对计算一次 CKA（O(h²·d_model²·d_k)），结果用于贪心分组。也可用于 layer pruning、head pruning 等需要表征相似度分析的场景。

涉及论文标题：
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration
- xKV: Cross-Layer SVD for KV-Cache Compression

**xKV 中的 CKA 使用**：xKV 将 CKA 用于衡量不同 Transformer 层 KV-Cache 之间的整体结构相似度（而非 head 间相似度）。对于层 ℓ 的 KV-Cache X_ℓ ∈ R^{n×d}（n 为 token 数），在 token 维度计算 centered Gram matrix G_ℓ = H X_ℓ X_ℓ^T H（H 为 centering matrix），CKA(X_ℓ1, X_ℓ2) = Tr(G_ℓ1 G_ℓ2) / √(Tr(G_ℓ1²)·Tr(G_ℓ2²))。高 CKA 值表明层间的主导左奇异向量高度对齐（详见 xKV 论文附录 A: CKA = Σ_{i,j} σ_i² σ_j² (u_i·v_j)² / ...），即使 token-wise cosine similarity 很低。xKV 利用此发现设计跨层 SVD 压缩——通过对齐的奇异向量共享压缩基。

---

## Matrix Fusion for Low-Rank Value Cache Output Projection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Matrix Fusion 是将低秩 Value 压缩的右因子 R_v 预融合进 output projection W_o 的技术。标准 attention: output = Attention(Q, K, X·W_v) · W_o。低秩压缩后: output = Attention(Q, K, X·L_v·R_v) · W_o。Fusion: W_o_fused = R_v·W_o，推理时 output = Attention(Q, K, X·L_v) · W_o_fused，消除在线 Value 重建步骤。

从算法pipeline角度拆解术语：

```
// Offline: W_o_fused = R_v @ W_o  [r_v × d_model]
// 推理: V_latent = X @ L_v [seq_len, r_v], 存入 KV cache
// O = softmax(QK^T/√d) @ V_latent  [seq_len, r_v]
// output = O @ W_o_fused  [seq_len, d_model]
// 节省: 无 Value 重建, O 矩阵缩小, KV cache 从 h·d_k 降至 r_v
```

LLaMA-2-7B, r_v=2048 (50% 压缩)，O 从 [1,4096] 缩至 [1,2048]。

术语一般如何实现？如何使用？

PyTorch 一行: `W_o_fused = torch.mm(R_v, W_o)`，完全 offline。融合后数学等价于先重建再投影，无精度损失。替换原 attention 层的 output projection 权重即可。

涉及论文标题：
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration

---

## Fisher Information-Guided Layer-wise Compression Allocation for KV Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Fisher Information 引导的逐层压缩率分配继承自 Palu (2024)，通过计算每层 Key/Value 投影矩阵的 Fisher Information F(θ)=E[(∂log p(y|x;θ)/∂θ)²] 来估计层重要性。高 Fisher 层保留更多 rank，低 Fisher 层可更激进压缩，实现在固定全局 budget 下的最优分配。

从算法pipeline角度拆解术语：

```
// Fisher 计算: 在 X_calib 上前向传播
for x in X_calib:
    loss = compute_loss(model, x)
    fisher = (∂loss/∂W_kv)²  // gradient squared
// 分配: r[layer] ∝ fisher[layer] / sum(fisher)
```

ReCalKV 使用与 Palu 相同的策略，为 Key 和 Value 分别或联合分配 rank。256 WikiText-2 样本完成计算。

术语一般如何实现？如何使用？

PyTorch: `loss.backward()` + `param.grad`，128-256 样本，offline 一次性计算。不仅用于 KV 压缩，可推广到 weight pruning per-layer sparsity 分配等场景。

涉及论文标题：
- ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration

## Block-Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Block-Sparse Attention（块稀疏注意力）是一种将 KV cache 划分为固定大小的连续 block，并在每个 decode step 中仅选择性地与部分 block 计算注意力的推理优化技术。与 token-level 的稀疏注意力不同，block-sparse 以 block 为粒度（通常 b=16 或 64 tokens），利用 KV cache 在内存中的连续布局特性实现高效 HBM 访问——GPU 从 HBM 加载数据的最小单位为 cache line（128 bytes），token-level 的随机访问无法利用连续内存带宽，而 block 级别的连续读取可大幅提升 memory throughput。

Block-sparse attention 的核心问题是如何快速判断哪些 block 与当前 query 相关。主流方案包括：(a) Quest 的 min/max 描述符——用每个 block 的 key 向量元素级 min/max 计算 approximate upper-bound score，复杂度 O(M·d/b) 而非 O(n·d)；(b) SeerAttention 的可学习 block keys——通过微调训练 block-level key 投影来替代统计描述符；(c) ClusterKV 的聚类方法——将语义相似的 token 聚类后以 cluster 为 block 选择单元。

从算法pipeline角度拆解术语：

```
// Block-Sparse Attention decode step
Input: query q, KV cache K/V, block descriptors B
Output: attention output o

// Step 1: Block selection (approximate matching)
scores = []
for each block i in 0..M:
    block_score = Σ_d max(q[d] × B[i].k_max[d], q[d] × B[i].k_min[d])
    scores.append(block_score)

// Step 2: Top-n selection
selected = top_n(scores) + [n_local recent blocks]

// Step 3: Sparse attention
k_sel = gather(K, selected)  // [n·b, d]
v_sel = gather(V, selected)
o = softmax(q @ k_sel^T / √d) @ v_sel

// Step 4: Online block descriptor update
B[last_block].k_min = elementwise_min(B[last_block].k_min, new_k)
B[last_block].k_max = elementwise_max(B[last_block].k_max, new_k)
```

术语一般如何实现？如何使用？

核心实现要点：(a) kernel 中按 block indices 做 gather/scatter 以仅加载选中 block 的 KV；(b) block descriptor 需在线增量更新（每个 decode step O(1)）；(c) 使用 Flash Decoding 的 split-execution 框架将选中的 block 均匀分配到各 SM。Quest 基于 vLLM PagedAttention 实现，将 block 映射到 virtual page；ReSA 基于 TileLang 实现 GBSA kernel，并新增 GQA group 内共享 attention pattern（shared grouping）减少 block selection 重复计算。Block size 通常取 b=16（Quest/ReSA 默认值）或 b=64。

**Star Attention 的两阶段 Block-Sparse 变体**：Star Attention 使用固定的结构稀疏模式（非动态 per-head 选择）：阶段一采用 block-diagonal local attention（每 block 只 attend 到自身 + anchor block），阶段二对 query token 使用 full global attention。稀疏度由 block_size 决定（建议序列长度的 1/4），阶段一的稀疏性提供加速、阶段二的全局注意力保证精度。与 Quest/MInference 的动态块选择不同，Star Attention 的稀疏模式是算法预设的结构化 mask。

涉及论文标题：
- Rectified Sparse Attention
- Star_Attention__Efficient_LLM_Inference_over_Long_Sequences
- XAttention: Block Sparse Attention with Antidiagonal Scoring

## GEAR (KV Cache Quantization Error Mitigation with Low-Rank Matrix)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

GEAR（Kang et al., 2024）是一种针对 KV cache 量化的误差修正算法。标准量化（如 INT4）将 FP16 KV cache 压缩为低精度表示，但会引入量化误差（即 $\hat{X} - X$，其中 $\hat{X}$ 是反量化后的值，$X$ 是原始值）。当量化误差集中在 outlier 位置时，会严重损害 LLM 输出质量。GEAR 通过两种机制共同近似和补偿量化误差：(1) **low-rank matrix** 用低秩分解近似整体量化误差的主体部分（低秩矩阵用 $UV^T$ 表示，其中 $U \in \mathbb{R}^{d \times r}$, $V \in \mathbb{R}^{r \times L}$，rank $r$ 控制近似精度）；(2) **sparse matrix** 保留少数全精度 outlier 值（sparsity ratio $s$ 控制保留比例），以处理量化误差中极端值集中的部分。两者叠加形成对量化误差的近似 $\tilde{E} \approx UV^T + S$，在推理时加回反量化结果以恢复精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**GEAR 在 KV cache 量化推理中的流程**：
```
# Prefill 阶段：正常计算并存储 FP16 KV cache
K_cache_fp16 = X @ W_K
V_cache_fp16 = X @ W_V

# Decoding 每步的 GEAR 量化+误差修正流程：
for each decode step:
    # 1. 量化 KV cache
    K_int4 = quantize(K_cache_fp16, group_size=G)
    V_int4 = quantize(V_cache_fp16, group_size=G)

    # 2. 反量化得到近似值
    K_approx = dequantize(K_int4)
    V_approx = dequantize(V_int4)

    # 3. 计算量化误差（对过去 window 内的 token）
    E_K = K_cache_fp16[recent_window] - K_approx[recent_window]
    # 不计算全量——仅对近期 window 做误差修正

    # 4. Low-rank 近似：对误差矩阵做 SVD，保留 top-r 奇异值
    U_K, Sigma_K, Vt_K = svd(E_K)
    U_K = U_K[:, :r] @ diag(sqrt(Sigma_K[:r]))
    V_K = Vt_K[:r, :]

    # 5. Sparse 保留：选 |E_K| 最大的 s% 位置保留全精度值
    outlier_indices = topk(abs(E_K), ratio=s)
    S_K = zeros_like(E_K)
    S_K[outlier_indices] = E_K[outlier_indices]

    # 6. Attention 计算时恢复：
    K_recovered = dequantize(K_int4) + (U_K @ V_K) + S_K

    # 7. 与 Q 做 Attention
    scores = Q @ K_recovered^T / sqrt(d_head)
    output = softmax(scores) @ V_recovered
```

**Annotations**: `G` = group_size（量化粒度），`r` = low-rank rank（典型 2%），`s` = sparse ratio（典型 2%）。Low-rank matrix 存储开销 = $r \times (d+L)$ 个 FP16 值，Sparse matrix 仅存储 outlier 位置和值。GEAR 的额外计算开销来自 SVD（一次性的误差分解）和 low-rank/sparse 矩阵的加法恢复，在 prefill 阶段会降低吞吐（论文 Table 3 显示 GEAR prefill 仅有 baseline 的 0.80-0.90×）。

术语一般如何实现？如何使用？

GEAR 开源实现：https://github.com/opengear-project/GEAR。关键参数：sparsity ratio $s$（默认 2%，控制保留全精度的 outlier 数量）和 rank $r$（默认 2%，控制 low-rank 近似矩阵的秩/精度）。GEAR 通过 $s$ 和 $r$ 控制误差修正的精度—内存tradeoff。论文 "Rethinking KV Cache Compression" 的评估显示 GEAR 在 prefill 阶段有显著吞吐下降（因额外 SVD + low-rank 计算开销），在 decode 阶段低 batch size/短 KV length 下可能与 FP16 baseline 持平，但在大 batch size/长 KV length 下吞吐收益有限。GEAR 在 LMDeploy v6.0.1 上的吞吐评估显示：LLaMA-7B TP=1 prefill 仅 0.86× FP16 baseline，decode 1.02×。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

## KIVI (Tuning-Free Asymmetric 2-bit KV Cache Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KIVI（Liu et al., 2024e）是一种免调参（tuning-free）的 KV cache 量化算法，核心设计是非对称量化策略：对 **Key 使用 per-channel 量化**，对 **Value 使用 per-token 量化**。这一设计基于以下观察：Key 张量的不同 channel 之间数值分布差异大（某些 channel 的数值范围比其他 channel 宽很多），因此需要 per-channel 量化为每个 channel 独立计算 scale/zero-point；Value 张量的不同 token 之间数值分布差异大（被 attention 高度关注的 token 的 value 数值范围更宽），因此需要 per-token 量化为每个 token 独立计算 scale/zero-point。KIVI 使用 2-bit 或 4-bit 精度（可配置），并保留最近 $R$ 个 token 为全精度（FP16）以保护近期上下文精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**KIVI 量化/反量化流程**：
```
# 超参数：group_size=G=32 (key), residual_length=R=128

# Key 量化（per-channel）：
# K_cache shape: [num_tokens, num_kv_heads, head_dim]
# Q: 每 channel（head_dim 维度）独立计算 min/max
for c in range(head_dim):
    channel_data = K_cache[:, :, c]           # [num_tokens, num_kv_heads]
    max_val = channel_data.max()
    min_val = channel_data.min()
    scale[c] = (max_val - min_val) / (2^bits - 1)
    zero[c] = round(-min_val / scale[c])
    K_int4[:, :, c] = quantize(channel_data, scale[c], zero[c])

# Value 量化（per-token）：
# Q: 每 token 独立计算 min/max
for t in range(num_tokens - R):    # 跳过最近 R 个 token（保留 FP16）
    token_data = V_cache[t, :, :]           # [num_kv_heads, head_dim]
    max_val = token_data.max()
    min_val = token_data.min()
    scale[t] = (max_val - min_val) / (2^bits - 1)
    zero[t] = round(-min_val / scale[t])
    V_int4[t, :, :] = quantize(token_data, scale[t], zero[t])

# 注意：最近 R=128 个 token 的 KV 保持 FP16 不量化
# 这是为了在近期上下文上保持全精度 Attention

# Attention 计算时反量化：
K_deq = dequantize_per_channel(K_int4, scale_K, zero_K)
V_deq = dequantize_per_token(V_int4, scale_V, zero_V)
# 前 R 个 FP16 token 和其余 INT4 token 拼接
K_full = concat([K_deq, K_cache_fp16[-R:]])
V_full = concat([V_deq, V_cache_fp16[-R:]])

scores = Q @ K_full^T / sqrt(d_head)
output = softmax(scores) @ V_full
```

**Annotations**: Per-channel 量化意味着同一 channel 的所有 token 共享 scale/zero，适合 channel 间分布差异大的 Key。Per-token 量化意味着同一 token 的所有 channel 共享 scale/zero，适合 token 间分布差异大的 Value。Residual length $R$ 保留最近 token 为全精度，兼顾近期上下文质量。

术语一般如何实现？如何使用？

KIVI 开源：https://github.com/jy-yuan/KIVI。关键参数：group_size $G$（key per-channel 量化时 channel 分组大小，默认 32）、residual length $R$（保留 FP16 的最近 token 数，默认 128）、bit-width（2-bit 或 4-bit）。KIVI 作者曾指出 integrate into vLLM 存在困难（GitHub issue #4），主要因为 window-based quantization（保留最近 R token 为 FP16 + 其余为 INT4）与 PagedAttention 的 fixed-type page block 管理不兼容。论文 "Rethinking KV Cache Compression" 将 KIVI 集成到 LMDeploy v6.0.1 中评估：LLaMA-7B TP=1 prefill 1.06×（略超 FP16 baseline），但 decode 仅 0.98×（TP=1）到 0.88×（TP=2），说明量化带来的 memory reduction 在实际 serving 框架中的吞吐收益有限。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

## H2O / Heavy Hitter Oracle (基于累积注意力分数的 KV Cache 动态逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

H2O（Heavy Hitter Oracle, Zhang et al., 2024f）是一种基于注意力分数的 KV cache 动态 eviction 算法。其核心思想是：在自回归生成过程中，追踪并累积每对 (query_head, key_token) 的注意力分数，少量 token（heavy hitters）会持续获得大部分注意力权重。通过保留这些 heavy hitters 的 KV cache 并 evict 其余不重要的 token，可以在几乎不损失精度的情况下大幅减少 KV cache 内存占用。H2O 在每个 decode step 更新 attention score 累积值（通过求和或指数衰减平均），然后用 top-K 选择保留最重要的 KV entries。与 StreamingLLM 不同，H2O 的 eviction 策略是动态的——heavy hitters 可能在生成过程中发生变化，H2O 会自适应调整保留哪些 token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**H2O 动态 KV cache eviction 流程**：
```
# 参数：heavy_hitter_size=64, recent_size=448, total_cache=512

# Prefill 阶段：
K_cache = X @ W_K           # [L, heads, d_head]
V_cache = X @ W_V
Q = X @ W_Q

scores = Q @ K_cache^T / sqrt(d_head)    # [heads, L, L]
# 累积注意力分数（沿 query 维度求和，得到每 key_token 被关注的总分）
attn_accum = scores.sum(dim=-2)           # [heads, L]

# Decode 每步：
for each decode step t:
    # 1. 当前 token 的 Q, K, V
    q_t = x_t @ W_Q    # [heads, d_head]
    k_t = x_t @ W_K
    v_t = x_t @ W_V

    # 2. 与所有缓存的 K 计算注意力分数
    scores_t = q_t @ K_cache^T / sqrt(d_head)   # [heads, 1, len(K_cache)]

    # 3. 更新累积注意力分数（指数移动平均或直接累加）
    attn_accum += scores_t.squeeze(1)            # [heads, len(K_cache)]

    # 4. 选择 heavy hitters + recent tokens
    heavy_idx = topk(attn_accum, k=64)           # 累积分数最高的 64 个
    recent_idx = [len(K_cache)-448, ..., len(K_cache)-1]  # 最近 448 个
    keep_idx = union(heavy_idx, recent_idx)

    # 5. Evict 不重要的 KV cache
    K_cache = K_cache[keep_idx]
    V_cache = V_cache[keep_idx]
    attn_accum = attn_accum[keep_idx]             # 同步裁剪分数

    # 6. 追加新 KV
    K_cache = concat([K_cache, k_t])
    V_cache = concat([V_cache, v_t])

    # 7. 在保留的 KV 上计算 Attention
    output_t = softmax(q_t @ K_cache^T / sqrt(d_head)) @ V_cache
```

**Annotations**: Heavy hitter 选择用 `topk(attn_accum, k=64)` 动态确定每步最重要的 64 个历史 token。Recent window（448 token）保证近期上下文完整性。总 KV cache 大小 ≈ 512（64+448），相比全量 KV cache（可能数万 token）压缩显著。H2O 的关键开销：每步需计算一次完整 attention scores（以更新 attn_accum），这与 FlashAttention 的单 pass 设计冲突——FlashAttention 不保存中间 attention scores，导致 H2O 需要额外的 multi-pass attention 和内存访问。

术语一般如何实现？如何使用？

H2O 开源：https://github.com/FMInference/H2O。关键参数：heavy hitter oracle token size（默认 64）、recent size（默认 448）、total cache size（默认 512）。动态 eviction 每步计算 attention scores → 更新累积分数 → top-K 选择 → evict 不需要的 KV。论文 "Rethinking KV Cache Compression" 的评估显示：H2O 与 FlashAttention 不兼容（需要 multi-pass attention 获取 attention scores），因此在 LMDeploy（含 FlashAttention）上 prefill 吞吐仅 0.51-0.58× FP16 baseline（LLaMA-7B TP=1/2/4），decode 为 0.85-1.34×，依赖于 batch size 和 KV length。H2O 的动态 eviction 导致 KV cache length 不单调增长，与 PagedAttention 的 fixed-size page 管理冲突。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

## Negative Sample in KV Cache Compression (KV Cache 压缩的负样本分析)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Negative sample（负样本）是在 KV cache 压缩上下文中的一个评估概念：指那些在原始未压缩 LLM 下表现正常（benign），但在应用 KV cache 压缩后精度显著退化的样本。论文 "Rethinking KV Cache Compression" 首次系统性地引入此概念，并使用 Algorithm 1 定义收集流程：给定数据集 $\mathcal{D}$、阈值 $\theta$、LLM $\mathcal{M}$、baseline 算法 $\mathcal{A}_b$ 和压缩算法集合 $\mathcal{A}$，若某样本 $d_i$ 在所有压缩算法下的 accuracy 均低于 $(1-\theta) \times p_{\text{base}}$（其中 $p_{\text{base}}$ 为 baseline 下的 accuracy），则该样本被标记为 negative sample，加入 $\mathcal{D}_{neg}$。关键发现：即使压缩算法整体平均 accuracy 损失很小（<1%），仍存在大量 negative samples（例如 threshold=10% 时数百个），揭示压缩算法在不同样本和任务类型上的脆弱性不均衡。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Negative Sample 收集流程（论文 Algorithm 1 复述）**：
```
Input: Dataset D, Threshold theta (e.g., 0.10),
       LLM M, Baseline Algorithm A_base, Compression Algorithm Set A = {A1, A2, ...}
Output: Negative Dataset D_neg

D_neg = empty_set()

for each sample d_i in D:
    # Step 1: 获取 baseline 精度
    p_base = Accuracy(A_base, M, d_i)

    # Step 2: 只评估 benign samples（baseline 精度 >= 平均值）
    if p_base < average(D, A_base):
        continue

    # Step 3: 检查所有压缩算法是否都 fail
    negative = true
    for each A_j in A:
        p_comp = Accuracy(A_j, M, d_i)
        if p_comp >= (1 - theta) * p_base:
            negative = false    # 至少一个算法通过
            break

    # Step 4: 所有算法都 fail → negative sample
    if negative:
        D_neg.insert(d_i)

return D_neg
```

**Annotations**: 阈值 $\theta$ 控制严格度——$\theta=0.10$ 意味着压缩后 accuracy 不得低于 baseline 的 90%。论文使用 LongBench 数据集和 LLaMA-3.1-8B-instruct 评估，发现 negative samples 集中在 summarization 和 QA 任务类型（这些任务严重依赖长上下文信息，KV cache 压缩导致的关键信息丢失对它们影响最大）。多个压缩算法（KIVI+GEAR 或 H2O+StreamingLLM）联合时 negative samples 减少但不能完全消除。

术语一般如何实现？如何使用？

论文提供的工具链（https://github.com/LLMkvsys/rethink-kv-compression）包含 negative sample evaluator：从 LongBench 收集的、经 10% threshold 筛选的 negative samples 构成 benchmark 数据集，用于评估新的 KV cache 压缩方法在困难样本上的表现。论文 Table 7 显示：LLaMA-3.1-8B-instruct 上 baseline FP16 在 summarization/QA/code 上得分为 31.6/52.0/97.0，而 KIVI 降至 24.8/28.8/30.0——code 任务退化最严重（97→30）。论文推荐的方向：(1) 用 lightweight model 预测请求的 task type，(2) 开发 task-specific 压缩策略，(3) 对不同 task 使用不同压缩级别。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

## Response Length Distribution Shift from KV Cache Compression (KV Cache 压缩引起的响应长度分布偏移 / Verbose Output)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV cache 压缩引起的响应长度分布偏移是论文 "Rethinking KV Cache Compression" 首次系统揭示的现象：有损 KV cache 压缩（量化或稀疏 eviction）会导致 LLM 生成比 FP16 baseline 更长的响应（verbose output），且高压缩比加剧此效应。论文通过定义响应长度差异 $D = (L^{un} - L^{cs})/L^{un}$（$L^{un}$ = 未压缩时的响应长度，$L^{cs}$ = 压缩后的响应长度），负值表示压缩导致更长输出。论文用 ShareGPT 1000 样本和 LLaMA-3.1-8B-instruct 测量发现：KIVI/GEAR/H2O/StreamingLLM 均导致 >20% 样本的输出长度增加 ≥50%（1.55-1.76× 平均 length increase）。语义相似度测试（Table 4）进一步表明：压缩后的更长输出并非质量提升，而是在相似或略低的语义质量下更 verbose。这一发现对实际部署有直接含义：即使压缩提升了 tokens/s 吞吐，更长的输出可能完全抵消甚至逆转端到端延迟收益。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Verbose Output 判定流程（论文 Section 4.3）**：
```
# 给定：同一 prompt，FP16 baseline 和 压缩算法分别生成
response_fp16 = generate(LLM, prompt, compression=None)
response_comp = generate(LLM, prompt, compression=algo)

Q_fp16 = semantic_score(response_fp16, reference)
Q_comp = semantic_score(response_comp, reference)
L_fp16 = len(response_fp16)
L_comp = len(response_comp)

# Verbose 判定：
is_verbose = (Q_comp <= Q_fp16) AND (L_comp >= L_fp16)
# 即：质量没提升（甚至下降），但输出更长

# 长度差异度量：
D = (L_fp16 - L_comp) / L_fp16
# D < 0 → 压缩导致更长输出
# D > 0 → 压缩导致更短输出
```

**Annotations**: 论文 Table 4 显示：FP16 semantic score=49.6，KIVI=50.7（略高），GEAR/H2O/Stream=46.2-46.3（略低），但所有压缩算法的 length increase 均为 1.55-1.76×。温度参数 T=0.9 和 T=1.1 分别导致 ~45% 样本变长和 ~20% 样本变短——大致对称。而 KV cache 压缩则显著非对称地偏向更长输出（>20% samples with 1.5×+ length increase）。

术语一般如何实现？如何使用？

论文提出 **Length Predictor** 作为工具来预测给定压缩算法下某 prompt 的可能响应长度：使用 LongFormer (max_seq_len=4096) 作为 BERT-based classifier，输入为 prompt text，输出为 response_length/prompt_length ratio，训练数据来自压缩算法在 ShareGPT 上的实际生成。精度 >85%（Table 6/10）。在请求路由器中，length predictor 结合 throughput predictor 估计每请求的端到端延迟，用于路由决策。论文 Table 8 显示：仅用 throughput predictor 可加速 1.18-1.48×，加上 length predictor 后可进一步提升至 1.45-1.80× E2E latency speedup。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving

## KV Cache Error Accumulation (KV Cache 误差累积)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV Cache Error Accumulation 指在稀疏解码过程中，由于每个 decode step 使用近似稀疏注意力而非精确 dense attention，生成 token 及其 KV cache 条目包含近似误差，这些误差被写入 KV cache 后随 decoding 步数持续累积的现象。传统 KV cache 假设每步由精确 attention 计算——稀疏解码打破这一假设：t₀ 时刻的近似 attention 产生带误差的 token₀，token₀ 的 K₀/V₀ 写入 cache；t₁ 时刻的 attention 基于不精确的 K₀/V₀ 做近似计算，产生更大误差的 token₁ 和 K₁/V₁，形成"误差累积"正反馈闭环。

这解释了为何 sparse decoding（如 Quest、InfLLM）的性能随解码长度增长而下降（ReSA Figure 1）：短解码时仅有少量 token 经过稀疏 attention，误差小；长解码时绝大多数 token 都经过稀疏 attention，误差逐级放大。

从算法pipeline角度拆解术语：

```
// KV Cache 误差累积的数学刻画
t=0 (prefill):  K_0 = K_dense, V_0 = V_dense   (e_0 = 0)
t=1:            token_1 = SparseAttn(q_1, K_0, V_0) + ε_1
                K_1 = K_0 ∪ {k_1 + ε_k1}
t=2:            token_2 = SparseAttn(q_2, K_1, V_1) + ε_2  (|ε_2| > |ε_1|)
t=T:            token_T = SparseAttn(q_T, K_{T-1}, V_{T-1}) + ε_T
                累积误差 ≈ Σ ε_i (单调增长)

// ReSA 方案: 每 f 步 dense rectification 限制误差窗口
if t % f == 0:
    K_t, V_t = DenseAttn(tokens[t-f:t], K_{t-f}, V_{t-f})
    max_error ≤ f · avg(|ε_i|)
```

术语一般如何实现？如何使用？

检测方法：对比 sparse vs dense decoding 在相同 prompt 下生成质量随 decode length 变化曲线。缓解方法：(a) ReSA 的 periodic dense rectification；(b) TriForce/MagicDec 的 self-speculation（sparse KV drafting + dense KV verification）；(c) Quest 的跳过前两层策略，但 ReSA 实验表明该方法改善有限。

涉及论文标题：
- Rectified Sparse Attention

## Dense Rectification

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dense Rectification（密集校正）是 ReSA 论文的核心机制：在稀疏解码每生成 f 个 token 后，将这 f 个 token 通过一次并行 dense attention forward pass 重新编码，刷新 KV cache 中对应的条目，将稀疏误差的累积范围限制在最近 f 步以内。其设计关键：(a) 批量并行——f 个 token 拼成 mini-batch，一次 dense forward 同时重编码；(b) 同步刷新 block key cache（block descriptors），否则新稀疏解码的 block selection 基于过时描述符会加剧误差；(c) 频率 f 权衡质量/效率——f=32 近 dense 精度，f=128 保留大部分增益但 overhead 更低。

从算法pipeline角度拆解术语：

```
Algorithm: Rectified Sparse Decoding
Input: P(prompt), M(model), f(frequency), T(max steps)

K, B = Prefill(P)  // dense prefill
for i = 1 to T:
    t = SparseForward(G[i-1], K, B)  // GBSA with block selection
    G.append(t); K.update(t); B.update(t)
    if i % f == 0:
        K, B = DenseForward(G[i-f:i], K, B)  // batch rectification
```

Memory access 成本：Avg(mem) = mem(KV cache) × (1/b + p + 1/f)，三项分别对应 block descriptor scan、sparse attention、rectification 摊销。256K context 下 rectification 占 attention 总延迟 32.7%。

术语一般如何实现？如何使用？

需支持在同一 session 中交替使用 sparse 和 dense attention kernel，dense forward 仅作用于最近 f 个 token。ReSA kernel 基于 TileLang + Flash Decoding split-execution。与 speculative decoding 的区别：rectification 无条件接受 sparse decoding 产生的所有 token 并用 dense 刷新 KV，避免 per-token accept/reject latency 惩罚（ReSA Table 3: 平均 1.92× faster than self-speculation）。天然兼容 continuous batching 和 chunked prefill。

涉及论文标题：
- Rectified Sparse Attention

## Group Block Sparse Attention (GBSA) with Shared Grouping

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

GBSA 是 ReSA 在 Quest block-sparse attention 基础上为 GQA 模型优化的变体。利用 GQA 结构——同一 GQA group 内多个 query heads 共享同一组 KV heads——在 group 级别统一做 block selection，组内所有 query heads 复用相同 selected block indices。关键步骤：(1) 对 GQA group 内所有 query heads 做平均池化得 q_pool；(2) 仅用 q_pool 与 block descriptors 计算一次 similarity scores；(3) 选 top-n block indices；(4) 组内所有 heads 共享这组 indices 做 sparse attention。

从算法pipeline角度拆解术语：

```
for each GQA group j in 0..h_kv:
    q_pool = mean(Q[j, :, :, :], dim=0)  // g query heads pooling
    // 仅一次 block selection per group
    selected = top_n([Σ_d max(q_pool[d]×k_max_i[d], q_pool[d]×k_min_i[d]) for i in 0..M])
    // g query heads 共享 selected blocks
    for each head q in group j:
        o = softmax(q @ K[j,selected,:]^T / √d) @ V[j,selected,:]
```

与 per-head block selection 相比，GBSA 的 block selection 计算从 O(h_query × M × d) 降至 O(h_kv × M × d)。Qwen2.5 7B 配置下（28 query heads / 4 KV heads），head 维度减少 7×。同一 SM 上为多个 query head 加载的 KV block 数据可在 warp/thread 间通过 shared memory 复用。

术语一般如何实现？如何使用？

每个 GQA group 固定分配到一个 SM，SM 内为 g 个 query heads 计算 sparse attention。KV block 数据加载一次后在 SM 内多 warp 间共享。Block indices 在同 group 多 head 间直接复用。ReSA 使用 TileLang 实现该 kernel。

涉及论文标题：
- Rectified Sparse Attention

---

## Scale-invariant Attention (尺度不变注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Scale-invariant Attention 是一种对 Transformer attention logits 施加位置依赖变换的注意力机制，由 Anson et al. (2025) 提出。核心思想源于自然图像的尺度不变（scale-invariant）统计特性——图像中存在所有空间尺度的重要特征。类比到文本 attention，token 范围也划分为不同尺度（1-10 tokens、10-100 tokens、100-1000 tokens 等），每个尺度的信息都应被保留。

Scale-invariant Attention 满足两个数学性质：
1. **Scale-invariant Total Attention**：在任意 token 范围 $[t, t\Delta)$ 内的 expected total unnormalized attention $\mathbb{E}[Z_t^{t\Delta}] = \Theta(1)$，即各范围的总注意力渐进恒定。
2. **Weak Scale-invariant Attention Sparsity**：$\mathbb{E}[H_t^{t\Delta}] = o(\log t)$，即注意力熵随 $t$ 亚对数增长，稀疏性随上下文变长而增加。

实现方式：在 attention score $S_t$ 上施加位置依赖变换 $L_t = a_t \cdot S_t + m_t$，其中（在 IID Gaussian logits 假设下，边界条件 $a_0^2=1, m_0=0$ 得 $\alpha=\beta=e^{0.5}$）：
$$a_t = \sqrt{2[\log(t/\tau+1) - \log\alpha + \beta/\alpha]}, \quad m_t = -a_t^2 + \beta/\alpha$$

唯一超参数 $\tau$（长度尺度，最优约 10）控制"局部区域"大小。$t \ll \tau$ 时 $a_t \approx 1, m_t \approx 0$（局部近似标准 attention）；$t \gg \tau$ 时 $a_t^2$ 对数增长（分布尖锐化），$m_t$ 对数下降（压低远距离总权重）。

从算法pipeline角度拆解术语，给出具体例子。

**Scale-invariant attention forward pass（使用 FlexAttention）**：

```
输入: Q [B, H, T, d], K [B, H, T, d], V [B, H, T, d]
超参数: τ = 10, α = β = e^{0.5}

# score_mod 函数（FlexAttention）
def scale_invariant_score_mod(score, b, h, q_idx, kv_idx):
    t = q_idx - kv_idx  # 距离
    if t >= 0:
        f_t = log(t/τ + 1) - log(α)
        a_t = sqrt(2 * (f_t + β/α))
        m_t = -a_t**2 + β/α
        return a_t * score + m_t
    return score

# FlexAttention 自动处理 causal mask、block-sparse 编译、反向传播
output = flex_attention(Q, K, V, score_mod=scale_invariant_score_mod)
```

**与 LogN/ALiBi 的关键区别**：
- LogN: $L_t = s\log N \cdot S_t$（位置无关，全局缩放）→ 牺牲局部注意力
- ALiBi: $L_t = S_t - m \cdot t$（线性刚性偏置）→ 无法灵活控制熵
- Scale-invariant: $L_t = a_t S_t + m_t$（位置依赖）→ 局部稠密 + 全局稀疏

术语一般如何实现？如何使用？

基于 modded-nanogpt（PyTorch），使用 FlexAttention API 定义 score_mod。论文在 GPT-2-style 162M/304M 和 Llama 2 7B 上验证。训练：FineWeb 数据集，Muon（线性层）+ Adam（embedding）。评估：验证 loss + needle-in-a-haystack。性能（162M Train@4k/Val@64k）：Val loss=3.247 vs LogN+RoPE 3.378 vs ALiBi 3.270，Needle@64k 准确率 0.969。

涉及论文标题：
- Scale-invariant Attention

---

## LogN Scaling / SSMax (Scalable Softmax / LogN Trick)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LogN Scaling（SSMax / Scalable Softmax / LogN trick）是对 attention logits 施加与序列长度 $N$ 相关的全局缩放的技术：$L_t = s \log N \cdot S_t$，其中 $s$ 为可学习标量。最早由 Jianlin (2021) 从熵不变性角度提出，后经 Chiang & Cholak (2022)、Nakanishi (2025) 系统研究。动机：随 $N$ 增长，标准 softmax 分布趋于均匀（熵增高），注意力分散到过多无关 token。LogN 通过 $\log N$ 因子放大 logits 使 softmax 更尖锐，在长上下文时保持聚焦。

从算法pipeline角度拆解术语，给出具体例子。

```
# LogN attention forward pass
N = seq_len
scale = s * log(N)     # s 为可学习参数
S = Q @ K^T / sqrt(d)  # attention scores
L = scale * S           # LogN 缩放
A = softmax(L)          # 更尖锐的分布
output = A @ V
```

**核心缺陷**（Scale-invariant Attention 论文指出）：LogN 是位置无关的全局缩放——对近处 token（$t=1-100$）和远处 token（$t=10000+$）施加相同缩放因子。这导致：(1) 局部上下文的总注意力随 $N$ 增长快速衰减；(2) 无法实现"局部稠密 + 全局稀疏"的理想模式。实验显示：LogN+RoPE 在 @64k 时 Val loss=3.378，而 Scale-invariant p-RoPE=3.247。

术语一般如何实现？如何使用？

实现极为简单——softmax 前乘以 $s \log N$。$s$ 通过梯度下降学习或手工设定。与 p-RoPE 组合优于与 RoPE 组合。适用于不需要精细局部注意力控制的场景。

涉及论文标题：
- Scale-invariant Attention

---

## p-RoPE (Partial Rotary Position Embedding / 部分旋转位置编码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

p-RoPE (Barbero et al., 2024b) 是 RoPE 的一种变体，通过降低 effective base $\theta_{\text{eff}}$（如 1024 vs 标准 10000）来排除低频/高波长成分。RoPE 的频率为 $\theta_i = \theta_{\text{base}}^{-2i/d}$，高 $i$ 对应低频/长波长（编码远距离位置）。这些低频成分在长上下文泛化时可能有害：其周期超过训练时最大序列长度，推理时遇到未见过的相位。p-RoPE 通过降低 $\theta_{\text{base}}$ 将所有波长限制在更短范围。

从算法pipeline角度拆解术语：

p-RoPE 的实现仅需修改 RoPE 的 $\theta_{\text{base}}$。对 $d=128$：RoPE 最大波长 $\approx 56000$ tokens，p-RoPE ($\theta=1024$) 最大波长 $\approx 5730$ tokens。

**与 Scale-invariant Attention 的关系**：论文发现 scale-invariant RoPE 在长上下文泛化时不如 scale-invariant p-RoPE。假设：RoPE 的低频成分会干扰位置依赖的 logit 变换 $a_t, m_t$，而 p-RoPE 通过移除低频成分消除此冲突。

术语一般如何实现？如何使用？

在 HuggingFace transformers 或自定义 RoPE 中设置 `rope_theta=1024`。p-RoPE 会略微降低短上下文 in-distribution 性能，但在长上下文泛化场景收益显著。LogN+p-RoPE 优于 LogN+RoPE，Scale-invariant p-RoPE 优于 Scale-invariant RoPE。

涉及论文标题：
- Scale-invariant Attention

---

## Attention Entropy Control (注意力熵控制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention Entropy Control 是一类通过控制 softmax 后注意力分布熵来实现长上下文泛化的技术族。核心问题：从训练长度 $N_{\text{train}}$ 扩展到推理长度 $N_{\text{inf}} \gg N_{\text{train}}$ 时，标准 attention 分布趋于均匀（高熵），注意力浪费在无关 token 上。

Token 范围 $[t_1, t_2]$ 内的熵定义为：
$$H_{t_1}^{t_2} = -\sum_{t=t_1}^{t_2-1} \frac{\tilde{A}_t}{Z_{t_1}^{t_2}} \log\left(\frac{\tilde{A}_t}{Z_{t_1}^{t_2}}\right)$$

现有方法分类：
1. **全局缩放**（LogN）：$L_t = s\log N \cdot S_t$ —— 位置无关
2. **加性偏置**（ALiBi）：$L_t = S_t - m \cdot t$ —— 线性刚性
3. **位置依赖变换**（Scale-invariant）：$L_t = a_t S_t + m_t$ —— 局部恒等、全局稀疏

从算法pipeline角度拆解术语：

三种方法的 entropy-scaling 行为比较（IID Gaussian logits）：

| 方法 | $H_t^{t\Delta}$ | 局部注意力保持 | 全局稀疏性 |
|------|----------------|--------------|-----------|
| 无缩放 | $\Theta(\log t)$ | 差 | 无 |
| LogN | sub-log | 差 (随 $N$ 衰减) | 强 |
| Scale-invariant | $\sim \sqrt{\log t}$ | 好 | 弱 (sub-linear) |

术语一般如何实现？如何使用？

通过 attention score modification 实现（FlexAttention score_mod）。选择指南：LogN 适合不需要精细局部控制的场景（实现最简单）；ALiBi 适合极端长度外推（零额外参数）；Scale-invariant 适合需同时保持局部稠密和全局稀疏的场景（仅增一个超参数 $\tau$）。

涉及论文标题：
- Scale-invariant Attention

## Attention Gate (AttnGate)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention Gate（AttnGate）是 SeerAttention/SeerAttention-R 提出的轻量级可学习门控模块，插入预训练 Transformer 注意力层之前，用于预测哪些 KV blocks 对当前 query token（或 query 序列）最重要，从而实现 block-level 稀疏注意力。AttnGate 受到 MoE gating 机制启发，但用于注意力稀疏性预测而非专家路由。

AttnGate 的核心计算流程（decode 阶段，SeerAttention-R）：
1. **Q 分支**：取 pre-RoPE 的 Q tensor（multi-head），通过线性层 W_q_gate 将 GQA group 内 query heads 聚合为 KV-head 数量，再应用 RoPE → Q_gate ∈ R^{1, num_kv_heads, d_gate}
2. **K 分支**：取 pre-RoPE 的 K tensor，进行 Max/Min/Avg 三种非重叠块级 pooling（pooling kernel size = block_size），concat 后通过线性层 W_k_gate，再应用 RoPE → K_gate ∈ R^{num_blocks, num_kv_heads, d_gate}
3. **块级注意力分数**：S = softmax(Q_gate @ K_gate^T / sqrt(d_gate))，输出每块的激活分数
4. **稀疏化**：通过 Top-K（token budget）或阈值过滤将软分数转换为二进制块掩码/块索引

AttnGate 的参数量极小：对 8B 模型约 66MB，仅原始模型参数的 ~0.8%。训练时仅更新 AttnGate 参数（冻结原始模型权重），使用 KL 散度损失将 AttnGate 输出对齐到原始模型注意力分布的 block-level ground truth。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# AttnGate 在 decode 阶段的推理流程（SeerAttention-R）
# 输入: 单 token decode, Q ∈ R^{1, num_heads, d_head}, K_cache ∈ R^{seq_len, num_kv_heads, d_head}

# === AttnGate 前向: 稀疏块选择 ===
def attngate_forward(Q, K_compression_cache, block_size, token_budget):
    # Step 1: Q 分支 — GQA head aggregation
    Q_nope, Q_pe = split_rope(Q)                    # 分离 RoPE 部分
    Q_reshaped = reshape(Q_nope, [num_kv_heads, g*d_head])  # 按 GQA group 重组
    Q_gate_ = W_q_gate @ Q_reshaped                 # [num_kv_heads, d_gate]
    Q_gate = RoPE(Q_gate_, Q_pe[0])                 # 重新应用 RoPE
    
    # Step 2: K 分支 — 使用 K Compression Cache
    # K_compression_cache 已存储压缩后的 K 表示
    K_gate = K_compression_cache                    # [num_blocks, num_kv_heads, d_gate]
    
    # Step 3: 块级注意力分数
    S = softmax((Q_gate @ K_gate.T) / sqrt(d_gate))  # [1, num_kv_heads, num_blocks]
    
    # Step 4: Top-K 选择
    block_budget = token_budget // block_size
    selected_blocks = topk(S, k=block_budget, dim=-1)  # [1, num_kv_heads, block_budget]
    selected_blocks = selected_blocks ∪ {last_incomplete_block}  # 始终包含最后不完整块
    
    return selected_blocks

# === 块稀疏 Attention ===
def block_sparse_attention(Q, K_cache, V_cache, selected_blocks, block_size):
    O = zeros_like(Q)
    m_prev = -inf
    for block_idx in selected_blocks:
        K_block = K_cache[block_idx*block_size : (block_idx+1)*block_size]
        V_block = V_cache[block_idx*block_size : (block_idx+1)*block_size]
        S_block = Q @ K_block.T / sqrt(d_head)
        # FlashAttention online softmax rescaling
        m_new = max(m_prev, rowmax(S_block))
        O = diag(exp(m_prev - m_new)) * O + exp(S_block - m_new) @ V_block
        m_prev = m_new
    return O
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

AttnGate 通过 HuggingFace Transformers 的模型修改实现：在原始 attention 层的 forward 函数中插入 AttnGate 模块。训练时使用修改版 FlashAttention-2 kernel 同时计算 attention output 和 block-level ground truth（column-wise 1D maxpooled attention scores），用 KL divergence 训练。推理时，AttnGate 的 K 分支利用 K Compression Cache 避免重复计算历史 token 的压缩表示。预训练好的 AttnGate 权重已发布在 HuggingFace（https://huggingface.co/SeerAttention）。

涉及论文标题：
- SeerAttention-R: Sparse Attention Adaptation for Long Reasoning
- SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs (NeurIPS 2025)

---

## Self-Distilled Attention Sparsity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Self-Distilled Attention Sparsity（自蒸馏注意力稀疏）是 SeerAttention 系列提出的 post-training 训练范式：让一个轻量级 AttnGate 模块通过蒸馏学习原始预训练模型自身的注意力稀疏模式，无需修改原始模型参数。与 knowledge distillation（大模型教小模型）不同，这里"教师"和"学生"是同一个模型——AttnGate 学习的是原始模型注意力分布中的稀疏结构。

训练流程：
1. 用原始模型对训练数据做完整 attention forward
2. 对完整 attention scores 做 block-level maxpooling（prefill 阶段 2D maxpool，decode 阶段 1D column-wise maxpool）
3. 对 GQA group 内 query heads 再做一次 maxpool，得到 KV-head 级别的 ground truth
4. 归一化 ground truth 使和为 1
5. AttnGate 通过 KL divergence loss 学习预测与 ground truth 一致的块激活分布
6. 仅更新 AttnGate 参数（通常 <1% 模型参数），原始模型权重冻结

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Self-Distillation Training Loop
for batch in dataloader:  # e.g. OpenR1-MATH-220K, packed to 32K seq
    # === Teacher: 完整 attention forward（修改版 FA2 kernel）===
    # modified_fa2_kernel 同时输出 attention output 和 block-level ground truth
    O_dense, ground_truth = modified_fa2_forward(Q, K, V, block_size)
    # ground_truth 生成:
    #   1. 计算 full attention scores A = QK^T/sqrt(d)
    #   2. Column-wise 1D maxpool: A_pooled[t] = max(A[t, b*s : (b+1)*s])
    #   3. GQA group 内 maxpool: gt = max over query heads in each group
    #   4. Normalize: gt = gt / sum(gt)
    
    # === Student: AttnGate 预测 ===
    S_pred = attngate_forward(Q, K, block_size)  # [num_kv_heads, num_blocks]
    
    # === Loss ===
    loss = KL_divergence(S_pred, ground_truth)
    loss.backward()  # 仅 AttnGate 参数有梯度
    
# 训练配置: 0.4B tokens, batch_size=16, 800 steps, lr=1e-3, cosine decay
# 硬件: AMD MI300x GPU, DeepSpeed ZeRO-2
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现上需要：(1) 修改 FlashAttention kernel 以在计算 attention 的同时生成 block-level ground truth（复用 FlashAttention 的 block-level rowmax 等中间结果，几乎零额外开销）；(2) 将 AttnGate 模块插入每层 attention layer。与从头预训练稀疏注意力（如 NSA、MoBA）相比，自蒸馏方法可以将稀疏注意力以 plug-in 方式添加到任意预训练模型中，训练开销极小（8B 模型仅需 12 GPU hours on MI300x）。

涉及论文标题：
- SeerAttention-R: Sparse Attention Adaptation for Long Reasoning
- SeerAttention: Learning Intrinsic Sparse Attention in Your LLMs (NeurIPS 2025)

---

## K Compression Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

K Compression Cache 是 SeerAttention-R 中为加速 AttnGate 推理而设计的压缩 key 缓存。类似于标准 KV Cache 缓存原始 key/value 以避免重复计算，K Compression Cache 缓存的是经过 pooling + 线性投影压缩后的 key 表示，用于 AttnGate 的 K 分支快速计算。

核心设计：
- 只缓存压缩后的 K_gate（经 Max/Min/Avg pooling + W_k_gate 线性层），而非原始 K
- 更新策略：accumulate 直到生成 block_size 个新 token，才计算这 block_size 个 token 的压缩表示并追加到 cache
- 在 accumulate 期间，最后一个不完整的 block 始终被标记为"选中"（补偿 K Compression Cache 信息滞后）
- block_size=64 时，K Compression Cache 内存仅占原始 KV cache 的 1/128 (<1%)：cache 中每个 block 存 d_gate 维向量（而非 block_size × d_head × num_kv_heads 维完整 K）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# K Compression Cache 更新与使用
class KCompressionCache:
    def __init__(self, num_layers, num_kv_heads, d_gate, block_size):
        # 内存: num_layers × num_blocks × num_kv_heads × d_gate
        # vs KV cache: num_layers × seq_len × num_kv_heads × (d_head × 2)
        # 比例: (1/block_size) × (d_gate / (2*d_head)) ≈ 1/128 (block_size=64)
        self.cache = []           # list of [1, num_kv_heads, d_gate] per block
        self.accumulated_tokens = 0
    
    def update(self, new_K_tokens, K_gate_projector):
        # new_K_tokens: 自上次更新后新生成的 K (pe + nope)
        self.accumulated_tokens += len(new_K_tokens)
        
        if self.accumulated_tokens >= block_size:
            # 取最近 block_size 个 token 的 K
            K_block = new_K_tokens[-block_size:]
            # Pooling + 线性投影
            K_pooled = concat([
                MaxPool(K_block, kernel=block_size),
                MinPool(K_block, kernel=block_size),
                AvgPool(K_block, kernel=block_size)
            ])  # [1, num_kv_heads, 3*d_head]
            K_gate_new = K_gate_projector(K_pooled)  # [1, num_kv_heads, d_gate]
            self.cache.append(K_gate_new)
            self.accumulated_tokens = 0
            return True  # cache 已更新
        return False  # 还在 accumulate，cache 未更新
    
    def get_full_cache(self):
        return stack(self.cache)  # [num_blocks, num_kv_heads, d_gate]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

K Compression Cache 在推理时与 KV Cache 并行维护。每次 decode step 中：AttnGate 先读取 K Compression Cache 计算块级分数，Top-K 选择后，再仅从 KV Cache 中加载被选中的原始 K/V blocks 计算 attention。由于 K Compression Cache 极小，它可以始终驻留在 GPU 显存中，而完整的 KV Cache 可以 offload 到 CPU/SSD，按需加载被选中的 blocks。这为极端长序列推理（如 128K+）提供了高效的混合内存管理方案。

涉及论文标题：
- SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

---

## Oracle Block Sparse Selection (Oracle 块稀疏选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Oracle Block Sparse Selection 是一种评估稀疏注意力方法准确率上界的实验技术：使用真实完整注意力分数（ground truth attention scores）来选择哪些 KV blocks 参与计算，而非使用任何近似或预测方法。由于需要先计算完整 attention 再做选择，oracle 方法本身无法加速推理，但可以回答"如果稀疏选择是完美的，模型准确率能保持到什么程度？"

在 SeerAttention-R 中，oracle sparsity 的实现：
1. 对每个 decode step，先计算完整 attention scores (Q @ K^T)
2. 对 attention scores 做 column-wise 1D maxpooling（每个 block 取最大值）
3. 对 GQA group 内做 maxpool 得到 KV-head 级别的分数
4. Top-K 选择分数最高的 blocks
5. 仅用选中的 blocks 重新计算 attention（实际上可以复用第一次的结果）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Oracle Block Sparse Selection 评估流程
def oracle_sparse_eval(model, prompt, block_size=64, token_budget=4096):
    block_budget = token_budget // block_size
    
    for each decode step t:
        # Step 1: 计算完整 attention（仅在评估时做，实际部署不这么做）
        Q = current_query_token        # [1, num_heads, d_head]
        K = past_kv_cache              # [t, num_kv_heads, d_head]
        A_full = Q @ K.T / sqrt(d_head)  # [1, num_heads, t]
        
        # Step 2: Block-level maxpooling + GQA group pooling
        A_blocks = column_maxpool(A_full, block_size)  # [1, num_heads, num_blocks]
        A_kv = maxpool_over_gqa_group(A_blocks)         # [1, num_kv_heads, num_blocks]
        
        # Step 3: Oracle Top-K（完美选择）
        selected = topk(A_kv, k=block_budget)            # ground truth 最优选择
        
        # Step 4: 计算 sparse attention（可用完整 A_full 中对应 block 的结果）
        O = compute_attention_on_selected(Q, K, V, selected)
        
    return model_accuracy  # 这就是 sparse attention 的准确率上界
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Oracle sparsity 主要用于：(1) 验证 attention 是否本身具有稀疏性——即确定是否仅需一小部分 KV blocks 即可保持准确率；(2) 为稀疏预测方法（如 AttnGate）提供准确率上界参考。SeerAttention-R 的 oracle 实验显示：Qwen3-14B 在 AIME 上，block_size=64 时 2k token budget 达 lossless，验证了推理 attention 的内在稀疏性。稀疏预测方法（AttnGate）达到 4k budget 才能 lossless，与 oracle 的 2k budget 有 gap，反映了稀疏预测的近似误差。

涉及论文标题：
- SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

## Lazy Layers (懒惰层) / Lazy Ratio (懒惰比例)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Lazy Layers 是 LightTransfer 论文提出的概念：在长上下文 LLM 推理中，某些 Transformer 层的注意力主要集中在两类"语义不重要"的 token 上——(1) 初始几个 token（$X_{\text{initial}}$，即 attention sink），(2) 最近的 token（$X_{\text{recent}}$，即 sliding window 内的 token）。这种注意力模式被类比为"读论文只读摘要和结论"，称为"懒惰行为"（lazy behavior）。表现出这种行为的层称为 Lazy Layers。

懒惰比例（Lazy Ratio）$r_i$ 是量化第 i 层懒惰程度的指标：
$$r_i = \frac{1}{w_{\text{last}}} \sum_{\hat{x} \in X_{\text{last}}} \sum_{x \in \{X_{\text{initial}}, X_{\text{recent}}\}} A_i(\hat{x}, x)$$

其中 $A_i(\hat{x}, x)$ 是第 i 层所有 head 平均后的注意力权重，从 query token $\hat{x}$ 到 key token $x$。$w_{\text{last}}$ 是用于评估的最后几个 query token 数量。$r_i$ 越高，说明该层越多注意力集中在 sink + recent token 上。

关键发现：(1) 对于给定的输入 prompt，懒惰层行为在生成过程中跨 token 相对一致；(2) 不同 prompt 下懒惰层的 index 位置可能不同，因此需要 test-time 动态识别。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Lazy Ratio 计算（利用 FlashAttention LSE 避免重算完整 attention 矩阵）**：
```python
def lazy_ratio_calculation(q, k, v, w_last, w_sink, w_recent):
    # q, k, v: [bs, num_heads, seq_len, head_dim]
    attn_out, lse = flash_attn(q, k, v, causal=True, return_lse=True)
    # lse: [bs, num_heads, seq_len] -- log-sum-exp of attention scores

    q_last = q[:, :, -w_last:, :]   # 最后 w_last 个 query token
    k_comb = torch.cat([k[:, :, :w_sink, :],    # sink tokens
                         k[:, :, -w_recent:, :]], dim=2)  # recent tokens

    # O(w_last * (w_sink+w_recent)) 小矩阵乘法，常数复杂度
    log_lazy_ratio = torch.matmul(q_last, k_comb.transpose(-1, -2)).logsumexp(dim=-1) - lse
    return log_lazy_ratio  # 高值 → layer 懒惰
```

**LightTransfer-TEST 流程（Prefilling 阶段动态识别）**：
```
优先队列 Q (max-heap, 容量 P = 50% 总层数)

for layer i in 0..L-1:
    计算当前层的 full attention 并获取 KV cache
    计算 lazy ratio r_i
    将 (r_i, i) 加入 Q
    
    if Q 容量 > P:
        (r_max, lazy_layer) = Q.pop()  # 弹出 ratio 最高的层
        将 lazy_layer 的 KV cache 缩减为 {X[:w_sink], X[-w_recent:]}
```

术语一般如何实现？如何使用？

通过 FlashAttention 的 `return_lse=True` 参数获取 LSE 值作为注意力分布代理，避免 $O(n^2)$ 重计算。推荐超参数：$w_{\text{sink}}=4$, $w_{\text{recent}}=1020$, $w_{\text{last}}=32$。Lazy ratio 计算的额外开销极小（相对吞吐仅降低 0.0014-0.0058×），且序列越长开销占比越低（识别复杂度 O(1)）。LightTransfer-TEST 适用于输入足够长的任务（long-context understanding），LightTransfer-TRAIN 通过训练集预选懒惰层后 SFT 微调，适用于输入短但推理链长的任务（o1-like reasoning）。

涉及论文标题：
- LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation

## Streaming Attention with Sink Tokens (带注意力汇的流式注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Streaming Attention 是 StreamingLLM（Xiao et al., ICLR 2024）提出的注意力机制：将 full causal attention 替换为固定大小的注意力窗口，仅保留 (1) 前 $w_{\text{sink}}$ 个 attention sink token（因 softmax 归一化累积了大量注意力权重），(2) 最近 $w_{\text{recent}}$ 个 token（局部上下文窗口）。KV cache 大小从 $O(L)$（随序列长度线性增长）降为 $O(w_{\text{sink}} + w_{\text{recent}})$ = 常数，支持无限长上下文流式推理。

与 "Streaming Attention Heads"（DuoAttention/PruLong 的 head 级分类）的区别：streaming attention 是整层粒度的 attention 机制替换而非逐 head 分类。

从算法pipeline角度拆解术语。

```
# Full attention（第 n 步）: Q_n=[1,d], K_cache=[n,d], V_cache=[n,d]
S = Q_n @ K_cache^T / sqrt(d)    # [1, n] — 随 n 增长

# Streaming attention（固定 KV cache 大小）
K_stream = concat([K_cache[:w_sink], K_cache[-w_recent:]])  # [w_sink+w_recent, d]
V_stream = concat([V_cache[:w_sink], V_cache[-w_recent:]])
S_stream = Q_n @ K_stream^T / sqrt(d)  # [1, w_sink+w_recent] — 常数
A_stream = softmax(S_stream)
O_stream = A_stream @ V_stream
```

术语一般如何实现？如何使用？

StreamingLLM 将所有层替换为 streaming attention，典型配置 $w_{\text{sink}}=4$, $w_{\text{recent}}=1020$，窗口 ≈ 1024 tokens。全部替换导致 LongBench 平均下降 3.5-11.5%（全局信息捕获能力完全移除）。LightTransfer 的创新：仅在"懒惰层"使用 streaming attention，非懒惰层保留 full attention 作为全局信息锚点——50% 层替换时吞吐提升 2.17×，LongBench 仅下降 <1.5%。开源：https://github.com/mit-han-lab/streaming-llm。

涉及论文标题：
- LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation
- StreamingLLM: Efficient Streaming Language Models with Attention Sinks

## Hybrid Model Architecture (混合模型架构)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Hybrid Model Architecture 是将异构架构层混合在同一模型中的设计范式。在 LLM 中特指将标准 Transformer attention 层与内存高效替代层（Mamba/SSM、sliding/streaming attention、lightning attention）混合使用，平衡表达力与内存效率。已知实例：Jamba（Transformer + Mamba）、Gemma 2（sliding window + full attention 交替）、Minimax-01（lightning attention + full attention）、LightTransfer（streaming attention + full attention）。

从算法pipeline角度拆解术语。

**LightTransfer 的 Hybrid 转换（prefilling 阶段）**：
```
标准 Transformer → Hybrid Model:
  Layer_0: FullAttn  → 识别为 lazy → StreamingAttn (KV cache 缩减)
  Layer_1: FullAttn  → 非 lazy     → FullAttn (保留完整 KV cache)
  ...
  Layer_L: FullAttn  → 识别为 lazy → StreamingAttn (KV cache 缩减)
```

关键设计决策：Layer-wise（非 head-wise）hybrid。在 TP 下，head-wise hybrid 导致不同 GPU 的 KV cache 大小不一致产生同步瓶颈；layer-wise 保持同层内所有 head 一致，与 vLLM/SGLang 的 KV cache 粒度兼容。

术语一般如何实现？如何使用？

两种路径：(1) 从头训练（Jamba, Gemma 2——需大规模预训练）；(2) 从预训练 Transformer 转换（LightTransfer——~5K 训练样本或 zero-shot）。LightTransfer 仅修改 attention mask pattern（不改变权重），在 lazy 层丢弃 {sink + recent} 之外的 KV cache。

涉及论文标题：
- LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation

## LightTransfer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LightTransfer 是将预训练 Transformer 转换为 Hybrid 模型（full attention + streaming attention 层混合）的轻量框架（arxiv 2410.13846）。核心洞察：利用 LLM 不同层在长上下文推理中的功能差异——某些层是"懒惰层"（注意力集中在 sink + recent tokens），将其 full attention 替换为 streaming attention 以降低 KV cache 显存；非懒惰层保留 full attention 维持全局信息捕获。

两种模式：(1) LightTransfer-TEST：test-time 在线转换，prefilling 阶段利用 FlashAttention LSE 值计算 lazy ratio，优先队列动态识别懒惰层——无需任何训练；(2) LightTransfer-TRAIN：训练集统计懒惰层频率预选，然后 SFT 微调（~5K 样本，原用于 long-reasoning 蒸馏的数据）——适用于短输入长推理。

理论保证：Theorem 5.1 证明输出误差 ≤ 被移除 KV 对的注意力分数之和 × 常数，而 lazy ratio 算法恰好优化该上界的 greedy 版本。

从算法pipeline角度拆解术语。

**LightTransfer-TEST 算法**：
```
Q = PriorityQueue(maxsize = P_ratio * L)  # max-heap, key = lazy_ratio
for i in 0..L-1:
    O_i, lse_i = FlashAttention(LN(X_{i-1}), causal=True, return_lse=True)
    r_i = compute_lazy_ratio(lse_i, ...)
    Q.push((r_i, i))
    if Q.is_full():
        r_max, lazy = Q.pop()
        K_cache[lazy] = K_cache[lazy][:w_sink] + K_cache[lazy][-w_recent:]  # 缩减
        V_cache[lazy] = V_cache[lazy][:w_sink] + V_cache[lazy][-w_recent:]

# Decoding: 使用已缩减的 KV cache
```

术语一般如何实现？如何使用？

基于 PyTorch + HuggingFace Transformers + FlashAttention。开源：https://github.com/sail-sg/LightTrans，HuggingFace 模型：cxdu/QwQ-32B-LightTransfer。超参数：$w_{\text{sink}}=4$, $w_{\text{recent}}=1020$, $w_{\text{last}}=32$, 标准层保留比例 50%-75%。结果：50% 层替换时吞吐 2.17× (16K seqlen)，LongBench 下降 <1.5%，AIME24 达 53.3%（QwQ-STILL baseline 46.7%：+6.6%）。

涉及论文标题：
- LightTransfer: Your Long-Context LLM is Secretly a Hybrid Model with Effortless Adaptation

## Attention Allocation Pattern (注意力分配模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention Allocation Pattern（注意力分配模式）是 SnapKV 论文（Li et al., 2024）通过系统实验发现的一项关键观察：在 LLM 自回归生成过程中，每个 attention head 对 prompt 中各 token 的注意力分配表现出高度一致的规律——仅有少数 prompt token 是真正对回答生成"重要"的，且这些重要 token 的集合在生成过程开始之前就可以被识别。

SnapKV 通过两项实验验证了这一模式：(1) 将 prompt 末尾多个 window 的 queries 选出的 "重要 attention features"（高 attention weights 的 KV 位置）与生成阶段实际使用的重要 features 计算 overlap rate，发现 prompt 最后一个 window 与生成阶段的 overlap rate 最高（Fig. 2）；(2) 将生成过程分为多个 window，计算各 window 选出的重要 features 与 prompt 最后一个 window 选出的 overlap rate，发现 overlap rate 在生成全过程中保持高位（Fig. 3），说明模式稳定。

这一发现的核心含义是：**LLM 在生成之前就知道哪些 prompt tokens 对其回答至关重要**（LLMs know what you are looking for before generation），因此可以在 prefill 阶段完成 KV cache 压缩，而不需等待生成过程。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**注意力分配模式的分析流程**：

```
# 实验设置：利用 Ultrachat 数据集，筛选 response > 512, prompt > 3K
# 每个 window = 128 tokens

# === 实验1: 模式是否可在生成前识别 ===
for layer in model.layers:
    # 取 prompt 最后 20 个 window 的 queries
    for w in last_20_windows:
        Q_w = Q[layer, w_start:w_end, :]     # [128, D]
        scores = Q_w @ K_prefix^T / sqrt(D)  # [128, L_prefix]
        avg_weights = scores.mean(dim=0)      # [L_prefix] 平均注意力权重
        important_w = avg_weights > threshold  # 标记的重要 features

    # 实际生成中使用的重要 features
    for gen_step in generation:
        Q_gen = Q[layer, gen_step, :]
        scores_gen = Q_gen @ K_prefix^T / sqrt(D)
        important_gen = scores_gen > threshold

    # 计算 overlap rate
    overlap_rate = |important_w ∩ important_gen| / |important_gen|

# 结果：最后一个 window 的 overlap rate 最高 → 模式可在生成前识别

# === 实验2: 模式是否在生成中保持稳定 ===
# 取 prompt 最后一个 window 的重要 features
important_last_window = get_important(Q_last_window, K_prefix)

# 将生成过程分为 4 个 window，每 window 128 tokens
for gen_window in [1..4]:
    important_gen_w = get_important(Q_gen_window, K_prefix)
    overlap = |important_last_window ∩ important_gen_w| / |important_gen_w|
    # 结果：overlap rate 在所有 window 中保持高位 → 模式稳定
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

注意力分配模式的具体实现基于对 attention weights 的分析。关键步骤：(1) 将 prompt 划分为 prefix 和 observation window（末尾部分）；(2) 用 observation window 内的 queries 计算对所有 prefix keys 的注意力权重；(3) 沿 query 维度聚合得到每个 prefix token 的"重要性投票分数"；(4) 通过 TopK 选出得分最高的 token 位置。这一模式被 SnapKV 用于驱动 KV cache 压缩——仅保留被选中的 prefix KV pairs 和完整的 observation window KV pairs。

该模式的适用条件：(a) 要求模型具备长上下文理解能力；(b) prompt 末尾的指令/问题能有效驱动对不同 prefix 区域的差异化注意力；(c) 不同指令下注意力模式会变化（Fig. 4），因此需要动态识别而非使用静态重要位置。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation

## Hit Rate in KV Cache Context (KV Cache 上下文中的命中率)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hit Rate（命中率，记作 H）是 SnapKV 论文专门设计的一个量化指标，用于评估 observation window-based voting 机制在识别重要 attention features 方面的有效性。它衡量的是：通过 observation window 投票选出的"重要"attention features 中，有多大比例在后续生成阶段确实保持了高 attention weights。

形式化定义（Eq. 4-8）：给定注意力阈值 θ，将生成阶段 attention weights 超过 θ 的 prefix 位置标记为"实际重要的"（M_threshold_cur），将 observation window 投票选出的位置标记为"预测重要的"（M_vote_obs）。Hit Rate 是两者的交集大小与实际重要位置总数的比值：H = |M_threshold_cur ∩ M_vote_obs| / |M_threshold_cur|。H ∈ [0, 1]，越接近 1 表示投票机制越准确。

SnapKV 使用 hit rate 进行了两项鲁棒性分析：(a) Contextual Dependency——不同指令在相同文档上选出的重要特征差异较大（hit rate 下降），证明 KV 压缩需要 context-aware 策略；(b) Invariance to Instruction Positions——无论指令在 prompt 开头还是末尾，hit rate 均保持高位，证明 SnapKV 的 observation window 机制对指令位置鲁棒。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Hit Rate 计算流程（per head, per layer）
# 输入: A_cur ∈ R^{L_prefix}       当前生成 query 对 prefix keys 的 attention weights
#       M_vote_obs ∈ {0,1}^{L_prefix} observation window 投票选出的位置掩码
#       θ                           attention 阈值

def compute_hit_rate(A_cur, M_vote_obs, theta):
    # Step 1: 标记当前生成中"实际重要的"features
    M_threshold_cur = (A_cur > theta).float()  # {0,1}^{L_prefix}

    # Step 2: 计算命中(交集)
    O = M_threshold_cur * M_vote_obs  # 逐元素与, {0,1}^{L_prefix}

    # Step 3: 计算命中率
    H = O.sum() / (M_threshold_cur.sum() + eps)

    return H  # ∈ [0, 1]

# SnapKV 论文中用于鲁棒性分析的变体：
# H(M_vote_A, M_vote_B) — 两组不同投票结果的命中率
# 用于衡量不同 instruction-response pairs 在同一文档上的重要特征一致性
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Hit Rate 主要作为分析工具而非运行时指标——在运行时不需要计算 hit rate（因为生成阶段的实际 attention weights 此时未知）。其用途包括：(1) 验证观察窗口大小选择的合理性；(2) 比较不同 voting strategy 的预测质量；(3) 分析不同数据集、不同指令类型对注意力模式的影响。实现上仅在离线分析/消融实验中使用，不产生运行时开销。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation

## Induction Heads (归纳头)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Induction Heads（归纳头）是 Transformer 模型中一类特殊的功能性 attention head，最早由 Olsson et al. (2022, "In-Context Learning and Induction Heads") 系统描述。Induction heads 的核心行为是 "copy-paste"：当序列中出现模式 [A][B]...[A] 时，induction head 会在最后一个 [A] 处将高 attention weight 分配给紧随前一个 [A] 出现的 token [B]，从而"预测" [B] 应该在此处重复出现。

其底层机制涉及 K-Composition：前一层的 attention head 将 token [B] 的信息写入 token [A] 的残差流中（通过 "previous token head"），当前层的 induction head 通过 Query 匹配 [A] 的 Key，使 [A] 的 Value（包含 [B] 的信息）被高权重读出。这实现了跨位置的模式匹配和复制。

SnapKV 论文在解释为何需要 pooling 聚类时引用了 induction heads（Sec. 4.3）：LLM 的信息检索和生成不仅依赖高 attention weight 的特征本身，还依赖 induction heads 将 attention weight 高的特征周围 token 的上下文信息一并"复制"到输出中。若仅保留孤立的 top attention 位置（不保留周围 token），会破坏 induction heads 的复制机制——例如在电话号码检索中，模型可能仅获取了国家代码却"补全"了错误的其余数字。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Induction Head 的工作机制（简化）
# 模式: 输入序列 "...A B ... A" → 预测 "B"

# Layer L-1 (Previous Token Head):
#   在位置 i 处，将位置 i 的 token 信息写入位置 i+1 的残差流
#   即 token A 的残差流中包含 B 的信息

# Layer L (Induction Head):
#   当前 token 是第二个 "A"
#   Q_A = W_Q @ residual_A  # Query 来自 A
#   K_prefix = W_K @ residual_all  # Key 来自所有位置的残差流
#   scores = Q_A @ K_prefix^T
#   # induction head 给第一个 "A" 的位置高分
#   # → 第一个 "A" 的 Value 中包含 B 的信息
#   # → 输出中复制 B 的信息 → 预测下一个 token 为 B

# SnapKV 的 pooling 设计与 induction heads 的关系:
# 仅 TopK 选择（无 pooling）:
important_positions = TopK(attention_scores, k)  # {100, 200, 350}
# → 位置 99, 101, 199, 201 等相邻 token 的 KV 被丢弃
# → induction heads 无法从 100 的上下文中复制完整信息

# Pooling 聚类（保留邻域）:
pooled_scores = MaxPool1d(attention_scores, kernel_size=5)
important_positions = TopK(pooled_scores, k)  # {99, 100, 101, 199, 200, 201}
# → 相邻 token 被集群保留
# → induction heads 可以正确复制完整上下文
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Induction heads 不是人工设计的模块，而是在 Transformer 预训练过程中自发涌现的功能性电路（circuits）。它们通常在多层 Transformer 中由两层 attention head 组合形成（K-Composition pattern）。在 mechanistically interpretable 的研究中，可通过 activation patching、attention pattern analysis 和 knock-out experiments 来识别。

对于 KV cache 压缩方法设计，induction heads 的存在意味着：(a) 压缩时不能仅保留孤立的高注意力 token，需要保留其邻域 token 以维持复制机制；(b) pooling/clustering 策略（如 SnapKV 的 max pooling 或 PyramidKV 的区块 chunk）是必要的设计选择。实践中 SnapKV 通过 1D max pooling（kernel_size=5~13）实现，PyramidKV 将序列分为固定大小 chunks 并保留完整 chunks。

涉及论文标题：
- SnapKV: LLM Knows What You are Looking for Before Generation

---

## Speculative Prefill (SPECPREFILL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Speculative Prefill 是一种 training-free 的 LLM prefill 加速框架。核心思想：利用同系列中较小的"推测器模型"（speculator，如 8B）估计 prompt 中各 token 对主模型（如 70B/405B）的重要性，仅将筛选出的 token 子集（保留原 position IDs）送入主模型 prefill，跳过其余 token 的 attention + MLP 计算。加速比近似正比于 token 丢弃率（实测 405B 模型 10% 保持率下达 7.66× TTFT 加速、7× QPS 提升）。与 speculative decoding 天然兼容：speculator 同时服务 prefill token 选择和 decode draft proposal。

从算法pipeline角度拆解术语：

```
// Speculative Prefill Pipeline
Input: Base model M, Speculator S, prompt P
// Phase 1: Look-ahead (N=8 steps)
for i = 1 to N:
    Q_i, K_i, V_i = S.forward(P, store_q=True)
    P.append(argmax(S.lm_head(Q_last)))

// Phase 2: Attention score aggregation
A = compute_attention(Q_saved, K_saved)  // [N, L, S, H]
score = mean_over_N(max_over_L_H(A))     // → [S]

// Phase 3: Chunk selection
score_smoothed = AvgPool1D(score)
chunks = split_into_chunks(score_smoothed)
selected = TopK(chunk_avg(chunks))
T = tokens_in(selected)

// Phase 4: Main model forward (selected tokens only, with original pos IDs)
output = M.forward(T, original_positions[T])
```

术语一般如何实现？如何使用？

基于 vLLM 0.6.3.post1 monkey patch 实现。需要同模型家族的 speculator（如 Llama-3.1-8B → 70B/405B）。适用于长上下文可压缩 prompts；不适用信息密集短 prompts。开源代码：https://github.com/anonymous/speculative_prefill。

涉及论文标题：
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

---

## Token Importance Estimation via Cross-Model Attention Transfer

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Token Importance Estimation via Cross-Model Attention Transfer 是 SPECPREFILL 的核心机制假设：同一模型家族中不同规模的模型（如 Llama-3.1-8B 和 405B），其对 prompt token 重要性的注意力分布具有可迁移性。因此可用小型 speculator 的注意力分数作为 proxy 估计哪些 token 对大型 base model 重要，无需对大模型做任何额外 forward pass 或训练。

从算法pipeline角度拆解术语：

给定 prompt 长度 M，speculator 层数 L，头数 H，look-ahead N 步：
1. 对每个 prompt token i，第 j 步解码产生的注意力：a_{ij} = Softmax(Q_{M+j} K^T)_i
2. 聚合：importance(i) = (1/N) Σ_j max_{l} max_{h} a_{ij}^{(l,h)}
3. 基于 importance 选择 Top-K chunks（而非 Top-K tokens，利用邻近 token 重要性相关）

术语一般如何实现？如何使用？

实现要求 speculator 和 base model 同 tokenizer、同家族。speculator FLOPs 仅为主模型 14.24%（70B）或 2.96%（405B）。适用场景：prompt 含冗余 token 的长上下文任务；不适用信息密集短 prompts（如数学题）。

涉及论文标题：
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

---

## Proximity Bias in Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Proximity Bias 是 LLM 自注意力中的偏差现象：在生成新 token 时，prompt 中位置更靠近末尾（更接近当前解码 token）的 token 倾向于获得更高注意力分数，即使这些 token 在语义上不如某些远端 token 重要。与 Attention Sink（首 token 偏爱）共同构成使用原始注意力分数估计 token 重要性的两个主要偏差源。

从算法pipeline角度拆解术语：

在 token dropping 场景中，仅依赖最后 token 的注意力会导致偏向选择末尾 token 而忽略前段重要信息。SPECPREFILL 通过两种策略缓解：
(1) Look-ahead decoding — 解码 N 步后聚合多个位置的注意力，削弱单一位置偏差
(2) Max-mean aggregation — max over layers/heads 让被任意层关注的 token 浮现，mean over look-ahead steps 公平对待各步

术语一般如何实现？如何使用？

除 look-ahead 和聚合策略外，邻近 token 重要性相关的观察（Concurrent work CritiPrefill/Lv et al. 也发现此现象）促使 chunk-based selection 作为额外去噪手段。SPECPREFILL 的消融实验（Figure 2, 8）验证这些策略对短上下文任务的提升比长上下文更显著。

涉及论文标题：
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

---

## Prompt Compressibility Types

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Prompt Compressibility 是 SPECPREFILL 通过实验发现的 prompt 对 token dropping 的不同响应模式分类。queries 分为三类：(1) Information-dense — 信息密集短 prompts，token dropping 效果差；(2) Compressible — 含大量冗余，删除大部分 token 后质量不下降；(3) Noisy — 删除部分噪声 token 后性能反而提升。

从算法pipeline角度拆解：

通过质量-保持率曲线分类：
```
compressible: quality(10%) ≈ quality(100%)     // 质量稳定
info_dense:   quality(10%) << quality(100%)    // 显著下降
noisy:        quality(50%) > quality(100%)     // 先升后降
```

术语一般如何实现？如何使用？

论文未给出自动分类算法（列为 future work）。实践中可让用户根据延迟/质量权衡决定保持率，或开发自适应策略动态调整。固定保持率已在 LongBench 多数类别中有效。

涉及论文标题：
- Speculative Prefill: Turbocharging TTFT with Lightweight and Training-Free Token Importance Estimation

---

## Dense Preference Score (密集偏好分数 / Layer Classification Metric for KV Cache Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dense Preference Score (P) 是 TailorKV 提出的离线度量指标，用于判断 Transformer 每一层注意力应该采用量化还是稀疏选择的压缩策略。核心公式：使用最近的 n_q 个 query 向量 Q_{last_q} ∈ R^{n_q × d_h} 与全部 key 向量 K ∈ R^{n × d_h} 计算完整 attention score 矩阵，取每行 Top-k attention scores 之和的补数。即 P = n_q - Σ_{(i,j)∈Î} Â_{i,j}，其中 Î 是每 query 行的 Top-k 位置集合。P 值高 → 注意力分布均匀（密集） → quantization-friendly；P 值低 → 注意力集中在少量 token → sparsity-friendly。阈值 τ=0.2 通过 synthetic LongBench 实验确定，该 metric 跨数据集一致（同一模型的 P 分布在 different datasets 下几乎相同），因此可离线一次标定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// === 离线标定阶段（仅执行一次） ===
// 输入：校准数据集 prompt，模型权重
// 超参：n_q（最近query数），k（top-k数），τ（阈值=0.2）

for each Transformer layer l in 0..L-1:
    // Step 1: 收集 prefilling 阶段的 attention 信息
    Q_last_q = recent_n_q_query_vectors(l)    // shape: (n_q, d_h)
    K_all = all_key_vectors(l)                 // shape: (n, d_h)

    // Step 2: 计算完整 attention score
    A_hat = Softmax(Q_last_q @ K_all.T / sqrt(d_h))  // shape: (n_q, n)

    // Step 3: 计算 Dense Preference Score
    for i in 1..n_q:
        topk_vals[i] = Top_k(A_hat[i, :], k)
    P_l = n_q - sum(topk_vals)                 // Eq.(8)

    // Step 4: 层分类
    if P_l > τ:
        layer_type[l] = "Quantization-Friendly"   // 浅层（layer 0, 有时 layer 1）
    else:
        layer_type[l] = "Sparsity-Friendly"       // 深层

// 结果示例（Llama-3.1-8B, L=32）:
// Q = {0} → 仅 layer 0 是 quantization-friendly
// Llama-2-7B / Yi-6B / Yi-9B: Q = {0, 1} → layer 0 和 1 是 quantization-friendly
```

直觉解释：P 捕捉了"有多少 attention mass 不在 top-k 中"——密集层中 attention 分散，top-k 只覆盖少量 mass → P 大；稀疏层中 attention 集中在 few tokens → P 小。

术语一般如何实现？如何使用？

实现：(1) 使用校准数据集（如 synthetic LongBench 的一个子集）运行一次完整 prefill → 记录每层最近 n_q 个 query 和全部 key → 计算 P；(2) τ 通过 grid search 在 LongBench 验证集上确定（TailorKV 发现 τ=0.2 对所有模型通用）；(3) 离线标定结果（即每层的类型 label）在 serving 时作为静态配置使用，不需要在线重新计算。

适用场景：任何需要对 Transformer 层进行差异化 KV cache 压缩策略的方法（不仅限于量化+稀疏，也可扩展到不同剪枝率、不同卸载策略等）。与 PyramidKV 的"金字塔信息漏斗"假设互补——PyramidKV 假设信息从浅层向深层集中（所有层用同一策略只是不同预算），TailorKV 发现浅层和深层需要根本不同的策略（浅层适合保留全部信息的量化、深层适合只保留关键 token 的稀疏）。

涉及论文标题：
- TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization

---

## Critical Channel-Driven Token Retrieval (关键通道驱动的Token检索)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Critical Channel-Driven Token Retrieval 是 TailorKV 提出的动态 token 选择技术，利用 query 和 key 中 outlier channel（值显著大于其他 channel 的 hidden dimension）来近似 attention score，从而精准选择需要从 CPU 取回 GPU 的 Top-K 个 KV token。核心洞察：(1) attention score = q·K^T 中，每个 channel 对 attention 的贡献为 |q_i| · |K_i|；(2) 少数 channel 在 query 和 key 中呈现大幅度值（outlier）、主导 attention 计算；(3) 仅用这些 critical channels 的 query/key 子集计算近似 attention，便可高精度识别最重要的 token。通道数 d_s = 8（LongBench）/ 12（InfiniteBench/RULER），覆盖的 token 检索准确率接近完整 attention 检索。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
// === Decoding 阶段，sparsity-friendly 层 l ===

// Stage 1（在 layer l-1 执行，用于预取）:
h_{l-1} = hidden_state_after_layer_l_minus_1        // shape: (1, d)
q_hat = W_q[l] @ h_{l-1}                             // inter-layer query 预估

// 计算 channel 重要性
for i in 1..d_h:
    s_i = |q_hat_i| * max(|K_{CPU}[i, :]|)           // Eq.(10)

critical_ch = Top_indices(s, d_s)                     // 选 d_s 个 critical channels
Prefetch_async(K_{CPU}[critical_ch, :])               // 从 CPU 异步预取 critical key

// Stage 2（在 layer l 执行）:
q = W_q[l] @ h_l                                     // 真实 query
q_crit = q[critical_ch]                              // d_s 维
K_crit = K_prefetched[critical_ch, :]                // (d_s, n)，已预取完成

// 近似 attention scores（用 d_s 而非 d_h 维计算）
a_approx = q_crit @ K_crit.T / sqrt(d_s)             // shape: (1, n)
topk_idx = TopK(a_approx, k=n_topk)                   // 选 Top-K

// 从 CPU 取完整 KV
Fetch_sync(K_full[topk_idx], V_full[topk_idx])        // 唯一不可 overlap 的操作

// 合并 local + fetched tokens 做完整 FlashAttention
output = FlashAttn(q, cat(K_local, K_fetched), cat(V_local, V_fetched))
```

为什么 dynamic channel selection 优于 static：TailorKV 实验（Figure 9b）显示 query/key 的 outlier 位置不是固定的，它们可能出现在任何 channel（Figure 2 Bottom），因此离线标定的 static channel set 召回率低于运行时动态选择。

术语一般如何实现？如何使用？

实现要点：(1) `max(|K_i|)` 在 prefill 后计算一次并存储在 CPU 元数据中（不随 decoding 变化），仅在每次有新 token 加入时更新；(2) Stage 1 的 q_hat 利用 inter-layer 相似性（余弦相似度 >0.99 between adjacent hidden states, Appendix B Figure 11）提前一层预估，使 critical key 预取可以与 layer l-1 的计算重叠；(3) Stage 2 的 attention 近似使用 d_s 维，计算量仅为完整 attention 的 d_s/d_h（约 8/128 = 6.25%）；(4) Double buffering——GPU 上有两个 buffer（读/写），一边写入 layer l 的 prefetch 数据、一边读取 layer l-1 预取好的数据。

该技术与 ANN-based token retrieval（如 Faiss/LSH）的区别：不需要额外 CPU 端索引构建和检索计算，所有选择逻辑在 GPU 上完成（仅 critical channel 的 K 从 CPU→GPU，然后 GPU 上近似 attention+排序），避免了 PQCache/MagicPIG 的 CPU 端 K-Means clustering 或 LSH hashing 开销。

涉及论文标题：
- TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization

---

## CodeBook-Based KV Cache Compression (基于码本的 KV Cache 压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

CodeBook-Based KV Cache Compression 是一种利用向量量化 (Vector Quantization) 方法压缩 KV Cache 的技术。核心思想：将 KV Cache 中高度相似的 Key/Value 向量聚类为少数"码本条目" (codebook entries)，仅存储码本 ($C_K$, $C_V$)、每个 token 到码本条目的索引 ($r_K$, $r_V$，整数类型) 和每个 token 的 L2 magnitude ($m_K$, $m_V$，浮点类型)，推理时通过查表+缩放重建原始向量：$\Gamma_r = C_\Gamma[r_\Gamma] \otimes m_\Gamma$。

SpindleKV 首次在 KV Cache 压缩中提出基于余弦相似度的贪心码本构建方法。与传统 VQ 需要离线训练不同，SpindleKV 的码本是在 prefill 阶段 Just-in-Time (JIT) 在线构建的，无需额外训练。构建过程：(1) 对保留的 KV cache 归一化（除以 L2 magnitude）；(2) 计算 token 间余弦相似度矩阵 $S_\Gamma$；(3) 设定阈值 $\theta_\Gamma$ 构建邻接矩阵 $G_\Gamma = \text{where}(S_\Gamma > \theta_\Gamma, 1, 0)$；(4) 贪心迭代：每次选图中度数最高节点加入码本，将其邻居映射到该码本条目，从图中移除已覆盖节点；(5) 记录每个 token 的 L2 magnitude 用于重建。

这项技术专门针对 KV Cache 在浅层中的"构成性冗余"——浅层 token 之间 KV 向量余弦相似度极高（超过 0.9），但这些 token 各自都获得较高注意力分数，传统 eviction 方法无法有效压缩。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**SpindleKV 码本构建伪代码（Algorithm 1）**：

```
Input:  归一化后的 KV Cache Γ_r [l_c, h, d_h]
        Key 阈值 θ_K = 0.98, Value 阈值 θ_V = 0.95
Output: CodeBook C_Γ, 引用索引 r_Γ, Magnitudes m_Γ

C_Γ = []                              # 空码本
r_Γ = [-1, -1, ..., -1]              # 每个 token 的引用，-1 表示未分配
m_Γ = L2_Norm(Γ, dim=-1)              # 记录原始 magnitude
Γ_r = Γ_r / m_Γ                       # 归一化到单位向量

S_Γ = cos_sim(Γ_r, Γ_r)              # [N, N] 余弦相似度矩阵
G_Γ = where(S_Γ > θ_Γ, 1, 0)         # 邻接矩阵：相似度 > 阈值 → 有边

while G_Γ != 0:                      # 直到所有 token 都被覆盖
    s_Γ = sum(G_Γ, dim=1)            # 每个节点的度数
    ι = argmax(s_Γ)                  # 选度数最高的 token
    C_Γ.append(Γ_r[ι])               # 加入码本
    η_ι = argwhere(G_Γ[ι] == 1)      # 邻居
    r_Γ[η_ι] = len(C_Γ) - 1          # 引用指向码本
    mask_Γ = matmul(¬G_Γ[ι]^T, ¬G_Γ[ι])
    G_Γ = G_Γ & mask_Γ               # 移除已覆盖节点

# 推理时重建
Γ_reconstructed = C_Γ[r_Γ] * m_Γ    # 查码本 × 恢复 magnitude
# 对重建后的 K 重新应用 RoPE
```

**KV Cache 最终存储空间计算**：

$$r^\lambda = r_1^\lambda \times r_2^\lambda \times r_3^\lambda$$

其中 $r_1^\lambda$ 是 eviction 保留率，$r_2^\lambda = |C_K^\lambda \cup C_V^\lambda| / \sum(|K_{j,r}^\lambda| + |V_{j,r}^\lambda|)$ 是码本压缩率，$r_3^\lambda$ 是 dtype 转换率（索引用 int、magnitude 用 float 替代完整 FP16 向量）。

术语一般如何实现？如何使用？

SpindleKV 开源实现见 https://github.com/tyxqc/SpindleKV。码本构建仅在 prefill 阶段执行一次。超参数：Key 阈值 $\theta_K = 0.98$，Value 阈值 $\theta_V = 0.95$。实验中仅用码本（无 eviction）即可压缩 50% KV Cache 而准确率无损，验证了浅层构成性冗余假说。该方法与 eviction 方法互补——eviction 处理深层注意力稀疏性，码本处理浅层向量相似性。

GQA 兼容性：对 GQA 模型，SpindleKV 先将 KV head 展开（repeat $h_n$ 次）再构建码本。Expand 引入的重复向量余弦相似度为 1，极容易被码本合并，额外开销被消除。

涉及论文标题：
- SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

---

## Pyramid-Shaped KV Cache Allocation (金字塔形 KV Cache 分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Pyramid-Shaped KV Cache Allocation 是一种层间（跨层）KV Cache 预算分配策略。核心思想：浅层 Transformer 层的注意力分布更均匀（不稀疏），深层注意力则高度集中在少数 token 上（注意力稀疏性）。因此，在不同层之间统一保留相同数量 token 的 KV Cache 是次优的——应为浅层分配更多 KV Cache 预算、为深层分配更少，形成"金字塔形"分配。PyramidInfer (Yang et al., 2024) 和 PyramidKV (Cai et al., 2024) 率先提出此策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**金字塔形 KV Cache 分配的数学定义**：

全局参数：$r$ 为总 KV cache 保留率，$l$ 为序列总长度，$l_w$ 为观察窗口长度，$l_c = l - l_w$ 为上下文中 eviction 候选 token 数。

上下文的保留率：
$$r_c = \frac{r \cdot l - l_w}{l_c}$$

设 $\beta = 0.05$（最小保留率），$\alpha = \frac{1}{2}(1+\beta) = 0.525$。层 0 和层 $m-1$ 的保留率：
$$r_c(0) = \begin{cases} 2 \times r_c - 0.05, & \beta < r_c \le \alpha \\ 1, & \alpha < r_c \le 1 \end{cases}$$

$$r_c(m-1) = \begin{cases} 0.05, & \beta < r_c \le \alpha \\ 1 - 2 \times r_c, & \alpha < r_c \le 1 \end{cases}$$

第 $\lambda$ 层保留率（线性插值）：
$$r_c(\lambda) = r_c(0) + \frac{r_c(m-1) - r_c(0)}{m-1} \cdot \lambda$$

**具体例子**（$m=32$, $l=4096$, $l_w=128$, $r=0.40$）：
```
r_c = (0.40 × 4096 - 128) / (4096 - 128) = 0.381
r_c(0) = 2 × 0.381 - 0.05 = 0.712      # Layer 0 保留 71.2%
r_c(31) = 0.05                            # Layer 31 保留 5.0%
r_c(15) = 0.712 + (0.05-0.712)/31 × 15 = 0.392
```

推理时每层执行：
```
ac_i = accumulated_attention_scores(Q, K, l_w)
k = floor(r_c(λ) × l_c)
η_i = argTopK(ac_i, k)
Γ_{r,i} = Γ_i[η_i]
```

术语一般如何实现？如何使用？

PyramidKV 开源见 https://github.com/Linking-ai/PyramidKV，PyramidInfer 见 https://github.com/mutonix/pyramidinfer。主要区别：PyramidInfer 的 eviction 是逐层的——layer i 被 evict 的 token 在 layer i+1 也不重新计算；PyramidKV 的 KV cache 在不同层独立管理。SpindleKV 沿用了 PyramidKV 方式，并在其基础上增加浅层码本压缩。

涉及论文标题：
- SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

---

## Constituent Redundancy in KV Cache (KV Cache 构成性冗余)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Constituent Redundancy (构成性冗余) 是 SpindleKV 首次识别并提出的一种 KV Cache 冗余类型。与传统的"token 间冗余"（不同 token 注意力贡献不同，低贡献的可被 evict）不同，构成性冗余描述的是：在 Transformer 浅层中，不同 token 的 Key/Value 向量之间存在极高的余弦相似度——这些向量的"构成成分"高度重叠，可被分解为一组有限"基础向量"的线性组合。SpindleKV 实验发现 LLaMA2-7B-chat 浅层中大量 token 对的 KV 余弦相似度超过 0.9（Key），而在深层这种相似性急剧下降。

从算法pipeline角度拆解术语：

**构成性冗余 vs. 注意力稀疏性**：

| 维度 | 浅层 (0-10) | 深层 (20-31) |
|------|------------|-------------|
| 注意力稀疏性 | 低（分布均匀） | 高（集中在少数 token） |
| KV 余弦相似度 | 高（>0.9） | 低 |
| 适合的压缩方法 | CodeBook replacement | Token eviction |

产生原因：浅层 token 经历较少的 Transformer 编码迭代，上下文信息整合有限，KV 向量仍是相对"原始"的基础表示。深层 token 经多次 self-attention + FFN 变换后被上下文信息"分化"，相似度下降。

术语一般如何实现？如何使用？

利用构成性冗余进行压缩的实现是 CodeBook-Based KV Cache Compression。实验验证：仅用码本压缩（无 eviction）在 LLaMA2-7B-chat 上 50% KV Cache 保留率下 LongBench 准确率无下降。构成性冗余的发现区分了 SpindleKV 与纯 eviction 方法的关键洞察：eviction 仅对深层有效，码本压缩弥补了浅层不足，两者结合使 SpindleKV 在所有保留率下均优于 PyramidKV/PyramidInfer。

涉及论文标题：
- SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers

## 补充：ShadowKV 的 Online Pre-RoPE Key SVD 低秩分解

ShadowKV 文献 Low-Rank Decomposition for KV Cache Compression 增加了 key cache 在线 SVD 分解的新方式——与传统方法对权重矩阵做离线分解不同，ShadowKV 对 **pre-RoPE key cache**（而非权重 $W_k$）做在线、prompt-dependent 的 SVD 截断分解。

具体发现：pre-RoPE keys 的奇异值衰减最快（比 post-RoPE keys、values、权重矩阵都要低秩），同一序列内 key 的低秩子空间高度共享（内序列相似度 ~0.8-1.0），不同序列间低秩子空间不同（跨序列相似度 ~0.2-0.4）。因此，对 pre-RoPE key cache 直接逐序列做 SVD（rank r=160 for d=128）比 data-independent 的 weight decomposition 更精准，实现 6× 压缩而无精度损失。

```
// ShadowKV Pre-RoPE Key SVD（online, per-sequence）
K = X @ W_k^T                    // pre-RoPE key, shape [s, d]
A, B = SVD(K, rank=r)            // A: [s, r], B: [h_kv, r, d]
// 低秩存储替代完整 K：
// GPU 存储: A [s, r] + B [h_kv, r, d]
// 解码时按需重建: K_selected = Gather(A, I) @ B → [k*c, d]
// 重建仅针对选中的 top-k chunk（~1.56% tokens）
```

ShadowKV 的低秩分解在 pre-filling 阶段完成，SVD 开销占比随序列长度递减（64K: 6.65%, 128K: 3.25%, 256K: 1.75%, 512K: 0.97%），因为 attention 计算为 $O(S^2d)$ 而 SVD 为 $O(Sdr)$。

涉及论文标题：
- ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference
- xKV: Cross-Layer SVD for KV-Cache Compression

**xKV 的扩展——从单层到跨层在线 SVD**：xKV 将 ShadowKV 的单层 pre-RoPE key SVD 扩展到**跨层**维度。与 ShadowKV 对每层独立做 SVD（Single SVD）不同，xKV 将多个相邻层的 pre-RoPE KV-Cache 水平拼接后做一次统一的跨层 SVD：concat([K_ℓ1, ..., K_ℓ{∣G∣}]) = U S V^T，提取跨层共享的左奇异向量作为共享基。xKV 同样在 prefill 阶段按请求在线执行，SVD 开销在 128K context 下 <10% prefill time。跨层 SVD 的关键优势：由于层间主导奇异向量高度对齐（由 CKA 验证），共享基比每层独立 SVD 更高效——相同压缩比下跨层 SVD 保留更多信息，相同 rank 下跨层 SVD 压缩率更高（≈ G× vs per-layer SVD）。xKV 还对 keys 和 values 分配不同 rank ratio（1:1.5），并对 pre-RoPE states 分解后重新施加 RoPE。

## 补充：ShadowKV 的 Landmark-based Chunk Sparse Attention

ShadowKV 为 Chunk Sparsity of Attention 增加了 Landmark-based 的 chunk 近似方法。与 InfiniteHiP 的 per-chunk 代表 token 选择不同，ShadowKV 使用 chunk 均值作为 compressed landmark，并通过 cosine similarity 检测 outlier chunk。

```
// ShadowKV Landmark 构建（pre-filling）
K_RoPE = RoPE(K)                     // post-RoPE key
C = Reduce(K_RoPE, chunk_size=c)     // chunk mean landmarks [h_kv, s/c, d]
S = CosineSimilarity(C, K_RoPE)      // 每 chunk 内 cosine similarity
I_outlier = ArgTopK(-Min(S, dim=-1), o) // 最差近似的 o 个 chunk
L = C \ Gather(C, I_outlier)         // 非 outlier landmarks 保留 GPU

// ShadowKV Landmark 解码查询
P = Q @ L^T                          // 近似 attention scores [h_q, 1, n_c]
S = Softmax(P / sqrt(d))
S_agg = max_kv_group(sum(S, dim=-2)) // GQA 聚合到 KV heads
I = ArgTopK(S_agg, k)                // 选择 top-k chunk
```

与 InfiniteHiP chunk sparsity 对比：ShadowKV 的 landmark 是固定均值（无需 per-query 重选代表），但通过 static outlier cache 弥补均值近似的误差。Outlier 仅占 0.2-0.3% 的 chunk，保留其完整 KV 对在 GPU 保证精度。

涉及论文标题：
- ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

## Star Attention (Two-Phase Block-Sparse Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Star Attention 是一种两阶段 block-sparse 注意力算法（Acharya et al., ICML 2025），专为 Transformer 大语言模型的长序列推理优化设计。它基于一个关键观察：LLM 推理通常包含 (1) prompt encoding 阶段和 (2) token generation 阶段，而许多长上下文任务中，context token 只需 local context、query token 需要 global context。利用这一观察，Star Attention 将注意力计算分解为两阶段：

- **阶段一（Context Encoding）**：将长 context 切分为连续性 blocks，分发到多个 hosts 并行处理。每个 block（除第一个外）前缀拼接 anchor block（第一个 context block），对 2b-token augmented block 做 blockwise-local self-attention。此阶段无跨 host 通信，attention 复杂度 O(L·b) vs full attention O(L²)。多 hosts 完全 embarrassingly parallel。

- **阶段二（Query Encoding & Token Generation）**：Query 被广播到所有 hosts，各 host 使用 Flash Attention 计算 local attention A_h 和 softmax sum s_h，query-host 通过 gather + online softmax（log-sum-exp trick）聚合为 global attention A_global。每 token 仅通信 O(d) 数据（scalar + vector），与 context 长度无关。仅 query-host 更新 KV cache。

Star Attention 兼容几乎所有使用 global attention 训练的 Transformer LLM，无需 fine-tuning。在 RULER benchmark 上，Star Attention 实现 Ring Attention 的 1.1-16.9× 加速，同时保持 97-100% 准确率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**阶段一：Context Encoding 伪代码**：
```
Require: Context c, Block size b
L ← length(c)
Split c into n = ⌈L/b⌉ blocks: c = [c1, c2, ..., cn]
for i = 2 to n:
    c'_i ← concat(c1, ci)          // prefix anchor block to each context block
end for
Distribute [c'_1, c'_2, ..., c'_n] across H hosts
for each host h concurrently:
    for each assigned block c'_i:
        attn_out = self_attention(c'_i)  // FlashAttention over 2b tokens
        KV = generate_kv_cache(attn_out)
        discard KV for anchor block portion (c1)
        retain KV for ci portion → append to kv_h
    end for
end for
```

**阶段二：Query Encoding & Token Generation 伪代码**：
```
Require: Query q, num output tokens n_o, KV cache kv_h for all hosts
Designate query-host h_q
Broadcast q to all hosts
for t = 1 to n_o:
    for each transformer layer:
        for each host h concurrently:
            A_h = FlashAttention(Q_h, K_h, V_h)     // local attention
            s_h = Σ exp(Q_h K_{h,k}^T / √d)          // softmax denominator
        end for
        Gather all (A_h, s_h) at h_q
        // Online softmax (log-sum-exp) aggregation:
        s_global ← s_1, A_global ← A_1
        for h = 2 to H:
            s_global ← s_global + log(1 + exp(s_h - s_global))
            A_global ← exp(s_h - s_global)·A_global + exp(A_h - s_global)·A_h
        end for
    end for
    next_token = generate(A_global)
    if next_token = EOS: break
    update KV cache at h_q only  // context hosts' KV cache remains frozen
end for
```

**Anchor Block 机制的关键性**：Blockwise-only attention（无 anchor block）会使每个 block 独立产生 attention sink，导致多 sink 分布与 global attention 的单 sink 分布不一致。Star Attention 通过插入 anchor block 使 attention sink 集中在 anchor token 上，丢弃 anchor KV 后分布逼近 global attention。消融实验（Table 4）量化了 anchor block 的作用：无 anchor 时 64K NIAH 准确率 60.1%（vs 99.5% full attention），有 anchor 时恢复至 97.6%。

**Speedup 来源**：加速来自两方面——(a) 阶段一 blockwise-local attention 将复杂度从 O(L²) 降至 O(L·b)，且多 host 并行无通信；(b) 阶段二 distributed softmax 通信量仅 O(d) per token（vs Ring Attention 的 O(L·d) per layer）。当 block size 固定 32K、序列长度从 128K 增长到 1M 时，speedup 从 2.7× 增长到 16.9×。

术语一般如何实现？如何使用？

Star Attention 在 HuggingFace Transformers 和 NVIDIA TRT-LLM 中实现。开源代码：https://github.com/NVIDIA/Star-Attention。使用时关键参数：block_size（建议为序列长度的 1/4，128K 以上固定 32K）、anchor_block_size（建议等于 block_size）。支持 Llama-3.1-8B/70B 和 gradientai 扩展上下文模型（256K/1M）。在 A100 GPU bfloat16 上运行，8B 模型需 8-32 GPU，70B 需 8-32 GPU 取决于序列长度。Flash Attention 被用作阶段一和阶段二的底层 kernel。

涉及论文标题：
- Star_Attention__Efficient_LLM_Inference_over_Long_Sequences

## FlexPrefill (动态阈值Prefill稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

FlexPrefill（Lai et al., ICLR 2025）是一种针对 prefilling 阶段的 training-free 稀疏注意力方法。其核心创新在于**阈值驱动（threshold-based）的动态 budget 分配**：不同于 Vertical-Slash 给每个 head 分配固定数量的 vertical columns 和 slashes，FlexPrefill 通过设置 coverage 参数 α（如 α=0.8 表示覆盖 80% attention mass）和一个 min_budget 参数，让每个 head 自动决定需要保留多少 QK 交互对来达到目标 coverage。当动态分配在高稀疏度下失效时，回退到 α=0（等价于均匀 Vertical-Slash 分配）。

关键机制：(1) 首先用近端 query window（256/512 tokens）估计注意力分布；(2) 对每个 head，按 attention score 降序选择 top tokens 直至累积覆盖率 ≥ α；(3) 每 head 最低保留 min_budget 个 tokens 作为连通性保证。Sparse Frontier 实验发现：(a) FlexPrefill 在多数任务中 matching 或略低于 Vertical-Slash 的均匀分配——threshold-based 选择捕获高 attention token 但漏掉 attention 分布长尾中的重要信息（"attention sink phenomenon"效应）；(b) min_budget=512 显著改善性能；(c) 高压缩比下动态分配失效需回退。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# FlexPrefill prefilling 稀疏注意力
Input: Q, K, V ∈ R^{S×d_h}, α=0.7, min_budget=512, window=256

# Step 1: 使用近端 query 窗口估计注意力
Q_recent = Q[-window:, :]
S_approx = Q_recent @ K^T / sqrt(d)             # [window, S]

# Step 2: 沿 query 维度聚合得 per-token importance
importance = S_approx.sum(dim=0)                 # [S]
importance = softmax(importance)                  # normalize to dist

# Step 3: 保留 attention sinks (prefix + local)
preserved = [0:4] ∪ [S-64:S]                     # 固定保留

# Step 4: Threshold-based 动态选择（关键差异）
sorted_imp = sort(importance[4:S-64], descending=True)
cumsum = cumsum(sorted_imp)
num_selected = max(min_budget, argmin(cumsum >= α))  # 至少min_budget
i_vs = top_indices(sorted_imp, num_selected)

# Step 5: 稀疏 attention 计算（仅 selected QK pairs）
O = sparse_attention(Q, K, V, indices = preserved ∪ i_vs)
```

术语一般如何实现？如何使用？

FlexPrefill 开源（Apache-2.0）：https://github.com/xxxx。实现基于 Vertical-Slash 基础设施，增加 coverage-based selection 逻辑。配置参数：(α, min_budget)——低 α 等价均匀分配，高 α 保留更多 attention mass。Sparse Frontier 推荐使用场景：中等稀疏度（0.5-0.7），min_budget=512，此时动态分配略有优势。高稀疏度下建议退回到均匀 Vertical-Slash（α=0）。

涉及论文标题：
- FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

## Ada-SnapKV (自适应Budget KV Cache逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Ada-SnapKV（Feng et al., 2024）是 SnapKV 的增强版，核心创新在于**跨 head 的动态 token budget 分配**：不同于 SnapKV 给所有 attention head 分配相同数量的 KV token，Ada-SnapKV 允许各 head 获得不同数量的 token——关键 head 保留更多 token，非关键 head 保留更少，在相同总 budget 下提高信息保留率。

Sparse Frontier 的实现使用 **max-aggregation**（而非 SnapKV 的 mean-aggregation）跨 query positions 和 heads 进行分数计算，经验证明这对自适应分配更有效（但对均匀分配无影响）。每 head 最低 budget 设为 20% 容量——消融显示 10-50% 范围内性能良好，但接近 100%（等价均匀 SnapKV）时退化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Ada-SnapKV prefill阶段 KV 选择（自适应 budget 版本）
Input: Q, K, V, total_budget=2048, n_heads=32, min_ratio=0.2

# Step 1: 使用 observation window 计算 per-head per-token importance
for each head h:
    Q_obs = Q[h, -256:, :]                         # observation window
    attn_h = Q_obs @ K[h]^T / sqrt(d)              # [256, S]
    imp_h = max(attn_h, dim=0)                      # max-aggregation [S]
    imp_h = AvgPool1d(imp_h, kernel_size=21)         # smoothing

# Step 2: 跨 head 聚合得全局 importance（用于 adaptive budget 分配）
global_imp = max_pool_over_heads(imp_1..imp_H)      # [S]

# Step 3: Adaptive budget 分配
for each head h:
    # 该 head 对 top 全局重要 token 的覆盖度决定 budget
    overlap = intersection(topk_global_indices, topk_h_indices)
    budget_h = max(total_budget * 0.2,  # 最低 20%
                   total_budget * |overlap|/total_budget)
    budget_h = min(budget_h, total_budget * n_heads)  # 上限

# Step 4: 每 head 独立选择 TopK
for each head h:
    selected_h = sort(preserved ∪ TopK(imp_h, budget_h))
    K_compress[h] = K[h, selected_h]
    V_compress[h] = V[h, selected_h]
```

术语一般如何实现？如何使用？

Ada-SnapKV 开源（MIT 许可证）。与 SnapKV 共享 infrastructure，差异仅在于 budget 分配策略。使用方式：设置 token_capacity（同 SnapKV）、min_budget_ratio=0.2、kernel_size=21、observation_window=128。Sparse Frontier 评估表明 Ada-SnapKV 始终优于均匀 SnapKV（尤其 multi-query 任务），但两者均弱于 Quest（full-cache 方法）因 eviction 的不可逆信息损失。推荐在内存受限场景（无法保留全 KV cache）使用。

涉及论文标题：
- Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

## isoCost Pareto Frontier Analysis for Sparse Attention (稀疏注意力等成本Pareto前沿分析)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

isoCost Pareto Frontier Analysis 是 Sparse Frontier 论文引入的稀疏注意力评估方法论。核心思想：在相同计算成本下（prefilling 用 FLOPs、decoding 用 memory transfers），比较不同模型大小 × 稀疏度配置的 accuracy，识别 Pareto 最优配置——即不被任何其他配置同时在成本和性能上支配的配置。这种方法回答了一个实际问题："给定固定计算 budget，应该用大稀疏模型还是小密集模型？"

方法论关键步骤：(1) 对每个 (model_size, sparsity_level) 配置计算计算成本——prefilling 用 FLOPs 公式（含 attention/QKV投影/MLP/embedding 以及 sparse indexing overhead），decoding 用 memory transfers 公式（含 weight loading + KV cache 加载）；(2) 对每个配置在多个任务上评估 accuracy；(3) 在 cost-accuracy 空间中绘制所有配置点；(4) 识别 Pareto 前沿——不被任何其他点支配的边界点集。

Sparse Frontier 的核心发现：对于长序列（128K），只有高稀疏度配置（0.8-0.93 sparsity, i.e. 1/5-1/15 attention budget）处于 Pareto 前沿上。大稀疏模型在等成本下始终优于小密集模型。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# isoCost Pareto Frontier 分析流程
# Step 1: 成本计算（prefilling FLOPs 示例）
for model_size in [7B, 14B, 32B, 72B]:
    for sparsity in [0, 0.33, 0.5, 0.6, 0.7, 0.8, 0.87, 0.9, 0.93, 0.95]:
        ρ = 1 - sparsity  # attention density
        # Attention FLOPs (Eq. 3 from paper):
        FLOPs_attn = N_layers * (2*L*d*(d + 2*d_h*n_kv + d) 
                     + ρ * (2*h*L²*d_h + 3*h*L² + 2*h*L²*d_h))
        # Total prefill FLOPs:
        FLOPs_total = FLOPs_embed + FLOPs_attn + FLOPs_mlp + FLOPs_logits

# Step 2: 计算 accuracy（所有 9 个任务平均）
accuracy = mean(task_accuracy over 9 tasks for this config)

# Step 3: 绘制并识别 Pareto frontier
points = {(cost_i, acc_i) for all configs}
pareto_frontier = []
for p in points:
    if not exists q: q.cost < p.cost AND q.acc >= p.acc:
        pareto_frontier.append(p)  # p is Pareto-optimal

# Step 4: 沿着 Pareto frontier 比较效率交叉点
# "efficiency crossover" = 大稀疏模型开始优于小密集模型的点
```

术语一般如何实现？如何使用？

isoCost 分析方法的关键价值在于提供 hardware-agnostic 的效率比较框架——FLOPs 和 memory transfers 在优化实现下与 wall-clock time 高度相关，但避免了特定硬件/实现的具体延迟测量偏差。适用场景：(a) 稀疏注意力方法的理论效率对比；(b) 部署决策——选择给定 budget 的最优 (model_size, sparsity) 配置；(c) 指导未来稀疏注意力设计方向。局限性：未考虑 batch size 效应（batch_size=1 时 decode KV cache 占比低，稀疏收益小）和 memory hierarchy 效应（cache hit/miss）。

涉及论文标题：
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

## Sublinear Token Budget Scaling for Sparse Attention (稀疏注意力Token预算的次线性缩放)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Sublinear Token Budget Scaling 是 Sparse Frontier 论文发现的一个关键规律：随着序列长度 L 增长，维持相同 accuracy degradation 所需的 token budget 以**次线性（sublinear）**速率增长，而非线性（linear）或固定（fixed）。具体来说，对于目标相对误差 ≈0.2，所需的 attention budget fraction 从 16K 时的 1/10 降至 32K 时的 1/15 再到 64K 时的 1/20——即翻倍序列长度不需要翻倍 token budget。

理论基础：Herdan's Law（Herdan, 1960）——自然语言中，序列越长，新信息的出现频率越低，允许更高的稀疏度。从信息论角度，更长上下文的 token 级信息密度递减（diminishing marginal information density），使得注意力模式更集中在少数关键 token 上。

实践意义：(1) 当前 production 中常用的固定 budget 方法（如固定 token_budget=4096）是次优的——应该在长度增长时增加预算但不必翻倍；(2) 固定 budget fraction（如总是 10% attention）则是过于保守——实际可以随着长度增长逐步提高 sparsity；(3) 最优预算函数应遵循 budget(L) ∝ L^k, k<1（如 k≈0.7-0.8）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Sublinear Budget Model (基于 Sparse Frontier 的 iso-error 曲线)
# 目标：给定目标 relative error ε，求预算函数 B(L)
# 
# 从 Figure 3 提取的近似 iso-error 点：
#   ε=0.2:  B(16K)≈0.1L, B(32K)≈0.067L, B(64K)≈0.05L
#   ε=0.1:  B(16K)≈0.2L, B(32K)≈0.2L, B(64K)≈0.17L

# Sublinear model fitting:
#   B(L) = c * L^k, where k < 1
#   For ε=0.2: k ≈ log(0.05/0.1) / log(64/32) ≈ log(0.5)/log(2) ≈ -1 → 
#              B ∝ L^0 (constant fraction decreasing!)
# Wait, the fraction itself decreases. Let me clarify:
#   budget_fraction(16K)=1/10, budget_fraction(64K)=1/20
#   token_count(16K)=1600, token_count(64K)=3200
#   So doubling L from 16K→32K→64K only needs 1.33×→1.33× more tokens

# Practice: dynamic budget schedule for serving
def get_token_budget(seq_len, base_budget_16k=1600):
    """次线性 token budget 调度器"""
    # 确保 budget 至少不减少
    return max(base_budget_16k, base_budget_16k * (seq_len/16384)^0.5)
```

术语一般如何实现？如何使用？

实际部署时:(1) 根据任务的 accuracy 要求确定 ε 容忍度；(2) 通过离线 profiling 建立 L→B(L) 查找表（对每种稀疏注意力方法和模型）；(3) serving 时根据实际 prompt length 查表和设置 token budget。Sparse Frontier 建议未来方向：开发可靠的动态 budget 分配机制（目前 dynamic 方法如 FlexPrefill 缺乏鲁棒性），使 sublinear scaling 能自动实现。

涉及论文标题：
- The Sparse Frontier: Sparse Attention Trade-offs in Transformer LLMs

## Token Importance Prediction via Attention Distillation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token Importance Prediction via Attention Distillation 是一种通过知识蒸馏训练轻量级预测器来估计 LLM 解码过程中每个 token 对当前 query 重要性的方法。核心思想：冻结预训练 LLM 作为 teacher，用一个极小的外部 MLP 预测器（<1% LLM 参数量）作为 student，蒸馏 teacher 每层每个 head 的 masked causal attention distribution。训练时，teacher 产出每层每 head 的真实注意力分布 A_true，student 预测低维 importance queries 并与降维后的真实 KV-cache keys 做点积得到 A_pred，最小化 softmax 化后的 cross-entropy loss：L_CE = -E[Σ P_k log(Q_k)]，其中 P = softmax(A_true), Q = softmax(A_pred)。推理时，预测器输出 token 重要性分数，在固定 budget 下选择 top-k token 参与注意力计算。训练数据仅需 1K 长度的通用语料（C4、FineWeb-Edu、CodeParrot、BABILong），预测器通过 key-cache 投影机制泛化到 64K 长上下文。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Token Importance Prediction 插入 LLM decode pipeline 中，位于每层 attention 计算之前：

```
# 训练阶段
For each training sequence (length ≤ 1K):
    # 冻结 LLM 前向，收集 teacher attention
    with torch.no_grad():
        for layer in LLM.layers:
            Q, K, V = project(layer.input)
            A_true[layer] = masked_attention(Q, K)  # teacher logits
    
    # 仅训练预测器参数
    for producer_layer in {0, G, 2G, ...}:
        H = hidden_states[producer_layer]             # [B, L, E]
        Q_imp = MLP(LayerNorm(H))                      # [B*H, G, L, d']
        
        for consumer_layer in [producer+1, producer+G]:
            slot = (consumer_layer - 1) % G
            K_proj = K_cache[consumer_layer] @ W_K[l]  # [B, H_kv, L, d']
            A_pred = Q_imp[:, slot] @ K_proj.transpose  # [B*H, L, L]
            
            # 蒸馏 loss：teacher-student 交叉熵
            loss += CE(softmax(A_true[consumer_layer] + mask),
                       softmax(A_pred + mask))
    
    loss.backward()  # 仅更新 MLP 和 W_K 参数

# 推理阶段（decode step t）
if t % prediction_interval == 0:
    H = hidden_states[producer_layer][:, -1:, :]  # 仅最新 token
    Q_imp = MLP(LayerNorm(H))                       # [B*H, G, 1, d']
    for consumer_layer in consumer_layers:
        K_proj = K_cache[consumer_layer] @ W_K
        scores = Q_imp[:, slot, 0, :] @ K_proj.T    # [H, L_kv]
        # 排除 sink + window tokens，取 top-B
        selected = topk(scores[candidate_mask], B)
        migrate_to_important_buffer(selected)

# Attention: 拼接 [Sink | Important | Local_Window]
attn_out = FlashAttention(Q, K[selected_all], V[selected_all])
```

Key dimensions: d'=16 (interaction dimension), G=4 (producer frequency), MLP hidden=512. Predictor params: 29.4M for Llama-8B (0.368%), 20.9M for Qwen2.5-7B (0.299%). Training cost: ~9h on single A6000 for Llama-8B.

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：在 HuggingFace Transformers 模型上挂载外部预测器模块。预测器结构为 LayerNorm → Linear(E→512) → GELU → Linear(512→B*H*G*d') → Reshape。K-cache 投影矩阵为每层独立参数 W_K^(l) ∈ R^{D×d'}（使用 GQA 时 H_kv < H，需 broadcast）。训练时使用 row-subsampling 加速：仅对序列尾部 R 个位置的 query 计算 loss（R << L），将 O(L²) 降为 O(RL)。推理时预测器在 producer layer 处每 prediction_interval 步触发一次，中间步复用上次选择。可与 FlashAttention 标准 kernel 无缝集成。代码开源：https://github.com/abdelfattah-lab/TokenButler。

涉及论文标题：
- TokenButler: Token Importance is Predictable

## Low-Dimensional Importance Query and Key-Cache Projection (低维重要性查询与键缓存投影)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
一种将高维 KV-cache key（维度 D=128/head）和预测器输出的 importance query 同时投影到低维交互空间（d'=16）进行高效 token 重要性评分的机制。核心由两部分组成：(1) Query Prediction：producer layer 的 hidden states H ∈ R^{B×L×E} 经 LayerNorm + 二层 MLP 生成 G 个 slot-specific 低维 importance queries Q_imp ∈ R^{(B·H)×G×L×d'}，每个 slot 对应一个 consumer layer；(2) Key-Cache Projection：对每个 consumer layer l，其真实 KV-cache keys K^(l) ∈ R^{B×H_kv×L×D} 通过学习投影矩阵 W_K^(l) ∈ R^{D×d'} 降维到同样的 d' 维空间，得到 K_imp^(l) = K^(l) · W_K^(l) ∈ R^{B×H_kv×L×d'}。Token 重要性分数 = Q_imp[slot] @ K_imp^T，计算复杂度为 O(L·d')，仅为完整 attention O(L·D) 的 d'/D = 16/128 = 1/8。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Transformer pipeline 中的定位：

```
Transformer Layer (consumer layer l):
    # Stage 0: Importance scoring（仅在预测步运行）
    if is_prediction_step:
        # 在 producer layer 已计算好 Q_imp
        K_proj = K_cache[l] @ W_K[l]        # GEMM: [L, D] × [D, d'] → [L, d']
        scores = Q_imp[slot] @ K_proj.T      # MatMul: [H, 1, d'] × [H, d', L] → [H, L]
        # scores[h, t] = Σ_{k=1}^{d'} Q_imp[h,slot,k] · K_proj[h,t,k]
        topk_indices = argtopk(scores, B)     # 选 top-B token
    
    # Stage 1: 构建稀疏 KV 集合
    K_sparse = gather(K_cache[l], [sink | topk_indices | local_window])
    V_sparse = gather(V_cache[l], [sink | topk_indices | local_window])
    
    # Stage 2: 标准 Attention (FlashAttention)
    output = FlashAttention(Q_current, K_sparse, V_sparse)
```

重要性评分计算的具体张量操作：
- Q_imp shape: [B, H, G, d'] (per-producer step, per-token)
- K_proj shape: [B, H_kv, L_kv, d'] (预先计算并缓存)
- scores shape: [H, L_kv] per consumer layer
- 使用 GQA 时：H_kv 个 key head 的 score broadcast 到 H 个 query head

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：K-cache 投影在 token 离开 local window 时批量执行（每 N 步一批），利用 cuBLASS GEMM 批量处理 N 个 token 的投影，充分利用 GPU HBM 带宽。Q_imp 预测使用单次 MLP forward（约 512×d'×G×H 次乘加），远小于一层 transformer 的 attention + FFN。d'=16 的选择在精度和效率间平衡：更小的 d' 更快但 recall 下降（ablation 显示 d'=16 时 Recall@50% ≈ 67% for 3.48M predictor）。投影矩阵 W_K 与 token 位置无关（position-agnostic），仅依赖层号和 key head，因此可预先计算并缓存投影结果。

涉及论文标题：
- TokenButler: Token Importance is Predictable

## Co-referential Token Retrieval in KV-Cache Sparsity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Co-referential Token Retrieval 指在长文本解码过程中，模型需要回溯并精确检索之前在上下文中出现过的特定实体（如人名、地名、术语）的能力。在 KV-cache 稀疏化场景中，这是一个核心挑战：较早出现的 co-referential token 可能在当前步的注意力分数较低（因为它与中间插入的 distractor 内容无关），但在未来的查询步中可能突然变得非常重要。例如：对话开头提到 "wraithspire" 这个地点名，中间插入大量无关内容，当后续问题问及地点时，"wraithspire" 的每个 token 都需要被完整检索到。TokenButler 论文设计了一个 synthetic co-reference benchmark（100 个虚构地点名，10^8 组合空间）专门测试这一能力，评估 accuracy（所有 token 被完整检索的比例）和 coverage（被检索到的 token 比例）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Co-referential retrieval 的失败模式分析：
- **驱逐型方法（H2O, SnapKV）**：token 在当前步注意力分数低时被永久驱逐，后续无法检索 → accuracy 仅 1-10%（Llama-8B）
- **分页方法（Quest）**：co-referential 实体的 token 可能跨越 page 边界，page 级别选择可能丢失部分 token → coverage 仅 19-58%
- **Oracle（理想）**：49-81% accuracy（上限取决于模型本身能力）
- **TokenButler**：逐 token 细粒度选择 + 不驱逐任何 token → accuracy 48-80%，接近 Oracle

```python
# Co-referential benchmark 样本结构
sample = {
    "contextual_lead": "Shrouded in fog, place is:",
    "location": "wraithspire",          # 需要被检索的实体
    "philosophical": "...distractor...",
    "culinary": "...distractor...",
    "math_problem": "...distractor...",
    "location_prelude": "Which location up the shore?",
    "answer": "wraithspire"            # 期望输出
}
# 评估：检查 "wraithspire" 的所有 token 是否完整被 attention 机制访问到
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
该 benchmark 使用 OpenAI gpt-4o-mini 生成 100 个独立组件（地点名、上下文引导语、哲学陈述、烹饪描述、数学问题），随机组合生成测试样本。评估时不依赖生成质量（不检查输出正确性），而是直接检查 attention mask 中的 token selection accuracy 和 coverage — 这是 token 级别的检索精度衡量。可用于评估任何 KV-cache 稀疏化方法在 retrieval-intensive 场景下的表现。

涉及论文标题：
- TokenButler: Token Importance is Predictable

## Dynamic Mask Attention (DMA / 动态掩码注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic Mask Attention (DMA) 是一种可训练的 content-position 双感知稀疏注意力机制，由港科大(广州)与智源(BAAI)于 2025 年提出（arXiv: 2508.02124，提交 NeurIPS 2025）。DMA 将稀疏注意力的计算分解为两个解耦的阶段：(1) Content-Aware Dynamic Mask Generation：从 value 向量表示中采样生成 per-head 动态 mask，使模型能自适应识别关键 token；(2) Position-Aware Sparse Weight Computation：利用动态 mask 对 scaled dot-product attention 做稀疏化，mask 值为 −∞ 的位置直接跳过计算。整个过程完全可微，支持端到端训练。DMA 的核心创新在于将"哪些 token 需要关注"（内容感知）和"如何高效计算这些关注"（位置感知/硬件友好）解耦，使有效复杂度从 O(n²d_h) 降为 O(nwd_h)，内存从 O(n²) 降为 O(nw)。

注意区分：DMA（本术语）与 DAM (Dynamic Attention Mask, Zhang et al., 2025) 是不同的方法。DAM 是一种针对推理阶段的动态注意力掩码方法（被 DMA 论文列为 baseline），而 DMA 是端到端可训练的稀疏注意力机制。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

DMA 算法 pipeline 分为两个阶段：

**Phase 1 — Content-Aware Dynamic Mask Generation**：
```
Input: V ∈ R^{b, n_h, n, d_h}  (value 矩阵)
Params: Δ ∈ R^{n_h, d_h, n_h}  (采样权重), A ∈ R^{n_h} (门控系数)
Hyperparams: w (per-head window size), τ(·) (非负激活, 如 softplus)

1. V_flat = V.transpose(1,2).reshape(b, n, n_h*d_h)   # 展平 head 维度
2. dt = V_flat @ W_dt                                  # 线性投影 → [b, n, n_h]
3. dt = exp(A * τ(dt))                                 # 门控 + 激活 + exp → 非负分数
4. dt = dt.transpose(-1, -2)                          # → [b, n_h, n]
5. m_t = dt.expand(-1, -1, q_len, -1)                 # broadcast 到 query 维度
6. m_t = m_t.masked_fill(causal_mask, -inf)           # 施加 causal mask
7. topk_indices = topk(m_t, w, dim=-1).indices        # per-head 选择 top-w
8. m_t = m_t.masked_fill(not in topk_indices, -inf)   # 非 top-w 位置置 −∞
Output: m_t ∈ R^{b, n_h, q_len, n}  (dynamic mask)
```
A 可设计为 query-dependent：A = f(q_t)，使 gating coefficient 随输入自适应。

**Phase 2 — Position-Aware Sparse Attention**：
```
Input: Q, K, V, m_t, topk_indices
Output: O_t

for each (batch, head, query_pos):
    indices = topk_indices[b, h, q]           # top-w 位置索引, shape [w]
    K_sel = K[b, h, indices, :]               # [w, d_h]
    V_sel = V[b, h, indices, :]               # [w, d_h]
    m_sel = m_t[b, h, q, indices]             # [w]
    scores = (Q[b,h,q,:] @ K_sel^T) / sqrt(d_h) + m_sel  # [w]
    attn_w = softmax(scores)                   # [w]
    O[b,h,q,:] = attn_w @ V_sel               # [d_h]
```
Kernel 实现中，若 mask block 全为 −∞，则直接跳过整个 block 的加载和矩阵乘。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

开源实现：CUDA kernel 在 https://github.com/HKUSTDial/flash-sparse-attention（亦为 flash-algo/flash-sparse-attention），Triton 参考实现在 https://github.com/SmallDoges/flash-dmattn。

使用方式：替换标准 Transformer 的 self-attention 模块。训练配置：AdamW + WSD LR scheduler，RoPE 位置编码（base freq 从 10K 调整到 100K 用于长上下文适应），NeoX tokenizer。关键超参数 w（per-head 保留 key 数量）——在 Scaling Laws 实验中 w=1024（80M-680M）和 w=2048（1.7B）。DMA 可近似 full attention：当 n_h × w ≥ n 时，所有 token 都可能被某些 head 选中。

局限：(1) 固定 window size w 无法自适应任务复杂度变化；(2) RoPE 位置编码仍是外推瓶颈；(3) 目前仅在 text domain 验证，多模态扩展尚未实现；(4) 实验规模最大 1.7B 参数，更大模型（7B+）效果待验证。

涉及论文标题：
- Trainable_Dynamic_Mask_Sparse_Attention

## Content-Aware Dynamic Mask Generation from Value Vectors (基于Value向量的内容感知动态掩码生成)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Content-Aware Dynamic Mask Generation 是 DMA 的核心子模块，从 Transformer 的 value 向量表示中生成内容感知的稀疏掩码，决定每个 attention head 应关注哪些历史 token。不同于传统方法：(1) SWA 使用固定局部窗口——对内容无感知；(2) NSA 使用压缩后的静态 token 选择——虽可训练但模式固定；(3) H2O/Quest 等使用启发式重要度估计——不可微。DMA 的 mask 生成完全基于可微操作（线性投影 + 激活 + exp + top-w），gradient 可经 m_t 回传到门控参数 A 和采样权重 Δ。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**三步计算流程**：
```
Step 1 — Value 投影：δ = v · Δ
  每个 token 的 d_h 维 value 向量经采样权重 Δ 投影为 n_h 维标量
  v ∈ R^{n_h × n × d_h}, Δ ∈ R^{n_h × d_h × n_h}
  输出：投影分数 ∈ R^{n_h × n}（每个 head 每个 token 的初始重要性估计）

Step 2 — 门控激活：δ = exp(τ(v·Δ) × A)
  τ(·) 为非负激活（如 softplus），确保分数非负
  A ∈ R^{n_h} 为 per-head 门控系数——控制每个 head 的稀疏程度
    A 可设为 query-dependent：A = f(q_t)，使稀疏度自适应输入
  exp(·) 放大分数差异，促进 head 特化（不同 head 学习不同的 A 值）

Step 3 — Top-w 稀疏化：f(δ)
  f(δ_{h,j}) = δ_{h,j} if δ_{h,j} ∈ top_w(δ_h) else −∞
  per-head 独立选择 top-w，不同 head 可关注不同的 token 子集
  因果语言建模中额外施加 causal mask（通过 broadcast，无额外内存）
```

**关键设计选择**：
- 从 V（而非 Q 或 K）生成 mask 的理论动机：V 携带了每个 token 的语义内容信息，从中采样的重要性分数直接反映"该 token 的内容对当前预测有多相关"。
- per-head 独立 top-w：允许 head 特化——有的 head 专注局部（local heads），有的关注远距离（range-dependency heads），有的进行全局采样（global context heads）。
- top-w 操作在 forward pass 中是离散的，但 backward pass 中仅对选中位置传播梯度，未选中位置的梯度自然为零——这是正确行为而非近似。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PyTorch 实现伪代码（来自论文 Listing 1）：
```python
# V: [b, n_h, n, d_h], W_dt: [n_h*d_h, n_h], A: [n_h]
dt = W_dt(V.transpose(1,2).reshape(b, n, -1))  # [b, n, n_h]
dt = torch.exp(A * F.softplus(dt)).transpose(-1, -2)  # [b, n_h, n]
# broadcast + causal mask
m_t = dt[:, :, None, :].expand(-1, -1, q_len, -1)
m_t = m_t.masked_fill(causal_mask != 0, -float('inf'))
# top-w per head
topk_indices = torch.topk(m_t, w, dim=-1, sorted=False).indices
m_t = m_t.masked_fill(scatter_mask == 0.0, -float('inf'))
```
CUDA kernel 中，mask 生成和 attention 计算融合为单个 kernel，mask 在 SRAM 中分块生成和消费，避免物化完整 n×n mask 矩阵。

涉及论文标题：
- Trainable_Dynamic_Mask_Sparse_Attention

## Fully Differentiable Sparse Attention (完全可微稀疏注意力 / 端到端可训练稀疏注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fully Differentiable Sparse Attention 是指稀疏模式可以端到端通过梯度下降学习的注意力机制。DMA 首次在 content-position 双感知的稀疏注意力中证明了：(1) 动态 mask 和稀疏权重不阻塞梯度；(2) 保留路径的梯度与 full attention 严格一致；(3) 即使 mask 生成涉及 top-w 离散操作，梯度也能完整流向所有参数（Δ, A, Q, K, V）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**梯度流分析**（单 head, 单 time step）：

设 I_h 为选中的 w 个位置索引集。前向：
```
s_{h,j} = (q_h · k_{h,j})/√d_h + m_{h,j}
p_{h,j} = exp(s_{h,j}) / Σ_{j'∈I_h}exp(s_{h,j'})  for j∈I_h, else 0
o_h = Σ_{j∈I_h} p_{h,j} · v_{h,j}
```

反向梯度流：
```
1. dv_{h,j} = p_{h,j} · do_h          (j∈I_h), 0 (j∉I_h)
2. dp_{h,j} = v_{h,j} · do_h
3. ds_{h,j} = p_{h,j} · (dp_{h,j} - Σ_{j'∈I_h} p_{h,j'} · dp_{h,j'})
   — 对 mask 位置 p_{h,j}=0 → ds_{h,j}=0 (自然为零，非近似)
4. dm_{h,j} = ds_{h,j}                — 梯度直接流向 mask 参数
5. dq_h = Σ_{j∈I_h} ds_{h,j} · k_{h,j}/√d_h
6. dk_{h,j} = ds_{h,j} · q_h/√d_h    (j∈I_h), 0 (j∉I_h)
```

**关键洞察**：
- 对选中位置 j∈I_h，梯度计算与 full attention 完全一致——DMA 仅裁剪了对可忽略贡献位置的算子链。
- top-w 操作在 backward pass 中不需要梯度：未选中位置 p_{h,j}=0 → ds_{h,j}=0，跳过计算和梯度传播是数学上正确的结果。
- dM = dS 的等价关系使 kernel 只需局部重算 S 而不需额外存储中间 mask 梯度张量。
- 门控参数 A 和权重 Δ 直接接收 attention weights 的梯度，快速 shaping head 特化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DMA 的 CUDA kernel（flash-sparse-attention）实现了完整的 forward+backward 可微 pipeline。Backward pass（Algorithm 2 in paper）与 forward 共享统一 skip logic——相同 mask block judge 决定是否加载 K/V tile。gradient chain 包含 fused bias gradients。与 MagicPIG（使用离散 LSH 采样→不可微）、ClusterKV（k-means 聚类→梯度阻断）等方法对比，DMA 的每个操作都是可微的。

涉及论文标题：
- Trainable_Dynamic_Mask_Sparse_Attention

## Multi-Head Latent Attention (MLA / 多头潜在注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Multi-Head Latent Attention (MLA) 是 DeepSeek-V2/V3 引入的注意力机制，通过低秩分解将 KV cache 压缩为低维 latent vector，在保持模型质量的同时大幅降低推理时的内存和带宽开销。核心思想：将 Key 和 Value 的投影分解为两步——先将输入 x_t 通过下投影矩阵 W^{DKV} ∈ R^{r_kv × D} 映射到 r_kv 维 latent 空间，生成 c_t^{KV} = W^{DKV} x_t（仅缓存此 latent vector，而非完整 K/V）；再通过上投影矩阵 W^{UK}, W^{UV} ∈ R^{hd × r_kv} 将 latent vector 还原为每个 attention head 的 K/V。Query 同样做低秩分解（W^{DQ} + W^{UQ}）以减少训练激活内存。位置编码采用 Decoupled RoPE——额外使用独立的多头 query q_{t,i}^R 和共享 key k_t^R 携带 RoPE 位置信息，与 content 部分 [q_{t,i}^C; q_{t,i}^R] 和 [k_{t,i}^C; k_t^R] 拼接。

MLA 支持两种计算范式切换：(1) 训练/高计算阶段使用类 MHA 范式（Equation 9）——各 head 独立计算完整 K/V，计算开销略低于标准 MHA；(2) 推理/高通信阶段使用 Absorb 操作切换到类 MQA 范式（Equation 10）——将 W^{UK} 吸收进 query projection，所有 head 共享一个 latent KV head，仅需缓存 c_t^{KV}（r_kv 维），类似 MQA 的极致 KV cache 压缩。

TransMLA 论文理论证明了 MLA 的表达能力严格强于 GQA：相同 KV cache 大小下，GQA 仅是 MLA 的一个稀疏子集（W^{UK}/W^{UV} 必须是 block-selector 矩阵），而 MLA 的 dense 上投影矩阵允许跨 head 混合信息。GQA < MLA_Factorized < MQA。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**MLA 推理范式（Absorb 操作后，Equation 10）**：
```
// latent feature（缓存此项，r_kv 维）
c_t^{KV} = W^{DKV} @ x_t          // [r_kv]

// Absorb: W^{UK} 吸收进 query
q̂_{t,i} = [W_i^{UK}^T @ q_{t,i}^C; q_{t,i}^R]  // per head, [r_kv + d^R]
k̂_t = [c_t^{KV}; k_t^R]                         // shared, [r_kv + d^R]

// 所有 head 共享一个 KV head（类似 MQA）
ô_{t,i} = Σ_j softmax(q̂_{t,i}^T @ k̂_j / √(d+d^R)) @ c_j^{KV}  // [r_kv]

// W^{UV} 合并到 output projection
y_t = W^O @ [W_1^{UV} @ ô_{t,1}; ...; W_h^{UV} @ ô_{t,h}]
```

**KV Cache 对比（d=128, h=32, g=8, r_kv=512）**：
| 机制 | KV Cache 维/token | 相对于 GQA |
|------|-------------------|-----------|
| MHA | 2×32×128=8192 | 4× |
| GQA | 2×8×128=2048 | 1× |
| MLA (r_kv=512) | 512 | 0.25× |
| MLA (r_kv=144) | 144 | 0.07× (93% 压缩) |

术语一般如何实现？如何使用？

MLA 在 DeepSeek-V2/V3/R1 中全面部署，配合 FlashMLA kernel 实现高效推理。开源实现：DeepSeek 官方仓库（FlashMLA: https://github.com/deepseek-ai/FlashMLA），vLLM 和 SGlang 均有 MLA 优化支持。TransMLA 提供 GQA→MLA 转换工具（https://github.com/MuLabPKU/TransMLA），生成的 MLA checkpoint 可直接在 DeepSeek 生态中运行。一般使用流程：训练时使用 full MHA-like 范式（Equation 9）以利用 GPU 算力；推理时切换到 Absorb 范式（Equation 10）以减少 KV cache 内存和带宽。兼容 FP8 量化、Multi-Token Prediction 等 DeepSeek 优化。

涉及论文标题：
- TransMLA: Multi-Head Latent Attention Is All You Need
- xKV: Cross-Layer SVD for KV-Cache Compression

**xKV 对 MLA 架构的进一步压缩**：xKV 证明了其跨层 SVD 压缩方法可以直接应用于 MLA 架构的 latent KV-Cache，实现**已压缩 cache 的再压缩**。具体做法：对 MLA 的 non-RoPE latent representations（c_t^{KV}，已压缩到 r_kv 维）按组做跨层 SVD，解耦的 RoPE keys（k_t^R）不压缩。在 DeepSeek-Coder-V2-Lite-Instruct（16B MoE, 2.4B activated, MLA）上，xKV-4 在 RepoBench-P 上实现 3× 压缩率、LCC 上 3.5× 压缩率，均无精度损失。作为对比，MiniCache 和 Single SVD 在此 MLA 架构上连更低的压缩率都无法保持精度。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Decoupled RoPE 是 DeepSeek MLA 中处理位置编码的策略：将 RoPE 位置信息从 content 计算路径中分离出来，使 content 部分的 K/V 可以安全地做低秩分解和 Absorb 操作。核心问题：标准 RoPE 直接施加在 Q 和 K 上（旋转操作与 token 位置 t 相关），如果 K 通过 W^{UK} 从 latent c^{KV} 上投影得到，则 RoPE 旋转与矩阵乘法不满足交换律——(Rot(W^{UK} c)) 无法被吸收为 (W^{UK}_rot c)。Decoupled RoPE 的解决方案：content 部分 [q_{t,i}^C; k_{t,i}^C] 不施加 RoPE（可安全吸收/压缩），额外创建独立的 RoPE 通道——多头 query q_{t,i}^R = RoPE(W^{QR} c_t^Q, t) 和共享 key k_t^R = RoPE(W^{KR} x_t, t)，将位置信息编码在这些独立通道中。最终 attention score = content_score + position_score，其中 content_score = (q_{t,i}^C)^T k_{j,i}^C（可吸收），position_score = (q_{t,i}^R)^T k_j^R（MQA 结构）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Decoupled RoPE 的完整计算流程**：
```
// 输入 token x_t
c_t^{KV} = W^{DKV} @ x_t           // latent KV, content-only (无 RoPE)
c_t^Q = W^{DQ} @ x_t               // latent Q, content-only

// Content 通道（可安全吸收，因不含 RoPE）
k_{t,i}^C = W_i^{UK} @ c_t^{KV}    // per-head content key
q_{t,i}^C = W_i^{UQ} @ c_t^Q       // per-head content query

// Position 通道（独立 RoPE）
k_t^R = RoPE(W^{KR} @ x_t, t)      // shared position key
q_{t,i}^R = RoPE(W_i^{QR} @ c_t^Q, t)  // per-head position query

// 拼接后计算 attention
k_{t,i} = [k_{t,i}^C; k_t^R]       // content + shared position
q_{t,i} = [q_{t,i}^C; q_{t,i}^R]

// Attention score = content interaction + position interaction
score = (q_{t,i}^C)^T @ k_{j,i}^C + (q_{t,i}^R)^T @ k_j^R
```

**为何 Content 部分可吸收而 Position 部分不可**：
- Content: q_{t,i}^C = W_i^{UQ} @ W^{DQ} @ x_t, k_{j,i}^C = W_i^{UK} @ W^{DKV} @ x_j
  → (q_{t,i}^C)^T @ k_{j,i}^C = x_t^T @ (W^{DQ})^T @ (W_i^{UQ})^T @ W_i^{UK} @ W^{DKV} @ x_j
  → W_i^{UK} 可吸收进 query: q̂_{t,i} = (W_i^{UK})^T @ q_{t,i}^C, k̂_j = c_j^{KV}
- Position: 因 RoPE 旋转矩阵 R(t) 与 W 不交换: R(t) @ W ≠ W @ R(t)
  → 无法吸收，需独立处理

术语一般如何实现？如何使用？

Decoupled RoPE 在 DeepSeek-V2/V3 和所有 MLA-based 模型（包括 TransMLA 转换后的模型）中使用。实现时需额外分配 per-head dimension d^R 给 RoPE 通道（如 d^R = d/2 = 64），Content 维度为 d^C = d - d^R。总 KV cache 中的 content 部分可压缩（低秩 latent），position 部分 k_t^R（d^R 维，所有 head 共享）不可压缩但开销极小。TransMLA 的 RoRoPE 技术进一步将 GQA 模型中分散在各 head 的 RoPE 信息集中到第一 head，实现 Decoupled RoPE 的等价转换。

涉及论文标题：
- TransMLA: Multi-Head Latent Attention Is All You Need

---

## RoRoPE (Rotational RoPE PCA / 旋转式 RoPE 主成分分析)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

RoRoPE 是 TransMLA 论文提出的将 GQA 模型中分散在多个 KV head 的 RoPE 位置信息集中到第一个 attention head 的技术，是实现 GQA→MLA 转换中解耦 RoPE 的关键步骤。核心原理：当多个 KV head 合并为一个 latent head 后，每个 head 的同一 RoPE 频率维度在各自 head 内独立旋转。RoRoPE 利用正交旋转在 RoPE 内积下的不变性（Theorem/Equation 19）：对于第 l 个 RoPE 频率对应的 2D 子空间（real + imaginary），将各 head 中该子空间的分量拼接为 g 维向量，用正交矩阵 U_l ∈ R^{g×g} 旋转。因为 U_l^T U_l = I 且 real/imag 分量使用相同的 U_l，内积不变。选择 U_l 使得旋转后第一 head 捕获最大方差（PCA），其余 head 的位置信息可忽略，从而安全移除其 RoPE。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**RoRoPE 计算流程**（g 个 KV head，每 head d 维）：
```
// Step 1: 合并所有 KV head 为一个 latent head
// 引入 W_i^{UK} 作为 selector 矩阵（等价变换）

// Step 2: 对每个 RoPE 频率 l ∈ {1,...,d/2}：
For l = 1 to d/2:
    // 从所有 g 个 head 收集第 l 个 RoPE 子空间
    K_x_real = []  // shape: [N, g], real 分量
    K_y_imag = []  // shape: [N, g], imag 分量
    For head_idx in 0..g-1:
        K_x_real[:, head_idx] = key_activations[head_idx, 2l-1, :]  // 第 (2l-1) 维
        K_y_imag[:, head_idx] = key_activations[head_idx, 2l, :]    // 第 (2l) 维

    // 构建联合协方差矩阵
    Σ_l = K_x_real^T @ K_x_real + K_y_imag^T @ K_y_imag  // [g, g]

    // 特征分解得到最优正交旋转矩阵 U_l
    eigenvalues, U_l = eig(Σ_l)  // 按特征值降序排列

    // 旋转 W^K 和 W^{UK}（等价变换，不改变 attention 输出）
    // 旋转后第一 head 捕获 max variance → K_rope
    // 其余 head 位置信息可忽略 → K_nope

// Step 3: 移除 K_nope 的 RoPE
// K_rope（第 1 head）保留 RoPE；K_nope（第 2~g head）去除 RoPE
```

**内积不变性证明（Equation 19 核心）**：
```
S_l = cos((t-j)θ_l) · (q_x^T k_x + q_y^T k_y) + sin((t-j)θ_l) · (q_x^T k_y - q_y^T k_x)
S'_l = cos((t-j)θ_l) · ((U_l q_x)^T (U_l k_x) + (U_l q_y)^T (U_l k_y)) 
     + sin((t-j)θ_l) · ((U_l q_x)^T (U_l k_y) - (U_l q_y)^T (U_l k_x))
     = cos((t-j)θ_l) · (q_x^T U_l^T U_l k_x + q_y^T U_l^T U_l k_y) + ...
     = cos((t-j)θ_l) · (q_x^T k_x + q_y^T k_y) + sin((t-j)θ_l) · (q_x^T k_y - q_y^T k_x)
     = S_l  // 因 U_l^T U_l = I
```

术语一般如何实现？如何使用？

RoRoPE 在校准数据集（如 WikiText-2 子集）上离线执行。收集每层 key 激活值 → 构建联合协方差矩阵 → 特征分解得到 U_l → 旋转 W^K 和 W^{UK}。整个过程为等价变换（不改变 attention 输出），training-free。关键约束：同一 RoPE 子空间的 real 和 imag 分量必须使用相同的 U_l（否则内积不保持不变）。选择保留的主成分数 m：m=1 表示仅第一 head 保留 RoPE（最激进），m>1 表示更多 head 共享 RoPE 信息。TransMLA 实验证明 LLaMA-3-8B 上 RoRoPE 在 90% RoPE 去除率下仍保持 log-perplexity ≈ 2，而 MHA2MLA 方法升至约 6。

涉及论文标题：
- TransMLA: Multi-Head Latent Attention Is All You Need

---

## FreqFold (Frequency Folding / 频率折叠)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

FreqFold 是 TransMLA 论文提出的增强 RoRoPE 的技术：利用 RoPE 中相邻频率 θ_l = 10000^{-2(l-1)/d} 的数值接近性，将多个频率维度的 key 段合并后做联合 PCA，使 K_rope 能占用更多维度（而非仅第一 head 的 d 维），保留更丰富的位置信息。核心原理（Proposition 2）：将 M 个原始 RoPE 频率组的 d' 维数据段拼接为 M·d' 维后进行 PCA，保留 M 个主成分所捕获的方差 V_2 严格大于分别对各组做 PCA 各保留 1 个主成分的方差之和 V_1。即合并后做 PCA 的方差保留效率更高。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**FreqFold 计算流程（以 4D FreqFold 为例，g=2 heads, d_head=8）**：
```
// 原始 RoRoPE（无 FreqFold）：4 个频率索引，各自独立 PCA
// φ₁: dims {1,2} head1 + {9,10} head2 → 4D vectors → PCA → keep 1PC
// φ₂: dims {3,4} head1 + {11,12} head2 → 4D vectors → PCA → keep 1PC
// φ₃: dims {5,6} head1 + {13,14} head2 → 4D vectors → PCA → keep 1PC
// φ₄: dims {7,8} head1 + {15,16} head2 → 4D vectors → PCA → keep 1PC
// → 4 个独立 PCA，各保留 1 个主成分

// FreqFold (M=2)：将 φ₁≈φ₂, φ₃≈φ₄ 合并
// 合并组 Φ_A (φ₁+φ₂): dims {1,2,3,4} head1 + {9,10,11,12} head2
//   → 8D vectors → PCA → keep 2 PCs（对应 M=2 个主成分）
// 合并组 Φ_B (φ₃+φ₄): dims {5,6,7,8} head1 + {13,14,15,16} head2
//   → 8D vectors → PCA → keep 2 PCs
// → K_rope 现在可使用 4×d 维（而非 d 维）保留位置信息
```

**Trade-off 分析**：
- M 越大 → PCA 方差保留越多（Proposition 2） → 位置信息损失越小
- M 越大 → 频率近似的偏差越大 → RoPE 位置编码精度损失越大
- 存在 sweet spot：TransMLA 在 LLaMA-3-8B 上发现 4D FreqFold 最优（Figure 3b）

术语一般如何实现？如何使用？

FreqFold 在校准数据集上离线执行，作为 RoRoPE 的可选增强步骤。实现：对 key 激活按 RoPE 频率索引分组 → 将频率值相近（如 |θ_a - θ_b|/θ_a < threshold）的组拼接 → 在拼接后的高维向量上做 PCA → 保留 M 个主成分作为 K_rope 的多维位置表示。保留的主成分数 M = 合并的原始频率组数（如 4D FreqFold 中 M=4）。需权衡 M（更好的位置信息保留 vs 更大的近似误差）。TransMLA 实验报告：4D FreqFold 在 extreme RoPE removal (90%) 下显著优于无 FreqFold 的 RoRoPE，但过度 FreqFold（如更大 M）可能因近似误差累积而退化。

涉及论文标题：
- TransMLA: Multi-Head Latent Attention Is All You Need

---

## Balanced Key-Value PCA (BKV-PCA / 平衡键值主成分分析)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Balanced Key-Value PCA (BKV-PCA) 是 TransMLA 论文提出的对 K_nope 和 V 做联合低秩压缩时解决 norm 不平衡问题的技术。问题：经过 RoRoPE 后，K_nope（去除了 RoPE 的 key，不含第一 head）的 ℓ₂-norm 显著大于 V 的 ℓ₂-norm（因 key 保留了主要信息成分）。如果直接对 [K_nope; V] 拼接做 PCA，主成分方向会被 norm 更大的 K_nope 主导，导致 value 子空间信息在压缩中严重丢失。BKV 解决：计算平衡因子 α = E[||K_nope||₂] / E[||V||₂]，在校准数据集上将 K_nope 缩放 1/α 使两者 norm 对齐后再拼接做联合 PCA，得到平衡的低秩投影矩阵 R_KV。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**BKV-PCA 计算流程**：
```
// Step 1: 计算 norm 平衡因子
For each calibration sample x_t:
    k_nope_t = W_NoPE^{DK} @ x_t    // K_nope activation, [ (g-1)d ]
    v_t = W^{DV} @ x_t              // V activation, [ gd ]
α = mean(||k_nope_t||_2) / mean(||v_t||_2)  // 标量

// Step 2: 平衡后拼接
For each calibration sample x_t:
    k'_t = (1/α) · k_nope_t                          // 缩放 K_nope
    c_t = concat([k'_t; v_t])                         // [(2g-1)d]
// 此时 ||k'_t||_2 ≈ ||v_t||_2

// Step 3: 联合 PCA
Σ = covariance_matrix({c_t})                          // [(2g-1)d, (2g-1)d]
eigenvalues, eigenvectors = eig(Σ)                     // 按特征值降序
R_KV = eigenvectors[:, :r_kv]                          // [(2g-1)d, r_kv]

// Step 4: 低秩分解（应用于权重矩阵）
W^{DKV'} = R_KV^T @ [W_NoPE^{DK}; W^{DV}]              // [r_kv, D]
W^{UKV'} = [W_NoPE^{UK}, 0; 0, W^{UV}] @ R_KV           // [2hd, r_kv]
// 推理时仅缓存 c_t^{KV'} = W^{DKV'} @ x_t ∈ R^{r_kv}
```

**BKV 的等价性**：因 W^{UK} 相应缩放 α 倍：(1/α · W_NoPE^{DK}) × (α · W_NoPE^{UK}) = W_NoPE^{DK} × W_NoPE^{UK}，数学上等价于原始计算，不改变模型输出。BKV 仅改变 PCA 阶段的数据分布（使 K/V 的 norm 平衡），使主成分方向更均衡地捕获两者的方差。

术语一般如何实现？如何使用？

BKV-PCA 在校准数据集（WikiText-2 子集）上离线执行。实现要点：(1) α 基于校准集上期望 norm 比值计算，使用 running average；(2) BKV 后 PCA 可选择 weight-based（对 W 做 SVD）或 activation-based（对激活值做 PCA，TransMLA 证明效果更好，Figure 4b）；(3) r_kv 选择决定压缩率——KV cache 从 2gd 压缩到 r_kv + d（K_rope 的 d 维不参与压缩）。TransMLA 实验中 BKV 显著降低联合 PCA 的 perplexity 损失（Figure 4b），是 training-free 转换低损失的关键因素。与 MHA2MLA（直接联合 SVD，无 BKV）相比，BKV-PCA 在相同压缩率下性能显著更好。

涉及论文标题：
- TransMLA: Multi-Head Latent Attention Is All You Need

---

## Token Transition Variation (TTV)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

TTV（Token Transition Variation）是 TransPrune 提出的用于评估 LVLM 中视觉 token 重要性的训练无关准则。TTV 的核心思想是：token 在 Transformer 模块中传播时的表征变化（transition）幅度和方向能够反映该 token 的语义重要性，而无需依赖 token 间的 attention 计算。具体而言，对每个 Transformer 子模块（self-attention 或 FFN），TTV 同时测量两个维度：(1) **幅度变化** m = ||T_out||₂ / ||T_in||₂，衡量 token 表征在模块传递后的 L2 norm 变化率；(2) **方向变化** d = cos(T_out, T_in) = (T_out · T_in) / (||T_out||₂ · ||T_in||₂)，衡量表征向量方向的旋转程度。最终的 TTV 计算为：TTV = Softmax(1 - |d|) · m，其中 Softmax 在所有 token 上归一化方向变化值，乘以幅度变化作为权重。每层 l 的 TTV = TTV(Attention) + TTV(FFN)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**TTV 的核心计算流程**（TransPrune 在每层 Transformer 中）：

```
# 输入: visual tokens T_I [N, d], 当前层索引 l
# 当前层 self-attention 模块
T_attn_out = SelfAttention(T_I)  # 标准 QKV attention + projection
# 计算 self-attention 的 TTV
m_attn = ||T_attn_out||_2 / ||T_I||_2        # [N], 幅度变化率
d_attn = (T_attn_out · T_I) / (||T_attn_out||_2 · ||T_I||_2)  # [N], 余弦相似度
TTV_attn = Softmax(1 - |d_attn|) * m_attn    # [N], Equation (2)

# 当前层 FFN 模块
T_ffn_out = FFN(T_attn_out)  # SwiGLU 或其他 FFN
m_ffn = ||T_ffn_out||_2 / ||T_attn_out||_2
d_ffn = cos(T_ffn_out, T_attn_out)
TTV_ffn = Softmax(1 - |d_ffn|) * m_ffn

# 当前层 TTV = attention + FFN 贡献之和
TTV[l] = TTV_attn + TTV_ffn                     # Equation (3)

# Accumulation: 从 accumulation start 到当前层累积 TTV
if l in pruning_layers:
    TTV_acc = sum(TTV[j] for j in range(acc_start, l+1))  # Equation (4)
```

关键设计动机：(1) 使用 1-|d| 而非 d —— 实验中 1-|d| 效果更好（见论文 supplementary）；(2) Softmax 归一化使方向变化在不同 token 间可比；(3) 乘以 m 赋予幅度变化更大的 token 更高权重（幅度变化大的 token 通常语义更丰富）；(4) TTV 仅依赖 token 自身的输入→输出变化，不计算 inter-token 依赖，天然避免 attention 三角 mask 的 positional bias。

术语一般如何实现？如何使用？

TTV 在 TransPrune 中通过 hook Transformer 子模块的输入/输出 tensor 实现，无需修改模型结构或进行训练。具体实现要点：(1) TTV 在 accumulation layers（TransPrune 默认 layers 7-12）计算——论文实验表明中间层（而非浅层 1-6 或深层 13+）的 token transition 最能反映语义重要性（Table 10，中间层 MME^P=1540 vs 浅层 MME^P=1515）；(2) TTV accumulation 跨层累积避免单层噪声——消融实验（Table 11）显示引入 accumulation 后 MME^P 从 1530 提升到 1540；(3) TTV 额外计算开销 O(sd) 与 stage 数 s 和维度 d 线性相关，在总计算中占比可忽略；(4) 与 FlashAttention 完全兼容——TTV 仅需模块输入/输出 tensor，不访问内部 attention matrix；(5) TTV 的幅度和方向组件均带来增益，magnitude 贡献更大（Table 12：IGA+Magnitude MME^P=1532 vs IGA+Direction MME^P=1521 vs IGA+TTV MME^P=1540）。代码将开源于 https://github.com/liaolea/TransPrune。

涉及论文标题：
- TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model

---

## Instruction-Guided Attention (IGA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

IGA（Instruction-Guided Attention）是 TransPrune 中与 TTV 互补的 token 重要性评估准则。IGA 通过计算 instruction tokens 对 image tokens 的单向 attention 权重来评估每个 visual token 在给定指令下的语义相关性。与需要完整 N×N attention matrix 的传统方法不同，IGA 仅计算 instruction→image 的单向 attention（L×N 而非 N×N，L 为 instruction token 数通常仅几十个），因此计算开销极小。IGA = mean(softmax(Q_inst @ K_img^T / sqrt(d)), dim=instruction)，即对所有 instruction token 的 attention 权重取平均，得到每个 image token 的重要性分数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**IGA 的计算流程**（在 pruning layer p_i，使用下一层 p_{i+1} 的 attention weights）：

```
# 输入: Q_inst [L, d] (第 p_{i+1} 层 instruction tokens 的 query)
#        K_img [N_retained, d] (第 p_{i+1} 层保留 visual tokens 的 key)
# 输出: IGA [N_retained] (每个保留 visual token 的重要性分数)

# Step 1: 计算 instruction → image 的原始 attention scores
A_raw = Q_inst @ K_img^T           # [L, N_retained]

# Step 2: Scaled softmax (与标准 attention 一致)
A = softmax(A_raw / sqrt(d), dim=-1)  # [L, N_retained], 沿 image token 维度

# Step 3: 对所有 instruction tokens 取平均
IGA = mean(A, dim=0)               # [N_retained], Equation (5)
```

**与 TTV 的组合**（Equation 6）：
```
Score = α * TTV_acc + (1-α) * IGA  # α=0.5, 等权平衡
# 按 Score 升序排列，剪除得分最低的 tokens
```

IGA 引入任务相关的语义监督——TTV 仅依赖 token 自身的 transition 信号（与指令无关），IGA 补充了"该 token 是否与当前指令相关"的信息。消融实验（Table 12）显示：仅用 IGA 时 MME^P=1514；添加 TTV 的 magnitude 和 direction 组件后分别提升到 1532 和 1521；两者联合达到最优 1540。

术语一般如何实现？如何使用？

IGA 在 TransPrune 中的实现要点：(1) IGA 在每个 pruning layer 使用**下一层（p_i+1）**的 attention weights（因当前层的 attention 已在 forward 中计算，取下一层的 attention 在计算上更自然）；(2) IGA 不使用 accumulation 机制（仅 TTV 使用 accumulation），因为 IGA 直接反映当前指令的语义相关性，不需要"历史"信息；(3) IGA 的额外 FLOPs 为每个 pruning stage 的 L×n_i×d（L 为 instruction token 数，n_i 为当前保留的 visual token 数，d 为 hidden dim），在 VQA 场景中通常 L≈几十，与 N×N attention 相比开销极小；(4) IGA 不能完全消除 attention 的位置偏差问题——论文 Figure 4(a) 显示 IGA 仍呈现一定的首尾位置偏好（因为底层仍是 attention 计算），但通过与 TTV 结合可部分缓解（Figure 4 对比，TTV 保留 token 位置分布更均匀集中在图像中央语义区域）；(5) 参数 α 默认 0.5 使 TTV 和 IGA 等权贡献——Table 13 消融显示 α=0.4 时 MMB^en=65.5，α=0.5 时 MMB^en=66.0，α=0.6 时 MMB^en=65.9，α=0.5 最优。

## Sinkhorn Normalization (for Balanced Group Assignment in Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sinkhorn Normalization 是一种将任意非负矩阵迭代变换为双随机矩阵（doubly stochastic matrix）的算法，在 Focus 论文中被用于强制注意力分组均衡。算法流程：给定 token-group 得分矩阵 S ∈ R^{T×K}（T 个 token，K 个 group），首先 Q ← exp(S/τ)（temperature τ 控制软硬程度），然后交替进行行归一化（Q ← Q / sum(Q, dim=tokens)，使每个 token 的总 assignment 为 1）和列归一化（Q ← Q / sum(Q, dim=groups)，使每个 group 的总 mass 均衡），迭代 N 次后 Q 近似双随机——所有行和列的和均为 1。与 softmax 归一化（仅行归一化，无列约束）不同，Sinkhorn 阻止任何单个 group 吸收所有 token（group dominance），同时仍允许 LM 梯度学习哪个 token 属于哪个 group。在 Focus 中 N=10 次迭代足以产生平衡分组，τ=0.1 控制 assignment 的软硬程度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Focus 的每个 attention 层，Sinkhorn 归一化作为 group assignment 的核心步骤，位于 token-to-centroid 亲和度计算之后、门控注意力之前：

```
# 输入: h [T, d] hidden states, C [K, d_g] centroids, W_g [d, d_g]
# 输出: g [T, K] doubly-stochastic group assignments

def sinkhorn_group_assignment(h, C, W_g, tau=0.1, N=10):
    # Step 1: 投影到 centroid 空间
    S = (W_g @ h.T).T @ C.T    # [T, K] token-centroid 亲和度得分
    
    # Step 2: Sinkhorn 迭代
    Q = exp(S / tau)            # [T, K] 指数化
    for i in range(N):
        Q = Q / Q.sum(dim=0, keepdim=True)    # 列归一化: 均衡 group mass
        Q = Q / Q.sum(dim=1, keepdim=True)    # 行归一化: 每个 token sum=1
    
    # Q 现在是近似双随机矩阵
    # 每行: token i 对各 group 的软分配
    # 每列: 各 group 的 token 质量均衡 (≈T/K)
    return Q  # [T, K]
```

Pipeline 中 Sinkhorn 的位置：
1. token hidden states → W_g 投影 → centroid 空间 (d_g=16)
2. 计算 token-centroid 亲和度得分 S
3. **Sinkhorn 归一化** → 双随机 group assignment g
4. g 用于门控注意力: s_ij = q_i^T k_j · (1_local + (1-1_local) · σ(λ · g_i^T g_j))
5. 仅同组远距离 token 参与注意力

Sinkhorn 阻止三条 group dominance escape pathway:
- Path A (centroid drift): 即使 centroid 漂移导致所有 token 偏向同一 centroid，列归一化强制重新分配
- Path B (representational bypass): 即使 hidden states 偏移，行归一化保持 per-token assignment 分布
- Path C (projection bypass): 即使 W_g 映射所有 token 到同一方向，双随机约束仍强制均衡

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Sinkhorn 归一化最初来自最优传输（optimal transport）理论，用于高效近似计算 Wasserstein 距离。在深度学习中的使用方式：
- Focus 论文中实现为 Python/PyTorch，每次前向传播在 attention 层内执行 10 次迭代的行/列归一化
- 温度 τ 控制 assignment 的置信度：τ=0.1 时 assignment 接近 hard（高置信度），τ 增大则趋于均匀
- N=3 次迭代不足以均衡（尤其在低 τ 下 exp(scores/τ) 分布极尖锐），论文推荐 N≥10
- 与 softmax + entropy loss 方法对比：Sinkhorn 是结构约束（非软损失），因此不依赖梯度来学习均衡——即使梯度推动 collapse，Sinkhorn 迭代仍强制重新分布
- 超参稳健性：Table 9 显示 fine-tuned PPL 在 16 种配置下仅波动 0.6

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

## Learned Centroids for Attention Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Learned Centroids for Attention Routing 是 Focus 论文的核心机制：在预训练 Transformer 的每个 attention 层中添加少量可学习的 centroid 向量 C ∈ R^{K×d_g}（K 个 group，d_g 维 centroid 空间）和一个轻量投影矩阵 W_g ∈ R^{d×d_g}，用于将 token 分配到语义 group，进而控制哪些 token pair 可以互相关注（routing）。关键设计原则是 separation of routing and attention：centroid 仅决定"谁可以关注谁"（routing 决策），QKV 注意力决定"关注多少"（content 传输）。预训练 QKV 权重完全冻结，centroid 参数低至 148K（d_g=16 时仅占模型 0.1%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Centroid 在 Focus 推理 pipeline 中的角色：

```
# 每个 attention 层
def focus_attention_layer(h, Q_weight, K_weight, V_weight, O_weight, C, W_g):
    # === Routing Phase (centroid-only, 可训练) ===
    # d_g=16 维投影，仅需 148K 参数
    token_repr = h @ W_g.T              # [T, d_g]
    group_scores = token_repr @ C.T     # [T, K]  token-centroid 亲和度
    g = sinkhorn(group_scores / τ)      # [T, K]  双随机 group assignment
    
    # === Content Phase (预训练权重, 冻结) ===
    q = h @ Q_weight.T                  # [T, d_head]
    k = h @ K_weight.T                  # [T, d_head]
    v = h @ V_weight.T                  # [T, d_head]
    
    # === Gated Attention ===
    for i in range(T):
        for j in range(i+1):
            if i - j <= w:              # 局部窗口 → 全注意力
                s_ij = q_i @ k_j.T
            else:                       # 远距离 → 组门控
                gate = sigmoid(λ * g[i] @ g[j].T)  # 同组≈1, 异组≈0
                s_ij = q_i @ k_j.T * gate
    
    attn_out = softmax(s) @ v          # 仅有效 pair 参与 softmax
    return attn_out @ O_weight.T
```

关键设计决策：
- **d_g=16**：token grouping 是低维任务，16 维足矣。d_g=768（全维）vs d_g=16，参数差 50 倍（7.1M vs 148K），PPL 无差异（均为 34.5）
- **K=4 或 8**：group 数量。K=4 时每个 token 属于 2 个 group（top-k=2），约 50% 远距离 pair 被剪枝，2× 加速；K=8 时 8.6× 加速（1M token）
- **λ**：gate steepness 参数，控制 sigmoid 的锐度

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Focus centroid 的使用方式：
- **Composable training**（仅 centroid 训练）：加载预训练模型 → 每层插入 C 和 W_g → 冻结所有原权重 → 仅训练 centroid 参数（4000 steps on PG-19 with GPT-2 124M）
- **Full fine-tuning**（两阶段）：Phase 1 仅 centroid 训练建立 group 结构 → Phase 2 解冻所有权重联合微调
- **Inference**：hard top-k assignment（每个 token 选 top-k 个 group），仅同组 token pair 计算注意力
- 从零训练：centroid + 标准 QKV 权重一起随机初始化、一起训练（Mistral 7B from scratch on 2B tokens）
- 跨架构通用：MHA、GQA、GQA+bias、MHA+QK-norm、interleaved+softcap 五种 attention 架构均适用

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

## Group Dominance (in Sparse Attention Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Group Dominance 是 Focus 论文发现的一种训练不稳定性：当使用 softmax 归一化训练 centroid-based 稀疏注意力时，一个 group 在约 600 步内吸收所有 token，导致稀疏性崩溃——Focus 退化为昂贵的 full attention。这类似于 Mixture of Experts 中的 expert collapse / load imbalance 问题（Fedus et al., 2022），但发生在注意力路由而非 FFN 路由中。论文识别出三条独立 escape pathway：(A) centroid drift——LM 梯度推动 centroid 漂移使所有 token 匹配同一 centroid；(B) representational bypass——即使 centroid 冻结，hidden states 也会向同一 centroid 方向偏移；(C) projection bypass——即使 centroids 和 hidden states 都被约束，W_g 投影也会学习将所有 token 映射到同一方向。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Group dominance 的根本原因：full attention 始终最小化 training loss（因模型可访问所有 token），因此梯度总是推动移除注意力限制。这形成了一个悖论——稀疏注意力在推理时提升质量，但训练时梯度推动消除稀疏性。

三种 escape pathway 的关系：
```
       训练前的均衡状态 (K=8, 每组 ~12.5%)
                |
       ┌───────┼───────┬──────────────┐
       v       v       v              v
   Path A   Path B   Path C         Combined
   centroid hidden    W_g            full FT
   drift    shift     collapse       all active
       |       |       |              |
       v       v       v              v
   1 group absorbs all tokens → sparsity lost
```

Dominance 度量：最大 group 中的 token 占比。K=8 时完美均衡 = 12.5%，collapse = 100%。

论文尝试的缓解方法及其失败原因：
- **Entropy + balance loss**：仅处理 Path A，第 600 步 collapse
- **Stop-gradient on inputs**：阻断 Path B 但不阻断 A/C
- **EMA centroids + detached projection**：阻断 A 但 projection 抹除结构 (Path C)
- **Periodic reclustering (每 100 步)**：周期性重置平衡但 group 不稳定
- **Balance weight ×5**：8 个 group 中 6 个死亡
- **Sinkhorn (论文方案)**：同时阻断三条路径——即使 centroids 漂移/representations 偏移/projection collapse，Sinkhorn 迭代强制重新分布

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Group dominance 的检测与防御：
- 监控每个 group 的 token 占比，发现某个 group 持续超过 30-40%（K=8 时代）即为 dominance 信号
- Sinkhorn 归一化作为结构性约束（非软损失），在每次前向传播中强制执行均衡分组
- Sinkhorn 在 full fine-tuning（最严峻的测试）中保持 15.9% dominance（K=8，完美 = 12.5%），而 softmax 方案 collapse 到 99.4%
- Sinkhorn 对超参稳健：16 种配置下 fine-tuned PPL 波动仅 0.6

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

## Group-Gated Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Group-Gated Attention 是 Focus 论文提出的注意力门控机制：将标准 QK^T 注意力分数乘以一个 group-based gate，使得远距离的 token pair 仅在属于同一 learned group 时才参与注意力。Gate 公式：s_ij = q_i^T k_j · (1_local(i,j) + (1 - 1_local(i,j)) · σ(λ · g_i^T g_j))，其中 1_local 表示 i-j ≤ w（局部窗口），g_i 为 token i 的 group assignment 向量，σ 为 sigmoid。对局部 token，gate=1（全注意力）；对远距离 token，同组 gate≈1（保留注意力），异组 gate≈0（剪枝注意力）。Gate 仅决定信息是否流动（binary routing），q_i^T k_j 决定流动多少（content weighting）。这与 token selection 方法不同——selection 挑选 top-k token 但使用标准 softmax；Focus gate 在 softmax 之前将异组 pair 缩放到零。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 标准 attention: s_ij = q_i^T k_j  (所有 T^2 个 pair)
# Group-gated attention: 
for i in range(T):
    for j in range(i+1):   # causal
        if i - j <= w:
            # 局部窗口内 → 全注意力 (gate = 1)
            s_ij = q_i^T @ k_j
        else:
            # 远距离 → 组门控
            affinity = g[i] @ g[j]        # 同组 ≈ 1.0, 异组 ≈ 0.0
            gate = sigmoid(lambda * affinity)
            s_ij = (q_i^T @ k_j) * gate    # 异组 → s_ij ≈ 0

attn_weights = softmax(s, dim=-1)          # 异组 pair 权重 ≈ 0
output = attn_weights @ V
```

门控机制的关键特性：
- **不重归一化（no re-normalization）**：异组 pair 被 gate 缩放到零，但 softmax 仍在全部 token 上归一化（包括零权重对）——这保留了预训练模型的 softmax 分母统计，是 composability 的来源之一
- **Gate steepness λ**：控制同组/异组的区分锐度。λ 过小 → 门控无力；λ 过大 → 近似 hard assignment
- **Local window 豁免**：局部 token 无论 group 归属均保留全注意力，保证短程依赖不丢失

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Group-gated attention 的推理效率优化：
- 训练时：soft gate 计算全部 O(n²) pair，无训练加速
- 推理时：使用 hard top-k assignment，异组远距离 pair 被完全剪枝（不计算），同组 pair 通过 FlashAttention 分解加速
- K=4, top-k=2: 约 50% 远距离 pair 被保留，约 25% 总 pair 被计算 → 2× 加速
- K=8, top-k=1: 约 12.5% pair 保留 → 理论 8× 加速，实际 8.6×（因 FlashAttention 在短序列上更高效）
- 门控的质量效应：top-k=2 时 PPL 优于 top-k=3/4（更稀疏产生更好质量），验证了"less attention can be more"的核心主张

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

## Composable Sparse Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Composable Sparse Attention 是 Focus 论文定义的一种新的高效注意力范式：稀疏注意力方法可以**叠加到任何预训练模型上**（类似插件），仅训练稀疏路由参数，所有原始权重保持冻结，且叠加后不退化任何下游 benchmark。这是与其他高效注意力方法的关键区别——结构化稀疏（Longformer）固定 pattern 无法适配、近似方法（Performer）误差累积、token selection（SparQ）退化 PPL 5-10 点——而 composable 方法要求零 benchmark 退化 + 改善或匹配 PPL。Focus 实现了这一性质的核心原因是 routing-attention separation：centroid 仅决定路由，预训练 QKV 注意力完整保留，因此模型不会"忘记"任何预训练知识。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Composability 的技术基础：
```
原始预训练模型: h → Q(h), K(h), V(h) → softmax(QK^T/√d)·V
Composable Focus: 
  h → W_g·h · C^T → Sinkhorn → g (group assignment, 仅 148K 新参数)
  h → Q(h), K(h), V(h) (原权重, 冻结)
  → group-gated attention = softmax(QK^T · gate(g))·V
```

Composability 的三层保证：
1. **权重冻结**：centroid 训练不修改 Q/K/V/O 权重，不破坏预训练表征
2. **精确 softmax 保留**：同组内仍使用标准 softmax（无近似、无重归一化），预训练计算模式不变
3. **Routing-content 分离**：路由（centroid）与内容（QKV attention）独立，互不干扰

验证范围：GPT-2 124M/774M、Mistral 7B、Qwen2.5 7B、OLMo-27B、LLaMA-2 13B/70B 七种模型五种 attention 架构，all benchmark 零退化（最差 -0.3%，噪声范围内）。

与 LoRA 对比：LoRA 修改权重矩阵（ΔW=AB），在相同参数预算下（147K vs 148K）退化所有 benchmark；Focus 仅添加路由不修改任何原权重，零退化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Composable 部署流程：
1. 加载任意预训练 HuggingFace 模型
2. 对每个 attention 层插入 centroid 参数（C, W_g）
3. 训练：仅 centroid 参数更新（AdamW, 4000 steps），所有原权重 requires_grad=False
4. 推理：使用 hard top-k assignment + FlashAttention 分解，获得加速
5. 跨模型可迁移：GPT-2 上训练的 centroid 思想可直接应用于 Mistral/Llama/Qwen，仅需重新训练 centroid 参数

限制：
- 训练时无加速（soft gate 计算全 O(n²)）
- ≤4K token 时路由开销（sort ~12ms）抵消加速
- 大模型（≥7B）上 PPL 收益递减（从 +1.1 降至 -0.7）

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

## Softmax Dilution

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Softmax Dilution 是 Focus 论文附录 A 中解释"为何稀疏注意力能超越全注意力"的三个机制之一：在全注意力中，softmax 将概率质量分布在全部 n 个 token 上。当一个 token（如位置 800 的代词）需要关注其语义相关 token（如位置 200 的先行词）时，它必须与其余 n-2 个无关 token 竞争 softmax 概率质量。结果是对相关 token 的注意力权重被稀释，每个无关 token 都从真正重要的 pair 那里夺取一小部分注意力权重。Focus 通过限制 softmax 到同组 token + 局部窗口，将竞争集合从 n 缩小到约 n/K + w，概率质量集中在更小但更相关的候选集上，产生更锐利、更有信息量的注意力分布。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Softmax dilution 的数学本质：
```
全注意力:
  scores = [q_i·k_1, q_i·k_2, ..., q_i·k_n]    # n 个 score
  weights = softmax(scores)                       # n 个权重, sum=1
  # 当 n 很大时, 即使最强的 score 也只得到 1/n ~ 少量概率质量
  # 每个无关 token 贡献: weight_j * v_j (虽小但 n 个累积)

Focus:
  scores = [q_i·k_j for j where g(i)=g(j) or |i-j|≤w]  # ~n/K + w 个
  weights = softmax(scores)                              # ~n/K + w 个权重
  # 竞争集缩小 ~K 倍, 关键 token 获得更多概率质量
```

论文 Appendix A 给出的三个机制：
1. **Softmax dilution**（上述）：减少竞争 token 数量，集中概率
2. **Noise removal**：无关 KV pair 不仅浪费计算，还主动降质——每个无关 token 向输出添加微量噪声，12 层 × 12 heads 累积后显著。Focus 完全消除这些 pair
3. **Implicit structural constraint**：全注意力在小模型（124M）上会记忆训练数据中的虚假长程相关性。限制注意力到语义相关 group 充当结构性先验（类似 L1 正则化），防止过拟合噪声注意力 pattern

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Softmax dilution 的实践意义：
- 解释了为什么 top-k=2（约 50% 远距离 pair）产生比 top-k=3/4（更多 pair）更好的 PPL——更多 pair 重新引入 dilution
- 在小模型（124M）上效果最显著：Focus FT 30.3 vs full attention FT 31.4 (+1.1 PPL)
- 在大模型（7B+）上效果递减：更大模型天然有更强能力区分相关/无关 token，dilution 影响较小
- 设计启示：稀疏注意力的目标不是近似全注意力，而是通过消除稀释和噪声来超越它

涉及论文标题：
- Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

涉及论文标题：
- TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model

## Attention Upcycling（注意力上循环/注意力升级）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention Upcycling（注意力上循环/注意力升级）指在相对较小的额外计算预算下，将预训练 Transformer 中的现有注意力模块（MHA、GQA）升级为更高效的注意力形式（如 MLA、linear attention、Mamba/SSM），而无需完整的重新预训练。该术语源自 sparse upcycling（Komatsuzaki et al., 2022 将 dense 模型转换为 MoE），在注意力上下文中特指：(1) 从预训练注意力权重中提取知识（如通过 SVD 分解）；(2) 将提取的低秩结构用于初始化新注意力模块的参数；(3) 通过轻量级训练（知识蒸馏、DPO 或连续预训练）弥合架构差异。

逻辑链：预训练模型在自注意力权重 W^Q、W^K、W^V 中积累了丰富的"dark knowledge" → 通过 SVD 等矩阵分解方法，将这些权重投影到低秩空间，提取最具信息量的主成分 → MLA 等高效注意力形式的降维/升维矩阵由此初始化（而非随机初始化）→ 最后通过知识蒸馏，由 teacher 模型指导 student 模型适应新的低秩注意力模式。相比于从零开始的 pre-training（DeepSeek-V3 需 2.664M H800 GPU hours），attention upcycling 仅需数十至数百 GPU hours。

从算法pipeline角度拆解术语，给出具体例子。

**X-EcoMLA Attention Upcycling Pipeline（以 Llama3.2-1B GQA → MLA 为例）：**

```
# === Stage 0: 从预训练权重初始化 MLA ===
# 输入: GQA 权重 W_Q[d, n_h*d_h], W_K[d, n_kv*d_h], W_V[d, n_kv*d_h]
# 若为 GQA，先将 K/V 复制到与 Q head 数一致：W_K, W_V → [d, n_h*d_h]

# 1. Query 侧 SVD 初始化
U_q, Σ_q, V_q^T = SVD(W_Q)           # W_Q = U_q Σ_q V_q^T
W_DQ = U_q                           # [d, r_q] 查询下投影
W_UQR_bar = (Σ_q @ V_q^T).view(r_q, n_h, d_h)
W_UQ = W_UQR_bar[:,:,:d_qk].view(r_q, n_h*d_qk)   # NoPE 查询上投影
W_QR = W_UQR_bar[:,:,-d_r:].view(r_q, n_h*d_r)    # RoPE 查询上投影

# 2. KV 侧 Joint SVD 初始化
W_KV = concat(W_K, W_V, dim=-1)      # [d, 2*n_h*d_h]
U_kv, Σ_kv, V_kv^T = SVD(W_KV)
W_DKV = U_kv                         # [d, r_kv] KV 下投影
W_UKV = Σ_kv @ V_kv^T                # [r_kv, 2*n_h*d_h]
W_UK = W_UKV[:, :n_h*d_h].view(r_kv, n_h, d_h)[:,:,:d_qk].view(r_kv, n_h*d_qk)
W_UV = W_UKV[:, n_h*d_h:]            # value 上投影

# 3. RoPE Key 初始化（所有 head 共享）
W_K_avg = W_K.view(d, n_kv, d_h).mean(dim=1)  # [d, d_h]
W_KR = W_K_avg[:, -d_r:]                      # [d, d_r]

# === Stage 1: 端到端知识蒸馏 ===
for batch in SFT_dataloader:          # OpenHermes + GenQA + Infinity-Instruct (~6.8B tokens)
    student_logits = X_EcoMLA(batch.input_ids)
    teacher_logits = frozen_teacher(batch.input_ids)
    loss = KL(teacher_logits || student_logits)  # KL 散度损失
    optimizer.step()

# === Stage 2: DPO 偏好对齐 ===
for batch in DPO_dataloader:          # ultrafeedback + orca_dpo (~0.2B tokens)
    π_student = X_EcoMLA(); π_ref = copy(X_EcoMLA).freeze()
    loss = -log σ(β[log(π_student(y_w)/π_ref(y_w)) - log(π_student(y_l)/π_ref(y_l))])
    optimizer.step()
```

术语一般如何实现？如何使用？

Attention Upcycling 的实现关键包括：(1) **权重初始化策略**——SVD 初始化（X-EcoMLA）、Joint SVD（MHA2MLA）、随机初始化+蒸馏（MOHAWK）；(2) **训练策略**——端到端 KL 蒸馏（X-EcoMLA, MambaInLlama）、连续预训练（GQA upcycling）、中间层蒸馏（MOHAWK）、DPO 偏好微调（X-EcoMLA）；(3) **架构映射**——MHA→MLA（X-EcoMLA, MHA2MLA, TransMLA）、MHA→GQA（Ainslie et al. 2023）、MHA→Linear Attention（Hedgehog）、MHA→Mamba/SSM（MambaInLlama, MOHAWK）。

使用场景：需要在不牺牲模型精度的前提下大幅压缩 KV cache 以降低推理显存成本，但无法承受从零预训练的计算开销。典型应用如将 Llama3.2-1B 升级为 MLA 仅需 70 GPU hours on MI300，而预训练原模型需 370K GPU hours——约 5000× 的训练成本差异。

涉及论文标题：
- X-EcoMLA__Upcycling_Pre-Trained_Attention_into_MLA_for_Efficient_and_Extreme_KV_Compression

---

## SVD-based Weight Initialization for MLA Upcycling（面向 MLA 上循环的 SVD 权重初始化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

SVD-based Weight Initialization for MLA Upcycling 是 X-EcoMLA 提出的将预训练 MHA/GQA 注意力权重通过奇异值分解（SVD）初始化为 MLA 参数的方法。核心思路：MHA 的 W^Q、W^K、W^V 与 MLA 的 down/up-projection 矩阵在数学上近似等价（MLA 是 MHA 的低秩近似），通过 SVD 提取预训练权重中能量最高的主成分方向，将其直接赋值给 MLA 的各投影矩阵，使 MLA 在训练开始时就继承了预训练模型的大部分知识（dark knowledge），显著优于随机初始化。

逻辑链：预训练的 W^Q（或 W^K, W^V）矩阵包含了模型学到的 token 语义投影方向 → SVD 分解为 W = UΣV^T，U 的列向量是输入空间的正交基（down-projection 方向），ΣV^T 的行向量是输出方向的加权组合 → U 直接作为 down-projection W^{DQ}（或 W^{DKV}），ΣV^T 重塑后分割为各 up-projection 矩阵 → 训练开始时 MLA 的输出已接近原始 MHA（忽略位置编码差异），知识蒸馏仅需微调弥合残余差异。

从算法pipeline角度拆解术语，给出具体例子。

**X-EcoMLA SVD 初始化伪代码（对应论文 Algorithm 1）：**

```
# 输入: MHA/GQA 权重 W_Q, W_K, W_V ∈ R^{d × n_h·d_h}
# 参数: r_q, r_kv (KV rank), d_qk (query-key dim), d_r (RoPE dim)
# 输出: MLA 权重

# === 1. Query 侧 ===
U_q, Σ_q, V_q = SVD(W_Q)                            # 经济型 SVD
W_DQ = U_q[:, :r_q]                                  # [d, r_q]
W_UQR_bar = (Σ_q[:r_q,:r_q] @ V_q[:r_q,:]).view(r_q, n_h, d_h)
W_UQ = W_UQR_bar[:, :, :d_qk].reshape(r_q, n_h * d_qk)  # NoPE query
W_QR = W_UQR_bar[:, :, -d_r:].reshape(r_q, n_h * d_r)   # RoPE query

# === 2. KV Joint SVD ===
W_KV = torch.cat([W_K, W_V], dim=-1)                 # [d, 2·n_h·d_h]
U_kv, Σ_kv, V_kv = SVD(W_KV)
W_DKV = U_kv[:, :r_kv]                               # [d, r_kv]
W_UKV = Σ_kv[:r_kv,:r_kv] @ V_kv[:r_kv,:]            # [r_kv, 2·n_h·d_h]

# Key up-proj: 取前 n_h*d_h 列，每 head 取前 d_qk 维
W_UK_bar = W_UKV[:, :n_h*d_h].view(r_kv, n_h, d_h)
W_UK = W_UK_bar[:, :, :d_qk].reshape(r_kv, n_h * d_qk)

# Value up-proj: 取后 n_h*d_h 列（全 d_h 维）
W_UV = W_UKV[:, n_h*d_h:]                            # [r_kv, n_h*d_h]

# === 3. 共享 RoPE Key ===
W_K_reshaped = W_K.view(d, n_kv_heads, d_h)          # [d, n_kv, d_h]
W_K_avg = W_K_reshaped.mean(dim=1)                    # 所有 KV head 平均 → [d, d_h]
W_KR = W_K_avg[:, -d_r:]                              # 取最后 d_r 维 → [d, d_r]

# === 4. 其他参数（W_O, FFN 等）===
# 直接从预训练模型复制，不做 SVD 分解
```

**Joint SVD 与 Separate SVD 的区别**：
- Separated SVD: 分别对 W^K 和 W^V 做 SVD，各取 r_kv/2 个主成分 → 丢失 K 和 V 之间的相关性
- Joint SVD: 将 [W^K, W^V] 拼接后做统一 SVD → 捕获 K-V 联合空间的低秩结构，保真度更高

术语一般如何实现？如何使用？

SVD 初始化使用标准的 `torch.linalg.svd` 或 `numpy.linalg.svd` 即可实现（经济型 SVD，仅计算前 r 个奇异向量）。对于大数据模型（如 Llama3-8B），完整 SVD 计算开销可忽略（远小于训练时间），且仅需执行一次。与 ASVD（Activation-aware SVD）不同，X-EcoMLA 的 SVD 直接作用于权重矩阵本身（而非用激活值校准），因此在没有校准数据的情况下也能工作。

涉及论文标题：
- X-EcoMLA__Upcycling_Pre-Trained_Attention_into_MLA_for_Efficient_and_Extreme_KV_Compression

---

## Dynamic Rank Selection (Energy-based)（基于能量的动态秩选择）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dynamic Rank Selection 是 X-EcoMLA 中提出的一种基于奇异值累积能量的自适应 rank 选择策略。与 Fixed Rank Selection（所有 transformer 层使用统一的 r_q 和 r_kv）不同，Dynamic Rank Selection 根据每层权重矩阵的奇异值分布，自动确定该层所需的 rank 值——信息丰富的层（奇异值衰减慢）自动分配更高 rank，冗余的层（奇异值衰减快）分配更低 rank。

逻辑链：对 W^Q（或 [W^K, W^V]）做 SVD 得到奇异值序列 σ_1 ≥ σ_2 ≥ ... ≥ σ_min(d, n_h·d_h) → 计算总能量 E = Σ σ_j² → 设定能量阈值 δ（如 0.90 或 0.95）→ 选择最小的 rank R 使得 Σ_{j=1}^R σ_j² ≥ δ·E → R 是该层在保留 δ 比例信息的条件下的最优低秩维度。

从算法pipeline角度拆解术语，给出具体例子。

**动态 Rank 选择算法：**

```
def dynamic_rank_selection(W, delta=0.95):
    """
    W: weight matrix [d_out, d_in]
    delta: energy preservation threshold (0 < delta <= 1)
    """
    _, Σ, _ = torch.linalg.svd(W, full_matrices=False)  # 经济型 SVD
    energies = Σ ** 2                                     # 奇异值平方
    total_energy = energies.sum()                         # 总能量 E
    cumulative_energy = torch.cumsum(energies, dim=0)     # 累积能量
    # 找到最小 rank 使得累积能量 >= delta * total_energy
    mask = cumulative_energy >= delta * total_energy
    rank = mask.nonzero(as_tuple=True)[0][0].item() + 1  # 1-indexed
    return rank

# 使用
r_q_i = dynamic_rank_selection(W_Q_layer_i, delta_q)      # 第 i 层 Q 的 rank
r_kv_i = dynamic_rank_selection(
    torch.cat([W_K_layer_i, W_V_layer_i], dim=-1), 
    delta_kv
)                                                          # 第 i 层 KV 的 rank
```

**X-EcoMLA 实验中的动态 rank 选择效果（论文 Table 1、Table 6-7）**：
- δ=0.95 时 KV size 约 54.7%（Llama3.2-1B），Avg score 53.12（vs baseline 52.85）——动态 rank 自动平衡了压缩与精度的 tradeoff
- 动态 rank 效果与固定 rank 相当或略优，但无需手动调参，对超参数不敏感

术语一般如何实现？如何使用？

Dynamic Rank Selection 可与 Fixed Rank Selection 互换使用（论文 Table 10 显示二者性能接近）。实际使用中：(1) 先用动态 rank 确定每层的 r_q、r_kv 值（一次性计算，耗时极短）；(2) 然后按确定的 rank 值进行 SVD 初始化；(3) 后续训练与固定 rank 相同。能量阈值 δ 通常设为 0.85-0.95（论文中的典型配置）。

涉及论文标题：
- X-EcoMLA__Upcycling_Pre-Trained_Attention_into_MLA_for_Efficient_and_Extreme_KV_Compression

---

## Joint SVD for KV Weights（KV 权重的联合 SVD）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Joint SVD for KV Weights 是 X-EcoMLA 中用于初始化 MLA 的 W^{DKV}、W^{UK}、W^{UV} 矩阵的 SVD 策略。与分别对 W^K 和 W^V 做 SVD 不同，Joint SVD 先将 W^K 和 W^V 沿列方向拼接为 [W^K, W^V] ∈ R^{d × 2·n_h·d_h}，再对拼接矩阵执行统一 SVD 分解，从而捕获 Key 和 Value 之间的相关性信息。

逻辑链：MHA/GQA 中 W^K 和 W^V 独立训练，但它们处理相同的 hidden state H，存在隐式的跨空间关联 → 拼接后做 SVD，U_kv 提取的是同时对 K 和 V 投影方向都重要的输入方向 → W^{DKV} = U_kv（共享 down-projection）→ Σ_kv V_kv^T 的前 n_h·d_h 列用于构造 W^{UK}，后 n_h·d_h 列用作 W^{UV} → 相比分别 SVD，Joint SVD 在相同 r_kv 下能更好地保留原始attention的KV联合信息。

从算法pipeline角度拆解术语：

```
# Joint SVD (X-EcoMLA)
W_KV = concat(W_K, W_V, dim=-1)      # [d, 2*n_h*d_h] —— 拼接
U, Σ, V^T = SVD(W_KV)                # 统一分解
W_DKV = U[:, :r_kv]                   # 共享 down-proj
W_UKV = Σ[:r_kv,:r_kv] @ V^T[:r_kv,:]  # [r_kv, 2*n_h*d_h]
W_UK = W_UKV[:, :n_h*d_h]   (截取前部) # key up-proj
W_UV = W_UKV[:, n_h*d_h:]   (截取后部) # value up-proj

# vs. Separate SVD (MHA2MLA 的变体)
U_k, Σ_k, V_k^T = SVD(W_K)           # 分别分解 W_K
U_v, Σ_v, V_v^T = SVD(W_V)           # 分别分解 W_V
# W_UK 仅来自 W_K 的信息，W_UV 仅来自 W_V 的信息
# 丢失了 K 和 V 之间的联合结构
```

术语一般如何实现？如何使用？

Joint SVD 的实现与普通 SVD 完全相同，仅需在调用 `torch.linalg.svd()` 前做一次 `torch.cat()`。对于大数据模型（d=4096, n_h=32, d_h=128），拼接后矩阵维度为 [4096, 8192]，经济型 SVD 的复杂度约 O(d × (2·n_h·d_h) × r_kv)，在 GPU 上计算时间 <1 秒/层。Joint SVD 特别适用于 GQA-based 模型（如 Llama 系列），因为 W_K 和 W_V 的维度天然较小。

涉及论文标题：
- X-EcoMLA__Upcycling_Pre-Trained_Attention_into_MLA_for_Efficient_and_Extreme_KV_Compression

## Dynamic Sparsity in Attention (注意力动态稀疏度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Dynamic Sparsity（动态稀疏度）指注意力计算中稀疏度（sparsity ratio / density）不是固定值，而是根据输入内容、序列长度、注意力头特性等因素自适应变化的策略。

在 XAttention 中，动态稀疏度通过 Threshold Block Selection 和 Minimum Threshold Prediction 两个机制共同实现：(1) Threshold-based 选择天然产生动态稀疏度——不同输入产生不同的反对角线分数分布，累计概率超过 τ 所需的 block 数自然不同；(2) 长序列的注意力天然更稀疏（信息分散在更多 token 上），阈值方法自动适配——128k 序列密度 ~6.89%，4k 序列密度 ~52.16%；(3) Per-head threshold optimization 进一步引入头间差异——不同功能头（retrieval head vs. streaming head）天然有不同的稀疏特性。

从算法pipeline角度拆解术语：

```
# 动态稀疏度的自适应行为

# 场景1: 短序列 (4k tokens)
# 注意力相对密集——信息集中，需更多 block 参与
# τ=0.9 时 density ≈ 52%，每个 query block 关注 ~33 个 key blocks

# 场景2: 长序列 (128k tokens)
# 注意力高度稀疏——信息分散，仅少数 block 含有效信息
# τ=0.9 时 density ≈ 6.89%，每个 query block 仅关注 ~141 个 key blocks

# 场景3: Per-head variation
# Head A (retrieval): 关注特定位置，稀疏度高 → τ_A=0.95
# Head B (streaming): 关注连续区域，稀疏度低 → τ_B=0.75
```

与固定稀疏度方法（Top-K: 固定 K 个 block；Top-Ratio: 固定比例）的对比（XAttention Table 8）：固定方法在短序列浪费计算（保留过多 block），在长序列丢失信息（保留不足），且无法适应不同输入内容。动态阈值按 attention mass 保留，自动匹配实际信息分布。

术语一般如何实现？如何使用？

实现方式：(a) 基于累积 softmax 概率的 Threshold Block Selection——无需预设任何稀疏度参数，仅需一个全局阈值 τ；(b) 离线 DP 搜索 per-head τ 值——一次性搜索后保存为配置文件，推理时零额外开销；(c) 也可以采用更简单的方案——所有头使用相同 τ（论文的默认 baseline，τ=0.9）。

与其他方法的动态稀疏度对比：MInference 通过 Kernel-Aware Search 为每个头分配固定稀疏模式但参数固定（k_v, k_s 不变）；FlexPrefill 使用 coverage α 参数控制动态 budget 分配但效果有限；XAttention 的 τ-based 方法是最直接的——累积概率超过阈值即停止，稀疏度完全由输入数据的注意力分布自然决定。

涉及论文标题：
- XAttention: Block Sparse Attention with Antidiagonal Scoring

## Residual Token Merging (残差 Token 合并 for KV Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Residual Token Merging 是 ZSMerge 提出的 KV Cache 压缩中的核心机制——将"被驱逐的 token"通过相似度驱动的增量均值聚合动态合并入有限个残差 slot，而非永久丢弃。与纯驱逐方法的根本区别在于：(1) 驱逐方法永久删除 KV 对，导致不可逆信息损失和 attention distribution drift；(2) 残差合并将驱逐转换为压缩编码——用 Br 个 slot 表示大量被驱逐 token 的聚合信息。

合并流程三步（per evicted token (k_t, v_t))：
1. **slot 选择**（Eq. 6）：通过最大 key 内积选择最兼容的 residual slot：$\hat{r} = \arg\max_{r \in \{1,\dots,B_r\}} \mathbf{k}_r^\top \mathbf{k}_t$。内积度量 key 向量的方向相似度。
2. **增量均值聚合**（Eq. 7）：$\mathbf{k}_{\hat{r}} \leftarrow \frac{w_{\hat{r}}\mathbf{k}_{\hat{r}} + \mathbf{k}_t}{w_{\hat{r}} + 1}, \quad \mathbf{v}_{\hat{r}} \leftarrow \frac{w_{\hat{r}}\mathbf{v}_{\hat{r}} + \mathbf{v}_t}{w_{\hat{r}} + 1}$——滑动平均格式，O(d) 增量更新无需存储历史 token。
3. **权重递增**：$w_{\hat{r}} \leftarrow w_{\hat{r}} + 1$，记录 slot r 已合并的 token 数量。

每个 attention head 独立维护 B_r 个残差 slot（K_r, V_r ∈ R^{B_r×d}）和权重向量 w ∈ R^{B_r}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# 初始化（per attention head）
K_r = zeros(B_r, d)  # 残差 key cache
V_r = zeros(B_r, d)  # 残差 value cache
w = zeros(B_r)       # 合并计数

# 每个解码步 T，处理被驱逐 token
def merge_evicted_token(k_t, v_t):
    # Step 1: 选择最兼容 slot
    scores = K_r @ k_t         # [B_r]，key 内积
    r_hat = argmax(scores)     # 最大内积 → 最相似

    # Step 2: 增量均值更新
    K_r[r_hat] = (w[r_hat] * K_r[r_hat] + k_t) / (w[r_hat] + 1)
    V_r[r_hat] = (w[r_hat] * V_r[r_hat] + v_t) / (w[r_hat] + 1)

    # Step 3: 权重递增
    w[r_hat] += 1

# 拼接压缩 cache
K_B = concat([K_p, K_c, K_r])  # proximity + context + residual
V_B = concat([V_p, V_c, V_r])
```

ZSMerge 实验验证：残差合并比纯驱逐在 ≤20% cache size 下减少 attention 输出误差 37-89%。Br=0 时退化为纯驱逐策略。

术语一般如何实现？如何使用？

ZSMerge 在 HuggingFace Transformers 中实现，替换 `scaled_dot_product_attention` 函数。每个 attention head 独立维护 B_r 个残差 slot（通常 B_r=2，为极小分配）。合并计算量 O(B_r·d) per evicted token，相比 full attention O(T²) 可忽略。残差 slot 在每次解码步动态更新，无需额外存储历史 token。代码开源：https://github.com/SusCom-Lab/ZSMerge。

涉及论文标题：
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

## Compensated Attention Scoring (补偿注意力评分)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Compensated Attention Scoring 是 ZSMerge 中用于修正 token 合并后 KV Cache 表示偏差的注意力评分机制。核心公式（Eq. 8）：

$$\hat{a}_t^{(T)} = \frac{\exp\left(\mathbf{q}_T^\top \mathbf{k}_t / \sqrt{d} + \alpha \log w_t\right)}{\sum_{i=1}^T \exp\left(\mathbf{q}_T^\top \mathbf{k}_i / \sqrt{d} + \alpha \log w_i\right)}$$

其中 $w_t$ 为 token t 的融合计数（未压缩 token w_t=1，残差 slot w_t = 该 slot 合并的 token 数），$\alpha \in [0,1]$ 为 scale factor。

补偿机制解决两个关键问题：
1. **表示偏差修正**：合并 token 的 key 为多个原始 token 的均值，与原 value 分布不匹配——log w_t 偏置项修正此偏差。
2. **attention mass 守恒**：保证压缩 token 的 attention 占比不"过度膨胀"——Theorem 1 证明：$\forall$ 未压缩 token i，$\hat{a}_i^{(T)} \geq a_i^{(T)}$（原 attention 分数），即未压缩 token 在压缩后仍保持相对优势。

定理 1 的证明依赖于 Jensen 不等式（指数函数的凸性）：残差 slot r 的 attention numerator 上界为 sum of individual token attention numerators，因此分母受压缩影响有限，未压缩 token 的相对分数在压缩后不降低。

$\alpha=0$ 退化为纯驱逐式注意力（无补偿），$\alpha=1$ 为完全补偿。ZSMerge 实验固定 $\alpha=1$，消融实验显示 $\alpha$ 从 0→1 的 ROUGE 提升 1-5%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# 标准 attention（无补偿）
scores = Q @ K_B.T / sqrt(d)       # [1, B]
attn = softmax(scores)
output = attn @ V_B

# 补偿 attention（ZSMerge Eq. 8）
w_all = [1, 1, ..., w[0], w[1], ...]  # 未压缩=1, 残差 slot = 合并数
log_bias = alpha * log(w_all)          # 对数偏置项
scores_compensated = Q @ K_B.T / sqrt(d) + log_bias  # [1, B]
attn_compensated = softmax(scores_compensated)
output = attn_compensated @ V_B
```

关键性质：log w 偏置使得合并 slot 的 attention 分数增加（因为 log w > 0），但 Theorem 1 保证此增加不会压倒未压缩 token——Jensen 不等式约束 compressed token 的 attention numerator ≤ sum of individual numerators。

术语一般如何实现？如何使用？

实现为对标准 softmax 的一行修改——在 logit 中加 α·log w_all 后调 softmax。O(B) 额外开销。无需训练或校准数据，α=1.0 为推荐默认值。与任何 attention kernel（SDPA、FlashAttention）兼容——只需改变 softmax 输入。

涉及论文标题：
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

## Tripartite KV Cache Budget Allocation (三分区 KV Cache 预算分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Tripartite KV Cache Budget Allocation 是 ZSMerge 提出的 KV Cache 内存管理策略——将总缓存预算 B 划分为三个功能互补的子预算：$B = B_p + B_c + B_r$（Eq. 3）。

1. **Proximity Component** ($B_p$)：保留最近 B_p 个 token 的 KV 对（sliding window），捕获局部上下文模式和短程依赖。这是所有 KV cache 管理方法的标准组件（StreamingLLM、H2O 均保留最近 token）。
2. **Context Component** ($B_c$)：按贡献分数 s^{(T)} 排序保留 top-B_c 个历史 token——s^{(T)}_t = λ·s^{(T-1)}_t + a^{(T)}_t，λ=0.98 为指数衰减因子（Eq. 5）。从全局历史中选出最具信息量的 KV 对。
3. **Residual Component** ($B_r$)：动态维护 B_r 个残差合并 slot，将被驱逐 token 通过 key 相似度匹配 + 增量均值聚合（Eq. 6-7）压缩编码——这是 ZSMerge 区分于纯驱逐方法的核心创新。

最终压缩 cache 为三部分拼接（Eq. 4）：$\mathbf{K}_B = [\mathbf{K}_p \| \mathbf{K}_c \| \mathbf{K}_r], \quad \mathbf{V}_B = [\mathbf{V}_p \| \mathbf{V}_c \| \mathbf{V}_r]$。

预算分配分两步：
1. Proximity ratio $B_p/B$ 控制局部/全局比（推荐 0.5）
2. Residual ratio $B_r/(B-B_p)$ 控制剩余预算中残差占比（推荐 0.02）
3. 剩余为 Context budget $B_c$

当 $B_r=0$ 时退化为纯驱逐策略（H2O-like）：仅保留 proximity + context，其余永久删除。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# 预算分配（固定配置）
B = total_cache_budget    # 如 512, 1024, 18K
B_p = 0.5 * B             # proximity: 一半用于最近 token
B_r = 0.02 * (B - B_p)    # residual: 剩余预算的 2%
B_c = B - B_p - B_r       # context: 剩余全部用于高分 token

# 每个解码步构建压缩 cache
def build_compressed_cache(K, V, s, T):
    # Proximity: 最近 B_p 个 token
    K_p, V_p = K[-B_p:], V[-B_p:]

    # Context: top-B_c 按贡献分数 s
    candidate_tokens = T - B_p  # 排除 proximity 的 token
    idx_c = topk(s[:candidate_tokens], B_c)
    K_c, V_c = K[idx_c], V[idx_c]

    # Residual: 剩余 token (被驱逐) 合并入 B_r 个 slot
    evicted_mask = all tokens NOT in proximity NOR context
    for (k_t, v_t) in evicted_tokens:
        merge_evicted_token(k_t, v_t)  # → update K_r, V_r

    # 拼接
    K_B = concat([K_p, K_c, K_r])  # [B, d]
    V_B = concat([V_p, V_c, V_r])
    return K_B, V_B
```

消融实验推荐配置：$B_p/B=0.5$, $B_r/(B-B_p)=0.02$, $\alpha=1.0$。极端 budget（B_p/B < 0.3 或 > 0.7）显著损害性能。B_r > 0 持续优于 B_r=0（纯驱逐），验证残差合并的有效性。

术语一般如何实现？如何使用？

在实际部署中，三个 budget 参数为静态配置——推理前设定，生成过程中不改变。ZSMerge 基于 Transformers 库实现，通过 `change_mode` 方法支持运行中切换配置。不同任务/模型可独立调参，但由于 ZSMerge 的零样本/无参数特性，默认配置在多数场景下可直接使用。

涉及论文标题：
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs

## Diffusion Language Models (DLMs)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Diffusion Language Models (DLMs) 是基于扩散过程生成文本的语言模型，区别于传统的自回归语言模型（AR）。核心思想来自连续域扩散模型（DDPM）：通过前向噪声（逐步掩盖/破坏 token）和逆向去噪（逐步恢复 token）实现文本生成。对于长度为 L 的序列，DLM 生成从全 [MASK] 序列开始，经过 T 个去噪步逐步解码 token，每步可选择多个位置并行解码。与 AR 的关键区别：(1) 非自回归解码——每步可并行解码任意数量 token，而非逐个左到右生成；(2) 双向注意力——每步所有 token 互相 attend，而非因果掩码；(3) 灵活解码顺序——不限于左到右，可任意顺序填充位置。

文本域中的扩散可分为两类：(a) **连续扩散**（continuous diffusion）：在连续词嵌入空间进行扩散和去噪，如 Diffusion-LM、SSD-LM；(b) **离散扩散**（discrete diffusion）：在离散 token 空间通过转移矩阵定义前向/逆向过程，如 D3PM、SEDD、MDLM。当前主流的 scaling 方向是掩码离散扩散（masked discrete diffusion），其中前向过程以概率 β_t 将 token 替换为 [MASK]，逆向过程学习恢复原始 token。LLaDA（8B）和 Dream（7B）是首个 scaling 到数十亿参数的掩码扩散 LLM，性能已可比肩同等规模的 AR LLM。

从算法pipeline角度拆解术语。

**掩码离散扩散语言模型的前向和逆向过程**：

```
# 前向过程：逐步掩盖 token
# x_0: 原始序列的 one-hot 编码，V 为词表大小
# U_t: 转移矩阵，定义从 token i 到 token j 在步 t 的概率
# 掩码扩散中使用 absorbing state [MASK]
# \bar{α}_t = ∏_{i=1}^t (1 - β_i)，β_i 为掩码概率

前向过程：
  q(x_{c(t)} | x_0) = Cat(x_{c(t)}; p = x_0 \bar{U}_t)
  其中：
    [\bar{U}_t]_{ij} = 
      1                          if i = j = [MASK]
      \bar{α}_t                  if i = j ≠ [MASK]
      1 - \bar{α}_t              if j = [MASK], i ≠ [MASK]

逆向过程（去噪）：
  p_θ(x_{c(t-1)} | x_{c(t)}) ≈ q(x_{c(t-1)} | x_{c(t)}, x_0)
  模型 θ 预测 x_0，再与 x_{c(t)} 联合确定 x_{c(t-1)}

# 采样循环
x_T = [MASK, MASK, ..., MASK]  # 初始全掩码序列
for t = T down to 1:
    # 1. 调用模型预测干净 token
    p_θ(x_{c(t-1)} | x_{c(t)})  → 预测每个位置的 x_0
    # 2. Remasking: 根据置信度/随机策略选择保留哪些 token
    #    高置信度或随机选中的 token 成为 "decoded" (unmasked)
    #    其余 token 继续保持 [MASK]
    for position i in 1..L:
        if confidence_i > threshold OR position in selected_set:
            x_{c(t-1)}^i = predicted_token_i  # 解码
        else:
            x_{c(t-1)}^i = [MASK]              # 保持掩码
```

**DLM 与 AR 的推理复杂度对比**：

| 属性 | AR (with KV-Cache) | DLM (no cache) |
|------|-------------------|----------------|
| 每步计算 | O(n) 个 token（仅新 token） | O(L) 个 token（全序列） |
| 注意力类型 | Causal（单向） | Bidirectional（双向） |
| 总步数 | L（每步 1 token） | T（去噪步，通常≈L） |
| 总复杂度 | O(L³)（累积 O(L²)） | O(L² × T) ≈ O(L³) |
| 实际速度 | 更快（每步仅算 1 token） | 更慢（每步算全部 L token） |

术语一般如何实现？如何使用？

主流掩码扩散语言模型基于 Transformer Decoder 架构（如 LLaDA 基于 LLaMA 架构），主要修改：(1) 去掉 causal mask，使用 bidirectional mask；(2) 训练时使用掩码预测损失（masked prediction loss）而非 next-token prediction；(3) 推理时使用迭代采样+remasking 流程。LLaDA-8B 和 Dream-7B 均通过 HuggingFace Transformers 实现，使用标准 Transformer blocks + FlashAttention。采样策略影响生成质量：置信度 remasking（keep top-k confidence）常优于随机 remasking。DLM 的去噪步数 T 通常设置为序列长度 L 的 1-2×。可通过减少去噪步数（Few-Steps/Half-Steps）加速，但以生成质量为代价。最新加速方法如 dKV-Cache 通过引入 KV 缓存进一步缩小与 AR 的速度差距。

涉及论文标题：
- dKV-Cache: The Cache for Diffusion Language Models

---

## dKV-Cache (Delayed KV-Cache for Diffusion Language Models)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

dKV-Cache（Delayed KV-Cache）是首个为 Diffusion Language Models (DLMs) 设计的 KV-Cache 机制，由 NUS xML Lab 提出（NeurIPS'25）。传统 KV-Cache 在 DLM 中不可用，因为 DLM 的双向注意力（每步 K/V 变化）和非顺序解码（无法预知下一步位置）破坏了缓存复用的前提。dKV-Cache 的核心洞见来自对 DLM 去噪过程中 token 表征动态的实证分析：已解码 token 的 K/V 在后续步趋于稳定（可缓存），而 [MASK] token 持续波动（需重新计算）；最大的表征变化发生在 token 从 [MASK] 变为解码状态的那一步。

基于此提出三个核心机制：(1) **延迟缓存**（delayed caching）：仅缓存已解码 token 的 K/V，未解码（掩码）token 每步重新计算，解决双向注意力下 K/V 时变问题；(2) **一步延迟**（one-step delay）：使用上一步的掩码集合 M_{t-1} 而非当前步 M_t 决定缓存对象，避免刚解码 token 在表征剧变时被过早缓存——实验显示无延迟时性能崩溃至接近零；(3) **缓存刷新**（cache refreshing）：每 N 步清空并重新计算全序列 KV，防止长时间累积的近似误差导致质量退化。

设计两种互补变体：(a) **dKV-Cache-Decode**：近乎无损，刷新间隔大（N=4-8），复杂度 O(L³)；(b) **dKV-Cache-Greedy**：激进缓存，仅计算当前 token D_t、上一步 token D_{t-1} 和局部窗口 W(t)（≤6 个 token）的 QKV，将复杂度降至 O(L²)，但质量略有下降。另有 dKV-Cache-Prefill（预填充 token 永久缓存不刷新）和 dKV-Cache-PD（prefill 永久+decode 间歇刷新）处理长 prefill 场景。方法为 training-free，直接应用于现有预训练 DLM。加速比 2-10×，GPU batch size 越大加速比越高。

从算法pipeline角度拆解术语。

**dKV-Cache-Decode 伪代码**（步 t）：

```
Require: x^{1:L}_{c(t)}, M_t (掩码 token 位置集合), 
         K_{t-1}^{I\M_{t-1}} (缓存 K), V_{t-1}^{I\M_{t-1}} (缓存 V)

// 一步延迟：使用上一步的掩码集 M_{t-1}
1: x' ← x[M_{t-1}]                          // 仅取掩码 token 子序列
2: PE' ← [PE[I\M_{t-1}]; PE[M_{t-1}]]       // 重排位置编码：缓存侧在左

3: Transformer(x') → Q_t^{M_t}, K_t^{M_t}, V_t^{M_t}  // 仅计算掩码 token
4: K_t^I ← Concat(K_{t-1}^{I\M_{t-1}}, K_t^{M_{t-1}}) // 拼接缓存与新 K
5: V_t^I ← Concat(V_{t-1}^{I\M_{t-1}}, V_t^{M_{t-1}}) // 拼接缓存与新 V
6: Reorder(K_t^I, V_t^I, mapping_to_I\M_t)   // 提取下一步的缓存集
7: p' ← Attention(Q_t^{M_t}, K_t^I, V_t^I)   // 双向注意力（全 K/V）
8: p ← Scatter(p', M_{t-1})                  // logits 散播回原位置

// 每 N 步刷新：设 M_{t-1}=∅，重新计算全序列 KV
```

**dKV-Cache-Greedy 的对齐计算集**：
```
// M_t = 上一步掩码集（所有未解码 + [MASK] token）
// Greedy 变体：M_t = {D_t, D_{t-1}} ∪ W(D_{t-1})
// 其中 W 是以 D_{t-1} 为中心、半径 floor(w/2) 的局部窗口
// w ≤ 6， |M_t| = O(1)，从而 O(L³) → O(L²)
```

**cache ratio 度量**：
```
cache_ratio = (1/T) Σ_{i=1}^T |T_i^{cache}| / L
其中 T_i^{cache} = 步 i 从缓存复用的 token 数 = |I \ M_{i-1}|
```

术语一般如何实现？如何使用？

开源实现：https://github.com/horseee/dKV-Cache（Python/PyTorch）。修改 HuggingFace 模型的 forward 函数，插入 concat_reorder 逻辑：重排 token 位置→仅计算掩码 token→concat 缓存 KV→注意力→scatter→更新缓存。使用方式：

```python
# Dream 模型
from models.dream import DreamModel
model = DreamModel.from_pretrained(
    "Dream-7B", use_cache=True, 
    cache_type="decode",       # 或 "greedy", "prefill", "pd"
    cache_steps=4               # 刷新间隔
)

# LLaDA 模型
from models.llada import LLaDAModelLM  
model = LLaDAModelLM.from_pretrained(
    "LLaDA-8B-Instruct", use_cache=True,
    cache_type="decode", cache_steps=8
)
```

关键实现细节：(1) concat_reorder 通过重排序列使缓存 token 连续，将索引开销从 K/V 矩阵 [B,L,D] 层级转移到 token [B,L] 层级；(2) 位置编码随序列重排而同步调整，每次仅需一次 PE reorder、跨层共享；(3) batch size 对加速比影响显著——batch=1 时 memory-bound 导致缓存可能反而减速，batch>1 时加速效果显著。支持 Dream 的三种缓存策略：Un-Shift（标准）、Right-Shift（右移一位）、Un&Right-Shift（两者条件组合）。实测 A6000/H20 GPU 上 1.75-10.19× 加速。

涉及论文标题：
- dKV-Cache: The Cache for Diffusion Language Models

## Cross-Layer SVD for KV-Cache Compression (xKV / 跨层SVD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

xKV（Cross-Layer SVD for KV-Cache Compression）是一种无需训练的 plug-and-play KV-Cache 压缩方法。其核心观察是：尽管相邻层 KV-Cache 的 token-wise cosine similarity 很低，但通过 Centered Kernel Alignment (CKA) 分析发现，不同层的**主导左奇异向量（dominant left singular vectors）高度对齐**。xKV 利用这一跨层奇异向量对齐特性，将多个相邻层的 KV-Cache 水平拼接后执行一次统一的 SVD，提取共享的低秩子空间基（A = U_r @ S_r），各层仅保留独立的低秩重构矩阵（B_ℓ_i），从而显著减少 KV-Cache 存储。

xKV 的关键公式：
$$
\left[\mathbf{X}_{\ell_1}, \dots, \mathbf{X}_{\ell_{|G|}}\right] \in \mathbb{R}^{L \times (|\mathcal{G}| \cdot d)} \approx \mathbf{U}_r \, \mathbf{S}_r \, \mathbf{V}_r^{\top} = \mathbf{A} \left[ \mathbf{B}_{\ell_1}, \dots, \mathbf{B}_{\ell_{|\mathcal{G}|}} \right]
$$
其中 A ∈ R^{L×r} 是被组内所有层共享的 left singular vectors（共享基），B_ℓ_i ∈ R^{r×d} 是层特定的重构矩阵。压缩后仅需存储 A 和 {B_ℓ_i}，原始存储 |G|·L·d → 压缩后 L·r + |G|·r·d。

从算法pipeline角度拆解术语：

```
// xKV Cross-Layer SVD 压缩算法
// 输入: 一组相邻层的 pre-RoPE key/value caches
// G: stride size (如 2, 4), L: sequence length, d: hidden dim, r: rank

def xkv_compress(group_layers, G, r_key, r_val):
    # 1. 水平拼接所有层的 KV-Cache
    K_cat = concat_horizontal([K_li for li in group_layers])  # [L, G*d]
    V_cat = concat_horizontal([V_li for li in group_layers])  # [L, G*d]
    
    # 2. 分别对 Key 和 Value 做跨层 SVD
    U_k, S_k, Vt_k = SVD(K_cat)
    U_v, S_v, Vt_v = SVD(V_cat)
    
    # 3. 保留 top-r 成分（key:value rank ratio = 1:1.5）
    A_k = U_k[:, :r_key] @ S_k[:r_key, :r_key]     # [L, r_key] 共享基
    A_v = U_v[:, :r_val] @ S_v[:r_val, :r_val]     # [L, r_val] 共享基
    
    B_k_li = [Vt_k[:r_key, i*d:(i+1)*d] for i in range(G)]  # 各层 key 重构矩阵
    B_v_li = [Vt_v[:r_val, i*d:(i+1)*d] for i in range(G)]  # 各层 value 重构矩阵
    
    # 4. 存储: A_k, A_v + {B_k_li, B_v_li}
    return (A_k, A_v, B_k_li, B_v_li)

# Decode 阶段: 重构并重新应用 RoPE
def xkv_decode(A_k, B_k_li, A_v, B_v_li, layer_idx):
    K_recon = A_k @ B_k_li[layer_idx]    # [L, d] 重构 key
    K_recon = apply_rope(K_recon)        # 重新施加 RoPE
    V_recon = A_v @ B_v_li[layer_idx]    # [L, d] 重构 value
    return attention(Q, K_recon, V_recon)
```

**Stride-based Grouping**：将 N 层 Transformer 按相邻层分组，每组大小 G（stride=G），共 N/G 组。论文实验验证 G=2 和 G=4。

**压缩率**：当 L >> r·d 时，压缩率 ≈ L/r（近似）。论文在 Llama-3.1-8B 上实现 2.5×-8× 压缩率，xKV-4 在 8× 压缩下仍保持 87.8% avg accuracy（vs Single SVD 的 35.3%）。

术语一般如何实现？如何使用？

通过 HuggingFace Transformers 实现，无需模型修改或微调。在 prefill 阶段按请求在线执行 SVD 分解（on-the-fly，非离线统计），更好地捕捉每个请求上下文的动态。SVD 开销在 128K context 下 <10% prefill time。keys 和 values 具有不同的压缩敏感度，固定 rank ratio(keys:values) = 1:1.5。对 pre-RoPE key states 执行 SVD，decode 时重新应用 RoPE。新生成的 tokens 不压缩（因长上下文场景下生成部分占比 <2%）。代码开源：https://github.com/abdelfattah-lab/xKV。

涉及论文标题：
- xKV: Cross-Layer SVD for KV-Cache Compression

## Tree-structured KV Cache Compression / TreeKV (树形结构 KV 缓存压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

TreeKV 是一种 training-free 的 KV cache 压缩方法（He et al., 2025），核心创新是用**树形结构（tree structure）**替代全局贪心淘汰，实现"左侧稀疏、右侧密集"的平滑 cache 压缩。其设计动机来自 wavelet 分析：对 attention-weighted values 做 multi-level Haar wavelet 分解发现，token 的信息贡献从远到近平滑递增，且与邻居 token 的差异性也逐渐增大（高频分量增长显著）。基于此，TreeKV 设计了一种循环淘汰范围（cyclic eviction scope）机制：在 decoding 每 step，当 cache 满时仅在相邻两个 token {idx, idx+1} 间淘汰重要性较低者，idx 从 1 到 c 循环递增，使得淘汰均匀分布在序列全程。

与 H2O/TOVA 全局贪心排序（O(t log t)，产生区域偏差）不同，TreeKV 每 step 淘汰 O(1)（仅比较两个值），且循环结构自然产生 coarse-to-fine 的信息层次。TreeKV 同时适用于 decoding（token 级）和 prefilling（block 级）两个阶段。Ablation 实验表明树结构本身（而非 attention-weight-based selection）才是性能的核心来源。

从算法pipeline角度拆解术语。

**TreeKV Decoding Stage 伪代码**（论文 Algorithm 1）：

```
参数: cache_size = c (含 4 sink + 508 recent + 512 selected)
S = zeros(c), C = zeros(c), idx = 1
K_cache, V_cache = [], []

for t in 1..T:
    q, k, v = x[t] @ W_Q, x[t] @ W_K, x[t] @ W_V
    K_cache.append(k); V_cache.append(v)
    a = softmax(q @ K_cache^T / sqrt(d))

    C = (C union {0}) + 1
    S = (S union {0}) + a

    if len(K_cache) > c:
        S_avg = S / C  # mean attention weight
        if S_avg[idx] > S_avg[idx+1]:
            evict (idx+1)-th elements
        else:
            evict idx-th elements
        idx = (idx + 1) mod c + 1

    # Position encoding re-assignment per relative order
```

**Prefilling 阶段差异**: prompt 切分为 blocks（block size = b），用最后一个 block query 得到 per-block importance，在 block 级别并行执行上述树形淘汰。

**Tree-structured KV Cache Competition 树形竞争示意**:
```
初始: [T1, T2, T3, T4, T5, T6, T7, T8]
idx=1: T1 vs T2 → 淘汰较低分者
idx=2: T3 vs T4 → 淘汰较低分者（假设前一淘汰后 cache 重新索引）
...
多轮后形成 "左疏右密" 的树形结构
```

术语一般如何实现？如何使用？

论文声明开源 https://github.com/ZiweiHe/TreeKV（截至检索时为空）。HuggingFace Transformers 使用，Llama-2-7B + Llama-3.2-1B-Instruct，NVIDIA RTX 4090 bf16。Cache 组成: 4 sink + 508 recent + 512 TreeKV-selected = 1024 total。16k context 下 16× 压缩。10M token 序列 NLL 稳定（H2O/TOVA 退化）。Longbench 6% budget 达最优效率。每 step O(1) 淘汰开销。

涉及论文标题：
- TreeKV: Smooth Key-Value Cache Compression with Tree Structures

## TOVA / Token Omission Via Attention (基于末端注意力的 KV Cache 动态逐出)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

TOVA（Oren et al., 2024）是一种 KV cache eviction 方法，核心思想：使用**最后一个 token 的 attention scores** 评估所有历史 token 的重要性，保留 attention 最高的 top-k 个 token。其理论基础是论文核心发现 "Transformers are Multi-State RNNs"——transformer decoder 可视为多状态 RNN，attention 机制维护有限数量的隐藏状态（保留的 token 的 KV pairs）。

与 H2O 使用累积 attention（跨多步历史）不同，TOVA 每步只使用最新 attention scores 做淘汰决策，无需维护 per-head 累积数组，计算和内存开销更低。与 TreeKV 的循环淘汰范围不同，TOVA 每次在全局选 top-k，产生强烈区域偏差（Figure 1 显示被选 token 集中于少数注意力高峰区域）。

从算法pipeline角度拆解术语。

**TOVA Decoding 流程**：

```
参数: cache_size, sink_size, recent_size

for t in 1..T:
    q, k, v = x[t] @ W_Q, x[t] @ W_K, x[t] @ W_V
    K_cache.append(k); V_cache.append(v)
    a = softmax(q @ K_cache^T / sqrt(d))

    if len(K_cache) > cache_size:
        mid_scores = a[0, sink_size:-recent_size]
        topk_idx = topk(mid_scores, k=cache_size - sink_size - recent_size)
        keep = [0:sink_size] + topk_idx + [-recent_size:]
        K_cache, V_cache = K_cache[keep], V_cache[keep]
```

**Annotations**: 与 H2O 的 `attn_accum += scores` 跨步累积不同，TOVA 的 scoring 不跨步——每步用最新的 `a[0, :]` 独立评估。Sink tokens（前几个）+ recent tokens（后几个）固定保留，仅中间区域参与动态淘汰。由于不跨步累积，老 token 的"历史高分"无法保护其不被淘汰（优劣参杂——减少偏差但可能误淘汰曾重要的 token）。

术语一般如何实现？如何使用？

论文 "Transformers are Multi-State RNNs" (arXiv 2401.06104)。在 Llama-2-7B 上 PG19 perplexity: 4k context PPL 7.00 (TOVA) vs 7.06 (H2O) vs 6.84 (Full)。16k 时 TOVA 7.15 vs TreeKV 6.91（3.6% 差距），显示全局贪心在长序列下的局限性。TOVA 的简化设计使其计算高效于 H2O（无累积维护），但 TreeKV 在超长序列（10M）和复杂上下文任务上远优。实践中 TOVA 与 FlashAttention 兼容性问题与 H2O 类似——需要多 pass attention 获取 attention scores。

涉及论文标题：
- TreeKV: Smooth Key-Value Cache Compression with Tree Structures

## Cyclic Eviction Scope for KV Cache (KV 缓存循环淘汰范围 / 树形淘汰机制)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cyclic Eviction Scope 是 TreeKV 的核心淘汰机制。在 decoding 每 step，当 KV cache 容量满时，淘汰决策局限在相邻两个 token {idx, idx+1} 构成的"淘汰范围（eviction scope）"内——比较两者的平均 attention weight，淘汰较低分者。idx 每 step 循环递增 (idx mod c + 1)，使得淘汰范围从 cache 头部平滑移到尾部再回到头部。

与 H2O/TOVA 的全局贪心排序（每次在所有 token 中选最差者淘汰）不同，cyclic eviction scope 有三个关键特性：(1) **O(1) 每步开销**——仅比较两个标量值，无排序；(2) **均匀空间覆盖**——每轮完整循环（c 步）中每个位置恰好参与两次比较（一次作为 idx 左端点，一次作为 idx+1 右端点），避免了 H2O 的区域偏差；(3) **树形竞争**——连续多轮形成二叉树竞争，相邻 token 逐级比较，胜者保留到下一轮，最终形成 coarse-to-fine 的信息层次（左侧远距离 token 淘汰率高，保留密度低；右侧近距离 token 保留密度高）。

从算法pipeline角度拆解术语。

**Cyclic Eviction Scope 树形竞争示意**:

```
初始 cache (c=8): [T1, T2, T3, T4, T5, T6, T7, T8]

第 1 轮循环 (idx=1→2→...→8):
  idx=1: scope={1,2}, 比较 T1 vs T2 → 淘汰低分者
  idx=2: scope={3,4}, 比较 T3 vs T4 → 淘汰低分者
  ... 每步 idx 循环递增，淘汰后 cache 重新索引

树形结构示意（连续多轮后）:
          [T1]                    ← 最左侧，每轮都面临淘汰，存活概率最低
         /    \
      [T1]   [T3]                ← 中间距离，已被淘汰多轮
             /    \
          [T3]   [T5,T6]         ← 近端，保留密度最高（"右密"）
```

**伪代码**:
```
idx = 1
for each step when cache full:
    S_avg = S / C                # 每个 token 的平均 attention weight
    if S_avg[idx] > S_avg[idx+1]:
        evict (idx+1)-th KV pair
    else:
        evict idx-th KV pair
    idx = (idx + 1) % c + 1      # 循环递增
```

**Annotations**:
- `S_avg[idx]` 和 `S_avg[idx+1]` 是 cache 中第 idx 和 idx+1 个 token 的平均 attention weight（非原始序列位置）
- idx 的循环范围是 1..c（cache 容量），保证每步淘汰后 cache 大小回到 c
- 新 token 追加到 cache 尾部，其初始 S=0, C=0，在首次参与淘汰时与左侧老 token 比较——若老 token 重要性低则被淘汰（新 token 得以保留），若新 token 尚无足够重要性证据则被淘汰
- Ablation 证实（Figure 5）：即使完全不用 attention weight（每次固定淘汰左侧 token），仅靠循环淘汰范围的树形空间分布，perplexity 已远超 H2O 和 TOVA

术语一般如何实现？如何使用？

HuggingFace Transformers 实现，per-layer 或 per-head 维护 idx (int) 和 S_avg (float array, size c)。每 decode step 开销为 1 次标量比较 + 1 次模运算 = 可忽略。与 H2O top-k 排序 O(c log c) 相比，在 batch serving 场景下优势显著。TreeKV_Select_Left_Token 变体（固定淘汰左侧，零 attention 开销）在 PG19 65k 书上 perplexity 与完整 TreeKV 差距极小，证明树结构本身是性能核心驱动力。

涉及论文标题：
- TreeKV: Smooth Key-Value Cache Compression with Tree Structures

## Tree Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Tree Attention 是一种拓扑感知的多 GPU 精确注意力并行解码算法，由 Zyphra 和 EleutherAI 提出。核心思想是将 self-attention 重新表述为能量函数（moment generating function）的梯度，利用 logsumexp 和 max 操作的结合律（associative property），将序列维度上的归约操作通过树形归约（tree reduction）并行化。该算法专门针对解码阶段（单个 query token），将 KV cache 在序列维度分片到 p 个 GPU，每 GPU 使用 Flash Attention 2 计算局部 attention，再通过 3 次 AllReduce（max + sum×2）合并全局结果。

理论创新链条：
(1) **Observation 1**：证明 self-attention = ∂F/∂ζ|_{ζ=0}，其中 energy function F(ζ) = log Σ_a exp(q·k_a^T + ζ·v_a^T)，ζ 是 "source" 辅助向量。这是首次推导出 self-attention block 的完整 scalar energy function。
(2) **Associative Reduction (Theorem 1)**：logsumexp 和 max 是 associative 操作，对大小为 N 的数组在 p 个并行处理器上的归约时间为 O(N/p + log p)，当 p=N 时降至 O(log N)。
(3) **Tree Decoding (Algorithm 3)**：解码时 K,V 分片，每 GPU 本地 Flash Attention 2 → AllReduce(max) 获取全局 max → 本地数值稳定化 → AllReduce(sum)×2 获取全局分子分母 → 归一化输出。

从算法pipeline角度拆解术语。
Tree Attention 的算法 pipeline（单 token 解码，p 个 GPU）：
```
输入: q ∈ R^{1×d_h}, K ∈ R^{N×d_h}, V ∈ R^{N×d_h}
分片: 每 GPU_i 持有 K_i, V_i ∈ R^{t×d_h}, t = N/p

Step 1: q 广播到所有 p 个 GPU
  scatter(q, all_gpus)

Step 2: 每 GPU 本地 Flash Attention 2
  o_i, lse_i = FlashAttention2(q, K_i, V_i)
  # o_i ∈ R^{1×d_h}, lse_i = log Σ_j exp(q·k_{ij}^T) ∈ R

Step 3: AllReduce(max) → 全局 max (tree reduction)
  m_global = max(lse_1, lse_2, ..., lse_p)
  # 通信步数 O(log p), 传输 1 个标量

Step 4: 本地数值稳定化
  n_i = o_i × exp(lse_i - m_global)  # 分子, [1, d_h]
  d_i = exp(lse_i - m_global)        # 分母, [1]

Step 5: AllReduce(sum) × 2 → 全局分子分母
  n_global = Σ_i n_i  # [1, d_h], tree reduction
  d_global = Σ_i d_i  # [1],      tree reduction

Step 6: 归一化
  z = n_global / d_global  # 精确 attention 输出
```

与 Ring Attention 的对比：
```
Ring Attention pipeline (p GPU, 单 token 解码):
  for step = 0..p-1:
    o_i, lse_i = FlashAttention2_and_accumulate(q, K_current, V_current)
    Send(K_current, V_current) → GPU_{(i+1)%p}  # P2P
    Recv(K_current, V_current) ← GPU_{(i-1)%p}  # P2P
  # 通信步数 O(p), 每次传输 K,V chunk (2bt×d_h elements)
```

关键性质：Tree Attention 是精确计算（exact attention），数值结果与标准 attention 前向传播完全一致，是 Ring Attention 的 drop-in replacement。

术语一般如何实现？如何使用？
实现：开源在 https://github.com/Zyphra/tree_attention，使用 JAX + Flash Attention 2 (JAX binding) + shard_map。通过 `lax.pmax` (max reduction) 和 `lax.psum` (sum reduction) 调用 NCCL AllReduce。NCCL 自动选择 intra-node ring reduce（NVLink 高带宽）和 inter-node tree reduce（InfiniBand 低带宽），实现拓扑感知通信。

使用场景：长上下文 LLM 解码（>32K tokens），跨多个 GPU 的注意力并行化。适用于 DGX H100 集群（NVLink 4.0）、AMD MI300X（Infinity Fabric）、RTX 4090（PCIe）等硬件。在 Llama 3.1-8B、128K context、8×H100 上，解码延迟比 Ring Attention 快 2-4×。

涉及论文标题：
- Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters

## Energy Function for Self-Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Energy Function for Self-Attention 是 Tree Attention 论文推导出的将 self-attention 操作表达为标量能量函数梯度的数学公式。能量函数定义为：
$$F(\zeta) = \log \sum_{a=1}^{N} \exp(q \cdot k_a^T + \zeta \cdot v_a^T)$$
其中 $\zeta \in \mathbb{R}^{d_h}$ 是辅助 "source" 向量，q,k,v 是 query/key/value。self-attention 操作是 F 关于 ζ 在 ζ=0 处的梯度：
$$\sum_{a=1}^{N} \operatorname{softmax}(q \cdot k_a) v_a = \left. \frac{\partial F}{\partial \zeta} \right|_{\zeta=0}$$

这个公式的物理意义来自统计力学：F 是 cumulant-generating function（类比 Helmholtz 自由能），softmax 后的 attention scores 构成概率分布 P_a = exp(q·k_a^T) / Σ_i exp(q·k_i^T)（partition function Z = Σ exp(q·k^T)），引入 source ζ 后 Z(ζ) = Σ exp(q·k_a^T + ζ·v_a^T)，attention 输出 = ⟨v⟩ = (1/Z) ∂Z/∂ζ|_{ζ=0} = ∂logZ/∂ζ|_{ζ=0}。

从算法pipeline角度拆解术语。
能量函数在 Tree Attention pipeline 中的作用：
```
给定: q, K_i (本地 key chunk), V_i (本地 value chunk)

# 能量函数 (forward, Algorithm 1):
r_i = q·K_i^T + ζ·V_i^T          # 每个 chunk 的 "能量贡献", [t]
m = TreeReduce(max, r_i)         # 全局 max (numerical stability)
r_i' = r_i - m                    # stable shift
F = TreeReduce(logsumexp, r_i')  # 全局 logsumexp = 能量函数值

# 梯度 (backward w.r.t ζ, Algorithm 2):
∂F/∂ζ = TreeReduce(sum, exp(r_i' - F) · V_i)  = attention 输出
```

关键洞察：自动微分的经典结论——∇f(x) 可以用与 f(x) 相同的渐进时间复杂度计算。因此，如果能高效计算能量函数 F（通过 tree reduction O(N/p + log p)），就能高效计算 attention（F 的梯度）。F 的 computational graph 很浅（仅 3 次 AllReduce），因此反向传播的内存开销可忽略。

术语一般如何实现？如何使用？
实现：在代码中，Algorithm 1 和 Algorithm 2 被合并为单一函数 `tree_flash_decode`（Appendix D），同时返回能量函数值和梯度（即 attention 输出）。ζ 实际上不会被 materialize——ζ=0 时，F 退化为仅关于 q·k 的 logsumexp，梯度计算简化为 exp(q·k - lse)·v 的加权和。

用途：能量函数表述不仅是理论好奇心——它直接揭示了 attention 计算中的结合律结构（logsumexp 的 associative property），从而证明了 tree reduction 的可行性，为 Tree Attention 算法提供了数学正确性保证。

涉及论文标题：
- Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters

## LogSumExp Reduction

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
LogSumExp (LSE) Reduction 是一种基于 logsumexp 操作结合律的并行归约策略。logsumexp 操作定义为：
$$\operatorname{logsumexp}(\{x_1, ..., x_N\}) = \log \sum_{i=1}^{N} \exp(x_i)$$
LSE 的关键性质是 **associativity**：
$$\operatorname{logsumexp}(\{A, \operatorname{logsumexp}(\{B, C\})\}) = \operatorname{logsumexp}(\{\operatorname{logsumexp}(\{A, B\}), C\})$$
这意味着 LSE 可以像加法一样进行树形归约：先将数据分块，每块独立计算局部 LSE，再通过树形结构层层合并局部结果得到全局 LSE。同理，max 操作也具有结合律。在 attention 计算中，LSE 出现在 softmax 的分母归一化项中：
$$z = \frac{\sum \exp(q \cdot k_i^T) v_i}{\sum \exp(q \cdot k_i^T)} = \frac{\sum \exp(q \cdot k_i^T) v_i}{\exp(\operatorname{logsumexp}(\{q \cdot k_i^T\}))}$$

从算法pipeline角度拆解术语。
LSE Reduction 在 Tree Attention 中的具体计算过程：
```
# 假设 p=4 GPU，序列分 4 个 chunk
# 每 GPU: lse_i = log Σ_{j in chunk_i} exp(q·k_j^T)

# == Tree Reduction of max (2 层) ==
# Level 1 (intra-node, NVLink):
m_12 = max(lse_1, lse_2)   # GPU 1,2 归约
m_34 = max(lse_3, lse_4)   # GPU 3,4 归约
# Level 2 (inter-node, InfiniBand):
m_global = max(m_12, m_34) # 跨节点归约

# == 数值稳定化 (用 m_global 稳定所有局部值) ==
# r_i = q·k_i^T - m_global
# n_i = Σ exp(r_i) * v_i  (局部分子)
# d_i = Σ exp(r_i)        (局部分母)

# == Tree Reduction of sum (2 层) ==
# Level 1:
n_12 = n_1 + n_2; d_12 = d_1 + d_2
n_34 = n_3 + n_4; d_34 = d_3 + d_4
# Level 2:
n_global = n_12 + n_34; d_global = d_12 + d_34

# 输出: z = n_global / d_global
```

时间复杂度：Theorem 1 证明 associative reduction 在 p 个处理器上的时间为 O(N/p + log p)，其中 O(N/p) 是每处理器本地计算，O(log p) 是树形归约的通信步数。

术语一般如何实现？如何使用？
实现：通过 NCCL 的 AllReduce 操作，在 reduce 阶段使用树形归约算法。在 JAX 中通过 `lax.pmax`（对应 AllReduce(max)）和 `lax.psum`（对应 AllReduce(sum)）调用。NCCL 自动检测网络拓扑——intra-node 使用 ring reduce（NVLink 高带宽 900 GBps），inter-node 使用 tree reduce（InfiniBand 较低带宽 ~50 GBps per link）。

关键区别：传统的 attention 计算中，LSE 归约是隐式的（Flash Attention 的 online softmax 在单 GPU 内的 SM 间归约）。Tree Attention 将 LSE 归约显式化并扩展到跨 GPU 场景，揭示了 attention 的 logsumexp 归约在分布式环境中可高效并行化——与 Flash Decoding 在 GPU 内 SM 级别做 split-KV+归约的思想类似，但在跨 GPU 层级。

涉及论文标题：
- Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters

## Sequence Parallelism for Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Sequence Parallelism for Attention 是将 Transformer 的 self-attention 计算在序列维度上跨多个设备（GPU）并行化的一类技术。核心思路是将长序列的 K,V 分片到多个设备，避免单个设备内存无法容纳完整 KV cache。不同于 Tensor Parallelism（沿 head/隐藏维度切分）和 Pipeline Parallelism（沿层切分），Sequence Parallelism 沿 token 序列维度切分工作和数据。

主要方法包括：
- **Ring Attention**（Liu et al., 2023）：KV 在 GPU 间 P2P 环形传递，每 GPU 依次处理所有 chunk。
- **Tree Attention**（本论文）：KV 不移动，通过 AllReduce 归约部分结果，通信步数 O(log p)。
- **Ulysses/DeepSpeed-Ulysses**：通过 All-to-All 在序列维度和 head 维度之间转换分片方式。
- **Star Attention**：两阶段——blockwise-local attention + distributed query-anchor softmax。

从算法pipeline角度拆解术语。
Sequence Parallelism 中不同方法的 pipeline 对比（以 p=4 GPU 解码为例）：
```
Ring Attention (O(p) 通信步):
  GPU_0: [q, K_0, V_0] → attn → send(K_0,V_0)→GPU_1, recv(K_1,V_1)←GPU_3
  GPU_1: [q, K_1, V_1] → attn → send(K_1,V_1)→GPU_2, recv(K_2,V_2)←GPU_0
  ...循环 p 次...
  每次传输 2btd elements (K+V chunk)
  总通信量: p × 2btd

Tree Attention (O(log p) 通信步):
  GPU_0..3: [q, K_i, V_i] → FlashAttn2 → (o_i, lse_i)
  AllReduce(max, lse_i)           # tree, O(log p) 步, 1 elem
  AllReduce(sum, n_i)             # tree, O(log p) 步, d_h elems
  AllReduce(sum, d_i)             # tree, O(log p) 步, 1 elem
  总通信量: 2(p-1)/p × (d_h + 2)

Ulysses (All-to-All):
  GPU 间通过 All-to-All 在 seq 维度和 head 维度间转换
  Attention 在 head-parallel 模式下计算
  每个 attention 需要 2 次 All-to-All
```

术语一般如何实现？如何使用？
实现：Ring Attention 通过 NCCL P2P send/recv。Tree Attention 通过 NCCL AllReduce（JAX `lax.pmax`/`lax.psum`）。Ulysses 通过 NCCL All-to-All（DeepSpeed 或 PyTorch `dist.all_to_all`）。选择哪种方法取决于：(a) 硬件拓扑——homogeneous 带宽适合 Ring，两层拓扑适合 Tree/AllReduce；(b) 序列长度——长序列 Ring 通信量大，Tree 通信量与序列长度无关；(c) 是否训练或仅解码——训练时有多个 query，Ring 可 overlap，解码时单 query 无法 overlap。

涉及论文标题：
- Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters

---

## Q/K Concentration (Pre-RoPE Q/K Concentration)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Q/K Concentration 是 TriAttention（Mao et al., 2025）发现的 pre-RoPE 空间中的注意力现象：在 Rotary Position Embedding (RoPE) 旋转之前，大量 attention head 的 Query 和 Key 向量高度聚集于固定的非零中心周围。该聚集使用 Mean Resultant Length (MRL) $R = \|\mathbb{E}[q]\|/\mathbb{E}[\|q\|]$ 量化：R→1 表示完美聚集，R→0 表示均匀分散。在 Qwen3-8B 全 1152 个 attention head 中，约 90% 的 head 的 R > 0.95，证实聚集具有普遍性。

关键特性：(1) 跨位置稳定——同一 head 中不同位置 tokens 的 Q/K 向量围绕同一中心聚集；(2) 跨上下文稳定——Math/Coding/Chat 三种不同领域的 MRL 几乎相同（0.977-0.980）；(3) 跨架构普遍——GQA (Qwen3, Llama3) 和 MLA (GLM-4.7-Flash) 均存在，MLA 中 96.6% heads 的 R > 0.95；(4) 模型内在属性——校准数据质量（HTML vs Chat）和数据量（50K-960K tokens）几乎不影响聚集度量。

从算法pipeline角度拆解术语：

Q/K Concentration 使 attention logit 变为可预测的三角函数级数：
```
# 无聚集时（标准 RoPE Attention）：
logit(q, k) = Σ_f ‖q_f‖·‖k_f‖·cos(ω_f·Δ + (arg(q_f)-arg(k_f)))
# q_f, k_f 随 token 位置和内容变化 → 需要实时计算

# 有聚集时：q_f ≈ q̄_f, k_f ≈ k̄_f（近似为常数中心）
logit(Δ) ≈ Σ_f ‖q̄_f‖·‖k̄_f‖·cos(ω_f·Δ + φ̄_f)
# 其中 φ̄_f = arg(q̄_f) - arg(k̄_f) 是固定相位差
# 结果：attention 退化为仅依赖距离 Δ 的三角函数级数
```

术语一般如何实现？如何使用？

实现：离线校准阶段收集少量 tokens（50K 即可）的 pre-RoPE Q/K 向量。对每个 head 的每个频段 f，计算：(1) Q 中心 E[q_f]（复数均值）；(2) 期望 Q 范数 E[‖q_f‖]；(3) Mean Resultant Length R_f = ‖E[q_f]‖/E[‖q_f‖]。统计量在推理前离线计算一次，以 JSON/numpy 格式存储，推理时直接使用。

核心用途：(1) Q 中心用作"通用 proxy query"——通过三角函数级数预测任意位置的 key 会收到多少 attention（S_trig 评分）；(2) R_f 用作 S_trig 和 S_norm 的自适应加权因子；(3) 诊断工具——识别哪些 head 有强距离偏好。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---

## Mean Resultant Length (R / MRL) in Attention Context

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Mean Resultant Length (MRL/R) 是方向统计学（Directional Statistics, Mardia & Jupp, 1999）中的标准度量，TriAttention 将其引入注意力分析，用于量化 pre-RoPE 空间中 Q/K 向量围绕其均值方向的聚集程度。对频段 f 的 Q 向量：$R_f = \|\mathbb{E}[q_f]\|/\mathbb{E}[\|q_f\|]$。R_f = 1 表示所有向量指向完全相同方向（完美聚集，三角函数级数精确）；R_f = 0 表示向量均匀分布（无聚集，三角函数级数不可用）。

MRL 在 TriAttention 中的双重角色：
(1) 量化聚集强度——判断三角函数级数对每个频段的可靠性；
(2) 自适应加权因子——在 S_norm 中 (1-R_f) 决定了 norm-based 信号的贡献：R_f 高时 (1-R_f) 小，S_trig 主导；R_f 低时，S_norm 贡献更大。

从算法pipeline角度拆解术语：
```
for each head h, frequency band f:
    E_q_f = mean(calib_Q[h, :, f])     # Q 中心（复数）
    R_f = |E_q_f| / mean(|q_f|)       # MRL: 0 ≤ R_f ≤ 1
    S_norm 中的自适应权重 = (1 - R_f)  # 聚集强 → 权重小；聚集弱 → 权重大
```

术语一般如何实现？如何使用？

实现：离线校准阶段计算，与 Q/K 中心同时完成。在 Qwen3-8B 上典型 MRL 约 0.98，约 90% heads 的 R > 0.95。MRL 跨领域数据（Math/Coding/Chat）几乎相同，证明其为模型内在属性。使用方式：(1) 诊断哪些 head 适合纯 S_trig；(2) 自适应平衡两个评分组件；(3) 跨架构验证聚集现象的普遍性。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---

## Trigonometric Series for KV Scoring (S_trig)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

S_trig 是 TriAttention 提出的 KV cache 重要性评分组件，利用 pre-RoPE Q/K 聚集现象，通过三角函数级数预测 key 在未来 query 位置会收到多少 attention。核心公式（Eq 6）：

$$S_{\text{trig}}(k, \Delta) = \sum_f \|\mathbb{E}[q_f]\| \cdot \|k_f\| \cdot \cos(\omega_f \Delta + \phi_f)$$

其中 E[q_f] 是 Q 中心（校准数据均值），k_f 是 cache 中 key 在频段 f 的 pre-RoPE 复数表示，ω_f = θ^{-2f/d} 是 RoPE 频率，Δ = p_q - p_k 是 Q-K 距离，φ_f = arg(E[q_f]) - arg(k_f) 是相位差。

物理意义：当 Q 高度聚集时，用 Q 中心替代未来任意位置的 query，三角函数级数给出该 key 在距离 Δ 处收到的平均 attention。与 post-RoPE 方法的根本区别：S_trig 是 model-intrinsic 预测——仅依赖 Q 中心和 key 自身的 pre-RoPE 表示——不依赖观测任何实际 attention scores，因此不受 RoPE 旋转限制的"小观察窗口"问题影响。

术语一般如何实现？如何使用？

实现：(1) 离线校准阶段——收集校准数据的 pre-RoPE Q 向量，计算 Q 中心 E[q_f]。(2) 推理时每 128 tokens 触发一次 scoring，遍历 cache 中所有 key，对每个 key 和每个 future offset δ∈{1,2,4,...,2^16} 计算 S_trig(k, Δ+δ)，取平均。计算量 O(|cache| × 17 × d/2)，但因仅每 128 步执行一次，实际 overhead 极低。

S_trig 能捕获距离偏好——某些 head 偏好近距离 key（S_trig 在 Δ 小时 peak），某些 head 偏好远距离 key（attention sink, S_trig 在 Δ 大时 peak）。跨域校准验证：校准数据用 coding data 时，AIME24 准确率 44.2%（vs reasoning 校准 42.1%）——证明 Q 中心是模型内在属性。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---

## TriAttention Scoring Function (TriAttention KV Scoring)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

TriAttention 评分函数是 TriAttention KV cache 压缩方法的核心，结合两个互补信号（Eq 10）：$S(k, \Delta) = S_{\text{trig}}(k, \Delta) + S_{\text{norm}}(k)$。S_trig 捕获距离偏好（通过三角函数级数预测 key 在距离 Δ 处的 attention），S_norm 捕获低范数 key（S_norm = Σ_f (1-R_f)·E[‖q_f‖]·‖k_f‖，以 MRL 自适应加权）。

GQA 聚合策略：每个 KV head 被 G 个 query head 共享，产生 G 个不同尺度的评分。处理方式——per-head z-score normalize 后 max 聚合（Eq 12-13）——只要任一 query head 认为 key 重要就保留。Window-based Pruning：每 128 tokens 触发一次评分+pruning，保留 top-B keys。

消融验证（Qwen3-8B, KV budget 2048, AIME）：
- 去掉 S_trig（仅 Snorm）：AIME24 42.1% → 18.8% (-23.3%)
- 去掉 Snorm（仅 S_trig）：AIME24 45.8% → 40.4% (-5.4%)
- 去掉 R 加权：AIME25 32.9% → 28.7% (-4.2%)

术语一般如何实现？如何使用？

实现：集成到 vLLM 作为 plugin（通过 monkeypatch scheduler 和 worker），也支持 SGLang 和 MLX。校准离线处理一次。评分无需计算实际 attention（仅需 key 的 pre-RoPE 表示和离线 Q 中心），远低于 post-RoPE 方法（需计算完整 QK^T attention matrix）。在 AIME25 上匹配 Full Attention 准确率（40.8%）同时实现 2.5x throughput 或 10.7x KV memory reduction。代码开源：https://github.com/WeianMao/triattention。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---

## Pre-RoPE / Post-RoPE Space for KV Cache Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Pre-RoPE 和 Post-RoPE 空间是 RoPE 位置编码前后的向量空间区分。Pre-RoPE 空间：Q/K 向量在 RoPE 旋转之前的空间，仅编码内容信息，不受位置影响。Post-RoPE 空间：Q/K 向量经 RoPE 旋转（乘以 e^{iωp}）后的空间，编码"内容+位置"混合信息。

这一区分对 KV cache 压缩方法至关重要。TriAttention 首次明确指出 post-RoPE 方法的系统性限制：query 经 RoPE 旋转后方向随位置连续变化，只有最近的 query 具有"当前"朝向，导致观察窗口极小（约 25 个 query 最优）——这是 post-RoPE 方法固有的，无法通过增加窗口大小解决（Zhang et al., 2025 确认：增加到 25 个 query 后性能下降）。

Pre-RoPE 空间不受位置旋转影响——Q/K 围绕固定中心聚集（Q/K Concentration 现象），跨位置稳定。TriAttention 回到 pre-RoPE 空间，利用 Q 中心替代未来 query 预测 attention 模式，完全绕过观察窗口限制。

术语一般如何实现？如何使用？

Pre-RoPE 向量的获取：在模型 attention layer 中 RoPE 旋转之前截取 Q/K 中间表示。在 vLLM 中通过 monkeypatch attention forward 实现。使用场景：任何需要考虑 Q/K 方向信息且不希望受位置编码污染的注意力分析——KV 压缩（TriAttention）、attention head 功能分类、模型诊断。Pre-RoPE 空间的 Q/K 向量跨位置稳定，是分析 head 语义功能比 post-RoPE 更优的信息源。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---

## Reconstruction Correlation (r-bar / Attention Reconstruction Correlation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Reconstruction Correlation r̄ 是 TriAttention 提出的评估指标，用于量化三角函数级数从 Q/K 中心预测实际 attention pattern 的准确度（Eq 5）：$\bar{r} = \frac{1}{N} \sum_{i=1}^N \rho(\mathbf{a}_i, \hat{\mathbf{s}})$，其中 a_i 是 query i 的实际 attention logits（在对数间隔的距离上采样），ŝ 是三角函数级数从 Q/K 中心预测的 attention logits，ρ 是 Pearson 相关系数。

物理意义：r̄ 量化"Q/K Concentration → 可预测距离偏好"这条因果链的强度。r̄ 高表示 Q/K 聚集确实导致了可被三角函数级数捕获的 attention 模式。跨模型分布：Qwen3-8B、Qwen2.5-7B、Llama-3-8B 的 r̄ 均右偏，均值 > 0.5，峰值在 0.6-0.9。

术语一般如何实现？如何使用？

实现：纯 Python + numpy，在大约 10K token 序列上计算一次。对数间隔采样 {1,2,4,8,...} 确保跨距离尺度平衡覆盖（避免相邻距离的样本非独立而高估相关性）。

使用场景：(1) 验证 Q/K Concentration 的因果效应；(2) 诊断哪些 head 适合用 S_trig——r̄ > 0.5 的 head（约 53.5% in Qwen3）三角函数级数预测有效；(3) 跨模型/跨架构比较——在 GQA 和 MLA 上 r̄ 分布相似，证明聚集现象是架构无关的通用规律。

涉及论文标题：
- TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

---

## Task-Adaptive KV Cache Window Selection

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Task-Adaptive KV Cache Window Selection 是 WindowKV（Zuo et al., 2025）提出的 KV cache 压缩方法，核心创新在于将逐 token 选择替换为以**连续语义窗口**为单位的保留/逐出决策，并根据任务类型自适应调整每个窗口内的 token 保留比例。

传统方法（H2O、PyramidKV、StreamingLLM）对每个 token 独立评分和选择——相邻 token 可能被不同步逐出，打断语义连贯性。WindowKV 观察到人类阅读以信息块（窗口）处理长文本（Rayner, 1998），因此将 context 切分为固定大小 ω 的 review windows，以 window 为粒度做 KV cache 保留。同时将下游任务分为 Information Localization（QA 类，p=ω，保留全窗口）和 Information Aggregation（摘要类，p<ω，仅保留 top-p 高分 token），训练 bert-base-cased 分类器自动判断。

算法流程：(1) Context 分为 observation window (最后 α tokens) 和 review context (切为 K 个 ω-token windows)；(2) 计算 observation window 对各 review token 的累积注意力 t_j = Σ A_ij；(3) 窗口级打分 s_k = (1/min(p,ω)) · sum(Top-p(W_k))；(4) 按 dynamic budget 选 top-n windows。

从算法pipeline角度拆解术语：

```
# WindowKV 完整 pipeline
# Input: n tokens, ω window size, α observation size, b_total KV budget

# === Phase 1: Task Classification ===
task_type = Classifier(input_context)  # bert-base-cased → localization/aggregation
p = ω if task_type=="localization" else p_small  # 决定窗口内保留比例

# === Phase 2: Per-Group Window Selection (仅 group-first layers) ===
for group_first_layer l_g in [0, γ, 2γ, ...]:
    Q, K = W_q @ h_lg, W_k @ h_lg           # [n, d_head]
    A = softmax(Q @ K^T / sqrt(d_k))        # [n, n]
    t_j = sum(A[n-α:n, j]) for j in [0,n-α) # observation → token scores
    windows = chunk(tokens[0:n-α], ω)       # partition review context
    scores = [(1/min(p,ω))*sum(top_p(w,p)) for w in windows]
    I_lg = indices_of(topk(scores, b_h/ω))  # retain top windows

    # Group sharing: for l in [l_g+1, l_g+γ-1]:
    #     I_l = I_lg  (直接复用首层 indices)
```

术语一般如何实现？如何使用？

开源：https://github.com/optim996/WindowKV。基于 HuggingFace Transformers，prefill 后执行 window selection。Group-first layer 执行 full attention + scoring，组内其余层复用 indices。Qwen2.5-1.5B: γ=7, ω=32; LLaMA3-8B: γ=8, ω=8/16。λ=14 控制金字塔形状。12% KV cache 下 LongBench 保持 41.35 vs FKV 41.51。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---

## Intra-Group Layer KV Cache Indices Sharing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Intra-Group Layer KV Cache Indices Sharing 是 WindowKV 的计算效率策略：将 m 层分为 H=m/γ 组，仅每组首层 l_g 执行 window selection 获得 indices I_{lg}，组内其余层直接复用，省去重复的 attention 计算。理论基础是 LLM 相邻层 attention 分布相似（Ma et al., 2024; Liu et al., 2025, ChunkKV），并通过同组内层间 KV cache index 的 Jaccard similarity 实验验证。

消除实验：γ=1（无共享，32.13）vs γ=7（共享，32.75 on Qwen2.5）——适度共享因预算更均匀分布而略有提升；γ=14 时 budget 过于均匀破坏金字塔结构（27.83）。LLaMA3-8B 最优 γ=8。

公式：$H = m/\gamma$, $\mathbb{I}_{l_{h\gamma}}$ 仅首层计算，$\mathbb{I}_{l_{h\gamma+k}} = \mathbb{I}_{l_{h\gamma}}$ for $k \in [1, \gamma-1]$。计算开销 O(m·n²) → O(H·n²) = 1/γ ×。

术语一般如何实现？如何使用？

实现：在 Transformer forward pass 中 `if layer_idx % γ == 0` 则执行完整 window selection，否则复用上一首层 indices 对 KV cache 做 gather 操作。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---

## Observation Window-Driven KV Cache Importance Scoring

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Observation Window-Driven KV Cache Importance Scoring 是 WindowKV 提出的 token 重要性评估机制：以输入末尾 α 个 token（紧邻生成位置）作为 observation window，计算其对 review context 中各 token 的累积注意力 $t_j = \sum_{i \in [n-\alpha, n]} \mathbf{A}_{ij}$，用于后续窗口级重要性评分。

优势对比：(1) vs H2O 的全 query 平均注意力——不易被 attention outliers 主导；(2) vs PyramidKV 仅用 instruction tokens——observation window 紧邻生成位置，天然携带当前生成阶段最相关的上下文需求。

模型相关配置：Qwen2.5-1.5B α=4(loc)/16(agg)；LLaMA3-8B α=16(loc)/32(agg)。选择逻辑：aggregation 任务需要更大的 observation window 来识别各窗口中的关键 token，localization 任务只需足够定位相关窗口即可。

术语一般如何实现？如何使用？

取 attention 矩阵 A[n-α:n, :n-α] 子矩阵，沿 query dim sum 得 score vector t ∈ R^{n-α}。仅 group-first layer 执行，其余层共享 indices。与 FlashAttention 兼容（full attention 仅首层）。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---

## Information Localization vs Aggregation Task

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Information Localization vs Aggregation 是 WindowKV 提出的长上下文下游任务二分法，驱动 KV cache 窗口选择的 p 参数：

1. **Information Localization**（信息定位）：QA 类任务——在长文本中定位关键段落并回答。需完整窗口语义 → p=ω（保留窗口中全部 token）。示例：NarrativeQA, Qasper, HotpotQA, 2WikiMQA, Musique。

2. **Information Aggregation**（信息聚合）：摘要类任务——从多段落提取显著信息并浓缩。仅需窗口内关键 token → p<ω（仅保留 top-p 高分 token）。示例：GovReport, QMSum, MultiNews, Code completion, Few-shot tasks。

分类器：bert-base-cased, 9551 样本（8:1:1），accuracy 92.69%, recall 95.19%, F1 94.75%。消融验证（Figure 4）：分类错误导致策略-任务不匹配时性能显著下降。

术语一般如何实现？如何使用？

分类器训练：batch_size=16, lr=1e-6, dropout=0.5, 10 epochs, 8×A100 40G。推理时分类器输出 task_type 控制 p 参数，进而影响窗口内 Top-p token 选择和窗口得分 s_k。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---

## Heavy Hitter Oracle (H2O)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Heavy Hitter Oracle (H2O) 是 Zhang et al. (NeurIPS 2023) 提出的 KV cache 逐出方法。核心观察：attention 分数高度不均匀，少数 token（Heavy Hitters）贡献大部分 attention score。算法：每步 decode 后保留最近 w 个 token + 历史中累积 attention scores 最高的 (b-w) 个 token。

评分方式：score_k = Σ_q A_{q,k}（所有 query 对该 key 的累积注意力）。所有层使用统一 budget。局限性：token 级离散选择破坏语义连贯性，所有任务统一策略，不区分层间注意力密度差异。

伪代码：
```
for each decode step:
    A = softmax(Q @ K_cache^T / sqrt(d_k))
    scores += sum(A, dim=query)
    keep = topk(scores, budget-w) ∪ recent_tokens
    K_cache, V_cache = K_cache[keep], V_cache[keep]
```

术语一般如何实现？如何使用？

集成于 HuggingFace Transformers，与 FlashAttention 兼容。PyramidKV 仓库提供统一实现：https://github.com/Zefan-Cai/PyramidKV。WindowKV 将 H2O 作为 baseline 对比，在 KV cache=2048 下 LongBench avg：H2O 31.34 vs WindowKV 32.75 (Qwen2.5)，H2O 41.08 vs WindowKV 41.35 (LLaMA3)。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---

## LongBench

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

LongBench（Bai et al., ACL 2024）是首个中英双语多任务长上下文 benchmark，21 数据集/4750 样本，6 大类：Single-Doc QA、Multi-Doc QA、Summarization、Few-shot Learning、Synthetic、Code。英文 avg 6711 词，中文 avg 13386 字符。提供 LongBench-E 子集（均匀长度分布：0-4k/4k-8k/8k+）。

在 WindowKV 中用于评估 Qwen2.5-1.5B 和 LLaMA3-8B 在 KV size=512/1024/2048 下的 16 子任务。WindowKV 以 12% KV cache 取得最多次 SOTA。后续版本：LongBench v2 (ACL 2025, 503 多选题 8K-2M 词) 和 LongBench Pro (2026, 1500 样本 8K-256K)。

术语一般如何实现？如何使用？

官方仓库 https://github.com/THUDM/LongBench。WindowKV 使用标准 prompt，贪心解码。QA 类 F1，Summarization Rouge-L，Few-shot Accuracy，Code Edit Similarity。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---

## Needle-in-a-Haystack (NIAH)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Needle-in-a-Haystack (NIAH, Kamradt 2023) 测试 LLM 长上下文检索：在大量无关文本中嵌入关键信息（needle），要求模型在所有长度和深度位置检索并复现。Rouge-1 F1 评估。揭示了 "lost-in-the-middle" 现象——中间位置准确率大幅下降。

在 KV cache 压缩评估中广泛使用。WindowKV 在 LLaMA3-8B, context=8K, KV size=512 下评估，热力图显示 WindowKV 在所有深度位置的检索准确率优于 StreamingLLM/H2O/PyramidKV。

术语一般如何实现？如何使用？

Needle 放置在指定深度百分比（0%-100%），格式如 "The pass key is <N>."。Haystack 为重复填充文本。Rouge-1 F1 匹配 needle 原文。WindowKV 中 context=8K, KV size=512。

涉及论文标题：
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference
