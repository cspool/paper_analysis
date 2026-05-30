涉及论文标题：
- SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs

## Q-Former (Querying Transformer / 查询变换器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Q-Former 是 BLIP-2 (Li et al., ICML 2023) 提出的轻量级 Transformer 模块，用于在冻结的图像编码器和冻结的 LLM 之间建立桥梁。核心思想是使用一组可学习的固定长度 query vectors（查询向量），通过 cross-attention 从编码器输出中提取/蒸馏最相关的信息，输出固定长度的 soft tokens，再通过线性投影送入 LLM 的语言空间。Q-Former 本质上是一个信息瓶颈（information bottleneck）：它将任意长度的编码器输出压缩为固定数量（如 32 或 64 个）的 condensed tokens，解决了不同模态编码器输出长度不固定、维度过大的问题。

Q-Former 内部结构：N 个标准 Transformer block，每层包含 multi-head self-attention (MSA)、cross-attention (CA) 和 FFN。输入是 learnable query vectors X_Q ∈ ℝ^{M×d}（M 为 query 数量），cross-attention 中 query vectors 作为 query，冻结编码器输出 hidden states 作为 key 和 value。输出是经过 N 层处理后的 refined query vectors，保留了编码器输出中的关键信息。

Uni-MoE 中使用 Audio-QFormer（4 层）和 Speech-QFormer（4 层）分别处理 BEATs 音频编码器输出和 Whisper-small 语音编码器输出，每种配置独立的 learnable query vectors 和线性投影层。

从算法pipeline角度拆解术语：

Q-Former 的单层计算流程（以 Audio-QFormer 为例，式 7-11）：

```
输入: X_Q ∈ ℝ^{AM×d} (AM 个 learnable query vectors, AM 为 query 数)
      h_B = BEATs(audio) ∈ ℝ^{T×d'} (冻结音频编码器输出)

对每层 (共 4 层):
  # Step 1: Self-Attention among query vectors
  h_S = MSA(LN(X_Q)) + X_Q                    # 式(9)
  
  # Step 2: Cross-Attention with encoder output
  #   Query: h_S, Key/Value: h_B
  h_C = CA(LN(h_S), h_B) + h_S                # 式(10)
  
  # Step 3: FFN
  X_Q = MLP(h_C)                               # 式(11)

# 最终: 线性投影到 LLM 空间
A = Linear(X_Q_final)                          # 式(4) 的一部分
```

整个多模态 pipeline 中 Q-Former 的位置：
```
audio → BEATs Encoder (frozen) → h_B → Q-Former (4 layers) → Linear → Audio Tokens → LLM
speech → Whisper-small (frozen) → h_S → Q-Former (4 layers) → Linear → Speech Tokens → LLM
```

与 LLaVA 的对比：LLaVA 使用单个线性投影层连接视觉编码器和 LLM（更简单），而 Q-Former 使用 Transformer 架构的交叉注意力蒸馏（更强的信息提取能力，但参数更多）。Q-Former 适用于需要从长序列编码器输出中压缩信息的场景（如音频、语音），线性投影适用于编码器输出已较紧凑的场景（如 CLIP 的图像特征）。

术语一般如何实现？如何使用？

典型实现基于 BLIP-2 的 Q-Former 架构（HuggingFace Transformers）。在 Uni-MoE 中：(1) 为 Audio 和 Speech 分别初始化独立的 Q-Former（4 层 Transformer），各有独立的 learnable query vectors；(2) 阶段一训练时仅训练 Q-Former 参数和投影层，冻结编码器和 LLM；(3) 学习率 2e-5，AdamW 优化器，cosine scheduler。Q-Former 的训练目标是 cross-entropy generation loss：生成的 text 与 ground truth 之间的交叉熵。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts

## Cross-Modality Alignment (跨模态对齐)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cross-Modality Alignment（跨模态对齐）是多模态大模型训练的第一阶段（Alignment Stage），目标是将不同模态（图像、视频、音频、语音等）的编码器输出映射到统一的 LLM 语言表示空间，使 LLM 能"理解"非文本模态的输入。该阶段仅训练模态连接器（Connector），冻结编码器和 LLM，通过最小化生成文本的交叉熵损失来学习模态-语言映射。

Uni-MoE 的跨模态对齐阶段针对每种模态独立执行：(1) 视觉对齐——使用 CLIP-V + MLP 线性投影（沿用 LLaVA 预训练权重）；(2) 语音对齐——使用 Whisper-small + Speech-QFormer + 线性投影，训练数据为 Common Voice (1.7M 短语音)；(3) 音频对齐——使用 BEATs + Audio-QFormer + 线性投影，训练数据为 WavCaps/AudioCaps/MELD/Clotho (194K)。

从算法pipeline角度拆解术语：

跨模态对齐训练（以语音对齐为例，对应 Algorithm 1 Stage 1）：

```
for each step:
    (x, y) = sample(PD_speech)          # 采样语音-文本对
    x_speech = Whisper(x)               # 冻结语音编码
    x_q = Speech-QFormer(x_speech)      # Q-Former 蒸馏
    x_tokens = Linear(x_q)              # 线性投影到 LLM 空间
    prediction = LLM(x_tokens)          # 冻结 LLM 前向
    loss = CE(prediction, tokenize(y))  # 交叉熵生成损失
    # 仅更新: Q-Former + Linear projection 参数
    θ = θ - α ∇_θ loss
```

此阶段的关键性质：
- 仅训练 Connector（Q-Former + 投影层），编码器和 LLM 冻结
- 每种模态独立训练，互不干扰
- 使用模态-文本配对数据（如 speech-transcription pairs）
- Loss 为标准语言建模交叉熵
- 学习率 2e-5，global batch size=32，AdamW

术语一般如何实现？如何使用？

在 Uni-MoE 中，跨模态对齐在 2 块 A100 GPU 上进行，分别处理 1.7M 短语音数据和 194K 音频字幕数据。视觉对齐部分复用 LLaVA 已有的 CLIP+MLP 视觉连接器（预训练完成）。跨模态对齐是多模态 LLM 训练的必要第一步——没有此阶段，LLM 无法将非文本模态的 continuous embeddings 解释为有意义的语义信息。与 Meta-Transformer 的统一 tokenizer 思路不同，Q-Former 方法通过 cross-attention 机制实现了更具表达力的模态特征蒸馏。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts

## Modality-Specific Expert (模态特定专家)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Modality-Specific Expert（模态特定专家）是 Uni-MoE 提出的 MoE 多模态 LLM 训练策略中的核心概念：在将 MoE 架构引入多模态 LLM 时，不同专家（FFN）在不同的单模态/跨模态数据上分别预训练，使其发展出对不同模态的偏好和处理专长。训练完成后，每个 expert 在其专业模态的 token 上被 router 优先激活（如音频 tokens → Expert 4，图像 tokens → Expert 2）。

Uni-MoE 定义了 8 个单模态专家训练任务（Task1-Task8），分别训练不同用途的专家：Task2 用 LLaVA-Instruct-150K (T-I，文本-图像) 训练图像专家；Task3 用 LLaVA-Instruct-150K (I-A，语音-图像) 训练语音-图像专家；Task7 用 RACE-Audio + LibriSpeech 训练长语音专家；Task8 用 WavCaps/AudioCaps/MELD/Clotho 训练音频专家。

与标准 MoE（所有专家初始相同，简称 pure MoE）的对比：pure MoE 中专家缺乏模态区分性，routing 分布在各模态间更均匀（Figure 8, Figure 10），无法有效利用多模态数据的结构差异；modality-specific 专家天然形成模态偏好分布（Figure 4-5），router 学习到特定路由模式。

从算法pipeline角度拆解术语：

Modality-Specific Expert 训练流程（对应 Algorithm 1 Stage 2）：

```
# 对每种模态 M，独立训练对应专家
for each modality M:
    # 复制阶段一训练好的权重
    copy_weights_from_stage1()
    
    for each step:
        (x, y) = sample(D_M)                    # 采样该模态的跨模态指令数据
        x_M = Connector(x)                       # 模态投影
        prediction = LLM(x_M, E[h(i_M)])        # 前向，激活目标专家 E
        loss = CE(prediction, tokenize(y))
        # 仅更新: LoRA (MLP in LLM) + 投影层参数
        θ = θ - α ∇_θ loss

# 得到: {Expert_1→image, Expert_2→image-text, Expert_3→speech-image, Expert_4→audio, ...}
```

阶段三加载这些预训练专家到 MoE layers，通过 LoRA 联合微调。

术语一般如何实现？如何使用？

在 Uni-MoE 的具体实现中：(1) 阶段二从阶段一 checkpoint 初始化，每个专家独立训练 1 epoch；(2) 使用 LoRA（rank=64, alpha=16）仅微调 MLP 在 LLM 中的参数和投影层；(3) 学习率 4e-5（LoRA）和 3e-5（投影层），global batch size=16，2 块 A100 GPU；(4) 训练后将各专家 FFN 权重分别保存，阶段三加载到 MoE layers 的不同 expert slots。这种方法的优势在于：(a) 各专家发展出明确的模态专长，实现自然 load balancing；(b) router 更容易学习到有意义的 token-to-expert 映射；(c) 在混合多模态数据上训练收敛更快更稳定（Figure 3 蓝色线 vs 橙色线）。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts

## Sparse Router / Token-Level Gating (稀疏路由 / Token级门控)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparse Router（稀疏路由器）是 MoE 架构中决定每个 token 分配给哪些专家的组件。它是一个可学习的线性函数 f(x) = W · x（W ∈ ℝ^{d×M}，d 为 hidden state 维度，M 为专家数），输出每个专家对当前 token 的分配概率（通过 softmax 归一化），然后通过 top-k 选择激活概率最高的 k 个专家。在 Uni-MoE 中，每个 MoE 层有独立的 Router，对每个 token 独立进行 top-2 专家选择，router 参数在阶段三与 LoRA 参数一起训练。

与传统 dense FFN 每个 token 激活所有参数不同，sparse router 实现了 token-level conditional computation——不同 token（不同模态、不同语义）激活不同的专家子集。其在多模态 MoE 中的特殊意义在于：router 可以学习到"模态感知"的路由策略，将图像 tokens 路由到图像专长专家，音频 tokens 路由到音频专长专家。

从算法pipeline角度拆解术语：

Sparse Router 在每个 MoE 层的计算流程（Uni-MoE 式 16-17）：

```
输入: X_l^s ∈ ℝ^{T×d}   # 经过 self-attention 后的 hidden states (T 个 tokens)

# Step 1: Router 计算每个 token 的专家分配概率
logits = X_l^s @ W_router              # W_router ∈ ℝ^{d×M}, logits ∈ ℝ^{T×M}
P = softmax(logits, dim=-1)            # P ∈ ℝ^{T×M}, 每行和为1

# Step 2: Top-K 选择
P_topk, indices = top_k(P, k=2, dim=-1)  # 每 token 选 top-2 专家

# Step 3: 归一化选中概率（可选）
P_topk = P_topk / sum(P_topk, dim=-1)    # 使选中概率和为 1

# Step 4: 加权累加专家输出
output = zeros_like(X_l^s)
for each token t:
    for each selected expert e_i (i=1..k):
        output[t] += P_topk[t, i] * Expert_FFN_{e_i}(X_l^s[t])   # 式(17)
```

Uni-MoE 可视化分析（Figure 4-5）揭示的 router 行为：
- 在 text-audio 输入下，专家 2 和 4 几乎主导所有 token 分配
- 在 text-image 输入下，专家 2（图像预训练）在初始层大幅领先
- 在 video 输入下（含音频+视觉），负载在各层更均衡
- 专家 1（原始 LLaVA MLP）在各场景下参与度最低——暗示预训练对专家专业化至关重要

术语一般如何实现？如何使用？

典型实现：Router 是一个简单的 `nn.Linear(hidden_size, num_experts)`，输出经 softmax 后 top-k。训练时与 LoRA 参数共同更新。在 Uni-MoE 中，router 在阶段三中与 LoRA 参数一起训练（学习率 4e-5），不使用 auxiliary balancing loss 时仍能学到有效的 routing 模式（因为 modality-specific 预训练专家提供了自然的路由信号）。关键权衡：k 值增大（如从 1→2）提升模型表达能力但增加计算量；Uni-MoE 消融实验（Table 7a）显示 top-2 在各 benchmark 上优于 top-1。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models
- Upcycling Large Language Models into Mixture of Experts

## Mixture-of-Experts (MoE) in Large Language Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixture-of-Experts (MoE) 是一种神经网络架构范式，通过将模型的 FFN（前馈网络）层替换为多个并行的"专家"子网络（Expert FFNs），并引入路由机制（Router/Gating Network）动态地为每个输入 token 选择激活哪些专家。与传统 Dense 模型每个 token 激活全部参数不同，MoE 实现了条件计算（conditional computation）：每个 token 仅激活全部专家中的 top-k 个（通常 k=2），从而在保持模型总参数量（capacity）极大的同时，控制实际激活的参数量（compute cost）。MoE LLM 通常由 m 个 Transformer 层组成，每层包含 n 个专家（Expert FFN），Router 为每个 token 输出 n 维的 softmax 概率分布，选择 top-k 个专家。最终输出为 k 个选中专家输出的加权和：$\text{output} = \sum_{i=1}^{k} P_{\text{topk},i} \cdot \text{Expert}_{\text{FFN},i}(x)$。典型 MoE LLM 包括 Mixtral 8x7B、DeepSeek-MoE-16B 等。

MoE 的核心优势在于：通过增加专家数量（扩容量）而非激活参数量（控计算），在固定计算预算下获得更强的表示能力。但代价是：(1) 专家之间的协作机制尚不清晰；(2) 大量专家导致巨大的存储和部署压力；(3) Router 坍塌需要 auxiliary load balancing loss 来缓解。

从算法pipeline角度拆解术语：

MoE 层的计算流程（以 DeepSeek-MoE 为例，normal experts + shared experts）：

```
输入: x in R^{T x d}  # T 个 token 的 hidden states

# Step 1: Router 计算专家分配
logits = x @ W_router                 # W_router in R^{d x (n_normal + n_shared)}
probs = softmax(logits, dim=-1)       # probs in R^{T x (n_normal + n_shared)}

# Step 2: Top-K 选择
topk_probs, topk_indices = top_k(probs[:, :n_normal], k=2)

# Step 3: Shared experts 始终激活
shared_out = sum_{j=1}^{n_shared} SharedExpert_FFN_j(x)

# Step 4: Routed experts 加权输出
routed_out = sum_{i=1}^{k} topk_probs[:,i] * NormalExpert_FFN_{topk_indices[:,i]}(x)

# Step 5: 最终输出
output = shared_out + routed_out + x   # residual connection
```

MoE 模型分析中的关键矩阵——Expert Activation Matrix（专家激活矩阵）：对于 m 层、每层 n 个普通专家的模型（共 $N_e = m \times n$ 个专家），在 $N_s$ 个样本上收集每个 token 的 router 分配权重 $\alpha(i)_{t,j,k}$，按句子聚合为：$v_{i,j,k} = \sum_{t=1}^{T} \alpha(i)_{t,j,k}$，构造 $X \in \mathbb{R}^{N_e \times N_s}$。该矩阵是分析专家协作模式的基础数据。

术语一般如何实现？如何使用？

典型实现：基于 HuggingFace Transformers，MoE 层通过 `MixtralSparseMoeBlock` 或 DeepSeek MoE 模块实现，Router 为 `nn.Linear(hidden_size, num_experts)`。训练时使用辅助负载均衡损失（auxiliary load balancing loss）防止路由坍塌。推理时，MoE 支持专家并行（Expert Parallelism, EP）——将不同专家分布在不同 GPU 上，通过 all-to-all 通信完成 token dispatch 和 combine。

涉及论文标题：
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models
- Upcycling Large Language Models into Mixture of Experts
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts

Uni-MoE 将 MoE 引入统一多模态 LLM 场景，基于 Vicuna-7B（LLaMA-7B），将 LLM 中部分 FFN 层替换为稀疏 MoE 层。每层包含 4~8 个专家（Expert FFN），Router 为线性层 $W \in \mathbb{R}^{d \times M}$，对每个 token 计算 softmax 概率并选择 top-2 专家。配置包括 Uni-MoE-7B×4-Top2（16 层 MoE，4 专家/层，激活 8.9B/总 13.2B）和 Uni-MoE-7B×4-Top2†（32 层 MoE，激活 11.1B/总 19.7B）。Uni-MoE 的特殊之处在于：(1) 每个专家在不同模态数据上分别预训练（阶段二），发展出模态偏好；(2) 使用 LoRA 微调替代全量专家参数更新，rank=8/alpha=16；(3) 支持 expert-level model parallelism 和 modality-level data parallelism；(4) 实验发现 auxiliary balancing loss 在 pure MoE（相同初始专家）中有效，但在 mixture MoE（预训练多样化专家）中不加 aux loss 反而更好——因为专家已自然发展出模态分化。

## Expert Collaboration Patterns (专家协作模式)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Collaboration Patterns（专家协作模式）是指在 MoE LLM 中，跨层（甚至同层）的不同专家之间存在的稳定的、可重复的共激活（co-activation）关系。与以往将每个专家视为独立实体的分析范式不同，协作模式视角认为：MoE 模型的最终输出并非由单个专家独立完成，而是由一组跨层的专家协同工作产生。例如，当处理数学推理任务时，可能同时出现 Layer 5 Expert 21（负责数值提取）和 Layer 6 Expert 3（负责逻辑运算）的频繁共激活，它们共同构成了一个"数学推理"的功能模块。

协作模式可以通过专家激活矩阵 X 的分析来发现：对 X 进行稀疏字典学习分解 $X \approx D \cdot R$，字典 D 的每个 atom（列向量）编码了一组共激活的专家集合（即一个协作模式），稀疏编码 R 控制各模式在不同输入样本上的参与度。实验验证：(1) 60% 的字典模式对应于穷举搜索中 top 10% 最高频的专家组合；(2) 语义相近领域（数学/物理/计算机科学）的协作模式分布相似度高，语义不同的领域（数学/法律）分布差异大；(3) 层级分解揭示从粗到细的语义层级——高层字典捕获"数学计算"等大类，深层字典细化为"日期识别"、"符号处理"等子任务。

从算法pipeline角度拆解术语：

协作模式发现的完整流程：

```
# 输入：MoE LLM（m 层，n 专家），数据集 S（N_s 个样本）
# 输出：协作模式字典 D 和稀疏编码 R

# Phase 1: 构建 Expert Activation Matrix
for each sample i in S:                  # i = 1..N_s
    for each token t in sample:
        alpha_t = Router(x_t)              # router 为每个 token 输出 n x m 个权重
    v_{i,j,k} = sum_t alpha(i)_{t,j,k}     # 句子级聚合，式(1)
X = stack(v)                             # X in R^{N_e x N_s}, N_e = m x n

# Phase 2: 层级稀疏字典学习 (HSDL)
D_1, R_1 = sparse_dict_learn(X, N_p1)    # Layer 1: X ≈ D_1 * R_1
for k in 2..K:
    D_k, R_k = sparse_dict_learn(D_{k-1}, N_pk)  # 递归分解
    L = L_sparse + lambda1*L_hier + lambda2*L_rec

# Phase 3: 结果解读
# D_1 的每个 atom: 一组粗粒度协作专家集合
# D_K 的每个 atom: 细粒度的子模式
# R_k 的每列: 各模式在不同样本上的激活强度
```

术语一般如何实现？如何使用？

实现方式：对每个输入样本前向传播，在 MoE 层的 router 输出位置插入 hook 记录激活权重，按句子求和得到激活矩阵 X。然后使用稀疏字典学习算法对 X 进行分解。该模式分析可用于：(1) 模型可解释性——可视化哪些专家协作处理何种语义任务；(2) 专家剪枝——识别并保留高贡献的协作模式；(3) 领域自适应——分析不同领域输入下的协作模式差异。

涉及论文标题：
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models

## Hierarchical Sparse Dictionary Learning (HSDL) for MoE Analysis

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hierarchical Sparse Dictionary Learning (HSDL) 是一种专门用于从 MoE LLM 的专家激活数据中提取多粒度协作模式的层级无监督学习方法。它扩展了传统的单层稀疏字典学习，通过对字典矩阵递归分解来捕获从粗到细的专家协作层次结构：$D_{k-1} \approx D_k \cdot R_k$（式 3）。第一层从原始激活矩阵 X 开始分解 $X \approx D_1 \cdot R_1$，随后每一层对上一层字典进一步分解。

HSDL 引入三个关键约束：(1) 稀疏性约束 $L_{\text{sparse}} = \|R_{k,i,:}\|_{\infty}$——防止某些字典元素主导，确保稀疏激活；(2) 层间一致性约束 $L_{\text{hier}} = \sum_j \|R_{k+1,j}\|_1 \cdot \|R_{k,j}\|_1 / N$——控制跨层字典学习的影响传递；(3) 重构误差项 $L_{\text{rec}} = \sum_j \|D_{k,j} - (D_{k+1}R_{k+1})_j\|_1 \cdot \|R_{k,j}\|_1 / N$——保证层间关系一致。总损失 $L_{\text{total}} = L_{\text{sparse}} + \lambda_1 L_{\text{hier}} + \lambda_2 L_{\text{rec}}$（式 7）。

从算法pipeline角度拆解术语：

HSDL 的层级计算流程：

```
# Layer 1: 从原始激活矩阵开始
D_1, R_1 = argmin_{D,R} ||X - D*R||_F^2 + alpha*||R||_1   # 标准稀疏编码
s.t. ||D_j||_2 <= 1  for all columns j

# Layer k (k >= 2): 对上一层字典递归分解
D_k, R_k = argmin_{D,R} ||D_{k-1} - D*R||_F^2 + alpha*||R||_1

# 多目标联合优化（每层同时考虑三个损失）：
# L_sparse:  R_k 的每一行的 L_inf 范数，鼓励稀疏激活
# L_hier:    跨层 R 矩阵的 L1 范数乘积
# L_rec:     字典重构误差的加权 L1 范数

# 优化: 交替更新 D_k 和 R_k
# - 固定 D_k, 用 Lasso/CD 更新 R_k
# - 固定 R_k, 用 block-coordinate descent 更新 D_k
```

术语一般如何实现？如何使用？

可使用 scikit-learn 的 `MiniBatchDictionaryLearning` 或 SPAMS 库实现单层字典学习，HSDL 在此基础上增加递归分解和三个约束的联合优化。论文在 phi-moe 模型上用 MMLU-pro 数据集（2,812 样本，5 领域）验证——60% 的字典模式对应 top 10% 最高频穷举组合。HSDL 的应用场景：(1) MoE 模型可解释性——层级语义标注揭示模型如何从粗到细理解任务；(2) 专家剪枝的输入信号；(3) 领域特化分析。

涉及论文标题：
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models

## Contribution-Aware Expert Pruning (CAEP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Contribution-Aware Expert Pruning (CAEP) 是一种基于专家协作模式的 MoE 模型剪枝算法。与传统的独立专家剪枝方法（SEER-MoE 基于路由分数截断、GEM 基于输出影响力排序）不同，CAEP 利用 HSDL 提取的字典矩阵 D 和稀疏编码 R 计算每个专家的综合贡献分数，考虑专家在协作模式中的结构角色，而非仅按个体指标排序。核心理念：删除专家时应考虑其所属的协作模式是否完整——如果一个专家是关键协作模式中的必要成员，即使其个体路由分数不高也应保留。

算法流程：(1) 从 R 计算模式级贡献 $R_{\text{sum}} = \sum_{j} R_{:,j}$，结合 D 计算专家的总贡献分数 $e$，降序排序；(2) 以 $k_1$-分位数为阈值生成初始 mask；(3) 迭代：找出贡献最小的模式并移除，重算贡献分数并更新 mask，直到保留专家数达到目标 $(1-k_2) \cdot N_e$。实验效果：在 DeepSeek-MoE-16B 上，CAEP 剪枝 25% 专家后平均 accuracy 为 0.612，优于 SEER-MoE (0.5872) 和 GEM (0.5870)，在 OBQA 上从 0.420 提升至 0.473。

从算法pipeline角度拆解术语：

```
输入: D in R^{N_e x N_p}, R in R^{N_p x N_s}, k_1, k_2
输出: expert mask m in {0,1}^{N_e}

# Step 1: 计算贡献分数
R_sum = sum_{j=1}^{N_s} R[:,j]         # 每个 pattern 的样本级总激活
D_sum = D @ R_sum^T                     # 专家-模式贡献矩阵
e = sum_{i=1}^{N_p} D_sum[:,i]          # 每个专家的总贡献分数

# Step 2: 初始阈值 mask
e_sorted = sort_descending(e)
threshold = e_sorted[ceil(k_1 * N_e)]
m = (e >= threshold)

# Step 3: 迭代剪枝
while count_ones(m) > (1 - k_2) * N_e:
    i* = argmin_i R_sum[i]              # 最少使用的协作模式
    D = delete_column(D, i*)
    R = delete_row(R, i*)
    recompute R_sum, D_sum, e
    m = (e > threshold)

return m
```

术语一般如何实现？如何使用？

基于 PyTorch/NumPy 实现。剪枝后参数量（DeepSeek-MoE-16B）：仅剪枝 normal experts，保留 shared experts，新参数量 = 16.4 - 14.7 x k_2 B（式 10）。适用场景：(1) MoE 部署压缩；(2) 领域特化剪枝——针对特定领域保留相关协作模式；(3) 替代独立评估的剪枝方法。

涉及论文标题：
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models

## Expert Activation Matrix (专家激活矩阵)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Activation Matrix（专家激活矩阵）是从 MoE LLM 中提取的、记录每个专家在每个输入样本上激活强度的二维数据矩阵，是分析 MoE 模型内部行为的基础数据结构。给定 m 层、每层 n 个专家的 MoE LLM（总 $N_e = m \times n$ 个专家），在 $N_s$ 个样本前向传播时，对每个 token 的 Router softmax 输出进行记录和聚合，构造矩阵 $X \in \mathbb{R}^{N_e \times N_s}$，其中 $X_{e,i}$ 表示第 e 个专家在第 i 个样本上的句子级总激活强度。

构造过程：对样本 $S_i$ 的第 t 个 token，Router 为第 j 层的第 k 个专家分配权重 $\alpha(i)_{t,j,k}$。句子级激活通过求和聚合：$v_{i,j,k} = \sum_{t=1}^{T} \alpha(i)_{t,j,k}$（式 1）。将每个 $(j,k)$ 展开为一行、每个样本为一列，得到 $X \in \mathbb{R}^{N_e \times N_s}$。矩阵具有非负性（激活值在 [0,T] 范围）和天然稀疏性（每 token 仅激活 top-k 个专家）。

从算法pipeline角度拆解术语：

```
for each sample s_i in S:
    for each token t in s_i:
        for each layer j in 1..m:
            alpha_j = softmax(W_router[j] @ x_t)   # Router 输出
            for each expert k in 1..n:
                record alpha(i)_{t,j,k}

    for each (j,k):
        v_{i,j,k} = sum_{t=1}^{T} alpha(i)_{t,j,k}  # 式(1): 句子级聚合

X[e, i] = v_{i,j,k}   where e = index(j, k)
# X shape: (N_e, N_s), N_e = m * n
```

术语一般如何实现？如何使用？

使用 PyTorch forward hook 在 MoE 层 Router softmax 后捕获激活值。对 shared experts 激活值恒为 1。激活矩阵可用于：(1) HSDL 分解发现协作模式；(2) 专家使用频率统计；(3) 领域偏好分析（计算不同领域激活分布的 cosine similarity）；(4) CAEP 剪枝的贡献评估。

涉及论文标题：
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models

## Upcycling (Sparse Upcycling / 稠密到MoE转换)

术语是什么？

Upcycling（在 MoE 语境中）是将已预训练的稠密（dense）语言模型转换为稀疏 MoE（Mixture of Experts）模型的训练技术。核心流程是：(1) 复制稠密模型的 MLP（FFN）权重来初始化 MoE 的多个 expert；(2) 随机初始化 Router；(3) 用相对较少的 token（通常是预训练 token 数的 10% 以内）继续训练，使 Router 学会合理的路由策略、expert 逐步分化。Upcycling 的目标是：与从头训练 MoE 相比，大幅降低总计算量；与续训稠密模型相比，利用 MoE 架构获得更高模型容量，从而得到更好的下游任务性能。

其理论基础在于：预训练的稠密 MLP 层已经学到了丰富的通用知识（language understanding、reasoning 等），将这些知识作为多个 expert 的初始状态比随机初始化更高效。每个 expert 从相同的起点出发但因为在 upcycling 阶段收到的 token 不同，逐渐通过梯度更新发生分化，最终形成不同专长的专家。

从算法pipeline角度拆解：

Upcycling 的完整流程（以 Nemotron-4 15B → E8G1T2 为例）：

```
# === 阶段 0: 准备稠密 Checkpoint ===
dense_model = load_checkpoint("Nemotron-4-15B")
# dense_model 已预训练 8T tokens, MMLU 59.3

# === 阶段 1: 初始化 MoE 架构 ===
# 对每个需要替换为 MoE 的 Transformer 层 (每 2 层中的 1 层):
for layer in moe_layers:
    # 复制 MLP 权重 E 次 → 初始化为 E 个 expert
    layer.experts = [copy(dense_model[layer].mlp) for _ in range(E)]
    # 随机初始化 Router
    layer.router = random_init((d_model, E))
    # 应用 Weight Scaling
    scale = (E * G**2 / T) ** (1/3)
    for expert in layer.experts:
        expert.W1 *= scale
        expert.W2 *= scale

# === 阶段 2: Upcycling 训练 ===
# 训练数据: 续训数据 blend，1T tokens
# 学习率: warmup → peak 3e-4 → cosine decay → 1/100 of pretraining min LR
# Load balancing aux loss: coeff = 1e-2
# 分布式策略: Megatron-LM (DP + TP + EP)
for batch in training_data:
    # 标准 MoE forward pass
    output = moe_model(batch)
    loss = L_LM + 0.01 * L_aux  # 语言模型损失 + 负载均衡损失
    loss.backward()
    optimizer.step()
```

关键设计决策：
1. **学习率重置**：upcycling 必须使用高学习率（如 2e-4 或 3e-4），而非 fine-tuning 的小学习率。原因是 MoE 从稠密模型的局部最优出发，高学习率帮助逃离该局部最优，促进 expert 分化。
2. **Router 设计**：推荐 softmax-then-topK 而非 topK-then-softmax，因为前者保留了 Router 输出的绝对值信息。
3. **大批量**：推荐 4M+ tokens 的 batch size，因为每个 expert 只收到部分 tokens，大量样本能稳定梯度并降低负载均衡损失的噪声。
4. **Weight Scaling**：对 fine-grained MoE 至关重要，补偿因 expert 拆分导致的输出缩放。

术语一般如何实现？

NVIDIA 开源实现位于 Megatron-LM: [moe/upcycling](https://github.com/NVIDIA/Megatron-LM/tree/0431153bf1b5c405057b158189c260107d8b7c3a/megatron/core/transformer/moe#upcycling)。NeMo 也集成了 online upcycling 功能（用户提供 dense checkpoint + parallel training config，自动产生 MoE 模型）。后续工作 "Llama 3 Meets MoE" 将 upcycling 应用于 Llama 3-8B，使用 <1% 预训练计算量实现 ~1.2% benchmark 平均提升。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts

## Virtual Group Initialization (虚拟组初始化)

术语是什么？

Virtual Group Initialization 是 NVIDIA 为 fine-grained MoE upcycling 提出的 Router 和 Expert 初始化策略。在 fine-grained MoE 中，每个 expert 的 hidden size 被缩小为原始 FFN 的 1/G（G 为 granularity），需要 T 个 expert 的加权输出才能重建完整 dense MLP 的功能。普通的随机 Router 初始化无法保证初始的 top-T 选择恰好覆盖所有 G 个 shard，导致 upcycling 初始阶段 loss 极高且无法收敛。Virtual Group Init 通过将 Router 权重分组复制，确保每一个 expert group 内部初始权重相同，从而保证 top-T 的初始选择均匀分布在 G 个 group 上。

核心原理——两个保证：
1. **Router 分组**：N = E×G 个 expert 分为 G 个 group，每组 E 个 expert 都是同一个 dense MLP shard 的副本。Router 权重在组内初始化相同，组间可以不同。
2. **Top-T 均匀覆盖**：因为每组内 Router 权重相同，top-T 操作自然从不同组中选择 expert（当 T ≥ G 时），保证每个 shard 都被选到至少一次。

从算法pipeline角度拆解：

以 E2G2T2（4 experts, 2 shard, top-2）为例：

```
# 稠密 FFN: y = FFN_0(x) + FFN_1(x)  (分成 2 个 shard)

# === 错误做法 (Naive) ===
FFN = [FFN_0, FFN_1, FFN_0, FFN_1]  # 复制
router = random_init([0.4, 0.2, 0.3, 0.1])
router_top2 = [0.4, 0.0, 0.3, 0.0]  # 选 expert 0 和 2 (都是 FFN_0!)
# MoE 输出 = 0.4*FFN_0 + 0.3*FFN_0 != FFN(x)  ← 出错!

# === Virtual Group Init (正确做法) ===
# Group 0 (experts 0,1): 都是 FFN_0
# Group 1 (experts 2,3): 都是 FFN_1
router = [0.3, 0.3, 0.2, 0.2]  # 组内相同
router_top2 = [0.3, 0.3, 0.0, 0.0]  # 选 expert 0 (Group 0) 和 expert 1 (Group 0)
# 但这里只选了 Group 0! topK=2 不足以覆盖 G=2 组 → 需要 topK >= G

# === 实际使用情况 (E8G8T8) ===
# G=8 个 shard, topK=8 → 恰好每个 group 选 1 个
# 初始 MoE 输出 = (1/(E*G)) * (T/G) * dense_output ≈ dense_output / (E*G)
# Weight Scaling 补偿该缩放因子
```

术语一般如何实现？

在 Megatron-LM upcycling 模块中实现。先按 intermediate dimension 切分 dense FFN 权重 → 复制 shards → 构建 Virtual Group Router（每个 group 内复制相同权重）→ 应用 Weight Scaling。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts

## Fine-grained MoE (Granular MoE / 细粒度混合专家)

术语是什么？

Fine-grained MoE（细粒度混合专家）是 MoE 架构的一个子类，通过缩小每个 expert 的 FFN hidden size（乘以因子 1/G）并增加激活的 expert 数量（乘以因子 G），在保持总计算 FLOPs 不变的前提下，使用更多但更小的 expert。由三个关键超参数定义：(1) E（Expansion Rate）：MoE 层总参数量是 dense MLP 的 E 倍；(2) G（Granularity）：expert hidden size 是 dense FFN hidden size 的 1/G；(3) T（TopK）：每 token 路由到的 expert 数量。总 expert 数 N = E × G。

与 coarse-grained MoE（如 Mixtral 8x7B，E=8, G=1, T=2）相比，fine-grained MoE（如 E8G8T8，64 experts 每个 1/8 hidden size）的理论优势在于：更多 expert 提供更细粒度的路由选择，理论上可实现更精准的 expert 专业化。

从算法pipeline角度拆解：

Fine-grained MoE 层的计算（E8G8T8 为例）：

```
# 参数规模：
# dense FFN: W1(d,h), W2(h,d) 其中 h = intermediate hidden (如 4*d)
# fine-grained expert: W1_i(d, h/G), W2_i(h/G, d)

# 前向传播
x = attention_output   # (S, d)
gate = softmax(x @ W_r)  # (S, N), N=64
topk_val, topk_idx = topk(gate, T=8)  # 选 8 个 expert

# 每个 expert 计算 (hidden smaller by G=8)
output = zeros(d)
for (val, idx) in zip(topk_val, topk_idx):
    h_i = activation(x @ W1_{idx})          # (h/8,)
    output += val * (h_i @ W2_{idx})        # (d,)
```

实际挑战：
1. **Upcycling 困难**：不能简单复制 dense MLP 权重，因为 expert 尺寸不匹配。Virtual Group Init 解决了这个问题。
2. **输出缩放**：多个小 expert 的加权输出需要缩放补偿。Weight Scaling 解决。
3. **系统开销**：更多 expert 意味着更多 all-to-all 通信轮次和更小的 GEMM 尺寸，降低 GPU FLOP utilization。

论文实验结论（Nemotron-2B 和 Nemotron-4 15B）：增加 granularity 到 G=8 有收益，但继续增加到 G=16/32 有 diminishing returns。64 experts (E8G8T8) 在 0.1T tokens 消融中最优，但在 1T+ tokens 大规模训练中 coarse-grained (E8G1T2) 和 fine-grained (E8G8T8) 最终 loss 趋同。Fine-grained MoE 的 scaling law 详见 Krajewski et al., 2024 "Scaling Laws for Fine-Grained Mixture of Experts"。

术语一般如何实现？

Megatron-LM 和 NeMo 框架支持 fine-grained MoE 的配置和训练。关键实现要点：(1) expert 权重按 G 因子缩小 intermediate dimension；(2) 需要 Virtual Group Init + Weight Scaling 来稳定 upcycling 训练；(3) 使用 scattermoe (Tan et al., 2024) 等优化来降低 fine-grained MoE 的内存和通信开销。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

## Weight Scaling for MoE Upcycling (MoE Upcycling 中的权重缩放)

术语是什么？

Weight Scaling 是 NVIDIA 为 MoE upcycling 提出的权重初始化补偿技术。在 upcycling 过程中，expert 输出因 Router softmax 和 fine-grained sharding 的综合效应被缩小——对 E×G 个 expert、top-T 路由，初始每个 expert 的输出大约被缩放为原来的 1/(E×G)，而 T 个 expert 的加权和大致等于 (T/(E×G²)) × dense_output。为了补偿这个缩放，对每个 expert 的 W1 和 W2 权重同时乘以缩放因子：

$$\text{scale} = \sqrt[3]{\frac{E \times G^2}{T}}$$

该公式对 Squared-ReLU 激活函数从第一性原理推导得出，但实验证明对 SwiGLU 激活同样有效。Weight Scaling 同时适用于 coarse-grained MoE（G=1 时 scale = ³√(E/T)）和 fine-grained MoE。

从算法pipeline角度拆解：

Weight Scaling 的推导（以 Squared-ReLU 为例）：

```
# MoE 激活（uniform distribution 假设, iteration 0）:
# P = P_1 = P_2 = ... = P_T = 1/(E*G)
MoE_activation = P * sum_{i=1}^{T} E_i(x)
               = (1/(E*G)) * (T/G) * dense_activation
               = T/(E*G^2) * dense_activation

# Squared-ReLU: output = W2 @ (ReLU(W1 @ x))^2
# 性质: squared_relu(k*w) = k^2 * squared_relu(w)
# 所以: 若 W1 *= k1, W2 *= k2
#       expert_output *= k1^2 * k2

# 需要 k1^2 * k2 = E*G^2/T  (补偿缩放)
# 取 symmetrically: k1 = k2 = (E*G^2/T)^{1/3}
```

实验验证：
- 对 E8G8T8 (64 experts top-8, 1/8 size): scale = ³√(8×64/8) = ³√64 = 4.0
- 对 E8G1T1 (8 experts top-1): scale = ³√(8×1/1) = 2.0
- w/ weight scaling 比 w/o 低 1.5% loss (Nemotron-4 15B E8G1T1)

术语一般如何实现？

在 Megatron-LM upcycling 初始化代码中，计算缩放因子后对每个 expert 的 W1 和 W2 进行 element-wise 乘法。论文同时尝试了替代方案（MoE output scaling 和 post expert layernorm），但 weight scaling 效果最优且实现最简单（不改变模型架构）。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts

## Load Balancing Auxiliary Loss (负载均衡辅助损失)

术语是什么？

Load Balancing Auxiliary Loss（负载均衡辅助损失）是 MoE 训练中用于防止"专家坍塌"（expert collapse，即大部分 token 被路由到极少数 expert 而其他 expert 闲置）的辅助损失函数。最早由 Switch Transformer (Fedus et al., 2022) 和 ST-MoE (Zoph et al., 2022) 提出。其核心思想是：在语言模型 loss 之外添加一项鼓励均匀路由的惩罚项，使 Router 在优化语言模型的同时也更均匀地分配 token。

标准公式（本文使用）：
$$L_{\text{aux}} = E \cdot \sum_{e=1}^{E} f_e \cdot P_e$$

其中：
- $f_e = \frac{1}{T} \sum_{t \in \text{batch}} \mathbf{1}[\text{token } t \text{ routed to expert } e]$（expert e 实际收到的 token 比例）
- $P_e = \frac{1}{T} \sum_{t \in \text{batch}} \text{softmax\_prob}[t, e]$（Router 分配给 expert e 的 softmax 概率均值）
- $E$ 为 expert 总数

当所有 expert 被均等利用时 $f_e = P_e = 1/E$，$L_{\text{aux}} = 1$。$L_{\text{aux}}$ 越大表示负载越不均。

从算法pipeline角度拆解：

```
# 在每个训练 step 的 forward pass 中:
gate_probs = softmax(x @ W_router)     # (S, E), Router softmax 输出
topk_probs, topk_idx = topk(gate_probs, T)  # Top-T 选择

# 计算 f_e (实际负载):
f_e = zeros(E)
for t in range(S):
    for k in range(T):
        f_e[topk_idx[t, k]] += 1 / (S * T)

# 计算 P_e (Router 分配比例):
P_e = gate_probs.mean(dim=0)           # (E,)

# 辅助损失:
L_aux = E * sum(f_e * P_e)             # 标量

# 总损失:
L_total = L_LM + alpha * L_aux
# 本文: alpha = 1e-2, 不带 Z loss
```

关键 hyperparameter 权衡：
- alpha 太小 (如 1e-4)：不充分的负载均衡 → 出现 "dead experts"（某些 expert 永远未被路由），导致训练 loss 提早 plateau
- alpha 太大 (如 1e-1)：aux loss 主导 language modeling loss → 模型质量下降
- 本文推荐范围：1e-2 到 1e-3

术语一般如何实现？

在 Megatron-LM / NeMo 等训练框架中，aux loss 在 MoE 层的 forward pass 中计算并加到 total loss 中。训练时与主 loss 同步反向传播。Upcycling 场景下特别重要，因为初始 Router 是随机初始化的，没有 aux loss 会导致少数 expert 迅速接收大部分 token 而其他 expert 完全不被训练。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts

## Softmax-then-TopK vs TopK-then-Softmax Routing

术语是什么？

在 MoE Router 中，将 Router logits 映射为 expert 选择有两种顺序：(1) **Softmax-then-TopK**：先对整个 logit 向量做 softmax 得到概率分布，再从概率分布中选 top-K（标准 MoE 做法，Shazeer et al., 2017）；(2) **TopK-then-Softmax**：先从 logit 中选 top-K，仅对这 K 个 logit 做 softmax（Mixtral 8x7B 的做法，Jiang et al., 2024）。

两者的本质区别在于：Softmax-then-TopK 给所有 E 个 expert 都分配了非零概率（虽然只有 top-K 被激活），保留了"非 top-K 的 logit 有多接近被选中"的信息；TopK-then-Softmax 则完全丢弃了非 top-K logits 的绝对值信息，因为 softmax 仅作用于 top-K 个 logit，对于 topK=1 的特殊情况，softmax of single element = 1（常数），梯度为零。

从算法pipeline角度拆解：

```
# 方法 1: Softmax-then-TopK (本文推荐)
r = x @ W_r               # (E,) Router logits
s = softmax(r)            # (E,) probability distribution
[p1, p2], [e1, e2] = topk(s, 2)  # select top-2
# 输出: p1*Expert(e1) + p2*Expert(e2)
# p1, p2 是原始 softmax 值，通常不归一化到和为1

# 方法 2: TopK-then-Softmax (Mixtral 风格)
r = x @ W_r               # (E,) Router logits
[val1, val2], [e1, e2] = topk(r, 2)  # select top-2 logits
[s1, s2] = softmax([val1, val2])  # 仅对选中的 logit 做 softmax
# 输出: s1*Expert(e1) + s2*Expert(e2)
# s1 + s2 = 1 (保证)
```

本文发现 softmax-then-topK 在 upcycling 场景下一致优于 topK-then-softmax。推测原因是保留所有 expert 的相对信息有助于 Router 梯度更丰富。但 softmax-then-topK 也有缺点：upcycling 初始阶段 MoE 输出与 dense 模型不等价（而 topK-then-softmax 在 topK > 1 时可以使输出 sum to 1），这一缺点被 Weight Scaling 方法弥补。

术语一般如何实现？

在 Megatron-LM 中默认使用 softmax-then-topK。在 NeMo 中可配置。切换方式：修改 Router 模块中 softmax 和 topK 的调用顺序。

涉及论文标题：
- Upcycling Large Language Models into Mixture of Experts

## Weight-Only Quantization for MoE (MoE 专家权重的仅权重量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Weight-Only Quantization 是一种只对模型权重进行量化、保持激活值为浮点精度的推理优化技术。在 MoE 场景下，选择仅量化 expert 权重而非所有参数（也不量化 activations），因为：(1) MoE 的 expert 权重占模型总参数 90% 以上，是内存的主要消耗者；(2) 只量化权重避免了 activation quantization 所需的 calibration，使得量化策略无需 post-training calibration 即可直接应用到不同语言家族；(3) 所有 activations 和 biases 保持 FP16，dequantized weights 也恢复到 FP16，因此矩阵乘法仍使用浮点运算，无需整数 Tensor Core。

论文 "Who Says Elephants Can't Run" 使用 symmetric range-based per-channel quantization：对每个 expert 权重矩阵 $W \in \mathbb{R}^{E \times M \times N}$（E 个专家，M×N 权重），沿输出 channel（N 维度）计算 per-channel scales $S \in \mathbb{R}^{E \times 1 \times N}$。INT8 使用 scale factor = max(|W[:,:,n]|) / 127，INT4 使用 max(|W[:,:,n]|) / 7。量化后权重加常量偏移（INT8: +128, INT4: +8）转为无符号数，简化后续 dequantize 的位操作。

从算法pipeline角度拆解术语：

MoE 模型的 INT4/INT8 推理 pipeline：
```
离线量化阶段（训练完成后一次性执行）：
for expert e in 0..E-1:
    for output_channel n in 0..N-1:
        max_abs = max(|W_fp16[e, :, n]|)
        S[e, 0, n] = max_abs / max_val_int     # 127 (INT8) or 7 (INT4)
        W_quant[e, :, n] = round(W_fp16[e, :, n] / S[e, 0, n])
        W_plus[e, :, n] = W_quant[e, :, n] + offset  # 128 or 8

在线推理阶段（每个 MoE layer 执行时）：
for each MoE layer:
    gate_logits = x @ W_router                  # FP16 matmul (Router)
    expert_idx, expert_scale = top_k(gate_logits, k=1)
    # Token routing: CUB radix sort + permute (FP16 activations)
    tokens_perm = radix_sort_and_permute(x, expert_idx)
    # Fused GEMM + Dequantize per expert
    for expert e with active tokens:
        W_deq = int_to_fp16_fast(W_plus[e] - offset)
        W_deq = W_deq * S[e]                    # FP16 乘 scale
        out_e = tokens_e @ W_deq                # FP16 GEMM
    output = unpermute_and_scale(out, expert_scale)
```
核心 insight：不量化 activation 避免了 calibration，所有中间结果保持 FP16，只有 weight load 和 dequantize 在 GEMM 内部 fused 处理。

术语一般如何实现？如何使用？

通常配合 CUTLASS 或 cuBLAS 实现 fused kernel。LLM.int8() 使用混合精度分解，GPTQ 使用 optimal brain quantization。MoE 场景中 weight-only 量化尤其有效——expert 权重冗余度高（大量独立 expert FFN 参数），量化 bit 损失被稀释。当前论文显示 INT4 实现 8× 模型压缩（5B→~625MB expert weights），INT4 GEMM 最高 1.85× 加速，BLEU 质量几乎无损（10 语言对平均 ΔBLEU = -0.167）。

涉及论文标题：
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

## Deep Encoder Shallow Decoder Architecture (深编码浅解码架构)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Deep Encoder Shallow Decoder 是一种为机器翻译任务设计的 encoder-decoder Transformer 架构变体，将 encoder 层数设为 decoder 层数的约 2 倍。传统 Transformer 的 encoder 和 decoder 通常层数对称，但在自回归推理中 decoder 是性能瓶颈——encoder 只执行一次而 decoder 每生成一个 token 都要执行一次 beam search。通过减少 decoder 层数来降低自回归解码计算开销，同时保持深度 encoder 的编码能力以保证翻译质量。

论文 "Who Says Elephants Can't Run" 使用 24 encoder layers + 12 decoder layers（2:1），embedding dim 1024, FFN hidden dim 4096，每 2 层使用 MoE FFN 层代替 dense FFN。该配置来自 Kim et al. (2021) 和 Kasai et al. (2020)，实验验证为 quality-performance 的 trade-off 最优点。

从算法pipeline角度拆解术语：

Auto-regressive 推理的层执行模式：
```
# === Encoder: 执行 1 次 ===
h_enc = embed(input_tokens)           # B × S_in tokens
for l in 1..24:                       # 24 layers
    h_enc = TUPE_self_attn(h_enc) + h_enc
    if l % 2 == 0:                    # 12 MoE layers in encoder
        h_enc = MoE_FFN(h_enc) + h_enc

# === Decoder: 每 token 执行 1 次（自回归 bottleneck） ===
for t in 1..T_out:
    for l in 1..12:                   # 12 layers (half!)
        h_dec = TUPE_self_attn(h_dec) + h_dec
        h_dec = cross_attn(h_dec, h_enc) + h_dec
        if l % 2 == 0:                # 6 MoE layers in decoder
            h_dec = MoE_FFN(h_dec) + h_dec
    next_token = argmax(lm_head(h_dec[:,-1,:]))
```

为什么有效：在 beam search 中，decoder 执行成本 = B × K × T_out × L_dec × cost_per_layer，encoder 执行成本 = B × S_in × L_enc × cost_per_layer。由于 T_out × K 通常远大于 S_in，decoder 深度影响巨大。L_dec 减半 ≈ decoder 计算减半 ≈ 总延迟约减半。

术语一般如何实现？如何使用？

在 PyTorch/HuggingFace 中通过 `EncoderDecoderModel` 或自定义 `nn.Module` 配置不同的 encoder/decoder 层数参数。Kim et al. (2019) 最早在 CPU 部署中使用此架构实现极快机器翻译。Kasai et al. (2020) 发现 encoder 至少 2× decoder 深度以保证非自回归蒸馏训练质量。

涉及论文标题：
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

## TUPE (Transformer with Untied Positional Encoding / 解耦位置编码Transformer)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

TUPE 是 Ke et al. (2020) 提出的位置编码方案，将 self-attention 中的 content 相关性和 position 相关性解耦为两个独立项。传统 Transformer 将 positional encoding 与 word embedding 相加后一同输入 attention（$A = (X+P)W_Q W_K^T (X+P)^T$），导致 content-content、content-position、position-content、position-position 四种交互混合。TUPE 将 attention 分解为：$A = \text{softmax}(X W_Q W_K^T X^T + P U_Q U_K^T P^T)$，content 和 position 使用各自独立的 Q/K 投影矩阵（$W_Q, W_K$ vs $U_Q, U_K$），去除交叉项噪声。

论文 5B MoE 模型使用 TUPE attention，来自 Kim et al. (2021) 的 DeepSpeed MoE 配置。

从算法pipeline角度拆解术语：

TUPE 在 MoE Transformer 层中的计算：
```
输入: x ∈ R^{S×D}, position_ids ∈ Z^S

# Content attention
Q_c, K_c = x @ W_Q, x @ W_K           # 标准 Q/K 投影
A_c = Q_c @ K_c^T / sqrt(d)            # content-content score

# Position attention（独立参数）
pos = learnable_pos_embed[position_ids]
Q_p, K_p = pos @ U_Q, pos @ U_K       # position 专属 Q/K
A_p = Q_p @ K_p^T / sqrt(d)            # position-position score

# 组合（无交叉项）
A = softmax(A_c + A_p)                 # 仅 content-content + position-position
output = A @ (x @ W_V)                 # value 投影
```

优势：(1) 去除 content-position 交叉项噪声，attention 更专注于语义内容；(2) position 学习独立的 Q/K 参数，更好捕获绝对和相对位置关系；(3) 对不同长度序列更鲁棒；(4) 可分别处理 content 和 position 的 Q/K。

术语一般如何实现？如何使用？

Ke et al. (2020) 开源实现在 GitHub（https://github.com/guolinke/TUPE）。Kim et al. (2021) 将其集成到 DeepSpeed MoE 训练框架。在 PyTorch 中修改标准 `nn.MultiheadAttention`：分别计算 content attention 和 position attention，softmax 前相加。

涉及论文标题：
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production

## Expert-Specialized MoE (专家特化混合专家 / DeepSeek-Style MoE)

术语是什么？

Expert-Specialized MoE 是 Mixture-of-Experts 架构的一个新兴子类，以 DeepSeek-MoE 为代表。其核心设计思想是将传统 MoE 中的 coarse-grained experts 拆分为大量 fine-grained experts（细粒度专家），同时增大 top-k 路由值（每 token 激活更多专家）。具体来说，若标准 MoE 有 E 个 expert、每 expert FFN hidden dim = HFFN、top-k = k，则 Expert-Specialized MoE 引入 fine-grained factor m，将 expert 数量扩展为 E × m，每 expert hidden dim 缩减为 HFFN / m，top-k 增大为 k × m。这保持了总参数量和 per-token 计算量大致不变，但 token 可见的 expert 组合数从 C(E, k) 暴增至 C(E×m, k×m)。

例如 DeepSeek-v3 使用 256 experts/layer、top-k=8，而传统 MoE 可能仅用 8 experts、top-k=2。这种设计让每个 expert 可以专注于更细粒度的语义概念（expert specialization），大幅提升模型的表达能力。

从算法pipeline角度拆解：

Expert-Specialized MoE 的 forward pass 伪代码：

```
# 输入 tokens: [S, H]
# E=256 experts, k=8, m=8, HFFN=2048 (vs 传统 HFFN=16384)

# Step 1: Gating
logits = softmax(Linear(tokens), dim=-1)  # [S, 256]
combine_weights, top_experts = topk(logits, k=8)  # [S, 8] each

# Step 2: 每 token 被路由到 8 个 fine-grained expert
# Expert i 的 FFN: Linear(H -> HFFN) + Act + Linear(HFFN -> H)
# 但 HFFN=2048 远小于传统 MoE 的 16384

# Step 3: 8 个 expert 的输出加权合并
output = sum(combine_weights[j] * expert_j(token) for j in top_experts)
```

与标准 MoE 的关键差异：
- Expert 数量 8→256 (+m×)，每 expert hidden dim 16384→2048 (/m)
- Token 可见的 expert 组合空间从 C(8,2)=28 增至 C(256,8)≈4.89×10^14
- 激活内存瓶颈从中间 FFN 激活（Ainterm）转移到 dispatch/combine 激活（Adispatch, Acombine），因为后者随 k（正比于 m）线性增长，而前者保持不变

术语一般如何实现？

Expert-Specialized MoE 的训练需要专门的系统优化：
1. **Zero-padding 问题加剧**：数百 expert + large top-k 使传统 GShard 式 capacity-based padding pipeline 的内存开销急剧膨胀（dispatch mask 和 intermediate buffers 占 >70% 激活内存）
2. **通信冗余**：Large top-k 使同一 token 被发往多个跨节点 expert，产生大量跨节点重复传输
3. **并行策略需调整**：传统 TP+EP 不减少 Adispatch/Acombine，需要 SSMB 等新技术

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

## PFT (Padding-Free Token Buffers / 无填充Token缓冲区)

术语是什么？

PFT 是 X-MoE 提出的一种稀疏数据结构，用于替代传统 MoE 训练中的 zero-padded expert buffers。传统 GShard 式 pipeline 为每个 expert 分配固定容量 C 的 token buffer [E, C, H]，不足 C 的槽位零填充。PFT 仅存储实际路由到各 expert 的有效 token，通过配套的 ERI-arrays 追踪路由信息。

PFT 结构包含：
- **token_buffer x**：[B, H]，B 为实际路由 token 总数（不含 padding），仅存有效 token
- **ERI-arrays**（Expert Routing Information Arrays）：
  - `token_ids` [B]：每个 token 在原始序列中的位置索引
  - `expert_ids` [B]：每个 token 被路由到的 expert 编号
  - `tokens_per_expert` [E]：每个 expert 接收的 token 数量
  - `combine_weights` [B]：每个 token 的 gating 概率权重

从算法pipeline角度拆解：

PFT 构造和使用流程：

```
# === PFT Construction ===
# Input: top_experts [S, K], combine_weights [S, K], max_token_count
flat_top_experts = flatten(top_experts)  # [S*K]
flat_weights = flatten(combine_weights)  # [S*K]
sorted_idx = argsort(flat_weights)  # 按权重排序以决定drop哪些token

# One-hot + Cumsum 过滤超出容量的token
one_hot = one_hot(sorted_top_experts, num_classes=E)  # [S*K, E]
rank = cumsum(one_hot, axis=0)  # 每expert内的token序号
mask = rank <= max_token_count  # 超出capacity的drop

# 构建ERI-arrays（仅保留有效token）
token_ids = original_ids[mask]  # [B]
expert_ids = flat_experts[mask]  # [B]
combine_weights = flat_weights[mask]  # [B]
tokens_per_expert = histogram(expert_ids, bins=E)  # [E]

# === Padding-free Dispatch ===
# Gather: 按 token_ids 从 gate_out [S,H] gather → dispatch_in [B,H]
# Uneven AlltoAll: 仅传输 B 个有效token（无padding）
dispatch_out = alltoallv(dispatch_in, tokens_per_expert)  # [Bexp, H]

# === Padding-free Combine ===
combine_in = alltoallv(mlp_out, tokens_per_expert)  # [B, H]
# Scatter: 按 token_ids 放回原始位置 + 乘以 combine_weights
output[token_ids[i], :] += combine_in[i, :] * combine_weights[i]
```

优化技巧：PFT construction 中的 cumsum 原为 inner dimension 操作（memory uncoalesced），X-MoE 将 one_hot 转置为 [E, S*K] 在 outer dimension 做 cumsum，加速 10×。

术语一般如何实现？

PFT 需要配套的 kernel 支持：
- **Triton Gather Kernel**：B thread-blocks, 每 block 256 threads，沿 hidden dimension 循环复制，coalesced read
- **Triton Scatter Kernel**：逆向操作 + 加权，coalesced write
- **Sequential GeMM**：按 tokens_per_expert 切片，每 expert 独立 launch GeMM

复杂度：GShard O(ckbsh)+O(ckb²s²) → PFT O(kbsh)

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

## ERI-arrays (Expert Routing Information Arrays / 专家路由信息数组)

术语是什么？

ERI-arrays 是 PFT 数据结构中的元数据组件，包含四个数组，用于在 padding-free MoE pipeline 中追踪每个 token 的路由信息，使得 dispatch、MLP 和 combine 各阶段可以在无 zero-padding 的情况下正确操作。

四个 ERI-array：
1. **token_ids** [B]：token 在原始输入序列中的位置索引，用于 gather/scatter 操作
2. **expert_ids** [B]：token 被路由到的目标 expert 编号
3. **tokens_per_expert** [E]：每个 expert 分配到的有效 token 数量，驱动 uneven alltoall 和 Sequential GeMM 的切片
4. **combine_weights** [B]：gating 输出的概率权重，在 combine 阶段缩放 expert 输出

在 RBD 中还有扩展的 pilot/local replica ERI-arrays 和 s1_mapping_indices。

术语一般如何实现？

ERI-arrays 在 PFT construction 阶段生成（gating 之后、dispatch 之前），随后贯穿整个 MoE layer forward pass：dispatch 用 token_ids + expert_ids 做 gather → alltoall 用 tokens_per_expert 确定传输量 → MLP 用 tokens_per_expert 切片 → combine 用 token_ids + combine_weights 做 scatter。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms

## LoRA (Low-Rank Adaptation / 低秩适配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LoRA（Low-Rank Adaptation）是一种参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）方法，通过在预训练模型的权重矩阵旁添加低秩分解矩阵来实现任务适配，而无需更新原始权重。核心思想：对于预训练权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$，LoRA 将权重更新表示为低秩分解 $\Delta W = B \cdot A$，其中 $B \in \mathbb{R}^{d \times r}$、$A \in \mathbb{R}^{r \times k}$，秩 $r \ll \min(d,k)$。前向传播时：$h = W_0 x + \Delta W x = W_0 x + B A x$。训练时仅更新 $A$ 和 $B$（加 adapter 输出），而 $W_0$ 冻结。$A$ 通常用随机高斯初始化，$B$ 用零初始化，使训练开始时 $\Delta W = 0$，不破坏预训练权重。

从算法pipeline角度拆解术语：

Uni-MoE 中 LoRA 的应用——在不同阶段使用不同配置：

```
# 阶段二（训练模态特定专家）LoRA
r = 64, alpha = 16
仅应用于 LLM 中 MLP 层的 LoRA 微调
# 前向：
x = input_tokens
h_original = Expert_FFN(x)      # 冻结专家参数
h_lora = B @ A @ x               # B in R^{d x 64}, A in R^{64 x k}
output = h_original + (alpha/r) * h_lora

# 阶段三（MoE 联合训练）LoRA
r = 8, alpha = 16
应用于所有专家 + self-attention 层
# 对于每个 token 被 router 分配给 expert e1：
h_e1 = e1(X_E1)                   # 冻结的专家 FFN
h_e1_LoRA = LoRA-e1(X_E1)         # LoRA 适配器
  = W_0 @ X_E1 + (B @ A) @ X_E1   # 式(19)-(20)
h_e1 = h_e1 + h_e1_LoRA            # 式(21)
```

核心：LoRA 使 Uni-MoE 能在不更新全部专家参数（最多 37B 总参数）的情况下高效微调，阶段三仅需更新少量 LoRA 参数 + Router + 投影层，训练成本显著降低。

术语一般如何实现？如何使用？

通过 HuggingFace PEFT 库或手动实现：对目标线性层（nn.Linear）注册 forward hook 或替换为 LoRA 包装类，定义 `self.lora_A = nn.Linear(in_features, r, bias=False)` 和 `self.lora_B = nn.Linear(r, out_features, bias=False)`。常用配置：对 attention 层的 Q/K/V/O 投影和 FFN 的 W1/W2/W3 应用 LoRA，r 取 4~64，alpha 取 8~32。LoRA 权重可与预训练权重合并（merge）进行无额外开销的推理：$W = W_0 + \frac{\alpha}{r}BA$。在 Uni-MoE 的 MoE 场景中，LoRA 不 merge——Router 控制哪些 token 激活哪些专家，LoRA 参数始终在线适配。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts

## Auxiliary Balancing Loss (辅助平衡损失 / Load Balancing Loss)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Auxiliary Balancing Loss（辅助平衡损失）是 MoE 模型训练中的一种正则化项，源于 GShard（Lepikhin et al., ICLR 2021），用于防止 Router（路由器）"坍塌"——即 Router 将所有 tokens 分配给少数几个专家，导致其余专家不被训练、模型退化为 Dense 模型。该损失鼓励所有专家获得大致相等的 token 分配量，从而保证模型容量被充分利用。

GShard 中的典型形式：$L_{\text{aux}} = \alpha \cdot \sum_{i=1}^{E} f_i \cdot P_i$，其中 $f_i = \frac{1}{T}\sum_{t} \mathbf{1}\{\text{token}_t \text{ routed to expert } i\}$ 是 expert i 实际接收的 token 比例，$P_i = \frac{1}{T}\sum_{t} p_{t,i}$ 是 Router 分配给 expert i 的平均概率。$L_{\text{aux}}$ 在 $f_i$ 与 $P_i$ 不均衡时增大，通过梯度引导 Router 均匀分配。

从算法pipeline角度拆解术语：

Uni-MoE 中的平衡损失实验流程（表 8-9）：

```
# MoE 训练阶段的标准 forward + 辅助平衡损失
for each batch:
    x = input_tokens                     # shape: T x d
    # Router 计算
    logits = x @ W_router                # W_router in R^{d x M}
    probs = softmax(logits, dim=-1)      # probs in R^{T x M}
    # Top-K 选择
    topk_probs, topk_idx = top_k(probs, k=2)
    # Expert FFN 计算 (normal forward)
    output = sum_{i in top_k} topk_probs_i * Expert_FFN_i(x)
    # 辅助平衡损失（如果启用）
    f_i = (1/T) * sum_{t} indicator(token_t -> expert_i)
    P_i = (1/T) * sum_{t} probs[t, i]
    L_aux = alpha * sum_i (f_i * P_i)
    # 总损失
    L_total = L_CE + L_aux
```

Uni-MoE 的关键发现：
1. **Mixture MoE（预训练多样化专家）**：不加 aux loss 时 Avg. 49.2%，加了 aux loss 降至 48.5%（表 8 a vs a'）——因为专家已在阶段二各自发展出模态偏好，Router 天然学会将不同模态 tokens 分配给对应专家，aux loss 反而干扰了自然分化。
2. **Pure MoE（相同初始专家）**：不加 aux loss 时 Avg. 47.5%，加了 aux loss 升至 48.4%（表 8 b vs b'）——相同初始专家缺乏差异化，aux loss 强制 Router 探索不同专家组合。
3. **扩展到 8 专家时**：aux loss 的作用增强（表 9），因为路由搜索空间从 C(4,2)=6 增至 C(8,2)=28 种组合，aux loss 帮助优化专家组合选择。

术语一般如何实现？如何使用？

在 HuggingFace Transformers 的 MoE 实现中（如 Mixtral、Switch Transformers），auxiliary loss 通常以 `load_balancing_loss` 或 `router_z_loss` 的形式内置在 MoE 模块的 forward 中。典型做法：$L_{\text{aux}} = \text{num\_experts} \cdot \sum_i (f_i \cdot P_i)$，超参数 $\alpha$ 通常取 0.01~0.1（平衡主任务损失和平衡损失的量级）。Uni-MoE 的实验表明该损失不是万能的：是否使用需根据专家是否有预训练差异化来决定——当专家已通过预训练发展出明确模态偏好时，aux loss 可能适得其反。

涉及论文标题：
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts
