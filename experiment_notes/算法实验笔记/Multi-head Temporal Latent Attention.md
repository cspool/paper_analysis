## Multi-head Temporal Latent Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  MTLA（Multi-head Temporal Latent Attention）在 MLA 的 latent 维度压缩基础上，进一步在 temporal 维度压缩 KV cache。核心实现：(1) 用共享低秩 latent 向量 C ∈ R^{T×r}（r=256 ≪ d=512）压缩跨 head 的 key/value 信息（继承 MLA）；(2) 用 hyper-network 动态生成 temporal merging weights w_i = Sigmoid(Linear(c_i) · Linear(pe_j))，将每 s 个 temporally adjacent latent vector 加权合并为一个 compressed vector ĉ_j，使 KV cache 序列长度从 T 降至 t = ⌈T/s⌉；(3) 用 stride-aware causal mask 解决 training 时 compressed KV cache 长度与 sequence length 不匹配的问题，确保 parallel training 与 incremental inference 的 attention pattern 一致；(4) decoupled RoPE 的 key 同样沿 temporal 维压缩，仅保留 position-specific RoPE key 用于 attention score 增强。

  实验比较：(a) MTLA vs MHA vs MLA on ST（MuST-C En-De, BLEU）、text summarisation（XSum, ROUGE）、ASR（AMI, WER）、SLU（SLURP, Accuracy）；(b) MTLA vs MQA/GQA/MLA w/ SnapKV/Mamba-2 on ST；(c) MTLA 不同 temporal compression ratio s=2/3/4 的 tradeoff；(d) MTLA + FlashAttention-2 vs MHA + FlashAttention-2；(e) MTLA on LRA benchmark vs 11 种 efficient attention；(f) MTLA vs MLA on MT（WMT14 En-De）。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA RTX 6000 Ada GPU（48GB 显存）。训练：ST 每 epoch ~13 分钟（78M params），text summarisation 每 epoch ~20 分钟（79M params），ASR 每 epoch ~50 分钟（67M params），SLU 每 epoch ~15 分钟（103M params）。推理：所有模型使用相同的 batch size 和 beam size，报告 inference time 和 average GPU memory usage。

- 模型是什么。数据集和bench分别是什么。
  模型：Decoder-only Transformer（encoder 输出 prepend 到 self-attention 输入，移除 cross-attention），9 层 decoder + 12 层 encoder，d=512，n_h=8，d_h=64，FFN d_ff=2048。MTLA/MLA 使用 r=256，d_h^R=32。默认 s=2。MHA/MLA/MTLA 参数量：ST 78M，summarisation 79M，ASR 67M，SLU 103M。MQA/GQA 约 74M。

  数据集：MuST-C v1.0 En-De（400h TED Talks speech translation），XSum（226K BBC news articles summarisation），AMI Meeting Corpus（100h meetings ASR），SLURP（120K utterances, 18 domains SLU）。LRA benchmark 和 WMT14 En-De 用于补充评估。

  Benchmarks：BLEU（ST）、ROUGE-1/2/L（summarisation）、WER（ASR）、Accuracy（SLU intent classification）。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/D-Keqi/mtla。基于 Fairseq 实现。

  **MTLA 单层前向—推理 pipeline（张量级，s=2, r=256, d=512, n_h=8）**：

  ```
  Step 1 - 输入: x_i ∈ R^{1×d}（单 token, incremental decoding）

  Step 2 - 多头 Query（标准 MHA）:
    q_i = x_i @ W_Q           # W_Q ∈ R^{d×(n_h·d_h)}, q_i ∈ R^{1×512}

  Step 3 - 低秩 Latent Vector（继承 MLA）:
    c_i = LayerNorm(x_i @ W_r)  # W_r ∈ R^{d×r}, c_i ∈ R^{1×256}

  Step 4 - Hyper-network 动态生成 merge weight:
    j = ceil(i / s)           # 当前所属的 compressed slot
    pe_j: positional embedding at step j
    w_i = Sigmoid(Linear(c_i) · Linear(pe_j))  # 两个 Linear 各映射 256→64, · 逐元素乘
    w_i ∈ R^{64}（实际实现中可能标量）

  Step 5 - 更新压缩 temporal-latent KV cache:
    if i % s == 1:  # 新 slot
      Ĉ = Concat(Ĉ, w_i ⊙ c_i)    # 添加新 compressed vector
    else:            # 合并到当前 slot
      Ĉ_j = Ĉ_j + w_i ⊙ c_i      # 动态融合相邻 token 信息

  Step 6 - Absorbed Attention（不显式计算 K, V）:
    # W_Q·W_K^T 预计算为吸收矩阵（训练时）
    # W_V·W_O 预计算为吸收矩阵
    scores = (x_i @ W_Q_absorbed) @ Ĉ^T / sqrt(d_h)
    # W_Q_absorbed = W_Q @ W_K^T, 形状 ∈ R^{d×(r)} → 但实际等价于 query-key 直接算
    attn_out = softmax(scores) @ (Ĉ @ W_V_absorbed)
    # Ĉ @ W_V_absorbed = Ĉ @ (W_V @ W_O), 直接输出无需中间 K, V
  ```

  **MTLA 训练—stride-aware causal mask（并行，s=2）**：

  ```
  Step 1 - Full sequence 输入: X ∈ R^{T×d}

  Step 2 - 低秩 latent: C = LayerNorm(X @ W_r) ∈ R^{T×256}

  Step 3 - Hyper-network 批量生成权重矩阵:
    PE = replicated positional embeddings: (pe_1,...,pe_1,...,pe_t,...,pe_t)
    每个 pe_j 重复 s 次，总长 T
    W = Sigmoid(Linear(PE) × Linear(C)) ∈ R^{T×T}
    chunk_mask(W)  # 仅保留对角线附近 chunk，去除跨 chunk 连接
    Ĉ' = W @ C     # 生成 extended compressed sequence ∈ R^{T×256}
    # Ĉ' 长度仍为 T（仅训练时），每 s 个 position 有相同的 compressed vector

  Step 4 - Absorbed attention with stride-aware causal mask:
    scores = (X @ W_Q_absorbed) @ Ĉ'^T / sqrt(d_h)  # ∈ R^{T×T}
    mask[n, m] = 0 if n==m or (m < n and m % s == 0) else -∞
    attn = softmax(scores + mask) @ (Ĉ' @ W_V_absorbed) ∈ R^{T×d}
  ```

  **KV cache 大小对比**（per token, n_h=8, d_h=64, s=2）：
  - MHA: 2 · d_h · n_h · l = 2 · 64 · 8 · l = 1024l elements
  - MQA: 2 · d_h · 1 · l = 128l elements
  - MLA: 9d_h · l / 2 = 288l elements
  - MTLA (s=2): 9d_h · l / (2s) = 2.25d_h · l = 144l elements（接近 MQA 水平）
  - MTLA (s=4): 9d_h · l / 8 = 72l elements

- 属于算法pipeline的实现是什么？实验比较什么？
  MoM（Mixture-of-Memories）是一种新的线性序列建模架构，核心创新是用多个独立的 memory state 替代传统线性模型中单一固定大小的 memory state，并通过 router 网络将每个 token 路由到 top-k 个 memory state 进行选择性更新，最后将多个 memory 加权混合后得到输出。该方法受生物神经元 theta-gamma 振荡机制和 MoE 思想的启发。MoM 支持多种 memory update 机制（Linear Attn、RetNet、GLA、DeltaNet、G-DeltaNet、TTT、Titans、Mamba2、HGRN2、RWKV6/7，详见 Table 1）。论文选择 Gated DeltaNet 作为主要的 memory update 方法。默认配置：4 个 memory state + 1 个 shared memory，top-2 激活（activation ratio=0.5）。Router 使用线性层计算 importance scores，经 softmax+TopK 后归一化得到 routing weights。Shared memory 始终被激活，用于捕获全局长期信息。硬件高效实现：将 tokens 按 routing 结果重排序分组 → concat 为 varlen 输入 → Triton kernel 计算 → 输出拆分回原始顺序 → 加权求和。训练使用 auxiliary loss（类似 Switch Transformer 的 load balancing loss）。

  实验比较：MoM vs Transformer++、RetNet、HGRN2、GLA、GSA、Gated DeltaNet。实验维度包括：(1) Recall-intensive tasks（FDA, SWDE, SQuAD, NQ, TriviaQA, Drop）；(2) Commonsense reasoning（WikiText ppl, LAMBADA ppl/acc, ARC-e, ARC-c, HellaSwag, PIQA, WinoGrande）；(3) LongBench long-context benchmark（单文档QA、多文档QA、摘要、少样本学习、合成任务、代码补全）；(4) 推理效率对比（inference time, GPU memory vs sequence length）；(5) Length extrapolation（2K → 32K perplexity）；(6) Memory scaling（memory 数量 1-8，activation ratio 0.25/0.5）；(7) Ablation on auxiliary loss scale 和 shared memory；(8) MoM-Transformer hybrid（每 7 层 MoM + 1 层 Transformer）；(9) Fairness comparison（等 activated params 和等 memory capacity）。两个模型规模：380M（24 layers, d=1024, 15B tokens）和 1.3B（24 layers, d=2048, 100B tokens）。

- 硬件平台是什么，配置是什么。
  32 张 NVIDIA A800 GPU。380M 模型训练约 10 小时，1.3B 模型训练约 6 天。使用 AdamW optimizer（lr=3e-4, cosine schedule, weight decay=0.01, gradient clipping=1.0）。380M 模型 batch size=0.5M tokens, warmup=0.25M tokens；1.3B 模型 batch size=2M tokens, warmup=1B tokens。硬件高效实现基于 Triton kernel（varlen operations）。

- 模型是什么。数据集和bench分别是什么。
  模型：MoM 380M（24 layers, hidden size=1024, hidden ratio=3, 4 memory states + shared memory, top-2 activation）和 MoM 1.3B（24 layers, hidden size=2048, 同样 memory 配置）。Baseline 模型包括：Transformer++（RoPE + GLU）、RetNet、HGRN2、GLA、GSA、Gated DeltaNet。

  数据集：训练使用 SlimPajama 数据集（627B tokens 的清洗版本），采样 100B tokens。Tokenizer: Mistral tokenizer。Length extrapolation 使用 Fineweb 数据集。

  Benchmarks：(1) Recall-intensive: FDA, SWDE, SQuAD, NQ, TriviaQA, Drop（max length 2K tokens）；(2) Commonsense reasoning: WikiText ppl, LAMBADA ppl/acc, ARC-easy, ARC-challenge, HellaSwag, PIQA, WinoGrande（使用 lm-evaluation-harness 评估）；(3) LongBench: 中英双语长文本理解 benchmark。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  开源：https://github.com/OpenSparseLLMs/MoM 和 https://github.com/OpenSparseLLMs/Linear-MoE。

  **MoM 单层前向 pipeline（张量级，top-k=2, M=4 memories + shared）**：

  ```
  输入: X ∈ R^{T×d}  (T=sequence length, d=hidden dim)
  参数: Router W_g ∈ R^{d×M}, memory-specific W_k^m, W_v^m ∈ R^{d×d}, W_q ∈ R^{d×d}
       M: total memory count (默认4), k: activated count (默认2)

  Step 1 - Router:
    scores = softmax(X @ W_g)        # scores ∈ R^{T×M}
    topk_scores, topk_indices = TopK(scores, k)  # 选择 top-k 个 memory
    g_t = topk_scores / sum(topk_scores)  # 归一化 importance weights ∈ R^{T×k}

  Step 2 - Hardware-Efficient Reordering (varlen):
    按 routing 结果将 tokens 分组到各 memory bucket:
    I_{b,m} = {t : token t routed to memory m for batch b}
    将同 bucket tokens concat 为 varlen sequence:
    X̃ = concat([X[I_{1,1}], ..., X[I_{B,M}]])

  Step 3 - Per-Memory QKV Projection & Update:
    对于每个 memory m:
      q̃_u = X̃_u @ W_q             # shared query projection
      k̃_u^m = X̃_u @ W_k^m         # memory-specific key
      ṽ_u^m = X̃_u @ W_v^m         # memory-specific value
      用 Gated DeltaNet update:
        M_t^m = a_t (I - (k_t^m)^T k_t^m) M_{t-1}^m + b_t (k_t^m)^T v_t^m
      或用其他 update rule（Table 1）

  Step 4 - Output Computation:
    对每个 token 的每个激活 memory:
      o_t^m = q_t @ M_t^m          # per-memory output
    加权混合:
      ỹ_t = Σ_{m in activated} g_t^{(m)} · o_t^m
    最终输出:
      o_t = activation(norm(ỹ_t)) @ W_o
  ```

  关键设计点：(1) Memory states 是 R^{d×d} 矩阵，类似 linear attention 的 KV 外积累积；(2) 非激活 memory 保持上一时刻状态不变，避免当前输入干扰；(3) Shared memory 始终被所有 token 激活，通过全局信息提升长程依赖；(4) 稀疏激活仅作用于 K/V projection（占参数量小），MLP 部分保持不变；(5) 计算复杂度：training O(n)（每 memory 线性），inference O(1)（constant memory state per step）。
