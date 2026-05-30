## GoldFinch__High_Performance_RWKV_Transformer_Hybrid_with_Linear_Pre-Fill_and_Extreme_KV-Cache_Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是GoldFinch——一种混合RNN-Attention架构，将Finch-C2（改进版RWKV-6）线性注意力层与GOLD Transformer层结合，通过TokenCat机制实现极致KV-Cache压缩。核心创新：(1) Finch-C2：移除Finch的gate，将GroupNorm替换为跨所有head的LayerNorm，对key乘以(1−decay)以保持kv-state行归一化，用数据依赖的独立token-shifted第二Value替代Finch的u(bonus)项；(2) GOLD Key Compression：取Finch-C2最终层输出x_t ∈ R^D，通过全局可学习矩阵W^{KD} ∈ R^{D×(D/16)}压缩为c_t = x_t·W^{KD} ∈ R^{D/16}；(3) TokenCat解压：将压缩key c_t与原始token embedding x_t^0拼接后乘W^{KU} ∈ R^{(D+D/16)×D}并RMSNorm，得到各GOLD层共享的proto-keys；(4) GOLD Attention：在proto-keys和原始embedding上通过DDLoRAdapt (loradapt(x)=x+tanh(xC)D)和data-dependent token shift (ddlerp)生成每层的q/k/v，执行标准MHA，无需存储value cache。预填充(fill)仅运行Finch-C2部分（O(1) per token），最后2G-1个token才需跑完整模型。
  实验比较：(a) 1.5B class模型(L24 D2048 ctx2048)：GoldFinch vs Finch vs Llama，在minipile 1.5T tokens上训练；(b) 消融实验(L12 D768 ctx1024)：Finch-C2的各种变体（无k×(1-w)、无第二Value）、GPTAlpha with RoPE、不同GOLD层比例（1/2, 1/3, 1/6）、Finch-C2/GPTAlpha无压缩hybrid vs GoldFinch with/without RoPE；(c) MQAR associative recall：对比GoldFinch与Linear Transformer architectures；(d) Long context：在PG19上测试Finch和GoldFinch在65536 context下的loss，含RoPE插值extrapolation；(e) Checkpoint Upgrade：从预训练Finch 1.6B checkpoint升级为GoldFinch的可行性实验。

- 硬件平台是什么，配置是什么。
  主实验：单节点8×NVIDIA RTX 4090 GPU，3090s用于升级实验。训练配置：4 GPU，per-GPU batch size 8，2步gradient accumulation，10步warmup后cosine decay anneal（3e-5→1e-5）。消融实验：单GPU，per-step batch size 32，2步gradient accumulation，10步warmup后cosine decay（6e-5→2e-5）。Adam optimizer，beta=(0.9, 0.99)，epsilon=1e-8，weight decay=0.001（仅应用于非LoRA/非压缩矩阵参数）。

- 模型是什么。数据集和bench分别是什么。
  模型：GoldFinch (L24 D2048 ctx2048, 1.45B参数)，Finch (L24 D2048 ctx2048, 1.60B)，Llama (L24 D2048 ctx2048, 1.47B)，所有使用RWKV World tokenizer。消融模型：L12 D768 ctx1024。Finch-C2改进：移除gate（约减少参数量以抵消新增的LoRA参数），第二Value复用W^V权重+额外LoRA。GPTAlpha改进：用RWKV channel mixer替代FFN，在attention层添加RWKV style token shift和额外LayerNorm。GOLD = GPTAlpha Over Linear transformer Decoder，移除key/value权重矩阵，改为从压缩cache和原始embedding生成。
  数据集：minipile（1.5T tokens for main, full for ablation）。Long context：PG19 (older books)。Checkpoint upgrade：2.5T token预训练的Finch 1.6B + 7.5B token新数据集。Benchmark：lambada (ppl+acc)、piqa、hellaswag、winogrande、arc_challenge、arc_easy、sciq。MQAR synthetic task。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/recursal/GoldFinch-paper (Apache 2.0)，模型权重：https://huggingface.co/recursal/GoldFinch-paper。基于修改版Linear Attention Arena代码仓库，88.2% Python + 8.9% CUDA + 2.9% C++。

  GoldFinch前向传播算法pipeline（推理时生成一个token，T=当前timestep）：
  ```
  Input: token idx_t, previous hidden state h_{t-1}, compressed key cache {c_1..c_{t-1}},
         original token indices [idx_1..idx_{t-1}]

  // ====== Finch-C2 layers (前2/3层, 如Layer 0-15 for L24) ======
  x = embedding_lookup(idx_t)
  For layer l = 0..F-1:  // F = 前2/3的层数
      x_prev = h_{t-1}[l]

      // Finch-C2 Time Mixing (per-head, 后续concat):
      d_t = lora_d(ddlerp_d(x, x_prev))               // Eq.4, data-dependent decay factor
      w_t = exp(-exp(d_t))                              // Eq.5, per-channel decay ∈ (0,1)
      r_t = ddlerp_r(x, x_prev) @ W^R                  // Eq.6, receptance
      k_t = ddlerp_k(x, x_prev) @ W^K · (1 - w_t)     // Eq.7, key × (1-decay)  [创新]
      v_t = ddlerp_{i,i}(x, x_prev) @ W^V              // Eq.8, first value
      u_t = ddlerp_u(x, x_prev)                        // Eq.9
      u'_t = u_t @ W^V + tanh(u_t @ W^{UD}) @ W^{UU}  // Eq.10, second value [创新]

      // WKV linear attention (recurrent update):
      wkv_t = diag(w_t) @ wkv_{t-1} + k_t^T @ v_t     // Eq.12, O(H²) state update
      o_t = LayerNorm(concat(r_t @ wkv_t + u'_t)) @ W^O  // Eq.13, output

      // Finch Channel Mixing (same for all layers):
      r'_t = lerp_r'(x, x_prev) @ W^{R'}              // Eq.22
      k'_t = lerp_k'(x, x_prev) @ W^{K'}              // Eq.23, ∈ R^{3.5D}
      v'_t = ReLU(k'_t)² @ W^{V'}                     // Eq.24
      c_out = σ(r'_t) ⊙ v'_t                           // Eq.25

      x = o_t + c_out  // residual

  // ====== Key Compression (after last Finch-C2 layer) ======
  c_t = x @ W^{KD}                                     // Eq.14, W^{KD} ∈ R^{D×(D/16)}, global
  store c_t into compressed key cache

  // ====== GOLD layers (后1/3层, 如Layer 16-23 for L24) ======
  For layer l = F..L-1:
      x_prev = h_{t-1}[l]

      // GOLD Key Decompression (TokenCat):
      x_t^0 = embedding_lookup(idx_t)                  // original embedding
      k_t^D = RMSNorm(concat(x_t^0, c_t) @ W^{KU})    // Eq.15, proto-keys, global W^{KU}

      // GOLD Attention Time Mixing:
      q_t = LayerNorm(ddlerp_q(x, x_prev) @ W^Q)      // Eq.17, per-layer
      a_t = lerp(x_t^0, x_{t-1}^0, μ_a)               // Eq.18, embedding token shift

      // Per-token keys from proto-keys (DDLoRAdapt):
      k_t = LayerNorm(loradapt_k(
               lerp(k_t^D, k_{t-1}^D, lora_k(a_t))    // Eq.19, data-dependent shift
           ))

      // Values from embeddings (no value cache needed):
      v_t = LayerNorm(loradapt_v(
               lerp(x_t^0, x_{t-1}^0, lora_v(a_t))    // Eq.20
           ))

      // Multi-Head Attention over full context:
      o_t = LayerNorm(concat(attention(q_t, K_{1:t}, V_{1:t}))) @ W^O  // Eq.21
      // K contains decompressed keys from cache, V from stored embeddings

      // Finch Channel Mixing (same as above)
      // ...

  Output: logits = lm_head(x)
  ```

  关键点：(1) Finch-C2部分每token O(1)计算，仅需维护wkv_t状态矩阵；(2) 压缩key cache仅需D/16 per token存储（vs 传统2·D·n_layer）；(3) 预填充时仅需跑Finch-C2部分（除最后2G-1个token），实现O(1) per token pre-fill；(4) GOLD层通过pooled key cache (k_t^D)加原始embedding (x_t^0)重构每层key，消除了per-layer key cache；(5) Value直接从embedding生成，消除了value cache。

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是H-Net——一种端到端的分层网络架构，通过Dynamic Chunking (DC) 机制以可微分方式学习数据依赖的chunk边界，完全替代传统BPE tokenization。核心创新包含三个组件：(1) Routing Module：基于相邻encoder输出的cosine similarity预测chunk边界概率p_t和离散边界指示b_t；(2) Smoothing Module：使用EMA (z̄_t = P_t·ẑ_t + (1-P_t)·z̄_{t-1}) 将离散chunk操作转为连续可微分计算，保证梯度流动；(3) Ratio Loss (L_ratio)：类似MoE的load balancing loss，控制压缩比N。H-Net采用U-Net风格的层级结构：encoder (E) + main network (M) + decoder (D)，其中M可递归嵌套为另一个H-Net实现多级hierarchy。E/D使用Mamba-2层（因SSM的压缩归纳偏置），M使用Transformer层（与BPE Transformer baseline公平对比）。额外技术包括Norm Balance（post-network RMSNorm）、Separation of Two Streams（仅对residual path加projection）、LR Modulation（按stage的batch size和hidden dim缩放学习率）。
  实验对比多种baseline：(a) BPE Transformer (GPT-3 L/XL scale, GPT-2 tokenizer)；(b) isotropic byte-level模型：LlamaByte (Transformer)、MambaByte (Mamba-2)；(c) hierarchical static chunking: H-Net (pool) — 固定k-width pooling；(d) hierarchical external chunking: SpaceByte (spacelike delimiter)、SpaceByte++ (SpaceByte + Mamba E/D + 架构改进)、H-Net (space) (SpaceByte++ + 信号传播技术)；(e) hierarchical dynamic chunking: H-Net (1-stage, N=6)、H-Net (2-stage, N=(3,3))。所有模型在bytes-per-batch和FLOPs-per-byte上严格匹配。评估指标：bits-per-byte (BPB)、zero-shot downstream accuracy (LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, OpenBookQA)、Chinese XWinograd、DNA perplexity、HellaSwag robustness (5种文本扰动)。

- 硬件平台是什么，配置是什么。
  Nvidia GPU集群。训练精度论文未显式声明（推测BF16/FP16混合精度）。Large scale模型760M-870M参数，XL scale模型1.3B-1.6B参数。训练配置：sequence length 8192 bytes (byte-level models) / 1792 tokens (BPE Transformer)，batch size 256，数据量100B tokens (FineWeb-Edu subset)。使用FlashAttention-2处理变长序列和variable-length batching。论文提到当前实现比isotropic模型慢约2×。

- 模型是什么。数据集和bench分别是什么。
  模型：所有模型FLOP-matched到GPT-3 Large (760M)和XL (1.3B)两个scale。详细架构见表1。(a) Transformer: T24/D1536 (Large), T24/D2048 (XL)，GPT-2 tokenizer；(b) LlamaByte: T16/D1024；(c) MambaByte: M28/D1024；(d) SpaceByte: T8+T16+T8, D768/D1536；(e) SpaceByte++: M4+T28+M4, D1024/D1536 (Large)；M4+T31+M4, D1024/D2048 (XL)；(f) H-Net (pool/space): M4+T28+M4；(g) H-Net (1-stage, 6-DC): M4+T22+M4, D1024/D1536 (Large); M4+T24+M4, D1024/D2048 (XL)；(h) H-Net (2-stage, (3,3)-DC): M4+T1M4+T26+M4T1+M4, D1024/D1024/D1536 (Large); M4+T1M4+T27+M4T1+M4, D1024/D1536/D2048 (XL)。E/D中的Transformer层使用Sliding Window Attention (W=1024)，所有Transformer层使用gated MLP (SwiGLU)。
  数据集：(a) English LM: FineWeb-Edu (100B tokens sub-sampled)，validation BPB on FineWeb-Edu validation set；(b) Chinese: FineWeb-Edu-Chinese-V2.1 (46B tokens)；(c) Code: Github subset from Pile (46B tokens)；(d) DNA: HG38 human genome。Benchmark: LM Evaluation Harness评测LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, OpenBookQA；Chinese: XWinograd-zh；Robustness: HellaSwag with 5 perturbation types (AntSpeak, Drop, RandomCase, Repeat, UpperCase)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/goombalab/hnet，预训练checkpoints：https://huggingface.co/cartesia-ai。

  H-Net (1-stage) 前向传播算法pipeline：
  ```
  Input: raw bytes x ∈ R^{L^0×D^0}  (L^0=8192, D^0=1024)
  Target compression: N^0 = 6

  // Stage 0: Encoder (Mamba-2, 4 layers)
  x̂^0 = E^0(x)  // x̂^0 ∈ R^{L^0×D^0}

  // Chunking Layer (Dynamic Chunking)
  // -- Step 1: Routing Module --
  For t = 1..L^0:
    q_t = W_q · x̂^0_t    // query projection
    k_t = W_k · x̂^0_t    // key projection
    p_t = 0.5 * (1 - cos_sim(q_t, k_{t-1}))  // boundary probability ∈ [0,1]
    b_t = 1 if p_t >= 0.5 else 0            // discrete boundary indicator
    c_t = p_t^b_t * (1-p_t)^{1-b_t}          // confidence score

  // -- Step 2: Downsampler --
  x^1 = select(x̂^0_t where b_t == 1)  // compressed sequence ∈ R^{L^1×D^0}
  p^1 = select(p_t where b_t == 1)     // compressed probabilities

  // Dimension expansion: D^0 → D^1 (append trainable pad vector)
  x^1 = append_pad(x^1, D^1 - D^0)  // x^1 ∈ R^{L^1×D^1}

  // Stage 1: Main Network (Transformer, 22-24 layers)
  ẑ^1 = M(x^1)  // ẑ^1 ∈ R^{L^1×D^1}

  // Dechunking Layer
  // -- Step 3: Smoothing Module (EMA) --
  For t = 1..L^1:
    z̄_t = P^1_t · ẑ^1_t + (1 - P^1_t) · z̄_{t-1}  // EMA interpolation
    // High confidence (P≈1.0): z̄_t ≈ ẑ^1_t (discrete boundary preserved)
    // Low confidence (P≈0.5): z̄_t blended with previous chunk

  // Dimension reduction: D^1 → D^0 (take first D^0 dims)
  z̄ = z̄[:, :, :D^0]

  // -- Step 4: Upsampler --
  For t = 1..L^0:
    // Find which compressed chunk this position belongs to
    chunk_idx[t] = sum_{k=1}^t b_k
    z̃_t = z̄[chunk_idx[t]]  // causal expansion (repeat until next boundary)
    // Straight-Through Estimator for gradient flow
    STE_c = c_t + stop_gradient(1 - c_t)  // forward: 1.0, backward: c_t
    upsampled_t = STE_c · z̃_t

  // Residual connection with projection
  z^0 = upsampled + Linear(x̂^0)  // z^0 ∈ R^{L^0×D^0}

  // Stage 0: Decoder (Mamba-2, 4 layers)
  output = D^0(z^0)  // output ∈ R^{L^0×D^0}

  // Logit prediction
  logits = Linear_head(output)  // logits ∈ R^{L^0×vocab_size}

  // Ratio Loss (per stage)
  F = mean(b_t)          // fraction selected
  G = mean(p_t)          // average boundary probability
  L_ratio = N/(N-1) * ((N-1)*F*G + (1-F)*(1-G))
  L_total = L_CE + 0.03 * L_ratio
  ```

  H-Net (2-stage) 递归执行两次DC过程：
  ```
  // Stage 0: 与1-stage相同，但M本身也是H-Net
  x̂^0 = E^0(x)  →  DC(N^0=3)  →  x^1, p^1

  // Stage 1: 第二个chunking阶段
  x̂^1 = E^1(x^1)  →  DC(N^1=3)  →  x^2, p^2

  // Stage 2: Main Network (Transformer)
  ẑ^2 = M(x^2)  →  Dechunk(ẑ^2, p^2)  →  z^1
  →  D^1(z^1)  →  Dechunk(z^1, p^1)  →  z^0
  →  D^0(z^0)  →  output
  ```

  推理时H-Net类似speculative decoding：encoder每token步进，routing module决定是否需要main network处理，若不需要则跳过main network——实现了per-token动态计算分配。

  Mamba-2层在E/D中的作用：
  - 每个Mamba-2层: XZ projection (2×expand) → short convolution (window=4) → SiLU → selective SSM scan → SiLU gating → output projection
  - 参数量: ≈6D² per layer（无MLP），Transformer层≈12D²（含MLP）
  - SSM的压缩归纳偏置使其天然适合将多个输入token压缩为固定状态，与DC机制协调
