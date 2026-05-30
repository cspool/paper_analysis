## Gated_Delta_Networks__Improving_Mamba2_with_Delta_Rule

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Gated DeltaNet——一种将Mamba2的gated更新规则与DeltaNet的delta规则统一为"gated delta rule"的新型线性RNN架构。核心创新：(1) gated delta rule: S_t = S_{t-1}(α_t(I - β_t k_t k_t^T)) + β_t v_t k_t^T，其中α_t∈(0,1)控制状态衰减（gating），β_t∈(0,1)控制写入强度（delta rule）；(2) 基于WY表示扩展的硬件高效chunkwise并行训练算法；(3) 混合架构GatedDeltaNet-H1（Gated DeltaNet + Sliding Window Attention）和GatedDeltaNet-H2（Mamba2 + Gated DeltaNet + SWA）。

  实验对比：(a) 语言建模和常识推理（400M/1.3B参数，100B tokens FineWeb-Edu训练）：对比RetNet、HGRN2、Mamba、Mamba2、DeltaNet、Transformer++、Samba，Gated DeltaNet在所有纯循环模型中最佳（1.3B avg 55.32 vs Mamba2 54.89），混合模型进一步改进（GatedDeltaNet-H1 avg 56.40）；(b) S-NIAH合成benchmark（1K-8K序列长度）：验证gating与delta rule的互补性——DeltaNet在S-NIAH-1上近完美但S-NIAH-2/3较长序列下性能下降（缺乏遗忘机制），Mamba2在S-NIAH-2/3上较好但S-NIAH-1长序列崩溃（过度遗忘），Gated DeltaNet在两者间取得最佳平衡；(c) 真实世界in-context retrieval（SWDE, SQuAD, FDA, TriviaQA, Drop, NQ, 2K输入截断）：Gated DeltaNet 30.6 avg vs Mamba2 29.8 vs DeltaNet 26.2；(d) 长度外推（6个长上下文benchmark，最大20K tokens）：Gated DeltaNet在RNN模型中困惑度最低；(e) LongBench（14个任务）：Gated DeltaNet 16.6 avg vs Mamba2 13.5，在single-doc QA、few-shot ICL、Code任务上尤其突出（TREC 30.0 vs Mamba2 13.0）；(f) 训练吞吐量对比（单H100 GPU，1.3B模型）：Gated DeltaNet吞吐量与DeltaNet几乎持平，仅比Mamba2慢2-3K tokens/sec；(g) 消融实验（400M参数，15B tokens）：验证short conv（+1.8ppl）、output gate（+1.8ppl）、L2-norm+SiLU组合、head dim=128为最优配置。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 GPU（训练吞吐量benchmark），NVIDIA GPU集群（主训练）。所有模型1.3B参数，训练100B tokens（FineWeb-Edu），训练长度4K tokens，batch size 0.5M tokens。优化器AdamW（peak LR=4e-4, weight decay=0.1, gradient clipping=1.0），cosine annealing + 1B token warmup。400M消融模型训练15B tokens。训练精度论文未明确说明（推测BF16/FP16混合精度）。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) Gated DeltaNet: 遵循Llama宏观架构，将self-attention替换为gated delta rule token mixing。Block设计：q/k通过线性投影→short convolution→SiLU→L2 norm；v通过线性投影→short convolution→SiLU；α/β通过线性投影+sigmoid；输出通过RMSNorm+SiLU gate后输出投影。1.3B参数，head dim=128。(2) GatedDeltaNet-H1: Gated DeltaNet + Sliding Window Attention (window=2K)交替。(3) GatedDeltaNet-H2: Mamba2 + Gated DeltaNet + SWA三层交替（Mamba2→GatedDeltaNet→SWA为最优顺序）。Baselines: RetNet、HGRN2、Mamba、Mamba2、DeltaNet、Samba（Mamba+SWA hybrid）、Transformer++（Llama-style with RoPE/SwiGLU）。
  数据集：FineWeb-Edu（100B tokens训练子集）。Benchmark：(a) 语言建模: WikiText ppl, LAMBADA ppl/acc；(b) 常识推理: PIQA, HellaSwag, WinoGrande, ARC-easy, ARC-challenge, SIQA, BoolQ（使用lm-evaluation-harness）；(c) 合成in-context retrieval: S-NIAH-1 (passkey), S-NIAH-2 (number in haystack), S-NIAH-3 (UUID in haystack)，序列长度1K-8K（来自RULER benchmark）；(d) 真实in-context retrieval: SWDE, FDA, SQuAD, TriviaQA, Drop, NQ（Cloze Completion Formatting prompts）；(e) 长上下文: LongBench 14任务 + 6个长上下文长度外推benchmark（最大20K tokens）。使用Llama2 tokenizer (vocab=32K)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/NVlabs/GatedDeltaNet

  Gated DeltaNet chunkwise训练算法pipeline（单层，单head，chunk size C）：
  ```
  Input: Q, K, V ∈ R^{L×d}  (L: sequence length, d: head dim)
         C: chunk size
         α_t = sigmoid(proj_α(x_t) + bias_α)  // 数据依赖gate, ∈ (0,1)
         β_t = sigmoid(proj_β(x_t))           // 数据依赖writing strength, ∈ (0,1)

  // 1. 预计算全局累积decay product
  γ_t = ∏_{j=1}^t α_j

  // 2. 分块处理
  For each chunk t = 0, 1, ..., L/C-1:
    // Chunk-local累积decay（从chunk开头重新计算）
    γ_{[t]}^r = ∏_{j=1}^r α_{[t]}^j  // r=1..C

    // Chunk decay mask Γ_{[t]} ∈ R^{C×C}
    (Γ_{[t]})_{ij} = γ_{[t]}^i / γ_{[t]}^j  if i ≥ j else 0

    // 3. 计算T矩阵（WY表示）
    // T_{[t]} = [I + strictLower(diag(β_{[t]}) (Γ_{[t]} ⊙ K_{[t]} K_{[t]}^T))]^{-1} diag(β_{[t]})
    M_{[t]} = diag(β_{[t]}) · (Γ_{[t]} ⊙ K_{[t]} K_{[t]}^T)  // ∈ R^{C×C}
    T_{[t]} = solve_triangular(I + tril(M_{[t]}, diagonal=-1), diag(β_{[t]}))

    // 4. W矩阵和Ũ矩阵
    W_{[t]} = T_{[t]} K_{[t]}       // ∈ R^{C×d}
    Ũ_{[t]} = T_{[t]} V_{[t]}       // ∈ R^{C×d}

    // 5. 应用decay
    ⟵W_{[t]}^r = γ_{[t]}^r · W_{[t]}^r     // decay to chunk start
    ⟶K_{[t]}^r = γ_{[t]}^C/γ_{[t]}^r · K_{[t]}^r  // decay to chunk end
    ⟶S_{[t]} = γ_{[t]}^C · S_{[t]}          // decay state over chunk
    ⟵Q_{[t]}^r = γ_{[t]}^r · Q_{[t]}^r     // decay to chunk start

    // 6. 状态更新
    S_{[t+1]} = ⟶S_{[t]} + (Ũ_{[t]} - ⟵W_{[t]} S_{[t]}^T)^T · ⟶K_{[t]}  // ∈ R^{d×d}

    // 7. Chunk输出
    O_{[t]} = ⟵Q_{[t]} S_{[t]}^T + (Q_{[t]} K_{[t]}^T ⊙ M) · (Ũ_{[t]} - ⟵W_{[t]} S_{[t]}^T)
    // M ∈ R^{C×C}: causal mask (下三角=1, 上三角=0)
  ```

  推理时recurrent形式（生成一个token，O(d²) per token）：
  ```
  Input: x_t ∈ R^D, S_{t-1} ∈ R^{d×d}

  // Token mixer block
  q_t = L2Norm(SiLU(ShortConv(Linear_q(x_t))))  // ∈ R^d
  k_t = L2Norm(SiLU(ShortConv(Linear_k(x_t))))  // ∈ R^d
  v_t = SiLU(ShortConv(Linear_v(x_t)))           // ∈ R^d
  α_t = sigmoid(Linear_α(x_t) + bias_α)          // ∈ (0,1)
  β_t = sigmoid(Linear_β(x_t))                   // ∈ (0,1)

  // Gated Delta Rule update
  S_t = α_t · S_{t-1} · (I - β_t k_t k_t^T) + β_t · v_t k_t^T
  //    ^^^^^^^ 全局衰减      ^^^^^^^^^^^^^^^^^ delta擦除旧值  ^^^^^^^^^ 写入新值

  // Output
  o_t = S_t q_t                       // ∈ R^d
  g_t = SiLU(Linear_g(x_t))
  output_t = Linear_o(RMSNorm(o_t) ⊙ g_t)  // ∈ R^D
  ```

  Gated delta rule的双重优势（S-NIAH case study验证）：
  - α_t→0: 快速清除整个记忆（context switch场景）
  - α_t→1: 纯delta rule，精确更新特定key-value对（memorization场景）
  - 在线学习视角：α_t为adaptive weight decay，β_t为SGD learning rate

  与baseline的关键公式对比：
  - Mamba2: S_t = α_t S_{t-1} + v_t k_t^T  (仅有全局衰减, 无精确更新)
  - DeltaNet: S_t = S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T  (仅有精确更新, 无全局衰减)
  - Gated DeltaNet: S_t = S_{t-1}(α_t(I - β_t k_t k_t^T)) + β_t v_t k_t^T  (两者兼有)
