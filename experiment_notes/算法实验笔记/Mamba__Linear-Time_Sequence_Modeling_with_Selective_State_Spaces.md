## Mamba__Linear-Time_Sequence_Modeling_with_Selective_State_Spaces

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Mamba——一种基于选择性状态空间模型（Selective SSM / S6）的新型序列建模架构，无需attention甚至MLP block。核心创新：(1) Selection Mechanism：将SSM参数（Δ, B, C）变为输入x的函数（s_B(x)=Linear_N(x), s_C(x)=Linear_N(x), s_Δ(x)=Broadcast_D(Linear_1(x)), τ_Δ=softplus），使模型能根据当前token选择性传播或遗忘信息；(2) Mamba Architecture：简化H3和MLP block为单一同质Mamba block（expand factor E=2, SiLU/Swish激活, 可选normalization layer），2×Mamba blocks参数匹配1×Transformer layer(12D²)；(3) Real-valued SSM：默认使用S4D-Real参数化和实数值state（A_n = -(n+1)），A矩阵为对角结构。Mamba block前向流程：x → Linear投影expand到2×D → 分两路：一路CausalConv1d(kernel=4)→SiLU→SSM selective scan，另一路SiLU gate → element-wise乘 → Linear输出投影 → residual。
  实验比较：(a) Language Model Scaling Laws（Figure 4, ≈125M-1.3B参数, Pile数据集）：vs Transformer(GPT3)、Transformer++(RoPE+SwiGLU+RMSNorm+improved recipe)、Hyena、H3++、RWKV、RetNet——Mamba是首个匹配强Transformer++的attention-free模型；(b) Zero-shot下游评估（Table 3, trained 300B tokens）：Mamba-130M/370M/790M/1.4B/2.8B vs Pythia/RWKV/OPT/Hybrid H3/GPT-Neo在LAMBADA/HellaSwag/PIQA/Arc-E/Arc-C/WinoGrande——Mamba each size best-in-class, Mamba-2.8B avg 63.3 匹配 Pythia-6.9B 的 61.7；(c) DNA Modeling（Figure 5）：HG38基因组pretraining scaling laws(≈250K-40.7M params) + Great Apes物种分类(最高1M序列长度, Table 13)；(d) Audio Modeling（Figure 7, Table 4-5）：YouTubeMix钢琴音乐pretraining + SC09语音生成(vs SaShiMi/WaveNet/DiffWave等)；(e) Synthetic Tasks：Selective Copying(Table 1, seqlen=4096)和Induction Heads(Table 2, Mamba可外推到1M tokens，4000×训练长度)；(f) Ablations（Table 6-10, ≈350M LM）：H3 vs Mamba block、S4(LTI) vs S6(selective)、Δ/B/C selective参数组合、A矩阵初始化(S4D-Real vs S4D-Lin vs Random)、Δ投影维度、SSM state dimension N的scaling。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB PCIe GPU。训练精度BF16/FP16混合精度（部分操作float32）。LM scaling laws使用AdamW(β1=0.9, β2=0.95, weight decay=0.1, gradient clip=1.0, no dropout, linear warmup+cosine decay)。Scaling law sizes(Table 12): 125M(L12/D768, LR=6e-4, batch=0.5M tokens), 350M(L24/D1024, LR=3e-4), 760M(L24/D1536, LR=2.5e-4), 1.3B(L24/D2048, LR=2e-4)。Transformer++使用improved recipe(peak LR=5×GPT3, no bias, RMSNorm, β=(0.9,0.95), cosine decay to 1e-5)。Efficiency benchmark: 单A100 80GB PCIe, BF16, batch=1, D=1024, N=16, prompt=2048, gen=128。Memory benchmark: 125M model, seqlen=2048, 1 A100 80GB。

- 模型是什么。数据集和bench分别是什么。
  模型：Mamba series (125M/350M/760M/1.3B per GPT-3 specs, 以及130M/370M/790M/1.4B/2.8B for downstream)。Mamba block: input projection(2ED²)→conv1d(kernel=4)→SiLU→SSM(Δ proj dim=R=D/16, state N=16)→SiLU gate→output proj(ED²)。Downstream用GPT-NeoX tokenizer(50277 vocab, ctx=2048), scaling用GPT-2 tokenizer。Baselines: Transformer(GPT3)、Transformer++(RoPE+SwiGLU+RMSNorm)、Hyena(H3+global conv)、H3++(H3+improved recipe)、RWKV、RetNet。DNA: Mamba(4-12 blocks, D=64-512, ≈250K-40.7M params per doubled blocks)。Audio: Mamba-UNet(15 blocks/stage, pool=16 p=16, D=64, 3.5M; 或15 layers/stage D=96 pool=4, 24.3M)。
  数据集：Pile(LM pretraining); HG38 human genome(≈4.5B tokens DNA pretraining); YouTubeMix(4h piano music, 16000Hz, mu-law 8bit); SC09(1s speech "zero"-"nine", 16000Hz)。Benchmark: LM Evaluation Harness(LAMBADA/HellaSwag/PIQA/ARC-Easy/ARC-Challenge/WinoGrande); DNA Species Classification(5 great apes, seqlen 2^10-2^20); SC09 automated metrics(NLL/FID/IS/mIS/AM); Synthetic Selective Copying(seqlen=4096, vocab=16, 16 data tokens)和Induction Heads(seqlen=256 train, 2^6-2^20 test, vocab=16)。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  完全开源(Apache 2.0)。代码: https://github.com/state-spaces/mamba，预训练checkpoints在HuggingFace。

  S4(LTI) vs S6(Selective) 核心算法对比：
  ```
  # Algorithm 1: SSM (S4) — LTI, time-invariant
  Input: x: (B,L,D)
  A: (D,N) ← Parameter          # constant
  B: (D,N) ← Parameter          # constant
  C: (D,N) ← Parameter          # constant
  Δ: (D) ← τ_Δ(Parameter)     # constant
  A_bar, B_bar: (D,N) ← discretize(Δ, A, B)
  y ← SSM(A_bar, B_bar, C)(x)  # recurrence OR convolution
  # 卷积模式训练: y = x * K, K = (CB, CAB, CA²B, ...)

  # Algorithm 2: SSM + Selection (S6) — time-varying
  Input: x: (B,L,D)
  A: (D,N) ← Parameter          # still constant
  B: (B,L,N) ← s_B(x)          # input-dependent!  s_B = Linear_N
  C: (B,L,N) ← s_C(x)          # input-dependent!  s_C = Linear_N
  Δ: (B,L,D) ← τ_Δ(Parameter + s_Δ(x))  # input-dependent! s_Δ = Broadcast_D(Linear_1)
  A_bar, B_bar: (B,L,D,N) ← discretize(Δ, A, B)
  y ← SSM(A_bar, B_bar, C)(x)  # recurrence (scan) ONLY, 失去与卷积的等价性
  ```

  推理时循环模式——O(1) per-token, 无KV cache:
  ```
  # Mamba autoregressive inference (per token generation)
  Input: x_t ∈ R^D, h_{t-1} ∈ R^{ED×N} (fixed-size SSM state)
  # Mamba block:
  x_proj, z = split(Linear_in(x_t))           # ∈ R^{ED} each
  x_conv = causal_conv1d(x_proj, state)       # kernel=4
  x_act = SiLU(x_conv)
  Δ_t = softplus(Linear_Δ(x_act) + bias)      # ∈ R^{ED}
  B_t = Linear_B(x_act)                       # ∈ R^N
  C_t = Linear_C(x_act)                       # ∈ R^N
  A_bar_t = exp(Δ_t ⊙ A)                      # A ∈ R^{ED×N}
  B_bar_t = Δ_t ⊗ B_t
  h_t = A_bar_t ⊙ h_{t-1} + B_bar_t ⊗ x_act  # O(1) update
  y_t = C_t^T h_t                             # O(1) output
  y_t = y_t * SiLU(z)
  out_t = Linear_out(y_t)                     # ∈ R^D
  # 仅需存储固定大小h_t (ED×N), 无随seqlen增长的KV cache
  ```

  性能摘要：
  - Mamba-2.8B zero-shot avg 63.3, 超越 Pythia-6.9B (61.7) 和 GPT-J-6B (63.0)
  - Mamba inference: 5× higher throughput than Transformers (no KV cache, higher batch sizes)
  - Mamba-6.9B (untrained) throughput > 5× smaller Transformer-1.3B
  - Scan speed: faster than FlashAttention-2 beyond seqlen 2K, up to 40× faster than standard scan
