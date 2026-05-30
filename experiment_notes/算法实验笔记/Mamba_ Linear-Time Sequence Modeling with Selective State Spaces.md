## Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Mamba 提出一种选择性状态空间模型（S6），核心是将 SSM 的连续参数（Δ, B, C）由静态改为输入依赖（input-dependent），使模型能沿序列维度"选择性地"传播或遗忘信息。具体包括：i) 选择机制（Selection Mechanism）：将 Δ = τ_Δ(Parameter + s_Δ(x)), B = s_B(x), C = s_C(x) 参数化为输入 x 的函数，其中 s_B(x) = Linear_N(x), s_C(x) = Linear_N(x), s_Δ(x) = Broadcast_D(Linear_1(x)), τ_Δ = softplus；ii) Mamba 架构：将 H3 的 SSM 块与传统 MLP 块合并为同质化单一模块（图3），使用 gate 分支 + SiLU 激活 + 卷积 + 选择性 SSM 的主分支，无 attention 甚至无传统 MLP 块，扩展因子 E=2，两个 Mamba 块匹配一个 Transformer（MHA+MLP）的参数数（≈12D²）；iii) 硬件感知并行扫描算法（见 kernel调度 层）。
  - 实验比较：在语言、DNA、音频三个模态上与以下 baseline 对比：
    - 语言建模（Pile, GPT2/NeoX tokenizer）：vs Transformer (GPT3)、Transformer++（LLaMa 风格，RoPE + SwiGLU + RMSNorm + 高 LR）、Hyena、H3++、RWKV、RetNet；下游零样本评估 vs Pythia、OPT、GPT-Neo、RWKV
    - DNA 建模（HG38 human genome）：vs Transformer++、HyenaDNA
    - 音频建模（YouTubeMix piano, SC09 speech）：vs SaShiMi (S4+MLP UNet)、WaveNet、SampleRNN、WaveGAN、DiffWave
    - 合成任务：Selective Copying（序列长度 4096）和 Induction Heads（训练长度 256，测试外推至 1M）
  - 指标：perplexity（语言/DNA）、bits per byte（音频）、下游准确率（LAMBADA/HellaSwag/PIQA/Arc-E/Arc-C/WinoGrande）、物种分类准确率、FID/IS/mIS/AM（语音生成质量）、训练速度（scan 速度 vs FlashAttention-2）、推理吞吐量、内存消耗

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB PCIe GPU
  - 训练配置：语言模型 125M–1.3B（Chinchilla 缩放律，总计 2.5B–26B tokens），大模型下游评估延伸至 2.8B 参数、300B tokens；DNA 模型 ~250K–40M 参数、上下文长度 1024–1M；音频模型 ~3.5M–24M 参数
  - 优化器：AdamW，β=(0.9, 0.95)，weight decay 0.1，gradient clip 1.0，cosine LR schedule with linear warmup
  - 混合精度：BF16

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mamba（125M/350M/760M/1.3B/1.4B/2.8B 用于语言；~250K–40M 用于 DNA；~3.5M–24M 用于音频）。所有模型使用 real-valued diagonal SSM（S4D-Real 初始化），状态维度 N=16，Δ 投影维度 R=64（默认为 D 的小比例）。
  - 数据集：
    - 语言：The Pile（800GB），GPT2 tokenizer（缩放律）/ GPT-NeoX tokenizer（下游评估）
    - DNA：HG38 human genome（~4.5B 碱基对训练集）
    - 音频：YouTubeMix（4 小时钢琴独奏，16000Hz，mu-law 8-bit 编码）；SC09（1 秒语音片段，16000Hz，数字"zero"~"nine"）
  - Benchmark：
    - 语言：Pile 验证集 perplexity + 零样本：LAMBADA, HellaSwag, PIQA, Arc-Easy, Arc-Challenge, WinoGrande（使用 EleutherAI lm-evaluation-harness）
    - DNA：HG38 验证集 perplexity + Great Apes 五物种分类（{human, chimpanzee, gorilla, orangutan, bonobo}，共享 99% DNA）
    - 音频：YouTubeMix bits per byte (BPB)；SC09 无条件生成 NLL, FID, IS, mIS, AM

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/state-spaces/mamba
  - 算法 pipeline（选择性 SSM 前向传播，以单通道为例）：

  ```
  输入: x ∈ R^{B×L×D}  (batch, sequence length, model dimension)
  输出: y ∈ R^{B×L×D}

  参数:
    A ∈ R^{D×N}  (diagonal state matrix, N ≈ 16)
    Δ 投影权重: W_Δ ∈ R^{D×R}, b_Δ ∈ R^D  (R=64 典型值)
    B 投影权重: W_B ∈ R^{D×N}
    C 投影权重: W_C ∈ R^{D×N}
    输入投影: W_in ∈ R^{D×2ED}, W_out ∈ R^{ED×D}  (E=2)

  前向过程 (per Mamba block):
    1. 输入投影 (gate + main branch):
       x_proj = x @ W_in^T  →  R^{B×L×2ED}
       x_gate, x_main = split(x_proj, dim=-1)  → 各 R^{B×L×ED}

    2. 1D 卷积 (short convolution on main branch):
       x_conv = SiLU(Conv1d(x_main))  →  R^{B×L×ED}

    3. 选择性 SSM (S6) — 对每个通道独立执行:
       Δ = softplus(W_Δ @ x_conv + b_Δ)  →  R^{B×L×D}
       B = W_B @ x_conv  →  R^{B×L×N}
       C = W_C @ x_conv  →  R^{B×L×N}

       离散化 (Zero-Order Hold):
       Ā = exp(Δ ∘ A)  →  R^{B×L×D×N}  (∘ 表示 broadcast element-wise)
       B̄ = Δ ∘ B       →  R^{B×L×D×N}  (一阶近似)

       并行关联扫描 (recurrent form, fused in SRAM):
       h_0 = 0
       h_t = Ā_t ⊙ h_{t-1} + B̄_t ⊙ x_conv_t  (⊙ = element-wise)
       y_ssm_t = C_t ⊙ h_t

    4. Gate:
       y = (y_ssm * SiLU(x_gate)) @ W_out^T  →  R^{B×L×D}

    5. Residual connection + LayerNorm (如前文所述可选)
  ```

  - 核心洞察：
    - **选择性**：Δ 控制"关注当前输入 vs 保持历史状态"的平衡（广义 RNN gating），大 Δ≈reset 并关注新输入，小 Δ≈忽略当前输入保持历史。B 和 C 的选择性提供更细粒度的输入→状态、状态→输出控制
    - **时间复杂度**：训练 O(BLDN) 比 Transformer O(BL²D) 在长序列上更高效；自回归推理 O(1) 每步，无需 KV cache
    - **Theorem 1**：当 N=1, A=-1, B=1 时，选择性 SSM 退化为经典 gated RNN: g_t = σ(Linear(x_t)), h_t = (1-g_t)h_{t-1} + g_t·x_t

  - 关键结果：
    - Mamba-2.8B 在零样本评估中平均 accuracy 63.3%，超过 Pythia-6.9B (61.7%) 和同规模所有 baseline
    - Mamba-1.4B 生成吞吐量 5× 于同规模 Transformer
    - Induction Heads 训练长度 256 时完美外推至 ≥1M 长度（4000× 训练长度）
    - DNA 1M 上下文下 Great Apes 分类达 81.31% (7M 参数模型)
    - SC09 语音生成 FID 0.67 (24.3M)，超越 WaveGAN (2.03)、DiffWave (1.92) 等 GAN/扩散方法
