## xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：xLSTM 7B 是一个基于 mLSTM（Matrix LSTM）cell 的纯递归 LLM 架构，具有线性计算复杂度和常量内存消耗。核心实现包括：(1) **Post-up projection block**：mLSTM cell 在 embedding 维度（4096）而非更高维度空间运行，每层 mLSTM 后接 SwiGLU MLP（projection factor 2.66），32 个 block 堆叠；(2) **Multi-head mLSTM**：8 个 head（head dim d_hv=512, d_qk=256），每个 head 有独立的矩阵记忆状态 C_t ∈ R^{256×512}；(3) **mLSTM cell**：使用标量指数输入门 i_t 和遗忘门 f_t（通过 soft-capping a=15 稳定）、向量输出门 o_t，记忆更新为 C_t = f_t C_{t-1} + i_t k_t v_t^T；(4) **训练稳定性改进**：RMSNorm 替代 LayerNorm 作为 pre-norm、输入门 bias 初始化为 -10、输出 logits soft-capping a=30；(5) 训练时使用 chunkwise-parallel 模式（类似 FlashLinearAttention），推理时使用 recurrent 模式（常量记忆）；(6) 使用 GPT-NeoX-20B tokenizer（vocab size 50257）；(7) 序列打包时在 EOD token 处通过置遗忘门为零来重置记忆状态。
  - 实验比较：(1) **语言建模性能**：在 Huggingface Open LLM Leaderboard v2 上与 Llama-3.1-8B、Llama-2-7B、OLMo-7B、Gemma-7B、Ministral-8B、Bloom-7B1、GPT-J-6B、Pythia-6.9B、Qwen2.5-7B、Gemma-2-9B、DCLM-7B、Zamba2-7B、Falcon-Mamba-7B、MambaCodestral-7B、RWKV-v5-Eagle-7B、RWKV-v6-Finch-7B 对比 6 项指标（BBH、MMLU-Pro、MATH、MuSR、GPQA、IFEval），Tab. 1；(2) **长上下文评估**：在 RULER benchmark 上（4K 到 131K token）与 Llama-2-7B、Llama-3.1-8B、CodestralMamba、FalconMamba、RWKV-5/6 对比，包括标准版 xLSTM 7B 和长上下文版 xLSTM 7B LCTX，Fig. 3；(3) **推理速度 benchmark**：在单 NVIDIA H100 GPU、batch size 1 下，测量生成吞吐 vs prefill 长度（Fig. 4）、生成时间和 GPU 内存 vs 生成长度（Fig. 5）、Time To First Token 延迟 vs prefill 长度（Fig. 6）、prefill 吞吐 vs batch size 和 context length（Fig. 7），与 Llama-2-7B、Llama-3.1-8B、Falcon-Mamba-7B、Codestral-Mamba-7B 对比；(4) **消融实验**：Pre-up vs Post-up Projection Block 的验证 perplexity 和吞吐对比（160M/400M/1.4B/7B 参数，Tab. 2）；Head 数量（4/8/16/32）对记忆状态大小、训练步时、验证 perplexity 的影响（7B 参数 160B tokens，Tab. 3）；Norm Layer 类型（LayerNorm vs RMSNorm）对训练稳定性的影响（Fig. 9）；Soft-capping 效果（Fig. 10）；输入门 bias 初始化值的影响（Fig. 11）；学习率调度器类型的影响（Fig. 12）；记忆状态大小和输入门对长上下文的影响（RULER, Fig. 13, Tab. 7, 8）。

- 硬件平台是什么，配置是什么。
  - 训练：128× NVIDIA H100 GPU，使用 FSDP（Fully Sharded Data Parallel）和 activation checkpointing，batch size 512，context length 8192，550K 训练步（2.3T tokens），AdamW optimizer（peak lr=5e-4, β1=0.99, β2=0.95, ε=1e-8, weight decay=0.1, gradient clipping=0.5）
  - 推理 benchmark：单 NVIDIA H100 GPU，batch size 1，使用 HuggingFace transformers 模型实现 + torch.compile + PyTorch CUDA Graphs
  - 消融训练：160M/400M 模型用 16 GPU batch size 128；1.4B 用 32 GPU batch size 256；7B ablation 用 128 GPU batch size 256，76K 步（160B tokens）

- 模型是什么。数据集和bench分别是什么。
  - 模型：xLSTM 7B（6,865,424,896 参数），32 blocks，embedding dim 4096，8 heads，head dim 512，d_qk=256
  - 训练数据集（第一阶段 500K 步）：DCLM（DataComp-LM）
  - 训练数据集（第二阶段 50K 步）：DCLM (40%)、FineWeb-Edu (15%)、Cosmopedia (10%)、ProofPile-2 (15%)、TheStack (15%)、SFT datasets (5%，包括 NuminaMath CoT、MetaMathQA、Tulu v3.1、OpenHermes 2.5、GSM8K、Smoltalk)
  - 长上下文 cool-down：DCLM (20%)、FineWeb-Edu (15%)、Cosmopedia (10%)、ProofPile-2 (15%)、TheStack (15%)、LongDataCollections (15%)、SFT (5%)、LongAlign10k (1%)、AntiHayStack (1%)、LongAlpaca12k (2%)
  - Benchmarks：Open LLM Leaderboard v2 (BBH, MMLU-Pro, MATH, MuSR, GPQA, IFEval)、Open LLM Leaderboard v1 (ARC-C, MMLU, HellaSwag, WinoGrande, TruthfulQA, OpenBookQA, PiQA)、RULER（长上下文 4K-131K）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 模型权重：https://huggingface.co/NX-AI/xLSTM-7b
  - 模型代码：https://github.com/NX-AI/xlstm 和 https://github.com/NX-AI/xlstm-jax
  - Triton kernel：https://github.com/NX-AI/mlstm_kernels
  - **算法伪代码（推理时/mLSTM Cell Recurrent Mode）**：
    ```
    # 输入: x_t ∈ R^d, 上一状态 (h_{t-1}, C_{t-1}, n_{t-1}, m_{t-1})
    # 参数: W_q, W_k, W_v, b_q, b_k, b_v; w_i, w_f, b_i, b_f; W_o, b_o
    # 超参: d_qk = d_hv / 2, softcap_a = 15

    # 1. 投影
    q_t = W_q @ x_t + b_q   # ∈ R^{d_qk}
    k_t = W_k @ x_t + b_k   # ∈ R^{d_qk}
    v_t = W_v @ x_t + b_v   # ∈ R^{d_hv}

    # 2. Gate pre-activations (with soft-capping)
    i_tilde = softcap_a(w_i^T @ x_t + b_i)   # scalar
    f_tilde = softcap_a(w_f^T @ x_t + b_f)   # scalar
    o_tilde = W_o @ x_t + b_o                # ∈ R^{d_hv}

    # 3. Gate activations
    m_t = max(log(σ(f_tilde)) + m_{t-1}, i_tilde)
    f_t = exp(log(σ(f_tilde)) + m_{t-1} - m_t)
    i_t = exp(i_tilde - m_t)
    o_t = σ(o_tilde)

    # 4. Memory State Update
    C_t = f_t * C_{t-1} + i_t * (k_t ⊗ v_t)  # outer product, C_t ∈ R^{d_qk × d_hv}
    n_t = f_t * n_{t-1} + i_t * k_t            # n_t ∈ R^{d_qk}

    # 5. Hidden State Retrieval
    q_norm = q_t / sqrt(d_qk)
    h_tilde = C_t^T @ q_norm / max(|n_t^T @ q_norm|, exp(-m_t))

    # 6. Output
    h_t = o_t ⊙ Norm(h_tilde)   # Norm = RMSNorm or LayerNorm
    ```
  - **Block 结构（Post-up Projection）**：
    ```
    # 输入: x ∈ R^{T×d}
    # 32个相同的block, 每个block:
    z = x + mLSTM(RMSNorm(x))      # mLSTM: multi-head mLSTM cells, concatenate + project
    y = z + SwiGLU_MLP(RMSNorm(z))  # SwiGLU with proj_factor=2.66
    ```
  - **训练时 Chunkwise-Parallel 模式**：将序列分块，块内并行计算（类似 FlashLinearAttention），块间通过 recurrent state 传递。避免了显式存储中间 (C_t, n_t, m_t) 状态。

## Samba__Simple_Hybrid_State_Space_Models_for_Efficient_Unlimited_Context_Language_Modeling

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：SAMBA 是一种简单混合神经架构，按层交替组合 Mamba（选择性 SSM）、Sliding Window Attention（SWA，窗口大小 2048）和 SwiGLU MLP。Mamba 层捕获时间依赖语义，通过选择性状态空间实现长程记忆的递归压缩；SWA 层提供精确的短中期记忆召回；MLP 层负责非线性变换和事实知识召回。每层均使用 Pre-Norm（RMSNorm）和残差连接。Mamba 层参数：扩展维度 d_e = 2*d_m，低秩维度 d_r = d_m/16，状态维度 d_s = 16，短卷积核大小 k=4，选择性门控 [Δ_min, Δ_max] = [0.001, 0.1]，使用 S4D-Real 初始化 A 矩阵（A_ij = log(j)）。SWA 层使用 RoPE（base frequency=10,000）和 FlashAttention 2。Samba 层排列：重复 [Mamba → MLP → SWA → MLP] 的模式。最大的 3.8B 模型有 64 层，模型宽度 d_m=2816，11 个 query head，1 个 KV head。
  - 实验比较：(1) 1.7B 规模：在 Phi-2 数据集（230B tokens）上与 Llama-3 1.6B、Mistral 1.6B、Mamba 1.8B、Mamba-SWA-MLP 1.6B、Mamba-MLP 1.9B 对比 15 个下游 benchmark（Table 2）；(2) 438M/1.3B 规模：在 SlimPajama 上与 Llama-2、Llama-2-SWA、Mamba、Sliding GLA、Sliding RetNet、Mega-S6、Mamba-SWA-MLP、MLP2-SWA-MLP、Samba-NoPE 对比 perplexity 和训练吞吐（Table 3）及下游任务（Table 4）；(3) 3.8B 规模：在 Phi-3 数据集（3.2T tokens）上与 Phi-3-mini、Llama 2、Mistral、Mamba、Gemma、RecurrentGemma、Llama 3、TFM++ 对比（Table 1, 7, 8）；(4) 长度外推：在 Proof-Pile 上评估 1M 上下文 perplexity（Figure 2），1.7B 模型在 Passkey Retrieval 上微调 500 步后外推到 256K（Figure 3），Phonebook 任务上外推到 8K（Figure 4）；(5) 效率：在单 A100 GPU 上测量 128K prompt 吞吐和 64K 解码吞吐（Figure 2, Figure 6）；(6) 消融：全注意力替代（Table 5）、注意力头数优化（Table 6）、SWA 训练长度 vs batch size（Table 9）、短卷积效果（Table 10）。

- 硬件平台是什么，配置是什么。
  - 训练：8×H100 GPU（训练速度测量）、64×H100 GPU（1.3B 模型 100B tokens 训练）、8×A100 GPU（438M 模型训练速度测量）
  - 推理吞吐：单张 A100 GPU，bfloat16 精度，batch size 16（解码），重复 10 次取平均
  - 训练基础设施基于 TinyLlama 代码库修改版（https://github.com/jzhang38/TinyLlama）
  - 优化器：AdamW，weight decay=0.1，gradient clipping=1.0

- 模型是什么。数据集和bench分别是什么。
  - 模型：SAMBA（421M/1.3B/1.7B/3.8B 参数），对比模型包括 Llama-2、Llama-3、Mistral、Mamba、Mamba-SWA-MLP、Mamba-MLP、Sliding GLA、Sliding RetNet、Mega-S6、MLP2-SWA-MLP、Phi-3-mini、TFM++、Gemma、RecurrentGemma、Jamba-1.5-Mini、FalconMamba
  - 训练数据集：SlimPajama（627B tokens，清洗版 RedPajama）、Phi-2 数据集（230B tokens）、Phi-3 数据集（3.2T tokens，教科书质量数据）
  - Benchmarks：常识推理（ARC-Easy/ARC-Challenge、PIQA、WinoGrande、SIQA、HellaSwag）、语言理解（BoolQ、OpenbookQA、SQuAD、MMLU、MMLU-Pro、GPQA）、真实性（TruthfulQA MC1/MC2）、数学与代码（GSM8K、MBPP、HumanEval）、长文本摘要（GovReport、SQuALITY via ZeroSCROLLS）、记忆召回（Passkey Retrieval 256K、Phonebook）、perplexity（SlimPajama 验证集、Proof-Pile 测试集）、LAMBADA

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/microsoft/Samba
  - 算法流程伪代码：
    ```
    # 输入: X ∈ R^{n × d_m}, n=序列长度, d_m=隐藏维度
    
    # === Mamba 层 ===
    H = X @ W_in                    # [n, d_e], d_e = 2*d_m
    U = SiLU(DepthwiseConv(H))      # 短卷积平滑, kernel=4
    Delta = Softplus(U @ W_r @ W_q + b)  # 选择性门控, W_r: [d_e, d_r], d_r=d_m/16
    B = U @ W_b                     # [n, d_s], d_s=16
    C = U @ W_c                     # [n, d_s]
    # S6 选择性状态空间, Z ∈ R^{d_e × d_s}
    Z_t = exp(-Delta_t ⊙ exp(A)) ⊙ Z_{t-1} + Delta_t ⊙ (B_t ⊗ U_t)
    Y_t = Z_t @ C_t + D ⊙ U_t      # D 初始化为 1
    # 门控输出
    O_mamba = (Y ⊙ SiLU(X @ W_g)) @ W_out   # [n, d_m]
    
    # === SWA 层 (窗口=2048, RoPE, FlashAttention 2) ===
    Q, K, V = X @ W_q, X @ W_k, X @ W_v
    Q, K = apply_rotary_pos_emb(Q, K, base=10000)
    # 对每个位置 t, 只在 [max(0, t-2048), t] 范围内计算 attention
    O_swa = FlashAttention2(Q, K, V, window_size=2048) @ W_o
    
    # === SwiGLU MLP 层 ===
    O_mlp = (SiLU(X @ W_gate) ⊙ (X @ W_up)) @ W_down
    
    # === 层排列 (Samba, N=48 for 1.7B) ===
    # Layer 0:  Mamba
    # Layer 1:  MLP (after Mamba)
    # Layer 2:  SWA
    # Layer 3:  MLP (after SWA)
    # Layer 4:  Mamba
    # ... 重复 12 个 block
    ```
  - 关键设计：Mamba 层中的选择性状态空间 S6 通过输入依赖的门控 Δ 实现软序列选择，使模型能够在递归状态中记住重要信息；SWA 层通过直接注意力机制精确召回窗口内的信号，弥补 Mamba 在精确记忆召回方面的不足；两种层的交替排列使 Mamba 专注于捕获递归结构而非执行精确检索。RoPE 对于长度外推至关重要（Samba-NoPE 在超过训练长度后 perplexity 爆炸）。训练使用 Mamba 的硬件感知并行扫描算法实现高效并行化。

## SSMLoRA__Enhancing_Low-Rank_Adaptation_with_State_Space_Model

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：SSMLoRA 提出基于状态空间模型（SSM）增强的低秩适应方法。核心是 Time Module，沿时间轴连接跨层插入的低秩矩阵，通过状态转移方程实现层间信息传递。Time Module 内部包含四个矩阵：W_a (d×r)、W_b (r×d)、W_c (r×r, 状态矩阵)、W_d (r×r, 控制矩阵)。前向过程：(1) 低秩投影 `x_new = x × W_a`；(2) 状态导数 `h_t' = h_{t-1} × W_c + x_new × W_d`；(3) Taylor 展开近似 `h_t = h_t' + h_{t-1}`；(4) min-max 归一化后将 `h_t_norm` 作为偏置：`y = x × W_0 + (x_new + h_t_norm) × W_b`。与 S4 不同，SSMLoRA 使用 Taylor 展开直接离散化而非离散化状态矩阵 W_c 和 W_d，避免了 FFT 开销（需时可开放）。稀疏插入策略：只在 attention 的 query 和 value 矩阵交替间隔插入 Time Module（层 l 激活 query，层 l+1 跳过，query 和 value 分别独立时间轴）。初始化策略：W_a 用 scaled Gaussian，W_b/W_c/W_d/h 为零初始化，使模型初始退化为稀疏 LoRA。
  - 实验比较：(1) GLUE benchmark（CoLA/SST-2/MRPC/STS-B/QQP/MNLI/QNLI/RTE）上 RoBERTa-base 对比 Fine-tune/BitFit/LoRA/QLoRA/MixLoRA（Table 1）；(2) LLaMA2-7B/13B 对比 LoRA（Table 2: RTE/BoolQ/WSC/WiC/MultiRC/COPA）；(3) 长文本能力——DeBERTaV3-base 在 SQuAD/NarrativeQA 上对比 Fine-tune/BitFit/LoRA（Table 3），RoBERTa-base 在 RACE 上对比 LoRA（Table 4）；(4) 不同长度区间 NarrativeQA 性能（Figure 2/Table 5）；(5) 稀疏性消融——RoBERTa-large 和 GPT-2 在 GLUE 上对比 LoRA 的不同 rank r=1/2/4/8/16（Table 6/7），SuperGLUE 对比（Table 8）；(6) 内存效率——LLaMA2-7B 上序列长度 16→7000 的 GPU 内存和推理耗时对比（Table 12）；(7) 参数效率对比（Table 9）；(8) 训练 wallclock time（Table 10/11）。

- 硬件平台是什么，配置是什么。
  - 小型模型（DeBERTaV3-base/RoBERTa-base/RoBERTa-large/GPT-2）：单卡 NVIDIA RTX 3090 (24GB)。
  - 大型模型（LLaMA2-7B/LLaMA2-13B）：单卡 NVIDIA RTX A6000 (48GB)。
  - 学习率范围：[5e-4, 1e-6]，采用动态学习率调度和 early stopping。LoRA 类方法统一超参：α=16, rank=8（消融实验除外），dropout=0.1。

- 模型是什么。数据集和bench分别是什么。
  - 模型：RoBERTa-base (124M)、RoBERTa-large (355M)、GPT-2、DeBERTaV3-base、LLaMA2-7B、LLaMA2-13B。
  - GLUE benchmark：CoLA (MCC)、SST-2 (Acc)、MRPC (F1/Acc)、STS-B (Pearson/Spearman Corr)、QQP (F1/Acc)、MNLI-m/mm (Acc)、QNLI (Acc)、RTE (Acc)。
  - SuperGLUE benchmark：RTE/BoolQ/WSC/WiC/MultiRC/COPA/CB/ReCoRD。
  - 长文本/推理：SQuAD (F1/EM)、NarrativeQA (ROUGE-L)、RACE (Acc)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/yuhkalhic/SSMLoRA（NAACL 2025 main conference）
  - 安装：Python 3.10, `pip install -r requirements.txt`。训练命令：`python src/main.py --dataset BoolQ`。
  - 算法 pipeline 张量计算流程（以 RoBERTa-base 为例，单层 attention 的 query 矩阵 W_Q 插入 Time Module）：
    ```
    # 初始化
    W_a: d×r = randn(d, r) * scale_gaussian  # scaled Gaussian
    W_b: r×d = zeros(r, d)  # 零初始化
    W_c: r×r = zeros(r, r)  # 状态矩阵，零初始化
    W_d: r×r = zeros(r, r)  # 控制矩阵，零初始化
    h:   1×r  = zeros(1, r)  # 状态向量，零初始化（时间轴起点的 Time Module）

    # Time Module 前向传播（每个被激活的 attention 层）
    x_new = x @ W_a                # [batch, seq, d] × [d, r] → [batch, seq, r]
    h_prime = h @ W_c + x_new @ W_d  # [1, r] × [r, r] + [batch, seq, r] × [r, r] = [batch, seq, r]
    h_new = h_prime + h            # Taylor 展开近似：[batch, seq, r]
    # min-max 归一化（per-batch）
    h_norm = (h_new - h_new.min()) / (h_new.max() - h_new.min() + 1e-8)
    y_lora = (x_new + h_norm) @ W_b  # [batch, seq, r] × [r, d] → [batch, seq, d]
    y = x @ W_Q + y_lora           # 最终输出

    # 状态传递（h 脱离计算图，仅 W_c/W_d/x 参与训练）
    h = h_new.detach()

    # 稀疏插入策略（query 和 value 独立时间轴，交替间隔）：
    # 对于 attention 中的 query 矩阵 W_Q：
    #   Layer l:   插入 Time Module（激活），接收来自激活层的 state
    #   Layer l+1: 跳过（不插入）
    #   Layer l+2: 插入 Time Module（激活）
    # ...
    # 对于 attention 中的 value 矩阵 W_V：独立的另一条时间轴，同样的交替间隔模式
    # 非 attention 层（如 FFN、classifier）：标准 LoRA 稠密插入（W_a + W_b，无 W_c/W_d/h）
    ```
    关键特性：(1) 参数仅约为 LoRA 的 50%（因交替间隔稀疏 + 仅 q/v 插入）；(2) FFT 可选——公式 (2) 计算为矩阵乘法而非卷积，但如果需要可启用 S4 的 FFT 加速；(3) 跨层状态传递使模型能关联不同层的低秩映射信息；(4) 零初始化使训练起点退化为稀疏 LoRA，渐进学习 SSM 连接。

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：RWKV（Receptance Weighted Key Value）模型架构，结合RNN和Transformer优势。核心设计包括：(1) Token Shift——每个block的时间混合输入由当前和前一个时间步的线性插值产生（`r_t = W_r · (μ_r ⊙ x_t + (1-μ_r) ⊙ x_{t-1})`）；(2) WKV Operator——通道级时间衰减线性注意力，`wkv_t = Σe^{-(t-1-i)w+k_i}⊙v_i / Σe^{-(t-1-i)w+k_i}`，将Transformer的O(T²d)降为O(Td)；(3) Output Gating——使用σ(r_t)⊙wkv_t作为输出；(4) Channel-Mixing Block——使用squared ReLU激活：`o'_t = σ(r'_t) ⊙ (W'_v · max(k'_t, 0)²)`。训练时使用time-parallel模式（类似Transformer并行），推理时使用time-sequential模式（类似RNN逐token递归）。
  - 实验比较：(1) FLOP匹配的零样本NLP评估——与Pythia/OPT/BLOOM在12个benchmark上比较；(2) 扩展上下文微调——从1024→2048→4096→8192逐步增加序列长度；(3) 长序列LRA benchmark与S4等模型比较；(4) Enwik8字符级语言建模；(5) 推理时延和内存与Transformer家族（BLOOM/OPT/GPT-Neo/Pythia）在CPU和GPU上比较；(6) 与ChatGPT/GPT-4在RTE/WNLI/GoEmotions等任务上的提示工程对比。

- 硬件平台是什么，配置是什么。
  - 训练：StabilityAI提供的GPU集群（论文未具体说明GPU型号和数量）
  - 推理benchmark：CPU (x86) 和 GPU (NVIDIA A100 80 GB)，使用float32精度

- 模型是什么。数据集和bench分别是什么。
  - 模型规模：RWKV 169M (12层/d=768)、430M (24层/d=1024)、1.5B (24层/d=2048)、3B (32层/d=2560)、7B (32层/d=4096)、14B (40层/d=5120)
  - 训练数据：The Pile (800GB, 330B tokens)，训练1个epoch
  - 评估benchmark：ARC (Easy/Challenge)、BoolQ、COPA、HeadQA、HellaSwag、LAMBADA、OpenBookQA、PIQA、ReCoRD、SciQ、Winogrande、LRA (ListOps/Text/Retrieval/Image/Pathfinder/Path-X)、Enwik8

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/BlinkDL/RWKV-LM
  - 预训练模型：https://huggingface.co/RWKV
  - Chat接口：https://github.com/BlinkDL/ChatRWKV
  - 算法pipeline伪代码（每个RWKV block的前向过程）：

```
# 输入: x_t (当前token), x_{t-1} (上一token), 状态 (a_{t-1}, b_{t-1}, p_{t-1})
# === Time-Mixing Block ===
# Token Shift
r_t = W_r @ (μ_r * x_t + (1-μ_r) * x_{t-1})       # [d]
k_t = W_k @ (μ_k * x_t + (1-μ_k) * x_{t-1})       # [d]
v_t = W_v @ (μ_v * x_t + (1-μ_v) * x_{t-1})       # [d]

# WKV Operator (channel-wise, 每个通道独立)
q = max(p_{t-1}, u + k_t)                           # 数值稳定
wkv_t = (e^{p_{t-1}-q} * a_{t-1} + e^{u+k_t-q} * v_t) /
        (e^{p_{t-1}-q} * b_{t-1} + e^{u+k_t-q})    # [d], element-wise

# 状态更新
q' = max(p_{t-1} - w, k_t)
a_t = e^{p_{t-1}-w-q'} * a_{t-1} + e^{k_t-q'} * v_t   # [d]
b_t = e^{p_{t-1}-w-q'} * b_{t-1} + e^{k_t-q'}        # [d]
p_t = q'

# Output Gating
o_t = W_o @ (σ(r_t) * wkv_t)                         # [d]

# === Channel-Mixing Block ===
r'_t = W'_r @ (μ'_r * x_t + (1-μ'_r) * x_{t-1})     # [d]
k'_t = W'_k @ (μ'_k * x_t + (1-μ'_k) * x_{t-1})     # [d]
o'_t = σ(r'_t) * (W'_v @ max(k'_t, 0)²)              # [d], squared ReLU
```
  关键张量形状：输入x为[B,T,d]（训练）或[1,1,d]（推理逐token）；W_r/k/v/o为[d,d]；μ为[d]（可学习token shift参数）；w为[d]（可学习通道时间衰减，非负）；u为[d]（当前token bonus参数）。训练时WKV通过串行扫描（可用parallel scan优化至O(B log T d)），推理时递归更新仅需O(d)空间和O(d)时间。

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

## RWKV-X__A_Linear_Complexity_Hybrid_Language_Model

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是RWKV-X——一种结合RWKV-7线性RNN block与Top-k Chunk Sparse Attention block的线性复杂度混合语言模型架构。核心创新：(1) Top-k Chunk Sparse Attention：将输入序列划分为n个等大chunk（size=B），对每个query q计算与各chunk的mean-pooled key的内积得分s_i=q·(1/B Σ_j k_j^(i))，通过TopK选择得分最高的k个chunk，仅在这k个chunk上计算softmax attention Attn(q,K_I,V_I)=softmax(qK_I^T/√d_k)V_I，将二次attention变为O(kBN)≈O(N)；(2) KV Cache Management：将past cache分为earlier cached states (K_past,V_past)和recent observation window (K_obs,V_obs)，通过softmax attention scores累积重要性得分C=Σ_i softmax(Q_obs K_past^T/√d_k)[i,:]，保留top-m最相关entries，与observation window拼接为固定大小cache（灵感来自SnapKV），实现解码阶段O(1)空间复杂度；(3) RWKV-7 block作为主干：基于generalized Delta Rule的state evolution S_t=S_{t-1}M_t+v_t^T·k̃_t，其中M_t=diag(w_t)-κ̂_t^T(a_t⊙κ̂_t)，通过channel-wise state更新实现高效short-range建模；(4) Block Expansion Method：从RWKV-7 checkpoint出发，interleaved插入Sparse Attention block，零初始化新参数（借鉴LLaMA Pro），先alignment training（freeze RWKV-7 blocks，MiniPile 1.5B tokens，1024 context）再long-context continual pretraining（unfreeze all，ProLong-64K 1B tokens，64K context）；(5) LongCE Loss：对传统CE loss中各token施加动态权重（critical tokens weight>1, ordinary tokens weight≈1），使模型自动聚焦长程依赖关键token。
  实验比较：(a) Long Context Evaluation（Table 2）：S-NIAH benchmark（S-NIAH-1 Passkey Retrieval, S-NIAH-2 Number in Haystack, S-NIAH-3 UUID in Haystack），在1K/2K/4K/8K context上对比RWKV-7(0.19B/2.9B)、RWKV-X(0.22B/3.6B)、DeltaNet-1.3B、Mamba2-1.3B、Gated DeltaNet-1.3B、RWKV-6(1.6B/3B)——RWKV-X-3.6B在S-NIAH-2 8K达到99.8（RWKV-7-2.9B仅88.0），在S-NIAH-3 8K达到95.6（RWKV-7-2.9B仅79.0）；(b) Short Context Evaluation（Table 3）：LAMBADA/HellaSwag/PIQA/ARC-E/ARC-C/Winogrande/SciQ/MMLU，对比RWKV-5/6/7、SmoLLM2-135M、Llama3.2-3B、Qwen2.5-3B——RWKV-X-3.6B avg 71.9 vs RWKV-7-2.9B 72.8 vs Qwen2.5-3B 71.4；(c) Efficiency Analysis（Figure 3,4）：prefill latency vs Flash-Attention v3（RWKV-X at 128K 1.37× speedup over Flash-Attention v3），decoding latency stability up to 1M tokens（固定64K KV cache，constant time）vs RWKV-7-2.9B；(d) Ablation Study：LongCE loss消融（Table 4，S-NIAH-2 8K: w/ LongCE 99.8 vs w/o 67.0），attention layer比例消融（Figure 5，25% optimal），model size scaling消融（Table 5，RWKV-X vs GPT-2 at 10B tokens），positional encoding消融（Table 6，No Pos优于Abs Pos/ROPE）；(e) Training Efficiency（Figure 6）：RWKV-X vs RWKV-7 across 1K-32K sequence lengths。

- 硬件平台是什么，配置是什么。
  GPU: 8×H20（0.22B model Long Context阶段）, 4×H20（0.22B Alignment阶段）, 8×H200（3.6B model Long Context阶段）。Optimizer: AdamW, constant learning rate 1e-5, no warmup, no weight decay。DeepSpeed Stage 1。训练精度：论文未明确说明（推测BF16/FP16混合精度）。0.22B模型Alignment阶段: batch size=1.024M tokens, context=4096, trained 1.5B tokens, GPU hours=6。0.22B模型Long Context阶段: batch size=4.096M tokens, context=64K, trained 20B tokens, GPU hours=576。3.6B模型Long Context阶段: batch size=8.192M tokens, context=64K, trained 1B tokens, GPU hours=80。总trained tokens: 0.22B=1.6B (alignment pretraining only), 3.6B=1B long-context tokens。Efficiency benchmark: 使用Flash-Attention v3 for full-attention baseline，RWKV-X用sparse attention实现。

- 模型是什么。数据集和bench分别是什么。
  模型：RWKV-X series (0.22B, 3.6B)，此外有消融实验用的126M/355M/786M variants。架构：RWKV-7 blocks + periodically inserted Sparse Attention blocks（~25% layers为attention layers时validation loss最优，Figure 5）。使用RWKV-7 checkpoint初始化，block expansion方法插入新层。无positional encoding。Sparse Attention: chunk size=B, selected chunks=k, KV cache budget=m（论文未给具体常数，解码阶段cache压缩至固定大小64K）。
  数据集：(1) Alignment Phase: MiniPile dataset (Kaddour, 2023), context=1024, 1.5B tokens；(2) Long-context Phase: ProLong-64K dataset (Gao et al., 2025a), context=64K, 1B tokens（3.6B model）或20B tokens（0.22B model）。Benchmark：(1) Long Context: S-NIAH benchmark from RULER (Hsieh et al., 2024) — S-NIAH-1 (Passkey Retrieval), S-NIAH-2 (Number in Haystack), S-NIAH-3 (UUID in Haystack)，evaluated at 1K/2K/4K/8K context；(2) Short Context: LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, Winogrande, SciQ, MMLU；(3) Ablation: validation loss on language modeling (MiniPile/ProLong data)。Baselines: RWKV-5/6/7, DeltaNet, Mamba2, Gated DeltaNet, GPT-2, SmoLLM2, Llama3.2, Qwen2.5。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  完全开源。代码: https://github.com/howard-hou/RWKV-X，包含预训练checkpoints。

  RWKV-X核心算法pipeline——Token到logits前向过程：
  ```
  # Algorithm: RWKV-X Hybrid Block Forward Pass
  Input: x ∈ R^{B×L×D}, total layers L_total, Sparse Attention layers at indices I_attn
  h_0 = x  # embedding input
  
  For layer l = 1 to L_total:
      if l in I_attn:  # Sparse Attention Block
          # ---- Top-k Chunk Sparse Attention ----
          q, k, v = Linear_Q(h_{l-1}), Linear_K(h_{l-1}), Linear_V(h_{l-1})  # ∈ R^{B×L×d_head}
          
          # Step 1: Divide into chunks
          n = L // B  # B=chunk_size
          k_chunks = reshape(k, (B, n, B, d_head))  # (B,n,B,d_head)
          k_mean = mean(k_chunks, dim=2)             # (B,n,d_head)
          
          # Step 2: Compute chunk relevance scores
          s = einsum("bld,bnd->bln", q, k_mean)      # (B,L,n)
          
          # Step 3: Select top-k chunks
          I = topk(s, k, dim=-1)                      # (B,L,k)
          k_selected = gather(k_chunks, I)             # only selected chunks
          v_selected = gather(v_chunks, I)
          
          # Step 4: Sparse attention on selected chunks
          attn_scores = softmax(q @ k_selected^T / sqrt(d_k))
          h_attn = attn_scores @ v_selected
          h_l = h_{l-1} + Linear_O(h_attn)
          
      else:  # RWKV-7 Block (Time-Mixing + Channel-Mixing)
          # ---- Time-Mixing (Generalized Delta Rule) ----
          # Input projections
          r, k, v = receptance(x), key(x), value(x)  # via Linear layers
          
          # w: data-dependent decay vector
          w = exp(-exp(Linear_w(x)))  # ∈ R^{B×L×D}
          
          # a: context-dependent learning rate
          a = Linear_a(x)  # ∈ R^{B×L×D}
          
          # κ̂: normalized removal key
          κ = Linear_κ(x)
          κ̂ = κ / ||κ||_2
          
          # k̃: replacement key  
          k̃ = k  # (simplified)
          
          # State evolution (recurrent form):
          # S_t = S_{t-1} * diag(w_t) - S_{t-1} * (κ̂_t^T (a_t ⊙ κ̂_t)) + v_t^T * k̃_t
          
          # Implementation with parallel scan/chunked form:
          # M_t = diag(w_t) - κ̂_t^T(a_t ⊙ κ̂_t)  # transition matrix
          # S_{t+1} = S_t * M_{t+1} + v_{t+1}^T * k̃_{t+1}
          
          # Output: gating with receptance
          h_time = r * (S_t output via WKV linear attention)
          
          # ---- Channel-Mixing (FFN with gating) ----
          h_mlp = Linear_out(gate * SiLU(Linear_in(x)))
          
          h_l = h_{l-1} + h_time + h_mlp
  
  # ---- Decoding with KV Cache Management (Figure 7) ----
  # At each decode step t:
  # Past cache split:
  K_past, V_past = cache[:m_old]  # earlier cached states
  K_obs, V_obs = recent_window     # observation window
  
  # Importance scoring:
  C = sum(softmax(Q_obs @ K_past^T / sqrt(d_k)), dim=0)  # cumulative attn scores
  
  # Top-m selection:
  idx = topk(C, m)
  K_compressed = K_past[idx] || K_obs  # concatenate
  V_compressed = V_past[idx] || V_obs
  
  # Sparse attention on compressed cache (constant size m + L_obs):
  attn = softmax(q_new @ K_compressed^T / sqrt(d_k))
  output = attn @ V_compressed
  ```

  Block Expansion训练两阶段流程：
  ```
  # Stage 1: Alignment Pretraining (RWKV-7 blocks frozen)
  model = load_checkpoint("RWKV-7")
  model = insert_sparse_attention_blocks(model, indices=every_4th_layer, init="zero")
  
  for batch in MiniPile(context=1024):
      # Only sparse attention block params receive gradients
      frozen_params = model.rwkv7_blocks.parameters()  # no grad
      trainable_params = model.sparse_attn_blocks.parameters()
      loss = LongCE_loss(model(batch))
      loss.backward()  # updates only sparse attention blocks
  
  # Stage 2: Long-context Continual Pretraining (all params unfrozen)
  model.unfreeze_all()
  for batch in ProLong-64K(context=64000):
      loss = LongCE_loss(model(batch))  # LongCE assigns dynamic weights per token
      loss.backward()  # updates all parameters
  ```

  性能摘要：
  - RWKV-X-3.6B S-NIAH-2 8K: 99.8 (RWKV-7-2.9B: 88.0)，S-NIAH-3 8K: 95.6 (RWKV-7-2.9B: 79.0)
  - Short-context avg: 71.9 vs Qwen2.5-3B 71.4 vs Llama3.2-3B 69.7
  - 128K prefill: 1.37× speedup over Flash-Attention v3
  - Decoding latency stable up to 1M tokens (fixed 64K KV cache)
  - Training complexity: O(kBN+N) ≈ O(N)，Decoding complexity: O(1) per token

## Linearizing_Large_Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是SUPRA（Scalable UPtraining for Recurrent Attention）——一种将预训练softmax Transformer（Llama2、Mistral）通过有限的继续训练（uptraining）转换为线性RNN的技术。核心创新：(1) 用MLP kernel（φ(x)=relu(Wx+b)，queries和keys共享权重）替换softmax计算；(2) 用GroupNorm（借鉴RetNet）替换传统线性注意力的分母归一化，解决大规模uptraining的数值不稳定问题；(3) 引入RoPE相对位置编码增强位置建模；(4) 使用固定衰减向量γ∈(0,1)^h（借鉴RetNet的decay机制）。最终的注意力形式为 v'_i = GroupNorm(Σ_{j=1}^{i} γ^{i-j}·sim(q_i,k_j)·v_j)，其中sim(q_i,k_j)=RoPE(φ(q_i))·RoPE(φ(k_j))。训练时使用5%的预训练token量（20B-100B tokens），训练新引入的参数（MLP kernel权重），同时联合微调全部网络参数。
  实验比较：(a) 短上下文NLU benchmark（Table 1）：对比Mamba（1.4B/2.8B/7B，从零预训练）、RWKV-5（1.5B/7B）、RetNet（6.7B）等循环模型，以及Transformer baseline（Llama2-7B, Mistral-7B, Gemma等），评测HellaSwag/PIQA/WG/ARC-E/ARC-C/MMLU；(b) 长上下文评测（Table 2）：在SCROLLS benchmark的Qasper（2-shot）和NarrativeQA（0-shot）上，不同context cut-off长度（2048/4096/8192/16384）下对比Llama2、Mistral、RWKV-5、Mamba、RecurrentGemma；(c) 消融实验（Table 3）：对比Mamba/T2R/SUPRA从零训练和从预训练Transformer uptraining，验证归一化策略的关键性（T2R uptraining不稳定导致性能崩溃），以及两阶段微调策略的效果。

- 硬件平台是什么，配置是什么。
  Nvidia H100 GPU集群。根据模型规模使用4到32个节点，每个节点8块GPU。使用PyTorch FSDP（Fully Sharded Data Parallel）进行分布式训练。混合精度策略由OpenLM自动选择（bfloat16和float32混合）。7B参数线性模型uptraining吞吐量约为4300 tokens/秒/GPU。训练精度bfloat16为主（部分操作用float32）。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) Llama2-SUPRA 7B（从Llama2-7B uptraining，+20B tokens，2048 context length）；(2) Mistral-SUPRA 7B（从Mistral-7B uptraining，+20B/+100B tokens，2048 context length）；(3) 消融用1B模型（从1.6T token预训练的Transformer uptraining +10B tokens）；(4) 从头训练的Mamba-7B（在RefinedWeb上训练1.2T tokens作为强baseline）；(5) 从零训练的对比模型：Mamba 1B、SUPRA 1B、T2R 1B、Transformer 1B（均在100B tokens上训练）。训练使用Adam optimizer（β1=0.9, β2=0.95），学习率cosine decay（7B: 3e-5→1e-5, 1B: 3e-4→1e-5），1000步linear warmup，mini-batch 2M tokens，默认RoPE频率10^4（长序列用10^6）。
  数据集：RefinedWeb（Penedo et al., 2023），2个epoch用于Mamba训练，单epoch用于uptraining。使用预训练模型的tokenizer（Llama2或Mistral tokenizer），从零训练时使用GPT-NeoX-20B tokenizer。序列打包（sequence packing），默认序列长度2048。
  Benchmark：(1) 短上下文NLU：使用Eleuther Evaluation Harness（Gao et al., 2023），评测HellaSwag、PIQA、WinoGrande、ARC-Easy、ARC-Challenge（0-shot），MMLU（5-shot）；(2) 长上下文：SCROLLS benchmark（Shaham et al., 2022），具体使用Qasper（2-shot）和NarrativeQA（0-shot），在不同context cut-off长度下评测。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  完全开源（MIT License）。代码：https://github.com/TRI-ML/linear_open_lm（基于OpenLM fork，包含修改后的linear attention函数和Lightning Attention 2的Triton kernel集成），模型权重：Mistral-SUPRA（https://huggingface.co/TRI-ML/mistral-supra）和Mamba-7B（https://huggingface.co/TRI-ML/mamba-7b-rw）。

  SUPRA算法pipeline核心——训练时的线性注意力前向传播（基于Lightning Attention 2 Triton kernel）：
  ```
  # 参数: W_Q, W_K, W_V, W_O ∈ R^{D×D} (原始Transformer的QKV投影)
  #        W ∈ R^{D×D}, b ∈ R^D (MLP kernel的线性层, queries和keys共享)
  #        γ ∈ (0,1)^h (固定衰减向量, h为head数, 借鉴RetNet)
  #        RoPE: rotary positional embedding

  Input: X ∈ R^{B×S×D} (batch, seq_len, hidden_dim)
         use_decay: bool = True
         normalize: bool = False (使用GroupNorm替代, 不在此函数内)

  For each head h in 0..num_heads-1:
    # 1. 原始Q/K/V投影
    q = X @ W_Q  # ∈ R^{B×S×d_h}, d_h = D/num_heads
    k = X @ W_K  # ∈ R^{B×S×d_h}
    v = X @ W_V  # ∈ R^{B×S×d_h}

    # 2. MLP kernel: 用可学习的线性层φ替换ELU非线性
    #    关键: queries和keys共享同一MLP权重 W, b
    phi_q = ReLU(q @ W + b)  # ∈ R^{B×S×d_h}, W共享
    phi_k = ReLU(k @ W + b)  # ∈ R^{B×S×d_h}, W共享

    # 3. RoPE位置编码（应用到kernel输出）
    phi_q_rope = RoPE(phi_q)  # 旋转位置嵌入
    phi_k_rope = RoPE(phi_k)

    # 4. 带decay的线性注意力（Lightning Attention 2高效实现）
    #    decay slope: s[h] ∈ (0,1) 控制该head的衰减速度
    output = lightning_attn_ops(phi_q_rope, phi_k_rope * qk_scale, v, slope_tensor)
    # lightning_attn_ops计算: O_i = Σ_{j=1}^{i} γ^{i-j} (φ(q_i)·φ(k_j)) v_j

    # 5. 输出: GroupNorm per head (替代传统线性注意力的分母除法)
    output = GroupNorm_h(output)  # h个group, 按head独立归一化

  # 6. 拼接所有head输出
  output = concat(output_heads)  # ∈ R^{B×S×D}
  output = output @ W_O  # 最终输出投影
  ```

  论文中提供的实际linear_attn_func代码（Section 7）：
  ```python
  def linear_attn_func(q, k, v, qk_scale: float, use_decay: bool = True,
                       normalize: bool = False) -> torch.Tensor:
      # q, k, v: (batch_size, num_heads, seq_len, dim)
      h = q.shape[1]
      if use_decay:
          s = slope_tensor(h, q.device, q.dtype)  # γ衰减向量
      else:
          s = no_slope_tensor(h, q.device, q.dtype)
      output = lightning_attn_ops(q, k * qk_scale, v, s)  # Triton kernel
      if normalize:
          norm = torch.clamp_min(
              torch.einsum("nhld,nhld->nhl", q, k.cumsum(2) * qk_scale), 1e-7)
          return output / norm.unsqueeze(-1)
      return output
  # 注: 实际SUPRA使用GroupNorm替代normalize分支, 不在此函数内处理
  ```

  推理时的循环（Recurrent）形式——O(1) per-token：
  ```
  # 初始化: s_0 = 0 (KV-state矩阵), z_0 = 0 (归一化向量)
  For each generated token i:
    φ_k_i = ReLU(k_i @ W + b)    # shared MLP kernel for key
    φ_q_i = ReLU(q_i @ W + b)    # shared MLP kernel for query
    φ_k_i = RoPE(φ_k_i)          # 应用RoPE
    φ_q_i = RoPE(φ_q_i)
    # 更新循环状态（等价于在线性attention中的KV累积）
    s_i = diag(γ) · s_{i-1} + φ_k_i · v_i^T   # matrix-valued state update
    # GroupNorm + 读取
    v'_i = GroupNorm(φ_q_i^T · s_i)  # 一个token的输出
  ```

  SUPRA与T2R的关键区别：
  - T2R: sim(q,k) = φ(q)·φ(k) 带分母归一化（sum of sim），类似于softmax的模拟，训练不稳定
  - SUPRA: sim(q,k) = RoPE(φ(q))·RoPE(φ(k)) 带GroupNorm + 固定decay γ，训练稳定，可扩展至7B规模

  训练流程：
  ```
  Step 1: 加载预训练Transformer (Llama2或Mistral)
  Step 2: 添加MLP kernel参数 (W, b, 每层每head共享query/key)
  Step 3: 替换注意力计算: softmax(QK^T/√d)·V → GroupNorm(Σγ^{i-j}·sim(q_i,k_j)·v_j)
  Step 4: 在RefinedWeb上uptraining 20B-100B tokens
          - 训练所有参数 (新增的MLP kernel + 原有Transformer参数)
          - Adam optimizer, cosine LR schedule
          - 1000步warmup
  Step 5: 推理时切换到循环（Recurrent）模式
  ```

  性能摘要（Table 1, 7B scale）：
  - Mistral-SUPRA (+100B): HellaSwag 77.1, PIQA 80.4, WG 70.3, ARC-E 75.9, ARC-C 45.8, MMLU 34.2, Avg 64.0
  - Mamba-7B (从零训练, 1.2T tokens): Avg 64.7
  - RWKV-5-1.7T (1.7T tokens): Avg 63.5
  - Mistral-7B (原始Transformer): Avg 72.4
  - SUPRA仅用5%训练成本，达到从零训练Mamba/RWKV的竞争性性能

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

## An_Empirical_Study_of_Mamba-based_Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是三种基于Mamba的8B参数LLM架构：纯Mamba（56层，hidden dim 4096，state dim 128）、纯Mamba-2（同Mamba架构参数，但使用Mamba-2 block，head dim 64, expansion factor 2, 8 groups, conv window 4）以及Mamba-2-Hybrid（56层：24 Mamba-2 + 4 Self-Attention(GQA, 8 groups, 32 heads, 128 KV-Channels) + 28 MLP层，按Appendix A算法均匀分布）。所有基于SSM的模型均不使用位置编码、不使用Dropout、不使用bias、使用untied embeddings。实验对比这些架构与同参数规模的纯Transformer baseline（32层，4096 hidden dim，32 attention heads，128 KV-channels，SwiGLU activation，RoPE位置编码，LayerNorm），在同一训练数据（1.1T和3.5T tokens）和相同超参数下进行apple-to-apple比较，评测涵盖12个标准短上下文任务、9个自然长上下文任务、13个RULER合成任务和Phonebook复制任务。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 GPU集群。训练配置：tensor parallel size=4，data parallel size=256（共1024 GPUs），micro batch size=4，global batch size=1024（3.5T数据集）或256（1.1T数据集）。训练精度BF16。

- 模型是什么。数据集和bench分别是什么。
  模型：8B参数级别的Transformer、Mamba、Mamba-2、Mamba-2-Hybrid四种架构，及16K/32K/128K长上下文扩展版本。数据集：1.1T和3.5T token数据集，成分70% English + 15% non-English + 15% code（Nemotron-4数据前身），使用SentencePiece 256K词表。Benchmark：LM Evaluation Harness (commit 94cc1850) 评测WinoGrande/PIQA/HellaSwag/ARC-Easy/ARC-Challenge/MMLU/OpenBookQA/TruthfulQA/PubMedQA/RACE/NQ/SquadV2共12项；LongBench (commit 48798083) 评测MultiFieldQA/HotpotQA/2WikiMQA/Musique/TREC/TriviaQA + NarrativeQA/Qasper/QuALITY；RULER评测13个合成任务；Phonebook合成复制任务。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/NVIDIA/Megatron-LM/tree/ssm/examples/mamba（Megatron-LM固定快照），模型权重在Hugging Face发布。
  
  算法pipeline核心——Mamba-2-Hybrid前向传播伪代码：
  ```
  Input: x ∈ R^{B×S×D} (batch, seq_len, hidden_dim=4096)
  
  For each layer l in 0..55:
    x_norm = RMSNorm(x)  // pre-norm
    if layer_type[l] == MAMBA-2:
      // Mamba-2 SSM block (head_dim=64, state_dim=128, n_groups=8, expand=2)
      x_branch = x_norm
      // 1. Input projection: expand to 2*D
      x_proj, z_proj = Linear(x_norm)  // split into two D-dim branches
      // 2. Short convolution (window=4)
      x_conv = CausalConv1d(x_proj)
      // 3. SiLU activation
      x_act = SiLU(x_conv)
      // 4. SSM scan (structured state-space duality, parallel scan)
      //    Discretize continuous SSM: A ∈ R^{D×state_dim}, B ∈ R^{D×state_dim}, C ∈ R^{D×state_dim}
      //    Δ = softplus(Linear(x_act) + bias)
      //    A_bar, B_bar = discretize(A, B, Δ)  // zero-order hold
      //    h_t = A_bar * h_{t-1} + B_bar * x_act[t]  // recurrent state update
      //    y[t] = C * h_t
      y = selective_scan(x_act, Δ, A, B, C)
      // 5. Gating with z branch
      y = y * SiLU(z_proj)
      // 6. Output projection back to D
      x = x + Linear_out(y)  // residual
    elif layer_type[l] == ATTENTION:
      // Group-Query Attention (8 KV groups, 32 Q heads, 128 KV-ch)
      Q, K, V = Linear(x_norm)  // project to Q, K, V
      attn_out = GQA(Q, K, V)  // no RoPE position encoding
      x = x + Linear_out(attn_out)  // residual
    elif layer_type[l] == MLP:
      // MLP with GELU, 4x expansion
      h = GELU(Linear_1(x_norm))
      x = x + Linear_2(h)  // residual
  
  // Final output
  logits = Linear_lm_head(x)  // untied embedding weights
  ```
  
  Mamba-2 tensor parallelism关键差异：Mamba-2每层仅需1次all-reduce（vs Mamba的2次），但需使用GroupNorm（groups=8, group_size=512 > 256）替代LayerNorm作为内部归一化。Hybrid模型的层分配遵循Appendix A Algorithm 1：先均匀散布self-attention层，再在非attention层的Mamba层中均匀替换为MLP层，确保首层为Mamba层。

## Associative_Recurrent_Memory_Transformer

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是ARMT（Associative Recurrent Memory Transformer），在RMT（Recurrent Memory Transformer）的segment-level recurrence基础上，每层添加基于quasi-linear key-value memory（delta-rule, 源自Schlag et al. 2021）的层间关联记忆（associative memory）。每个segment的memory token通过线性映射生成key/value，经DPFP-3非线性变换后存入关联矩阵A_s^l，同时用γ-correction更新归一化向量z_s^l防止灾难性遗忘。实验对比：(1) BABILong benchmark上QA1-QA5任务（最高50M tokens），对比Mamba (130M)、RMT (137M)、RMT-R、GPT-4 (few-shot)、GPT-4+RAG；(2) Associative Retrieval Remember和Rewrite任务（最多200 key-value pairs），对比Mamba和RMT，评估记忆容量和动态改写能力；(3) Language Modeling (Wikitext-103)对比RMT；(4) 消融实验包括γ-correction有无和PRMT（Parallel Memory RMT，仅层间记忆无关联矩阵）对比。

- 硬件平台是什么，配置是什么。
  论文未明确说明具体GPU型号和配置。致谢中提到"SberDevices for granting us access to additional computational resources"。基于训练设置（GPT-2 137M backbone、最大16k tokens训练）推测使用商用NVIDIA GPU集群（如A100或类似）。训练精度论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) BABILong实验：GPT-2 (137M) + ARMT/RMT扩展，segment size=512，总参数145M（ARMT）、137M（RMT）；对比Mamba-130M。训练最大长度16k tokens（32 segments × 512）。(2) Associative Retrieval实验：小模型约500k参数，4层，hidden dim=128，memory dim=32。训练最大200 pairs (Remember) / 50 pairs (Rewrite)。(3) LM实验：GPT-2 + ARMT/RMT，segment size=128，训练8 segments（1024 tokens）。
  数据集：(1) BABILong benchmark — 合成QA任务，从单事实(QA1)到三参数关系(QA5)，噪声句子+事实混合，支持生成1k-50M tokens序列；(2) Associative Retrieval — 自建数据集：Remember任务（唯一key-value对，评估记忆容量）、Rewrite任务（非唯一key，评估动态更新能力）；(3) Wikitext-103（语言建模评估）。(4) 训练数据使用curriculum learning：BABILong从2 segments递增至32 segments（16k tokens），Associative Retrieval从1对递增至200对。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/RodkinIvan/associative-recurrent-memory-transformer（Apache-2.0 license），基于PyTorch + Hugging Face Transformers + Accelerate，包含modeling_amt.py和modeling_rmt.py核心模块。

  ARMT算法pipeline核心——逐层前向传播（输入一个segment）：
  ```
  Input: X_s^l ∈ R^{seg_len×D} (segment s 在层l的hidden states)
         M_s^l ∈ R^{mem_tokens×D} (层l的memory tokens)
         A_{s-1}^l ∈ R^{D×D}, z_{s-1}^l ∈ R^D (上一层segment的关联记忆和归一化向量)
  
  For each layer l:
    // 1. 拼接输入和memory tokens
    input_concat = concat([X_s^l; M_s^l])  // ∈ R^{(seg_len+mem_tokens)×D}
  
    // 2. 从关联记忆中读取（类似线性注意力）
    for each token x_j in input_concat:
      q_j = W_Q x_j                    // ∈ R^D
      y_j = A_s^l φ(q_j) / (z_s^l)^T φ(q_j)  // 关联回忆, φ=DPFP-3非线性
    // y_j加到token表示中
  
    // 3. Transformer block处理（包含self-attention + FFN）
    [X_s^{l+1}; M_s^{l+1}] = TransformerBlock(input_concat + y)
    // self-attention仅在当前segment内（local context），不需要attend历史token
  
    // 4. 更新关联记忆（用新产生的memory tokens）
    for each memory token m_i ∈ M_s^{l+1}:
      k_i = W_K m_i                   // key投影
      v_i = W_V m_i                   // value投影
      β_i = σ(W_β m_i)                // importance scalar（sigmoid门控）
      
      // γ-correction: 从归一化向量中移除旧key的贡献
      v̄_i = A_{s-1}^l φ(k_i) / (z_{s-1}^l)^T φ(k_i)  // 回忆旧value
      γ_i = 1 - (z_{s-1}^l)^T φ(k_i) / ||φ(k_i)||²     // 归一化修正系数
      
      // Delta-rule更新关联矩阵
      A_s^l = A_{s-1}^l + Σ_i β_i (v_i - v̄_i) ⊗ φ(k_i)  // 外积更新, D×D
      z_s^l = z_{s-1}^l + Σ_i γ_i φ(k_i)                // 归一化向量更新
  
  Output: X_s^{l+1} (用于下一层或输出), M_s^{l+1} (传给下一segment同层), A_s^l, z_s^l (传给下一segment同层)
  ```

  复杂度分析（per segment, per layer）：
  - Local self-attention: O(seg_len² × D) — 仅segment内，与总序列长度无关
  - Associative memory read: O((seg_len+mem_tokens) × D²) — D×D矩阵-向量乘
  - Associative memory update: O(mem_tokens × D²) — 外积更新
  - 总空间: O(D²) per layer 存储固定大小A矩阵，与序列长度无关
  - 每segment处理时间和空间均为常量O(1)，与总序列长度无关

  训练配置：
  - Curriculum learning: 从短序列开始逐步增加segment数量
  - BABILong: 训练最大32 segments × 512 tokens = 16k tokens
  - Associative Retrieval: 训练最大200 pairs (Remember) / 50 pairs (Rewrite)
  - LM: 训练8 segments × 128 tokens = 1024 tokens
  - γ-correction的γ在训练时detach以改善收敛

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Artificial Hippocampus Networks (AHNs) —— 一种轻量级架构组件，在标准Transformer的每层attention旁添加RNN-like压缩模块。具体而言：模型保留滑动窗口attention（默认窗口W=32k）作为lossless short-term memory；当序列长度超过W时，AHN对离开窗口的KV pair (k_{t-W}, v_{t-W}) 进行recurrent压缩，更新固定大小的compressed long-term memory h_{t-W} = AHN((k_{t-W}, v_{t-W}), h_{t-W-1})。最终输出为attention输出与AHN压缩记忆输出的加和：y_t = y_AHN,t + Attention({(k_i,v_i)}_{i=t-W+1}^t, q_t)。AHN有三种实例化：AHN-Mamba2（基于Mamba2 SSM）、AHN-DN（基于DeltaNet的delta rule更新）、AHN-GDN（基于GatedDeltaNet的gated delta rule）。训练采用self-distillation：冻结base LLM全部参数，仅训练AHN参数（约0.4%参数量），以KL(p_teacher || p_student)为loss。实验在Qwen2.5-Instruct系列（3B/7B/14B）上对比：Full Attention、Sinks+SWA、Compressive Transformer (CT-Max/CT-Average pooling, 4x压缩率)、AHN-Mamba2/AHN-DN/AHN-GDN。评测包括LV-Eval (128k)、InfiniteBench (128k)、LongBench (6个>8k任务)、PG19 perplexity、RULER NIAH任务。

- 硬件平台是什么，配置是什么。
  训练：32块NVIDIA A100 GPU，训练7B模型约10小时。训练精度论文未明确说明（推测BF16/FP16混合精度）。推理评测：论文未明确说明推理硬件（推测同一A100集群）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-Instruct系列（3B、7B、14B），在其每层attention上添加AHN模块。AHN instantiation参数：每head的W_α∈R^{D×1}、W_β∈R^{D×1}、W_γ∈R^{D×1}（gating参数）和W_o∈R^{H×H}（per-head输出投影），总计约0.4% base model参数量。数据集：ChatQA2（1B tokens，开源长上下文任务集合），训练时最大序列长度24k，随机化滑动窗口大小（从[32,64,...,8192]中采样过滤后候选）。Benchmark：LV-Eval（128k context, 11 datasets）、InfiniteBench（128k context）、LongBench（6个平均长度>8k的任务：DuReader, HotpotQA, MuSiQue, NarrativeQA, QMSum, TriviaQA）、PG19（perplexity评测）、RULER（needle-in-a-haystack评测）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/ByteDance-Seed/AHN，模型：https://huggingface.co/ByteDance-Seed。实现基于PyTorch、LLaMA-Factory、Flash Linear Attention。

  AHN-GDN算法pipeline（per head, per layer）：
  ```
  Input: x_t ∈ R^D (当前token hidden state), QKV from attention projections
         KV cache window: {(k_i, v_i)}_{i=t-W+1}^t
         h_{t-W-1} ∈ R^{H×H} (上一步压缩记忆矩阵)

  // Step 1: 从attention的KV projection获取当前离开窗口的KV pair
  k_{t-W}, v_{t-W} ∈ R^H  // 第t-W位置的key和value, H为head_dim

  // Step 2: AHN-GDN 压缩记忆更新（Gated Delta Rule）
  α(x_{t-W}) = x_{t-W} W_α   // W_α ∈ R^{D×1}, α: scalar per head
  β(x_{t-W}) = x_{t-W} W_β   // W_β ∈ R^{D×1}, β: scalar per head

  h_{t-W} = α(x_{t-W}) * (I - β(x_{t-W}) * k_{t-W}^T k_{t-W}) * h_{t-W-1}
          + β(x_{t-W}) * k_{t-W}^T v_{t-W}
  // 即：记忆衰减 + 新信息写入，h ∈ R^{H×H}

  // Step 3: Query访问压缩记忆
  q_t = slice from attention's Q projection  // q_t ∈ R^H
  γ(x_t) = x_t W_γ   // W_γ ∈ R^{D×1}, gate scalar
  y_AHN,t = γ(x_t) * q_t * h_{t-W} * W_o   // W_o ∈ R^{H×H}, grouped by heads

  // Step 4: 与窗口attention输出求和
  y_t = y_AHN,t + Softmax(q_t {k_i}_{i=t-W+1}^T / √H) {v_i}_{i=t-W+1}

  复杂度（vs Full Attention）:
  - Memory cache: O(W) vs O(L), W=32k固定
  - FLOPs per token: O(W) vs O(L)
  - 当L ≤ W时，AHN不激活，模型等同于标准Transformer
  ```

  训练流程（Self-Distillation）：
  ```
  Teacher: 原始Qwen2.5-Instruct（Full Attention, 参数冻结）
  Student: Qwen2.5-Instruct + AHN（window attention + AHN, 仅AHN参数可训练）

  For each training step:
    x_teacher = x_student = input_sequence (max 24k tokens)
    p_teacher = Teacher(x_teacher)    // full attention forward
    p_student = Student(x_student)    // window attn + AHN forward
    loss = KL(p_teacher || p_student)
    loss.backward()  // 仅AHN参数有梯度

  超参数：
    - Optimizer: AdamW, LR=1e-4, linear warmup 10% steps + cosine decay
    - Batch size: 128 (global), 740 update steps (1 epoch over 1B tokens)
    - 训练窗口随机化：attention sink size ∈ [0,32,64,128,512,2048,4096]
      总lossless memory (sinks + window) ∈ [32,64,...,8192]
    - 推理默认：128 attention sinks + 32640 sliding window = 32768 lossless memory
  ```

## Eagle_and_Finch__RWKV_with_Matrix-Valued_States_and_Dynamic_Recurrence

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Eagle（RWKV-5）和Finch（RWKV-6）两种新型RNN架构，沿袭RWKV-4的线性注意力思路。核心创新：(1) Eagle引入multi-headed matrix-valued states（WKV state从向量d扩展为矩阵(D/h)×(D/h)），用LayerNorm（等效GroupNorm）替代分母归一化，添加SiLU attention gating，移除receptance的Sigmoid激活，改进参数初始化；(2) Finch在Eagle基础上引入data-dependent dynamic recurrence：Token Shift从静态lerp升级为ddlerp（data-dependent linear interpolation），通过LoRA（λ + tanh(xA)B）使token shift量依赖输入内容；decay rate w从固定learned vector变为时间可变的w_t = exp(-exp(d_t))，其中d_t也由LoRA生成。同时引入新tokenizer（RWKV World Tokenizer, V=65536, Trie-based greedy matching）和新数据集（RWKV World v2, 1.12T tokens, 70% English + 15% multilingual + 15% code）。
  实验对比：(a) multilingual benchmarks (LAMBADA Multilingual, XCOPA, XNLI, PAWS-X, XStoryCloze, xWinogrande) 和 English benchmarks (LAMBADA, HellaSwag, PIQA, ARC-Easy, ARC-Challenge, GLUE, WinoGrande, SciQ, COPA) 对比Pythia/Mamba/RWKV-4/Llama-2/Falcon/Mistral/MPT等；(b) MQAR合成任务对比RWKV-4和Mamba；(c) PG19长上下文loss vs position对比RWKV-4；(d) Bamboo长上下文推理benchmark；(e) 速度和内存benchmark对比Mamba和Flash Attention (A100 80GB)；(f) 多模态：VisualRWKV (GQA, ScienceQA-IMG, Text-VQA, POPE)、Music modelling、AudioRWKV (AudioSet mAP)；(g) 架构消融（170M参数在Pile上训练330B tokens对比Mamba/RWKV-4/Pythia）；(h) DDLerp消融；(i) AlignBench中文对齐、MTBench、Self-Learning Capability、零样本推理。

- 硬件平台是什么，配置是什么。
  训练：NVIDIA A100 80GB GPU（Eagle 0.4B: 24 GPUs, 1.5B/3B: 48 GPUs; Finch 1.6B/3B同配置），Eagle 7B使用64×H800 GPU。训练精度bfloat16（WKV计算用float32保证数值稳定性）。优化器Adam（β1=0.9, β2=0.99, weight decay=0.001仅用于linear和embedding）。pretraining context length=4096。学习率：linear 10-step warmup (20%→100%) + cosine decay。详细超参数见表17（max LR: 0.4B=4e-4, 1.5B=3e-4, 3B=2e-4, 7B=1.5e-4; micro batch size: 0.4B/1.5B=8, 3B=4, 7B=9; global batch size: 7B=2359296）。速度benchmark: 单张A100 80GB, batch size=8, model dim=4096, head size=64。

- 模型是什么。数据集和bench分别是什么。
  模型：Eagle (RWKV-5) 0.4B (L24/D1024), 1.5B (L24/D2048), 3B (L32/D2560), 7B (L32/D4096); Finch (RWKV-6) 1.6B (L24/D2048), 3B (L32/D2560)。参数公式：#Params_E = 13D²L+14DL+4D+2DV, #Params_F = 13D²L+464DL+4D+2DV (V=65536)。Head dim恒为64，h=D/64。Finch的LoRA矩阵: A∈R^{D×32}, B∈R^{32×D} (A_ω∈R^{D×64}, B_ω∈R^{64×D})。内部状态大小: L(2D+D²/h)=66DL（约5×DL RWKV-4的5.2倍）。Channel Mixing hidden dim从4D缩减至3.5D（Eagle）以保持等参数关系。
  训练数据集：RWKV World v2（1.12T tokens，70% English + 15% multilingual + 15% code），组件包括Wikipedia/SlimPajama/peS2o/BigPatent/Pile of Law/StarCoder/OSCAR23.01/TED2020/PhilPapers/Books3/Gutenberg/OpenSubtitles等（见表9）。Benchmark：(1) LM Evaluation Harness: 多语言(LAMBADA-M, XCOPA, XNLI, PAWS-X, xStoryCloze, xWinogrande) + 英语(LAMBADA, HellaSwag, PIQA, ARC-E/C, GLUE, WinoGrande, SciQ, COPA, OpenBookQA, HeadQA, ReCoRD)；(2) MQAR合成；(3) PG19测试集长上下文；(4) Bamboo长上下文推理（9/10任务有效）；(5) VisualRWKV: GQA, ScienceQA-IMG, Text-VQA, POPE；(6) AUDIOSET mAP；(7) AlignBench, MTBench；(8) 自学习能力(SLC Score); (9) 零样本（Aggression, MathQA, Sarcasm等13个数据集）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  完全开源（Apache 2.0）。训练代码：https://github.com/RWKV/RWKV-LM，推理代码：https://github.com/RWKV/ChatRWKV，time-parallel训练：https://github.com/RWKV/RWKV-infctx-trainer，模型权重：https://huggingface.co/RWKV。

  Eagle (RWKV-5) Time Mixing算法pipeline (single head, recurrent form):
  ```
  # 参数: W_r, W_k, W_v, W_g ∈ R^{D/h × D/h} (投影矩阵)
  #        μ_r, μ_k, μ_v, μ_g ∈ R^{D/h} (learnable token-shift系数)
  #        ω ∈ R^{D/h} (trainable log-decay), u ∈ R^{D/h} (boost)
  #        s_t ∈ R^{(D/h)×(D/h)} (matrix-valued recurrent state)
  #        W_o ∈ R^{(D/h)·h × D} (output projection)

  Input: x_t ∈ R^D (当前token embedding)
         x_{t-1} (前一token, 用于token-shift)
         s_{t-1} ∈ R^{(D/h)×(D/h)} (上一时刻的WKV state)

  # Step 1: Token Shift (per-head lerp)
  r_t = (x_t + (x_{t-1} - x_t) ⊙ μ_r) @ W_r      # ⊙ element-wise
  k_t = (x_t + (x_{t-1} - x_t) ⊙ μ_k) @ W_k
  v_t = (x_t + (x_{t-1} - x_t) ⊙ μ_v) @ W_v
  g_t = (x_t + (x_{t-1} - x_t) ⊙ μ_g) @ W_g
  # r_t, k_t, v_t, g_t ∈ R^{D/h}

  # Step 2: Decay rate (固定, per-channel)
  w = exp(-exp(ω))  # w ∈ (0,1)^{D/h}, contraction matrix

  # Step 3: WKV computation (加权key-value矩阵)
  #   当前token (特殊处理, boost u):
  wkv_cur = diag(u) · k_t^T · v_t     # ∈ R^{(D/h)×(D/h)}
  #   历史state:
  wkv = wkv_cur + s_{t-1}

  # Step 4: 更新state (decay + accumulate)
  s_t = diag(w) · s_{t-1} + k_t^T · v_t

  # Step 5: Receptance + LayerNorm + SiLU gating + Output
  o_t = LayerNorm(r_t @ wkv)          # ∈ R^{D/h}, LN per head (=GroupNorm, h groups)
  o_t = concat(SiLU(g_t) ⊙ o_t)      # 拼接所有head
  output = o_t @ W_o                  # ∈ R^D

  Return: output, s_t, x_t (for next time step)
  ```

  Finch (RWKV-6) Time Mixing算法pipeline (single head, recurrent form):
  ```
  # 额外参数: A_□ ∈ R^{D×32}, B_□ ∈ R^{32×D} for □∈{r,k,v,g,d}
  #           (decay的A_ω∈R^{D×64}, B_ω∈R^{64×D} 加倍)
  #           μ_x ∈ R^D (ddlerp内的token-shift系数)
  #           λ_□ ∈ R^D (LoRA bias)

  Input: x_t, x_{t-1}, s_{t-1} (同Eagle)

  # Step 1: Data-Dependent Token Shift (ddlerp)
  #   LoRA对token差值进行数据依赖调制:
  lora_□(x) = λ_□ + tanh(x @ A_□) @ B_□   # ∈ R^{D/h}
  ddlerp_□(a,b) = a + (b-a) ⊙ lora_□(a + (b-a) ⊙ μ_x)

  r_t = ddlerp_r(x_t, x_{t-1}) @ W_r
  k_t = ddlerp_k(x_t, x_{t-1}) @ W_k
  v_t = ddlerp_v(x_t, x_{t-1}) @ W_v
  g_t = ddlerp_g(x_t, x_{t-1}) @ W_g

  # Step 2: Data-Dependent Time-Varying Decay
  d_t = lora_d(ddlerp_d(x_t, x_{t-1}))    # ∈ R^{D/h}
  w_t = exp(-exp(d_t))                    # 时间可变decay, ∈ (0,1)^{D/h}

  # Step 3: WKV computation
  wkv_cur = diag(u) · k_t^T · v_t
  wkv = wkv_cur + s_{t-1}

  # Step 4: 更新state (data-dependent decay)
  s_t = diag(w_t) · s_{t-1} + k_t^T · v_t

  # Step 5: Output (同Eagle)
  o_t = LayerNorm(r_t @ wkv)
  output = concat(SiLU(g_t) ⊙ o_t) @ W_o

  Return: output, s_t, x_t
  ```

  Channel Mixing（Eagle和Finch共享, 同RWKV-4但hidden dim减至3.5D）:
  ```
  r'_t = lerp_{r'}(x'_t, x'_{t-1}) @ W_{r'}     # ∈ R^D
  k'_t = lerp_{k'}(x'_t, x'_{t-1}) @ W_{k'}     # ∈ R^{3.5D}
  v'_t = ReLU(k'_t)^2 @ W_{v'}                   # squared ReLU激活
  o'_t = σ(r'_t) ⊙ v'_t                          # sigmoid gating

  # Finch中token-shift也升级为ddlerp:
  r'_t = ddlerp_{r'}(x'_t, x'_{t-1}) @ W_{r'}
  k'_t = ddlerp_{k'}(x'_t, x'_{t-1}) @ W_{k'}
  ```

  训练时并行化：WKV计算在时间维度可通过associative scan或FlashAttention类技术并行化。论文选择沿非时间维度并行，使用custom CUDA kernel将state操作保持在SRAM中。另外提供纯PyTorch time-parallel实现（基于GLA方法）。

  关键设计差异RWKV-4→5→6：
  - RWKV-4: 向量state s∈R^D, head size=1, scalar decay, Sigmoid receptance, 有分母归一化
  - Eagle: 矩阵state s∈R^{(D/h)×(D/h)}, head size=64, per-channel decay w, LayerNorm替代分母, SiLU gating, 无Sigmoid receptance
  - Finch: 矩阵state + data-dependent w_t和ddlerp token-shift

  推理效率：O(1) per-token time, O(D²/h) memory per layer for state。Finch训练时16k序列比Flash Attention快4.2×，比Mamba省17%内存、比Flash Attention省40%内存（A100 80GB, batch=8, D=4096, head=64）。

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

## Just_read_twice__closing_the_recall_gap_for_recurrent_language_models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现包含两部分：(1) JRT-Prompt——极简的prompting策略，将context在prompt中重复多次，使模型看到所有数据顺序，从而绕过causal模型对输入顺序的依赖。例如Î = A(C, Q, C, Q)后再生成答案；(2) JRT-RNN——受Prefix-LM启发的encoder-decoder循环架构，使用非因果的Prefix Linear Attention (PLA)处理prompt前缀（encoder区域），再因果解码输出。JRT-RNN使用两套独立的key/value投影(k_e,v_e用于encoder, k_d,v_d用于decoder)，结合NTP+MLM联合训练目标。基于Based架构（混合gated convolution + sliding window attention + linear attention），将linear attention层替换为JRT-RNN的PLA层。
  实验比较：(a) JRT-Prompt: 16个off-the-shelf循环LM（Based 1.3B, Mamba 130M/370M/1.4B/2.8B, Mamba-2 130M/370M/1.3B/2.7B, GLA 1.3B/2.7B）在6个recall-intensive ICL任务（FDA, SWDE, NQ, SQUAD, TriviaQA, Drop）上对比default vs JRT-Prompt zero-shot prompting，也对比Transformer++；(b) JRT-RNN: 360M/30B和1.3B/10B/50B参数训练，对比Based、Mamba、Transformer++在同token量下的ICL质量和SuperGLUE通用语言理解；(c) 合成SD任务: causal vs non-causal Based变体，不同state size下数据顺序敏感性验证；(d) Pile perplexity slicing (AR vs Other slices)；(e) 推理吞吐量prefill benchmark对比FlashAttention-2和多种线性注意力实现。

- 硬件平台是什么，配置是什么。
  训练：NVIDIA A100-80GB GPU集群。JRT-RNN训练基于FlashAttention代码库（https://github.com/Dao-AILab/flash-attention/tree/main），使用NVidia A100-80GB。推理吞吐量benchmark：单张NVIDIA H100 GPU，prefill latency测量sequence length 2048-32768和batch size 2-64，取20次迭代平均。合成任务Based模型：论文未具体说明GPU型号（推测商用NVIDIA GPU）。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) JRT-Prompt评测：Based 1.3B (10B/50B Pile tokens)、Mamba 130M/370M/1.4B/2.8B (300B Pile tokens)、Mamba-2 130M/370M/1.3B/2.7B (300B Pile tokens)、GLA 1.3B/2.7B (100B SlimPajama tokens)、Transformer++ 1.3B (10B/50B Pile tokens)；(2) JRT-RNN训练：360M参数 (30B tokens) 和1.3B参数 (10B/50B tokens)，encoder区域长度M=1024，总序列长度N=2048。Based架构混合了gated convolution (kernel size=3)、sliding window attention (window=128)、linear attention (Taylor feature map, feature dim=16, 2nd order approximation)。JRT-RNN的PLA层feature map使用2阶Taylor近似: φ(q)^Tφ(k) = 1 + q^T k + (q^T k)²/2；(3) 合成SD任务：Based架构4层，交替gated convolution和linear attention，无位置编码。
  数据集：Pile（预训练，GPT2BPETokenizer tokenize，所有模型相同数据顺序）。Benchmark：(a) Recall-intensive ICL: FDA (FDA报告信息抽取，1102 examples/avg 1999.9 tokens)、SWDE (HTML网页信息抽取，1111/1036.1)、SQUADv2 (文档QA，2984/151.9)、Natural Questions (3157/8857.7)、TriviaQA (1698/310.1)、Drop (2084/236.6)，所有用cloze completion格式（Llama-3-70B改写）；(b) SuperGLUE: BoolQ, CB, COPA, MultiRC, ReCoRD, RTE, WiC, WSC；(c) Pile test set perplexity slicing；(d) 合成Set Disjointness任务（|V|=2048）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/HazyResearch/prefix-linear-attention，模型权重在HuggingFace发布。JRT-Prompt评测使用LM-Eval Harness，基于HuggingFace上的开源模型权重。JRT-RNN训练代码集成在Based实现中（https://github.com/HazyResearch/based）。

  **JRT-RNN Prefix Linear Attention (PLA) 算法pipeline**:

  ```
  Input: u ∈ R^{N×d} (输入序列, N=2048, M=1024为encoder区域)
         特征图 φ: R^d → R^{d̃} (Taylor 2nd-order: φ(q)^T φ(k) = 1 + q^T k + (q^T k)²/2)

  // Encoder投影 (前M个token)
  k_e = φ(W_{ke} · u_{1:M})  // encoder key, ∈ R^{M×d̃}
  v_e = W_{ve} · u_{1:M}     // encoder value, ∈ R^{M×d}

  // Decoder投影 (全部N个token, 后N-M个causal)
  q_d = φ(W_{qd} · u)       // decoder query, ∈ R^{N×d̃}
  k_d = φ(W_{kd} · u)       // decoder key, ∈ R^{N×d̃}
  v_d = W_{vd} · u          // decoder value, ∈ R^{N×d}

  // Prefix Linear Attention (Eq.3)
  For i = 1 to N:
    num_i = q_d[i] · (Σ_{j=1}^{i} k_d[j]^T v_d[j] + Σ_{j=1}^{M} k_e[j]^T v_e[j])
    den_i = q_d[i] · (Σ_{j=1}^{i} k_d[j] + Σ_{j=1}^{M} k_e[j])
    y_i = num_i / den_i

  // Recurrent view (推理):
  // Prefill: 并行计算encoder KV-state和decoder初始state
  s_M = Σ_{j=1}^{M} (k_e[j]^T v_e[j] + k_d[j]^T v_d[j])  // KV-state ∈ R^{d×d̃}
  z_M = Σ_{j=1}^{M} (k_e[j] + k_d[j])                    // K-state ∈ R^{d̃}

  // Decoding (i > M, 每token O(1)):
  s_i = s_{i-1} + k_d[i]^T v_d[i]
  z_i = z_{i-1} + k_d[i]
  y_i = (q_d[i] · s_i) / (q_d[i] · z_i)
  ```

  **训练目标 (Eq.5)**:
  ```
  L = (w1 * L_NTP + w2 * L_MLM) / (w1 + w2)
  // L_NTP: 标准next token prediction loss, 仅计算causal区域 {u_M..u_N}
  // L_MLM: masked language modeling loss, 随机mask encoder区域比例P的token
  // 推理时不使用[MASK] token
  ```

  **JRT-Prompt策略**:
  ```
  // 标准ICL: Ŷ = A(C, Q)
  // JRT-Prompt: Ŷ = A(C, Q, C, Q)
  // 第二轮时模型以完整context view决定存储什么信息
  ```

  JRT-RNN与标准decoder-only循环LM的关键差异：
  - 使用两套独立KV投影（encoder和decoder分开），而非Prefix-LM的单套共享投影
  - 训练时encoder区域额外计算MLM loss（比例P mask），Prefix-LM仅计算NTP loss
  - Encoder区域使用非因果sum（非causal cumsum），让encoder区域token互相可见
  - prefill阶段的recurrent state初始化包含encoder的非因果KV-state (Eq.4)
  - 解码步骤与标准causal linear attention完全相同 (Eq.2)，无额外开销

  JRT-Prompt效率分析：虽将context长度翻倍(2N vs N)，但使用sub-quadratic循环架构仍渐进优于Transformer的O(N²)。N=32768, batch=16, H100上Based+JRT-Prompt提供11.9×于FlashAttention-2的prefill吞吐量。

## LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是LongMamba——一种training-free技术，通过扩大Mamba模型中全局通道（global channels）的感受野来增强长上下文能力。核心创新分两步：(1) Channel Classification：基于训练长度上的累积衰减 ∏_{k=1}^L Ā_k 将隐藏状态通道分类为全局通道和局部通道——若累积衰减 > θ 则归类为全局通道；(2) Receptive Field Enlargement via Token Filtering：对于识别的全局通道，过滤掉Δ_t低于阈值g(S)的token（不更新也不衰减隐藏状态），使目标序列长度上的累积衰减与训练长度对齐：∏_{i=1}^S Ā'_i ≈ ∏_{i=1}^L Ā_i。实现时从训练集（Pile）采样序列标定Δ_t分布，预先构建per-channel查找表g(S)，以1000-token为间隔离线计算。Token过滤规则：若Δ_t < g，设置(Ā'_t, B̄'_t) = (1, 0)使H_t = H_{t-1}（跳过该token）；否则正常更新。
  实验比较：(a) Language Modeling：PG-19数据集上评测perplexity，Mamba-1.4B、Mamba2-1.3B、Zamba2-1.2B，序列长度最高60k tokens；(b) RULER合成数据集：13个长上下文任务，16k/24k/32k序列长度，对比vanilla模型；(c) LongBench-E：13个真实世界长上下文应用（Single-doc QA、Multi-doc QA、Summary、Few-shot、Coding等），对比vanilla模型和DeciMamba；(d) 消融研究：不同channel selection阈值θ影响（表3），不同标定序列组影响（表4，STD 0.42%）；(e) LongBench：Falcon-Mamba-7B上对比Llama2-7B-chat-4k、XGen-7B-8k、Vicuna-v1.5-7B-16k Transformer baselines；(f) 延迟开销：A5000和A100 GPU上prefilling延迟测量，4k-96k tokens。

- 硬件平台是什么，配置是什么。
  NVIDIA A5000和NVIDIA A100 GPU。延迟测量使用batch size=1的prefilling场景，评测4k到96k tokens序列长度。训练相关：使用预训练模型的官方checkpoint直接加载，不进行任何微调或参数调整。标定过程随机采样5个来自Pile数据集的序列。论文未明确说明训练原始模型所用的GPU型号（模型为开源预训练模型：Mamba-1.4B、Mamba2-1.3B、Zamba2-1.2B、Falcon-Mamba-7B）。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) Mamba-1.4B（Gu & Dao, 2023），训练长度2k tokens；(2) Mamba2-1.3B（Dao & Gu, 2024a），训练长度2k tokens；(3) Zamba2-1.2B（Glorioso et al., 2024a），混合Transformer-SSM模型，训练长度4k tokens；(4) Falcon-Mamba-7B（Zuo et al., 2024），8B参数纯SSM。所有实验直接加载官方模型checkpoint，无微调。
  数据集：PG-19（语言建模perplexity评测）、RULER（13个合成长上下文任务，包括passkey retrieval、question answering等，每任务/每长度生成100条序列）、LongBench-E（13个真实世界长上下文任务：Passage Count, PassageRetrieval-en, GovReport, MultiNews, MultiFieldQA-en, Qasper, 2WikiMQA, HotpotQA, SAMSum, TREC, TriviaQA, LCC, RepoBench-P）、LongBench（更多任务额外评测）。标定数据集：Pile（Gao et al., 2020）。
  Benchmark：PG-19 perplexity评测；RULER各任务accuracy；LongBench-E各任务accuracy（按Single-doc QA、Multi-doc QA、Summary、Few-shot、Synthetic、Coding类别分组）；LongBench各任务accuracy。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  完全开源。代码：https://github.com/GATECH-EIC/LongMamba。基于PyTorch实现，直接加载HuggingFace上的预训练Mamba/Mamba2/Zamba2模型。

  LongMamba算法pipeline核心——推理时的两阶段前向传播（对每一层Mamba block）：
  ```
  # Phase 0: 离线标定阶段（仅运行一次）
  Input: 从Pile数据集随机采样5条序列（各训练长度L）
         候选clamping值C ∈ {0,5,10,15,20}
         候选阈值θ ∈ {10^-40, 10^-30, ..., 10^-1, 5×10^-1}

  For each 通道 c in d_e (hidden state channels):
    # 1. 计算训练长度L上的累积衰减（Eq.12）
    decay_c = ∏_{k=1}^L Ā_k[c]  # ∈ (0,1), 沿d_s维度取平均

    # 2. 通过grid search在LongBench-E上选最优θ确定全局/局部分类
    if decay_c > θ:
      标记c为全局通道(global channel)
    else:
      标记c为局部通道(local channel)

    # 3. 对每个全局通道标定Δ_t分布
    For each timestep t in 采样序列:
      记录 Δ_t[c] 值
    # Clamp极值到top C%最大值的边界
    clamp_threshold = percentile(Δ_values, 100 - C)
    Δ_t_clamped[c] = min(Δ_t[c], clamp_threshold)

    # 4. 构建per-channel查找表g_c(S)以1000-token为间隔
    For S = 1000, 2000, 3000, ..., max_context:
      找到g使得: ∏_{i=1}^S Ā'_i(g)[c] ≈ ∏_{i=1}^L Ā_i[c]
      # 在假定的Δ_t分布下数值求解g
      g_c[S] = optimal_g
  ```

  ```
  # Phase 1: 推理时修改Mamba SSM前向传播
  # 原Mamba计算（Eq.4-5）:
  #   H_t = Ā_t ⊙ H_{t-1} + B̄_t ⊙ X_t
  #   Y_t = C_t^T H_t
  # LongMamba修改（仅对全局通道）:

  Input: 输入序列X ∈ R^{S×d_e}, S > L (训练长度)
         每个通道c的全局/局部标签
         查找表 g_c(S)  (S向下取整至最近1000-token间隔)

  For each Mamba layer l:
    # 标准Mamba预处理
    X_input = σ(Conv1D(Linear_1(I)))  # ∈ R^{S×d_e}
    Δ, B, C = compute_from(X_input)   # 标准Mamba投影 (Eq.6)
    Ā = exp(Δ ⊙ A)                    # 标准decay计算 (Eq.7)
    B̄ = Δ ⊗ B                         # 标准输入投影 (Eq.7)

    For each timestep t = 1..S:
      For each 通道 c in d_e:
        if 通道c是全局通道:
          # Token filtering (Eq.14)
          if Δ_t[c] < g_c(S):
            Ā'_t[c] = 1               # 不衰减
            B̄'_t[c] = 0               # 不更新
          else:
            Ā'_t[c] = Ā_t[c]           # 正常衰减
            B̄'_t[c] = B̄_t[c]           # 正常更新
        else:  # 局部通道
          Ā'_t[c] = Ā_t[c]             # 保持原值
          B̄'_t[c] = B̄_t[c]             # 保持原值

      # 修改后的隐藏状态更新 (Eq.4 with filtering)
      H_t = Ā'_t ⊙ H_{t-1} + B̄'_t ⊙ X_t
      # 当Δ_t[c] < g_c(S)时: H_t[c] = H_{t-1}[c] (直接传递历史状态)

    # 标准Mamba输出 (Eq.3)
    Y = SSM_filtered(X_input)  # 使用修改后的Ā', B̄'替代Ā, B̄
    O = Linear_3(σ(Linear_2(I)) ⊙ Y)

  Output: O ∈ R^{S×d_m}
  ```

  LongMamba的超参数搜索策略：
  ```
  # θ (channel selection threshold)
  # 在LongBench-E上独立搜索各模型:
  #   候选: {10^-40, 10^-30, 10^-20, 10^-10, 10^-5, 10^-4, 10^-3,
  #           10^-2, 5×10^-2, 10^-1, 5×10^-1}
  #   Mamba-1.4B:  θ = 10^-30
  #   Mamba2-1.3B: θ = 5 × 10^-2
  #   Zamba2-1.2B: θ = 10^-5

  # C (Δ_t clamping percentile)
  #   候选: {0, 5, 10, 15, 20}
  #   Mamba2-1.3B & Zamba2-1.2B: C = 5
  #   Mamba-1.4B: C = 20

  # g(S) lookup table
  #   间隔: 1000-token (1000, 2000, 3000, ...)
  #   输入S先向下取整到最近1000-token间隔再查表
  ```

  关键数值示例——累积衰减对齐（Eq.13）：
  - 训练长度L=2000：∏Ā_i ≈ 某个值（取决于通道）
  - 测试长度S=16000：∏Ā_i 若不过滤 ≈ 接近0（八倍衰减）
  - LongMamba通过过滤Δ_t<g的token（约占(1-L/S)的比例）使筛选后的∏Ā'_i仍然≈∏_{trained}_Ā_i

  延迟开销：A5000上prefilling延迟增加≤4.5%（表6），A100上prefilling延迟增加≤3.8%（表7），均为batch size=1场景。

## Attamba__Attending_To_Multi-Token_States

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是Attamba——一种将State-Space Models (SSMs) 直接集成到Transformer自注意力机制内部的混合架构。核心设计：用SSM block替换标准的Key (K) 和Value (V) 投影矩阵，SSM将连续的P个token压缩为单个表示，attention仅在这些压缩后的chunk边界表示上进行计算。Query (Q) 投影保持标准方式以保留自回归性质。具体技术包括：(1) 训练时保留所有SSM输出用于causal next-word prediction，推理时仅缓存chunk边界输出（P× KV-Cache缩减）；(2) cyclic chunk boundary——不同层使用不同的chunk边界偏移，减少固定边界引入的偏差；(3) 支持leading tokens (L) 保持对最近token的完整attention（类似sliding-window attention）；(4) pseudo-chunking模式（不裁剪attention map，仅替换K/V投影为SSM）可略微提升Transformer困惑度。实验对比：(a) standard Transformer（iso-parameter/iso-KV/iso-FLOPs变体，通过调整attention model dimension F来匹配Attamba的KV-Cache或FLOPs）；(b) SSM类模型Mamba、minGRU、Hawk；(c) 消融实验：KV投影矩阵有无、SSM state dimension大小、chunking方法（Uniform/Random/Cyclic/FAttn/FSSM）、chunk size (4/8/64/128)、leading tokens数量。

- 硬件平台是什么，配置是什么。
  单张NVIDIA RTX A6000 GPU。训练模型约60M参数（8层、8 heads、512 model dimension），batch size=16，sequence length=1024，约982M tokens训练（100k步≈1B tokens）和100k步8B tokens的扩展实验。框架使用Meta Lingua（Facebook开源的PyTorch LLM训练库）。

- 模型是什么。数据集和bench分别是什么。
  模型：60M参数的Attamba（8层、8 heads、512 model dimension）。默认配置：chunk size P=4/8，leading tokens L=0/P，SSM state dimension D_s=16（总SSM参数开销约4M）。对比Transformer baseline同为8层8 heads但attention model dimension F根据iso-KV/iso-FLOPs条件调整（如表1，P=4时isoF=128/isoFLOPs=160，P=8时isoF=64/isoFLOPs=104）。对比模型：Mamba、minGRU、Hawk均在60-64M参数预算内。数据集：10%subset of dclm-baseline-1.0（Li et al., 2024），最大训练8B tokens。Benchmark：WikiText2 test-set perplexity。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/abdelfattah-lab/attamba（BSD-3-Clause license，基于Meta Lingua框架，Python 99.8%）。训练日志：https://wandb.ai/akhauriyash/attamba_arxiv。论文明确标注"experiments should be reproducible from the current state of this repo."

  Attamba算法pipeline核心——单个attention head的前向传播（训练时）：
  ```
  Input: X ∈ R^{n×e}  (n: sequence length, e: model dimension, P: chunk size)
         Mask ∈ R^{n×n} (causal + chunk boundary mask)

  // 1. Query projection保持不变（标准线性投影）
  Q = X W_Q    // W_Q ∈ R^{e×e}, Q ∈ R^{n×e}

  // 2. Key和Value通过SSM block而非线性投影
  //    X被分为n/P个chunk: X^{(1)}, X^{(2)}, ..., X^{(n/P)}  each ∈ R^{P×e}
  For each chunk p = 1, 2, ..., n/P:
    // SSM_K: 对chunk内P个token做autoregressive SSM扫描
    //          使用Mamba-style selective SSM:
    //          h_t = A_t * h_{t-1} + B_t * x_t  (recurrent state更新)
    //          k_t = C_t * h_t
    K^{(p)} = SSM_K(X^{(p)})  // ∈ R^{P×e}, 每行是causally valid的压缩表示

    // SSM_V: 同理处理value
    V^{(p)} = SSM_V(X^{(p)})  // ∈ R^{P×e}

  // 3. 拼接所有chunk的SSM输出
  K_SSM = concat[K^{(1)}; K^{(2)}; ...; K^{(n/P)}]  // ∈ R^{n×e}
  V_SSM = concat[V^{(1)}; V^{(2)}; ...; V^{(n/P)}]  // ∈ R^{n×e}

  // 4. 构造chunk attention mask (Equation 5)
  //    M_train[i,j] = 0 iff:
  //      - j和i在同一chunk内且j≤i (chunk内causal self-attention), OR
  //      - j≤i且j是某个chunk的最后一个token (跨chunk仅attend chunk边界)
  //    M_train[i,j] = -∞ otherwise

  // 5. Attention（仅attend chunk边界+当前chunk内）
  S = Q K_SSM^T / √d           // S ∈ R^{n×n}, d = e/num_heads
  A = Softmax(S + M_train)     // 仅chunk边界和当前chunk内非-masked
  Y = A · V_SSM                // Y ∈ R^{n×e}

  Output: Y (经过output projection和residual后送入MLP)
  ```

  推理时（test-time）简化版：
  ```
  // 推理仅需保存每个chunk的最后一条SSM输出
  For each chunk p:
    K_SSM_test.append(K^{(p)}[-1])  // 仅最后一个token的K, ∈ R^{1×e}
    V_SSM_test.append(V^{(p)}[-1])  // 仅最后一个token的V

  // Attention mask (Equation 7): 每个query仅attend到已完成的chunk边界
  M_test[i,j] = 0 if j ≤ floor(i/P), else -∞

  // KV-Cache大小: n/P × e × 2  (vs Transformer: n × e × 2)
  // Attention FLOPs: O(n²/P)  (vs Transformer: O(n²))
  // 当L>1时，额外保存每个chunk最后L个token的KV用于sliding window attention
  ```

  具体配置参数含义：
  - P (Chunk size): 每个SSM压缩的token数，P=4→4× KV-Cache缩减，P=8→8×缩减
  - L (Leading tokens): 保持完整attention的最近token数，L=P时保留整个最新chunk
  - D_s (SSM state dimension): 默认16，>32后对P=8收益递减（<1% perplexity差异）
  - Cyclic chunking: 第layer层chunk边界偏移layer个token位置，使不同层处理不同token分组
  - Pseudo-chunking: L=seq_len时的退化情况，SSM替代K/V投影但保持全注意力mask

  复杂度分析（vs 标准Transformer）：
  - KV-Cache: (2n/P + 2L)E vs 2nE（L为leading tokens数）
  - Attention FLOPs: O(n²/P) vs O(n²)
  - SSM FLOPs overhead: O(n × D_s × E) per SSM block (linear, 可忽略不计)
  - SSM参数开销: ~4M（60M模型的6.7%）

  Cyclic chunking的实现：
  ```
  # 第layer_num层的chunk边界从 layer_num % P 位置开始
  For layer in range(num_layers):
    offset = layer % P
    # chunks: [{offset, offset+1, ..., offset+P-1},
    #          {offset+P, offset+P+1, ..., offset+2P-1}, ...]
    # 不同层从不同偏移量开始划分chunk，打破固定边界偏差
  ```

  论文主要结果（WikiText2）：
  - Attamba (P=4) vs iso-KV+SWA Transformer: 困惑度显著改善（~24%）
  - Attamba (P=8) vs iso-KV Transformer: 5% perplexity trade-off for 8× KV-Cache压缩
  - 8B tokens训练：Attamba (P=4, L=4) 困惑度 ~18.5 vs Mamba ~20.2 vs Transformer ~20.8
  - Cyclic chunking比Uniform chunking提升约5%
  - 随机chunk边界工作与均匀分块相当（说明SSM对chunk边界鲁棒）
  - Pseudo-chunking（替换K/V投影但不裁剪attention）比标准Transformer困惑度略优

## M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是M1——一种基于Mamba架构的hybrid线性RNN推理模型，通过多阶段训练pipeline将Transformer的推理能力迁移到Mamba架构。核心训练流程分为三阶段：(1) Stage 1 Distillation：用MambaInLlama框架将Llama3.2-3B-Instruct蒸馏为hybrid Mamba模型。将attention层的Q/K/V/O投影权重初始化为Mamba层的C/B/X/O投影，新增MLP（生成Δ_t）和A参数（dynamic parameter）。与原始MambaInLlama不同，引入两个额外线性层从head_dim*kv_head扩展到head_dim*n_head（因Transformer使用GQA而Mamba不需要KV cache）。蒸馏loss使用reverse KL divergence D_KL(p(·;θ) || p(·;θ_T))，具有mode-seeking特性。训练框架基于Axolotl，使用data packing合并序列至max_len=8192，仅计算assistant输出token的loss。(2) Stage 2 SFT：先用OpenMathInstruct-2做通用数学SFT（两epoch），再用推理数据集（OpenR1-Math-220k + OpenThoughts-114k-math + ServiceNow-AI-R1-Distill + Magpie-Reasoning-250K，共10B tokens）做推理SFT（五epoch），训练长度扩展至24576。(3) Stage 3 Reasoning RL：使用GRPO（Group Relative Policy Optimization）进行RL训练，移除KL penalty项（实验发现不稳定），添加entropy bonus鼓励策略多样性。Batch size=128，PPO batch size=64（µ=2 iterations），每序列生成8个rollout，最大生成长度32k。使用Adam optimizer with LR=1e-6，训练50步后选最高critic reward的checkpoint。GRPO训练集成进VeRL框架，修复了CUDA graph与PyTorch FSDP的兼容性问题。
  实验比较：(a) 数学推理benchmark（AIME25/AIME24/MATH500/AMC23/OlympiadBench）：M1-3B vs DeepSeek-R1-Distill-Qwen-1.5B及Qwen2.5-Math-7B-Instruct/rStar-Math-7B等；(b) 推理速度benchmark（vLLM 0.6.3，H100 GPU）：M1-3B vs Llama-3.2-3B vs DeepSeek-R1-Distill-Qwen-1.5B，变化batch size (8-512)和decoding length（固定prompt=256, decode=4096或固定batch=128, 变化decode length）；(c) Test-time scaling：majority vote accuracy vs sample count（最多64 samples）和generation time budget；(d) 消融实验：各训练阶段后的MATH500/AIME24 accuracy，验证distillation→SFT(MATH)→SFT(Reason)→RL各阶段贡献。

- 硬件平台是什么，配置是什么。
  推理速度benchmark：单张NVIDIA H100 GPU，使用vLLM 0.6.3推理引擎，greedy decoding（ignore_eos=True保证生成到最大长度），warmup两次后平均三次测量。训练硬件论文未明确说明具体GPU型号及数量（来自TogetherAI）。RL训练通过VeRL框架（https://github.com/volcengine/verl）进行，修复了Mamba+FSDP的CUDA graph兼容性问题，使CUDA graph启用时训练速度提升5x。

- 模型是什么。数据集和bench分别是什么。
  模型：M1-3B，基于Llama3.2-3B-Instruct蒸馏。架构为hybrid Mamba：28层total，其中6层为interleaved attention层（~21%），其余为Mamba层。SSM state size=16，SSM groups=3072/16=192。对比模型：(a) DeepSeek-R1-Distill-Qwen-1.5B（transformer推理模型，1.5B参数）；(b) Llama-3.2-3B（同参数transformer baseline，非推理模型）；(c) Qwen2.5-Math-7B-Instruct/rStar-Math-7B/Eurus-2-7B-PRIME/Qwen2.5-7B-SimpleRL（更大模型的参考对比）。训练数据集：(a) 蒸馏阶段——基于Llama3.2-3B的token-level KL divergence（数据集为通用预训练语料，论文未明确说明具体数据）；(b) SFT-MATH阶段——OpenMathInstruct-2；(c) SFT-Reason阶段——OpenR1-Math-220k + OpenThoughts-114k-math + ServiceNow-AI-R1-Distill + Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B（总计10B reasoning tokens）；(d) RL阶段——数学问题训练集（论文未明确说明具体数据集）。总训练token数<50B。Benchmark：(a) MATH500（Hendrycks et al., 2021）；(b) AIME25（MAA, 2025）；(c) AIME24（MAA, 2024）；(d) AMC23（MAA, 2023）；(e) OlympiadBench（He et al., 2024）。评估使用VeRL的evaluation tools，temperature=0.7，max sequence length=32k。Pass@1取64次平均，majority voting重复100次计算。评估prompt统一为"Let's think step by step and output the final answer within \boxed{}"。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/jxiw/M1，模型checkpoint在HuggingFace发布。训练框架基于Axolotl（https://github.com/axolotl-ai-cloud/axolotl，蒸馏和SFT）和VeRL（https://github.com/volcengine/verl，GRPO RL训练）。

  M1训练pipeline核心——三阶段训练算法：

  **Stage 1: 蒸馏（Distillation）**
  ```
  # 基于MambaInLlama (Wang et al., 2024a) 的跨架构蒸馏
  # Teacher: Llama3.2-3B-Instruct (Transformer with GQA)
  # Student: Hybrid Mamba model (28 layers, 6 attention + 22 Mamba)

  # Step 1: 权重初始化（Algorithm 1 - MAMBAINLLAMA）
  For each attention layer to be converted to Mamba:
    # 从Teacher的attention投影初始化Mamba参数
    # QKV投影 → C/B/X投影（Mamba不使用KV cache，需扩展GQA的KV heads）
    W_C = W_Q  # Q投影 → C投影
    W_B = Linear_expand(W_K)  # K投影 → 扩展至full heads → B投影
    W_X = Linear_expand(W_V)  # V投影 → 扩展至full heads → X投影
    W_O = W_O  # 输出投影直接复用

    # 新增Mamba参数（随机初始化）
    A ∈ R^{N×N'}  # dynamic parameter, N=head_dim, N'=expand_dim
    MLP_Δ: R^N → R^{N'}  # 生成采样率Δ

    # 保留MLP层（直接复用Transformer的MLP权重）
    # 保留6个interleaved attention层不变

  # Step 2: Token-level reverse KL蒸馏
  For each training step:
    # 前向传播
    p_teacher = Teacher(input_ids)  # Teacher输出完整概率分布
    p_student = Student(input_ids)  # Student输出完整概率分布

    # Reverse KL divergence（mode-seeking）
    loss = D_KL(p_student || p_teacher)
         = Σ_t Σ_v p_student(t,v) * log(p_student(t,v) / p_teacher(t,v))

    # Data packing: 合并多个序列至max_len=8192
    # Chat template: mask user prompt部分，仅计算assistant token的loss
    loss.backward()

  # Optimizer: AdamW, LR=1e-5, cosine decay, β=(0.9,0.95), weight_decay=0.1
  ```

  **Stage 2: SFT（Supervised Fine-Tuning）**
  ```
  # Sub-stage 2a: Math SFT on OpenMathInstruct-2
  For epoch in range(2):
    For each (question, solution) in OpenMathInstruct-2:
      input = apply_chat_template(question)
      target = solution
      loss = CrossEntropy(Student(input), target)  # 仅计算assistant token
      # Optimizer: 同蒸馏阶段

  # Sub-stage 2b: Reasoning SFT with 10B reasoning tokens
  # 数据集混合: OpenR1-Math-220k + OpenThoughts-114k-math
  #            + ServiceNow-AI-R1-Distill + Magpie-Reasoning-250K
  For epoch in range(5):
    For each (question, reasoning_solution) in mixed_reasoning_data:
      input = apply_chat_template(question)
      max_seq_len = 24576  # 覆盖99%数据
      target = reasoning_solution  # 包含完整chain-of-thought
      loss = CrossEntropy(Student(input), target)
      # Optimizer: AdamW, LR=6e-6 (降低peak LR), 其余同蒸馏
  ```

  **Stage 3: Reasoning RL（GRPO Training）**
  ```
  # 使用VeRL框架 + GRPO loss (modified, 无KL penalty)
  # L_GRPO(θ) = E[π_θ(a|s)/π_θold(a|s) * Â(s,a)] + η·H(π_θ)

  For step in range(50):
    # 1. Rollout generation (batch_size=128)
    For each question in batch:
      # 每个问题生成8个rollout
      prompt = "Let's think step by step and output the final answer within \\boxed{}"
      For g in range(8):
        output_g = model.generate(question + prompt, max_len=32k, temperature=0.7)
        reward_g = critic(output_g, ground_truth)  # 基于答案正确性的reward

    # 2. Advantage computation
    # Â(s,a) = (reward - mean(rewards)) / std(rewards)  # group-relative advantage
    advantages = compute_group_advantages(rewards)  # per-group normalization

    # 3. PPO update (µ=2 iterations, ppo_batch_size=64)
    For iter in range(2):
      For mini_batch in split(rollouts, batch_size=64):
        # Policy gradient with importance sampling
        ratio = π_θ(a|s) / π_θold(a|s)
        loss = ratio * advantages + η * entropy(π_θ)
        loss.backward()
        optimizer.step()  # Adam, LR=1e-6

    # 4. Checkpoint selection
    if reward_critic(current_model) > best_reward:
      save_checkpoint()

  # CUDA graph优化: 修复FSDP+CUDA graph兼容性 → 5x训练加速
  ```

  **M1 Hybrid Mamba单层前向传播（推理时生成一个token）:**
  ```
  Input: x_t ∈ R^D, h_{t-1} ∈ R^{N×N'} (Mamba state)

  # Mamba层（28层中的22层）
  # Step 1: Input projection
  x_proj = Linear_x(x_t)  # ∈ R^N
  z = Linear_z(x_t)       # ∈ R^N, for gating

  # Step 2: 1D convolution + SiLU
  x_conv = CausalConv1d(x_proj)  # kernel=4
  x_act = SiLU(x_conv)

  # Step 3: SSM parameters
  Δ_t = softplus(Linear_Δ(x_act) + bias_Δ)  # ∈ R^N'
  B_t = Linear_B(x_act)  # ∈ R^{N×1}
  C_t = Linear_C(x_act)  # ∈ R^{N×1}

  # Step 4: Discretization + State update
  A_bar, B_bar = discretize(A, B_t, Δ_t)  # Zero-order hold
  h_t = A_bar ⊙ h_{t-1} + B_bar ⊗ x_act  # Selective SSM, O(N×N')
  y_t = C_t^T h_t  # ∈ R^N

  # Step 5: Gating + Output
  y_t = y_t ⊙ SiLU(z)
  output = Linear_O(y_t)  # ∈ R^D

  # Attention层（28层中的6层，interleaved）
  # 标准Multi-Head Attention（保留的Transformer层）
  Q, K, V = Linear_QKV(x_t)
  attn_out = Softmax(Q @ K^T / √d) @ V
  output = Linear_O_attn(attn_out)

  # MLP层（所有层共享MLP设计）
  output = output + SiLU(Linear_gate(x_t)) ⊙ Linear_up(x_t)
  ```

  **性能摘要（Table 1/2）:**
  | Model | AIME25 | AIME24 | MATH500 | AMC23 | OlympiadBench |
  |-------|--------|--------|---------|-------|---------------|
  | DeepSeek-R1-Qwen-1.5B | 23.0 | 28.8 | 82.8 | 62.9 | 43.3 |
  | M1-3B | 23.5 | 28.9 | 82.1 | 62.8 | 47.3 |

  **速度摘要（Figure 1/2, H100+ vLLM 0.6.3）:**
  - Batch=512, decode=4096, prompt=256: M1 3× faster than Llama-3.2-3B
  - Batch=128, 变化decode length: M1始终2×+ faster than Llama-3.2-3B
  - 最优吞吐量: M1 = 15169 T/s vs DeepSeek-R1-Qwen-1.5B = 7263 T/s

## ML-Mamba__Efficient_Multi-Modal_Large_Language_Model_Utilizing_Mamba-2

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是ML-Mamba——一种基于Mamba-2状态空间模型（SSM）的多模态大语言模型（MLLM），用预训练Mamba-2 LLM替换传统Transformer backbone，通过新提出的Mamba-2 Scan Connector（MSC）桥接2D非因果视觉特征与SSM的1D因果建模能力。核心组件：(1) 双视觉编码器DINOv2（ViT-Large, 304M参数）+ SigLIP（shape-optimized ViT），输出分辨率384×384，729个visual tokens；(2) Mamba-2 Scan Connector（MSC），包含Mamba-2 Visual Selective Scanning（MVSS）模块和SwiGLU模块。MVSS探索两种2D扫描机制：Bidirectional-Scan Mechanism（BSM，前后方向扫描互补特征）和Cross-Scan Mechanism（CSM，四方向对角线扫描）；(3) MLP Projector（三层MLP）对齐视觉和文本特征；(4) Mamba-2 2.7B LLM作为语言主干。MSC有三种变体：MLP（纯三层MLP）、MSC-MLP Basic（MSC不含SwiGLU + MLP）、MSC-MLP Advanced（MSC含SwiGLU + MLP）。
  实验比较：(a) 六项benchmark对比SoTA方法（Table 2）：VQAv2、GQA、TextVQA、POPE、VizWiz、VSR，对比BLIP-2、MiniGPT-4、InstructBLIP、Shikra、IDEFICS、Qwen-VL、LLaVA-1.5（7B/13B）、TinyLLaVA（Phi2-2.7B）、LLaVA-Phi（Phi-2-2.7B）、MobileVLM-3B、Cobra（Mamba LLM-2.8B）、VL-Mamba（Mamba LLM-2.8B）；(b) 推理速度对比（Table 3）：ML-Mamba vs TinyLLaVA 3B vs MobileVLM v2 3B，单卡A100 PCIe 80GB，统一图片336×336输入，ML-Mamba实际处理729个tokens（分辨率384×384），测量evalavg tokens/s和total latency；(c) 消融实验——语言模型变体（Table 4：Mamba2-780m/1.3b/2.7b）、视觉编码器组合（Table 5：DINOv2 vs SigLIP vs DINOv2+SigLIP）、多模态连接器结构（Table 6：MLP vs MSC-MLP Basic vs MSC-MLP Advanced）、扫描机制（Table 7：BSM vs CSM）。

- 硬件平台是什么，配置是什么。
  训练：8× NVIDIA A100 80GB GPU，总训练时间约31小时。使用PyTorch FSDP（Fully Sharded Data Parallel）分布式训练框架，自动混合精度FP32+BF16。batch size=64，优化器AdamW，学习率2e-5，cosine decay（decay factor 0.1），warmup ratio 0.03，weight decay 0.1。对齐阶段1 epoch（558K样本），微调阶段1 epoch（665K样本）。
  推理速度测试：单卡NVIDIA A100 PCIe 80GB GPU，统一图片分辨率336×336（CLIP encoder处理），设定输出256 tokens。

- 模型是什么。数据集和bench分别是什么。
  模型：ML-Mamba，Vision Encoder = DINOv2（ViT-Large）+ SigLIP，LLM Backbone = Mamba-2 2.7B（在Pile数据集300B tokens上预训练），MSC = MSC-MLP Advanced（含MVSS模块BSM扫描 + SwiGLU）。对比模型包括TinyLLaVA 3B（Phi-2 2.7B backbone）、MobileVLM v2 3B（MobileLLaMA 2.7B backbone）、LLaVA-Phi（Phi-2-2.7B）、Cobra（Mamba LLM-2.8B）、VL-Mamba（Mamba LLM-2.8B）。
  数据集：对齐阶段——558K LAION-CC-SBU子集；微调阶段——665K Mixed Dataset（来自LLaVA v1.5，包含视觉多轮对话和纯文本对话数据）。LLM预训练数据——Pile数据集300B tokens。
  Benchmark：(1) 开放VQA任务：VQAv2（通用视觉推理，验证集）、GQA（空间理解和多步推理，test-dev partition）、TextVQA（OCR和光学字符推理，验证集）、VizWiz（常识+不可回答问题，验证集）；(2) 闭集预测任务：POPE（物体幻觉检测，二元分类，evaluation partition）、VSR（空间关系理解，zero-shot test partition）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  完全开源（MIT License）。代码：https://github.com/WenjunHuang94/ML-Mamba，项目页：https://wenjunhuang94.github.io/ML-Mamba，模型权重在Hugging Face发布。

  ML-Mamba前向传播算法pipeline（MSC-MLP Advanced, BSM扫描，推理时处理一张图片+文本问题）：

  ```
  Input: 图片 X_v ∈ R^{C×384×384}, 文本查询 Q_text

  // Step 1: 双视觉编码器特征提取
  patches = patchify(X_v, P=14)  // 384/14 ≈ 27, N_v = 27×27 = 729 patches
  V_siglip = SigLIP_ViT(patches)  // ∈ R^{729×D_sig}
  V_dino = DINOv2_ViT(patches)    // ∈ R^{729×D_dino}
  V_img = concat([V_siglip; V_dino], dim=-1)  // ∈ R^{729×D_v}

  // Step 2: Mamba-2 Scan Connector (MSC) - BSM
  // Forward scan: 原始patch顺序
  V_f = Mamba2_Block(V_img)  // 1D SSM scan along 729 patches
  // Backward scan: 反转patch顺序
  V_b = Mamba2_Block(flip(V_img))
  V_scan_bsm = V_f + flip(V_b)  // 合并前后向扫描 ∈ R^{729×D_v}

  // Step 3: SwiGLU feature extraction
  V_gate = Linear_gate(V_scan_bsm)     // ∈ R^{729×D_v}
  V_proj = Linear_proj(V_scan_bsm)    // ∈ R^{729×D_v}
  V_scan = SiLU(V_gate) ⊙ V_proj      // gated activation

  // Step 4: MLP Projector (三层MLP)
  V_out = MLP_3layer(V_scan)  // ∈ R^{729×D_llm}

  // Step 5: 文本token化与拼接
  T_tokens = Tokenize(Q_text)
  T_emb = Embedding(T_tokens)  // ∈ R^{L_text×D_llm}
  Input_emb = concat([V_out; T_emb], dim=0)

  // Step 6: Mamba-2 LLM 自回归生成
  // Mamba-2 block（每层）:
  //   x_proj, z_proj = Linear_in(x_norm)  // expand 2×D
  //   x_conv = CausalConv1d(x_proj, window=4)
  //   x_act = SiLU(x_conv)
  //   Δ, B, C = split(Linear_dt(x_act))  // 数据依赖参数
  //   A_bar, B_bar = discretize(A, B, Δ)  // ZOH离散化
  //   h_t = A_bar ⊙ h_{t-1} + B_bar ⊗ x_act[t]  // recurrent state update
  //   y[t] = C ⊗ h_t
  //   y = y ⊙ SiLU(z_proj)
  //   output = Linear_out(y) → residual

  Response = AutoregressiveGenerate(Mamba2_LLM, Input_emb, max_tokens)
  ```

  推理时关键特性：
  - Mamba-2每层每token O(1)计算，固定大小hidden state（无KV-Cache增长）
  - 速度对比（Table 3）：ML-Mamba 171 tokens/s, 总时间1.47s（256 tokens），远超TinyLLaVA 38 tokens/s（6.45s）和MobileVLM v2 50 tokens/s（5.15s）
  - 即使处理729个visual tokens（远多于TinyLLaVA的576和MobileVLM的144），ML-Mamba仍因RNN-like特性维持高速

  训练流程：
  ```
  Step 1 (Alignment): 冻结Vision Encoder + Mamba-2 LLM，仅训练MSC + MLP Projector
                     数据: 558K LAION-CC-SBU子集，1 epoch
  Step 2 (Fine-tuning): 训练MSC + Projector + Mamba-2 LLM（全参数监督微调）
                        数据: 665K Mixed Dataset（LLaVA v1.5格式），1 epoch
  总计: ~31小时 on 8× A100 80GB
  ```

  性能摘要（Table 2，仅用LLaVA-1.5 7B约40%参数）：
  - ML-Mamba: VQAv2 75.26, GQA 60.68, TextVQA 52.2, POPE 88.3, VizWiz 45.17, VSR 51.5
  - LLaVA-1.5 7B: VQAv2 78.5, GQA 62.0, TextVQA 58.2, POPE 85.9, VizWiz 50.0
  - 在POPE（88.3 vs 85.9）上超越LLaVA-1.5 7B，VSR上表现优异（51.5）

## Rethinking_Token_Reduction_for_State_Space_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：面向SSM（Mamba系列）的统一后训练Token Reduction方法——UTRC（Unified Token Reduction by token importance Classification）。核心流程为：(1) 从SSM层的隐藏状态y提取token重要性度量 `S = Σ_{d=1}^{D'} max(0, y_{::d}) / D'`（每个token所有通道的正值求和并平均）；(2) 基于重要性将所有token分为两等份——集合M_A（重要性低的N/2个token）和集合M_B（重要性高的N/2个token）；(3) 为每个M_A中token a_i计算它到M_B中最相似的token f_i：`f_i = argmax_{b_j∈M_B} sim(a_i, b_j)`，得到最大相似度g_i；(4) 保留相似度最高的p%连接对；(5) 对保留的连接对执行UTR操作：对(p×q)%的连接进行pruning（删除a_i，保留f_i），对剩余[p×(1-q)]%的连接进行merging（`f_i = (a_i + f_i) / 2`，删除a_i）；(6) 重新组合token集。默认q=0.5时效果最佳。在hidden states上使用hybrid（q=0.5），在residual connections上只使用merging，避免去除关键残差信息。(7) 层次化应用：从第10~12层开始，每5层执行一次token reduction（如Mamba-2-2.7B在[12,17,22,27,32,37,42]层），使用固定的压缩率。
  - 实验比较：(1) 与baseline方法PuMer和EViT在Mamba-2-1.3B、Mamba-2-2.7B、Mamba-2.8B、Mamba-1.4B上比较，分别在10%/20%/30%的FLOPS Reduction下评估；(2) 消融实验：不同token重要性度量（ℓ1-norm、ℓ2-norm、无Clip、带Clip）；不同reduction位置配置（如[10,15,20,...]/[12,17,22,...]等6种配置）；不同design choices（P-only、M-only、不同q值组合）；(3) GPU峰值内存和吞吐量测量；(4) 附录中与LTMP方法对比。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB GPU
  - 所有实验均在单卡A100上完成（论文未明确说明多卡训练/推理）

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mamba-2-1.3B、Mamba-2-2.7B、Mamba-2.8B（基于Mamba-2架构，Dao and Gu, 2024）、Mamba-1.4B（基于Mamba架构，Gu and Dao, 2023）
  - 数据集/benchmark：LAMBADA（Perplexity + Accuracy）、HellaSwag、PIQA、Arc-Easy、Arc-Challenge、WinoGrade（零样本评估，无微调）
  - 推理配置：生成2048 tokens、batch size 96（峰值内存测量）；prompt length 2048、batch size 16、生成100 tokens（吞吐量测量）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/wuyushuwys/ToR_SSM
  - 框架：PyTorch + HuggingFace Transformers
  - 算法pipeline伪代码（对Mamba第l层执行UTRC token reduction）：

```
# 输入: T_{l-1} ∈ R^{B×N×D} (上一层的token序列，B=batch, N=序列长度, D=特征维度)
#       reduction_rate: 目标压缩率 (如0.2表示减少20% FLOPs)
#       start_layer: 开始reduction的层号
#       interval: reduction间隔层数
#       q: hybrid比例参数 (默认0.5)

# === Step 0: 判断是否在当前层执行reduction ===
if layer_id < start_layer or (layer_id - start_layer) % interval != 0:
    return standard_mamba_layer(T_{l-1})  # 不执行reduction

# === Step 1: 获取SSM隐藏状态并计算Token重要性 ===
x = Linear_proj(T_{l-1})          # 投影到D'维: [B, N, D']
y = SSM(A, B, C)(x)              # 通过SSM, y∈[B, N, D']
# 计算token重要性 (Equation 5)
S_i = sum(max(0, y_{i,:})) / D'  # 对每个token i, 正通道值求平均
                                  # S ∈ R^{B×N×1}

# === Step 2: Token重要性分类 ===
# 按S降序排列所有N个token
sorted_idx = argsort(S, descending=True)  # [B, N]
M_B = sorted_idx[:, :N//2]               # 重要性高的前50% token
M_A = sorted_idx[:, N//2:]               # 重要性低的后50% token

# === Step 3: 建立相似度连接 ===
for each a_i in M_A:
    similarities = [cosine_sim(a_i, b_j) for b_j in M_B]  # cosine相似度
    f_i = M_B[argmax(similarities)]   # Equation 6: 最相似的M_B中token
    g_i = max(similarities)            # Equation 7: 最大相似度值

# === Step 4: 保留top-p%相似连接 ===
# 按g_i降序排序所有连接对
num_keep = int(p * len(M_A))
keep_pairs = sort_by_g_desc({(a_i, f_i, g_i)})[:num_keep]

# === Step 5: Unified Token Reduction (UTR) ===
num_prune = int(q * num_keep)    # q=0.5 by default
num_merge = num_keep - num_prune

# 取前num_prune对做pruning
for (a_i, f_i) in keep_pairs[:num_prune]:
    M_A = M_A \ {a_i}            # 删除a_i, f_i保持不变
# 取后num_merge对做merging  
for (a_i, f_i) in keep_pairs[num_prune:]:
    T_l[f_i] = (T_l[a_i] + T_l[f_i]) / 2   # 平均融合到f_i
    M_A = M_A \ {a_i}                       # 删除a_i

# === Step 6: 重新组装token序列 ===
# 将M_B和缩减后的M_A合并
if reduction_on_hidden_states:
    T_l_hidden = reassemble(M_B, M_A_reduced)
if reduction_on_residual:
    T_l_residual = merge_only(M_B, M_A)  # 残差只用merging

# === Step 7: 最终输出 ===
T_l = Linear(y) + T_l_residual    # 标准Mamba层输出的简化版
return T_l
```

  - 关键代码实现细节：
    - **Token重要性计算**：在Mamba block的SSM输出处（after selective scan），读取y hidden states，按`max(0, y).sum(dim=-1)`计算每个token的importance score，除以D'归一化
    - **Similarity计算**：使用余弦相似度 `cosine_similarity(a_i, b_j)`，通过矩阵乘法实现快速计算：`sim_matrix = norm(M_A) @ norm(M_B).T`
    - **残差连接处理**：论文发现在residual上只用merging（不删除任何残差信息）比hybrid/pruning-only效果更好（PPL 40.61 vs 42.61），因为残差保留了上一层的关键信息
    - **层次化reduction**：不在每层都做reduction（相邻层token重要性相似），每5层做一次；不从太早层开始（前几层尚未充分捕获token重要性），Mamba-2-2.7B从第12层开始
    - **PPL/Accuracy评估适配**：由于token数量减少，评估时需对应调整label logits，只取前(1-m%)的logits计算PPL和Accuracy

## Stuffed_Mamba__State_Collapse_and_State_Capacity_of_RNN-Based_Long-Context_Modeling

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：论文分析了 Mamba-2 等 RNN 架构在长上下文泛化失败的根本原因——"无法遗忘（inability to forget）"，并提出两种无需训练的遗忘诱导方法。核心发现：(1) Mamba-2 的某些 head 对首 token 的记忆保留强度 α_{1:t} 在整个训练长度窗口内始终 > 0.997，导致状态过参数化（state overparameterization）；(2) 状态均值/方差在超过训练长度后发生爆炸（variance explosion），主要由少数 outlier channel 驱动；(3) 遗忘阈值 T_forget 与状态大小 N_S 成线性关系：T_forget = 5.172 · N_S − 4.469 (R² > 0.999)；(4) 最大召回上下文长度 T_recall 与状态大小成指数关系：T_recall = 4.756 · (1.365^{N_S} − 1) − 0.742 (R² > 0.999)。两种遗忘诱导方法：(a) RRI（Reduced Memory Retention and Insertion）：将记忆保留强度 α_t 缩放 0.9999 倍，记忆插入强度 B_t 缩放 0.75 倍（超参由 32K 验证集选择）；(b) Sliding Window：利用状态加权和性质，通过 h_t^{(r)} = h_t − α_{t−r+1:t} · h_{t−r} 精确模拟滑动窗口，维护 Δ_{t−r:t} 避免浮点不稳定。
  - 实验比较：(1) LM loss vs token position：Mamba-2 130M/370M/780M 在 8K 训练长度下，context > 8K 后 perplexity 急剧爆炸（Figure 1, 10）；(2) Passkey Retrieval：Mamba-2 130M/370M/780M 在 1K-256K context，近乎完美准确率仅在 ≤8K，>16K 后几乎为零（Figure 2）；(3) Inducing forgetting：RRI 和 Sliding Window 方法在 32K context 下将 LM loss 从 ~15 降至 ~8-10，LongMamba 也有类似改善但牺牲短上下文性能（Figure 4）；(4) 训练长度 vs 状态大小：sweep 6 种模型规模（36M/47M/85M/130M/370M/780M）和多种训练长度（最高 256K），验证 T_forget 线性关系和 T_recall 指数关系（Figure 9, 11）；(5) 更多训练 ⇒ 更少遗忘：Mamba-2 370M 从零训练过程中，passkey retrieval 精度在短训练长度内随数据量增加反而下降（Figure 8），呈现过拟合行为；(6) 其他架构比较：RWKV-5、RWKV-6、Mamba-1、HGRN-2 在 passkey retrieval 和 "newlines" prompt 上的对比（Figure 13-15, Appendix H）。

- 硬件平台是什么，配置是什么。
  - 训练：NVIDIA A800 80GB GPU，部分实验多节点、部分单节点多 GPU
  - 训练精度：BF16 为主，部分激活值用 FP32（与官方 Mamba-2 实现一致）
  - 推理评估：FP32 精度（确保精度误差不引入噪声），greedy decoding
  - 优化器：AdamW，weight decay=0.1，gradient clipping=1.0
  - LR scheduler：WSD（warmup-stable-decay），10% decay steps，1000 步 linear warmup，50K 步 linear decay
  - Batch size：0.5M tokens/step
  - 学习率 sweep：{1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3}，passkey retrieval 验证选择最优

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mamba-2 官方 checkpoint（130M: L24/D768/H24, state size 4.8M；370M: L48/D1024/H32, state size 12.9M；780M: L48/D1536/H48, state size 19.3M）+ 从零训练的 checkpoint（36.4M: L6/D512/H16, state size 0.8M；47.0M: L12/D512/H16, state size 1.6M；84.6M: L12/D768/H24, state size 2.4M）。Mamba-2 配置：P=64（head dim），N=128（state dim），H=2d/P（head 数），expansion factor=2，conv kernel=4。还评估了 RWKV-5、RWKV-6、Mamba-1、HGRN-2 作为对比。
  - 训练数据集：RedPajama-V2（30T tokens，过滤短于 4K tokens 的文档——过滤掉约 97.6% 数据），使用 Truncated BPTT（12 序列拼接，等价于 concatenation + 截断梯度）
  - Benchmark：(1) Language Modeling：RedPajama 验证集，perplexity/loss 随 token position 变化；(2) Passkey Retrieval：5-digit passkey，均匀分布 needle position，context 1K-256K，greedy decoding；(3) "newlines" prompt：纯换行符序列，用于检测状态分布稳定性（mean/variance explosion）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 官方 Mamba-2 checkpoint：https://huggingface.co/state-spaces/mamba2-130m / mamba2-370m / mamba2-780m
  - 论文未明确给出独立代码仓库，但方法和分析均基于公开检查点
  - 算法 pipeline（RRI + Sliding Window 的推理时遗忘诱导）：
    ```
    # ===== Mamba-2 原始推理（一个 head，t 时刻） =====
    # 输入: u_t ∈ R^d
    # 参数: W_B, W_C, W_x, W_Δ, b_Δ, A ∈ R
    x_t = SiLU(Conv(u_t @ W_x))^T      # [1, P]
    B_t = σ(Conv(u_t @ W_B))            # [N, 1]
    C_t = σ(Conv(u_t @ W_C))^T          # [1, N]
    Δ_t = Softplus(u_t @ W_Δ + b_Δ)     # scalar
    α_t = exp(-Δ_t * exp(A))            # decay multiplier ∈ (0,1)
    h_t = h_{t-1} * α_t + Δ_t * B_t * x_t  # [N, P] state update
    y_t = C_t @ h_t + D ⊙ x_t           # [1, P]

    # ===== RRI (Reduced Memory Retention and Insertion) =====
    # 干预 α_t 和 B_t，无需重新训练
    α_t' = α_t ** 0.9999       # scale retention closer to 0
    B_t' = B_t * 0.75           # weaken insertion strength
    h_t = h_{t-1} * α_t' + Δ_t * B_t' * x_t

    # ===== Sliding Window (精确窗口状态) =====
    # 窗口大小 w
    # 维护三个量: h_{t-1}, h_{t-w-1}, Δ_sum = Σ_{i=t-w}^t Δ_i
    # 在 t 时刻:
    h_t = h_{t-1} * α_t + Δ_t * B_t * x_t        # 正常更新
    Δ_sum = prev_Δ_sum * (1 − reset_flag) + Δ_t  # 维护 Δ 累积和
    α_window = exp(-Δ_sum * exp(A))              # 窗口衰减因子
    h_t^{(w)} = h_t - α_window * h_{t-w}          # 精确窗口状态

    # 推理时使用 h_t^{(w)} 替代 h_t 进行 query
    y_t = C_t @ h_t^{(w)} + D ⊙ x_t
    ```
    关键洞察：(1) Mamba-2 的 state h_t 是加权和形式（h_t = Σ α_{i:t} B̄_i x_i），因此 Sliding Window 可精确计算为两个状态的差，无需重新处理窗口内所有 token；(2) 直接计算 α_{t−r:t} 可能因浮点精度不稳定，改为维护 Δ 累积和并每步重新计算 α_window；(3) 方法适用于所有可表为加权和的 RNN（GLA、RWKV、RetNet 等）。

  - 训练时 Truncated BPTT 实现：
    ```
    # 等价于序列拼接 + 梯度截断
    # 12 序列拼接，总长 ≈ 12 * T_train
    for batch in dataloader:
        h_0 = zeros(N, P)          # 初始化为零
        for seq in batch:          # batch 内 12 个序列
            for t in range(len(seq)):
                h_t = update(h_{t-1}, seq[t])
                loss += CE(linear_head(h_t), seq[t+1])
            h_0 = h_t.detach()     # 截断梯度，继续用当前 state
            # 下一个序列从 h_t 开始（状态延续）
    ```

## VisualRWKV__Exploring_Recurrent_Neural_Networks_for_Visual_Language_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：VisualRWKV 是首个将线性 RNN（RWKV）应用于多模态视觉语言模型的架构。核心创新包括三部分：(1) Data-dependent Recurrence：将 RWKV 的数据独立 token shift 和 time mixing 升级为数据依赖版本——Token Shift 通过 ddlerp (data-dependent linear interpolation) + LoRA 实现，`ddlerp_α(a,b)=a+(b-a)⊙(λ_α+tanh(a+(b-a)⊙μ_x)A_α)B_α`；Time Mixing 的 decay 从固定 w 变为动态 w_t，`d_t=lora_d(ddlerp_d(x_t,x_{t-1})), w_t=exp(-exp(d_t))`；(2) Sandwich Prompt：在 instruction token 中间插入 image token，使模型先阅读指令再处理图像再继续指令，解决 RNN 无法回溯的问题；(3) 2D Image Scanning：将 RWKV 的因果单向扫描扩展为双向(BiDir: Forward+Backward)和多向(MultiDir: Forward+Backward+Upward+Downward)交替排列，不增加计算量但增强 2D 视觉信息提取。训练流程为两阶段：(1) 视觉-语言对齐预训练（冻结 vision encoder 和 RWKV LLM，仅更新 projector）；(2) 视觉指令微调（同时更新 projector 和 RWKV LLM）。使用 CLIP-L (0.3B) 作为 vision encoder，RWKV-5/RWKV-6 系列作为 LLM backbone。
  - 实验比较：(1) Main Results（Table 2）：VisualRWKV 1.6B/3B/7B 在 8 个 benchmark（VQA-v2, GQA, ScienceQA, TextVQA, POPE, MME, MMBench, MMBench-CN）上对比 LLaVA-1.5、BLIP-2、InstructBLIP、MiniGPT-4、Qwen-VL、Shikra、MobileVLM 等 SOTA Transformer VLM；(2) Scaling 消融（Table 1）：VisualRWKV-Base → +Data-dep Recurrence → +Bidirection + Sandwich → +Better LR → Scale up 3B → Scale up 7B；(3) Prompting 方法消融（Table 3）：Image First vs Image Last vs Sandwich Prompt；(4) Scanning 方法消融（Table 4）：UniDir vs BiDir vs MultiDir；(5) 学习率消融（Table 10）：1.6B/3B/7B 各 scale 的最优学习率搜索；(6) 效率分析（Figure 1）：VisualRWKV 7B vs LLaVA-1.5 7B 的推理速度和 GPU 内存随序列长度变化（最高 24K tokens），VisualRWKV 速度优势 3.98×，节省 54% GPU 内存；(7) Text-only 能力（Table 5）：验证视觉指令微调后纯文本能力不退化；(8) 单阶段 vs 两阶段训练（Figure 5）；(9) Loss reduction 方法消融（Table 7/8）：batch-level vs sample-level reduction；(10) Weight decay 消融（Table 13）；(11) VisualRWKV Hybrid（Table 14）：添加 Tiny Attention layer 的混合模型。

- 硬件平台是什么，配置是什么。
  - 训练：8× NVIDIA A100-80GB GPU（标准训练和 benchmark 评估），VisualRWKV 7B 使用 6× A100 GPU（因 8 GPU 显存不够）
  - 效率分析：单张 L20-48GB GPU
  - 优化器：AdamW，cosine decay LR schedule，无 weight decay
  - 训练框架：NVIDIA PyTorch NGC Container (23.07-py3)，lightning 1.9.5，DeepSpeed 0.12.6
  - 计算预算：VisualRWKV 1.6B 单 epoch 53.6 GPU hours（8×A100）；3B 单 epoch 90.4 GPU hours（8×A100）；7B 单 epoch 159 GPU hours（6×A100）

- 模型是什么。数据集和bench分别是什么。
  - 模型：VisualRWKV-Base（RWKV-5 1.6B LLM backbone）、VisualRWKV 1.6B（RWKV-6 1.6B + CLIP-L 0.3B = 1.9B 总参数）、VisualRWKV 3B（RWKV-6 3.1B + CLIP-L = 3.4B）、VisualRWKV 7B（RWKV-6 7.6B + CLIP-L = 7.9B）、VisualRWKV-Hybrid（7B + Tiny Attention layer）。对比模型：LLaVA-1.5（Vicuna-7B/13B）、BLIP-2（Vicuna-13B）、InstructBLIP（Vicuna-7B/13B）、MiniGPT-4、Shikra、Qwen-VL/Chat、MobileVLM-3B、VL-Mamba、LLaVA-Phi、IDEFICS-9B/80B、Otter、mPLUG-Owl
  - 训练数据（与 LLaVA-1.5 完全一致）：(1) 视觉-语言对齐预训练：558K subset of LAION-CC-SBU；(2) 视觉指令微调：150K GPT-generated multimodal instruction-following data + ~515K academic VQA datasets（OK-VQA, TextVQA, GQA, VQA-v2）
  - Benchmark：VQA-v2（test-dev split）、GQA（test-dev split）、ScienceQA-IMG（test set, zero-shot）、TextVQA（validation set）、POPE（test set, COCO random/common/adversarial, avg F1）、MME-Perception、MMBench（development set）、MMBench-CN（中文版）、LAMBADA、English benchmarks（PIQA/StoryCloze16/HellaSwag/WinoGrande/ARC-Challenge/Easy/HeadQA/OpenBookQA/SciQ）、Multilingual benchmarks（xLAMBADA/xStoryCloze/xWinoGrande/xCOPA）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/howard-hou/VisualRWKV（Apache-2.0 license）
  - RWKV 预训练模型：https://huggingface.co/BlinkDL/rwkv-5-world（RWKV-5）、https://huggingface.co/BlinkDL/rwkv-6-world（RWKV-6）
  - 算法 pipeline 伪代码（VisualRWKV forward pass, 推理时逐 token 生成）：
    ```
    # === Step 1: Vision Encoding ===
    # Input: 原始图像 I → CLIP-L (ViT-L/14, 336×336 resolution)
    Z_v = CLIP_ViT_L14(I)  # penultimate layer features: [577, 1024]

    # === Step 2: Projector ===
    H_v = Projector(Z_v)  # 2-layer MLP: [577, D_llm], 576 image tokens

    # === Step 3: Sandwich Prompt ===
    # 格式: [System] [Q_prefix] <image_tokens> [Q_suffix]
    Input = concat([S_embed, Q_prefix_embed, H_v, Q_suffix_embed])
    # Input ∈ R^{L×D_llm}

    # === Step 4: VisualRWKV Blocks (RWKV-6 backbone) ===
    # 根据 layer index 决定 scanning 方向，交替排列
    For layer l = 0..L-1:
        # ---- Data-dependent Token Shift ----
        # ddlerp_α(a,b) = a + (b-a) ⊙ (λ_α + tanh((a+(b-a)⊙μ_α)A_α)B_α)
        r_t = ddlerp_r(x_t, x_prev) @ W_R  # receptance
        g_t = ddlerp_g(x_t, x_prev) @ W_G  # SiLU gate
        k_t = ddlerp_k(x_t, x_prev) @ W_K  # key
        v_t = ddlerp_v(x_t, x_prev) @ W_V  # value

        # ---- Data-dependent Time Decay ----
        d_t = lora_d(ddlerp_d(x_t, x_prev))
        w_t = exp(-exp(d_t))  # ∈ (0,1), channel-wise dynamic decay

        # ---- WKV Linear Attention (per-head, head_dim=64) ----
        # Recurrent state update:
        wkv_cur = diag(u) @ k_t^T @ v_t       # current token bonus
        wkv_state = diag(w_t) @ wkv_state + k_t^T @ v_t  # accumulated past
        wkv_total = wkv_cur + wkv_state

        # LayerNorm per head + output gating:
        o_t = concat(SiLU(g_t) ⊙ LayerNorm(r_t @ wkv_total)) @ W_O

        # ---- Channel Mixing (FFN) ----
        r'_t = ddlerp_r'(x_t, x_prev) @ W_R'
        k'_t = ddlerp_k'(x_t, x_prev) @ W_K'
        v'_t = ReLU(k'_t)^2 @ W_V'  # squared ReLU
        c_out = σ(r'_t) ⊙ v'_t

        x_t = x_t + o_t + c_out  # residual

    # === Step 5: Output ===
    logits = LM_Head(x_t)  # next token prediction
    ```

    关键张量形状（7B, D_llm=4096, h=64 heads）：
    - x_t ∈ R^{4096}，ddlerp 矩阵 A_α ∈ R^{4096×32}, B_α ∈ R^{32×4096}（LoRA rank=32）
    - WKV state 矩阵 S ∈ R^{64×64} per head（矩阵状态，替代标量状态）
    - 推理时无 KV cache，仅需维护 L × h 个 64×64 的 state 矩阵（恒定量，与序列长度无关）
    - GPU 内存恒定为 54% of LLaVA-1.5 @ 24K tokens
    - 推理速度恒定 O(1) per token，24K tokens 时 3.98× faster than LLaVA-1.5

    数据依赖 vs 数据独立的对比：
    ```
    # Data-independent (RWKV-5 / VisualRWKV-Base):
    α_t = (μ_α ⊙ x_t + (1-μ_α) ⊙ x_{t-1}) W_α     # μ 固定可学习
    w = exp(-exp(ω))                                   # ω 固定可学习
    wkv_t = sum(diag(w)^{t-1-i} @ k_i^T @ v_i)       # 固定 decay

    # Data-dependent (RWKV-6 / VisualRWKV):
    α_t = ddlerp_α(x_t, x_{t-1}) W_α                  # ddlerp with LoRA
    w_t = exp(-exp(lora_d(ddlerp_d(x_t, x_{t-1}))))  # 动态 decay
    wkv_t = sum(diag(Π w_j) @ k_i^T @ v_i)            # 时变 decay
    ```
