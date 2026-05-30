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
