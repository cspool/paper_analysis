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
