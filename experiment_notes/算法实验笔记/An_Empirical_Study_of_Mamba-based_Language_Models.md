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
