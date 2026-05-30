## DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 DuoAttention，将 attention head 分为 Retrieval Heads（关键长上下文处理，需 full attention across all tokens）和 Streaming Heads（主要关注 recent tokens 和 attention sinks，仅需 constant-length KV cache）。核心设计：(1) **基于优化的 Retrieval Head 识别**：为每个 KV head 分配可训练 gate value α_{i,j} ∈ [0,1]，前向 pass 中混合 full attention 和 streaming attention 输出：attn_{i,j} = α_{i,j}·full_attn + (1-α_{i,j})·streaming_attn。在合成 passkey retrieval 数据集上以 L2 distillation loss + L1 正则化项训练 gate values（仅数千参数，所有模型权重冻结），2,000 steps on 8×A100 完成。(2) **部署时二值化**：按阈值 τ（由 sparsity quantile 决定）将 gate value 二值化，高于 τ 为 retrieval head（使用 full KV cache），否则为 streaming head（仅保留 sink + recent tokens，constant memory）。(3) **Head 重排序**：预处理时按 head 类型重排 Q/K/V 投影的输出通道，将 retrieval/streaming heads 分为连续簇，以高效 slicing/concat 替代 scatter/gather。(4) **Chunked Pre-filling 兼容**：streaming heads 的 pre-filling 复杂度从 O(L²) 降至 O(LK)（K 为 chunk size），memory 从 O(L) 降至 O(K)。实验比较 Needle-in-a-Haystack（NIAH）、LongBench（14 任务）、MMLU/MBPP/MT-Bench（短上下文）上与 H2O、TOVA、StreamingLLM、FastGen 在相同 KV cache budget 下的准确率；以及单 A100 上不同 context length 下的 decoding/pre-filling latency 和 memory。

- 硬件平台是什么，配置是什么。
  Retrieval head identification training: 8× NVIDIA A100 GPU servers。Decoding/pre-filling efficiency 评测: 单张 NVIDIA A100 GPU（80GB）。默认数值格式：BFloat16 权重和激活。KV cache pre-allocation 避免动态内存分配开销。结合量化时使用 QServe（8-bit weight + 4-bit KV cache quantization）。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-2-7B-chat、Llama-2-7B-32K-Instruct（MHA，32 heads/layer）、Llama-3-8B-Instruct、Llama-3-8B-Instruct-Gradient-1048k、Llama-3-70B-Instruct（GQA，8 KV heads/layer）、Mistral-7B-Instruct-v0.2（GQA，8 KV heads/layer）。
  Retrieval head identification 训练数据：BookSum 数据集嵌入 10 个 32-word passkeys，50 个长度区间（1K tokens → 模型最大长度），passkeys 随机插入 1000 个位置。
  Benchmark：
  - Long-context: Needle-in-a-Haystack (NIAH, 至 1048K tokens)、LongBench（21 任务含 Single-Doc QA, Multi-Doc QA, Summarization, Few-shot Learning, Synthetic, Code）
  - Short-context: MMLU（1-shot）、MBPP（0-shot）、MT-Bench（0-shot）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源地址：https://github.com/mit-han-lab/duo-attention。基于 PyTorch + FlashInfer（RoPE/RMSNorm kernels）+ FlashAttention-2。

  **Phase 1: Retrieval Head Identification（训练，仅优化 gate values）**
  ```
  # 初始化
  α_{i,j} = 1.0 for all heads  # 初始假设所有 head 都是 retrieval head
  optimizer = AdamW(lr=0.02, warmup 400 steps 0.002→0.02, decay 400 steps 0.02→0.002)

  # 合成数据集构造
  for each sample:
      context = BookSum excerpt (1K→model_max_len tokens)
      for i in 1..10:
          insert random 32-word passkey at random position in context
      target = recall all 10 passkeys
      # 仅计算最后 l 个 passkey token 的 loss

  # 前向 pass (per KV head j in layer i)
  full_attn = softmax(Q @ K^T ⊙ M_causal) @ V          # 标准 causal attention
  streaming_attn = softmax(Q @ K^T ⊙ M_streaming) @ V  # Λ-like mask: 仅 sink + recent tokens
  attn_{i,j} = α_{i,j} · full_attn + (1-α_{i,j}) · streaming_attn  # 混合输出

  # Loss
  L_distill = (1/N) Σ_i Σ_{j=T-l+1}^T (H_full[i][j] - H_mixed[i][j])²  # L2 on last hidden states
  L_reg = Σ_i Σ_j |α_{i,j}|                                                # L1 sparsity
  L = L_distill + 0.05 · L_reg

  # 可训练参数仅数千个浮点数（N_layers × N_heads），所有模型权重冻结
  # 使用 FSDP2 + DeepSpeed Ulysses sequence parallelism 支持长序列
  # 2,000 steps on 8×A100 完成
  ```

  **Phase 2: Deployment**
  ```
  # 二值化（按 sparsity quantile 阈值 τ）
  for each head (i,j):
      if α_{i,j} > τ:
          type_{i,j} = "retrieval"    # full attention, 全量 KV cache
      else:
          type_{i,j} = "streaming"    # streaming attention, constant KV cache

  # Head 重排序 — 预处理时重排 Q/K/V 投影权重
  # 将 retrieval heads 和 streaming heads 分为两个连续簇
  # 推理时使用 slicing/concat 而非 scatter/gather

  # Decoding（per layer）
  Q_ret, Q_str = split(Q, head_dim)  # 沿 head 维度切分
  K_ret = full_KV_cache_ret          # 全量历史 KV
  V_ret = full_KV_cache_ret
  K_str = const_KV_cache_str         # 仅 sink (64 tokens) + recent (256 tokens)
  V_str = const_KV_cache_str

  out_ret = FlashAttention(Q_ret, K_ret, V_ret)      # full attention for retrieval heads
  out_str = FlashAttention(Q_str, K_str, V_str)      # streaming attention for streaming heads
  output = concat([out_ret, out_str], head_dim) @ W_O

  # Chunked Pre-filling（streaming heads 优化）
  for each chunk of K tokens:
      K_chunk, V_chunk = compute_KV(chunk)
      # 仅保留 sink + recent tokens，其余立即 evict
      K_str = prune_to_sink_and_recent(K_str, K_chunk)
      V_str = prune_to_sink_and_recent(V_str, V_chunk)
      # 下一 chunk 仅需 attend 到 constant 数量的 contextual tokens
  # 复杂度: time O(LK) instead of O(L²), memory O(K) instead of O(L)
  ```

  **张量维度**（以 Llama-2-7B MHA 为例，25% retrieval ratio）：
  - Full KV cache: 所有 32 heads × 32K tokens × 128 dim × 2 (K+V) × 2 bytes (BF16) = ~512 MB
  - DuoAttention KV cache: 8 retrieval heads × 32K × 128 × 2 × 2 = ~128 MB + 24 streaming heads × (64+256) × 128 × 2 × 2 = ~39 MB → 总计 ~167 MB（节省 ~2.55×）

  **关键配置**：
  - Llama-2-7B (MHA): retrieval ratio 25%（可选更低至 ~10% for 2.55× memory/2.18× latency reduction）
  - Llama-3-8B (GQA): retrieval ratio 50%（可选更低至 ~50% for 1.67× memory/1.50× latency reduction）
  - Sink tokens: 64（deployment, from 128 in identification）
  - Recent tokens: 256（deployment, from 256 in identification）
  - Pre-filling chunk size: 32,000

  **结合量化**：DuoAttention + QServe (W8A8KV4) → Llama-3-8B 单 A100 容纳 3.3M tokens（6.4× capacity increase vs full attention BF16）。
