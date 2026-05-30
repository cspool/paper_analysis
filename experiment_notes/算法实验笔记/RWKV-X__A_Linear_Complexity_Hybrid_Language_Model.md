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
