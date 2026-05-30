## Why Attend to Everything? Focus is the Key (Composing Sparse Attention via Learned Grouping)

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现 Focus 推理时的 FlashAttention 分解（Appendix D），将 Focus 稀疏注意力 mask 分解为两个不相交的 FlashAttention 调用，无需编写任何自定义 CUDA kernel。注意力 mask: M(i,j) = 1[j≤i] ∧ (1[g(i)=g(j)] ∨ 1[i-j≤w])。分解为：
  - A 集合（same-group causal）：{(i,j): j≤i ∧ g(i)=g(j)} — 按 group 对 token 做 stable sort（保持 causal order），reshape 为 K 个独立序列，对每个调用 flash_attn_func(causal=True)。复杂度 O(n²/K)。
  - B 集合（cross-group local）：{(i,j): j≤i ∧ i-j≤w ∧ g(i)≠g(j)} — 对每个 query 提取 local key，mask 同组 pair 为 -∞。复杂度 O(nw)。
  合并：o[i] = (e^{ℓA[i]}·oA[i] + e^{ℓB[i]}·oB[i]) / (e^{ℓA[i]} + e^{ℓB[i]})，其中 ℓA, ℓB 为 per-query logsumexp。两集合互斥且完备（A∩B=∅, A∪B=M），merge 数学精确（cosine similarity 1.0000 验证）。

  实验比较：(1) Wall-clock speedup（Table 6）：H100-80GB 上 Focus 相比 full attention (均使用 FlashAttention) 在 1K 到 1M 上下文下的加速比，K=4 时 0.2×→4.1×，K=8 时 0.2×→8.6×；(2) Speed-quality tradeoff（Table 7）：不同 top-k (1/2/3/4) 下的 PPL 与 speedup 关系，top-k=2 时 2× 加速 + PPL 改善；(3) 短序列开销：sort 和 gather/scatter 约 12ms 常数开销，序列 ≤4K 时无加速。

- 后端平台是什么，配置是什么。
  NVIDIA H100-80GB GPU。使用 PyTorch + FlashAttention（flash_attn_func），无自定义 CUDA kernel / Triton / 编译。完整实现约 320 行 Python。

- 评估性能的软件/脚本是什么。修改了什么。
  使用 flash_attn_func（FlashAttention-2/3）进行标准注意力计算。未修改 FlashAttention 本身；修改的是 attention mask 的分解方式——将 Focus 的分组稀疏 mask 通过 token sort + group reshape 转化为标准 FA 调用。具体修改：(1) 实现 stable sort by group 保留 causal order；(2) 实现 disjoint decomposition 避免 double-counting；(3) 实现 logsumexp merge 数学精确合并两个 FA 输出。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未明确提供开源链接。

  评估原理与流程（Focus FlashAttention 分解）：
  ```
  # 输入
  q, k, v: [batch, heads, T, d_head]  # 标准 QKV
  group_ids: [T]                       # 每个 token 的 group assignment (0..K-1)
  w: int                               # local window size

  # === A 集合：same-group causal attention ===
  # Step 1: Stable sort by group (保持 causal order)
  sorted_idx = argsort(group_ids, stable=True)     # [T]
  reverse_idx = argsort(sorted_idx)                # inverse permutation
  q_A = q[:, :, sorted_idx, :]
  k_A = k[:, :, sorted_idx, :]
  v_A = v[:, :, sorted_idx, :]

  # Step 2: Reshape into K sequences, pad to same length
  # group_sizes: [K], max_len = max(group_sizes)
  q_A_padded = pad_and_reshape(q_A, group_sizes)   # [K, batch, heads, max_len, d_head]
  k_A_padded = pad_and_reshape(k_A, group_sizes)
  v_A_padded = pad_and_reshape(v_A, group_sizes)

  # Step 3: FlashAttention per group
  o_A_parts = []
  lse_A_parts = []
  for k_idx in 0..K-1:
      # flash_attn_func 内部: QK^T/√d → softmax → ×V
      # 复杂度 O(max_len^2)，总复杂度 O(K·(n/K)^2) = O(n^2/K)
      o_g, _, lse_g = flash_attn_func(
          q_A_padded[k_idx], k_A_padded[k_idx], v_A_padded[k_idx],
          causal=True
      )
      o_A_parts.append(o_g)
      lse_A_parts.append(lse_g)

  # Step 4: Unpad and unsort
  o_A = unsort(concat_and_unpad(o_A_parts, group_sizes), reverse_idx)
  lse_A = unsort(concat_and_unpad(lse_A_parts, group_sizes), reverse_idx)

  # === B 集合：cross-group local attention ===
  # Step 5: 为每个 query 按 group 构造 local mask
  # 仅当 i-j ≤ w 且 g(i) ≠ g(j) 时保留
  o_B, _, lse_B = flash_attn_func(
      q, k, v, causal=True, window_size=(w, 0),
      # 内部：对同组 token 设 attn_mask = -inf
      # 复杂度 O(nw)
  )
  # 注：B 集合同组 local 被 mask 掉后需与 A 合并
  # 论文中 B 的实际实现是对 cross-group local 的计算
  # 合并 A 与 B: o = merge(lse_A, o_A, lse_B, o_B)

  # === Merge (logsumexp 空间) ===
  def merge(lse_A, o_A, lse_B, o_B):
      w_A = exp(lse_A) / (exp(lse_A) + exp(lse_B))
      w_B = exp(lse_B) / (exp(lse_A) + exp(lse_B))
      return w_A * o_A + w_B * o_B
  ```

  性能评估原理：测量 end-to-end wall-clock time（含 sort ~12ms + 两个 FA kernel launch + merge），与 full attention（单次 FA 调用）对比。理论加速 K×：K 个 group 各 attend n/K token，K·(n/K)²=n²/K。实测 K=4 达 4.1×，K=8 达 8.6×，略超理论值因 FA 在较短 per-group 序列上更高效。总代码量 320 行 Python，无需自定义 CUDA/Triton kernel。

## TailorKV: A Hybrid Framework for Long-Context Inference via Tailored KV Cache Optimization

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现两个 CUDA kernel 和异步数据传输管线用以支持混合 KV Cache 压缩：(1) **1-bit 量化 kernel**：对 quantization-friendly 层的 KV cache 执行 per-channel key + per-token value 的 1-bit 静态量化，group size=64，zero-point 和 scaler 存 FP16；(2) **FP16×INT1 GEMV kernel**：实现 FP16 输入矩阵与 INT1 权重矩阵的矩阵乘 kernel，支持高压缩率下的高效解码，通过直接使用 FP16×INT1 矩阵乘法替代 FP16×FP16，减少计算量并充分利用内存带宽；(3) **DGL-based 直接行传输**：使用 Deep Graph Library (DGL, Wang et al., 2019) 的稀疏 tensor API 直接从 CPU tensor 按行索引传输选中的 KV cache rows 到 GPU 设备，避免先 gather 到 CPU 连续内存再传输的中间步骤（PQCache 的做法）；(4) **异步 double buffering 预取管线**：在 decoding 阶段，layer l-1 计算时异步预取 layer l 的 critical key cache（基于 inter-layer query 预估），使用读写双缓冲区实现 computation 与 CPU-GPU communication 的 overlap。

  实验比较：(1) Latency breakdown 对比（Figure 8）：在 A100 上单 Transformer block 16k 序列长度下，TailorKV 相比 PQCache 降低 retrieval latency 27.8% (GQA) / 40.5% (MHA)，降低 data transfer latency 83.5% (GQA) / 82.2% (MHA)；(2) End-to-end decoding latency 对比（Table 4, Table 13）：TailorKV 延迟接近 Full Cache (FlashAttention-2)，相比 OffloadCache 加速数十倍；(3) 不同 critical channel 数量 (2/4/8/12) 的性能/latency trade-off（Figure 9c）。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 3090 (24GB, PCIe 1.0, 4GB/s) + Intel Xeon Gold 6240；NVIDIA A100 (80GB, PCIe 4.0, 32GB/s) + Intel Xeon Platinum 8369B。使用 PyTorch 2.4.0 + FlashAttention-2.6.3 + DGL。多线程执行异步任务，实现 GPU 计算与 CPU→GPU 数据传输的 overlap。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 HuggingFace Transformers 4.46.1 的推理脚本修改。主要修改在 Transformer 的 forward pass 中：
  1. 在 decoder layer 的 attention 模块中插入混合压缩逻辑——根据离线确定的层类型 C(l) 分支：quantization-friendly 层执行 per-layer 量化 + FP16×INT1 GEMV；sparsity-friendly 层执行动态检索（critical channel selection + CPU-GPU 异步预取 + Top-K token fetch）
  2. 使用 DGL 的稀疏 tensor row-wise transfer API（dgl.sparse.from_coo / dgl.ops.gspmm）将 CPU tensor 中指定行索引的数据直接传输到 GPU
  3. 实现 CUDA kernel 的 1-bit quantize/dequantize GEMV，用于 quantization-friendly 层的 attention 计算
  4. prefilling 阶段每层 KV cache 逐层 offload 到 CPU memory

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/ydyhello/TailorKV（ACL 2025 Findings，代码已发布）。

  **评估原理**：使用 PyTorch 模型推理 + `torch.cuda.Event` / `time.perf_counter()` 测量 per-layer 和端到端 latency。peak memory 通过 `torch.cuda.max_memory_allocated()` 记录。latency breakdown 在 A100 上单测每个 Transformer block 各操作的耗时分布（attention compute vs CPU-GPU transfer vs retrieval computation）。

  **Kernel 执行流程**（sparsity-friendly 层的动态检索）：
  ```
  // === Stage 1 (at layer l-1, on GPU) ===
  h_l_minus_1 = hidden_state  // shape: (1, d)
  q_hat = W_q[l] @ h_l_minus_1  // 预估算当前层 query, shape: (d_h,)
  // 计算 channel-wise importance
  s = abs(q_hat) * max(abs(K_cpu), dim=seq)  // shape: (d_h,)
  critical_ch = argsort(s)[-d_s:]  // 选 d_s 个 critical channels
  // 启动异步预取: CPU→GPU 传输 critical key cache
  // K_critical shape: (d_s, n), 使用 double buffering

  // === Stage 2 (at layer l, on GPU) ===
  q = W_q[l] @ h_l  // 真实 query
  q_crit = q[critical_ch]  // shape: (d_s,)
  // K_critical 已预取完成（double buffer read）
  a_approx = q_crit @ K_critical.T  // 近似 attention, shape: (1, n)
  topk_idx = TopK(a_approx, k=n_topk)  // GPU 端排序选 Top-K

  // Fetch Top-K tokens: CPU→GPU, 传输完整 K[topk_idx], V[topk_idx]
  // 唯一不可 overlap 的操作——取决于当前层 query

  // 完整 attention: concat(local_KV, fetched_KV) → FlashAttention
  output = flash_attn(q, cat(local_K, fetched_K), cat(local_V, fetched_V))
  ```

## SageAttention2++: A More Efficient Implementation of SageAttention2

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现 CUDA kernel，在 SageAttention2 的基础上将 P×V 矩阵乘法的 MMA 指令从 mma.f32.f8.f8.f32（FP32 累加器）替换为 mma.f16.f8.f8.f16（FP16 累加器）。具体改动：(1) FP8 量化 scale factor 重新设定：$δ_P = \max(|P̃|)/224$, $δ_V = \text{colmax}(|V|)/4.5$，使量化后值满足 $|32 × p × v| ≤ 65504$；(2) Delayed FP32 Buffering：连续两次 mma.m16n8k32 结果在 FP16 中累加后再执行 FP32 类型转换 PTX 指令，减少转换开销。

  实验比较：(1) Kernel speed benchmark：RTX4090 和 RTX5090 上，headdim=64/128，带/不带 Causal Mask，对比 FlashAttention2、SageAttention、SageAttention2 的 kernel 吞吐量（图 1-4）；(2) 端到端模型指标（Table 3）：LLaMA3.1-8B、CogvideoX-2B、HunyuanVideo、Wan、Flux、Stable-Diffusion3.5。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 4090 GPU 和 NVIDIA RTX 5090 GPU。RTX 4090 基于 Ada Lovelace 架构，RTX 5090 基于 Blackwell 架构。两者均支持 FP8 数据类型和 mma.f16.f8.f8.f16 指令（FP8 Tensor Core with FP16 accumulator）。注意 FlashAttention3 仅能在 Hopper GPU (H100/H800) 上运行，RTX4090/5090 不支持，所以 FlashAttention2 是这些消费级 GPU 上的最快 baseline。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 SageAttention2 的 CUDA kernel 代码库修改。在 SageAttention2 的 P×V Matmul kernel 中：
  1. 将 MMA 指令从 mma.f32.f8.f8.f32（PTX: mma.sync.aligned.m16n8k32.row.col.f32.f8.f8.f32）替换为 mma.f16.f8.f8.f16（PTX: mma.sync.aligned.m16n8k32.row.col.f16.f8.f8.f16）
  2. 修改 FP8 量化 scale factor 计算：$\max(|x|)/P_r$ 和 $\max(|x|)/V_r$，其中 $P_r=224, V_r=4.5$
  3. 实现 Delayed FP32 Buffering：每两次 MMA 结果在 FP16 中累加后统一 convert 到 FP32
  Kernel 使用 CUDA C++ 编写，直接调用 PTX 内联汇编实现 Tensor Core 指令。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/thu-ml/SageAttention（SageAttention2++ 合入同一仓库）。

  **评估原理**：使用 CUDA Event (cudaEventRecord) 测量 attention kernel 执行时间（latency），在多种 (batch_size, seq_len, num_heads, head_dim) 配置下对比各方法的 kernel 耗时。端到端评估使用 PyTorch 模型推理 + torch.cuda.Event 计时。

  **Kernel 输入**：Q, K, V ∈ R^{batch×heads×seq×headdim}，已按 FlashAttention 风格 tiling。Q,K 已量化为 INT4/INT8（per-block），P̃ 和 V 待 FP8 量化。

  **Kernel 执行流程**（P×V 部分）：
  ```
  // 输入：P̃ (FP16/FP32), V (FP16), 已分块为 P̃_i, V_i
  for each block (P̃_i, V_i) on SM:
      // 1. 计算 per-block scale factors
      δ_P = max(|P̃_i|) / 224        // 缩小的量化范围
      δ_V = colmax(|V_i|) / 4.5     // 缩小的量化范围

      // 2. FP8 量化 (E4M3)
      P̂_i = cvt_fp8_e4m3(P̃_i / δ_P)   // 范围约束在 [-224, 224]
      V̂_i = cvt_fp8_e4m3(V_i / δ_V)   // 范围约束在 [-4.5, 4.5]

      // 3. Tensor Core MMA with FP16 accumulator
      acc_fp16 = 0 (FP16)
      for k in range(K_dim / 32):
          p_tile = P̂_i[k*16:(k+1)*16, :]   // 16×32 FP8 tile
          v_tile = V̂_i[k*32:(k+1)*32, :]   // 32×8 FP8 tile
          // mma.sync.aligned.m16n8k32.row.col.f16.f8.f8.f16
          acc_fp16 += mma_f16_f8_f8_f16(p_tile, v_tile)

      // 4. Delayed FP32 Buffering: 两次 MMA 后才转 FP32
      if (mma_count % 2 == 0):
          acc_fp32_tmp = cvt_f16_to_f32(acc_fp16)
          acc_fp32 += acc_fp32_tmp
          acc_fp16 = 0

      // 5. 反量化
      O_i = acc_fp32 * δ_P * δ_V
  ```

  **性能结果**：
  - RTX4090, headdim=128: SageAttn2++(4+8) ≈ 3.9× FlashAttention2, SageAttn2++(8+8) ≈ 3.0× FlashAttention2
  - RTX4090, headdim=64: 类似加速趋势
  - RTX5090 上加速效果更显著（Blackwell 架构对 FP8 支持更好）

## Rectified Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  ReSA 实现了一个基于 Flash Decoding split-execution 策略的自定义 CUDA kernel，用于加速 GBSA（Group Block Sparse Attention）的 decoding 阶段。核心设计：
  (1) **GQA-aware SM 分配**：每个 GQA group 分配到一个独立的 streaming multiprocessor (SM)，确保高效利用 GPU 资源和最小化 SM 间通信。
  (2) **Block-level workload splitting**：decoding workload（batch_size × num_kv_heads）按 block indices 维度拆分到所有可用 SM。每个 step 激活 k 个 memory block，这些 block 被均匀分到约 k/split 个 block per SM——每个 SM 独立 fetch 对应 KV 并执行 sparse attention。
  (3) **Shared KV fetching**：intra-GQA key 共享机制，同一 GQA group 内的 query heads 复用加载的 KV cache block，减少 HBM 访问。
  (4) 使用 TileLang 库实现（论文 Acknowledgments 致谢 TileLang 团队）。

  实验比较：
  (a) Kernel-level latency breakdown（Section 3.5.1）：16K/64K/256K context 下 ReSA vs Dense FlashAttention 的 CUDA kernel 执行时间对比，延迟分解为 sparse estimation、attention computation、rectification 三部分。
  (b) End-to-end throughput（Section 3.5.2）：FP16 和 INT4（Marlin kernel）下，4K/16K/64K/256K context 的 end-to-end speedup。结果表明 256K 下 ReSA 达 FP16 2.28×、INT4 2.44× speedup。

- 后端平台是什么，配置是什么。
  NVIDIA A100-80G GPU。kernel 参考 Flash Decoding（crfm.stanford.edu/2023/10/12/flashdecoding.html）的 split-execution 策略实现。INT4 实验使用 Marlin kernel 进行 low-bit matmul（group-wise scaling, group size=128）。Custom decoding kernel 使用 TileLang 库实现。Batch size 默认为 8。评测时仅报告 CUDA kernel 时间，排除 CPU overhead。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch 框架和自定义 CUDA kernel（通过 TileLang 实现）。修改内容：
  (a) 在 Flash Decoding kernel 基础上增加 block-sparse attention 支持：在 split-execution pipeline 中，外层循环不再遍历全部 KV blocks，而是只遍历由 GBSA block selection 选中的 top-n block indices（算法见 Appendix A）。
  (b) GQA group SM 绑定：每个 SM 固定处理一个 GQA group 的 query，减少 SM 间 KV 传输。
  (c) Block key cache 在线更新：每个 decode step 后增量更新新 token 所在 block 的 min/max 描述符，无需从头计算。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源在 https://aka.ms/ReSA-LM。kernel 与 Flash Decoding 的集成见论文 Appendix A 的 Algorithm 2。

  **评估原理**：测量 CUDA kernel 执行时间（latency）和 token 生成吞吐量（throughput）。延迟排除 CPU scheduling overhead，仅计 kernel 实际执行。Throughput 测量包括完整 decode step（sparse + rectification 摊销）。

  **Kernel 输入**：Queries Q ∈ R^{h×g×n×d}，Keys K ∈ R^{h×n×d}，Values V ∈ R^{h×n×d}，block_indices（由 GBSA block selection 产生），num_splits（SM 拆分粒度）。

  **Kernel 执行流程**（Flash Decoding + Block-Sparse）：
  ```
  for each (num_splits, num_kv_heads, batch_size) in grid:
      1. Load query vectors q in a GQA group
      2. Compute partial_block_indices = block_indices[split_slice]
         # 只取当前 split 负责的 block 子集
      3. Initialize accumulators: mi ← -∞, li ← 1.0, acc ← 0
      4. for each block_id in partial_block_indices:
           Load k, v from KV cache at block block_id
           qk = q @ k^T * sm_scale
           qk[invalid_positions] = -1e6
           Update mi, li, acc (online softmax)
      5. Store partial logsum and attention outputs
  end for
  6. Combine(logsum_partial, out_partial) → final output O
  ```

  **性能结果**：
  - 256K context：FP16 2.28×，INT4 2.42× end-to-end speedup vs dense FlashAttention
  - Rectification overhead：256K 时占 attention 总延迟的 32.7%，64K 时 28.9%
  - Sparse estimation 和 attention computation 耗时相当（均 ≈ mem(KV cache) × p ≈ mem(KV cache) × 1/b）

## ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  ReCalKV 使用 Triton 实现了自定义 fused attention kernel（Section 4.5），将低秩压缩的 Key 路径和 Value 路径集成到单个 kernel 中：
  (1) **Key 路径的 HSR 在线置换**：HSR 对 head 的重排序在推理时需要通过在线 permutation 来恢复原始 head 顺序，该置换操作在 fused kernel 中作为运行时步骤对每个 token 执行。
  (2) **Value 路径的离线 Matrix Fusion**：R_v 已预先融合进 W_o（W_o_fused = R_v·W_o），kernel 只需计算低秩 Value latent z_v = x·L_v 后直接与 fused output projection 相乘，无需在线重建完整 Value。
  (3) **RoPE 兼容性**：kernel 支持 rotary position embedding（RoPE），并保持与 causal attention 的完全兼容。

  实验比较：在单张 NVIDIA A800 GPU 上，测量 4K、16K、65K 三种 prompt 长度下的单次 attention 模块延迟（100 次运行取平均），对比 baseline full attention 的加速比。结果：70% 压缩率下，4K 加速 1.22×，16K 加速 1.59×，65K 加速 1.80×——压缩率越高、prompt 越长，加速越显著。

- 后端平台是什么，配置是什么。
  NVIDIA A800 GPU。Triton 语言实现 fused attention kernel。模型使用 LLaMA-2-7B 等标准 Transformer 架构（MHA 和 GQA 均支持）。Kernel 在 causal attention 模式下运行，集成 RoPE。

- 评估性能的软件/脚本是什么。修改了什么。
  使用 Triton 编写自定义 fused attention kernel。修改内容：
  (a) 将低秩 Key 压缩（HSR with grouped SVD）的在线 head permutation 步骤集成到 attention kernel 的前向路径中——每个 token 的 query/key 计算后，先对压缩后的 key head 执行 inverse reordering 恢复原始 head 顺序，再应用 RoPE 位置编码和 attention score 计算。
  (b) 将 Value 压缩的离线 Matrix Fusion 结果（W_o_fused = R_v·W_o）作为 static weight 嵌入 kernel，跳过显式的 Value 重建步骤。
  (c) Kernel 以 block-sparse 形式执行低秩重建，减少全局内存访问。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码将在 https://github.com/XIANGLONGYAN/ReCalKV 发布。评估原理和 kernel 执行流程：

  **评估原理**：对比 ReCalKV 的 fused attention kernel 与标准 FlashAttention 在相同 prompt 长度下的端到端延迟。延迟测量基于单次 attention 模块（不含 embedding、FFN 等其他组件），排除模型其他部分的噪声。测量采用 100 次运行取平均的方式。

  **Kernel 输入**：Input hidden states X ∈ R^{seq_len × d_model}，预压缩权重 L_k (per head group) 和 R_k (per head group)，Value latent 投影 L_v，fused output projection W_o_fused = R_v·W_o，以及原始 Q 投影 W_q。

  **Kernel 执行流程**（Triton fused kernel）：
  ```
  1. Q = X @ W_q                           # 标准 query 投影
  2. K_latent = X @ L_k                    # Key 低秩投影 [seq, r_k]
     K_full = K_latent @ R_k               # 重建完整 key
     K_reordered = inverse_reorder(K_full)  # 在线 HSR inverse permutation
     K_rope = apply_rope(K_reordered)       # RoPE 位置编码
  3. V_latent = X @ L_v                    # Value 低秩投影 [seq, r_v]
     存入 KV cache: V_latent（而非完整 V）
  4. S = Q @ K_rope.T / sqrt(d_k)          # attention scores
     A = softmax(S, causal=True)
  5. Output = A @ V_latent @ W_o_fused     # fused: 无需重建完整 Value
     # W_o_fused = R_v @ W_o (预计算，offline)
  ```
  加速来源：(a) 推理时 Key 从 `X·W_k` 变为 `X·L_k` + 重建，L_k 更窄（低秩）；(b) Value cache 存储 low-rank latent 而非完整 Value，内存占用从 d_h 降至 r_v；(c) Output projection 已预融合（W_o_fused），消除在线重建步骤。内存访问减少在高压缩率（70%）和长 prompt（65K）时尤为明显，达到 1.80× 加速。

## PowerAttention: Exponentially Scaling of Receptive Fields for Effective Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  POWERATTENTION 使用 PyTorch FlexAttention（Dong et al., 2024）库实现自定义稀疏 attention mask，结合 Triton（Tillet et al., 2019）实现 kernel 级别的优化和序列并行训练（RingAttention, Liu et al., 2024）。核心 kernel 实现是通过 FlexAttention 的编程模型将自定义 mask（power-of-2 + window + sink）编译为优化的 GPU kernel，使用 256-token blocks 对齐 CUDA compute cores 的内存访问模式。FlexAttention 自动将 mask 定义转换为 block-sparse attention kernel 的 tiling 和内存访问策略。

  实验比较：(a) Kernel 前向时间对比：POWERATTENTION vs Full Attention vs MInference，测量 16K-128K context 下每次 attention forward pass 的时间（Figure 4b），128K 时 POWERATTENTION kernel 比 MInference 快 5.3×，比 Full Attention 快 21.6×；(b) 端到端延迟：prefilling 阶段和 decoding 阶段（1024 steps）的完整推理延迟（Figure 4a），128K 时 prefilling 比 Full Attention 快 3.0×，decoding 仅需 Full Attention 58% 的时间。

- 后端平台是什么，配置是什么。
  NVIDIA A800 GPU。模型 Qwen2-7B（28 layers, 32K context）。Kernel 配置：block_size=256 tokens（对齐 GPU compute core 内存访问），sparsity ratio ≈ 94%（每个 token 最多关注 ~10 blocks）。POWERATTENTION 内核因 O(N log² N) 时间复杂度，增长曲线接近滑动窗口的线性复杂度。

- 评估性能的软件/脚本是什么。修改了什么。
  使用 PyTorch FlexAttention 库定义稀疏 attention mask，Triton 结合 RingAttention 用于序列并行以扩展到更长序列。修改内容：(a) 使用 FlexAttention 的 `create_block_mask` 或等效接口定义 power-of-2 mask pattern；(b) 将 POWERATTENTION 的 mask（sink + window + power）编译为 block-sparse kernel，利用 FlexAttention 自动将 mask 映射到 GPU tiling；(c) 通过 Triton kernel 将 KV cache 分块在序列维度上并行计算。MInference 仅在 prefilling 阶段使用（按原论文建议），decoding 阶段回退到 FlashAttention。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未明确提供开源链接。实现基于 PyTorch FlexAttention 库。kernel 执行原理如下：

  **评估原理**：FlexAttention 提供一种编程模型，用户只需用 PyTorch 张量操作定义 attention score mask（`score_mod` 函数），框架自动将其编译为融合的 block-sparse CUDA kernel。对于 POWERATTENTION，mask 定义在 block 级别（block_size=256），因此所有计算和内存访问均为 block-aligned。

  **Kernel 输入**：Query tensor Q [M, d_k]，Key tensor K [N, d_k]（N 个 prefill token 或 1 个 decode token + KV cache），Value tensor V [N, d_v]，block-wise mask（从 FlexAttention score_mod 函数推导）。

  **Kernel 执行流程**（FlexAttention block-sparse 模式）：
  ```
  1. Mask 分析阶段（offline 或首次调用时）:
     - FlexAttention 接收 score_mod(q_idx, kv_idx) 函数
     - 将输入按 block_size=256 分块，预计算哪些 (query_block, kv_block) 对需要计算
     - 生成稀疏的 block 索引列表（仅 mask=1 的 block 对）
     - POWERATTENTION 的 block mask：sink(block 0) + window(5 blocks) + power-of-2 blocks
  
  2. Kernel 执行阶段（Triton grid）:
     for each query_block (Grid-level parallel):
       load Q_block [B_q, d_k] into SRAM
       for each kv_block in sparse_block_list[query_block]:
         load K_block [B_k, d_k], V_block [B_k, d_v] into SRAM
         S = Q_block @ K_block^T / sqrt(d_k)  # [B_q, B_k]
         online softmax + accumulate: o = softmax_update(S, V_block)
       write o to HBM
  ```
  POWERATTENTION 的 block 数 ≈ O(log n) per query，因此总计算复杂度 O(N log² N)，内存访问量远低于 Full Attention 的 O(N²)，与滑动窗口的 O(N) 接近。128K context 时 kernel 比 Full Attention 快 21.6×。

## Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  NSA 在 Triton 上实现了硬件对齐的稀疏注意力 kernel，专为 GQA/MQA 架构设计。核心 kernel 设计针对 selection attention（compression 和 sliding window 直接复用 FlashAttention-2 kernel）。关键优化：(1) Group-Centric Data Loading：不同于 FlashAttention 按时间连续 query block 加载到 SRAM，NSA kernel 对每个 query 位置 t，将同一 GQA group 内所有 H 个 query head 的 Q ∈ R^{[H, d_k]} 一同加载到 SRAM，因为它们共享相同的稀疏 KV block 索引 I_t；(2) Shared KV Fetching：在内循环中按 I_t 顺序加载连续的 key/value block K ∈ R^{[B_k, d_k]}, V ∈ R^{[B_k, d_v]} 到 SRAM（B_k 为 kernel block size 且 B_k | l'），消除同一 group 内 head 间的冗余 KV 传输；(3) Outer Loop on Grid：将 query/output 循环放到 Triton 的 grid scheduler 中，因为各 query block 的 inner-loop 长度（正比于选中的 block 数 n）几乎一致，有利于 GPU SM 间负载均衡。kernel 通过消除冗余 KV 传输和均衡 SM 负载实现近最优 arithmetic intensity。

  实验比较：(a) NSA Triton kernel vs FlashAttention-2 Triton kernel（同一后端），测量 forward 和 backward latency（8k/16k/32k/64k context），forward 最高 9.0× speedup @64k，backward 最高 6.0× speedup @64k；(b) 解码阶段 memory access volume 对比（Table 4），NSA 在 64k 时仅需加载 ~5632 等效 token 量（vs Full Attention 65536），预期 11.6× speedup。

- 后端平台是什么，配置是什么。
  NVIDIA A100 GPU（8-GPU 系统）。kernel 配置：GQA group=4，每 group heads H=16，d_q=d_k=192，d_v=128。NSA 超参：compression block size=32，stride=16，selected block size l'=64，selected block count n=16，sliding window=512。kernel block size B_k 整除 l'。

- 评估性能的软件/脚本是什么。修改了什么。
  使用 Triton（Tillet et al., 2019）实现 NSA kernel。对比 baseline 为同样用 Triton 实现的 FlashAttention-2 kernel（保证同后端公平比较）。修改内容：(a) 实现了新的 Triton kernel for grouped-query sparse attention，核心修改 query 加载方式——从 FlashAttention 的「按时间连续 query block 加载」改为「按 GQA group 的 query heads 加载」；(b) 内循环按 I_t 索引顺序加载 KV blocks，每条 KV cache line 加载一次后供 group 内所有 heads 共享；(c) 将 query 遍历外移到 Triton grid scheduler 中。compression attention 和 sliding window attention 直接复用 FA2 kernel 无需修改。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文出自 DeepSeek-AI，kernel 使用 Triton 实现但未在论文中提供显式开源链接。kernel 执行原理如下：

  **Kernel 输入**：Query tensor Q_{t} ∈ R^{[H, d_k]}（同一 GQA group 在位置 t 所有 heads 的 query）；稀疏 KV block 索引 I_t（由算法层的 Top-n selection 预先计算）；全局 K/V cache（HBM 中）。

  **Kernel 执行流程**（参见 Figure 3）：
  ```
  Grid Loop (Triton grid scheduler, 每个 program 处理一个 query 位置 t):
    1. 加载 Q_{t} ∈ R^{[H, d_k]} 到 SRAM （Group-Centric Loading）
    2. 初始化 o ∈ R^{[H, d_v]} = 0, l ∈ R^{H} = 0 （online softmax 状态）
    Inner Loop (遍历 I_t 中每个选中的 KV block):
      3. 从 HBM 加载连续 KV block：K_blk ∈ R^{[B_k, d_k]}, V_blk ∈ R^{[B_k, d_v]} 到 SRAM
         （Shared KV Fetching — 同一 group 所有 H heads 共享此加载）
      4. 计算 S = Q_{t} @ K_blk^T / sqrt(d_k) ∈ R^{[H, B_k]}
      5. Online Softmax 更新：m_new = max(m_old, rowmax(S))
         l_new = exp(m_old - m_new) * l_old + rowsum(exp(S - m_new))
         o_new = exp(m_old - m_new) * o_old + exp(S - m_new) @ V_blk
    End Inner Loop
    6. 写出 o = o_new / l_new ∈ R^{[H, d_v]} 到 HBM
  End Grid Loop
  ```

  **关键性能原理**：(a) 算术强度优化——每个 inner loop iteration 中，HBM 加载量 = B_k × (d_k + d_v) 个元素，计算量 = H × B_k × (2d_k + 3d_v) FLOPs（含 online softmax），H=16 时算术强度约为 16× (2d_k+3d_v) / (d_k+d_v) ≈ 14，超过 A100 的 critical arithmetic intensity，从 memory-bound 变为 compute-bound；(b) 消除冗余 KV 传输——同一 GQA group 的 H 个 query heads 共享相同 KV block，FlashAttention 方式会让每个 head 独立加载，NSA kernel 一次加载供 H 个 heads 使用，减 少 H-1 倍冗余；(c) Grid 负载均衡——所有 query position 的 inner loop 长度 = n × (l'/B_k) 几乎恒定，无 warp divergence。

## Multi-head Temporal Latent Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  由于官方 FlashAttention-2 不直接支持 MTLA 的 temporal compressed KV cache 结构，论文扩展了 FlashAttention-2 并实现了自定义 CUDA kernel 用于 MTLA 推理。核心改动：CUDA kernel 需要适配 MTLA 的 compressed KV cache Ĉ ∈ R^{t×r}（而非标准 attention 的 K, V ∈ R^{T×(n_h·d_h)}），用 Ĉ 直接参与 attention 计算，避免了显式 up-projection 到 K, V 的计算开销。吸收权重后，kernel 计算逻辑：scores = (X @ W_Q_absorbed) @ Ĉ^T / sqrt(d_h)，其中 W_Q_absorbed = W_Q @ W_K^T ∈ R^{d×r} 预计算，W_V_absorbed = W_V @ W_O ∈ R^{r×d} 同理。

  实验比较：(a) MTLA + extended FlashAttention-2 vs MHA + FlashAttention-2 on ST task（BLEU, inference time, GPU memory）；(b) MTLA + FlashAttention-2 vs MHA baseline（无 FlashAttention）。结果：MTLA w/ FlashAttention-2 相比 MHA w/ FlashAttention-2 实现 3.99× speedup（36.5s vs 145.7s），GPU 内存降低 7.34×（1259 MiB vs 9244 MiB），且 BLEU 略有提升（23.29 vs 23.16）。

- 后端平台是什么，配置是什么。
  单张 NVIDIA RTX 6000 Ada GPU（48GB, bfloat16 推理）。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 Fairseq toolkit + 自研 CUDA kernel（参考文献 [1] 指向 https://github.com/D-Keqi/mtla）。核心修改：

  1. **Custom CUDA kernel for MTLA inference with FlashAttention-2**：官方 FlashAttention-2 kernel 的设计假设 input K, V 序列长度与 Q 相同（标准 MHA），MTLA 的 K, V 需由 compressed Ĉ 经 absorption up-projection 得到（长度 t = T/s < T）。自定义 CUDA kernel 需：
     - 以吸收后的权重矩阵 W_Q_absorbed ∈ R^{d×r} 和 W_V_absorbed ∈ R^{r×d} 作为 kernel 输入
     - 对每个 query token 计算 softmax score 时，使用 stride-aware mask（仅允许 m % s == 0 的 KV cache position 被 attend）
     - Tiling 策略需考虑 Ĉ 的压缩比例 s：每 s 个 query 可共享同一个 Ĉ 行的 tiling 模式
     - 内存访问模式：Ĉ 的加载频率降为 1/s（vs 标准 attention 中 K, V 每 query 都加载）

  2. **Absorbed weight pre-computation**（training 后 inference 前）：W_Q @ W_K^T 和 W_V @ W_O 被预计算并存储，避免推理时显式 up-project Ĉ 到完整 K, V 维度再 down-project。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://github.com/D-Keqi/mtla（包含 extended FlashAttention-2 CUDA kernel）。
  
  **MTLA FlashAttention-2 kernel 执行全流程（ST task, s=2, T=1024 speech frames, r=256, d=512, n_h=8）**：

  ```
  输入：
    X ∈ R^{T×d}（或 incremental 时 x_i ∈ R^{1×d}）
    Ĉ ∈ R^{t×r}, t = ceil(T/s) = 512（compressed KV cache）
    W_Q_absorbed = W_Q @ W_K^T ∈ R^{d×r}（pre-computed, 吸收后的 query-key 投影）
    W_V_absorbed = W_V @ W_O ∈ R^{r×d}（pre-computed, 吸收后的 value-output 投影）
    stride s = 2

  Step 1 - Kernel 内存加载:
    将 Ĉ ∈ R^{512×256} 加载到 GPU shared memory（按 FlashAttention 分块策略）
    将 W_Q_absorbed 和 W_V_absorbed 加载到 GPU registers

  Step 2 - Q @ K^T equivalent（无显式 K）:
    对每个 query token i（GPU thread block 级别并行）:
      q_i_absorbed = X[i] @ W_Q_absorbed  # 等价于常规的 q_i @ W_K^T
      # q_i_absorbed ∈ R^{1×r}，而非标准 FlashAttention 中的 R^{1×(n_h·d_h)}
      
    逐 block 加载 Ĉ 的 tiles:
      for each tile of Ĉ[j:j+B]:
        scores_block = q_i_absorbed @ Ĉ[j:j+B]^T / sqrt(d_h)
        # 此处 Ĉ[j:j+B] ∈ R^{B×r} 而非标准 K ∈ R^{B×(n_h·d_h)}
        # r = 256 vs n_h·d_h = 512，减少了 2× 的计算量和带宽

  Step 3 - Stride-aware causal masking（kernel 内联实现）:
    对每个 attention 位置 (m, n):
      if n == m or (n < m and n % s == 0): 保持 scores 值
      else: mask = -inf

  Step 4 - Softmax + V @ O equivalent（无显式 V）:
    # 同样逐 block 加载 Ĉ 的 tiles，但需吸收 W_V_absorbed
    # 标准 FlashAttention: P @ V @ W_O → 需中间 V ∈ R^{t×(n_h·d_h)}
    # MTLA kernel: P @ Ĉ @ W_V_absorbed → 中间 Ĉ ∈ R^{t×r}, r 更小
    for each tile of Ĉ and corresponding P block:
      o_i += P_block @ (Ĉ_block @ W_V_absorbed)
    # Ĉ_block @ W_V_absorbed 可用 shared memory 预计算一次，复用给所有 query

  评估原理：
    - 用 PyTorch 计时器 wraparound 测量完整 encoder-decoder 推理时间
    - 用 nvidia-smi / PyTorch CUDA memory API 采样 average GPU memory usage
    - 对比 MHA/MLA/MTLA 在相同 batch size、beam size 下的 speedup 和 memory reduction
    - 速度提升来自两个维度：(a) KV cache 的 temporal 压缩降低了 per-token attention O(T) → O(T/s)；(b) absorption 避免显式 up-project Ĉ

  性能：
    MTLA + FlashAttention-2: 36.5s, 1259 MiB, BLEU 23.29
    MHA + FlashAttention-2: 145.7s, 9244 MiB, BLEU 23.16
    Speedup: 3.99×, Memory reduction: 7.34×, Quality: +0.13 BLEU
  ```

## MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了三种针对不同稀疏注意力模式的优化 GPU kernel：(1) A-shape kernel（静态稀疏，保留 global tokens + local windows）；(2) Vertical-Slash kernel（混合 block-sparse + column-sparse，两个子 kernel：VS Sparse Index kernel + VS FlashAttention kernel）；(3) Block-Sparse kernel（基于 block-level top-k 选择）。Kernel 基于 Triton 语言 + PIT 动态稀疏编译器 + FlashAttention 实现，针对 A100 GPU 优化。

  实验比较：(a) 三种 kernel vs FlashAttention 的 micro-benchmark latency（Fig. 10）：Block-Sparse 最快（1M context 下 30× speedup），A-shape 次之（10K 下 <1ms，1M 下 164ms），Vertical-Slash 最慢但仍有 13× speedup；(b) 端到端 pre-filling latency breakdown（Fig. 1b）：100K→1.8×, 300K→4.1×, 500K→6.8×, 1M→10× speedup；(c) 三种模式的 sparsity distribution（Fig. 12）：>200K 时实际计算稀疏度 >90%，>500K 时 >95%。

- 后端平台是什么，配置是什么。
  单张 NVIDIA A100 80GB GPU（bfloat16）。Kernel 基于 Triton，可移植到 H100、MI300X。单 A100 优化：Tensor Splitting（按 head 拆分 Attention、按 sequence 维度拆分 MLP）、消除中间变量（mask logic 直接在 kernel 内实现 causal mask）、仅计算最后 token 的 logits。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 FlashAttention Triton 实现 + PIT（Permutation Invariant Transformation）动态稀疏编译器。核心修改：

  1. **Block-Sparse FlashAttention kernel**（Appendix C.4.1）：以 selected block index 为额外输入，每个 thread block 循环遍历每行的 top-K blocks。速度比 $s_p = S / (2B \times k_b)$，B=64 为 block size。

  2. **Vertical-Slash Sparse Index kernel**（Algorithm 4）：对每行 blocks 构建稀疏索引——point-range two-way merge 算法，垂直索引视为 points、斜线索引按行索引转为 ranges。输出 merged ranges（block indexes）+ separate column indexes。时间复杂度 O(k_v + k_s) per row，GPU 并行化。

  3. **Vertical-Slash FlashAttention kernel**（Algorithm 5）：混合 kernel——先循环 block indexes（Block-Sparse FlashAttention 方式），再循环 column indexes grouped by block size（PIT sparse attention 方式）。PIT 将稀疏数据通过 Permutation Invariant Transformation 加载到 dense compute blocks。

  4. **A-shape kernel**：静态稀疏掩码（固定保留 1K global tokens + 4K local window tokens），直接使用 FlashAttention 但仅计算静态掩码内的区域。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://aka.ms/MInference（GitHub）。基于 PyTorch + Triton + PIT + FlashAttention。

  **Kernel 执行全流程（以 Vertical-Slash kernel 为例，LLaMA-3-8B, 128K context）**：

  ```
  输入：
    Q, K, V ∈ R^{131072×128}（S=128K, d_h=128）
    i_v ∈ N^{30}（top-30 垂直列索引）
    i_s ∈ N^{2000}（top-2000 斜线索引）
    block_size B = 64

  Step 1: 在线稀疏索引构建（Vertical-Slash Sparse Index kernel, Algorithm 4）
    N = ceil(S/B) = 2048 行 blocks
    GPU 并行 for i ← 1 to N:
      # 找到当前行 i 对应的斜线范围
      j_s ← biset_left(i_s, i×B)         # 二分查找第一条穿过行 i 的斜线
      r_start ← (i-1)×B - i_s[j_s]
      r_end ← i×B - i_s[j_s]
      # Point-range two-way merge
      while 垂直列和斜线范围存在:
        如果垂直列在范围外 → 记录 column index
        如果斜线范围结束 → 记录 block index 并更新范围
      输出: c_blk^i（block count）, i_blk^i（block indices）,
            c_col^i（column count）, i_col^i（column indices）

  Step 2: 稀疏 FlashAttention（Vertical-Slash FlashAttention kernel, Algorithm 5）
    GPU 并行 for i ← 1 to N (2048 行 blocks):
      Load Q_chip ← Q[i×B:(i+1)×B]  [B=64, 128]
      Init O_chip = 0, m = -inf, l = 0

      # Part A: Block-sparse attention（循环 block indexes）
      for j ← 1 to c_blk^i:
        s ← i_blk[i, j]                      # block 起始位置
        Load K_chip ← K[s:s+B]               # [64, 128]
        Load V_chip ← V[s:s+B]               # [64, 128]
        S ← τ × Q_chip @ K_chip^T            # [64, 64], τ=1/√128
        S ← mask(S)                           # causal mask
        m_new ← max(m, rowmax(S))
        S ← S - m_new; P ← exp(S)
        l_new ← rowsum(P(S))
        α ← exp(m - m_new)
        l ← α·l + l_new
        O_chip ← α·O_chip + P @ V_chip

      # Part B: PIT column-sparse attention（循环 column indexes）
      j ← 0
      while j < c_col^i:
        cols ← i_col[i, j:j+B]              # [B] column indices
        Load K_chip ← K[cols]               # [64, 128]
        Load V_chip ← V[cols]               # [64, 128]
        S ← τ × Q_chip @ K_chip^T           # 同上流程
        S ← mask(S); m_new ← max(m, rowmax(S))
        S ← S - m_new; P ← exp(S)
        l_new ← rowsum(P(S))
        α ← exp(m - m_new)
        l ← α·l + l_new
        O_chip ← α·O_chip + P @ V_chip
        j ← j + B

      O_chip ← diag(l)^{-1} × O_chip         # 归一化
      Save O[i×B:(i+1)×B] ← O_chip

  输出：O ∈ R^{131072×128}
  ```

  **延迟分解**：
  - 动态索引构建时间：Vertical-Slash ~5-15%，Block-Sparse ~25%（主要开销来自 MeanPooling + block-level matmul）
  - 稀疏计算时间：占总时间主要部分
  - Memory overhead：1M context 下 <160MB（LLaMA-3-8B）

  **实际加速比**（端到端 pre-filling，单 A100）：
  - 10K context：接近 FlashAttention（索引构建开销占比高，~30%）
  - 100K context：1.8× speedup
  - 500K context：6.8× speedup
  - 1M context：10× speedup（从 30 min 降至 3 min）

## Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  在 FlashInfer（Ye et al., 2024）attention kernel 库之上实现针对 GQA 模型的定制化稀疏 attention kernel。核心优化：利用 LessIsMore 的跨 head 统一 token 选择（CUSA）特性，所有 query head 共享同一 token 索引集 ρ，避免了 TidalDecode/Quest 等 per-head 独立选择方案在 GQA 下的冗余 KV loading——在 TidalDecode 中，同一 KV group 的不同 query head 可能选择不同的 token 集合，导致 KV cache 需要加载更多 token。LessIsMore kernel 仅加载统一的 ρ token 集合（大小 K），单次加载即可服务整个 KV group 的所有 query heads，减少 global-to-shared memory 传输。实验比较：(a) kernel 级延迟：LessIsMore vs TidalDecode vs Quest/SeerAttn-R vs StreamingLLM vs Full Attention（FlashInfer），在 DeepSeek-R1-8B、2K budget、16K context 下的 FLOPs/global-to-shared memory/Mem/Latency（Table 4）；(b) 稀疏 attention kernel latency vs TidalDecode 在不同 token budget 下的 speedup（Figure 6b）；(c) 端到端 TBT speedup（Figure 6a）。

- 后端平台是什么，配置是什么。
  单张 NVIDIA A100 80GB GPU。DeepSeek-R1-Distill-Llama-8B 模型。所有 kernel 基于 FlashInfer attention kernel 库实现。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 FlashInfer（https://flashinfer.ai/）attention kernel 库。核心修改：
  1. **统一 token 索引的稀疏 attention kernel**：修改 FlashInfer 的 GQA attention kernel，将 per-head token 索引替换为单一统一索引 ρ，所有 query heads 共享同一个 KV 子集。在 GQA 架构下（如 DeepSeek-R1-8B: 32 query heads, 8 KV heads, group=4），避免同一 KV group 内不同 query head 的独立 KV loading。
  2. **KV cache 加载优化**：Sparse Attention Layers 从 KV cache 仅加载 K[ρ] 和 V[ρ]（仅 K 个 token），存储于 shared memory，随后所有 query head 共享此 KV tile。
  3. **GQA query grouping**：将同一 KV group 的多个 query head 在 kernel 内并行化，利用 Tensor Core 批量 GEMM（Q_g [r, d] @ K[ρ]^T [d, K] -> S [r, K]），饱满 Tensor Core 利用率。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://github.com/DerrickYLJ/LessIsMore（含 FlashInfer 定制 kernel）

  **Kernel 执行全流程（以 DeepSeek-R1-8B GQA 单 decode step 为例）**：

  ```
  输入：
    Q ∈ R^{32×1×128}（32 query heads, d=128）
    K_cache ∈ R^{L_kv×8×128}（8 KV heads, d=128）
    V_cache ∈ R^{L_kv×8×128}
    ρ ∈ N^{K}（统一 token 索引, K=2000）
  输出：O ∈ R^{32×1×128}

  Step 1: Token Selection Layer（在指定 layer 如 Layer 12 执行一次）
    P = Q @ K_cache^T            # [32, 1, L_kv], FlashInfer full attention
    # CUSA: 跨 head 统一选择
    for each KV group g (0..7):  # 4 query heads per KV group
        idx_group = []
        for h in [4g, ..., 4g+3]:
            idx_h = TopK(P[h,:,:], k=K·0.75)  # each query head proposes
            idx_group.append(idx_h)
        # Union across 4 query heads sharing same KV head
    ρ_unified = unique(flatten(all 32 heads' proposals))
    ρ = sort_by_score(ρ_unified)[:K·0.75] ∪ [L_kv-K·0.25, ..., L_kv-1]

  Step 2: Sparse Attention Kernel（后续层复用 ρ）
    # KV cache 加载（仅加载 ρ 中 K 个 token，而非全量 L_kv 个）
    for each KV head g (0..7):
        K_sparse = load_kv_tile(K_cache, ρ, head=g)    # [K, 128] from HBM → SMEM
        V_sparse = load_kv_tile(V_cache, ρ, head=g)    # [K, 128] from HBM → SMEM
        # 同一 KV group 的 4 个 query heads 共享 K_sparse, V_sparse
        Q_g = Q[4g:4g+4]                                # [4, 1, 128]
        S_g = Q_g @ K_sparse^T / √128                    # [4, 1, K], Tensor Core mma
        P_g = online_softmax(S_g)                        # [4, 1, K]
        O_g = P_g @ V_sparse                             # [4, 1, 128], Tensor Core mma
    O = concat([O_0, ..., O_7], dim=0)                  # [32, 1, 128]
  ```

  **Kernel 效率对比**（Table 4, DeepSeek-R1-8B, 2K budget, 16K context）：

  | Method | FLOPs | G2S Memory | On-device Mem | Latency |
  |--------|-------|------------|---------------|---------|
  | LessIsMore | 1.05M | 1.04MB | 8.38MB | 20.1µs |
  | TidalDecode | 1.05M | 2.34MB | 8.38MB | 32.1µs |
  | Quest/SeerAttn-R | 1.05M | 2.34MB | 8.38MB | 32.1µs |
  | StreamingLLM | 1.05M | 1.04MB | 1.04MB | 20.1µs |
  | Full Attention | 8.40M | 8.38MB | 8.38MB | 76.4µs |

  **关键差异**：LessIsMore 与 TidalDecode 计算量（FLOPs）相同，但 Global-to-Shared memory 传输仅 1.04MB vs 2.34MB（减少 55%），因为统一 token 选择避免了同一 KV group 内不同 query head 的冗余 KV loading。这使得 kernel latency 从 32.1µs 降至 20.1µs（1.6× speedup）。

  **Sparse Attention Kernel Speedup**（Figure 6b, vs TidalDecode）：
  - 各 token budget 下 LessIsMore kernel 比 TidalDecode kernel 快 1.3×-1.72×

  **端到端 Speedup**（Figure 6a, vs Full Attention, A100）：
  - 16K context: 1.09×-1.1×
  - 32K context: 1.22×-1.3×
  - 64K context: 1.48×-1.58×

## KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction (Softmax-Free Kernel Variant)

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  KVzip 在附录 C.3 中提出 softmax-free 重要性评分变体。标准 KVzip 在 Algorithm 1 中使用 Softmax 归一化后的注意力分数作为重要性得分，这需要在前向传播之外额外读取注意力矩阵。Softmax-free 变体通过实现自定义 Triton-based FlashAttention CUDA kernel，直接在 fused attention kernel 内部使用未归一化的 QK product（logits）作为重要性得分，省略 Softmax 归一化步骤，从而将评分步骤嵌入前向传播，消除冗余计算。实验比较 softmax-free 变体与标准 KVzip 在 Retr.KV (SCBench) 上的压缩性能（LLaMA3.1-8B）。

- 后端平台是什么，配置是什么。
  NVIDIA A100 80GB GPU，Bfloat16 精度。使用 Triton DSL 编写 CUDA kernel。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：基于 FlashAttention-2 的 LLM 推理 pipeline。修改点：在 FlashAttention fused kernel 内部，在 Softmax 之前截取 QK product 矩阵中 KV_c 对应部分，直接沿 query 维度取 max 作为重要性得分，绕过 Softmax 归一化和后续的注意力矩阵物化。评分过程原本约占 forward 总时间的 10%，该 kernel 将此 10% 开销消除。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://github.com/snu-mllab/KVzip（论文未明确说明 softmax-free kernel 是否在开源仓库中独立提供）
  
  评估原理：标准 FlashAttention 的分块算法在 on-chip SRAM 中计算 QK^T → Softmax → ×V，中间注意力矩阵不写回 HBM。KVzip 需要在 Softmax 之后沿 query 维度取 max，这与 FlashAttention 的逐块计算模式不兼容（需要跨 query 维度的全局 max）。Softmax-free 变体直接跳过 Softmax，将 QK^T logits 作为得分，可在分块计算时直接累积 per-chunk 的最大值，与 FlashAttention 的 online softmax rescaling 逻辑兼容。
  
  Kernel 输入：Query tensor Q ∈ R^{G×H×n_in×d}，Key tensor K ∈ R^{H×(n_c+n_in)×d}（从 KV_c + 当前 input 拼接）。
  Kernel 输出：attention output + importance score S_{l,h,t}（每个分块的 QK^T logits 沿 query 轴 max）。
  
  性能 trade-off：消除 ~10% 评分开销，但压缩比下降约 10%（Figure 15），因未归一化的 logits 不能准确反映注意力权重分布。 by Passing Compressed Context Blocks across GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了一个定制化的 FLASHATTN kernel 以及优化的分布式通信调度策略，用于支持 APB 的 approximate attention 机制。具体包括：(1) 修改 FLASHATTN kernel 的 attention mask，以支持 [anchor block, passing block, local context block] 三部分联合的注意力计算，在 tiling 层面正确实现 M' 遮罩；(2) 在每层 Transformer 中内嵌两次 AllGather 通信（分别对压缩后的 K^C 和 V^C），与 QKV 投影和 attention 计算协同调度；(3) Decoding 阶段实现 Gather + MergeScore（online softmax lse 合并）的分布式精确注意力。实验比较了 APB 与 baselines（FLASHATTN、ULYSSES、RINGATTN、MINFERENCE、STARATTN）在多种 context length（32K-512K）下的 wall-time 分解和推理速度。

- 后端平台是什么，配置是什么。
  8× NVIDIA A800-80GB GPU（NVLink 3.0 互联），104 核 Intel Xeon Platinum 8470 CPU，跨机 HDR InfiniBand，CentOS Linux 7 (Core)。GPU 间通信利用 NVLink 进行 AllGather（intra-node），跨节点使用 InfiniBand。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 HuggingFace Transformers 框架（https://github.com/huggingface/transformers）进行推理实验。核心修改：
  1. **FLASHATTN kernel 修改**：修改 attention mask 为 M'，在 tiling 计算中正确实现 [A, P_h, B_h] 的因果/跨块注意力遮罩
  2. **通信调度**：在每层 attention 计算前后插入 AllGather（K^C, V^C），实现通信-计算流水线
  3. **Decoding 阶段**：实现 STARATTN stage-2 的 Gather+MergeScore 分布式解码，各 host 独立计算 partial attention 后合并

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/thunlp/APB

  **评估原理与 Kernel 执行全流程（以单层 Transformer Prefill 为例）**：

  ```
  输入：每 host 上的 H_a（anchor hidden states）和 H_h（local context hidden states）
  输出：H_a^out, H_h^out, K_a, K_h, V_a, V_h

  Step 1: QKV Projection
    [Q_a, Q_h], [K_a, K_h], [V_a, V_h] = layer.qkv_proj([H_a, H_h])
    // 在 A800 GPU 上利用 cuBLAS 执行矩阵乘法

  Step 2: KV Cache Compression (retaining heads)
    s_1, ..., s_{l_b} = layer.R([Q_h, K_h, V_h])  // MLP 推理
    indices = ArgTop-l_p(s_1, ..., s_{l_b})       // GPU top-k
    K_h^C, V_h^C = K_h[indices], V_h[indices]      // gather 操作

  Step 3: AllGather Communication
    K_{1:H}^C = AllGather(K_h^C)   // NCCL AllGather on NVLink/IB
    V_{1:H}^C = AllGather(V_h^C)

  Step 4: Construct Passing Block
    K_p = concat(K_1^C, ..., K_{h-1}^C)   // host h 只取前序 host 的压缩块
    V_p = concat(V_1^C, ..., V_{h-1}^C)

  Step 5: Modified FLASHATTN Attention
    // 在 SRAM 中分 tile 计算，带 M' attention mask
    // K,V layout: [K_a, K_p, K_h], [V_a, V_p, V_h]
    // Q layout: [Q_a, Q_h]
    A_a, A_h = flash_attn_with_mask([Q_a, Q_h], [K_a, K_p, K_h], [V_a, V_p, V_h], M')

  Step 6: FFN（仅 anchor 和 local context，不含 passing block）
    H_a^out, H_h^out = FFN(A_a, A_h)

  Step 7: Discard passing blocks
    // P_h 在 attention 计算后丢弃，不缓存
  ```

  **Wall-time 分解（128K 输入，Llama-3.1-8B，8 hosts，每 Transformer Block）**：
  - QKV Projection: 4.01 ms
  - Retaining Head: 1.72 ms
  - Communication (AllGather): 0.62 ms
  - Attention (Modified FLASHATTN): 34.07 ms
  - O Projection: 2.67 ms
  - FFN: 30.76 ms
  - Others: 6.33 ms
  - **Total/Block: 80.18 ms**

  **速度评估指标**：
  $$speed = \frac{\#input\_tokens + \#output\_tokens}{prefill\_time + decoding\_time}$$

  **关键性能数据（128K，Llama-3.1-8B，RULER avg）**：
  - FLASHATTN: 4,086 tok/s（单 GPU，OOM at >128K）
  - ULYSSES: 26,200 tok/s
  - RINGATTN: 17,822 tok/s
  - MINFERENCE: 4,545 tok/s（单 GPU）
  - STARATTN: 29,600 tok/s
  - APB: **37,575 tok/s**（最高）

  **通信开销分析**：APB 的 AllGather 通信仅占每 block 总时间 0.62 ms（<1%），远小于 RINGATTN 的 P2P ring 通信（18.40 ms，占 ~9%），这是因为 APB 只传输 top-l_p 个压缩 KV pair（l_p=2K vs 原始 l_b=16K）。

## InfiniteHiP: Extending Language Model Context Up to 3 Million Tokens on a Single GPU

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  使用 Triton 语言实现了三个核心 GPU kernel：(1) **Pruning Stage Kernel**：单一 Triton kernel 实现了完整的剪枝 stage（SelectRep + chunk score estimation + top-K chunk selection），通过参数化设计可复用于所有 stage（不同 l_c、k、b_q），利用 key sequence dimension 并行度（类似 FlashDecode 的 split-KV）避免全局同步，相比 HiP Attention 的迭代式 top-k 算法消除了内部 global thread synchronization；(2) **Block Sparse Attention Kernel**：基于 FlashAttention 风格实现 prefill 的 BSA kernel，基于 FlashDecoding 风格实现 decoding 的 BSA kernel，利用 PagedAttention 管理 KV cache 内存，仅对 ~2K-4K 选中的 key token 计算完整注意力；(3) **UVM Offloading Kernel**：实现基于 Nvidia UVM 的动态 KV cache 加载/驱逐，在 attention kernel 执行期间通过 PCIe 访问 CPU memory，维护 GPU key bank（两个独立 bank：mask-selection 用和 BSA 用）和 page table（global-to-local index mapping），LRU 驱逐策略，整个 offloaded attention 操作实现为 CUDA graph capturable。实验比较：(1) kernel-level attention latency：vs FA2 (1M window)、InfLLM (12K)、HiP (1K) 的 prefill/decoding 延迟，Triton kernel 内各阶段耗时拆解（Stage 0/1/2/BSA/Extra 占比）；(2) decoding latency with KV offloading：vs FA2 (estimated)、InfLLM 在 256K/512K/1024K 下的带 offloading 解码延迟，含 mask hit ratio 和 SA hit ratio 分析。

- 后端平台是什么，配置是什么。
  (1) NVIDIA RTX 4090 24GB（PCIe 4.0 x8），搭配 AMD Ryzen 7950X + 128GB DDR5 + Ubuntu 22.04；(2) NVIDIA L40S 48GB（AWS g6e.48xlarge）。KV cache offloading 的 PCIe 带宽：PCIe 4.0 x8 约 16 GB/s，访存延迟较 VRAM 高 31.5×。attention latency 测试使用 AWQ Llama 3.1 8B + FP8 KV cache。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 SGLang（https://github.com/sgl-project/sglang）的推理框架进行 kernel-level 和 end-to-end 评测。核心修改：
  1. **Pruning Stage Triton Kernel**：用 Triton 语言重写 HiP Attention 的层次化剪枝算法，核心改进是将 SelectRep 的迭代二分搜索展开为无全局同步的单 kernel 实现——由于每次迭代仅访问 2 个 token（左右分支首 token），消除了 HiP 中 internal top-k 导致的 global thread sync
  2. **BSA Triton Kernel**：实现 FlashAttention-style（prefill: tiling + recompute）+ FlashDecoding-style（decoding: split-KV parallel）+ PagedAttention（block-based KV memory）的 block sparse attention
  3. **UVM Offloading**：基于 CUDA UVM 实现运行时 KV cache 动态换页，通过 page table 管理 GPU↔CPU 的 token 迁移，LRU 驱逐策略
  4. **Mask 缓存与刷新**：实现 per-stage mask cache，refresh interval 设为 16/8/4（fast: 32/16/8, flash: 96/24/8）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：(1) hip-attention: https://github.com/DeepAuto-AI/hip-attention/；(2) SGLang 集成: https://github.com/DeepAuto-AI/sglang/

  **Kernel 评估原理与全流程（以单层 attention decoding 为例）**：

  ```
  输入：Query Q ∈ R^(H×1×d)（单 token decoding）、Key K ∈ R^(H×T_kv×d)、Value V（同）
       当前 mask 缓存状态（各 stage 的 I^(l,i) 和 counter c^(i)）
  输出：Attention output O ∈ R^(H×1×d)

  Step 1: Mask Cache Check
    For each stage i = 1..3:
      if c^(i) % n_refresh^(i) == 0:
        运行 PruningStage kernel → 生成/更新 I^(l,i)
        记录 GPU cache miss → fetch missing key from CPU UVM
      else:
        复用缓存的 I^(l,i)（mask temporal locality）

  Step 2: PruningStage Kernel（Triton, 单 kernel）
    - 输入: Q block, K indices I^(i-1), stage params (l_c, b_q, k)
    - Kernel 内操作（以 Stage 1 为例，l_c=256, k=32K）:
      a) 将 I^(i-1) 划分为 n_chunk = |I|/256 个 chunk
      b) 对每个 chunk j (parallel over j):
         - SelectRep: log₂(256)=8 次迭代，每次取 2 token 与 Q 做点积
         - 收敛到代表 token r_j，计算 s_j = max_{head,t}(q_t^T * k_{r_j})
      c) Top-K chunk selection: 保留分数最高的 K = 32000/256 = 125 个 chunk
    - 时间复杂度: O(T_q * T_kv) for Stage 0, O(T_q) for Stage 1-2
    - 关键优化: SelectRep 每次迭代仅 2 次点积，无需全局同步，key sequence dim 并行

  Step 3: Block Sparse Attention Kernel（Triton, FlashDecoding-style）
    - 输入: Q (1 token), K/V (仅 I^(3) 中约 2K-4K tokens)
    - 使用 PagedAttention block-based KV 管理
    - Tiling over key sequence dim (类似 FlashDecoding split-KV)
    - Online softmax rescaling (FlashAttention 风格)
    - 复杂度: O(H * d * k^(3))，其中 k^(3) ≈ 2K-4K

  Step 4: KV Cache Management
    - 记录 BSA 过程中的 GPU cache miss
    - LRU eviction: 驱逐 cold token → CPU UVM，加载 miss token → GPU bank
    - Page table 更新: 维护 global_idx → local_bank_idx 的映射
  ```

  **Attention Latency 拆解（Table 3, 1M context decoding, RTX 4090, 3K preset）**：
  - Total (AR, 无 mask cache): 936 µs
  - Stage 0: 28.2% (264 µs) — 最昂贵的初始剪枝
  - Stage 1: 4.0% (37 µs)
  - Stage 2: 5.3% (50 µs)
  - BSA: 2.2% (21 µs) — 仅对 2K-4K token 计算
  - Extra: 60.3% (565 µs) — UVM offloading 的 PCIe 传输开销

  **Mask Cache 加速效果（Table 4, 256K decoding）**：
  - No cache（3 stage 全重算）: 9,803 µs/token
  - All cache（仅 BSA + offload）: 110 µs/token → **89× speedup**

  **性能对比总结（1M context decoding vs baselines）**：
  - vs FA2: 18.95× faster（245 µs vs 4,645 µs）
  - vs InfLLM (12K): 4.98× faster
  - vs HiP (1K): 92% faster（245 µs vs 450 µs）

## BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了三个核心 GPU kernel：(1) **Residual Kernel**：fused computation + quantization + packing，将新生成 token 的 FP16 KV tensor 通过 ldmatrix 加载到寄存器，在 Tensor Core fragment 上执行 mma 后，各线程在寄存器内完成量化（tensor-wise 或 channel-wise）和 INT16 bit-packing，输出 interleaved layout-compatible 低比特数据到 global memory；(2) **Packing Kernel**：fused dequantization + Tensor Core GEMM，用与 Residual Kernel 相同的 ldmatrix/mma 配置加载 packed 低比特数据，经 lop3 指令高效 dequantization（75316420 pattern remapping），随后送入 Tensor Core mma，并实现了 CUDA Core 与 Tensor Core 的 register-level 异步流水线（ldmatrix+Dequant 与 mma 重叠）；(3) **Multi-warp Cooperative Softmax**：多 warp 并行 softmax，用 shared memory buffer（sTMP, sAcc）做 cross-warp reduction 和跨 warp 同步，在 Hopper 上利用 WGMMA 直接从 shared memory 消费数据。实验比较：(1) kernel-level：vs FP16 FlashDecoding-v2（speedup baseline）、Kivi（non-fused low-bit）、QServe/Atom（CUDA Core-only fused）、FlashDecoding-v3（Hopper optimized），在 Single/Batches/Page 三种 setting 下，跨 Blackwell(H100 equiv. RTX 5090/RTX PRO 6000)、Hopper(H100)、Ada(RTX 4090)、Ampere(A100) 四代 GPU；(2) end-to-end：vs Kivi 和 QServe 在 LLaMA-2-7B、LLaMA-3.1-8B/70B、Qwen3-8B/14B 上的解码延迟和吞吐。

- 后端平台是什么，配置是什么。
  - Blackwell: RTX 5090、RTX PRO 6000（原生支持 MXFP4/NVFP4 低精度 Tensor Core，消除 dequantization 开销）
  - Hopper: NVIDIA H100（80GB HBM，支持 WGMMA 指令、warp-specialized pipeline、TMA 异步拷贝）
  - Ada: NVIDIA RTX 4090（带宽受限 GPU）
  - Ampere: NVIDIA A100（80GB HBM，高带宽 GPU）。多 GPU 实验：8×A100 用于 LLaMA-3.1-70B
  - 所有 kernel 使用 CUDA + PTX 内联汇编编写，基准 kernel（FlashDecoding-v2/v3、QServe、Kivi）对比评估

- 评估性能的软件/脚本是什么。修改了什么。
  基于 CUDA/PTX 手工编写的自定义 attention kernel，集成到 PyTorch 推理 pipeline 中作为 attention 后端。核心修改/实现：

  1. **Residual Kernel（量化+打包）**：
  - 输入：prefill 后的 FP16 KV tensor 和新生成 token 的 FP16 K/V
  - 过程：ldmatrix 加载 FP16 KV tile → Tensor Core mma 执行 QK^T 或 PV → 线程级 min/max reduction（`__shfl_xor_sync` 做 warp-level reduction）→ 计算 scale/zero-point → 各线程在寄存器内量化并 pack 为 INT16 → 写出到 low-bit KV cache global memory
  - 关键优化：利用 ldmatrix 建立的 interleaved register layout，量化后自动保持 Tensor Core compatible layout，无需额外 layout transformation；scale/zero-point 存储为 compact half2 格式

  2. **Packing Kernel（去量化+GEMM）**：
  - 输入：Low-bit packed KV cache（K_pack, V_pack）、FP16 Q tile、量化参数（K_p, V_p）
  - 异步数据移动：Q 和 K_pack/V_pack 用 `cp.async.cg` 从 global→shared memory；K_p/V_p 用 `cp.async.ca` 做 byte-aligned 细粒度 copy；Hopper 上用 `tma.copy` 替代
  - Shared→Register：ldmatrix 加载 K_pack/V_pack 到 Tensor Core register layout；用 sizzling scheme（col_id = row_id ^ col_id）消除 bank conflict
  - Dequantization：lop3 指令执行 75316420 pattern bitwise 映射，高效转换 INT4/INT2→FP16；对齐 Tensor Core 期望的 interleaved fragment layout
  - 异步流水线：第 i 个 tile 在 Tensor Core 上执行 mma 的同时，第 i+1 个 tile 通过 ldmatrix 加载并 dequantize（CUDA Core）——CUDA Core 和 Tensor Core 重叠执行
  - 输出：FP16 attention output O tile

  3. **Multi-warp Cooperative Softmax（Algorithm 1）**：
  - W_m=1（解码 Q length 小），增加 W_n 提高并行度
  - sTMP ∈ R^{W_n}：cross-warp reduction 计算 row-wise max（先 intra-warp register reduction，再 inter-warp shared memory reduction）
  - sAcc ∈ R^{T_m×T_n}：暂存 Tensor Core 寄存器中的 attention scores P，通过 ldmatrix 重载确保 MMA alignment
  - sTMP 和 sAcc 复用同一 shared memory 指针（因 W_n 小）

  4. **Hopper 优化**：
  - 利用 `STSM` PTX 指令将 dequantized FP16 值写入 shared memory
  - 利用 `wgmma_SS` 指令（B 矩阵在 shared memory）执行 Tensor Core GEMM
  - 异步特性使存储与计算重叠

  5. **Blackwell 优化**：
  - 使用原生 MXFP4/NVFP4 mma 指令，直接在 packed 4-bit 数据上执行 GEMM
  - 跳过 lop3-based register remapping（无需显式 dequantization）
  - Block-scaling factor 布局由 Sect. IV-A 的 layout-agnostic 方法自动对齐

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/OpenBitSys/BitDecoding

  **评估原理与 Kernel 执行全流程（以单层 GQA attention decode step 为例）**：

  ```
  输入：
    Q ∈ R^{1×hq×d}（当前 token 的 query，decode 阶段 seq_len=1）
    K_pack ∈ R^{L×hkv×d_packed}（低比特 packed Key cache，含 scale/zp metadata）
    V_pack ∈ R^{L×hkv×d_packed}（低比特 packed Value cache）
    K_res, V_res ∈ R^{res_len×hkv×d}（FP16 residual KV cache）
    量化参数：bit_width β ∈ {2,4}, group_size（channel-wise 时）

  输出：
    O ∈ R^{1×hq×d}（当前 token 的 attention output）

  Step 1: Query Transformation
    // 将 Q 从 [1, gq, hkv] reshape 为 [gq, hkv]（gq = hq/hkv）
    // 在 GQA 下饱满 Tensor Core tile，提高 warp occupancy

  Step 2: Residual Kernel（可选，当 res_len == N_r 时触发）
    // 将 residual KV cache 中满 N_r 个 token 量化并迁移到 packed cache
    ldmatrix 加载 FP16 residual K/V → 线程内 min/max reduction
    → __shfl_xor_sync warp-level reduction → scale/zp 计算
    → 各线程在寄存器内 quantize + pack → 写入 K_pack/V_pack global memory
    // 每 decode step 仅当 res_len == N_r 时触发一次

  Step 3: Packing Kernel（主要 decode kernel）
    // Block-wise tiling: Q tile T_m, KV tile T_n
    for c = 0 to ceil(L/T_n):
        // --- Asynchronous Global→Shared Memory ---
        cp.async.cg: Q tile, K_pack[c], V_pack[c] → SMEM
        cp.async.ca: K_p[c], V_p[c]（量化参数）→ SMEM

        // --- Pipeline Stage 1: Load + Dequant (CUDA Cores) ---
        ldmatrix: K_pack[c] → registers (interleaved Tensor Core layout)
        ldmatrix: K_p[c]   → registers (量化 scale/zp)
        // lop3 bitwise remapping (INT4/INT2 → FP16): 75316420 pattern
        K_fp16 = dequant(K_pack[c], K_p[c])

        // --- Pipeline Stage 2: MMA (Tensor Cores) ---
        // 与下一个 tile 的 ldmatrix+dequant 重叠
        S = Q_fp16 @ K_fp16^T            // Tensor Core mma, output T_m × T_n
        // Cooperative Softmax (cross-warp via sTMP, sAcc)
        m_new = max(m_old, rowmax(S))    // sTMP cross-warp reduction
        P = exp(S - m_new)               // element-wise on CUDA Cores
        sAcc = P                          // store to SMEM for re-alignment
        P' = ldmatrix(sAcc)               // reload for MMA alignment
        O_new = P' @ V_fp16 + diag(exp(m_old - m_new)) @ O_old
        // (Hopper: wgmma_SS 直接从 sAcc shared memory 消费 P')

    // --- Residual KV Cache Attention ---
    // 对 res_len 个 FP16 residual token 执行标准 FlashAttention
    O += FlashAttention(Q, K_res, V_res)

  输出 O 到下一层 Transformer
  ```

  **评估指标与原理**：
  - Kernel-level speedup：以 FP16 FlashDecoding-v2 latency 为基准，normalized speedup = T_baseline / T_method
  - 三种 setting：
    - Single：batch_size=1，模拟边缘长上下文推理
    - Batches：大 batch_size + simple padding
    - Page：大 batch_size + paged attention memory management
  - Nsight Compute profiling：分析 dequantization overhead、Tensor Core utilization、memory throughput
  - End-to-end：HuggingFace Transformers 中替换 attention backend，测量 token/s 吞吐和逐 token 延迟

  **关键性能数据**：
  - Blackwell (RTX 5090, NVFP4): up to 8.6× vs FP16 FlashDecoding-v2, up to 4.3× vs QServe
  - Hopper (H100): up to 8.0× (v3 with wgmma), 4.1× (v2)
  - Ada (RTX 4090): ~4× (4-bit), ~7× (2-bit)
  - Ampere (A100): up to 3×; Kivi 和 QServe 在 A100 上甚至比 FP16 baseline 更差
  - End-to-end (LLaMA-3.1-8B, 128K): 3× latency reduction; >4× higher throughput than QServe
  - Dequantization overhead: BitDecoding <15% (4-bit), <35% (2-bit) vs Atom/QServe ~50%

## Elastic Attention: Test-time Adaptive Sparsity Ratios for Efficient Transformers

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了一个 fused attention kernel，在同一 kernel launch 中同时处理 retrieval heads（FA mode）和 sparse heads（SA mode），替代传统的 serial dispatch 方案。核心设计：(1) **Unified Kernel Launch**：将 routing decisions r 直接作为轻量元数据 m 传入 kernel，kernel 内部通过 thread-block level branching 动态判断每个 head 的类型并执行对应的 attention logic（FA 或 SA）；(2) **Eliminate Tensor Splitting**：无需像 Serial Dispatch 那样在 kernel 外先 split Q/K/V 为 full 和 sparse 两组、再分别 launch 两个 kernel，避免了非连续 tensor fragment 的内存分配和拷贝开销；(3) **Grid Integrity**：保持 grid 维度不变（Batch × Heads × Sequence Blocks），允许 GPU hardware scheduler 最优地分布 sequence blocks 到各 streaming multiprocessor。基于 Block Sparse Attention (BSA) Kernel（Guo et al., 2024, https://github.com/mit-han-lab/Block-Sparse-Attention）实现。实验比较 fused kernel 与 Torch-based sequential、layer-wise hybrid attention 实现在不同 sequence length 下的 prefill-time 加速比（Figure 4）。

- 后端平台是什么，配置是什么。
  单 GPU 部署（无跨设备通信），具体 GPU 型号论文未明确说明（fused kernel 加速测试环境）。基于 Block Sparse Attention (BSA) Kernel，block_size=64, chunk_size=16384, sink_size=128。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch + Block Sparse Attention (BSA) Kernel。核心修改：
  1. **Fused Hybrid Attention Kernel**：替代 PyTorch baseline 的三步 serial dispatch（split Q → FlashAttn + SlidingWin → merge O），实现 unified kernel:
  ```
  # PyTorch Baseline (Serial Dispatch)
  r = Router(x)
  I_full = {h | r[h]=0}, I_sp = {h | r[h]=1}
  Q_full = Q[:, I_full]
  O_full = FlashAttn(Q_full, K_full, V_full)      # kernel 1
  O_sp = SlidingWin(Q[:, I_sp], K_sp, V_sp)       # kernel 2
  O[:, I_full] = O_full; O[:, I_sp] = O_sp        # merge

  # Fused Kernel (Parallel via BSA)
  r = Router(x)
  m = Map(r)  # lightweight metadata
  O = BSA_Kernel(Q, K, V, m)  # single kernel
  # Inside kernel:
  # par for h do:
  #   if m[h]==SP: O[h] = Sparse(Q[h], K, V)
  #   else:        O[h] = Full(Q[h], K, V)
  ```
  2. **Thread-block Level Branching**：每个 thread block 从 metadata m 中动态获取所分配 head 的类型，根据类型执行对应的 attention 计算逻辑（FA 或 SA），避免 kernel 外部的 tensor rearrangement。
  3. **Sequence-level Parallelism**：当输入序列足够长时，parallelism 沿 sequence dimension 主导执行。GPU 在完成一个 head 的大部分 sequence blocks 后才切换到下一个 head。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/LCM-Lab/Elastic-Attention。BSA Kernel 来自 https://github.com/mit-han-lab/Block-Sparse-Attention。

  **评估原理与 Kernel 执行全流程（以单层 prefill 为例）**：

  ```
  输入：Q, K, V ∈ R^{s×H×d'}（H KV heads via GQA）, router decisions r ∈ {0,1}^H
  输出：O ∈ R^{s×H×d'}

  Step 1: Attention Router 计算 routing decisions
    x_K = K  # Key hidden states as input
    x_K' = BoundaryPooling(x_K)  # [H, d'], 聚合前100+后100 tokens
    z = MLP_router(MLP_task(x_K'))  # [H, 2]
    r_hard[h] = argmax(softmax(z[h]))  # 0=FA, 1=SA

  Step 2: Map routing decisions to metadata
    m = {h: "full" if r_hard[h]=0 else "sparse" for h in 1..H}

  Step 3: Unified BSA Kernel Launch
    grid = (Batch, Heads, ceil(s / T_s))  # T_s = tile size along sequence
    # Single kernel launch — no pre-splitting of tensors

    # Inside kernel, each thread block:
    block_idx = (b, h, seq_tile)
    if m[h] == "full":
        # Standard FlashAttention-like tiling for FA
        Q_tile = Q[b, h, seq_tile]       # load to SRAM
        K_tile = K[b, h, :]              # load to SRAM (full K)
        V_tile = V[b, h, :]
        S = Q_tile @ K_tile^T / sqrt(d')
        P = online_softmax(S)
        O_tile += P @ V_tile
    else:  # m[h] == "sparse"
        # Sparse attention: only attend to sink + recent + selected tokens
        K_sparse = K[b, h, sp_indices]   # sp_indices = {sink, recent, selected}
        V_sparse = V[b, h, sp_indices]
        S = Q_tile @ K_sparse^T / sqrt(d')
        P = online_softmax(S)
        O_tile += P @ V_sparse

  Step 4: Output concatenation
    O = concat all head outputs along head_dim
    # No post-kernel merge needed — output already in correct order
  ```

  **性能优势（Figure 4）**：
  - Fused kernel 相比 Torch-based sequential 实现在 prefill 阶段实现加速
  - 加速收益随序列长度增加而增大（较长的序列使 sequence-level parallelism 更充分地利用 GPU SMs）
  - 两种主要 overhead 被消除：(1) Memory overhead — 不再需要 allocate 和 copy 非连续 tensor fragment（split Q_full/Q_sp）；(2) Kernel Launch & Scheduling overhead — 不再需要多次 kernel launch，避免 work fragmentation 和 GPU SM 调度中断

  **Router Latency（Figure 10, 消融）**：
  - Attention Router 产生 negligible overhead：平均 0.196 ms/router call
  - 延迟不随序列长度增长（512 → 1M tokens 保持恒定），因为 router 的 pooling 仅处理 boundary tokens（首部+尾部各100）

## AdaSplash: Adaptive Sparse Flash Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了一套基于 Triton 语言的自定义 GPU kernel，用于高效计算 α-entmax 稀疏注意力。包括：(1) 前向 kernel：block-wise tiling 避免 materialize 完整的 S ∈ R^{n×n} 和 P ∈ R^{n×n}，recomputation 策略以 FLOPs 换 memory，Halley-bisection block version 在 SRAM 中累积 f, f', f'' 不写出 S；(2) 反向 kernel：分 dK/dV kernel 和 dQ kernel 两个独立 kernel，利用 α-entmax 稀疏 Jacobian (U_i^{(j)} = P_i^{(j)}^{2-α}) 计算 dS = U ⊙ (dP - δ)；(3) 稀疏调度优化：前向 pass 最后迭代中动态构建 block mask M ∈ {0,1}^{T_r×T_c}，通过 pointer-increment lookup tables (K_j, Q_i) 跳过 null blocks 的 HBM 读写。实验比较：(1) ADASPLASH Triton kernel vs. FlashAttention-2 (CUDA + Triton) 在 synthetic data 上的 runtime vs. input sparsity 关系；(2) 不同序列长度 (512-8192) 和不同注意力实现 (Torch sorting, Torch bisection, Halley-bisection Triton, ADASPLASH Triton) 的 ModernBERT training step 时间与内存。

- 后端平台是什么，配置是什么。
  - Efficiency benchmark (synthetic data, Figures 1, 3) 和 GPT-2 训练：单张 Nvidia H100 GPU (80GB HBM, large SRAM)
  - ModernBERT runtime 和下游任务训练：Nvidia RTX A6000 GPU (48GB VRAM)
  - 所有 kernel 均用 Triton 语言编写（https://github.com/triton-lang/triton），利用 Triton 的 block-level programming 模型实现对 GPU SRAM 和 HBM 的精细控制

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch (torch.compile 未使用，因 attention 太复杂需手动优化) + Triton 语言 + HuggingFace Transformers 框架。核心 Triton kernel 修改/实现：
  
  1. **Halley-bisection block version kernel（Algorithm 3）**：
  - 输入：Q_1..T_r, K_1..T_c 分块，HBM 中
  - 过程：对每 Q_i block，循环 T_c K_j blocks，在 SRAM 中计算 S_i^{(j)} = Q_i K_j^T，但不写出；直接累积 f(τ), f'(τ), f''(τ) 的 block 部分贡献；用 Halley-bisection 在 M 次迭代后输出 τ_i
  - 关键优化：block-accumulated f/f'/f'' 避免了 materialize 完整的 S 矩阵，仅需 O(B_r × B_c) SRAM workspace
  
  2. **ADASPLASH forward kernel（Algorithm 2）**：
  - block sizes: B_c（column block size for K,V）, B_r（row block size for Q）
  - 每 Q_i block：先调 Halley-bisection block kernel 得 τ_i；再 loop K_j/V_j blocks，在 SRAM 中 re-compute S_i^{(j)}，计算 P_i^{(j)} = [(α-1)S_i^{(j)} - τ_i]_+^{1/(α-1)}；累积 O_i += P_i^{(j)} V_j
  - 比 FlashAttention-2 多 2 次 K 加载（用于 τ 计算），故前向 pass 永远稍慢于 FA2
  
  3. **Block Mask 生成与 Lookup Table 构造**：
  - 在 Halley-bisection 最后迭代中：对每个 (i,j) block pair，check any(S_i^{(j)} > τ_i)，存为 binary M_{ij}
  - 用 torch.argwhere 提取 M_{ij}=1 的 (i,j) 索引 → K_j, Q_i lookup tables
  - M 为二进制值且跨 attention 层可共享，内存开销 O(T_r×T_c) 远小于 P ∈ R^{n×n}
  
  4. **ADASPLASH backward dK/dV kernel（Algorithm 4）**：
  - 外层 loop j=1..T_c（K_j/V_j 粒度），内层 loop i ∈ K_j（仅有效 Q_i）
  - 在 SRAM 中计算：S_i^{(j)} → P_i^{(j)} → U_i^{(j)} (= P_i^{(j)}^{2-α}) → dS_i^{(j)} = U_i^{(j)} ⊙ (dP_i^{(j)} - δ_i)（利用 α-entmax 稀疏 Jacobian）
  - dV_j += (P_i^{(j)})^T dO_i, dK_j += (dS_i^{(j)})^T Q_i
  - Block masking 跳过 M_{ij}=0 的 block，大幅减少无效 HBM 访问
  
  5. **ADASPLASH backward dQ kernel（Algorithm 5）**：
  - 外层 loop i=1..T_r（Q_i 粒度），内层 loop j ∈ Q_i（仅有效 K_j）
  - dQ_i += dS_i^{(j)} K_j
  - 与 dK/dV kernel 分离，允许独立的并行化和 block masking

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/deep-spin/adasplash（ICML 2025）

  **评估原理与 Kernel 执行全流程（以一个 attention head 的前向+反向 pass 为例）**：

  ```
  输入：Q, K, V ∈ R^{n×d}（HBM 中），block sizes B_r, B_c，参数 α, iterations T
  输出：O ∈ R^{n×d}（前向），dQ, dK, dV ∈ R^{n×d}（反向）

  === 前向 Pass ===
  Step 1: 分块
    将 Q 分为 T_r = ⌈n/B_r⌉ blocks，K,V 分为 T_c = ⌈n/B_c⌉ blocks
    分配 τ ∈ R^n 的 T_r blocks（HBM 中）
  
  Step 2: τ 计算（Halley-Bisection Block Kernel，Algorithm 3）
    for i = 1..T_r:
        Load Q_i (B_r × d) from HBM → SRAM
        Initialize τ_i, τ_lo_i, τ_hi_i
        for iter = 1..T:  // T=3 即可达到 machine precision
            f, f', f'' = 0 (in SRAM, shape B_r)
            for j = 1..T_c:
                Load K_j (B_c × d) from HBM → SRAM
                Compute S_i^{(j)} = Q_i @ K_j^T on SRAM  // B_r × B_c
                Accumulate f, f', f'' using block contributions  // Equations 3,6,7
                // S_i^{(j)} NOT written to HBM (recomputed later)
            // Store binary mask M_{ij} if last iteration
            M_{ij} = 1 if any(S_i^{(j)} > τ_i) else 0
            // Update τ_i via Halley-bisection (Algorithm 1)
            if Halley update within [τ_lo, τ_hi]:
                τ_i = Halley_update
            else:
                τ_i = bisection_update
        Write τ_i to HBM
  
  Step 3: 构造 Lookup Tables
    M_full = M (T_r × T_c, bool, on GPU)
    indices = torch.argwhere(M)  // (i,j) pairs where M_{ij}=1
    K_j lookup: per-column j, row indices i where M_{ij}=1
    Q_i lookup: per-row i, column indices j where M_{ij}=1
  
  Step 4: 前向 Attention 计算（Algorithm 2 + Block Masking）
    for i = 1..T_r:
        Load Q_i, τ_i from HBM → SRAM
        Initialize O_i = 0 (B_r × d, on SRAM)
        for j in Q_i:  // only non-null blocks!
            Load K_j, V_j from HBM → SRAM
            Recompute S_i^{(j)} = Q_i @ K_j^T  // re-computation (not stored)
            Compute P_i^{(j)} = [(α-1)S_i^{(j)} - τ_i]_+^{1/(α-1)}
            O_i += P_i^{(j)} @ V_j  // accumulate in SRAM
        Write O_i to HBM

  === 反向 Pass ===
  Step 5: δ 计算（Separate Kernel, Equation 25）
    需要前向存下的 O^{(2)} ∈ R^{n×d}（存储增量 vs softmax 的 O）
    for i = 1..T_r:
        δ_i = dO_i @ O_i^{(2)} / ||U_i||_1  // per-row normalization
  
  Step 6: dK, dV Kernel（Algorithm 4 + Block Masking）
    for j = 1..T_c:
        Load K_j, V_j from HBM → SRAM
        Initialize dK_j, dV_j = 0 (on SRAM)
        for i in K_j:  // only Q_i rows that contribute
            Load Q_i, dO_i, τ_i, δ_i from HBM → SRAM
            Recompute S_i^{(j)}, P_i^{(j)}, U_i^{(j)}
            dP_i = dO_i @ V_j^T
            dS_i^{(j)} = U_i^{(j)} ⊙ (dP_i - δ_i)  // sparse Jacobian product
            dK_j += (dS_i^{(j)})^T @ Q_i
            dV_j += (P_i^{(j)})^T @ dO_i
        Write dK_j, dV_j to HBM
  
  Step 7: dQ Kernel（Algorithm 5 + Block Masking）
    for i = 1..T_r:
        Load Q_i, dO_i, δ_i, τ_i from HBM → SRAM
        Initialize dQ_i = 0 (on SRAM)
        for j in Q_i:  // only non-null blocks
            Load K_j, V_j from HBM → SRAM
            Recompute S_i^{(j)}, P_i^{(j)}, U_i^{(j)}
            dP_i = dO_i @ V_j^T
            dS_i^{(j)} = U_i^{(j)} ⊙ (dP_i - δ_i)
            dQ_i += dS_i^{(j)} @ K_j
        Write dQ_i to HBM
  ```

  **额外内存开销（vs FlashAttention-2）**：
  - O^{(2)} ∈ R^{n×d}：取代 softmax 中存 O 的需求，实际与 FA2 的 O 等大
  - M ∈ {0,1}^{T_r×T_c}：二进制 block mask，可跨 attention 层共享
  - τ ∈ R^n：每行一个标量阈值，O(n) 额外开销

  **评估指标**：
  - 平均 training step 时间（前向+反向）（单位：s 或 ms/step）
  - Peak GPU memory usage（GB）
  - 50 steps 的平均（warmup 后）

  **ModernBERT-base runtime 对比（Table 5, A6000 GPU）**：
  | 算法 | 512 | 1024 | 2048 | 4096 | 8192 |
  |------|-----|------|------|------|------|
  | Sorting (Torch) | 0.09s | 0.11s | 0.26s | 0.76s | OOM |
  | Bisection (Torch) | 0.11s | 0.15s | 0.42s | 1.35s | 4.99s |
  | Halley-bisection (Triton) | 0.10s | 0.11s | 0.26s | 0.46s | 1.61s |
  | **ADASPLASH (Triton)** | 0.10s | 0.12s | 0.21s | 0.48s | **1.53s** |

  **GPT-2 training step（Table 4, H100 GPU, 1024 ctx）**：
  - FlashAttention-2 (softmax): 0.98 s/step, 52.5 GB
  - ADASPLASH (α=1.5): 1.03 s/step, 52.5 GB
  - Torch sorting (α=1.5): 3.61 s/step, 73.8 GB
  - Torch bisection (α=1.5): 7.78 s/step, 77.6 GB

## CommVQ: Commutative Vector Quantization for KV Cache Compression

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现 CommVQ 的 Triton kernel 以在 GPU 上实际节省 KV cache 显存并优化解码延迟。核心实现包括：(1) **可交换码本的解码 kernel**：利用 RoPE-可交换码本在 self-attention 中融合 KV cache 解码，kernel 通过预计算 (qR_t)C_K^T（一次计算，跨所有 token 复用）加上每个 token 的轻量 R_i^T s_i^T 旋转操作替代先前需要逐 token 完整解码的全量矩阵乘法；(2) **Value 解码重排 kernel**：先计算 Softmax(A) × S_V（小矩阵乘法），再乘以 C_V，将复杂度从 O(d N_c N + dN) 降至 O(N_c N + d N_c)；(3) **量化 KV cache 的压缩存储**：在 GPU 全局内存中以 1-bit/2-bit 精度存储量化后的 KV cache，加载时按需解压。实验比较 naive 实现（decode-then-self-attention）与优化实现（commutative codebook + reordering）在 8K/32K/128K context length 下的单层单 token 延迟（Table 5），以及 FP16 与 CommVQ-1bit 在不同 context length（至 128K）和 batch size（至 128）下的 per-token 解码显存使用量（Figure 3）。

- 后端平台是什么，配置是什么。
  NVIDIA H100-80GB GPU（主要实验与显存测量平台）；NVIDIA RTX 4090（验证单卡 128K context 推理可行性）。显存测量在 LLaMA-3.1-8B-Instruct 模型上进行。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch + Triton 自定义 kernel 实现。核心修改：
  1. **Key cache 解码-注意力融合 kernel（Triton）**：将 key 码本解码从独立的前置步骤融合到 attention score 计算中。预计算 q_pre = (qR_t) C_K^T（[1, d] × [d, N_c] → [1, N_c]），然后对每个已缓存 token i，仅需计算 R_i^T s_i^T 的旋转操作并点积。原始朴素实现的复杂度为 O(2d N_c N)，优化后为 O((Rd + N_c + 1)N + d(N_c + R N_c'))。
  2. **Value cache 解码 kernel（Triton）**：重排计算顺序，先计算注意力权重与 S_V 的乘积（小矩阵 [1, N] × [N, N_c] → [1, N_c]），再乘以 C_V（[1, N_c] × [N_c, d] → [1, d]），避免逐 token 先解码再聚合的 O(d N_c N) 开销。
  3. **量化 KV cache 存储管理**：在 GPU 显存中以 uint8/打包位存储量化后的 s_i 向量，解码时按索引查表还原。
  4. **Codebook 常驻显存**：码本大小固定（2-bit 时 ~9.25 MB，1-bit 时 ~4.75 MB），与 token 数量无关，在长 context 下可忽略。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/UMass-Embodied-AGI/CommVQ

  **评估原理与 Kernel 执行全流程（以单层 decoding step 为例）**：

  ```
  输入：当前 query 向量 x ∈ R^{1×d}，量化 KV cache S_K, S_V，码本 C_K, C_V，RoPE 参数
  输出：self-attention 输出 O ∈ R^{1×d}

  Step 1: Query Projection + RoPE
    q = x @ W_Q                                # cuBLAS gemm, [1,d] @ [d,d] -> [1,d]
    q = apply_rope(q, position=t)              # Triton: 逐 2D 子空间旋转

  Step 2: Key 解码-注意力融合 (Triton fused kernel)
    # Kernel Launch: grid=(num_heads,), block=(d/2,)
    # 预计算（once per decoding step, shared across all cached tokens）
    q_pre = q @ C_K^T                          # [1,d] @ [d,2R·log2(N_c')] -> [1, K_dim]
    # 逐 token 计算 attention score（Triton 内并行）
    for i in range(num_cached_tokens):
        # 从量化的 S_K[i] 中查表取出旋转后的子码本贡献
        # 计算 R_i^T s_i^T 并与 q_pre 点积
        alpha[i] = fused_rope_decode_dot(q_pre, S_K[i], position=i)

  Step 3: Softmax
    alpha = softmax(alpha / sqrt(d))           # [1, N], PyTorch

  Step 4: Value 解码-聚合融合 (Triton reordering kernel)
    # 先小矩阵乘，再大矩阵乘（重排以降低复杂度）
    temp = alpha @ S_V                         # [1,N] @ [N,N_c] -> [1,N_c]
    O = temp @ C_V                             # [1,N_c] @ [N_c,d] -> [1,d]
    # 等价于原 attention 输出，但复杂度从 O(d·N_c·N + d·N) 降至 O(N_c·N + d·N_c)

  Step 5: Output Projection
    O = O @ W_O                                # cuBLAS gemm
  ```

  **延迟对比（Table 5, H100 GPU, LLaMA-3.1-8B, per-layer per-token, ms）**：
  | 实现 | 8K ctx | 32K ctx | 128K ctx |
  |------|--------|---------|----------|
  | Naive (decode-then-attn) | 2.4 | 9.2 | 36.6 |
  | Optimized (commutative CB) | 0.4 | 1.1 | 3.8 |
  | Speedup | 6.0× | 8.4× | 9.6× |

  **显存节省（Figure 3, H100-80GB, LLaMA-3.1-8B, per-token decoding）**：
  - 120K ctx: FP16 需 60 GB → CommVQ-1bit 仅 20 GB
  - 32K ctx + batch=8: FP16 OOM → CommVQ-1bit 可扩展至 batch=128
  - 128K ctx 在单 RTX 4090 上可运行（FP16 无法）

## HATA: Trainable and Hardware-Efficient Hash-Aware Top-k Attention for Scalable Large Model Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  HATA实现了三项硬件高效GPU kernel优化：(1) Kernel Fusion for Hash Encoding：将HashEncode阶段涉及的linear projection、sign function、BitPack和cache update等操作融合为单个CUDA kernel，消除CPU-GPU同步开销（PyTorch原生dispatch每个op需tens of microseconds）；(2) High-Performance Hamming Score Operator：自研高效GPU operator计算query和key hash codes之间的Hamming距离——用XOR+popc/popc11指令计算bit mismatch数，通过coalesced memory access从HBM到SRAM优化带宽；(3) Fused Gather with FlashAttention：将sparse attention的gather操作与FlashAttention kernel融合，消除selected K/V在HBM和SRAM间的冗余数据搬运。实验比较：(a) 完整优化HATA vs Simple PyTorch实现在不同组件增量添加下的speedup——Score减少53.2% latency, FusedAttn减少23.8%, Encode减少7.6%，最终6.53× speedup；(b) HATA vs baseline (Dense vLLM, Loki Triton, Quest open-source)在decode step latency上的比较，batch_size=1~8, seq_len=8K~256K。

- 后端平台是什么，配置是什么。
  48GB HBM GPU (最高149.7 TFLOPS FP16)，96 cores。Ubuntu 24.04，CUDA 12.1，PyTorch 2.4，FlashInfer。Efficiency benchmarking在单GPU上进行，覆盖batch_size 1~8，sequence length 8K~256K。

- 评估性能的软件/脚本是什么。修改了什么。
  基于PyTorch + FlashInfer + 自定义CUDA/PTX kernel（1470行C++/CUDA）。核心实现/修改：

  1. **Fused Hash Encode Kernel（单CUDA kernel）**：
  - 输入：K ∈ R^{s×d}, W_H ∈ R^{d×128}
  - 过程：单kernel内完成 MatMul(K,W_H) → Sign → BitPack → Cache Update，替代4个独立PyTorch op
  - 减少CPU kernel launch overhead：从4次dispatch（每次tens of μs GPU + tens of μs CPU）合并为1次
  - 输出：K_H ∈ N^{s×4}（128 bits packed为4个INT32）
  - Speedup贡献：end-to-end latency减少7.6%

  2. **High-Performance Hamming Score Operator（CUDA kernel）**：
  - 输入：Q_H ∈ N^{1×4}, K_H_cache ∈ N^{s×4}（128-bit = 4 INT32 per token）
  - 过程：
    a. Coalesced memory access：从HBM加载连续的K_H_cache tile到SRAM
    b. bitwise_xor(Q_H, each K_H)：M个整数同时XOR，'1'→mismatch, '0'→match
    c. popc/popc11指令：对每个XOR结果计数'1'的数量（硬件级bit-count）
    d. Reduction：高效reduction operator聚合各整数count → final Hamming score S[i]
  - 关键优化：避免逐bit比较，用整数级+硬件popc实现O(s×4)而非O(s×128)的复杂度
  - Speedup贡献：end-to-end latency减少53.2%

  3. **Fused Gather with FlashAttention Kernel**：
  - 输入：Q ∈ R^{1×d}, K_cache ∈ R^{s×d}, V_cache ∈ R^{s×d}, indices ∈ N^N
  - 过程：将Gather(K/V, indices)操作融合到FlashAttention kernel内部：
    a. 在FlashAttention tiling中，根据indices直接加载选中的K/V tiles到SRAM
    b. 避免先将gathered K/V写入HBM再读回（节省2×带宽）
    c. 保留FlashAttention的online softmax + recomputation优化
  - 关键优化：消除HBM↔SRAM的冗余数据搬运
  - Speedup贡献：end-to-end latency减少23.8%

  **与FlashInfer集成**：HATA作为pluggable attention后端集成到FlashInfer推理框架，用户仅需替换标准attention为HATA attention。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/gpzlx1/HATA

  **评估原理与Kernel执行全流程（以单层GQA decode step为例）**：

  ```
  输入：Q ∈ R^{1×hq×d}, K_new ∈ R^{1×hkv×d}, V_new ∈ R^{1×hkv×d},
       K_cache ∈ R^{s×hkv×d}, V_cache ∈ R^{s×hkv×d},
       K_H_cache ∈ R^{s×4}（128-bit packed）
  输出：O ∈ R^{1×d_model}

  Step 1: Hash Encode (Fused CUDA Kernel)
    # 单kernel完成以下操作：
    [GPU Kernel Launch 1 — Fused Hash Encode]
    K_H_new = Sign(K_new @ W_H)       # MatMul on Tensor Cores → Sign
    K_H_new = BitPack(K_H_new)        # 128 bits → 4 INT32
    Q_H = Sign(Q @ W_H)              # same for query
    Q_H = BitPack(Q_H)
    # Cache update (in SRAM, direct write):
    K_H_cache = [K_H_cache; K_H_new]
    # Output: Q_H[1, 4], K_H_new[1, 4], updated K_H_cache[s+1, 4]

  Step 2: Hamming Score (CUDA Kernel)
    [GPU Kernel Launch 2 — Hamming Score]
    # Grid: (batch × num_KV_heads,)
    for each KV head:
        # Coalesced load K_H_cache tile from HBM → SRAM
        xor_result = bitwise_xor(Q_H, K_H_cache_tile)   # [tile_size, 4] INT32
        # popc on each INT32 element
        for i in 0..3:
            count[i] = popc(xor_result[:, i])           # hardware instruction
        S[tile] = sum(count) / 128                      # normalize to [0,1]
    # GQA aggregation: sum S across shared KV head query heads
    Output: S[s] # Hamming scores for all cached keys

  Step 3: TopK Selection (standard GPU op)
    Idx = TopK(S, N)   # N = token_budget, e.g., 1.56% × s
    # GPU: parallel radix sort or bitonic top-k

  Step 4: Fused Gather + FlashAttention (CUDA Kernel)
    [GPU Kernel Launch 3 — Fused Gather + FlashAttention]
    # Modified FlashAttention kernel:
    for each KV head:
        # Instead of loading full K_cache, V_cache:
        for each attention tile:
            # Selectively load only the K,V tokens indexed by Idx
            K_tile = Gather_tile(K_cache, Idx[tile_start:tile_end])
            V_tile = Gather_tile(V_cache, Idx[tile_start:tile_end])
            S_tile = Q @ K_tile^T / sqrt(d)
            P_tile = online_softmax(S_tile)
            O += P_tile @ V_tile
    Output: O[1, d_model]

  Step 5: Output Projection
    O = O @ W_O   # standard linear, cuBLAS
  ```

  **评估指标与原理**：
  - **Decode latency（ms/token）**：测量单decode step的wall-clock时间，包括hash encoding + scoring + top-k + sparse attention
  - **End-to-end latency（s）**：prefill + N个decode steps的总时间
  - **Speedup over Dense**：latency_dense / latency_HATA
  - **Ablation分析**：增量启用以测量各优化组件的独立贡献

  **Ablation结果（Llama2 attention module, 128K input）**：
  | 配置 | Latency (relative) | Speedup |
  |------|-------------------|---------|
  | Simple PyTorch HATA | 1.00× | 1.00× |
  | + Score Operator | 0.47× | 2.14× |
  | + FusedAttn | 0.36× | 2.81× |
  | + Encode Fusion (Full HATA) | 0.15× | 6.53× |

  **关键性能数据**：
  - Llama2 batch=8 seq=32K: 7.20× speedup over Dense, 1.99× over Loki
  - Llama2 batch=1 seq=256K: 6.51× over Dense, 2.21× over Loki, 1.19× over Quest
  - Prefill overhead < 1%（rbit=128 ≪ s）

## HISA: Efficient Hierarchical Indexing for Fine-Grained Sparse Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  使用 TileLang (Wang et al., 2025) 实现了 HISA 两个阶段的 GPU kernel，并对比 DSA 原始 TileLang indexer kernel 的延迟。HISA kernel 分解为两个阶段：(1) Block-level filtering kernel：对 ⌈L/B⌉ 个 pooled block 代表向量做 attention 式打分，选出 top-m blocks；(2) Token-level refinement kernel：仅对候选 block 内的 token（最多 mB 个）做 token-level indexer 计算，选出最终 top-k tokens。实验比较 HISA kernel vs DSA kernel (flat token scan) 在 8K-64K context length 下的 indexer latency，以及两种预算模式：(a) fixed block budget m=64 (B=128)，(b) fixed compression ratio M:m = 4:1。所有 kernel 在单张 NVIDIA A100 GPU 上测试，query lens=1024, k=2048。结果以 indexer kernel level 报告，不包含 Sparse MLA operator 和其他系统组件。

- 后端平台是什么，配置是什么。
  单张 NVIDIA A100 GPU。使用 TileLang (https://github.com/tile-ai/tilelang) 作为 kernel 编程语言。DSA baseline kernel 遵循 TileLang 官方参考实现 (https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32)。配置：query length=1024, final top-k=2048 tokens, block size B=128。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 TileLang 编写自定义 kernel。核心实现/修改：

  1. **Block-level Filtering Kernel**：
  - 输入：pooled block representative keys k̃_b^I ∈ R^{M×d}，query indexing representations q_{t,j}^I ∈ R^{H^I×d}，gating weights w_{t,j}^I
  - 过程：计算 J_{t,b} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k̃_b^I)，即 query 对所有 M = ⌈L/B⌉ 个 block 代表向量的 attention score
  - TopK 选出 m 个 block，同时强制包含首尾 block
  - TileLang tiling：沿 M 维分 tile 做 block-level matmul，M ≪ L（如 64K/B=128 → M=500），计算量远小于 token-level indexer
  - 输出：候选 block 索引集 C_t 和候选 token 集 Ω_t

  2. **Token-level Refinement Kernel**：
  - 输入：仅候选 token 集 Ω_t（≤ mB）的 indexing keys k_s^I，query representations q_{t,j}^I, gating weights w_{t,j}^I
  - 过程：使用与 DSA 相同的 scoring 公式 I_{t,s} = Σ_j w_{t,j}^I · ReLU(q_{t,j}^I · k_s^I)，但仅在 Ω_t 上计算
  - TopK 选出最终 k=2048 tokens
  - TileLang tiling：沿候选 token 维分 tile，|Ω_t| ≤ mB = 8192 (4:1 ratio) 或 2048 (fixed budget)，远小于全前缀 L
  - 在 fixed 8K budget 模式下，第二阶段输入输出长度均固定，计算图更易优化，进一步提速

  3. **与 DSA kernel 的差异**：
  - DSA kernel：对所有 L 个 token 执行一次完整 token-level indexer scan，复杂度 O(L)
  - HISA kernel：先对 M=⌈L/B⌉ 个 block 做轻量粗过滤 (O(L/B))，再对 mB 个候选 token 做精筛 (O(mB))
  - HISA 增加了 block filtering 阶段的开销，但该阶段仅在 pooled 摘要上操作（M ≪ L），代价远小于跳过大量不相关 token 带来的收益

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/MuLabPKU/TransArch（论文声称仓库，HISA 代码标记为"Release HISA code ☐"尚未发布）。DSA 参考 TileLang kernel：https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32。

  **评估原理与 Kernel 执行全流程（以单层 HISA indexer 为例）**：

  ```
  输入: indexing query representations Q^I ∈ R^{H^I × d}（per query position）
        token indexing keys K^I ∈ R^{L × d}
        gating weights w^I ∈ R^{H^I}
        block pooled keys K̃^I ∈ R^{M × d}（M = ceil(L/B)，增量维护）
  输出: selected token indices T（size k=2048）

  Step 1: Block-level Filtering (TileLang kernel)
    // Grid: (H^I_heads, ceil(M / T_M))
    for each indexing head j:
        q_j = Q^I[j, :]                                // [d]
        for each block tile:
            K̃_tile = load K̃^I[tile] from HBM → SRAM    // [T_M, d]
            S_tile = q_j @ K̃_tile^T                     // [T_M], block scores
            // ReLU activation
            S_tile = ReLU(S_tile)
            // Gating weight multiply
            J[:, tile] += w^I[j] * S_tile              // accumulate across heads
    // Global TopK across all blocks
    C = TopK(J, m)                                      // m selected blocks
    C = C ∪ {0, M-1}                                   // force include first/last
    Ω = all token indices in selected blocks

  Step 2: Token-level Refinement (TileLang kernel)
    // Grid: (H^I_heads, ceil(|Ω| / T_tok))
    for each indexing head j:
        q_j = Q^I[j, :]
        for each candidate token tile in Ω:
            K_tile = gather_and_load K^I[Ω_tile]        // [T_tok, d], selective load
            S_tile = q_j @ K_tile^T                     // [T_tok]
            S_tile = ReLU(S_tile)
            I[:, Ω_tile] += w^I[j] * S_tile            // accumulate across heads
    // Global TopK across candidate tokens
    T = TopK(I, k)                                      // k=2048 final tokens

  Step 3: Sparse MLA（与 DSA 完全相同，不修改）
    u_t = SparseMLA(h_t, {c_s | s ∈ T})
  ```

  **评估指标与原理**：
  - Indexer kernel latency (ms)：纯 indexer 阶段的 wall-clock time
  - Speedup = latency_DSA / latency_HISA
  - 两种预算模式：
    - Fixed top-m=64 (B=128): 随 seq_len 增长，candidate pool 从 ~完全覆盖 变为 ~部分覆盖
    - Fixed compression ratio M:m=4:1: 随 seq_len 增长，m 自适应增长以保持恒定压缩比

  **关键性能数据（A100, query lens=1024, k=2048）**：
  | Context Len | DSA Indexer | HISA (4:1 ratio) | HISA (fixed 8K budget) |
  |-------------|-------------|-------------------|------------------------|
  | 8K          | ~0.7 ms     | ~0.5 ms           | ~0.5 ms                |
  | 64K         | ~5.6 ms     | ~2.6 ms (2.16×)   | ~1.5 ms (3.75×)        |

  Sparse MLA operator 自身约 1.6 ms（与 seq_len 无关），表明 DSA 的瓶颈在 indexer 而非 Sparse MLA。

  **Block filtering 开销分析**：HISA 增加了 block filtering 阶段，但该阶段仅在 M = ceil(L/B) 个 pooled 摘要上操作。以 64K context 为例：M = 64K/128 = 500，远小于 L = 64K。Block filtering 的额外开销约 0.2 ms，但节省了跳过 ~56K 个 token 的 token-level 计算，净收益显著。

## Hardware-Efficient Attention for Fast Decoding

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了针对 MLA、GTA 和 GLA 三种注意力变体的高性能 CUDA 解码 kernel，核心优化包括三项：(1) **异步软件流水线 + Warp Specialization**：使用独立的 producer warp（TMA 或 cp.async 指令执行 HBM→SRAM 内存加载）和 consumer warp（执行 Tensor Core MMA），通过 warp scheduler 的异步特性实现内存加载与计算的重叠，保持 Tensor Core 始终处于满负荷状态。(2) **分布式偏移量计算（Distributed Offset Calculation）**：针对 Paged KV 场景，将地址计算任务分布到同一 warp 内的多个线程——128 线程分为 8 组（每组 16 线程），每组负责加载 8 行，每线程仅计算 1 行的 global memory 地址（而非 16 行），各线程通过 warp shuffle 共享地址。消除 page size 1 相对于 page size 64 的性能退化（1.2-1.5× speedup），解锁 RadixAttention prefix caching（需 page size 1）场景。(3) **Cooperative Softmax**：多 warp 协作执行 online softmax，通过 cross-warp shared memory reduction（sTMP 做 row-max，sAcc 暂存 attention scores 再 ldmatrix 重载保证 MMA alignment）实现跨 warp 的正确性。实验比较：(a) GLA kernel vs FlashMLA（Li, 2025）在 L_q=1（标准解码）和 L_q=2（推测解码）下的 TFLOPs 和带宽利用率；(b) Page size 1 vs page size 64 在有/无 distributed offset calculation 下的速度；(c) kernel latency（Table 44, 45）下 MLA DP vs GLA TP=2 在 2 GPU 上的延迟对比。

- 后端平台是什么，配置是什么。
  NVIDIA H100 80GB SXM5 GPU（BF16 峰值 989 TFLOPS/s，HBM 带宽 3350 GB/s，132 SMs，每 SM 256KB SRAM，总 on-chip SRAM 带宽 ~33 TB/s）。kernel 使用 CUDA + PTX 内联汇编编写，使用 FlashAttention-3 的 warp-specialized pipeline。多 GPU kernel latency 测试：2× H100 GPU。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 FlashAttention-3 CUDA kernel 框架修改，核心实现：

  1. **异步流水线 + Warp Specialization kernel**：
  - 输入：Q ∈ R^{B×1×hq×d}，K/V page table + paged KV cache
  - Producer warp：使用 TMA (Tensor Memory Accelerator) 或 cp.async 指令从 HBM 异步加载下一 KV block 到 shared memory
  - Consumer warp：执行 Tensor Core MMA（QK^T 和 PV），与 producer 的内存加载重叠
  - 基于 Bauer et al. (2014) 的 warp specialization 和 Thakkar et al. (2023) 的矩阵乘法优化，软件流水线参考 Lam (1988)

  2. **Distributed Offset Calculation kernel**（针对 Paged KV）：
  ```
  # 128 threads 加载 128×128 block
  # 分组：8 groups × 16 threads
  for t in 0..127 (thread index):
      g = t / 16                           # group 0..7
      # Step 1: 读 page table entry
      row = g + (t % 16) * 8
      page_idx = page_table[row]
      addr = compute_global_addr(page_idx, row)  # 64-bit 整数地址
      # Step 2: warp shuffle 共享地址
      for r in g, g+8, ..., g+120:
          src_thread = g*16 + (r-g)/8
          load_addr = __shfl_sync(addr, src_thread)
          # cp.async 加载 KV 元素
          cp.async(shared_mem[r], load_addr)
  ```
  关键优化：每个线程仅存储 1 行的地址（而非 16 行），通过 warp shuffle 在组内共享。消除 page size 1 下的地址计算瓶颈。

  3. **Cooperative Softmax**（Algorithm 1）：
  - W_m=1（解码 query length 小），增加 W_n 提高 warp 并行度
  - sTMP ∈ R^{W_n}：cross-warp reduction 计算 row-wise max（先 intra-warp __shfl_xor_sync，再 inter-warp shared memory reduction）
  - sAcc ∈ R^{T_m×T_n}：暂存 Tensor Core 寄存器中的 attention scores，通过 ldmatrix 重载确保 MMA interleaved layout 对齐
  - Online rescaling：O_new = P' @ V + diag(exp(m_old - m_new)) @ O_old

  4. **Hopper 特定优化**：
  - TMA (Tensor Memory Accelerator) 做 contiguous block 加载的地址计算和边界检查
  - cp.async 指令做 byte-aligned 非连续内存拷贝（用于 Paged KV page table 间接寻址）
  - 利用 Hopper 的异步执行模型：producer warp 的 TMA/cp.async 与 consumer warp 的 wgmma 自动重叠

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/Dao-AILab/grouped-latent-attention

  **评估原理与 Kernel 执行全流程（以 GLA-2 单解码步为例）**：

  ```
  输入：
    Q ∈ R^{1×hq×d}（当前 token 的 query）
    page_table ∈ R^{L}（paged KV 的页表，page size 可任意，含 1）
    K_cache, V_cache ∈ R^{L×hkv×d}（分页存储的 KV cache）
    latent heads: c_0^{KV}, c_1^{KV} ∈ R^{L×2dh}（GLA 的 latent caches）

  输出：
    O ∈ R^{1×d}（当前 token 的 attention output）

  Step 1: Query projection（在 FlashAttention-3 框架中）
    Q_proj = X @ W^Q  # cuBLAS gemm

  Step 2: GLA Decoding Kernel Launch
    grid = (batch, num_heads, seq_blocks)

    # --- Warp Specialization 异步流水线 ---
    # Producer Warp (warp 0):
    #   异步加载 KV block (i+1) 到 shared memory
    #   TMA: contiguous block → TMA copy
    #   cp.async: 非连续（paged）→ distributed offset calculation

    # Consumer Warp (warp 1,2,3):
    #   对当前 KV block (i) 执行:
    #   a. ldmatrix 从 shared memory 加载 KV tile 到寄存器
    #   b. mma.sync (Tensor Core): QK^T → S (attention scores)
    #   c. Cooperative Softmax（cross-warp via sTMP + sAcc）
    #   d. mma.sync: P @ V → partial O
    #   e. Online rescaling with running m (log-sum-exp)

    # Producer 的 load(i+1) 与 Consumer 的 compute(i) 重叠执行

  Step 3: 输出累积
    O = accumulate partial O across heads/ranks
    O = O @ W^O  # output projection
  ```

  **性能指标与原理**：
  - Roofline 分析（Figure 3, 4 Left）：测量实际 TFLOPs vs 内存带宽，判断 kernel 是 memory-bound 还是 compute-bound。H100 roofline: compute roof 989 TFLOPS, memory roof 3350 GB/s → BF16 算术强度转折点 ~295 FLOPs/byte
  - Kernel speed comparison：以 FlashMLA（Li, 2025, 28 March 2025 version）为 baseline，比较 decoding latency (μs)
  - Page size impact：固定 sequence length，变化 page size (1 vs 64)，测量有/无 distributed offset calculation 的 kernel 延迟

  **关键性能数据**：
  - GLA kernel L_q=1（标准解码）：~360 TFLOPS（vs FlashMLA ~610 TFLOPS，因 GLA 尚未触及 compute roof），但 GLA memory bandwidth 利用 93%，FlashMLA ~72%
  - GLA kernel L_q=2（推测解码）：~2× faster than FlashMLA——GLA 算术强度 ~128 FLOPs/byte 触及 H100 compute roof 而 MLA 已超出
  - Distributed offset calculation：page size 1 无减速（匹配 page size 64），1.5× speedup for page size 1，1.2× speedup for page size 64
  - Kernel latency（2 GPUs, batch=1, seqlen=131072）：GLA TP=2 55.0 μs vs MLA DP 81.0 μs（1.47× faster）
  - Imbalanced workload（batch=[1024×15 + 65536]）：GLA TP=2 42.6 μs vs MLA DP 56.0 μs（1.31× faster）

## MagicPIG: LSH Sampling for Efficient LLM Generation

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  GPU-CPU异构计算调度：GPU负责compute-bound的线性投影和随机投影（HashEncode），CPU负责memory-bound的KV cache哈希表查询和稀疏注意力计算。具体运行时流程：GPU上PyTorch执行所有线性层和LSH随机投影，然后将hash code和新KV传输到CPU；CPU上运行FBGEMM bfloat16内核执行稀疏qK^T和weighted V求和；CPU结果通过recursive attention与GPU上的on-device cache结果合并。实验比较了不同(K,L)超参数下的延迟和吞吐量（Table 7），以及三种硬件场景的系统性能（Figure 8）。

- 后端平台是什么，配置是什么。
  GPU: NVIDIA A100 (80GB HBM), L20 (48GB, 864GB/s带宽), 模拟RTX 4090 (24GB, ~1TB/s带宽)。CPU: Intel Platinum 8480+ / Intel 8563C。CPU DRAM带宽按150GB/s估算（论文实测group query attention size=4下的经验带宽）。

- 评估性能的软件/脚本是什么。修改了什么。
  GPU端：原生PyTorch (Paszke et al., 2019)。CPU端：FBGEMM (Khudia et al., 2021) bfloat16精度。修改内容：(1) 新增GPU端HashEncode kernel：q_code = Sign(q @ W)，将d维向量投影到K×L bit哈希码，所有attention head共享W，内存开销400KB~825KB；(2) 新增CPU端LSH哈希表数据结构：每个KV head L张哈希表，存储所有key的索引和哈希码，内存开销随context length和head数线性增长（如Llama-3.1-8B 96K context: 14GB for (10,150)）；(3) CPU端稀疏注意力计算：FBGEMM执行q·K_S^T的内积计算和加权求和。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源链接: https://github.com/Infini-AI-Lab/MagicPIG。评估原理和流程：
  评估原理：测量端到端LLM解码的wall-clock性能指标（token间延迟TBT、最大吞吐量tokens/sec、200ms延迟约束下的吞吐量Throughput200ms），量化不同(K,L)配置下的计算开销(Cost_2 = 采样后稀疏注意力FLOPs / 全注意力FLOPs)与系统性能的关系。
  Kernel输入到性能输出全过程：
    输入：prefill后的KV cache (n×d)、模型参数、LSH超参数(K,L)、随机投影矩阵W
    GPU端：每个decode step执行
      - 线性层投影 (compute-bound, GPU利用率高)
      - HashEncode: q_code = Sign(q @ W) → 传输到CPU
    CPU端：
      - HashTable查询：L次查找，收集S = {i | collision_count_i ≥ 2}
      - 稀疏注意力：加载K_S, V_S → FBGEMM计算q·K_S^T → softmax(· - log(u)) → Σ w_i·v_i
      - 结果传回GPU → recursive attention合并 → 输出
    性能输出：TBT (time between tokens, ms)、最大吞吐量(可容纳的最大batch size × 每秒tokens)、Throughput200ms (200ms延迟SLO下的最大吞吐量)
    关键发现：(K,L)=(10,150)下TBT=18.31ms、最大吞吐53.78 tokens/s、Throughput200ms=48.89 tokens/s；KV cache offload使batch size可达baseline的12×以上。

## MoBA: Mixture of Block Attention for Long-Context LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了融合 FlashAttention 和 MoE 优化技术的 MoBA attention kernel（Algorithm 1）。核心实现包含五个步骤：(1) 根据 gating network 和 causal mask 确定 query-to-KV-block 分配；(2) 按分配的 KV block 重排 query tokens 顺序；(3) 对每个 KV block 和分配给它的 queries 使用 FlashAttention varlen 分别计算 block-wise attention；(4) 将 attention outputs 重排回原始顺序；(5) 使用 online softmax (tiling) 合并 output——因为一个 query 可能关注当前 block 和多个历史 block。Attention 计算被分为两部分：self-attention block（当前 block, causal=True）通过 `get_self_attn_block` 处理，MoBA blocks（top-k 选中的历史 blocks, causal=False）通过 `index_select_moba_attn_block` 处理，两部分用 `combine_with_online_softmax` 合并。实验比较 MoBA kernel vs FlashAttention 在 1M model 上的 forward pass 时间（seqlen 8K-1M, Figure 2a）和固定 sparsity 95.31% 下的 scaling（8K-10M, Figure 2b）。

- 后端平台是什么，配置是什么。
  具体 GPU 型号论文未明确说明（标注为 Moonshot AI 内部集群）。使用 tensor parallelism 扩展至 query head level：将 K、V tensors broadcast 到 distributed query heads 以解决 10M context 下的 GPU 显存限制。基于 FlashAttention (Dao et al. 2022) 和 DeepSpeed-MoE (Rajbhandari et al. 2022) 实现。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch + FlashAttention + DeepSpeed-MoE。核心实现/修改：
  
  1. **KV Block Splitting + Mean Pooling**（Algorithm 1 Lines 1-4）：将 K, V ∈ R^{N×h×d} 按 block size B 划分为 n=N/B 个 block K̃_i ∈ R^{B×h×d}，mean_pool 沿 sequence 维度计算 K̄ ∈ R^{n×h×d} 作为 block-level key representation。

  2. **Gating Score + Top-k Selection**（Lines 5-8）：Q @ K̄^T → S ∈ R^{N×h×n}，加 causal mask M（future blocks = -∞），topk(S+M, k) → G ∈ {0,1}^{N×h×n}，得到稀疏 query-to-block mapping。

  3. **Varlen FlashAttention Computation**（Lines 9-14）：`index_select_moba_attn_block` 根据 G 将 queries 分组到各 KV block，输出变长序列 Q^m/K̃^m/Ṽ^m；`flash_attention_varlen` 对每个 (query_group, kv_block) 对执行 FlashAttention；当前 block attention（`causal=True`）和历史 block attention（`causal=False`）分别计算。

  4. **Online Softmax Combining**（Line 16）：使用 online softmax tiling (Milakov et al. 2018; Liu et al. 2023) 将 self-attention output O^s 和 MoBA attention output O^m 合并——因为一个 query 可能同时关注当前 block 和多个历史 blocks，需从不同 attention 分片的 partial softmax 合并出最终结果。

  5. **Tensor Parallelism for 10M context**：将 K、V broadcast 到不同 query heads（tensor parallelism over heads），各 head 独立持有完整 K/V 但仅计算自己负责的 Q heads 的 attention，有效突破单 GPU 显存限制。

  6. **MoBA/Full Attention Switching**：attention 层可在 MoBA 和 full attention 间动态切换，gating 计算仅在 MoBA 模式下触发。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://github.com/MoonshotAI/MoBA

  **评估原理与 Kernel 执行全流程（以单个 MoBA attention layer prefill 为例）**：

  ```
  输入: Q, K, V ∈ R^{N×h×d}（N=context length, h=num_heads, d=head_dim）
       B=block_size, k=top_k
  输出: O ∈ R^{N×h×d}

  Step 1: Block Partitioning [GPU: split]
    n = N / B
    for i in 1..n:
        K̃_i = K[(i-1)B : iB]  # [B, h, d], slice from HBM
        Ṽ_i = V[(i-1)B : iB]

  Step 2: Block Mean Pooling [GPU: reduce]
    K̄ = mean_pool(K, dim=0, group_size=B)  # reduce along seq dim
    # K̄ ∈ R^{n×h×d}, each entry is mean of B key vectors

  Step 3: Gating Score [GPU: bmm]
    S = Q @ K̄.transpose(-1, -2)  # [N, h, n]
    # S[q_idx, head, block_idx] = affinity of query q to KV block block_idx

  Step 4: Causal Mask + TopK [GPU: mask + topk]
    M[pos][i] = -inf if pos < i*B  # no future blocks
    G = topk(S + M, k, dim=-1)  # [N, h, n], binary
    # G[q][b] = 1 if block b is selected for query q

  Step 5: Query-to-Block Assignment [GPU: index_select]
    # Self-attention block
    for each block i:
        Q_i^s = queries whose position ∈ I_i  # queries in this block
        K_i^s = K̃_i, V_i^s = Ṽ_i
    # MoBA history blocks
    Q^m, K̃^m, Ṽ^m = index_select_moba_attn_block(Q, K̃, Ṽ, G)
    # Group queries by their assigned blocks, output varlen tensors

  Step 6: FlashAttention Varlen [GPU: FlashAttn kernel]
    # Self-attention (current block only, with causal mask)
    O^s = flash_attn_varlen(Q^s, K̃^s, Ṽ^s, causal=True)
    # MoBA attention (selected history + current blocks, no causal mask
    #   because causal is already enforced by block-level routing)
    O^m = flash_attn_varlen(Q^m, K̃^m, Ṽ^m, causal=False)

  Step 7: Online Softmax Combining [GPU: fused kernel]
    O = combine_with_online_softmax(O^s, O^m)
    # Principle: lse_i (log-sum-exp) from each partial attention,
    # re-weight and re-scale O^s and O^m with their respective lse
    # Equivalent to:
    #   lse_total = logsumexp([lse_s, lse_m])
    #   O = (exp(lse_s - lse_total) * O^s + exp(lse_m - lse_total) * O^m)
    # In practice: online tiling, no explicit lse materialization
  ```

  **评估指标与性能**（Figure 2）：
  - 1M model speedup (Figure 2a): 8K→1M 序列长度，MoBA vs FlashAttention forward pass time
    - 1M tokens: MoBA ~0.15s vs FlashAttn ~1.0s → **6.5× speedup**
    - 随序列增长 MoBA 呈 sub-quadratic scaling
  - Fixed sparsity ratio scaling (Figure 2b): 8K→10M, fixed blocks=64, top-k=3, sparsity=95.31%
    - 10M tokens: MoBA ~1.5s vs FlashAttn ~24s → **16× speedup**
    - 小序列（32K-512K）两者接近，长序列下优势显著

  **FlashAttention 集成要点**（Figure 1b）：
  - FlashAttention blocks 内嵌 MoBA 的 block routing 逻辑
  - Q^m, K̃^m, Ṽ^m 已经按 block 分组排列，FlashAttention varlen 直接处理
  - "Varlen" 指各 block 的 query count 不同（取决于 top-k routing 结果）
  - 最终 online softmax combine 保证数值等价于完整 softmax attention

## Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了三个核心 GPU kernel/操作：(1) **Triton 压缩 kernel**：在 decode 阶段实时对剪枝后的 KV cache 进行 bitmap-based 压缩，将非零元素按 tile（1×64 列）打包，利用 GPU 并行加速压缩过程；(2) **Custom CUDA SpMV kernel**：对 bitmap-compressed 稀疏 KV cache 执行 batch SpMV（稀疏矩阵-向量乘），采用 load-as-compressed, compute-as-dense 范式——从 GPU global memory 以压缩格式加载到寄存器，在 shared memory 中解压为稠密 tile，然后执行 tile-wise dense 计算，有效减少 memory-bound decode 阶段的 global memory 数据搬运；(3) **Dense local window MV**：对最近 32 token 的 dense KV cache 执行标准 batch MV。

  实验比较：(a) Mustafar sparse attention kernel 各组件延迟拆解：SpMV vs dense MV of local window vs runtime pruning vs compression，与 cuBLAS dense batch MV 对比；(b) 不同稀疏度 (50%/70%) 下的 kernel 加速比；(c) 不同 batch size (1-8) 下的吞吐量 (tokens/sec) vs dense FlashAttention decode；(d) KV cache 压缩比 vs LongBench 精度，与 ThinK 对比 Pareto 曲线；(e) 不同 input:output token ratio 下的 decode speed（decode 512/1024/2048 tokens）；(f) Llama-2-7B (MHA) 与 Llama-3-8B (GQA) 的不同表现。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 6000 Ada GPU（48GB VRAM）。性能测量使用 NVIDIA Nsight Profiling Tool。推理使用 bfloat16 精度。实验配置：Llama-2-7B 输入 seqlen=2048，生成 1024/2048 tokens；Llama-3-8B 输入 seqlen=4096，生成 1024/2048/4096 tokens。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch + Triton（压缩 kernel）+ CUDA（SpMV kernel）实现。核心修改：

  1. **Triton 压缩 kernel**：接收剪枝后的稀疏 KV cache 和 binary mask，输出 bitmap-compressed 格式。每 tile 64 个元素，GPU 并行处理所有 token × tiles。输出格式包含：tile_offset（寻址正确非零起始位置）、bitmap（64-bit 表示非零位置）、compressed nonzeros。

  2. **Custom CUDA SpMV kernel**：基于 Coruscant [20] 的 bitmap-based 稀疏格式，遵循 FlashLLM [43] 的 load-as-compressed, compute-as-dense 范式：
  - 每个 warp thread 处理一个 1×64 thread-tile
  - 每 warp 操作一个 64×64 matrix tile
  - Pipeline：gmem2reg（压缩格式从 global memory 加载到寄存器）→ extract（解压到 shared memory）→ smem2tc（送入 Tensor Core 计算）
  - 未使用的 N 维度补零对齐 Tensor Core FP16 计算
  - 每 warp thread 每 pipeline stage 解压 2 个 thread-tile，用 bitmap 确定非零位置

  3. **KV Cache Management**：
  - Key cache：列 tiling 沿 token 维度（因 Key 乘在 channel 维）
  - Value cache：列 tiling 沿 channel 维度（因 Value 乘在 token 维）
  - Channel-major 遍历：新 token 的压缩 KV 可追加到末尾
  - Token group of 64：新增 KV cache 需累积满 64 token 组才压缩追加

  4. **Prefill-Decode 兼容**：
  - Prefill 使用 FlashAttention [6]（不受影响）
  - Prefill 结束后 KV cache 被剪枝并压缩
  - Decode 生成的 KV cache 先 dense 保留在 local window，退出 window 后剪枝压缩

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码开源：https://github.com/dhjoo98/mustafar（包含 pruning mechanism + dense local attention + custom CUDA SpMV kernel）。

  **Kernel 评估原理与执行全流程（以 Llama-2-7B MHA decode step, K_s=0.5, V_s=0.5 为例）**：

  ```
  输入：
    Q_t ∈ R^{1×d}（当前 token query, d=4096/32heads×128dim）
    K_C ∈ bitmap-compressed sparse format（T-W 个 token 的压缩 Key cache）
    V_C ∈ bitmap-compressed sparse format（T-W 个 token 的压缩 Value cache）
    K_L, V_L ∈ R^{W×d}（local dense window, W=32 tokens）
    bitmap 格式: [(tile_offset_uint16, bitmap_uint64, nonzeros_fp16[popcount]) ... ]

  输出：
    O_t ∈ R^{1×d}（当前 token attention output）

  Step 1: Runtime Pruning + Compression（Triton kernel，新退出 local window 的 token）
    # 对刚退出 local window 的 token（第 T-W 个 token）:
    mask_K = |K_cache[T-W]| >= threshold_K     # per-token magnitude threshold
    mask_V = |V_cache[T-W]| >= threshold_V
    # Compress into bitmap format:
    for tile in 0..ceil(d/64):
        bitmap = pack_64bit(mask_K[tile*64:(tile+1)*64])
        nonzeros = gather and pack non-zero elements
    # Append to K_C, V_C (channel-major traversal ensures contiguous append)

  Step 2: Compressed KV Cache Attention — SpMV（Custom CUDA kernel）
    # Kernel grid: num_heads × ceil((T-W) / 64)
    # Load-as-compressed, compute-as-dense:
    for each head h:
        for each warp-tile of 64 compressed tokens:
            # Pipeline Stage 1: gmem2reg
            Load compressed tile from HBM to registers
            # 包含: tile_offset[], bitmap[], compressed nonzeros[]

            # Pipeline Stage 2: extract（decompression）
            for each thread (1×64 thread-tile):
                use bitmap to place nonzeros into dense 1×64 in shared memory
            # shared memory now holds dense K_C_tile ∈ R^{64×d_h}

            # Pipeline Stage 3: smem2tc (Tensor Core GEMM)
            S_C_partial = Q_t[h] @ K_C_tile^T  # [1×d_h] @ [d_h×64] → [1×64]
            # Online max rescaling for numerical stability

  Step 3: Dense Local Window Attention — Batch MV（cuBLAS or custom kernel）
    S_L = Q_t @ K_L^T         # [1×d_h] @ [d_h×32] → [1×32]

  Step 4: Softmax Merge
    S_full = concat(S_C, S_L)  # [1×(T-W)+32]
    A = softmax(S_full / sqrt(d_h))
    [A_C, A_L] = split(A, at T-W)

  Step 5: Output Computation
    # Sparse: A_C @ V_C via batch SpMV（同 Step 2 pipeline 但用 V_C）
    O_C = batch_SpMV(A_C, V_C_bitmap)  # [1×d_h]
    # Dense: standard MV for local window
    O_L = A_L @ V_L                    # [1×d_h]
    O_t[h] = O_C + O_L
  ```

  **延迟拆解（Llama-2-7B, T=2048, KV 50% sparsity, vs cuBLAS dense batch MV）**：
  - Pruning overhead: 1.84% of cuBLAS time
  - Compression overhead: 6.25% of cuBLAS time
  - Dense local window MV: 0.62% of cuBLAS time
  - SpMV (main computation): 81.07% of cuBLAS time
  - **Total: 89.78% of cuBLAS time → 1.11× speedup**

  **KV 70% sparsity**：
  - SpMV: 61.87% of cuBLAS time → ~1.25× speedup（含 overhead）

  **GQA 架构（Llama-3-8B）**：pruning overhead 1.47%, compression 0.47% — GQA 减少 KV head 数降低了剪枝和压缩开销。

  **吞吐量提升（end-to-end vs dense FlashAttention）**：
  - Llama-2-7B, batch=8, seq in=2048, gen=2048: KV 50% sparsity → tokens/sec 高于 dense
  - Llama-3-8B, batch=8 vs dense batch=6: Mustafar 50% sparsity → 2.23× tokens/sec（支持更大 batch + 更快推理）
  - Batch=1 时 Mustafar 吞吐低于 dense（SpMV kernel 未充分利用 GPU SMs，threadblock 数 < SM 数）
  - TTFT（time-to-first-token）增加（prefill 后 pruning+compression 开销）但被 decode 加速补偿

  **KV Cache 压缩比**：
  - K_s=0.5, V_s=0.5: 65% of dense KV cache（bitmap+tile offset 约 15% overhead）
  - K_s=0.7, V_s=0.7: 45% of dense KV cache
  - Single-cache 70%: 72.5% of dense

## NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了 Reduce Attention Scores CUDA kernel，与 FlashAttention-2 兼容，用于在 encoding 阶段高效计算 attention scores 的列向归约（per-token importance scores），避免重新计算完整 attention matrix。两种实现方式：(1) **重计算方式**：利用 FlashAttention-2 forward 返回的 log-sum-exp（LSE），按 backward pass 的方式重算 attention scores 矩阵，再做 column-wise sum 得到 reduced attention scores（Algorithm 2）；(2) **小矩阵重计算方式**：仅对 proxy tokens（~10% tokens）重新计算 attention scores，因 proxy token 数量远小于总 token 数，额外开销可忽略。128K context 下 evict 20% 维持 15GB 稳定显存。实验：128K context 推理的显存和速度兼容性验证。

- 后端平台是什么，配置是什么。
  单张 NVIDIA A100 80GB GPU。基于 FlashAttention-2 的 CUDA kernel 实现。128K context length 测试。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 FlashAttention-2 的 backward pass 逻辑实现。核心修改：
  1. **Reduce Attention Scores Kernel（Algorithm 2）**：利用 FlashAttention-2 forward 输出的 LSE vector L ∈ R^{N_q}，在 SRAM 上分 tile 重算 P_i^{(j)} = exp(S_{ij} - L_i)，再做 column-wise reduce R_j += Reduce(P_i^{(j)})，输出 per-key 的累积 attention scores O ∈ R^{N_k}。分块策略与 FlashAttention-2 一致（T_r = ceil(N_q/B_r), T_c = ceil(N_k/B_c)）。
  2. **小矩阵重计算方式**：仅对 proxy token subset P 和完整 K 计算 attention scores，Q_proxy ∈ R^{|P|×d}，K ∈ R^{N_k×d}，计算量仅 O(|P|·N_k·d) 而非 O(N_q·N_k·d)。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://github.com/PaddlePaddle/Research/tree/master/NLP/ACL2024-NACL

  **Reduce Attention Scores Kernel 执行全流程（Algorithm 2, FlashAttention-2 兼容）**：

  ```
  输入：
    Q ∈ R^{N_q×d}, K ∈ R^{N_k×d} in HBM
    Logsumexp L ∈ R^{N_q} in HBM（FlashAttention-2 forward 输出）
    block sizes B_c, B_r
  输出：O ∈ R^{N_k}（per-key 的 reduced attention scores）

  Step 1: 分块
    Q → T_r = ceil(N_q/B_r) blocks
    K → T_c = ceil(N_k/B_c) blocks
    L → T_r blocks
    O = zeros(N_k) in HBM → T_c blocks

  Step 2: 逐 K block 计算（外层循环）
    for j = 1..T_c:
      Load K_j from HBM → SRAM     # K_j ∈ R^{B_c×d}
      R_j = zeros(B_c) on Register  # per-block reduced scores

  Step 3: 逐 Q block 计算（内层循环）
      for i = 1..T_r:
        Load Q_i, L_i from HBM → SRAM  # Q_i ∈ R^{B_r×d}, L_i ∈ R^{B_r}
        S_i^{(j)} = Q_i @ K_j^T          # ∈ R^{B_r×B_c}, on-chip matmul
        P_i^{(j)} = exp(S_i^{(j)} - L_i) # ∈ R^{B_r×B_c}, online rescale
        R_j = R_j + Reduce(P_i^{(j)})    # column-wise sum, ∈ R^{B_c}

  Step 4: atomicAdd to HBM
      atomicAdd(O_j, R_j)               # 累加到全局输出

  返回 O ∈ R^{N_k}，即每个 key token 的累积 attention score
  ```

  **评估原理**：128K context 下 NACL evict 20% KV cache，测量 GPU 显存使用量（维持 ~15GB 稳定），确认 kernel 开销不显著影响推理吞吐。小矩阵重计算方式因 |P| ≪ N_q 而开销可忽略。

## Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  Quest 在 FlashInfer kernel 库（Ye et al., 2024）基础上实现了三个专用 CUDA kernel：(1) **Criticality Estimation kernel**：将每 page 的 channel-wise min/max Key 元数据（2 个 float16/通道/page）与当前 Query 向量做 element-wise 乘法和 max 选择，求和得到 per-page criticality score；(2) **Top-K Filtering kernel**：使用 RAFT（RAPIDS, Zhang et al. 2023a）的 batched Top-K CUDA operator，从所有 page score 中选出 Top-K 索引，延迟仅 5-10 µs（seq_len < 128K）；(3) **Approximate Attention kernel**：基于 PageAttention 实现，将 Top-K page 索引传入 FlashInfer 的 sparse page 加载接口，仅对选中 page 执行标准 FlashAttention。

  实验比较：(a) Per-kernel micro-benchmark（NVBench）：criticality estimation 延迟 vs FlashInfer full attention，不同 seq_len 和 page_size；(b) Approximate attention 延迟 vs FlashInfer，不同 token budget 和 seq_len，page_size=16；(c) Combined kernel time breakdown（PyTorch Profiler）：Quest attention（estimation + Top-K + approximate）vs FlashInfer full attention，Llama2-7B 配置，seq_len=8K/16K/32K，token budget=2048。

- 后端平台是什么，配置是什么。
  GPU：NVIDIA RTX 4090（kernel 级 micro-benchmark），NVIDIA Ada 6000 48GB（端到端长 context 评估）。CUDA 12.2。模型配置：Llama2-7B（32 layers, 32 heads, d_head=128, FP16）。FlashInfer 作为 attention kernel 库。量化：支持 4-bit 权重量化（端到端场景）。

- 评估性能的软件/脚本是什么。修改了什么。
  软件：FlashInfer（kernel library for LLM serving），RAFT（GPU-accelerated vector search library，提供 batched Top-K CUDA kernel），NVBench（NVIDIA GPU kernel micro-benchmarking tool），PyTorch Profiler（端到端 kernel 时间分解）。修改内容：在 FlashInfer 中新增三个 CUDA kernel——criticality estimation（读取 page metadata 替代完整 KV cache 加载），Top-K filtering（调用 RAFT），approximate attention（基于 PageAttention sparse page loading）。同时维护 per-page metadata buffer（min/max Key values, 2 × d_head × page_size 个 FP16）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  Quest 开源：https://github.com/mit-han-lab/Quest。基于 FlashInfer（https://flashinfer.ai）和 RAFT（https://github.com/rapidsai/raft）。

  **Kernel 评估流程（NVBench，Fig. 8）**：

  (a) **Criticality Estimation kernel 评估**：
  ```
  输入：Query Q ∈ R^{B×d_head}, Metadata M,m ∈ R^{num_pages×d_head}
  原理：测量 element-wise max(Q*m, Q*M) + reduce-sum 的 GPU 延迟
  对比 FlashInfer full attention（加载完整 K cache + QK^T matmul）
  评估参数：seq_len ∈ {4K, 8K, 16K, 32K, 64K}, page_size ∈ {16, 32, 64}
  输出：latency (µs)，随 seq_len 增长 latency 相对 FlashInfer 比例趋近 1/PageSize
  ```

  (b) **Approximate Attention kernel 评估**：
  ```
  输入：Q ∈ R^{B×d_head}, 选中 page 的完整 K,V ∈ R^{K×S×d_head}
  原理：PageAttention sparse loading + FlashAttention 计算
  对比：FlashInfer 标准 attention at seq_len=B (token budget)
  输出：给定 token budget B，latency 为常数（与总 seq_len 无关），接近 FlashInfer@seq_len=B
  ```

  (c) **Combined Quest Attention 评估（PyTorch Profiler, Fig. 9）**：
  ```
  输入：完整推理的单步 decode（Llama2-7B, single batch）
  测量：Criticality estimation + Top-K filtering + Approximate attention 的合计 wall-clock 时间
  对比：FlashInfer full attention（加载完整 KV cache + GQA attention）
  结果：seq_len=32K, token budget=2048 → 7.03× self-attention speedup
  时间分解：estimation + Top-K ≪ approximate attention ≪ full attention KV loading
  ```

  **Kernel 数据流**：
  ```
  GPU HBM:
    [Full KV Cache: 2 × seq_len × d_head × FP16]
    [Page Metadata: 2 × num_pages × d_head × FP16]  ← Quest 额外存储

  Step 1: Criticality Estimation (memory-bound, O(num_pages × d_head))
    HBM → SM registers: Q (d_head), M_p (d_head), m_p (d_head)
    Compute: per-channel Q_i * max(m_i^p, M_i^p), sum → scalar score_p
    Output: scores ∈ R^{num_pages} (SRAM)

  Step 2: Top-K Filtering (compute-bound, O(num_pages × log K))
    scores → RAFT batched Top-K → top_k_indices ∈ Z^K
    Latency: 5-10 µs (negligible vs KV loading)

  Step 3: Approximate Attention (memory-bound, O(K × S × d_head))
    HBM → SM: Q, K[top_k_indices], V[top_k_indices] (via PageAttention sparse load)
    Compute: S = QK^T/√d_head, softmax, O = SV (FlashAttention tiling)
    Output: O ∈ R^{B×d_head}
  ```

  Memory load reduction: Quest 加载 1/PageSize + K/PageNum of total KV cache。如 page_size=16, 64K context (4096 pages), K=256 pages → ~1/16 + 256/4096 ≈ 12.5%，约 8× 减少。

## SeerAttention-R: Sparse Attention Adaptation for Long Reasoning

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了块稀疏 Flash Decoding Kernel（TileLang 和 Triton 两个版本），专门为 block-sparse attention 在 decode 阶段设计。该 kernel 扩展了 FlashAttention decoding pattern，支持动态块稀疏性，接收 AttnGate 输出的选中 block indices，在 kernel 内部只遍历选中的 KV blocks，跳过无效 entries。实验对比了 TileLang kernel、Triton kernel 和 FA3（FlashAttention-3）dense baseline，在不同序列长度（8k-128k）、batch sizes（1-16）、sparsity ratios（0.5-0.9）下的加速比。

- 后端平台是什么，配置是什么。
  NVIDIA H100 GPU（Section 4.4）。
  GQA 配置：64 attention heads, 8 key-value heads, head dimension 128。
  利用 H100 的 wgmma 指令提升 Tensor Core 利用率，将 query head group 数 padding 到 64。

- 评估性能的软件/脚本是什么。修改了什么。
  软件：TileLang（https://github.com/tile-ai/tilelang）和 Triton（https://github.com/triton-lang/triton）。Baseline：FlashAttention-3 (FA3) dense decoding kernel。
  修改：从零实现了 block-sparse 版本的 flash decoding kernel，具体修改包括：
  - grid scheduling strategy: 3D launch space over (batch, heads_kv, num_split)
  - 只遍历 selected_block_indices，跳过无效 entries
  - num_split 维度按 max_selected_blocks 分割（而非 total_blocks），改善 SM 负载均衡
  - H100 专用优化：wgmma 指令 + query head group padding to 64

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  代码仓库：https://github.com/microsoft/SeerAttention

  Kernel 执行全流程：

  **输入阶段：**
  ```
  Inputs:
    Q ∈ R^{batch, num_kv_heads, d_head}         (decode 单 token)
    K_cache ∈ R^{seq_len, num_kv_heads, d_head}  (HBM 中的 KV cache)
    V_cache ∈ R^{seq_len, num_kv_heads, d_head}
    blocked_indices ∈ R^{batch, num_kv_heads, max_selected_blocks}  (来自 AttnGate)
    block_size ∈ {64, 128}  (block sparsity 粒度)
    sm_scale = 1/sqrt(d_head)
  ```

  **调度策略（Grid Launch）：**
  ```
  grid = (batch, num_kv_heads, num_splits)
  num_splits = ceil(max_selected_blocks / BLOCKS_PER_SPLIT)
  // 关键优化：按 max_selected_blocks 而非 total_blocks 划分 split，
  // 确保 sparsity 不均匀时 SM 间负载均衡
  ```

  **Kernel 内部执行（每个 SM）：**
  ```
  For each selected block_idx in blocked_indices[batch, head_kv, :]:
    // 1. HBM → SRAM: 加载对应 block 的 K, V
    K_block = load_tile(K_cache[block_idx * block_size : (block_idx + 1) * block_size, :])
    V_block = load_tile(V_cache[block_idx * block_size : (block_idx + 1) * block_size, :])

    // 2. 计算 QK^T (Tensor Core, wgmma on H100)
    S_block = Q @ K_block^T * sm_scale  // [1, block_size]

    // 3. Online softmax rescaling (FlashAttention 标准流程)
    m_new = max(m_prev, rowmax(S_block))
    O = diag(exp(m_prev - m_new)) * O_prev + exp(S_block - m_new) @ V_block
    m_prev = m_new

  Output: O ∈ R^{batch, num_kv_heads, d_head}
  ```

  **TileLang 相比 Triton 的优化：**
  TileLang 自动应用以下优化（基于 target architecture）：
  - Tiling: 自动确定最优 tile size
  - Warp specialization & pipelining: 计算与访存重叠
  - Tensorization, rasterization, swizzling: HBM 访存模式优化

  **评估原理：**
  对固定的 (seqlen, batch_size, sparsity) 组合，分别运行 TileLang kernel、Triton kernel 和 FA3 dense kernel，测量 wall-clock time，计算 speedup = T_FA3 / T_sparse。理论加速比 = 1 / (1 - sparsity)。例如 90% sparsity → 1/(1-0.9) = 10× 理论上限。

  **关键结果：**
  - bs=16, seqlen≥32k, 90% sparsity: TileLang kernel 达到 8.6× 加速（接近 10× 理论上限），比 Triton kernel 快 1.7×
  - bs=4, seqlen=32k, 90% sparsity: 仍有 6× 加速
  - 序列越长、batch 越大，加速越接近理论上限（decode kernel 为 I/O-bound）

## ShadowKV__KV_Cache_in_Shadows_for_High-Throughput_Long-Context_LLM_Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  ShadowKV 的实现中包含多个自定义 CUDA kernel 和运行时调度优化：(a) **Attention approximation kernel**：基于 landmarks（chunk 均值）的近似注意力分数计算，包括 chunk-level MatMul(Q, L^T)、Softmax、聚合、ArgTopK 的 fused kernel；(b) **Key cache low-rank reconstruction kernel**：GPU 端从低秩投影 A、B 和 chunk indices I 通过 MatMul(Gather(A, I), B) 重建稀疏 key cache 的 fused gather+matmul kernel；(c) **Value cache fetching kernel**：CPU→GPU 的 value cache 批量 gather 操作，通过 PCIe 传输调度；(d) **Cache mechanism kernel**：利用 KV cache temporal locality 的 index scan kernel，比较相邻 decoding step 的 chunk 选择索引，检测 miss chunks 以跳过命中部分的重复重建和 PCIe 传输；(e) **Multi-stream overlap 调度**：CUDA multi-stream 将 key 低秩重建（GPU compute-bound）与 value CPU 抓取（PCIe I/O-bound）放入不同 stream 并发执行，实现计算与数据传输重叠。实验比较不同 chunk size 下的 batch size 上限和命中率，不同 context 长度下 decoding 的逐操作延迟分解（GEMM+Softmax、Max、TopK、Recompute K (Overlapped)、Fetch V、Attention、FFN、QKV），以及不同 chunk size / rank / sparse budget 下的准确率消融。

- 后端平台是什么，配置是什么。
  NVIDIA A100 80GB PCIe GPU，CUDA 多流并发，GPU 内存带宽 2 TB/s，PCIe 带宽 31.5 GB/s。

- 评估性能的软件/脚本是什么。修改了什么。
  评估基于 PyTorch + FlashAttention（FlashAttention-2）+ FlashInfer（fused layer norm 等）+ CUTLASS + vLLM（PagedAttention）的集成框架。修改/新增的 kernel 和运行时组件：
  1. **SVD kernel**：prefilling 阶段对 pre-RoPE key cache 的在线截断 SVD 计算
  2. **Landmark 构建 kernel**：post-RoPE key 按 chunk 规约求均值 + cosine similarity 检测 outlier
  3. **Fused attention approximation kernel**：Q 与 landmarks L 的 MatMul + Softmax + 聚合 + TopK，减少 kernel launch 和中间数据搬运
  4. **Fused key reconstruction kernel**：Gather(A, I) + MatMul(gathered_A, B) 的 fused kernel，减少显存搬运
  5. **Cache index scan kernel**：相邻 decoding step chunk indices 的 miss detection
  6. **Multi-stream 编排**：将 key 低秩重建与 value CPU 抓取分配到不同 CUDA stream 实现 overlap

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV（Apache 2.0，CUDA 内核占代码量 17.4%）。Kernel 运行时执行流程（以 decoding 阶段单 Transformer block 为例）：

  **输入**：Q ∈ R^{b×h_q×1×d}（batch=24, 8 query heads, 1 token, 128 dim），landmarks L ∈ R^{b×h_kv×n_c×d}（n_c=16K chunks for 128K context/chunk_size=8），低秩投影 A ∈ R^{b×128K×160}、B ∈ R^{b×h_kv×160×128}，V_CPU ∈ CPU memory，K_outlier、V_outlier ∈ GPU

  **Stage 1 — Approx Attention（Fused Kernel）**：
  ```
  // CUDA kernel launch 1: fused Gemm-Softmax-AggTopK
  P[h, nc] = sum_d(Q[h, d] * L[h, nc, d])     // MatMul, compute-bound
  S[h, nc] = softmax(P / sqrt(d))               // online softmax
  S_agg[nc] = sum_h(S[h, nc])                   // aggregate across query heads
  S_kv[nc] = max over kv_group(S_agg)            // GQA mapping
  I = TopK(S_kv, k=256)                          // select top-256 chunk indices
  I_miss = CompareAndMiss(I, I_prev)             // cache mechanism scan
  // 输出：I_miss ∈ R^{h_kv × k'}（k' ≈ k × (1-0.6) ≈ 102）
  ```

  **Stage 2 — Key Reconstruct + Value Fetch（Multi-Stream Overlap）**：
  ```
  Stream 1 [GPU compute]:
    A_selected = Gather(A, I_miss)               // index gather
    K_lowrank = MatMul(A_selected, B)            // reconstruct keys
    K_recon = RoPE(K_lowrank)                    // apply RoPE
    // 耗时 ~1.36ms for batch=24×128K

  Stream 2 [PCIe→GPU]:
    V_selected = cudaMemcpyAsync(V_CPU[I_miss], H2D, Stream2)
    // PCIe 传输 (256+48) * 8 * 1024 bytes = ~2.4MB per head
    // 总传输量取决于 batch × h_kv
    // 耗时 ~1.66ms for batch=24×128K

  cudaStreamSynchronize(both)
  // 总延迟 = max(1.36ms, 1.66ms) ≈ 1.84ms（含同步开销）
  // 若无 overlap：1.36 + 1.66 = 3.02ms
  ```

  **Stage 3 — Sparse Attention**：
  ```
  K_final = Concat(K_outlier, K_recon, K_new_token)
  V_final = Concat(V_outlier, V_selected, V_new_token)
  O = FlashAttention(Q, K_final, V_final)
  // 仅对 (48+256)*8 ≈ 2432 tokens 做 attention，而非完整 128K tokens
  // 耗时 ~0.21ms vs Full Attention ~0.5ms+
  ```

  **性能评估原理**：通过延迟分解（Table 13）测量每操作的 wall-clock time：GEMM+Softmax 0.58ms、Max+TopK 0.22ms、Recompute K(overlapped) 1.25ms、Fetch V 1.84ms、Attention 0.21ms、FFN 0.29ms、QKV 0.05ms，总计约 4.44ms/decoding step/Transformer block。对比 Full Attention 约 10ms+，加速主要来自：(1) sparse attention 只计算 1.56% tokens；(2) multi-stream overlap 隐藏 key 重建延迟；(3) cache mechanism 减少 60% 重复操作；(4) fused kernel 减少中间数据搬运和 launch overhead。chunk size 对性能的影响：chunk size 越大，landmark 数量越少（n_c=S/C 越小），近似注意力计算更快但精度下降，论文选择 C=8 作为准确率-效率平衡点。

## ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现多个自定义 CUDA kernel 用于 ShadowKV 的 sparse attention 解码流程：(1) **Landmark Attention Approximation Kernel**：将 Q 与 landmarks L 的矩阵乘法和 softmax 融合为单个 kernel，使用 chunk-level 粒度减少计算量（从 O(S×d) 降至 O(S/c×d)），输出 top-k chunk indices；(2) **Key Cache Low-Rank Reconstruction Kernel**：基于预存储的低秩投影 A、B 和选中的 chunk indices，通过 Gather(A[I]) × B 的 GEMM 重建 sparse K cache，利用 Tensor Core 加速；(3) **Value Cache Fetching + Overlap Kernel**：使用 CUDA multiple streams 将 CPU→GPU 的 value 数据传输（cudaMemcpy H2D via PCIe）与 key 重建 GEMM 重叠执行，通过 CUDA event 同步；(4) **Temporal Locality Cache Kernel**：维护 chunk index 的环形缓冲区，通过 index scan 检测相邻解码步的 chunk 选中重复，跳过已驻留 GPU 的 KV 对的取回和重建；(5) **融合 Kernel**：将 attention approximation、key cache reconstruction、value cache fetching、cache mechanism 等操作用 CUTLASS 融合以减少 memory movement 和 kernel launch overhead。

  实验比较：(1) 吞吐量实验：A100 上多模型多上下文下的 tokens/s 对比；(2) 延迟分解实验：48×64K / 24×128K / 12×256K / 6×512K 下各操作（GEMM+Softmax, Max, TopK, Recompute K, Fetch V, Attention, FFN, QKV）的延迟分解，显示重叠后 key 重建延迟被 value 取回延迟所掩盖；(3) Pre-filling 延迟分解：64K-512K 上下文下 SVD、Reduce、CosineSimilarity、TopK、Gather 的 overhead 占比分析。

- 后端平台是什么，配置是什么。
  NVIDIA A100 GPU (80GB)。GPU 内存带宽 2 TB/s (HBM2e)，Tensor Core 支持 FP16/BF16/FP8 等。PCIe 4.0 x16 连接 CPU，理论带宽 31.5 GB/s。配套 CPU 端为 x86 服务器平台，用于存放 offloaded value cache 和执行异步数据传输。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 PyTorch 框架，使用 CUTLASS 编写自定义 CUDA kernel，集成 FlashAttention (v2) 和 FlashInfer 的高效 attention 和 layer norm kernel。评估脚本：自定义的 latency/throughput benchmark 脚本，测量每层的 attention/FFN/SVD 等操作延迟（ms），以及整体生成吞吐（tokens/s）。

  修改的 kernel 操作：
  - Pre-filling 新增 kernel：SVD (cuSOLVER gesvd)、Reduce（chunk-mean 归约）、CosineSimilarity（chunk 内 cosine similarity + TopK argmin）、Gather（按 index 选取 outlier）
  - Decoding 替换/新增 kernel：Q×L^T 的 chunk-level attention approximation（融合 GEMM+Softmax+TopK）、Gather(A[I])×B 的低秩 key 重建 GEMM、CPU→GPU value 取回的异步 cudaMemcpy、Temporal cache index scan + merge

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源地址：https://github.com/ByteDance-Seed/ShadowKV 。Kernel 评估原理和全流程：

  **评估原理**：
  评估脚本测量每个 Transformer block 中各操作在给定 batch size × context length 下的延迟（ms），使用 CUDA event (cudaEventRecord) 精确计时。生成吞吐通过端到端运行固定数量的 decode steps，统计总 tokens 数 / 总耗时得到 tokens/s。

  **Kernel 输入到性能输出全过程**（以 decoding 为例）：

  输入状态：
  - GPU memory: A (low-rank key 左奇异向量), B (右奇异向量), L (landmarks), K_outlier, V_outlier
  - CPU pinned memory: V_CPU (全量 value cache)
  - 输入 Q ∈ R^{b×h_q×1×d}

  Step 1 — Attention Approximation Kernel：
  - 输入：Q [b, h_q, 1, d], L [b, h_kv, n_c, d]
  - 计算：P = Q @ L^T → [b, h_q, 1, n_c]; S = softmax(P/√d); 沿 h_q 聚合（GQA）→ [b, h_kv, n_c]
  - 输出：I = ArgTopK(S, k) → [b, h_kv, k]（selected chunk indices）
  - 性能：CUTLASS GEMM + custom softmax/topk fusion，延迟 ~0.56ms (48×64K)

  Step 2 — Key Reconstruction + Value Fetching (Overlapped)：
  - Stream 1 (GPU compute): A_selected = Gather(A, I) → [b, k*c, r]; K_sparse = A_selected @ B → [b, h_kv, k*c, d]
  - Stream 2 (PCIe H2D): V_sparse = cudaMemcpy(V_CPU[I], H2D) → [b, h_kv, k*c, d]
  - CUDA event 同步确保两者完成后继续
  - 性能：Recompute K ~1.25ms（overlapped），Fetch V ~1.84ms（overlapped 后 net latency 即 max(1.25, 1.84) 而非 sum）

  Step 3 — Attention Compute：
  - 输入：Q, K_combined = [K_outlier; RoPE(K_sparse); K_new], V_combined = [V_outlier; V_sparse; V_new]
  - 计算：FlashAttention (exact) on selected subset → O [b, h_q, 1, d]
  - 性能：~0.23ms (48×64K, selected chunk 占比 1.56%)

  Step 4 — Temporal Cache Mechanism：
  - 维护 prev_indices ring buffer，index_scan 检测 I 与 prev_indices 的交集
  - 跳过命中的 chunk，仅对未命中 chunk 执行 Step 2
  - Hit rate ~60%，有效减少 60% 的重建和取回开销

  SVD Overhead 分析（pre-filling）：64K→128K→256K→512K 上下文下，SVD 延迟分别为 17.19/26.62/50.56/108.38ms，占总 pre-filling 时间 6.65%/3.25%/1.75%/0.97%，随序列长度增加占比递减（因为 attention 计算为 O(n²) 而 SVD 为 O(n×d×r)）。

## Trainable_Dynamic_Mask_Sparse_Attention

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现一个专用 CUDA kernel（Flash Dynamic Mask Attention），融合 FlashAttention 风格的 tiling 与 DMA 的可训练稀疏性。核心机制：(1) **Block-level mask skip**：outer loop 中加载 K/V block 前先加载对应 mask block M_j，调用 Judge(M_j) 判断该 block 是否全零——若 active=0，则 advance stream pointers 跳过该 K/V block，直接进入下一 block，避免矩阵乘法和内存访问。(2) **Forward pass**（Algorithm 1）：Q/K/V/M 分块加载到 SRAM，对 K block 未被屏蔽的位置计算 S_ij = Q_i K_j^T / sqrt(d_h) + M_j，使用 online softmax（m/l/O 递推）保证数值稳定性，只在不全零的 block 上执行计算。(3) **Backward pass**（Algorithm 2）：与 Forward 共享统一 skip logic，只在必要时 fetch K/V tiles。backward 中 dM=dS，kernel 只需局部重算 S 而无需额外存储中间 mask 梯度张量。梯度链包含 fused bias gradients。整个 pipeline 完全可微，支持端到端训练。使用 shared memory aliasing、pipelined prefetching、coalesced memory accesses 优化带宽和 occupancy。实验比较 MHA (FlashAttention)、SWA、MLA (FlashMLA)、NSA 的 forward/backward/decoding kernel 性能。

- 后端平台是什么，配置是什么。
  NVIDIA A100-SXM4-80GB GPU。Benchmark 配置：32 heads、8 KV heads、d_h=128、bf16 精度。各变体统一参数对比，forward pass 在 token 长度 8192/16384/32768 下，decode phase 在 key 长度 65536/131072/262144/524288 下，backward pass 在 8192/16384/32768 下测试。3 次 warmup + 1000 次 run 取平均。

- 评估性能的软件/脚本是什么。修改了什么。
  使用 PyTorch + CUDA 自定义 kernel。相对于标准 FlashAttention 的修改：增加了 block-level mask skip 逻辑——在 K/V block 加载前检查 mask block，跳过全 zero block；增加了 mask 和 bias 的 batch/head/query broadcasting 支持；forward/backward 共享 skip logic；backward 中融合 bias 梯度和 mask 梯度（dM=dS，直接从 dS 推导而不额外存储）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/flash-algo/flash-sparse-attention。kernel 评估原理基于 tiled FlashAttention 架构。**输入到输出全过程**：
  (1) HBM→SRAM：Q/K/V/M 矩阵分块（T_r×T_c blocks，block size B）从 HBM 加载到 SRAM。每次 outer loop 加载一个 M block 到 SRAM。
  (2) Judge 判断：计算 active = Judge(M_j)。若 M block 所有元素为 −∞（全 zero），active=0，则 advance stream pointers 跳过该 K_j/V_j block，不执行任何 M×M（matrix multiply）操作。
  (3) 非跳过 block 计算：加载 K_j/V_j 到 SRAM，遍历所有 Q blocks。计算 S_ij = Q_i K_j^T × d_h^{−0.5} + M_j，使用 online softmax 递推公式合并到累计输出 O_i（m_new = max(m, rowmax(S_ij))，l_new = exp(m−m_new)·l + exp(S_ij−m_new)·rowsum）。
  (4) SRAM→HBM：将更新后的 O_i、l_i、m_i 写回 HBM。最终输出 O 为所有非跳过 block 的加权累积结果。有效复杂度 O(n·w·d_h)，内存 O(n·d_h)（无需物化完整 attention matrix）。
  Forward 速度提升：相对 MHA 在 8192/16384/32768 token 长度分别约 26.1×/10.2×/21.5×。Decode 提速：在 65536/131072/262144 key 长度分别约 49.6×/92.7×/171.1×。Backward 提速：在 8192/16384/32768 分别约 2.5×/4.4×/7.9×。

## XAttention: Block Sparse Attention with Antidiagonal Scoring

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  基于 FlashInfer 框架实现 block-sparse attention 的 GPU kernel，核心包含三部分 kernel 级实现：(1) **Strided Antidiagonal Scoring Kernel**：对每个 B×B 大小的 attention block，沿反对角线以步长 S 重排 Q 和 K 后计算近似注意力分数。Q 的 reshape 从 [B, d] 按 stride S 沿反对角线交错读取为 [S, B//S, d]；K 从全局内存按 stride S 分步读取。计算 Q_reshaped @ K_reshaped^T 得到 [S, S] 近似注意力矩阵，其反对角线和作为 block 重要性得分；(2) **Block Selection Kernel**：对 softmax 归一化后的反对角线分数执行 find_blocks——选择累计概率超过阈值 τ 的最小 block 子集。基于 cumulative probability threshold 实现动态稀疏度（Top-K 和 Top-Ratio 无法处理变化序列长度）；(3) **Block-Sparse Attention Kernel**：仅对选中的 block 子集执行完整的 FlashAttention 风格精确注意力计算，kernel 级别的 block tiling 跳过未选中区域。

  实验比较：(a) Prefill attention speedup vs FlashAttention (FlashInfer 实现)、MInference、FlexPrefill——256k 上下文下最高 13.5× 加速（S=16, τ=0.9, 密度 7.32%）；(b) Pattern selection 时间对比——XAttention 的 antidiagonal pattern selection 比 MInference 的 vertical-slash index search 快 24.9×，比 FlexPrefill 快 5.9×；(c) Attention time breakdown（Figure 5）：XAttention 将 pattern selection + sparse attention 的总开销控制在最低水平；(d) 消融研究：对比 antidiagonal vs random vs diagonal 模式的密度和准确率。

- 后端平台是什么，配置是什么。
  NVIDIA GPU（DGX 服务器）。基于 FlashInfer（https://flashinfer.ai/）注意力 kernel 库实现，使用其 FlashAttention 实现作为 dense baseline。精度为 BF16/FP16。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 FlashInfer 框架的 attention kernel 进行修改。核心修改：
  1. **Antidiagonal Scoring 实现**：在标准 FlashAttention prefill kernel 前插入轻量级 antidiagonal score 计算——对 Q 按 stride S 沿反对角线取子序列（Q[i::S,:] for i=S-1..0），对 K 按 stride S 正向取子序列（K[i::S,:] for i=0..S-1），计算 Q_reshaped @ K_reshaped^T / sqrt(d_h) / S 作为近似注意力分数。该步骤计算量仅为完整注意力的 1/S²。
  2. **Block Selection 实现**：基于反对角线 softmax 概率累积和选择 block，实现 greedy cumulative threshold 算法——按反对角线得分降序排列 block，从高到低累积直到超过 τ。
  3. **Sparse Attention 计算**：将选中的 block indices 传入 FlashInfer 的 block-sparse attention kernel，仅计算 mask 为 1 的 (query_block, key_block) 对。
  4. **Dynamic Threshold Prediction**：离线使用动态规划为每个 head 预测最优 τ 值，通过逐步降低 10% 的搜索策略（M=1000 steps）探索最优配置。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://github.com/mit-han-lab/x-attention

  **评估原理**：使用 CUDA Event 测量 attention 模块的 prefill kernel 执行时间（含 pattern selection + sparse attention 两部分），与 FlashInfer 的 FlashAttention 实现对比。Density 定义为选中 block 占总 block 数的比例（Table 5：128k 时 S=4→6.20%, S=8→6.89%, S=16→7.32%）。

  **Kernel 输入**：Q, K, V ∈ R^{L×d}（prefill 阶段完整序列），block size B，stride S，threshold τ。Dynamic threshold prediction 模式下每个 head 独立 τ_h。

  **Kernel 执行流程**（以 prefill 128k tokens, B=64, S=16 为例）：

  ```
  // N_B = 128k / 64 = 2048 blocks

  // === Phase 1: Antidiagonal Scoring (轻量级) ===
  for each block b in 0..N_B-1 (parallel over blocks):
      // Q reshape: [64, d] -> [16, 4, d]
      Q_slice = Q[b*64:(b+1)*64, :]
      Q_reshaped = []
      for i = 15 down to 0:
          Q_reshaped.append(Q_slice[i::16, :])  // 取反对角线元素

      // K reshape: [L, d] -> [16, L//16, d]
      K_reshaped = []
      for i = 0 to 15:
          K_reshaped.append(K[i::16, :])

      // 近似注意力: [16, 4, d] @ [16, L//16, d]^T -> [16, 16, L//16]
      // 简化为 per-block 的反对角线得分
      A_approx = Softmax(Q_reshaped @ K_reshaped^T / sqrt(d_h) / 16)
      score[b] = sum of antidiagonal values in A_approx

  // === Phase 2: Block Selection ===
  sorted_blocks = argsort(scores, descending=True)
  cumsum = 0
  selected_blocks = []
  for b in sorted_blocks:
      cumsum += scores[b]
      selected_blocks.append(b)
      if cumsum >= τ: break
  // 例如 τ=0.9, density ≈ 7%, 选中 ~143/2048 blocks

  // === Phase 3: Block-Sparse Attention ===
  M = zeros(N_B, N_B)  // 2048 x 2048 block mask
  M[:, selected_blocks] = 1  // 每行 query block 只关注选中的 key blocks
  // 实际 kernel 实现中直接传入 selected_blocks 索引列表

  for each query_block in 0..N_B-1 (grid-level parallel):
      load Q_blk [64, d] into SRAM
      for each key_block in selected_blocks:
          if key_block > query_block: continue  // causal mask
          load K_blk [64, d], V_blk [64, d] into SRAM
          S = Q_blk @ K_blk^T / sqrt(d_h)  // [64, 64]
          P = online_softmax(S)
          O_blk += P @ V_blk
      write O_blk to HBM

  // 总计算量: N_B * |selected| * B² * d
  //           = 2048 * 143 * 64² * 128 ≈ 1.5 × 10^11 FLOPs
  // vs dense: 2048 * 2048 * 64² * 128 ≈ 2.2 × 10^12 FLOPs
  // 加速比 ≈ 14.5×（接近实测 13.5×）
  ```

  **Pattern Selection 效率**：
  - XAttention antidiagonal scoring: O(N_B × S × d × B/S) = O(L × d) per block → 轻量级
  - MInference vertical-slash index search: O(L × k_v × k_s) → 需遍历垂直/斜线索引
  - 论文报告 pattern selection 快 24.9× vs MInference, 5.9× vs FlexPrefill

  **稀疏度随序列长度变化**（Table 5, S=8）：
  - 4k: density 52.16%（短序列注意力较密集）
  - 32k: density 20.97%
  - 128k: density 6.89%（长序列注意力高度稀疏）

  **With/Without Dynamic Threshold**：
  - Fixed τ=0.9: S=8, 32k density 23.06%
  - Minimum τ (DP optimized, avg 0.8): S=8, 32k density 20.97%

## Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现 Tree Attention 的 distributed attention kernel，在每 GPU 上使用 Flash Attention 2 (Dao, 2023) 进行局部 attention 计算，通过 NCCL AllReduce 进行跨设备通信，利用拓扑感知的 collective communication 调度。核心实现：
  (1) **Per-GPU Flash Attention 2 kernel**：每个 GPU 使用 JAX 绑定的 Flash Attention 2 (`flash_attn_jax.flash._flash_mha_vjp`) 对其本地 K,V chunk 执行精确 attention 计算，返回局部输出 o_local 和 logsumexp lse_local。
  (2) **NCCL AllReduce 调度**：通过 `lax.pmax`（max reduction）和 `lax.psum`（sum reduction）调用 NCCL AllReduce。NCCL 自动检测网络拓扑——intra-node 使用 ring reduce（NVLink 高带宽），inter-node 使用 tree reduce（InfiniBand 低带宽）——Tree Attention 的 AllReduce 模式自然享受此优化。
  (3) **通信-计算解耦**：与 Ring Attention 的 P2P KV chunk 传输不同，Tree Attention 不移动 K,V chunks，仅传输标量 lse 和部分归约结果（分子/分母/Max三个 tensor），通信体积极小。
  (4) **JAX shard_map 实现**：使用 JAX 的 `shard_map` + `Mesh` + `NamedSharding` 在序列维度上分片 K,V，指定 `in_specs=(P(None, None, None, None), P(None, 'i', None, None), P(None, 'i', None, None))` 和 `out_specs=P(None, None, None)`。

  实验比较：
  (a) Latency benchmark：标准 16-head attention block (head dim=128)，varying sequence lengths (80K-5.12M) 和 GPU counts (8-128 H100 nodes)——Tree Attention vs Ring Attention 执行时间对比（Section 6.1）。
  (b) Peak memory：JAX memory profiler 在 2×RTX 4090 上测量单 attention block 的峰值内存（Section 6.2）。
  (c) Communication volume：理论分析 + 实证对比，AllReduce vs P2P ring 的数据传输量（Section 6.3）。
  (d) End-to-end throughput：Llama 3.1-8B on 8×H100 / 4×MI300X / 2×RTX 4090 的解码延迟（Section 6.4 + Appendix C.3）。

- 后端平台是什么，配置是什么。
  (1) DGX H100 集群：16 节点 × 8 H100 GPU，NVLink 4.0 (900 GBps) intra-node，8× InfiniBand NDR (400 Gbps per GPU) inter-node。
  (2) AMD MI300X 集群：4 GPU，AMD Infinity Fabric intra-node + RoCE inter-node。
  (3) NVIDIA RTX 4090：2 GPU，PCIe interconnect。
  所有 kernel 使用 BF16 精度。JAX + Flash Attention 2 JAX binding。

- 评估性能的软件/脚本是什么。修改了什么。
  基于 JAX 框架 + Flash Attention 2 (`flash_attn_jax`) + NCCL collective communication。核心修改：
  (1) 实现 `tree_flash_decode` 函数（Appendix D）：使用 `shard_map` 将 K,V 沿序列维度（轴 'i'）分片到各 GPU，每个 GPU 调用 `_flash_mha_vjp.fwd` (Flash Attention 2) 计算局部 attention → 通过 `lax.pmax` / `lax.psum`（NCCL AllReduce）合并全局结果。
  (2) 对比 baseline Ring Attention：同样使用 Flash Attention 2 per-GPU，但采用 P2P send/recv 在 GPU 间环形传递 K,V chunks。
  (3) 使用 JAX memory profiler 测量峰值内存，使用 wall-clock timing 测量延迟。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/Zyphra/tree_attention（JAX 实现）。Flash Attention 2 JAX binding：https://github.com/nshepperd/flash_attn_jax。

  **评估原理**：对比 Tree Attention（AllReduce-based）和 Ring Attention（P2P ring-based）在相同硬件上处理相同 attention workload 的 wall-clock 延迟。两者均使用 Flash Attention 2 per-GPU，差异仅在跨设备通信策略。

  **Kernel 输入到性能输出的全过程**（单 token decoding, p 个 GPU）：
  ```
  # === 数据分布 ===
  # K, V ∈ R^{N×d_h} 沿序列维度分片到 p 个 GPU
  # GPU_i 持有 K_i, V_i ∈ R^{t×d_h}, t = N/p
  # q ∈ R^{1×d_h} 广播到所有 GPU

  # === Step 1: 每 GPU 本地 Flash Attention 2 ===
  # 输入: q [1, d_h], K_i [t, d_h], V_i [t, d_h]
  # 内部流程 (Flash Attention 2):
  #   - Tiling: 将 K_i, V_i 按 B_r×d_h, B_c×d_h 分块加载到 SRAM
  #   - QK^T: S_block = q_block @ K_block^T / sqrt(d_h)  [B_r, B_c]
  #   - Online softmax: m_new = max(m, rowmax(S))
  #     l_new = exp(m_old - m_new)*l_old + rowsum(exp(S - m_new))
  #     o_new = exp(m_old - m_new)*o_old + exp(S - m_new) @ V_block
  #   - 输出: o_i [1, d_h], lse_i [1] (scalar logsumexp)
  o_i, lse_i = flash_attn2(q, K_i, V_i)

  # === Step 2: AllReduce(max) — tree reduction ===
  # NCCL 执行: intra-node ring reduce + inter-node tree reduce
  # lse_i 在 p 个 GPU 间归约 → m_global = max(lse_1, ..., lse_p)
  # 通信复杂度: O(log p) 步 (inter-node tree), 每步传输 1 个标量
  m_global = lax.pmax(lse_i, axis_name='i')

  # === Step 3: 本地数值稳定化 ===
  # n_i = o_i * exp(lse_i - m_global)  [1, d_h]
  # d_i = exp(lse_i - m_global)        [1]
  n_i = o_i * jnp.exp(lse_i - m_global)
  d_i = jnp.exp(lse_i - m_global)

  # === Step 4: AllReduce(sum) — tree reduction (×2) ===
  # 两个独立的 AllReduce 调用，NCCL 可能合并
  # n_global = Σ_i n_i  [1, d_h]
  # d_global = Σ_i d_i  [1]
  # 通信量: d_h + 1 个元素 (≈ 129 for d_h=128)
  n_global = lax.psum(n_i, axis_name='i')
  d_global = lax.psum(d_i, axis_name='i')

  # === Step 5: 归一化 ===
  z = n_global / d_global  # [1, d_h], 精确 attention 输出
  ```

  **Ring Attention 对比流程**（同样输入）：
  ```
  # Ring Attention: 在 p 个 GPU 上环形传递 K,V chunks
  # GPU_i 持有 K_i, V_i, 同时也接收来自 GPU_{i-1} 的 K_{i-1}, V_{i-1}
  for step in range(p):
      # 每个 GPU 计算 flash_attn2(q, K_current, V_current)
      o_i, lse_i = flash_attn2_and_accumulate(q, K_current, V_current, o_i, lse_i)
      # P2P send/recv: 将 K_current, V_current 发送到下一 GPU
      send(K_current, V_current) to GPU_{(i+1)%p}
      recv(K_current, V_current) from GPU_{(i-1)%p}
  # 通信量: p × 2btd 个元素 (传输所有 K,V chunks 各一次)
  # 关键瓶颈: intra-node NVLink 和 inter-node InfiniBand 的带宽差异
  #   Ring 的每一步都需等待最慢链路 → inter-node 带宽成为瓶颈
  ```

  **Tree Attention 为什么更快**：
  (1) **通信量更少**：AllReduce 传输 d_h×2 个元素（分子+分母） vs Ring 传输 t×d_h×2 个元素（K+V chunk），当 t >> 1 时差异巨大（例如 t=80K, d_h=128 → Tree 传输 ~256 elements, Ring 传输 ~20M elements）。
  (2) **拓扑感知**：AllReduce 的 tree reduction 利用 inter-node 低带宽连接的层次结构，Ring 的均匀 P2P 无法区分 intra-node 和 inter-node 带宽差异。
  (3) **解码场景下计算太快无法 overlap 通信**：单 token 解码时 per-GPU Flash Attention 仅需 O(10^{-5})s，而 P2P 传输 K,V chunk 需 O(10^{-3})s (intra-node) 到 O(10^{-2})s (inter-node)，Ring Attention 无法 overlap。Tree Attention 的 AllReduce 仅传输标量级数据，通信延迟远小于 K,V chunk 传输。
