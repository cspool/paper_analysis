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
