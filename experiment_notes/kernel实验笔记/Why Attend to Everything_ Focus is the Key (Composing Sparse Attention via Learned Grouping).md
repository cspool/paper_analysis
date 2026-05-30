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
