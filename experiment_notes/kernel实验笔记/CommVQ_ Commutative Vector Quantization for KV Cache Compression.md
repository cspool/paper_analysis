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
