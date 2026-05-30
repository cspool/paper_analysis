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
