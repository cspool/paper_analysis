## YOCO (You Only Cache Once): Gated Retention Triton Kernel

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现了 gated retention 的 Triton kernel，支持三种计算范式：(1) **Parallel** 模式用于训练（充分利用 GPU 并行）；(2) **Recurrent** 模式用于自回归生成（O(1) 常量 KV 内存）；(3) **Chunkwise Recurrent** 模式用于 prefill（结合并行和 recurrent 优势，chunk 内并行 + chunk 间 recurrent，降低 FLOPs 和迭代次数）。prefill 阶段使用 chunkwise（chunk_size=256），生成阶段切换到 recurrent。基于 FLA (Flash-Linear-Attention) 库实现。同时 baseline Transformer 使用了 Flash-Decoding 和 kernel fusion 进行公平比较。实验比较了 YOCO_gRet 与优化 Transformer（GQA + Flash-Decoding + kernel fusion）在 H100-80GB 上的推理性能（GPU memory, prefill latency, throughput），序列长度从 32K 到 1M。

- 后端平台是什么，配置是什么。
  NVIDIA H100-80GB GPU。Triton kernel 基于 FLA 库（https://github.com/sustcsonglin/flash-linear-attention）实现。Baseline Transformer 使用 Flash-Decoding 做优化 attention kernel。

- 评估性能的软件/脚本是什么。修改了什么。
  评估方式：profiling 测量 GPU memory breakdown（model weights + intermediate activation + KV cache）、prefill latency（编码输入 prompt 的时间）、throughput（tokens/s，包含 prefill + generation）。
  修改内容（Gated Retention Triton kernel 设计）：
  1. **Chunkwise Recurrent Kernel（prefill 用）**：
     - 将序列分为 chunk_size=256 的块
     - Inner-Chunk 部分用 parallel 计算（chunk 内 QK^T ⊙ D + V，利用 Tensor Core）
     - Cross-Chunk 部分用 recurrent state R_i 传递跨 chunk 信息
     - 输出 = (Q_{[i]} K_{[i]}^T ⊙ D_{[i]}) V_{[i]} + (Q_{[i]} R_{i-1}) ⊙ β_{[i]}
     - 比 fully parallel 节省 FLOPs，比 fully recurrent 减少迭代轮次
  2. **Recurrent Kernel（decode 用）**：
     - S_n = γ_n · S_{n-1} + K_n^T V_n（state 更新，O(d²) per step）
     - O_n = Q_n · S_n（vector-matrix multiply）
     - 仅维护 single state matrix S ∈ R^{d×d}，不存储 per-token KV cache
  3. **数据依赖门控优化**：γ 使用 head-wise decay（而非 element-wise），使计算可充分利用 NVIDIA Tensor Core

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源：https://aka.ms/YOCO。Triton kernel 基于 FLA 库 (https://github.com/sustcsonglin/flash-linear-attention)。

  **Kernel 执行全过程（以 YOCO_gRet 3B, H100-80GB, 512K 输入 + 1024 token 生成 为例）**：

  **阶段 1: Prefill（Chunkwise Recurrent Kernel）**
  - 输入：embedding X ∈ R^{512K×3072}，权重 W_Q, W_K, W_V, W_γ ∈ R^{3072×3072}
  - Step 1a: 投影 Q = XW_Q ⊙ Θ, K = XW_K ⊙ Θ̄, V = XW_V（利用 Tensor Core GEMM）
  - Step 1b: 计算 gate γ = sigmoid(XW_γ)^{1/τ}（head-wise, 24 heads）
  - Step 1c: 分 chunk（256 tokens/chunk, 2000 chunks），对每个 chunk：
    - Inner-Chunk: (Q_c K_c^T ⊙ D_c) V_c（parallel, Tensor Core matmul）
    - Cross-Chunk: (Q_c R_{c-1}) ⊙ β_c（state R 传递，O(d²) 计算）
    - 更新 state: R_c = K_c^T (V_c ⊙ value_decay) + chunk_decay · R_{c-1}
  - Step 1d: GroupNorm + swish gate 输出
  - 输出：Self-Decoder 最终 activation M = X^{L/2} ∈ R^{512K×3072}
  - 关键：此时 prefill 可提前退出，无需进入 Cross-Decoder

  **阶段 2: KV Cache 生成（单次）**
  - 输入：M（Self-Decoder 最终输出）
  - K̂ = LN(M) W_K, V̂ = LN(M) W_V
  - 存储：K̂, V̂ ∈ R^{512K×3072}（单层全局 KV cache，约 512K×3072×2×2bytes = 6.3GB, 3B 模型）

  **阶段 3: Decode（Recurrent Kernel for Self-Decoder, Standard Attention for Cross-Decoder）**
  - 每个新 token：
    - Self-Decoder(recurrent): S_n = γ_n · S_{n-1} + K_n^T V_n → O_n = Q_n · S_n（极小内存，仅维护 S）
    - Cross-Decoder: Q̂ = XW_Q → Attention(Q̂, K̂, V̂)（标准 Flash-Decoding kernel, 复用全局 cache）

  **性能输出（H100-80GB, 3B model）：**
  - GPU Memory: 1M context 仅 12.4GB（Transformer 需 9.4× more）；32K context 节省 ~2×
  - KV Cache per token: 128K tokens 仅需 1GB KV cache（65B model 时 Transformer 仅支持 1.6K tokens）
  - Prefill Latency: 512K 从 180s（Transformer）降至 <6s（71.8× 加速 for 1M, 2.87× for 32K）
  - Throughput: 512K queries 43.1 token/s vs Transformer 4.5 token/s（9.6× 加速）
  - 加速来源：(a) prefill 仅需 L/2 层 + 高效 attention；(b) KV cache 内存节省允许更大 batch size
