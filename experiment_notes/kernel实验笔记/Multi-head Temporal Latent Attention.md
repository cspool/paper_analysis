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
