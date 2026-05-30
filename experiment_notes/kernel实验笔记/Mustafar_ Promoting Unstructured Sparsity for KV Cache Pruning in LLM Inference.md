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
