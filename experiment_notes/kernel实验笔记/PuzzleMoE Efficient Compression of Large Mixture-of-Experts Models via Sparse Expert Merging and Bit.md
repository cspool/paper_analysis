## PuzzleMoE Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：PuzzleMoE 设计了一个自定义 CUDA GEMV kernel 用于 on-the-fly 解码 bit-packed 权重。核心逻辑：(1) Bit-packed 存储——利用 Bfloat16 中 3 个 underutilized exponent bits 嵌入 binary mask 和 sign bit，压缩后的 W_merged 直接以标准 Bfloat16 格式存储，无需额外 metadata 存储；(2) On-the-fly Decoding GEMV Kernel——每个 weight W[i,j] 在矩阵乘法使用前即时从 packed 格式解码，解码操作（bit shift + mask + exponent 恢复）在 kernel 的 data-loading path 上 piggyback，利用 warp-level scheduling 和 coalesced memory access 实现零额外延迟；(3) 消除 decoded matrix 的 materialization——不在内存中创建独立的解码后权重矩阵，避免额外内存分配和访存开销。
  - 实验比较：(1) 推理加速——Mixtral-8x7B 50% sparsity 下 1.28× speedup，Qwen3-MoE 50% sparsity 下 1.19× speedup（vs full model on same GPU count）；(2) 内存节省——Mixtral-8x7B 从需要 2×A100-80GB 降至 1×A100-80GB，Qwen3-MoE 从 2×A100-40GB 降至 1×A100-40GB；(3) 压缩时间——Mixtral-8x7B 仅 2 分钟，Deepseek-MoE（64 experts）仅 10 分钟；D2 需 55 分钟（因 SVD），NAEE 对 Deepseek-MoE 需 10^18 次 forward pass 不可行。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100-80GB (Mixtral-8x7B) 和 A100-40GB (Qwen3-MoE)。CUDA kernel 基于 Bfloat16 计算路径，kernel 融合 decoding + GEMV，prefill length=1024, decode length=512。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 CUDA GEMV kernel。修改内容：(1) 新增 bit-packed weight 的 on-the-fly decoding 逻辑（Algorithm 1: mask_bit 提取 → 零值判断 → sign_bit 提取 → exponent 恢复 → Bfloat16 重构）；(2) 将 decoding 逻辑嵌入 GEMV data-loading path，利用 warp 级并行和 coalesced memory access；(3) 标准 Bfloat16 格式兼容——packed 后的数据仍可被 PyTorch 作为 Bfloat16 tensor 加载，仅在内核执行时通过自定义 kernel 进行解码。
  - 推理评估：基于论文自研推理框架（含自定义 CUDA kernel），对比 full model 和压缩后模型的 latency 和 memory usage。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/Supercomputing-System-AI-Lab/PuzzleMoE
  - 评估原理：自定义 GEMV kernel 加载 bit-packed Bfloat16 tensor（输入 X ∈ R^{B×d}，packed W ∈ R^{d×h} 含嵌入式 mask 和 sign），在 CUDA thread block 中每个 warp 处理一行 d 维的输入向量。具体流程：
    1. **Kernel Launch**：每个 MoE expert FFN 层的 weight 矩阵以 packed Bfloat16 格式存储在 GPU global memory。输入 activation X 从上一层的 FP16/Bfloat16 tensor 传入。
    2. **Data Load**：每个 warp 从 global memory 加载 packed W[i,j]（16 bits）进入寄存器。W[i,j] 的 bit layout：[15: sign_packed][14:13: mask bits for expert 0/1][12:7: shifted exponent (5 bits)][6:0: mantissa (7 bits)]。
    3. **On-the-fly Decode**（见 Algorithm 1）：
       - mask_bit ← (W ≫ (13 - expert_pos)) & 1
       - 若 mask_bit = 0 → 该 weight 对当前 expert 无效，W_decoded = 0
       - 否则 sign_bit ← (W ≫ (15 - expert_pos)) & 1
       - exp ← (W & 0x0F80) + (112 ≪ 7) 恢复原始 exponent
       - W_decoded ← (sign_bit ≪ 15) | exp | (W & 0x007F) → Bfloat16
    4. **FMA Compute**：W_decoded 作为 Bfloat16 值直接参与 FMA (Fused Multiply-Add) 计算 Y[p] += X[p,k] × W_decoded[k,j]。
    5. **性能输出**：end-to-end inference latency（prefill + decode phases）、GPU memory usage（通过 nvidia-smi 或 PyTorch memory stats 测量）、speedup ratio（latency_compressed / latency_full）。
  - 关键技术点：Decoding 在 data-loading path 上与 warp-level memory access 高度融合——解码的 bit ops 远小于 global memory read 延迟，因此解码开销被访存延迟隐藏。Expert 选择通过 gate network 的标准 Top-K routing 完成，gate 计算不涉及 packed weight——gate weights 保持原始精度。
