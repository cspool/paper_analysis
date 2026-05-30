## KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - KIVI 的 **System Support** 部分提供了 GPU 上的硬件友好实现，包含两个关键 kernel：
    1. **Q_MatMul（Fused Dequantization + Matrix Multiplication）**：使用 CUDA 实现，将 KV cache 的反量化过程与矩阵乘法在 tiling 级别融合，避免将反量化后的全精度数据写回全局内存。在计算 attention score 和 attention output 时使用，减少内存访问开销。
    2. **Group-wise Quantization Kernel**：使用 Triton 实现，执行 group-wise round-to-nearest 量化（per-channel 用于 key，per-token 用于 value）。支持 streaming 场景下将新到达的 KV tensor 动态量化并追加到已有 quantized cache。
    3. **Tiled Matrix Multiplication**：将 grouped quantized 部分和 residual FP16 部分的矩阵乘法分块执行后 Concat。
  - 实验比较：
    - KIVI（residual length 32/128）vs FP16 baseline 在 Llama-2-7B 上的峰值内存和吞吐量（ShareGPT workload）
    - KIVI 可使 batch size 增大 4×，吞吐量提升 2.35× ∼ 3.47×

- 后端平台是什么，配置是什么。
  - GPU：单张 NVIDIA A100 GPU（80GB）
  - 计算后端：CUDA（用于 fused dequantization + MatMul kernel）、Triton（用于 group-wise quantization kernel）

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **Hugging Face Transformers** 代码库实现 KIVI 算法
  - 使用 **LM-Eval** 框架评估准确率（CoQA, TruthfulQA, GSM8K）
  - 使用 **LongBench** 评估长上下文性能
  - 使用 **ShareGPT** 真实对话数据合成 workload，参考 **vLLM** 的方式评估吞吐量和内存
  - 修改内容：
    - 在 attention 层中插入 KIVI 的量化/反量化逻辑
    - 实现了 CUDA fused dequantization+MatMul kernel（Q_MatMul）
    - 实现了 Triton group-wise quantization kernel
    - 修改了 KV cache 的数据结构为 grouped quantized + residual FP16

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/jy-yuan/KIVI
  - **Kernel 执行流程详解**：

  **Q_MatMul（Fused Dequantization + MatMul）**：
  ```
  输入: t_Q ∈ R^{1×d} (query), Q(X_K_g) ∈ int2 (quantized grouped key)
  过程:
    1. 将 t_Q 按 tile 加载到 GPU SRAM
    2. 将对应的 Q(X_K_g) tile 加载到 SRAM
    3. 在 SRAM 中即时反量化 int2 → FP16：
       X_K_g' = Q(X_K_g) × s_K + z_K  // s_K是scaling factor, z_K是zero-point
    4. 直接在 SRAM 中计算 tile 的矩阵乘法: A_tile = t_Q_tile × X_K_g'_tile^T
    5. 输出 A_tile 写回全局内存
  输出: A_g (attention logits for grouped part)
  ```
  避免将 FP16 大小的 X_K_g' 整体写回全局内存，节省 HBM 带宽。

  **Group-wise Quantization Kernel**：
  ```
  输入: X ∈ R^{l×d} (float16), G=32 (group size), dim (quantization axis)
  过程 (dim=channel 时):
    1. 将 X 沿 channel 维度分成 d/G 个 group
    2. 每个 group 包含 G 个连续的 channel
    3. 对每个 group 计算: min, max → s_X = (max-min)/3, z_X = min
    4. 执行量化: Q(X) = round((X - z_X) / s_X)，clamp 到 [0, 3] (2bit)
    5. 将 Q(X), s_X, z_X 存储
  输出: Q(X) ∈ int2, 每 G 个元素共享一组 (s_X, z_X)
  ```

  **Tiled Attention 全流程**：
  ```
  KV cache = {Q(X_K_g): int2, X_K_r: FP16, Q(X_V_g): int2, X_V_r: FP16}
  
  1. Q_MatMul(t_Q, Q(X_K_g)) → A_g    // fused dequant+matmul, grouped部分
  2. t_Q × X_K_r^T → A_r               // 标准 matmul, residual FP16部分
  3. A = Concat([A_g, A_r])            // 拼接 attention logits
  4. A_g_sm = Softmax(A)[:-R], A_r_sm = Softmax(A)[-R:]
  5. Q_MatMul(A_g_sm, Q(X_V_g)) → t_O_g  // fused dequant+matmul
  6. A_r_sm × X_V_r → t_O_r              // 标准 matmul
  7. t_O = t_O_g + t_O_r
  ```
