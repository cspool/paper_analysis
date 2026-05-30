## Optimal Brain Restoration for Joint Quantization and Sparsification of LLMs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：基于 **CUTLASS** 库实现 **INT4 2:4 semi-structured sparse GEMM kernel**，利用 NVIDIA Ampere/Hopper 架构的硬件原生支持（如 Sparse Tensor Cores）。对 joint OBR 压缩后的 LLM（W4A4KV4 + 50% sparsity），权重 W 使用 INT4 量化 + 2:4 结构化稀疏，激活 X 使用 INT4 量化，执行 batched INT4 sparse GEMM。
  - 实验比较：INT4 2:4 Sparse GEMM vs. INT4 Dense GEMM vs. FP16 Dense GEMM。在单张 NVIDIA A100-SXM4-80GB 上评测不同序列长度（128/512/1024/2048/4096）下的 **latency (ms)**、**theoretical FLOPs (GFLOPs)** 和 **TOPS**。权重矩阵尺寸为 W ∈ ℝ^{4096×4096}，激活为 X ∈ ℝ^{32×seq_len×4096}，模拟典型 LLM 推理场景。

- 后端平台是什么，配置是什么。
  - NVIDIA A100-SXM4-80GB GPU（Ampere 架构），80GB HBM2e，支持第三代 Tensor Cores 和 2:4 structured sparsity。

- 评估性能的软件/脚本是什么。修改了什么。
  - **CUTLASS** (https://github.com/NVIDIA/cutlass)：NVIDIA 开源 CUDA C++ 模板库，用于实现高性能 GEMM kernel。论文使用 CUTLASS 实现 INT4 sparse GEMM，将 2:4 结构化稀疏与 INT4 量化结合到单个 kernel 中。修改/新增了支持 INT4 数据类型 + 2:4 sparse 模式的 GEMM kernel。
  - 评估方式：测量给定 batch=32 下的 wall-clock latency；计算理论 FLOPs（2×M×N×K - 2×M×N×K×sparsity）；计算 TOPS (Tera Operations Per Second) = FLOPs / latency。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：OBR 算法代码 https://github.com/csguoh/OBR；CUTLASS kernel 库 https://github.com/NVIDIA/cutlass。
  - Kernel 输入到性能输出的全过程：
    1. **输入**：经 OBR 压缩后的 INT4 量化权重 Ŵ ∈ ℤ^{M×K}（存储为 packed INT4，每 2 个 4-bit 值打包为 1 byte），2:4 稀疏 mask M（每 4 个连续元素中恰好 2 个非零）；INT4 量化激活 X̂ ∈ ℤ^{K×N}；对应的量化 scale/zero-point。
    2. **Kernel 编译**：使用 CUTLASS 模板定义 `Gemm<Int4Sparse>`，指定 warp tile size、threadblock shape、MMA (Matrix Multiply-Accumulate) 指令（利用 Sparse Tensor Cores 的 `mma.sp.sync` 指令，每次取 4 个值中自动跳过 2 个零）。
    3. **Global → Shared Memory**：将 packed INT4 Ŵ 和 sparse metadata 从 global memory load 到 shared memory；将 FP16/INT4 X̂ load 到 shared memory。利用 2:4 结构化稀疏将权重访存量减半。
    4. **MMA 计算 (Tensor Core)**：warp-level Tensor Core 指令执行 `D = A * B + C`，其中 A (激活, INT4 → FP16 反量化), B (权重, INT4 2:4 sparse, 利用硬件跳过零值), C/D (累加器, FP32)。2:4 稀疏使有效计算减少 2×，内存带宽需求降低 2×。
    5. **输出**：FP32 累加器结果 → 写回 global memory 作为输出矩阵 Y ∈ ℝ^{M×N}。
    6. **性能测量**：用 CUDA Events (cudaEventRecord) 计时，排除 kernel launch overhead。测量纯 GEMM 计算时间，计算 TOPS = (2 × M × N × K × (1 - sparsity)) / latency。
  - 结果：在 seq_len=4096 时，INT4 2:4 Sparse GEMM 比 FP16 Dense GEMM 快 **5.9×**，比 INT4 Dense GEMM 快 **1.4×**；理论 FLOPs 比 INT4 Dense 减半（因 50% 稀疏）；GPU 资源饱和（seq_len > 2048）时 TOPS 更高。

 (Learn at Test Time): RNNs with Expressive Hidden States

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：为 TTT-Linear 和 TTT-MLP 编写**自定义 GPU 推理 kernel**，分别针对 forward（prefill，可并行化）和 generate（decode，序列化）两种模式。训练 kernel 未实现（因 TPU 上使用 JAX，不兼容 GPU kernel）。
    - **Forward (prefill) kernel**：使用 dual form，将内循环梯度计算转化为 matmul 操作以最大化 TensorCore 利用率。dual form 等价于 primal form 的输出，但通过 `W_b = W_0 - 2η(W_0X̂ - Y)X̂^T` 和 `Z = W_0X̄ - 2η(W_0X̂ - Y)mask(X̂^TX̄)` 将所有操作转为 matmul，训练速度比 primal form 快 **5× 以上**（TPU 上）。
    - **Generate (decode) kernel**：使用 primal form，因为逐个 token 生成本质上是序列化的，每次只需对单个 token 计算梯度 `G_t = ∇ℓ(W_{t-1}; x_t)` 并更新 `W_t = W_{t-1} - ηG_t`，无需 dual form 的批处理优势。
  - 实验比较：在 NVIDIA A100 80G PCIe 上评测 1.3B 模型的 forward latency（prompt processing）和 generate latency（token 解码），对比 Transformer（vLLM serving）、Mamba、TTT-Linear、TTT-MLP。随上下文长度从 1k 到 32k 增加，Transformer latency 线性增长，TTT 和 Mamba 基本恒定。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100 80G HBM，PCIe 连接
  - 训练平台：TPU v5e-256 pod（JAX 实现，与 GPU kernel 路径分开）

- 评估性能的软件/脚本是什么。修改了什么。
  - Transformer baseline：使用 **vLLM**（state-of-the-art LLM serving 系统），而非 HuggingFace Transformer，确保 baseline 具有竞争力。
  - Mamba baseline：使用作者公开的 PyTorch+Triton+CUDA 代码。
  - TTT 方法：自写 GPU inference kernel，**未实现训练 kernel**（训练在 TPU 上用 JAX 完成）。修改/新增了 forward (prefill) 的 dual form kernel 和 generate (decode) 的 primal form kernel。
  - 评估方式：测量单次 forward（给定 prompt 长度）和单 token generate 的 wall-clock latency。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：JAX 代码 https://github.com/test-time-training/ttt-lm-jax，PyTorch 代码 https://github.com/test-time-training/ttt-lm-pytorch。GPU inference kernel 位于 PyTorch 仓库中。
  - 评估原理与 kernel 执行流程：
    1. **输入**：输入序列 X ∈ R^{d×T}（T 为 prompt 长度，forward 模式）或单个 token x_t ∈ R^d（generate 模式）
    2. **Forward (prefill) kernel 流程**：
       - 按 mini-batch size b=16 将 T 个 token 分块
       - 对每个 mini-batch，执行 dual form：
         a. 投影：X̂ = θ_K @ X_block, Y = θ_V @ X_block, X̄ = θ_Q @ X_block（matmul, 利用 TensorCore）
         b. 计算 W_end：W_b = W_0 - 2η(W_0 @ X̂ - Y) @ X̂^T（matmul）
         c. 计算输出：Z = W_0 @ X̄ - 2η * (W_0 @ X̂ - Y) * mask(X̂^T @ X̄)（matmul + 上三角 mask）
       - 将 W_b 传入下一个 mini-batch 作为新的 W_0
    3. **Generate (decode) kernel 流程**：
       - 对单个新 token x_t，使用 primal form：
         a. 投影：x̂ = θ_K x_t, y = θ_V x_t, x̄ = θ_Q x_t（向量操作）
         b. 梯度：G = 2(W_current x̂ - y) x̂^T（外积）
         c. 更新：W_new = W_current - η(x_t) · G
         d. 输出：z = W_new @ x̄（矩阵-向量乘）
    4. **输出**：output tokens Z ∈ R^{d×T}（forward）或 z_t ∈ R^d（generate），性能指标为 wall-clock latency（ms）
