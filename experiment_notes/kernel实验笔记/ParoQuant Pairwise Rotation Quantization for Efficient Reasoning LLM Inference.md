## ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：为 scaled pairwise rotation transform 编写了**单一 fused CUDA kernel**，在推理时对激活张量应用逆变换 T^{-1}。kernel 在三个层次上完全并行化：(1) **token 维度**——沿 batch×seq_len 维度并行；(2) **channel group**——不同 CUDA block 处理不同的 channel group（group size=128）；(3) **rotation pair**——同一 group 内不同 CUDA thread 处理不同 Givens 旋转对。所有 Givens 旋转彼此独立（无数据依赖），因此同一 rotation 内的所有 pair 可**无同步**（synchronization-free）并发执行。由于 group size 小（128），激活张量可放入 on-chip shared memory，旋转参数（pair indices 和 angles）可放入寄存器，多个独立旋转（K=8）在一次内存加载后即可融合执行，无需多次 global memory 访问。
  - 实验比较：ParoQuant fused kernel vs **Fast Hadamard Transform** (Dao, 2024) 在不同 channel 维度上的 speedup (RTX A6000)。与 AWQ（无额外 transform kernel）、QTIP（Hadamard transform kernel）对比端到端 decode 吞吐（tokens/s）。在 RTX A6000、RTX 6000 Ada、RTX 4090 上的完整吞吐表（Table A4）。

- 后端平台是什么，配置是什么。
  - NVIDIA RTX A6000 (48GB, Ampere)、NVIDIA RTX 6000 Ada (48GB, Ada Lovelace)、NVIDIA RTX 4090 (24GB, Ada Lovelace)。推理测速使用 PyTorch 2.6.0 + torch.compile (max-autotune) + CUDA Graphs。W4A16 GEMM kernel 使用 AWQ 仓库的实现。

- 评估性能的软件/脚本是什么。修改了什么。
  - **Transformers library** (HuggingFace)：仅替换原始 FP16 线性层为量化线性层实现。AWQ、QTIP、ParoQuant 分别使用各自的官方开源仓库提供的量化层实现。
  - **Fast Hadamard Transform** (https://github.com/Dao-AILab/fast-hadamard-transform)：作为 Hadamard-based 方法的 transform kernel baseline。
  - ParoQuant 修改/新增：实现了 fused CUDA kernel 将 channel-wise scaling 逆变换和 K 个 independent rotations 融合为单次 kernel 调用。Kernel 采用三步并行策略：(1) token-level grid stride loop、(2) group-level CUDA block 分配、(3) pair-level CUDA thread 处理。旋转参数（2K 个 float per group × group 内 pair 数）预加载到寄存器，激活 tile 加载到 shared memory，8 个旋转在一个 kernel 内融合完成。
  - 评估方式：decode throughput (tokens/s)，batch size=1，测量端到端 token 生成速度。各方法使用统一的 Transformers 推理框架，仅修改量化层的权重变换和反量化代码。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：GitHub https://github.com/z-lab/paroquant（MIT License），PyPI `pip install paroquant[vllm]`。W4A16 GEMM kernel 复用 AWQ 开源仓库。
  - Kernel 输入到性能输出的全过程：
    1. **输入**：INT4 量化权重 W_q ∈ ℤ^{C_in/2 × C_out}（packed，每 2 个 4-bit 值打包为 1 byte）+ 量化参数 (s ∈ R^{C_out}, z ∈ ℤ^{C_out})；FP16 激活 X ∈ R^{T×C_in}（T=tokens）；旋转参数：每组 128 channels 存储 K=8 个 independent rotation 的 pair indices (2×64 个 uint8 per rotation) 和 angles (64 个 float per rotation)。
    2. **Kernel Launch**：对每个 linear 层，先调用 ParoQuant transform kernel 对激活 X 应用逆变换 T^{-1}，再调用 AWQ W4A16 GEMM kernel 执行 INT4 矩阵乘法。
    3. **Transform Kernel (Global→Shared Memory)**：按 token 和 group 分配 grid/blocks。每个 thread block 负责一个 (token, group) pair，将对应激活片 X[token, group_start:group_start+128]（128 × FP16 = 256 bytes）从 global memory load 到 shared memory。旋转参数（pair indices + angles）预取到寄存器。
    4. **Transform Kernel (Givens 旋转计算)**：按 pair 分配 threads。对每个 rotation r=1..K：
       - 读取 pair (i, j)、angle θ
       - `x_i' = cosθ * x_i - sinθ * x_j`
       - `x_j' = sinθ * x_i + cosθ * x_j`
       - 所有 pairs 并发执行（无依赖，无同步）。先做 8 次旋转，最后乘以 1/α 完成逆缩放。
    5. **Transform Kernel (Shared→Global)**：变换后的激活 tile 写回 global memory，作为后续 GEMM kernel 的输入。
    6. **GEMM Kernel**：AWQ 的 W4A16 GEMM kernel 读取 packed INT4 权重 + FP16 激活 → dequantize → FP16 matmul → FP16 output，写回 global memory。
    7. **性能测量**：CUDA Events 计时，包含 transform kernel + GEMM kernel 的端到端时间。计算 throughput = num_tokens / total_time (tokens/s)。
  - 关键优势：(1) Group 内 shared memory 常驻——128 个 FP16 仅 256 bytes，远小于 typical shared memory (48-100KB)，8 次旋转全部在 shared memory 上完成；(2) 无同步开销——同一 rotation 内 pairs 互不重叠，threads 间无数据竞争；(3) 随 channel 维度扩大，相比 Hadamard transform（有全局依赖）加速比递增（Figure 4）。
  - 结果：ParoQuant decode 吞吐仅比 AWQ 慢约 10%（如 Qwen3-4B: 160 vs 176 tokens/s），比 QTIP 快 15%-30%（如 Qwen3-4B: 160 vs 117 tokens/s）。
