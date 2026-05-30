## QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：QuIP# 设计了针对 E8P 编解码的高效 CUDA kernel，用于量化线性层的矩阵-向量乘法（decode_matvec_e8p_kernel），结合 Fast Walsh-Hadamard Transform (FWHT) 实现完整推理管线。Kernel 核心设计：
    - **E8P 解码 + MMA 融合**：在一个 warp 级 kernel 中完成 E8P 码字解码和 Tensor Core MMA 累加。(1) 从压缩权重 `weights_compressed` 读取 uint2 码字，每 uint2 包含 4 个 16-bit E8P 码字；(2) 用 `codebook_abs[256]` 查找源码书绝对值条目（4-bit 整数存储，128B L1 cache）；(3) 通过 XOR 操作和移位实现符号翻转解码（7+1 位 sign bits）；(4) 生成 FP16 权重后直接送入 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` PTX 指令做矩阵乘法累加。Kernel 使用 warp 级数据共享（`__shfl_sync`）和 bank conflict 避免（32× 复制 codebook 到 L1）。
    - **RHT（Fast Walsh-Hadamard Transform）**：O(n log n) 时间复杂度，仅涉及 ±1 加减法运算，无浮点乘法。在推理时对激活向量做两次 Hadamard 变换（前/后），用于撤销量化时的非相干处理。
    - **压缩格式**：权重以 2-bit E8P 码字打包存储——每个 16-bit 码字编码 8 个权重共 16 bits（2 bits/weight × 8），E8P 源码书仅 256×8 条目 = 1KiB，可完全放入 L1 cache（即使在 32× bank conflict 复制后）。
  - 实验比较：(a) NVIDIA RTX 4090 上 Llama 2 7B/13B/70B 和 Llama 1 30B 的 2-bit/4-bit 生成吞吐（tok/s）及峰值显存带宽利用率（2-bit 下 70B 达 56.84%）; (b) RTX 4090 上 QuIP# vs AQLM vs FP16 的 Llama 2 7B/70B 吞吐对比——QuIP# 2-bit 显著快于 FP16 和 AQLM；(c) A6000 上 QuIP# vs QuIP 吞吐对比——同 bitrate 下 QuIP# 约为 QuIP 的 2 倍。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA RTX 4090（Ada Lovelace 架构，1TB/s 峰值显存带宽，L1 cache 128KB/SM）；NVIDIA A6000（Ampere 架构）。
  - CUDA PTX：使用 mma.sync.aligned.m16n8k16 指令，针对 Ampere 及更新架构 Tensor Core（FP16 输入→FP32 累加）。
  - 软件栈：FlashAttention 库的 Llama 实现和 HuggingFace Llama 实现分别用于吞吐测试。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估脚本：基于 FlashAttention 库（Dao et al., 2022; 2023）的 Llama 模型实现进行生成吞吐测试；HuggingFace Llama 实现用于与 AQLM 的公平对比。
  - 修改内容：(1) 将标准 FP16 线性层替换为 QuIP# 量化线性层（包含两次 FWHT + E8P GEMV kernel）；(2) 自研 E8P decode_matvec CUDA kernel，编译为 quiptools_e8p_gemv.cu；(3) 权重预量化为压缩码字格式后以 uint2 数组加载。
  - 优化程度：论文描述为 "proof of concept" 实现，minimal kernel fusion in the RHT，仍有进一步融合优化空间（Hadamard transform 与后续操作融合）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/Cornell-RelaxML/quip-sharp 包含 `quiptools/quiptools_e8p_gemv.cu`；预量化模型：https://huggingface.co/relaxml
  - Kernel 输入到性能输出全过程（以 Llama 2 70B 2-bit QuIP# decode 为例，RTX 4090）：
    1. **输入**：压缩权重 `weights_compressed`（uint2 数组，每个 uint2 含 4 个 16-bit E8P 码字，平均 2 bits/weight）、`codebook_abs`（256 条 uint32_t 绝对值条目，常驻 L1 cache）、`input`（当前 token 的 hidden state，FP16 向量）。
    2. **线程组织**：Grid 按输出维度 N 分块（每 16 行一个 block），Warp 内 32 个线程处理 K 维度上连续的权重块。warpId = threadIdx.y 确定处理 K 维度哪个 chunk，laneId = threadIdx.x 确定 Warp 内线程 ID。
    3. **E8P 码字解码**（per weight group of 8 weights）：
       - 从 `weights_compressed` 读取 uint2 w_compr（a = w_compr.x, b = w_compr.y）
       - Parity 计算：s = b XOR (b>>4) XOR (b>>8) XOR (b>>16)，sb = s & 15 为每 8 个权重的奇偶性位
       - 对 4 个 E8P 码字（a 的 4 个字节）分别解码：
         a. `x = codebook_abs[(a >> shift) & 255]` — 查表得 8 个权重的绝对值（4-bit pack）
         b. `x = x XOR ((s & mask) * factor)` — 解码 7 位符号翻转
         c. `o = BASE_OFFSET | ((sb & mask) << shift2)` — ±1/4 偏移
         d. `w00..w03 = add_as_half2(mask_lop3(...))` — 生成 4 组 FP16 权重对
       - 使用 `lop3` 指令融合多个位操作，< 5 指令/权重
    4. **MMA 累加**：`__shfl_sync` 广播激活片段 → `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` Tensor Core MMA 指令，输入 FP16 权重（w00-w13）和 FP16 激活片段（x_in0, x_in1），FP32 累加到 z0-z3。
    5. **输出**：`atomicAdd` 将各 Warp 的 FP32 累加器写回 output 数组 → 经第二个 FWHT (S_U) → 最终 hidden state。
    6. **性能结果**：Llama 2 70B 2-bit 达 32.74 tok/s（56.84% peak mem BW），7B 2-bit 达 170.50 tok/s（29.60%），均远超 AQLM（2-7B 20.6 tok/s, 2-70B 8.27 tok/s）和 FP16（2-7B 33.1 tok/s）。
    7. **关键设计优势**：E8P codebook 仅 1KiB → 可放 L1 cache → 避免 AQLM 1MiB codebook 导致的 L1 cache miss；E8 格高 packing density → 量化误差低于 D4 格和 K-Means。
