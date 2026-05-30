## QTIP: Quantization with Trellises and Incoherence Processing

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：QTIP 设计了硬件高效的 **bitshift trellis 解码 kernel**，在 GPU 上将 TCQ 编码的权重实时解码用于矩阵乘法。核心设计：(1) **Bitshift Trellis 解码**——每个 group of V 权重仅依赖连续 L-bit 窗口，解码时仅需 kV-bit 位移（bitshift by kV）获取下一组权重，硬件原生支持且完全并行化；(2) **Compute-based 解码 kernel**——1MAD 码仅需 2 GPU 指令（MAD + vabsdiff4）、3INST 码仅需 3 GPU 指令（MAD + lop3 + add）、HYB 码摊销 2 指令（MAD hash + lop3 sign-flip + LUT lookup），均 ≤4 指令/权重；(3) **MMA Tile 映射**——T_x=T_y=16 使每个 trellis 序列精确对应一个 16×16 MMA tile（NVIDIA Tensor Core 基础 tile 尺寸），矩阵乘法直接利用硬件加速单元；(4) **Tail-biting** 使编码比特总数能被 32-bit 字长整除，避免读取浪费比特；(5) **Codebook Cache 优化**——HYB 码 codebook 仅 2KiB (2^9×2 FP16)，可放入 L1 cache 甚至多次复制（32×）以消除 bank conflicts。
  - 实验比较：(1) 端到端推理吞吐（Table 4）：RTX 6000 Ada 上 batch_size=1 decode，Llama 2 7B/70B 的 QTIP vs QuIP# vs AQLM vs FP16，QTIP 2-bit 达 188/23.5 tok/s vs QuIP# 186/22.2、AQLM 81.5/8.78；(2) 跨 GPU 解码速度（Table 17）：RTX 3090、RTX A6000 Ampere、RTX 6000 Ada 上的 2/3/4-bit tok/s；(3) 与峰值带宽对比：QTIP 解码达到 >80% 峰值显存带宽。

- 后端平台是什么，配置是什么。
  - NVIDIA RTX 6000 Ada (960GB/s 显存带宽, Ada Lovelace)、NVIDIA RTX 3090 (Ampere)、NVIDIA RTX A6000 Ampere。CUDA/PTX 实现。利用 16×16 MMA tile（Tensor Core）进行矩阵-向量乘法，decode 阶段 memory-bound。

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 CUDA kernel 实现 bitshift trellis 解码 + dequantization + GEMV 融合。修改内容：
    1. **Bitshift 解码逻辑**：从 packed bitstream 读取当前 L-bit 窗口 → 通过 compute-based code (1MAD/3INST/HYB) 即时生成伪随机高斯权重 → 执行 FP16 GEMV。下一组权重通过 kV-bit 位移获取，无需存储完整 trellis 结构。
    2. **1MAD Kernel**：x = (a*x + b) mod 2^32 → 求和四个 8-bit unsigned ints → scale/shift → 输出近似高斯。2 指令: MAD (mul-add) + vabsdiff4（求和 4 个 8-bit 整数）。
    3. **3INST Kernel**：x = (a*x + b) mod 2^32 → 取 bottom/top 16 bits 分别 XOR magic FP16 m 的尾数/指数/符号位 → m1 + m2 → 输出近似高斯。3 指令: MAD + lop3 (logic op 3-input) + FADD。
    4. **HYB Kernel**：x = x²+x mod 2^32 → 取 bits (14-Q+1):14 作为 LUT index → 查表得 2D 向量 → XOR bit 15 翻转第二分量符号。摊销 2 指令。LUT 2KiB 常驻 L1 cache。
    5. **Tail-biting 对齐**：通过 Algorithm 4 近似 tail-biting，使 kT 能被 32 整除，无浪费比特读取。
  - 评估方式：测量 batch_size=1 decode 的端到端吞吐量 (tokens/s)，比较各量化方法的推理速度。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/Cornell-RelaxML/qtip
  - Kernel 输入到性能输出全过程（以 Llama 2 7B 2-bit QTIP HYB decode 为例，RTX 6000 Ada）：
    1. **输入**：packed bitstream（每 L=16 bits 编码一组权重，tail-biting 对齐 32-bit word），HYB codebook C ∈ R^{2^9×2} (2KiB, 常驻 L1 cache), LCG 参数 a, b。
    2. **Bitstream 加载**：从 DRAM 读取 32-bit word 到寄存器，通过 tail-biting 结构每 kT = 2×16×16 = 512 bits 对齐一个 16×16 tile。
    3. **Bitshift Trellis 解码**：对 Tx×Ty=16×16 tile 内每个权重位置，通过 bitshift 操作从 bitstream 窗口提取 L=16 bits 状态字，并行处理——每个权重仅依赖 16-bit 连续窗口。
    4. **HYB Code 解码**（per weight, 摊销 2 instrs）：
       - x ← x²+x mod 2^32 (MAD: 1 inst hash)
       - idx ← (x >> 6) & 511 (bitmask, fused in lop3)
       - v ← C[idx] (L1 cache lookup, 2×FP16)
       - sign-flip v[1] via XOR bit 15 (lop3: 1 inst)
       - 输出 2 个 FP16 权重值
    5. **MMA 计算**：16×16 tile 的 FP16 权重 × FP16 激活向量 → Tensor Core MMA (matrix-vector multiply accumulate) → FP32 accumulator → FP16 output。
    6. **输出**：当前 token 的 hidden state，传入下一 Transformer 层。
    7. **性能结果**：Llama 2 7B 2-bit 188 tok/s (>3× FP16 55.9 tok/s)，70B 2-bit 23.5 tok/s (FP16 OOM)。QTIP 与 QuIP# 吞吐相当，但有效维度为 256（QuIP# 仅 8），量化质量更高而无额外推理开销。
  - 关键优化：compute-based codes 消除了 VQ 方法需要的大 LUT 存储（AQLM 1MiB codebook 无法放入 L1 cache），HYB codebook 仅 2KiB → 32× 复制消除 bank conflicts。Bitshift trellis 的并行解码消除了 naive TCQ 的顺序依赖。
