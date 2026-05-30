## GPTVQ: The Blessing of Dimensionality for LLM Quantization

- **属于kernel调度/运行时计算的实现是什么？实验比较什么？**
  实现了 VQ 解码 kernel（CPU 端使用 TBL 指令，GPU 端使用 CUDA vector types），将 VQ 编码的索引解压缩为 native data type。CPU kernel：利用移动 CPU 的 hardware lookup table instruction (TBL)，将 6-bit index 映射到 8-bit signed integer（2D VQ 需 2 条 TBL 指令链接），解码后的整数用于矩阵-向量乘（SIMD 加速）。GPU kernel：使用 CUDA vector types（char4/uchar4 乃至 char128）快速加载/写回数据。

  实验比较（Table 6, Appendix B）：
  - CPU 端：Data Transfer 实验：对比 Uniform INT4、Uniform INT8、VQ 2D（3/2.75/2.25 bpv）的相对延迟和相对 footprint。Token Generation 实验：VQ 1D 3.125 bpv vs Uniform。
  - GPU 端（RTX 3080）：Data Transfer 实验：对比 Uniform INT4、Uniform INT8、FP16、VQ 2D（2.125/3.125 bpv）、VQ 4D（2.125 bpv）的相对延迟和 footprint。
  - 端到端推理（Table 1, Section 5.1）：Llama-v3-8B 在 Snapdragon X Elite 上，对比 llama.cpp INT4、自有引擎 INT4(g128)、自有引擎 VQ 2D(3.125 bpv)，测量 Model Footprint (GB) 和 Throughput (tok/s)。

- **后端平台是什么，配置是什么。**
  - CPU 端：Snapdragon X Elite 平台（mobile CPU），Windows OS，Clang 18.1 with Polly。利用 ARM TBL (Table Lookup) 指令支持 6-bit→8-bit 映射。
  - GPU 端：NVIDIA GeForce RTX 3080 GPU，CUDA。
  - 推理引擎：自研 C 语言实现（含 vector intrinsics、SIMD 扩展、polyhedral compiler capabilities），支持高度参数化 transformer 架构。

- **评估性能的软件/脚本是什么。修改了什么。**
  - 评估方式：自定义 VQ decoding kernel 集成到自研 LLM 推理引擎，测量 data transfer/decoding 延迟和端到端 token generation rate。
  - 修改内容（VQ 解码 kernel 设计）：
    1. **CPU VQ Decode Kernel**：
       - 6-bit indices 紧凑打包（packed tightly）存储，与 LUT（lookup tables）和量化 scale 按 block 组织以实现高效向量化
       - 每个 block 加载流程（Section 3.2）：DRAM → SoC cache → VQ decode kernel 使用 TBL 指令解码 → 输出 signed 8-bit int → 矩阵-向量乘
    2. **GPU VQ Decode Kernel**：
       - 使用 CUDA native vector types（char4/uchar4 + 自定义 char128 agglomerations）并行加载/写回
       - 支集 2D VQ 和 4D VQ 的解码
    3. **Packing 格式**：6-bit indices 紧凑打包（每个 weight 占 6 bits），LUT（64 entries × 8-bit），scale（FP16）；block size=8192（移动端实测配置）

- **开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。**
  论文声明 GPU kernel 代码 "will be made available in the future"（尚未开源）。推理引擎为 Qualcomm 内部实现。

  **CPU VQ Decode Kernel 执行全过程（Llama-v3-8B 2D VQ 3.125 bpv, Snapdragon X Elite）**：

  1. **输入**：packed 6-bit indices（每个 weight 占 6 bits），per-block 64-entry LUT（8-bit signed int per entry），per-block FP16 scale
  2. **数据加载**：DRAM 中读取 block tuple (indices + LUT + scale) → SoC CPU cache
  3. **Decode 阶段**：
     - 从 packed indices 解包 6-bit index（维度1和维度2各一个）
     - TBL 指令：dimension 1 index → LUT lookup → 8-bit signed int (v1)
     - TBL 指令：dimension 2 index → LUT lookup → 8-bit signed int (v2)
     - 合并：v_decoded = v1 + v2（2D VQ 合并两维结果）
     - 反量化：w_fp = scale × v_decoded
  4. **矩阵-向量乘**：SIMD 加速的 INT8/INT32 乘法累加
  5. **输出**：下一层 activation

  **性能结果**（Table 1, Llama-v3-8B, Snapdragon X Elite）：
  - llama.cpp INT4: Footprint 4.64GB, Throughput 17.95 tok/s
  - Ours INT4 g128: Footprint 4.33GB, Throughput 23.81 tok/s
  - Ours VQ 2D 3.125 bpv: Footprint 3.52GB (-19%), Throughput 26.15 tok/s (+10% vs Ours INT4)

  **CPU Data Transfer 结果**（Table 6, gate_proj 层 11008×4096）：
  - Uniform INT4: Rel. FP 1.00×, Rel. Lat 1.00×
  - VQ 2D 2.25 bpv: Rel. FP 0.56×, Rel. Lat 0.87×（延迟更低 + footprint 更小）

  **GPU Data Transfer 结果**（Table 6, RTX 3080）：
  - VQ 2D 2.125 bpv: Rel. FP 0.53×, Rel. Lat 1.03×（footprint 减半，延迟近似持平 FP16）
  - VQ 4D 2.125 bpv: Rel. FP 0.53×, Rel. Lat 0.71×（footprint 减半 + 延迟降低 29%）
