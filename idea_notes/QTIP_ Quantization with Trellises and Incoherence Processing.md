## QTIP: Quantization with Trellises and Incoherence Processing

- baseline方法是什么？
  - Baseline 是当前 SOTA 的 VQ-based LLM PTQ 方法 **QuIP#** 和 **AQLM**。它们使用 Vector Quantization (VQ) 将 d 维向量量化到 2^{kd} 大小的 codebook。由于 codebook 大小随维度指数增长，这些方法被硬件限制在 d≤8 维度：AQLM 使用 8D codebook (1MiB, 无法放入 L1 cache)，QuIP# 使用 8D E8 格点 codebook（高对称性可压缩 256×, 勉强放入 L1 cache）。低维度限制了 VQ 的 shaping/packing 优势，导致量化失真较高。
  - Baseline 全栈执行例子（QuIP# 2-bit 量化 Llama 2 7B 推理一个 token）：
    - **算法层**：RHT 使权重近似 i.i.d. 高斯 → BlockLDLQ 逐块量化（group size g, 8D VQ 每 8 个权重选 E8 格点最近邻） → 每 8 维存储 kd=2×8=16 bits → 反量化时查 8D codebook 恢复 FP16 权重。
    - **系统框架层**：PyTorch + 自定义 CUDA kernel。QuIP# 的 E8 codebook 压缩 256× 后可放入 L1 cache (约 8Kb)，通过查表实现快速反量化。
    - **编译框架/kernel调度层**：自定义 CUDA kernel 实现 on-the-fly E8 格点反量化 + FP16 GEMV。由于 d=8 维度过小，VQ 的 shaping 优势未充分发挥。
    - **硬件架构层**：NVIDIA GPU（RTX 6000 Ada 等），E8 codebook 查找在 GPU L1 cache 中完成。
  - Baseline 核心缺陷：VQ 维度被硬件 codebook 缓存大小限制在 ≤8，而信息论表明更高维度可显著降低量化失真（256D TCQ MSE 0.069 vs 8D VQ 0.089 vs D_R=0.063）。codebook 大小 O(2^{kd} d) 的指数增长在硬件上不可行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - QTIP 用 **Trellis Coded Quantization (TCQ)** 替代 VQ，利用 TCQ 的 **线性复杂度（O(2^L T)）** 实现超高维量化（有效维度 256），同时设计硬件高效的 bitshift trellis + compute-based codes 消除 TCQ 的推理开销。三个对应设计：
    1. **Bitshift Trellis → 解决 TCQ 的顺序解码和 trellis 存储问题**：bitshift trellis 中节点 i→j 有边当且仅当 j=(i·2^{kV} mod 2^L)+c，第 t 组权重仅依赖连续 L-bit 窗口，解码时仅需 kV-bit 位移（硬件原生支持）且完全并行化，无需存储 trellis 图结构（对比 naive TCQ 需存储 2^L×2^{kV} 边信息）。
    2. **Compute-based Codes → 解决 codebook 存储问题**：1MAD/3INST/HYB 码均为 lookup-free 或小 LUT 设计，在 GPU 上仅需 ≤4 指令/权重即时生成伪随机高斯值。HYB codebook 仅 2KiB（2^9×2 FP16），比 AQLM 的 1MiB 小 512 倍，可完全放入 L1 cache。这消除了 TCQ 需要存储 2^L×V 大小 codebook 的瓶颈。
    3. **RHT Incoherence Processing → 使权重适合 TCQ**：RHT 将 LLM 权重转化为近似 i.i.d. 高斯分布，而 TCQ 对 i.i.d. 高斯源天然高效（256D TCQ MSE 0.069 接近 D_R=0.063）。
  - QTIP 全栈执行例子（Llama 2 7B 2-bit HYB 码推理一个 token，L=16, V=2, Tx=Ty=16, Q=9）：
    - **算法层**：离线——RHT 变换 W̃ ← V_m S_m W S_n V_n^T → BlockLDLQ 逐块量化：每 Tx×Ty=16×16=256 维序列用 Viterbi 算法在 (L=16, k=2, V=2) bitshift trellis 上最小化 MSE 失真（O(2^16 × 256) ≈ O(1.7M) 操作/序列） → 输出每 256 维权重的 kT=2×16×16=512 bits 编码。在线——从 packed bitstream 读取 512-bit 块 → 通过 bitshift 操作逐 2D 向量提取 16-bit 状态 → HYB code 即时解码为 FP16 权重对（hash → LUT lookup → sign-flip, 摊销 2 指令/权重）。
    - **系统框架层**：PyTorch + QuIP# BlockLDLQ 框架 + 自研 CUDA kernel。QTIP 作为 BlockLDLQ 中 VQ 的 drop-in 替换（Algorithm 5），g=Ty 但有效维度 TxTy=256 >> g。HYB codebook 常驻 GPU L1 cache (2KiB, 可 32× 复制消除 bank conflicts)。
    - **编译框架层**：论文未明确说明。
    - **kernel 调度层**：自定义 CUDA kernel：bitshift trellis 解码（每步 kV=4 bit 位移）→ 1MAD/3INST/HYB code 即时生成高斯权重 → 16×16 MMA tile 矩阵-向量乘（Tensor Core）。支持 tail-biting（Algorithm 4）对齐 32-bit word。解码达 >80% 峰值显存带宽。1MAD: 2 GPU instr (MAD+vabsdiff4)、3INST: 3 instr (MAD+lop3+FADD)、HYB: 摊销 2 instr。
    - **硬件架构层**：NVIDIA GPU (RTX 6000 Ada, 3090, A6000 Ampere)。利用 16×16 MMA tile 和 L1 cache。ARMv8 CPU 也可用 NEON vqtbl4q_u8 查表实现 6-bit 1D HYB code（Q=6, V=1），达到与 3INST 相当的质量。
  - 效果：QTIP 256D TCQ 2-bit 量化 i.i.d. 高斯源 MSE 0.069（vs QuIP# 8D VQ 0.089, 改善 22%）；Llama 2 70B 2-bit perplexity gap 约减半；端到端推理速度与 QuIP# 持平（188 vs 186 tok/s, ≤4 instr/weight），同时量化质量更高。代码开源：https://github.com/Cornell-RelaxML/qtip
