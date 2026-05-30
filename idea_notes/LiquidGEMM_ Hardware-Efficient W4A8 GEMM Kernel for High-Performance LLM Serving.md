## LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving

- baseline方法是什么？
  baseline是QServe [15]的W4A8 GEMM实现，使用QoQ（Quality over Quantity）dequantization算法。QoQ通过progressive quantization将INT8限制在[-119, 119]范围避免乘法溢出，再通过"先乘后减"策略（Q_u4 · s_i8 - s_i8 · z_i8）处理dequantization。但减法步骤依赖vadd伪指令——每条vadd被lowering为十余条底层指令——导致CUDA Cores上dequantization开销巨大（占warp stalls的21%）。baseline的GEMM执行采用简单的load→dequantize→MMA串行pipeline，dequantization无法被有效重叠，导致：(1) memory-bound场景下W4A8与W8A8性能相当（理论应更快）；(2) compute-bound场景下W4A8比W8A8慢2x（理论应相当）。

  全栈执行例子（以QServe W4A8 GEMM处理LLaMA2-7B FFN层，batch=256为例）：
  - 算法层：QServe两级量化——FP16→INT8 (per-channel, [-119,119]) → UINT4 (per-group, group_size=128)。激活SmoothQuant动态per-token量化FP16→INT8。
  - 系统框架层：QServe serving系统通过PyTorch调用QServe GEMM kernel。权重离线量化存储为UINT4，激活在线量化。KV cache 4-bit量化。
  - 编译框架层：未使用编译框架自动生成kernel。QServe hand-crafted GEMM kernel。
  - kernel调度层：QServe GEMM kernel执行简单pipeline——从GMEM加载UINT4 weight (LDG.32) → unpack 4-bit到8-bit → QoQ dequantization (CUDA Cores, vadd → dozen instructions) → WGMMA INT8 MMA (Tensor Cores)。dequantization与MMA串行，CUDA Cores成为瓶颈。LDS.32加载（非LDS.128）浪费一半带宽。
  - 硬件架构层：NVIDIA H800 GPU。CUDA Cores throughput远低于Tensor Cores（H100: 60 TFLOPS CUDA vs 990 TFLOPS TC INT8, 16.5x差距）。Tensor Cores在dequantization期间空闲，CUDA Cores在MMA期间空闲——两种硬件单元交替闲置。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出LiquidGEMM，包含两个co-designed技术解决W4A8 dequantization瓶颈：

  (1) **LiquidQuant (LQQ)量化算法**：通过rotation-based transformation将INT8先shift到UINT8域再量化为UINT4，利用two's complement同余性质（i ≡ j mod 2^8 → 相同binary representation）设计dequantization。dequantization仅需两条32-bit硬件指令（IMAD + XOR）处理四个元素，无overflow、无vadd伪指令开销。从QoQ的10+条指令/元素降至0.5条指令/元素，α=2（而非QServe的α≥10），远低于与memory load重叠所需的阈值α≤5.07。

  (2) **Implicit Fine-Grained Pipeline (ImFP)**：采用single-producer multiple-consumer模型替代ExCP的多WG显式同步。Load WG通过TMA加载weight到SMEM后切分为fine-grained tasks，多个Compute WG竞争获取task并各自完成dequantization+MMA，dequantization与MMA跨Compute WG自然重叠。消除ExCP的SMEM↔RF round-trip数据搬运和barrier同步开销。配合Dual-MMA packed layout让每个线程用单条LDS.128加载32个UINT4元素。

  全栈执行对比baseline（以LiquidGEMM处理同一LLaMA2-7B FFN层，batch=256为例）：
  - 算法层：LiquidQuant替代QoQ——FP16→INT8 (per-channel, [-119,119]) → shift到UINT8 → UINT4 (per-group, group_size=64)。Dequantization: Q_i8 = (Q_u4 * s_u8 + a) XOR 0x80，两条指令四元素。准确率通过WikiText2/zero-shot评估保持与QServe相当。
  - 系统框架层：LiquidServe自建serving系统。与QServe不同的是KV cache使用INT8（非4-bit），batch size可扩展到更大（如LLaMA2-70B batch=184 vs QServe batch=64），GEMM不再是瓶颈。
  - 编译框架层：基于CUTLASS/Cute构建，WGMMA/TMA/barrier用PTX包装，dequantization用CUDA直接实现。计算重构为Y=(WX^T)^T以利用WGMMA的m=64固定维度。
  - kernel调度层：ImFP替代串行pipeline。Load WG (TMA) → SMEM task queue → Compute WG_0 (LDS.128→unpack→IMAD+XOR dequantization→WGMMA MMA) 与 Compute WG_1 (同样pipeline, 不同task) 并发执行。CUDA Cores做dequantization期间Tensor Cores在另一Compute WG做MMA，反之亦然。无SMEM↔RF round-trip，无显式barrier同步（硬件task scheduling管理）。彻底消除CUDA Core瓶颈。
  - 硬件架构层：NVIDIA H800 GPU。TMA、CUDA Cores、Tensor Cores三种异构硬件通过ImFP实现pipeline-parallel执行——Weight loading (TMA) ∥ Dequantization (CUDA Cores, WG_0) ∥ MMA (Tensor Cores, WG_1)。从"交替闲置"变为"持续饱和"。2.90x kernel speedup vs QServe，4.94x system speedup。
