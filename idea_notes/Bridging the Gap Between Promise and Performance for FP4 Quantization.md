## Bridging the Gap Between Promise and Performance for FP4 Quantization

- baseline方法是什么？
  Baseline 是标准 **RTN（Round-to-Nearest）量化**直接应用于 MXFP4 和 NVFP4 微缩放格式，配合 absmax scaling。具体流程：(1) 将权重/激活按 G=32（MXFP4）或 G=16（NVFP4）分组；(2) 每组用 absmax 计算 shared scale（MXFP4 scale 量化为 E8M0 即 power-of-two，NVFP4 scale 量化为 E4M3 即完整 FP8）；(3) 以 FP4 E2M1 格式对归一化后的元素执行 RTN 量化。也对比了 GPTQ（标准 INT GPTQ 直接套用到 FP4）、SmoothQuant（对角 rescaling 迁移激活异常值到权重）、QuaRot/SpinQuant（全局 Hadamard 旋转后 RTN）。

  Baseline 全栈执行例子（Llama-3.1-8B-Instruct MXFP4 RTN W4A4）：
  - 算法pipeline：加载 FP16 权重 → 逐层线性层：权重按 G=32 分组 → 每组 absmax scale s_G → s_G 量化为 E8M0（power-of-two）→ 权重归一化后 RTN 量化到 E2M1 FP4 网格 → 激活同理 → 推理：Q(WH_k)@Q(XH_k)^T（无旋转时 H_k=I）。MXFP4 RTN 平均 accuracy recovery 仅 87.83%（FP16=78.93, RTN=69.32）。NVFP4 RTN recovery=94.67%。
  - 系统框架：PyTorch 模拟量化（fake quantization），HuggingFace Transformers。校准集 FineWeb 1024 sequences。
  - 编译框架：论文未明确说明。
  - kernel调度：标准 PyTorch FP16 GEMM，模拟量化仅用于精度测量。
  - 硬件架构：论文未明确说明（实验在 GPU 上执行 PyTorch 模拟量化）。

  **Baseline 的核心缺陷（通过量化误差分析揭示）：**
  1. **MXFP4 的 E8M0 power-of-two scale 引发严重量化误差**：scale 量化为 power-of-two（E8M0）在保持硬件乘法简化的同时，引入了较大近似误差。MXFP4 RTN 下 MLL 平均下降 ~10%，显著劣于 NVFP4 和 INT4。MSE 分析显示 MXFP4 的 top-element 误差随 group size 增大而保持恒定（受限于 E2M1 而非 E8M0），而 per-element MSE 随 G 增大增长。
  2. **NVFP4 的小 group size（G=16）天生做异常值抑制**：传统异常值缓解技术（如 SmoothQuant 的 per-channel scaling）在 NVFP4 的 G=16 下被证明无效——小 group 已经通过细粒度 absmax scaling 隐式处理了异常值。因此 NVFP4 RTN 即使不加任何额外技术已表现良好。
  3. **Hadamard 旋转对 NVFP4 精度有负面影响**：分析证明（Lemma 1-2），对 Laplace 分布（原生权重/激活）应用 Hadamard 旋转转为 Normal 分布后，在小 G 下 MSE 增大（因为 top-element 误差被均匀扩散到整组）。NVFP4 的 G=16 恰好在此区间，所以 RTN+HT 比 RTN 精度更差。这解释了为何 QuaRot/SpinQuant 在 NVFP4 上无效甚至有害。
  4. **标准 GPTQ 未针对 FP4 格式优化**：直接套用 INT GPTQ 的 absmax scaling + uniform grid 到 FP4 非均匀网格，未利用 MSE-optimized grid、未处理旋转后的格式适配。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **MR-GPTQ（Micro-Rotated-GPTQ）**，通过量化误差理论分析驱动三个核心创新逐一解决 baseline 缺陷，并配套 **QuTLASS** GPU kernel 实现零开销部署：

  **(1) 量化误差分析驱动方法选择（解决缺陷 1-3 的根源问题）**
  论文建立了 MXFP4/NVFP4 的理论 MSE 模型（Laplace→原生权重、Normal→旋转后权重），推导 per-element MSE 和 top-element MSE 的渐近收敛率：
  - Laplace（原生）: R_L(G) = Θ((log G)² G^(-δ))，小 G 下 MSE 低
  - Normal（旋转后）: R_N(G) = Θ(√(log G) G^(-δ²))，大 G 下 MSE 低
  由于 0 < δ² < δ < 1，存在 crossover 现象：小 G 时 Laplace MSE 更低（NVFP4 G=16 不应旋转），大 G 时 Normal MSE 更低（MXFP4 G=32 应旋转）。这直接指导了设计决策：MR-GPTQ-MXFP4 必须旋转，MR-GPTQ-NVFP4 可选旋转（若配合 scale 优化可补偿旋转引入的局部误差）。

  **(2) MSE-Optimized Grids + Static Act-Order + Block-wise Rotations（解决缺陷 4）**
  - **MSE-Optimized Grids**：替代标准 absmax scale + RTN grid。对每个 tensor 求解 min_{s_T, s_{G_1...G_k}} Σ_i ||X̂_i - X_i||²，通过交替优化 per-tensor scale s_T 和 per-group scales s_G 最小化量化 MSE。NVFP4 无旋转时此优化产生一致改善；MXFP4 旋转后使用统一静态值。
  - **Static Activation Reordering**：标准 GPTQ 的 dynamic act-order 在推理时需实时重排列，产生 10-20% 延迟开销。MR-GPTQ 改为：先确定 scales/grid → 再按 Hessian 重排列 → 应用 GPTQ 量化 → 恢复原始列序。与 dynamic 效果相同，零推理开销。
  - **Block-wise Hadamard Rotations**：对 MXFP4（G=32），旋转将 Laplace 分布转为 Normal，降低 per-element MSE（与大 G 一致）。旋转大小匹配 group size，形成 "micro-rotation" 设计（k=32 for MXFP4, k=16 for NVFP4），区别于 QuaRot 的全局旋转。

  **(3) QuTLASS Fused Kernel 实现零推理开销（将理论加速兑现为实际加速）**
  - 权重端旋转离线预融合：W_rot = W·H_k，量化存储为 Q(W_rot)，无运行时旋转开销
  - 激活端 fused online rotation：QuTLASS kernel 将 H_k 加载（k<256 时 memory-bound，任意矩阵同成本）+ 旋转 + 量化 + scale 计算融合为单 kernel，epilogue 直接输出 FP4 量化值
  - MXFP4 在 B200 上 matmul throughput **超过** NVFP4 ~15%（power-of-two scales 降低硬件乘法开销）

  论文方法全栈执行例子（Llama-3.1-8B MXFP4 MR-GPTQ W4A4）：
  - 算法pipeline：加载 FP16 权重 → 离线阶段：对每个线性层（Q/K/V/O/gate/up/down）→ ① block-wise Hadamard 旋转 W_rot = W·H_32（k=32 匹配 MXFP4 G=32）→ ② FineWeb 1024 校准集前向计算 Hessian H=2X^T X → ③ 按原始列序计算 MSE-optimized scales & grid（MXFP4 使用统一静态 s_T）→ ④ Static act-order：按 Hessian 对角线重排列 → GPTQ 逐列量化 + 误差补偿（OBS 框架，各列共享 H^{-1}）→ 恢复原始列序 → ⑤ 存储 MXFP4 packed 权重（4.25 bits/elem）。推理时：FP16 激活 X → QuTLASS fused kernel: X_rot=X·H_32 → MXFP4 quantize(X_rot) → scale rearrangement（Triton kernel for tcgen05.mma）→ FP4 matmul（Blackwell hardware）→ 输出。结果：Average Recovery 93.31%（RTN=87.83%, GPTQ=89.47%），接近 NVFP4 水平。
  - 系统框架：PyTorch 模拟量化（精度实验）/ vLLM + QuTLASS kernel（性能实验）。量化代码：FP-Quant（https://github.com/IST-DASLab/FP-Quant）。
  - 编译框架：论文未明确说明。
  - kernel调度：QuTLASS v1.0（https://github.com/IST-DASLab/qutlass）。B200 单层 speedup 3.6×（ideal 4×），端到端 2.2×（vLLM Llama-3.3-70B）。RTX 5090 单层 6×（ideal 8×），端到端 4×。
  - 硬件架构：NVIDIA B200（SM100）/ RTX 5090（SM120）Blackwell GPU。利用 tcgen05.mma 硬件 FP4 矩阵乘指令。

  关键设计动机映射：
  - MXFP4 E8M0 scale 误差大 → MSE-optimized grids 交替优化 s_T 和 s_G 最小化整体 MSE + scale fitting (×4/3 unbiased estimate)
  - MXFP4 大 G=32 下 Normal 分布 MSE 更低 → block-wise Hadamard 旋转（G=32 匹配旋转 block size 32）
  - NVFP4 小 G=16 下 Hadamard 旋转有害 → 分析指导 NVFP4 无旋转 + MSE grid 优化（利用 NVFP4 E4M3 scale 精度优势）
  - Standard GPTQ dynamic act-order 有推理开销 → Static act-order：先定 grid → 重排量化 → 恢复原序，零开销
  - 在线旋转可能抵消 FP4 硬件加速收益 → QuTLASS fused kernel：k<256 时旋转 memory-bound，任意矩阵同成本，epilogue 直接量化无中间写入
