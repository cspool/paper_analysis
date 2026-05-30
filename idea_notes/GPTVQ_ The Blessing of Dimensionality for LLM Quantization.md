## GPTVQ: The Blessing of Dimensionality for LLM Quantization

- baseline方法是什么？
  Baseline 是 **均匀 INT4 量化 + GPTQ 后训练量化**（例如 GPTQ W3/W4 g128）。标准 LLM 推理全栈中，权重以 INT4 存储，每个 group（128 个权重）共享一个 FP16 scale，推理时通过 scale 反量化到 FP16 后执行矩阵乘法。

  Baseline 全栈执行例子（Llama-2-7B INT4 g128, 移动 CPU/GPU）：
  - **算法pipeline**：FP16 权重 → GPTQ 逐列量化 + Hessian 误差补偿 → 每 group 128 个权重共享一个 scale → 推理时读取 INT4 packed 权重 → scale 反量化到 FP16 → FP16 GEMM。均匀量化的 grid 是等间隔的，优化空间仅为 2^4=16 个等间隔值。
  - **系统框架**：llama.cpp（开源）或自研推理引擎。Llama.cpp 使用 Q4_0 INT4 量化（block size 32）。
  - **编译框架**：论文未明确说明。
  - **kernel调度**：CPU 端使用 SIMD 指令（ARM NEON / x86 AVX）加速 INT4 反量化 + GEMM。GPU 端使用 CUDA kernel 或 Triton kernel。
  - **硬件架构**：移动端 Snapdragon X Elite CPU / NVIDIA GPU（RTX 4090/A100/H100），无自定义硬件。

  **Baseline 的核心缺陷：**
  1. **均匀量化 grid 表达能力受限**：均匀量化将每个 group 映射到等间隔的 2^b 个值上，无法自适应权重分布。当权重分布非均匀（如长尾分布、多峰分布）时，等间隔 grid 浪费大量量化级别在低密度区域，高密度区域精度不足。
  2. **DRAM 带宽瓶颈限制 token rate**：LLM 自回归推理中每生成一个 token 需从 DRAM 读取所有权重一次，DRAM 带宽是主要瓶颈。INT4 虽已将 FP16 压缩 4×，但 8B 模型仍需约 4.3 GB footprint，DRAM 带宽余额有限。
  3. **现有 VQ 方法在移动端低效**：AQLM 等方法使用 8D VQ + 16-bit 索引 + 大 codebook（2^16 个 8D entries），无法利用移动 CPU 的 TBL 指令（仅支持 5-6 bit index → 8-bit value）。解码 latency 过高，抵消了 footprint 减小的带宽收益。
  4. **AQLM 压缩时间长**：Llama-v2-7B 需约 35 小时 on H100（含 block FT），GPU 资源消耗巨大。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **GPTVQ**，通过 VQ 表示 + 移动 CPU 实现 + 快速 PTQ 算法的协同设计，系统性解决 baseline 各项缺陷：

  **(1) VQ 替代均匀量化，提升 representational accuracy（解决缺陷 1）**
  将权重按 d 维向量分组量化到非均匀 codebook。对于给定 bit budget（如 3.125 bpv），2D VQ 的 64 个质心可自由分布在二维空间中，形成非均匀 grid，比均匀 INT4 的 16 个等间隔点表达能力更强。论文用 SQNR metric 验证：d 越高（1D→2D→4D），SQNR 越高，且实验证实 2D 和 4D VQ 在几乎所有模型和 bitwidth 下比 1D VQ 和均匀量化有更低 perplexity 和更高零样本准确率。

  **(2) 硬件友好的 VQ 参数选择 + LUT 解码 kernel（解决缺陷 3）**
  与移动 CPU 硬件协同设计 VQ 参数：固定 2D VQ + 6-bit index（3 bits/dimension），codebook 最多 64 entries。这直接匹配移动 CPU 的 TBL 指令规范（6-bit index → 8-bit value），每个维度仅需 1 条 TBL 指令解码。对比 AQLM 的 16-bit index 需要 SVE gather 指令（性能更差），GPTVQ 的解码延迟极低，使 VQ 的 footprint 减小（19%）能转化为实际的 token rate 提升（10%）。Table 6 验证：CPU 端 VQ 2D 2.25 bpv 延迟 0.87× vs INT4，吞吐反超。

  **(3) GPTVQ 算法 = GPTQ 扩展 + 加权 EM 初始化 + Codebook Update（解决缺陷 4）**
  - 将 GPTQ 的逐列量化扩展为逐 d 维向量量化，误差沿 d 维累积后一次性补偿
  - EM 初始化：用 Hessian 加权的马氏距离（公式 4-6），E-step 分配质心，M-step 伪逆闭式解更新质心，比标准 k-means 更好地利用校准数据信息
  - Codebook update（附录 A）：GPTVQ 完成后通过梯度下降（PyTorch）层内微调 codebook 值，以极小开销（~30% 额外时间）提升精度
  - 结果：Llama-v2-7B 2D VQ 压缩时间仅 2.5h（H100），vs AQLM no BFT 18.3h（7.3× 加速），且精度 competitive（WikiText2 PPL 7.11 vs AQLM 7.49）

  **(4) 正交组合 LoRA adapters 恢复精度（额外贡献）**
  GPTVQ 量化后的 base model 可与 LoRA adapters 结合：frozen adapter（FP16 模型训练的 LoRA 直接挂载）或 trained adapter（在量化模型上训练 LoRA）。GPTVQ 4D 2.125 bpv + LoRA trained 在 GSM8k 上达 32.5-35.0%（L2-7B），显著超越 LoftQ（20.9%）。

  论文方法全栈执行例子（Llama-v3-8B 2D VQ 3.125 bpv, Snapdragon X Elite）：
  - **算法pipeline**：FP16 权重 → 校准集（WikiText2 128×2048）前向收集 Hessian → 逐 column block：每 256 columns 进入新 group → EM 初始化该 group 的 8-bit codebook（64 entries, 2D）→ GPTVQ 逐 d=2 列量化 + Hessian 补偿 → Codebook update（梯度下降 fine-tune）→ Codebook 量化到 INT8（或 INT4）→ 输出：packed 6-bit indices + per-block 64-entry INT8 LUT + per-block FP16 scale → **推理**：DRAM → SoC cache → TBL 指令解码 6-bit index → 2D 值合并 → scale × decoded int = FP16 → SIMD GEMM。Footprint 3.52GB (-19% vs INT4 4.33GB)，Throughput 26.15 tok/s (+10% vs Ours INT4 23.81 tok/s, +45.7% vs llama.cpp INT4 17.95 tok/s)。
  - **系统框架**：Qualcomm 自研 C 语言推理引擎（vector intrinsics + SIMD + polyhedral compiler）。移动端 Snapdragon X Elite + Clang 18.1 + Polly。
  - **编译框架**：Polyhedral compiler（Polly）用于细粒度向量化编排。
  - **kernel调度**：CPU TBL kernel（解码）+ SIMD GEMM kernel。GPU CUDA kernel（char4/uchar4/char128 vector types）。VQ 2D CPU 数据加载延迟测试：3.125 bpv = 0.96× 延迟 vs INT4（同 footprint 仅 0.78×），2.25 bpv = 0.87× 延迟 vs INT4（footprint 仅 0.56×）。
  - **硬件架构**：Snapdragon X Elite CPU（ARM TBL 指令 6-bit→8-bit）+ NVIDIA GPU（RTX 3080 + H100）。无自定义硬件，仅利用现有 CPU ISA 扩展。

  关键设计动机映射：
  - Uniform quantization 表达力低（16 等间隔点）→ VQ codebook 64 个任意分布质心，提高 SQNR + 降低 PPL
  - DRAM 带宽瓶颈 → 更小的 bpv（3.125 vs 4.125）减少 footprint 19%，直接转化为 10% token rate 增益
  - 移动 CPU TBL 指令特性（6-bit→8-bit）→ 2D VQ + 6-bit index 配置匹配硬件，解码快于 DRAM 带宽
  - AQLM 压缩时间长 → GPTVQ 单次从左到右扫描（复用 GPTQ lazy update）+ 闭式解 EM，2.5h vs 18.3h
  - VQ 固有精度损失 → Codebook update 梯度下降 + LoRA adapter 正交补偿
