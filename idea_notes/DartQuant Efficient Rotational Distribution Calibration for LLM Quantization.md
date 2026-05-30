## DartQuant Efficient Rotational Distribution Calibration for LLM Quantization

- baseline方法是什么？
  Baseline 是端到端微调旋转矩阵的 LLM PTQ 方法，具体包括 **SpinQuant** 和 **OSTQuant**。核心流程：(1) 在 Transformer block 中插入 4 个旋转矩阵 R1-R4（基于 Computational Invariance）；(2) 将 R1, R2 视为可学习网络参数，插入 pseudo-quantizers；(3) 使用 Cayley SGD（黎曼优化器，在 Stiefel 流形上优化保证正交性）在 calibration set 上端到端微调；(4) 优化目标为 task-specific loss（如 KL divergence）。校准完成后将 R1, R2 融合入相邻权重矩阵实现零推理开销。

  Baseline 全栈执行例子（SpinQuant, LLaMA-2 70B, w4a4kv16）：
  - 算法pipeline：加载 FP16 LLaMA-2 70B → 在 Transformer block 中插入 pseudo-quantizers（W4A4）→ 初始化 R1,R2 为随机 Hadamard 矩阵 → 用 WikiText2 128 samples 作为校准集 → Cayley SGD 端到端微调 R1,R2（每步计算梯度 → Cayley 变换→ 投影回 Stiefel 流形保证正交 → 更新） → 量化 GPTQ 重建权重 → 融合 R1/R2/R1^T/R2^T 入相邻权重。校准需 42.9 GPU-hours、238.89 GiB（A800）。
  - 系统框架：PyTorch + HuggingFace Transformers。使用 Cayley SGD 优化器（`cayley_optimizer`）。
  - 编译框架：论文未明确说明。
  - kernel调度：推理时使用快速 Hadamard kernel 处理 R3, R4（在线旋转）。R1, R2 已离线融合，无额外推理开销。
  - 硬件架构：NVIDIA A800 GPU 服务器。

  **Baseline 的核心缺陷（通过图 3 和表 1 揭示）：**
  1. **资源开销巨大**：70B 模型旋转优化需 42.9 GPU-hours + 238.89 GiB 显存（SpinQuant），OSTQuant 更高达 44 GPU-hours + 583.86 GiB。与 PTQ "快速部署" 目标矛盾。
  2. **端到端微调引入过拟合**：小校准集上的 task-specific loss 微调导致过拟合——表 1 显示 SpinQuant 在 PTB 校准集上 PPL 提升明显（37.91→38.24），但在其他数据集退化严重（WikiText2: 5.47→6.02, C4: 7.26→8.13）。零样本任务上 SpinQuant 和 OSTQuant 反而不如 QuaRot（随机 Hadamard）。
  3. **无法显著降低 outliers 和量化误差**：图 3 显示端到端微调后的旋转矩阵在减少 outliers 数量和降低量化误差方面改进有限——变换后的激活与随机 Hadamard 差异不大。
  4. **Cayley SGD 正交优化器计算昂贵**：需在 Stiefel 流形上做复杂投影计算（约 6n³ 额外计算量），优化时间为标准 SGD 的约 2 倍。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **DartQuant**，通过三个创新设计从根本解决 baseline 的问题：

  **(1) Rotational Distribution Calibration 替代端到端微调（解决缺陷 1-3）**
  将旋转矩阵优化从 "网络参数端到端微调" 重新定义为 "将激活分布变换为最适合量化的分布"。不再使用 task-specific loss，而是直接约束旋转后激活的分布特性——最小化 outliers 数量。这消除了对 task-specific loss 的依赖，从根本上避免过拟合。通过仅 128 条校准样本收集激活值（无需标签），校准过程不涉及反向传播至模型输出，大幅降低资源消耗。

  **(2) Whip Loss 驱动激活趋向均匀分布（解决缺陷 3）**
  Baseline 的量化 loss、方差、峰度等目标均无法有效优化激活分布。Whip Loss 的数学设计：`Whip = Σ exp(-|x_i|)`。受 Laplace→Uniform 的 CDF 变换 `U_X(x) = τ[exp(x/b)-1]` 启发，Whip 在零附近有较大梯度（将小值推开），在 norm-invariance 约束（||Rx|| = ||x||）下产生 "峰平滑" 效应：小值增大 → outliers 被迫减小以保持 L2 范数不变 → 整体分布趋向均匀。图 6 直方图验证：Whip 优化后的激活分布最接近均匀分布，outliers 被有效消除。图 7a 验证：Whip 的量化误差下降曲线远优于量化 loss、方差、峰度。

  **(3) QR-Orth 替代 Cayley SGD（解决缺陷 4）**
  不再直接在 Stiefel 流形上优化 R，而是引入隐参数 Z（任意矩阵），通过 QR 分解获得正交矩阵 `R = QR(Z)` 作为实际旋转矩阵。优化 Z 可以用任何标准优化器（SGD/Adam）在欧几里得空间进行，无需投影操作。校准完成后丢弃 Z，仅保留 R 融合入模型。QR-Orth 的额外计算量约 4/3 n³（vs Cayley 6n³），实测 100 步 SGD 耗时 5.7h vs Cayley 8.2h（1.44×）。由于收敛更快，QR-Orth SGD 仅 6 步即达到 Cayley SGD 100 步效果（41× effective 加速）。

  论文方法全栈执行例子（DartQuant, LLaMA-2 70B, w4a4kv16）：
  - 算法pipeline：加载 FP16 LLaMA-2 70B → 前向传播 128 WikiText2 samples 收集各层激活 X → Token sampling 10% → 初始化 Z_0 为随机 Hadamard 矩阵 → **每层独立校准（Algorithm 1）**：for k=0..T (T=10 epochs, lr=1e-3, SGD, batch=64) → R = QR_decomposition(Z) → O = X @ R → L = Whip(O) → Z = Z - η ∂L/∂Z → 最终 R = QR(Z) → **融合**: R1 融入 W_q/W_k/W_v/W_up/W_gate/W_o/W_down/W_embedding/W_lm_head，R2 融入 W_v/W_o → GPTQ 量化权重为 INT4 → 推理时激活 INT4 量化 → INT4×INT4 TensorCore GEMM → INT32 结果转换为 FP16。校准耗时 0.91 GPU-hours (A800) / 2.90 GPU-hours (3090)，内存 23.47 GiB。Llama-2 70B w4a4kv16 零样本 avg=69.02 (FP16=69.53, loss only 0.5%)。
  - 系统框架：PyTorch + HuggingFace Transformers。基于 SpinQuant/QuaRot 代码修改。标准 SGD + QR-Orth 替代 Cayley SGD。
  - 编译框架：论文未明确说明。
  - kernel调度：推理使用快速 Hadamard kernel 处理在线 R3, R4（与 SpinQuant 相同）。R1, R2 预融合无推理开销。
  - 硬件架构：NVIDIA A800 GPU 服务器 / RTX 3090 单卡。首次在 3090 上完成 70B 旋转校准。

  关键设计动机映射：
  - 端到端微调过拟合 + 资源高 → Rotational Distribution Calibration：用激活分布约束替代 task-specific loss，仅需前向传播收集激活
  - 量化 loss/方差/峰度无法有效减少 outliers → Whip Loss：数学上驱动 Laplace 分布趋向 Uniform，在 norm-invariance 约束下产生 "峰平滑" 效应
  - Cayley SGD 投影计算昂贵（6n³ 额外）→ QR-Orth：隐参数 Z → QR 分解得 R，欧几里得空间优化，仅 4/3 n³ 额外
  - 端到端微调不能显著降低量化误差（图 3）→ Whip 直接降低量化误差（图 7a 验证快速收敛），图 6 直方图验证分布均匀化效果
