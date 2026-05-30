## QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

- baseline方法是什么？
  - Baseline：BF16 LoRA RL 训练——使用 BF16 精度的预训练 LLM 权重，仅训练 LoRA adapter（rank=32），通过 GRPO/DAPO 进行 RL。BF16 LoRA 的缺陷：(1) **rollout 速度慢**：BF16 模型在推理时需要 15.2GB（7B），受限于显存带宽和计算量，rollout 吞吐量仅 115.4 tokens/s（batch=2, H100）；(2) **内存占用大**：7B 模型 15.2GB，32B 模型 62.3GB，无法在单 H100 80GB 上做 batch≥4 的 RL 训练（OOM）；(3) **收敛慢**：需 500+ steps 才能看到 reward 上升（BigMath 上的 7B 模型），因为 BF16 模型采样熵低，探索不足；(4) **对高学习率敏感**：LR>5e-6 就训练不稳定/崩溃；(5) **QLoRA (NF4) 更慢**：虽然内存更小（5.7GB），但 NF4 的 unpack+lookup table 操作使 rollout 比 BF16 还慢 0.7×−0.8×。
  - 全栈执行例子（Baseline: BF16 LoRA + GRPO on Qwen2.5-7B-Instruct, single H100）：
    - **算法pipeline**：预训练 Qwen2.5-7B-Instruct BF16 权重 → 添加 LoRA adapter (rank=32, α=32) → policy model 与 reference model (BF16 副本) 并存于 GPU → rollout 阶段 BF16 FP16 GEMM 推理 → 生成 G=8 个候选 → 计算 reward + group advantage → KL 散度约束 → AdamW-8bit 更新 LoRA → 每步约 600ms rollout + 200ms 其他计算，总 BF16 模型 15.2GB + LoRA + optimizer state ≈ 30GB+。
    - **系统框架**：论文未明确说明特定 Serving 框架用于 baseline rollout，但暗示使用 vLLM 作为 rollout 引擎；训练框架为自研 GRPO/DAPO 实现。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：BF16 推理使用标准 cuBLAS GEMM kernels，无特殊优化。NF4 baseline（QLoRA）使用 NF4→FP16 转换的 dequant kernel + cuBLAS GEMM。
    - **硬件架构**：NVIDIA H100 80GB GPU，Tensor Cores 执行 BF16×BF16 GEMM。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - 论文方法：QeRL = NVFP4 权重量化 + LoRA + Adaptive Quantization Noise (AQN)。解决 Baseline 缺陷的方式：(1) **NVFP4 加速 rollout**：用 NVFP4 量化权重（7B 仅 5.9GB，~61% 显存节省），结合 Marlin kernel 的 NVFP4×BF16 快速矩阵乘法，实现 1.2×−1.5× rollout 加速（batch=8 时从 1641→2091 tokens/s）；(2) **内存节省支持更大模型**：32B 仅 20.7GB，首次实现在单 H100 80GB 上训练 32B 模型（BF16 需 62.3GB，OOM）；(3) **量化噪声增强探索**：核心发现——量化误差 Δϵ = Q(θ)−θ 系统性增加采样熵 H(π(|q))，使输出分布更"平坦"，鼓励探索更广的 token 空间，从而在 BigMath 上仅需 ~200 steps 即可看到 reward 快速增长（vs BF16 LoRA 的 500+ steps）；(4) **AQN 动态噪声调度**：静态量化噪声对后期训练不利→引入 AQN，通过 RMSNorm 注入可控高斯噪声 Z_noisy∼N(0,σ²I)，按指数衰减 σ(k)=σ_start×(σ_end/σ_start)^((k-1)/(K-1)) 从探索过渡到利用，使 7B QeRL+AQN 在 GSM8K 达到 90.8%（vs BF16 LoRA 88.1%）；(5) **高学习率鲁棒性**：量化噪声的稳定化效应使 QeRL 能在 LR=3e-5 下稳定训练，reward 增长速率接近 BF16 LoRA 的 2 倍。
  - 全栈执行例子（QeRL: NVFP4+LoRA+AQN + GRPO on Qwen2.5-7B-Instruct, single H100）：
    - **算法pipeline**：Qwen2.5-7B BF16 → AWQ calibration (OpenThoughts-114k) → NVFP4 量化：\tilde{W} (4-bit) + S_FP32 + S_E4M3(E4M3, block=16) → 添加 LoRA adapter (rank=32, α=32) → QeRL pipeline：step 1. 判断 stage k，计算 σ(k)（stage 0=0 无额外噪声）；step 2. AQN 注入：Z_noisy∼N(0,σ²I) → w_noise = Z_noisy+w → RMSNorm_noise(x)=w_noise⊙x/√(mean(x²)+δ)（等价于乘法噪声 (Z_noise/w+I)⊙\hat{W}）；step 3. rollout：policy model（NVFP4 量化+AQN 噪声）生成 G=8 个候选，Marlin kernel 执行 NVFP4×BF16 GEMM；step 4. 计算 group advantage A_i = (r_i−mean(r))/std(r)；step 5. 仅更新 LoRA A,B（量化权重冻结）。模型 5.9GB+LoRA+optimizer≈~12GB，远小于 BF16 baseline。
    - **系统框架**：vLLM 作为 rollout 引擎（memory utilization=0.30 for 7B），训练框架为自研 GRPO/DAPO 实现。
    - **编译框架**：论文未明确说明。
    - **kernel调度**：Marlin kernel 加速 NVFP4×BF16 推理——kernel 从 global memory 加载 4-bit packed weights 到 shared memory → dequant: ŵ=S_FP32×S_E4M3[block]×unpack_4bit(w̃) → BF16 GEMM → 加 LoRA 输出。AQN 噪声经 RMSNorm 注入，不破坏 kernel 的 NVFP4×BF16 op 路径。
    - **硬件架构**：NVIDIA H100 80GB GPU，Tensor Cores 执行 BF16 GEMM + Marlin 优化的 NVFP4 dequant+compute。
