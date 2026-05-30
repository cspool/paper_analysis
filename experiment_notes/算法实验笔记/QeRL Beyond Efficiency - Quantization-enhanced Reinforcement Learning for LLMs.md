## QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QeRL 将 NVFP4 权重量化与 LoRA 低秩适配结合，用于 LLM 强化学习（RL）训练。核心 pipeline：(1) 用 AWQ 对预训练 LLM 权重做 NVFP4 后训练量化（calibration: OpenThoughts-114k 的 256 序列×2048 tokens）；(2) 在量化模型 Q/K/V/O/gate/up/down 层添加 LoRA adapter（rank=32，约 1% 可训参数）；(3) 通过 GRPO 或 DAPO 进行 RL 训练。(4) 创新 AQN 机制：高斯噪声融入 RMSNorm scale 参数，按指数衰减调度（σ_start=1e-2, σ_end=5e-4, K=10 阶段），实现从探索到利用的动态过渡。伪代码：每步确定 stage k → σ(k)=σ_start×(σ_end/σ_start)^((k-1)/(K-1)), stage 0 无噪声 → 注入噪声到旧策略 π_θold ← π_θ+N(0,σ²) → rollout 生成 G 个候选 → 计算 group relative advantage → 用 GRPO/DAPO 目标更新 LoRA。前向推理用 Marlin kernel 加速 NVFP4×BF16 矩阵乘法，梯度仅回传 LoRA 层。
  - 实验比较：(a) QeRL vs BF16 LoRA vs BF16 Full FT vs QLoRA(NF4+LoRA) 在 GSM8K 准确率；(b) BigMath 训练的 MATH500/AIME24/25/AMC23 上比较；(c) 量化格式消融 NVFP4 vs MXFP4 vs NF4；(d) noise scheduler 消融 exponential vs linear vs cosine vs logarithmic；(e) LoRA rank 消融 16/32/64/128；(f) 学习率消融；(g) rollout 吞吐量/内存对比（batch=2/4/8）。

- 硬件平台是什么，配置是什么。
  - NVIDIA H100 80GB GPU。速度测试在单 H100，最终评估模型在 8×H100 训练。vLLM 引擎用于 rollout（memory utilization: 3B=0.20, 7B=0.30, 14B=0.45, 32B=0.40）。环境：CUDA≥12.4.1, Linux, 64GB RAM。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen2.5-3B/7B/14B/32B-Instruct（基础通用模型，未做数学微调）
  - 训练数据：GSM8K（7500 样本, generation number=8）, BigMath（122000 样本, generation number=16, difficulty level 3-5 或 4-5）
  - Benchmarks：GSM8K, MATH500, AIME 2024, AIME 2025, AMC 23, 均报 Pass@1
  - RL 算法：GRPO（GSM8K 训练）, DAPO（BigMath 训练）
  - RL 超参数：AdamW-8bit, LR=1e-5(QeRL/QLoRA)/5e-6(LoRA BF16), Batch=128, Samples per prompt=8(GSM8K)/16(BigMath), Max response=4096/8192, temperature=1.0, Clip(0.2,0.28)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/NVlabs/QeRL (Apache 2.0)
  - 张量计算流程：(1) NVFP4 量化权重 \tilde{W} + FP32 全局 scale S_FP32 + FP8(E4M3) block-wise scale S_E4M3（block=16 元素），dequant: \hat{W}=S_FP32·(S_E4M3⊙\tilde{W})；(2) LoRA: ΔW=BA, B∈R^{d×r}, A∈R^{r×k}, r=32；(3) AQN: Z_noisy∈R^{1×d}∼N(0,σ²I) 并入 RMSNorm: RMSNorm_noise(x)=w_noise⊙x/√(mean(x²)+δ)，w_noise=Z_noise+w（等价变换为乘法噪声 (Z_noise/w+I)⊙\hat{W}）；(4) 前向：x_attn=RMSNorm_noise(x)·\hat{W}_{q,k,v}+LoRA_output, x_ffn=RMSNorm_noise(x)·\hat{W}_{gate,up}+LoRA_output；(5) 仅更新 LoRA 参数 A,B，量化权重冻结。使用方法：`python quantize_nvfp4.py --model Qwen/Qwen2.5-7B-Instruct` → `bash training/dapo_qwen2.5-7b_nvfp4_single_gpu.sh`。
