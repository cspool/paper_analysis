## MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 MoEQuant 量化框架，包含 EBSS（Expert-Balanced Self-Sampling）和 AGQ（Affinity-Guided Quantization）两个插件式模块，可与 GPTQ、AWQ 等现有 PTQ 方法无缝集成。EBSS 利用 LLM 自采样能力，通过累积概率和专家平衡因子引导搜索，生成专家分布均衡的校准集；AGQ 将 token-expert 亲和力（gating coefficient）纳入量化误差计算和 Hessian 统计，改进逐层量化过程中的权重更新精度。
  - 实验比较了 FP16、RTN、AWQ、GPTQ 和 MoEQuant（基于 AWQ 和 GPTQ 的变体）在 4-bit 和 3-bit 权重量化下的表现，涵盖 PPL（WikiText2、C4）和 7 个下游任务（MMLU、HumanEval、GSM8K、BoolQ、HellaSwag、OpenBookQA、MathQA），并在 Qwen-MoE-14B-Chat 和 DeepSeek-MoE-16B-Chat 上验证了对 instruction-tuned 模型的量化性能。

- 硬件平台是什么，配置是什么。
  - NVIDIA A6000 GPU。所有实验在 NVIDIA A6000 上完成，不涉及微调。

- 模型是什么。数据集和bench分别是什么。
  - 模型：DeepSeek-MoE-16B、Qwen-MoE-14B（Qwen1.5-MoE-A2.7B-14B）、Mixtral-8x7B，以及它们对应的 instruction-tuned 版本（Qwen-MoE-14B-Chat、DeepSeek-MoE-16B-Chat）。
  - 校准集：WikiText2（baseline 方法所用，128 segments），EBSS 使用模型自采样生成校准集（branch number w=4，temperature τ=1.2）。
  - 评估数据集：WikiText2、C4（perplexity）；MMLU、HumanEval、GSM8K、BoolQ、HellaSwag、OpenBookQA、MathQA（下游任务）。复杂推理任务（MMLU、GSM8K、HumanEval）基于官方 repository 评估，其他 zero-shot 任务使用 lm-evaluation-harness v0.4.4。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文声称代码将开源于 https://anonymous.4open.science/r/MoEQuant-DDFD/README.md（ICML 2025 匿名仓库）。
  - 算法 pipeline：
    1. **EBSS 校准集生成**：给定 MoE 模型 M，设定 beam width w、sequence length n、温度 τ。初始化 w 个空序列 S={}。对每个 step i=1..n，对每个 beam S^t，从词汇表 V 中计算 score(S^t||v) = (-1/(i+1)) * (R_{S^t} + log P(v|S^t)) + σ(M, S^t)/τ。取 top-w 候选作为新的 beam。完成后得到 w 个长度为 n 的序列作为校准集 D*。此过程将搜索复杂度从 O(m^n) 降至 O(wn)。
    2. **AGQ - 亲和力感知量化误差**：传统 layer-wise 量化损失为 L(W_hat) = ||WX - W_hat X||_F^2。AGQ 将其重新定义为 L(W_hat) = Σ_i c_i · ||W x_i - W_hat x_i||_F^2，其中 c_i 是 token i 对该 expert 的 gating coefficient。对于 Hessian-based 方法（如 GPTQ），改进后的 Hessian 为 H = (X ⊙ √c)(X ⊙ √c)^T = (X ⊙ c) X^T，使高亲和力 token 在计算 sensitivity metrics 时贡献更大。
    3. **集成流程**：MoEQuant 首先用 EBSS 生成专家均衡校准集 D*，然后对每个 MoE 层中每个 expert 的权重矩阵，用 AGQ 改进的量化损失/Hessian 执行标准 GPTQ 或 AWQ 量化。量化采用 per-channel 对称均匀量化：Q(W) = clamp(⌈W/s⌋, q_min, q_max)，W_hat = Q(W)·s。
    4. **性能结果**：4-bit MoEQuant++ 相比 GPTQ 在 Qwen-MoE-14B 上平均分提升 0.59pts（49.59 vs 49.00），在 DeepSeek-MoE-16B 上提升 1.00pts（40.01 vs 39.01），在 Mixtral-8x7B 上提升 2.16pts（55.58 vs 53.42）。HumanEval 上 DeepSeek-MoE-16B 在 4-bit 下提升超 10 个点。3.2x 以上内存节省，1.2x 以上推理加速。
