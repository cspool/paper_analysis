## MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

- baseline方法是什么？
  - Baseline 方法包括三种 PTQ 方案：
    - **RTN（Round-to-Nearest）**：直接对权重做 per-channel 对称均匀量化，Q(W) = clamp(⌈W/s⌋, q_min, q_max)，无任何误差补偿。
    - **AWQ（Activation-aware Weight Quantization）**：利用激活分布选择平滑系数和剪枝权重，量化损失为 L(W_hat) = ||WX - W_hat X||_F^2，通过最小化输出误差指导量化。
    - **GPTQ**：基于 OBQ 的 Hessian 误差补偿方法，Hessian = X X^T，逐列量化和补偿误差，是当前最强的 LLM PTQ 方法。
    - 核心缺陷：这些方法均为 layer-wise 量化，忽略了 MoE 架构的两个关键特性：(1) 校准集中不同 expert 负载极不均衡——使用 WikiText2 或 C4 作为校准集时，部分 expert 被大量 token 路由到而其他 expert 收到的 token 极少，导致欠载 expert 校准不足；(2) gating network 为不同 token 分配不同的 expert 亲和力 c_i，但传统量化对所有 token 一视同仁（每个 token 的量化误差贡献相同权重），导致高亲和力 token 的量化误差被低估。
  - 全栈执行例子（以 GPTQ 在 Qwen-MoE-14B 上的 4-bit 量化为例）：
    - **算法层**：从 WikiText2 取 128 条 512-token 序列作为校准集 → 逐层 forward，收集每层每个 expert 的输入激活 X → 对每个 expert 的每个线性层（W_gate, W_up, W_down），计算 Hessian = X X^T → 逐列量化 W 的每一列，用 Hessian 逆矩阵补偿剩余列的量化误差。**问题**：WikiText2 校准集下，某些 expert 可能只收到不到 5% 的 token，Hessian 估计严重不足；gating weight c_i 被完全忽略，所有 token 对 Hessian 的贡献等权。
    - **系统框架层**：论文未明确说明具体推理框架。量化后模型通过标准 PyTorch/HuggingFace Transformers 加载推理。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。量化后使用标准 INT4 dequant + FP16 matmul kernel。
    - **硬件架构层**：NVIDIA A6000 GPU（48GB）。4-bit 量化后内存从 27.88GB 降至 8.51GB（Qwen-MoE-14B），3.28x 节省；解码速度从 8.35 提升至 10.60 tokens/s（1.27x 加速）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **MoEQuant 的核心设计**：通过两个插件式模块 EBSS 和 AGQ，分别解决 inter-expert 和 intra-expert 的不均衡问题，可无缝集成到 GPTQ/AWQ 等现有 PTQ 方法中。
  - **解决 baseline 的两个缺陷**：
    1. **Inter-expert 校准不均衡 → EBSS（Expert-Balanced Self-Sampling）**：不再依赖固定的校准集（如 WikiText2），而是利用 LLM 自身能力自采样生成校准数据。从词汇表 V 开始，维护 w 个 beam，每步用 score(S^t||v) = (-1/(i+1))(R_S + log P(v|S)) + σ(M, S)/τ 对候选 token 排序，保留 top-w。此过程同时优化 perplexity（保证与预训练分布一致）和 expert balance（σ 即 expert 使用频率的 std），将搜索复杂度从 O(m^n) 降至 O(wn)。EBSS 生成的校准集中各 expert 分配到的 token 数基本均衡（参见 Figure 2），确保每个 expert 都有足够的校准样本。
    2. **Intra-expert 亲和力缺失 → AGQ（Affinity-Guided Quantization）**：将 token-expert 亲和力（即 gating coefficient c_i）纳入量化过程。传统量化损失 L = Σ_i ||W x_i - W_hat x_i||_F^2，AGQ 重定义为 L = Σ_i c_i · ||W x_i - W_hat x_i||_F^2，使高亲和力 token 的量化误差惩罚更大。对 Hessian-based 方法，改进 Hessian 为 H = (X ⊙ √c)(X ⊙ √c)^T = (X ⊙ c)X^T，物理含义是 token i 对 Hessian 的贡献按其 gating weight c_i 缩放，使得 router 更信任的 token 在误差补偿时占据更大权重。
  - 全栈执行例子（以 MoEQuant++（基于 GPTQ）在 Qwen-MoE-14B 上的 4-bit 量化为例）：
    - **算法层**：EBSS 以 w=4 branches、τ=1.2、sequence length n=512 自采样生成 expert-balanced 校准集 D* → 将 D* 输入模型，逐层 forward 收集每个 expert 的输入激活 X 和 gating coefficient c → AGQ 计算带亲和力权重的 Hessian H = (X ⊙ c)X^T → 对每个 expert 的每个线性层执行标准 GPTQ 逐列量化+误差补偿，但使用 AGQ 改进的 Hessian。量化参数：per-channel 对称均匀量化，4-bit（q_min=-8, q_max=7）。
    - **系统框架层**：基于 GPTQ 和 AWQ 官方仓库修改，集成 EBSS 校准集生成模块和 AGQ Hessian 计算模块。评估使用 lm-evaluation-harness v0.4.4（zero-shot 任务）和 MMLU/GSM8K/HumanEval 官方仓库（复杂推理任务）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：论文未明确说明。量化后推理使用标准 INT4 矩阵乘 kernel。
    - **硬件架构层**：NVIDIA A6000 GPU。MoEQuant++ 在 Qwen-MoE-14B 上 4-bit 量化后平均分 49.59（vs GPTQ 49.00，+0.59），在 DeepSeek-MoE-16B 上 40.01（vs GPTQ 39.01，+1.00），在 Mixtral-8x7B 上 55.58（vs GPTQ 53.42，+2.16）。HumanEval 上 DeepSeek-MoE-16B 4-bit 下 GPTQ 得分 22.56，MoEQuant++ 提升至 25.00（+10.8%）。instruction-tuned 模型上 MoEQuant++ 效果更显著：Qwen-MoE-14B-Chat 上 HumanEval 从 GPTQ 的 15.24 提升至 21.95（+44%）。
