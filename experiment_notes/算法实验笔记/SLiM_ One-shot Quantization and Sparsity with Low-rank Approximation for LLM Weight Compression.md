## SLiM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  SLiM 是一个 one-shot 压缩框架，将硬件友好的量化（SLiM-Quant）、半结构化稀疏（Wanda 2:4 sparsity）和基于显著性的低秩近似（SLiM-LoRA）整合为统一 pipeline。实验比较：（1）zero-shot 下游任务平均准确率（MMLU、Piqa、Arc-Easy、Arc-Challenge、WinoGrande、OpenBookQA）；（2）WikiText2 语言建模困惑度；（3）NVIDIA RTX 3060 和 A100 GPU 上的逐层推理加速比；（4）端到端内存缩减比；（5）floating-point operation（FLOP）缩减比。Baseline 包括：SparseGPT+OPTQ、Wanda+Group AbsMax/AWQ/OmniQuant/AffineQuant、JSQ、L²QER、Magnitude Pruning、MaskLLM。

- 硬件平台是什么，配置是什么。
  NVIDIA RTX 3060（consumer GPU，加速比实验）、NVIDIA A100-40GB（data center GPU，加速比和主要实验）、NVIDIA H100（微调开销实验，单卡）。所有实验运行于 University of Toronto。

- 模型是什么。数据集和bench分别是什么。
  模型：OPT 家族（125M, 350M, 1.3B, 2.7B, 6.7B, 13B）、LLaMA-2（7B, 13B）、LLaMA-3.1-405B（仅加速比）。数据集与 Benchmark：zero-shot 下游任务（MMLU, Piqa, Arc-Easy, Arc-Challenge, WinoGrande, OpenBookQA）使用 Language Model Evaluation Harness（lm-eval-harness）；WikiText2 用于困惑度评估；C4 数据集（128 条序列用于校准，300,000 tokens 用于可选微调）；SlimPajama 作为备选校准数据集。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/Mohammad-Mozaffari/slim

  **SLiM 三阶段 compression pipeline（逐层执行）：**

  **阶段一：SLiM-Quant（概率量化）**
  输入：权重矩阵 W ∈ R^{d_in × d_out}，量化位宽 q（通常 q=4）
  1. 构建权重绝对值直方图 f_abs（bin 数 = max(512, min(d_in*d_out/1000, 20000))）
  2. 多网格搜索最优 scaling factor α：
     - 低分辨率网格：在 [0, max(|W|)] 范围内取 10 个均匀样本，计算每个 α 的 E(α) = E_quant(α) + E_clip(α)
       - E_quant(α) = ∫_0^α f_abs(x) |α × round(x/α) × 2^{1-q} - x|² dx
       - E_clip(α) = ∫_α^∞ f_abs(x) |α - x|² dx
     - 在最低误差 α_low 附近高分辨率细化搜索
  3. 最优 α* = argmin_α E(α)
  4. W^Q = round(clip(W/α*)) × 2^{q-1}
  可选 SLiM-Quant^O（activation-aware）：对 1% 最高显著性的 channel（saliency = |diag(x_mean) × W|），scale up 权重 × s，scale down 对应激活 ÷ s，降低输出误差。

  **阶段二：Sparsification（剪枝）**
  使用 Wanda 在量化权重 W^Q 上施加 2:4 半结构化稀疏或 50% 非结构化稀疏：
  对于每行 weight w_i ∈ R^{d_out}，重要性 score_ij = |w_ij| × ||x_j||_2，保留 score 最高的 50%（2:4 模式：每 4 个连续元素保留 2 个）。

  **阶段三：SLiM-LoRA（显著性低秩适配）**
  1. 计算压缩误差 E_C = W^C - W（其中 W^C = W + E_Q + E_S，E_Q 为量化误差，E_S 为稀疏误差）
  2. 构建 saliency 函数 F(W) = diag(x)W，其中 x ∈ R^{d_in} 为校准集输入的平均绝对值（+ min(|x|) 避免零元素）
  3. 计算误差显著性 S_C = diag(x) × E_C
  4. SVD 分解：S_C = U Σ V^T，取 rank r = 0.1 × d 得到 L̃ = U_r Σ_r^{1/2}, R̃ = Σ_r^{1/2} V_r^T
  5. 逆 saliency 变换：L = diag(1/x) × L̃，R = R̃
  最终近似：W ≈ W^C + LR

  **阶段四（可选）：低秩适配器量化 + PEFT 微调**
  - 对 LR 适配器使用 AbsMax group quantization（group size=128，4-bit）压缩至 4×
  - 冻结量化稀疏权重，仅微调低秩适配器（300K tokens C4，batch size 64，seq len 1024）
  - 量化适配器微调使用 STE（straight-through estimator）+ Triton 自定义量化/反量化 kernel
  - 优化器：AdaFactor + 线性 LR schedule，BF16 精度

  **推理加速路线：**
  - 量化稀疏矩阵乘法使用 Sparse Marlin kernel（集成 vLLM）
  - 低秩适配器使用 Dense Quantized Marlin（量化时）或 PyTorch kernel（全精度时）
  - 小 batch size decode 模式
  - 加速比：RTX 3060 上最高 4.3×，A100 上最高 3.8×（逐层测量）
  - 内存缩减：SLiM^Q 达 0.23×（vs dense），SLiM 达 0.33×（含全精度适配器）
