## ResQ: Mixed-Precision Quantization of Large Language Models with Low-Rank Residuals

- 属于算法pipeline的实现是什么？实验比较什么？
  ResQ 是一种后训练量化（PTQ）方法，通过 PCA 识别激活中方差最高的低秩子空间（rank r = d/8），将子空间内系数保持 8-bit 高精度，其余量化到 4-bit；并在每个子空间内应用不变随机旋转抑制 outliers，将权重、激活和 KV cache 均量化到 W/A/KV=4/4/4。实验比较 ResQ 与 RTN、GPTQ、SmoothQuant+、QUIK、QuaRot、SpinQuant 在 Wikitext 困惑度、0-shot common sense reasoning（8 任务）、MMLU、GSM8K、LongBench（qmsum/samsum/repobench-p）、MMMU（多模态）上的表现。ResQ 相比 SpinQuant 在 Wikitext 困惑度上降低 4-33%，0-shot 精度提升 0.1-5.4%，无额外训练。

- 硬件平台：单张 NVIDIA A100 80GB GPU 用于量化和评估（Meta-Llama-3-70B 使用 4 张 GPU 评估）；NVIDIA RTX 3090 用于硬件加速测试。

- 模型：Llama 2 (7B, 13B)、Meta-Llama-3 (8B, 70B)、Llama 3.2 (1B, 3B)、Qwen2.5 (0.5B, 3B, 72B)、Qwen2-VL (2B, 7B Instruct)。数据集与 benchmark：Wikitext（困惑度）、ARC-c/e, BoolQ, HellaSwag, OpenBook QA, PIQA, SIQA, WinoGrande（0-shot common sense reasoning）、MMLU（语言理解）、GSM8K 5-shot（数学推理）、samsum/qmsum/repobench-p from LongBench（对话摘要和代码补全）、MMMU（多模态理解）。校准数据使用 Wikitext 512 随机样本获取投影矩阵，GPTQ 使用 128 随机样本。

- 开源情况：代码开源 https://github.com/utkarsh-dmx/project-resq。基于 HuggingFace Transformers + PyTorch 实现，使用 CUDA 11.8 + CUTLASS 进行 INT4/INT8 GEMM 操作，评估使用 lm_evaluation_harness v0.4.5 和 LongBench。

- 算法 pipeline 详解（张量计算级别）：
  给定激活 X∈R^{n×d} 和权重 W∈R^{d×d}：
  1. **PCA 投影矩阵构造**：U = PR，其中 P 由 X 协方差矩阵 XX^T 的特征向量按特征值递增排列组成（后 r 列为高精度子空间 P_h，前 d-r 列为低精度子空间 P_l），R 为随机正交矩阵（Hadamard 或随机旋转）。
  2. **投影与量化**：X_q = Q_L(X·U_l) + Q_H(X·U_h)，W_q = Q_L(U_l^T·W) + Q_H(U_h^T·W)，其中 Q_L 为 4-bit 量化，Q_H 为 8-bit 量化。
  3. **输出计算**：X_q·W_q = Q_L(XU_l)·Q_L(U_l^T·W) + Q_H(XU_h)·Q_H(U_h^T·W)，交叉项因正交性消失。
  4. **推理时投影融合**：U_A 通过右乘 o_proj/down_proj 权重矩阵融入前一层；U_B/U_C 处理注意力块内 KV cache 量化投影；U_D 为 Hadamard 矩阵，通过快速 Hadamard 变换实现；U_C 因 RoPE 存在需运行时显式计算（8-bit 量化）。
  5. **理论保证**：Theorem 4.2 证明 PCA 基选择最小化量化误差上界。

  校准流程：从 Wikitext 采样 512 条校准数据 → 前向传播收集各层激活 X → 对 X 做 PCA 得特征向量 P → 生成随机正交旋转矩阵 R → 构造 U=PR → 将 U_A/U_B/U_C/U_D 融合到权重 → 用 GPTQ 对权重做进一步优化量化。Meta-Llama-3-8B 完整流程在单张 A100 上耗时 35 分钟。
