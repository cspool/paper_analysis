## LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - **LoftQ**：一种量化 + LoRA 联合初始化的框架，通过交替优化（量化与 SVD 低秩近似）来近似原始高精度预训练权重，解决 QLoRA 中量化误差导致 LoRA 初始化偏移的问题。核心流程：给定预训练权重 W，交替执行：(1) 量化当前残差 Q_t = q_N(W - A_{t-1}B_{t-1}^T)，(2) 对量化残差做 SVD 获得 top-r 低秩近似 A_t B_t^T。最终输出量化 backbone Q_T 和 LoRA 适配器初始化 A_T, B_T，满足 W ≈ Q_T + A_T B_T^T。LoftQ 与量化函数无关，支持 NormalFloat (NF2/NF4) 和 Uniform quantization。
  - 实验比较 QLoRA（量化 backbone + 零初始化 LoRA）、Full-precision LoRA（16-bit backbone + LoRA）、Full fine-tuning（全参数微调）作为基线。覆盖 2-bit、4-bit、混合精度（前几层 4-bit + 剩余 2-bit）多种精度级别。评估 encoder-only（DeBERTaV3-base）、encoder-decoder（BART-large）、decoder-only（LLAMA-2-7b/13b）三类模型。

- 硬件平台是什么，配置是什么。
  - NVIDIA A100 GPU（所有训练和推理实验）
  - 量化时间测试：Intel Xeon CPU E5-2650 v4 @ 2.20GHz

- 模型是什么。数据集和bench分别是什么。
  - 模型：DeBERTaV3-base（encoder-only, ~183M）、BART-large（encoder-decoder, ~406M）、LLAMA-2-7b、LLAMA-2-13b
  - 数据集与 Benchmark：
    - NLU: GLUE benchmark（MNLI, QNLI, RTE, SST-2, MRPC, CoLA, QQP, STS-B），SQuADv1.1，ANLI
    - 摘要生成：XSum, CNN/DailyMail（评估 ROUGE-1/2/L）
    - NLG: WikiText-2（评估 Perplexity），GSM8K（评估数学推理 accuracy）
  - LoRA rank：DeBERTaV3 用 rank 16/32，BART 用 rank 8/16，LLAMA-2 用 rank 64

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/yxli2123/LoftQ
  - 模型开源：https://huggingface.co/LoftQ
  - 基于 HuggingFace Transformers + PyTorch 实现

  **LoftQ 核心算法（对单个权重矩阵 W ∈ R^{d1×d2}）：**

  ```
  # 输入: 预训练权重 W, 目标 rank r, N-bit 量化函数 q_N, 交替步数 T
  A_0 = 0  # ∈ R^{d1×r}
  B_0 = 0  # ∈ R^{d2×r}

  for t = 1 to T:
      # Step 1: 量化残差（减去上一步的低秩近似）
      residual = W - A_{t-1} @ B_{t-1}^T   # ∈ R^{d1×d2}
      Q_t = q_N(residual)                    # N-bit 量化, Q_t ∈ R_N^{d1×d2}

      # Step 2: SVD 分解量化残差，取 top-r 分量
      R_t = W - Q_t                         # 量化误差矩阵
      U, Σ, V^T = SVD(R_t)                  # Σ = diag(σ_1, σ_2, ..., σ_d)
      # 取 top-r:
      A_t[:, i] = sqrt(σ_i) * U[:, i]       # i = 1..r
      B_t[:, i] = sqrt(σ_i) * V[:, i]       # i = 1..r

  输出: Q_T (量化 backbone), A_T, B_T (LoRA 适配器初始化)
  ```

  **关键特例 T=1**：Q_1 等于 QLoRA 的量化权重（因 A_0B_0^T=0），A_1B_1^T 是量化残差 W-Q_1 的 top-r SVD。仅 T=1 即可显著减轻量化差异，更多迭代进一步缩小初始化差距。

  **LoRA Fine-tuning 使用方式**：
  - 存储：Q_T 编码为整数矩阵 M（通过公式 X_INT = round((2^N-1) F(X_HP))）和查找表 T
  - 初始化：backbone 用整数矩阵 M（freeze），LoRA 适配器用 A_T, B_T（可训练）
  - 前向：Y = X · dequant(T, M) + X · A_T B_T^T
  - 推理时 adapter 可 merge 回 backbone：W_final = dequant(M) + A_T B_T^T

  **计算成本**：LoftQ 逐权重矩阵独立执行，可并行化。例如 LLAMA-2-13b 单矩阵 (5120×5120, T=5, NF4) 耗时 43s（CPU），总量化时间可接受。

  **关键结果**：
  - DeBERTaV3-base 2-bit Uniform：MNLI-m 88.0%（QLoRA 79.9%, +8.1%），CoLA 60.5（QLoRA N.A.）
  - BART-large 4-bit NF4 rank=8 XSum Rouge-1 44.08（QLoRA 42.91, +1.17）
  - LLAMA-2-7b 2-bit WikiText-2 PPL 7.85（QLoRA N.A.，不收敛）
  - LLAMA-2-13b 2/4-bit 混合精度 GSM8K 38.1%（纯 2-bit QLoRA N.A.）
  - 4-bit 场景接近 Full fine-tuning；T 不敏感（T=1~10 均可），T=1 已有显著增益
