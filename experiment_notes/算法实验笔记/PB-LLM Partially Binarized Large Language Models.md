## PB-LLM Partially Binarized Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PB-LLM 提出部分二值化 LLM 权重矩阵的方法，核心包括：(1) **部分二值化权重矩阵**——通过 magnitude 或 Hessian 指标按元素粒度检测 salient weights（显著权重），将小比例（如 5%-30%）salient weights 保留在高位宽（如 8-bit），其余 90%+ 权重二值化为 ±1（乘以 optimal scaling factor）；(2) **PB-GPTQ（PTQ 方法）**——将 GPTQ 的 Hessian 引导迭代量化扩展到部分二值化场景，逐列量化权重矩阵，对 unsalient 权重二值化、salient 权重高比特量化后，计算 Hessian 补偿并应用到剩余列；(3) **QAT 方法**——冻结 salient weights 保持全精度（不参与训练），对 residual binary weights 使用可解析推导的 optimal scaling factor α* = ||w_F||_1 / n（column-wise L1 norm 平均）来最小化 L2 binarization error。
  - 实验比较：(a) 5 种已有 binarization 方法（BNN, XNOR, Bi-Real, ReCU, FDA）直接应用于 OPT-1.3B 的效果（均在随机猜测以下）；(b) PB-GPTQ vs RTN 在不同 salient fraction（50%/20%/10%/5%）下的 C4 perplexity，以及 Magnitude vs Hessian 检测准则对比；(c) PB-GPTQ layer-wise vs group-wise (g=128) 对比；(d) PB-LLM QAT vs LLM-QAT、SmoothQuant、RTN、PB-GPTQ 在 LLaMA-7B 上的 7 个零样本常识推理任务（BoolQ, PIQA, HellaSwag, WinoGrande, ARC-E, ARC-C, OBQA）和 perplexity（C4, WikiText2, PTB）；(e) 训练效率：PB-LLM 仅需 1-10K iterations 恢复性能，而 LLM-QAT 需要 100K iterations。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号。使用 PyTorch 框架进行训练和评估。QAT 训练使用 AdamW 优化器（zero weight decay），batch size=1 per GPU，learning rate=2e-5，cosine learning rate decay。由于计算资源限制，方法论探索主要使用 OPT-1.3B，评估扩展到 LLaMA-7B。

- 模型是什么。数据集和bench分别是什么。
  - 模型：**OPT-1.3B**（方法论探索和消融实验）、**LLaMA-7B**（主要实验结果）。
  - 训练数据：**RedPajama-simple-1B**（RedPajama-1T 的 0.1% 子集，包含 Commoncrawl, C4, GitHub, Wikipedia, Books3, ArXiv, Stackexchange）。
  - 评估数据集/benchmark：(a) 7 个零样本常识推理任务：**BoolQ, PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge, OBQA**（使用 lm-eval-harness）；(b) Perplexity：**WikiText2, C4, PTB**。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：GitHub **https://github.com/hahnyuan/PB-LLM**。
  - 算法 pipeline 核心流程：

  **部分二值化权重矩阵格式**：
  ```
  W ∈ R^{d_o × d_i} (全精度预训练权重)

  # Step 1: Salient Weight Detection
  For PTQ (Hessian 准则):
    H = 2 X X^T  (Hessian 矩阵，X 为 calibration 特征)
    v_i = w_i^2 / [H^{-1}]_{ii}^2  (saliency metric)
    选 saliency top-k% 为 salient weights W^{sal}

  For QAT (Magnitude 准则):
    按 |w_i| 排序，选 top-k% 为 salient weights W^{sal}

  # Step 2: 部分二值化
  W^{unsal} → sign(W^{unsal}) → {-1, +1}
  W^{sal} → 保留高比特 (asymmetric per-channel INT8)

  存储: N_bit ≤ 1 * r_binary + 8 * (1 - r_binary) + 1 (bitmap)
  10% salient → 最多 2.7-bit 等效量化
  ```

  **PB-GPTQ (PTQ) 算法流程**：
  ```
  输入: W ∈ R^{d_o × d_i}, calibration data X, salient fraction k%
  1. 计算 H = 2 X X^T, H^{-1}
  2. 逐列迭代量化:
     For column q in [1, d_i]:
       # 识别当前列 salient/unsalient
       W_q^{sal}, W_q^{unsal} = split_by_saliency(W[:, q])

       # 二值化 unsalient: α_q = mean(|W_q^{unsal}|), Ŵ_q^{unsal} = α_q * sign(W_q^{unsal})
       # 量化 salient: Ŵ_q^{sal} = MinMaxQuant(W_q^{sal}, bit=8)

       # GPTQ Hessian 误差补偿
       δ = (W[:, q] - Ŵ[:, q]) / [H^{-1}]_{qq} * (H^{-1})_{:, q}
       W[:, q+1:] += δ
  3. 输出: partially-binarized 权重矩阵 Ŵ
  ```

  **QAT 训练流程**：
  ```
  1. Salient Weights Frozen:
     W^{sal} = top-k% by |W|  (一次性检测)
     训练全程 freeze W^{sal}，仅更新 W_F^{unsal}

  2. Optimal Scaling Factor (column-wise):
     w̄_B = sign(w_F)
     α* = ||w_F||_1 / n  (闭式解, minimize ||w_F - α w̄_B||_2^2)
     前向: y = W^{sal} x + α* · sign(W_F^{unsal}) x

  3. 反向 (STE): ∂L/∂x = ∂L/∂sign(x) if |x| ≤ 1 else 0

  4. 训练: AdamW, lr=2e-5, cosine decay, 10K iters
  ```

  关键结果：PB-LLM 30% salient (等效 ~3.7 bit) 在 LLaMA-7B 上 Avg 66.9 vs FP 68.7；PB-LLM 10% salient (等效 ~1.7 bit) Avg 60.6。QAT 可大幅恢复 PTQ 性能（PB-GPTQ 10% Avg 36.5 → PB-LLM 10% Avg 60.6）。
