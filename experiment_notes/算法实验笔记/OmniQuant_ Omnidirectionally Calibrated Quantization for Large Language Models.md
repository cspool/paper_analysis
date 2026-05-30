## OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：OmniQuant 是一种可微分后训练量化（PTQ）方法，在冻结原始 FP16 权重的前提下，引入少量可学习量化参数（Learnable Weight Clipping 的 γ/β 和 Learnable Equivalent Transformation 的 s/δ/s_a），通过 block-wise 量化误差最小化用 SGD 优化，求得最优量化配置。包含两个核心模块：
    1. **Learnable Weight Clipping (LWC)**：学习权重截断强度 γ ∈ [0,1], β ∈ [0,1]（Eq.2），通过 `h = (γ max(W) - β min(W)) / (2^N - 1)` 动态调整量化步长，将 MinMax 量化推广到可学习版本；γ=1, β=1 时退化为 MinMax。
    2. **Learnable Equivalent Transformation (LET)**：在 attention 和 FFN 层中学习通道级缩放 s 和偏移 δ（Eq.3-5），将激活量化的难度等效迁移到权重量化上；同时将等效变换扩展到 Q/K 的矩阵乘法（Eq.5），使 KV cache 也可被量化。
  - 所有可学习参数量化后可融合进权重，推理时不引入额外计算或参数。
  - 实验比较：
    - Weight-only quantization（W2A16, W3A16, W4A16，含 per-channel 和 group-wise g128/g64 变体）：对比 RTN、GPTQ、AWQ，指标为 WikiText2/PTB/C4 perplexity。
    - Weight-activation quantization（W6A6, W4A4）：对比 SmoothQuant、Outlier Suppression+ (OS+)、RPTQ、LLM-QAT，指标为 PIQA、ARC-e、ARC-c、BoolQ、HellaSwag、Winogrande 零样本任务准确率，以及 WikiText2/C4 perplexity。
    - 指令微调模型（LLaMA-2-chat）W3A16g128：在 Vicuna-Bench 上对比 RTN、AWQ（GPT-4 评估，win rate）。
    - MMLU 零样本评估（Table A16）。
    - 真实设备加速：通过 MLC-LLM 部署在 A100-80G 上，测试 weight memory、running memory 和 token/s 吞吐量。

- 硬件平台是什么，配置是什么。
  - 量化训练：单卡 NVIDIA A100-40G GPU，LLaMA-7B W4A4 约 1.6 小时，LLaMA-65B W4A4 约 14.4 小时
  - 部署测试：单卡 NVIDIA A100-80G GPU，通过 MLC-LLM 评测推理吞吐和显存占用
  - 校准数据：WikiText2 中随机 128 个 2048-token 段落，batch size=1
  - 训练配置：AdamW 优化器（weight decay=0），LWC 学习率 5e-3，LET 学习率 1e-2，默认 20 epochs（W2A16 用 40 epochs）

- 模型是什么。数据集和bench分别是什么。
  - 模型：OPT（125M–66B）、LLaMA-1（7B–65B）、LLaMA-2（7B–70B）、LLaMA-2-chat（7B/13B，指令微调版）、Falcon-180B
  - 校准数据集：WikiText2（128 segments × 2048 tokens）
  - 评估数据集（perplexity）：WikiText2、PTB、C4
  - 评估 benchmark（零样本准确率）：PIQA、ARC-easy、ARC-challenge、BoolQ、HellaSwag、Winogrande、MMLU
  - 生成质量评估：Vicuna-Bench（80 个问题，GPT-4 评分）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 代码开源：https://github.com/OpenGVLab/OmniQuant
  - 部署框架：https://github.com/mlc-ai/mlc-llm
  - 基于 PyTorch + HuggingFace Transformers 实现，lm-eval-harness 用于零样本评估

  **OmniQuant 核心算法（伪代码）：**

  ```
  Input: calibration dataset X, pre-trained LLM model M
  Output: quantized model

  X_fp = X_q = X  # full-precision and quantized model inputs

  for each transformer block B_i in M:  # block-wise calibration
      X_fp = B_i(X_fp)
      init LWC parameters Θ1 = {γ=1, β=1}
      init LET parameters Θ2 = {s=SmoothQuant_init, δ=OS+_init, s_a=1}

      for k in range(epochs):
          for (x_q, x_fp) in (X_q, X_fp):
              # Eq.(3): tilde_X = (X - δ) ⊘ s, tilde_W = s ⊙ W, tilde_B = B + δW
              # Eq.(5): tilde_Q = Q ⊘ s_a, tilde_K^T = s_a ⊙ K^T
              B_i' = LET(B_i, Θ2)

              # Eq.(2): h = (γ*max(W)-β*min(W))/(2^N-1)
              #          W_q = clamp(round(W/h)+z, 0, 2^N-1)
              B_i' = Quantize(B_i', Θ1)

              x_q' = B_i'(x_q)
              loss = ||x_fp - x_q'||^2
              loss.backward()
              update Θ1, Θ2 via AdamW

      # Fuse and finalize
      B_i = LET(B_i, Θ2)  # absorb scaling into weights
      B_i = Quantize(B_i, Θ1)
      X_q = B_i(X_q)

  return quantized model M
  ```

  **LET 等效变换详解**：
  - 线性层（Eq.3）：Y = XW + B = [(X-δ)⊘s] · [s⊙W] + [B+δW]，s, δ ∈ R^{1×Cin} 为通道级 scale/shift
  - Attention（Eq.5）：P = Softmax(Q@K^T) = Softmax((Q⊘s_a) @ (s_a⊙K^T))，s_a ∈ R^{1×Cout}
  - 融合：tilde_X 中的 s, δ 吸收到前一层 LayerNorm/Linear 中；tilde_W 中的 s 和 tilde_Q/tilde_K 中的 s_a 融入原始权重矩阵
  - 应用位置（四对 LET）：[ln1, (q_proj,k_proj,v_proj)]、[v_proj,out_proj]、[Q,K]、[ln2,fc1]（第二层 FFN 除外）

  **LWC vs PACT/LSQ**：
  - PACT 直接学习绝对截断阈值，LSQ 学习绝对 scale/zero-point；当 LET 每轮改变权重分布时两者收敛失败（Table A14, Figure A5）
  - LWC 学习相对截断强度 γ/β ∈ [0,1]，处理权重分布变化时更稳定

  **校准效率**：LLaMA-7B 在 W3A16 下仅需 16 个样本即可收敛（Table A11）；校准数据集切换为 C4/Pile 时 perplexity 波动仅 0.0006-0.17（Table A10）。
