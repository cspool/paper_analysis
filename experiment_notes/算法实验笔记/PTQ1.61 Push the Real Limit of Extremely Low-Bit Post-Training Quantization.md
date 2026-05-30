## PTQ1.61 Push the Real Limit of Extremely Low-Bit Post-Training Quantization

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：PTQ1.61 是一种极低位（1.61-bit）后训练量化（PTQ）方法，包含三个核心创新：(1) **一维结构化掩码（Structured Mask）**——基于输入激活的通道级幅值，通过数学推导证明量化误差上界与输入激活通道幅值强相关，提出按通道保留 top-20% 显著权重为 4-bit、其余二值化为 1-bit，掩码仅额外引入 0.0002-bit/权重；(2) **分块缩放因子优化（Block-wise Scaling Factors Optimization）**——将缩放因子设为可学习参数，联合 MSE loss（幅值差距）和负对数余弦相似度 loss（方向偏差）进行分块优化，考虑行间隐式相关性和角度偏差；(3) **量化预处理（Quantization Preprocessing）**——使用轻量级 restorative LoRA（rank=64, 20K steps）在预训练数据集 RedPajama 上微调，将显著权重的分布从散乱模式转化为行集中模式，使模型更适合逐通道 PTQ。
  - 实验比较：PTQ1.61 vs **PB-LLM**（1.7+1 bit，10% 8-bit + 非结构化掩码）vs **BiLLM**（1+1.1 bit，多组二值化 + 非结构化掩码）vs **OmniQuant**（2-bit）vs **AWQ**（2-bit）vs **GPTQ**（2-bit）vs **QuIP**（2-bit），在 LLaMA/LLaMA-2/LLaMA-3/OPT 系列模型上比较 WikiText2/C4 困惑度（PPL）和 8 个推理 benchmark 的零样本准确率。消融实验验证结构化掩码、可学习缩放因子和量化预处理各自贡献。

- 硬件平台是什么，配置是什么。
  - 2 张 Nvidia A800 GPU。PTQ 阶段：校准集来自 WikiText2（128 个随机 2048-token 片段），分块训练 20 epochs，batch size=1。量化预处理阶段：LoRA rank=64，20K steps，单张 A100 GPU 耗时 <1.2 小时。整体 PTQ1.61 在 LLaMA-7B 上总耗时约 2h，GPU 内存 15GB；LLaMA-13B 上约 4.2h，GPU 内存 19GB。使用 lm-evaluation-harness 工具包进行推理评估。

- 模型是什么。数据集和bench分别是什么。
  - **模型**：LLaMA-1（7B, 13B, 30B, 65B）、LLaMA-2（7B, 13B, 70B）、LLaMA-3（8B）、OPT（2.7B, 6.7B, 13B）。所有线性层权重均被量化。
  - **数据集/Benchmark（语言生成）**：WikiText2、C4（困惑度 PPL 评估）。
  - **数据集/Benchmark（推理）**：PIQA、ARC-e、ARC-c、HellaSwag、Winogrande、Race、LAMBADA（使用 lm-evaluation-harness）；MMLU、GSM8K、LongBench（附录）。
  - **数据集（预处理）**：RedPajama（LLaMA 系列的预训练数据集）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码 https://github.com/zjq0455/PTQ1.61
  - PTQ1.61 量化流程（以 LLaMA-7B 单层线性层 W∈R^{4096×4096} 为例）：
    ```
    输入: 预训练权重 W, 校准数据 X (128×2048 tokens)
    阶段1 - 量化预处理（可选）:
      W_preprocessed = W + LoRA(W, RedPajama)  // rank=64, 20K steps
      // 目标: 将显著权重转化为行集中分布模式
    
    阶段2 - 结构化掩码生成:
      对每层计算输入激活 X 的通道幅值 ||x_i||  // x_i ∈ R^n
      选择 top-20% 通道作为显著通道 → mask ∈ {0,1}^{4096×1}  // 一维掩码
      // 仅额外 0.0002-bit/权重
    
    阶段3 - 分块优化量化:
      for each transformer block:
        // 显著通道量化
        W_q[salient] = round(W[salient] / S_q) + Z_q  // 4-bit 量化, Eq.(1)
        // 非显著通道二值化
        W_q[non-salient] = α * sign(W[non-salient])   // 1-bit, Eq.(2)
        // α 为可学习缩放因子, 初始化 α^w = ||w||_1 / n_w
        for epoch = 1 to 20:
          前向: X_q → Block_quant → output_q
          损失: L = ||output_fp - output_q||_2  // MSE
                 + (-log(cos_sim(output_fp, output_q)))  // NLC loss, Eq.(6)
          更新: α = AdamW(L, lr=5e-4 或 1e-3)  // 分块优化, Eq.(7)
        end for
      end for
    输出: 量化权重 W_q (平均 1.61-bit)
    ```
  - 量化误差上界推导（Section 3.2, Eq.(4)）：E = |X(W_q^T - W^T)| ≤ Σ_i (|x_i| * Σ_j |w_{i,j}^q - w_{i,j}|)，证明第 i 通道的量化误差上界与输入激活幅值 |x_i| 和权重行量化误差乘积成正比。由于激活幅值约为权重的 1000 倍（尤其 top-20% 通道），因此保护高激活通道对应的权重行可最大程度降低量化误差上界。
  - 分块优化目标函数（Eq.(7)）：min_{α_s, α_r} E(F(W_q'), X) + E(F(W_q'), X_q)，其中第一分支减轻量化误差传播（量化和全精度 block 输出对比），第二分支量化同一输入对量化前后 block 的输出差异。W_q' 为考虑缩放因子的反量化权重（Eq.(9)）。
  - 推理时内存对比：LLaMA-7B 量化后 PTQ1.61 仅需 1.41GB（vs PB-LLM 2.36GB, BiLLM 1.83GB），因为无需加载额外的非结构化掩码。
