## SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  提出SliM-LLM，一个基于salience驱动的group-wise混合精度PTQ框架。核心包含：(1) **Salience-Determined Bit Allocation (SBA)**：基于group内平均salience排序，通过双指针搜索最小化输出KL散度来优化bit-width分配。(2) **Salience-Weighted Quantizer Calibration (SQC)**：通过引入calibration参数τ，在三倍标准差规则筛选的salient权重子集上优化加权量化误差，增强对局部重要权重的感知。SliM-LLM以GPTQ为backbone，SliM-LLM⁺以OmniQuant为backbone（仅用SBA，保留learnable weight clipping替代SQC）。实验比较2/3/4-bit weight-only量化下WikiText2和C4 perplexity（per-group size=128），zero-shot任务（PIQA, ARC-e, ARC-c, BoolQ, HellaSwag, Winogrande），以及MMLU、MathQA。对比方法包括RTN、GPTQ、AWQ、QuIP、PB-LLM、OmniQuant、AffineQuant、APTX、LLM-MQ。

- 硬件平台是什么，配置是什么。
  量化在单张NVIDIA A800-80GB GPU上完成（SliM-LLM无梯度，SliM-LLM⁺使用AdamW优化器）。部署测试同样在A800上使用修改版AutoGPTQ进行。量化框架基于GPTQ (Frantar et al., 2022) 和 OmniQuant (Shao et al., 2023)，PyTorch实现。

- 模型是什么。数据集和bench分别是什么。
  模型：OPT (1.3B, 2.7B, 6.7B, 13B, 30B, 66B)，LLaMA-1 (7B, 13B, 30B, 65B)，LLaMA-2 (7B, 13B, 70B)，LLaMA-3 (8B, 70B)，Gemma2-9B，Mixtral 8×7B，Vicuna-13B（对话评估），LLaVA-Next-8B（多模态评估）。校准数据集：从WikiText2随机选取128个样本，每个2048 tokens。评估数据集：WikiText2、C4（perplexity）；PIQA、ARC-e/ARC-c、BoolQ、HellaSwag、Winogrande（zero-shot）；MMLU、MathQA；AI2D、ChartQA、DocVQA、MMBench（VLM评估）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/Aaronhuang-778/SliM-LLM
  
  算法Pipeline（以LLaMA-7B 2-bit量化为例）：
  1. **校准数据采集**：从WikiText2选取128个2048-token样本，前向传播收集每层输入激活x_F。
  2. **Hessian计算**（逐层）：H = (1/P) Σ x_F^[k] x_F^[k]^T，计算Cholesky分解 H^in = Cholesky((H + λI)^(-1))。
  
  3. **SBA (Salience-Determined Bit Allocation)**，对每层权重W ∈ R^{n×m}（group_size=128，共k=m/128个group）：
     ```
     # 计算每个group的平均salience
     for each group g_i (i=0..k-1):
         S[i] = mean(W_g^2 / [diag(H^in)]_g^2)
     # 按salience排序groups
     sort groups by S descending
     # 双指针搜索最优混合精度比例
     for p = 1 to ceil(k/2):
         将p个最低salience的group量化为1-bit，p个最高salience量化为3-bit，其余2-bit
         计算KL_div(xW^T || xŴ_q^T)
         选择KL_div最小的p*作为最优配置
     ```
     约束条件: |G_{N-1}| = |G_{N+1}|（即1-bit和3-bit group数量相等，维持2-bit平均位宽）
  
  4. **SQC (Salience-Weighted Quantizer Calibration)**，对每个group:
     ```
     # 用3-σ规则筛选salient元素
     w_s = {w | w < μ-3σ 或 w > μ+3σ}  # 约占group内1%元素
     w_us = 其余元素
     # 在[1-λ, 1+λ]内搜索最优τ（λ=0.1, n=50 candidates）
     for τ in linearly spaced [0.9, 1.1] (50 steps):
         Δ = τ(w_max - w_min) / (2^b - 1)
         z = -⌊τ w_min / Δ⌋
         ŵ = fakequant(W, b, Δ, z)
         loss = ||w_s - ŵ_s||₂² + ||w_us - ŵ_us||₂²
     选择最小化loss的τ*, Δ*, z*
     ```
  
  5. **GPTQ Error Compensation**（逐列）：
     ŵ_q^b = fakequant(W_{:,b:b+β}, g_b, Δ*, z*)
     E = (W_{:,b:b+β} - ŵ_q^b) / diag(H^in_{b:b+β,b:b+β})
     W_{:,b+β:} = W_{:,b+β:} - E · H^in_{b:b+β,b+β:}

  6. **SliM-LLM⁺变体**：SBA保持不变，量化器部分用OmniQuant的Learnable Weight Clipping (LWC)和Learnable Equivalent Transformation (LET)替代SQC，使用AdamW优化器进行梯度优化。
