## Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

- 属于算法pipeline的实现是什么？实验比较什么？
  Squat（EdgeQAT）提出了面向边缘设备SLM的粗粒度QAT框架，核心包含三个组件：(1) **Entropy-Guided & Distribution-Aligned Distillation**：熵损失 L_E 最大化量化后 query/key 的信息熵（等价于最小化量化误差），分布损失 L_D 通过余弦相似度对齐量化注意力图与FP16注意力图，解决量化自注意力模块的表征退化；(2) **Token Adaptive Quantization**：基于每个token对初始token的平均注意力分数评估重要性，TopK选择 ρ 比例的重要token分配8-bit、其余分配4-bit，通过TCLM模块实现动态分组+拼接+分别量化；(3) **Adaptive Training Pipeline**：FP16教师模型蒸馏量化学生模型，总损失 L_total = L_distill + r_E·L_E + r_D·L_D。
  实验比较了NIPQ、PACT、LLM-QAT三种QAT baseline，覆盖W8A8、W4A8、W4A4三种位宽配置。在BLiMP零样本评估和(Super)GLUE微调评估上验证精度，在OnePlus 11和Raspberry Pi 5上验证硬件加速。

- 硬件平台是什么，配置是什么。
  - 训练：论文未明确说明训练GPU型号（基于PyTorch框架）。
  - 推理延迟测试：OnePlus 11（Snapdragon 8 Gen 2，全部核心多线程），Raspberry Pi 5（BCM2712四核Arm Cortex A76，四核全用）。延迟基于1000次迭代取平均，输入序列长度128。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-58M（BabyLLaMA架构）、GPT2-97M。
  数据集：预训练数据来自BabyLLaMA工作[46]并经regex清洗，BPE tokenizer（vocab=16000）。
  Benchmark：BLiMP（零样本评估，含BLiMP Main 12个子集+BLiMP Supplement 5个子集）、(Super)GLUE（微调评估，11个子任务：CoLA, SST-2, MRPC, QQP, MNLI, MNLIm, QNLI, RTE, BoolQ, MultiRC, WSC）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/shawnricecake/squant

  **算法pipeline（QAT训练流程）：**

  1. **对称逐层量化前向（Preliminary）**：
     - 权重量化：Q(w) = clip(round(w/α_w), -2^{b_w-1}, 2^{b_w-1}-1)，ŵ = Q(w)·α_w
     - 激活量化：Q(x) = clip(round(x/α_x), -2^{b_x-1}, 2^{b_x-1}-1)，x̂ = Q(x)·α_x
     - 线性层：F_Linear(x, w) = α_x·α_w·[Q(x) × Q(w)]
     - 反向传播使用STE近似梯度

  2. **熵引导优化（Entropy Loss）**：
     - 假设query q ~ N(μ_q, σ_q²)，key k ~ N(μ_k, σ_k²)
     - 熵 H(q) = ½log(2πeσ_q²)，H(k) = ½log(2πeσ_k²)
     - 损失：L_E = -log(Σ_{l=1}^L Σ_{h=1}^H log(1 + σ_q²·σ_k²))
     - 最大化熵等价于最小化量化误差（MOE ≈ MAE for Gaussian）

  3. **分布对齐优化（Distribution Loss）**：
     - L_D = log(Σ_{l=1}^L Σ_{h=1}^H (attn_q · attn_f) / (||attn_q||₂ · ||attn_f||₂))
     - 对齐量化注意力图与FP16注意力图的余弦相似度

  4. **Token自适应量化（Token Adaptive Quantization）**：
     ```
     输入: activations x ∈ R^{N×d}, attention map attn, important ratio ρ
     1. scores = attn[:, 0]  // 每个token对初始token的平均注意力
     2. threshold = TopK(scores, Int(ρ*N))  // Heapsort取第k大
     3. for i = 0 to N-1:
     4.     if scores[i] >= threshold:
     5.         x_8bit.append(x[i])  // 重要token → 8-bit
     6.     else:
     7.         x_4bit.append(x[i])  // 非重要token → 4-bit
     8.   x_q = concat(layer_wise_quant8(x_8bit), layer_wise_quant4(x_4bit))
     9.   output = MKMP_multiplier(x_q, w_q)  // 混合精度MAC
     ```

  5. **蒸馏训练总损失**：
     - L_distill = (1-γ)·L_CE + γ·τ²·L_KL
     - L_total = L_distill + 0.5·L_E + 1.0·L_D

  **关键结果**：LLaMA-58M W4A8 BLiMP avg=69.4%（FP16=69.7%，仅↓0.3%），W4A4 avg=67.8%。GPT2-97M W4A4 BLiMP avg=69.2%（FP16=69.9%）。OnePlus 11上GPT2-97M INT4加速2.26×，Raspberry Pi 5上2.37×。
