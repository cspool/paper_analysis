## Dimple Discrete Diffusion Multimodal Large Language Model with Parallel Decoding

- 属于算法pipeline的实现是什么？实验比较什么？
  实现分为两部分：(1) 混合训练范式"Autoregressive-then-Diffusion"：Phase I 使用自回归训练（causal attention mask + next-token prediction loss）进行视觉-语言对齐和指令微调；Phase II 切换为扩散训练（full bidirectional attention mask + masked language modeling loss，仅对answer部分mask）恢复并行解码能力。
  (2) 推理技术：Confident Decoding（基于置信度阈值γ动态选择每步解码token数）、Prefilling（复用prompt的KV cache将注意力复杂度从O((L_prompt+L_answer)²)降至O(L_answer²)）、Structure Prior（通过预置token控制输出结构和长度）。
  实验比较：(i) 13个MLLM benchmark上与LLaVA-1.5-7B、LLaVA-NEXT-7B、Eagle-7B、Qwen-VL-7B、Eagle2-9B、Qwen2.5-VL-7B的性能对比（Table 1）；(ii) 纯扩散训练(DA+DT) vs 自回归后扩散(AA+AT+DT)的消融实验，6种策略比较（Table 2）；(iii) Prefilling对性能和速度的影响（Table 3），batch size=1和32下测量TPS加速比；(iv) Length Bias分析——纯扩散模型对response length敏感度（ChartQA上从42.7%降至8.6%）；(v) Confident Decoding、Structure Prior的定性展示（Table 4-6）。

- 硬件平台是什么，配置是什么。
  NVIDIA H100 GPU集群，训练总计约100 GPU hours。Table 3的Prefilling消融实验在单张H100上进行（batch size=1模拟低GPU利用率，batch size=32模拟高GPU利用率）。

- 模型是什么。数据集和bench分别是什么。
  模型：Dimple-7B。组件：Qwen2.5-VL的vision encoder（ViT）、Dream（从Qwen2.5微调的DLM）作为LLM backbone、随机初始化的两层projector。
  训练数据：Phase I (alignment): LLaVA-CC3M-Pretrain (559k样本)；Phase I (instruction tuning) + Phase II: LLaVA-NEXT Instruction Tuning data (739k样本)。总计约1.3M训练样本，0.8B训练tokens。
  评估库：lmms-eval（https://github.com/EvolvingLMMs-Lab/lmms-eval）。
  Benchmarks：GQA、MMBench_en_test、MME (perception/cognition/total)、POPE、MMMU_val、SQA_img (ScienceQA)、AI2D、ChartQA、TextQA_eval、OCRBench、MathVista_test_mini、MMVet。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/yu-rp/Dimple；模型权重：https://huggingface.co/rp-yu/Dimple-7B；在线体验：https://huggingface.co/spaces/rp-yu/Dimple-7B。

  算法pipeline——训练（Autoregressive-then-Diffusion）：
  1. **初始模型**：Vision Encoder (Qwen2.5-VL ViT, 冻结) + 随机初始化2层MLP projector + Dream DLM (Qwen2.5 fine-tuned)。
  2. **Phase I - Autoregressive Alignment（1 epoch, lr=0.001, batch=256, 数据=LLaVA-CC3M-Pretrain 559k）**：
     a. 将Dream的full bidirectional attention替换为causal attention mask。
     b. 使用标准next-token prediction loss: L_AR = -Σ log p_θ(x_i | x_{<i})，对所有answer tokens计算loss。
     c. 训练projector（vision-language对齐），冻结vision encoder和LLM部分参数。
  3. **Phase I - Autoregressive Instruction Tuning（1 epoch, lr=5e-6, batch=128, 数据=LLaVA-NEXT 739k）**：
     a. 同causal attention，在instruction-following数据上继续训练。
     b. 使用LLaVA-NEXT training recipe和datasets。
  4. **Phase II - Diffusion Tuning（1 epoch, lr=5e-7, batch=128, 数据=LLaVA-NEXT 739k 复用）**：
     a. 恢复Dream的full bidirectional attention mask。
     b. 将[EOS]替换为随机数量的[padding] tokens（公式：n~RandomInteger(n_min, n_max)，其中n_min=l+1，n_max根据l确定）。
     c. Loss: L_D = E_t[(1/t) * E_{q(x_t|x_0)}[-Σ δ_{x_t^n, m} * (x_0^n)^T log f_θ(x_t)^n]]，仅mask answer部分。
     d. 时间步t∈(0,1]随机采样，使用absorbing-state (mask)扩散。

  算法pipeline——推理（Confident Decoding + Prefilling）：
  1. **输入**：图片经过Vision Encoder → visual tokens (projector映射)；文本prompt编码为token序列x。总长度L = L_prompt + L_answer。
  2. **Prefilling（首次forward）**：完整计算attention得到prompt的K/V cache。后续步骤仅需O(L_answer²)的attention计算。
  3. **Confident Decoding迭代（Alg. 1）**：
     输入：当前序列x_t、logits z_t、温度τ、阈值γ、fallback位置数K、masked token数N
     a. 计算pre-revision概率 p_t = softmax(z_t)；post-revision概率 p̃_t = softmax(z_t/τ)。
     b. 对每个masked位置i，计算confidence c_t^(i) = max(p_t^(i))（使用pre-revision概率）；采样候选token x̃_t^(i) ~ Categorical(p̃_t^(i))。
     c. 选择：若∃i满足c_t^(i)≥γ，则更新所有满足条件的位置（batch update）。否则fallback：随机选K个位置更新。
     d. 重复直到所有masked位置被填充或达到最大迭代次数。
  4. **Structure Prior**：在生成前预置特定位置的token（如"Thus, the answer is \box{"），这些位置在解码时标记为已确定，不参与mask/更新。
