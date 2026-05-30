## M1__Towards_Scalable_Test-Time_Compute_with_Mamba_Reasoning_Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是M1——一种基于Mamba架构的hybrid线性RNN推理模型，通过多阶段训练pipeline将Transformer的推理能力迁移到Mamba架构。核心训练流程分为三阶段：(1) Stage 1 Distillation：用MambaInLlama框架将Llama3.2-3B-Instruct蒸馏为hybrid Mamba模型。将attention层的Q/K/V/O投影权重初始化为Mamba层的C/B/X/O投影，新增MLP（生成Δ_t）和A参数（dynamic parameter）。与原始MambaInLlama不同，引入两个额外线性层从head_dim*kv_head扩展到head_dim*n_head（因Transformer使用GQA而Mamba不需要KV cache）。蒸馏loss使用reverse KL divergence D_KL(p(·;θ) || p(·;θ_T))，具有mode-seeking特性。训练框架基于Axolotl，使用data packing合并序列至max_len=8192，仅计算assistant输出token的loss。(2) Stage 2 SFT：先用OpenMathInstruct-2做通用数学SFT（两epoch），再用推理数据集（OpenR1-Math-220k + OpenThoughts-114k-math + ServiceNow-AI-R1-Distill + Magpie-Reasoning-250K，共10B tokens）做推理SFT（五epoch），训练长度扩展至24576。(3) Stage 3 Reasoning RL：使用GRPO（Group Relative Policy Optimization）进行RL训练，移除KL penalty项（实验发现不稳定），添加entropy bonus鼓励策略多样性。Batch size=128，PPO batch size=64（µ=2 iterations），每序列生成8个rollout，最大生成长度32k。使用Adam optimizer with LR=1e-6，训练50步后选最高critic reward的checkpoint。GRPO训练集成进VeRL框架，修复了CUDA graph与PyTorch FSDP的兼容性问题。
  实验比较：(a) 数学推理benchmark（AIME25/AIME24/MATH500/AMC23/OlympiadBench）：M1-3B vs DeepSeek-R1-Distill-Qwen-1.5B及Qwen2.5-Math-7B-Instruct/rStar-Math-7B等；(b) 推理速度benchmark（vLLM 0.6.3，H100 GPU）：M1-3B vs Llama-3.2-3B vs DeepSeek-R1-Distill-Qwen-1.5B，变化batch size (8-512)和decoding length（固定prompt=256, decode=4096或固定batch=128, 变化decode length）；(c) Test-time scaling：majority vote accuracy vs sample count（最多64 samples）和generation time budget；(d) 消融实验：各训练阶段后的MATH500/AIME24 accuracy，验证distillation→SFT(MATH)→SFT(Reason)→RL各阶段贡献。

- 硬件平台是什么，配置是什么。
  推理速度benchmark：单张NVIDIA H100 GPU，使用vLLM 0.6.3推理引擎，greedy decoding（ignore_eos=True保证生成到最大长度），warmup两次后平均三次测量。训练硬件论文未明确说明具体GPU型号及数量（来自TogetherAI）。RL训练通过VeRL框架（https://github.com/volcengine/verl）进行，修复了Mamba+FSDP的CUDA graph兼容性问题，使CUDA graph启用时训练速度提升5x。

- 模型是什么。数据集和bench分别是什么。
  模型：M1-3B，基于Llama3.2-3B-Instruct蒸馏。架构为hybrid Mamba：28层total，其中6层为interleaved attention层（~21%），其余为Mamba层。SSM state size=16，SSM groups=3072/16=192。对比模型：(a) DeepSeek-R1-Distill-Qwen-1.5B（transformer推理模型，1.5B参数）；(b) Llama-3.2-3B（同参数transformer baseline，非推理模型）；(c) Qwen2.5-Math-7B-Instruct/rStar-Math-7B/Eurus-2-7B-PRIME/Qwen2.5-7B-SimpleRL（更大模型的参考对比）。训练数据集：(a) 蒸馏阶段——基于Llama3.2-3B的token-level KL divergence（数据集为通用预训练语料，论文未明确说明具体数据）；(b) SFT-MATH阶段——OpenMathInstruct-2；(c) SFT-Reason阶段——OpenR1-Math-220k + OpenThoughts-114k-math + ServiceNow-AI-R1-Distill + Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B（总计10B reasoning tokens）；(d) RL阶段——数学问题训练集（论文未明确说明具体数据集）。总训练token数<50B。Benchmark：(a) MATH500（Hendrycks et al., 2021）；(b) AIME25（MAA, 2025）；(c) AIME24（MAA, 2024）；(d) AMC23（MAA, 2023）；(e) OlympiadBench（He et al., 2024）。评估使用VeRL的evaluation tools，temperature=0.7，max sequence length=32k。Pass@1取64次平均，majority voting重复100次计算。评估prompt统一为"Let's think step by step and output the final answer within \boxed{}"。

- 开源情况。基于开源文档和论文，使用例子解释，算法pipeline，至少具体到伪代码或张量计算。
  已开源。代码：https://github.com/jxiw/M1，模型checkpoint在HuggingFace发布。训练框架基于Axolotl（https://github.com/axolotl-ai-cloud/axolotl，蒸馏和SFT）和VeRL（https://github.com/volcengine/verl，GRPO RL训练）。

  M1训练pipeline核心——三阶段训练算法：

  **Stage 1: 蒸馏（Distillation）**
  ```
  # 基于MambaInLlama (Wang et al., 2024a) 的跨架构蒸馏
  # Teacher: Llama3.2-3B-Instruct (Transformer with GQA)
  # Student: Hybrid Mamba model (28 layers, 6 attention + 22 Mamba)

  # Step 1: 权重初始化（Algorithm 1 - MAMBAINLLAMA）
  For each attention layer to be converted to Mamba:
    # 从Teacher的attention投影初始化Mamba参数
    # QKV投影 → C/B/X投影（Mamba不使用KV cache，需扩展GQA的KV heads）
    W_C = W_Q  # Q投影 → C投影
    W_B = Linear_expand(W_K)  # K投影 → 扩展至full heads → B投影
    W_X = Linear_expand(W_V)  # V投影 → 扩展至full heads → X投影
    W_O = W_O  # 输出投影直接复用

    # 新增Mamba参数（随机初始化）
    A ∈ R^{N×N'}  # dynamic parameter, N=head_dim, N'=expand_dim
    MLP_Δ: R^N → R^{N'}  # 生成采样率Δ

    # 保留MLP层（直接复用Transformer的MLP权重）
    # 保留6个interleaved attention层不变

  # Step 2: Token-level reverse KL蒸馏
  For each training step:
    # 前向传播
    p_teacher = Teacher(input_ids)  # Teacher输出完整概率分布
    p_student = Student(input_ids)  # Student输出完整概率分布

    # Reverse KL divergence（mode-seeking）
    loss = D_KL(p_student || p_teacher)
         = Σ_t Σ_v p_student(t,v) * log(p_student(t,v) / p_teacher(t,v))

    # Data packing: 合并多个序列至max_len=8192
    # Chat template: mask user prompt部分，仅计算assistant token的loss
    loss.backward()

  # Optimizer: AdamW, LR=1e-5, cosine decay, β=(0.9,0.95), weight_decay=0.1
  ```

  **Stage 2: SFT（Supervised Fine-Tuning）**
  ```
  # Sub-stage 2a: Math SFT on OpenMathInstruct-2
  For epoch in range(2):
    For each (question, solution) in OpenMathInstruct-2:
      input = apply_chat_template(question)
      target = solution
      loss = CrossEntropy(Student(input), target)  # 仅计算assistant token
      # Optimizer: 同蒸馏阶段

  # Sub-stage 2b: Reasoning SFT with 10B reasoning tokens
  # 数据集混合: OpenR1-Math-220k + OpenThoughts-114k-math
  #            + ServiceNow-AI-R1-Distill + Magpie-Reasoning-250K
  For epoch in range(5):
    For each (question, reasoning_solution) in mixed_reasoning_data:
      input = apply_chat_template(question)
      max_seq_len = 24576  # 覆盖99%数据
      target = reasoning_solution  # 包含完整chain-of-thought
      loss = CrossEntropy(Student(input), target)
      # Optimizer: AdamW, LR=6e-6 (降低peak LR), 其余同蒸馏
  ```

  **Stage 3: Reasoning RL（GRPO Training）**
  ```
  # 使用VeRL框架 + GRPO loss (modified, 无KL penalty)
  # L_GRPO(θ) = E[π_θ(a|s)/π_θold(a|s) * Â(s,a)] + η·H(π_θ)

  For step in range(50):
    # 1. Rollout generation (batch_size=128)
    For each question in batch:
      # 每个问题生成8个rollout
      prompt = "Let's think step by step and output the final answer within \\boxed{}"
      For g in range(8):
        output_g = model.generate(question + prompt, max_len=32k, temperature=0.7)
        reward_g = critic(output_g, ground_truth)  # 基于答案正确性的reward

    # 2. Advantage computation
    # Â(s,a) = (reward - mean(rewards)) / std(rewards)  # group-relative advantage
    advantages = compute_group_advantages(rewards)  # per-group normalization

    # 3. PPO update (µ=2 iterations, ppo_batch_size=64)
    For iter in range(2):
      For mini_batch in split(rollouts, batch_size=64):
        # Policy gradient with importance sampling
        ratio = π_θ(a|s) / π_θold(a|s)
        loss = ratio * advantages + η * entropy(π_θ)
        loss.backward()
        optimizer.step()  # Adam, LR=1e-6

    # 4. Checkpoint selection
    if reward_critic(current_model) > best_reward:
      save_checkpoint()

  # CUDA graph优化: 修复FSDP+CUDA graph兼容性 → 5x训练加速
  ```

  **M1 Hybrid Mamba单层前向传播（推理时生成一个token）:**
  ```
  Input: x_t ∈ R^D, h_{t-1} ∈ R^{N×N'} (Mamba state)

  # Mamba层（28层中的22层）
  # Step 1: Input projection
  x_proj = Linear_x(x_t)  # ∈ R^N
  z = Linear_z(x_t)       # ∈ R^N, for gating

  # Step 2: 1D convolution + SiLU
  x_conv = CausalConv1d(x_proj)  # kernel=4
  x_act = SiLU(x_conv)

  # Step 3: SSM parameters
  Δ_t = softplus(Linear_Δ(x_act) + bias_Δ)  # ∈ R^N'
  B_t = Linear_B(x_act)  # ∈ R^{N×1}
  C_t = Linear_C(x_act)  # ∈ R^{N×1}

  # Step 4: Discretization + State update
  A_bar, B_bar = discretize(A, B_t, Δ_t)  # Zero-order hold
  h_t = A_bar ⊙ h_{t-1} + B_bar ⊗ x_act  # Selective SSM, O(N×N')
  y_t = C_t^T h_t  # ∈ R^N

  # Step 5: Gating + Output
  y_t = y_t ⊙ SiLU(z)
  output = Linear_O(y_t)  # ∈ R^D

  # Attention层（28层中的6层，interleaved）
  # 标准Multi-Head Attention（保留的Transformer层）
  Q, K, V = Linear_QKV(x_t)
  attn_out = Softmax(Q @ K^T / √d) @ V
  output = Linear_O_attn(attn_out)

  # MLP层（所有层共享MLP设计）
  output = output + SiLU(Linear_gate(x_t)) ⊙ Linear_up(x_t)
  ```

  **性能摘要（Table 1/2）:**
  | Model | AIME25 | AIME24 | MATH500 | AMC23 | OlympiadBench |
  |-------|--------|--------|---------|-------|---------------|
  | DeepSeek-R1-Qwen-1.5B | 23.0 | 28.8 | 82.8 | 62.9 | 43.3 |
  | M1-3B | 23.5 | 28.9 | 82.1 | 62.8 | 47.3 |

  **速度摘要（Figure 1/2, H100+ vLLM 0.6.3）:**
  - Batch=512, decode=4096, prompt=256: M1 3× faster than Llama-3.2-3B
  - Batch=128, 变化decode length: M1始终2×+ faster than Llama-3.2-3B
  - 最优吞吐量: M1 = 15169 T/s vs DeepSeek-R1-Qwen-1.5B = 7263 T/s
