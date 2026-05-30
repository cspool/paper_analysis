## Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

- 属于算法pipeline的实现是什么？实验比较什么？
  - Hunyuan-Large 提出多项算法 pipeline 创新来训练大规模 MoE 模型（389B 总参数量, 52B 激活参数, 256K 上下文）：
    1. **MoE 混合专家路由策略（Shared + Specialized Experts + Recycle Routing）**：使用 1 个 Shared Expert（被所有 token 消费）和 16 个 Specialized Experts（每个 token 激活 top-1）。提出 Recycle Routing 策略：对传统 top-k 路由中因 capacity overflow 被丢弃的 token，随机重新分配到未超 capacity 的其他 specialized experts，避免关键信息丢失。
    2. **KV Cache 压缩（GQA + CLA）**：联合使用 Grouped-Query Attention（8 组 KV heads）和 Cross-Layer Attention（每 2 层共享 KV cache），将 KV cache 内存开销相比 MHA 减少约 95%（从 4nhdhl 降至 2ngdhl）。
    3. **Expert-Specific Learning Rate Scaling**：不同 expert（shared vs specialized）处理的 token 数不平衡（shared expert 处理所有 token, specialized expert 处理 1/16 的 token），因此 effective batch size 不同。为 shared expert 分配最优学习率 ε_opt(B)，为 specialized experts 按比例 ε_opt(B)/ε_opt(B/n) ≈ 0.31 缩小学习率。
    4. **MoE Scaling Laws**：训练 10M-1B 激活参数的 MoE 模型系列，拟合 N_opt = N_c * C_min^α 和 D_opt = D_c * C_min^β，确定最优激活参数数量（约 58.1B, 选 52B）和最优训练 token 数（约 5.6T, 选 7T）。
    5. **四步合成数据 pipeline**：Instruction Generation → Instruction Evolution → Response Generation → Response Filtering，生成约 1.5T tokens 高质量合成数据（包含数学、代码、低资源语言、高教育价值领域）。
  - 实验比较：
    - Pre-training baselines: LLama3.1-405B, LLama3.1-70B, Mixtral-8x22B, DeepSeek-V2
    - Post-training baselines: LLama3.1-405B-Instruct, LLama3.1-70B-Instruct, Mixtral-8x22B-Instruct, DeepSeek-V2.5-Chat
    - 评估指标：MMLU, MMLU-Pro, BBH, HellaSwag, CommonsenseQA, WinoGrande, PIQA, NaturalQuestions, DROP, ARC-C, TriviaQA, CMMLU, C-Eval, C3, GSM8K, MATH, CMATH, HumanEval, MBPP, AlignBench, MT-Bench, IFEval, Arena-Hard, AlpacaEval-2.0, RULER, LV-Eval, PenguinScrolls

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练硬件配置（未给出 GPU 型号、数量、节点数等具体信息）
  - 推理评估使用与 baseline 一致的配置，具体 GPU 和节点数论文未明确说明

- 模型是什么。数据集和bench分别是什么。
  - **模型**：Hunyuan-Large, Transformer-based MoE。64 layers, 80 attention heads, 8 KV heads (GQA), 1 shared expert, 16 specialized experts (top-1 activated), hidden size 6400, SwiGLU activation, RoPE position embedding, vocabulary 128K tokens, total 389B params, 52B activated params
  - **数据集**：预训练数据共 7T tokens，其中 ~1.5T 为合成数据（数学、代码、低资源语言、高教育价值领域），其余为自然文本语料（中英文为主）；SFT 数据超 100 万条；长上下文预训练用 ~10B tokens × 2 stages（32K → 256K）；论文未公开具体数据集名称
  - **Benchmarks**：MMLU, MMLU-Pro, BBH, HellaSwag, CommonsenseQA, WinoGrande, PIQA, NaturalQuestions, DROP, ARC-C, TriviaQA, CMMLU, C-Eval, C3, GSM8K, MATH, CMATH, HumanEval, MBPP (pre-training)；AlignBench, MT-Bench, IFEval, Arena-Hard, AlpacaEval-2.0, GPQA_diamond, RULER, LV-Eval, PenguinScrolls (post-training + long-context)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - **开源**：代码 https://github.com/Tencent/Tencent-Hunyuan-Large, 模型 https://huggingface.co/tencent/Tencent-Hunyuan-Large
  - **MoE 混合路由算法 pipeline（Recycle Routing 伪代码）**：
    ```
    # 输入: tokens x (B×L×d), router weights W_r (d×E), experts E[0..15], shared expert E_shared
    # 超参数: capacity_factor C, expert_capacity = (B×L / E) * C
    
    # Step 1: All tokens through shared expert
    shared_out = E_shared(x)  # (B×L, d)
    
    # Step 2: Router scores for specialized experts
    scores = softmax(x @ W_r, dim=-1)  # (B×L, 16)
    top1_vals, top1_indices = topk(scores, k=1, dim=-1)  # each token gets top-1 expert
    
    # Step 3: Recycle routing
    expert_counts = count(top1_indices)  # [16]
    expert_out = zeros(B×L, d)
    for token_i in range(B×L):
        expert_id = top1_indices[token_i]
        if expert_counts[expert_id] < expert_capacity[expert_id]:
            expert_counts[expert_id] += 1
            expert_out[token_i] = E[expert_id](x[token_i])
        else:
            # Recycle: randomly assign to any expert under capacity
            available = [e for e in range(16) if expert_counts[e] < expert_capacity[e]]
            if available:
                new_expert = random_choice(available)
                expert_counts[new_expert] += 1
                expert_out[token_i] = E[new_expert](x[token_i])
            # else: truly dropped (rare)
    
    # Step 4: Combine
    output = shared_out + expert_out
    return output
    ```
  - **KV Cache 压缩张量计算**：MHA 原始 KV cache = 4 × n_h × d_h × l bytes (bf16)；GQA 后 = 4 × n_g × d_h × l (8 groups, 80→8 heads 压缩)；CLA 后 = 2 × n_g × d_h × l (每 2 layers 共享, l→l/2)；最终 GQA+CLA = 2 × n_g × d_h × l，相比 MHA = (2 × n_g)/(4 × n_h) = n_g/(2×n_h) = 8/(2×80) = 5% 的 KV cache。
  - **Expert-Specific LR 张量计算**：给定 batch size B，噪声 batch size B_noise，最大学习率 ε_max。shared expert LR = ε_opt(B) = 2ε_max / (sqrt(B_noise/B) + sqrt(B/B_noise))。specialized expert LR = ε_opt(B/16)。ratio ≈ 0.31（代入 B 和 B_noise 计算）。
  - **MoE Scaling Law 计算预算**公式 C ≈ 9.59ND + 2.3×10^8 D（N=激活参数量, D=训练 tokens），结合临界 batch size B_crit(L) 得到最小计算预算 C_min = C / (1 + B/B_crit(L))，拟合 N_opt = 5.9×10^-3 × C_min^0.5305, D_opt = 3.2 × C_min^0.50。
