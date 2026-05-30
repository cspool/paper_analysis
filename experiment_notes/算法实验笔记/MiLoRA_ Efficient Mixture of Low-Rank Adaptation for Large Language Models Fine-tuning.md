## MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MiLoRA：一种结合 MoE 机制与 LoRA 的 PEFT 方法。核心设计：(1) 将每个 LoRA 模块（Q/K/V/O/G/U/D 共 7 个）视为一个 expert，每层只激活一个 LoRA expert（k=1，经 Top-k=3 softmax 概率分布实现）；(2) prompt-aware routing：router 仅在输入 prompt 首次通过 backbone 时计算一次（hidden states → Pooler → Rational Activation → MOE router → Top-k），后续 token 生成全部复用该路由决策；(3) 使用 Rational Activation Functions（有理函数激活，阶数 m=6, n=5）替代固定 ReLU/GeLU，通过 bi-level optimization（DARTS 风格）为每层学习不同激活函数。
  - 实验比较 MiLoRA vs 多组 PEFT baseline：LoRA、AdaLoRA、MOELoRA、DoRA、Parallel-Adapter、Learned-Adapter、P-tuning v2、IAPT、BitFit、(IA)^3、SSP，以及 MiLoRA+DoRA 组合（MiDoRA）。
  - 评估维度：(a) 单任务学习——5 个常识推理（ARC-e, ARC-c, BoolQ, OBQA, PIQA）+ 2 个数学推理（AQuA, GSM8k）准确率；(b) 多任务学习——混合 ARC/BoolQ/OBQA/PIQA 训练后分别评估；(c) 通用指令微调——Alpaca 训练后评估 MT-Bench（GPT-4 score）、MMLU、BBH；(d) 推理效率——GPU 内存占用（MiB）和 tokens/s（tps），beam size=1 和 3；(e) ablation——pooler 类型、激活函数、k 值、λ_lb、可调参数量、不同 backbone。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A40 (48GB)。
  - 训练精度：论文未明确说明（基于 HuggingFace Transformers，推断为 BF16/FP16 mixed-precision）。
  - 解码策略：beam search，beam size=1 和 3（推理效率实验中）。

- 模型是什么。数据集和bench分别是什么。
  - 主模型：LLaMA-2 7B。ablation 扩展至 LLaMA-2 13B 和 Gemma 2B。
  - 数据集：常识推理（ARC-e 2251 训练, ARC-c 1119 训练, OBQA 4957 训练, PIQA 16000 训练, BoolQ 9427 训练），数学推理（AQuA 97467 训练, GSM8k 7473 训练，使用 GPT-3.5 zero-shot CoT 生成的 rationale），指令微调（Alpaca 50k）。
  - Benchmark 评估集：MT-Bench（80 条, GPT-4 score），MMLU（14042 条, acc），BBH（6511 条, acc）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供官方开源代码仓库。Web 搜索到 github.com/sufenlp/MiLoRA 为同名不同论文（NAACL 2025，关于 minor singular components 初始化 LoRA），非本论文代码。
  - 实现框架：HuggingFace Transformers + PEFT 库（"We use the HuggingFace Transformers, PEFT, or the original code repositories for implementing all the methods"）。
  - 算法pipeline 核心计算流程（MiLoRA 在 LLaMA-2 的单层 forward，基于论文 §3.3 和 Figure 1）：
    ```
    阶段一：Prompt 编码（仅执行一次，在生成第一个新 token 前）
    输入: H^l ∈ R^{n_p × d}  （layer l 的输入 hidden states，n_p=prompt token 数）
    
    1. Pooler: h^l = SelfAttnPool(H^l)
       - 初始化 W_sa ∈ R^{d×1}
       - U = H^l·W_sa          [n_p × d] × [d × 1] → [n_p × 1]
       - A = Softmax(U)         [n_p × 1], 沿序列维度归一化
       - h^l = A^T·H^l          [1 × n_p] × [n_p × d] → [1 × d]
       （备选：last-token pooling / average pooling / max pooling）
    
    2. Rational Activation: g^l = Ra(h^l)
       - Ra(x) = Σ_{j=0}^{m} a_j·x^j / (1 + ||Σ_{i=1}^{n} b_i·x^i||)
       - m=6, n=5, a_j 和 b_i 可学习，初始化为逼近 GeLU
       - 每层有独立的 Rational Activation 参数
    
    3. LoRA Router: expert_idx = Top-k(Softmax(g^l · W_r^l))
       - W_r^l ∈ R^{d × N_mod}, N_mod=7 (Q/K/V/O/G/U/D)
       - k=3 → 激活 top-1 expert（选最高概率的 LoRA 模块）
       - 仅在此阶段调用一次
    
    阶段二：Transformer 层标准计算 + 被选中 LoRA 模块
    4. 执行标准 attention/FFN 计算，仅在 expert_idx 对应的模块 m 附加 LoRA：
       x' = x·W_m + x·W_m^A·W_m^B + b_m
       - W_m^A ∈ R^{d1×r}, W_m^B ∈ R^{r×d2}, r=32
       - 若模块 m 未被选中，则 x' = x·W_m + b_m（原始 backbone）
    
    阶段三：后续 Token 生成（所有 auto-regressive 步骤）
    5. 复用步骤 1-3 的路由决策 expert_idx
    6. 仅对被选中的 LoRA 模块执行步骤 4 的 LoRA forward
       - 跳过 Pooler、Rational Activation、Router 计算
       - 每层仅激活 1/7 个 LoRA 模块（~25.2M activated params vs 80.9M tunable params）
    
    Load Balancing（训练时）:
    L_lb = N_mod · Σ_{i=1}^{N_mod} f_i^l · p̂_i^l
    - f_i^l = 被路由到 expert i 的 prompt 比例
    - p̂_i^l = expert i 的平均概率质量
    - λ_lb = 1e-2（加入 cross-entropy loss）
    
    Bi-level Optimization（训练 Rational Activation 参数 Θ vs LoRA 参数 Ω）:
    - inner: Ω* = argmin L(D_train, Ω, Θ)
    - outer: min L(D_val, Ω*, Θ)
    - 交替优化，Ω 用 lr=1e-4，Θ 用 lr=1e-6
    ```
  - 训练超参数：AdamW (lr=1e-4, linear warmup 6% steps + linear decay), max epoch=10, batch size=16~128, max seq len=768, patience=10 (dev perplexity 不降则早停)。
