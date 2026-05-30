## QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：QA-LoRA 提出分组量化 + 低秩适应的联合方法，核心在于平衡量化与适应的自由度。具体设计：(1) 对预训练权重 W ∈ R^{D_in × D_out} 的每一列划分为 L 组（组大小 g = D_in / L），每组独立使用缩放因子 α_{l,j} 和零点 β_{l,j} 进行 INT4/INT3/INT2 量化；(2) 输入 x 经过 AvgPool/group-sum 聚合操作 A(x)，将维度从 D_in 降至 L；(3) LoRA 适配器 A ∈ R^{L × D_int}，B ∈ R^{D_int × D_out}，其中 A 的行数与分组数 L 对齐，不再对每行自由优化，而是组内共享；(4) 前向传播：y = W̃^T x + s · A(x)^T · A^T B^T；(5) 微调后通过更新零点矩阵 B' = B - s · (L1 L2) ⊘ A 将 LoRA 权重合并到量化模型中，保持 INT 格式用于推理。此设计实现微调时使用量化权重节省显存和时间，推理时直接使用 INT 格式无需 PTQ。
  - 实验比较：QA-LoRA vs QLoRA vs QLoRA w/ GPTQ vs PEQA，在不同模型规模（LLaMA 7B/13B/33B/65B）、不同量化位宽（INT4/INT3/INT2）、不同微调数据集（Alpaca 52K、FLAN v2 320K、Self-instruct、Longform、Chip2）下比较 MMLU 0-shot/5-shot 准确率、CommonSense QA（HellaSwag、PIQA、WinoGrande、ARC-e、ARC-c、BoolQ、OBQA）0-shot 准确率。消融实验：分组大小（g=32/64/128）、数据集大小（160K-480K）对 MMLU 准确率的影响。

- 硬件平台是什么，配置是什么。
  - GPU: Tesla V100。7B/13B/33B 模型使用 1 块 V100，65B 模型使用 2 块 V100。训练步数 Alpaca 10K、FLAN v2 20K。batch size 16。paged AdamW optimizer，max gradient norm 0.3，constant LR schedule，7B/13B LR=2e-5，33B/65B LR=1e-5。

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA（7B、13B、33B、65B）和 LLaMA2（7B、13B）。
  - 数据集：微调用 Alpaca（52K）、FLAN v2（320K subset）、Self-instruct、Longform、Chip2。
  - Benchmark：MMLU（0-shot 和 5-shot，57 个语言任务含 STEM、Humanities、Social Science、Other），CommonSense QA（HellaSwag、PIQA、WinoGrande、ARC-easy、ARC-challenge、BoolQ、OpenBookQA），使用 lm-eval-harness 评测。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/yuhuixu1993/qa-lora（论文中提供）。
  - 量化方法：采用 GPTQ 进行预训练权重的分组不对称量化（group size=32），act-order=false，true-sequential=true。支持 INT4/INT3/INT2，但方法框架也兼容其他 PTQ 方法（如 AWQ、SPQR）。
  - 核心伪代码（Algorithm 1，PyTorch-like）：
    ```
    # D_in, D_out, D_int: dimensions
    # L: number of quantization groups (D_in // L = group size)
    # s: adaptation coefficient; N: bit width
    
    QA = nn.AvgPool1d(D_in // L)          # 组内平均聚合
    lora_A = nn.Parameter(torch.empty((D_int, L)))
    lora_B = nn.Parameter(torch.empty((D_out, D_int)))
    
    def qalora_forward(x, W, lora_A, lora_B):
        W_tilde = pre_quantization(W, alpha, beta)
        result = x @ W_tilde
        result += (QA(x) * (D_in // L)) @ lora_A.T @ lora_B.T * s
        return result
    
    def pre_quantization(W, alpha, beta):
        # alpha: shape (L, D_out), beta: shape (L, D_out)
        W_hat = torch.round(W / alpha) + beta
        return alpha * (W_hat - beta)
    
    def merge_with_quantization(beta, lora_A, lora_B):
        # 合并 LoRA 到零点矩阵，保持 INT 格式
        beta_new = beta - s * (lora_B @ lora_A).T / alpha
        return beta_new
    ```
  - 张量计算流程：给定输入 x ∈ R^{D_in} 和分组量化权重 W̃ = [α_{l,j} · ⌊(w_{i,j} - β_{l,j}) / α_{l,j}⌉ + β_{l,j}]（其中 l = ⌊i/g⌋ 为组索引），聚合操作 A(x) 对每组内 g 个元素求和输出 L 维向量。forward 计算 y_j = Σ_i x_i · W̃_{i,j} + s · Σ_k (Σ_{r=1}^g x_{(k-1)g+r}) · a_{k,mid} · b_{mid,j}。合并推理时，只需更新 β'_{l,j} = β_{l,j} - s · (Σ_{mid} b_{mid,j} · a_{l,mid}) / α_{l,j}，Ŵ 和 α 不变，模型仍为 INT 格式。
