## Mixture of Diverse Size Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoDSE（Mixture of Diverse Size Experts），一种新的 MoE FFN 层结构。与传统 MoE 中所有 expert 尺寸相同不同，MoDSE 在每个 FFN 层内设置不同 hidden dimension 的 expert：大专家（如 4.5× input size）处理高难度 token 预测，小专家（如 0.5× input size）处理低难度 token 预测。专家按对分组 $(i_k^1, i_k^2)$，每对 hidden dimension 之和保持 $2 \times h$（h 为 baseline 的 hidden dimension），保证总参数量与 baseline 一致。提出 expert-pair allocation 策略将每对 expert 放置在同一 GPU 上，保证各 GPU 的参数量均衡。
  - 实验比较：(1) MoDSE vs Baseline（相同尺寸专家的 MoE）在 300M×8 和 700M×8 两个规模下，训练过程中的 cross-entropy loss 曲线和验证 loss；(2) 九个下游 benchmark 上的少样本 in-context learning 评估；(3) MoDSE vs Baseline 105% 参数模型（与 MoDSE 运行时平均 workload 相等）的 loss 对比以消除 workload 差异影响；(4) 推理耗时对比（9 个 benchmark 上的端到端解码时间）；(5) 困难 token（高 CE loss）的路由分布分析。

- 硬件平台是什么，配置是什么。
  - GPU 集群：NVIDIA A800（80GB），每节点 8 GPU，节点内通过 NVLink 和 NVSwitch 互连
  - 300M×8 设置：2 节点（16 GPU）
  - 700M×8 设置：8 节点（64 GPU）
  - 分布式训练框架：ZeRO 优化（论文未明确说明具体 stage）

- 模型是什么。数据集和bench分别是什么。
  - 模型架构：基于 Llama 2 的 decoder-only Transformer，将 dense FFN 替换为 MoE expert 层
  - 300M×8 模型：dim=1536, n_layers=8, #heads=12, #expert=8, top-k=2, h=3840
  - 700M×8 模型：dim=2048, n_layers=12, #heads=32, #expert=8, top-k=2, h=5120
  - MoDSE expert 尺寸对（300M×8）：[(6912,768), (6144,1536), (4608,3072), (3840,3840)]，比例为 (4.5,0.5), (4.0,1.0), (3.0,2.0), (2.5,2.5) 相对于 input dim
  - MoDSE expert 尺寸对（700M×8）：[(9216,1024), (8192,2048), (6144,4096), (5120,5120)]
  - 训练数据：100B tokens，中英双语，来源包括 CommonCrawl、代码、学术论文、书籍、数学、Q&A
  - Tokenizer：BPE（Byte Pair Encoding），中英双语训练
  - Benchmark（9个）：AGIEval（5-shot Acc.）、MMLU（5-shot Acc.）、INTENT（5-shot Acc.）、GSM8K（8-shot EM）、LAMBADA（5-shot EM）、MATH（5-shot EM）、TriviaQA（5-shot EM）、PIQA（5-shot EM）、SIQA（5-shot EM）
  - 优化器：Adam（β1=0.9, β2=0.95, ε=1e-8），weight decay=0.1，gradient clipping=1.0，cosine LR schedule（初始 2e-7，最小 3e-5，warmup 2000 步）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未开源（Limitations 章节说明训练数据和 tokenizer 未开源，计划未来将 MoDSE 设计应用于开源资源）。
  - 算法 pipeline 示例 — MoDSE FFN Layer 前向计算（以 300M×8, top-k=2 为例）：
    ```
    # Input: hidden_states x [batch_size, seq_len, dim=1536]
    # Gate network weights: W_g [dim, num_experts=8], W_n [dim, num_experts=8]
    # Expert pairs with different hidden dims:
    #   pair_0: E_{4.5} (h_0=6912), E_{0.5} (h_1=768)   -> avg=3840
    #   pair_1: E_{4.0} (h_2=6144), E_{1.0} (h_3=1536)  -> avg=3840
    #   pair_2: E_{3.0} (h_4=4608), E_{2.0} (h_5=3072)  -> avg=3840
    #   pair_3: E_{2.5} (h_6=3840), E_{2.5} (h_7=3840)  -> avg=3840
    # Each expert E_i: w1_i [dim, h_i], w2_i [h_i, dim]
    # Total params = sum_i (dim * h_i + h_i * dim) = 2 * dim * sum_i h_i
    #              = 2 * dim * (N * h) = same as baseline

    # Step 1: Gating (same as standard MoE, Switch Transformer style)
    logits = x @ W_g                          # [B, S, 8]
    noise = RMSNorm(Softplus(x @ W_n))        # [B, S, 8]
    H = logits + noise
    probs = Softmax(KeepTopK(H, k=2))         # [B, S, 8]

    # Step 2: Diverse-size expert computation
    output = zeros([B, S, dim])
    for each expert i in {0..7}:
        tokens_i = tokens where expert i is in top-2
        if tokens_i not empty:
            hidden = SiLU(tokens_i @ w1_i)    # [n_tokens_i, h_i] — h_i varies per expert!
            out_i = hidden @ w2_i              # [n_tokens_i, dim]
            output[routed_indices] += probs_i * out_i

    # Step 3: Load balance loss (Switch Transformer auxiliary loss)
    # f_i = fraction of tokens dispatched to expert i
    # P_i = average router probability for expert i
    L_aux = α * N * sum_i(f_i * P_i)
    # Total = L_CE + L_aux, 论文用 α 作为乘数系数（具体值未明确说明）
    ```
  - Expert-pair 加载均衡策略：每对 expert $(\hat{E}_{i_k^1}, \hat{E}_{i_k^2})$ 放置在同一 GPU 上。每个 GPU 分配等量的 expert 对（总参数量一致：每个 pair 的 h_i1 + h_i2 = 2h），确保即使单个 expert 尺寸不同，每个 GPU 上的计算负载（以参数量衡量）均衡。
