## mHC Manifold-Constrained Hyper-Connections

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 **Manifold-Constrained Hyper-Connections (mHC)**，将 HC 的残差连接矩阵投影到 Birkhoff polytope（双随机矩阵流形）上，恢复 identity mapping 性质。核心设计：
    - **残差映射约束**：将 HC 的 $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$ 通过 Sinkhorn-Knopp 算法投影为双随机矩阵（行和=列和=1，元素非负），使 $\mathcal{H}_l^{\text{res}} \mathbf{x}_l$ 成为特征的凸组合，保证信号均值的保持和范数的正则化。
    - **输入/输出映射非负约束**：对 $\mathcal{H}_l^{\text{pre}}$ 和 $\mathcal{H}_l^{\text{post}}$ 施加 Sigmoid 函数约束（前者经过 $\sigma(\cdot)$，后者经过 $2\sigma(\cdot)$），防止正负系数组合导致的信号抵消。
    - **参数化**：先 flatten $\mathbf{x}_l \in \mathbb{R}^{n \times C}$ 为 $\vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}$，经 RMSNorm 后通过线性投影 $\varphi_l^{\text{pre}}, \varphi_l^{\text{post}} \in \mathbb{R}^{nC \times n}$ 和 $\varphi_l^{\text{res}} \in \mathbb{R}^{nC \times n^2}$ 获得动态映射，加上可学习 bias 作为静态映射，再进行约束投影得到最终映射矩阵。
    - **Sinkhorn-Knopp 迭代**：从 $\mathbf{M}^{(0)} = \exp(\tilde{\mathcal{H}}_l^{\text{res}})$ 开始，交替行归一化和列归一化，$t_{\text{max}} = 20$ 次迭代得到近似双随机矩阵。
    - **理论保证**：(1) 谱范数 $\|\mathcal{H}_l^{\text{res}}\|_2 \leq 1$，非膨胀；(2) 双随机矩阵的乘法封闭性保证跨层复合映射仍为双随机；(3) Birkhoff polytope 是所有置换矩阵的凸包，残差映射可解释为"置换的凸组合"，反复作用趋向单调增加跨流信息混合。
  - 实验比较：
    - **Baseline**：标准残差连接（Pre-Norm Transformer）
    - **HC**：Hyper-Connections（Zhu et al., 2024），expansion rate n=4
    - **mHC**：本文方法，expansion rate n=4，Sinkhorn-Knopp 20 次迭代
    - 下游 benchmark：BBH (3-shot EM), DROP (3-shot F1), GSM8K (8-shot EM), HellaSwag (10-shot Acc.), MATH (4-shot EM), MMLU (5-shot Acc.), PIQA (0-shot Acc.), TriviaQA (5-shot EM)
    - 缩放实验：Compute Scaling（3B→9B→27B 参数模型）、Token Scaling（3B 模型训练 1T tokens）

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号和集群规模。系统级开销评测指出训练引入了仅 6.7% 的额外时间开销（n=4），暗示使用大规模 GPU 集群进行训练。
  - 利用 DualPipe pipeline parallelism schedule（DeepSeek-V3 技术），涉及 NVLink 和 NIC 通信。

- 模型是什么。数据集和bench分别是什么。
  - 模型：基于 DeepSeek-V3 架构的 MoE 模型，使用 MLA（Multi-Head Latent Attention）、Loss-Free Load Balancing、RMSNorm、RoPE。
    - 3B：12 layers, 1280 dim, 64 routed experts + 2 shared experts, 6 active experts, 16 attention heads
    - 9B：18 layers, 1920 dim, 64 routed experts + 2 shared experts, 6 active experts, 24 attention heads
    - 27B：30 layers, 2560 dim, 72 routed experts + 2 shared experts, 6 active experts, 32 attention heads
  - 数据集：论文未明确说明预训练数据集名称，仅提及按参数比例缩放训练 tokens。
    - 3B: 39.3B tokens, 9B: 105B tokens, 27B: 262B tokens
    - 3B (1T): 1.05T tokens 用于 token scaling 实验
  - Benchmarks: BBH, DROP, GSM8K, HellaSwag, MATH, MMLU, PIQA, TriviaQA

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明是否开源。DeepSeek 系列工作的代码尚未在公开仓库完全发布。
  - mHC 算法 pipeline 的伪代码：
    ```
    # 输入: x_l (n, C) - n-stream residual hidden state at layer l
    # 参数: phi_pre (nC, n), phi_post (nC, n), phi_res (nC, n^2)
    #        b_pre, b_post (1, n), b_res (n, n)
    #        alpha_pre, alpha_post, alpha_res (scalar)
    
    # Step 1: Flatten and normalize
    x_flat = flatten(x_l)  # shape: (1, nC)
    x_norm = RMSNorm(x_flat)  # shape: (1, nC)
    
    # Step 2: Compute raw mappings
    H_pre_raw  = alpha_pre  * (x_norm @ phi_pre)  + b_pre   # (1, n)
    H_post_raw = alpha_post * (x_norm @ phi_post) + b_post  # (1, n)
    H_res_raw  = alpha_res  * reshape(x_norm @ phi_res, (n, n)) + b_res  # (n, n)
    
    # Step 3: Manifold projection
    H_pre  = sigmoid(H_pre_raw)          # (1, n), non-negative
    H_post = 2 * sigmoid(H_post_raw)     # (1, n), non-negative
    H_res  = SinkhornKnopp(H_res_raw)    # (n, n), doubly stochastic
    
    # Sinkhorn-Knopp: H_res_raw -> M_0 = exp(H_res_raw)
    #   for t=1..20: M_t = normalize_rows(normalize_cols(M_{t-1}))
    #   return M_20
    
    # Step 4: Apply mappings
    layer_input  = H_pre @ x_l                    # (C,) - aggregate n streams to 1
    layer_output = F(layer_input, W_l)            # (C,) - standard layer computation
    x_{l+1}      = H_res @ x_l + H_post^T * layer_output  # (n, C) - update stream
    ```
  - **张量计算关键特性**：
    - 当 n=1 时，H_res 退化为标量 1，H_pre=H_post=1，mHC 完全恢复为原始残差连接。
    - 当 n>1 时，H_res 的双随机性保证行和=列和=1，$\|H_res\|_2 \leq 1$，跨层不影响信号范数。
    - 复合映射 $\prod_{i=1}^{L-l} H_{L-i}^{res}$ 仍为双随机矩阵（封闭性），Amax Gain Magnitude 约 1.6（vs HC 的 ~3000）。

## Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出 Sparsely-Gated Mixture-of-Experts (MoE) Layer，核心组件包括：
    - **Noisy Top-K Gating**：对每个输入 x，计算 H(x)_i = (x·W_g)_i + StandardNormal()·Softplus((x·W_noise)_i)，然后 KeepTopK 保留最大的 k 个值（其余设为 -∞），再经 Softmax 得到稀疏门控向量 G(x)。k 个专家中未被选中的专家无需计算 E_i(x)，实现条件计算。
    - **Hierarchical MoE**：两级层次结构。主门控网络 G_primary 选择稀疏加权组合的"次级专家"，每个次级专家本身包含一个次级 MoE（含次级门控网络）。输出 y_H = Σ_i Σ_j G_primary(x)_i · G_i(x)_j · E_{i,j}(x)。第一级分支因子通常等于 GPU 数量，第二级在每个设备内部选择。
    - **Load Balancing**：两个辅助损失——L_importance (CV of expert importance squared) 和 L_load (CV of smooth load estimator squared)，防止门控网络收敛到少数几个专家。
    - **混合数据并行与模型并行**：标准层和门控网络用数据并行，每个 expert 只保留一份共享副本（模型并行），同一组设备同时充当数据并行副本和模型并行分片。专家接收来自所有数据并行输入批次中相关样本的组合批次，batch size 放大 d 倍。
    - **卷积式应用 MoE**：在 stacked LSTM 之间插入 MoE 层，等待前一层对所有时间步完成后再将 MoE 卷积式应用于所有时间步，将时间步维度折叠进 batch 维度，进一步增大 expert 的 batch size。
  - 实验比较：
    - 1 Billion Word LM Benchmark（Chelba et al., 2013）：MoE-4/32/256/256-h/1024-h/4096-h（计算预算约 8M ops/timestep，k=4）vs 计算匹配的 baseline（LSTM-2048-512, 4xLSTM-512, MoE-1-Wide, MoE-1-Deep, MoE-4），以及高计算预算 MoE-34M/143M vs Jozefowicz et al. (2016) 的 best published result。
    - 100 Billion Word Google News Corpus：MoE-32/256-h/1024-h/4096-h/16384-h/65536-h/131072-h（最大 137B 参数）vs 4xLSTM-512 baseline 和 Kneser-Ney 5-gram。
    - WMT'14 En→Fr 和 En→De 机器翻译：MoE-2048（encoder 和 decoder 各一个 MoE 层）vs GNMT (Wu et al., 2016)、PBMT、DeepAtt、LSTM-6-layer。
    - Google Production En→Fr：MoE-2048 vs GNMT。
    - 多语言机器翻译（12 语言对）：MoE-Multi（8.7B 参数）vs GNMT-Mono（278M/model）和 GNMT-Multi（278M）。

- 硬件平台是什么，配置是什么。
  - NVIDIA Tesla K40 GPU 集群（单 GPU 理论峰值 4.29 TFLOPS）
  - 1 Billion Word 实验：16 K40 GPUs；高计算预算模型：32 K40 GPUs
  - 100 Billion Word 实验：32 K40 GPUs（最大两个模型用 64 和 128 K40 GPUs）
  - 机器翻译实验：最多 64 K40 GPUs

- 模型是什么。数据集和bench分别是什么。
  - 模型架构：
    - 语言模型：Word Embedding (512) → LSTM (512) → MoE → LSTM (512) → Softmax。每层后加 dropout 和残差连接。MoE 层内每个 expert 为单隐藏层（1024 ReLU）全连接网络，输入/输出 512 维，约 1M 参数/expert。高计算预算模型用 1024/2048 维 embedding、8192 维 expert hidden layer。
    - 机器翻译模型：基于 GNMT (Wu et al., 2016) 修改版，encoder 3 层 LSTM + 1 层 MoE，decoder 2 层 LSTM + 1 层 MoE。LSTM hidden 2048、output projection 512。各 expert 单隐藏层 2048 ReLU，约 2M 参数/expert。使用 wordpieces（32K 词汇表）、beam search。
  - 数据集：
    - 1 Billion Word Language Modeling Benchmark (Chelba et al., 2013)：~829M words，词汇表 793,471 词
    - 100 Billion Word Google News Corpus：~100B words，来自 Google 内部新闻语料
    - WMT'14 En→Fr：36M 句对训练集，newstest2014 测试集
    - WMT'14 En→De：5M 句对训练集，newstest2014 测试集
    - Google Production En→Fr：Google 内部生产数据
    - 多语言翻译：12 语言对组合数据集（Johnson et al., 2016），约 3B 句对
  - Benchmark 指标：Perplexity（语言建模），BLEU score（机器翻译，multi-bleu.pl）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源状态：原始 TensorFlow 实现是 Google Brain 内部代码，**未作为独立仓库开源**。论文发表于 ICLR 2017。后续 Google 将 MoE 概念集成到 tensor2tensor 库中。社区有多项 PyTorch 复现（如 davidmrau/mixture-of-experts，lucidrains/mixture-of-experts）。
  - 算法 Pipeline 伪代码：
    ```
    # 输入: x, shape [batch_size, seq_len, d_model]
    # 参数: W_g [d_model, n_experts], W_noise [d_model, n_experts]
    # 超参数: k (每个输入选择的专家数)
    
    def MoE_Layer(x):
        # 1. 计算门控信号 H(x)
        clean_logits = x @ W_g                    # [B*T, n]
        noise_std = Softplus(x @ W_noise)         # [B*T, n]
        noise = StandardNormal(clean_logits.shape) * noise_std
        H = clean_logits + noise                  # [B*T, n]
        
        # 2. Noisy Top-K 稀疏化
        top_k_vals, top_k_indices = KeepTopK(H, k)  # [B*T, k]
        # KeepTopK: 保留最大的 k 个值，其余设为 -inf
        G = Softmax(top_k_vals)                    # [B*T, k]，稀疏 gate 权重
        
        # 3. 条件执行 experts
        MoE_out = zeros([B*T, d_model])
        # 将输入按选中的 expert 分组（组 batch）
        for i in range(n_experts):
            mask_i = (top_k_indices contains i)    # 哪些输入选中了 expert i
            if mask_i.any():
                expert_inputs = x[mask_i]           # expert i 的子 batch
                expert_outputs = Expert_i(expert_inputs)  # W_in @ x -> ReLU -> W_out @ x
                gates_for_i = G[mask_i, i]          # 对应 gate 权重
                MoE_out[mask_i] += gates_for_i[:, None] * expert_outputs
        
        return MoE_out
    
    # Hierarchical MoE:
    # G_primary: [B*T] -> top-k1 选择 a 个次级 group
    # G_secondary_i: 仅在 G_primary(x)_i > 0 的样本上计算 -> top-k2
    # output = Σ_i Σ_j G_primary(x)_i * G_i(x)_j * E_{i,j}(x)
    
    # Load Balancing Losses:
    # Importance(X) = Σ_{x∈X} G(x)                    # [n], per-expert gate sum
    # L_importance = w * CV(Importance(X))^2           # CV = σ/μ
    # P(x,i) = Φ((clean_logits_i - kth_excluding(H,k,i)) / noise_std_i)
    # Load(X)_i = Σ_{x∈X} P(x,i)
    # L_load = w * CV(Load(X))^2
    # Total Loss = CrossEntropy + L_importance + L_load
    ```

## Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：OEA（Opportunistic Expert Activation）是一种无需重新训练的 batch-aware MoE 路由算法，用于降低 decode 阶段的延迟。核心是一个两阶段路由策略：
    - **Phase 1（Baseline Expert Selection）**：对 batch 中每个 token，激活其 top-k0 个最优先的专家，保证每个 token 的独立质量基线。
    - **Phase 2（Opportunistic Piggybacking）**：对每个 token，遍历其 top-k0 之后的低优先级专家，若该专家已在 Phase 1 中被其他 token 选入 S_base（即专家权重已被加载到 SRAM），则免费将该专家分配给当前 token（但不超过 k_max 上界），保持激活专家总数 T = |S_base| 不变，从而在不增加延迟的前提下恢复模型性能。
    - 最终路由权重按式 (1) 重归一化：moe(x) = sum_{i in S} (R(x)_i / sum_{j in S} R(x)_j) * E_i(x)。
    - 简化版本：消融实验表明 k_max=k=8、maxP=128（不限制）、p=1.0（固定 top-k0）效果最优，最终简化为 Algorithm 1 —— 仅在 Phase 1 用固定 k0 且 Phase 2 可填入 top-k 中的 S_base 专家。
  - 实验比较：(1) Cross-entropy vs. 平均激活专家数：FineWeb-Edu 子集上扫描 k0、k_max、p、maxP，batch size B ∈ {8,16,32,64}，对比 vanilla top-8 routing、Phase-1-only（pruned）和 OEA 的 Pareto 前沿；(2) 下游 Benchmark：AIME24、MATH500、GPQA、LiveCodeBench，对比 vanilla、pruned（top-k0）和 OEA 的准确率；(3) MoE 层延迟 vs. 激活专家数：测量所有 decode step 和所有 layer 的 (T, latency) 对，验证线性关系；(4) Qwen3-30B-A3B 和 Qwen3-235B-A22B 两个模型规模。

- 硬件平台是什么，配置是什么。
  - Qwen3-30B-A3B：单卡 NVIDIA H100 80GB，bfloat16 精度。
  - Qwen3-235B-A22B：8 张 H100 80GB，单节点 HGX H100，NVSwitch 互联（每 GPU pair 18 条 NVLink），tensor parallelism degree=8。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Qwen3-30B-A3B（48 layers, N=128 experts, k=8 activated per token, embedding dim=2048, expert hidden dim=768, SwiGLU FFN）；Qwen3-235B-A22B（96 layers, embedding dim=4096, expert hidden dim=1536, 同样 top-8/128 routing）。
  - 数据集：FineWeb-Edu 子集（2048 条 sequence，每条 ≥8192 token，用于 cross-entropy 评估）。
  - Benchmark：AIME24（数学竞赛）、MATH500、GPQA（研究生级问答）、LiveCodeBench（代码生成）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供独立开源仓库链接。路由算法集成进 SGLang 框架。
  - 伪代码（简化版 Algorithm 1）：
    ```
    输入: token embeddings x_{1..B}, 每个token的top-k0基线专家数k0,
          排序专家索引 e_{i,j}（按router score降序）
    
    Phase 1: 为每个token i, S_i_base = {e_{i,1}, ..., e_{i,k0}}
    Phase 2: S_base = union_i S_i_base  // 所有必需专家的并集
             对每个token i:
               S_i = S_i_base
               对 j = k0+1 到 N:
                 若 |S_i| >= k: break
                 若 e_{i,j} in S_base:
                   S_i = S_i ∪ {e_{i,j}}
    输出: 最终专家集合 S_1, ..., S_B
    ```
  - 张量计算示例：对 batch 中 B=16 个 token，k0=5，k=8，N=128。
    Phase 1 每 token 选 top-5，S_base 约含 30-40 个不同专家（远小于 128）。
    Phase 2 对每个 token 检查 S_base 中是否有其 6-8 位排名的专家，若有则免费附加。
    最终每 token 仍激活约 8 个专家，但 T ≈ |S_base| ≈ 30-40（而非 vanilla 的 ~48-82）。
    MoE 层输出 = sum_{i in S} softmax(R(x)_S) * E_i(x)，延迟从 b*43 + a*16*8 降至 b*30 + a*16*8（b >> a 时约降 30-50%）。

## No Need to Talk: Asynchronous Mixture of Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：SMALLTALK LM 是一种异步混合语言模型训练方法，核心是一个两阶段 EM 训练流程：（1）**Router 训练阶段**——使用 E 个极小的语言模型（4.4M 参数，仅为 expert 的约 1.3%）作为 router，通过 EM 算法交替优化 router 的负对数似然（仅用 prefix 前 M=256 token）和数据的 hard assignment（根据 router 对 prefix 的 log-likelihood 选择最优 expert），并在 assignment 时使用 balanced assignments 策略（按 min log-likelihood 排序后贪心分配，保证每个 expert 获得等量数据）；（2）**Expert 训练阶段**——训练好的 router 将完整数据集划分为 E 个不相交的子集，每个 expert 在自己的子集上完全独立训练，无需任何梯度同步。推理时，用 router 对输入 prefix 评分，选择得分最高的单个 expert 执行自回归生成，仅激活总参数的 1/E。
  - 实验比较：(1) Perplexity vs. FLOPs：335M 参数模型（4/8/16/32 experts）和 1.3B 参数模型（4/16/32 experts）对比同规模 dense baseline，在相同训练 FLOPs 和数据量下比较 test perplexity；(2) 335M × 32 experts（perplexity 9.07）对比 1.3B dense baseline（9.11），训练 FLOPs 相近但推理 FLOPs 仅 1/3；(3) Downstream 零样本评估：ARC Challenge、ARC Easy、HellaSwag、SciQ、MMLU（56 个子任务）；(4) Router 消融：router 大小（4.4M/64M/110M/335M）、prefix 长度（32-256 token）、对比 TF-IDF+SVD+K-Means 聚类路由（Gururangan et al., 2023）；(5) Expert 专业化分析：每个 expert 在其分配数据上的 perplexity 对比 dense baseline。

- 硬件平台是什么，配置是什么。
  - GPU 训练，具体型号论文未明确说明。根据 Table 2，dense baseline 训练使用 8-128 GPUs（batch size 512-2048），expert 训练每个 expert 用 8 GPUs（batch size 128），router 训练用 1 GPU（batch size 32）。Router 训练 128k steps，expert 训练 256k-512k steps。

- 模型是什么。数据集和bench分别是什么。
  - 模型：基于 Transformer decoder + RoPE 的纯 decoder-only 架构。Expert 有两种规模：335M（hidden=1024, layers=24, heads=16, FFN expansion=4）和 1.3B（hidden=2048, layers=24, heads=16, FFN expansion=4）。Router 默认 4.4M 参数（hidden=96, layers=12, heads=12, FFN expansion=4）。使用 SentencePiece tokenizer（vocab=32000）。训练用 AdamW（β1=0.9, β2=0.99, weight decay=0.1, grad clip=0.1）。Expert 用 linear warmup 3000 steps → cosine decay（peak lr=5e-4），router 用 constant lr=1e-4。序列长度 1024 token，router prefix M=256。
  - 数据集：RedPajama-V2（84 个 Common Crawl 爬取周期）。Benchmark：perplexity（held-out test set）、ARC Challenge、ARC Easy、HellaSwag、SciQ、MMLU（使用 lm-eval-harness 评估）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明开源链接（Apple 出品，arxiv:2410.03529，被 ICLR 2025 接收）。实现细节：router 训练用 PyTorch（EM scheme），expert 训练用 JAX（独立训练），评估用 lm-eval-harness。全部 bfloat16 训练，optimizer state 和操作在 float32。
  - 算法 pipeline 伪代码（来自论文 Algorithm 1）：
    ```
    # Stage 1: Train routers via EM
    X = N new sequences from dataset
    X_{1:E} = random_assignments(X)  # initial random split
    for i = 1 ... T:
        for e = 1 ... E:
            θ^{r,e} ≈ argmin_θ L(X_e; θ^{r,e})  # SGD on NLL (Eq.9)
        X = N new sequences from dataset
        X_{1:E} = balanced_assignments(X, θ^r)
            # 1. For each seq x_{1:M} in X:
            #    compute score_e = log p(x_{1:M} | θ^{r,e}) for all e
            # 2. Sort sequences by -max_e score_e
            # 3. Greedy assign: each expert gets |X|/E seqs
    # Stage 2: Train experts independently
    X = M new sequences (full training data)
    X_{1:E} = balanced_assignments(X, θ^r)
    for e = 1 ... E:
        θ^e ≈ argmin_θ L(X_e; θ^e)  # independent SGD, no sync
    ```
  - 张量计算：Router 对每个序列 x_{1:M} 计算 NLL = -Σ_{s=1}^{M-1} log p(x_{s+1}|x_{1:s}; θ^{r,e})。Assignment 选择 e* = argmax_e log p(x_{1:M}|θ^{r,e})（假设 uniform prior）。推理时仅激活 expert e*，计算 p(x_{M+1:S}|x_{1:M}; θ^{e*})。通信开销：router 训练期间约 100 次 all-gather，每次每节点 <6MB（传输 16-bit loss 值）；expert 训练零通信。

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoExtend 提出了一种为 MoE LLM 扩展新模态（以视觉为例）的方法，通过三个阶段的 pipeline：（1）**Alignment Stage（对齐阶段）**——使用 CLIP 视觉编码器提取图像特征，经过可训练的 MLP 投影层将视觉 token 与文本 token 拼接，仅训练该 MLP，用 LLaVA 1.5-558k 图像-标题对数据做模态对齐；（2）**Extension Stage（扩展阶段）**——构造 Extender 自适应决定哪些 MoE 层需要新增 expert：先用新模态子集微调 router 得到 κ'，再分别将验证集输入原始模型 κ 和微调 router 后的 κ'，统计各层各 expert 被选中次数 R_κ 和 R_κ'，计算每层的专家选择分布差异 d_j = Std(¯r_ij^κ - ¯r_ij^κ')，选 d_j 最大的 ⌊pL⌋ 层（p=0.5）新增 expert，并将新 expert 权重初始化为该层中原有最活跃 expert（被选次数最多）的权重复制；（3）**Fine-tuning Stage（微调阶段）**——冻结所有原有参数，仅训练新增 expert、对应的 router 列参数 v_new 和 Calibration Module（对每个 expert 输出的校正，使用 GELU 两层网络，确保加 expert 后 softmax 概率分布不变），使用 LLaVA 1.5-mix-665k 数据集。
  - 实验比较：（1）Image QA：在 SQA、TextVQA、VQA^V2 上与 LLaVA-1.5（7B/13B）、MoE-LLaVA、BLIP-2、InstructBLIP、Qwen-VL、SPHINX-MoE 等对比；（2）Multimodal Benchmarks：POPE、MM-Vet、MMBench、MMBench-Chinese、MME；（3）Catastrophic Forgetting 评估：在纯文本 benchmark（ARC-e、HellaSwag、PIQA、WinoG、MBPP、MMLU、GSM8K）上对比原始 LLM、LLaVA-1.5、MoExtend-Full、MoE-LLaVA；（4）Ablation：不同专家插入策略（All layer / First-half / Second-half / Interval / First-quarter / Ours）、不同初始化方法（Copy(i)、Zero、Mean）、不同 Calibration 模块结构（Type1/Type2 × addition/multiplication）。

- 硬件平台是什么，配置是什么。
  - GPU: 8× NVIDIA A800-80G
  - 精度: BF16
  - 分布式框架: DeepSpeed stage 2（预训练阶段）、DeepSpeed stage 3（指令微调阶段）
  - 优化器: AdamW，cosine decay lr schedule，warmup ratio=0.03，weight decay=0

- 模型是什么。数据集和bench分别是什么。
  - 模型：Base LLM 为 Mixtral 8x7B（32 层 MoE，每层 8 experts，top-k=2，总参数 46.7B，每 token 激活 12.9B 参数）；Vision Encoder 为 CLIP ViT-L/14@336px；Vision Projection 为两层线性层 + GELU。
  - 数据集：LLaVA 1.5-558k（Alignment 阶段预训练，图像-标题对），LLaVA 1.5-mix-665k（Fine-tuning 阶段指令微调，多模态指令数据）
  - Benchmark（多模态）：ScienceQA-IMG (SQA)、TextVQA (VQA^T)、VQA^V2、POPE、MM-Vet、MMBench (MMB)、MMBench-Chinese (MMB^CN)、MME
  - Benchmark（文本/遗忘评估）：ARC-Easy、HellaSwag、PIQA、Winogrande、MBPP、MMLU、GSM8K，使用 OpenCompass 工具包评估

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/zhongshsh/MoExtend
  - 算法 pipeline 核心执行流程（以视觉模态扩展为例）：
    1. **输入**：图像 I 经 CLIP ViT-L 编码为视觉 token V=[v_i]_{i=1}^P ∈ R^{P×D}，文本标题 c 经 word embedding 投影为文本 token T=[t_i]_{i=1}^N ∈ R^{N×D}，拼接为 x_0 = [T; V] ∈ R^{(N+P)×D}
    2. **阶段 1 - Alignment**：冻结 CLIP 和 MoE LLM 所有参数，仅训练新增 MLP projector，损失为标准的 next-token prediction（与 LLaVA 一致），将视觉特征空间与文本特征空间粗对齐
    3. **阶段 2 - Extension（Extender 决策）**：
       a. 从 LLaVA 1.5-mix-665k 随机抽样 10,000 条作为验证集 S_e，其余为子训练集 S_t
       b. 使所有 MoE 层的 router 可训练，冻结其他参数，用 S_t 微调 1,000 步得到 κ'
       c. 将 S_e 分别输入 κ 和 κ'，统计每层每个 expert 被选中的次数矩阵 R_κ, R_κ' ∈ R^{m×L}
       d. 归一化得到概率分布 ¯R_κ, ¯R_κ'，计算每层分布差异 d_j = Std_{i=1}^m(¯r_ij^κ - ¯r_ij^κ')
       e. 选 d_j 最大的 ⌊0.5L⌋ 层，为每层新增一个 expert FFN_{m+1}
       f. 新 expert 权重初始化：复制该层中 R_κ 统计中最活跃 expert（argmax_i r_ij^κ）的权重
    4. **阶段 3 - Fine-tuning**：冻结所有原有参数，仅训练新增 expert FFN_{m+1}、新 router 列 v_new 和 Calibration module s_c(x)
       - Calibration：MoE(x) = Σ_{j=1}^k s(x)_j · [1 + s_c(x)] · FFN(x)_j，其中 s_c 为 W_1(GELU(W_2(x)))，W_1 零初始化（使 s_c(x)=0 初始无干扰），W_2 正态初始化
       - Router 扩展：W_new = [W; v_new] ∈ R^{D×(m+1)}，v_new 从最活跃专家对应的 router 列复制
    5. **推理**：与原始 MoE 推理流程完全一致，仅 router 在新增 expert 的层从 m 选 k 变为 m+1 选 k，无额外推理开销

## MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoESys 提出了四项训练/推理系统优化策略：（1）**Hierarchical Storage（分级存储）**——将 MoE 模型的 sparse 参数（expert FFN 层）存储在 SSD/CPU 内存、dense 参数（attention 层）存储在 GPU HBM，通过理论公式（GPU-Node/CPU-Node/SSD-Node 内存约束方程）确定各存储层的容量分配；（2）**2D Prefetch Scheduling**——在 NVLink（水平维度）和 PCIe（垂直维度）上同时预取 dense 和 sparse 参数，与当前层的计算/通信重叠，使用类似 LFU 的 CPU cache 机制管理 sparse 参数的缓存命中；（3）**Elastic MoE Training**——根据各 task 的 batch size 动态调整计算节点数量（轻量 task 合并节点，重量 task 增加节点），消除负载不均造成的 "bubble" 空闲；（4）**Embedding Partition in Data Parallelism**——在 data parallelism 下对 embedding table 做 column-wise 切分（按 hidden_size 维度而非 vocab 维度），通过 3 次 AlltoAll 通信替代 AllReduce 同步，减少 memory footprint。
  - 实验比较：（1）large-scale MoE training：对比 DeepSpeed，在不同参数规模（13.9B-207.2B）、不同 GPU 数（8-128）下的 training throughput（tokens/s）和 GPU memory usage；（2）Elastic MoE Training：load imbalanced vs load balanced 配置下的 per-GPU throughput（samples/s），以及 UFO 模型上的 throughput 和 memory 对比 PyTorch v1.10；（3）Embedding Partition：不同 vocab/hidden/expert 配置下对比 baseline non-segment embedding 的 memory usage 和 speed；（4）Cross-wise comparison：各优化策略的 peak memory 和 computation speed 对比。

- 硬件平台是什么，配置是什么。
  - GPU: NVIDIA A100 80GB（training），A100 40GB（部分 inference 实验）。单节点 8 GPU，通过 NVLink 互联；多节点通过 NIC + switch 互联。
  - Storage: HBM (GPU memory)、CPU DRAM、SSD、Intel Optane Persistent Memory（AppDirect 模式，FSDAX namespace，绕过 page cache 和 kernel 做 DAX 直接 load/store）。
  - Framework: PaddlePaddle / PaddleFleetX。

- 模型是什么。数据集和bench分别是什么。
  - 模型：GPT 系列 MoE 模型（参数 13.9B 到 207.2B，attention heads=64, hidden size=4096, vocab size=50304, layers=12, experts=8-128）；UFO（Unified Feature Optimization）视觉模型（12B sparse-gated MoE）；VIMER-UFO 2.0（billion-scale visual model）。
  - 数据集/benchmark：text generation 任务用于 MoE inference 评估；UFO 多任务训练（4 任务，batch sizes 512/256/128/128 模拟不平衡）。
  - 优化器：ADAMW，pure fp16 precision。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - MoESys 基于开源 PaddlePaddle/PaddleFleetX（https://github.com/PaddlePaddle/PaddleFleetX）实现。论文称 MoESys 代码将发布于 PaddlePaddle GitHub，截至搜索未找到独立 MoESys 仓库。
  - 算法 pipeline 核心——2D Prefetch + Hierarchical Storage 的执行流程：
    1. **参数分类**：MoE 模型参数分为 Dense（multi-head attention，始终激活）和 Sparse（expert FFN，选择性激活）。Dense 参数总量 D，Sparse 参数总量 S。
    2. **存储分配**（基于 ADAM optimizer，每个参数需 fp16 param + fp16 grad + fp32 master + fp32 momentum + fp32 variance = 16 bytes）：
       - GPU-Node 存储：全部 dense 参数状态 16D + 激活批次的 sparse 参数 4αS/L ≤ M_GPU × N
       - CPU-Node 缓存：高频 sparse 参数状态 16αS ≤ M_CPU × N
       - SSD-Node 全量：sparse 参数 master+动量+方差 12S ≤ M_SSD × N
       - 其中 α 为 sparse 参数激活概率（0<α<1），L 为 MoE 层数。
    3. **2D Prefetch**：水平维度（NVLink）→ AllGather 预取下一层 dense 参数（Algorithm 1），垂直维度（PCIe）→ 从 CPU cache 或 SSD 预取下一层 sparse 参数（Algorithm 2）。sparse 参数使用 hash table 记录命中频率（hits），CPU cache 满时淘汰最低命中频率且超过 threshold 的参数，使用 moving average 衰减（每 K step，hits × β）。
    4. **并行执行**：GPU 计算当前第 i 层 → 同时 NVLink 预取 dense 第 (i+1) 层参数 + PCIe 预取 sparse 第 (i+1) 层参数 → 下一层参数就绪无缝衔接。
  - Elastic MoE Training 流程：Gate network AlltoAll 收集 expert 选择结果 → 评估各 task workload 估算 → 合并轻量 task（combine nodes，比例 2:2）或拆分重量 task（add nodes，比例 1:1:1:1）→ 重分配 data partition → 同步参数。
  - Embedding Partition：embedding table [V, H] 沿 hidden_size 维度列切分 → 每个 worker 持有 [V, H/N] shard → Forward: AlltoAll 交换 input data → 本地 lookup → AlltoAll 交换结果 → Backward: AlltoAll 交换 gradients → 本地更新。无需 AllReduce。

## MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：将现有的 Speculative Decoding（SD）算法（standalone draft model 和 Eagle speculation head）应用于稀疏 MoE 模型推理，并通过修改 MoE 模型的 `num_experts_per_token`（K）参数研究 MoE 稀疏度 ρ 对 SD 加速效果的影响。核心创新在于（1）理论分析：推导全激活专家数 N(t) 和每专家平均 token 数 Texp(t;ρ)，证明中等 batch size 下所有专家已激活时 SD 验证不会带来额外参数加载开销，且更稀疏的 MoE 延迟 memory-bound→compute-bound 转变；（2）新指标 target efficiency = T_T(B,1)/T_T(B,γ)，用于解耦系统瓶颈与算法优化；（3）基于 roofline model 的 SD speedup 性能建模（Algorithm 1），通过参数拟合预测任意 workload 下的 SD 加速比。
  - 实验比较：（1）不同 batch size 下 MoE SD speedup 趋势（先升后降，验证理论预测）；（2）不同 sparsity ρ（K=1,2,4,8,16）对 SD speedup 的影响；（3）MoE vs dense model 的 target efficiency 和 end-to-end speedup 对比；（4）不同 γ、temperature、dataset 下的 speedup；（5）性能模型拟合 vs GPU 实测的对比。

- 硬件平台是什么，配置是什么。
  - 2xGPU-A, 2xGPU-B, 4xGPU-A, 4xGPU-C（论文对 GPU 型号做了匿名化处理，GPU-A/GPU-B/GPU-C 为不同 ridge point 的 GPU 平台）。

- 模型是什么。数据集和bench分别是什么。
  - Target 模型：Qwen2-57B-A14B-Instruct（sparsity ρ=8/14），Mixtral-8x7B-Instruct-v0.1（ρ=2/8）。
  - Draft 模型：Qwen2-0.5B-Instruct（standalone small model），Eagle speculation head（trained head integrated in target model）。
  - 稀疏度实验：通过修改 Qwen2-57B-A14B-Instruct 的 config.json 中 `num_experts_per_token` 为 K=1,2,4,8,16 来模拟不同 ρ。
  - Dense 对比模型：Opt-30b（target）+ Opt-350m（draft）。
  - 数据集：HumanEval（code generation）和 MT-bench（conversation）。Tokenized prompt 长度：HumanEval 38-391 tokens，MT-bench 5-356 tokens。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供独立开源代码仓库。实验基于开源框架 vLLM（支持 batched SD、cudagraph optimization）。
  - 算法 pipeline 伪代码：
    1. 输入：B 个 requests 的 prompt tokens，MoE target model M_T，draft model M_D
    2. for each decoding round r = 1, 2, ..., R:
       a. Draft: M_D 自回归生成 γ 个 draft tokens，耗时 γ × T_D(B, 1)
       b. Verify: M_T 并行处理 B × γ 个 (prompt + draft) tokens
          - MoE Gate 路由每个 token 到 K 个 expert
          - N(Bγ) = E × (1 - ((E-K)/E)^{Bγ}) 个专家被激活
          - 若 Bγ 足够大使得 N(Bγ) ≈ E（全激活），验证时间 T_T(B, γ) ≈ T_T(B, 1)
          - 否则 T_T(B, γ) > T_T(B, 1)（额外 expert 参数加载开销）
       c. Rejection Sampling: 基于 target/draft logits 比较丢弃错误预测 token
       d. 本轮接受 token 数 S/R = σ × (γ+1)，σ = (1-α^{γ+1})/((1-α)(γ+1))
    3. Speedup = (S/R) / (γ × T_D(B,1)/T_T(B,1) + T_T(B,γ)/T_T(B,1) + T_reject/T_T(B,1))
    4. Target Efficiency = T_T(B,1) / T_T(B,γ) 作为系统瓶颈度量
    5. 稀疏度调整：修改 config.json 中 `num_experts_per_token` → 影响 ρ 和 N(t) 曲线

## MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoEBlaze 提出一种内存高效的 MoE 训练算法，核心创新有三点：
    1. **记忆高效的 token dispatch**：不创建传统的 per-expert materialized token buffer（大小为 L×K×d，在 DeepSeek 规模下约 94GB），而是生成四组轻量级索引数据结构——expert_token_indices（L×K）、expert_token_offsets（E+1）、token_expert_indices（L×K）、token_index_map（L×K）——通过 on-the-fly gather/scatter 直接从原始未重排激活张量进行 expert 计算和结果聚合。
    2. **前向传播**：Token dispatch 仅生成索引数据结构而不分配路由 token 显存；Expert 计算通过 per-expert token list 做 on-the-fly gather；Output aggregation 通过 per-token expert list 做 on-the-fly reduction。
    3. **反向传播**：利用相同的逆向映射索引，避免将 (L,d) 梯度"展开"为 (L×k,d) 路由梯度的中间步骤，通过 scatter 操作直接将输出梯度映射回对应位置。
  - 实验比较：(1) 训练速度（forward+backward 的 speedup vs MegaBlocks），(2) 激活内存消耗（PyTorch saved tensor hooks 追踪的中间激活张量总大小），在 SiLU 和 SwiGLU 两种激活函数下、7 种 MoE 配置（见表 1）下对比。

- 硬件平台是什么，配置是什么。
  - 单张 NVIDIA H100 Tensor Core GPU（80GB HBM）。软件栈：PyTorch 2.0.1 + CUDA 12.1。

- 模型是什么。数据集和bench分别是什么。
  - 模型：MoE 配置共 7 种（表 1）：input dim d ∈ {512, 1024, 2048}，expert 数 E ∈ {4, 8, 16}，top-k K ∈ {1, 2, 4}，batch size B ∈ {16, 32}，seq len L ∈ {512, 1024, 2048}。FFN hidden dim = 4×d。配置模拟常见 LLM 设定（如 DeepSeek 参数规模）。
  - 数据集/bench：论文使用这些配置的合成数据/随机张量进行单层 MoE 的 Sparse-to-Sparse 计算阶段评测，未使用具体 NLP benchmark。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明开源链接。HuggingFace papers 页面（https://huggingface.co/papers/2601.05296）无代码仓库链接。
  - 算法 pipeline 伪代码（核心前向）：
    1. 输入：x ∈ R^{L×d}，gate 权重 W_g ∈ R^{E×d}
    2. topk_experts = TopK(softmax(W_g · x))  // 形状 (L, k)
    3. 构建 expert_token_indices[L×k]：对每个 expert e，顺序记录路由到 e 的 token ID
    4. 构建 expert_token_offsets[E+1]：前缀和，记录每个 expert 的 token 起止位置
    5. 构建 token_expert_indices[L×k]：按 token ID 排列的 expert ID
    6. 构建 token_index_map[L×k]：每个 token 在 expert_token_indices 中的位置
    7. for each expert e_i:
         token_ids = expert_token_indices[offsets[i]:offsets[i+1]]
         x_ei = x[token_ids]  // on-the-fly gather，不 materialize
         h_ei = σ(W1_i · x_ei)  // 仅保存中间结果用于 backward
         y_ei = W2_i · h_ei
    8. for each token j:
         for each routed expert e_k:
             y_j += g_{j,k} · y_ei[token_index_map[j][k]]  // on-the-fly reduction
    9. 输出：y ∈ R^{L×d}
    关键记忆节省：不分配 L×K×d 的 routed token buffer，仅分配 4×L×K 的 int32 索引（vs bf16 激活的数百 GB）。

## MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-Pruner 是一种针对 MoE LLM 的一站式（one-shot）后训练剪枝方法，核心创新在于将 MoE router 的 gating 权重引入剪枝重要性度量：S = |W_ij| * ||X_j * Gate_j||，即对每个输出神经元，剪掉 weight magnitude × 输入激活 norm × router 权重最小的权值。相比 Wanda（S = |W_ij| * ||X_j||），MoE-Pruner 多乘了一个 router 权重项，利用 MoE routing 信息识别 expert 层中不重要的权值。
  - 实验比较：（1）One-shot 剪枝：MoE-Pruner vs SparseGPT vs Wanda，在 Mixtral-8x7B (base/instruct) 和 Mixtral-8x22B (base/instruct) 上以 50% 非结构化稀疏度和 2:4 半结构化稀疏度进行对比，指标为 WikiText perplexity 和 9 个 zero-shot 任务准确率（ARC-c, ARC-e, Boolq, HellaSwag, MMLU, OBQA, PIQA, RTE, WinoGrande）；（2）Expert-wise Knowledge Distillation 恢复：以未剪枝 pretrained model 为 teacher，对剪枝后 student 做逐 expert 的 MSE 蒸馏，评测 zero-shot 准确率恢复；（3）消融：校准样本数量（2-256）和剪枝率（10%-70%）对 perplexity 的影响。

- 硬件平台是什么，配置是什么。
  - 剪枝实验：单张 NVIDIA H100-80GB GPU。
  - 微调/蒸馏实验：2 台服务器，每台 8×NVIDIA H100-80GB GPU（共 16 卡）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral-8x7B、Mixtral-8x7B-Instruct、Mixtral-8x22B、Mixtral-8x22B-Instruct（Jiang et al., 2024）。
  - 校准数据：C4（Raffel et al., 2020），固定 128 条序列用于所有 one-shot 剪枝实验。
  - 评估数据集：WikiText 验证集（perplexity）。
  - Benchmarks：EleutherAI LM Harness（Gao et al., 2023）上的 9 个 zero-shot 任务 — ARC-easy、ARC-challenge、Boolq、HellaSwag、MMLU、OpenBookQA、PIQA、RTE、WinoGrande。
  - 蒸馏训练集：C4 子集，仅需 1000 条训练样本。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供官方代码仓库链接，arXiv 页面和 HuggingFace papers 页面均未找到 GitHub URL。
  - 算法 Pipeline（基于论文 Algorithm 1）：
    1. **初始化**：给定 MoE 模型 M（l 个 MoE layer，每层 n 个 expert），校准数据 X ∈ R^{b×d_col}，目标稀疏度 p%。
    2. **逐层处理**：对每一层 t=1,...,l：
       a. Forward 前一层：X', G ← forward(layer_t, X)，得到当前层的输入激活 X' 和 router 权重 Gate ∈ R^{b×n}。
       b. 对每个 expert e=1,...,n：
          - 初始化 binary pruning mask M ← 1_{d_row × d_col}
          - 计算重要性分数：S ← |W_ij| * ||X_j * Gate_j||（对每个输出神经元 j，Gate_j 是 router 分配给该 expert 的归一化权重广播到所有输入维度，X_j 是输入激活的第 j 列，逐元素乘法后取 L2 norm）
          - 沿 dim=1 对 S 排序，取最不重要的 d_col*p% 个位置
          - M 中对应位置置 0，W ← M ⊙ W（剪枝后的权重为零）
       c. X ← X' 传递给下一层。
    3. **返回**：剪枝后的模型 M'。

## MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-SpeQ 的核心算法创新是将**量化 MoE 模型作为高保真 draft 模型**，与 expert offloading 协同设计的 speculative decoding 方法。具体包括：
    1. **量化 Draft Model**：对 target FP16 MoE 模型用 GPTQ 进行 INT4 对称量化（group size=128），量化后的 draft 模型全量驻留在 GPU VRAM，作为极低开销的"oracle"。量化草稿模型以 90.9% total fidelity（44.1% hard match + 46.8% soft match）预测 target 模型的 expert selection，优于专门训练的 one-layer-ahead predictor（84.7%）。
    2. **Hybrid-Precision 策略**：FP16 保持 gating networks、attention layers、共享 experts 的全精度（router 量化误差会通过 softmax 放大导致错误 routing）；INT4 量化所有 MLP expert 的非共享部分（主体参数），兼顾草稿速度与 routing 保真度。
    3. **Parameter/KV Cache Sharing**：draft 与 target 模型共享 non-expert 参数（embeddings, attention, layer norm）和 KV Cache，draft 在 target 之前生成的高精度 KV cache 上运行，进一步提高预测质量。VRAM 节省 43%（Qwen1.5-MoE: 13.40GB→7.68GB）。
    4. **Speculative Decoding with MoE Target**：draft 模型自回归生成 k 个候选 token → 从 ELB 提取每 token 每层的 expert 预测 → Expert Scheduler 预取 experts → target 模型单次并行 forward 验证 k+1 个 token → 接受匹配前缀 + 在分歧点从 target 分布采样。
  - 实验比较：（1）End-to-end 推理吞吐（TPOT）对比：MoE-SpeQ vs HuggingFace Transformers（with device_map offloading）vs Mixtral-Offloading-SC vs Mixtral-Offloading-SM，在 low-memory 和 high-memory 两种 GPU 内存约束下；（2）Speculative prefetching 策略命中率对比：MoE-SpeQ speculative vs LRU vs LRU(scaled) vs Single Prefetch(sooner/later)，在 16/24/32GB expert cache 容量下；（3）消融实验：Full vs 无异步预取 vs 无 fused kernel vs 两者都无；（4）五数据集上 token 接受率验证：C4, WikiText-2, HumanEval, GSM8K, GPQA。

- 硬件平台是什么，配置是什么。
  - 单卡 NVIDIA A100-40GB GPU（HBM memory），PCIe 4.0 x16（理论双向 32GB/s 聚合带宽）。
  - 24-core Intel Xeon Silver 4310 CPU，256GB RAM。
  - 多级 GPU 内存预算模拟：16GB（RTX 4080 级）、24GB（RTX 4090 级）、32GB（H20 级）、40GB（A100 全量）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Phi-3.5-MoE（41.9B 参数/6.6B 激活，32 MoE layers, 8 experts/layer, top-2, MoE inter. size=6400）、Qwen1.5-MoE-A2.7B（14.3B 参数/2.7B 激活，24 MoE layers, 60 experts/layer, top-4, 1 shared expert, MoE inter. size=1408）、DeepSeek-V2-Lite（15.7B 参数/2.4B 激活，26 MoE layers, 64 experts/layer, top-6, 1 shared expert, MoE inter. size=1408）。
  - 数据集/Benchmark：C4（web-crawled corpus）、WikiText-2-v1（语言建模）、HumanEval（代码生成）、GSM8K（数学推理）、GPQA（多学科问答）。论文以缩写 GK/WT/HE/GP/C4/avg 引用。
  - Draft 模型量化：GPTQ 方法，expert 内所有线性层 symmetric INT4 量化，group size=128。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供开源代码仓库链接。
  - 算法 Pipeline（基于论文 §2.2, §3.2-3.4）：
    1. **Draft 阶段**（与 I/O 重叠）：给定输入 prefix token 序列 X[1:p] → 量化 draft model (INT4, on-GPU) 自回归生成 k 个候选 token t_{p+1},...,t_{p+k}。对每个生成 token t_i 和每层 j，router 记录 (expert_id, confidence_score) → 构建 Expert Lookahead Buffer (ELB): ELB[i][j] = (expert_id, score), shape=k×L。
       - Draft 中每个 token 的 MoE 计算：hidden state h → Router(W_gate * h) → softmax → top-k selection → expert_i FFN computation（使用 fuseMoE CUDA kernel）：h_out = sum(router_score_i * W_down_i * SiLU(W_gate_i * h) ⊙ (W_up_i * h))。
       - 关键：Draft 生成第一 token 的 latency 被 T_{pcie,init}（首个 expert fetch）掩盖，后续 tokens 与 prefetch overlap。
    2. **Expert Scheduler 三阶段预取**（与 draft 并发）：
       - Phase I（locality-aware cache priming）：利用 cache 中已有 experts，通过 ELB 前部条目做本地命中服务。
       - Phase II（adaptive bandwidth-guided prefetch）：对 ELB 中部条目选择性预取高置信度 experts，控制 VRAM 压力。
       - Phase III（activation-driven cache saturation）：Draft 完成后，对 ELB 尾部所有缺失 experts 做 aggressive prefetch，饱和 VRAM cache 以消除 verify 阶段的 I/O stall。
    3. **验证阶段**：拼接 X[1:p+k] → Target FP16 model 单次 forward（computation reordering 将 tokens 按 expert 重排以最大化 cache locality）→ 逐 token 与 draft 序列比对 → 接受匹配前缀 → 分歧处从 target 分布采样 → 回滚 KV cache 和 logits。
    4. **自适应控制**：Speculative Governor 用 Amortization Roofline Model 每步在线计算 argmax_k Θ(k) = k_accept(k) / T_cycle(k)，其中 T_cycle = max(T_draft(k), T_pcie,init) + T_pcie,new(k) + T_verify(k+1)，受离线 SLO 约束上限 k_SLO 限制。
  - 张量计算示例（Mixtral-8x7B 某 expert 层的 W_gate 矩阵）：
    W ∈ R^{d_row × d_col}（如 14336×4096），X ∈ R^{b×d_col}（b 个 token 的 hidden states），Gate ∈ R^{b×n}（router softmax 输出，n=8）。对每个 expert e，Gate[:,e] 广播为 Gate_broadcast ∈ R^{b×d_col}，计算 X_gated = X ⊙ Gate_broadcast，取列范数 ||X_gated_j||，则重要性矩阵 S_{ij} = |W_{ij}| * ||X_gated_j||。按输出神经元（行）比较，每行保留 (1-p%) 重要性最高的权值，其余置零。
  - 扩展：支持 N:M 半结构化稀疏（如 2:4），在每 M 个连续权值中用同一度量比较。本文 Algorithm 1 是非结构化版本，论文描述通过修改 comparison group 即可扩展为结构化剪枝。

## MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-Prism 的 Offline Refactoring Engine，通过以下三步将预训练 MoE 的 monolithic expert 分解为 fine-grained sub-expert，实现无重训练的模型弹性化：
    1. **Neuron Activation Profiler**：在 Wikitext-2-raw-v1 校准数据集上运行模型，从每个 expert 的 SwiGLU FFN 层捕捉中间激活矩阵 M（B×C，B=tokens, C=neurons），利用 FFN 每列计算独立性将"neuron"定义为负责 A 矩阵单列+对应 W_down 行的权重集合。
    2. **Partitioning Optimization Solver**：组合优化问题——寻找将 C 个 neuron 划分到 N 个子 expert 的最优分区 P*，目标是最小化所有 tokens 上被 deactivated sub-experts 的 L1 norm 之和。求解器分两阶段：Greedy Initialization（贪心负载均衡初始分区）+ Simulated Annealing Refinement（T0=100.0, α=0.995, I=100,000 次迭代）。
    3. **Gating Mechanism Reconstructor**：两种策略——(a) Training-Free Proxy Gating：构建 neuron 共激活矩阵 C_co = B^T·B，选择每个 sub-expert 内 centrality 最高的 r=4 个 neuron 作为 gate neurons，用其平均 L1 norm 估算 sub-expert 输出 norm；(b) Low-cost Router Finetuning：仅微调 router（<0.1% 参数），采用 curriculum training 逐步增加 k（8→24/32），在 SlimPajama 的 200K 序列上训练，LR=1e-5。
  - 实验比较：与原始 MoE 模型在同等激活参数量下的 Perplexity（Wikitext）和下游任务准确率（Winogrande 3-shot, ARC-C 5-shot, SciQ 0-shot, BoolQ 0-shot）对比。每个 expert 被划分为 N=4 个子 expert。
- 硬件平台是什么，配置是什么：NVIDIA H800 GPU（训练/评估），PyTorch 2.7.0 + CUDA 12.6。
- 模型是什么。数据集和bench分别是什么：模型为 OLMoE-1B-7B（7B, 64→256 experts, k=8→32）、DeepSeek-V2-Lite（16B, 64→256 experts, k=6→24, 含 2 shared experts）、Qwen3-30B-A3B（30B, 128→512 experts, k=8→32）。校准数据集为 Wikitext-2-raw-v1。微调数据集为 SlimPajama（200K 序列）。评估 benchmark 使用 lm-eval (Eleuther AI)，vLLM 作为推理后端。
- 开源情况：论文未明确说明开源链接。基于论文描述，算法流水线为：(1) 对预训练 MoE 在 Wikitext-2-raw-v1 上前向传播，从 FFN 中间层收集激活矩阵 M_e；(2) 对每个 expert，运行 SA 优化器求解最优分区 P*，输入为 M，输出为 N 个子 expert 的 neuron 索引映射；(3) 从 M 计算共激活矩阵 C_co = B^T·B，B 为 top-k_a 激活二值化矩阵；(4) 对每个子 expert S_n，选择 centrality 最高的 r 个 neuron 作为 gate neurons；(5) 可选：在 SlimPajama 上用 curriculum training 仅微调 linear gate（线性层），冻结其余所有参数；(6) 推理时，gate neurons 对其所属子 expert 的 L1 norm 做代理估算，router 按 softmax 分数选择 top-k 个子 expert，加权求和输出。

## MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-Inference-Bench 是一个综合性 MoE 推理 benchmark 套件，不提出新的算法，而是系统性地评估多种已有算法优化技术在 MoE 模型上的推理性能。评估的算法优化包括：
    1. **FP8 量化**：将 Mixtral-8x7B 从 FP16 量化到 FP8 精度（使用 GPTQ/AWQ 方法），对比不同 batch size 和 sequence length 下的吞吐量。
    2. **Intra-expert 剪枝**：在每个 expert 内部减少 FFN dimension（12.5%/25%/50%），保持 expert 数量不变但降低每个 expert 的计算量。
    3. **Inter-expert 剪枝**：移除整个 expert 及其路由权重（12.5%/25%/50%），保持相同数量的 active experts 但减少总 expert 数量以降低显存占用。
    4. **投机解码（Speculative Decoding）**：使用 Qwen3 系列的小型 draft model（0.6B/1.7B/4B/8B）为 Qwen3-30B-A3B target model 生成候选 token，通过验证-接受机制加速解码。
  - 实验比较：(1) FP16 vs FP8 在不同 batch size（1/16/32/64）和 input/output length（128/256/512/1024/2048）下的吞吐量；(2) OLMoE-1B-7B 和 Qwen1.5-MoE-A2.7B 在不同剪枝比例（12.5%/25%/50%）和 TopK（1 到 baseline）下的吞吐量变化；(3) Qwen3-30B-A3B 搭配四种不同大小 draft model 在不同 input length 和 draft token count 下的投机解码吞吐量；(4) 六种 LLM（Mixtral-8x7B, DeepSeek-V2-Lite, Phi-3.5-MoE, OLMoE-1B-7B, Qwen1.5-MoE-A2.7B, Qwen3-30B-A3B）在九个 lm-eval 任务上的准确率 vs 吞吐量/延迟 trade-off；(5) 三种 DeepSeek-VL2 模型（Tiny/Small/Base）在八个 VLMEvalKit 任务上的准确率 vs 吞吐量/延迟 trade-off。

- 硬件平台是什么，配置是什么。
  - 主要平台：NVIDIA H100 SXM5 80GB GPU（基于 TSMC 4N 工艺，80B 晶体管，80GB HBM3，50MB L2 cache，第四代 Tensor Cores，NVLink）
  - 多 GPU 实验：4× H100 GPUs（用于超参数 scaling 分析、剪枝实验、并行策略评估、Fused MoE 实验）
  - 对比平台：Cerebras CS-3 cloud inference system（WSE-3 wafer-scale engine，FP8 weight storage + FP16 computation）
  - 推理框架：vLLM（所有实验统一使用）

- 模型是什么。数据集和bench分别是什么。
  - LLM 模型（7种）：Mixtral-8x7B（47B total/12.9B active）、Qwen-1.5-MoE-A2.7B（14.3B total/2.7B active）、Qwen3-30B-A3B（30B total/5B active）、DeepSeek-V2-Lite（15.7B total/2.4B active）、Phi-3.5-MoE（41B total/9B active）、OLMoE-1B-7B（7.2B total/1.3B active）、Llama-4-Scout-17B-16E
  - VLM 模型（3种）：DeepSeek-VL2-Tiny（3B total/1B active）、DeepSeek-VL2-Small（16B total/2.8B active）、DeepSeek-VL2（27B total/4.5B active）
  - LLM Benchmark：lm-eval suite — ARC-c, ARC-e, BoolQ, HellaSwag, MMLU, OpenBookQA, RTE, WinoGrande
  - VLM Benchmark：VLMEvalKit — MME, TextVQA, AI2D, DocVQA, MMMU, InfoVQA, RealWorldQA, ScienceQA
  - 推理性能评估：自定义脚本基于 vLLM，通过限制 max output length=1 测量 TTFT，计算 ITL 和 throughput

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未明确说明 benchmark 代码是否开源。所有评估基于开源框架 vLLM（https://github.com/vllm-project/vllm）和开源模型（Mixtral、DeepSeek、Qwen、Phi、OLMoE 均可在 HuggingFace 获取）。
  - 算法 pipeline 示例 — FP8 量化 MoE 推理路径（以 Mixtral-8x7B on H100 + vLLM 为例）：
    ```
    # MoE Layer with FP8 Quantization (pseudocode)
    # Input: hidden_states [batch_size, seq_len, hidden_dim=4096]
    
    # Step 1: Router (kept in FP16 for accuracy)
    router_logits = hidden_states @ W_gate_fp16  # [B, S, num_experts=8]
    topk_weights, topk_indices = topk(softmax(router_logits), k=2)
    
    # Step 2: Expert FFN computation (FP8 quantized weights)
    for expert_id in range(8):
        token_mask = (topk_indices == expert_id)  # tokens routed to this expert
        if token_mask.sum() == 0: continue
        
        expert_input = hidden_states[token_mask]  # [num_tokens, 4096]
        
        # FP8 weight dequantization + INT8 matmul on Tensor Core
        # W_gate_fp8: [4096, 14336] stored as FP8, dequantized on-the-fly
        gate_out = fp8_matmul(expert_input, W_gate_fp8[expert_id])  # [n, 14336]
        gate_out = silu(gate_out)
        
        # W_up_fp8, W_down_fp8 FP8 matmul
        up_out = fp8_matmul(expert_input, W_up_fp8[expert_id])  # [n, 14336]
        expert_out = gate_out * up_out  # element-wise
        expert_out = fp8_matmul(expert_out, W_down_fp8[expert_id])  # [n, 4096]
        
        # Accumulate weighted output
        weight = topk_weights[token_mask].unsqueeze(-1)
        output[token_mask] += weight * expert_out
    
    # Step 3: Combine with residual
    final_output = output + residual
    ```
  - 投机解码 pipeline（Qwen3 系列为例）：
    ```
    # Target: Qwen3-30B-A3B, Draft: Qwen3-1.7B (shared vocabulary)
    
    for each decoding step:
        # Phase 1: Draft model generates k candidate tokens
        draft_tokens = []
        draft_kv_cache = copy(target_kv_cache)
        for i in range(num_draft_tokens):
            draft_logits = draft_model.forward(current_token, draft_kv_cache)
            next_token = sample(draft_logits)
            draft_tokens.append(next_token)
        
        # Phase 2: Target model verifies all draft tokens in parallel
        target_logits = target_model.forward([current_token] + draft_tokens, target_kv_cache)
        
        # Phase 3: Accept/reject with speculative sampling
        accepted_tokens = []
        for i, draft_token in enumerate(draft_tokens):
            p_draft = draft_model_probs[i][draft_token]
            p_target = softmax(target_logits[i])[draft_token]
            if random() < min(1, p_target / p_draft):
                accepted_tokens.append(draft_token)
            else:
                # Reject, sample from adjusted target distribution
                corrected_token = sample(max(0, p_target - p_draft))
                accepted_tokens.append(corrected_token)
                break
        
        # Output accepted tokens and advance
    ```

## MoE-ERAS: Expert Residency Aware Selection

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-ERAS 提出两种 residency-aware 的 expert 选择算法，在 MoE 推理时修改 gating network 的输出，使路由器偏向选择已驻留在 HBM（fast memory）上的 expert，减少 host-to-device 的 expert 加载：
    1. **Thresholding（阈值法）**：对已驻留在 fast memory（HBM）中的 expert，在其 softmax 概率上添加用户定义的超参数 α（Weights_i += α），人工提升 on-chip expert 的激活概率，使得"足够好"的 on-chip expert 能够击败略微更好的 off-chip expert。
    2. **Biasing（偏置法）**：在 softmax 之前，对 off-chip expert 的 logits 施加惩罚 β(1 - freq(E_i))，其中 freq(E_i) 是 expert 在 profiling 阶段收集的归一化激活频率。频率越低的 off-chip expert 惩罚越重——因为冷门 expert 被加载到 HBM 后大概率很快被换出，导致两次 swap。相比 thresholding，biasing 额外考虑了 expert 的热度信息。
  - 实验比较：(1) Top-K routing baseline（含 quantization + LRU caching）vs Thresholding（α=0.05, 0.15, 0.25）vs Biasing（β=1）的解码延迟和 expert swap 次数；(2) 不同 offload per layer 设置下的 speedup；(3) WikiText2-PPL、C4-PPL、MMLU-Acc 的 quality-speedup trade-off。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA H100（用于图 2 的 CPU vs GPU expert read time 对比）
  - 主机内存：CPU DRAM 用于 offload expert 参数
  - Baseline 框架（dvmazur/mixtral-offloading）可在 Tesla T4 16GB 上运行 Mixtral-8x7B
  - 计算精度：论文未明确说明具体精度（baseline 使用 quantization）

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Mixtral-8x7B：32 hidden layers，每层 8 experts，Top-K=2（主要实验模型）
    - Switch Transformer-32E：6 hidden layers，每层 32 experts（仅 profiling）
  - 数据集：
    - CNN DailyMail（profiling expert activation patterns，139k tokens for Mixtral，500k tokens for Switch Transformer）
    - WikiText2（test set，perplexity 评估）
    - C4（validation set，perplexity 评估）
  - Benchmark：
    - MMLU（5-shot accuracy，完整数据集）
    - 解码延迟（wall clock time）、throughput（tokens/sec）、expert swaps saved

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文自身代码未开源。Baseline 基于开源项目 `dvmazur/mixtral-offloading`（https://github.com/dvmazur/mixtral-offloading），该 baseline 已包含 quantization 和 LRU caching。
  - 算法 pipeline 伪代码：
    ```
    # 标准 MoE Gating（Baseline Top-K）
    # H_i: 第 i 层的 self-attention 输出
    Logits = H_i @ W_exp                    # (seq_len, num_experts)
    Weights = Softmax(Logits)               # (seq_len, num_experts)
    Activated = SelectTopK(Weights, k=2)    # 选择 Top-2 experts

    # === MoE-ERAS Thresholding ===
    # residency[e]: True 表示 expert e 当前在 HBM 上
    Weights = Softmax(Logits)
    for e in range(num_experts):
        if residency[e]:  # expert 在 fast mem (HBM)
            Weights[:, e] += alpha           # 添加阈值偏置 α
    Activated = SelectTopK(Weights, k=2)

    # === MoE-ERAS Biasing ===
    # freq[e]: 从 profiling 收集的归一化激活频率
    Logits = H_i @ W_exp
    for e in range(num_experts):
        if not residency[e]:  # expert 在 slow mem (CPU)
            Logits[:, e] -= beta * (1 - freq[e])  # 频率越低惩罚越大
    Weights = Softmax(Logits)
    Activated = SelectTopK(Weights, k=2)
    ```
    关键张量维度：H_i ∈ R^{seq_len × hidden_dim}, W_exp ∈ R^{hidden_dim × num_experts}。MoE-ERAS 在 softmax 前后修改 logits/weights，不改变模型参数，仅在推理时生效。

## MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-GPS 提出两种 expert 预测策略用于 MoE 推理时的动态 expert duplication 实现负载均衡：
    1. **Distribution-Only Prediction**：仅预测 coarse-grained token 分布（各 expert 被激活的比例），不预测具体 token-to-expert 映射。使用 Multinomial Distribution + MLE (Maximum Likelihood Estimation) 对每层 MoE 的 expert 激活概率建模：$\hat{p}_i^l = n_i^l / N$，其中 $n_i^l$ 为训练集第 l 层 expert i 被激活的次数。预测无运行时 overhead（offline 估计），可平衡 FFN compute 但不能减少 All-to-All 通信。
    2. **Token-to-Expert Prediction**：将 expert selection 建模为多分类问题，预测每个 token 的激活 expert。探索三类模型：(a) Probability Model——始终选训练集中频率最高的 expert；(b) Conditional Probability Model——按 token index 或 position index 条件化选择最频繁 expert；(c) Neural Networks——FFN（2 层 MLP, 4096→128→64→8 logits）和 LSTM with Sparse Attention（2-layer LSTM, hidden 64, sparse attention + residual connection）。Token-to-Expert Prediction 可同时平衡 compute 和通信，但有 predictor inference overhead。
    3. 两种策略均配合 Algorithm 1（Expert Duplication 贪心算法）：通过迭代将 overloaded GPU 上的热门 expert 复制到 underloaded GPU，直至所有 GPU token 数差 ≤ 1。
  - 实验比较：(1) Baseline（无 prediction）vs Distribution-Only Prediction vs Token-to-Expert Prediction（多精度点）；(2) 不同 skewness 下的预测准确率（error rate）和系统性能（normalized performance）；(3) Token-to-Expert Prediction 不同 predictor 类型的 accuracy-overhead trade-off（probability model / conditional probability / FFN / LSTM）；(4) 不同 interconnect（NVLink 2TB/s vs PCIe 32GB/s）下的端到端 latency 对比。

- 硬件平台是什么，配置是什么。
  - GPU：4× NVIDIA A100，fully connected via NVLink 3.0（2 TB/s bandwidth）
  - 低带宽配置：PCIe 4.0（32 GB/s bandwidth）
  - 模拟器：LLMCompass [36]（block-level LLM inference simulator, ISCA 2024, validated with silicon measurements）
  - 增强：添加 MoE + Expert Parallelism 支持、Mixtral 架构支持（GQA, SwiGLU, Sliding Window）、Prediction Strategy modeling

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Mixtral 8×7B（主要实验，32 layers, 8 experts/layer, Top-K=2）
    - LLaMA-MoE [37]（Appendix C，cross-validation）
    - Switch Transformer [7]（Appendix C，cross-validation）
  - 数据集：
    - MMLU（skewness=1.39, error rate=1.80%）
    - Alpaca Eval（skewness=1.40, error rate=0.98%）
    - SST2（skewness=1.99, error rate=16.00%）
  - 配置：batch size=1, sequence length=512（prefill stage）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文代码未开源。使用的 LLMCompass 模拟器（https://github.com/PrincetonUniversity/LLMCompass）开源（ISCA 2024）。
  - 算法 pipeline 伪代码：
    ```
    # === Distribution-Only Prediction (Offline) ===
    # 训练阶段：统计各层 expert 激活频率
    for layer l in 1..L:
        for batch in training_data:
            tokens = batch  # seq_len × N batches
            expert_assignments = MoE_Router(tokens, layer=l)  # Top-K routing
            for expert e in 1..E:
                n_e[l] += count(expert_assignments == e)
        # MLE estimation
        p_hat[e][l] = n_e[l] / total_tokens[l]

    # 推理阶段：使用 p_hat 指导 expert duplication
    for layer l in 1..L:
        # 预测各 GPU 应处理的 token 比例
        target_tokens_per_gpu = total_tokens / G  # G = num GPUs
        # 使用 Algorithm 1: Expert Duplication
        P, d = ExpertDuplication(f=token_expert_map, p_hat[:,l],
                                  M=GPU_memory, C_max=max_copies)
        # Scatter tokens（通信：随机分发，未针对 expert 位置优化）
        tokens = AllToAllScatter(tokens, d)
        # FFN compute（计算已均衡）
        for gpu in 1..G:
            output[gpu] = FFN_Experts(tokens[gpu], P[gpu])

    # === Token-to-Expert Prediction ===
    # 训练 Predictor（以 FFN 为例）
    # input: token embeddings ∈ R^{seq_len × 4096}
    # output: expert logits ∈ R^{seq_len × 8}
    class FFNPredictor:
        def forward(x):  # x: (batch, seq_len, 4096)
            h = ReLU(Linear(x, 4096→128))     # (batch, seq_len, 128)
            h = ReLU(Linear(h, 128→64))        # (batch, seq_len, 64)
            logits = Linear(h, 64→8)            # (batch, seq_len, 8)
            return logits  # 每层独立 classifier head

    # 推理阶段：predictor 插入 Attention 之前
    for layer l in 1..L:
        predicted_experts = Predictor[l](hidden_states)  # overhead
        # 直接路由 token 到对应 GPU（跳过 Scatter 通信）
        tokens = DirectRoute(tokens, predicted_experts)
        output = FFN_Experts(tokens)

    # === Algorithm 1: Expert Duplication ===
    # P: expert→GPU placement, d: token→GPU dispatch
    def ExpertDuplication(f, M, P_init, C_max):
        d[t] = min{g | (f(t), g) in P}  # assign token to any GPU with its expert
        L[g] = |{t | d(t)=g}|           # load per GPU
        while max(L) - min(L) > 1:
            g_h = argmax(L); g_c = argmin(L)
            Δ = ceil((L[g_h] - L[g_c]) / 2)
            e* = most_popular_expert_on(g_h)
            if (e*, g_c) not in P and copies(e*) < C_max:
                copy_weights(e* → g_c)   # 复制 expert 权重
                P = P ∪ {(e*, g_c)}
                reassign first Δ tokens of e* from g_h to g_c
            update L[g_h], L[g_c]
        return P, d
    ```
  - 关键设计：
    - Distribution-Only Prediction 的 overhead 为零（offline 估计，MLE 公式 $\hat{p}_i = n_i/N$ 极简单）
    - Token-to-Expert Prediction 的 overhead 来自 predictor 前向推理，accuracy 越高通常 overhead 越大（更复杂模型）
    - Error rate 建模：Distribution-Only 用 $|\hat{p}-p|/(1/E)$；Token-to-Expert 用 1−accuracy。性能影响模型分 Optimistic/Typical/Pessimistic 三档，默认使用 Typical（errors uniformly distributed）

## MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE-DisCo，一种基于 Block Coordinate Descent (BCD) 和 SimulParallel SGD 的 MoE 分阶段训练框架，由四个阶段组成：
    1. **Model Decoupling（模型解耦）**：将完整 MoE 模型（E 个 expert）分解为 E 个独立的 dense 子模型，每个子模型 = 完整共享 backbone（embedding, attention, LayerNorm 等）+ 单个 expert。MoE 层中移除 gating 机制，仅保留一个 expert，形成紧凑的 dense 子模型。
    2. **Data Decoupling（数据解耦）**：使用预训练 embedding 层对每个句子提取 token embedding 并做 mean pooling 得到句子向量 h_x，通过 K-Means（K=E）将训练数据聚类为 E 个语义区分的子集，每个子集分配给一个 expert 子模型。
    3. **Independent Parallel Training（独立并行训练）**：各子模型在其分配的数据子集上独立训练，无任何跨设备通信（无 gradient/parameter 交换），可在低成本 GPU（RTX 4090）上并行执行。
    4. **Model Reintegration & Fine-Tune（模型重组与微调）**：采用 "direct integration" 策略将各 expert 参数拼接为完整 expert 层；共享参数按 WP-SGD 加权平均融合；最后在完整数据集上进行短时间 global fine-tune（A100）恢复协调的 gating 行为。
  - 实验比较：(1) MoE-DisCo vs Full-Parameter MoE training，按 training loss、PPL、downstream tasks 和训练经济成本比较；(2) 消融：K-Means clustering vs random data assignment；(3) 消融：2 experts vs 4 experts 对收敛的影响。

- 硬件平台是什么，配置是什么。
  - S-phase（子模型训练）：NVIDIA RTX 4090 × 4（并行，无通信），价格 $0.35/GPU·hour
  - F-phase（fine-tune）：NVIDIA A100 80GB × 1，价格 $2.28/GPU·hour
  - Full-Parameter baseline：NVIDIA A100 80GB × 1
  - 计算精度：bfloat16
  - 序列长度：1024
  - Batch size：16

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Qwen1.5-MoE-2.7B：约 2.7B 激活参数，性能与 Mistral-7B 相当，实验中设 E=4 experts
    - LLaMA-MoE-3.5B：基于 LLaMA 架构的 MoE 设计，实验中设 E=4 experts
  - 数据集（预训练）：C4、WikiText-2、OpenWebText
  - Benchmark/评估指标：
    - 语言建模：Training Loss、Perplexity (PPL)
    - Downstream：ARC-e（5-shot）、MMLU（5-shot）、HellaSwag（0-shot）、PIQA（0-shot）
    - 经济成本：GPU 租用费用（$）、训练时长（hours）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://anonymous.4open.science/r/MoE-DisCo-4835/
  - 框架：PyTorch（论文未明确说明具体版本）
  - 算法 Pipeline 伪代码（对应 Algorithm 1）：

```
# ===== MoE-DisCo 算法流程 =====
# 输入: 原始数据集 D, MoE 共享参数 θ_shared, E 个 expert 参数 θ_1..θ_E
# 输出: 训练完成的全局 MoE 模型 M(Θ, D)

# --- Stage 1: Data Clustering ---
for x in D:
    # 对句子中所有 token 取 embedding 后 mean pooling
    h_x = MeanPool(Embed(x))    # h_x shape: [d_embed]

# K-Means 聚类，K = E
{D_1, ..., D_E} = KMeans({h_x}, K=E)

# --- Stage 2: Independent Submodel Training (S-phase, RTX 4090) ---
for k in 1..E:    # 完全并行，无跨设备通信
    θ_shared^(k) = θ_shared          # 复制共享参数
    Θ_k = (θ_shared^(k), θ_k)       # 子模型参数 = 共享 backbone + 第 k 个 expert
    Train M(Θ_k, D_k)               # 在数据子集 D_k 上训练子模型

# --- Stage 3: Reintegration ---
θ_exp* = Concat(θ_1, ..., θ_E)     # 拼接所有 expert 参数
θ_shared* = (1/E) * Σ_{k=1}^{E} θ_shared^(k)   # 共享参数加权平均

# --- Stage 4: Global Fine-Tune (F-phase, A100) ---
Θ = (θ_shared*, θ_exp*)             # 组装完整 MoE 参数
FineTune M(Θ, D)                    # 全数据集短时间微调
```

  - 超参数（S-phase）：Optimizer=AdamW, LR=1e-4, scheduler=constant, batch=16, bf16
  - 超参数（F-phase/Full-Param）：Optimizer=AdamW, LR=3e-4, weight_decay=0.01, warmup_ratio=0.03, scheduler=Cosine, batch=16, bf16

## MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoE Jetpack，一种将预训练 dense checkpoint 转换为 MoE 模型的 fine-tuning 框架，由两部分组成：
    1. **Checkpoint Recycling（检查点回收）**：将预训练 dense 模型（predecessor）的 MLP 权重通过四种策略分配到 MoE 模型（successor）的多个 expert 中，避免从零训练 MoE。默认策略为 Importance-Based Weight Sampling：通过在 predecessor 上跑一批图像获取每层每个 channel 和 hidden neuron 的 activation 值，对 channel 按跨层平均 activation 选 top-d'，对 hidden neuron 按 activation 概率分布采样分配给不同 expert。其他策略包括 Co-Activation Graph Partitioning（用 Metis 图分割将共激活神经元分入同一 expert）、Uniform Selection（等距采样）和 Random Sampling。
    2. **SpheroMoE Layer（超球面自适应 MoE 层）**：优化 dense checkpoint 到 MoE 的 fine-tuning，包含三个改进：(a) SpheroMoE Routing：用 cross-attention 将 input token 分配到 expert slots，查询向量 Q 随机初始化并 L2-normalize 投影到超球面（避免随机初始化的数值不稳定），key 由 input token 的 LayerNorm 后线性投影得到，在超球面计算相似度 logits；(b) Expert Regularization：learnable softmax temperature T（早期大→均匀分散注意，逐步减小→专精）+ expert noise + stochastic expert dropout（概率 p 随机停用 expert）；(c) Adaptive Dual-path MoE：核心专家（Core experts，数量少参数大）处理高重要性 token + 通用专家（Universal experts，数量多参数约 1/4）处理低重要性 token。通过 checkpoint recycling 获得的 dense 先验知识帮助区分重要/非重要 token。
  - 实验比较：(1) MoE Jetpack vs Dense ViT/ConvNeXt（from scratch 和 ImageNet-21k pretrained）vs Soft MoE（from scratch）；(2) 消融：Checkpoint Recycling + Soft MoE vs 单独 SpheroMoE；(3) Checkpoint Recycling 四种策略 vs Sparse Upcycling [16]；(4) Core Experts Ratio 消融（1/3 最优）；(5) MoE layer 配置消融：层数范围（7:12 最优）、expert 数量、dense checkpoint 基础模型大小。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA RTX 4090。
  - V-JetMoE-T 训练 ImageNet-1K：120 GPU hours；CIFAR-100：2.5 GPU hours。
  - C-JetMoE-F 训练 ImageNet-1K：156 GPU hours；CIFAR-100：2.5 GPU hours。
  - V-JetMoE-S 训练 ImageNet-1K：200 GPU hours；CIFAR-100：8 GPU hours。
  - 论文总训练 GPU hours：约 3300 GPU hours（含探索验证约 8000）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Vision Transformer (ViT-S/16, ViT-T) 和 ConvNeXt (ConvNeXt-T, ConvNeXt-F)。Dense predecessor 使用 ImageNet-21K 预训练权重（来自 timm）。
  - MoE 后继模型：V-JetMoE-T（FLOPs 1.1G, core experts 98, universal experts 196, MoE layers 7:12）、C-JetMoE-F（FLOPs 1.1G）、V-JetMoE-S（FLOPs 4.3G）。
  - 数据集（8 个图像分类）：ImageNet-1K, CIFAR-10, CIFAR-100, Flowers, Pets, STL-10, Food-101, DTD。
  - Benchmark 指标：Top-1 Accuracy (%)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Adlith/MoE-Jetpack（NeurIPS 2024）
  - 框架：PyTorch 2.1.0 + MMCV 2.1.0 + MMPretrain（OpenMMLab）
  - 算法 Pipeline 伪代码：

```
# === Phase 1: Checkpoint Recycling（离线，一次性） ===

# 输入：predecessor dense 模型 P (N layers, channel dim d, hidden dim 4d)
# 输出：successor MoE 模型 S (N layers, channel dim d', hidden dim 4d', 
#        前半 N/2 dense layers + 后半 N/2 SpheroMoE layers)

# Importance-Based Weight Sampling:
images = sample_batch(dataset)  # 一批图像
activations = forward_pass(P, images)  # 获取每层各 channel/hidden neuron 的 activation

# Channel 选择：跨层平均 activation，取 top-d'
for c in range(d):
    A_c = mean([activations[l][c] for l in range(N)])
selected_channels = top_k(A_c, d')  # 选 activation 最高的 d' 个 channel

# Hidden neuron 选择：按 activation 概率分布为每个 expert 采样
for h in hidden_neurons:
    P_h = A_h / sum(all_A)  # 概率分布
for each expert e:
    expert_neurons[e] = sample_from_distribution(P_h, 4d')

# 从 predecessor 权重矩阵中提取相应行/列构造 expert 的 MLP 权重

# === Phase 2: SpheroMoE Layer 前向传播（训练 + 推理） ===

# 输入：X ∈ R^{b×n×d}（batch, token_num, channel）
# Q ∈ R^{e*s×d}（随机初始化，e experts * s slots per expert）

def spheromoe_forward(X, Q, T, core_experts, univ_experts):
    # 1. 继承 dense checkpoint 的 LayerNorm
    X_norm = inherit_layer_norm(X, dim=-1)  # b×n×d
    
    # 2. 超球面投影：Q 通过 LayerNorm + L2 normalize
    Q_norm = l2_norm(inherit_layer_norm(Q, dim=-1))  # e*s×d
    
    # 3. Key 投影
    K = linear(X_norm, W_k)  # b×n×d
    
    # 4. 超球面相似度计算（点积）
    S = einsum(K, Q_norm, "b n d, e s d -> b n e s")  # b×n×e×s
    
    # 5. Expert Regularization
    S = S + normal_noise(S) * noise_mult  # 加噪声
    dispatch = softmax(S / T, dim=1)       # temperature-scaled, b×n×e×s
    combine = softmax(S / T, dim=[-1,-2])  # b×n×e×s
    
    # 6. Token 分发到 expert slots
    X_hat = einsum(dispatch, X_norm, "b n d, b n e s -> b e s d")  # b×e×s×d
    
    # 7. Adaptive Dual-path: 分离 core 和 universal experts
    X_core = X_hat[:, :core_num, :, :]    # b×core_num×s×d
    X_univ = X_hat[:, core_num:, :, :]    # b×univ_num×s×d
    
    # 8. 并行 expert 前向（合并所有 expert 权重为一个大矩阵，单次 matmul）
    # parallel_expert_forward 等价于:
    #   x = einsum(x, experts.weight_1, "b e s d1, e d2 d1 -> b e s d2")
    #   x = x + rearrange(experts.bias_1, "e d2 -> () e () d2")
    #   x = experts.act(x)
    #   x = einsum(x, experts.weight_2, "b e s d1, e d1 d2 -> b e s d1")
    #   x = x + rearrange(experts.bias_2, "e d1 -> () e () d1")
    Y_core = parallel_expert_forward(X_core, core_experts)
    Y_univ = parallel_expert_forward(X_univ, univ_experts)
    Y_hat = concat([Y_core, Y_univ], dim=1)  # b×e×s×d
    
    # 9. Expert dropout（随机停用 expert）
    Y_hat = expert_dropout(Y_hat, p)
    
    # 10. Token 重组
    Y = einsum(combine, Y_hat, "b n e s, b e s d -> b n d")  # b×n×d
    return Y
```

- 关键设计要点：
  - **继承 LayerNorm**：X 的 LayerNorm 直接从 dense checkpoint 继承，Q 也通过相同 LayerNorm + L2 norm，保证 MoE 层与 dense checkpoint 的分布一致性。
  - **超球面相似度**：Q 经 L2 normalize 后 ‖Q_norm‖ = 1，与 K 做点积等价于 cosine similarity（因 ‖K‖ 未归一化保留了 scale 信息），解决了随机初始化 Q 的数值不稳定。
  - **并行 Expert 前向**：将所有 expert 的 weight_1 合并为一个大矩阵（shape e×d2×d1），通过单次 einsum 完成 b×e×s 个 slot 的并行计算，替代传统 for-loop 逐 expert 处理。
  - **Adaptive Dual-path**：core expert 数量 = 总 expert 数的 1/3（最优比例来自消融实验），core expert 有完整 hidden dim 4d'，universal expert hidden dim ≈ d'（约 1/4 参数）。

## MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoDES，首个面向 MoE MLLM 的 training-free 动态 expert skipping 框架。核心由三部分组成：
    1. **Globally-Modulated Local Gating (GMLG)**：离线校准层级别全局重要性因子 $\alpha^{(l)}$（通过对 calibration set 中每条数据计算原始模型与跳过第 l 层 expert 后模型输出的 KL 散度均值），推理时将 $\alpha^{(l)}$ 与局部 routing probability $\pi_i^{(l)}$ 相乘得到 expert importance score $s_i^{(l)} = \alpha^{(l)} \cdot \pi_i^{(l)}$。$\alpha^{(l)}$ 实际使用时跨层归一化为 $\widetilde{\alpha^{(l)}} = \frac{\alpha^{(l)}}{\sum_{l'=1}^L \alpha^{(l')}}$。
    2. **Dual-Modality Thresholding (DMT)**：为 text token 和 vision token 分别设置阈值 $\tau_t$ 和 $\tau_v$，对 token $\mathbf{x}^{(l)}$ 跳过 $s_i^{(l)} < \tau_t \cdot \mathbb{I}_t + \tau_v \cdot \mathbb{I}_v$ 的 expert。
    3. **Frontier Search**：利用 $f(\tau_t, \tau_v)$（输出 KL 散度）和 $g(\tau_t, \tau_v)$（expert skipping ratio）的单调性，以 $\mathcal{O}(ND)$ 时间（D=100 个 grid 点，N 个 calibration 样本）找到满足 skipping ratio 约束下最小化输出差异的最优阈值对。比 naive $\mathcal{O}(ND^2)$ 搜索快约 45×。
  - 实验比较：(1) 与 expert skipping baseline（NAEE [42]、MC-MoE [22]、DiEP [6]，均重新实现适配 MLLM top-k 场景）及直接减少 top-k 的 k 值比较，在 Kimi-VL-A3B-Instruct 上对比 50%/67%/83% 三种 skipping ratio；(2) 跨 backbone 对比（Qwen3-VL-MoE-30B-A3B-Instruct 88% skip, InternVL-3.5-30B-A3B-HF 88% skip, InternVL-3.5-GPT-OSS-20B-A4B-Preview-HF 75% skip）；(3) 与量化结合（2.5-bit 和 1.5-bit 混合精度量化）；(4) 消融实验（Thresholding vs Thresholding+GMLG vs DMT vs DMT+GMLG）；(5) Calibration 数据选择鲁棒性（GQA vs COCO vs VMMMU）；(6) 样本数 N 和 grid 点数 D 的消融。

- 硬件平台是什么，配置是什么。
  - 校准和搜索：8×H200 GPU。
  - 推理速度测试：单张 H200 GPU。Prefilling batch size=8，decoding sequence length=1024。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Kimi-VL-A3B-Instruct（3B active, 128 experts per MoE layer, k=6, MoE 在第 1-26 层）、Qwen3-VL-MoE-30B-A3B-Instruct（3B active, 128 experts per layer, k=8）、InternVL-3.5-30B-A3B-HF（128 experts, k=8）、InternVL-3.5-GPT-OSS-20B-A4B-Preview-HF（32 experts per layer, k=4）。
  - Calibration 数据：GQA 数据集中随机抽取 1024 样本。
  - Image 理解 benchmark (8个)：TextVQA_val, ChartQA, MMStar, MMBench_dev,en, MMVet, MME, RealWorldQA, COCO2017-Cap_val (CIDEr 评分)。
  - Video 理解 benchmark (5个)：MVBench, EgoSchema, VideoMME (VMME), LongVideoBench_val,v (LVB), VideoM-MMU (VMMMU)。
  - 评估框架：lmms-eval。MMBench 和 MMVet 使用 DeepSeek-V3.1 打分。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/ModelTC/MoDES
  - 算法 Pipeline 伪代码：

```
# === 离线阶段：校准 GMLG 因子 + Frontier Search ===

# Step 1: 计算每层的全局重要性因子 alpha
calib_set C = GQA 中随机 1024 样本
for each MoE layer l in [1..L]:
    prob_orig = model.forward(C)  # 原始模型输出概率分布
    prob_l_skip = model.forward(C, skip_all_experts_at_layer_l)
    alpha[l] = mean(KL(prob_orig || prob_l_skip))  # Eq.(4)
# 归一化
alpha_tilde = alpha / sum(alpha)

# Step 2: Frontier Search for optimal (tau_t, tau_v)
B = rectified_sigmoid(linspace(0, 1, D=100))  # search grid
for q = 1 to D:   # tau_t 候选
    for p from D down to 1:  # tau_v 候选, 单调递减
        g = compute_skip_ratio(model, C, tau_t=B[q], tau_v=B[p])
        if g < rho_target: break
    if p+1 <= D:
        record f(q, p+1) = mean(KL(prob_orig || prob_skip_with(tau_t=B[q], tau_v=B[p+1])))
(q*, p*) = argmin f(q, p) on frontier
tau_t_opt, tau_v_opt = B[q*], B[p*]

# === 推理阶段：动态 Expert Skipping ===

for each token x in input:
    for each MoE layer l:
        # Step A: 标准 MoE routing
        r = router(x)                          # (M,) routing logits
        pi = softmax(r)                        # (M,) routing probs, Eq.(1)
        S = topk_indices(pi, k)                # top-k expert indices

        # Step B: GMLG importance scoring
        for i in S:
            s_i = alpha_tilde[l] * pi[i]       # Eq.(3)

        # Step C: DMT threshold-based skipping
        modality_flag = (is_text_token(x) ? tau_t : tau_v)
        kept_experts = {i in S | s_i >= modality_flag}  # Eq.(5)

        # Step D: 加权聚合（仅激活保留的 expert）
        y = 0
        for i in kept_experts:
            y += pi[i] * Expert_i(x)           # Eq.(2)
    x = y
```

算法核心思想：浅层 expert 对最终输出影响更大（$\alpha^{(l)}$ 在浅层更大），应保守跳过；深层 expert 可激进跳过。Vision token 的 FFN 更新幅度小于 text token（$\cos(\text{pre-FFN}, \text{post-FFN})$ 更高），vision expert 冗余度更大，可更激进跳过。因此 DMT 中 $\tau_v > \tau_t$（vision 阈值更高，跳过更多 expert）。

## MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoDE（Mixture of Dyadic Experts），一种新的多任务 PEFT 算法。核心创新：(1) 所有 expert 共享同一个 down-projection 矩阵 A（基于 PCA 分析发现 down-projection 向量跨任务聚类，即 task-agnostic），消除 LoRA-MoE 中的参数冗余；(2) 将 LoRA 更新分解为 rank-one dyadic product 之和 $\Delta\mathbf{W} = \sum_{j=1}^r (\mathbf{a}_j \otimes \mathbf{b}_j)$，每个 rank 维度独立路由（fine-grained routing），允许 $m^r$ 种专家组合（传统 LoRA-MoE 仅 m 种）；(3) 广义 MoDE 支持 rank-p adapter，router 选择 p 列为一组。
  - 实验比较：(1) Multi-task 全量评估（756 tasks, SNI）：LoRA 64 vs MoLORA 16×4 vs MoLORA-SD 16×4 vs MoDE 16×4/8×4/6×4/4×4/4×6/4×8/4×16；(2) 广义 MoDE ablation：固定 m/r 变化 expert rank p (1→16)；(3) Iso-parametric 配置：固定总参数量变化 LoRA rank r、expert rank p；(4) Case study：15 类任务、固定参数预算约 6M，比较 LoRA 15×4（baseline，每任务独立 LoRA）vs LoRA 1×60 vs MoLORA vs MoLORA-SD vs MoDE。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明硬件平台。作者来自 Google DeepMind，推测使用 Google Cloud TPU 或 GPU。Gemma 2B 模型规模较小，可在单卡运行。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Gemma 2B（Google 开源的 2B 参数 decoder-only LLM）。
  - 数据集：Supernatural Instructions (SNI)，含 1,616 个指令遵循任务。实验使用 756 个英文任务的训练集，每任务 90/10 切分。Case study 选取 15 类任务（QuestionAnswering, WrongCandidateGeneration, QuestionGeneration, GrammarErrorDetection 等，每类 ≥5k 训练样本）。
  - 评估指标：ROUGE-L。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：**论文未开源代码**（Papers with Code 显示 "No code implementations yet"）。GitHub 上存在同名的 MoDE 项目（CLIP Data Experts 和 Diffusion Policy），与本论文无关。
  - 算法 Pipeline 伪代码：

```
# MoDE 前向传播 (per transformer layer)
# 输入: x ∈ R^{1×P}, 冻结权重 W0 ∈ R^{P×Q}
# 可训练参数: A ∈ R^{P×r} (共享), B_j^i ∈ R^{Q×1} (per rank per expert),
#              W_R;j ∈ R^{P×m} (per rank router)

def mode_forward(x):
    # 1. 冻结层输出
    y = x @ W0  # R^{1×Q}

    # 2. 共享 down-projection
    h = x @ A  # R^{1×r}, 其中 A = [a_1, ..., a_r], a_j ∈ R^{P×1}

    # 3. 对每个 rank 维度独立路由
    dyadic_sum = 0
    for j in range(r):  # 遍历每个 rank
        # 路由权重: softmax per-rank
        R_j = softmax(x @ W_R_j)  # R^(1×m)

        for i in range(m):  # 遍历每个专家
            # B_j^i ∈ R^{Q×1}, h_j 为标量
            dyadic_sum += R_j[i] * (h[:, j] * B_j^i)  # R^{1×Q}

    return y + dyadic_sum
```

张量计算等效形式：

$$\mathbf{y} = \mathbf{x}\mathbf{W_0} + \sum_{i=1}^m \sum_{j=1}^r \mathcal{R}_j^i(\mathbf{x}) \cdot (\mathbf{x} (\mathbf{a}_j \otimes \mathbf{b}_j^{iT}))$$

其中 $\mathcal{R}_j^i(\mathbf{x}) = \text{softmax}(\mathbf{x} \cdot \mathbf{W}_{\mathcal{R};j})_i$，$\mathbf{a}_j$ 是共享 down-projection 矩阵 A 的第 j 列，$\mathbf{b}_j^i$ 是第 i 个 expert 在第 j 个 rank 的 up-projection 向量。

广义 MoDE (rank-p adapter)：

$$\mathbf{y} = \mathbf{x}\mathbf{W_0} + \sum_{i=1}^{m} \sum_{k=1}^{r/p} \mathcal{R}_k^i(\mathbf{x}) \cdot \mathbf{x}\mathbf{A}_k \mathbf{B}_k^{iT}$$

其中 $\mathbf{A}_k \mathbf{B}_k^{iT} = \sum_{j=1}^p (\mathbf{a}_{j+p(k-1)} \otimes \mathbf{b}_{j+p(k-1)}^i)$。

- 训练配置：Adafactor 优化器，lr=1e-3，total sequence length=1024，batch size=128，训练 20,000 steps。
- MoDE $1 \times r \times r$ 等价于标准 LoRA rank r；MoDE $m \times r \times r$ 等价于 LoRA-MoE-SD。

## MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出三方面通信优化加速 MoE 大规模训练：(1) 为 attention 和 FFN 分别定制通信高效并行策略——attention 用 Sequence Parallelism (SP，基于 DeepSpeed-Ulysses 的 all-to-all 风格)、FFN/experts 用 Expert Parallelism (EP)，替代传统 Tensor Parallelism (TP)；(2) 通信压缩——BF16 训练中将 DP 梯度同步精度从 FP32 降至 BF16（all-to-all 替代 reduce-scatter + FP32 本地累积），FP8 训练中用 FP8 all-to-all 替代 BF16 reduce-scatter（per-token activation quantization + per-channel/group quantization）；(3) selective activation rematerialization，仅保留计算密集的中间激活，低成本的通过重计算/重通信获得，节省约 50% 激活内存。
  - 实验比较 MegaScale-MoE vs Megatron-LM（commit f1f03922），包括 strong scaling（240-1440 GPU，固定 global batch 720）、weak scaling（480-1440 GPU，batch 360→1080 等比增长）、不同 GPU 平台（H800/A100/H20）性能分解、ablation study 逐步启用 SP+EP → inter-operator overlap → intra-operator overlap。
  - 评估六种 MoE 模型：Internal-352B（60 layers, h=4096, 32 experts, top-k=3）、Mixtral-8×7B、Mixtral-8×22B、Hunyuan-Large、Phi-3.5-MoE、DeepSeekMoE。

- 硬件平台是什么，配置是什么。
  - 主要平台：NVIDIA H800 SXM GPU（Compute 989 TFLOPS, 80 GB HBM, 3.4 TB/s 内存带宽, NVLink 400 GB/s），最多 1,440 GPUs。
  - 对比平台：NVIDIA A100（312 TFLOPS, 80 GB, 2.0 TB/s, NVLink 600 GB/s）、NVIDIA H20（148 TFLOPS, 96 GB, 4.0 TB/s, NVLink 900 GB/s），各 32 GPUs。
  - 训练精度：BF16 mixed-precision 和 FP8（E4M3）。
  - 网络：intra-node NVLink + inter-node RDMA（NIC 50 GB/s 量级）。
  - Sequence length=8192, vocabulary size=65536。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Internal-352B MoE（60 layers, h=4096, 32 experts, top-k=3, GQA m=4, SwiGLU FFN h_ffn=14336），以及 Mixtral-8×7B、Mixtral-8×22B、Hunyuan-Large、Phi-3.5-MoE、DeepSeekMoE 等五个开源 MoE 模型。
  - 数据集：论文未明确说明具体训练数据集名称。用于验证 FP8 收敛性的 35B 和 176B MoE 模型也未指定数据集。
  - Benchmark：训练吞吐量（tokens/s）、MFU（Model FLOPs Utilization）、iteration time、loss curve 收敛性、内存占用。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供开源链接（论文本身发表在 EuroSys 2026，未在文中提供 GitHub 仓库地址）。
  - 系统基于 Megatron-LM 构建（开源：github.com/NVIDIA/Megatron-LM）。
  - 算法pipeline 核心计算流程（单 MoE 层 forward，基于论文 §4.1 Figure 8-9）：
    1. Input: hidden [b, s/n, h] → RMSNorm → ln1_out [b, s/n, h]
    2. QKV Projection: qkv = MatMul(ln1_out, qkv_weight) → [b, s/n, h(1+2/m)]
    3. RoPE on q, k → q_rope [b, s/n, h], k_rope [b, s/n, h/m]
    4. SP Attention: All-to-All(q_rope, k_rope, v) → qkv_a2a [b, s, h(1+2/m)/n]
    5. SelfAttention(qkv_a2a) → attn [b, s, h/n]
    6. All-to-All(attn) → attn_a2a [b, s/n, h]
    7. Output Projection: attn_out = MatMul(attn_a2a, out_weight) → [b, s/n, h]
    8. Residual: ln2_in = Add(hidden, attn_out) → RMSNorm → ln2_out [b, s/n, h]
    9. Expert dispatch: All-Gather(ln2_out) → ln2_out_ag [b, s, h] → Scatter → ffn_in [b*s*k/n, h]
    10. SwiGLU FFN: fc1_out = GroupedGEMM(ffn_in, fc1_weight), fc3_out = GroupedGEMM(ffn_in, fc3_weight), fc2_in = SiLU(fc1_out) * fc3_out, fc2_out = GroupedGEMM(fc2_in, fc2_weight) → [b*s*k/n, h]
    11. Gather(fc2_out) → fc2_out_rs [b, s, h] → Reduce-Scatter → ffn_out [b, s/n, h]
    12. Residual: hidden(next) = Add(ln2_in, ffn_out)
  - 当 top-k > n 时，EP 通信从 all-to-all 切换为 all-gather + reduce-scatter（环形通信更高效）。
  - DP 通信压缩：梯度本地 FP32 累积后 cast 到 BF16 → all-to-all（替代 reduce-scatter）→ 本地 FP32 聚合，通信量减半。

## M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 M3oE 框架，通过三个 Mixture-of-Experts 模块（共享专家 S、域专家 D、任务专家 T）以解耦方式学习 common/domain-aspect/task-aspect 用户偏好，并使用两级融合机制（第一级：域间/任务间融合；第二级：三类专家间融合）实现精确的信息聚合控制，再通过 AutoML（Bi-Level Optimization）自适应优化融合权重 α_d, α_t, β_d, β_t。
  - 实验比较 M3oE 与四类 baseline：(a) 单域单任务 MLP；(b) 多任务方法（ShBot-MTL, PLE-MTL, MMoE-MTL, AdaTT, AdaTT-sp）；(c) 多域方法（ShBot-MDL, MMoE-MDL, PLE-MDL, STAR）；(d) 多域多任务方法（ShBot-MDMT, MMoE-MDMT, PLE-MDMT, M2M）。评估指标为 AUC 和 LogLoss。
  - 消融实验：w/o AutoML、Concat modules、Fully gated modules、w/o domain module、w/o task module、w/o domain&task module。
  - 可视化：T-SNE 分析解耦嵌入和融合嵌入。
  - 超参数分析：learning rate、shared expert 数量 N。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练/评估所用的 GPU 或 CPU 硬件配置。

## Mixture of LoRA Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 **Mixture of LoRA Experts (MOLE)**，将每个已训练的 LoRA 的每一层视为一个独立 expert，在每层引入可学习的 gating function，通过 hierarchical weight control 对多个 LoRA 的层输出加权组合，实现灵活、动态、低开销的多 LoRA 组合。训练时仅优化 gating function 参数（冻结所有 LoRA 和预训练模型参数），推理时支持两种模式：(1) 使用全部已训练 LoRA 并自动分配权重；(2) 手动 mask 不需要的 LoRA 后按比例重新分配权重。
  - V&L 域实验比较 MOLE vs (a) Normalized Linear Arithmetic composition (NLA, Eq.2) 和 (b) SVDiff（均为 LoRA composition baseline），以及 full-parameter training baseline (Custom, Textual Inversion)。评估指标为 CLIP feature space 下的 Text-alignment 和 Image-alignment。
  - NLP 域实验比较 MOLE vs (a) LoRAHub 和 (b) PEMs。评估任务包括 Translation（WMT14/16, BLEU）、Struct to Text（CommonGen/DART/E2ENLG/WebNLG, Rouge-1/2/L）、Closed-Book QA（ARC-c/ARC-e/NQ/TQA, EM）、BBH（7 subtasks, EM）、NLI（ANLI-R1/R2/R3/QNLI, EM）。
  - 消融实验：w/ vs w/o gating balancing loss L_balance；仅调大温度 τ 替代 L_balance 的多组对比；coarse-to-fine gating granularity（matrix-wise m-MoLE / layer-wise l-MoLE / block-wise b-MoLE / network-wise n-MoLE）；LoRA 数量扩展（8/24/48/128 在 NLP，3/4/5/6 在 V&L）；跨任务泛化（NLI 任务训练 → BBH 评估）。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练/评估所用的具体 GPU 硬件配置。
  - V&L 域：DreamBooth 基于 Stable Diffusion V2.1，图像分辨率 512×512，DDPM sampler 50 steps，scale=7.5。
  - NLP 域：基于 FLAN-T5（Chung et al., 2022），具体参数量论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  - V&L 域模型：DreamBooth (Ruiz et al., 2023) 基于 Stable Diffusion V2.1，以 Stable Diffusion V2.1 为 base generator。
  - NLP 域模型：FLAN-T5（Chung et al., 2022）。
  - V&L 域数据集：15 组不同三概念组合（如"Fancy boot + Monster + Clock"等，见 Table 1），每组 200 张生成图像 × 5 个 text prompt 评估。训练数据未明确说明（使用 CLIP 的 local + global guidance 做无监督训练优化 MoLE）。
  - NLP 域数据集/benchmark：Translation（WMT'14 En↔Fr, WMT'16 En↔De/En↔Ro）、Struct to Text（CommonGen, DART, E2ENLG, WebNLG）、Closed-Book QA（ARC-c, ARC-e, NQ, TQA）、Big-Bench Hard（Boolean Expressions, Causal Judgement, Date Understanding, Disambiguation, Penguins in a Table, Reasoning about Colored Objects, Ruin Names）、NLI（ANLI-R1, ANLI-R2, ANLI-R3, QNLI, WNLI）。
  - NLP 域 LoRA 训练数据：各 LoRA 从 FLAN 数据集的不同子集训练获得。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文声明代码开源在 https://github.com/yushuiwx/MoLE.git，但该链接返回 404（2026-05-27 确认，可能已迁移或删除）。
  - 算法pipeline 核心计算流程（单 transformer block，基于论文 §3.2 Eq.5-13）：

  ```
  # 输入: x ∈ R^{L×d}, 预训练 block θ, 已训练 LoRA 集合 Ω={Δθ_i}_{i=0}^{N-1}
  
  # 1. 预训练 block 前向
  x_θ'    = x + f_Attn(LN(x) | θ)                         # Eq.5
  F_θ(x)  = x_θ' + f_FFN(LN(x_θ') | θ)                    # Eq.6
  
  # 2. 每个 LoRA expert i 的前向
  x_Δθi'       = x + f_Attn(LN(x) | Δθ_i)                  # Eq.7
  E_Δθi(x)     = x_Δθi' + f_FFN(LN(x_Δθi') | Δθ_i)        # Eq.8
  
  # 3. Gating 函数计算组合权重
  E_Ω(x) = Normalization(E_Δθ0(x) ⊕ ... ⊕ E_Δθ{N-1}(x))   # Eq.9, concat: R^{N·L·d}
  ε      = Flatten(E_Ω(x))^T · e                           # Eq.10, e ∈ R^{N·L·d × N}
  G_i    = exp(ε_i/τ) / Σ_j exp(ε_j/τ)                     # Eq.11, τ learnable
  
  # 4. 加权组合
  Ẽ_Ω(x) = Σ_i G_i · E_Δθi(x)                              # Eq.12

  # 5. 最终输出
  O(x) = F_θ(x) + Ẽ_Ω(x)                                   # Eq.13
  ```

  - 训练时仅优化 gating function 参数 e 和 τ（冻结 θ 和所有 Δθ_i），总可训练参数量为 O(N·L·d·N) + 1（仅 Eq.10 的 e 和 Eq.11 的 τ）。
  - V&L 域训练目标：L = L_CLIP（local + global guidance） + α · L_balance（α=0.5），400 iterations，lr=1e-5，batch size=2。
  - NLP 域训练目标：L = L_FLAN-T5（cross-entropy） + α · L_balance（α=0.5），800 iterations，lr=1e-5，batch size=12。
  - Gating balancing loss：L_balance = -log(Π_i q^(i))，其中 q^(i) = (1/M)·Σ_k exp(ε_i^k/τ) / Σ_j exp(ε_j^k/τ)，M 为嵌入 gating 的 block 数。该 loss 在 gating 均匀分布时最小化，防止 gating 坍塌到少数 LoRA。

- 模型是什么。数据集和bench分别是什么。
  - 模型：M3oE，包含 Domain Representation Extraction Layer（含 domain-specific/shared 权重矩阵元素乘 + domain-agnostic mapping）、Multi-View Expert Learning Layer（共享专家 N 个、域专家 D 个、任务专家 T 个，均为单层 MLP + LayerNorm + ReLU）、MDMT Objective Prediction Layer（D×T 个两层 MLP prediction tower，Sigmoid 输出）。
  - 数据集：(1) MovieLens-1M（~100万评分，~3900电影，用 "age" 特征切分为 3 个域，"click"/"like" 2 个任务）；(2) KuaiRand-Pure（快手短视频平台数据，用 "tab" 特征切分为 3 个域，"click"/"long-view" 2 个任务）。训练/验证/测试分割比例为 8:1:1。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Applied-Machine-Learning-Lab/M3oE
  - 算法pipeline 伪代码：

```
输入: 样本 x_d (来自域 d), 域数 D, 任务数 T
输出: 所有域-任务对的预测 y_hat_{d,t}

// 1. Domain Representation Extraction
for each domain d:
    W_hat_d = W_d ⊙ W_sh                          // element-wise product
    h_d = W_c * (W_hat_d * x_d + b_d + b_sh) + b_c + f_DA(x_d)

// 2. Shared Expert Module (N experts)
for each shared expert e in {1..N}:
    f_E^e(h_d) = ReLU(LayerNorm(W_e * h_d + b_e))
// Shared fusion with D×T gates
for each (d,t):
    S_{d,t}(h_d) = softmax(f_gate_{d,t}(h_d)) · [f_E^1(h_d), ..., f_E^N(h_d)]

// 3. Domain Expert Module (D experts)
for each domain expert k in {1..D}:
    f_E^k(h_d) = ReLU(LayerNorm(W_k * h_d + b_k))
// Domain fusion (biased)
D(h_d) = β_d * f_E^d(h_d) + (1-β_d)/(D-1) * Σ_{k≠d} f_E^k(h_d)

// 4. Task Expert Module (T experts)
for each task expert k in {1..T}:
    f_E^k(h_d) = ReLU(LayerNorm(W_k * h_d + b_k))
// Task fusion (biased)
T(h_d) = β_t * f_E^t(h_d) + (1-β_t)/(T-1) * Σ_{k≠t} f_E^k(h_d)

// 5. Multi-View Representation Balancing (Two-Level Fusion)
h̄_d = S_{d,t}(h_d) + α_d * T(h_d) + α_t * D(h_d)

// 6. Prediction
for each (d,t):
    y_hat_{d,t} = Sigmoid(W2_{d,t} * ReLU(W1_{d,t} * h̄_d + b1_{d,t}) + b2_{d,t})

// 7. AutoML - Bi-Level Optimization
for epoch in 1..E:
    更新模型参数 W = argmin_W L(W, α, β)         // 外层
    更新融合权重 α, β = argmin_{α,β} L(W*, α, β)  // 内层（基于一个 mini-batch）
    // α_d, α_t, β_d, β_t 由可训练标量经 Sigmoid 生成: w = Sigmoid(e_w)
```

- 关键参数配置：
  - embedding size = 16
  - MovieLens: N=1 shared experts, lr=1e-2
  - KuaiRand-Pure: N=4 shared experts, lr=3e-3
  - D=3 domain experts, T=2 task experts, D×T=6 prediction towers
  - Loss: Binary Cross Entropy，所有域和任务加和

## MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MC（Mixture Compressor），一种 training-free 的 MoE-LLM 混合压缩策略，包含两个阶段：
    - **PMQ（Pre-Loading Mixed-Precision Quantization）**：基于 expert 重要性（访问频率 ϕ × 激活权重 w × 量化重构误差 ϵ）构建 Integer Programming 模型，为每个 expert 分配 1/2/3-bit 的最优位宽，使用 GPTQ 执行量化。其余 attention/gating 模块统一 4-bit。
    - **ODP（Online Dynamic Pruning）**：基于 routing weight ratio w₁/w₀ 动态剪枝低置信度 expert，同时引入 token importance Iⱼ = ‖tⱼ‖₁ · (Σ Aⱼ,ᵢ)/(L-j) 保护关键 token（仅需保护 2%），防止 attention decay。
  - 实验比较：
    - PMQ vs Uni（GPTQ uniform quantization）、BSP（block score predictor, Li et al. 2024）、Hessian-based（HAWQ V2, Dong et al. 2020），在 1.57~2.54-bit 范围。
    - PMQ+ODP vs PMQ-only vs Uni，在不同 bit-width 下。
    - 压缩后 MoE vs 同规模 FP16 dense LLM（LLaMA2-7b/13b）。
    - 消融：bit-width 分配指标（random/routing weight/activation frequency/Hessian/F-norm/PMQ）、token protection ratio、pruning threshold μ、专家显著性权重 α/β/γ。
    - 不同量化技术兼容性：GPTQ vs Omniquant。
    - 挑战性 benchmark：GSM8K, HumanEval, Needle-in-a-haystack。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100-80GB（Mixtral 8×7b 用 2 卡，Mixtral 8×22b 用 4 卡用于 FP16 baseline；量化后模型在单张 A100-80GB 上测试），也测试了 RTX 3090。
  - CPU/内存：论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Mixtral 8×7b：总参数 49B（~96.8 GB FP16），32 decoder blocks，hidden_dim=4096，8 experts/layer，每 token 激活 top-2，激活参数 13B（~26.3 GB）。
    - Mixtral 8×22b：总参数 141B（~281.2 GB FP16），56 decoder blocks，hidden_dim=6144，8 experts/layer，每 token 激活 top-2，激活参数 39B（~76.5 GB）。
    - 对比 dense model：LLaMA2-7b, LLaMA2-13b（16-bit）。
  - 数据集/benchmark：
    - 校准数据：C4（128 组随机序列，每组 2048 tokens），用于计算 expert 显著性指标和 bit-width 配置。
    - 评估数据：
      - Perplexity（PPL↓）：WikiText2
      - 8 个 zero-shot benchmark（EleutherAI LM Harness, ↑）：PIQA, ARC-easy, ARC-challenge, BoolQ, HellaSwag, Winogrande, MathQA, MMLU
      - Few-shot：MMLU（5-shot）
      - 挑战性 benchmark：GSM8K（推理↑）, HumanEval（pass@10↑）, Needle-in-a-haystack（长上下文检索↑, NIAH）
    - 额外分析：MATH 数据集（用于观察 expert 激活分布差异）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/Aaronhuang-778/MC-MoE
  - 量化工具链：GPTQ（Frantar et al. 2022）执行 expert 量化，HQQ（Badri & Shaji 2024）用于保存量化权重和反量化，Omniquant（Shao et al. 2023）作为替代量化方法验证兼容性。
  - 算法pipeline 伪代码：

```
输入: MoE-LLM 模型 M（含 L 层 MoE block，每层 N 个 expert），校准数据集 C4，目标平均位宽 k
输出: 压缩后模型 M_compressed

// ===== 阶段0: Expert 显著性分析（在原始 16-bit 模型上）=====
for each MoE block l in {1..L}:
    for each expert e_i in {1..N}:
        // 计算访问频率
        ϕ_i = n_i / N_calib                    // n_i: expert i 被激活的总次数
        // 计算激活权重和
        w_i = Σ_{j=1}^{N_calib} σ_i^j / N_calib  // σ_i^j: 第 j 次推理中 expert i 的 routing weight
        // 计算量化重构误差（F-norm）
        for each bit j in {1, 2, 3}:
            ε_{i,j} = ||F(θ) - F(θ[e_i → Q(e_i, j)])||_F

// ===== 阶段1: PMQ — Integer Programming 求解最优位宽分配 =====
for each MoE block l in {1..L}:
    // 定义 binary 决策变量 x_{i,j} ∈ {0,1}: expert i 分配 j-bit
    // 求解 Integer Programming:
    MINIMIZE  Σ_i Σ_j ϕ_i^α · w_i^β · (ε_{i,j} · x_{i,j})^γ
    Subject to:
        Σ_i Σ_j j · x_{i,j} = N · k           // 平均位宽约束
        Σ_j x_{i,j} = 1, ∀i                    // 每个 expert 只分配一个位宽
        Σ_i x_{i,3} ≥ 1, Σ_i x_{i,2} ≥ 1      // 至少一个 3-bit 和 2-bit expert
        x_{i,j} ∈ {0,1}
    // 得到位宽配置 B_i ∈ {1,2,3} for each expert i

// ===== 阶段1b: 应用 GPTQ 量化 =====
for each MoE block l:
    for each expert e_i:
        位宽 b = B_i
        if b == 1:
            // 二值化（见附录 A.2）
            B̃ = (sign(W) + 1) / 2             // 映射到 {0,1}
            s = ||W||_ℓ1 / (d × m)             // scaling factor
            存储: B̃ (bool) + s (float)
        else:  // b ∈ {2,3}
            使用 GPTQ 量化: W_q = GPTQ(W, X, b)
            // GPTQ: Hessian H=2XX^T + 逐列量化 + 误差补偿
    // Attention/gating 模块统一 4-bit GPTQ

// ===== 阶段2: ODP — Online Dynamic Pruning =====
// 在推理时对每个 token t 动态执行：

for each MoE block l in {1..L}:
    // 2a. 计算 token importance（基于上一层 attention map）
    for each token j:
        I_j = ||t_j||_1 · (Σ_{i≥j} A_{j,i}) / (L - j)
    // 保护 top-2% 重要 token：这些 token 的所有 top-k expert 都保留

    // 2b. 对非保护 token，基于 routing weight 剪枝
    {w_0, w_1} = Top-2{G(t)}                  // routing scores
    if token 未被保护 AND w_1/w_0 < μ:        // μ 取 calibration 数据的中位数
        剪枝 w_1 对应的 expert，仅用 w_0 对应的 expert 计算
        y = w_0 · E_0(t)                       // 从 top-2 降为 top-1
    else:
        y = w_0 · E_0(t) + w_1 · E_1(t)       // 保留 top-2

// ===== 一比特权重反量化（推理时）=====
// 对 b=1 的 expert，反量化为:
// s · xB = s(Σ_{j: B̃_{ij}=1} x_j - Σ_{j: B̃_{ij}=0} x_j)
// MACs: 仅 m 次乘法（vs FP16 的 d×m 次），复杂度 O(m) vs O(m²)
```

  - 关键超参数配置：
    - α=1, β=1, γ=2（expert 显著性权重因子，消融实验验证稳定）
    - token 保护比例：2%（ODP 阶段）
    - pruning threshold μ：取 calibration 数据上 w₁/w₀ 的中位数
    - 校准数据：C4，128 序列 × 2048 tokens
    - 量化时间：Mixtral 8×7b 约 90 分钟（GPTQ）
  - 1-bit 权重存储格式：通过 B̃ = (sign(W)+1)/2 将 ±1 映射到 {0,1}，真正用 1-bit 内存存储每个元素。反量化仅需 m 次乘法（vs FP16 的 dm 次）。

## MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MPMoE，一个基于 PyTorch 的高性能 MoE 训练库，核心实现包括三个组件：
    - **Micro-Batch Pipeline Parallelism**：将 mini-batch 按 batch size 维度切分为多个 micro-batch，在 MoE 层的三个阶段（S: 第一个 All-to-All dispatch → C: Expert 计算 → R: 第二个 All-to-All collect）之间实现 pipeline 并行，使通信和计算重叠执行。与 FasterMoE 按 node 维度切分不同，MPMoE 按 batch 维度切分，保留了 NCCL All-to-All 的优化能力，且 pipeline granularity n 可灵活调整。
    - **Memory Reuse Strategies**：针对 MoE 训练中 activation tensors 和 temporary buffers 占主要内存的观察，提出 4 种内存复用策略（S1-S4），通过不同方式恢复前向中被覆盖的 tensors（T_DI 和 T_M）：S1（T_DI/T_M 均 offload 到 CPU）、S2（T_DI 通信恢复 + T_M offload）、S3（T_DI offload + T_M recompute）、S4（T_DI 通信恢复 + T_M recompute）。将所需 activation buffer 从 n 份压缩为 1 份。
    - **Joint Optimization**：配置 (n, S) 的联合优化——n 为 pipeline granularity，S 为内存复用策略。提供两种方法：(a) MPMoE-pb：profile-based 搜索算法（Algorithm 1），利用单调性和抛物线假设减少搜索空间；(b) MPMoE-pm：基于 3 种 pipeline paradigm（范式1/2/3，如图 8）和 piecewise 性能模型（如图 9），在运行时估算不同配置的执行时间。
  - 实验比较：
    - 端到端训练速度：MPMoE-pb vs MPMoE-pm vs FasterMoE vs FastMoE，在 Adira（64 A100）和 Valor（16 V100）两个集群上。
    - 内存占用：MPMoE vs FastMoE vs FasterMoE vs PMoE（无内存复用的 MPMoE 变体），在不同 pipeline stage 数 n=2/4/8 下。
    - 理论内存节省上限 vs 实际内存节省（Equation 6 验证）。
    - 消融实验：(a) 通信效率 micro-benchmark（FasterMoE vs MPMoE 不同 n 下的 All-to-All dispatch/recovery 时间）；(b) Pipeline granularity 敏感度分析（不同 B 和 n 的性能变化）；(c) 内存复用策略开销分析（S1-S4 在不同 N 和 B 下的表现）。
    - 性能分解与开销分析（TensorCore 加速率、data partition 开销、profiling 开销）。
    - 多节点可扩展性（1/2/4/8 nodes on Adira，throughput 对比）。

- 硬件平台是什么，配置是什么。
  - **Adira 集群**：8 台 NVIDIA DGX A100 服务器，每节点 8×A100 40GB GPU（共 64 GPU），200 Gbps HDR InfiniBand 互联，节点内第 3 代 NVLink。
  - **Valor 集群**：4 节点，16× NVIDIA Tesla V100 16GB HBM GPU，每节点 4 GPU，56 Gbps HDR InfiniBand 互联，节点内第 2 代 NVLink。
  - 软件栈：PyTorch 1.9、CUDA Toolkit 11.1、NCCL 2.7、Ubuntu 18.04。

- 模型是什么。数据集和bench分别是什么。
  - 模型：3 种 MoE 配置（见表 3）：
    - MoE-GPT-S: d_model=768, d_hidden=3072, #experts=64 或 16
    - MoE-GPT-XL: d_model=2048, d_hidden=8192, #experts=64 或 16
    - MoE-BERT-L: d_model=1024, d_hidden=4096, #experts=64 或 16
  - 数据集：Dummy dataset（随机生成的 tokens），因为评估目标是训练系统的 throughput 和 memory footprint，而非模型精度。
  - 优化器：Adam。
  - 评估指标：平均训练时间（用于 speedup 计算）、峰值内存占用（peak memory footprint）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未提供开源链接，经搜索（论文页面、武大 ICS 实验室页面）未发现公开代码仓库。论文的 conference 版本 MPipeMoE 发表于 IPDPS 2023。
  - 算法 pipeline 伪代码：

```
输入: 当前 MoE layer 的 input tensor T_I, shape (N, B, M)
      N = device 数, B = batch size, M = model dimension
      配置: pipeline granularity n, memory reuse strategy S ∈ {S1, S2, S3, S4}
输出: MoE layer 的 output tensor T_O, shape (N, B, M)

// ===== 0. 确定最优配置 (Section 4) =====
// MPMoE-pb: profile-based (Algorithm 1)
if MPMoE-pb:
    if B not in cache C:
        在 G 中找最近区间 (n_l, n_h)
        n_best = searchBestGran(B, (n_l, n_h))  // 抛物线终止搜索
        更新 G 和 C
    n = C[B]
    for each S in {S1, S2, S3, S4}:
        用 n 执行 profiling，取执行时间最短的 S
// MPMoE-pm: performance model
else:  // MPMoE-pm
    对每种配置 (n, S):
        根据 paradigm (1/2/3) 和 piecewise 速度模型估算 T(n,S)
        // Piecewise 速度: W_comp(B/n), W_comm(B/n), W_mem(B/n)
        // α 干扰因子: α(comm,comp), α(comp,comm), α(comm,mem)
    取 T 最小的 (n, S)

// ===== 1. Micro-Batch Pipeline Parallelism (Section 3.2) =====
将 T_I 沿 batch 维度切分为 n 个 micro-batch:
    T_I[0], T_I[1], ..., T_I[n-1], 每个 shape (N, B/n, M)

定义 pipeline stages:
    S(i): 第 i 个 micro-batch 的 All-to-All dispatch (T_I[i] → T_DI[i])
    C(i): 第 i 个 micro-batch 的 Expert FFN 计算 (T_DI[i] → T_M[i] → T_DO[i])
    R(i): 第 i 个 micro-batch 的 All-to-All collect (T_DO[i] → T_O[i])

Pipeline 调度 (如图 7 timeline，交替执行 S 和 R 以增强 memory locality):
    时间 t0: S(0) 启动
    时间 t1: S(0) 完成 → 同时启动 C(0) 和 S(1)
    时间 t2: R(0) 在 C(0) 完成后启动, S(2) 启动
    ...
    // Tensor Shape Flow (以 expert FFN 为例):
    // T_DI[i]: (B/n, M) → Linear1: W1(M, H) → T_M[i]: (B/n, H)
    // T_M[i] → GeLU(in-place) → Linear2: W2(H, M) → T_DO[i]: (B/n, M)

// ===== 2. Memory Reuse (Section 3.3) =====
// 原本每个 partition 独立分配 buffer，现改为共享:
Buffer_DI = alloc(B/n * M * sizeof(fp16))   // n partitions 共享
Buffer_M  = alloc(B/n * H * sizeof(fp16))   // n partitions 共享
Buffer_DO = alloc(B/n * M * sizeof(fp16))   // n partitions 共享

// 前向: 各 micro-batch 的 tensors 依次复用同一 buffer
// 后向: 需恢复被覆写的 T_DI, T_M，根据策略 S:
switch S:
    case S1:  // offload T_DI, offload T_M
        forward:  // Paradigm 2
            C(i) 完成后: D2H_copy(T_DI[i])   // 异步拷贝到 CPU
            S(i) 完成后: D2H_copy(T_M[i])   // 异步拷贝到 CPU
        backward:  // Paradigm 3
            H2D_copy(T_M[i])                // 先从 CPU 取回
            H2D_copy(T_DI[i])
            计算梯度
    case S2:  // comm restore T_DI, offload T_M
        forward:  // Paradigm 2
            仅 offload T_M[i]
        backward:  // Paradigm 3
            H2D_copy(T_M[i])
            T_DI[i] = All-to-All_replay(T_I[i])  // 重新通信
            计算梯度
    case S3:  // offload T_DI, recompute T_M
        forward:  // Paradigm 2
            仅 offload T_DI[i]
        backward:
            H2D_copy(T_DI[i])
            T_M[i] = FFN_forward(T_DI[i])  // 重新计算
            计算梯度
    case S4:  // comm restore T_DI, recompute T_M
        forward:  // Paradigm 1 (无 memory copy)
            不 offload 任何 tensor
        backward:
            T_DI[i] = All-to-All_replay(T_I[i])
            T_M[i] = FFN_forward(T_DI[i])
            计算梯度

// ===== 3. Memory Footprint Calculation (Section 2.2) =====
原始 M_act = 4*B*M + B*H          // Equation 2
原始 M_buf = B*M + B*H            // Equation 3
Pipeline 后的 M_act^pipe = M_buf^pipe = 4*B*M + B*H  // Equation 4
Memory reuse 后节省:
    ΔM_act = ΔM_buf = B * (2M*(n-2)/n + H*(n-1)/n)  // Equation 5
Memory 节省率:
    φ = (ΔM_act + ΔM_buf) / (M_ms + M_act^pipe + M_buf^pipe)  // Equation 6
// 其中 M_ms = 4 * (E*M + 2*H*M)  // 包含 params, grads, momentum, variance

// ===== 4. Performance Model (Section 4.2) =====
// Piecewise 速度函数 (Figure 9):
W_comp(volume) = { k1_comp * volume,  if volume < V_threshold_comp
                 { k2_comp * volume,  otherwise
W_comm(volume)  = { k1_comm * volume,  if volume < V_threshold_comm
                 { k2_comm * volume,  otherwise
// 带 α 干扰因子的实际执行时间:
// 以 Paradigm 1 的 P2 阶段为例:
T_P2 = max( (t_S + t_R) / α(comm,comp), t_C / α(comp,comm) )
```

- 关键配置与结果：
  - Pipeline granularity n: 2/4/8（B < 8k 时 n=2 最优，8k-22k 时 n=4，>22k 时 n=8，Figure 14 验证了单调性假设）。
  - 内存复用策略选择：N 小（如 8 GPU）时 S1/S2 更优，N 大（如 64 GPU）时 S4 更优（Figure 15）。
  - MPMoE-pb 平均 1.66× speedup vs FasterMoE，MPMoE-pm 平均 1.55×；vs FastMoE 分别 2.34× 和 2.20×（Figure 10）。
  - 内存节省：n=2/4/8 时分别平均节省 23%/34%/38%，最高比 FastMoE/FasterMoE 节省 53%（Figure 11）。
  - 实际内存节省达到理论上限的约 95%（Figure 12）。
  - 8 节点扩展比：MPMoE 5.76×（72% ideal），FasterMoE 5.4×（Figure 17）。
  - Profiling overhead: MPMoE-pb <3%，MPMoE-pm <1%（Figure 16）。

## MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MPipeMoE，基于 PyTorch 1.9 + CUDA 11.1 的 MoE 训练库，包含三个核心实现：
    - **Adaptive Pipeline Parallelism**：将 mini-batch 的 tokens 沿 batch 维度切分为 n 个 micro-batch，对 MoE 层的三个阶段（S: All-to-All dispatch → C: Expert FFN 计算 → R: All-to-All collect）进行 pipeline 并行，重叠通信与计算。与 FasterMoE 沿 node 维度切分不同，MPipeMoE 沿 batch 维度切分，保留 NCCL All-to-All 集体通信优化能力。
    - **Adaptive Granularity Configuration (Algorithm 1)**：基于"n 随 B 单调递增"的假设，将 B 的值域划分为不相交区间{R_n}，每个区间一对一映射到最优 n。通过二分搜索树维护 (n, R_n) 映射集，以 O(log n) 复杂度查找。当 cache miss 时调用 searchBestGran(B) 做 trial-based 搜索。
    - **Memory Reusing Strategies (S1-S4)**：识别 pipeline parallelism 中的 "memory bubbles"——不同 micro-batch 的 T_DI、T_M、T_DO 在不同时刻激活，可共享同一 buffer。n 个 partition 的 activation buffer 从 O(n) 压缩为 O(1)，节省 ΔM_act = B*(2M*(n-2)/n + H*(n-1)/n)。为恢复 backward pass 所需的被覆写 tensors，设计 4 种策略：S1 (T_DI/T_M 均 CPU offload)、S2 (T_DI 通信重发 + T_M offload)、S3 (T_DI offload + T_M 重计算)、S4 (T_DI 通信重发 + T_M 重计算)，通过性能模型在运行时选择最优策略。
  - 实验比较：
    - 端到端训练速度：PipeMoE vs FastMoE vs FasterMoE，MPipeMoE vs PipeMoE vs FastMoE vs FasterMoE（Figure 8, 9）。
    - 内存占用：MPipeMoE vs FastMoE vs FasterMoE，归一化到 FastMoE（Figure 9）。
    - 理论内存节省 bound vs 实际节省（Figure 10），n=2/4/8 及 B=4k-32k。
    - Pipeline granularity 有效性：不同 n（1/2/4/8/16）在不同 B（2k-32k）下的性能（Figure 12）。
    - 内存复用策略开销：S1-S4 在不同 (N, B) 下的表现（Figure 13）。
    - 性能分解（Figure 11）：memory-time 坐标系下各方法对比。
  - 变体：*PipeMoE* 仅含 pipeline parallelism（无 memory reuse），*MPipeMoE* = PipeMoE + memory reuse。

- 硬件平台是什么，配置是什么。
  - 8 台 NVIDIA DGX A100 服务器，每节点 8×A100 SXM 40GB GPU，200 Gbps HDR InfiniBand 互联，96×第 2 代 AMD EPYC CPU 核，1.9 TiB 内存。节点内第 3 代 NVLink + NVSwitch。跨节点 1,600 Gbps InfiniBand 自适应路由。
  - 软件栈：PyTorch 1.9.0、CUDA 11.1、NCCL（版本论文未明确说明）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：3 种 MoE 配置（Table III）：
    - MoE-GPT3-S: d_model=768, d_hidden=3072, #experts=64
    - MoE-GPT3-XL: d_model=2048, d_hidden=8192, #experts=64
    - MoE-BERT-L: d_model=1024, d_hidden=4096, #experts=64
    - Expert 为 FFN（Linear1 → GeLU in-place → Linear2），gating 为 top-1 routing。
  - 数据集：Dummy dataset（随机生成 tokens），评估目标是训练系统的 throughput 和 memory footprint。
  - 优化器：Adam（momentum + variance 各占参数量内存）。
  - 评估指标：平均训练时间（计算 speedup）、峰值内存占用（peak memory footprint）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/whuzhangzheng/MPipeMoE
  - 算法 pipeline 伪代码：

```
输入: MoE layer input T_I, shape (N, B, M)
      N=device数, B=batch size, M=model dim
      n=pipeline granularity, S∈{S1,S2,S3,S4}=memory reuse策略
输出: MoE layer output T_O

// 1. Adaptive Granularity Search (Algorithm 1)
if B in cache_table:
    n = cache_table[B]
else:
    在集合S = {(n, range(B_lower, B_upper))} 中查找B所属区间
    if 未找到:
        n = searchBestGran(B)  // trial-based搜索
        插入新的(n, range(B,B))到S
    cache_table[B] = n

// 2. Micro-Batch Pipeline (按batch维度切分)
将T_I沿dim=1切为n个micro-batch: T_I[0..n-1], each (N, B/n, M)
Pipeline调度 (交替S和R以增强memory locality):
  stream_comm: S(0)→S(1)→R(0)→S(2)→R(1)→...
  stream_comp:         C(0)→C(1)→C(2)→...
  // S: All-to-All dispatch, C: Expert FFN, R: All-to-All collect
  // C(i) = Linear2(GeLU(Linear1(T_DI[i])))

// 3. Memory Reuse (共享buffer)
Buffer_DI = alloc(B/n * M * sizeof(fp16))  // n个partition共享
Buffer_M  = alloc(B/n * H * sizeof(fp16))
Buffer_DO = alloc(B/n * M * sizeof(fp16))
// 前向: 各micro-batch依次复用同一buffer, 后写入覆盖前写入

// 4. Backward Tensor Recovery (按策略S)
switch S:
  case S1:  // T_DI, T_M 均CPU offload
    fwd: D2H_copy(T_DI[i], T_M[i])
    bwd: H2D_copy(T_M[i], T_DI[i]), 计算梯度
  case S2:  // T_DI通信恢复, T_M offload
    fwd: D2H_copy(T_M[i])
    bwd: H2D_copy(T_M[i]), T_DI[i]=AlltoAll_replay(T_I[i]), 计算梯度
  case S3:  // T_DI offload, T_M重计算
    fwd: D2H_copy(T_DI[i])
    bwd: H2D_copy(T_DI[i]), T_M[i]=FFN_fwd(T_DI[i]), 计算梯度
  case S4:  // T_DI通信恢复, T_M重计算
    fwd: (无额外操作)
    bwd: T_DI[i]=AlltoAll_replay(T_I[i]), T_M[i]=FFN_fwd(T_DI[i]), 计算梯度

// 5. 性能模型选择最优S (Eq 10)
v0_comp = b*H*M, v0_comm = b*M, v0_mem = b*M, b=B/n
C(S) = max(q1*v0_comp/(σ*W_comp), q2*v0_comm/(μ*W_comm), q3*v0_mem/(η*W_mem))
// Q_fw=[q1,q2,q3]见表II, μ/σ/η为干扰slowdown因子, 选C最小的S

// 6. Memory Footprint
M_act^pipe = M_buf^pipe = 4*B*M + B*H          // Eq 4
ΔM_act = ΔM_buf = B*(2M*(n-2)/n + H*(n-1)/n)  // Eq 5
φ = (ΔM_act+ΔM_buf)/(M_ms+M_act^pipe+M_buf^pipe) // Eq 6, M_ms = 4*(E*M+2*H*M)
```

  - 关键结果：
    - PipeMoE 平均 2.26× speedup vs FasterMoE，最高 3.4× vs FasterMoE，最高 3.7× vs FastMoE（Figure 8）。
    - MPipeMoE 内存节省：平均 23%（最高 40%）vs FastMoE，平均 27%（最高 47%）vs FasterMoE，同时 3.1× speedup（Figure 9）。
    - 实际内存节省约达理论上限的 95%（Figure 10）。
    - Pipeline granularity: B<8k 时 n=2 最优, 8k-22k 时 n=4, >22k 时 n=8（Figure 12）。
    - 内存复用策略选择：N 小时 S1/S2 更优（I/O bound 容忍），N 大时 S4 更优（避免 memory bandwidth 竞争）（Figure 13）。
    - 策略 S1/S2 在小 N（如 8 GPU）表现好，S4 在 N=32/64 时表现好（通信瓶颈场景）。

## Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 LEGO（LLM Judge with Graph Mixture-of-expert）框架，核心是将 LLM 作为 context-aware routing function 融入 Graph MoE 框架，实现动态系统在环境变化下的鲁棒预测。具体实现包括三个组件：
    - **Hierarchical Prompt Engineering**：从 system level（系统参数/背景描述）、object level（各物体初始位置和速度向量，数值作为 token）、edge level（边连接关系文本化描述）三个层次提取环境上下文，编码为 LLM prompt。
    - **Graph Mixture-of-Expert（MoE）**：K 个同构 EGNN（E(n) Equivariant GNN）作为 graph experts，各自拥有独立参数 θ¹,...,θᴷ。每个 expert 通过迭代消息传递（Eq. 1-3：ϕ 学习交互 → AGG 聚合 → COM 组合 node/coordinate representation）生成隐藏表示 Hᴷ = f_{θᴷ}(G, X⁽⁰⁾)。最终预测通过 MoE routing function 加权组合：x̂ᵢ⁽ᵗ⁾ = Decoder(Σ ω(k)·hᵢᵏ)。
    - **LLM Judge for Context-Aware Routing**：将 LLM 作为 routing function（而非直接生成预测）。LLM 接收 hierarchical prompt 后评估 K 个 experts 的候选预测，选择最适应当前环境的一个。使用 one-hot routing + label smoothing（Eq. 7：选中 expert 权重 α，其余 (1-α)/(K-1)），配合 diversity-enhanced contrastive loss（Eq. 9-10）确保不同 expert 学习多样化动力学模式。
    - 优化采用交替更新（Algorithm 1）：每隔若干 epoch 更新 LLM 生成的 routing weights，内部循环通过梯度下降优化 graph expert 参数。
  - 实验比较：
    - Baseline：Linear、Dynamic（物理匀速模型）、GNN（Kipf & Welling 2017）、Radial Field（Köhler et al. 2019）、EGNN（Satorras et al. 2022）、EGNO（Xu et al. 2024）。
    - LEGO 变体组合：EGNN+LEGO、EGNO+LEGO、Radial Field+LEGO。
    - 消融实验（Ablation）：V1（仅 system level prompt）、V2（system + edge level prompt，无 object level）、V3（完整三层 prompt）。
    - LLM Judge vs LLM Forecasting 对比。
    - 敏感度分析：不同 LLM（规模对比）、不同 expert 数量 K ∈ {3,5,10,15,20}、不同 LLM temperature ∈ {0,0.25,0.5,0.75,1}。
    - Case Study：LLM Judge 的逐步推理过程分析。
    - 更多结果：ETH-UCY 上 vs Eq-Motion，MD17 上 vs Se3-Transformer/TFN，不同原子数分子间的迁移。

- 硬件平台是什么，配置是什么。
  - 训练硬件：论文未明确说明 GPU/CPU 具体型号和数量。
  - LLM：Llama 3.1 8B 版本作为 LLM Judge（推理用），论文未说明 LLM 推理所用的具体 GPU 配置。
  - 优化器：Adam，学习率 0.0005，batch size 100。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - 基础 GNN expert：EGNN（Satorras et al. 2022）—— E(n) 等变图神经网络，消息传递包含 node representation h 和 coordinate x 的联合更新；EGNO（Xu et al. 2024）—— 等变图神经算子，结合 Fourier 神经算子；Radial Field（Köhler et al. 2019）—— 仅操作位置坐标的 E(n) 等变模型。
    - LEGO 框架可构建于任意基础 GNN 模型之上。
    - LLM Judge：Llama 3.1 8B（Dubey et al. 2024），用于 context-aware routing。
    - 默认 K=5 个 graph experts。
  - 数据集：
    - **Spring**（Satorras et al. 2022）：N-body 弹簧系统，粒子通过弹簧力相互作用（F=k·x）。5 个粒子，3D 空间。训练集 strength=1.0, start_state=30, end_state=40。Hard/Soft/Temporal Shift 三种环境变化。时间窗口 ΔT=10。3000/2000/2000 train/val/test。
    - **Charged**（Satorras et al. 2022）：N-body 电荷系统，粒子通过库仑力相互作用（F=k·q₁·q₂/r²）。5 个粒子，3D 空间。类似设置，含无环境变化、多种 strength 和 temporal shift 场景。
    - **MD17**（Chmiela et al. 2017）：分子动力学数据集。训练用 salicylic acid，测试用 naphthalene（不同分子 = OOD 环境变化）。去除氢原子。时间窗口 ΔT=50。500/2000/2000 train/val/test。
    - **Motion**（CMU 2003）：人体运动捕捉。训练用 Subject #35（Walk），测试用 Subject #9（Run）。关节点为边，关节点交点为节点。200 train / 240 val / 240 test 轨迹。时间窗口 ΔT=30。
    - **ETH-UCY**（Li et al. 2016）：行人轨迹预测（Appendix D.4），评估指标 ADE/FDE。
  - 评估指标：MSE（Mean Square Error）×10⁻²，ADE（Average Displacement Error），FDE（Final Displacement Error）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/jdp22/LEGO.git
  - 算法pipeline 伪代码：

```
输入: 初始状态 X⁽⁰⁾ ∈ R^{N×d}, 交互图 G=(V,E), 环境参数
      K 个 graph experts {θ¹,...,θᴷ}, 预训练 LLM, 时间 t
输出: 预测状态 X̂⁽ᵗ⁾

// 1. Hierarchical Prompt Extraction
prompt_system = f"System has {N} balls with spring constant k={coeff}"
prompt_object = f"Ball 0: pos=({x₀},{y₀},{z₀}), vel=({vx₀},{vy₀},{vz₀})..."
prompt_edge   = f"Ball 2 connects ball 0, ball 1, ball 3..."
hierarchical_prompt = [prompt_system, prompt_object, prompt_edge]

// 2. Graph Expert Forward Pass (per expert k)
for each expert k in {1..K}:
    h_i⁰ = MLP(x_i⁽⁰⁾)  // 初始 node embedding
    for l in {1..L}:     // L 层 EGNN
        for each edge (i,j):
            e_ij^l = ϕ(h_j^{l-1}, x_j^{l-1}, h_i^{l-1}, x_i^{l-1})  // Eq.1
        for each node i:
            h_i^l = COM^H(h_i^{l-1}, AGG({e_ij^l | j∈N(i)}))        // Eq.2
            x_i^l = COM^X(x_i^{l-1}, AGG({e_ij^l | j∈N(i)}))        // Eq.3
    H^k = [h_1^L, ..., h_N^L] = f_{θᴷ}(G, X⁽⁰⁾)

// 3. LLM Judge: Context-aware Routing
for each expert k:
    candidate_prediction X̂⁽ᵗ⁾,ᴷ = Decoder(h_i^k)  // Eq.6, one-hot routing
// 将 hierarchical_prompt + candidate_predictions 送入 LLM
LLM_input = hierarchical_prompt + descriptions of K candidate predictions
chosen_expert = LLM(LLM_input)  // LLM 选择最合适的 expert

// 4. Label Smoothing Routing Weights (Eq.7)
for k in {1..K}:
    ê^k(k) = α                if k == chosen_expert
    ê^k(j) = (1-α)/(K-1)      if j != chosen_expert

// 5. Final Prediction with Smoothed Weights (Eq.8)
for each node i:
    h_i_combined = Σ_{k=1}^K ê^k(k) · h_i^k
    x̂_i⁽ᵗ⁾ = Decoder(h_i_combined)  // Decoder: 另一层 EGNN

// 6. Loss Computation (Eq.11)
loss_mse = ||X⁽ᵗ⁾ - X̂⁽ᵗ⁾||²
// Diversity Loss (Eq.9-10): 同 expert 内的表征相近，不同 expert 间的表征远离
for each node i and expert k:
    S_i^k = {h_i^k from training data for expert k}  // activated representations
    ℓ_i^k = -1/C * Σ log(exp(h_i^k·h̃_i^k/τ) / Σ_{h∈S_i} exp(h_i^k·h/τ))
loss_div = (1/(K*N)) * Σ_k Σ_i ℓ_i^k
loss = loss_mse + loss_div

// 7. Alternative Optimization (Algorithm 1)
while not converged:
    更新 routing weights（通过 LLM 推理，每隔若干 epoch）
    for epochs in {1..E}:
        固定 routing weights
        通过梯度下降优化 {θ¹,...,θᴷ}（Adam, lr=0.0005）
```

  - 关键超参数：
    - K=5 个 graph experts（默认，来自参数敏感度实验）
    - α（label smoothing 系数）∈ (0,1)
    - τ（contrastive loss 温度系数）
    - LLM temperature = 0（推理阶段低 temperature 更优）
    - 交替更新间隔：每隔若干 epoch 更新一次 LLM routing weights
    - 优化器：Adam, lr=0.0005, batch_size=100

## MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 **dropless-MoE (dMoE)**，将 MoE 层的 expert 计算从 batched matrix multiplication 重新表述为 block-sparse matrix multiplication。核心思想：标准 MoE 使用 batched GEMM 要求所有 expert 分配相同数量 tokens（导致 token dropping 或 padding 浪费）；MegaBlocks 将 expert 计算视为 block diagonal matrix multiplication，允许可变大小 block（即负载不均衡的 token 分配），通过将大 block 分解为多个 128×128 固定大小小 block 的 block-sparse 矩阵乘法来实现。
  - 实验比较：
    - dMoE (MegaBlocks) vs dMoE (Tutel, dynamic capacity factor) vs Dense Transformer (Megatron-LM)：在 The Pile 上训练 decoder-only Transformer 语言模型 (MoE-XS/Small/Medium)，比较端到端训练时间和 validation loss（Figure 7）。MegaBlocks 实现 1.38×、2.0×、4.35× 加速。
    - dMoE (MegaBlocks) vs token-dropping MoE (Tutel, capacity_factor=1/1.5/2)：在相同模型配置下，以 loss-equivalent Pareto 前沿比较训练时间（Figure 8）。MegaBlocks 减少训练时间 1.18×–1.38×。
    - Block-sparse matrix multiplication kernel micro-benchmarks vs cuBLAS batched GEMM（Figure 9）：18 种问题配置（3 模型 × 6 operations），平均达到 cuBLAS 98.6% 吞吐量。
    - MoE layer forward pass vs Megatron-LM SwitchMLP（sequential expert, Appendix A）：num_experts=128 时 20× 加速。
    - Block-sparse kernels vs Triton Blocksparse（Appendix C）：平均 9× 吞吐量优势。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 SXM4 80GB。单卡实验（§3 motivation）用 1×A100。端到端训练实验（§6.1）用 8×A100 SXM4 80GB（8-way expert model parallelism for MoE layers + data parallelism for other layers）。
  - 软件：CUDA 11.5、CUTLASS 2.5、PyTorch + Megatron-LM。Mixed-precision training (FP16 + FP32 accumulation)。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Decoder-only Transformer 语言模型，配置如表 1 & 2：
    - Transformer-XS (46M), Transformer-Small (125M), Transformer-Medium (356M), Transformer-Large (760M), Transformer-XL (1316M)
    - MoE-XS (839M), MoE-Small (3,693M), MoE-Medium (13,041M)：将 Transformer 的 FFN 层替换为 64-expert MoE 层，top-1 routing，每个 expert 为 2 层 MLP（ffn_hidden_size=4×hidden_size）
    - 所有模型 vocabulary_size=51200, sequence_length=1024, attention_head_size=64
  - 数据集：The Pile（Gao et al. 2020），使用 GPT2 tokenization（Radford et al. 2019）。训练 10B tokens，batch size 512 sequences。训练/验证集划分按 The Pile 标准划分。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/stanford-futuredata/megablocks，Apache-2.0 许可证
  - 算法pipeline 伪代码（dMoE forward，对应 Figure 4）：

```
# x.shape: (num_tokens, hidden_size)
def dmoe_forward(self, x):
    # (1) Router: Assign tokens to experts
    # indices.shape: (num_tokens), weights.shape: (num_tokens)
    indices, weights = router(x)  # top-k greedy selection

    # (2) Create block-sparse matrix topology
    # Constructs the variable-size block diagonal matrix
    # as many 128x128 fixed blocks (Figure 3C)
    topology = make_topology(indices)

    # (3) Permute tokens to group by expert assignment
    # Pad each expert batch to multiple of block size (128)
    x = padded_gather(x, indices)

    # (4) Compute expert layers via block-sparse ops
    # self.w1.shape: (hidden_size, ffn_hidden_size * num_experts)
    # self.w2.shape: (ffn_hidden_size * num_experts, hidden_size)
    # SDD: Sparse = Dense x Dense (Figure 3C forward)
    x = sdd(x, self.w1, topology)       # output: block-sparse
    # DSD: Dense = Sparse x Dense (second layer)
    x = dsd(x, self.w2)                 # output: dense

    # (5) Un-permute tokens and scale by router probabilities
    x = padded_scatter(x, indices)
    return x * weights
```

  - 关键：SDD 操作中，sparse output matrix 对应图 3C 的 block diagonal structure。每个 expert 的 token batch 被分解为 ceil(num_tokens_expert/128)×128 行的多个 block。w1 和 w2 的列维度按 expert 拼接（concatenated），使得单次 block-sparse 矩阵乘法等价于并行计算所有 expert。
  - 向后传播：对 MLP expert (2-layer FFN)，需要 SDD^T (第二层 weight grad)、DS^T D (第二层 data grad)、DSD^T (第一层 data grad)、DD^T S (第一层 weight grad) 四种操作。

## MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出三类 MoE 合并技术：(1) 同构专家合并：用 Dare/Ties 合并替代 BTX 的简单平均（unweighted averaging）处理参数干扰（parameter interference），减少 post-merge fine-tuning 需求；(2) 无 fine-tuning 路由启发式：PPL（perplexity）路由 + 分离 attention 层替代训练的 router network，仅需一次额外 forward pass 计算 PPL 即完成路由决策；(3) 异构专家合并：通过 Proj-in/Proj-out projector（随机初始化 MLP）+ sequence-level router 将不同架构的专家模型合并为统一 MoE。
  - 实验比较：
    - 同构合并（Table 1）：BTX merging vs Ties merging vs Dare merging vs Random Routing vs Router Fine-tuning vs MoE Upcycling，6 benchmark 评估。
    - 无 fine-tuning 合并（Table 3）：Dare Dense vs Ties Dense vs merge attention + PPL routing vs separate attention + PPL routing vs separate attention + task vector routing，验证分离 attention 层和 PPL 路由的有效性。
    - 异构合并（Table 4）：3-expert MoE vs MoE w/ Math Olmo vs MoE w/ Math TinyLlama，验证 projector-based 异构合并性能。
    - 消融分析：路由概率分析（Figure 5/6/7/8/9）、不同 fine-tuning token 数量下的性能变化（Figure 10）、训练成本对比（Table 6/7/8）。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练/评估所用的 GPU 具体型号和数量。论文提到 "limitations of computation resources" 限制在 1B 级别模型实验，BTX 论文（Sukhbaatar et al. 2024）的 MoE fine-tuning 需多 GPU 支持（因 expert 间 GPU 通信开销），但 MergeME 未说明具体集群配置。推测使用了至少支持 4-expert MoE（~3.7B 参数）训练的 GPU 集群。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Base-1B：Llama-2 架构，24 层，hidden_dim=2048，从 RedPajama 数据集（Arxiv, CommonCrawl, C4, StackExchange, 前一半 Wikipedia）random init 预训练 250B tokens。
    - Math Expert：Base-1B 在 OpenWebMath 上 CPT 100B tokens。
    - Code Expert：Base-1B 在 RedPajama GitHub 数据上 CPT 100B tokens。
    - Knowledge Expert：Base-1B 在 RedPajama 后一半 Wikipedia 数据上 CPT 100B tokens。
    - Math TinyLlama：TinyLlama-1.1B（22 层, hidden_dim=2048）在 Math Expert 相同数据上 CPT。
    - Math Olmo：Olmo-1B（16 层, hidden_dim=2048）在 Math Expert 相同数据上 CPT。
    - 同构 MoE：合并 Base-1B + Math Expert + Code Expert + Knowledge Expert，top-2 routing，总参数 ~3.7B。
    - 异构 MoE：合并 Base-1B + Code Expert + Knowledge Expert + (Math TinyLlama 或 Math Olmo)，总参数 ~4B。
  - 数据集/Benchmark：GSM8K (8-shot)、MATH (4-shot)、MBPP (0-shot)、HumanEval (0-shot)、Natural Questions (NQ, 5-shot)、TriviaQA (5-shot)。
  - 训练数据：CPT 用 100B tokens/专家，MoE fine-tuning 用额外 40B tokens（混合所有数据源，按 Table 5 比例采样）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供开源链接。经 web search，未发现该论文的公开代码仓库（截至论文阅读时）。论文作者来自 University of Maryland 和 Amazon AGI。
  - 算法pipeline 核心流程（三种合并模式）：

```
输入: l 个 dense expert 模型参数 [θ₁,θ₂,...,θₗ]，base 模型参数 θ_b
输出: 合并后的 MoE 模型 θ_m

// ====== 模式1: 同构专家合并（Dare/Ties 替代平均）======

// 步骤 1: 计算 task vectors
for each expert i in {1..l}:
    τᵢ = θ_b - θᵢ  // task vector = base - expert 参数差

// 步骤 2a: Ties merging
// 参数: 保留比例 p (默认 80%)
for each task vector τᵢ:
    按 magnitude 排序，将 bottom (100-p)% 参数重置为 0  // 剪枝冗余参数
for each parameter position j:
    确定所有 τᵢ[j] 中总 magnitude 最大的符号方向
    τ_m[j] = Σ_{i: sign(τᵢ[j]) == 主导符号} τᵢ[j]      // 仅累加同符号值

// 步骤 2b: Dare merging（替代 Ties）
for each task vector τᵢ:
    随机 drop (100-p)% 参数（重置为 0）
    τᵢ = τᵢ / (0.01 * p)  // rescale 补偿丢弃
τ_m = Σ_{i=1}^{l} τᵢ  // 简单求和

// 步骤 3: 合并回 base model
θ_m = θ_b + λ · τ_m  // λ = 1/3（scaling term）

// 步骤 4: MoE 结构构建
// 非 FFN 层（embedding, attention, normalization, head）用 θ_m
// FFN 层保持各 expert 独立
// 插入 router network: θ_r * v → SoftMax(top-K(·))
// FF_MoE(v) = Σ_{i=1}^{K} SoftMax(top-K(θ_r · v)) · FF_i(v)

// 步骤 5: Fine-tuning
// 在混合数据源（Table 5 比例）上 fine-tune 全部参数（含 router），40B tokens

// ====== 模式2: 无 Fine-tuning MoE（PPL 路由 + 分离 attention）======

// 输入: 推理 prompt x_inf（t 个 tokens）
// 预处理: 各 expert 的 attention 层不合并，保持独立

// PPL 路由（替代 router network）:
for each expert i in {1..l}:
    PPL(x_inf | θᵢ) = exp( -1/t * Σ_{j=1}^{t} log P(xⱼ | x_{<j}, θᵢ) )
confidence_i = 1 / PPL(x_inf | θᵢ)

α = SoftMax(top-K(confidence_1, confidence_2, ..., confidence_l))
// α 即为各 expert 的权重

// Token 处理:
// 输入按 α 权重分配给 top-K expert
// 每个 expert 使用自己的 attention + FFN 处理
// 输出: Σ αᵢ · expert_outputᵢ

// ====== 模式3: 异构专家合并（Projector + Sequence-level Router）======

// 各 expert 架构不同（层数、hidden_dim 不同）
// 设最大 hidden_dim = d_m，各 expert hidden_dim = dᵢ

// 共享层:
// 1. Embedding 层 M_e: V → R^{d_m}（各 expert embedding/head 平均，小维度 padding 0）
// 2. Head 层 M_h: R^{d_m} → R^{|V|}

// 投影层（per expert，随机初始化 MLP）:
Proj-inᵢ:  R^{d_m} → R^{dᵢ}
Proj-outᵢ: R^{dᵢ} → R^{d_m}

// Sequence-level Routing:
// 将输入所有 token 的 embedding 平均
avg_e = 1/t * Σ_{j=1}^{t} eⱼ  // eⱼ = M_e(vⱼ)
α = SoftMax(top-K(θ_r · avg_e))

// Forward Pass:
for each selected expert k:
    e_proj[k] = Proj-in_k(e₁, e₂, ..., e_t)  // 投影到 expert k 的维度
    h_k = Expert_k.forward(e_proj[k])         // 标准 forward（含 attention + FFN）
    r_k = Proj-out_k(h_k)                     // 投影回 d_m

// 组合输出:
combined = Σ_{k in top-K} α_k · r_k
output_logits = M_h(combined)  // head 层输出 token 概率分布

// Fine-tuning: 所有参数（含 projector）在混合数据上 fine-tune
```

  - 关键超参数配置：
    - Dare/Ties: p=80% (retain ratio), λ=1/3 (scaling term)
    - Top-2 routing for all MoE models
    - CPT learning rate=1e-5, weight decay=0.01
    - Fine-tuning: 40B tokens on mixed data sources
    - Inference: temperature=0.0 (greedy decoding), max generated tokens=512

## MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MergeMoE，一种基于输出合并观点的 MoE 模型压缩算法。核心将 expert merging 重新解释为在 forward computation 中插入额外矩阵（A、B、T1、T2、T3）的优化问题：
    - 聚类阶段：基于 expert 权重矩阵 W_U 和 W_G 拼接结果的相似度进行聚类（选取 top-M 使用频率的 experts 作为聚类中心，其余按距离归类）。
    - 合并阶段：簇内使用相对使用频率作为加权权重（经理论证明最优），T2/T3 按 M-SMoE 方式设为加权平均矩阵（式4），T1 通过对采样输入做最小二乘法求解（式6：T1 = QP†）。
    - 最终输出 W'_D T1, T2 W'_G, T3 W'_U 作为合并 expert 的权重矩阵。
  - 实验比较 MergeMoE vs M-SMoE（主 baseline）、Average（简单平均）、ZipIt 三种合并方案；额外与同激活参数规模的 dense 模型（Qwen3-4B, Qwen1.5-1.8B/4B）对比。
  - 消融实验：(1) 不同压缩比例的影响（减少 experts 数量 vs 增加压缩层数）；(2) 输入样本数量对最小二乘法质量的影响（临界阈值 ~32 samples）；(3) 跨数据集泛化能力（单数据集采样 → 全 benchmark 评估）；(4) 压缩误差消融（w/o merging errors vs w/ merging errors）；(5) 合并时间开销对比（MergeMoE vs M-SMoE）；(6) IFEval 指令遵循 benchmark + knowledge distillation 验证。
  - 所有比较实验固定相同压缩层和压缩比确保公平，所有合并算法使用相同数量输入样本。

- 硬件平台是什么，配置是什么。
  - 合并执行：单张 NVIDIA H20 96GB 显存。
  - 评估执行：两张 NVIDIA H20 96GB。
  - 精度：BFloat16（合并阶段的压缩矩阵计算在 GPU 内存中进行）。
  - 合并算法在单 GPU 上运行，每层处理时间 <1 分钟。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - **Qwen3-30B-A3B**：14B 参数，48 layers，128 routed experts，每 token 激活 8 experts，无共享 experts。压缩：layers 28-47, experts 128→64, 总参数 30B→25B，激活参数 ~3B。
    - **Qwen1.5-MoE-A2.7B**：14B 参数，24 layers，60 routed experts，每 token 激活 4 experts，有共享 experts。压缩：layers 10-23, experts 60→30, 总参数 14B→10B，激活参数 ~2.7B。
    - **DeepSeekMoE**：16B 参数，28 layers，64 routed experts，每 token 激活 6 experts，有共享 experts。压缩：layers 16-27, experts 64→28, 总参数 16B→12B。
  - 数据集/Benchmark（7 个 NLP 任务）：
    - MRPC（语义等价判断）
    - WinoGrande（指代消解）
    - SQuAD（抽取式问答）
    - Hellaswag（常识推理）
    - PIQA（物理交互推理）
    - ARC easy / ARC challenge（科学推理）
    - 额外：IFEval（指令遵循 benchmark）+ ShareGPT（知识蒸馏数据）
  - 评估框架：DCLM（DataComp-LM）执行下游任务评估。
  - 输入采样数据来源：各 benchmark 数据本身（self-sourced），或单一数据集跨任务评估。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供开源链接。经搜索未发现公开代码仓库。论文作者来自 Peking University 和 ByteDance。
  - 算法 pipeline 伪代码（单 MoE layer，将 N 个 experts 压缩为 M 个）：

```
输入: MoE layer 含 N 个 routed experts {E_1..E_N}, target M 个 experts
      calibration 输入样本 X̂ (来自 target task 数据)
输出: M 个合并后的 experts {E'_1..E'_M}

// ===== 步骤 1: 聚类 =====
// 计算每个 expert 的使用频率 f_i = 该 expert 被 top-K 选中的次数
// 选取 top-M 使用频率的 experts 作为聚类中心
for each non-center expert j:
    // 距离度量: 拼接矩阵 [W_U || W_G] 的相似度
    dist(j, center_k) = ||[W_Uj||W_Gj] - [W_Uk||W_Gk]||
    将 expert j 分配给距离最近的聚类中心

// 聚类结果 → 确定矩阵 A (式 2):
// A ∈ R^{M×N}, A_{ij}=1 iff 第 j 个 expert 归入第 i 个 cluster

// ===== 步骤 2: 确定合并权重 (Theorem 1, 使用频率最优) =====
for each cluster C_i:
    for each expert j in C_i:
        B_{ji} = f_j / Σ_{k∈C_i} f_k   // 簇内相对使用频率作为权重
// B ∈ R^{N×M} 的列 v_i 仅在 C_i 的索引位置非零

// ===== 步骤 3: 构造扩展参数的合并 expert =====
for each cluster i:
    // 构造中间扩展矩阵 (无维度缩减):
    W'_{Di} = [B_{1i}W_{D1}, B_{2i}W_{D2}, ..., B_{Ni}W_{DN}]  // 水平拼接
    W'_{Gi} = [W_{G1}; W_{G2}; ...; W_{GN}]                     // 垂直拼接
    W'_{Ui} = [W_{U1}; W_{U2}; ...; W_{UN}]                     // 垂直拼接

// ===== 步骤 4: 设置 T2, T3 (式 4, 加权平均) =====
// T2, T3 ∈ R^{E·N × E}  (E = 单个 expert 的 intermediate dim)
T2 = [B_{1i}I, B_{2i}I, ..., B_{Ni}I]  // block diagonal with weights
T3 = [B_{1i}I, B_{2i}I, ..., B_{Ni}I]  // 同上

// ===== 步骤 5: 最小二乘法计算 T1 (式 5-6) =====
// 在前向过程中用 torch hooks 获取中间激活
// 对采样输入 X̂ 做一次前向:
P = σ(T2 · W'_{Gi} · X̂) ⊙ (T3 · W'_{Ui} · X̂)   // 压缩路径的中间激活
Q = σ(W'_{Gi} · X̂) ⊙ (W'_{Ui} · X̂)             // 原始扩展路径的中间激活
// 最小二乘闭式解:
T1 = Q · P^†     // P^† 为 Moore-Penrose 伪逆
// T1 ∈ R^{E×E}, 将扩展维度压缩回单个 expert 维度

// ===== 步骤 6: 构造最终压缩 expert 权重 =====
W^final_Di = W'_{Di} · T1   // shape: (out_dim, E)
W^final_Gi = T2 · W'_{Gi}   // shape: (E, in_dim)
W^final_Ui = T3 · W'_{Ui}   // shape: (E, in_dim)

// ===== 步骤 7: 路由权重更新 =====
// 合并后路由权重 = A · 原始路由权重 (相当于原簇内 experts 路由权重求和)
merged_routing_weights = A · original_routing_weights
```

  - 关键实现细节：
    - 压缩按层从后往前执行（后层不影响前层激活），逐层获取中间激活 → 做最小二乘 → 释放内存。
    - BFloat16 精度最大化输入样本量，同时避免 GPU OOM。
    - 类似 M-SMoE，保留 N 个 expert 引用但指向 M 个实际 merged expert（矩阵 A 隐式编码）。
    - 对于含共享 experts 的 MoE 模型（DeepSeekMoE, Qwen1.5-MoE），仅压缩 routed experts，共享 experts 保持不变。
  - 批量大小与样本数配置：
    - Qwen3: ARC challenge/HellaSwag/PIQA/SQuAD 用 16 samples, WinoGrande/MRPC 用 40
    - Qwen1.5: PIQA/SQuAD 用 32 samples, 其余用 64
    - DeepSeekMoE: WinoGrande/MRPC 用 128, ARC easy/challenge/HellaSwag 用 64, 其余用 40
  - 合并时间：MergeMoE 比 M-SMoE 慢（因最小二乘法），但仍在 1 分钟内完成单任务合并（batch_size=128, Qwen1.5, WinoGrande）。

## Making MoE-based LLM Inference Resilient with Tarragon

- 属于算法pipeline的实现是什么？实验比较什么？
  TARRAGON 提出了一系列算法层面的故障恢复机制，核心包括：
  1. **异步增量 KV Cache Checkpointing**：AW 在每层 attention 计算完成后，利用 AW-EW 通信间隙（link idle 时段），通过 one-sided RDMA write 将新增的 KV cache segment（每 token 每层一个小 segment）异步写入 checkpoint store。使用 "async log + commit record" 设计保证顺序（基于单调递增的 RDMA work request ID 作为 sequence number），避免干扰正常 AW-EW 流量。
  2. **Per-Request KV Cache Restoration**：故障时仅恢复受影响请求的 KV cache。Checkpoint store 通过 GPUDirect one-sided RDMA write 将 KV cache segments 直接注入替代 AW 的 GPU 显存，替代 AW 从 committed token 继续 decoding，无需重放 prefill/decoding。
  3. **AW 侧自愈算法（EW 故障容忍）**：AW 对 EW 响应设置超时；超时后 REFE 立即将请求重路由到替代 EW（健康 primary 或 shadow expert），重播相同 token embeddings + metadata。因 expert 计算是 stateless 和 deterministic 的，重播产生相同结果。
  4. **EW 侧自愈算法（AW 故障容忍）**：EW 不再等待所有 AW 的输入。当收到"足够数量"健康 AW 的输入（或 batch 达到配置的最小大小）时即开始 expert 计算，省略未响应 AW 的 slots。
  5. **Shadow Experts**：在 EW GPU 显存中预加载 expert 权重的 inactive 副本，primary 故障时可立即激活，避免从存储重新加载权重（数百毫秒到秒级延迟）。
  实验比较了：
  - 不同 checkpointing 方案（No-CKPT / Pause-Checkpoint-Resume / TARRAGON incremental）的吞吐量开销
  - 不同 AW 恢复策略（Sequential replay / Parallel replay / TARRAGON per-request restoration）在 restoration time、transfer data volume、GPU recomputation cost 三个维度上的表现
  - 不同 failure point（已 decoding token 数量）下的恢复代价

- 硬件平台是什么，配置是什么。
  GCP A3 Ultra 节点，每节点 8x NVIDIA H200 GPUs (141 GB 显存), 8x 400 Gbps ConnectX-7 RDMA NICs (GPUDirect RDMA), NVLink 3.6 Tbps。3 节点：AWs 1 节点 + EWs 1 节点 + Checkpoint store 1 节点。Ubuntu 22.04, Linux 5.15, CUDA 12.8, PyTorch 2.6.0。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Mixtral-8×7B（32 层 MoE transformer, 8 experts/layer, top-2 expert selection），MoE 层 hidden_size=4096。
  - 数据集/Workload：
    - ShareGPT：自然变化的 prompt 长度，测试 prefill 和 decode 的真实请求异构性
    - Random（synthetic）：固定 10 input tokens + 128 generated tokens，强调 decoding 阶段
  - 请求到达：Poisson 过程，varied rates (30-70 RPS)
  - 评估指标：TTFT (Time-to-First-Token), TBT (Time-Between-Tokens, median + P95), Output-token throughput, T_stall (failure-induced stall time), GPU recomputation cost (GPU-time)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文声明将开源（"We will open-source TARRAGON"），截止论文阅读时尚未公开链接。
  
  **KV Cache Checkpointing 算法流程**：
  
  ```
  输入: AW 上每层 attention 完成后新产生的 KV cache segment
        segment 大小 C = 2 * H_kv * (N_hidden_size / H_attn) * S_elem
        对于 Mixtral-8×7B (GQA): C ≈ 12.5% of expert traffic V
        V = 2 * Top_k * N_hidden_size * S_elem
  
  // 初始化
  AW 分配连续 GPU KV cache region，通过 RDMA 注册
  Checkpoint store 分配对应 bucket，返回 base address
  seq_num = 0  // 单调递增的 RDMA work request ID
  
  // 增量更新（每 token 每层）
  on_attention_done(layer_l, token_t):
      segment = KV_cache[token_t][layer_l]  // 刚写入的 segment
      // 等待 AW-EW link idle（opportunistic interleaving）
      wait_until_aw_ew_link_idle()
      // One-sided RDMA write，不涉及 receiver CPU
      rdma_write(
          src = segment.gpu_addr,
          dst = checkpoint_store.bucket_base + offset,
          size = C,
          wr_id = seq_num  // sequence number 保证顺序
      )
      seq_num += 1
      // 写入 commit record（标记此 segment 已持久化）
      rdma_write_commit_record(token_t, layer_l)
  ```
  
  **Per-Request KV Cache Restoration 算法流程**：
  
  ```
  输入: failed_aw_id, 该 AW 上的活跃请求列表 requests[]
  输出: 恢复后的请求在替代 AW 上继续 decoding
  
  on_aw_failure(failed_aw_id):
      // 1. Orchestrator 识别 failed AW 上所有活跃请求
      for each request r in failed_aw.active_requests:
          committed_token = checkpoint_store.get_latest_commit(r)
          // committed_token: 最后一个已 checkpoint 的 token index
  
      // 2. 负载均衡分发到健康 AWs
      for each request r in round_robin:
          alt_aw = select_healthy_aw()
          assign(r, alt_aw)
  
      // 3. Per-request 恢复（并行执行）
      for each (r, alt_aw):
          // Step a: Checkpoint store 通知 alt_aw 恢复所需信息
          checkpoint_store → alt_aw: committed_token_id, kv_state_size
          
          // Step b: alt_aw 分配 KV cache 区域
          kv_region = alt_aw.allocate_kv_cache_region(kv_state_size)
          alt_aw → checkpoint_store: kv_region.offset
          
          // Step c: Checkpoint store 通过 GPUDirect RDMA 注入 KV cache
          for each layer l in {1..L}:
              for each segment s in {1..committed_token}:
                  rdma_write(
                      src = checkpoint_store.bucket[r][l][s],
                      dst = alt_aw.gpu_mem[kv_region.offset + l * segment_stride + s],
                      size = C
                  )
          
          // Step d: 确认完成，resume decoding
          checkpoint_store → alt_aw: HTTP restore_complete(r)
          alt_aw.resume_decoding(r, from_token=committed_token + 1)
  ```
  
  **AW 侧自愈（EW 故障）**：
  
  ```
  on_ew_timeout(ew_id, request_metadata, token_embeddings):
      // REFE 探测到 EW 无响应
      alt_ew = ERT.lookup_alternative(expert_id)
      if alt_ew is None:
          // 激活 shadow expert（已预加载在 GPU 显存中）
          alt_ew = ERT.activate_shadow(expert_id)
      // 重播请求到替代 EW（带优先级标记，避免 straggler）
      rdma_write_prioritized(alt_ew, metadata, token_embeddings)
  ```
  
  **EW 侧自愈（AW 故障）**：
  
  ```
  on_expert_batch_ready(layer_l, expert_e):
      // EW 收集来自各 AW 的 tokens
      received = buffer[expert_e][layer_l].tokens
      healthy_aws = liveness_probe_all_aws()
      if len(received) >= min(threshold, len(healthy_aws)):
          // 开始计算，不等所有 AW
          outputs = expert_ffn_forward(received)
          // 返回结果给各自的 AW
          for each aw in received.sources:
              rdma_write(aw, outputs[aw])
  ```
  
  **Shadow Expert 机制**：
  
  ```
  // 初始化：在 EW GPU 显存中预加载 inactive expert 副本
  for each expert e in primary_experts:
      shadow_e = load_weights(e.weights)  // 仅占 GPU 显存，不消耗 compute
      // 对于 DeepSeek-R1: 单个 expert 约 2.5 GB
      // 多个 active + shadow experts 可舒适放入 A100/H200 40-141 GB 显存
  
  // 故障时激活
  on_primary_ew_failure(failed_ew):
      for each expert e in failed_ew.experts:
          shadow = find_shadow_replica(e)
          shadow.activate()  // 开始接受请求
          ERT.update(e.id → shadow.physical_location)
  ```

  关键参数：
  - KV cache segment 大小 C：对于 Mixtral-8×7B (GQA)，仅为 expert traffic V 的 ~12.5%
  - Failure detection probing interval: 10 ms
  - 连续超时阈值: 3（RDMA QP 级别配置）
  - Shadow expert: inactive 时不消耗 compute，仅占 GPU 显存

## Mixture of Lookup Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MoLE（Mixture of Lookup Experts），一种训练与推理结构不同的 MoE 架构。训练时 routed experts 是 FFN，但以 embedding tokens（embedding 层输出）为输入，且所有 experts 同时激活。推理前将 routed experts 重参数化为 lookup table（LUT），LUT 存储所有 vocabulary 中每个 token 对应的 expert 输出 v_j^i = FFN_j(Embedding(i))，离线预计算后 offload 到存储设备。推理时从 LUT 直接检索 expert 输出，无需计算，仅需一次 lookup + router 加权求和，通信开销可忽略。共享 expert FFN_shared 保持标准计算。
  - 实验比较 Dense baseline（Pythia 架构）、MoE baseline（Mixtral 风格，top-2 routing, 10/34 experts）、MoLE（4/16 experts），在 160M/410M/1B 激活参数规模下。
  - 评估指标：8 个 zero-shot benchmark（ARC-C, ARC-E, BoolQ, HellaSwag, PIQA, RACE, SIQA, LAMBADA）的 accuracy、per-step decoding latency（V100 + HuggingFace Transformers）、#Param Offloaded、#Param Loaded per Token。
  - 消融实验：(a) 训练 loss（LM loss only vs +load_balance vs +z-loss）；(b) routed expert hidden dimension D_r（d/4d/16d）；(c) routed expert 数量 N（2/4/8/16/32）；(d) Architecture Design 逐步演进（MoE-10E → +Full Activation → +Reconfiguration → +Embedding as inputs → +Re-param. = MoLE-4E）；(e) LUT 后训练量化（FP16/NF4/NF3）。

- 硬件平台是什么，配置是什么。
  - 训练硬件：论文未明确说明 GPU 型号。使用 bf16 精度，global batch size=1024，seq length=2048，50000 training iterations。
  - 推理延迟测量：NVIDIA V100 GPU，使用 HuggingFace Transformers。参数加载延迟按 V100 最大 PCIe 带宽 16 GB/s 估算。
  - 训练软件栈：PyTorch + HuggingFace Transformers，基于 Pythia 代码库。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Dense（Pythia 架构）、MoE（Mixtral 风格，无共享 expert，top-2 routing，D_r = Dense FFN hidden_dim / 2）、MoLE（共享 expert = Dense FFN，routed expert D_r = 共享 expert hidden_dim，所有 expert 激活）。具体配置见 Table 2（160M: L=12/d=768/D_s=3072; 410M: L=24/d=1024/D_s=4096; 1B: L=16/d=2048/D_s=8192）。
  - 数据集：100B-token subset of deduped Pile dataset，GPT-NeoX tokenizer（vocab size 50k）。
  - Benchmark：ARC-C, ARC-E, BoolQ, HellaSwag, PIQA, RACE, SIQA, LAMBADA（通过 lm-evaluation-harness 评估，zero-shot accuracy）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/JieShibo/MoLE（含 modeling_dense.py, modeling_moe.py, modeling_mole.py, modeling_mole_rep.py）。HuggingFace checkpoint：JieShibo/MoLE-{160M,410M}-{4E,16E}。
  - 算法pipeline 核心流程（训练 → 重参数化 → 推理）：

**训练阶段（MoLE Decoder Layer forward）:**
```
输入: hidden_states ∈ R^{b×s×d}, input_ids ∈ R^{b×s}
      embedding_states = Embedding(input_ids) ∈ R^{b×s×d}

1. Attention:
   residual = hidden_states
   hidden_states = RMSNorm(hidden_states)
   hidden_states = SelfAttention(hidden_states)   // QKV + attention + output proj
   hidden_states = residual + hidden_states

2. Shared Expert (始终激活，接受中间特征):
   residual = hidden_states
   hidden_states = RMSNorm(hidden_states)
   shared_output = FFN_shared(hidden_states)       // [b, s, d_s] → SwiGLU → [b, s, d]

3. Routed Experts (接受 embedding tokens，全激活):
   router_value = SoftMax(Router(hidden_states))   // [b, s, N]
   embedding_states = RMSNorm(embedding_states)
   routed_output = stack([FFN_j(embedding_states) for j in 1..N], dim=2)  // [b, s, N, d]
   routed_output = sum(routed_output * router_value.unsqueeze(-1), dim=2) // [b, s, d]

4. 输出:
   hidden_states = residual + shared_output + routed_output
```
关键差异：routed experts 的输入是 `embedding_states`（仅依赖 input ids），而非 `hidden_states`（中间特征，含上下文）。无 auxiliary loss（因所有 experts 始终激活且可微）。

**重参数化阶段（训练后、推理前）:**
```
# 对每个 expert j 和每个 vocabulary token i，预计算 expert 输出
for j in 1..N:
    for i in 1..|V|:
        e_i = Embedding(i)                         // [d]
        v_j^i = FFN_j(e_i)                         // [d], 只需一次 forward
LUT_l = {v_j^i}_{j=1..N, i=1..|V|}                // size: N × |V| × d
// 实际实现：以 embedding weights 为输入做单次 FFN_j forward
// W_emb ∈ R^{|V|×d} → FFN_j(W_emb) → R^{|V|×d}
```

**推理阶段（MoLE Decoder Layer forward）:**
```
1. Lookup:
   lookup_results = LUT(input_ids)                 // [b, s, N*d]
   lookup_results = lookup_results.view(b, s, N, d)

2. Attention: 同训练

3. Shared Expert: 同训练

4. Routed Expert (计算-free):
   router_value = SoftMax(Router(hidden_states))   // [b, s, N]
   routed_output = sum(lookup_results * router_value.unsqueeze(-1), dim=2) // [b, s, d]

5. 输出: 同训练
```
推理时 routed experts 零 FLOPs，仅 lookup + 加权求和。每 token 加载参数量：dN（仅加载 |V| 中当前 token 对应的 N 个 expert 输出），与 MoE expert offloading 的 2dkD_r 相比，小 1000× 以上。

**复杂度对比（Table 1）:**
- Dense: FLOPs=4dD_s, Offloaded=0, Loaded/token=0
- MoE: FLOPs=4d(kD_r+D_s), Offloaded=2dND_r, Loaded/token=2dkD_r
- MoLE: FLOPs=4dD_s, Offloaded=dN|V|, Loaded/token=dN

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

## MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MixLoRA，一种基于 LoRA 的 MoE 参数高效微调方法。核心实现：(a) **MoE 构建**：从预训练 dense 模型的 FFN 层构造稀疏 MoE——每个 expert = 共享的冻结 FFN 权重 + 独立 LoRA 适配器（作为 expert 的更新参数存储），替代传统将 LoRA 直接作为 expert 的方式；(b) **Top-K Router**：线性层 + Softmax + KeepTop-2，为每个 token 选择最合适的 2 个 LoRA expert；(c) **负载均衡**：受 Switch Transformers 启发的 auxiliary load balance loss，L_aux = a·N·Σ F_i·P_i，a=1e-2；(d) **Attention 层 LoRA**：额外在 self-attention 的 q,k,v,o 投影上添加独立 LoRA 适配器（非 MoE），提升性能；(e) **MixDoRA**：用 DoRA 替代 LoRA 作为 expert 基础单元的变体。
  - **性能优化**：
    - **(I) 计算复杂度降低**：共享 FFN 的 W1 和 W3 计算结果跨 expert 复用，先将输入送入 W1/W3 做线性投影，再按路由权重切片分发给各 expert 的 LoRA 计算，减少约 30% token 计算延迟。
    - **(II) 多模型高吞吐**：受 m-LoRA 启发，多个 MixLoRA 模型的 multi-task 输入 pack 为单 batch，共享预训练权重，per-model peak GPU memory 降低约 45%。
  - 实验比较：(a) 单任务学习——MixLoRA/MixDoRA vs LoRA/DoRA（r=80），8 个 commonsense reasoning 数据集 accuracy；(b) 多任务学习——混合 ARC/BoolQ/OBQA/PIQA 训练后分别评估，对比 single-task→multi-task 性能退化；(c) 消融：auxiliary loss coefficient a、LoRA rank r、expert load distribution；(d) 计算效率——token compute latency (µs) 和 peak GPU memory (GB)，对比 LoRA/DoRA/vanilla MixLoRA/optimized MixLoRA，含单模型和多模型（×2）场景。

- 硬件平台是什么，配置是什么。
  - 7B 模型：24GB 显存 GPU（RTX 3090, RTX A5000, RTX 4090）。
  - 8B/13B 模型：48GB 显存 GPU（RTX A6000）。
  - 软件栈：Python 3.10, Ubuntu 22.04, x86-64 CPU。
  - 训练精度：half precision（FP16/BF16，论文未细分说明）。
  - 训练超参：cutoff length=512, lr=2e-4, AdamW optimizer, batch size=16, accumulation steps=8, dropout=0.05, epochs=2。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Gemma 2B, LLaMA-2 7B, LLaMA-2 13B, LLaMA-3 8B。
  - MixLoRA 配置：r=16, alpha=32, 8 experts, top-2 router, LoRA 应用于 q,k,v,o（attention）+ w1,w2,w3（FFN expert）。
  - Baseline LoRA/DoRA 配置：r=80, alpha=160, LoRA 应用于 q,k,v,o + w1,w2,w3（控制等量可训练参数）。
  - 数据集（均从 HuggingFace DATASETS 下载）：
    - ARC-e (2250 train / 2380 test), ARC-c (1120 train / 1170 test) — 科学问答
    - BoolQ (9427 train / 3270 test) — 文本分类
    - OpenBookQA (4957 train / 500 test) — 科学事实问答
    - PIQA (16100 train / 1840 test) — 物理交互推理
    - SIQA (33410 train / 1954 test) — 社交交互推理
    - HellaSwag (39905 train / 10042 test) — 句子补全
    - WinoGrande (9248 train / 1267 test) — 填空
  - 评估指标：Accuracy（所有数据集）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/TUDB-Labs/MixLoRA
  - 算法 pipeline 伪代码（MixLoRA 单层 forward，基于 §3.2 公式 5-7 和 Algorithm 1 / Appendix A.7）：

```
输入: hidden states h^{l-1} ∈ R^{B×N×D}  (B=batch, N=seq_len, D=hidden_dim)
      pretrained FFN weights W1,W2,W3 (共享、冻结)
      K 个 LoRA expert: {A_i^{W1}, B_i^{W1}, A_i^{W2}, B_i^{W2}, A_i^{W3}, B_i^{W3}}_{i=1..K}
      每层 router: W_r^l ∈ R^{D×K}
输出: h^l ∈ R^{B×N×D}

// 1. Attention (标准 MSA + LoRA on Q,K,V,O)
z^l = MSA(LN(h^{l-1})) + h^{l-1}
// MSA 中使用 LoRA 修正 Q,K,V,O: W' = W + B·A

// 2. MixLoRA MoE FFN (替代原 FFN)
x = LN(z^l)                              // [B, N, D]

// 2a. Router 计算 (per token)
r = W_r^l · x                            // [B, N, K] logits
r' = KeepTop-2(Softmax(r))               // [B, N, K], 仅 top-2 位置非零

// 2b. [优化] 共享计算：先对全输入做 W1/W3
h_W1 = x · W1^T                           // [B, N, D']  D'=intermediate_dim
h_W3 = x · W3^T                           // [B, N, D']

// 2c. 可选：多模型 batch 模式（Multi-MixLoRA）
// 将来自 M 个 MixLoRA 模型的输入 pack 为一个 batch，共享 W1/W3 计算

// 2d. 逐 expert 计算
h^l = 0                                   // 初始化为零
for k in {1..K}:
    // Expert k 的 LoRA 增量
    h_W1_k = h_W1 + x · (A_k^{W1})^T · (B_k^{W1})^T   // [B, N, D']  W1+LoRA
    h_W3_k = h_W3 + x · (A_k^{W3})^T · (B_k^{W3})^T   // [B, N, D']  W3+LoRA
    // SwiGLU activation
    h_gate = SiLU(h_W1_k) ⊙ h_W3_k                     // [B, N, D']
    // W2 + LoRA
    h_out_k = h_gate · W2^T + h_gate · (A_k^{W2})^T · (B_k^{W2})^T  // [B, N, D]
    // Router 加权累加
    h^l += h_out_k ⊙ r'[:, :, k:k+1]                   // 按 token 的路由权重

// 3. Residual connection
h^l = h^l + z^l

// Training Loss:
L_total = L_CE + a · N · Σ_{i=1}^{N} F_i · P_i
// F_i = 被路由到 expert i 的 token 比例
// P_i = router 分配给 expert i 的概率均值
// a = 1e-2, N = 8 (expert 数)
```

  - **性能优化要点**：
    - 朴素 MixLoRA：每 expert 独立执行 W1·x, SiLU, W2, W3·x 全流程 → 输入序列长时开销大。
    - 优化后：先对全输入计算共享的 W1·x 和 W3·x，再按 expert 切片分发；W2 因依赖 W1/W3 输出无法共享。
    - 多模型模式：M 个 MixLoRA 模型的输入 batch 合并，共享同一份预训练权重，各模型 router 独立路由各自 tokens。训练时 peak GPU memory 从 15.1GB 降至 8.8GB（LLaMA-2 7B + 2 models），推理时从 13.7GB 降至 7.2GB。
  - 单 token 计算延迟（LLaMA-2 7B, µs）：LoRA 245.3, DoRA 659.4, MixLoRA 535.2, †MixLoRA 462.5（优化后降低约 30%）。

## FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 FineEP，一种基于线性规划的 token scheduling 策略实现 MoE 训练中的细粒度 GPU 负载均衡。核心算法包括：
  1. **LPP-based Load Balancing（§5.1）**：将每 micro-batch 的负载均衡建模为线性规划问题。变量 `x_e^g`（expert e 在 GPU g 上的 replica load），约束 `Σ_g x_e^g = load_e`（每个 expert 的总 load 分配到其 replicas），目标 `min max_g Σ_e x_e^g`（最小化最大 GPU load）。使用 HiGHs 求解器在 CPU 单线程求解，利用 warm-start 跨 micro-batch 复用。
  2. **Locality-Aware Token Routing（§5.2, Algorithm 1）**：贪婪路由策略——优先将 GPU g 上的 tokens 路由到同在 GPU g 的 local replica（减少通信），再路由到 remote replica。
  3. **Graph-Theoretic Expert Placement（§6）**：
     - Symmetric Placement（§6.2）：无先验 load 知识时，用 Cayley graphs 构造对称 expert placement（保证图密度最小化 max induced subgraph density）。
     - Asymmetric Placement（§6.3）：已知 load 分布时，greedy 确定 replica counts + Monte Carlo sampling 确定 placement graph（选 max induced subgraph density 最小的图）。
     - Adaptive Replacement（§6.4）：后台监控 load 分布 → 时间序列预测 → Equation 3 评估 → 触发 placement 更新。
  4. **Communication-Aware Scheduling（Appendix A.1）**：扩展 LPP 目标函数为 `min comp + α·comm`，区分 intra-node (α₁) 和 inter-node (α₂) 通信权重。
  5. **Pipelining（Appendix A.2）**：将 tokens 拆分为 EP（前者）和 FineEP（后者）两部分，用 EP 的 all-to-all 通信覆盖 FineEP 的调度时间。
  实验比较 FineMoE vs Megatron-LM/SmartMoE/FlexMoE/DeepSpeed 的端到端吞吐量、负载均衡（Zipfian skewness s∈[0,2]）、执行时间分解、调度开销（vs experts/GPUs scaling）、ablation（warm solving/locality-aware routing/overlapping）。

- 硬件平台是什么，配置是什么。
  4 节点，每节点 8×NVIDIA H100 80GB SXM GPU（共 32 GPU），900 GB/s NVLink intra-node，2×400 Gbps InfiniBand NIC per node。BF16 精度。PP=节点数（仅 inter-node），DP=8, EP=4, d=2。禁用 TP。Selective activation recomputation（仅 MoE FFN）。Distributed optimizers（类 ZeRO-1）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - GPT 32×1.3B: 24 layers, h=2048, FFN_h=8192, 32 experts, top-2
    - GPT 16×3.2B: 16 layers, h=4096, FFN_h=16384, 16 experts, top-2
    - GPT 8×6.7B: 32 layers, h=4096, FFN_h=16384, 8 experts, top-2
    - Mixtral 16×2B: 32 layers, h=2048, FFN_h=8192, 16 experts, top-2
    - Mixtral 8×7B: 32 layers, h=4096, FFN_h=14336, 8 experts, top-2
  - 数据集：Wikipedia（预训练）。
  - Benchmark：端到端训练吞吐量(tokens/s)、max GPU load / avg GPU load（负载均衡指标）、dispatch time（通信性能）、调度时间（overhead）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未声明开源，经 web search 未发现公开仓库。基于 Megatron-LM（github.com/NVIDIA/Megatron-LM）实现。

  **FineEP Token Scheduling 算法伪代码**：

```
输入: {input_e^g} (GPU g 上分配给 expert e 的 token 数), expert placement {G_EDP^e}
输出: token-to-(GPU, replica) mapping

// === Step 1: LPP Solving (per micro-batch) ===
// Solve: min max_{g in G_FineEP} sum_{e: g in G_EDP^e} x_e^g
// s.t.   sum_{g in G_EDP^e} x_e^g = load_e, ∀e
//        x_e^g >= 0
// where load_e = sum_g input_e^g
// Warm-start: reuse previous solution as initial simplex state
{x_e^g} = HiGHs_solve_warmstart(LPP, {load_e}, prev_solution)

// === Step 2: Locality-Aware Token Routing (Algorithm 1) ===
{remain_input_e^g} = {input_e^g}
{remain_x_e^g} = {x_e^g}

for each expert e in E:
    // Phase A: Route local tokens to local replicas (reduce all-to-all)
    for each GPU g in G_EDP^e:
        y = min(remain_input_e^g, remain_x_e^g)
        route next y tokens of expert e from GPU g to local replica on GPU g
        remain_input_e^g -= y
        remain_x_e^g -= y

    // Phase B: Route remaining tokens to remote replicas
    for each GPU g in G_FineEP:
        for each GPU g' in G_EDP^e:
            y = min(remain_input_e^g, remain_x_e^{g'})
            route next y tokens of expert e from GPU g to replica on GPU g'
            remain_input_e^g -= y
            remain_x_e^{g'} -= y

// === Step 3: Distributed Execution ===
// All GPUs execute all-gather({input_e^g})
// Each GPU independently runs Steps 1-2 (deterministic algorithm)
// Each GPU produces identical token-to-replica dispatching plan

// === Step 4: Optional Communication-Aware Variant ===
// Extended LPP 4:
// minimize comp + α·comm
// where comp = max_g sum_e x_e^g
//       comm = max_g max(send_g, recv_g)
//       send_g = (sum_e input_e^g) - local_g
//       recv_g = (sum_e x_e^g) - local_g
//       local_g = sum_e min(x_e^g, input_e^g)
// For topology-aware: split α into α₁ (intra-node) and α₂ (inter-node)
```

  **Graph-Theoretic Expert Placement 算法**：

```
// === Symmetric Placement (no prior load knowledge, §6.2) ===
// Use Cayley graphs: group = (Z_{2^p}, +) or product groups
// Example: 8 GPUs, 8 experts → (Z_8, +), generators {1, -1} → cycle graph
// Example: 16 GPUs, 32 experts → (Z_4×Z_4, +), generators {(0,1),(0,-1),(1,0),(-1,0)} → 4x4 toroidal grid
// Property: all edges (experts) are uniformly distributed, minimizing max induced subgraph density

// === Asymmetric Placement (known loads, §6.3) ===
// Step 1: Greedy replica count allocation
heap = max-heap of (expert e, load_e / replica_count_e)
while remaining_replicas > 0:
    (e, max_load_per_replica) = heap.pop()
    replica_count_e += 1
    heap.push(e, load_e / replica_count_e)

// Step 2: Monte Carlo placement sampling
best_placement = nil, best_score = inf
for iter in 1..M:  // M Monte Carlo iterations
    placement = random_assign_experts_to_gpus(replica_counts)
    // Equation 3: compute max density
    m = max_{G_max subset G} (1/|G_max| * sum_{e: G_EDP^e subset G_max} load_e)
    if m < best_score:
        best_placement = placement
        best_score = m

// === Adaptive Replacement (§6.4) ===
// Every ~50 iterations:
predicted_loads = moving_average(historical_loads)
future_m = Equation3_simulate(current_placement, predicted_loads)
if future_m > threshold:
    new_placement = asymmetric_placement(predicted_loads)
    reinitialize_model_states(new_placement)  // migrate expert params + optimizer states
```

  **关键超参数与结果**：
  - FineEP d=2 (DP_degree/EP_degree = 8/4)
  - HiGHs solver: 单 CPU thread, LP 变量数 O(|E|d), 约束数 O(|E|+|G|)
  - Scheduling overhead: ~100 μs (min) 到 <1 ms (64 GPUs, 256 experts)
  - Warm-start LPP solving: 进一步减少求解时间
  - Adaptive replacement interval: 50 iterations（训练初期），数百 iterations（训练后期）
  - 端到端加速：最多 47.6% vs Megatron-LM，平均 36.9%，超 FlexMoE 13.9%
  - 负载均衡：s<1 时 FineMoE (w/o AR) 完美均衡；s>1 时 FineMoE (with AR) 借助 asymmetric placement 保持完美均衡
  - 调度额外开销：仅 0.4 ms dispatch time vs vanilla Megatron-LM

## Mixture of A Million Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出了 PEER（Parameter Efficient Expert Retrieval）层，使用 product key retrieval 技术实现从超过一百万（1024²）个 tiny expert（单神经元 MLP）中稀疏检索 top-k 专家。核心创新：(1) product key 将 N 个 d 维 key 分解为两组各 √N 个 d/2 维 sub-key 的笛卡尔积，将 top-k 检索复杂度从 O(Nd) 降至 O((√N + k²)d)；(2) 每个 expert 是 singleton MLP（仅一个隐藏神经元）：e_i(x) = σ(u_i^T x) v_i，权重存储在 Embedding 层中通过索引检索；(3) multi-head retrieval：h 个独立 query network 各自检索 k 个 expert，共享同一 expert pool，输出直接求和，等效于动态组装一个 h 神经元 MLP。
  - 实验通过 isoFLOP 分析（固定 FLOP 预算 6e18 和 2e19）比较 PEER vs Dense FFW vs Coarse-grained MoE（expert-choice routing, 128 experts）vs PKM（1024² memories, h=8, k=32）。在 C4 验证集上绘制 isoFLOP 曲线（模型大小 vs perplexity），并评估 compute-optimal 模型在 Curation Corpus、Lambada、Pile、Wikitext、C4 上的 perplexity。
  - Ablation 研究：(1) 变化总 expert 数量 N（128², 256², 512², 1024²）保持 hk=128 不变；(2) 变化 active expert 数量 hk（32, 64, 128, 256, 512）保持 N=1024² 不变，联合变化 h 和 k；(3) Query BatchNorm 对 expert usage 和 unevenness 的影响。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号或硬件配置。作者单位为 Google DeepMind，致谢中提到使用内部代码库训练模型，推测使用了 Google 内部 TPU/GPU 集群。
  - 训练配置：batch size=128, sequence length=2048。
  - 精度：论文未明确说明训练精度（BF16/FP32），推测为标准混合精度训练。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Transformer backbone，将中间 block 的 FFW 层替换为 PEER/MoE/PKM 层。模型大小通过变化层数、attention heads 和模型维度来控制（具体范围论文未列详细表格）。
  - PEER 配置：N=1024² experts, h=8 heads, k=16 experts/head, query BatchNorm 启用。MoE 配置：expert-choice routing, 128 experts, 每个 expert 大小与对应 dense 模型 FFW 相同。PKM 配置：1024² memories, h=8 heads, k=32 memories/head, query BatchNorm 启用。
  - 数据集：C4（预训练 isoFLOP 分析），Curation Corpus、Lambada、Pile、Wikitext、C4（语言建模评估）。Benchmark 指标：perplexity（验证集）。
  - Expert usage 评估指标：Expert Usage（被检索 expert 比例）、Unevenness（expert 分布与均匀分布的 KL 散度）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供专门的开源仓库。作者引用 PKM 实现为参考：https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb。PEER 的 get_indices 和 query_proj 函数实现可参考该 PKM 实现。
  - 算法 pipeline（基于论文 Algorithm 1 和公式）：

  **Step 1: Query 投影** — 输入 x ∈ R^(b×t×d_model)，通过 h 个独立 query network 映射为 h 个 query 向量 q^i(x) ∈ R^(b×t×d)，其中 d 为 product key 维度。

  **Step 2: Product Key 检索** — 将每个 query q 拆分为两个子查询 q_1, q_2 ∈ R^(d/2)。计算子查询与两组子密钥 C, C'（各含 √N 个 d/2 维向量）的内积：
  ```
  I_C = TopK({q_1^T c_i | c_i ∈ C})  # k 个候选子密钥索引
  I_C' = TopK({q_2^T c'_j | c'_j ∈ C'})  # k 个候选子密钥索引
  ```
  候选 product key 集合 K' = {(c_i, c'_j) | i ∈ I_C, j ∈ I_C'}，共 k² 个候选。计算每个候选 key 与完整 query 的内积 = q_1^T c_i + q_2^T c'_j，再次 TopK 选出最终 k 个 expert 索引。总复杂度 O((√N + k²)d)。

  **Step 3: Expert 权重检索** — 通过 Embedding 层按索引检索 expert 的 down/up projection 权重：
  ```python
  w_down = w_down_embed(indices)  # shape: (b, t, h, k, d_model)
  w_up = w_up_embed(indices)      # shape: (b, t, h, k, d_model)
  ```

  **Step 4: Expert 计算与聚合** — 每个 expert 为单神经元 MLP: e_i(x) = σ(u_i^T x) v_i:
  ```python
  x = einsum('btd, bthkd -> bthk', x, w_down)  # 等价于 u_i^T x
  x = activation(x)                              # σ 非线性
  x = x * softmax(scores)                        # router score 加权
  x = einsum('bthk, bthkd -> btd', x, w_up)    # 输出投影
  ```
  其中 scores 来自 query-key 内积经 softmax/sigmoid 归一化。h 个 head 的输出直接求和（已在 einsum 的 h 维度上隐式完成）。
  - 论文指出"efficient implementation may require specialized hardware kernels to accelerate embedding lookup and fusion with the einsum operations"，当前实现为 JAX 原型。

## Mixture-of-Experts with Expert Choice Routing

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 Expert Choice（专家选择）路由方法：与传统的 token-choice routing（每个 token 选择 top-k 专家）相反，让每个专家独立选择 top-k 个 token。该方法通过 Softmax 计算 token-to-expert affinity 矩阵 S，然后对 S^T 的每一行（每个专家）取 TopK，实现完美负载均衡并允许每个 token 获得可变数量的专家。
  - 实验比较：(1) EC-CF2（容量系数 c=2）vs Switch Transformer Top-1 gating vs GShard Top-2 gating，评估预训练 perplexity 收敛速度和下游 GLUE/SuperGLUE 11 任务 fine-tuning 性能；(2) 扩展专家数量（16→32→64→128）对 perplexity 的影响；(3) 变体：EC-CAP2/CAP3（限制每个 token 最多 2/3 个专家）vs 无约束 EC-CF2；(4) 对比 Hash Layer 路由；(5) 容量系数 ablation（c=0.5, 1, 2）；(6) 与同规模 Dense 模型的预训练比较。

- 硬件平台是什么，配置是什么。
  - 训练平台：Google TPU V4 chips。最大模型（8B/64E）使用 512 TPU V4 chips。
  - 使用 GSPMD 的 2D sharding 算法进行模型分区，充分利用 TPU 集群的 2D 拓扑。
  - 训练精度/优化器：Adafactor optimizer（β1=0, β2=0.99），无 auxiliary load balancing loss，dropout rate=0。

- 模型是什么。数据集和bench分别是什么。
  - 模型：基于 Transformer 架构，每两层替换一层 FFN 为 MoE 层。100M 规模系列（expert size=100M，专家数 16/32/64/128）和 8B/64E（8B activated params，每 token 9.8B，总参数 143B，32 layers, M=4096, H=16384, 32 heads, d_head=128）。非 MoE FFN 层使用 Gated Linear Unit (GLU) + GeLU。使用 per-layer relative positional bias。SentencePiece tokenizer（vocab 256K）。
  - 数据集：GLaM 数据集——1.6 trillion tokens，由高质量网页子集、书籍、Wikipedia、对话、论坛和新闻混合而成（详见 GLaM 论文 Table 3）。
  - Benchmark：GLUE 和 SuperGLUE 的 11 个任务——BoolQ, CB, CoLA, MNLI, MRPC, QNLI, QQP, RTE, SST2, WiC, WNLI。主要指标为 accuracy。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文本身来自 Google，未在论文中提供直接开源链接，但相关实现在 Google Research 仓库中：`google-research/sparse_mixers/routing.py` 包含 ExpertsChoose 路由逻辑（JAX/Flax）；Flaxformer/T5X 中也包含 "Experts choose" routing 机制。
  - 第三方 PyTorch 实现：`pytorch-mixtures`（`pip install pytorch-mixtures`）提供 `ExpertChoiceRouter`，可直接插入 MoE 层。

  **算法 pipeline（张量计算伪代码）**：

  ```
  输入: X ∈ R^{n×d}  (n = batch_size × seq_len, d = hidden_dim)
        W_g ∈ R^{d×e}  (专家嵌入矩阵, e = 专家数)
        c (容量系数), e (专家数)

  Step 1: 计算 token-to-expert affinity
      S = Softmax(X @ W_g)  ∈ R^{n×e}

  Step 2: 专家选择 token（对 S^T 的每行取 top-k）
      k = n × c / e  （每个专家的容量）
      G, I = TopK(S^T, k)
      # G ∈ R^{e×k}: 门控权重
      # I ∈ R^{e×k}: I[i,j] = 第 i 个专家选择的第 j 个 token 的索引
      P = OneHot(I)  ∈ R^{e×k×n}  (排列矩阵)

  Step 3: 按专家排列输入
      X_in = P @ X  ∈ R^{e×k×d}
      # X_in[i] ∈ R^{k×d}: 第 i 个专家的输入 token 集合

  Step 4: 每个专家独立计算 FFN
      for i in range(e):
          X_e[i] = GeLU(X_in[i] @ W_1[i]) @ W_2[i]^T
      # X_e ∈ R^{e×k×d}

  Step 5: 反排列回原始 token 顺序
      X_out[l, d] = Σ_{i,j} P[i,j,l] × G[i,j] × X_e[i,j,d]
      # 等价于: X_out = unshuffle(G ⊙ X_e, P)
  ```

  **可选的约束版本（EC-CAP）**：
  使用熵正则化线性规划限制每个 token 最多 b 个专家：
  ```
  max_A ⟨S^T, A⟩ + λH(A)
  s.t. Σ_j A[i,j] = k (每个专家选k个), Σ_i A[i,j] ≤ b (每个token最多b个专家)
  求解: Dykstra's 交替投影算法 (λ=0.001, max 100 iterations)
  然后: I = TopK(A, k)
  ```

  **使用例子（pytorch-mixtures）**：
  ```python
  from pytorch_mixtures.routing import ExpertChoiceRouter
  from pytorch_mixtures.moe_layer import MoELayer

  router = ExpertChoiceRouter(dim=768, num_experts=64)
  moe = MoELayer(
      num_experts=64,
      router=router,
      experts=experts,        # 64 个 FFN 专家
      capacity_factor=2.0     # 匹配 GShard top-2 的计算量
  )
  output = moe(input_tokens)  # input_tokens shape: (batch, seq, dim)
  ```

## MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  实现：在标准 MoE 层中引入三种零计算专家（zero expert、copy expert、constant expert）与 FFN 专家混合，构建异构 MoE 框架 MoE++。Zero expert 输出零向量（丢弃），copy expert 输出输入本身（跳过），constant expert 用可训练向量替换输入。同时引入基于 gating residuals 的 pathway-aware router（将前一层路由分数通过可训练矩阵 W_g 融入当前层）和异构负载均衡损失（超参数 τ 控制零计算专家与 FFN 专家的 token 分配比例）及异构专家容量分配。
  实验比较：MoE++ vs. vanilla MoE（相同参数量级别的标准 Top-2 MoE），比较下游 benchmark 准确率和 expert forward throughput。

- 硬件平台是什么，配置是什么。
  训练：4 节点 32× NVIDIA A100 GPU 集群。7B 模型使用 8-way pipeline parallel（tensor parallel=1）。小模型（0.6B/1B/2B）不使用模型并行。

- 模型是什么。数据集和bench分别是什么。
  模型：MoE++ 0.6B/(8+4)E、1B/(16+4)E、2B/(32+8)E、7B/(16+4)E，其中每个模型包含 1 zero expert + 1 copy expert + n_const constant experts，Top-K=2。baseline 为相同参数量级别的 vanilla MoE（Top-2，纯 FFN 专家）。所有 FFN 层替换为 MoE/MoE++ 层。
  数据集：RedPajama、Dolma、Pile 按不同采样比例混合。Tokenizer：LLaMA2（65,536 vocab）。训练 budget：100B tokens（Tab.3 所有模型）或 1T tokens（7B MoE++ 大模型）。
  Benchmarks：SciQ、PIQA、WinoGrande、ARC-E (0-shot)、HellaSwag (10-shot)、LogiQA (0-shot)、BoolQ (32-shot)、LAMBADA (0-shot)、NQ (32-shot exact match)、ARC-C (25-shot)、MMLU (5-shot)。使用 lm-evaluation-harness 评估。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码仓库 https://github.com/SkyworkAI/MoE-plus-plus（ICLR 2025，Apache 2.0）。当前仅发布推理代码和评估代码，训练代码待 Skywork-MoE 开源后一并发布。HuggingFace 上已发布 MoE++7B-Base 模型权重。
  算法 pipeline 伪代码（单个 MoE++ 层的 forward）：
  ```
  # 输入: x (shape: [B, S, D]), 前一层路由分数 G_prev (shape: [B, S, N])
  # N = N_FFN + N_ZC 总专家数, N_ZC = n_zero + n_copy + n_const

  # 1. Router with gating residuals
  logits = W @ x  # [B, S, N], W in R^{N x D}
  if layer_idx > 1:
      logits += W_g @ G_prev  # W_g in R^{N x N}, gating residuals
  G_curr = softmax(logits, dim=-1)  # 当前层路由分数

  # 2. Top-2 selection (考虑异构专家容量 C_i)
  # C_i = γ * τT / (τ*N_FFN + N_ZC) for FFN experts
  # C_i = γ * T / (τ*N_FFN + N_ZC) for zero-computation experts
  selected_indices, selected_probs = top_k_with_capacity(G_curr, k=2, capacities=C)

  # 3. Expert computation (异构专家)
  outputs = []
  for idx, prob in zip(selected_indices, selected_probs):
      if expert_type[idx] == FFN:
          out = FFN[idx](x)  # 标准 Feed-Forward
      elif expert_type[idx] == ZERO:
          out = 0  # 零输出
      elif expert_type[idx] == COPY:
          out = x  # 直通
      elif expert_type[idx] == CONST:
          alpha = softmax(W_c @ x)  # W_c in R^{2 x D}
          out = alpha[0] * x + alpha[1] * v  # v 是可训练向量
      outputs.append(prob * out)

  y = sum(outputs)  # 加权聚合
  # 异构负载均衡损失:
  L_b = sum_i η_i * f_i * P_i
  # η_i = 1 for FFN, τ for ZC expert
  # f_i = 选中频率, P_i = 平均 softmax 分数
  L_total = L_ce + 0.01 * L_b
  ```

  MoE++ 的计算复杂度仅为相同参数量 vanilla MoE 的 `τ*N_FFN / (τ*N_FFN + N_ZC)` 倍（Tab. 1）。典型 τ=0.75 时，MoE++ 0.6B/(8+4)E 的 expert forward throughput 从 535.3ms 降至 427.6ms（提升 25.2%），同时 average benchmark 从 44.3 提升至 45.6。τ=0.10 时 throughput 提升可达 164.5%。

## MoE-Compression: How the Compression Error of Experts Affects the Inference Accuracy of MoE Model?

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：首次将 error-bounded lossy compression（SZ3 for CPU、CuSZp for GPU）应用于 MoE 推理中的非激活 expert 压缩，以减少 PCIe offloading 的数据传输开销。核心方法为模拟压缩误差（以 Normal 分布 N~(0, ê) 随机生成），添加到 expert 参数上，从 7 个维度系统分析压缩误差对不同层次 expert 的推理精度影响：
    1. 单一 expert（单层内，如 expert-0 in layer 1）
    2. 最高频激活 expert（单层内，如 layer 1 的 expert-26）
    3. 不同层的最高频激活 expert（layer 1/13/20/26）
    4. Top-K 最高频激活 expert（layer 1/layer 26 的 top-6 experts）
    5. 单层全部 expert（layer 1/13/20/26 的全部 64 experts）
    6. 跨层 group 的最高频 expert（Group1: L1-L10, Group2: L9-L18, Group3: L17-L26，每组选 10 个最高频 expert）
    7. 跨数据集泛化（GSM8K → MATH dataset）
  - 误差 bound 设置为：ê = (10%/30%/50%/80% * ||θ_{ℓ,expert}||_1 / n_{ℓ,expert})，即 expert 参数 L1 范数平均值的百分比。
  - 评估指标：Instruction Compliance Accuracy (ICA，输出格式+内容均正确)、Pure Inference Accuracy (PIA，仅内容正确性，忽略格式)。
  - 实验比较：baseline（无误差注入的原始模型）vs 不同误差 bound 下的模型性能。论文也总结了量化方法的比较（Table 1: MC-MoE、MoE-CSP、MoQE、QMoE、CMoE、MoE-MPTQS、HOBBIT、EdgeMoE），但主要贡献是误差敏感性分析而非实现完整压缩系统。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明具体 GPU 型号。论文讨论的 motivation 场景为：GPU 内存有限 + PCIe offloading（如 PCIe 4.0 32 GB/s vs GPU 内部 300 GB/s on-chip bandwidth），推理阶段涉及 expert 参数在 GPU 内存和主内存之间的传输。
  - 推理实验的具体 GPU 配置论文未给出。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Moonlight（MoE 架构，26 个 expert layers，每层 64 expert submodules，inference 时每层 top-6 routing 激活 6 个 expert）。
  - 数据集：
    - GSM8K（数学推理 benchmark，作为主要分析数据集）
    - MATH dataset（Hendrycks et al. 2021，更难的数学数据集，用于泛化评估）
  - 评估指标：ICA (Instruction Compliance Accuracy)、PIA (Pure Inference Accuracy)。此外还自定义了 Imbalance Score、Expert Utilization、Entropy (Normalized)、Gini Coefficient 等指标来量化 expert 激活分布。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未提供开源代码链接。该工作发表于 SC'25 workshop。
  - 算法 Pipeline 伪代码（误差敏感性分析流程）：

```
# === 准备阶段：分析 Expert 激活分布 ===
# 在 GSM8K/MATH 数据集上对 Moonlight 模型做完整推理
for each question q in dataset:
    model.forward(q)
    for each MoE layer l in {1..26}:
        for each selected expert (top-6) in layer l:
            record expert_index, routing_weight
# 计算每个 expert 的总激活次数 ϕ_{l,e}
# 输出激活热力图（Fig. 3）和各层利用统计（Fig. 4）

# === 误差注入：模拟 error-bounded lossy compression 的误差 ===
# 对于给定 expert (layer l, expert e)，误差 bound ê：
θ = model.layers[l].experts[e].parameters
n = numel(θ)  # expert 参数总数
L1_avg = ||θ||_1 / n  # L1 范数平均值

# 生成遵循 Normal 分布的随机误差（模拟 SZ3/CuSZp 压缩误差分布）
ê = error_bound_pct * L1_avg  # 如 ê = 80% * L1_avg
errors = normal(mean=0, std=ê, shape=θ.shape)  # N(0, ê)

# 注入误差到 expert 参数
θ_perturbed = θ + errors

# === 评估：推理 + 指标计算 ===
output = model.forward_with_perturbed_experts(dataset)
# 计算 ICA: 检查输出格式（如 \boxed{}）和答案正确性
# 计算 PIA: 仅检查答案正确性，忽略格式要求

# === 跨实验维度 ===
# 1. 单一 expert 注入 (Section 3.2.1): expert-0 in layer 1
# 2. 最高频 expert (Section 3.2.2-3.2.3): layer 1/13/20/26 中 ϕ 最大的 expert
# 3. Top-K expert (Section 3.3): layer 1/26 中 ϕ 最大的 6 个 expert
# 4. 全层 expert (Section 3.4): layer 1/13/20/26 的全部 64 experts
# 5. 跨层 group expert (Section 3.5): Group1 L1-10/Group2 L9-18/Group3 L17-26
# 6. 跨数据集 (Section 3.6): 在 MATH dataset 上重复实验
```

- 关键实验发现（9 条 Takeaway）：
  1. 单一 expert 参数误差对推理影响小，但完全随机化参数导致严重退化——即使"不重要"的 expert 也 critical。
  2. 高频 expert 即使误差大（ê=80%），模型仍保持较高 PIA（如 layer 1 expert-26: PIA=0.95），误差首先影响 ICA 再影响 PIA。路由机制可自适应保护核心推理能力。
  3. 不同层的 expert 误差对性能影响呈非单调分布——shallow 层 ICA 降 10-20%，deep 层 ICA 反升 7-10%（layer 26 expert-40: ICA=0.96 vs baseline 0.86）。
  4. 浅层 expert 负责 attention + token→vector 转换（误差影响小），中层负责核心推理（误差影响最大），深层负责指令遵循+输出整合（可控误差可能带来增益）。
  5. 中层（layer 13）全层注入误差时 ICA 降至 0.38（vs baseline 0.86），说明中层对模型推理最关键。
  6. Deep layer 注入可控误差可提升性能——一种隐式集成效应（implicit integration effect），自动生成多样化 ensemble 提升鲁棒性。
  7. 多 expert 同时注入误差产生累积效应（layer 1 top-6: ICA 79%→74%），但 layer 26 top-6 的 ICA 仍高于 baseline（0.90 vs 0.85）。
  8. 误差传播呈非线性级联放大效应——cross-layer 注入（多组 expert 同时扰动）影响远超 single-layer。
  9. 当 ê=80% 时跨层 group 注入导致模型完全失效（所有 group 均无法输出有效结果），仅 ê≤50% 时才产生有效输出。

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

## MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Roster of Experts (RoE)，一种无需训练的 MoE 推理算法。核心为三个组件：
    1. **Gumbel-Top-K 随机路由**：在标准 MoE router logits $\mathbf{R} \in \mathbb{R}^E$ 上添加 Gumbel 噪声后做 TopK 选择——$\text{Indices} = \text{TopK}(\mathbf{R} + \tau \cdot \mathbf{G}, k)$，其中 $\mathbf{G}$ 为 Gumbel(0,1) i.i.d. 采样向量，$\tau$ 为逐层温度超参数。当 $\tau=0$ 时退化为标准确定性路由；$\tau>0$ 时产生受控随机性，等价于从 router 隐含的 categorical 分布中无放回采样 k 个 expert。
    2. **多路径聚合**：对每个 token 生成 n 个候选输出 logits（n 次独立的 Gumbel-Top-K 采样），通过概率平均（probability averaging）聚合成最终预测 logits。
    3. **Clean Cache 策略**：将 n 个样本的 forward pass 合并为一个 batch，batch 中第一个样本（index 0）使用 $\tau=0$ 确定性路由产生"clean path"，其 KV-cache 被其余所有样本共享。由此 KV-cache 内存开销与单样本完全相同。
  - 实验比较：(1) RoE vs baseline standard MoE greedy decoding 在三类模型的 12 个 benchmark 上的准确率（exact match 和 pass@1）；(2) 计算开销分析：不同样本数 K 下的 GPU 内存、功耗、延迟增长；(3) 效率对比：RoE with K=32 的 OLMoE-7B vs 等价的 10.5B 标准 MoE，对比延迟和内存。

- 硬件平台是什么，配置是什么。
  - GPU：NVIDIA A100 80GB（单卡）。RoE 计算开销实验使用单卡 A100 跑 GSM8K 前 100 题。
  - 超参数搜索框架：Optuna（TPE，Tree-structured Parzen Estimator）。

- 模型是什么。数据集和bench分别是什么。
  - 模型：OLMoE-1B-7B-Instruct（7B total, 1B active）、Mixtral-8x7B-Instruct-v0.1（47B total, ~13B active）、GPT-OSS-20B。
  - 数学推理 benchmark：GSM8K, SVAMP, AddSub, SingleEQ, MultiArith。
  - 常识推理 benchmark：ARC-Easy, ARC-Challenge, OpenBookQA, Social-I-QA, Hellaswag。
  - 代码生成 benchmark：HumanEval, HumanEvalPlus（pass@1）。
  - 效率分析数据集：WikiText-103（perplexity 评估 equivalent model size）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：论文未提供官方开源代码仓库。作者来自 Apple 和 UCSD，arXiv:2509.17238。
  - 算法 pipeline 伪代码（单 token 生成步骤）：

```
# ===== 单 Token 生成: RoE Forward =====
# 输入: hidden_state h (当前 token), KV-cache (共享)
# 参数: 逐层温度 tau[l] (经 Optuna TPE 调优), 样本数 n
# 输出: 下一个 token 的 logits

def roe_forward(h, kv_cache, tau, n, model):
    all_logits = []

    # Step 1: 将 n 个样本的 forward 打包为 batch
    batch_h = h.repeat(n, 1)  # (n, d_model)

    for layer in model.layers:
        # Step 2: Attention (共享 KV-cache)
        # Sample 0 使用 tau[l]=0 (clean path); 其余样本使用相同的 KV-cache
        if is_first_sample:
            out, new_kv = attention(batch_h[0], kv_cache)  # 单次计算
            kv_cache = new_kv  # 更新 clean KV-cache

        # Step 3: MoE Layer with Gumbel-Top-K Routing
        for each MoE layer l in model.moe_layers:
            router_logits = W_router[l] @ batch_h  # (n, E)

            # Gumbel noise for diversity
            for i in range(n):
                if i == 0 and use_clean_cache:
                    tau_eff = 0.0   # Clean path: deterministic
                else:
                    tau_eff = tau[l]  # Temperature from TPE tuning
                G = sample_gumbel(E)
                noisy_logits = router_logits[i] + tau_eff * G

                # Top-K 选择
                topk_vals, topk_idx = topk(softmax(noisy_logits), k)
                expert_out = 0
                for idx, weight in zip(topk_idx, topk_vals):
                    expert_out += weight * expert_ffn[idx](batch_h[i])
                batch_h[i] += expert_out

    # Step 4: Logit 聚合 (probability averaging)
    for i in range(n):
        all_logits.append(model.lm_head(batch_h[i]))
    final_logits = mean(softmax(all_logits), dim=0)  # 概率平均

    return final_logits, kv_cache

# ===== 温度搜索: Optuna TPE =====
# 搜索空间: 每 MoE 层一个 tau_i ∈ [0, 0.5]
# 中间层 (skip first/last k layers) 参与搜索
# 优化目标: validation perplexity (数学任务) 或 validation accuracy (常识/代码任务)

def search_temperature(model, val_data, task_type):
    def objective(trial):
        tau = []
        for l in range(L_moe):
            if l < skip_first or l >= L_moe - skip_last:
                tau.append(0.0)  # 首尾层固定为 0
            else:
                tau.append(trial.suggest_float(f"tau_{l}", 0.0, 0.5))
        score = evaluate_roe(model, val_data, tau)
        return score

    study = optuna.create_study(
        direction="minimize" if task_type == "math" else "maximize",
        sampler=TPESampler()
    )
    study.optimize(objective, n_trials=50)
    return study.best_params
```

  - 关键张量维度与计算：
    - Router logits: $\mathbf{R} \in \mathbb{R}^{E}$ (E 为 expert 数)
    - Gumbel 噪声: $\mathbf{G} \sim \text{Gumbel}(0,1)$，即 $G_i = -\log(-\log(U_i)), U_i \sim \text{Uniform}(0,1)$
    - 温度控制方程: $\text{Indices} = \text{TopK}(\mathbf{R} + \tau \cdot \mathbf{G}, k)$
    - 当 $\tau=0$ 时退化为 Standard Top-K；当 $\tau$ 中等时保留高 logit expert 被选中的优势（Gumbel-Max 性质）
    - Clean Cache: batch[0] 使用 $\tau=0$ 产生共享 KV-cache，batch[1:] 使用 TPE 调优的 $\tau_l$
    - 最终 logits: $\text{logits}_{\text{final}} = \text{softmax}^{-1}(\frac{1}{n}\sum_{i=1}^{n} \text{softmax}(\text{logits}_i))$
  - 超参数配置（Table 1）:
    - 数学任务：OLMoE N=32/T=0.5/L=1, Mixtral N=64/T=0.25/L=5, GPT-OSS N=64/T=0.2/L=5，PPL 优化
    - 常识任务：OLMoE N=32/T=0.5/L=3, Mixtral N=64/T=0.3/L=3, GPT-OSS N=64/T=0.2/L=5，Accuracy 优化
    - 代码任务：OLMoE N=32/T=0.5/L=1, Mixtral N=64/T=0.25/L=5, GPT-OSS N=64/T=0.2/L=5，Accuracy 优化


## MoH: Multi-Head Attention as Mixture-of-Head Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoH（Mixture-of-Head Attention）将 multi-head attention 中的每个 attention head 视为 MoE 框架中的 expert，通过一个 router 为每个 token 动态选择 Top-K 个 attention head 进行激活。核心组件：（1）**Heads as Experts**——将 h 个 attention head 视为 experts，router 对每个 token 产生 routing score g_i，仅 Top-K 个 routed head 被激活（g_i 非零），输出为加权求和 MoH(X, X') = Σ g_i · H^i · W_O^i；（2）**Shared Heads**——指定前 h_s 个 head 为共享 head（始终激活），捕获跨上下文的通用知识（如语法规则），剩余 h − h_s 个 head 为动态路由 head；（3）**Two-Stage Routing**——共享 head 的路由分数由 W_s 计算，路由 head 的路由分数由 W_r 计算并经 Top-K 筛选，再通过 W_h 产生 α_1, α_2 系数动态平衡两类 head 的权重；（4）**Load Balance Loss**——L_b = Σ P_i · f_i，防止路由坍塌到少数 head；（5）总 loss L = L_task + β · L_b（β=0.01）。
  - 实验比较：（a）**ViT 图像分类**（ImageNet-1K）：MoH-ViT-S/B 基于 TransNeXt 框架，仅替换 multi-head attention 为 MoH，与 DeiT、Swin、PVTv2、CoAtNet、FocalNet、CAFormer、TransNeXt 等对比 Top-1 Acc；（b）**DiT 图像生成**（ImageNet-1K 256×256）：MoH-DiT-S/B/XL 替换 DiT 中的 attention，对比 FID/sFID/IS/Precision/Recall；（c）**LLM 从头训练**：MoH-LLM-S(186M)/B(881M) 对比 vanilla LLM，6 个 benchmark（SciQ/PIQA/WinoGrande/OpenbookQA/LogiQA/TruthfulQA）；（d）**LLaMA3-8B Continue-Tuning**：MoH-LLaMA3-8B vs LLaMA3-8B，14 个 benchmark（MMLU/CEVAL/CMMLU/GSM8K/TruthfulQA/HellaSwag/LogiQA/BoolQ/LAMBADA/SciQ/PIQA/WinoGrande/NQ/ARC-C）；（e）**Ablation**：shared heads 消融、two-stage routing 消融、shared heads ratio 消融（13.9%~74.0%）、activated head ratio 消融（50%~80%）、inference time 对比（seq len 256/512，head num=32）。

- 硬件平台是什么，配置是什么。
  - ViT 训练：8 GPUs（论文未明确说明 GPU 型号，基于 TransNeXt 训练设置），自动混合精度（AMP）
  - DiT 训练：论文未明确说明 GPU 配置
  - LLM 从头训练：Megatron 框架，Tensor Parallel=1，Pipeline Parallel=1，batch size 4M tokens，序列长度 2048
  - LLaMA3-8B Continue-Tuning：Tensor Parallel=2（第一阶段）/ 1（第二阶段），Pipeline Parallel=1（第一阶段）/ 8（第二阶段），batch size 16M tokens，序列长度 8192

- 模型是什么。数据集和bench分别是什么。
  - 模型：MoH-ViT-S(50M)/B(90M) 基于 TransNeXt；MoH-DiT-S(33M)/B(130M)/L(458M)/XL(675M) 基于 DiT；MoH-LLM-S(186M, 12 layers, hidden=768, heads=12)/B(881M, 24 layers, hidden=1536, heads=16)；MoH-LLaMA3-8B（从 LLaMA3-8B continue-tune）
  - 数据集：ImageNet-1K（~1.2M images, 1000 classes）；LLM 训练用 RedPajama(Books 4.24%/Wikipedia 3.50%/ArXiv 4.37%/StackExchange 3.19%/C4 10.94%)、Dolma(61.28%)、Pile(12.48%) 按采样比例混合；LLaMA2 tokenizer（65,536 vocab）
  - Benchmark：ViT→ImageNet-1K Top-1 Acc；DiT→FID/sFID/IS/Precision/Recall；LLM→SciQ/PIQA/WinoGrande/OpenbookQA/LogiQA/TruthfulQA；LLaMA3→MMLU/CEVAL/CMMLU/GSM8K/TruthfulQA/HellaSwag/LogiQA/BoolQ/LAMBADA/SciQ/PIQA/WinoGrande/NQ/ARC-C

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码 https://github.com/SkyworkAI/MoH（Apache 2.0），预训练权重在 HuggingFace（Chat-UniVi/MoH-ViT-*、Chat-UniVi/MoH-DiT-*、Chat-UniVi/MoH-LLaMA3-8B）
  - 算法伪代码流程：
    ```
    # 输入: X ∈ R^{T×d_in} (T tokens, d_in dims)
    # 超参: h (总head数), h_s (共享head数), K (激活路由head数)
    
    # 1. 计算共享head路由分数
    s_s = Softmax(W_s @ X)  # W_s ∈ R^{h_s×d_in}, s_s ∈ R^{h_s×T}
    
    # 2. 计算路由head路由分数
    s_r = Softmax(W_r @ X)  # W_r ∈ R^{(h-h_s)×d_in}, s_r ∈ R^{(h-h_s)×T}
    
    # 3. Top-K 选择路由head
    topk_indices = TopK(s_r, K)  # 对每个token选K个路由head
    
    # 4. 两阶段系数
    [α_1, α_2] = Softmax(W_h @ x_t)  # W_h ∈ R^{2×d_in}
    
    # 5. 组装 routing score g_i
    for i in 1..h_s:     g_i = α_1 * s_s[i]
    for i in h_s+1..h:   g_i = (i in topk_indices) ? α_2 * s_r[i-h_s] : 0
    
    # 6. 计算每个head的attention输出
    for i in 1..h:
      Q_i = X @ W_Q^i, K_i = X' @ W_K^i, V_i = X' @ W_V^i
      H^i = Softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i
    
    # 7. MoH 加权求和输出
    MoH(X, X') = Σ_{i=1}^{h} g_i · H^i · W_O^i
    
    # 8. Load Balance Loss
    P_i = mean(Softmax(W_r @ X)[i-h_s])  对路由head
    f_i = mean(token选择head_i的indicator)
    L_b = Σ P_i * f_i
    
    # 总loss: L = L_task + 0.01 * L_b
    ```
    关键张量计算：对于每个 token x_t，router 计算 routing score 选择 Top-K head。shared head 始终参与计算，routed head 按需激活。输出是 activated head 的加权和。在 ViT 和 DiT 中，head 激活预算在各层不均匀分布——浅层激活较少 head，深层激活较多 head。

## MoLA: MoE LoRA with Layer-wise Expert Allocation

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：MoLA 提出一种层级别专家分配的 MoE-LoRA 参数高效微调方法。在 Transformer 的每层中，将 LoRA 适配器作为 MoE expert（即每层有多个低秩矩阵对 {A_i, B_i}），通过 router 进行 top-K 路由选择。关键创新是为不同 Transformer 层分配不同数量的 LoRA expert，而非传统 MoE 中各层 expert 数量相同。具体公式：S_i^{jt}(x) = TopK(Softmax(W_r^{jt}x), K)_i / Σ TopK(Softmax(W_r^{jt}x), K)_i，h^{jt} = W_0^{jt}x + Σ_{i=1}^K S_i^{jt}(x) A_i^{jt} B_i^{jt} x。对每个 dense weight matrix（attention 的 W_q/W_k/W_v/W_o 和 MLP 的 W_gate/W_down/W_up）都应用 LoRA expert。提出五种层级别配置：MoLA-△(8642, triangle，底层多 expert)、MoLA-▽(2468, inverted-triangle，高层多 expert)、MoLA-▷◁(8228, hourglass，底层和高层多)、MoLA-✸(2882, diamond，中层多)、MoLA-□(5555, rectangle，各层相同)。
  - 实验比较：（1）与 baseline PEFT 方法（Prompt Tuning、LLaMA-Adapter、LoRA rank=64）和 Full-Parameter Fine-tuning 在 6 个 benchmark 上的精度对比；（2）五种 MoLA 层级别配置之间的对比（每种总 expert 数相同，仅分配方式不同，总 config sum 为 20 或 16）；（3）极端配置（10-2-2-2 / 2-10-2-2 / 2-2-10-2 / 2-2-2-10）分析各层段 expert 冗余度；（4）Transfer Learning：instruction tuning → downstream fine-tuning；（5）Continuous Learning：跨 5 个 ScienceQA 领域连续学习，用 OP 和 PD 指标评估；（6）Frobenius Norm 分析各层 expert 相似度以量化冗余；（7）PiSSA 初始化方法的兼容性实验。

- 硬件平台是什么，配置是什么。
  - GPU: 8× NVIDIA A100-40G + 3× NVIDIA A6000
  - 训练时间: COLA 数据集约 4 小时
  - 精度: 论文未明确说明（推断为 Hugging Face Transformers 默认混合精度设置）
  - 分布式框架: 论文未明确说明（基于 Hugging Face Transformers + PyTorch 训练）

- 模型是什么。数据集和bench分别是什么。
  - 模型：LLaMA-2-7B（32 层）、Mistral-7B（32 层）、Gemma（28 层，Appendix E）
  - NLP 数据集：MRPC（5801 句对，二分类）、RTE（2490 train / 277 val，二分类）、COLA（8551 train / 1043 val，二分类）
  - Commonsense QA 数据集：ScienceQA（6508 train / 2224 test text-only）、CommonsenseQA（9740 train / 1221 val）、OpenbookQA（4957 train / 500 val / 500 test）
  - Instruction Tuning 数据：OpenOrca 随机采样 50,000 条

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源地址：https://github.com/GCYZSL/MoLA（179 stars，5 commits）
  - 开源内容：训练脚本（mola_training.py、mola_training_instruction.py）、评估脚本（evaluation_scienceqa.py）、推理 notebook、数据预处理脚本。基于 Hugging Face Transformers、mm-cot、alpaca-lora 构建。
  - 伪代码示意（MoLA 单层前向传播，对每个 linear module 执行）：
    ```
    # 超参数: rank=8, top_k=2, N_j=该层expert数
    # 对于第j层，预训练权重W_0冻结

    # 1. 原始预训练前向
    h_base = W_0 @ x  # [d_q, d_p] @ [B, L, d_p] -> [B, L, d_q]

    # 2. Router 计算每个 expert 的选择概率
    router_logits = W_r @ x  # W_r: [d_q, N_j], -> [B, L, N_j]
    router_probs = Softmax(router_logits, dim=-1)  # [B, L, N_j]

    # 3. Top-K 选择
    topk_vals, topk_idx = TopK(router_probs, K=2)  # 各 [B, L, K]
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)  # 归一化

    # 4. 每个 expert 计算 LoRA delta
    h_expert = 0
    for i in range(N_j):
        # A_i: [d_q, r], B_i: [r, d_p], r=8
        if i in topk_idx:
            delta_i = A_i @ B_i @ x  # [d_q, r] @ [r, d_p] @ [B,L,d_p] -> [B,L,d_q]
            weight_i = topk_vals[topk_idx == i]
            h_expert += weight_i * delta_i

    # 5. 输出 = 预训练 + LoRA expert 组合
    h = h_base + h_expert  # [B, L, d_q]

    # 6. Load balancing loss（每层计算）
    # f_i = 1/T Σ_t Indicator(token_t选择expert_i)
    # P_i = 1/T Σ_t router_probs[t][i]
    # L_balance = N_j * Σ_i f_i * P_i
    ```
  - 关键超参数：LoRA rank=8, top-K=2, LoRA alpha=16, LoRA dropout=0.05, optimizer=AdamW, lr=3e-4, batch_size=128, cutoff_length=256, epochs={10,15,20}, seed=10。可训练参数量（config sum=20, 即 5555 等配置）：105,635,840（LLaMA-2-7B 总参数的 ~1.5%）。

## Muon is Scalable for LLM Training

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：对 Muon 优化器进行三项关键扩展，使其适用于大规模 LLM 训练：(1) 引入 AdamW 风格的 weight decay（λ=0.1），解决原始 Muon 在大规模训练中权重 RMS 持续增长超出 bf16 范围的问题；(2) 提出 Consistent Update RMS 机制，对每个矩阵参数按 √(max(A,B)) 缩放更新量，使不同 shape 的矩阵参数具有一致的更新尺度，避免大矩阵（如 MLP 矩阵 [H, 2.6H] 或 [H, 4H]）更新过小而限制模型容量、小矩阵（如 GQA/MLA 中独立 KV head）更新过大导致训练不稳定；(3) 将 Muon 更新 RMS 匹配到与 AdamW 相同的 ~0.2 范围（scale factor = 0.2），使得 Muon 可以直接复用为 AdamW 调优的 learning rate 和 weight decay。分布式方面提出 Distributed Muon（Algorithm 1），基于 ZeRO-1 在 DP 组上分片 optimizer state，引入 DP Gather（bf16 全矩阵收集）和 Newton-Schulz 迭代在 bf16 精度下计算全矩阵更新，通信量为 Distributed AdamW 的 1~1.25 倍。
  - 实验比较：(a) Scaling Law 实验：在 399M~1.5B 参数 Llama 架构密集模型上，按 compute-optimal 设置对比 Muon vs AdamW，Muon 仅需约 52% 训练 FLOPs 即可匹配 AdamW 性能；(b) 大规模预训练：基于 DeepSeek-V3-Small 架构训练 3B/16B MoE 模型 Moonlight（5.7T tokens），对比 Moonlight-A（同架构+AdamW）和业界模型（Llama3.2-3B, Qwen2.5-3B, DSV2-Lite），Moonlight 在 MMLU 达 70.0 vs DSV2-Lite 58.3；(c) 消融实验：对比 Baseline（仅匹配 AdamW RMS）、Update Norm（直接归一化）和 Adjusted LR（按 shape 缩放）三种更新 RMS 控制策略；(d) SFT 实验：验证 pretrain 和 SFT 阶段优化器互换性，以及在 Qwen2.5-7B 上 SFT 时 Muon vs AdamW；(e) Spectral Analysis：通过 SVD entropy 分析，Muon 训练的权重矩阵具有更高的 SVD entropy，验证其提供更多样化的优化方向。

- 硬件平台是什么，配置是什么。
  - GPU 集群，支持 Megatron-LM 的 TP/PP/EP/DP 并行策略。具体 GPU 型号、数量、集群规模论文未明确说明。训练使用 bf16 混合精度。分布式 Muon 的 Newton-Schulz 迭代在 bf16 下计算，通信量相比 fp32 减半。

- 模型是什么。数据集和bench分别是什么。
  - Scaling Law 模型：Llama 架构密集模型，参数量从 399M 到 1.5B（不含 embedding），hidden size 1536~2560，层数 12~20，训练 tokens 8.92B~38.91B，batch size 96~256（8K context length）。Learning rate 8.3e-4~9.5e-4。
  - Moonlight 模型：基于 DeepSeek-V3-Small 架构的 MoE 模型，2.24B activated / 15.29B total params（含 embedding 为 3B/16B），使用 SwiGLU MLP、GQA、MLA。修改：去除 MTP、修改 auxfree bias 更新规则为 b_i = b_i + u × (sign(e_i) − sign(e).mean())、gate scaling factor=2.446。
  - 预训练数据：Moonshot AI 自研数据集（参见 K. Team 2025），最大 context length 8K。SFT 数据：tulu-3-sft-mixture（Lambert et al. 2024, 4K seq length）。
  - Benchmarks：English (MMLU 5-shot, MMLU-pro 5-shot, BBH 3-shot, TriviaQA 5-shot), Code (HumanEval pass@1, MBPP pass@1), Math (GSM8K 4-shot, MATH, CMATH), Chinese (C-Eval 5-shot, CMMLU 5-shot)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：分布式 Muon 实现将以 PR 形式贡献给 Megatron-LM（https://github.com/NVIDIA/Megatron-LM）；预训练 checkpoint、SFT checkpoint 和中间 checkpoint 均已发布。
  - 算法 Pipeline 核心（Muon + Weight Decay + Consistent Update RMS）：
  ```
  # 对每个矩阵参数 W ∈ R^{A×B}，每步迭代：
  # Nesterov momentum: 先外推再计算正交化
  M_t = mu * M_{t-1} + ∇L(W_{t-1})

  # Newton-Schulz 迭代 (N=5, a=3.4445, b=-4.7750, c=2.0315)
  X_0 = (mu * M_t + ∇L(W_{t-1})) / ||·||_F    # 注意：Nesterov 风格
  for k=1 to 5:
      X_k = a*X_{k-1} + b*(X_{k-1} @ X_{k-1}^T) @ X_{k-1}
            + c*(X_{k-1} @ X_{k-1}^T)^2 @ X_{k-1}
  O_t = X_5  # ≈ (M_t M_t^T)^{-1/2} M_t = U V^T

  # 更新：Matching AdamW RMS + Weight Decay
  W_t = W_{t-1} - lr * (0.2 * O_t * sqrt(max(A,B)) + lambda * W_{t-1})
  ```
## Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Nexus 提出一种增强型 MoE 架构，核心创新是**基于域嵌入的自适应路由器**：
    1. **Router 设计**：用 2 层 MLP（SwiGLU 激活）作为投影层 P_r，将预计算的域嵌入 d_i ∈ R^m（由 Cohere Embed v3 对每域名数据集编码后取平均得到）投影为专家嵌入 e_i ∈ R^h。路由概率通过 s_i = softmax(x · e_i) 计算，即输入 token 与各专家嵌入的点积相似度。这与超网络（hypernetwork）密切相关——投影层在运行时为给定输入生成路由参数。
    2. **Upcycling 阶段**：分别在不同域（ArXiv, Books, C4, StackExchange, Wikipedia）上独立训练 dense expert 模型，之后将各 expert 的 FFN 层沿新维度拼接为 MoE 层 FFN_{moe} = FFN_s + [FFN_e1, ..., FFN_en]。Seed 模型的原始 FFN 作为共享专家（始终激活），非 FFN 参数（attention 等）通过简单权重平均合并：φ_{moe} = Σ φ_i / n。
    3. **扩展阶段**：新域到来时，计算新域嵌入 d_new，通过已训练的投影层得到 e_new = P_r(d_new)，新 expert FFN 直接拼接到已有 expert 数组后，非 FFN 参数用加权平均 φ_f = (1-λ)·φ_moe + λ·φ_new（λ=1/(n+1)），然后用 1B token 做轻量微调。
  - 实验比较：
    1. **初始 Upcycling 性能**：对比 Dense Merging（BTM 风格等权平均）和 upcycled MoE with Linear Router（标准线性路由器的 upcycled MoE），在 470M 和 2.8B 两个 seed model 规模上评估 Knowledge / Science / Reasoning / MMLU 四类共 15 个下游任务。
    2. **扩展新 Expert（Code）**：在 2.8B seed model 的 upcycled MoE 上新增 Code expert（Starcoder 数据训练），比较 200M / 500M / 1B finetuning tokens 下 Nexus vs MoE (Linear Router) 的 Code 性能和通用任务性能。
    3. **Ablations**：load balancing loss factor 变化（0.05 vs 0.0005）、训练数据采样策略（按域大小比例 vs 均匀采样）、域嵌入投影前后的 cosine similarity 可视化。
    4. **Expert 专业化度量**：按域计算各 expert 的平均路由概率（routing frequency），验证 domain specialization 是否在 upcycling 后保持。

- 硬件平台是什么，配置是什么。
  - 训练平台：论文未明确说明具体 GPU 型号/数量
  - 精度：论文未明确说明（推测为 BF16 或 FP32）
  - 优化器：AdamW（论文提到使用 AdamW — 参考 Nemotron-4 的 recipe，但未单独列出 Nexus 使用的优化器；Section 4.1 提到 cosine decay schedule 但未指定优化器名）
  - 学习率：linear warmup 10% steps → max lr 1e-3 → cosine decay → 3e-4（dense expert 训练阶段）; cos decay to 3e-5（upcycling 最后 1B tokens）
  - 分布式框架：论文未明确说明

- 模型是什么。数据集和bench分别是什么。
  - 模型：Decoder-only autoregressive Transformer，470M 和 2.8B 参数两种规模的 seed model，使用 parallel attention layers、SwiGLU activation、no biases、BPE tokenizer（vocab 256k）。
    - 470M MoE：1 shared expert + 6 routed experts → total 1.3B params, 605M active（top-2 routing）
    - 2.8B MoE：1 shared expert + 4 routed experts → total 9.1B params, 4.3B active（top-2 routing）
  - 训练数据集：SlimPajama（627B token English corpus），包含 ArXiv, Books, C4, StackExchange, Wikipedia 子集，排除了 Github/StackExchange 用于后续 Code domain ablation
  - 扩展数据集：StarCoder code documents（Code expert 训练）
  - Dense expert 训练 token 量：470M scale 用 25B tokens/expert，2.8B scale 用 40B tokens/expert
  - MoE 训练 token 量：25B (470M) / 40B (2.8B)，最后 1B tokens 上做 upweight 原始预训练数据
  - Seed model 在 full SlimPajama 750B tokens 上训练
  - Benchmark：
    - Knowledge: OpenBookQA, Natural Questions, TriviaQA, QUAC (0-shot), SQuAD (4-shot)
    - Science: ARC-Easy, ARC-Challenge, SciQ (0-shot)
    - Reasoning: CommonSenseQA, SIQA, PIQA, WinoGrande, HellaSwag (0-shot)
    - General: MMLU (5-shot)
    - Code: MBPP, LBPP, HumanEval-Pack (Cpp, JS, Java, Go, Python, Rust) (0-shot)

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源情况：**未开源**。截至搜索日期（2026/05），无公开 GitHub 仓库或官方代码实现。论文在 Papers with Code 上显示 "No code implementations yet"。
  - 算法 pipeline 核心执行流程（基于论文 Section 3 及 Figure 2 伪代码）：

  **Phase 1: Dense Expert 训练**
  1. 从 seed model（已在 750B tokens 上预训练）初始化
  2. 对每个域（ArXiv, Books, C4, StackExchange, Wikipedia）独立训练 dense expert：用各自域的数据做 continue training（25B/40B tokens）
  3. 对每个域的数据用 Cohere Embed v3 编码所有样本，取平均得到域嵌入 d_i ∈ R^m

  **Phase 2: Upcycling（合并为 MoE）**
  1. MoE 层构造：
     - FFN_{moe} = concat([FFN_e1, ..., FFN_en]) along new dimension
     - Shared expert = FFN_seed（始终激活）
     - 非 FFN 参数（attention, norms, embedding）: φ_{moe} = mean(φ_1, ..., φ_n)
  2. Router 训练（每个 Transformer block 一个独立 router）：
     ```
     # 域到专家嵌入投影 (2-layer MLP, SwiGLU)
     # d_i: domain embedding [m], W1: [2h x m], W2: [h x h]
     expert_embeddings[i] = W2 @ SwiGLU(W1 @ d_i)  # [h]
     # 按 token 路由
     router_probs = softmax(inputs @ expert_embeddings)  # [batch, seq, n_experts]
     # Top-1 选路由专家 (+ shared expert 始终激活 = top-2)
     index, gate = topk(router_probs, k=1)
     # 输出
     out = shared_expert_ffn(inputs) + gate * routed_expert_ffns[index](inputs)
     ```
  3. 继续训练：用所有域 + 原始预训练数据的 mix 训练 25B/40B tokens，最后 1B tokens 上做 upweight 原始预训练数据 + cos decay lr to 3e-5

  **Phase 3: Extension（添加新 expert）**
  1. 用 StarCoder 数据训练新的 dense Code expert（8B tokens）
  2. 计算 Code domain embedding d_code，通过投影层得到 e_code = P_r(d_code)
  3. 追加 FFN_code 到 MoE 层，加权平均非 FFN 参数
  4. 轻量微调（up to 1B tokens）：data mix = 50% 旧域+预训练数据 + 50% Code 数据

  **张量流动**（以 upcycling 阶段一个 Transformer block 的 forward 为例）：
  - Input: x ∈ R^{s×h}（s 序列长度, h 隐藏维度）
  - Router 计算：预存 domain_embeddings ∈ R^{m×n} → 投影层 P_r (2-layer SwiGLU MLP) → expert_embeddings ∈ R^{h×n} → router_probs = softmax(x @ expert_embeddings) ∈ R^{s×n} → Top-1 gate → selected expert index ∈ Z^s
  - Shared expert: always → y_shared = FFN_seed(x) ∈ R^{s×h}
  - Routed expert: 按 index gather → y_routed = FFN_{index[i]}(x[i]) ∈ R^{s×h}
  - Output: y = y_shared + gate * y_routed ∈ R^{s×h}

  - 分布式 Muon (Algorithm 1) 张量流动：全梯度 G(fp32) → DP reduce-scatter 分片 → 本地动量更新(fp32) → DP gather 恢复全梯度矩阵(bf16) → Newton-Schulz N=5 迭代 → 取本地参数分片 → apply_update(p, u) with weight decay → DP all-gather 同步(fp32)。非矩阵参数（RMSNorm、LM head、embedding）仍用 AdamW。
  - Lemma 1：对 shape [A,B] 满秩矩阵，Muon 理论更新 RMS = √(1/max(A,B))。因此 √(max(A,B)) 缩放抵消此效应。

## Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models

- **属于算法pipeline的实现是什么？实验比较什么？**
  提出两种后训练 expert-level 稀疏化方法：(1) **Expert Pruning**：逐层枚举 expert 组合，以最小化 token 重建损失（Frobenius norm）选择保留 r 个 expert，永久丢弃 n−r 个不重要 expert；(2) **Dynamic Expert Skipping**：推理时根据 routing weight 比值 w_{e1}/w_{e0} < β 动态跳过次要 expert，β 从校准集每层中位数确定。实验比较：(a) 与 Wanda 2:4 结构化剪枝的性能/内存/速度对比；(b) 与 Random Pruning、Frequency-based Pruning 等 expert 剪枝 baseline 的 zero-shot 精度对比；(c) task-agnostic (C4 校准) vs task-specific (MATH 校准) 的 domain 效果对比；(d) expert pruning + dynamic skipping 组合的 LM-eval 精度与 token 生成速度 trade-off。

- **硬件平台是什么，配置是什么。**
  NVIDIA A100-80G GPU。原始 Mixtral 8x7B (bf16) 需 2 块 A100-80G 加载；prune 2 个 expert（r=6）后仅需 1 块 80G GPU；prune 4 个 expert（r=4）内存降至 46,879 MB。fine-tuning 实验使用 16 块 A100-80G GPU。推理速度测试基于 AutoGPTQ speed benchmark 脚本修改。

- **模型是什么。数据集和bench分别是什么。**
  模型：Mixtral 8x7B 和 Mixtral 8x7B Instruct。校准集：task-agnostic 用 C4（128 序列×2048 tokens），task-specific 用 MATH training set。Benchmarks：(a) EleutherAI LM Harness 8 项 zero-shot（ARC-c, ARC-e, BoolQ, HellaSwag, MMLU, OBQA, RTE, WinoGrande）；(b) GSM8K 5-shot；(c) MATH zero-shot；(d) fine-tuning 用 MetaMathQA（训练 900 steps, lr=2e-5, cosine scheduler）。

- **开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。**
  代码开源：https://github.com/Lucky-Lance/Expert_Sparsity。使用 HuggingFace Transformers，prune 后仅需修改 model config 即可加载。

  **Expert Pruning 伪代码（逐层）：**
  ```
  # 第1步: 用校准集对原始模型做推理，缓存每层 MoE 输入输出
  for each sample in calibration_set:
      for each MoE_layer l:
          cache: X_l (input tokens), Y_l = F_l(X_l) (original output)

  # 第2步: 逐层枚举 expert 组合
  for each layer l:
      best_loss = inf
      for each subset C of {expert_0,...,expert_{n-1}} with |C| = r:
          # 构建 prune 后 MoE 层 F'_l(·, C)，仅保留 C 中 expert 及对应 routing weight
          Y'_l = F'_l(X_l, C)
          loss = ||Y'_l - Y_l||_F   # Frobenius norm 重建损失
          if loss < best_loss:
              best_loss = loss
              best_C = C
      保留 best_C，丢弃其余 n−r 个 expert

  # 逐层拼接得到 r-expert MoE 模型
  ```

  **Dynamic Skipping 伪代码（推理时逐 token, top-2 场景）：**
  ```
  for each token x in sequence:
      计算 routing weights w = Softmax(l)
      选 top-2 expert: e0 (w_{e0} 最大), e1 (w_{e1} 次大)
      if w_{e1} < β * w_{e0}:    # β per-layer 超参，取校准集中位数
          仅使用 expert e0：z = E_{e0}(x)
      else:
          使用两个 expert：z = w̃_{e0}·E_{e0}(x) + w̃_{e1}·E_{e1}(x)
  ```

  张量计算流程：input token x ∈ R^{d} → Router 计算 logits l ∈ R^n → Softmax → w ∈ R^n → top-k 选择 → 对选中 expert e_j 计算 SwiGLU FFN：x → W_gate·x ⊙ SiLU(W_up·x) → W_down·(result) → output = Σ w̃_{e_j}·E_{e_j}(x)。Prune 后仅保留 r 个 expert，移除其他 expert 的权重矩阵及 routing weight。Dynamic skipping 在不修改模型参数的前提下运行时决定调用 1 或 2 个 expert。

## Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：提出两种量化 MoE 模型局部路由一致性（local routing consistency）的指标：（1）**Segment Routing Best Performance (SRP)**——基于 segment 的路由器对原始 token-level 路由器决策的逼近上限 F1 分数。对 single expert，定义激活序列为 binary classification tasks，segment estimator 在长度为 m 的 segment 内统一预测（全激活或全不激活），对所有 segment 计算 F1，当且仅当对所有 activation frequency ≥ α_e^m 的 segment 给出"激活"预测时 F1 最大化。对 expert group（layer 或 model level），用 segment router 预测 group 内所有 expert 的激活情况，同样求最大 F1。辅助指标 ρ̂（segment routing size ratio）衡量为达最佳 F1 所需激活 expert 数与原有激活数的比值。（2）**Segment Cache Best Hit Rate (SCH)**——模拟 oracle segment cache：缓存上限为 ρ·k（k 为原始每 token 激活 expert 数），驱逐未来 m 个 token 中激活次数最少的 expert，SCH 为其 hit rate，桥接 SRP 与实际 expert offloading 系统。
  - 实验比较：(1) 20 个 MoE LLM（3B-57B 参数）在不同 segment 长度 m（4/16/64/256）下的 SRP 和 ρ̂，按 SRP 将模型分为 4 组；(2) 11 个 TOY 模型（基于 OLMoE 修改，~1.43B 参数）验证 load balance、shared experts、expert combination space 等因素对 SRP 的影响；(3) 领域级（11 domains）SRP 与 expert specialization 分析（domain/prediction/vocabulary specialization）；(4) SCH 与实际 cache 算法（LRU、LFU）hit rate 的相关性（Pearson correlation），以及 SCH vs. 最优 Belady cache 的相对差距；(5) base vs. post-trained 模型 SRP 一致性；(6) layer-wise、position-wise、per-expert SRP 细粒度分析。

- 硬件平台是什么，配置是什么。
  - REAL 模型：NVIDIA A100 PCIe 80GB GPU（用于 router decisions 收集和 offloading throughput benchmark）。TOY 模型训练：基于 Megatron-DeepSpeed 框架，使用 OLMoE 预训练代码，序列长度 4096，全局 batch size 1024（~4M tokens/batch），10000 steps（约 40B tokens），learning rate cosine decay from 4×10⁻⁴ to 5×10⁻⁵，bfloat16 混合精度。

- 模型是什么。数据集和bench分别是什么。
  - REAL 模型（20 个，3B-57B total params）：LLaMA-MoE-v2（3.80B/8.03B act/total）、Yuan2.0-M32（3.70B/39.94B）、PowerMoE-3B（0.88B/3.30B）、Qwen3-30B-A3B（3.35B/30.53B）、Phi-3.5-MoE（6.64B/41.87B）、OLMoE-1B-7B（1.28B/6.92B）、GRIN-MoE（6.64B/41.87B）、Mixtral-8x7B（12.88B/46.70B）、MiniCPM-MoE-8x2B（4.32B/13.87B）、JetMoE-8B（2.33B/8.52B）、LLaMA-MoE-v1-3.5B（3.50B/6.74B）、XVERSE-MoE-A4.2B（4.23B/25.78B）、Jamba-Mini-1.6（12.11B/51.57B）、DeepSeek-V2-Lite（2.66B/15.71B）、DeepSeekMoE（2.83B/16.38B）、Qwen2-57B-A14B（14.25B/57.41B）、NLLB-MoE-54B（3.75B/54.50B）、Qwen1.5-MoE-A2.7B（2.69B/14.32B）、OpenMoE-8B（3.80B/11.86B）、SwitchTransformers-Base-128（0.22B/7.42B）。
  - TOY 模型（11 个，~1.43B total params，从 OLMoE 配置修改）：Baseline（8 layers, hidden=1280, 64 experts activate 8）、FewerExp（32 experts, activate 4）、ActMore/ActFewer（activate 16/2）、1ShrExp/2ShrExp（1 or 2 shared experts）、DenseFst/DenseHlf（第 1 层或第 1/3/5/7 层替换为 dense MLP）、NoLB（load balance loss coeff = 0）、OverLB（load balance loss coeff = 0.1）。
  - 数据集：从 RedPajama（C4、CommonCrawl、Books、Wikipedia、ArXiv、StackExchange、GitHub）和下游应用数据（LMArena arena-human-preference-140k、OpenMathInstruct-2、OpenCode-Instruct、OpenScienceReasoning-2）中抽取，每域 2048 个 512-token 样本，总计 22,528 输入样本。Benchmark 为 SRP 和 SCH 指标本身。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：https://github.com/ljcleo/moe-lrc（论文明确给出），承诺发布采样 corpus 和确定性实验配置。
  - 算法 pipeline 伪代码（SRP 计算）：
    ```
    # Input: MoE model M, corpus S (22,528 samples × 512 tokens), segment length m
    # Output: SRP(E, m), ρ̂

    for each sequence T in S:
        for each MoE layer l in M:
            record expert activation matrix A_l[T] where A_l[T][i][e] ∈ {0,1}

    # Step 1: Per-expert SRP
    for each expert e in each layer l:
        # count activation frequency f in every segment
        for each segment [p, p+m-1] in each T:
            f[e,T,p,m] = Σ_{i=p}^{p+m-1} A_l[T][i][e]

        # find α that maximizes F1 (proved in Appendix C.3)
        for α in [0, m]:
            TP_α = Σ f[e,T,p,m] for segments where f[e,T,p,m] >= α
            FP_α = Σ m for segments where f[e,T,p,m] >= α  minus TP_α
            FN_α = Σ f[e,T,p,m] for segments where f[e,T,p,m] < α
            F1_α = 2*TP_α / (2*TP_α + FP_α + FN_α)
        SRP(e, m) = max_α(F1_α)

    # Step 2: Expert group SRP (layer/model level)
    for each expert group E:
        # Joint optimization over all experts in E
        # Equations 5-6: F1 maximized iff segment router predicts "active"
        # for all (e, segment) where f[e,T,p,m] >= α_e^m
        SRP(E, m) = joint_max_F1_over_all_e_in_E

    # Step 3: ρ̂ computation
    ρ̂ = (avg predicted active experts at optimal F1) / (avg original active experts)
    ```

  - SCH 计算伪代码：
    ```
    for each layer l, segment length m, cache ratio ρ:
        cache_size = ρ * k  # k = number of active experts per token
        for each segment start position p in all T:
            cache = empty_set()
            for token t in segment[p : p+m]:
                demanded = top_k(router_weights_l[t])
                hit = True
                for expert e in demanded:
                    if e not in cache:
                        evict_k = e  # mark as missed for this expert
                        hit = False
                if not hit:
                    # evict experts least activated in remaining future of segment
                    future_activation_counts = count_activations_past_t(expert, T, t+1, p+m)
                    evict_experts = bottom_N(cache, future_activation_counts, N=|missed|)
                    cache = (cache \ evict_experts) ∪ missed_experts
                    record miss_count
                else:
                    record hit_count
        SCH = hit_count / (hit_count + miss_count)
    ```

## Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：Oracle-MoE 提出了一种全新的 MoE 路由机制，用基于 Oracle Space（语义组嵌入空间）的路由替代传统的 token-level 路由。核心流程分为以下阶段：
    - **Warm-up 阶段**：先对 token-level MoE 模型进行短期预训练，随机采样 N 条数据提取每层的语义组嵌入（Semantic Group Embedding），构成初始 Oracle Space。语义组的划分基于因果注意力分数矩阵：对连续 token，若注意力分数 a_ij > ε（阈值），则归入同一语义组。语义组嵌入为该组内所有 token 嵌入的平均值 z_S = (1/|S|) * Σ t_j。为提高计算效率，对 Oracle Space 中的嵌入做 SVD 降维。
    - **预训练/推理 Prefill 阶段路由**：在 Oracle Space 上运行 K-means 聚类（k = 专家数量），每个聚类中心对应一个专家。对每条新数据，先根据注意力分数划分语义组，计算语义组嵌入（用相同的 SVD 变换矩阵降维），计算该语义组嵌入到各聚类中心的欧氏距离，将距离最近的聚类对应的专家分配给该语义组的所有 token：e_t = argmin_k ||z_S(t) - c_k||。
    - **推理 Decode 阶段路由**：新 token 到来时，根据其与已缓存 token 的注意力分数决定所属语义组，更新该语义组嵌入，路由到该语义组对应聚类中心的专家。语义组变化缓慢，因此连续 token 往往路由到相同专家，大幅减少 expert swapping。
    - **Expert Prediction 优化（可选）**：用第一层的 embedding 预测后续层的专家激活，预测准确率达 85%-95%，进一步减少 10%-15% 的 expert loading 延迟。
  - 实验比较：(1) Expert Activation 模式对比——Oracle-MoE vs Switch Transformer，可视化连续 token 生成时的专家激活变化；(2) Memory-Latency 曲线——四种模型规模（195M/295M/729M/2.06B）在不同 memory budget 下对比 FIFO/LRU/SwapMoE 策略的每样本处理延迟；(3) First Token Latency——765M 模型在 50% 内存预算下对比各策略的首 token 延迟；(4) Downstream Task 性能——TriviaQA (F1)、GLUE (Acc)、MAG (Acc)、Sci-Cite (Acc)、XSum (Rouge-1) 上的零样本性能对比；(5) 激活不一致性——DeepSeekMoE-16B、Qwen1.5-MoE-A2.7B、Switch Transformer 和 Oracle-MoE 各层的激活不一致性对比；(6) 细粒度专家 MoE 扩展实验（3B 参数，12 MoE 层，64 experts，top-6 激活）。

- 硬件平台是什么，配置是什么。
  - NVIDIA Jetson Xavier NX（边缘设备）：384 核 NVIDIA Volta 架构 GPU，8 GiB GPU 内存，约 21 TOPS AI 算力。

- 模型是什么。数据集和bench分别是什么。
  - 模型：基于 GPT-2 架构的 MoE 模型，四种规模：
    - 2*4(195M)：12 层 Transformer，2 个 MoE 层，每层 4 个专家（top-1），hidden dim=768
    - 4*8(295M)：12 层 Transformer，4 个 MoE 层，每层 8 个专家（top-1），hidden dim=768
    - 8*16(729M)：12 层 Transformer，8 个 MoE 层，每层 16 个专家（top-1），hidden dim=768
    - 9*24(2.06B)：24 层 Transformer，9 个 MoE 层，每层 32 个专家（top-1），hidden dim=1024
    - 扩展实验：3B 模型，12 MoE 层，64 experts，top-6 激活，hidden dim=1536，expert intermediate dim=1024（仿 DeepSeekMoE 设计）
  - Baseline：Switch Transformer（token-level MoE routing）
  - 数据集：OpenWeb-Text（预训练）；下游任务——Trivia QA（问答）、GLUE（分类）、MAG（分类）、Sci-Cite（分类）、XSum（摘要）
  - Benchmark 指标：Expert Activation Variation、Memory-Latency Curve、First Token Latency、下游任务性能指标（F1/Accuracy/Rouge-1）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未明确说明开源链接，ICML 2025 论文页面无 GitHub 仓库链接。
  - 算法 Pipeline 伪代码：

    ```
    # === Oracle Space Initialization (Warm-up Phase) ===
    # 在 token-level MoE 短期预训练后执行一次

    oracle_space = []  # 收集所有语义组嵌入
    for each sampled_sequence in N_samples:
        # Step 1: 计算注意力分数矩阵
        A = compute_attention_scores(sequence)  # [T, T] lower-triangular

        # Step 2: 贪心划分语义组
        semantic_groups = []
        for t in range(T):
            # 找到最大的 j < t 使得 A[t][j] > epsilon
            # 且 token j 所在组内所有 token k 都有 A[t][k] > epsilon
            merged = False
            for group in reversed(semantic_groups):
                if all(A[t][k] > epsilon for k in group):
                    group.append(t)
                    merged = True
                    break
            if not merged:
                semantic_groups.append([t])

        # Step 3: 计算语义组嵌入
        for group in semantic_groups:
            z = mean(token_embeddings[group])  # z ∈ R^d
            oracle_space.append(z)

    # Step 4: SVD 降维
    U, S, Vt = SVD(oracle_space)  # 保留 top-r 奇异值
    W_svd = Vt[:r, :]  # 降维变换矩阵

    # Step 5: K-means 聚类
    reduced_embeddings = oracle_space @ W_svd.T
    cluster_centers = KMeans(reduced_embeddings, k=num_experts)

    # === Oracle-MoE Routing (Training & Prefill) ===
    # 对每个输入序列：
    def oracle_moe_forward(token_embeddings):
        # Step 1: 划分语义组（同上述贪心算法）
        groups = partition_semantic_groups(attention_scores, epsilon)

        # Step 2: 计算语义组嵌入并降维
        for group in groups:
            z = mean(token_embeddings[group])      # 组嵌入
            z_reduced = W_svd @ z                  # SVD 降维

            # Step 3: 最近邻聚类中心 -> 专家选择
            expert_id = argmin(||z_reduced - cluster_centers[k]|| for k in range(num_experts))

            # Step 4: 组内所有 token 路由到同一专家
            for token in group:
                router_probs[token] = one_hot(expert_id, num_experts)

        # 标准 MoE FFN 计算
        output = sum(router_probs[i] * expert_ffn_i(token_embeddings[i])
                     for i in range(seq_len))
        return output

    # === Decode Stage Routing ===
    def oracle_moe_decode(new_token, kv_cache):
        # Step 1: 计算新 token 与缓存 token 的注意力分数
        attn_scores = compute_new_token_attention(new_token, kv_cache)  # [1, len(cache)]

        # Step 2: 决定语义组归属
        assigned_group = None
        for group in existing_groups:
            if all(attn_scores[0][k] > epsilon for k in group):
                assigned_group = group
                break
        if assigned_group is None:
            assigned_group = create_new_group([new_token_idx])

        # Step 3: 更新语义组嵌入
        assigned_group.append(new_token_idx)
        z = mean(token_embeddings[assigned_group])
        z_reduced = W_svd @ z

        # Step 4: 路由到最接近聚类中心的专家
        expert_id = argmin(||z_reduced - cluster_centers[k]||)
        return expert_id

    # === Expert Prediction Optimization ===
    # 用第一层 embedding 预测深层专家激活
    def predict_deep_experts(first_layer_hidden):
        pred_experts = []
        for layer in range(1, num_layers):
            expert_pred = expert_predictor[layer](first_layer_hidden)
            pred_experts.append(expert_pred)
        return pred_experts  # 准确率 85%-95%
    ```

  - 张量计算核心：传统 token-level MoE 的 gate 为 g(t) = softmax(W_g * t) ∈ R^N，选择 top-k；Oracle-MoE 替换为 z_S(t) = (1/|S(t)|) * Σ t_j（语义组内平均），e_t = argmin_k ||W_svd * z_S(t) - c_k||（最近聚类中心）。对比之下，Oracle-MoE 的路由输入从 per-token embedding（受 token-identity 主导，高方差）变为 per-semantic-group embedding（保留高层语义，低方差），使得连续 token 的 CSD_oracle << CSD_token，从而减少 expert swapping。

## MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现是 MoEQuant 量化框架，包含 EBSS（Expert-Balanced Self-Sampling）和 AGQ（Affinity-Guided Quantization）两个插件式模块，可与 GPTQ、AWQ 等现有 PTQ 方法无缝集成。EBSS 利用 LLM 自采样能力，通过累积概率和专家平衡因子引导搜索，生成专家分布均衡的校准集；AGQ 将 token-expert 亲和力（gating coefficient）纳入量化误差计算和 Hessian 统计，改进逐层量化过程中的权重更新精度。
  - 实验比较了 FP16、RTN、AWQ、GPTQ 和 MoEQuant（基于 AWQ 和 GPTQ 的变体）在 4-bit 和 3-bit 权重量化下的表现，涵盖 PPL（WikiText2、C4）和 7 个下游任务（MMLU、HumanEval、GSM8K、BoolQ、HellaSwag、OpenBookQA、MathQA），并在 Qwen-MoE-14B-Chat 和 DeepSeek-MoE-16B-Chat 上验证了对 instruction-tuned 模型的量化性能。

- 硬件平台是什么，配置是什么。
  - NVIDIA A6000 GPU。所有实验在 NVIDIA A6000 上完成，不涉及微调。

- 模型是什么。数据集和bench分别是什么。
  - 模型：DeepSeek-MoE-16B、Qwen-MoE-14B（Qwen1.5-MoE-A2.7B-14B）、Mixtral-8x7B，以及它们对应的 instruction-tuned 版本（Qwen-MoE-14B-Chat、DeepSeek-MoE-16B-Chat）。
  - 校准集：WikiText2（baseline 方法所用，128 segments），EBSS 使用模型自采样生成校准集（branch number w=4，temperature τ=1.2）。
  - 评估数据集：WikiText2、C4（perplexity）；MMLU、HumanEval、GSM8K、BoolQ、HellaSwag、OpenBookQA、MathQA（下游任务）。复杂推理任务（MMLU、GSM8K、HumanEval）基于官方 repository 评估，其他 zero-shot 任务使用 lm-evaluation-harness v0.4.4。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文声称代码将开源于 https://anonymous.4open.science/r/MoEQuant-DDFD/README.md（ICML 2025 匿名仓库）。
  - 算法 pipeline：
    1. **EBSS 校准集生成**：给定 MoE 模型 M，设定 beam width w、sequence length n、温度 τ。初始化 w 个空序列 S={}。对每个 step i=1..n，对每个 beam S^t，从词汇表 V 中计算 score(S^t||v) = (-1/(i+1)) * (R_{S^t} + log P(v|S^t)) + σ(M, S^t)/τ。取 top-w 候选作为新的 beam。完成后得到 w 个长度为 n 的序列作为校准集 D*。此过程将搜索复杂度从 O(m^n) 降至 O(wn)。
    2. **AGQ - 亲和力感知量化误差**：传统 layer-wise 量化损失为 L(W_hat) = ||WX - W_hat X||_F^2。AGQ 将其重新定义为 L(W_hat) = Σ_i c_i · ||W x_i - W_hat x_i||_F^2，其中 c_i 是 token i 对该 expert 的 gating coefficient。对于 Hessian-based 方法（如 GPTQ），改进后的 Hessian 为 H = (X ⊙ √c)(X ⊙ √c)^T = (X ⊙ c) X^T，使高亲和力 token 在计算 sensitivity metrics 时贡献更大。
    3. **集成流程**：MoEQuant 首先用 EBSS 生成专家均衡校准集 D*，然后对每个 MoE 层中每个 expert 的权重矩阵，用 AGQ 改进的量化损失/Hessian 执行标准 GPTQ 或 AWQ 量化。量化采用 per-channel 对称均匀量化：Q(W) = clamp(⌈W/s⌋, q_min, q_max)，W_hat = Q(W)·s。
    4. **性能结果**：4-bit MoEQuant++ 相比 GPTQ 在 Qwen-MoE-14B 上平均分提升 0.59pts（49.59 vs 49.00），在 DeepSeek-MoE-16B 上提升 1.00pts（40.01 vs 39.01），在 Mixtral-8x7B 上提升 2.16pts（55.58 vs 53.42）。HumanEval 上 DeepSeek-MoE-16B 在 4-bit 下提升超 10 个点。3.2x 以上内存节省，1.2x 以上推理加速。
