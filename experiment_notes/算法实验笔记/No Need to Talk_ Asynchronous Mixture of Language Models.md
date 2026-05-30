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
