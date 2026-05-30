## DeRS Towards Extremely Efficient Upcycled Mixture-of-Experts Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是 **DeRS (Decompose, Replace, Synthesis)** 范式，通过分解 upcycled MoE 专家为专家共享基础权重 + 专家专属 delta 权重，并对 delta 权重使用稀疏化或量化等轻量表示来提升参数效率。包含两种应用：
  
  **DeRS Compression**（推理阶段压缩）：对已训练的 vanilla upcycled MoE 模型，将 N 个专家权重 {W_1,...,W_N} 分解为共享基础权重 W_base 和 N 个 delta 权重 {Δ_1,...,Δ_N}（W_i = W_base + Δ_i），然后对 delta 权重应用后处理轻量技术：
  - DeRS-Sparsification：随机丢弃 delta 权重中比例为 p 的元素，以紧凑向量存储稀疏矩阵。MoE 层参数从 N·d·d_h 降至 (1+N·(1-p))·d·d_h
  - DeRS-Quantization：将 delta 权重从 K 比特量化为 k 比特。存储成本从 N·K 降至 K+N·k
  推理时按需合成专家权重：Ŵ_i = W_base + F_post(Δ_i)
  
  **DeRS Upcycling**（训练阶段高效 upcycling）：不复制原始 FFN N 次构建专家，而是将 N 个专家分解为一个可训练共享权重 W_shared 和一个专家专属增量权重 F_pre(Δ_i)，训练和推理时通过 W_i = W_shared + F_pre(Δ_i) 合成专家权重：
  - DeRS-SM（Sparse Matrix）：使用索引向量 I 和值向量 V 两个紧凑行向量表示稀疏矩阵，通过 torch.scatter 映射回去。训练参数从 N·d·d_h 降至 (1+N·(1-p))·d·d_h
  - DeRS-LM（Low-rank Matrix）：使用两个低秩矩阵 A∈R^{d×r} 和 B∈R^{r×d_h} 表示增量权重，F_pre(Δ_i) = A_i·B_i。训练参数从 N·d·d_h 降至 d·d_h + N·r·(d+d_h)
  
  实验比较（三大任务六种 MoE 架构）：
  
  **通用多模态任务（MoE-LLaVA）**：
  - DeRS Compression：在 MoE-LLaVA-StableLM/Qwen/Phi 上对比不同 drop rate (0.2~0.99) 和量化位宽 (8/4/2/1 bit) 下的性能。drop rate 0.9 时 MoE 层参数减少 65% 无性能损失；2-bit 量化可达 16→2 位宽降低。极端设置 (0.99 drop rate / 1-bit) 仍无明显性能退化
  - DeRS Upcycling：对比 Vanilla Upcycling vs DeRS-SM vs DeRS-LM 在三个 MoE-LLaVA 架构上的 Added Params 和 Overall 性能
  
  **医学多模态任务（Med-MoE）**：
  - DeRS Compression：在 Med-MoE-StableLM/Phi 上对比不同压缩率。极端压缩 (remove 99% 元素或 1-bit 量化) 对性能影响可忽略
  - DeRS Upcycling + Extended DeRS Upcycling（扩展至 universal FFN）
  
  **代码生成任务（Coder-MoE）**：
  - DeRS Compression：delta 权重冗余度比医学任务低（因 dense model 未进行先验微调），drop rate 0.6 或 2-bit 量化无性能损失
  - DeRS Upcycling + Extended DeRS Upcycling
  
  **消融实验**：
  - 冻结共享基础 FFN vs 不冻结：冻结导致 DeRS-SM 下降 1.3%，DeRS-LM 下降 1.6%
  - 稀疏率/秩超参数扫描：DeRS-SM 低稀疏率更好；DeRS-LM rank 1/4/16 效果相近，rank 64 性能退化
  - 成本分析：DeRS-LM (4rank) 减少模型大小 52.7%、训练内存 21.2%、推理内存 43.8%，性能提升 0.7%

- 硬件平台是什么，配置是什么。
  - 通用多模态任务：8× NVIDIA A100 80GB，Bfloat16 精度，Training Batch size per GPU=4，Gradient Accumulation Steps=4
  - 医学多模态任务：4× NVIDIA A100 80GB，Bfloat16 精度，Training Batch size per GPU=8，Gradient Accumulation Steps=2
  - 代码生成任务：8× NVIDIA A100 80GB，Bfloat16 精度，Training Batch size per GPU=4，Gradient Accumulation Steps=2
  - 使用 CFFF platform of Fudan University

- 模型是什么。数据集和bench分别是什么。
  **模型**：
  - MoE-LLaVA 框架：CLIP-Large 视觉编码器 + 语言骨干（StableLM-2-1.6B / Qwen-1.8B / Phi-2-2.7B），每间隔一个 block 的 FFN 层 upcycled 为 4 专家的 MoE 层，top-2 激活
  - Med-MoE 框架：CLIP-Large 视觉编码器 + 语言骨干（StableLM-2-1.6B / Phi-2-2.7B），每间隔一个 block 的 FFN 替换为 universal FFN + 4 专家 MoE 层的并行结构，top-1 激活
  - Coder-MoE 框架：DeepSeek-Coder-Base-1.3B，每个 block 的 FFN 替换为 universal FFN + 4 专家 MoE 层的并行结构，top-1 激活
  
  **数据集与 Benchmark**：
  - 通用多模态：微调 LLaVA-mix-665k；评估 VQA-v2, GQA, VisWiz, ScienceQA-IMG, TextVQA + POPE, MMBench, MM-Vet
  - 医学多模态：微调/评估 VQA-RAD, SLAKE, PathVQA（含开放/封闭式问答）
  - 代码生成：微调 evol-codealpaca-v1 (110K instruction-output pairs)；评估 HumanEval, HumanEval+, MBPP, MBPP+

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供开源代码链接，arXiv 页面亦无代码可用声明。以下基于论文描述给出算法 pipeline 伪代码：
  
  **DeRS Compression 流程**：
  ```
  # 输入：已训练的 vanilla upcycled MoE 模型，包含 N 个专家 {W_1,...,W_N}
  # 输出：压缩后的 MoE 模型
  
  # Step 1: Decompose
  W_base = 原始 FFN 权重  # shape: [d, d_h]
  for i in 1..N:
      Δ_i = W_i - W_base  # shape: [d, d_h]
  
  # Step 2: Replace (Sparsification 示例)
  for i in 1..N:
      M_i ~ Bernoulli(p)  # shape: [d, d_h]，每个元素独立 Bernoulli
      F_post(Δ_i) = (1 - M_i) ⊙ Δ_i / (1-p)  # 元素级 drop + rescale
      # 以紧凑格式存储：仅保存非零元素的值和索引
  
  # Step 3: Synthesis（推理时按需合成）
  # 当 Router 选择 expert E_k 时：
  Ŵ_k = W_base + F_post(Δ_k)  # shape: [d, d_h]
  y = x @ Ŵ_k  # 使用合成权重进行 FFN 计算
  ```
  
  **DeRS Upcycling (DeRS-LM) 流程**：
  ```
  # 输入：预训练 Dense 模型的 FFN 权重 W_orig，shape [d, d_h]
  # 输出：参数高效的 MoE 模型
  
  # 初始化
  W_shared = W_orig.clone()  # 专家共享基础权重，可训练
  for i in 1..N:
      A_i = random_init([d, r])  # 低秩矩阵 A，可训练
      B_i = zeros([r, d_h])      # 低秩矩阵 B，可训练（零初始化确保初始 Δ=0）
  
  # 训练/推理时合成专家权重
  for each input x:
      Router_score = TopK(softmax(x @ W_R), k)  # 路由计算
      for i in selected_experts:
          F_pre(Δ_i) = A_i @ B_i          # shape: [d, d_h]，低秩分解
          W_i = W_shared + F_pre(Δ_i)      # 合成专家权重
          y_i = FFN(x, W_i)                # 使用合成权重计算
      y = Σ Router_score_i * y_i
  
  # 可训练参数对比：
  # Vanilla: N * d * d_h 个参数
  # DeRS-LM: d * d_h + N * r * (d + d_h) 个参数
  # DeRS-SM: d * d_h + N * d * d_h * (1-p) 个参数（紧凑存储）
  ```
  
  以 MoE-LLaVA-Phi (d=2560, d_h=10240, N=4) 为例：
  - Vanilla Upcycling：4 × 2560 × 10240 ≈ 2.52B 新增参数
  - DeRS-SM (p=0.9999)：2560×10240 + 4×(2560×10240×0.0001) ≈ 26.2M + 0.26M ≈ 1.11M 有效新增参数（2270× 减少）
  - DeRS-LM (r=1)：2560×10240 + 4×1×(2560+10240) ≈ 26.2M + 2.42M = 2.42M 新增参数（1041× 减少）
