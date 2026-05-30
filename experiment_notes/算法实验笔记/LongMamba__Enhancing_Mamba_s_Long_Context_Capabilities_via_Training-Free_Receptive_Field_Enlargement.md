## LongMamba__Enhancing_Mamba_s_Long_Context_Capabilities_via_Training-Free_Receptive_Field_Enlargement

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是LongMamba——一种training-free技术，通过扩大Mamba模型中全局通道（global channels）的感受野来增强长上下文能力。核心创新分两步：(1) Channel Classification：基于训练长度上的累积衰减 ∏_{k=1}^L Ā_k 将隐藏状态通道分类为全局通道和局部通道——若累积衰减 > θ 则归类为全局通道；(2) Receptive Field Enlargement via Token Filtering：对于识别的全局通道，过滤掉Δ_t低于阈值g(S)的token（不更新也不衰减隐藏状态），使目标序列长度上的累积衰减与训练长度对齐：∏_{i=1}^S Ā'_i ≈ ∏_{i=1}^L Ā_i。实现时从训练集（Pile）采样序列标定Δ_t分布，预先构建per-channel查找表g(S)，以1000-token为间隔离线计算。Token过滤规则：若Δ_t < g，设置(Ā'_t, B̄'_t) = (1, 0)使H_t = H_{t-1}（跳过该token）；否则正常更新。
  实验比较：(a) Language Modeling：PG-19数据集上评测perplexity，Mamba-1.4B、Mamba2-1.3B、Zamba2-1.2B，序列长度最高60k tokens；(b) RULER合成数据集：13个长上下文任务，16k/24k/32k序列长度，对比vanilla模型；(c) LongBench-E：13个真实世界长上下文应用（Single-doc QA、Multi-doc QA、Summary、Few-shot、Coding等），对比vanilla模型和DeciMamba；(d) 消融研究：不同channel selection阈值θ影响（表3），不同标定序列组影响（表4，STD 0.42%）；(e) LongBench：Falcon-Mamba-7B上对比Llama2-7B-chat-4k、XGen-7B-8k、Vicuna-v1.5-7B-16k Transformer baselines；(f) 延迟开销：A5000和A100 GPU上prefilling延迟测量，4k-96k tokens。

- 硬件平台是什么，配置是什么。
  NVIDIA A5000和NVIDIA A100 GPU。延迟测量使用batch size=1的prefilling场景，评测4k到96k tokens序列长度。训练相关：使用预训练模型的官方checkpoint直接加载，不进行任何微调或参数调整。标定过程随机采样5个来自Pile数据集的序列。论文未明确说明训练原始模型所用的GPU型号（模型为开源预训练模型：Mamba-1.4B、Mamba2-1.3B、Zamba2-1.2B、Falcon-Mamba-7B）。

- 模型是什么。数据集和bench分别是什么。
  模型：(1) Mamba-1.4B（Gu & Dao, 2023），训练长度2k tokens；(2) Mamba2-1.3B（Dao & Gu, 2024a），训练长度2k tokens；(3) Zamba2-1.2B（Glorioso et al., 2024a），混合Transformer-SSM模型，训练长度4k tokens；(4) Falcon-Mamba-7B（Zuo et al., 2024），8B参数纯SSM。所有实验直接加载官方模型checkpoint，无微调。
  数据集：PG-19（语言建模perplexity评测）、RULER（13个合成长上下文任务，包括passkey retrieval、question answering等，每任务/每长度生成100条序列）、LongBench-E（13个真实世界长上下文任务：Passage Count, PassageRetrieval-en, GovReport, MultiNews, MultiFieldQA-en, Qasper, 2WikiMQA, HotpotQA, SAMSum, TREC, TriviaQA, LCC, RepoBench-P）、LongBench（更多任务额外评测）。标定数据集：Pile（Gao et al., 2020）。
  Benchmark：PG-19 perplexity评测；RULER各任务accuracy；LongBench-E各任务accuracy（按Single-doc QA、Multi-doc QA、Summary、Few-shot、Synthetic、Coding类别分组）；LongBench各任务accuracy。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  完全开源。代码：https://github.com/GATECH-EIC/LongMamba。基于PyTorch实现，直接加载HuggingFace上的预训练Mamba/Mamba2/Zamba2模型。

  LongMamba算法pipeline核心——推理时的两阶段前向传播（对每一层Mamba block）：
  ```
  # Phase 0: 离线标定阶段（仅运行一次）
  Input: 从Pile数据集随机采样5条序列（各训练长度L）
         候选clamping值C ∈ {0,5,10,15,20}
         候选阈值θ ∈ {10^-40, 10^-30, ..., 10^-1, 5×10^-1}

  For each 通道 c in d_e (hidden state channels):
    # 1. 计算训练长度L上的累积衰减（Eq.12）
    decay_c = ∏_{k=1}^L Ā_k[c]  # ∈ (0,1), 沿d_s维度取平均

    # 2. 通过grid search在LongBench-E上选最优θ确定全局/局部分类
    if decay_c > θ:
      标记c为全局通道(global channel)
    else:
      标记c为局部通道(local channel)

    # 3. 对每个全局通道标定Δ_t分布
    For each timestep t in 采样序列:
      记录 Δ_t[c] 值
    # Clamp极值到top C%最大值的边界
    clamp_threshold = percentile(Δ_values, 100 - C)
    Δ_t_clamped[c] = min(Δ_t[c], clamp_threshold)

    # 4. 构建per-channel查找表g_c(S)以1000-token为间隔
    For S = 1000, 2000, 3000, ..., max_context:
      找到g使得: ∏_{i=1}^S Ā'_i(g)[c] ≈ ∏_{i=1}^L Ā_i[c]
      # 在假定的Δ_t分布下数值求解g
      g_c[S] = optimal_g
  ```

  ```
  # Phase 1: 推理时修改Mamba SSM前向传播
  # 原Mamba计算（Eq.4-5）:
  #   H_t = Ā_t ⊙ H_{t-1} + B̄_t ⊙ X_t
  #   Y_t = C_t^T H_t
  # LongMamba修改（仅对全局通道）:

  Input: 输入序列X ∈ R^{S×d_e}, S > L (训练长度)
         每个通道c的全局/局部标签
         查找表 g_c(S)  (S向下取整至最近1000-token间隔)

  For each Mamba layer l:
    # 标准Mamba预处理
    X_input = σ(Conv1D(Linear_1(I)))  # ∈ R^{S×d_e}
    Δ, B, C = compute_from(X_input)   # 标准Mamba投影 (Eq.6)
    Ā = exp(Δ ⊙ A)                    # 标准decay计算 (Eq.7)
    B̄ = Δ ⊗ B                         # 标准输入投影 (Eq.7)

    For each timestep t = 1..S:
      For each 通道 c in d_e:
        if 通道c是全局通道:
          # Token filtering (Eq.14)
          if Δ_t[c] < g_c(S):
            Ā'_t[c] = 1               # 不衰减
            B̄'_t[c] = 0               # 不更新
          else:
            Ā'_t[c] = Ā_t[c]           # 正常衰减
            B̄'_t[c] = B̄_t[c]           # 正常更新
        else:  # 局部通道
          Ā'_t[c] = Ā_t[c]             # 保持原值
          B̄'_t[c] = B̄_t[c]             # 保持原值

      # 修改后的隐藏状态更新 (Eq.4 with filtering)
      H_t = Ā'_t ⊙ H_{t-1} + B̄'_t ⊙ X_t
      # 当Δ_t[c] < g_c(S)时: H_t[c] = H_{t-1}[c] (直接传递历史状态)

    # 标准Mamba输出 (Eq.3)
    Y = SSM_filtered(X_input)  # 使用修改后的Ā', B̄'替代Ā, B̄
    O = Linear_3(σ(Linear_2(I)) ⊙ Y)

  Output: O ∈ R^{S×d_m}
  ```

  LongMamba的超参数搜索策略：
  ```
  # θ (channel selection threshold)
  # 在LongBench-E上独立搜索各模型:
  #   候选: {10^-40, 10^-30, 10^-20, 10^-10, 10^-5, 10^-4, 10^-3,
  #           10^-2, 5×10^-2, 10^-1, 5×10^-1}
  #   Mamba-1.4B:  θ = 10^-30
  #   Mamba2-1.3B: θ = 5 × 10^-2
  #   Zamba2-1.2B: θ = 10^-5

  # C (Δ_t clamping percentile)
  #   候选: {0, 5, 10, 15, 20}
  #   Mamba2-1.3B & Zamba2-1.2B: C = 5
  #   Mamba-1.4B: C = 20

  # g(S) lookup table
  #   间隔: 1000-token (1000, 2000, 3000, ...)
  #   输入S先向下取整到最近1000-token间隔再查表
  ```

  关键数值示例——累积衰减对齐（Eq.13）：
  - 训练长度L=2000：∏Ā_i ≈ 某个值（取决于通道）
  - 测试长度S=16000：∏Ā_i 若不过滤 ≈ 接近0（八倍衰减）
  - LongMamba通过过滤Δ_t<g的token（约占(1-L/S)的比例）使筛选后的∏Ā'_i仍然≈∏_{trained}_Ā_i

  延迟开销：A5000上prefilling延迟增加≤4.5%（表6），A100上prefilling延迟增加≤3.8%（表7），均为batch size=1场景。
