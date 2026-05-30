## WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 WindowKV —— 一种任务自适应的 KV cache 窗口选择方法。核心实现包含三个组件：(1) **Task-Adaptive Window Selection**：将输入 context 分为 observation window（最后 α 个 token）和 review windows（其余 context 按固定窗口大小 ω 切分）。使用 observation window 对 review context 中各 token 计算累积注意力分数 t_j = Σ A_ij（i ∈ observation window），再按窗口聚合为 window-level score s_k = (1/min(p,ω)) · sum(Top-p(W_k))。训练一个 bert-base-cased 任务自适应分类器（9551 样本，8:1:1 划分）将输入 context 分为 Information Localization 任务（QA 类，p=ω，保留整个窗口以理解完整语义）和 Information Aggregation 任务（摘要/代码类，p<ω，仅保留窗口内 top-p 高注意力 token）；(2) **Intra-Group Layer KV Cache Indices Sharing**：将 m 层 Transformer 分为 H=m/γ 组，每组 γ 层。仅在每组第一层 l_g 执行完整的 task-adaptive window selection，组内其余层共享同一套 KV cache indices I_lg，大幅降低计算开销；(3) **Dynamic Budget Allocation**：受 PyramidKV 启发，按等差数列跨组分配 budget——底层组分配更多 budget，顶层组分配更少，形成金字塔结构。总 budget b^total 分布在 H 个组上，通过超参数 λ（实验中 λ=14）控制金字塔形状。

  实验比较：(1) LongBench benchmark 上对比 StreamingLLM (SLM)、H2O、PyramidKV (PKV)、Full KV (FKV)，在 Qwen2.5-1.5B-Instruct 和 LLaMA3-8B-Instruct 上分别测试 KV cache size=512/1024/2048 下 6 类任务（Single-Doc QA、Multi-Doc QA、Summarization、Few-shot Learning、Synthetic、Code）的 16 个子数据集，WindowKV 在 LongBench 上以 12% 原始 KV cache 取得最多 SOTA 结果；(2) Needle-in-a-Haystack 测试 LLaMA3-8B-Instruct 在 8K context length、512 KV cache size 下的长上下文检索能力（Rouge-1 F1）；(3) Throughput test：在单张 A100 40G 上对比 Vanilla、Vanilla+WindowKV、Vanilla+WindowKV+Classifier 的吞吐量和延迟；(4) 消融实验：不同 γ（共享层数 = 1/4/7/8/14/16）对 LongBench 性能的影响；不同 review window size（8/16/32/64/128）的影响；任务类型（localization vs aggregation）匹配窗口选择策略的必要性。

- 硬件平台是什么，配置是什么。
  训练：8× NVIDIA A100 40GB GPU（训练 task-adaptive classifier）。
  推理/评估：单张 NVIDIA A100 40GB GPU（throughput test 和 benchmark 评估）。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen2.5-1.5B-Instruct（28 层，γ=7）、LLaMA3-8B-Instruct（32 层，γ=8）。
  Benchmark：LongBench（包含 6 类 16 子任务——Single-Doc QA: NarrativeQA/Qasper/MultiFieldQA-en; Multi-Doc QA: HotpotQA/2WikiMQA/Musique; Summarization: GovReport/QMSum/MultiNews; Few-shot Learning: TREC/TriviaQA/SAMSum; Synthetic: PCount/PassageCount; Code: RepoBench-P/RepoBench-L）；Needle-in-a-Haystack（Rouge-1 F1 评估检索准确率）。
  分类器训练数据集：自建数据集，9551 样本，train/val/test = 8:1:1。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源：代码公开在 https://github.com/optim996/WindowKV

  算法 Pipeline 伪代码级解释：
  ```
  # 超参数
  n = input_context_length
  α = observation_window_size       # Qwen2.5: α=4(loc) or 16(agg); LLaMA3: α=16 or 32
  ω = review_window_size            # Qwen2.5: ω=32; LLaMA3: ω=8(loc) or 16(agg)
  p = top_p_for_aggregation         # p=ω for localization; p<ω for aggregation
  γ = shared_layers_per_group       # Qwen2.5: γ=7; LLaMA3: γ=8
  H = num_layers / γ                # number of groups
  λ = 14.0                          # pyramid shape control
  b_total = target_kv_cache_size

  # Step 1: 任务分类
  task_type = classifier(input_context)  # "localization" or "aggregation"

  # Step 2: 前向传播 - 第一层计算 attention scores
  # 对于每组的第一层 l_g:
  Q, K = W_q @ h, W_k @ h                    # [n, d_head]
  A = softmax(Q @ K^T / sqrt(d_k))            # [n, n], causal masked

  # Step 3: Observation window 评估 review context token 重要性
  # observation window tokens: [n-α, n]; review context: [0, n-α]
  for j in [0, n-α]:
      t_j = sum_{i in [n-α, n]} A[i, j]      # token-level attention score

  # Step 4: Window-level scoring
  num_windows = ceil((n - α) / ω)
  for k in [1, num_windows]:
      W_k = {t_j, ..., t_{j+ω-1}}             # review window k
      top_p = select_top_p(W_k, p)             # Top-p tokens within window
      s_k = (1 / min(p, ω)) * sum(top_p)       # window score

  # Step 5: 根据 group budget 选择 top windows
  b_0 = (2 * b_total) / H - b_{H-1}           # bottom group budget (largest)
  b_{H-1} = b_total / (λ * H)                  # top group budget (smallest)
  for h in [0, H-1]:
      b_h = b_0 - (b_0 - b_{H-1}) / (H-1) * h  # arithmetic sequence
      per_layer_budget = b_h / γ
      n_windows_h = per_layer_budget / ω
      I_h = indices_of_top_n_windows(scores, n_windows_h)

  # Step 6: 组内共享 indices
  for h in [0, H-1]:
      group_layers = [h*γ, h*γ+1, ..., h*γ+γ-1]
      for layer in group_layers:
          if layer == group_layers[0]:
              selected_KV = gather(KV_cache[layer], I_h)    # 仅首层计算
          else:
              selected_KV = gather(KV_cache[layer], I_h)    # 复用首层 indices
  ```

  关键张量计算路径（以 LLaMA3-8B-Instruct、n=7950、KV size=2048 为例）：
  1. 输入 token embedding [7950, 4096] 经 32 层 Transformer
  2. 每层 attention: Q,K,V = Linear(h), dim=[7950, 128] per head (32 heads in 8 KV groups via GQA)
  3. 第一层计算 full attention A = softmax(Q@K^T/√128), [7950, 7950]
  4. Observation window (最后 α=16 tokens) → 累积 attention → t_j per review token [7950-16]
  5. 切分 review windows (ω=8 for localization) → ~992 windows → s_k per window
  6. 4 groups (γ=8), b_total=2048, λ=14 → b^0=704, b^3=320 → 每层 budget: group0=704, group1=576, group2=448, group3=320
  7. Window selection + indices sharing → 仅 4 次 selection（每组首层），其余 28 层复用 → 以 12% 原始 KV cache 保持 LongBench 性能 41.35 vs FKV 41.51
