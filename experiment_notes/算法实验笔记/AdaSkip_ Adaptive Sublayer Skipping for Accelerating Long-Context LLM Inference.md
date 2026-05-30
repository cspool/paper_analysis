## AdaSkip: Adaptive Sublayer Skipping for Accelerating Long-Context LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 AdaSkip，一种 training-free、自适应的 sublayer-wise skipping 策略，专为长上下文 LLM 推理设计。核心思想：(1) 利用 IO Similarity（输入输出向量的余弦相似度）评估 sublayer 重要性——相似度高表示 sublayer 输出接近输入，该 sublayer 对前向传播贡献小，可以被跳过；(2) Offline Importance Learning：从历史推理任务中学习各 sublayer 的平均 IO Similarity 和 Scale Factor，用于 Prefilling 阶段的 sublayer-wise skipping；(3) Online Importance Learning：利用 Decoding 阶段前 P 个 token（online learning window）计算当前上下文的 IO Similarity，额外再跳过部分 FFN sublayer（因为 Observation 3 发现 FFN 在 decoding 阶段相似度更高）。实验比较 AdaSkip 与三种 layer-wise skipping baseline——Early Exit（跳过后几层）、SkipDecode（跳过前几层）、Unified Skipping（均匀跳过中间层）——在 Prefilling 任务（Doc QA + Few-shot Learning）、Decoding 任务（Text Summarization）和 End-to-End（prefill+decode 同时跳过）下的生成质量（F1/ACC/Rouge-L）和加速比（SU, SpeedUp）。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA L20 GPU，CUDA 12.1。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA3.1-8B-128k、InternLM-7B-8k、Vicuna-v1.5-7B-16k。
  Prefilling 任务（输出长度 cap 32）：MultiFieldQA-en（F1，avg input 6493）、TriviaQA（F1，avg input 8677）、TREC（ACC，avg input 8208）。
  Decoding 任务（输出长度 limit 512）：GovReport（Rouge-L，avg input 9214）、MultiNews（Rouge-L，avg input 8265）。
  End-to-End 任务：prefill 和 decode 均跳过 sublayer。
  Offline Importance Learning 所用数据：2WikiMQA、MultiFieldQA-en、TriviaQA（来自 Stanford Alpaca 数据集）。
  Online Importance Learning 所用数据：TREC、GovReport。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/ASISys/AdaSkip

  **核心概念 — IO Similarity 作为重要性度量**：
  给定两个 n 维向量 a⃗ 和 b⃗，余弦相似度定义：
  $$Similarity(\vec{a}, \vec{b}) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\| \|\vec{b}\|} = \frac{\sum_{i=1}^{n} a_i b_i}{\sqrt{\sum_{i=1}^{n} a_i^2} \sqrt{\sum_{i=1}^{n} b_i^2}}$$

  IO Similarity 越高 → sublayer 输出越接近输入 → sublayer 越不重要 → 应被跳过。论文通过 LeastSkip vs MostSkip 实验验证：跳过 IO Similarity 最高的 sublayer 的 GPT 评分（跳 1/3/5 层：8.9/6.1/4.2）远优于跳过最低的（跳 1 层即低于 1.0）。

  **Phase 1: Offline Importance Learning（Prefilling 阶段）**：

  伪代码：
  ```
  // 输入：N 个历史推理样本（sample i 含 |T_i| 个 token）
  // 模型：M 个 Transformer Layer，含 M 个 Attention sublayer + M 个 FFN sublayer（共 2M 个 sublayer）

  // Step 1: 对每个 sublayer j，累积 IO Similarity 和 Scale Factor
  for each sample i in 1..N:
      for each token t in 1..|T_i|:
          for each sublayer j in 1..2M:
              Simi_j += cosine_similarity(a_it^j, b_it^j)  // 公式(2)
              Scale_j += ||b_it^j|| / ||a_it^j||            // 公式(3)

  // Step 2: 取平均
  Simi_j = Simi_j / sum(|T_i|)
  Scale_j = Scale_j / sum(|T_i|)

  // Step 3: 按 Simi_j 对所有 2M 个 sublayer 降序排序，得 sorted list
  sorted = argsort(Simi_j, descending=True)

  // Step 4: 根据加速比 α 确定跳过的 sublayer 数量
  m = M - M/α  // 跳过的 layer 数
  // 跳过前 2m 个 sublayer（IO Similarity 最高的）
  skipped = sorted[0:2m]

  // Step 5: 用 Scale_j 补偿被跳过 sublayer 的信息损失
  b_approx_it^j = Scale_j * a_it^j  // 公式(4)
  ```

  **Phase 2: Online Importance Learning（Decoding 阶段额外 FFN 跳过）**：

  伪代码：
  ```
  // 输入：当前新上下文的解码 token、Phase 1 的 skipped set
  // P = online learning window size（前 P 个 decoded token）

  // Step 1: 用前 P 个 token 计算每个 FFN sublayer 的当前上下文 IO Similarity
  for token t in 1..P:
      for each FFN sublayer j in index:  // index = 所有 FFN sublayer 索引
          Simi_j^P += cosine_similarity(a_t^j, b_t^j)

  Simi_j^P = Simi_j^P / P  // 公式(5)

  // Step 2: 计算阈值 β（skipped set 中最低的 Similarity）
  β = min{Simi_j | j in skipped}

  // Step 3: 找出当前上下文中 Similarity 高于 β 的额外 FFN sublayer
  for each FFN sublayer j in index:
      if Simi_j^P > β:
          skipped_extra.append(j)

  // Step 4: 合并
  skipped^P = skipped ∪ skipped_extra  // 最终的 sublayer-wise skipping set

  // Step 5: 同样用 Scale_j 补偿
  ```

  **数据复用交叉验证（Table 1）**：
  Offline 学习的 IO Similarity 特征在不同数据集间有高 hit rate：
  - Src=TriviaQA, Dest=MFieldQA: 3.76/4（top-4 hit）, 4.86/6, 9.31/10
  - Src=MFieldQA, Dest=Wiki: 3.80/4, 5.54/6, 9.90/10
  - FFN 跨数据集 hit rate 略低于 ATTN 但仍在 9.38-9.56/10 水平

  **Online Window Size 消融（Table 2）**：
  - TREC: size=5 → 0.84/2, size=20 → 1.08/2, size=40 → 1.07/2（20 起趋于饱和）
  - GovReport: size=5 → 1.01/2, size=20 → 1.14/2, size=40 → 1.19/2

  **执行流程全貌**：
  1. Offline：在历史数据集上跑推理，累积各 sublayer 的 Simi_j 和 Scale_j
  2. Prefilling：用 sorted + α 确定 skipped set，跳过高 Similarity sublayer，用 Scale 补偿
  3. Decoding：前 P 个 token 正常执行所有 sublayer → 计算当前上下文的 Simi_j^P → 用 β 阈值筛选额外 FFN → 后续 token 用 skipped^P 跳过；用 Scale 补偿
  4. 支持 Prefilling-only、Decoding-only、End-to-End（两阶段同时 skip）三种模式

  **关键性能数据（Table 3）**：
  - Prefilling task（LLaMA3.1-8B-128k, skip 8 sublayers）：AdaSkip TREC ACC 72.8%（Full: 75.0%），远超 SkipDecode 0.0% / Unified Skipping 2.2%
  - Decoding task（LLaMA3.1-8B-128k, skip 8 sublayers）：AdaSkip GovReport Rouge-L 30.9（Full: 34.2），实测加速 SU=1.15；SkipDecode 19.3（SU=1.07）
  - End-to-End（LLaMA3.1-8B-128k, skip 16 sublayers）：AdaSkip GovReport/MultiNews 18.9/17.8，baselines <5
  - Decoding 加速比：最高达 17% acceleration improvement over baselines（跳过更多 attention sublayer + 额外 FFN）
  - Prefilling 加速比：InternLM 上 >10% speedup advantage，LLaMA 上 attention 已有优化故略低于 baseline
