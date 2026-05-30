## Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是一个研究 LLM 自适应计算的框架，分为三个阶段：(1) **Stage 1: 训练 Duo FFN 模块**：在每层 FFN 中并排放置一个 big FFN（inner dim=10240）和一个 small FFN（inner dim=640，小 16 倍），共享 attention；训练时以 0.5 概率随机路由 token 到 big 或 small 模块，使两个模块可互换使用。(2) **Stage 2: Oracle 引导的最优路由**：对每条输入序列，穷举所有可能的路由路径（仅 big/small 二选一时为 2^n 条，加入 skip 时为 3^n 条），在固定计算预算约束下选择使 perplexity 最低的路由，作为理论最优路由上界。(3) **Stage 3: 学习路由近似 Oracle**：训练一个类似 MoE 的 learnable router（每层一个线性层 W_{r,l}），通过 soft routing（带温度 τ 的 softmax）结合 budget loss 约束全局 big 模块使用比例（而非 per-layer load balancing），学习近似 oracle 的最优路由。实验比较：(a) Oracle vs 最优随机模式 vs trained router 在不同预算下的 perplexity；(b) Oracle 在不同 big layer 预算下的路由模式（C4 + Code holdout）；(c) Trained router 的路由模式与 oracle 的对比；(d) Token difficulty 分析（small model loss vs loss gap）；(e) Duo-LLM 与同等 FLOPs dense 模型的 accuracy 对比（arc_easy, hellaswag）；(f) 从 scratch 训练 vs freeze big + fine-tune small 的对比。

- 硬件平台是什么，配置是什么。
  论文未明确说明训练硬件平台和 GPU 配置。论文提到未来可以考虑基于 Megablocks 的 block-sparse matrix multiplication 在单 GPU 上高效执行 Duo-LLM，但论文未提供具体硬件规格。

- 模型是什么。数据集和bench分别是什么。
  模型：12 层 Llama 2 风格架构，hidden dim=2560，共 1.399B 参数（big FFN: 944M，small FFN: 59M，attention: 314M，其余为 embeddings）。训练数据：300B tokens，来源包括 FineWeb、Wiki、Flan（来自 Dolma）、Python code（Stack-v2）。评估 holdout sets：(1) C4 validation set 中随机采样 1024 条；(2) GitHub MIT license 的 Python code 中随机采样 1024 条。Benchmarks：arc_easy、hellaswag（用于与 dense 模型对比）。附录中在 OPT-1.3B 上进行了额外路由实验。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文来自 Apple，论文未提供开源代码仓库。算法核心流程如下：

  ```
  === Stage 1: Duo FFN Training (Random Routing) ===
  For each layer l in [1..L]:
      h = Attention(x)              # shared attention
      # Random routing with p=0.5
      if random() < 0.5:
          h_out = BigFFN_l(h)       # inner_dim = 10240
      else:
          h_out = SmallFFN_l(h)     # inner_dim = 640 (16x smaller)
      x = x + h_out                 # residual connection

  === Stage 2: Oracle Optimal Routing (Exhaustive Search) ===
  Given: input sequence x, budget B (e.g., 4 big layers out of 12)
  For each possible route r in {0,1}^L (0=small, 1=big):
      if sum(r) == B:               # meet budget constraint
          forward(x, route=r)       # execute with chosen modules
          loss_r = CrossEntropy(output, labels)
  optimal_route = argmin_r(loss_r)

  Oracle routing with skip: {0=small, 1=big, 2=skip}^L
  (3^L total routes, exhaustively searched)

  === Stage 3: Learned Router Training ===
  For each layer l:
      h_l = x @ W_{r,l}                       # router logits
      P_{l,big} = softmax(h_l / τ)            # routing probabilities
      P_{l,small} = 1 - P_{l,big}
      H_big = BigFFN_l(x)
      H_small = SmallFFN_l(x)
      output_l = P_{l,big} * H_big + P_{l,small} * H_small  # soft combination

  Budget loss (across all layers, not per-layer):
      L_budget = (mean(P_{:,big}) - target_budget)^2
      L_total = L_CE + α * L_budget

  Temperature τ gradually increased for hard assignment.
  ```
