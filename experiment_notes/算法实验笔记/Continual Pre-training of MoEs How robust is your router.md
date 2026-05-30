## Continual Pre-training of MoEs How robust is your router

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是针对 MoE transformer 的大规模持续预训练（Continual Pre-training, CPT）策略。论文系统研究了两种路由算法（Penalty-Balanced Top-k, PBTk 和 Sinkhorn-Balanced Top-k, SBTk）和两种 MoE 架构（Granular MoE: 31 routed experts, K=3 active, 1 shared expert; Switch MoE: 8 routed experts, K=1 active, 无 shared expert）在分布偏移下的 CPT 行为。CPT 策略使用：(1) Infinite LR schedule (CosineInf)，从非衰减 checkpoint 恢复训练；(2) Replay 机制（30%/40% 旧数据回放）；(3) Learning rate re-warming + re-decaying（从衰减 checkpoint 开始时）。

  实验比较：(a) 4 种 MoE 架构（PB Granular, SB Granular, PB Switch, SB Switch） vs FLOP-matched Dense Baseline (570M) 在 FineWeb→Stack(Code) 和 FineWeb→German 两个分布偏移下的 CPT 表现；(b) CPT vs full re-training baseline（从头在 FineWeb∪Stack/German 联合数据上训练）；(c) 不同 replay 比例（0%/10%/40%）对遗忘和适应的影响；(d) 从衰减 vs 非衰减 checkpoint 开始 CPT 的对比；(e) 路由行为分析：Router Saturation (路由饱和率), Vocabulary Specialization (词汇专精), Expert Co-activation (专家共激活), Maximum Routing Imbalance (MRI, 最大路由不均衡度)。

- 硬件平台是什么，配置是什么。
  64 张 NVIDIA A100 GPU，使用 data parallelism + ZeRO-1（Rajbhandari et al., 2020）。显存和互联配置：论文未明确说明单卡显存（应为 A100-80GB 或 A100-40GB）。

- 模型是什么。数据集和bench分别是什么。
  模型：
  - Dense Baseline: 24层 570M 参数 decoder-only transformer，Llama3 架构（但使用 GeLU 激活），Llama3 tokenizer，GEGLU FFN（中间维度 2816），hidden size 1024，16 attention heads，RoPE
  - Granular MoE: 570M active / 2B total，E=31 routed experts + 1 shared expert，K=3 active，FFN 中间维度 704（为 dense 的 1/4），GEGLU
  - Switch MoE: 570M active / 2B total，E=8 routed experts，K=1 active，无 shared expert，FFN 中间维度 2816（与 dense 相同），GEGLU
  所有 MoE 使用 Top-k routing（k=1 for Switch, k=3 for Granular），不 drop token。PBTk 使用 z-loss coefficient 0.001 + Aux-loss coefficient 0.01。SBTk 使用 tolerance 0.01。

  数据集：
  - Pre-training: FineWeb (English web crawl, 400B tokens)
  - CPT: The Stack (code, 200B tokens) 和 German Common Crawl (200B tokens)
  - Replay: FineWeb 数据按比例回放（30% for Stack, 40% for German）

  Benchmarks:
  - English: HellaSwag, Winogrande, PIQA, ARC-Easy, ARC-Challenge, SWAG, LAMBADA (OpenAI), SciQ, PubMedQA, MathQA
  - German (GPT-3.5 翻译): HellaSwag-DE, ARC-Challenge-DE, TruthfulQA-DE
  - Code: HumanEval (pass@k, k∈{1,10,50,100,150,200})
  - Validation loss: FineWeb, Stack, German 测试集上的 log perplexity
  - Routing metrics: MRI, Router Saturation, Vocabulary Specialization, Expert Co-activation

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供独立的代码仓库。实验基于 GPT-NeoX 库 (https://github.com/EleutherAI/gpt-neox) 和 Megablocks grouped GEMM kernel (https://github.com/tgale96/grouped_gemm)。训练使用 AdamW 优化器 (β₁=0.9, β₂=0.95)，weight decay 0.1，gradient clipping 1.0，batch size 1024，sequence length 2048。

  **算法 pipeline 伪代码（MoE CPT 训练循环）**：

  ```
  # Step 1: 初始化 MoE 模型 (以 Granular PBTk 为例)
  model = MoETransformer(
      num_layers=24, hidden_size=1024,
      num_routed_experts=31, num_active_experts=3,
      shared_expert=True, ffn_intermediate=704,
      router_type="PBTk"  # 或 "SBTk"
  )
  # 使用 Llama3 tokenizer, vocab_size=128000

  # Step 2: Pre-training Phase (FineWeb, 400B tokens)
  scheduler = CosineInf(  # Infinite LR schedule
      total_iters=192720, eta_max=3e-4, eta_min=3e-5,
      eta_const=1.65e-4, T_warmup=0.01, T_cooldown=0.70
  )
  for step in range(192720):
      batch = sample_fineweb(batch_size=1024, seq_len=2048)
      loss = model(batch).loss + aux_loss(model.router_logits, batch) * 0.01
                                  + z_loss(model.router_logits) * 0.001
      optimizer.step(loss)  # AdamW
      scheduler.step()

  # Step 3: CPT Phase (FineWeb→German or FineWeb→Stack)
  scheduler_cpt = CosineInf(  # 从非衰减 checkpoint 恢复
      total_iters=95370, eta_max=3e-4, eta_min=3e-5,
      eta_const=1.65e-4, T_warmup=0.01, T_constant=0.80
  )
  for step in range(95370):
      # Replay: 40% FineWeb + 60% German (或 30% FineWeb + 70% Stack)
      batch_replay = sample_fineweb(batch_size * 0.4, seq_len=2048)
      batch_new = sample_german(batch_size * 0.6, seq_len=2048)
      batch = concat([batch_replay, batch_new])
      loss = model(batch).loss + aux_loss * 0.01 + z_loss * 0.001
      optimizer.step(loss)
      scheduler_cpt.step()
  ```

  **MoE 层前向传播（以 token x 为例）**：
  ```
  # Router: W_r ∈ R^{H×E} (H=hidden_size, E=num_experts)
  logits = W_r @ x                    # [E]  线性投影
  probs = softmax(logits)             # [E]  PBTk: 恒等; SBTk: Sinkhorn re-weight
  topk_indices = topk(probs, k=3)     # 选择 top-3 experts
  topk_probs = probs[topk_indices]    # 对应的概率

  # Expert computation (GEGLU FFN)
  shared_out = SharedFFN(x)           # shared expert 输出
  expert_outs = []
  for idx in topk_indices:
      expert_outs.append(GEGLU_FFN_expert[idx](x))

  # Weighted combination
  combined = sum(topk_probs[i] * expert_outs[i] for i in range(k))
  combined = combined / sum(topk_probs)  # 归一化

  output = shared_out + combined      # MoE 层输出
  ```

  **Maximum Routing Imbalance (MRI) 计算**：
  ```
  # 对于 MoE 层 j，batch B 中的 tokens
  def compute_MRI(layer_j, batch_B):
      E = layer_j.num_experts
      k = layer_j.num_active_experts
      loads = zeros(E)
      for token in batch_B:
          topk_indices = layer_j.route(token)
          for idx in topk_indices:
              loads[idx] += 1
      loads = loads / len(batch_B)    # 归一化
      return max(loads)               # MRI
  ```
