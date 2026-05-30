## Continual Pre-training of MoEs: How robust is your router?

- 属于算法pipeline的实现是什么？实验比较什么？
  实现是对 **MoE 持续预训练（Continual Pre-training of MoEs）** 的系统性实证研究。研究了两种主流路由算法（Penalty-Balanced Top-k / Sinkhorn-Balanced Top-k）和两种 MoE 架构（Switch MoE / Granular MoE）在经历从英文到代码/德语的分布偏移时的持续预训练行为。核心技术包括：Infinite LR Schedule（CosineInf）、Replay（回放旧数据）、LR Re-warming + Re-decaying。提出新指标 **MRI (Maximum Routing Imbalance)** 衡量最坏情况延迟下的路由不平衡。
  
  实验比较：
  - **CPT MoE vs FLOP-matched Dense Baseline**：验证 MoE 在持续预训练中是否保持样本效率优势
  - **CPT MoE vs Full Re-training MoE**：验证 CPT 是否能以更低成本匹配完全重训练的性能
  - **PBTk vs SBTk 路由算法**：比较两种路由在分布偏移下的鲁棒性（性能、MRI、路由行为变化）
  - **Switch MoE vs Granular MoE 架构**：比较两种架构的 CPT 表现
  - **Replay 百分比消融（0%, 10%, 30%, 40%）**：分析 replay 对遗忘和适应的 trade-off
  - **Decayed vs Non-decayed checkpoint CPT**：比较从衰减后 checkpoint 和不衰减 checkpoint 开始 CPT
  - **路由行为变化分析**：Router Saturation、Vocabulary Specialization、Expert Co-activation 三指标分析分布式偏移前后的路由决策变化

- 硬件平台是什么，配置是什么。
  64× NVIDIA A100 GPU，使用数据并行（Data Parallelism）和 ZeRO-1（Rajbhandari et al., 2020）。为加速 dropless MoE 前向传播使用了 Megablocks kernel（Gale et al., 2023）。代码基于 GPT-NeoX 库（Andonian et al., 2023）实现。训练精度论文未明确说明。

- 模型是什么。数据集和bench分别是什么。
  模型（共 5 个，均为 decoder-only，Llama3 架构骨架，GeLU 激活，Llama3 tokenizer，序列长度 2048）：
  - **Dense Baseline**：24 层，570M 参数，hidden size=1024，FFN intermediate=2816，GEGLU FFN
  - **PB Switch MoE**：8 个 routed experts，K=1 active，无 shared expert，full-sized FFN (2816)，Penalty-Balanced（Z-loss coeff=0.001 + Aux-loss coeff=0.01）路由，~2B total / 570M active
  - **SB Switch MoE**：同上但使用 Sinkhorn-Balanced 路由（tolerance=0.01），~2B total / 570M active
  - **PB Granular MoE**：31 个 routed experts，K=3 active，1 个 shared expert，fine-grained FFN (intermediate=704，dense 的 1/4)，Penalty-Balanced 路由，~2B total / 570M active
  - **SB Granular MoE**：同上但使用 Sinkhorn-Balanced 路由，~2B total / 570M active
  所有模型使用 AdamW optimizer（β1=0.9, β2=0.95），weight decay=0.1，gradient clipping=1.0，batch size=1024，Rotary positional embedding（PCT=0.25），vocab size=128000。

  数据集：
  - **Pre-training (Task 1)**：FineWeb（英文 Web），400B tokens，采样自 2916.65B 子集
  - **CPT (Task 2)**：Stack（Code，251.819B 子集）200B tokens，German Common Crawl（169.291B）200B tokens

  Benchmarks：
  - **英文（0-shot）**：HellaSwag, Winogrande, PIQA, ARC-Easy, ARC-Challenge, SWAG, LAMBADA, SciQ, PubMedQA, MathQA
  - **德文（0-shot，GPT-3.5 翻译版）**：HellaSwag-DE, ARC-Challenge-DE, TruthfulQA-DE
  - **代码**：HumanEval（pass@1/10/50/100/150/200）

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  论文未提供独立开源代码仓库。训练基于开源库 GPT-NeoX (https://github.com/eleutherai/gpt-neox) 和 Megablocks grouped GEMM kernel (https://github.com/tgale96/grouped_gemm)。

  **MoE CPT 算法 Pipeline**：

  ```
  # === MoE Layer 前向传播（每层） ===
  # x: [S, H] token hidden states, S=seq_len, H=hidden_dim
  # W_r: [H, E] router weight, E=num_experts
  
  def moe_layer_forward(x, experts, router, shared_expert=None, routing="PBTk"):
      logits = x @ W_r  # [S, E]
      if routing == "SBTk":
          probs = sinkhorn_balance(logits)  # Sinkhorn-Knopp 迭代
      else:  # PBTk
          probs = softmax(logits)
      
      # Top-k 专家选择
      topk_vals, topk_idx = topk(probs, k)  # [S, k]
      
      # MoE 输出计算
      moe_out = zeros_like(x)
      for each token s:
          norm = sum(probs[s, topk_idx[s]])
          for i in topk_idx[s]:
              moe_out[s] += probs[s,i] * experts[i](x[s]) / norm
      
      # Shared Expert (Granular MoE only)
      if shared_expert is not None:
          moe_out += shared_expert(x)
      
      return moe_out
  ```

  ```
  # === CPT Training Loop ===
  # Phase 1: Pre-training on FineWeb (400B tokens)
  for step in range(192720):
      batch = sample(FineWeb, batch_size=1024)
      # Loss = LM loss + α * Aux Loss + β * Z-Loss (仅 PBTk)
      loss = lm_loss + 0.01 * aux_loss + 0.001 * z_loss  
      optimizer.step()  # CosineInf schedule, lr_const=1.65e-4
  
  # Phase 2: CPT (200B tokens, 30-40% replay)
  for step in range(95370):
      batch_fw = sample(FineWeb, batch_size=1024 * replay_pct)
      batch_new = sample(target, batch_size=1024 * (1-replay_pct))
      batch = concat(batch_fw, batch_new)
      loss = model(batch)
      optimizer.step()  # CosineInf: 从 const LR 继续，无 cooldown
      
      # MRI 监控
      for layer in moe_layers:
          mri = max(token_load_per_expert / total_tokens)  # Eq. (1)
  ```

  **MRI 定义**（Eq. 1）：
  $$MRI(t,j) := \max_{i \in [1,\dots,E]} \left[ \frac{\sum_{x \in B} \mathbb{1}\{i \in I_k(x)\}}{|B|} \right]$$
  其中 $B$ 为一个 batch 中的所有 tokens，$I_k(x)$ 为 token $x$ 的 top-k 专家索引集合。MRI 越大表示最繁忙 expert 承载越多 tokens → 最坏情况延迟越高。

  **核心 CPT 策略**：
  - **CosineInf Schedule**：预训练阶段用 CosineInf（constant 80% + cooldown 70%），CPT 阶段从 $\eta_{const}=1.65\times10^{-4}$ 继续，cooldown=0%（LR 始终保持 constant），$\eta_{max}=3\times10^{-4}$，$\eta_{min}=3\times10^{-5}$
  - **Replay**：每 batch 中 X% 的样本来自旧分布，(100-X)% 来自新分布。Compute Equivalent Replay：增加 replay 不增总 token budget 而减少新数据量
  - **PBTk Routing**：$L_{total} = L_{LM} + \alpha \cdot L_{aux} + \beta \cdot L_z$，$\alpha=0.01, \beta=0.001$
  - **SBTk Routing**：softmax 前应用 Sinkhorn-Knopp 迭代近似求解线性分配问题，推理时去掉 balancing step（不兼容自回归生成）
