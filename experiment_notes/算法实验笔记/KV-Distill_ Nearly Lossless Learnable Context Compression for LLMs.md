## KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

- 属于算法pipeline的实现是什么？实验比较什么？
  KV-Distill 提出一种可训练的 KV cache 压缩框架，通过 token 重要性打分（FFN scorer）、LoRA 条件计算适配、以及前向+反向 KL 散度蒸馏目标来将长上下文的 KV cache 压缩为更短的表示。具体流程：(1) 通过一个 FFN 对第 η=6 层的 hidden states 打分得到每个 token 的重要性分数 s ∈ R^N；(2) 取 top-k 重要 token 索引，通过 hard selection matrix S ∈ {0,1}^{k×N} 从 KV cache 中提取 ˜X = SX；(3) 将上下文通过带有 LoRA adapters 的 LM_θ 编码，其中被选中的 token 路由到可训练的 W^Q/W^O 矩阵，未选中的 token 通过冻结的原始矩阵；(4) 使用加权 KL 散度 L(θ) = λ·D_KL(p||q_θ) + (1-λ)·D_KL(q_θ||p) 匹配压缩前后的 next-token 分布。实验比较 KV-Distill 与 H2O (H2A 问题感知/H2I 问题无关)、DODO、ICAE 在提取式 QA、长文本 QA、抽象式摘要、Needle-in-a-Haystack 上的性能。

- 硬件平台是什么，配置是什么。
  训练：8 × NVIDIA A100 80GB GPU 集群，使用 DeepSpeed Stage 2 分布式训练，bf16 精度。推理评估：论文未明确说明推理硬件，但从模型规模（7B-27B）推断在单/多 GPU 上进行。

- 模型是什么。数据集和bench分别是什么。
  模型：LLAMA-2 7B、LLAMA-3 8B、MISTRAL 7B、GEMMA-2 9B、GEMMA-2 27B（均使用 instruction-tuned 版本）。训练数据：从 Self-Instruct、P3、LongAlpaca、Super-Natural Instructions 中 curated 的大规模指令数据集，拆分为 (Context, Instruction, Answer) 三元组。基准测试：SQuAD（提取式 QA，平均长度 225 tokens）、QuALITY（长文本多选题 QA，平均 6K tokens）、SQuALITY（长文本抽象式摘要，平均 7K tokens）、GovReport（长文档摘要，平均 10K tokens）、Needle-in-a-Haystack（长文本检索）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  官方仓库 https://github.com/vnchari/kv-distill，论文声明代码和 checkpoint 即将发布，但截至本文时仅有 README，完整代码尚未公开。算法 pipeline 如下：
  ```
  # === KV-Distill 压缩流程 ===
  # 输入: context c ∈ V^N, 预训练 LM, LoRA adapter LM_θ

  # Step 1: Token Importance Scoring
  # η=6 层的 hidden states X'_η ∈ R^{N×d}
  s = FFN_θ(X'_η)              # s ∈ R^N, 每个 token 的重要性分数
  indices_topk = topk(s, k)     # 取 top-k 索引, k = N × retention_ratio

  # Step 2: 编码上下文通过 LoRA-adapted LM_θ
  for each transformer layer l:
      # 正常 forward pass
      X_l^K, X_l^V = encode(context)  # [N, d]

      # 条件计算路由:
      for selected tokens i ∈ indices_topk:
          Q_i = (z_i @ W^Q_lora)       # 使用可训练的 LoRA W^Q
          O_i = attention(Q_i, K, V) @ W^O_lora  # 使用可训练的 LoRA W^O
      for unselected tokens j:
          Q_j = (z_j @ W^Q_frozen)     # 使用冻结的原始 W^Q
          O_j = attention(Q_j, K, V) @ W^O_frozen  # 使用冻结的原始 W^O

  # Step 3: 从 LM_θ 的输出中提取压缩 KV cache
  ˜X = S @ X    # S ∈ {0,1}^{k×N}, hard selection
  # 即只保留 indices_topk 对应 token 在所有层的 KV

  # Step 4: 梯度传播（non-differentiable topk 的替代方案）
  # 在注意力计算中，对 attention weights 按重要性衰减:
  α' = σ(s) ⊙ α    # σ=sigmoid, ⊙=Hadamard product
  # 被选中 token 的 attention weight 不变（sigmoid(高分)≈1）
  # 未选中 token 的 attention weight 被衰减

  # Step 5: KL 散度蒸馏损失
  p = softmax(LM(y | X_full) / T)    # teacher: 完整 cache
  q_θ = softmax(LM(y | ˜X) / T)    # student: 压缩 cache
  L(θ) = λ·Σp·log(p/q_θ) + (1-λ)·Σq_θ·log(q_θ/p)
  # λ=0.6 (偏向 forward KL 以稳定训练)

  # Step 6: 训练细节
  # LoRA: rank=128, 应用于 Q,K,V,O 矩阵 (rsLoRA)
  # 优化器: AdamW, lr=5e-5, batch_size=32
  # 训练时随机采样 retention ratio ∈ [0.1%, 80%]
  # 长上下文 (>1536 tokens) 折叠为 batch of N×1536
  # 前几个 tokens (<10) 始终保留（sink tokens）
  ```
  关键设计：KV retention ratio 在训练时随机采样（0.1%-80%），因此单个 KV-Distill 模型支持任意压缩率。训练参数仅 150M（LoRA adapter），压缩后的 KV cache 在自回归解码时零额外开销。前 k% 的 token 选择可跨层共享索引。forward + reverse KL 混合损失（λ=0.6）优于纯 forward KL（λ=1, SQuAD 83.4%）、纯 reverse KL（λ=0, 82.7%）和 auto-encoding + CE loss（79.1%）。No routing（用可学习 embedding 替代条件计算 routing）= 67.4%。

  关键结果：LLAMA-3 8B SQuAD: KVD 25% retention 86.6%（vs uncompressed 87.6%, H2A 25% 84.0%, H2I 25% 56.6%）。Needle-in-a-Haystack: 90% compression 下近乎完美准确率。QuALITY: 10x compression 下与 uncompressed 性能接近。SQuALITY: >20% retention 时 ROUGE-L 等于或超过 uncompressed。GovReport fine-tuning: 1% retention (100x compression) ROUGE-L 22.8（vs uncompressed 23.7）。各模型蒸馏训练 3-4 天（GEMMA 27B 需 4 天）。
