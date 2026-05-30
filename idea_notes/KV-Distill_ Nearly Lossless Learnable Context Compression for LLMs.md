## KV-Distill: Nearly Lossless Learnable Context Compression for LLMs

- baseline方法是什么？
  Baseline 分为两类：(a) 训练无关方法：H2O (H2A/H2I) 和 SnapKV。H2O 基于累积注意力分数选出 "heavy-hitter" tokens 作为 KV cache 中保留的 top-k 键值对。H2A（问题感知）将问题和上下文拼接后计算累积注意力，能利用问题扫描上下文中的关键信息；H2I（问题无关）仅在上下文内部计算累积注意力。SnapKV 使用最近 token 窗口的注意力模式选择重要 token。(b) 可训练方法：ICAE 使用 auto-encoding + language modeling 目标预训练上下文压缩器，将长上下文编码为少量 memory slots，再用 frozen LLM 解码。DODO 将 KV cache 子选择为 "nugget" tokens，训练时使用 auto-encoding 或 LM 目标，但压缩率固定。

  全栈执行例子（H2O on LLAMA-3 8B，上下文 N=6000 tokens，20% retention，问题无关范式）：
  - **算法层**：将上下文 tokens 通过 LLAMA-3 forward pass 得到所有层的 attention weights → 对每层每头累加 attention scores → 选 top-k=1200 tokens 保留在 KV cache → 被 evict 的 token KV 从 cache 删除。解码时：每个新 token 仅对保留的 1200 tokens attend（而非 6000）。
  - **系统框架层**：标准 HuggingFace Transformers 的 KV cache 机制，修改 cache 的 `past_key_values` 元组在 prefill 后裁剪。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 Flash Attention kernel（PyTorch SDPA 或 flash-attn），无特殊 kernel 修改。
  - **硬件架构层**：NVIDIA A100 80GB / H100。

  Baseline 的关键缺陷：
  (a) H2I 在问题无关范式下性能急剧下降——因为缺少问题信号引导，上下文内部的 heavy-hitter 分布与实际需要回答的问题无关。例如 LLAMA-3 SQuAD: H2I 25% retention 准确率仅 56.6%（vs uncompressed 87.6%）。
  (b) ICAE/DODO 使用 auto-encoding 预训练目标，与下游推理时 next-token prediction 存在分布不一致（pretraining-inference mismatch），导致高压缩率下性能损失大。
  (c) 训练无关方法无法利用领域先验知识进一步提升压缩性能（H2O 没有 fine-tuning 机制）。
  (d) ICAE/DODO 的压缩率固定，不支持灵活的任意压缩率推理。
  (e) 训练无关方法的 token 选择仅在 prefill 后执行一次，被丢弃的 token 中的信息永久丢失，未被选中的 token 无法向被选中的 token "传递"信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  KV-Distill 提出三组件协同的 KV cache 压缩框架：

  **1. Learnable Token Importance Scorer + Conditional LoRA Routing → 解决缺陷(a)(e)**：
  训练一个 FFN scorer 从第 η 层的 hidden states 预测每个 token 的重要性分数 s = FFN_θ(X'_η)，取 top-k 作为保留 token。更重要的是，通过 LoRA-adapted LM_θ 进行条件计算路由：被选中的 token 使用可训练的 LoRA W^Q/W^O 矩阵，使其能 "吸收" 来自未选中 token 的信息（通过 cross-attention 机制），实现被选中 token 表示的语义增强。这与训练无关方法的纯 token 选择有本质区别——KV-Distill 不仅选择 token，还增强了被选 token 的表示质量。

  **2. Forward + Reverse KL Divergence Distillation → 解决缺陷(b)**：
  使用加权 KL 散度 L(θ) = λ·D_KL(p||q_θ) + (1-λ)·D_KL(q_θ||p) 直接匹配压缩前后的 next-token 分布（p=完整 cache 的分布，q_θ=压缩 cache 的分布）。forward KL (λ=0.6 主导) 为 mean-seeking 行为，确保压缩模型覆盖完整模型的所有可能输出；reverse KL 为 mode-seeking 行为，避免分布模式坍缩。这与 auto-encoding loss 有本质区别：KL 散度直接在 token 预测分布层面优化，与下游推理时的 next-token prediction 任务一致，消除了 pretraining-inference mismatch。

  **3. Multi-Ratio Training → 解决缺陷(c)(d)**：
  训练时随机采样 KV retention ratio ∈ [0.1%, 80%]，使单一 KV-Distill 模型支持任意压缩率推理。此外支持在领域数据上通过相同 KL 损失 fine-tune，进一步提升特定领域下的压缩率上限（GovReport 上 1% retention fine-tuned ROUGE-L=22.8 vs uncompressed 23.7）。

  全栈执行例子（KV-Distill on LLAMA-3 8B, N=6000 tokens, 20% retention=1200 tokens）：
  - **算法层**：
    (a) Pre-compression: context tokens → LM_θ 第 6 层 hidden states X'_6 → FFN scorer 输出 s ∈ R^6000 → top-1200 索引。
    (b) Encoding with routing: context tokens 通过 LoRA-adapted LM_θ (rank=128, Q/K/V/O 应用 LoRA) 编码。每层中：被选 token i 的 query 通过 LoRA W^Q 计算 → attention weights α_i = (z_i @ W^Q_lora)(K)^T → 可选地通过 α'_i = σ(s_i) ⊙ α_i 衰减 attention → output 通过 LoRA W^O 变换。未选 token j 使用冻结原始 W^Q/W^O 计算（其 KV 会参与被选 token 的 attention 计算，传递信息，但自身 KV 最终被丢弃）。
    (c) KL distillation: compressed KV ˜X (仅 1200 个 token) → LM 解码 → p_compressed；full KV X → LM 解码 → p_full；L = 0.6·D_KL(p_full||p_compressed) + 0.4·D_KL(p_compressed||p_full)，反向传播更新 LoRA + FFN scorer 的 150M 参数。
    (d) 解码时：使用原始 frozen LM，仅对 compressed KV cache ˜X 做 attention，无额外计算开销（LoRA adapter 已融合到 encoding 阶段）。
    (e) 长上下文折叠：N>1536 的上下文 pad 到 1536 的倍数，reshape 为 batch 后分别压缩再 unfold，保持 1536 的训练上下文窗口。
  - **系统框架层**：HuggingFace Transformers + DeepSpeed Stage 2 (8×A100 80GB) + LoRA adapter 管理。压缩后的 KV cache 格式与原始 KV cache 完全相同（仅 sequence length 维度缩小），因此任何支持 KV cache 的推理框架可直接使用。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：标准 Flash Attention (SDPA)，无特殊 kernel 修改。训练时梯度传播通过 α' = σ(s) ⊙ α 实现 scorer 的可微分（绕过 non-differentiable topk）。推理时 zero overhead——compressed KV cache 直接配合标准 attention kernel。
  - **硬件架构层**：训练：8 × NVIDIA A100 80GB。推理：任意支持 HuggingFace Transformers 的 GPU 平台。

  **对比 baseline 的关键差异**：
  - H2I 纯 attention-based token selection → KV-Distill 可训练 scorer + conditional LoRA routing 增强选中 token 表示（信息从丢弃 token 传递到保留 token）
  - ICAE/DODO 的 auto-encoding loss → KV-Distill 的 next-token KL divergence loss（消除 pretraining-inference mismatch）
  - H2O/SnapKV 无领域适应能力 → KV-Distill 支持领域 fine-tune 实现 100x 压缩
  - ICAE/DODO 固定压缩率 → KV-Distill 单模型支持 0.1%-100% 任意压缩率
  - LLAMA-3 8B SQuAD 20% retention: KVD 86.0% vs H2I 51.7%, H2A 83.0%, DODO 73.3%
  - GovReport 1% retention fine-tuned: ROUGE-L 22.8 vs H2I 18.3
  - 1000x compression 下仍能产生有意义输出（定性分析）
