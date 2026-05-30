## Trainable_Dynamic_Mask_Sparse_Attention

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 Dynamic Mask Attention (DMA)，一种可训练的 content-position 双感知稀疏注意力机制。核心由两部分组成：(1) **Content-Aware Dynamic Mask**：从 value 向量表示采样生成动态 mask。使用采样权重矩阵 Δ∈R^{n_h×d_h×n_h} 和 per-head 门控系数 A∈R^{n_h}，通过 δ=exp(τ(v·Δ)×A) 计算重要性分数，其中 τ 为非负激活函数（softplus），sparsification 函数 f(·) 保留 per-head Top-w 位置的原始分数、其余置 −∞。门控系数 A 可做成 query-dependent 以自适应输入。(2) **Position-Aware Sparse Weights**：用动态 mask 对 scaled dot-product attention 做稀疏化，mask 值为 −∞ 的位置 attention weight≈0，kernel 实现对这些位置直接填零跳过计算。整个过程完全可微，支持端到端训练。复杂度从 O(n²d_h) 降为 O(nwd_h)，内存从 O(n²) 降为 O(nw)。实验比较所有 attention 变体（MHA、SWA、MLA、NSA）在 Scaling Laws（80M→1.7B 参数下的 SmolLMCorpus Perplexity）、Multi-Query Associative Recall（512 KV pairs，更长序列+更小模型维度）、下游 Benchmark（LAMBADA、MMLU、TriviaQA、ARC、PIQA、HellaSwag、OBQA、Winogrande、LongBench、RULER、BBH、GSM8K、MATH、MBPP）以及 Needle-in-a-Haystack 长文本信息检索上的表现。

- 硬件平台是什么，配置是什么。
  NVIDIA A100-SXM4-80GB GPU。Kernel 性能 benchmark 使用 32 heads、8 KV heads、d_h=128、bf16 精度，3 次 warmup + 1000 次 run 取平均。

- 模型是什么。数据集和bench分别是什么。
  模型：Qwen3 1.7B 结构（仅修改 self-attention 部分为 DMA）。Scaling Laws 实验从 80M/200M/680M/1.7B 参数规模，12/16/24/28 层，d_model=768/1024/1536/2048，heads=6/8/12/16。训练数据集：SmolLMCorpus。下游 benchmark 实验：先用 32B tokens（Web/TextBook/Code/Math）做基础预训练（seq_len=2048），再用 8B tokens（seq_len=8K）做长上下文续训（RoPE base freq 10K→100K）；微调阶段 seq_len=16K（RoPE base freq→400K）。Benchmark：LAMBADA、MMLU、TriviaQA、ARC、PIQA、HellaSwag、OBQA、Winogrande、LongBench（英文任务）、RULER、BBH、GSM8K、MATH、MBPP。优化器：AdamW + WSD LR scheduler。Tokenizer：NeoX。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  开源代码：https://github.com/flash-algo/flash-sparse-attention（CUDA kernel 实现）。论文提供了完整 PyTorch 参考实现（Listing 1）。算法 pipeline 分两步：
  **Step 1 — 动态 Mask 生成（Content-Aware）**：
  ```
  # V ∈ R^{b, n_h, n, d_h}，Δ ∈ R^{n_h, d_h, n_h}，A ∈ R^{n_h}
  dt = W_dt( V.transpose(1,2).reshape(b, n, n_h*d_h) )  # → [b, n, n_h]
  dt = exp( A * softplus(dt) ).transpose(-1, -2)          # → [b, n_h, n]
  m_t = dt.expand(-1, -1, q_len, -1)                      # → [b, n_h, q_len, n]
  m_t = m_t.masked_fill(causal_mask, -inf)                 # apply causal mask
  topk_indices = topk(m_t, w, dim=-1).indices              # per-head top-w selection
  m_t = m_t.masked_fill(not_in_topk, -inf)                 # mask out non-top-w positions
  ```
  **Step 2 — 稀疏 Attention（Position-Aware）**：对每个 (b, h, q_idx)，取 topk_indices 位置的 K/V 向量 [w, d_h]，计算 q_elem·K_selected^T / sqrt(d_h) + mask_selected，softmax 后加权求和 V_selected，输出 o_t∈R^{n_h×d_h}。复杂度从 O(n²d_h) 降为 O(nwd_h)。kernel 在 block 级别判断，若整个 K block 对应的 mask 全为 −∞，则直接跳过该 block 的加载和计算。
