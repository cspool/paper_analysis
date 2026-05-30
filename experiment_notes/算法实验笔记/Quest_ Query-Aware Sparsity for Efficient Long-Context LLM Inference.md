## Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference

- 属于算法pipeline的实现是什么？实验比较什么？
  Quest 提出 query-aware KV cache 稀疏注意力算法。核心思想：KV cache 中 token 的关键性高度依赖于当前 query token（如 Fig. 2 所示 "A is B. C is D. A is" 中，"B" 仅在 query="is" 时关键），因此不能预先静态裁剪 KV cache。Quest 采用 PageAttention（vLLM, Kwon et al. 2023）的 page 粒度管理 KV cache，为每 page 维护 per-channel Key 向量的最小值 $m_i$ 和最大值 $M_i$ 作为元数据。推理时，对给定 Query 向量 $Q$，Quest 计算每 page 的 attention score 上界：$U_i = \max(Q_i \cdot m_i, Q_i \cdot M_i)$，求和 $\sum_i U_i$ 作为该 page 的 criticality 估计。然后选择 Top-K pages 执行近似 self-attention（仅加载选中 page 的 K、V），其余 page 不加载。前两层保持 full attention（因稀疏度低 <10%），其余层使用 Quest。

  实验比较：(a) PG19 语言建模困惑度（32K tokens, token budget=4096）：Quest vs H2O/TOVA/Full Cache；(b) Passkey Retrieval（10K 和 100K 长度）：Quest vs H2O/TOVA/StreamingLLM，不同 token budget (32/64/128/256/512 和 256/512/1024/2048/4096)；(c) LongBench 六数据集（NarrativeQA, HotpotQA, Qasper, TriviaQA, GovReport, MultifieldQA）：Quest vs H2O/TOVA/StreamingLLM，不同 KV cache budget；(d) Kernel 级效率：Quest vs FlashInfer，NVBench 测量 criticality estimation / Top-K filtering / approximate attention 延迟；(e) 端到端推理延迟：Quest vs FlashInfer，单 batch decode 阶段平均每 token 延迟（16K-32K）。

- 硬件平台是什么，配置是什么。
  Kernel 评估：NVIDIA RTX 4090，CUDA 12.2。端到端评估：NVIDIA Ada 6000 GPU（48GB）。模型：LongChat-7b-v1.5-32k（基于 Llama-2-7B，32 layers），Yarn-Llama-2-7b-128k（基于 Llama-2-7B，128K context），Llama2-7B。FP16 权重，同时测试 4-bit 权重量化。

- 模型是什么。数据集和bench分别是什么。
  模型：LongChat-7b-v1.5-32k（Li et al., 2023），Yarn-Llama-2-7b-128k（Peng et al., 2023），Llama2-7B（Touvron et al., 2023）。数据集：PG19 测试集（100 本书，平均 70K tokens，perplexity 评估），Passkey Retrieval（Yarn, Peng et al. 2023，10K 和 100K 长度），LongBench（Bai et al., 2023，含 NarrativeQA/HotpotQA/Qasper/TriviaQA/GovReport/MultifieldQA 共六个子任务）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/mit-han-lab/Quest。基于 FlashInfer（https://flashinfer.ai）kernel 库实现，使用 RAFT（https://github.com/rapidsai/raft）的 batched Top-K CUDA operator。

  算法 pipeline（两阶段）：

  **阶段 0：KV Cache 插入时维护元数据**
  ```
  对于每个 page p（含 S 个 token）每个 channel i:
    M_i^p = max(M_i^p, k_i)   # page p 中所有 token Key 的 channel-wise 最大值
    m_i^p = min(m_i^p, k_i)   # page p 中所有 token Key 的 channel-wise 最小值
  ```
  元数据大小：2 × d_head × num_pages（相比完整 KV cache 的 2 × seq_len × d_head，压缩比为 1/PageSize，如 page_size=16 时仅 1/16）。

  **阶段 1：推理时 Criticality Estimation**
  ```
  输入：Query 向量 Q ∈ R^{d_head}, 所有 page 的 M^p, m^p
  for each page p:
    score_p = 0
    for each channel i:
      U_i = max(Q_i * m_i^p, Q_i * M_i^p)  # 保证 U_i ≥ 任意 K_i * Q_i
      score_p += U_i
  top_k_pages = TopK({score_p}, k=K)
  ```
  张量计算：给定 $Q \in \mathbb{R}^{d}$，$M^p \in \mathbb{R}^{d}$，$m^p \in \mathbb{R}^{d}$，每 page 上界 $s_p = \sum_{i=1}^{d} \max(Q_i \cdot m_i^p, Q_i \cdot M_i^p)$。由于 $\max(Q_i m_i^p, Q_i M_i^p) \geq Q_i K_i^{(t)}$ 对 page 内所有 token $t$ 成立，$s_p$ 是 page 内任意 token attention score 的上界。选 score 最高的 K 个 page。

  **阶段 2：Approximate Attention on Top-K Pages**
  ```
  加载 Top-K pages 的完整 K, V 向量
  S = Q @ K_selected^T / sqrt(d_head)  # 仅计算选中 tokens 的 attention
  A = softmax(S)
  O = A @ V_selected
  ```

  内存加载量：完整 KV cache = 2M × L bytes；Quest = 2M × L/S（元数据）+ 2M × K × S（Top-K 页）= 1/PageSize + K/PageNum of total KV cache。例如 page_size=16, 64K context, K=4K pages → 约 8× 内存加载减少。
